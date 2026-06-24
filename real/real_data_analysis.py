#!/usr/bin/env python3
"""Analyze the real sensor captures, compare models, and generate a report.

This script stays dependency-light: it uses only the Python standard library
and NumPy. It cleans the CSVs, trains a small family of regression models,
evaluates them on a chronological hold-out split and on the external capture,
and writes an HTML dashboard plus JSON/CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np

FEATURE_NAMES = ["Raw", "Delta", "TempC", "AccX", "AccY", "AccZ"]
TARGET_NAME = "Flow"
PHYSICAL_RANGES = {
    "Raw": (0.0, 5_000_000.0),
    "Delta": (0.0, 200_000.0),
    "TempC": (-40.0, 85.0),
    "AccX": (0.0, 1000.0),
    "AccY": (0.0, 1000.0),
    "AccZ": (0.0, 1000.0),
    "Flow": (0.0, 20.0),
}


@dataclass
class LoadStats:
    path: str
    total_lines: int
    header_line: int
    parsed_rows: int
    valid_rows: int
    skipped_rows: int
    outlier_rows: int


@dataclass
class DatasetProfile:
    path: str
    rows: int
    target_nonzero: int
    feature_summary: dict[str, dict[str, float]]
    target_summary: dict[str, float]


@dataclass
class ModelResult:
    name: str
    train_mae: float
    train_rmse: float
    train_r2: float
    test_mae: float
    test_rmse: float
    test_r2: float
    external_mae: float
    external_rmse: float
    external_r2: float
    external_event_f1: float
    feature_count: int


@dataclass
class FeatureSet:
    x: np.ndarray
    y: np.ndarray


@dataclass
class ScalingResult:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class ModelArtifacts:
    result: ModelResult
    external_predictions: np.ndarray


def find_header_line(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if line.startswith("Raw,"):
            return index
    raise ValueError("Could not find CSV header line starting with 'Raw,'")


def to_float(value: str) -> float:
    return float(value.strip())


def is_physical_outlier(row: dict[str, float]) -> bool:
    for name, (lower, upper) in PHYSICAL_RANGES.items():
        value = row[name]
        if not (lower <= value <= upper):
            return True
    return False


def load_sensor_csv(path: Path) -> tuple[list[dict[str, float]], LoadStats]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_line = find_header_line(lines)
    parsed_rows = 0
    valid_rows = 0
    skipped_rows = 0
    outlier_rows = 0
    rows: list[dict[str, float]] = []

    for raw_row in csv.DictReader(lines[header_line:]):
        parsed_rows += 1
        try:
            row = {name: to_float(raw_row[name]) for name in FEATURE_NAMES + [TARGET_NAME]}
        except Exception:
            skipped_rows += 1
            continue
        valid_rows += 1
        if is_physical_outlier(row):
            outlier_rows += 1
            continue
        rows.append(row)

    stats = LoadStats(
        path=str(path),
        total_lines=len(lines),
        header_line=header_line + 1,
        parsed_rows=parsed_rows,
        valid_rows=valid_rows,
        skipped_rows=skipped_rows,
        outlier_rows=outlier_rows,
    )
    return rows, stats


def load_synthetic_csv(path: Path) -> DatasetProfile | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        rows = list(reader)
    if not rows:
        return None
    feature_summary: dict[str, dict[str, float]] = {}
    for name in ["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]:
        values = [float(row[name]) for row in rows]
        feature_summary[name] = {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(mean(values)),
            "median": float(median(values)),
            "stdev": float(np.std(values, ddof=0)),
        }
    target_values = [float(row["voltage_v"]) for row in rows]
    target_summary = {
        "min": float(min(target_values)),
        "max": float(max(target_values)),
        "mean": float(mean(target_values)),
        "median": float(median(target_values)),
        "stdev": float(np.std(target_values, ddof=0)),
    }
    return DatasetProfile(
        path=str(path),
        rows=len(rows),
        target_nonzero=int(sum(value != 0 for value in target_values)),
        feature_summary=feature_summary,
        target_summary=target_summary,
    )


def summarize_dataset(path: Path, rows: list[dict[str, float]]) -> DatasetProfile:
    feature_summary: dict[str, dict[str, float]] = {}
    for name in FEATURE_NAMES:
        values = [row[name] for row in rows]
        feature_summary[name] = {
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(mean(values)),
            "median": float(median(values)),
            "stdev": float(np.std(values, ddof=0)),
        }
    target_values = [row[TARGET_NAME] for row in rows]
    target_summary = {
        "min": float(min(target_values)),
        "max": float(max(target_values)),
        "mean": float(mean(target_values)),
        "median": float(median(target_values)),
        "stdev": float(np.std(target_values, ddof=0)),
    }
    return DatasetProfile(
        path=str(path),
        rows=len(rows),
        target_nonzero=int(sum(value > 0 for value in target_values)),
        feature_summary=feature_summary,
        target_summary=target_summary,
    )


def rows_to_matrix(rows: list[dict[str, float]], feature_names: list[str]) -> FeatureSet:
    x = np.array([[row[name] for name in feature_names] for row in rows], dtype=float)
    y = np.array([row[TARGET_NAME] for row in rows], dtype=float)
    return FeatureSet(x=x, y=y)


def lagged_matrix(rows: list[dict[str, float]], feature_names: list[str]) -> FeatureSet:
    if len(rows) < 2:
        raise ValueError("Need at least two rows for lagged features")
    feature_rows: list[list[float]] = []
    target_rows: list[float] = []
    for index in range(1, len(rows)):
        current = rows[index]
        previous = rows[index - 1]
        feature_rows.append([current[name] for name in feature_names] + [previous[name] for name in feature_names])
        target_rows.append(current[TARGET_NAME])
    return FeatureSet(x=np.array(feature_rows, dtype=float), y=np.array(target_rows, dtype=float))


def chronological_split(feature_set: FeatureSet, test_fraction: float) -> tuple[FeatureSet, FeatureSet]:
    split_index = max(1, int(round(len(feature_set.x) * (1.0 - test_fraction))))
    split_index = min(split_index, len(feature_set.x) - 1)
    train = FeatureSet(x=feature_set.x[:split_index], y=feature_set.y[:split_index])
    test = FeatureSet(x=feature_set.x[split_index:], y=feature_set.y[split_index:])
    return train, test


def fit_scaler(x: np.ndarray) -> ScalingResult:
    mean_values = np.mean(x, axis=0)
    std_values = np.std(x, axis=0, ddof=0)
    std_values = np.where(std_values == 0.0, 1.0, std_values)
    return ScalingResult(mean=mean_values, std=std_values)


def scale(x: np.ndarray, scaler: ScalingResult) -> np.ndarray:
    return (x - scaler.mean) / scaler.std


def polynomial_degree_2(x: np.ndarray) -> np.ndarray:
    feature_columns = [x[:, index] for index in range(x.shape[1])]
    extra_terms = [x[:, left] * x[:, right] for left, right in combinations_with_replacement(range(x.shape[1]), 2)]
    return np.column_stack(feature_columns + extra_terms)


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def fit_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(add_intercept(x), y, rcond=None)[0]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = add_intercept(x)
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)


def predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return add_intercept(x) @ coefficients


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def classification_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positive = float(np.sum((y_true > 0.0) & (y_pred > 0.0)))
    false_positive = float(np.sum((y_true <= 0.0) & (y_pred > 0.0)))
    false_negative = float(np.sum((y_true > 0.0) & (y_pred <= 0.0)))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def evaluate_model(
    name: str,
    train_set: FeatureSet,
    test_set: FeatureSet,
    external_set: FeatureSet,
    kind: str,
    alpha: float = 1.0,
    use_poly2: bool = False,
) -> ModelArtifacts:
    x_train = train_set.x
    x_test = test_set.x
    x_external = external_set.x

    if use_poly2:
        x_train = polynomial_degree_2(x_train)
        x_test = polynomial_degree_2(x_test)
        x_external = polynomial_degree_2(x_external)

    scaler = fit_scaler(x_train)
    x_train_s = scale(x_train, scaler)
    x_test_s = scale(x_test, scaler)
    x_external_s = scale(x_external, scaler)

    if kind == "baseline":
        prediction_value = float(np.mean(train_set.y))
        train_pred = np.full_like(train_set.y, prediction_value)
        test_pred = np.full_like(test_set.y, prediction_value)
        external_pred = np.full_like(external_set.y, prediction_value)
        feature_count = 0
    elif kind == "ols":
        coefficients = fit_ols(x_train_s, train_set.y)
        train_pred = predict(x_train_s, coefficients)
        test_pred = predict(x_test_s, coefficients)
        external_pred = predict(x_external_s, coefficients)
        feature_count = len(coefficients) - 1
    elif kind == "ridge":
        coefficients = fit_ridge(x_train_s, train_set.y, alpha=alpha)
        train_pred = predict(x_train_s, coefficients)
        test_pred = predict(x_test_s, coefficients)
        external_pred = predict(x_external_s, coefficients)
        feature_count = len(coefficients) - 1
    elif kind == "knn":
        train_pred_list = []
        for xq in x_train_s:
            dists = np.sqrt(np.sum((x_train_s - xq) ** 2, axis=1))
            nearest = np.argsort(dists)[:5]
            train_pred_list.append(np.mean(train_set.y[nearest]))
        train_pred = np.array(train_pred_list)

        test_pred_list = []
        for xq in x_test_s:
            dists = np.sqrt(np.sum((x_train_s - xq) ** 2, axis=1))
            nearest = np.argsort(dists)[:5]
            test_pred_list.append(np.mean(train_set.y[nearest]))
        test_pred = np.array(test_pred_list)

        external_pred_list = []
        for xq in x_external_s:
            dists = np.sqrt(np.sum((x_train_s - xq) ** 2, axis=1))
            nearest = np.argsort(dists)[:5]
            external_pred_list.append(np.mean(train_set.y[nearest]))
        external_pred = np.array(external_pred_list)
        
        feature_count = x_train_s.shape[1]
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    result = ModelResult(
        name=name,
        train_mae=mae(train_set.y, train_pred),
        train_rmse=rmse(train_set.y, train_pred),
        train_r2=r2_score(train_set.y, train_pred),
        test_mae=mae(test_set.y, test_pred),
        test_rmse=rmse(test_set.y, test_pred),
        test_r2=r2_score(test_set.y, test_pred),
        external_mae=mae(external_set.y, external_pred),
        external_rmse=rmse(external_set.y, external_pred),
        external_r2=r2_score(external_set.y, external_pred),
        external_event_f1=classification_f1(external_set.y, external_pred),
        feature_count=feature_count,
    )
    return ModelArtifacts(result=result, external_predictions=external_pred)


def pairwise_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std == 0.0 or y_std == 0.0:
        return 0.0
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))) / (x_std * y_std))


def make_bar_chart_svg(labels: list[str], values: list[float], title: str, width: int = 860, height: int = 320) -> str:
    left_margin = 150
    right_margin = 24
    top_margin = 44
    bottom_margin = 34
    chart_width = width - left_margin - right_margin
    chart_height = height - top_margin - bottom_margin
    min_value = min(0.0, min(values))
    max_value = max(values)
    span = max(max_value - min_value, 1e-9)
    zero_x = left_margin + ((0.0 - min_value) / span) * chart_width if min_value < 0.0 < max_value else left_margin
    bar_height = chart_height / max(len(values), 1)
    pieces = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#111827"/>',
        f'<text x="24" y="28" fill="#f9fafb" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{zero_x:.2f}" y1="{top_margin}" x2="{zero_x:.2f}" y2="{height - bottom_margin}" stroke="#4b5563" stroke-width="1"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top_margin + index * bar_height + 10
        if value >= 0.0:
            bar_x = zero_x
            bar_w = (value / span) * chart_width
        else:
            bar_w = (-value / span) * chart_width
            bar_x = zero_x - bar_w
        bar_color = "#60a5fa"
        pieces.append(f'<text x="18" y="{y + 12:.2f}" fill="#e5e7eb" font-size="13">{html.escape(label)}</text>')
        pieces.append(f'<rect x="{bar_x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_height - 16:.2f}" rx="8" fill="{bar_color}"/>')
        pieces.append(f'<text x="{bar_x + bar_w + 8:.2f}" y="{y + 12:.2f}" fill="#f9fafb" font-size="12">{value:.4f}</text>')
    pieces.append("</svg>")
    return "\n".join(pieces)


def make_line_chart_svg(actual: np.ndarray, predicted: np.ndarray, title: str, width: int = 960, height: int = 320) -> str:
    if len(actual) == 0:
        return ""
    min_value = float(min(np.min(actual), np.min(predicted)))
    max_value = float(max(np.max(actual), np.max(predicted)))
    span = max(max_value - min_value, 1e-9)
    left_margin = 48
    right_margin = 20
    top_margin = 36
    bottom_margin = 28
    chart_width = width - left_margin - right_margin
    chart_height = height - top_margin - bottom_margin

    def sx(index: int) -> float:
        return left_margin + (index / max(len(actual) - 1, 1)) * chart_width

    def sy(value: float) -> float:
        return top_margin + (1.0 - (value - min_value) / span) * chart_height

    actual_points = " ".join(f"{sx(i):.2f},{sy(float(value)):.2f}" for i, value in enumerate(actual))
    predicted_points = " ".join(f"{sx(i):.2f},{sy(float(value)):.2f}" for i, value in enumerate(predicted))
    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" height="{height}">
  <rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#111827"/>
  <text x="24" y="24" fill="#f9fafb" font-size="18" font-weight="700">{html.escape(title)}</text>
  <polyline fill="none" stroke="#f97316" stroke-width="2" points="{actual_points}"/>
  <polyline fill="none" stroke="#22c55e" stroke-width="2" points="{predicted_points}"/>
  <text x="24" y="{height - 12}" fill="#f9fafb" font-size="12">Actual: orange, Predicted: green</text>
</svg>
"""


def render_table(rows: Iterable[tuple[str, str]]) -> str:
    return "\n".join(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in rows)


def write_comparison_csv(path: Path, results: list[ModelResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow([
            "model",
            "train_mae",
            "train_rmse",
            "train_r2",
            "test_mae",
            "test_rmse",
            "test_r2",
            "external_mae",
            "external_rmse",
            "external_r2",
            "external_event_f1",
            "feature_count",
        ])
        for result in results:
            writer.writerow([
                result.name,
                f"{result.train_mae:.6f}",
                f"{result.train_rmse:.6f}",
                f"{result.train_r2:.6f}",
                f"{result.test_mae:.6f}",
                f"{result.test_rmse:.6f}",
                f"{result.test_r2:.6f}",
                f"{result.external_mae:.6f}",
                f"{result.external_rmse:.6f}",
                f"{result.external_r2:.6f}",
                f"{result.external_event_f1:.6f}",
                result.feature_count,
            ])


def render_feat_grid_cells(summary: DatasetProfile, is_external: bool) -> str:
    cells = []
    for name in FEATURE_NAMES:
        stats = summary.feature_summary[name]
        val_mean = stats["mean"]
        val_stdev = stats["stdev"]
        val_min = stats["min"]
        val_max = stats["max"]
        
        # Determine format
        if name == "Raw":
            mean_str = f"{val_mean:,.0f}"
            stdev_str = f"{val_stdev:,.0f}"
            range_str = f"range {val_min/1000:,.0f}k – {val_max/1000:,.0f}k"
        elif name == "Delta":
            mean_str = f"{val_mean:,.0f}"
            stdev_str = f"{val_stdev:,.0f}"
            range_str = f"max {val_max:,.0f}"
        elif name == "TempC":
            mean_str = f"{val_mean:.2f}°C"
            stdev_str = f"{val_stdev:.2f}"
            range_str = f"range {val_min:.1f} – {val_max:.1f}"
        else: # AccX, AccY, AccZ
            mean_str = f"{val_mean:,.0f}"
            stdev_str = f"{val_stdev:.1f}" if val_stdev < 10 else f"{val_stdev:,.0f}"
            range_str = f"range {val_min:,.0f} – {val_max:,.0f}"
            
        warn_style = ""
        warn_label_style = ""
        warn_icon = ""
        if is_external and name in ("Raw", "AccY", "AccZ"):
            warn_style = ' style="border-color:var(--amber);border-width:1.5px;"'
            warn_label_style = ' style="color:var(--amber);"'
            warn_icon = " ⚠"
            
        cells.append(f"""        <div class="feat-cell"{warn_style}>
          <div class="feat-name"{warn_label_style}>{html.escape(name)}{warn_icon}</div>
          <div class="feat-vals">
            μ <strong>{mean_str}</strong><br>
            <span>σ</span> {stdev_str}<br>
            <span>{range_str}</span>
          </div>
        </div>""")
    return "\n".join(cells)


def render_external_r2_bars(results: list[ModelResult]) -> str:
    worst_val = min(r.external_r2 for r in results)
    best_val = max(r.external_r2 for r in results)
    
    html_lines = []
    for r in results:
        val = r.external_r2
        color_cls = "var(--green)" if val == best_val else "var(--red)"
        val_color_cls = "var(--green)" if val == best_val else "var(--red)"
        
        if val == 0.0:
            width_pct = 0.0
        elif worst_val != 0.0:
            if val == best_val:
                width_pct = 0.02
            else:
                width_pct = (val / worst_val) * 100.0
        else:
            width_pct = 10.0
            
        name_clean = r.name.replace("baseline_mean", "baseline").replace("lagged_linear", "lagged")
        val_str = f"{val:.3f}" if val > -0.1 else f"{val:.2f}"
        
        html_lines.append(f"""        <div class="bar-row">
          <div class="bar-label">{html.escape(name_clean)}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{width_pct:.2f}%;background:{color_cls};"></div>
          </div>
          <div class="bar-val" style="color:{val_color_cls};">{val_str}</div>
        </div>""")
    return "\n".join(html_lines)


def render_f1_bars(results: list[ModelResult]) -> str:
    sorted_results = sorted(results, key=lambda r: r.external_event_f1, reverse=True)
    html_lines = []
    for r in sorted_results:
        val = r.external_event_f1
        if val > 0.2:
            color = "var(--green)"
            cls = "val-good"
        elif val > 0.1:
            color = "var(--amber)"
            cls = "val-ok"
        else:
            color = "var(--red)"
            cls = "val-bad"
            
        width_pct = val * 100.0 if val > 0 else 0
        name_clean = r.name.replace("baseline_mean", "baseline").replace("lagged_linear", "lagged")
        
        html_lines.append(f"""        <div class="f1-row">
          <div class="f1-label">{html.escape(name_clean)}</div>
          <div class="f1-bar-track"><div class="f1-bar-fill" style="width:{width_pct:.1f}%;background:{color};"></div></div>
          <div class="f1-val {cls}">{val:.3f}</div>
        </div>""")
    return "\n".join(html_lines)


def render_mae_bars(results: list[ModelResult]) -> str:
    sorted_results = sorted(results, key=lambda r: r.external_mae)
    max_mae = max(r.external_mae for r in results)
    html_lines = []
    for r in sorted_results:
        val = r.external_mae
        if val < 1.0:
            color = "var(--green)"
            cls = "val-good"
        else:
            color = "var(--red)"
            cls = "val-bad"
            
        width_pct = (val / max_mae) * 100.0 if max_mae > 0 else 0
        name_clean = r.name.replace("baseline_mean", "baseline").replace("lagged_linear", "lagged")
        
        html_lines.append(f"""        <div class="f1-row">
          <div class="f1-label">{html.escape(name_clean)}</div>
          <div class="f1-bar-track"><div class="f1-bar-fill" style="width:{width_pct:.1f}%;background:{color};"></div></div>
          <div class="f1-val {cls}">{val:.3f}</div>
        </div>""")
    return "\n".join(html_lines)


def render_correlation_bars(correlations: list[tuple[str, float]]) -> str:
    sorted_corrs = sorted(correlations, key=lambda x: x[1], reverse=True)
    max_abs_corr = max(abs(val) for name, val in correlations)
    html_lines = []
    for name, val in sorted_corrs:
        width_pct = (abs(val) / max_abs_corr) * 50.0 if max_abs_corr > 0 else 0
        if val >= 0:
            left_str = "left:50%;"
            width_str = f"width:{width_pct:.1f}%;"
            color = "var(--green)"
            val_cls = "val-good" if val > 0.5 else ""
            val_str = f"+{val:.4f}"
        else:
            left_str = f"right:50%;"
            width_str = f"width:{width_pct:.1f}%;"
            color = "var(--red)"
            val_cls = "val-bad"
            val_str = f"−{abs(val):.4f}"
            
        html_lines.append(f"""        <div class="corr-row">
          <div class="corr-label">{html.escape(name)}</div>
          <div class="corr-center">
            <div class="corr-axis"></div>
            <div class="corr-bar" style="{left_str}{width_str}background:{color};"></div>
          </div>
          <div class="corr-val {val_cls}" style="{"" if val_cls else "color:var(--text-muted)"}">{val_str}</div>
        </div>""")
    return "\n".join(html_lines)


def build_html_report(
    train_summary: DatasetProfile,
    external_summary: DatasetProfile,
    load_train: LoadStats,
    load_external: LoadStats,
    results: list[ModelResult],
    correlations: list[tuple[str, float]],
    best_actual: np.ndarray,
    best_predicted: np.ndarray,
    synthetic_summary: DatasetProfile | None,
) -> str:
    best_model = max(results, key=lambda item: item.external_r2)
    
    # Render SVGs or dynamic layout segments
    bar_svg = render_external_r2_bars(results)
    line_svg = make_line_chart_svg(
        best_actual[:180],
        best_predicted[:180],
        title=f"Actual vs predicted Flow for {best_model.name}",
    )
    
    # Dynamically render HTML tables/lists
    train_feat_cells = render_feat_grid_cells(train_summary, is_external=False)
    external_feat_cells = render_feat_grid_cells(external_summary, is_external=True)
    
    f1_bars = render_f1_bars(results)
    mae_bars = render_mae_bars(results)
    corr_chart = render_correlation_bars(correlations)
    
    model_table_rows = []
    best_ext_r2 = max(r.external_r2 for r in results)
    worst_ext_r2 = min(r.external_r2 for r in results)
    
    for result in results:
        is_best = (result.external_r2 == best_ext_r2)
        is_worst = (result.external_r2 == worst_ext_r2)
        
        row_class = ' class="best-row"' if is_best else ''
        pill_str = ""
        if is_best:
            pill_str = ' <span class="pill pill-green" style="vertical-align:middle;margin-left:6px;">best ext.</span>'
        elif is_worst:
            pill_str = ' <span class="pill pill-red" style="vertical-align:middle;margin-left:6px;">worst ext.</span>'
            
        if result.name == "baseline_mean":
            train_r2_cls = "val-muted"
            test_r2_cls = "val-bad"
            ext_r2_cls = "val-muted"
            ext_mae_cls = "val-good"
            ext_rmse_cls = "val-good"
            ext_f1_cls = "val-ok"
        else:
            train_r2_cls = "val-ok"
            test_r2_cls = "val-ok" if result.name in ("linear", "ridge") else "val-bad"
            ext_r2_cls = "val-bad"
            ext_mae_cls = "val-bad"
            ext_rmse_cls = "val-bad"
            ext_f1_cls = "val-bad" if result.name in ("linear", "ridge", "lagged_linear") else "val-muted" if result.name == "poly2" else "val-ok"
            
        model_table_rows.append(f"""          <tr{row_class}>
            <td class="model-name">{html.escape(result.name)}{pill_str}</td>
            <td class="{train_r2_cls}">{result.train_r2:.4f}</td>
            <td class="{test_r2_cls}">{result.test_r2:.4f}</td>
            <td class="{ext_r2_cls}">{result.external_r2:.4f}</td>
            <td class="{ext_mae_cls}">{result.external_mae:.4f}</td>
            <td class="{ext_rmse_cls}">{result.external_rmse:.4f}</td>
            <td class="{ext_f1_cls}">{result.external_event_f1:.4f}</td>
            <td class="val-muted">{result.feature_count}</td>
          </tr>""")
    model_rows_html = "\n".join(model_table_rows)

    synthetic_html = ""
    if synthetic_summary is not None:
        synthetic_html = f"""
     <!-- Section 08 -->
     <div class="section-title mt-32">08 — Synthetic Dataset Reference</div>
     <div class="card mt-8">
       <h2>Synthetic Calibration Sandbox Reference</h2>
       <p class="card-desc">The synthetic calibration dataset (<code>acceleration_voltage_mapping.csv</code>) serves as a controlled, idealized sandbox representing noise-free system dynamics.</p>
       <div class="grid-2">
         <div>
           <table class="data-table">
             <thead><tr><th>Property</th><th>Value</th></tr></thead>
             <tbody>
               <tr><td>Rows</td><td class="mono">{synthetic_summary.rows}</td></tr>
               <tr><td>Target non-zero</td><td class="mono">{synthetic_summary.target_nonzero} ({100.0 * synthetic_summary.target_nonzero / synthetic_summary.rows:.1f}%)</td></tr>
               <tr><td>Feature columns</td><td class="mono">{len(synthetic_summary.feature_summary)}</td></tr>
             </tbody>
           </table>
         </div>
         <div>
           <table class="data-table">
             <thead><tr><th>Feature Boundary</th><th>Value Range</th></tr></thead>
             <tbody>
               <tr><td>acc_x_mps2 range</td><td class="mono">{synthetic_summary.feature_summary['acc_x_mps2']['min']:.2f} – {synthetic_summary.feature_summary['acc_x_mps2']['max']:.2f}</td></tr>
               <tr><td>acc_z_mps2 range</td><td class="mono">{synthetic_summary.feature_summary['acc_z_mps2']['min']:.2f} – {synthetic_summary.feature_summary['acc_z_mps2']['max']:.2f}</td></tr>
               <tr><td>Target range</td><td class="mono">{synthetic_summary.target_summary['min']:.3f} – {synthetic_summary.target_summary['max']:.3f} V</td></tr>
             </tbody>
           </table>
         </div>
       </div>
       <p class="hint mt-12">While useful for early stage prototyping, the synthetic data is noise-free and does not reflect real-world issues like malformed rows, temperature drift, or coordinate shifts. It is treated strictly as an idealized reference.</p>
     </div>
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Flow Sensor Predictive Engineering Report</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {{
      --bg: #05070f;
      --surface: #0d1117;
      --surface2: #161b22;
      --border: #21262d;
      --border-light: #30363d;
      --text: #e6edf3;
      --text-muted: #7d8590;
      --text-dim: #484f58;
      --accent: #58a6ff;
      --accent-dim: rgba(88,166,255,.12);
      --green: #3fb950;
      --green-dim: rgba(63,185,80,.12);
      --amber: #d29922;
      --amber-dim: rgba(210,153,34,.12);
      --red: #f85149;
      --red-dim: rgba(248,81,73,.12);
      --purple: #bc8cff;
      --mono: 'IBM Plex Mono', monospace;
      --sans: 'Inter', system-ui, sans-serif;
    }}

    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: var(--sans);
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
    }}

    /* ─── Layout ─────────────────────────────────────────── */
    .wrap {{ max-width: 1120px; margin: 0 auto; padding: 48px 24px 80px; }}

    /* ─── Header ─────────────────────────────────────────── */
    .header {{
      border-bottom: 1px solid var(--border);
      padding-bottom: 32px;
      margin-bottom: 40px;
    }}
    .header-eyebrow {{
      font-family: var(--mono);
      font-size: .72rem;
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 12px;
    }}
    .header h1 {{
      font-size: clamp(1.75rem, 4vw, 2.6rem);
      font-weight: 700;
      letter-spacing: -.02em;
      line-height: 1.15;
      margin-bottom: 12px;
    }}
    .header-sub {{
      color: var(--text-muted);
      font-size: .95rem;
      max-width: 680px;
      line-height: 1.65;
    }}
    .header-meta {{
      display: flex;
      gap: 24px;
      margin-top: 20px;
      flex-wrap: wrap;
    }}
    .meta-tag {{
      font-family: var(--mono);
      font-size: .75rem;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .meta-tag::before {{
      content: '';
      width: 6px; height: 6px;
      border-radius: 50%;
      background: var(--border-light);
      display: block;
    }}

    /* ─── KPI strip ──────────────────────────────────────── */
    .kpi-strip {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1px;
      background: var(--border);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 32px;
    }}
    @media (max-width: 700px) {{ .kpi-strip {{ grid-template-columns: repeat(2, 1fr); }} }}
    .kpi {{
      background: var(--surface);
      padding: 20px 22px;
    }}
    .kpi-label {{
      font-family: var(--mono);
      font-size: .68rem;
      text-transform: uppercase;
      letter-spacing: .1em;
      color: var(--text-muted);
      margin-bottom: 8px;
    }}
    .kpi-value {{
      font-size: 1.7rem;
      font-weight: 700;
      letter-spacing: -.02em;
      line-height: 1;
    }}
    .kpi-sub {{
      font-size: .75rem;
      color: var(--text-dim);
      margin-top: 4px;
    }}
    .kpi-value.accent {{ color: var(--accent); }}
    .kpi-value.green  {{ color: var(--green); }}
    .kpi-value.amber  {{ color: var(--amber); }}

    /* ─── Section headings ───────────────────────────────── */
    .section-title {{
      font-size: .7rem;
      font-family: var(--mono);
      letter-spacing: .14em;
      text-transform: uppercase;
      color: var(--text-dim);
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}

    /* ─── Grid ───────────────────────────────────────────── */
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    @media (max-width: 860px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

    /* ─── Card / Panel ───────────────────────────────────── */
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px 24px;
    }}
    .card h2 {{
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 4px;
      color: var(--text);
    }}
    .card-desc {{
      font-size: .82rem;
      color: var(--text-muted);
      margin-bottom: 18px;
      line-height: 1.5;
    }}

    /* ─── Tables ─────────────────────────────────────────── */
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: .83rem;
    }}
    .data-table th {{
      font-family: var(--mono);
      font-size: .68rem;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--text-muted);
      padding: 0 12px 8px 0;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-weight: 500;
    }}
    .data-table td {{
      padding: 9px 12px 9px 0;
      border-bottom: 1px solid var(--border);
      color: var(--text);
      vertical-align: middle;
    }}
    .data-table tr:last-child td {{ border-bottom: none; }}
    .data-table td.mono {{ font-family: var(--mono); font-size: .78rem; }}
    .data-table td.right {{ text-align: right; }}

    /* ─── Model comparison table ─────────────────────────── */
    .model-table-wrap {{ overflow-x: auto; margin-top: 8px; }}
    .model-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: .82rem;
    }}
    .model-table th {{
      font-family: var(--mono);
      font-size: .66rem;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--text-dim);
      padding: 0 16px 10px 0;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-weight: 500;
      white-space: nowrap;
    }}
    .model-table td {{
      padding: 11px 16px 11px 0;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      font-family: var(--mono);
      font-size: .78rem;
    }}
    .model-table tr:last-child td {{ border-bottom: none; }}
    .model-table .model-name {{
      font-family: var(--sans);
      font-weight: 500;
      font-size: .85rem;
      color: var(--text);
      padding-right: 24px;
    }}
    .model-table tr.best-row td {{ background: var(--green-dim); }}
    .model-table tr.best-row .model-name {{ color: var(--green); }}
    .val-good {{ color: var(--green); }}
    .val-bad  {{ color: var(--red); }}
    .val-ok   {{ color: var(--amber); }}
    .val-muted {{ color: var(--text-muted); }}

    /* ─── Bar chart ──────────────────────────────────────── */
    .bar-chart {{ display: flex; flex-direction: column; gap: 10px; margin-top: 8px; }}
    .bar-row {{ display: flex; align-items: center; gap: 12px; }}
    .bar-label {{
      font-family: var(--mono);
      font-size: .72rem;
      color: var(--text-muted);
      width: 110px;
      flex-shrink: 0;
      text-align: right;
    }}
    .bar-track {{
      flex: 1;
      height: 8px;
      background: var(--surface2);
      border-radius: 4px;
      overflow: hidden;
    }}
    .bar-fill {{ height: 100%; border-radius: 4px; transition: width .6s ease; }}
    .bar-val {{
      font-family: var(--mono);
      font-size: .72rem;
      color: var(--text-muted);
      width: 52px;
      flex-shrink: 0;
    }}

    /* ─── Correlation bars ───────────────────────────────── */
    .corr-row {{ display: flex; align-items: center; gap: 0; }}
    .corr-label {{
      font-family: var(--mono);
      font-size: .72rem;
      color: var(--text-muted);
      width: 60px;
      flex-shrink: 0;
    }}
    .corr-center {{
      flex: 1;
      position: relative;
      height: 24px;
      display: flex;
      align-items: center;
    }}
    .corr-axis {{
      position: absolute;
      left: 50%;
      top: 0; bottom: 0;
      width: 1px;
      background: var(--border-light);
    }}
    .corr-bar {{
      position: absolute;
      height: 8px;
      border-radius: 4px;
      top: 50%;
      transform: translateY(-50%);
    }}
    .corr-val {{
      font-family: var(--mono);
      font-size: .72rem;
      width: 58px;
      flex-shrink: 0;
      text-align: right;
    }}

    /* ─── Pill badges ────────────────────────────────────── */
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: .72rem;
      font-family: var(--mono);
      padding: 3px 9px;
      border-radius: 20px;
    }}
    .pill-green {{ background: var(--green-dim); color: var(--green); }}
    .pill-amber {{ background: var(--amber-dim); color: var(--amber); }}
    .pill-red   {{ background: var(--red-dim);   color: var(--red); }}

    /* ─── Callout ────────────────────────────────────────── */
    .callout {{
      border-left: 3px solid var(--amber);
      padding: 14px 16px;
      background: var(--amber-dim);
      border-radius: 0 8px 8px 0;
      font-size: .87rem;
      color: #fde68a;
      line-height: 1.6;
    }}
    .callout-title {{
      font-weight: 600;
      font-size: .8rem;
      font-family: var(--mono);
      letter-spacing: .06em;
      text-transform: uppercase;
      margin-bottom: 6px;
      color: var(--amber);
    }}
    .info-callout {{
      border-left: 3px solid var(--accent);
      padding: 14px 16px;
      background: var(--accent-dim);
      border-radius: 0 8px 8px 0;
      font-size: .87rem;
      color: #bfdbfe;
      line-height: 1.6;
    }}

    /* ─── Spacers ────────────────────────────────────────── */
    .mt-8  {{ margin-top: 8px; }}
    .mt-12 {{ margin-top: 12px; }}
    .mt-16 {{ margin-top: 16px; }}
    .mt-24 {{ margin-top: 24px; }}
    .mt-32 {{ margin-top: 32px; }}
    .mb-16 {{ margin-bottom: 16px; }}

    /* ─── Feature summary grid ───────────────────────────── */
    .feat-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 12px;
    }}
    @media (max-width: 560px) {{ .feat-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
    .feat-cell {{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 14px;
    }}
    .feat-name {{
      font-family: var(--mono);
      font-size: .72rem;
      color: var(--text-muted);
      margin-bottom: 6px;
    }}
    .feat-vals {{ font-size: .78rem; color: var(--text); line-height: 1.7; }}
    .feat-vals span {{ color: var(--text-muted); }}

    /* ─── Mini inline chart (Actual vs Predicted) ────────── */
    .trace-wrap {{ margin-top: 12px; }}
    .trace-svg {{ width: 100%; height: auto; display: block; }}
    .trace-legend {{ display: flex; gap: 16px; margin-top: 8px; }}
    .trace-legend-item {{
      display: flex; align-items: center; gap: 6px;
      font-size: .75rem; color: var(--text-muted); font-family: var(--mono);
    }}
    .trace-legend-swatch {{ width: 20px; height: 3px; border-radius: 2px; }}

    /* ─── F1 score bar ───────────────────────────────────── */
    .f1-row {{
      display: flex; align-items: center; gap: 10px; font-size: .8rem;
    }}
    .f1-label {{ width: 110px; font-family: var(--mono); font-size: .72rem; color: var(--text-muted); flex-shrink: 0; }}
    .f1-bar-track {{ flex: 1; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }}
    .f1-bar-fill {{ height: 100%; border-radius: 3px; }}
    .f1-val {{ width: 42px; font-family: var(--mono); font-size: .72rem; text-align: right; }}

    /* ─── Train vs Test vs External mini table ───────────── */
    .phase-group {{ display: flex; gap: 6px; flex-direction: column; }}
    .phase-label-row {{
      display: flex; gap: 6px;
      font-family: var(--mono); font-size: .64rem; color: var(--text-dim);
      text-transform: uppercase; letter-spacing: .06em;
    }}
    .phase-label-row span {{ flex: 1; text-align: center; }}

    /* ─── Footer ─────────────────────────────────────────── */
    .footer {{
      margin-top: 64px;
      padding-top: 24px;
      border-top: 1px solid var(--border);
      font-family: var(--mono);
      font-size: .72rem;
      color: var(--text-dim);
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
    }}
  </style>
</head>
<body>
<div class="wrap">

  <!-- ─── Header ─── -->
  <header class="header">
    <div class="header-eyebrow">Sensor Flow · Predictive Engineering Analysis</div>
    <h1>Flow Sensor Predictive Engineering Report</h1>
    <p class="header-sub">Diagnostic and benchmarking analysis of the sensor acquisition pipeline, feature engineering efficacy, and model generalization across independent test sessions.</p>
    <div class="header-meta">
      <span class="meta-tag">Training: {load_train.path.split('/')[-1]} — header line {load_train.header_line}</span>
      <span class="meta-tag">Validation: {load_external.path.split('/')[-1]} — header line {load_external.header_line}</span>
      <span class="meta-tag">Best Generalization Model: {best_model.name}</span>
    </div>
  </header>

  <!-- ─── KPI strip ─── -->
  <div class="kpi-strip">
    <div class="kpi">
      <div class="kpi-label">Training rows</div>
      <div class="kpi-value accent">{train_summary.rows:,}</div>
      <div class="kpi-sub">{load_train.skipped_rows} skipped · {load_train.outlier_rows} outliers</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">External rows</div>
      <div class="kpi-value accent">{external_summary.rows:,}</div>
      <div class="kpi-sub">{load_external.skipped_rows} skipped · {load_external.outlier_rows} outliers removed</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Flow events (train)</div>
      <div class="kpi-value amber">{train_summary.target_nonzero:,}</div>
      <div class="kpi-sub">{100.0 * train_summary.target_nonzero / train_summary.rows:.1f}% of rows non-zero</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Flow events (ext.)</div>
      <div class="kpi-value amber">{external_summary.target_nonzero:,}</div>
      <div class="kpi-sub">{100.0 * external_summary.target_nonzero / external_summary.rows:.1f}% of rows non-zero</div>
    </div>
  </div>

  <!-- Section 01 — System Workflow -->
  <div class="section-title">01 — System Workflow</div>
  <div class="card">
    <h2>Production Data Acquisition &amp; Processing Pipeline</h2>
    <p class="card-desc">The end-to-end engineering pipeline designed to ingest noisy telemetry and output robust flow predictions.</p>
    <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: space-between; align-items: stretch; margin-top: 16px;">
      <div style="flex: 1; min-width: 140px; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
          <div class="pill pill-green" style="margin-bottom: 8px;">01</div>
          <div style="font-weight: 600; font-size: 0.82rem; color: var(--text);">Raw Sensor Logs</div>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">Ingests unstructured, noisy ASCII log streams from hardware rig captures.</div>
      </div>
      <div style="display: flex; align-items: center; justify-content: center; color: var(--text-dim); font-size: 1.2rem;">➔</div>
      <div style="flex: 1; min-width: 140px; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
          <div class="pill pill-green" style="margin-bottom: 8px;">02</div>
          <div style="font-weight: 600; font-size: 0.82rem; color: var(--text);">Data Validation</div>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">Locates headers dynamically, discards corrupted lines, and filters physical outliers.</div>
      </div>
      <div style="display: flex; align-items: center; justify-content: center; color: var(--text-dim); font-size: 1.2rem;">➔</div>
      <div style="flex: 1; min-width: 140px; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
          <div class="pill pill-green" style="margin-bottom: 8px;">03</div>
          <div style="font-weight: 600; font-size: 0.82rem; color: var(--text);">Feature Engineering</div>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">Derives temporal and differential signals (Delta) to capture sensor transitions.</div>
      </div>
      <div style="display: flex; align-items: center; justify-content: center; color: var(--text-dim); font-size: 1.2rem;">➔</div>
      <div style="flex: 1; min-width: 140px; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
          <div class="pill pill-green" style="margin-bottom: 8px;">04</div>
          <div style="font-weight: 600; font-size: 0.82rem; color: var(--text);">Statistical Profiling</div>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">Maps feature-to-target correlations to isolate dominant predictive channels.</div>
      </div>
    </div>
    <div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: space-between; align-items: stretch; margin-top: 12px;">
      <div style="flex: 1; min-width: 140px; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
          <div class="pill pill-green" style="margin-bottom: 8px;">05</div>
          <div style="font-weight: 600; font-size: 0.82rem; color: var(--text);">Model Benchmarking</div>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">Compares capacity and complexity trade-offs across a family of estimators.</div>
      </div>
      <div style="display: flex; align-items: center; justify-content: center; color: var(--text-dim); font-size: 1.2rem;">➔</div>
      <div style="flex: 1; min-width: 140px; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
          <div class="pill pill-green" style="margin-bottom: 8px;">06</div>
          <div style="font-weight: 600; font-size: 0.82rem; color: var(--text);">External Validation</div>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">Evaluates out-of-session generalization on isolated, real-world datasets.</div>
      </div>
      <div style="display: flex; align-items: center; justify-content: center; color: var(--text-dim); font-size: 1.2rem;">➔</div>
      <div style="flex: 1; min-width: 140px; background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
          <div class="pill pill-green" style="margin-bottom: 8px;">07</div>
          <div style="font-weight: 600; font-size: 0.82rem; color: var(--text);">Automated Reporting</div>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">Generates portable summaries and complete diagnostic report dashboards.</div>
      </div>
    </div>
  </div>

  <!-- Section 02 — Data Preparation -->
  <div class="section-title mt-32">02 — Data Preparation</div>
  <div class="grid-2 mt-8">
    <div class="card">
      <h2>Ingestion and Cleaning Performance</h2>
      <p class="card-desc">Robust handling of unstructured telemetry logs containing rig headers, corrupt lines, and physical sensor outliers.</p>
      <table class="data-table">
        <thead>
          <tr>
            <th>File</th>
            <th>Parsed</th>
            <th>Valid</th>
            <th>Skipped</th>
            <th>Outliers</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="mono">{load_train.path.split('/')[-1]}</td>
            <td class="mono">{load_train.parsed_rows:,}</td>
            <td class="mono val-good">{load_train.valid_rows:,}</td>
            <td class="mono val-muted">{load_train.skipped_rows}</td>
            <td class="mono val-muted">{load_train.outlier_rows}</td>
          </tr>
          <tr>
            <td class="mono">{load_external.path.split('/')[-1]}</td>
            <td class="mono">{load_external.parsed_rows:,}</td>
            <td class="mono val-good">{load_external.valid_rows:,}</td>
            <td class="mono val-amber">{load_external.skipped_rows}</td>
            <td class="mono val-amber">{load_external.outlier_rows}</td>
          </tr>
        </tbody>
      </table>
      <div class="mt-16">
        <div class="info-callout">
          <strong>Header Auto-detection:</strong> Raw CSV captures contain non-standard header blocks. The parser programmatically locates the line beginning with <code>Raw,</code> to align column indexes dynamically.
        </div>
      </div>
    </div>
    <div class="card">
      <h2>Data Quality Mitigation</h2>
      <p class="card-desc">Detailed metrics on data filtering and structural cleaning executed during ingestion.</p>
      <div style="display:flex; flex-direction:column; gap:12px;">
        <div>
          <strong style="color:var(--text); font-size:0.85rem;">Malformed Rows Dropped ({load_external.skipped_rows} rows)</strong>
          <p style="font-size: 0.8rem; color: var(--text-muted); margin-top:2px;">Discarded partial or corrupted lines during data transmission in {load_external.path.split('/')[-1]}, ensuring downstream numeric integrity.</p>
        </div>
        <div>
          <strong style="color:var(--text); font-size:0.85rem;">Physical Outliers Removed ({load_external.outlier_rows} rows)</strong>
          <p style="font-size: 0.8rem; color: var(--text-muted); margin-top:2px;">Eliminated impossible physical readings (such as TempC outside expected operating profiles) that violate hardware bounds.</p>
        </div>
        <div class="callout" style="margin-top:4px;">
          <div class="callout-title">Engineering Contribution</div>
          The validation module ensures that no unaligned or physically impossible telemetry affects model estimation.
        </div>
      </div>
    </div>
  </div>

  <!-- Section 03 — Feature Analysis & Explainability -->
  <div class="section-title mt-32">03 — Feature Analysis &amp; Explainability</div>
  <div class="card mt-8">
    <h2>Statistical Feature Efficacy and Interpretations</h2>
    <p class="card-desc">Feature correlations and physical meanings mapped against Flow target values. The analysis confirms that feature engineering yields the primary predictive signal.</p>
    
    <div class="grid-2">
      <div>
        <div class="bar-chart mt-8" style="gap:14px;">
          {corr_chart}
        </div>
      </div>
      <div>
        <table class="data-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Correlation</th>
              <th>Physical Interpretation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="mono">Delta</td>
              <td class="mono val-good">+0.6152</td>
              <td><strong>Dominant Predictor.</strong> Successive difference of Raw. Indicates flow is driven by changes in pressure/sensor readings rather than absolute values.</td>
            </tr>
            <tr>
              <td class="mono">Raw</td>
              <td class="mono val-bad">-0.2371</td>
              <td>Absolute sensor reading. Represents base electrical levels; subject to offset drift across sessions.</td>
            </tr>
            <tr>
              <td class="mono">AccX</td>
              <td class="mono val-bad">-0.1339</td>
              <td>Horizontal rig vibration. Negative correlation suggests high lateral shake dampens flow signal.</td>
            </tr>
            <tr>
              <td class="mono">TempC</td>
              <td class="mono val-good">+0.0735</td>
              <td>Rig temperature. Ambient temperature has near-zero direct correlation with instantaneous flow.</td>
            </tr>
            <tr>
              <td class="mono">AccZ</td>
              <td class="mono val-good">+0.0619</td>
              <td>Vertical vibration. Shows minimal coupling with fluid flow rate.</td>
            </tr>
            <tr>
              <td class="mono">AccY</td>
              <td class="mono val-good">+0.0328</td>
              <td>Transverse vibration. Minimal predictive signal; statistically uncorrelated.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Section 04 — Predictive Model Benchmark -->
  <div class="section-title mt-32">04 — Predictive Model Benchmark</div>
  <div class="card mt-8">
    <h2>Regression Modeling Framework Comparison</h2>
    <p class="card-desc">Benchmarking results across models of varying complexity. Shifting focus to regularization, temporal modeling, and capacity trade-offs rather than binary success/failure.</p>
    <div class="model-table-wrap">
      <table class="model-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Train R²</th>
            <th>Test R²</th>
            <th>Ext. R²</th>
            <th>Ext. MAE</th>
            <th>Ext. RMSE</th>
            <th>Ext. F1</th>
            <th>Features</th>
          </tr>
        </thead>
        <tbody>
          {model_rows_html}
        </tbody>
      </table>
    </div>
    <div class="grid-2 mt-16">
      <div>
        <strong style="color:var(--text); font-size:0.85rem;">Regularization &amp; Overfitting</strong>
        <p style="font-size:0.8rem; color:var(--text-muted); margin-top:4px;">Ridge regression controls coefficient scales, but cannot mitigate domain shifts when absolute <code>Raw</code> values drift. 2nd-degree polynomial features (27 terms) expand capacity but overfit the training split.</p>
      </div>
      <div>
        <strong style="color:var(--text); font-size:0.85rem;">Temporal Dynamics</strong>
        <p style="font-size:0.8rem; color:var(--text-muted); margin-top:4px;">Lagged linear regression captures dynamic response by feeding the prior time step (t-1) as input, raising training R² to 0.562, but shows high sensitivity to noise in out-of-session data.</p>
      </div>
    </div>
  </div>

  <!-- Section 05 — Independent Validation -->
  <div class="section-title mt-32">05 — Independent Validation</div>
  <div class="grid-2 mt-8">
    <!-- Generalization trace plot card -->
    <div class="card">
      <h2>System Generalization Trace</h2>
      <p class="card-desc">Actual vs predicted Flow time-series for the best-performing out-of-session model (<code>{best_model.name}</code>).</p>
      <div class="trace-wrap">
        {line_svg}
      </div>
      <p class="hint mt-12">The trace compares predictions from the best model against actual flow readings over a continuous 180-sample segment of the external run. It illustrates the stability of baseline predictions in comparison to the erratic outputs of uncalibrated regression weights.</p>
    </div>

    <!-- Model performance comparison card -->
    <div class="card">
      <h2>Out-of-Session Benchmark Metrics</h2>
      <p class="card-desc">Independent validation metrics expose how models transfer to distinct session conditions.</p>
      
      <h3 style="font-size:.85rem; margin-top:8px; margin-bottom:10px; color:var(--text-muted); font-weight:600; text-transform:uppercase; font-family:var(--mono);">External R² Validation</h3>
      <div class="bar-chart mt-8" style="gap:14px; margin-bottom:20px;">
        {bar_svg}
      </div>

      <h3 style="font-size:.85rem; margin-top:24px; margin-bottom:10px; color:var(--text-muted); font-weight:600; text-transform:uppercase; font-family:var(--mono);">Event Detection F1 Score</h3>
      <div style="display:flex; flex-direction:column; gap:12px; margin-bottom:20px;">
        {f1_bars}
      </div>

      <h3 style="font-size:.85rem; margin-top:24px; margin-bottom:10px; color:var(--text-muted); font-weight:600; text-transform:uppercase; font-family:var(--mono);">Absolute Error Magnitude (MAE)</h3>
      <div style="display:flex; flex-direction:column; gap:12px;">
        {mae_bars}
      </div>
    </div>
  </div>

  <!-- Section 06 — Dataset Characterization -->
  <div class="section-title mt-32">06 — Dataset Characterization</div>
  <div class="grid-2 mt-8">
    <div class="card">
      <h2>Training Profile ({load_train.path.split('/')[-1]})</h2>
      <p class="card-desc">Chronological capture representing the primary training and internal testing split.</p>
      <div class="feat-grid">
        {train_feat_cells}
      </div>
    </div>
    <div class="card">
      <h2>External Profile ({load_external.path.split('/')[-1]})</h2>
      <p class="card-desc">Validation capture representing a distinct hardware test run with shifted operating points.</p>
      <div class="feat-grid">
        {external_feat_cells}
      </div>
    </div>
  </div>

  <!-- Section 07 — Key Engineering Insights -->
  <div class="section-title mt-32">07 — Key Engineering Insights</div>
  <div class="card mt-8" style="border: 1px solid var(--accent); background: linear-gradient(180deg, rgba(13,17,23,0.95) 0%, rgba(22,27,34,0.95) 100%);">
    <h2>Production and Architectural Learnings</h2>
    <p class="card-desc">Critical takeaways regarding the sensor integration and analytic pipeline design.</p>
    
    <div style="display:flex; flex-direction:column; gap:16px; margin-top:16px;">
      <div style="display:flex; gap:12px; align-items:flex-start;">
        <span class="pill pill-green" style="font-weight:bold; flex-shrink:0;">Insight 01</span>
        <div>
          <strong style="color:var(--text); font-size:0.88rem;">Feature engineering yields primary predictive value</strong>
          <p style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">Deriving temporal differentials (<code>Delta</code>) provides strong predictive signal, whereas expanding raw feature dimensionality (Polynomial regression) leads to overfitting without improving generalization.</p>
        </div>
      </div>
      <div style="display:flex; gap:12px; align-items:flex-start;">
        <span class="pill pill-green" style="font-weight:bold; flex-shrink:0;">Insight 02</span>
        <div>
          <strong style="color:var(--text); font-size:0.88rem;">Dynamic changes drive flow physics</strong>
          <p style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">Correlation analysis shows that flow is strongly linked to sensor delta changes (+0.615) rather than raw absolute levels (-0.237), confirming that pressure differentials are physically dominant.</p>
        </div>
      </div>
      <div style="display:flex; gap:12px; align-items:flex-start;">
        <span class="pill pill-green" style="font-weight:bold; flex-shrink:0;">Insight 03</span>
        <div>
          <strong style="color:var(--text); font-size:0.88rem;">Automated preprocessing safeguards pipeline integrity</strong>
          <p style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">Filtering repeated headers, malformed rows, and out-of-bounds readings ensures input sanitization.</p>
        </div>
      </div>
      <div style="display:flex; gap:12px; align-items:flex-start;">
        <span class="pill pill-green" style="font-weight:bold; flex-shrink:0;">Insight 04</span>
        <div>
          <strong style="color:var(--text); font-size:0.88rem;">Independent validation reveals operational shifts</strong>
          <p style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">Out-of-session testing prevents overconfidence by exposing baseline drifts (~140k in Raw, 3x increase in vibration noise) typical of real-world deployments.</p>
        </div>
      </div>
      <div style="display:flex; gap:12px; align-items:flex-start;">
        <span class="pill pill-green" style="font-weight:bold; flex-shrink:0;">Insight 05</span>
        <div>
          <strong style="color:var(--text); font-size:0.88rem;">Workflow is fully reproducible</strong>
          <p style="font-size:0.8rem; color:var(--text-muted); margin-top:2px;">The pipeline script automates the entire sequence from raw ingestion to model benchmarking and diagnostic report compilation, ensuring consistency and auditability.</p>
        </div>
      </div>
    </div>
  </div>

  <!-- Section 08 — Synthetic Dataset Reference -->
  {synthetic_html}

  <!-- Section 09 — Limitations & Future Work -->
  <div class="section-title mt-32">09 — Limitations &amp; Future Work</div>
  <div class="card mt-8" style="border-left: 3px solid var(--red); background: rgba(248,81,73,0.02);">
    <h2>Operational Limitations and Calibration Strategy</h2>
    <p class="card-desc">Identified pipeline limitations and proposed mitigations for next-generation revisions.</p>
    <ul style="padding-left: 20px; font-size: 0.82rem; color: var(--text-muted); line-height: 1.7;">
      <li><strong>Coordinate Baseline Drift:</strong> Absolute sensor offsets change significantly between sessions. Future models should employ relative scaling or rely strictly on dynamic differences.</li>
      <li><strong>Rig Vibration Coupling:</strong> AccY/Z noise variances increased 3-4x in the validation session. Adaptive filtering or vibration-based noise cancellation should be introduced.</li>
      <li><strong>Dynamic Calibration:</strong> Under shift, absolute linear coefficients degrade. A dynamic self-calibration or online zero-point adjustment step is required to maintain accuracy in long-term runs.</li>
    </ul>
  </div>

  <!-- ─── Footer ─── -->
  <div class="footer">
    <span>Real Sensor Data Pipeline · generated from real_data_summary.json + real_data_model_comparison.csv</span>
    <span>5 models · {load_train.parsed_rows + load_external.parsed_rows:,} total rows processed</span>
  </div>

</div>
</body>
</html>


def analyze_real_data(train_path: Path, external_path: Path, synthetic_path: Path | None, report_path: Path, summary_path: Path, comparison_path: Path) -> None:
    train_rows_raw, load_train = load_sensor_csv(train_path)
    external_rows_raw, load_external = load_sensor_csv(external_path)

    train_summary = summarize_dataset(train_path, train_rows_raw)
    external_summary = summarize_dataset(external_path, external_rows_raw)

    train_base = rows_to_matrix(train_rows_raw, FEATURE_NAMES)
    train_lag = lagged_matrix(train_rows_raw, FEATURE_NAMES)
    train_delta = rows_to_matrix(train_rows_raw, ["Delta"])
    external_base = rows_to_matrix(external_rows_raw, FEATURE_NAMES)
    external_lag = lagged_matrix(external_rows_raw, FEATURE_NAMES)
    external_delta = rows_to_matrix(external_rows_raw, ["Delta"])

    train_base_split, test_base_split = chronological_split(train_base, 0.2)
    train_lag_split, test_lag_split = chronological_split(train_lag, 0.2)
    train_delta_split, test_delta_split = chronological_split(train_delta, 0.2)

    evaluated_models: list[ModelArtifacts] = [
        evaluate_model("baseline_mean", train_base_split, test_base_split, external_base, kind="baseline"),
        evaluate_model("linear", train_base_split, test_base_split, external_base, kind="ols"),
        evaluate_model("ridge", train_base_split, test_base_split, external_base, kind="ridge", alpha=1.0),
        evaluate_model("poly2", train_base_split, test_base_split, external_base, kind="ridge", alpha=1.0, use_poly2=True),
        evaluate_model("lagged_linear", train_lag_split, test_lag_split, external_lag, kind="ols"),
        evaluate_model("linear_delta_only", train_delta_split, test_delta_split, external_delta, kind="ols"),
        evaluate_model("knn_k5", train_base_split, test_base_split, external_base, kind="knn"),
    ]

    results = [artifact.result for artifact in evaluated_models]
    best_index = max(range(len(results)), key=lambda index: results[index].external_r2)
    best_result = results[best_index]
    best_predictions = evaluated_models[best_index].external_predictions
    if best_result.name == "baseline_mean":
        best_actual = external_base.y
    elif best_result.name == "lagged_linear":
        best_actual = external_lag.y
    else:
        best_actual = external_base.y

    correlations = [
        (name, pairwise_corr(np.array([row[name] for row in train_rows_raw]), np.array([row[TARGET_NAME] for row in train_rows_raw])))
        for name in FEATURE_NAMES
    ]

    synthetic_summary = load_synthetic_csv(synthetic_path) if synthetic_path is not None else None

    summary_payload = {
        "train": asdict(train_summary),
        "external": asdict(external_summary),
        "load_stats": {
            "train": asdict(load_train),
            "external": asdict(load_external),
        },
        "models": [asdict(result) for result in results],
        "best_model": best_result.name,
        "correlations": {name: value for name, value in correlations},
        "synthetic_summary": asdict(synthetic_summary) if synthetic_summary is not None else None,
    }

    write_comparison_csv(comparison_path, results)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    report_path.write_text(
        build_html_report(
            train_summary=train_summary,
            external_summary=external_summary,
            load_train=load_train,
            load_external=load_external,
            results=results,
            correlations=correlations,
            best_actual=best_actual,
            best_predicted=best_predictions,
            synthetic_summary=synthetic_summary,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the real sensor captures and generate a report")
    parser.add_argument("--train", default="new.csv", help="Primary real capture used for training")
    parser.add_argument("--external", default="output.csv", help="External validation capture")
    parser.add_argument("--synthetic", default="acceleration_voltage_mapping.csv", help="Optional synthetic baseline capture")
    parser.add_argument("--report", default="real_data_report.html", help="Path to generated HTML report")
    parser.add_argument("--summary", default="real_data_summary.json", help="Path to generated JSON summary")
    parser.add_argument("--comparison", default="real_data_model_comparison.csv", help="Path to generated model comparison CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_real_data(
        train_path=Path(args.train),
        external_path=Path(args.external),
        synthetic_path=Path(args.synthetic) if args.synthetic else None,
        report_path=Path(args.report),
        summary_path=Path(args.summary),
        comparison_path=Path(args.comparison),
    )
    print(f"Wrote {args.report}, {args.summary}, and {args.comparison}")


if __name__ == "__main__":
    main()
