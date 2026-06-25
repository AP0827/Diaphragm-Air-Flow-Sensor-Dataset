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

if __name__ == "__main__":
    main()
