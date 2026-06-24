#!/usr/bin/env python3
"""Train a linear regression model that maps acceleration to voltage."""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np


FEATURE_NAMES = ["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]
TARGET_NAME = "voltage_v"


def load_dataset(path: str) -> dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data found in {path}")

    features = np.array([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=float)
    target = np.array([float(row[TARGET_NAME]) for row in rows], dtype=float)

    return {
        "features": features,
        "target": target,
    }


def train_test_split_indices(n_samples: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    test_size = max(1, int(round(n_samples * test_fraction)))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return train_indices, test_indices


def fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    design_matrix = np.column_stack([np.ones(len(x_train)), x_train])
    coefficients, *_ = np.linalg.lstsq(design_matrix, y_train, rcond=None)
    return coefficients


def predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design_matrix = np.column_stack([np.ones(len(x)), x])
    return design_matrix @ coefficients


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def save_model(path: str, coefficients: np.ndarray) -> None:
    payload = {
        "feature_names": FEATURE_NAMES,
        "intercept_v": float(coefficients[0]),
        "coefficients_v_per_ms2": {
            FEATURE_NAMES[0]: float(coefficients[1]),
            FEATURE_NAMES[1]: float(coefficients[2]),
            FEATURE_NAMES[2]: float(coefficients[3]),
        },
    }
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear model mapping acceleration to voltage")
    parser.add_argument("--data", default="acceleration_voltage_mapping.csv", help="Path to dataset CSV")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of samples used for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split")
    parser.add_argument("--model-out", default="acceleration_voltage_model.json", help="Where to save fitted coefficients")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.data)
    x = data["features"]
    y = data["target"]

    train_indices, test_indices = train_test_split_indices(len(x), args.test_fraction, args.seed)
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    coefficients = fit_linear_regression(x_train, y_train)
    y_pred = predict(x_test, coefficients)

    print("Linear regression: acceleration -> voltage")
    print(f"Intercept: {coefficients[0]:.6f} V")
    for name, value in zip(FEATURE_NAMES, coefficients[1:]):
        print(f"{name}: {value:.6f} V per m/s^2")

    print()
    print(f"Test R^2: {r2_score(y_test, y_pred):.4f}")
    print(f"Test MAE: {mae(y_test, y_pred):.6f} V")
    print(f"Test RMSE: {rmse(y_test, y_pred):.6f} V")

    save_model(args.model_out, coefficients)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
