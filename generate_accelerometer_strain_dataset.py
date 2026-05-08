#!/usr/bin/env python3
"""Generate a simple acceleration-to-voltage mapping dataset."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    samples: int = 360
    dt_seconds: float = 0.02
    random_seed: int = 42


def generate_rows(cfg: DatasetConfig) -> list[dict[str, float | int]]:
    rng = __import__('random').Random(cfg.random_seed)
    rows: list[dict[str, float | int]] = []

    for sample_id in range(cfg.samples):
        time_s = sample_id * cfg.dt_seconds

        acc_x = (
            2.2 * math.sin(2.0 * math.pi * 0.45 * time_s)
            + 0.6 * math.sin(2.0 * math.pi * 1.7 * time_s + 0.4)
            + rng.gauss(0.0, 0.12)
        )
        acc_y = (
            1.6 * math.cos(2.0 * math.pi * 0.33 * time_s + 0.9)
            + 0.5 * math.sin(2.0 * math.pi * 1.2 * time_s + 1.1)
            + rng.gauss(0.0, 0.10)
        )
        acc_z = (
            9.81
            + 0.85 * math.sin(2.0 * math.pi * 0.52 * time_s + 0.2)
            + 0.28 * math.cos(2.0 * math.pi * 1.4 * time_s)
            + rng.gauss(0.0, 0.08)
        )

        voltage_v = (
            0.018 * acc_x
            - 0.014 * acc_y
            + 0.022 * (acc_z - 9.81)
            + rng.gauss(0.0, 0.0025)
        )

        rows.append(
            {
                "sample_id": sample_id,
                "acc_x_mps2": round(acc_x, 6),
                "acc_y_mps2": round(acc_y, 6),
                "acc_z_mps2": round(acc_z, 6),
                "voltage_v": round(voltage_v, 6),
            }
        )

    return rows


def write_csv(rows: list[dict[str, float | int]], path: str) -> None:
    if not rows:
        raise ValueError("No rows generated")

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate acceleration-to-voltage mapping dataset")
    parser.add_argument("--out", default="acceleration_voltage_mapping.csv", help="Output CSV path")
    parser.add_argument("--samples", type=int, default=360, help="Number of samples to generate")
    parser.add_argument("--dt", type=float, default=0.02, help="Sample spacing in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetConfig(samples=args.samples, dt_seconds=args.dt, random_seed=args.seed)
    rows = generate_rows(cfg)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
