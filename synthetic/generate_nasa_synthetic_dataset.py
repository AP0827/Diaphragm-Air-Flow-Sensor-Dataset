#!/usr/bin/env python3
"""Generate a physics-informed synthetic dataset for vibration-distorted pressure sensing.

Model assumptions reflect the NASA insights in Context.txt:
- Distortion enters at displacement level: x_total = x_pressure + x_vibration
- Vibration sensitivity is resonance and Q dependent
- Mechanical and electrical losses are structured bias terms
- Pressure relationship is nonlinear with continuum/free-molecular blending
- Sensor has memory with tau = Q / (pi * f0)
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from typing import List


@dataclass
class SimulationConfig:
    num_sequences: int = 300
    seq_len: int = 80
    dt_seconds: float = 0.004
    pressure_min_pa: float = 50.0
    pressure_max_pa: float = 120000.0
    pressure_transition_pa: float = 8000.0
    blend_sharpness: float = 2.4
    q_ref: float = 100.0
    bandwidth_k: float = 2.0
    random_seed: int = 42


@dataclass
class SequenceParams:
    f0_hz: float
    q_base: float
    k_pressure_um: float
    k_vibration_um: float
    pressure_slew_pa_per_s: float
    pressure_wave_amp_pa: float
    pressure_wave_freq_hz: float
    vib_base_accel_mps2: float
    vib_freq_hz: float
    vib_mod_freq_hz: float
    pm_base: float
    pr_base: float
    k_energy_to_voltage: float
    gain_v_per_um: float
    offset_v: float


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def pressure_to_displacement_um(
    pressure_pa: float,
    pressure_transition_pa: float,
    blend_sharpness: float,
    k_pressure_um: float,
) -> float:
    """Nonlinear pressure->displacement map with two-regime blending."""
    p = max(pressure_pa, 1e-9)

    # Regime blending: low pressure favors free molecular response,
    # high pressure favors continuum response.
    w_cont = 1.0 / (1.0 + (pressure_transition_pa / p) ** blend_sharpness)

    # Simple nonlinear basis for each regime.
    x_free = k_pressure_um * 0.55 * math.sqrt(p / pressure_transition_pa)
    x_cont = k_pressure_um * (0.25 + 0.95 * (p / pressure_transition_pa) ** 0.92)

    return (1.0 - w_cont) * x_free + w_cont * x_cont


def pressure_loss_term(pressure_pa: float, pressure_transition_pa: float, blend_sharpness: float) -> float:
    """Structured pressure-dependent damping/loss proxy f(P, gas)."""
    p = max(pressure_pa, 1e-9)
    w_cont = 1.0 / (1.0 + (pressure_transition_pa / p) ** blend_sharpness)
    loss_free = 0.08 * math.sqrt(p)
    loss_cont = 0.00095 * p
    return (1.0 - w_cont) * loss_free + w_cont * loss_cont


def q_from_pressure(q_base: float, pressure_loss: float) -> float:
    """Higher dissipation lowers Q; keep Q in a practical range."""
    q = q_base / (1.0 + 0.035 * pressure_loss)
    return clamp(q, 8.0, 450.0)


def resonance_gain(freq_hz: float, f0_hz: float, q_value: float) -> float:
    """Second-order response magnitude near resonance."""
    r = freq_hz / max(f0_hz, 1e-9)
    denom = math.sqrt((1.0 - r * r) ** 2 + (r / max(q_value, 1e-9)) ** 2)
    return 1.0 / max(denom, 1e-9)


def resonance_bandwidth_weight(freq_hz: float, f0_hz: float, q_value: float, bandwidth_k: float) -> float:
    """Softly suppress vibration coupling outside the resonance bandwidth region."""
    bw_hz = max(f0_hz / max(q_value, 1e-9), 1e-9)
    limit_hz = max(bandwidth_k, 1e-9) * bw_hz
    delta_hz = abs(freq_hz - f0_hz)
    if delta_hz <= limit_hz:
        return 1.0
    z = (delta_hz - limit_hz) / limit_hz
    return math.exp(-3.0 * z * z)


def sample_sequence_params(rng: random.Random) -> SequenceParams:
    return SequenceParams(
        f0_hz=rng.uniform(250.0, 1400.0),
        q_base=rng.uniform(40.0, 220.0),
        k_pressure_um=rng.uniform(0.9, 2.0),
        k_vibration_um=rng.uniform(0.004, 0.024),
        pressure_slew_pa_per_s=rng.uniform(-26000.0, 26000.0),
        pressure_wave_amp_pa=rng.uniform(300.0, 9000.0),
        pressure_wave_freq_hz=rng.uniform(0.2, 2.4),
        vib_base_accel_mps2=rng.uniform(0.2, 26.0),
        vib_freq_hz=rng.uniform(15.0, 2200.0),
        vib_mod_freq_hz=rng.uniform(0.2, 2.2),
        pm_base=rng.uniform(-0.028, 0.028),
        pr_base=rng.uniform(-0.022, 0.022),
        k_energy_to_voltage=rng.uniform(0.00015, 0.0012),
        gain_v_per_um=rng.uniform(0.35, 1.45),
        offset_v=rng.uniform(0.08, 0.65),
    )


def generate_dataset(cfg: SimulationConfig) -> List[dict]:
    rng = random.Random(cfg.random_seed)
    rows: List[dict] = []

    for seq in range(cfg.num_sequences):
        p = rng.uniform(cfg.pressure_min_pa, cfg.pressure_max_pa)
        params = sample_sequence_params(rng)

        x_total_dyn_um = 0.0
        x_pressure_prev_um = 0.0

        for t in range(cfg.seq_len):
            time_s = t * cfg.dt_seconds

            # Evolving true pressure profile per sequence.
            p += params.pressure_slew_pa_per_s * cfg.dt_seconds
            p += params.pressure_wave_amp_pa * math.sin(2.0 * math.pi * params.pressure_wave_freq_hz * time_s) * cfg.dt_seconds
            p = clamp(p, cfg.pressure_min_pa, cfg.pressure_max_pa)

            x_pressure_um = pressure_to_displacement_um(
                pressure_pa=p,
                pressure_transition_pa=cfg.pressure_transition_pa,
                blend_sharpness=cfg.blend_sharpness,
                k_pressure_um=params.k_pressure_um,
            )

            loss_pressure = pressure_loss_term(
                pressure_pa=p,
                pressure_transition_pa=cfg.pressure_transition_pa,
                blend_sharpness=cfg.blend_sharpness,
            )
            q_val = q_from_pressure(params.q_base, loss_pressure)
            gain_res = resonance_gain(params.vib_freq_hz, params.f0_hz, q_val)
            bw_weight = resonance_bandwidth_weight(
                freq_hz=params.vib_freq_hz,
                f0_hz=params.f0_hz,
                q_value=q_val,
                bandwidth_k=cfg.bandwidth_k,
            )

            # Vibration acceleration has amplitude modulation.
            accel_mps2 = params.vib_base_accel_mps2 * (
                1.0
                + 0.33 * math.sin(2.0 * math.pi * params.vib_mod_freq_hz * time_s)
                + 0.08 * math.sin(2.0 * math.pi * 0.5 * params.vib_mod_freq_hz * time_s + 1.4)
            )
            accel_mps2 = max(0.0, accel_mps2)

            q_coupling = q_val / max(cfg.q_ref, 1e-9)
            x_vibration_um = params.k_vibration_um * accel_mps2 * gain_res * q_coupling * bw_weight
            x_total_inst_um = x_pressure_um + x_vibration_um

            # Time response: first-order relaxation with tau = Q/(pi*f0).
            tau = q_val / (math.pi * max(params.f0_hz, 1e-6))
            alpha = cfg.dt_seconds / max(tau, cfg.dt_seconds)
            alpha = clamp(alpha, 0.0, 1.0)
            x_total_dyn_um = x_total_dyn_um + alpha * (x_total_inst_um - x_total_dyn_um)

            # NASA-inspired structured losses (bias terms), slow drift plus pressure coupling.
            pm = params.pm_base + 0.012 * math.sin(2.0 * math.pi * 0.06 * time_s + 0.3 * seq)
            pr = params.pr_base + 0.010 * math.sin(2.0 * math.pi * 0.09 * time_s + 0.7 * seq)

            # Drive-energy proxy from NASA relation (V0*V1/x1 = f(P)+Pm+Pr).
            # We expose it as a feature for physics-informed models.
            drive_energy_proxy = loss_pressure + pm + pr

            # Voltage from displacement + structured terms + small measurement noise.
            v_clean = params.gain_v_per_um * x_pressure_um + params.offset_v
            v_measured = (
                params.gain_v_per_um * x_total_dyn_um
                + params.offset_v
                + 0.10 * pm
                + 0.08 * pr
                + params.k_energy_to_voltage * drive_energy_proxy
                + rng.gauss(0.0, 0.003)
            )

            rows.append(
                {
                    "sequence_id": seq,
                    "time_step": t,
                    "time_s": round(time_s, 6),
                    "pressure_true_pa": round(p, 6),
                    "measured_voltage_v": round(v_measured, 8),
                    "clean_voltage_v": round(v_clean, 8),
                    "vibration_accel_mps2": round(accel_mps2, 8),
                    "vibration_freq_hz": round(params.vib_freq_hz, 8),
                    "f0_hz": round(params.f0_hz, 8),
                    "q_factor": round(q_val, 8),
                    "resonance_gain": round(gain_res, 8),
                    "resonance_bandwidth_weight": round(bw_weight, 8),
                    "x_pressure_um": round(x_pressure_um, 8),
                    "x_vibration_um": round(x_vibration_um, 8),
                    "x_vibration_to_x_pressure_ratio": round(abs(x_vibration_um) / max(abs(x_pressure_um), 1e-9), 8),
                    "x_total_dyn_um": round(x_total_dyn_um, 8),
                    "pm": round(pm, 8),
                    "pr": round(pr, 8),
                    "drive_energy_proxy": round(drive_energy_proxy, 8),
                    "d_xpressure_dt": round((x_pressure_um - x_pressure_prev_um) / max(cfg.dt_seconds, 1e-9), 8),
                }
            )

            x_pressure_prev_um = x_pressure_um

    return rows


def write_csv(rows: List[dict], path: str) -> None:
    if not rows:
        raise ValueError("No rows generated")

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NASA-informed synthetic pressure-vibration dataset")
    parser.add_argument("--out", default="synthetic_pressure_vibration_dataset.csv", help="Output CSV path")
    parser.add_argument("--num-sequences", type=int, default=300, help="Number of sequences")
    parser.add_argument("--seq-len", type=int, default=80, help="Timesteps per sequence")
    parser.add_argument("--dt", type=float, default=0.004, help="Timestep in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig(
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        dt_seconds=args.dt,
        random_seed=args.seed,
    )
    rows = generate_dataset(cfg)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
