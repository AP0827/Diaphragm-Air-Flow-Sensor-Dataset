# Diaphragm-Air-Flow-Sensor-Dataset
# Synthetic Dataset Generation Documentation

This document describes the full physics-informed generation process implemented in [generate_nasa_synthetic_dataset.py](generate_nasa_synthetic_dataset.py), why each term exists, and how trustworthy the generated data is for real-world modeling.

Important modeling note:

NASA discusses an energy-like observable proportional to $V_0V_1/x_1$. In this generator, voltage is modeled as a displacement-readout surrogate (with additional energy-proxy coupling), not as the raw NASA energy observable itself.

## 1) Objective

Build a synthetic dataset for learning:

- Inputs: measured voltage and vibration conditions
- Target: true pressure

under the NASA-consistent principle:

Pressure is corrupted by vibration at the displacement level, not by post-measurement additive white noise.

## 2) Physics mapping used

The generator follows this chain:

Pressure -> displacement -> dynamic displacement under vibration -> voltage

with resonance, Q dependence, and structured bias terms.

### 2.1 Displacement decomposition

$$
x_{total} = x_{pressure} + x_{vibration}
$$

This is the core structural choice.

### 2.2 Pressure to displacement mapping (nonlinear, regime-blended)

We blend low-pressure and high-pressure response regimes:

$$
w_{cont}(P) = \frac{1}{1 + (P_t/P)^s}
$$

$$
x_{pressure}(P) = (1-w_{cont}) \cdot x_{free}(P) + w_{cont} \cdot x_{cont}(P)
$$

where:

- $x_{free}(P) \propto \sqrt{P}$
- $x_{cont}(P) \propto P^{0.92}$

This enforces nonlinearity and transition behavior.

### 2.3 Pressure-dependent dissipation proxy

$$
loss_{pressure} = f(P)
$$

with the same regime blend idea. This represents pressure-dependent damping/load term.

### 2.4 Q factor dependence on dissipation

$$
Q(P) = \frac{Q_{base}}{1 + c \cdot loss_{pressure}}
$$

clamped to a practical range.

### 2.5 Resonance gain

$$
r = \frac{f_{vib}}{f_0}
$$

$$
gain_{res} = \frac{1}{\sqrt{(1-r^2)^2 + (r/Q)^2}}
$$

Only vibration near resonance becomes strongly amplified.

### 2.6 Vibration-induced displacement

$$
x_{vibration} = k_a \cdot a(t) \cdot gain_{res} \cdot g(Q) \cdot w_{bw}
$$

where:

- $g(Q) = Q/Q_{ref}$
- $w_{bw}$ is a soft resonance-bandwidth weight

Acceleration is slowly modulated over time to avoid trivial constant-excitation sequences.

Bandwidth weighting enforces narrow-band sensitivity around resonance:

$$
bw = f_0/Q
$$

$$
w_{bw} =
\begin{cases}
1, & |f_{vib} - f_0| \le k \cdot bw \\
\exp\{-3z^2\}, & \text{otherwise}
\end{cases}
$$

with $z = (|f_{vib}-f_0| - k\cdot bw)/(k\cdot bw)$.

### 2.7 Dynamic response (sensor memory)

$$
tau = \frac{Q}{\pi f_0}
$$

$$
x_{dyn}[t] = x_{dyn}[t-1] + \alpha \cdot (x_{inst}[t] - x_{dyn}[t-1])
$$

with $\alpha = dt/\max(tau, dt)$.

### 2.8 Structured bias terms

$$
drive\_energy\_proxy = f(P) + P_m + P_r
$$

where $P_m$ and $P_r$ are slow-varying sequence-level drift terms (not i.i.d. white noise).

### 2.9 Voltage generation

$$
V_{clean} = gain \cdot x_{pressure} + offset
$$

$$
V_{measured} = gain \cdot x_{dyn} + offset + 0.10P_m + 0.08P_r + k_e \cdot drive\_energy\_proxy + \epsilon
$$

with small Gaussian measurement noise $\epsilon$.

## 3) Parameter sampling strategy

Each sequence samples fixed hardware/environment parameters to mimic sensor-to-sensor and condition-to-condition variability:

- Natural frequency $f_0$: 250 to 1400 Hz
- Baseline Q: 40 to 220
- Vibration frequency: 15 to 2200 Hz
- Base acceleration: 0.2 to 26 m/s^2
- Gain and offset randomized per sequence
- Pressure trajectory includes slope plus low-frequency oscillation

This creates broad but structured coverage.

## 4) Output schema and meaning

Generated CSV columns:

- sequence_id: sequence identity for grouped splits
- time_step, time_s: temporal index
- pressure_true_pa: regression target
- measured_voltage_v: distorted observable
- clean_voltage_v: no-vibration reference channel
- vibration_accel_mps2, vibration_freq_hz: disturbance descriptors
- f0_hz, q_factor, resonance_gain: dynamic sensitivity descriptors
- resonance_bandwidth_weight: narrow-band coupling strength factor
- x_pressure_um, x_vibration_um, x_total_dyn_um: latent physics states
- x_vibration_to_x_pressure_ratio: corruption-level indicator
- pm, pr: structured bias components
- drive_energy_proxy: pressure-plus-loss proxy feature
- d_xpressure_dt: pressure-state derivative proxy

## 5) Data split guidance

Do not random-row split. Use grouped splits by sequence_id.

- Train: 70%
- Validation: 15%
- Test: 15%

This avoids leakage from temporal adjacency.

## 6) Modeling guidance

Baseline regressors:

- Gradient boosting regressor
- Random forest regressor

Temporal models:

- 1D CNN
- LSTM or GRU

Core input set:

- measured_voltage_v
- vibration_accel_mps2
- vibration_freq_hz

Useful extended set:

- f0_hz
- q_factor
- resonance_gain
- resonance_bandwidth_weight
- drive_energy_proxy
- x_vibration_to_x_pressure_ratio

## 7) Realism assessment: does it make sense with real data?

Short answer: yes for structure, not yet for absolute metrology.

### 7.1 What is physically credible already

- Distortion enters at displacement before readout.
- Vibration impact increases near resonance.
- Sensitivity depends on pressure through Q and dissipation.
- Bias is structured drift, not only random noise.
- Dynamic lag exists through $\tau = Q/(\pi f_0)$.

These are the right causal ingredients.

### 7.2 What is still synthetic/assumed

- Coefficients are plausible but not calibrated to one exact hardware unit.
- Gas composition, temperature, packaging modes, and electronic transfer details are simplified.
- Excitation spectrum is parametric, not measured PSD from a shaker profile.
- Bias drift laws are heuristic sinusoidal forms.

Therefore this dataset is best treated as a physics-informed pretraining and ablation dataset, not a final certification dataset.

### 7.3 Quantitative sanity checks on current generated CSV

From [synthetic_pressure_vibration_dataset.csv](synthetic_pressure_vibration_dataset.csv):

- Rows: 28,800 (+ header)
- Pressure range: 50 to 120,000 Pa
- Q range: 8.68 to 191.23
- Resonance gain range: 0.020 to 57.98
- Mean resonance bandwidth weight: 0.069
- Fraction with non-negligible bandwidth weight (>1e-6): 0.124
- Mean ratio $|x_{vibration}|/|x_{pressure}|$: 0.035
- Corr(pressure, clean_voltage): 0.740
- Corr(pressure, measured_voltage): 0.710
- Corr(|measured-clean|, resonance_gain): 0.430
- Sequence-wise monotonic tendency of clean voltage vs pressure: 1.0 mean

Interpretation:

- The corrupted channel is measurably less pressure-faithful than the clean channel.
- Distortion rises with resonance gain as expected.
- Pressure-to-clean-voltage relation remains physically ordered.

So the generated data is internally consistent with the intended physics behavior.

## 8) Calibration path to make it closer to real deployment

Use this sequence when real data becomes available:

1. Fit static pressure-to-clean-voltage coefficients from bench calibration.
2. Fit resonance transfer shape using shaker sweep around measured $f_0$.
3. Fit Q vs pressure from bandwidth measurements at multiple pressures.
4. Fit drift model $P_m, P_r$ from long stable holds.
5. Re-run generation with calibrated coefficient priors.
6. Evaluate sim-to-real gap on held-out real segments.

## 9) Run instructions

Generate CSV:

python3 generate_nasa_synthetic_dataset.py --out synthetic_pressure_vibration_dataset.csv

Main options:

- --num-sequences
- --seq-len
- --dt
- --seed

## 10) Recommended documentation use in your report

Use this synthetic process as:

- A physics-informed data engine for controlled experiments
- A way to test whether models can separate pressure from vibration coupling
- A pretraining source before limited real-data fine-tuning

Avoid claiming direct field-level absolute accuracy until coefficients are calibrated to real hardware measurements.
