# Diaphragm-Air-Flow-Sensor-Dataset
# Synthetic Dataset Generation Documentation

## Acceleration-to-Voltage Mapping Dataset

This is a simple dataset and calibration framework that maps tri-axial acceleration measurements to the voltage they induce on a strain-gauge sensor.

## 1) The Model

A three-axis accelerometer provides the input acceleration vector:

$$
\mathbf{a}(t) = \begin{bmatrix} a_x(t) \\ a_y(t) \\ a_z(t) \end{bmatrix}.
$$

The voltage induced by this acceleration is approximately linear over typical operating ranges:

$$
V(t) = \beta_0 + \beta_x a_x(t) + \beta_y a_y(t) + \beta_z a_z(t) + \varepsilon(t).
$$

The goal is to learn the coefficients $\beta_0, \beta_x, \beta_y, \beta_z$ from training data using linear regression, then use the fitted model to predict voltage from new acceleration measurements.

## 2) Dataset

The dataset is stored in [acceleration_voltage_mapping.csv](acceleration_voltage_mapping.csv) and contains:

- `sample_id`: row index
- `acc_x_mps2`: x-axis acceleration (m/s²)
- `acc_y_mps2`: y-axis acceleration (m/s²)
- `acc_z_mps2`: z-axis acceleration (m/s²)
- `voltage_v`: the voltage measured on the strain gauge (V)

Each row is one simultaneous measurement of acceleration and the voltage it produces.

## 3) Training

Run the training script to fit the linear model:

```bash
python3 train_acceleration_linear_regression.py --data acceleration_voltage_mapping.csv
```

The script will:

1. Load the dataset.
2. Split it into training and test sets (80/20 by default).
3. Fit a linear regression model on the training set.
4. Evaluate on the test set and report $R^2$, MAE, and RMSE.
5. Save the fitted coefficients to `acceleration_voltage_model.json`.

## 4) Generating Synthetic Data

To create a new dataset with different parameters:

```bash
python3 generate_accelerometer_strain_dataset.py --samples 500 --out new_dataset.csv
```

Main options:

- `--samples`: Number of data points to generate (default 360)
- `--dt`: Sample spacing in seconds (default 0.02)
- `--seed`: Random seed for reproducibility (default 42)
- `--out`: Output CSV filename

## 5) Model Output

After training, the coefficients are saved to JSON in a format like:

```json
{
  "feature_names": ["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"],
  "intercept_v": -0.2164,
  "coefficients_v_per_ms2": {
    "acc_x_mps2": 0.0181,
    "acc_y_mps2": -0.0141,
    "acc_z_mps2": 0.0221
  }
}
```

Use these values to predict voltage from new acceleration vectors.

## 6) Quick Summary

This is a direct mapping from tri-axial acceleration to induced voltage:
- **Input**: $a_x, a_y, a_z$
- **Output**: $V$
- **Method**: Linear regression
- **No decomposition, no latent variables, just a straightforward calibration.**
