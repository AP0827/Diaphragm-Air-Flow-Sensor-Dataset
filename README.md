# Diaphragm-Air-Flow-Sensor-Dataset
# Synthetic Dataset Generation Documentation

## Diaphragm Strain-Gauge Calibration Dataset

This project now uses a simpler calibration workflow:

1. Measure tri-axial acceleration with an accelerometer.
2. Measure the diaphragm strain-gauge voltage at the same time.
3. Learn how much of the voltage is caused by acceleration.
4. Subtract the predicted acceleration voltage from the measured reading to estimate the true signal.

The key idea is that, for small strains, the strain-gauge bridge output is approximately linear. That makes linear regression a good first calibration model.

## 1) Measurement model

Let the accelerometer output be the acceleration vector

$$
\mathbf{a}(t) = \begin{bmatrix} a_x(t) \\ a_y(t) \\ a_z(t) \end{bmatrix}.
$$

This is the input feature vector used for calibration.

The diaphragm strain is approximated as a linear function of the three axes:

$$
\epsilon_g(t) = k_x a_x(t) + k_y a_y(t) + k_z a_z(t).
$$

This says that acceleration bends the diaphragm and produces strain in the gauge.

For a Wheatstone-bridge strain gauge, the small-signal voltage is approximately proportional to strain:

$$
V_{acc}(t) \approx V_{exc} \cdot \frac{GF}{4} \cdot \epsilon_g(t).
$$

Here:

- $V_{exc}$ is the bridge excitation voltage
- $GF$ is the gauge factor
- the factor $1/4$ comes from the small-signal bridge approximation

After expanding the constants, the acceleration-induced voltage can be written in regression form:

$$
V_{acc}(t) = \beta_0 + \beta_x a_x(t) + \beta_y a_y(t) + \beta_z a_z(t) + \varepsilon(t).
$$

This is the model we train.

The measured voltage is then:

$$
V_{meas}(t) = V_{true}(t) + V_{acc}(t).
$$

So the corrected estimate of the true voltage is:

$$
\hat V_{true}(t) = V_{meas}(t) - \hat V_{acc}(t).
$$

That subtraction is the whole calibration idea.

## 2) Linear regression objective

Given training samples $(a_x^{(i)}, a_y^{(i)}, a_z^{(i)}, V_{acc}^{(i)})$, the linear model parameters are chosen to minimize mean squared error:

$$
\hat\beta = \arg\min_{\beta} \sum_{i=1}^{N} \left(V_{acc}^{(i)} - (\beta_0 + \beta_x a_x^{(i)} + \beta_y a_y^{(i)} + \beta_z a_z^{(i)})\right)^2.
$$

This gives the best straight-line fit in the least-squares sense.

## 3) Dataset columns

The synthetic dataset is stored in [synthetic_pressure_vibration_dataset.csv](synthetic_pressure_vibration_dataset.csv) and contains:

- `sample_id`: row index
- `time_s`: timestamp for the sample
- `acc_x_mps2`: x-axis acceleration
- `acc_y_mps2`: y-axis acceleration
- `acc_z_mps2`: z-axis acceleration
- `true_voltage_v`: the underlying diaphragm voltage before acceleration distortion
- `accel_induced_voltage_v`: voltage contribution caused by acceleration
- `measured_voltage_v`: observed voltage, equal to true plus acceleration-induced voltage

## 4) Training script

Train the regression model with [train_acceleration_linear_regression.py](train_acceleration_linear_regression.py).

The script:

1. Loads the CSV.
2. Fits a linear regression model to predict `accel_induced_voltage_v` from the three acceleration axes.
3. Evaluates the model on a holdout split.
4. Subtracts the predicted acceleration voltage from `measured_voltage_v`.
5. Reports how close the corrected voltage is to `true_voltage_v`.

## 5) Why this method is appropriate

This approach matches the sensor physics at small strain:

- strain gauges are approximately linear over a normal operating range
- the bridge output is proportional to strain
- acceleration can be modeled as a disturbance term that enters the voltage reading
- tri-axial acceleration gives the model enough information to learn direction-dependent coupling

That makes the model simple, explainable, and practical for calibration.

## 6) Run instructions

Generate the dataset:

```bash
python3 generate_accelerometer_strain_dataset.py --out synthetic_pressure_vibration_dataset.csv
```

Train the model:

```bash
python3 train_acceleration_linear_regression.py --data synthetic_pressure_vibration_dataset.csv
```

## 7) Short presentation version

You can describe the method like this:

"We measure acceleration in three axes and learn a linear calibration model that predicts the voltage caused by vibration. During testing, we subtract that predicted vibration voltage from the sensor reading to estimate the true diaphragm signal."
