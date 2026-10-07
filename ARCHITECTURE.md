# GPU Bit-Flip Risk Monitor

A lightweight model that watches GPU telemetry during AI training and warns the user when a bit flip may have occurred and could be corrupting the run.

---

## 1. Problem

Silent data corruption (SDC) on GPUs is dangerous because nothing crashes. A flipped bit in a weight, activation, or gradient can quietly skew a training run, and by the time the loss looks wrong, hours of compute are wasted.

Bit flips can't be observed directly, but the conditions that make them likely can: heat, voltage instability, power spikes, rising ECC counts, clock throttling.

**Core question:**

> Given noisy, possibly incomplete GPU telemetry, how likely is it that a bit flip occurred in the last window, and how much should we trust that estimate?

**What the system outputs:**

```text
bit_flip_risk   ∈ [0,1]   expected likelihood of a flip in the current window
confidence      ∈ [0,1]   how trustworthy the telemetry (and therefore the estimate) is
alert           yes/no    "A bit flip may have occurred. Training results may be affected."
```

Low risk and low confidence are different things. A risk of 0.05 with confidence 0.95 means the GPU is probably fine. A risk of 0.05 with confidence 0.30 means we're mostly blind. The system must report both.

---

## 2. Architecture

```text
 ┌─────────────────────┐
 │   GPU Simulator     │  normal behavior, injected bit flips,
 │                     │  hidden corruption state, telemetry dropouts
 └──────────┬──────────┘
            ▼
 ┌─────────────────────┐
 │     Telemetry       │  temperature, voltage, power, clocks,
 │                     │  utilization, ECC counts
 └──────────┬──────────┘
            ▼
 ┌─────────────────────┐
 │   Preprocessing     │  cleaning, missing-value handling,
 │                     │  rolling windows, feature extraction
 └──────────┬──────────┘
            ▼
   ┌────────┴─────────┐
   ▼                  ▼
┌──────────────┐  ┌──────────────┐
│ Bit-Flip     │  │ Observability│
│ Risk         │  │ Confidence   │
│ Regressor    │  │              │
└──────┬───────┘  └──────┬───────┘
       └────────┬────────┘
                ▼
 ┌─────────────────────┐
 │ Calibration +       │  calibrated risk, uncertainty,
 │ Alert Logic         │  threshold → user warning
 └──────────┬──────────┘
            ▼
 ┌─────────────────────┐
 │    Evaluation       │  compared against hidden ground truth
 └─────────────────────┘
```

---

## 3. GPU Simulator

No physical GPU is needed. The simulator generates telemetry and a hidden ground truth the model never sees.

```text
Simulated GPU
├── Observable: temperature, voltage, power, clock, utilization, ECC count
├── Hidden:     fault stress level, corruption_state, actual bit-flip events
└── Monitoring: sensor availability, data freshness, missing samples
```

**Fault modes that raise flip probability:**

| Mode | Telemetry signature |
|---|---|
| Thermal stress | Temperature climbs faster than normal, sustained high plateau |
| Voltage / power instability | Spikes, oscillation, unexpected drops |
| ECC degradation | Correctable error count rising over time |
| Silent flip | Flip occurs with **no** ECC event and no obvious anomaly |

The silent case is deliberate: it forces the model to learn indirect signals, not just ECC counts.

**Monitoring failures** are also simulated (missing sensors, stale or delayed readings, agent dropout). These drive the confidence score.

---

## 4. Features

The model uses temporal behavior, not just raw readings.

```text
Raw:       temperature, voltage, power, clock_speed, utilization, ecc_errors
Temporal:  temp_mean / temp_std / temp_slope (rolling window)
           voltage_variance, power_variance
           ecc_error_rate, clock_throttle_rate
History:   gpu_age, past ECC events, past thermal events
```

---

## 5. Models

### Bit-flip risk regressor
Predicts flip risk for the next window from the features above. It's meant to be small enough to run alongside training without noticeable overhead.

Start with a simple baseline, then compare against stronger models:

```text
Linear / Ridge regression  →  Random Forest  →  Gradient Boosting (XGBoost)
```

### Calibration
Raw outputs are not probabilities. Calibrate so that predicted risk matches observed flip frequency (Platt scaling or isotonic regression).

### Observability confidence
A score in [0,1] built from:

```text
data completeness · data freshness · sensor availability · sensor consistency
```

Example: 98% complete, fresh, reliable sensors → confidence ≈ 0.96. 61% complete, 42% fresh → confidence ≈ 0.48.

Missing data must reduce confidence. The system should not impute values and then act as if the data were trustworthy.

### Uncertainty
Start with calibrated probabilities plus ensemble disagreement. Move to conformal prediction or bootstrap intervals only if needed.

---

## 6. Alerting

```text
risk high, confidence high  →  "A bit flip may have occurred. Consider validating or rerunning this step."
risk high, confidence low   →  "Possible flip, but telemetry is unreliable. Check GPU health."
risk low,  confidence low   →  "No flip detected, but monitoring is degraded. Can't rule one out."
risk low,  confidence high  →  no alert
```

Alert output: `gpu_id, timestamp, risk, confidence, severity`.

---

## 7. Evaluation

Everything is scored against the simulator's hidden ground truth.

| Area | Metrics |
|---|---|
| Prediction quality | MAE / RMSE on risk; precision, recall, F1, PR-AUC on the flip-or-not decision (flips are rare, so PR-AUC matters most) |
| Calibration | Brier score, ECE, reliability diagram |
| Detection | Flips caught vs. missed, false alarms, detection delay / warning lead time |
| Silent flips | Detection rate on flips with no ECC event |
| Observability | Does confidence drop as telemetry quality drops, and does it reduce overconfident wrong predictions? |

**Experiments vary:**

```text
Fault frequency:    low / medium / high
Telemetry quality:  100% / 75% / 50% / 25% available
Fleet size:         8 / 32 / 128 GPUs
```

---

## 8. Hypothesis

> A model that combines flip-risk estimation with explicit observability confidence will give more trustworthy warnings than one using raw telemetry alone, and will reduce false confidence when monitoring data is missing or stale.

---

## 9. Project Structure

```text
gpu-bitflip-monitor/
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── config/            # simulation, fault, and experiment settings (yaml)
├── simulator/         # gpu.py, faults.py, telemetry.py, monitoring.py
├── preprocessing/     # cleaning.py, windows.py, features.py
├── models/            # risk_model.py, observability.py, calibration.py
├── alerts/            # alert_logic.py
├── evaluation/        # metrics.py, calibration.py, experiments.py
├── data/
├── notebooks/
└── tests/
```

**Stack:** Python, NumPy, Pandas, scikit-learn, XGBoost, Matplotlib. Parquet or CSV for storage.

---

## 10. Build Order

1. **Simulator.** GPU objects, normal telemetry, time series.
2. **Fault injection.** Thermal, voltage/power, ECC, silent bit flips.
3. **Partial observability.** Missing, stale, and failed telemetry.
4. **Features.** Rolling windows, trends, rates.
5. **Baseline regressors.** Ridge, Random Forest, XGBoost; compare.
6. **Calibration.** Brier, ECE, reliability plots.
7. **Observability confidence.** Combine with risk and wire up alert logic.
8. **Evaluation.** Run the experiment grid and plot results.

---

## 11. Out of Scope (for now)

- GPU scheduling, job migration, quarantine decisions
- Real GPU fault injection or production deployment
- Dashboard / UI
- Loss and gradient anomaly detection, redundant computation, checksums (possible later extensions for confirming a suspected flip)