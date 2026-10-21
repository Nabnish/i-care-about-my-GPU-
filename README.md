# GPU Bit-Flip Risk Model

Predicts the probability that a **bit flip** happens on a GPU in the **next 15 minutes**, and reports **how much the telemetry can be trusted** at that moment. The two are reported separately, because *low risk* and *low confidence* are different things.

> **Important:** this project is trained and tested on a **synthetic dataset**. Results mean "this works if flips behave the way the simulator says". See [Limitations](#10-limitations).

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [Key ideas in plain language](#2-key-ideas-in-plain-language)
3. [Project structure](#3-project-structure)
4. [Setup](#4-setup)
5. [Data](#5-data)
6. [How to run (step by step)](#6-how-to-run-step-by-step)
7. [How the pipeline works](#7-how-the-pipeline-works)
8. [Results summary](#8-results-summary)
9. [Reading the outputs](#9-reading-the-outputs)
10. [Limitations](#10-limitations)
11. [Troubleshooting](#11-troubleshooting)
12. [Not done yet](#12-not-done-yet)
13. [Glossary](#13-glossary)

---

## 1. What this project does

GPUs can suffer **bit flips**: a stored bit changes from 0 to 1 or back, usually from a hardware fault or radiation. Some are visible (the GPU's ECC memory logs a correction) and some are **silent** (no trace is left). A silent flip can corrupt a training run without anyone knowing.

This project builds a model that, for every GPU every minute, outputs:

| Output | Meaning |
|---|---|
| `risk` | Probability of a flip in the next 15 minutes |
| `confidence` | How much the current telemetry can be trusted (0 to 1) |
| `alert` | A message for an operator, combining risk and confidence |

---

## 2. Key ideas in plain language

- **Risk vs confidence.** A GPU whose sensors stopped reporting can *look* safe to the model. Confidence tells you when "low risk" cannot be trusted.
- **Training target vs evaluation target.** The model is trained to predict the true hidden probability (`bit_flip_risk`), and is graded against what actually happened (`flip_in_next_window`, 0 or 1).
- **Splits by GPU.** Train, validation and test contain *different GPUs*. A GPU never appears in two splits, so the test measures how the model does on GPUs it has never seen.
- **Hidden columns.** Columns starting with `hidden_` hold the simulator's secrets (the true fault state, true hazard, and so on). They are never given to the model. They are only used to check that the confidence score works.
- **ECC history dominates.** In this synthetic world, the recent count of single-bit ECC errors (`sbe_24h`) carries almost all of the signal. A one-line rule using it ties with every trained model. That is a finding, not a bug.

---

## 3. Project structure

```text
i-care-about-my-GPU-/
├── data/
│   ├── cleaned/                 # synthetic_gpu_dataset.csv.gz (training input) + cleaned real datasets
│   └── generated/               # regenerated dataset variants
├── preprocessing/
│   ├── windows.py               # sorts per GPU, adds rolling columns, drops raw ECC totals
│   ├── features.py              # FEATURES / NEVER_USE lists, guards, build_features()
│   └── cleaning.py
├── models/
│   ├── risk_model.py            # Ridge, Random Forest, Gradient Boosting
│   ├── calibration.py           # trains the main model, fits isotonic calibrator, saves both
│   ├── observability.py         # confidence score
│   └── saved/                   # saved model, calibrator, alert threshold
├── evaluation/
│   ├── metrics.py               # score(): ROC-AUC, PR-AUC, MAE, Brier
│   └── results/                 # plots (reliability diagram)
├── alerts/
│   └── alert_logic.py           # risk threshold + four-way alert table
├── simulator/                   # synthetic data generator
├── config/                      # simulation / fault / experiment settings
├── notebooks/                   # exploration only
├── ARCHITECTURE.md              # system design
├── Training_architecture.md     # training plan
└── RESULTS.md                   # results of this run
```

**Rule of thumb:** `preprocessing/` makes features, `models/` fits things, `evaluation/` only measures. Evaluation code never fits anything.

---

## 4. Setup

**Requirements:** Python 3.10 or newer.

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install pandas numpy scikit-learn joblib matplotlib
```

(or `pip install -r requirements.txt` if it lists these).

Create the output folders if they don't exist:

```bash
mkdir models/saved
mkdir evaluation/results
```

---

## 5. Data

### Training data
`data/cleaned/synthetic_gpu_dataset.csv.gz`

| Property | Value |
|---|---|
| Rows | 966,960 |
| GPUs | 48 (each 20,145 one-minute rows, 14 days) |
| Raw columns | 47 |
| After preprocessing | 966,960 x 51 |
| Splits | train 644,640 rows (32 GPUs), val 161,160 (8 GPUs), test 161,160 (8 GPUs) |
| Base flip rate | 1.97% |

### Real datasets (not used for training)
The real workload and Eagle datasets have **no flip labels**, so they cannot train the model. They shaped the simulator (outage lengths, workload patterns). "No ECC event" does not mean "no flip", because silent flips leave no trace, so labelling those rows as negatives would teach the model something false. They are intended only for a later false-alarm check.

---

## 6. How to run (step by step)

**Always run from the project root** using `python -m ...`. Do not use an editor's Run button, because it changes the working directory and breaks imports like `preprocessing.windows`.

### Step 1: check the features
```bash
python -m preprocessing.features
```
Expected: shape `(966960, 31)`, split counts 644640 / 161160 / 161160, and a flip rate near `0.0197`.

### Step 2: baselines and the metrics ruler
```bash
python -m evaluation.metrics
```
Scores two baselines that need no training:
- **Constant:** every row gets the train-set mean risk.
- **ECC rule:** the score is just `sbe_24h`.

Expected: ECC rule at about 0.843 ROC-AUC / 0.125 PR-AUC on val and 0.899 / 0.448 on test. If these don't match, there is a bug in the features or splits. Fix it before going further.

### Step 3: train the models
```bash
python -m models.risk_model
```
Trains Ridge, Random Forest and Gradient Boosting on the train split (every 3rd minute) and prints val and test scores. Random Forest takes a few minutes.

### Step 4: calibration
```bash
python -m models.calibration
```
Trains the main model (Gradient Boosting), fits an isotonic calibrator on val, and saves:
- `models/saved/gb_model.joblib`
- `models/saved/isotonic_calibrator.joblib`
- `evaluation/results/reliability_test.png`

**This step must run before step 5**, because the alert step loads the saved model.

### Step 5: confidence score check
```bash
python -m models.observability
```
Prints the confidence range and checks it against the hidden monitor-fault flag.

### Step 6: alerts and the four-way table
```bash
python -m alerts.alert_logic
```
Picks the risk threshold on val, applies it to test, and prints the four-way alert table plus alert events per GPU per day. Saves `models/saved/alert_threshold.json`.

> There is no single `train.py` yet. Run the steps above in order.

---

## 7. How the pipeline works

### 7.1 Features (31 columns)

| Group | Columns |
|---|---|
| Current sensors (7) | `temp_core_c`, `temp_mem_c`, `power_pct_tdp`, `util_gpu`, `util_mem`, `voltage_v`, `throttle_flag` |
| Rolling, past-only (6) | `temp_core_c`, `voltage_v`, `power_w` at 15 and 60 minutes |
| ECC history (1) | `sbe_24h` |
| Static (1) | `gpu_model` (H100 = 1, B200 = 0) |
| Monitoring quality (16) | 8 `*_missing` flags and 8 `*_age` columns |

Raw `power_w` is deliberately left out (only its rolling averages are used).

**Never used as features:** every `hidden_*` column, the targets `bit_flip_risk` and `flip_in_next_window`, and the identifiers `run_id`, `node_id`, `gpu_id`, `source`, `split`, `timestamp`. `features.py` has guards that stop the program if any of these slip in.

**Rolling windows** are computed per GPU, looking backwards only, with a minimum of one observation. They start filling from a GPU's first minute rather than being empty for the first hour.

### 7.2 Missing values
Sensors go missing during simulated outages. This is expected and meaningful.
- **Gradient Boosting** handles NaNs natively.
- **Ridge and Random Forest** use the **train median** to fill them. The `*_missing` flags stay as features, so the model knows a value was filled.

### 7.3 Models

| Model | Purpose |
|---|---|
| Constant (base rate) | Floor |
| ECC rule (`sbe_24h`) | The bar to beat |
| Ridge | Simple linear baseline |
| Random Forest (200 trees, min leaf 50) | Non-linear baseline |
| Gradient Boosting (200 iterations, learning rate 0.05) | Main model |

All are trained on `bit_flip_risk` and scored against `flip_in_next_window`. Predictions are clipped to [0, 1].

### 7.4 Calibration
Isotonic regression is fit on **validation** predictions against the 0/1 outcome, then applied to test. Predictions are grouped into bins by quantiles (not equal widths), because positives are rare.

### 7.5 Confidence score
Built from monitoring columns only, never from the model's risk:

```
confidence = 0.4 * completeness + 0.3 * freshness + 0.3 * coverage
```

| Part | Definition |
|---|---|
| Completeness | Share of the last 60 minutes where core temperature was present |
| Freshness | `1 - (worst *_age, capped at 60) / 60` |
| Coverage | Share of sensors present right now |

The weights are placeholders and were not tuned.

### 7.6 Alert logic

The **risk threshold** is chosen on validation only, so that there is about 1 alert per GPU per day (8 GPUs x 14 days = 112 alerts). The same threshold is then applied to test. The **confidence cut** is fixed at 0.5 in advance.

| Risk | Confidence | Message |
|---|---|---|
| high | high | "A bit flip may have occurred. Consider validating or rerunning this step." |
| high | low | "Possible flip, but telemetry is unreliable. Check GPU health." |
| low | low | "No flip detected, but monitoring is degraded. Can't rule one out." |
| low | high | no alert |

The low-risk, low-confidence message is a **monitoring notice**, and is reported separately from risk alerts.

**Events vs rows.** Alerts fire every minute during a risky episode. Alerts within 15 minutes of each other are merged into one **event**, which is what an operator actually experiences.

---

## 8. Results summary

Full tables are in `RESULTS.md`.

**Model comparison (test, ROC-AUC / PR-AUC)**

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| ECC rule | 0.899 | 0.448 |
| Ridge | 0.903 | 0.456 |
| Random Forest | 0.905 | 0.450 |
| Gradient Boosting | 0.902 | 0.438 |

**Main findings**
- All models tie with each other and with the one-line ECC rule. ECC history carries nearly all the signal.
- The raw regressor is already well calibrated (test ECE 0.005). Isotonic calibration fit on val made test slightly worse, because val and test have different base rates (1.7% vs 3.9%).
- Threshold 0.4936 gave 1.0 alert rows per GPU per day on val. On test, **all** 2,247 alert rows fell on the single weak GPU and none on the 7 healthy GPUs. They merge into **44 events (0.39 per GPU per day)**.
- Flip rate in alert rows is 0.71 (high confidence) and 0.56 (low confidence), against 0.03 in the no-alert row.
- Confidence is 0.135 when the hidden monitor fault is on and 0.996 when it is off, so the score detects degraded monitoring.
- Confidence flags **data quality, not extra flip probability**: the low-risk low-confidence group flips at 0.033 vs 0.030 for the no-alert group.

---

## 9. Reading the outputs

| Metric | What it tells you | Good direction |
|---|---|---|
| ROC-AUC | Do real flips get higher scores than non-flips? 0.5 is random, 1.0 is perfect | Higher |
| PR-AUC | How many top-scored rows are real flips, across thresholds. Baseline is the flip rate (about 0.02), not 0.5 | Higher |
| MAE | Average gap between predicted risk and the true `bit_flip_risk` | Lower |
| Brier | Average squared gap between predicted risk and the 0/1 outcome | Lower |
| ECE | Average gap between predicted risk and observed flip rate, by bin | Lower |
| Precision | Of the alerts, how many were followed by a real flip | Higher |
| Recall | Of all real flip rows, how many got an alert | Higher |

PR-AUC is the more honest ranking metric here, because flips are rare and ROC-AUC can look good even when most alerts are false alarms.

**Red flags**
- ROC-AUC far above about 0.92: probable leakage. Check that no `hidden_*` column or target is in the features.
- ROC-AUC near 0.5 for a trained model: it isn't learning, usually wrong rows or wrong target.
- A calibrated test prediction with only one unique value: isotonic collapsed because val has too few positives.

---

## 10. Limitations

- **Simulated data.** The link from conditions to flips is written by us. Results mean "works if flips behave like this".
- **ECC history dominates by design.** This agrees with real studies that report only modest temperature effects, but it means trained models do not beat the simple rule here.
- **One weak GPU per split.** Val and test each hold about one weak GPU, so scores swing between splits (test flip rate 3.9% vs val 1.7%). Do not choose settings on test.
- **Alert budget does not transfer at row level.** 1.0 per GPU per day on val became 20 on test as rows, and only held (0.39) after merging into events.
- **Low row-level recall.** Recall is 0.236 on test at this strict budget.
- **Confidence does not predict flips or model error** in this simulator, only data quality.
- **Placeholder severity table** until real fault-injection runs.
- **Voltage** comes from one consumer GPU (RTX 3060) and is applied to all GPUs.

---

## 11. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'windows'` | Old import. Use `from preprocessing.windows import window` and run from the project root |
| `ModuleNotFoundError: No module named 'preprocessing'` | Running from the wrong folder or with an editor's Run button. Use `python -m ...` from the project root, and check `preprocessing/__init__.py` exists |
| `ValueError: Unmapped gpu_model labels` | A GPU model other than H100 or B200 is in the data. Extend `GPU_MODEL_MAP` or switch to one-hot columns |
| `could not convert string to float: 'none'` | `hidden_monitor_fault` holds text labels. Use `!= "none"` to make a flag instead of converting to float |
| `FileNotFoundError: models/saved/gb_model.joblib` | Run `python -m models.calibration` first, and make sure `models/saved/` exists |
| Ridge predicts negative values | Expected. Predictions are clipped to [0, 1] |
| Random Forest is slow | Normal. Training on every 3rd minute keeps it to a few minutes |

---

## 12. Not done yet

- ECC-history-removed ablation (an earlier probe reported PR-AUC about 0.024, near chance, but it was not reproduced in this run)
- Confidence check by group (confidence below 1 vs equal to 1) on error, ROC-AUC and PR-AUC
- Ensemble and uncertainty (5 seeds)
- Grouped 4-fold cross-validation
- Availability sweep (100%, 75%, 50%, 25% telemetry)
- Multiple simulator seeds
- Real-data false-alarm checks on Eagle and the workload files
- A single `train.py` that runs the whole pipeline

---

## 13. Glossary

| Term | Meaning |
|---|---|
| Bit flip | A stored bit changes value (0 to 1 or 1 to 0) |
| Silent flip | A flip that leaves no ECC trace |
| ECC | Error-correcting code memory, which logs and fixes some errors |
| SBE / DBE | Single-bit / double-bit ECC errors |
| `sbe_24h` | Count of single-bit ECC errors over the past 24 hours |
| Weak GPU | A GPU the simulator makes more flip-prone |
| Observability | How well the monitoring can see the GPU's real state |
| Calibration | Making predicted probabilities match observed frequencies |
| Isotonic regression | A monotone curve used to calibrate predictions |
| Leakage | Information from the answer or the future sneaking into the features |
| Event | Alert rows merged when they are within 15 minutes of each other |