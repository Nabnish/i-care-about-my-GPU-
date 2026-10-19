# Training Architecture: GPU Bit-Flip Risk Model

How to train, calibrate, and evaluate the risk model on the synthetic dataset. It builds on `ARCHITECTURE.md` (the system), `DATA_SPEC.md` (the columns), and `IMPLEMENTATION.md` (the data generation). Plain-language steps; no code.

Numbers marked *(from my probe)* come from a quick experiment on the generated dataset (4-fold cross-validation, whole GPUs held out, training on every 3rd minute). Your numbers should land in the same ballpark. A big gap means a bug or leakage.

---

## 1. What training produces

| Output | Meaning | Built by |
|---|---|---|
| `risk` | Probability a flip happens in the next 15 minutes | Risk model + calibration |
| `confidence` | How much the telemetry can be trusted right now | Observability score |
| `uncertainty` | How much the models disagree | Small ensemble |
| `alert` | "A bit may have flipped; training may be affected" | Threshold on risk, adjusted by confidence |

Low risk and low confidence are different things. The system must report both.

---

## 2. Where everything lives (your current folders)

```text
i-care-about-my-GPU-/
├── data/
│   ├── cleaned/            # synthetic_gpu_dataset.csv.gz (input to training)
│   └── generated/          # regenerated variants (e.g. different availability)
├── preprocessing/
│   ├── windows.py          # trailing windows per GPU: rolling means, ECC window counts
│   ├── features.py         # builds the final feature table; FEATURES / NEVER_USE lists
│   └── cleaning.py         # loading, sorting, basic checks
├── models/
│   ├── risk_model.py       # baselines + main regressor + small ensemble
│   ├── calibration.py      # fits the calibrator (isotonic) on validation GPUs
│   ├── observability.py    # confidence score from missing / stale data
│   └── saved/              # trained models (create this folder)
├── evaluation/
│   ├── metrics.py          # PR-AUC, Brier, ECE, alert and lead-time metrics
│   ├── calibration.py      # reliability diagrams
│   ├── experiments.py      # CV, ablations, availability sweep
│   └── results/            # metrics tables and plots (create this folder)
├── notebooks/              # exploration and plots only, not the real pipeline
└── train.py                # (create at project root) runs the whole pipeline in order
```

Rule of thumb: **`preprocessing/` makes features, `models/` fits things, `evaluation/` only measures.** Evaluation code never fits anything.

---

## 3. Input and preprocessing contract

**Input:** `synthetic_gpu_dataset.csv.gz`, 966,960 rows, 46 columns (48 GPUs, 1-minute steps, 14 days).

**After your preprocessing** (ECC totals dropped, 6 rolling columns added): **966,960 rows x 50 columns.**

### Feature list (the only columns the model may use)

| Group | Columns |
|---|---|
| Current sensors | `temp_core_c`, `temp_mem_c`, `power_pct_tdp`, `util_gpu`, `util_mem`, `voltage_v`, `throttle_flag` |
| Rolling (per GPU, past only) | `temp_core_c`, `voltage_v`, `power_w` at 15 and 60 minutes (6 columns) |
| ECC history | `sbe_24h` (required). Optional: 1 h and 6 h error counts, 24 h double-bit count |
| Static | `gpu_model` (encode as 0/1) |
| Monitoring quality | `*_missing` (8) and `*_age` (8). Used mainly by the confidence model |

### Never-use list (never goes into features)
- everything starting with `hidden_`
- the targets `bit_flip_risk` and `flip_in_next_window`
- identifiers: `run_id`, `node_id`, `gpu_id`, `source`, `split`, `timestamp`

### Notes on the optional ECC window counts
In my probe, adding 1 h, 6 h, and 24 h counts and the rolling averages did **not** beat `sbe_24h` plus a few current sensors (PR-AUC 0.214 vs 0.215, which is noise). `sbe_24h` is enough to start. To add the others, compute them from the cumulative ECC totals **before** dropping them, per GPU and past-only, and forward-fill the totals first.

### Missing values
Sensors are missing during simulated outages. That is expected and meaningful.
- **Gradient boosting** handles missing values natively. Leave them.
- **Linear models and random forest** need filling: use the **training** median, and keep the `*_missing` flags so the model knows it was filled. Fit the median on train only.

---

## 4. Splits and leakage rules

- Use the **`split` column** (train / val / test), assigned **by GPU**. A GPU never appears in two splits.
- Because val and test each hold only about 1 weak GPU, single-split scores are noisy (test positives are 3.9% vs val 1.7%). For decisions, use **4-fold cross-validation grouped by GPU**, stratified by whether a GPU is weak. Report the pooled out-of-fold result.
- Train on val GPUs only to choose settings; **touch the test split once, at the end.**
- Fit every imputer, scaler, calibrator, and threshold on training or validation data only.
- Features at time t use only data up to t. Never use resampling over the whole table.
- Training rows are 1-minute steps and highly correlated. Using every 3rd minute gave similar results ~3x faster *(from my probe)*, so use that while experimenting.

---

## 5. Targets

- **Main target (regression): `bit_flip_risk`.** The true probability of a flip over the next 15 minutes if conditions persist. Most rows are near zero; a few are large.
- **Evaluation target: `flip_in_next_window`** (0 or 1), the sampled outcome. The regressor's predicted risk is scored against it as a flip score.
- Keep both. The regressor is the product; the outcome is how it is graded against reality.

---

## 6. Models to train (in this order)

| # | Model | Purpose | Starting settings |
|---|---|---|---|
| 0 | Constant (base rate 1.97%) | Floor | none |
| 1 | **ECC rule**: score = `sbe_24h` | The bar to beat | none |
| 2 | Ridge regression (and logistic as a classifier check) | Simple linear baseline | alpha 1.0, standardize inputs |
| 3 | Random Forest | Non-linear baseline | 200 trees, min leaf 50 |
| 4 | **Gradient boosting (HistGradientBoosting)** | Main model | 150 to 300 rounds, learning rate 0.05 to 0.1 |
| 5 | XGBoost (optional) | Only if 4 is clearly short | similar settings |

Train each on `bit_flip_risk`, then score the predictions against `flip_in_next_window`. Also train a classifier version on the 0/1 outcome and compare, because it is a useful sanity check.

### Expected results *(from my probe, pooled 4-fold grouped CV, base rate 1.97%)*

| What | PR-AUC | ROC-AUC | Notes |
|---|---|---|---|
| `sbe_24h` rule alone | 0.30 | 0.87 | the bar |
| Regressor (all features), used as flip score | 0.30 | 0.88 | Brier 0.0159 |
| Classifier (all features) | 0.21 | 0.87 | Brier 0.0181 |
| True hidden risk (ceiling) | 0.40 | 0.91 | |
| Regressor, ECC history removed | 0.024 | 0.60 | about chance |
| Regressor vs the true risk | Spearman 0.735, mean error 0.0105 | | |
| Constant guess | | | Brier 0.0193 |

**Read these honestly.** In this synthetic world, ECC history carries almost all the signal, and temp, power, and voltage barely help (they were written as small effects). The regressor tracks the true risk well but does not beat the one-line ECC rule as a detector. Report that as a finding. **Do not change the data generator to make the model look better.**

---

## 7. Calibration

The model's raw output is not a trustworthy probability yet.
1. Predict on the **validation** GPUs.
2. Fit isotonic regression mapping predicted risk to the observed `flip_in_next_window`.
3. Apply it to test predictions, and compare before and after with a reliability diagram, Brier score, and ECE.

Because positives are rare, group predictions into bins by quantiles, not equal widths, or most bins will be empty.

---

## 8. Observability confidence

**Goal:** a number in [0, 1] saying how much the current telemetry can be trusted. Built from monitoring columns only, never from the model's risk.

**Starting rule** *(placeholder weights; tune them against section 12 tests)*:
- completeness: the share of the last 60 minutes where core temp was present
- freshness: how large the biggest `*_age` is right now, capped at 60 minutes
- coverage: the share of sensors present right now

Combine them (for example, a weighted average) into one score. Add `avail60` and the biggest age as columns in `preprocessing/features.py` so the rule has inputs.

**Checks that it works:**
- Confidence drops as telemetry availability goes 100% to 75% to 50% to 25%.
- Prediction error is larger in the low-confidence bins than in the high-confidence bins.
- A **high risk with low confidence** case is reported differently from high risk with high confidence.

A learned version (a small model that predicts the risk model's error from the missing and age columns) is a later upgrade.

---

## 9. Uncertainty

Train 5 copies of the main model with different random seeds and different sub-samples of the training GPUs. Use the **spread of their predictions** as uncertainty. This is separate from observability: one is "models disagree", the other is "data is shaky".

---

## 10. Alert logic

| Risk | Confidence | Message |
|---|---|---|
| high | high | "A bit flip may have occurred. Consider validating or rerunning this step." |
| high | low | "Possible flip, but telemetry is unreliable. Check GPU health." |
| low | low | "No flip detected, but monitoring is degraded. Can't rule one out." |
| low | high | no alert |

**Choosing the risk threshold:** do it on validation GPUs only. Pick the threshold that gives at most about **1 alert per GPU per day**, or hits a target recall at an acceptable precision. Alerts per GPU per day is the operational number to report.

---

## 11. Evaluation

| Area | Metrics |
|---|---|
| Ranking | PR-AUC (main), ROC-AUC |
| Probability quality | Brier score, ECE, reliability diagram |
| Fit to the truth | Spearman and mean error against the true `bit_flip_risk` |
| Alerts | precision, recall, alerts per GPU per day |
| **Warning lead time** | flip time minus the first alert time in the 60 minutes before it; flips with no alert are "missed" |
| By GPU class | weak, moderate, healthy |
| **Silent vs visible flips** | detection rate for each. Silent flips are the hard case and the heart of the project |
| Observability | everything above at 100, 75, 50, 25% telemetry availability |

---

## 12. Experiments and ablations

1. **Model comparison** (section 6), with 4-fold grouped CV.
2. **Feature-group ablations:** remove ECC history, remove missing/age, remove rolling features, remove current sensors. (My probe: ECC history removal drops PR-AUC to about chance; the others change it only slightly.)
3. **Availability sweep:** regenerate the dataset with the simulator's `availability` setting at 1.0, 0.75, 0.5, and 0.25. Train once on the default (0.90) data and test on each variant. Check the confidence score falls and false confidence drops with it.
4. **Hypothesis test** (from `ARCHITECTURE.md`): does adding the confidence score reduce false confidence when telemetry is missing, compared with risk alone?
5. **Seeds:** repeat with 3 different simulator seeds. With only 5 weak GPUs, one seed can mislead.

---

## 13. Artifacts to save

- `models/saved/`: the trained models, the calibrator, and the ensemble.
- `evaluation/results/`: a metrics table (CSV), reliability and PR plots, the availability-sweep table, and an out-of-fold predictions file (`run_id`, `timestamp`, `risk`, `confidence`, `uncertainty`, `flip_in_next_window`).
- A short log of which dataset file and which seed produced each result.

---

## 14. Build order

1. Finish preprocessing in `preprocessing/windows.py` and `features.py`; check shape (966,960 x 50) and that no `hidden_` column is in the features.
2. Baselines 0 and 1 (constant, ECC rule). Confirm their numbers match section 6.
3. Ridge, then random forest, then gradient boosting, in `models/risk_model.py`.
4. Calibration (`models/calibration.py`, `evaluation/calibration.py`).
5. Observability score (`models/observability.py`).
6. Ensemble and uncertainty.
7. Alert threshold and metrics (`evaluation/metrics.py`).
8. Ablations, availability sweep, seeds (`evaluation/experiments.py`).
9. Write up results, with the honest limits below.

---

## 15. Limits to state in the report
- Data is simulated, and the link from conditions to flips is written by us. Results mean "works if flips behave like this".
- The severity table is a placeholder until the laptop injection runs.
- ECC history dominating is a design outcome, in line with real studies reporting only modest temperature effects.
- Val and test hold few weak GPUs, so use cross-validation and several seeds.
- Voltage comes from one consumer GPU (RTX 3060) and is applied to all GPUs.

---

## 16. Open decisions
1. Alert threshold policy: alerts per GPU per day vs a target recall.
2. Whether to add the optional ECC window counts (my probe says they are not needed).
3. How strongly the confidence score should adjust the alert message.
4. How many simulator seeds to run.