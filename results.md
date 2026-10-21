# Results: GPU Bit-Flip Risk Model

Trained on the synthetic dataset (966,960 rows, 48 GPUs, 14 days, 1-minute steps). Split by GPU: 32 train / 8 val / 8 test. Models were trained on every 3rd minute of the train split.

---

## 1. Setup

| Item | Value |
|---|---|
| Features | 31 (current sensors, 15/60 min rolling, `sbe_24h`, `gpu_model`, 8 `*_missing`, 8 `*_age`) |
| Excluded | all `hidden_*` columns, both targets, identifiers (guarded in code) |
| Training target | `bit_flip_risk` |
| Evaluation target | `flip_in_next_window` |
| Base flip rate | 1.97% overall (val 1.7%, test 3.9%) |

Val and test each hold about one weak GPU, so single-split scores are noisy.

---

## 2. Model comparison

| Model | Split | ROC-AUC | PR-AUC | MAE | Brier |
|---|---|---|---|---|---|
| Constant (train mean) | val | 0.5000 | 0.0171 | 0.0185 | 0.0168 |
| Constant (train mean) | test | 0.5000 | 0.0393 | 0.0413 | 0.0383 |
| ECC rule (`sbe_24h`) | val | 0.8431 | 0.1248 | n/a | n/a |
| ECC rule (`sbe_24h`) | test | 0.8987 | 0.4477 | n/a | n/a |
| Ridge | val | 0.8460 | 0.1334 | 0.0099 | 0.0159 |
| Ridge | test | 0.9032 | 0.4562 | 0.0193 | 0.0272 |
| Random Forest | val | 0.8246 | 0.1037 | 0.0097 | 0.0160 |
| Random Forest | test | 0.9045 | 0.4495 | 0.0202 | 0.0278 |
| Gradient Boosting | val | 0.8400 | 0.1084 | 0.0093 | 0.0161 |
| Gradient Boosting | test | 0.9023 | 0.4383 | 0.0205 | 0.0281 |

**Findings**
- All models tie with each other and with the one-line ECC rule (test ROC-AUC about 0.90).
- ECC history carries nearly all the signal. Ridge matches gradient boosting, so extra complexity buys nothing.
- MAE is roughly half the constant's on val (about 0.0095 vs 0.0185), so the models track true risk better than a flat guess.
- Val and test differ because each holds about one weak GPU. Settings should be chosen on val or cross-validation, not test.

---

## 3. Calibration (isotonic, fit on val)

| Split | Brier raw | Brier calibrated | ECE raw | ECE calibrated |
|---|---|---|---|---|
| Val | 0.0160 | 0.0157 | 0.0035 | 0.0000 |
| Test | 0.0281 | 0.0291 | 0.0052 | 0.0091 |

- Val ECE of 0.0000 is meaningless because the calibrator was fit on val.
- On test, calibration made results slightly worse. Val and test have different base rates (1.7% vs 3.9%), so a calibrator fit on val pulls test predictions too low.
- The raw regressor was already close to calibrated (test ECE 0.0052), since it was trained on the true risk.
- Decision: raw predictions are used for alerts. The calibrator is saved but not relied on.

---

## 4. Risk alerts

Threshold chosen on val only, for a budget of 1 alert per GPU per day (112 alerts over 8 GPUs and 14 days): **0.4936**.

| Split | Alert rows | Per GPU/day (rows) | Precision | Recall |
|---|---|---|---|---|
| Val | 112 | 1.00 | 0.330 | 0.013 |
| Test | 2,247 | 20.06 | 0.665 | 0.236 |

- Counted as rows, test looks 20x over budget. This is misleading.
- **All 2,247 alert rows are on `sim_gpu11`**, the test weak GPU. The other 7 test GPUs have zero alerts.
- Merging alerts within 15 minutes gives **44 events**, which is **0.39 events per GPU per day**, under the budget.
- On the weak GPU alone that is about 3 events per day.
- Recall is low because the budget is strict. Many flip-risk minutes on the weak GPU fall below the threshold.

---

## 5. Observability confidence

Built from monitoring columns only: 0.4 x completeness (share of last 60 min with core temp present) + 0.3 x freshness (worst `*_age`, capped at 60) + 0.3 x coverage (share of sensors present). Weights are placeholders and were not tuned.

| Check | Result |
|---|---|
| Range | 0.0 to 1.0, no NaNs |
| Mean confidence, hidden monitor fault ON | 0.135 |
| Mean confidence, hidden monitor fault OFF | 0.996 |

- The score clearly detects degraded monitoring. `hidden_monitor_fault` was used only to validate it, never as a feature.
- The 8 `*_age` columns have identical statistics (mean 87, max 2,695), so sensors drop out together. Freshness and coverage therefore overlap.
- The quartile-error check was inconclusive: most rows have confidence exactly 1.0, so quartiles 2 to 4 are arbitrary splits of identical rows. The fair comparison (confidence below 1 vs equal to 1) was not run.

---

## 6. Four-way alert table (test)

Risk cut: 0.4936 (from val). Confidence cut: 0.5 (fixed in advance, not tuned).

| Category | Rows | Flips | Flip rate |
|---|---|---|---|
| 1. High risk, high confidence: "flip may have occurred" | 1,570 | 1,116 | 0.7108 |
| 2. High risk, low confidence: "possible flip, telemetry unreliable" | 677 | 378 | 0.5583 |
| 3. Low risk, low confidence: "monitoring degraded, can't rule one out" | 20,796 | 688 | 0.0331 |
| 4. Low risk, high confidence: no alert | 138,117 | 4,151 | 0.0301 |

| Message type | Rows | Events (15 min merge) | Events per GPU/day |
|---|---|---|---|
| Risk alert, category 1 | 1,570 | 44 | 0.39 |
| Risk alert, category 2 | 677 | 1 | 0.01 |
| Monitoring notice, category 3 (reported separately) | 20,796 | not merged | not computed |

**Findings**
- Categories 1 and 2 are meaningful: flip rates of 0.71 and 0.56 vs 0.03 in the no-alert row.
- About 13% of test rows have degraded monitoring. During high risk this changes the message. During low risk it produces a separate notice.
- Categories 3 and 4 have essentially the same flip rate (0.0331 vs 0.0301). **Confidence flags data quality, not extra flip probability.** ECC history stays available during outages (`sbe_24h` has no NaNs), so the risk number holds up.
- The no-alert row still contains 4,151 flips (3.0%), which matches the low row-level recall.

---

## 7. Limits

- Data is simulated and the link from conditions to flips is written by us. Results mean "works if flips behave like this".
- ECC history dominating is a design outcome of the simulator, in line with real studies reporting modest temperature effects.
- Val and test each hold about one weak GPU, so scores swing between splits.
- The alert budget did not transfer across splits at row level (1.0 on val, 20 on test) and held only after merging rows into events.
- Row-level recall is low (0.236 on test) at this strict budget.
- Confidence does not predict model error or flip probability in this simulator, only data quality.
- The severity table is a placeholder until real fault-injection runs.
- Voltage comes from one consumer GPU (RTX 3060) and is applied to all GPUs.

---

## 8. Not done (future work)

- ECC-history-removed ablation. The earlier probe reported PR-AUC about 0.024 (near chance); this was **not reproduced in this run**, so cite it as a probe result.
- Two-group confidence check (confidence below 1 vs equal to 1) on error, ROC-AUC and PR-AUC.
- Ensemble and uncertainty (5 seeds).
- Grouped 4-fold cross-validation.
- Availability sweep (100%, 75%, 50%, 25% telemetry).
- Multiple simulator seeds.
- Real-data false-alarm checks on Eagle and the workload files.