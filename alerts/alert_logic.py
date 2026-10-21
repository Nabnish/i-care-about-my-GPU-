# alerts/alert_logic.py
import json
import joblib
import numpy as np
import pandas as pd

from preprocessing.windows import window
from preprocessing.features import build_features
from models.observability import confidence_score

N_GPUS = 8
DAYS = 14
CONF_CUT = 0.5  # fixed up front, not tuned on test

if __name__ == "__main__":
    df = window()                                  # load once, reuse everywhere
    X, y_risk, y_flip, split = build_features(df)
    model = joblib.load("models/saved/gb_model.joblib")

    va = (split == "val").values
    te = (split == "test").values

    pred_val = np.clip(model.predict(X[va]), 0, 1)
    pred_test = np.clip(model.predict(X[te]), 0, 1)

    # 1. risk threshold from val only (1 alert per GPU per day)
    budget = N_GPUS * DAYS * 1
    thr = np.quantile(pred_val, 1 - budget / len(pred_val))
    with open("models/saved/alert_threshold.json", "w") as f:
        json.dump({"threshold": float(thr), "conf_cut": CONF_CUT}, f)
    print("risk threshold:", round(float(thr), 4))

    # 2. confidence: compute on the full table (per-GPU rolling), then slice test
    conf = confidence_score(df)
    conf_test = conf[te].values
    y_test = y_flip[te].values

    # 3. four-way categories on test
    high_risk = pred_test >= thr
    low_conf = conf_test < CONF_CUT

    cat = np.select(
        [high_risk & ~low_conf, high_risk & low_conf, ~high_risk & low_conf],
        ["1 flip may have occurred",
         "2 possible flip, telemetry unreliable",
         "3 monitoring degraded, can't rule one out"],
        default="4 no alert",
    )

    # 4. counts and flip rate per category
    t = pd.DataFrame({"cat": cat, "flip": y_test})
    table = t.groupby("cat").agg(rows=("flip", "size"),
                                 flips=("flip", "sum"),
                                 flip_rate=("flip", "mean")).round(4)
    print(table)

    # 5. report the two message types separately
    risk_rows = high_risk.sum()
    notice_rows = (~high_risk & low_conf).sum()
    print("\nrisk alert rows:", risk_rows)
    print("monitoring-degraded notice rows (separate):", notice_rows)

    # 6. merge risk alerts into events (15-minute gap) per category
    ts = pd.to_datetime(df.loc[te, "timestamp"]).values
    run = df.loc[te, "run_id"].values
    e = pd.DataFrame({"run_id": run, "ts": ts, "cat": cat})
    for c in ["1 flip may have occurred", "2 possible flip, telemetry unreliable"]:
        a = e[e["cat"] == c].sort_values(["run_id", "ts"])
        gap = a.groupby("run_id")["ts"].diff()
        new_ev = gap.isna() | (gap > pd.Timedelta("15min"))
        print(f"{c}: events = {int(new_ev.sum())}, "
              f"per GPU/day = {new_ev.sum() / N_GPUS / DAYS:.2f}")