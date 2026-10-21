# models/observability.py
import joblib
import numpy as np
import pandas as pd

from preprocessing.windows import window
from preprocessing.features import build_features, MISSING, AGE


def confidence_score(df, w=(0.4, 0.3, 0.3)):
    """0-to-1 trust in current telemetry. Uses monitoring columns only."""
    # completeness: share of the last 60 minutes where core temp was present
    miss = df["temp_core_c_missing"].astype(float)
    recent_missing = miss.groupby(df["run_id"]).transform(
        lambda s: s.rolling(60, min_periods=1).mean())
    completeness = 1 - recent_missing

    # freshness: worst staleness right now, capped at 60 (assumes age is in minutes)
    worst_age = df[AGE].max(axis=1).clip(upper=60)
    freshness = 1 - worst_age / 60

    # coverage: share of sensors present right now
    coverage = 1 - df[MISSING].astype(float).mean(axis=1)

    return w[0] * completeness + w[1] * freshness + w[2] * coverage


if __name__ == "__main__":
    df = window()
    X, y_risk, y_flip, split = build_features(df)

    print(df[AGE].describe().loc[["min", "mean", "max"]].round(1))  # check units

    conf = confidence_score(df)
    print("confidence range:", conf.min().round(3), conf.max().round(3),
          "| NaNs:", conf.isna().sum())

    # check on val using the saved model
    model = joblib.load("models/saved/gb_model.joblib")
    va = (split == "val").values
    pred = np.clip(model.predict(X[va]), 0, 1)

    d = pd.DataFrame({
        "conf": conf[va].values,
        "err": np.abs(pred - y_risk[va].values),
        "fault": (df.loc[va, "hidden_monitor_fault"] != "none").astype(float).values,
    })
    # rank first so ties at 1.0 don't collapse the bins
    d["q"] = pd.qcut(d["conf"].rank(method="first"), 4,
                     labels=["Q1 low", "Q2", "Q3", "Q4 high"])

    print(d.groupby("q", observed=True).agg(mean_conf=("conf", "mean"),
                                            mean_err=("err", "mean"),
                                            fault_rate=("fault", "mean")).round(4))
    print("confidence when monitor fault on / off:",
          d.loc[d.fault > 0, "conf"].mean().round(3),
          d.loc[d.fault == 0, "conf"].mean().round(3))