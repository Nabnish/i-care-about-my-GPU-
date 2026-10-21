# alerts/alert_logic.py
import json
import joblib
import numpy as np

from preprocessing.features import build_features

N_GPUS = 8
DAYS = 14

if __name__ == "__main__":
    X, y_risk, y_flip, split = build_features()
    model = joblib.load("models/saved/gb_model.joblib")

    va = (split == "val").values
    te = (split == "test").values

    pred_val = np.clip(model.predict(X[va]), 0, 1)
    pred_test = np.clip(model.predict(X[te]), 0, 1)

    
    budget = N_GPUS * DAYS * 1
    thr = np.quantile(pred_val, 1 - budget / len(pred_val))

   
    for name, pred, y in [("val", pred_val, y_flip[va].values),
                          ("test", pred_test, y_flip[te].values)]:
        alert = pred >= thr
        tp = (alert & (y == 1)).sum()
        print(name,
              "| alerts:", alert.sum(),
              "| per GPU/day:", round(alert.sum() / N_GPUS / DAYS, 2),
              "| precision:", round(tp / max(alert.sum(), 1), 3),
              "| recall:", round(tp / y.sum(), 3))

    with open("models/saved/alert_threshold.json", "w") as f:
        json.dump({"threshold": float(thr)}, f)
    print("threshold:", thr)
    import pandas as pd
from preprocessing.windows import window

df = window()
t = df[te][["run_id", "timestamp"]].copy()
t["alert"] = pred_test >= thr
t["timestamp"] = pd.to_datetime(t["timestamp"])

# check 1: alert rows per GPU (top 10)
per_gpu = t.groupby("run_id")["alert"].sum().sort_values(ascending=False)
print(per_gpu.head(10))

# check 2: merge alerts within 15 minutes into one event
a = t[t["alert"]].sort_values(["run_id", "timestamp"])
gap = a.groupby("run_id")["timestamp"].diff()
new_event = gap.isna() | (gap > pd.Timedelta("15min"))
events = new_event.groupby(a["run_id"]).sum()
print(events[events > 0])
print("events per GPU/day:", round(events.sum() / 8 / 14, 2))