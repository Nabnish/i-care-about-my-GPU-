# models/calibration.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.windows import window
from preprocessing.features import build_features


def ece_quantile(y, p, bins=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(p=("p", "mean"),
                                            y=("y", "mean"),
                                            n=("y", "size"))
    ece = (abs(g["p"] - g["y"]) * g["n"]).sum() / g["n"].sum()
    return ece, g


if __name__ == "__main__":
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("evaluation/results", exist_ok=True)

    df = window()
    X, y_risk, y_flip, split = build_features(df)

    keep = (df.groupby("run_id").cumcount() % 3 == 0).values
    tr = (split == "train").values & keep
    va = (split == "val").values
    te = (split == "test").values

    # 1. train main model on train only
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                          random_state=0)
    model.fit(X[tr], y_risk[tr])

    pred_val = np.clip(model.predict(X[va]), 0, 1)
    pred_test = np.clip(model.predict(X[te]), 0, 1)

    # 2. fit calibrator on val only, target = 0/1 outcome
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    cal.fit(pred_val, y_flip[va])

    cal_val = cal.transform(pred_val)
    cal_test = cal.transform(pred_test)

    # 3. save
    joblib.dump(model, "models/saved/gb_model.joblib")
    joblib.dump(cal, "models/saved/isotonic_calibrator.joblib")

    # 4. compare before / after
    rows = []
    for name, y, raw, cald in [("val", y_flip[va], pred_val, cal_val),
                               ("test", y_flip[te], pred_test, cal_test)]:
        e_raw, _ = ece_quantile(y, raw)
        e_cal, _ = ece_quantile(y, cald)
        rows.append({"split": name,
                     "brier_raw": brier_score_loss(y, raw),
                     "brier_cal": brier_score_loss(y, cald),
                     "ece_raw": e_raw,
                     "ece_cal": e_cal})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("unique calibrated values (test):", len(np.unique(cal_test)))

    # 5. reliability plot (test)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, g_raw = ece_quantile(y_flip[te], pred_test)
    _, g_cal = ece_quantile(y_flip[te], cal_test)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 0.3], [0, 0.3], "k--", label="perfect")
    plt.plot(g_raw["p"], g_raw["y"], "o-", label="raw")
    plt.plot(g_cal["p"], g_cal["y"], "s-", label="calibrated")
    plt.xlabel("mean predicted risk")
    plt.ylabel("observed flip rate")
    plt.legend()
    plt.title("Reliability (test)")
    plt.savefig("evaluation/results/reliability_test.png", dpi=120)