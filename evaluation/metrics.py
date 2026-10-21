
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             mean_absolute_error, brier_score_loss)


def score(y_flip, y_risk, pred, is_probability=True):
   
    out = {
        "roc_auc": roc_auc_score(y_flip, pred),
        "pr_auc": average_precision_score(y_flip, pred),
        "mae": np.nan,
        "brier": np.nan,
    }
    if is_probability:
        out["mae"] = mean_absolute_error(y_risk, pred)
        out["brier"] = brier_score_loss(y_flip, pred)
    return out


if __name__ == "__main__":
    import os
    import sys


    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


    from preprocessing.features import build_features

    X, y_risk, y_flip, split = build_features()
    tr = (split == "train").values

    const = y_risk[tr].mean()  

    for name in ["val", "test"]:
        m = (split == name).values
        c = score(y_flip[m], y_risk[m], np.full(m.sum(), const))
        e = score(y_flip[m], y_risk[m], X.loc[m, "sbe_24h"], is_probability=False)
        print(f"{name} constant:", {k: round(v, 4) for k, v in c.items()})
        print(f"{name} ECC rule:", {k: round(v, 4) for k, v in e.items()})