import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import score
from preprocessing.features import build_features
from preprocessing.windows import window


def get_models():
    models={
       "Ridge": Pipeline([
           ('imputer', SimpleImputer(strategy="median")),
           ('Scale' , StandardScaler()),
           ('model', Ridge()) 
       ]),
       "Random Forest": Pipeline([
            ('imputer',SimpleImputer(strategy="median")),
            ('model',RandomForestRegressor(n_estimators=200,min_samples_leaf=50,n_jobs=-1,random_state=0))
       ]),
       "Gradient Boosting": Pipeline([
           ("model",HistGradientBoostingRegressor(max_iter=200,learning_rate=0.05))
       ])
    }

    return models



if __name__  == "__main__":
    df=window()
    X, y_risk, y_flip, split = build_features(df)
    keep = df.groupby("run_id").cumcount() % 3 == 0
    tr = (split == "train").values & keep
    rows=[]
    for y,x in get_models().items():
        x.fit(X[tr],y_risk[tr])
        for i in ["val","test"]:
            m=(split==i).values
            pred = np.clip(x.predict(X[m]), 0, 1)
            rows.append({"model": y, "split": i, **score(y_flip[m], y_risk[m], pred)})

    print(pd.DataFrame(rows).round(4).to_string(index=False))