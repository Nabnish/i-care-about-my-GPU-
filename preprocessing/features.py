
import pandas as pd
from preprocessing.windows import window

CURRENT = ["temp_core_c", "temp_mem_c", "power_pct_tdp", "util_gpu",
           "util_mem", "voltage_v", "throttle_flag"]

ROLLING = ["temp_core_c_15mins_avg", "temp_core_c_60mins_avg",
           "voltage_v_15mins_avg", "voltage_v_60mins_avg",
           "power_w_15mins_avg", "power_w_60mins_avg"]

MISSING = ["temp_core_c_missing", "temp_mem_c_missing", "power_w_missing",
           "util_gpu_missing", "util_mem_missing", "voltage_v_missing",
           "ecc_sbe_total_missing", "ecc_dbe_total_missing"]

AGE = ["temp_core_c_age", "temp_mem_c_age", "power_w_age", "util_gpu_age",
       "util_mem_age", "voltage_v_age", "ecc_sbe_total_age", "ecc_dbe_total_age"]

FEATURES = CURRENT + ROLLING + ["sbe_24h", "gpu_model"] + MISSING + AGE  
NEVER_USE = [
    "hidden_monitor_fault", "hidden_weak_class", "hidden_weak_factor",
    "hidden_fault_mode", "hidden_fault_stress", "hidden_hazard_per_min",
    "hidden_flip_now", "hidden_flip_type", "hidden_flip_bit_class",
    "hidden_flip_severity", "hidden_flip_target",
    "bit_flip_risk", "flip_in_next_window",
    "run_id", "node_id", "gpu_id", "source", "split", "timestamp",
]

GPU_MODEL_MAP = {"H100": 1, "B200": 0}  


assert len(FEATURES) == 31, f"expected 31 features, got {len(FEATURES)}"
assert len(set(FEATURES)) == len(FEATURES), "duplicate feature names"
assert not set(FEATURES) & set(NEVER_USE), "feature overlaps NEVER_USE"
assert not any(c.startswith("hidden_") for c in FEATURES), "hidden_ column in FEATURES"


def build_features(df=None):
   
    if df is None:
        df = window()

    X = df[FEATURES].copy()
    X["gpu_model"] = X["gpu_model"].map(GPU_MODEL_MAP)

    if X["gpu_model"].isna().any():
        bad = df.loc[X["gpu_model"].isna(), "gpu_model"].unique()
        raise ValueError(f"Unmapped gpu_model labels: {bad}")

    return X, df["bit_flip_risk"], df["flip_in_next_window"], df["split"]


if __name__ == "__main__":
    X, y_risk, y_flip, split = build_features()
    print(X.shape)                     
    print(split.value_counts())
    print(y_flip.mean())                 