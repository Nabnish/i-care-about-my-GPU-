import pandas as pd

def window():
    df = pd.read_csv(r"data\cleaned\synthetic_gpu_dataset.csv.gz")
    df = df.sort_values(by="run_id")
    df = df.sort_values(by="timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.drop(columns=["ecc_dbe_total","ecc_sbe_total"])
    df = df.sort_values(["run_id", "gpu_id", "timestamp"])

    groups = df.groupby(["run_id", "gpu_id"])

    for col in ["temp_core_c", "voltage_v", "power_w"]:
        df[f"{col}_15mins_avg"] = groups[col].transform(
            lambda x: x.rolling(window=15, min_periods=1).mean()
        )

        df[f"{col}_60mins_avg"] = groups[col].transform(
            lambda x: x.rolling(window=60, min_periods=1).mean()
        )
    return df

df = window()
'''print(df.shape)
print(df.describe)
print(df.columns)
print(df.isna().any())'''
