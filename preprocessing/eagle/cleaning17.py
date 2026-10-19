import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NODES = ["r103u17", "r104u29"]
DATA_DIR = Path("data")
OUT_CSV = Path("eagle_clean.csv")
CHUNK_SIZE = 5_000_000

METRICS = {
    "temp": ("temp_core_c", 10, 110),
    "power_usage_report": ("power_w", 0, 450),
    "mem_util": ("util_mem", 0, 100),
}
WANTED = [f"gpu{g}_{m}" for g in (0, 1) for m in METRICS]
METRIC_NAMES = list(METRICS)


def read_filtered(node):
    zip_path = DATA_DIR / f"esif.hpc.eagle.ganglia.gpu.sixnodes_{node}.csv.zip"
    inner = f"esif.hpc.eagle.ganglia.gpu.sixnodes_{node}.csv"
    parts, total_rows = [], 0

    with zipfile.ZipFile(zip_path) as z, z.open(inner) as f:
        for i, chunk in enumerate(pd.read_csv(
                f, usecols=["__time", "mt", "vl"], chunksize=CHUNK_SIZE)):
            total_rows += len(chunk)
            sub = chunk[chunk["mt"].isin(WANTED)]

            parts_mt = sub["mt"].str.extract(r"^gpu(\d)_(.+)$")
            out = pd.DataFrame({
                "timestamp": pd.to_datetime(sub["__time"], utc=True)
                               .dt.round("min").dt.tz_localize(None),
                "gpu_id": parts_mt[0].astype("int8"),
                "metric": pd.Categorical(parts_mt[1], categories=METRIC_NAMES),
                "vl": pd.to_numeric(sub["vl"], errors="coerce").astype("float32"),
            })
            parts.append(out)
            print(f"  {node} chunk {i + 1}: kept {len(out)}")

    long_df = pd.concat(parts, ignore_index=True)
    print(f"  {node}: read {total_rows} rows, kept {len(long_df)}")
    return long_df


def to_wide(long_df):
    """One row per GPU per minute; average readings that land in the same minute."""
    wide = (long_df.groupby(["timestamp", "gpu_id", "metric"], observed=True)["vl"]
                   .mean().unstack("metric").reset_index())
    wide.columns.name = None
    return wide.rename(columns={m: v[0] for m, v in METRICS.items()})


def range_check(df):
    """Out-of-range readings become NaN (never 0). Existing NaNs are not counted."""
    for _, (col, lo, hi) in METRICS.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        print(f"  range check {col}: {int(bad.sum())} replaced with NaN")
        df.loc[bad, col] = np.nan
    return df


def add_flags_and_ids(df, node):
    cols = [v[0] for v in METRICS.values()]
    for c in cols:
        df[f"{c}_missing"] = df[c].isna().astype("int8")
        print(f"  {c}: {int(df[c].isna().sum())} NaN cells")

    df = df.sort_values(["gpu_id", "timestamp"]).reset_index(drop=True)
    df["gap_min"] = df.groupby("gpu_id")["timestamp"].diff().dt.total_seconds() / 60

    df.insert(0, "run_id", node + "_gpu" + df["gpu_id"].astype(str))
    df.insert(1, "node", node)
    df["gpu_model"] = "Eagle-16GB"
    df["source"] = "eagle"
    return df


frames = []
for node in NODES:
    print(f"== {node} ==")
    df = to_wide(read_filtered(node))
    print("  shape after pivot:", df.shape)
    df = range_check(df)
    df = add_flags_and_ids(df, node)
    print("  rows per gpu:", df.groupby("gpu_id").size().to_dict())
    print("  biggest gap (min):", df["gap_min"].max())
    frames.append(df)

eagle = pd.concat(frames, ignore_index=True)
eagle.to_csv(OUT_CSV, index=False)

print("\n== FINAL ==")
print("shape:", eagle.shape, "| run_ids:", eagle["run_id"].nunique())
print(eagle[["temp_core_c", "power_w", "util_mem"]].agg(["min", "mean", "max"]).round(1))
print(eagle.columns)