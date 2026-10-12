import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZIP_PATH = Path("data") / "High-resolution-AI-Data-Center-Training-Workloads-Dataset_FigShare.zip"
ROOT = "High-resolution-AI-Data-Center-Training-Workloads-Dataset_FigShare/"
NODE = ROOT + "Node_Dataset/"
SINGLE = ROOT + "Single_Machine_Dataset/"

NODE_FILES = [
    NODE + "Text Generation LLMs/B200/Batch_size/LLM_B200_8GPU_Node_Lamma8B_SeqLength2048_Deepspeedds_z3_BatchSize32.csv",
    NODE + "Text Generation LLMs/H100/Batch_size/LLM_H100_8GPU_Node_Lamma8B_SeqLength2048_Deepspeedds_z3_BatchSize32.csv",
    NODE + "Image Generation Diffusion Models/H100/Image_size_impact/Image_generation_H100_8GPU_Node_ModelSize128_ImageSize64_BatchSize128.csv",
    NODE + "Image Generation Diffusion Models/B200/Image_size_impact/Image_generation_B200_8GPU_Node_ModelSize128_ImageSize128_BatchSize128.csv",
]
SINGLE_FILES = [
    SINGLE + "Text Generation/Sequance Length/Single_machine_SeqLen500_EmbedDim100_batchsize128.CSV",
    SINGLE + "Feature Forecasting/Model Size/Single_machine_NumLayer6_SeqLength96_LayerSize1500_batchsize100.CSV",
]

NODE_COLS = {
    "utilization_percent": ("util_gpu", 0, 100),
    "mem_utilization": ("util_mem", 0, 100),
    "Power_TDP": ("power_pct_tdp", 0, 150),
    "power_W": ("power_w", 0, 1300),
    "temp_C": ("temp_core_c", 10, 110),
}
SINGLE_COLS = {
    "GPU Temperature [": ("temp_core_c", 10, 110),
    "GPU Hot Spot Temperature [": ("temp_hotspot_c", 10, 130),
    "GPU Core Voltage [": ("voltage_v", 0.3, 1.5),
    "GPU Power [W]": ("power_w", 0, 500),
    "GPU Core Load [": ("util_gpu", 0, 100),
    "GPU Memory Controller Load [": ("util_mem", 0, 100),
    "Total GPU Power [% of TDP]": ("power_pct_tdp", 0, 150),
}


def range_check(df, ranges, label):
    for col, lo, hi in ranges:
        bad = df[col].notna() & ~df[col].between(lo, hi)
        if bad.any():
            print(f"  {label}: {col} {int(bad.sum())} out of range -> NaN")
        df.loc[bad, col] = np.nan


def add_missing_flags(df, cols):
    for c in cols:
        df[f"{c}_missing"] = df[c].isna().astype("int8")


def clean_node_file(z, path):
    stem = Path(path).stem
    gpu_model = re.search(r"_(H100|B200)_", stem).group(1)
    workload = "diffusion" if stem.startswith("Image_generation") else "llm"

    with z.open(path) as f:
        df = pd.read_csv(f, usecols=lambda c: c == "timestamp" or c.startswith("gpu"))

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[ts.notna()].copy()
    ts = ts[ts.notna()]
    keep = ~ts.duplicated()
    df, ts = df[keep], ts[keep]
    order = ts.argsort()
    df, ts = df.iloc[order], ts.iloc[order]
    t0 = ts.iloc[0]
    sec = ((ts - t0).dt.total_seconds()).astype(int).values  

    ranges = [(v[0], v[1], v[2]) for v in NODE_COLS.values()]
    frames = []
    for g in range(8):
        sub = pd.DataFrame({"sec": sec})
        for suffix, (name, _, _) in NODE_COLS.items():
            sub[name] = pd.to_numeric(df[f"gpu{g}_{suffix}"], errors="coerce").values
        range_check(sub, ranges, f"{stem[:40]} gpu{g}")
        one = sub.groupby("sec").mean().reset_index()     
        one["gpu_id"] = g
        frames.append(one)

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = t0.floor("s") + pd.to_timedelta(out["sec"], unit="s")
    out = out.rename(columns={"sec": "t_sec"})
    add_missing_flags(out, ["temp_core_c", "power_w", "util_gpu"])
    out["run_id"], out["gpu_model"], out["workload"], out["source"] = stem, gpu_model, workload, "workload_zip"
    return out


def clean_single_file(z, path):
    stem = Path(path).stem
    application = path.split("/")[2]

    with z.open(path) as f:
        header = list(pd.read_csv(f, encoding="latin-1", nrows=0).columns)
    pick = {name: next(c for c in header if c.startswith(prefix))
            for prefix, (name, _, _) in SINGLE_COLS.items()}

    with z.open(path) as f:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False,
                         usecols=["Date", "Time"] + list(pick.values()))

    df = df[df["Date"].str.match(r"^\d+\.\d+\.\d{4}$", na=False)].copy()

    out = pd.DataFrame({name: pd.to_numeric(df[col], errors="coerce") for name, col in pick.items()})
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")
    if ts.isna().any():
        print(f"  WARNING {stem}: {int(ts.isna().sum())} rows with unreadable time dropped")
    out["ts"] = ts
    out = out[out["ts"].notna()].drop_duplicates("ts").sort_values("ts")

    ranges = [(v[0], v[1], v[2]) for v in SINGLE_COLS.values()]
    range_check(out, ranges, stem[:40])

    t0 = out["ts"].iloc[0]
    out["t_sec"] = ((out["ts"] - t0).dt.total_seconds()).astype(int)   
    out = out.drop(columns="ts").groupby("t_sec").mean().reset_index() 
    out["timestamp"] = t0.floor("s") + pd.to_timedelta(out["t_sec"], unit="s")

    add_missing_flags(out, ["temp_core_c", "power_w", "util_gpu", "voltage_v"])
    out["run_id"], out["application"] = stem, application
    out["gpu_id"], out["gpu_model"], out["source"] = 0, "RTX3060", "workload_zip"
    return out


with zipfile.ZipFile(ZIP_PATH) as z:
    print("== node files ==")
    node_df = pd.concat([clean_node_file(z, p) for p in NODE_FILES], ignore_index=True)
    print("== single-machine files ==")
    single_df = pd.concat([clean_single_file(z, p) for p in SINGLE_FILES], ignore_index=True)

node_df.to_csv("workload_node_clean.csv", index=False)
single_df.to_csv("workload_single_machine_clean.csv", index=False)

print("\nNODE: rows", len(node_df), "| per run:", node_df.groupby("run_id").size().to_dict())
print(node_df.groupby("run_id")[["temp_core_c", "power_w"]].max().round(1))
print("\nSINGLE: rows", len(single_df), "| per run:", single_df.groupby("run_id").size().to_dict())
print(single_df.groupby("run_id")[["voltage_v", "temp_core_c"]].agg(["min", "max"]).round(3))