"""
GPU bit-flip simulator: builds the synthetic, labeled dataset.

Real data supplies the shapes (workload traces, voltage curve, idle temps, outage lengths).
The simulator adds what no dataset has: ECC errors, bit flips, history, hidden labels,
and monitoring faults. Every number marked PLACEHOLDER is a tunable guess, not a fact.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- config
PATHS = {                       # cleaned real data (csv or parquet both work)
    "node": "data/cleaned/workload_node_clean.csv",
    "single": "data/cleaned/workload_single_machine_clean.csv",
    "eagle": "data/cleaned/eagle_clean.csv",
}
OUT_DIR = Path("data/generated")

CFG = {
    "seed": 42,
    "n_gpus": 48,               # 2 GPUs per node
    "days": 14,
    "window": 15,               # target: flip within the next 15 minutes
    "target_pos_rate": 0.02,    # share of windows that contain a flip
    "silent_share": 0.35,       # PLACEHOLDER: share of flips with no ECC event
    "availability": 0.90,       # share of time a node's telemetry is present (1.0 = no outages)
    "class_share": {"weak": 0.10, "moderate": 0.20},          # rest are healthy
    "class_factor": {"healthy": (1, 1), "moderate": (3, 6), "weak": (15, 40)},   # PLACEHOLDER risk multipliers
    "fault_rate_per_day": {"healthy": 0.05, "moderate": 0.15, "weak": 0.6},      # PLACEHOLDER
    "sbe_per_day_healthy": 0.1,  # PLACEHOLDER: correctable ECC errors per day on a healthy GPU
    "dbe_ratio": 0.02,           # PLACEHOLDER: double-bit errors per single-bit error
    "visible_flip_sbe_share": 0.9,  # PLACEHOLDER: a visible flip raises an SBE (else a DBE); Eagle shows DBEs are very rare
    "split_fraction": 0.15,      # val and test each get about 15% of GPUs per class
}
TDP = {"H100": 700.0, "B200": 1000.0}      # from the data: peak power / peak % of limit
SCHEDULE_PROBS = {"heavy": 0.50, "medium": 0.25, "light": 0.10, "idle": 0.15}      # PLACEHOLDER
BIT_CLASSES = ["sign", "exp_high", "exp_low", "mant_high", "mant_low"]
BIT_CLASS_P = np.array([1, 4, 4, 8, 15]) / 32.0   # share of the 32 bits in each class
# PLACEHOLDER severity table (none, mild, severe). Replace with the laptop injection results.
SEVERITY = {
    "sign": (0.15, 0.45, 0.40),
    "exp_high": (0.05, 0.10, 0.85),
    "exp_low": (0.55, 0.30, 0.15),
    "mant_high": (0.80, 0.18, 0.02),
    "mant_low": (0.97, 0.03, 0.00),
}
SENSORS = ["temp_core_c", "temp_mem_c", "power_w", "util_gpu", "util_mem",
           "voltage_v", "ecc_sbe_total", "ecc_dbe_total"]
HEAT_REF_C = 50.0            # temperature above which risk starts to rise (small effect)
HOTSPOT_OFFSET_C = 10.3      # measured: hot-spot minus core on the RTX 3060 runs


def read_any(p):
    p = Path(p)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


# ----------------------------------------------------------------------------- real-data libraries
def build_libraries(node, single, eagle):
    node = node.copy()
    node["minute"] = (node["t_sec"] // 60).clip(upper=14)
    cols = ["temp_core_c", "power_pct_tdp", "util_gpu", "util_mem"]
    per_min = node.groupby(["gpu_model", "run_id", "gpu_id", "minute"])[cols].mean().reset_index()
    traces = {}
    for (model, _, _), d in per_min.groupby(["gpu_model", "run_id", "gpu_id"]):
        d = d.sort_values("minute")
        arr = d[cols].to_numpy()
        if len(arr) < 15 or np.isnan(arr).any():
            continue
        u = arr[:, 2].mean()
        cls = "heavy" if u >= 70 else "medium" if u >= 40 else "light"
        traces.setdefault((model, cls), []).append(arr[:15])

    # voltage curve vs GPU load, learned from the RTX 3060 runs
    s = single.dropna(subset=["util_gpu", "voltage_v"]).copy()
    s["bin"] = (s["util_gpu"] // 5 * 5).clip(upper=95)
    g = s.groupby("bin")["voltage_v"].agg(["median", "std"]).reset_index()
    volt = {"x": g["bin"].to_numpy() + 2.5, "med": g["median"].to_numpy(), "sd": g["std"].fillna(0).to_numpy()}

    idle = eagle[(eagle["util_mem"] == 0) & (eagle["power_w"] < 70)]["temp_core_c"].dropna().to_numpy()

    # outage lengths from Eagle gaps (minutes). Drop multi-day gaps, they would swamp a short run.
    e = eagle[eagle["run_id"].str.endswith("gpu0")]
    gaps = (e["gap_min"][e["gap_min"] > 5] - 1).to_numpy()
    pool = gaps[(gaps >= 5) & (gaps <= 2880)]
    return traces, volt, idle, pool


def pick_trace(traces, model, cls, rng):
    for key in [(model, cls)] + [(m, cls) for m in TDP if m != model]:
        if traces.get(key):
            lst = traces[key]
            return lst[rng.integers(len(lst))]
    allv = [t for lst in traces.values() for t in lst]
    return allv[rng.integers(len(allv))]


# ----------------------------------------------------------------------------- one GPU, stage 1
def trapezoid(n, level, kind):
    x = np.arange(n) / max(n, 1)
    if kind == "ecc":
        return np.interp(x, [0, 0.5, 1], [0, level, level])
    return np.interp(x, [0, 0.2, 0.8, 1], [0, level, level, 0])


def simulate_state(spec, T, libs, rng):
    traces, volt, idle, _ = libs
    model, cls = spec["gpu_model"], spec["class"]

    # --- workload schedule from real 15-minute traces
    temp_s, pw_s, ug_s, um_s = (np.empty(T) for _ in range(4))
    types, probs = list(SCHEDULE_PROBS), np.array(list(SCHEDULE_PROBS.values()))
    t = 0
    while t < T:
        typ = types[rng.choice(len(types), p=probs / probs.sum())]
        for _ in range(int(rng.integers(1, 17))):
            if t >= T:
                break
            n = min(15, T - t)
            if typ == "idle":
                temp_s[t:t + n] = rng.choice(idle)
                pw_s[t:t + n] = rng.uniform(14, 20)
                ug_s[t:t + n] = 0.0
                um_s[t:t + n] = 0.0
            else:
                a = pick_trace(traces, model, typ, rng)
                temp_s[t:t + n], pw_s[t:t + n] = a[:n, 0], a[:n, 1]
                ug_s[t:t + n], um_s[t:t + n] = a[:n, 2], a[:n, 3]
            t += n

    # --- fault episodes (hidden)
    st_th, st_v, st_e, temp_add = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    for _ in range(rng.poisson(CFG["fault_rate_per_day"][cls] * T / 1440)):
        mode = rng.choice(["thermal", "voltage", "ecc"], p=[0.4, 0.4, 0.2])
        dur = int(rng.integers(1440, 4320)) if mode == "ecc" else \
              int(rng.integers(30, 180)) if mode == "thermal" else int(rng.integers(20, 120))
        a0 = int(rng.integers(0, T))
        a1 = min(T, a0 + dur)
        prof = trapezoid(a1 - a0, rng.uniform(0.5, 1.0), "ecc" if mode == "ecc" else "up_down")
        if mode == "thermal":
            st_th[a0:a1] = np.maximum(st_th[a0:a1], prof)
            temp_add[a0:a1] += prof * rng.uniform(8, 15)
        elif mode == "voltage":
            st_v[a0:a1] = np.maximum(st_v[a0:a1], prof)
        else:
            st_e[a0:a1] = np.maximum(st_e[a0:a1], prof)
    stress = np.vstack([st_th, st_v, st_e])
    fault_stress = stress.max(axis=0)
    fault_mode = np.where(fault_stress < 0.05, "none", np.array(["thermal", "voltage", "ecc"])[stress.argmax(axis=0)])

    # --- sensors
    temp = pd.Series(temp_s + spec["temp_offset"] + temp_add).ewm(alpha=0.35, adjust=False).mean().to_numpy() \
        + rng.normal(0, 0.4, T)
    power_pct = np.clip(pw_s + rng.normal(0, 1.0, T) + st_v * rng.normal(0, 4, T), 0, 110)
    util_gpu = np.clip(ug_s + rng.normal(0, 1.5, T), 0, 100)
    util_mem = np.clip(um_s + rng.normal(0, 1.5, T), 0, 100)
    power_w = power_pct * TDP[model] / 100.0
    throttle = ((temp >= 83) | (power_pct >= 99.5)).astype(np.int8)

    droop = np.zeros(T)
    for s0 in np.flatnonzero(rng.random(T) < (0.0015 + 0.25 * st_v)):
        end = min(T, s0 + int(rng.integers(1, 4)))
        droop[s0:end] = np.maximum(droop[s0:end], rng.uniform(0.03, 0.08))
    v_med = np.interp(util_gpu, volt["x"], volt["med"])
    v_sd = np.interp(util_gpu, volt["x"], volt["sd"])
    voltage = np.clip(v_med + rng.normal(0, 1, T) * 0.4 * v_sd * (1 + 2 * st_v) - droop, 0.55, 1.15)
    temp_mem = pd.Series(temp + HOTSPOT_OFFSET_C).ewm(alpha=0.5, adjust=False).mean().to_numpy() + rng.normal(0, 0.4, T)

    # --- hazard pieces (what makes a GPU risky)
    f_temp = 1 + 0.03 * np.maximum(0, temp - HEAT_REF_C)          # small effect
    f_volt = 1 + 2 * (droop > 0) + 2 * st_v
    f_load = 0.5 + util_gpu / 100.0
    f_fault_ecc = 1 + 1 * st_th + 6 * st_e
    mu = CFG["sbe_per_day_healthy"] / 1440.0 * spec["weak_factor"] * f_temp * f_volt * f_load * f_fault_ecc
    sbe_bg = rng.poisson(mu)
    dbe_bg = (rng.random(T) < CFG["dbe_ratio"] * mu).astype(int)
    prior = np.concatenate([[0], np.cumsum(sbe_bg)[:-1]])          # errors up to the previous minute
    sbe_24h_true = prior - np.concatenate([np.zeros(1440), prior[:-1440]])
    f_hist = 1 + 0.15 * np.minimum(sbe_24h_true, 20)               # error history matters a lot
    f_fault_flip = 1 + 3 * st_th + 3 * st_v + 5 * st_e
    base_h = spec["weak_factor"] * f_temp * f_volt * f_load * f_hist * f_fault_flip

    return dict(temp_core_c=temp, temp_mem_c=temp_mem, power_w=power_w, power_pct_tdp=power_pct,
                util_gpu=util_gpu, util_mem=util_mem, voltage_v=voltage, throttle_flag=throttle,
                sbe_bg=sbe_bg, dbe_bg=dbe_bg, base_h=base_h,
                fault_stress=fault_stress, fault_mode=fault_mode)


def window_sum(x, w):
    c = np.concatenate([[0], np.cumsum(x)])          # c[i] = sum of x[:i]
    n = len(x) - w
    return c[np.arange(n) + w + 1] - c[np.arange(n) + 1]      # sum of x[t+1 .. t+w]


def calibrate_lambda(all_w, target):
    lo, hi = -14.0, 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.mean(1 - np.exp(-(10 ** mid) * all_w)) < target:
            lo = mid
        else:
            hi = mid
    return 10 ** ((lo + hi) / 2)


# ----------------------------------------------------------------------------- stage 2: flips + targets
def add_flips_and_targets(state, spec, lam, T, rng):
    W = CFG["window"]
    h = lam * state["base_h"]
    flip = rng.random(T) < 1 - np.exp(-h)
    silent = flip & (rng.random(T) < CFG["silent_share"])
    visible = flip & ~silent
    as_sbe = visible & (rng.random(T) < CFG["visible_flip_sbe_share"])
    as_dbe = visible & ~as_sbe

    ftype = np.where(silent, "silent", np.where(visible, "visible", "none")).astype(object)
    bit = np.full(T, "none", dtype=object)
    sev = np.full(T, "none", dtype=object)
    tgt = np.full(T, "none", dtype=object)
    for i in np.flatnonzero(flip):
        b = BIT_CLASSES[rng.choice(len(BIT_CLASSES), p=BIT_CLASS_P)]
        bit[i] = b
        sev[i] = ["none", "mild", "severe"][rng.choice(3, p=SEVERITY[b])]
        tgt[i] = ["weight", "activation", "gradient"][rng.integers(3)]

    n = T - W
    cf = np.concatenate([[0], np.cumsum(flip)])
    out = pd.DataFrame({
        "temp_core_c": state["temp_core_c"], "temp_mem_c": state["temp_mem_c"], "power_w": state["power_w"],
        "power_pct_tdp": state["power_pct_tdp"], "util_gpu": state["util_gpu"], "util_mem": state["util_mem"],
        "voltage_v": state["voltage_v"], "throttle_flag": state["throttle_flag"],
        "ecc_sbe_total": np.cumsum(state["sbe_bg"] + as_sbe).astype(float),
        "ecc_dbe_total": np.cumsum(state["dbe_bg"] + as_dbe).astype(float),
        "hidden_weak_class": spec["class"], "hidden_weak_factor": spec["weak_factor"],
        "hidden_fault_mode": state["fault_mode"], "hidden_fault_stress": state["fault_stress"],
        "hidden_hazard_per_min": h, "hidden_flip_now": flip.astype(np.int8), "hidden_flip_type": ftype,
        "hidden_flip_bit_class": bit, "hidden_flip_severity": sev, "hidden_flip_target": tgt,
        "bit_flip_risk": 1 - np.exp(-W * h),     # flip probability over the next 15 min if conditions persist
    })
    nxt = np.zeros(T)
    nxt[:n] = cf[np.arange(n) + W + 1] - cf[np.arange(n) + 1]
    out["flip_in_next_window"] = (nxt > 0).astype(np.int8)
    return out.iloc[:n].reset_index(drop=True)            # last 15 minutes have no full window


# ----------------------------------------------------------------------------- monitoring faults (applied last)
def node_outages(T, pool, rng):
    mask = np.zeros(T, bool)
    kind = np.full(T, "", dtype=object)
    need = (1 - CFG["availability"]) * T
    while mask.sum() < need and len(pool):
        d = int(rng.choice(pool))
        s0 = int(rng.integers(0, T))
        mask[s0:s0 + d] = True
        kind[s0:s0 + d] = "agent_outage" if rng.random() < 0.7 else "gpu_blackout"
    return mask, kind


def apply_monitoring(df, outage, okind, rng):
    n = len(df)
    days = n / 1440
    obs = {s: df[s].to_numpy(float).copy() for s in SENSORS}
    missing = {s: outage.copy() for s in SENSORS}
    stale = {s: np.zeros(n, bool) for s in SENSORS}
    stuck_any, stale_any, drop_any = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)

    for s in SENSORS:                                              # short per-sensor dropouts
        for _ in range(rng.poisson(0.3 * days)):
            a = int(rng.integers(0, n))
            b = a + int(rng.integers(1, 6))
            missing[s][a:b] = True
            drop_any[a:b] = True
    for _ in range(rng.poisson(0.05 * days)):                      # stuck sensor: constant but looks fresh
        s = ["temp_core_c", "power_w", "util_gpu"][rng.integers(3)]
        a = int(rng.integers(1, n))
        b = min(n, a + int(rng.integers(60, 480)))
        obs[s][a:b] = obs[s][a - 1]
        stuck_any[a:b] = True
    for _ in range(rng.poisson(0.1 * days)):                       # stale: old value repeated, age grows
        for s in rng.choice(SENSORS, size=int(rng.integers(1, 4)), replace=False):
            a = int(rng.integers(1, n))
            b = min(n, a + int(rng.integers(30, 240)))
            stale[s][a:b] = True
            stale_any[a:b] = True

    idx = np.arange(n)
    res = {}
    for s in SENSORS:
        fresh = ~(missing[s] | stale[s])
        last = np.maximum.accumulate(np.where(fresh, idx, -1))
        age = np.where(last >= 0, idx - last, idx + 1).astype(np.int32)
        val = np.where(last >= 0, obs[s][np.maximum(last, 0)], np.nan)
        val = np.where(missing[s], np.nan, np.where(stale[s], val, obs[s]))
        res[s] = val
        res[f"{s}_missing"] = missing[s].astype(np.int8)
        res[f"{s}_age"] = age
    out = pd.DataFrame(res)

    out["power_pct_tdp"] = np.where(np.isnan(out["power_w"]), np.nan, df["power_pct_tdp"])
    out["throttle_flag"] = np.where(out["temp_core_c_missing"].astype(bool) | out["power_w_missing"].astype(bool),
                                    np.nan, df["throttle_flag"])
    s_obs = out["ecc_sbe_total"].ffill().fillna(0)
    out["sbe_24h"] = s_obs - s_obs.shift(1440).fillna(0)           # history computed only from what was observed
    fault = np.where(outage, okind, np.where(stale_any, "stale", np.where(stuck_any, "stuck_sensor",
                     np.where(drop_any, "sensor_dropout", "none"))))
    out["hidden_monitor_fault"] = fault
    return out


# ----------------------------------------------------------------------------- fleet + main
def make_fleet(rng):
    n = CFG["n_gpus"]
    n_weak = round(CFG["class_share"]["weak"] * n)
    n_mod = round(CFG["class_share"]["moderate"] * n)
    classes = ["weak"] * n_weak + ["moderate"] * n_mod + ["healthy"] * (n - n_weak - n_mod)
    rng.shuffle(classes)
    models = np.array(["H100", "B200"] * (n // 2 + 1))[:n]
    rng.shuffle(models)
    fleet = [dict(idx=i, **{"class": c}, gpu_model=models[i]) for i, c in enumerate(classes)]
    for f in fleet:
        lo, hi = CFG["class_factor"][f["class"]]
        f["weak_factor"] = float(rng.uniform(lo, hi)) if hi > lo else 1.0
        f["temp_offset"] = float(rng.normal(0, 2.0))
    for c in ["weak", "moderate", "healthy"]:                       # split by GPU, stratified by class
        grp = [f for f in fleet if f["class"] == c]
        rng.shuffle(grp)
        k = max(1, round(CFG["split_fraction"] * len(grp))) if len(grp) >= 3 else 0
        for j, f in enumerate(grp):
            f["split"] = "val" if j < k else "test" if j < 2 * k else "train"
    for sp in ["train", "val", "test"]:                             # pair GPUs into nodes within a split
        grp = [f for f in fleet if f["split"] == sp]
        for j, f in enumerate(grp):
            f["node"], f["gpu_in_node"] = f"{sp}_node{j // 2:02d}", j % 2
    return fleet


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ss = np.random.SeedSequence(CFG["seed"])
    fleet_rng, *gpu_rngs = [np.random.default_rng(s) for s in ss.spawn(1 + 3 * CFG["n_gpus"] + 64)]
    libs = build_libraries(read_any(PATHS["node"]), read_any(PATHS["single"]), read_any(PATHS["eagle"]))
    print({k: len(v) for k, v in libs[0].items()}, "| outage-length pool:", len(libs[3]))
    T = CFG["days"] * 1440
    fleet = make_fleet(fleet_rng)

    states = [simulate_state(f, T, libs, gpu_rngs[i]) for i, f in enumerate(fleet)]
    all_w = np.concatenate([window_sum(s["base_h"], CFG["window"]) for s in states])
    lam = calibrate_lambda(all_w, CFG["target_pos_rate"])
    print(f"calibrated base flip rate: {lam:.3e} per GPU-minute")

    frames = {}
    n = T - CFG["window"]
    for i, (f, st) in enumerate(zip(fleet, states)):
        frames[i] = add_flips_and_targets(st, f, lam, T, gpu_rngs[CFG["n_gpus"] + i])

    nodes = {}
    for f in fleet:
        nodes.setdefault(f["node"], []).append(f["idx"])
    ts = pd.date_range("2026-01-01", periods=n, freq="min")
    parts = []
    for ni, (node, members) in enumerate(sorted(nodes.items())):
        mrng = gpu_rngs[2 * CFG["n_gpus"] + ni]
        outage, okind = node_outages(n, libs[3], mrng)
        for i in members:
            f, d = fleet[i], frames[i]
            obs = apply_monitoring(d, outage, okind, gpu_rngs[2 * CFG["n_gpus"] + 10 + i])
            head = pd.DataFrame({"split": f["split"], "run_id": f"sim_gpu{i:02d}", "node_id": node,
                                 "gpu_id": f["gpu_in_node"], "gpu_model": f["gpu_model"],
                                 "source": "simulated", "timestamp": ts})
            hidden = d[[c for c in d.columns if c.startswith("hidden_")]]
            keep = ["power_pct_tdp", "throttle_flag", "sbe_24h"] + SENSORS + \
                   [f"{s}_missing" for s in SENSORS] + [f"{s}_age" for s in SENSORS] + ["hidden_monitor_fault"]
            parts.append(pd.concat([head, obs[keep], hidden, d[["bit_flip_risk", "flip_in_next_window"]]], axis=1))

    data = pd.concat(parts, ignore_index=True)
    out_path = OUT_DIR / "synthetic_gpu_dataset.csv.gz"
    data.to_csv(out_path, index=False, compression="gzip", float_format="%.4g")
    json.dump({"config": CFG, "lambda": lam, "rows": len(data), "placeholder_severity_table": SEVERITY},
              open(OUT_DIR / "synthetic_config.json", "w"), indent=1)

    print("\nrows", len(data), "| GPUs", data["run_id"].nunique(), "| file", out_path)
    print(data.groupby("split").agg(gpus=("run_id", "nunique"), rows=("run_id", "size"),
                                    pos_rate=("flip_in_next_window", "mean")).round(4))
    print(data.groupby("hidden_weak_class").agg(gpus=("run_id", "nunique"), pos_rate=("flip_in_next_window", "mean"),
                                                sbe=("ecc_sbe_total", "max"), dbe=("ecc_dbe_total", "max")).round(3))


if __name__ == "__main__":
    main()