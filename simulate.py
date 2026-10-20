"""Stylised 'measurement' simulator.

IMPORTANT: this is NOT hardware ground truth. It encodes effects that the plain Roofline
model ignores, so the pipeline can be exercised end to end:

  * L2 residency / warm-cache reuse (DRAM traffic < compulsory traffic),
  * tile-reload traffic for GEMM-like kernels when the working set exceeds L2,
  * wave quantisation and small-dimension under-utilisation of compute,
  * poor coalescing for gathers, per-chip hidden efficiency (throttling, memory controller),
  * overlap imperfection and measurement noise.

Any 'ML beats Roofline' result on this data only shows the models can recover the effects
encoded here. Use measure_gpu.py for real labels.
"""
import math
import zlib
import numpy as np
import pandas as pd

from .hw_specs import HARDWARE, effective_peak
from .workloads import generate_workloads
from .features import featurize, labels_from_ratio

LAUNCH_S = 4e-6
MEM_LAT_S = 1.5e-6
RELOAD_ABSORB = 0.03     # fraction of tile reloads that actually reach DRAM (L2 rasterisation)
NOISE_SIGMA = 0.06


def _hidden_eff(name):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    return rng.uniform(0.72, 0.92), rng.uniform(0.78, 0.93)  # compute, memory


def simulate_one(w, hw, rng=None):
    """rng=None gives the noise-free (deterministic) estimate used for diagnostics."""
    nz = (lambda: rng.lognormal(0, NOISE_SIGMA)) if rng is not None else (lambda: 1.0)
    P, used_tc = effective_peak(hw, w["precision"], bool(w["tensor_eligible"]))
    B = hw.bw_gbs * 1e9
    L2 = hw.l2_mb * 2**20
    hid_c, hid_m = _hidden_eff(hw.name)

    cover = min(1.0, L2 / w["ws"])
    warm = 0.8 * cover**2
    miss = 1.0 - cover
    q_dram = (w["bytes"] + RELOAD_ABSORB * w["reload"] * miss) * (1.0 - warm)
    q_dram = max(q_dram, 0.02 * w["bytes"])

    tiles = w["tiles"]
    wave_eff = tiles / (hw.sms * math.ceil(tiles / hw.sms))
    dim_thr = 64 if used_tc else 16
    dim_pen = min(1.0, w["min_dim"] / dim_thr) ** 0.5
    eff_c = max(0.02, w["base_eff"] * wave_eff * dim_pen * hid_c)

    if w["irregular"]:
        eff_m = 0.8 * min(1.0, w["row_bytes"] / 128.0) * hid_m
    else:
        eff_m = 0.9 * hid_m
    eff_m = max(0.05, eff_m)

    t_c = w["flops"] / (P * eff_c) * nz()
    t_m = (q_dram / (B * eff_m) + MEM_LAT_S) * nz()

    ov = 0.5 if w["irregular"] else 0.8
    total = max(t_c, t_m) + (1 - ov) * min(t_c, t_m) + LAUNCH_S
    return dict(t_c=t_c, t_m=t_m, r_ratio=t_c / t_m, time_s=total,
                launch_frac=LAUNCH_S / total,
                ach_gflops=w["flops"] / total / 1e9, ach_gbs=q_dram / total / 1e9,
                q_dram=q_dram, l2_cover=cover, wave_eff=wave_eff, dim_pen=dim_pen,
                eff_c=eff_c, eff_m=eff_m)


def build_sim_dataset(seed=0, drop_launch_dominated=False):
    rng = np.random.default_rng(seed + 1)
    rows = []
    for w in generate_workloads(seed):
        for hw in HARDWARE.values():
            row = dict(family=w["family"], desc=w["desc"], hw=hw.name, precision=w["precision"],
                       flops=w["flops"], bytes=w["bytes"])
            row.update(featurize(w, hw))
            row.update(simulate_one(w, hw, rng))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["y_bin"], df["y_3"] = labels_from_ratio(df["r_ratio"].values)
    if drop_launch_dominated:
        df = df[df["launch_frac"] <= 0.5].reset_index(drop=True)
    return df
