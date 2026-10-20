"""Real measured kernel latencies (PipeWeave dataset) -> unified table of Roofline efficiency.

For every measured kernel:
    ideal_time = max(FLOPs / peak, bytes / bandwidth)          (Roofline, analytic)
    efficiency = ideal_time / measured_time                    (1.0 = on the roofline)
Target for learning: y = log(measured_time / ideal_time) = -log(efficiency).

FLOPs / bytes / peak are ANALYTIC and rest on assumptions listed in OP_NOTES - the dataset has
runtimes only (no bottleneck labels). Only inputs a user can know in advance are used as features
(shape, FLOPs, bytes, peak, bandwidth, SM count); cuBLAS tile choices and the dataset's derived
cycle/op-count columns are deliberately excluded (they are outputs, not inputs).
"""
import os

import numpy as np
import pandas as pd

OPS = ["gemm", "gemm_fp8", "attn", "rmsnorm", "silu_mul", "moe_w13"]
OP_ID = {o: i for i, o in enumerate(OPS)}
FILES = {
    "gemm": ["gemm_train", "gemm_test"], "gemm_fp8": ["gemm_fp8_train", "gemm_fp8_test"],
    "attn": ["attn_train", "attn_test"], "rmsnorm": ["rmsnorm_train", "rmsnorm_test"],
    "silu_mul": ["siluandmul_train", "siluandmul_test"], "moe_w13": ["moe_train"],
}
OP_NOTES = {
    "gemm": "BF16 tensor-core GEMM. FLOPs=2MNK, bytes=2(MK+KN+MN), peak=tensor BF16.",
    "gemm_fp8": "FP8 GEMM (H20/H800 only). bytes=MK+KN+2MN (BF16 out ASSUMED), peak=2x BF16 tensor peak (Hopper).",
    "attn": "FlashAttention-style (fa2/fa3, paged/ragged). FLOPs=sum 4*q*kv*nh*hd (full, causal NOT discounted), "
            "bytes=2B*(Q+O+K+V), peak=tensor BF16.",
    "rmsnorm": "bytes=2B*(read+write seq*dim)+weights, FLOPs=4*seq*dim (ASSUMED), peak=FP32 CUDA cores.",
    "silu_mul": "reads seq x 2dim, writes seq x dim: bytes=2B*3*seq*dim, FLOPs=6*seq*dim (ASSUMED), peak=FP32.",
    "moe_w13": "fused-MoE first GEMM. ASSUMED: weights [E, N, H], uniform routing so expected distinct experts "
               "E*(1-(1-1/E)^(M*topk)) are read; FLOPs=2*M*topk*N*H. Least certain of the six.",
}
FEATURES = [
    "op_id", "log_flops", "log_bytes", "log_ai", "log_peak", "log_bw", "log_ridge", "log_ratio",
    "log_t_ideal", "log_sms", "log_peak_per_sm",
    "log_M", "log_N", "log_K", "log_seq", "log_dim",
    "bs", "log_sum_q", "log_sum_kv", "log_max_q", "log_max_kv", "nh", "nkv", "hd", "frac_q1", "attn_type",
    "E", "topk", "log_H", "distinct_experts",
]
# The dataset lists RTX 6000 Ada at 1024 tensor FLOP/cycle/SM, but on identical compute-bound GEMMs it
# delivers the same TFLOPS as the L40 (148 vs 154, same silicon) while the L40 is listed at 512. BF16 uses
# FP32 accumulate, which runs at half rate on Ada, so the effective peak is taken as half the listed one.
# This is an inference from the data, not vendor-confirmed.
PEAK_FIX = {"RTX6000Ada": 0.5}
ATTN_TYPES = {"fa2_paged": 0, "fa2_ragged": 1, "fa3_paged": 2, "fa3_ragged": 3}


def short_gpu(h):
    return (str(h).replace("NVIDIA ", "").replace(" Generation", "").replace(" Blackwell Server Edition", "")
            .replace("A100-SXM4-80GB", "A100-80").replace("RTX 6000 Ada", "RTX6000Ada")
            .replace("RTX PRO 6000", "RTX-PRO-6000").replace("RTX A6000", "RTXA6000"))


def hw_from_dataset(d):
    f = d["sm_freq"].values * 1e6
    return dict(peak_tc=d["num_sms"].values * f * d["tc_bf16"].values,
                peak_fp32=d["num_sms"].values * f * d["fma_fp32"].values * 2,
                bw=d["mem_bandwidth"].values * 1e9, sms=d["num_sms"].values.astype(float))


def _lens(x):
    if isinstance(x, str):
        return np.fromstring(x.strip("[] "), sep=",")
    return np.asarray(x, dtype=float)


def build_op(op, p, hw):
    """p: dict of op parameters (arrays or scalars); hw: dict peak_tc, peak_fp32, bw, sms (arrays or scalars).
    Returns DataFrame with flops, bytes, peak, ideal_s and all shape features (NaN where n/a)."""
    one = np.isscalar(hw["bw"])
    hw = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in hw.items()}
    out = {}
    nan = None
    if op in ("gemm", "gemm_fp8"):
        M, N, K = (np.atleast_1d(np.asarray(p[k], dtype=float)) for k in "MNK")
        flops = 2 * M * N * K
        if op == "gemm":
            nbytes, peak = 2 * (M * K + K * N + M * N), hw["peak_tc"]
        else:
            nbytes, peak = (M * K + K * N + 2 * M * N), 2 * hw["peak_tc"]
        out.update(log_M=np.log10(M), log_N=np.log10(N), log_K=np.log10(K))
    elif op == "rmsnorm":
        s, dm = (np.atleast_1d(np.asarray(p[k], dtype=float)) for k in ("seq", "dim"))
        flops, nbytes, peak = 4 * s * dm, 2 * (2 * s * dm) + 2 * dm, hw["peak_fp32"]
        out.update(log_seq=np.log10(s), log_dim=np.log10(dm))
    elif op == "silu_mul":
        s, dm = (np.atleast_1d(np.asarray(p[k], dtype=float)) for k in ("seq", "dim"))
        flops, nbytes, peak = 6 * s * dm, 2 * (3 * s * dm), hw["peak_fp32"]
        out.update(log_seq=np.log10(s), log_dim=np.log10(dm))
    elif op == "attn":
        q_all = [p["q_lengths"]] if one and not _is_seq_of_seq(p["q_lengths"]) else list(p["q_lengths"])
        kv_all = [p["kv_lengths"]] if one and not _is_seq_of_seq(p["kv_lengths"]) else list(p["kv_lengths"])
        nh = np.atleast_1d(np.asarray(p["nh"], dtype=float))
        nkv = np.atleast_1d(np.asarray(p["nkv"], dtype=float))
        hd = np.atleast_1d(np.asarray(p["hd"], dtype=float))
        n = len(nh)
        flops, nbytes = np.empty(n), np.empty(n)
        sq, skv, mq, mkv, f1 = (np.empty(n) for _ in range(5))
        for i in range(n):
            q, kv = _lens(q_all[i]), _lens(kv_all[i])
            flops[i] = np.sum(4 * q * kv) * nh[i] * hd[i]
            nbytes[i] = 2 * (2 * q.sum() * nh[i] * hd[i] + 2 * kv.sum() * nkv[i] * hd[i])
            sq[i], skv[i], mq[i], mkv[i], f1[i] = q.sum(), kv.sum(), q.max(), kv.max(), np.mean(q == 1)
        peak = hw["peak_tc"]
        bs = np.atleast_1d(np.asarray(p["bs"], dtype=float))
        at = p["attention_type"]
        at = np.atleast_1d(np.array([ATTN_TYPES.get(a, -1) for a in np.atleast_1d(at)], dtype=float))
        out.update(bs=bs, log_sum_q=np.log10(sq), log_sum_kv=np.log10(skv), log_max_q=np.log10(mq),
                   log_max_kv=np.log10(mkv), nh=nh, nkv=nkv, hd=hd, frac_q1=f1, attn_type=at)
    elif op == "moe_w13":
        M, E, topk, H, N = (np.atleast_1d(np.asarray(p[k], dtype=float)) for k in ("M", "E", "topk", "H", "N"))
        flops = 2 * M * topk * N * H
        distinct = E * (1 - (1 - 1 / E) ** (M * topk))
        nbytes = 2 * (distinct * N * H + M * H + M * topk * N)
        peak = hw["peak_tc"]
        out.update(log_M=np.log10(M), E=E, topk=topk, log_H=np.log10(H), log_N=np.log10(N), distinct_experts=distinct)
    else:
        raise ValueError(op)
    bw, sms = hw["bw"], hw["sms"]
    ai, ridge = flops / nbytes, peak / bw
    ideal = np.maximum(flops / peak, nbytes / bw)
    df = pd.DataFrame(dict(op=op, op_id=OP_ID[op], flops=flops, bytes=nbytes, peak=peak, bw=bw, sms=sms,
                           ideal_s=ideal, ai=ai, ridge=ridge, ratio=ai / ridge,
                           log_flops=np.log10(flops), log_bytes=np.log10(nbytes), log_ai=np.log10(ai),
                           log_peak=np.log10(peak), log_bw=np.log10(bw), log_ridge=np.log10(ridge),
                           log_ratio=np.log10(ai / ridge), log_t_ideal=np.log10(ideal),
                           log_sms=np.log10(sms), log_peak_per_sm=np.log10(peak / sms)))
    for k, v in out.items():
        df[k] = v
    for c in FEATURES:
        if c not in df:
            df[c] = np.nan
    return df


def _is_seq_of_seq(x):
    return isinstance(x, (list, tuple, np.ndarray, pd.Series)) and len(x) > 0 and \
        isinstance(x[0], (str, list, tuple, np.ndarray))


def _is_pointer(path):
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def load_real(data_dir="data/real", ops=None, log=print):
    frames, skipped = [], []
    for op in (ops or OPS):
        for name in FILES[op]:
            path = os.path.join(data_dir, name + ".csv")
            if not os.path.exists(path) or _is_pointer(path):
                skipped.append(name)
                continue
            d = pd.read_csv(path)
            hw = hw_from_dataset(d)
            gname = d.hardware.map(short_gpu).values
            for g, fac in PEAK_FIX.items():     # data-quirk correction, see PEAK_FIX
                hw["peak_tc"] = np.where(gname == g, hw["peak_tc"] * fac, hw["peak_tc"])
            if op in ("gemm", "gemm_fp8"):
                p = dict(M=d.M, N=d.N, K=d.K)
                key = op + "|" + d.M.astype(int).astype(str) + "," + d.N.astype(int).astype(str) + "," + d.K.astype(int).astype(str)
            elif op in ("rmsnorm", "silu_mul"):
                p = dict(seq=d.seq, dim=d.dim)
                key = op + "|" + d.seq.astype(int).astype(str) + "," + d.dim.astype(int).astype(str)
            elif op == "attn":
                p = dict(q_lengths=d.q_lengths, kv_lengths=d.kv_lengths, nh=d.nh, nkv=d.nkv, hd=d.hd, bs=d.bs,
                         attention_type=d.attention_type)
                key = op + "|" + d.attention_type + "|" + d.q_lengths + "|" + d.kv_lengths + "|" + \
                    d.nh.astype(str) + "," + d.nkv.astype(str) + "," + d.hd.astype(str)
            else:
                p = dict(M=d.M, E=d.E, topk=d.topk, H=d.H, N=d.N)
                key = op + "|" + d.M.astype(int).astype(str) + "," + d.E.astype(int).astype(str) + "," + \
                    d.topk.astype(int).astype(str) + "," + d.H.astype(int).astype(str) + "," + d.N.astype(int).astype(str)
            o = build_op(op, p, hw)
            o["gpu"] = d.hardware.map(short_gpu).values
            o["t_s"] = d.avg_duration.values * 1e-6          # dataset unit: microseconds
            o["shape_key"] = key.values
            o["file"] = name
            if "overall_perf" in d:
                o["overall_perf"] = d["overall_perf"].values
            frames.append(o)
    if not frames:
        raise SystemExit(f"no data found in {data_dir}; run `python fetch_real_data.py`")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df.t_s > 0) & np.isfinite(df.ideal_s) & (df.ideal_s > 0)].reset_index(drop=True)
    df["eff"] = df.ideal_s / df.t_s
    df["y"] = np.log10(df.t_s / df.ideal_s)                 # = -log10(efficiency)
    if skipped:
        log(f"[note] files not available (skipped): {', '.join(skipped)}")
    return df
