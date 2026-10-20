"""Predict how long a kernel takes on a GPU, and how far below its Roofline it will run,
using a model trained on REAL measured latencies (PipeWeave dataset, 11 NVIDIA GPUs).

    python train_real.py                                   # once (needs `python fetch_real_data.py` first)

    python predict_real.py --gpu H100 --op gemm --M 4096 --N 4096 --K 4096
    python predict_real.py --gpu A100-80 --op gemm --M 1 --N 8192 --K 8192            # decode GEMV
    python predict_real.py --gpu H100 --op rmsnorm --seq 8192 --dim 4096
    python predict_real.py --gpu H100 --op silu_mul --seq 8192 --dim 14336
    python predict_real.py --gpu H100 --op moe --M 512 --E 64 --topk 6 --H 2048 --N 1536
    python predict_real.py --gpu H100 --op attn --q-lens 1,1,1,1 --kv-lens 512,900,300,4000 --nh 32 --nkv 8 --hd 128
    python predict_real.py --tensor-tflops 312 --fp32-tflops 19.5 --bw-gbs 2039 --sms 108 --op gemm --M 2048 --N 2048 --K 2048

Hardware: --gpu NAME (built-in table), or --tensor-tflops (dense BF16, FP32-accumulate) --fp32-tflops
--bw-gbs --sms. Ops: gemm, gemm_fp8, attn, rmsnorm, silu_mul, moe (first MoE GEMM, w13).

Applicability: trained on tensor-core datacenter/workstation GPUs (Ampere to Blackwell) with BF16/FP8 kernels.
GPUs without tensor cores (e.g. GTX 1650 Ti) or FP32 GEMMs are OUT of this model's domain: use predict.py.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np

from rooflinelab.hw_specs import HARDWARE
from rooflinelab.realdata import FEATURES, OP_NOTES, build_op

OPMAP = {"gemm": "gemm", "gemm_fp8": "gemm_fp8", "attn": "attn", "rmsnorm": "rmsnorm", "silu_mul": "silu_mul",
         "moe": "moe_w13"}
NEEDS = {"gemm": "MNK", "gemm_fp8": "MNK", "rmsnorm": ("seq", "dim"), "silu_mul": ("seq", "dim"),
         "moe": ("M", "E", "topk", "H", "N")}
TENSOR_OPS = {"gemm", "gemm_fp8", "attn", "moe_w13"}


def si(v, unit):
    for div, p in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= div:
            return f"{v/div:.3g} {p}{unit}"
    return f"{v:.3g} {unit}"


def tfmt(s):
    return f"{s*1e6:.1f} us" if s < 1e-3 else f"{s*1e3:.3f} ms"


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu")
    ap.add_argument("--hw-json")
    for f in ("tensor-tflops", "fp32-tflops", "bw-gbs"):
        ap.add_argument("--" + f, type=float)
    ap.add_argument("--sms", type=int)
    ap.add_argument("--op", required=False)
    for f in ("M", "N", "K", "seq", "dim", "E", "topk", "H", "nh", "nkv", "hd"):
        ap.add_argument("--" + f, type=lambda s: int(float(s)))
    ap.add_argument("--q-lens")
    ap.add_argument("--kv-lens")
    ap.add_argument("--attn-type", default="fa2_paged", choices=["fa2_paged", "fa2_ragged", "fa3_paged", "fa3_ragged"])
    ap.add_argument("--model", default="models/real_efficiency.joblib")
    ap.add_argument("--json", action="store_true")
    return ap


def get_hw(a, ap):
    if a.gpu:
        if a.gpu not in HARDWARE:
            ap.error(f"unknown --gpu {a.gpu}. Known: {', '.join(HARDWARE)}")
        h = HARDWARE[a.gpu]
        return dict(name=h.name, peak_tc=h.fp16_tensor_tflops * 1e12, peak_fp32=h.fp32_tflops * 1e12,
                    bw=h.bw_gbs * 1e9, sms=float(h.sms))
    if a.hw_json:
        d = json.load(open(a.hw_json))
        return dict(name=d.get("name", "custom"), peak_tc=d.get("fp16_tensor_tflops", 0.0) * 1e12,
                    peak_fp32=d["fp32_tflops"] * 1e12, bw=d["bw_gbs"] * 1e9, sms=float(d["sms"]))
    need = dict(tensor_tflops=a.tensor_tflops, fp32_tflops=a.fp32_tflops, bw_gbs=a.bw_gbs, sms=a.sms)
    miss = [k for k, v in need.items() if v is None]
    if miss:
        ap.error("hardware: use --gpu, --hw-json, or all of --tensor-tflops --fp32-tflops --bw-gbs --sms "
                 f"(missing {miss})")
    return dict(name="custom", peak_tc=a.tensor_tflops * 1e12, peak_fp32=a.fp32_tflops * 1e12,
                bw=a.bw_gbs * 1e9, sms=float(a.sms))


def get_params(a, ap):
    if a.op not in OPMAP:
        ap.error(f"--op must be one of {', '.join(OPMAP)}")
    if a.op == "attn":
        need = dict(q_lens=a.q_lens, kv_lens=a.kv_lens, nh=a.nh, nkv=a.nkv, hd=a.hd)
        miss = [k for k, v in need.items() if v is None]
        if miss:
            ap.error(f"--op attn needs --q-lens --kv-lens --nh --nkv --hd (missing {miss})")
        q = [float(x) for x in a.q_lens.split(",")]
        kv = [float(x) for x in a.kv_lens.split(",")]
        if len(q) != len(kv):
            ap.error("--q-lens and --kv-lens must have the same number of requests")
        return dict(q_lengths=[q], kv_lengths=[kv], nh=a.nh, nkv=a.nkv, hd=a.hd, bs=len(q),
                    attention_type=a.attn_type)
    keys = NEEDS[a.op] if a.op != "gemm" and a.op != "gemm_fp8" else ("M", "N", "K")
    miss = [k for k in keys if getattr(a, k) is None]
    if miss:
        ap.error(f"--op {a.op} needs " + " ".join("--" + k for k in keys) + f" (missing {miss})")
    return {k: getattr(a, k) for k in keys}


def main():
    ap = parser()
    a = ap.parse_args()
    if not a.op:
        ap.error("--op is required")
    hw = get_hw(a, ap)
    op = OPMAP[a.op]
    p = get_params(a, ap)
    row = build_op(op, {k: ([v] if isinstance(v, (list, tuple)) and k in ("q_lengths", "kv_lengths") and False else v)
                        for k, v in p.items()}, {k: hw[k] for k in ("peak_tc", "peak_fp32", "bw", "sms")})
    r = row.iloc[0]
    warns = []
    ideal = float(r.ideal_s)
    side = "compute-bound" if r.ratio >= 1 else "memory-bound"

    bundle = joblib.load(a.model) if os.path.exists(a.model) else None
    pred = None
    if op in TENSOR_OPS and hw["peak_tc"] <= 0:
        warns.append("This GPU has no tensor cores; the learned model was trained only on tensor-core GPUs, so no "
                     "ML prediction is made. Use predict.py (FP32/FP16 CUDA-core analysis) instead.")
    elif bundle is None:
        warns.append(f"No trained model at {a.model}: run `python fetch_real_data.py && python train_real.py`.")
    else:
        X = row[FEATURES].values
        ys = {q: float(bundle["models"][q].predict(X)[0]) for q in (0.1, 0.5, 0.9)}
        lo, mid, hi = min(ys[0.1], ys[0.5]), ys[0.5], max(ys[0.9], ys[0.5])
        pred = dict(t_med=ideal * 10 ** mid, t_lo=ideal * 10 ** lo, t_hi=ideal * 10 ** hi,
                    eff_med=10 ** -mid, eff_lo=10 ** -hi, eff_hi=10 ** -lo)
        rng = bundle["meta"].get("hw_range_per_op", {}).get(op)
        if rng:
            for key, label in (("log_peak", "compute peak"), ("log_bw", "bandwidth"), ("log_sms", "SM count")):
                v = float(r[key])
                lo_r, hi_r = rng[key]
                if v < lo_r - 0.1 or v > hi_r + 0.1:
                    warns.append(f"{label} is outside the range of the training GPUs "
                                 f"[{10**lo_r:.3g}, {10**hi_r:.3g}] for this op: extrapolating, expect larger error.")
    if r.ratio > 1 / 3 and r.ratio < 3:
        warns.append("Within 3x of the ridge: efficiency is lowest and most variable here (see README/summary).")
    if ideal < 20e-6:
        warns.append("Ideal time < 20 us: fixed launch/latency costs dominate; Roofline is a poor guide.")

    out = dict(hardware=hw, op=a.op, params={k: v for k, v in p.items()},
               roofline=dict(flops=float(r.flops), bytes=float(r.bytes), ai=float(r.ai), ridge=float(r.ridge),
                             ratio=float(r.ratio), side=side, ideal_s=ideal),
               prediction=pred, warnings=warns)
    if a.json:
        print(json.dumps(out, indent=2, default=float))
        return
    L = ["=" * 74, f" REAL-DATA PREDICTION   {a.op} {json.dumps({k: v for k, v in p.items() if k not in ('q_lengths','kv_lengths')})}  on {hw['name']}", "=" * 74, "",
         "ROOFLINE (analytic)",
         f"  FLOPs {si(r.flops, 'FLOP')} | bytes {si(r.bytes, 'B')} | AI {r.ai:.3g} FLOP/B | ridge {r.ridge:.3g} "
         f"| AI/ridge {r.ratio:.3g}",
         f"  Roofline side: {side.upper()} | ideal time {tfmt(ideal)} (assumes 100% of peak, no overheads)"]
    L += ["", "LEARNED PREDICTION (trained on real measurements)"]
    if pred:
        L += [f"  predicted time  {tfmt(pred['t_med'])}   (80% interval {tfmt(pred['t_lo'])} - {tfmt(pred['t_hi'])})",
              f"  predicted efficiency  {pred['eff_med']*100:.0f}% of Roofline   "
              f"(80% interval {pred['eff_lo']*100:.0f}% - {pred['eff_hi']*100:.0f}%)"]
        L.append("  note: intervals are only calibrated for new shapes on known GPUs; on unseen hardware they cover far "
                 "less than 80% (see validation below) - treat them as a floor on the uncertainty.")
        e = pred["eff_med"]
        if e > 1.3:
            L.append("  -> faster than the analytic ideal: real DRAM traffic is below the assumed bytes (data stays "
                     "in L2 cache) or this op's FLOP/byte formula is approximate; Roofline over-estimates the time")
        elif e < 0.3:
            L.append("  -> far below Roofline: overheads (launch, small tiles, tail effects) dominate, not the ceilings")
        elif e < 0.65:
            L.append("  -> well below Roofline: real implementation loses a lot to non-ideal effects")
        else:
            L.append("  -> close to Roofline: the ceiling is a good guide here")
        cv = [c for c in bundle["meta"].get("cv", []) if c["op"] == op and c["method"] == "GradBoost"]
        base = {(c["split"]): c for c in bundle["meta"].get("cv", []) if c["op"] == op and c["method"] == "Roofline"}
        if cv:
            L += ["", "VALIDATION FOR THIS OP (cross-validated; median relative time error, lower is better)"]
            for c in sorted(cv, key=lambda c: ["unseen_arch", "unseen_gpu", "unseen_shape", "random(leaky)"].index(c["split"])):
                b = base.get(c["split"])
                cov = f", 80% interval covered {c['cover80']*100:.0f}% of truths" if c.get("cover80") is not None else ""
                L.append(f"  {c['split']:<14} model {c['mape_median']*100:.0f}% (within 25%: {c['within25']*100:.0f}%)"
                         + (f" vs plain Roofline {b['mape_median']*100:.0f}%" if b else "") + cov)
        L += ["", f"  trained on {bundle['meta']['n_rows']:,} measured kernels, GPUs: {', '.join(bundle['meta']['gpus'])}"]
    else:
        L.append("  (not available)")
    L += ["", "FORMULA ASSUMPTIONS FOR THIS OP", f"  {OP_NOTES[op]}"]
    if warns:
        L += ["", "WARNINGS"] + [f"  - {w}" for w in warns]
    print("\n".join(L))


if __name__ == "__main__":
    main()
