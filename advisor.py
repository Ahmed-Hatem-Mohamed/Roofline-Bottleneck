"""Roofline analysis + ML verdict + what-if sensitivity + optimisation advice.

Everything labelled 'roofline' is analytic and hardware-agnostic. The ML verdict is only as
good as the labels the model was trained on (see meta['label_source'] in the model file).
Speed-up figures are Roofline UPPER BOUNDS (ideal times), never promises.
"""
import dataclasses
import math

import numpy as np

from .features import featurize
from .hw_specs import HARDWARE, effective_peak
from .simulate import LAUNCH_S, simulate_one
from .workloads import make_workload

CLASS = {0: "memory-bound", 1: "mixed", 2: "compute-bound"}
NEAR_RIDGE = 3.0
LAUNCH_DOMINATED_S = 2 * LAUNCH_S  # launch overhead > half the runtime (matches training filter)
MIXED_BAND = 1.5


# ----------------------------------------------------------------------------- analytics
def roofline(w, hw):
    P, tc = effective_peak(hw, w["precision"], bool(w["tensor_eligible"]))
    B = hw.bw_gbs * 1e9
    ai, ridge = w["flops"] / w["bytes"], P / B
    t_c, t_m = w["flops"] / P, w["bytes"] / B
    ratio = ai / ridge
    band = "mixed" if 1 / MIXED_BAND < ratio < MIXED_BAND else ("compute-bound" if ratio >= 1 else "memory-bound")
    return dict(peak_flops=P, used_tensor_cores=tc, bandwidth=B, ai=ai, ridge=ridge, ratio=ratio,
                t_compute=t_c, t_memory=t_m, t_ideal=max(t_c, t_m), attainable_flops=min(P, ai * B),
                frac_of_peak=min(1.0, ratio), bound="compute-bound" if ratio >= 1 else "memory-bound",
                bound3=band, near_ridge=bool(1 / NEAR_RIDGE < ratio < NEAR_RIDGE),
                launch_dominated=bool(max(t_c, t_m) < LAUNCH_DOMINATED_S))


def ml_verdict(bundle, w, hw):
    if bundle is None:
        return None
    f = featurize(w, hw)
    x = np.array([[f[k] for k in bundle["features"]]])
    m = bundle["model"]
    p = m.predict_proba(x)[0]
    probs = {CLASS[int(c)]: float(pp) for c, pp in zip(m.classes_, p)}
    for name in CLASS.values():
        probs.setdefault(name, 0.0)
    label = max(probs, key=probs.get)
    top2 = sorted(probs.values())[-2:]
    return dict(label=label, probs=probs, confidence=probs[label], margin=top2[1] - top2[0])


def diagnostics(w, hw):
    """Heuristic structural effects (noise-free simulator formulas, NOT measurements)."""
    d = simulate_one(w, hw, None)
    return dict(l2_cover=d["l2_cover"], wave_eff=d["wave_eff"], dim_pen=d["dim_pen"],
                dram_traffic_est=d["q_dram"], dram_vs_compulsory=d["q_dram"] / w["bytes"],
                est_time_s=d["time_s"], est_gflops=d["ach_gflops"],
                est_compute_eff=d["eff_c"], est_mem_eff=d["eff_m"])


def hw_out_of_range(hw):
    notes = []
    tab = list(HARDWARE.values())
    checks = {"fp32_tflops": "FP32 peak", "bw_gbs": "bandwidth", "l2_mb": "L2 size", "sms": "SM count",
              "mem_gb": "memory capacity"}
    for k, label in checks.items():
        v = getattr(hw, k)
        lo, hi = min(getattr(h, k) for h in tab), max(getattr(h, k) for h in tab)
        if v < lo / 2 or v > hi * 2:
            notes.append(f"{label} ({v:g}) is far outside the training range [{lo:g}, {hi:g}]: ML output unreliable")
    return notes


# ----------------------------------------------------------------------------- what-if
def _scaled(hw, **mult):
    return dataclasses.replace(hw, **{k: getattr(hw, k) * v for k, v in mult.items()})


def _params(w):
    p = w.get("params")
    return dict(p) if p else None


def _evaluate(bundle, w, hw, base_t):
    rf = roofline(w, hw)
    ml = ml_verdict(bundle, w, hw)
    return dict(roofline=rf["bound3"], ratio=rf["ratio"], t_ideal=rf["t_ideal"],
                ml=(ml["label"] if ml else None), ml_conf=(ml["confidence"] if ml else None),
                speedup_ub=base_t / rf["t_ideal"])


def what_ifs(bundle, w, hw):
    base_t = roofline(w, hw)["t_ideal"]
    out = {}
    out["2x memory bandwidth"] = _evaluate(bundle, w, _scaled(hw, bw_gbs=2), base_t)
    out["2x compute peak"] = _evaluate(
        bundle, w, _scaled(hw, fp32_tflops=2, fp16_vec_tflops=2, fp16_tensor_tflops=2), base_t)
    out["4x L2 cache"] = _evaluate(bundle, w, _scaled(hw, l2_mb=4), base_t)
    p = _params(w)
    if p:
        op = p.pop("op")
        try:
            if w["precision"] == "fp32":
                out["fp16 instead of fp32"] = _evaluate(bundle, make_workload(op, "fp16", **p), hw, base_t)
            if op == "gemm":
                q = dict(p, M=p["M"] * 8)
                w8 = make_workload(op, w["precision"], **q)
                e = _evaluate(bundle, w8, hw, base_t)
                e["speedup_ub"] = None  # different amount of work; speed-up per item is not comparable
                e["per_item_ub"] = base_t * 8 / roofline(w8, hw)["t_ideal"]
                out["8x larger M (batch/tokens)"] = e
            if op == "conv":
                w8 = make_workload(op, w["precision"], **dict(p, N=p["N"] * 8))
                e = _evaluate(bundle, w8, hw, base_t)
                e["speedup_ub"] = None
                e["per_item_ub"] = base_t * 8 / roofline(w8, hw)["t_ideal"]
                out["8x larger batch"] = e
        except Exception:
            pass
    if w["family"] in ("softmax", "layernorm", "elementwise", "depthwise_conv", "custom"):
        wf = dict(w, bytes=w["bytes"] * 0.5, ws=w["ws"] * 0.5)
        out["fuse with neighbour op (halves DRAM traffic)"] = _evaluate(bundle, wf, hw, base_t)
    return out


def ridge_crossing_M(w, hw):
    """Smallest M (power of two) at which this GEMM becomes compute-bound by Roofline."""
    p = _params(w)
    if not p or p.get("op") != "gemm":
        return None
    p.pop("op")
    for k in range(0, 22):
        q = dict(p, M=2 ** k)
        if roofline(make_workload("gemm", w["precision"], **q), hw)["ratio"] >= 1:
            return 2 ** k
    return None


# ----------------------------------------------------------------------------- advice
def advise(w, hw, rf, verdict, diag, wi, bundle_present):
    adv = []

    def add(title, detail, ub=None):
        adv.append(dict(title=title, detail=detail, speedup_ub=ub))

    label = verdict["label"] if verdict else rf["bound3"]
    p = _params(w)
    op = p["op"] if p else None
    fam = w["family"]

    if rf["launch_dominated"]:
        add("Kernel is launch/latency-bound, not limited by either ceiling",
            f"Ideal runtime is {rf['t_ideal']*1e6:.1f} us versus ~{LAUNCH_S*1e6:.0f} us of launch overhead. "
            "Batch many small ops into one launch, fuse neighbours, or use CUDA Graphs.")

    if label == "memory-bound":
        if w["precision"] == "fp32" and "fp16 instead of fp32" in wi:
            e = wi["fp16 instead of fp32"]
            add("Use FP16/BF16 (halves bytes moved)",
                f"Bytes halve, AI doubles. New Roofline bound: {e['roofline']}"
                + (f", ML: {e['ml']}" if e["ml"] else "") + ".", e["speedup_ub"])
        if "fuse with neighbour op (halves DRAM traffic)" in wi:
            e = wi["fuse with neighbour op (halves DRAM traffic)"]
            add("Fuse with the adjacent elementwise/normalisation op (torch.compile, custom fused kernel)",
                "An isolated streaming op reads and writes DRAM once; fusing removes the intermediate "
                "round trip (modelled as halving traffic).", e["speedup_ub"])
        if op in ("gemm", "conv"):
            cross = ridge_crossing_M(w, hw) if op == "gemm" else None
            key = "8x larger M (batch/tokens)" if op == "gemm" else "8x larger batch"
            e = wi.get(key)
            txt = ("Weights/activations are re-read per call; more work per byte raises arithmetic intensity. "
                   f"With 8x the batch, Roofline says {e['roofline']}." if e else "")
            if cross:
                txt += f" Roofline crossover to compute-bound at M ~ {cross} (this precision/hardware)."
            add("Increase batch / tokens per launch", txt.strip(), e.get("per_item_ub") if e else None)
        if fam == "embedding_gather" or w["irregular"]:
            rb = w["row_bytes"]
            if rb < 128:
                add("Improve access coalescing for the gather",
                    f"Each row is only {rb} B but a DRAM burst is 128 B, so most of every transaction is "
                    "wasted. Sort/bucket indices, process several rows per warp, or pad/pack rows to >=128 B.")
            else:
                add("Gather is bandwidth-limited by random row access",
                    f"Rows are {rb} B (>= one 128 B burst), so coalescing within a row is fine; remaining "
                    "levers are sorting indices for locality, caching hot rows (shared memory/L2), and "
                    "FP16/INT8 tables.")
        if op == "gemm" and p.get("M", 99) <= 64:
            b = 4 if w["precision"] == "fp32" else 2
            wbytes = p["K"] * p["N"] * b * p.get("batch", 1)
            if wbytes / w["bytes"] > 0.8:
                for q, name in ((1, "INT8"), (0.5, "INT4")):
                    new = w["bytes"] - wbytes * (1 - q / b)
                    add(f"Weight-only quantisation to {name}",
                        f"Weights are {wbytes/w['bytes']*100:.0f}% of traffic at M={p['M']}; "
                        f"{name} weights cut bytes to {new/w['bytes']*100:.0f}%. Needs dequant kernels "
                        "(e.g. bitsandbytes, GPTQ/AWQ, Marlin) - check your GPU's support.", w["bytes"] / new)
        if op in ("gemm", "conv", "attn_scores") and diag["l2_cover"] < 1 and w["min_dim"] >= 64:
            add("Block/tile for L2 reuse",
                f"Working set {w['ws']/2**20:.1f} MiB exceeds L2 ({hw.l2_mb:g} MiB). Use L2-aware thread-block "
                "swizzling / larger tiles, or split K, so operand tiles are reused from cache.")
        if diag["l2_cover"] >= 1:
            add("Working set fits in L2 - keep it there",
                "Repeated calls can hit L2; avoid evicting it (fuse consumers right after, don't interleave "
                "large streams).")
        if fam in ("softmax",) or op == "attn_scores":
            add("Use a fused attention kernel (FlashAttention-style)",
                "Materialising the L x L score matrix in DRAM dominates traffic; fusing QK^T, softmax and "
                "PV keeps it on chip.")
        add("Use vectorised, aligned accesses (float4 / 128-bit loads) in custom kernels",
            "Needed to reach the achievable bandwidth ceiling; run detect_specs.py to see what your GPU reaches.")

    elif label == "compute-bound":
        if w["precision"] == "fp32" and "fp16 instead of fp32" in wi:
            e = wi["fp16 instead of fp32"]
            what = ("tensor cores" if hw.fp16_tensor_tflops > 0 and w["tensor_eligible"]
                    else f"{hw.fp16_vec_tflops/hw.fp32_tflops:.1f}x FP16 CUDA-core rate")
            add(f"Switch to FP16/BF16 to use {what}",
                f"Raises the compute ceiling; new Roofline bound: {e['roofline']}"
                + (f", ML: {e['ml']}" if e["ml"] else "") + ".", e["speedup_ub"])
        elif w["precision"] == "fp16" and hw.fp16_tensor_tflops > 0 and not rf["used_tensor_cores"]:
            add("Make the kernel tensor-core eligible",
                "Use cuBLAS/cuDNN paths with dimensions multiple of 8 (64+ ideal) and matching dtypes.")
        if diag["wave_eff"] < 0.85:
            add("Fix wave quantisation (tail effect)",
                f"{w['tiles']} tiles over {hw.sms} SMs leaves only {diag['wave_eff']*100:.0f}% of the last wave "
                "busy. Change tile size or problem split so tile count is a multiple of the SM count.",
                1 / max(diag["wave_eff"], 1e-3))
        if diag["dim_pen"] < 0.9:
            add("Small dimension limits utilisation",
                f"Smallest dim is {w['min_dim']}; pad to multiples of 8/16/64 or batch several problems "
                "(grouped/batched GEMM).")
        if fam == "conv3x3":
            add("Try Winograd / FFT convolution algorithms",
                "Reduces multiplications for 3x3 kernels (cuDNN benchmark mode picks automatically).")
        add("Reduce FLOPs algorithmically",
            "Pruning/sparsity, low-rank factorisation, smaller sequence length, or skipping redundant work "
            "- only these help once you are at the compute ceiling.")

    else:  # mixed
        cap = max(rf["t_compute"], rf["t_memory"]) / max(min(rf["t_compute"], rf["t_memory"]), 1e-30)
        add("Both ceilings are close - optimise both sides",
            f"Compute and memory ideal times differ only {cap:.2f}x, so fixing one side alone caps the gain "
            f"near {cap:.2f}x. Reduce bytes (FP16, fusion) AND raise compute efficiency.", cap)
        if w["precision"] == "fp32" and "fp16 instead of fp32" in wi:
            e = wi["fp16 instead of fp32"]
            add("Use FP16/BF16", f"Halves bytes and raises compute peak; Roofline then says {e['roofline']}.",
                e["speedup_ub"])

    adv.sort(key=lambda a: -(a["speedup_ub"] or 0))
    return adv


# ----------------------------------------------------------------------------- driver
def analyze(bundle, w, hw):
    rf = roofline(w, hw)
    ml = ml_verdict(bundle, w, hw)
    diag = diagnostics(w, hw)
    wi = what_ifs(bundle, w, hw)
    final = ml["label"] if ml else rf["bound3"]
    notes = []
    if ml and (final == "compute-bound") != (rf["bound"] == "compute-bound") and final != "mixed":
        why = []
        if final == "compute-bound":  # Roofline said memory-bound
            if diag["dram_vs_compulsory"] < 0.7:
                why.append(f"real DRAM traffic is estimated at {diag['dram_vs_compulsory']*100:.0f}% of the "
                           "compulsory bytes (L2 reuse), so Roofline overestimates memory time")
            if diag["wave_eff"] < 0.85 or diag["dim_pen"] < 0.9:
                why.append("tile/wave/dimension effects cut compute efficiency, so Roofline underestimates "
                           "compute time")
        else:  # Roofline said compute-bound
            if diag["dram_vs_compulsory"] > 1.05:
                why.append(f"tile-reload traffic raises DRAM bytes to {diag['dram_vs_compulsory']*100:.0f}% of "
                           "compulsory, so Roofline underestimates memory time")
            if w["irregular"]:
                why.append("irregular access lowers achievable bandwidth")
            if not why:
                why.append("memory-side efficiency/latency effects the plain ceilings ignore")
        notes.append("ML disagrees with plain Roofline" + (": " + "; ".join(why) if why else "."))
    if w["family"] == "custom":
        notes.append("Custom kernel: tile count, smallest dimension and access regularity are default guesses, "
                     "so the ML verdict is less reliable than for a built-in --op; trust the Roofline section more.")
    if rf["near_ridge"]:
        notes.append(f"Within {NEAR_RIDGE:g}x of the ridge point: neither bottleneck is clear-cut; "
                     "real measurement is the tiebreaker.")
    if rf["launch_dominated"]:
        notes.append("Launch-dominated: the ML model was trained without such kernels; treat its verdict as unreliable.")
    if w["ws"] > hw.mem_gb * 2**30 * 0.8:
        notes.append("Working set may not fit in device memory.")
    notes += hw_out_of_range(hw)
    advice = advise(w, hw, rf, ml, diag, wi, bundle is not None)
    return dict(hardware=dataclasses.asdict(hw), workload={k: w[k] for k in
                ("family", "desc", "precision", "flops", "bytes", "ws", "tiles", "min_dim", "irregular",
                 "tensor_eligible")},
                roofline=rf, ml=ml, final=final, diagnostics=diag, what_if=wi, advice=advice, notes=notes)


# ----------------------------------------------------------------------------- text report
def _si(v, unit):
    for div, pre in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= div:
            return f"{v/div:.3g} {pre}{unit}"
    return f"{v:.3g} {unit}"


def _t(s):
    return f"{s*1e6:.2f} us" if s < 1e-3 else f"{s*1e3:.3f} ms"


def format_report(r, meta=None):
    hw, w, rf, ml, dg = r["hardware"], r["workload"], r["roofline"], r["ml"], r["diagnostics"]
    L = []
    bar = "=" * 74
    L += [bar, f" BOTTLENECK REPORT   {w['desc']}  [{w['precision']}]   on  {hw['name']}", bar]
    L += ["", "HARDWARE",
          f"  FP32 {hw['fp32_tflops']:g} TFLOPS | FP16(vec) {hw['fp16_vec_tflops']:g} | "
          f"FP16(tensor) {hw['fp16_tensor_tflops']:g} | BW {hw['bw_gbs']:g} GB/s | "
          f"L2 {hw['l2_mb']:g} MB | {hw['sms']} SMs | {hw['mem_gb']:g} GB"]
    L += ["", "ROOFLINE ANALYSIS (analytic)",
          f"  FLOPs {_si(w['flops'], 'FLOP')} | compulsory bytes {_si(w['bytes'], 'B')} | "
          f"working set {_si(w['ws'], 'B')}",
          f"  arithmetic intensity {rf['ai']:.3g} FLOP/B   ridge point {rf['ridge']:.3g} FLOP/B   "
          f"AI/ridge = {rf['ratio']:.3g}",
          f"  compute ceiling {_si(rf['peak_flops'], 'FLOP/s')}"
          f"{' (tensor cores)' if rf['used_tensor_cores'] else ''} | bandwidth ceiling {_si(rf['bandwidth'], 'B/s')}",
          f"  ideal time: compute {_t(rf['t_compute'])}, memory {_t(rf['t_memory'])} -> {_t(rf['t_ideal'])}",
          f"  attainable {_si(rf['attainable_flops'], 'FLOP/s')} = {rf['frac_of_peak']*100:.1f}% of compute peak",
          f"  Roofline verdict: {rf['bound3'].upper()}"]
    L += ["", "ML VERDICT"]
    if ml:
        pr = ml["probs"]
        L.append(f"  {ml['label'].upper()}  (confidence {ml['confidence']*100:.0f}%)")
        L.append("  P(memory) {:.2f} | P(mixed) {:.2f} | P(compute) {:.2f}".format(
            pr["memory-bound"], pr["mixed"], pr["compute-bound"]))
        if meta:
            L.append(f"  model trained on: {meta.get('label_source', 'unknown')}")
            ev = meta.get("eval", {})
            if ev:
                L.append("  accuracy of this model family on your real rows: " + ", ".join(
                    f"{k}={v:.3f}" for k, v in ev.items()))
            if "SIMULATED only" in meta.get("label_source", ""):
                L.append("  ! Labels are simulated: this verdict reflects my simulator, not measured hardware."
                         " Collect real data (measure_gpu.py) and retrain (train.py --real ...).")
    else:
        L.append("  (no trained model found - run `python train.py`; showing Roofline only)")
    L += ["", "STRUCTURAL DIAGNOSTICS (heuristic formulas, not measurements)",
          f"  L2 coverage of working set {min(dg['l2_cover'],1)*100:.0f}% | est. DRAM traffic "
          f"{dg['dram_vs_compulsory']*100:.0f}% of compulsory | wave efficiency {dg['wave_eff']*100:.0f}% | "
          f"dimension penalty factor {dg['dim_pen']:.2f}",
          f"  heuristic runtime estimate {_t(dg['est_time_s'])} (~{dg['est_gflops']:.0f} GFLOP/s)"]
    L += ["", "WHAT-IF (Roofline upper bounds on speed-up)",
          f"  {'change':<46}{'Roofline':<15}{'ML':<15}{'speed-up<=':<10}"]
    for k, e in r["what_if"].items():
        sp = e["speedup_ub"]
        sp_s = f"{sp:.2f}x" if sp else (f"{e['per_item_ub']:.2f}x/item" if e.get("per_item_ub") else "-")
        L.append(f"  {k:<46}{e['roofline']:<15}{(e['ml'] or '-'):<15}{sp_s:<10}")
    L += ["", f"OPTIMISATIONS (for a {r['final'].upper()} kernel, ranked by upper-bound gain)"]
    for i, a in enumerate(r["advice"], 1):
        ub = f"  [<= {a['speedup_ub']:.2f}x]" if a["speedup_ub"] else ""
        L.append(f"  {i}. {a['title']}{ub}")
        L.append(f"     {a['detail']}")
    if r["notes"]:
        L += ["", "NOTES"] + [f"  - {n}" for n in r["notes"]]
    return "\n".join(L)
