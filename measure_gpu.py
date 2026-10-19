"""Collect REAL bottleneck labels on an NVIDIA GPU with Nsight Compute (ncu).

Writes a CSV with the same schema as the simulator, so `python run_all.py --data <csv>` works.

Why ncu and not just timing: with only a kernel time T, the utilisations F/(T*P) and Q/(T*B)
have ratio exactly AI/ridge, i.e. the Roofline rule itself - labels would be circular. ncu's
"speed of light" counters measure what the hardware actually did (real DRAM traffic after
cache effects, real pipe utilisation):

    r = sm__throughput % / gpu__dram_throughput %        (compute-side / memory-side)

and the label bands are the same as the simulator's (labels_from_ratio in features.py).
The dominant kernel (longest duration) of the op is used.

STATUS: written without access to a GPU - not yet run on hardware. Expect to adjust metric
names for your ncu version (`ncu --query-metrics`) and to eyeball a few rows against the
Nsight Compute GUI before trusting a full sweep.

Usage (needs PyTorch + CUDA + ncu on PATH, and permission to read GPU counters):
    python measure_gpu.py --out data/measured_a100.csv
    python measure_gpu.py --list                       # dry run, no GPU needed
    python measure_gpu.py --out x.csv --families gemm,softmax --max-configs 40
    python measure_gpu.py --cache-control none ...     # keep L2 warm like a benchmark loop
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys

M_SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_DRAM = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
M_TIME = "gpu__time_duration.sum"
M_BYTES = "dram__bytes.sum"
METRICS = ",".join([M_SM, M_DRAM, M_TIME, M_BYTES])


# ---------------------------------------------------------------- op construction (torch)
def build_op(params, prec, device="cuda"):
    import torch
    import torch.nn.functional as F
    dt = torch.float16 if prec == "fp16" else torch.float32
    p = params
    op = p["op"]
    r = lambda *s: torch.randn(*s, device=device, dtype=dt)
    if op == "gemm":
        if p["batch"] > 1:
            a, b = r(p["batch"], p["M"], p["K"]), r(p["batch"], p["K"], p["N"])
            return lambda: torch.bmm(a, b)
        a, b = r(p["M"], p["K"]), r(p["K"], p["N"])
        return lambda: a @ b
    if op == "conv":
        x, w = r(p["N"], p["C"], p["H"], p["H"]), r(p["K"], p["C"], p["R"], p["R"])
        pad = p["R"] // 2
        return lambda: F.conv2d(x, w, stride=p["stride"], padding=pad)
    if op == "depthwise":
        x, w = r(p["N"], p["C"], p["H"], p["H"]), r(p["C"], 1, p["R"], p["R"])
        return lambda: F.conv2d(x, w, padding=p["R"] // 2, groups=p["C"])
    if op == "attn_scores":
        bh = p["B"] * p["heads"]
        q, k = r(bh, p["L"], p["d"]), r(bh, p["d"], p["L"])
        return lambda: torch.bmm(q, k)
    if op == "softmax":
        x = r(p["rows"], p["length"])
        return lambda: torch.softmax(x, dim=-1)
    if op == "layernorm":
        x = r(p["tokens"], p["H"])
        return lambda: F.layer_norm(x, (p["H"],))
    if op in ("add", "gelu", "silu"):
        x, y = r(p["n"]), r(p["n"])
        return {"add": lambda: x + y, "gelu": lambda: F.gelu(x), "silu": lambda: F.silu(x)}[op]
    if op == "sum":
        x = r(p["n"])
        return lambda: x.sum()
    if op == "gather":
        table = r(max(p["rows"] * 4, 1024), p["D"])  # larger than the touched set
        idx = torch.randint(0, table.shape[0], (p["rows"],), device=device)
        return lambda: table.index_select(0, idx)
    raise ValueError(op)


def single_mode(spec):
    """Runs one op under ncu; only the single timed call is profiled."""
    import torch
    fn = build_op(spec["params"], spec["precision"])
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


# ---------------------------------------------------------------- ncu driver / parsing
def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except ValueError:
        return float("nan")


def parse_ncu_csv(text):
    """-> dict(sm_pct, dram_pct, time_s, dram_bytes) of the longest kernel."""
    lines = [l for l in text.splitlines() if l.startswith('"')]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    kernels = {}
    for r in rows:
        k = kernels.setdefault(r["ID"], {})
        k[r["Metric Name"]] = _num(r["Metric Value"])
    if not kernels:
        return None
    best = max(kernels.values(), key=lambda k: k.get(M_TIME, 0.0))
    return dict(sm_pct=best.get(M_SM, float("nan")), dram_pct=best.get(M_DRAM, float("nan")),
                time_s=best.get(M_TIME, float("nan")) * 1e-9, dram_bytes=best.get(M_BYTES, float("nan")))


def run_ncu(spec, cache_control):
    cmd = ["ncu", "--csv", "--print-units", "base", "--profile-from-start", "off",
           "--cache-control", cache_control, "--metrics", METRICS,
           sys.executable, os.path.abspath(__file__), "--single", json.dumps(spec)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-500:] or out.stdout[-500:])
    return parse_ncu_csv(out.stdout)


# ---------------------------------------------------------------- hardware detection
def detect_hw(name_override=None):
    from rooflinelab.hw_specs import HARDWARE
    if name_override:
        return HARDWARE[name_override]
    import torch
    dev = torch.cuda.get_device_name(0).replace(" ", "").upper()
    for key, hw in HARDWARE.items():
        if key.replace("-", "").upper() in dev:
            return hw
    raise SystemExit(f"Unknown GPU '{dev}': add it to rooflinelab/hw_specs.py or pass --hw-name")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", help=argparse.SUPPRESS)
    ap.add_argument("--out", default="data/measured.csv")
    ap.add_argument("--hw-name")
    ap.add_argument("--families")
    ap.add_argument("--max-configs", type=int, default=0, help="per family, 0 = all")
    ap.add_argument("--cache-control", default="all", choices=["all", "none"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-mem-frac", type=float, default=0.3,
                    help="skip workloads whose working set exceeds this fraction of GPU memory")
    a = ap.parse_args()

    if a.single:
        return single_mode(json.loads(a.single))

    import pandas as pd
    from rooflinelab.workloads import generate_workloads
    from rooflinelab.features import featurize, labels_from_ratio

    ws = generate_workloads(0)
    if a.families:
        keep = set(a.families.split(","))
        ws = [w for w in ws if w["family"] in keep]
    if a.max_configs:
        seen, out = {}, []
        for w in ws:
            seen[w["family"]] = seen.get(w["family"], 0) + 1
            if seen[w["family"]] <= a.max_configs:
                out.append(w)
        ws = out
    if a.list:
        print(f"{len(ws)} workloads would be profiled")
        return

    hw = detect_hw(a.hw_name)
    # leave headroom: inputs, outputs and cuDNN/cuBLAS workspaces must all fit (matters on a 4 GB card)
    limit = a.max_mem_frac * hw.mem_gb * 2**30
    n_before = len(ws)
    ws = [w for w in ws if w["ws"] <= limit]
    print(f"GPU: {hw.name}; profiling {len(ws)} workloads ({n_before - len(ws)} skipped: too big for "
          f"{hw.mem_gb:g} GB at max-mem-frac {a.max_mem_frac})")
    rows = []
    for i, w in enumerate(ws):
        spec = dict(params=w["params"], precision=w["precision"])
        try:
            m = run_ncu(spec, a.cache_control)
        except Exception as e:  # e.g. OOM on a small GPU
            print(f"[{i}] skipped {w['desc']}: {str(e)[:120]}")
            continue
        if not m or not (m["dram_pct"] == m["dram_pct"]):
            continue
        ratio = m["sm_pct"] / max(m["dram_pct"], 0.1)
        row = dict(family=w["family"], desc=w["desc"], hw=hw.name, precision=w["precision"],
                   flops=w["flops"], bytes=w["bytes"])
        row.update(featurize(w, hw))
        row.update(r_ratio=ratio, time_s=m["time_s"], launch_frac=min(1.0, 4e-6 / max(m["time_s"], 1e-9)),
                   ach_gflops=w["flops"] / m["time_s"] / 1e9, ach_gbs=m["dram_bytes"] / m["time_s"] / 1e9,
                   sm_pct=m["sm_pct"], dram_pct=m["dram_pct"])
        rows.append(row)
        if i % 25 == 0:
            print(f"[{i}/{len(ws)}] {w['desc']} {w['precision']}: r={ratio:.2f}")
    df = pd.DataFrame(rows)
    df["y_bin"], df["y_3"] = labels_from_ratio(df["r_ratio"].values)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out}\nRun this on several GPUs, concatenate the CSVs, "
          f"then: python run_all.py --data <combined.csv>")


if __name__ == "__main__":
    main()
