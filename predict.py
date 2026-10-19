"""Give hardware specs + a workload, get a compute/memory-bound verdict, statistics and advice.

HARDWARE - pick one:
  --gpu GTX1650Ti                       built-in table (see --list-gpus)
  --hw-json hw/my_gpu.json              written by detect_specs.py (measured ceilings)
  or raw specs: --fp32-tflops X --bw-gbs X --l2-mb X --sms N --mem-gb X
                [--fp16-tflops X] [--tensor-tflops X] [--hw-name NAME]

WORKLOAD - pick one:
  --op <name> with its size flags (see --list-ops), --precision fp32|fp16
  or --op custom --flops F --bytes Q [--ws W --irregular --tensor-eligible]   (any kernel you profiled by hand)

Examples:
  python predict.py --gpu GTX1650Ti --op gemm --M 4096 --N 4096 --K 4096
  python predict.py --gpu GTX1650Ti --op gemm --M 1 --N 8192 --K 8192 --precision fp16   # LLM decode GEMV
  python predict.py --hw-json hw/my_gpu.json --op softmax --rows 98304 --length 2048
  python predict.py --fp32-tflops 3 --bw-gbs 192 --l2-mb 1 --sms 16 --mem-gb 4 --op conv \
         --N 32 --C 256 --H 14 --K 256 --R 3
Add --json for machine-readable output.
"""
import argparse
import json
import os
import sys

from rooflinelab.advisor import analyze, format_report
from rooflinelab.hw_specs import HARDWARE, custom_hw
from rooflinelab.workloads import OPS, make_workload

REQUIRED = {
    "gemm": ["M", "N", "K"], "conv": ["N", "C", "H", "K", "R"], "depthwise": ["N", "C", "H", "R"],
    "attn_scores": ["B", "heads", "L", "d"], "softmax": ["rows", "length"], "layernorm": ["tokens", "H"],
    "add": ["n"], "gelu": ["n"], "silu": ["n"], "sum": ["n"], "gather": ["rows", "D"],
}
OPTIONAL = {"gemm": ["batch"], "conv": ["stride"]}
SIZE_FLAGS = ["M", "N", "K", "batch", "C", "H", "R", "stride", "B", "heads", "L", "d", "rows", "length",
              "tokens", "n", "D"]


def _int(s):
    return int(float(s))


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu")
    ap.add_argument("--hw-json")
    ap.add_argument("--hw-name", default="custom")
    for f in ("fp32-tflops", "fp16-tflops", "tensor-tflops", "bw-gbs", "l2-mb", "mem-gb"):
        ap.add_argument("--" + f, type=float)
    ap.add_argument("--sms", type=int)
    ap.add_argument("--op")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    for f in SIZE_FLAGS:
        ap.add_argument("--" + f, type=_int)
    ap.add_argument("--flops", type=float)
    ap.add_argument("--bytes", type=float)
    ap.add_argument("--ws", type=float)
    ap.add_argument("--irregular", action="store_true")
    ap.add_argument("--tensor-eligible", action="store_true")
    ap.add_argument("--model", default="models/model.joblib")
    ap.add_argument("--no-ml", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list-gpus", action="store_true")
    ap.add_argument("--list-ops", action="store_true")
    return ap


def get_hw(a, ap):
    if a.gpu:
        if a.gpu not in HARDWARE:
            ap.error(f"unknown --gpu '{a.gpu}'. Known: {', '.join(HARDWARE)}")
        return HARDWARE[a.gpu]
    if a.hw_json:
        d = json.load(open(a.hw_json))
        return custom_hw(d["fp32_tflops"], d["bw_gbs"], d["l2_mb"], d["sms"], d["mem_gb"],
                         d.get("fp16_vec_tflops"), d.get("fp16_tensor_tflops", 0.0),
                         d.get("name", "custom"), d.get("arch", "custom"))
    need = {"--fp32-tflops": a.fp32_tflops, "--bw-gbs": a.bw_gbs, "--l2-mb": a.l2_mb,
            "--sms": a.sms, "--mem-gb": a.mem_gb}
    miss = [k for k, v in need.items() if v is None]
    if miss:
        ap.error("hardware not specified. Use --gpu, --hw-json, or give all of: " + ", ".join(need)
                 + f"  (missing: {', '.join(miss)})")
    return custom_hw(a.fp32_tflops, a.bw_gbs, a.l2_mb, a.sms, a.mem_gb, a.fp16_tflops,
                     a.tensor_tflops or 0.0, a.hw_name)


def get_workload(a, ap):
    if not a.op:
        ap.error("no workload: give --op (see --list-ops) or --op custom --flops F --bytes Q")
    if a.op == "custom":
        if a.flops is None or a.bytes is None:
            ap.error("--op custom needs --flops and --bytes")
        ws = a.ws or a.bytes
        return dict(family="custom", desc=f"custom kernel ({a.flops:.3g} FLOP, {a.bytes:.3g} B)",
                    precision=a.precision, flops=a.flops, bytes=a.bytes, ws=ws, reload=0.0,
                    irregular=int(a.irregular), row_bytes=128, tiles=max(1, int(ws // 16384)),
                    min_dim=1024, tensor_eligible=int(a.tensor_eligible), base_eff=0.5, params=None)
    if a.op not in REQUIRED:
        ap.error(f"unknown --op '{a.op}'. Supported: {', '.join(REQUIRED)}, custom")
    miss = [k for k in REQUIRED[a.op] if getattr(a, k) is None]
    if miss:
        ap.error(f"--op {a.op} needs: " + " ".join("--" + k for k in REQUIRED[a.op])
                 + f"  (missing: {', '.join('--' + m for m in miss)})")
    kw = {k: getattr(a, k) for k in REQUIRED[a.op] + OPTIONAL.get(a.op, []) if getattr(a, k) is not None}
    return make_workload(a.op, a.precision, **kw)


def main():
    ap = build_parser()
    a = ap.parse_args()
    if a.list_gpus:
        print(f"{'name':<11}{'FP32':>7}{'FP16v':>7}{'FP16tc':>8}{'BW GB/s':>9}{'L2 MB':>7}{'SMs':>5}{'GB':>5}")
        for h in HARDWARE.values():
            print(f"{h.name:<11}{h.fp32_tflops:>7g}{h.fp16_vec_tflops:>7g}{h.fp16_tensor_tflops:>8g}"
                  f"{h.bw_gbs:>9g}{h.l2_mb:>7g}{h.sms:>5}{h.mem_gb:>5g}")
        return
    if a.list_ops:
        for k, v in OPS.items():
            print(f"  {k:<12}{v}")
        print("  custom      --flops F --bytes Q [--ws W --irregular --tensor-eligible]")
        return
    hw = get_hw(a, ap)
    w = get_workload(a, ap)

    bundle = None
    if not a.no_ml:
        if os.path.exists(a.model):
            import joblib
            bundle = joblib.load(a.model)
        else:
            print(f"[note] no trained model at {a.model}; run `python train.py` for the ML verdict.\n",
                  file=sys.stderr)
    r = analyze(bundle, w, hw)
    if a.json:
        print(json.dumps(dict(r, model_meta=bundle["meta"] if bundle else None), indent=2, default=float))
    else:
        print(format_report(r, bundle["meta"] if bundle else None))


if __name__ == "__main__":
    main()
