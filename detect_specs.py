"""Measure what YOUR GPU actually achieves and write a hardware JSON for predict.py.

Datasheet peaks are rarely reached (clocks, power limits, laptop cooling). Roofline analysis
is only as good as its ceilings, so this measures them with plain PyTorch (no Nsight needed):

    achievable FP32 / FP16 GEMM TFLOPS   (large square matmuls)
    achievable DRAM bandwidth GB/s       (large device-to-device copy)

Usage (needs PyTorch with CUDA):
    python detect_specs.py --out hw/my_gpu.json
    python predict.py --hw-json hw/my_gpu.json --op gemm --M 4096 --N 4096 --K 4096

Not run on hardware by me - if a number looks implausible, compare with the datasheet value
printed alongside it and re-run with the laptop plugged in / in performance mode.
"""
import argparse
import json
import os


def _time(fn, iters=20, warm=5):
    import torch
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(3):  # best of 3 batches rejects thermal/boost noise
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / 1e3 / iters)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hw/my_gpu.json")
    ap.add_argument("--n", type=int, default=4096, help="GEMM size (reduce if you run out of memory)")
    ap.add_argument("--copy-mb", type=int, default=256)
    a = ap.parse_args()

    import torch
    from rooflinelab.hw_specs import HARDWARE
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to PyTorch.")
    props = torch.cuda.get_device_properties(0)
    name = props.name
    key = next((k for k in HARDWARE if k.replace("-", "").upper() in name.replace(" ", "").upper()), None)
    ref = HARDWARE.get(key) if key else None
    print(f"GPU: {name} ({props.multi_processor_count} SMs, {props.total_memory/2**30:.1f} GB)")

    torch.backends.cuda.matmul.allow_tf32 = False
    n = a.n
    res = {}
    for prec, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        x = torch.randn(n, n, device="cuda", dtype=dt)
        y = torch.randn(n, n, device="cuda", dtype=dt)
        t = _time(lambda: x @ y)
        res[prec] = 2 * n**3 / t / 1e12
        print(f"achieved {prec} GEMM: {res[prec]:.2f} TFLOPS")
        del x, y
    nbytes = a.copy_mb * 2**20
    src = torch.empty(nbytes // 4, device="cuda", dtype=torch.float32).normal_()
    dst = torch.empty_like(src)
    t = _time(lambda: dst.copy_(src))
    bw = 2 * nbytes / t / 1e9
    print(f"achieved copy bandwidth: {bw:.0f} GB/s")

    has_tensor_ref = ref is not None and ref.fp16_tensor_tflops > 0
    hw = dict(name=key or name.replace(" ", ""), arch=ref.arch if ref else "custom",
              fp32_tflops=round(res["fp32"], 3),
              bw_gbs=round(bw, 1), sms=props.multi_processor_count,
              mem_gb=round(props.total_memory / 2**30, 1),
              l2_mb=getattr(props, "L2_cache_size", 0) / 2**20 or (ref.l2_mb if ref else 0))
    if ref is None:
        print("WARNING: GPU not in the built-in table: fp16_tensor_tflops=0 and l2_mb may be wrong; "
              "edit the JSON if your GPU has tensor cores / a different L2.")
    if has_tensor_ref:
        hw["fp16_tensor_tflops"] = round(res["fp16"], 3)
        hw["fp16_vec_tflops"] = ref.fp16_vec_tflops
    else:
        hw["fp16_tensor_tflops"] = 0.0
        hw["fp16_vec_tflops"] = round(res["fp16"], 3)
    if hw["l2_mb"] == 0:
        hw["l2_mb"] = 1.0
        print("WARNING: could not read L2 size, using 1 MB - edit the JSON.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(hw, f, indent=2)
    if ref:
        print(f"datasheet(table) fp32 {ref.fp32_tflops} TFLOPS, bw {ref.bw_gbs} GB/s  ->  "
              f"achieved {hw['fp32_tflops']} / {hw['bw_gbs']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
