"""Hardware database.

Numbers are approximate vendor-datasheet peak values (dense, no sparsity), entered from
memory. VERIFY against the official datasheets before citing them in a paper or report.
Tensor-core numbers assume FP16 inputs with FP32 accumulate where the vendor distinguishes.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class HW:
    name: str
    arch: str
    fp32_tflops: float      # CUDA-core FP32
    fp16_vec_tflops: float  # FP16 on CUDA cores (no tensor cores)
    fp16_tensor_tflops: float  # 0 if the chip has no tensor cores
    bw_gbs: float           # DRAM/HBM bandwidth
    l2_mb: float
    sms: int
    mem_gb: float


HARDWARE = {h.name: h for h in [
    HW("P100",     "Pascal", 9.3,  18.7,  0.0,   732,  4,   56,  16),
    HW("T4",       "Turing", 8.1,  16.2,  65.0,  320,  4,   40,  16),
    HW("V100",     "Volta",  15.7, 31.4,  125.0, 900,  6,   80,  32),
    HW("A10",      "Ampere", 31.2, 31.2,  125.0, 600,  6,   72,  24),
    HW("RTX3090",  "Ampere", 35.6, 35.6,  71.0,  936,  6,   82,  24),
    HW("A100-40",  "Ampere", 19.5, 78.0,  312.0, 1555, 40,  108, 40),
    HW("A100-80",  "Ampere", 19.5, 78.0,  312.0, 2039, 40,  108, 80),
    HW("L4",       "Ada",    30.3, 30.3,  121.0, 300,  48,  58,  24),
    HW("RTX4090",  "Ada",    82.6, 82.6,  165.0, 1008, 72,  128, 24),
    HW("H100",     "Hopper", 67.0, 133.8, 989.0, 3350, 50,  132, 80),
    # Laptop GTX 1650 Ti (TU117): 1024 FP32 cores, FP16 at 2x FP32 on CUDA cores, NO tensor
    # cores, 128-bit GDDR6 @ 12 Gbps. Clocks vary by laptop (power limit) - run detect_specs.py
    # and edit if your measured numbers differ a lot.
    HW("GTX1650Ti", "Turing", 3.0, 6.0, 0.0, 192, 1, 16, 4),
]}


def custom_hw(fp32_tflops, bw_gbs, l2_mb, sms, mem_gb, fp16_vec_tflops=None,
              fp16_tensor_tflops=0.0, name="custom", arch="custom"):
    """Build a HW from raw specs (the minimum set a user must provide)."""
    return HW(name, arch, float(fp32_tflops),
              float(fp16_vec_tflops if fp16_vec_tflops is not None else fp32_tflops),
              float(fp16_tensor_tflops), float(bw_gbs), float(l2_mb), int(sms), float(mem_gb))


def effective_peak(hw: HW, precision: str, tensor_eligible: bool):
    """Peak FLOP/s the kernel could use on this chip, and whether tensor cores are used.

    The Roofline baseline is deliberately given this (workload-aware) peak: it is the most
    favourable fair version of the analytical model.
    """
    if precision == "fp16":
        if tensor_eligible and hw.fp16_tensor_tflops > 0:
            return hw.fp16_tensor_tflops * 1e12, True
        return hw.fp16_vec_tflops * 1e12, False
    return hw.fp32_tflops * 1e12, False
