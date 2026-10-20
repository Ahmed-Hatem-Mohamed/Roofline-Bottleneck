# Can ML predict compute- vs memory-bound kernels better than Roofline?

Research question: given an AI kernel and a GPU, can a classifier predict whether it is
compute-bound or memory-bound (and optionally "mixed"), and does it capture bottlenecks the
analytical Roofline model misses?

    f(workload features, hardware features) -> {memory-bound, compute-bound [, mixed]}

## Interactive demo (no GPU required)

The fastest way to try the project is the Streamlit interface. It lets you enter a preset or custom architecture, define a supported AI kernel, and compare the analytical Roofline verdict with the simulator-trained ML verdict. It also displays the Roofline plot, structural heuristics, optimization suggestions, and what-if changes such as 2× compute or 2× memory bandwidth.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app trains its small classifier in memory from the built-in simulator on first launch (typically a few seconds on CPU); it does not require a GPU, pretrained model file, or downloaded real-data dataset. The generated `models/*.joblib` files are intentionally not included in the source release; use `python train.py` if you also want to run the command-line classifier. **The ML bottleneck verdict is learned from simulated labels, not measured physical bottleneck labels.** Use it as a working demonstration of the workflow, not as proof that the model predicts real hardware better than Roofline. The what-if speed-ups are idealized Roofline bounds, not performance guarantees.

## The predictor (`predict.py`)

Give it hardware specs and a workload; it prints a Roofline analysis, an ML verdict with
probabilities, what-if sensitivity and ranked optimisations.

    python train.py                                   # once; saves models/model.joblib
    python predict.py --list-gpus                     # built-in hardware
    python predict.py --list-ops                      # supported workloads
    python predict.py --gpu GTX1650Ti --op gemm --M 4096 --N 4096 --K 4096
    python predict.py --gpu GTX1650Ti --op gemm --M 1 --N 8192 --K 8192 --precision fp16   # LLM decode
    python predict.py --gpu GTX1650Ti --op softmax --rows 98304 --length 2048
    python predict.py --fp32-tflops 3 --bw-gbs 192 --l2-mb 1 --sms 16 --mem-gb 4 --op conv --N 32 --C 256 --H 14 --K 256 --R 3
    python predict.py --gpu GTX1650Ti --op custom --flops 2e9 --bytes 4e8   # any kernel you sized by hand
    add --json for machine-readable output

**Hardware specs it needs** (all in the report header): FP32 TFLOPS, memory bandwidth (GB/s),
L2 size (MB), SM count, memory capacity (GB); optional FP16 CUDA-core TFLOPS and FP16 tensor-core
TFLOPS (0 if the GPU has none).

**What it outputs:** FLOPs, bytes, arithmetic intensity, ridge point, ideal compute/memory time,
attainable % of peak, Roofline verdict, ML verdict + P(memory/mixed/compute), structural
diagnostics (L2 coverage, wave efficiency), what-if table (2x BW, 2x compute, 4x L2, FP16, larger
batch, fusion) and ranked optimisations with Roofline upper-bound speed-ups (never promises).

**Limits you must know:** only the 11 op types in `--list-ops` (or `custom`) are modelled, at kernel
granularity, not whole networks; the ML verdict is only as real as the training labels (below);
inputs far outside the training hardware range trigger a warning.

## The real-data model (`predict_real.py`) - trained on measurements, not a simulator

Data: [PipeWeave](https://github.com/zksainx/pipeweave) measured kernel latencies (arXiv 2601.14910;
repo is Apache-2.0, **the dataset's own license is not stated - cite the paper and check with the authors
before redistributing**). 452,120 kernels on 11 NVIDIA GPUs (A100, A40, H100, H20, H200, H800, L20, L40,
RTX 6000 Ada, RTX A6000, RTX PRO 6000): BF16 GEMM, FP8 GEMM, attention (FA2/FA3), RMSNorm, SiLU*Mul, fused-MoE.
`gemm_train.csv` (131 MB, Git LFS) could not be downloaded in my environment; `fetch_real_data.py` pulls it if
you have git-lfs, and `train_real.py` then uses it automatically.

    python fetch_real_data.py      # downloads the CSVs to data/real/ (git-ignored)
    python train_real.py           # ~12 min on a 2-core machine; saves models/real_efficiency.joblib
    python make_real_figs.py
    python predict_real.py --gpu H100 --op gemm --M 4096 --N 4096 --K 4096

**What it learns:** not a bottleneck label (the data has runtimes only, and a label derived from runtime plus
datasheet peaks would just reproduce Roofline). It learns *how far below its Roofline a kernel runs*:
`y = log10(measured time / Roofline ideal time)`, from shape, FLOPs, bytes, peak, bandwidth and SM count.
`predict_real.py` outputs the Roofline side, predicted time with an 80% interval, predicted efficiency, and the
validation numbers for that op.

**Results** (median relative error of predicted kernel time; lower is better; all numbers cross-validated):

| Test | Plain Roofline | Per-op constant efficiency | Learned model |
|---|---|---|---|
| New shapes, known GPUs | 58% | 25% | **3.3%** (96% within 25%; 80% interval covers 79%) |
| New GPU (siblings in training) | 58% | 28% | **12.8%** (76% within 25%) |
| New architecture (whole family held out) | 58% | 27% | **21.3%** (56% within 25%) |

Honest reading:
* Interpolation (new shapes on known hardware) is very good. Hardware extrapolation is much weaker:
  on a new architecture the model beats a "constant efficiency per op" guess only modestly, and for plain BF16
  GEMM it does **not** beat it (19% vs 17%). Its gains come mostly from small, latency-bound kernels.
* The 80% intervals are **not trustworthy on new hardware**: they cover only ~27% of truths on unseen
  architectures (45% on unseen GPUs). They are calibrated only for new shapes on known GPUs.
* "Unseen GPU" is optimistic because same-die siblings (H100/H200/H800, L20/L40/RTX 6000 Ada, A40/A6000) stay in
  training; "unseen architecture" is the fair hardware test. Only 4 architectures exist in the data.
* Efficiency is lowest *near the ridge point* for GEMM and attention (`results/real/figs/efficiency_vs_ratio.png`):
  Roofline is least reliable exactly where compute and memory ceilings meet.
* FLOP/byte formulas are analytic and some are assumptions (see `OP_NOTES` in `rooflinelab/realdata.py` and
  the `predict_real.py` output). GEMM/FP8 match the dataset's own `overall_perf` (median ratio 1.00); MoE,
  RMSNorm and SiLU*Mul do not track it, so treat those ops' *absolute* efficiency with caution (the model
  still learns their actual times). SiLU*Mul shows efficiency >100% because small tensors stay in L2.
* RTX 6000 Ada's listed tensor peak looks doubled (it delivers the same TFLOPS as the L40 on identical GEMMs),
  so its peak is halved (`PEAK_FIX`). This is an inference from the data, not vendor-confirmed.
* **Domain limits:** tensor-core GPUs, BF16/FP8 kernels. `predict_real.py` refuses GEMM/attention/MoE for GPUs
  without tensor cores (e.g. GTX 1650 Ti); use `predict.py` there. Hardware outside the training range triggers a warning.

## Your GTX 1650 Ti workflow (getting real numbers)

1. `python detect_specs.py --out hw/my_gpu.json` - needs PyTorch+CUDA, no Nsight. Measures the FP32/FP16
   GEMM and copy-bandwidth your laptop actually reaches (usually below datasheet). Then use
   `--hw-json hw/my_gpu.json` instead of `--gpu`. Roofline results are only as good as these ceilings.
2. Install Nsight Compute (`ncu`), allow GPU counter access (on Windows: NVIDIA Control Panel ->
   Developer -> Manage GPU Performance Counters -> allow all users), plug in the laptop.
3. `python measure_gpu.py --out data/measured_1650ti.csv --max-configs 60` (skips workloads too big for 4 GB;
   `--list` shows the count; a full sweep is ~1000 ncu runs, so start with `--max-configs 5` and check
   a few rows against the Nsight GUI).
4. `python train.py --real data/measured_1650ti.csv` prints, BEFORE the final fit, how Roofline and the
   simulation-trained model do on your real rows - that is the honest test of whether the simulator
   resembles your GPU. The final model blends simulated rows with your real rows (weight x5).

Caveat: one GPU can calibrate the model for that GPU but cannot show generalisation to unseen hardware
(experiment B needs several GPUs).

## Read this first: what is and is not real here

| Part | Status |
|---|---|
| Roofline baseline, features, 5-model ladder, experiments A/B/C, figures | Implemented and run |
| `predict.py`, `train.py`, advisor | Implemented and tested on CPU with simulated labels |
| Labels in `results/` (top level) and `models/model.joblib` | **Simulated** (`rooflinelab/simulate.py`), not hardware measurements |
| `predict_real.py`, `train_real.py`, `results/real/` | **Real measured data** (PipeWeave, 452k kernels, 11 GPUs); formulas partly assumed; no GPU of yours involved |
| `measure_gpu.py`, `detect_specs.py` | Written, **never run on a GPU** (parser unit-tested on a synthetic CSV) |
| Hardware spec table (`hw_specs.py`) | Entered from memory; **verify against vendor datasheets** (1650 Ti is a laptop part; clocks/memory vary) |

The simulator encodes effects Roofline ignores (L2 residency / warm-cache reuse, GEMM tile
reload traffic, wave quantisation, small-dimension under-utilisation, gather coalescing,
per-chip efficiency, imperfect overlap, noise). So "ML beats Roofline" on this data means only
that the models can recover effects I put there. It validates the pipeline and methodology;
it is not an architectural finding. The real result needs real labels.

## Quick start

    pip install -r requirements.txt
    python run_all.py                        # simulated data, ~8 min, writes results/
    # real data:
    python measure_gpu.py --out data/a100.csv    # on each GPU (needs torch+CUDA+ncu)
    # concatenate the per-GPU CSVs into data/all.csv
    python run_all.py --data data/all.csv

## Design decisions

* **Roofline is the baseline**: `AI = F/Q` vs ridge `P_peak/B`. It is given the correct
  precision/tensor-core peak, the most favourable fair version.
* **Labels from a ratio r = compute-side / memory-side**: simulator uses time components;
  the GPU harness uses Nsight "speed of light" counters (`sm__throughput` / `gpu__dram_throughput`).
  Binary: compute-bound iff r > 1. Three-class: "mixed" iff r within 1.5x of 1 (quantitative, per your note).
* **Why not label from kernel time alone**: F/(T*P) divided by Q/(T*B) equals AI/ridge exactly,
  so the label would just be the Roofline rule. Real labels need counters.
* **Features have no workload ID**, only structural properties (FLOPs, bytes, working set,
  tiles, min dimension, irregularity, precision) and hardware properties (peak, BW, L2, SMs...),
  so the unseen-workload experiment is meaningful. Two feature sets: `full` (includes
  Roofline-derived AI/ridge columns) and `raw` (without them). Note a logistic regression on
  the raw log features can represent the Roofline rule exactly (log AI - log ridge is linear in them).
* **Kernels dominated by launch overhead (>50% of time) are dropped** (16% of simulated rows):
  neither ceiling is the limiter there.
* **Experiments**: random split (leaky contrast), A leave-one-workload-family-out (12 folds),
  B leave-one-GPU-out (10 folds), C hold out 20% of (family, GPU) pairs (5 repeats).
* Metrics: accuracy, balanced accuracy, macro-F1, and accuracy near the ridge (within 3x), where Roofline should struggle.

## Simulated-data results (2-class, accuracy; raw features)

| Experiment | Roofline | LogReg | RandForest | GradBoost | MLP |
|---|---|---|---|---|---|
| A unseen workload | 0.890 | 0.944 | 0.915 | 0.936 | 0.963 |
| B unseen hardware | 0.864 | 0.948 | 0.943 | 0.964 | 0.961 |
| C unseen pair | 0.880 | 0.946 | 0.956 | 0.971 | 0.971 |
| random split (leaky) | 0.872 | 0.947 | 0.972 | 0.980 | 0.974 |

Near the ridge Roofline drops to ~0.65-0.70 vs ~0.81-0.95 for the ML models. Full tables for
both feature sets and the 3-class task are in `results/summary.md`; per-fold numbers in `results/all_folds.csv`.

Observations (all conditional on the simulator):
* Unseen-workload (A) is the hardest generalisation; with raw features RandomForest even
  trails LogReg there, i.e. tree models extrapolate worse to unseen families.
* The leaky random split inflates tree models by only ~1-2 points here because the simulator is
  smooth; do not assume the gap is this small on real measurements.
* "Mixed" is only ~9% of rows and is where all models are weakest (balanced accuracy 0.66-0.88).
* Permutation importance (`results/figs/feature_importance.png`) ranks tensor-core eligibility,
  FLOPs, min dimension, peak throughput on top. That reflects how I built the simulator, not hardware.

## Layout

    rooflinelab/hw_specs.py     GPU table + effective peak
    rooflinelab/workloads.py    12 kernel families, analytic FLOPs/bytes/working set
    rooflinelab/features.py     feature vector, label bands
    rooflinelab/simulate.py     stylised label generator (replace with real data)
    rooflinelab/baseline.py     Roofline classifier
    rooflinelab/experiments.py  model ladder + splits + metrics
    rooflinelab/advisor.py      Roofline stats, ML verdict, what-if, optimisation advice, report
    rooflinelab/plots.py        figures
    predict.py                  CLI: hardware specs + workload -> report (simulated-label classifier; any GPU)
    train.py                    trains/saves models/model.joblib (simulated, or + real CSVs)
    predict_real.py             CLI: learned time/efficiency prediction from REAL data (tensor-core GPUs)
    train_real.py               trains + cross-validates models/real_efficiency.joblib
    fetch_real_data.py          downloads the PipeWeave CSVs into data/real/
    make_real_figs.py           figures in results/real/figs/
    rooflinelab/realdata.py     loader: analytic FLOPs/bytes/peak per op, efficiency target
    detect_specs.py             measures your GPU's achievable ceilings -> hw/*.json (untested on hardware)
    measure_gpu.py              Nsight Compute label collection (untested on hardware)
    run_all.py                  research experiments (Roofline vs ML ladder, A/B/C)
    results/                    summary.md, summary.csv, all_folds.csv, figs/

## Before trusting real data

* Run `measure_gpu.py --max-configs 5` on one GPU first and compare a few rows to the Nsight GUI.
* `ncu` flushes caches by default (`--cache-control all`), which removes the warm-L2 effect the
  simulator includes; try both settings and report which you used.
* ncu locks clocks to base by default; decide whether that matches the peaks in `hw_specs.py`.
* You need >= 4-5 GPUs for experiment B to mean anything; Colab/Kaggle/cloud rentals cover T4, L4, A100, H100.
* Natural extensions: extra classes (latency / cache-bound), whole-model workloads (ResNet, BERT) by summing kernels.
