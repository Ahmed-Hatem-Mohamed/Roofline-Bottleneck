"""Download the real measured kernel-latency data used by train_real.py.

Source: PipeWeave (https://github.com/zksainx/pipeweave, Apache-2.0 repo; the dataset's own
license is not stated - cite the paper, arXiv 2601.14910, and check with the authors before
redistributing). Measured with PyTorch Profiler on 11 NVIDIA datacenter/workstation GPUs.

    python fetch_real_data.py

Copies the operator CSVs to data/real/. One file, dataset/gemm_train.csv (~131 MB), is stored
with Git LFS; this script tries `git lfs pull` and tells you if it is missing. train_real.py
works without it (it uses gemm_test.csv, 118,800 GEMM rows) and picks it up automatically if
present.

Needs: git (and git-lfs for the big file). ~400 MB of disk. Data is not committed to your repo
(data/real/ is in .gitignore).
"""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/zksainx/pipeweave.git"
CLONE = os.path.join("data", "pipeweave_repo")
OUT = os.path.join("data", "real")
WANT = ["gemm_train", "gemm_test", "gemm_fp8_train", "gemm_fp8_test", "attn_train", "attn_test",
        "rmsnorm_train", "rmsnorm_test", "siluandmul_train", "siluandmul_test", "moe_train"]


def run(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def is_pointer(path):
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


def main():
    os.makedirs("data", exist_ok=True)
    if not os.path.isdir(os.path.join(CLONE, ".git")):
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        print("cloning (shallow)...")
        r = subprocess.run(["git", "clone", "--depth", "1", REPO, CLONE], env=env, text=True)
        if r.returncode:
            sys.exit("clone failed")
    big = os.path.join(CLONE, "dataset", "gemm_train.csv")
    if is_pointer(big):
        r = run(["git", "lfs", "pull", "--include", "dataset/gemm_train.csv"], cwd=CLONE)
        if r.returncode or is_pointer(big):
            print("NOTE: gemm_train.csv (Git LFS, ~131 MB) was NOT downloaded:\n  "
                  + (r.stderr or r.stdout).strip().splitlines()[-1][:200]
                  + "\n  Install git-lfs (`git lfs install`) and rerun, or continue without it.")
    os.makedirs(OUT, exist_ok=True)
    got, missing = [], []
    for n in WANT:
        src = os.path.join(CLONE, "dataset", n + ".csv")
        if os.path.exists(src) and not is_pointer(src):
            shutil.copy(src, os.path.join(OUT, n + ".csv"))
            got.append(n)
        else:
            missing.append(n)
    print(f"copied {len(got)} files to {OUT}: {', '.join(got)}")
    if missing:
        print(f"not available: {', '.join(missing)}")


if __name__ == "__main__":
    main()
