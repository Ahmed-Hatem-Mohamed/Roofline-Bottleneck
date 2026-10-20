"""Analytical workload descriptions (FLOPs, compulsory bytes, working set, structure).

Each workload is one GPU kernel-level op (the granularity at which a bottleneck is
well defined). `family` groups configs and is the unit held out in Experiment A.
`params` holds exactly what the real profiling harness needs to run the op.
"""
import math
import numpy as np

BYTES = {"fp32": 4, "fp16": 2}
TILE = 128
STREAM_BLOCK = 4096


def _ceil(a, b):
    return -(-int(a) // int(b))


def _gemm_like(family, desc, prec, flops, a_bytes, b_bytes, c_bytes, m_tiles, n_tiles,
               batch, min_dim, tensor, params):
    nbytes = a_bytes + b_bytes + c_bytes
    reload = a_bytes * (n_tiles - 1) + b_bytes * (m_tiles - 1)
    return dict(family=family, desc=desc, precision=prec, flops=float(flops),
                bytes=float(nbytes), ws=float(nbytes), reload=float(reload),
                irregular=0, row_bytes=TILE * BYTES[prec], tiles=int(m_tiles * n_tiles * batch),
                min_dim=int(min_dim), tensor_eligible=int(tensor), base_eff=0.8 if tensor else 0.7,
                params=params)


def gemm(family, M, N, K, prec, batch=1):
    b = BYTES[prec]
    return _gemm_like(family, f"gemm M{M} N{N} K{K}", prec, 2 * M * N * K * batch,
                      M * K * b * batch, K * N * b * batch, M * N * b * batch,
                      _ceil(M, TILE), _ceil(N, TILE), batch, min(M, N, K), True,
                      dict(op="gemm", M=M, N=N, K=K, batch=batch))


def conv(family, Nb, C, H, K, R, stride, prec):
    b = BYTES[prec]
    Ho = max(1, H // stride)
    in_b, w_b, out_b = Nb * C * H * H * b, K * C * R * R * b, Nb * K * Ho * Ho * b
    M = Nb * Ho * Ho
    return _gemm_like(family, f"conv N{Nb} C{C} H{H} K{K} R{R}", prec,
                      2 * Nb * K * Ho * Ho * C * R * R, in_b, w_b, out_b,
                      _ceil(M, TILE), _ceil(K, TILE), 1, min(K, C * R * R), True,
                      dict(op="conv", N=Nb, C=C, H=H, K=K, R=R, stride=stride))


def depthwise(Nb, C, H, R, prec):
    b = BYTES[prec]
    n_out = Nb * C * H * H
    nbytes = (2 * n_out + C * R * R) * b
    return dict(family="depthwise_conv", desc=f"dwconv N{Nb} C{C} H{H}", precision=prec,
                flops=float(2 * n_out * R * R), bytes=float(nbytes), ws=float(nbytes), reload=0.0,
                irregular=0, row_bytes=H * b, tiles=max(1, _ceil(n_out, 1024)), min_dim=64,
                tensor_eligible=0, base_eff=0.4,
                params=dict(op="depthwise", N=Nb, C=C, H=H, R=R))


def attn_scores(Bt, heads, L, d, prec):
    b = BYTES[prec]
    bh = Bt * heads
    a = bh * L * d * b
    out = bh * L * L * b
    return _gemm_like("attention_scores", f"qk^T B{Bt} h{heads} L{L} d{d}", prec,
                      2 * bh * L * L * d, a, a, out, _ceil(L, TILE), _ceil(L, TILE), bh, min(L, d),
                      True, dict(op="attn_scores", B=Bt, heads=heads, L=L, d=d))


def _stream(family, desc, prec, n, flops_per, rw, params, irregular=0, row_bytes=128, extra_bytes=0):
    b = BYTES[prec]
    nbytes = rw * n * b + extra_bytes
    return dict(family=family, desc=desc, precision=prec, flops=float(flops_per * n),
                bytes=float(nbytes), ws=float(nbytes), reload=0.0, irregular=irregular,
                row_bytes=int(row_bytes), tiles=max(1, _ceil(n, STREAM_BLOCK)), min_dim=1024,
                tensor_eligible=0, base_eff=0.5, params=params)


def softmax(Bt, heads, L, prec):
    n = Bt * heads * L * L
    return _stream("softmax", f"softmax B{Bt} h{heads} L{L}", prec, n, 5, 2,
                   dict(op="softmax", rows=Bt * heads * L, length=L))


def layernorm(tokens, H, prec):
    n = tokens * H
    return _stream("layernorm", f"layernorm T{tokens} H{H}", prec, n, 8, 2,
                   dict(op="layernorm", tokens=tokens, H=H), extra_bytes=2 * H * BYTES[prec])


def elementwise(kind, n, prec):
    k, rw = {"add": (1, 3), "gelu": (8, 2), "silu": (6, 2)}[kind]
    return _stream("elementwise", f"{kind} n{int(n)}", prec, int(n), k, rw, dict(op=kind, n=int(n)))


def reduction(n, prec):
    return _stream("reduction", f"sum n{int(n)}", prec, int(n), 1, 1, dict(op="sum", n=int(n)))


def gather(rows, D, prec):
    b = BYTES[prec]
    nbytes = 2 * rows * D * b + rows * 8
    return dict(family="embedding_gather", desc=f"gather rows{rows} D{D}", precision=prec,
                flops=float(rows * D), bytes=float(nbytes), ws=float(nbytes), reload=0.0,
                irregular=1, row_bytes=int(D * b), tiles=max(1, _ceil(rows * D, STREAM_BLOCK)),
                min_dim=1024, tensor_eligible=0, base_eff=0.3,
                params=dict(op="gather", rows=rows, D=D))


def make_workload(op, prec, **k):
    """Build a workload from the same param names stored in w['params'] (used by the CLI,
    by what-if analysis, and consistent with measure_gpu.build_op)."""
    if op == "gemm":
        return gemm("gemm", k["M"], k["N"], k["K"], prec, k.get("batch", 1))
    if op == "conv":
        return conv("conv", k["N"], k["C"], k["H"], k["K"], k["R"], k.get("stride", 1), prec)
    if op == "depthwise":
        return depthwise(k["N"], k["C"], k["H"], k["R"], prec)
    if op == "attn_scores":
        return attn_scores(k["B"], k["heads"], k["L"], k["d"], prec)
    if op == "softmax":
        return _stream("softmax", f"softmax rows{k['rows']} len{k['length']}", prec,
                       k["rows"] * k["length"], 5, 2, dict(op="softmax", rows=k["rows"], length=k["length"]))
    if op == "layernorm":
        return layernorm(k["tokens"], k["H"], prec)
    if op in ("add", "gelu", "silu"):
        return elementwise(op, k["n"], prec)
    if op == "sum":
        return reduction(k["n"], prec)
    if op == "gather":
        return gather(k["rows"], k["D"], prec)
    raise ValueError(f"unknown op '{op}'. Supported: {sorted(OPS)}")


OPS = {
    "gemm": "M N K [batch]            matrix multiply / linear layer / FFN / decode GEMV (M small)",
    "conv": "N C H K R [stride]       2-D convolution (N batch, C in-ch, H spatial, K out-ch, R kernel)",
    "depthwise": "N C H R               depthwise convolution",
    "attn_scores": "B heads L d          Q.K^T of attention (B batch, L sequence, d head dim)",
    "softmax": "rows length            row-wise softmax",
    "layernorm": "tokens H              LayerNorm over H",
    "add": "n                          elementwise add of two vectors",
    "gelu": "n                         elementwise GELU",
    "silu": "n                         elementwise SiLU",
    "sum": "n                          reduction",
    "gather": "rows D                  embedding-style row gather",
}


def _logint(rng, lo, hi):
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def _grid(prec):
    out = []
    for M in (128, 512, 2048, 8192):
        for N in (128, 1024, 4096):
            for K in (64, 512, 4096):
                out.append(gemm("gemm", M, N, K, prec))
    for batch in (1, 2, 4, 8, 16, 32, 64, 128):
        for H in (2048, 4096, 8192, 12288):
            out.append(gemm("decode_gemv", batch, H, H, prec))
    for tokens in (128, 512, 2048, 8192, 32768, 131072):
        for K, N in ((768, 3072), (3072, 768), (1024, 4096), (4096, 1024)):
            out.append(gemm("bert_ffn", tokens, N, K, prec))
    for Nb in (1, 8, 32, 128):
        for C, K, H in ((64, 64, 56), (128, 128, 28), (256, 256, 14), (512, 512, 7)):
            out.append(conv("conv3x3", Nb, C, H, K, 3, 1, prec))
        for C, K, H in ((256, 64, 56), (512, 128, 28), (1024, 256, 14), (2048, 512, 7)):
            out.append(conv("conv1x1", Nb, C, H, K, 1, 1, prec))
        for C, H in ((32, 112), (96, 56), (192, 28), (384, 14), (960, 7)):
            out.append(depthwise(Nb, C, H, 3, prec))
    for Bt in (1, 8, 32):
        for L in (128, 512, 2048, 4096):
            out.append(attn_scores(Bt, 12, L, 64, prec))
            out.append(softmax(Bt, 12, L, prec))
    for tokens in (128, 1024, 8192, 65536, 262144):
        for H in (768, 1024, 4096):
            out.append(layernorm(tokens, H, prec))
    for n in (1e4, 1e5, 1e6, 1e7, 1e8, 4e8):
        for kind in ("add", "gelu", "silu"):
            out.append(elementwise(kind, n, prec))
        out.append(reduction(n, prec))
    for rows in (1000, 10000, 100000, 1000000):
        for D in (128, 512, 2048):
            out.append(gather(rows, D, prec))
    return out


def _random(prec, rng, per_family=30):
    out = []
    for _ in range(per_family):
        out.append(gemm("gemm", _logint(rng, 16, 16384), _logint(rng, 16, 16384), _logint(rng, 16, 16384), prec))
        out.append(gemm("decode_gemv", _logint(rng, 1, 256), _logint(rng, 1024, 16384), _logint(rng, 1024, 16384), prec))
        out.append(gemm("bert_ffn", _logint(rng, 64, 262144), int(rng.choice([3072, 4096, 768, 1024])),
                        int(rng.choice([768, 1024, 3072, 4096])), prec))
        C = int(rng.choice([32, 64, 128, 256, 512, 1024])); H = int(rng.choice([7, 14, 28, 56, 112]))
        out.append(conv("conv3x3", _logint(rng, 1, 256), C, H, int(rng.choice([64, 128, 256, 512])), 3, 1, prec))
        out.append(conv("conv1x1", _logint(rng, 1, 256), C, H, int(rng.choice([64, 128, 256, 512])), 1, 1, prec))
        out.append(depthwise(_logint(rng, 1, 256), C, H, int(rng.choice([3, 5])), prec))
        Bt, L = _logint(rng, 1, 64), _logint(rng, 64, 8192)
        out.append(attn_scores(Bt, int(rng.choice([8, 12, 16, 32])), L, int(rng.choice([64, 128])), prec))
        out.append(softmax(Bt, int(rng.choice([8, 12, 16, 32])), _logint(rng, 64, 4096), prec))
        out.append(layernorm(_logint(rng, 64, 500000), int(rng.choice([768, 1024, 2048, 4096, 8192])), prec))
        out.append(elementwise(str(rng.choice(["add", "gelu", "silu"])), _logint(rng, 1e4, 6e8), prec))
        out.append(reduction(_logint(rng, 1e4, 6e8), prec))
        out.append(gather(_logint(rng, 500, 2000000), int(rng.choice([64, 128, 256, 512, 1024, 2048, 4096])), prec))
    return out


def generate_workloads(seed=0):
    rng = np.random.default_rng(seed)
    ws = []
    for prec in ("fp32", "fp16"):
        ws += _grid(prec)
        ws += _random(prec, rng)
    return ws


FAMILIES = ["gemm", "decode_gemv", "bert_ffn", "conv3x3", "conv1x1", "depthwise_conv",
            "attention_scores", "softmax", "layernorm", "elementwise", "reduction",
            "embedding_gather"]
