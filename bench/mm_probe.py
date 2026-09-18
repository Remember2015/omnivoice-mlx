"""Cost model of the matmuls a step is made of: fp16 GEMM vs 4/8-bit quantised
matmul (g64 / g128), K=1024 (and 3072 for down_proj), N in the shapes the
model uses, M from 1 to 312 rows.

    bench/benchlock.sh -- .venv/bin/python bench/mm_probe.py
"""
from __future__ import annotations

import time

import mlx.core as mx


def bench(fn, n=40):
    mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [fn() for _ in range(n)]
    mx.eval(*outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    mx.random.seed(0)
    shapes = [(1024, 1024), (1024, 2048), (1024, 3072), (1024, 4096), (1024, 6144), (3072, 1024)]
    Ms = [1, 39, 78, 156, 312, 624]
    print("ms per matmul; rows = M; columns = (K,N); fp16 | q8g64 | q4g64 | q8g128 | q4g128")
    for K, N in shapes:
        w16 = mx.random.normal((N, K)).astype(mx.float16)
        q = {}
        for bits in (8, 4):
            for gs in (64, 128):
                q[(bits, gs)] = mx.quantize(w16, group_size=gs, bits=bits)
        mx.eval(w16, *[a for t in q.values() for a in t])
        print(f"\n(K,N)=({K},{N})  weight fp16 {N * K * 2 / 1e6:.1f} MB, q8 {N * K / 1e6:.1f} MB")
        for M in Ms:
            x = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x)
            t16 = bench(lambda: x @ w16.T)
            cells = [f"{t16:7.3f}"]
            for bits in (8, 4):
                for gs in (64, 128):
                    wq, sc, bi = q[(bits, gs)]
                    cells.append(f"{bench(lambda: mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=gs, bits=bits)):7.3f}")
            flops = 2 * M * K * N
            print(f"  M={M:<4} " + " ".join(cells) + f"   (fp16 {flops / t16 / 1e9:6.0f} GFLOP/s)")


if __name__ == "__main__":
    main()
