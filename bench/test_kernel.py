"""Custom small-M quantised matmul: correctness against mx.quantized_matmul and
speed in a dependent chain (every rep evaluated), at the model's shapes.

    bench/benchlock.sh -- .venv/bin/python bench/test_kernel.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from omnivoice_mlx.kernels import qmm_small, rows_per_pass  # noqa: E402


def chain_bench(fn, x, L, reps=5):
    """Per-call µs of fn in a dependent chain of L calls, all reps evaluated."""
    def run():
        y = x
        for i in range(L):
            y = fn(y, i)
        return y
    mx.eval(run())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(run())
    mx.synchronize()
    return (time.perf_counter() - t0) / reps / L * 1e6


def main():
    mx.random.seed(0)
    print("correctness (max |diff| / max |ref|):")
    for M, K, N in [(78, 1024, 4096), (39, 1024, 1024), (126, 3072, 1024), (252, 1024, 6144), (7, 1024, 2048), (300, 2048, 1024)]:
        x = (mx.random.normal((M, K)) * 0.5).astype(mx.float16)
        w = mx.random.normal((N, K)).astype(mx.float16) * 0.05
        wq, sc, bi = mx.quantize(w, group_size=64, bits=8)
        ref = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=8)
        out = qmm_small(x, wq, sc, bi)
        err = float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max()) / float(mx.abs(ref).max())
        print(f"  M={M:<4} K={K:<5} N={N:<5} MT={rows_per_pass(M):<3} rel err {err:.2e}")

    print("\nspeed, dependent chain of 28 GEMMs (µs per GEMM): mlx qmm | custom | fp16 matmul")
    L = 28
    for K, N in [(1024, 4096), (2048, 1024), (1024, 6144), (3072, 1024)]:
        ws = [mx.quantize(mx.random.normal((N, K)).astype(mx.float16), group_size=64, bits=8) for _ in range(L)]
        w16 = [mx.random.normal((N, K)).astype(mx.float16) for _ in range(L)]
        mx.eval(*[a for t in ws for a in t], *w16)
        print(f"(K,N)=({K},{N})")
        for M in (39, 78, 126, 252):
            x = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x)
            def keep(y):  # bring [M, N] back to [M, K] for the next link without extra kernels of note
                return y[:, :K] if N >= K else mx.concatenate([y] * (K // N), axis=1)
            t_mlx = chain_bench(lambda y, i: keep(mx.quantized_matmul(y, *ws[i], transpose=True, group_size=64, bits=8)), x, L)
            t_cus = chain_bench(lambda y, i: keep(qmm_small(y, *ws[i])), x, L)
            t_f16 = chain_bench(lambda y, i: keep(y @ w16[i].T), x, L)
            tflops = 2 * M * K * N / (t_cus * 1e-6) / 1e12
            print(f"  M={M:<4} mlx {t_mlx:7.1f}  custom {t_cus:7.1f} ({t_mlx / t_cus:4.2f}x, {tflops:4.1f} TFLOPS)  fp16 {t_f16:7.1f}")


if __name__ == "__main__":
    main()
