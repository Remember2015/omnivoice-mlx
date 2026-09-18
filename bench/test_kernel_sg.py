"""Simdgroup-matrix quantised matmul: correctness against mx.quantized_matmul
and speed in a dependent chain (every rep evaluated), at the model's shapes.

    bench/benchlock.sh --load 6 -- .venv/bin/python bench/test_kernel_sg.py

Reuses ``chain_bench`` from bench/test_kernel.py so the numbers are directly
comparable with the earlier multi-row-GEMV attempt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from omnivoice_mlx.kernels_sg import pick_config, qmm_sg  # noqa: E402
from test_kernel import chain_bench  # noqa: E402

KS = (1024, 2048, 3072)
NS = (1024, 2048, 4096, 6144)
BENCH_SHAPES = ((1024, 4096), (2048, 1024), (1024, 6144), (3072, 1024))
TOL = 2e-3


def correctness() -> int:
    print("correctness: max |diff| / max |ref| against mx.quantized_matmul")
    print(f"  {'M':>5} " + " ".join(f"{'K=%d,N=%d' % (K, N):>13}" for K in KS for N in NS))
    bad = 0
    ws = {}
    for K in KS:
        for N in NS:
            w = (mx.random.normal((N, K)) * 0.05).astype(mx.float16)
            ws[(K, N)] = mx.quantize(w, group_size=64, bits=8)
    for M in (7, 39, 78, 126, 252, 300):
        cells = []
        for K in KS:
            x = (mx.random.normal((M, K)) * 0.5).astype(mx.float16)
            for N in NS:
                wq, sc, bi = ws[(K, N)]
                ref = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=8)
                out = qmm_sg(x, wq, sc, bi)
                err = (float(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max())
                       / float(mx.abs(ref).max()))
                bad += err > TOL
                cells.append(f"{err:13.2e}" + ("!" if err > TOL else " "))
        print(f"  {M:>5} " + " ".join(c[:13] for c in cells))
    print(f"  tolerance {TOL:.0e}: {'FAIL' if bad else 'all pass'}")
    return bad


def speed() -> None:
    L = 28
    print(f"\nspeed: dependent chain of {L} GEMMs, us per GEMM (every rep evaluated)")
    print(f"  {'K':>5} {'N':>5} {'M':>5} {'mlx':>8} {'qmm_sg':>8} {'ratio':>6} "
          f"{'TFLOPS':>7} {'config':>24}")
    for K, N in BENCH_SHAPES:
        ws = [mx.quantize(mx.random.normal((N, K)).astype(mx.float16), group_size=64, bits=8)
              for _ in range(L)]
        mx.eval(*[a for t in ws for a in t])

        def keep(y):  # bring [M, N] back to [M, K] for the next link
            return y[:, :K] if N >= K else mx.concatenate([y] * (K // N), axis=1)

        for M in (39, 78, 126, 252):
            x = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x)
            cfg = pick_config(M, K, N)
            t_mlx = chain_bench(lambda y, i: keep(mx.quantized_matmul(
                y, *ws[i], transpose=True, group_size=64, bits=8)), x, L)
            t_sg = chain_bench(lambda y, i: keep(qmm_sg(y, *ws[i], config=cfg)), x, L)
            tflops = 2 * M * K * N / (t_sg * 1e-6) / 1e12
            print(f"  {K:>5} {N:>5} {M:>5} {t_mlx:8.1f} {t_sg:8.1f} {t_mlx / t_sg:5.2f}x "
                  f"{tflops:7.2f} {str(cfg):>24}")


def main() -> None:
    mx.random.seed(0)
    bad = correctness()
    speed()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
