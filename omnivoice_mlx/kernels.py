"""Custom Metal kernels (``mx.fast.metal_kernel``) for the shapes this model
runs at: a handful of rows (M = 40–300) against 8-bit affine-quantised weights
(group 64, MLX packing: element k of row n in word k//4, byte k%4, LSB first,
w = scale * q + bias).

``qmm_small``: multi-row GEMV. One thread owns a slice of one weight column
(row n of W, 16 of the K elements per chunk), dequantises it once into
registers and dots it against every input row staged in threadgroup memory.
Weights are streamed exactly once per pass of ``MT`` rows, the FMAs run from
registers, eight lanes per column split K and reduce with simd shuffles.
MLX's steel qmm tiles over (M, N) with simdgroup matrices, which at M ≈ 78 in
a dependent chain runs at ~5 TFLOPS; this kernel targeted the ALU rate instead.

Result (2026-09-09, bench/test_kernel.py, M2 Max): numerically right (rel err
5e-4, fp16 rounding) but 1.3–5x SLOWER than mx.quantized_matmul at every shape
(2–3 TFLOPS at M ≤ 78, register spills above). Scalar/half4 FMAs from
threadgroup memory cannot compete with simdgroup-matrix tiles; a second
generation would have to use ``simdgroup_half8x8`` with x staged for 8+
simdgroups and 8-row M tiles (less padding waste than steel's 32/64) — a
multi-day effort with uncertain payoff. Kept as the documented negative
result; not used by the model.
"""
from __future__ import annotations

import math

import mlx.core as mx

_SRC = r"""
    constexpr int KC = 128;      // K elements per chunk staged in threadgroup memory
    constexpr int LANES = 8;     // lanes per output column, 16 elements each per chunk
    constexpr int COLS = 32;     // columns per threadgroup (256 threads)
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid % LANES;
    const uint col = tid / LANES;
    const int n = int(threadgroup_position_in_grid.x) * COLS + int(col);
    const int M = x_shape[0];
    const int K = x_shape[1];
    const int N = w_shape[0];
    const int KW = K / 4;
    const int G = K / 64;
    threadgroup half xs[MT * KC];
    const device uint4* wrow = (const device uint4*)(w + n * KW);
    for (int m0 = 0; m0 < M; m0 += MT) {
        const int rows = min(MT, M - m0);
        float acc[MT];
        #pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] = 0.0f;
        for (int kc = 0; kc < K; kc += KC) {
            // stage x[m0 : m0+MT, kc : kc+KC]; rows past M read as zero
            for (int i = int(tid); i < MT * (KC / 4); i += 256) {
                const int m = i / (KC / 4);
                const int k4 = i % (KC / 4);
                half4 v = half4(0.0h);
                if (m < rows) {
                    v = *((const device half4*)(x + (m0 + m) * K + kc) + k4);
                }
                ((threadgroup half4*)xs)[i] = v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int k0 = kc + int(lane) * 16;
            const int g = k0 / 64;
            const float s = float(scales[n * G + g]);
            const float b = float(biases[n * G + g]);
            const uint4 packed = wrow[k0 / 16];
            half4 wv[4];
            {
                const uint words[4] = {packed.x, packed.y, packed.z, packed.w};
                const half hs = half(s), hb = half(b);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint word = words[j];
                    wv[j] = half4(half(word & 0xFFu), half((word >> 8) & 0xFFu),
                                  half((word >> 16) & 0xFFu), half((word >> 24) & 0xFFu)) * hs + hb;
                }
            }
            #pragma unroll
            for (int m = 0; m < MT; ++m) {
                const threadgroup half4* xr = (const threadgroup half4*)(xs + m * KC + int(lane) * 16);
                acc[m] += float(dot(xr[0], wv[0]) + dot(xr[1], wv[1])) + float(dot(xr[2], wv[2]) + dot(xr[3], wv[3]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma unroll
        for (int m = 0; m < MT; ++m) {
            float v = acc[m];
            v += simd_shuffle_xor(v, 1);
            v += simd_shuffle_xor(v, 2);
            v += simd_shuffle_xor(v, 4);
            if (lane == 0 && m < rows) out[(m0 + m) * N + n] = T(v);
        }
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omnivoice_qmm_small",
            input_names=["x", "w", "scales", "biases"],
            output_names=["out"],
            source=_SRC,
            ensure_row_contiguous=True,
        )
    return _KERNEL


def rows_per_pass(M: int, max_rows: int = 64) -> int:
    """Smallest multiple of 8 such that ceil(M / MT) passes waste the least work."""
    passes = math.ceil(M / max_rows)
    return min(max_rows, ((math.ceil(M / passes) + 7) // 8) * 8)


def qmm_small(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, *,
              out_dtype: mx.Dtype = mx.float16) -> mx.array:
    """x [M, K] float16 @ dequant(w, scales, biases).T -> [M, N]; K % 128 == 0, N % 32 == 0."""
    M, K = x.shape
    N = w.shape[0]
    if K % 128 or N % 32:
        raise ValueError(f"qmm_small needs K % 128 == 0 and N % 32 == 0, got K={K}, N={N}")
    MT = rows_per_pass(M)
    return _kernel()(
        inputs=[x, w, scales, biases],
        template=[("T", out_dtype), ("MT", MT)],
        grid=(N // 32 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[out_dtype],
    )[0]
