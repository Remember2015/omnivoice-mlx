"""Simdgroup-matrix quantised matmul for the small-M shapes this model runs at.

``qmm_sg(x, w, scales, biases)`` computes ``x @ dequant(w, scales, biases).T``
for ``x`` [M, K] float16 and 8-bit affine-quantised weights (group 64, MLX
packing) with M in the 39-300 range, using Metal ``simdgroup_half8x8``
fragments and 8-row M tiles, so M = 39 pads to 40 rather than to the 32/64 that
MLX's steel qmm_t pads to.

Result (2026-09-10, bench/test_kernel_sg.py, M2 Max, MLX 0.32.2): numerically
right (rel err <= 1.3e-3, fp16 rounding) and a **partial** win.  Against
``mx.quantized_matmul`` in a dependent chain of 28 GEMMs:

    (K, N)          M=39    M=78    M=126   M=252
    (1024, 4096)    1.18x   1.02x   0.90x   0.85x
    (1024, 6144)    1.22x   1.02x   0.85x   0.89x
    (2048, 1024)    0.92x   0.90x   0.89x   0.90x
    (3072, 1024)    0.88x   0.86x   0.87x   0.89x

So: worth using only at M <= 80 with N >= 4096 (up to 1.22x, 5.0 TFLOPS);
MLX wins everywhere else.  Why, measured rather than guessed:

* MLX's only structural waste is padding M to a multiple of 32, which is
  1.6x at M=39 and 1.23x at M=78 and nothing at M >= 126.  Its qmm_t already
  issues mma at 6.5 (M=40) to 9.2 (M=300) Gmma/s out of a ~12.4 Gmma/s machine
  peak, so outside those two M values there is nothing structural left to take.
* This kernel's inner loop tops out around 5-8 Gmma/s.  The gap is the
  dequantise-and-stage phase, which does not overlap the mma: a GPU thread runs
  serially, so staging only hides behind *other* threadgroups on the same core,
  and every tiling that gives a thread enough mma work to amortise staging also
  costs occupancy.  Double buffering the tiles made this worse, not better -
  it doubles threadgroup memory and halves the threadgroups resident per core.
* With N = 1024 there are only N/BN threadgroups, which starves the GPU; the
  sweep answers with tiny M tiles (BM = 8) that re-stream the weights once per
  8 rows.  Split-K would fix the parallelism but costs an extra kernel launch,
  ~10-14 us here, which is most of the margin.

Things that measurably mattered, in order: manual two-element fragment loads
instead of ``simdgroup_load`` (up to 1.4x - ``simdgroup_load`` from threadgroup
memory costs about as much as an mma), single-buffered tiles with the next K
block prefetched into registers, one scale/bias pair and one wide load per
thread in the weight loader, and writing the accumulators straight to device
memory from ``thread_elements()`` instead of via a threadgroup scratch.
Things that did not: a k-major weight tile (contiguous B fragments but strided
dequant stores), deeper prefetch, and swapping the mma operands so the weights
are the contiguous one (helps only a couple of tile shapes, kept as ``SW``).
"""
from __future__ import annotations

import mlx.core as mx

_HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

# TM, TN : 8x8 output tiles per threadgroup (BM = 8*TM rows, BN = 8*TN cols)
# WM, WN : simdgroup grid inside the threadgroup (WM*WN simdgroups);
#          each simdgroup owns RM = TM/WM by RN = TN/WN tiles, strided.
# BK     : K elements staged per iteration.
# XT     : 1 stages x in threadgroup memory, 0 builds the x fragments straight
#          from device memory (x is small and stays in cache).
# SW     : 1 feeds the weights in as the mma A operand (cheap contiguous
#          fragment loads) and x as B; better whenever RN > RM.
#
# One threadgroup walks K once for its BM x BN output tile.  Threadgroup memory
# is single buffered - doubling it halves the number of threadgroups resident
# per core, which costs more than the latency it hides - and the device loads
# for block i+1 are issued into registers before the mma of block i, so the
# weight load latency is covered without spending more threadgroup memory.
_SRC = r"""
    constexpr int BM = 8 * TM;
    constexpr int BN = 8 * TN;
    constexpr int RM = TM / WM;
    constexpr int RN = TN / WN;
    constexpr int BKp = BK + 8;                 // padded row stride, halves
    constexpr int NT = WM * WN * 32;            // threads per threadgroup
    constexpr int KV = BK / 4;                  // half4 / uint32 units per row
    constexpr int WPT = (BN * KV >= NT) ? (BN * KV) / NT : 1;  // weight words per thread
    constexpr int XPT = XT ? (BM * KV + NT - 1) / NT : 1;

    const int M = x_shape[0];
    const int K = x_shape[1];
    const int N = w_shape[0];
    const int KW = K >> 2;                      // uint32 words per weight row
    const int G = K >> 6;                       // groups per weight row

    const int tid = int(thread_position_in_threadgroup.x);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int lane = int(thread_index_in_simdgroup);
    const int sgm = sg / WN;
    const int sgn = sg % WN;
    const int n0 = int(threadgroup_position_in_grid.x) * BN;
    const int m0 = int(threadgroup_position_in_grid.y) * BM;

    // fragment coordinates of this lane inside an 8x8 simdgroup matrix
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;

    threadgroup half xs[XT ? BM * BKp : 1];
    threadgroup half ws[BN * BKp];

    simdgroup_float8x8 acc[RM * RN];
    #pragma clang loop unroll(full)
    for (int i = 0; i < RM * RN; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // per-thread staging slots, fixed for the whole K walk.  Each thread owns
    // WPT *consecutive* weight words, so they all sit in one row of W and share
    // one scale/bias pair and can be pulled with a single wide load.
    const int wbase = tid * WPT;
    const int wrow = (wbase < BN * KV) ? wbase / KV : -1;
    const int wcol = (wrow >= 0) ? wbase - wrow * KV : 0;
    const device uint* wp = w + (n0 + max(wrow, 0)) * KW + wcol;
    const device half* scp = scales + (n0 + max(wrow, 0)) * G;
    const device half* bip = biases + (n0 + max(wrow, 0)) * G;
    int xm[XPT], xk[XPT], xr[XPT];
    #pragma clang loop unroll(full)
    for (int p = 0; p < XPT; ++p) {
        const int i = tid + p * NT;
        const int r = (XT && i < BM * KV) ? i / KV : -1;
        xr[p] = r;
        xm[p] = (r >= 0 && m0 + r < M) ? r : -1;
        xk[p] = (r >= 0) ? i - r * KV : 0;
    }

    uint wv[WPT]; half wsc[1], wbi[1]; half4 xv[XPT];

#define OMNI_LOAD(KB)                                                            \
    {                                                                            \
        if (wrow >= 0) {                                                         \
            const device uint* wq = wp + ((KB) >> 2);                            \
            if (WPT == 4) {                                                      \
                const uint4 v = *((const device uint4*)wq);                      \
                wv[0] = v.x; wv[1] = v.y; wv[2] = v.z; wv[3] = v.w;              \
            } else if (WPT == 2) {                                               \
                const uint2 v = *((const device uint2*)wq);                      \
                wv[0] = v.x; wv[1] = v.y;                                        \
            } else {                                                             \
                _Pragma("clang loop unroll(full)")                                \
                for (int p = 0; p < WPT; ++p) wv[p] = wq[p];                     \
            }                                                                    \
            wsc[0] = scp[(KB) >> 6];                                             \
            wbi[0] = bip[(KB) >> 6];                                             \
        }                                                                        \
        if (XT) {                                                                \
            _Pragma("clang loop unroll(full)")                                    \
            for (int p = 0; p < XPT; ++p) {                                      \
                xv[p] = (xm[p] >= 0)                                             \
                    ? *((const device half4*)(x + (m0 + xm[p]) * K + (KB)) + xk[p]) \
                    : half4(0.0h);                                               \
            }                                                                    \
        }                                                                        \
    }

#define OMNI_STORE()                                                             \
    {                                                                            \
        if (wrow >= 0) {                                                         \
            const half s = wsc[0], b = wbi[0];                                   \
            threadgroup half4* d = (threadgroup half4*)(ws + wrow * BKp + wcol * 4); \
            _Pragma("clang loop unroll(full)")                                    \
            for (int p = 0; p < WPT; ++p) {                                      \
                const uint u = wv[p];                                            \
                const half4 q = half4(half(u & 0xFFu), half((u >> 8) & 0xFFu),   \
                                      half((u >> 16) & 0xFFu), half(u >> 24));   \
                d[p] = q * s + b;                                                \
            }                                                                    \
        }                                                                        \
        if (XT) {                                                                \
            _Pragma("clang loop unroll(full)")                                    \
            for (int p = 0; p < XPT; ++p) {                                      \
                if (xr[p] >= 0)                                                  \
                    *((threadgroup half4*)(xs + xr[p] * BKp + xk[p] * 4)) = xv[p]; \
            }                                                                    \
        }                                                                        \
    }

#define OMNI_MMA(KB)                                                             \
    if (SW) {                                                                    \
        /* weights are the A operand: one contiguous half2 per fragment, and     \
           x is the strided one.  Cheaper when RN > RM.  acc holds C^T. */       \
        _Pragma("clang loop unroll(full)")                                        \
        for (int kk = 0; kk < BK; kk += 8) {                                     \
            simdgroup_half8x8 wmat[RN];                                          \
            _Pragma("clang loop unroll(full)")                                    \
            for (int j = 0; j < RN; ++j) {                                       \
                const threadgroup half2* wp = (const threadgroup half2*)          \
                    (ws + ((j * WN + sgn) * 8 + fm) * BKp + kk + fn);            \
                reinterpret_cast<thread half2&>(wmat[j].thread_elements()) = *wp; \
            }                                                                    \
            _Pragma("clang loop unroll(full)")                                    \
            for (int t = 0; t < RM; ++t) {                                       \
                simdgroup_half8x8 xmat;                                          \
                const int mm = m0 + (t * WM + sgm) * 8 + fn;                     \
                if (XT) {                                                        \
                    const threadgroup half* xp =                                 \
                        xs + ((t * WM + sgm) * 8 + fn) * BKp + kk + fm;          \
                    reinterpret_cast<thread half2&>(xmat.thread_elements()) =    \
                        half2(xp[0], xp[BKp]);                                   \
                } else {                                                         \
                    const device half* xp = x + mm * K + (KB) + kk + fm;         \
                    reinterpret_cast<thread half2&>(xmat.thread_elements()) =    \
                        half2(mm < M ? xp[0] : half(0.0h),                       \
                              mm + 1 < M ? xp[K] : half(0.0h));                  \
                }                                                                \
                _Pragma("clang loop unroll(full)")                                \
                for (int j = 0; j < RN; ++j)                                     \
                    simdgroup_multiply_accumulate(acc[t * RN + j], wmat[j], xmat,\
                                                  acc[t * RN + j]);              \
            }                                                                    \
        }                                                                        \
    } else {                                                                     \
        _Pragma("clang loop unroll(full)")                                        \
        for (int kk = 0; kk < BK; kk += 8) {                                     \
            simdgroup_half8x8 bmat[RN];                                          \
            _Pragma("clang loop unroll(full)")                                    \
            for (int j = 0; j < RN; ++j) {                                       \
                const threadgroup half* bp =                                     \
                    ws + (j * WN + sgn) * 8 * BKp + kk + fm + fn * BKp;          \
                reinterpret_cast<thread half2&>(bmat[j].thread_elements()) =     \
                    half2(bp[0], bp[BKp]);                                       \
            }                                                                    \
            _Pragma("clang loop unroll(full)")                                    \
            for (int t = 0; t < RM; ++t) {                                       \
                simdgroup_half8x8 amat;                                          \
                if (XT) {                                                        \
                    const threadgroup half2* ap = (const threadgroup half2*)      \
                        (xs + (t * WM + sgm) * 8 * BKp + kk + fm * BKp + fn);    \
                    reinterpret_cast<thread half2&>(amat.thread_elements()) = *ap; \
                } else {                                                         \
                    const int mm = m0 + (t * WM + sgm) * 8 + fm;                 \
                    const device half2* ap = (const device half2*)               \
                        (x + mm * K + (KB) + kk + fn);                           \
                    reinterpret_cast<thread half2&>(amat.thread_elements()) =    \
                        (mm < M) ? *ap : half2(0.0h);                            \
                }                                                                \
                _Pragma("clang loop unroll(full)")                                \
                for (int j = 0; j < RN; ++j)                                     \
                    simdgroup_multiply_accumulate(acc[t * RN + j], amat, bmat[j],\
                                                  acc[t * RN + j]);              \
            }                                                                    \
        }                                                                        \
    }

    OMNI_LOAD(0)
    for (int kb = 0; kb < K; kb += BK) {
        OMNI_STORE()
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kb + BK < K) OMNI_LOAD(kb + BK)
        OMNI_MMA(kb)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (int t = 0; t < RM; ++t) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < RN; ++j) {
            const float2 v = reinterpret_cast<thread float2&>(acc[t * RN + j].thread_elements());
            if (SW) {
                // fragment row is n, fragment column is m
                const int m = m0 + (t * WM + sgm) * 8 + fn;
                const int n = n0 + (j * WN + sgn) * 8 + fm;
                if (m < M) out[m * N + n] = T(v.x);
                if (m + 1 < M) out[(m + 1) * N + n] = T(v.y);
            } else {
                const int m = m0 + (t * WM + sgm) * 8 + fm;
                if (m < M) {
                    device T* o = out + m * N + n0 + (j * WN + sgn) * 8 + fn;
                    o[0] = T(v.x);
                    o[1] = T(v.y);
                }
            }
        }
    }
"""

_CACHE: dict = {}


def _kernel():
    if "k" not in _CACHE:
        _CACHE["k"] = mx.fast.metal_kernel(
            name="omnivoice_qmm_sg",
            input_names=["x", "w", "scales", "biases"],
            output_names=["out"],
            source=_SRC,
            header=_HEADER,
            ensure_row_contiguous=True,
        )
    return _CACHE["k"]


def tg_bytes(TM: int, TN: int, WM: int, WN: int, BK: int, XT: int = 1, SW: int = 0) -> int:
    return 8 * ((TM if XT else 0) + TN) * (BK + 8) * 2


def valid(TM: int, TN: int, WM: int, WN: int, BK: int, XT: int = 1, SW: int = 0) -> bool:
    NT = WM * WN * 32
    KV = BK // 4
    words = 8 * TN * KV
    if words >= NT and (words % NT or words // NT > KV):
        return False  # the consecutive-word loader needs whole rows per thread
    return (TM % WM == 0 and TN % WN == 0 and NT <= 1024
            and tg_bytes(TM, TN, WM, WN, BK, XT) <= 32000)


def qmm_sg(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, *,
           out_dtype: mx.Dtype = mx.float16,
           config: tuple[int, int, int, int, int, int, int] | None = None) -> mx.array:
    """x [M, K] fp16 @ dequant(w, scales, biases).T -> [M, N] (8 bit, group 64)."""
    M, K = x.shape
    N = w.shape[0]
    TM, TN, WM, WN, BK, XT, SW = config if config is not None else pick_config(M, K, N)
    return _kernel()(
        inputs=[x, w, scales, biases],
        template=[("T", out_dtype), ("TM", TM), ("TN", TN), ("WM", WM), ("WN", WN),
                  ("BK", BK), ("XT", XT), ("SW", SW)],
        grid=((N // (8 * TN)) * WM * WN * 32, (M + 8 * TM - 1) // (8 * TM), 1),
        threadgroup=(WM * WN * 32, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[out_dtype],
    )[0]


# Winners of the config sweep (bench/test_kernel_sg.py --sweep style search over
# 440 tile shapes at the four (K, N) this model uses).  Wide N and narrow N want
# different shapes: with N = 1024 there are only N/BN threadgroups, so small
# tiles that spread the work over more of the GPU win, while wide N can afford
# fat tiles that walk K once.
_TABLE = [
    # (M upper bound, wide N config, narrow N config)
    (40, (5, 8, 1, 8, 64, 1, 0), (1, 4, 1, 4, 64, 1, 0)),
    (96, (2, 8, 2, 4, 64, 1, 1), (2, 4, 2, 4, 64, 1, 0)),
    (160, (4, 4, 4, 1, 64, 0, 0), (4, 4, 4, 2, 64, 1, 1)),
    (1 << 30, (8, 4, 4, 1, 64, 0, 0), (8, 8, 2, 4, 64, 1, 0)),
]


def pick_config(M: int, K: int, N: int) -> tuple[int, int, int, int, int, int, int]:
    """Tile shape for these dimensions; see _TABLE."""
    for hi, wide, narrow in _TABLE:
        if M <= hi:
            TM, TN, WM, WN, BK, XT, SW = wide if N >= 2048 else narrow
            break
    TM = min(TM, max(1, (M + 7) // 8))
    while TM % WM:
        WM //= 2
        WN *= 2
    while N % (8 * TN) or TN % WN:
        TN //= 2
    while K % BK:
        BK //= 2
    return (TM, TN, WM, WN, BK, XT, SW)
