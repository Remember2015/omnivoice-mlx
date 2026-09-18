"""Where does one unmasking step go? Host graph-building vs GPU execution,
attention vs MLP vs heads+sampling, at the row lengths the recommended config
sees (cached steps: 2T rows; refresh steps: P+2T).

    bench/benchlock.sh -- .venv/bin/python bench/profile_step.py --model models/mlx-q8-fp16
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from omnivoice_mlx.backbone import Segment  # noqa: E402
from omnivoice_mlx.model import load_model  # noqa: E402


def timed(fn, n=5):
    """(host build ms, gpu ms): build the graph, then eval; medians over n."""
    build, gpu = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        mx.eval(out)
        t2 = time.perf_counter()
        build.append((t1 - t0) * 1000)
        gpu.append((t2 - t1) * 1000)
    return statistics.median(build), statistics.median(gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/mlx-q8-fp16"))
    ap.add_argument("--P", type=int, default=130)
    ap.add_argument("--T", type=int, nargs="+", default=[39, 63, 126])
    args = ap.parse_args()
    m = load_model(args.model)
    H = m.config.llm.hidden_size
    L = m.config.llm.num_hidden_layers
    dt = m.backbone.layers[0].mlp.gate_proj.scales.dtype if hasattr(m.backbone.layers[0].mlp.gate_proj, "scales") \
        else m.backbone.layers[0].mlp.gate_proj.weight.dtype
    print(f"model dtype {dt}, {L} layers")

    for T in args.T:
        P = args.P
        x2 = mx.random.normal((1, 2 * T, H)).astype(dt)
        xr = mx.random.normal((1, P + 2 * T, H)).astype(dt)
        pk = mx.random.normal((1, 8, P, 128)).astype(dt)
        pv = mx.random.normal((1, 8, P, 128)).astype(dt)
        mx.eval(x2, xr, pk, pv)
        segs_c = [Segment(T, offset=P), Segment(T)]
        prefix = [[(pk, pv), None] for _ in range(L)]
        segs_r = [Segment(P + T, keep=P), Segment(T)]
        tokens = mx.zeros((T, 8), dtype=mx.int32)
        mx.eval(tokens)

        def cached():
            h, _ = m.backbone(x2, segs_c, prefix=prefix)
            return h

        def refresh():
            h, kept = m.backbone(xr, segs_r, keep=True)
            return [h] + [kv for layer in kept for kv in layer[0]]

        def one_layer_attn():
            lyr = m.backbone.layers[0]
            return lyr.self_attn(lyr.input_layernorm(x2), segs_c, prefix[0], None)

        def one_layer_mlp():
            lyr = m.backbone.layers[0]
            return lyr.mlp(lyr.post_attention_layernorm(x2))

        def all_mlp():
            h = x2
            for lyr in m.backbone.layers:
                h = h + lyr.mlp(lyr.post_attention_layernorm(h))
            return h

        def all_attn():
            h = x2
            for li, lyr in enumerate(m.backbone.layers):
                h = h + lyr.self_attn(lyr.input_layernorm(h), segs_c, prefix[li], None)
            return h

        def heads_and_sampling():
            hid = mx.random.normal((1, 2 * T, H)).astype(dt)
            lg = m.logits(hid)[0]
            c_lp = nn.log_softmax(lg[:T], axis=-1)
            u_lp = nn.log_softmax(lg[T:], axis=-1)
            lp = nn.log_softmax(c_lp + 2.0 * (c_lp - u_lp), axis=-1)
            lp = mx.where(mx.arange(1025) == 1024, mx.array(-float("inf")), lp)
            pred = mx.argmax(lp, axis=-1)
            conf = mx.max(lp, axis=-1) - (mx.arange(8, dtype=mx.float32) * 5.0)[None]
            u = mx.random.uniform(shape=conf.shape)
            scores = conf / 5.0 + (-mx.log(-mx.log(u + 1e-10) + 1e-10))
            scores = mx.where(tokens == 1024, scores, mx.array(-float("inf")))
            idx = mx.argpartition(-scores.reshape(-1), kth=T)[:T]
            flat = mx.put_along_axis(tokens.reshape(-1), idx, mx.take(pred.reshape(-1).astype(mx.int32), idx), axis=0)
            return flat

        def embed():
            return m.embed_audio(tokens)

        for _ in range(2):
            mx.eval(cached(), refresh(), all_mlp(), all_attn(), heads_and_sampling(), embed())
        rows = []
        for name, fn in [("cached step backbone (2T rows)", cached), ("refresh step backbone (P+2T)", refresh),
                         ("28x MLP only", all_mlp), ("28x attention only", all_attn),
                         ("1 layer attention", one_layer_attn), ("1 layer MLP", one_layer_mlp),
                         ("heads + CFG + sampling", heads_and_sampling), ("audio embedding", embed)]:
            b, g = timed(fn)
            rows.append((name, b, g))
        print(f"\nT={T}  P={P}  cached rows {2 * T}, refresh rows {P + 2 * T}")
        print(f"  {'stage':<32} {'host build':>10} {'gpu':>8}")
        for name, b, g in rows:
            print(f"  {name:<32} {b:10.2f} {g:8.2f}")


if __name__ == "__main__":
    main()
