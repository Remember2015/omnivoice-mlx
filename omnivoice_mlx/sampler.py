"""Iterative unmasking (the OmniVoice sampler) in MLX.

Port of ``OmniVoice._generate_iterative`` / ``_predict_tokens_with_scoring``
(official, torch), verified token-for-token against it in float32
(bench/parity_mlx.py), plus the knobs the bench studies:

* ``cache_refresh``  — Fast-dLLM-style prefix KV cache: recompute the prompt's
                       K/V every n steps and, in between, run only the target
                       tokens through the transformer (0 = off, exact);
* ``uncond_every``   — in cached steps recompute the unconditional CFG branch
                       only every n-th step and reuse its logits in between
                       (1 = official);
* ``cfg_until``      — run the unconditional branch only for the first fraction
                       of steps (1.0 = every step, as official);
* ``conf_threshold`` — reveal, on top of the schedule, every still-masked slot
                       whose top probability exceeds the threshold, stop when
                       nothing is masked (0 = off). One host sync per step;
* ``fast_path``      — cached steps as padded rows through backbone.batched
                       (one masked SDPA per layer whatever the batch size);
                       False = the per-segment path.

Batching: any number of utterances go through one loop. Refresh steps pack
them as segments (no padding, per-segment attention); cached steps put every
row (cond and uncond of every utterance) into one ``[R, T_max, H]`` batch,
padded to the longest, with a boolean mask that hides the padding and each
row's prompt K/V, so the GEMMs run at M = R * T_max and attention is one
kernel. Pad waste is Σ(T_max − T_i); pipeline.generate_batch sorts by length
and buckets to keep it small.

Per step the graph is built lazily and handed to the GPU with
``mx.async_eval``; the host goes on to build the next step on the still
unevaluated tokens. One blocking eval at the end.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .backbone import Segment
from .model import OmniVoice


@dataclass
class SamplerConfig:
    """Defaults are this repo's pick (2026-09-09, README §11–12): 16 steps, the
    prompt K/V refreshed every 8, the unconditional branch recomputed every 3
    cached steps — same CER / speaker similarity / UTMOS as the official 32
    steps at a quarter of the cost. ``SamplerConfig(num_steps=32,
    cache_refresh=0, uncond_every=1)`` is the official sampler exactly."""

    num_steps: int = 16
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    cfg_until: float = 1.0
    cache_refresh: int = 8
    conf_threshold: float = 0.0
    sync_every_step: bool = False
    fast_path: bool = True
    uncond_every: int = 3

    def tag(self) -> str:
        parts = [f"s{self.num_steps}"]
        if not self.fast_path:
            parts.append("slow")
        if self.cfg_until != 1.0:
            parts.append(f"cfg{self.cfg_until:g}")
        if self.cache_refresh:
            parts.append(f"kv{self.cache_refresh}")
        if self.conf_threshold:
            parts.append(f"th{self.conf_threshold:g}")
        if self.uncond_every > 1:
            parts.append(f"ue{self.uncond_every}")
        return "-".join(parts)


@dataclass
class Prompt:
    """Everything before the target: style + text (+ reference audio tokens)."""

    ids: mx.array         # [P, C] int32
    audio_mask: mx.array  # [P] bool

    @property
    def length(self) -> int:
        return int(self.ids.shape[0])


@dataclass
class UnmaskStats:
    steps: int = 0                    # steps actually run (max over items when batched)
    forwards: int = 0
    tokens_through_backbone: int = 0  # sum over forwards of the rows the backbone processed


def time_steps(num_step: int, t_shift: float) -> list[float]:
    ts = [i / num_step for i in range(num_step + 1)]
    return [t_shift * t / (1.0 + (t_shift - 1.0) * t) for t in ts]


def reveal_schedule(total: int, num_step: int, t_shift: float) -> list[int]:
    """How many of ``total`` masked slots to reveal at each step (official bookkeeping)."""
    ts = time_steps(num_step, t_shift)
    rem = total
    sched = []
    for step in range(num_step):
        if step == num_step - 1:
            num = rem
        else:
            num = min(math.ceil(total * (ts[step + 1] - ts[step])), rem)
        sched.append(int(num))
        rem -= int(num)
    return sched


# --- small elementwise chains, compiled once (shapeless: no shape is baked in) ---

def _cfg_mix_impl(c: mx.array, u: mx.array, g: mx.array) -> mx.array:
    # official: log_softmax(c_lp + g (c_lp - u_lp)) with c_lp/u_lp log-softmaxed; per position
    # that is (1+g) c - g u minus a constant that the final log_softmax removes.
    return (1.0 + g) * c - g * u


def _gumbel_apply_impl(x: mx.array, u: mx.array, inv_temp: mx.array) -> mx.array:
    return x * inv_temp - mx.log(-mx.log(u + 1e-10) + 1e-10)


_cfg_mix = mx.compile(_cfg_mix_impl, shapeless=True)
_gumbel_apply = mx.compile(_gumbel_apply_impl, shapeless=True)


def _gumbel(x: mx.array, temperature: float) -> mx.array:
    return _gumbel_apply(x, mx.random.uniform(shape=x.shape), mx.array(1.0 / temperature, dtype=x.dtype))


def _filter_top_k(log_probs: mx.array, ratio: float = 0.1) -> mx.array:
    V = log_probs.shape[-1]
    k = math.ceil(ratio * V)
    kth = mx.sort(log_probs, axis=-1)[..., V - k: V - k + 1]
    return mx.where(log_probs >= kth, log_probs, mx.array(-float("inf")))


class _Item:
    """Per-utterance state of the loop."""

    __slots__ = ("prompt", "T", "prefix_emb", "tokens", "sched", "cache", "done", "steps", "u_lg")

    def __init__(self, model: OmniVoice, prompt: Prompt, T: int, cfg: SamplerConfig):
        C = model.config.num_audio_codebook
        self.prompt = prompt
        self.T = T
        self.prefix_emb = model.embed(prompt.ids, prompt.audio_mask)  # [1, P, H], constant
        self.tokens = mx.full((T, C), model.config.audio_mask_id, dtype=mx.int32)
        self.sched = reveal_schedule(T * C, cfg.num_steps, cfg.t_shift)
        self.cache: list | None = None  # per layer (K, V) [1, nkv, P, D] of the prompt positions
        self.done = False
        self.steps = 0
        self.u_lg: mx.array | None = None  # last unconditional logits, for uncond_every > 1


class _BatchCache:
    """The active items' prompt K/V padded to P_max and stacked per layer, once per refresh.

    Rows: the cond rows of every item (``kv_c``), and the same followed by zero
    prefixes for the uncond rows (``kv_cu``), with matching RoPE offsets and
    attention masks. Built lazily (arrays stay unevaluated until the next step).
    """

    def __init__(self, items: list[_Item], n_layers: int):
        self.items = items
        n = len(items)
        self.T_max = max(it.T for it in items)
        self.P_max = max(it.prompt.length for it in items)
        T_max, P_max = self.T_max, self.P_max
        kv_c, kv_cu = [], []
        for li in range(n_layers):
            ks, vs = [], []
            for it in items:
                k, v = it.cache[li]  # type: ignore[index]
                pad = P_max - k.shape[2]
                if pad:
                    k = mx.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)])
                    v = mx.pad(v, [(0, 0), (0, 0), (0, pad), (0, 0)])
                ks.append(k)
                vs.append(v)
            K = mx.concatenate(ks, axis=0) if n > 1 else ks[0]
            V = mx.concatenate(vs, axis=0) if n > 1 else vs[0]
            kv_c.append((K, V))
            z = mx.zeros_like(K)
            kv_cu.append((mx.concatenate([K, z], axis=0), mx.concatenate([V, z], axis=0)))
        self.kv_c, self.kv_cu = kv_c, kv_cu
        self.off_c = mx.array([it.prompt.length for it in items], dtype=mx.int32)
        self.off_cu = mx.concatenate([self.off_c, mx.zeros((n,), dtype=mx.int32)])
        # masks [rows, 1, T_max, P_max + T_max]: cond row i sees keys < P_i and targets < T_i,
        # uncond row i only targets < T_i; padded queries use the same row (finite softmax)
        key = mx.arange(P_max + T_max)
        rows_c, rows_u = [], []
        for it in items:
            P, T = it.prompt.length, it.T
            tgt_ok = (key >= P_max) & (key < P_max + T)
            rows_c.append(((key < P) | tgt_ok)[None, None, None, :])
            rows_u.append(tgt_ok[None, None, None, :])
        mc = mx.concatenate(rows_c, axis=0) if n > 1 else rows_c[0]
        mu = mx.concatenate(rows_u, axis=0) if n > 1 else rows_u[0]
        self.mask_c = mx.broadcast_to(mc, (n, 1, T_max, P_max + T_max))
        self.mask_cu = mx.broadcast_to(mx.concatenate([mc, mu], axis=0), (2 * n, 1, T_max, P_max + T_max))
        self.needs_mask_c = n > 1 or any(it.T != T_max or it.prompt.length != P_max for it in items)


def unmask_batch(model: OmniVoice, items: list[tuple[Prompt, int]], cfg: SamplerConfig, *,
                 seed: int | None = None, stats: UnmaskStats | None = None) -> list[mx.array]:
    """Generate audio tokens ``[T_i, C]`` for every ``(prompt, T_i)`` in one loop."""
    C = model.config.num_audio_codebook
    V = model.config.audio_vocab_size
    MASK = model.config.audio_mask_id
    H = model.config.llm.hidden_size
    n_layers = model.config.llm.num_hidden_layers
    if seed is not None:
        mx.random.seed(seed)
    stats = stats if stats is not None else UnmaskStats()

    penalty = (mx.arange(C, dtype=mx.float32) * cfg.layer_penalty_factor)[None, :]  # [1, C]
    mask_col = mx.arange(V) == MASK
    neg_inf = mx.array(-float("inf"))
    log_thresh = math.log(cfg.conf_threshold) if cfg.conf_threshold > 0 else None
    cfg_steps = int(round(cfg.num_steps * cfg.cfg_until)) if cfg.guidance_scale != 0 else 0

    state = [_Item(model, p, T, cfg) for p, T in items]
    bcache: _BatchCache | None = None

    for step in range(cfg.num_steps):
        active = [it for it in state if not it.done and it.sched[step] > 0]
        if not active:
            continue
        use_cfg = step < cfg_steps
        refresh = cfg.cache_refresh > 0 and step % cfg.cache_refresh == 0
        use_cache = cfg.cache_refresh > 0 and not refresh
        stale_u = use_cfg and cfg.uncond_every > 1 and use_cache and step % cfg.uncond_every != 0
        run_u = use_cfg and not stale_u

        # Padded rows beat the segment path only while padding is cheap: with one or two
        # utterances (or equal lengths). Measured on the 20-sentence set (README §16):
        # B=1 0.106 either way, B=2 0.093 either way, B=8 padded 0.085 vs segments 0.081.
        padded = cfg.fast_path and use_cache and (len(active) <= 2 or len({it.T for it in active}) == 1)
        if padded:
            # ---- cached step: padded rows [R, T_max, H], one masked SDPA per layer
            if bcache is None or bcache.items != active:
                bcache = _BatchCache(active, n_layers)
            n, T_max = len(active), bcache.T_max
            embs = []
            for it in active:
                e = model.embed_audio(it.tokens)  # [1, T, H]
                if it.T < T_max:
                    e = mx.pad(e, [(0, 0), (0, T_max - it.T), (0, 0)])
                embs.append(e)
            x_c = mx.concatenate(embs, axis=0) if n > 1 else embs[0]
            if run_u:
                x = mx.concatenate([x_c, x_c], axis=0)
                hidden = model.backbone.batched(x, bcache.off_cu, bcache.kv_cu, bcache.mask_cu)
            else:
                hidden = model.backbone.batched(x_c, bcache.off_c, bcache.kv_c,
                                                bcache.mask_c if bcache.needs_mask_c else None)
            R = hidden.shape[0]
            stats.forwards += 1
            stats.tokens_through_backbone += R * T_max
            lg = model.logits(hidden.reshape(1, R * T_max, H))[0]  # [R*T_max, C, V]
            pairs = []
            for i, it in enumerate(active):
                c = lg[i * T_max: i * T_max + it.T]
                u = lg[(n + i) * T_max: (n + i) * T_max + it.T] if run_u else None
                pairs.append((c, u))
            _reveal(cfg, step, active, pairs, penalty, mask_col, neg_inf, log_thresh, MASK, C, stale=stale_u)
            continue

        # ---- refresh step (or fast_path off): segments, no padding
        bcache = None
        parts, segs, prefix_l, slices = [], [], [], []
        pos = 0
        for it in active:
            P, T = it.prompt.length, it.T
            tgt = model.embed_audio(it.tokens)
            if use_cache:
                parts.append(tgt)
                segs.append(Segment(T, offset=P))
                prefix_l.append(it.cache)
                c_slice = (pos, pos + T)
                pos += T
            else:
                parts += [it.prefix_emb, tgt]
                segs.append(Segment(P + T, keep=P if cfg.cache_refresh else 0))
                prefix_l.append(None)
                c_slice = (pos + P, pos + P + T)
                pos += P + T
            u_slice = None
            if run_u:
                parts.append(tgt)
                segs.append(Segment(T))
                prefix_l.append(None)
                u_slice = (pos, pos + T)
                pos += T
            slices.append((c_slice, u_slice))
        x = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        if use_cache:
            prefix = [[None if kv is None else kv[li] for kv in prefix_l] for li in range(n_layers)]
            hidden, _ = model.backbone(x, segs, prefix=prefix)
        else:
            hidden, kept = model.backbone(x, segs, keep=bool(cfg.cache_refresh))
            if cfg.cache_refresh:
                si = 0
                for it in active:
                    it.cache = [layer[si] for layer in kept]  # type: ignore[index]
                    si += 2 if run_u else 1
        stats.forwards += 1
        stats.tokens_through_backbone += int(x.shape[1])
        rows = []
        for (c0, c1), u in slices:
            rows.append(hidden[:, c0:c1])
            if u is not None:
                rows.append(hidden[:, u[0]:u[1]])
        lg = model.logits(mx.concatenate(rows, axis=1))[0]  # [sum, C, V]
        pairs, off = [], 0
        for it, (_, u) in zip(active, slices):
            c = lg[off: off + it.T]
            off += it.T
            uu = None
            if u is not None:
                uu = lg[off: off + it.T]
                off += it.T
            pairs.append((c, uu))
        _reveal(cfg, step, active, pairs, penalty, mask_col, neg_inf, log_thresh, MASK, C, stale=stale_u)

    results = []
    for it in state:
        results.append(mx.where(it.tokens == MASK, mx.zeros_like(it.tokens), it.tokens))
    mx.eval(*results)
    stats.steps = max(it.steps for it in state)
    return results


def _reveal(cfg, step, active, pairs, penalty, mask_col, neg_inf, log_thresh, MASK, C, *, stale=False):
    """Score every active item's logits and reveal its k slots for this step.
    ``pairs``: per item (cond logits [T, C, V], uncond logits or None); ``stale``
    means reuse the item's last uncond logits."""
    g = mx.array(cfg.guidance_scale, dtype=mx.float32)
    outs = []
    for it, (c_lg, u_lg) in zip(active, pairs):
        T = it.T
        if u_lg is not None:
            if cfg.uncond_every > 1:
                it.u_lg = u_lg
            lp = nn.log_softmax(_cfg_mix(c_lg, u_lg, g), axis=-1)
        elif stale and it.u_lg is not None:
            lp = nn.log_softmax(_cfg_mix(c_lg, it.u_lg, g), axis=-1)
        else:
            lp = nn.log_softmax(c_lg, axis=-1)
        lp = mx.where(mask_col, neg_inf, lp)
        if cfg.class_temperature > 0.0:
            pred = mx.argmax(_gumbel(_filter_top_k(lp), cfg.class_temperature), axis=-1)
        else:
            pred = mx.argmax(lp, axis=-1)  # [T, C]
        conf = mx.max(lp, axis=-1)          # [T, C]
        masked = it.tokens == MASK
        scores = conf - penalty
        if cfg.position_temperature > 0.0:
            scores = _gumbel(scores, cfg.position_temperature)
        scores = mx.where(masked, scores, neg_inf)
        k = it.sched[step]
        if log_thresh is not None:
            n_conf = int(mx.sum((conf > log_thresh) & masked).item())
            n_masked = int(mx.sum(masked).item())
            k = min(max(k, n_conf), n_masked)
            if k >= n_masked:
                it.done = True
            if k <= 0:
                it.done = True
                continue
        flat_scores = scores.reshape(-1)
        idx = mx.argpartition(-flat_scores, kth=k - 1)[:k] if k < T * C else mx.arange(T * C)
        flat = mx.put_along_axis(it.tokens.reshape(-1), idx,
                                 mx.take(pred.reshape(-1).astype(mx.int32), idx), axis=0)
        it.tokens = flat.reshape(T, C)
        it.steps += 1
        outs.append(it.tokens)
    if cfg.sync_every_step or log_thresh is not None:
        mx.eval(*outs)
    else:
        mx.async_eval(*outs)


def unmask(model: OmniVoice, prompt: Prompt, T: int, cfg: SamplerConfig, *,
           seed: int | None = None, stats: UnmaskStats | None = None) -> mx.array:
    """Single utterance: the generated audio tokens ``[T, C]`` (int32)."""
    return unmask_batch(model, [(prompt, T)], cfg, seed=seed, stats=stats)[0]
