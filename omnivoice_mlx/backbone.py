"""Qwen3-0.6B bidirectional transformer for OmniVoice, in MLX.

Differences from a causal Qwen3:

* no causal mask: every token attends to every token of its own segment;
* inputs are embeddings (text + summed audio-codebook embeddings), never ids.

Two forward paths, same arithmetic:

``__call__`` (segments)
    One row ``x`` of shape ``[1, S, H]`` that is the concatenation of
    independent *segments* (conditional branch, unconditional branch, more
    sentences). GEMMs run on the packed row; attention is done per segment,
    so branches of different length share the big kernels without padding.
    A segment may carry a cached prefix ``(K, V)`` per layer plus a RoPE
    ``offset`` (Fast-dLLM-style prompt cache, see sampler.py), and ``keep``
    asks for the first ``keep`` roped K/V back for a later step.

``batched`` (rows)
    ``x`` of shape ``[B, T, H]``: B equal-length rows, each with its own RoPE
    offset and an optional shared prefix K/V that a boolean mask hides from
    the rows that must not see it. This is the cached step of the sampler
    (cond row with the prompt's K/V, uncond row without): one RoPE call for
    q, one for k, one SDPA, no per-segment slicing — about a third fewer
    kernels per layer, which is what a step at these sizes is made of.

Fused projections
    ``fuse_linears`` (model.py) concatenates q/k/v into one ``qkv_proj`` and
    gate/up into one ``gate_up_proj`` at load time (quantised weights included:
    affine groups run along the input dim, so stacking output rows is exact).
    Fewer, larger GEMMs; bit-identical results.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class BackboneConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 3072
    vocab_size: int = 151676
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    @classmethod
    def from_llm_config(cls, d: dict) -> "BackboneConfig":
        d = dict(d)
        if "rope_theta" not in d and isinstance(d.get("rope_parameters"), dict):
            d["rope_theta"] = d["rope_parameters"].get("rope_theta", cls.rope_theta)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Segment:
    """One independent attention span inside the packed row."""

    length: int          # fresh tokens in this segment
    offset: int = 0      # RoPE position of the first fresh token
    keep: int = 0        # hand back roped K/V of the first ``keep`` fresh positions


KV = tuple[mx.array, mx.array]  # [1, n_kv, P, D] each, RoPE already applied


class Attention(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.scale = cfg.head_dim ** -0.5
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, h, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_theta)

    def _qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """x [B, S, H] -> normed q [B, nh, S, D], k [B, nkv, S, D], v [B, nkv, S, D]."""
        B, S, _ = x.shape
        nh, nkv, D = self.n_heads, self.n_kv_heads, self.head_dim
        if "qkv_proj" in self:
            out = self.qkv_proj(x).reshape(B, S, nh + 2 * nkv, D).transpose(0, 2, 1, 3)
            q, k, v = out[:, :nh], out[:, nh:nh + nkv], out[:, nh + nkv:]
        else:
            q = self.q_proj(x).reshape(B, S, nh, D).transpose(0, 2, 1, 3)
            k = self.k_proj(x).reshape(B, S, nkv, D).transpose(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, S, nkv, D).transpose(0, 2, 1, 3)
        return self.q_norm(q), self.k_norm(k), v

    def __call__(self, x: mx.array, segments: list[Segment],
                 prefix: list[KV | None] | None = None,
                 keep_out: dict[int, KV] | None = None) -> mx.array:
        S = x.shape[1]
        q, k, v = self._qkv(x)
        outs = []
        start = 0
        single = len(segments) == 1
        for i, seg in enumerate(segments):
            end = start + seg.length
            qs = q if single else q[:, :, start:end]
            ks = k if single else k[:, :, start:end]
            vs = v if single else v[:, :, start:end]
            qs = self.rope(qs, offset=seg.offset)
            ks = self.rope(ks, offset=seg.offset)
            if keep_out is not None and seg.keep:
                keep_out[i] = (ks[:, :, :seg.keep], vs[:, :, :seg.keep])
            if prefix is not None and prefix[i] is not None:
                pk, pv = prefix[i]
                ks = mx.concatenate([pk, ks], axis=2)
                vs = mx.concatenate([pv, vs], axis=2)
            outs.append(mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=self.scale, mask=None))
            start = end
        out = outs[0] if single else mx.concatenate(outs, axis=2)
        out = out.transpose(0, 2, 1, 3).reshape(1, S, -1)
        return self.o_proj(out)

    def batched(self, x: mx.array, offsets: mx.array, prefix: KV | None, mask: mx.array | None) -> mx.array:
        """x [B, T, H], offsets [B] int32, prefix (K, V) [1 or B, nkv, P, D] (shared or per row),
        mask [B, 1, T, P+T] bool (True = attend) or None."""
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.rope.base, scale=1.0, offset=offsets)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.rope.base, scale=1.0, offset=offsets)
        if prefix is not None:
            pk, pv = prefix
            if pk.shape[0] != B:
                pk = mx.broadcast_to(pk, (B, *pk.shape[1:]))
                pv = mx.broadcast_to(pv, (B, *pv.shape[1:]))
            k = mx.concatenate([pk, k], axis=2)
            v = mx.concatenate([pv, v], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, T, -1))


def _swiglu_impl(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up


# shapeless compile fuses silu and the product into one kernel; the split stays outside
# (shapeless tracing cannot infer split shapes)
_swiglu = mx.compile(_swiglu_impl, shapeless=True)


class MLP(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if "gate_up_proj" in self:
            gate, up = mx.split(self.gate_up_proj(x), 2, axis=-1)
            return self.down_proj(_swiglu(gate, up))
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x, segments, prefix=None, keep_out=None):
        h = x + self.self_attn(self.input_layernorm(x), segments, prefix, keep_out)
        return h + self.mlp(self.post_attention_layernorm(h))

    def batched(self, x, offsets, prefix, mask):
        h = x + self.self_attn.batched(self.input_layernorm(x), offsets, prefix, mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class Backbone(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def __call__(self, x: mx.array, segments: list[Segment] | None = None,
                 prefix: list[list[KV | None]] | None = None,
                 keep: bool = False) -> tuple[mx.array, list[dict[int, KV]] | None]:
        """``x`` [1, S, H] -> (normed hidden [1, S, H], kept K/V per layer or None)."""
        if segments is None:
            segments = [Segment(x.shape[1])]
        kept: list[dict[int, KV]] | None = [] if keep else None
        for li, layer in enumerate(self.layers):
            layer_keep: dict[int, KV] | None = {} if keep else None
            x = layer(x, segments, None if prefix is None else prefix[li], layer_keep)
            if kept is not None:
                kept.append(layer_keep)  # type: ignore[arg-type]
        return self.norm(x), kept

    def batched(self, x: mx.array, offsets: mx.array, prefix: list[KV] | None,
                mask: mx.array | None) -> mx.array:
        """``x`` [B, T, H] equal-length rows -> normed hidden [B, T, H]; ``prefix[layer]`` shared K/V."""
        for li, layer in enumerate(self.layers):
            x = layer.batched(x, offsets, None if prefix is None else prefix[li], mask)
        return self.norm(x)
