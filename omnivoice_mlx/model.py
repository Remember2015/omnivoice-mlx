"""OmniVoice (k2-fsa) in MLX: audio-codebook embeddings, the Qwen3 backbone,
the 8 prediction heads, and loading (cast / quantize) straight from the
k2-fsa safetensors or from a directory written by ``scripts/convert.py``.

Layout follows the official checkpoint: one ``audio_embeddings`` table of
``8 * 1025`` rows indexed with ``id + codebook * 1025``, one ``audio_heads``
Linear of ``8 * 1025`` outputs reshaped to ``[..., 8, 1025]``. The heads are
applied to the target positions only; per position they are independent, so
that is exact and saves the prompt positions' share (~1 % of a step).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .backbone import Backbone, BackboneConfig, Segment  # noqa: F401  (re-export)

DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


@dataclass
class OmniVoiceConfig:
    num_audio_codebook: int = 8
    audio_vocab_size: int = 1025
    audio_mask_id: int = 1024
    sample_rate: int = 24000
    hop_length: int = 960
    llm: BackboneConfig = field(default_factory=BackboneConfig)
    quantization: dict | None = None  # present when the weights on disk are quantized

    @classmethod
    def from_dir(cls, path: Path) -> "OmniVoiceConfig":
        raw = json.loads((path / "config.json").read_text())
        return cls(
            num_audio_codebook=raw.get("num_audio_codebook", 8),
            audio_vocab_size=raw.get("audio_vocab_size", 1025),
            audio_mask_id=raw.get("audio_mask_id", 1024),
            llm=BackboneConfig.from_llm_config(raw.get("llm_config", {})),
            quantization=raw.get("quantization"),
        )

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length  # 25 Hz


class _Const:
    """Arrays a Module must not treat as parameters (MLX walks every mx.array attribute).

    Materialised here: ``mx.eval(model.parameters())`` never sees them, and a
    lazy array built on one thread cannot be evaluated on another (MLX 0.32:
    "There is no Stream(gpu, 0) in current thread") — a service that builds the
    engine on its main thread and synthesises on a worker thread would fail on
    the first clause (bench/test_thread.py)."""

    __slots__ = ("offsets",)

    def __init__(self, offsets: mx.array):
        mx.eval(offsets)
        self.offsets = offsets


class OmniVoice(nn.Module):
    def __init__(self, config: OmniVoiceConfig, head_dtype: mx.Dtype = mx.float32):
        super().__init__()
        self.config = config
        C, V, H = config.num_audio_codebook, config.audio_vocab_size, config.llm.hidden_size
        self.backbone = Backbone(config.llm)
        self.audio_embeddings = nn.Embedding(C * V, H)
        self.audio_heads = nn.Linear(H, C * V, bias=False)
        self.head_dtype = head_dtype
        self._const = _Const(mx.arange(C, dtype=mx.int32) * V)

    # ---- embeddings -------------------------------------------------------
    def embed(self, ids: mx.array, audio_mask: mx.array) -> mx.array:
        """``ids`` [S, C] int32, ``audio_mask`` [S] bool -> [1, S, H] in the backbone dtype.

        Text positions use ``embed_tokens`` on codebook row 0 (the rows are
        copies); audio positions sum the 8 codebook embeddings. Same arithmetic
        as the official ``_prepare_embed_inputs``.
        """
        text = self.backbone.embed_tokens(ids[:, 0])
        shifted = ids * audio_mask[:, None].astype(ids.dtype) + self._const.offsets[None, :]
        audio = self.audio_embeddings(shifted).sum(axis=1)
        return mx.where(audio_mask[:, None], audio, text)[None]

    def embed_audio(self, ids: mx.array) -> mx.array:
        """All-audio positions: ``ids`` [S, C] -> [1, S, H]."""
        return self.audio_embeddings(ids + self._const.offsets[None, :]).sum(axis=1)[None]

    # ---- heads ------------------------------------------------------------
    def logits(self, hidden: mx.array) -> mx.array:
        """``hidden`` [..., H] -> [..., C, V] float32 logits over the audio vocabulary."""
        out = self.audio_heads(hidden.astype(self.head_dtype))
        C, V = self.config.num_audio_codebook, self.config.audio_vocab_size
        return out.reshape(*out.shape[:-1], C, V).astype(mx.float32)

    # ---- weights ----------------------------------------------------------
    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        out = {}
        for k, v in weights.items():
            if k == "codebook_layer_offsets":
                continue
            if k.startswith("llm."):
                k = "backbone." + k[4:]
            out[k] = v
        return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _quantize_predicate(quantize_embed: bool, quantize_heads: bool):
    def pred(path: str, module: nn.Module) -> bool:
        if not hasattr(module, "to_quantized"):
            return False
        if path.startswith("backbone.layers."):
            return isinstance(module, nn.Linear)
        if path == "backbone.embed_tokens":
            return quantize_embed
        if path in ("audio_embeddings", "audio_heads"):
            return quantize_heads
        return False
    return pred


class SmallMQuantizedLinear(nn.QuantizedLinear):
    """QuantizedLinear that routes small-M, wide-N calls to the custom simdgroup
    kernel (kernels_sg.qmm_sg) and everything else to mx.quantized_matmul.

    The custom kernel only wins where MLX pads M to 32 for nothing: 1.24–1.30x at
    M ≤ 40 and 1.02–1.04x at M ≈ 78 for N ∈ {4096, 6144}; it loses 10–15 %
    elsewhere (README §13). So: rows ≤ 80 and N ≥ 4096, 8-bit group 64 only.
    """

    max_rows: int = 80
    min_cols: int = 4096

    def __call__(self, x: mx.array) -> mx.array:
        rows = 1
        for d in x.shape[:-1]:
            rows *= d
        N = self.weight.shape[0]
        if (rows <= self.max_rows and N >= self.min_cols and self.bits == 8 and self.group_size == 64
                and x.dtype == mx.float16 and "bias" not in self):
            from .kernels_sg import qmm_sg
            y = qmm_sg(x.reshape(rows, x.shape[-1]), self.weight, self.scales, self.biases, out_dtype=x.dtype)
            return y.reshape(*x.shape[:-1], N)
        return super().__call__(x)


def _cat_linears(mods: list[nn.Module], custom_gemm: bool = False) -> nn.Module:
    """One Linear / QuantizedLinear whose output rows are the given ones stacked (exact)."""
    first = mods[0]
    if isinstance(first, nn.QuantizedLinear):
        cls = SmallMQuantizedLinear if custom_gemm else nn.QuantizedLinear
        out = cls.__new__(cls)
        nn.Module.__init__(out)
        out.group_size, out.bits, out.mode = first.group_size, first.bits, getattr(first, "mode", "affine")
        out.weight = mx.concatenate([m.weight for m in mods], axis=0)
        out.scales = mx.concatenate([m.scales for m in mods], axis=0)
        if "biases" in first:
            out.biases = mx.concatenate([m.biases for m in mods], axis=0)
        out.freeze()
        return out
    out = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(out)
    out.weight = mx.concatenate([m.weight for m in mods], axis=0)
    return out


def fuse_linears(model: "OmniVoice", custom_gemm: bool = False) -> None:
    """q/k/v -> qkv_proj and gate/up -> gate_up_proj in every layer (see backbone.py).
    ``custom_gemm`` routes the two wide projections through SmallMQuantizedLinear."""
    for layer in model.backbone.layers:
        att, mlp = layer.self_attn, layer.mlp
        if "qkv_proj" not in att:
            att.qkv_proj = _cat_linears([att.q_proj, att.k_proj, att.v_proj], custom_gemm)
            for name in ("q_proj", "k_proj", "v_proj"):
                del att[name]
        if "gate_up_proj" not in mlp:
            mlp.gate_up_proj = _cat_linears([mlp.gate_proj, mlp.up_proj], custom_gemm)
            for name in ("gate_proj", "up_proj"):
                del mlp[name]
    mx.eval(model.parameters())


def load_model(model_dir: str | Path, *, dtype: str = "bfloat16", bits: int = 0,
               group_size: int = 64, quantize_embed: bool = False, quantize_heads: bool = False,
               head_dtype: str = "float32", fuse: bool = True, custom_gemm: bool = False) -> OmniVoice:
    """Build the model and load weights.

    ``model_dir`` is either the k2-fsa checkpoint (fp32; cast to ``dtype`` and,
    with ``bits``, quantized on the fly) or a directory from
    ``scripts/convert.py`` whose config carries ``quantization`` (loaded as is;
    ``bits``/``dtype`` are then ignored). ``fuse`` concatenates q/k/v and gate/up
    after loading (exact, faster). Missing weights are an error, never a silent
    fallback.
    """
    model_dir = Path(model_dir)
    cfg = OmniVoiceConfig.from_dir(model_dir)
    weights_path = model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{weights_path} not found. Fetch with: huggingface-cli download k2-fsa/OmniVoice "
            f"--local-dir {model_dir}")
    model = OmniVoice(cfg, head_dtype=DTYPES[head_dtype])
    raw = OmniVoice.sanitize(dict(mx.load(str(weights_path))))

    if cfg.quantization is not None:
        q = cfg.quantization
        nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                    class_predicate=lambda p, m: f"{p}.scales" in raw)
        if "audio_heads.weight" in raw and raw["audio_heads.weight"].dtype != DTYPES[head_dtype]:
            raw["audio_heads.weight"] = raw["audio_heads.weight"].astype(DTYPES[head_dtype])
        model.load_weights(list(raw.items()))
    else:
        target = DTYPES[dtype]
        casted = {}
        for k, v in raw.items():
            if k == "audio_heads.weight":
                casted[k] = v.astype(DTYPES[head_dtype])
            elif v.dtype in (mx.float32, mx.float16, mx.bfloat16):
                casted[k] = v.astype(target)
            else:
                casted[k] = v
        model.load_weights(list(casted.items()))
        if bits:
            nn.quantize(model, group_size=group_size, bits=bits,
                        class_predicate=_quantize_predicate(quantize_embed, quantize_heads))
    mx.eval(model.parameters())
    if fuse:
        fuse_linears(model, custom_gemm=custom_gemm)
    return model


def model_bytes(model: nn.Module) -> int:
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))
