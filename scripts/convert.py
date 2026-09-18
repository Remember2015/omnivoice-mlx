"""Write an MLX-ready OmniVoice directory: cast (bf16 / fp16) and optionally
quantise the backbone, keep the heads in their own dtype, link the codec.

    .venv/bin/python scripts/convert.py --out models/mlx-bf16
    .venv/bin/python scripts/convert.py --out models/mlx-q8 --bits 8
    .venv/bin/python scripts/convert.py --out models/mlx-q4 --bits 4 --quantize-embed

The output loads with ``load_model(dir)``: ``config.json`` carries a
``quantization`` block when quantised (modules found by their ``.scales``
weight, mlx-lm convention) and ``dtype``. ``audio_tokenizer`` and
``tokenizer.json`` are symlinks to the source checkpoint, so the codec weights
are not duplicated (806 MB, fp32).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from omnivoice_mlx.model import load_model, model_bytes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "models/k2-fsa-OmniVoice"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--head-dtype", default="float32", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--bits", type=int, default=0, choices=[0, 4, 6, 8])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--quantize-embed", action="store_true", help="also quantise the 151k text embedding")
    ap.add_argument("--quantize-heads", action="store_true", help="also quantise audio embeddings + heads")
    ap.add_argument("--codec", default="float16", choices=["float16", "float32", "none"],
                    help="also write the decode-only codec (codec-decoder-<dtype>.safetensors, 45 MB fp16)")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(src, dtype=args.dtype, bits=args.bits, group_size=args.group_size,
                       quantize_embed=args.quantize_embed, quantize_heads=args.quantize_heads,
                       head_dtype=args.head_dtype)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out / "model.safetensors"), weights)

    cfg = json.loads((src / "config.json").read_text())
    cfg["dtype"] = args.dtype
    cfg["head_dtype"] = args.head_dtype
    if args.bits:
        cfg["quantization"] = {"group_size": args.group_size, "bits": args.bits}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    for name in ("tokenizer.json", "tokenizer_config.json", "audio_tokenizer"):
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(src / name, out), link)
    print(f"{out}: {model_bytes(model) / 1e6:.0f} MB in {len(weights)} tensors "
          f"({args.dtype}, bits={args.bits}, heads {args.head_dtype})")
    if args.codec != "none":
        from omnivoice_mlx.codec import write_slim_decoder
        slim = write_slim_decoder(out, args.codec)
        print(f"  codec decoder: {slim.name} ({slim.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
