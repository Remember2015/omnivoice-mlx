"""Compare the MLX port with the official torch run saved by ``parity_ref.py``.

    .venv/bin/python bench/parity_mlx.py [--dtype float32|bfloat16|float16] [--bits 0|8|4]
                                         [--cache-refresh N] [--cfg-until F]

Three checks, each against ``out/parity``:
  1. reference prompt: our preprocessing + codec encode vs the official tokens;
  2. prompt ids: our style/text/ref assembly vs the official ``input_ids``;
  3. deterministic unmasking (position/class temperature 0, same T, same
     official ref tokens): token agreement per codebook and overall.
Then the wav is decoded from *our* tokens for listening. A float32 run should
agree almost everywhere; bf16 / quantised runs show how far the arithmetic
drifts, and the cache / CFG knobs show how far the approximation drifts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from cases import CASES, REF_TEXT, REF_WAV  # noqa: E402
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig, VoicePrompt  # noqa: E402
from omnivoice_mlx.audio import write_wav  # noqa: E402
from omnivoice_mlx.sampler import unmask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/k2-fsa-OmniVoice"))
    ap.add_argument("--ref", default=str(ROOT / "out/parity"))
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--head-dtype", default="float32")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cache-refresh", type=int, default=0)
    ap.add_argument("--cfg-until", type=float, default=1.0)
    ap.add_argument("--cases", nargs="*", default=["opener", "sentence", "long"])
    args = ap.parse_args()
    ref = Path(args.ref)

    tts = OmniVoiceTTS(args.model, dtype=args.dtype, bits=args.bits, head_dtype=args.head_dtype)
    tag = f"{args.dtype}{'-q' + str(args.bits) if args.bits else ''}"
    print(f"model {tag}  load {tts.load_model_s:.1f}s  codec {tts.load_codec_s:.1f}s")

    # 1. reference prompt
    off = np.load(ref / "ref_prompt.npz")
    ours = tts.make_prompt(REF_WAV, REF_TEXT)
    ot = np.asarray(ours.ref_tokens)
    print(f"ref: official {off['tokens'].shape} text={str(off['ref_text'])!r} rms={float(off['ref_rms']):.4f}")
    print(f"ref: ours     {ot.shape} text={ours.ref_text!r} rms={ours.ref_rms:.4f}")
    if ot.shape == off["tokens"].shape:
        agree = (ot == off["tokens"]).mean(axis=0)
        print("ref token agreement per codebook: " + " ".join(f"{a:.3f}" for a in agree))
    else:
        print("ref token length differs -> preprocessing differs")

    # use the official reference tokens from here on, so 2/3 isolate the model
    vp = VoicePrompt(mx.array(off["tokens"].astype(np.int32)), str(off["ref_text"]), float(off["ref_rms"]))
    for name in args.cases:
        text = dict(CASES)[name]
        d = np.load(ref / f"{name}.npz")
        p = tts.build_prompt(text, vp, language="zh")
        ids = np.asarray(p.ids)
        L = int(d["input_ids"].shape[0])
        T = int(d["T"])
        same_ids = ids.shape[0] == L - T and np.array_equal(ids, d["input_ids"][: L - T])
        est_T = tts.estimate_tokens(text, vp)
        print(f"{name:<9} prompt ids {'match' if same_ids else 'DIFFER'} (P={ids.shape[0]} vs {L - T})  "
              f"T official {T} ours {est_T}")
        cfg = SamplerConfig(num_steps=args.steps, position_temperature=0.0, class_temperature=0.0,
                            cache_refresh=args.cache_refresh, cfg_until=args.cfg_until)
        tok = np.asarray(unmask(tts.model, p, T, cfg))
        agree_cb = (tok == d["tokens"]).mean(axis=0)
        print(f"          token agreement {(tok == d['tokens']).mean():.4f}  per codebook "
              + " ".join(f"{a:.3f}" for a in agree_cb))
        wav = np.asarray(tts.codec.decode(mx.array(tok)))
        wav = tts.post_process(wav, vp.ref_rms)
        write_wav(ref / f"{name}-mlx-{tag}-{cfg.tag()}.wav", wav, 24000)


if __name__ == "__main__":
    main()
