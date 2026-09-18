"""Ground truth from the official torch implementation (CPU, fp32).

    .venv-ref/bin/python bench/parity_ref.py [--steps 8] [--out out/parity]

Writes, for the shared reference clip and each test sentence:
  ref_prompt.npz        reference audio tokens [Tr, 8], ref_text (punctuated), ref_rms
  <case>.npz            cond input_ids [L, 8], audio_mask [L], target T, generated tokens [T, 8]
  <case>.wav            decoded + post-processed audio
Generation is made deterministic (position_temperature=0, class_temperature=0)
so the MLX port can be compared token by token; the timing is a torch-CPU data
point only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cases import CASES, REF_TEXT, REF_WAV  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models/k2-fsa-OmniVoice"))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "out/parity"))
    ap.add_argument("--cases", nargs="*", default=["opener", "sentence", "long"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from omnivoice import OmniVoice, OmniVoiceGenerationConfig

    torch.manual_seed(0)
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained(args.model, device_map="cpu")
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    vp = model.create_voice_clone_prompt(ref_audio=str(REF_WAV), ref_text=REF_TEXT)
    ref_tokens = vp.ref_audio_tokens.cpu().numpy().T.astype(np.int32)  # [Tr, 8]
    np.savez(out / "ref_prompt.npz", tokens=ref_tokens, ref_text=vp.ref_text, ref_rms=vp.ref_rms)
    print(f"ref tokens {ref_tokens.shape}  ref_text={vp.ref_text!r}  rms={vp.ref_rms:.4f}", flush=True)

    gen = OmniVoiceGenerationConfig(num_step=args.steps, position_temperature=0.0, class_temperature=0.0)
    summary = {}
    for name in args.cases:
        text = dict(CASES)[name]
        task = model._preprocess_all(text=text, language="zh", voice_clone_prompt=vp)
        inputs = model._prepare_inference_inputs(task.texts[0], task.target_lens[0], task.ref_texts[0],
                                                 task.ref_audio_tokens[0], task.langs[0],
                                                 task.instructs[0], gen.denoise)
        ids = inputs["input_ids"][0].cpu().numpy().T.astype(np.int32)  # [L, 8]
        amask = inputs["audio_mask"][0].cpu().numpy()
        t0 = time.perf_counter()
        with torch.no_grad():
            tokens = model._generate_iterative(task, gen)[0]  # [8, T]
            unmask_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            audio = model._decode_and_post_process(tokens.detach(), vp.ref_rms, gen)
            decode_s = time.perf_counter() - t0
        tok = tokens.cpu().numpy().T.astype(np.int32)
        np.savez(out / f"{name}.npz", input_ids=ids, audio_mask=amask, T=task.target_lens[0], tokens=tok)
        sf.write(str(out / f"{name}.wav"), audio, model.sampling_rate)
        raw_s = task.target_lens[0] * 960 / model.sampling_rate
        print(f"{name:<9} L={ids.shape[0]} T={task.target_lens[0]} raw {raw_s:.2f}s  "
              f"unmask {unmask_s:.2f}s decode {decode_s:.2f}s  RTF {(unmask_s + decode_s) / raw_s:.2f}", flush=True)
        summary[name] = dict(L=int(ids.shape[0]), T=int(task.target_lens[0]), unmask_s=unmask_s, decode_s=decode_s)
    (out / "ref_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
