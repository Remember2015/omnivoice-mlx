"""The first generation may happen on a worker thread.

MLX 0.32: an array built lazily on one thread cannot be evaluated on another
("There is no Stream(gpu, 0) in current thread"). A service constructs the
engine on its main thread and synthesises on a pump thread, so every array the
engine keeps between calls has to be materialised at construction — the
codebook offsets in OmniVoice.__init__ were not (2026-09-10, found when the
a TTS service warmed up on its own pump thread).

    .venv/bin/python bench/test_thread.py
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from cases import REF_TEXT, REF_WAV  # noqa: E402
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig  # noqa: E402


def main() -> int:
    tts = OmniVoiceTTS(str(ROOT / "models/mlx-q8-fp16"))
    vp = tts.make_prompt(REF_WAV, REF_TEXT, release_encoder=True)
    cfg = SamplerConfig(position_temperature=0.0)  # deterministic: the two threads must agree
    out: dict[str, object] = {}

    def worker():
        try:
            out["tokens"] = np.asarray(tts.generate("你好，今天天气不错。", vp, sampler=cfg, seed=3, postprocess=False).tokens)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=worker)   # first generation ever, off the main thread
    t.start()
    t.join()
    if "error" in out:
        print(f"FAIL first generation on a worker thread: {out['error']}")
        return 1
    main_tokens = np.asarray(tts.generate("你好，今天天气不错。", vp, sampler=cfg, seed=3, postprocess=False).tokens)
    same = main_tokens.shape == out["tokens"].shape and bool((main_tokens == out["tokens"]).all())
    print(f"{'ok  ' if same else 'FAIL'} worker-thread generation matches the main thread's ({main_tokens.shape[0]} tokens)")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
