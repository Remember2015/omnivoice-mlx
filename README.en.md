# OmniVoice on MLX

[简体中文](README.md) | English

An MLX port of [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice). Inference pulls in neither torch nor
transformers, and in fp32 it matches the official implementation token for token.

| implementation | weights / activations | steps | RTF short | RTF long | sentence latency |
|---|---|---:|---:|---:|---:|
| official torch, CPU | fp32 | 32 | ≈2–3 | — | — |
| official torch, MPS | fp32 | 32 | 0.98 | — | — |
| official torch, MPS | fp16 | 32 | 1.14 | — | — |
| this port | fp16 | 32 | 0.23 | 0.10 | 0.5–1.0 s |
| **this port (default)** | **8-bit g64 + fp16** | **16** | **0.106** | **≈0.08** | **0.23–0.44 s** |
| this port (fast, UTMOS −5 %) | 8-bit g64 + fp16 | 8 | 0.075 | 0.057 | 0.16–0.32 s |

Short = three sentences of 7 / 17 / 34 characters pooled, long = one 46 s paragraph. Batched (B=4–8)
short-sentence RTF is 0.080. M2 Max 12-core CPU / 38-core GPU / 32 GB, macOS 26.6, MLX 0.32.2. Method, ablations and
dead ends: [docs/research-log.md](docs/research-log.md) (Chinese).

## Install

```bash
pip install git+https://github.com/Remember2015/omnivoice-mlx
hf download remember2015/omnivoice-mlx-q8-fp16 --local-dir models/mlx-q8-fp16
```

The codec weights have their own licence and are not redistributed with the checkpoint above; take them from
upstream:

```bash
hf download k2-fsa/OmniVoice --local-dir models/k2-fsa-OmniVoice
ln -s ../k2-fsa-OmniVoice/audio_tokenizer models/mlx-q8-fp16/audio_tokenizer
python -c "from omnivoice_mlx.codec import write_slim_decoder; write_slim_decoder('models/mlx-q8-fp16', 'float16')"
```

Decoding reads only that 44 MB branch; the full tokenizer is loaded when encoding a reference clip. Other flavours
come from `scripts/convert.py`: bf16, fp16, 4-bit, and a 345 MB build with embeddings and heads quantised too.

## Use

```python
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig

tts = OmniVoiceTTS("models/mlx-q8-fp16")
voice = tts.make_prompt("my-voice.wav", "exactly what that clip says.")
r = tts.generate("今天天气不错，我们出去走走吧。", voice, language="zh")   # r.audio float32 24 kHz, r.rtf
```

`SamplerConfig()` defaults to `num_steps=16, cache_refresh=8, uncond_every=3`. The official sampler is
`SamplerConfig(32, cache_refresh=0, uncond_every=1)`, the fast one `SamplerConfig(8, cache_refresh=4)`. There is also
`generate_batch` (length-bucketed packing), `generate_long` (the official chunking path) and `omnivoice_mlx.stream`
(clause-ahead synthesis: 165 ms to first audio, no underruns after that).

The reference clip is yours to supply — a recorded voice belongs to whoever spoke it, so none ships here. 3–4 s of
clean mono, with a transcript that matches the audio (it goes into the prompt). `bench/` reads it from the
environment:

```bash
export OMNIVOICE_REF_WAV=assets/my-voice.wav
export OMNIVOICE_REF_TEXT="what the clip says, punctuation included."
```

## Where the speed comes from

- fp16 activations. The M2 GPU has no native bf16, MLX emulates it, and bf16 runs at fp32 speed; fp16 is 12–15 % faster.
- 32 → 16 steps. CER, speaker similarity and UTMOS are all unchanged; UTMOS only drops 5 % at 8 steps.
- Prefix KV cache. The prompt's K/V is recomputed every 8 steps, and the steps in between only push target tokens.
- Stale CFG. The unconditional branch runs every 3 steps: −20 %, same three metrics.

Quantisation is not a speedup. With bf16 activations 8-bit is 15–30 % *slower* than fp16: each step is a GEMM with
M ≈ 200–400, bound by kernel dispatch rather than bandwidth, and `quantized_matmul` has no edge at that size. It
only overtakes fp16 (by 3–12 %) once the activations are fp16 too. What it really buys is resident memory,
1.99 → 1.56 GB.

The dead ends are all written up in the log: custom Metal GEMM kernels (the simdgroup version wins 1.05–1.3× only at
M ≤ 80 and N ≥ 4096, and loses 10–15 % elsewhere), CFG truncation (speaker similarity −0.05 to −0.08), a
confidence-threshold adaptive step count, 4 steps, clause-to-clause prompt continuation (UTMOS 2.95 → 2.65), and
`mx.compile`.

## Layout

```
omnivoice_mlx/     backbone / model / sampler / pipeline / codec / stream / textnorm
  higgs/           Higgs-audio v2 tokenizer, vendored from mlx-audio (MIT), bit-identical
scripts/convert.py writes MLX weight directories
bench/             parity_* (token-for-token against the official torch), bench* (timing, interleaved in one
                   process), benchlock.sh (exclusive lock + wait for an idle machine)
docs/              the research log
```

Every timing run goes through `bench/benchlock.sh`. Absolute numbers drift 10–40 % with load on this machine; only
the interleaved in-process ones compare.

## Licence

The code is Apache-2.0 (`LICENSE`, `NOTICE`), following upstream. `omnivoice_mlx/higgs/` is vendored from
[mlx-audio](https://github.com/Blaizzy/mlx-audio) under MIT; the full text is in `THIRD-PARTY-LICENSES.md`.

**The weights are not covered by that licence.** k2-fsa releases the code under Apache-2.0 and the pre-trained
weights under CC-BY-NC, because of constraints in the training data. Whatever `scripts/convert.py` produces, and the
HF checkpoint above, are derivatives of those weights and carry the same terms: non-commercial, with attribution.
The Higgs codec weights are under the Boson Higgs Audio 2 Community License, which is why neither this repository
nor that checkpoint redistributes them.
