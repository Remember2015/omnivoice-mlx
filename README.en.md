# OmniVoice on MLX

[简体中文](README.md) | English

An MLX port of [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice). Inference pulls in neither torch nor
transformers, and in fp32 it matches the official implementation token for token.

| source | framework | device | precision | steps | RTF short | RTF long | sentence latency | |
|---|---|---|---|---:|---:|---:|---:|---|
| official | torch | CPU | fp32 | 32 | ≈2–3 | — | — | |
| official | torch | MPS | fp32 | 32 | 0.98 | — | — | |
| official | torch | MPS | fp16 | 32 | 1.14 | — | — | |
| this port | MLX | GPU | fp16 | 32 | 0.23 | 0.10 | 0.5–1.0 s | |
| **this port** | **MLX** | **GPU** | **8bit+fp16** | **16** | **0.106** | **≈0.08** | **0.23–0.44 s** | **default** |
| this port | MLX | GPU | 8bit+fp16 | 16 | 0.080 | — | — | batched B=4–8 |
| this port | MLX | GPU | 8bit+fp16 | 8 | 0.075 | 0.057 | 0.16–0.32 s | |

M2 Max 12-core CPU / 38-core GPU / 32 GB, macOS 26.6, MLX 0.32.2. Method, ablations and dead ends are in
[docs/research-log.md](docs/research-log.md) (Chinese).

## Install

```bash
pip install git+https://github.com/Remember2015/omnivoice-mlx
hf download remember2015/omnivoice-mlx-q8-fp16 --local-dir models/mlx-q8-fp16
```

The codec weights have their own licence and are not in that checkpoint; take them from upstream:

```bash
hf download k2-fsa/OmniVoice --local-dir models/k2-fsa-OmniVoice
ln -s ../k2-fsa-OmniVoice/audio_tokenizer models/mlx-q8-fp16/audio_tokenizer
python -c "from omnivoice_mlx.codec import write_slim_decoder; write_slim_decoder('models/mlx-q8-fp16', 'float16')"
```

Decoding reads only that 44 MB branch; the full tokenizer is loaded when encoding a reference clip.
`scripts/convert.py` also writes bf16, fp16 and fp32.

## Use

```python
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig
from omnivoice_mlx.stream import generate_stream

tts = OmniVoiceTTS("models/mlx-q8-fp16")
voice = tts.make_prompt("my-voice.wav", "exactly what that clip says.")

r = tts.generate("今天天气不错，我们出去走走吧。", voice)   # r.audio: float32 24 kHz, r.rtf
tts.generate_batch(["第一句。", "第二句。"], voice, max_batch=4)
tts.generate_long(paragraph, voice)                      # official chunking path
for piece in generate_stream(tts, paragraph, voice):     # clause-ahead, 165 ms to first audio
    play(piece.audio)

SamplerConfig()                                     # num_steps=16, cache_refresh=8, uncond_every=3
SamplerConfig(32, cache_refresh=0, uncond_every=1)  # official
SamplerConfig(8, cache_refresh=4, uncond_every=1)   # the 8-step row above
```

No reference clip ships here, bring your own: 3–4 s of clean mono, with a transcript that matches the audio (it
goes into the prompt). `bench/` reads it from the environment:

```bash
export OMNIVOICE_REF_WAV=assets/my-voice.wav
export OMNIVOICE_REF_TEXT="what the clip says, punctuation included."
```

## Optimisations

RTF as each step is added (three sentences pooled, 8-bit + fp16):

| | RTF | CER / speaker sim / UTMOS |
|---|---:|---|
| 32 steps, no KV cache | 0.381 | baseline |
| 32 → 16 steps | 0.194 | unchanged |
| + prefix KV cache, refreshed every 8 | 0.133 | unchanged |
| + unconditional branch every 3 | **0.106** | unchanged |

fp16 activations are 12–15 % faster than bf16 (the M2 has no native bf16, so bf16 runs at fp32 speed). 8-bit weights
are not a speedup; they buy 1.99 → 1.56 GB resident.

## Layout

```
omnivoice_mlx/
├── backbone.py       # Qwen3 bidirectional backbone
├── model.py          # 8 codebook embeddings + 8 heads
├── sampler.py        # the unmasking loop
├── pipeline.py       # the inference path
├── codec.py          # Higgs codec
├── stream.py         # clause-ahead synthesis
├── textnorm.py       # text normalisation
├── kernels*.py       # custom Metal GEMM, only wins on some shapes, off by default
└── higgs/            # tokenizer, vendored from mlx-audio (MIT)
scripts/convert.py    # writes MLX weight directories
bench/                # parity, timing, benchlock.sh
docs/                 # the research log
```

## Licence

| what | licence |
|---|---|
| this repository's code | Apache-2.0, following upstream |
| `omnivoice_mlx/higgs/` | MIT, vendored from [mlx-audio](https://github.com/Blaizzy/mlx-audio) |
| model weights | **CC-BY-NC**, non-commercial, attribution required |
| Higgs codec weights | Boson Higgs Audio 2 Community License |
