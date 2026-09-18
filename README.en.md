# OmniVoice on MLX

[简体中文](README.md) | English

An MLX port of [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice). Inference pulls in neither torch nor
transformers, and in fp32 it matches the official implementation token for token.

| source | framework | device | weights | activations | steps | RTF short | RTF long | sentence latency | notes |
|---|---|---|---|---|---:|---:|---:|---:|---|
| official | torch | CPU | fp32 | fp32 | 32 | ≈2–3 | — | — | |
| official | torch | MPS | fp32 | fp32 | 32 | 0.98 | — | — | |
| mlx-audio | MLX | GPU | bf16 | bf16 | 32 | 0.68 | — | — | |
| this port | MLX | GPU | 8-bit | fp16 | 32 | 0.381 | 0.25 | 0.5–1.0 s | |
| **this port** | **MLX** | **GPU** | **8-bit** | **fp16** | **16** | **0.106** | **≈0.08** | **0.23–0.44 s** | **default** |
| this port | MLX | GPU | 8-bit | fp16 | 16 | 0.080 | — | — | batched B=4–8 |

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

That last command writes the 44 MB decode branch, which is all inference reads; the full tokenizer is loaded when
encoding a reference clip.
`scripts/convert.py` also writes bf16, fp16 and fp32.

## Use

```python
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig
from omnivoice_mlx.stream import generate_stream

tts = OmniVoiceTTS("models/mlx-q8-fp16")
voice = tts.make_prompt("my-voice.wav", "exactly what that clip says.")

r = tts.generate("今天天气不错，我们出去走走吧。", voice)   # r.audio: float32 24 kHz, r.rtf
tts.generate_batch(["第一句。", "第二句。"], voice, max_batch=4)
tts.generate_long(paragraph, voice)                      # long text, split at punctuation
for piece in generate_stream(tts, paragraph, voice):     # clause-ahead, 165 ms to first audio
    play(piece.audio)

SamplerConfig()                                     # num_steps=16, cache_refresh=8, uncond_every=3
SamplerConfig(32, cache_refresh=0, uncond_every=1)  # official
SamplerConfig(8, cache_refresh=4, uncond_every=1)   # faster, slightly less natural
```

No reference clip ships here, bring your own: 3–4 s of clean mono, with a transcript that matches the audio. `bench/` reads it from the environment:

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

The M2 has no native bf16, so bf16 runs at fp32 speed and fp16 is 12–15 % faster than both. 8-bit weights are not a
speedup; they buy 1.99 → 1.56 GB resident.

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
├── kernels*.py       # custom Metal GEMM, off by default
└── higgs/            # tokenizer, vendored from mlx-audio
server.py             # HTTP server, one worker + dynamic batching
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
