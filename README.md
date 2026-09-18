# OmniVoice on MLX

简体中文 | [English](README.en.md)

[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) 的 MLX 移植。推理不依赖 torch / transformers，
fp32 下与官方实现逐 token 一致。

| 来源 | 框架 | 设备 | 精度 | 步数 | RTF 短句 | RTF 长段 | 整句延迟 | |
|---|---|---|---|---:|---:|---:|---:|---|
| 官方 | torch | CPU | fp32 | 32 | ≈2–3 | — | — | |
| 官方 | torch | MPS | fp32 | 32 | 0.98 | — | — | |
| 本项目 | MLX | GPU | fp16 | 32 | 0.23 | 0.10 | 0.5–1.0 s | |
| **本项目** | **MLX** | **GPU** | **8bit+fp16** | **16** | **0.106** | **≈0.08** | **0.23–0.44 s** | **默认** |
| 本项目 | MLX | GPU | 8bit+fp16 | 16 | 0.080 | — | — | batch B=4–8 |
| 本项目 | MLX | GPU | 8bit+fp16 | 8 | 0.075 | 0.057 | 0.16–0.32 s | |

M2 Max 12 核 / 38 核 GPU / 32 GB，macOS 26.6，MLX 0.32.2。测量方法、消融与负结果见 [docs/research-log.md](docs/research-log.md)。

## 安装

```bash
pip install git+https://github.com/Remember2015/omnivoice-mlx
hf download remember2015/omnivoice-mlx-q8-fp16 --local-dir models/mlx-q8-fp16
```

codec 权重另有许可，不在上面那份里，从上游下载：

```bash
hf download k2-fsa/OmniVoice --local-dir models/k2-fsa-OmniVoice
ln -s ../k2-fsa-OmniVoice/audio_tokenizer models/mlx-q8-fp16/audio_tokenizer
python -c "from omnivoice_mlx.codec import write_slim_decoder; write_slim_decoder('models/mlx-q8-fp16', 'float16')"
```

解码只用导出的这 44 MB 分支，完整 tokenizer 只在编码参考音时加载。
`scripts/convert.py` 还能导出 bf16 / fp16 / fp32。

## 使用

```python
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig
from omnivoice_mlx.stream import generate_stream

tts = OmniVoiceTTS("models/mlx-q8-fp16")
voice = tts.make_prompt("my-voice.wav", "这段录音念的那句话，标点照写。")

r = tts.generate("今天天气不错，我们出去走走吧。", voice)   # r.audio: float32 24 kHz；r.rtf
tts.generate_batch(["第一句。", "第二句。"], voice, max_batch=4)
tts.generate_long(paragraph, voice)                      # 官方 chunk 路径
for piece in generate_stream(tts, paragraph, voice):     # 分句预生成，首音频 165 ms
    play(piece.audio)

SamplerConfig()                                     # num_steps=16, cache_refresh=8, uncond_every=3
SamplerConfig(32, cache_refresh=0, uncond_every=1)  # 官方
SamplerConfig(8, cache_refresh=4, uncond_every=1)   # 表里 8 步那行
```

参考音需要自己准备，仓库里不带：3–4 s 干净单声道，转写要与音频一致（它会进 prompt）。`bench/` 走环境变量：

```bash
export OMNIVOICE_REF_WAV=assets/my-voice.wav
export OMNIVOICE_REF_TEXT="它念的那句话，标点照写。"
```

## 性能优化

从官方的 32 步逐级改到默认配置，RTF（3 句合并，8-bit + fp16）：

| 手段 | RTF | CER / speaker sim / UTMOS |
|---|---:|---|
| 32 步，无 KV cache | 0.381 | 基线 |
| 步数 32 → 16 | 0.194 | 不变 |
| + 前缀 KV cache，每 8 步重算 | 0.133 | 不变 |
| + uncond 每 3 步重算 | **0.106** | 不变 |

精度上 fp16 比 bf16 快 12–15 %（M2 无原生 bf16，bf16 与 fp32 同速）；8-bit 量化不提速，省的是常驻内存
1.99 → 1.56 GB。

## 目录

```
omnivoice_mlx/
├── backbone.py       # Qwen3 双向 backbone
├── model.py          # 8 个 codebook 的 embedding + 8 个 head
├── sampler.py        # unmask 循环
├── pipeline.py       # 推理流程
├── codec.py          # Higgs codec
├── stream.py         # 分句预生成
├── textnorm.py       # text normalization
├── kernels*.py       # 自定义 Metal GEMM，只在特定形状上更快，默认不用
└── higgs/            # tokenizer，vendor 自 mlx-audio（MIT）
scripts/convert.py    # 导出 MLX 权重
bench/                # parity、计时、benchlock.sh
docs/                 # 研究记录
```

## 许可

| 内容 | 许可 |
|---|---|
| 本仓库代码 | Apache-2.0，同上游 |
| `omnivoice_mlx/higgs/` | MIT，vendor 自 [mlx-audio](https://github.com/Blaizzy/mlx-audio) |
| 模型权重 | **CC-BY-NC**，非商用、需署名 |
| Higgs codec 权重 | Boson Higgs Audio 2 Community License |
