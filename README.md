# OmniVoice on MLX

简体中文 | [English](README.en.md)

[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) 的 MLX 移植。推理不依赖 torch / transformers，
fp32 下与官方实现逐 token 一致。

| 来源 | 框架 | 设备 | 精度 | 步数 | RTF 短句 | RTF 长段 | 整句延迟 | |
|---|---|---|---|---:|---:|---:|---:|---|
| 官方 | torch | CPU | fp32 | 32 | ≈2–3 | — | — | |
| 官方 | torch | MPS | fp32 | 32 | 0.98 | — | — | |
| 官方 | torch | MPS | fp16 | 32 | 1.14 | — | — | |
| 本项目 | MLX | GPU | fp16 | 32 | 0.23 | 0.10 | 0.5–1.0 s | |
| **本项目** | **MLX** | **GPU** | **8bit+fp16** | **16** | **0.106** | **≈0.08** | **0.23–0.44 s** | **默认** |
| 本项目 | MLX | GPU | 8bit+fp16 | 16 | 0.080 | — | — | 打包 B=4–8 |
| 本项目 | MLX | GPU | 8bit+fp16 | 8 | 0.075 | 0.057 | 0.16–0.32 s | |

M2 Max 12 核 / 38 核 GPU / 32 GB，macOS 26.6，MLX 0.32.2。测法、消融与负结果见
[docs/research-log.md](docs/research-log.md)。

## 安装

```bash
pip install git+https://github.com/Remember2015/omnivoice-mlx
hf download remember2015/omnivoice-mlx-q8-fp16 --local-dir models/mlx-q8-fp16
```

codec 权重另有许可，不随上面那份分发，从上游取：

```bash
hf download k2-fsa/OmniVoice --local-dir models/k2-fsa-OmniVoice
ln -s ../k2-fsa-OmniVoice/audio_tokenizer models/mlx-q8-fp16/audio_tokenizer
python -c "from omnivoice_mlx.codec import write_slim_decoder; write_slim_decoder('models/mlx-q8-fp16', 'float16')"
```

解码只用导出的这 44 MB 分支，完整 tokenizer 仅编码参考音时加载。
`scripts/convert.py` 还能导出 bf16 / fp16 / 4-bit，以及嵌入和头一起量化的 345 MB 版本。

## 使用

```python
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig
from omnivoice_mlx.stream import generate_stream

tts = OmniVoiceTTS("models/mlx-q8-fp16")
voice = tts.make_prompt("my-voice.wav", "这段录音念的那句话，标点照写。")

r = tts.generate("今天天气不错，我们出去走走吧。", voice)   # r.audio: float32 24 kHz；r.rtf
tts.generate_batch(["第一句。", "第二句。"], voice, max_batch=4)
tts.generate_long(paragraph, voice)                      # 官方分块路径
for piece in generate_stream(tts, paragraph, voice):     # 分句预生成，首音频 165 ms
    play(piece.audio)

SamplerConfig()                                     # num_steps=16, cache_refresh=8, uncond_every=3
SamplerConfig(32, cache_refresh=0, uncond_every=1)  # 官方
SamplerConfig(8, cache_refresh=4, uncond_every=1)   # 表里 8 步那行
```

参考音仓库不带，自己准备：3–4 s 干净单声道，转写须与音频一致（它进 prompt）。`bench/` 走环境变量：

```bash
export OMNIVOICE_REF_WAV=assets/my-voice.wav
export OMNIVOICE_REF_TEXT="它念的那句话，标点照写。"
```

## 性能优化

| 用了 | 效果 |
|---|---|
| 激活 fp16 | 快 12–15 %；M2 无原生 bf16，bf16 与 fp32 同速 |
| 步数 32 → 16 | CER / 声纹 / UTMOS 不变 |
| 前缀 KV 缓存 | prompt 的 K/V 每 8 步重算一次 |
| uncond 隔步复用 | −20 %，三项指标不变 |
| 8-bit 量化 | 不提速，每步 GEMM 的 M ≈ 200–400，瓶颈在 kernel 发射；省常驻内存 1.99 → 1.56 GB |

| 没用 | 结果 |
|---|---|
| 自定义 Metal GEMM | 仅 M ≤ 80 且 N ≥ 4096 快 1.05–1.3×，其余慢 10–15 % |
| CFG 截断 | 声纹 −0.05～−0.08 |
| 置信度阈值自适应步数 | 几乎不触发，每步同步反而慢 |
| 4 步 | 声纹 −0.03，出整句错 |
| 分句续接 | UTMOS 2.95 → 2.65 |
| `mx.compile` | 可融合的只有几条逐元素链，≤ 5 % |

## 目录

```
omnivoice_mlx/     backbone / model / sampler / pipeline / codec / stream / textnorm
  higgs/           Higgs-audio v2 tokenizer，vendor 自 mlx-audio（MIT），逐位一致
scripts/convert.py 导出 MLX 权重目录
bench/             parity_*（对官方逐 token 对拍）、bench*（计时，同进程交错）、benchlock.sh（排他锁 + 等空载）
docs/              研究记录
```

计时一律走 `bench/benchlock.sh`：绝对值随负载漂 10–40 %，只有同进程交错的数可比。

## 许可

代码 Apache-2.0（`LICENSE`、`NOTICE`），与上游一致。`omnivoice_mlx/higgs/` vendor 自
[mlx-audio](https://github.com/Blaizzy/mlx-audio)，MIT，全文见 `THIRD-PARTY-LICENSES.md`。

**权重不归这个许可管。** k2-fsa 代码 Apache-2.0，预训练权重 CC-BY-NC（训练数据约束）。
`scripts/convert.py` 的产物和上面那份 HF 权重都是其衍生物，同样非商用、需署名。
Higgs codec 权重另有 Boson Higgs Audio 2 Community License，故本仓库与那份 HF 权重都不重分发。
