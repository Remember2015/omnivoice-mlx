# OmniVoice on MLX

[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) — 0.6B masked-diffusion (non-autoregressive) TTS, Qwen3
bidirectional backbone + Higgs-audio v2 codec — reimplemented for Apple silicon in MLX. No torch, no transformers at
inference: five packages. In fp32 it reproduces the official implementation token for token.

**RTF 0.106 on an M2 Max**, with 8-bit weights + fp16 activations, 16 denoising steps, a prefix KV cache refreshed
every 8 steps and the unconditional CFG branch reused every 3 — against 0.98 for the official torch code on MPS and
≈2–3 on CPU, with CER, speaker similarity and UTMOS unchanged from the official 32-step sampler. There is no
streaming in the architecture, so the whole sentence is the first packet: **230–440 ms**.

| | RTF 短句 | RTF 长段 | 出声延迟 |
|---|---:|---:|---|
| 官方 torch，CPU fp32，32 步 | ≈2–3 | — | — |
| 官方 torch，MPS fp32 / fp16，32 步 | 0.98 / 1.14 | — | — |
| 本移植 fp16，32 步（官方那套采样） | 0.23 | 0.10 | 0.5–1.0 s |
| **本移植 8-bit + fp16，16 步（默认）** | **0.106** | ≈0.08 | **0.23–0.44 s** |
| 同上 8 步（快档，自然度 −5 %） | 0.075 | 0.057 | 0.16–0.32 s |
| 同上，多句一起算（吞吐档） | 0.080 | — | — |

- **RTF** = 合成耗时 ÷ 生成音频的时长，越小越快：0.1 表示一秒钟的话花 0.1 秒算出来。
- **短句 / 长段**：短句是 7、17、34 字三句合起来算的，长段是一段读出来 46 秒的文字。
- **步**是去掩码的迭代次数。这个模型不是一个字一个字往外蹦的，整句一起反复去掩码，步数越少越快。
  官方默认 32 步；默认档的 16 步另外还省了两处重复计算（前缀 KV 缓存、无条件分支隔步复用），都在研究记录里。
- **出声延迟**是从调用到**整句**音频出来。架构上没有流式，所以没有「先出一点再边放边算」这回事，第一个包就是整句。

M2 Max 12 核 / 38 核 GPU / 32 GB，macOS 26.6，MLX 0.32.2。测法、消融和所有负结果在
**[docs/research-log.md](docs/research-log.md)**（17 节，每节写明怎么量的）。

## 装

```bash
pip install omnivoice-mlx
hf download remember2015/omnivoice-mlx-q8-fp16 --local-dir models/mlx-q8-fp16   # 推荐档，829 MB
```

codec 不在那份权重里（Higgs tokenizer 有自己的许可），从上游拿一次：

```bash
hf download k2-fsa/OmniVoice --local-dir models/k2-fsa-OmniVoice
ln -s ../k2-fsa-OmniVoice/audio_tokenizer models/mlx-q8-fp16/audio_tokenizer
python -c "from omnivoice_mlx.codec import write_slim_decoder; write_slim_decoder('models/mlx-q8-fp16', 'float16')"
```

最后那行写出 44 MB 的解码分支，推理只用它；完整的 `audio_tokenizer` 只在编码参考音时读。
想自己转权重（bf16 / fp16 / 4-bit / 4-bit 全量化 345 MB）用 `scripts/convert.py`。

## 用

```python
from omnivoice_mlx import OmniVoiceTTS, SamplerConfig

tts = OmniVoiceTTS("models/mlx-q8-fp16")
voice = tts.make_prompt("my-voice.wav", "这段录音念的那句话，标点照写。")
r = tts.generate("今天天气不错，我们出去走走吧。", voice, language="zh")
# r.audio: float32 24 kHz；r.rtf
```

`SamplerConfig()` 默认 16 步 + `cache_refresh=8` + `uncond_every=3`；官方采样是
`SamplerConfig(32, cache_refresh=0, uncond_every=1)`，快档是 `SamplerConfig(8, cache_refresh=4)`。
还有 `tts.generate_batch([...])`（多句打包）、`tts.generate_long(paragraph)`（官方分块路径）、
`omnivoice_mlx.stream`（按分句提前一句合成，首音频 165 ms，之后不断流）。

**参考音自己准备**——一段录下来的嗓子属于说话的那个人，仓库里不带。3–4 秒干净单声道就够，转写必须是它真说的那句
（它进 prompt，不是标签）。跑 `bench/` 时用环境变量指过去：

```bash
export OMNIVOICE_REF_WAV=assets/my-voice.wav
export OMNIVOICE_REF_TEXT="它念的那句话，标点照写。"
```

## 怎么快起来的

四件事按贡献排：**fp16 激活**（M2 没有原生 bf16，bf16 和 fp32 一样慢）、**16 步**（32 → 16 三个指标都不动，
UTMOS 要到 8 步才掉 5 %）、**前缀 KV 缓存**（prompt 的 K/V 每 8 步算一次）、**uncond 隔步复用**（无条件分支每 3 步
算一次，−20 %，指标不动）。8-bit 量化本身不提速——每步是 M≈200–400 的 GEMM，是发射瓶颈不是带宽瓶颈——它省的是内存。

试了没用的都留在日志里：自定义 Metal GEMM 内核（simdgroup 版只在 M≤80 且 N≥4096 赢 5–30 %，其余慢 10–15 %）、
CFG 截断（声纹 −0.05～−0.08）、置信度阈值自适应步数、4 步（出整句错）、分句续接（UTMOS 2.95 → 2.65）、
`mx.compile`。

## 布局

```
omnivoice_mlx/     移植本体：backbone / model / sampler / pipeline / codec / stream / textnorm
  higgs/           Higgs-audio v2 tokenizer，从 mlx-audio vendor 而来（MIT），逐位相同
scripts/convert.py 写 MLX 权重目录（bf16 / fp16 / 8-bit / 4-bit）
bench/             parity_*（对官方 torch 逐 token 对拍）、bench*（计时，同进程交错）、benchlock.sh（排他锁 + 等空载）
docs/              研究记录
```

任何计时都走 `bench/benchlock.sh`：这台机器上绝对数随负载漂 10–40 %，只有同进程交错的数字可比。

## 许可

代码 **Apache-2.0**（`LICENSE`、`NOTICE`），跟上游一致。`omnivoice_mlx/higgs/` 来自
[mlx-audio](https://github.com/Blaizzy/mlx-audio)，MIT，全文在 `THIRD-PARTY-LICENSES.md`。

**权重不归这个许可管。** k2-fsa 的代码是 Apache-2.0，预训练权重是 **CC-BY-NC**（训练数据约束）。
`scripts/convert.py` 转出来的、以及上面那份 HF 权重，都是它的衍生物，同样 **非商用 + 署名**。
Higgs codec 权重另有 Boson Higgs Audio 2 Community License，所以本仓库和那份 HF 权重都不重分发它。
