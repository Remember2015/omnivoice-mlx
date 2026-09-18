安装和用法见 [../README.md](../README.md)。本文是测量记录，没走通的尝试也记着。

测试环境：M2 Max（12 核 CPU / 38 核 GPU / 32 GB），macOS 26.6，MLX 0.32.2。
RTF =（unmask + codec 解码）/ 原始时长，同进程交错取 3 轮中位数，绝对值随负载漂 10–40 %；
CER、speaker similarity、UTMOS 分别用 Fun-ASR-Nano、CAM++、UTMOS22-strong 测，换一套模型数值就不一样。

## 结果

| 对照 | 框架 | 设备 | 精度 | 步数 | RTF 短句 |
|---|---|---|---|---:|---:|
| 官方 | torch | CPU | fp32 | 32 | ≈2–3 |
| 官方 | torch | MPS | fp32 | 32 | 0.98 |
| mlx-audio | MLX | GPU | bf16 | 32 | 0.68 |

本项目，8-bit + fp16，列名是 `SamplerConfig` 的参数（`cache_refresh=0` 表示不用 KV cache）：

| num_steps | cache_refresh | uncond_every | RTF 短句 | 整句延迟 | CER | sim | UTMOS | 备注 |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 32 | 0 | 1 | 0.381 | 0.5–1.0 s | 0.73 % | 0.746 | 2.826 | |
| **16** | **8** | **3** | **0.106** | **0.23–0.44 s** | 0.40 % | 0.747 | **2.835** | **默认** |
| 16 | 8 | 3 | 0.080 | — | — | — | — | batch B=4–8 |
| 8 | 4 | 1 | 0.075 | 0.16–0.32 s | 0.40 % | 0.738 | 2.687 | |
| 6 | 6 | 1 | 0.060 | 0.13–0.22 s | 0.44 % | 0.729 | 2.532 | |

## 1. 移植正确性

fp32 下 8 步确定性生成逐 token 与官方 100 % 一致，解码后波形时长和 RMS 也相同。

bf16 / fp16 只有 15–25 % token 一致，所以精度的影响只看 CER / speaker similarity / UTMOS。

## 2. 精度与量化

| 权重 | 激活 | 常驻 | 权重文件 | ms/step（短/中/长） | RTF 32 步 |
|---|---|---:|---:|---|---:|
| fp32 | fp32 | 3.19 GB | — | 36.1 / 38.6 / 58.9 | 0.481 |
| bf16 | bf16 | 1.99 GB | 1242 MB | 35.8 / 38.3 / 56.9 | 0.469 |
| fp16 | fp16 | 1.99 GB | 1242 MB | 31.6 / 33.9 / 48.7 | 0.409 |
| 8-bit g64 | bf16 | 1.56 GB | 829 MB | 37.1 / 42.3 / 64.4 | 0.514 |
| **8-bit g64** | **fp16** | 1.56 GB | 829 MB | **27.8 / 31.5 / 47.1** | **0.381** |

- M2 没有原生 bf16，bf16 与 fp32 同速，fp16 比它们快 12–15 %。
- 量化更慢只对 bf16 激活成立，换 fp16 激活后与 fp16 GEMM 持平或略快；默认取 8-bit + fp16。
- 每步约 15 ms 固定开销：一行从 198 token 加到 254 只多 7 %，254 → 391 才多 44 %。

## 3. 采样：KV cache、CFG 截断、置信度阈值

| 手段 | 做法 | ms/step | CER | speaker sim | 采用 |
|---|---|---|---|---|---|
| **前缀 KV cache** | prompt 的 K/V 每 n 步算一次 | −30～37 % | 不变 | −0.01～−0.02 | ✅ |
| CFG 截断 | 后半步不跑 uncond 分支 | −12～18 % | 出错字 | −0.05～−0.08 | ❌ |
| 置信度阈值提前揭示 | 超阈值的位提前定下来 | +5～13 % | — | — | ❌ |

## 4. 步数

| num_steps | cache_refresh | ms/step（短/中/长） | RTF | CER | sim 均值 / 最低 | UTMOS |
|---:|---:|---|---:|---:|---|---:|
| 32 | 0 | 27.7 / 31.3 / 46.9 | 0.381 | 0.73 % | 0.746 / 0.653 | 2.826 |
| 16 | 0 | 27.8 / 31.4 / 47.2 | 0.194 | 0.98 % | 0.746 / 0.613 | 2.838 |
| **16** | **8** | **17.0 / 20.8 / 34.0** | **0.133** | 0.73 % | 0.737 / 0.596 | **2.835** |
| 8 | 0 | 27.9 / 31.7 / 47.2 | 0.101 | 0.90 % | 0.735 / 0.607 | 2.701 |
| 8 | 4 | 18.7 / 22.6 / 36.2 | 0.075 | 0.40 % | 0.738 / 0.639 | 2.687 |
| 6 | 6 | — | 0.060 | 0.44 % | 0.729 / 0.583 | 2.532 |
| 4 | 0 | — | 0.058 | 1.62 % | **0.700** / 0.550 | 2.327 |

- speaker similarity 全程在 0.735–0.746 之间，是噪声底；CER 也分不出 8 步和 32 步。只有 UTMOS 能：
  16 → 8 步降 0.13，8 → 6 → 4 步再各降 0.1 和 0.26。
- 16 步上加 KV cache 不影响 UTMOS，6 步起会降 0.06。
- 4 步 speaker similarity 降 0.03–0.04，并出现整句错误。论文的步数消融里英文 WER 在 8 步翻倍，这批中文短句上
  没有出现。单句 RTF 下限约 0.06。

## 5. 无条件分支隔步重算

`uncond_every=n`：每 n 步重算一次无条件分支，中间的步复用上次的 log-prob。

| num_steps | cache_refresh | uncond_every | RTF | CER | sim 均值 / 最低 | UTMOS |
|---:|---:|---:|---:|---:|---|---:|
| 16 | 8 | 1 | 0.133 | 0.33 % | 0.746 / 0.580 | 2.826 |
| 16 | 8 | 2 | 0.109 | 0.33 % | 0.744 / 0.632 | 2.762 |
| **16** | **8** | **3** | **0.106** | 0.40 % | 0.747 / 0.647 | **2.835** |
| 12 | 6 | 1 | 0.104 | 0.75 % | 0.739 / 0.602 | 2.748 |
| 12 | 6 | 2 | 0.086 | 0.83 % | 0.730 / 0.534 | 2.741 |
| 10 | 5 | 1 | 0.089 | 1.07 %（1 句 25 %） | 0.747 / 0.626 | 2.732 |

一行从 142 降到 110 token，三项指标与基线一致。
**默认 `SamplerConfig(16, cache_refresh=8, uncond_every=3)`，RTF 0.106。**

## 6. 多句 batch

20 句集按 B 句一组走 `generate_batch`。

| B | RTF | rows/step | GEMM 速率（TFLOPS） | 峰值内存（GB） |
|---:|---:|---:|---:|---:|
| 1 | 0.106 | 128 | 5.9 | 2.57 |
| 2 | 0.093 | 258 | 6.9 | — |
| 4 | 0.086 | 527 | 7.7 | 3.11 |
| 8 | **0.081** | 911 | 8.0 | 3.68 |
| 16 | 0.080 | 1429 | 8.0 | 4.80 |

B≥4 之后不再提升，再大只是多占内存。交互式场景用不上，首批延迟按 B 倍增。

**MLX 的缓冲区缓存默认无上限**，大 batch 之后会把后续分配拖慢，连 B=1 都从 0.107 退到 0.162。
`mx.set_cache_limit(512 MB)` 已写进 `OmniVoiceTTS.__init__`，不设这个上限就复现不出上面这张表。

## 7. 长文本与流式

- **长文本** `generate_long`：304 字、约 46 s，切 4 块，`num_steps=8, cache_refresh=4` 下顺序 0.057、chunk batch 0.053。
  chunk 并行只再省 6–8 %，每步已经 700+ token。整段 2.5 s 合成完。
- **流式** `stream.py`：按标点切分句，边合成边播放。304 字 → 22 句，首音频 **165 ms**（8 步）、286 ms（16 步），
  断流 0 次。代价是分句边界的韵律接不上，总时长比整段合成多 10 %。
- **分句续接**（没走通）：把上一句生成的 token 接进参考。UTMOS 2.949 → **2.653**，还慢 10 %。
  默认关，`generate_stream(continuity=True)` 可开。

## 8. 算子层

一个 unmask step 的耗时分布（ms）：

| T | rows | step | attention ×28 | MLP ×28 | head + CFG + sample | host graph |
|---:|---:|---:|---:|---:|---:|---:|
| 39 | 78 | 16.0 | 9.0 | 7.4 | 0.8 | 1.0 |
| 126 | 252 | 33.4 | 16.9 | 17.7 | 1.5 | 2.5 |

四个 8-bit GEMM 占一个 step 的 67 %，T=126 时 77 %。GPU 占用率 98–99 %；GEMM 速率 5.9 TFLOPS，是峰值 13.6 的 43 %，
长文本时 7.1，占 52 %。

MLX 的量化 GEMM mma 已跑满峰值的 55–75 %，唯一的浪费是把 M 补到 32：M=32 → 33 时耗时从 56 跳到 90 µs。
单句延迟在算子层已无空间。

自己写过两版 Metal GEMM（`kernels.py` 的 GEMV 式、`kernels_sg.py` 的 simdgroup 版），都没能超过 MLX 内置的，
代码留着但默认不用，`load_model(custom_gemm=True)` 可开。

## 9. codec 瘦身

- 只装解码分支（fp16，44 MB，SNR 59 dB），编码分支在 `make_prompt` 时懒加载、可释放：常驻 1.57 → 0.87 GB。
  原来装的是整个 806 MB 的 tokenizer。
- codec vendor 进 `omnivoice_mlx/higgs/`（6 个文件，MIT），输出与 mlx-audio 逐位相同；运行时不再需要 transformers。

## 10. text normalization

34 句，涵盖数字、日期、金额、单位等写法；「底噪」是直接合成手写口语形式的结果。

| | 原文 | zh_normalization | wetext | **wetext + 修补** | 底噪 |
|---|---:|---:|---:|---:|---:|
| 平均 CER | 7.09 % | 2.09 % | 1.79 % | **0.33 %** | 0.33 % |
| CER > 10 % 的句子 | 8 / 34 | 2 | 2 | **0** | 0 |

- 不规整会丢信息：`-5 度` → 五度、`3.5%` → 丢「百分之」、`14:30` → 十四三零。
- 两个现成规整器各有错漏：`3500 台` → 三五零零台、`¥` → U、带空格的 `2024 年` 按基数读、`GPT-4` → GPT 负四。
  共 5 种写法，补齐后接近底噪。
- `omnivoice_mlx/ttstext.py` = markdown-it + wetext + 修补。`generate(normalize=True)` 才启用，它另需两个包。

## 11. 与 mlx-audio 的对比

| | ms/step（短/中/长） | RTF 32 步 |
|---|---|---:|
| mlx-audio bf16 | 56.8 / 58.6 / 77.3 | 0.680 |
| 本项目 bf16 | 35.8 / 38.3 / 56.9 | 0.469 |
| 本项目 fp16 | 31.6 / 33.9 / 48.7 | 0.409 |

## 12. 与官方 torch 的对比（CPU / MPS）

`bench/bench_ref_device.py`，MPS 计时前 `torch.mps.synchronize()`。未设 `PYTORCH_ENABLE_MPS_FALLBACK`，
全程没有算子回落 CPU；codec 是官方在 MPS 上强制留在 CPU 的。这组数两份模型同进程，CPU 那行比单独跑时高，单独跑约 2–3，见「结果」。

| 栈 | 精度 | RTF 32 步 | RTF 8 步 | ms/step（短/中/长） |
|---|---|---:|---:|---|
| torch CPU | fp32 | 4.02 | 1.045 | 308 / 355 / 483 |
| torch MPS | fp32 | 0.980 | 0.363 | 71 / 85 / 115 |
| **本项目 MLX** | **fp16** | **0.227** | 0.063 | **15.4 / 16.9 / 29.6** |
| 本项目 MLX | 8-bit + fp16 | 0.225 | — | 15.4 / 16.9 / 29.6 |

- 同精度同步数下本项目快 5.0 倍，默认配置对 MPS 快 9.6 倍。
- 同一进程里比三种精度，三份模型常驻所以绝对值整体偏高：fp32 1.195、fp16 1.136、bf16 1.540。
  fp16 只比 fp32 快 5 %，MLX 里是 15 %。
- 第 2 节的 fp16 32 步是 0.409，这里同口径是 0.227，中间做了 fast path 和 batch 改动。

## 13. 服务化：单进程 RPS

`server.py`：一个 worker 线程持模型和 GPU，HTTP 线程只排队。MLX 只有一条 GPU stream，而且懒数组不能跨线程求值，
所以并发只能转成 batching：worker 每次把已经排在队里的请求一起走 `generate_batch`。20 句集，每档压 20 s
（`bench/bench_rps.py`）：

| 并发 | max_batch | RPS | 音频秒/墙钟秒 | p50 ms | p95 ms | 实际批大小 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2.78 | 9.3 | 367 | 404 | 1.00 |
| 4 | 1 | 2.69 | 9.0 | 1472 | 1615 | 1.00 |
| 8 | 1 | 2.71 | 9.0 | 2938 | 3151 | 1.00 |
| 1 | 4 | 2.88 | 9.6 | 364 | 393 | 1.00 |
| 4 | 4 | 3.30 | 11.0 | 1193 | 1350 | 2.48 |
| 8 | 4 | **3.56** | **11.8** | 2193 | 2397 | 3.96 |

- 不做 batching 时并发加到 8 也还是 2.7 RPS，只是排队变长：GPU 本来就满，多开客户端不产生吞吐。
- 开了 batching 后 8 并发 3.56 RPS，比单并发高 **24 %**，与第 6 节 B=4–8 拿到的 −25 % 对得上。
- 并发 2 时批大小仍是 1.00：一个请求在算的时候只来得及排进一个，worker 取到它时队列已空。要在低并发下也凑成批
  得让 worker 等一个时间窗，那是拿延迟换吞吐，没做。
- 音频秒/墙钟秒 9.3–11.8，即一台 M2 Max 能同时喂 9–11 路实时语音。

## 复现

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-bench.txt

hf download k2-fsa/OmniVoice --local-dir models/k2-fsa-OmniVoice          # 原版 fp32，各 mlx 目录的 tokenizer 软链指向它
.venv/bin/python scripts/convert.py --out models/mlx-q8-fp16 --dtype float16 --bits 8

export OMNIVOICE_REF_WAV=assets/my-voice.wav      # 自己的 3–4 s 干净单声道录音
export OMNIVOICE_REF_TEXT="它念的那句话，标点照写。"
```

变体名 `s<步数>[-kv<n>][-ue<n>]` 对应上面各表的三列；`uncond_every` 的默认值是 3，所以表里 uncond 写「每步」
的行要显式加 `-ue1`。

各节的数据分别来自：

```bash
lock="bench/benchlock.sh --"        # 排他锁 + 等空载，每条都要走

# 1 移植正确性
$lock .venv-ref/bin/python bench/parity_ref.py
$lock .venv/bin/python bench/parity_mlx.py --dtype float32

# 2 精度与量化
$lock .venv/bin/python bench/bench_models.py --tag flavours \
    --spec fp16=models/k2-fsa-OmniVoice:float16 q8fp16=models/mlx-q8-fp16

# 3 / 4 / 5 采样、步数、隔步重算
$lock .venv/bin/python bench/bench.py --tag steps --model models/mlx-q8-fp16 --runs 3 \
    --variants s32 s16 s16-kv8-ue1 s16-kv8-ue3 s8-kv4-ue1

# 6 多句 batch
$lock .venv/bin/python bench/bench_batch.py --tag batch --model models/mlx-q8-fp16

# 7 长文本 / 流式
$lock .venv/bin/python bench/bench_long.py --tag long --model models/mlx-q8-fp16
$lock .venv/bin/python bench/demo_stream.py

# 8 算子层
$lock .venv/bin/python bench/profile_step.py
$lock .venv/bin/python bench/mm_probe.py
$lock .venv/bin/python bench/gpu_util.py

# 11 与 mlx-audio（另需装 mlx-audio）
$lock .venv/bin/python bench/bench_mlxaudio.py --tag mlxaudio

# 12 与官方 torch
$lock .venv-ref/bin/python bench/bench_ref_device.py --tag ref-dev --devices cpu mps --steps 8 32 --runs 3

# 13 服务化 RPS：先起 server.py，再压
.venv/bin/python server.py --model models/mlx-q8-fp16 --port 8123 &
$lock .venv/bin/python bench/bench_rps.py --url http://127.0.0.1:8123/tts --concurrency 1 2 4 8
```

官方实现要单独一个 venv：`uv venv --python 3.12 .venv-ref` 后装
`torch==2.8.0 torchaudio==2.8.0 omnivoice soundfile`。

CER、speaker similarity 和 UTMOS 需要另外三个模型（ASR、说话人、MOS），本仓库不含。`bench/bench.py` 会把合成的
wav 写到 `out/<tag>/`，用自己的模型打完分之后，`bench/aggregate.py --quality` 能读回结果出表。

## 许可

见 [../README.md](../README.md#许可)。代码 Apache-2.0，权重 CC-BY-NC。
