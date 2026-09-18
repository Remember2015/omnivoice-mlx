# 研究记录：OmniVoice 在 MLX 上的移植与 RTF

安装和用法见 [../README.md](../README.md)。本文记录每个结论是怎么测出来的，负结果一并保留。

测试环境：M2 Max（12 核 CPU / 38 核 GPU / 32 GB），macOS 26.6，MLX 0.32.2。

## 方法

- **正确性**：`bench/parity_ref.py` 用官方实现（torch CPU fp32）在温度全 0 的确定性设置下存下参考 token、prompt ids、
  目标长度、生成 token，`bench/parity_mlx.py` 拿同样输入跑本移植比对。
- **计时**：`bench/bench.py` 一个进程加载一次模型，所有变体按句轮转交错，3 轮取中位数，先各预热一次。
  RTF =（unmask + codec 解码）/ 生成的原始时长（T × 40 ms），不含后处理。所有作业走 `bench/benchlock.sh`（排他锁 +
  等空载）。绝对值随负载漂 10–40 %，只比同进程交错的数。
- **质量**：20 句中文集各合成一次，Fun-ASR-Nano 回读算 CER，CAM++ 算对参考音的余弦相似度，UTMOS22-strong 算自然度。
  这三个模型和跑它们的脚本都不在本仓库，换任何一套都行，绝对值不能跨套比。参考音是一段 3.7 s 的真人录音（92 token）；
  2026-09-16 用 Demucs 减掉了底下的背景音乐，之前之后的质量数不能混着比。
- 变体名 `s<步数>[-kv<n>][-ue<n>][-cfg<比例>]`：kv = 每 n 步刷新一次前缀 KV，ue = 无条件分支每 n 步算一次。
  **注意** `uncond_every` 的默认值后来从 1 改成 3，所以下面早期表里的 `s8-kv4` 今天要写成 `s8-kv4-ue1` 才能复现。

## 1. 移植正确性

参考音预处理后同为 92 token（4 个 codebook 各差 1 个，RVQ 残差路径的浮点差）；用官方参考 token 后 prompt ids 和目标
长度完全一致；fp32 下 8 步确定性生成 3 句逐 token 100 % 一致，解码 + 后处理后波形时长和 RMS 与官方相同。

bf16 / fp16 只有 15–25 % token 一致：迭代 unmask 是混沌过程，一处 argmax 翻转就级联。所以精度的影响只能看 CER /
speaker similarity / UTMOS，不能看 token 一致率。

## 2. 精度与量化：fp16 是关键，量化只省内存

3 句（7 / 17 / 34 字，T = 39 / 63 / 126）× 3 轮中位数，同进程交错。RTF 是三句合并。

| 权重 | 激活 | 常驻 | 权重文件 | 32 步 ms/step（短/中/长） | RTF 32 步 |
|---|---|---:|---:|---|---:|
| fp32 | fp32 | 3.19 GB | — | 36.1 / 38.6 / 58.9 | 0.481 |
| bf16 | bf16 | 1.99 GB | 1242 MB | 35.8 / 38.3 / 56.9 | 0.469 |
| fp16 | fp16 | 1.99 GB | 1242 MB | 31.6 / 33.9 / 48.7 | 0.409 |
| 8-bit g64 | bf16 | 1.56 GB | 829 MB | 37.1 / 42.3 / 64.4 | 0.514 |
| **8-bit g64** | **fp16** | 1.56 GB | 829 MB | **27.8 / 31.5 / 47.1** | **0.381** |
| 4-bit g64 全量化 | fp16 | 1.11 GB | **345 MB** | 28.0 / 31.9 / 47.8 | 0.385 |

- **bf16 和 fp32 一样慢**：M2 的 GPU 没有原生 bf16，MLX 用转换模拟，fp16 比它快 12–15 %。
- **量化更慢只对 bf16 激活成立**（MLX 的 qmm 走 bf16 累加在 M2 上吃亏）。换成 fp16 激活后，量化 matmul 在 M ≈ 200 行时
  比 fp16 GEMM 略快（权重读得少），M ≈ 400 时持平。**量化真正的价值是内存和包体，不是速度。**
- 每步有约 15 ms 固定开销（行长 198 → 254 只多 7 %，254 → 391 多 44 %）：28 层 × ~26 个 kernel 的启动和调度，短句上
  占一步的一半。
- 4-bit 全量化（连 151k 词表 embedding 和 8 个 head）压到 345 MB，但 60 句里出过 1 句错字率 43 %，speaker similarity
  也低 0.01–0.02：拿质量换内存，不是免费的。8-bit + fp16 没有这个代价，定为默认。

## 3. 三种省算量的采样改动：只有 KV cache 可用

三种 NAR 特有的省算量法，fp16 单句，16 个变体同进程交错：

| 手段 | 做法 | 结果 |
|---|---|---|
| **前缀 KV cache** | prompt（120–140 token）的 K/V 每 n 步算一次，中间步只送目标 token | 行长 198→93，每步 −30～37 %；CER 不变，speaker similarity −0.01～−0.02。**用** |
| CFG 截断 | 后半步不跑 uncond 分支 | 省 12–18 %，但 speaker similarity 掉 0.05–0.08（0.75 → 0.67），最低 0.52，还出错字。**弃** |
| 置信度阈值提前揭示 | 每步把置信度超阈值的位提前定下来 | 几乎不触发，每步还要两次 `.item()` 同步，反而慢 5–13 %。**弃** |

双向注意力下 prompt 的隐状态其实依赖目标 token，所以 KV cache 是近似；但 20 句上 CER、相似度都在噪声内。
CFG 截断则是把引导本身去掉：无条件分支在后期步仍在把音色往参考音上推。

## 4. 步数：16 步与 32 步等效，8 步掉自然度，6 步是下限

`models/mlx-q8-fp16`，3 句 × 3 轮计时 + 20 句 × 3 轮质量（n = 60）：

| 变体 | 每步 ms（短/中/长） | RTF | CER | sim 均值 / 最低 | UTMOS |
|---|---|---:|---:|---|---:|
| s32 | 27.7 / 31.3 / 46.9 | 0.381 | 0.73 % | 0.746 / 0.653 | 2.826 |
| s16 | 27.8 / 31.4 / 47.2 | 0.194 | 0.98 % | 0.746 / 0.613 | 2.838 |
| **s16-kv8** | **17.0 / 20.8 / 34.0** | **0.133** | 0.73 % | 0.737 / 0.596 | **2.835** |
| s8 | 27.9 / 31.7 / 47.2 | 0.101 | 0.90 % | 0.735 / 0.607 | 2.701 |
| s8-kv4 | 18.7 / 22.6 / 36.2 | 0.075 | 0.40 % | 0.738 / 0.639 | 2.687 |
| s6-kv6 | — | 0.060 | 0.44 % | 0.729 / 0.583 | 2.532 |
| s4 | — | 0.058 | 1.62 % | **0.700** / 0.550 | 2.327 |

**CER 和 speaker similarity 分不出 8 步和 32 步，UTMOS 能。** 相似度 0.735–0.746 之间全是噪声（±0.005），
但 UTMOS 在 16 → 8 步稳定掉 0.13（约 5 %），每轮种子都一致，8 → 6 → 4 再各掉 0.1、0.26。所以默认取 16 步：
三个指标都与官方 32 步同，RTF 少三分之二。KV cache 在 16 步上不掉 UTMOS，8 步以下开始有代价（s6 → s6-kv6 −0.06）。

4 步相似度掉 0.03–0.04 且开始整句出错。论文的步数消融（英文 WER 8 步翻倍）在这批中文短句上没出现，
**8 → 6 才是这台机器上速度 / 质量的拐点**，单句 RTF 下限约 0.06。

## 5. 无条件分支隔步复用：RTF 降 20 %，三个指标不变

stale CFG（`uncond_every=n`）：缓存步里只每 n 步重算无条件分支，中间步复用上次的无条件 log-prob 做引导。

| 变体 | RTF 3 句 | CER | sim 均值 / 最低 | UTMOS |
|---|---:|---:|---|---:|
| s16-kv8 | 0.133 | 0.33 % | 0.746 / 0.580 | 2.826 |
| s16-kv8-ue2 | 0.109 | 0.33 % | 0.744 / 0.632 | 2.762 |
| **s16-kv8-ue3** | **0.106** | 0.40 % | 0.747 / 0.647 | **2.835** |
| s12-kv6 | 0.104 | 0.75 % | 0.739 / 0.602 | 2.748 |
| s12-kv6-ue2 | 0.086 | 0.83 % | 0.730 / 0.534 | 2.741 |
| s10-kv5 | 0.089 | 1.07 %（1 句 25 %） | 0.747 / 0.626 | 2.732 |

每 3 步算一次 uncond：每步行长 142 → 110，三个指标都与基线同。和第 3 节的 CFG 截断对比，差别在于**引导方向留着、
只是更新得慢**，所以不掉音色。12 步以下 UTMOS 掉 3 %，和步数那节一致。

**默认定为 `SamplerConfig(num_steps=16, cache_refresh=8, uncond_every=3)`，RTF 0.106。**

## 6. 吞吐：多句 batch B=4–8 快 25 %，以及 MLX 的一个缓存坑

20 句集按 B 句一组走 `generate_batch`，cond / uncond 行堆成一个 batch：

| B | RTF | 每步行数 | 折算 GEMM | 峰值内存 |
|---:|---:|---:|---:|---:|
| 1 | 0.106 | 128 | 5.9 TFLOPS | 2.57 GB |
| 2 | 0.093 | 258 | 6.9 | — |
| 4 | 0.086 | 527 | 7.7 | 3.11 |
| 8 | **0.081** | 911 | 8.0 | 3.68 |
| 16 | 0.080 | 1429 | 8.0 | 4.80 |

**B≥4 就到平台**，再大只是多占内存。交互式场景用不上（首批延迟按 B 倍增），这是段落和离线合成的选项。

第一版测出来的却是「越 batch 越慢」（B=1 0.107 → B=8 0.157，GPU 忙闲 97 % → 65 %），主机搭图只 11–74 ms，排除。
原因是 **MLX 的缓冲区缓存默认无上限**：一次大 batch 之后缓存里滞留几 GB 尺寸各异的空闲 buffer，之后每次分配
都走慢路径，连之后的 B=1 都从 0.107 变 0.162，`mx.clear_cache()` 立刻恢复。`mx.set_cache_limit(512 MB)` 已写进
`OmniVoiceTTS.__init__`。

## 7. 长文本和假流式

**长文本**（`generate_long` 复刻官方 chunk 路径）：304 字会议通知估 46 s → 4 块。s8-kv4 顺序 RTF 0.057，把所有块
打进一个 unmask 循环 0.053，**chunk 并行只再省 6–8 %**，每步已经 700+ token，GPU 已经满载。一段 46 s 的文字 2.5 s
出整段。长块 RTF 比短句低是因为每步的固定开销被摊薄，backbone 本身对行长是线性的（200 → 1600 token 每 token 116–143 µs）。

**假流式**（`omnivoice_mlx/stream.py`）：模型内部没有流式，但一句 0.15–0.3 s 就整句出来，放要 1.5–3 s。按标点切分句、
一句一句合成边合边播，304 字 → 22 个分句：s8-kv4 首音频 **165 ms**、s16-kv8 286 ms，两者播放器断流次数都是 0，
余量从第 2 句起就是秒级并一路累加。代价是分句边界的韵律接不上，总时长比整段合成多 10 %。

**分句续接是负结果**：把第 i−1 句自己生成的 token 接进第 i 句的参考，想让韵律跨过分句点。22 个分句逐句 UTMOS：
不续接 2.949、续接 **2.653**（最低 1.29），还慢 10 %。模型自己的输出当参考是分布外的，越接越糟。默认关。

## 8. 算子层已经没有余地

**一步的时间在哪**（`bench/profile_step.py`，q8+fp16 的缓存步）：

| T | 行 | 缓存步 GPU | 28 层注意力 | 28 层 MLP | 头 + CFG + 采样 | 主线程搭图 |
|---:|---:|---:|---:|---:|---:|---:|
| 39 | 78 | 16.0 ms | 9.0 | 7.4 | 0.8 | 1.0 |
| 126 | 252 | 33.4 | 16.9 | 17.7 | 1.5 | 2.5 |

主线程 1–2.5 ms 不是瓶颈（`async_eval` 已经把它藏在 GPU 后面）。四个 8-bit GEMM 占一步的 **67 %**（T=126 时 77 %），
其余是访存型的胶水算子。GPU 忙闲 98–99 %，**没有调度气泡可填**；折算 GEMM 速率 5.9 TFLOPS（峰值 13.6 的 43 %），
长块 7.1（52 %）。效率不到一半是小 M 下 tile 利用率低，不是排队问题。

做了没收益的：qkv / gate_up 沿 N 拼成一个 GEMM、cond/uncond 排成 batch-2、一次带 mask 的 SDPA（每层少 4–6 个 kernel）
→ 16.9 vs 17.3 ms/step，噪声内；沿 N 拼 GEMM 不缩短 K 循环延迟。CFG 三次 log_softmax 合一次（代数恒等）~1 %。
头 GEMM 改 fp16：速度无差但 UTMOS 2.794 vs 2.835，**弃**。`mx.compile` 融合 elementwise 链：噪声内。

**自定义 Metal GEMM 写了两代，都没赢过 MLX。** 第一代多行 GEMV 式（`kernels.py`）数值正确但慢 1.3–5 倍：标量 FMA 每 4 个
乘加配一次 threadgroup 内存读，指令里一半不是 FMA。第二代用 `simdgroup_half8x8` 矩阵指令 + 8 行 M tile
（`kernels_sg.py`，440 种 tile 组合扫参），**只在 M ≤ 80 且 N ≥ 4096 快 1.05–1.3×，其余慢 10–15 %**。
接进模型按形状分派后端到端：7 字句快 6 %，17 字以上一行都碰不到赢的区间，默认关（`custom_gemm=True` 可开）。

两代算子证明的是：**MLX 的量化 GEMM 在这个模型的形状上已经接近硬件上限**，唯一的结构性浪费是把 M 补到 32
（M=32 → 33 时 56 → 90 µs），8 行 tile 拿到的就是这部分；它的 mma 已经跑满机器峰值的 55–75 %。
单句延迟在算子层已无空间，剩下的收益全在算法层（第 4、5 节）和多句 batch。

## 9. codec 瘦身：常驻 1.57 → 0.87 GB，同时去掉 transformers 依赖

解码只要 quantizer + fc2 + acoustic_decoder（90 MB fp32），以前却把 806 MB 的整个 tokenizer（含 HuBERT 编码分支）
fp32 全装进来。改成解码分支单独装 fp16（44 MB 文件，SNR 59 dB）、编码分支只在 `make_prompt` 时懒加载并可释放。

codec 从 mlx-audio vendor 进 `omnivoice_mlx/higgs/`（6 个文件，MIT，改动都标了 `# vendored:`），编码 token 与解码波形
与 mlx-audio 逐位相同。在只装 `mlx numpy soundfile tokenizers safetensors` 的最小 venv 里加载 0.2 s、常驻 0.87 GB、
正常合成，进程里没有 transformers / torch / mlx_audio。

## 10. text normalization：CER 从 7.09 % 降到 0.33 %

34 句（年份、数量、小数、百分比、负数、金额、温度、时间、日期、电话、序号、分数、英文缩写、单位），每句合成后
同一个 ASR 回读算 CER；第四臂直接合成手写的口语形式，是 TTS+ASR 本身的底噪：

| | 原文 | zh_normalization 底座 | wetext 裸跑 | **wetext + 修补** | 底噪 |
|---|---:|---:|---:|---:|---:|
| 平均 CER | 7.09 % | 2.09 % | 1.79 % | **0.33 %** | 0.33 % |
| CER > 10 % 的句子 | 8 / 34 | 2 | 2 | **0** | 0 |

不做规整会丢信息：`-5 度` 念「五度」、`3.5%` 丢「百分之」、`14:30` 念「十四三零」、`1/3` 念「一四三」。
两个现成的规整器各有缺口，互有胜负：zh_normalization 把 `3500 台` 念成「三五零零台」、`¥199` 里的 ¥ 念成「U」；
wetext 把带空格的 `2024 年` 按基数读；两边都把 `GPT-4` 念成「GPT 负四」。两边加起来只有 5 种形状，补齐后就能接近底噪。
最终选 wetext，前面加一层修补（`omnivoice_mlx/ttstext.py`），默认关，因为它另需两个包。

## 11. 对比 mlx-audio 的移植：每步快 26–44 %

mlx-audio 0.5.x 自带的 OmniVoice，同参考音、同目标长度、同 3 句、同步数、同计时定义：

| | 每步 ms（短/中/长） | RTF 32 步 |
|---|---|---:|
| mlx-audio bf16 | 56.8 / 58.6 / 77.3 | 0.680 |
| 本移植 bf16 | 35.8 / 38.3 / 56.9 | 0.469 |
| 本移植 fp16 | 31.6 / 33.9 / 48.7 | 0.409 |

差距来自少做事，不是算子更快：mlx-audio 每步跑两次前向（cond、uncond 各一次 batch-1），8 个 head 对整行算 logits 再切
目标位，每步 `mx.eval` 同步一次再用 `concatenate` 重建 ids 和 embedding，argsort 两次求 rank。本移植把 cond + uncond
拼成一行一次前向（注意力按段算），prompt 的 embedding 只算一次，头只算目标位，reveal 用 `argpartition`，
整步图 `async_eval` 交给 GPU、主线程接着搭下一步。

## 12. 官方 torch 在 MPS 上：能跑，本移植仍快 5 倍（2026-09-18）

前面的官方基准都是 torch CPU。`bench/bench_ref_device.py` 按（设备 × dtype × 步数）同进程交错，MPS 计时前
`torch.mps.synchronize()`，**故意不设 `PYTORCH_ENABLE_MPS_FALLBACK`**：不支持的算子必须抛错，不能悄悄回落 CPU
假装是 MPS 的数。全程没抛错。官方加载器在 MPS 上把 Higgs codec 强制留在 CPU（卷积输出通道 > 65536），所以 MPS
行的 decode 是 CPU 时间，纯 GPU 的比较看 unmask。

| 栈 | 精度 | RTF 32 步 | RTF 8 步 | 每步 ms（短/中/长） |
|---|---|---:|---:|---|
| torch CPU | fp32 | 4.02 | 1.045 | 308 / 355 / 483 |
| torch MPS | fp32 | 0.980 | 0.363 | 71 / 85 / 115 |
| torch MPS | fp16 | 1.136 | 0.385 | 83 / 94 / 126 |
| torch MPS | bf16 | 1.540 | 0.469 | 110 / 125 / 175 |
| **本移植 MLX** | **fp16** | **0.227** | 0.063 | **15.4 / 16.9 / 29.6** |
| 本移植 MLX | 8-bit + fp16 | 0.225 | — | 15.4 / 16.9 / 29.6 |

MPS 比 CPU 快 4.1 倍。torch 在 MPS 上 fp16 只比 fp32 快 5 %（MLX 里是 15 %），bf16 慢 29 %，和第 2 节一致。
**同精度同步数本移植快 5.0 倍（每步 126 → 29.6 ms）；算上 16 步 + kv8 + ue3，对 MPS 最快的一组是 9.6 倍
（1.136 → 0.118）。** 差距在每步的固定开销：MLX 28 层 ~770 个 kernel 就发完，torch MPS 每步多花 4–5 倍的调度。

注：本移植 fp16 32 步在第 2 节是 0.409，这里同口径重测是 0.227，中间的 fast path 和 batch 改动把它提了近 2 倍。
第 5 节的 0.106 与今天的 0.118 在漂移内。

## 没做 / 想过不值得

- **真流式**：架构上没有（NAR，整句一起 unmask），首包 = 整句耗时。s6-kv6 下 7 字句 130 ms、34 字句 224 ms，
  这已经是「首包」。分句并行压得了吞吐，压不了首包。
- **split-K 量化 GEMM**：把 K 循环切段并行再归约，估计短句每步 −30 %；但 N=1024 的形状并行度本来就不够，
  多一次 kernel 启动（10–14 µs）就吃掉差价，没做。
- **`mx.compile`**：可融合的只有 silu×up、残差加、Gumbel 那几条 elementwise 链，≤ 5 %，而且按段切片的整数会固化
  进图里，要按 (P, T) 重 trace。
- 只测了中文单音色克隆；voice design（`instruct`）和其他语言按官方 prompt 格式接了但没评。

## 复现

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -r requirements-bench.txt                # 基准另加 mlx-audio 等
hf download k2-fsa/OmniVoice --local-dir models/k2-fsa-OmniVoice                  # 各 mlx 目录的 tokenizer 软链指向它
.venv/bin/python scripts/convert.py --out models/mlx-q8-fp16 --dtype float16 --bits 8

export OMNIVOICE_REF_WAV=assets/my-voice.wav
export OMNIVOICE_REF_TEXT="它念的那句话，标点照写。"

bench/benchlock.sh -- .venv/bin/python bench/bench.py --tag demo --model models/mlx-q8-fp16 --variants s32 s16-kv8-ue3 s8-kv4-ue1 --runs 3
.venv/bin/python bench/test_thread.py                                             # 工作线程里跑不炸（MLX 跨线程懒数组）
```

官方真值：`uv venv --python 3.12 .venv-ref && uv pip install --python .venv-ref/bin/python torch==2.8.0 torchaudio==2.8.0 omnivoice soundfile`，
然后 `.venv-ref/bin/python bench/parity_ref.py && .venv/bin/python bench/parity_mlx.py --dtype float32`。

CER / speaker similarity / UTMOS 要你自己的三个模型，见「方法」。

## 许可

见 [../README.md](../README.md#许可)。代码 Apache-2.0，权重 CC-BY-NC。
