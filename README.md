# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-08 07:05:47 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [BWTA: Accurate and Efficient Binarized Transformer by Algorithm-Hardware Co-design](https://arxiv.org/abs/2604.03957)

**Authors**: Yifu Ding, Xianglong Liu, Shenghao Jin, Jinyang Guo, Jiwen Lu  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2604.03957v1  

#### Abstract
Ultra low-bit quantization brings substantial efficiency for Transformer-based models, but the accuracy degradation and limited GPU support hinder its wide usage. In this paper, we analyze zero-point distortion in binarization and propose a Binary Weights & Ternary Activations (BWTA) quantization sc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BWTA: Accurate and Efficient Binarized Transformer by Algorithm-Hardware Co-design

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **Zero-point distortion**：在极低比特量化（如二值化）中，激活值分布集中在零附近，传统二值化（binary activation）会强制将接近零的小值映射为 ±1，破坏原始分布，导致显著的量化误差。
- **训练不稳定与收敛困难**：超低位宽模型（如1-bit权重+1-bit激活）难以收敛，且精度下降严重。
- **缺乏GPU原生支持**：现有的超低位宽（ultra-low bitwidth）计算核多依赖FPGA/ASIC，而主流LLM推理平台是GPU，缺乏高效的GPU级支持。

### 提出的新方法与思路
- **Binary Weights & Ternary Activations (BWTA)**：
  - 权重采用 **binary**（{-1, 1}），激活采用 **ternary**（{-1, 0, 1}），保留 `0` 显式表示小值，缓解 zero-point distortion。
- **Smooth Multi-Stage Quantization (SMSQ)**：
  - **Levelwise Degradation Strategy**：逐步减少量化层级（如从17 → 15 → 13 → … → 3 → 1），避免指数级压缩带来的剧烈容量下降。
  - **Magnitude Alignment Projection Factor**：在阶段切换时引入缩放因子 $ p = \frac{\sum |A_{l-1}|}{\sum |A_l|} $，对齐相邻阶段的幅值，实现平滑过渡。
- **定制化的 BWTA MatMul CUDA Kernel**：
  - 支持 binary weight 与 ternary activation 的高效矩阵乘法。
  - 包含 **Instruction-Level Parallel Bitpack**，实现运行时浮点到二/三值的并行打包。
  - 支持多种算术规则（Linear、Attention Score、Value MatMul），兼容各类Transformer结构。

### 相比现有方法的优势
| 方面 | BWTA优势 |
|------|---------|
| **精度** | 显著优于传统二值化方法（如BiBERT、BiT），BERT上平均仅比全精度低3.5%，LLM上接近SOTA量化方法。 |
| **效率** | GPU上实现 **16–24× kernel级加速**（vs FP16），端到端推理达 **216–330 tokens/s**（prefill阶段）。 |
| **通用性** | 可无缝集成至各类Transformer架构（BERT、LLM等），支持线性层与注意力机制。 |
| **硬件适配** | 首个面向GPU的完整超低位宽（<2-bit）算法-硬件协同设计系统，填补了软件与硬件之间的空白。 |

---

## 2. 核心实验方法和设置

### 数据集
- **BERT模型**：
  - **GLUE benchmark**：包含MNLI、QQP、QNLI、SST-2、CoLA、STS-B、MRPC、RTE共8项任务（排除WNLI）。
- **LLM模型**：
  - **Perplexity评估**：Wikitext2 和 C4。
  - **准确性评估**：CommonsenseQA（多个子任务：BoolQ, PIQA, HellaSwag, Winogrande, ARC-E/C, OBQA）。

### 实验设置与评估指标
| 类别 | 设置说明 |
|------|--------|
| **模型** | - BERT：基于DynaBERT蒸馏训练。<br>- LLM：基于Bitnet系列（0.7B, 1.3B, 3B）进行微调。 |
| **量化配置** | - 权重：binary（sign函数）<br>- 激活：ternary（非负用bool，其余用ternary）<br>- 非线性层（Softmax、LayerNorm、GELU）保持高精度（BF16/FP32）以保证稳定性。 |
| **训练策略** | - 多阶段量化：初始阶段使用较多整数量化等级，逐步退化至ternary。<br>- 使用知识蒸馏（中间层输出、logits、attention概率）。<br>- AdamW优化器，cosine学习率调度。 |
| **评估指标** | - BERT：各任务准确率 / Pearson相关系数，取8任务平均。<br>- LLM：Perplexity（越低越好）、CommonsenseQA平均准确率。<br>- 效率：kernel延迟（μs）、TFLOPs、端到端吞吐（tokens/s）、显存占用（GB）。 |

### 基线方法对比
| 类型 | 对比方法 |
|------|--------|
| **BERT量化** | BinaryBERT、BiBERT、BiT、BE-BERT、MLBERT、TernaryBERT、Q2BERT |
| **LLM量化** | GPTQ、AWQ、RTN、PB-LLM、BiLLM、Bitnet、BWN（Binary Weight Network） |
| **硬件加速** | bnb.nn.Linear4bit（bitsandbytes）、cuBLAS HGEMM、TensorRT-LLM、vLLM |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **BERT平均准确率** | **80.4%**（vs 全精度83.9%，仅↓3.5%） |
| **BERT存储大小** | **13.4 MB**（与其他1-bit方法相当） |
| **LLM Perplexity** | 在Wikitext2上达 **15.58**（Bitnet-b1.58-1.3B），显著优于GPTQ/PB-LLM |
| **CommonsenseQA准确率** | 达 **44.8%**（30%层替换），接近Bitnet（47.2%） |
| **Kernel加速比** | 相比FP16，**Linear层加速23.6×**，**Attention加速16.7–18.5×** |
| **端到端吞吐（Prefill）** | 批大小16时达 **330 tokens/s**（比Bitnet提升近2倍） |
| **解码速度（Decode）** | 达 **12–15 tokens/s**，优于所有4-bit权重量化方法 |

### 与基线方法的对比结果
#### ✅ BERT结果（Table 1）
- BWTA在所有任务上均优于其他1-bit方法，平均达 **80.4%**，远超BiT（77.5%）、BiBERT（67.0%）。
- 即使相比更高位宽方法（如W1A4），仍具竞争力。

#### ✅ LLM结果（Table 2）
- 在相同模型规模下，BWTA在更低平均bit数（W1.3/A6）下实现了更优的perplexity和accuracy。
- 例如，在Bitnet-b1.58-1.3B上，BWTA (30% layers) 在更低bit下达到 **44.8%** 准确率，优于BWN（42.5%）。

### 消融实验结果
#### 🔍 多阶段量化策略对比（Fig. 9）
- **Bitwise vs Levelwise**：
  - Bitwise（指数退化）在阶段切换时出现明显loss spike和accuracy drop。
  - Levelwise（线性退化）过渡平稳，恢复更快。
- 加入 **Projection Factor** 后，loss曲线更平滑，收敛更快，scaling factor初始化更接近最优。

#### 🔍 投影因子有效性（Fig. 10）
- “None”、“Mean”、“Search”三种初始化方式均不如提出的 **Projection Factor**。
- Projection Factor 能有效对齐幅值，避免重新搜索，加快收敛。

#### 🔍 激活分布可视化（Fig. 12）
- Bitwise策略导致量化bin利用不均衡，信息损失大。
- Levelwise策略能更好保留原始分布形态，尤其在零值附近。

#### 🔍 损失曲面分析（Fig. 13）
- BWTA模型的loss surface更平滑，标准差更低（0.091 vs BiT的0.117），表明其对扰动更鲁棒，收敛到更优解。

---

## 4. 关键结论和发现

### 主要发现
1. **Ternary Activation 有效缓解 Zero-point Distortion**：
   - 显式保留 `0` 可显著降低量化误差，尤其是在注意力分数和中间激活中。
2. **Smooth Multi-Stage Quantization 提升训练稳定性**：
   - 渐进式退化 + 幅值对齐，使超低位宽模型可在有限epoch内稳定收敛。
3. **算法-硬件协同设计带来真实加速**：
   - 定制CUDA kernel结合bitwise MMA指令与并行bitpack，实现高达 **24× kernel加速**。
4. **BWTA兼具高精度与高效率**：
   - 是首个在BERT和LLM上同时逼近全精度性能并实现显著加速的1-bit级方案。

### 方法的局限性
- **非线性层未量化**：Softmax、LayerNorm等仍使用BF16/FP32，限制了极致压缩潜力。
- **部分层选择性替换**：在LLM中仅替换30%敏感度低的层，未能完全全模型二值化。
- **依赖预训练低比特模型**：实验基于Bitnet初始化，可能影响泛化性。
- **当前仅支持NVIDIA GPU**：未扩展至FPGA/ASIC或其他硬件平台。

### 未来工作方向
- **Fully Quantized Nonlinear Operators**：
  - 探索 integer-only Softmax（如SoftmAP）、AILayerNorm 等低比特非线性算子。
- **跨硬件平台迁移**：
  - 将BWTA思想迁移到FPGA/ASIC，进一步优化能效比。
- **自动化层选择与量化粒度**：
  - 动态识别可安全量化的层，实现自适应混合精度。
- **扩展至更多模态**：
  - 应用于Vision Transformer、Multimodal Models等更广泛架构。

---

> **一句话总结**：  
> BWTA通过提出 **Binary Weights & Ternary Activations** 架构与 **Smooth Multi-Stage Quantization** 训练框架，结合定制化 **CUDA kernel**，首次实现了在GPU上兼具高精度（接近全精度）与高效率（16–24×加速）的超低位宽Transformer推理，推动了1-bit级模型实用化进程。

</details>

---

### 2. [Diagonal-Tiled Mixed-Precision Attention for Efficient Low-Bit MXFP Inference](https://arxiv.org/abs/2604.03950)

**Authors**: Yifu Ding, Xinhao Zhang, Jinyang Guo  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2604.03950v1  

#### Abstract
Transformer-based large language models (LLMs) have demonstrated remarkable performance across a wide range of real-world tasks, but their inference cost remains prohibitively high due to the quadratic complexity of attention and the memory bandwidth limitations of high-precision operations. In this...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Diagonal-Tiled Mixed-Precision Attention for Efficient Low-Bit MXFP Inference*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Transformer 模型中的 **self-attention** 因其 $O(N^2)$ 的序列长度复杂度成为推理瓶颈。尽管低比特量化（如 MXFP4、MXFP8）可提升计算效率，但直接应用会导致显著的 **accuracy degradation** 和 **quantization error**。此外，传统分离式的量化流程引入额外内存访问和 kernel launch 开销，削弱了低比特带来的收益。

本文针对以下两个核心挑战提出解决方案：
- **Challenge 1**: 低比特 MXFP 格式（尤其是 MXFP4）在 attention score 计算中引入严重量化误差。
- **Challenge 2**: 非融合的量化操作导致冗余内存访问和调度开销，影响端到端效率。

---

### 🚀 提出的新方法与创新点

#### （1）**Diagonal-Tiled Mixed-Precision Attention (DMA)**  
一种新型混合精度注意力机制，在 **tiling 层面**动态划分高/低精度区域：
- **对角线窗口（diagonal window）内**：保留关键 token 对（如近期上下文），使用高精度格式（如 MXFP8）进行计算。
- **其余区域**：采用高效低比特格式（如 MXFP4/NVFP4）加速处理。
- 设计灵感来自观察：attention score 中最重要的信息集中在矩阵对角线附近。

#### （2）**Fully Fused Quantization Kernel（基于 Triton 实现）**
将整个低比特预处理流水线集成进一个 GPU kernel：
- 包括：softmax scaling、quantization scale 计算、MXFP 编码、FP4 packing、scale conversion（to E8M0）等步骤。
- 消除中间结果存储，减少 memory traffic 与 kernel 启动延迟。

#### （3）兼容多种 attention 类型
- 支持 **causal attention** 和 **non-causal attention**，通过调整 tile 迭代范围实现。

---

### 🔍 相比现有方法的优势

| 方面 | 现有方法（如 SageAttention、INT-FlashAttention） | 本文 DMA |
|------|---------------------------------------------|--------|
| 精度恢复 | 使用额外 GEMV 补偿误差 → 增加计算负担 | 对角线区域原生高精度 → 更高效准确 |
| 内存效率 | 多阶段 kernel 分离执行 → 冗余访存 | 全流程融合 kernel → 极大降低 memory footprint |
| 性能增益 | 通常 2–3× speedup | 在 B200 上达 74.2× ~ 80.1× kernel speedup（vs unfused） |
| 实用性 | 多为 plug-and-play 模块 | 端到端优化，适配下一代 Blackwell 架构 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **LongBench**：用于评估长文本理解能力，涵盖多语言、多任务场景。
  - 序列长度范围：2.5K ~ 30K tokens
  - 子任务包括：`repobench-p`, `samsum`, `triviaqa`, `passage_retrieval`, `narrativeqa` 等。

### ⚙️ 模型
- **LLaMA-3.1-8B**
- **LLaMA-3.2-3B**

### 💻 实验平台
- 单张 **NVIDIA B200 GPU**
- 使用 **Triton** 编写 fused kernel
- 对比基线运行于 PyTorch SDPA（BF16）

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **准确性** | Average score on LongBench, Cosine Similarity, PSNR, RMSE, Rel. L1 Distance |
| **效率** | Attention kernel latency (ms), Total runtime, Throughput (TOPS) |
| **消融分析** | 不同 block size、fusion stage、quantization granularity 下的表现 |

### 🆚 基线方法
- **Native SDPA**（PyTorch 默认，BF16）
- **MXFP4**
- **NVFP4**
- **MXFP8**
- **Unfused pipeline**（各量化步骤独立执行）

---

## 3. 主要实验结果和性能指标

### 📈 准确性结果（LongBench 平均得分）

| Model | Baseline (SDPA) | DMA (Ours) | 提升 |
|-------|------------------|------------|------|
| LLaMA-3.1-8B | 44.11 | **46.43** | +2.32 |
| LLaMA-3.2-3B | 35.84 | **37.20** | +1.36 |

> ✅ 在多个子任务上均有明显提升，尤其在 `repobench-p`（+11.77）、`triviaqa`（+10.88）、`trec`（+6.5）等需要强上下文建模的任务中表现突出。

---

### ⏱ 效率结果（Latency Breakdown）

| Format | Attn (ms) | Quant (ms) | **Total (ms)** |
|--------|-----------|------------|----------------|
| MXFP4 | 12.491 | 0.242 | 12.980 |
| NVFP4 | 12.941 | 0.204 | 13.404 |
| MXFP8 | 16.480 | 0.044 | 16.771 |
| **DMA (128/128)** | **7.110** | **0.382** | **7.776** ✅ |
| DMA (256/256) | 15.056 | 0.382 | 15.720 |

> ✅ **总延迟降低至 7.776ms**，相较 MXFP4 快 **~1.7×**，相较 MXFP8 快 **~2.2×**

---

### 🔬 消融实验结果

#### （1）Kernel Fusion 贡献（L=2k / L=8k）

| Fusion Stage | L=2k (μs) | L=8k (μs) | Speedup vs Unfused |
|--------------|-----------|-----------|--------------------|
| Fully Unfused | 7262.41 | 22628.96 | 1× |
| +Encode | 802.90 | 1113.77 | ~9× |
| +Pack | 740.64 | 942.67 | ~10× |
| +Scale Cvt | 179.97 | 299.69 | ~40× |
| **Full DMA (MP)** | **97.87** | **282.46** | **74.2× / 80.1×** ✅ |

> ✅ **kernel fusion 是性能飞跃的关键**，特别是 scale conversion 与 mixed-precision quantization 的融合。

#### （2）Mixed-Precision Tile Size 影响（相似性 vs 延迟）

| Diag./Sink | BitHigh (%) | Cos Sim ↑ | Rel. L1 ↓ | RMSE ↓ | PSNR ↑ | Latency |
|------------|-------------|-----------|----------|--------|--------|---------|
| 0 / 0 | 0.0 | 0.778 | 0.620 | 0.065 | 43.715 | — |
| 128 / 128 | 2.30 | **0.822** | **0.539** | **0.059** | **44.657** | ✅ |
| 512 / 512 | 9.22 | 0.826 | 0.542 | 0.058 | 44.731 | ↑↑↑（不划算）|

> ✅ **128/128 是最佳平衡点**：仅 2.3% 高精度区域即可接近最优表示质量，同时保持高性能。

#### （3）Quantization Granularity 对比

| Granularity | Latency | Cos Sim ↑ | RMSE ↓ | PSNR ↑ |
|-------------|---------|-----------|--------|--------|
| Per-Tensor | 6.276ms | 0.732 | 0.067 | 43.479 |
| Per-Block | 6.366ms | 0.736 | 0.067 | 43.531 |
| **Per-Token** | **7.131ms** | **0.822** | **0.059** | **44.657** |

> ✅ **Per-Token granularity 最优精度**，但代价是更高延迟；DMA 结合 diagonal tiling 可在较低成本下逼近该性能。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **对角线区域主导 attention 质量**：保留对角线附近 token 的高精度表示可有效缓解低比特量化误差。
2. **kernel fusion 极致优化至关重要**：分离式量化带来不可忽视的系统开销，全融合 kernel 可实现数量级加速。
3. **DMA 实现 lossless 推理**：在 LongBench 上不仅未损失性能，反而平均得分提升 1.3~2.3 分。
4. **128×128 tile size 为最优配置**：兼顾精度与并行效率，适合当前硬件架构。

---

### ⚠️ 局限性
- 当前验证集中于 **text-only LLMs**，尚未扩展至 vision 或 multimodal 模型。
- 实验主要基于 **LLaMA 系列模型**，缺乏在其他架构（如 Mistral、Phi）上的泛化测试。
- **极长序列（>32K）下的 scalability** 尚未充分验证。
- 混合精度策略为固定窗口设计，**未支持动态 adaptive tiling**。

---

### 🔮 未来工作方向
- 扩展至 **vision transformer** 和 **multimodal models**（如 LLaVA）。
- 探索 **adaptive diagonal window sizing**，根据输入内容动态调整高精度区域。
- 在更多硬件平台（如 H100、GB200 NVL）上验证跨设备兼容性。
- 将 DMA 思路应用于 **training phase**，探索低比特训练可行性（参考 SageAttention3）。

---

> 🔗 **代码已开源**：[https://github.com/yifu-ding/MP-Sparse-Attn](https://github.com/yifu-ding/MP-Sparse-Attn)

</details>

---

### 3. [See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs](https://arxiv.org/abs/2604.05650)

**Authors**: Yicheng Ji, Jun Zhang, Jinpeng Chen, Cong Wang, Lidan Shou, Gang Chen, Huan Li  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.05650v1  

#### Abstract
Video Large Language Models (Video-LLMs) excel in video understanding but suffer from high inference latency during autoregressive generation. Speculative Decoding (SD) mitigates this by applying a draft-and-verify paradigm, yet existing methods are constrained by rigid exact-match rules, severely l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **高推理延迟瓶颈**：Video-LLMs 在视频理解任务中表现出色，但由于其自回归生成机制，在处理长序列视觉 token 时面临严重的推理延迟问题。
- **传统 Speculative Decoding (SD) 效率受限**：现有的 SD 方法依赖严格的“精确匹配”（exact-match）验证规则，导致大量语义上正确但字面不一致的 draft token 被错误拒绝，限制了加速潜力。

### 提出了什么新方法或新思路
提出 **LVSPEC** ——首个专为 Video-LLMs 设计的**训练免费（training-free）松散推测解码（loosely speculative decoding）框架**，其核心思想是：
- **视觉相关性驱动的差异化验证策略**：基于一个关键洞察——生成质量由少数关键的“视觉锚点 token”（如颜色、物体名称）决定，而大量语法填充词（如“the”, “and”）对语义完整性影响极小。
- **双轨验证机制**：
  - 对 **Visual-Relevant Tokens**（视觉相关 token）进行**严格验证**（strict verification），确保事实准确性。
  - 对 **Visual-Irrelevant Tokens**（视觉无关 token）进行**宽松接受**（loose verification），提升接受长度。
- **位置偏移容忍机制（Position Shift-Tolerant, PST）**：允许因顺序差异（如“wooden deck and multiple sails” vs “multiple wooden deck and multiple sails”）导致的位置错位 token 被接受，进一步提高利用率。

### 相比现有方法的优势
- **打破 rigid exact-match 瓶颈**：首次将“松散匹配”引入 Video-LLMs 推理，超越传统 SD 的理论上限。
- **无需额外训练**：完全 training-free，部署成本低。
- **高效轻量**：视觉相关性识别模块计算开销极小（仅占总延迟 0.42%）。
- **通用性强**：适用于不同模型家族（Qwen2.5-VL, LLaVA-OneVision）和多种任务。

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个主流视频理解基准上进行评估：
| 数据集 | 任务类型 | 输出长度 | 帧数 | 实例数 |
|--------|--------|--------|--------|--------|
| **VDC** (Video Detail Caption) | 视频详细描述 | 长 | 64 | 120 |
| **VDD** (Video Detail Description) | 视频整体描述 | 长 | 128 | 30 |
| **MovieChat** | 开放式问答 | 中等 | 128 | 100 |
| **Video-MME** | 多项选择题 | 短 | 64 | 500 |

### 实验设置和评估指标
- **目标模型（Verifier）**：
  - `Qwen2.5-VL-32B`
  - `LLaVA-OneVision-72B`
- **草稿模型（Drafter）**：
  - `Qwen2.5-VL-7B` / `LLaVA-OneVision-7B`（带 90% 视觉 token 剪枝）
- **两种 SD 设置**：
  - **Standard-SD (Std.-SD)**：小模型作 draft，大模型作 verify。
  - **Self-SD**：同一模型剪枝后作 draft，原模型作 verify。
- **评估指标**：
  - **效率指标**：Speedup Ratio（加速比）、Mean Accepted Length $ \mathbb{E}[T] $
  - **性能指标**：Accuracy、Rating Score（通过 LMMs-Eval 使用 LLM judge 评分）

### 基线方法对比
| 类别 | 方法 | 说明 |
|------|------|------|
| **Lossless SD** | NAIVE SD | 直接使用小模型作为 draft |
| | SPECVLM | 当前 SOTA，采用 uniform 视觉 token 剪枝 |
| **Loosely SD** | FLY | 基于熵和延迟窗口的松散接受机制 |
| | FLY⁺ | FLY 的更宽松变体（仅用熵门控） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
LVSPEC 在保持近乎无损性能的前提下实现了显著加速：

| 目标模型 | 方法 | Speedup | Mean $ \mathbb{E}[T] $ | 性能保留率 |
|--------|------|--------|------------------|------------|
| Qwen2.5-VL-32B | LVSPEC (Ours) | **2.70×** | **7.76** | >99.8% |
| LLaVA-OneVision-72B | LVSPEC (Ours) | **2.94×** | **7.34** | >99.8% |

> 注：性能保留率指相对于目标模型原始输出的准确率/得分保留比例。

### 与基线方法的对比结果
- **相比 SPECVLM（当前最优训练免费 SD）**：
  - 平均 **Mean Accepted Length 提升 136%**
  - 平均 **Speedup 提升 35%**
  - 在 VDC 上，LVSPEC 的 $ \mathbb{E}[T] = 7.76 $，远高于 SPECVLM 的 3.29。
- **相比松散方法 FLY/FLY⁺**：
  - 加速效果更优（如在 VDC 上提速达 **1.58×** 于 FLY）
  - 性能稳定性更高（FLY 平均保留率仅 ~87%，而 LVSPEC >99.8%）
- **Pareto 前沿优势**：LVSPEC 显著推动了 accuracy-speedup 的帕累托前沿，实现更高加速的同时维持更高性能。

### 消融实验结果
#### （1）视觉语义引导的有效性（w/o PST）
- 移除 PST 后，$ \mathbb{E}[T] $ 从 7.76 降至 7.27，Speedup 从 2.70× 降至 2.54×。
- 表明 PST 有效回收了因位置偏移被误拒的 token。

#### （2）超参数敏感性分析
| 超参数 | 设置 | $ \mathbb{E}[T] $ | Speedup | 保留率 |
|-------|------|------------------|---------|--------|
| $ \lambda $ (松弛比例) | 0.0 | 3.41 | 1.40× | 100% |
| | 0.7 | 7.27 | 2.54× | 98.5% |
| | 0.9 | 8.97 | 3.03× | 92.8% ↓ |
| $ N $ (Top-N 视觉 token) | 1 | 7.15 | — | ↓ |
| | 10 | **7.27** | **2.54×** | **98.5%** ✅ |
| | 100 | 7.47 | 2.65× | 93.4% ↓ |

- 结论：适度松弛（$ \lambda=0.7 $）和适中 $ N=10 $ 可在速度与精度间取得最佳平衡。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **视觉锚点稀疏性假设成立**：视频生成的质量主要由少量视觉相关的“锚点 token”决定，其余 token 可安全地进行宽松验证。
2. **松散验证可突破理论瓶颈**：通过利用视觉稀疏性，LVSPEC 将有效失败率从 $ e $ 降低至 $ pe $（$ p $ 为视觉密度），理论上可线性提升接受长度。
3. **视觉语义指导优于纯语言启发**：相比仅依赖熵的 FLY，LVSPEC 利用跨模态相似性进行 token 级别的验证决策，更加精准有效。

### 方法的局限性
1. **超参数需调优**：对于视觉信息密度差异大的任务，可能需要调整 $ \lambda $ 和 $ N $。
2. **复杂逻辑推理场景未覆盖**：对于涉及否定词（如“not”）或逻辑连接词的任务，当前的宽松策略可能不够安全。
3. **PST 模块适用范围有限**：位置偏移容忍机制依赖于 draft 与 target 输出模式相似，泛化性有待验证。
4. **未结合 fine-tuned draft model**：当前 focus 在 verification 策略，未来可探索与专门训练的 draft model 结合。

### 未来工作方向
- 扩展至更复杂的视觉推理任务（如 Video-of-Thought）。
- 引入自适应机制动态调整 $ \lambda $。
- 结合 KV Cache 压缩、In-context draft 等其他加速技术。
- 探索与 retrieval-based 或 tree-based speculation 的深度融合。

--- 

> **总结**：LVSPEC 通过“见林不见树”（See the Forest for the Trees）的哲学，提出了一种基于视觉语义指导的松散推测解码范式，成功打破了 Video-LLMs 推理中的精确匹配瓶颈，在几乎无损性能的前提下实现了高达 **2.94×** 的端到端加速，为高效视频理解提供了新的实用路径。

</details>

---

### 4. [HybridKV: Hybrid KV Cache Compression for Efficient Multimodal Large Language Model Inference](https://arxiv.org/abs/2604.05887)

**Authors**: Bowen Zeng, Feiyang Ren, Jun Zhang, Xiaoling Gu, Ke Chen, Lidan Shou, Huan Li  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.05887v1  

#### Abstract
Multimodal Large Language Models (MLLMs) have advanced unified reasoning over text, images, and videos, but their inference is hindered by the rapid growth of key-value (KV) caches. Each visual input expands into thousands of tokens, causing caches to scale linearly with context length and remain re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：HYBRIDKV: Hybrid KV Cache Compression for Efficient Multimodal Large Language Model Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Multimodal Large Language Models (MLLMs) 在处理图像、视频等多模态输入时，会将每个视觉输入扩展为数千个 tokens，导致 **Key-Value (KV) cache** 随上下文长度线性增长。这些 KV cache 必须在 GPU 内存中全程保留以支持自回归解码，造成严重的 **内存开销和延迟问题**，限制了 MLLMs 在长上下文场景下的实际部署。

尽管已有多种 KV cache 压缩方法（如 token-level、layer-level、head-level），但它们大多仅关注 **预算分配（budget allocation）**，忽略了不同 attention head 具有异构行为这一关键事实——有些 head 可安全剪枝，而另一些则需动态检索。

---

### **提出的新方法与新思路**
作者提出了 **HYBRIDKV**，一种基于 **混合策略的 KV cache 压缩框架**，其核心思想是：

> **不仅要进行精细化的预算分配，更要根据 attention head 的行为特性，采用不同的压缩策略。**

该方法分为三个阶段：

1. **Head 分类（Text-centric Head Classification）**  
   利用 **text-centric attention sparsity** 在 prefill 阶段预测每个 head 是“静态”还是“动态”：
   - **Static Heads**：注意力集中在少数稳定 token 上 → 可安全剪枝（pruning）
   - **Dynamic Heads**：注意力随时间变化广泛分布 → 需要运行时检索（retrieval）

2. **分层预算分配（Top-down Budget Allocation）**  
   - 第一层：在 **head type 层面** 分配总预算（`B_stat`, `B_dyn`），通过系数 `r` 控制平衡
   - 第二层：在 **individual head 层面** 自适应分配，static heads 使用 sparsity score 加权分配，dynamic heads 均匀分配

3. **混合压缩机制（Hybrid KV Cache Compression）**  
   - 对 **static heads**：采用 **text-prior pruning**，保留文本 token 和局部显著视觉 token
   - 对 **dynamic heads**：采用 **chunk-wise retrieval**，将 KV cache 分块存储于 CPU，按需加载重要 chunk 至 GPU

---

### **相比现有方法的优势**
| 维度 | 现有方法 | HYBRIDKV |
|------|--------|----------|
| **粒度** | 多数停留在 token/layer/head-level 分配 | 引入 **head behavior-aware** 设计 |
| **策略统一性** | 所有 head 使用相同策略（全剪枝或全检索） | **差异化策略**：静态剪枝 + 动态检索 |
| **上下文感知** | 多为离线或固定模式 | **在线分类 + 输入自适应** |
| **效率-精度权衡** | 易因过度剪枝导致性能下降 | 更精准保留关键信息，甚至提升性能 |

> ✅ **核心优势**：首次系统性地识别并利用了 MLLM 中 attention head 的 **异构行为模式**，实现了更高效且准确的 KV cache 压缩。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
#### 图像任务（Image Tasks）：
- **MileBench** 包含以下子任务：
  - CL-CH（CLEVR-Change）: 视觉差异描述
  - DocVQA / SlideVQA：文档理解
  - WebQA / WikiVQA：网页知识问答
  - MMCoQA / MM-QA：跨模态对话问答
  - STD（Spot-the-Diff）：图像差异检测

#### 视频任务（Video Tasks）：
- **VATEX**：多语言视频描述生成
- **NextQA**：基于视频的因果与时序推理
- **Video-ChatGPT**：长视频对话理解（含 CI, DO, CU, TU, CO 五个维度评分）

---

### **实验设置与评估指标**

| 项目 | 设置 |
|------|------|
| **模型** | Qwen2.5-VL-3B, Qwen2.5-VL-7B, LLaVA-OneVision-7B |
| **硬件平台** | NVIDIA L40S GPU |
| **帧采样** | 视频输入每段采样 64 帧 |
| **KV 缓存比例** | 主要测试 10% 和 20% 极高压缩比 |
| **注意力实现** | 使用 FlashAttention-2 加速计算 |

#### **评估指标**
| 任务类型 | 指标 |
|---------|------|
| 文本生成一致性 | ROUGE-L |
| QA 准确率 | Exact Match Accuracy |
| 视频描述质量 | BLEU-4, METEOR, ROUGE-L, CIDEr |
| 回答语义相似度 | WUPS（Wu-Palmer Similarity） |
| 对话能力评分 | GPT-4o-mini 自动生成的 CI/DO/CU/TU/CO 分数（0–5） |

---

### **基线方法对比**
| 方法 | 类型 | 特点 |
|------|------|------|
| **SNAPKV** | Token-level | 基于累积注意力保留关键 token |
| **LOOK-M** | Token-level | 优先保留文本 token，合并 KV |
| **MADAKV** | Layer-level | 按模态偏好分配各层预算 |
| **SPARSEMM** | Head-level | 使用离线视觉 head score 进行非对称分配 |
| **FULL CACHE** | —— | 不压缩，作为性能上限基准 |

> ⚠️ 所有 baseline 均未区分 head 行为类型，仅使用单一压缩策略。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 指标 | 结果 |
|------|------|
| **KV Cache 内存减少** | 最高达 **7.9×**（10% budget 下） |
| **解码速度提升** | 达到 **1.52×** 加速 |
| **平均准确率损失** | <1.3%（图像任务），部分任务反超 full cache |
| **最高性能增益** | 在 VATEX 上 **超过 full cache 基线**（METEOR 提升 0.01） |

---

### **与基线方法的对比结果**

#### ✅ **图像任务（Table 1）**
- 在 **8 项图像任务** 上，HYBRIDKV 平均表现优于所有 baseline
- 相比 SPARSEMM（最强 head-level 方法）：
  - Qwen2.5-VL-7B 上平均得分高出 **1.45%**
  - WebQA 上达到 **76.00 vs. 70.50**
- 即使在极端压缩（10% KV）下仍保持接近 full cache 性能

#### ✅ **视频任务（Table 2）**
- 在 **VATEX** 上，HYBRIDKV 实现全面领先：
  - BLEU-4: **0.2266 vs. 0.2181**（full cache）
  - METEOR: **0.2236 vs. 0.2209**
  - CIDEr: **0.4774 vs. 0.4628**
- 在 **NextQA** 和 **Video-ChatGPT** 上也显著优于其他方法
- SPARSEMM 在 VATEX 上出现 **5% 平均准确率下降**，而 HYBRIDKV 无此问题

#### ✅ **效率结果（Table 3）**
| 方法 | KV Budget | GPU Memory | Latency (ms/token) |
|------|-----------|------------|---------------------|
| FULL CACHE | 100% | 1.73 GB | 58.94 |
| HYBRIDKV | 20% | 0.40 GB | 42.08 |
| HYBRIDKV | 10% | **0.22 GB** | **38.65** |

> 💡 **结论**：仅用 10% KV cache，即可实现 **7.9× 内存节省 + 1.52× 解码加速**

---

### **消融实验结果（Ablation Study）**

#### 🔍 **Head 分类的影响（Table 4）**
| 变体 | STD | WebQA | NextQA | Latency |
|------|-----|-------|--------|--------|
| 所有 head 设为 static | ↓ | ↓ | ↓ | 34.13 |
| 所有 head 设为 dynamic | ↔ | ↔ | ↔ | ↑ 48.15 |
| **HYBRIDKV（真实分类）** | ✅ | ✅ | ✅ | 38.65 |

> ❗ 将所有 head 视为同一类型会导致性能下降或延迟上升；**动态分类至关重要**

#### 🔍 **预算分配的影响（Table 5）**
| 变体 | STD | WebQA | NextQA | Latency |
|------|-----|-------|--------|--------|
| w/o H（无 head-level 分配） | ↓ | ↔ | ↓ | ↔ |
| w/o (H&HT)（无 head-type & head-level） | ↓↓ | ↓ | ↓↓ | ↑ 44.67 |
| **HYBRIDKV** | ✅ | ✅ | ✅ | ✅ |

> ✅ **分层预算分配** 显著提升准确率并控制延迟，验证了设计有效性

---

## **4. 关键结论和发现**

### **主要发现**
1. **Attention Heads 存在异构行为模式**：
   - 可明确划分为 **static** 和 **dynamic** 两类
   - 二者在 prefill 阶段的 **text-centric attention sparsity** 可有效预测其后续行为

2. **单一压缩策略不适用于所有 heads**：
   - 强行统一剪枝或检索会造成信息丢失或冗余传输
   - **hybrid 策略（pruning + retrieval）更符合实际需求**

3. **上下文感知分类 + 分层预算分配 = 更优性能**：
   - 在线分类 + 自适应分配可显著提升压缩效果
   - 甚至能在高压缩比下 **超越 full cache 模型性能**（如 VATEX）

4. **极高压缩比下仍可保持高质量生成**：
   - 仅保留 10% KV cache，性能几乎无损
   - 适用于长视频、多图推理等内存敏感场景

---

### **方法的局限性**
1. **依赖预设阈值进行 head 分类**：
   - 当前使用固定 threshold `θ=0.9`，可能不适用于所有模型规模或领域
   - 分类鲁棒性有待进一步增强

2. **未结合训练过程优化**：
   - 完全为 inference-time 方法，未探索与 fine-tuning 联合优化的可能性

3. **评估集中于图像与视频任务**：
   - 尚未验证在音频或多轮长文本等其他模态上的泛化能力

4. **chunk-wise retrieval 引入额外 I/O 开销**：
   - 虽然总体更快，但在某些低带宽环境下可能成为瓶颈

---

### **未来工作方向**
- 探索 **learned 或 adaptive classifier** 替代手工阈值
- 将 head 分类机制融入 **training/fine-tuning 流程**，实现 co-design
- 扩展至 **audio-grounded MLLMs** 或 **extremely long-context reasoning**
- 结合其他压缩技术（如 **quantization**, **speculative decoding**）形成复合优化方案
- 支持 **adaptive thresholding** 和 **joint strategy learning**

---

## **总结**
HYBRIDKV 是首个从 **attention head 行为异质性** 出发设计的 KV cache 压缩框架。它通过 **text-guided head classification + hierarchical budget allocation + hybrid pruning/retrieval** 三阶段机制，在仅保留 10% KV cache 的情况下实现了 **7.9× 内存压缩** 和 **1.52× 解码加速**，且多数任务性能持平甚至超越 full cache 模型。该工作不仅提供了高效的 MLLM 推理工具，也为未来的 **context-aware、strategy-driven 压缩范式** 奠定了基础。

</details>

---

### 5. [Scalable Variational Bayesian Fine-Tuning of LLMs via Orthogonalized Low-Rank Adapters](https://arxiv.org/abs/2604.03388)

**Authors**: Haotian Xiang, Bingcong Li, Qin Lu  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.03388v1  

#### Abstract
When deploying large language models (LLMs) to safety-critical applications, uncertainty quantification (UQ) is of utmost importance to self-assess the reliability of the LLM-based decisions. However, such decisions typically suffer from overconfidence, particularly after parameter-efficient fine-tu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scalable Variational Bayesian Fine-Tuning of LLMs via Orthogonalized Low-Rank Adapters

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在将 **Large Language Models (LLMs)** 部署到安全关键应用（如医疗、法律）时，**Uncertainty Quantification (UQ)** 至关重要，以评估模型决策的可靠性。然而，现有的 **Parameter-Efficient Fine-Tuning (PEFT)** 方法（如 LoRA）在小样本下游任务上微调后，常导致模型**过度自信（overconfidence）**，无法准确表达预测不确定性。

现有两种主流 UQ 方法存在显著缺陷：
- **Laplace Approximation (LA)**：基于 MAP 估计进行后处理，仅捕捉单个模式附近的局部不确定性，效果受限于训练轨迹。
- **Variational Bayesian (VB) 方法**（如 BLoB）：虽能联合优化均值和协方差，但推理时需对整个 LLM 主干网络进行多次前向传播（Monte Carlo 采样），**计算开销大，难以部署**。

此外，标准 LoRA 存在 **rank collapse** 问题，即其有效秩（stable rank）接近 1，导致特征空间被压缩，破坏了语义距离，损害了后续贝叶斯推理的质量。

### 提出了什么新方法或新思路

本文提出 **PoLAR-VBLL**，一个统一且可扩展的框架，结合以下两个核心技术：

1. **Polar-decomposed Low-rank Adapter Representation (PoLAR)**  
   - 采用正交分解参数化：`ΔW = UAV^T`，其中 `U ∈ St(m, r)` 和 `V ∈ St(n, r)` 是正交矩阵（Stiefel 流形约束）。
   - 通过施加**正交性约束**，防止方向多样性坍缩（directional diversity collapse），保留多方向特征几何结构，提升特征提取器的表达能力。
   - 使用 **Landing Field** 方法进行高效的流形优化，避免昂贵的 SVD 或 QR 分解操作，实现 3× 到 18× 的训练加速。

2. **Variational Bayesian Last Layer (VBLL)**  
   - 采用 **Bayesian Last Layer (BLL)** 范式：LLM 主干作为确定性特征提取器，仅最后一层权重为随机变量。
   - 在训练阶段，通过最大化一个**闭式的、Jensen 加紧的 ELBO**，联合优化 PoLAR 参数和最后一层权重的近似后验分布（变分参数）。
   - 推理时，只需一次主干前向传播，随后对轻量级的最后一层进行多次采样，**极大提升推理效率**。

3. **Hybrid Refinement with Post-hoc Laplace (LA)**  
   - 可选地，在 VBLL 找到高质量后验模式后，应用 LA 对最后一层的协方差进行精细化校准，利用精确的 Hessian 信息进一步提升 UQ 性能。

### 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **UQ 质量** | 显著优于 LoRA-based LA/BLoB，校准误差（ECE）更低，负对数似然（NLL）更优，尤其在 OOD 场景下表现突出。 |
| **计算效率** | 推理速度比 BLoB 快约 **7–10 倍**（见 Table 3），因无需多次主干前向传播。 |
| **特征表达** | PoLAR 有效缓解 rank collapse，稳定秩（stable rank）从 LoRA 的 ~1.5 提升至 ~2.9，保留更多特征方向。 |
| **训练稳定性** | VBLL 的闭式 ELBO 避免了 MC 采样的高方差梯度，训练更稳定。 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **In-Distribution (ID) 数据集**（常识推理任务）：
  - Winogrande-Small/Medium (WG-S/WG-M)
  - ARC-Challenge/Easy (ARC-C/ARC-E)
  - OpenBookQA (OBQA)
  - BoolQ
- **Out-of-Distribution (OOD) 数据集**（评估泛化与 OOD 检测）：
  - OBQA → ARC-C / ARC-E （较小分布偏移）
  - OBQA → Chemistry / Physics （来自 MMLU，较大领域偏移）

### 实验设置和评估指标

- **模型**：LLaMA-3.1-8B 和 LLaMA-2-7B
- **PEFT 设置**：在输出层及所有注意力层的 query 和 value 上应用 PoLAR/LoRA，秩 `r = 8`
- **训练步数**：5000 步，batch size = 4
- **评估指标**：
  - **ACC (%)**：准确率
  - **ECE (%)**：Expected Calibration Error（越低越好）
  - **NLL**：Negative Log-Likelihood（越低越好）

### 基线方法对比

| 类别 | 方法 |
|------|------|
| **标准 PEFT** | MLE, MAP |
| **UQ on LoRA** | MCD, ENS, LA |
| **近期贝叶斯 LoRA** | BLoB, ScalaBL, C-LoRA, TFB |
| **PoLAR 化变体** | PoLAR-MLE, PoLAR-LA, PoLAR-BLoB 等 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（LLaMA-3.1-8B，部分摘录自 Table 1）

| 方法 | WG-S ACC | OBQA ACC | OBQA ECE | OBQA NLL |
|------|----------|----------|---------|---------|
| **MLE** | 77.92 | 88.30 | 8.95 | 0.74 |
| **LA** | 77.92 | 88.30 | 4.51 | 0.61 |
| **BLoB** | 72.36 | 87.53 | 3.77 | 0.58 |
| **PoLAR-BLoB** | 76.49 | 87.67 | 3.71 | 0.53 |
| **PoLAR-VBLL (w/o LA)** | **77.26** | **88.43** | **3.71** | **0.53** |
| **PoLAR-VBLL** | **77.26** | **88.43** | **3.71** | **0.53** |

> ✅ **PoLAR-VBLL 在多个 ID 数据集上达到最佳或次佳 ACC，同时 ECE 和 NLL 显著优于其他方法**。

#### OOD 性能（OBQA → Chem）

| 方法 | ACC | ECE | NLL |
|------|-----|-----|-----|
| **MLE** | 28.73 | 1.75 | 1.94 |
| **LA** | 11.69 | 1.19 | 1.22 |
| **PoLAR-VBLL** | **49.30** | **1.16** | **1.20** |

> ✅ **PoLAR-VBLL 在 OOD 任务上不仅保持更高 ACC，且校准质量更优**。

### 消融实验结果

#### (1) **PoLAR vs. LoRA 的稳定秩分析**（Table 2, Fig 2）
- **LoRA**：平均稳定秩 ≈ 1.53（接近 1，严重坍缩）
- **PoLAR**：平均稳定秩 ≈ 2.86（显著更高，保留更多方向）

#### (2) **VBLL 是校准性能的主要驱动力**（Table 6）
- 固定 PoLAR 适配器，比较不同 UQ 方法：
  - **PoLAR-LA-LL**：ECE = 14.63%
  - **PoLAR-BLoB**：ECE = 12.06%
  - **PoLAR-VBLL (w/o LA)**：ECE = **8.26%**
  - **PoLAR-VBLL (full)**：ECE = **7.31%**

> 🔍 **VBLL 本身已提供强校准能力，LA 仅为“锦上添花”**。

#### (3) **Jensen Bound 的紧致性验证**（Table 7）
- Jensen 闭式 ELBO 与 50 样本 MC 估计的绝对差距在 50 步后迅速收敛至 < 0.35，并在整个训练过程中保持稳定。
- 表明闭式优化是高效且可靠的。

#### (4) **推理效率对比**（Table 3）
| 方法 | 推理时间 (s) | 内存 (MB) |
|------|--------------|----------|
| **BLoB** | 80–90 | ~19,800 |
| **PoLAR-VBLL** | **12** | 18,423 |
> ⚡ **PoLAR-VBLL 推理速度快约 7 倍**。

---

## 4. 关键结论和发现

### 主要发现

1. **PoLAR 有效缓解了 LoRA 的 rank collapse 问题**，通过正交约束保留了多方向特征几何，为高质量 UQ 提供了基础。
2. **VBLL 是实现良好校准的核心机制**，其闭式 ELBO 优化稳定，推理高效。
3. **PoLAR-VBLL 实现了 ACC 与 UQ 的双赢**，未出现传统方法中的“准确率-校准权衡”。
4. **后处理 LA 可进一步提升性能**，但前提是 VBLL 已找到高质量后验模式，而非依赖 LA 修复糟糕的初始化。
5. **该框架具有良好的可扩展性和实用性**，适合资源受限场景下的部署。

### 方法的局限性

- 当前仅关注分类任务，对生成任务的 UQ 尚未探索。
- PoLAR 的正交优化可能增加训练复杂性，对超参（如惩罚系数 `λ`）有一定敏感性（尽管实验显示鲁棒）。
- 未在更大规模模型（如 LLaMA-3 70B）上验证。

### 未来工作方向

- 将 PoLAR-VBLL 扩展到文本生成、对话系统等任务。
- 探索更高效的变分推理策略，如结构化协方差假设。
- 结合其他 UQ 技术，如 deep kernel learning 或 conformal prediction。
- 研究如何自动选择最优的秩 `r` 和先验尺度 `σ₀`。

--- 

> **总结**：PoLAR-VBLL 成功地将**架构增强优化**（PoLAR）与**可扩展贝叶斯推理**（VBLL）相结合，为 LLM 提供了一种高效、可靠且实用的不确定性量化方案，兼具高性能与高可信度，为安全关键应用铺平了道路。

</details>

---

### 6. [TinyNina: A Resource-Efficient Edge-AI Framework for Sustainable Air Quality Monitoring via Intra-Image Satellite Super-Resolution](https://arxiv.org/abs/2604.04445)

**Authors**: Prasanjit Dey, Zachary Yahn, Bianca Schoen-Phelan, Soumyabrata Dev  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.04445v1  

#### Abstract
Nitrogen dioxide (NO$_2$) is a primary atmospheric pollutant and a significant contributor to respiratory morbidity and urban climate-related challenges. While satellite platforms like Sentinel-2 provide global coverage, their native spatial resolution often limits the precision required, fine-grain...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TinyNina: A Resource-Efficient Edge-AI Framework for Sustainable Air Quality Monitoring via Intra-Image Satellite Super-Resolution

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于卫星的 **NO₂**（Nitrogen Dioxide）浓度监测面临以下挑战：
- 卫星影像（如 Sentinel-2）空间分辨率有限（10–60 米），难以支持细粒度污染分析；
- 高分辨率参考数据稀缺且昂贵，限制了超分辨率（Super-Resolution, SR）模型的训练与部署；
- 现有深度学习模型参数量大、计算开销高，不适合在边缘设备（Edge Devices）上实时运行；
- 通用图像质量指标（如 PSNR、SSIM）与下游任务（如 NO₂ 预测）性能脱节。

### 🚀 提出的新方法与创新
作者提出 **TinyNina** —— 一种面向环境监测任务的轻量化、资源高效的 **Edge-AI 超分辨率框架**，其核心创新包括：

#### （1）**任务感知的轻量级架构设计**
- 模型仅含 **51K 参数**，远小于主流模型（如 EDSR: 40.7M，RCAN: 15.4M）；
- 引入 **depthwise separable convolutions** 显著降低计算复杂度；
- 采用 **multi-scale residual upsampling** 和 **PixelShuffle** 实现高效上采样。

#### （2）**Intra-Image Spectral Super-Resolution 学习范式**
- 利用 Sentinel-2 自身多光谱层级结构进行自监督训练：
  - 使用 10m 分辨率波段（如 B4）作为低分辨率波段（如 B5–B7）的空间引导信号；
  - 完全无需外部高分辨率标签数据（external HR references），提升可扩展性和适用性。

#### （3）**波长敏感的注意力机制（Spectral Attention Gates）**
- 动态加权对 NO₂ 敏感的关键波段（如红边波段 B4–B7，700–800nm）；
- 保留污染物相关的光谱特征，避免信息失真。

#### （4）端到端 NO₂ 预测流水线
- 将超分辨率模块与基于 **ResNet50** 的回归模型结合，形成完整预测流程；
- 支持从原始卫星图像直接输出地面 NO₂ 浓度估计。

### 🔍 相比现有方法的优势
| 维度 | TinyNina | 传统方法（如 EDSR/RCAN） |
|------|---------|------------------------|
| **参数量** | 51K（极小） | 数千万级别 |
| **是否依赖外部数据** | 否（intra-image learning） | 是（需 RapidEye 等配准数据） |
| **推理速度** | 快 **47×**（vs EDSR） | 慢，不适用于边缘部署 |
| **任务相关性能** | 优化于 NO₂ 预测 | 优化于 PSNR/SSIM，与实际应用脱节 |
| **部署场景** | 可部署于 **IoT 边缘网关、车载系统** | 仅适合云端批量处理 |

---

## 2. 核心实验方法和设置

### 📁 数据集
- 使用由 Scheibenreif et al. [22] 构建的数据集：
  - 包含 **27 个 EPA 地面监测站**（美国西海岸：加州、俄勒冈州、华盛顿州）；
  - 时间跨度：**2018–2020 年**，涵盖四季变化；
  - 总样本数：**3,276 对** 卫星-地面匹配观测；
  - 卫星数据：Sentinel-2 Level-2A 多光谱影像（12 个波段，排除 B10）；
    - 分辨率层次：10m（可见光/NIR）、20m（红边/SWIR）、60m（大气校正波段）；
  - 地面真值：小时级 NO₂ 浓度，与卫星过境时间精确对齐。

### ⚙️ 实验设置
- 输入格式：每个样本为 **200×200 像素的 12 波段图像块**（约 1.2×1.2 km²）；
- 预处理步骤：
  - 云掩膜（SCL）
  - 大气校正（SEN2COR）
  - 几何配准至 WGS84（误差 < 0.5 像素）
  - 所有波段双三次插值至 10m 分辨率作为输入基准；
- 超分辨率目标：恢复 20m 波段（B5–B7, B8A, B11–B12）的高频细节。

### 📊 评估指标
- 主要关注 **下游任务性能**，而非图像重建质量：
  - **MAE**（Mean Absolute Error）：预测 NO₂ 与实测值之间的平均绝对误差；
  - **MSE**（Mean Squared Error）：均方误差；
- 不使用 PSNR/SSIM，因其与 NO₂ 预测性能相关性弱。

### 🆚 基线方法对比
| 模型 | 类型 | 参数量 | 是否使用外部数据 |
|------|------|--------|------------------|
| EDSR | CNN-based SR | 40.7M | 是 |
| RCAN | Channel Attention SR | 15.4M | 是 |
| NinaB1 | Hybrid SR | 1.02M | 是 |
| **TinyNina (Ours)** | Task-aware Lightweight SR | **51K** | **否** |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
| 方法 | MSE (μg/m³²) | MAE (μg/m³) |
|------|--------------|------------|
| EDSR (Naive SR) | 112 | 8.2 |
| RCAN (Naive SR) | 98 | 7.8 |
| **TinyNina (Channel SR)** | **97** | **7.4** |

👉 **TinyNina 在 MAE 上达到 7.4 μg/m³，优于所有基线模型**，满足 EPA 监测精度要求（典型城市 NO₂ 浓度为 50–100 μg/m³，误差 <15%）。

### 🔁 推理效率对比
- **推理速度快 47× vs EDSR，28× vs RCAN**；
- 在 Intel Core i7 CPU 上处理 500 张图像耗时仅 **45 秒**（≈90ms/图）；
- 内存占用极低（模型大小 ~0.2MB）。

### 📉 收敛速度优势
- NO₂ 预测模型在 TinyNina 增强图像上训练时，收敛速度快 **40–50 个 epoch**；
- 表明 TinyNina 生成的图像更利于捕捉与 NO₂ 相关的空间-光谱特征。

### 🔬 消融实验结果（Ablation Study）
| 变体 | Attention | MSE | MAE |
|------|-----------|-----|-----|
| TinyNina（无 attention） | ❌ | 102 | 7.9 |
| **TinyNina（完整版）** | ✅ | **97** | **7.4** |

✅ 结果表明：**spectral attention 显著提升性能**（MAE ↓0.5），且几乎不增加模型负担。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **任务驱动的设计优于通用超分辨率**：
   - TinyNina 牺牲通用图像质量，专注保留 NO₂ 敏感波段的光谱一致性，在下游任务中表现更优；
   - 验证了“Green AI”理念：小而精 > 大而慢。

2. **intra-image learning 是解决数据稀缺的有效路径**：
   - 无需外部高分辨率标签即可实现有效训练，极大提升了模型在全球范围内的可部署性。

3. **Edge-AI 可支撑可持续环境监测**：
   - 极低延迟与能耗使其适用于智能交通、智慧城市等实时系统；
   - 支持部署于 **Jetson Nano/Xavier NX** 等边缘平台（预计吞吐达 4–12 tiles/s）。

4. **能效与碳足迹显著降低**：
   - 单次推理能耗约 **5.85 Joules**；
   - 百万级图像处理总能耗仅约 **1.6 kWh**，相比传统模型节能两个数量级以上。

### ⚠️ 局限性
- **受制于卫星重访周期**：无法捕捉突发污染事件（如交通高峰、野火烟雾）；
- **气象因素未显式建模**：风速、逆温层等影响扩散过程的因素未被纳入；
- **云污染残留风险**：尽管进行了云掩膜，残余大气效应仍可能干扰光谱重建；
- **域迁移问题**：在气候或城市结构差异大的区域可能性能下降。

### 🔮 未来工作方向
- 引入 **multi-temporal 观测** 以缓解时间错配问题；
- 融合 **meteorological data**（温度、风速）增强预测鲁棒性；
- 扩展至其他污染物（如 PM2.5、CO）或多任务联合建模；
- 探索 **hybrid edge-cloud 架构**，实现大规模动态空气质量地图生成；
- 应用于 **eco-routing、排放区监管、ITS 控制中心** 等智能交通场景。

---

> 💡 **一句话总结**：  
> **TinyNina 通过“任务感知 + 光谱优化 + 极致轻量化”的设计，在无需外部数据的前提下实现了高效、精准、可持续的卫星 NO₂ 监测，为 Green AI 与 Edge-AI 在环境科学中的落地提供了典范。**

</details>

---

### 7. [BlazeFL: Fast and Deterministic Federated Learning Simulation](https://arxiv.org/abs/2604.03606)

**Authors**: Kitsuya Azuma, Takayuki Nishio  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.03606v1  

#### Abstract
Federated learning (FL) research increasingly relies on single-node simulations with hundreds or thousands of virtual clients, making both efficiency and reproducibility essential. Yet parallel client training often introduces nondeterminism through shared random state and scheduling variability, fo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：BlazeFL: Fast and Deterministic Federated Learning Simulation**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
Federated Learning (FL) 研究高度依赖单节点模拟器来运行数百甚至数千个虚拟客户端。然而，当前的 FL 框架在并行执行时面临两大挑战：
- **效率瓶颈**：由于进程间通信（IPC）和参数序列化的开销，尤其是在通信密集型任务中，模拟速度受限。
- **可复现性差**：并行训练引入了非确定性行为，如共享随机状态（RNG）、调度顺序变化以及浮点累加顺序不一致，导致即使使用相同种子也无法获得比特级一致的结果。

这些问题迫使研究人员在吞吐量和可复现性之间做出权衡。

---

### **提出了什么新方法或新思路**
作者提出 **BlazeFL** —— 一个轻量级、面向单节点 FL 模拟的新框架，其核心设计基于以下两个关键技术：

#### ✅ **Free-threaded Shared-Memory Execution**
- 利用 Python 的 free-threading 构建（PEP 703 / PEP 779），在单个进程中以线程方式运行客户端。
- 所有线程共享内存空间，服务器与客户端之间的模型参数通过共享内存直接传递，避免了跨进程序列化和 IPC 开销。

#### ✅ **Controlled Deterministic Execution**
- 为每个客户端分配独立的 RNG 流（isolated RNG streams），确保随机操作与调度解耦。
- 客户端结果按采样顺序消费（而非完成顺序），消除因浮点累加顺序不同带来的数值差异。

---

### **相比现有方法的优势**
| 维度 | 传统框架（如 Flower + Ray） | BlazeFL |
|------|-------------------------------|---------|
| 并行模型 | 多进程 / 分布式运行时（multiprocessing/distributed runtime） | 单进程多线程（free-threaded） |
| 参数交换 | 需要序列化 + IPC 或对象存储 | 共享内存直传，零序列化 |
| 可复现性 | 即使设全局种子仍存在偏差 | 支持比特级重复（bitwise-identical） |
| 依赖复杂度 | 依赖 Ray/MPI/NCCL 等重型组件 | 仅需标准库 + PyTorch，依赖极简 |
| 易用性 | 需继承特定类或适配框架生命周期 | 基于 `typing.Protocol` 接口，兼容任意 PyTorch 训练逻辑 |

> 🔍 **核心优势总结**：BlazeFL 在保持高性能的同时实现了**高并发下的确定性执行**，解决了“速度快就不能复现”的经典矛盾。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **CIFAR-10** 图像分类任务
- 数据划分：non-IID 设置，每客户端仅包含两类图像（2 classes per client）
- 总共 100 个虚拟客户端

---

### **实验设置**
- **算法**：FedAvg [7] 聚合策略
- **训练配置**：
  - 每轮选择部分客户端参与训练
  - 每个选中客户端本地训练 5 epochs，使用 500 个样本
  - 每轮后进行全局评估（10,000 测试样本）
  - 总共运行 5 communication rounds
- **并行度变量**：$ P \in \{1, 2, 4, 8, 16, 32, 64\} $
- **模型规模**：从轻量 CNN 到深层 ResNet（ResNet-18, ResNet-50, ResNet-101）

---

### **评估指标**
1. **Wall-clock time**：五轮 FL 循环的总耗时（不含数据下载与划分）
2. **Throughput**：随并行度提升的时间下降趋势
3. **Reproducibility**：
   - 最终准确率的标准差（Final Accuracy Std. Dev.）
   - 每轮全局模型权重的 SHA-256 哈希是否一致（round-wise hash agreement）

---

### **基线方法对比**
| 配置 | 描述 |
|------|------|
| **BlazeFL (free-threaded)** | 主推方案：单进程 free-threading + 共享内存 |
| **BlazeFL (process-based)** | 对照组：多进程 + `torch.multiprocessing` + 共享内存张量 |
| **Flower (Ray backend)** | 当前主流开源框架代表，作为主要性能与可复现性基准 |

> ⚠️ 注意：由于依赖栈限制，BlazeFL 使用 Python 3.14.3（支持 free-threading），而 Flower 使用 Python 3.13.7，因此比较是端到端的实际部署对比，而非理想化微基准。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### 📈 **性能加速（High-performance Server）**
在配备 NVIDIA H100 GPU 和 48 核 CPU 的服务器上：

| 模型 | 最大加速比（vs. Flower） |
|------|--------------------------|
| Lightweight CNN | **3.1× 更快** |
| ResNet-18 | **1.4× 更快** |
| ResNet-50 | **1.1× 更快** |

- 加速效果在通信密集型任务中最显著（小模型 + 高并行度）
- BlazeFL 的 free-threaded 模式随着 $P$ 增加持续受益，并未出现性能平台期；而 Flower 在高并发下趋于饱和甚至退化。

#### 💻 **工作站级设备表现**
在 NVIDIA Quadro RTX 6000（24GB VRAM）上：
- 小模型（CNN）仍明显优于 Flower
- 大模型（ResNet-50/101）上 Flower 略优 → 推测原因是 **CUDA caching allocator 的全局锁争用**成为瓶颈
- 此场景下建议切换至 **BlazeFL 的 process-based 模式**以规避单进程内核锁竞争

> 🔍 结论：**BlazeFL 的优势集中在通信主导型负载**，计算密集且显存紧张时需权衡模式选择。

---

### **与基线方法的对比结果**

| 指标 | Flower (no seed) | Flower (global seed) | BlazeFL (both modes) |
|------|------------------|------------------------|------------------------|
| 最终准确率标准差 | 1.24 pp | 0.18 pp | **0.00 pp** |
| 每轮模型哈希一致性 | ❌ 不一致 | ❌ 不一致 | ✅ 完全一致 |

> ✅ **BlazeFL 实现了比特级可复现性**，无论线程还是进程模式，在固定软硬件环境下均能产生完全相同的训练轨迹。

---

### **消融实验结果**

#### ✅ **跨并行度的可复现性测试（Ablation on Parallelism Level）**
- 固定种子、分区、软件栈，改变 $P = 1, 2, ..., 64$
- 所有情况下：
  - 与 $P=1$ 的参考运行相比，**最终准确率无任何变化（ΔAcc = 0.0 pp）**
  - 所有五轮的全局模型 SHA-256 哈希完全匹配

> 📌 发现：**BlazeFL 的执行结果对并行度变化具有不变性**，这是现有框架难以实现的关键特性。

#### ✅ **Flower 中的误差传播分析**
- 可视化某客户端在各轮输出 logits 与平均值的 L2 距离
- 第一轮完全一致 → 第二轮聚合后出现 ~1e-6 差异 → 后续轮次呈“扇形发散”
- 原因：**completion-order-dependent aggregation** 导致浮点累加顺序不同，引发舍入误差累积

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **共享内存 + free-threading 可显著降低通信开销**，尤其适用于通信密集型 FL 模拟。
2. ✅ **隔离 RNG 流 + 固定消费顺序** 是实现高并发下比特级可复现性的有效路径。
3. ✅ BlazeFL 在合理设置下，能在不同并行度下生成完全一致的结果，极大提升了实验可信度。
4. ✅ 相比 Flower + Ray，BlazeFL 实现最高 **3.1× 的端到端加速**，同时保持更小的依赖面和更高的可移植性。

---

### **方法的局限性**
| 局限性 | 说明 |
|-------|------|
| **单节点限制** | 不适用于多机分布式训练，目标定位为本地原型开发与调试 |
| **跨平台不可复现** | 仅保证同一软硬件环境下的比特级一致，跨机器/操作系统/驱动版本可能失效 |
| **vision pipeline 兼容性要求** | 若数据增强算子内部使用全局 RNG（如某些 torchvision transforms），则需手动适配为显式 generator 输入才能保证端到端确定性 |
| **生态系统成熟度** | 依赖 Python free-threading（3.14+），目前生态尚未普及，部分第三方库暂不支持 |

---

### **未来工作方向**
1. **扩展至多节点模拟环境**，探索如何将确定性控制延伸至网络通信层。
2. **集成自动检测机制**，识别用户代码中潜在的全局 RNG 使用，提示修改建议。
3. **优化 CUDA allocator 锁竞争问题**，例如引入 per-thread memory pool 或异步释放策略。
4. **推动社区采纳 free-threading 构建**，促进整个 ML 生态向更高并发、更低开销演进。

---

> ✅ **总体评价**：  
> **BlazeFL 是一个面向 FL 系统研究者的实用工具**，特别适合需要快速迭代、精细调参、严格对比算法变体的研究场景。它不仅提升了效率，更重要的是重建了“一次运行即可信赖”的科研基础，有望成为可复现 FL 研究的新标准基础设施之一。

🔗 开源地址：[https://github.com/kitsuyaazuma/blazefl](https://github.com/kitsuyaazuma/blazefl)

</details>

---

### 8. [Sampling Parallelism for Fast and Efficient Bayesian Learning](https://arxiv.org/abs/2604.04736)

**Authors**: Asena Karolin \"Ozdemir, Lars H. Heyen, Arvid Weyrauch, Achim Streit, Markus G\"otz, Charlotte Debus  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.04736v1  

#### Abstract
Machine learning models, and deep neural networks in particular, are increasingly deployed in risk-sensitive domains such as healthcare, environmental forecasting, and finance, where reliable quantification of predictive uncertainty is essential. However, many uncertainty quantification (UQ) methods...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Sampling Parallelism for Fast and Efficient Bayesian Learning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **采样型贝叶斯学习（如 BNN、MCD）计算开销大**：由于需要多次采样模型参数并进行前向传播，导致训练过程内存占用高、计算时间长。
- **现有并行化策略（如 DDP）难以有效支持大规模贝叶斯学习**：
  - DDP 要求每个 GPU 存储完整模型副本，当模型本身已接近显存极限时，多份采样无法容纳。
  - 大批量训练会引发“large batch effects”，损害泛化能力。

### 🚀 提出的新方法：**Sampling Parallelism**
- 将 **采样维度（sample dimension）作为新的并行轴**，将 $ S $ 个参数样本分布到 $ P $ 个 GPU 上，每个 GPU 只负责 $ S/P $ 个样本的前向与反向传播。
- 所有 GPU 共享相同的输入 batch，通过 `all-reduce` 同步梯度以更新共享的变分参数（mean 和 std）。

### 🔍 创新点
1. **提出“采样并行”这一正交于 DDP/MP 的新并行范式**：
   - 不改变模型架构或引入复杂通信协议。
   - 显著降低单卡显存压力（最多减少 $ P $ 倍）。
2. **利用重复数据加载实现增强多样性（augmentation diversity）**：
   - 每个 GPU 对同一 batch 应用独立的随机数据增强（如 crop、flip），相当于在不增加 batch 数量的情况下提升了数据多样性。
3. **提出混合并行策略（Hybrid Parallelism）**：
   - 在节点内使用 **Sampling Parallelism** 分配样本，在节点间使用 **DDP** 分配数据，兼顾可扩展性与效率。

### ⚖️ 相比现有方法的优势
| 方法 | 内存优势 | 训练速度 | 收敛速度 | 适用场景 |
|------|----------|----------|-----------|------------|
| **DDP** | ❌ 需复制整个模型 | ✅ 快速（线性加速） | ⚠️ 受 large batch 影响 | 小模型 + 大数据 |
| **Sampling Parallelism** | ✅ 显著降低 per-GPU 显存 | ⚠️ 数据加载重复带来开销 | ✅ 更快收敛（因增强多样性） | 大模型 + 强 UQ 需求 |
| **Hybrid** | ✅ 最佳平衡 | ✅✅ 综合优化 | ✅✅ 加速且稳定 | 超大规模任务 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与任务
| Use Case | 模型 | 数据集 | 任务类型 |
|--------|-------|--------|---------|
| **Use Case 1** | Bayesian ViT | CIFAR-10 | 图像分类 |
| **Use Case 2** | MLP + MCD | ENTSO-E de | 时间序列预测（电力消耗） |
| **Use Case 3** | SWIN Transformer | ERA5 | 天气预报（高维回归） |

> 所有实验均基于 PyTorch 2.9.0 和 `torch.distributed` 实现，并使用 A100 GPUs（40GB）集群运行。

### 🧪 实验设置与评估指标

#### 并行策略对比
- **Sampling Parallelism (SP)**：样本跨 GPU 分布，batch 复制。
- **Distributed Data Parallelism (DDP)**：数据分片，每卡一个 micro-batch。
- **Hybrid**：节点内 SP，节点间 DDP。

#### 缩放实验设计
1. **Proportional-sample Scaling**  
   - 工作负载随 GPU 数量成比例增长（样本数 ∝ GPU 数）
   - 评价指标：**Efficiency** = $ T(1)/[T(P) \cdot P] $
2. **Fixed-sample Scaling**  
   - 固定总样本数（如 16 samples），增加 GPU 数
   - 评价指标：**Speed-up** = $ T(1)/T(P) $

#### 两种 batch 设置
- **Constant Global Batch Size (GBS)**：保持全局 batch 不变 → 单卡负载下降
- **Increasing GBS**：提升 GBS 使每卡满载 → 更公平比较硬件利用率

#### 评估指标
| 指标 | 描述 |
|------|------|
| Top-1 Accuracy | 分类任务性能 |
| RMSE / MSE | 回归任务误差 |
| Wall-clock Time to Target Accuracy | 实际训练耗时 |
| Negative Log Likelihood (NLL) | 不确定性校准质量 |
| Mean Absolute Calibration Error (MACE) | 校准偏差度量 |

---

## 3. 主要实验结果和性能指标

### 📈 Use Case 1: ViT on CIFAR-10（BNN + VI）

#### ✅ Proportional-sample Scaling 结果
- **近乎理想的扩展效率**：
  - 效率维持在 90% 以上（见 Figure 4 下图），表明采样过程高度可并行化。
- **验证了“采样即并行”的可行性**。

#### ⏱ Fixed-sample Scaling（16 samples, 16 GPUs）
| 方法 | Speed-up | Epoch Time | Epochs to Convergence | Wall-clock Time |
|------|----------|------------|------------------------|------------------|
| DDP (incr GBS) | ~14x | 极低 | 较多 | 中等 |
| SP (incr GBS) | ~8–10x | 较高（重复加载） | **显著更少** | **相当甚至略优** |
| Hybrid | 居中 | —— | —— | —— |

> 尽管 SP 的 epoch time 更长，但由于每轮学习效果更强，最终达到目标精度的时间与 DDP 相当甚至更快。

#### 🎯 收敛与不确定性质量
- **Accuracy vs Epoch**（Figure 5）：
  - SP 收敛速度明显快于 DDP，尤其在高 GBS 下。
- **不同增强策略对比**（Figure 6）：
  - 若所有 GPU 使用相同 augmentation → 收敛慢；
  - 若各 GPU 独立增强 → **收敛加速显著**，证明多样性增益。
- **不确定性校准表现更优**（Table 1 & Figure 8）：
  - SP 的 **MACE 更低**（最低达 0.0534 vs DDP 的 0.1525）
  - NLL 下降更快，说明预测不确定性更可靠。

---

### ⚙️ Use Case 2: MLP on ENTSO-E de（MCD）

#### ⚠️ 小模型下的局限性
- **Proportional-sample Scaling 效率下降明显**（Figure 9）：
  - 超过 4 GPUs 后效率急剧下滑。
- 原因：小模型计算量小，**通信开销占主导**（gradient all-reduce 成为瓶颈）。
- **DDP 表现优于 SP**：因其能并行化数据加载，而 SP 重复加载反而浪费资源。

> ➤ 结论：**Sampling Parallelism 更适合大模型**，其中计算成本足以覆盖通信开销。

---

### 🌐 Use Case 3: SWIN Transformer on ERA5（大型天气建模）

#### 💥 突破传统限制的关键案例
- 模型参数 >200M，单个数据样本 ≈17.7MB，**极难放入单卡显存**。
- 在非分布式设置下，仅能运行 2 个样本；使用 SP 可扩展至 16 样本（8 GPUs × 2 samples/GPU）。
- **Proportional-sample Scaling 效率达 80–90%**（Figure 10），显示良好可扩展性。

> ➤ 这是 **DDP 完全不可行的场景**：因为即使 batch=1，也无法容纳多个样本副本。唯有 SP 能突破此壁垒。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Sampling Parallelism 是一种高效、实用的新并行范式**：
   - 显著缓解贝叶斯学习中的显存瓶颈。
   - 实现近似完美的 proportional scaling。
2. **重复数据加载不是缺陷，而是机会**：
   - 允许各 GPU 施加独立数据增强，提升训练多样性，加快收敛。
   - 在图像等任务中形成“免费的数据增强并行化”。
3. **与 DDP 正交，可组合为 Hybrid 方案**：
   - 实现跨节点数据并行 + 节点内采样并行，适用于超大规模部署。
4. **对大模型特别有价值**：
   - 当模型或数据项极大时（如气象、医疗影像），SP 成为唯一可行路径。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **不适合小型网络** | 通信开销占比过高，效率低于 DDP（如 MLP 实验所示） |
| **非精确算法（Approximate）** | 梯度平均可能导致 ELBO 中非线性项（如标准差聚合）失真 |
| **依赖 loss 函数结构** | 某些 loss（如基于预测均值的标准差）需额外通信才能精确同步 |
| **数据加载冗余** | 每卡重复读取相同 batch，带宽利用率较低 |

---

### 🔮 未来工作方向
1. **开发专用库支持 Sampling Parallelism**
   - 提供经过验证的、保证数值一致性的 loss 实现（如支持跨 GPU 统计量同步）。
2. **探索更高效的通信机制**
   - 如 fused communication kernels 或异步梯度同步，降低开销。
3. **与其他并行技术深度集成**
   - 与 FSDP、Tensor Parallelism 结合，构建完整的“贝叶斯训练栈”。
4. **理论分析 approximation error bounds**
   - 量化梯度平均带来的偏差，指导实际应用中的可靠性判断。

---

## 总结

> **Sampling Parallelism** 不仅是一种工程优化，更是打开了 **贝叶斯深度学习规模化的大门**。它通过将“采样”本身并行化，解决了长期以来阻碍 BNN/MCD 实际落地的核心瓶颈——显存与计算成本。虽然在小模型上不如 DDP 高效，但在大模型尤其是数据密集型任务（如气候建模）中展现出不可替代的价值。更重要的是，其带来的 **增强多样性红利** 使得该方法不仅是“更省”，而且是“更好”。随着硬件规模持续扩大，Sampling Parallelism 有望成为下一代 UQ 系统的标准组件之一。

</details>

---

### 9. [Bypassing the CSI Bottleneck: MARL-Driven Spatial Control for Reflector Arrays](https://arxiv.org/abs/2604.05162)

**Authors**: Hieu Le, Oguz Bedir, Mostafa Ibrahim, Jian Tao, Sabit Ekin  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.05162v1  

#### Abstract
Reconfigurable Intelligent Surfaces (RIS) are pivotal for next-generation smart radio environments, yet their practical deployment is severely bottlenecked by the intractable computational overhead of Channel State Information (CSI) estimation. To bypass this fundamental physical-layer barrier, we p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Bypassing the CSI Bottleneck: MARL-Driven Spatial Control for Reflector Arrays*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Reconfigurable Intelligent Surfaces (RIS)** 在实际部署中面临严重的 **Channel State Information (CSI) estimation bottleneck**。精确的 CSI 获取需要大量导频开销和计算资源，尤其在毫米波 (mmWave) 和非直视 (NLOS) 场景下，导致系统复杂度高、难以扩展。

此外，大多数基于 **Deep Reinforcement Learning (DRL)** 的 RIS 控制方法仍依赖部分 CSI 或需离线训练数据，限制了其在动态环境中的实用性。

### 🚀 提出的新方法与创新思路
本文提出一种 **AI-native、data-driven 的新型范式**，通过以下核心设计绕过 CSI 瓶颈：

- **Mechanically Adjustable Metallic Reflector Arrays**  
  使用可机械调节方位角 (azimuth) 和俯仰角 (elevation) 的金属反射板阵列，替代传统的电子相位调控 RIS，实现对电磁波前的物理操控，无需复杂的 RF 电路或相移器。

- **Focal Point Control Abstraction**  
  将高维控制空间（每个 tile 的独立角度）抽象为低维的 **虚拟焦点 (virtual focal point)** 空间。每个智能体控制一个分段反射器的焦点位置，从而几何地决定该段内所有 tiles 的反射方向，显著降低优化维度。

- **Multi-Agent Reinforcement Learning (MARL) 框架 + CTDE 架构**  
  采用 **Centralized Training with Decentralized Execution (CTDE)** 范式的 **Multi-Agent PPO (MAPPO)** 算法：
  - **训练阶段**：集中式 critic 利用全局状态进行稳定学习；
  - **执行阶段**：各 agent 仅凭本地观测（用户坐标、自身段落位置）独立决策，无需通信或中心控制器。

- **CSI-Free Operation**  
  完全依赖 **用户坐标信息**（可通过 UWB 等现代定位技术获取），不依赖任何信道估计，实现真正的 **spatial intelligence** 驱动无线传播控制。

### 🔍 相比现有方法的优势
| 方面 | 本文方法 | 传统方法 |
|------|---------|----------|
| **CSI 依赖** | 完全无依赖（CSI-free） | 强依赖，带来巨大开销 |
| **硬件成本** | 商用伺服电机即可，无需高精度相移器 | 需要昂贵的 RF 链路和同步机制 |
| **可扩展性** | 多智能体分解任务，天然支持大规模阵列 | 单智能体难以处理高维动作空间 |
| **鲁棒性** | 对定位噪声具有强韧性（至 1.0m） | 性能随 CSI 不准确迅速下降 |

---

## 2. 核心实验方法和设置

### 📊 数据集与仿真环境
- **未使用真实数据集**，而是构建了一个高保真 (**high-fidelity**) 的 **ray-tracing 仿真环境**。
- 使用 **NVIDIA Sionna** 进行 60 GHz mmWave 下行链路建模，考虑多径效应和材料属性（石膏板、混凝土、木材等，依据 ITU 推荐值）。
- 场景为 **L-shaped hallway**，存在严重 NLOS 阻塞。

### ⚙️ 实验设置
- **AP 发射功率**：5 mW
- **用户数量**：3 个移动用户
- **反射阵列**：由 $ N_r \times N_c = 72 $ 个六边形金属 tile 组成
- **控制粒度**：阵列划分为 $ L $ 个 segment，每段由一个 agent 控制其虚拟焦点
- **动作空间**：每个 agent 输出三维位移向量 $\Delta f_t \in \mathbb{R}^3$ 更新焦点位置
- **观测空间**：仅包含分配用户的坐标、本段位置、当前焦点
- **奖励函数**：混合奖励  
  $$
  R_k(s_t,a_t) = \frac{1}{K}\sum_{k'} P_{t,k'}(s_t,a) + P_{t,k(l)}(s_t,a)
  $$
  包含全局平均 RSSI 与本地目标用户 RSSI，鼓励协作与专注并存。

### 🧪 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **No Reflector** | 无辅助的纯 NLOS 环境 |
| **Flat Reflector** | 静态平面反射器（固定角度） |
| **beam-focusing-sa** | 单智能体控制整个阵列（Single-Agent DRL） |
| **column-based-ma** | 多智能体但受限于列级 azimuth 控制（降低成本，减少自由度） |

### 📈 评估指标
- **Average Received Signal Strength Indicator (RSSI)**（dBm）
- **Spatial Selectivity**：信号是否聚焦于用户所在区域
- **Temporal Stability**：面对用户移动时的响应速度与波动程度
- **Robustness to Localization Noise**：加入 0.1–1.0 米高斯噪声下的性能退化情况
- **Convergence Speed & Final Cumulative Reward**

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
| 方法 | 平均 RSSI (dBm) | 相比 Flat Reflector 提升 |
|------|------------------|----------------------------|
| No Reflector | -110.50 dBm | — |
| Flat Reflector | -94.40 dBm | 基准 |
| beam-focusing-sa (SA) | -72.48 dBm | +21.92 dB |
| column-based-ma | -74.79 dBm | +19.61 dB |
| **beam-focusing-ma (本文)** | **-67.54 dBm** | **+26.86 dB** ✅ |

> 💡 最大提升达 **26.86 dB**，远超静态反射器。

### 🔁 收敛性表现（图3）
- **beam-focusing-ma**：快速收敛，在约 1000 轮后趋于稳定，最终累积奖励 ~42
- **beam-focusing-sa** 和 **column-based-ma**：分别收敛于 ~24 和 ~27，明显低于多智能体方案
- 表明 **多智能体任务分解有效缓解了“维度灾难”**

### 🕐 时间稳定性与适应能力（图5）
- 在 **300 步动态用户移动测试** 中：
  - **beam-focusing-ma** 平均 RSSI 达 **-66.83 dBm**，波动小，恢复快（通常只需 1 步重新对齐）
  - **column-based-ma**：-73.65 dBm，稳定性尚可但性能较低
  - **beam-focusing-sa**：-79.70 dBm，方差大，难以跟踪动态变化

> ✅ 多智能体架构具备更强的实时适应性和协调能力。

### 🛡️ 抗定位噪声鲁棒性（图6）
引入不同标准差的高斯噪声模拟定位误差：
| 定位噪声 (σ) | 平均 RSSI | 性能退化分析 |
|-------------|-----------|--------------|
| 0 m（理想） | -68.15 dBm | 最优性能 |
| 0.1 m       | -68.27 dBm | 几乎无影响 |
| 0.3 m       | -68.62 dBm | 微弱下降 |
| 0.5 m       | -71.21 dBm | 开始明显退化 |
| **1.0 m**   | **-72.36 dBm** | 仍优于单智能体基线！ |

> ✅ 即使在 **1.0 米定位误差** 下仍保持可用信号质量，验证了系统的 **部署韧性 (deployment resilience)**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MARL + Spatial Abstraction 可有效绕过 CSI 瓶颈**  
   通过将物理控制抽象为虚拟焦点操作，并结合 MAPPO 框架，实现了完全无需 CSI 的智能反射控制。

2. **Decentralized Spatial Task Decomposition 是关键**  
   多智能体架构相比单智能体在收敛速度、信号增益和动态适应性上全面占优，证明了任务分解的有效性。

3. **Mechanical Reflectors 具备实用潜力**  
   尽管是机械系统，但在商用伺服支持下仍能实现毫秒级响应，适用于室内慢时变场景。

4. **系统对现实不确定性高度鲁棒**  
   在高达 1.0 米的定位误差下仍能维持良好覆盖，表明该方法适合真实部署。

### ⚠️ 局限性
- 当前基于 **仿真环境**（Sionna），尚未有硬件原型验证。
- 用户与 agent 之间为 **预分配关系**，缺乏动态负载均衡机制。
- 机械调整速度有限，可能不适用于高速移动场景（如车载通信）。
- 聚焦策略假设用户位置已知，若定位失效则性能骤降。

### 🔮 未来工作方向
1. **构建物理原型系统**，实现实验室或现场验证；
2. **集成商业级 Indoor Positioning Systems (IPS)** 如 UWB，形成闭环系统；
3. 设计 **动态用户-智能体映射机制**，应对用户密集或突发流量；
4. 探索 **hybrid electronic-mechanical architectures** 以兼顾灵活性与响应速度；
5. 扩展至 **上行链路** 和 **多 AP 协同场景**。

---

> ✅ **总体评价**：本文提出了一条极具前景的路径——**以空间智能取代信道感知**，推动 RIS 向真正智能化、可扩展、低成本的方向演进。MARL 驱动的反射阵列控制有望成为下一代 AI-Empowered Wireless Networks 的核心技术之一。

</details>

---

### 10. [Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs](https://arxiv.org/abs/2604.05643)

**Authors**: Hongyuan Yuan, Xinran He, Run Shao, Bolei He, Xianwei Xue, Mengke Chen, Qiutong Pan, Haiwei Wang, Haifeng Li  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.05643v1  

#### Abstract
Extending CoT through RL has been widely used to enhance the reasoning capabilities of LLMs. However, due to the sparsity of reward signals, it can also induce undesirable thinking patterns such as overthinking, i.e., generating redundant intermediate reasoning content. In this work, we argue that a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Reinforcement Learning (RL)** 扩展 **Chain-of-Thought (CoT)** 的推理模型（如 o1、R1）虽然提升了复杂任务的表现，但由于 reward 信号稀疏且延迟，容易引发“**过思考**”（overthinking）现象——即生成大量冗余的中间推理内容，尤其是无效的自我反思（reflection），导致推理成本上升而收益有限。

作者指出，这类冗余主要源于两种低效的反思行为：
- **Indiscriminate Reflection**（无差别反思）：对每个步骤都进行检查，即使该步骤是平凡的。
- **Repetitive Reflection**（重复反思）：反复验证已确认的结论。

### 🚀 提出的新方法与创新思路
提出一种 **图结构化的 CoT 优化框架**（graph-based CoT optimization framework），通过以下方式精准识别并剪枝冗余反思：

1. **将线性 CoT 转换为有向无环图 (DAG)**  
   - 每个推理步骤作为节点，显式建模其语义角色（`progress` 或 `review`）和依赖关系。
   - 利用外部 LLM（如 qwen-turbo）迭代构建图结构。

2. **双层级剪枝策略 (Dual Pruning Strategy)**：
   - **Branch-Level Pruning**：移除子节点数少于阈值 $k=2$ 的 `review` 节点，因其形成窄分支，贡献小。
   - **Depth-Level Pruning**：移除在推理后期出现的 `review` 节点（深度超过最大深度的 90%），通常为事后回溯，不增加新信息。

3. **三阶段训练流程** 实现高效推理能力内化：
   - **SFT**：在剪枝后的简洁轨迹上监督微调，初始化高效推理范式。
   - **DPO**：偏好学习，鼓励选择正确但更简短的路径。
   - **GRPO + length penalty**：联合优化答案正确性和推理效率，进一步压缩冗余。

### 🔍 相比现有方法的优势
| 方法类别 | 代表工作 | 局限性 | 本文优势 |
|--------|---------|-------|----------|
| 长度控制 | TALE, CoT-Valve | 粗粒度限制，可能截断有效推理 | 结构感知剪枝，保留核心逻辑 |
| 冗余检测 | TokenSkip, Think Clearly | 基于 token 注意力或重要性评分，缺乏全局依赖理解 | 显式建模依赖关系，精准定位冗余反思 |
| RL 优化 | EfficientReasoning | 仅优化长度目标 | 多阶段协同优化，兼顾准确性与效率 |

> ✅ **核心创新**：首次从**反思行为模式**出发，结合**图结构建模**与**结构化剪枝**，系统性减少冗余推理。

---

## 2. 核心实验方法和设置

### 📚 数据集
在五个主流数学推理基准上评估，覆盖不同难度级别：
- **AIME24**, **AIME25**：美国邀请数学竞赛题，高难度。
- **AMC23**：美国数学竞赛题，中等挑战。
- **MATH500**：精选 500 道涵盖代数、数论、几何、概率的题目。
- **OlympiadBench**：双语奥赛级科学推理题，测试高级推理能力。

每道题采样 10 个解法，报告平均 accuracy 和平均 reasoning token 数量。

### ⚙️ 实验设置
- **模型基础**：基于 `DeepSeek-R1-Distill-Qwen-1.5B` 和 `7B` 进行 post-training。
- **训练流程**：
  1. **SFT**：在 Light-R1 数据集上进行初始微调。
  2. **DPO**：使用自动生成的偏好对（低冗余 vs 高冗余但正确的轨迹）进行偏好对齐。
  3. **GRPO with length penalty**：在 dapo-17k 上进行强化学习，奖励函数设计为：
     $$
     R(x,y) = V(x,y) - \lambda \cdot \mathbb{1}[V(x,y)=1] \cdot R_{\text{length}}(x,y)
     $$
     其中 $R_{\text{length}}$ 是归一化超长惩罚项。

- **硬件**：单台机器，配备 4×NVIDIA A800 GPU。

### 📊 评估指标
- **Accuracy ↑**：最终答案正确率（经 math-verify 验证）
- **Average Reasoning Tokens ↓**：推理部分平均 token 数量，衡量效率
- **Accuracy-Length Trade-off**：综合评价准确率与推理开销的平衡

### 🆚 基线方法对比
| 类型 | 基线方法 |
|------|--------|
| 开源 R1 类模型 | Skywork-OR1-7B, OREAL-7B, AReaL-boba-RL-7B, Light-R1-DS-7B |
| 效率优化方法 | O1-Pruner*, TokenSkip*, EfficientReasoning, AdaptThink |
> *注：O1-Pruner 和 TokenSkip 被适配到 DeepSeek-R1-Distill-Qwen 模型上以公平比较*

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（DeepSeek-R1-Distill-Qwen-7B）

| 方法 | 平均 Accuracy ↑ | 平均 Reasoning Tokens ↓ |
|------|------------------|--------------------------|
| Base (DeepSeek-AI et al., 2025) | 59.72 | 8134 |
| EfficientReasoning | 59.36 | 5515 |
| AdaptThink | 60.15 | 5972 |
| **Ours** | **60.95** | **4660** |

✅ **提升效果显著**：
- **准确率提升**：+1.23 pts（从 59.72 → 60.95）
- **推理长度下降**：**42.7% 减少**（8134 → 4660 tokens）
- 在最难的 **AIME25** 上，accuracy 提升 2.67%，token 数减少近一半（12779 → 6977）

### 🔁 与其他方法对比优势
| 对比维度 | 我们的方法 | 其他方法 |
|--------|-----------|---------|
| 准确率 | 最高 | 多数低于或持平 |
| 推理长度 | 最短 | 普遍较长 |
| trade-off 曲线 | 明显左上方偏移（见 Figure 1） | 更靠近右下角 |

> 图 1 显示，在所有方法中，我们的方法实现了最高的 accuracy 同时拥有最少的 reasoning tokens。

### 🔍 消融实验（Ablation Study）
逐步添加模块的效果（Figure 3）：
| 阶段 | Accuracy | Token Ratio (vs Base) |
|------|--------|------------------------|
| Base | 59.72 | 1.00 |
| +SFT | ~60.0 | ~0.90 |
| +DPO | ~60.5 | ~0.70 |
| +GRPO | **60.95** | **~0.57** |

📌 发现：
- SFT 提供了简洁推理的冷启动；
- DPO 显著降低冗余倾向；
- GRPO 进一步压缩长度并稳定性能。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **冗余反思是推理低效的重要根源**，特别是 Indiscriminate 和 Repetitive Reflection。
2. **图结构建模能有效揭示推理单元间的依赖关系**，从而精准识别可剪枝的低价值 `review` 节点。
3. **结构化剪枝 + 三阶段训练** 可显著压缩推理长度（**平均减少 42%**），同时**保持甚至提升准确率**。
4. 剪枝后仍能维持较高的 **consistency**（表 3 中 90.69% vs Full-CoT 的 99.60%），说明核心逻辑未被破坏。

### ⚠️ 局限性
1. **依赖强教师模型构建图结构**：需调用外部 LLM（如 qwen-turbo）进行图构造，带来预处理开销（约 \$20 成本处理 3335 条样本）。
2. **progress-review 分类较粗粒度**：未能捕捉更细粒度的推理语义差异。
3. **目前主要验证于数学领域**：是否适用于开放域（如常识推理、规划）尚待验证。

### 🔮 未来工作方向
- 设计轻量化图构建机制（如 prompt engineering 或小型专家模型）
- 引入更丰富的语义标签体系（如 hypothesis, contradiction, lemma 等）
- 将方法扩展至代码生成、对话推理等更多场景
- 探索在线动态剪枝（dynamic pruning at inference time）

---

## 总结
> 本文提出了一个新颖的 **graph-based CoT pruning** 框架，通过将线性推理链转化为 DAG 并实施双层级剪枝，成功减少了 LLM 推理中的冗余反思行为。实验表明，该方法可在 **不牺牲准确性的前提下，将平均推理 token 数减少 42%**，显著优于现有效率优化方法，为构建高效、紧凑的 reasoning LLM 提供了一条可行路径。

</details>

---

### 11. [BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs](https://arxiv.org/abs/2604.05942)

**Authors**: Abbas Ghaddar, Ivan Kobyzev, Boxing Chen, Yufei Cui  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.05942v1  

#### Abstract
Post-training hybridization of large language models (LLMs) often replaces quadratic self-attention with sliding-window attention (SWA) to reduce KV cache usage and improve latency. Existing hybridization schemes are typically defined either at the layer level (e.g., interleaving) or at the head lev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs —— 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLMs）中的 **self-attention** 具有 **二次复杂度**，导致在处理长上下文时 **KV Cache 占用高、推理延迟大**。为缓解此问题，研究者采用 **Sliding Window Attention (SWA)** 替代部分全局注意力，实现 **post-training hybridization**。

然而，现有方法存在两大缺陷：
- **Layer-level 方法**（如 interleave、BME）：忽略同一层内不同 attention head 可能分别负责局部与全局依赖，粗粒度替换会破坏关键全局信息。
- **Static head-level 方法**（如 DCAM、FISHER）：基于预训练模型静态分析 head 的“局部性”，但在实际 hybridization 后，head 的行为可能因 **entanglement（纠缠）效应** 发生变化，导致选择失效。

### **提出了什么新方法或新思路**

作者提出 **BOSCH**（Black-Box Binary Optimization for Short-Context Head Selection），一种 **无需训练的黑盒二元优化框架**，用于在 LLM 中进行细粒度的 SWA attention head 选择。

其核心思想是将问题建模为 **Large Neighborhood Search (LNS)**，并分解为三个阶段：

1. **Layer Importance Detection**  
   使用小预算的黑盒搜索（black-box probe）从顶层到底层逐层评估每层对局部化的敏感度，构建“重要性”排序。

2. **Adaptive Per-Layer SWA Ratio Assignment**  
   基于上一步的敏感度，动态分配每层应转换为 SWA 的 head 比例（ratio），确保在总预算下优先在“更安全”的层中进行局部化。

3. **Grouped Head-Level Optimization**  
   将具有相同目标 ratio 的层分组，在组内联合优化 head 级别的选择，以捕捉 head 间的交互并缓解 entanglement。

### **相比现有方法的优势**

- ✅ **优于 layer-level 方法**：避免整层替换带来的信息损失。
- ✅ **优于 static head-level 方法**：通过黑盒搜索直接优化下游任务表现，而非依赖易受 entanglement 影响的静态指标。
- ✅ **训练免费（training-free）**：仅需少量校准数据（calibration set）进行前向传播评估。
- ✅ **自适应性强**：针对每个目标 SWA ratio 独立运行，避免“一刀切”的静态排名。
- ✅ **可扩展性好**：通过分阶段搜索，有效控制搜索空间爆炸问题。

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### **校准数据集（Calibration Set D）**
- **NIAH（Needle-in-a-Haystack）**：合成生成的长文本中定位特定答案的任务，用于探测模型的长距离关联能力。
- 包含 64 个样本，每条长度为 32k tokens。
- 用于执行 BOSCH 的黑盒优化过程。

#### **评估基准（Evaluation Benchmarks）**
- **NIAH Benchmark**：平均 6 个 NIAH 子任务的表现。
- **LongBench**（Bai et al., 2024）：包含 30 个长上下文 QA 任务，涵盖单/多文档问答、摘要、少样本学习、代码补全等。
  - 额外加入 **GSM8K** 作为数学推理任务。
  - 最终报告 **7 个类别的未加权平均分**。

### **实验设置和评估指标**

- **模型**：Qwen3 系列，共 4 个规模：
  - `1.7B`, `8B`, `14B`, `30B` 参数。
  - 均支持 32k 或更长序列（30B 支持 40k）。
  - 使用 **Grouped-Query Attention (GQA)**，因此 head 分组决策需一致。

- **SWA 设置**：
  - SWA window size 固定为 **1024**（为最大长度的 1/32）。
  - 测试 SWA ratio $ p \in \{0.25, 0.5, 0.75, 0.875\} $，即 25% 到 87.5% 的 head 转换为 SWA。

- **评估方式**：
  - **Zero-shot 性能**：直接评估 hybrid 模型在 NIAH 和 LongBench 上的表现。
  - **Continual Pretraining**：在 hybrid 模型基础上继续预训练 2.5B tokens，观察性能恢复情况。

- **评估指标**：
  - 平均准确率（Accuracy）。
  - 性能恢复速度与最终水平（continual training 曲线）。

### **基线方法对比**

#### **Layer-level Heuristics**
- `RAND`：随机选择 layer 进行替换。
- `BME`（Begin-Middle-End）：选择开头、中间、结尾的 layer。
- `INTR`（Interleave）：交替选择 layer。

#### **Head-level Static Methods**
- `DCAM`：基于注意力质量分布判断局部性。
- `APL`：基于答案位置的峰值滞后。
- `PROXY`：代理注意力机制估计重要性。
- `QADA`：查询自适应注意力判据。
- `RAZOR`：基于重复 token 探针识别检索头。
- `FISHER`：基于 Fisher 信息估计 head 重要性。

#### **BOSCH 消融变体**
- `B-single`：仅使用第一阶段结果。
- `B-multi`：第一阶段扩展到多层。
- `B-layer`：第一阶段在 layer 级别运行。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **Zero-shot Performance（Tables 1 & 2）**

| 方法 / Ratio | p=0.25 (NIAH) | p=0.5 (NIAH) | p=0.75 (NIAH) | p=0.875 (NIAH) |
|-------------|---------------|--------------|---------------|----------------|
| **BOSCH (1.7B)** | **91.8** | **78.3** | **58.0** | **30.3** |
| **BOSCH (8B)**   | **98.9** | **90.3** | **72.7** | **42.5** |
| **BOSCH (14B)**  | **99.2** | **94.0** | **83.6** | **47.2** |
| **BOSCH (30B)**  | **97.5** | **86.3** | **50.2** | **26.9** |

> 注：所有配置下，BOSCH 在 **NIAH** 和 **LongBench** 上均显著优于所有基线，且优势随 $ p $ 增大而扩大。

#### **典型对比示例（p=0.75, 14B）**
- **BOSCH**: 83.6 (NIAH), 38.0 (LongBench)
- **FISHER**（最强基线）: 71.6, 31.6
- **RAZOR**: 71.3, 35.4
- **INTR**: 11.9, 11.9

👉 **BOSCH 在 NIAH 上领先超过 12 个百分点**。

---

### **与基线方法的对比结果**

- ✅ **全面超越 layer-level 方法**：`INTR`, `BME`, `RAND` 表现极差，甚至不如随机。
- ✅ **显著优于所有 static head-level 方法**：即使是表现最好的 `FISHER` 和 `RAZOR`，也明显落后于 BOSCH。
- ✅ **优势随 SWA ratio 增大而增强**：说明 BOSCH 在高压缩比下更具鲁棒性。
- ✅ **模型大小影响小**：在 1.7B 到 30B 范围内表现稳定，表明方法具有 **model-agnostic** 特性。

---

### **消融实验结果**

| 变体 | NIAH (p=0.5, 8B) | LongBench (p=0.5, 8B) |
|------|------------------|------------------------|
| **BOSCH** | **90.3** | **41.8** |
| `B-single` | 88.9 | 36.4 |
| `B-multi` | 88.8 | 38.0 |
| `B-layer` | 89.7 | 36.3 |

- ❗ **仅使用第一阶段（B-single）性能大幅下降**，说明后续两阶段至关重要。
- ✅ **B-multi 和 B-layer 表现接近 FISHER/RAZOR**，但仍低于完整 BOSCH，说明 **head-level entanglement mitigation 是关键增益来源**。

---

## 4. 关键结论和发现

### **主要发现**

1. **Entanglement 效应真实存在且严重影响性能**  
   - 静态方法假设 head 局部性不变，但 hybridization 后行为改变。
   - BOSCH 通过黑盒优化直接响应这种变化，获得更优解。

2. **Turnover 现象显著**  
   - 不同 $ p $ 下 BOSCH 选出的 SWA head 集合差异大（Table 3）。
   - 几何归一化 turnover rate 达 **26%-44%**，说明不能复用单一静态排名。

3. **Jaccard Distance 与性能正相关**  
   - BOSCH 与其他方法的 Jaccard 距离大，且与性能差距一致。
   - 表明其发现了不同的、更有效的 head 组合。

4. **Continual Pretraining 中恢复更快更高**  
   - 图 3 显示，BOSCH 初始化的 hybrid 模型在继续训练后：
     - 收敛速度更快。
     - 最终性能更高，尤其在 LongBench 上。
   - 说明 **更好的初始化带来更强泛化能力**。

5. **零样本长度外推表现稳健**  
   - 在 64k 和 128k 序列上测试（via YaRN），BOSCH 仍保持领先。

---

### **方法的局限性**

- **计算开销较高**：相比静态方法，BOSCH 需要多次前向传播进行搜索（最长约 14 小时/配置）。
- **仅限 Qwen3 家族验证**：实验集中在 Qwen3 模型，缺乏跨架构泛化验证。
- **未结合量化等其他效率技术**：未探索与 KV Cache 压缩、weight pruning 的协同效应。
- **不适用于 zero-shot search 的混合模块**：如 SSM-based 混合需重新设计。

---

### **未来工作方向**

1. **扩展至其他混合原语（hybrid primitives）**  
   - 如整合 **SSM layers**（如 Mamba），但需解决无法零样本搜索的问题。

2. **适配 Multi-Latent Attention**  
   - 将所有 attention head 压缩为连续向量，进一步降低内存。

3. **降低搜索成本**  
   - 设计更高效的代理模型或 warm-start 策略，减少评估次数。

4. **探索与其他效率技术的联合优化**  
   - 如与 **quantization**, **KV-cache compression**, **pruning** 联合设计。

5. **跨模型家族迁移**  
   - 验证 BOSCH 是否可在 LLaMA、Phi 等架构上通用。

---

> **总结一句话**：  
> BOSCH 通过 **分阶段黑盒优化**，实现了 **无需训练、自适应、抗 entanglement 的 SWA head 选择**，在多个模型和任务上 **系统性超越** 现有 layer-level 和 static head-level 方法，为高效长上下文 LLM 部署提供了新范式。

</details>

---

### 12. [Hardware-Oriented Inference Complexity of Kolmogorov-Arnold Networks](https://arxiv.org/abs/2604.03345)

**Authors**: Bilal Khalid, Pedro Freire, Sergei K. Turitsyn, Jaroslaw E. Prilepsky  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.03345v1  

#### Abstract
Kolmogorov-Arnold Networks (KANs) have recently emerged as a powerful architecture for various machine learning applications. However, their unique structure raises significant concerns regarding their computational overhead. Existing studies primarily evaluate KAN complexity in terms of Floating-Po...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Hardware-Oriented Inference Complexity of Kolmogorov-Arnold Networks**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题  
当前对 **Kolmogorov-Arnold Networks (KANs)** 的复杂度分析主要依赖于 **Floating-Point Operations (FLOPs)**，这种指标适用于 GPU 上的训练和推理场景，但**无法准确反映在专用硬件（如 FPGA 或 ASIC）上的实际推理开销**。此外，已有硬件实现研究使用平台相关的资源消耗指标（如 LUT、BRAM、FF），这些指标需要完整的设计与综合流程，**不适用于早期架构探索和跨平台比较**。

因此，本文旨在解决以下问题：
- 如何建立一个**平台无关、可快速计算、能真实反映硬件推理成本**的复杂度评估框架？
- 如何公平地比较 KAN 与其他神经网络（如 MLP）在硬件部署中的效率？

---

### ✅ 提出了什么新方法或新思路  
作者提出了一套**面向硬件推理的通用复杂度评估框架**，定义并推导了三种**平台无关的分析型复杂度指标**：

| 指标 | 含义 |
|------|------|
| **Real Multiplications (RM)** | 仅统计乘法操作次数，因乘法器在硬件中远比加法器昂贵 |
| **Bit Operations (BOP)** | 考虑定点量化下不同位宽的操作代价，区分乘法与累加的成本 |
| **Number of Additions and Bit Shifts (NABS)** | 将乘法用“移位+加法”实现时所需的等效加法器数量 |

该方法可直接从网络结构参数（如层数、宽度、基函数类型及参数）计算得出，无需实际硬件设计。

---

### ✅ 相比现有方法的优势  

| 对比维度 | 传统方法（FLOPs / 硬件资源） | 本文方法（RM/BOP/NABS） |
|--------|-------------------------------|--------------------------|
| **适用性** | GPU为中心 / 平台相关 | 面向专用硬件 / 平台无关 |
| **计算便捷性** | FLOPs 易算；硬件资源需综合 | 可解析推导，支持快速设计空间搜索 |
| **准确性** | 忽略稀疏性优化（如 B-spline 局部支撑） | 显式建模硬件优化策略（如 LUT 查表） |
| **可比性** | 不同平台间难以横向对比 | 支持跨架构、跨平台公平比较 |

> 🔍 特别指出：传统的 FLOPs 分析会高估 B-spline KAN 的复杂度，因为它假设所有基函数都需计算，而实际上利用**局部支撑性质（local support）** 和查表优化后，只有 $k+1$ 个非零项参与运算。

---

## 2. **核心实验方法和设置**

### ✅ 分析的 KAN 变体  
论文系统分析了四种主流 KAN 架构：
- **B-spline KAN**
- **Gaussian Radial Basis Function (GRBF) KAN**
- **Chebyshev Polynomial KAN**
- **Fourier Basis KAN**

每种均基于其数学表达式推导出对应的 RM、BOP 和 NABS 公式（见 Table I）。

---

### ✅ 实验设置与评估指标  

#### 🧪 网络架构设定  
采用统一的多层结构进行对比分析：
- 架构形式：`[3, X, X, 2]`，即输入维数 3，两个隐藏层各 $X$ 个神经元，输出维数 2
- 控制变量分析中令 $X$ 从 4 到 64 变化，观察复杂度随规模增长趋势

#### ⚙️ 参数配置（代表性取值）
| 类型 | 参数设置 |
|------|---------|
| B-spline | 阶数 $k=3$, 网格大小 $G=5$ |
| GRBF | 中心数 $N_c=5$ |
| Chebyshev | 最大阶次 $n=5$ |
| Fourier | 网格大小 $G=5$（共 $2G=10$ 项） |

> 注：这些参数用于说明性分析，并非全局最优。

#### 📏 评估指标  
- **RM**: Real Multiplications per layer
- **BOP**: Bit Operations（设为 8-bit 均匀量化）
- **NABS**: Number of Additions and Bit Shifts（设 $X_u = b - 1$）

---

### ✅ 基线方法对比  
以标准 **Multi-Layer Perceptron (MLP)** 作为基准模型，在相同网络宽度和深度下进行对比。

---

## 3. **主要实验结果和性能指标**

### ✅ 关键性能数据（图 3）

在 `[3,16,16,2]` 架构下，与 MLP 相比：

| 模型 | RM 开销倍数 | BOP 开销倍数 | NABS 开销倍数 |
|------|-------------|--------------|---------------|
| **B-spline KAN** ($k=3$) | ~6× | ~5.5× | ~5.1× |
| **GRBF KAN** ($N_c=5$) | 更高 | 更高 | 更高 |
| **Chebyshev KAN** ($n=5$) | 更高 | 更高 | 更高 |
| **Fourier KAN** ($G=5$) | 最高（~13× RM） | 最高 | 最高 |

> 💡 结论：**所有 KAN 变体在相同结构下均显著高于 MLP 的硬件推理成本**。

---

### ✅ 与基线方法的对比结果  

#### 🔺 图 4：复杂度随网络宽度的变化  
- 所有模型呈现 **二次方增长趋势**（由 $n_{\text{in}} \times n_{\text{out}}$ 主导）
- KAN 与 MLP 的**相对开销比例保持恒定**，表明单位边的额外负担是固定的

#### 🔺 图 5：Iso-Complexity Analysis（等复杂度分析）  
目标：找出在**与 `[3,64,64,2]` MLP 相同硬件预算下**，各类 KAN 能达到的最大宽度 $X$

| 模型 | 最大可行 $X$（近似） |
|------|---------------------|
| **MLP**（参考） | 64 |
| **B-spline KAN** | **25–29** |
| **GRBF KAN** | **23–27** |
| **Chebyshev KAN** | **19–26** |
| **Fourier KAN** | **18–19** |

> ✅ 发现：尽管 KAN 单边开销大，但由于更强的逼近能力，可用更窄结构达到相似精度，仍具竞争力。

---

### ✅ 消融实验结果（隐含分析）  

虽然未明确命名“消融实验”，但文中通过参数敏感性分析实现了类似目的：

- **B-spline KAN 的优势来自局部支撑特性**：仅需计算 $k+1=4$ 项，不受网格大小 $G$ 影响
- **Fourier/GRBF/Chebyshev KAN 无局部支撑**：必须计算全部基函数，导致线性增长
- **LUT 优化极大降低 RM/BOP**：将递归计算替换为查表后，B-spline 的 $C_{\text{basis}} \to 0$
- **不同量化方案影响 NABS**：APoT vs PoT 会影响 $X_u$，从而改变 NABS 数值

---

## 4. **关键结论和发现**

### ✅ 主要发现  

1. **KAN 的硬件推理成本显著高于同等 MLP**  
   - 在相同结构下，KAN 的 RM/BOP/NABS 普遍高出 5–13 倍
   - 成本主要来源于每条边上需执行多个基函数的加权组合

2. **B-spline KAN 是最高效的变体之一**  
   - 得益于**局部支撑性**和**查表优化潜力**，其每边激活函数仅需 $k+3$ 次实数乘法
   - 在等复杂度条件下可支持最宽网络结构

3. **提出的 RM/BOP/NABS 指标高度一致且可信**  
   - 尽管三个指标衡量不同物理量，但在 iso-complexity 分析中给出几乎相同的等效网络宽度
   - 表明公式正确捕捉了底层硬件行为

4. **KAN 的高参数效率可能补偿其高硬件成本**  
   - 文献表明 KAN 可用更少参数达到甚至超越 MLP 性能
   - 因此可通过缩小网络宽度来平衡硬件开销，实现“小而强”的设计

---

### ⚠️ 方法的局限性  

1. **未考虑内存带宽与缓存效应**  
   - 当前模型忽略 LUT 大小对片上存储的压力
   - 大规模 GRBF/Fourier KAN 可能耗尽 BRAM 资源

2. **假设理想化硬件优化**  
   - 默认所有非线性函数（如 exp, tanh, sin/cos）均可用 LUT 实现
   - 实际中可能存在精度损失或延迟瓶颈

3. **未涵盖所有 KAN 变体**  
   - 如 Wavelet-KAN、Fractional KAN 等新型结构未被纳入分析

4. **缺乏真实硬件验证**  
   - 所有结论基于理论推导，尚未在 FPGA/ASIC 上实测验证

---

### 🔮 未来工作方向  

1. **扩展至更多 KAN 架构**  
   - 将分析框架推广到 Wav-KAN、FKAN、GraphKAN 等新兴结构

2. **结合能效建模**  
   - 引入功耗估计模型（如每 RM/BOP 的能耗），构建能效感知的设计工具

3. **自动化架构搜索（NAS）集成**  
   - 将 RM/BOP/NABS 作为约束条件，用于联合优化 KAN 结构与硬件映射

4. **软硬协同设计指南**  
   - 提供“给定硬件预算下如何选择 KAN 类型与参数”的实用建议手册

---

## ✅ 总结一句话  
> 本文提出了首个面向硬件推理的 **平台无关 KAN 复杂度评估框架（RM/BOP/NABS）**，揭示了 KAN 相较 MLP 的高昂硬件代价，但也证明通过合理缩放网络宽度，KAN 依然可在有限资源下发挥其强大的函数逼近优势，为未来高效 KAN 加速器设计提供了理论基础。

</details>

---

### 13. [Simple yet Effective: Low-Rank Spatial Attention for Neural Operators](https://arxiv.org/abs/2604.03582)

**Authors**: Zherui Yang, Haiyang Xin, Tao Du, Ligang Liu  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.03582v1  

#### Abstract
Neural operators have emerged as data-driven surrogates for solving partial differential equations (PDEs), and their success hinges on efficiently modeling the long-range, global coupling among spatial points induced by the underlying physics. In many PDE regimes, the induced global interaction kern...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Simple yet Effective: Low-Rank Spatial Attention for Neural Operators

---

## 1. 论文的主要贡献和创新点

### 解决的问题
神经算子（Neural Operators）在求解偏微分方程（PDEs）时面临**高效建模空间点之间长距离、全局耦合关系**的挑战。传统方法如 Fourier Neural Operator（FNO）依赖预定义的解析基函数（如傅里叶基），难以适应不规则几何；而基于注意力机制的方法虽然灵活，但全注意力计算复杂度为 $O(N^2)$，且常需引入非标准归一化或聚合模块以保证稳定性，导致实现复杂、硬件兼容性差。

### 提出的新方法与新思路
本文提出 **Low-Rank Spatial Attention (LRSA)**，一种简洁高效的全局混合模块，其核心思想是：
- 将多种神经算子中的全局混合机制统一到一个**低秩压缩-处理-重建（compress-process-reconstruct）模板**下。
- 在此框架下，设计了一个**完全由标准 Transformer 组件构成**的模块：cross-attention 用于压缩与重建，self-attention 和 FFN 用于潜空间内的非线性交互。

具体流程如下：
1. **Compression**: 使用可学习的 latent queries 通过 cross-attention 将 $N$ 个空间点特征压缩为 $M$ 个紧凑的 latent tokens。
2. **Processing**: 在 latent space 内应用标准 Transformer block（self-attention + FFN）进行全局信息融合。
3. **Reconstruction**: 再次使用 cross-attention 将处理后的 latent tokens 映射回 $N$ 个输出点，完成上下文重建。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **架构简洁性** | 仅使用标准 attention、normalization、FFN，无需自定义 slice、assignment 或 renormalization 操作。 |
| **硬件友好性** | 可直接利用 FlashAttention-2 等高度优化的 fused kernels，提升训练效率。 |
| **数值稳定性** | 避免了显式的 $N \times M$ 权重矩阵归一化，在 FP16/BF16 下依然稳定训练。 |
| **表达能力更强** | 解耦压缩与重建路径，并保留 latent space 中的非线性 self-attention，提升了模型灵活性。 |
| **分辨率不变性** | 设计天然支持不同离散化下的泛化能力。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖三大类任务，涵盖从规则网格到工业级复杂几何体：

#### （1）标准基准（Standard Benchmarks）
| 数据集 | 类型 | 分辨率 | 描述 |
|--------|------|--------|------|
| Darcy Flow | Regular Grid | 85×85 | 多孔介质流，输入为扩散系数场 |
| Navier-Stokes | Regular Grid | 64×64 | 不可压粘性流，时间序列预测 |
| Airfoil / Pipe / Plasticity | Structured Mesh | ~10k pts | 翼型绕流、变形状管道速度场、塑性材料变形 |
| Elasticity | Point Cloud | 972 pts | 弹性体应力分布预测 |

#### （2）不规则域基准（Irregular Domains）
| 数据集 | 几何复杂度 | 节点数 | 特点 |
|--------|------------|--------|------|
| Irregular Darcy | 复杂边界 | 2290 | 标准 FFT 方法失效 |
| Pipe Turbulence | 弯曲管道 | 2673 | 湍流建模 |
| Heat Transfer / Composite | 3D 结构 | >7000 | 温度驱动形变预测 |

#### （3）工业级大规模案例
| 数据集 | 规模 | 应用场景 |
|--------|------|----------|
| ShapeNet Car | ~32k mesh points | 汽车气动阻力 $C_D$ 预测 |
| AirfRANS | ~32k mesh points | 高精度翼型升力 $C_L$ 预测 |

---

### 实验设置和评估指标

#### 主要评估指标
- **Relative L2 Error**: $\frac{\|u_{\text{pred}} - u_{\text{true}}\|_2}{\|u_{\text{true}}\|_2}$，用于物理场预测。
- **MSE on Volume/Surface Fields**: 体积与表面区域误差。
- **Relative Error & Spearman’s Rank Correlation ($\rho$)**: 对于 $C_D$, $C_L$ 等工程量，衡量排序一致性（对设计优化至关重要）。

#### 基线方法对比
| 类别 | 方法 |
|------|------|
| **Spectral/Basis-based** | FNO, F-FNO, LSM, HPM, NORM |
| **Attention-based** | Transolver, Transolver++, LinearNO, LNO |
| **GNN-based** | GraphSAGE, MeshGraphNet, GNO |

所有模型控制参数量相近，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### ✅ 标准基准表现（Relative L2 Error ↓）
| Method | Airfoil | Pipe | Plasticity | NS | Darcy | Elasticity |
|--------|-------|------|-----------|-----|-------|-----------|
| **Ours (LRSA)** | **0.0042** | **0.0023** | **0.0005** | **0.0484** | **0.0043** | **0.0033** |
| Second Best | 0.0049 | 0.0024 | 0.0011 | 0.0699 | 0.0050 | 0.0050 |

👉 **平均相对误差降低超过 17%**，尤其在动态系统（如 Navier-Stokes）上优势显著。

#### ✅ 不规则域表现（Table 2）
| Method | Irregular Darcy | Pipe Turbulence | Heat Transfer | Composite |
|--------|------------------|------------------|---------------|-----------|
| **Ours (LRSA)** | 7.56e-3 | **4.89e-3** | **1.49e-4** | **8.71e-3** |
| HPM (best prior) | 7.36e-3 | 8.26e-3 | 1.84e-4 | 9.34e-3 |

👉 在多数任务上超越依赖几何先验的 HPM/NORM，证明**无需显式 Laplace 特征函数即可达到高性能**。

#### ✅ 工业级任务表现（Table 3）
| Model | AirfRANS (Surface) | AirfRANS ($C_L$ rel err) | AirfRANS ($\rho_L$) |
|-------|--------------------|----------------------------|---------------------|
| Transolver | 0.0085 | 0.1230 | 0.9978 |
| LinearNO | 0.0077 | 0.0491 | 0.9992 |
| **Ours** | **0.0010** | **0.0396** | **0.9997** |

👉 在高分辨率工业仿真中大幅领先，**rank correlation 接近完美（0.9997）**，具备实际设计指导价值。

---

### 消融实验结果（Ablation Study）

#### （1）组件分析（Table 12）
| Variant | Navier-Stokes (rel L2) |
|---------|------------------------|
| Full LRSA | **4.84e-2** |
| w/o Intra Attention (replace with MLP) | 5.44e-2 (+12.4%) |
| Enforce Symmetric (like Transolver) | 5.93e-2 (+22.5%) |

👉 **latent space 中的 self-attention 至关重要**，且**解耦压缩与重建优于对称绑定**。

#### （2）潜空间维度 $M$ 敏感性（Figure 5）
- 当 $M$ 达到一定值后误差趋于饱和，验证了“有效低秩”的假设。
- 在 Navier-Stokes 上，LRSA 随 $M$ 增加持续增益，而 Transolver 收敛缓慢。

#### （3）混合精度训练稳定性（Figure 4）
| Method/Precision | FP32 | BF16 | FP16 |
|------------------|------|------|------|
| Transolver | ✓ | ✓ (误差上升) | ✗ (**发散**) |
| **LRSA** | ✓ | ✓ | ✓ (**稳定**) |

👉 **LRSA 在 FP16 下仍保持高精度，而 Transolver 完全无法收敛**。

#### （4）效率提升（Table 11）
在 ShapeNet Car 上（N≈32k）：
- **训练延迟下降 3.2×**
- **峰值内存减少 2.1×**

得益于 FlashAttention-2 的自动调度与数值鲁棒性。

---

## 4. 关键结论和发现

### 主要发现
1. **统一视角**：大多数高效神经算子（FNO、Transolver 等）均可视为低秩因子分解 $K \approx UGV^\top$ 的特例，LRSA 提供了一种通用解释框架。
2. **简单即强大**：仅使用标准 Transformer 组件即可构建高性能、高稳定性的神经算子模块，无需复杂手工设计。
3. **硬件协同设计有效**：采用标准 attention 可无缝接入 FlashAttention 等优化内核，带来显著的速度与内存收益。
4. **解耦优于对称**：允许压缩与重建使用不同的 basis（via decoupled cross-attention）能更好地捕捉复杂的物理耦合模式。
5. **混合精度可行**：LRSA 在 FP16 下稳定训练，解锁了大规模、高分辨率 PDE 仿真的实用潜力。

### 方法的局限性
- **潜空间大小 $M$ 需手动设定**，缺乏自适应机制。
- 当前设计未显式编码守恒律或对称性等物理归纳偏置。
- 在极小样本场景下（如 <200 训练样本），带有强几何先验的方法（如 HPM）可能更具优势（见 Table 5）。

### 未来工作方向
- 探索 **principled capacity allocation strategy**，动态分配 latent processing 与 pointwise update 的容量。
- 结合 **large-scale multi-physics pretraining** 以增强跨任务泛化能力。
- 引入 **物理约束正则化** 或 **equivariant design** 进一步提升保真度。
- 扩展至更高维（3D+time）或更复杂耦合系统（如 multiphase flow）。

--- 

> 💡 **一句话总结**：  
> LRSA 通过一个**纯标准组件构成的低秩注意力块**，实现了**高精度、高效率、高稳定性**的神经算子设计，在多样化的 PDE 任务上全面超越现有方法，推动了 operator learning 向实用化迈进。

</details>

---

### 14. [APPA: Adaptive Preference Pluralistic Alignment for Fair Federated RLHF of LLMs](https://arxiv.org/abs/2604.04261)

**Authors**: Mahmoud Srewa, Tianyu Zhao, Salma Elmalaki  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.04261v1  

#### Abstract
Aligning large language models (LLMs) with diverse human preferences requires pluralistic alignment, where a single model must respect the values of multiple distinct groups simultaneously. In federated reinforcement learning from human feedback (FedRLHF), these groups align a shared policy without ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：APPA: Adaptive Preference Pluralistic Alignment for Fair Federated RLHF of LLMs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Federated Reinforcement Learning from Human Feedback (FedRLHF)** 场景中，如何公平地对齐 **Large Language Models (LLMs)** 与多个异构群体（如不同国家、文化、人口统计群体）的多样化偏好是一个关键挑战。  
现有聚合策略存在明显缺陷：
- **Average aggregation**：导致“多数偏见”（majority bias），系统性地忽视表现最差的群体（worst-performing groups）。
- **Min aggregation**：虽提升最差群体的表现，但牺牲了整体对齐效果（overall alignment），且忽略其他群体的有用反馈。

该问题本质上是 **pluralistic alignment**（多元价值对齐）与 **group fairness**（群体公平性）之间的权衡。

---

### 提出了什么新方法或新思路
作者提出 **APPA (Adaptive Preference Pluralistic Alignment)** 框架，其核心思想是：
> **基于历史对齐奖励动态重加权各群体的奖励信号，在不损害已良好对齐群体的前提下优先关注对齐不足的群体。**

#### 主要创新点：
1. **自适应奖励聚合机制（Adaptive Alpha Aggregation）**
   - 引入 **group-specific adaptive weights**，取代传统固定标量 α 的聚合方式（如 Park et al., 2024 中的 α-aggregation）。
   - 权重由每个群体的历史对齐表现通过 **指数移动平均（EMA）** 动态计算，并使用 **reverse softmax** 赋予低对齐群体更高权重。

2. **公平性感知切换机制（Fairness-aware Threshold Switching）**
   - 定义 **Fairness Index (FI)** 衡量跨群体奖励差异（基于变异系数 CoV）。
   - 当 FI ≥ τ（阈值，默认 0.99）时，采用简单平均；否则启用自适应 log-sum-exp 加权，防止训练后期不必要的扰动。

3. **完全去中心化与隐私保护**
   - 仅需群体级奖励（group-level rewards），无需访问原始偏好数据，符合联邦学习的隐私要求。

---

### 相比现有方法的优势
| 特性 | Average Aggregation | Min Aggregation | APPA (Ours) |
|------|---------------------|------------------|-------------|
| 对齐所有群体能力 | ✗（偏向多数） | ✗（牺牲整体） | ✓（兼顾平均与最差） |
| 支持持续改进 | ✗（梯度被主导群体捕获） | ✗（只更新最差组） | ✓（所有组始终参与） |
| 无需原始数据 | ✓ | ✓ | ✓ |
| 自适应调整 | ✗ | ✗ | ✓（基于历史动态调整） |

> APPA 成功缓解了“平均 vs 最小”的两难困境，在大多数配置下实现了更优的 **fairness-alignment trade-off**。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 | 群体划分 | 任务特点 |
|--------|------|----------|---------|
| **GLOBALQA** | Pew Research 全球态度调查数据集（DURMUS et al., 2024） | 8个国家（如 Nigeria, Germany, USA） | 跨国多元对齐，答案多为名义型（nominal） |
| **OQA (OpinionQA)** | Santurkar et al. (2023) 构建的美国民意数据集 | 8个美国人口统计群体（如 Republican, Male, $100k+） | 国内人口多样性对齐，答案多为序数型（ordinal） |

---

### 实验设置
- **模型家族**：在三种规模不同的开源 LLM 上验证泛化性：
  - `Gemma-2-2B`
  - `Llama-3.2-3B`
  - `Qwen3-0.6B`

- **训练流程**：
  1. 先进行 **Supervised Fine-Tuning (SFT)** 得到初始策略模型。
  2. 在 **Proximal Policy Optimization (PPO)** 框架下执行 FedRLHF，每轮服务器广播 rollout 到各客户端。
  3. 各客户端使用本地冻结的 **PluralLLM** 模块生成群体偏好分布并返回奖励。
  4. 服务器使用 APPA 进行奖励聚合后更新全局策略。

- **实现细节**：
  - 使用 Hugging Face TRL 库。
  - 所有模型以 4-bit 量化运行于单个 A100 GPU。
  - 超参数统一设定（如 EMA decay λ=0.8, 温度 T=0.1, FI 阈值 τ=0.99）。

---

### 评估指标
#### （1）对齐质量（Alignment Score）
- **Per-group Alignment Score (AS)**：每个群体在测试集上的平均奖励。
- **Avg AS**：所有群体 AS 的均值。
- **Min AS**：所有群体 AS 的最小值（反映最弱群体表现）。

#### （2）公平性指标
- **Fairness Index (FI)**：基于每题各群体奖励的变异系数（CoV）定义，范围 [0,1]，越接近 1 表示群体间差异越小。
  $$
  \text{FI} = \frac{1}{|X_t|} \sum_{q_i \in X_t} \frac{1}{1 + \text{CoV}(r_g^{(i)})}
  $$

#### （3）任务类型
- **DPA (Distributional Preference Alignment)**：预测完整偏好分布，使用以下指标：
  - **JS (Jensen-Shannon Divergence)**：主用于 GLOBALQA
  - **Was. (Wasserstein Distance)**：主用于 OQA（因有序选项）
  - **Cos. (Cosine Similarity)**
- **OPA (Ordinal Preference Alignment)**：输出排序列表，使用 **Borda Score** 作为主指标。

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **SFT** | 在多数标签上微调，代表主流做法，易引入多数偏见 |
| **PPO + Average** | 各群体奖励取平均，平等对待但可能掩盖弱势群体 |
| **PPO + Min** | 取最小奖励作为优化目标，追求最坏情况下的公平性 |
| **PPO + APPA (Ours)** | 提出的自适应加权方法 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Tables 2 & 3）

#### ✅ GLOBALQA – DPA（JS 为主指标）
| Model | 方法 | Avg AS (JS) | Min AS (JS) | FI |
|-------|------|------------|------------|----|
| Gemma-2-2B | APPA | **0.861** | **0.843** | **0.9994** |
| | Min | 0.834 | 0.808 | 0.9995 |
| | Average | 0.839 | 0.812 | 0.9695 |

> APPA 在 **Avg 和 Min AS 上同时领先**，FI 接近完美。

#### ✅ GLOBALQA – OPA（Borda 为主指标）
| Model | 方法 | Avg AS (Bor.) | Min AS (Bor.) | FI |
|-------|------|--------------|--------------|----|
| Gemma-2-2B | APPA | **0.511** | **0.461** | **0.891** |
| | Min | 0.475 | 0.420 | 0.873 |
| | Average | 0.469 | 0.359 | 0.854 |

> APPA 显著优于所有基线，尤其在 **Min AS 上比 Average 提升达 28%**。

#### ✅ OQA – DPA（Was. 为主指标）
| Model | 方法 | Avg AS (Was.) | Min AS (Was.) | FI |
|-------|------|---------------|---------------|----|
| Llama-3.2-3B | APPA | **0.872** | **0.841** | **0.9940** |
| | Min | 0.856 | 0.811 | 0.9937 |
| | Average | 0.859 | 0.824 | 0.9645 |

> 再次证明 APPA 在高容量模型上的优越性。

#### ⚠️ OQA – DPA on Qwen3-0.6B（例外情况）
| 方法 | Avg AS (Was.) | Min AS (Was.) |
|------|---------------|---------------|
| Min | **0.823** | 0.781 |
| APPA | 0.780 | **0.735** |

> 小模型上 Min 表现更好，作者归因于 **模型容量限制导致无法有效响应 Wasserstein 梯度信号**，削弱了自适应权重的作用。

---

### 与其他方法的对比总结
- APPA 在 **绝大多数配置下实现了最高的 Avg AS 和 Min AS**。
- 相比 Average，**Min AS 提升高达 28%**。
- 相比 Min，**Avg AS 更高，避免了整体性能退化**。
- FI 普遍达到 ≥0.99，说明最终模型在群体间高度公平。

---

### 消融实验与分析（隐含在设计中）
虽然未显式列出消融表，但从机制设计可推断以下关键组件作用：
| 组件 | 作用 | 若移除的影响 |
|------|------|-------------|
| **Historical EMA (h_g)** | 平滑短期波动，提供长期对齐趋势 | 权重易受噪声干扰，不稳定 |
| **Reverse Softmax (1−h)** | 确保低对齐群体获得高权重 | 失去“优先关注落后者”机制 |
| **FI Threshold (τ=0.99)** | 避免收敛后过度调整 | 可能引发后期震荡或过拟合 |
| **Non-zero weights for all groups** | 所有群体持续贡献梯度 | 类似 Min 的极端选择行为 |

---

## 4. 关键结论和发现

### 主要发现
1. **奖励聚合方式显著影响 FedRLHF 的公平性与有效性**。
2. **APPA 能够动态平衡平均与最差群体的对齐表现**，解决了传统方法中的根本权衡。
3. **历史对齐信息可用于构建自适应、公平感知的聚合机制**，而无需访问原始数据。
4. **APPA 在多种模型规模（0.6B–3B）、数据集（跨国/国内）、任务类型（DPA/OPA）上均表现出强鲁棒性和泛化能力**。
5. **只有当模型具备足够表达能力时，APPA 的优势才能充分体现**（Qwen3-0.6B 是例外）。

---

### 方法的局限性
1. **依赖群体定义清晰且稳定的前提**：若群体边界模糊或动态变化，PluralLLM 和权重分配将面临挑战。
2. **小模型上性能受限**：低容量模型难以捕捉复杂偏好分布，削弱了自适应机制的效果。
3. **仅适用于结构化输出任务**：当前实验集中在多选问答（MCQ），尚未扩展到开放文本生成等非结构化任务。
4. **超参数敏感性未知**：尽管固定了 λ, T, τ，但未进行全面超参搜索，实际部署中可能需要调优。

---

### 未来工作方向
1. **扩展至开放域任务**：应用于长文本生成、代码合成、创意写作等场景，探索更丰富的群体级 reward signal 设计。
2. **动态群体发现**：结合聚类技术自动识别潜在偏好群体，而非预设分组。
3. **理论分析深化**：形式化证明 APPA 的收敛性、公平性保证及 Pareto 改进性质。
4. **跨模态联邦对齐**：将框架推广至图像、音频等多模态大模型的对齐任务。
5. **在线学习机制**：支持增量加入新群体而不重新训练整个系统。

---

> 💡 **一句话总结**：  
> **APPA 通过历史驱动的自适应奖励加权，在 FedRLHF 中实现了更公平、更高效的多元偏好对齐，为构建真正包容的 LLM 提供了一条可行路径。**

</details>

---

### 15. [FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control](https://arxiv.org/abs/2604.04539)

**Authors**: Donghu Kim, Youngdo Lee, Minho Park, Kinam Kim, I Made Aswin Nahendra, Takuma Seno, Sehee Min, Daniel Palenicek, Florian Vogt, Danica Kragic, Jan Peters, Jaegul Choo, Hojoon Lee  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.04539v1  

#### Abstract
Reinforcement learning (RL) is a core approach for robot control when expert demonstrations are unavailable. On-policy methods such as Proximal Policy Optimization (PPO) are widely used for their stability, but their reliance on narrowly distributed on-policy data limits accurate policy evaluation i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统强化学习在机器人控制中的两大范式存在显著瓶颈：
- **On-policy 方法**（如 PPO）虽然训练稳定，但严重依赖 on-policy 数据，样本效率低，在高维状态-动作空间中难以实现充分的状态覆盖，导致策略评估不准确。
- **Off-policy 方法**（如 SAC, TD3）虽能复用经验回放缓冲区（replay buffer）提升数据效率，但在高维任务中常因 **critic error accumulation**（值函数误差通过 bootstrapping 不断累积）而出现训练缓慢且不稳定的问题。

尤其在**高维机器人任务**（如灵巧手操作、人形机器人行走）中，上述问题被进一步放大，限制了 off-policy RL 在 sim-to-real 场景中的应用。

### 提出的新方法与核心思想
本文提出 **FlashSAC**，一种基于 Soft Actor-Critic (SAC) 的快速且稳定的 off-policy 强化学习算法，其核心创新在于结合了 **scaling law** 思想与显式的稳定性控制机制：

#### 主要创新点：
1. **Fast Training via Scaling Law**  
   受监督学习中 scaling law 启发，FlashSAC 采用“**大模型 + 大批量 + 少梯度更新**”的策略：
   - 使用高达 2.5M 参数的网络（远超传统 off-policy 方法的 0.2–0.5M）
   - 批量大小为 2048，充分利用 GPU 资源
   - 极低的更新频率（update-to-data ratio = 2/1024），即每收集 1024 条新数据仅进行 2 次梯度更新
   - 配合大规模并行仿真（1024 并行环境）和大容量 replay buffer（10M）

2. **Stable Training via Explicit Norm Control**  
   为防止大模型在 bootstrapping 过程中引发的误差累积，FlashSAC 引入多项架构级约束以稳定 critic 更新动态：
   - **Inverted Residual Backbone**：增强梯度传播稳定性
   - **Pre-activation Batch Normalization**：缓解非平稳输入分布问题
   - **Cross-Batch Value Prediction**：确保目标值与预测值共享归一化统计量
   - **Distributional Critic + Adaptive Reward Scaling**：平滑优化景观，避免极端回报影响
   - **Weight Normalization**：将权重投影到单位球面，抑制方差增长

3. **Enhanced Exploration Mechanisms**
   - **Unified Entropy Target**：通过固定 action standard deviation $ \sigma_{\text{tgt}} = 0.15 $ 自动设定目标熵，无需 per-task 调参
   - **Noise Repetition**：在多个连续步骤中重复同一噪声向量，生成时间相关的探索轨迹，提升稀疏奖励下的探索效率

### 相比现有方法的优势
| 维度 | FlashSAC | PPO | FastTD3 |
|------|--------|-----|---------|
| **训练速度** | ⭐⭐⭐⭐⭐（极快） | ⭐⭐ | ⭐⭐⭐⭐ |
| **最终性能** | ⭐⭐⭐⭐⭐（最优） | ⭐⭐⭐ | ⭐⭐⭐ |
| **稳定性** | ⭐⭐⭐⭐⭐（高） | ⭐⭐⭐⭐ | ⭐⭐ |
| **高维任务表现** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

> ✅ **核心优势总结**：FlashSAC 成功解决了 off-policy RL “快而不稳” 或 “稳而慢”的两难困境，在保持高稳定性的同时实现了前所未有的训练效率，尤其适用于高维、接触密集型的机器人控制任务。

---

## 2. 核心实验方法和设置

### 使用的数据集与模拟器
实验覆盖超过 **60 项任务**，来自 **10 个主流机器人仿真平台**，涵盖多种控制范式：

| 类别 | 模拟器 | 代表任务 |
|------|-------|----------|
| **GPU-based Simulators** | IsaacLab, ManiSkill3, Genesis, MuJoCo Playground | 抓取、四足行走、灵巧手操作、人形机器人行走 |
| **CPU-based Simulators** | MuJoCo, DeepMind Control Suite (DMC), HumanoidBench, MyoSuite | 连续控制基准任务、全身运动控制、肌骨系统建模 |
| **Vision-based RL** | DMC-Visual | 图像输入的操控与双足行走 |
| **Sim-to-Real** | IsaacLab + Unitree G1 人形机器人 | 盲目地形行走、楼梯攀爬 |

### 实验设置与评估指标
- **硬件配置**：单张 RTX 5090 GPU，AMD Ryzen 9 9950X3D CPU
- **评估维度**：
  - **Wall-clock time (min/hr)**：真实世界训练耗时（核心指标）
  - **Environment steps**：样本效率
  - **Normalized score / Episode return**：任务完成度
- **训练步数**：
  - Off-policy 方法（FlashSAC, FastTD3）：50M 步
  - PPO：200M 步（约 3× 计算预算）
- **统一超参数**：除折扣因子外，所有任务使用相同配置，体现泛化能力

### 基线方法对比
| 基线方法 | 类型 | 特点 |
|--------|------|------|
| **PPO** | On-policy | 当前 sim-to-real 主流方法，稳定但数据效率低 |
| **FastTD3** | Off-policy | 专为高速训练设计，小模型，速度快但性能受限 |
| **XQC, SimbaV2** | Off-policy | 注重样本效率与稳定性 |
| **TD-MPC2, MR.Q** | Model-based / Hybrid | 利用世界模型或辅助目标提升表示学习 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）高维任务性能碾压
在 **dexterous manipulation** 和 **humanoid locomotion** 等高维任务中：
- FlashSAC 在 **Isaac Shadow Hand Cube Repose** 上达到 **~10,000 分**，显著优于 PPO (~6,000) 和 FastTD3 (~7,000)
- 在 **Humanoid-v4** 上，FlashSAC 仅用 **40 分钟** 即超越 PPO 在 3 小时后的性能

#### （2）Sim-to-Real 训练时间大幅缩短
| 任务 | 方法 | 达到稳定行走所需训练时间 |
|------|------|--------------------------|
| **Flat Terrain** | FlashSAC | **~20 分钟** |
| | PPO | **~3 小时** |
| **Rough Terrain (Stairs)** | FlashSAC | **~4 小时** |
| | PPO | **~20 小时** |

> 🔥 **提速近一个数量级**（order of magnitude reduction）

#### （3）视觉控制任务表现优异
在 DMC-Visual 的 `finger-turn-hard` 等任务中：
- FlashSAC 收敛速度明显快于 DrQ-v2 和 MR.Q
- 最终性能更高且更稳定，无崩溃现象

---

### 与基线方法的对比结果
| 对比维度 | 结果 |
|--------|------|
| **vs. PPO** | 在低维任务上性能相当；在高维任务上全面超越，训练速度快 **5–10 倍** |
| **vs. FastTD3** | 更稳定（无训练崩溃），性能更高，尤其在 humanoid 任务上优势明显 |
| **vs. XQC/SimbaV2** | 在 CPU 单环境设置下仍保持领先，说明其设计具有通用性 |
| **vs. TD-MPC2/MR.Q** | 无需复杂的世界模型或辅助目标，计算开销更低，性能持平甚至反超 |

---

### 消融实验结果
#### （1）Scaling Components（图 8）
- **增大 replay buffer 至 10M** 显著提升稳定性与性能
- **增加 batch size 与 model capacity** 加速收敛
- **降低 UTD ratio**（减少更新次数）反而加快训练，验证 scaling law 有效性

#### （2）Architecture Ablation（图 9）
逐步添加以下组件后：
- Weight Norm → Feature Norm → Gradient Norm 全部得到有效控制
- Critic loss landscape condition number 明显下降
- 最终任务得分持续提升，证明各模块协同作用

#### （3）Exploration Ablation（图 10）
- **Entropy target $ \sigma_{\text{tgt}} $**：在 0.15–0.2 区间内鲁棒性强，支持统一设置
- **Noise Repetition**：启用后收敛更快、最终性能更高，验证时间相关探索的重要性

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Off-policy RL 可以既快又稳**：通过 scaling law + 显式稳定性控制，FlashSAC 打破了传统认知中 off-policy 方法“慢且不稳定”的刻板印象。
2. ✅ **大模型 + 少更新 是可行路径**：在足够大的数据吞吐下，减少梯度更新次数不会损害性能，反而有助于防止误差累积。
3. ✅ **架构级正则化至关重要**：weight/feature/gradient norm 控制是支撑大模型稳定训练的关键。
4. ✅ **sim-to-real 可分钟级完成**：在人形机器人控制中，FlashSAC 将训练时间从小时级压缩至分钟级，极大推动实际部署。

### 方法的局限性
- 当前主要面向 **state-based** 和 **vision-based** 输入，尚未扩展到多模态（如触觉 + 视觉）融合场景
- 虽然超参数统一性高，但在极端稀疏奖励任务中仍可能需要微调 exploration 参数
- 对极高频率物理仿真（>1kHz）的支持有待验证

### 未来工作方向
- 扩展至 **tactile-based reinforcement learning**
- 探索 **multi-modal input fusion**（vision + touch + proprioception）
- 应用于 **real-world online adaptation** 与 lifelong learning
- 结合 **offline RL** 与 demonstration data 进一步提升数据效率

---

> 📌 **一句话总结**：  
> **FlashSAC 通过“大模型、大批量、少更新 + 显式稳定性控制”，首次实现了快速、稳定且高性能的 off-policy 强化学习框架，在高维机器人控制与 sim-to-real 转移中展现出革命性潜力。**

</details>

---

### 16. [Do Domain-specific Experts exist in MoE-based LLMs?](https://arxiv.org/abs/2604.05267)

**Authors**: Giang Do, Hung Le, Truyen Tran  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.05267v1  

#### Abstract
In the era of Large Language Models (LLMs), the Mixture of Experts (MoE) architecture has emerged as an effective approach for training extremely large models with improved computational efficiency. This success builds upon extensive prior research aimed at enhancing expert specialization in MoE-bas...

---

### 17. [Towards Intelligent Energy Security: A Unified Spatio-Temporal and Graph Learning Framework for Scalable Electricity Theft Detection in Smart Grids](https://arxiv.org/abs/2604.03344)

**Authors**: AbdulQoyum A. Olowookere, Usman A. Oguntola, Ebenezer. Leke Odekanle, Maridiyah A. Madehin, Aisha A. Adesope  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.03344v1  

#### Abstract
Electricity theft and non-technical losses (NTLs) remain critical challenges in modern smart grids, causing significant economic losses and compromising grid reliability. This study introduces the SmartGuard Energy Intelligence System (SGEIS), an integrated artificial intelligence framework for elec...

---

### 18. [SLaB: Sparse-Lowrank-Binary Decomposition for Efficient Large Language Models](https://arxiv.org/abs/2604.04493)

**Authors**: Ziwei Li, Yuang Ma, Yi Kang  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.04493v1  

#### Abstract
The rapid growth of large language models (LLMs) presents significant deployment challenges due to their massive computational and memory demands. While model compression, such as network pruning, offers potential solutions, most existing methods often fail to maintain good performance at high compr...

---

### 19. [One Model for All: Multi-Objective Controllable Language Models](https://arxiv.org/abs/2604.04497)

**Authors**: Qiang He, Yucheng Yang, Tianyi Zhou, Meng Fang, Mykola Pechenizkiy, Setareh Maghsudi  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.04497v1  

#### Abstract
Aligning large language models (LLMs) with human preferences is critical for enhancing LLMs' safety, helpfulness, humor, faithfulness, etc. Current reinforcement learning from human feedback (RLHF) mainly focuses on a fixed reward learned from average human ratings, which may weaken the adaptability...

---

### 20. [Learning to Focus: CSI-Free Hierarchical MARL for Reconfigurable Reflectors](https://arxiv.org/abs/2604.05165)

**Authors**: Hieu Le, Mostafa Ibrahim, Oguz Bedir, Jian Tao, Sabit Ekin  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.05165v1  

#### Abstract
Reconfigurable Intelligent Surfaces (RIS) has a potential to engineer smart radio environments for next-generation millimeter-wave (mmWave) networks. However, the prohibitive computational overhead of Channel State Information (CSI) estimation and the dimensionality explosion inherent in centralized...

---

### 21. [COSMO-Agent: Tool-Augmented Agent for Closed-loop Optimization,Simulation,and Modeling Orchestration](https://arxiv.org/abs/2604.05547)

**Authors**: Liyuan Deng, Shujian Deng, Yongkang Chen, Yongkang Dai, Zhihang Zhong, Linyang Li, Xiao Sun, Yilei Shi, Huaxi Huang  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.05547v1  

#### Abstract
Iterative industrial design-simulation optimization is bottlenecked by the CAD-CAE semantic gap: translating simulation feedback into valid geometric edits under diverse, coupled constraints. To fill this gap, we propose COSMO-Agent (Closed-loop Optimization, Simulation, and Modeling Orchestration),...

---

### 22. [MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU](https://arxiv.org/abs/2604.05091)

**Authors**: Zhengqing Yuan, Hanchi Sun, Lichao Sun, Yanfang Ye  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.05091v1  

#### Abstract
We present MegaTrain, a memory-centric system that efficiently trains 100B+ parameter large language models at full precision on a single GPU. Unlike traditional GPU-centric systems, MegaTrain stores parameters and optimizer states in host memory (CPU memory) and treats GPUs as transient compute eng...

---

### 23. [Improving Feasibility via Fast Autoencoder-Based Projections](https://arxiv.org/abs/2604.03489)

**Authors**: Maria Chzhen, Priya L. Donti  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.03489v1  

#### Abstract
Enforcing complex (e.g., nonconvex) operational constraints is a critical challenge in real-world learning and control systems. However, existing methods struggle to efficiently enforce general classes of constraints. To address this, we propose a novel data-driven amortized approach that uses a tra...

---

### 24. [k-Maximum Inner Product Attention for Graph Transformers and the Expressive Power of GraphGPS The Expressive Power of GraphGPS](https://arxiv.org/abs/2604.03815)

**Authors**: Jonas De Schouwer, Haitz S\'aez de Oc\'ariz Borde, Xiaowen Dong  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.03815v1  

#### Abstract
Graph transformers have shown promise in overcoming limitations of traditional graph neural networks, such as oversquashing and difficulties in modelling long-range dependencies. However, their application to large-scale graphs is hindered by the quadratic memory and computational complexity of the ...

---

### 25. [Three Phases of Expert Routing: How Load Balance Evolves During Mixture-of-Experts Training](https://arxiv.org/abs/2604.04230)

**Authors**: Charafeddine Mouzouni  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.04230v1  

#### Abstract
We model Mixture-of-Experts (MoE) token routing as a congestion game with a single effective parameter, the congestion coefficient gamma_eff, that quantifies the balance-quality tradeoff. Tracking gamma_eff across training checkpoints of two open-source MoE models, OLMoE-1B-7B (20 checkpoints, with ...

---

### 26. [MUXQ: Mixed-to-Uniform Precision MatriX Quantization via Low-Rank Outlier Decomposition](https://arxiv.org/abs/2604.04701)

**Authors**: Seoungsub Lee, In Seo Kim, Seon Wook Kim  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.04701v1  

#### Abstract
Large language models (LLMs) have achieved outstanding performance across a wide range of natural language processing tasks, but their enormous parameter counts impose ubstantial memory and computational overheads. This challenge is particularly critical in NPU-based on-device environments, where FP...

---

### 27. [Forgetting to Witness: Efficient Federated Unlearning and Its Visible Evaluation](https://arxiv.org/abs/2604.04800)

**Authors**: Houzhe Wang, Xiaojie Zhu, Chi Chen  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.04800v1  

#### Abstract
With the increasing importance of data privacy and security, federated unlearning has emerged as a novel research field dedicated to ensuring that federated learning models no longer retain or leak relevant information once specific data has been deleted. In this paper, to the best of our knowledge,...

---

### 28. [Just Pass Twice: Efficient Token Classification with LLMs for Zero-Shot NER](https://arxiv.org/abs/2604.05158)

**Authors**: Ahmed Ewais, Ahmed Hashish, Amr Ali  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.05158v1  

#### Abstract
Large language models encode extensive world knowledge valuable for zero-shot named entity recognition. However, their causal attention mechanism, where tokens attend only to preceding context, prevents effective token classification when disambiguation requires future context. Existing approaches u...

---

### 29. [Multi-Drafter Speculative Decoding with Alignment Feedback](https://arxiv.org/abs/2604.05417)

**Authors**: Taehyeon Kim, Hojung Jung, Se-Young Yun  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.05417v1  

#### Abstract
Speculative decoding (SD) accelerates large language model (LLM) inference by using a smaller model to draft future tokens, which are then verified by the target LLM. This preserves generation quality by accepting only aligned tokens. However, individual drafters, often trained for specific tasks or...

---

### 30. [DRAFT: Task Decoupled Latent Reasoning for Agent Safety](https://arxiv.org/abs/2604.03242)

**Authors**: Lin Wang, Junfeng Fang, Dan Zhang, Fei Shen, Xiang Wang, Tat-Seng Chua  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.03242v1  

#### Abstract
The advent of tool-using LLM agents shifts safety monitoring from output moderation to auditing long, noisy interaction trajectories, where risk-critical evidence is sparse-making standard binary supervision poorly suited for credit assignment. To address this, we propose DRAFT (Task Decoupled Laten...

---

## 🔧 Configuration

This bot is configured to look for papers containing the following keywords:
- LLM, RL, RLHF, Inference, Training, Attention, Pipeline, MOE, Sparse, Quantization, Speculative, Efficient, Efficiency, Framework, Parallel, Distributed, Kernel, Decode, Decoding, Prefill, Throughput, Fast, Network, Hardware, Cluster, FP8, FP4, Optimization, Scalable, Communication

## 📅 Schedule

The bot runs daily at 12:00 UTC via GitHub Actions to fetch the latest papers.

## 🚀 How to Use

1. **Fork this repository** to your GitHub account
2. **Customize the configuration** by editing `config.json`:
   - Add/remove arXiv categories (e.g., `cs.AI`, `cs.LG`, `cs.CL`)
   - Modify keywords to match your research interests
   - Adjust `max_papers` and `days_back` settings
3. **Enable GitHub Actions** in your repository settings
4. **The bot will automatically run daily** and update the README.md

## 📝 Customization

### arXiv Categories
Common categories include:
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning
- `cs.CL` - Computation and Language
- `cs.CV` - Computer Vision
- `cs.NE` - Neural and Evolutionary Computing
- `stat.ML` - Machine Learning (Statistics)

### Keywords
Add keywords that match your research interests. The bot will search for these terms in paper titles and abstracts.

### Exclude Keywords
Add terms to exclude certain types of papers (e.g., "survey", "review", "tutorial").

## 🔍 Manual Trigger

You can manually trigger the bot by:
1. Going to the "Actions" tab in your repository
2. Selecting "arXiv Bot Daily Update"
3. Clicking "Run workflow"

---
*Generated automatically by arXiv Bot* 
