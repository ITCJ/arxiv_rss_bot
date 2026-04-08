# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-08 07:03:53 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs](https://arxiv.org/abs/2604.05650)

**Authors**: Yicheng Ji, Jun Zhang, Jinpeng Chen, Cong Wang, Lidan Shou, Gang Chen, Huan Li  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2604.05650v1  

#### Abstract
Video Large Language Models (Video-LLMs) excel in video understanding but suffer from high inference latency during autoregressive generation. Speculative Decoding (SD) mitigates this by applying a draft-and-verify paradigm, yet existing methods are constrained by rigid exact-match rules, severely l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

- **高推理延迟瓶颈**：Video Large Language Models (Video-LLMs) 在视频理解任务中表现出色，但由于其自回归生成机制，在处理长序列视觉 token 时面临严重的推理延迟问题。
- **现有 Speculative Decoding (SD) 的局限性**：当前主流的 SD 方法依赖于严格的“精确匹配”（exact-match）验证规则，导致大量语义上等价但字面不一致的 draft token 被错误拒绝，限制了加速潜力。

### **提出了什么新方法或新思路**

提出 **LVSPEC** —— 首个专为 Video-LLMs 设计的、无需训练的 **Loosely Speculative Decoding** 框架，其核心思想是：

> **并非所有生成 token 对视觉理解都同等重要，因此不应采用统一的严格验证标准。**

具体创新点如下：

- ✅ **视觉相关性感知的松散验证机制**：
  - 引入轻量级的 **Visual-Relevant Text Token Identification** 机制，通过计算文本隐藏状态与视觉嵌入之间的余弦相似度，识别出对视觉理解至关重要的“锚点 token”（如颜色、物体名称）。
  - 对这些 **Visual-Relevant Tokens** 进行 **Strict Verification**（严格验证），确保关键信息准确无误。
  - 对 **Visual-Irrelevant Tokens**（如语法词、连接词）进行 **Loose Verification**（宽松接受），提升接受率。

- ✅ **位置偏移容忍机制 (Position Shift-Tolerant, PST)**：
  - 允许在 draft 序列附近窗口内出现的 mismatch token 被接受，缓解因描述顺序不同或插入短语导致的位置错位问题。

- ✅ **理论支撑**：
  - 提出 **Visual Anchor Sparsity Hypothesis**：生成质量由稀疏的关键视觉 token 决定，而非密集的语言填充词。
  - 理论证明：利用该稀疏性可将有效失败率从 $e$ 降低至 $pe$（$p$ 为视觉密度），从而突破传统 SD 的加速上限。

### **相比现有方法的优势**

| 维度 | LVSPEC | 传统 SD / FLY 类方法 |
|------|--------|------------------|
| **验证策略** | 视觉语义引导的差异化验证 | 统一规则（精确匹配 或 语言熵启发） |
| **适用性** | 专为 Video-LLMs 设计，解决“视觉盲区”问题 | 主要针对纯文本 LLMs，忽略视觉重要性差异 |
| **效率** | 显著提升平均接受长度 $T$ 和加速比 | 受限于 draft 模型原始对齐精度 |
| **保真度** | >99.8% 性能保留，关键信息不丢失 | 松散策略可能导致事实性错误（尤其在视觉描述中） |

---

## 2. 核心实验方法和设置

### **使用的数据集**

在四个视频理解基准上进行全面评估：

| 数据集 | 任务类型 | 输出长度 | 帧数 | 实例数 |
|-------|---------|----------|------|--------|
| **VDC** (Video Detail Caption) | 视频详细描述 | 长 | 64 | 120 |
| **VDD** (Video Detail Description) | 视频整体描述 | 长 | 128 | 30 |
| **MovieChat** | 开放式视频问答 | 中 | 128 | 100 |
| **Video-MME** | 多项选择题问答 | 短 | 64 | 500 |

> 所有评估均使用 **LMMs-Eval** 框架，并以大模型作为 judge 进行评分。

---

### **实验设置和评估指标**

#### **目标模型与 draft 模型**

- **Target Models**：
  - `Qwen2.5-VL-32B`
  - `LLaVA-OneVision-72B`

- **Draft Models**：
  - 同系列小模型（如 `Qwen2.5-VL-7B`）
  - 应用 **video token pruning** 以减少 KV cache 开销

#### **两种 SD 设置**

1. **Standard-SD (Std.-SD)**：使用更小的同系列模型作为 draft model。
2. **Self-SD**：使用同一模型但剪枝后的版本作为 draft model。

#### **评估指标**

| 类别 | 指标 |
|------|------|
| **效率** | Speedup Ratio, Mean Accepted Length $T$, Tokens/s |
| **性能** | Rating Score, Accuracy (Acc%), Retention (%) |

---

### **基线方法对比**

| 类别 | 方法 | 描述 |
|------|------|------|
| **Lossless SD** | NAIVE SD | 直接使用小模型作为 draft model |
| | SPECVLM (Ji et al., 2025) | 当前 SOTA 的训练免费 SD 方法，采用 uniform video token pruning |
| **Loosely SD** | FLY (Li et al., 2025b) | 基于熵和延迟窗口的训练免费松散解码 |
| | FLY⁺ | 仅使用熵门控的更宽松变体 |

> 所有 loosely SD 方法均使用相同的 draft 结构（chain-like, $K=10$）以公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 方法 | 模型设置 | Speedup | Mean $T$ | 性能保留 |
|------|----------|---------|----------|----------|
| **LVSPEC (Ours)** | Std.-SD-Qwen2.5-VL | **2.70×** | **7.76** | >99.8% |
| **LVSPEC (Ours)** | Std.-SD-LLaVA-OV | **2.94×** | **7.34** | >99.8% |
| SPECVLM | Std.-SD-Qwen2.5-VL | 2.00× | 3.29 | 100% |
| FLY | Std.-SD-Qwen2.5-VL | 2.32× | 6.81 | ~97% |

> 🔺 **LVSPEC 将平均接受长度 $T$ 提升了 136%，加速比提升 35%**，显著优于当前 SOTA 训练免费 SD 方法。

---

### **与基线方法的对比结果**

- ✅ **速度优势明显**：
  - 在 `Qwen2.5-VL-32B` 上实现 **2.70× 加速**，在 `LLaVA-OneVision-72B` 上达到 **2.94×**。
  - 相比 SPECVLM，$T$ 提升超过 **2.15 倍**。

- ✅ **性能高度保真**：
  - 平均性能保留率达 **>99.8%**，几乎无损。
  - 在多项选择任务上完全 lossless，在开放生成任务上保留原模型绝大部分语义完整性。

- ✅ **超越通用松散策略**：
  - FLY 等基于语言熵的方法在视频任务上表现不稳定，且可能引入事实错误。
  - LVSPEC 因结合视觉信号，能在更高松弛度下保持稳定性。

---

### **消融实验结果**

#### **(1) PST 机制的作用（Table 2）**

| 方法 | $T$ | Speedup | Retention |
|------|-----|--------|----------|
| LVSPEC (w/o PST) | 7.27 | 2.54× | 98.5% |
| **LVSPEC (w/ PST)** | **7.76** | **2.70×** | **99.6%** |

> ✅ PST 在不损害准确性的前提下，进一步提升了接受长度和加速比。

#### **(2) 超参数敏感性分析（Table 3）**

- **松弛参数 $\lambda$**：
  - $\lambda = 0.7$ 时取得最佳平衡：$T=7.27$, Speedup=2.54×, Retention=98.5%
  - $\lambda = 0.9$ 时加速更高（3.03×），但性能下降至 92.8%

- **关键视觉 token 数量 $N$**：
  - $N=10$ 为最优选择，过多或过少都会降低判别能力。

#### **(3) 与其他策略对比（Figure 7）**

- LVSPEC 明显推动了 **Accuracy-Speedup Pareto Frontier**，优于：
  - FLY（熵驱动）
  - Random Relaxation（随机松弛）
  - Lossless SD（保守但慢）

---

## 4. 关键结论和发现

### **主要发现**

1. 🎯 **视觉语义具有稀疏性与关键性**：
   - 仅有约 **15–20%** 的 token 是视觉相关的，但它们决定了生成质量；移除它们会导致性能崩溃。

2. 🧩 **验证应区别对待**：
   - 对视觉无关 token 施加严格匹配是计算浪费，而对视觉相关 token 放宽则是危险的。

3. ⚙️ **LVSPEC 成功打破“精确匹配”瓶颈**：
   - 通过视觉语义引导的松散验证 + 位置偏移容忍，实现了高效且可靠的加速。

4. 📈 **理论与实践一致**：
   - 实验观测到的失败率稀释效果（从 ~66% 到 ~22%）与理论预测（$e \to pe$）高度吻合。

---

### **方法的局限性**

1. **超参数需调优**：
   - $\lambda$ 和 $N$ 需根据任务和模型调整，在极端视觉密度任务中可能需要重新校准。

2. **复杂逻辑推理场景未覆盖**：
   - 对于涉及否定词（如 "not"）、逻辑连接词等“视觉无关但逻辑关键”的 token，当前机制可能过于宽松。

3. **PST 是辅助模块**：
   - 仅适用于存在位置偏移的少数情况，不能作为主要加速来源。

4. **未整合 fine-tuned draft model**：
   - 当前 focus 在验证机制改进，未来可结合专门训练的 draft model 进一步提升性能。

---

### **未来工作方向**

- 探索 **adaptive $\lambda$** 策略，根据输入动态调整松弛程度。
- 引入 **逻辑敏感性检测**，避免对关键非视觉 token 进行过度放松。
- 结合 **KV cache compression** 与 **in-context draft capability** 实现端到端优化。
- 扩展至 **complex video reasoning**、**long video summarization** 等更具挑战的任务。
- 探索 **parallel speculative decoding** 架构以支持更大批量推理。

--- 

> 💡 **一句话总结**：  
> LVSPEC 首次将“视觉语义引导”引入 Speculative Decoding，提出“看森林而非树木”的理念——只对关键视觉 token 严加看管，其余则宽松放行，从而在近乎无损的前提下，将 Video-LLMs 的推理速度提升近 **3 倍**，为高效视频理解提供了实用化路径。

</details>

---

### 2. [BWTA: Accurate and Efficient Binarized Transformer by Algorithm-Hardware Co-design](https://arxiv.org/abs/2604.03957)

**Authors**: Yifu Ding, Xianglong Liu, Shenghao Jin, Jinyang Guo, Jiwen Lu  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 14.0  
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
当前基于Transformer的模型（如BERT、LLMs）面临两大挑战：
- **精度瓶颈**：超低位宽（ultra low-bit）量化（尤其是二值化）会严重损害模型表示能力，导致训练困难和显著的准确率下降。
- **硬件支持缺失**：尽管GPU是主流计算平台，但现有的低位宽（如1-bit）算子缺乏高效的GPU内核支持，多数方案依赖定制硬件（如FPGA/ASIC），难以在通用GPU上部署。

此外，作者观察到**零点失真（zeropoint distortion）** 是二值化中的关键问题：当激活值分布集中在零附近时，强制将其映射为±1会导致大量小值被错误放大，破坏原始分布，引入巨大量化误差。

### 提出的新方法与创新
为解决上述问题，本文提出 **Binary Weights & Ternary Activations (BWTA)** 框架，结合算法-硬件协同设计（algorithm-hardware co-design），实现高效且高精度的极低比特Transformer推理。

#### 主要创新点：
1. **BWTA量化方案**  
   - **权重二值化（Binary Weights）**：使用 `sign` 函数将权重压缩至1-bit。
   - **激活三值化（Ternary Activations）**：对非负激活（如ReLU输出）使用 `bool` 函数（0/1），对服从高斯分布的激活使用 `ternary` 函数（-1/0/1），保留零点以缓解零点失真问题。
   - 该设计在保持极致压缩的同时，有效维持了激活的统计特性。

2. **Smooth Multi-Stage Quantization（平滑多阶段量化）训练框架**
   - **Levelwise Degradation Strategy**：逐步减少激活的量化等级数（如从17级 → 15 → 13 → … → 3），避免传统“指数衰减”策略带来的剧烈容量跳跃，实现更平滑的收敛。
   - **Magnitude Alignment Projection Factor**：在阶段切换时引入一个基于前后阶段激活幅值比的投影因子，用于调整缩放因子（scaling factor），防止数值突变，提升稳定性。

3. **定制化的BWTA MatMul CUDA Kernel**
   - **Instruction-Level Parallel Bitpack**：在运行时并行地将FP16张量打包为二进制/三进制位流，支持实时量化。
   - **Comprehensive MatMul Implementation**：针对不同操作（Linear、Attention Score、Value Projection）设计专用的低位宽矩阵乘法内核，利用 `xor`, `and`, `popcount` 等位运算指令实现高效计算。
   - 支持完整的Transformer架构（包括MHA和FFN），可无缝集成。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **精度** | 显著优于传统二值化方法（如BiBERT、BiT），接近全精度模型性能。 |
| **效率** | 在NVIDIA GPU上实现16–24×的内核级加速，端到端吞吐量提升显著。 |
| **兼容性** | 可广泛应用于各类Transformer架构（BERT、LLM等），无需修改模型结构。 |
| **实用性** | 首次在通用GPU上实现了**权重+激活均为极低比特**的完整推理系统，填补了软硬件之间的空白。 |

---

## 2. 核心实验方法和设置

### 数据集
- **BERT类模型**：在 **GLUE benchmark** 上进行评估，包含9个自然语言理解任务（MNLI, QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE），排除WNLI。
- **LLM类模型**：
  - **Perplexity评估**：Wikitext-2 和 C4 数据集。
  - **准确性评估**：CommonsenseQA 基准。

### 实验设置与评估指标
| 类型 | 设置 |
|------|------|
| **模型** | 
  - BERT：基于DynaBERT蒸馏训练。
  - LLM：基于Bitnet系列预训练权重（0.7B, 1.3B, 3B）微调。
| **量化配置** | 权重1-bit，激活平均1.5-bit（即ternary），部分层替换为BWTA模块（如10%/30%）。 |
| **训练策略** | 多阶段量化 + 知识蒸馏（中间层、logits、attention概率）。 |
| **评估指标** |
  - BERT：各任务准确率 / Pearson相关系数，以及平均得分。
  - LLM：Perplexity（越低越好）、CommonsenseQA准确率。
  - 效率：内核延迟（μs）、TFLOPs、端到端吞吐量（tokens/s）、显存占用（GB）。 |

### 基线方法对比
| 类别 | 对比方法 |
|------|----------|
| **BERT量化** | Q2BERT, TernaryBERT, BinaryBERT, BiBERT, BiT, BEBERT, BiPFT, MLBERT |
| **LLM量化** | RTN, GPTQ, PB-LLM, BiLLM, Bitnet, BWN |
| **工具链** | bitsandbytes (bnb-4bit), AutoGPTQ, vLLM, TensorRT-LLM |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ BERT在GLUE上的表现（Table 1）
- **平均准确率**：BWTA达到 **80.4%**，显著高于其他1-bit方法（如BiT: 77.5%，BE-BERT: 74.7%）。
- **与全精度差距**：仅比全精度低约 **3.5%**，在多个任务（如SST-2, QQP, QNLI）上差距小于2%。
- **存储开销**：与其它1-bit方法相同（仅13.4MB），无额外负担。

#### ✅ LLM在CommonsenseQA上的表现（Table 2 & 9）
- 在Bitnet-b1.58-1.3B上，使用30% BWTA层时，准确率达到 **44.8%**，优于同配置下的BWN（42.5%）和Bitnet原生int8xint2（44.4%）。
- 表明即使在更低平均bitwidth下（1.3/6），仍能保持竞争力。

#### ✅ 内核级效率（Table 3 & Fig. 11）
- 在NVIDIA H800 GPU上测试：
  - **Linear层**：BWTA相比FP16提速 **23.6×**。
  - **Attention层**：提速 **16.7–18.5×**。
  - **最大加速比可达50×**（大矩阵场景）。
- **分解分析**：
  - 数据传输（Data Trans.）略快于FP16。
  - Bitpack耗时极少，几乎无额外开销。
  - MMA Core（核心计算）是主要加速来源，得益于低位宽指令和更高并行度。

#### ✅ 端到端推理速度（Table 4）
| 场景 | 性能 |
|------|------|
| **Prefill阶段**（batch=16, seq_len=2k） | 达到 **216–330 tokens/s**，远超bnb-4bit和AutoGPTQ（后者在batch≥8时OOM）。 |
| **Decode阶段**（生成长度~150） | 吞吐达 **12–15 tokens/s**，优于Bitnet-b1.58。 |
| **显存占用** | 显著低于4-bit方法，例如Gemma-2B从5.0GB降至~1.8GB。 |

### 消融实验结果

#### 🔍 多阶段策略对比（Fig. 9）
- **Levelwise vs Bitwise**：Levelwise策略在阶段转换时损失上升更小，恢复更快，验证其平滑性。
- **加入Projection Factor后**：进一步降低跳变幅度，加快收敛。

#### 🔍 缩放因子初始化策略（Fig. 10）
- “Search” 和 “Mean” 初始化效果差，需长时间恢复。
- **Projection Factor** 能使缩放因子起点更接近最优，实现快速稳定收敛。

#### 🔍 激活可视化（Fig. 12）
- Bitwise策略导致量化桶未充分利用，分布不平衡。
- Levelwise策略能更好保留原始分布形态，尤其在零点附近。

#### 🔍 损失曲面分析（Fig. 13）
- BWTA模型的损失曲面更平滑，标准差更低（0.091 vs BiT的0.117），说明其对扰动更鲁棒，收敛质量更高。

---

## 4. 关键结论和发现

### 主要发现
1. **零点失真是二值化性能退化的重要原因**，通过引入**三值激活**可有效缓解。
2. **平滑的多阶段量化策略**（Levelwise Degradation + Magnitude Alignment）对于超低位宽模型的稳定训练至关重要。
3. **算法-硬件协同设计是释放极低比特潜力的关键**：仅靠算法改进无法实现实质性加速，必须配合底层CUDA优化才能发挥位运算优势。
4. BWTA首次在**通用GPU上实现了权重+激活均极低比特的高效推理系统**，兼具高精度与高性能。

### 方法的局限性
1. **非线性函数仍使用高精度**：Softmax、LayerNorm、GELU等仍用BF16/FP32，限制了进一步压缩空间。
2. **目前主要用于推理**：训练成本虽降低但仍较高，尚未完全适配大规模分布式训练。
3. **依赖特定GPU架构**：高度优化的kernel可能在不同GPU型号上表现不一致。
4. **仅部分替换有效**：实验中通常只替换30%左右的层，全部替换可能导致性能崩溃。

### 未来工作方向
1. **扩展至更多非线性函数的量化**：探索Softmax、LayerNorm等的整数量化方案（参考SOLE、SoftmAP）。
2. **适配更多硬件平台**：将BWTA思想迁移到FPGA、ASIC等专用芯片，追求极致能效。
3. **全自动层选择机制**：开发更智能的敏感度分析方法，自动决定哪些层适合替换为BWTA。
4. **支持更大规模LLM**：在10B+级别模型上验证BWTA的可扩展性和稳定性。
5. **开放生态建设**：提供PyTorch/TensorFlow插件，便于社区集成与应用。

---

> **总结一句话**：  
> BWTA通过**算法-硬件协同设计**，提出了首个在通用GPU上可行的**权重1-bit + 激活1.5-bit** Transformer推理方案，在几乎不牺牲精度的前提下，实现了高达 **24× 的内核加速** 和 **330 tokens/s 的端到端吞吐**，为超低位宽AI模型的实际落地铺平了道路。

</details>

---

### 3. [HybridKV: Hybrid KV Cache Compression for Efficient Multimodal Large Language Model Inference](https://arxiv.org/abs/2604.05887)

**Authors**: Bowen Zeng, Feiyang Ren, Jun Zhang, Xiaoling Gu, Ke Chen, Lidan Shou, Huan Li  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2604.05887v1  

#### Abstract
Multimodal Large Language Models (MLLMs) have advanced unified reasoning over text, images, and videos, but their inference is hindered by the rapid growth of key-value (KV) caches. Each visual input expands into thousands of tokens, causing caches to scale linearly with context length and remain re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：HYBRIDKV: Hybrid KV Cache Compression for Efficient Multimodal Large Language Model Inference**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
Multimodal Large Language Models (MLLMs) 在处理图像、视频等视觉输入时，会将每个视觉帧展开为数百甚至数千个 **visual tokens**，导致上下文长度急剧膨胀。这直接引发 **Key-Value (KV) Cache** 的线性增长，这些缓存必须在 GPU 内存中全程保留以支持自回归解码。

这一现象带来了两大瓶颈：
- **巨大的 GPU 内存开销**：例如，一个 72B 模型处理 20 张图片即可超过 13GB 缓存。
- **严重的解码延迟**：频繁的内存访问拖慢推理速度。

尽管已有多种 KV Cache 压缩方法（如 token-level、layer-level、head-level），但它们大多仅关注“**预算分配**”（budget allocation），即决定保留多少 token 或分配给哪一层/头，却忽略了不同 **attention head** 具有本质不同的行为模式，需要**差异化的压缩策略**。

---

### 🚀 **提出了什么新方法或新思路**
本文提出 **HYBRIDKV** ——一种基于注意力头异质性的混合 KV Cache 压缩框架，其核心思想是：

> **不仅要合理分配缓存预算，更要根据 head 的行为特性选择合适的压缩策略。**

该方法分为三个阶段：

#### （1）**Head 分类：静态 vs 动态**
通过分析 **prefill 阶段的 text-centric attention sparsity** 来预测每个 attention head 在 decoding 阶段的行为：
- **Static Heads**：注意力集中在少数稳定 token 上 → 可安全剪枝（pruning）
- **Dynamic Heads**：注意力随时间动态变化 → 必须支持运行时检索（retrieval）

分类依据是一个可学习的 **text-centric sparsity score**，具有任务感知能力。

#### （2）**分层预算分配（Top-down Budget Allocation）**
采用两级分配机制：
- **Level 1: Head Type Level**  
  根据 `r` 系数在 static 和 dynamic heads 之间分配总预算。
- **Level 2: Individual Head Level**  
  - 对 static heads：按 sparsity score 自适应分配（越重要保留越多）
  - 对 dynamic heads：均匀分配 + 对齐 CUDA 块大小

#### （3）**混合压缩策略（Hybrid Compression）**
针对不同类型 heads 应用不同策略：
- **Static Heads**：使用 **text-prior pruning**，优先保留文本和局部显著视觉 token。
- **Dynamic Heads**：采用 **chunk-wise retrieval**，将 KV 缓存卸载到 CPU，在解码时按需加载关键 chunk。

最终所有重要 token 被统一维护在一个小型 GPU KV Buffer 中。

---

### 🔍 **相比现有方法的优势**
| 维度 | 现有方法（如 SNAPKV, MADAKV, SPARSEMM） | HYBRIDKV |
|------|----------------------------------------|----------|
| **粒度** | 多为 token/layer/head 级别预算分配 | 细致到 head 类型识别与策略定制 |
| **策略灵活性** | 统一使用 pruning 或 retrieval | 混合使用 pruning + retrieval，按需适配 |
| **上下文感知** | 多依赖固定规则或离线评分 | 利用 text-centric attention 实现在线、任务感知分类 |
| **效率-精度平衡** | 极端压缩下性能下降明显 | 即使仅保留 10% KV Cache，仍接近甚至超越 full cache 性能 |

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
涵盖图像与视频两类多模态任务：

#### 图像任务（来自 MileBench）：
- **CL-CH**（CLEVR-Change）：视觉差异描述
- **STD**（Spot-the-Diff）：找图差异
- **DocVQA / SlideVQA**：文档理解
- **WebQA / WikiVQA / MMCoQA / MM-QA**：跨模态问答

#### 视频任务：
- **VATEX**：多语言视频描述生成
- **NextQA**：基于视频的因果与时序推理
- **Video-ChatGPT**：长视频对话理解（含 CI, DO, CU, TU, CO 五个维度评分）

---

### ⚙️ **实验设置与评估指标**

| 项目 | 设置说明 |
|------|---------|
| **模型** | Qwen2.5-VL-3B/7B, LLaVA-OneVision-7B |
| **硬件** | NVIDIA L40S GPU |
| **KV Cache 预算** | 主要测试 10% 和 20%，极端情况低至 5% |
| **评估方式** | 平均多个 decoding steps 的 cumulative focus count |
| **主要指标** | - 图像任务：Exact Match Accuracy, ROUGE-L<br>- 视频任务：BLEU-4, METEOR, ROUGE-L, CIDEr, WUPS, GPT Score（LLM-as-a-judge）<br>- 效率指标：GPU Memory (GB), Latency (ms/token) |

---

### 🆚 **基线方法对比**
| 方法 | 类型 | 特点 |
|------|------|------|
| **SNAPKV** | Token-level | 基于累计 attention score 保留关键 token |
| **LOOK-M** | Token-level | 优先保留 text token，合并 KV |
| **MADAKV** | Layer-level | 按模态偏好分配各层预算 |
| **SPARSEMM** | Head-level | 使用离线 visual head score 进行非对称分配 |
| **FULL CACHE** | 上限基准 | 不进行任何压缩 |

> 所有 baseline 均未结合 head 行为分析与差异化压缩策略。

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据**

#### ✅ **压缩比与加速效果（Table 3）**
在 **Qwen2.5-VL-7B + Video-ChatGPT** 上的结果：

| 方法 | KV Cache 预算 | GPU Memory | Latency (ms/token) | 加速比 |
|------|----------------|-------------|---------------------|--------|
| FULL CACHE | 100% | 1.73 GB | 58.94 | 1× |
| HYBRIDKV | 20% | 0.40 GB | 42.08 | 1.40× |
| **HYBRIDKV** | **10%** | **0.22 GB** | **38.65** | **1.52×** |

👉 **实现最高达 7.9× 的 GPU 内存节省**，同时提升解码速度。

---

#### ✅ **准确率表现（Tables 1 & 2）**

##### 图像任务平均准确率（Table 1）：
| 方法 | 平均得分（Qwen2.5-VL-7B） |
|------|----------------------------|
| FULL CACHE | 67.53 |
| SNAPKV | 63.74 |
| LOOK-M | 63.61 |
| MADAKV | 63.59 |
| SPARSEMM | 64.86 |
| **HYBRIDKV** | **66.68**（仅比 full cache 低 0.85%）|

✅ 在 10% 缓存下保持几乎无损性能。

##### 视频任务平均得分（Table 2）：
| 方法 | VATEX (Avg) | NextQA (Avg) | Video-ChatGPT (Avg) |
|------|--------------|---------------|-----------------------|
| FULL CACHE | 0.3321 | 33.79 | 2.93 |
| SPARSEMM | 0.3169 | 33.84 | 2.75 |
| **HYBRIDKV** | **0.3399** | **33.86** | **2.85** |

👉 **在 VATEX 上甚至超过了 full cache 基线！**

原因在于：HYBRIDKV 成功过滤噪声 token，聚焦于最相关的视觉区域，反而提升了生成质量。

---

### 🔬 **消融实验结果（Ablation Studies）**

#### （1）Head 分类的重要性（Table 4）
| 方法 | STD | WebQA | NextQA | Latency |
|------|-----|--------|--------|--------|
| w/ all static heads | 28.72 | 73.00 | 33.61 | 34.13 |
| w/ all dynamic heads | 29.77 | 75.50 | 33.76 | **48.15**↑ |
| **HYBRIDKV（完整）** | **29.84** | **76.00** | **33.86** | **38.65** |

- 移除分类会导致性能下降；
- 全设为 dynamic 导致 I/O 开销大增，延迟显著上升。

#### （2）预算分配机制的作用（Table 5）
| 方法 | STD | WebQA | NextQA | Latency |
|------|-----|--------|--------|--------|
| w/o H（无 head-level 分配） | 29.67 | 76.00 | 33.70 | 38.65 |
| w/o (H&HT)（无 head-type 分配） | 29.58 | 74.50 | 33.29 | **44.67**↑ |
| **HYBRIDKV** | **29.84** | **76.00** | **33.86** | **38.65** |

- 分层分配对 accuracy 和 efficiency 均有正向影响。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **Attention Heads 存在显著异质性**：
   - 可明确划分为 **static** 与 **dynamic** 两种模式。
   - 二者在 prefill 阶段的 **text-centric attention sparsity** 可作为可靠判据。

2. **任务依赖性强**：
   - 同一头在不同任务中可能表现出不同行为（见 Fig. 3），因此必须设计**上下文感知**的分类机制。

3. **混合策略优于单一策略**：
   - 单纯 pruning 或 retrieval 都无法兼顾效率与精度。
   - HYBRIDKV 通过 **pruning + retrieval 结合 + 分类引导**，实现了最优权衡。

4. **高压缩比下仍可提效增益**：
   - 在仅 10% KV Cache 下，不仅没有损失性能，有时还能**超越 full cache**，说明原始缓存中存在大量冗余或噪声。

---

### ⚠️ **方法的局限性**
1. **分类依赖阈值 `θ`**：
   - 当前使用固定阈值（经验证设为 0.9 最优），缺乏自适应调整能力。
   - 不同模型规模或领域可能需重新调参。

2. **未参与训练过程**：
   - 完全为 inference-time 方法，未联合优化压缩策略与模型参数。
   - 若能在训练中引入压缩感知，可能进一步增强鲁棒性。

3. **主要验证于图像/视频任务**：
   - 尚未扩展至音频或多轮超长文本等其他模态场景。

---

### 🔮 **未来工作方向**
- 引入 **learnable classifier** 替代手工阈值，实现动态 head 分类。
- 探索 **joint training + compression co-design**，让模型更适应压缩环境。
- 扩展至 **audio-grounded MLLMs** 或 **extremely long-context reasoning** 场景。
- 结合其他技术如 **quantization**, **speculative decoding**, **attention sink** 等形成综合优化方案。

---

## ✅ 总结一句话
> **HYBRIDKV 是首个基于 attention head 异质性设计的混合 KV Cache 压缩框架，通过“分类 + 分配 + 差异化压缩”三步走，在仅保留 10% 缓存的情况下实现高达 7.9× 内存节省和 1.52× 解码加速，且性能不降反升，为高效 MLLM 推理提供了新范式。**

</details>

---

### 4. [Diagonal-Tiled Mixed-Precision Attention for Efficient Low-Bit MXFP Inference](https://arxiv.org/abs/2604.03950)

**Authors**: Yifu Ding, Xinhao Zhang, Jinyang Guo  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 13.5  
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
Transformer 模型中的 **self-attention** 由于其 $O(n^2)$ 的序列长度复杂度，成为大模型推理的主要瓶颈。尽管低比特量化（如 MXFP4、MXFP8）能显著降低计算和内存开销，但直接应用会导致严重的 **accuracy degradation** 和 **quantization error**。此外，传统量化流程中分离的操作（如量化、格式转换）引入大量冗余内存访问和 kernel launch 开销。

本文针对以下两个核心挑战提出解决方案：
- **Challenge 1**: 低比特 MXFP 格式（尤其是 MXFP4）在 attention score 计算中引入显著量化误差。
- **Challenge 2**: 非融合的量化操作破坏了低比特计算的整体效率。

---

### 🚀 提出的新方法：Diagonal-Tiled Mixed-Precision Attention (DMA)

DMA 是首个专为混合 MXFP 格式设计的端到端 attention 工作流，包含两大核心技术：

#### （1）对角分块混合精度计算（Diagonal-Tiled Mixed-Precision）
- 将 attention 矩阵划分为高精度与低精度区域。
- **关键思想**：attention 中沿对角线附近的得分最为敏感（如因果注意力中的自回归依赖），因此保留这些区域使用高精度（如 MXFP8 或 FP16）表示。
- 其余外围区域采用高效低比特格式（如 MXFP4）进行计算。
- 引入可调参数 `T`（diagonal window size）控制高精度区域大小，在精度与速度间灵活权衡。

#### （2）全融合量化内核（Fully Fused Quantization Kernel）
- 在一个 Triton 实现的 kernel 中集成：
  - FP16 → MXFP 的量化
  - 微缩放（microscaling）变换
  - 低比特编码与打包（如两个 FP4 合并为一个 UINT8）
  - attention 计算（含 OnlineSoftmax）
- 避免中间结果写回显存，极大减少 memory traffic 和 kernel launch overhead。

---

### 🔍 相比现有方法的优势

| 对比维度 | 现有方法（如 SageAttention、INT-FlashAttention） | 本文 DMA 方法 |
|--------|---------------------------------------------|--------------|
| **精度保持** | 多数全图低比特量化，易损失关键信息 | 选择性保留对角高精度，有效维持生成质量 |
| **量化策略** | 通常 token-wise 或 block-wise 单一粒度 | 支持 per-token 更细粒度量化，提升保真度 |
| **实现方式** | 多阶段非融合操作，存在额外开销 | 完全融合 kernel，消除冗余访存 |
| **硬件适配** | 不完全利用 Blackwell 架构原生 MXFP 支持 | 充分利用 MXFP8/MXFP4/NVFP4 原生加速能力 |

> ✅ **优势总结**：DMA 实现了 **lossless generation quality + 显著加速** 的双重目标。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 主要评测任务：**LongBench**  
  - 覆盖长上下文理解场景（sequence length: 2.5K ~ 30K tokens）
  - 包括多语言问答、摘要、检索等子任务（如 `repobench-p`, `samsum`, `triviaqa`, `passage_retrieval` 等）

### ⚙️ 模型与硬件平台
- **模型**：
  - LLaMA-3.1-8B
  - LLaMA-3.2-3B
- **硬件**：
  - 单张 **NVIDIA B200 GPU**（Blackwell 架构，原生支持 MXFP）
- **实现工具**：
  - 使用 **Triton** 编写高性能 CUDA-like kernel
  - 代码已开源：[GitHub - yifu-ding/MP-Sparse-Attn](https://github.com/yifu-ding/MP-Sparse-Attn)

### 📊 评估指标
| 类别 | 指标 |
|------|------|
| **准确性** | LongBench 各项任务得分、平均分（越高越好） |
| **效率** | Attention kernel 延迟（ms）、总运行时间、吞吐量（TOPS） |
| **相似性分析** | Cosine Similarity、Rel. L1 Distance、RMSE、PSNR（衡量量化后 attention score 与原始 FP16 的接近程度） |

### 🆚 基线方法对比
- **Native Attention**：PyTorch 内置 SDPA（BF16 精度）
- **固定格式 Baseline**：
  - MXFP4
  - NVFP4
  - MXFP8
- **消融实验对照组**：
  - 不同 tile size（128 vs 256 vs 512）
  - 不同量化粒度（Per-Tensor / Per-Block / Per-Token）
  - 是否启用 kernel fusion 各组件

---

## 3. 主要实验结果和性能指标

### 📈 准确性结果（LongBench 平均得分）

| 模型 | 方法 | 平均得分 |
|------|------|---------|
| LLaMA-3.1-8B | Native (SDPA) | 44.11 |
| | **Ours (DMA)** | **46.43** ↑ |
| LLaMA-3.2-3B | Native (SDPA) | 35.84 |
| | **Ours (DMA)** | **37.20** ↑ |

> ✅ **结论**：DMA 不仅没有损失精度，反而在多数任务上表现更优，尤其在 `repobench-p`、`triviaqa`、`samsum` 上提升明显。

---

### ⏱ 效率结果（延迟对比）

| 方法 | Attention Time (ms) | Quant Overhead (ms) | **Total Runtime (ms)** |
|------|---------------------|------------------------|--------------------------|
| MXFP4 | 12.491 | 0.242 | 12.980 |
| NVFP4 | 12.941 | 0.204 | 13.404 |
| MXFP8 | 16.480 | 0.044 | 16.771 |
| **Ours (DMA, 128/128)** | **7.110** | **0.382** | **7.776** ↓ |
| Ours (DMA, 256/256) | 15.056 | 0.382 | 15.720 |

> ✅ **关键发现**：
> - DMA 在 **128/128 设置下达到最低总延迟（7.776ms）**
> - 相比 MXFP8 快 **~2.15×**，相比 MXFP4 快 **~1.67×**

---

### 🔬 消融实验结果

#### （1）Kernel Fusion 消融（L=2k / L=8k）

| 配置 | L=2k (μs) | L=8k (μs) | 加速比 |
|------|-----------|-----------|--------|
| Fully Unfused | 7262.41 | 22628.96 | ×1 |
| +Encode | 802.90 | 1113.77 | ×9 |
| +Encode +Pack | 740.64 | 942.67 | ×9.8 |
| +Scale Cvt. | 179.97 | 299.69 | ×40 |
| **Full Fusion (MP)** | **97.87** | **282.46** | **×74.2 / ×80.1** |

> ✅ **结论**：完整 kernel fusion 至关重要，带来 **超过 74× 的端到端加速**。

#### （2）Mixed-Precision Tile Size 影响（相似性 vs 性能）

| Diag./Sink | BitHigh (%) | Cos Sim ↑ | Rel. L1 ↓ | RMSE ↓ | PSNR ↑ | Total Latency |
|------------|-------------|-----------|----------|--------|--------|---------------|
| 128/128 | 2.30% | 0.822 | 0.539 | 0.059 | 44.657 | **7.776ms** |
| 512/512 | 9.22% | 0.826 | 0.542 | 0.058 | 44.731 | 15.720ms |
| 2048/2048 | 36.87% | 0.852 | 0.521 | 0.054 | 45.352 | >30ms |

> ✅ **结论**：增大 tile size 可略微提升 similarity，但代价是显著增加延迟；**128/128 是最佳平衡点**。

#### （3）量化粒度影响（Granularity Ablation）

| Granularity | Latency | Cos Sim | Rel. L1 | RMSE | PSNR |
|------------|--------|--------|--------|------|------|
| Per-Tensor | 6.276ms | 0.732 | 0.560 | 0.067 | 43.479 |
| Per-Block | 6.366ms | 0.736 | 0.558 | 0.067 | 43.531 |
| **Per-Token** | **7.131ms** | **0.822** | **0.539** | **0.059** | **44.657** |

> ✅ **结论**：**Per-Token 量化粒度最优**，虽延迟略高，但保真度远超其他方案。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **对角线附近 attention score 最敏感**，应优先保留高精度表示。
2. **混合精度分块策略（DMA）可在极小高精度占比下（<3%）恢复几乎完整的生成质量**。
3. **kernel fusion 是低比特 attention 高效化的关键**，非融合流程会严重抵消低比特带来的理论加速。
4. **Per-Token 量化粒度 + 对角保护机制** 是当前最有效的低比特 attention 设计范式。
5. 在 NVIDIA B200 上，DMA 实现了 **无损甚至增益的 accuracy + 超过 1.6× 的 end-to-end 加速**。

---

### ⚠️ 局限性
- 当前验证集中在文本类任务（LongBench），尚未扩展至 **vision 或 multimodal 场景**。
- 实验主要基于 LLaMA 系列模型，未覆盖更多架构变体（如 MoE、MQA/GQA）。
- 混合精度策略未在 **极端长序列（>100K）** 下充分验证。
- 所有测试均在单卡 B200 上完成，缺乏分布式或多节点场景下的评估。

---

### 🔮 未来工作方向
- 探索 **动态调整 diagonal window size** 的机制，根据输入内容自适应分配精度资源。
- 将 DMA 扩展至 **vision transformer 和 multimodal models**（如 LLaVA、Qwen-VL）。
- 结合 **sparsity 与 mixed-precision**，进一步压缩计算图密度。
- 在多种硬件平台（如 H100、TPU、移动端）上验证通用性和移植性。
- 探索 **training-time aware 的 mixed-precision attention 设计**，推动 8-bit training 实用化（参考 SageAttention3）。

</details>

---

### 5. [TinyNina: A Resource-Efficient Edge-AI Framework for Sustainable Air Quality Monitoring via Intra-Image Satellite Super-Resolution](https://arxiv.org/abs/2604.04445)

**Authors**: Prasanjit Dey, Zachary Yahn, Bianca Schoen-Phelan, Soumyabrata Dev  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2604.04445v1  

#### Abstract
Nitrogen dioxide (NO$_2$) is a primary atmospheric pollutant and a significant contributor to respiratory morbidity and urban climate-related challenges. While satellite platforms like Sentinel-2 provide global coverage, their native spatial resolution often limits the precision required, fine-grain...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《TinyNina: A Resource-Efficient Edge-AI Framework for Sustainable Air Quality Monitoring via Intra-Image Satellite Super-Resolution》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **高分辨率 NO₂ 监测难题**：卫星影像（如 Sentinel-2）虽具有全球覆盖能力，但其原生空间分辨率（10–60m）不足以支持细粒度的氮氧化物（NO₂）污染监测。
- **依赖外部高分辨率标签数据**：传统超分辨率（Super-Resolution, SR）方法通常需要昂贵且难以获取的高分辨率参考图像进行训练，限制了在资源匮乏地区的可扩展性。
- **计算开销大、难以边缘部署**：主流深度学习模型（如 EDSR、RCAN）参数量巨大，推理速度慢，不适用于实时、低功耗的 Edge AI 场景。

### 🚀 提出的新方法与创新思路
- **TinyNina 框架**：一种超轻量级、面向任务优化的 Edge-AI 超分辨率框架，专为可持续空气质量监测设计。
- **Intra-Image Spectral Super-Resolution（图像内光谱超分）**：
  - 利用 Sentinel-2 自身多光谱层级结构作为内部监督信号（例如用 10m 分辨率波段指导 20m 波段的重建），无需任何外部高分辨率参考数据。
  - 实现“数据自洽”训练，提升模型在无标注区域的泛化能力。
- **任务感知架构设计（Task-Aware Architecture）**：
  - 引入 **Spectral Attention Gates**，动态加权对 NO₂ 敏感的关键波段（如红边 B5–B7 和可见光 B4）。
  - 采用 **Depthwise Separable Convolutions** 显著降低参数量和计算复杂度。
  - 设计 **Multi-scale Residual Upsampling** 结构，在保持光谱保真度的同时恢复空间细节。

### ⚖️ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 仅含 **51K 参数**，相比 EDSR（40.7M）减少约 **95% 模型大小**，实现 **47× 更快推理速度**。 |
| **实用性** | 不依赖外部高分辨率数据，适合全球范围部署，尤其适用于缺乏地面观测或辅助数据的地区。 |
| **任务导向性** | 直接优化下游 NO₂ 预测性能，而非传统图像质量指标（PSNR/SSIM），更符合环境监测的实际需求。 |
| **可持续性** | 极低能耗，支持 Green Computing 和边缘设备部署，契合可持续发展目标。 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **来源**：基于 Scheibenreif et al. [22] 构建的数据集，实现 Sentinel-2 卫星观测与地面监测站的时空对齐。
- **地理覆盖**：美国西海岸 27 个 EPA 监测站点（涵盖城市、郊区、农村）。
- **时间跨度**：2018 年 1 月至 2020 年 12 月，共 **3,276 对匹配样本**。
- **卫星数据**：
  - Sentinel-2 Level-2A 表面反射率产品（12 个波段，排除 B10）。
  - 包括 10m（B2/B3/B4/B8）、20m（B5/B6/B7/B8A/B11/B12）、60m（B1/B9）三种分辨率波段。
- **地面真值**：EPA 提供的小时级 NO₂ 浓度，并与卫星过境时间精确对齐。

### 🔍 实验设置与评估指标
#### 评估目标
- 下游任务性能：**地面 NO₂ 浓度预测精度**，而非图像视觉质量。

#### 主要评估指标
| 指标 | 定义 |
|------|------|
| **MAE**（Mean Absolute Error） | $ \frac{1}{n}\sum_{i=1}^{n} |y_i - g_i| $，单位：μg/m³ |
| **MSE**（Mean Squared Error） | $ \frac{1}{n}\sum_{i=1}^{n} (y_i - g_i)^2 $，单位：μg/m³ |

#### 基线方法对比
| 模型 | 类型 | 参数量 | 是否使用外部数据 |
|------|------|--------|------------------|
| **EDSR** | CNN-based SR | 40.7M | 是（如 RapidEye） |
| **RCAN** | Attention-enhanced SR | 15.4M | 是 |
| **NinaB1** | Lightweight hybrid | 1.02M | 是 |
| **TinyNina (Ours)** | Task-aware + Intra-image | **51K** | ❌（仅用 Sentinel-2 内部信息） |

#### 训练策略
- **两种训练范式**：
  1. **Naive SR**：统一处理所有 12 个通道。
  2. **Channel SR**（推荐）：选择性增强 20m 波段（B5–B7, B8A, B11–B12），以对应 10m 波段为引导（如 B4 → B5–B7）。
- **损失函数**：
  - Naive SR：L1 Reconstruction Loss
  - Channel SR：结合 L1 + L2 正则项（λ=1e-4）
- **预测模型**：基于 ResNet50 改造的回归网络，加入波长注意力机制与时序嵌入。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
| 模型 | MSE (μg/m³) | MAE (μg/m³) |
|------|-------------|-------------|
| **EDSR (Naive SR)** | 112 | 8.2 |
| **RCAN (Naive SR)** | 98 | 7.8 |
| **TinyNina (Channel SR)** | **97** | **7.4** ✅ |

> ✅ **TinyNina 在 MAE 上达到 7.4 μg/m³，优于所有基线模型，满足 EPA 监测标准（误差 <15%）**

### 🔁 与基线方法对比结果
- **精度更高**：
  - MAE 比 RCAN 降低 **5.1%**（7.8 → 7.4），尽管参数仅为后者的 **0.3%**。
- **收敛更快**：
  - NO₂ 预测模型在 TinyNina 输出上训练时，**提前 40–50 轮完成收敛**。
- **推理极快**：
  - 处理 500 张 200×200 卫星图块：
    - **TinyNina**: 45 秒（≈11 tiles/s）
    - **EDSR**: 35 分钟（≈0.24 tiles/s）
    - **加速比达 47×**
- **稳定性更强**：
  - 在城市复杂排放区，TinyNina 的 MAE 标准差低于 **2.1 μg/m³**，仅为 EDSR 的一半（4.2 μg/m³）。

### 🔬 消融实验结果（Ablation Study）
| 变体 | Spectral Attention | MSE | MAE |
|------|--------------------|-----|-----|
| TinyNina（无 attention） | ❌ | 102 | 7.9 |
| TinyNina（完整模型） | ✅ | **97** | **7.4** |

> ✅ **引入 Spectral Attention 可使 MAE 下降 0.5 μg/m³，验证其有效聚焦于污染物敏感波段的能力。**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **任务导向设计胜过模型规模**：
   - 尽管 TinyNina 参数极少（51K），但在 NO₂ 预测任务中表现优于数十倍大的模型，证明“**少即是多**”的设计哲学在特定领域更具优势。
2. **Intra-Image 学习是可行且高效的替代方案**：
   - 成功摆脱对外部高分辨率数据的依赖，利用 Sentinel-2 自身多尺度波段关系即可实现高质量超分。
3. **边缘部署完全可行**：
   - 在 Intel Core i7 CPU 上单图推理延迟仅 **~90ms**，可在 Jetson Nano/Xavier NX 等边缘设备实现实时运行（预计吞吐量 4–12 tiles/s）。
4. **绿色 AI 实践典范**：
   - 单次推理能耗约 **5.85 J**，百万级处理仅需 **~1.6 kWh**，显著降低大规模遥感分析的碳足迹。

### ⚠️ 局限性
- **云层干扰风险**：尽管预处理包含云掩膜（SCL），但残留大气效应仍可能影响光谱重建准确性。
- **时间错配问题**：固定重访周期（5天）可能导致错过短期污染事件（如交通高峰、野火烟雾）。
- **气象因素未显式建模**：风速、温度逆温等扩散条件未纳入输入，可能影响空间分布推断。
- **域迁移挑战**：在气候、地形差异较大的新区域应用时可能出现性能下降。

### 🔮 未来工作方向
- **融合多源数据**：集成气象数据（风速、湿度）、交通流、土地利用等辅助变量以增强鲁棒性。
- **引入时序建模**：利用多时相 Sentinel-2 序列捕捉动态污染演化过程。
- **扩展至其他污染物**：将框架推广至 PM2.5、CO、O₃ 等多种空气污染物的联合监测。
- **端到端联合训练**：探索超分辨率模块与 NO₂ 回归头的联合优化路径，进一步提升任务一致性。
- **智能交通系统集成**：应用于 eco-routing、排放区管理、ITS 控制中心，推动智慧低碳城市建设。

---

> 💡 **总体评价**：  
> TinyNina 不仅是一项技术突破，更是 **Green AI** 与 **Sustainable Engineering** 的典范——它通过精巧的任务感知设计，在极致压缩模型的同时实现了卓越的应用性能，为全球范围内低成本、高时效的空气质量监测提供了切实可行的技术路径。

</details>

---

### 6. [Scalable Variational Bayesian Fine-Tuning of LLMs via Orthogonalized Low-Rank Adapters](https://arxiv.org/abs/2604.03388)

**Authors**: Haotian Xiang, Bingcong Li, Qin Lu  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.03388v1  

#### Abstract
When deploying large language models (LLMs) to safety-critical applications, uncertainty quantification (UQ) is of utmost importance to self-assess the reliability of the LLM-based decisions. However, such decisions typically suffer from overconfidence, particularly after parameter-efficient fine-tu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scalable Variational Bayesian Fine-Tuning of LLMs via Orthogonalized Low-Rank Adapters

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在将 **Large Language Models (LLMs)** 部署到安全关键应用（如医疗诊断、法律分析）时，**不确定性量化 (Uncertainty Quantification, UQ)** 至关重要，以评估模型决策的可靠性。然而，现有的 **Parameter-Efficient Fine-Tuning (PEFT)** 方法（如 LoRA）在有限数据下进行微调后，常表现出严重的 **过度自信 (overconfidence)**，导致 UQ 不准确。

现有 UQ 方法存在以下局限：
- **Laplace Approximation (LA)**：依赖于 MAP 估计后的局部近似，若训练轨迹不佳，则 UQ 效果受限。
- **Variational Bayesian (VB) 方法**（如 BLoB）：虽能联合优化均值和协方差，但推理时需对整个 LLM 主干进行多次前向传播（Monte Carlo 采样），计算开销大，难以部署。

此外，标准 LoRA 存在 **rank collapse** 问题，即其有效秩（stable rank）接近 1，导致特征空间被压缩，破坏了语义距离，不利于下游贝叶斯推理。

### 提出的新方法
本文提出 **PoLAR-VBLL**，一个结合了架构优化与可扩展贝叶斯推理的统一框架，旨在解决上述问题。

#### 核心创新点：
1. **Polar-decomposed Low-rank Adapter Representation (PoLAR)**  
   - 采用正交分解参数化：$ \Delta W = UAV^T $，其中 $ U \in \text{St}(m, r), V \in \text{St}(n, r) $ 为 Stiefel 流形上的正交矩阵，$ A \in \mathbb{R}^{r\times r} $ 为自由参数。
   - 引入 **Riemannian optimization** 和 **landing field** 方法，避免昂贵的 SVD 或 QR 分解，实现高效优化。
   - 该设计显著缓解了 LoRA 的 rank collapse，保留了多方向特征几何，有利于距离感知的 UQ。

2. **Variational Bayesian Last Layer (VBLL)**  
   - 采用 **Bayesian Last Layer (BLL)** 范式：LLM 主干作为确定性特征提取器，仅最后一层权重为随机变量。
   - 使用变分推断优化 **Jensen-tightened ELBO**，闭式表达避免了 Monte Carlo 采样，训练高效。
   - 推理时只需一次主干前向传播 + 多次轻量级最后一层采样，显著提升速度。

3. **Hybrid Post-hoc Laplace Refinement**  
   - 在 VBLL 训练得到高质量后验模式后，**可选地** 应用 LA 对最后一层协方差进行精细化校准。
   - 利用精确 Hessian 信息修正变分协方差，进一步提升 UQ 质量。

### 相比现有方法的优势
| 维度 | PoLAR-VBLL | BLoB / TFB 类方法 | Laplace-LoRA |
|------|------------|------------------|-------------|
| **UQ 准确性** | ✅ 最佳（高 ACC + 低 ECE/NLL） | ❌ 有精度-校准权衡 | ⚠️ 依赖 MAP 初始化质量 |
| **推理效率** | ✅ 单次主干前传 | ❌ 多次主干前传 | ✅ 单次主干前传 |
| **特征表达能力** | ✅ 高稳定秩，保留几何 | ❌ LoRA 秩坍缩 | ❌ LoRA 秩坍缩 |
| **训练效率** | ✅ 闭式 ELBO，无 MC 采样 | ❌ 需 MC 采样 | ✅ MAP 训练 |

---

## 2. 核心实验方法和设置

### 数据集
- **In-Distribution (ID)**：6 个常识推理任务：
  - Winogrande-S/M (WG-S, WG-M)
  - ARC-Challenge/Easy (ARC-C, ARC-E)
  - OpenBookQA (OBQA)
  - BoolQ
- **Out-of-Distribution (OOD)**：评估分布偏移下的泛化能力：
  - OBQA → ARC-C/E（小偏移）
  - OBQA → MMLU 化学/物理（大偏移）

### 实验设置
- **模型**：LLaMA-3.1-8B 和 LLaMA-2-7B
- **Adapter 设置**：PoLAR/LoRA rank = 8，应用于输出层及所有注意力层的 query/value 投影
- **训练步数**：5000 步，batch size = 4
- **评估指标**：
  - **ACC (%)**：准确率
  - **ECE (%)**：期望校准误差（越低越好）
  - **NLL**：负对数似然（越低越好）

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **标准 PEFT** | MLE, MAP |
| **UQ on LoRA** | MCD, ENS, LA |
| **贝叶斯 LoRA** | BLoB, ScalaBL, C-LoRA, TFB |
| **PoLAR 变体** | PoLAR-MLE, PoLAR-LA, PoLAR-BLoB 等 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（LLaMA-3.1-8B，ID 平均）
| 方法 | ACC (%) | ECE (%) | NLL |
|------|--------|--------|-----|
| **PoLAR-VBLL (ours)** | **81.79** | **5.86** | **0.64** |
| PoLAR-BLoB | 80.03 | 7.02 | 0.73 |
| BLoB | 79.42 | 9.59 | 0.78 |
| PoLAR-LA | 81.21 | 9.69 | 1.03 |
| MLE | 81.05 | 17.95 | 1.75 |

> ✅ PoLAR-VBLL 在保持最高 ACC 的同时，实现了最佳的 ECE 和 NLL，**无精度-校准权衡**。

### OOD 性能（OBQA → Chem/Phy）
| 方法 | Chem ACC | Phy ACC |
|------|---------|--------|
| **PoLAR-VBLL** | **49.30** | **48.91** |
| PoLAR-BLoB | 46.18 | 46.88 |
| BLoB | 45.67 | 45.67 |
| MLE | 48.00 | 43.33 |

> ✅ 在大分布偏移下，PoLAR-VBLL 显著优于其他方法，表明其 UQ 更可靠。

### 消融实验结果

#### (1) PoLAR vs. LoRA 特征几何
- **稳定秩 (stable rank)**：
  - LoRA：≈ 1.53（严重坍缩）
  - PoLAR：≈ 2.86（显著更高）
> ✅ PoLAR 有效缓解了 rank collapse，保留了更丰富的特征表示。

#### (2) VBLL 是校准的核心驱动
- 在相同 PoLAR 架构下比较：
  | 方法 | ECE ↓ |
  |------|------|
  | PoLAR-VBLL (w/o LA) | **5.86** |
  | PoLAR-LA-LL | 8.36 |
  | PoLAR-BLoB | 7.02 |
> ✅ 即使不加 LA，VBLL 本身已提供强校准，证明其是 UQ 的主要驱动力。

#### (3) 后处理 LA 的增益
- 加入 LA 后，ECE 进一步下降（如从 5.86 → 4.92），且 **不影响 ACC**。
> ✅ LA 作为“精加工”步骤，有效提升协方差估计质量。

#### (4) 计算效率
| 方法 | 推理时间 (s) | 内存 (MB) |
|------|------------|----------|
| **PoLAR-VBLL** | **12** | 18,423 |
| BLoB | 80–90 | ~19,800 |
| TFB | 90 | 24,432 |
> ✅ 实现约 **7× 推理加速**，适合资源受限场景。

---

## 4. 关键结论和发现

### 主要发现
1. **PoLAR 有效缓解了 LoRA 的 rank collapse**，通过正交约束保留了多方向特征几何，为高质量 UQ 提供了基础。
2. **VBLL 是实现良好校准的核心机制**，其闭式 ELBO 优化避免了昂贵的 MC 采样，训练高效且效果优于后处理 LA。
3. **PoLAR-VBLL 实现了 ACC 与 UQ 的双赢**，在多个 ID/OOD 任务上均取得 SOTA 表现，且无传统方法中的精度-校准权衡。
4. **后处理 LA 可进一步提升性能**，但前提是 VBLL 已找到高质量后验模式，否则 LA 效果有限。

### 方法的局限性
- 当前框架聚焦于 **分类任务**，对生成任务（如文本生成中的 UQ）的适配尚待探索。
- PoLAR 的正交约束虽然提升了表达能力，但增加了优化复杂性，尽管 landing field 缓解了此问题。
- 实验主要基于 LLaMA 系列模型，对其他架构（如 Mistral、Qwen）的泛化性需进一步验证。

### 未来工作方向
- 将 PoLAR-VBLL 扩展至 **序列生成任务**，实现 token-level 的不确定性估计。
- 探索 **动态 rank allocation**，根据任务需求自适应调整 PoLAR 的秩。
- 结合 **active learning** 或 **rejection option**，利用高质量 UQ 提升人机协作系统的安全性与效率。

--- 

> **总结**：PoLAR-VBLL 通过 **架构增强 (PoLAR)** 与 **可扩展贝叶斯推理 (VBLL)** 的协同设计，为 LLM 的高效、可靠微调提供了新范式，在保持高性能的同时实现了**可部署的、校准良好的不确定性量化**。

</details>

---

### 7. [Bypassing the CSI Bottleneck: MARL-Driven Spatial Control for Reflector Arrays](https://arxiv.org/abs/2604.05162)

**Authors**: Hieu Le, Oguz Bedir, Mostafa Ibrahim, Jian Tao, Sabit Ekin  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 9.5  
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
传统 **Reconfigurable Intelligent Surfaces (RIS)** 在实际部署中面临严重的 **Channel State Information (CSI) estimation bottleneck**。精确获取 CSI 需要大量导频开销和计算资源，尤其在毫米波（mmWave）非直视（NLOS）环境中，其复杂度随反射单元数量呈指数增长，严重限制了系统可扩展性和实时性。

此外，依赖电子相位调控的 RIS 还需要高精度的 **phase shifters** 和同步机制，增加了硬件成本与能耗。

### 🚀 提出的新方法与创新思路
本文提出一种 **AI-native、data-driven 的全新范式**，通过以下关键技术绕过 CSI 瓶颈：

- **机械可调金属反射器阵列（Mechanically Adjustable Metallic Reflector Arrays）**  
  使用伺服电机控制每个六边形金属瓦片的 **elevation** 和 **azimuth** 角度，物理上重定向 mmWave 波束，完全避免复杂的 RF 电路和电子相移器。

- **空间智能替代信道建模（Spatial Intelligence over Electromagnetic Modeling）**  
  不再依赖 CSI，而是利用用户的 **位置坐标（user coordinates）** 作为输入，实现 **CSI-free operation**。

- **基于虚拟焦点的空间抽象（Focal Point Control Abstraction）**  
  将原本 $2N_pN_c$ 维的独立角度控制问题，压缩为仅 $3L$ 维的 **virtual focal point** 控制空间（$L \ll N_pN_c$），大幅降低优化维度并自然满足机械约束。

- **多智能体强化学习框架（MARL with CTDE 架构）**  
  采用 **Centralized Training with Decentralized Execution (CTDE)** 范式，结合 **Multi-Agent Proximal Policy Optimization (MAPPO)**，使各 segment 上的 agent 自主协作完成 beam focusing。

### 🔍 相比现有方法的优势
| 方面 | 本方法优势 |
|------|-----------|
| **无需 CSI** | 完全摆脱对信道估计的依赖，适用于动态、NLOS 场景 |
| **宽带兼容性** | 机械反射不依赖频率敏感元件，支持宽频段操作 |
| **部署可行性高** | 使用商用伺服系统（COTS servos），易于集成 |
| **可扩展性强** | MARL + 空间抽象支持大规模阵列扩展 |
| **鲁棒性强** | 对用户定位噪声具有良好的容忍能力 |

---

## 2. 核心实验方法和设置

### 🧪 数据集与仿真环境
- **未使用真实数据集**，而是构建了一个高保真 **ray-tracing 仿真平台**。
- 使用 **NVIDIA Sionna** 引擎进行 60 GHz mmWave 下行链路传播建模。
- 环境建模工具：**Blender** 构建三维 L 形走廊场景，包含 concrete、plasterboard、wood 等材料（符合 ITU 推荐值）。

### ⚙️ 实验设置
- **AP 设置**：发射功率 5 mW，位于 NLOS 区域外。
- **用户配置**：3 名移动用户，在每 4 个仿真步更新位置，模拟动态场景。
- **反射器阵列**：由 72 个六边形金属瓦片组成的 NT × Nc 阵列，划分为多个 segments，每个 segment 分配一个智能体控制。
- **状态空间（State Space）**：
  - 全局状态：所有用户位置 + 各 agent 当前 focal point
  - 局部观测（Local Observation）：仅限自身负责用户的坐标、segment 位置和当前 focal point
- **动作空间（Action Space）**：连续三维位移向量 $\mathbf{a}_{l,t} = [\Delta f_{x}, \Delta f_{y}, \Delta f_{z}]$，用于更新 focal point
- **奖励函数（Reward Function）**：混合奖励
  $$
  R_l(s_t, a_t) = \frac{1}{K}\sum_{k'=1}^K P_{r,k'}(s_t,a_t) + P_{r,k(l)}(s_t,a_t)
  $$
  即兼顾全局平均 RSSI 与本地目标用户的信号强度。

### 📊 评估指标
- **Received Signal Strength Indicator (RSSI)**（dBm）
- **空间信号热图（Spatial RSSI Heatmap）**
- **时间域 RSSI 变化曲线（Temporal RSSI over 300 steps）**
- **收敛速度与稳定性（Training convergence curve）**
- **抗噪能力测试（Robustness to localization noise）**

### 🔁 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **No Reflector** | 无任何反射辅助的纯 NLOS 场景 |
| **Flat Reflector** | 固定平面反射器（静态） |
| **beam-focusing-sa** | 单智能体（Single-Agent）DRL 控制整个阵列 |
| **column-based-ma** | 多智能体但硬件受限：azimuth 控制按列共享（减少 servo 数量） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
| 场景 | 平均 RSSI (dBm) | 相对增益 |
|------|------------------|---------|
| No Reflector | -110.50 dBm | — |
| Flat Reflector | -94.40 dBm | +16.1 dB |
| **Proposed (beam-focusing-ma)** | **-67.54 dBm** | **+26.86 dB vs flat**, **+42.96 dB vs no reflector** |
| Single-Agent (sa) | -72.48 dBm | 比 MA 差约 5 dB |
| Column-based-ma | -74.79 dBm | 比全自由度 MA 差约 7 dB |

> ✅ **最大提升达 26.86 dB 超过静态反射器**

### 📊 与基线方法对比结果
- **训练收敛性（Fig. 3）**：
  - **beam-focusing-ma** 快速上升并在 ~1000 episode 后收敛至 ~42 的累积奖励
  - **beam-focusing-sa** 和 **column-based-ma** 分别停滞在 ~24 和 ~27，显著低于所提方法
  > 表明 **多智能体分解任务能有效缓解维度灾难**

- **空间聚焦能力（Fig. 4）**：
  - 所提方法生成高度集中的“信号口袋”（signal pockets）覆盖用户位置
  - 单智能体和列控方法信号分布更弥散，空间选择性较差

- **时序适应性（Fig. 5）**：
  - **beam-focusing-ma** 在用户移动后通常只需 **1 个仿真步**即可重新对准
  - 平均 RSSI 达 **-66.83 dBm**，远高于 sa（-79.70 dBm）和 column-based（-73.65 dBm）
  - 时间稳定性更高，波动小

### 🔍 消融实验与鲁棒性分析（Fig. 6）
引入不同标准差的高斯定位噪声（0.1m ~ 1.0m）验证系统鲁棒性：

| 定位噪声（σ） | 平均 RSSI | 性能退化情况 |
|---------------|----------|-------------|
| 0 m（理想） | -68.15 dBm | 最优性能 |
| 0.1 m | -68.27 dBm | 几乎无影响 |
| 0.3 m | -68.62 dBm | 微弱下降 |
| 0.5 m | -71.21 dBm | 明显下降但仍可用 |
| **1.0 m** | **-72.36 dBm** | **仍优于单智能体基线，系统未崩溃** |

> ✅ **关键发现**：系统表现出“优雅降级”（graceful degradation），即使在 **1米定位误差下仍维持稳定通信**，证明其具备强部署韧性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MARL + 空间抽象是突破 CSI 瓶颈的有效路径**  
   利用 **user location** 替代 CSI，结合 **focal point abstraction** 和 **MAPPO**，实现了高效、自主的无线传播控制。

2. **多智能体架构优于集中式控制**  
   decentralized task decomposition 显著提升了学习效率、空间聚焦能力和动态适应性。

3. **机械反射器具备实用潜力**  
   尽管响应速度慢于电子 RIS，但在 **宽带、低成本、易部署** 场景中优势明显。

4. **系统对现实不确定性具有强鲁棒性**  
   在高达 1 米的定位误差下仍能保持可用信号质量，适合实际室内定位技术（如 UWB）集成。

### ⚠️ 方法的局限性
- **延迟问题**：机械调整速度受限于伺服电机响应时间（毫秒级），可能难以应对极高速移动用户。
- **局部最优风险**：MAPPO 可能在复杂多径环境中陷入次优解。
- **泛化能力待验证**：目前仅在一个特定几何环境下测试，跨场景迁移能力未知。
- **未考虑遮挡动态变化**：障碍物运动未纳入模型。

### 🔮 未来工作方向
1. **硬件原型验证**：搭建物理实验平台，实测机械反射器的实际性能。
2. **与 UWB 定位系统集成**：实现端到端的 real-time indoor positioning + MARL 控制闭环。
3. **引入预测机制**：结合轨迹预测提升对快速移动用户的跟踪能力。
4. **探索 hybrid 架构**：部分机械 + 部分电子控制，平衡性能与成本。
5. **扩展至多 AP 多 RIS 场景**：研究更大规模智能环境协同控制。

---

> 💡 **总体评价**：该论文开创性地将 **MARL 与机械反射器设计相结合**，提出了一条脱离 CSI 依赖、面向实际部署的 **AI-empowered wireless networking 新路径**，兼具理论深度与工程价值。

</details>

---

### 8. [Learning to Focus: CSI-Free Hierarchical MARL for Reconfigurable Reflectors](https://arxiv.org/abs/2604.05165)

**Authors**: Hieu Le, Mostafa Ibrahim, Oguz Bedir, Jian Tao, Sabit Ekin  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.05165v1  

#### Abstract
Reconfigurable Intelligent Surfaces (RIS) has a potential to engineer smart radio environments for next-generation millimeter-wave (mmWave) networks. However, the prohibitive computational overhead of Channel State Information (CSI) estimation and the dimensionality explosion inherent in centralized...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning to Focus: CSI-Free Hierarchical MARL for Reconfigurable Reflectors*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Reconfigurable Intelligent Surfaces (RIS)** 在毫米波（mmWave）网络中面临两大瓶颈：
- **Channel State Information (CSI) 估计开销巨大**：随着反射单元数量增加，信道估计所需的导频信号导致严重的谱效率损失。
- **集中式优化维度爆炸**：对每个反射单元进行独立控制带来极高的计算复杂度，难以扩展到大规模部署。

此外，现有的基于 **Deep Reinforcement Learning (DRL)** 或 **Multi-Agent Reinforcement Learning (MARL)** 的方法仍依赖于显式的 CSI 获取，未能根本解决上述问题。

---

### 🚀 提出的新方法与创新思路

本文提出一种全新的 **“CSI-free” 范式**，结合 **机械可重构反射器（mechanically reconfigurable metallic reflectors）** 与 **分层多智能体强化学习（Hierarchical Multi-Agent Reinforcement Learning, HMARL）** 架构，实现无需 CSI 的高效波束聚焦控制。

#### 主要创新点包括：

1. **Eliminate CSI Dependency via Spatial Intelligence**
   - 不再依赖电磁级 CSI 估计，转而利用用户定位数据（localization data）进行空间感知驱动的宏观信号传播管理。
   - 利用几何先验（如距离、反射角）构建 **Compatibility Matrix**，作为高阶控制器的归纳偏置（inductive bias），引导用户-反射器分配。

2. **Hierarchical Task Decomposition**
   - 将控制任务分解为两个层级：
     - **High-Level Controller**：执行离散的、时间扩展的 **user-to-reflector 分配决策**（每 $T$ 步更新一次）。
     - **Low-Level Controllers**：分布式地优化连续的 **focal point 位移动作**（每步更新），以最大化接收信号强度（RSSI）。
   - 显著降低控制维度从 $D_{\text{tile}} = K + 2N_rN_c$ 到 $D_{\text{focal}} = K + 3L$，提升训练效率和可扩展性。

3. **MAPPO + CTDE 架构**
   - 使用 **Multi-Agent Proximal Policy Optimization (MAPPO)** 在 **Centralized Training with Decentralized Execution (CTDE)** 框架下训练策略。
   - 训练时使用全局 critic 进行准确优势估计；部署时仅依赖局部观测，无通信开销，适合实际应用。

4. **Mechanical Reflectors over Electronic Metasurfaces**
   - 采用机械式金属反射板阵列（类似线性菲涅尔反射器），避免复杂的 RF 电路和相位调控硬件。
   - 具备宽频带、低成本、低功耗等优势，适用于室内 NLOS 场景。

---

### 🔍 相比现有方法的优势

| 维度 | 本工作 | 现有主流方法 |
|------|--------|-------------|
| CSI 依赖 | ❌ 完全消除 | ✅ 强依赖，需大量导频 |
| 控制粒度 | Segment-level focal point | Per-element phase shift |
| 可扩展性 | 高（HMARL + 维度压缩） | 低（集中式优化维度爆炸） |
| 硬件成本 | 低（机械结构，无需 RF IC） | 高（电子相控阵） |
| 学习稳定性 | 高（兼容矩阵引导 + 时间抽象） | 易受稀疏奖励影响 |

---

## 2. 核心实验方法和设置

### 🧪 数据集与仿真环境
- **未使用公开数据集**，而是构建了一个高保真 **60 GHz 室内 mmWave 仿真环境**。
- 使用 **NVIDIA Sionna** 的确定性 **ray-tracing 引擎** 结合 **Blender** 建模会议室场景。
- 材料参数遵循 **ITU-R P.2040-1** 标准（混凝土墙、大理石地板、金属反射板）。
- AP 外置于房间外，服务区域内 $K \in \{2,4\}$ 名用户，两个机械反射阵列部署在角落。

### ⚙️ 实验设置
| 参数 | 设置值 |
|------|-------|
| 频率 | 60 GHz |
| 发射功率 | 5 dBm |
| 区域大小 | 10 m × 10 m |
| 用户移动速度 | 1 m/s（测试阶段引入动态性） |
| 反射器配置 | 2 个阵列，每个由多个 tile 组成 |
| 观测输入 | 用户位置、反射器位置、焦点位置（局部屏蔽） |
| 动作空间 | 高层：离散分配；底层：连续焦点位移 $\Delta f_l \in \mathbb{R}^3$ |

### 📊 评估指标
- **主指标**：
  - **Received Signal Strength Indicator (RSSI)**：单位 dBm，衡量系统整体覆盖质量。
  - **Episode-Averaged Reward**：累计接收功率作为奖励函数。
- **鲁棒性测试**：
  - 注入不同水平的 **Gaussian localization error**（$\sigma_{\text{error}} \in \{0.0, 0.1, 0.3, 0.5, 1.0, 2.0\}$ 米），验证定位误差下的性能退化情况。

### 🔁 基线方法对比
1. **Allocator (Full HMARL)**：本文完整框架（含兼容矩阵）。
2. **No_compat**：去除兼容矩阵的 HMARL 变体。
3. **No_allocator**：传统集中式 PPO 控制器，直接联合优化所有变量。
4. **Random**：随机用户-反射器分配策略。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 性能增益显著
- 在 **4 用户场景** 下，本文方法达到平均 **RSSI = -66.47 dBm**。
- 相比集中式 PPO 基线 (**No_allocator**: -74.26 dBm)，实现高达 **+7.79 dB** 的 RSSI 提升。
  > 这是目前同类研究中报道的最大增益之一。

#### 🔄 收敛速度更快
- **Training Convergence 图显示**：
  - **Allocator** 在前 500 episodes 内快速上升并收敛至约 reward=70。
  - **No_compat** 和 **No_allocator** 均停滞在 reward≈50 左右。
  - **Random** 最终仅达 reward≈39。

#### 🧩 消融实验结果（Ablation Study）
| 方法 | 平均 RSSI (dBm) | 相对增益 |
|------|------------------|---------|
| Random | -80+（极差） | — |
| No_allocator (Centralized PPO) | -74.26 | 基线 |
| No_compat (w/o compatibility matrix) | -70.59 | +3.67 dB |
| **Allocator (Proposed)** | **-66.47** | **+7.79 dB vs baseline** |

> 表明：
> - 分层架构本身带来约 **3.5 dB** 增益；
> - 兼容矩阵额外贡献 **>4.3 dB**，防止陷入次优分配策略。

#### 🛡️ 鲁棒性表现优异（Localization Error Test）
| 定位误差 $\sigma$ (m) | 平均 RSSI (dBm) | 是否可用？ |
|------------------------|------------------|------------|
| 0.0（理想） | -64.57 | ✅ 最佳 |
| 0.1（UWB级） | -65.01 | ✅ 几乎无损 |
| 0.3（WiFi/BLE级） | -70.54 | ✅ 可接受 |
| 0.5 | -77.18 | ⚠️ 开始恶化 |
| 1.0~2.0 | -78.08 ~ -83.64 | ❌ QoS严重下降 |

> **结论**：系统在 **≤0.3米误差下保持稳健运行**，适用于当前主流商用定位技术（如 UWB、BLE）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **CSI-free 是可行且高效的路径**：
   - 利用用户定位 + 几何先验完全可以替代传统 CSI，在保证性能的同时彻底消除信道估计开销。

2. **Hierarchical MARL 架构具有天然优势**：
   - 成功解耦“谁连哪个反射器”与“如何聚焦”的双重挑战，显著提升学习效率与系统可扩展性。

3. **Mechanical RIS 是极具潜力的替代方案**：
   - 相比电子 RIS，具备更宽频带、更低功耗、更高可靠性，特别适合静态或半动态室内环境。

4. **Inductive Bias 至关重要**：
   - 单纯依靠 MARL 探索难以在巨大组合空间中找到最优解；引入 **compatibility matrix** 极大加速收敛并提高最终性能。

---

### ⚠️ 局限性
1. **依赖用户定位精度**：
   - 当定位误差超过 0.5 米时性能急剧下降，限制其在低精度定位系统中的应用。
   
2. **适用于 NLOS 场景为主**：
   - 主要设计用于直视路径被遮挡的场景，LOS 场景增益可能有限。

3. **机械响应速度较慢**：
   - 机械调整存在物理延迟，不适合超高速移动用户场景（如车载通信）。

4. **仿真验证为主**：
   - 所有结果基于 ray-tracing 仿真，尚未在真实硬件平台上部署验证。

---

### 🔮 未来工作方向
1. **集成轻量级在线校准机制**：
   - 引入 feedback loop 对定位误差进行补偿，增强鲁棒性。

2. **拓展至三维空间与动态拓扑**：
   - 支持更多反射器、非平面布局及动态环境变化。

3. **硬件原型开发与实测验证**：
   - 构建物理测试平台，验证算法在真实世界中的可行性。

4. **融合 Sensing-Communication-HMARL 一体化架构**：
   - 探索 ISAC（Integrated Sensing and Communication）与 HMARL 的协同优化。

5. **探索联邦学习或边缘协作训练模式**：
   - 实现跨区域反射器集群的分布式学习与协调。

---

> 💡 **一句话总结**：  
> 该论文开创性地提出了一个 **无需 CSI、基于空间智能与 HMARL 的机械可重构反射面控制框架**，在仿真中实现了高达 **7.79 dB 的 RSSI 提升**，并在亚米级定位误差下表现出强鲁棒性，为下一代智能无线环境提供了 **低成本、可扩展、实用性强** 的全新解决方案。

</details>

---

### 9. [Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs](https://arxiv.org/abs/2604.05643)

**Authors**: Hongyuan Yuan, Xinran He, Run Shao, Bolei He, Xianwei Xue, Mengke Chen, Qiutong Pan, Haiwei Wang, Haifeng Li  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 9.5  
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
该论文针对当前基于 **Reinforcement Learning (RL)** 扩展 **Chain-of-Thought (CoT)** 的推理大模型中存在的“**过思考（overthinking）**”现象，即模型生成大量冗余的中间推理步骤，尤其是无效的自我反思（redundant reflections），导致推理效率低下、推理成本上升。

作者指出，这种冗余主要来源于两种低效的反思行为：
- **Indiscriminate Reflection（无差别反思）**：对每个中间步骤都进行检查，即使其明显正确或无关紧要。
- **Repetitive Reflection（重复反思）**：反复验证已经确认的结论，造成逻辑回溯和内容重复。

### 🚀 提出的新方法与创新思路
提出了一种 **图结构化的 CoT 优化框架（Graph-Based CoT Pruning）**，通过以下方式系统性识别并剪枝冗余反思：

1. **将线性 CoT 转换为有向无环图（DAG）**  
   - 将每个推理步骤建模为节点（node），依赖关系建模为边（edge）。
   - 节点分为两类：`progress`（推进推理）和 `review`（执行检查/验证）。
   - 利用外部 LLM（如 qwen-turbo）进行迭代图构建，恢复推理过程中的非线性结构（如分支探索、回溯等）。

2. **双层级剪枝策略（Dual Pruning Strategy）**
   - **Branch-Level Pruning**：移除子节点数少于阈值 $k=2$ 的 `review` 节点，因其形成狭窄侧支，贡献小。
   - **Depth-Level Pruning**：移除出现在推理后期（相对深度 > 0.9）的 `review` 节点，通常为事后重复验证，不增加新信息。

3. **三阶段训练流程实现高效推理能力内化**
   - **SFT（Supervised Fine-Tuning）**：在剪枝后的简洁 CoT 上初始化策略，建立简洁推理先验。
   - **DPO（Direct Preference Optimization）**：偏好对齐，鼓励选择正确且更简短的轨迹。
   - **GRPO with length penalty**：结合最终答案正确性和推理长度惩罚，联合优化准确率与效率。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | 本文优势 |
|------|--------|----------|
| **TokenSkip / Think Clearly** | 基于 token 或 attention 的局部冗余检测，缺乏全局结构理解 | 显式建模依赖关系，精准定位结构性冗余 |
| **TALE / CoT-Valve** | 控制总长度或预算，可能截断有效推理 | 结构感知剪枝，保留主干逻辑，仅删减低价值反思 |
| **O1-Pruner** | 仅压缩长轨迹，未区分语义重要性 | 基于图拓扑分析，针对性去除两类典型冗余 |

> ✅ **核心创新**：首次从“**反思行为模式**”角度分析冗余来源，并利用**图结构显式建模依赖关系**，实现**可解释、可控制的冗余剪枝**。

---

## 2. 核心实验方法和设置

### 📚 数据集
在五个主流数学推理基准上评估，涵盖不同难度级别：
- **AIME24 & AIME25**：美国数学邀请赛题，竞赛级难题。
- **AMC23**：美国数学竞赛题，中等挑战。
- **MATH500**：精选500道数学题，覆盖代数、数论、几何、概率。
- **OlympiadBench**：双语奥赛级科学推理题，高阶复杂任务。

每题采样10个解，报告平均 accuracy 和平均 reasoning token 数。

### ⚙️ 实验设置
- **基础模型**：`DeepSeek-R1-Distill-Qwen-1.5B` 和 `7B`
- **训练流程三阶段**：
  1. **SFT**：在图剪枝后的简洁 CoT 上微调（使用 Light-R1 数据集）
  2. **DPO**：基于冗余评分构造偏好对（低冗余 vs 高冗余），提升偏好一致性
  3. **GRPO + length penalty**：在 dapo-17k 上进行强化学习，奖励 = 正确性 − λ × 归一化长度超量

- **评估指标**：
  - **Accuracy ↑**：最终答案正确率
  - **Average Reasoning Tokens ↓**：推理部分平均 token 数，衡量效率

### 🆚 对比的基线方法
| 类型 | 方法 |
|------|------|
| **效率导向 RL 方法** | EfficientReasoning, AdaptThink |
| **CoT 压缩方法** | O1-Pruner*, TokenSkip*（适配到目标模型） |
| **开源 R1 类模型** | Skywork-OR1, OREAL-7B, AReaL-boba-RL-7B, Light-R1-DS-7B |

> *注：O1-Pruner 和 TokenSkip 经作者适配至 DeepSeek-R1-Distill-Qwen 架构*

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 7B 模型为例）

| 方法 | 平均 Accuracy | 平均 Reasoning Tokens | 提升 vs Base |
|------|----------------|------------------------|-------------|
| **Base** | 59.72 | 8134 | — |
| **Ours** | **60.95** (+1.23) | **4660** (-42.7%) | ✅ 同时提准降耗 |

#### 在关键数据集上的表现：
| 数据集 | Base Acc → Ours Acc | Base Len → Ours Len |
|-------|--------------------|---------------------|
| **AIME25** | 29.00% → **31.67%** | 12779 → **6977** (-45.6%) |
| **OlympiadBench** | 56.77% → **59.85%** | 5252 → **3786** (-28.0%) |
| **MATH500** | 89.16% → **88.80%** | 3065 → **2080** (-32.1%) |
| **AMC23** | 81.00% → **82.34%** | 6849 → **3211** (-53.1%) |

✅ **结论**：在显著减少推理开销的同时，多数任务精度**稳定甚至提升**，说明冗余反思被有效去除而非误删关键逻辑。

### 🔍 消融实验（Ablation Study）
逐步添加模块的效果如下（Figure 3）：

| 阶段 | Accuracy 变化 | Token 数变化（归一化） |
|------|--------------|-------------------------|
| **Base** | 100% | 1.0 |
| **+ SFT (pruned CoT)** | +0.5pt | ↓15% |
| **+ DPO** | +0.8pt | ↓30% |
| **+ GRPO + length penalty** | **+1.23pt** | **↓42.7%** |

> 💡 发现：三阶段协同作用明显，**GRPO 阶段对效率压缩贡献最大**，而 DPO 成功引导模型偏好简洁路径。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **冗余反思是推理低效的重要根源**，特别是 Indiscriminate 和 Repetitive Reflection。
2. **图结构能有效揭示推理依赖关系**，使冗余节点可被精确定位。
3. **结构化剪枝 + 三阶段训练**可在不牺牲准确性前提下大幅压缩推理长度（**最高达 42.7%**）。
4. 模型行为发生正向转变：
   - 反思类 token（如 "wait", "maybe", "check"）频率显著下降；
   - 推进类连接词（如 "therefore"）增多，推理更直接高效（见 Figure 4）。

### ⚠️ 方法局限性
1. **依赖强教师模型构建图结构**：需调用外部 LLM（如 qwen-turbo）进行图解析，带来预处理开销（文中约 $20 成本）。
2. **progress-review 分类较粗粒度**：未能捕捉更细粒度的推理语义差异。
3. **泛化性待验证**：目前主要在数学类可验证任务上验证，是否适用于开放域（如创意写作、对话）尚不明确。

### 🔮 未来工作方向
- 设计轻量化图构建方法（如规则引擎或小型模型替代 LLM）。
- 引入更丰富的语义标签体系（如假设、反例、归纳等）。
- 探索在非数学领域（如代码生成、规划决策）的应用。
- 结合 early-exit 机制，在推理过程中动态判断是否需要继续反思。

---

## 总结
> 本文提出了一个新颖的 **graph-based CoT pruning 框架**，通过将线性推理转化为结构化图并设计双层级剪枝策略，成功识别并去除了两大类冗余反思行为。结合 SFT → DPO → GRPO 的三阶段训练范式，实现了**在保持甚至提升 accuracy 的同时，将 reasoning tokens 减少超过 40%**，为构建高效、稳健的 reasoning LLM 提供了新范式。

</details>

---

### 10. [Towards Intelligent Energy Security: A Unified Spatio-Temporal and Graph Learning Framework for Scalable Electricity Theft Detection in Smart Grids](https://arxiv.org/abs/2604.03344)

**Authors**: AbdulQoyum A. Olowookere, Usman A. Oguntola, Ebenezer. Leke Odekanle, Maridiyah A. Madehin, Aisha A. Adesope  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.03344v1  

#### Abstract
Electricity theft and non-technical losses (NTLs) remain critical challenges in modern smart grids, causing significant economic losses and compromising grid reliability. This study introduces the SmartGuard Energy Intelligence System (SGEIS), an integrated artificial intelligence framework for elec...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**智能电网中的电力盗窃检测**（Electricity Theft Detection）这一关键挑战展开研究。电力盗窃属于非技术性损耗（NTLs），导致严重的经济损失并威胁电网可靠性，尤其在发展中国家尤为严重。传统方法如人工巡检和规则系统效率低、难以扩展，而现有机器学习方法多集中于**时间序列分析**，忽视了电网的**空间拓扑结构**和**多源异构数据融合**。

### 提出的新方法与新思路
作者提出了名为 **SmartGuard Energy Intelligence System (SGEIS)** 的统一框架，其核心创新在于构建了一个**集成化、多维度的智能检测体系**，具体包括：

- **统一的时空图学习架构**：首次将 **Spatio-Temporal Modeling** 与 **Graph Learning** 深度结合，同时建模电力消耗的时间动态性和电网节点间的空间依赖关系。
- **混合异常检测机制**：
  - 时间维度：采用 **LSTM**、**Temporal Convolutional Network (TCN)** 和 **Autoencoder** 进行深度时序建模。
  - 分类维度：引入 **Random Forest**、**Gradient Boosting**、**XGBoost**、**LightGBM** 等集成学习模型进行监督分类。
  - 空间维度：利用 **Graph Neural Networks (GNNs)**（如 GCN、GAT）建模变压器-电表网络拓扑，识别跨节点传播的协同窃电行为。
- **增强可解释性**：集成 **Non-Intrusive Load Monitoring (NILM)** 模块，实现从总用电信号中解耦出电器级用电模式，提升检测结果的可解释性。
- **综合风险评分机制**：通过融合时间异常得分、分类概率和图节点风险得分，生成统一的风险评分，提升检测鲁棒性。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **全面性** | 超越单一时间或静态表格模型，整合时间、空间、统计与物理特征，提供更完整的电网智能视图。 |
| **鲁棒性** | 多模型融合降低误报率，尤其在处理不平衡数据和噪声方面表现更强。 |
| **可扩展性** | 框架设计支持大规模部署，适用于真实世界的智能电网环境。 |
| **实用性** | 引入 NILM 和图分析，为运维人员提供“为何可疑”的洞察，便于优先排查高风险区域。 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用一个**高分辨率智能电网数据集**，包含以下多源信息：
  - 电气参数：电压（Voltage）、电流（Current）、有功/无功功率（Power Consumption, Reactive Power）
  - 可再生能源输入：太阳能与风能发电量（Solar/Wind Power）
  - 环境变量：温度（Temperature）、湿度（Humidity）、电价（Electricity Price）
  - 运行状态：过载标志（Overload Condition）、故障信号（Fault Signal）
  - 时间戳：以15分钟为间隔聚合
- 数据经过清洗、缺失值填补（数值用中位数，类别用众数）和标准化处理。

### 实验设置与评估指标
#### 评估指标
由于数据高度不平衡（正常样本占比 ~81.75%，异常 ~18.25%），仅靠 Accuracy 不足以反映性能，因此采用以下综合指标：
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC**

#### 基线方法对比
论文对比了多种主流模型作为基线：
- **传统机器学习**：Random Forest, SVM
- **深度学习时序模型**：LSTM, TCN, Autoencoder
- **集成方法**：Gradient Boosting, XGBoost, LightGBM
- **图模型**：GCN, GAT
- 同时参考文献中已有研究（见 Table 1）进行横向比较。

#### 消融实验设计
虽然未明确列出“消融实验”章节，但在 **Section 4.7** 中通过模块组合分析实现了类似功能：
- 单独使用监督学习模型
- 加入时间序列模型（LSTM/TCN/Autoencoder）
- 再加入图神经网络（GNN）
- 最终整合所有组件形成完整 SGEIS 框架

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 表 2 & 表 3：监督学习模型性能对比

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 0.911 | 0.697 | 0.887 |
| **Gradient Boosting** | **0.910** | **0.693** | **0.894** ✅ |
| XGBoost | 0.909 | 0.694 | 0.885 |
| LightGBM | 0.910 | 0.697 | 0.894 ✅ |

- 所有模型准确率均达约 **91%**，表明特征工程有效。
- **Gradient Boosting 和 LightGBM 在 ROC-AUC 上达到最高 0.894**，显示其在区分正负样本方面的优越能力。

#### 图模型性能
- **GNN 模型在识别高风险节点上的准确率超过 96%**。
- 成功识别出多个高概率异常节点（如 Meter_451, Meter_322 等），并通过可视化展示其在网络中的聚集趋势，验证了窃电行为可能具有**空间传播性**。

#### 整体框架效果
- **SGEIS 框架通过融合时间、分类与图模型，显著提升了检测鲁棒性与可靠性**。
- Ensemble Anomaly Detection（至少两个模型一致判定为异常）有效减少了假阳性。
- NILM 成功实现冰箱等设备的负载分解（见 Fig. 7），证明了其在设备级异常识别上的潜力。

### 与基线方法对比结果
- 相较于仅使用时间序列或表格分类的方法，SGEIS 在 **F1-score 和 ROC-AUC 上均有提升**，尤其是在捕捉复杂、隐蔽的窃电模式上更具优势。
- 相比文献中多数单模型方法（如纯 LSTM 或 RF），本框架在处理**类不平衡**和**特征重叠**问题上表现更好（见 Fig. 5 显示两类样本存在明显重叠）。

---

## 4. 关键结论和发现

### 主要发现
1. **电力盗窃是时空联合现象**：不仅是时间上的异常波动，更是空间上互联节点间的协调行为，必须同时考虑 Temporal 与 Spatial 维度。
2. **混合建模优于单一模型**：结合 Supervised Learning、Deep Time-Series Models 与 GNN 的框架显著优于任何单独组件。
3. **特征工程至关重要**：构造的衍生特征（如 Supply-Demand Imbalance、Rolling Mean/Std、Apparent Power）极大增强了模型判别力。
4. **NILM 提升可解释性**：能够将异常追溯到具体电器行为，对实际运维极具价值。
5. **现实场景需多指标评估**：在高度不平衡的数据下，**ROC-AUC 和 F1-score 比 Accuracy 更具指导意义**。

### 方法的局限性
- **计算复杂度较高**：尤其是 GNN 与深度模型并行运行，带来较高的计算开销（文中提及 "High computational complexity"）。
- **依赖高质量拓扑数据**：GNN 性能受限于电网连接关系（Adjacency Matrix）的准确性。
- **标签依赖规则机制**：虽提出 Rule-Based Labeling，但仍可能存在误标风险，且未完全解决标注稀缺问题。
- **尚未在更大规模真实公用事业数据上验证**：当前实验基于特定数据集，泛化能力有待进一步检验。

### 未来工作方向
1. **引入实时流数据处理**：支持近实时监控与快速响应。
2. **探索 Temporal Graph Networks (TGNs)**：更好地建模动态变化的电网结构与时序行为。
3. **改进不平衡学习策略**：尝试 Cost-Sensitive Learning、Advanced Resampling（如 SMOTE-GAN）等方法优化少数类检测。
4. **扩展 NILM 至多设备类型**：提升负载分解粒度，增强诊断能力。
5. **联邦学习与隐私保护集成**：结合 Federated Learning（如 FedDetect）实现在不共享原始数据下的跨区域协同检测。
6. **真实场景部署验证**：与电力公司合作，在真实 Smart Grid 环境中测试 SGEIS 的实用性与经济效益。

---

> **总结**：  
> 本文提出的 **SGEIS 框架**代表了电力盗窃检测领域的一次重要推进——从“孤立分析”迈向“系统智能”。它不仅提高了检测精度，更重要的是构建了一个**可解释、可扩展、融合时空感知的智能能源安全平台**，为下一代智能电网的安全管理提供了坚实的技术基础。

</details>

---

### 11. [BlazeFL: Fast and Deterministic Federated Learning Simulation](https://arxiv.org/abs/2604.03606)

**Authors**: Kitsuya Azuma, Takayuki Nishio  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.03606v1  

#### Abstract
Federated learning (FL) research increasingly relies on single-node simulations with hundreds or thousands of virtual clients, making both efficiency and reproducibility essential. Yet parallel client training often introduces nondeterminism through shared random state and scheduling variability, fo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# BlazeFL: Fast and Deterministic Federated Learning Simulation 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Federated Learning (FL) 研究中，单节点仿真常用于算法原型设计、超参数搜索和消融研究。然而，随着虚拟客户端数量增加至数百甚至数千，**效率**（throughput）和**可复现性**（reproducibility）成为两大挑战：

- **效率瓶颈**：传统基于 multiprocessing 或分布式运行时（如 Ray）的方法引入了进程间通信（IPC）和参数序列化的开销，尤其在通信密集型任务中显著拖慢训练速度。
- **非确定性问题**：并行执行下，由于共享随机状态（RNG）、调度顺序差异以及浮点累加顺序不一致，即使固定随机种子也无法保证多次运行结果完全相同（bitwise-identical），严重影响实验分析的可靠性。

### 提出了什么新方法或新思路
作者提出 **BlazeFL** —— 一个轻量级、面向单节点 FL 仿真的框架，其核心创新在于结合两个关键技术：

- **Free-threaded Shared-Memory Execution**  
  利用 Python 的 free-threading 架构（PEP 703 / PEP 779），将所有 client 作为 worker threads 在单一进程中执行，通过 shared memory 进行模型参数交换，避免跨进程序列化与 IPC 开销。

- **Controlled Deterministic Execution via Isolated RNG Streams**  
  为每个 client 分配独立的 RNG 流（random number generator stream），确保 client 局部的随机行为（如数据增强、dropout、mini-batch shuffle）不受调度顺序影响；同时 server 按照采样顺序消费结果，避免 completion-order-dependent aggregation 引入的浮点误差累积。

### 相比现有方法的优势
| 维度 | BlazeFL | 传统方法（如 Flower + Ray） |
|------|--------|-----------------------------|
| **执行效率** | 显著降低通信开销，尤其在通信密集型场景下 | 存在 IPC 和序列化开销 |
| **可复现性** | 支持 bitwise-identical 多次运行结果 | 即使设全局 seed 仍存在微小差异 |
| **依赖复杂度** | 轻量级，仅依赖标准库 + PyTorch | 依赖外部调度器（如 Ray、MPI） |
| **接口灵活性** | 使用 `typing.Protocol` 实现低耦合，易于集成已有 PyTorch 代码 | 需继承特定基类或适配框架生命周期 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CIFAR-10**：图像分类任务，共 10 类。
- 数据划分方式：non-IID 设置，每 client 分配两个类别（two classes per client），模拟真实联邦场景下的数据异质性。

### 实验设置
- 客户端总数：100
- 通信轮数（communication rounds）：5
- 每轮每个选中 client 本地训练：5 epochs，500 samples
- 全局评估：每轮后在 10,000 测试样本上进行
- 并行度 $ P \in \{1, 2, 4, 8, 16, 32, 64\} $
- 模型架构对比：
  - Lightweight CNN
  - ResNet-18, ResNet-50, ResNet-101

### 评估指标
1. **Wall-clock time**：五轮 FL loop 的总运行时间（不含数据下载与划分）
2. **Throughput scalability**：随并行度提升的时间变化趋势
3. **Reproducibility**：
   - 最终准确率的标准差（Final Acc. Std. Dev.）
   - 每轮全局模型权重的 SHA-256 哈希一致性（round-wise hash agreement）

### 基线方法对比
| 配置 | 描述 |
|------|------|
| **BlazeFL (free-threaded)** | 主要提案，使用 free-threading + shared memory + isolated RNG |
| **BlazeFL (process-based)** | 对照组，multiprocessing + shared memory tensors（torch.multiprocessing） |
| **Flower (Ray backend)** | 当前主流开源框架代表，使用 Ray 实现分布式仿真 |

> 注：因框架生态限制，BlazeFL 使用 Python 3.14.3（支持 free-threading），Flower 使用 Python 3.13.7，解释器版本不同但性能差距主要归因于架构而非语言版本。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）运行效率（Wall-clock Time）
在高性能服务器（H100 GPU, 48 CPU cores）上的表现如下：

| 模型 | 最大加速比（vs. Flower） |
|------|--------------------------|
| CNN | **3.1×** |
| ResNet-18 | **1.4×** |
| ResNet-50 | **1.1×** |

- 加速效果在 **通信密集型 workload**（如轻量 CNN）中最明显；
- 随着模型变大（计算占比上升），优势缩小但仍存在；
- BlazeFL (process-based) 表现介于两者之间，说明 **process management 本身仍有额外开销**。

#### （2）可复现性测试（Workstation Server, P=32, 10 runs）

| 配置 | Final Acc. Std. Dev. | Round-wise Hash Agreement |
|------|------------------------|----------------------------|
| Flower (no seed control) | 1.24 pp | ❌ |
| Flower (global seed) | 0.18 pp | ❌ |
| **BlazeFL (process-based)** | **0.00 pp** | ✅ |
| **BlazeFL (free-threaded)** | **0.00 pp** | ✅ |

> ✅ 表示所有运行间哈希完全一致，即实现 **bitwise-identical** 执行。

#### （3）跨并行度可复现性（BlazeFL free-threaded）
| 并行度 $ P $ | vs. $ P=1 $ Final Acc. Δ | Hash Agreement |
|---------------|----------------------------|----------------|
| 1~64 | 0.0 pp | ✅ 所有 round 哈希匹配 |

→ 表明 BlazeFL 的执行结果对并行度变化 **完全不变**，进一步验证其 determinism 设计的有效性。

#### （4）Flower 中的发散诊断
通过对 logits 输出的 L2 距离分析发现：
- 第一轮聚合后即出现 ~1e-6 级别的微小差异；
- 差异随通信轮次逐步放大（compounding effect）；
- 可视化显示轨迹“扇形展开”（fan out），符合 **completion-order-dependent floating-point accumulation** 导致的非确定性。

---

## 4. 关键结论和发现

### 主要发现
1. **共享内存 + free-threading 可显著提升单节点 FL 仿真效率**，尤其适用于通信密集型任务（如小模型、高频通信）。
2. **隔离 RNG 流 + 固定消费顺序** 是实现高并发下 bitwise-identical 执行的关键，解决了现有框架难以复现的问题。
3. BlazeFL 在相同硬件/软件栈下，无论并行度如何变化，均可产生完全一致的结果，极大增强了实验可信度。
4. 在 VRAM 受限设备上，heavy computation + high concurrency 场景可能触发 CUDA allocator 锁竞争，此时 process-based 模式反而更优。

### 方法的局限性
| 局限性 | 说明 |
|-------|------|
| **单节点限定** | 不适用于多机分布式训练或生产部署，定位是本地研究工具。 |
| **跨平台不可复现** | 仅保证同一软硬件环境下的 bitwise-identical，不同平台/库版本可能导致数值差异。 |
| **vision pipeline 兼容性要求** | 若自定义 transform 内部使用全局 RNG（如某些 torchvision ops），需手动适配以接受显式 generator 才能保持 determinism。 |
| **生态系统成熟度限制** | 依赖 Python free-threading 新特性，部分第三方库尚未兼容，影响通用性和公平比较。 |

### 未来工作方向
- 将 free-threading 思路扩展到其他 ML 仿真场景（如 RL、multi-agent systems）；
- 探索 hybrid 模式：在内存充足时用 free-threading，在 compute-heavy 场景自动 fallback 到 process-based；
- 构建标准化的 deterministic vision data pipeline 工具链；
- 推动更多库对 free-threaded Python 的支持，提升生态兼容性。

---

> 🔗 开源地址：[https://github.com/kitsuyaazuma/blazefl](https://github.com/kitsuyaazuma/blazefl)  
> 📄 发表信息：已被 FedVision @ CVPR 2026 (CVPRW) 接收，© 2026 IEEE

</details>

---

### 12. [APPA: Adaptive Preference Pluralistic Alignment for Fair Federated RLHF of LLMs](https://arxiv.org/abs/2604.04261)

**Authors**: Mahmoud Srewa, Tianyu Zhao, Salma Elmalaki  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.04261v1  

#### Abstract
Aligning large language models (LLMs) with diverse human preferences requires pluralistic alignment, where a single model must respect the values of multiple distinct groups simultaneously. In federated reinforcement learning from human feedback (FedRLHF), these groups align a shared policy without ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# APPA: Adaptive Preference Pluralistic Alignment for Fair Federated RLHF of LLMs — 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Federated Reinforcement Learning from Human Feedback (FedRLHF)** 场景下，如何公平地对齐 **Large Language Models (LLMs)** 与多个异构群体（如不同国家、文化、人口统计群体）的多样化偏好是一个关键挑战。传统聚合方法存在明显缺陷：
- **Average aggregation**：导致“多数主导”，系统性地忽视表现最差的群体（worst-performing groups），加剧 **majority bias**。
- **Min aggregation**：虽提升最差群体的表现，但牺牲整体对齐效果（overall alignment），且忽略其他群体的有用反馈。

因此，核心问题是：**如何在不集中原始偏好数据的前提下，实现公平且高效的多群体奖励聚合？**

### 提出了什么新方法或新思路
作者提出 **APPA (Adaptive Preference Pluralistic Alignment)**，一种用于 **Fair FedRLHF** 的自适应奖励聚合框架。其核心思想是：
- 动态重加权各群体的奖励（dynamic reweighting），优先关注历史对齐度较低的群体。
- 仅依赖于群体级别的奖励信号（group-level rewards），无需访问原始偏好数据，保护隐私。
- 引入 **Fairness Index (FI)** 和阈值机制，决定何时启用自适应加权。

#### 关键技术组件：
1. **Adaptive Alpha Aggregation**：
   - 改进 Park et al. (2024) 的 α-aggregation，将全局固定标量 α 替换为 **每群体动态更新的权重**。
   - 使用 **reverse softmax** 基于历史对齐得分计算权重：历史表现越差，当前权重越高。

2. **Fairness Index (FI)**：
   - 基于群体间奖励的变异系数（Coefficient of Variation, CoV）定义，量化跨群体的不公平程度。
   - 当 FI < 0.99 时启用自适应聚合；否则回退到平均聚合，防止训练后期过度调整。

3. **Exponential Moving Average (EMA) 历史追踪**：
   - 维护每个群体的历史对齐得分 $ h_g^t $，平滑短期波动，提供长期趋势估计。

### 相比现有方法的优势
| 特性 | Average Aggregation | Min Aggregation | APPA (Ours) |
|------|---------------------|------------------|-------------|
| 平均对齐 (Avg AS) | 高 | 低 | ✅ 更高或相当 |
| 最差群体对齐 (Min AS) | 低 | 中等 | ✅ 显著更高 |
| 公平性 (FI) | 低 | 高 | ✅ 高且稳定 |
| 利用所有群体信息 | ✅ 是 | ❌ 否（只用最差组） | ✅ 是（非零权重） |
| 自适应性 | ❌ 固定 | ❌ 固定 | ✅ 动态调整 |
| 数据隐私 | ✅ 满足 | ✅ 满足 | ✅ 满足 |

> ✅ APPA 成功平衡了 **worst-group performance** 与 **overall alignment**，避免了二者之间的 trade-off。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **GLOBALQA**：基于 Pew Research 的全球态度调查，涵盖来自多个国家（如美国、德国、印度等）的受访者。测试 **跨国家（cross-national）** 多样化偏好建模能力，答案多为 **名义型（nominal）**。
- **OQA (OpinionQA)**：美国国内按人口统计特征划分的群体（如宗教、收入、教育水平等）。测试 **国内多样性（intra-national demographic）** 对齐，答案多为 **序数型（ordinal Likert scale）**。

### 实验设置
- **模型家族**：
  - `Gemma-2-2B`
  - `Llama-3.2-3B`
  - `Qwen3-0.6B`
- **任务类型**：
  1. **Distributional Preference Alignment (DPA)**：预测群体对各选项的偏好分布（概率向量）。
     - 主要指标：Jensen-Shannon divergence (JS)，Wasserstein distance (Was.)
  2. **Ordinal Preference Alignment (OPA)**：输出选项的排序（ranking）。
     - 主要指标：Borda score
- **评估流程**：
  - 使用 **PluralLLM** 作为冻结的轻量级联邦偏好预测器，生成每个群体的目标分布。
  - 政策模型输出预测分布或排名，与目标比较得到 per-group reward。
  - 在服务器端进行奖励聚合后执行 PPO 更新。

### 评估指标
- **Per-group Alignment Score (AS)**：每个群体在测试集上的平均奖励。
- **Avg AS**：所有群体 AS 的均值。
- **Min AS**：所有群体 AS 的最小值（反映最弱势群体表现）。
- **Fairness Index (FI)**：基于 CoV 定义，范围 [0,1]，越接近 1 表示群体间差异越小，越公平。
- **Format Score**：衡量输出格式正确率（如是否输出 K 个数字、是否归一化等）。

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **SFT** | 在多数派标签上微调，学习模态偏好，天然偏向主流群体。 |
| **Average** | 所有群体奖励取均值，平等对待但易受 majority bias 影响。 |
| **Min** | 取最差群体的奖励作为优化目标，极端关注 worst-case，可能牺牲整体性能。 |
| **APPA (Ours)** | 自适应加权，动态提升低对齐群体权重，保持所有群体参与。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

#### GLOBALQA – DPA (JS 指标)
| Model | 方法 | Avg AS (JS) | Min AS (JS) | FI (JS) |
|-------|------|-------------|------------|---------|
| Gemma-2-2B | Average | 0.839 | 0.812 | 0.9695 |
| | **APPA** | **0.861** | **0.843** | **0.9994** |
| | Min | 0.834 | 0.808 | 0.9995 |
| → APPA 将 **Min AS 提升 +3.5 pts**，同时 **Avg AS 超过 Min 和 Average**。

#### GLOBALQA – OPA (Borda 指标)
| Model | 方法 | Avg AS (Bor.) | Min AS (Bor.) | FI (Bor.) |
|-------|------|----------------|---------------|-----------|
| Gemma-2-2B | Average | 0.469 | 0.359 | 0.8539 |
| | **APPA** | **0.511** | **0.461** | **0.8911** |
| | Min | 0.475 | 0.420 | 0.8725 |
| → APPA 实现 **最高 Avg 和 Min AS**，且 **FI 最高**，全面领先。

#### OQA – DPA (Wasserstein 指标)
| Model | 方法 | Avg AS (Was.) | Min AS (Was.) | FI (Was.) |
|-------|------|----------------|----------------|-----------|
| Llama-3.2-3B | Average | 0.859 | 0.824 | 0.9645 |
| | **APPA** | **0.872** | **0.841** | **0.9940** |
| | Min | 0.856 | 0.811 | 0.9937 |
| → APPA 在 **Avg、Min 和 FI 上均最优**。

> 💡 **例外情况**：对于 `Qwen3-0.6B` 在 OQA DPA 上，**Min 方法获得了更高的 Avg AS (0.823 vs 0.780)**，作者认为这是由于小模型容量限制，导致 Wasserstein 梯度信号不足，削弱了自适应权重的有效性。

### 与基线方法的对比结果
- **相比 Average**：
  - Worst-group alignment 提升高达 **28%**。
  - FI 显著提高（普遍 >0.99 vs ~0.96–0.97）。
- **相比 Min**：
  - 在大多数配置下，**APPA 同时实现了更高的 Avg AS 和 Min AS**。
  - 避免了 Min 方法“牺牲整体以保底线”的代价。
- **可视化分析（Figure 2 & 3）**：
  - **Spider Plot** 显示 APPA 的多边形更大更圆，表明增益分布更均匀。
  - **Fairness-Alignment Trade-off 图** 显示 APPA 占据右上角 Pareto 前沿区域，优于其他方法。

### 消融实验结果（隐含分析）
虽然未明确列出消融表，但从设计可推断以下关键组件的作用：
- **EMA 历史追踪**：确保权重更新稳定，避免因单次低奖励而被永久高赋权。
- **Reverse Softmax + Low Temperature (T=0.1)**：增强对落后群体的关注强度。
- **FI Threshold (τ=0.99)**：防止训练后期不必要的扰动，提升稳定性。
- 若移除这些机制，预期会观察到：
  - 更剧烈的训练波动
  - FI 下降或收敛变慢
  - Min AS 提升但 Avg AS 下降

---

## 4. 关键结论和发现

### 主要发现
1. **奖励聚合方式显著影响 FedRLHF 的公平性与效率**。
2. **APPA 能有效缓解 majority bias 和 minimax 的过度保守问题**，实现双赢。
3. **通过历史对齐状态动态调整权重，可在不牺牲整体性能的前提下大幅提升最弱群体表现**。
4. **该方法在多种模型规模（0.6B–3B）、数据集（GLOBALQA/OQA）、任务类型（DPA/OPA）上具有鲁棒性和泛化能力**。
5. **最终高 FI 值表明 APPA 成功推动模型达到跨群体高度一致的高质量对齐状态**。

### 方法的局限性
1. **依赖 PluralLLM 提供群体偏好分布**：若 PluralLLM 本身存在偏差，则会影响整个 pipeline。
2. **小模型上性能受限**：如 Qwen3-0.6B 在某些任务中未能超越 Min 方法，说明梯度信号强度可能制约自适应机制的效果。
3. **仅适用于结构化输出任务**：目前实验集中在多选题和排序任务，尚未验证于开放生成场景。
4. **超参数敏感性未知**：EMA 衰减率 λ、温度 T、阈值 τ 固定使用，未进行广泛调参。

### 未来工作方向
1. **扩展至开放文本生成任务**（如长文本创作、代码生成），探索更丰富的群体偏好建模方式。
2. **设计适用于非结构化输出的 group-level reward signals**。
3. **结合 DPO 或其他 reward-free 方法**，进一步简化训练流程。
4. **研究个性化与公平性的联合优化机制**，支持个体与群体双重对齐。
5. **在真实世界部署中评估社会影响**，例如政策建议、新闻摘要中的文化代表性。

--- 

> ✅ **总结一句话**：  
> **APPA 通过引入基于历史表现的自适应奖励加权机制，在 FedRLHF 中实现了更公平、更高效、更稳健的多群体对齐，显著优于传统的平均或最小化聚合策略。**

</details>

---

### 13. [ResearchEVO: An End-to-End Framework for Automated Scientific Discovery and Documentation](https://arxiv.org/abs/2604.05587)

**Authors**: Zhe Zhao, Haibin Wen, Jiaming Ma, Jiachang Zhan, Tianyi Xu, Ye Wei, Qingfu Zhang  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.05587v1  

#### Abstract
An important recurring pattern in scientific breakthroughs is a two-stage process: an initial phase of undirected experimentation that yields an unexpected finding, followed by a retrospective phase that explains why the finding works and situates it within existing theory. We present ResearchEVO, a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ResearchEVO: An End-to-End Framework for Automated Scientific Discovery and Documentation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前自动化科学研究系统存在显著割裂：
- **Evolution Systems**（如FunSearch、AlphaEvolve）能通过LLM驱动的进化搜索发现高性能算法，但输出仅为代码片段，缺乏科学解释、文献关联和可发表的研究叙事。
- **Writing Systems**（如AI Scientist、CycleResearcher）能生成完整科研论文，但其“发现”源于LLM参数记忆中的知识重组，而非基于真实实验反馈的**原则性算法搜索**，易产生幻觉引用和虚构结果。

该论文指出，这一割裂导致现有系统无法复现真实科学突破中“先发现、后解释”（discover-then-explain）的经典范式。

### 提出了什么新方法或新思路
提出 **ResearchEVO**，首个端到端（end-to-end）实现“发现→解释→文档化”全链条自动化的框架，包含两个核心阶段：

#### （1）Evolution Phase：双维度协同进化（Bi-Dimensional Co-Evolution）
- **功能维度（Functional Dimension）**：优化模块内部逻辑（如数学公式、条件分支）。
- **结构维度（Structural Dimension）**：优化整体架构（控制流、模块组织、新增/删除计算阶段）。
- 采用 **LLM-guided evolution**，无需预定义模板（template-free），通过短时反思（short-term reflection）和长时合成（long-term synthesis）指导搜索方向。
- 引入**领域自适应沙箱评估**（domain-adaptive sandbox evaluation），返回堆栈追踪、部分输出等结构化错误反馈，用于纠正后续代次。

#### （2）Writing Phase：基于检索增强生成的科学写作
- 输入为进化阶段产出的最佳算法 $W^*$。
- 三阶段流程：
  1. **文献爬取与向量索引**：从种子参考文献出发，扩展构建Chroma向量数据库。
  2. **自动实验设计与执行**：生成分析视角（如消融研究、超参扫描），编写并运行实验脚本，可视化结果。
  3. **句子级RAG写作 + 反幻觉验证**：每句话独立进行密集检索（dense retrieval）与交叉编码器重排序（cross-encoder reranking），确保每个 `\cite{}` 键均对应数据库中真实条目，**强制杜绝虚构引用**。

### 相比现有方法的优势
| 维度 | ResearchEVO | 其他系统 |
|------|-------------|----------|
| **算法演化自由度** | ✅ 双维度协同进化（功能+结构） | ❌ 多限于单函数优化或固定架构 |
| **科学文档生成** | ✅ 完整论文（含实验、图表、引用） | ❌ 仅代码 或 虚构性论文 |
| **引用真实性保障** | ✅ 句子级RAG + 显式反幻觉验证 | ❌ 文档级检索，易幻觉 |
| **发现来源** | ✅ 基于fitness的真实搜索 | ❌ 来自LLM记忆的知识重组 |
| **输出形式** | ✅ 编译通过的LaTeX/PDF论文 | ❌ 仅文本草稿 |

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **Quantum Error Correction (QEC)**  
   - 数据来源：Google量子硬件真实探测事件数据（来自[2]）。
   - 配置：表面码（surface code, d=3），8个测量轮次（R ∈ {3,5,...,17}），4个空间中心，共32种配置。
   - 任务：优化MWPM解码器的边权重函数。

2. **Physics-Informed Neural Networks (PINN)**  
   - 数据来源：DeepXDE库提供的三个2D泊松方程基准：
     - `Poisson2D_Classic`
     - `PoissonBoltzmann2D`
     - `Poisson2D_ManyArea`
   - 任务：优化LRA框架下的自适应损失加权函数以提升训练稳定性。

### 实验设置和评估指标

#### Evolution Phase 设置
- 初始种群大小 $N_{init}=10$，活跃种群 $N=10$
- 进化迭代次数 $T=20$
- 沙箱超时时间：30秒/次
- LLM模型：GPT-4o
- 选择机制：基于排名的概率选择（$p \propto 1/(rank+1+|P|)$）

#### Writing Phase 设置
- 文献数据库：Semantic Scholar + arXiv，使用BAAI/bge-m3嵌入 + Chroma索引
- 检索：Top-20稠密检索 → FlagEmbedding交叉编码器重排序 → Top-8用于生成
- 实验脚本最多重试3次
- 生成论文需编译为PDF且无引用错误

#### 评估指标
| 领域 | 主要指标 | 辅助/诊断指标 |
|------|--------|--------------|
| QEC | Logical Error Rate (LER) ↓ | Bootstrap CI, Paired Sign Test, 改进一致性（improvement consistency） |
| PINN | L2 Relative Error (L2RE) ↓ | Avg95RelUpdate ↓, AGIR ↓, MSCR ↓ |

### 基线方法对比
| 领域 | 基线方法 |
|------|---------|
| QEC | 标准MWPM（均匀/距离权重） |
| PINN | LRA [31], Adam [13] |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### QEC 结果（表4）
| 测量轮次 R | MWPM (LER) | DOA-MWPM (LER) | Δ_abs | Δ_rel |
|-----------|------------|----------------|-------|-------|
| 3         | 0.089      | 0.088          | -1.1e-3 | +0.9% |
| 5         | 0.155      | 0.154          | -1.1e-3 | +0.7% |
| 7         | 0.211      | 0.208          | -2.7e-3 | +1.3% |
| 9         | 0.254      | 0.250          | -3.3e-3 | +1.3% |
| ...       | ...        | ...            | ...   | ...   |
| **Mean**  | **0.257**  | **0.254**      | **-2.1e-3** | **+0.9%** |

- 所有8个轮次均取得改进，相对LER降低 **0.4%–1.3%**。
- 改进具方向一致性：6/8轮次中，全部4个中心同时改善。
- Bootstrap置信区间与配对符号检验均显示统计显著性。

#### PINN 结果（图3）
- 在三个PDE基准上，**ResLRA-PINN** 均优于LRA和Adam基线：
  - 中位L2RE更低
  - Avg95RelUpdate 更小（更新更稳定）
  - AGIR 和 MSCR 更低（梯度失衡与曲率更优）
- 性能提升与稳定性指标强相关（见图3b）。

### 与基线方法的对比结果
- **QEC**：DOA-MWPM在所有配置下均优于标准MWPM，且改进具有物理可解释性（边界、可观测量连接、孤立检测器等）。
- **PINN**：ResLRA-PINN在L2RE上全面胜出，并显著提高训练稳定性。

### 消融实验结果（PINN）
- 移除**信任区域约束**（Eq. 8）→ Avg95RelUpdate 和 MSCR 明显上升，L2RE恶化。
- 移除**裁剪操作**（Eq. 7）→ AGIR升高，L2RE轻微退化。
- 移除**残差连接**（Eq. 9）→ AGIR、MSCR上升，L2RE明显增加。
- 表明两个组件（trust-region adaptor + residual backbone）各自独立贡献，共同提升性能。

---

## 4. 关键结论和发现

### 主要发现
1. **成功实现“发现→解释”闭环**：
   - Evolution Phase 在无任何领域知识输入的情况下，**盲搜**出人类可解释的新机制：
     - QEC中：DOA-MWPM 自动发现了边界敏感、可观测连接、孤立节点等拓扑特征的重要性。
     - PINN中：ResLRA-PINN 发现了信任区域控制与残差连接对稳定训练的关键作用。
   - Writing Phase 成功将这些“盲发现”回溯至已有理论：
     - 将DOA因子与T1/T2弛豫不对称性、测量误差传播等联系起来。
     - 将信任区域约束与Levenberg-Marquardt等经典优化方法建立联系。

2. **生成论文具备高质量科学属性**：
   - 自动生成非平凡实验设计（如Bootstrap CI用于n=4场景）。
   - 设计新型诊断指标（如Avg95RelUpdate, AGIR, MSCR）提供机制性解释。
   - 所有引用均经验证存在于索引库中，**零虚构引用**。

3. **方法具备跨学科泛化能力**：
   - 在量子计算（QEC）与科学机器学习（PINN）两个截然不同领域均取得有效发现。
   - 验证了框架对真实硬件数据（Google量子芯片）和复杂PDE求解的有效性。

### 方法的局限性
1. **计算成本高**：LLM调用数千次，全流程耗时数小时。
2. **依赖初始文献质量**：Writing Phase的文献定位效果受限于种子参考文献B的质量。
3. **未经历正式同行评审**：生成论文尚未提交期刊接受实际审稿考验。
4. **顺序流水线设计**：当前两阶段为串行，缺乏写作反馈引导进化的闭环机制。
5. **模板依赖性**：LaTeX生成仍需一定格式模板支持，完全自由格式尚难保证。

### 未来工作方向
1. **紧耦合反馈机制**：让Writing Phase的分析洞察（如识别理论缺口）反向指导Evolution Phase的搜索方向。
2. **多目标优化**：在准确率、效率、可解释性之间进行Pareto前沿探索。
3. **扩展至实验科学**：应用于湿实验（wet-lab）协议设计与生物实验自动化。
4. **正式同行评审验证**：推动生成论文投稿并接受真实学术社区评价。
5. **轻量化与开源推广**：进一步降低资源需求，促进学术界广泛复现与应用。

--- 

> **总结一句话**：  
> ResearchEVO 是首个真正实现“从问题定义到可发表论文”全自动化的科研AI系统，它通过**双维度协同进化**发现新算法，并利用**句子级RAG+反幻觉验证**自动生成可信、可读、可复现的科学叙述，标志着自动化科学研究迈向“完整科学生命周期”的重要一步。

</details>

---

### 14. [Do Domain-specific Experts exist in MoE-based LLMs?](https://arxiv.org/abs/2604.05267)

**Authors**: Giang Do, Hung Le, Truyen Tran  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.05267v1  

#### Abstract
In the era of Large Language Models (LLMs), the Mixture of Experts (MoE) architecture has emerged as an effective approach for training extremely large models with improved computational efficiency. This success builds upon extensive prior research aimed at enhancing expert specialization in MoE-bas...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Do Domain-specific Experts exist in MoE-based LLMs?》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文探讨了一个基础性问题：**在基于 Mixture of Experts (MoE) 架构的大型语言模型（LLMs）中，是否存在“领域特定专家”（domain-specific experts）？**  
尽管已有大量研究致力于提升 MoE 中专家的专业化程度，但这些“专业化”的本质及其可解释性仍不明确。

### 🚀 提出的新方法与新思路
作者提出了 **Domain Steering Mixture of Experts (DSMoE)**，一种**无需训练**（training-free）、**零推理开销**的框架，用于增强 MoE-based LLM 在目标领域的表现。

其核心思想是：
- 定义并识别 **domain-specific tokens** 和 **domain-specific experts**。
- 通过动态调整路由器（router）权重，放大与目标领域相关的专家的影响，从而实现“领域引导”。

### 🔍 相比现有方法的优势
| 方面 | DSMoE | 现有方法（如 RICE、SFT） |
|------|-------|------------------------|
| 是否需要重新训练 | ❌ 否 | ✅ 是（SFT 需要微调） |
| 推理成本增加 | ❌ 无额外开销 | ✅ RICE 每样本需实时计算 |
| 泛化能力 | ✅ 强，在非目标域也有效 | ⚠️ RICE 仅适用于推理密集型模型 |
| 数据依赖 | ❌ 不需要标注数据 | ✅ SFT 严重依赖高质量训练数据 |

> ✅ **核心优势**：DSMoE 在不修改模型参数、不增加推理时间的前提下，显著提升了多个领域上的性能，甚至优于 Supervised Fine-Tuning（SFT）等强基线。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 描述 | 特点 |
|--------|------|------|
| **MMLU-Pro** | 多任务理解基准，涵盖 STEM、法律、人文等领域 | 12K 个问题，10 选项设计减少随机猜测影响 |
| **GPQA Diamond** | 博士级别科学问答数据集 | 极高难度，人类专家准确率约 65% |
| **AIME** | 美国数学邀请赛题目集合 | 数学竞赛题，要求多步推理，答案为整数（0–999） |

### ⚙️ 实验设置与评估指标
- **评估模型范围**：共测试 **10 个开源 MoE-based LLMs**，参数量从 **3.8B 到 120B**，包括：
  - `PhiMoE-Tiny`, `OLMoE`, `Qwen1.5-MoE`, `DeepSeek-MoE`, `GPT-OSS-20B/120B` 等。
- **评估指标**：主要使用 **Accuracy (%)**。
- **目标领域**：聚焦于 **Math, Biology, Physics, Chemistry** 四个 STEM 领域。
- **domain-specific token 识别**：采用梯度归因法（gradient-based attribution），避免 Leave-One-Out 的高计算成本。
- **domain-specific expert 识别**：基于公式 $ g(e_i) = P(e_i|D) \cdot [P(s \in S|e_i) - P(s \in C|e_i)] $

### 🆚 基线方法对比
| 基线方法 | 类型 | 说明 |
|---------|------|------|
| **Original MoE Models** | 原始模型 | 未进行任何干预 |
| **RICE (Wang et al., 2025a)** | 训练自由 steering 方法 | 聚焦“thinking experts”，每样本在线 steering |
| **Supervised Fine-Tuning (SFT)** | 微调范式 | 使用 LoRA 进行参数高效微调（约 2.7% 参数可更新） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Tables 2–4）

#### ✅ 在 MMLU-Pro 上的表现（Table 2）
| 模型 | 平均提升（DSMoE vs Original） |
|------|-------------------------------|
| Qwen3-30B-Instruct | +1.5 pp |
| GPT-OSS-120B | **+14.5 pp** |
| Qwen3-30B-Thinking | +3.6 pp |
| GPT-OSS-20B | +3.7 pp |

> 💡 在 **Chemistry** 子任务上，GPT-OSS-120B 的 DSMoE 实现了高达 **+29.1%** 的绝对增益！

#### ✅ 在 GPQA Diamond 上的泛化能力（Table 3）
| 模型 | 平均提升 |
|------|--------|
| GPT-OSS-20B | **+27.1 pp** |
| 其他模型 | +4.8 ~ +10.4 pp |

> 表明 DSMoE 可有效迁移到更难、未见过的博士级科学问题。

#### ✅ 在 AIME 数学竞赛题上的表现（Table 4）
| 模型 | 提升幅度 |
|------|--------|
| GPT-OSS-20B (AIME 2024) | **+27.3 pp** |
| GPT-OSS-20B (AIME 2025) | +12.3 pp |
| Qwen3-30B-Instruct | +13.3 pp |

> 显示 DSMoE 对复杂数学推理任务具有强大增强效果。

### 🔬 消融实验结果

#### （1）不同数量的 domain-specific experts（K）的影响（Table 5）
- 测试模型：GPT-OSS-20B（Biology 领域）
- 最佳配置：**K=20**（约占总专家数的 1%）
- 结果趋势：先升后降，过多专家反而降低性能 → 存在最优稀疏性。

#### （2）steering coefficient α 的影响（Table 6）
- 最优值：**α = 5.0**
- 过小（如 0.1）→ 引导不足；过大（如 50.0）→ 过度约束，破坏多样性。
- 推荐范围：**[2.0, 5.0]** 内效果稳定。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **存在 domain-specific experts**：
   - 经过对 10 个 MoE-based LLM 的系统分析，实证表明：**确实存在对特定领域高度敏感且功能专精的专家模块**。
   - 可视化热图（Figures 3–6）显示某些层中出现明显的专家集群。

2. **DSMoE 高效且通用**：
   - 作为一种 **training-free** 方法，DSMoE 在多个领域、多种规模模型上均取得显著提升。
   - 性能超越 SFT，说明利用内在专家结构可能比外部微调更高效。

3. **具备强泛化能力**：
   - 在非目标数据集（如 GPQA、AIME）上依然表现优异，证明其不是过拟合，而是真正激活了领域相关知识路径。

4. **计算效率极高**：
   - 识别 domain-specific experts 为一次性离线操作（O(L)），远低于 RICE 的 O(L×M)。
   - 推理阶段无额外延迟，适合部署。

### ⚠️ 局限性
- 当前实验受限于算力，最大只验证到 **120B 参数模型**（GPT-OSS-120B），尚未扩展至 400B+ 规模。
- 尚未在更多非 STEM 领域（如法律、医学）充分验证。
- 依赖预定义的领域文本（如 MMLU-Pro）来提取 domain-specific tokens，若初始数据偏差大，则可能误导专家选择。

### 🔮 未来工作方向
- 扩展至更大规模模型（>400B）及更多推理专用架构（如 DeepSeek-R1）。
- 探索自动化的 domain-specific token 发现机制。
- 结合动态 routing 机制，实现自适应领域切换。
- 研究 bias mitigation 策略，防止因 web-sourced 数据带来的性别、种族偏见被放大。

---

## ✅ 总结一句话
> 本文首次系统证实了 **MoE-based LLM 中存在 domain-specific experts**，并提出 **DSMoE** —— 一个无需训练、零开销、高性能的领域引导框架，在多项挑战性基准上超越 SFT 与 RICE，揭示了**挖掘模型内部结构潜力**作为替代传统微调的新范式。

</details>

---

### 15. [SLaB: Sparse-Lowrank-Binary Decomposition for Efficient Large Language Models](https://arxiv.org/abs/2604.04493)

**Authors**: Ziwei Li, Yuang Ma, Yi Kang  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.04493v1  

#### Abstract
The rapid growth of large language models (LLMs) presents significant deployment challenges due to their massive computational and memory demands. While model compression, such as network pruning, offers potential solutions, most existing methods often fail to maintain good performance at high compr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SLaB: Sparse-Lowrank-Binary Decomposition for Efficient Large Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（**Large Language Models, LLMs**）由于其巨大的参数量和计算开销，在资源受限设备上的部署面临严峻挑战。尽管已有多种**model compression**技术（如剪枝、量化、低秩分解），但在高压缩比下往往导致显著的性能下降。

现有方法存在以下不足：
- **One-shot pruning** 方法（如 SparseGPT、Wanda）在高稀疏度时准确率急剧下降；
- 单纯结合 **sparsity** 和 **low-rank decomposition** 效果不佳（见图1）；
- 多数方法需要微调（fine-tuning）来恢复性能，增加成本。

### 🚀 提出的新方法：SLaB
作者提出 **Sparse-Lowrank-Binary Decomposition (SLaB)**，一种无需训练的一次性（one-shot）压缩框架，将每个线性层权重矩阵 $ W $ 分解为三个互补成分：

$$
W \approx W_s + W_L \odot W_B
$$

其中：
- $ W_s $：**sparse matrix**（稀疏矩阵），保留重要权重；
- $ W_L $：**low-rank matrix**（低秩矩阵），补偿稀疏化带来的信息损失；
- $ W_B $：**binary matrix**（二值矩阵，元素 ∈ {+1, -1}），与 $ W_L $ 做 Hadamard 积以简化计算；
- $ \odot $ 表示逐元素乘法（Hadamard product）。

该设计充分利用了硬件友好的特性（如二值运算加速、稀疏存储节省内存）。

### 🔍 创新点与优势
| 特性 | 说明 |
|------|------|
| **无需重训练** | 完全基于校准数据进行一次性剪枝，不需任何 fine-tuning 或再训练。 |
| **激活感知剪枝（activation-aware pruning）** | 使用类似 Wanda 的评分机制 $ S_{ij} = |Y_s^{(i,j)}| \cdot \|X_j\|_2 $，考虑输入激活的影响，提升剪枝质量。 |
| **联合优化策略** | 采用交替优化（alternating optimization）方式迭代更新 $ W_s, W_L, W_B $，实现协同收敛。 |
| **高效低秩表示** | 引入 binary matrix 显著降低所需 rank（实验表明 rank=1 即可取得优异效果）。 |

相比现有方法，SLaB 在相同压缩比下：
- 更好地保持模型性能；
- 更适合硬件部署（支持稀疏+二值+低秩协同优化）；
- 性能提升显著且稳定。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Calibration Dataset**：用于剪枝前的激活收集  
  → 从 C4 数据集的第一个 shard 中随机采样 **128 条长度为 2048 的序列**（与 SparseGPT 一致，确保公平比较）。

- **Evaluation Datasets**：
  - **WikiText-2**：评估语言建模能力，使用 **perplexity (ppl)** 作为指标（越低越好）；
  - **Zero-shot Tasks**：通过 **LM-Eval-Harness** 测试以下任务的准确率（越高越好）：
    - ARC-C, ARC-E
    - BoolQ
    - HellaSwag
    - PIQA
    - RTE
    - WinoGrande

### ⚙️ 实验设置
| 设置项 | 配置 |
|--------|-------|
| **模型** | Llama-2 7B, Llama-3 8B, Llama-3.2 1B |
| **压缩类型** | - Unstructured Sparsity (US): 50%, 60%, 70%, 80%<br>- Structured Sparsity: 2:4 和 4:8 模式（CR=50%） |
| **目标层** | 所有 linear layers（排除 embedding 层和分类头） |
| **bit-width** | FP16（b=16） |
| **comparison group size** | (1, Din)，即按列分组剪枝 |
| **SLaB 参数** | 迭代次数 s = 20，rank = 1 |
| **实现工具** | PyTorch + Hugging Face Transformers |

### 🆚 基线方法对比
- **SparseGPT** [10]：基于二阶信息的一次性剪枝方法；
- **Wanda** [11]：激活感知的 channel-wise 剪枝方法；
- 两者均为当前 SOTA 的 one-shot pruning 方法。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table I）

#### 在 **Llama-3.2 1B @ 50% 压缩比（Unstructured）** 下：
| 方法 | Perplexity ↓ | Zero-shot Acc ↑ |
|------|---------------|------------------|
| Dense (原始) | 9.06 | 58.3% |
| SparseGPT | 18.09 | 51.2% |
| Wanda | 21.37 | 48.5% |
| **SLaB (Ours)** | **11.57** | **55.8%** |

➡️ **相比最优基线（SparseGPT）**：
- **Perplexity 降低 36.04%**
- **Zero-shot 准确率提升 8.98个百分点**

#### 在其他模型上同样领先：
| 模型 | 指标 | SLaB vs 最佳基线 |
|------|------|------------------|
| Llama-2 7B (US, 50%) | ppl: 5.49 vs 6.45 (Wanda) | ↓15% |
| Llama-3 8B (US, 50%) | ppl: 6.67 vs 8.73 (SparseGPT) | ↓23.6% |
| 所有模型在 80% 压缩比下仍保持可用性，而 Wanda 已严重退化（ppl > 1000） |

### 🔁 不同结构化稀疏模式下的表现（CR=50%）
| 结构模式 | 方法 | Llama-2 7B ppl |
|----------|------|----------------|
| 4:8 | SparseGPT | 7.94 |
|     | Wanda     | 8.03 |
|     | **SLaB**  | **5.61** ✅ |
| 2:4 | SparseGPT | 10.37 |
|     | Wanda     | 11.40 |
|     | **SLaB**  | **5.77** ✅ |

→ SLaB 在结构化稀疏场景下也大幅领先，显示其通用性和鲁棒性。

### 🔍 消融实验（Ablation Study, Table III）
在 Llama-2 7B 上进行组件分析（2:4, 50% CR）：

| 组件组合 | Avg Accuracy (%) |
|---------|--------------------|
| $ W_s $ only | 49.8 |
| $ W_s + W_L $ (r=16) | 51.2 |
| $ W_s + \text{factor} \odot W_B $ | 58.2 |
| $ W_s + W_L \odot W_B $ (**完整 SLaB**) | **58.9** |

✅ 发现：
- 单独使用 $ W_L $ 或 $ W_B $ 提升有限；
- **三者协同作用明显**，尤其 $ W_L \odot W_B $ 能有效补偿稀疏损失；
- 证明 binary + low-rank 的耦合设计是关键创新。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **稀疏 + 低秩 + 二值三者可以高效互补**：SLaB 成功构建了一个无需训练即可实现高质量压缩的框架。
2. **rank=1 已足够有效**：得益于 $ W_B $ 的引入，即使极低秩也能很好逼近残差，极大减少参数量和计算负担。
3. **无需微调即可超越 SOTA**：在多个 Llama 模型和不同压缩模式下，SLaB 均显著优于 SparseGPT 和 Wanda。
4. **对高稀疏度更鲁棒**：在 70%-80% 压缩比下仍保持合理性能，而现有方法已崩溃。

### ⚠️ 局限性
- 当前仅应用于 **decoder-only 架构**（如 Llama），未验证 encoder-decoder 模型（如 T5）；
- 分解形式固定为 $ W_s + W_L \odot W_B $，可能限制表达能力；
- 对 extremely low-rank（如全局共享 $ W_L $）未深入探索；
- 实际推理速度/能耗测试尚未提供（仅关注模型大小和精度）。

### 🔮 未来工作方向
- 探索 **动态 rank 分配**：根据不同层的重要性自适应调整 $ W_L $ 的秩；
- 扩展到 **multi-modal models** 和 **MoE 架构**；
- 结合 **quantization** 进一步压缩 $ W_s $ 和 $ W_L $；
- 开发专用硬件支持 SLaB 的稀疏-二值-低秩混合计算；
- 将此思想推广至其他压缩范式（如蒸馏、知识迁移）。

---

## ✅ 总结一句话
> **SLaB 提出了一种新颖的三元分解结构（sparse + low-rank + binary），实现了无需训练的高效 LLM 压缩，在多种模型和压缩模式下均达到 SOTA 性能，为未来轻量化大模型提供了新思路。**

</details>

---

### 16. [COSMO-Agent: Tool-Augmented Agent for Closed-loop Optimization,Simulation,and Modeling Orchestration](https://arxiv.org/abs/2604.05547)

**Authors**: Liyuan Deng, Shujian Deng, Yongkang Chen, Yongkang Dai, Zhihang Zhong, Linyang Li, Xiao Sun, Yilei Shi, Huaxi Huang  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.05547v1  

#### Abstract
Iterative industrial design-simulation optimization is bottlenecked by the CAD-CAE semantic gap: translating simulation feedback into valid geometric edits under diverse, coupled constraints. To fill this gap, we propose COSMO-Agent (Closed-loop Optimization, Simulation, and Modeling Orchestration),...

---

### 17. [MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU](https://arxiv.org/abs/2604.05091)

**Authors**: Zhengqing Yuan, Hanchi Sun, Lichao Sun, Yanfang Ye  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.05091v1  

#### Abstract
We present MegaTrain, a memory-centric system that efficiently trains 100B+ parameter large language models at full precision on a single GPU. Unlike traditional GPU-centric systems, MegaTrain stores parameters and optimizer states in host memory (CPU memory) and treats GPUs as transient compute eng...

---

### 18. [BOSCH: Black-Box Binary Optimization for Short-Context Attention-Head Selection in LLMs](https://arxiv.org/abs/2604.05942)

**Authors**: Abbas Ghaddar, Ivan Kobyzev, Boxing Chen, Yufei Cui  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.05942v1  

#### Abstract
Post-training hybridization of large language models (LLMs) often replaces quadratic self-attention with sliding-window attention (SWA) to reduce KV cache usage and improve latency. Existing hybridization schemes are typically defined either at the layer level (e.g., interleaving) or at the head lev...

---

### 19. [Hardware-Oriented Inference Complexity of Kolmogorov-Arnold Networks](https://arxiv.org/abs/2604.03345)

**Authors**: Bilal Khalid, Pedro Freire, Sergei K. Turitsyn, Jaroslaw E. Prilepsky  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.03345v1  

#### Abstract
Kolmogorov-Arnold Networks (KANs) have recently emerged as a powerful architecture for various machine learning applications. However, their unique structure raises significant concerns regarding their computational overhead. Existing studies primarily evaluate KAN complexity in terms of Floating-Po...

---

### 20. [k-Maximum Inner Product Attention for Graph Transformers and the Expressive Power of GraphGPS The Expressive Power of GraphGPS](https://arxiv.org/abs/2604.03815)

**Authors**: Jonas De Schouwer, Haitz S\'aez de Oc\'ariz Borde, Xiaowen Dong  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.03815v1  

#### Abstract
Graph transformers have shown promise in overcoming limitations of traditional graph neural networks, such as oversquashing and difficulties in modelling long-range dependencies. However, their application to large-scale graphs is hindered by the quadratic memory and computational complexity of the ...

---

### 21. [Sampling Parallelism for Fast and Efficient Bayesian Learning](https://arxiv.org/abs/2604.04736)

**Authors**: Asena Karolin \"Ozdemir, Lars H. Heyen, Arvid Weyrauch, Achim Streit, Markus G\"otz, Charlotte Debus  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.04736v1  

#### Abstract
Machine learning models, and deep neural networks in particular, are increasingly deployed in risk-sensitive domains such as healthcare, environmental forecasting, and finance, where reliable quantification of predictive uncertainty is essential. However, many uncertainty quantification (UQ) methods...

---

### 22. [Forgetting to Witness: Efficient Federated Unlearning and Its Visible Evaluation](https://arxiv.org/abs/2604.04800)

**Authors**: Houzhe Wang, Xiaojie Zhu, Chi Chen  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.04800v1  

#### Abstract
With the increasing importance of data privacy and security, federated unlearning has emerged as a novel research field dedicated to ensuring that federated learning models no longer retain or leak relevant information once specific data has been deleted. In this paper, to the best of our knowledge,...

---

### 23. [Optimizing LLM Prompt Engineering with DSPy Based Declarative Learning](https://arxiv.org/abs/2604.04869)

**Authors**: Shiek Ruksana, Sailesh Kiran Kurra, Thipparthi Sanjay Baradwaj  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.04869v1  

#### Abstract
Large Language Models (LLMs) have shown strong performance across a wide range of natural language processing tasks; however, their effectiveness is highly dependent on prompt design, structure, and embedded reasoning signals. Conventional prompt engineering methods largely rely on heuristic trial-a...

---

### 24. [Multi-Agent Pathfinding with Non-Unit Integer Edge Costs via Enhanced Conflict-Based Search and Graph Discretization](https://arxiv.org/abs/2604.05416)

**Authors**: Hongkai Fan, Qinjing Xie, Bo Ouyang, Yaonan Wang, Zhi Yan, Jiawen He, Zheng Fang  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.05416v1  

#### Abstract
Multi-Agent Pathfinding (MAPF) plays a critical role in various domains. Traditional MAPF methods typically assume unit edge costs and single-timestep actions, which limit their applicability to real-world scenarios. MAPFR extends MAPF to handle non-unit costs with real-valued edge costs and continu...

---

### 25. [Multi-Drafter Speculative Decoding with Alignment Feedback](https://arxiv.org/abs/2604.05417)

**Authors**: Taehyeon Kim, Hojung Jung, Se-Young Yun  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.05417v1  

#### Abstract
Speculative decoding (SD) accelerates large language model (LLM) inference by using a smaller model to draft future tokens, which are then verified by the target LLM. This preserves generation quality by accepting only aligned tokens. However, individual drafters, often trained for specific tasks or...

---

### 26. [Controlling Distributional Bias in Multi-Round LLM Generation via KL-Optimized Fine-Tuning](https://arxiv.org/abs/2604.05756)

**Authors**: Yanbei Jiang, Amr Keleg, Ryandito Diandaru, Jey Han Lau, Lea Frermann, Biaoyan Fang, Fajri Koto  
**Category**: cs.CL  
**Published**: 2026-04-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.05756v1  

#### Abstract
While the real world is inherently stochastic, Large Language Models (LLMs) are predominantly evaluated on single-round inference against fixed ground truths. In this work, we shift the lens to distribution alignment: assessing whether LLMs, when prompted repeatedly, can generate outputs that adhere...

---

### 27. [DP-OPD: Differentially Private On-Policy Distillation for Language Models](https://arxiv.org/abs/2604.04461)

**Authors**: Fatemeh Khadem, Sajad Mousavi, Yi Fang, Yuhong Liu  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.04461v1  

#### Abstract
Large language models (LLMs) are increasingly adapted to proprietary and domain-specific corpora that contain sensitive information, creating a tension between formal privacy guarantees and efficient deployment through model compression. Differential privacy (DP), typically enforced via DP-SGD, prov...

---

### 28. [Vision-Guided Iterative Refinement for Frontend Code Generation](https://arxiv.org/abs/2604.05839)

**Authors**: Hannah Sansford, Derek H. C. Law, Wei Liu, Abhishek Tripathi, Niresh Agarwal, Gerrit J. J. van den Burg  
**Category**: cs.AI  
**Published**: 2026-04-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.05839v1  

#### Abstract
Code generation with large language models often relies on multi-stage human-in-the-loop refinement, which is effective but very costly - particularly in domains such as frontend web development where the solution quality depends on rendered visual output. We present a fully automated critic-in-the-...

---

### 29. [CoStream: Codec-Guided Resource-Efficient System for Video Streaming Analytics](https://arxiv.org/abs/2604.06036)

**Authors**: Yulin Zou, Yan Chen, Wenyan Chen, JooYoung Park, Shivaraman Nitin, Luo Tao, Francisco Romero, Dmitrii Ustiugov  
**Category**: cs.DC  
**Published**: 2026-04-08  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.06036v1  

#### Abstract
Video streaming analytics is a crucial workload for vision-language model serving, but the high cost of multimodal inference limits scalability. Prior systems reduce inference cost by exploiting temporal and spatial redundancy in video streams, but they target either the vision transformer (ViT) or ...

---

### 30. [Improving Feasibility via Fast Autoencoder-Based Projections](https://arxiv.org/abs/2604.03489)

**Authors**: Maria Chzhen, Priya L. Donti  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.03489v1  

#### Abstract
Enforcing complex (e.g., nonconvex) operational constraints is a critical challenge in real-world learning and control systems. However, existing methods struggle to efficiently enforce general classes of constraints. To address this, we propose a novel data-driven amortized approach that uses a tra...

---

## 🔧 Configuration

This bot is configured to look for papers containing the following keywords:
- kv cache, offload, State Space, SSM, framework, System, Generation, Video, Linear, LLM, RL, RLHF, Inference, Training, Attention, Pipeline, MOE, Sparse, Quantization, Speculative, Efficient, Efficiency, Framework, Parallel, Distributed, Kernel, Decode, Decoding, Prefill, Throughput, Fast, Network, Hardware, Cluster, FP8, FP4, Optimization, Scalable, Communication

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
