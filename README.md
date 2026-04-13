# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-13 07:35:27 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Modality-Aware Zero-Shot Pruning and Sparse Attention for Efficient Multimodal Edge Inference](https://arxiv.org/abs/2604.08971)

**Authors**: Yueyuan Sui, Payal Mohapatra, Do\u{g}a\c{c} Eldenk, Haodong Yang, Yiting Zhang, Haoyan Zhang, Qi Zhu, Stephen Xia  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.08971v1  

#### Abstract
Edge devices increasingly run multimodal sensing pipelines that must remain accurate despite fluctuating power budgets and unpredictable sensor dropout. Existing pruning methods fail under these conditions: they generally require fine-tuning after compression, consuming over $10\times$ the deploymen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Modality-Aware Zero-Shot Pruning and Sparse Attention for Efficient Multimodal Edge Inference*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前在边缘设备上部署多模态模型面临两大挑战：
- **Fine-tuning依赖**：现有剪枝方法通常需要在压缩后进行微调以恢复精度，这在资源受限的边缘设备上不可行，因为反向传播消耗大量内存和能量。
- **静态重要性评分**：传统剪枝方法为模型组件分配固定的、与模态无关的重要性分数，无法适应推理时传感器模态动态缺失的情况，导致在模态不完整时性能严重下降。

### 提出的新方法
作者提出 **SentryFuse** 框架，包含两个核心组件：
- **SentryGate**：一种**模态感知的零样本剪枝**机制。通过在训练阶段学习基于梯度显著性的监督信号，SentryGate 学习一个轻量级的门控模块，该模块能根据当前激活的模态集合输出结构化组件（如注意力头、前馈通道）的重要性分数。在部署时，无需微调即可根据可用模态和计算预算动态生成精简子网络。
- **SentryAttend**：一种高效的稀疏分组查询注意力机制。它结合了 **Grouped-Query Attention (GQA)** 和**稀疏查询选择**，通过共享Key-Value投影和仅处理高显著性查询来降低自注意力的计算开销。

### 相比现有方法的优势
- **零样本压缩**：完全消除了对部署后微调的需求，解决了边缘设备上的能量和内存瓶颈。
- **模态感知**：剪枝决策是动态的，能够根据实际存在的传感器模态进行调整，显著提升了在模态缺失场景下的鲁棒性。
- **高效且即插即用**：SentryAttend 是一个可直接替换标准密集自注意力的模块，平均减少 **15%** 的 GFLOPs。
- **端到端收益**：SentryFuse 在提升效率的同时，还能保持甚至提高预测精度。

---

## 2. 核心实验方法和设置

### 数据集
在三个多模态时间序列基准数据集上进行了评估：
- **WESAD**：用于压力检测，包含10种生理信号。
- **DaliaHAR**：用于人体活动识别。
- **DSADS**：用于日常活动识别。

### 多模态骨干网络
使用了三种代表性的多模态骨干模型：
- **FlexMoE**
- **FuseMoE**
- **MAESTRO**

### 实验设置和评估指标
- **评估场景**：在不同剪枝比例（6%-53%）和模态缺失程度（0-4个模态缺失）下进行测试。
- **评估指标**：
  - **Accuracy**：分类准确率。
  - **GFLOPs**：单次推理的浮点运算次数，衡量计算成本。
  - **Memory**：序列化检查点大小，衡量存储占用。
  - **Latency**：端到端墙钟推理时间，衡量延迟。
- **硬件平台**：在多种异构硬件上进行部署测试，包括 NVIDIA L40 GPU、CPU、Jetson TX2、iPhone 13 Pro 和 Google Pixel 8。

### 基线方法对比
- **剪枝基线**：
  - **Random**：随机移除。
  - **Magnitude**：基于权重幅值剪枝。
  - **SynFlow**：一种无数据的突触流评分方法。
- **注意力基线**：原始的密集自注意力模型。
- **端到端对比**：比较原始模型、`+SentryGate` 和完整的 `SentryFuse`。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **SentryGate 性能**：在144种不同的骨干-数据集-剪枝-缺失组合中，SentryGate 在 **92.4%** 的情况下优于最强的基线（SynFlow）。其平均准确率为 **0.68**，而 SynFlow 仅为 **0.60**，实现了 **12.7%** 的相对提升。在模态缺失严重时（4个模态缺失），提升高达 **18%**。
- **SentryAttend 效率**：将密集自注意力替换为 SentryAttend 后，平均减少了 **15%** 的 GFLOPs，在某些模型上最高可达 **29%** 的节省，同时预测性能得以维持甚至提升。
- **SentryFuse 端到端性能**：
  - 内存占用从 **6.07 MB** 减少到 **4.36 MB**（**-28.2%**）。
  - GFLOPs 从 **6.83** 减少到 **4.41**（**-35.4%**）。
  - 推理延迟在 Jetson TX2 上从 295.87ms 降至 254.77ms（**-1.16×**），在 iPhone 13 Pro 上从 167.82ms 降至 113.32ms（**-1.48×**）。
  - 在所有平台上，准确率均得到提升（例如，在 WESAD 上从 0.75 提升至 0.76-0.77）。

### 与基线方法的对比结果
- **剪枝效果**：SentryGate 在所有设置下均显著优于 Random、Magnitude 和 SynFlow 剪枝，尤其是在模态缺失的情况下，优势更加明显。
- **效率对比**：SentryAttend 在所有骨干模型上都稳定地降低了 GFLOPs，且 MAESTRO + SentryAttend 组合在准确率和效率上均达到最佳。
- **综合优势**：完整的 SentryFuse 框架在准确率、内存、延迟和计算量上全面超越了原始模型和仅使用 SentryGate 的变体。

### 消融实验结果
- **SentryGate 有效性**：图11显示，SentryFuse 的性能始终非常接近于使用昂贵的泰勒展开（Taylor-based）显著性作为教师的“理想”剪枝器，证明了其学习到的重要性分数的有效性。
- **SentryAttend 配置敏感性**：图12表明，使用 **2组**（2 groups）的 GQA 在参数共享和表示灵活性之间取得了最佳平衡，因此被选为默认配置。

---

## 4. 关键结论和发现

### 主要发现
1. **模态感知的剪枝至关重要**：在多模态边缘推理中，静态的、与模态无关的剪枝策略是次优的。SentryGate 通过学习模态条件的重要性分数，实现了更鲁棒和高效的压缩。
2. **零样本剪枝是可行且必要的**：SentryFuse 成功地将复杂的微调过程“蒸馏”到一个简单的前向门控模块中，使得模型可以在部署时即时、零成本地适应不同的硬件约束和模态配置。
3. **SentryAttend 显著缓解了注意力瓶颈**：通过稀疏化和分组查询，SentryAttend 有效降低了多模态模型中最耗时的自注意力层的开销。
4. **SentryFuse 是一个实用的端到端解决方案**：该框架在真实世界的边缘硬件上验证了其有效性，能够在不牺牲精度的前提下，大幅降低内存、延迟和能耗。

### 方法的局限性
- **依赖于特定的骨干架构**：虽然 SentryAttend 可以作为即插即用模块，但整个框架的设计（尤其是与 MoE 结构的集成）可能对特定的多模态架构有更强的耦合性。
- **训练复杂性增加**：引入了额外的损失函数（对齐损失、二值化正则化）和课程学习策略，增加了训练的复杂性。
- **门控模块的泛化能力**：SentryGate 的性能依赖于训练时暴露的模态掩码模式，对于训练中未见的极端模态组合，其泛化能力有待进一步研究。

### 未来工作方向
- **扩展到其他模型族**：将 SentryFuse 的思想应用于非 Transformer 架构或其他类型的多模态模型。
- **探索更高效的门控机制**：设计更小、更快的门控模块，以进一步降低其开销。
- **结合更多压缩技术**：与知识蒸馏、量化等其他模型压缩技术进行更深入的联合优化。
- **在线自适应**：探索在部署期间根据实时反馈微调门控策略的可能性，实现更细粒度的自适应。

</details>

---

### 2. [SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks](https://arxiv.org/abs/2604.08865)

**Authors**: Tianyi Wang, Yixia Li, Long Li, Yibiao Chen, Shaohan Huang, Yun Chen, Peng Li, Yang Liu, Guanhua Chen  
**Category**: cs.AI  
**Published**: 2026-04-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.08865v1  

#### Abstract
Proximal Policy Optimization (PPO) is central to aligning Large Language Models (LLMs) in reasoning tasks with verifiable rewards. However, standard token-level PPO struggles in this setting due to the instability of temporal credit assignment over long Chain-of-Thought (CoT) horizons and the prohib...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在长链推理任务（如数学问题求解）中，传统的 **token-level PPO** 存在两个核心问题：
- **Temporal Credit Assignment 不稳定**：由于奖励稀疏且延迟（仅在最终答案处给出），标准 PPO 使用 GAE 进行时序信用分配时，难以将奖励信号有效回传到早期 token，导致训练不稳定。
- **Critic “Tail Effect”**：价值模型（Critic）倾向于“过拟合”序列末尾的语义线索，在正确路径上提前收敛至高值、在错误路径上低估中间步骤，导致优势函数（Advantage）消失。

同时，虽然 **critic-free 方法（如 GRPO）** 能缓解信用分配问题，但其依赖对同一 prompt 多次采样（N>1）来构建组内统计基线，带来巨大的计算开销，严重限制了训练吞吐量。

### 提出了什么新方法或新思路
本文提出 **SPPO (Sequence-Level PPO)**，其核心思想是：
- 将推理过程从 **token-level MDP** 重构为 **Sequence-Level Contextual Bandit (SL-CB)** 问题。
- Prompt 作为静态上下文（context），整个 Chain-of-Thought 输出被视为一个原子动作（atomic action）。
- 引入一个**解耦的标量价值函数** $V(s_p)$，用于估计给定 prompt 的可解性概率（solvability），而非 token-level 的未来回报。

### 相比现有方法的优势
- **稳定性高**：避免了长时序信用分配的偏差，解决了“Tail Effect”。
- **高效性强**：仅需单样本（N=1）即可更新策略，无需 GRPO 类方法的多采样（N=8），显著提升训练速度。
- **资源友好**：支持使用轻量级 Critic（如 1.5B）来对齐大模型（如 7B），大幅降低显存占用。
- **性能优越**：在多个数学推理基准上超越标准 PPO，并达到甚至超过 GRPO 的性能水平。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练数据**：
  - `DeepScaleR`（Luo et al., 2025）
  - `DAPO-17K`（Yu et al., 2025）
- **测试基准（held-out benchmarks）**：
  - AIME24, AIME25
  - AMC23
  - MATH500
  - Minerva Math

所有模型均基于 `DeepSeek-R1-Distill-Qwen` 系列进行微调。

### 实验设置和评估指标
- **模型规模**：
  - 1.5B 和 7B 参数量的模型分别用于验证不同尺度下的表现。
- **评估指标**：
  - **Average@16 Accuracy**：在每个测试集上取 16 次采样的平均准确率。
  - 部分实验使用 **Avg@8** 或 **Avg@k** 作为 pass rate 的代理。
- **硬件配置**：
  - 1.5B 模型：4 × A100 GPUs
  - 7B 模型：4 × H100 GPUs
- **实现框架**：基于 `verl` 框架实现所有算法。

### 基线方法对比
| 方法 | 特点 |
|------|------|
| **Base Model** | 未经 RL 微调的初始模型 |
| **Standard PPO (token-level)** | 使用 GAE 和 token-level Critic 的传统 PPO |
| **ReMax** | 序列级方法，基于 KL 正则化 |
| **RLOO** | Leave-one-out 形式的 REINFORCE 变体 |
| **GRPO (N=8)** | 组相对策略优化，需每 prompt 采样 8 条响应 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 1.5B 模型上的平均得分（Avg）：
| 方法 | 平均得分 |
|------|--------|
| Base Model | 44.96 |
| PPO | 44.06 |
| ReMax | 46.74 |
| RLOO | 46.15 |
| GRPO (N=8) | 47.08 |
| **SPPO (Ours)** | **48.06** ✅ |

#### 在 7B 模型上的平均得分（Avg）：
| 方法 | 平均得分 |
|------|--------|
| Base Model | 52.49 |
| PPO | 56.44 |
| ReMax | 57.09 |
| RLOO | 57.02 |
| GRPO (N=8) | 57.44 |
| **SPPO (Ours)** | **58.11** ✅ |
| **SPPO + Small Critic (1.5B Critic)** | **58.56** ✅✅ |

> 💡 **关键发现**：SPPO 在两种规模下均取得最优性能，尤其在 7B 上结合小规模 Critic 达到最高分。

### 与基线方法的对比结果
- **相比 Standard PPO**：
  - 显著克服了训练不稳定性（见 Figure 4），避免性能崩溃。
  - 在所有任务上全面超越，尤其是在 AIME24/25 上提升明显。
- **相比 GRPO**：
  - 性能持平甚至略优（48.06 vs 47.08 @1.5B；58.11 vs 57.44 @7B）。
  - **训练效率提升 5.9×**（见 Figure 5），达到峰值性能所需时间大幅缩短。
- **相比其他 sequence-level 方法（ReMax/RLOO）**：
  - 全面领先，说明统一优势信号传播优于 token-mask 或 leave-one-out 设计。

### 消融实验结果
#### （1）控制变量实验：PPO + BCE Loss
- 实验设计：将 SPPO 中使用的 Binary Cross-Entropy (BCE) 损失应用于标准 token-level PPO。
- 结果（Figure 4）：
  - 该变体仍出现训练崩溃，性能低于原始 PPO。
  - 表明性能提升并非来自损失函数本身，而是源于 **Sequence-Level Contextual Bandit 架构设计**。

#### （2）Decoupled Critic 消融
- 使用 1.5B Critic 对齐 7B Policy。
- 结果：
  - 性能未下降，反而略有提升（58.11 → 58.56）。
  - 显存占用减少 **12.8%**（见 Figure 6），证明价值函数建模复杂度远低于生成任务。

#### （3）Value Model 分析（Figure 7）
- 批判网络预测的 $V(s_p)$ 与实证成功率（AVG@64）呈正相关：
  - Pearson r = 0.642
  - Spearman ρ = 0.664
- 表明价值模型成功学习到了 prompt 难度的排序能力，尽管趋于保守（回归均值），但仍提供了有效的低方差基线。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **GRPO 成功的本质不是去除了 Critic，而是隐式地将任务建模为 Sequence-Level Contextual Bandit**。
2. **显式采用 SL-CB 范式并配合一个可学习的标量价值函数，可以在保持稳定性的同时摆脱多采样瓶颈**。
3. **SPPO 实现了 sample efficiency 与 optimization stability 的统一**：
   - 单样本更新（N=1）
   - 无需复杂的 group normalization
   - 支持轻量级 Critic，降低部署门槛

### 方法的局限性
- 当前方法依赖于**可验证的奖励信号**（verifiable rewards），即存在明确的对错判断（如数学题）。
- 不适用于开放域生成任务（如创意写作、对话），因为无法定义全局二元奖励。
- 对 reward shaping 敏感，若奖励不可靠可能导致价值模型误导。

### 未来工作方向
- 探索如何将 SL-CB 范式扩展到**非确定性或主观性任务**中，例如通过引入不确定性建模或多视角评分。
- 研究更高效的 value model 架构，进一步压缩 Critic 规模。
- 将 SPPO 应用于更大规模模型（>10B）及多模态推理场景。

---

> 🔗 **代码开源地址**：https://github.com/sustech-nlp/SPPO  
> 📚 **推荐阅读图示**：Figure 3（SPPO 架构概览）、Figure 5（训练效率对比）、Figure 7（价值模型校准分析）

</details>

---

### 3. [Attention-Based Sampler for Diffusion Language Models](https://arxiv.org/abs/2604.08564)

**Authors**: Yuyan Zhou, Kai Syun Hou, Weiyu Chen, James Kwok  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.08564v1  

#### Abstract
Auto-regressive models (ARMs) have established a dominant paradigm in language modeling. However, their strictly sequential decoding paradigm imposes fundamental constraints on both inference efficiency and modeling flexibility. To address these limitations, diffusion-based large language models (dL...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Attention-Based Sampler for Diffusion Language Models**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
当前的 **Diffusion Large Language Models (dLLMs)** 虽然支持并行解码和灵活生成顺序，但其主流的解码策略（如基于置信度、熵或margin的token-level采样）存在以下问题：
- 仅依赖局部输出概率信息，忽略全局序列结构；
- 缺乏与**目标序列对数似然最大化**之间的理论联系；
- 导致次优的解码路径和生成质量。

因此，如何**系统性地选择最优解码顺序以最大化序列似然**成为一个核心挑战。

---

### 🚀 提出的新方法与思路
作者提出了一种**理论驱动的注意力引导解码算法——Attn-Sampler**，其核心思想是：

> **通过Transformer模型中的自注意力矩阵列和（column sum）来衡量token的重要性，并按重要性降序进行解码。**

#### 主要创新点包括：
1. **理论建模**：将解码顺序选择形式化为一个优化问题，旨在最小化“排列依赖间隙”（Permutation Dependency Gap, PDG），即实际因子分解与理想全上下文预测之间的差距。
2. **理论证明**：在合理假设下，**按注意力矩阵列和从高到低排序解码可近似最小化PDG上界**，从而理论上保证更高的生成质量。
3. **训练免费算法设计**：Attn-Sampler无需额外训练，直接利用预训练dLLM的注意力权重动态决定解码顺序。
4. **高效实现机制**：
   - **Block Attention Approximation**：避免完整计算 $n \times n$ 注意力矩阵，适配FlashAttention等高性能内核；
   - **Dynamic Attention Thresholding**：结合概率阈值与动态注意力阈值，自适应控制并行解码粒度，在保持质量的同时提升吞吐量。

---

### 🔍 相比现有方法的优势
| 维度 | Attn-Sampler | 传统Token-Level方法（如Confidence/Entropy） |
|------|---------------|---------------------------------------------|
| **理论基础** | 明确连接注意力结构与似然最大化 | 缺乏理论支撑，启发式设计 |
| **信息来源** | 全局注意力结构（sequence-level） | 局部输出分布（token-level） |
| **灵活性** | 动态调整解码顺序与并行程度 | 固定策略（如top-k、固定阈值） |
| **效率-精度权衡** | 更优的Pareto前沿（更高准确率+更高吞吐） | 难以兼顾速度与质量 |

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集
涵盖两大类典型语言任务：
- **数学推理**：
  - **GSM8K**：小学级数学应用题
  - **MATH**：复杂高中级别数学问题
- **代码生成**：
  - **HumanEval**：函数级代码生成评估
  - **MBPP**：面向编程行为的Python代码生成

---

### ⚙️ 实验设置
- **模型**：
  - `Fast-dLLM v2`（1.5B 和 7B 参数）
  - `LLaDA-1.58B`
- **硬件平台**：单张 NVIDIA A6000 GPU
- **评估指标**：
  - 准确率（Accuracy）：Pass@1
  - 吞吐量（Throughput）：Tokens Per Second (TPS)
  - 平均得分（Avg.）：四项任务平均性能
- **块大小（block size）**：默认为8个MASK token组成的block

---

### 🆚 基线方法对比
比较了多种主流dLLM解码策略：
| 方法 | 类型 | 特点 |
|------|------|------|
| **Top-1 Confidence** | Token-level | 选最高预测概率token |
| **Margin Sampler** | Token-level | 选top1与top2概率差最大者 |
| **Entropy Sampler** | Token-level | 选不确定性最低（熵最小）token |
| **Fast-dLLM** | Parallel | 基于置信度阈值的并行解码 |
| **EB-Sampler** | Adaptive | 基于熵边界去掩码 |
| **KLASS Sampler** | Adaptive | 基于KL散度控制采样节奏 |

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（来自 Table 1）

| 模型 | 方法 | GSM8K | MATH | HumanEval | MBPP | **Avg.** |
|------|------|-------|------|-----------|--------|---------|
| Fast-dLLM v2 7B | Confidence | 82.71 | 50.96 | 54.27 | 30.69 | 54.66 |
| Fast-dLLM v2 7B | Entropy Sampler | 82.87 | 51.92 | 55.49 | 35.98 | **56.57** |
| Fast-dLLM v2 7B | **Attn-Sampler (Seq)** | **84.00** | **52.50** | **57.93** | **36.24** | **57.67** |
| Fast-dLLM v2 7B | **Attn-Sampler (Par)** | 84.23 | 51.88 | 58.54 | 35.98 | **57.66** |

> ✅ **Attn-Sampler 在所有任务上均达到SOTA水平**，平均超越最强基线（Entropy Sampler）约 **1.1个百分点**，在 HumanEval 上提升达 **+2.44%**。

---

### 🔁 与其他规模模型的结果一致性
| 模型 | 最佳基线 Avg. | **Attn-Sampler Avg.** |
|------|----------------|--------------------------|
| LLaDA-1.58B | 50.98 (Confidence) | **52.84** (Sequential) |
| Fast-dLLM v2 1.5B | 39.71 (KLASS) | **41.26** (Sequential) |

> 表明 Attn-Sampler 在不同参数尺度下均有显著增益，具备良好泛化能力。

---

### ⏱️ 推理速度与精度权衡（Figure 2a）
- **Attn-Sampler 实现更优的 Pareto 前沿**：
  - 在相同吞吐量 **95 TPS** 下，Attn-Sampler 达到 **84.2%** 准确率，而 Fast-dLLM 仅为 **82.1%**；
  - 可配置至 **107 TPS（加速3.06×）** 仍维持 **82.6%** 准确率，接近置信度方法精度但速度快三倍；
  - KLASS 虽达 83.2% 准确率，但吞吐仅 **51 TPS**，效率远低于本方法。

---

### 🔍 消融实验结果

#### （1）**Dynamic Attention Thresholding vs. 静态策略**
- 对比：
  - **Top-k Selection**（k=2,3,4）
  - **Static Thresholding**（阈值=0.8~1.0）
- 结果（Figure 2b）：
  - 当吞吐提升至 ~120 TPS，Top-k 准确率下降超 **15%**；
  - Attn-Sampler 在 **118 TPS** 时仍保持 **81.35%** 准确率；
  - 表明**动态门控机制能有效保留关键语义信息**，避免盲目并行导致性能崩溃。

#### （2）**注意力层与头的数量影响**（Figure 3）
- **层数越多越好**：
  - 单层：82.26%
  - 前7层：84.08%
  - 所有28层均值：**84.23%**（最佳）
- **注意力头数量增加持续提效**：
  - 单头：83.32%
  - 所有28头均值：**84.23%**
- ✅ **结论：聚合全部layer和head的信息可最大化性能**，说明高层语义与分布式表示至关重要。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **注意力矩阵列和是衡量token重要性的有效代理**，其排序与最大化序列似然具有理论关联；
2. **Attn-Sampler 是首个将 self-attention 结构显式用于解码顺序决策的方法**，实现了从“输出空间”向“表示空间”的范式转变；
3. **动态注意力阈值机制显著优于静态并行策略**，能够在高吞吐下维持高质量生成；
4. **无需微调即可大幅提升现有dLLM性能**，是一种通用、即插即用的推理增强方案。

---

### ⚠️ 方法的局限性
1. **依赖注意力矩阵稳定性假设**（Assumption 3.1）：假设在一个block内注意力不变，可能在深层或多步去噪中不完全成立；
2. **对多头多层注意力需聚合处理**：虽然实验证明使用所有heads/layers效果最好，但增加了轻量化部署难度；
3. **未探索非Transformer架构适用性**：目前仅适用于标准Transformer-based dLLMs。

---

### 🔮 未来工作方向
1. **扩展至连续扩散模型或其他生成框架**（如AR+Diffusion混合模型）；
2. **研究注意力之外的中间表示作为解码指导信号**（如MLP激活、梯度信息）；
3. **进一步压缩注意力计算开销**，实现端侧实时推理；
4. **结合强化学习或元学习自动优化阈值策略**，实现完全自适应解码控制。

---

## ✅ 总结
> **Attn-Sampler 提供了一个理论严谨、实践高效的新型解码范式，首次建立了 attention structure 与 sequence likelihood maximization 之间的桥梁。它不仅在多个基准上刷新SOTA，还揭示了“注意力即重要性”的深层洞见，有望成为未来 dLLM 推理的标准组件之一。**

🔗 开源地址：[https://github.com/YuyanZhoul/Attn_Sampling](https://github.com/YuyanZhoul/Attn_Sampling)

</details>

---

### 4. [Think Less, Know More: State-Aware Reasoning Compression with Knowledge Guidance for Efficient Reasoning](https://arxiv.org/abs/2604.09150)

**Authors**: Yi Sui, Chaozhuo Li, Dawei Song  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.09150v1  

#### Abstract
Large Reasoning Models (LRMs) achieve strong performance on complex tasks by leveraging long Chain-of-Thought (CoT), but often suffer from overthinking, leading to excessive reasoning steps and high inference latency. Existing CoT compression methods struggle to balance accuracy and efficiency, and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Think Less, Know More: State-Aware Reasoning Compression with Knowledge Guidance for Efficient Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型推理模型（Large Reasoning Models, LRMs）虽然通过长链式思维（Chain-of-Thought, CoT）在复杂任务上表现出色，但普遍存在“过度思考”（overthinking）现象：
- 推理步骤冗长，导致响应延迟高、计算成本大；
- 在不确定或存在偏差的推理状态下，错误会通过自我反思机制被放大；
- 现有 CoT 压缩方法难以在**准确性**与**效率**之间取得良好平衡，且缺乏对不同推理阶段细粒度冗余源的动态适应。

### 🚀 提出的新方法：STACK
作者提出 **State-Aware Reasoning Compression with Knowledge Guidance (STACK)**，一种结合状态感知与知识引导的动态 CoT 压缩框架。其核心思想是：
- **按步决策**：在每个推理步骤中识别当前“推理状态”，并据此选择最优压缩策略；
- **知识介入**：引入检索增强生成（RAG）作为外部知识源，在关键节点纠正偏差；
- **早期停止**：基于答案分布收敛判断是否终止冗余验证。

### 🔍 创新点
1. **状态感知的压缩策略切换机制**  
   - 动态检测“犹豫状态”（hesitation state），即模型不确定性高的时刻；
   - 若处于犹豫状态 → 启用 **Knowledge-Guided Compression**，利用外部知识校正方向；
   - 若仅是冗长但置信度高 → 启用 **Self-Prompted Compression**，去除重复表达。

2. **知识引导的对比解码机制（KGCD）**  
   - 引入 prompt asymmetry 设计：原始模型保持完整推理行为，而检索条件下的模型被约束为简洁、证据驱动；
   - 通过对比两个模型的输出概率分布，抑制无依据扩展，提升准确性和紧凑性。

3. **基于答案收敛的早期停止机制（Answer-Convergence-Based Early Stopping）**  
   - 监控连续步骤间候选答案分布的 KL 散度；
   - 当信息增益低于阈值时，判定推理已稳定，提前终止后续验证步骤。

4. **奖励差驱动的训练策略（Reward-Difference-Driven Training）**  
   - 结合 PPO 和 DPO 的优势，设计 **MDPO（Margin-enhanced DPO）损失函数**；
   - 将奖励差异 $ \Delta R $ 作为动态 margin，使模型不仅能学习偏好方向，还能感知改进强度。

### ⚖️ 相比现有方法的优势
| 维度 | 现有方法局限 | STACK 改进 |
|------|----------------|-------------|
| **压缩粒度** | 多为任务级或全局压缩 | 实现**步级**（step-level）细粒度控制 |
| **冗余处理** | 忽视不同来源的冗余（如重复 vs 不确定） | 区分**置信冗余**与**偏差传播**，分别应对 |
| **知识整合** | RAG 被动嵌入上下文，易被内部推理淹没 | 主动在关键时刻注入知识，实现**精准纠偏** |
| **训练稳定性** | 强压缩易导致语言退化或崩溃 | 通过参考策略锚定 + 自适应 margin 提升鲁棒性 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在三个数学推理基准上进行评估，难度递增：
- **GSM8K**：小学级别应用题，侧重基本逻辑；
- **MATH500**：中学至大学水平数学题，涵盖代数、几何等；
- **AIME24**：奥赛级别题目，挑战性强，测试模型极限能力。

### ⚙️ 实验设置
- **基础模型**：
  - `DeepSeek-R1-Distill-Qwen-1.5B`
  - `DeepSeek-R1-Distill-Qwen-7B`
- **训练框架**：PyTorch，4×NVIDIA A100-80GB GPU；
- **学习率**：2e-6，批量大小：64；
- **外部知识获取**：使用 Bing Web Search API 获取 top-5 检索结果；
- **冷启动预训练**：先在标注数据上进行 SFT，避免直接 DPO 导致模式崩溃。

### 📊 评估指标
| 指标 | 定义 | 说明 |
|------|------|------|
| **Acc** | 准确率（Accuracy） | 正确解答的比例 |
| **Len** | 平均响应长度（token 数） | 衡量推理效率 |
| **Lat(s)** | 推理延迟（秒） | 实际运行时间 |
| **Token Efficiency (TE)** | $ \text{Acc} \times 100 / \text{Len} $ | 综合衡量精度与效率的平衡 |

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Prompt** | 输出优化 | 注入轻量提示限制长度 |
| **ConCISE** | 模型优化 | 基于信心与早停构建高质量压缩样本 |
| **MuTIS** | 模型优化 | 多轮干预采样 + DPO 微调 |
| **TokenSqueeze** | 模型优化 | 自适应深度选择 + 分布对齐精炼 |

---

## 3. 主要实验结果和性能指标

### 📈 性能总览（Table 1）
在所有模型规模下，**STACK 均实现最佳准确率-效率权衡**：

| 方法 | Avg Acc ↑ | Avg Len ↓ | Lat(s) ↓ | 压缩率 |
|------|----------|-----------|----------|--------|
| Original | — | — | 16.14 (1.5B) / 13.29 (7B) | — |
| Prompt | +0.77 | -7.35% | ~1s↓ | 极低 |
| ConCISE | +0.83 | -51.5% | ~6.7s↓ | 中等 |
| MuTIS | +2.07 | -56.7% | ~4.5s↓ | 高 |
| **STACK (Ours)** | **+4.80** | **-59.9%** | **7.23s (1.5B) / 6.73s (7B)** | **最高** |

> ✅ **关键成果**：平均准确率提升 **4.8 分**，同时将响应长度减少近 **60%**，显著优于所有基线。

#### 🔍 按数据集表现
- 在简单任务（GSM8K）上，STACK 能自适应生成更短链（平均 684 tokens）；
- 在困难任务（AIME24）上仍保持高准确率（36.7 → 57.4），表明其**强泛化能力**；
- Token Efficiency 最高，说明单位 token 的推理价值最大。

### 🔬 消融实验（Ablation Study, Table 2）

| 设置 | MATH500 Acc/Len | AIME24 Acc/Len |
|------|------------------|---------------|
| Original | 91.2 / 4010 | 53.8 / 13178 |
| Ablate Early Stopping (End Signal) | 86.7 / 3374 | 31.7 / 10693 |
| Ablate Early Stopping (Consistency) | 93.3 / 1980 | 57.5 / 8157 |
| **STACK (Full)** | **93.5 / 1733** | **57.4 / 7274** |

> ❗ 发现：
- 单纯增加“结束信号”概率会导致**过早截断**，严重损害准确率；
- “答案一致性”虽有效，但仍不如基于分布收敛的方法紧凑；
- **完整 STACK 显著优于各变体**，证明各组件协同作用。

| 训练范式 | MATH500 Acc/Len | AIME24 Acc/Len |
|---------|------------------|---------------|
| SFT | 92.4 / 3016 | 56.7 / 9761 |
| SFT + DPO | 93.2 / 1895 | 57.1 / 7586 |
| **SFT + MDPO (STACK)** | **93.5 / 1733** | **57.4 / 7274** |

> ✅ 证明：**奖励差驱动的 MDPO 损失函数**进一步提升了压缩效率与稳定性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **推理状态感知至关重要**  
   不同阶段的冗余成因不同（重复 vs 不确定），需采用差异化压缩策略。

2. **知识应主动而非被动使用**  
   外部知识不应只是静态上下文，而应在模型出现犹豫时**动态注入**，才能有效纠偏。

3. **早期停止可大幅提升效率**  
   基于答案分布收敛的机制能在不牺牲正确性的前提下，消除大量冗余验证步骤。

4. **在线对比采样优于离线构造**  
   STACK 在线构建 long-short 对比样本，能更好适应模型演化过程，避免分布漂移。

5. **小模型也能高效推理**  
   即使在 1.5B 模型上，STACK 也实现了接近甚至超越大模型的性能，具备部署潜力。

### ⚠️ 局限性
1. **额外开销**：在线检索与对比采样带来更高的训练时间和资源消耗；
2. **知识质量依赖**：检索结果若不准确或无关，可能误导模型；
3. **模态单一**：目前仅支持文本知识，未集成符号求解器或多模态工具；
4. **领域限制**：实验集中于数学推理，其他领域（如代码、规划）尚待验证。

### 🔮 未来工作方向
- 联合优化 retrieval 模块与推理模型，提升知识可靠性；
- 扩展至多模态知识源（图表、公式数据库、symbolic solver）；
- 探索更高效的 contrastive decoding 实现方式，降低推理延迟；
- 应用于真实场景（如教育辅导、自动编程助手）中的长期交互推理系统。

---

> 💡 **一句话总结**：  
> **STACK 通过“看状态、引知识、早停止”的三重机制，实现了“少想一点，多懂一些”的高效推理新范式，在大幅压缩 CoT 的同时反而提升了准确率，为下一代高效 LRMs 提供了可行路径。**

</details>

---

### 5. [Efficient RL Training for LLMs with Experience Replay](https://arxiv.org/abs/2604.08706)

**Authors**: Charles Arnal, Vivien Cabannes, Taco Cohen, Julia Kempe, Remi Munos  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.08706v1  

#### Abstract
While Experience Replay - the practice of storing rollouts and reusing them multiple times during training - is a foundational technique in general RL, it remains largely unexplored in LLM post-training due to the prevailing belief that fresh, on-policy data is essential for high performance. In thi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient RL Training for LLMs with Experience Replay**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代大型语言模型（LLMs）在推理任务中广泛采用 **Reinforcement Learning (RL)** 进行后训练（post-training），例如数学解题和代码生成。然而，当前主流的 **on-policy RL** 方法（如 PPO、GRPO）遵循“生成即丢弃”（generate-then-discard）范式，导致极高的推理计算开销——在许多系统中，**推理成本占总训练GPU时间的80%以上**。

尽管 **Experience Replay (ER)** 在传统RL中被广泛用于提升样本效率，但在LLM训练中却长期被忽视，其背后假设是：**off-policy 数据会因策略过时（staleness）而导致性能下降**。

本文挑战了这一共识，系统地研究了在LLM RL训练中引入Experience Replay的可行性与效益。

---

### **提出的新方法与新思路**
- **首次系统性论证**：在LLM RL训练中，**适度的off-policiness并非有害，反而可作为正则化手段稳定训练**。
- **理论框架建模**：将Experience Replay的设计形式化为一个三重权衡（trade-off）：
  - **Staleness-induced variance**（过时样本带来的梯度方差）
  - **Sample diversity**（样本多样性损失）
  - **Inference cost**（推理生成成本）
- **提出异步训练中的ER实现方案**：在标准的异步RLHF架构中引入**共享回放缓冲区（replay buffer）**，允许trainer重复采样历史rollouts，从而显著降低推理负载。

---

### **相比现有方法的优势**
| 维度 | 传统方法（On-policy） | 本文方法（Experience Replay） |
|------|------------------------|-------------------------------|
| **计算效率** | 极低，每条轨迹仅用一次 | 高，通过复用大幅减少推理次数 |
| **样本效率** | 低 | 显著提升 |
| **训练稳定性** | 易崩溃，波动大 | 更稳定，防止过拟合 |
| **实现复杂度** | 简单但浪费 | 极简修改即可集成进现有pipeline |
| **目标优化** | 最大化每步性能 | 最大化单位算力下的性能 |

> ✅ **核心优势**：**无需改变训练算法或损失函数，仅通过加入简单replay buffer即可节省高达40%的计算资源，同时保持甚至提升最终准确率**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **OpenR1-Math-220k**：大规模数学推理数据集，用于主实验。
- **MATH**：标准数学评测集，用于测试泛化能力。
- **miniF2F**：Lean定理证明任务子集，用于跨任务验证。
- **Llama3.2-3B on OpenR1-Math-220k**：用于验证方法在不同模型上的普适性。

---

### **实验设置**
- **模型**：
  - Qwen3-0.6B（小规模）
  - Qwen2.5-7B（中等规模）
  - Qwen3-8B 和 Llama3.2-3B（扩展验证）
- **训练方式**：
  - 异步RL训练：`W`个inference worker生成rollouts，`T`个trainer从buffer中采样并更新。
  - 使用 **GRPO** 作为默认RL算法。
- **缓冲区设计**：
  - FIFO队列，大小为 `N`
  - 支持均匀采样、正样本偏置采样（positive-bias sampling）
  - 可配置 `(W, T)` 比例以控制 **replay ratio** 和 **compute ratio**
- **关键超参数**：
  - Buffer size: `N ∈ {64, 256, ..., 20736}`
  - Worker-Trainer ratio: `(W,T) ∈ {(6,2), (5,3), (4,4)}`
  - 学习率：对每个模型进行ablation调优

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Accuracy on MATH/OpenR1-Math** | 主要性能指标 |
| **Pass@k (k>1)** | 衡量输出多样性（diversity preservation） |
| **Compute spent** | 单位为“标准化GPU计算单元”，定义为：<br>`compute = C × (1 + W/T)`，其中`C`为单次backward代价 |
| **Wall-time** | 实际运行时间（小时） |
| **Replay Ratio** | 平均每个样本被使用的次数 |
| **Off-policiness** | 样本生成步数与使用步数之差 |
| **Entropy / Stability** | 训练过程中的策略熵变化与崩溃频率 |

---

### **基线方法对比**
- **Baseline**：无buffer的标准on-policy训练（如GRPO/PPO）
- **对比维度**：
  - 相同compute预算下的最高accuracy
  - 达到相同accuracy所需的compute
  - 训练稳定性（是否崩溃）
  - 输出多样性（pass@k）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | 方法 | Compute 节省 | Accuracy 提升 | 备注 |
|------|------|---------------|----------------|------|
| Qwen2.5-7B | ER (`N=84`, `(W,T)=(5,3)`) | **↑40% compute efficiency** | **持平或略优** | 图1显示所有compute水平下均优于baseline |
| Qwen3-0.6B | ER (`N=2304`, `(W,T)=(4,4)`) | 最高节省~40% | 达到更高peak accuracy | 图3 |
| Qwen3-8B | ER on miniF2F | 显著加速收敛 | 性能相当 | 图16 |
| Llama3.2-3B | ER on Math | 同样趋势 | 验证通用性 | 图17 |

> 📈 **图1核心结果**：在达到相同准确率时，**带buffer的方法所需compute仅为baseline的60%左右**。

---

### **与基线方法的对比结果**
- **计算效率**：
  - 当 `(W,T) = (1,7)` 时，`γ = 0.18`，意味着每次参数更新的计算成本仅为baseline的18%。
- **准确性**：
  - 所有合理配置的ER方法在**相同compute下均优于on-policy baseline**。
  - 某些配置甚至达到更高的**峰值准确率**。
- **训练稳定性**：
  - ER显著减少训练崩溃（crash）现象。
  - 训练曲线更平滑，不易过拟合后坍塌。

---

### **消融实验结果**
#### （1）Buffer Size 影响
- **太小（N=64）**：效果不明显
- **适中（N=256~2304）**：最佳平衡点，既能复用又不过时
- **太大（N>10k）**：训练变慢，但稳定性增强

> 🔍 发现：**增大buffer size会增加平均off-policiness，但能提升局部多样性（local diversity）**。

#### （2）Worker-Trainer Ratio `(W,T)`
| `(W,T)` | Replay Ratio | Compute Ratio γ | 效果 |
|--------|--------------|------------------|------|
| (6,2) | ~1.8 | ~0.65 | 稳定高效 |
| (5,3) | ~3.4 | ~0.43 | 更省算力，仍有效 |
| (4,4) | ~7.0 | ~0.32 | 极端复用，略有性能下降但性价比高 |

> ⚠️ 过度复用（replay ratio >10）会导致性能下降，归因于**局部多样性丧失**。

#### （3）Positive-Bias Sampling + AsymRE Loss
- **Positive-bias sampling**（保留更多正确rollout）：
  - 显著提升性能，尤其在高replay ratio下
- **替换GRPO为AsymRE loss**：
  - 避免重要性采样带来的方差爆炸
  - 结合ER后表现更鲁棒（见图5）

> ✅ **组合改进可进一步推高Pareto前沿**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **“Generate-then-discard”不是最优选择**：
   - 在推理成本高昂时，**适度复用旧数据比严格on-policy更高效**。
2. ✅ **Experience Replay可作正则化工具**：
   - 复用历史策略生成的数据有助于防止过拟合，提升训练稳定性。
3. ✅ **Preserve Output Diversity**：
   - ER方法在 **pass@k（k>1）** 上表现更好，说明其有助于维持策略熵。
4. ✅ **Compute Efficiency与Accuracy可兼得**：
   - 通过调节 `buffer size` 和 `(W,T)` 比例，可在不牺牲性能的前提下节省**最多40%计算资源**。
5. ✅ **理论指导实践**：
   - 理论分析给出最优 `N/R`（staleness horizon）和 `B/R`（replay ratio）比例，与实验吻合良好（见图6）。

---

### **方法的局限性**
- **当前验证集中在中小模型**（≤8B），尚未在百亿级以上模型上充分验证。
- **极端复用（replay ratio >10）可能导致性能下降**，需谨慎调参。
- **未考虑动态难度课程学习或复杂过滤机制**，仅使用最简uniform buffer。
- **理论分析基于同步设定**，而实验在异步环境中进行，存在一定gap。

---

### **未来工作方向**
- 探索更智能的**采样策略**：
  - Prioritized Experience Replay（按reward/TD error排序）
  - Hindsight Experience Replay for reasoning（失败prompt重标注为成功路径）
- 设计专用于ER的**off-policy稳定损失函数**（如结合AsymRE与clip机制）
- 将ER应用于**多阶段训练 pipeline**（exploration → exploitation）
- 扩展至更大规模模型（如70B+）和更多任务（coding, planning, agent）
- 结合**offline RL**思想，构建混合on/off-policy训练框架

---

## **总结**
> 💡 **本文颠覆了LLM RL训练中“必须on-policy”的固有认知，证明了一个简单的replay buffer不仅能大幅节省计算成本（↓40%），还能提升训练稳定性与输出多样性。这标志着LLM强化学习正从“最大化每步收益”转向“最大化单位算力收益”的新范式。**

该方法易于部署、改动极小，具有极强的工业落地潜力，有望成为下一代高效RLHF系统的标配组件。

</details>

---

### 6. [Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](https://arxiv.org/abs/2604.08926)

**Authors**: Taojie Zhu, Dongyang Xu, Ding Zou, Sen Zhao, Qiaobo Hao, Zhiguo Yang, Yonghong He  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.08926v1  

#### Abstract
Post-training paradigms for Large Language Models (LLMs), primarily Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), face a fundamental dilemma: SFT provides stability (low variance) but suffers from high fitting bias, while RL enables exploration (low bias) but grapples with high gradi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在推理能力提升上的后训练（post-training）主要依赖两种范式：
- **Supervised Fine-Tuning (SFT)**：提供稳定、低方差的学习信号，但存在**高拟合偏差（high fitting bias）**，限制了对新推理路径的探索和 Out-of-Distribution (OOD) 泛化能力。
- **Reinforcement Learning (RL)**：通过奖励信号实现低偏差的探索，但因采样随机性导致**梯度方差高（high gradient variance）**，训练不稳定。

现有统一优化策略（如简单加权损失）忽略了 SFT 和 RL 梯度信号之间的**统计冲突**，无法有效平衡偏差与方差。

### 提出了什么新方法或新思路
本文提出 **DYPO (Dynamic Policy Optimization)**，一种结构化融合 SFT 与 RL 的统一框架，其核心思想是**基于样本难度动态路由优化路径**，而非全局混合目标。

DYPO 包含三个核心组件：
1. **Dynamic Difficulty Grading**  
   基于多轨迹回放（group rollout）结果将查询分为三类：
   - **Easy**：全部成功 → 忽略（无学习信号）
   - **Hard**：全部失败 → 路由至 SFT 进行知识注入
   - **Mid**：部分成功 → 路由至 RL 进行探索
2. **Multi-Teacher Distillation (用于 SFT)**  
   在 Hard 样本上采用多个教师模型生成多样化推理路径，聚合监督信号以**降低单源 SFT 的拟合偏差**。
3. **Group Alignment Loss (GAL) (用于 RL)**  
   在 Mid 样本上引入对比性损失，利用组内正负样本对拉近正确路径、推远错误路径，显著**降低 RL 梯度方差**。

此外，DYPO 引入 **Dynamic Exploitation-Exploration Gating** 机制，根据奖励反馈自适应地仲裁 SFT 与 RL 的使用。

### 相比现有方法的优势
- **结构化解耦**：不同于简单的损失加权或切换策略，DYPO 实现了按样本难度的**实例级路由（instance-level routing）**，更精细地管理学习过程。
- **理论保障**：理论上证明 DYPO 可线性减少拟合偏差，并最小化整体梯度方差。
- **高效稳定**：避免了传统“SFT-then-RL”流程中的偏差传播和计算开销，训练更稳定且样本利用率更高。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **In-Distribution (ID) 数据集**：
  - AIME 2024 / 2025
  - AMC
  - MATH-500
  - Minerva
- **Out-of-Distribution (OOD) 数据集**：
  - ARC-c
  - GPQA-Diamond (博士级科学问答)

所有数据均基于 `OpenR1-Math-220k` 子集构建，并使用 **DeepSeek-R1** 和 **Qwen3-235B-A22B** 作为教师模型生成辅助推理轨迹，支持 Multi-Teacher Distillation。

### 实验设置和评估指标
- **基础模型**：Qwen2.5-Math-7B 和 Qwen3-4B-Base
- **硬件配置**：2 节点 × 8 × NVIDIA A800 GPU (80GB)
- **训练细节**：
  - 每个提示生成 8 条轨迹（rollouts）
  - 最大响应长度：8,192 tokens
  - 学习率：1e-6
  - 使用 bfloat16 精度
  - 基于 `verl` 框架实现
- **评估指标**：
  - AIME/AMC：pass@32
  - 其他任务：pass@1
  - 推理温度：0.6，启用选项打乱防止数据泄露

### 基线方法对比
共四类基线：
1. **标准 SFT 基线**：vanilla SFT
2. **Zero-shot RL 方法**：
   - SimpleRL-Zero
   - OpenReasoner-Zero
   - PRIME-Zero
   - Oat-Zero
3. **多阶段优化方法**：
   - SFT→RL
   - SuperRL
   - LUFFY
   - ReLIFT
   - SRFT
   - CHORD

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | ID 平均得分 | OOD 平均得分 |
|------|-------------|--------------|
| **DYPO (Qwen2.5-Math-7B)** | **52.5** | **61.6** |
| 第二名 (CHORD) | 50.2 | 60.8 |

在 **Qwen2.5-Math-7B** 上：
- **平均提升**：相比 SFT 提升 **+8.4%**，相比 SFT→RL 提升 **+4.8%**
- **OOD 任务提升**：达到 **61.6%**，较 SFT 基线提升 **+13.3%**

在 **Qwen3-4B-Base** 上：
- ID 平均得分达 **66.9%**，显著优于 SFT→RL 的 56.1%
- OOD 平均得分达 **68.5%**

### 与基线方法的对比结果
- **vs SFT**：DYPO 在所有 ID 和 OOD 任务上全面超越，尤其在复杂推理任务（如 AIME 25）上表现突出。
- **vs Zero-shot RL**：避免了“探索陷阱（exploration trap）”，稳定性更强，在 AIME 24/25 上领先 SimpleRL-Zero 和 Oat-Zero 超过 **19.4 分**。
- **vs 多阶段流水线**：优于 SuperRL、LUFFY、SRFT、CHORD 等先进方法，说明其动态路由机制优于静态混合或两阶段流程。

### 消融实验结果
| 阶段 | AIME 25 | GPQA-D |
|------|--------|--------|
| +SFT | 22.3 | 24.7 |
| +Multi-Teacher | 23.3 | 33.3 |
| +RL | 26.6 | 34.8 |
| +Dynamic Grading | 28.7 | 36.4 |
| **+GAL (完整 DYPO)** | **28.7** | **41.4** |

- **Multi-Teacher** 明显提升泛化能力。
- **Dynamic Grading** 对最难任务增益最大。
- **GAL** 进一步稳定训练并提升最终性能。

> 即使使用较弱的 8B 教师模型，DYPO 仍能将 AIME 25 从 22.0 提升至 27.8，表明其增益不仅来自更强教师。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **SFT 与 RL 的融合需结构性解耦**：简单加权无法解决梯度信号间的统计冲突，应根据样本难度动态分配优化目标。
2. **Multi-Teacher Distillation 可线性降低 SFT 拟合偏差**：通过聚合多个教师的多样化推理路径，削弱个体模型的特异性偏差。
3. **GAL 是有效的梯度方差控制器**：其对比机制天然具备方差抑制特性，随模型判别能力增强而自动退火。
4. **DYPO 实现更优的偏差-方差权衡**：在保持训练稳定性的同时，显著提升复杂推理与 OOD 泛化能力。

### 方法的局限性
1. **领域局限性**：当前评估集中于逻辑密集型任务（如数学推理），在开放域任务（如创意写作、闲聊）中的有效性尚待验证。
2. **训练效率较低**：每提示需生成 8 条轨迹进行动态估计，带来较高的在线采样开销，**样本效率低于纯离线方法**。
3. **依赖高质量教师模型**：Multi-Teacher Distillation 的效果受限于教师模型的质量与多样性。

### 未来工作方向
- 将 DYPO 扩展至更多任务类型（如代码生成、对话系统）。
- 探索更高效的动态估计机制以降低计算成本。
- 结合离线数据与在线探索，进一步提升样本效率。
- 研究如何自动化选择最优教师组合与超参数配置。

---

> ✅ **代码已开源**：https://github.com/Tocci-Zhu/DYPO  
> 📄 **论文链接**：https://arxiv.org/abs/2604.08926

</details>

---

### 7. [MT-OSC: Path for LLMs that Get Lost in Multi-Turn Conversation](https://arxiv.org/abs/2604.08782)

**Authors**: Jyotika Singh, Fang Tu, Miguel Ballesteros, Weiyi Sun, Sandip Ghoshal, Michelle Yuan, Yassine Benajiba, Sujith Ravi, Dan Roth  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.08782v1  

#### Abstract
Large language models (LLMs) suffer significant performance degradation when user instructions and context are distributed over multiple conversational turns, yet multi-turn (MT) interactions dominate chat interfaces. The routine approach of appending full chat history to prompts rapidly exhausts co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《MT-OSC: Path for LLMs that Get Lost in Multi-Turn Conversation》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在多轮对话（multi-turn, MT）中表现显著下降，主要原因包括：
- **信息分散**：用户指令和关键上下文分布在多个回合中，导致模型难以准确整合信息。
- **上下文窗口压力**：传统方法将完整对话历史拼接到提示（prompt）中，迅速耗尽 context window，增加延迟和计算成本。
- **信息丢失与错误累积**：随着对话延长，模型容易遗忘早期细节或基于错误假设进行推理。

### 提出了什么新方法或新思路
提出 **MT-OSC**（One-off Sequential Condensation），一种**后台自动对话历史压缩框架**，其核心思想是：
- 在不干扰用户体验的前提下，动态地对对话历史进行**逐段压缩**（w turns at a time）。
- 引入 **Condenser Agent**，由两个组件构成：
  - **Condenser**：基于 few-shot 推理的 LLM Agent，通过精心设计的示例（exemplars）学习保留关键信息（如原始用户指令、数值、否定词等）。
  - **Decider**：轻量级决策模块，基于内容重叠度和 token 数量决定是否执行压缩，防止在高密度信息对话中丢失关键内容。

### 相比现有方法的优势
| 方法 | 局限性 | MT-OSC 的优势 |
|------|--------|----------------|
| **Concatenation (MT-baseline)** | 上下文线性增长，资源消耗大 | 显著减少 token 数量（最高达 72%） |
| **Ad-hoc summarization** | 风险遗漏关键细节，泛化能力差 | Few-shot Condenser 更鲁棒，能识别并保留关键片段 |
| **Fine-tuning / Retrieval-based** | 需要训练或额外检索系统，部署复杂 | **无需微调、无架构修改、可插拔集成** |
| **RECAP / SNOWBALL** | 只在最后一步干预，无法处理持续对话 | 支持**持续、渐进式压缩**，适用于任意长度对话 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
共评估 **10 个多样化多轮数据集**，涵盖两类场景：

#### （1）Sharded Datasets（模拟信息碎片化）
将单轮任务人为拆分为多轮，测试模型整合能力：
- **GSM8K-shrd**：数学推理
- **BFCL-V3 Parallel-shrd**：函数调用
- **HumanEval-shrd (HEval)**：代码生成
- **Spider-shrd**：Text-to-SQL
- **ToTTo-shrd**：表格到文本生成
- **Summary of Haystack-shrd (SoH)**：长文档摘要

#### （2）MT-EVAL Datasets（真实对话流）
来自 MT-Eval benchmark 的四类自然对话模式：
- **Recollection**：回忆型问答
- **Refinement**：逐步完善答案
- **Expansion**：扩展已有信息
- **Follow-up**：后续追问

此外，为测试鲁棒性，在上述数据上注入三种噪声：
- **Repetition Infusion**：重复随机对话轮次
- **Filler Injection**：插入“Um”、“Well”等无意义填充语
- **Contextual Diversion**：添加主题相关但无关的任务干扰句

### 实验设置和评估指标

| 维度 | 设置 |
|------|------|
| **主模型** | Llama-3.3-70B-Instruct（默认） |
| **窗口大小 w** | 2, 3, 4 轮压缩一次 |
| **评估方式** | 平均三次运行结果 |
| **背景压缩机制** | 在第 `w+2` 轮时启动压缩，替换前 `w` 轮历史 |

#### 评估指标（按任务定制）
| 数据集类型 | 指标 |
|-----------|------|
| GSM8K, BFCL, HEval, Spider, Recoll | **Accuracy**（精确匹配） |
| Refinement, Follow-up, Expansion | **LLM-as-a-Judge 评分**（10 分制） |
| ToTTo | **BLEU Score** |
| SoH | **Joint Score F1**（覆盖 + 归因准确性） |

### 基线方法对比
- **MT-baseline**：直接拼接所有历史（concatenate all turns）
- **Simple Summarization (Summ)**：使用简单摘要 prompt 替代 Condenser（消融对照）
- **RECAP / SNOWBALL / CONCAT**：来自 Laban et al. (2025) 的基线（文中指出其不可行）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **平均 token 减少率** | **30.9%**（所有样本）<br>**45.5%**（≥6 轮对话）<br>**最高达 72%**（10 轮对话） |
| **端到端延迟降低** | 平均节省 **1,789 tokens/轮** → **约 1–1.2 秒 TTFT 降低** |
| **总体性能变化** | **保持或提升精度**，无统计显著下降（p=0.19） |

### 与基线方法的对比结果（见 Table 1 & Figure 3）

#### 性能 vs Token 消耗双优
| Dataset | MT-baseline Acc | MT-OSC (w=4) Acc | Token Reduction |
|--------|------------------|------------------|------------------|
| BFCL-shrd | 81.13% | **86.79%** ↑ | 263 → 134 (**49%↓**) |
| GSM-shrd | 83.45% | **84.80%** ↑ | 1026 → 879 (**14%↓**) |
| HEval-shrd | 74.67% | **77.33%** ↑ | 332 → 194 (**42%↓**) |
| Spider-shrd | 76.95% | **79.44%** ↑ | 167 → 167 (=) |
| SoH-shrd | 0.13 | 0.13 (=) | 16k → 16k (=) |

> 注：SoH 和 ToTTo 未压缩是因为 Decider 判断信息密集而跳过。

#### 多模型验证（13 个 SOTA LLMs）
- 在 **GPT-5, GPT-4o, Llama-3/4, Grok-3/4** 等模型上一致有效
- GPT-5 基线已达 90%，MT-OSC 进一步提升至 **92%**
- 所有模型平均 accuracy 提升 **+4%**

### 消融实验结果

#### （1）Decider 模块作用（Table 2）
| Dataset | w/ Decider Acc | w/o Decider Acc | Token Reduction |
|--------|----------------|------------------|------------------|
| ToTTo-shrd | 0.18 | **0.09** ↓ | 4670 → 177 (**96%↓**) |
| SoH-shrd | 0.13 | **0.08** ↓ | 16063 → 2360 (**85%↓**) |

👉 **结论**：移除 Decider 导致严重性能下降，说明其有效防止关键信息被过度压缩。

#### （2）Condenser vs Simple Summarization（Figure 3）
- Simple summarization 在所有 window size 下均**低于 MT-OSC**
- 原因：无法识别需保留的原始指令或数值，易误删关键信息
- 示例见 Appendix E：simple summarizer 错误合并 assistant 假设，导致后续推理混乱

#### （3）跨 Condenser 模型泛化性（Table 5）
| Condenser Model | BFCL-shrd Acc | HEval-shrd Acc |
|----------------|---------------|----------------|
| Llama-3.3-70B | 91.36% | 90.12% |
| Llama-4-Maverick | **92.59%** | 85.19% |
| GPT-4.1 | 85.19% | **88.89%** |
| Gemini-2.5-Flash | 82.72% | 80.25% |

👉 **结论**：MT-OSC 框架对不同 Condenser LLM 具有良好泛化性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **MT-OSC 显著缓解了 multi-turn 性能退化问题**，在多数任务上实现**精度持平或提升**。
2. ✅ **token 消耗大幅降低**（平均 30.9%，最长达 72%），有效释放 context window。
3. ✅ **完全后台运行**，不影响用户交互体验，适合生产环境部署。
4. ✅ **对噪声和干扰具有强鲁棒性**：在重复、填充、干扰句存在时仍能维持甚至扩大性能优势（Figure 7）。
5. ✅ **通用性强**：适用于多种任务类型（math, code, SQL, summarization）和多种 LLM 架构。

### 方法的局限性
- 当前评估基于 **≤12 轮**的对话，尚未验证超长对话（如 >20 轮）效果。
- 数据集以**单主题、清洁对话为主**，缺乏复杂多轮 agentic 行为（如工具调用链、长期记忆管理）。
- **未探索压缩过程本身的 cost**：虽然总 token 下降，但 Condenser 本身也消耗资源（尽管后台执行）。
- Decider 规则较简单，依赖词重叠度，可能无法捕捉语义层面的信息密度。

### 未来工作方向
- 构建更复杂的 **longer, multi-topic, agent-driven 多轮对话数据集**。
- 将 MT-OSC 与 **retrieval-augmented generation (RAG)** 或 **memory-augmented agents** 结合。
- 探索 **topic-aware compression** 或 **intent-driven condensation policy**。
- 研究如何在 **edge devices 或低延迟系统**中优化 Condenser 的轻量化实现。

--- 

> **一句话总结**：  
> MT-OSC 提供了一种**无需训练、可插拔、高效且鲁棒的多轮对话压缩方案**，解决了 LLMs “在多轮对话中迷失”的核心痛点，在降低延迟的同时提升了任务性能，为现实世界 chat system 的规模化部署提供了实用路径。

</details>

---

### 8. [Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](https://arxiv.org/abs/2604.08964)

**Authors**: Shun Zou, Yong Wang, Zehui Chen, Lin Chen, Chongyang Tao, Feng Zhao, Xiangxiang Chu  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.08964v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have recently become a promising alternative to autoregressive large language models (ARMs). Semi-autoregressive (Semi-AR) decoding is widely employed in base dLLMs and advanced decoding strategies due to its superior performance. However, our observations rev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本文针对 **Diffusion Large Language Models (dLLMs)** 中广泛采用的 **Semi-autoregressive (Semi-AR) decoding** 存在的关键缺陷——**块边界延迟（Block-Boundary Delay）**。

- 在 Semi-AR 解码中，序列被划分为多个 block，必须按顺序解码，即使某些 token 已经提前稳定（即其预测值不再变化），也需等待所在 block 被处理才能释放。
- 这些“跨块稳定 token”（cross-block stable tokens）的延迟解码导致大量冗余计算，浪费解码步数，并抑制局部区域的收敛（radiative effects），影响生成质量和效率。

### 🚀 提出了什么新方法或新思路

作者提出 **Anchor-based History-stable Decoding (AHD)**，一种无需训练、即插即用（training-free, plug-and-play）的动态解码策略。

#### 核心思想：
- **基于动态锚点（dynamic anchor）监控历史轨迹**，识别 token 是否进入“绝对稳定性趋势”（absolute stability trend）。
- 一旦检测到某 token 在未来 block 中已稳定，即可**提前解锁并参与当前解码**，打破 block 边界限制。

#### 三大关键洞察（Insights）驱动设计：
1. **Naive lookahead decoding is unreliable**：仅依赖单步置信度（如 confidence 或 entropy）判断稳定性易受局部波动干扰，可能导致误判。
2. **Token stability correlates with convergence trend**：稳定 token 在首次稳定后，其置信度呈现快速单调上升趋势（绝对稳定性趋势），应通过轨迹而非峰值判断。
3. **Historical information is isolated**：标准 dLLM 解码忽略中间步骤的历史预测，最终输出仅由最后一步决定，造成历史信息隔离。

### 🔍 相比现有方法的优势

| 维度 | AHD | 现有加速方法（如 Fast-dLLM, KLASS, Saber） |
|------|-----|----------------------------------------|
| **是否牺牲性能换速度** | ❌ 否 —— 同时提升性能与效率 | ✅ 是 —— 多数方法加速伴随性能下降 |
| **能否跨 block 提前解码** | ✅ 是 —— 主动识别并释放未来 block 的稳定 token | ❌ 否 —— 仍受限于 block 顺序 |
| **是否需要训练** | ✅ 否 —— 完全推理阶段优化 | 多数否，但部分需微调缓存机制 |
| **鲁棒性** | ✅ 高 —— 利用历史一致性降低局部波动影响 | 低 —— 易受单步噪声干扰 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

实验覆盖三大领域共 **17 个基准任务**：

#### 语言领域（Language Domain）
- **代码生成**：`HumanEval`, `MBPP`
- **通用能力**：`BBH`, `MMLU-Pro`, `TruthfulQA`
- **数学推理**：`Math`, `Asdiv`

#### 视觉-语言领域（Vision-Language）
- `MATH-Vision`, `MathVista`, `ScienceQA`, `GQA`, `MME`

#### 音频-语言领域（Audio-Language）
- `VoiceBench` 上的五个任务：`AlpacaEval`, `OpenBookQA`, `CommonEval`, `BBH`, `Wildvoice`

---

### ⚙️ 实验设置和评估指标

#### 模型
- **LLaDA-8B-Instruct**, **LLaDA-1.5**（语言模型）
- **MMaDA-8B-MixCoT**（多模态模型）
- **DIFFA**（音频语言模型）

#### 设置
- 生成长度：默认 256（长序列扩展至 1024 和 2048）
- Block 长度：默认 32
- 历史缓冲区长度 $ H $：默认 6
- 历史一致性阈值 $ \epsilon $：默认 0.01

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Score ↑** | 各任务的标准准确率或通过率（如 Pass@1） |
| **Decoding Steps ↓** | 所需迭代解码步数，越少越好 |
| **Latency / Speed-up ×** | 推理延迟或加速比，反映实际效率 |
| **TPF (Time Per Forward)** | 单次前向传播时间，用于归一化比较 |

---

### 🔁 基线方法对比

与以下主流解码策略进行对比：
- **Vanilla / Naive decoding**：原始 Semi-AR 解码
- **PC-sampler**：位置感知校准采样器
- **Fast-dLLM**：基于 confidence-aware 并行解码
- **KLASS**：基于 KL 散度引导的快速推理
- **Saber**：自适应加速 + 回溯重掩码机制

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 LLaDA-8B-Instruct 为例）

| Benchmark | 方法 | Score ↑ | Steps ↓ | 对比 Vanilla 步骤减少 |
|----------|------|--------|--------|------------------|
| **BBH** | AHD (Ours) | **56.78** (+3.67) | **51.48** | **-80%** |
| **HumanEval** | AHD | **43.29** (+2.44) | **77.24** | -70% |
| **MMLU-Pro** | AHD | **37.42** (+1.85) | **133.06** | -48% |
| **TruthfulQA** | AHD | **41.49** (+1.10) | **52.91** | -79% |

> 💡 **亮点**：AHD 在所有 7 个语言任务上均实现 **性能提升 + 解码步数显著下降**，而其他先进方法大多以性能为代价换取速度。

---

### 🔬 多模态与音频领域结果

#### 多模态（MMaDA on MathVista-mini）
| 方法 | Score ↑ | Speed-up × |
|------|--------|------------|
| Vanilla | 32.90 | 1.00× |
| AHD (Ours) | **36.00** (+3.10) | **2.37×** |

#### 音频（DIFFA on VoiceBench）
| 任务 | 方法 | Score ↑ | Steps ↓ | 减少 |
|------|------|--------|--------|-----|
| OpenBookQA | AHD | **38.50** (+2.00) | **28.67** | -78% |

> ✅ 表明 AHD 具备良好的跨模态泛化能力。

---

### 🔍 消融实验结果（Ablation Study）

#### 不同生成长度的影响（Generation Length）
| Length | 方法 | BBH Score | Steps |
|-------|------|---------|-------|
| 128 | AHD | 50.07 | 33.11 (-74%) |
| 256 | AHD | 56.78 | 51.48 (-80%) |
| 512 | AHD | 59.05 | 60.68 (-88%) |

> 📈 随着长度增加，AHD 的优势更加明显，说明其对长序列更有效。

#### 不同 Block 长度的鲁棒性
| Block Size | 方法 | BBH Score | Steps ↓ |
|-----------|------|----------|--------|
| 16 | AHD | 56.12 | 59.88 |
| 32 | AHD | 55.69 | 52.62 |
| 64 | AHD | 55.57 | 48.93 |
| 128 | AHD | 53.49 | 48.08 |

> ✅ AHD 在不同 block 配置下均优于 baseline，表现出强鲁棒性。

#### 历史缓冲长度 $ H $ 与阈值 $ \epsilon $
- 最优 $ H = 6 $：太小无法捕捉趋势，太大无益且增加开销。
- 最优 $ \epsilon = 0.02 $：过小则释放慢，过大则释放不稳定 token 导致错误。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Semi-AR 解码存在严重 block-boundary 延迟**：大量 token 在 block 开始前已稳定，却被迫等待，造成资源浪费。
2. **token 稳定性可通过历史轨迹识别**：引入“绝对稳定性趋势”概念，比单步指标更可靠。
3. **AHD 可同时提升性能与效率**：打破 block 边界，提前释放稳定 token，不仅提速，还增强生成连贯性。
4. **该现象具有跨模态普适性**：在文本、视觉、音频等任务中均取得一致增益。

> 🧠 “**减少解码步数不一定导致性能下降**”——AHD 通过消除冗余迭代、保留必要演化过程，实现了质量与速度的双赢。

---

### ⚠️ 方法的局限性

1. **参数需轻微调优**：虽然无需训练，但 $ \epsilon $ 和 $ H $ 在不同任务间略有差异，增加了部署成本。
2. **尚未验证超大规模模型**：实验集中在 ~8B 参数模型，未在 72B 或 256B 级别测试有效性。
3. **额外计算开销虽小但仍存在**：需维护历史 buffer 并计算 KL 散度，尽管占比 <2%，仍有优化空间。

---

### 🔮 未来工作方向

1. **自动化超参选择**：设计轻量级模块动态调整 $ \epsilon $ 和 $ H $，提升零样本适应能力。
2. **扩展至更大规模模型**：在百B级 dLLMs 上验证 AHD 的可扩展性。
3. **结合 KV Cache 优化**：将 AHD 与 `dkv-cache`、`dllm-cache` 等机制融合，进一步提升端到端效率。
4. **探索完全非自回归路径**：基于 AHD 的早期稳定信号，尝试构建更激进的 non-AR 解码范式。

---

> 🔗 **开源地址**：[https://github.com/zs1314/AHD](https://github.com/zs1314/AHD)  
> 📄 **论文链接**：[arXiv:2604.08964](https://arxiv.org/abs/2604.08964)

</details>

---

### 9. [Sensor Placement for Tsunami Early Warning via Large-Scale Bayesian Optimal Experimental Design](https://arxiv.org/abs/2604.08812)

**Authors**: Sreeram Venkat, Stefan Henneking, Omar Ghattas  
**Category**: cs.DC  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.08812v1  

#### Abstract
Real-time tsunami early warning relies on distributed sensor networks to infer seismic sources and seafloor motion. Optimizing these networks via Bayesian optimal experimental design (OED) is exceptionally challenging for systems governed by hyperbolic partial differential equations, which lack the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sensor Placement for Tsunami Early Warning via Large-Scale Bayesian Optimal Experimental Design

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**海啸早期预警系统中的传感器布设优化问题**，即在有限预算下如何选择最优的海底压力传感器位置，以最大化对地震引发的海床运动（seafloor motion）推断的准确性，并最小化参数场的不确定性。

传统方法面临以下挑战：
- 海啸传播由**双曲型偏微分方程**（hyperbolic PDEs）控制，缺乏平滑性导致标准低秩近似失效；
- 参数空间维度极高（超过 $10^9$），使得贝叶斯最优实验设计（Bayesian OED）计算上不可行；
- 经典OED框架需要嵌套大量反演求解，成本极高。

### 🚀 提出的新方法与创新思路

1. **基于数据空间重构的大规模贝叶斯OED框架**
   - 利用 **Sherman-Morrison-Woodbury恒等式** 将高维参数空间的贝叶斯反演转化为低维“数据空间”（data space）中的矩阵操作；
   - 构造显式的**数据空间Hessian矩阵 $K$**（尺寸为 $N_d N_t \times N_d N_t$，远小于参数维度），将OED问题转化为**稠密矩阵子集选择问题**。

2. **Schur补更新驱动的贪婪算法**
   - 设计了一种基于**块Schur补更新**（block Schur complement update）的高效贪婪算法，避免每次候选传感器评估时重新进行完整的Cholesky分解；
   - 显著降低每步时间复杂度从 $O(k^3)$ 到 $O(k^2)$，总复杂度从 $O(|C|B^4N)$ 下降到 $O(|C|B^3N)$。

3. **多GPU并行架构与I/O流水线优化**
   - 实现了一个**分布式内存、多GPU并行框架**，结合PyTorch和MPI；
   - 引入**双缓冲机制**（double-buffered architecture）和隔离的CUDA/HIP流，实现I/O与GPU计算完全重叠；
   - 使用**独立POSIX I/O**访问HDF5格式的稠密矩阵 $K$，避免MPI-IO同步瓶颈。

4. **首次解决极端尺度PDE约束下的贝叶斯OED问题**
   - 成功应用于一个拥有**超十亿自由度参数场**的真实数字孪生模型（digital twin），无需降阶建模或代理模型。

### 🔍 相比现有方法的优势

| 方面 | 本文方法 | 传统方法 |
|------|----------|---------|
| 可扩展性 | 支持 $O(10^9)$ 参数维度 | 仅适用于低维或椭圆/抛物系统 |
| 计算效率 | 多GPU强/弱可扩展性接近理想 | 难以并行，易受内存限制 |
| 模型保真度 | 使用全保真PDE模型（无近似） | 常依赖低秩/降阶模型 |
| I/O处理 | 完全隐藏I/O延迟 | I/O成为主要瓶颈 |

---

## 2. 核心实验方法和设置

### 📊 数据集与物理背景
- **应用场景**：Cascadia Subduction Zone（CSZ）海啸预警数字孪生系统（获2025年Gordon Bell奖）；
- **前向模型**：声重力波方程（acoustic-gravity wave equations），通过MFEM有限元求解；
- **候选传感器集合**：共 **600个候选位置**，均匀分布在CSZ区域（见图1）；
- **目标参数场**：时空连续的海床位移场，离散后具有 **超过10亿自由度**；
- **观测数据**：合成的海底压力传感器时间序列（每秒采样，持续7分钟 → $N_t = 420$）；

### ⚙️ 实验设置
- **OED准则**：采用 **D-optimal design**，即最小化后验协方差矩阵行列式，等价于最大化信息增益（EIG）；
- **目标函数**（简化形式）：
  $$
  \Phi(S) = -\log \det(K_S)
  $$
  其中 $K_S$ 是对应传感器子集的数据空间Hessian；
- **传感器预算**：从600个候选中选出 $B = 175$ 个最优位置；
- **算法流程**：使用Algorithm 1中的贪婪算法逐个添加传感器。

### 💻 硬件平台
- **Perlmutter**（NERSC）：NVIDIA A100 GPU集群（最多512 GPU）
- **Frontier**（OLCF）：AMD MI250X GPU集群（最多1024 GCDs）
- 存储：Lustre文件系统，$K$ 矩阵以chunked HDF5格式存储（464 GB，双精度）

### 📈 评估指标
- 单候选评估时间（per-candidate evaluation time）
- 总运行时间（wall-clock time）
- 并行效率（strong/weak scaling）
- 内存占用（peak GPU memory）
- 最终选型的目标函数值 vs 随机配置

### 🔀 基线方法对比
- **Naive greedy算法**：每次对测试矩阵从头进行Cholesky分解（非增量更新）；
- **随机选择策略**：生成100组随机的175传感器组合，统计其目标函数分布；
- （注：由于问题规模极大，无法与其他OED方法直接比较）

---

## 3. 主要实验结果和性能指标

### 📈 性能表现

#### ✅ 单GPU性能（图2）
| GPU型号 | 最大支持预算 $B$ | Schur方法加速比（vs Naive） |
|--------|------------------|----------------------------|
| NVIDIA A100 (80GB) | 340 | $\sim 10^3\times$ 更快 |
| AMD MI250X (64GB) | 300 | 同量级加速 |
| NVIDIA GH200 (96GB) | 370 | 支持更大预算 |

- Schur方法展现出 $O(k^2)$ 时间增长趋势，而朴素方法为 $O(k^3)$；
- 内存使用也显著更低，允许更大的传感器预算。

#### ✅ 多GPU可扩展性（图3）
- **强可扩展性**（Fixed problem size: 8192 candidates）：
  - 在 **512 GPUs / 1024 GCDs** 上实现近乎理想的线性加速；
  - 效率维持在 **97%-101%** 范围内（考虑测量误差）；
- **弱可扩展性**（Per-GPU workload固定为256 candidates）：
  - 规模扩展至 **262,144候选评估任务**（Frontier）；
  - 同样保持接近完美的扩展效率。

> 💡 关键原因：**I/O与计算完全重叠**，消除了传统实现中I/O等待导致的性能塌陷。

#### ✅ 实际部署结果（CSZ数字孪生）
- **计算耗时**：
  - 构造 $K$ 矩阵：约3小时（512 A100 GPUs）
  - 运行OED算法（$B=175$）：**仅需1.5小时（16 A100 GPUs）**
- **结果质量**（图4）：
  - 贪婪最优配置的目标函数值远优于100次随机配置（高出数量级）；
  - 不确定性随传感器增加逐步下降（图5），验证了信息增益的有效积累。

### 🔍 消融实验与理论分析
- **子模性（Submodularity）验证**：
  - D-optimal目标函数是子模函数，保证贪婪算法至少达到全局最优的 $(1 - 1/e) \approx 63\%$；
- **精度影响**：
  - 使用单精度（float32）进行线性代数运算，相比双精度（float64）**结果一致**，但内存减半、吞吐更高；
- **内存优化效果**：
  - 全局预分配Cholesky因子缓冲区，消除动态分配开销；
  - 零分配（zero-allocation）内循环防止GPU内存碎片化。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **大规模贝叶斯OED可用于真实世界极端尺度PDE系统**：
   - 首次成功解决了基于全保真模型、参数维度超十亿的OED问题；
   - 无需任何模型简化、降阶或代理建模。

2. **数据空间重构 + Schur更新是突破性技术路径**：
   - 将原本不可行的OED问题转化为可计算的组合优化；
   - 数学重构与高性能计算协同设计至关重要。

3. **软硬件协同优化释放极致性能潜力**：
   - 多GPU并行 + 双缓冲I/O流水线实现了接近理想的可扩展性；
   - 为未来科学机器学习中的“外环优化”提供了范式参考。

4. **为CSZ海啸监测网络建设提供决策支持**：
   - 提供了数学上严谨、不确定性感知的传感器布局方案；
   - 可灵活集成成本模型、区域优先权重等实际约束（见Section III-G）。

### ⚠️ 局限性
- 当前方法主要适用于 **线性时不变系统**（LTI dynamical systems）；
- 对非线性或时变系统需引入局部线性化或其他近似；
- 贪婪算法虽具理论保证，但仍为近似解法，不能保证全局最优；
- 构造 $K$ 矩阵本身仍需大量离线计算资源（~500小时 on 512 GPUs）。

### 🔮 未来工作方向
1. 扩展至**非线性贝叶斯OED**，结合迭代线性化策略（如Gauss-Newton）；
2. 探索**lazy greedy** 或 **stochastic greedy** 算法在更大候选池下的应用；
3. 结合**主动学习**框架，在线更新传感器布局；
4. 应用于其他LTI系统场景：如大气污染溯源、地下水流监测、GNSS地震预警等；
5. 开发开源软件框架，推广至更广泛的地球物理与工程领域。

---

> **一句话总结**：  
> 本论文通过**数据空间重构 + Schur补更新 + 多GPU流水线优化**，首次实现了在**超十亿参数场**上的PDE约束贝叶斯OED，为Cascadia海啸预警系统的传感器布设提供了高效、可扩展且数学严谨的解决方案，标志着高性能计算与不确定性量化融合的重大进展。

</details>

---

### 10. [Integrated electro-optic attention nonlinearities for transformers](https://arxiv.org/abs/2604.09512)

**Authors**: Luis Mickeler, Kai Lion, Alfonso Nardi, Jost Kellner, Pierre Didier, Bhavin J. Shastri, Niao He, Rachel Grange  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.09512v1  

#### Abstract
Transformers have emerged as the dominant neural-network architecture, achieving state-of-the-art performance in language processing and computer vision. At the core of these models lies the attention mechanism, which requires a nonlinear, non-negative mapping using the Softmax function. However, al...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Integrated Electro-Optic Attention Nonlinearities for Transformers*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **Softmax瓶颈问题**：在基于Transformer的模型（如LLM和ViT）中，尽管Softmax操作仅占总计算量（FLOPs）的不到1%，但由于其非线性特性及对Special Function Units (SFUs)的依赖，其执行速度远低于矩阵乘法等线性运算，导致成为推理延迟的关键瓶颈。
- **数字硬件的效率限制**：现代GPU中，Softmax的吞吐量可能比Tensor Core低256倍，严重制约了Transformer的整体性能。

### 提出了什么新方法或新思路
- **提出电光非线性单元**：利用**薄层铌酸锂**（TFLN）平台上的**马赫-曾德尔调制器**（MZM）作为模拟非线性计算元件，直接实现Softmax和Sigmoid函数的物理近似。
- **两种新型电光注意力机制**：
  - **Optmax**：用MZM的上升沿近似指数函数（numerator），下降沿近似归一化倒数（denominator），实现对Softmax的替代。
  - **Optmoid**：利用MZM从最小到最大的完整电压摆幅来近似Sigmoid函数，实现更高效的逐元素注意力。
- **混合数字-模拟架构**：提出一种**共封装光学**（co-packaged optics）方案，将高速DAC/ADC与TFLN MZM集成，形成“光学作为非线性单元”的混合计算系统。

### 相比现有方法的优势
- **极低延迟**：相比现有电子和光子加速器，Optmax和Optmoid可将注意力非线性计算的延迟降低一个数量级以上（Optmax）甚至两个数量级（Optmoid）。
- **高能效**：在10 GBaud下，Optmax和Optmoid的能量消耗分别为10.0 pJ和4.7 pJ每序列，具有竞争力。
- **高鲁棒性**：即使在**4-bit输入输出量化**条件下，模型仍能保持高度竞争性的准确率。
- **避免复杂光子组件**：不同于需要半导体光放大器、波长路由查找表或多级光电转换的全光方案，该方法仅需标准MZM，简化了系统并提升了可扩展性。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **计算机视觉任务**：
  - **MNIST**：手写数字识别
  - **CIFAR-10**：10类图像分类
  - **SVHN**：街景门牌号识别
- **自然语言处理任务**：
  - **FineWeb-Edu**：用于因果语言建模（Causal Language Modeling, CLM）的大规模教育文本数据集。

### 实验设置和评估指标
- **模型架构**：
  - **Vision Transformer (ViT)**：用于图像分类任务。
  - **GPT-2 (124M参数)**：用于语言建模任务。
- **评估指标**：
  - **图像分类**：测试集准确率（Test Accuracy）
  - **语言建模**：测试集负对数似然（Test Loss）
- **量化设置**：对输入（DAC）和输出（ADC）进行2-bit至全精度的均匀量化，重点分析4-bit条件下的性能。
- **噪声鲁棒性测试**：在推理阶段注入加性高斯噪声 $ \mathcal{N}(0, \sigma) $，评估模型在不同噪声水平下的性能退化。
- **训练细节**：
  - 所有模型均使用相同的超参数（除注意力非线性外）。
  - 使用拟合的MZM响应函数（而非原始测量数据）进行反向传播训练，以保证可微性。

### 基线方法对比
- **数字基线**：
  - **Softmax**：标准注意力机制。
  - **Sigmoid**：无归一化的逐元素注意力变体。
- **硬件基线**（用于延迟/能耗对比）：
  - **nMOS SMA**（电子模拟）
  - **Softermax**（数字电子）
  - **SOFTONIC**（集成光子）
  - **VEXP**（RISC-V指令扩展）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | Optmax | Optmoid | Softmax/Sigmoid (Baseline) |
|------|--------|---------|----------------------------|
| **ViT 测试准确率 (CIFAR-10, 全精度)** | 75.34% | 74.06% | 75.34% / 74.06% |
| **ViT 测试准确率 (4-bit)** | 74.6% | 69.9% | 76.3% / 75.9% |
| **GPT-2 测试损失 (FineWeb-Edu, 全精度)** | 4.08 | 4.22 | 4.07 / 4.18 |
| **GPT-2 测试损失 (4-bit)** | 5.85 | 5.89 | 5.97 / 5.97 |
| **延迟 (n=64, 10 GBaud)** | 13 ns | 6.5 ns | — |
| **能量 (n=64, 10 GBaud)** | 10.0 pJ | 4.7 pJ | — |

### 与基线方法的对比结果
- **准确性**：
  - 在ViT任务中，**Optmax**在全精度下与Softmax**完全持平**，在4-bit下仅略降。
  - 在GPT-2任务中，**Optmax**的测试损失（4.08）与Softmax（4.07）几乎相同，**表现极具竞争力**。
  - **Optmoid**在语言模型中表现出优于Sigmoid的量化鲁棒性。
- **延迟与能效**：
  - 如Table I所示，**Optmax**和**Optmoid**在延迟上显著优于所有对比的电子和光子方案（如SOFTONIC、VEXP等）。
  - **Optmoid**因仅需单次调制，延迟和能耗均为最低。

### 消融实验结果
- **量化敏感性**：
  - Optmax在4-bit下性能下降较小，表明其对低精度接口友好。
  - Optmoid在4-bit下性能下降较多，归因于偏置（bias）设置导致大量激活值被截断为零。
- **噪声鲁棒性**：
  - **加性噪声是主要挑战**：当噪声水平 $ \sigma > 0.02 $ 时，4-bit模型性能急剧下降，因为噪声会将本应为零的权重“激活”。
  - **乘性噪声更鲁棒**：模型对乘性噪声（如增益波动）表现出更强的鲁棒性。
  - **噪声感知训练有效**：在训练中引入噪声可显著提升模型在噪声环境下的推理性能，甚至在某些情况下提升准确率。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **电光非线性可行且高效**：TFLN MZM可以作为高性能的模拟非线性计算单元，有效替代Transformer中的Softmax和Sigmoid操作。
2. **延迟瓶颈可突破**：通过将非线性计算“卸载”到光学域，可从根本上解决GPU中SFU带来的延迟瓶颈。
3. **低比特量化下仍具竞争力**：即使在4-bit量化下，模型仍能保持接近全精度的性能，证明了该架构适用于高速、低功耗场景。
4. **注意力机制对动态范围不敏感**：尽管MZM的正弦响应无法完美复现无限动态范围的指数函数，但只要保留**非负性**和**非线性重加权**这两个关键属性，模型性能即可维持。

### 方法的局限性
- **加性噪声敏感**：系统对加性噪声（来自RF放大器、EDFA、PD等）极为敏感，尤其是在低比特量化下，易导致注意力分布失真。
- **动态范围受限**：MZM的周期性和单位上限传输特性限制了其逼近理想指数函数的能力。
- **需要校准与拟合**：实际部署需对MZM响应进行精确建模和校准，并在训练中使用拟合函数，增加了系统复杂性。
- **当前为概念验证**：实验基于离散器件搭建，尚未实现大规模集成。

### 未来工作方向
- **降低加性噪声**：优化RF链路、采用更低噪声的放大器和探测器。
- **开发噪声感知训练协议**：在训练中显式建模和对抗硬件噪声，提升部署鲁棒性。
- **探索其他模拟非线性**：如微环谐振器的洛伦兹响应、电吸收调制器的指数衰减等。
- **实现全集成芯片**：将TFLN MZM、激光器、PD、TIA等集成在同一芯片上，迈向实用化。
- **扩展至其他非线性函数**：将该范式推广至GeLU、Swish等其他神经网络激活函数。

</details>

---

### 11. [E3-TIR: Enhanced Experience Exploitation for Tool-Integrated Reasoning](https://arxiv.org/abs/2604.09455)

**Authors**: Weiyang Guo, Zesheng Shi, Liye Zhao, Jiayuan Ma, Zeen Zhu, Junxian He, Min Zhang, Jing Li  
**Category**: cs.AI  
**Published**: 2026-04-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.09455v1  

#### Abstract
While Large Language Models (LLMs) have demonstrated significant potential in Tool-Integrated Reasoning (TIR), existing training paradigms face significant limitations: Zero-RL suffers from inefficient exploration and mode degradation due to a lack of prior guidance, while SFT-then-RL is limited by ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# E3-TIR: Enhanced Experience Exploitation for Tool-Integrated Reasoning 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前在 **Tool-Integrated Reasoning (TIR)** 领域中，主流训练范式面临两大瓶颈：

- **Zero-RL**（零样本强化学习）：缺乏先验指导，导致探索效率低下（inefficient exploration），容易陷入“react mode”（过度依赖工具调用而忽略推理），并出现模式退化（mode degradation）。
- **SFT-then-RL**（监督微调后接强化学习）：依赖大量合成数据进行 SFT，成本高昂；且由于策略熵过低，易发生 **low-entropy collapse**（能力僵化），限制了后续 RL 阶段的提升空间。

这些问题共同导致训练效率低、样本利用率差、最终性能受限。

---

### 提出了什么新方法或新思路

作者提出 **E3-TIR**（Enhanced Experience Exploitation for Tool-Integrated Reasoning），一种面向智能体早期训练的“热身”（warm-up）范式，其核心思想是：

> **动态融合三种经验来源**：Expert Prefixes（专家前缀）、Expert Guided（专家引导探索）、Self-Exploration（自我探索），实现高效的经验利用。

#### 核心机制包括：

- **Expert-Guided Branch Sampling**  
  在高质量专家轨迹的高熵节点（high-entropy “anchors”）处进行分支采样，围绕“锚点”展开多样化探索，显著提升探索效率和梯度价值。

- **Dynamic Experience Filtering**  
  基于奖励方差和相对性能对混合轨迹进行动态筛选，确保训练数据的质量与多样性平衡。

- **Mix Policy Optimization with Advantage-Aware Gradient Detaching (AAGD)**  
  引入混合优势估计（hybrid advantage estimation），结合全局优势（Aglobal）与子树内优势（Atree），并通过 **AAGD** 技术解决共享前缀上的梯度冲突，保证训练稳定性。

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **训练效率** | 显著加速收敛，避免 Zero-RL 的冷启动问题 |
| **数据成本** | 仅需不到 10% 的合成数据即可超越传统方法 |
| **性能上限** | 缓解 SFT+RL 的低熵崩溃，实现持续性能增长 |
| **ROI（投资回报率）** | 综合性能、数据成本与训练效率，达到基线的 **1.46× ROI 增益** |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

#### 任务类型与评测基准：

- **数学推理任务（Mathematical Reasoning）**：
  - `AIME24`, `AIME25`, `AMC23`, `GSM8K`, `MATH500`

- **知识密集型问答任务（Knowledge-Intensive Reasoning）**：
  - `HotpotQA`, `2WikiMultiHopQA`, `Musique`, `Bamboogle`, `SimpleQA`

#### 训练数据集：

- **Tool-Star-SFT-54K**：用于 SFT 阶段的高质量工具集成推理数据。
- **Tool-Star-RL-10K**：用于 RL 阶段的约 10,000 个交互任务样本。
- **Expert Trajectories**：来自 Tool-Star 的 SFT 数据中提取的专家轨迹作为“锚点”。

---

### 实验设置和评估指标

#### 模型架构
- 主要基于 `Qwen2.5-3B/7B-Instruct` 和 `Llama3.1-8B-Instruct` 进行实验。

#### 评估指标
- **Avg. Score**：各任务平均得分（准确率或 F1）
- **Avg. RMR**（React-Mode Ratio）：衡量工具滥用程度
- **Tool-calling Behavior**：平均调用次数、失败率、冗余率
- **ROI**（Return on Investment）：综合性能、训练步数、数据规模的复合效率指标

#### 训练流程
- **E3-TIR 分两阶段**：
  1. **Warm-Up**：使用少量专家数据 + 自我探索进行混合策略优化
  2. **Post-RL**：切换至标准 RL 数据继续训练

---

### 基线方法对比

| 类别 | 基线方法 |
|------|--------|
| **训练范式** | Only SFT, SFT-then-RL, Zero-RL |
| **LLM-based Math Agent** | ToRL, SimpleTIR |
| **LLM-based Search Agent** | Search-ol, Search-R1, Tree-GRPO |
| **Multi-Tool Integrated Agent** | Tool-Star, ReCall, ARPO |
| **离线融合方法** | Luffy, HPT（用于 ROI 对比） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 方法 | 平均得分（3B） | ROI 提升 |
|------|----------------|----------|
| SFT-then-RL | 44.2 | 1.00× |
| Zero-RL | 43.2 | — |
| **E3-TIR (Ours)** | **46.7** | **1.46×** |

> 在 `Qwen2.5-3B-Instruct` 上，E3-TIR 实现 **6% 的绝对性能提升**，同时仅使用 **<10% 合成数据**。

---

### 与基线方法的对比结果

- 在所有模型尺度下（3B/7B/8B），E3-TIR 均优于各类 SOTA 方法。
- 在数学与知识类任务上均取得最佳表现，尤其在复杂多跳推理任务（如 Musique、Bamboogle）上优势明显。
- 相比 SFT-then-RL，E3-TIR 避免了后期性能停滞甚至下降的问题，表现出更稳定的上升趋势。

---

### 消融实验结果

| 消融配置 | HotQA ↓ | Musique ↓ | AMC23 ↓ | MATH ↓ | 结论 |
|---------|--------|---------|--------|-------|------|
| w/o Branch | -4.2 | -3.8 | -3.7 | -1.6 | 分支探索贡献最大 |
| w/o AAGD | -36.1 | -18.7 | -31.2 | -36.6 | 导致训练崩溃，至关重要 |
| w/o Hybrid Adv. | -1.9 | -1.3 | -0.8 | -1.1 | 影响较小但仍重要 |
| Mix SFT+RL（加权融合） | -29.0 | -15.6 | -23.5 | -28.4 | 完全失效，验证设计必要性 |

> ✅ **关键发现**：Advantage-Aware Gradient Detaching 是稳定训练的关键；分支探索是性能增益的主要驱动力。

---

## 4. 关键结论和发现

### 论文的主要发现

1. **Zero-RL 存在严重的探索低效与模式退化问题**，表现为高工具失败率与 react mode 泛滥。
2. **SFT-then-RL 虽然起始性能高，但易陷入低熵崩溃**，限制长期发展。
3. **以专家轨迹为“锚点”进行分支探索**，能有效缩小搜索空间、提高梯度质量，实现“快热身、稳增长”。
4. **混合策略优化 + AAGD 可解决共享前缀的梯度冲突**，使不同来源的经验协同而非干扰。
5. **极小量专家数据（如 2k–4k）即可带来显著收益**，验证了“质优于量”的经验利用理念。

---

### 方法的局限性

1. **对初始专家前缀质量敏感**：若种子轨迹质量差或多样性不足，可能误导整个探索过程。
2. **当前评估集中在数学与 QA 领域**，尚未充分验证在真实世界复杂场景中的泛化能力。
3. **长周期、多工具协同任务的适用性有待加强**，例如需要维护状态一致性的任务。

---

### 未来工作方向

- 探索自动筛选高质量“锚点”的机制，降低对人工标注专家轨迹的依赖。
- 将 E3-TIR 扩展到更多模态与工具组合（如视觉、语音、API 调用等）。
- 研究如何在动态环境中在线生成“虚拟专家”以支持终身学习。
- 构建更具挑战性的长视野、多代理协作任务基准，进一步验证框架极限。

---

> 🔗 **代码开源地址**：[https://github.com/yuki-younai/E3-TIR](https://github.com/yuki-younai/E3-TIR)

</details>

---

### 12. [UIPress: Bringing Optical Token Compression to UI-to-Code Generation](https://arxiv.org/abs/2604.09442)

**Authors**: Dasen Dai, Shuoqi Li, Ronghao Chen, Huacan Wang, Biao Wu, Qizhen Lan  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.09442v1  

#### Abstract
UI-to-Code generation requires vision-language models (VLMs) to produce thousands of tokens of structured HTML/CSS from a single screenshot, making visual token efficiency critical. Existing compression methods either select tokens at inference time using task-agnostic heuristics, or zero out low-at...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：UIPRESS: Bringing Optical Token Compression to UI-to-Code Generation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
UI-to-Code（UI2CODE）任务要求视觉语言模型（VLMs）从单张截图生成数千个 token 的结构化 HTML/CSS 代码，导致输入的视觉 token 序列极长（如 Qwen3-VL-8B 可产生约 6,700 个视觉 token）。这带来了严重的 **prefill 延迟** 和 **GPU 显存开销**，限制了实际部署效率。

现有压缩方法存在以下缺陷：
- **推理时压缩（inference-time methods）** 如 FastV、VisionZip 等仅通过零化特征或选择重要 token 进行“伪压缩”，并未真正缩短序列长度，因此无法减少 time-to-first-token（TTFT）。
- **启发式策略缺乏任务感知能力**，例如基于 L2 范数或注意力分数的选择方式未考虑 UI 截图中信息密度高度不均的问题（文本/按钮区域远比背景重要）。
- **光学压缩（optical compression）虽在 OCR 领域有效**（如 DeepSeek-OCR），但尚未被引入到 UI-to-Code 场景，且存在表示空间不匹配问题。

---

### 🚀 提出的新方法与创新思路
作者提出 **UIPRESS**（UI Pipeline for Representation-Efficient Screenshot Synthesis），是首个将 **encoder-side learned optical compression** 成功应用于 UI-to-Code 任务的方法。

#### 核心设计：
1. **轻量级光学压缩器（Optical Compressor）** 插入在冻结的 ViT 编码器与 LLM 解码器之间：
   - 使用 **depthwise-separable convolutions** 实现空间下采样；
   - 引入 **element-guided spatial reweighting**，利用 OmniParser 检测 UI 元素（如按钮、文本框），对高语义区域赋予更高权重；
   - 添加 **Transformer refinement layer** 恢复因池化丢失的长程依赖关系；
   - 将 ~6,700 视觉 token 压缩至固定预算 **K=256**。

2. **解码器适配机制：Low-Rank Adaptation (LoRA)**  
   在 LLM 解码器的所有 query 和 value 投影层中注入 LoRA 适配器（共 ~7.7M 参数），以桥接原始 token 与压缩后 token 的表示差距。

3. **端到端联合训练**
   - 冻结 ViT 和大部分 LLM 参数；
   - 仅训练 Compressor 和 LoRA 模块；
   - 总可训练参数仅 **~21.7M（占 8B 模型的 0.26%）**。

---

### 🔍 相比现有方法的优势
| 方法类型 | 是否真正减少序列长度？ | 是否任务感知？ | 是否降低 TTFT？ | 是否需额外训练？ |
|--------|----------------------|--------------|---------------|----------------|
| Feature Zeroing (FastV) | ❌（保留位置） | ❌ | ❌ | ❌ |
| Token Selection (VisionZip) | ✅ | ❌（通用启发式） | ✅ | ❌ |
| Resolution Scaling | ✅ | ❌（均匀降分辨率） | ✅ | ❌ |
| **UIPRESS (本文)** | ✅ | ✅（基于 UI 结构引导） | ✅✅✅（显著加速） | ✅（少量训练） |

> ✅ **首次实现“超越无损”（beyond lossless）效果**：在大幅压缩的同时反而提升了生成质量。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Design2Code [42]**：包含 485 个真实网页截图及其对应 HTML，用于主评估。
  - 划分：50 页作为验证集（消融实验用），其余 435 页为测试集。
- **WebSight [16]**：合成数据集，含 823K 对截图-HTML。
  - 使用其中 **50K 样本**用于训练 UIPRESS 模型；
  - 另取 100 页用于跨数据集泛化分析。

---

### ⚙️ 实验设置
- **基础模型**：Qwen3-VL-8B-Instruct（统一基线，确保公平比较）
- **输入分辨率**：原生分辨率下 ViT 输出 ~6,517–6,700 个视觉 token
- **目标压缩数量**：K = 256（压缩比达 25.5×）
- **训练细节**：
  - 使用 AdamW，学习率分别为：
    - Compressor: `2e-4` → `1e-6`（cosine decay）
    - LoRA: `2e-5` → `1e-6`
  - Batch size: 48（6×A40 GPU，梯度累积）
  - 训练周期：20 epochs
  - 最佳 checkpoint 按 Design2Code 验证集 CLIP 分数选取

---

### 📊 评估指标

#### ✅ 质量指标
- **CLIP Score**：使用 ViT-B/32（OpenCLIP）计算生成页面渲染图与原图之间的余弦相似度，衡量全局视觉保真度（布局、颜色、内容排布等）。

#### ⏱️ 效率指标
- **Time-to-First-Token (TTFT)**：反映 prefill 阶段延迟，直接受视觉 token 数影响；
- **End-to-end Latency**：完整生成时间；
- **Peak VRAM**：峰值显存占用。

所有测量均在单张 NVIDIA A40（48GB）上完成，控制变量一致。

---

### 🔁 基线方法对比
| 基线方法 | 类型 | 是否训练 | 特点 |
|--------|------|---------|------|
| Qwen3-VL (native) | 无压缩 | ❌ | 原始 ~6,517 tokens，性能上限参考 |
| Resolution Scaling (480px) | 分辨率压缩 | ❌ | 控制输入尺寸，输出 ~845 tokens |
| VisionZip-256 [61] | 推理时选择 | ❌ | 按 L2 范数选主导 token |
| EfficientUICoder [64] | 元素感知剪枝 | ❌ | 基于边缘检测保留关键 patch |
| FastV [5] | 特征零化 | ❌ | 注意力得分低则置零，不删序列 |

> 所有方法均在同一 Qwen3-VL-8B 上运行，仅 UIPRESS 经过微调。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Design2Code 验证集）

| 方法 | Tokens | CLIP Score | TTFT (ms) | 压缩比 |
|------|--------|------------|-----------|--------|
| No Compression | 6,517 | 0.7563 | 384 | 1.0× |
| Resolution Scaling (480px) | 845 | 0.7768 | 69 | 7.7× |
| VisionZip-256 | 256 | 0.7333 | 45 | 25.5× |
| FastV (75%) | 6,517* | 0.7540 | 385 | 1.0× |
| **UIPRESS-256 (Ours)** | **256** | **0.8127** | **42** | **25.5×** |

> ✅ **核心成果总结**：
- **+7.5% CLIP 提升**：相比无压缩基线（0.8127 vs. 0.7563）；
- **+4.6% 超越最强推理方法**：优于 resolution scaling（0.7768）；
- **+10.8% 超越同 token 数方法**：优于 VisionZip-256；
- **9.1× TTFT 加速**：从 384ms 降至 42ms；
- **显存与延迟全面优化**：peak VRAM 显著下降，端到端延迟降低约 27%。

---

### 🔍 消融实验结果（Ablation Study）

#### 表：组件消融（50-page subset）

| 配置 | CLIP Score | ΔCLIP |
|------|-----------|-------|
| Full UIPRESS (Conv + Pool + Refine + LoRA) | **0.8127** | — |
| -LoRA | 0.7046 | -0.108 |
| -Transformer Refinement | 0.7940 | -0.019 |
| -Depthwise-Sep Conv → Std Conv | 0.8020 | -0.011 |
| LoRA only (no compressor) | 0.7610 | -0.052 |
| Compressor only (no LoRA) | 0.7046 | -0.108 |

> 💡 发现：
- **LoRA 是最大贡献者**（单独带来 +10.8% CLIP 提升），说明解码器适配至关重要；
- **Transformer refinement 和 depthwise-sep conv 均有正向增益**；
- 各模块间存在强协同效应。

#### 目标 token 数量分析（K）
| K | CLIP | Latency (s) |
|----|------|-------------|
| 64 | 0.6890 | 48.2 |
| 128 | 0.7250 | 49.1 |
| **256** | **0.8127** | **52.5** |
| 512 | 0.8150 | 65.3 |

> ✅ **K=256 是最优平衡点**：
- 继续增加至 512 仅提升 0.3%，但延迟上升 24%；
- 下降到 128 导致 CLIP 下降 10.8%，表明低于一定阈值会损失关键结构信息。

---

## 4. 关键结论和发现

### 🎯 主要发现
1. **Learned Optical Compression 可实现“超越无损”效果**：
   - 在 UI-to-Code 中，大量高频噪声（如抗锯齿、渐变纹理）对生成无益；
   - 学习性压缩能主动过滤冗余信息，聚焦结构信号，从而提升生成质量。

2. **Decoder Adaptation 至关重要**：
   - 即使压缩得当，若不解码器进行适配，也无法发挥潜力；
   - LoRA 以极低成本（<0.1% 参数）解决了表示空间错配问题。

3. **非均匀信息密度需要结构感知压缩**：
   - UI 页面的信息分布极度不均；
   - element-guided reweighting 使得压缩过程更智能地保留语义关键区域。

4. **编码阶段压缩才是真正的效率突破**：
   - 推理时方法虽快，但无法改变 prefill 复杂度；
   - 只有在 encoder side 缩短序列，才能实现 **O(K²D)** 注意力成本的实质性下降。

---

### ⚠️ 局限性
1. **依赖额外训练数据**：需约 50K screenshot-HTML pair 进行训练，在小众领域可能难以获取；
2. **未完全复现 EfficientUICoder 全流程**：仅实现了其输入侧压缩模块，未包含输出去重（ADTS），可能低估其整体性能；
3. **评估指标偏宏观**：CLIP score 衡量整体视觉相似性，缺乏细粒度评估（如字符准确率、CSS 属性还原精度）；
4. **固定压缩数量 K=256**：未支持动态调整压缩比例以适应不同复杂度页面。

---

### 🔮 未来工作方向
- 开发 **adaptive token budget allocation** 机制，根据页面复杂度自动分配 token 数量；
- 构建面向移动端 App 或 Figma 设计稿的专用压缩数据集；
- 引入更精细的评估协议，如 **CSS property matching rate**、**DOM tree edit distance**；
- 探索将 optical compression 与其他高效推理技术（如 KV Cache 压缩）结合。

---

## ✅ 总结
UIPRESS 是首个成功将 **learned optical compression** 引入 UI-to-Code 任务的工作。它通过一个轻量级、任务感知的压缩模块，在仅增加 **0.26% 可训练参数** 的前提下，实现了：
- **25.5× token 压缩**
- **CLIP score +7.5% 提升**
- **9.1× TTFT 加速**

不仅显著提升了效率，还反向增强了生成质量，验证了“**压缩即增强**”的可能性，为后续高效 VLM 架构设计提供了新范式。

</details>

---

### 13. [MATCHA: Efficient Deployment of Deep Neural Networks on Multi-Accelerator Heterogeneous Edge SoCs](https://arxiv.org/abs/2604.09124)

**Authors**: Enrico Russo, Mohamed Amine Hamdi, Alessandro Ottaviano, Francesco Conti, Angelo Garofalo, Daniele Jahier Pagliari, Maurizio Palesi, Luca Benini, Alessio Burrello  
**Category**: cs.DC  
**Published**: 2026-04-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.09124v1  

#### Abstract
Deploying DNNs on System-on-Chips (SoC) with multiple heterogeneous acceleration engines is challenging, and the majority of deployment frameworks cannot fully exploit heterogeneity. We present MATCHA, a unified DNN deployment framework that generates highly concurrent schedules for parallel, hetero...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MATCHA: Efficient Deployment of Deep Neural Networks on Multi-Accelerator Heterogeneous Edge SoCs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代边缘系统级芯片（Edge SoC）通常集成多个异构加速器（如 SIMD 单元、向量处理器、GEMM 加速器等），以满足 DNN 推理在延迟、能效和成本上的严苛要求。然而，现有的 DNN 部署框架（如 TVM、MATCH）大多采用**层粒度映射**和**同步串行执行**策略，导致硬件资源利用率低下——同一时间只有一个加速器处于活跃状态，其余空闲。

### 提出的新方法与创新思路
本文提出 **MATCHA**，一个面向多加速器异构边缘 SoC 的统一 DNN 部署框架，其核心创新包括：

- ✅ **细粒度并行化支持**：  
  同时探索**层间并行**（inter-layer parallelism，利用 DNN 图中的独立分支）和**层内分块并行**（intra-layer tiling-based parallelism），将单个算子（如 Conv2D）划分为多个 tile，并分配到不同加速器上并发执行。

- ✅ **基于约束编程（Constraint Programming, CP）的联合优化引擎**：  
  将 pattern matching、设备分配、tile 分配和调度统一建模为一个 CP 问题，目标是最小化端到端推理延迟（makespan）。该模型考虑了设备性能差异、内存容量限制和数据依赖关系。

- ✅ **异步运行时设计**：  
  设计轻量级、无操作系统的异步运行时，支持跨设备的任务分发与完成通知（通过中断/事件机制），实现真正的并发执行，避免主机轮询开销。

- ✅ **统一的 tile-centric 映射范式**：  
  打破传统“一层一设备”的映射模式，引入 tile 变量 $ t_{p,m} $ 表示每个 pattern match 处理的 tile 数量，使映射决策更灵活，提升负载均衡能力。

### 相比现有方法的优势
| 特性 | MATCH / TVM | MATCHA |
|------|-------------|--------|
| 映射粒度 | 层或 fused pattern | **Tile-level + Layer-level** |
| 执行方式 | 同步、串行 | **异步、并发** |
| 并行机制 | 仅图级并行（多分支） | 支持图级 + **算子级分块并行** |
| 内存规划 | 分离处理 | 联合调度与内存分配（L2/L3） |
| 异构支持 | 是 | 更精细的异构适配（效率因子 $\eta_p$） |

MATCHA 是首个支持 **OS-less 异构 SoC 上并发异步执行** 的开源 DNN 编译器。

---

## 2. 核心实验方法和设置

### 实验平台
- **硬件平台**：开源的 **Carfield HSoC** 架构，部署于 Xilinx VCU118 FPGA，主频 50MHz。
- **架构组成**：
  - **Host (Device 0)**：双核 RV64GCH RISC-V CPU，共享 1MiB L2 SPM。
  - **Pulp Cluster (Device 1)**：8 个 RI5CY 核心，256KiB L1 SPM，支持 FP16。
  - **Spatz Cluster (Device 2)**：2 个 RISC-V 标量核心 + 2 个 RVVU 向量单元（VLEN=512），128KiB L1 SPM，支持 FP8–FP64 和 bfloat16。
- **通信机制**：AXI4 总线 + DMA 引擎 + PLIC 中断控制器，支持低延迟异步任务分发。

### 数据集与模型
- **基准测试套件**：
  - **MLPerf Tiny Benchmark**：AutoEncoder、DS-CNN、MobileNet、ResNet18
  - **微基准模块**（Microbenchmarks）：
    - ResNet-50 第一个残差块
    - ResNeXt-50 第一个 block
    - Transformer Encoder Layer（hidden size=128）

### 评估指标
- 端到端推理周期数（Cycles）
- 推理延迟（Runtime, ms）
- 实际达到的 FLOPS
- 加速器利用率（Utilization）
- 与基线相比的**延迟降低百分比**

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **TVM (Host-only)** | 所有计算在主机 CPU 上执行 |
| **MATCH** | 层粒度映射至最优加速器，**串行执行**（state-of-the-art 基线） |
| **MATCHA (No Tiling)** | 关闭 tile 分割，仅启用异步层卸载 |
| **MATCHA (Ours)** | 完整版本：启用 tile-centric 匹配 + 异步并发执行 |

所有方法使用相同的 pattern 库和 FP16 精度。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 和 Figure 7）

#### MLPerf Tiny 结果汇总（vs. MATCH）
| Model | MATCHA vs MATCH 延迟降低 |
|-------|--------------------------|
| **AutoEncoder** | **33.3%** ↓ |
| **DS-CNN** | 0% （无改善） |
| **MobileNet** | 0% （无改善） |
| **ResNet18** | **28.8%** ↓ |

> ⚠️ 注：DS-CNN 和 MobileNet 主要由 depthwise convolution 构成，其算术强度低，slice/concat 开销超过并行收益，故未受益。

#### 微基准结果（vs. MATCH）
| 模块 | 延迟降低 |
|------|---------|
| **ResNet Block** | **35.02%** ↓ |
| **ResNeXt-50 Block** | **17.55%** ↓ |
| **Transformer Encoder** | **23.65%** ↓ |

> ✅ 最高提速达 **35%**，显著优于基线。

#### 绝对性能增益（vs. TVM Host-only）
- 速度提升范围：**4.61× ~ 12.28×**
- 在 ResNet 块上最高达 **40.34× FLOPS 提升**

### 消融实验结果
- **异步执行 alone（No Tiling）**：
  - 在 ResNet-50 残差路径上带来约 **13–18%** 的延迟下降（得益于图级并行）。
- **启用 tile-centric 分块后**：
  - 进一步带来 **额外 15–20%** 的延迟优化，证明分块对负载均衡至关重要。
- **内存计划有效性**：
  - 图 4 展示了动态 swapping 与静态分配结合的调度方案，在有限 L2 容量下仍可实现高效并发。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Tile-centric 异步执行是提升异构 SoC 利用率的关键**：  
   仅靠 layer-wise 映射无法充分利用多加速器潜力；通过 operator tiling + 异步调度，可显著减少空闲时间。

2. ✅ **MATCHA 显著优于现有编译器**：  
   在真实异构平台上，MATCHA 相比 state-of-the-art 的 MATCH 编译器，**最高降低 35% 推理延迟**，平均提升 20–30%。

3. ✅ **并非所有网络都能从 tiling 中受益**：  
   对于 depthwise conv-heavy 的模型（如 DS-CNN），slice/concat 开销可能抵消并行优势，需智能选择是否分块。

4. ✅ **负载均衡能力增强**：  
   图 6 显示 ResNet 推理中，Pulp 和 Spatz 集群被均衡使用，且与主机协同工作，体现良好的 workload distribution。

### 方法的局限性
1. ❌ **helper operator 开销未精确建模**：  
   当前 latency model 忽略了 `slice` 和 `concat` 的运行时开销，影响优化准确性。

2. ❌ **分块维度选择依赖人工经验**：  
   当前默认按输出特征图行或输出神经元维度分块，缺乏自动维度搜索机制。

3. ❌ **两阶段优化分离**：  
   pattern matching 与底层 tiling（ZigZag）分开进行，未能在统一 cost model 中联合优化，存在次优风险。

4. ❌ **未评估能耗表现**：  
   虽然强调能效场景，但实验未报告能量数据，未来需补充功耗分析。

### 未来工作方向
1. 🔧 **统一优化框架**：  
   将高层 pattern matching 与底层 L1/L2 tiling 联合建模，实现端到端 joint optimization。

2. 🔧 **消除 slice/concat 开销**：  
   引入 view-based tiling 或连续地址布局规划，避免显式数据重组操作。

3. 🔧 **自动化分块策略搜索**：  
   基于算子几何特性与设备特性，自动决定最佳 tiling dimension 和粒度。

4. 🔧 **支持硬件闭环调优**：  
   引入 profiling-driven 参数校准（如 $\alpha_d$, $\eta_p$），提升 latency model 准确性。

5. 🔧 **扩展至更多异构组合**：  
   支持 GPU+NPU+FPGA 等复杂异构配置，验证通用性。

--- 

> 📌 **总结一句话**：  
> **MATCHA 通过 tile-centric pattern matching + 异步并发调度，在异构边缘 SoC 上实现了高达 35% 的推理延迟降低，是迈向高效边缘 AI 部署的重要一步。**

</details>

---

### 14. [The nextAI Solution to the NeurIPS 2023 LLM Efficiency Challenge](https://arxiv.org/abs/2604.09034)

**Authors**: Gyuwon Park, DongIl Shin, SolGil Oh, SangGi Ryu, Byung-Hak Kim  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.09034v1  

#### Abstract
The rapid evolution of Large Language Models (LLMs) has significantly impacted the field of natural language processing, but their growing complexity raises concerns about resource usage and transparency. Addressing these challenges, we participated in the NeurIPS LLM Efficiency Challenge, aiming to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《The nextAI Solution to the NeurIPS 2023 LLM Efficiency Challenge》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对 **NeurIPS 2023 LLM Efficiency Challenge** 中提出的现实挑战：在**单张 A100 40GB GPU** 上、**24小时内**完成对大规模语言模型（如 LLaMA2 70B）的高效微调。这一任务面临以下核心难题：
- **资源极度受限**：70B 参数模型通常需要多卡并行训练，难以部署于单卡；
- **透明性与可复现性缺失**：许多先进 LLM 缺乏开源细节，阻碍社区发展；
- **计算效率与性能之间的权衡**：如何在极低资源下保持高推理与问答能力。

### 提出的新方法与新思路
团队提出了一个**高度集成且资源友好的微调框架**，其核心创新在于：
- **QLoRA + Flash Attention 2 联合优化**：首次在挑战设定下成功将 QLoRA 与 Flash Attention 2 结合应用于 LLaMA2 70B 模型，在单卡上实现完整训练；
- **定制化高质量开源数据集构建策略**：基于 Open-Platypus 构建指令微调数据集，并通过 **Sentence-BERT 去重（cosine similarity > 0.9 过滤）** 提升数据质量；
- **精细化的 LoRA 配置探索**：系统性测试不同 LoRA r 值、目标模块组合（如 `attention + FFN output layers`），以最小化 GPU 占用而不牺牲性能。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **资源效率** | 仅用 **1×A100 40GB** 完成训练，总显存占用 **39.56GB**，远低于常规需求（通常需 8×A100 80GB）；训练时间仅 **17小时** |
| **参数效率** | 可训练参数比例低至 **0.0247%（约1700万参数）**，极大降低存储与更新开销 |
| **开放性与可复现性** | 全程使用开源数据与工具链（HuggingFace + Axolotl），符合挑战精神，易于复现 |
| **性能表现** | 在多个 QA 基准上达到高水平，尤其在 MMLU 和 TruthfulQA 上表现稳健 |

---

## 2. 核心实验方法和设置

### 使用的数据集
所有数据均为**开源且非由 LLM 自动生成**（遵循比赛规则），主要包括：

| 数据集 | 类型 | 数量 | 特点 |
|-------|------|------|------|
| **Open-Platypus** [17] | 综合指令数据 | 主体来源 | 高质量 STEM 与逻辑推理任务 |
| PRM800K | Math QA | 13k | 数学过程监督 |
| ScienceQA | Science QA | 1.3k | 多模态科学问答 |
| ReClor | Logical Reasoning QA | 4.5k | 需要逻辑推理的选择题 |
| TheoremQA | STEM QA | 0.5k | 定理驱动问答 |
| CodeGen | Coding Problem Solving | 4k | 补充被排除的编程数据 |
| Dolly | 多任务对话 | 15k | 包含创意写作、摘要等 |
| LIMA | Conversations | 1k | 高质量人类偏好数据 |
| FaithDial | Multi-turn Conversation | 4.5k | 信仰一致性对话 |
| Multi-News | Summarization | 4k | 多文档摘要 |

> ✅ 最终采用 **Version 5** 数据组合（见 Table 7）：  
> `Dolly(100%) + Filtering Platypus + CodeGen(10%) + FaithDial(20%) + Multi-News(10%) + LIMA(100%)`

### 实验设置
- **模型**：LLaMA2-70B（主）、LLaMA-30B/65B、LLaMA2-13B、Mistral-7B（对比）
- **硬件平台**：1×NVIDIA A100 40GB GPU
- **时间限制**：≤24 小时
- **微调技术**：
  - **QLoRA**（Quantized Low-Rank Adaptation）
  - **Flash Attention 2** 加速注意力计算
  - 使用 **Axolotl** 框架进行工程优化
- **量化配置**：
  - `bnb_4bit_quant_type = nf4`
  - `use_double_quant = True`
  - `compute_dtype = bfloat16`

### 评估指标
- **训练阶段指标**：
  - GPU 显存占用（GPU Usage）
  - 训练损失（Train Loss）
  - 总训练时间
- **基准测试（Benchmark）**：
  - **HELM** 测试框架下的子集：
    - **MMLU**（17/57 科目）
    - **BBQ**（Bias Benchmark for QA）
    - **TruthfulQA**
    - **CNN/DailyMail**（ROUGE 分数 + Bias 检测）
- **最终评分机制**：
  - **mean-win-rate** [34]：衡量模型在各项比较中胜出频率

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 9）
| 指标 | 数值 |
|------|------|
| 模型 | LLaMA2-70B |
| 数据集版本 | Version 5 |
| GPU 显存峰值 | **39.56 GB** |
| 可训练参数比例 | **0.0247%（~17M）** |
| 实际训练耗时 | **17 小时** |

> ⚠️ 成功在单卡 A100 上完成训练，未超限！

### Benchmark 测试结果（Table 8）
| 版本 | MMLU | BBQ | TruthfulQA | CNN/DailyMail (ROUGE-2) |
|------|------|-----|------------|--------------------------|
| Version 1 | 0.70 | 0.92 | 0.74 | 0.14 |
| Version 2 | 0.71 | 0.92 | 0.76 | 0.16 |
| Version 3 | 0.71 | 0.96 | 0.54 | 0.15 |
| Version 4 | 0.72 | 0.84 | 0.72 | 0.17 |
| **Version 5 (Ours)** | **0.73** | **0.90** | **0.76** | **0.17** |
| Version 6 | 0.69 | 0.96 | 0.70 | 0.15 |
| Version 7 | 0.70 | 0.84 | 0.64 | 0.14 |

✅ **Version 5 在 MMLU 和 TruthfulQA 上均取得最高分**，且整体表现最均衡。

### 与基线方法的对比
| 方法 | GPU Usage | Train Loss | Time |
|------|-----------|------------|------|
| QLoRA Only | 68.13 GB | 0.820 | 36m 12s |
| **QLoRA + Flash Attention 2** | **46.95 GB** | **0.824** | **19m 18s** |

➡️ 引入 Flash Attention 2 后：
- **GPU 内存下降 31%**
- **训练时间缩短近一半**
- 训练损失几乎不变（+0.004）

### 消融实验结果

#### （1）LoRA r 值的影响（Table 4）
| LoRA r | 4 | 8 | 16 | 64 |
|--------|----|----|----|-----|
| GPU Usage | **46.49** | 46.95 | 47.43 | 51.27 |
| Train Loss | 0.822 | 0.824 | 0.825 | 0.825 |

➡️ **r=4 即可获得最优资源-性能平衡**，更高秩带来边际收益递减。

#### （2）LoRA Target Layers 对比（Table 5）
| Layer Target | GPU Usage | Time | Train Loss |
|--------------|-----------|------|------------|
| all layers | 42.09 | 24m 4s | 0.794 |
| **attention + FFN output layers** | **41.46** | **22m 30s** | 0.798 |

➡️ **特定层适配优于全层微调**，节省资源且速度更快。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **大规模 LLM 可在极端资源约束下高效微调**：LLaMA2-70B 能在单张 A100 上完成训练，验证了 **QLoRA + Flash Attention 2** 的强大协同效应。
2. ✅ **数据质量比数量更重要**：通过去重与精选高质量数据（如 LIMA、Open-Platypus），即使小规模数据也能取得优异泛化能力。
3. ✅ **合理的 LoRA 配置能显著提升效率**：选择 `r=4` 并聚焦于关键层（attention + FFN output）可在不损失性能前提下大幅降低资源消耗。
4. ✅ **Flash Attention 2 是性能加速的关键**：几乎将训练时间减半，是达成 24 小时限的关键技术。

### 方法的局限性
- **秘密测试排名偏低**：在 secret evaluation 中排名第11（共17队），部分指标出现 `NULL` 值，表明模型在未知分布数据上的鲁棒性有待加强；
- **依赖高质量开源数据**：若缺乏类似 Open-Platypus 的整合数据集，复现难度上升；
- **未支持超长上下文**：当前设置最大序列长度为 1024，未利用 LongLoRA 等扩展方案处理更长输入。

### 未来工作方向
- 改进模型在跨领域、跨分布任务中的鲁棒性，避免“NULL”类异常输出；
- 探索 **LongLoRA** 或 **Shift Short Attention** 以支持更长上下文微调；
- 进一步压缩可训练参数规模，尝试 **IA³** 或 **Prefix-Tuning** 等更轻量级 PEFT 方法；
- 构建自动化数据筛选 pipeline，动态优化 dataset composition；
- 推动更多高质量、多样化、无偏见的开源指令数据发布，促进公平竞争环境。

--- 

> 📌 **总体评价**：该研究展示了在严格资源限制下高效微调超大规模语言模型的可行性路径，为边缘设备部署、中小企业应用及绿色 AI 发展提供了重要参考。其方法论具有高度实用性和推广价值。

</details>

---

### 15. [Event-Driven Temporal Graph Networks for Asynchronous Multi-Agent Cyber Defense in NetForge_RL](https://arxiv.org/abs/2604.09523)

**Authors**: Igor Jankowski  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.09523v1  

#### Abstract
The transition of Multi-Agent Reinforcement Learning (MARL) policies from simulated cyber wargames to operational Security Operations Centers (SOCs) is fundamentally bottlenecked by the Sim2Real gap. Legacy simulators abstract away network protocol physics, rely on synchronous ticks, and provide cle...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题**: *Event-Driven Temporal Graph Networks for Asynchronous Multi-Agent Cyber Defense in NetForge_RL*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Multi-Agent Reinforcement Learning (MARL)** 在从模拟环境迁移到真实 **Security Operations Centers (SOCs)** 时面临严重的 **Sim2Real Gap**，主要原因在于传统模拟器存在三大结构性缺陷：
1. **同步时间建模**：使用离散步长（synchronous ticks），无法反映真实网络攻击中动作具有随机持续时间（stochastic duration）的特性。
2. **理想化状态输入**：提供“干净”的 one-hot 编码状态向量，而现实 SOC 分析师依赖的是噪声大、非结构化的日志文本（如 Sysmon、Windows Event Logs）。
3. **静态拓扑结构**：固定网络节点数量和连接关系，缺乏对动态变化企业网络的泛化能力。

这些限制导致训练出的策略在真实环境中失效。

---

### 🚀 提出的新方法与创新思路

#### （1）**NetForge_RL**：首个面向真实场景的高保真连续时间 POSMDP 模拟器
- 将网络攻防建模为 **Partially Observable Semi-Markov Decision Process (POSMDP)**，支持异步事件驱动。
- 引入 **NLP-encoded telemetry**：将原始 Windows Event XML 日志通过 TF-IDF + LSA 或 Transformer 编码为 **128-dim LSA embeddings**，更贴近真实 SIEM 数据流。
- 实施 **Cryptographic Zero-Trust Network Access (ZTNA)** 约束，要求红队必须窃取身份令牌才能横向移动。
- 内置 **Green Agent** 模拟背景噪声（如用户登录、服务心跳），生成高达 20x 的 **false-positive alerts**，防止策略“奖励作弊”（reward hacking）。

#### （2）**CT-GMARL**：新型事件驱动图神经 MARL 架构
- 结合 **Graph Attention Network (GAT)** 和 **Neural ODE-RNN**，实现空间-时间联合建模：
  - **GAT** 处理动态拓扑结构，支持零样本扩展到不同规模网络；
  - **Neural ODE** 对不规则到达的日志进行连续时间积分，解决长时间“潜伏期”下隐藏状态退化问题。
- 提出 **Continuous-Time MAPPO** 优化框架，采用指数衰减折扣因子 $ \gamma(\Delta t) = e^{-\beta \Delta t} $ 进行 **Generalized Advantage Estimation (GAE)**，确保信用分配符合物理时间流逝。

#### （3）**Sim2Real 双引擎验证机制**
- **MockHypervisor**：高速模拟模式（~10,000 steps/sec），用于高效训练；
- **DockerHypervisor**：对接真实 Vulhub 容器漏洞（如 EternalBlue、BlueKeep），执行实际 exploit 脚本，实现 **zero-shot transfer evaluation**。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（CybORG/NASim） | 本文方法（NetForge_RL + CT-GMARL） |
|------|--------------------------|------------------------------------|
| 时间模型 | 同步离散 MDP/POMDP | 异步连续时间 POSMDP |
| 观测输入 | 清晰状态向量 | NLP 编码的日志嵌入（128-dim） |
| 拓扑结构 | 固定大小 | 动态可扩展图结构 |
| Sim2Real 支持 | 无 | 双引擎无缝切换，支持 zero-shot 验证 |
| 防御策略质量 | 易陷入“scorched earth” | 主动修复，兼顾安全与业务连续性 |

---

## 2. 核心实验方法和设置

### 📊 数据集与环境
- **自研模拟器 NetForge_RL**，构建于一个包含 **3个子网（DMZ、Corporate、Secure Vault）共100节点** 的虚拟网络。
- 攻击行为基于 **MITRE ATT&CK** 框架设计，涵盖远程利用（ExploitEternalBlue）、凭证转储（DumpLSASS）、票据传递（PassTheTicket）等战术。
- 日志来源：模拟生成的 **Windows Event XMLs**，经由 **TF-IDF + TruncatedSVD (LSA)** 投影为 **128维稠密向量**。
- 支持高级配置使用 **all-MiniLM-L6-v2 Transformer** 编码器。

### ⚙️ 实验设置
- **训练模式**：对抗性自博弈（adversarial self-play），红蓝双方均使用相同算法训练；
- **执行模式**：去中心化决策（decentralized execution），每个 Blue agent 仅能访问其所在区域的日志；
- **硬件平台**：NVIDIA A100 GPU 集群，使用 PyTorch + torchdiffeq + CleanRL 实现；
- **训练时长**：约 8 小时完成 2.5M 步训练（10 个随机种子平均）；
- **评估方式**：最后 20% 训练阶段取中位数指标。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Blue Reward** | 防守方综合得分，平衡响应速度、误操作惩罚与业务可用性 |
| **Services Restored** | 成功恢复的服务数量（衡量主动修复能力） |
| **Total Successful Exploits** | 攻击成功次数（需结合“服务恢复”判断是否为“焦土防御”） |
| **ODE NFE** | Neural ODE 积分函数调用次数（计算开销） |
| **KL Divergence** | 策略更新稳定性 |
| **Steps Per Second (SPS)** | 环境吞吐量（效率） |

### 🆚 基线方法对比
- **R-MAPPO**：标准 MARL 算法，使用 LSTM 处理序列，MLP 输入；
- **QMIX**：值分解型 MARL 方法，适用于合作任务；
- 所有基线均适配环境：将图结构展平并补零至 100 节点，且显式拼接 $\Delta t$ 到观测向量中。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（收敛后中位数）

| 方法 | Blue Reward | Services Restored | Exploits | SPS |
|------|-------------|--------------------|----------|-----|
| **CT-GMARL (Ours)** | **57,135** | **144** | 1.0 | 10.0 |
| R-MAPPO | 28,347 | 13 | 2 | 17.2 |
| QMIX | 26,649 | 5 | 0 | 11.7 |
| **Sim2Real (Docker)** | **98,026** | — | — | 5.1 |

> ✅ **CT-GMARL 达成 2.0× 蓝方奖励提升**（vs R-MAPPO），**服务恢复数达 12× 更多**

---

### 🔬 消融实验结果（Ablation Studies）

| 消融变体 | Blue Reward | Services Restored | 说明 |
|---------|-------------|--------------------|------|
| **No-ODE** | 51,783 | 32 | 移除 Neural ODE 导致时间感知断裂，服务恢复能力大幅下降 |
| **No-GAT** | 56,565 | 73 | 保留部分性能但空间精度丧失，行动分布趋于均匀 |
| **No-Beta ($\gamma$ fixed)** | 56,656 | 48 | 固定折扣导致信用分配失真，影响长期策略学习 |
| **Full CT-GMARL** | **57,135** | **144** | 完整架构表现最优，尤其在主动防御上显著领先 |

> 💡 发现：即使移除 GAT，仍可通过 TopologyMessagePasser 维持一定性能；但 **ODE 模块对“服务恢复”至关重要**，证明连续时间建模是关键。

---

### 🧪 Zero-Shot Sim2Real Transfer 结果
- 在 **未经过任何微调或梯度更新** 的情况下，将在 MockHypervisor 中训练好的 CT-GMARL 策略直接部署到 **DockerHypervisor** 上运行：
  - **中位奖励达到 98,026**，比模拟训练高出 **71%**
  - 表明 NetForge_RL 的双引擎设计有效弥合了 Sim2Real Gap
- 原因分析：Docker 环境中的真实 CVE 执行路径更具确定性和结构性，提供了更强的学习信号。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **异步连续时间建模优于同步离散假设**：真实网络攻防本质上是非周期、突发性的，Neural ODE 能有效维持隐藏状态的时间一致性。
2. **GAT 提升空间推理能力**：使策略学会忽略 DMZ 区域的噪声干扰，聚焦于内部 ZTNA 拐点进行精准布防。
3. **避免“scorched earth”失败模式**：传统方法通过隔离所有主机来压低攻击成功率，牺牲业务连续性；而 CT-GMARL 实现了 **主动修复 + 控制暴露面** 的平衡策略。
4. **Sim2Real 桥梁可行**：通过物理约束 + NLP telemetry + live exploit injection，实现了无需微调即可迁移的高鲁棒策略。

---

### ⚠️ 局限性
1. **Neural ODE 计算开销大**：尽管使用 RK4 固定步长控制 NFE=4，但仍比 LSTM 消耗更多 FLOPs，影响实时性；
2. **Fixed-step solver 存在语义模糊风险**：当事件风暴密集发生时，非自适应求解器可能丢失细粒度时间信息；
3. **Red Agent 动作受限于预定义 CVE 集合**：缺乏对 zero-day 或组合式新型攻击的泛化能力；
4. **Live Evaluation 吞吐极低**：DockerHypervisor 仅 ~5 SPS，难以支撑大规模训练。

---

### 🔮 未来工作方向
1. 扩展 Red Agent 动作空间以支持 **dynamic zero-day exploit generation**；
2. 替换 TF-IDF pipeline 为轻量化 **Transformer-based telemetry encoder**（如 MiniLM）；
3. 将模型扩展至 **千级节点规模**，验证企业级网络适用性；
4. 探索 **offline RL + human feedback** 混合范式，增强策略可解释性与安全性。

---

> 📦 **代码与数据开源地址**：
> - NetForge_RL: [https://github.com/xaiqo/NetForge_RL](https://github.com/xaiqo/NetForge_RL)
> - CT-GMARL: [https://github.com/xaiqo/ct-gmarl](https://github.com/xaiqo/ct-gmarl)  
> （MIT License，含完整复现实验脚本与预训练权重）

</details>

---

### 16. [TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](https://arxiv.org/abs/2604.08948)

**Authors**: Gang Hu, Yating Chen, Haiyan Ding, Wang Gao, Jiajia Huang, Min Peng, Qianqian Xie, Kun Yu  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.08948v1  

#### Abstract
While Large Language Models (LLMs) excel in various general domains, they exhibit notable gaps in the highly specialized, knowledge-intensive, and legally regulated Chinese tax domain. Consequently, while tax-related benchmarks are gaining attention, many focus on isolated NLP tasks, neglecting real...

---

### 17. [Wireless Communication Enhanced Value Decomposition for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2604.08728)

**Authors**: Diyi Hu, Bhaskar Krishnamachari  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.08728v1  

#### Abstract
Cooperation in multi-agent reinforcement learning (MARL) benefits from inter-agent communication, yet most approaches assume idealized channels and existing value decomposition methods ignore who successfully shared information with whom. We propose CLOVER, a cooperative MARL framework whose central...

---

### 18. [Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](https://arxiv.org/abs/2604.08944)

**Authors**: Benjamin Amoh, Geoffrey Parker, Wesley Marrero  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.08944v1  

#### Abstract
Multi-agent coordination under partial observability requires agents to share complementary private information. While recent methods optimize messages for intermediate objectives (e.g., reconstruction accuracy or mutual information), rather than decision quality, we introduce \textbf{SeqComm-DFL}, ...

---

### 19. [DiffHLS: Differential Learning for High-Level Synthesis QoR Prediction with GNNs and LLM Code Embeddings](https://arxiv.org/abs/2604.09240)

**Authors**: Zedong Peng, Zeju Li, Qiang Xu, Jieru Zhao  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.09240v1  

#### Abstract
High-Level Synthesis (HLS) compiles C/C++ into RTL, but exploring pragma-driven optimization choices remains expensive because each design point requires time-consuming synthesis. We propose \textbf{\DiffHLS}, a differential learning framework for HLS Quality-of-Result (QoR) prediction that learns f...

---

### 20. [Distributed Online Convex Optimization with Compressed Communication: Optimal Regret and Applications](https://arxiv.org/abs/2604.09276)

**Authors**: Sifan Yang, Dan-Yue Li, Lijun Zhang  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.09276v1  

#### Abstract
Distributed online convex optimization (D-OCO) is a powerful paradigm for modeling distributed scenarios with streaming data. However, the communication cost between local learners and the central server is substantial in large-scale applications. To alleviate this bottleneck, we initiate the study ...

---

### 21. [Stochastic-Dimension Frozen Sampled Neural Network for High-Dimensional Gross-Pitaevskii Equations on Unbounded Domains](https://arxiv.org/abs/2604.09361)

**Authors**: Zhangyong Liang  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.09361v1  

#### Abstract
In this paper, we propose a stochastic-dimension frozen sampled neural network (SD-FSNN) for solving a class of high-dimensional Gross-Pitaevskii equations (GPEs) on unbounded domains. SD-FSNN is unbiased across all dimensions, and its computational cost is independent of the dimension, avoiding the...

---

### 22. [Quantisation Reshapes the Metacognitive Geometry of Language Models](https://arxiv.org/abs/2604.08976)

**Authors**: Jon-Paul Cacioli  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.08976v1  

#### Abstract
We report that model quantisation restructures domain-level metacognitive efficiency in LLMs rather than degrading it uniformly. Evaluating Llama-3-8B-Instruct on the same 3,000 questions at Q5_K_M and f16 precision, we find that M-ratio profiles across four knowledge domains are uncorrelated betwee...

---

### 23. [Adaptive Simulation Experiment for LLM Policy Optimization](https://arxiv.org/abs/2604.08779)

**Authors**: Mingjie Hu, Siyang Gao, Jian-qiang Hu, Enlu Zhou  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.08779v1  

#### Abstract
Large language models (LLMs) have significant potential to improve operational efficiency in operations management. Deploying these models requires specifying a policy that governs response quality, shapes user experience, and influences operational value. In this research, we treat LLMs as stochast...

---

### 24. [Finite-Sample Analysis of Nonlinear Independent Component Analysis:Sample Complexity and Identifiability Bounds](https://arxiv.org/abs/2604.08850)

**Authors**: Yuwen Jiang  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.08850v1  

#### Abstract
Independent Component Analysis (ICA) is a fundamental unsupervised learning technique foruncovering latent structure in data by separating mixed signals into their independent sources. While substantial progress has been made in establishing asymptotic identifiability guarantees for nonlinear ICA, t...

---

### 25. [Synthesizing real-world distributions from high-dimensional Gaussian Noise with Fully Connected Neural Network](https://arxiv.org/abs/2604.09091)

**Authors**: Joanna Komorniczak  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.09091v1  

#### Abstract
The use of synthetic data in machine learning applications and research offers many benefits, including performance improvements through data augmentation, privacy preservation of original samples, and reliable method assessment with fully synthetic data. This work proposes a time-efficient syntheti...

---

### 26. [Truncated Rectified Flow Policy for Reinforcement Learning with One-Step Sampling](https://arxiv.org/abs/2604.09159)

**Authors**: Xubin Zhou, Yipeng Yang, Zhan Li  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.09159v1  

#### Abstract
Maximum entropy reinforcement learning (MaxEnt RL) has become a standard framework for sequential decision making, yet its standard Gaussian policy parameterization is inherently unimodal, limiting its ability to model complex multimodal action distributions. This limitation has motivated increasing...

---

### 27. [StaRPO: Stability-Augmented Reinforcement Policy Optimization](https://arxiv.org/abs/2604.08905)

**Authors**: Jinghan Zhang, Fengran Mo, Tharindu Cyril Weerasooriya, Ruimin Dai, Xiaoyan Han, Yanjie Fu, Dakuo Wang, Kunpeng Liu  
**Category**: cs.AI  
**Published**: 2026-04-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.08905v1  

#### Abstract
Reinforcement learning (RL) is effective in enhancing the accuracy of large language models in complex reasoning tasks. Existing RL policy optimization frameworks rely on final-answer correctness as feedback signals and rarely capture the internal logical structure of the reasoning process. Conseque...

---

### 28. [Hierarchical Alignment: Enforcing Hierarchical Instruction-Following in LLMs through Logical Consistency](https://arxiv.org/abs/2604.09075)

**Authors**: Shu Yang, Zihao Zhou, Di Wang, Wenda Li  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.09075v1  

#### Abstract
Large language models increasingly operate under multiple instructions from heterogeneous sources with different authority levels, including system policies, user requests, tool outputs, and retrieved context. While prior work on instruction hierarchy highlights the importance of respecting instruct...

---

### 29. [EthicMind: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](https://arxiv.org/abs/2604.09265)

**Authors**: Jiawen Deng, Wei Li, Wentao Zhang, Ziyun Jiao, Fuji Ren  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.09265v1  

#### Abstract
Intelligent dialogue systems are increasingly deployed in emotionally and ethically sensitive settings, where failures in either emotional attunement or ethical judgment can cause significant harm. Existing dialogue models typically address empathy and ethical safety in isolation, and often fail to ...

---

### 30. [Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](https://arxiv.org/abs/2604.09377)

**Authors**: Hui Liu, Bin Zou, Kecheng Chen, Jie Liu, Wenya Wang, Haoliang Li  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.09377v1  

#### Abstract
Large language models (LLMs) exhibit substantial variability in performance and computational cost across tasks and queries, motivating routing systems that select models to meet user-specific cost-performance trade-offs. However, existing routers generalize poorly in cold-start scenarios where in-d...

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
