# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-28 09:07:48 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training](https://arxiv.org/abs/2605.27740)

**Authors**: Keqi Deng, Shaoshi Ling, Ruchao Fan, Jinyu Li  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2605.27740v1  

#### Abstract
Long-context inference in large language models (LLMs) is bottlenecked by the linear growth of the self-attention key-value (KV) cache. Top-k sparse attention alleviates this by loading only a small fraction of the KV cache, but accurately and cheaply estimating cache importance, for both training-f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：UNIQUE: Universal Top-k Sparse Attention for Training-free Inference and Sparsity-aware Training

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在长上下文推理中面临 **Key-Value (KV) cache** 的内存瓶颈。随着上下文长度增长到数十万甚至上百万 token，KV cache 的线性增长导致自回归解码时的内存带宽压力剧增，显著拖慢推理速度。现有的稀疏注意力方法存在以下两类问题：
- **Training-free 方法**（如 Quest、InfLLM）：无需训练即可部署，但在高稀疏度下性能下降明显，存在训练-推理不一致（train-inference gap）。
- **Trainable 方法**（如 NSA、DSA）：通过引入辅助损失或专用架构实现更好的稀疏适应性，但增加了训练复杂性和资源开销。

此外，大多数稀疏注意力研究集中在文本模态，对语音等连续信号的泛化能力未知。

### 提出了什么新方法或新思路
本文提出 **UNIQUE**，一个统一的、通用的 top-k 稀疏注意力框架，支持：
- **Training-free 推理**：无需任何模型微调即可高效运行。
- **Sparsity-aware 训练**：可通过微调进一步优化模型对稀疏注意力的适应性，且无需额外参数或辅助损失。

其核心创新包括：

#### （1）Offset-Augmented Page Scoring（偏移增强页面评分）
在 KV page 粒度上估计重要性，每个 page 的得分由两部分组成：
$$
\text{score}(p) = q \cdot \text{mean}_p + \lambda \|q\|_2 \cdot \text{std}_p
$$
- `mean_p`：page 内所有 key 的均值向量，代表平均语义。
- `std_p`：key 向量的标准差（标量化），作为“偏移项”补偿均值可能稀释高信息量 token 的问题。
- 该设计简单、计算廉价，却能更准确地捕捉 page 中潜在的关键信息。

#### （2）Soft-Mask Sparsity-aware Training（软掩码稀疏感知训练）
为缩小训练-推理差距，提出一种可微的 soft mask：
- 以 top-k 分数边界为中心，应用 sigmoid 函数生成 soft mask。
- 将其作为 additive log-bias 加入 attention softmax 中，使梯度可反向传播至 page score。
- **无需辅助损失、无需额外模块、无架构改动**，即可让模型主动学习更好的 top-k 选择策略。

#### （3）高效的 CUDA 实现
- **融合 criticality estimation kernel**：将 matmul、offset 添加与 max reduction 融合为单个 kernel，减少 HBM 访问。
- **基于 radix 的 top-k 选择**：采用两轮 8-bit radix selection，时间复杂度 $O(P)$，远快于传统 $O(P \log P)$ 方法。
- 支持与 paged attention 无缝集成。

### 相比现有方法的优势
| 维度 | UNIQUE | 现有方法 |
|------|--------|---------|
| **通用性** | ✅ 支持 text & speech 多模态 | 多数仅验证于 text |
| **灵活性** | ✅ 同时支持 training-free 和 fine-tuning | 通常只能二选一 |
| **简洁性** | ✅ 无额外参数/损失/架构修改 | 需要路由网络、蒸馏损失等 |
| **效率** | ✅ CUDA 层面高度优化 | 多数依赖标准库 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
#### 文本任务
- **LongBench-Pro**：双语（英/中）、11类主任务 + 25类子任务，输入最长达 256K tokens。
- **RULER-32K**：涵盖检索、多跳追踪、聚合问答等，测试真实长上下文理解能力。
- **LongAlign-10k**：用于 sparsity-aware 微调的英文长上下文对齐数据集。

#### 语音任务
- **葡萄牙语 ASR 数据集**：
  - 训练：69K 小时语音（最长 1 小时）。
  - 测试：约 10 分钟长语音片段，分巴西葡语和欧洲葡语两个子集。

### 实验设置和评估指标
| 设置项 | 描述 |
|-------|------|
| **模型** | 文本：Ministral-3-8B-Instruct-2512；语音：Qwen3-8B + Conformer 编码器 |
| **KV Page Size** | $S = 8$ |
| **KV Budget** | 默认 512 tokens（即最多访问 64 个 pages） |
| **Batch Size** | 80（端到端测试） |
| **硬件** | NVIDIA H100 GPU |
| **评估指标** | 
| - 文本：Accuracy (%) on LongBench-Pro / RULER |
| - 语音：Word Error Rate (WER)，Entity Error Rate (EER) ↓ |
| - 效率：Attention kernel latency, End-to-end decoding latency |

### 基线方法对比
#### Training-free 方法
- **Quest**：基于 key 的 min-max 统计选择 top-k pages。
- **H2O**：基于 attention score 的 token 级淘汰机制。
- **InfLLM**：扩展 Quest 至超长上下文，支持 CPU offload。

#### Trainable 方法
- **InfLLM-v2**：InfLLM 的可训练版本，支持从 dense checkpoint 微调。
- **DSA (DeepSeek Sparse Attention)**：基于蒸馏的稀疏注意力，原为 MLA 设计，本文适配至 GQA。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Training-free 文本结果（LongBench-Pro @ 512 budget）
| 方法 | Overall Accuracy |
|------|------------------|
| Full Attention | 37.70 |
| InfLLM | 34.99 |
| **UNIQUE (Ours)** | **36.58** |

→ **恢复 97.0% 的全注意力性能**，优于最佳 baseline 1.59 pts。

#### ✅ Training-free 语音结果（ASR @ 512 budget）
| 方法 | Avg EER/WER |
|------|-------------|
| Full Attention | 18.25 |
| Quest | 21.92 |
| InfLLM | 57.72 |
| **UNIQUE (Ours)** | **18.60** |

→ **几乎匹配全注意力表现**（仅差 0.35 pts），是唯一在语音上有效的稀疏方法。

#### ✅ Sparsity-aware Fine-tuning 结果
| 方法 | LongBench-Pro (EN) | ASR Macro-Avg |
|------|--------------------|---------------|
| Dense Fine-tuned | 36.55 | 18.25 |
| InfLLM-v2 | 36.11 | 19.10 |
| DSA | 36.79 | 20.91 |
| **UNIQUE (Ours)** | **37.25** | **17.89** |

→ 在两项任务上均 **超越全注意力 baseline**，表明稀疏训练反而有助于去除噪声。

### 与基线方法的对比结果
- 在 **LongBench-Pro 和 RULER 上同时优于所有 baseline**，而其他方法各有短板：
  - Quest 在 RULER 表现好但在 LongBench-Pro 下降剧烈。
  - H2O 因永久删除 token 导致 MK3 等任务失败（得分为 0）。
- 在 **语音 ASR 上大幅领先**，InfLLM 性能崩溃，Quest 也有明显退化，唯独 UNIQUE 稳定。
- 在 **低预算（如 128 pages）下优势更显著**，其他方法性能断崖式下跌。

### 消融实验结果

#### 🔹 移除 std 偏移项的影响（Table 6）
| 变体 | ASR Macro-Avg |
|------|---------------|
| UNIQUE w/o std | 19.11 |
| **UNIQUE (完整)** | **18.60** |

→ 证明 `std` 项有效捕获 page 内部方差，提升关键信息识别能力。

#### 🔹 Soft Mask vs Hard Mask（Table 7）
| 变体 | ASR Macro-Avg |
|------|---------------|
| Hard Mask | 18.32 |
| **Soft Mask (Ours)** | **17.89** |

→ 软掩码因可微性，允许 criticality estimator 自适应优化，带来持续增益。

#### 🔹 效率提升（Figure 2 & 3）
| 指标 | 提升倍数 |
|------|----------|
| **Attention Kernel Speedup** (@32K context) | **11.4×** vs FlashInfer dense |
| **End-to-end Decoding Speedup** (@32K context) | **5.3×** vs avLLM-based dense model |

→ 显著降低推理延迟，尤其在长上下文场景下优势巨大。

---

## 4. 关键结论和发现

### 主要发现
1. **UNIQUE 是首个真正通用的稀疏注意力框架**，在 text 和 speech 模态上均能保持高性能，验证了其 **modality-agnostic** 特性。
2. **简单的统计量（mean + std）足以实现高质量的 page 重要性估计**，无需复杂路由或蒸馏机制。
3. **soft-mask 训练策略有效闭合了 train-inference gap**，且实现极其简洁，适合工业部署。
4. **稀疏注意力不仅加速推理，还可能提升最终性能** —— 在 fine-tuning 后，UNIQUE 超越了全注意力 baseline，说明 full-context 并非总是最优。

### 方法的局限性
1. **未与其他 KV 压缩技术（如量化、低秩压缩）进行联合比较**，这些方向正交但可结合。
2. 当前评估集中于 text 和 speech，尚未扩展到 vision 或 vision-language 模型。
3. 未在从头预训练（pre-training from scratch）场景下验证，因计算成本过高；目前仅覆盖 fine-tuning 场景。
4. radix top-k kernel 在超过 ~23K pages 后需 fallback 到 FlashInfer，仍有优化空间。

### 未来工作方向
- 将 UNIQUE 扩展至多模态大模型（MLLMs），尤其是视觉长序列建模。
- 探索与 KV quantization、low-rank approximation 等技术的协同优化。
- 研究在 full pre-training setting 下的应用潜力。
- 进一步优化 extreme-long context（>1M tokens）下的 offload 与调度策略。

--- 

> 💡 **总结一句话**：  
> **UNIQUE 以极简设计实现了训练即用（training-free）与可微调优（sparsity-aware）的统一，在文本与语音长上下文任务中均达到接近甚至超越全注意力的性能，同时带来高达 11.4× 的注意力内核加速，为高效、通用的稀疏注意力提供了新范式。**

</details>

---

### 2. [HRBench: Benchmarking and Understanding Thinking-Mode Switch Strategies in Hybrid-Reasoning LLMs](https://arxiv.org/abs/2605.28398)

**Authors**: Yansong Ning, Mianpeng Liu, Jingwen Ye, Weidong Zhang, Hao Liu  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.28398v1  

#### Abstract
Hybrid-reasoning large language models (LLMs) expose explicit controls over reasoning effort, allowing users or systems to trade off answer quality against inference cost. However, existing methods for adaptive thinking-mode selection are typically evaluated under different models, datasets, and imp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HRBench: Benchmarking and Understanding Thinking-Mode Switch Strategies in Hybrid-Reasoning LLMs 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前针对 **Hybrid-Reasoning LLMs**（混合推理大模型）中的 **thinking-mode switch**（思维模式切换）策略的研究存在严重碎片化问题：
- 不同方法在不同的 **LLM 模型、数据集、训练方式、评估指标** 下进行比较；
- 缺乏统一标准，导致无法公平判断哪种策略更优。

这使得研究者难以回答两个核心问题：
1. 哪种 switching strategy 最有效？
2. 训练过程如何影响不同策略的表现？

### 🚀 提出的新方法与创新
作者提出了 **HRBench** —— 首个系统性的、统一的评估框架，用于全面理解 hybrid-reasoning LLM 中的思维模式切换机制。

#### 主要创新点：
- **构建了正交的评估分类体系（taxonomy）**：
  - 3 种 **switching strategy families**：Prompt-Tuning (PT), Routing (RT), Speculative (SPEC)
  - 4 种 **training regimes**：Training-Free, SFT, Offline RL, Online RL
  - 组合形成 **12 种受控实验配置**，实现跨策略、跨训练方式的公平比较。

- **开源平台与可复现性**：
  - 提供完整的代码库、基准实现和统一 pipeline（基于 `verl` 和 `vLLM`）
  - 支持即插即用（plug-and-play），促进社区后续研究
  - 所有外部方法均被重新实现并纳入统一评估流程

- **首次揭示策略间的根本差异与交互效应**：
  - 发现不同策略在 **efficiency-effectiveness trade-off** 上具有本质区别
  - 揭示训练收益是 **strategy-dependent** 的
  - 明确指出最优策略选择依赖于 **模型规模（scale）** 和 **任务领域（domain）**

### 🔍 相比现有方法的优势
| 方面 | 现有工作 | HRBench |
|------|--------|---------|
| 可比性 | 各自为政，不可比 | 统一 pipeline，公平对比 |
| 覆盖范围 | 单一策略或少量方法 | 覆盖 3 类策略 + 4 种训练范式 |
| 分析深度 | 局部性能报告 | 系统性消融 + 多维分析（scale/domain） |
| 开放性 | 多数未开源或难复现 | 完整开源，支持扩展 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
共使用 **5 个 benchmark 数据集**，覆盖三大推理密集型任务：

| 数据集 | 领域 | 规模 | 难度 |
|-------|-----|------|------|
| **MATH500** | 数学 | 500题 | 高中水平 |
| **AIME 2025** | 数学 | 30题 | 竞赛级 |
| **GPQA-Diamond** | 科学 | 198题 | 研究生水平（物理/化学/生物） |
| **LiveCodeBench (LCB)** | 编程 | 167题 | 实际编程任务（执行验证） |
| **Codeforces** | 编程 | 366题 | 竞赛级编程 |

> 所有任务均采用自动化评估（Pass@1）或 LLM-as-judge 进行评分。

### ⚙️ 实验设置
- **模型范围**：涵盖从 **2B 到 1.1T 参数** 的 6 个主流 hybrid-reasoning LLM：
  - Qwen3.5-2B, Qwen3.5-9B
  - gpt-oss-20B
  - Seed-OSS-36B
  - DeepSeek-V3.1-671B
  - Kimi-K2.5-1.1T

- **思维模式控制形式多样**：
  - Binary switch: `think` / `no_think`
  - Discrete effort: `High/Medium/Low`
  - Numeric budget: `≤ 4096 tokens`

- **评估指标**
  - **Acc**: Pass@1 准确率（%）
  - **Tok**: 平均输出 token 成本（含 CoT）

- **统一解码参数**：
  - 温度 = 0（greedy decoding），最大输出长度 = 32,768 tokens
  - 所有方法在同一 pipeline 下运行以确保公平性

### 🧪 基线方法对比
#### 固定基线（Fixed Baselines）：
- **Full-Think**：始终启用深度推理
- **No-Think**：始终直接作答
- **Budget-Aware**：手动设定推理等级（如 High/Medium/Low）

#### 内部实现（Our Implementations）：
对每种 (strategy × training regime) 组合提供参考实现：
- **Training-Free**：PT-TF, RT-TF, Spec-TF
- **SFT / DPO / GRPO**：分别训练对应策略

#### 外部集成方法（12+ representative prior works）：
| 类别 | 方法 |
|------|------|
| Prompt-Tuning | S1, TALE, Budget-Guidance, SoT, CoD, DynaThink, DEER, RASC |
| Routing | AdaptThink, HDFlow |
| Speculative | MixReasoning, ADR |

> 所有外部方法均在 HRBench 统一 pipeline 中重现实验。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（以 Qwen3.5-9B 为例）

| 方法 | Avg Acc (%) | Avg Tok | Token Reduction vs Full-Think |
|------|-------------|---------|-------------------------------|
| Full-Think | 40.9 | 16.4k | — |
| No-Think | 42.4 | 5.5k | +66.5% |
| **PT-TF** | **47.6** | **12.4k** | **+24.4%** |
| RT-TF | 44.1 | 14.3k | +12.5% |
| Spec-TF (Entropy) | 45.8 | 21.2k | -29.6% |

> ✅ **PT-TF 实现“双赢”**：准确率 ↑ + token 成本 ↓  
> ⚠️ **SPEC 增加开销但提升准确率**

### 🔁 与基线方法的对比结果
#### 在统一评估下的表现排序（图5）：
- **Prompt-Tuning 方法整体 Pareto 最优**：
  - 如 **RASC** 达到 53.4% 准确率（最高），但 token 消耗极高
  - **Chain-of-Draft (CoD)** 实现最强压缩（token ↓79.9%），适合高效率场景
- **Routing 方法稳定节省 token（约 18–21%）且保持准确性**
- **Speculative 方法普遍增加 token 消耗（-11% ~ -30%）**，仅 **ADR** 实现轻微节省（+15.2%）

#### 按任务领域的最佳方法不同（表6）：
| 领域 | 最佳方法 | 性能亮点 |
|------|----------|-----------|
| 数学（AIME） | RASC | 83.3% 准确率 |
| 数学（MATH500） | Budget-Guidance | 87.8% 准确率 |
| 科学（GPQA） | Sketch-of-Thought | 表现领先 |
| 编程 | 多样分布 | 无单一主导 |

> ❗ 结论：不应依赖 aggregate score 选方法，应结合目标 domain 选择

### 🔍 消融实验结果（Training Effects on Qwen3.5-9B）

| 训练方式 | Acc 提升（vs TF） | Token Reduction（vs TF） |
|----------|------------------|--------------------------|
| SFT | +0.2~+0.4 pp | 12–33% |
| DPO | +0.5~+0.9 pp | 15–31% |
| **GRPO** | +0.3~+0.8 pp | **高达 65%**（尤其对 RT） |

> 💡 **关键发现**：
> - 所有训练方式都能显著降低 token 消耗（效率增益远大于准确率增益）
> - **GRPO 对 Routing 效果最明显**（token ↓65%），因其 binary decision 更易通过强化学习优化
> - **DPO 更利于 accuracy 提升**，GRPO 更利于 efficiency 优化

---

## 4. 关键结论和发现

### 🎯 主要发现（Key Findings）

#### ✅ **不同策略具有本质不同的 trade-off 特征**
| 策略 | 特点 | 适用场景 |
|------|------|----------|
| **Prompt-Tuning (PT)** | “Win-Win”：提高准确率同时减少 token 使用 | 通用首选，尤其数学任务 |
| **Routing (RT)** | 稳定节省 token（~12–20%），保持 accuracy | 强调成本控制的部署场景 |
| **Speculative (SPEC)** | 提升 accuracy，但增加 token 开销 | 对精度敏感、资源充足的场景 |

#### ✅ **训练效果高度依赖于策略类型**
- **RT 受训练增益最大**：GRPO 可带来高达 **65% token 节省**
- **PT 和 SPEC 收益较小**：因涉及细粒度决策（如触发时机），难以通过稀疏奖励优化

#### ✅ **最优策略随模型规模和任务领域变化**
| 维度 | 影响 |
|------|------|
| **模型规模** | 
| - 小模型（2B–9B）：PT 最优 | 
| - 中大型模型（20B, 671B）：SPEC 反超 PT |
| - 超大规模（1.1T）：PT 再次领先 |
| **任务领域** |
| - 数学/科学：PT 表现最好 |
| - 编程任务：SPEC 因“try-then-verify”机制表现优异 |

> 📌 **没有一种策略通吃所有情况！必须根据模型 size 和 target domain 动态选择。**

---

### ⚠️ 方法的局限性（Limitations）

1. **训练实验受限于计算资源**：
   - 所有训练实验仅在 **Qwen3.5-9B** 上完成
   - 更大模型（如 20B+）上的训练行为仍需探索

2. **仅考虑单轮推理（single-turn）**：
   - 未覆盖多步推理或多智能体协作等复杂场景
   - 在 agent workflow 中的累积决策效应尚未研究

3. **任务领域有限**：
   - 当前聚焦于数学、科学、编程
   - 创意写作、多语言、常识推理等领域可能呈现不同规律

4. **外部方法复现可能存在偏差**：
   - 尽管尽力还原原论文设置，但超参、数据划分等细节可能导致微小差异

---

### 🔮 未来工作方向（Future Work）

1. **扩展至更大模型与更多训练范式**：
   - 探索 20B+ 模型上的训练 scalability
   - 引入 ICL-based 或 zero-shot routing 方法

2. **支持 multi-turn 与 agentic 场景**：
   - 构建动态 workflow 中的 mode-switching benchmark
   - 研究长期推理路径中的 cost 控制

3. **引入更多评估维度**：
   - latency、energy consumption、carbon footprint
   - human evaluation for non-structured tasks

4. **开发自动推荐系统**：
   - 基于模型 size + task type 自动推荐最优 switching strategy + training regime

5. **拓展至非英语和其他模态**：
   - 多语言 reasoning benchmark
   - 视觉-语言 hybrid reasoning（如 Kimi-K2.5 支持视觉输入）

---

## ✅ 总结

**HRBench 是首个系统性、标准化的 hybrid-reasoning LLM 思维切换评估框架**，其核心价值在于：

- ✅ **统一了混乱的研究生态**，实现了跨策略、跨训练方式的公平比较；
- ✅ **揭示了深层规律**：策略选择必须结合模型规模与任务领域；
- ✅ **推动了可复现研究**：开源平台极大降低了该领域的入门门槛。

> 🔗 项目地址：[https://github.com/usail-hkust/HRBench](https://github.com/usail-hkust/HRBench)

</details>

---

### 3. [Adaptive Multimodal Agents-Based Framework for Automatic Workflow Execution](https://arxiv.org/abs/2605.28607)

**Authors**: Susanna Cifani, Mario Luca Bernardi, Marta Cimitile  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.28607v1  

#### Abstract
Modern information systems require autonomous agents capable of navigating complex workflows, yet current methodologies often struggle with the transition from structured metadata parsing to general environmental perception. While the integration of MLLMs has enabled agents to interact directly with...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Adaptive Multimodal Agents-Based Framework for Automatic Workflow Execution

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Multimodal Large Language Models (MLLMs)** 的 GUI 导航代理在执行跨应用、多步骤工作流任务时面临以下关键挑战：
- **线性片段化记忆**：现有方法将任务序列视为独立、离散的“链式动作”（chain-of-action），缺乏对系统状态转移拓扑结构的整体感知。
- **上下文丢失与无限循环**：代理无法有效维护跨应用的状态历史，容易重复执行已完操作或陷入循环。
- **错误传播严重**：开放回路（open-loop）决策机制导致单个错误会引发后续所有步骤失败。

这些问题限制了代理在复杂、非平稳界面环境中的泛化能力和鲁棒性。

---

### 提出了什么新方法或新思路
本文提出了一种**自适应多模态多智能体框架**，通过两个核心阶段实现自动工作流执行：

#### （1）离线发现阶段（Offline Discovery Phase）
- 利用自动化流水线从碎片化的执行日志中构建一个**有向图形式的知识图谱 $G=(V,E)$**：
  - 节点 $V$ 表示唯一的 GUI 状态；
  - 边 $E$ 表示触发页面跳转的动作。
- 引入双重相似性检查机制（BGE-M3 语义嵌入 + 视觉对比）防止节点冗余，提升图谱紧凑性。
- 连续的页内操作被压缩为单一边，简化路径同时保留功能轨迹。

#### （2）推理执行阶段（Inference Execution Phase）
采用**分层多智能体架构**，包含三大模块：
- **Planning Layer**：全局规划器生成高层策略，子目标规划器动态分解并结合语义历史调整计划。
- **Execution Layer**：观察代理编码当前 GUI 状态，决策代理提出具体动作。
- **Verification Layer**：验证代理作为闭环反馈机制，审核提议动作是否逻辑一致；若拒绝则返回反馈，触发重试循环（最多 $M=4$ 次）。

此外，引入**Goal-aware Semantic History Mechanism**：
- 不再记录原子动作（如“Scroll down”），而是生成描述状态演进的叙事文本（如“Scrolled down to reveal song lyrics...”），增强上下文连贯性。

---

### 相比现有方法的优势
| 维度 | 传统方法（如 Mind2Web, SeeClick） | 本论文方法 |
|------|-------------------------------|-----------|
| 记忆结构 | 线性 episode trace | 图结构知识库（Graph-based RAG） |
| 上下文管理 | 原子动作历史 | Goal-aware 叙事型历史 |
| 决策机制 | Open-loop 执行 | Closed-loop 自我纠正（Verifier Agent） |
| 泛化能力 | 依赖相似历史路径 | 可利用拓扑关系进行动态路径发现 |

> ✅ **优势总结**：该框架实现了更强的**语义感知**、**任务分解能力**和**自适应导航性能**，尤其适用于训练数据有限的新颖或非稳态场景。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GUIOdyssey**：一个公开可用的移动端跨应用 GUI 导航基准数据集。
  - 包含 **1,666 个测试 episode**
  - 涵盖五大功能领域：
    - Information Management
    - Web Shopping
    - Media Entertainment
    - Social Sharing
    - Multi-Apps（最复杂，需跨多个领域协同）

---

### 实验设置和评估指标

#### 主要评估指标
| 指标 | 公式 | 说明 |
|------|------|------|
| **Action Matching Score (AMS)** | $\text{AMS} = \frac{1}{T_i}\sum_{t=1}^{T_i} [[a_t = a_t^*]]$ | 步骤级精度，衡量每一步预测动作与最优动作的一致性 |
| **Success Rate (SR)** | $\text{SR} = \frac{N_s}{N_{\text{total}}}$ | 任务完成率，仅当所有步骤均正确才算成功（更严格） |

> 注：由于 SR 对错误极度敏感，且当前 SOTA 在此指标上普遍偏低（<2%），作者以 AMS 为主，但仍报告 SR 以体现端到端效果。

#### 计算资源
- 硬件：4 × NVIDIA A100 (40GB)
- 框架：vLLM + Tensor Parallelism
- 平均处理时间：~173 秒/episode，总耗时约 80 小时

---

### 基线方法对比
参与比较的方法包括：
- **Gemini Pro Vision**
- **CogAgent**
- **GPT-4V**
- **Qwen2.5-VL-72B**
- **PG-Agent (w/ Qwen-72B & Qwen-30B)** —— 当前最先进的图结构代理框架

> 本文在其基础上进一步优化了历史机制与验证协议。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table I）

| Method | Overall AMS | Overall SR |
|--------|-------------|----------|
| PG-Agent (w/72B) | 47.7% | — |
| PG-Agent (w/30B) | 43.6% | 0.42% |
| **Proposed (w/30B)** | **56.4%** | **1.55%** |

#### 显著成果：
- ✅ **AMS 提升 +12.85个百分点**，绝对领先。
- ✅ **SR 超过三倍于最强基线**（1.55% vs 0.42%）。
- ✅ 在最具挑战性的 **Multi-Apps 类别中**：
  - PG-Agent (30B) 完全失败（SR = 0.0%）
  - 本文方法达到 **1.54% SR**

> 更惊人的是：**使用 30B 参数模型超越了 72B 模型的表现**，表明架构创新优于单纯扩大模型规模。

---

### 替代 backbone 的扩展实验（Table III）
在 16 个代表性 episode 上测试不同 LLM 后端：

| Model | AMS | SR |
|-------|-----|----|
| Qwen3-VL-32B-Thinking | **61.67%** | **6.67%** |
| GLM-4.6V-Flash | 46.38% | 0.0% |
| Qwen-30B（原主干） | 54.05% | 6.25% |

> 结果显示框架具有良好的模型兼容性，且更强的 backbone 可进一步释放潜力。

---

### 消融实验结果（Ablation Study, Table II）

研究两个核心组件的影响：
- **Informative Temporal Context**（语义历史）
- **Self-correction Mechanism**（Verifier Agent）

| 配置 | Overall AMS | Overall SR |
|------|-------------|------------|
| Full（完整系统） | 53.65% | 1.0% |
| Context-only（仅有语义历史） | **60.17%** | 1.0% |
| Verifier-only（仅有验证器） | 53.72% | 1.0% |

#### 发现：
- 🟢 **语义历史机制贡献最大**：相比 baseline 提升 **+6.52 pp AMS**
- 🔴 **验证器单独作用微弱**：仅提升 0.07 pp，甚至在某些领域表现略差
- ⚠️ **组合后出现负干扰**：完整系统 AMS 反而低于 context-only，推测因验证器过于保守导致误拒正确动作（false positive）

> ➤ 表明：**强大的上下文是自我纠正的前提**；未来需设计**基于置信度的自适应验证阈值**

---

## 4. 关键结论和发现

### 主要发现
1. **图结构知识库显著提升导航能力**：
   - Graph-based RAG 使代理能理解状态间的拓扑关系，支持跨任务路径复用与动态探索。
2. **Goal-aware 语义历史机制至关重要**：
   - 是避免“上下文丢失”和“无限循环”的核心技术，也是高 AMS 的主要驱动力。
3. **闭环验证机制潜力巨大但需精细调校**：
   - 能有效拦截不一致动作，但在缺乏足够语义支撑时可能产生负面干扰。
4. **架构创新可超越模型缩放效应**：
   - 30B 模型 > 72B 基线，证明合理的系统设计比盲目追求大模型更重要。

---

### 方法的局限性
1. **整体 Success Rate 仍较低**（1.55%），距离实用仍有差距。
2. **验证机制存在 false positive 问题**（占失败案例 14%），影响流畅性。
3. **计算开销大**：图谱构建与多智能体协作带来较高延迟，难以实时部署。
4. **视觉定位仍是瓶颈**：38% 的失败源于 UI 元素定位不准（尤其是小图标或相似控件）。
5. **跨应用状态保持不足**：尽管有语义历史，仍有 26% 失败源于信息未正确传递。

---

### 未来工作方向
1. **集成专用 GUI Grounding 模型**：
   - 改进视觉元素识别与定位能力，缓解视觉 grounding 错误。
2. **开发自适应验证策略**：
   - 根据动作置信度动态调节验证强度，减少不必要的干预。
3. **扩展图谱的层次抽象能力**：
   - 构建更高层级的抽象状态表示，支持更大规模的操作环境。
4. **轻量化与加速推理**：
   - 探索图谱剪枝、缓存机制、蒸馏技术以降低部署成本。
5. **多基准泛化验证**：
   - 在 OSWorld、AITW 等其他环境中测试，验证通用性。

---

> 💡 **总结一句话**：  
> 本文通过**图结构知识库 + Goal-aware 语义历史 + 多智能体闭环验证**，构建了一个更具语义感知与自我纠错能力的多模态代理框架，在 GUIOdyssey 上实现了 AMS 和 SR 的显著突破，为复杂工作流自动化提供了新的系统范式。

</details>

---

### 4. [DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting, Precise Verification, and Fully Parallel Execution](https://arxiv.org/abs/2605.28678)

**Authors**: Yunhai Hu, Zining Liu, Xiangyang Yin, Tianhua Xia, Bo Bao, Eric Sather, Vithursan Thangarasa, Sai Qian Zhang  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.28678v1  

#### Abstract
Speculative reasoning has recently been proposed as a means to accelerate reasoning-intensive generation in large multimodal models, but its effectiveness is often constrained by misalignment between speculative drafts and target-verified reasoning. In this work, we introduce DREAM-R, a framework th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DREAM-R: Multimodal Speculative Reasoning with RL-Based Refined Drafting, Precise Verification, and Fully Parallel Execution》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **speculative reasoning** 方法在语言模型（LLMs）上表现良好，但在 **multimodal large reasoning models (MLRMs)** 上效果不佳。主要原因包括：
- 多模态推理中存在 **视觉 grounding 错误**（如忽略图像线索或错误识别），导致 draft model 生成看似合理但与图像不符的推理步骤。
- 传统基于离散评分（discrete score）的验证机制对多模态不确定性敏感，容易产生不稳定判断。
- 推理过程中的串行执行方式限制了硬件利用率和端到端延迟优化。

### 提出的新方法与创新思路
作者提出 **DREAM-R**，一个专为多模态场景设计的 speculative reasoning 框架，包含三大核心技术：

#### ✅ **Speculative Alignment Policy Optimization (SAPO)**
- 一种基于 **Reinforcement Learning (RL)** 的训练目标，用于优化 draft model。
- 引入复合奖励函数：
  - `R_outcome`：最终答案正确性
  - `R_draft`：draft steps 被 target model 接受的比例
  - `R_length`：惩罚过长的 reasoning chain
- 目标是让 draft model 生成既 **faithful（忠实于目标轨迹）** 又 **concise（简洁）** 的推理步骤。

#### ✅ **Contrastive Probability Normalization (CPN)**
- 改进传统的 scalar score-based 验证机制。
- 使用 target model 对“positive”和“negative”两个关键词的概率比值 $ p = \frac{s_+}{s_+ + s_-} $ 来决定是否接受 draft step。
- 设定阈值 $ \alpha = 0.7 $，仅当正向证据明显占优时才接受，提升验证稳定性与可解释性。

#### ✅ **Fully Parallel Speculative Reasoning (FPSR)**
- 实现 **draft generation、target reasoning 和 verification 的完全并行化**。
- 允许早期终止（early stopping）和干净回滚（clean fallback），避免冗余计算。
- 显著降低 wall-clock latency，提高 GPU 利用率。

### 相比现有方法的优势
| 维度 | 现有方法（如 SpecReason、Lookahead Reasoning） | DREAM-R |
|------|---------------------------------------------|--------|
| **验证机制** | 基于离散分数（0–9），易受 prompt bias 影响 | 基于概率比值的 CPN，更稳定、可解释 |
| **draft 训练方式** | 通常不进行专门对齐训练 | 使用 RL-based SAPO 进行目标对齐 |
| **执行模式** | 多数为顺序或部分重叠 | 完全并行，支持 early stop 与 rollback |
| **适用性** | 主要面向文本模型 | 明确针对 **multimodal reasoning** 场景 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个主流的多模态推理 benchmark 上进行全面评估：
- **MathVerse**：视觉数学题理解与求解
- **MMBench**：综合多模态能力评测
- **RealWorldQA**：现实世界常识推理
- **MMMU**：跨学科专家级多模态理解

所有实验均使用完整测试集。

### 实验设置
- **Target Models**：
  - Qwen3-VL-32B (Q32B)
  - Qwen3-VL-235B-A22B (Q235B)
- **Draft Models**：
  - Qwen3-VL-2B (Q2B)
  - Qwen3-VL-4B (Q4B)
  - R-4B (R4B)
  - MiMo-VL-7B-RL (M7B-RL)

- **部署配置**：
  - Target model：4×NVIDIA L40S GPUs，AWQ INT4 量化
  - Draft model：2×NVIDIA L40S GPUs
  - 训练环境：8×NVIDIA H200 GPUs，BF16 精度

- **关键参数**：
  - Lookahead window: 4
  - Acceptance threshold $ \alpha $: 0.7
  - Max sequence length: 8196 tokens
  - Batch size: 64，学习率 $1 \times 10^{-6}$，AdamW 优化器，训练 15 轮

### 评估指标
| 指标 | 含义 |
|------|------|
| **Accuracy (Acc.)** | 最终任务准确率 |
| **Acceptance Rate (Acpt.)** | draft steps 被接受的比例 |
| **Speedup** | 相对于标准自回归 decoding 的加速比（以 wall-clock time 计算） |

### 基线方法对比
- **Standard SD**：传统 speculative decoding
- **SpecReason (Pan et al., 2025)**：首个将 speculation 扩展至 reasoning step 层面的方法
- **LR (Lookahead Reasoning, Fu et al., 2025)**：异步重叠 drafting 与 verification
- **DREAM-R-NS**：DREAM-R 的消融版本，不含 SAPO 模块（用于验证 SAPO 的作用）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）
| 方法 | Target | Draft | 数据集 | Accuracy (%) | Speedup |
|------|--------|-------|--------|--------------|---------|
| **DREAM-R** | Q32B | Q2B | MathVerse | **75.98** | **2.38×** |
| **DREAM-R** | Q32B | Q4B | MMBench | **89.44** | **2.48×** |
| **DREAM-R** | Q32B | R4B | RealWorldQA | **75.25** | **2.26×** |
| **DREAM-R** | Q235B | Q2B | MathVerse | **80.00** | **2.28×** |

> ⚡ 最高实现 **2.48× 加速**，同时保持接近目标模型的 accuracy（如从 76.00% → 75.98%）。

### 与基线方法对比
| 方法 | 特点 | 表现 |
|------|------|------|
| **SpecReason** | 使用离散打分机制 | 在强 target 下 accuracy 显著下降（如 MathVerse 从 76% → 44.57%），speedup 不稳定甚至负增益 |
| **LR** | 支持异步并行 | 有一定加速（~1.7×），但 accuracy 损失较大，尤其在 weak draft 情况下 |
| **DREAM-R-NS**（无 SAPO） | 包含 CPN + FPSR | 性能优于 SpecReason/LR，但 acceptance 和 speedup 仍低于完整版 DREAM-R |
| **DREAM-R**（完整） | SAPO + CPN + FPSR | 在所有 benchmark 上取得最佳平衡：高 accuracy + 高 speedup + 高 acceptance rate |

> 📊 示例：在 Q32B + Q2B 设置下，DREAM-R 的 acceptance rate 达到 **52.61%**，而 SpecReason 仅为 **14.60%**。

### 消融实验结果

#### 🔍 Ablation on Reward Function Design
- 当 `w1:w2:w3 = 1:1:1`（均衡权重）时整体表现最优。
- 单独增强 `R_outcome` 导致输出变长、speedup 下降（2.16× vs 2.38×）。
- 过度强调 `R_length` 会损害 acceptance（MMBench 下降至 38.78%）。
- 结论：**三者需协同调节，不可偏废**。

#### 🔍 Ablation on FPSR
- 相较于 SpecReason 和 LR，在 Q32B + Q2B 设置下：
  - SpecReason: ~1.2× speedup
  - LR: ~1.5× speedup
  - **FPSR**: **1.86× (MathVerse), 1.77× (MMBench)**
- 在更大模型（Q235B）上优势更显著，最高达 **2.38×**。
- 证明 **fully parallel execution 架构有效释放硬件潜力**。

#### 🔍 Ablation on CPN Threshold ($\alpha$)
- $\alpha = 0.5$：接受率高，速度快，但 accuracy 下降（因容忍低质量推理）
- $\alpha = 0.9$：accuracy 提升，但 speedup 明显下降
- $\alpha = 0.7$：**最佳权衡点**，兼顾效率与准确性

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **多模态 speculative reasoning 必须考虑视觉 grounding 一致性**，否则 draft model 易产生“逻辑连贯但视觉错误”的幻觉。
2. ✅ **传统基于分数的验证机制在多模态场景下不稳定**，而 CPN 提供了一种更鲁棒、可解释的替代方案。
3. ✅ **通过 RL-based SAPO 训练 draft model，可显著提升其与 target model 的 alignment**，从而提高 acceptance rate 和最终 accuracy。
4. ✅ **FPSR 实现了真正的全流程并行化**，相比 lookahead 类方法进一步压缩延迟，且支持 rollback 保证正确性。
5. ✅ DREAM-R 在多种 target/draft 组合下均表现出色，具备良好的泛化性和实用性。

### 方法的局限性
- ❗ **依赖高质量的 step-level annotated training data**（如 Geo3K、OCR-VQA、ScienceQA）来训练 SAPO。
- ❗ 当 draft model 能力远弱于 target（如 2B vs 235B）时，即使经过 SAPO 微调，acceptance 仍有上限。
- ❗ 当前框架假设 draft 和 target 拥有相同 tokenizer 和输入格式，跨架构适配尚需额外处理。
- ❗ 实验集中在静态图像+文本任务，未覆盖视频或多轮交互场景。

### 未来工作方向
- 🔄 将 DREAM-R 扩展至 **video-language reasoning** 和 **embodied agents** 中的动态决策流。
- 🤖 探索 **self-improving draft models**，利用 target model 的反馈持续迭代更新 draft policy。
- 💡 研究 **zero-shot 或 few-shot alignment** 方法，减少对大量标注数据的依赖。
- ⚙️ 开发 **hardware-aware scheduler**，结合 GPU 内存带宽与计算密度动态调整 lookahead window 和 threshold。
- 🔐 引入 **uncertainty-aware verification**，使系统能在高风险场景自动切换为保守策略。

---

> ✅ **总结一句话**：  
> DREAM-R 是首个专为 **multimodal speculative reasoning** 设计的高效推理框架，通过 **SAPO + CPN + FPSR** 三位一体的技术组合，在几乎不损失 accuracy 的前提下实现了高达 **2.48× 的推理加速**，为大规模多模态推理系统的实用化提供了重要路径。

🔗 **代码开源地址**：[https://github.com/HuYunhai-Alex/DREAM-R](https://github.com/HuYunhai-Alex/DREAM-R)

</details>

---

### 5. [Rethinking Visual Neglect: Steering via Context-Preference for MLLM Hallucination Mitigation](https://arxiv.org/abs/2605.27993)

**Authors**: Jingwen Wu, Xijun Zhang, Ge Song  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.27993v1  

#### Abstract
Object hallucination remains a primary obstacle to the reliable deployment of Multimodal Large Language Models (MLLMs). Current inference-time mitigation methods mainly assume hallucinations stem from visual neglect, steering models to enhance visual reliance. In contrast, our systematic interventio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Rethinking Visual Neglect: Steering via Context-Preference for MLLM Hallucination Mitigation**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于 **Multimodal Large Language Models (MLLMs)** 中普遍存在的 **object hallucination**（对象幻觉）问题，即模型在生成描述时虚构图像中不存在的对象。当前主流方法通常假设幻觉源于“**视觉忽视**”（visual neglect），即模型过度依赖参数化知识或文本先验而忽略图像输入，并因此提出增强视觉注意力的干预策略。

然而，作者指出这一假设是**不充分的**：简单地“增强视觉依赖”并不总能缓解幻觉，有时反而会加剧问题。

---

### 🚀 提出的新方法与新思路

作者提出了一个全新的视角——将图像视为一种**上下文信号**，它同时与两种内部语义信号竞争：
1. **Parametric Knowledge**（来自预训练的内部知识）
2. **Textual Context**（由指令和生成历史构成的文本上下文）

基于此，论文提出了一种解耦的视觉偏好建模方式，并引入两个独立的控制维度：

- **Visual Fidelity Vector (VFV)**：衡量模型对**外部视觉输入 vs 内部参数知识**的偏好（external-vs-internal）。
- **Modality Reliance Vector (MRV)**：衡量模型对**视觉上下文 vs 文本上下文**的偏好（cross-modal）。

由此提出 **Context-Preference Activation Steering (CAS)** 框架：

- **无需训练**（training-free）
- 利用少量设计好的冲突样本（conflict samples）提取 VFV 和 MRV
- 在推理阶段通过单次前向传播，在中间 MLP 层注入带符号残差（signed residual injection）进行激活引导

---

### 🔍 相比现有方法的优势

| 特性 | CAS | 其他主流方法（如 PAI, VCD, OPERA 等） |
|------|-----|-------------------------------|
| 是否需重新训练 | ❌ 否 | ❌ 否（多数为 inference-time） |
| 干预效率 | 单次前向，无额外解码开销 | 多数需要双通路、回溯或多步搜索 |
| 对文本质量影响 | 几乎无损，保持原生生成流畅性 | 易导致重复退化（repetitive degeneration） |
| 适应性 | 可针对不同模型定制干预方向（正/负） | 统一“加强视觉”策略，缺乏灵活性 |
| 数据需求 | 极低（约百个样本） | 部分依赖大量对比样本或外部模型 |

> ✅ **核心优势**：轻量、高效、自适应、不牺牲生成质量。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 类型 | 数据集 | 任务说明 |
|------|--------|----------|
| **生成式评估** | **CHAIR** (500 COCO 图像) | 详细图像描述中的幻觉检测，报告 CHAIRs, CHAIRr, F1 |
| | **AMBER** (1,004 图像) | 多维幻觉评测基准，报告 Hal, CHAIR, Cog, Cover |
| **判别式评估** | **POPE** (3k 问题 × 3 子集) | 二分类判断对象是否存在，报告 Acc 和 F1 |

---

### ⚙️ 实验设置与评估指标

#### 模型
在四个主流开源 MLLM 上测试：
- LLaVA-1.5
- Shikra
- Qwen-VL
- InstructBLIP

所有模型均为 7B 参数规模。

#### 评估指标
| 指标 | 含义 |
|------|------|
| **CHAIRs / CHAIRr ↓** | 幻觉率（越低越好） |
| **Hal ↓** | AMBER 中的整体幻觉得分 |
| **Cover ↑** | 描述覆盖率（越高越好） |
| **Acc / F1 ↑** | POPE 判别任务准确率与F1 |
| **Rep ↓** | n-gram 重复率（用于检测 degeneration） |
| **TTR ↑** | Type-Token Ratio，衡量词汇多样性 |

#### 基线方法对比（共8种 training-free 方法）
按发表顺序排列：
- DoLa
- VCD
- OPERA
- PAI
- Code
- DeCo
- AttnReal
- SSL

所有方法均采用原始推荐配置。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ 表1：CHAIR 结果（生成式幻觉）

| 方法 | LLaVA-1.5 (CHAIRs↓) | Shikra | Qwen-VL | InstructBLIP |
|------|---------------------|--------|---------|---------------|
| Vanilla | 57.6 | 61.0 | 50.8 | 52.6 |
| **CAS (Ours)** | **35.8** | **44.0** | **38.0** | **36.0** |
| 最佳降幅 | **-37.8%** | -27.9% | -25.2% | -31.6% |

> ✅ CAS 在所有模型上取得最低或并列最低的 CHAIRs，且 **Rep 值接近 vanilla**，显著优于 PAI、OPERA 等出现严重重复的方法。

#### ✅ 表2：AMBER 生成结果

| 方法 | LLaVA-1.5 (Hal↓) | Qwen-VL | InstructBLIP |
|------|------------------|---------|---------------|
| Vanilla | 31.1 | 29.6 | 35.3 |
| **CAS** | **17.2** | **23.5** | **28.8** |
| 降幅 | **-44.7%** | -20.6% | -18.4% |

> ✅ CAS 显著降低 Hal 分数，同时 **Cover 接近 vanilla**，避免了 PAI 导致的覆盖崩溃（如 Shikra 上从 51.4 → 33.5）。

#### ✅ 表3：POPE 判别任务结果（平均 F1）

| 方法 | LLaVA-1.5 | Shikra | Qwen-VL | InstructBLIP |
|------|-----------|--------|---------|--------------|
| Vanilla | 83.49 | 78.46 | 87.10 | 79.51 |
| **CAS** | **86.11** | **81.20** | **87.53** | **84.74** |
| 提升幅度 | +2.62 | +2.74 | +0.43 | +5.23 |

> ✅ CAS 在所有模型上达到最高 F1，尤其在 InstructBLIP 上提升达 **6.6%**。

---

### 🔬 消融实验结果

#### （1）VFV 与 MRV 的独立性验证
- 计算两向量夹角余弦：`cos(v_VFV, v_MRV)` ∈ [-0.06, +0.11]
- 平均绝对值 ≈ 0.033，接近随机向量在 R^4096 空间的期望
> ✅ **结论**：VFV 与 MRV 在隐空间中近似正交，代表两个独立控制维度。

#### （2）每模型最优干预方向不同（见图3）
- 不同模型对 VFV/MRV 的响应符号各异：
  - 如 LLaVA-1.5：需 `α=-2`（减少 parametric reliance），`β=+1`（增加 visual reliance）
  - Qwen-VL：需 `α=+1`, `β=-1`
> ✅ **结论**：不能统一施加“更强视觉关注”，必须**按模型定制干预方向**。

#### （3）层分析（Table 4）
- 干预位置选择 **L11–L14**（mid-early MLP layers）
- 更浅层（L1–9）会导致严重退化（Repetition ×280）
- 更深层（L16+）效果归零
> ✅ **结论**：存在一个有效且稳定的干预窗口。

---

## 4. 关键结论和发现

### 🧠 主要发现

1. **“视觉忽视”不是唯一原因**  
   单纯增强视觉注意力可能适得其反，因图像作为上下文需与 parametric knowledge 和 textual context 三方博弈。

2. **视觉偏好应被解耦为两个独立维度**  
   - VFV（external-vs-internal）
   - MRV（cross-modal）
   二者几何独立，行为响应也相互独立。

3. **不同 MLLM 对干预有异质性反应**  
   同一干预方向在不同模型上可能导致相反效果，强调个性化调节的重要性。

4. **CAS 实现高效、无损幻觉抑制**  
   - 单次前向完成干预
   - 不增加解码延迟（latency ≈ vanilla）
   - 不引发重复退化（Rep ≪ PAI/OPERA）

---

### ⚠️ 方法的局限性

1. **仅针对 object hallucination**  
   尚未验证对 attribute 错误、空间关系误判、逻辑推理幻觉的有效性。

2. **限于 7B 规模模型**  
   更大模型（如 70B）可能存在不同的最优干预层，当前 L11–L14 可能不再适用。

3. **CPV 样本质量受限**  
   - MRV 使用 SD 1.5 生成图像，存在轻微视觉偏差
   - VFV 虽用高质量 Qwen-Image-2.0，但仍非完美

4. **缺乏理论机制解释**  
   当前为经验性有效，尚未深入分析其如何作用于 multimodal fusion pathway。

---

### 🔮 未来工作方向

1. 扩展至更复杂的幻觉类型（attribute, relation, reasoning）
2. 探索更大模型下的 layer drift 与自适应定位机制
3. 使用更高保真度 T2I 模型重构 CPV 样本以减少偏置
4. 建立理论框架解释 CAS 如何调控多模态特征融合路径
5. 探索动态在线更新 CPV 的可能性（adaptation to domain shift）

---

> 💡 **总结一句话**：  
> 本文挑战了“越多视觉=越少幻觉”的共识，提出 **CAS** 框架通过解耦上下文偏好、精准注入控制信号，在无需训练的前提下实现高效、稳定、无损的幻觉缓解，为 MLLM inference-time mitigation 提供了新范式。

</details>

---

### 6. [Meta-Attention: Bayesian Per-Token Routing for Efficient Transformer Inference](https://arxiv.org/abs/2605.28384)

**Authors**: Alan Ferrari  
**Category**: cs.LG  
**Published**: 2026-05-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.28384v1  

#### Abstract
Standard transformer architectures apply a single attention mechanism uniformly across all tokens and sequence positions, irrespective of local context or computational budget. We propose Meta-Attention, a framework that dynamically routes each token to the most appropriate attention strategy -- ful...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Meta-Attention: Bayesian Per-Token Routing for Efficient Transformer Inference》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
标准的 Transformer 架构对所有 token 统一应用相同的 attention 机制（如 full softmax attention），无论其上下文复杂度或计算成本如何。这种“一刀切”的设计导致：
- 在局部上下文简单的 token 上浪费大量计算资源（O(n²) 复杂度）；
- 在需要长程依赖的关键位置可能因使用近似 attention 而损失精度。

现有高效 attention 方法（如 Sparse、Local、Linear attention）通常在架构层面固定一种策略，无法动态适应不同 token 的需求。

### 提出的新方法与新思路
提出 **Meta-Attention** 框架，引入一个 **Bayesian Meta-Controller**，实现 per-token 的自适应 attention 路由：

- **动态路由机制**：每个 token 可被路由至最合适的 attention 专家（expert）：
  - `E1`: Full Softmax Attention（高表达力，高成本）
  - `E2`: Linear (Kernel) Attention（低复杂度 O(n)，低成本）
  - `E3`: Sliding-Window Local Attention（中等成本）
- **贝叶斯视角建模路由决策**：将每 token 的机制选择视为后验推断问题，在一个 **compute-aware Dirichlet prior** 下进行。
- **ELBO 训练目标**：联合优化任务性能与计算成本，通过 KL 散度正则项编码先验偏好（偏向低成本专家）。
- **不确定性驱动软硬切换**：利用变分后验的熵作为 per-token 不确定性估计，指导从 soft routing 向 hard routing 的过渡。

### 相比现有方法的优势
| 方面 | Meta-Attention 的优势 |
|------|------------------------|
| **灵活性** | 动态按需分配计算，而非全局固定策略 |
| **原则性** | 使用贝叶斯框架替代启发式或无先验的 learned routing，避免 routing collapse |
| **效率-性能权衡** | 显著降低预期 FLOPs，同时保持任务性能 |
| **可解释性** | 输出 per-token 路由权重和不确定性，提供模型行为洞察 |
| **组合性** | 与 MoD（Mixture of Depths）、AttnRes 正交且可堆叠 |

---

## 2. 核心实验方法和设置

### 数据集
- **Phase 1 实验**：使用 **WikiText-2 子集（1MB）** 上训练的字符级语言模型（Tiny LM）进行消融研究。
- **未来计划（Phase 2）**：将在 **WikiText-103** 上进行大规模验证，报告绝对 perplexity 和真实 FLOP 节省。

### 实验设置
- **模型结构**：2 层 Transformer，隐藏维度 D=128，序列长度 T=64。
- **训练配置**：2000 步，batch size=32，Adam 优化器。
- **Meta-Attention 配置**：
  - 三个 attention 专家并行运行（soft routing）
  - Bayesian Meta-Controller 是一个两层 MLP，输出 Dirichlet 分布参数
  - 先验设置为 `β = ε + β₀(1−c)`，其中 `ε=0.01` 防止退化，`β₀=1.0`
- **对比设置**：
  - **Bayesian 版本**：`β_elbo = 1`，使用完整 ELBO 目标
  - **Non-Bayesian / Prior-Free 基线**：`β_elbo → 0`，即移除 KL 正则项

### 评估指标
| 指标 | 描述 |
|------|------|
| **Normalized PPL** | 相对于非贝叶斯基线的相对困惑度（baseline 设为 1.0） |
| **Routing Entropy (%)** | 路由分布的平均熵，衡量路由是否集中（越低越好） |
| **Projected FLOP Cost (%)** | 基于当前路由概率加权的成本估计（∑αᵢcᵢ），反映 Phase 3 硬路由下的理论节省 |
| **Forward-pass Correctness** | 输出形状、数值稳定性、初始路由偏置等正确性检查 |

### 基线方法对比
- 主要对比对象是 **Prior-Free Learned Routing**（即去掉 Dirichlet prior 和 KL 正则的版本）
- 隐含对比其他静态高效 attention 方法（如 Longformer、Performer），但未直接复现

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

| 指标 | Bayesian (Ours) | Non-Bayesian (Baseline) | 对比结果 |
|------|------------------|--------------------------|-----------|
| **Normalized PPL** | 1.07 | 1.00（ref） | +6.3% 开销 |
| **Routing Entropy (%)** | **43.3%** | 55.8% | ↓12.5 个百分点 |
| **Projected FLOP Cost (%)** | **25.1%** | 59.3% | ↓34.2 个百分点 |
| **Projected Cost Ratio** | — | — | **2.4× 更低** |

> 注：“Projected” 表示这是基于当前 soft routing 分布预测的 Phase 3 硬路由成本，并非实测 wall-clock 时间。

### 与基线方法的对比结果
- **显著提升计算效率**：贝叶斯控制器将预计 FLOP 成本从 59.3% 降至 **25.1%**，意味着在硬路由下可实现约 **64% 的 FLOP 减少**（相对于 full attention）。
- **有效防止 routing collapse**：非贝叶斯模型趋向于退化为接近 full attention 的路由模式（高成本），而贝叶斯模型成功引导更多 token 使用廉价专家（E2/E3）。
- **更稳定的路由决策**：更低的 routing entropy 表明贝叶斯后验更“果断”，有利于后续 hard routing 实现真正加速。
- **可控的性能代价**：仅付出 **6.3% 的相对 PPL 上升**，换取 2.4 倍的效率增益，作者认为这是一个非常有利的 trade-off。

### 消融实验结果
- **控制变量对比**：唯一差异是是否启用 ELBO 中的 KL 正则项（即是否有 compute-aware prior）。
- 结果明确显示：
  - 加入贝叶斯 prior 后，模型不再默认选择昂贵的 full attention；
  - 初始阶段即形成对低成本专家的偏好（见初始化检查表 2）；
  - 路由熵下降更快，表明结构化路由更早出现（与 [18] 的“相变”预测一致）。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **贝叶斯先验能有效防止 routing collapse**：通过 `β = ε + β₀(1−c)` 的设计，天然鼓励模型优先尝试低成本 attention 机制。
2. ✅ **更好的 compute-performance trade-off**：相比 prior-free 方法，在轻微性能损失下获得巨大计算节省（2.4× 投影成本优势）。
3. ✅ **不确定性可用于软硬路由过渡**：变分后验的熵提供了 principled 的信号，用于决定何时切换到单专家执行。
4. ✅ **初期实验证实理论预测**：Phase 1 结果支持 [24]（VMoER）关于“贝叶斯路由提升稳定性和校准性”的结论。
5. ✅ **框架具有高度可扩展性与组合性**：可与 MoD、AttnRes 等正交技术叠加，构建更复杂的 conditional computation pipeline。

### 方法的局限性（作者承认）
| 局限性 | 说明 |
|--------|------|
| **尚未实测真实加速** | 当前 soft routing 并行运行三专家，无实际速度提升；Phase 3 才会实现 hard routing 和真实 FLOP 减少 |
| **小规模实验环境** | Tiny LM 实验仅为原理验证，缺乏 WikiText-103 规模的绝对 PPL 支持 |
| **梯度方差未知** | Dirichlet reparameterization 的梯度方差可能在大模型训练中成为问题 |
| **先验敏感性** | `β₀` 超参可能需根据任务和深度调整，目前为手动设定 |
| **salience 特征简单** | 当前使用 `[x; ||x||/√D; pos]` 作为控制器输入，可能不足以捕捉语义重要性 |
| **缺少 SSM 专家** | 如 Mamba 等 State Space Model 尚未集成，因其状态机制与 attention 不兼容（Phase 2 计划） |

### 未来工作方向（Phased Roadmap）
| 阶段 | 目标 |
|------|------|
| **Phase 2** | - 在 WikiText-103 上训练，报告绝对 PPL 和 KL 曲线<br>- 分析 posterior concentration 与 collapse 动力学<br>- 引入 repetition curriculum 测试 routing emergence<br>- 对比 Bayesian vs Prior-Free 在相同 FLOP 预算下的表现 |
| **Phase 3** | - 实现 uncertainty-gated hard routing<br>- 进行真实 wall-clock FLOP 测量<br>- 探索阈值 `η` 的敏感性分析和校准误差（ECE） |
| **长期方向** | - 集成 SSM 专家（如 Mamba）<br>- 升级 E1 为 Gated Attention [17]<br>- 设计 per-layer 自适应 `β₀` 或 empirical Bayes 估计<br>- 探索 routing emergence 的相变现象 |

---

> 🔗 **代码开源地址**：https://github.com/KFEAL/meta-attention  
> 📄 **论文状态**：Preprint，Under Review

该工作为高效 Transformer 推理提供了一个**原则性强、可解释、可组合**的新范式，有望推动 conditional computation 在大模型中的深入应用。

</details>

---

### 7. [LaneRoPE: Positional Encoding for Collaborative Parallel Reasoning and Generation](https://arxiv.org/abs/2605.27570)

**Authors**: Gabriele Cesa, Thomas Hehn, Aleix Torres-Camps, \`Alex Batlle Casellas, Jordi Ros-Giralt, Arash Behboodi, Tribhuvanesh Orekondy  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.27570v1  

#### Abstract
Parallel LLM test-time scaling techniques (e.g., best-of-$N$) require drawing $N>1$ sequences conditioned on the same input prompt. These methods boost accuracy while exploiting the computational efficiency of batching $N$ generations. However, each sequence in the batch is traditionally generated i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LaneRoPE: Positional Encoding for Collaborative Parallel Reasoning and Generation

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的并行推理方法（如 best-of-N、self-consistency）在生成多个响应时，各序列是**独立采样**的，无法共享中间推理过程或计算结果。这导致：
- 重复计算，资源利用效率低；
- 难以利用问题的可分解结构进行协作求解；
- 限制了多路径探索中的信息交互。

### 提出的新方法：LaneRoPE
作者提出 **LaneRoPE**（Lane-based Rotary Positional Encoding），一种支持**协作式并行推理与生成**的位置编码方案，其核心思想包括两点：

1. **跨序列注意力掩码（Inter-sequence Attention Mask）**  
   允许一个序列在生成下一个 token 时，能够因果地关注其他所有并行序列的历史 token，实现 token 级别的协作。

2. **扩展的 RoPE 编码机制**  
   在标准 Rotary Positional Embedding (RoPE) 的基础上，引入**第二维旋转矩阵**来编码“序列索引”（lane index），从而建模两个维度上的相对位置：
   - 序列内 token 之间的相对位置；
   - 不同序列之间 token 的相对位置。

该设计使得模型能够在保持批处理高效性的同时，动态学习跨 lane 的依赖关系。

### 相比现有方法的优势
| 方法 | 特点 | LaneRoPE 优势 |
|------|------|----------------|
| **Best-of-N / Self-consistency** | 并行但独立采样 | 支持跨序列信息共享，提升准确率 |
| **GroupThink [12]** | 将多个序列拼接为虚拟长序列 | LaneRoPE 更灵活，避免负虚拟位置问题；可通过 NTK-aware 初始化缓解分布偏移 |
| **Hogwild! [27]** | 动态重排序序列以避免负索引 | 需定制 attention 内核；LaneRoPE 可直接集成到标准架构中 |
| **Bridge [7]** | 引入额外 attention 层实现 axial attention | 增加推理延迟；LaneRoPE 几乎无额外开销 |

> ✅ **核心优势总结**：
> - 极小改动即可集成进现有 LLM 推理流程；
> - 引入的计算开销可忽略不计（见 Table 2）；
> - 支持 fine-tuning，且仅需微调极少量参数（<0.5%）；
> - 统一框架涵盖多种已有方法（如 GroupThink 是其特例）。

---

## 2. 核心实验方法和设置

### 使用的数据集
在多个数学推理基准上进行评估：
- **MATH500**：500道高难度数学题；
- **AMC 23**：美国数学竞赛题目；
- **AIME 24 & AIME 25**：美国邀请赛数学考试题。

这些任务对多步推理能力要求高，适合测试协作推理效果。

### 实验设置
- **模型基础**：基于开源的 `DeepSeek-R1-Distill-Qwen` 模型，测试 **1.5B 和 7B** 参数版本。
- **并行数量 N**：测试 N = 1, 2, 4 条 lane（即同时生成 N 个相关序列）。
- **训练策略**：
  - **Supervised Fine-Tuning (SFT)**：使用合成的协作对话数据；
  - **KTO (Kahneman-Tversky Optimization)**：基于成功/失败推理轨迹优化偏好。
- **推理配置**：
  - 温度 = 0.6，top-p = 0.95；
  - 最大生成长度 = 4096；
  - 使用 Flash Attention 加速。

### 评估指标
- **maj@4**：从每组查询的 4 个并行样本中取多数投票结果作为最终答案，用于公平比较不同 N 下的性能（固定预算 B=4）；
- **Pass@1 (Accuracy)**：单个样本正确率，反映个体生成质量；
- **推理时间**：测量生成速度，验证实际部署可行性。

### 基线方法对比
- **Baseline**: 原始 DeepSeek 模型（sequential）
- **+Hogwild! [27]**：官方实现
- **+Bridge [7]**：作者自行复现并训练
- **+LaneRoPE(GT)**：GroupThink 初始化
- **+LaneRoPE(NTK)**：NTK-aware 初始化（改进版）
- **+LaneRoPE(NTK*)**：频率可学习 + KTO/SFT 微调

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1 & Table 3）

#### ✅ **7B 模型上的 maj@4 性能汇总（平均得分）**

| 方法 | N=1 | N=2 | N=4 | Avg |
|------|-----|-----|-----|-----|
| Base (DS-Qwen-7B) | 52.1 | — | — | 52.1 |
| +Bridge [7] | 59.3 | 60.0 | 61.9 | ↑~10 pts |
| **+LaneRoPE(NTK*) + KTO** | **63.0** | **63.9** | **64.1** | **↑12 pts** |

> 🔺 在 N=4 时达到 **64.1 avg maj@4**，显著优于所有基线。

#### ✅ Pass@1 单样本准确率（Table 3）
- LaneRoPE 训练后不仅未降低个体性能，反而有所提升（例如 MATH500 上从 77.2 → 84.9）；
- 表明协作并未牺牲个体推理能力，而是增强了整体表现。

### 与基线方法的对比结果
| 对比项 | 结果 |
|--------|------|
| **vs. Bridge** | LaneRoPE 在相同训练条件下性能更高，且推理更快（↓~25% latency） |
| **vs. Hogwild!** | 官方实现反而性能下降（可能因 base model 差异），而 LaneRoPE 稳定增益 |
| **vs. GroupThink** | 原始 GroupThink 初始化会导致性能崩溃（如 AIME24→0.0），而 NTK-aware 初始化有效缓解此问题 |

### 消融实验结果（Table 4）
| 配置 | AIME24 | AIME25 | AMC23 | Avg |
|------|-------|--------|--------|-----|
| GT 初始化 + SFT | 29.6 | 28.0 | 73.5 | 43.7 |
| GT 初始化 + KTO | 46.2 | 33.8 | 82.5 | 54.2 |
| NTK 初始化 + KTO | 44.5 | 33.6 | 83.1 | 53.8 |
| **NTK* + KTO (可学习 freq.)** | **46.5** | **33.3** | **84.2** | **54.7** |

> 🔍 **关键发现**：
> - **KTO > SFT**：KTO 利用更大规模非成功轨迹数据，效果更优；
> - **NTK-aware 初始化 > GroupThink**：有效防止负位置带来的训练不稳定；
> - **可学习频率 (* ) 提升明显**：说明 LaneRoPE 的灵活性至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **协作式并行生成确实能提升复杂推理任务的表现**，尤其是在数学类需要多路径探索的任务上；
2. ✅ **LaneRoPE 是轻量高效的解决方案**：几乎零推理开销（+6% 内），兼容标准 RoPE 架构；
3. ✅ **初始化策略至关重要**：原始 GroupThink 易导致负位置问题，NTK-aware 插值法可有效缓解；
4. ✅ **KTO 是有效的训练目标**：无需人工标注标签，利用成功/失败轨迹即可引导协作行为；
5. ✅ **大模型更具协作潜力**：7B 模型收益远大于 1.5B，符合“更大的模型更能学会协作”的趋势。

### 方法的局限性
- 当前实验限于 **N ≤ 4** lanes，大规模并行下的上下文膨胀问题尚未深入研究；
- 未设计专门的输出融合机制（如 consensus module），仍依赖 majority voting；
- 合成数据依赖强提示工程，真实场景下的泛化能力有待验证；
- 虽然支持 fine-tuning，但目前仍需外部数据或 distillation 流程激活能力。

### 未来工作方向
- 探索 **dedicated merging heads** 或 **consensus modules** 来自动整合多 lane 输出；
- 扩展至 **N >> 8** 场景，结合 hierarchical attention 或 sparse cross-lane 连接；
- 引入 **Reinforcement Learning with Verifiable Feedback (RLVF)** 进一步挖掘协作策略空间；
- 将 LaneRoPE 应用于 coding、planning 等更广泛的 reasoning 任务；
- 研究如何在 **edge devices** 上高效部署多 lane 推理（利用其低 overhead 特性）。

---

> 📌 **总体评价**：  
> LaneRoPE 是一个简洁而强大的创新，它通过**重新思考 positional encoding 的作用边界**，将“序列身份”也纳入位置建模范畴，从而自然支持跨 lane 协作。其设计理念优雅，工程友好，有望成为下一代 LLM 并行推理的标准组件之一。

</details>

---

### 8. [Bridging the Detection-to-Abstention Gap in Reasoning Models under Insufficient Information](https://arxiv.org/abs/2605.28070)

**Authors**: Renjie Gu, Jiaxu Li, Yihao Wang, Yun Yue, Hansong Xiao, Yefei Chen, Yuan Wang, Chunxiao Guo, Pei Wei, Jinjie Gu, Yixin Cao  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.28070v1  

#### Abstract
We highlight a failure mode of large reasoning models on questions with insufficient information: models may recognize that a problem is under-specified, yet still continue reasoning and produce unsupported final answers instead of abstaining. We formalize this mismatch as the detection-to-abstentio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Bridging the Detection-to-Abstention Gap in Reasoning Models under Insufficient Information 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于大型推理模型（Large Reasoning Models, LRMs）在面对**信息不足的问题（under-specified questions）**时的一个关键失败模式：  
尽管模型在中间推理过程中可能已经识别到前提缺失（如缺少关键症状、数值或条件），但仍会继续推理并生成一个无依据的最终答案，而不是选择**abstain（拒绝回答）**。

作者将这一现象形式化为 **“detection-to-abstention gap”（检测到拒绝的差距）** —— 即模型能“检测”到信息不充分，却未能将其转化为“拒绝作答”的行为。这在医疗等高风险领域尤为危险，因为错误的回答可能比不回答更具危害性。

### 提出了什么新方法或新思路
为解决该问题，作者提出了一种新的轨迹级推理控制框架：**Judge-Then-Solve (JTS)**。

其核心思想是：
- 将 **abstention** 不再视为一种最终输出风格，而是作为对整个推理轨迹的**控制决策**。
- 在推理开始前，强制模型先进行一个显式的 **answerability judgment（可回答性判断）**，封装在一个 `<answerability_judge>` 结构块中。
- 判断流程包括三个部分：
  1. **Contextual Audit**：分析已有上下文是否足够；
  2. **Integrity Check**：验证是否存在缺失或模糊的前提；
  3. **Conclusion**：明确输出 `ANSWERABLE` 或 `UNANSWERABLE`。
- 若判断为 `UNANSWERABLE`，则立即终止推理并返回拒绝回答；否则才进入正式求解过程。

### 相比现有方法的优势
- **结构化控制优于后处理或提示工程**：相比简单的 prompting 或仅优化最终输出的 RL 方法，JTS 显式地将判断前置，实现了从“被动拒绝”到“主动控制”的转变。
- **引入 conditional length-shaping reward**：鼓励在不可回答问题上尽早终止，避免“过度思考”（overthinking）；同时在可回答问题上保留足够的推理长度。
- **提升推理效率与可靠性**：不仅提高了 abstention 的准确性，还显著减少了无效长推理，提升了推理效率。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **Missing-Premise (MIP) dataset**：包含数学类 under-specified 和 well-defined 问题，如 Math-500 和 GSM8K 子集。
- **AbstentionBench**：跨领域的拒绝回答评测基准，包含四个子集：
  - MMLU-History
  - MMLU-Math
  - GPQA-Diamond
  - **MedIQ**（医疗诊断场景，来自 [20]）
- **Omni-Math**：用于评估在困难但可回答数学题上的泛化能力。

### 实验设置和评估指标

#### 模型架构
- **Qwen3-30B-A3B-Thinking**：MoE 架构的 LRM
- **DeepSeek-R1-Distill-Qwen-14B**：dense 架构的 LRM

#### 训练方法
- **Supervised Warm-up (SFT)**：先用构造好的 JTS 格式数据进行轻量微调，教会模型遵循结构。
- **Group Relative Policy Optimization (GRPO)**：强化学习阶段，采用结构化奖励函数：
  - `R_format`：确保输出符合 JTS 结构
  - `R_consistency`：保证判断结论与最终行为一致
  - `R_task`：任务正确性奖励
  - `R_length`：基于长度的 shaping 奖励（仅用于失败样本）

#### 评估指标
| 指标 | 含义 |
|------|------|
| **DR (Detection Rate)** | 模型是否在推理中明确指出信息缺失 |
| **OAR (Overall Abstention Rate)** | 最终是否拒绝回答 |
| **A@D (Abstention@Detection)** | 在已检测到信息缺失的前提下，是否成功 abstain：<br>$$ A@D = \frac{\text{Detected} \cap \text{Abstained}}{\text{Detected}} $$ |
| **Answer Rate / Correct Rate** | 在 well-defined 问题上的作答率与正确率 |
| **Corr./1K Tok** | 每千 token 的正确率，衡量推理效率 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

| Method | Model | DR ↑ | OAR ↑ | **A@D ↑** | Avg. Len ↓ |
|--------|-------|-----|------|----------|------------|
| Base | DeepSeek | 45.3 | 18.6 | 41.1 | 2605.9 |
| Plain RL | DeepSeek | 64.7 | 52.7 | 81.4 | 1765.1 |
| Prompting | DeepSeek | 56.5 | 52.7 | 93.3 | 714.6 |
| **Ours: JTS** | **DeepSeek** | **88.7** | **88.5** | **99.8** | **349.0** |
| Base | Qwen3 | 52.9 | 21.1 | 40.0 | 2839.4 |
| Plain RL | Qwen3 | 69.4 | 63.6 | 91.7 | 1200.0 |
| Prompting | Qwen3 | 72.8 | 65.8 | 90.4 | 959.9 |
| **Ours: JTS** | **Qwen3** | **72.9** | **72.4** | **99.3** | **342.9** |

> ✅ **A@D 接近饱和（>99%）**，表明 JTS 成功弥合了 detection-to-abstention gap。

#### 在 well-defined 问题上的表现（Table 2）
| Method | Model | Ans. Rate ↓ | Correct Rate ↓ | **Corr./1K Tok ↑** |
|--------|-------|-------------|----------------|------------------|
| Base | DeepSeek | 100.0 | 91.8 | 0.544 |
| Plain RL | DeepSeek | 99.4 | 90.3 | 0.510 |
| **Ours: JTS** | **DeepSeek** | **92.4** | **86.8** | **1.028** |
| Base | Qwen3 | 100.0 | 97.5 | 0.367 |
| **Ours: JTS** | **Qwen3** | **94.7** | **93.4** | **1.152** |

> ⚠️ 虽然 raw 正确率略有下降，但 **Corr./1K Tok 提升超过 2 倍**，说明 JTS 显著提升了单位 token 的推理效率。

### 与基线方法的对比结果
- **vs Base Model**：大幅提高 OAR 和 A@D，尤其在 DeepSeek 上 A@D 从 41.1 → 99.8。
- **vs Plain RL**：尽管 RL 已改善 abstention，但仍有明显 gap（如 Qwen3 上 DR=69.4 vs OAR=63.6），而 JTS 几乎完全闭合该 gap。
- **vs Prompting**：虽然 prompting 在 A@D 上表现尚可，但整体 OAR 较低，且缺乏训练层面的稳定性。

### 消融实验结果（Table 4 & 5）
| 变体 | A@D (DeepSeek) | OAR | Avg. Len |
|------|----------------|-----|---------|
| Plain SFT-RL + Length | 90.0 | 61.1 | 800.2 |
| JTS-SFT-RL | 99.1 | 79.7 | 522.3 |
| **JTS (Full)** | **99.8** | **88.5** | **349.0** |

> 🔍 结果显示：
> - 仅靠 SFT + length shaping 无法完全闭合 gap；
> - JTS-SFT-RL 已大幅提升 A@D；
> - 加入 length shaping 后进一步压缩响应长度，实现最优平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **Detection ≠ Abstention**：当前主流 LRM 能检测信息缺失，但常因“过度思考”而仍输出答案，存在明显的 detection-to-abstention gap。
2. **JTS 有效闭合该 gap**：通过将 answerability 判断前置并作为控制门控，使模型一旦检测即刻 abstain，A@D 达到近 100%。
3. **提升推理效率**：JTS 大幅缩短了在 under-specified 问题上的响应长度（平均从 ~2700 → ~350 tokens），减少计算浪费。
4. **增强不确定性感知**：token-level entropy 分析显示，JTS 模型在关键缺失处保持更高不确定性，而非盲目假设。
5. **副作用正向**：在 Omni-Math 上发现，missing-premise 训练还能减少无效自省（hesitation），提升 hard math 的推理质量。

### 方法的局限性
- **仅测试两种模型家族**：实验集中在 Qwen 和 DeepSeek，未覆盖更多架构。
- **全参数微调成本高**：每个配置只运行一次，缺乏多次随机种子验证。
- **依赖 LLM-as-a-judge**：虽经人工校验一致性高（>95%），但仍非完全替代人类标注。
- **可能过保守**：在 well-defined 问题上略降低作答率，需权衡安全与可用性。

### 未来工作方向
- 扩展至更多领域（如法律、金融）和多模态场景。
- 探索更低成本的适配方式（如 LoRA-based JTS）。
- 动态调整 abstention 阈值以适应不同应用场景。
- 结合外部知识检索，在可能时主动补全信息而非直接拒绝。

---

> 📌 **总结一句话**：  
> JTS 将“能否回答”的判断从隐式推理变为显式控制机制，首次系统性解决了 reasoning model 在信息不足时“明知故犯”的问题，为高风险场景下的可信 AI 部署提供了重要路径。

</details>

---

### 9. [Agentic Active Omni-Modal Perception for Multi-Hop Audio-Visual Reasoning](https://arxiv.org/abs/2605.28192)

**Authors**: Ke Xu, Yuhao Wang, Ziyang Cheng, Hongcheng Liu, Yanfeng Wang, Yu Wang  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.28192v1  

#### Abstract
Multi-hop audio-visual reasoning remains challenging for Omni-LLMs, as relevant evidence is often sparse, temporally dispersed, and distributed across both audio and visual streams. Existing benchmarks provide limited investigation of this setting, typically involving only a limited number of modali...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Agentic Active Omni-Modal Perception for Multi-Hop Audio-Visual Reasoning

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Omni-LLMs** 在处理 **multi-hop audio-visual reasoning**（多跳音视频推理）任务时面临显著挑战。这类任务要求模型整合在时间上分散、跨模态（音频与视觉）分布的稀疏证据。然而，现有方法通常采用被动、端到端的全视频编码方式，难以有效定位和集成这些分散的关键线索，导致推理性能低下。

此外，现有的视频理解基准（如 Video-MME、OmniVideoBench）任务构成复杂且异构，无法专门评估模型在**跨模态多跳推理**上的能力。

### 提出了什么新方法或新思路
为解决上述问题，本文提出两大核心贡献：

- **MOV-Bench**：一个全新的、精心构建的基准，专注于评估跨多个时间片段和模态的多跳音视频推理能力。该基准包含 **519 个高质量的多选题**，每个问题都需要连接多个时间点的音频和视觉证据，并覆盖五种推理类型（因果、指代、关系、假设、意图推理）。

- **AOP-Agent**：一种高效的 **agentic framework**（智能体框架），用于实现低资源条件下的主动全模态感知（active omni-modal perception）。该框架无需额外训练或依赖专有模型，即可赋能开源 Omni-LLMs 进行迭代式观察与推理。

### 相比现有方法的优势
- **无需训练与专有模型**：AOP-Agent 完全基于开源 Omni-LLMs 构建，通过设计机制降低主动感知的难度，避免了现有 agentic 方法对昂贵专有模型（如 GPT-4、Gemini）或大规模强化学习训练的依赖。
- **高效分层记忆结构**：引入 **hierarchical omni-modal memory**，将长视频组织为全局摘要、片段级描述、关键点（keypoints）等多层次表示，支持从粗到细的证据定位。
- **协作式观察-反思-重规划循环**：通过 **observe-reflect-replan loop**，由 Planner Agent 决定观察目标，Reflector Agent 评估证据充分性并反馈，使模型能动态调整策略，避免无效探索。
- **通用性强**：在 MOV-Bench 和 OmniVideoBench 上均取得一致提升，尤其在长视频和高推理强度问题上表现突出。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **MOV-Bench**（本文提出）：
  - 包含 519 个多跳音视频问答样本。
  - 平均视频时长 240.26 秒，其中超过 5 分钟的长视频占 36.42%。
  - 问题平均需要 **2.6 跳推理**，涵盖 2-hop（295）、3-hop（138）、4-hop（86）。
  - 视频类别覆盖 15 个现实场景（如烹饪、游戏、教育等）。
- **OmniVideoBench**（外部基准）：
  - 作为通用音视频理解基准，用于验证 AOP-Agent 的泛化能力。

### 实验设置和评估指标
- **评估指标**：准确率（Accuracy），按以下维度细分报告：
  - 推理类型：Causal, Referential, Hypothetical, Relational, Intent
  - 视频长度：Short (<150s), Medium (150–300s), Long (>300s)
  - 推理跳数：2-hop, 3-hop, 4-hop
- **模型输入**：仅提供原始视频和问题文本，不提供人工标注的剪辑或提示。
- **上下文长度**：最大 32,768 tokens，以支持长视频处理。

### 基线方法对比
- **直接推理基线（Direct Inference）**：
  - 将完整视频输入标准 Omni-LLM（如 Qwen3-Omni-Instruct）进行端到端推理。
- **其他 agentic 框架**：
  - **OmniAgent** (Tao et al., 2026)
  - **Active Video Perception** (Wang et al., 2025e)
  - 所有框架均使用 **Qwen3-Omni-Instruct** 作为 backbone 模型，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
| Model | Overall (MOV-Bench) | Long Video | 4-hop |
|-------|----------------------|------------|--------|
| Qwen3-Omni-Instruct (Direct) | 52.79% | 45.50% | 51.16% |
| **AOP-Agent (Qwen3-Omni-Instruct)** | **62.62%** | **60.85%** | **56.98%** |

- AOP-Agent 在 **long video 子集上提升达 +15.35%**（45.50% → 60.85%）。
- 在 **4-hop 高推理强度问题上提升 +5.82%**。
- 在 **所有推理类型上均有提升**，尤其在 Causal (+23.08%) 和 Intent (+15.00%) 上增益显著。

### 与其他 agentic 框架的对比（Table 1 & 2）
| Method | MOV-Bench (Overall) | MOV-Bench (Long) |
|--------|----------------------|------------------|
| Direct Inference | 52.79% | 45.50% |
| OmniAgent | 38.34% | 33.86% |
| Active Video Perception | 35.27% | 32.80% |
| **AOP-Agent (Ours)** | **62.62%** | **60.85%** |

> ⚠️ **关键发现**：现有 agentic 方法在开源模型上表现**不如直接推理**，而 AOP-Agent 显著优于两者。

### 消融实验结果（Ablation Study）

#### 组件消融（Table 3）
| 组件组合 | MOV-Bench (Long) | OmniVideoBench (Long) |
|---------|-------------------|------------------------|
| Reasoner Only (Baseline) | 45.50% | 28.45% |
| + Planner | 50.26% | 34.48% |
| **+ Planner + Reflector (Full)** | **60.85%** | **40.52%** |

- 加入 **Planner** 可提升约 5%，说明主动选择观察目标有益。
- 加入 **Reflector** 后带来最大增益（+10.59%），证明**迭代式反思**对稳定推理路径至关重要。

#### 模型分配影响（Table 4）
| Planner & Reflector | Reasoner | MOV-Bench (Long) |
|--------------------|----------|-------------------|
| Qwen2.5-Omni-7B | Qwen2.5-Omni-7B | 41.27% |
| Qwen3-Omni-30B | Qwen2.5-Omni-7B | 48.68% |
| Qwen2.5-Omni-7B | Qwen3-Omni-30B | 58.20% |
| **Qwen3-Omni-30B** | **Qwen3-Omni-30B** | **60.85%** |

- **Planner 和 Reflector 的质量对最终性能影响更大**，即使 Reasoner 较弱，强 Planner 也能带来显著提升。
- 表明 **高质量的 planning 与 reflection 是成功的关键前置步骤**。

#### 观察轮次影响（Figure 5）
- 性能在 **3 轮观察时达到峰值**。
- 超过 3 轮后性能趋于饱和甚至轻微下降，表明过多轮次可能引入噪声。
- 结论：**bounded observation**（有限轮次）更优，避免冗余探索。

---

## 4. 关键结论和发现

### 主要发现
1. **当前 Omni-LLMs 的瓶颈在于证据获取而非纯推理能力**：MOV-Bench 揭示，模型失败主因是无法有效定位和整合**时空分散的稀疏证据**，尤其是在长视频和多跳任务中。
2. **主动感知（active perception）是解决长视频多跳推理的有效范式**：相比被动处理，AOP-Agent 通过迭代观察显著提升了性能。
3. **现有 agentic 方法难以迁移到开源模型**：OmniAgent 和 Active Video Perception 在开源设置下表现差于直接推理，因其依赖强大的规划能力，而开源模型较弱。
4. **AOP-Agent 成功降低了主动感知的门槛**：通过分层记忆和多智能体协作，使开源模型也能实现高效、可靠的主动推理。

### 方法的局限性
1. **幻觉与错误传播风险**：若 hierarchical memory 中存在错误描述（如 Figure 8 所示），AOP-Agent 可能基于错误线索做出错误推断。
2. **推理开销增加**：multi-round observe-reflect-replan 循环带来额外的 inference overhead，延迟高于直接推理。
3. **离线内存构建**：当前的 omni-modal memory 依赖多阶段离线处理，**不支持流式输入**，限制其在实时场景（如智能眼镜、直播分析）的应用。
4. **数据规模受限**：MOV-Bench 因标注成本仅包含 519 个样本，未来需扩展更大规模数据集。

### 未来工作方向
- 开发更强的 **hallucination detection** 和 **evidence verification** 机制，提升系统鲁棒性。
- 设计 **streaming-friendly memory construction** 方法，支持实时增量更新。
- 探索 **轻量化 reflection 机制**，减少推理延迟。
- 构建更大规模、更多样化的 multi-hop audio-visual reasoning benchmark。
- 将 AOP-Agent 扩展至更多应用场景，如机器人交互、医疗视频分析等。

> ✅ **总体评价**：本文提出了一个**实用且可复现的 agentic 框架**，为开源社区提供了在低资源条件下实现高性能多跳音视频推理的新路径，推动了 **scalable long-video understanding** 的发展。

</details>

---

### 10. [Diffusion Large Language Models for Visual Speech Recognition](https://arxiv.org/abs/2605.28456)

**Authors**: Jeong Hun Yeo, Chae Won Kim, Hyeongseop Rha, Yong Man Ro  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.28456v1  

#### Abstract
Existing Visual Speech Recognition (VSR) systems commonly rely on left-to-right autoregressive decoding, which can force premature decisions on visually ambiguous tokens before sufficient context is available. We propose DLLM-VSR, to the best of our knowledge, the first Diffusion Large Language Mode...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Diffusion Large Language Models for Visual Speech Recognition**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **Visual Speech Recognition (VSR)** 系统普遍采用 **left-to-right autoregressive decoding**，这种固定顺序解码在视觉语音识别中存在显著缺陷：
- 视觉线索具有高度不确定性（如多个音素对应相同口型，即 **viseme ambiguity**）；
- 模型可能在上下文不足时过早做出决策，导致早期错误无法被后续上下文修正。

### **提出了什么新方法或新思路**
本文提出 **DLLM-VSR**，是首个基于 **Diffusion Large Language Model (DLLM)** 的 VSR 框架，其核心思想是将转录任务建模为 **iterative masked denoising** 过程，支持灵活的生成顺序。

#### **关键创新点包括：**
1. **Confidence-based unmasking**  
   - 高置信度位置优先解码，利用已确定的 token 作为双向上下文来消歧模糊 token。
   - 实现“先易后难”的渐进式推理，避免早期错误传播。

2. **Two-stage masked-denoising training strategy**  
   - **Stage 1**: 仅对 transcript 和 EOS 进行去噪训练，专注于内容对齐；
   - **Stage 2**: 引入 padding token，完成长度建模（length modeling），解决固定画布下的长度预测问题。

3. **Length-guided candidate decoding**  
   - 利用视频时长预测合理的目标长度范围 $ K_{\text{pred}} \pm R $；
   - 在多个长度假设下并行解码候选序列；
   - 使用联合重排序得分选择最终输出：
     $$
     s(k) = \sum \log c_i + \lambda \log p_k - \beta n_k
     $$
     （分别衡量解码置信度、长度合理性、迭代效率）

### **相比现有方法的优势**
| 维度 | 传统 AR 方法 | DLLM-VSR |
|------|-------------|----------|
| 解码顺序 | 固定左到右 | 灵活顺序，高置信优先 |
| 上下文利用 | 单向 | 双向 refinement |
| 长度建模 | 自动终止于 EOS | 显式长度预测 + 多假设搜索 |
| 错误纠正能力 | 不可逆 | 后续步骤可修正前期模糊项 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LRS3**（主基准）：433 小时 TED/TEDx 视频，用于训练与测试；
- **LRS2**：223 小时 BBC 节目，用于跨数据集泛化评估。

### **实验设置**
- **视觉编码器**：
  - 冻结的 **USR 2.0 Huge** 或 **AV-HuBERT**；
- **语言解码器**：
  - 冻结的 **Dream-7B**（DLLM）；
  - 使用 **LoRA** 微调所有线性层（rank=16, α=32）；
- **投影模块**：
  - Conv1d 下采样（25fps → 12.5fps）；
  - 两层 MLP 映射至 3584-dim LLM 空间；
- **画布长度**：$ T = 32 $；
- **训练策略**：
  - 两阶段训练：Stage 1（42k 步，lr=1e-4），Stage 2（4k 步，lr=5e-5）；
  - 使用 AdamW、bf16 混合精度、DeepSpeed ZeRO-2；
- **推理解码**：
  - Confidence threshold = 0.9；
  - 长度候选窗口 $ R = 5 $，最多 11 个候选；
  - 重排序权重通过验证集网格搜索确定。

### **评估指标**
- **Word Error Rate (WER)**：主要评价指标；
- **Real Time Factor (RTF)**：解码时间 / 视频时长，衡量效率；
- **Length predictor accuracy**：Acc@N, MAE。

### **基线方法对比**
- **Fully supervised**：Afouras et al. (2018), Ma et al. (2023)；
- **Self-supervised encoders**：AV-HuBERT, USR 2.0 + Transformer decoder；
- **LLM-augmented**：Cappellazzo et al. (2026) 使用 Qwen2.5-7B autoregressive LLM；
- **Oracle 设置**：已知真实长度，作为性能上界。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 方法 | 编码器 | 解码器 | 数据集 | WER (%) |
|------|--------|--------|--------|---------|
| Cappellazzo et al. (2026) | AV-HuBERT | Qwen2.5-7B (AR) | LRS3 | 24.9 |
| **Ours** | AV-HuBERT | Dream-7B (DLLM) | LRS3 | **21.9** |
| Reproduced baseline | USR 2.0 | Transformer | LRS3 | 25.8 |
| **Ours** | **USR 2.0** | **Dream-7B** | **LRS3** | **19.5** ✅ |
| Auto-AVSR (SOTA) | — | — | LRS3 | 19.1*（使用 3,448h 标注数据） |

> 💡 **DLLM-VSR 在仅使用 LRS3 433h 训练数据的情况下达到 19.5% WER，为当前 state-of-the-art**。

### **与基线方法的对比结果**
- 相比同编码器下的 AR-LM 方法（Cappellazzo et al., 2026）：
  - **降低 3.0% WER（24.9 → 21.9）**；
- 使用更强编码器 USR 2.0 后：
  - 比自实现 Transformer baseline（25.8%）**降低 6.3% WER**；
- 接近使用超大规模标注数据（3,448h）的 Auto-AVSR（19.1%），但训练数据量仅为其约 1/8。

### **消融实验结果**
#### **表3：两阶段训练与长度引导解码的影响（LRS3）**
| 解码策略 | AV-HuBERT WER (%) | USR 2.0 WER (%) |
|----------|-------------------|------------------|
| Stage-1 only + oracle length | 21.9 | 17.8 |
| Stage-1 only + implicit length | 188.0 ❌ | 275.0 ❌ |
| Implicit length (full training) | 23.1 | 20.5 |
| + Length-guided reranking | 22.7 | 20.2 |
| + Iteration penalty ($-\beta n_k$) | **21.9** | **19.5** |

> 🔍 **Stage 2 对稳定长度预测至关重要**；  
> 🔍 **Length-guided decoding 显著缩小与 oracle 的差距**。

#### **表2：生成顺序的影响**
| 方法 | AV-HuBERT WER (%) | USR 2.0 WER (%) |
|------|-------------------|------------------|
| AR (baseline) | 24.9 | — |
| AR + bidirectional attn | 24.3 | 22.2 |
| Block-wise (size=8) | 23.3 | 20.7 |
| **Full-parallel (confidence-based)** | **23.1** | **20.5** |

> ✅ **双向注意力 + 灵活解码顺序带来持续增益**。

#### **LRS2 泛化性能（Table 4）**
| 方法 | LRS2 WER (%) | RTF ↓ |
|------|--------------|-------|
| USR 2.0 + AR (beam=5) | 20.5 | 0.34 |
| DLLM-VSR (implicit length) | 17.8 | 0.14 |
| **DLLM-VSR (length-guided)** | **16.8** | 1.53 |

> ⚖️ **隐式长度解码速度快且准确；长度引导更准但代价高**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **灵活解码顺序优于固定左到右**  
   - Confidence-based unmasking 能有效利用高置信 token 提供双向上下文，缓解 viseme 模糊性；
   - Full-parallel 解码比 block-wise 更优。

2. **两阶段训练提升稳定性**  
   - 分离内容学习与长度建模可避免 padding token 主导损失函数；
   - Stage 2 对实际部署中的长度预测至关重要。

3. **视频时长是可靠的长度先验**  
   - 表5显示长度预测器 Acc@5 达 99.9%，MAE ≈ 0.7 token；
   - 支持局部窗口内多假设解码的设计合理性。

4. **Length-guided decoding 显著缩小与 oracle 的差距**  
   - Oracle WER (USR 2.0): 17.7% vs 实际 19.5%，仍有改进空间；
   - 多假设 + 重排序机制有效整合长度先验与解码置信度。

### **方法的局限性**
- **仍存在长度建模瓶颈**：即使使用 length-guided decoding，距离 oracle 性能仍有约 2% WER 差距；
- **推理成本较高**：length-guided decoding 需并行处理多个候选，RTF 从 0.14 升至 1.53；
- **高置信错误难以纠正**：若错误 token 本身语义合理（如 "harvested" vs "harvest"），模型仍可能保留。

### **未来工作方向**
1. 设计更高效的长度搜索策略（如 adaptive window）；
2. 探索动态画布大小或可变长度扩散机制；
3. 结合音频或其他模态进行多模态联合优化；
4. 降低 DLLM 推理延迟，推动实时应用落地。

---

> 📌 **代码开源地址**：[https://bit.ly/DLLM-VSR](https://bit.ly/DLLM-VSR)

</details>

---

### 11. [CaMBRAIN: Real-time, Continuous EEG Inference with Causal State Space Models](https://arxiv.org/abs/2605.28792)

**Authors**: Abhilash Durgam, Nyle Siddiqui, Jeffrey A. Chan-Santiago, Qiushi Fu, Elakkat D. Gireesh, Mubarak Shah  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.28792v1  

#### Abstract
Electroencephalography (EEG) is a critical, non-invasive method to monitor electrical brain activity. EEGs can span anywhere from a couple seconds to multiple hours, posing a major hurdle for existing deep learning methods due to two major factors: (1) existing EEG models are predominantly built upo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CaMBRAIN: Real-time, Continuous EEG Inference with Causal State Space Models**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统基于 **Transformer** 的 EEG 分析模型存在两大瓶颈：
- **计算复杂度高**：注意力机制导致时间复杂度为 $O(T^2)$，难以处理长达数小时的连续 EEG 信号。
- **滑动窗口处理方式**：将长序列切分为固定长度窗口进行独立推理，造成重复计算、上下文断裂，无法建模全局时序依赖。

此外，现有自监督预训练目标（如信号重建）侧重局部保真度，不利于隐藏状态长期记忆关键上下文。

### **提出了什么新方法或新思路**
提出 **CaMBRAIN** —— 首个面向实时、连续 EEG 推理的 **因果型、Mamba-based 状态空间模型（SSM）**，其核心创新如下：

#### ✅ **统一架构设计：因果流式推理**
- 采用 **单向、因果的 Mamba-3 架构**，以 **62.5ms 小块（patch）** 流式输入 EEG 数据。
- 维护一个 **持久化隐藏状态（persistent hidden state）**，逐步压缩历史信息，实现真正的“在线”推理，无需回看或重算。

#### ✅ **新型自监督训练框架：Representation-Level 学习**
提出两阶段预训练策略，替代传统的信号重建目标：
1. **Stage 1: 因果预测预训练（Causal Predictive Pretraining）**
   - 结合 **自回归预测（ARM）** 和 **掩码重建（Masked Reconstruction）**，建立基础时序理解。
2. **Stage 2: 无解码器潜在表示学习（Reconstruction-Free Latent Prediction）**
   - 引入 **JEPA-style 学生-教师框架**，学生模型在掩码输入下预测教师模型输出的潜在表示。
   - 包含两个子任务：
     - **掩码潜在预测（Masked Latent Prediction）**
     - **多步未来潜在预测（Multi-step Future Latent Prediction）**
   - 显式鼓励隐藏状态保留具有预测能力的长程上下文。

### **相比现有方法的优势**
| 维度 | CaMBRAIN | 传统方法（如 Transformer/Bi-LSTM） |
|------|---------|-------------------------------|
| **计算效率** | 线性时间复杂度 $O(T)$，每步常量计算 | 二次复杂度 $O(T^2)$ 或需滑窗重算 |
| **上下文建模** | 全局上下文持续累积，支持任意长度信号 | 局部窗口隔离，上下文受限 |
| **部署延迟** | 支持真正实时流式推理（real-time streaming） | 必须等待完整窗口，延迟高 |
| **内存占用** | 固定大小隐藏状态，不随时间增长 | 缓存整个窗口或 KV Cache，内存线性增长 |
| **训练目标对齐** | 表征级学习，强调长期可预测性 | 信号重建，偏向局部保真 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个多样化 EEG 基准上评估，涵盖多种临床任务：
| 数据集 | 任务 | 通道数 | 类别平衡 | 应用场景 |
|-------|------|--------|----------|----------|
| **TUAR** | 多类 Artifact 检测（6类） | 22 (bipolar) | 不平衡 | 脑电伪迹识别 |
| **TUAB** | 异常脑电检测（二分类） | 22 (bipolar) | ~50/50 | 临床异常筛查 |
| **MAT** | 心理压力检测（二分类） | 19 (10-20) | 平衡 | 认知状态监测 |
| **CHB-MIT** | 癫痫发作检测（二分类） | 16 (bipolar) | 极不平衡（0.42% 正例） | 癫痫预警系统 |

所有数据统一预处理：带通滤波（0.1–75 Hz）、陷波滤波（60 Hz）、重采样至 256 Hz，并使用 **因果滑动四分位归一化（RQN）** 进行标准化。

### **实验设置和评估指标**
- **预训练**：在 **TUEG (~21k 小时)** 上进行两阶段自监督预训练。
- **微调**：端到端微调整个模型用于下游任务。
- **评估频率**：以 **16 Hz（62.5ms/patch）** 输出逐块预测，模拟真实流式场景。
- **主要指标**：
  - **AUROC**（Area Under ROC Curve）
  - **AUPR / AUC-PR**（Precision-Recall 曲线下面积）
  - **Balanced Accuracy (BAC)**
  - **GFLOPs/s**（持续计算负载，衡量效率）

### **基线方法对比**
涵盖主流监督与自监督 EEG 模型：
- **监督模型**：EEGNet, EEG-GNN, ST-Transformer
- **自监督模型**：
  - BENDR, BIOT, LaBraM
  - CBraMod, LUNA, REVE
  - EEGFormer, GraphS4mer

特别关注是否支持 **流式推理** 及其 **计算开销**。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **在多个任务上达到 SOTA 性能**
| 数据集 | 指标 | CaMBRAIN | 最佳基线 | 提升 |
|-------|------|----------|----------|------|
| **TUAR** | AUROC | **0.936** | LUNA-H (0.921) | +1.5% |
| **CHB-MIT** | AUROC | **0.921** | REVE (0.908) | +1.3% |
| **MAT** | AUROC | **0.876** | REVE-B (0.845) | +3.1% |
| **MAT** | Bal. Acc. | **0.778** | REVE-B (0.766) | +1.2% |

> 注：CaMBRAIN 参数量仅为 37M，远小于部分基线（如 REVE 72M, CEReBrO 85M），但仍表现更优。

#### ✅ **显著提升计算效率**
| 模型 | GFLOPs/s（越低越好） | 相对节省 |
|------|---------------------|-----------|
| **CaMBRAIN** | **1.23** | — |
| CBraMod | 7.51 | ×6.1 |
| REVE | 194.79 | ×158 |

👉 在保持甚至超越性能的同时，**计算负载降低超过 10 倍以上**，是首个能在资源受限设备上运行的高效流式 EEG 模型。

### **与基线方法的对比结果**
- 在 **CHB-MIT** 上，尽管事件稀疏且不平衡，CaMBRAIN 凭借长期上下文整合能力，在 **发作前概率上升更快、更稳定**。
- 在 **MAT** 上，优于更大规模的基础模型（如 REVE），说明表征质量更高。
- 所有对比模型均采用 **滑动窗口 + 重叠计算**，而 CaMBRAIN 完全避免冗余计算。

### **消融实验结果**

#### 🔹 **Ablation on Hidden State（表 3a）**
比较“持续状态” vs “每 5 秒重置状态”：
| 指标 | 持续状态 | 窗口式重置 | 差异 |
|------|----------|------------|------|
| AUROC（稀疏分类） | 88.9 | 87.4 | **+1.5** |
| 发作时概率峰值 | 38.3 | 27.6 | **+10.8** |
| 发作块内概率 | 21.6 | 6.2 | **+15.4** |

✅ **结论**：持久隐藏状态显著增强对罕见事件（如癫痫）的检测能力和鲁棒性。

#### 🔹 **Ablation on Pretraining Stages（表 3b）**
| 阶段 | AUROC | AUPR | 相对于随机初始化提升 |
|------|------|------|------------------|
| 仅微调（Random Init） | 93.7 | 75.0 | — |
| + Stage 1（ARM + Mask Rec） | 94.3 (+0.6) | 77.8 (+2.8) | 有限提升 |
| + Stage 1 + Stage 2（JEPA） | **96.0 (+2.3)** | **85.9 (+10.9)** | 显著跃升 |

✅ **结论**：Stage 2 的表征级预测训练对性能提升起决定性作用。

---

## **4. 关键结论和发现**

### **主要发现**
1. **EEG 天然是因果信号**，应采用 **单向流式建模**，而非双向 Transformer。
2. **隐藏状态即记忆**：通过 SSM 架构维护的 **persistent hidden state** 是实现长程上下文建模的关键。
3. **训练目标必须匹配推理需求**：传统“重建”目标不适合流式场景；**JEPA-style 表征预测** 更有利于学习可传播的语义信息。
4. **效率与性能可兼得**：CaMBRAIN 在大幅提升吞吐量（>10×）的同时，仍取得 SOTA 性能。

### **方法的局限性**
1. **在 TUAB 上未明显领先最强基线**（如 CBraMod），表明在某些任务上仍有追赶空间。
2. **预训练数据量相对较小**（~21k 小时 TUEG），远低于 REVE 等大规模跨中心数据集。
3. 当前模型尚未探索 **多模态融合**（如结合 fMRI 或行为标签）。

### **未来工作方向**
1. **扩展预训练规模**：在更大、更多样化的 EEG corpus 上训练，提升泛化能力。
2. **适配更多采集范式**：支持非标准导联布局、干电极等可穿戴设备场景。
3. **嵌入式部署优化**：进一步压缩模型，实现在边缘设备（如头戴式 EEG 设备）上的实时运行。
4. **引入外部先验知识**：结合神经科学知识指导表征学习（如特定频段建模）。

---

> 📌 **一句话总结**：  
> **CaMBRAIN 是首个真正意义上支持实时、连续、高效 EEG 推理的深度模型，通过“因果 SSM 架构 + 表征级自监督训练”的协同设计，打破了传统滑窗与注意力机制的桎梏，为临床床旁监测与可穿戴脑机接口提供了全新解决方案。**

</details>

---

### 12. [SiDP: Memory-Efficient Data Parallelism for Offline LLM Inference](https://arxiv.org/abs/2605.28095)

**Authors**: Alan Zhao, Cyril Y. He  
**Category**: cs.DC  
**Published**: 2026-05-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.28095v1  

#### Abstract
The rapid adoption of large language models (LLMs) has shifted a substantial portion of inference workloads into throughput-oriented offline regimes, where fully utilizing GPU compute requires large batch sizes. However, existing deployments face a structural tension. Data parallelism (DP) scales th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SiDP: Memory-Efficient Data Parallelism for Offline LLM Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在离线大语言模型（LLM）推理中，**吞吐量（throughput）是核心性能指标**。为了最大化吞吐，系统需要尽可能增大批处理大小（batch size, B），以使 GPU 进入计算饱和的“高效”区域。然而，**GPU 显存容量（HBM）成为瓶颈**，因为：
- **静态权重（weights）** 在 Data Parallelism（DP）下被完全复制到每个 GPU 上，占用大量显存；
- **动态 Key-Value（KV）缓存** 随着 batch size 和序列长度增长而线性增加。

这导致即使计算资源未饱和，也无法进一步扩大 batch size，从而限制了吞吐量提升。

传统方案如 Model Parallelism（MP）虽能减少每卡权重，但引入细粒度同步，破坏了 DP 的独立性和调度灵活性。

---

### 提出了什么新方法或新思路
论文提出 **SiDP（Shared-weight Intra-node Data Parallelism）**，一种内存高效的 DP 范式，其核心思想是：

> **将模型权重视为由带宽支撑的共享资源，在 DP 组内构建一个分布式权重池**。

具体设计包括：
- **分层所有权（Layer Ownership）**：每个 Transformer 层的 FFN 权重仅由一个 GPU 拥有（owner），其他副本按需访问。
- **双模式执行机制**：
  - **Weight-as-a-Service (WaS)**：在大 batch 场景下，非拥有者通过 NVLink 异步预取远程权重，在本地执行计算，保持 DP 独立性。
  - **Compute-as-a-Service (CaS)**：在小 batch 尾部阶段，非拥有者将激活值发送给权重所有者，由其统一计算并返回结果，避免为少量计算频繁拉取权重。
- **异步缓存管理与峰值偏移（Peak Shifting）**：通过预分配缓存槽位、后台线程管理、以及错开各 rank 的权重拉取顺序，缓解 NVLink 冲突。

---

### 相比现有方法的优势
| 方面 | SiDP | 传统 DP | FSDP / ZeRO |
|------|------|--------|-------------|
| 显存效率 | ✅ 大幅减少冗余权重 | ❌ 完全复制权重 | ✅ 分片权重 |
| 计算独立性 | ✅ 保留 DP 独立性（WaS） | ✅ 完全独立 | ❌ 需要同步 all-gather |
| 通信开销 | ✅ 可重叠、轻量级 | ✅ 无额外通信 | ❌ 固定高开销 all-gather |
| 适用场景 | ✅ 特别适合离线吞吐优先任务 | ✅ 通用 | ⚠️ 更适合训练 |

> **SiDP 在不牺牲 DP 可扩展性的前提下，实现了接近 MP 的显存效率**。

---

## 2. 核心实验方法和设置

### 使用了哪些模型（非数据集）
由于是**推理系统优化**研究，实验基于以下主流 LLM 模型进行：
- **Qwen3-32B**
- **Llama-3.1-70B**
- **Qwen2.5-72B**

这些均为 dense Transformer 架构，适用于典型的离线任务（如评测、日志处理、合成数据生成等）。

---

### 实验设置和评估指标

#### 硬件平台（见 Table 1）
| 节点类型 | GPU 数量 | 单卡显存 | 互联技术 |
|---------|----------|-----------|------------|
| H20     | 8× H20   | 144 GB    | NVLink 4   |
| H200    | 8× H200  | 144 GB    | NVLink 4   |
| B200    | 8× B200  | 180 GB    | NVLink 5   |

#### 并行配置
- 对比多种并行策略：
  - Pure TP（Tensor Parallelism）
  - TP + DP
  - TP + PP（Pipeline Parallelism）
  - SiDP（默认使用 DP=8）
- 所有方法均集成于 **vLLM** 框架之上。

#### 工作负载（Workload）
模拟典型离线场景，固定序列长度：
- **Summarization**: S = 1K
- **Code Generation**: S = 2K
- **Conversation Evaluation**: S = 4K

采用两种 batching 策略并报告最优结果：
1. 固定大 batch size
2. 自适应 batch size（受限于显存）

#### 评估指标
- **可用 KV 缓存容量（in tokens）**
- **端到端吞吐量（tokens/sec）**
- **Per-iteration decode time**
- 消融实验中的模块影响分析

---

### 基线方法对比
- **vLLM-TP**：纯张量并行
- **vLLM-TP+DP**：主流的混合并行方案
- **vLLM-TP+PP**：管道并行作为补充对比
- **FSDP-style 实现**：用于消融实验，验证设计必要性

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ KV Cache 容量提升
- 在相同配置下，SiDP **最多将可用 KV 容量提高至 1.8×**。
- 图 5 显示，在 TP=1, DP=8 设置下：
  - vLLM **无法运行 Llama-3.1-70B 或 Qwen2.5-72B**（显存不足）；
  - SiDP 仍可支持约 **1.0M KV tokens**。

#### ✅ 吞吐量显著提升
- 在 H20、H200、B200 上，SiDP 相比 vLLM-TP+DP 基线实现 **最高达 1.5× 的端到端吞吐提升**。
- 提升随序列长度增加而增强：
  - S=1K：接近基线（已 compute-bound）
  - S=4K：优势明显（memory-bound 场景）

> 如图 6–8 所示，SiDP 在长上下文、高并行度场景下始终优于任何 vLLM 支持的 TP/PP/DP 组合。

---

### 与基线方法的对比结果

| 指标 | 结果 |
|------|------|
| **最大 KV 容量** | ↑ 最多 1.8× |
| **端到端吞吐** | ↑ 最多 1.5× |
| **能否运行更大模型** | ✅ 可在 DP=8 下运行原不可行的大模型 |
| **小 batch 尾部性能** | ✅ CaS 模式有效缓解尾延迟 |

> 特别是在 `TP=2, DP=4` 和 `TP=1, DP=8` 配置下，收益最大，说明 **权重冗余越严重，SiDP 收益越大**。

---

### 消融实验结果

#### ✅ Peak Shifting 至关重要
- 图 10 显示，在 DP=8 时，**启用 peak shifting 可使吞吐提升 3.4×**。
- 不启用会导致多个 rank 同时从同一 owner 拉取权重，造成 NVLink incast 拥塞。

#### ✅ Mode Switching 提升整体效率
- 图 13 显示：
  - 仅用 WaS：吞吐提升 ~7–9%
  - 加入 CaS 动态切换：吞吐提升 **27–32%**
- 表明 **CaS 对尾部阶段优化至关重要**。

#### ✅ CaS 各组件逐步优化效果显著
- 图 14 对比不同版本：
  - FSDP-style all-gather：33s（极慢）
  - CaS V1（异步 P2P）：25s（↓24%）
  - CaS V2（GEMM fusion）：19s（↓24%）
  - CaS V3（skip dummy）：12s（↓37%）
- 总计提速 **2.8×**，证明各设计细节不可或缺。

#### ✅ SiDP 在短上下文也有效
- 图 12 显示，在显存利用率仅为 0.6 的常见部署条件下（≈ A100/H100），SiDP 仍可带来 **24–51% 吞吐提升**。

---

## 4. 关键结论和发现

### 主要发现
1. **权重冗余是离线 LLM 推理中显存浪费的主要来源**，尤其在高 DP 度下。
2. **利用高速互联（NVLink）将权重转为共享资源是可行且高效的**。
3. **WaS + CaS 双模式设计可在不同 batch regime 下自动选择最优路径**：
   - WaS 适用于主体大 batch 阶段；
   - CaS 是尾部的小而关键的安全网。
4. **SiDP 在不修改模型架构的前提下，提升了系统的“有效 batch size”上限**，从而进入更高吞吐的操作区间。

---

### 方法的局限性
1. **依赖 NVLink 等高带宽低延迟互联**：
   - 在 PCIe 或以太网连接的集群中可能失效。
2. **节点内机制，不解决跨节点问题**：
   - 虽可扩展至多节点（通过复制 SiDP 组），但未优化跨节点通信。
3. **目前仅支持 dense 模型**：
   - MoE 架构尚未适配，尽管作者指出有潜力结合 EPLB。
4. **故障域变大**：
   - 类似 TP/PP，任一 GPU 故障可能导致整个组失效，削弱了 DP 原有的容错优势。

---

### 未来工作方向
1. **扩展至异构集群与低速互联环境**：
   - 动态调整 WaS/CaS 切换阈值，适配不同网络条件。
2. **支持 MoE 架构**：
   - 在 Expert-Parallel 组内应用 SiDP 减少专家权重冗余。
3. **与跨节点调度协同设计**：
   - 构建全局感知的参数服务层。
4. **探索更智能的缓存替换与预取策略**：
   - 基于 workload pattern 学习 prefetching policy。

---

> **总结**：  
> SiDP 提出了一种优雅的“资源套利”思想——**用闲置的带宽换取宝贵的 HBM**，在保持 DP 简洁性的同时突破显存墙，为离线 LLM 推理提供了新的系统设计范式。其实验充分、工程扎实，具有很强的实际部署价值。

</details>

---

### 13. [ZipRL: Adaptive Multi-Turn Context Compression with Hindsight Response Replay](https://arxiv.org/abs/2605.28069)

**Authors**: Zhexin Hu, Li Wang, Xiaohan Wang, Jiajun Chai, Xiaojun Guo, Wei Lin, Guojun Yin  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.28069v1  

#### Abstract
Adaptive context compression is vital for scaling Large Language Models (LLMs) to complex, multi-turn agent tasks. However, rule-based compression methods may discard task-critical nuances, while Reinforcement Learning (RL) approaches usually struggle to balance information retention and token effic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《ZipRL: Adaptive Multi-Turn Context Compression with Hindsight Response Replay》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前在基于 **Large Language Models (LLMs)** 的多轮代理任务中，**context window** 有限，而检索到的文档和交互历史会迅速累积，导致上下文过载。现有的 **context compression** 方法面临两大挑战：

1. **Uniform Processing Granularity**：大多数方法对所有文档采用相同的压缩粒度，忽略了不同文档与查询的相关性差异，可能导致关键信息丢失或无关噪声保留。
2. **Sparse Rewards in RL**：强化学习（RL）训练中，奖励信号稀疏（通常只在最终任务成功时给出），难以有效指导中间步骤的压缩策略优化。

---

### **提出的新方法与创新思路**
为解决上述问题，论文提出了 **ZipRL**，一个专为 **Reinforcement Learning from Verifiable Rewards (RLVR)** 设计的自适应多轮上下文压缩框架，其核心创新包括：

#### **(1) Multi-Granularity Compression Mechanism**
- 引入**多粒度压缩机制**，允许模型根据文档与查询的**相关性动态选择压缩级别**（共5级：从 `Ultra-coarse` 到 `Ultra-fine`）。
- 高相关性文档分配更高压缩级别（保留更多细节），低相关性文档则进行粗粒度压缩以减少噪声。
- 通过 **in-context prompting** 实现主动感知和决策。

#### **(2) Hindsight Response Replay (HRR)**
- 受 **Hindsight Experience Replay (HER)** 启发，提出 **HRR**，用于在 RL 训练中**稠密化奖励信号**。
- 不依赖外部 **Process Reward Models (PRMs)**，而是利用一个**启发式压缩质量评分函数** $ Q_{\text{com}} $，将轨迹级优势（advantage）按每轮压缩质量重新分配。
- 实现方式：  
  $$
  A^{(i,j)} = A^{\text{GRPO}} + w \cdot (Q_{\text{com}}^{(i,j)} - \bar{Q}_{\text{com}})
  $$
  其中 $ w $ 是 reshaping 系数，$ \bar{Q}_{\text{com}} $ 是该轨迹的平均压缩质量。

#### **(3) Compression Quality Scoring Function $ Q_{\text{com}} $**
设计了一个多维度评分函数，综合衡量压缩质量：
- **$ Q_{\text{ratio}} $**：长度匹配度（软惩罚机制）
- **$ Q_{\text{level}} $**：压缩级别一致性（如句子密度是否符合预期）
- **$ Q_{\text{info}} $**：信息保留度（关键词覆盖 + 内容保留）
- **$ Q_{\text{sem}} $**：语义完整性（句法、标点、连贯性）

最终得分：
$$
Q_{\text{com}} = 0.3 Q_{\text{ratio}} + 0.1 Q_{\text{level}} + 0.4 Q_{\text{info}} + 0.2 Q_{\text{sem}}
$$

---

### **相比现有方法的优势**
| 维度 | ZipRL | 现有方法（如 ASearcher, AgentFold） |
|------|-------|-------------------------------|
| **压缩粒度** | 自适应、多粒度 | 固定或单粒度 |
| **RL 优化信号** | 稠密化（via HRR） | 稀疏（仅最终奖励） |
| **压缩质量评估** | 多维启发式评分，无需外部 PRM | 依赖人工设计或外部模型 |
| **长程任务鲁棒性** | 支持高达 256 轮外推测试 | 在长轮次下性能饱和或下降 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Multi-hop QA**：
  - **MusiQue**, **SQuAD**, **Frames**, **Bamboogle**
- **Web Browsing**：
  - **BrowseComp (BC-Plus)**：动态网页检索基准

### **实验设置**
- **模型**：基于 **Qwen2.5** 和 **Qwen3** 系列（3B, 4B, 7B, 8B, 14B, 32B）
- **训练**：
  - 先进行 **Cold Start SFT**（使用 GPT-4o 合成的 1,155 条高质量轨迹）
  - 再进行 **GRPO + HRR** 的 RL 微调
  - 最大训练轮次：20 轮
- **评估指标**：
  - **EM (Exact Match)** 和 **F1**
  - **Token Efficiency**（token 使用量）
  - **Long-Horizon Extrapolation**：测试至 256 轮

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **ReAct-based** | Qwen3-235B-ReAct, Gemini-3-Pro-ReAct |
| **Summary-only** | GPT-4o-Summary, LongCat-Flash-Summary |
| **专用搜索代理** | ASearcher, AgentFold, NestBrowse, WebSailor |
| **内存管理方法** | MemAgent, MEM1 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模型 | 平均 EM | 平均 F1 |
|------|--------|--------|
| **ZipRL-8B** | **30.3%** | **41.2%** |
| Qwen3-235B-ReAct | 20.9% | 32.4% |
| ASearcher-7B | 22.5% | 32.7% |
| AgentFold-8B | 22.5% | 31.9% |

> ✅ **ZipRL-8B 以 29× 更少参数超越 Qwen3-235B-ReAct**

---

### **与基线方法的对比结果**
- 在 **Qwen3-4B** 和 **Qwen3-8B** 上，ZipRL 分别比最强基线提升：
  - **+27.9%** 和 **+34.7%** 的平均 EM
- 在 **极端 256 轮外推测试**中：
  - ZipRL-8B 持续提升性能，而 Qwen3-235B-ReAct 在 16 轮后即饱和
  - 显示出卓越的**长程推理鲁棒性**

---

### **消融实验结果（Ablation Study）**
使用 **Qwen3-8B** 进行消融，结果如下：

| 变体 | 平均 EM | 相比完整版下降 |
|------|--------|----------------|
| **ZipRL (完整)** | 30.3% | — |
| w/o RL | 26.4% | ↓ 3.9% |
| w/o Level 2 & 4 | 28.9% | ↓ 1.4% |
| w/o $ Q_{\text{info}} $ | 26.9% | ↓ 3.4% |
| w/o $ Q_{\text{sem}} $ | 27.7% | ↓ 2.6% |

> 🔍 结论：
> - **RL 训练至关重要**（↓3.9%）
> - **多粒度机制有效**（移除两级导致性能下降）
> - **信息保留（$ Q_{\text{info}} $）和语义完整性（$ Q_{\text{sem}} $）是关键**

---

## **4. 关键结论和发现**

### **主要发现**
1. **多粒度压缩优于均匀压缩**：理论证明（Theorem 4.1）和实验验证均表明，在相同资源预算下，**相关性感知的资源分配能显著提升任务效用**。
2. **HRR 有效缓解稀疏奖励问题**：通过压缩质量评分实现优势重塑，使模型能从非最优轨迹中学到有用行为。
3. **ZipRL 具备强泛化能力**：在多种模型规模和任务上均表现领先，且在 256 轮极限测试中仍持续提升。
4. **高 token 效率**：相比 ReAct 方法，ZipRL 在保持甚至超越性能的同时，显著降低 token 消耗。

---

### **局限性**
1. **语言依赖**：$ Q_{\text{info}} $ 依赖英文停用词表，在多语言或专业领域（如法律、代码）可能失效。
2. **可信度未建模**：$ Q_{\text{com}} $ 忽略文档可信度，在完全对抗性检索下性能骤降 85–99%。
3. **冷启动依赖单一 QA 数据集**：可能限制在结构化或代码类任务中的迁移能力。

---

### **未来工作方向**
- 扩展 $ Q_{\text{com}} $ 以支持**多语言和跨领域压缩评估**
- 引入**可信度估计模块**，增强对噪声和对抗性检索的鲁棒性
- 探索更复杂的**分层 RL 架构**，进一步优化长程信用分配
- 将 ZipRL 思路应用于 **multi-agent collaboration** 和 **tool learning** 场景

---

> 📌 **代码已开源**：https://github.com/huzhexin/ZipRL

</details>

---

### 14. [CIVIC: End-to-End Sequence Compactness for Efficient Vision-Language Models](https://arxiv.org/abs/2605.28115)

**Authors**: Fengze Yang, Bo Yu, Xuewen Luo, Cathy Liu, Chenxi Liu  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.28115v1  

#### Abstract
Vision-Language Models (VLMs) face severe memory and latency bottlenecks due to high-resolution visual tokens. While current token reduction methods theoretically save FLOPs, post-hoc pruning introduces structural overhead, failing to yield proportional wall-clock acceleration. However, enforcing a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CIVIC: End-to-End Sequence Compactness for Efficient Vision-Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Vision-Language Models (VLMs)** 在处理高分辨率图像或长视频时会产生大量视觉 token，导致：
- **计算开销大**（FLOPs 高）
- **KV Cache 内存占用高**
- **推理延迟严重**

尽管已有许多 token 压缩方法（如 token pruning、merging）在理论上减少了 FLOPs，但由于以下原因，**实际硬件加速效果不明显**：
- 后处理压缩（post-hoc pruning）引入额外操作（如 scoring、routing、gather/scatter），造成运行时开销；
- 压缩后的紧凑表示常需恢复为密集格式以兼容下游模块（如 LLM prefill），破坏端到端效率；
- 非连续内存访问模式降低 GPU 利用率。

这种“**理论压缩 vs 实际加速脱节**”的现象被称为 **compression-realization gap**。

---

### 🚀 提出的新方法：CIVIC
本文提出 **CIVIC**（Compact Inference for Vision-Language Integrated Compression），一种**路径一致的端到端紧凑推理框架**，其核心思想是：
> 将紧凑序列作为从视觉输入到语言模型生成全过程中的**主推理路径**，避免任何中间恢复或动态路由。

#### 主要技术创新点：
1. **Anchor-Based Compact Visual Aggregation**
   - 在视觉编码前，通过可学习的 anchor 向量对 dense patch embeddings 进行聚合；
   - 输出为连续、紧凑的 token 序列，保持空间映射关系，防止几何失真。

2. **KV-Compressed Attention in Vision Encoder**
   - 在视觉 Transformer 中压缩 Key 和 Value 到固定数量的 memory anchors；
   - 显著减少 self-attention 的内存交互复杂度（从 $ M_e \times M_e $ → $ M_e \times S $）。

3. **Adaptive Spatial Retention Floor**
   - 设置最小保留比例（min keep ratio），确保细粒度定位信息不被过度压缩丢失；
   - 平衡效率与感知精度。

4. **Text-Aligned KL Distillation**
   - 由于原始 LLM 接受的是 dense visual placeholder，而 CIVIC 输入的是 compact tokens；
   - 提出基于文本位置对齐的 KL 散度蒸馏策略，仅在非视觉 token 上进行 logit 对齐，绕过结构不匹配问题；
   - 不修改预训练 LLM 参数，实现高效迁移。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 DyMU, DynamicViT） | CIVIC |
|------|-------------------------------|-------|
| 压缩时机 | Post-hoc（编码后） | Pre-encoding（编码前） |
| 路径一致性 | ❌ 中间压缩 + 下游恢复 | ✅ 全流程保持 compact 表示 |
| 内存访问 | ❌ 非连续 gather/scatter | ✅ 连续内存布局 |
| 训练兼容性 | ❌ 需复杂适配机制 | ✅ 文本对齐蒸馏，无需修改 LLM |
| 实际加速 | ⚠️ 理论 FLOP 下降 ≠ 延迟下降 | ✅ FLOP 下降直接转化为 wall-clock 加速 |

> 💡 **核心优势总结**：CIVIC 实现了真正的“**理论压缩 → 物理加速**”转化，解决了现有方法中存在的“**虚假效率**”问题。

---

## 2. 核心实验方法和设置

### 📚 数据集
在多个多模态基准上评估，涵盖不同任务类型：
- **Reasoning**：MMMU, MathVision
- **Perception & Localization**：ODinW-13, RealWorldQA (RWQA)
- **Sequential Context Tracking**：VideoMME short split

这些任务覆盖了视觉理解、数学推理、目标检测、常识问答和视频时序建模能力。

---

### ⚙️ 实验设置
- **基础模型**：Qwen3-VL-2B-Instruct（开源版本）
- **硬件平台**：单张 NVIDIA RTX 4090 GPU（强调资源受限环境下的实用性）
- **评估方式**：
  - 所有生成过程使用确定性参数（`T=0`, `top-k=1`, `top-p=1`），保证 timing 可复现；
  - 使用 PyTorch + Transformers 默认配置。

---

### 📊 评估指标
分为两类：

#### （1）性能指标（Accuracy）
- 多项任务平均得分（归一化于 baseline）

#### （2）系统效率指标（Efficiency）
| 指标 | 描述 |
|------|------|
| `Total Latency` | 端到端推理时间（ms） |
| `Prefill Tokens` | 输入 LLM 的视觉 token 数量 |
| `KV Cache Memory` | 占用显存大小（MB） |
| `Overhead` | 压缩逻辑引入的额外开销（ms） |
| `Throughput` | 单位时间内处理样本数 |

---

### 🆚 基线方法对比
共比较五种主流 token reduction 方法：
1. **DyMU**：动态合并 + 虚拟解合并
2. **DiffRate**：可微分压缩率控制
3. **DynamicViT**（hard/soft）：基于重要性评分的 token drop
4. **VisionTrim**：统一视觉 token 压缩
5. **ZOO-Prune**：零阶梯度估计剪枝

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| 模型 | 总延迟 (ms) | Prefill Tokens | KV Cache (MB) | Overhead (ms) |
|------|-------------|----------------|---------------|----------------|
| Baseline | 3543.0 | 1122.2 | 122.7 | 0.00 |
| DyMU | 3688.2 | 1122.2 | 122.7 | 9.28 |
| DiffRate | 3804.9 | 1122.2 | 122.7 | 1.26 |
| VisionTrim | 4207.5 | 848.5 | 92.81 | 3.96 |
| ZOO-Prune | 3564.3 | 848.5 | 92.81 | 18.45 |
| **CIVIC (Ours)** | **2514.9** | **407.9** | **44.61** | **0.49** |

> ✅ **关键提升**：
- **KV Cache 减少至约 1/3**（122.7 → 44.61 MB）
- **Prefill token 减少超过 60%**
- **总延迟降低 ~29%**（3543 → 2515 ms）
- **压缩开销极低**（仅 0.49ms）

---

### 📊 与其他方法对比结果
- **几乎所有基线方法反而增加了总延迟**（如 DyMU ↑3688ms），说明其“理论压缩”未带来实际收益；
- 尽管 VisionTrim 和 ZOO-Prune 成功减少了视觉 token 数量，但因未改变 LLM 输入长度，**无法缓解 decoding 阶段的注意力负担**；
- CIVIC 是唯一一个将压缩传播到 **LLM prefill 和 KV-cache** 层面的方法，从而显著缩短 decoding 时间（3128 → 2229 ms）。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）Compact Token Budget（C）
| C 值 | 总延迟 (ms) | Decode Time (ms) |
|------|------------|------------------|
| 64 | 2823.7 | 2514.4 |
| 256 | 2440.8 | 2151.3 |
| 512 | 2327.6 | 2038.0 |

> ⚠️ 极端压缩（C=64）虽减少 prefill，但导致 decode 时间上升 —— **信息不足迫使模型反复猜测上下文**。

#### （2）Minimum Keep Ratio（Min）
| Min | 总延迟 (ms) | 定位性能 |
|-----|------------|----------|
| 0.0 | 2591.8 | 下降明显 |
| 0.2 | 2635.8 | 略升 |
| 0.5 | 2616.6 | 最佳平衡 |

> ✅ 存在一个最优保留阈值（约 0.2–0.5），用于保护关键细节区域。

#### （3）KV Compression Anchors 数量
| Anchors | 总延迟 (ms) |
|--------|------------|
| None | 2598.9 |
| 128 | 2543.4 |
| 512 | 2660.3 |

> ✅ 使用少量 anchors（如 128）即可有效压缩 KV，过多反而增加计算负担。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **理论 FLOP 下降 ≠ 实际加速**  
   当前多数 token reduction 方法因引入 runtime overhead（如 scoring、routing、restore）而无法实现物理加速。

2. **端到端路径一致性至关重要**  
   只有当 compact 表示贯穿整个 pipeline（vision encoder → projector → LLM prefill → KV-cache），才能真正释放硬件效率潜力。

3. **KV Cache 是瓶颈的关键载体**  
   减少视觉 token 若不能缩小 LLM 的 context length，则无法减轻 autoregressive decoding 的 attention 开销。

4. **CIVIC 实现了“无损换高效”**  
   - 在所有 benchmark 上性能持平甚至略优（尤其 ODinW 定位任务）；
   - 显著降低 latency 与 memory footprint；
   - 开销几乎可以忽略（<0.5ms）。

---

### ⚠️ 方法的局限性
- **静态 token 预算**：当前采用固定压缩率，尚未支持 instance-adaptive 动态调整；
- **单图限制**：目前实验集中在 single-image 场景，未验证 multi-image 或 long video 流；
- **架构依赖性**：虽然设计通用，但具体实现仍基于 Qwen3-VL 架构，跨模型泛化需进一步验证。

---

### 🔮 未来工作方向
1. 扩展至 **dynamic, instance-adaptive token budget** 控制；
2. 支持 **multi-image 输入** 和 **long-form video streaming** 场景；
3. 探索更高效的 anchor 更新机制与轻量化 aggregation 设计；
4. 结合 on-device safety mechanisms，推动边缘部署的安全高效推理。

---

> 🧩 **一句话总结**：  
> CIVIC 通过构建一条**全程紧凑、路径一致的视觉推理通路**，成功打通了从“理论压缩”到“真实加速”的最后一公里，为高效 VLM 部署提供了新的范式。

</details>

---

### 15. [The Missing Piece in Pre-trained Model Evaluation: Reward-Guided Decoding Unlocks Task-Oriented Behavior Without Parameter Updates](https://arxiv.org/abs/2605.28020)

**Authors**: Shaobo Wang, Guo Chen, Ziyue Wang, Zhengyang Tang, Qingyang Liu, Xingzhang Ren, Dayiheng Liu, Linfeng Zhang  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.28020v1  

#### Abstract
With the rapid progress of large language models (LLMs), reliably evaluating the capabilities of pre-trained LLMs has become increasingly important. The challenge is that base pre-trained models are optimized for next-token prediction and often fail to follow instructions or produce well-formed answ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：The Missing Piece in Pre-trained Model Evaluation: Reward-Guided Decoding Unlocks Task-Oriented Behavior Without Parameter Updates

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前对 **pre-trained LLMs** 的评估存在严重偏差。标准的 **direct decoding** 和提示工程往往无法有效激发模型的任务导向行为（task-oriented behavior），因为预训练模型本质上是为 **next-token prediction** 优化的，倾向于生成文本延续（continuation）而非遵循指令、给出结构化答案。

这导致在 **AlpacaEval2.0** 等任务导向型基准上的表现被低估，而这种“失败”可能并非源于模型缺乏知识，而是 **decoding-induced failure**（解码导致的失败）。因此，现有评估混淆了模型能力与解码策略的有效性。

### 提出了什么新方法或新思路
本文提出了 **Energy-Based Decoding (EBD)**，一种无需参数更新、基于奖励引导的推理时解码框架，用于从冻结的预训练 LLM 中激活任务导向行为。

- **核心思想**：将解码过程建模为从一个 **reward-tilted posterior** 分布中采样：
  $$
  \pi_\beta(y|x) \propto p_e(y|x) \exp(\beta R(y,x))
  $$
  其中 $p_e(y|x)$ 是预训练模型的先验分布，$R(y,x)$ 是外部轻量级 **reward model** 给出的奖励分数，$\beta$ 控制奖励与先验之间的权衡。

- **实现机制**：采用 **block-wise Metropolis-Hastings (MH) 采样器** 进行迭代优化：
  1. **初始化阶段**：从预训练模型生成少量候选响应，用 reward model 打分并选择最优者作为起点。
  2. **精细化阶段**：重复执行以下步骤：
     - 随机选择一个切点（cut position）
     - 保留前缀，从预训练模型重新生成后缀
     - 用 reward model 对新提案打分
     - 根据 MH 接受规则决定是否接受该提案

### 相比现有方法的优势
| 方面 | EBD | 现有方法（如 Power Sampling） |
|------|-----|-----------------------------|
| **有效性** ✅ | 显著提升任务表现，尤其在开放域任务上 | 在主观任务上常因过度搜索而退化 |
| **忠实性** ✅ | 通过 KL 正则化锚定于原始模型先验，避免产生低概率的“奖励黑客”输出 | 可能集中于低先验高奖励的异常输出 |
| **效率** ✅ | 每步仅需一次后缀重生成和一次 reward 评估，计算开销小 | Best-of-N 类方法需大量独立采样，成本高昂 |
| **通用性** ✅ | 适用于多种任务（数学、代码、对话等）且无需任务特定 verifier | 很多方法依赖任务特定的验证器 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
#### 客观任务（Objective Benchmarks）
- **GPQA**：研究生水平的科学推理
- **Math500**：数学问题求解
- **HumanEval**：代码生成

#### 主观任务（Subjective Benchmarks）
- **AlpacaEval2.0**：指令跟随质量（通过 LLM-as-a-judge 评估）
- **MT-Bench**：多轮对话质量
- **WritingBench**：开放式写作能力

### 实验设置和评估指标
- **模型**：5 个主流开源 LLMs
  - `Meta-Llama-3-8B`, `Mistral-7B-v0.3`, `Qwen2.5-7B`, `Qwen3-8B-Base`, `Olmo-3-1025-7B`
- **评估方式**：
  - 所有模型均以 **base model** 形式测试（未经过 SFT 或 RLHF）
  - 使用统一超参数：`β=3.5`, `K=12` refinement steps, `n_init=4`, `M=12` blocks
- **硬件**：8×NVIDIA A100-SXM4-80GB GPUs
- **软件**：PyTorch + HuggingFace Transformers，不使用 vLLM 加速

### 基线方法对比
- **Direct**：标准的直接采样（baseline）
- **Power Sampling**：近期提出的基于采样的解码方法（来自 Karan & Du, 2025）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Backbone | 方法 | AlpacaEval2.0 ↑ | Math500 ↓ | HumanEval ↑ | MT-Bench ↑ |
|---------|------|------------------|------------|--------------|-------------|
| Qwen3-8B-Base | Direct | 8.8 | 0.690 | 0.598 | 6.115 |
| | **EBD (ours)** | **44.5** (+405%) | **0.792** | **0.689** | **7.700** |
| Mistral-7B | Direct | 2.153 | 0.147 | 2.294 | 2.868 |
| | **EBD (ours)** | **7.249** | **0.232** | **4.776** | **5.375** |
| Llama3-8B | Direct | 1.525 | 0.299 | 3.000 | 1.709 |
| | **EBD (ours)** | **3.862** | **0.506** | **3.258** | **3.683** |

> 💡 **亮点**：EBD 将 `Qwen3-8B-Base` 在 AlpacaEval2.0 上的表现从 **8.8 提升至 44.5**，实现了近 **405% 的增长**。

### 与基线方法的对比结果
- **全面超越**：EBD 在所有 5 个模型、6 个基准上均显著优于 Direct 和 Power Sampling。
- **效率优势巨大**（见 Table 2）：
  - 在 `Mistral-7B` 上，EBD 相比 Power Sampling **降低延迟 18.9×**。
  - 平均每题推理时间远低于采样密集型方法。
- **主观任务提升最显著**：在开放域任务（如 AlpacaEval2.0）上增益最大，说明 EBD 能有效纠正“延续式输出”的缺陷。

### 消融实验结果
#### （1）行为相似性分析（Table 3）
- 衡量 base model + EBD 的输出与对应 **instruct-tuned model** 的行为一致性（Pearson 相关性）。
- 结果显示，EBD 使行为模式更接近 post-trained 模型：
  - 平均相关性从 **0.256 (Direct)** 提升至 **0.385 (EBD)**，绝对增益 **+0.129**。

#### （2）奖励模型规模影响（Table 5）
- 测试不同大小的 reward model（0.6B 到 8B）的影响。
- 发现：即使使用 **0.6B 的轻量级 reward model**，也能达到与 8B 模型相近的性能。
- 例如，在 `Qwen2.5-7B` 上，0.6B RM 达到 0.720，8B RM 达到 0.722。
- ✅ **结论**：EBD **对 reward model 规模不敏感**，具备良好鲁棒性。

#### （3）格式遵从性（Table 4）
- **Valid Response Rate (VRR)**：输出中包含正确 boxed answer 的比例。
- EBD 显著提升 VRR，例如 `Llama3-8B` 从 3.2% 提升至 28.5%。
- 同时准确率也提升，说明改进不仅是格式变化，更是内容质量提升。

---

## 4. 关键结论和发现

### 主要发现
1. **Decoding is a crucial part of evaluation**  
   当前对 pre-trained model 的评估严重低估其潜力，问题不在模型本身，而在 **decoding 策略**。EBD 揭示了“缺失的一环”。

2. **Reward-guided decoding unlocks latent capabilities**  
   无需任何参数更新，仅通过推理时的 reward 引导，即可从冻结模型中激活高质量的任务导向行为。

3. **EBD shifts behavior towards post-trained models**  
   EBD 不仅提升分数，还系统性地改变模型的行为模式，使其更接近经过 SFT/RLHF 的 instruct model。

4. **High efficiency and robustness**  
   - 相比采样密集型方法，EBD 效率高出一个数量级（10× 以上）。
   - 对 reward model 大小、超参数设置均表现出强鲁棒性。

### 方法的局限性
- 依赖一个有效的 **reward model**，若 reward model 本身有偏见或错误，会误导生成。
- 当前 reward model 仍需额外训练，虽可复用，但仍增加部署复杂度。
- MH 采样过程引入随机性，输出有一定波动性，难以完全 deterministic。

### 未来工作方向
- 探索更高效的 MH 变体或替代采样策略。
- 研究如何自适应选择切点（cut position）以进一步提升效率。
- 将 EBD 与少量参数微调结合，探索“推理时增强 + 参数更新”的混合范式。
- 构建通用、可迁移的 reward model，减少对任务特定 reward 的依赖。

---

> 📌 **一句话总结**：  
> **Energy-Based Decoding (EBD)** 证明了通过 **reward-guided inference-time decoding**，可以从冻结的 pre-trained LLM 中高效、稳定地激活出接近 post-trained 模型的任务能力，解决了当前评估体系中的关键偏差问题，为公平评估基础模型提供了新范式。

</details>

---

### 16. [AdaDPO: Self-Adaptive Direct Preference Optimization with Balanced Gradient Updates](https://arxiv.org/abs/2605.28440)

**Authors**: Shaolong Chen, Madalina Ciobanu, Qingqing Mao, Ritankar Das  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.28440v1  

#### Abstract
DPO has become a widely adopted alternative to RLHF for aligning LLMs with human preferences, eliminating the need for a separate reward model or RL loop. Recent theoretical analysis uncovers an asymmetric gradient behavior in DPO: the loss suppresses dispreferred responses substantially faster than...

---

### 17. [Faster Thermal Profiling of a Lunar Rover with Machine Learning Adapted Finite Difference Model](https://arxiv.org/abs/2605.27651)

**Authors**: Samuel Weber, Zaki Hasnain, Souma Chowdhury  
**Category**: cs.LG  
**Published**: 2026-05-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.27651v1  

#### Abstract
Autonomous space systems operating in extreme thermal environments require accurate and efficient thermal modeling to support both pre-mission system design and onboard autonomy. For lunar rovers, large temperature gradients, radiative heat transfer, and variable surface conditions make reliable the...

---

### 18. [Long Live The Balance: Information Bottleneck Driven Tree-based Policy Optimization](https://arxiv.org/abs/2605.28109)

**Authors**: Hao Jiang, Shurui Li, Tianpeng Bu, Bowen Xu, Xin Liu, Qihua Chen, Hongtao Duan, Lulu Hu, Bin Yang, Minying Zhang  
**Category**: cs.LG  
**Published**: 2026-05-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.28109v1  

#### Abstract
Recent advances in online reinforcement learning (RL) for large language models (LLMs) have demonstrated promising performance in complex reasoning tasks. However, they often exhibit an imbalanced exploration-exploitation trade-off, resulting in unstable optimization and sub-optimal performance. We ...

---

### 19. [Efficient Post-training of LLMs for Code Generation With Offline Reinforcement Learning](https://arxiv.org/abs/2605.28409)

**Authors**: Mingze Wu, Abhinav Anand, Shweta Verma, Mira Mezini  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.28409v1  

#### Abstract
Post-training using online reinforcement learning (RL) is an important training step for LLMs, including code-generating models. However, online RL for code generation involves LLM inference and verification of the generated output, which can take considerable time and resources. In this paper, we e...

---

### 20. [ICG: Improving Cover Image Generation via MLLM-based Prompting and Personalized Preference Alignment](https://arxiv.org/abs/2605.27374)

**Authors**: Zhipeng Bian, Jieming Zhu, Qijiong Liu, Wang Lin, Guohao Cai, Zhaocheng Du, Jiacheng Sun, Zhou Zhao, Zhenhua Dong  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.27374v1  

#### Abstract
Recent advances in multimodal large language models (MLLMs) and diffusion models (DMs) have opened new possibilities for AI-generated content. Yet, personalized cover image generation remains underexplored, despite its critical role in boosting user engagement on digital platforms. We propose ICG, a...

---

### 21. [Analyzing Quality-Latency-Resource Trade-offs in a Technical Documentation RAG Assistant Using LoRA Adaptation](https://arxiv.org/abs/2605.28222)

**Authors**: Evgenii Palnikov, Elizaveta Gavrilova  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.28222v1  

#### Abstract
We study quality-latency-resource trade-offs in a documentation-grounded retrieval-augmented generation (RAG) system that uses Low-Rank Adaptation (LoRA) of the generator. We build a manually verified benchmark of 5,144 question-answer pairs over the official Kubernetes documentation and combine it ...

---

### 22. [Roles with Rails: Contract-Preserving Role Evolution in Multi-Agent Structured Reasoning](https://arxiv.org/abs/2605.28433)

**Authors**: Ling-Yue Ge, Lan-Zhe Guo  
**Category**: cs.CL  
**Published**: 2026-05-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.28433v1  

#### Abstract
Role-based LLM multi-agent systems need adaptive role pools, yet adapting such systems is not merely a matter of prompt optimization: roles often carry structural obligations, including capability coverage, message compatibility, validation, final-answer aggregation, and parser-compatible output pro...

---

### 23. [Addressing Variable Heterogeneity in Distributed Multimodal Training with Entrain](https://arxiv.org/abs/2605.27918)

**Authors**: Insu Jang, Mosharaf Chowdhury  
**Category**: cs.DC  
**Published**: 2026-05-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.27918v1  

#### Abstract
Multimodal LLM datasets are inherently heterogeneous, with significant data variability. Although each modality exhibits independent variability, sample-level entanglement makes it difficult to balance workloads across both modalities and batches. We present Entrain, a distributed MLLM training fram...

---

### 24. [Fault Tolerance of Accelerated Asynchronous Fixed-Point Iterations on Flexible Computing Infrastructure](https://arxiv.org/abs/2605.28426)

**Authors**: Evan Coleman, Masha Sosonkina  
**Category**: cs.DC  
**Published**: 2026-05-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.28426v1  

#### Abstract
Asynchronous iterative methods tolerate straggling processors by allowing workers to proceed with stale data, but at a cost: the iterates become inconsistent, potentially degrading convergence. We investigate whether convergence accelerators such as Anderson acceleration compensate for this degradat...

---

### 25. [Multi-Mixer Models: Flexible Sequence Modeling with Shared Representations](https://arxiv.org/abs/2605.28769)

**Authors**: Kevin Y. Li, Asher Trockman, Ananda Theertha Suresh, Ziteng Sun  
**Category**: cs.LG  
**Published**: 2026-05-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.28769v1  

#### Abstract
Softmax attention is the cornerstone of modern large language models, but its memory scales linearly and compute quadratically with sequence length. Linear recurrent models, such as linear attention and state space models, have become widely studied as alternatives to attention due to their linear c...

---

### 26. [Operational AI Deployment Assurance: Governance-State Orchestration Under Threshold-Sensitive Deployment Conditions -- A Governance Framework for High-Stakes AI Systems](https://arxiv.org/abs/2605.27827)

**Authors**: Khalid Adnan Alsayed  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.27827v1  

#### Abstract
AI governance frameworks increasingly emphasize fairness, transparency, accountability, and lifecycle risk management in high-stakes domains. However, many current approaches remain observational, relying on static metric reporting, post-hoc auditing, and monitoring dashboards without directly gover...

---

### 27. [A Unified Framework for the Evaluation of LLM Agentic Capabilities](https://arxiv.org/abs/2605.27898)

**Authors**: Pengyu Zhu, Lijun Li, Yaxing Lyu, Qianxin Luo, Jingyi Yang, Yi Liu, Tingfeng Hui, Xinyu Yuan, Li Sun, Sen Su, Jing Shao  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.27898v1  

#### Abstract
As LLMs are increasingly deployed as agents, reliable assessment of their agentic capabilities has become essential. However, reported benchmark scores often jointly reflect model capability and the implementation choices each benchmark is packaged with, making cross-benchmark results difficult to i...

---

### 28. [SKILLC: Learning Autonomous Skill Internalization in LLM Agents via Contrastive Credit Assignment](https://arxiv.org/abs/2605.27899)

**Authors**: Hongxiang Lin, Zhirui Kuai, Erpeng Xue, Lei Wang  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.27899v1  

#### Abstract
Structured skill prompts improve exploration in long-horizon agentic reinforcement learning (RL). Skill-augmented RL methods retain external skills at inference, while skill-internalization RL methods withdraw them during training to enable autonomous performance. However, existing internalization a...

---

### 29. [Adaptive Reservoir Computing for Multi-Scenario Chaotic System Forecasting](https://arxiv.org/abs/2605.28145)

**Authors**: Shadmehr Zaregarizi, Khashayar Yavari  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.28145v1  

#### Abstract
We present an adaptive reservoir computing framework for the CTF-4-Science Lorenz benchmark, which evaluates machine learning models across twelve distinct tasks spanning five qualitatively different scenarios: baseline forecasting, noisy signal reconstruction, forecasting under noise, few-shot lear...

---

### 30. [DenoiseRL: Bootstrapping Reasoning Models to Recover from Noisy Prefixes](https://arxiv.org/abs/2605.28421)

**Authors**: Caijun Xu, Changyi Xiao, Zhongyuan Peng, Yixin Cao  
**Category**: cs.AI  
**Published**: 2026-05-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.28421v1  

#### Abstract
Reinforcement learning has become a central paradigm for advancing reasoning in large language models, yet most existing methods still depend on stronger teacher models or heavily curated difficult datasets, limiting scalable capability improvement. In this paper, we introduce DenoiseRL, a reinforce...

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
