# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-09 06:44:57 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [EvoESAP: Non-Uniform Expert Pruning for Sparse MoE](https://arxiv.org/abs/2603.06003)

**Authors**: Zongfang Liu, Shengkun Tang, Boyang Sun, Zhiqiang Shen, Xin Yuan  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2603.06003v1  

#### Abstract
Sparse Mixture-of-Experts (SMoE) language models achieve strong capability at low per-token compute, yet deployment remains memory- and throughput-bound because the full expert pool must be stored and served. Post-training expert pruning reduces this cost, but most methods focus on which experts to ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# EvoESAP: Non-Uniform Expert Pruning for Sparse MoE 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **Sparse Mixture-of-Experts (SMoE)** 模型虽然通过条件计算机制降低了每 token 的计算成本，但在部署时仍需存储全部专家（expert），导致内存占用高、吞吐受限。
- 现有的 **expert pruning** 方法大多采用**均匀层间稀疏度分配**（uniform layer-wise sparsity），即每一层剪枝相同比例的专家，忽略了不同层对模型能力影响的差异性。
- 同时，如何高效评估剪枝后模型与原始模型的行为一致性（尤其是生成质量）缺乏有效手段。

### 提出的新方法与新思路
1. **Expected Speculative Acceptance Proxy (ESAP)**  
   - 受 speculative decoding 启发，提出一种基于 teacher-forcing 的代理指标，用于衡量剪枝模型与原模型在 next-token 分布上的相似性。
   - ESAP 是一个有界、稳定且计算高效的 fitness 函数，避免了昂贵的自回归解码过程，适用于大规模候选模型比较。

2. **EvoESAP：进化搜索框架**
   - 将 expert pruning 解耦为两个独立步骤：
     - **层内选择**（within-layer selection）：使用已有标准（如 Frequency, EAN, SEER, REAP）确定各层专家的重要性排序。
     - **层间分配**（across-layer allocation）：在固定全局剪枝预算下，使用进化算法优化非均匀的层间稀疏度分布。
   - 利用 **level-switch mutation** 在保持总剪枝数不变的前提下进行局部调整，以探索更优的非均匀配置。

### 相比现有方法的优势
| 维度 | 传统方法 | EvoESAP |
|------|--------|--------|
| 层间稀疏策略 | 默认 uniform | 显式优化 non-uniform 分配 |
| 评估方式 | 多依赖 MCQ 或完整生成测试 | 引入 ESAP 快速代理评估生成一致性 |
| 通用性 | 通常绑定特定剪枝准则 | Plug-and-play，兼容多种 within-layer 排序标准 |
| 效率 | 高成本 autoregressive 评估不可行于搜索 | ESAP 节省 ~18× 搜索时间（见 Table 4） |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 类别 | 数据集 |
|------|-------|
| **Calibration / Search Set** | `evol-codealpaca-v1`（用于计算专家重要性）、`tulu-3-sft-personas-math`（用于 ESAP fitness 评估） |
| **Open-ended Generation** | EvalPlus（code）、LiveCodeBench（code）、GSM8K & MATH-500（math）、WildBench（creative writing） |
| **Multiple Choice (MC)** | ARC-C/E, BoolQ, HellaSwag, MMLU, OpenBookQA, RTE, WinoGrande |

> 所有任务均采用 zero-shot 设置。

### 实验设置
- **模型规模**：覆盖 7B–30B 参数的 SMoE LLMs：
  - OLMoE-1B-7B-0125-Instruct (7B)
  - ERNIE-4.5-21B-A3B-PT (21B)
  - Qwen3-30B-A3B-Instruct-2507 (30B)
- **剪枝比例**：25% 和 50% 全局 sparsity
- **within-layer 排序标准**：Frequency, SEER, EAN, REAP
- **搜索参数**：
  - Population size: 32
  - Elite size: 4
  - Max transfer Δ: 4
  - Max mutation steps: 3
  - Generations: 10–50（依模型大小而定）

### 评估指标
| 指标类型 | 指标名称 | 说明 |
|--------|--------|------|
| **主性能指标** | Code Avg, Math Avg, MC Avg | 各子任务平均得分 |
| **fitness 函数对比** | SPEC-DEC vs ESAP | 真实 speculative acceptance vs 本文提出的代理指标 |
| **消融实验** | 不同 fitness（KL/NLL/SAP/ESAP）、样本量（8–128）的影响 | 验证 ESAP 的有效性与鲁棒性 |

### 基线方法对比
- **Uniform Pruning (UNI)**：相同 within-layer 排序下，每层按比例均匀剪枝。
- **其他剪枝准则本身**：如 REAP、EAN 等作为排序依据，但不改变其默认 uniform 分配。
- **真实 speculative decoding (SPEC-DEC)**：作为 upper-bound 参考，验证 ESAP 的近似效果。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）
#### ✅ 在 **开放生成任务** 上显著提升：
| 模型 | Sparsity | 方法 | Code Avg ↑ | Math Avg ↑ | MATH-500 ↑ | MC Avg |
|------|---------|------|-----------|------------|-------------|--------|
| ERNIE-21B | 50% | REAP + UNI | 0.730 | 0.598 | 0.798 | 0.575 |
| ERNIE-21B | 50% | REAP + **EvoESAP** | **0.737** | **0.608** | **0.818** | 0.575 |
| ERNIE-21B | 50% | Freq + UNI | 0.647 | 0.272 | 0.542 | 0.565 |
| ERNIE-21B | 50% | Freq + **EvoESAP** | **0.781** | **0.468** | **+19.6%** | **0.557** |

> 💡 **最大增益达 +19.6% on MATH-500**（ERNIE @ 50% sparsity, Frequency-based pruning）

#### ✅ 多个模型和剪枝标准上一致优于 uniform：
- **OLMoE @ 25%**：REAP + EvoESAP → Code Avg +2.9%, Math Avg +2.8%
- **Qwen3 @ 50%**：Frequency + EvoESAP → Code Avg +6.6%, WildBench +7.2%, Math Avg +9.5%
- 即使是较弱的 pruning metric（如 Frequency），通过优化 allocation 也能大幅提升表现。

#### ⚠️ Multiple Choice 性能基本持平或轻微波动
- MC Avg 改变通常小于 ±1%，表明 EvoESAP **主要增强生成能力而不损害判别任务表现**。

### 消融实验结果（Table 5）
| Fitness Function | Code Avg | Math Avg | MC Avg | 结论 |
|------------------|----------|----------|--------|------|
| KL | 0.331 | 0.405 | 0.582 | 较差 |
| NLL | 0.334 | 0.424 | 0.576 | 中等 |
| SAP (Monte Carlo) | 0.339 | 0.420 | 0.584 | 波动大 |
| **ESAP (Ours)** | **0.344** | **0.426** | **0.581** | **最优且最稳定** |

> ✅ **ESAP 在生成任务上全面领先，同时保持 MC 竞争力**

#### 搜索样本量敏感性分析
- 使用仅 **32–64 个样本**即可获得高质量搜索结果；
- 超过 128 个样本未带来收益，甚至略有下降 → 表明 ESAP 对噪声鲁棒。

### 搜索效率对比（Table 4）
| Fitness | GPU | Time (h) | Code Avg | MC Avg |
|--------|-----|----------|---------|--------|
| SPEC-DEC | 2×L40S | 29.49 | 0.171 | 0.565 |
| **ESAP** | **1×L40S** | **1.64** | **0.173** | **0.557** |

> 🔥 **ESAP 将搜索时间从近 30 小时压缩至 1.6 小时，提速 ~18×，性能几乎无损**

---

## 4. 关键结论和发现

### 主要发现
1. **非均匀层间稀疏度分配至关重要**
   - 即使 within-layer 排序固定，**如何分配剪枝预算到不同层**会显著影响模型性能。
   - “看似合理”的启发式（如按路由频率全局排序再分配）可能反而损害性能（见 Figure 1）。

2. **EvoESAP 是 plug-and-play 的通用优化模块**
   - 可无缝集成到任何基于 importance scoring 的剪枝流程中（如 REAP、EAN 等）。
   - 不需要微调（finetuning-free），适合 post-training 压缩场景。

3. **ESAP 是有效的生成行为代理指标**
   - 与真实 speculative decoding 高度相关，但成本极低。
   - 可视为 total variation distance 的补集：`ESAP(x) = 1 - TV(p, q)`，具有理论解释性。

4. **开放生成任务对 allocation 更敏感**
   - 尤其在高稀疏度（50%）下，生成质量对专家保留位置高度敏感。
   - MC 任务相对鲁棒，因此 EvoESAP 实现了 **generation-preserving compression**。

### 方法的局限性
- **假设 within-layer 排序已知且固定**：未联合优化层内选择与层间分配。
- **进化搜索引入额外开销**：尽管 ESAP 高效，但仍需数百次前向传播进行搜索。
- **结果受 calibration 数据影响大**：例如使用 C4 替代 evol-codealpaca-v1 会导致代码任务性能大幅下降（Appendix D）。

### 未来工作方向
- 开发更高效的搜索策略（如基于梯度或强化学习）。
- 探索 joint optimization of within-layer selection and across-layer allocation。
- 研究跨任务/领域下的迁移性：是否可在某一任务上搜索的 allocation 泛化到其他任务？

---

> 📌 **总结一句话**：  
> **EvoESAP 揭示了 SMoE 剪枝中被忽视的关键维度——层间稀疏度分配，并通过轻量级代理指标 ESAP 与进化搜索，实现了无需微调、显著提升开放生成能力的非均匀剪枝方案，在多个大模型上验证了其有效性与通用性。**

</details>

---

### 2. [MoE Lens -- An Expert Is All You Need](https://arxiv.org/abs/2603.05806)

**Authors**: Marmik Chaudhari, Idhant Gulati, Nishkal Hundia, Pranav Karra, Shivam Raval  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.05806v1  

#### Abstract
Mixture of Experts (MoE) models enable parameter-efficient scaling through sparse expert activations, yet optimizing their inference and memory costs remains challenging due to limited understanding of their specialization behavior. We present a systematic analysis of expert specialization in MoEs t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MoE Lens — An Expert Is All You Need 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Mixture of Experts (MoE) 模型虽然通过稀疏激活实现了参数高效的扩展，但在推理效率、内存开销和专家行为理解方面仍面临挑战。特别是：
- **专家专业化程度不明确**：大量专家中是否只有少数真正承担关键计算？
- **知识冗余与负载不均衡**：许多专家可能贡献极小，导致资源浪费。
- **缺乏对专家贡献的可解释性分析工具**。

本文旨在系统分析 MoE 中专家的**专业化行为**，并探索能否在保持性能的前提下大幅减少活跃专家数量以优化推理。

---

### 🚀 提出的新方法与新思路

1. **双路径专家行为分析框架**：
   - **Domain-specific routing pattern analysis**：量化不同领域输入下各专家被路由的频率，识别领域专用专家。
   - **Extended LogitLens for early decoding**：将 LogitLens 扩展至 MoE 架构，追踪单个专家输出如何影响中间层表示和最终预测。

2. **提出“Top-Weighted Expert Sufficiency”假设**：
   - 单个最高权重专家（top-1 expert）结合 residual stream 足以逼近全量 top-k 专家组合的输出表示。

3. **基于专家贡献度的可解释性剪枝路径**：
   - 发现大多数专家贡献微弱，为后续动态专家选择和静态剪枝提供理论依据。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | 本工作 |
|------|--------|--------|
| 分析粒度 | 宏观负载均衡指标（如路由熵） | 细粒度专家贡献追踪（逐层、逐域） |
| 可解释性工具 | 缺乏针对 MoE 的中间表示解码手段 | 引入并扩展 LogitLens 到 MoE 层级 |
| 推理优化潜力 | 固定 top-k 激活所有专家 | 支持仅激活 top-1 专家而不显著损失性能 |

> ✅ **优势总结**：首次从**表示一致性**和**预测收敛性**两个角度验证了“一个专家就足够”的现象，揭示了 MoE 内部高度集中化的知识分布特性。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

共使用七个领域的子集，涵盖多语言、多任务场景：

| 数据集 | 领域描述 |
|-------|--------|
| **Gutenberg English Dataset** | 英文文学文本 |
| **GitHub Code Subset (Paloma)** | 编程代码 |
| **French-QA (FQuAD)** | 法语问答任务 |
| **GSM8K** | 小学数学应用题（英文） |
| **AIME Problem Sets** | 数学竞赛题目 |
| **Chinese Fineweb Edu** | 高质量中文教育语料 |
| **arXiv Dataset** | 多学科科研论文预印本 |

> 实验重点关注 **English**, **French-QA**, 和 **GSM8K** 三个代表性 domain。

---

### ⚙️ 实验设置

- **模型**：`DeepSeekMoE`（2 shared + 64 routed experts, top-k=6）
- **对比模型**（附录中扩展）：`OLMoE`, `Qwen 1.5 MoE`
- **主要分析对象**：routed experts（排除 shared expert，因其负责通用知识）

#### 主要实验设计：

1. **Expert Specialization Analysis**
   - 定义：  
     $$
     \text{Expert Specialization}(E_i, D) = \frac{N^{(k)}_{E_i,D}}{N_D}
     $$
     其中 $N^{(k)}_{E_i,D}$ 是 domain $D$ 中 $E_i$ 进入 top-k 的 token 数。
   - 基线：均匀路由期望 ≈ 9.4%（6/64）

2. **Extended LogitLens Decoding**
   - 对每层 hidden state 进行 early decoding：
     $$
     \text{LogitLens}_\text{ext}(h^l_t) = \text{LayerNorm}(h^l_t + u^l_t) W_u
     $$
     其中 $h^l_t$: expert 输出；$u^l_t$: attention 后 residual stream；$W_u$: unembedding matrix。
   - 比较三种情况的预测一致性：
     - Top-1 expert + residual → $H_1^l$
     - Top-6 experts weighted sum + residual → $H_6^l$
     - 最终层输出 → $h_f$

3. **评估指标**
   - **Cosine Similarity**：比较 $H_1^l$ 与 $H_6^l$ 在各层的隐藏状态相似性。
   - **Normalized Log Perplexity**：衡量减少活跃专家数（从 k=6 → k=1）时的语言建模性能下降。
   - **Routing Distribution Visualization**：可视化各 domain 下专家路由占比。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

| 指标 | 结果 |
|------|------|
| **Top-1 Expert Routing Coverage** | 少数专家处理 >50% 的特定 domain 输入（远高于 9.4% 均匀基线） |
| **Hidden State Cosine Similarity ($H_1^l$ vs $H_6^l$)** | 层间平均高达 **0.95**，部分层接近 1.0 |
| **Perplexity Increase (k=6 → k=1)** | 平均仅上升约 **5%**，表明预测质量几乎不变 |
| **Prediction Convergence via LogitLens** | $H_1^l$ 逐层解码即可快速收敛到正确 next-token，与完整输出一致 |

> 图 3 显示，在所有 27 层中，$H_1^l$ 与 $H_6^l$ 的 cosine similarity 始终维持高位，说明 top-1 expert 主导了表示构建。

---

### 🔁 与基线方法对比结果

| 对比维度 | 本方法表现 |
|--------|----------|
| **vs. Uniform Routing Baseline** | 多个专家路由率显著高于 9.4%，证明存在强 specialization |
| **vs. Full Ensemble Prediction** | 单专家 + residual 输出已能近似完整 ensemble 输出 |
| **vs. Prior Interpretability Tools** | Extended LogitLens 成功揭示 expert-level 表示演化过程，优于仅看 attention 或梯度的方法 |

---

### 🔍 消融实验结果（隐含于主实验）

- **消融专家数量（top-k ablation）**：
  - 当从 top-6 减少到 top-1 时，perplexity 上升有限（<5%），且下游任务准确率未明显下降。
- **消融专家身份（specialized vs non-specialized）**：
  - 非专业化专家即使被移除，对整体输出影响极小。
- **跨 domain 泛化性测试**：
  - 不同 domain 下均观察到类似模式，说明结论具有一定普适性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **专家高度专业化且贡献集中**：
   - 尽管有 64 个 routed experts，但每个 domain 主要由 **少数几个专家主导**。
   - 存在明显的 domain-specialized experts（如法语 QA 中某专家高频激活）。

2. **Top-Weighted Expert Sufficiency Hypothesis 成立**：
   - **单个 top-1 expert + residual stream** 足以生成与全部 6 个专家加权和高度一致的表示。
   - 表示空间中 $H_1^l \approx H_6^l$，cosine similarity 达 0.95+。

3. **Minimal Contribution from Non-Dominant Experts**：
   - 其他 5 个专家对最终输出贡献极小，存在严重知识冗余。

4. **Early Decoding 验证预测一致性**：
   - 使用 extended LogitLens 可见，仅靠 top-1 expert 已能在深层准确预测 next token。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖特定 MoE 架构** | 实验集中在 DeepSeekMoE，其他 MoE（如密集路由型）可能行为不同 |
| **未实现端到端剪枝部署** | 当前为分析性研究，尚未发布实际加速推理的轻量化版本 |
| **Shared Expert 影响未深入探讨** | shared expert 被排除在外，其与 routed experts 的交互机制尚待研究 |
| **动态输入复杂度适应缺失** | 当前固定选择 top-1，未考虑根据输入难度动态调整 k 值 |

---

### 🔮 未来工作方向

1. **Dynamic Expert Selection**：
   - 设计基于输入复杂度或 uncertainty 的 adaptive routing，动态决定激活专家数。

2. **Expert Pruning & Sparse Inference**：
   - 开发基于贡献度的剪枝策略，永久移除低效专家，降低显存占用和计算成本。

3. **Knowledge Localization in Experts**：
   - 探索专家内部表示稀疏性，定位 factual knowledge 存储位置（如“化学化合物”由哪个 expert 编码）。

4. **Extension to Other MoE Variants**：
   - 应用于 OLMoE、DeepSeek-V2、Qwen-MoE 等架构，建立通用专家行为理论。

5. **Integration with TunedLens**：
   - 替代原始 LogitLens，使用 layer-wise learned transformation 提升 early decoding 准确性。

---

> 💡 **一句话总结**：  
> 该论文通过 **extended LogitLens** 与 **routing pattern analysis** 揭示了 MoE 模型中“**一个专家就足够**”的现象——尽管结构上激活多个专家，但实质上预测由**单一主导专家驱动**，为高效、可解释的 MoE 推理打开了新路径。

</details>

---

### 3. [Real-Time AI Service Economy: A Framework for Agentic Computing Across the Continuum](https://arxiv.org/abs/2603.05614)

**Authors**: Lauri Lov\'en, Alaa Saleh, Reza Farahani, Ilir Murturi, Miguel Bordallo L\'opez, Praveen Kumar Donta, Schahram Dustdar  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.05614v1  

#### Abstract
Real-time AI services increasingly operate across the device-edge-cloud continuum, where autonomous AI agents generate latency-sensitive workloads, orchestrate multi-stage processing pipelines, and compete for shared resources under policy and governance constraints. This article shows that the stru...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Real-Time AI Service Economy: A Framework for Agentic Computing Across the Continuum

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**实时AI服务在设备-边缘-云连续体（device-edge-cloud continuum）上的资源管理挑战**提出系统性解决方案。随着自主AI代理（autonomous AI agents）的兴起，这些代理会主动生成任务、编排多阶段处理流水线，并在策略和治理约束下竞争共享资源。传统集中式调度难以应对跨域、多主体、动态依赖的复杂性，而简单的市场机制在存在强互补性（complementarities）时也会失效。

核心问题是：**如何在去中心化环境下实现高效、稳定且激励相容（incentive-compatible）的资源分配？**

### 提出的新方法与创新思路
论文提出了一个**统一的框架**，其核心创新在于将**服务依赖图（service-dependency DAG）的拓扑结构**与**经济机制设计（mechanism design）** 相结合，揭示了结构对系统可管理性的决定性作用。

1.  **结构性可管理性理论（Structural Regimes for Efficient Allocation）**  
    首次证明：当服务依赖图（DAG）具有**树状（tree）或串并联（series-parallel, SP）结构**时，可行分配空间（feasible allocation set）是**拟阵（polymatroid）**。这一性质至关重要，因为它保证了：
    -   分配优化可在多项式时间内完成。
    -   存在瓦尔拉斯均衡（Walrasian equilibrium），即价格能收敛到稳定状态。
    -   在适当的机制设计下（如VCG或clenching auction），真实报价（truthful bidding）是占优策略（DSIC），防止代理虚报估值。

2.  **混合管理架构（Hybrid Management Architecture）**  
    针对更复杂的任意DAG（引入强互补性，导致市场不稳定），提出了一种**封装（encapsulation）** 架构：
    -   **跨域集成商（Cross-Domain Integrators）**：将复杂的子图封装成一个“资源切片（resource slice）”，对外暴露一个**简化、可替代（substitutable）的容量接口**。
    -   **本地市场（Local Marketplaces）**：在集成商内部，通过本地市场协调底层的同质化资源（fungible resources）。
    -   这种架构使得面向代理的市场层始终保持**拟阵结构**，从而继承了上述所有优势。

3.  **治理感知模型（Governance-Aware Management Model）**  
    将信任阈值、数据本地性等治理约束（governance constraints）建模为对可行分配集的**坐标截断（coordinate-wise truncations）**，并证明这种操作**保持了拟阵结构**，实现了效率与合规的权衡。

### 相比现有方法的优势
| 方面 | 现有方法 | 本文方法 |
| :--- | :--- | :--- |
| **依赖关系处理** | 通常假设资源独立或简单链式 | 显式建模复杂DAG，区分可管理（tree/SP）与不可管理（entangled）结构 |
| **去中心化协调** | 假设独立商品，不适用于捆绑资源 | 通过结构分析和封装，使去中心化市场在复杂场景下依然有效 |
| **激励兼容性** | 在一般互补性下难以保证 | 在tree/SP或封装后，严格证明DSIC，确保系统鲁棒性 |
| **治理整合** | 多为事后过滤或独立模块 | 将治理作为第一类设计要素，直接嵌入可行性定义 |

---

## 2. 核心实验方法和设置

### 实验设置
-   **模拟环境**：构建了一个异构的三层（设备-device、边缘-edge、云-cloud）计算环境，各层具有不同的容量和基础延迟。
-   **代理行为**：50个自主代理以泊松过程生成**延迟敏感型任务**，任务价值随延迟指数衰减。
-   **服务依赖图（DAG）**：实验了四种拓扑结构：
    -   `Linear` / `Tree`：拟阵结构，理论上可管理。
    -   `Series-Parallel (SP)`：拟阵结构，但负载分布可能导致拥塞。
    -   `Entangled`：非拟阵结构，存在跨层强耦合，理论上难管理。
-   **市场机制**：采用**Tâtonnement**（试探性）价格调整机制进行市场出清，作为价格发现的代理。
-   **变量控制**：系统性地移除四个关键组件进行消融研究（ablation study）：
    -   `-S`：破坏结构纪律（从tree到entangled）
    -   `-H`：移除混合架构（从hybrid到naive）
    -   `-G`：移除治理（从strict到none）
    -   `-M`：移除市场机制（从market到value-greedy）

### 评估指标
-   **价格波动（Price Volatility, σ）**：衡量市场稳定性，计算价格对数收益率的标准差。
-   **丢包率（Drop Rate）**：未能完成的任务比例。
-   **中位延迟（Median Latency）**：成功任务的端到端延迟。
-   **社会福利（Welfare）**：实现的价值减去成本。
-   **服务覆盖率（Service Coverage）**：获得任何分配的任务比例。
-   **可扩展性（Scaling Ceiling）**：系统从稳定到退化的临界代理数量。

### 基线方法对比
-   **Naive Allocation**：无封装的直接按层定价。
-   **Hybrid Architecture**：本文提出的集成商封装架构。
-   **Allocation Rules**（用于`-M`实验）：
    -   `Random`：随机分配。
    -   `EDF`：最早截止时间优先。
    -   `Value-Greedy`：按预期价值贪心分配（无价格信号）。
    -   `Market`：基于价格的Tâtonnement机制。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比
1.  **结构决定稳定性（-S Ablation）**：
    -   在高负载下，`Tree`拓扑保持**零价格波动（σ=0）** 和约50%的丢包率。
    -   `Entangled`拓扑则出现**严重的价格振荡（σ=0.273）** 和**近乎100%的丢包率**。
    -   结论：**依赖图拓扑是系统稳定性和可扩展性的首要决定因素**。

2.  **混合架构的有效性（-H Ablation）**：
    -   在`SP`高负载下，朴素架构（naive）的价格波动为 **σ=0.34**。
    -   采用混合架构（hybrid）后，价格波动降至 **σ=0.10**，**降低了70-75%**，且未牺牲吞吐量。
    -   结论：**封装架构能有效恢复价格稳定性**。

3.  **治理的权衡（-G Ablation）**：
    -   严格的治理（strict governance）在`entangled`高负载下，将中位延迟从**404ms降低34%至266ms**。
    -   代价是服务覆盖率**减半**。
    -   结论：**治理通过排除高拥塞分配来提升服务质量，但牺牲了吞吐量**。

4.  **市场机制的作用（-M Ablation）**：
    -   在真实报价假设下，`Market`机制与`Value-Greedy`规则产生的社会福利差异**小于1%**。
    -   这表明，在非策略环境下，市场机制的信息价值有限，其核心价值在于**激励对齐**（incentive alignment），而非信息聚合。

### 消融实验结果
| 组件 | 条件 | 移除前 | 移除后 | 主要影响 |
| :--- | :--- | :--- | :--- | :--- |
| `-S` | Tree → Entangled | σ=0 | σ=0.273 | 破坏稳定性，导致市场崩溃 |
| `-H` | Hybrid → Naive | σ=0.10 | σ=0.34 | 价格波动增加70-75% |
| `-G` | None → Strict | Latency=404ms | Latency=266ms | 延迟降低34%，但覆盖率减半 |
| `-M` | Market → Value-greedy | Welfare diff <1% | | 证实市场在真实报价下冗余 |

---

## 4. 关键结论和发现

### 主要发现
1.  **结构是第一性原理**：服务依赖图的**拓扑结构**（tree/SP vs. entangled）从根本上决定了去中心化资源市场的**可管理性**。这是本文最核心的洞见。
2.  **封装是关键桥梁**：对于无法避免的复杂依赖，**跨域集成商的封装**是一种有效的工程实践，它将深层的互补性吸收在内部，向市场暴露一个可管理的、拟阵化的接口。
3.  **去中心化可复制中心化最优**：在满足结构性条件（或经过封装）且代理真实报价的前提下，去中心化市场可以达到与拥有全局视图的中心化规划者**同等质量的分配**。
4.  **治理是主动的设计参数**：治理不仅是合规过滤器，更是主动塑造系统运行模式的工具，它与拓扑和负载共同决定了**效率-合规权衡曲线**。

### 方法的局限性
1.  **实验假设**：当前模拟假设代理真实报价，尚未测试在**策略性行为**（strategic misreporting）下的表现。这是验证DSIC机制价值的关键下一步。
2.  **静态拓扑**：假设DAG结构在运行时是固定的，而实际中AI流水线可能动态重组。
3.  **集成商非策略性**：假设集成商是非策略性的基础设施，未考虑其自身可能成为自私主体并操纵容量报告。
4.  **简化模型**：使用的DAG拓扑和延迟模型是简化的，与生产级AI管道的复杂性仍有差距。

### 未来工作方向
1.  **引入策略性代理**：用**上升式拍卖（ascending clinching auctions）** 或**强化学习（RL）代理**替换Tâtonnement，验证在策略行为下DSIC机制的鲁棒性。
2.  **动态管道演化**：研究服务组合在运行时动态变化时，系统的稳定性和适应性。
3.  **集成商的机制设计**：将集成商本身建模为自利的参与者，设计机制防止垄断定价和容量虚报。
4.  **语义切片（Semantic Slicing）**：探索更具语义感知能力的切片边界划分方法。
5.  **真实世界部署**：在多域测试平台中部署该架构，验证其互操作性、开销和实际性能。

</details>

---

### 4. [Attention Meets Reachability: Structural Equivalence and Efficiency in Grammar-Constrained LLM Decoding](https://arxiv.org/abs/2603.05540)

**Authors**: Faruk Alpay, Bilge Senturk  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.05540v1  

#### Abstract
We study grammar-constrained decoding (GCD) as a coupling between an autoregressive next-token distribution and a reachability oracle over a pushdown system compiled from a context-free grammar (CFG). We prove an oracle invariance theorem: language-equivalent grammars induce identical admissible nex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Attention Meets Reachability: Structural Equivalence and Efficiency in Grammar-Constrained LLM Decoding*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于 **Grammar-Constrained Decoding (GCD)** 中一个核心矛盾：**语言等价性（language equivalence）与解码效率之间的脱节**。尽管两个上下文无关文法（CFG）可能生成相同的语言 $ L(G) $，它们在实际用于约束 LLM 解码时，可能导致显著不同的运行时开销（如内存、计算量、延迟）。这种差异源于文法结构对底层 **pushdown 系统可达性分析（reachability）** 的影响。

传统 GCD 工具通常将文法视为“黑盒”，忽视其内部结构对性能的影响，导致即使语义相同，不同文法实现的推理速度可能相差数倍。

---

### 提出的新方法与新思路

论文提出了一套**统一的理论框架**，将 GCD 形式化为 **Transformer 模型与基于 CFG 编译的 pushdown 可达性 oracle 的耦合系统**，并在此基础上做出以下六大贡献：

#### 1. **Oracle Invariance 定理**
- **核心思想**：若两个 CFG $ G $ 和 $ G' $ 是语言等价的（$ L(G) = L(G') $），则对于任意前缀 $ u $，其允许的下一个 token 集合 $ \Omega_G(u) = \Omega_{G'}(u) $ 完全一致。
- **意义**：从理论上证明了 GCD 的输出行为仅依赖于语言本身，而非文法的具体形式 —— 即“**语义不变性**”。

#### 2. **Structural Ambiguity Cost (SAC) 度量**
- **定义**：SAC 衡量每一步新增 token 后，打包解析森林（packed parse forest）的增长量，反映在线解码过程中的结构性歧义成本。
- **创新性**：首次提出左到右增量式的解析结构增长度量，用于量化不同文法带来的动态复杂度差异。

#### 3. **引擎无关的下界分析（Engine-Independent Lower Bounds）**
- 证明任何满足 soundness、parse-preserving 和 retrieval-efficient 的 GCD 引擎，在处理某些常数大小的 CFG（如 $ G_4: S \to SS \mid a \mid b $）时，必须承受 $ \Omega(t^2) $ 的每步更新代价（累计 $ \Omega(n^3) $）。
- 这一结果独立于具体实现，揭示了 SAC 成本的本质性来源。

#### 4. **状态空间膨胀分析（State-Space Blowup）**
- 对经典语言 $ a^n b^n $ 给出精确控制状态计数：
  - $ G_1: S \to aSb \mid \varepsilon $：编译后有 8 个控制状态；
  - $ G_2: S \to aAb \mid \varepsilon, A \to aAb \mid \varepsilon $：编译后有 15 个控制状态。
- 揭示了冗余非终结符会直接导致 **15/8 ≈ 1.875 倍的状态空间膨胀**，增加内存和缓存压力。

#### 5. **Decoding-Cost Equivalence Classes 与 Canonical Low-SAC Forms**
- 定义了新的等价关系 $ G =_{\text{dec}} H $：不仅要求语言等价，还要求最优可实现的 SAC 渐近一致。
- 在有限重写族（bounded rewrite family）内，证明存在 **最小 SAC 的代表文法（minimal-SAC representative）**，为自动化文法优化提供数学基础。

#### 6. **Grammar-Conditioned Autoregressive Process 的概率建模**
- 使用 **Doob h-transform** 刻画真正的条件分布 $ p(\cdot | T(y) \in L) $。
- 推导出硬掩码（hard masking）相对于真实条件采样的失真界限：
  - KL 散度上界为 $ \log I(y_{<t}) $，其中 $ I = h_{\max}/h_{\min} $ 是存活概率的 spread。
  - 当不同合法 token 导致完成概率差异大时，硬掩码引入显著偏差。

#### 7. **神经架构集成与性能预测模型**
- 将 SAC 扩展为可校准的运行时预测模型，结合：
  - Transformer/MoE 架构下的延迟包络（latency envelope）
  - Beam Search 下的符号工作放大效应
  - 实验 trace 驱动的仿射时间代理模型：$ T_{\text{mask}}(t) \approx a \cdot S(w) + b $

---

### 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **理论深度** | 超越工程层面，建立 GCD 的形式化复杂性理论，连接 CFL parsing、reachability 与 LLM inference |
| **性能洞察** | 揭示“相同语言 ≠ 相同效率”，指出文法结构是隐藏的性能瓶颈 |
| **优化指导** | 提供自动文法优化路径：通过等价改写（如 inlining）降低 SAC/K(G)，提升推理速度 |
| **通用性** | 理论适用于所有基于 CFG 的 GCD 引擎（如 XGrammar, LLGuidance, Pre3） |

---

## 2. 核心实验方法和设置

> 注：本文以**理论推导为主**，未进行传统意义上的端到端任务实验（如 accuracy/benchmark 测试），而是通过构造性证明、复杂度分析与实证工具链支持验证。

### 数据集
- **合成语言构造**：
  - $ L = \{a^n b^n \mid n \geq 0\} $：用于展示状态空间膨胀（$ G_1 $ vs $ G_2 $）
  - $ L = \Sigma^* $（全语言）：使用 $ G_3: S \to aS \mid bS \mid \varepsilon $ 与 $ G_4: S_0 \to S \mid \varepsilon, S \to SS \mid a \mid b $ 对比 SAC 差异
- **真实 workload 支持**：
  - 引用 [JSONSchemaBench](https://arxiv.org/abs/2501.10868) 和 MaskBench 作为实证基准
  - 使用 [Nsight Systems](https://developer.nvidia.com/nsight-systems), [perf](https://linux.die.net/man/1/perf), [PyTorch Profiler](https://pytorch.org/tutorials/recipes/profiler_recipe.html) 等工具采集 trace

---

### 实验设置与评估指标

#### 设置
- 分析对象：不同结构但语言等价的 CFG 对
- 模拟环境：抽象的 bitset-style active-set engine 或 packed-forest engine
- 输入长度：从 $ t=1 $ 到 $ n $ 的逐步增长序列
- 控制变量：固定 vocabulary size、beam width $ B $、KV caching 条件

#### 评估指标
| 指标 | 描述 |
|------|------|
| $ K(G) $ | 编译后的控制状态数量（静态成本） |
| $ \text{SAC}_G(t) $ | 第 $ t $ 步的结构歧义成本（动态成本） |
| $ |\mathcal{P}_G(u)| $ | partial parse structures 数量（输出敏感检索成本） |
| $ T_{\text{step}}(t) $ | 单步总耗时（critical path） |
| $ T_{\text{mask}}(t) $ | 引擎侧掩码计算时间 |
| $ S(w) $ | SAC proxy（来自 instrumentation 的计数器向量） |

#### 基线方法对比
虽然没有直接列出 baselines，但隐含对比了以下典型策略：
- **Naive Earley/GLR parsing**：高歧义文法下产生指数级解析项
- **XGrammar-style persistent stack**：虽高效但仍受文法结构影响
- **Pre3’s DPDA compilation**：减少运行时探索，对应于缩小 active configuration fanout
- **标准 hard masking**：忽略 completion probability 差异，造成采样偏差

---

## 3. 主要实验结果和性能指标

### 关键性能数据与理论边界

| 文法 | 类型 | 每步 SAC | 累计成本 | 控制状态数 $ K(G) $ |
|------|------|----------|----------|------------------------|
| $ G_1: S \to aSb \mid \varepsilon $ | 右递归 | $ O(1) $ | $ O(n) $ | 8 |
| $ G_2: S \to aAb \mid \varepsilon, A \to aAb \mid \varepsilon $ | 冗余委托 | $ O(1) $ | $ O(n) $ | 15 (**↑87.5%**) |
| $ G_3: S \to aS \mid bS \mid \varepsilon $ | 线性无歧义 | $ O(1) $ | $ O(n) $ | — |
| $ G_4: S \to SS \mid a \mid b $ | 拼接歧义 | $ O(t^2) $ | $ O(n^3) $ | — |

> ✅ **Theorem 3**：任何 sound + parse-preserving + retrieval-efficient 引擎在 $ G_4 $ 上必须承受 $ \Omega(t^2) $ 每步更新成本。

> ✅ **Corollary 1**：$ G_4 $ 的累计 packed structure 增长为 $ O(n^3) $，而 $ G_3 $ 仅为 $ O(n) $。

> ✅ **Proposition 8**：Beam Search 下符号工作被放大 $ B $ 倍，即 $ \sum_i S(w_i) = O(B \cdot S) $

---

### 消融实验（Ablation Insights）

尽管无显式消融表，文中通过多个构造性对比实现“逻辑消融”：

| 改动 | 观察结果 |
|------|---------|
| 添加冗余非终结符（$ G_1 \to G_2 $） | $ K(G) $ 从 8 → 15，bitset 扫描开销上升 87.5% |
| 使用拼接规则 $ S \to SS $（$ G_3 \to G_4 $） | SAC 从 $ O(1) $ 恶化至 $ O(t^2) $，packed node 数量爆炸 |
| 移除 $ \varepsilon $-expansion 歧义（引入 $ S_0 $） | 避免空串无限推导，保证 parse forest 有限 |
| 应用 inlining rewrite | 减少中间非终结符，降低 $ K(G) $ 和潜在的 $ \text{SAC} $ |

---

## 4. 关键结论和发现

### 主要发现

1. 🔹 **语言等价 ≠ 解码效率等价**  
   即使两个 CFG 生成完全相同的字符串集合，其在 GCD 中引发的内部搜索空间、状态数、解析结构增长可以天差地别。

2. 🔹 **SAC 是衡量文法“友好程度”的关键指标**  
   右递归、线性文法具有恒定 SAC；而含有 $ S \to SS $ 结构的文法会导致二次增长，成为性能瓶颈。

3. 🔹 **硬掩码不是最优采样策略**  
   其忽略不同合法 token 后续成功完成的概率差异（survival probability spread），导致与真实条件分布 $ p(\cdot | T(y)\in L) $ 存在可量化的 KL 失真。

4. 🔹 **存在可优化的 canonical low-SAC 文法形式**  
   在局部重写范围内（如 inlining、recursion normalization），可以通过搜索找到最小 SAC 的等价文法。

5. 🔹 **SAC 可转化为可测量的运行时代理**  
   通过 instrumented engine 输出的计数器（如 Earley items 新增数），可构建 $ T_{\text{mask}} \sim a \cdot S + b $ 的预测模型，支持自动化调优。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **非端到端实验验证** | 缺乏在真实 LLM + downstream task 上的 latency/speedup 数据 |
| **假设理想 KV caching** | 忽略 attention computation 的细节变化 |
| **focus on worst-case** | 多数 bound 为 worst-case，平均情况需额外建模 |
| **implementation gap** | 最小 SAC 文法的存在性 ≠ 易于构造或可扩展搜索 |

---

### 未来工作方向

1. 🚀 **自动化 Grammar Optimizer Compiler**
   - 基于 equality saturation + e-graphs 实现文法重写与成本估计一体化
   - 支持 inlining、left-factoring、right-recursion normalization 等 rewrite rules

2. 🧪 **Empirical Validation Pipeline**
   - 构建 controlled ablation suite on JSONSchemaBench
   - 测量不同文法在 XGrammar / LLGuidance 下的实际 speedup

3. 🔄 **Hybrid Control: Symbolic + Neural Feedback**
   - 将 $ \text{CoReach}(u) $ 编码为 embedding 输入 MoE router 或 attention bias
   - 实现 syntax-aware dynamic computation routing

4. 📈 **Beyond Hard Masking: Soft Biasing with h-estimation**
   - 使用轻量模型估计 $ h(y_{<t}v) $ 并调整 logits，逼近 Doob h-transform
   - 减少因 uniform weighting 引入的采样偏差

5. ⚙️ **Integration with Tokenizer Alignment**
   - 联合优化 subword segmentation 与 terminal boundaries（参考 DOMINO）
   - 避免跨 token boundary 的语法断裂问题

---

## 总结

> ✅ **一句话总结**：  
> 本文建立了首个将 **文法结构、解析复杂性与 LLM 解码效率** 联系起来的严格理论体系，揭示了“相同语言可以有不同的代价”，并提出了 **SAC** 作为核心度量，推动 GCD 从“功能正确”走向“性能最优”。

> 💡 **实践启示**：  
> 在部署结构化生成时，不应随意编写 CFG，而应优先选择 **右递归、低歧义、紧凑结构** 的文法，并利用工具自动优化文法以降低 SAC 和 $ K(G) $，从而获得显著推理加速。

</details>

---

### 5. [ReflexiCoder: Teaching Large Language Models to Self-Reflect on Generated Code and Self-Correct It via Reinforcement Learning](https://arxiv.org/abs/2603.05863)

**Authors**: Juyong Jiang, Jiasi Shen, Sunghun Kim, Kang Min Yoo, Jeonghoon Kim, Sungju Kim  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.05863v1  

#### Abstract
While Large Language Models (LLMs) have revolutionized code generation, standard "System 1" approaches, generating solutions in a single forward pass, often hit a performance ceiling when faced with complex algorithmic tasks. Existing iterative refinement strategies attempt to bridge this gap at inf...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ReflexiCoder: Teaching Large Language Models to Self-Reflect on Generated Code and Self-Correct It via Reinforcement Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前主流的 **Large Language Models (LLMs)** 在代码生成任务中普遍采用“**System 1**”范式，即通过单次前向推理生成代码。尽管这类模型在简单任务上表现良好，但在处理复杂算法题时往往受限于一次性生成的局限性，容易产生逻辑错误或功能缺陷。

现有改进策略如 **re-ranking**、**external repairers** 或 **feedback-guided refinement**（如 Reflexion）依赖外部信号（如编译器反馈、测试用例执行结果或冻结模型的批评），存在以下问题：
- 依赖外部环境（execution engine），部署成本高；
- 推理延迟大，token 开销高；
- 难以泛化到缺乏完整测试用例的真实开发场景。

因此，如何让 LLM 在**不依赖外部反馈**的前提下，具备**内在的自我反思与纠错能力**，是本研究要解决的核心问题。

---

### **提出的新方法与创新思路**
作者提出了 **ReflexiCoder**，一个基于 **Reinforcement Learning (RL)** 的新型框架，其核心思想是将“**生成 → 反思 → 自我修正**”这一完整的推理轨迹（reasoning trajectory）内化到模型权重中。

#### **关键创新点：**
- ✅ **首次将 self-reflection 和 self-correction 轨迹作为 RL 优化目标**  
  不同于以往 RL 方法仅优化单次生成策略（single-pass generation policy），ReflexiCoder 将整个多步反思-修正过程建模为可学习的决策路径，并通过 RL 进行端到端优化。

- ✅ **完全自主的内在调试机制（intrinsic self-debugging）**  
  模型无需依赖外部执行环境或额外 critic 模型，即可完成 bug 检测、修复和代码优化，实现真正的“**全内省式推理**”。

- ✅ **RL-zero 训练范式 + 结构化奖励设计**  
  采用无监督微调（SFT-free）的 RL-zero 范式，在训练阶段利用细粒度奖励函数引导模型学会“如何调试”，而非仅仅“生成正确代码”。

- ✅ **高效的 token 利用机制**  
  实验表明，该方法不仅性能更强，反而比基线模型更 **token-efficient**，推理阶段平均减少约 **40% 的 token 消耗**。

---

### **相比现有方法的优势**
| 维度 | 传统方法（如 Reflexion） | ReflexiCoder |
|------|--------------------------|-------------|
| 是否依赖外部反馈 | 是（需执行引擎/测试用例） | 否（完全内省） |
| 推理开销 | 多轮 prompt-response 循环，高延迟 | 单一 prompt 内完成多步推理，低延迟 |
| 泛化性 | 依赖高质量测试用例 | 更适用于真实世界缺乏测试的场景 |
| 学习目标 | 仅优化最终输出正确性 | 优化整个“思考-修正”认知轨迹 |
| 效率 | 易陷入冗余迭代或震荡 | 学会“最优轨迹”：一次成功为主，最多一次优化 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
训练数据来源于开源项目 **DeepCoder** 的语料库，包含：
- **TACO-Verified**: 7,436 个经过验证的编程问题
- **LiveCodeBench**: 599 个去污染后的挑战性编程题
- **CodeForces**: 6,128 个竞赛级题目
- **LeetCode**: 2,641 个常见算法题

所有训练数据均经过质量过滤与去重处理，确保与主流评测基准无重叠。

---

### **实验设置与评估指标**

#### **模型架构**
- 基础模型：**Qwen3-8B**
- 训练方式：两轮 RL 训练，使用 **TRL (Transformer Reinforcement Learning)** 框架
- GPU 配置：8×NVIDIA H200

#### **评估配置**
定义两种推理模式以公平比较：
- **ReflexiCoder-8B (Single)**：移除系统提示（system prompt），禁用反思循环，用于与基线在相同 token 预算下对比
- **ReflexiCoder-8B (Multiple)**：启用完整反思-修正流程，允许最多 5 次迭代

#### **评估指标**
- 主要指标：**pass@1**（首次生成即通过所有测试）
- 评测工具：**EvalChemy**（统一评估框架，避免因工具差异导致偏差）

#### **系统提示（System Prompt）**
强制要求模型遵循 `<think> → <answer> → <reflection> → <answer>` 的结构化输出格式，并引入状态标记：
- `STATUS: BUG_DETECTED`：触发修正
- `STATUS: OPTIMIZATION_ONLY`：触发一次且仅一次优化后终止

---

### **基线方法对比**
涵盖多个代表性模型类别：
| 类别 | 基线模型 |
|------|--------|
| 开源通用模型 | Qwen3-8B, Qwen2.5-Coder-7B, Seed-Coder-8B |
| 闭源商用模型 | GPT-5.1, GPT-4.1, Claude-Sonnet-4.5, Gemini-2.5-Pro |
| RL 增强模型 | Ledex-RL-13B, DeepCoder-14B-Preview |
| 其他代码模型 | DeepSeek-Coder-7B, CodeGemma-7B, CodeLlama-7B |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（pass@1 %）**

| 模型 | HumanEval | HumanEval+ | MBPP | MBPP+ | BigCodeBench | LiveCodeBench | CodeForces |
|------|-----------|------------|------|-------|---------------|----------------|------------|
| **GPT-5.1** | 95.12 | 87.20 | 84.00 | 79.10 | 39.56 | 48.03 | 34.70 |
| **ReflexiCoder-8B (Single)** | **94.51** | **87.20** | **81.80** | **78.57** | **35.00** | **52.21** | **37.34** |
| **ReflexiCoder-8B (Multiple)** | **95.73** | **87.80** | **82.00** | **79.10** | **36.84** | **54.12** | **37.68** |

> 💡 **说明**：  
> - 在 **Single** 设置下，已超越多数开源模型，接近甚至持平 GPT-5.1；
> - 在 **Multiple** 设置下，全面超越 GPT-5.1，尤其在高难度任务（如 LiveCodeBench 和 CodeForces）上优势显著。

---

### **与基线方法的对比结果**
- 相比基础模型 **Qwen3-8B**，**ReflexiCoder-8B (Single)** 在：
  - **LiveCodeBench** 上提升 **+14.46%**
  - **CodeForces** 上提升 **+13.64%**
- 相比更大规模的 **DeepCoder-14B-Preview**（参数多 75%）：
  - 在 **LiveCodeBench** 上领先 **+18.16%**
  - 在 **CodeForces** 上领先 **+23.10%**
- 在 **8B 规模**下达到甚至超越 **GPT-5.1** 表现，证明小模型也能通过 RL 实现“认知跃迁”。

---

### **消融实验结果（Ablation Study）**

| 方法变体 | HumanEval | BigCodeBench | LiveCodeBench | CodeForces |
|---------|-----------|--------------|----------------|------------|
| **Full (完整版)** | 94.51 | 35.00 | 52.21 | 37.34 |
| w/o Format Gating `F(T)` | 84.75 (-9.76) | 32.02 | 39.07 | 24.81 |
| w/o Cycle Regulation `P(n)` | 92.68 | 33.68 | 52.09 | 35.84 |
| w/o Efficiency Reward `E(n)` | 91.46 | 33.42 | 42.41 | 29.92 |
| w/o Progressive Improvement `mt` | 93.29 | 34.74 | 39.19 | 34.10 |

> 🔍 **结论**：
> - **格式门控 `F(T)` 最关键**：缺失导致性能暴跌，说明结构化输出对学习至关重要；
> - **效率奖励 `E(n)` 显著影响高阶任务表现**：缺少则模型倾向于无效迭代；
> - 所有组件协同作用，缺一不可。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **内化的 self-reflection 是可行且高效的**  
   通过 RL 可教会 LLM 在没有外部反馈的情况下进行有效的自我调试，形成类似人类程序员的“内部审查”机制。

2. ✅ **优化“轨迹”优于只优化“结果”**  
   传统 RL 仅关注是否通过测试，而 ReflexiCoder 优化的是“从错误到正确的演化路径”，从而提升了根本的推理能力。

3. ✅ **更高效而非更昂贵**  
   尽管支持多步推理，但模型学会了“快速定位核心逻辑 + 极简修正”，实际 token 消耗更低（见下表）。

4. ✅ **小模型也能挑战大模型**  
   8B 模型在复杂任务上超越 14B+ 的 DeepCoder 和部分闭源模型，显示 **RL 内省机制具有强大的缩放潜力**。

---

### **Token 消耗分析（效率优势）**

| 模型 | 平均总 token 数 (HumanEval) | 平均推理 token 数 | 反思次数分布 |
|------|----------------------------|--------------------|--------------|
| Qwen3-8B | 4,170 | 3,134 | — |
| ReflexiCoder (Single) | 3,455 | 2,701 | 0 次反思 (164/164) |
| **ReflexiCoder (Multiple)** | **2,215** | **1,743** | **1 次反思 (164/164)** |

> 📉 **关键洞察**：  
> - **ReflexiCoder (Multiple)** 虽然进行了反思，但总 token 更少；
> - 因为它学会了“精准思考”，减少了冗余语言输出；
> - 几乎所有任务都精确执行 **一次反思即终止**，体现高度纪律性。

---

### **局限性**
1. ⚠️ **仍受 token 预算限制**  
   尽管更高效，但在极低延迟场景下，多步推理仍可能带来轻微延迟。

2. ⚠️ **聚焦单文件算法任务**  
   当前方法主要针对函数级代码生成，尚未扩展至跨文件重构、依赖管理或多模块协作等仓库级开发任务。

3. ⚠️ **依赖单元测试衡量正确性**  
   对无法被测试覆盖的非功能性需求（如安全性、可维护性）支持有限。

4. ⚠️ **迁移性待验证**  
   当前基于 Qwen3 系列训练，是否能无缝迁移到其他 base model 或编程语言尚需进一步研究。

---

### **未来工作方向**
- 🔮 扩展至 **multi-file context** 和 **tool-augmented programming**（如调用 Git、IDE API）
- 🔮 引入 **interactive specification evolution** 支持动态需求变更
- 🔮 探索 **cross-language generalization** 的内省能力
- 🔮 构建 **repository-level self-improvement agent**，实现长期自演进代码系统

---

## **总结**
ReflexiCoder 成功地将“**自我反思与修正**”这一高级认知能力内化为 LLM 的固有技能，标志着代码生成从“被动翻译”迈向“主动调试”的重要一步。其实验结果不仅刷新了开源模型在多个基准上的 SOTA，更重要的是展示了 **RL 在构建自主智能体方面的巨大潜力**——未来的代码模型不再是“写完就交卷的学生”，而是会“检查作业、发现问题、自己改错”的成熟工程师。

</details>

---

### 6. [Implicit Style Conditioning: A Structured Style-Rewrite Framework for Low-Resource Character Modeling](https://arxiv.org/abs/2603.05933)

**Authors**: Chanhui Zhu  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.05933v1  

#### Abstract
Large Language Models (LLMs) have demonstrated impressive capabilities in role-playing (RP); however, small Language Models (SLMs) with highly stylized personas remains a challenge due to data scarcity and the complexity of style disentanglement. Standard Supervised Fine-Tuning (SFT) often captures ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Implicit Style Conditioning: A Structured Style-Rewrite Framework for Low-Resource Character Modeling*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在低资源场景下，小型语言模型（SLMs）在角色扮演（Role-Playing, RP）任务中难以生成风格一致、符合人物设定的对话。主要原因包括：

- **数据稀缺**：大多数虚构角色仅有少量语料（如动漫角色），不足以训练鲁棒的风格化生成模型。
- **风格解耦困难**：角色风格是高维且复杂的，涉及词汇偏好（lexical）、句法模式（syntactic）和语用倾向（pragmatic），标准监督微调（SFT）往往只能捕捉表层语义，导致“Out-Of-Character”（OOC）生成。

### **提出了什么新方法或新思路**

作者提出了一种 **Structured Style-Rewrite Framework**，其核心创新如下：

1. **结构化风格表示（Structured Style Representation）**  
   将角色风格显式解耦为三个可解释维度：
   - **Lexical**：通过 TF-PMI 提取角色特异性关键词（如“喵”、“沐沐”）；
   - **Syntactic**：基于 PCFG 规则统计句法结构频率，压缩为 13 维向量；
   - **Pragmatic**：利用多标签分类器预测语用风格分布（如“cute”, “energetic”）。

   最终构建一个统一的结构化风格向量 $ S = [L, S, P] $，实现细粒度、可解释的风格控制。

2. **上下文感知的风格精炼器（Context-Aware Style Refiner）**  
   在少样本条件下，伪标签噪声严重。该模块结合聚类原型和上下文嵌入，修正初始风格标签，提升监督信号可靠性。

3. **重写式数据增强（Rewrite-Based Data Augmentation）**  
   构建一个可扩展的重写流水线，将中性句子转换为角色风格化文本，生成大规模、风格一致的合成训练数据。

4. **隐式风格条件化（Implicit Style Conditioning via CoT Distillation）**  
   引入 Chain-of-Thought（CoT）蒸馏策略，在训练阶段注入显式推理链（如 `<think>... </think>`），作为强归纳偏置，使模型内化风格决策逻辑。**在推理时无需输出 CoT，即可实现高质量风格生成**，降低部署开销。

### **相比现有方法的优势**

| 方法 | 缺陷 | 本文优势 |
|------|------|----------|
| Vanilla SFT | 风格模仿以牺牲语义保真度为代价，易产生语义漂移（semantic drift） | 显著提升 Valid Style Score，同时保持高 Semantic Score |
| Prompt-based RP | 受限于提示工程，风格不稳定，输出方差大 | 结构化风格向量提供稳定控制，减少对提示的依赖 |
| Retrieval-Augmented Generation (RAG) | 倾向于直接复制检索结果，导致语义坍塌（semantic collapse） | 生成新颖表达而非记忆复现，兼顾语义与风格一致性 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **主训练数据**：基于 `ChatHaruhi-Expand-118K`（Li et al., 2023）构建的合成平行语料（Neutral, Stylized）对，共 **5,786 对原始样本**，经重采样后达 **6,997 对**。
- **测试数据**：
  - **Hybrid Test Set**（150 条）：
    - In-domain Daily Chat（42 条）：来自 LCCC 语料库的日常对话。
    - Cross-domain Stress Test（108 条）：来自未见过的角色（如 Raiden Shogun, Haruhi Suzumiya）的去风格化语句。
- **零样本泛化案例**：Frieren（N=25 极端冷启动设置）。

### **实验设置**

- **模型架构**：基于 Qwen-1.7B 进行 LoRA 微调。
- **训练目标**：
  $$
  \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{lm}} + \lambda_{\text{recon}} \mathcal{L}_{\text{recon}} + \lambda_{\text{style}} \mathcal{L}_{\text{style}}
  $$
  - $\mathcal{L}_{\text{lm}}$：语言建模损失（含 CoT 和输出）
  - $\mathcal{L}_{\text{recon}}$：句法重建损失（从隐藏状态还原 PCFG 向量）
  - $\mathcal{L}_{\text{style}}$：语用风格多标签分类损失

- **推理模式**：评估 `Model v2 (Inference-only)`，即关闭显式 CoT 输出。

### **评估指标**

| 指标 | 定义 |
|------|------|
| **Semantic Score** | 生成句与中性输入之间的 BGE-large-zh-v1.5 余弦相似度 |
| **Style Score (Raw)** | 生成句与目标角色风格质心的 RoBERTa 分类器相似度 |
| **Valid Style Score** | $ S_{\text{raw}} \times \mathbb{I}(Semantic > 0.75) $，惩罚语义偏离的风格得分 |
| **H-Score** | 调和平均数：$ 2 \cdot \frac{\text{Semantic} \cdot \text{Style}}{\text{Semantic} + \text{Style}} $ |
| **LLM-as-a-Judge** | 使用 DeepSeek-V3 对生成质量进行人工打分（1–5 分） |
| **Human Evaluation** | 四位熟悉动漫文化的标注者盲评，评分维度：语义忠实度、风格强度、整体质量 |

### **基线方法对比**

| 基线 | 描述 |
|------|------|
| **Baseline A (RAG+Few-shot)** | 检索增强 + 少样本提示，无训练 |
| **Baseline B (Per-Character SFT)** | 每角色单独微调，上限基准 |
| **Baseline C (Vanilla SFT)** | 多任务 SFT（同数据），无 CoT 或辅助损失，使用更大的 Qwen-4B |
| **Baseline D (Strong LLM Prompting)** | 使用 GLM-4.7 + 2-shot 提示，工业级大模型方案 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 表 2：自动评估结果（Hybrid Test Set）

| Model | Semantic | Style (Raw) | H-Score | Valid Style |
|-------|----------|-------------|---------|------------|
| **Model v2 (Inference-only)** | **0.88** | 0.63 | **0.69** | **0.57** |
| Baseline C (Vanilla SFT) | 0.71 | 0.76 | 0.71 | 0.36 |
| Baseline D (Prompting) | 0.77 | **0.87** | 0.79 | 0.52 |
| Baseline A (RAG) | 0.51 | 0.88 | 0.63 | 0.10 |

> ✅ **关键发现**：本文方法在 **Semantic Score 上显著领先**（0.88 vs. ≤0.77），且 **Valid Style Score 最高**（0.57），说明其在保持语义完整的同时实现了更可靠的风格迁移。

#### 图 4：Pareto Frontier 分析（高保真区域）

- 在 Semantic ≥ 0.75 的可用区域内，**Model v2 (Inference-only) 是唯一非支配点**，表明其在高语义保真前提下达到最优风格-语义平衡。

#### 表 4：人类评估结果（1–5 分）

| Model | Semantic | Style | Overall |
|--------|----------|--------|---------|
| **Model v2 (Inference-only)** | **4.24** | 3.51 | 3.47 |
| Baseline D | 4.03 | **4.40** | **3.88** |

> 📌 注：虽然 Baseline D 在风格强度上得分更高，但存在“创造性偏差”（creativity bias）——评委偏好夸张表达，即使伴随轻微语义幻觉。

#### 表 3：LLM-as-a-Judge 评分

| Model | Semantic | Style Logic | Naturalness |
|--------|----------|------------|--------------|
| **Model v2 (Inference-only)** | **4.29** | **2.86** | **3.00** |
| Baseline D | 4.40 | 3.89 | 4.03 |

> ⚠️ 注意：LLM 判断显示 Baseline D 存在更多 **语义幻觉** 和 **逻辑不一致**，验证了人工评估中的“风格溢价”现象。

### **消融实验结果（Table 5）**

| 模型变体 | Semantic | Style | Valid Style |
|----------|----------|--------|------------|
| Full Model v2 | 0.8387 | 0.5875 | **0.4385** |
| w/o Lexical | 0.8471 | 0.5312 | 0.4184 |
| w/o Pragmatic | 0.8344 | 0.5569 | 0.4135 |
| w/o Syntactic | 0.8356 | 0.5805 | 0.4253 |

> 🔍 发现：
> - 移除 **Lexical** 导致 Raw Style 下降最多 → 词汇是表面风格最直接标记；
> - 移除 **Pragmatic** 对 Valid Style 影响最大 → 语用一致性决定整体风格连贯性；
> - 移除 **Syntactic** 影响最小 → 模型已通过 CoT 内化句法习惯（“auto-completion”效应）。

---

## 4. 关键结论和发现

### **主要发现**

1. **结构化解耦 + CoT 蒸馏能有效提升低资源角色建模能力**  
   即使只有 25 条样本（如 Frieren 案例），也能提取可靠风格特征并实现高质量零样本风格迁移。

2. **隐式风格条件化优于显式提示控制**  
   CoT 训练使模型将复杂风格推理过程内化为参数知识，推理时无需额外推理 token，实现高效部署。

3. **语义保真是风格迁移的前提**  
   当前主流指标（如 Raw Style Score）容易奖励语义漂移的“风格化复读”，而 **Valid Style Score 更能反映真实可用性**。

4. **句法建模虽可放松推理输入，但在训练中不可或缺**  
   消融实验证明，显式句法监督有助于塑造稳健的潜在风格空间。

### **局限性**

- **长对话一致性不足**：当前框架聚焦单句重写，缺乏跨轮次记忆与风格演化机制。
- **微妙风格现象捕捉有限**：如讽刺、双关、文化隐喻等难以通过统计特征建模。
- **PCFG 维度选择依赖经验**：13 维映射未系统验证于所有风格分布。
- **源句非完全中性**：现有中文语料自带“网民风格”，影响风格纯粹性。
- **评估主观性强**：人类评价一致性仅为 fair level（Krippendorff’s α ≈ 0.4）。

### **未来工作方向**

- 引入动态风格向量，支持基于对话历史的风格演化；
- 探索自适应 PCFG 特征选择机制；
- 开发更客观的风格幻觉检测指标；
- 扩展至非虚构角色（如历史人物、专业人士）；
- 研究显式与隐式推理的混合部署策略，兼顾极端语境下的可控性与效率。

--- 

> 💡 **总结一句话**：  
> 本文提出了一种**数据高效、可解释、可部署**的低资源角色建模框架，通过**结构化风格解耦 + CoT 蒸馏**，让小模型也能学会“像角色一样思考与说话”，并在语义保真与风格表达之间取得帕累托最优。

</details>

---

### 7. [RouteGoT: Node-Adaptive Routing for Cost-Efficient Graph of Thoughts Reasoning](https://arxiv.org/abs/2603.05818)

**Authors**: Yuhang Liu, Ruijie Wang, Yunlong Chu, Bing Hao, Yumeng Lin, Shengzhong Liu, Minglai Shao  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.05818v1  

#### Abstract
Large Language Models (LLMs) excel at multi-step reasoning, yet increasing the structural complexity of inference does not consistently improve system-level returns. Methods such as Tree of Thoughts (ToT), Graph of Thoughts (GoT), and Adaptive Graph of Thoughts (AGoT) can boost accuracy on some benc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：RouteGoT: Node-Adaptive Routing for Cost-Efficient Graph of Thoughts Reasoning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **Graph of Thoughts (GoT)** 和 **Adaptive Graph of Thoughts (AGoT)** 虽然在复杂推理任务中提升了准确性，但存在以下问题：
- **计算成本高**：大量使用大模型（Large LLM）进行所有节点处理，导致 **token 消耗巨大、延迟高**。
- **收益不稳定**：更高的计算开销并不总是带来更高的准确率，有时甚至不如简单的 **Chain-of-Thought (CoT)** 或直接输入输出（IO）。
- **缺乏预算控制**：无法在用户指定的 token 预算下进行可控推理，难以部署于实际生产环境。

根本原因在于：**GoT 推理图中的节点难度高度异质（heterogeneous）** —— 规划（planning）和最终合成（synthesis）等全局步骤需要强模型，而许多中间子任务是局部且简单的，可用轻量模型高效解决。

---

### **提出的新方法：RouteGoT**
RouteGoT 是一个 **节点自适应路由框架（node-adaptive routing framework）**，核心思想是：
> **“按需分配”计算资源：对不同难度的节点动态选择合适的模型和策略，在满足预算的前提下最大化推理质量。**

#### **核心创新点**
1. **Node-Level Adaptive Routing（节点级自适应路由）**
   - 每个待处理的叶节点（leaf node）由一个 **router** 动态决定执行动作 `a ∈ {IO, CoT, Decompose}`。
   - 每种动作绑定不同的模型规模和提示方式：
     - `IO` → 小模型（Small），直接回答
     - `CoT` → 中模型（Medium），链式思考
     - `Decompose` → 大模型（Large），分解为子图

2. **Budget-Controlled Inference（预算控制推理）**
   - 引入 **Global Budget Scheduler**，强制遵守用户设定的总 token 预算 `B_total`。
   - 保留一部分预算用于最终合成（`B_syn`），防止因预算耗尽而无法聚合结果。
   - 动态调整图的扩展深度和分支宽度，避免无谓扩张。

3. **Learned Routing Modules（学习型路由模块）**
   - **Success Predictor**：预测每个动作的成功概率。
   - **Budget Predictor**：预测节点所需的最小计算等级（ordinal difficulty budget），分为 Low/Medium/High 三档。
   - **PolicyNet**：结合成功概率和预算限制，输出最优动作分布。

4. **Plan-Guided Fallback（计划引导回退机制）**
   - 当 `Decompose` 动作生成的子图超出剩余预算时，不完全放弃，而是将分解计划作为上下文，用轻量模型直接求解当前节点，**保留部分规划价值**。

---

### **相比现有方法的优势**
| 维度 | RouteGoT | AGoT / GoT | 说明 |
|------|--------|----------|------|
| **成本效率** | ✅ 极低 token 消耗 | ❌ 高昂 token 开销 | 减少冗余计算 |
| **预算可控性** | ✅ 支持硬性预算约束 | ❌ 成本不可控 | 更适合生产部署 |
| **推理灵活性** | ✅ 节点级动态决策 | ❌ 全局统一策略 | 更细粒度优化 |
| **鲁棒性** | ✅ 在低预算下仍保持高性能 | ❌ 低预算时性能骤降 | 更稳定可靠 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

#### **(1) 路由器训练池（Router Training Pool）**
- 包含来自 **12 个基准** 的 20,000 个实例，涵盖数学、多跳问答、常识推理等。
- 主要数据集包括：
  - **2WikiMultihopQA**, **MuSiQue**, **MATH**, **GSM8K**, **StrategyQA**, **TabFact** 等。
- 每个样本运行三种策略（IO/CoT/Decompose），记录正确性和 token 消耗，用于训练 router。

#### **(2) 最终评估基准（Evaluation Benchmarks）**
| 类别 | 数据集 | 特点 |
|------|-------|------|
| **高级推理** | GPQA (Diamond split) | 高难度多选题，需研究生水平知识 |
| **检索 & 多跳 QA** | HotpotQA, MoreHopQA, HybridQA | 多步推理，涉及文本和表格 |
| **探索性推理** | Game of 24, Crosswords | 需搜索和组合空间推理 |

---

### **实验设置与评估指标**

#### **模型配置**
- 使用 **Qwen3** 系列模型实现不同能力层级：
  - **Small (4B)** → IO
  - **Medium (8B)** → CoT
  - **Large (30B)** → Decompose / Synthesis
- 路由组件使用轻量级 **0.6B adapter**，降低路由开销。

#### **评估指标**
| 指标 | 定义 |
|------|------|
| **Accuracy (%)** | 正确率（Exact Match 或语义等价判断） |
| **Output Tokens** | 平均每查询输出 token 数（衡量成本） |
| **Input Tokens** | 输入 token 数（反映上下文开销） |
| **Latency (s)** | 单次推理平均耗时 |
| **Utility Regret** | 相对于最优策略的效用损失 |
| **Oracle Match Rate** | 达到最高效用的比例 |

---

### **基线方法对比**

#### **(1) 标准推理范式（Reasoning Baselines）**
- **IO**, **CoT**, **ToT**, **GoT\***, **AGoT**

#### **(2) 自适应路由基线（Adaptive Routing Baselines）**
- **Random Router**：随机选择动作
- **KNN-Router**：基于嵌入相似性检索最近邻并选择最佳专家
- **EmbedLLM**：选择预测成功率最高的模型
- **RouteLLM**：二元路由（弱/强模型）
- **RTR (Route-to-Reason)**：联合建模成功率与成本，优化效用函数

> 所有方法在相同的 **GoT-style executor** 中运行，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2 & Table 3）**

| 方法 | 平均 Accuracy ↑ | 平均 Output Tokens ↓ | 对比 AGoT 的提升 |
|------|----------------|---------------------|------------------|
| **AGoT** | 76.5% | 12,179 | 基线 |
| **RouteGoT** | **84.6%** | **2,583** | **+8.1pp, -79.1% tokens** |

#### **具体任务表现亮点**
- **GPQA**:  
  - RouteGoT: **65.7%** @ 3,352 tokens  
  - AGoT: 64.6% @ 12,179 tokens → **精度更高，成本仅 27.5%**
- **HotpotQA**:  
  - RouteGoT: **88.0%** @ 592 tokens  
  - AGoT: 72.0% @ 2,583 tokens → **+16.0pp, -77.1% tokens**
- **HybridQA**:  
  - RouteGoT: **91.0%** @ 700 tokens  
  - RTR: 68.0% → 显著优于其他路由方法

#### **推理速度（Table 3）**
- 在 **MoreHopQA** 上，RouteGoT 比 GoT* 快 **近 6 倍**（11.79s vs 70.38s）
- 比 AGoT 更快，同时更准确

---

### **与基线方法的对比结果**
- RouteGoT 在 **所有 7 个任务上均达到或超过最佳基线性能**。
- 相比 **RTR** 等成本感知路由方法：
  - **平均高出 9.8 个百分点准确率**
  - 在 **Crosswords** 上更快（37.09s vs 47.00s）
- 相比 **EmbedLLM**（只看成功率）：
  - 成本更低，且避免盲目扩展导致的预算超支

---

### **消融实验结果（Ablation Study, Table 4）**

| 变体 | HotpotQA Acc | Tokens |
|------|--------------|--------|
| **RouteGoT (Full)** | **88.0%** | **592** |
| w/o Budget Predictor | 86.0% | 2,438 |
| w/o BP + PolicyNet | 78.0% | 2,020 |

#### **分析**
- 移除 **Budget Predictor** 导致 token 消耗激增（×4），说明离散预算等级比连续回归更稳定。
- 进一步移除 **PolicyNet** 导致准确率大幅下降（-10pp），表明仅靠成功率不足以做出最优图结构决策。
- 结论：**三个学习模块协同作用至关重要**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **节点异质性是优化突破口**  
   图结构推理中，并非所有节点都需要大模型。识别简单子任务并用轻量模型处理，可显著降低成本而不牺牲性能。

2. **预算控制能提升鲁棒性**  
   在低预算场景下，RouteGoT 仍能保持较高准确率（如 GPQA 上 54.3%），远超 AGoT 和 CoT，证明其 **“深但精简”** 的推理策略有效。

3. **细粒度路由优于任务级路由**  
   现有路由方法多为任务入口处一次性决策，而 RouteGoT 在图内持续调度，能根据中间状态动态调整，实现更优的 **cost-accuracy trade-off**。

4. **计划不应因预算不足而浪费**  
   Plan-guided fallback 机制使得即使无法展开完整子图，也能利用已有规划信息进行本地求解，提高了资源利用率。

---

### **方法的局限性**
- **依赖预定义的动作空间 `{IO, CoT, Decompose}`**，扩展新策略需重新训练。
- **Budget Predictor 输出为离散等级**，可能不够精细。
- 当前实验集中在特定模型族（Qwen3），跨模型泛化能力有待验证。
- 对极端复杂的图结构（如深度 > 3）支持有限（受硬件限制）。

---

### **未来工作方向**
1. **Hierarchical Routing Pipeline**  
   引入顶层 gateway 判断是否需要启动图推理，对简单问题直接跳过 GoT 流程，进一步节省成本。

2. **强化学习驱动的动态策略生成**  
   让 router 学习生成新的推理模式，而非固定动作集合。

3. **端到端联合优化**  
   将 router 与 backbone LLM 联合微调，提升协同效率。

4. **支持更多模型和工具集成**  
   将外部 API、检索系统等纳入动作空间，构建更通用的推理路由框架。

---

> **总结一句话**：  
> **RouteGoT 通过“节点自适应路由 + 全局预算控制”，实现了比 AGoT 更准、更快、更省的图结构推理，在真实场景下的部署潜力巨大。**

</details>

---

### 8. [Provuse: Platform-Side Function Fusion for Performance and Efficiency in FaaS Environments](https://arxiv.org/abs/2603.06170)

**Authors**: Niklas Kowallik, Natalie Carl, Leon P\"ollinger, Wei Wang, Sharan Santhahanam, David Bermbach  
**Category**: cs.DC  
**Published**: 2026-03-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.06170v1  

#### Abstract
Function-as-a-Service (FaaS) platforms provide scalable and cost-efficient execution but suffer from increased latency and resource overheads in complex applications comprising multiple functions, particularly due to double billing when functions call each other. This paper presents Provuse, a trans...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Provuse: Platform-Side Function Fusion for Performance and Efficiency in FaaS Environments**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
- **FaaS（Function-as-a-Service）平台在复杂应用中存在显著的系统开销**，尤其是由函数间频繁调用导致的：
  - **高延迟**：远程函数调用引入网络通信、调度和初始化延迟。
  - **资源浪费**：每个函数运行在独立的容器/沙箱中，造成内存冗余。
  - **双重计费（double billing）**：同步调用链中的多个函数被分别计时计费，增加成本。

这些问题在由多个细粒度函数组成的组合型 FaaS 应用中尤为严重。

---

### ✅ 提出了什么新方法或新思路
提出 **Provuse** —— 一种**透明的、平台侧的自动函数融合（function fusion）机制**，其核心思想是：

- 在运行时动态识别频繁同步调用的函数对；
- 将这些函数“融合”到同一个执行单元（container）中，实现**本地内联调用**而非远程调用；
- 整个过程对开发者完全透明，无需修改代码或部署配置。

> 🔧 **关键技术组件**：
> - **Function Handler**：监控函数出口连接，检测是否为阻塞式（同步）调用。
> - **Merger**：接收到融合请求后，合并两个函数的文件系统，构建新的融合函数镜像并部署。

该设计实现了**执行合并（execution consolidation）**，同时保留逻辑上的函数边界。

---

### ✅ 相比现有方法的优势
| 维度 | 传统方式 / 客户端控制方案 | Provuse（平台侧方案） |
|------|--------------------------|------------------------|
| 控制主体 | 开发者手动标注或配置融合规则 | 平台自动决策，无需干预 |
| 透明性 | 需要修改代码或部署流程 | 完全透明，零代码改动 |
| 可扩展性 | 依赖特定编程模型或框架 | 支持通用 bring-your-own-code 模型 |
| 兼容性 | 多基于定制平台 | 已实现在 tinyFaaS 和 Kubernetes 上 |
| 优化粒度 | 多静态分析或离线决策 | 动态基于实际调用行为 |

👉 **核心优势**：将优化责任从开发者转移到平台，提升可用性和自动化程度，适用于大规模生产环境。

---

## 2. 核心实验方法和设置

### ✅ 使用了哪些数据集 / 应用场景
使用两个典型的 FaaS 微服务应用进行评估：

1. **TREE**  
   - 结构：二叉树形调用图，一侧为同步调用链，另一侧为异步分支。
   - 特点：简单但能清晰展示融合效果。

2. **IOT（Fusionize++ IoT 应用）**  
   - 场景：模拟物联网传感器数据分析流程。
   - 调用模式：输入 → 分析温度/空气质量/声音等 → 判断事故 → 触发响应。
   - 包含多个同步串行步骤，适合函数融合。

---

### ✅ 实验设置和评估指标

#### 🖥️ 实验环境
- **两种底层平台**：
  - **tinyFaaS**：轻量级边缘 FaaS 平台，低开销。
  - **Kubernetes + Knative/OpenFaaS 类架构**：代表主流云原生 FaaS 架构。
- **硬件配置**：
  - 测试主机（SUT）与客户端各运行于独立虚拟机（QEMU/KVM），4 vCPU + 16GB RAM。
  - 网络带宽：10 Gbit/s。

#### ⏱️ 请求负载
- 工具：`k6` 压测工具。
- 总请求数：10,000 次 HTTP 请求。
- 请求速率：恒定 5 req/s（与 prior work 保持一致）。

#### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **End-to-end Latency** | 端到端请求响应时间（重点关注中位数） |
| **RAM Usage** | 平台整体内存消耗（反映资源效率） |
| **Developer Transparency** | 是否需要更改代码或配置（作为设计属性而非测量项） |

#### 🔁 对比基线
- **Vanilla Deployment**：标准 FaaS 部署，无任何融合优化。
- **Function Fusion Enabled**：启用 Provuse 的融合机制。

> ❗ 所有对比均为同一应用在同一平台下的对照实验。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据汇总

| 应用 | 平台 | 中位延迟（Vanilla） | 中位延迟（Fusion） | **延迟降低** | RAM 减少 |
|------|-------|--------------------|---------------------|---------------|------------|
| IOT | tinyFaaS | 807 ms | 574 ms | **28.9%** | ~57% |
| TREE | tinyFaaS | 452 ms | 350 ms | **22.6%** | ~50% |
| IOT | Kubernetes | 815 ms | 551 ms | **32.4%** | ~57% |
| TREE | Kubernetes | 456 ms | 358 ms | **21.5%** | ~50% |

> 📈 **总体平均提升**：
> - **平均端到端延迟减少：26.33%**
> - **平均 RAM 使用量减少：53.57%**

---

### ✅ 与基线方法的对比结果
- 在所有测试场景下，**Provuse 显著优于 vanilla FaaS 部署**。
- 延迟下降趋势随融合事件逐步显现（见 Figure 5），表明动态融合有效且稳定。
- 内存节省直接来源于减少了并发运行的函数实例数量（合并后释放原容器）。
- Kubernetes 上收益略高于 tinyFaaS，可能因其初始调度开销更大，融合带来的相对增益更明显。

---

### ✅ 消融实验结果（如有）
论文未明确开展多变量消融实验，但通过以下方式间接验证有效性：

- **跨平台一致性验证**：在 tinyFaaS 和 Kubernetes 上均取得相似改进，说明方法具有良好的可移植性。
- **不同应用结构对比**：同步密集型（如 IOT）受益更多，异步主导型（如 TREE 的部分路径）增益较小，符合理论预期。
- **动态触发机制验证**：仅当检测到同步调用时才触发融合，避免无效合并。

👉 表明融合策略的有效性高度依赖于**调用模式特征**。

---

## 4. 关键结论和发现

### ✅ 论文的主要发现
1. **平台侧函数融合是一种高效且可行的优化手段**：
   - 可显著降低延迟和资源消耗，尤其适用于同步调用链密集的应用。
2. **自动、透明的运行时融合无需开发者参与即可带来性能提升**：
   - 是迈向“隐形优化”的重要一步，符合 serverless “免运维”理念。
3. **融合能有效缓解 double billing 问题**：
   - 合并后的函数共享执行时间，避免重复计费。
4. **该方法兼容主流容器编排系统（如 Kubernetes）**：
   - 具备向真实生产环境推广的潜力。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **信任域限制** | 融合需确保函数属于同一信任域，否则会削弱隔离性（security/isolation boundary）。 |
| **冷启动融合开销** | 融合过程涉及镜像重建与部署，有一定延迟，需后续调用摊销成本。 |
| **语言支持有限** | 当前原型仅支持 Python，其他语言需适配入口点监控机制。 |
| **不支持 BYOC（Bring-Your-Own-Container）** | 若用户自带完整容器，则平台无法访问内部代码结构以实施融合。 |
| **对异步工作流增益有限** | 异步或非阻塞调用不会触发融合，优化空间小。 |

---

### 🔮 未来工作方向
1. **扩展至多语言支持**（e.g., Node.js, Java, Go）——利用统一的 runtime hook 技术。
2. **支持混合部署模型**：
   - 探索 **hybrid optimization**，结合 bring-your-own-function-code 与 bring-your-own-container 的优势。
3. **融合策略智能化**：
   - 引入 ML 模型预测调用模式，提前预融合（proactive fusion）。
4. **与其他平台优化技术集成**：
   - 如与 **pre-warming**, **peak shaving**, **dependency sharing** 等协同优化。
5. **安全性增强机制**：
   - 在融合环境中引入轻量级隔离（如 WebAssembly、namespace 划分）以维持多租户安全。

---

## ✅ 总结

**Provuse** 成功展示了**平台侧自动函数融合**在提升 FaaS 性能与效率方面的巨大潜力。它通过动态合并高频同步调用的函数，在不改变开发体验的前提下，实现了：

> 💡 **平均 26.3% 的延迟降低 + 53.6% 的内存节省**

这一成果凸显了**基础设施层透明优化**的价值，为构建更高性能、更低成本的 serverless 平台提供了新范式。未来若能在安全性、通用性和智能性方面进一步突破，有望成为下一代 FaaS 平台的标准特性之一。

</details>

---

### 9. [Parallelization Strategies for Dense LLM Deployment: Navigating Through Application-Specific Tradeoffs and Bottlenecks](https://arxiv.org/abs/2603.05692)

**Authors**: Burak Topcu, Musa Oguzhan Cim, Poovaiah Palangappa, Meena Arunachalam, Mahmut Taylan Kandemir  
**Category**: cs.DC  
**Published**: 2026-03-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.05692v1  

#### Abstract
Breakthroughs in the generative AI domain have fueled an explosion of large language model (LLM)-powered applications, whose workloads fundamentally consist of sequences of inferences through transformer architectures. Within this rapidly expanding ecosystem, dense LLMs--those that activate all mode...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Parallelization Strategies for Dense LLM Deployment**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本文聚焦于**密集型大语言模型（Dense LLM）在推理部署中的并行化策略选择难题**。随着 LLM 参数量增长至数百亿甚至数千亿级别（如 Llama-3.1-70B 和 -405B），单个 GPU 已无法容纳完整模型权重和运行时的 KV Cache，导致必须依赖多设备部署。然而，不同的 **Model Parallelism** 策略（如 TP、PP 及其混合）在 **Latency**（延迟）与 **Throughput**（吞吐量）之间存在显著权衡，且受输入长度、批处理大小、硬件配置等多重因素影响。

现有研究多集中于训练阶段的并行优化，而对**推理场景下并行策略的系统性实证分析仍不足**，尤其缺乏对应用特定目标（如低延迟 vs 高吞吐）的指导原则。

---

### **提出了什么新方法或新思路**
本研究并未提出全新的并行算法，而是通过构建一个高保真的 **in-house simulator**，系统性地量化和揭示了不同并行策略在真实推理负载下的行为规律。其核心创新在于：

- **首次系统性评估了 TP、PP 及其 Hybrid 在 Dense LLM 推理中的 Latency-Throughput 权衡关系**。
- 提出了 **“Latency Flexibility”** 概念，即系统在不同批处理规模下维持低延迟的能力，并证明 TP 显著优于 PP。
- 揭示了 **all-reduce 通信开销是 TP 中不可忽视的瓶颈**，尤其是在多节点扩展时。
- 发现 **PP 虽不改善延迟，但能有效提升吞吐上限**，因其缓解了内存压力，允许更大 batch。
- 强调 **Hybrid TP+PP 是实现可调节性能目标的关键手段**：通过控制 TP 度数调节延迟，通过 PP 深度提升吞吐。

---

### **相比现有方法的优势**
| 维度 | 传统做法 | 本文优势 |
|------|--------|--------|
| 分析粒度 | 定性描述或小规模实验 | **大规模仿真 + 多维度参数扫描**（batch size, seq len, parallelism degree） |
| 性能洞察 | 关注单一指标（如 throughput） | **联合分析 Latency 与 Throughput 的动态互作关系** |
| 应用导向 | 缺乏明确设计指南 | 提供 **面向 SLA 的并行策略选择框架**（低延迟选 TP，高吞吐选 PP，平衡选 Hybrid） |
| 模型覆盖 | 多为中小模型 | 聚焦 **前沿超大规模 Dense LLM（70B/405B）** |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验基于两类代表性数据集，分别模拟长短上下文任务：
- **LongAlpaca-12K**: 包含约 12K 样本，平均输入序列长度（ISL）达 **9092 tokens**，用于测试长上下文推理能力。
- **MLPerf Inference Dataset**: 包含 8313 个提示，平均 ISL 为 **9428 tokens**，代表工业级基准负载。
- **Short-context 组合数据集**：BBH + GSM8K + HumanEval，平均 ISL 约 **100 tokens**，模拟交互式问答、编程等短任务。

> 所有实验均使用各数据集的平均 ISL 和 OSL 进行建模。

---

### **实验设置和评估指标**

#### **模型**
- **Llama 3.1-70B** 和 **Llama 3.1-405B**
- 采用 **FP8 / 4-bit 量化** 以适应设备内存
- 支持最大 **128K 上下文窗口**

#### **硬件平台（模拟）**
- 单节点内配备 **8× AMD MI325x 或 MI355x GPU**
- GPU 间通过 **全连接 mesh interconnect**（带宽 128–153.6 GB/s）
- 使用自研 **Roofline-based simulator**，经硅片实测验证误差 < 18%

#### **并行策略**
- **Tensor Parallelism (TP)**：按层内算子切分（如 GEMM、Attention Head）
- **Pipeline Parallelism (PP)**：按 Transformer 层划分流水线阶段
- **Hybrid TP+PP**：组合使用，如 TP=2, PP=4

#### **评估指标**
| 指标 | 定义 | 用途 |
|------|------|------|
| **TTFT** (Time to First Token) | 用户请求到生成第一个 token 的时间 | 衡量 **响应延迟** |
| **TPOT** (Time Per Output Token) | 解码阶段每个输出 token 的平均耗时 | 衡量 **流式生成效率** |
| **TPS** (Tokens Per Second) | 单位时间内生成的总 token 数 | 衡量 **系统吞吐能力** |

---

### **基线方法对比**
- **NoPar**：无并行，仅限极小 batch
- **Pure TP**（TP=2/4/8）
- **Pure PP**（PP=2/4/8）
- **Hybrid TP+PP**（如 TP=2+PP=2, TP=4+PP=2）

所有配置均在相同硬件资源下进行公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **Latency 结果（TTFT & TPOT）**
- **TP 显著降低延迟**：
  - 对 Llama-3.1-70B，在 LongAlpaca 数据集上，**TP8 相比 TP2 实现 3.61× 更快的 TTFT 和 3.01× 更快的 TPOT**。
  - 对 Llama-3.1-405B，**TP8 相比 TP2 加速 3.67×（prefill）和 2.81×（decode）**。
- **PP 几乎不影响延迟**：
  - PP2/PP4/PP8 的 TTFT/TPOT 曲线几乎重叠，说明其无法加速单个请求处理。
- **Hybrid 中 TP 主导延迟表现**：
  - TP4_PP2 比纯 TP4 略慢，因引入了额外 P2P 通信开销。

#### **Throughput 结果（TPS）**
- **PP 显著提升吞吐**：
  - 对 Llama-3.1-405B + MLPerf 数据集，**PP8 达到比无并行方案高 13.8× 的 TPS**。
  - 吞吐增益主要来自更大的 **nano-batch size**（每 stage 可承载更多请求）。
- **TP 反而降低吞吐**：
  - 尽管 TP 加速单步计算，但 **all-reduce 开销增长更快**，导致整体 TPS 下降。
  - 如 TP8 的 TPS 明显低于 PP8 和数据并行基线。
- **Hybrid 平衡性能**：
  - TP2+PP4 在保持较低延迟的同时，获得接近纯 PP 的高吞吐。

---

### **与基线方法的对比结果**

| 策略 | Latency (TTFT/TPOT) | Throughput (TPS) | 内存利用率 |
|------|---------------------|------------------|------------|
| **NoPar** | 最差（仅支持小 batch） | 极低 | 低 |
| **TP** | ✅ **最优**（随 degree 提升） | ❌ 劣于 PP | 中等 |
| **PP** | ❌ 无改进 | ✅ **最优**（随 depth 提升） | ✅ 高 |
| **Hybrid TP+PP** | ✅ 接近 TP | ✅ 接近 PP | ✅ 高 |

> 示例：在 Llama-3.1-405B 上，PP8 支持最大 **512 的 global batch size**，而无并行仅支持 32。

---

### **消融实验结果**
- **TP 深度影响**：
  - 增加 TP degree 可持续降低 TTFT，但收益递减，且 **all-reduce 占 TTFT 比例稳定在 ~30%**，成为瓶颈。
- **链路带宽影响**：
  - 将 aggregate link speed 从 256GB/s 提升至 608GB/s，**TTFT 下降 34%**，表明高速互联对 TP 至关重要。
- **PP 阶段间通信开销**：
  - P2P 传输延迟仅占 TTFT 的 **<0.5%**，远小于 all-reduce 影响，解释为何 PP 对延迟影响微弱。

---

## **4. 关键结论和发现**

### **主要发现**
1. **TP 是降低延迟的首选策略**  
   通过将计算负载分散到更多 GPU 并利用并行计算能力，TP 显著缩短 prefill 和 decode 时间，适合对 **Latency 敏感的应用**（如对话系统、实时搜索）。

2. **PP 是提升吞吐的有效途径**  
   通过缓解 per-GPU 内存压力，PP 允许更大的 batch size，从而提高 GPU 利用率，适用于 **高吞吐离线任务**（如批量摘要、文档处理）。

3. **Latency 与 Throughput 存在根本性张力**  
   - TP 牺牲通信效率换取计算加速；
   - PP 牺牲流水线效率换取内存容量；
   - 二者难以兼得，需根据应用场景权衡。

4. **Hybrid TP+PP 提供可调的设计空间**  
   - 固定 PP 深度以保障吞吐潜力；
   - 调整 TP 度数以精细控制延迟；
   - 是实现 **SLA 可定制化 LLM 服务** 的理想架构。

5. **all-reduce 是 TP 的关键瓶颈**  
   尤其在多节点扩展时，跨节点 all-reduce 将成为主导延迟的因素，亟需更高效的通信原语（如 fusing ops, topology-aware collectives）。

6. **PP 吞吐提升受限于 compute-bound 现象**  
   当 batch 增大到一定程度后，decode 阶段变为 compute-bound，进一步增大 batch 不再提升 TPS，反而增加延迟。

---

### **方法的局限性**
- **依赖模拟器而非真实部署**：虽经校准，但仍可能忽略某些底层调度、缓存效应或软件栈开销。
- **未考虑异构硬件**：假设所有 GPU 性能一致，未涉及 CPU offload 或混合精度协同。
- **聚焦 Dense LLM**：未深入探讨 MoE 模型中的 Expert Parallelism 与 TP/PP 的交互。
- **静态批处理假设**：未模拟动态批处理（Dynamic Batching）或连续批处理（Continuous Batching）下的复杂排队行为。

---

### **未来工作方向**
1. **扩展至 Multi-node Systems**  
   研究跨节点 TP/PP 的通信拓扑优化、带宽约束下的性能建模。

2. **结合 Dynamic Batching 与 Parallelism 设计**  
   探索如何在动态请求到达模式下自适应调整并行策略。

3. **集成更先进的并行范式**  
   如 **Sequence Parallelism**（处理长序列）、**Expert Parallelism**（用于 MoE 模型），并与 TP/PP 协同优化。

4. **开发自动化并行策略推荐引擎**  
   基于应用 SLA（如 max TTFT ≤ 500ms, min TPS ≥ 10k）自动推荐最优 TP/PP/Hybrid 配置。

5. **探索通信压缩与融合技术**  
   减少 all-reduce 数据量或将其与其他操作融合，以突破 TP 的通信瓶颈。

--- 

> **一句话总结**：  
> 该论文揭示了 Dense LLM 推理中 **TP 优在 Latency，PP 优在 Throughput** 的基本规律，并指出 **Hybrid 并行是实现灵活性能调控的核心路径**，为构建高效、可定制的 LLM 服务平台提供了坚实的理论依据与实践指南。

</details>

---

### 10. [SAHOO: Safeguarded Alignment for High-Order Optimization Objectives in Recursive Self-Improvement](https://arxiv.org/abs/2603.06333)

**Authors**: Subramanyam Sahoo, Aman Chadha, Vinija Jain, Divya Chaudhary  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.06333v1  

#### Abstract
Recursive self-improvement is moving from theory to practice: modern systems can critique, revise, and evaluate their own outputs, yet iterative self-modification risks subtle alignment drift. We introduce SAHOO, a practical framework to monitor and control drift through three safeguards: (i) the Go...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SAHOO: Safeguarded Alignment for High-Order Optimization Objectives in Recursive Self-Improvement

---

## 1. 论文的主要贡献和创新点

### 解决的问题
论文聚焦于**递归自改进（Recursive Self-Improvement, RSI）系统中的对齐漂移（alignment drift）风险**。尽管现代大模型已具备自我批评、修改和评估输出的能力，但在多轮迭代优化中，系统可能在提升能力的同时逐渐偏离原始对齐目标（如事实性、安全性、格式正确性），这种**细微且累积性的漂移难以察觉但可能导致严重后果**。

### 提出的新方法与思路
作者提出了 **SAHOO** 框架，一个用于监控和控制 RSI 过程中对齐漂移的**可部署、可测量、可验证的系统性框架**，其核心由三大互补机制构成：

1. **Goal Drift Index (GDI)**  
   一种**多信号融合的漂移检测器**，结合语义（semantic）、词法（lexical）、结构（structural）和分布（distributional）四个维度的信息，通过信息论散度与学习权重构建统一漂移指数，实现对齐状态的量化监测。

2. **Constraint Preservation Checks**  
   引入显式的**约束保持损失函数**，确保关键安全属性（如语法正确性、非幻觉生成）在整个改进周期中不被牺牲。任何违反都将触发惩罚机制，并作为硬性停止条件。

3. **Regression-Risk Quantification**  
   设计**回归风险评估机制**，预测当前改进是否可能导致质量回退，防止“进步-倒退”循环，保障长期稳定性。

此外，所有参数（如漂移阈值、组件权重）均从校准数据中**数据驱动地学习**，而非依赖人工设定，增强了普适性和适应性。

### 相比现有方法的优势
| 维度 | 现有方法局限 | SAHOO 的优势 |
|------|----------------|---------------|
| **理论基础** | 多为抽象理论（如 Gödel Machine），缺乏可操作实现 | 将对齐保护转化为**可计算、可监控、可干预的实际流程** |
| **漂移检测** | 单一维度（如仅分布偏移）易遗漏复杂漂移 | 四维联合检测，捕捉语义等深层变化 |
| **约束处理** | 软性正则化或事后修正 | 显式硬约束 + 违规即停（zero-tolerance） |
| **长期稳定性** | 缺乏对回归与震荡行为的建模 | 引入 Regression Risk 和 Capability Alignment Ratio（CAR）进行动态权衡分析 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在三个代表性任务领域共 189 个任务上进行评估：
- **HumanEval**（63 tasks）：代码生成，测试语法正确性和语义保真度。
- **TruthfulQA**（63 tasks）：真实性判断，衡量事实准确性 vs 流畅性幻觉。
- **GSM8K**（63 tasks）：数学推理，测试多步求解与中间验证一致性。

> **校准阶段**：使用每个领域各 2 个任务（共 6 个任务 × 3 cycles = 18 个观测）进行参数学习与阈值校准。

### 实验设置
- **基础模型**：Qwen3-8B
- **最大迭代周期**：20 cycles
- **样本量**：每任务 63 次采样（经功效分析确定）
- **温度**：固定为 √2/2 ≈ 0.707（最大化熵）
- **Bootstrap**：约 2000 次重采样以估计 95% CI

### 评估指标
| 类别 | 指标 | 描述 |
|------|------|------|
| **质量提升** | Pass@1 / Accuracy / Exact Match | 各任务特定的质量得分 |
| **对齐漂移** | Goal Drift Index (GDI) | 多信号综合漂移指数，阈值设为 0.44 |
| **约束保持** | Constraint Preservation Score (CPS) | 满足约束的比例，关键约束零容忍 |
| **效率与权衡** | Capability Alignment Ratio (CAR) | 单位漂移带来的质量增益，越高越好 |
| **稳定性** | Regression Risk, Stability Score | 预测倒退概率与长期稳定程度 |
| **收敛性** | Convergence Rate, Mean Convergence Cycle | 改进是否趋于稳定 |

### 基线方法对比
文中未直接运行传统基线（如无防护的 RSI），而是通过以下方式间接比较：
- 对比“无漂移监控”的隐含基线：预期会出现高漂移、低 CPS。
- 对比“极端保守策略”：几乎无改进但零漂移，位于 CAR 前沿劣势区。
- SAHOO 成功实现了**高质量增益 + 低漂移 + 高 CPS** 的帕累托前沿区域。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 指标 | Code Gen | Truthfulness | Math Reasoning | Overall |
|------|----------|--------------|----------------|---------|
| 初始质量 | 0.672 | 0.678 | 0.689 | 0.680 |
| 最终质量 | 0.795 | 0.704 | 0.805 | 0.768 |
| **质量提升 (%)** | **+18.3%** | **+3.8%** | **+16.8%** | **+13.0%** |
| GDI（均值） | 0.320 | 0.354 | 0.330 | 0.335 |
| GDI 阈值 | 0.440 | 0.440 | 0.440 | — |
| CPS | 1.000 | 0.987 | 1.000 | 0.996 |
| CAR | 0.671 | 0.599 | 0.675 | 0.648 |
| 收敛率 | 93.7% | 87.3% | 92.1% | 91.0% |

> ✅ 所有领域 GDI 均远低于临界阈值（0.44），表明漂移得到有效控制。

### 与基线方法的对比结果
- **相比无防护 RSI**：SAHOO 在大幅提升能力的同时避免了灾难性对齐失败（zero catastrophic failures）。
- **相比保守策略**：SAHOO 实现了显著质量增益（最高达 +18.3%），而保守策略无法获得此类收益。
- **CAR 分析显示**：SAHOO 落在帕累托最优前沿，即在可控漂移下实现了高效改进。

### 消融实验与关键分析（Ablation Insights）

#### （1）漂移成分分解（Drift Component Analysis）
- **语义漂移**（Semantic）贡献最大（权重 0.38），其次是分布（0.29）、结构（0.21）、词法（0.12）。
- 表明**意义层面的变化是漂移主因**，表面词汇变化影响较小。
- 权重在校准后跨任务通用，说明该模式具有泛化性。

#### （2）约束违规模式（Constraint Violation Patterns）
- **仅 Truthfulness 领域出现违规**（共 170 次），Code 与 Math 实现完美 CPS（1.00）。
- 主要违规类型：
  - **Fabrication（编造）**：91 次（53.5%）
  - **Overconfidence（过度自信）**：48 次（28.2%）
  - **System-call misuse（系统调用滥用）**：15 次
- 表明**真实性改进面临更大对齐张力**，需针对性设计约束（如不确定性提示）。

#### （3）回归风险与稳定性
- 总体回归事件占比仅 **4.5%**（3,780 cycles 中 170 次）。
- 排除一个异常震荡任务后，回归频率降至 **0.7%**。
- 框架成功识别高风险任务并提前预警（平均第 4.5 cycle 触发干预）。

#### （4）CAR 动态演化
- **早期 cycle CAR 接近 1.0**，表示低成本高质量增益；
- 第 2–3 轮后下降至 0.6–0.7 并趋于稳定；
- 表明**早期改进性价比最高，后期需接受更高漂移成本**。

---

## 4. 关键结论和发现

### 主要发现
1. **递归自改进可在保持对齐的前提下实现显著能力提升**：SAHOO 在 Code 和 Math 上实现 >16% 提升，Truthfulness 上也有 +3.8%，同时将 GDI 控制在安全范围内。
2. **不同任务类型的对齐成本差异显著**：
   - Code/Math：改进高效、约束易保持（CPS=1.0）
   - Truthfulness：改进困难、漂移代价高（CAR=0.599），存在**流畅性与事实性的内在张力**
3. **早期改进周期最具性价比**：前几轮即可捕获大部分增益，建议设置较短 cycle limit（如 5–7）以平衡效率与安全。
4. **多信号漂移检测有效捕捉深层变化**：语义和分布漂移为主导因素，单一指标不足以反映真实对齐状态。
5. **回归风险可建模且可预防**：绝大多数任务未发生倒退，异常行为能被及时检测。

### 方法的局限性
1. **依赖初始对齐的基准模型**：若 baseline 本身已 misaligned，GDI 只会测量“从错误起点出发的漂移”，无法纠正根本价值错位。
2. **约束需显式定义**：对于伦理、价值观等难以形式化的属性，当前框架支持有限。
3. **人类评估依赖性强**：尤其在 Truthfulness 等任务中仍需 human-in-the-loop 标注，限制完全自动化扩展。
4. **对抗鲁棒性未知**：未测试恶意规避检测的行为（如欺骗性漂移隐藏）。
5. **适用于 moderate capability gain**：对于极高智能或潜在欺骗性系统，人类干预有效性可能失效。

### 未来工作方向
1. **更丰富的约束表达方式**：超越二元满足，引入软约束、优先级分级、动态调整机制。
2. **增强对抗鲁棒性**：研究 drift detection 系统在面对主动规避时的表现。
3. **结合 mechanistic interpretability**：深入理解模型内部表征漂移机制，实现更精准干预。
4. **多智能体协同自改进**：拓展至 multi-agent setting，探索群体演化下的对齐动力学。
5. **自动化约束发现**：利用 LLM 自动归纳潜在风险模式并生成约束规则。

---

> 🔚 **总结**：SAHOO 是首个将**对齐保护机制系统化、可量化、可部署**于递归自改进流程的工作。它不仅证明了“能力提升 ≠ 对齐丧失”的可能性，还提供了实用工具包（GDI、CPS、CAR、Regression Risk）帮助开发者在实际场景中安全推进 RSI 应用，为通向可控 AGI 提供了一条可行路径。

</details>

---

### 11. [ROSE: Reordered SparseGPT for More Accurate One-Shot Large Language Models Pruning](https://arxiv.org/abs/2603.05878)

**Authors**: Mingluo Su, Huan Wang  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.05878v1  

#### Abstract
Pruning is widely recognized as an effective method for reducing the parameters of large language models (LLMs), potentially leading to more efficient deployment and inference. One classic and prominent path of LLM one-shot pruning is to leverage second-order gradients (i.e., Hessian), represented b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ROSE: Reordered SparseGPT for More Accurate One-Shot Large Language Models Pruning**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
- **问题背景**：在大型语言模型（LLMs）的 **one-shot pruning** 中，**SparseGPT** 是一种高效的无训练剪枝方法，利用 Hessian 信息进行权重重建补偿，显著提升了剪枝后的模型精度。
- **核心问题**：SparseGPT 采用固定的从左到右（left-to-right）的 **block-wise pruning order**，当权重矩阵呈现 **columnar pattern**（列状分布）时，这种顺序会导致次优的重建效果。具体表现为：高误差的块若在后期被剪枝，可用于补偿的剩余参数更少，导致最终误差更大。

### **提出了什么新方法或新思路**
- **提出 ROSE（Reordered SparseGPT）**：
  - **核心思想**：**优先剪枝潜在误差更大的权重块**，使其在仍有较多可调参数时完成误差补偿。
  - **两阶段重排序（Two-level Reordering）**：
    1. **列级重排序（Column Reordering）**：在每个 block 内部，按列损失（column loss）降序排列。
    2. **块级重排序（Block Reordering）**：将所有 blocks 按块损失（block loss）降序排列。
  - **柱状层识别机制**：引入 **relative range of block loss** 作为指标，自动识别具有 columnar pattern 的层，并仅对这些层执行重排序，实现全模型自适应处理。

### **相比现有方法的优势**
- **更优的剪枝顺序**：通过重排序，使高误差块尽早被剪枝，获得更强的补偿能力。
- **无需额外训练或微调**：仍保持 one-shot、post-training pruning 范式，计算开销极低。
- **通用性强**：可扩展至 semi-structured pruning（如 2:4、4:8 模式）和量化联合压缩场景。
- **性能提升显著**：在多种 LLMs 和 sparsity 水平下均优于 SparseGPT 及其他主流方法。

---

## 2. **核心实验方法和设置**

### **使用的模型与数据集**
- **模型**：
  - **LLaMA2系列**：7B、13B、70B
  - **LLaMA3系列**：8B
  - **Mistral-7B**
- **数据集**：
  - **校准数据（Calibration Data）**：C4 数据集的前 128 个样本，每样本 2048 tokens，用于计算 Hessian 矩阵。
  - **评估数据集**：
    - **WikiText-2-raw**：用于计算 **Perplexity（↓越低越好）**
    - **Commonsense Reasoning 任务**：BoolQ、WinoGrande、PIQA、OpenBookQA、HellaSwag、ARC-Easy、ARC-Challenge，使用 **zero-shot accuracy（↑越高越好）**

### **实验设置和评估指标**
- **Sparsity Levels**：60% ~ 90%
- **Block Size**：128（unstructured），4（2:4），8（4:8）
- **评估指标**：
  - **Perplexity**：主指标，衡量语言建模能力
  - **Zero-shot Accuracy**：综合平均准确率
  - **Pruning Time**：实际运行耗时（分钟）
  - **End-to-end Latency**：推理延迟（ms）与加速比（Speedup）

### **基线方法对比**
| 方法 | 类型 | 特点 |
|------|------|------|
| **Magnitude** | Unstructured | 基于权重大小剪枝 |
| **Wanda** | Unstructured | 权重 × 激活范数 |
| **DSnoT** | Unstructured | 动态掩码 + 无训练微调 |
| **OATS** | Unstructured | 稀疏 + 低秩分解 |
| **SparseGPT** | Unstructured | Hessian-based 重建补偿（基准方法） |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **WikiText Perplexity 对比（Table 2 & 3）**
| Model | Sparsity | SparseGPT | **ROSE (ours)** | 改进 |
|-------|----------|-----------|------------------|------|
| LLaMA3-8B | 80% | 203.45 | **172.14** | ↓31.31 |
| Mistral-7B | 80% | 78.69 | **78.96** | ≈持平（略差但非显著） |
| LLaMA2-7B | 70% | 27.68 | **26.38** | ↓1.30 |
| LLaMA2-13B | 70% | 19.78 | **19.54** | ↓0.24 |
| LLaMA2-70B | 70% | 9.34 | **9.29** | ↓0.05 |

> 🔍 **观察**：在高稀疏度（如 80%）下，ROSE 的 perplexity 显著低于 SparseGPT，尤其在 LLaMA3-8B 上表现突出。

#### ✅ **Zero-shot Accuracy 对比（Table 3）**
| Model | Task | SparseGPT | **ROSE (ours)** | 改进 |
|-------|------|-----------|------------------|------|
| LLaMA2-7B | ARC-c | 40.19 | **41.71** | ↑1.52 |
| LLaMA2-7B | ARC-e | 40.38 | **41.35** | ↑0.97 |
| LLaMA2-7B | Avg | 45.43 | **46.43** | ↑1.00 |
| LLaMA2-13B | Avg | 50.51 | **50.75** | ↑0.24 |

> 📈 **趋势**：ROSE 在多数任务上取得更高准确率，尤其在推理类任务（ARC）中优势明显。

#### ✅ **Semi-structured Pruning 结果（Table 4 & 5）**
| Pattern | Model | SparseGPT | **ROSE (ours)** |
|--------|-------|------------|------------------|
| 2:4 | LLaMA3-8B | 16.33 | **15.84** |
| 4:8 | LLaMA3-8B | 12.20 | **12.00** |

> ✔️ ROSE 在 semi-structured setting 下依然优于 SparseGPT。

#### ✅ **消融实验（Ablation Study）**
- **Figure 4**：验证重排序策略有效性
  - **Block Reordering** 贡献最大，显著降低 reconstruction error。
  - **Column + Block Reordering** 效果最佳。
  - 若反向排序（先剪低误差块），性能反而下降 → 证明 **pruning order 至关重要**。
- **Figure 5**：鲁棒性分析
  - 不同 **blocksize**、**calibration samples**、**sequence length** 下，ROSE 始终优于 SparseGPT。

#### ✅ **运行效率分析（Table 6 & 7）**
| Model | Method | Pruning Time (min) |
|-------|--------|--------------------|
| LLaMA2-7B | SparseGPT | 4.76 |
| LLaMA2-7B | ROSE | **5.15**（+0.39 min）|

| Method | Latency (ms) | Speedup |
|--------|---------------|---------|
| Dense | 1791 | – |
| SparseGPT | 1458 | 1.23× |
| ROSE | **1450** | **1.24×** |

> ⚙️ **结论**：ROSE 仅增加约 **8% 的剪枝时间**，但推理延迟几乎不变，说明重排序操作不增加推理开销。

---

## 4. **关键结论和发现**

### **主要发现**
1. **Pruning Order Matters**：首次系统研究 one-shot pruning 中 **pruning order 的影响**，发现其对最终重建误差有决定性作用。
2. **Columnar Pattern 存在普遍性**：主流 LLMs 的 `self_attn.o_proj` 层普遍存在 **columnar weight distribution**，这是导致固定顺序剪枝失效的关键原因。
3. **Early Pruning of High-Error Blocks Improves Compensation**：越早剪枝高误差块，可用补偿参数越多，最终误差越小。
4. **Relative Range of Block Loss 是有效判据**：该指标能稳定区分 columnar 与 non-columnar 层，支持自适应重排序。

### **方法的局限性**
- **依赖预剪枝估计**：使用 Wanda 式的重要性分数进行预剪枝，可能在极端稀疏下产生偏差。
- **仅适用于 SparseGPT 框架**：当前设计紧密耦合 SparseGPT 的 block-wise 补偿机制，难以直接迁移到其他框架（如 OBC）。
- **未探索动态重排序**：重排序在剪枝前一次性确定，未考虑剪枝过程中的动态变化。

### **未来工作方向**
- 将 ROSE 思想推广至其他 **second-order pruning 方法**（如 OBC、OBS）。
- 探索 **layer-dependent block size** 或 **adaptive threshold** 策略。
- 结合 **quantization-aware pruning**，构建统一的压缩 pipeline。
- 研究 **pruning order optimization** 的理论基础，建立形式化模型。

---

> ✅ **总结一句话**：  
> **ROSE 通过引入“先剪高误差块”的重排序策略，在几乎不增加计算成本的前提下，显著提升了 SparseGPT 的 one-shot 剪枝精度，是 LLM 剪枝领域对 pruning order 的一次重要探索。**

</details>

---

### 12. [SPOT: Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning in Large Language Models](https://arxiv.org/abs/2603.06222)

**Authors**: Yunlong Chu, Minglai Shao, Yuhang Liu, Bing Hao, Yumeng Lin, Jialu Wang, Ruijie Wang  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.06222v1  

#### Abstract
Explicit Chain-of-Thought improves the reasoning performance of large language models but often incurs high inference cost due to verbose token-level traces. While recent approaches reduce this overhead via concise prompting or step pruning, they largely truncate what the model says rather than inte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SPOT: Span-level Pause-of-Thought for Efficient and Interpretable Latent Reasoning in Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **显式 Chain-of-Thought (CoT)** 虽然提升了 LLM 的推理能力，但其冗长的 token-level 推理轨迹导致高昂的推理成本（即“overthinking”）。
- 现有的**隐式/潜变量推理方法**存在两个关键缺陷：
  1. **刚性对齐（rigid point-to-point alignment）**：强制将一个 latent token 对齐到某个推理步骤的终点表示，无法捕捉整个 variable-length 推理段落的密集语义。
  2. **缺乏可解释性（lack of interpretability）**：latent states 通常通过无约束优化或 embedding mixing 得到，难以通过预训练语言头（pretrained LM head）解码为人类可读的“思想”。

### 提出了什么新方法或新思路
提出 **SPOT (Span-level Pause Of Thought)** 框架，实现高效且可解释的潜变量推理：

#### 核心创新点：
1. **Span-level Semantic Alignment**
   - 引入基于 **Sinkhorn 最优传输（optimal transport）** 的软匹配目标，将每个 `<pause>` token 与整个推理段落（span）的语义进行对齐。
   - 避免了传统方法中“一步一对齐”的刚性限制，能更鲁棒地建模 variable-length 推理片段。

2. **Frozen-Head Decoding Constraint**
   - 在训练过程中冻结预训练的 LM head 和 token embedding 矩阵，确保 `<pause>` 的隐状态可以直接通过该 head 解码为 token 分布。
   - 实现了 **原生可解释性（natively interpretable）**：可通过 `TopK` 解码获得 `<pause>` 所代表的关键词汇，如 `"multiply"`、`"64"` 等。

3. **灵活的两阶段训练范式（Two-stage Training）**
   - **Stage I**: 使用 SpanDrop 数据 + OT 对齐损失训练 `<pause>` 表示。
   - **Stage II**: 采用 Rejection-Sampled Fine-Tuning (RFT)，筛选正确且更短的答案进行微调，提升模型对外部插入 `<pause>` 的鲁棒性。
   - 支持在推理时动态控制 `<pause>` 插入密度，从而调节隐式推理强度。

4. **无需固定模板（No Fixed Template）**
   - 不强制要求 `<pause>` 与显式文本交替出现，允许灵活组合，增强了实用性。

### 相比现有方法的优势
| 维度 | SPOT | 现有方法 |
|------|------|---------|
| **压缩效率** | 显著减少生成 token 数量（↓37.5%） | 多数仅轻微缩短 |
| **准确性** | 平均提升 2.3 个百分点 | 多数方法精度下降 |
| **可解释性** | 可直接解码 `<pause>` 为关键词 | 多数 latent states 难以审计 |
| **灵活性** | 支持外部控制推理密度 | 多依赖固定模式 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **数学推理基准**：
  - **GSM8K**：小学数学应用题（1,319 测试样本）
  - **MATH500**：竞赛级多学科数学题（500 测试样本）
  - **AIME 2024 / 2025**：美国数学邀请赛题目（各 30 题）
- **跨领域科学问答**：
  - **GPQA-Diamond**：研究生水平科学选择题（198 测试样本），用于评估 OOD 泛化能力

### 实验设置和评估指标
- **主干模型**：DeepSeek-R1-Distill-Qwen-7B
- **训练数据**：仅在 GSM8K 训练集上训练
- **推理设置**：
  - 温度 = 0.6，top-p = 0.95
  - 最大生成长度 = 16,384 tokens
  - 每个样本采样 10 次取平均
- **评估指标**：
  - **Pass@1 Accuracy (Acc)**：一次采样的准确率
  - **输出长度 (#L)**：完整响应中的总 token 数（含推理和答案）

### 基线方法对比
分为两类：

#### 显式推理压缩方法（Explicit Trace Control）
- **CCoT**, **ConciseHint**, **Step Entropy**, **L1-Max**, **DEER**

#### 隐式/潜变量推理方法（Implicit/Latent Reasoning）
- **COCONUT**, **CODI**, **LightThinker**, **Latent-SFT**

> 所有方法均基于相同 backbone 进行公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）
| 方法 | 平均 Acc ↑ | 平均 #L ↓ |
|------|-----------|----------|
| **Vanilla** | — | — |
| **SPOT-stage2** | **+2.3%** | **-37.5%** |

#### 各数据集详细表现：
| 数据集 | Acc 提升 | Token 减少 |
|-------|--------|---------|
| **GSM8K** | +3.1 pp | -52.1% |
| **MATH500** | +1.4 pp | -43.0% |
| **AIME2025** | +3.3 pp | -15.8% |
| **GPQA-Diamond** | **+4.5 pp** | **-49.3%** |

> ✅ SPOT 在所有任务上均优于 Vanilla，并显著优于其他基线。

### 与基线方法的对比结果
- **相比显式压缩方法**（如 DEER、ConciseHint）：
  - 尽管这些方法也能减少长度（↓10~35%），但在困难任务（如 AIME）上常出现明显精度下降。
  - SPOT 在大幅压缩的同时仍保持甚至提升精度。
- **相比隐式推理方法**（如 CODI、COCONUT）：
  - 这些方法虽压缩更强（↓65~85%），但精度严重受损（↓10~30 pp）。
  - SPOT 实现了**最佳的 accuracy-length trade-off**。

### 消融实验结果

#### （1）对齐目标消融（Table 2）
| 对齐方式 | GSM8K Acc | AIME2025 Acc | #L |
|--------|----------|------------|-----|
| **Sinkhorn OT (SPOT)** | **92.72** | **39.33** | **630** |
| End_KL（仅终点对齐） | 87.30 | 21.33 | 10223 |
| MSE（池化后回归） | 88.93 | 15.67 | 12785 |

> ❌ 移除 OT 对齐会导致严重性能退化，验证了 span-level alignment 的必要性。

#### （2）压缩粒度（Spans per `<pause>`）（Table 3）
| G | Acc (MATH500) | #L ↓ |
|---|---------------|------|
| 1 | **93.80** | -43.0% |
| 2 | 90.62 | -55.6% |
| 3 | 89.84 | -57.5% |

> ⚠️ 更高压缩率（G>1）带来更大长度缩减，但以牺牲精度为代价；**G=1 是最优平衡点**。

#### （3）超参数敏感性分析（Table 4）
- 对齐权重 $\lambda$ 设置为 1.0 时效果最好。
- 当 $\lambda=0$（无对齐损失）时，Acc 明显下降，说明仅靠 CE 损失不足以学习 span 语义。

---

## 4. 关键结论和发现

### 主要发现
1. **SPOT 成功实现了高效且可解释的潜变量推理**：
   - 通过 **span-level alignment** 和 **frozen-head decoding**，既压缩了推理轨迹，又保留了语义完整性与可审计性。
2. **推理长度可控性强**：
   - 通过调整 `<pause>` 插入频率（每 N 个 span 插入一个），可在推理时灵活权衡速度与精度（见 Figure 3）。
3. **可解释性得到验证**：
   - **训练时诊断**：`<pause>` 的 TopK 解码词与对应教师 span 的词汇高度重叠（coverage 达 0.8+）。
   - **推理时评估（LLM-as-a-Judge）**：SPOT 的 `<pause>` 边界处 **Joint@4** 达到 **83.6%**，远高于 Vanilla（17.2%），表明其确实承载了有意义的推理跳跃。

### 方法的局限性
1. **依赖启发式分段规则**：
   - 当前使用 `\n\n` 作为空白行分隔符来定义 reasoning span，属于启发式设计，可能不适用于所有格式。
2. **极端密集插入可能导致重复行为**：
   - 如 Appendix A.6 所示，当 `<pause>` 插入过密时，模型可能出现 restart/repetition，破坏局部连贯性。
3. **未完全自动化 span 划分**：
   - 分段策略是固定的，尚未引入 learnable 或 task-adaptive 的边界检测机制。

### 未来工作方向
1. **推广到可学习的 span 边界**：
   - 替代当前的空白行分割，探索基于语义或注意力的动态分段策略。
2. **扩展至复杂任务场景**：
   - 应用于 **planning**、**long-horizon decision-making** 等需要长期记忆与多步抽象的任务。
3. **增强对插入模式的鲁棒性**：
   - 进一步优化 Stage II 训练策略，使模型更能适应多样化的 `<pause>` 插入方式。
4. **探索多模态潜变量推理**：
   - 将 SPOT 思路拓展至 vision-language 模型中的中间推理过程压缩。

--- 

> 📌 **总结一句话**：  
> **SPOT 通过 span-level pause token 实现了高效、灵活、可解释的潜变量推理，在大幅提升推理效率的同时还提高了准确性，为构建“聪明而不啰嗦”的语言模型提供了新范式。**

</details>

---

### 13. [Edge Intelligence-Driven LegalEdge Contracts for EV Charging Stations: A Fedrated Learning with Deep Q-Networks Approach](https://arxiv.org/abs/2603.06041)

**Authors**: Rahim Rahmani, Arman Chianeh  
**Category**: cs.DC  
**Published**: 2026-03-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.06041v1  

#### Abstract
We introduce LegalEdge, an edge intelligence-driven framework that integrates Federated Learning (FL) and Deep Q-Networks (DQN) to optimize electric vehicle (EV) charging infrastructure. LegalEdge contracts are novel smart contracts deployed on the blockchain to manage dynamic pricing and incentive ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题：** *Edge Intelligence-Driven LegalEdge Contracts for EV Charging Stations: A Federated Learning with Deep Q-Networks Approach*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统电动汽车（EV）充电系统面临以下挑战：
- **缺乏信任与透明度**：依赖中心化第三方（如 eMSP、Clearing House），存在单点故障和隐私泄露风险。
- **低效能源调度**：无序充电导致电网峰谷差加剧，增加基础设施投资压力。
- **智能决策能力不足**：现有系统难以实现动态定价、实时负载均衡和个性化激励机制。
- **法律与技术脱节**：智能合约（Smart Contract）虽可自动化执行，但缺乏法律可读性和可执行性，存在合规性盲区。

### 🚀 提出的新方法与创新思路
本文提出 **LegalEdge Contracts** —— 一种融合边缘智能（Edge Intelligence）、联邦学习（Federated Learning, FL）与深度强化学习（Deep Q-Networks, DQN）的新型区块链合约框架，用于优化 EV 充电管理。

#### 主要创新点包括：
1. **LegalEdge Contract 架构设计**
   - 融合 **Ricardian Contract（RC）** 与 **Smart Contract（SC）**，实现“法律条款可读 + 执行过程自动”的双重保障。
   - 合约具备人类可读文本（legal terms）与机器可执行代码（Solidity），确保法律效力与去中心化执行统一。

2. **边缘驱动的联邦强化学习机制**
   - 多个 EV 充电站作为 **Edge Node** 部署本地 DQN Agent，在不共享原始数据的前提下通过 FL 协同训练全局模型。
   - 利用 **Experience Replay + Target Network + ε-greedy** 提升 DQN 学习稳定性。
   - 支持 **Quantization-Aware Training (QAT)**，适配边缘设备资源受限场景。

3. **基于 DFA 的合约状态机建模**
   - 使用 **Discrete Finite Automata (DFA)** 形式化描述 LegalEdge Contract 的生命周期（如 Drafted → Signed → Active → Completed/Terminated）。
   - 明确定义事件触发的状态转移逻辑，增强合约执行的确定性与安全性。

4. **闭环反馈机制：智能合约驱动 RL 奖励函数**
   - 将合约中的奖惩规则（如按时完成充电奖励、违约罚款）编码为 RL 的 **Reward Function**，形成“策略优化 ↔ 合约执行”闭环。

### 🔍 相比现有方法的优势
| 维度 | 传统方案 | LegalEdge-FL |
|------|--------|-------------|
| 数据隐私 | 中心化收集，易泄露 | 本地训练，仅上传模型更新（FL） |
| 决策延迟 | 云端处理，高延迟 | 边缘部署，低延迟实时响应 |
| 法律合规性 | 智能合约法律模糊 | RC+SC 双重结构，合法可诉 |
| 系统弹性 | 单点故障风险高 | 去中心化架构，抗故障能力强 |
| 动态适应性 | 规则静态，难调整 | AI 自主学习 + Oracle 动态更新 |

---

## 2. 核心实验方法和设置

### 📊 实验环境配置

#### 硬件平台
- **Edge Devices**: 3 × Raspberry Pi 5 + 1 × Jetson Nano（模拟边缘节点）
- **Coordinator & Blockchain Node**: Ubuntu 服务器

#### 软件栈（见 Table 1）
| 层级 | 工具/框架 |
|------|---------|
| FL 协调 | Flower, PySyft |
| DQN 实现 | PyTorch + Stable-Baselines3 |
| 区块链 | Ethereum (Ganache 测试网) + Solidity + Web3.py |
| 模拟环境 | SimPy, Custom Python Simulator |
| 监控 | Prometheus + Grafana, TensorBoard |

### ⚙️ 实验设置与参数（见 Table 4）

| 参数 | 设置值 |
|------|-------|
| 客户端数量（Clients） | 5（含 EV 与 CS） |
| FL 轮数（Rounds） | 20 |
| 本地训练轮次（Local Epochs） | 1–5 |
| 批量大小（Batch Size） | 32–128 |
| 学习率（Learning Rate） | 1e-4 ~ 1e-3 |
| 优化器 | Adam |
| 聚合算法 | FedAvg（Flower NumPyClient） |
| 奖励函数来源 | 部署在 Ganache 上的 Solidity 智能合约 |
| 定价模式 | 正弦波形 surge pricing（模拟昼夜负荷变化） |
| 最大充电容量 | 100 kWh |

### 🎯 评估指标
1. **效率（Efficiency）**：单位时间内分配的能量 / 决策成本
2. **学习收敛性（Convergence）**：平均 TD Error 下降趋势
3. **交易速度（Transaction Speed）**：智能合约调用延迟（秒）
4. **合约完整性（Contract Integrity）**：状态一致性、异常触发次数、惩罚追踪准确性
5. **通信开销压缩率**：量化前后模型上传体积对比

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **Traditional FL (FedAvg)** | 标准联邦平均，无智能合约反馈 |
| **Centralized DQN** | 所有数据集中训练，存在隐私泄露风险 |
| **Non-QAT Edge DQN** | 未进行量化感知训练的边缘 DQN |
| **Static Rule-based Charging** | 固定时间/价格策略，无自适应能力 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Figure 10 与 Table 3）

| 指标 | LegalEdge-FL 表现 | 对比提升 |
|------|------------------|----------|
| **系统效率** | 从 55% 提升至 **>90%** | ↑ ~63.6% |
| **学习收敛速度** | TD Error 在第 20 轮趋于稳定，呈指数衰减 | 比传统线性收敛快 2–3 倍 |
| **交易延迟** | 平均 **0.12 秒/次**（标准差 ±0.01s） | 满足实时控制需求 |
| **合约完整性得分** | 平均 **0.98 / 1.0** | 几乎无运行时错误 |
| **通信压缩率** | 使用 int8 量化后，模型上传量减少 **~75%** | 显著降低带宽消耗 |

### 📊 与基线方法对比结果（见 Table 3）

| 特性 | Traditional FL | LegalEdge-FL |
|------|---------------|--------------|
| FL 算法 | FedAvg | DQN + FL (Flower) |
| 智能合约反馈 | 无 | ✔️ On-policy reward |
| 区块链延迟 | N/A | <0.15s avg |
| 效率扩展性 | ~65% | **>90%** |
| 收敛动态 | 线性 | **指数型（DQN 加速）** |

> ✅ 结果表明：LegalEdge-FL 在效率、收敛速度和系统响应方面全面优于传统方法。

### 🔍 消融实验分析（隐含于实验设计中）
虽然文中未明确列出消融表，但从实验设计可推断以下关键组件的作用：
- **Experience Replay 与 Target Network**：显著降低 TD Error 波动，提升学习稳定性。
- **QAT 训练支持**：使模型在 int8 推理下仍保持 >95% 的原始精度，适合边缘部署。
- **智能合约作为 Reward Source**：引导 Agent 学会遵守合约义务（如及时支付、按计划充电），提高履约率。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LegalEdge Contract 成功桥接“法律”与“代码”**
   - 实现了合约的“双可读性”（human-readable + machine-executable），解决了智能合约法律效力缺失问题。
   - 支持动态合规更新（Dynamic Compliance），可通过 Oracle 注入新规并自动调整合约行为。

2. **边缘联邦强化学习有效提升充电调度智能化水平**
   - DQN Agent 能够根据本地 SoC、电价、电网负载等条件自主选择最优充电策略。
   - FL 框架保证了数据隐私的同时实现了跨站点协同优化。

3. **低延迟区块链交互成为可能**
   - 在 Ganache 测试网上实现 **<0.15s 的平均交易延迟**，证明其适用于近实时控制场景。
   - 智能合约成功承担资金托管（Escrow）、自动结算与争议处理功能。

4. **QAT 与 FL 深度融合，推动边缘 AI 部署**
   - LegalEdge 原生支持 Quantization-Aware Training，使得轻量化模型可在树莓派等设备上高效推理。

### ⚠️ 方法的局限性
1. **测试环境为仿真系统**：尚未在真实城市级 EV 网络中验证，实际网络延迟与节点异构性可能影响性能。
2. **依赖可信 Oracle 输入**：尽管减少了对第三方的信任，但仍需 Oracle 提供电价、充电完成信号等外部数据，存在潜在攻击面。
3. **计算资源要求较高**：DQN + FL + Blockchain 三重叠加对边缘设备算力有一定门槛，低端设备可能难以承载。
4. **缺乏大规模多主体博弈分析**：当前实验规模较小（≤5 客户端），复杂市场环境下策略竞争有待研究。

### 🔮 未来工作方向
1. **引入 Differential Privacy 与 Secure Aggregation**：进一步增强 FL 过程中的隐私保护。
2. **集成自动法律推理引擎（Legal Reasoning Engine）**：实现合同条款的自然语言理解与动态解释。
3. **开展实地试点（Field Trial）**：在真实 EV 充电桩网络中部署 LegalEdge，评估长期稳定性与经济效益。
4. **探索 V2G 场景下的双向能量交易机制**：将 LegalEdge 扩展至 Vehicle-to-Grid 应用，支持反向馈电与电网支撑服务。
5. **构建 DAO 治理层**：利用去中心化自治组织（DAO）实现合约升级与争议仲裁的社区共治。

---

## 总结
> **LegalEdge** 是首个将 **Edge Intelligence、Federated Learning、Deep Q-Networks 与 Ricardian-Smart Contract 融合** 的 EV 充电管理系统。它不仅提升了系统的智能化、安全性和效率，更重要的是开创了一种“**合规-by-design**”（compliance-by-design）的新型数字契约范式，为未来智能交通、能源互联网与 Web3 基础设施提供了可复制的技术蓝图。

</details>

---

### 14. [Adapter-Augmented Bandits for Online Multi-Constrained Multi-Modal Inference Scheduling](https://arxiv.org/abs/2603.06403)

**Authors**: Xianzhi Zhang, Yue Xu, Yinlin Zhu, Di Wu, Yipeng Zhou, Miao Hu, Guocong Quan  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.06403v1  

#### Abstract
Multi-modal large language model (MLLM) inference scheduling enables strong response quality under practical and heterogeneous budgets, beyond what a homogeneous single-backend setting can offer. Yet online MLLM task scheduling is nontrivial, as requests vary sharply in modality composition and late...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Adapter-Augmented Bandits for Online Multi-Constrained Multi-Modal Inference Scheduling

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**多模态大语言模型（MLLM）在异构执行后端上的在线推理调度问题**，解决以下核心挑战：
- **任务表示难题**：真实请求在模态组合、输入规模和隐含推理难度上差异巨大，传统手工特征（如token数、模态指示符）难以准确建模其语义和成本。
- **动态预算约束下的决策难题**：在线调度需在不可逆的多维预算（如延迟、金钱成本）下进行低开销、非短视的决策，避免过早耗尽资源。

### 提出的新方法：M2-CMAB
提出 **M2-CMAB (Multi-modal Multi-constraint Contextual Multi-Armed Bandit)** 框架，包含三个核心组件：
1. **aCLS-attentive Predictor**：
   - 在冻结的MLLM主干网络上，引入 `[CLS]` token 并通过 **multi-head self-attention** 进行池化，提取紧凑且语义丰富的任务表示 `z`。
   - 仅更新轻量级的 **reward/cost adapters** 来预测不同动作下的奖励和成本，实现高效、动作特定的估计，同时保留本地后端的生成能力。
2. **Primal-Dual Constrainer**：
   - 维护在线 **Lagrange multipliers (λ)**，通过 **primal-dual update** 和 **Online Mirror Descent (OMD)** 动态调整，将长期累积的多维预算约束解耦为每轮可处理的目标。
3. **Two-Phase Scheduler**：
   - **初始阶段 (Initial Phase)**：通过探索收集数据，估算对偶变量的有效范围 `A`，为后续阶段提供稳定基础。
   - **探索-利用阶段 (Exploration-Exploitation Phase)**：结合预测器的输出和约束反馈，计算带惩罚的拉格朗日分数 `S(a)`，并基于此进行概率采样，平衡探索与利用。

### 相比现有方法的优势
- **高效表示**：利用冻结MLLM自身提取语义表示，比手工特征更鲁棒，且仅微调adapters，计算开销远低于全模型微调。
- **理论保障**：建立了在多维背包约束下的 **regret guarantee**，形式为 $Reg(T) \leq O(\sqrt{T})$，提供了算法性能的理论支撑。
- **解耦控制**：通过拉格朗日乘子将复杂的长期约束优化转化为简单的每轮目标，实现了高效的在线决策。

---

## 2. 核心实验方法和设置

### 数据集
构建了一个综合性的多模态基准测试，包含 **六个数据集**：
- **五个独立数据集**：涵盖多样任务和模态需求。
  - `InfoVQA`：知识密集型视觉理解
  - `GSM8K`：数学推理
  - `SimpleVQA`：多模态事实性问答
  - `CoQA`：多轮对话与阅读理解
  - `AI2D`：基于图表的视觉推理
- **一个复合数据集 (COMPOSITE)**：将上述五个数据集合并，形成大规模、异构的任务流，用于全面评估。

### 实验设置和评估指标
- **执行后端 (Backends)**：共 **5个异构后端**，覆盖本地和云端部署。
  - **本地**：`Qwen3-VL-2B-Instruct` (轻量级，低延迟)
  - **云端**：`GPT-5-nano`, `Qwen3-VL-32B-Instruct`, `Qwen3-VL-30B-A3B-Instruct`, `GLM-4.6V-Thinking` (大型、推理增强模型)
- **预算制度 (Budget Regimes)**：定义了三种预算水平（基于五种候选动作的聚合成本）：
  - **Restricted**：最低聚合成本
  - **Normal**：第二低聚合成本
  - **Generous**：中位聚合成本
- **评估指标**：
  - **主要指标**：**平均推理奖励 (average inference reward)**，即在所有成功执行的轮次中的平均响应质量得分。
  - **次要指标**：与Oracle上限的差距、消融研究。

### 基线方法对比
与多种代表性基线方法进行比较：
1. **Random**：随机选择后端。
2. **Latency-first**：贪心选择预测延迟最低的后端。
3. **Money-first**：贪心选择预测金钱成本最低的后端。
4. **BGT-planner**：先进的基于CMAB的预算分配框架。
5. **Threshold-based**：基于预估效用-成本比率选择动作。
6. **Optimal**：Oracle辅助的上限，使用完美的每轮观测信息做出决策。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **COMPOSITE** 数据集上，M2-CMAB 相比第二好的基线方法，在不同预算制度下实现了显著提升：
  - **Restricted** 制度下：**+6.79%** 更高的平均奖励
  - **Normal** 制度下：**+13.08%** 更高的平均奖励
  - **Generous** 制度下：**+14.18%** 更高的平均奖励
- M2-CMAB 的性能**最接近 Oracle-aided 上限**，证明了其有效性。

### 与基线方法的对比结果
- M2-CMAB **在所有数据集和所有预算制度下均一致优于所有现实基线**。
- 随着预算增加，M2-CMAB 的相对优势更加明显，表明其能更有效地利用额外的调度灵活性来优化性能。
- 即使在极端受限的场景下（如 `CoQA` 在 Restricted 设置），M2-CMAB 与 Oracle 的差距也**小于 1.2%**，展现了极强的鲁棒性。

### 消融实验结果
进行了移除关键组件的消融研究（ablation study）：
- **移除任何适配器（adapter）都会导致性能下降**，验证了各组件的重要性。
- **移除 Reward Adapter 导致的性能下降最为严重**，说明准确的奖励预测是有效调度决策的核心。
- 移除 Money 或 Latency Adapter 的影响相对较小，因为两者作为预算约束预测器，其不准确性可以在一定程度上相互补偿。

---

## 4. 关键结论和发现

### 主要发现
1. **MLLM自身是最佳的任务表示器**：利用冻结的MLLM主干网络提取的语义表示，结合轻量级adapters进行预测，是一种高效且鲁棒的在线调度方案。
2. **解耦的约束控制至关重要**：通过 primal-dual 框架将长期、多维的预算约束解耦，使得在线决策既简单又有效。
3. **M2-CMAB具有强大的实际性能**：在高度异构和复杂的MLLM推理负载下，该方法能够显著超越现有技术，并逼近理论最优性能。

### 方法的局限性
- **理论保证的开放性**：虽然建立了regret bound，但该界依赖于reward/cost predictors的estimation regret。将此保证扩展到更通用的MLLM-based predictors（如文中使用的）仍是一个开放问题（Remark 4.2）。
- **初始阶段开销**：初始探索阶段需要消耗一定比例的轮次，虽然实验证明其影响可控，但在超短期任务流中可能成为瓶颈。

### 未来工作方向
1. **建立MLLM-based estimators的在线regret保证**：为所提出的基于MLLM的预测器提供严格的理论分析。
2. **探索更轻量级和细粒度的表示**：进一步优化任务和执行后端的表示方式，以提高大规模部署时的鲁棒性和效率。

</details>

---

### 15. [A Lock-Free Work-Stealing Algorithm for Bulk Operations](https://arxiv.org/abs/2603.05766)

**Authors**: Raja Sai Nandhan Yadav Kataru, Danial Davarnia, Ali Jannesari  
**Category**: cs.DC  
**Published**: 2026-03-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.05766v1  

#### Abstract
Work-stealing is a widely used technique for balancing irregular parallel workloads, and most modern runtime systems adopt lock-free work-stealing deques to reduce contention and improve scalability. However, existing algorithms are designed for general-purpose parallel runtimes and often incur over...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Lock-Free Work-Stealing Algorithm for Bulk Operations

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **work-stealing queue**（如 C++ Taskflow 中的实现）虽然在通用并行运行时中表现良好，但在特定领域（如基于 **Decision Diagrams (DDs)** 的混合整数规划求解器）中存在以下问题：
- 缺乏对 **bulk operations**（批量操作）的原生支持，导致频繁的单节点 push/steal 操作开销大；
- 固定大小或动态扩容的数组结构（如 ring buffer）在大规模、不规则任务生成场景下效率低下；
- 支持多个并发窃取者（multiple stealers），引入不必要的同步复杂度，而实际应用中往往只需一个中心化的 **master** 进行负载均衡。

### 提出的新方法
作者提出了一种新的 **lock-free work-stealing queue** —— `LF_Queue`，专为 **master-worker 架构下的 DD-based solver** 设计，其核心创新包括：

- ✅ **原生支持 bulk push 和 bulk steal 操作**  
  可一次性插入或窃取一批任务节点，显著减少函数调用和原子操作次数。

- ✅ **无界增长（unbounded growth）且无需 resize**  
  基于 **singly linked list** 实现，避免了数组扩容带来的复制开销。

- ✅ **简化并发模型：最多两个线程访问队列（one owner + one stealer）**  
  利用 master 作为唯一 stealer 的特性，大幅降低同步需求，提升性能。

- ✅ **完全 lock-free 且保证线性可序列化（linearizable）**  
  所有操作通过原子指令实现，无锁设计确保高并发下的可扩展性。

### 相比现有方法的优势
| 特性 | 传统方法（如 Chase-Lev / Taskflow） | 本文方法（LF_Queue） |
|------|-------------------------------|------------------------|
| 批量操作支持 | 需模拟（多次单节点操作） | 原生支持 |
| 内存管理 | 数组需 resize 或分段管理 | 单链表，自然无界 |
| 并发模型 | 多个潜在窃取者，同步复杂 | 仅一个 stealer，轻量同步 |
| 性能特征 | 小批量高效，大批量延迟上升 | 推送延迟几乎恒定 |

---

## 2. 核心实验方法和设置

### 数据集与工作负载
由于目标求解器仍在开发中，未使用真实 solver workload，而是采用两种替代方案：

1. **Microbenchmark**：独立测试 `push`, `pop`, `steal` 操作的延迟。
2. **Pseudo Workload**：基于大规模有向无环图（DAG）的并行探索任务模拟，构建接近真实行为的工作负载。
   - 图规模：2.5M 节点 和 300M 节点
   - 每个 worker 维护私有的 work-stealing queue
   - 当本地队列为空时，由 master 窃取其他 worker 队列的一半任务

### 实验设置
- **硬件平台**：
  - CPU: Intel Xeon Platinum 8358 ×2，共 64 核 128 线程
  - 编译器: GCC 15, C++20, `-O3 -march=native`
- **评估指标**：
  - 各 API 操作的平均延迟（latency）
  - 整体执行时间（execution time）
  - 可扩展性（speedup 随线程数变化）

### 基线方法对比
- `TF_BD_Queue`: Taskflow 的 **bounded deque**（固定大小 ring buffer）
- `TF_UB_Queue`: Taskflow 的 **unbounded deque**（动态扩展）
- `LF_Queue`: 本文提出的 lock-free 批量窃取队列

---

## 3. 主要实验结果和性能指标

### Microbenchmark 结果

#### 🔹 Push 操作延迟（随 batch size 变化）
| Batch Size | `LF_Queue` | `TF_BD/UB_Queue` |
|------------|------------|------------------|
| 1–128      | ~200–400 ns | 相似             |
| 1024       | **< 500 ns** | **~5000 ns**     |

✅ **LF_Queue 在大批量 push 下保持常数级延迟**，而 Taskflow 队列因逐节点操作和内存重分配导致延迟急剧上升。

#### 🔹 Steal 操作延迟（随窃取比例变化）
- 测试从队尾窃取 10% 到 60% 的任务
- `LF_Queue` 表现出 **近乎平坦的延迟曲线（约 40μs）**
- Taskflow 队列延迟随比例线性增长，最高达 **>112μs**

> 💡 原因：LF_Queue 一次遍历完成切割与计数；Taskflow 对每个被窃取的任务单独处理。

#### 🔹 Pop 操作
- 三者性能相近，约为 **213–216ns**
- 表明 pop 不是瓶颈，优化集中在 push/steal 更有意义

#### 🔹 优化版 steal（early-return variant）
- 若能确认 owner 未修改队列，则跳过第二次遍历以快速返回
- 实测延迟从 37.7μs（10%）降至 **12.5μs（60%）**
- ⬇️ **延迟降低高达 3倍**

> 📌 注：该优化未纳入主版本，用于展示常见场景下的潜力

### Pseudo Workload（DAG Exploration）结果

#### 🔹 可扩展性分析（Figure 9 & 10）
- 所有三种队列均表现出良好的线性可扩展性
- 双对数坐标下，执行时间随线程数翻倍而减半 → **近似 log-linear speedup**
- 不同图规模（2.5M vs 300M）下趋势一致，说明算法对输入规模不敏感

#### 🔹 总体性能差异
- 在此 workload 下，各队列总执行时间相近
- 原因：节点处理成本均匀，queue 开销未成为主导因素

> ⚠️ 但作者强调：这并不代表 LF_Queue 优势不明显 —— **真实 solver 中节点处理时间极不规则**，届时 queue 性能差异将更显著。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **针对特定 workload 定制 work-stealing queue 可带来显著性能收益**
2. ✅ **原生 bulk operations + 单窃取者模型 + 无锁链表结构** 是应对大规模、不规则任务生成的有效组合
3. ✅ `LF_Queue` 实现了 **constant-latency push 和 stable-latency steal**，优于通用队列
4. ✅ 微观层面优势虽在均匀 workload 中被掩盖，但在 **不规则 solver 场景中预期会被放大**

### 方法的局限性
1. ❌ **仅支持单一 stealer**，无法直接应用于允许多个 thief 并发窃取的系统
2. ❌ 当前尚未集成到完整 solver 中，缺乏真实环境验证
3. ❌ 对小比例、细粒度 steal 操作的优化空间有限（相比通用框架）

### 未来工作方向
1. ➤ 将 `LF_Queue` 集成进完整的 **DD-based MIP solver**，进行端到端性能评估
2. ➤ 扩展设计以支持 **多个并发 stealer**（如在分布式 MPI 环境中）
3. ➤ 探索更多优化路径，例如：
   - 更智能的 steal 比例自适应策略
   - early-return steal 的安全启用机制
4. ➤ 支持 **NUMA-aware** 内存布局，进一步提升多插槽系统的缓存局部性

---

> ✅ **总结一句话**：  
> 本文展示了“**专用优于通用**”的设计哲学 —— 通过结合 **bulk operations、lock-free linked list 和 single-stealer 模型**，`LF_Queue` 在特定高性能计算场景下实现了比主流 work-stealing 队列更优的可预测性和吞吐能力。

</details>

---

### 16. [Weak-SIGReg: Covariance Regularization for Stable Deep Learning](https://arxiv.org/abs/2603.05924)

**Authors**: Habibullah Akbar  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.05924v1  

#### Abstract
Modern neural network optimization relies heavily on architectural priorssuch as Batch Normalization and Residual connectionsto stabilize training dynamics. Without these, or in low-data regimes with aggressive augmentation, low-bias architectures like Vision Transformers (ViTs) often suffer from op...

---

### 17. [RoboLayout: Differentiable 3D Scene Generation for Embodied Agents](https://arxiv.org/abs/2603.05522)

**Authors**: Ali Shamsaddinlou  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.05522v1  

#### Abstract
Recent advances in vision language models (VLMs) have shown strong potential for spatial reasoning and 3D scene layout generation from open-ended language instructions. However, generating layouts that are not only semantically coherent but also feasible for interaction by embodied agents remains ch...

---

### 18. [Artificial Intelligence for Climate Adaptation: Reinforcement Learning for Climate Change-Resilient Transport](https://arxiv.org/abs/2603.06278)

**Authors**: Miguel Costa, Arthur Vandervoort, Carolin Schmidt, Jo\~ao Miranda, Morten W. Petersen, Martin Drews, Karyn Morrisey, Francisco C. Pereira  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.06278v1  

#### Abstract
Climate change is expected to intensify rainfall and, consequently, pluvial flooding, leading to increased disruptions in urban transportation systems over the coming decades. Designing effective adaptation strategies is challenging due to the long-term, sequential nature of infrastructure investmen...

---

### 19. [NOTAI.AI: Explainable Detection of Machine-Generated Text via Curvature and Feature Attribution](https://arxiv.org/abs/2603.05617)

**Authors**: Oleksandr Marchenko Breneur, Adelaide Danilov, Aria Nourbakhsh, Salima Lamsiyah  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.05617v1  

#### Abstract
We present NOTAI.AI, an explainable framework for machine-generated text detection that extends Fast-DetectGPT by integrating curvature-based signals with neural and stylometric features in a supervised setting. The system combines 17 interpretable features, including Conditional Probability Curvatu...

---

### 20. [InfoGatherer: Principled Information Seeking via Evidence Retrieval and Strategic Questioning](https://arxiv.org/abs/2603.05909)

**Authors**: Maksym Taranukhin, Shuyue Stella Li, Evangelos Milios, Geoff Pleiss, Yulia Tsvetkov, Vered Shwartz  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.05909v1  

#### Abstract
LLMs are increasingly deployed in high-stakes domains such as medical triage and legal assistance, often as document-grounded QA systems in which a user provides a description, relevant sources are retrieved, and an LLM generates a prediction. In practice, initial user queries are often underspecifi...

---

### 21. [FuseDiff: Symmetry-Preserving Joint Diffusion for Dual-Target Structure-Based Drug Design](https://arxiv.org/abs/2603.05567)

**Authors**: Jianliang Wu, Anjie Qiao, Zhen Wang, Zhewei Wei, Sheng Chen  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.05567v1  

#### Abstract
Dual-target structure-based drug design aims to generate a single ligand together with two pocket-specific binding poses, each compatible with a corresponding target pocket, enabling polypharmacological therapies with improved efficacy and reduced resistance. Existing approaches typically rely on st...

---

### 22. [Score-Guided Proximal Projection: A Unified Geometric Framework for Rectified Flow Editing](https://arxiv.org/abs/2603.05761)

**Authors**: Vansh Bansal, James G Scott  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.05761v1  

#### Abstract
Rectified Flow (RF) models achieve state-of-the-art generation quality, yet controlling them for precise tasks -- such as semantic editing or blind image recovery -- remains a challenge. Current approaches bifurcate into inversion-based guidance, which suffers from "geometric locking" by rigidly adh...

---

### 23. [Sparse Crosscoders for diffing MoEs and Dense models](https://arxiv.org/abs/2603.05805)

**Authors**: Marmik Chaudhari, Nishkal Hundia, Idhant Gulati  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.05805v1  

#### Abstract
Mixture of Experts (MoE) achieve parameter-efficient scaling through sparse expert routing, yet their internal representations remain poorly understood compared to dense models. We present a systematic comparison of MoE and dense model internals using crosscoders, a variant of sparse autoencoders, t...

---

### 24. [Dynamic Momentum Recalibration in Online Gradient Learning](https://arxiv.org/abs/2603.06120)

**Authors**: Zhipeng Yao, Rui Yu, Guisong Chang, Ying Li, Yu Zhang, Dazhou Li  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.06120v1  

#### Abstract
Stochastic Gradient Descent (SGD) and its momentum variants form the backbone of deep learning optimization, yet the underlying dynamics of their gradient behavior remain insufficiently understood. In this work, we reinterpret gradient updates through the lens of signal processing and reveal that fi...

---

### 25. [A recipe for scalable attention-based MLIPs: unlocking long-range accuracy with all-to-all node attention](https://arxiv.org/abs/2603.06567)

**Authors**: Eric Qu, Brandon M. Wood, Aditi S. Krishnapriyan, Zachary W. Ulissi  
**Category**: cs.LG  
**Published**: 2026-03-09  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.06567v1  

#### Abstract
Machine-learning interatomic potentials (MLIPs) have advanced rapidly, with many top models relying on strong physics-based inductive biases. However, as models scale to larger systems like biomolecules and electrolytes, they struggle to accurately capture long-range (LR) interactions, leading curre...

---

### 26. [The World Won't Stay Still: Programmable Evolution for Agent Benchmarks](https://arxiv.org/abs/2603.05910)

**Authors**: Guangrui Li, Yaochen Xie, Yi Liu, Ziwei Dong, Xingyuan Pan, Tianqi Zheng, Jason Choi, Michael J. Morais, Binit Jha, Shaunak Mishra, Bingrou Zhou, Chen Luo, Monica Xiao Cheng, Dawn Song  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.05910v1  

#### Abstract
LLM-powered agents fulfill user requests by interacting with environments, querying data, and invoking tools in a multi-turn process. Yet, most existing benchmarks assume static environments with fixed schemas and toolsets, neglecting the evolutionary nature of the real world and agents' robustness ...

---

### 27. [Agentic LLM Planning via Step-Wise PDDL Simulation: An Empirical Characterisation](https://arxiv.org/abs/2603.06064)

**Authors**: Kai G\"obel, Pierrick Lorang, Patrik Zips, Tobias Gl\"uck  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.06064v1  

#### Abstract
Task planning, the problem of sequencing actions to reach a goal from an initial state, is a core capability requirement for autonomous robotic systems. Whether large language models (LLMs) can serve as viable planners alongside classical symbolic methods remains an open question. We present PyPDDLE...

---

### 28. [Boosting deep Reinforcement Learning using pretraining with Logical Options](https://arxiv.org/abs/2603.06565)

**Authors**: Zihan Ye, Phil Chau, Raban Emunds, Jannis Bl\"uml, Cedric Derstroff, Quentin Delfosse, Oleg Arenz, Kristian Kersting  
**Category**: cs.AI  
**Published**: 2026-03-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.06565v1  

#### Abstract
Deep reinforcement learning agents are often misaligned, as they over-exploit early reward signals. Recently, several symbolic approaches have addressed these challenges by encoding sparse objectives along with aligned plans. However, purely symbolic architectures are complex to scale and difficult ...

---

### 29. [Towards Robust Retrieval-Augmented Generation Based on Knowledge Graph: A Comparative Analysis](https://arxiv.org/abs/2603.05698)

**Authors**: Hazem Amamou, St\'ephane Gagnon, Alan Davoust, Anderson R. Avila  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.05698v1  

#### Abstract
Retrieval-Augmented Generation (RAG) was introduced to enhance the capabilities of Large Language Models (LLMs) beyond their encoded prior knowledge. This is achieved by providing LLMs with an external source of knowledge, which helps reduce factual hallucinations and enables access to new informati...

---

### 30. [LIT-RAGBench: Benchmarking Generator Capabilities of Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2603.06198)

**Authors**: Koki Itai, Shunichi Hasegawa, Yuta Yamamoto, Gouki Minegishi, Masaki Otsuki  
**Category**: cs.CL  
**Published**: 2026-03-09  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.06198v1  

#### Abstract
Retrieval-Augmented Generation (RAG) is a framework in which a Generator, such as a Large Language Model (LLM), produces answers by retrieving documents from an external collection using a Retriever. In practice, Generators must integrate evidence from long contexts, perform multi-step reasoning, in...

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
