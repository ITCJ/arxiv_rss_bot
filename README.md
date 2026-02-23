# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-23 06:49:13 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [TempoNet: Slack-Quantized Transformer-Guided Reinforcement Scheduler for Adaptive Deadline-Centric Real-Time Dispatchs](https://arxiv.org/abs/2602.18109)

**Authors**: Rong Fu, Yibo Meng, Guangzhen Yao, Jiaxuan Lu, Zeyu Zhang, Zhaolu Kang, Ziming Guo, Jia Yee Tan, Xiaojing Du, Simon James Fong  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 15.0  
**Type**: new  
**ArXiv ID**: 2602.18109v1  

#### Abstract
Real-time schedulers must reason about tight deadlines under strict compute budgets. We present TempoNet, a reinforcement learning scheduler that pairs a permutation-invariant Transformer with a deep Q-approximation. An Urgency Tokenizer discretizes temporal slack into learnable embeddings, stabiliz...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《TempoNet: Slack-Quantized Transformer-Guided Reinforcement Scheduler for Adaptive Deadline-Centric Real-Time Dispatchs》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
实时调度系统在高动态负载、不确定执行时间及多核环境下，传统基于优先级的**analytic schedulers**（如 EDF、RM）在过载时性能急剧下降，难以保证严格的 deadline compliance。同时，现有的**Reinforcement Learning (RL)** 调度器存在以下问题：
- 依赖序列编码，引入顺序偏差，泛化能力差；
- 使用全注意力机制导致 $O(N^2)$ 复杂度，无法满足亚毫秒级推理延迟要求；
- 缺乏对 deadline 临近性的显式建模，优化稳定性差。

### 提出的新方法与创新思路
作者提出 **TempoNet**，一种面向 deadline-centric 实时调度的值函数型 RL 调度框架，其核心创新包括：

#### （1）**Urgency Tokenizer (UT)** —— 基于松弛量（slack）的可学习离散化嵌入层
- 将连续的 per-job slack $s_i(t)$ 通过 clip 和 floor 操作量化为离散索引 $q_i$；
- 使用可训练的 embedding 矩阵 $E \in \mathbb{R}^{Q \times d}$ 映射为 urgency token $x_i(t)$；
- **优势**：  
  - 减少梯度方差，提升训练稳定性；  
  - 强制模型按“紧迫等级”分组任务，增强 deadline-aware 表示能力；  
  - 理论上证明该设计相比连续输入具有更强的表达力（expressivity advantage）。

#### （2）**轻量级稀疏注意力 Transformer 编码器**
- 采用 permutation-invariant 架构处理无序任务集合；
- 引入 **blockwise top-k 选择 + locality-sensitive chunking** 的稀疏注意力机制；
- 实现近线性复杂度 $O(N^{1.1})$，支持大规模任务集下的全局推理；
- 推理延迟控制在 **sub-millisecond** 级别，适用于硬实时场景。

#### （3）**高效的多核映射层（Multicore Mapping Layer）**
- 将上下文化后的 Q-scores 映射到多核分配决策；
- 支持两种策略：
  - **Masked-Greedy Selection**：迭代选取最高未屏蔽 token，适合低延迟部署；
  - **Differentiable Matching**（如 Sinkhorn）：追求更优匹配质量，代价是更高延迟。

#### （4）端到端可扩展架构
- 支持与 **Actor-Critic** 框架兼容，无需修改推理流水线；
- 可结合 **Behavioral Cloning 预训练** 提升样本效率；
- 支持硬件闭环微基准测试（hardware-in-the-loop），确保实际部署可行性。

### 相比现有方法的优势
| 维度 | TempoNet | 传统方法（EDF/RM） | 其他深度学习方法（DQN/GNN/Transformer） |
|------|----------|---------------------|----------------------------------------|
| 过载鲁棒性 | ✅ 强 | ❌ 差 | ⚠️ 中等 |
| 泛化能力 | ✅ permutation-invariant | ✅ | ❌ 序列依赖 |
| 推理延迟 | ✅ <1ms（600任务） | ✅ 极快 | ❌ 高（尤其 dense attention） |
| deadline compliance | ✅ 最高 | ⚠️ 仅理想条件下最优 | ⚠️ 不稳定 |
| 可解释性 | ✅ 注意力可视化 + 政策蒸馏 | ✅ 规则明确 | ❌ 黑箱 |

---

## 2. 核心实验方法和设置

### 数据集
- **合成任务集**：生成 200 个随机化的 5 任务配置，利用率 $U \sim [0.6, 1.0]$；
- **工业混合关键性轨迹**：来自真实多核系统的 mixed-criticality 工作负载；
- **大规模多处理器场景**：最多包含 **600 个并发任务**，模拟云边协同或车载计算环境。

### 实验设置
- **平台**：NVIDIA V100 / Tegra Orin Nano / ARM Cortex-A78；
- **模型参数**：
  - Embedding dimension $d=128$
  - Attention heads $H=4$
  - Encoder depth $L=2$
  - Slack quantization levels $Q=128$
- **训练方式**：Deep Q-Learning with experience replay, soft target update, $\epsilon$-greedy exploration annealing.

### 评估指标
| 指标 | 定义 |
|------|------|
| **Deadline Compliance Rate** | 成功完成且不超期的任务占比 |
| **Average Response Time (ART)** | 从释放到完成的平均耗时 |
| **PITMD (Percentage of Important Tasks Meeting Deadlines)** | 关键任务的 deadline 满足率 |
| **Success Rate** | 所有关键任务均达标的运行比例 |
| **Inference Latency** | 单次调度决策的 wall-clock 时间 |
| **Complexity Scaling** | 随任务数增长的时间复杂度 |

### 基线方法对比
- **Analytic Schedulers**：EDF, RM, FCFS
- **Metaheuristic Methods**：Mo-QIGA, HQIGA, PSO-based
- **Deep RL Baselines**：
  - FF-DQN, Rainbow DQN
  - LSTM-PPO, DRL-based
  - GNN-based [26], Transformer-based DRL [11]
- **State-of-the-art Learned Schedulers**：MHQISSO, DIOS, ENF-S

---

## 3. 主要实验结果和性能指标

### 关键性能数据（综合表现）

| 方法 | Deadline Compliance (%) | ART (ms) | Inference Time (ms) | Complexity |
|------|--------------------------|---------|------------------------|-----------|
| **TempoNet (Ours)** | **87.0** | **12.4** | **0.375 @600** | $O(N^{1.1})$ |
| GNN-based [26] | 81.0 | 13.2 | 0.40 @600 | $O(N^{1.5})$ |
| Transformer-based DRL [11] | 83.0 | 13.5 | 0.52 @600 | $O(N^{2.2})$ |
| MHQISSO (EDF) | 87.5 | – | 0.53 @600 | $O(N^{2.2})$ |
| DIOS | 87.28 | 16.72 | – | $O(N^{1.8})$ |

> ✅ TempoNet 在 **600 任务规模下仍保持 sub-millisecond 推理速度**，并实现 **90.1% 成功率**（远超 MHQISSO 的 87.5%）。

### 与基线方法的对比结果
- 在 uniprocessor 场景中，TempoNet 达到 **79.00% deadline compliance**，比 EDF 提升 **67.33%**，比 FF-DQN 提升 **7.57%**；
- 在工业混合关键性负载中，PITMD 达到 **89.15%**，优于 DIOS（+1.87pp）、GNN-based（+0.35pp）；
- 平均响应时间降低 **25.7%**，峰值延迟改善高达 **37%**；
- 推理速度比同类方法快 **5.1–8.2×**（见 Figure 4）。

### 消融实验结果（Ablation Studies）

#### （1）Urgency Tokenizer 消融（Table 7）
| Variant | Hit Rate (%) | △ vs TempoNet |
|--------|---------------|----------------|
| FF-DQN-cont (raw slack) | 74.8 | -12.2 |
| FF-DQN-norm (z-score) | 76.1 | -10.9 |
| FF-DQN-MLP | 79.5 | -7.5 |
| **TempoNet w/o UT (concat)** | **81.3** | **-5.7** |
| **TempoNet (full)** | **87.0** | **–** |

> ✅ UT 模块带来显著增益（+5.7pp），且训练方差更低（$ \sigma^2 = 1.7\times10^{-3} $ vs. $2.4\times10^{-3}$）。

#### （2）注意力头数影响（Table 2）
| Heads | Hit Rate (%) |
|-------|--------------|
| 2 | 80.3 |
| **4** | **85.0** |
| 6 | 84.7 |
| 8 | 84.9 |

> ✅ 四头注意力即达最优，更多 heads 无明显收益反而增加开销。

#### （3）编码器深度分析（Table 3）
| Layers | Hit Rate (%) | Latency (ms) |
|--------|---------------|----------------|
| 1 | 76.2 | 0.42 |
| **2** | **85.0** | **0.51** |
| 3 | 86.1 | 0.71 |
| 4 | 85.7 | 0.94 |

> ✅ 两层编码器已达最佳性价比，更深网络收益递减。

#### （4）嵌入维度分析（Figure 7）
- $d=128$ 时达到性能-成本平衡点；
- 更大维度带来边际收益递减（diminishing returns）。

---

## 4. 关键结论和发现

### 主要发现
1. **Slack Quantization 是关键设计**：将连续 slack 离散化为 learnable tokens，不仅能提升表示质量，还能理论上有界地减少 approximation error（Theorem A.1）；
2. **稀疏注意力实现实用化**：通过 block Top-k + locality-aware chunking，实现 $O(N^{1.1})$ 复杂度，在 600 任务下仍维持 **<400μs** 推理时间；
3. **全局推理优于局部启发式**：Transformer 的 attention maps 显示其能有效聚焦于 deadline-critical 任务（correlation $r=0.98$）；
4. **政策可解释性强**：通过蒸馏得到的规则 `priority = 0.73/s + 0.27/c` 可解释 91% 决策，表明其融合了 EDF 与 SRPT 的优点；
5. **具备强鲁棒性与适应性**：
   - 在非平稳负载下 few-shot adaptation 可快速恢复性能；
   - 支持 runtime mitigation（如动态稀疏缩放）应对突发过载。

### 方法的局限性
- 当任务集极度不平衡（如大量短周期任务爆发）时，attention 可能过度集中于极低 slack 任务，导致长 slack 任务饥饿；
- 当系统利用率持续高于 1.25 时，性能开始显著下降（stress test 下 compliance 降至 ~71%）；
- 目前假设任务间无资源竞争（如内存带宽），未来需扩展至异构资源联合调度。

### 未来工作方向
1. 扩展至 **heterogeneous hardware**（CPU/GPU/FPGA）联合调度；
2. 引入 **energy-aware** 或 **multi-objective optimization**（如能效比、公平性）；
3. 探索 **distributed attention** 用于跨节点集群调度；
4. 结合 **offline RL** 与 **conservative fine-tuning** 提高安全性；
5. 开发专用 **sparse attention kernel** 进一步压榨边缘设备延迟。

---

> 📌 **总结一句话**：  
> **TempoNet 通过“离散化紧迫令牌 + 稀疏注意力 + 多核贪心映射”的协同设计，在保证亚毫秒级推理的前提下，实现了当前最先进的 deadline compliance 性能，为 Transformer 在硬实时系统中的落地提供了实用化路径。**

</details>

---

### 2. [SPQ: An Ensemble Technique for Large Language Model Compression](https://arxiv.org/abs/2602.18420)

**Authors**: Jiamin Yao, Eren Gultepe  
**Category**: cs.CL  
**Published**: 2026-02-23  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.18420v1  

#### Abstract
This study presents an ensemble technique, SPQ (SVD-Pruning-Quantization), for large language model (LLM) compression that combines variance-retained singular value decomposition (SVD), activation-based pruning, and post-training linear quantization. Each component targets a different source of inef...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SPQ: An Ensemble Technique for Large Language Model Compression

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）虽然在自然语言理解和生成任务上表现出色，但其庞大的参数量导致**高内存占用和计算开销**，限制了在资源受限设备上的部署。现有的单一压缩技术（如 SVD、Pruning、Quantization）在高压缩率下往往会导致显著的性能下降。

本文旨在通过**组合多种互补的压缩技术**，实现高效、高质量的 LLM 压缩，在大幅降低内存消耗的同时保持甚至提升模型性能。

---

### 🚀 提出的新方法：SPQ（SVD-Pruning-Quantization）
SPQ 是一种**模块化、层感知（layer-aware）的集成压缩框架**，结合三种核心技术：

| 技术 | 应用位置 | 目标 |
|------|--------|------|
| **SVD**（低秩分解） | Attention 层 | 利用注意力权重的低秩特性，将其分解为紧凑因子 |
| **Pruning**（结构化剪枝） | MLP 层 | 基于激活值移除冗余神经元 |
| **Quantization**（量化） | 所有线性层 | 使用 8-bit 对称线性量化统一压缩 |

> 🔍 **关键设计思想**：将每种压缩方法应用于其最有效的层类型，形成“结构对齐”的压缩策略。

---

### ⭐ 相比现有方法的优势

| 维度 | SPQ 的优势 |
|------|-----------|
| **方法论创新** | 首次系统性地将 SVD（Attention）、Pruning（MLP）、Quantization（全连接层）三者结合成一个统一管道，而非仅两两组合（如 GPTQ 只做量化，SVD-LLM v2 缺少剪枝）。 |
| **SVD 设计** | 使用**方差保留准则**决定截断秩，无需复杂的损失估计或逐层调参，更简洁鲁棒。 |
| **Pruning 设计** | 仅针对 MLP 层进行基于激活统计的结构化剪枝，使用 log-scale 映射确定剪枝比例，轻量且硬件友好。 |
| **Quantization 设计** | 采用无需校准数据的 **post-training 8-bit 线性量化**，支持 per-tensor / per-channel / hybrid 模式，兼容性强。 |
| **效率优势** | 不仅压缩率更高，还实现了比 GPTQ 更快的推理吞吐（最高达 1.9× 加速），端到端压缩时间也更快（比 GPTQ 快 20%）。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 类型 | 数据集 |
|------|-------|
| **语言建模评估** | WikiText-2, C4 |
| **下游推理任务** | OpenBookQA, ARC, WinoGrande, HellaSwag, PIQA |
| **真实性与数学能力** | TruthfulQA-1/2, GSM8K |
| **校准数据集** | 小规模 calibration dataset（用于计算激活均值） |

---

### 🧪 实验设置与评估指标

| 项目 | 设置说明 |
|------|---------|
| **模型** | 主要测试 LLaMA-2-7B，同时验证 LLaMA、OPT、Vicuna、Mistral 等系列（1B–7B 参数） |
| **硬件平台** | 2× NVIDIA A100-40GB GPU |
| **评估指标** | <ul><li>**Weight Memory (GB)**：权重存储大小</li><li>**Perplexity ↓**：语言建模质量（越低越好）</li><li>**Throughput (tokens/sec) ↑**：推理速度</li><li>**Accuracy (%) ↑**：下游任务准确率</li></ul> |
| **超参数范围** | <ul><li>SVD 方差阈值：ε ∈ [0.84, 0.96]</li><li>最大剪枝率：r_max ∈ [0.05, 0.30]</li><li>量化方式：Per-tensor, Per-channel, PBH, LNH, MSH hybrid</li></ul> |

---

### 🆚 基线方法对比

| 方法 | 类型 | 特点 |
|------|------|------|
| **ASVD** | SVD-only | 基于激活感知的奇异值分解 |
| **SparseGPT** | Pruning-only | 结构化稀疏化方法 |
| **GPTQ** | Quantization-only | 强大的 4/8-bit 后训练量化基线 |
| **SVD-only / Pruning-only / Quant-only** | 单一方法 | 用于消融研究 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 LLaMA-2-7B 为例）

| 指标 | SPQ | 原始模型 | GPTQ (8-bit) | SparseGPT | ASVD |
|------|-----|----------|-------------|------------|--------|
| **压缩率** | **75%** | 0% | 73% | 50% | 21% |
| **权重内存** | **6.86 GB** | 26.95 GB | 7.16 GB | 13.40 GB | 21.41 GB |
| **WikiText-2 Perplexity** | **4.91 ↓** | 5.47 | 5.48 | 7.76 | 6.54 |
| **C4 Perplexity** | 7.11 | 6.85 | 6.66 | 8.98 | 7.66 |
| **平均推理准确率** | ≈原始水平 | — | ≈GPTQ | 显著下降 | 下降明显 |
| **TruthfulQA BLEU** | 0.24 / 0.38 | 0.24 / 0.39 | 0.22 / 0.34 | 0.22 / 0.38 | 0.25 / 0.41 |
| **GSM8K 准确率** | 0.05 | 0.05 | 0.04 | 0.03 | 0.03 |

> ✅ **亮点**：SPQ 在 **75% 压缩率**下不仅内存最小（6.86 GB），而且在 WikiText-2 上 **perplexity 显著优于原始模型**（5.47 → 4.91）！

---

### 🔍 与基线方法的对比结果

- **相比 GPTQ**：
  - 内存减少 **2%**（6.86 vs 7.16 GB）
  - 推理吞吐提升 **1.3× (vs GPTQ-8bit)** 至 **1.9× (vs GPTQ-4bit)**
  - Perplexity 更优（尤其 WikiText-2）
  - 下游任务表现相当或略优

- **相比 SparseGPT 和 ASVD**：
  - 实现更高的压缩率（75% vs 50%/21%）
  - 在高压缩下仍保持稳定性能，而基线出现明显退化

---

### 🔬 消融实验结果（Ablation Studies）

#### （1）单个组件效果（图7）
- **SVD-only**：超过 15% 压缩后 perplexity 急剧上升
- **Pruning-only**：40% 压缩后性能崩溃
- **Quantization-only (8-bit)**：稳定但压缩有限；4-bit 时性能骤降（~85% 压缩）
- **SPQ**：即使压缩 >80%，perplexity 仍低于 15，展现出极强鲁棒性

#### （2）两两组合分析（图6 & 表2）
| 组合 | 内存节省 | Perplexity 变化 | 结论 |
|------|--------|----------------|------|
| SVD + Quant | 显著（p<0.001） | 无显著差异 | 互补有效 |
| Pruning + Quant | 显著（p<0.001） | 无显著恶化 | 可安全叠加 |
| SVD + Pruning | — | 控制得当可维持 baseline 水平 | 需精细调节 |

> ✅ **结论**：三个组件相互补充，联合使用可突破单一方法瓶颈。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **集成优于单一**：SVD、Pruning、Quantization 各自存在极限，但**组合使用能产生协同效应**，实现更高压缩而不牺牲性能。
2. **层感知设计至关重要**：将 SVD 用于 Attention、Pruning 用于 MLP，符合模型内在结构特性，是成功的关键。
3. **SPQ 实现“双赢”**：在 LLaMA-2-7B 上达到 **75% 压缩率 + 更低 perplexity + 更高吞吐**，打破了“压缩必损性能”的传统认知。
4. **通用性强**：在 LLaMA、OPT、Vicuna、Mistral 多种架构上均取得一致效果，尤其对大模型（7B级）增益更明显（见表4）。
5. **实用性强**：无需复杂优化或大量校准数据，压缩速度快于 GPTQ，适合实际部署。

---

### ⚠️ 方法的局限性

1. **依赖 LoRA 微调恢复性能**：尽管仅需 200 步，但仍需额外训练步骤；完全免微调版本可能性能略降。
2. **当前仅处理权重压缩**：未涉及 **activation quantization** 或 KV Cache 压缩，仍有进一步优化空间。
3. **配置非全自动**：虽然比其他方法简单，但 SVD 阈值、剪枝上限等仍需手动设定，缺乏自适应机制。
4. **未探索混合精度**：目前统一使用 8-bit，若引入 4-bit 或 mixed-precision 可能获得更大压缩。

---

### 🔮 未来工作方向

1. **扩展至 activation quantization**：进一步降低运行时内存和能耗。
2. **探索更灵活的集成策略**：尝试替换组件（如用知识蒸馏替代 LoRA）或动态调整各阶段强度。
3. **适配更多硬件平台**：针对边缘设备、移动端等特定场景优化部署流程。
4. **研究其他矩阵分解技术**：如 Tucker 分解、QR 分解等是否能在某些层提供更好近似。
5. **自动化压缩配置搜索**：开发算法自动选择最优的 SVD 阈值、剪枝率、量化模式组合。

---

> 💡 **总结一句话**：  
> **SPQ 通过“层感知 + 模块化集成”的设计理念，首次实现了 SVD、Pruning 与 Quantization 在 LLM 上的高效协同，在高达 75% 压缩率下仍保持甚至超越原始模型的语言建模能力和推理效率，为 LLM 在资源受限环境中的实用化部署提供了强有力的技术路径。**

🔗 代码已开源：[https://github.com/JiaminYao/SPQ_LLM_Compression/](https://github.com/JiaminYao/SPQ_LLM_Compression/)

</details>

---

### 3. [Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](https://arxiv.org/abs/2602.18403)

**Authors**: Orfeas Bourchas, George Papalambrou  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.18403v1  

#### Abstract
Accurate prediction of main engine power is essential for vessel performance optimization, fuel efficiency, and compliance with emission regulations. Conventional machine learning approaches, such as Support Vector Machines, variants of Artificial Neural Networks (ANNs), and tree-based methods like ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统纯数据驱动的机器学习模型（如 XGBoost、Neural Networks、PINNs）在船舶主发动机功率预测中虽然在训练数据密集区域表现良好，但在**稀疏数据区或外推场景**（如高速航行、极端风况）下往往违反物理规律（特别是 propeller law 的立方关系 $P \propto V^3$），导致预测结果不可靠。

此外，尽管 Physics-Informed Neural Networks (PINNs) 能部分保证物理一致性，但其训练困难，对噪声敏感，在真实运营数据中表现不稳定。

### 🚀 提出的新方法/新思路
提出一种**科学知识引导的混合建模框架（hybrid modeling framework）**，将物理先验与数据驱动残差学习相结合：

- **Baseline Component**：基于海试（sea trial）数据构建 calm-water 功率曲线 $P_{\text{sea trial}} = cV^n$，作为物理一致的基础预测。
- **Residual Learning Component**：使用非线性回归器（XGBoost / NN / PINN）仅学习观测功率与基线之间的**残差**（residual），即环境和操作因素（风、浪、船体污底、老化等）引起的变化。

该方法形式为：
$$
P(X) = P_{\text{sea trial}}(V, T) + f(X)
$$
其中 $f(X)$ 是 ML 模型预测的残差项。

### 🔍 相比现有方法的优势
| 优势维度 | 描述 |
|--------|------|
| **物理一致性增强** | 强制保留 $P \propto V^3$ 的主导趋势，避免非物理解释。 |
| **外推能力提升** | 在训练范围之外（如高航速）仍能生成平滑且合理的预测。 |
| **学习任务简化** | ML 模型只需拟合“偏差”，而非从零学习整个复杂映射，降低过拟合风险。 |
| **通用性强** | 可适配多种 ML 架构（XGBoost、NN、PINN），具有广泛适用性。 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 来源：作者先前研究 *(Bourchas and Papalambrou, 2025)*
- 规模：约 **40,000 条记录**，涵盖一艘船舶连续 **五个月的真实运营数据**
- 主要变量（见 Table 1）：
  - `P`: 主机制动功率 (kW)
  - `V`: 船速（Speed Through Water, kn）
  - `T`: 平均吃水 (m)
  - `Trim`: 纵倾 (m)
  - `WTS`, `WTD`: 风速与风向 → 分解为 `Wx`, `Wy` 分量（公式 12）

> 数据按 **80%/10%/10%** 划分为训练 / 验证 / 测试集。

---

### ⚙️ 实验设置

#### 模型架构对比
对以下三种主流模型分别实现 **Baseline（纯数据驱动）** 和 **Hybrid（带物理基线）** 版本进行比较：
1. **XGBoost**
2. **Simple Neural Network (NN)**
3. **Physics-Informed Neural Network (PINN)**

#### 超参数优化（HPO）
- **XGBoost**: 使用 `RandomizedSearchCV` 进行调参，搜索空间包括 learning rate、max depth、n_estimators、L1/L2 正则化。
- **NN & PINN**: 基于 PyTorch 实现，使用 Weights & Biases 的 Bayesian optimization 扫描超参数（learning rate、层数、每层神经元数）。
- 所有输入输出均标准化（StandardScaler）。

#### PINN 损失函数设计
总损失由两部分组成：
$$
\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{p.law}}
$$
其中：
- $\mathcal{L}_{\text{data}}$: MSE 回归损失
- $\mathcal{L}_{\text{p.law}}$: 基于 $\frac{\partial P}{\partial V} = 3cV^2$ 的物理导数约束（通过 PyTorch autograd 自动微分计算）
- $\lambda = 100$（固定权重）

---

### 📏 评估指标
- **定量指标**：
  - Mean Absolute Error (**MAE**)
  - Root Mean Square Error (**RMSE**)
- **定性分析重点**：
  - 外推行为（extrapolation behavior）：在 **8–17 kn 航速范围内**，不同风向下（0°, 90°, 180°）的功率-速度曲线形态是否合理、单调、符合 propeller law。

---

### 🔁 基线方法对比
| 类型 | 模型名称 | 是否含物理基线 |
|-----|---------|----------------|
| Baseline | XGBoost-only | ❌ |
| Baseline | NN-only | ❌ |
| Baseline | PINN-only | ❌ |
| Proposed | Hybrid-XGBoost | ✅ |
| Proposed | Hybrid-NN | ✅ |
| Proposed | Hybrid-PINN | ✅ |

> 所有模型经过统一严格的 HPO 后再比较，确保公平性。

---

## 3. 主要实验结果和性能指标

### 📈 定量性能对比（见 Table 5）

| 模型 | Test MAE [kW] | Test RMSE [kW] |
|------|---------------|----------------|
| XGBoost (Baseline) | **122.2** | **195.0** |
| XGBoost (Hybrid)    | 148.8         | 208.2         |
| NN (Baseline)       | **162.66**    | **225.10**    |
| NN (Hybrid)         | 219.32        | 284.33        |
| PINN (Baseline)     | **144.30**    | **211.89**    |
| PINN (Hybrid)       | 171.19        | 229.45        |

> 💡 注意：Hybrid 模型在全局误差指标上略逊于 Baseline 模型（误差稍高），但这并非核心关注点。

---

### 🔍 定性外推性能分析（关键发现）

#### 图表观察（Figures 3–5）
- **Baseline 模型问题显著**：
  - 出现**非单调、平坦甚至下降**的功率-速度曲线（尤其在 >15 kn 区域）
  - 对风向变化反应异常剧烈或不合理
  - 明显偏离 calm-water envelope，违背物理常识

- **Hybrid 模型优势明显**：
  - 功率随速度增加保持**平滑上升趋势**
  - 即使在无训练样本的高速区域，也能维持接近 $P \propto V^3$ 的增长模式
  - 不同风向下预测更具一致性与可解释性

#### 各模型外推表现排序（从优到劣）：
1. **Hybrid-PINN**：最佳物理一致性与稳定性
2. **Hybrid-NN**
3. **Hybrid-XGBoost**
4. 所有 Baseline 模型（尤其 XGBoost 最差）

> 尽管 Hybrid 模型在 MAE/RMSE 上略差，但其**在稀疏区域的行为更可靠、更可信**。

---

### 🧪 消融实验（隐含分析）
虽然未明确标注“ablation study”，但全文本质上是一次系统性的消融实验：

| 对比轴 | 发现 |
|-------|------|
| 是否引入物理基线 | 引入后显著改善外推鲁棒性和物理一致性 |
| 不同 ML 架构选择 | 所有架构均受益于 Hybrid 结构，说明方法通用 |
| PINN 加入物理损失 vs Hybrid 结构 | Hybrid-PINN 效果最好，表明“双重物理约束”（基线 + 导数损失）最有效 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **物理引导的混合建模显著优于纯数据驱动方法**，尤其是在外推场景下；
2. 将 ML 任务限定为“残差修正”能有效**正则化模型行为**，防止其学习到虚假相关性；
3. Hybrid-PINN 表现出最强的综合性能，在准确性和物理一致性之间取得最佳平衡；
4. **全局误差指标（MAE/RMSE）不能反映模型可靠性**，特别是在决策支持类应用中，**预测趋势的合理性比数值精度更重要**；
5. 该框架适用于实际海事应用场景，如天气航线规划（weather routing）、航速优化、能效管理等。

---

### ⚠️ 局限性
1. **依赖高质量海试数据**：若缺乏 ballast/laden 工况下的 sea trial 曲线，则无法构建可靠 baseline；
2. **中间吃水采用线性插值**（Eq. 3）是一种简化假设，可能忽略非线性效应；
3. 当前物理约束仅考虑 $P-V$ 关系，未显式建模其他物理方程（如阻力成分分解）；
4. PINN 训练成本较高，且 $\lambda$ 参数未动态调整，影响收敛效率。

---

### 🔮 未来工作方向
1. 探索自适应权重机制（如 learnable $\lambda$）以平衡 data loss 与 physics loss；
2. 将更多物理模块集成进 baseline（如风阻、兴波阻力模型），进一步减少残差复杂度；
3. 扩展至多船种建模，研究迁移学习结合物理基线的可能性；
4. 应用于实时 onboard 决策系统，验证其在闭环控制中的有效性；
5. 引入不确定性量化（Uncertainty Quantification）以提供置信区间。

---

## ✅ 总结一句话
> 本文提出的 **scientific knowledge-guided hybrid framework** 通过将物理先验嵌入模型结构，实现了在保持预测精度的同时大幅提升外推稳定性和物理一致性，为船舶性能建模提供了**兼具实用性与理论严谨性**的新范式。

</details>

---

### 4. [SeedFlood: A Step Toward Scalable Decentralized Training of LLMs](https://arxiv.org/abs/2602.18181)

**Authors**: Jihun Kim, Namhoon Lee  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.18181v1  

#### Abstract
This work presents a new approach to decentralized training-SeedFlood-designed to scale for large models across complex network topologies and achieve global consensus with minimal communication overhead. Traditional gossip-based methods suffer from message communication costs that grow with model s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《SeedFlood: A Step Toward Scalable Decentralized Training of LLMs》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统去中心化训练（decentralized training）面临两大可扩展性瓶颈：
- **通信开销随模型规模增长**：基于 gossip 的共识机制需要频繁交换高维参数或梯度，导致通信成本与模型维度 $d$ 成正比，在 billion-parameter LLMs 上变得不可行。
- **信息衰减导致全局共识效率低下**：在稀疏或大规模网络拓扑中，gossip 多跳传播导致信息延迟和衰减，难以实现高效全局同步。

此外，现有基于 zeroth-order optimization 的方法虽能压缩通信，但仍沿用 gossip 范式，无法摆脱其结构性限制。

---

### **提出的新方法与新思路**
作者提出 **SeedFlood**，一种全新的去中心化训练框架，核心思想是：
- 利用 **seed-reconstructible zeroth-order updates** 将通信内容压缩为极小的随机种子（seed）和标量（scalar），使通信负载与模型大小无关。
- 改用 **flooding-based dissemination** 替代传统的 gossip-based averaging，通过递归广播机制将每个更新快速传播至全网所有客户端，实现类 all-gather 的全局共识。

进一步引入 **Subspace Canonical-basis Gradient Estimation (SubCGE)** 技术，解决 flooding 下大量更新带来的计算聚合瓶颈：
- 所有扰动被限制在一个共享的低秩子空间（low-rank subspace）内，并映射到该子空间的标准基坐标上。
- 允许客户端以向量化矩阵操作批量聚合数千个零阶梯度更新，避免逐个重建带来的高昂计算开销。

---

### **相比现有方法的优势**
| 维度 | 传统 Gossip | Gossip + Shared Randomness | SeedFlood（本文） |
|------|-------------|----------------------------|------------------|
| 通信字节数 | $O(d)$ | $O(tn)$ | $O(n)$ |
| 应用计算复杂度 | $O(d)$ | $O(tnd)$ | $O(n + rd)$ |
| 是否实现完美共识 | ❌ | ❌ | ✅（拓扑不变） |
| 通信是否依赖模型大小 | 是 | 否（但随时间增长） | 否 |
| 可扩展至百级客户端？ | ❌ | ❌ | ✅ |

> **优势总结**：
> - 通信开销极低且恒定（仅传输种子），适用于超大模型（如 1B 参数以上）；
> - 实现拓扑不变的高质量全局共识，尤其适合稀疏网络（ring topology）；
> - 首次将 flooding 引入去中心化训练作为共识原语，突破 gossip 范式的根本局限。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 主要任务来自 **SuperGLUE benchmark** 子集：
  - `RTE`, `BoolQ`, `WiC`, `MultiRC`, `ReCoRD`
- 加上情感分析任务 `SST-2`
- 每个任务使用 **1,024 条训练样本**，均匀分布在各客户端
- 使用固定的验证集（500 样本）和测试集（1,000 样本）

---

### **模型与网络拓扑**
- 使用 **OPT 系列语言模型**：
  - `OPT-125M`、`OPT-1.3B`、`OPT-2.7B`
- 客户端数量：16 ~ 128
- 网络拓扑结构：
  - **Ring**（环形，稀疏）
  - **Meshgrid**（网格，较密集）

---

### **训练设置**
- **本地迭代步数**：
  - First-order 方法：500 步
  - Zeroth-order 方法：5,000 步（因更高方差需更多采样）
- **通信频率**：每 5 个本地更新执行一次通信
- **SeedFlood 特殊设置**：
  - Flooding 步数 = 网络直径
  - Subspace rank $r = 32$ 或 $64$
  - Subspace refresh period $T = 1000$ 或 $5000$

---

### **评估指标**
- **Global Model Performance (GMP)**：训练结束后对所有客户端模型取平均并评测
- **总通信成本（Total Communication Cost）**：整个训练过程中每条边上传输的总字节数
- 性能以百分比形式报告，部分结果归一化于 DSGD@16C

---

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **First-order (FO)** | DSGD, ChocoSGD（带 Top-K 压缩） |
| **LoRA-based FO** | DSGD-LoRA, ChocoSGD-LoRA |
| **Zeroth-order (ZO)** | DZSGD, DZSGD-LoRA |
| **本文方法** | SeedFlood（+ SubCGE） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **极端通信效率**
- SeedFlood 整个训练过程仅消耗 **400 KB** 通信量（见 Figure 3）
- 相比之下：
  - DSGD：**526 GB**
  - ChocoSGD-LoRA：**18.8 MB**
  - 即便高压缩率的 LoRA + Top-K，仍比 SeedFlood 高出 **3~5 个数量级**

> 💡 **通信成本排序**：  
> `SeedFlood << ChocoSGD-LoRA << DSGD-LoRA << DSGD`

---

#### ✅ **性能表现优异**
- 在多数任务上显著优于所有 zeroth-order 基线（DZSGD 等）
- 在稀疏拓扑（ring）下表现尤为突出，**几乎不受网络结构影响**
- 在某些任务（如 BoolQ）甚至超过通信高效的 first-order 方法（如 ChocoSGD-LoRA），高出约 **4%**

| 方法 | 平均性能（相对 DSGD） | 通信成本 |
|------|------------------------|----------|
| DSGD | 100% | 526 GB |
| ChocoSGD-LoRA | ~91% | 18.8 MB |
| SeedFlood | **~94–96%** | **400 KB** |

> ⚠️ 尽管略低于 full-parameter DSGD（差距约 4–6%），但这是已知的 ZO 方法样本效率问题所致。

---

#### ✅ **大规模网络下的鲁棒性**
- 当客户端从 16 扩展到 128 时（Table 2）：
  - 所有 gossip-based 方法性能明显下降
  - SeedFlood 表现稳定，甚至在 128-client ring 拓扑下反超 DSGD，达到 **100.24%**（归一化）

> 📈 这表明 SeedFlood 不仅可扩展，还能利用更多客户端带来的扰动多样性提升估计质量。

---

#### ✅ **消融实验结果**
##### （1）SubCGE 的有效性（Figure 5）
- 对比原始 MeZO 与 SubCGE 在处理多个 ZO 更新时的运行时间：
  - 当更新数达 1024 时，MeZO 耗时超 **100 秒**
  - SubCGE 仅需不到 **1 秒**，快 **两个数量级以上**
- 表明 SubCGE 成功解决了 flooding 下的聚合计算瓶颈

##### （2）SubCGE 参数敏感性（Figure 6）
- 子空间秩过小（rank < 32）或刷新周期太短会导致性能下降
- 最佳配置：rank ≥ 32，refresh period ≈ 1000~5000

##### （3）延迟 flooding 实验（Figure 7）
- 即使只进行有限步 flooding（如 k=4），性能也无明显下降
- 只有当 k=1 或 2 时才出现退化
> 👉 表明 SeedFlood 不依赖“完全同步”，只要更新能在合理延迟内传播即可生效

---

## **4. 关键结论和发现**

### **主要发现**
1. **通信不再是去中心化训练的瓶颈**：
   - SeedFlood 通过 seed-reconstructible 更新 + flooding，实现了 **通信开销与模型大小解耦**
   - 第一次让 billion-parameter LLM 的去中心化微调在现实中可行

2. **flooding 可作为有效的共识机制**：
   - 在 zeroth-order 设置下，flooding 比 gossip 更自然、更高效
   - 实现了 **all-gather-equivalent consensus**，无需牺牲收敛速度

3. **SubCGE 是关键使能技术**：
   - 解决了 flooding 引发的“更新洪泛”计算难题
   - 使得高吞吐量聚合成为可能，支撑了大规模部署

4. **方法具备强拓扑鲁棒性和可扩展性**：
   - 在稀疏网络中表现优于密集网络中的 gossip 方法
   - 支持扩展至数百客户端而性能不降反升

---

### **局限性**
- **计算开销转移至客户端**：虽然通信极低，但每个客户端需处理 $O(n)$ 个 ZO 更新，依赖 SubCGE 缓解
- **依赖同步随机数生成器（RNG）**：要求所有客户端能基于相同 seed 重建相同扰动，系统实现上有一定挑战
- **zeroth-order 固有的样本效率问题**：相比 first-order 方法仍存在约 4–6% 的性能差距

---

### **未来工作方向**
- 探索 **partial flooding** 与异步执行结合，降低对网络直径的依赖
- 将 SeedFlood 扩展至 **non-IID 数据分布** 和 **动态拓扑**
- 结合 **adaptive subspace selection** 提升 SubCGE 的优化效率
- 探索在 **federated learning** 中的应用，尤其是边缘设备场景下的低带宽训练

---

> 🔚 **一句话总结**：  
> SeedFlood 通过 **seed-based communication + flooding consensus + SubCGE aggregation**，首次实现了**通信开销近乎为零、拓扑无关、可扩展至百级客户端的大规模去中心化 LLM 训练**，为未来绿色、分布式 AI 提供了新路径。

</details>

---

### 5. [Pimp My LLM: Leveraging Variability Modeling to Tune Inference Hyperparameters](https://arxiv.org/abs/2602.17697)

**Authors**: Nada Zine, Cl\'ement Quinton, Romain Rouvoy  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.17697v1  

#### Abstract
Large Language Models (LLMs) are being increasingly used across a wide range of tasks. However, their substantial computational demands raise concerns about the energy efficiency and sustainability of both training and inference. Inference, in particular, dominates total compute usage, making its op...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Pimp My LLM: Leveraging Variability Modeling to Tune Inference Hyperparameters*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLM）在推理阶段具有极高的计算开销，其配置空间庞大且复杂，包含大量相互依赖的生成超参数（如 `temperature`, `top-p`, `num_beams`, `cache` 策略等）。这种**组合爆炸**使得手动调优困难，而穷举实验又不可行。现有研究通常只孤立地分析少数参数，缺乏对参数间交互作用的系统建模。

本文旨在解决以下挑战：
- 如何系统化管理 LLM 推理配置空间？
- 如何识别影响 **energy consumption**, **latency**, 和 **accuracy** 的关键参数及其交互？
- 如何基于有限测量预测未见配置的表现？

---

### 🚀 提出的新方法与创新思路

作者提出将 LLM 视为一个**高度可配置软件系统**（highly configurable system），并首次引入**变体建模**（variability modeling）技术来建模和探索其推理配置空间。

#### 主要创新点：
1. **Feature-based 建模 LLM 推理配置空间**
   - 构建了一个公开可用的 **Feature Model (FM)** 来表示 Hugging Face Transformers 库中的生成超参数及其约束关系。
   - 模型包含 96 个 features（其中 67 个具体），支持约 $9.37 \times 10^{12}$ 种有效配置。
   - 显式编码参数间的依赖与互斥关系（cross-tree constraints），确保所有采样配置合法、可执行。

2. **四步预测框架：Modeling → Sampling → Measurement → Learning**
   - **Modeling**: 使用 FM 结构化整个配置空间；
   - **Sampling**: 采用多种策略（YASA, ICPL, RANDOM）从巨大空间中选取代表性子集；
   - **Measurement**: 在真实硬件上测量 energy, latency, accuracy；
   - **Learning**: 利用 Random Forest Regression 建立预测模型，泛化至未见配置。

3. **系统性揭示参数影响与权衡**
   - 通过 **feature-wise** 和 **pairwise 分析** 揭示单个参数及参数组合的影响；
   - 发现 energy 与 latency 高度相关，而 accuracy 受不同机制主导；
   - 绘制 **Pareto front** 展示最优 trade-offs。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| **配置探索方式** | 手动试错 / 单因素实验 | 自动化、系统化、覆盖交互 |
| **参数建模能力** | 忽略约束与依赖 | 显式建模 constraints，避免无效配置 |
| **可扩展性** | 不适用于高维空间 | 支持自动化推理（如枚举、验证） |
| **预测能力** | 缺乏通用模型 | 建立准确预测模型（R² > 0.94） |
| **跨任务迁移潜力** | 弱 | 方法通用，可迁移到其他 inference server 或任务 |

> ✅ **核心优势**：将软件工程领域的 variability modeling 成功应用于机器学习系统优化，打通了 SE 与 ML 的桥梁。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **HumanEval+**：基于原始 HumanEval 的增强版本，包含 164 个手写 Python 编程题，附带单元测试。
- 用于评估生成代码的 **functional correctness**，即是否能通过所有测试用例。

### 🧠 选用的 LLMs
从 BigCodeBench 榜单中选择三个开源、小于 8B 参数、支持完整解码控制的模型：

| 模型名称 | 家族 | 参数量 |
|--------|-----|-------|
| OpenCoder-8B-Instruct | infly | 8B |
| Qwen2.5-Coder-7B-Instruct | Qwen | 7.62B |
| Qwen2.5-Coder-3B-Instruct | Qwen | 3.09B |

后续简称：Qwen-7B, Qwen-3B, OpenCoder-8B。

---

### ⚙️ 实验设置

| 项目 | 设置说明 |
|------|----------|
| **推理框架** | Hugging Face Transformers |
| **采样配置数** | 共 254 个配置：<br>• YASA (2-wise): 77<br>• ICPL (2-wise): 81<br>• RANDOM: 96 |
| **每配置运行次数** | 10 次重复 |
| **总提示数量** | 7,620 runs × 164 prompts = ~1.25 million prompts |
| **硬件平台** | AMD EPYC 7513 CPU, 4×NVIDIA A100-SXM4-40GB GPUs |
| **精度模式** | bfloat16 |
| **批大小** | 32 |
| **最大 token 数** | 512 |
| **设备映射** | `device_map="auto"` |

#### 测量流程设计
- **Calibration Phase**（30秒）：校准能耗系数 + 获取 idle power baseline。
- **Warm-up Phase**：6轮预热以消除冷启动效应（CUDA 编译、内存分配等）。
- **Energy Measurement 工具**：
  - CPU: `perf` + RAPL
  - GPU: `nvidia-smi`
- **Latency**：请求到响应的时间间隔。
- **Accuracy**：使用 **pass@1** 指标（Chen et al., 2021），衡量单次采样通过测试的概率。

---

### 🎯 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **预测性能** | R², MAE, MAPE | 衡量回归模型准确性 |
| **配置有效性** | Pareto Front | 展示 energy vs. accuracy 的最优边界 |
| **参数影响力** | Feature-wise / Pairwise Analysis | 量化各 feature 对指标的平均影响 |

---

### 🔁 基线方法对比
本文未直接对比传统调参方法（如 Grid Search、Bayesian Optimization），而是比较了三种**采样策略**作为训练数据来源时，所构建预测模型的性能差异：
- **YASA**：先进的 t-wise 覆盖采样器
- **ICPL**：另一种高效 t-wise 采样器
- **RANDOM**：随机采样（基准）
- **ALL**：三者合并（最佳数据多样性）

> 这种设计体现了“**采样质量决定预测上限**”的思想。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（来自 Table 5）

| Sampler | Energy (R² / MAE) | Latency (R² / MAE) | Accuracy (R² / MAE) |
|--------|-------------------|--------------------|----------------------|
| **YASA** | 0.94 / 12.23 kJ | 0.93 / 27.72 s | 0.99 / 0.01 |
| **ICPL** | 0.74 / 19.59 kJ | 0.78 / 37.77 s | 0.97 / 0.02 |
| **RANDOM** | 0.91 / 15.94 kJ | 0.91 / 32.99 s | 0.96 / 0.02 |
| **ALL** ✅ | **0.95 / 10.05 kJ** | **0.94 / 23.78 s** | **0.99 / 0.01** |

> ✅ **结论**：结合多种采样策略（ALL）效果最好；即使是简单随机采样也能取得不错结果，表明该方法对数据不敏感且鲁棒。

---

### 🔍 核心发现（RQ1–RQ3 回答）

#### ✅ RQ1: 变体建模能否系统识别关键参数？
- **Energy & Latency 强相关**（R² ≈ 0.9），主要受以下因素驱动：
  - `cache=offloaded` ➝ 最大能耗增加（+89.48 kJ）
  - `num_beams=4`, `num_beam_groups=4` ➝ 增加前向传播次数 ➝ 高耗能
- **Accuracy 主要由解码策略和重复控制决定**：
  - `decoding=greedy` ➝ 准确率最高（图5b）
  - `no_repeat_ngram_size=0` ➝ 更自由生成，提升准确率
  - `low_memory=True` ➝ 微降准确率（-0.23 pass@1）

#### ✅ RQ2: 最优权衡是什么？（Pareto Front 分析）
- **低能耗区（~20–25 kJ）**：
  - 小幅增能即可大幅提升 accuracy（0.66 → 0.75）
  - 最佳配置：`Qwen-7B + dynamic cache + greedy/contrastive decoding + low temp (~0.3)`
- **平衡区（~35–40 kJ）**：
  - accuracy 提升放缓（0.75 → 0.77）
  - 使用 `beam search (2 beams)` + `sampling=True` + `top_p=0.8–0.85`
- **高精度区（49–64 kJ）**：
  - accuracy 达峰值 0.80，但能耗上升 66%
  - 配置更复杂：3–4 beams, hybrid/static cache, higher temp/top-p

> 💡 **洞察**：大部分收益可在低能耗区域获得，边际效益递减明显。

#### ✅ RQ3: 是否可以准确预测未见配置？
- 是！使用仅 254 个样本训练的模型，在测试集上表现优异：
  - **Energy**: R² = 0.95, MAE = 10.05 kJ
  - **Latency**: R² = 0.94, MAE = 23.78 s
  - **Accuracy**: R² = 0.99, MAE = 0.01
- **采样策略显著影响性能**：
  - ICPL 表现最差 ➝ 说明仅靠“全覆盖”不够，需多样化
  - YASA 和 RANDOM 表现接近，ALL 最优 ➝ 多样性是关键

---

### 🔪 消融实验（隐含在采样策略对比中）
虽然没有明确称为“ablation”，但不同采样策略的对比本质上是一次消融研究：

| 实验 | 发现 |
|------|------|
| 仅用 ICPL | 性能最差 ➝ 单一采样器不足以捕捉全局行为 |
| 加入 RANDOM | 显著提升 ➝ 随机性有助于发现边缘案例 |
| 合并所有 | 达到最优 ➝ 数据多样性至关重要 |

> ✅ **结论**：**多样化的采样策略组合** 是构建高质量预测模型的关键。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **LLM 推理是一个典型的可配置系统**，适合用 **Feature Model** 进行建模。
2. **变体建模能有效管理配置复杂性**，防止非法配置，支持自动化分析。
3. **energy 与 latency 高度正相关**，但 accuracy 由独立机制控制，存在明显 trade-off。
4. **关键参数影响显著**：
   - `offloaded cache` 是能耗大户；
   - `greedy decoding` 同时高效且准确；
   - `beam search` 提升 accuracy 但代价高昂。
5. **少量代表性样本足以训练高精度预测模型**（R² > 0.94），尤其当数据来源多样时。
6. **Pareto front 可指导实际部署决策**：多数 accuracy 提升发生在低能耗区间。

---

### ⚠️ 局限性与有效性威胁

| 类型 | 说明 |
|------|------|
| **外部有效性（External Validity）** | • 仅使用 **HumanEval+**（代码生成任务）<br>• 仅测试 **Hugging Face Transformers**<br>• 仅评估 **3 个 LLMs** ➝ 结果可能不适用于摘要、翻译等任务或其他 inference engine（如 vLLM, TGI） |
| **构造有效性（Construct Validity）** | • 连续参数被离散化（如 temperature ∈ {0.1, 0.3, ..., 1.2}）<br>• 约 3% 的生成配置无法运行 ➝ 约束建模不完全 |
| **内部有效性（Internal Validity）** | • 实验在固定硬件（A100）上进行 ➝ 无法反映异构环境下的变化 |
| **维护成本** | • LLM 生态快速演进（新架构、量化方法）→ FM 需持续更新 |

---

### 🔮 未来工作方向（作者提出）

1. **扩展变体模型层级**：
   - 加入 **hardware configuration**（GPU 类型、内存）、**deployment options**（batch size, quantization）等维度。
2. **实现运行时自适应重配置**（adaptive reconfiguration）：
   - 动态调整配置以应对负载波动或目标切换（节能 vs. 高速）。
3. **推广至其他推理服务器**：
   - 构建 vLLM、TGI 等系统的 FM，并建立统一抽象接口（abstract FM）。
4. **自动约束推断**：
   - 利用 LLM 或 ML 技术自动从文档/日志中提取参数约束，降低建模成本。

---

## ✅ 总结一句话

> 本文开创性地将**变体建模**（variability modeling）引入 LLM 推理优化，通过构建 **Feature Model + 采样 + 测量 + 学习** 的闭环框架，实现了对超大规模配置空间的系统分析与精准预测，为绿色、可持续、智能化的 LLM 部署提供了新范式。

</details>

---

### 6. [Dual Length Codes for Lossless Compression of BFloat16](https://arxiv.org/abs/2602.17849)

**Authors**: Aditya Agrawal, Albert Magyar, Hiteshwar Eswaraiah, Patrick Sheridan, Pradeep Janedula, Ravi Krishnan Venkatesan, Krishna Nair, Ravi Iyer  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.17849v1  

#### Abstract
Training and serving Large Language Models (LLMs) relies heavily on parallelization and collective operations, which are frequently bottlenecked by network bandwidth. Lossless compression using e.g., Huffman codes can alleviate the issue, however, Huffman codes suffer from slow, bit-sequential decod...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Dual Length Codes for Lossless Compression of BFloat16*

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模语言模型（LLMs）的训练与推理过程中，**并行计算和集体通信操作**（如 AllReduce、AllGather）频繁发生，而这些操作通常受限于网络带宽。为缓解这一瓶颈，常采用**无损压缩技术**减少传输数据量。然而，传统方法存在以下问题：
- **Huffman Codes** 虽然压缩效率高（接近熵极限），但解码是**位串行（bit-sequential）** 的，依赖深度树遍历，导致**解码延迟高、硬件实现复杂**。
- **Universal Codes**（如 Exponential-Golomb、Elias Gamma）解码较快，不完全依赖逐位解析，但**未利用符号频率分布**，压缩率较低。

### 提出的新方法：Dual Length Codes
本文提出一种新型混合编码方案——**Dual Length Codes**，其核心思想是结合 Huffman 编码的频率感知优势与 Universal Codes 的快速解码特性。

#### 创新点：
- 将 256 个 BFloat16 符号划分为两个区域：
  - **高频区（Top-8 Symbols）**：占累计概率约 50%，分配 **4-bit 短码**。
  - **低频区（Remaining 248 Symbols）**：分配 **9-bit 长码**。
- 使用 **单个前缀 bit** 区分码长（0 表示短码，1 表示长码）。
- 编解码通过一个仅含 8 项的小型 **Look Up Table (LUT)** 实现，极大简化硬件设计。

### 相比现有方法的优势
| 特性 | Huffman Codes | Universal Codes | **Dual Length Codes** |
|------|----------------|------------------|------------------------|
| 压缩效率 | ✅ 最优（~21.3%） | ❌ 较低 | ⚠️ 略低（18.6%） |
| 解码速度 | ❌ 慢（树遍历） | ✅ 快 | ✅ 快（固定长度段读取） |
| 硬件复杂度 | ❌ 高（深树） | ✅ 低 | ✅ 极低（小 LUT + 固定格式） |
| 实现简易性 | ❌ 复杂 | ✅ 简单 | ✅ 简单 |

> **核心权衡**：以 **2.7% 的压缩率损失** 换取显著提升的**解码速度与硬件友好性**。

---

## 2. 核心实验方法和设置

### 数据集
- 来源于 **Gemma 2B 模型** 在 **Supervised Fine-Tuning (SFT)** 阶段的中间张量。
- 具体分析对象为 Feed-Forward 层中的 **FFN1 Activation Tensor**，数据类型为 **BFloat16**。
- 总共分析了 18 层 × 64 TPU shards = **1152 个 shard** 的激活值，结果具有统计代表性。
- 其他张量（如 FFN2 activation、gradients）也表现出类似分布趋势。

### 实验设置与评估指标
- **符号粒度**：将 BFloat16 视为 8-bit symbol（共 256 种可能值），便于统计频率分布。
- **分析工具**：
  - 绘制 **PMF（Probability Mass Function）** 和 **CDF（Cumulative Distribution Function）**。
  - 对比 Huffman 与所提方法的 **code length 分布**。
- **评估指标**：
  - **Compressibility（压缩率）**：定义为 `(原始比特数 - 平均编码长度) / 原始比特数`。
  - **Expected Bits per Symbol**：加权平均编码长度。
  - **Implementation Complexity**：基于是否需要树结构、LUT 大小等定性评估。

### 基线方法对比
- **Huffman Coding**：作为最优熵编码基准。
- （隐含对比）Universal Codes：虽未直接测试性能，但在背景中指出其缺乏频率建模能力。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 数值 |
|------|------|
| Top-8 符号累计概率 | ~50% |
| Huffman 编码压缩率 | **21.3%** |
| Dual Length Codes 压缩率 | **18.6%** |
| 理论预期压缩率（分析估算） | 18.75%（吻合良好） |
| 平均编码长度 | 6.5 bits/symbol |
| LUT 大小 | 仅需存储 8 个高频符号映射 |

### 与基线方法对比结果
- **压缩效率**：
  - Dual Length Codes 比 Huffman 低约 **2.7%**，但仍远优于通用编码。
- **解码性能**：
  - Huffman 需要最多 10 层树查找，延迟随路径增长。
  - Dual Length Codes 解码仅需：
    - 读取首位判断码长；
    - 若为短码（概率 50%），查 LUT 得原符号；
    - 若为长码，直接读后续 8 位恢复符号。
  - **避免了树遍历，支持并行位提取，显著降低延迟**。
- **硬件资源消耗**：
  - Huffman 需大容量编码表或复杂状态机。
  - Dual Length Codes 只需 **8-entry LUT + 固定逻辑电路**，适合 ASIC/FPGA 实现。

### 消融实验（间接体现）
虽然没有明确命名“ablation study”，但文中通过以下方式验证设计选择：
- 分析不同符号数量划分的影响（为何选 Top-8？）：
  - 图 2 显示 Top-8 正好达到 CDF ≈ 0.5，是自然分割点。
- 分析 code length 分布（图 3 & 4）：
  - 发现多数符号 Huffman code length 接近 9 bits → 支持将剩余符号统一用 9-bit 编码的合理性。

---

## 4. 关键结论和发现

### 主要发现
1. **BFloat16 张量具有强偏态频率分布**：
   - 少数符号（Top-8）占据约一半出现概率，适合针对性优化。
2. **Dual Length Codes 是高效的折中方案**：
   - 在可接受的压缩率损失下，换取了解码速度和硬件实现的巨大优势。
3. **实际部署友好性强**：
   - 小 LUT 可预生成并固化在芯片中，适用于 TPU/GPU 内部通信压缩模块。

### 方法的局限性
- **静态编码假设**：当前方案基于离线统计得到的固定 LUT，若不同阶段或不同模型分布变化较大，可能需动态更新 LUT。
- **精度限制**：仅针对 BFloat16 的 8-bit 分组处理，未考虑更细粒度或跨元素相关性。
- **适用范围**：主要验证于 Gemma 模型的 activation tensor，其他类型（如 gradients）虽趋势相似，仍需进一步验证泛化性。

### 未来工作方向
- 扩展至更多层级的 **Multi-Length Codes**（如三档长度），在压缩率与速度间进一步调优。
- 探索 **自适应 Dual Length Coding**，根据运行时分布动态调整高频符号集合。
- 将该编码集成到 **NCCL-like 通信库** 或 **TPU 内核层**，实现端到端加速。
- 结合量化或其他压缩技术进行联合优化。

---

> **总结一句话**：  
> *Dual Length Codes* 提出了一种面向 ML 场景的轻量级无损压缩编码，**以微小压缩率代价，换来了极简解码逻辑与卓越硬件效率**，特别适合用于 LLM 训练中高频率的 collective communication 场景。

</details>

---

### 7. [Learning Long-Range Dependencies with Temporal Predictive Coding](https://arxiv.org/abs/2602.18131)

**Authors**: Tom Potter, Oliver Rhodes  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.18131v1  

#### Abstract
Predictive Coding (PC) is a biologically-inspired learning framework characterised by local, parallelisable operations, properties that enable energy-efficient implementation on neuromorphic hardware. Despite this, extending PC effectively to recurrent neural networks (RNNs) has been challenging, pa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning Long-Range Dependencies with Temporal Predictive Coding*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **传统 BPTT 的缺陷**：Backpropagation Through Time (BPTT) 是训练 RNN 的主流方法，但其存在非局部计算、缺乏空间并行性、需存储完整激活历史等问题，导致高能耗，难以在低功耗的 **neuromorphic hardware** 上部署。
- **Predictive Coding (PC)** 虽然具有生物启发性和本地化、可并行的优点，但在处理 **long-range temporal dependencies** 时表现不佳，尤其在复杂序列任务中性能远低于 BPTT。

### 🚀 提出的新方法
- **tPC RTRL**：本文首次将 **Temporal Predictive Coding (tPC)** 与近似版本的 **Real-Time Recurrent Learning (RTRL)** 结合，提出 **tPC RTRL** 算法。
- 创新机制：
  - 引入 **influence matrix** $ M(t) $ 来追踪参数对当前隐藏状态的历史影响，从而实现跨时间步的信用分配（credit assignment）。
  - 修改 RTRL 的矩阵更新规则，使其适用于 tPC 中基于优化的隐状态收敛过程（即用实际收敛的 $ x(t) $ 替代预测值 $ \mu(t) $）。

### 🔍 相比现有方法的优势
| 特性 | BPTT | tPC (baseline) | tPC RTRL (本文) |
|------|------|----------------|------------------|
| 局部性 | ❌ 非局部反向传播 | ✅ 完全局部 | ✅ 保持局部性 |
| 并行性 | ❌ 时间上串行 | ✅ 可空间并行 | ✅ 支持空间并行 |
| 内存随序列长度增长 | ✅ 线性增长 | ✅ 固定 | ✅ 固定 |
| 能效潜力 | ❌ 高能耗 | ✅ 适合 neuromorphic | ✅ 更优，支持在线学习 |
| 长程依赖建模能力 | ✅ 强 | ❌ 弱 | ✅ 接近 BPTT |

> ✅ **核心优势**：在保留 PC 的生物合理性、局部性和能效优势的同时，显著提升了对长程依赖的学习能力。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
1. **Synthetic Copy Task**  
   - 输入：30 个数字（1–9），后接 10 步延迟输出原序列。
   - 目标：测试模型捕捉长距离依赖的能力（信息与目标相隔 10 步以上）。

2. **WikiText-2**  
   - 规模：约 200 万 token，用于语言建模。
   - 任务：next-token prediction。
   - 模型：Linear Recurrent Unit (LRU)，便于实现高效 RTRL。

3. **CCMatrix 英法翻译子集**  
   - 数据量：60 万句对（50 万训练，各 5 万验证/测试）。
   - 序列长度限制：≤128 tokens。
   - 任务：机器翻译，要求编码整个源句后再生成目标句，挑战长程依赖。

### ⚙️ 实验设置与评估指标
| 设置项 | 描述 |
|-------|------|
| 模型架构 | 多数采用单层 LRU 或简单 RNN，部分任务使用两层读出头 |
| 参数规模 | 最大达 **15 million**（翻译任务） |
| 优化器 | Adam（BP 方法）；Inference Learning + SGD（PC 方法） |
| 评估指标 |  
- Copy Task：Validation Accuracy, Cross-Entropy Loss  
- WikiText-2：Test Perplexity  
- 翻译任务：Test Perplexity, BLEU Score  

### 🆚 对比的基线方法
- **BPTT**：标准时序反向传播，性能上限参考。
- **Spatial BP**：仅使用当前时间步梯度，无历史依赖。
- **Baseline tPC**：原始 Temporal Predictive Coding，无 RTRL 增强。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ Copy Task（表 1）
| Method | Val Loss | Val Accuracy |
|--------|----------|--------------|
| BPTT | 0.0176 ± 0.0020 | 0.9993 ± 0.0003 |
| tPC RTRL | 0.0574 ± 0.0028 | **1.0000 ± 0.0000** |

> 💡 **结论**：tPC RTRL 成功解决长程复制任务，准确率媲美 BPTT，而 baseline tPC 和 spatial BP 完全失败。

#### ✅ WikiText-2（表 2）
| Method | Test Perplexity ↓ |
|--------|-------------------|
| Spatial BP | 103.38 ± 0.39 |
| tPC | 108.99 ± 0.54 |
| BPTT | 98.62 ± 0.23 |
| **tPC RTRL** | **99.19 ± 0.18** |

> 💡 **结论**：tPC RTRL 显著优于普通 tPC 和 spatial BP，接近 BPTT 表现，说明其具备建模一定时序依赖的能力。

#### ✅ 英法机器翻译（表 3）
| Method | Test Perplexity ↓ | Test BLEU ↑ |
|--------|-------------------|-------------|
| Spatial BP | 16.03 | 8.93 |
| tPC | 28.31 | 3.07 |
| BPTT | **7.49** | **21.11** |
| **tPC RTRL** | **7.62** | **20.71** |

> 💡 **结论**：
- tPC RTRL 在复杂任务上几乎达到 BPTT 性能（perplexity 仅差 1.7%，BLEU 差 ~1.9%）。
- baseline tPC 表现极差，甚至不如 spatial BP，凸显其长程建模缺陷。
- 图 3 显示 tPC RTRL 与 BPTT 收敛速度和稳定性高度一致。

### 🔬 消融分析（隐含）
- **是否引入 RTRL 影响巨大**：从 tPC 到 tPC RTRL 的提升是质变而非量变，尤其在翻译任务中。
- **influence matrix 近似有效**：尽管使用了 $ x(t) $ 替代 $ \mu(t) $ 的近似，仍能稳定训练并取得优异结果。
- **超参敏感性**：tPC 类方法需要仔细调整 inference learning rate 和 iteration 数，否则难以收敛（如翻译任务中 tPC 无法调通）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **tPC RTRL 可有效学习长程依赖**：首次证明 PC 框架可通过结合 RTRL 实现与 BPTT 相当的时序建模能力。
2. **性能逼近 BPTT**：在合成任务和真实世界任务（尤其是 15M 参数的翻译模型）上，tPC RTRL 的 perplexity 和 BLEU 分数均非常接近 BPTT。
3. **保持 PC 的理想属性**：算法仍由局部、可并行的操作构成，无需展开整个序列，内存不随长度增加，适合 **edge AI** 和 **neuromorphic hardware** 部署。
4. **为在线学习提供可能路径**：由于其固定内存开销和实时更新特性，tPC RTRL 是实现 **online learning** 的有力候选者。

### ⚠️ 方法的局限性
1. **计算复杂度仍较高**：虽然使用 LRU 实现了 $ O(P) $ 的 influence matrix 存储，但对于大规模多层网络，仍需扩展至多层近似。
2. **超参数调节困难**：PC-based 方法缺乏成熟的“最佳实践”，inference learning rate、iterations 等需手动调优，且不同任务差异大。
3. **尚未在硬件上验证能效**：虽理论上更节能，但尚未在 Loihi 等 neuromorphic chip 上实测能量消耗，效率主张仍待验证。
4. **目前限于单层 RNN**：多层 RNN 需维护多个 influence matrices，尚未在本工作中充分探索。

### 🔮 未来工作方向
1. **扩展至深层架构**：开发适用于 multi-layer RNN 的 layer-local RTRL approximation，并集成到 tPC RTRL 框架中。
2. **探索其他 recurrent cell**：尝试 element-wise LSTM 或其他稀疏/分解形式的 RTRL 近似以进一步降低开销。
3. **硬件实现与能效测量**：将 tPC RTRL 部署到 neuromorphic 平台（如 Loihi），进行真实的 energy profiling。
4. **改进训练稳定性**：借鉴近期工作（如 Qi et al., 2025）优化 free energy 设计，提升 PC 网络的收敛鲁棒性。
5. **应用于更多序列任务**：如语音识别、视频预测等，验证泛化能力。

---

## ✅ 总结
> **tPC RTRL 是一项兼具理论意义与工程潜力的重要进展** —— 它成功弥合了生物启发式学习框架（PC）与高性能序列建模之间的鸿沟，在不牺牲局部性与能效的前提下，实现了对 BPTT 的性能逼近。这为下一代低功耗、可在线学习的智能系统提供了关键技术路径。

</details>

---

### 8. [Parallel Complex Diffusion for Scalable Time Series Generation](https://arxiv.org/abs/2602.17706)

**Authors**: Rongyao Cai, Yuxi Wan, Kexin Zhang, Ming Jin, Zhiqiang Ge, Qingsong Wen, Yong Liu  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.17706v1  

#### Abstract
Modeling long-range dependencies in time series generation poses a fundamental trade-off between representational capacity and computational efficiency. Traditional temporal diffusion models suffer from local entanglement and the $\mathcal{O}(L^2)$ cost of attention mechanisms. We address these limi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Parallel Complex Diffusion for Scalable Time Series Generation

## 1. 论文的主要贡献和创新点

### 解决的问题
传统时间序列生成模型（如基于时间域的 Diffusion Models）面临两个核心挑战：
- **局部纠缠（Local Entanglement）**：时间序列中相邻时刻存在强依赖关系，导致建模长程依赖困难。
- **计算复杂度高**：主流架构（如 DiT）中的 Self-Attention 机制具有 $O(L^2)$ 的计算复杂度，难以扩展到长序列。

此外，现有频域扩散方法多为启发式设计，缺乏理论基础，且未充分利用频域信号的统计独立性进行并行化建模。

### 提出的新方法：PaCoDi (Parallel Complex Diffusion)
本文提出 **PaCoDi**，一种全新的、在复数频域中进行并行扩散的生成框架。其核心思想是将扩散过程从时域转移到频域，并利用傅里叶变换的对角化特性解耦信号。

#### 主要创新点：
- **频域扩散的数学基础**：
  - 提出 **Quadrature Forward Diffusion** 和 **Conditional Reverse Factorization Theorem**，证明在频域中，实部（Real）和虚部（Imaginary）的扩散过程可以完全解耦。
  - 这为并行建模提供了严格的数学依据，避免了复数神经网络中的全纯性（Holomorphicity）约束。

- **并行架构设计**：
  - 构建两个独立的分支分别处理实部和虚部，实现真正的并行训练与推理。
  - 通过 **Mean Field Theory (MFT)** 近似来桥接“理想解耦”与“实际数据耦合”之间的差距。

- **交互式修正机制（Interactive Correction Mechanism）**：
  - 在并行分支间引入轻量级的交叉注意力模块，恢复因 MFT 近似而丢失的相位等跨域依赖信息。

- **效率优化**：
  - 利用实信号的 **Hermitian Symmetry** 性质，将序列长度压缩一半，直接减少 50% 的 Attention FLOPs。
  - 推导出适用于压缩流形的 **Heteroscedastic Loss**，以处理非各向同性的噪声分布。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **理论性** | 首次建立频域扩散的严格数学体系，统一离散 DDPM 与连续 SDE 框架。 |
| **效率** | 实现 50% 的 Attention FLOPs 减少，显著提升长序列生成速度。 |
| **表达能力** | 绕过复数网络的 Holomorphic 约束，可自由使用 SiLU/GELU 等非线性激活函数。 |
| **生成质量** | 在多种任务上超越 SOTA，尤其在长序列场景下表现更稳定。 |

---

## 2. 核心实验方法和设置

### 数据集
#### 条件生成（Conditional Generation）
遵循 T2S 设置，在以下单变量数据集上进行片段级生成：
- **ETTh1**, **ETTm1**（电力变压器温度）
- **ECL**（电力消耗）
- **Exchange**（汇率）
- **Air Quality**（空气质量）

#### 无条件生成（Unconditional Generation）
在多变量数据集上合成完整时间序列：
| 数据集 | 样本数 | 维度 |
|--------|--------|------|
| **ETTh1** | 17,420 | 7 |
| **Stocks** | 3,773 | 6 |
| **Sines** | 10,000 | 5 |
| **Air Quality** | 9,333 | 15 |

### 实验设置
- **序列长度（Horizon L）**：24, 48, 96（条件），24, 64, 128, 256（无条件）
- **模型配置**：基于 DiT 架构，隐藏维度随 L 缩放以保持容量一致。
- **PaCoDi 变体**：
  - `PaCoDi DDPM`：离散扩散版本
  - `PaCoDi SDE`：连续 SDE 版本

### 评估指标
#### 无条件生成
1. **Discriminative Score**：判别器区分真实/合成样本的准确率（越低越好）
2. **Predictive Score (TSTR)**：在合成数据上训练、在真实数据上测试的预测误差
3. **Context-FID**：潜在表示的 Wasserstein 距离（越低越好）
4. **Correlational Score**：相关矩阵差异的 Frobenius 范数

#### 条件生成
- **MSE**（均方误差）
- **WAPE**（加权绝对百分比误差）

### 基线方法对比
| 类型 | 基线模型 |
|------|----------|
| **专用生成模型** | Diffusion-TS, TimeVAE, TimeGAN, Temporal DDPM |
| **监督模型** | T2S |
| **大语言模型（零样本）** | GPT-4o-mini, Llama3.1-8b |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（平均值）

#### 无条件生成（Table 3）
| Model | Discriminative ↓ | Predictive ↓ | Context-FID ↓ | Correlational ↓ |
|-------|------------------|--------------|---------------|-----------------|
| **PaCoDi SDE** | **0.202** | **0.041** | **0.074** | **0.107** |
| **PaCoDi DDPM** | **0.215** | **0.049** | **0.087** | **0.112** |
| Diffusion-TS | 0.927 | 0.086 | 0.136 | 0.117 |
| TimeVAE | 0.899 | 0.072 | 0.176 | 0.119 |
| TimeGAN | 15.771 | 1.440 | 0.438 | 0.161 |
| DDPM | 0.556 | 0.042 | 0.138 | 0.116 |

> ✅ **PaCoDi 在 16 项指标中获得 14 个第一**，全面领先。

#### 条件生成（Table 4）
| Model | WAPE ↓ | MSE ↓ |
|-------|--------|------|
| **PaCoDi SDE** | **0.131** | **0.010** |
| **PaCoDi DDPM** | **0.132** | **0.009** |
| T2S | 0.176 | 0.018 |
| Diffusion-TS | 0.763 | 0.058 |
| TimeVAE | 0.209 | 0.017 |
| GPT-4o-mini | 0.924 | 0.956 |
| Llama3.1-8b | 1.016 | 1.432 |

> ✅ 在 **19/20** 个评估类别中取得最佳或第二佳成绩。

### 与基线方法的对比结果
- **生成质量**：
  - PaCoDi 显著优于所有专用生成模型（Diffusion-TS、TimeVAE 等）。
  - 大语言模型（LLMs）在数值精度和高频波动建模上表现极差，不适合原始时间序列生成。
- **长序列稳定性**：
  - 当 $L=256$ 时，Diffusion-TS 性能明显下降，而 PaCoDi 保持稳定，验证其对“长程纠缠”的鲁棒性。
- **可视化效果**：
  - 图2显示，PaCoDi 生成的数据密度、PCA 和 t-SNE 分布与真实数据高度一致，优于 Diffusion-TS 和标准时域扩散。

### 消融实验结果（Table 5）
研究了不同组件的影响（在 Sines 数据集上）：

| 模型 | Context-FID ↓ | Discriminative ↓ |
|------|----------------|------------------|
| **PaCoDi (Full)** | **0.021** | **0.016** |
| Decoupled Only (Dec.) | 0.017 | 0.146 |
| Temporal Baseline (Temp.) | 0.023 | 0.031 |

- **仅解耦（Dec.）**：虽然计算高效，但生成质量严重退化，说明忽略跨象限依赖会导致相位信息丢失。
- **PaCoDi 完整版**：通过 **Interactive Correction** 成功恢复依赖关系，性能反超时域基线。

---

## 4. 关键结论和发现

### 主要发现
1. **频域是解耦的理想空间**：傅里叶变换作为“对角化算子”，天然地将纠缠的时间信号转换为统计独立的频谱分量。
2. **并行扩散可行且高效**：通过理论证明，频域扩散的实部与虚部可完全解耦，支持真正意义上的并行建模。
3. **50% Attention 加速可实现**：结合 Hermitian Symmetry 压缩与并行架构，PaCoDi 在理论上和实践中均实现了 50% 的 Attention FLOPs 减少。
4. **MFT + 交互修正 是关键**：完全解耦会损失相位一致性，必须通过交互机制补偿，才能兼顾效率与保真度。

### 方法的局限性
- **依赖傅里叶变换**：假设信号平稳或准平稳，对于剧烈非平稳信号可能需结合小波等时频分析工具。
- **复数运算开销**：尽管节省了 Attention，但增加了 FFT/iFFT 的固定开销，对短序列增益有限。
- **初始化敏感性**：压缩后的流形需要特殊处理噪声协方差（Heteroscedastic Loss），实现稍复杂。

### 未来工作方向
- 将 PaCoDi 扩展至 **Spatio-Temporal** 数据（如交通、气象）。
- 结合 **Wavelet** 或 **Learnable Basis** 以适应非平稳信号。
- 探索 **Hybrid Domain** 建模，融合时域局部细节与频域全局模式。
- 开发面向 PaCoDi 的专用硬件加速方案，最大化其并行潜力。

> **总结**：PaCoDi 不仅是一个高性能的时间序列生成器，更提供了一种“**通过变换改变问题拓扑**”的新范式，为高效、可扩展的生成建模开辟了新路径。

</details>

---

### 9. [El Agente Gr\'afico: Structured Execution Graphs for Scientific Agents](https://arxiv.org/abs/2602.17902)

**Authors**: Jiaru Bai, Abdulrahman Aldossary, Thomas Swanick, Marcel M\"uller, Yeonghun Kang, Zijian Zhang, Jin Won Lee, Tsz Wai Ko, Mohammad Ghazi Vakili, Varinia Bernales, Al\'an Aspuru-Guzik  
**Category**: cs.AI  
**Published**: 2026-02-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.17902v1  

#### Abstract
Large language models (LLMs) are increasingly used to automate scientific workflows, yet their integration with heterogeneous computational tools remains ad hoc and fragile. Current agentic approaches often rely on unstructured text to manage context and coordinate execution, generating often overwh...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：El Agente Gráfico: Structured Execution Graphs for Scientific Agents

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Large Language Models (LLMs)** 的科学智能体（Scientific Agents）在自动化科研流程时面临以下核心挑战：
- **上下文脆弱性**：多数系统依赖非结构化的文本传递信息，导致上下文膨胀、决策溯源困难、审计性差。
- **执行状态管理薄弱**：计算中间产物（如分子构型、电子态等）通常以原始文本或文件形式传递，难以高效复用和验证。
- **多工具异构集成困难**：不同软件包（如RDKit、PySCF、CREST）间的数据格式不兼容，依赖硬编码转换或不可靠的LLM“胶水”逻辑，易出错且难以扩展。

这些问题使得现有系统在复杂、多步、并行的科学任务中表现不稳定，尤其在需要高数值精度和状态一致性的量子化学与材料设计领域。

---

### 提出了什么新方法或新思路
本文提出 **EL AGENTE GRAFICO**，一个单智能体（single-agent）框架，其核心创新在于将 **结构化执行图（Structured Execution Graphs）** 和 **类型安全（type safety）** 引入科学智能体的设计中：

1. **结构化执行图（Execution Graphs）**  
   将科学工作流抽象为由节点（如单点能计算、几何优化）和有向边组成的图。每个节点代表一个可验证的状态变换，避免了自由文本描述带来的歧义。

2. **对象图映射器（Object Graph Mapper, OGM）**  
   开发了一个定制化的OGM，将Python中的科学对象（如`ConceptualAtoms`）序列化到外部 **知识图谱（Knowledge Graph, KG）** 中，并通过唯一的国际资源标识符（IRI）进行引用。这实现了：
   - 内存与持久化存储的统一视图
   - 跨工具、跨会话的轻量级状态传递

3. **概念原子抽象层（ConceptualAtoms）**  
   定义了一个统一的内存接口，用于表示分子和周期性体系，封装电荷、自旋多重度等元信息，并支持在RDKit、ASE、pymatgen等工具间无缝转换。

4. **路由控制器（Routing Agent）**  
   在图中动态选择下一个执行节点，使用schema-conditioned的结构化输出确保输入参数合法，实现安全的条件跳转和循环控制。

---

### 相比现有方法的优势
| 维度 | 传统多智能体/文本中心方法 | EL AGENTE GRAFICO |
|------|--------------------------|------------------|
| **架构** | 多智能体分解，协调开销大 | 单智能体，低通信开销 |
| **状态管理** | 文本/文件传递，易丢失 | 类型化对象 + KG持久化 |
| **可审计性** | 决策链模糊 | 全过程可追溯，支持KG查询 |
| **并行性** | 难以协调 | 支持GPU级并行（3并发/GPU） |
| **成本效率** | 上下文“雪球效应”严重 | Token消耗降低14倍以上 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **大学水平量子化学练习题**（源自 *El Agente Q*）：共6类任务，每类含两个难度等级（Level 1 & 2），总计12个任务，重复10次，形成120次运行的基准测试集。
  - 包括有机/无机分子分析、氢提取反应能、环烷烃环张力、pKa预测、TDDFT激发态能量等。
- **构象集合生成**（Conformer Ensemble Generation）
  - 使用 **CREST** 在GFN2-xTB/ALPB级别对merocyanine化合物在水和庚烷中的溶剂效应进行采样。
- **金属有机框架（MOF）设计空间探索**
  - 基于 **CoRE-MOF** 数据库中的实验结构（如IXEJIG, VEGBUG），结合 **PORMAKE** 构建假设MOF，并通过 **MACE** 和 **Zeo++** 进行几何优化与孔隙率分析。

---

### 实验设置和评估指标

#### 评估策略
采用 **双评估器设计（Dual-evaluator design）**：
1. **数值评估器（Deterministic Numerical Checker）**  
   验证输出对象的正确性，包括：
   - 功能泛函、基组、电荷、多重度是否匹配
   - 总能量误差 < 0.01 Ha
   - 几何RMSD < 0.15 Å
   - 是否存在虚频（稳定极小值要求）
   - 偶极矩、HOMO-LUMO间隙、点群对称性等衍生性质

2. **LLM裁判评估器（LLM-as-a-judge）**  
   使用独立LLM（如gpt-4o）评估任务完成度、推理质量、报告完整性，评分范围0–1。

#### 关键性能指标
| 指标 | 描述 |
|------|------|
| **Numerical Eval (%)** | 数值正确率 |
| **LLM Judge Eval (%)** | LLM裁判评分均值 |
| **Trace Tokens** | 整个代理轨迹的总Token数 |
| **Token Cost (USD)** | 按官方价格计算的总费用（未考虑缓存） |
| **Task Duration (s)** | 平均任务耗时（秒） |
| **Context Window Saturation (%)** | 最终API请求占最大上下文窗口的比例 |
| **Error Recovery Cost (%)** | 错误恢复相关的额外开销占比 |
| **Carryover Tokens (%)** | 可缓存Token占总量的比例 |

#### 基线方法对比
- **El Agente Q (5)**：前作，基于多智能体架构，使用 `sonnet-3.7`。
- **轻量级LLM代理（Bare LLM Agent）**：仅配备Web搜索和代码执行能力，作为“从零开始写脚本”的对照。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| LLM | Numerical (%) | LLM Judge (%) | Trace Tokens | Token Cost (USD) | Duration (s) |
|-----|---------------|----------------|--------------|------------------|--------------|
| **gpt-5** | **98.88** | **98.50** | **83,613** | **0.17** | **228** |
| gpt-4.1 | 93.71 | 96.52 | 113,175 | 0.25 | 208 |
| sonnet-4.5 | 96.07 | 95.67 | 320,397 | 1.09 | 273 |
| **El Agente Q** | 88.25 | — | **1,649,616** | **4.67** | **1,827** |

> ✅ **相比前作，GRAFICO 实现了：**
- **成本降低 96%**（$4.67 → $0.17）
- **时间提速 ≥6x**（1827s → ~200s）
- **Token消耗减少 ~14x**

---

### 与基线方法的对比结果

#### 对比 El Agente Q（多智能体）
- **效率飞跃**：单智能体消除了多智能体间的通信开销，大幅减少LLM API调用次数。
- **更强并行性**：充分利用GPU加速的PySCF流程，无需协调延迟。
- **更高可靠性**：结构化执行图防止非法状态转移，提升鲁棒性。

#### 对比轻量级LLM代理（Bare LLM）
| 任务 | Bare LLM 结果 | GRAFICO 正确结果 |
|------|----------------|------------------|
| **Inorganic L1** | ClF₃ 几何错误（平面三角形而非T型）<br>遗漏虚频检查 | ✅ 正确T型结构<br>✅ 无虚频 |
| **pKa L2** | pKa ≈ -5.0（错误协议）<br>未校准质子溶剂化能 | ✅ pKa ∈ [-2.7, 1.5] |
| **资源消耗** | ~650k tokens / 40分钟 | ~25k tokens / 3分钟 |

> ❌ 轻量级代理虽具备“自举”潜力，但缺乏结构约束，极易产生科学错误。

---

### 消融实验与深入分析（Supplementary）

#### Pass@k 与 Pass^k 分析（k=3）
- **pass@3 = 0.99**：3次尝试内至少成功一次的概率极高
- **pass^3 = 0.54**：3次全部成功的概率仍可观，表明系统具有较强鲁棒性

#### 推理行为差异
- **GPT系列**：倾向于批量调用工具，再汇总推理 → 更高效
- **Claude系列**：偏好串行调用 + 增量推理 → 成本更高，尤其`sonnet-3.7`

#### 缓存影响
- **Carryover Tokens**：gpt-5仅53%，而`sonnet-3.7`高达86%，说明其更难利用缓存机制。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **从Prompt工程转向Context工程**  
   科学智能体的可扩展性瓶颈不在提示词本身，而在如何**结构化地管理执行上下文**。通过将状态显式类型化并外化至KG，可显著提升可靠性和效率。

2. **单智能体优于多智能体**  
   在达到一定能力阈值后，增加智能体数量不仅不会提升性能，反而因协调失败导致**性能下降39–70%**。单智能体配合结构化执行图是更优路径。

3. **类型安全是信任基础**  
   通过`pydantic`强制schema验证、工具沙箱运行、实时分子可视化，使用户能共同观察内部状态，增强对输出的信任。

4. **知识图谱不仅是记忆，更是推理基底**  
   KG不仅用于持久化，还支持关系感知查询（如“找出所有含N625节点的MOF”），成为主动推理的基础设施。

---

### 方法的局限性
1. **当前为单会话、单运行时模型**  
   尚未解决长期运行、跨会话同步、分布式协作等问题。
2. **依赖特定工具生态**  
   虽然抽象层良好，但仍需为每个新工具开发适配器（如PORMAKE、MACE）。
3. **路由控制器仍为黑盒**  
   当前路由决策由LLM驱动，缺乏形式化保证，可能引入非预期路径。

---

### 未来工作方向（Roadmap in Figure 5）
1. **异步与资源感知执行环境**  
   引入容器化、隔离运行时，支持高风险工具的安全组合。
2. **语义边界演化（Semantic Boundary Evolution）**  
   实现工具与本体的自动演进，类似单元测试与代码协同进化。
3. **长视野与分布式智能体网络**  
   构建AI科学家团队，支持跨任务、跨时间的知识积累与协作。
4. **自我演化智能体**  
   智能体不仅能执行任务，还能改进自身架构与工具集。

---

> 🔮 **最终愿景**：将GRAFICO部署为云服务，作为 **EL AGENTE 家族** 的一部分，推动全球科学民主化。

</details>

---

### 10. [Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies](https://arxiv.org/abs/2602.18291)

**Authors**: Zhuoran Li, Hai Zhong, Xun Wang, Qingxin Xia, Lihua Zhang, Longbo Huang  
**Category**: cs.AI  
**Published**: 2026-02-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.18291v1  

#### Abstract
Online Multi-Agent Reinforcement Learning (MARL) is a prominent framework for efficient agent coordination. Crucially, enhancing policy expressiveness is pivotal for achieving superior performance. Diffusion-based generative models are well-positioned to meet this demand, having demonstrated remarka...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Diffusing to Coordinate: Efficient Online Multi-Agent Diffusion Policies

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Multi-Agent Reinforcement Learning (MARL)** 方法大多采用单峰策略分布（如高斯策略），难以建模复杂的**多模态协调行为**，限制了在复杂非平稳环境中的探索与协作能力。尽管基于扩散模型的策略（Diffusion Policy）在离线强化学习中展现出强大的表达能力，但由于其**不可计算的似然性**（intractable likelihoods），导致无法进行基于熵的探索和正则化，阻碍了其在**在线 MARL** 中的应用。

此外，在 **CTDE**（Centralized Training with Decentralized Execution）框架下，如何有效协调多个扩散策略、避免训练不稳定，也是一个未被充分解决的挑战。

### 提出的新方法与新思路
本文提出了 **OMAD**（Online off-policy Multi-Agent Diffusion policies），是首个将扩散策略成功应用于**在线、off-policy MARL** 的框架。其核心创新包括：

- **可计算的最大熵目标函数**：提出一个基于**缩放联合熵证据下界**（scaled joint entropy ELBO）的松弛策略目标，绕过扩散模型不可计算似然性的障碍，实现有效的熵正则化探索。
- **集中式分布值函数引导**：在 CTDE 框架下，引入一个**联合分布值函数**（Joint Distributional Critic），通过建模回报的完整分布而非期望值，提供更丰富、鲁棒的价值信号，指导多个扩散策略的同步更新。
- **同步策略优化机制**：所有智能体的扩散策略在统一的目标下进行联合优化，确保协调一致性，缓解非平稳性问题。
- **自动温度调优**：通过约束联合 ELBO 的下限，动态调节熵系数 $\alpha$，实现自适应的探索-利用权衡。

### 相比现有方法的优势
- **更强的策略表达能力**：相比传统高斯策略，扩散策略能捕捉多模态动作分布，支持更灵活、多样化的协作策略。
- **高效的探索机制**：通过可计算的 ELBO 实现熵驱动探索，避免陷入局部最优。
- **稳定协调训练**：集中式分布值函数与同步更新机制显著提升训练稳定性与收敛速度。
- **样本效率高**：实验表明 OMAD 在样本效率上达到 SOTA，相比基线提升 **2.5× 到 5×**。

---

## 2. 核心实验方法和设置

### 使用的数据集
在两个标准的连续控制多智能体基准上进行评估：

- **MPE**（Multi-Agent Particle Environments）：
  - Cooperative Navigation（3/4 agents）
  - Physical Deception（2 agents）
- **MAMuJoCo**（Multi-Agent MuJoCo）：
  - Ant（2×4, 2×4d, 4×2）
  - HalfCheetah（2×3, 6×1）
  - Walker2d（2×3）
  - Swimmer（2×1）

这些任务要求智能体在高维动作空间中协同完成导航、避障、奔跑等复杂控制任务。

### 实验设置和评估指标
- **训练步数**：多数任务训练至 $3 \times 10^6$ 步，部分基线延长至 $1 \times 10^7$ 步以确保收敛。
- **评估方式**：每固定步数评估一次，报告 **平均episode return**（平均累积奖励）。
- **统计可靠性**：所有实验运行 **5 个随机种子**，报告均值与标准差。
- **关键超参数**：
  - 扩散步数（denoising steps）：8
  - 分布值函数原子数（atoms）：100
  - 温度自动调优目标熵：$H_{\text{target}} = 4 \cdot \text{dim}(A)$

### 基线方法对比
- **主流在线 MARL 方法**：
  - HATD3
  - HASAC
- **扩散策略扩展方法**（作为对比）：
  - MADPMD（Multi-Agent Diffusion Policy with MADDPG-style training）
  - MASDAC（Multi-Agent SDAC）

这些基线用于验证仅引入扩散策略而不设计专用协调机制的不足。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 任务 | OMAD 最终性能（平均回报） |
|------|----------------------------|
| Cooperative Navigation (N=3) | **-23.9 ± 1.1** |
| Ant 2×4 | **7517.0 ± 279.2** |
| HalfCheetah 2×3 | **14368.5 ± 1166.0** |
| Swimmer 2×1 | **162.0 ± 22.5** |

> ✅ OMAD 在 **10 个不同任务** 上均达到新的 SOTA 性能。

### 与基线方法的对比结果
- **性能全面领先**：在所有 MPE 和 MAMuJoCo 任务中，OMAD 不仅最终性能更高，且**收敛速度更快**，达到基线峰值性能所需训练步数减少 **最多达 5×**。
- **样本效率提升显著**：相比最强基线（如 HASAC、HATD3），OMAD 实现 **2.5× 至 5× 的样本效率增益**。
- **扩散策略直接迁移效果有限**：MADPMD 和 MASDAC 虽受益于扩散建模，但因缺乏有效协调机制，表现不如 OMAD，甚至出现收敛慢或性能下降现象。

### 消融实验结果
#### （1）分布值函数超参数影响
- **Vmax 设置过低**（如 200）会截断回报分布，严重损害性能；**Vmax ≥ 1000** 后性能趋于稳定。
- **原子数过少**（< 100）无法捕捉分布细节，过多则收益递减。最终选择 **100 个原子**。

#### （2）去噪步数的影响
- 去噪步数为 **8 时性能已饱和**（~6000 回报），进一步增加至 16 步带来微弱提升但显著增加训练时间（从 22h → 26h）。
- 推理延迟也随步数线性增长，**8 步为效率与性能的最佳平衡点**。

#### （3）熵系数自动调优 vs 固定值
- 固定高熵系数（$\alpha=0.1$）导致过度随机，训练不稳定；
- 手动调优小值（如 0.001–0.01）虽可达高性能，但需大量搜索；
- **自动调优机制** 动态调整 $\alpha$，**匹配最佳固定值性能**（~6000），同时消除超参敏感性，提升鲁棒性。

#### （4）状态探索可视化
在 Ant 2×4 任务前 250k 步中：
- OMAD 状态覆盖率：**68.3%**（934 bins）
- HASAC：55.0%（753 bins）
- HATD3：48.4%（662 bins）

👉 OMAD 显著拓展了探索范围，尤其覆盖了其他方法未触及的状态区域（图中橙色区域），证明其更强的探索能力。

---

## 4. 关键结论和发现

### 主要发现
- **扩散策略可用于在线 MARL**：通过设计可计算的 ELBO 替代不可计算的熵，首次实现了扩散策略在在线、off-policy 多智能体场景下的高效训练。
- **集中式分布值函数至关重要**：建模回报分布而非期望值，能更好解耦智能体间的随机性干扰，提供更稳定的监督信号。
- **同步更新优于独立优化**：共享全局价值目标下的联合策略优化，显著提升协调效率与训练稳定性。
- **表达力 + 协调机制 = 高效探索**：扩散策略的多模态表达能力必须与有效的价值引导机制结合，才能真正释放其潜力。

### 方法的局限性
- **计算开销较高**：相比传统策略，扩散模型需要多次去噪迭代（即使仅 8 步），推理延迟仍高于确定性策略。
- **依赖集中式训练架构**：虽然执行是去中心化的，但训练阶段需要集中式 critic，对通信带宽有一定要求。
- **理论近似误差未知**：ELBO 是真实熵的下界，其近似偏差在训练过程中的影响尚难精确量化。

### 未来工作方向
- **加速推理**：研究更高效的扩散采样器（如 DDIM、DPM-Solver）以降低延迟。
- **离散动作空间扩展**：探索基于离散扩散模型（discrete diffusion）的多智能体策略。
- **通信受限场景适配**：设计轻量级 critic 或分布式估计机制，适应边缘设备部署。
- **理论分析深化**：研究 ELBO 近似的收敛性保证与偏差边界。

---

> 📌 **总结一句话**：  
> **OMAD 成功将扩散策略引入在线多智能体强化学习，通过可计算的熵下界与集中式分布值函数，实现了前所未有的样本效率与协调能力，在 MPE 和 MAMuJoCo 上全面超越现有方法。**

</details>

---

### 11. [Joint Training on AMD and NVIDIA GPUs](https://arxiv.org/abs/2602.18007)

**Authors**: Jon Hu, Thomas Jia, Jing Zhu, Zhendong Yu  
**Category**: cs.DC  
**Published**: 2026-02-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.18007v1  

#### Abstract
As large language models continue to scale, training demands on compute and system capacity grow rapidly, making single-vendor homogeneous clusters insufficient. This paper presents a technical solution for heterogeneous mixed training in AMD-NVIDIA environments. We first adopt a compatibility-orien...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Joint Training on AMD and NVIDIA GPUs》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着大语言模型（LLM）参数规模突破万亿级别，单一厂商的同构计算集群已难以满足日益增长的算力需求。现代数据中心普遍面临硬件异构性挑战，尤其是 **AMD** 与 **NVIDIA** 加速器共存的混合环境。然而，跨厂商 GPU 之间缺乏高效的直接通信机制，导致训练效率低下。

本文旨在解决 **AMD-NVIDIA 异构 GPU 集群中的联合训练难题**，特别是如何在不牺牲性能的前提下实现跨架构的高效通信。

---

### 🚀 提出的新方法与创新思路

论文提出了两种异构训练方案，并重点优化了高性能路径：

#### （1）**CPU-Forwarding Communication（兼容性优先方案）**
- 利用 CPU 作为中转代理进行跨厂商通信（如使用 Gloo 库）
- 在 **Pipeline Parallel (PP)** 组间采用 Gloo 支持异构连接
- 在 **Data Parallel (DP)** 和 **Tensor Parallel (TP)** 组内保留厂商专用库（NCCL / RCCL），维持高带宽
- 引入 **Multi-NIC Parallel Data Transfer (MPDT)**：为每个 GPU 分配独立 NIC，提升 PP 跨节点传输并行度

> ⚠️ 局限：频繁的 Host-Device 内存拷贝（H2D/D2H）成为性能瓶颈。

#### （2）**Device-Direct Communication（高性能核心方案）**
- **核心创新**：提出一种 **CPU-offloading P2P 机制**，实现跨厂商 GPU 之间的 **设备直连通信**
- 利用 **GPUDirect RDMA (GDR)** 技术绕过主机内存，避免中间拷贝开销
- 控制面由 CPU 处理（资源注册、连接管理等），数据面完全在设备侧完成
- 构建多适配器架构（Device / Net-Plugin / CCL Adaptors）抽象硬件差异
- 通过 PyTorch backend 插件集成，对上层框架（Megatron、DeepSpeed）透明

> ✅ 优势：消除 PCIe 瓶颈，显著提升通信效率，接近同构系统性能。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法（如 Gloo） | 本文 Device-Direct 方法 |
|------|------------------------|----------------------------|
| 通信路径 | Host 中转 → H2D/D2H 开销大 | Device-direct → 零主机内存拷贝 |
| 性能 | 显著低于同构系统 | 达到 NVIDIA 同构系统的 **98% 吞吐量** |
| 兼容性 | 支持异构但低效 | 支持异构且高效 |
| 可扩展性 | 受限于 CPU 带宽 | 利用 RDMA + 多 NIC 并行，可扩展性强 |
| 上层透明性 | 需修改训练逻辑 | 通过插件无缝接入，无需改动高层代码 |

---

## 2. 核心实验方法和设置

### 📊 使用的模型（Workloads）
- **LLaMA-8B**
- **Qwen2-7B**

> 注：未使用传统“数据集”概念，而是以完整 LLM 预训练任务为 workload。

---

### 🖥 实验平台配置（Testbed）

| 节点类型 | GPU 数量 | GPU 型号 | 互联技术 | 带宽 |
|---------|----------|----------|-----------|-------|
| NVIDIA Node | 8× | H200 | NVLink | 900 GB/s（双向） |
| AMD Node | 8× | Instinct MI325X | Infinity Fabric | 128 GB/s（双向） |
| 网络设备 | 每节点 8× | BlueField-3 DPU | RDMA over Converged Ethernet (RoCE) | 100 GB/s per DPU |

> 异构设置 = 连接 NVIDIA 和 AMD 节点形成双节点系统

---

### ⚙️ 并行策略（Parallelization Configuration）

| 并行维度 | 设置说明 |
|----------|----------|
| Tensor Parallelism (TP) | =1（即无 TP） |
| Pipeline Parallelism (PP) | =2（两阶段流水线） |
| Data Parallelism (DP) | Homogeneous: DP=4；Heterogeneous: DP=8 |

> 流水线划分根据性能差异动态调整：
- LLaMA-8B：AMD 侧 15 层，NVIDIA 侧 17 层
- Qwen2-7B：AMD 侧 12 层，NVIDIA 侧 16 层  
→ 实现负载均衡，避免 pipeline bubble

---

### 📈 评估指标
- **Average Training Throughput**（样本/秒 或 tokens/sec）
- **Training Stability**：迭代过程中的吞吐波动情况
- **Loss Convergence Behavior**：损失函数收敛曲线是否一致
- 对比基线包括：
  - NVIDIA-Homo（纯 NVIDIA 单节点）
  - AMD-Homo（纯 AMD 单节点）
  - Global Gloo-Hetero（原始 Megatron+Gloo 异构方案）
  - DCBS-Hetero（差异化后端选择）
  - DCBS&MPDT-Hetero（加多 NIC 优化）

---

## 3. 主要实验结果和性能指标

### 📉 吞吐性能对比（见 Figure 3）

| 配置 | LLaMA-8B 吞吐 | Qwen2-7B 吞吐 |
|------|----------------|----------------|
| NVIDIA-Homo（最强基线） | 557.4 | 526.4 |
| AMD-Homo | 500.0 | 514.5 |
| **Proposed Approach (Device-Direct)** | **549.7** | **497.0** |
| → 占 NVIDIA-Homo 比例 | **98.2%** | **94.4%** |
| → 超越 AMD-Homo | +0.9% | +2.3% |

> ✅ 结论：异构训练性能几乎媲美高端 NVIDIA 同构系统！

---

### 🔁 与其它异构方案对比

| 方法 | LLaMA-8B 吞吐 | 相对 Device-Direct 差距 |
|------|----------------|--------------------------|
| Global Gloo-Hetero | 11.1 | ↓ 98% |
| DCBS-Hetero | 236.5 | ↓ 57% |
| DCBS&MPDT-Hetero | 160.8 | ↓ 71% |
| **Device-Direct (Ours)** | **549.7** | —— |

> ❗ 明确显示：仅靠软件调度优化（DCBS/MPDT）无法弥补 Host-Copy 的根本瓶颈。

---

### 📈 稳定性分析（Figure 4）
- 在连续 500 个训练 iteration 中：
  - 异构训练吞吐保持稳定，无明显抖动或下降趋势
  - 波动幅度与同构系统相当
- 表明该方法具备工业级稳定性

---

### ✅ 正确性验证（Figure 5）
- Loss 曲线显示：
  - AMD-NVIDIA 异构训练的 loss 收敛轨迹与 NVIDIA-Homo 和 AMD-Homo 完全重合
  - 无发散、震荡或其他数值异常
- 证明：**Device-Direct 方法保证了训练动态和数值精度的一致性**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **异构训练可以接近同构性能**  
   → 借助 Device-Direct Communication，AMD-NVIDIA 混合集群可达 NVIDIA 同构系统 **98% 的吞吐**

2. **Host-memory staging 是主要瓶颈**  
   → 所有基于 CPU 转发的方法（Gloo、DCBS 等）因 H2D/D2H 开销严重受限

3. **P2P 是异构友好的切入点**  
   → 将异构性限制在 PP 组（仅需 P2P 通信），而 DP/TP 保持同构，可最大化性能与稳定性

4. **合理的模型切分至关重要**  
   → 不均衡 layer partitioning（将更多层分配给更强的 NVIDIA GPU）是实现高效的关键

5. **工程复杂性远高于接口兼容性**  
   → 尽管 NCCL 与 RCCL 接口相似，实际联调中仍出现大量 hang 问题，需深度系统级调试

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **仅支持 PP 层级异构** | 当前未将异构扩展至 DP/TP 组，因其依赖复杂的 AllReduce/AllGather，尚无跨厂商高效集体通信实现 |
| **依赖特定硬件支持** | 需要支持 GDR 和 RDMA 的 NIC（如 BlueField DPU），通用性受限 |
| **拓扑感知要求高** | 需精确掌握全局设备拓扑和元数据，部署复杂度上升 |
| **尚未测试更大规模集群** | 实验仅在双节点进行，大规模扩展性有待验证 |

---

### 🔮 未来工作方向

1. **拓展异构支持至 DP/TP 维度**  
   → 研究跨厂商 AllReduce 的高效实现，推动全栈异构训练

2. **支持更多厂商组合**  
   → 如 Intel Gaudi、Huawei Ascend 等加入混合训练生态

3. **自动化模型分区与调度**  
   → 结合 profiling 自适应地决定 layer 分配策略，提升易用性

4. **构建统一的异构通信抽象层**  
   → 类似 UCX 的跨厂商通信运行时，降低开发门槛

5. **探索 fault tolerance 与弹性训练能力**  
   → 在异构环境下实现容错恢复与动态扩缩容

---

## ✅ 总结一句话

> 本论文首次实现了 **高性能、稳定、正确的 AMD-NVIDIA 异构 LLM 训练**，提出的 **Device-Direct Communication** 方法通过 **CPU-offloading P2P + GDR** 技术，使混合集群达到 **98% 的 NVIDIA 同构系统吞吐**，为未来多元算力融合提供了可行路径。

</details>

---

### 12. [Joint Parameter and State-Space Bayesian Optimization: Using Process Expertise to Accelerate Manufacturing Optimization](https://arxiv.org/abs/2602.17679)

**Authors**: Saksham Kiroriwal, Julius Pfrommer, J\"urgen Beyerer  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.17679v1  

#### Abstract
Bayesian optimization (BO) is a powerful method for optimizing black-box manufacturing processes, but its performance is often limited when dealing with high-dimensional multi-stage systems, where we can observe intermediate outputs. Standard BO models the process as a black box and ignores the inte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Joint Parameter and State-Space Bayesian Optimization: Using Process Expertise to Accelerate Manufacturing Optimization*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Bayesian Optimization (BO)** 在处理高维、多阶段制造过程时存在显著局限：
- 将整个系统视为黑箱（black-box），忽略了中间输出（intermediate outputs）和已知的流程结构（如因果依赖关系）。
- 现有方法如 **Gaussian Process Networks (GPNs)** 和 **Partially Observable GPN (POGPN)** 虽能建模结构化流程，但无法有效处理高维状态空间时间序列数据（如传感器采集的温度、pH 随时间变化的数据）。

因此，在复杂工业场景中（如生物乙醇生产），如何利用**高维中间观测数据**并结合**专家知识**来加速优化，是一个关键挑战。

---

### 🚀 提出的新方法：POGPN-JPSS
本文提出 **POGPN-JPSS** —— 一种融合了以下两个关键技术的新型框架：

1. **Partially Observable Gaussian Process Network (POGPN)**  
   - 将制造过程建模为一个 **Directed Acyclic Graph (DAG)**，每个节点代表一个子过程。
   - 使用 **doubly stochastic variational inference (DSVI)** 进行联合推理，显式建模潜在输出 $ f^{(v)} $ 而非直接使用带噪声的观测 $ y^{(v)} $，提升鲁棒性。

2. **Joint Parameter and State-Space (JPSS) Modeling**  
   - 引入**工艺专家知识**，从高维状态空间数据 $ S^{(v)} $（如时间序列）中提取低维**潜特征** $ h^{(v)} $。
   - 例如：在生物反应器中，仅保留最终时刻的生物质浓度和乙醇浓度作为关键特征。

> 🔗 **创新整合**：首次将 **POGPN** 与 **JPSS** 结合，实现对“参数 + 状态空间”的联合建模，使 BO 可以充分利用结构先验和高维传感数据。

---

### ⚖️ 相比现有方法的优势
| 方法 | 局限性 | POGPN-JPSS 的优势 |
|------|--------|------------------|
| **Standard BO / STGP** | 黑箱建模，忽略结构和中间数据 | 显式建模流程结构，利用中间信息 |
| **GPN / POGPN** | 无法处理高维状态空间数据 | 通过 JPSS 提取低维特征，兼容高维输入 |
| **High-Dimensional GP [15]** | 仅处理高维输入，不利用中间状态 | 利用状态演化信息，增强泛化能力 |

> ✅ **核心优势**：**更快收敛 + 更高可靠性 + 更少资源消耗**

---

## 2. 核心实验方法和设置

### 📊 数据集
- **Multi-stage Bioethanol Production Process Simulation**（多阶段生物乙醇生产仿真）
  - 来源：基于 ODE 动力学模型模拟真实发酵过程（参考文献 [9,14,17]）
  - 包含三个子过程：
    - P(1): Flask（摇瓶活化）
    - P(2): Seed Fermenter（种子罐培养）
    - P(3): Production Fermenter（生产罐发酵）
  - 输入维度：$ D_x = 26 $（包括进料速率、温控曲线、pH 曲线、各阶段持续时间等）
  - 输出：高维状态空间时间序列 $ S^{(v)} $（如温度、体积、底物浓度随时间变化）

> 💡 每次仿真耗时极长（对应现实世界数天），故快速优化至关重要。

---

### 🧪 实验设置
- **优化目标**：最大化 **Space-Time Yield (STY)**，即单位时间单位体积产乙醇量。
- **STY 经济阈值**：≥ 0.5 g/(L·hr)
- **优化预算**：100 次迭代
- **初始数据**：50 组随机采样点
- **重复次数**：5 次独立运行，报告均值与标准差
- **预处理**：所有观测值进行 log transform 和标准化
- **实现工具**：**BoTorch** 框架，Matérn-5/2 kernel，Gaussian likelihood
- **诱导点策略**：
  - 目标节点使用 **Greedy Improvement Reduction (GIR)**
  - 其他节点使用 **Greedy Variance Reduction (GVR)**

---

### 📈 评估指标
- 主要指标：
  - **Best-observed STY** 随迭代的变化趋势（平均值 ± 标准差）
  - 达到 **STY ≥ 0.5** 所需迭代次数
- 对比方式：Boxplot 与收敛曲线

---

### 🔁 基线方法对比
| 方法 | 类型 | 是否使用结构 | 是否处理高维状态 |
|------|------|---------------|--------------------|
| **STGP** [15] | 单任务高维 GP | ❌ | ✅（仅输入） |
| **SVGP-PLL** [18,24] | 稀疏变分 GP | ❌ | ✅ |
| **GPN** [4] | 高斯过程网络 | ✅ | ❌（仅接受标量中间输出） |
| **POGPN-ELBO / POGPN-PLL** [21] | 概率图模型 + 结构 | ✅ | ❌（原生不支持高维状态） |
| ✅ **POGPN-JPSS (ours)** | **本文方法** | ✅ | ✅（通过 JPSS 提取特征） |

> 注：POGPN-JPSS 分别测试了两种训练目标函数：**ELBO** 和 **Predictive Log-Likelihood (PLL)**

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Figure 4）
| 方法 | 达到 STY ≥ 0.5 所需迭代数 | 最终平均 STY | 收敛稳定性（方差） |
|------|----------------------------|--------------|---------------------|
| **STGP** | ~95–100 | < 0.5 | 差（波动大） |
| **SVGP-PLL** | ~90 | ~0.48 | 中等 |
| **GPN** | ~85 | ~0.47 | 中等 |
| **POGPN-ELBO** | ~70 | ~0.52 | 较好 |
| ✅ **POGPN-PLL (JPSS)** | **~50** | **> 0.55** | **最优（最小方差）** |

> 🔥 **核心结果**：**POGPN-PLL + JPSS 在约 50 次迭代内达到经济可行阈值，速度是其他方法的两倍以上！**

---

### 🆚 与基线方法对比
- **收敛速度**：
  - POGPN-PLL (JPSS) 在第 50 次迭代就稳定超过 0.5 STY。
  - 其他最佳方法（如 POGPN-ELBO）需接近 100 次迭代才能达到相同水平。
- **最终性能**：
  - 平均 STY 提升 > 10%，且结果更可靠（标准差更低）。
- **资源节省估算**：
  - 每次仿真模拟现实中的数日至数周。
  - 加速优化可节省 **300–350 天**的实际生产时间，大幅降低机会成本、材料和能源消耗。

---

### 🔍 消融实验分析（隐含于设计中）
虽然未明确列出消融实验表格，但从方法构建逻辑可推断：

| 变体 | 性能预期下降原因 |
|------|------------------|
| 移除 JPSS（直接丢弃状态数据） | 丢失动态演化信息，导致决策依据不足 |
| 使用 GPN 替代 POGPN | 忽略测量噪声与真实输出差异，传播误差 |
| 使用 ELBO 而非 PLL | PLL 更关注预测准确性，更适合 BO 场景 |

> ✅ 实验表明：**PLL 目标优于 ELBO**，说明面向预测性能的目标函数更适合 BO。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **专家知识驱动的特征提取（JPSS）至关重要**  
   - 无需端到端学习高维数据，而是由领域专家指导提取物理意义明确的关键特征，显著提升样本效率。

2. **结构化建模（POGPN）+ 状态感知（JPSS）带来协同增益**  
   - 同时利用流程结构（DAG）和中间状态信息，使得 BO 能“理解”过程机理，做出更优决策。

3. **POGPN-PLL 是当前最优配置**  
   - 在复杂制造系统中，采用 **Predictive Log-Likelihood** 作为训练目标，比传统 ELBO 更有利于后续优化。

4. **加速优化 = 直接经济效益**  
   - 快速达到性能阈值意味着巨大的时间、材料和能源节约，尤其适用于 pilot-scale 或 industrial-scale 工艺成熟（process maturation）。

---

### ⚠️ 方法的局限性
1. **依赖专家知识进行特征工程**  
   - 若缺乏可靠的领域专家，则难以提取有效低维特征，限制通用性。
2. **假设 DAG 结构已知**  
   - 在实际系统中，流程结构可能部分未知或不确定，需额外探索。
3. **计算开销较高**  
   - POGPN 使用 DSVI，训练复杂度高于标准 GP，对超参调优敏感。

---

### 🔮 未来工作方向
1. **自动化特征提取机制**  
   - 探索自编码器、注意力机制等 DL 方法自动识别重要状态特征，减少人工干预。
2. **因果结构发现（Causal Discovery）集成**  
   - 将 POGPN-JPSS 与因果图学习结合，用于结构未知的系统。
3. **扩展至更多工业场景**  
   - 应用于制药、半导体制造、电池生产等同样具有多阶段、高维传感特性的行业。
4. **在线学习与闭环控制结合**  
   - 将 BO 与实时控制系统集成，实现动态调整与持续优化。

---

## ✅ 总结一句话
> **POGPN-JPSS 通过融合工艺专家知识与结构化概率建模，实现了对高维多阶段制造过程的高效贝叶斯优化，在生物乙醇生产案例中将达标速度提升一倍以上，展现出强大的工业应用潜力。**

</details>

---

### 13. [Continual-NExT: A Unified Comprehension And Generation Continual Learning Framework](https://arxiv.org/abs/2602.18055)

**Authors**: Jingyang Qiao, Zhizhong Zhang, Xin Tan, Jingyu Gong, Yanyun Qu, Yuan Xie  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.18055v1  

#### Abstract
Dual-to-Dual MLLMs refer to Multimodal Large Language Models, which can enable unified multimodal comprehension and generation through text and image modalities. Although exhibiting strong instantaneous learning and generalization capabilities, Dual-to-Dual MLLMs still remain deficient in lifelong e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Continual-NExT: A Unified Comprehension And Generation Continual Learning Framework**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对 **Dual-to-Dual MLLMs**（即支持文本/图像输入并生成文本/图像输出的多模态大语言模型）在持续学习（Continual Learning）场景下的以下挑战：
- **灾难性遗忘（Catastrophic Forgetting）**：学习新任务时严重丢失旧知识。
- **幻觉（Hallucination）** 和 **指令不遵循（Instruction Unfollowing）**：在生成任务中产生错误或偏离指令的内容。
- 缺乏统一的、支持多模态理解与生成的持续学习框架。

现有研究大多集中在 **Dual-to-Text MLLMs** 上，无法有效应对多模态输入输出带来的复杂性。

---

### **提出的新方法与新思路**
论文提出了两个核心贡献：

#### ✅ **1. Continual-NExT 框架**
- 首个面向 **Dual-to-Dual MLLMs** 的统一持续学习框架。
- 支持六类异构任务：
  - **Visual Question Answering (VQAv2)**
  - **Image Classification (ImageNet)**
  - **Image Generation (Flickr30k)**
  - **OCR Token Recognition (OCRVQA)**
  - **Visual Grounding (RefCOCO)**
  - **Image Editing (HQEdit)**
- 引入三种诊断性评估指标以深入分析遗忘行为：
  - **Avg.HAL**（平均幻觉率）
  - **Avg.IUF**（平均指令不遵循率）
  - **Avg.OTH**（其他错误率）

#### ✅ **2. MAGE 方法（Mixture and Aggregation of General LoRA and Expert LoRA）**
- 将 LoRA 分为两类：
  - **General LoRA**：负责输入模态的理解（如文本/图像编码）。
  - **Expert LoRA**：负责输出模态的推理与生成（如文本/图像解码）。
- 在训练时仅更新与当前任务相关的 LoRA 模块，冻结无关部分，从而减少干扰。
- 提出 **Parameter-wise EMA (PEMA)** 更新策略：
  - 替代传统的 DEMA 方法，基于 Fisher 信息矩阵计算参数级动态权重。
  - 更细粒度、更低内存开销，避免存储历史梯度。

---

### **相比现有方法的优势**
| 特性 | MAGE | 其他方法（如 EWC, CIA, MoELoRA） |
|------|------|-------------------------------|
| 多模态支持 | ✅ 支持图文双向输入输出 | ❌ 多为文本生成任务 |
| 参数效率 | ✅ 固定总秩，无模型扩展 | ⚠️ 部分需额外参数（如 CIA 的 instruction grouping） |
| 抗遗忘能力 | ✅ 显著优于所有 baseline | ⚠️ 存在遗忘或记忆负担重 |
| 内存消耗 | ✅ 无需缓存历史梯度/参数 | ❌ 如 DEMA 需高内存 |
| 跨模态迁移 | ✅ 观察到显著的知识转移现象 | ⚠️ 未明确建模 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 任务 | 数据集 | 主要特点 |
|------|--------|----------|
| VQAv2 | VQAv2 | 图像问答，开放答案 |
| Image Classification | ImageNet-1K | 千类图像分类 |
| Image Generation | Flickr30k | 文本到图像生成 |
| OCR Recognition | OCRVQA | 图中读取文字并回答问题 |
| Visual Grounding | RefCOCO | 定位描述中的物体 |
| Image Editing | HQEdit | 指令驱动的图像编辑 |

> 所有任务按顺序增量训练，模拟真实世界动态演化场景。

---

### **实验设置**
- **骨干模型**：SEED-X（基于 LLaMA2-13B + ViT）
- **可训练参数**：仅训练连接层（connection layers）和 LoRA 模块，其余冻结。
- **LoRA 设置**：
  - 总秩固定为 32（与其他 baseline 对齐）
  - MAGE 使用 4 个专家（General Text/Image LoRA + Expert Text/Image LoRA），每个秩为 8
- **训练顺序**：默认为 `VQAv2 → ImageNet → Flickr30k → OCRVQA → RefCOCO → HQEdit`

---

### **评估指标**
| 指标 | 含义 |
|------|------|
| **Avg.ACC** | 所有任务最终准确率的平均值（综合性能） |
| **Forgetting (↓)** | 旧任务性能下降程度（稳定性） |
| **New.ACC (↑)** | 新任务学习能力（可塑性） |
| **Avg.HAL (↓)** | 平均幻觉率 |
| **Avg.IUF (↓)** | 平均指令不遵循率 |
| **Avg.OTH (↓)** | 其他错误率 |

---

### **基线方法对比**
| 方法 | 类型 | 特点 |
|------|------|------|
| **Zero-Shot** | 零样本推理 | 不进行微调 |
| **LoRA Fine-Tune** | 标准 LoRA 微调 | 基线，易遗忘 |
| **MoELoRA** | 多专家混合 | 利用 MoE 结构缓解遗忘 |
| **EWC** | 正则化方法 | 基于 Fisher 保护重要参数 |
| **LAE / PGP / CIA / RegLoRA** | EMA 或投影方法 | 动态更新或约束参数变化 |

> 所有方法保持相同可训练参数量，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1 & Table 6）**

| 方法 | Avg.ACC ↑ | Forgetting ↓ | New.ACC ↑ |
|------|-----------|--------------|------------|
| Zero-Shot | 28.81 | — | — |
| LoRA Fine-Tune | 42.23 ± 0.83 | 20.36 ± 0.31 | 59.19 ± 0.73 |
| MoELoRA | 43.90 ± 0.47 | 19.08 ± 0.22 | 59.80 ± 0.50 |
| EWC | 47.00 ± 0.85 | 15.85 ± 0.19 | 60.21 ± 0.75 |
| CIA | 47.99 ± 0.18 | 14.89 ± 0.16 | 60.40 ± 0.55 |
| **MAGE (Ours)** | **49.58 ± 0.58** | **12.26 ± 0.13** | 59.79 ± 0.63 |
| Upper-Bound | 53.31 | — | — |

> ✅ MAGE 在 **Avg.ACC** 上超越最强基线 **CIA** 达 **+1.59**，且 **Forgetting 降低至 12.26**，表现最佳。

---

### **与基线方法的对比结果**
- **抗遗忘能力最强**：MAGE 的 Forgetting 最低（12.26），远低于 LoRA（20.36）、MoELoRA（19.08）。
- **综合性能最优**：Avg.ACC 持续领先，在后期阶段拉开差距。
- **内存更优**：无需保存中间变量（如 CIA），适合实际部署。
- **鲁棒性强**：在不同训练顺序下（Reverse / Alphabet），Forgetting 波动极小（12.17–12.35），说明方法稳定。

---

### **消融实验结果（Table 4）**
逐步加入模块验证有效性：

| 方法 | Avg.ACC | Forgetting |
|------|---------|------------|
| MoELoRA (baseline) | 43.90 | 19.08 |
| + Equal-Weight Sum (EW) | 46.75 | 15.66 |
| + General/Expert LoRA Fusion (EW+GA) | 47.67 | 14.90 |
| + PEMA 更新策略 (**MAGE**) | **49.58** | **12.26** |

> 每个组件均有独立贡献，尤其是 **PEMA** 对降低遗忘至关重要。

---

## **4. 关键结论和发现**

### **主要发现**
1. **理解与生成并非本质冲突**：
   - 传统认为 comprehension 与 generation 目标矛盾，但实验证明其差异源于 **输入/输出模态对参数更新的不同需求**。
   - 输入侧需稳定表示（General LoRA），输出侧需灵活解码（Expert LoRA）。

2. **参数更新具有模态特异性**：
   - 浅层参数受输入模态影响大（如图文输入 vs 纯文本）。
   - 深层参数受输出模态主导（如文本生成 vs 图像生成）。
   - 不同模态的 LoRA 可隔离更新，互不干扰。

3. **存在显著跨模态知识迁移**：
   - 在 OCRVQA 上训练后，VQAv2 性能回升（从 20.10 → 50.26），表明 **旧知识可通过新任务“唤醒”**。
   - MAGE 更好保留此能力（43.56 → 57.29），证明其更强的记忆保持机制。

4. **PEMA 优于 DEMA**：
   - 在相同任务上，PEMA 实现更高 Accuracy 且计算/存储成本更低。
   - 参数级加权比层级更精细，更适合多任务环境。

---

### **方法的局限性**
1. **模态数量增长带来扩展瓶颈**：
   - 当前 LoRA 专家数随模态组合线性增加，可能导致计算爆炸。
2. **未处理未知模态增量**：
   - 所有模态均为预定义，如何引入全新模态（如音频）仍是开放问题。
3. **依赖高质量 LoRA 初始化**：
   - 若 General/Expert LoRA 划分不合理，可能影响性能。

---

### **未来工作方向**
- 探索 **更高效的适配器架构**（如层次化 LoRA、动态路由）。
- 扩展至 **Any-to-Any MLLMs**（支持任意模态间转换）。
- 研究 **跨模态持续学习理论机制**，解释知识“恢复”现象。
- 应用于 **安全敏感领域** 时需加强可控性和偏见抑制。

---

> 📌 **总结一句话**：  
> **MAGE 通过将模态理解与生成解耦，并结合参数级 EMA 更新，在 Dual-to-Dual MLLMs 的持续学习中实现了当前最优的稳定性与性能平衡。**

</details>

---

### 14. [RAT+: Train Dense, Infer Sparse -- Recurrence Augmented Attention for Dilated Inference](https://arxiv.org/abs/2602.18196)

**Authors**: Xiuying Wei, Caglar Gulcehre  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.18196v1  

#### Abstract
Structured dilated attention has an appealing inference-time efficiency knob: it reduces the FLOPs of the attention and the KV cache size by a factor of the dilation size D, while preserving long-range connectivity. However, we find a persistent failure mode of them -- sparsifying a pretrained atten...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《RAT+: Train Dense, Infer Sparse -- Recurrence Augmented Attention for Dilated Inference》总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现代语言模型中，标准 **attention** 机制在长序列上存在 **quadratic cost**（计算量和内存随序列长度平方增长）的问题。虽然已有多种稀疏化方法（如局部窗口 attention、top-k block attention）来提升推理效率，但**将预训练好的 dense attention 模型直接稀疏化为 dilated attention 模式时，性能会严重下降**。

此外，现有的稀疏架构（如 RAT、Mamba）通常需要从头训练，缺乏灵活性，无法在同一个模型上支持多种稀疏模式（如不同 dilation size 或混合模式）。

### **提出了什么新方法或新思路**

作者提出 **RAT+**，一种“**train dense, infer sparse**”的新架构，其核心思想是：

- 在预训练阶段保持 **dense attention** 结构；
- 引入 **full-sequence recurrence** 和 **active recurrence learning (ARL)**，使模型具备构建完整感受野的能力；
- 在推理时灵活切换到各种稀疏模式（如 dilated attention、local window、hybrid 层/头设计、top-k block attention），仅需极短的 **resolution adaptation**（约 1B tokens）即可适配。

#### 关键技术点：
- **Full-sequence recurrence**：采用固定长度的重叠递归块（最终简化为全序列递归），确保不同 dilation 设置下递归输出分布一致。
- **Active Recurrence Learning (ARL)**：通过联合训练（batch 中同时包含 dense 和 sparse 配置）强制模型学习足够长的有效递归能力（如 L\*=64）。

### **相比现有方法的优势**

| 方面 | RAT+ | 传统稀疏架构（如 RAT、Mamba） | Dense-to-sparse 直接转换 |
|------|------|-------------------------------|--------------------------|
| **训练成本** | 只需一次 dense 预训练 | 每种稀疏配置需单独训练 | 无需重新训练 |
| **推理灵活性** | ✅ 支持多种稀疏模式动态切换 | ❌ 固定稀疏结构 | ✅ 但性能差 |
| **稀疏化效果** | ✅ 性能接近 dense 模型 | ✅ 优秀（但需从头训） | ❌ 严重退化 |
| **KV Cache / FLOPs** | 显著降低（最高达 64×） | 显著降低 | 显著降低 |

> **核心优势**：**一个模型，多种高效推理路径**，兼顾性能与效率。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **预训练数据**：
  - `FineWeb-Edu`（100B tokens）
  - 扩展至 200B tokens 进行更大规模训练
- **下游任务评估**：
  - **Commonsense Reasoning Tasks**（短上下文）：ARC-C, ARC-E, HellaSwag, PIQA, Winogrande, LAMBADA
  - **LongBench**：多语言、多任务长文本理解基准（如 NarrativeQA, HotpotQA, 2WikiMultihopQA）
  - **RULER Benchmark**：NIAH（Needle-in-a-Haystack）任务，测试检索能力（exact match）

### **实验设置**

- **模型规模**：
  - 主要：1.5B 参数
  - 扩展：2.6B 参数
  - 小规模验证：200M 参数（PG19 数据集）
- **上下文长度**：4096（部分扩展至 16384 via RoPE scaling）
- **稀疏模式**：
  - Dilated attention（D=1~128）
  - Local window attention（W=256, 512, 1024）
  - Hybrid（层间或头间混合稀疏）
  - Top-k block attention（K=8,16,64）
- **Adaptation**：使用 1B tokens 进行轻量级微调以适配稀疏模式

### **评估指标**

| 类型 | 指标 |
|------|------|
| **语言建模** | Perplexity (PPL) |
| **常识推理** | Average accuracy (%) |
| **长文本理解** | LongBench 平均得分 |
| **检索能力** | NIAH exact match accuracy |
| **效率** | Prefill/decode latency, throughput (tokens/sec), FLOPs reduction |

### **基线方法对比**

- **Attention-only**：标准 dense attention
- **RAT**：chunk-based recurrence + inter-chunk dilated attention（从头训练）
- **Mamba2**, **GatedDeltaNet**：state space models
- **StreamingLLM**：local window + attention sink
- **Top-k block attention**：基于 Quest/MoBA 的重要性选择

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **1.5B 模型在 Commonsense Reasoning 上的表现（Table 4）**

| 方法 | Dilation (D) | Avg. Accuracy ↓ | FLOPs Reduction |
|------|--------------|------------------|-----------------|
| Dense Attention | 1 | 58.33 | ×1 |
| RAT+ | 16 | 58.06 | ×16 |
| RAT+ | 64 | 57.46 | ×64 |
| Attention → D=16 | 16 | 32.60 | ×16 |
| RAT (from scratch) | 16 | 56.99 | ×16 |

> **结论**：RAT+ 在 D=64 时仅损失 ~2–3 个点，而直接稀疏化 attention 几乎崩溃。

#### ✅ **LongBench 表现（Table 5）**

| 方法 | Dilation | Avg Score |
|------|----------|-----------|
| RAT+ (D=1) | 1 | 19.37 |
| RAT+ (D=16) | 16 | 18.50 |
| RAT+ (D=64) | 64 | 16.56 |
| RAT+ (D=16, W=256) | 16+local | 18.67 |

> 不同任务偏好不同稀疏策略（如 RBP 更喜欢 local window），证明 **hybrid 设计的价值**。

#### ✅ **Top-k Block Attention 检索能力（Fig. 2, Table 11）**

在 NIAH-MK-2 任务上（T=4096, D=64, K=16）：

| 方法 | Accuracy |
|------|----------|
| Attention → top-k | 63.2 |
| **RAT+ → top-k** | **93.8** |
| RAT+ (no ARL in SFT) | 76.8 |

> **引入 recurrence 显著提升 block selection 质量**，尤其在长距离检索任务中。

#### ✅ **效率提升（Fig. 3, 4）**

- **Prefill 加速**：
  - D=16 时，temporal-mixing operator 达到 **6.3×~8.5×** 加速（H=2048~4096）
- **End-to-end decoding throughput**（Fig. 4）：
  - 在 1.5B 模型、context=16K 下，D=64 实现 **超过 60× 更高吞吐量**

---

### **消融实验结果（Table 7）**

| 变体 | 是否有效？ | 说明 |
|------|------------|------|
| **L=T, L\*=64 (RAT+)** | ✅ 最佳 | 全序列递归 + ARL，稳定且高性能 |
| L=64, L=D adapt | ❌ | 推理时 chunk size 变化导致分布偏移 |
| 无 ARL（仅 D=1 训练） | ❌ | 递归未充分学习长程依赖，大 D 下性能差 |
| 无 adaptation | ❌ | 即使是 D=64 也需要少量适配 |

> **证明 ARL 和 full-sequence recurrence 的必要性**。

---

## 4. 关键结论和发现

### **主要发现**

1. **Dilated attention 本身不足以支撑有效的稀疏推理**，必须配合显式的机制（如 recurrence）来构建完整的感受野。
2. **Recurrence 是实现“train dense, infer sparse”的关键桥梁**，它使得模型能在 dense 训练中学习 long-range 依赖，并在稀疏推理中复用该能力。
3. **RAT+ 架构实现了前所未有的推理灵活性**：单个模型可适配多种稀疏模式（dilation、local、hybrid、top-k），仅需 1B tokens 微调。
4. **Recurrence 不仅对 dilated attention 有益，在 top-k block attention 中也显著优于纯 attention 模型**，因其增强了 block 内容表征。
5. **模型越大，dense 与 dilated 之间的性能差距越小**（Fig. 5），表明 RAT+ 具有良好的 scaling property。

### **方法的局限性**

- 当前实现仍依赖 PyTorch 默认算子，**未进行 CUDA 级优化**，实际加速潜力尚未完全释放。
- **Hybrid 模式的设计尚属探索阶段**，最优层/头组合依赖于预训练模型特性，需进一步研究自动化搜索。
- **Recurrence 引入额外参数和计算开销**（尽管很小），在极低延迟场景可能成为瓶颈。

### **未来工作方向**

1. **CUDA-level kernel 优化**：为 recurrence + sparse attention 设计专用高效 kernel。
2. **自动搜索最优 hybrid sparse 配置**：结合 layer-wise sparsifiability 分析。
3. **扩展至其他领域**：如 vision、speech、tokenizer-free（byte-level）建模。
4. **更高效的 recurrence 形式**：探索轻量化或条件式 recurrence。
5. **理论分析**：形式化解释为何 recurrence 能桥接 dense 与 sparse attention。

---

> **总结一句话**：  
> **RAT+ 开辟了一条“一次训练，多种高效推理”的新范式，通过引入 full-sequence recurrence 和 active learning，成功解决了 dense-to-sparse 转换中 dilated attention 性能崩溃的问题，在保持 high accuracy 的同时实现数十倍的效率提升。**

</details>

---

### 15. [A Probabilistic Framework for LLM-Based Model Discovery](https://arxiv.org/abs/2602.18266)

**Authors**: Stefan Wahl, Raphaela Schenk, Ali Farnoud, Jakob H. Macke, Daniel Gedon  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.18266v1  

#### Abstract
Automated methods for discovering mechanistic simulator models from observational data offer a promising path toward accelerating scientific progress. Such methods often take the form of agentic-style iterative workflows that repeatedly propose and revise candidate models by imitating human discover...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# A Probabilistic Framework for LLM-Based Model Discovery 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于 LLM 的模型发现方法（如 FunSearch、AlphaEvolve）通常采用**启发式代理流程**（agentic workflows），通过固定角色、优化循环或进化算子来迭代生成和改进模型。这些方法缺乏明确的概率框架，导致以下问题：
- 推理目标不清晰，难以解释失败模式；
- 缺乏理论保证（如收敛性）；
- 组件设计依赖经验，难以进行系统性分析。

本文旨在将 **LLM-based model discovery** 归约为一个形式化的**概率推断问题**，从而为该领域提供统一、可分析的理论基础。

---

### 提出的新方法：ModelSMC
作者提出 **ModelSMC** ——一种基于 **Sequential Monte Carlo (SMC)** 的算法，用于从观测数据中发现可解释的机制性模拟器模型（mechanistic simulator models）。其核心思想是：

> 将模型发现视为对后验分布 $ p(m|x_o) $ 的采样问题，其中 $ m $ 是能解释观测数据 $ x_o $ 的程序化模型。

#### 方法架构
- **粒子表示**：每个候选模型 $ m^k $ 被表示为一个“粒子”，包含代码实现、权重和上下文反馈。
- **三大步骤**（每轮迭代）：
  1. **Resample**：按权重重采样高可能性模型；
  2. **Propagate**：用 LLM 修改或克隆模型（以概率 $ 1-\alpha $ 提出新变体）；
  3. **Weight**：基于似然估计 $ p(x_o|m) $ 更新权重。

- **LLM 角色**：作为 proposal distribution，生成符合科学规范的新模型结构。
- **Likelihood Estimation**：使用 **Neural Likelihood Estimation (NLE)** 或 **NLE-PFN** 进行无梯度似然近似，支持非可微模型。

---

### 相比现有方法的优势
| 特性 | ModelSMC | 传统 LLM Agents (e.g., FunSearch+) |
|------|---------|-------------------------------|
| **理论基础** | 明确的概率推断框架（Bayesian inference） | 启发式操作流程 |
| **收敛性保证** | 在理想条件下具有一致性（consistency） | 无理论保障 |
| **参数处理** | 边际化参数 $ \theta $，减少不确定性影响 | 通常优化点估计 |
| **输出形式** | 加权模型集合，反映 posterior uncertainty | 单一最优模型 |
| **语言通用性** | 支持任意编程语言（如 R、Python） | 多限于 Python |

此外，ModelSMC 可自然集成 SMC 理论中的成熟技术，如自适应重采样、退火策略等。

---

## 2. 核心实验方法和设置

### 数据集
在三个真实世界与合成任务上验证：

| 任务 | 类型 | 描述 |
|------|------|------|
| **SIR Model** | 合成流行病学模型 | 基于经典 SIR 动力学生成时间序列数据，用于验证推理机制 |
| **Pharmacological Kidney Model** | 真实药理学系统 | 钾离子与醛固酮调节模型（R 实现），原始机制被替换为常数项，测试能否恢复正确反馈机制 |
| **Hodgkin-Huxley (HH) Neuron Model** | 真实神经动力学 | 来自 Allen Cell Types Database 的膜电位记录，挑战在于检测并补充已高性能基准模型的细微缺陷 |

---

### 实验设置
- **粒子数量**：$ N = 50 $
- **迭代次数**：最多 20–150 轮（视任务而定）
- **克隆概率**：$ \alpha = 0.7 \sim 0.8 $，平衡探索与利用
- **LLM**：默认使用 GPT-5-mini（via DSPy）
- **Likelihood Estimator**：NLE-PFN（基于 TabPFN，无需训练即可密度估计）

---

### 评估指标
1. **负平均对数边际似然**（↓）  
   $$
   -\log p(x_o | m) = -\frac{1}{M} \sum_{j=1}^M \log p(x_j | m)
   $$
   衡量模型整体拟合能力。

2. **负平均对数条件似然**（↓）  
   $$
   -\log p(x_o | \hat{\theta}, m)
   $$
   使用 MAP 参数估计后的性能。

3. **Posterior Predictive Checks**：可视化预测轨迹与真实数据的一致性。

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **FunSearch+** | 改进版 FunSearch，引入参数估计与 likelihood-based scoring，但仍为单岛演化 |
| **ModelSMC N=1** | 移除种群机制，仅保留单一模型演化路径，检验 population 的必要性 |

---

## 3. 主要实验结果和性能指标

### 定量性能对比（Table 1）

| Task | 方法 | $-\log p(x_o|m)$ ↓ | $-\log p(x_o|\hat{\theta},m)$ ↓ |
|------|------|---------------------|-------------------------------|
| **SIR** | ModelSMC | **-503.37** | **-492.67** |
|        | FunSearch+ | -500.00 | -492.05 |
|        | ModelSMC N=1 | -464.17 | -455.13 |
| **Kidney** | ModelSMC | **43.58** | **55.05** |
|           | ModelSMC N=1 | 46.29 | 58.63 |
| **HH** | ModelSMC | **25.18** | **15.47** |
|        | FunSearch+ | 28.96 | 12.05 |
|        | ModelSMC N=1 | 25.95 | 15.64 |

> 注：数值越小越好；加粗表示当前任务下最佳表现。

#### 关键观察：
- ModelSMC 在所有任务上均达到或超越基线，尤其在 **SIR 和 Kidney** 上显著优于其他方法；
- **ModelSMC N=1 性能下降明显**，说明 population-based inference 至关重要；
- FunSearch+ 在 HH 任务上条件似然略优，但边际似然更差，表明可能过拟合特定参数；
- ModelSMC 更稳健，在有限数据下仍能集中 posterior mass 到合理机制族。

---

### 消融实验与关键发现
#### （1）LLM-Free 验证实验（Sec. 4.1）
- 在有限模型空间（20 个 GMM 候选）中关闭 LLM 提案，仅靠 resampling + weighting。
- 结果显示：即使使用近似似然估计，粒子也能快速集中在真值模型上（Fig. 2），验证了 **SMC 推理机制的有效性**。

#### （2）Token 效率分析（Fig. G-1）
- 尽管 ModelSMC 输入/输出 token 更多（因携带上下文反馈），但在相同 token 预算下性能持平甚至超越基线。
- 表明：**概率框架未牺牲效率**，反而提升了单位计算资源的信息利用率。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **模型发现可以且应当被形式化为概率推断问题**：
   - 提供了统一视角理解 proposal、refinement、selection；
   - 使系统设计更具原则性，避免“黑箱代理”陷阱。

2. ✅ **ModelSMC 成功实现了 posterior sampling over executable programs**：
   - 输出不是单一模型，而是**加权模型集合**，揭示哪些机制被数据支持、哪些不确定、哪些被排除；
   - 在药理学与神经科学任务中发现了与真实机制高度一致的反馈结构（如醛固酮调控公式、慢钾电流扩展）。

3. ✅ **Population-based inference 至关重要**：
   - ModelSMC N=1 明显劣于完整版本，说明多样性维持与竞争选择是成功的关键。

4. ✅ **理论可分析性强**：
   - 在理想假设下证明了算法一致性（Theorem 3.1），建立了与经典 SMC 理论的联系；
   - 为 future work 提供了严谨的数学工具（如 variance control、convergence analysis）。

---

### 局限性
| 限制 | 说明 |
|------|------|
| **计算成本高** | 每次迭代需多次模拟与 likelihood 估计，尤其当模型复杂时 |
| **似然代理存在偏差** | NLE/NLE-PFN 是近似，若 misspecified 可能误导搜索方向 |
| **LLM 提案能力瓶颈** | 若 LLM 无法生成某类结构（如特定 ODE 形式），则整个 posterior 将缺失该区域 |
| **缺乏模型空间几何结构** | 当前方法将模型视为离散点，难定义“相似性”或进行语义聚类 |

---

### 未来工作方向
1. **提升效率**：
   - 引入 multi-fidelity simulation；
   - 自适应粒子预算或 early rejection；
   - 利用 SMC 中的 rejuvenation moves 减少退化。

2. **增强鲁棒性**：
   - 开发 uncertainty-aware likelihood estimation；
   - 引入 robust Bayes 或 PAC-Bayes 框架。

3. **深化语义结构建模**：
   - 构建 program embedding space，支持基于语义的 proposal；
   - 结合 retrieval-augmented generation 提升相关结构生成概率。

4. **扩展应用场景**：
   - 应用于更多领域（气候建模、基因调控网络）；
   - 探索 discovery + control 联合优化任务。

---

> **总结一句话**：  
> 本文首次将 LLM-driven model discovery 系统地置于 **probabilistic inference** 框架之下，提出了 **ModelSMC** 这一兼具理论深度与实践效能的新范式，推动自动化科学发现迈向更可靠、可解释、可分析的新阶段。

</details>

---

### 16. [Analyzing LLM Instruction Optimization for Tabular Fact Verification](https://arxiv.org/abs/2602.17937)

**Authors**: Xiaotang Du, Giwon Hong, Wai-Chung Kwan, Rohit Saxena, Ivan Titov, Pasquale Minervini, Emily Allaway  
**Category**: cs.CL  
**Published**: 2026-02-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.17937v1  

#### Abstract
Instruction optimization provides a lightweight, model-agnostic approach to enhancing the reasoning performance of large language models (LLMs). This paper presents the first systematic comparison of instruction optimization, based on the DSPy optimization framework, for tabular fact verification. W...

---

### 17. [A reliability- and latency-driven task allocation framework for workflow applications in the edge-hub-cloud continuum](https://arxiv.org/abs/2602.18158)

**Authors**: Andreas Kouloumpris, Georgios L. Stavrinides, Maria K. Michael, Theocharis Theocharides  
**Category**: cs.DC  
**Published**: 2026-02-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.18158v1  

#### Abstract
A growing number of critical workflow applications leverage a streamlined edge-hub-cloud architecture, which diverges from the conventional edge computing paradigm. An edge device, in collaboration with a hub device and a cloud server, often suffices for their reliable and efficient execution. Howev...

---

### 18. [Optimal Multi-Debris Mission Planning in LEO: A Deep Reinforcement Learning Approach with Co-Elliptic Transfers and Refueling](https://arxiv.org/abs/2602.17685)

**Authors**: Agni Bandyopadhyay, Gunther Waxenegger-Wilfing  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.17685v1  

#### Abstract
This paper addresses the challenge of multi target active debris removal (ADR) in Low Earth Orbit (LEO) by introducing a unified coelliptic maneuver framework that combines Hohmann transfers, safety ellipse proximity operations, and explicit refueling logic. We benchmark three distinct planning algo...

---

### 19. [Causality by Abstraction: Symbolic Rule Learning in Multivariate Timeseries with Large Language Models](https://arxiv.org/abs/2602.17829)

**Authors**: Preetom Biswas, Giulia Pedrielli, K. Sel\c{c}uk Candan  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.17829v1  

#### Abstract
Inferring causal relations in timeseries data with delayed effects is a fundamental challenge, especially when the underlying system exhibits complex dynamics that cannot be captured by simple functional mappings. Traditional approaches often fail to produce generalized and interpretable explanation...

---

### 20. [The Statistical Signature of LLMs](https://arxiv.org/abs/2602.18152)

**Authors**: Ortal Hadad, Edoardo Loru, Jacopo Nudo, Niccol\`o Di Marco, Matteo Cinelli, Walter Quattrociocchi  
**Category**: cs.CL  
**Published**: 2026-02-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.18152v1  

#### Abstract
Large language models generate text through probabilistic sampling from high-dimensional distributions, yet how this process reshapes the structural statistical organization of language remains incompletely characterized. Here we show that lossless compression provides a simple, model-agnostic measu...

---

### 21. [GPU Memory and Utilization Estimation for Training-Aware Resource Management: Opportunities and Limitations](https://arxiv.org/abs/2602.17817)

**Authors**: Ehsan Yousefzadeh-Asl-Miandoab, Reza Karimzadeh, Danyal Yorulmaz, Bulat Ibragimov, P{\i}nar T\"oz\"un  
**Category**: cs.DC  
**Published**: 2026-02-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.17817v1  

#### Abstract
Collocating deep learning training tasks improves GPU utilization but causes drastic slowdowns due to resource contention and risks Out-of-Memory (OOM) failures. Accurate memory estimation is essential for robust collocation, while GPU utilization -- a key proxy for resource contention -- enables in...

---

### 22. [BioBridge: Bridging Proteins and Language for Enhanced Biological Reasoning with LLMs](https://arxiv.org/abs/2602.17680)

**Authors**: Yujia Wang, Jihong Guan, Wengen Li, Shuigeng Zhou, Xuhong Wang  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.17680v1  

#### Abstract
Existing Protein Language Models (PLMs) often suffer from limited adaptability to multiple tasks and exhibit poor generalization across diverse biological contexts. In contrast, general-purpose Large Language Models (LLMs) lack the capability to interpret protein sequences and fall short in domain-s...

---

### 23. [Multi-material Multi-physics Topology Optimization with Physics-informed Gaussian Process Priors](https://arxiv.org/abs/2602.17783)

**Authors**: Xiangyu Sun, Shirin Hosseinmardi, Amin Yousefpour, Ramin Bostanabad  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.17783v1  

#### Abstract
Machine learning (ML) has been increasingly used for topology optimization (TO). However, most existing ML-based approaches focus on simplified benchmark problems due to their high computational cost, spectral bias, and difficulty in handling complex physics. These limitations become more pronounced...

---

### 24. [Grassmannian Mixture-of-Experts: Concentration-Controlled Routing on Subspace Manifolds](https://arxiv.org/abs/2602.17798)

**Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.17798v1  

#### Abstract
Mixture-of-Experts models rely on learned routers to assign tokens to experts, yet standard softmax gating provides no principled mechanism to control the tradeoff between sparsity and utilization. We propose Grassmannian MoE (GrMoE), a routing framework that operates on the Grassmannian manifold of...

---

### 25. [Influence-Preserving Proxies for Gradient-Based Data Selection in LLM Fine-tuning](https://arxiv.org/abs/2602.17835)

**Authors**: Sirui Chen, Yunzhe Qi, Mengting Ai, Yifan Sun, Ruizhong Qiu, Jiaru Zou, Jingrui He  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.17835v1  

#### Abstract
Supervised fine-tuning (SFT) relies critically on selecting training data that most benefits a model's downstream performance. Gradient-based data selection methods such as TracIn and Influence Functions leverage influence to identify useful samples, but their computational cost scales poorly, makin...

---

### 26. [Hardware-Friendly Input Expansion for Accelerating Function Approximation](https://arxiv.org/abs/2602.17952)

**Authors**: Hu Lou, Yin-Jun Gao, Dong-Xiao Zhang, Tai-Jiao Du, Jun-Jie Zhang, Jia-Rui Zhang  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.17952v1  

#### Abstract
One-dimensional function approximation is a fundamental problem in scientific computing and engineering applications. While neural networks possess powerful universal approximation capabilities, their optimization process is often hindered by flat loss landscapes induced by parameter-space symmetrie...

---

### 27. [Provable Adversarial Robustness in In-Context Learning](https://arxiv.org/abs/2602.17743)

**Authors**: Di Zhang  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.17743v1  

#### Abstract
Large language models adapt to new tasks through in-context learning (ICL) without parameter updates. Current theoretical explanations for this capability assume test tasks are drawn from a distribution similar to that seen during pretraining. This assumption overlooks adversarial distribution shift...

---

### 28. [Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](https://arxiv.org/abs/2602.17898)

**Authors**: Jingquan Yan, Yuwei Miao, Peiran Yu, Junzhou Huang  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.17898v1  

#### Abstract
Attention-based regression models are often trained by jointly optimizing Mean Squared Error (MSE) loss and Pearson correlation coefficient (PCC) loss, emphasizing the magnitude of errors and the order or shape of targets, respectively. A common but poorly understood phenomenon during training is th...

---

### 29. [Parameter-Efficient Domain Adaptation of Physics-Informed Self-Attention based GNNs for AC Power Flow Prediction](https://arxiv.org/abs/2602.18227)

**Authors**: Redwanul Karim (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), Changhun Kim (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), Timon Conrad (Institute of Electrical Energy Systems, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Germany), Nora Gourmelon (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), Julian Oelhaf (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), David Riebesel (Institute of Electrical Energy Systems, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Germany), Tom\'as Arias-Vergara (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), Andreas Maier (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany), Johann J\"ager (Institute of Electrical Energy Systems, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Germany), Siming Bayer (Pattern Recognition Lab, Friedrich-Alexander-Universit\"at Erlangen-N\"urnberg, Erlangen, Germany)  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.18227v1  

#### Abstract
Accurate AC-PF prediction under domain shift is critical when models trained on medium-voltage (MV) grids are deployed on high-voltage (HV) networks. Existing physics-informed graph neural solvers typically rely on full fine-tuning for cross-regime transfer, incurring high retraining cost and offeri...

---

### 30. [Asking Forever: Universal Activations Behind Turn Amplification in Conversational LLMs](https://arxiv.org/abs/2602.17778)

**Authors**: Zachary Coalson, Bo Fang, Sanghyun Hong  
**Category**: cs.LG  
**Published**: 2026-02-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.17778v1  

#### Abstract
Multi-turn interaction length is a dominant factor in the operational costs of conversational LLMs. In this work, we present a new failure mode in conversational LLMs: turn amplification, in which a model consistently prolongs multi-turn interactions without completing the underlying task. We show t...

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
