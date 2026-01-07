# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-01-07 06:14:31 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [LoRA-Drop: Temporal LoRA Decoding for Efficient LLM Inference](https://arxiv.org/abs/2601.02569)

**Authors**: Hossein Rajabzadeh, Maryam Dialameh, Chul B. Park, Il-Min Kim, Hyock Ju Kwon  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2601.02569v1  

#### Abstract
Autoregressive large language models (LLMs) are bottlenecked by sequential decoding, where each new token typically requires executing all transformer layers. Existing dynamic-depth and layer-skipping methods reduce this cost, but often rely on auxiliary routing mechanisms or incur accuracy degradat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LoRA-Drop: Temporal LoRA Decoding for Efficient LLM Inference 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在自回归推理过程中面临严重的**计算瓶颈**，因为每个新 token 都需要通过全部 Transformer 层进行顺序解码，导致高延迟和高内存开销。尽管已有动态深度或层跳过方法尝试缓解此问题，但它们通常依赖额外的路由机制或在跳过层时缺乏补偿性更新，从而可能造成精度下降。

此外，传统的 KV-Cache 在长序列生成中占用大量显存，限制了实际部署效率。

### 提出的新方法：LoRA-Drop
本文提出 **LoRA-Drop** —— 一种即插即用（plug-and-play）的高效推理框架，其核心思想是利用 LLM 中隐藏状态的**时间冗余性**（temporal redundancy），对中间层实施**周期性的轻量级更新**而非完整计算。

#### 方法核心机制：
- **固定子集的可跳过层（droppable layers）**：选择部分中间层注入 LoRA 模块。
- **双模式交替调度**：
  - **Full Mode（刷新步）**：每 $k+1$ 步执行一次完整的前向传播，重新计算所有层并更新 KV-Cache，防止误差累积。
  - **LoRA Mode（轻量步）**：其余 $k$ 步中，仅对“可跳过层”应用 LoRA 更新，复用上一 token 的隐藏状态，并叠加一个低秩修正项 $\Delta = \alpha \cdot BA x^{t-1}$。
- **无需路由网络**：调度策略为固定周期，不引入额外参数或决策模块。
- **兼容 KV-Caching**：在 LoRA 步骤中跳过可跳过层的 KV 更新，显著减少 KV-Cache 写入频率和内存占用。

### 相比现有方法的优势
| 特性 | LoRA-Drop | Unified Layer Skipping / FlexiDepth | Speculative Decoding | Quantization |
|------|-----------|-------------------------------------|------------------------|-------------|
| 是否需辅助模型/路由 | ❌ 否 | ⚠️ 部分需要 | ✅ 是（draft model） | ❌ 否 |
| 是否保留语义一致性 | ✅ 是（复用+微调） | ❌ 完全跳过层 | ⚠️ 接受率影响质量 | ✅ 是 |
| 是否支持 KV-Cache 优化 | ✅ 显著减少写入 | ❌ 一般不涉及 | ✅ 可结合 | ✅ 可结合 |
| 部署复杂度 | ✅ 极低（post-hoc 微调即可） | ✅ 较低 | ❌ 高（两模型协同） | ⚠️ 中等（需校准） |
| 精度损失控制 | ✅ <0.5 pp | ⚠️ 通常 >1 pp | ⚠️ 不稳定 | ✅ 可控 |

> **创新亮点总结**：
> - 利用 **temporal redundancy** 设计 **temporal LoRA decoding** 调度；
> - 提出 **周期性 refresh + LoRA correction** 机制，在速度与稳定性之间取得平衡；
> - 实现 **KV-Cache footprint 减少 45–55%**，同时保持极小精度损失；
> - 完全兼容预训练模型，仅需少量持续微调（continual fine-tuning）即可集成。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖多类任务以验证泛化能力：

| 类别 | 数据集 |
|------|-------|
| **通用理解与知识** | MMLU（5-shot）、ARC-e/c、HellaSwag (HS)、WinoGrande、PIQA、OpenBookQA、RACE |
| **代码生成** | HumanEval、MBPP（Pass@1 / Pass@10） |
| **数学与推理** | GSM8K、MATH、BBH |
| **长文本与检索** | LongBench、Needle-in-a-Haystack |
| **多语言理解** | XNLI、XCOPA |

测试数据来自 `RefinedWeb` 和标准 benchmark 套件（如 `LM-Eval Harness v0.4.2`）。

### 实验设置
- **模型家族**：
  - LLaMA2-7B
  - LLaMA3-8B
  - Qwen2.5-7B
  - Qwen2.5-14B
- **LoRA 参数**：
  - Rank $r = 16$
  - Scaling $\alpha = 16$
  - 注入位置：Transformer block 输出端（block-level）
- **调度参数**：
  - Drop Ratio $p \in \{0.25, 0.5, 0.75\}$：表示可跳过层的比例
  - Temporal Window $k \in \{1,2,3,5\}$：连续使用 LoRA 模式的 token 数量，之后执行 full refresh
- **硬件环境**：
  - GPU：NVIDIA A100 / V100
  - 精度：BF16 混合精度
  - 序列长度：2048（文本）、1024（HumanEval）
- **微调阶段**：
  - 在约 15B tokens 的 RefinedWeb 上进行轻量级持续预训练，仅更新 LoRA 参数，冻结主干权重。

### 评估指标
| 指标类别 | 具体指标 |
|--------|---------|
| **准确性** | Zero-shot Accuracy (%)、Pass@1 (%) |
| **效率** | Tokens/sec（吞吐量）、Speedup ×（相对 baseline 加速比） |
| **资源消耗** | KV-Cache 内存（MB）、FLOPs（理论计算量） |
| **延迟分布** | p50（中位延迟）、p95（尾部延迟） |

### 基线方法对比
- **Full Model（Baseline）**：无任何优化的标准推理
- **Unified Layer Skipping [18]**：基于置信度跳过层
- **FlexiDepth [19]**：动态分配计算预算
- （注：未直接比较 speculative decoding 或 quantization，因其正交且工程实现差异大）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（见 Table I & V）

| 模型 | 方法 | 平均准确率 (%) | 相对准确率变化 | 速度提升 (×) | KV-Cache 节省 |
|------|------|----------------|----------------|--------------|---------------|
| LLaMA2-7B | Full | 64.6 | — | 1.00 | — |
| | LoRA-Drop ($p=0.5, k=3$) | 64.5 | -0.1 pp | **1.68×** | ~40% |
| | LoRA-Drop ($p=0.75, k=3$) | 62.8 | -1.8 pp | **2.35×** | ~65% |
| Qwen2.5-7B | Full | 66.5 | — | 1.00 | — |
| | LoRA-Drop ($p=0.5, k=3$) | 66.3 | -0.2 pp | **1.73×** | ~42% |
| | LoRA-Drop ($p=0.75, k=3$) | 64.1 | -2.5 pp | **2.42×** | ~68% |
| LLaMA3-8B | Full | 67.7 | — | 1.00 | — |
| | LoRA-Drop ($p=0.5, k=3$) | 67.4 | -0.3 pp | **1.70×** | ~43% |
| | LoRA-Drop ($p=0.75, k=3$) | 65.3 | -2.4 pp | **2.38×** | ~67% |
| Qwen2.5-14B | Full | 69.3 | — | 1.00 | — |
| | LoRA-Drop ($p=0.75, k=3$) | 66.8 | -2.5 pp | **2.60×** | ~70% |

> ✅ **最高加速达 2.6×，KV-Cache 最多减少 55%**

### 与基线方法对比（Table I）
- 在相同 drop ratio 下，**LoRA-Drop 精度优于 Unified Layer Skipping 和 FlexiDepth**，平均高出 0.5–1.0 pp。
- 速度方面，LoRA-Drop 在 $p=0.5$ 时达到 **1.68–1.73×**，略高于 FlexiDepth（1.50–1.58×），远超原始模型。
- 在 $p=0.75$ 极端设置下，LoRA-Drop 仍能维持合理输出质量，而其他方法已出现明显退化。

### 消融实验结果（Table III & IV）

#### 不同 $p$ 和 $k$ 的权衡分析
| $p$ | $k$ | 平均准确率变化 | 速度提升 | KV-Memory (LLaMA2-7B) |
|-----|----|----------------|----------|------------------------|
| 0.25 | 3 | +0.2 pp | 1.27× | 12.1 GB → ↓17% |
| 0.50 | 3 | -0.1 pp | 1.68× | 8.5 GB → ↓41% |
| 0.75 | 3 | -1.8 pp | 2.35× | 5.0 GB → ↓65% |
| 0.75 | 5 | -2.5 pp | 2.45× | 4.2 GB → ↓71% |

> 🔍 发现“安全区”（safe zone）：**$p \leq 0.5$, $k \leq 3$** 可保证精度损失 ≤0.5 pp，同时获得 1.6–1.8× 加速。

#### 延迟与尾部表现（Table IV）
- **p50 延迟降低至 6–8 ms/token**（baseline ~13 ms）
- **p95 尾延迟可控**：由于每 $k+1$ 步才有一次 full refresh，只要 $k > 19$，p95 即由 LoRA 步主导，适合高吞吐服务场景。

#### LoRA 注入位置消融（Table V）
| 注入方式 | 准确率 | 速度提升 |
|--------|--------|----------|
| Block-level（默认） | ✅ 64.6% | **1.55×** |
| Attention+MLP 分离注入 | ✅ 64.6% | 1.37× |

> 结论：**block-level 注入最优**，兼顾精度与最大吞吐。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 隐藏状态存在显著时间冗余**：相邻 token 的隐藏状态相似度高达 0.6–0.85，未来 3–6 个 token 仍高度相关（Fig. 1 & 4），支持“复用+微调”的合理性。
2. ✅ **LoRA-Drop 在多个任务上保持鲁棒性**：即使在 GSM8K、MATH 等复杂推理任务中，$p=0.5, k=3$ 配置也能将性能损失控制在 0.5 pp 以内。
3. ✅ **KV-Cache 节省可建模预测**：公式推导表明节省比例为：
   $$
   \text{Save\%}(p,w) = 100 \times \left(1 - \frac{a + (1-p)S + pS/w}{L}\right)
   $$
   其中 $w = k+1$，$S$: 可跳过层数，$a$: 永久激活层数（如首三层+末层）。
4. ✅ **存在清晰的 Pareto 前沿**：可通过调节 $p$ 和 $k$ 灵活控制效率-精度权衡，适用于不同应用场景（实时响应 vs. 高质量生成）。

### 方法局限性
- ❗ **依赖周期性 full refresh**：虽然避免了漂移，但在极端长序列中仍可能积累误差。
- ❗ **当前为固定调度**：尚未引入基于 token 复杂度的自适应调度（如 entropy-based triggering）。
- ❗ **对浅层/深层敏感层假设较强**：默认前几层和最后一层不可跳过，可能限制进一步压缩空间。
- ❗ **仅验证了文本任务**：在多模态或检索增强场景中的有效性待探索。

### 未来工作方向
- 🔄 **自适应调度策略**：基于 logit margin、entropy 或 attention divergence 动态决定是否 refresh。
- 🧠 **扩展至多模态 Transformer**：应用于视觉-语言模型中的跨模态推理加速。
- 🔍 **结合其他优化技术**：与量化（Quantization）、KV Cache 压缩、Speculative Decoding 联合使用，实现更深层次的端到端优化。
- 📈 **构建自动化配置推荐系统**：根据输入长度、任务类型自动选择最优 $(p, k)$ 组合。

---

> 💡 **一句话总结**：  
> LoRA-Drop 通过挖掘 LLM 解码过程中的**时间冗余性**，提出了一种简洁高效的**周期性 LoRA 更新机制**，在几乎不影响精度的前提下实现了高达 **2.6× 的推理加速** 和 **55% 的 KV-Cache 节省**，为大规模语言模型的实际部署提供了一个极具前景的轻量化解决方案。

GitHub 开源地址：[https://github.com/hosseinbv/LoRA-Drop.git](https://github.com/hosseinbv/LoRA-Drop.git)

</details>

---

### 2. [Lil: Less is Less When Applying Post-Training Sparse-Attention Algorithms in Long-Decode Stage](https://arxiv.org/abs/2601.03043)

**Authors**: Junhao Hu, Fangze Li, Mingtao Xu, Feifan Meng, Shiju Zhao, Tiancheng Hu, Ting Peng, Anmin Liu, Wenrui Huang, Chenxu Liu, Ziyue Hua, Tao Xie  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2601.03043v1  

#### Abstract
Large language models (LLMs) demonstrate strong capabilities across a wide range of complex tasks and are increasingly deployed at scale, placing significant demands on inference efficiency. Prior work typically decomposes inference into prefill and decode stages, with the decode stage dominating to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Lil: Less is Less When Applying Post-Training Sparse-Attention Algorithms in Long-Decode Stage

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文揭示了一个在**Post-Training Sparse-Attention (PTSD)** 算法中被忽视的关键问题，称为 **“Lil” (Less is Less)**。尽管稀疏注意力（sparse attention）通过减少每个解码步的计算量来加速推理，但由于其在解码过程中丢失上下文信息，导致模型需要生成更长的序列来补偿，从而反而增加了端到端的延迟和内存消耗。

这一现象在推理密集型任务（如数学推理、Chain-of-Thought）中尤为严重，使得原本旨在提升效率的技术可能适得其反。

### 提出的新方法与思路
为应对 Lil 问题，作者提出了 **Guardian**，一种用于解码阶段的**早停算法（early-stopping algorithm）**，其核心思想是：
- 利用 **LZ77 压缩算法**动态监测生成序列的信息增益；
- 当连续若干步的压缩长度增长低于阈值时，判定信息增益停滞，提前终止解码。

该方法基于一个关键洞察：**信息冗余的增加会显著降低可压缩性**，因此压缩率的变化可以作为信息增益的代理指标。

### 相比现有方法的优势
- **无需重新训练**：Guardian 是一个后训练（post-training）、即插即用的模块，适用于任何已部署的 LLM。
- **高效且轻量**：LZ77 压缩成本极低（约 34ms 处理 128k tokens），每 250 步执行一次，开销可忽略。
- **通用性强**：不仅适用于稀疏注意力场景，也能有效压缩全注意力下因冗余推理模式导致的过长输出（如 reward hacking 或人类偏好导致的 verbosity）。
- **高性价比**：在推理密集型基准上，**最高节省 90% 的 token 消耗，同时准确率下降小于 2%**。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个面向复杂推理的公开数据集上进行：
- **GSM8K**：小学数学应用题，需多步算术推理。
- **MATH-500**：高中数学竞赛题，涵盖代数、几何等五级难度。
- **AIME**：美国数学邀请赛题目，挑战顶尖高中生。

每个数据集选取前 200 个测试样例进行评估。

### 模型
使用三种主流开源模型，覆盖不同架构与规模：
- **DSR (DeepScaleR-1.5B-Preview)**：1.5B 参数，dense 架构。
- **DSL (DeepSeek-R1-Distill-Llama-8B)**：8B 参数，dense 架构。
- **Qwe (Qwen1.5-MoE-A2.7B-Chat)**：2.7B 参数，MoE 架构。

### 基线方法对比
比较了五种典型的 PTSD 算法：
- **H2O**, **Sink**：保留部分 KV 缓存，动态丢弃不重要 token。
- **infLLM**, **Quest**：保留全部 KV 向量，仅限制 attention 范围。
- **Full Attention**：完整注意力机制，作为性能上限。

所有算法均在不同 **cache budget**（128–1024）下测试。

### 评估指标
- **Token Savings**：应用 Guardian 后相比原方法节省的 token 数比例。
- **Accuracy**：答案与标准答案的数学等价性判断，报告正确率及变化（ΔAccuracy）。
- **Compression Ratio**：使用 LZ77 压缩后的长度 / 原始长度，衡量信息密度。

### 实验环境
- 单张 NVIDIA A100-80GB GPU
- CUDA 12.6, Ubuntu 20.04
- 使用 Hugging Face 实现框架

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）
| 方法 | 最大 Token 节省 | 准确率下降 |
|------|------------------|------------|
| **Guardian + PTSD** | **up to 90%** | **< 2%** |

具体表现如下：
- 在 **DSR + GSM8K + Sink (128)** 配置下，token 节省高达 **83.0%**，准确率仅降 **0.9%**。
- 多数配置下节省 **50%-80%** token，准确率波动在 ±2% 内。
- 在某些情况下（如 DSL on MATH-500），Guardian **反而提升了准确率**，原因是阻止了模型在正确答案后继续冗余验证并遗忘答案。

### 与基线方法的对比结果
- 所有 PTSD 方法在无 Guardian 时均表现出明显的 **output length 增加**（最多达 90%），而 accuracy 并未同步提升。
- Guardian 显著缓解了这一问题，在保持 accuracy 接近原水平的同时大幅缩短输出。
- 即使在保守配置（如 cache budget=1024）下，仍可观测到冗余，说明 Lil 问题是普遍存在的。

### 消融实验结果
#### （1）按正确/错误案例分别统计（Table 3）
| 类型 | Token 节省来源 |
|------|----------------|
| **Incorrect Cases** | 主要来自无限循环或无效重复生成的截断 |
| **Correct Cases** | 来自答案生成后不必要的反复验证 |

表明 Guardian 对两类冗余均有抑制作用。

#### （2）应用于 Full Attention 的效果（Table 2）
| 模型 | 平均 Token 节省 |
|------|------------------|
| DSR | 12.5% |
| DSL | 5.9% |
| Qwe | 1.3% |

说明即使在全注意力下也存在冗余推理，Guardian 可作为通用的 **Chain-of-Thought 压缩工具**。

#### （3）压缩成本分析
- LZ77 压缩 128k 随机字符串耗时约 **34ms**，接近单个 token 的 decode 时间。
- 设置 `f=250`（每 250 步压缩一次），总体 overhead 可忽略。

---

## 4. 关键结论和发现

### 主要发现
1. **Lil 问题是 PTSD 算法中的“房间里的大象”**：
   - 稀疏注意力虽加快单步推理，但因信息丢失迫使模型延长输出，最终导致 **端到端效率下降**。
   - 输出呈现“信息丢失 → 尝试重建 → 再次丢失”的恶性循环。

2. **压缩比是衡量信息增益的有效代理**：
   - 使用 LZ77 可量化生成过程中的信息密度变化。
   - 实验显示，当压缩长度增长趋于平缓时，信息增益基本停滞。

3. **Guardian 高效且鲁棒**：
   - 通过监测压缩增长实现早停，能以极小代价避免大量无效生成。
   - 在多种模型、数据集和稀疏算法上均有效。

4. **适用范围超出 PTSD**：
   - Guardian 同样适用于全注意力下的冗余 CoT 生成，具有通用压缩潜力。

### 方法的局限性
- **依赖压缩算法的敏感性**：阈值 `t` 和频率 `f` 需合理设置，虽然实验表明对 `t/f ∈ (0.02, 1)` 区间鲁棒，但仍需调参。
- **评估范围有限**：
  - 仅测试了 3 个模型和 3 个数据集，未覆盖更大上下文（如 Qwen2.5-Max）或代码生成任务（如 GPQA、Codeforces）。
  - PTSD 算法仅覆盖 4 种代表性方法，未包含所有变体（如 clusterKV）。
- **假设生成是顺序不可逆的**：一旦停止无法恢复，若早期误判可能导致遗漏关键推理步骤（但在实验中极少发生）。

### 未来工作方向
- 将 Guardian 应用于更广泛的 **prolonged CoT 场景**，尤其是由训练偏差（data quality, reward hacking）引起的冗长推理。
- 探索与其他 CoT 压缩方法（如 ThinkPrune, HALT-COT）的联合优化。
- 设计更细粒度的信息监控机制，例如结合语义相似度或逻辑一致性检测。
- 扩展至 prefill 阶段或其他稀疏化策略（如 DeepSeek NSA）的协同优化。

---

> **总结一句话**：  
> 本文指出，稀疏注意力可能导致“少即是更少”（Lil）的悖论——看似节省计算，实则因信息丢失引发更长生成；为此提出 Guardian，利用压缩感知实现智能早停，在几乎不影响准确率的前提下最多节省 90% 的 token 消耗，为高效长推理提供了一种简单而强大的解决方案。

</details>

---

### 3. [Jenius Agent: Towards Experience-Driven Accuracy Optimization in Real-World Scenarios](https://arxiv.org/abs/2601.01857)

**Authors**: Defei Xia, Bingfeng Pi, Shenbin Zhang, Song Hua, Yunfei Wei, Lei Zuo  
**Category**: cs.AI  
**Published**: 2026-01-07  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2601.01857v1  

#### Abstract
As agent systems powered by large language models (LLMs) advance, improving the task performance of an autonomous agent, especially in context understanding, tool usage, and response generation, has become increasingly critical. Although prior studies have advanced the overall design of LLM-based ag...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前基于大语言模型（LLM）的自主智能体系统在实际应用中面临三大挑战：
- **固定提示（prompt）导致意图理解偏差**：静态或通用的提示无法适应任务状态变化，容易造成行为不稳定和输出不一致。
- **工具调用缺乏上下文感知**：依赖预定义工具列表或手工规则，难以在模糊或多领域场景下准确选择合适工具，导致无效或错误调用。
- **长对话中的上下文冗余**：随着交互轮次增加，历史记录膨胀，增加了token消耗并稀释关键信息，影响推理质量。

这些问题限制了智能体在复杂、多步、真实世界任务中的准确性、效率和鲁棒性。

---

### **提出了什么新方法或新思路**
本文提出了一种名为 **Jenius-Agent** 的端到端智能体框架，围绕“经验驱动”的优化理念，集成三个核心模块：

#### **(1) 自适应提示生成（Adaptive Prompt Generation）**
- 动态融合角色指令、任务状态和用户上下文，生成与当前情境匹配的系统提示。
- 引入**意图分类机制**（intent taxonomy），将用户请求分为四类：社交互动、创意生成、事实回忆、工具增强推理，并据此定制响应策略。
- 包含安全控制层，防止幻觉、敏感内容输出及非法工具调用。

#### **(2) 上下文感知的工具编排（Context-Aware Tool Orchestration）**
- 构建基于 **Model Context Protocol (MCP)** 的标准化工具管理机制。
- 工具表示为高维嵌入向量（使用 Qwen3 Embedding 模型），通过语义相似度进行检索。
- 提出三阶段筛选流程：
  1. **Top-M候选检索**：按查询与工具嵌入的相似度排序；
  2. **拐点过滤（Inflection Point Filtering）**：结合相似度跳跃法和Kneedle算法确定相关性阈值；
  3. **动态截断**：保留最优数量 $ N = \min(N_{\text{jump}}, N_{\text{kneedle}}) $ 的工具，若不足则补足至10个。

#### **(3) 分层记忆管理（Hierarchical Memory Management）**
- **对话级对齐**：维护 `Human → AI → Tool → AI` 的标准交互序列，修复因失败调用导致的消息缺失。
- **会话级压缩**：当消息数超过阈值 $ K $ 时，自动摘要早期对话内容，仅保留最新一轮完整上下文。
- 总结后的语义信息以 `SystemMessage` 形式插入输入流前端，提升注意力保留效果。

---

### **相比现有方法的优势**
| 维度 | 传统方法缺陷 | Jenius-Agent 改进 |
|------|--------------|------------------|
| **Prompt工程** | 固定模板，缺乏动态调整 | 实现运行时自适应重构，提升任务对齐 |
| **Tool使用** | 静态列表或硬编码规则 | 语义检索+动态过滤，减少误调用 |
| **Memory管理** | 简单滑窗截断或全量保留 | 结构化分层压缩，兼顾完整性与效率 |
| **协议兼容性** | 各自为政，接口不统一 | 支持 MCP、ACP、A2A 等新兴通信协议 |
| **部署可行性** | 资源开销大，难落地 | 显著降低token成本与延迟，适合生产环境 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

#### **(1) APIGen（公开基准）**
- 来源：人工标注 + LLM生成混合
- 规模：60K样本，覆盖21类API，平均每调用2.3个参数
- 特点：单轮交互，关注函数调用准确性
- 本文改进：每条query注入100个无关工具，模拟高噪声环境，测试工具检索能力

#### **(2) Jenius-bench（新构建的真实任务数据集）**
- 规模：850高质量样本，涵盖38类工具
- 应用场景：旅行规划、票务预订、网页生成、学术检索等
- 多轮交互，包含完整执行轨迹（tool调用、输入、输出、agent响应）
- 工具附带丰富语义描述（目的、前提、示例等）
- 所有路径经专家审核，确保正确性和连贯性

> ✅ 对比见表1：Jenius-bench 更贴近现实，强调跨轮推理、上下文依赖和真实工具调用。

---

### **实验设置与评估指标**

#### **评估框架三大维度**
| 维度 | 指标体系 | 内容 |
|------|--------|------|
| **过程保真度（Procedural Fidelity）** | **4T Metrics** | TCR, TFR, TIR, TPS |
| **输出质量（Output Quality）** | **CRCFF Metrics** | Correctness, Relevance, Completeness, Fluency, Faithfulness |
| **效率（Efficiency）** | Token Consumption | 输入/输出token总量，反映计算开销 |

##### **4T Metrics 定义**
- **TCR (Task Completion Rate)**：所有必需工具按序正确调用的比例
- **TFR (Task Failure Rate)**：无有效执行（空调用或报错）的任务比例
- **TIR (Task Incompletion Rate)**：部分完成（漏调、错序、无关调用）的比例
- **TPS (Task Performance Score)**：综合考虑正确、错误、遗漏工具的加权得分

##### **CRCFF Metrics**
由 Qwen 和 DeepSeek 等 LLM evaluator 进行五维打分（0–10分制）

---

### **基线方法对比**
设计四个渐进式变体进行消融研究：

| 模型 | 描述 |
|------|------|
| **Base** | 标准 ReAct 框架，observe-think-act 循环 |
| **B-P** | Base + Adaptive Prompt Generation |
| **B-PT** | B-P + Context-Aware Tool Orchestration |
| **Jenius** | B-PT + Hierarchical Memory Management（完整版） |

所有模型共享相同架构与推理协议，仅组件不同，便于归因性能差异。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 执行保真度（Execution Fidelity）**

| 数据集 | 模型 | TCR ↑ | TFR ↓ | TIR ↓ | TPS ↑ |
|-------|------|------|------|------|------|
| **APIGen** | Base | 0.8150 | 0.1800 | 0.0050 | 0.8150 |
|          | B-P  | 0.8275 | 0.1675 | 0.0050 | 0.8275 |
|          | B-PT | 0.8375 | 0.1587 | 0.0038 | 0.8375 |
|          | **Jenius** | **0.8500** | **0.1362** | **0.0138** | **0.8500** |
| **Jenius-bench** | Base | 0.5659 | 0.0329 | 0.4012 | 0.5968 |
|                | B-P  | 0.7271 (+28.5%) | 0.0859 | 0.1871 | 0.7491 |
|                | B-PT | 0.7494 | 0.0718 | 0.1788 | 0.7740 |
|                | **Jenius** | **0.7647** | **0.0753** | **0.1600** | **0.7847** |

> 🔍 在简单任务（APIGen）上提升有限；但在复杂多轮任务（Jenius-bench）中，**TCR 提升近 35%**，显示模块组合的有效性。

---

#### **(2) 输出质量（CRCFF）——仅在 Jenius-bench 上评估**

| Evaluato r | Model | Correctness ↑ | Relevance ↑ | Completeness ↑ | Fluency ↑ | Faithfulness ↑ |
|----------|-------|---------------|-------------|----------------|-----------|----------------|
| **Qwen** | Base | 0.6741 | 0.8951 | 0.7722 | 0.9294 | 0.7919 |
|          | Jenius | **0.7580** | **0.9447** | **0.8088** | **0.9771** | **0.8766** |
| **DeepSeek** | Base | 0.7890 | 0.9245 | 0.7898 | 0.9546 | 0.8291 |
|              | Jenius | **0.8350** | **0.9400** | **0.8143** | **0.9686** | **0.8636** |

> ✅ 平均提升 **8–10%**，尤其在 Correctness 和 Fluency 上显著改善。

---

#### **(3) Token 消耗分析（效率）**

| 数据集 | Base | B-PT | Jenius |
|-------|------|------|--------|
| **APIGen** | 9.96M | 2.41M | 2.46M |
| **Jenius-bench** | 9.27M | — | **3.65M** |

> 📉 **总token消耗下降超60%**，得益于：
- 自适应提示减少冗余推理
- 工具检索避免无效调用
- 分层记忆压缩历史上下文

---

### **消融实验结果**
从 Base 到 Jenius 的逐步增强表明：
- **B-P（加入Prompt优化）**：TCR 提升最大（+16.12%），说明意图理解和提示对齐至关重要。
- **B-PT（加入Tool检索）**：进一步提升精度，同时降低TIR。
- **Jenius（加入Memory管理）**：虽TCR增幅较小，但显著降低token消耗，支持长期交互稳定性。

> ✅ 三者协同作用明显，缺一不可。

---

## **4. 关键结论和发现**

### **主要发现**
1. **经验驱动的设计优于静态配置**：动态适配提示、工具和记忆，能显著提升真实任务下的准确性与鲁棒性。
2. **上下文感知是工具调用的关键**：语义检索 + 拐点过滤机制有效应对高噪声、多候选场景。
3. **结构化记忆优于简单截断**：分层摘要既控制长度又保留关键语义，维持长程推理一致性。
4. **模块化设计可扩展性强**：支持MCP等开放协议，易于集成第三方工具和服务。
5. **真实部署验证有效性**：已在 [Jenius](https://www.jenius.cn) 上线，日活用户持续增长，具备工业级可用性。

---

### **方法的局限性**
1. **依赖高质量工具元数据**：若工具描述不完整或语义模糊，会影响检索效果。
2. **拐点检测对分布敏感**：极端相似度分布可能导致阈值判断不准。
3. **评估仍偏重结构化任务**：对开放式创造性任务的支持尚待加强。
4. **未完全解决多智能体协作问题**：当前聚焦单agent优化。

---

### **未来工作方向**
1. **更灵活的结果导向评估**：允许多条合法执行路径，引入用户满意度、决策成本、延迟等现实指标。
2. **动态模块重组机制**：根据任务复杂度自动启用/关闭某些模块，实现资源自适应分配。
3. **多智能体协同架构**：探索分布式问题求解与角色分工。
4. **更强的解释性与可调试性**：提供中间推理链可视化，增强用户信任。
5. **轻量化部署方案**：面向边缘设备或低资源场景的压缩版本。

---

> 💡 **总结**：  
> **Jenius-Agent** 通过 **自适应提示生成、上下文感知工具编排、分层记忆管理** 三大创新，在真实世界复杂任务中实现了 **约20%的任务准确率提升**，同时大幅降低 **token消耗、响应延迟和调用失败率**。其已在实际产品中部署，为构建高效、可靠、可扩展的 LLM-based Autonomous Agent 提供了一个强有力的实践范本。

</details>

---

### 4. [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780)

**Authors**: Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, Gang Xie, Hailin Zhang, Hanglong Lv, Hanyu Li, Heyu Chen, Hongshen Xu, Houbin Zhang, Huaqiu Liu, Jiangshan Duo, Jianyu Wei, Jiebao Xiao, Jinhao Dong, Jun Shi, Junhao Hu, Kainan Bao, Kang Zhou, Lei Li, Liang Zhao, Linghao Zhang, Peidian Li, Qianli Chen, Shaohui Liu, Shihua Yu, Shijie Cao, Shimao Chen, Shouqiu Yu, Shuo Liu, Tianling Zhou, Weijiang Su, Weikun Wang, Wenhan Ma, Xiangwei Deng, Bohan Mao, Bowen Ye, Can Cai, Chenghua Wang, Chengxuan Zhu, Chong Ma, Chun Chen, Chunan Li, Dawei Zhu, Deshan Xiao, Dong Zhang, Duo Zhang, Fangyue Liu, Feiyu Yang, Fengyuan Shi, Guoan Wang, Hao Tian, Hao Wu, Heng Qu, Hongfei Yi, Hongxu An, Hongyi Guan, Xing Zhang, Yifan Song, Yihan Yan, Yihao Zhao, Yingchun Lai, Yizhao Gao, Yu Cheng, Yuanyuan Tian, Yudong Wang, Zhen Tang, Zhengju Tang, Zhengtao Wen, Zhichao Song, Zhixian Zheng, Zihan Jiang, Jian Wen, Jiarui Sun, Jiawei Li, Jinlong Xue, Jun Xia, Kai Fang, Menghang Zhu, Nuo Chen, Qian Tu, Qihao Zhang, Qiying Wang, Rang Li, Rui Ma, Shaolei Zhang, Shengfan Wang, Shicheng Li, Shuhao Gu, Shuhuai Ren, Sirui Deng, Tao Guo, Tianyang Lu, Weiji Zhuang, Weikang Zhang, Weimin Xiong, Wenshan Huang, Wenyu Yang, Xin Zhang, Xing Yong, Xu Wang, Xueyang Xie, Yilin Jiang, Yixin Yang, Yongzhe He, Yu Tu, Yuanliang Dong, Yuchen Liu, Yue Ma, Yue Yu, Yuxing Xiang, Zhaojun Huang, Zhenru Lin, Zhipeng Xu, Zhiyang Chen, Zhonghua Deng, Zihan Zhang, Zihao Yue  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2601.02780v1  

#### Abstract
We present MiMo-V2-Flash, a Mixture-of-Experts (MoE) model with 309B total parameters and 15B active parameters, designed for fast, strong reasoning and agentic capabilities. MiMo-V2-Flash adopts a hybrid attention architecture that interleaves Sliding Window Attention (SWA) with global attention, w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MiMo-V2-Flash 技术报告核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该研究旨在解决当前大型语言模型（LLM）在**长上下文建模**中面临的“强推理能力”与“高效推理速度”之间的根本矛盾。具体挑战包括：
- **计算瓶颈**：全注意力机制（Full Attention）在长序列下具有 $O(n^2)$ 的计算复杂度，导致KV缓存和计算开销巨大。
- **后训练效率低下**：传统的后训练范式（如SFT、RLHF）存在“跷跷板效应”（capability imbalance），即提升某一项能力可能导致其他能力退化。
- **推理延迟高**：标准自回归解码限制了推理吞吐量，尤其是在强化学习（RL）rollout等高成本场景。

### 提出的新方法与创新思路
MiMo-V2-Flash 提出了一个集成多项技术创新的高效模型架构与训练范式：

#### （1）混合滑动窗口注意力架构（Hybrid Sliding Window Attention）
- **设计**：采用 **5:1 的局部-全局注意力比例**，即每5层滑动窗口注意力（SWA）后接1层全局注意力（GA），形成交替结构。
- **参数**：滑动窗口大小为 **128 tokens**，远小于常规设置。
- **关键技术**：引入 **可学习的Attention Sink偏置**（learnable attention sink bias），允许模型动态地忽略不相关的token，从而在极小的窗口下仍能保持强大的长程依赖建模能力。

#### （2）轻量级多令牌预测模块（Lightweight MTP）
- **设计**：在主干MoE模型上附加一个由 **3层轻量级MTP块** 组成的模块。
- **轻量化实现**：MTP块使用**密集FFN**而非MoE，并仅采用**SWA**而非GA，使其成为高效的“草稿模型”（draft model）。
- **双重用途**：
  - **训练增益**：作为预训练目标，提升模型质量。
  - **推理加速**：用于**投机性解码**（speculative decoding），显著提升解码速度。

#### （3）多教师在线策略蒸馏（Multi-Teacher On-Policy Distillation, MOPD）
- **三阶段范式**：
  1. **通用SFT**：建立基础指令遵循能力。
  2. **领域专用RL/SFT**：训练多个**领域专家教师模型**（如数学、编码、安全对齐等）。
  3. **MOPD蒸馏**：学生模型通过**在线策略学习**，从教师模型获取**密集的、token级别的奖励信号**（基于KL散度）和最终结果奖励。
- **优势**：
  - **避免能力失衡**：同时吸收多个教师的峰值能力，无传统“取舍”问题。
  - **高效稳定**：在线策略学习避免了离线数据集带来的暴露偏差（exposure bias）。
  - **可迭代进化**：蒸馏后的学生可反哺成为更强的教师，形成正向循环。

### 相比现有方法的优势
- **性能对标顶级闭源模型**：尽管总参数量仅为DeepSeek-V3.2和Kimi-K2的1/2和1/3，其推理和智能体能力与之相当。
- **长上下文鲁棒性强**：在32K至256K的极端长上下文任务（如GSM-Infinite）中表现优异，性能衰减极小。
- **推理速度快**：利用MTP进行投机解码，实现了最高 **2.6倍的解码加速**。
- **开源友好**：公开了模型权重和MTP权重，促进社区研究。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **预训练**：27万亿tokens的多样化高质量语料，涵盖网页、书籍、学术论文、代码、STEM等领域，特别强调长距离依赖数据（如完整的GitHub仓库、PR、issues）。
- **评估基准**：
  - **通用能力**：MMLU, BBH, TriviaQA, DROP, ARC, HellaSwag, WinoGrande。
  - **数学推理**：GSM8K, MATH, AIME2024/25, HMMT Feb.2025。
  - **代码能力**：HumanEval+, MBPP+, CRUXEval, LiveCodeBench, **SWE-Bench** (Verified & Multilingual)。
  - **长上下文**：LongBench V2, MRCR, GSM-Infinite, NIAH-Multi。
  - **智能体能力**：BrowseComp, t2-Bench, Terminal-Bench。
  - **多语言**：GlobalMMLU, INCLUDE, SWE-Bench Multilingual。

### 实验设置和评估指标
- **模型规模**：总参数 **309B**，激活参数 **15B**（稀疏MoE，每token激活8/256个专家）。
- **上下文长度**：原生支持 **32K**，扩展至 **256K**。
- **训练精度**：FP8混合精度训练。
- **主要评估指标**：
  - **准确率**（Accuracy）：用于分类、数学、代码修复等任务。
  - **接受长度**（Acceptance Length）：衡量MTP投机解码效率。
  - **解码速度提升**（Decoding Speedup）：与无MTP的基线相比。
  - **解决率**（Resolved Rate）：用于SWE-Bench等复杂任务。

### 基线方法对比
- **开源模型**：DeepSeek-V3.2-Thinking, Kimi-K2-Thinking, DeepSeek-V3.2-Exp-Base, Kimi-K2-Base。
- **闭源模型**：Claude Sonnet 4.5, GPT-5 (High), Gemini 3.0 Pro。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **SWE-Bench Verified**：**73.4%**，超越所有开源模型，接近GPT-5 High（74.9%）。
- **SWE-Bench Multilingual**：**71.7%**，确立了在多语言软件工程任务上的领先地位。
- **MTP加速效果**：
  - 平均**接受长度**达 **3.6 tokens**。
  - 实现最高 **2.6倍的解码速度提升**（见Table 10）。
- **长上下文检索**：在32K-256K长度下，NIAH-Multi任务成功率接近 **100%**。
- **数学推理**：AIME 2025得分为 **94.1%**，与顶尖模型水平相当。

### 与基线方法的对比结果
- **性能对标**：在大多数推理基准上，性能与 **Kimi-K2-Thinking** 和 **DeepSeek-V3.2-Thinking** 相当或更优。
- **长上下文优势**：在 **LongBench V2** 和 **MRCR** 上，超越了更大的全注意力模型，验证了混合SWA架构的有效性。
- **参数效率**：以约一半（vs DeepSeek-V3.2）甚至三分之一（vs Kimi-K2）的参数量，达到了相近的智能体和推理能力。

### 消融实验结果
- **Attention Sink的作用**（Table 2）：
  - 无Sink的Hybrid SWA（W=128）在MMLU上比全GA基线低2.4分。
  - 加入Sink后，性能全面反超，证明了其对小窗口SWA的关键作用。
- **滑动窗口大小的影响**（Table 3）：
  - W=128 在长上下文任务（GSM-Infinite, NoLiMa）上显著优于 W=512，表明过大的窗口会模糊局部与全局的分工，导致次优性能。
- **MOPD的有效性**（Table 7, Figure 6）：
  - MOPD成功继承了最强教师的能力，例如在SWE-Bench Verified上，学生模型（73.4%）几乎完全掌握了教师模型（74.2%）的能力。
  - 相比纯ORM的RL训练，MOPD在收敛速度和稳定性上表现更佳。

---

## 4. 关键结论和发现

### 主要发现
1. **小窗口SWA + Attention Sink是可行且高效的**：128-token的极小滑动窗口，在可学习sink的辅助下，不仅能匹配，甚至能在长上下文任务上超越全注意力模型，这得益于更好的正则化和清晰的“局部-全局”功能划分。
2. **MOPD范式解决了后训练的“能力失衡”难题**：通过从多个领域专家教师处获取token级监督，学生模型能够无损地融合各项专精能力，实现了能力的“叠加”而非“取舍”。
3. **轻量级MTP是实现高效推理的关键**：将MTP作为投机解码的草稿模型，能有效提升解码吞吐量，尤其在batch size受限的RL训练中价值巨大。
4. **大规模智能体RL训练具有强泛化性**：在代码智能体任务上的大规模RL训练，其学到的技能能有效迁移到数学、搜索等其他复杂任务上。

### 方法的局限性
- **知识容量受限**：由于参数量相对较小，在纯粹的知识问答任务（如SimpleQA）上，其知识容量低于更大的模型。
- **架构探索初步**：论文承认其对混合注意力架构的设计空间探索尚属初步，未进行更系统的权衡分析。
- **与顶尖闭源模型仍有差距**：尽管性能强劲，但与最强的闭源模型（如GPT-5 High）相比，仍存在一定差距。

### 未来工作方向
- **扩大模型规模**：计划通过增加模型尺寸和训练算力来缩小与顶级闭源模型的差距。
- **设计更优的智能体导向架构**：专注于开发更健壮、更高效的、面向智能体任务的模型架构。
- **扩展MOPD的迭代共进化**：规模化教师与学生的迭代共进化循环，以充分发挥MOPD范式的潜力。

</details>

---

### 5. [Falcon-H1R: Pushing the Reasoning Frontiers with a Hybrid Model for Efficient Test-Time Scaling](https://arxiv.org/abs/2601.02346)

**Authors**: Falcon LLM Team, Iheb Chaabane, Puneesh Khanna, Suhail Mohmad, Slim Frikha, Shi Hu, Abdalgader Abubaker, Reda Alami, Mikhail Lubinets, Mohamed El Amine Seddik, Hakim Hacid  
**Category**: cs.AI  
**Published**: 2026-01-07  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2601.02346v1  

#### Abstract
This work introduces Falcon-H1R, a 7B-parameter reasoning-optimized model that establishes the feasibility of achieving competitive reasoning performance with small language models (SLMs). Falcon-H1R stands out for its parameter efficiency, consistently matching or outperforming SOTA reasoning model...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Falcon-H1R 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**大型语言模型（LLMs）在复杂推理任务中推理效率低下**的问题。随着模型规模的增长，训练成本急剧上升，而纯靠扩大模型参数来提升推理能力已接近瓶颈。与此同时，测试时扩展（Test-Time Scaling, TTS）方法虽然能显著提升性能，但其高昂的推理开销限制了实际应用。

Falcon-H1R 提出了一种新的范式：通过**更小的模型 + 更高效的架构 + 针对性的训练策略**，实现与更大模型相当甚至超越的推理性能，同时大幅降低计算成本。

---

### 提出的新方法与思路

#### （1）Hybrid Architecture for Efficient Reasoning
- 基于 **Falcon-H1 架构**，采用 **Transformer-Mamba 混合架构**（Hybrid Transformer-SSM），结合了 Transformer 的强大表达能力和 Mamba 在长序列上的线性时间推理优势。
- 这种设计显著提升了**推理速度和内存效率**，尤其适合需要生成长 Chain-of-Thought（CoT）的并行推理场景。

#### （2）Robust Training Strategy
- **两阶段训练流程**：
  1. **Cold-start SFT**：在高质量、多样化的数学、代码、科学等领域的长推理轨迹数据上进行监督微调。
  2. **Reinforcement Learning with Verifiable Rewards (RLVR)**：使用 **GRPO** 算法进一步优化模型，目标是提升 `pass@1` 准确率，并控制输出质量（如长度、格式）。
- 数据处理强调“**难度感知加权**”（difficulty-aware weighting），即对更难的问题赋予更高权重，避免过拟合简单样本。

#### （3）Superior Efficiency via Test-Time Scaling (TTS)
- 将 Falcon-H1R 与最新的 TTS 方法 **DeepConf** 结合，动态剪枝低置信度的推理路径，实现早期停止。
- 利用模型良好的**置信度校准能力**，在保证高准确率的同时大幅减少 token 消耗。

---

### 相比现有方法的优势

| 维度 | Falcon-H1R 优势 |
|------|----------------|
| **参数效率** | 仅 7B 参数，性能媲美 8B–32B 甚至更大的 SOTA 推理模型（如 Qwen3-32B, GPT-OSS-20B） |
| **推理效率** | 混合架构带来更快的 inference 和更低的内存占用，支持高 batch size 下的高效并行推理 |
| **token 效率** | 在 DeepConf@512 设置下，相比同类模型减少高达 38% 的 token 使用量 |
| **准确性** | 在多个硬核推理基准上达到 SOTA 或第二名水平 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### SFT 阶段
- 多领域混合数据集，涵盖：
  - **数学**：MATH、AIME、AMO-Bench 等
  - **编程**：LiveCodeBench、SciCode、LeetCode 类题目
  - **科学**：GPQA-Diamond、STEM 问题
  - **其他**：指令遵循、工具调用、安全对话等

#### RL 阶段
- 严格去重且独立于 SFT 的高质量子集
- 主要聚焦 **数学与编程任务**
- 数据经过难度过滤（基于 SFT 模型 pass rate）
- 引入 **MATH-VERIFY** 和 **Sandbox-Fusion** 进行自动验证

---

### 实验设置与评估指标

#### 评估基准分类
| 类别 | 基准 |
|------|------|
| **数学推理** | AIME24, AIME25, HMMT25, AMO-Bench, MATH500 |
| **代码生成** | LiveCodeBench v6, SciCode, T2-Telecom, Terminal-Bench Hard |
| **通用推理** | GPQA-Diamond, MMLU-Pro, Humanity's Last Exam (HLE), IFBench |

#### 评估方式
- 所有结果报告为 **pass@1**
- 多数任务采样 16 条响应用于聚合分析（如 majority voting）
- 使用官方推荐的解码参数（temperature=0.6, top_p=0.95）

#### 测试时扩展（TTS）设置
- 方法：**DeepConf@512**
- 总 trace 数：512
- 初始 warm-up：16 条 trace
- 动态剪枝：基于滑动窗口（2048 tokens）内的最低 group confidence
- 回答提取：使用增强版 `math_verify` 解析器，优于简单的 `\boxed{}` 匹配

#### 基线对比模型
- **7B级**：Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B
- **14B级**：Phi-4-Reasoning-Plus-14B
- **15B级**：Apriel-1.5-15b-Thinker
- **20B级**：GPT-OSS-20B
- **32B级**：Qwen3-32B
- **47B级**：Nemotron-H-47B-Reasoning

---

## 3. 主要实验结果和性能指标

### 数学推理性能（Table 4）

| Model | AIME24 | AIME25 | HMMT25 | AMO-Bench | MATH500 |
|-------|--------|--------|--------|-----------|---------|
| **Falcon-H1R-7B** | **88.1** | **83.1** | **64.9** | **36.3** | **97.4** |
| Qwen3-32B | 79.4 | 71.0 | 49.8 | 21.3 | 96.8 |
| GPT-OSS-20B | 83.3 | 84.4 | 64.8 | 26.0 | 94.8 |
| Phi-4-R-Plus-14B | 77.2 | 71.2 | 47.7 | 15.0 | 95.4 |

> ✅ Falcon-H1R 在 AIME24、HMMT25、AMO-Bench 上均取得第一，在 AIME25 排名第二，全面超越多数大模型。

---

### 代码生成性能（Table 5）

| Model | LCB v6 | SciCode (sub/main) | T2-Telecom | TB Hard |
|-------|--------|--------------------|------------|---------|
| **Falcon-H1R-7B** | **68.6** | 28.3 / 3.9 | 25.4 | 4.9 |
| GPT-OSS-20B | 72.0 | 34.9 / 6.2 | 60.2* | 9.9* |
| Qwen3-32B | 61.0 | 36.4 / 9.2 | 29.8 | 2.8 |

> ✅ 在 LiveCodeBench 上仅次于 GPT-OSS-20B，远超同级别模型；在终端代理任务上表现稳健。

---

### 通用推理性能（Table 6）

| Model | GPQA-D | MMLU-Pro | HLE | IFBench |
|-------|--------|----------|-----|--------|
| **Falcon-H1R-7B** | 61.3 | 72.1 | **11.1** | **53.4** |
| Phi-4-R-Plus-14B | 67.9 | 79.2 | 5.9 | 51.7 |
| GPT-OSS-20B | 61.2 | 75.6 | 9.8 | 69.4 |

> ✅ 在 HLE 和 IFBench 上排名第二，显示强大的指令跟随和前沿推理能力；GPQA 和 MMLU-Pro 仍有提升空间。

---

### Test-Time Scaling 性能（Table 7, DeepConf@512）

| Model | AIME25 Acc↑ | AIME25 Tok↓ | AMO-Bench Acc↑ | AMO-Bench Tok↓ |
|-------|-------------|-------------|----------------|----------------|
| **Falcon-H1R-7B** | **96.7%** | **95.1M** | **35.9%** | **216.8M** |
| DS-R1-0528-Qwen3-8B | 82.8% | 174.5M | 25.6% | 487.9M |
| Qwen3-32B | 86.7% | 174.8M | 28.2% | 364.8M |

> 🔥 **关键突破**：在 AIME25 上达到 **96.7% 准确率**，同时 token 消耗比 DeepSeek-R1 降低 **38%**，体现了极高的 TTS 成本效益。

---

### 消融实验关键发现（Section 2.2 & 3.3）

| 实验维度 | 最优选择 | 发现 |
|--------|----------|------|
| **Learning Rate** | 1024×10⁻⁶ | 更大学习率带来更快收敛和更高下游性能 |
| **Rollout Multiplicity** | n=12 | 更多推理路径有助于学习复杂问题解决策略 |
| **Incorrect Rollouts** | 有限保留 | 对最难问题略有帮助，整体收益边际 |
| **Teacher Mixing** | 单教师 > 多教师 | 跨风格混合引入分布偏移，损害泛化 |
| **Data Weighting** | 难题加权（1.25–1.75×） | 显著提升整体性能，防止过拟合简单题 |
| **RL Curriculum** | Math-only → Code | 数学优先训练迁移效果最好 |

---

## 4. 关键结论和发现

### 主要发现

1. **Small Language Models (SLMs) can compete with LLMs in reasoning**
   - Falcon-H1R-7B 以仅 7B 参数，在多项硬核推理任务上超越 2× 至 7× 更大的模型，证明了**参数效率的可能性**。

2. **Careful data curation and training matter more than scale**
   - 高质量、多样化、难度感知的数据配合 SFT + RLVR 训练，是性能跃升的关键驱动力。
   - “数学技能可迁移性强”：数学主导的数据混合效果最佳。

3. **Hybrid architecture enables efficient TTS**
   - Falcon-H1R 的 Transformer-Mamba 架构在长序列、高 batch 场景下具有显著吞吐优势（Appendix B 图8显示 +20%~100% 吞吐提升）。

4. **Well-calibrated confidence supports aggressive pruning**
   - 模型具备良好的置信度估计能力，使得 DeepConf 可以有效剪枝弱路径，实现“高精度 + 低开销”的 TTS。

---

### 方法的局限性

1. **知识密集型任务仍有差距**
   - 在 GPQA-Diamond 和 MMLU-Pro 上未达顶尖水平，表明其训练侧重推理而非广博知识记忆。

2. **依赖高质量奖励信号**
   - RL 阶段严重依赖可验证的答案（如数学、代码执行），对于开放域或主观判断任务适用性受限。

3. **CoT 安全性需谨慎管理**
   - 安全评估（Appendix E）显示，Chain-of-Thought 中的安全违规率高于最终答案（92.6% vs 98.2%），说明推理过程可能暴露敏感内容，部署时建议隐藏原始 CoT。

---

### 未来工作方向

- 探索更小规模模型（<7B）是否也能实现强推理能力
- 设计更适合 TTS 的新型架构（如更深的 Mamba 层、动态路由机制）
- 提升模型在知识密集型任务中的表现
- 开发无需人工标注 reward 的自洽强化学习框架
- 将该范式推广至多模态推理系统

---

> 📌 **总结一句话**：  
> **Falcon-H1R 证明了“小模型 + 好数据 + 强训练 + 高效架构”可以打破“越大越好”的推理模型发展惯性，在准确性、token 效率和推理速度三个维度共同推进 reasoning frontier。**

</details>

---

### 6. [RadioDiff-Flux: Efficient Radio Map Construction via Generative Denoise Diffusion Model Trajectory Midpoint Reuse](https://arxiv.org/abs/2601.02790)

**Authors**: Xiucheng Wang, Peilin Zheng, Honggang Jia, Nan Cheng, Ruijin Sun, Conghao Zhou, Xuemin Shen  
**Category**: cs.LG  
**Published**: 2026-01-07  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2601.02790v1  

#### Abstract
Accurate radio map (RM) construction is essential to enabling environment-aware and adaptive wireless communication. However, in future 6G scenarios characterized by high-speed network entities and fast-changing environments, it is very challenging to meet real-time requirements. Although generative...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RadioDiff-Flux: Efficient Radio Map Construction via Generative Denoise Diffusion Model Trajectory Midpoint Reuse

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在6G网络中，**动态环境下的实时Radio Map（RM）构建**面临巨大挑战。尽管生成式扩散模型（DMs）在RM构造中实现了高精度，但其迭代去噪过程导致推理延迟较高（通常为数秒），难以满足高速移动场景（如无人机、车联网）对低延迟更新的需求。此外，传统方法无法有效利用连续场景之间的时空相关性，造成大量冗余计算。

### 提出的新方法与创新思路
本文提出 **RadioDiff-Flux** ——一种基于**隐式扩散模型轨迹中点重用**（midpoint reuse）的两阶段高效RM构建框架，其核心思想是：

- **发现并验证了一个关键结构性规律**：在语义相似的场景下（如同一建筑布局但基站位置不同），扩散过程中的**中间潜在变量（latent midpoint）高度一致**，这些中点编码了稳定的环境语义（如建筑结构、材料属性），而后期才细化发射机特性和动态障碍物等细节。
- 基于此，设计了**两阶段解耦架构**：
  1. **第一阶段**：仅使用静态环境特征（如建筑物布局）训练一个扩散模型，生成可缓存的“粗略潜变量表示”（即midpoint）；
  2. **第二阶段**：复用该midpoint作为起点，结合动态条件（如车辆、BS位置）进行后续去噪，从而跳过前半段重复计算。

### 相比现有方法的优势
| 方法类型 | 局限性 | RadioDiff-Flux 的优势 |
|--------|------|---------------------|
| **物理驱动方法（如ray tracing）** | 计算复杂度随场景指数增长，轻微变化需重新全局计算 | 避免物理仿真，实现快速适应 |
| **判别式神经网络（如RadioUNet）** | 难以生成空间连贯的全局RM，泛化能力弱 | 利用生成模型保证高质量、结构完整输出 |
| **GAN类方法（如RME-GAN）** | 存在模式崩溃、训练不稳定问题 | 改用更稳定、质量更高的DM架构 |
| **标准扩散模型（如RadioDiff）** | 迭代推理耗时长（~600ms），无记忆机制 | **通过midpoint reuse减少有效去噪步数，显著加速推理**

> ✅ **核心优势总结**：在保持高保真度的前提下，实现**高达50倍以上的推理加速**，适用于动态、实时无线感知系统。

---

## 2. 核心实验方法和设置

### 数据集
- 使用公开数据集 **RadioMapSeer**，源自ICASSP 2023路径损耗预测挑战赛。
- 包含700张城市地图（500训练 + 200测试），每张图有80个发射机位置及对应的真实RM。
- 地图来源：OpenStreetMap（Ankara, Berlin, Glasgow等）
- 分辨率：256×256像素，1米/像素；包含建筑物、道路、随机车辆（用于DRM）。
- 载频：5.9 GHz，发射功率：23 dBm。

### 实验设置
- 所有模型均在 **NVIDIA A40 GPU** 上运行，使用PyTorch 2.2.0 + CUDA 11.8。
- 扩散步数 $ T = 100 $（采样阶段），训练策略与SOTA模型 **RadioDiff [17]** 对齐。
- **评估指标**：
  - **NMSE / RMSE**：衡量整体误差
  - **SSIM**：评估结构相似性（高频细节保留）
  - **PSNR (dB)**：信号重建质量
  - **Time (ms)**：单次推理时间（反映延迟）
  - 缓存开销：每个latent midpoint大小为64KB（float32）

### 基线方法对比
| 模型 | 类型 | 特点 |
|------|------|------|
| **RadioUNet** | Discriminative CNN | 经典U-Net结构，直接回归RM |
| **UVM-Net** | Sequence Model | 使用State Space Model捕捉长程依赖 |
| **RME-GAN** | GAN-based | 条件生成对抗网络，存在训练不稳定性 |
| **RadioDiff** | Diffusion Model (SOTA) | 当前最优生成式RM构造方法，作为主基准 |

### 自研方法
- **Vanilla Midpoint Reuse (Ours)**：直接复用预训练RadioDiff的中间潜变量，无需额外训练。
- **RadioDiff-Flux (Ours)**：两阶段框架，第一阶段单独训练静态环境扩散模型，第二阶段复用pre-trained RadioDiff完成剩余去噪。

### 实验场景设计（三类动态变化）
1. **Scenario 1**：改变BS位置（静态环境不变）
2. **Scenario 2**：从静态环境过渡到含动态车辆的环境（BS固定）
3. **Scenario 3**：直接修改静态环境（建筑布局+BS同时变）

引入 **Reuse Ratio $ R_{\text{reuse}} $** 控制复用比例（例如$ R_{\text{reuse}} = 0.9 $ 表示前90%步骤复用midpoint）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 📊 Scenario 1: 改变BS位置（表I）
| $ R_{\text{reuse}} $ | NMSE | Speedup vs RadioDiff |
|-----------------------|------|------------------------|
| 0.7 | 0.00671 (+15.7%) | **3.47×** |
| 0.9 | 0.01542 (+166%) | **9.5×** |
| 0.98 | 0.13098 (>20×) | **50×** |

> ⚠️ 注意：当$ R_{\text{reuse}} > 0.9 $时，精度下降明显（出现“模糊叠加”效应）

#### ✅ 引入RadioDiff-Flux后显著改善（表II）
| $ R_{\text{reuse}} $ | NMSE (Vanilla) → (Flux) | Improvement |
|------------------------|----------------------------|-------------|
| 0.98 | 0.13098 → **0.02957** | ↓77.4% error |
| 0.95 | 0.04271 → **0.01292** | ↓69.7% error |

> 💡 结论：**RadioDiff-Flux能有效缓解高复用比下的失真问题**

#### 📈 Scenario 2: 加入动态车辆（表III）
| $ R_{\text{reuse}} $ | NMSE | SSIM | Speedup |
|------------------------|------|------|---------|
| 0.98 | **0.00776**（仅比baseline高20.7%） | 0.9509 | **58.07×** |

> ✅ 即使在极高复用比下仍保持极佳全局准确性，适合动态元素插入场景。

#### ⚠️ Scenario 3: 修改静态环境（表IV）
| $ R_{\text{reuse}} $ | NMSE ($ R_{\text{reuse}}=0 $: 0.0068) | 可接受范围 |
|------------------------|----------------------------------------|------------|
| 0.7 | 0.00889 (+30.7%) | ✔️ 合理平衡 |
| 0.9 | 0.04694 (>6×) | ❌ 不推荐 |
| 0.98 | 0.58418 (>85×) | 完全失效 |

> 🔍 发现：静态环境发生根本性变化时，初始latent产生“惯性偏差”，难以纠正。

---

### 性能对比总览（最高加速与精度损失）
| 方法 | 最大Speedup | 对应Accuracy Loss | 适用场景 |
|------|--------------|--------------------|----------|
| Vanilla Midpoint Reuse | ~50× | <0.15% SSIM drop (at $ R_{\text{reuse}} \approx 0.98 $) | 小幅扰动（BS微调、加车） |
| RadioDiff-Flux | ~50× | 更小失真，尤其在高$ R_{\text{reuse}} $ | 高速切换、边缘部署 |
| RadioDiff (Baseline) | 1× | N/A | 精度优先 |

> ✅ **最终结论**：RadioDiff-Flux可在**小于0.15%精度损失下实现最高50倍加速**。

---

## 4. 关键结论和发现

### 主要发现
1. **扩散过程中间状态具有强语义一致性**：在同一静态环境下不同BS或动态配置生成的RM，在扩散中段（latent midpoint）表现出高度相似的潜变量分布（通过NMSE/SSIM/KL散度验证）。
2. **midpoint可安全复用**：理论分析表明，随着扩散步数增加，语义相近场景的KL散度呈二次衰减，支持跨场景复用。
3. **两阶段解耦设计大幅提升效率**：
   - 第一阶段建模静态环境，结果可缓存共享；
   - 第二阶段专注动态调整，避免重复早期计算。
4. **实际部署友好**：
   - 缓存开销小（单个midpoint约64KB）；
   - 推理加速使边缘设备实现实时RM更新成为可能。

### 方法的局限性
- **对静态环境剧变敏感**：若建筑布局发生重大变更，midpoint复用会导致严重误差（“惯性偏差”）。
- **依赖预训练模型一致性**：要求第一阶段与第二阶段模型在latent space上兼容。
- **未处理多BS联合建模**：当前为单BS设计，多BS需分别生成后叠加（虽可加速但仍非端到端优化）。

### 未来工作方向
1. **自适应 $ R_{\text{reuse}} $ 策略**：根据环境变化程度自动选择最优复用比例（如通过cross-attention空间距离判断）。
2. **增强时序一致性**：面向连续RM序列生成，提升帧间平滑性，减少抖动。
3. **扩展至3D与毫米波场景**：适配更复杂的传播环境。
4. **多BS联合生成架构**：将多个BS位置作为条件输入，一次性生成复合RM。

---

## 总结
✅ **RadioDiff-Flux 是首个将扩散模型“中点复用”思想应用于无线信道建图的工作**，通过解耦静态与动态建模，在保持SOTA生成质量的同时，实现了**数量级的推理加速（最高50×）**，为6G环境中实时、可扩展的环境感知通信提供了切实可行的技术路径。

</details>

---

### 7. [MixTTE: Multi-Level Mixture-of-Experts for Scalable and Adaptive Travel Time Estimation](https://arxiv.org/abs/2601.02943)

**Authors**: Wenzhao Jiang, Jindong Han, Ruiqian Han, Hao Liu  
**Category**: cs.LG  
**Published**: 2026-01-07  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2601.02943v1  

#### Abstract
Accurate Travel Time Estimation (TTE) is critical for ride-hailing platforms, where errors directly impact user experience and operational efficiency. While existing production systems excel at holistic route-level dependency modeling, they struggle to capture city-scale traffic dynamics and long-ta...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MixTTE: Multi-Level Mixture-of-Experts for Scalable and Adaptive Travel Time Estimation 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

本文针对当前工业级 **Travel Time Estimation (TTE)** 系统在大规模城市交通网络中面临的两大挑战：

1. **有限的感受野（Limited Reception Field）**：现有以路线为中心（route-centric）的模型（如 DiDi 的 WDR 架构）将路线视为孤立序列，难以捕捉跨路线的全局交通动态传播（如周边高速拥堵扩散至主干道）。
2. **长尾场景预测不准（Long-tail Underperformance）**：单一模型架构对罕见但关键的交通模式（如大型活动、施工区）泛化能力差，导致尾部场景误差高。

此外，直接集成链路级（link-level）建模面临三大挑战：
- **可扩展性**：百万级道路网络上的全局时空依赖建模效率低；
- **异质性**：多样化的交通模式难以统一建模；
- **动态适应性**：频繁更新模型成本高昂且易过拟合。

---

### **提出了什么新方法或新思路**

作者提出 **MIxTTE** —— 一种**可扩展且自适应的多层级混合专家框架**，通过模块化方式将链路级建模无缝集成到现有的路线级 TTE 系统中。其核心创新包括：

#### （1）Spatio-Temporal External Attention (STEA)
- 引入一组小型外部记忆单元（external memory units），通过 **cross-attention** 机制实现高效全局依赖建模。
- 复杂度为 $O(N \cdot U_{\text{ex}})$，远优于传统成对建模的 $O(N^2)$，适用于百万级道路网络。
- 结合**空间分层建模**（spatial hierarchical modeling），增强同时间步内的全局相关性感知。

#### （2）Externally Stabilized Graph Mixture-of-Experts (ESGMoE)
- 设计图结构 MoE 层，利用多个专家捕捉异构交通模式。
- 提出**熵驱动的分层路由机制**（entropy-based hierarchical routing），结合 STEA 提供的外部知识，提升路由稳定性。
- 引入**零计算专家**（Zero-computation Experts）：
  - Identity / Constant / Null Experts，用于摊销常见模式的计算开销，避免占用图专家资源。
  - 实现“稀疏激活 + 高效泛化”平衡，尤其利于长尾场景。

#### （3）Asynchronous Incremental Learning (ASIL)
- 提出异步增量学习策略，仅在检测到周期性分布偏移时选择性更新参数。
- 使用 **Mahalanobis Distance (MD)** 衡量链路表征偏移程度，基于异常链路比例决定是否触发更新。
- 冻结旧版链路模型生成漂移检测表征，确保潜在空间一致性。
- 支持高频适应（每小时更新），同时控制计算成本。

---

### **相比现有方法的优势**

| 维度 | MIxTTE 优势 |
|------|------------|
| **准确性** | 显著降低 MAE、MAPE 和 BCR，尤其在长尾场景表现更优 |
| **可扩展性** | STEA 实现线性复杂度，支持大规模路网建模 |
| **鲁棒性** | ESGMoE 有效处理异质与长尾交通模式 |
| **适应性** | ASIL 实现稳定高效的在线自适应，避免全模型重训 |
| **部署兼容性** | 无需重构现有系统，可插件式集成至工业生产环境 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

在滴滴出行平台的真实数据上进行实验，涵盖三个中国大城市：

| 城市 | 时间跨度 | 总行程数 | 平均持续时间 | 路段数 | 热门路段数 |
|------|----------|-----------|----------------|--------|--------------|
| 北京 | 2024-07-21 ~ 2024-09-05 | 22.28M | 18.11 min | 2.32M | 156.7K |
| 南京 | 2024-12-18 ~ 2025-02-25 | 7.95M | 13.79 min | 0.70M | 83.3K |
| 苏州 | 2025-01-01 ~ 2025-04-14 | 16.78M | 13.63 min | 1.42M | 149.2K |

> 注：定义“热门路段”以减少噪声并提高效率（见 Appendix B.1）。

---

### **实验设置和评估指标**

#### **训练设置**
- **全量重训练（Full Retraining）**：使用前110天数据训练，最后7天测试。
- **增量学习（Incremental Learning, IL）**：过去1小时完成行程作为训练集，下一小时出发行程作为测试集，模拟高频更新场景。

#### **评估指标**
| 指标 | 定义 |
|------|------|
| **MAE** | 预测与实际旅行时间的平均绝对误差（秒） |
| **MAPE** | 平均绝对百分比误差 |
| **BCR (Bad Case Ratio)** | MAE > 300 秒且 MAPE > 20% 的查询占比 |

---

### **基线方法对比**

#### 全量重训练基线：
1. **Rule-based**: RouteETA（历史平均加总）
2. **Route-centric**: HierETA, WDR（DiDi 当前系统）
3. **Link-centric**: CompactETA, ConSTGAT, BigST

#### 增量学习基线：
- **iETA**：DiDi 现有的增量学习框架

所有方法输入相同特征集，链路级模型训练流程一致，保证公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Table 2）**

#### 在增量学习设置下，MIxTTE 相较最优基线（iETA）的相对提升：

| 城市 | MAE ↓ | MAPE ↓ | BCR ↓ |
|------|-------|--------|--------|
| 北京 | **2.39%** | **3.70%** | **10.32%** |
| 南京 | 2.38% | 3.70% | 10.32% |
| 苏州 | 2.41% | 3.70% | 10.32% |

> 所有指标均为显著领先，尤其 **BCR 下降超 10%**，说明对极端错误案例有强抑制能力。

---

### **与基线方法的对比结果**

- MIxTTE 在所有城市、所有指标上均优于所有基线。
- 相比当前工业系统 **WDR** 和 **iETA**，MIxTTE 在长尾场景（如节假日、突发事件）增益更大（见 Figure 6）。
- 链路级方法（如 CompactETA, ConSTGAT）多数不如 WDR，表明单纯引入链路建模不足以超越精心设计的 route-centric 系统。
- **BigST** 在部分指标上优于 WDR，验证了链路级建模的价值，但仍不及 MIxTTE。

---

### **消融实验结果（Ablation Study）**

在南京和苏州数据集上验证各模块贡献：

| 变体 | 描述 | 影响 |
|------|------|------|
| **-WoEA** | 移除 STEA 与分层路由 | MAE 显著上升 → STEA 对上下文感知至关重要 |
| **-WoHR** | 移除分层路由 | 性能下降 → 外部知识引导提升路由稳定性 |
| **-WoMoE** | 移除 ESGMoE 层 | MAE 和 BCR 上升 → MoE 对异质模式建模有效 |
| **-WoZE** | 移除零计算专家 | BCR 明显恶化（尤其南京）→ 零计算专家对长尾泛化重要 |
| **-WoPU** | 移除参数选择性更新（即禁用 ASIL） | 训练时间和可训练参数大幅增加 → ASIL 显著提升效率 |

> 结果证明：**STEA、ESGMoE、ASIL 三大模块缺一不可**。

---

## 4. 关键结论和发现

### **主要发现**

1. **链路级建模必须与路线级系统协同设计**，而非简单替换。MIxTTE 通过模块化集成实现双赢。
2. **全局依赖可通过外部注意力高效建模**，STEA 在保持线性复杂度的同时显著提升上下文感知能力。
3. **MoE 是处理交通异质性的有效范式**，尤其是引入零计算专家后，可在不增加推理负担的前提下提升长尾性能。
4. **异步增量学习是工业系统的刚需**：ASIL 实现了“按需更新”，兼顾实时性与稳定性。
5. **线上 A/B 测试验证实用价值**：在真实滴滴平台上，MIxTTE 相比当前系统带来 **MAE 下降 1.24%~3.03%，BCR 下降 4.30%~4.41%**，节假日增益更明显。

---

### **方法的局限性**

1. **依赖高质量链路特征工程**：虽然 MIxTTE 不改变原有 pipeline，但其效果仍受底层特征质量制约。
2. **外部记忆单元容量有限**：$U_{\text{ex}}$ 过小会限制全局模式表达能力（见敏感性分析）。
3. **专家数量需手动设定**：目前未实现自动调节专家规模，可能影响不同城市的迁移能力。
4. **未考虑多模态外部因素**：如天气、事件等未显式建模（尽管作者在 Future Work 中提及）。

---

### **未来工作方向（Future Work）**

1. **构建多模态基础模型**：融合天气、社会事件等信息，提升模型鲁棒性。
2. **跨城市迁移学习**：利用外部记忆和 MoE 抽象通用交通知识，降低新城市部署成本。
3. **轻量化蒸馏技术**：将大规模跨城模型压缩，便于边缘部署。
4. **扩展至其他任务**：将 MixTTE 框架推广至 ETA、路径规划、订单调度等决策服务。

---

> ✅ **总结**：  
> **MIxTTE 是一个面向工业落地的创新框架**，它不是追求极致复杂的模型，而是通过 **STEA + ESGMoE + ASIL** 三者协同，在**精度、效率、稳定性、可扩展性**之间取得良好平衡，并已成功部署于滴滴出行平台，具有显著的实际应用价值。

</details>

---

### 8. [ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation](https://arxiv.org/abs/2601.02535)

**Authors**: Hyeong Kyu Choi, Sharon Li  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2601.02535v1  

#### Abstract
Selecting a single high-quality output from multiple stochastic generations remains a fundamental challenge for large language models (LLMs), particularly in open-ended tasks where no canonical answer exists. While Best-of-N and self-consistency methods show that aggregating multiple generations can...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在开放生成任务（如文本摘要、代码生成、数学推理）中面临一个根本挑战：如何从多个随机生成的结果中选择一个高质量输出。传统方法依赖外部评估器（如reward model）、精确字符串匹配投票（exact string-match voting）或迭代修正机制（如Self-Refine），这些方法存在以下问题：
- **封闭性限制**：多数仅适用于有标准答案的闭合任务（如多选题）；
- **高计算开销**：需要额外推理步骤或辅助模型；
- **效率低下**：无法有效推广到语义等价但字面不同的开放文本。

### 提出的新方法与思路
作者提出 **Mode Extraction (ModeX)** 和其轻量版 **ModeX-Lite**，一种无需外部评估器的 Best-of-N 选择框架，核心思想是将“多数投票”（majority voting）推广到开放文本生成场景。

#### ModeX 的三步流程：
1. **Adjacency Matrix Construction**  
   构建加权相似度图，节点为生成结果，边权重基于 n-gram（unigram, bigram, trigram）的 Jaccard 相似度。
   
2. **Spectral Graph Clustering**  
   利用图拉普拉斯矩阵的 Fiedler 向量进行谱聚类，递归地将候选输出划分为子群，直到割的导率（conductance）超过阈值 $ T = 0.8 $，保留最大连通簇。

3. **Centroid Selection**  
   在最终簇中选择加权度最高的节点作为“模态输出”（modal output），即最能代表群体共识的生成结果。

#### ModeX-Lite 的改进：
- 引入**早期剪枝机制**（early pruning）：在生成过程中每隔固定步数（默认 $ T=100 $）执行一次非递归谱聚类，并保留代表性路径；
- 显著降低计算开销，同时保持性能优势。

### 相比现有方法的优势
| 特性 | ModeX / ModeX-Lite | Self-Consistency | LLM Judge | Best-of-N (w/ RM) |
|------|---------------------|------------------|-----------|--------------------|
| 是否需外部评估器 | ❌（evaluator-free） | ❌ | ✅（需另一个LLM） | ✅（需reward model） |
| 是否支持开放生成 | ✅ | ❌（依赖exact match） | ✅ | ✅ |
| 计算效率 | 高（并行生成 + 图算法） | 中等（串行 refine） | 低（二次推理） | 低（需RM打分） |
| 可扩展性（随N增加） | 强 | 弱 | 弱 | 强但成本高 |

> ✅ **核心创新**：首次将谱聚类引入 LLM 多路径选择，实现无监督、结构感知的“语义共识”提取。

---

## 2. 核心实验方法和设置

### 数据集
在三个典型的开放生成任务上验证方法有效性：
- **Text Summarization**: CNN/DailyMail（版本3.0.0，前300个测试样本）
- **Code Generation**: HumanEval（164个Python编程题）
- **Mathematical Reasoning**: Math-500（500道数学题，涵盖代数、几何等六领域，取前300样例）

### 模型
- 主要使用：`Qwen2.5-7b-instruct` 和 `Llama3.1-8b-instruct`
- 代码生成任务使用：`CodeLlama-7b-instruct`

### 评估指标
| 任务 | 主要指标 |
|------|---------|
| 文本摘要 | ROUGE-1, ROUGE-2, ROUGE-L, BLEU |
| 代码生成 | Pass@1（功能正确率）, BLEU |
| 数学推理 | Accuracy（最终答案准确率） |

### 基线方法对比
1. **Single Path**：单次生成，报告16次运行均值±标准差
2. **Self-Refine**：迭代优化4轮
3. **LLM Judge (N=4/16)**：用另一个LLM选出最佳输出
4. **Perplexity**：选择困惑度最低的输出
5. **Self-Certainty**：选择负对数似然最小的输出
6. **Best-of-N (Gold Standard)**：使用reward model评分后选择最优（视为上限）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 Qwen 上的表现（部分突出项）：
| 方法 | Text Sum (ROUGE-L) | Code Gen (Pass@1) | Math Reasoning (Acc) |
|------|---------------------|-------------------|------------------------|
| Single Path | 20.17 | 69.89% | 70.98% |
| LLM Judge (N=16) | 19.72 | 65.24% | **74.67%** |
| Perplexity BoN | 21.06 | 73.17% | 78.00% |
| Self-Certainty BoN | 19.32 | 55.49% | 67.00% |
| **ModeX (N=16)** | **21.06** | **75.61%** | **78.00%** |
| **ModeX-Lite (N=16)** | **21.89** | **78.66%** | 75.33% |

> 💡 ModeX-Lite 在代码生成上超越所有基线，甚至接近或优于依赖 reward model 的 Best-of-N。

#### 在 Llama 上的表现趋势一致：
- ModeX-Lite 在文本摘要上达到 **ROUGE-L=22.80**，显著高于 Single Path（21.30）
- 数学推理从 38.75% 提升至 **45.33%**（ModeX-Lite, N=16）

### 与基线方法的对比结果
- **优于 LLM Judge**：尽管后者使用更强的判断模型，但 ModeX 更稳定且不受“judge bias”影响。
- **远超 Self-Certainty**：说明仅靠内部置信度不足以捕捉语义一致性。
- **媲美甚至超越 Gold Standard Best-of-N**：在某些任务（如Qwen代码生成）中，ModeX-Lite 超过 reward model 指导的选择。

### 消融实验与敏感性分析（Section 5.1 & Table 2）

#### 不同剪枝频率 $ T $
- 在 $ T \in \{100, 200, ..., 500\} $ 范围内，ModeX-Lite 性能稳定，始终优于 Single Path。
- 表明早期剪枝策略鲁棒性强。

#### 导率阈值 $ T $ 敏感性
- 在 $ T = 0.5 \sim 0.8 $ 区间内性能波动小，说明聚类终止条件设计合理。

#### 相似度函数比较（Appendix E）
| 方法 | ROUGE-L (Text Sum) | Pass@1 (Code) |
|------|---------------------|---------------|
| ModeX-n-gram（本文） | **21.06** | **75.61%** |
| ModeX-cosine（embedding-based） | 20.26 | 75.00% |
> 使用 n-gram Jaccard 比 cosine embedding 更有效，可能因更贴近表面语义匹配。

#### 计算复杂度与延迟（Table 2）
| 方法 | 复杂度 | 延迟（秒） |
|------|--------|------------|
| Single Path | O(L) | 5.5s |
| Self-Refine | O(kL) | 31.7s ⛔️ |
| LLM Judge | O(NL + NL_judge) | 10.7s |
| **ModeX-Lite (N=4)** | O(NL + N²) | **7.2s** |
| **ModeX-Lite (N=16)** | O(NL + N²) | **9.1s** |

> ✅ ModeX-Lite 实现了 **3.5倍于 Self-Refine 的加速**，且延迟仅略高于单路径生成。

---

## 4. 关键结论和发现

### 主要发现
1. **“模态输出”是高质量生成的关键**：  
   高质量生成倾向于形成语义上的密集簇，而幻觉或错误输出往往是离群点。通过识别“主导语义共识”，可有效筛选优质结果。

2. **结构感知选择优于简单打分机制**：  
   即使没有 reward model，利用生成之间的关系结构（relation structure）也能实现更优选择。

3. **更多路径 ≠ 更好结果，关键是选择机制**：  
   如 LLM Judge 随 N 增加收益有限；而 ModeX-Lite 在数学推理任务上 N 从4增至16带来 +7.33% 提升，显示其良好的可扩展性。

4. **早期剪枝可行且高效**：  
   图4显示，在生成完成50%时即可区分优劣路径，支持 ModeX-Lite 的设计合理性。

### 方法的局限性
1. **依赖词法相似度（lexical similarity）**：  
   使用 Jaccard n-gram 可能忽略有效但措辞差异大的 paraphrase，导致误判。

2. **假设“多数即正确”**：  
   若 LLM 存在系统性偏见或模式崩溃（mode collapse），可能会错误强化错误共识。

3. **不适用于极短输出或高度发散任务**：  
   当所有生成差异极大时，谱聚类难以形成有意义簇。

### 未来工作方向
- 探索基于 embedding 的动态相似度函数（如SBERT、BGE）以提升语义理解能力；
- 将 ModeX 与推理时搜索（reasoning-time search）结合，进一步提升复杂任务表现；
- 扩展至多模态生成任务中的“模态选择”；
- 研究如何检测和缓解“错误共识”问题。

---

> 🔗 **代码开源地址**：https://github.com/deeplearning-wisc/ModeX  
> 📌 **一句话总结**：ModeX 提出了一种无需外部评估器的 Best-of-N 选择机制，通过谱聚类识别语义共识中的“模态输出”，在开放生成任务中实现了高效、稳健且优于主流基线的性能。

</details>

---

### 9. [Punctuation-aware Hybrid Trainable Sparse Attention for Large Language Models](https://arxiv.org/abs/2601.02819)

**Authors**: Junxiang Qiu, Shuo Wang, Zhengsu Chen, Hengheng Zhang, Jinda Lu, Changcheng Li, Qi Tian  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2601.02819v1  

#### Abstract
Attention serves as the fundamental mechanism for long-context modeling in large language models (LLMs), yet dense attention becomes structurally prohibitive for long sequences due to its quadratic complexity. Consequently, sparse attention has received increasing attention as a scalable alternative...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Punctuation-aware Hybrid Trainable Sparse Attention for Large Language Models

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLMs）中的 **dense attention** 机制在处理长序列时面临 **计算复杂度为 $O(L^2)$** 的瓶颈，导致推理延迟和内存占用急剧上升。虽然 **sparse attention** 被提出作为解决方案，但现有方法存在两个关键缺陷：

1. **粗粒度语义聚合导致信息丢失**：多数方法将连续token聚合成一个代表向量（如平均池化），模糊了块内语义边界，稀释关键实体或逻辑枢纽的信息。
2. **极端稀疏场景下性能严重下降**：当激活token比例极低时（如 < 5%），模型难以保留关键信息，限制了其在资源受限设备上的部署潜力。

---

### **提出了什么新方法或新思路**

本文提出 **Punctuation-aware Hybrid Sparse Attention (PHSA)**，一种原生可训练的稀疏注意力框架，核心创新如下：

#### ✅ 创新点一：标点感知的双分支聚合机制（Dual-branch Aggregation）
- 利用 **punctuation tokens**（如逗号、句号）作为自然的语义边界锚点。
- 对每个 key block 构造两种语义表示：
  - **Global Semantic Representation (Mo)**：对块内所有token进行平均池化，捕捉整体语义。
  - **Punctuation-aware Representation (Mp)**：仅对块内的标点token进行池化，强调语义转折点。
- 通过加权融合：  
  $$
  M(B_t) = \lambda \cdot M_o(B_t) + (1-\lambda) \cdot M_p(B_t)
  $$
  在几乎不增加计算开销的前提下，显著提升 Top-K 块选择的准确性。

#### ✅ 创新点二：面向极端稀疏的自适应训练与推理策略
- 设计了一套支持 **极低激活token比率**（如 Top-K=2）的训练-推理一致框架。
- 引入优先级机制：始终保留初始块（init block）和局部窗口（local window），确保基础上下文完整性。
- 支持从高稀疏度训练迁移到更低稀疏度推理，增强鲁棒性。

---

### **相比现有方法的优势**

| 维度 | PHSA优势 |
|------|---------|
| **语义精度** | 显著减少因平均池化造成的语义稀释，保留关键边界信息 |
| **极端稀疏表现** | 在极低Top-K下仍保持稳定性能，优于InfLLM v2等SOTA方法 |
| **通用性** | 在训练型与非训练型范式下均有效，兼容不同模型规模（0.6B / 8B） |
| **效率** | 推理复杂度降至 $O(L)$，适合长文本场景（32k+ tokens） |

---

## 2. 核心实验方法和设置

### **使用的数据集**

| 类别 | 数据集 |
|------|--------|
| **训练数据** | DCLM, MAP-CC, UltraChat, TuluV3, FineMath, MegaMath 及高质量私有教育类数据 |
| **通用评测基准** | GSM8K, MATH, MathQA, MMLU, CMMLU, HellaSwag, HumanEval, BBH, LAMBADA, XStoryCloze, C-Valid, PiPA 等共15项任务 |
| **长上下文评测** | 
| - **NIAH (Needle-in-a-Haystack)** | 测试长距离信息检索能力，直接反映信息损失边界 |
| - **LongBench** | 多语言、多任务长文本理解基准，包含单轮问答、少样本学习、摘要生成、代码任务四大模块 |

---

### **实验设置**

| 参数 | 设置说明 |
|------|----------|
| **模型基础** | Qwen3-0.6B-Base 和 Qwen3-8B |
| **序列长度** | 4k（通用任务）、32k（长上下文） |
| **块大小（block size）** | 16 tokens |
| **初始化块 & 局部窗口** | 4k场景：16 + 128；32k场景：128 + 512 |
| **Top-K范围** | 1~16（训练与推理分别测试） |
| **训练token量** | 20B 和 100B |
| **门控参数 $\lambda$** | 固定为0.5（平衡全局与标点表示） |

---

### **评估指标**

| 指标 | 含义 |
|------|------|
| **NIAH Score** | Needle-in-a-Haystack 准确率，越高表示信息保留越好 |
| **Average Score on General Benchmarks** | 多任务平均得分，衡量综合能力 |
| **LongBench Average** | 长文本任务平均得分 |
| **Sparsity Ratio** | 激活token占比，用于衡量效率 |

---

### **基线方法对比**

| 方法 | 类型 | 说明 |
|------|------|------|
| **Dense Attention** | 密集注意力 | 原始全连接注意力，$O(L^2)$ 复杂度 |
| **InfLLM v2 (Zhao et al., 2025)** | Trainable Sparse | 当前SOTA可训练稀疏注意力，支持短到长自适应切换 |
| **NSA (Yuan et al., 2025)** | Trainable Sparse | DeepSeek提出的分层动态稀疏策略 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔹 在 **Qwen3-0.6B, 32k 序列长度** 下的表现：
- **Sparsity Ratio 达 97.3%**（即仅激活约 2.7% 的token）
- **信息损失减少 10.8%** 相较于 InfLLM v2
- NIAH 得分达 **99.0**（vs. InfLLM v2 的 98.0）

> 表明 PHSA 在极端稀疏条件下仍能高效保留关键信息。

#### 🔹 通用基准性能（100B训练tokens）：

| 方法 | Inference Top-K | 平均得分 |
|------|------------------|----------|
| Dense | N/A | 48.96 |
| InfLLM v2 | 2 | 47.69 |
| **PHSA** | **2** | **48.81** |
| **PHSA** | **4** | **49.24** ✅（最优） |

✅ **PHSA 在 Top-K=4 时超越 dense attention**，实现“更稀疏 + 更强性能”。

#### 🔹 LongBench 总体表现：

| 方法 | Average Score |
|------|---------------|
| Dense | 27.86 |
| InfLLM v2 | 27.40 |
| **PHSA** | **28.13** ✅ |
| **PHSA_en+zh**（加入中文标点） | **28.15** ✅✅ |

PHSA 在大多数子任务上优于基线，尤其在代码相关任务（LCC, RBP）中领先明显。

#### 🔹 NIAH 对比（32k, Qwen3-8B）：

| 方法 | Top-K=1 | Top-K=2 | Top-K=4 |
|------|--------|--------|--------|
| InfLLM v2 | 64.6 | 85.8 | 98.0 |
| **PHSA** | **68.0** | **88.8** | **99.0** |

✅ 所有稀疏级别下全面领先，尤其在低Top-K时优势更大。

---

### **消融实验结果**

- **Training Top-K=2 是局部最优配置**：
  - 训练时 Top-K=2，在推理 Top-K=2 下取得最佳NIAH分数（~95），显著优于 Top-K=1 或 >4。
  - 过低（如1）会导致训练不稳定，过高则失去稀疏优势。

- **PHSA 在 dense-trained 模型上也有效**：
  - 即使未专门训练稀疏模式，PHSA 在 inference 阶段应用也能优于 InfLLM v2。

- **跨语言影响验证（RQ4）**：
  - 原始 PHSA 在中文任务（arc_c_zh, MQZ）表现略弱 → 归因于仅使用英文标点。
  - 加入中文标点训练后得到 **PHSA_en+zh**，MQZ 分数从 20.63 提升至 22.22，验证了标点语言适配的重要性。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **标点符号是天然的语义边界提示器**，可有效指导稀疏注意力中的块选择。
2. ✅ **双分支聚合机制** 在几乎零额外计算成本下显著提升 Top-K 选择质量。
3. ✅ **极端稀疏训练可行且有益**：PHSA 支持极低激活token比率下的稳定训练与推理。
4. ✅ **PHSA 不仅不牺牲性能，反而可能超越 dense attention**，特别是在充分训练后（100B tokens）。
5. ✅ **轻量化部署潜力巨大**：适用于边缘设备、移动端等资源受限场景。

---

### **局限性**

- **标点噪声问题**：并非所有标点都具有语义边界意义（如引号、括号），当前方法未做筛选，可能引入噪声。
- **语言依赖性**：性能受标点体系影响，需针对不同语言定制标点集合。
- **未探索最优标点子集**：哪些标点最有效？是否可以学习动态权重？尚未深入研究。

---

### **未来工作方向**

1. **构建更精细的标点过滤机制**，剔除无语义作用的符号（如 `"`、`(`）。
2. **设计可学习的标点重要性权重**，替代固定门控参数 $\lambda$。
3. **扩展至更多语言与书写系统**（如阿拉伯语、日语），验证普适性。
4. **结合硬件优化**，实现端到端的高效稀疏推理引擎。

---

> 📌 **总结一句话**：  
> **PHSA 通过“标点即语义锚点”的新颖视角，设计了高效、可训练、抗极端稀疏的注意力机制，在保持甚至超越 dense attention 性能的同时，大幅降低计算负担，为 LLM 的轻量化长上下文处理提供了新范式。**

</details>

---

### 10. [MedDialogRubrics: A Comprehensive Benchmark and Evaluation Framework for Multi-turn Medical Consultations in Large Language Models](https://arxiv.org/abs/2601.03023)

**Authors**: Lecheng Gong, Weimin Fang, Ting Yang, Dongjie Tao, Chunxiao Guo, Peng Wei, Bo Xie, Jinqun Guan, Zixiao Chen, Fang Shi, Jinjie Gu, Junwei Liu  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2601.03023v1  

#### Abstract
Medical conversational AI (AI) plays a pivotal role in the development of safer and more effective medical dialogue systems. However, existing benchmarks and evaluation frameworks for assessing the information-gathering and diagnostic reasoning abilities of medical large language models (LLMs) have ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*MedDialogRubrics: A Comprehensive Benchmark and Evaluation Framework for Multi-turn Medical Consultations in Large Language Models*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前医学大语言模型（LLMs）的评估主要依赖静态任务（如多项选择题 MedQA、摘要生成），无法有效衡量模型在**多轮医疗咨询中主动获取信息、动态推理和对话管理**的能力。此外，真实医疗对话数据因隐私法规（如 HIPAA/GDPR）稀缺，且现有基于 LLM 的患者模拟器易产生**幻觉（hallucinations）**，导致评估不可靠。

### 提出的新方法与创新点
本研究提出 **MedDialogRubrics**，一个全面的基准与评估框架，旨在解决上述挑战：

- **合成高质量、无隐私风险的患者病例**  
  采用多智能体系统（multi-agent system）从公开医学知识源（如百度健康百科、UpToDate）生成 5,200 个合成患者记录，无需访问真实电子病历（EHR），保障数据隐私。

- **构建抗幻觉的 Patient Agent**  
  将患者记录分解为“原子医学事实”（atomic medical facts），并设计**动态引导机制（guidance injection loop）**，实时检测并纠正模型输出中的矛盾，确保对话内部一致性和临床合理性。

- **基于循证医学（EBM）的细粒度评分标准（rubrics）生成**  
  利用 LLM 结合专家标注，从临床指南中提取超过 **60,000 条“必须提问项”（must-ask items）**，形成结构化、优先级明确的评估 rubrics，并通过拒绝采样（reject sampling）优化质量。

- **自动化、可扩展的 LLM-as-a-Judge 评估流程**  
  构建基于 LLM 的评分系统，结合多数投票（majority voting）等集成策略，实现对多轮对话轨迹的客观、高效打分，减少人工标注成本。

### 相比现有方法的优势
| 维度 | MedDialogRubrics 的优势 |
|------|------------------------|
| **数据隐私** | 完全合成数据，不依赖真实 EHR |
| **仿真可靠性** | 抗幻觉机制显著降低患者代理的行为不一致性 |
| **评估精细度** | 提供 >60k 细粒度 rubrics，覆盖诊断全过程 |
| **评估自动化** | 支持大规模、低成本自动评分，与人类专家判断高度对齐（Macro F1 > 76%） |
| **评估维度完整性** | 覆盖信息采集策略、推理逻辑、安全行为等多方面 |

---

## 2. 核心实验方法和设置

### 数据集
- **MedDialogRubrics 数据集**：包含 **5,200 个合成患者案例**，涵盖常见病、慢性病、精神疾病及急症。
- 每个案例配有：
  - 结构化患者记录（人口统计、病史、症状、检查结果）
  - 首要主诉（chief complaint）
  - 平均每个案例约 **12 条“必须提问”rubrics**

### 实验设置
- **Doctor Agent 模型**（被评测对象）：
  - 开源模型：`Qwen3-235B-A22B-Instruct-2507`, `DeepSeek-R1`
  - 商用模型：`GPT-5`, `Gemini-2.5-Pro`
- **Patient Agent**：基于 DeepSeek-V3 构建，受限于原子事实记忆，具备纠错能力。
- **对话流程**：
  - 最多进行 **12 轮交互**
  - Doctor Agent 可持续提问或以 “End Inquiry” 结束
- **评估方式**：LLM-as-a-Judge，使用三个高级 LLM（GPT-5, Gemini-2.5-Pro, DeepSeek-V3）组成评审团

### 评估指标
- **Rubric 匹配精度（Precision）**：医生提出的问题中有多少属于“必须提问”
- **召回率（Recall）**、**F1 分数**、**准确率（Accuracy）**
- **人类一致性分析**：使用 Macro F1 衡量 LLM 评分器与人类专家评分的一致性
- **消融实验**：验证 Patient Agent 各组件的有效性（如是否启用 Strict Adherence 或 Guidance Injection）

### 基线方法对比
本文未直接对比传统静态 QA 基准（如 MedQA），而是强调其框架相较于以下工作的进步：
- **HealthBench**：虽有 48k rubrics，但缺乏对抗患者幻觉的设计
- **MediQ / LLM-Mini-CEX**：缺少系统化的 rubric 构建流程和专家验证
- **AgentClinic**：侧重工具调用而非纯对话推理

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **最高表现模型**：`Gemini-2.5-Pro` 在第 9–10 轮达到峰值 **52% 的 rubric 匹配精度**
- **其他模型表现**：
  - `GPT-5`：增长缓慢，在第 13+ 轮接近 50%，表现出“晚熟型”特征
  - `DeepSeek-R1`：稳定提升，最终约 **40%**
  - `Qwen3-235B`：表现最弱，始终低于 **35%**
- 所有模型距离理想值（100%）仍有巨大差距，表明当前 LLM 在**战略信息采集**上存在严重缺陷

### 与基线方法的对比结果
- 相较于仅依赖 prompt engineering 的患者模拟器，MedDialogRubrics 的 Patient Agent 将**幻觉率从 0.129 降至 0.049**
- 自动评分系统与人类专家的 **Macro F1 达到 76%~79.6%**，其中：
  - **多数投票（Majority Voting）**：平衡性最佳（F1 ~77–79%）
  - **自由策略（Liberal Strategy）**：对 GPT-5 效果最好（F1 ~79.6%），适合高召回场景
  - **一致投票（Unanimous Voting）**：过于严格，导致召回下降明显

### 消融实验结果
| Patient Agent 配置 | 幻觉率 ↓ | 相关性 ↑ | 行为一致性 ↑ |
|--------------------|---------|----------|--------------|
| Basic（仅提示工程） | 0.129   | 0.92     | 0.689        |
| + Strict Adherence  | 0.076   | 0.951    | 1.000        |
| + Guidance Injection| **0.049** | **0.992**| **1.000**    |

👉 结论：**Strict Adherence + Guidance Injection** 显著提升了患者代理的真实性与稳定性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **当前 LLM 在多轮医疗咨询中普遍存在“询问不足”（inquiry deficit）问题**  
   即使增加上下文长度，多数模型也无法动态更新假设并提出最关键的下一步问题。

2. ✅ **对话管理架构是瓶颈，而非单纯的知识容量或指令微调**  
   实验显示，更大的模型不一定更优；`Gemini-2.5-Pro` 凭借更强的推理规划能力胜出，说明需要专门优化**对话策略模块**。

3. ✅ **MedDialogRubrics 成功捕捉到静态基准无法反映的行为差异**  
   例如 GPT-5 和 Qwen3 在静态 QA 中差距小，但在多轮互动中 rubric 覆盖率相差近 **20%**。

4. ✅ **自动化评估可行且可靠**  
   多 LLM 评审团 + 多数投票机制可在保证准确性的同时大幅降低成本，Macro F1 > 76% 支持其作为人类替代的合理性。

### 方法的局限性
- **合成数据仍可能偏离真实临床复杂性**：尽管经过专家审核，但仍难以完全模拟情绪波动、非理性表达等真实患者行为。
- **rubrics 依赖专家标注**：虽然自动化生成，但仍需大量临床专家参与筛选与反馈，限制扩展速度。
- **未涉及治疗建议或医患沟通情感维度**：目前聚焦于诊断前的信息收集阶段。

### 未来工作方向
- 扩展至**治疗决策、随访管理、心理支持**等后续环节
- 引入**多模态输入**（如影像报告、生命体征）
- 探索**强化学习驱动的 Doctor Agent**，优化主动提问策略
- 构建跨语言版本，推动全球适用性

---

> **总结一句话**：  
> MedDialogRubrics 是首个将**合成数据隐私保护、抗幻觉患者模拟、EBM 驱动 rubrics、自动化评分**整合于一体的多轮医疗对话评估框架，揭示了当前 LLM 在临床推理过程中的深层短板，为下一代 doctor agent 的研发提供了标准化、高保真的测试平台。

</details>

---

### 11. [ElecTwit: A Framework for Studying Persuasion in Multi-Agent Social Systems](https://arxiv.org/abs/2601.00994)

**Authors**: Michael Bao  
**Category**: cs.AI  
**Published**: 2026-01-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2601.00994v1  

#### Abstract
This paper introduces ElecTwit, a simulation framework designed to study persuasion within multi-agent systems, specifically emulating the interactions on social media platforms during a political election. By grounding our experiments in a realistic environment, we aimed to overcome the limitations...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《ElecTwit: A Framework for Studying Persuasion in Multi-Agent Social Systems》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决当前**多智能体系统（multi-agent systems）中关于说服力研究的现实性不足问题**。以往的研究多依赖于简化环境（如基于游戏的模拟，例如 *Among Us*），这些场景虽然可控，但难以反映真实社会媒体环境中复杂的交互动态，尤其是政治选举背景下的舆论形成、信任建立与说服行为。

此外，现有研究往往忽视了模型架构差异对说服策略选择的影响，也缺乏对**涌现行为**（emergent behaviors）如“回音室效应”（echo chambers）、“说服级联”（persuasion cascades）等现象的观察。

### 提出的新方法或新思路
作者提出了 **ElecTwit** ——一个基于现实社交媒体平台（类比 X/Twitter）的**多智能体社会模拟框架**，用于在更真实的环境中研究 LLM 智能体之间的**说服行为**。

其核心创新包括：
- **现实主义建模**：以美国政治选举为背景，构建了一个包含候选人、选民和事件生成者（eventor）的闭环社会系统。
- **细粒度背景设定**：为每个 agent 赋予基于 Pew Research 数据的政治立场（6 维度）和“大五人格”（Big 5 personality traits），增强行为多样性与真实性。
- **结构化交互机制**：引入唯一 ID 回复/点赞机制、280 字符限制、每日时间步长控制等，贴近真实社交平台操作逻辑。
- **开放源码框架**：所有代码公开（GitHub: `tcmmichaelb139/ai-electwit`），支持可复现研究。

### 相比现有方法的优势
| 方面 | 传统方法（如 Among Us 游戏框架） | ElecTwit |
|------|-------------------------------|---------|
| 环境真实性 | 低（抽象游戏规则） | 高（模拟真实社交媒体+选举） |
| 行为动机 | 外部任务驱动（赢游戏） | 内在政治偏好驱动（投票决策） |
| 社会结构 | 封闭小群体互动 | 开放信息流 + 全局 feed |
| 动态复杂性 | 有限 | 支持涌现行为（如“ink obsession”） |
| 可扩展性 | 弱 | 模块化设计，便于扩展 |

> ✅ **优势总结**：ElecTwit 提供了一个**更贴近现实、更具生态效度**（ecological validity）的测试床，能够揭示 LLM 在复杂社会语境中的真实说服能力与潜在风险。

---

## 2. 核心实验方法和设置

### 使用的数据集
本研究未使用外部真实世界数据集，而是通过以下方式构建**合成数据环境**：
- 所有 agent 的背景由随机采样生成，依据 Pew Research 中长期稳定的六大政治维度与 Big 5 人格模型。
- 事件由 `google/gemini-2.5-flash` 模型作为 eventor 自动生成，包含真实与虚假新闻（如候选人丑闻）。
- 所有交互记录（post、reply、like、vote、diary）均被保存为 JSON 格式日志。

### 实验设置
#### Agent 角色与数量
| 角色 | 数量 | 模型范围 |
|------|-----|----------|
| Voter Agents | 16 | 全部 8 种 LLM |
| Candidate Agents | 2 | 仅测试 3 种：`gpt-4.1-mini`, `gemini-2.5-flash`, `claude-3.5-haiku` |
| Eventor Agent | 1 | 固定为 `google/gemini-2.5-flash` |

#### 时间与交互机制
- 每“天”共 9 个时间步（9am–5pm），每小时一次交互机会。
- 每个 agent 有“行动概率”（chance to act）参数控制活跃度（voters/candidates: 0.4–0.9；eventor: 0.3–0.7）。
- 每天结束时汇总 diary 日志作为长期记忆输入次日。

#### 模拟分组
进行 11 场模拟，分为两组：
1. **Same Seed Group**（6 场）：固定随机种子，轮换候选人的 LLM 模型 → 分析**候选人模型影响**
2. **Different Seed Group**（6 场）：固定模型配置，改变随机种子 → 分析**背景与个体差异影响**

> 注：两组共享相同的 voter 模型组成。

### 评估指标
- **说服技术分类**：采用 [Idziejczak et al., 2025] 定义的 **25 类 persuasion techniques**，由独立 LLM 对所有 posts 和 comments 进行标注（允许一条消息属于多个类别）。
- 主要分析维度：
  - 各模型使用的 persuasion 技术频率分布
  - 投票结果与候选人相似性关系
  - 平台交互模式（posts vs replies vs likes）
  - 网络结构分析（reply/like 图谱）
- **最终输出指标**：
  - 总交互数（posts, comments, likes）
  - persuasion tag 总数及分布
  - 选举胜负结果
  - 涌现行为识别（qualitative）

### 基线方法对比
本文未直接与其他 multi-agent 框架进行定量性能对比，但明确指出其相对于如下工作的改进：
- **Casevo** [3]：虽有社会网络结构，但仍限于辩论场景。
- **AgentSociety** [4]：规模大但目标非聚焦 persuasion。
- **Among Us 类游戏框架** [8][9]：任务导向强，缺乏现实政治激励。

> ⚠️ 本质是提出一种**新的评估范式**，而非在已有 benchmark 上刷榜。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 数值 |
|------|------|
| 总交互次数 | 73,877 |
| Posts | 6,692 |
| Comments (replies) | 36,345 |
| Likes | 30,840 |
| Persuasion Tags 总数 | 125,254 |
| 平均每条消息标签数 | ~2.8 |

### 选举结果（Same Seed 组）
| 候选人模型 | 参与场次 | 胜出场次 |
|-----------|--------|--------|
| `gemini-2.5-flash` | 4 | 4 |
| `gpt-4.1-mini` | 4 | 2 |
| `claude-3.5-haiku` | 4 | 0 |

👉 **Gemini 2.5 Flash 明显胜率更高**，尽管样本量小，提示其更强的 persuasion 表现。

### 不同模型的 persuasion 技术使用情况
#### 最常见的 5 种 persuasion techniques（两组一致）：
1. **Appeal to Credibility**（诉诸可信度）——18,126 次
2. **Appeal to Logic**（诉诸逻辑）
3. **Appeal to Emotion**（诉诸情感）
4. **Vagueness**（模糊表述）
5. **Distraction**（转移注意力）

> 其余技术使用频率显著下降，呈现“长尾分布”。

#### 模型间差异显著：
- `gemini-2.5-flash`：产生最多 persuasion tags（无论作为 voter 或 candidate），表明高互动性和高频使用多种策略。
- `claude-3.5-haiku`：作为 candidate 时生成最少 persuasion tags。
- `grok-3-mini`：总体生成最少 persuasion tags。

📌 注意：**tag 数量多 ≠ 更具说服力**，更多反映的是**参与度高**。

### 与基线方法的对比结果
- 相比之前研究（如 Among Us 框架）仅观察到部分 persuasion 技术，**本研究首次在真实感环境中验证了全部 25 类技术的广泛存在**。
- 发现更大范围的技术组合使用，说明 LLM 在复杂社会情境下具备更丰富的 rhetorical 能力。

### 消融实验（间接体现）
虽然没有正式消融实验，但从设计中可推断：
- 若移除 agent 背景（political stance + personality），行为趋于同质化 → 验证了背景的重要性。
- 若取消 diary 机制，则长期一致性降低 → 验证 memory 设计的有效性。
- 同步更新 feed 虽不完全真实，但加速了 mass behavior 模拟 → 权衡效率与真实性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 普遍掌握并综合运用多种 persuasion techniques**，涵盖从理性论证到情绪操控的完整谱系。
2. ✅ **不同 LLM 架构在 persuasion 输出上有显著差异**，`gemini-2.5-flash` 表现出更强的主动性和策略多样性。
3. ✅ **agent 背景（background）对投票倾向的影响不明显**（Fig. 3 显示平均相似度接近零），暗示模型可能更受即时交互影响而非深层价值观匹配。
4. ✅ **出现意料之外的涌现行为**：
   - “**Kernel of Truth**” messages：包含部分事实但整体误导的信息。
   - “**Ink Obsession**” 现象：agents 集体要求候选人提供书面承诺（“no ink, no vote”），形成统一口号与压力机制。
5. ✅ **回音室效应未明显出现**：网络图显示，**低相似度 agents 反而互动更多**（可能是争论所致），挑战了传统 polarized 社交网络假设。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| Feed 无个性化 | 所有 agent 看同一 feed，缺乏算法推荐导致的 filter bubble 效应 |
| 缺乏人类参与者 | 评估依赖 LLM 自评，缺少 human-in-the-loop validation |
| 模拟规模较小 | 仅 16 voters，难以捕捉大规模社会动力学 |
| 温度设为 0 | 缺少随机性，行为过于确定，抑制 creativity 与多样性 |
| 成本限制实验设计 | 无法全面测试所有 candidate-voter 组合 |

### 未来工作方向
1. **引入个性化 feed 与推荐算法**，研究信息茧房与极化演化。
2. **加入 human evaluators**，实现 human-AI mixed society simulation。
3. **扩大 agent 数量至千级**，探索宏观社会趋势。
4. **引入 temperature 参数与 agent mutation 机制**，增强行为多样性。
5. **结合真实舆情数据初始化背景**，提升 ecological validity。
6. **研究恶意 persuasion 与 misinformation 传播路径**，服务于 AI safety 与 alignment 研究。

---

> 🔚 **总结一句话**：  
> **ElecTwit 是首个将 persuasion 研究置于高度仿真的社交媒体选举环境中的开源框架，揭示了 LLM 在复杂社会互动中丰富的说服策略与不可预测的集体行为，为未来 AI 社会模拟与安全治理提供了重要基础工具。**

</details>

---

### 12. [A New Benchmark for the Appropriate Evaluation of RTL Code Optimization](https://arxiv.org/abs/2601.01765)

**Authors**: Yao Lu, Shang Liu, Hangan Zhou, Wenji Fang, Qijun Zhang, Zhiyao Xie  
**Category**: cs.AI  
**Published**: 2026-01-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2601.01765v1  

#### Abstract
The rapid progress of artificial intelligence increasingly relies on efficient integrated circuit (IC) design. Recent studies have explored the use of large language models (LLMs) for generating Register Transfer Level (RTL) code, but existing benchmarks mainly evaluate syntactic correctness rather ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A New Benchmark for the Appropriate Evaluation of RTL Code Optimization

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于大语言模型（LLM）的 RTL 代码生成研究主要依赖于语法正确性作为评估标准，而忽视了对集成电路设计核心质量指标——**Power, Performance, and Area (PPA)** 的优化能力评估。现有的 RTL 优化基准（如 [26]）存在以下严重缺陷：
- **不切实际的设计**：子优 RTL 包含大量人为构造的冗余操作（如 `+0`、`*1`），这些在真实工程中几乎不会出现。
- **简化的综合流程**：依赖弱综合工具（如 Yosys），导致评估结果对表面代码变化过于敏感，无法反映工业级流程的真实效果。
- **评估维度单一**：仅关注面积相关指标，忽略功耗与时序等关键 PPA 维度。

这些问题使得现有基准容易高估 LLM 的优化能力，导致误导性结论。

### 提出的新方法与思路
本文提出了 **RTL-OPT**，一个专为评估 LLM 在 RTL 代码优化方面能力而设计的新基准。其核心创新包括：

- **高质量、现实的优化任务集**：包含 36 个手工编写的数字电路设计，覆盖组合逻辑、流水线数据路径、有限状态机（FSM）、存储器接口等多种类型。
- **真实的优化模式**：每个任务提供一对 RTL 代码——一个“子优”版本和一个由专家手工优化的“参考”版本。优化模式来源于工业实践，包括：
  - Bit-width Optimization
  - Precomputation & LUT Conversion
  - Operator Strength Reduction
  - Control Simplification
  - Resource Sharing
  - State Encoding Optimization
- **自动化评估框架**：集成完整的 EDA 工具链（Synopsys DC、Yosys、Formality、VCS），实现：
  - 功能等价性检查（Functional Equivalence Checking）
  - 多维度 PPA 量化评估（Power, WNS/TNS, Area/Cells）

### 相比现有方法的优势
| 特性 | 现有基准 [26] | RTL-OPT（本文） |
|------|----------------|------------------|
| 设计真实性 | 低（过度人为构造） | 高（反映真实优化机会） |
| 综合工具 | 弱（Yosys为主） | 强（DC compile_ultra） |
| 评估维度 | 单一（面积） | 全面（PPA） |
| 优化模式来源 | 不明确 | 工业实践验证 |
| 功能验证 | 缺失或不足 | 完整（Formality + VCS） |

RTL-OPT 能更可靠地衡量 LLM 是否真正提升了 RTL 的硬件实现质量。

---

## 2. 核心实验方法和设置

### 数据集
- **RTL-OPT 基准**：包含 36 个手写 RTL 优化任务，每个任务包含 suboptimal 和 optimized 两个版本。
- **对比基准**：复现并评估了先前工作 [26] 中的 43 对 RTL 设计（包括 human-optimized 和 LLM-optimized 版本）。

### 实验设置
- **综合工具**：
  - 商业工具：Synopsys Design Compiler (DC)，使用 `compile` 和 `compile_ultra` 模式。
  - 开源工具：Yosys。
- **时钟约束**：
  - 松散约束：1ns
  - 紧约束：0.1ns
- **技术库**：Nangate45 工艺库。
- **功能验证**：
  - 形式等价性检查：Synopsys Formality
  - 动态仿真验证：Synopsys VCS（针对带流水线的设计）

### 评估指标
- **PPA 指标**：
  - **Power**：总功耗（动态 + 泄漏）
  - **Performance**：Worst Negative Slack (WNS) 和 Total Negative Slack (TNS)
  - **Area**：单元数量（Cells）和物理面积（Area）
- **功能正确性**：通过 Formality/VCS 验证。
- **语法正确性**：能否通过 DC/Yosys 综合。

### 基线方法对比
- **基准对比**：将 [26] 的设计在相同流程下重新评估，验证其可靠性。
- **LLM 对比**：测试多个主流 LLM 在 RTL-OPT 上的表现：
  - GPT-4o-mini
  - Gemini-2.5
  - Deepseek V3
  - Deepseek R1

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 3）

#### ✅ RTL-OPT 自身有效性验证
在 `DC (compile_ultra, 1ns)` 设置下：
| 基准 | 更好 | 相同 | 更差 | 贸易-offs |
|------|-----|-----|-----|---------|
| [26] Human-Optimized | 16 | 13 | 7 | 7 |
| **RTL-OPT Human-Optimized** | **35** | **3** | **0** | **0** |

👉 结论：RTL-OPT 中的人工优化版本在绝大多数情况下（35/36）均优于子优版本，证明其优化是真实有效的；而 [26] 的优化在强综合下常被抹平。

#### ✅ 平均 PPA 改进（DC 结果）
| 指标 | 子优设计 | 优化设计 | 改进 (%) |
|------|--------|--------|--------|
| Cells | 1337.4 | 1047.7 | **27.6%** |
| Area | 1226.8 mW | 901.8 mW | **36.0%** |
| Power | 14.0 ns | 9.1 ns | **53.4%** |

👉 显示 RTL-OPT 中存在显著且一致的优化空间。

#### ✅ LLM 在 RTL-OPT 上的表现（Figure 3 & Table 4）
| LLM | 语法正确率 | 功能正确率 | PPA 优于子优 | PPA 优于人工优化 |
|-----|------------|------------|---------------|------------------|
| GPT-4o-mini | 97.2% | 75% | 19.4% | ~2.2% |
| Gemini-2.5 | ~100% | ~75% | ~27.8% | ~2.4% |
| Deepseek V3 | 100% | 69.4% | 23.3% | ~4.4% |
| **Deepseek R1** | 86.1% | 61.1% | **41.7%** | **13.9%** |

👉 **Deepseek R1 表现最强**，能在约 15 个任务上成功优化，其中 5 个甚至超过人工设计。

### 与基线方法的对比结果
- 在 [26] 的基准上，许多“优化”RTL 在 DC 下与原始版本无异，说明其优化无效。
- 在 RTL-OPT 上，所有 LLM 表现均远低于人类专家，表明该基准具有挑战性。
- 使用 Yosys 会夸大优化效果，而 DC 更能揭示真实优化价值。

### 消融实验 / 进一步分析
- **不同综合配置的影响**：使用更紧的时序约束（0.1ns）会导致更多 PPA trade-offs，但 RTL-OPT 仍保持有效性。
- **错误模式分析**：对失败案例的手动检查发现常见错误包括：
  - 控制逻辑错误（如 FSM 状态转移条件错误）
  - 过度流水线化（违反延迟要求）
  - 资源共享不当（寄存器重用导致数据冲突）

---

## 4. 关键结论和发现

### 主要发现
1. **现有 RTL 优化基准不可靠**：[26] 等工作的基准因设计不真实、评估流程薄弱，容易产生误导性结果。
2. **综合工具选择至关重要**：商业工具（如 DC）的强大优化能力会“抹平”表面优化，因此必须使用工业级流程进行评估。
3. **PPA 评估需全面**：不能只看面积，必须综合考虑 Power、Performance、Area 及其权衡。
4. **LLM 仍有巨大提升空间**：尽管 Deepseek R1 表现出色，但整体成功率仍较低，尤其在复杂控制逻辑上易出错。
5. **功能正确性是前提**：最激进的优化者（Deepseek R1）也最容易引入功能错误，凸显了“安全优化”的重要性。

### 方法的局限性
- 当前 RTL-OPT 规模为 36 个任务，虽具代表性，但仍可扩展。
- 所有设计基于 Nangate45 工艺库，跨工艺迁移性有待验证。
- 未涵盖高层次综合（HLS）或系统级优化场景。

### 未来工作方向
- 扩展 RTL-OPT 至更大规模和更多设计类别（如 CPU core、memory controller）。
- 构建面向特定应用场景（AI accelerator, IoT）的专用子基准。
- 探索结合 symbolic reasoning 或 formal methods 的 LLM 优化代理，提高功能安全性。
- 将 RTL-OPT 集成到 LLM 训练反馈循环中，用于强化学习驱动的优化策略学习。

> 🔗 **项目开源地址**：https://anonymous.4open.science/r/RTL-OPT-20C5

</details>

---

### 13. [Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](https://arxiv.org/abs/2601.02956)

**Authors**: Jeonghyun Park, Byeongjeong Kim, Seojin Hwang, Hwanhee Lee  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2601.02956v1  

#### Abstract
Multilingual Retrieval-Augmented Generation (mRAG) systems often exhibit a perceived preference for high-resource languages, particularly English, resulting in the widespread adoption of English pivoting. While prior studies attribute this advantage to the superior English-centric capabilities of La...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Multilingual RAG (mRAG)** 系统中存在的一个普遍现象——**对英语的偏好（English preference）** 进行深入分析。传统观点认为这种偏好源于大语言模型（LLMs）在英语上的更强能力（即“English-centric”优势），因此广泛采用 **English pivoting**（将非英语查询翻译为英语再检索）作为有效策略。

然而，本文指出，这种“英语偏好”很大程度上是由于 **评估基准中的结构性偏差（structural priors）** 所导致的假象，而非模型本身的内在语言偏好。这些偏差主要包括：
- **Gold Availability Prior**：正确答案（gold passage）在英语维基百科中覆盖率远高于其他语言。
- **Exposure Bias**：英语文档在检索结果中出现频率更高，造成其更容易被选中。
- **Cultural Prior**：某些问题与特定文化或地域强相关，其本地语言中的表面形式（如专有名词、别名、脚本）成为检索锚点，从而表现出“语言偏好”。

### 提出的新方法与新思路

#### （1）DeLP (Debiased Language Preference)
- **目标**：提出一种去偏的语言偏好度量方法，以揭示 mRAG 系统真实的内在语言偏好。
- **方法**：通过岭回归（ridge regression）从观测到的语言偏好分数中剥离出由上述三种结构性先验（exposure, gold availability, cultural）带来的影响，残差部分被视为“真实”的语言偏好。
- **意义**：首次系统性地识别并量化了 mRAG 中的结构性偏差，并提供了一个可校准的测量框架。

#### （2）DELTA (DEbiased Language preference-guided Text Augmentation)
- **目标**：基于 DeLP 发现的真实偏好（即 **monolingual alignment**，查询与文档同语言时表现最佳），设计一种轻量级的查询增强方法。
- **方法**：将原始查询（[LOCAL]）、英文翻译（[GLOB]）、跨语言标题桥接（[TITLE_BRIDGE]）、别名（[ALIASES]）和区域提示（[LOCALE_HINT]）融合成一个统一的查询。
- **机制**：使用 **repetition-based weighting** 策略，根据问题的文化特异性（`y`）和置信度（`c`）动态调整本地与全局组件的重复次数，实现偏好引导下的平衡。

### 相比现有方法的优势
- **无需修改模型或知识库**：DELTA 是纯文本级别的查询重构，不涉及模型微调或文档翻译。
- **高效低成本**：避免了 document-level translation 或多轮生成的成本。
- **性能更优**：在多个生成器上均显著优于 English pivoting 和其他主流 mRAG 基线。
- **揭示真相**：DeLP 揭示了“英语偏好”主要是数据分布偏差的结果，而非模型本质能力差异。

---

## 2. 核心实验方法和设置

### 数据集
- **主数据集**：**MKQA**（Multilingual Knowledge Questions and Answers），包含 10k 条专业翻译的问答对。
- **子集构建**：选取与 **KILT-NQ** 重叠的 2.7K 示例，以便获取标准化的文档来源（gold passage IDs）。
- **知识源**：使用 **Wikipedia** 的英文版及用户本地语言版本作为检索语料库。

### 实验设置与评估指标
- **检索器**：**BGE-m3**（多语言嵌入模型），用于初始检索与重排序。
- **生成器**：三个强大的多语言 LLM：
  - **Qwen3-235B**
  - **DeepSeek-v3.1**
  - **Gemini-2.5-Flash**
- **检索参数**：Top-50 文档，使用 Top-5 进行生成。
- **评估指标**：
  - **End-to-end Accuracy**：使用字符 3-gram recall 衡量生成答案与参考答案的匹配程度。
  - **Retriever Recall@50**：衡量黄金段落在前 50 个检索结果中的召回率。
  - **Latency**：端到端响应时间。

### 基线方法对比
| 方法 | 类型 | 简介 |
|------|------|------|
| **MultiRAG** | Document Level | 使用原语言查询进行多语言检索与生成 |
| **CrossRAG** | Document Level | 检索后将所有文档翻译为英语再生成 |
| **DKM-RAG** | Document Level | 翻译并使用 LLM 生成多个精炼版本 |
| **QTT-RAG** | Document Level | 添加翻译质量标签供生成器判断可信度 |
| **English Translation** | Query Level | 将查询翻译为英语后检索与生成 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 5）
在 **Qwen3-235B** 上，DELTA 的平均准确率为 **62.88**，显著优于：
- English Translation（58.81）
- MultiRAG（51.30）
- QTT-RAG（51.65）

在 **Gemini-2.5-Flash** 上，DELTA 平均准确率为 **56.28**，优于：
- English Translation（52.75）
- MultiRAG（43.80）

在 **DeepSeek-v3.1** 上，DELTA 平均准确率为 **58.23**，优于：
- English Translation（55.26）
- MultiRAG（45.46）

> ✅ **结论**：DELTA 在所有生成器上均取得最佳或接近最佳的整体性能，尤其在非英语语言上提升显著。

### 与基线方法的对比结果
- **相比 English pivoting**：DELTA 在所有非英语语言上均有明显提升（+5~10 pts），仅在英语上略有下降（因冗余信息稀释信号）。
- **相比 Document-level 方法**：DELTA 性能优于或媲美需要高昂计算成本的文档翻译方法（如 CrossRAG, DKM-RAG），且延迟更低。
- **效率优势**：DELTA 的平均延迟（1.13s）低于多数基线，甚至快于 English Translation（1.17s），因其保留本地锚点有助于快速聚焦相关文档。

### 消融实验结果（Table 7）
在固定证据条件下测试不同查询组件的影响：
| 方法 | 准确率（平均） |
|------|----------------|
| Original Query | 63.13 |
| + Global ([GLOB]) | 71.51 |
| + Title Bridge | 68.00 |
| + Aliases | 67.90 |
| + Locale Hint | 67.68 |
| **All Cues (DELTA)** | **72.89** |

> 🔍 **发现**：
- 英文翻译本身就能大幅提升生成效果（global cue）。
- 所有桥接组件（title, aliases, locale）均带来独立增益。
- 组合所有线索达到最优性能，说明它们共同提供了关键的消歧和实体定位功能。

---

## 4. 关键结论和发现

### 主要发现
1. **“英语偏好”是假象**：mRAG 系统中观察到的英语优势主要源于 **gold availability bias** 和 **cultural priors**，而非模型本身的语言能力差异。
2. **真实偏好是单语对齐（monolingual alignment）**：经过 DeLP 去偏后，发现检索器最倾向于匹配查询语言与文档语言一致的情况。
3. **DELTA 有效利用这一偏好**：通过融合本地与全局信息，并动态加权，实现了更精准的跨语言检索与生成。
4. **轻量高效胜过复杂流程**：DELTA 作为一种 query-level 方法，在性能上超越了多种依赖 document translation 的复杂框架。

### 方法的局限性
1. **仅处理检索侧偏好**：未解决生成器如何消费多语言证据的偏好问题（generator-level bias）。
2. **依赖 Wikipedia 设置**：结论基于通用百科知识库，是否适用于领域特定或多模态语料尚待验证。
3. **权重机制较粗糙**：当前使用重复次数控制权重，缺乏更精细的 attention 或 soft fusion 机制。
4. **可能过度本地化**：对于意图模糊的问题（如“朝鲜战争时期的总统”），注入过多本地线索可能导致错误导向（见 Table 15 失败案例）。

### 未来工作方向
- 将去偏思想扩展至 **generator-level preference modeling**。
- 探索更精细的 **adaptive weighting** 或 **intent disambiguation** 机制。
- 在更多样化的 **domain-specific corpora** 上验证 DeLP 与 DELTA 的泛化能力。
- 结合 **multimodal** 或 **code-switching** 场景下的语言偏好研究。

--- 

> 📌 **总结一句话**：  
> 本文揭示了 mRAG 中“英语偏好”的结构性偏差本质，提出了 DeLP 度量与 DELTA 框架，证明通过去偏后的单语对齐偏好指导查询融合，可在低成本下实现超越 English pivoting 的多语言 RAG 性能。

</details>

---

### 14. [LLM-Enhanced Reinforcement Learning for Time Series Anomaly Detection](https://arxiv.org/abs/2601.02511)

**Authors**: Bahareh Golchin, Banafsheh Rekabdar, Danielle Justo  
**Category**: cs.LG  
**Published**: 2026-01-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2601.02511v1  

#### Abstract
Detecting anomalies in time series data is crucial for finance, healthcare, sensor networks, and industrial monitoring applications. However, time series anomaly detection often suffers from sparse labels, complex temporal patterns, and costly expert annotation. We propose a unified framework that i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LLM-Enhanced Reinforcement Learning for Time Series Anomaly Detection 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**时间序列异常检测**中的三大挑战：
- **标签稀疏性**（sparse labels）：真实场景中异常样本极少且标注成本高；
- **复杂时序模式**（complex temporal patterns）：如周期性、趋势漂移、多变量耦合等；
- **探索效率低与奖励稀疏**（sparse rewards in RL）：传统强化学习在缺乏密集反馈时难以有效训练。

这些问题导致现有方法在数据受限条件下性能下降明显，尤其在工业监控、医疗健康等领域应用受限。

---

### 🚀 提出的新方法与创新思路
作者提出一个**统一框架**，将以下四个模块集成于一个端到端系统中：

| 模块 | 创新点 |
|------|--------|
| **LSTM-based RL Agent** | 使用LSTM建模长期依赖关系，在滑动窗口上进行序列决策，实现逐点分类（正常/异常）。 |
| **LLM-based Potential Function for Reward Shaping** | 引入Large Language Model（LLM）生成语义化的“潜在函数”（potential function），用于**基于语义的奖励塑形**（semantic reward shaping），无需人工设计特征即可注入领域知识。 |
| **VAE-enhanced Dynamic Reward Scaling** | 利用Variational Autoencoder（VAE）的重构误差作为无监督异常信号，并通过动态系数 `λ(t)` 自适应融合监督与非监督奖励，提升样本效率。 |
| **Active Learning + Label Propagation** | 主动学习选择不确定性最高的样本请求标注；随后利用相似性权重进行伪标签传播，显著减少人工标注负担。 |

> 🔑 核心思想：**用LLM提供可解释、上下文感知的异常严重度评分，结合PBRS理论指导RL探索方向，同时借助VAE和主动学习缓解标注稀缺问题。**

---

### ⭐ 相比现有方法的优势
| 对比维度 | 本文方法优势 |
|---------|-------------|
| **样本效率** | Active learning + label propagation 极大降低标注需求 |
| **探索引导能力** | LLM提供的语义奖励比随机探索更高效，解决RL中的稀疏奖励问题 |
| **泛化能力** | LLM预训练知识帮助识别未见模式，适应新型异常 |
| **策略不变性保障** | 使用Potential-Based Reward Shaping（PBRS），确保引入额外奖励不改变最优策略 |
| **多源信号融合** | 动态加权机制平衡分类准确率与重构误差信号，增强鲁棒性 |

---

## 2. 核心实验方法和设置

### 📊 数据集
在两个广泛使用的基准数据集上验证方法有效性：

| 数据集 | 类型 | 特征数 | 序列数量 | 异常比例 | 描述 |
|-------|------|--------|----------|-----------|------|
| **Yahoo-A1** | 单变量（univariate） | 1 | 67 | 1.76% | 来自Yahoo网站流量的真实时间序列，异常为突发峰值或均值偏移 |
| **SMD**（Server Machine Dataset） | 多变量（multivariate） | 38 | 28 | 4.16% | 工业服务器传感器数据，含季节性和渐变漂移，异常由多个传感器协同变化构成 |

---

### 🧪 实验设置
- **状态表示**：滑动窗口长度 `n_steps=25`，每个状态包含历史观测 + 动作标志位（action flag）
- **动作空间**：二分类 `{0: 正常, 1: 异常}`
- **奖励结构**：
  - 分类奖励 $ R_1 \in \{+5(TP), +1(TN), -1(FP), -5(FN)\} $
  - VAE重构误差奖励 $ R_2 = MSE(x, \hat{x}) $
  - 总奖励：$ R_{\text{total}} = R_1 + \lambda(t) \cdot R_2 $，其中 $\lambda(t)$ 通过比例控制动态调整
- **LLM提示工程**：采用few-shot prompting方式输入传感器数据，要求输出JSON格式 `{"severity": v}`，$v \in [0,1]$ 表示异常严重程度
- **测试的LLM模型**：GPT-3.5, Llama-3.2-3B, Phi-2

---

### 📈 评估指标
使用标准异常检测评价指标：
- **Precision（精确率）**
- **Recall（召回率）**
- **F1 Score（F1分数）**

所有结果均为跨多个时间序列的平均性能。

---

### 🆚 基线方法对比
包括多种主流及SOTA方法：
- **THOC**, **TS2Vec**, **TimesNet**, **TranAD**, **DCdetector**：深度学习时间序列模型
- **CARLA**：当前表现较强的RL-based anomaly detection方法

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自Table II）

#### 在 **Yahoo-A1**（单变量）上的表现：

| 方法 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| CARLA (best non-LLM) | 0.5747 | 0.9755 | **0.7233** |
| **Proposed + Llama-3** | **0.6051** | **0.9565** | **0.7413** ✅ |
| Proposed + GPT-3.5 | 0.0742 | 0.9130 | 0.1372 ❌（过敏感）|
| Proposed + Phi-2 | 0.6666 | 0.4761 | 0.5555 ❌（欠敏感）|

> ✅ **Llama-3版本在F1上超越CARLA，达到SOTA水平**

---

#### 在 **SMD**（多变量）上的表现：

| 方法 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| CARLA | 0.4276 | 0.6362 | **0.5114** |
| **Proposed + Llama-3** | 0.3813 | **0.8685** | **0.5300** ✅ |
| Proposed + GPT-3.5 | 0.5370 | 0.4061 | 0.4625 |
| Proposed + Phi-2 | **0.8461** | 0.2541 | 0.3908 |

> ✅ **Llama-3在F1上再次优于CARLA，尤其以高达86.85%的Recall领先**

---

### 🔍 关键观察与分析（Q1-Q3研究问题回答）

#### Q1: LLM用于奖励塑形是否优于现有方法？
✅ 是。特别是使用Llama-3时，无论在单变量还是多变量数据集中，F1均超过最强非LLM基线（CARLA），证明**LLM语义奖励能有效加速RL收敛并提高检测质量**。

#### Q2: 不同LLM的表现差异及原因？
| LLM | 表现特点 | 原因分析 |
|-----|----------|---------|
| **Llama-3** | 最佳平衡（高Recall + 合理Precision） | 遵循指令能力强，输出平滑、一致性好，对正常波动不过激 |
| **GPT-3.5** | 高Recall但极低Precision | 存在“过度报警”倾向，轻微波动也判为异常（prior太强） |
| **Phi-2** | 高Precision但低Recall | 过于保守，仅标记最明显的异常，漏检严重 |

> 💡 结论：**并非越大越好的LLM就更适合此任务，需关注其输出稳定性与校准能力**

#### Q3: 单变量 vs 多变量数据表现差异？
- **Yahoo-A1（单变量）**：异常通常是突变或尖峰，容易被LLM捕捉 → 整体性能更高
- **SMD（多变量）**：异常表现为多传感器协同偏离，需理解跨变量关系 → 更具挑战性
- 尽管如此，本方法仍取得SOTA结果，说明**LLM+VAE+动态奖励机制具备处理复杂模式的能力**

---

### 🔁 消融实验（隐含分析）
虽然文中未明确列出消融表，但从设计逻辑和讨论中可推断以下关键组件作用：
- 若去除LLM奖励塑形 → 探索效率下降，类似CARLA表现
- 若固定`λ`而非动态调整 → 无法适应不同episode性能波动，效果退化
- 若不用active learning → 标注成本上升，小样本下性能骤降

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM可用于生成语义驱动的潜在函数**，在不破坏策略不变性的前提下显著提升RL探索效率；
2. **Llama-3在异常严重度评估方面优于GPT-3.5和Phi-2**，因其输出更稳定、校准更好；
3. **动态奖励缩放机制有效融合监督与无监督信号**，使模型在有限标签下依然保持高性能；
4. 所提框架在**单变量与多变量数据集上均达到SOTA**，尤其在Recall方面表现突出，适合对漏报容忍度低的应用场景（如工业安全监测）；
5. **主动学习+标签传播大幅降低标注开销**，符合现实部署需求。

---

### ⚠️ 局限性
1. **计算开销较大**：每次决策需调用LLM推理，可能影响实时性；
2. **依赖LLM API可用性与成本**：若使用闭源模型（如GPT），存在服务中断或费用风险；
3. **多变量关联建模不足**：当前LLM输入为单个传感器窗口，未能显式建模跨传感器依赖；
4. **few-shot prompt设计敏感**：性能受示例选择影响，缺乏自动化优化机制。

---

### 🔮 未来工作方向
1. 探索轻量化LLM（如TinyLLM）或微调小型模型替代API调用；
2. 设计**多变量联合输入prompt**，让LLM理解传感器间因果或相关结构；
3. 引入**记忆机制或上下文缓存**，避免重复调用LLM；
4. 扩展至在线流式检测场景，支持增量学习；
5. 探索更多类型的内在奖励（intrinsic rewards）与LLM结合方式。

---

## 总结

> 🌟 本文开创性地将**LLM语义理解能力融入RL-based异常检测框架**，通过**PBRS理论保证策略一致性**，结合**VAE、动态奖励、主动学习**形成闭环系统，在**低标注预算下实现了SOTA性能**。不仅展示了LLM在非自然语言任务中的潜力，也为未来智能运维、工业AI提供了可扩展的技术路径。

</details>

---

### 15. [MAFS: Multi-head Attention Feature Selection for High-Dimensional Data via Deep Fusion of Filter Methods](https://arxiv.org/abs/2601.02668)

**Authors**: Xiaoyan Sun, Qingyu Meng, Yalu Wen  
**Category**: cs.LG  
**Published**: 2026-01-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2601.02668v1  

#### Abstract
Feature selection is essential for high-dimensional biomedical data, enabling stronger predictive performance, reduced computational cost, and improved interpretability in precision medicine applications. Existing approaches face notable challenges. Filter methods are highly scalable but cannot capt...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MAFS: Multi-head Attention Feature Selection for High-Dimensional Data via Deep Fusion of Filter Methods

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

高维生物医学数据（如基因表达、SNP、多组学数据）的特征选择面临三大挑战：
- **Filter 方法** 虽然高效可扩展，但无法捕捉非线性关系且容易遗漏冗余特征；
- **深度学习方法** 能建模复杂依赖，但缺乏稳定性、可解释性和初始化鲁棒性；
- **单头注意力机制（Single-head Attention）** 可解释性强，但只能从单一视角捕获特征依赖，对初始化敏感，导致结果不可复现。

此外，大多数现有方法未能有效结合统计方法的**可解释性**与深度学习的**表征能力**，尤其在超高维场景下表现不佳。

---

### **提出了什么新方法或新思路**

作者提出 **MAFS（Multi-head Attention-based Feature Selection）**，一种融合滤波方法与多头注意力机制的混合框架，其核心创新包括：

1. **Filter-based Prior Initialization**  
   利用多种经典 filter 方法（如 SIS、BCor-SIS、Kendall’s tau）生成初始特征重要性权重，作为多头注意力模块的“软先验”（soft priors），用于引导训练过程，缓解随机初始化带来的不稳定性。

2. **Multi-head Attention 架构**  
   每个 attention head 并行处理来自不同 filter 方法的先验信息，从多个统计视角（线性、非线性、秩相关等）独立评估特征重要性，增强模型对复杂、异质依赖关系的捕捉能力。

3. **Reordering Module**  
   设计了一个重排序模块，将各 attention head 输出的 Top-K 特征合并为候选集，并通过树模型（如 Random Forest）重新打分排序，解决多视角冲突、减少信息丢失，提升最终特征排名的鲁棒性。

4. **External Attention 降低计算复杂度**  
   使用 external attention 替代 self-attention，将计算复杂度由 $O(n^2p)$ 降至 $O(np)$，显著提高效率，适用于超高维场景。

---

### **相比现有方法的优势**

| 维度 | MAFS 优势 |
|------|-----------|
| **准确性** | 在多种函数关系（尤其是非线性）下显著优于 GRACES、EAR-FS、DeepLIFT 等基线方法 |
| **稳定性** | 覆盖率置信区间更窄（±2.2% vs ±2.5–3.1%），跨重复实验一致性更高 |
| **效率** | 比最强基线 GRACES 快一个数量级以上（如 117.6 min vs 1,569.9 min） |
| **可解释性** | 注意力权重提供透明的重要性评分，支持多视角比较分析 |
| **泛化性** | 在连续/分类特征、连续/二元输出、不同维度下均保持领先 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### （1）模拟数据（Simulation）
- 基于 **OmicsSIMLA** 和 **HAPGEN2** 生成：
  - 连续型：RNA-seq 表达数据（~26,000 基因）
  - 分类型：SNP 基因型数据（~350,000 SNPs）
- 功能形式：7 种特征-响应关系（线性、对数、余弦、指数、交互项等）
- 样本量：$n = 500$ 和 $n = 2,000$
- 维度：$p = 25K, 50K, 100K$

#### （2）真实数据
- **癌症基因表达数据集（6个）**：
  - Colon, Leukemia, ALLAML, GLI_85, Prostate_GE, SMK_CAN_187
  - 任务：二分类（肿瘤 vs 正常）
  - 特征：标准化表达值（实数）
- **ADNI 数据集（阿尔茨海默病神经影像计划）**：
  - 样本数：449
  - 特征：49,386 个基因表达
  - 输出：9 个脑区体积（连续型表型）
  - 任务：回归预测

---

### **实验设置和评估指标**

| 类别 | 设置 |
|------|------|
| **评估指标** | - **Coverage Rate**（模拟数据）：选中的因果特征比例<br>- **AUROC**（癌症数据）：分类性能<br>- **Pearson Correlation**（ADNI）：回归预测准确性 |
| **特征选择策略** | - 固定比例（top 2%）<br>- 固定数量（top 100 / 500） |
| **下游模型** | - 分类器：SVM, KNN, MLP<br>- 回归器：SVR, KNN Regressor, MLP Regressor |
| **验证方式** | - 模拟数据：20次重复，80%/20% train/validation split<br>- 癌症数据：5-fold CV + 20% test<br>- ADNI：60%/20%/20% split |
| **超参优化** | 使用 **Optuna** 进行 100 次贝叶斯搜索 |

---

### **基线方法对比**

MAFS 与以下四类主流深度学习特征选择方法进行比较：

| 方法 | 类型 | 特点 |
|------|------|------|
| **GRACES** | Graph-based | 基于图卷积网络（GCN）和梯度估计，性能强但极慢 |
| **EAR-FS** | Attention-based | 外部注意力机制，轻量高效 |
| **DeepLIFT** | Gradient-based | 归因方法，依赖参考输入，易受噪声影响 |
| **CancelOut** | Layer-based | 可学习门控层，简单快速但表现差 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### （1）模拟实验：Coverage Rate（n=2000, p=100K）

| 方法 | Linear | Logarithmic | Cosine | Interaction |
|------|--------|-------------|--------|------------|
| **MAFS** | 0.86 | **1.00** | **0.93** | **0.56** |
| GRACES | 0.84 | 0.25 | 0.33 | 0.43 |
| EAR-FS | 0.70 | 0.19 | 0.27 | 0.37 |
| DeepLIFT | 0.26 | 0.09 | 0.15 | 0.20 |
| CancelOut | 0.05 | 0.05 | 0.05 | 0.05 |

> ✅ MAFS 在所有非线性关系上大幅领先，尤其在 log/cosine 上接近完美覆盖。

#### （2）维度扩展实验（top 100 features, n=2000）

当维度从 25K 升至 100K，**覆盖率下降幅度**：

| 方法 | Continuous Y | Binary Y |
|------|--------------|----------|
| **MAFS** | ↓21% | ↓38% |
| GRACES | ↓47% | ↓44% |
| EAR-FS | ↓69% | ↓87% |
| DeepLIFT | ↓80% | ↓50% |
| CancelOut | ↓100% | ↓100% |

> ✅ MAFS 对维度增长最稳健，即使固定选择 top 100 特征仍能维持较高覆盖率。

#### （3）运行时间对比（p=100K, n=2000, continuous Y）

| 方法 | 时间（分钟） |
|------|---------------|
| **MAFS** | 117.6 |
| GRACES | **1,569.9** |
| EAR-FS | 29.1 |
| DeepLIFT | 13.2 |
| CancelOut | 20.2 |

> ⚠️ MAFS 比最慢的 GRACES 快 **13倍以上**，虽比 EAR-FS 慢约4倍，但性能远超。

---

### **与基线方法的对比结果**

- **在所有模拟场景中，MAFS 覆盖率最高**，尤其在非线性、交互效应场景优势明显；
- **在真实癌症数据中**，MAFS 在 18 个测试场景中赢得 17 场（3分类器×6数据集），仅在前列腺癌+SVM 下与 DeepLIFT 持平；
- **在 ADNI 回归任务中**，MAFS 在多数脑区达到最高 Pearson 相关性（如脑干：0.440 vs GRACES 0.418）；
- **特征利用效率更高**：在白血病数据中，MAFS 仅需 2–3 个特征即可达到 AUROC > 0.95，而 GRACES 需 4–6，EAR-FS 需 8–10。

---

### **消融实验结果（间接体现）**

虽然未明确列出消融实验表格，但文中指出：
- **仅使用 filter 权重初始化即可提升 EAR-FS 性能**（见 Supplementary Fig S12–S19），说明初始化质量至关重要；
- 多头注意力架构进一步提升了性能，表明“多视角并行评估”本身具有增益；
- Reordering 模块避免了简单投票导致的信息损失，使低频但高价值特征也能被保留。

---

## 4. 关键结论和发现

### **主要发现**

1. **Filter 方法 + 深度学习 = 更优平衡**  
   将 filter 方法的统计先验融入 attention 初始化，既保留了解释性，又增强了深度模型的稳定性和收敛速度。

2. **Multi-head Attention 显著提升鲁棒性**  
   不同 attention head 可分别关注线性、非线性、排序等不同类型的特征-响应关系，避免单一视角偏差。

3. **MAFS 在复杂关系下表现卓越**  
   尤其在 log、cosine、interaction 等非线性场景中，MAFS 几乎实现完全覆盖，远超其他方法。

4. **实际应用中更具实用性**  
   - 更少特征即可达到高性能 → 降低下游计算成本
   - 更稳定的排名 → 支持可靠 biomarker 发现
   - 合理的运行时间 → 可用于大规模基因组研究

---

### **方法的局限性**

1. **归一化可能导致尺度失配**  
   不同 filter 方法的权重范围差异可能因归一化而扭曲，影响先验信号强度。

2. **要求完整特征矩阵**  
   当前版本不支持缺失值处理，可能在真实临床数据中受限。

3. **Reordering 模块依赖额外模型**  
   使用 Random Forest/Gini 进行重排序引入了新的建模假设，可能带来偏倚。

---

### **未来工作方向**

- 探索更先进的 filter 权重融合策略（如 learnable fusion weights 或 rank-based aggregation）
- 引入缺失数据处理机制（如 imputation-aware attention）
- 扩展至多任务或多模态特征选择（如联合基因+影像+临床数据）
- 开发可视化工具以支持多视角特征重要性比较分析

---

> 🔗 **代码与数据可用性**  
> - GitHub: [https://github.com/xsun768/mafs_package](https://github.com/xsun768/mafs_package)  
> - 癌症数据: [https://jundongl.github.io/scikit-feature/datasets.html](https://jundongl.github.io/scikit-feature/datasets.html)  
> - ADNI 数据: [https://adni.loni.usc.edu/](https://adni.loni.usc.edu/)（需注册）

</details>

---

### 16. [Uni-FinLLM: A Unified Multimodal Large Language Model with Modular Task Heads for Micro-Level Stock Prediction and Macro-Level Systemic Risk Assessment](https://arxiv.org/abs/2601.02677)

**Authors**: Gongao Zhang, Haijiang Zeng, Lu Jiang  
**Category**: cs.LG  
**Published**: 2026-01-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2601.02677v1  

#### Abstract
Financial institutions and regulators require systems that integrate heterogeneous data to assess risks from stock fluctuations to systemic vulnerabilities. Existing approaches often treat these tasks in isolation, failing to capture cross-scale dependencies. We propose Uni-FinLLM, a unified multimo...

---

### 17. [From Memorization to Creativity: LLM as a Designer of Novel Neural-Architectures](https://arxiv.org/abs/2601.02997)

**Authors**: Waleed Khalid, Dmitry Ignatov, Radu Timofte  
**Category**: cs.LG  
**Published**: 2026-01-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2601.02997v1  

#### Abstract
Large language models (LLMs) excel in program synthesis, yet their ability to autonomously navigate neural architecture design--balancing syntactic reliability, performance, and structural novelty--remains underexplored. We address this by placing a code-oriented LLM within a closed-loop synthesis f...

---

### 18. [MindChat: A Privacy-preserving Large Language Model for Mental Health Support](https://arxiv.org/abs/2601.01993)

**Authors**: Dong Xue, Jicheng Tu, Ming Wang, Xin Yan, Fangzhou Liu, Jie Hu  
**Category**: cs.AI  
**Published**: 2026-01-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2601.01993v1  

#### Abstract
Large language models (LLMs) have shown promise for mental health support, yet training such models is constrained by the scarcity and sensitivity of real counseling dialogues. In this article, we present MindChat, a privacy-preserving LLM for mental health support, together with MindCorpus, a synth...

---

### 19. [WebAnchor: Anchoring Agent Planning to Stabilize Long-Horizon Web Reasoning](https://arxiv.org/abs/2601.03164)

**Authors**: Yu Xinmiao, Zhang Liwen, Feng Xiaocheng, Jiang Yong, Qin Bing, Xie Pengjun, Zhou Jingren  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2601.03164v1  

#### Abstract
Large Language Model(LLM)-based agents have shown strong capabilities in web information seeking, with reinforcement learning (RL) becoming a key optimization paradigm. However, planning remains a bottleneck, as existing methods struggle with long-horizon strategies. Our analysis reveals a critical ...

---

### 20. [Electricity Price Forecasting: Bridging Linear Models, Neural Networks and Online Learning](https://arxiv.org/abs/2601.02856)

**Authors**: Btissame El Mahtout, Florian Ziel  
**Category**: cs.LG  
**Published**: 2026-01-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2601.02856v1  

#### Abstract
Precise day-ahead forecasts for electricity prices are crucial to ensure efficient portfolio management, support strategic decision-making for power plant operations, enable efficient battery storage optimization, and facilitate demand response planning. However, developing an accurate prediction mo...

---

### 21. [FlowPlan-G2P: A Structured Generation Framework for Transforming Scientific Papers into Patent Descriptions](https://arxiv.org/abs/2601.02589)

**Authors**: Kris W Pan, Yongmin Yoo  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2601.02589v1  

#### Abstract
Over 3.5 million patents are filed annually, with drafting patent descriptions requiring deep technical and legal expertise. Transforming scientific papers into patent descriptions is particularly challenging due to their differing rhetorical styles and stringent legal requirements. Unlike black-box...

---

### 22. [LLM-Augmented Changepoint Detection: A Framework for Ensemble Detection and Automated Explanation](https://arxiv.org/abs/2601.02957)

**Authors**: Fabian Lukassen, Christoph Weisser, Michael Schlee, Manish Kumar, Anton Thielmann, Benjamin Saefken, Thomas Kneib  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2601.02957v1  

#### Abstract
This paper introduces a novel changepoint detection framework that combines ensemble statistical methods with Large Language Models (LLMs) to enhance both detection accuracy and the interpretability of regime changes in time series data. Two critical limitations in the field are addressed. First, in...

---

### 23. [UltraLogic: Enhancing LLM Reasoning through Large-Scale Data Synthesis and Bipolar Float Reward](https://arxiv.org/abs/2601.03205)

**Authors**: Yile Liu, Yixian Liu, Zongwei Li, Yufei Huang, Xinhua Feng, Zhichao Hu, Jinglu Hu, Jianfeng Yan, Fengzong Lian, Yuhong Liu  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2601.03205v1  

#### Abstract
While Large Language Models (LLMs) have demonstrated significant potential in natural language processing , complex general-purpose reasoning requiring multi-step logic, planning, and verification remains a critical bottleneck. Although Reinforcement Learning with Verifiable Rewards (RLVR) has succe...

---

### 24. [Scalable Tree Ensemble Proximities in Python](https://arxiv.org/abs/2601.02735)

**Authors**: Adrien Aumon, Guy Wolf, Kevin R. Moon, Jake S. Rhodes  
**Category**: cs.LG  
**Published**: 2026-01-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2601.02735v1  

#### Abstract
Tree ensemble methods such as Random Forests naturally induce supervised similarity measures through their decision tree structure, but existing implementations of proximities derived from tree ensembles typically suffer from quadratic time or memory complexity, limiting their scalability. In this w...

---

### 25. [Universal Conditional Logic: A Formal Language for Prompt Engineering](https://arxiv.org/abs/2601.00880)

**Authors**: Anthony Mikinka  
**Category**: cs.AI  
**Published**: 2026-01-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.00880v1  

#### Abstract
We present Universal Conditional Logic (UCL), a mathematical framework for prompt optimization that transforms prompt engineering from heuristic practice into systematic optimization. Through systematic evaluation (N=305, 11 models, 4 iterations), we demonstrate significant token reduction (29.8%, t...

---

### 26. [PCEval: A Benchmark for Evaluating Physical Computing Capabilities of Large Language Models](https://arxiv.org/abs/2601.02404)

**Authors**: Inpyo Song, Eunji Jeon, Jangwon Lee  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.02404v1  

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains, including software development, education, and technical assistance. Among these, software development is one of the key areas where LLMs are increasingly adopted. However, when hardware constraints are co...

---

### 27. [Towards Comprehensive Stage-wise Benchmarking of Large Language Models in Fact-Checking](https://arxiv.org/abs/2601.02669)

**Authors**: Hongzhan Lin, Zixin Chen, Zhiqi Shen, Ziyang Luo, Zhen Ye, Jing Ma, Tat-Seng Chua, Guandong Xu  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.02669v1  

#### Abstract
Large Language Models (LLMs) are increasingly deployed in real-world fact-checking systems, yet existing evaluations focus predominantly on claim verification and overlook the broader fact-checking workflow, including claim extraction and evidence retrieval. This narrow focus prevents current benchm...

---

### 28. [Mechanistic Knobs in LLMs: Retrieving and Steering High-Order Semantic Features via Sparse Autoencoders](https://arxiv.org/abs/2601.02978)

**Authors**: Ruikang Zhang, Shuo Wang, Qi Su  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.02978v1  

#### Abstract
Recent work in Mechanistic Interpretability (MI) has enabled the identification and intervention of internal features in Large Language Models (LLMs). However, a persistent challenge lies in linking such internal features to the reliable control of complex, behavior-level semantic attributes in lang...

---

### 29. [Dementia-R1: Reinforced Pretraining and Reasoning from Unstructured Clinical Notes for Real-World Dementia Prognosis](https://arxiv.org/abs/2601.03018)

**Authors**: Choonghan Kim, Hyunmin Hwang, Hangeol Chang, Jaemin Kim, Jinse Park, Jae-Sung Lim, Jong Chul Ye  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.03018v1  

#### Abstract
While Large Language Models (LLMs) have shown strong performance on clinical text understanding, they struggle with longitudinal prediction tasks such as dementia prognosis, which require reasoning over complex, non-monotonic symptom trajectories across multiple visits. Standard supervised training ...

---

### 30. [DIP: Dynamic In-Context Planner For Diffusion Language Models](https://arxiv.org/abs/2601.03199)

**Authors**: Yang Li, Han Meng, Chenan Wang, Haipeng Chen  
**Category**: cs.CL  
**Published**: 2026-01-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.03199v1  

#### Abstract
Diffusion language models (DLMs) have shown strong potential for general natural language tasks with in-context examples. However, due to the bidirectional attention mechanism, DLMs incur substantial computational cost as context length increases. This work addresses this issue with a key discovery:...

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
