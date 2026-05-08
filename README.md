# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-08 07:14:53 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Requests of a Feather Must Flock Together: Batch Size vs. Prefix Homogeneity in LLM Inference](https://arxiv.org/abs/2605.06046)

**Authors**: Saksham Rathi,  Preeti, Mythili Vutukuru  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2605.06046v1  

#### Abstract
Auto-regressive token generation in large language models is memory-bound because it requires "attending to" key and value tensors (KV cache) of all previous tokens. Prior work aims to improve the efficiency of this decode process by batching multiple requests together, and maximizing batch size sub...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Requests of a Feather Must Flock Together: Batch Size vs. Prefix Homogeneity in LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 LLM 推理调度器（如 vLLM、SGLang）在 decode 阶段主要追求**最大化 batch size**以提高 GPU 利用率。然而，这些方法忽略了另一个关键因素——**prefix homogeneity**（前缀同质性），即一个 batch 中所有请求是否共享相同的 prompt 前缀。

作者指出：
- 即使是极小的 prefix 差异也会破坏 **KV cache 的空间与时间局部性**，导致内存带宽利用率下降。
- 现有基于 radix tree 的 prefix detection 方法存在严重的 **CPU 开销**，有时甚至超过 GPU 执行时间。

因此，单纯追求大 batch 并不能带来最优吞吐量，尤其是在 prefix 共享型 workload 下。

---

### 🚀 提出的新方法与创新思路

论文提出 **FEATHER** ——一种新型的 prefix-aware 调度器，其核心思想是：  
> 在 batch size 和 prefix homogeneity 之间进行动态权衡，而非一味追求最大 batch。

#### 主要组件：

| 组件 | 功能 |
|------|------|
| **Chunked Hash Tree (CHT)** | 一种轻量级数据结构，用于快速检测请求间的 prefix 共享程度，避免昂贵的 radix tree 遍历 |
| **Reinforcement Learning (RL) Policy** | 学习何时停止向 batch 添加新请求，以最大化 end-to-end throughput |

---

### 🔍 相比现有方法的优势

| 方面 | FEATHER 的优势 |
|------|----------------|
| **性能** | 实现 **2–10× 更高的端到端吞吐量**（end-to-end throughput） |
| **效率** | CHT 将 prefix detection 的 CPU 开销降低 **高达 1000×**，远低于 DFS-based 方法 |
| **适应性** | RL 政策能自适应不同 workload 和硬件配置，无需手动调参 |
| **兼容性** | 完全硬件无关，可集成进 vLLM 和 SGLang 等主流框架 |
| **鲁棒性** | 当 workload 无 prefix 共享时，性能不劣于 FCFS；当有共享时则显著提升 |

---

## 2. 核心实验方法和设置

### 📚 数据集与工作负载
- **L-Eval**：真实的人工标注问答对，涵盖摘要、QA 等任务，序列长度从 2.7K 到 210.5K tokens。
- **LongBench**：长上下文基准测试，包含多文档 QA、代码补全等，context length 在 4K–10K tokens。
- 自定义合成 workload：
  - 控制 prefix 数量（1～100 groups）
  - 控制 shared prefix 长度（1K～10K tokens）
  - 控制 decode 长度（1～200 tokens）
  - 请求到达建模为 Poisson 过程

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| **模型** | Llama-3 8B, Qwen 0.5B/1.5B/8B, LongChat 13B |
| **GPU** | NVIDIA RTX 6000 Ada（48GB GDDR6, 96MB L2 cache）<br>A100-80GB（用于与 PAT 对比） |
| **推理引擎** | 集成于 vLLM 和 SGLang |
| **最大 batch size** | 默认 500（受限于 GPU 内存） |
| **KV cache 管理** | 使用 PagedAttention |

---

### 📊 评估指标

| 指标 | 含义 |
|------|------|
| **Throughput (toks/s)** | 输出 token 数每秒，衡量系统容量 |
| **Time Between Tokens (TBT)** | 解码两个 token 之间的平均延迟，反映响应速度 |
| **Average Batch Size** | 每次 forward pass 的平均请求数，关联调度策略 |
| **DRAM Bandwidth Utilization** | GPU 显存带宽使用率，反映内存访问效率 |
| **Scheduler CPU Overhead** | 调度决策所消耗的 CPU 时间 |

---

### 🆚 基线方法对比

| 基线 | 描述 |
|------|------|
| **vLLM FCFS** | 先来先服务，默认策略 |
| **Dynamic Batching** | 根据内存和延迟动态调整 batch size |
| **SGLang FCFS/LPM/DFS-W** | 支持 prefix-aware 调度：<br>- LPM: 最长前缀匹配<br>- DFS-W: 深度优先遍历加权分支 |
| **PAT [41]** | Prefix-aware attention kernel，需定制化 GPU kernel |
| **Oracle** | 理想情况：直接提供 prefix group ID，无 detection 开销 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 场景 | FEATHER 表现 |
|------|-------------|
| **标准 workload**<br>(5 prefix groups, 5K tokens, 100 req/s) | 达到 **~4× vLLM FCFS 吞吐量** |
| **高 prefix 分组数**<br>(100 groups) | 达到 **10× vLLM FCFS 吞吐量** |
| **长 decode 长度**<br>(200 tokens) | 达到 **~3.8K toks/s**，超第二名 3× |
| **LongChat 13B 模型** | 最高实现 **22× 吞吐提升**（vs FCFS） |
| **CPU 开销** | 调度开销始终低于 GPU 执行时间的 **1%**，而 SGLang LPM/DFS-W 可达 **10× GPU 时间** |

> 图 12(a) 显示，在所有请求速率下，FEATHER 均优于所有 baseline。

---

### 🔁 与基线方法的对比结果

| 对比项 | 结果 |
|--------|------|
| **vs vLLM FCFS** | 吞吐量提升 2–10×，TBT 更低 |
| **vs SGLang 系列** | 批处理更高效，batch size 更稳定，TBT 更优 |
| **vs PAT (kernel-level 优化)** | 性能相当甚至更好，且无需修改 kernel，更具通用性 |
| **vs Dynamic Batching** | 在 prefix 共享场景下大幅领先；无共享时持平 |

> 特别值得注意的是：FEATHER 的平均 batch size **通常小于 FCFS**，但仍取得更高 throughput，说明“小而同质”的 batch 比“大而异构”的 batch 更优。

---

### 🔍 消融实验结果

#### ✅ 替换 CHT 为 Radix Tree
- 吞吐量显著下降
- TBT 上升 → GPU 因调度延迟频繁空转
- 证明 **CHT 对降低 CPU 开销至关重要**

#### ✅ 移除 RL Policy（仅用 CHT + 贪心添加）
- 在低负载时接近 FCFS
- 在中高负载时无法及时停止 batch 构造，错过最佳同质 batch 时机
- 证明 **RL 能智能地在“利用”与“探索”间平衡**

#### ✅ 不同 chunk size 影响
- CHT 对 chunk size 不敏感（即使 chunk=1，也优于传统 radix tree）
- 大 chunk 减少哈希计算，但可能损失精度
- 推荐使用 **K=512 或 1024 tokens/chunk**

#### ✅ 无 prefix 共享 workload
- FEATHER 性能与 FCFS 相当，不会退化
- 证明其 **鲁棒性强，适用于各类 workload**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Prefix Homogeneity 是 decode 阶段的关键性能决定因素**
   - 即使只有一个请求偏离共同前缀，也会导致 **throughput 下降近 2×**
   - 同质 batch 提供更强的 **空间与时间局部性**，提升 DRAM 带宽利用率 >40%

2. **存在 batch size 与 prefix homogeneity 的权衡**
   - 太小的 batch 导致 compute under-utilization
   - 太大的 batch 引入 heterogeneity，破坏 locality
   - **适度大小的同质 batch 可胜过更大的异构 batch**

3. **现有 prefix detection 方法开销巨大**
   - SGLang 的 DFS-W / LPM 调度开销可达 GPU 时间的 **50–90%**
   - radix tree traversal 成为瓶颈，尤其在长 prompt 和高并发下

4. **FEATHER 实现了软硬协同设计**
   - 通过 CHT 实现 **低开销 prefix detection**
   - 通过 RL 实现 **自适应 stopping decision**
   - 二者结合实现了 **2–10× 的端到端加速**

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖 prefix 共享型 workload** | 若所有请求完全独立，则优势消失（但也不会变差） |
| **目前为单 GPU 设计** | 尚未扩展至 multi-GPU 或分布式环境 |
| **RL 收敛需要一定训练过程** | 在极端突变 workload 下可能存在短暂次优 |
| **chunk size 需经验设定** | 虽然鲁棒，但最优值仍依赖 workload 特征 |

---

### 🔮 未来工作方向

1. **扩展至 multi-GPU 和 distributed setting**
   - 跨节点协调 prefix 共享状态
   - 支持 disaggregated prefill/decode 架构（如 DistServe）

2. **更轻量的学习策略**
   - 探索无需训练的在线自适应算法
   - 减少对历史数据的依赖

3. **与 kernel-level 优化深度耦合**
   - 与 PAT、FlashAttention 等 prefix-aware kernel 联合优化
   - 实现 scheduler-kernel co-design

4. **支持动态 chunk size 调整**
   - 根据 workload 自动选择最优 chunk granularity

5. **开源计划**
   - 作者承诺将 FEATHER 贡献给 vLLM 和 SGLang 官方仓库

---

> 💬 **一句话总结**：  
> FEATHER 揭示了一个被忽视的真相：**在 LLM 推理中，“物以类聚”比“人多力量大”更重要**。它通过轻量化的 CHT 和智能的 RL 调度，在保持低开销的同时，实现了前所未有的 decode 吞吐提升。

</details>

---

### 2. [MDN: Parallelizing Stepwise Momentum for Delta Linear Attention](https://arxiv.org/abs/2605.05838)

**Authors**: Yulong Huang, Xiang Liu, Hongxiang Huang, Xiaopeng Lin, Zunchang Liu, Xiaowen Chu, Zeke Xie, Bojun Cheng  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2605.05838v1  

#### Abstract
Linear Attention (LA) offers a promising paradigm for scaling large language models (LLMs) to long sequences by avoiding the quadratic complexity of self-attention. Recent LA models such as Mamba2 and GDN interpret linear recurrences as closed-form online stochastic gradient descent (SGD), but naive...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MDN: Parallelizing Stepwise Momentum for Delta Linear Attention

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前主流的 **Linear Attention (LA)** 模型（如 Mamba2、GDN）虽然避免了传统 **Self-Attention** 的 $O(L^2)$ 复杂度，实现了 $O(L)$ 时间复杂度，但其更新机制基于 **naive SGD**，存在以下问题：

- **信息衰减过快**：历史信息在更新中迅速丢失。
- **优化轨迹不稳定**：对梯度噪声敏感，收敛效果不佳。
- **表达能力受限**：无法有效建模长程依赖和振荡模式。

尽管 **momentum-based optimizers** 在优化领域已被证明能提升稳定性和表达力，但在 LA 中难以高效并行化训练，尤其是保持严格因果性的 **stepwise momentum** 因其串行性而无法用于大规模预训练。

### **提出了什么新方法或新思路**

本文提出 **Momentum DeltaNet (MDN)**，其核心创新如下：

#### ✅ **Stepwise Momentum 并行化算法**
- 将 stepwise momentum 更新规则通过 **几何重排序（geometric reordering）** 解耦更新系数，实现 **chunkwise parallel** 训练。
- 该算法在保持严格因果性的同时，实现了高效的训练吞吐。

#### ✅ **动力系统视角下的稳定性分析**
- 将 momentum 机制建模为 **二阶动态系统（second-order dynamical system）**，揭示其引入了 **复共轭特征值（complex conjugate eigenvalues）**，从而支持 **阻尼振荡行为（damped oscillations）**。
- 这种机制增强了模型的表达能力，使其能够捕捉相位感知的记忆模式。

#### ✅ **稳定的门控约束设计**
- 分析发现，无约束的 momentum 参数可能导致 **负实部特征值**，引发数值不稳定（如 NaN）。
- 提出 **象限约束（quadrant constraint）**：通过限制 $\beta \leq 1 - \alpha$ 和 $\mu \in (e^{-1}, 1)$，确保特征值位于右半平面，保证训练稳定性。

#### ✅ **高效实现**
- 基于 **Triton kernels** 实现，训练效率与 Mamba2、KDA 相当，推理延迟与 GDN、Comba 接近。

### **相比现有方法的优势**

| 方面 | MDN 优势 |
|------|---------|
| **表达能力** | 引入 momentum 后支持振荡动态，超越传统一阶 LA 模型的实数衰减动态 |
| **训练效率** | chunkwise parallel 算法实现高效训练，避免 blockwise momentum 的训练-推理不一致 |
| **推理一致性** | 保持 stepwise 更新，确保训练与推理完全一致 |
| **稳定性** | 通过理论驱动的门控约束，保障大规模训练的鲁棒性 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **合成任务**：
  - **MQAR (Multi-Query Associative Recall)**：评估 in-context retrieval 能力。
- **语言建模与下游任务**：
  - **WikiText-4K**, **LAMBADA**：评估困惑度（PPL）和常识推理。
  - **Commonsense Reasoning Tasks**：ARC-e, ARC-c, PIQA, WinoGrande, BoolQ, SciQ。
  - **In-context Retrieval Tasks**：FDA, SWDE, SQuAD, NQ, TQA, DROP。
- **长上下文任务**：
  - **LongBench**：在 16K 长度上评估代码、摘要、问答等任务。
  - **NIAH (Needle-In-A-Haystack)**：从 RULER 基准评估长程检索能力。

### **实验设置**

- **模型规模**：400M 和 1.3B 参数。
- **训练配置**：
  - 序列长度：4K。
  - 训练 token 数：400M 模型训练 15B，1.3B 模型训练 100B。
  - 优化器：AdamW，学习率余弦退火，峰值 $3\times10^{-4}$。
  - 数据集：SlimPajama 子集（100B tokens）。
- **评估方式**：
  - 所有模型在相同配置下训练，公平比较。
  - 使用 **FLA** 和 **FLAME** 框架实现。

### **基线方法对比**

| 模型 | 类型 | 特点 |
|------|------|------|
| **Transformer** | 自注意力 | 基线，高计算成本 |
| **Mamba2** | Decay Rule LA | 数据相关衰减 |
| **GDN** | Delta Rule LA | 支持值修正（value correction） |
| **Comba** | Delta Rule + Query Correction | 引入查询修正 |
| **KDA** | Vector-valued Delta Rule | 通道级门控，专为检索优化 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **语言建模性能（400M / 1.3B）**

| 模型 | Wiki PPL ↓ | Lamb PPL ↓ | Commonsense Avg ↑ | In-context Retrieval Avg ↑ |
|------|------------|------------|-------------------|----------------------------|
| **Transformer** | 32.80 / 18.99 | 54.36 / 17.90 | 48.06 / 57.95 | 30.42 / 40.60 |
| **Mamba2** | 33.45 / 19.14 | 60.42 / 18.20 | 47.85 / 57.10 | 30.42 / 32.45 |
| **GDN** | 32.10 / 18.51 | 45.63 / 16.12 | 47.85 / 57.44 | 21.12 / 32.80 |
| **Comba** | 31.73 / 18.37 | 46.19 / 16.83 | 48.91 / 58.49 | 22.93 / 35.25 |
| **KDA** | 31.96 / 19.24 | 43.44 / 14.87 | 49.06 / 58.56 | 24.47 / 32.62 |
| **MDN (Ours)** | **31.51 / 18.03** | **41.62 / 14.87** | **49.42 / 58.82** | **26.76 / 36.14** |

> **结论**：MDN 在 **困惑度** 和 **常识推理** 上均达到最优，在 **in-context retrieval** 上显著优于其他 LA 模型。

#### ✅ **长上下文建模（LongBench, 1.3B）**

| 模型 | Code ↑ | Summarization ↑ | SingleQA ↑ | MultiQA ↑ | **Avg ↑** |
|------|--------|------------------|-------------|------------|----------|
| **Transformer** | 8.35 | — | — | — | 8.35 |
| **Mamba2** | 15.00 | — | — | — | 15.00 |
| **GDN** | 19.28 | — | — | — | 19.28 |
| **Comba** | 18.11 | — | — | — | 18.11 |
| **KDA** | 18.62 | — | — | — | 18.62 |
| **MDN (Ours)** | **20.18** | — | — | — | **20.18** |

> **结论**：MDN 在 **代码生成** 和 **摘要** 任务上表现尤为突出。

#### ✅ **长程检索（NIAH, 8K context）**

在多针（multi-needle）设置下，MDN 显著优于最强基线：

| 任务 | MDN | 最强基线 | 提升 |
|------|-----|----------|------|
| MK-NIAH | **38.60** | 25.20 (KDA) | +13.40 |
| MQ-NIAH | **35.15** | 23.70 (KDA) | +11.45 |
| MV-NIAH | **27.60** | 18.65 (KDA) | +8.95 |

> **结论**：MDN 在超长上下文检索中具有显著优势。

### **消融实验结果**

| 变体 | Wiki PPL | Retrieval Avg | 说明 |
|------|----------|----------------|------|
| **MDN (完整)** | 31.51 | 26.76 | 基线 |
| w/o Output Corr. | 31.72 | 25.52 | 仍优于 GDN，说明 momentum 是主因 |
| w/o Momentum | 32.11 | 20.12 | 性能大幅下降，验证 momentum 有效性 |
| w/o Clamp min | NaN | — | 无下界导致训练发散 |
| w/o $\alpha_{\text{max}}$ | NaN | — | 无约束导致不稳定 |
| w/o $\beta_{\text{max}}$ | 31.52 | 26.40 | 性能轻微下降 |
| $\mu = 2\cdot\text{sigmoid}(\cdot)$ | 31.89 | 25.54 | 激活函数影响性能 |

> **结论**：momentum 和稳定性约束是 MDN 成功的关键。

---

## 4. 关键结论和发现

### **主要发现**

1. **Stepwise Momentum 显著提升 LA 表达能力**：
   - 通过引入 momentum，MDN 能够建模更复杂的动态（如振荡），增强记忆鲁棒性。
2. **并行化与因果性可以兼得**：
   - 通过几何重排序，首次实现了 stepwise momentum 的高效 chunkwise parallel 训练。
3. **稳定性需理论指导**：
   - 二阶系统分析揭示了负实部特征值的风险，提出的象限约束有效防止训练崩溃。
4. **性能全面领先**：
   - 在困惑度、常识推理、长上下文建模和检索任务上，MDN 一致优于 Transformer、Mamba2、GDN、Comba 和 KDA。

### **方法的局限性**

1. **训练吞吐略低于 GDN/Comba**：
   - 由于维护 momentum state 和 correction value，内存开销略高，训练 throughput 低于部分一阶 LA 模型。
2. **未在更大规模（如 7B+）验证**：
   - 当前实验最大为 1.3B，更大模型上的扩展性有待验证。
3. **混合架构探索有限**：
   - 仅测试了 3:1 和 7:1 的 linear/full-attention 比例，更优的混合策略可能进一步提升效率。

### **未来工作方向**

1. **优化 kernel 实现**：
   - 开发更高效的 Triton kernels，减少 memory overhead，提升训练 throughput。
2. **探索更先进的优化器**：
   - 将 Nesterov Momentum、Adam 等更复杂优化器融入 LA 框架。
3. **扩展到多模态**：
   - 将 MDN 应用于语音、视频、基因组等长序列建模任务。
4. **深入研究混合架构**：
   - 系统研究不同层中 linear/full-attention 的放置策略，构建更高效的 hybrid LLM。

---

> **总结**：MDN 通过将 **stepwise momentum** 成功并行化，并结合 **动力系统分析** 设计稳定门控，显著提升了 Linear Attention 的表达能力和实际性能，为构建更高效、更强健的长序列模型提供了新范式。

</details>

---

### 3. [UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification](https://arxiv.org/abs/2605.06221)

**Authors**: Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, Ran He  
**Category**: cs.CL  
**Published**: 2026-05-08  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.06221v1  

#### Abstract
As large language models (LLMs) continue to advance rapidly, they are becoming increasingly capable while simultaneously demanding ever-longer context lengths. To improve the inference efficiency of long-context processing, several novel low-complexity hybrid architectures have recently been propose...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **long-context LLM 推理加速**研究集中于 **sparse attention** 方法（如 MInference、FlexPrefill），这些方法存在两大局限性：

1. **架构依赖性强**：仅在全 attention 架构上有效，在新兴的 **hybrid 架构**（如 linear/full 或 sliding window/full attention 混合）中收益显著下降。
2. **不兼容连续批处理（continuous batching）**：无法集成到现代推理引擎（如 vLLM）中，限制了其生产部署能力。

### 🚀 提出的新方法：UniPrefill
提出 **UniPrefill**，一种**架构无关（architecture-agnostic）** 的 prefill 加速框架，核心思想是：

- 在 **full attention 层**进行 **token importance 估计**，通过 **block-wise top-p selection** 动态筛选重要 token。
- 将稀疏性（sparsity）**跨所有后续层传播**（包括 linear attention、FFN 等），实现 **attention 和 GEMM FLOPs 的同步减少**。
- 在 **token 级别**而非 attention 子层级别进行加速，提升整体计算效率。

### 🔍 相比现有方法的优势
| 维度 | 传统 sparse attention 方法 | UniPrefill |
|------|----------------------------|----------|
| **适用架构** | 仅全 attention 有效 | 支持全 attention、linear/full、sliding window/full 等混合架构 |
| **加速范围** | 仅减少 attention FLOPs | 同时减少 attention 和 GEMM FLOPs |
| **系统集成** | 不兼容 continuous batching | 深度集成至 vLLM，支持 prefill-decode co-processing 和 tensor parallel |
| **扩展性** | 加速效果随上下文增长趋于饱和 | 加速比随并发请求增加而提升 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **RULER** [11]：一个全面的 long-context benchmark，涵盖检索、多跳推理、聚合、问答等任务，支持从 4K 到 128K 的上下文长度配置。

### ⚙️ 实验设置
- **模型架构**：
  - `LLaMA-3.1-8B-Instruct`：纯 full attention 架构
  - `Qwen3-Next-80B-A3B`：linear/full attention 混合（比例 3:1）
  - `Gemma-3-12B`：sliding window/full attention 混合（比例 5:1）
- **上下文长度**：4K, 8K, 16K, 32K, 64K, 128K
- **批大小（BSZ）**：1, 4, 16, 64
- **硬件设置**：基于 **vLLM v0.16.0** 实现，使用 **TP=8**（tensor parallelism），CUDA 12.8
- **关键参数**：
  - block size $ G = 64 $
  - 查询窗口长度 $ n = 128 $
  - top-p 阈值：0.99（LLaMA）、0.99（Qwen）、0.98（Gemma）
  - 前 128 个 token 强制保留（attention sinks）

### 🎯 评估指标
- **准确性**：RULER benchmark 上的平均得分
- **效率**：
  - **Time-To-First-Token (TTFT)** 加速比
  - **Prefill throughput**（单位：tokens/s）
- **对比方法**：
  - Baseline（标准 prefill）
  - LazyLLM [9]
  - SlimInfer [20]
  - MInference [13]
  - FlexPrefill [16]
  - XAttention [31]
  - ProxyAttn [27]

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 和 Figure 1）

#### ✅ TTFT 加速比（128K 上下文）
| 方法 | LLaMA-3.1-8B | Qwen3-Next-80B | Gemma-3-12B |
|------|--------------|----------------|-------------|
| **UniPrefill** | **2.26×** | **1.68×** | **1.49×** |
| MInference | 1.34× | 1.05× | 1.03× |
| FlexPrefill | 1.46× | 1.08× | 1.04× |
| LazyLLM | 2.51× | 1.74× | 1.64× |

> 💡 注意：虽然 LazyLLM 在纯速度上有一定优势，但其 **accuracy 下降明显**（见下表），而 UniPrefill 在保持高 accuracy 的前提下实现高效加速。

#### ✅ 准确性（RULER Score @ 128K）
| 方法 | LLaMA-3.1-8B | Qwen3-Next-80B | Gemma-3-12B |
|------|--------------|----------------|-------------|
| **Baseline** | 76.89 | 92.09 | 61.22 |
| **UniPrefill** | **79.87** (+3.9%) | **91.41** (-0.68) | **58.38** (-2.84) |
| LazyLLM | 49.71 (-27.18) | 55.17 (-36.92) | 43.38 (-17.84) |
| MInference | 78.21 (+1.32) | 91.81 (-0.28) | 59.31 (-1.91) |

> ✅ UniPrefill 在绝大多数情况下 **accuracy 损失极小甚至略有提升**，远优于 LazyLLM/SlimInfer 等 token pruning 方法。

#### ✅ Prefill Throughput 提升（Table 2，TP=8）
在 **LLaMA-3.1-8B** 上，随着 batch size 和 context length 增加，加速效果显著增强：

| Context | BSZ | Standard Prefill (t/s) | UniPrefill (t/s) | **Speedup** |
|--------|-----|------------------------|------------------|------------|
| 128K | 1 | 30,324 | 63,762 | **+110%** |
| 128K | 4 | 30,812 | 68,721 | **+123%** |
| 128K | 16 | 30,834 | 72,139 | **+134%** |
| 128K | 64 | 33,489 | 72,139 | **+115%** |

> 🔥 最高实现 **+134% 的吞吐提升**，且加速比随并发请求增长而上升。

### 🔍 消融实验结果（Ablation Study）

#### 📏 Block Size $ G $ 影响（Table 3）
- $ G=64 $：默认选择，平衡了 selection 开销与 drop 粒度。
- $ G=32 $：细粒度 drop 更多 token，长上下文下吞吐更高（128K 时 +121%）。
- $ G=128 $：短上下文更优，但长上下文 drop 不够精细。

#### 🔢 查询窗口长度 $ n $ 影响（Table 4）
- $ n=32 $：importance 估计方差大，accuracy 显著下降（↓2.7 pts）。
- $ n=512 $：accuracy 恢复但计算开销增加。
- $ n=128 $：最佳平衡点，被选为默认值。

#### 🎯 Random Seed 鲁棒性（Table 5）
- 在多个随机种子下性能稳定，表明方法对初始化不敏感。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **UniPrefill 是首个真正通用的 prefill 加速器**，适用于全 attention 和各类 hybrid 架构。
2. 通过 **block-wise token dropping + 跨层稀疏传播**，同时降低 attention 与 GEMM 计算量，突破了 sparse attention 的局限。
3. **深度集成 vLLM**，支持 continuous batching、tensor parallel 和 prefill-decode co-processing，具备生产级部署能力。
4. 在 **高并发、长上下文场景**下加速效果最显著，非常适合真实服务负载。
5. **accuracy 损失可忽略**，在多数情况下甚至优于 baseline，显著优于其他 token pruning 方法。

### ⚠️ 方法的局限性
- 当前聚焦于 **prefill 阶段加速**，未涉及 decoding 阶段优化。
- 对 extremely sparse attention 分布可能过度剪枝，需谨慎设置 top-p 阈值。
- 依赖 full attention 层进行 importance 估计，在完全无 full attention 的模型中不可用。

### 🔮 未来工作方向
- 扩展至 **decoding 阶段加速**，实现端到端推理优化。
- 探索 **训练时联合优化**，使模型更适应动态 token dropping。
- 结合 **KV cache 压缩**（如 SnapKV）进一步节省内存。
- 探索 **adaptive top-p** 策略，根据输入动态调整保留率。

---

> **GitHub 地址**：[https://github.com/qhfan/UniPrefill.git](https://github.com/qhfan/UniPrefill.git)  
> **一句话总结**：UniPrefill 通过 **block-wise 动态稀疏化 + 跨层传播**，实现了**架构通用、系统友好、高效准确**的 long-context prefill 加速，最高可达 **2.1× TTFT 加速**，是迈向高效 LLM 推理的重要一步。

</details>

---

### 4. [CCL-Bench 1.0: A Trace-Based Benchmark for LLM Infrastructure](https://arxiv.org/abs/2605.06544)

**Authors**: Eric Ding, Byungsoo Oh, Bhaskar Kataria, Kaiwen Guo, Jelena Gvero, Abhishek Vijaya Kumar, Arjun Devraj, Lindsey Bowen, Atharv Sonwane, Emaad Manzoor, Rachee Singh  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.06544v1  

#### Abstract
Evaluative claims about LLM infrastructure -- ``workload X is fastest on hardware Y with software Z'' -- depend on a complex configuration space spanning hardware accelerators, interconnect bandwidth, software frameworks, parallelism plans, and communication libraries. Current infrastructure evaluat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CCL-Bench 1.0: A Trace-Based Benchmark for LLM Infrastructure

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前的 **LLM 基础设施基准测试**（如 MLPerf、LLM-Perf）存在三大根本性局限：

- **仅报告结果，不提供解释**：只发布端到端的性能数字（如 step time、MFU），无法解释为何某个配置更快。
- **评估指标固化**：一旦实验完成，无法再用新指标分析旧数据，必须重新运行实验。
- **调优过程不可见**：未记录配置搜索路径，导致“最优配置”难以复现，且不同框架间的比较不公平。

这些问题使得硬件厂商、软件开发者和生产运维人员难以定位瓶颈、优化系统或做出可靠的技术选型决策。

---

### 提出了什么新方法或新思路

论文提出了 **CCL-Bench**，一个基于执行轨迹（execution trace）的 LLM 基础设施基准测试框架，其核心是 **从“outcome benchmarking”转向“explanation benchmarking”**。

#### 主要创新点：

- ✅ **证据驱动的提交范式（Evidence-based Schema）**  
  每次提交包含三个可复现的组件：
  - **Execution Trace**：细粒度操作符、内核、通信事件的时间序列（来自 Kineto 或 XProf）。
  - **Workload Card（YAML）**：标准化描述模型、硬件、并行策略、框架等完整上下文。
  - **Run Scripts**：启动脚本，确保可复现性。

- ✅ **社区可扩展的度量工具包（Community-Extensible Metric Toolkit）**  
  开源工具库支持从 trace 中计算多种细粒度指标（如 MFU、compute-communication overlap、memory-transfer overhead 等），新工具可对历史 trace 进行回溯分析。

- ✅ **支持下游高级分析插件**
  - **Post-hoc metric extension**：新增指标无需重跑实验。
  - **Trace-driven what-if analysis**：将 trace 转为 Chakra 图，输入 Astra-Sim 等模拟器预测带宽升级效果。
  - **自动化配置优化（CCL-Search）**：基于 LLM Agent 的自动调参系统，迭代探索最优配置，并全程记录为 benchmark entry。

---

### 相比现有方法的优势

| 维度 | 传统 Benchmark（如 MLPerf） | CCL-Bench |
|------|-------------------------------|----------|
| 可解释性 | ❌ 仅有 summary 数字 | ✅ 支持归因分析（compute vs comm vs memory） |
| 灵活性 | ❌ 指标固定 | ✅ 新 metric 可 retroactively 应用于所有历史 trace |
| 复现性 | ❌ 配置搜索路径缺失 | ✅ 完整记录调优过程（via CCL-Search） |
| 成本 | ❌ 每次新分析需重跑 | ✅ 分析可在 trace 上离线进行 |
| 社区共建 | ❌ 封闭榜单 | ✅ 开放提交、工具贡献、争议可公开讨论 |

---

## 2. 核心实验方法和设置

### 使用的模型与工作负载（Workloads）

CCL-Bench 1.0 包含以下代表性 workload（见 Table 4）：

| ID | Model | Phase | Batch Size | Sequence/Input Length |
|----|-------|--------|------------|------------------------|
| WL1 | Qwen3-4B | Inference | 128 | 1024 input |
| WL2 | Llama-3.1-8B | Inference | 128 | 1024 input |
| WL3 | DeepSeek-MoE-16B | Inference | 128 | 1024 input |
| WL4 | Llama-3.1-8B | Training | 4 | 512 sequence |
| WL5 | DeepSeek-V3-16B | Training | 8 | 1024 sequence |
| WL6 | DeepSeek-V3-16B | Training | 64 | 2048 sequence |
| WL7 | DeepSeek-V3-236B | Training | 64 | 1024 sequence |

覆盖了 dense 和 MoE 架构、训练与推理场景。

---

### 实验平台与硬件环境

- **GPU 平台**：NERSC Perlmutter 超算
  - 每节点 4× A100 GPU
  - NVLink 3.0（300 GB/s 单向，scale-up）
  - Slingshot-11（200 Gbps，scale-out）

- **TPU 平台**：Google TPU v6e
  - 8 chips/node，32 nodes/pod
  - 2D Torus 拓扑
  - ICI（Inter-Chip Interconnect）带宽 100 GB/s

---

### 评估指标（Metrics）

CCL-Bench 工具包支持多维度细粒度指标：

| 类别 | 指标 | 描述 |
|------|------|------|
| **Model Execution** | `avg_step_time`, `MFU` | 端到端步长时间、模型 FLOPs 利用率 |
| **Compute** | `SM_coverage`, `compute_bound_fraction` | 计算单元利用率、是否计算密集 |
| **Memory** | `memory_transfer_overhead` | 显存拷贝开销占比 |
| **Communication** | `comm_fraction`, `compute_comm_overlap`, `AllReduce BW` | 通信时间占比、与计算重叠程度、集体通信带宽 |
| **Utility** | `scale_up_bw_utility` | 带宽翻倍后的 step time 改善百分比（via Astra-Sim 模拟） |

---

### 基线方法对比

CCL-Bench 不依赖单一 baseline，而是支持灵活的对比模式：

- **Cross-System Comparison**：固定硬件与 workload，比较不同软件栈（如 NCCL vs MSCCL++，vLLM vs SGLang）
- **Cross-Architecture Comparison**：相同预算下比较 GPU vs TPU 表现
- **Auto-Tuning Comparison**：使用 CCL-Search 自动寻找各框架下的最优配置

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### Claim 1: 更高的 compute-communication overlap 不一定降低 step time

在 DeepSeek-V3-16B MoE 训练中（WL5）：
- **高 overlap 配置（EP=4）**：overlap 达 83.67%，但 step time 为 **12.5s**
- **低 overlap 配置（EP=8）**：overlap 仅 41.2%，但 step time 降至 **10.0s**

🔍 **原因诊断**：  
EP=4 导致专家复制在 DP 域，引发大量 AllGather/ReduceScatter 流量（**29.1 GB → 817.9 GB**），尽管部分可重叠，但总通信体积过大反而拖慢整体。

> 📌 **结论**：overlap 是误导性指标；应联合分析 traffic volume。

---

#### Claim 2: TPU 互连带宽提升收益远高于 GPU

通过 trace 驱动的 what-if 分析（Astra-Sim）得出：

| 工作负载 | 平台 | 带宽翻倍效用（Utility） |
|---------|------|------------------------|
| WL1–WL3（小中规模推理） | TPU v6e | 最高达 **102.57×** GPU scale-up 带宽效用 |
| WL4–WL5（中等训练） | TPU v6e | 最高达 **22.82×** GPU scale-up 效用 |
| WL6–WL7（大规模训练） | GPU | 扩展 scale-up domain（扩大拓扑）比提升带宽更有效（+53.9%） |

> 📌 **结论**：TPU 的低带宽 + Torus 拓扑使其 ICI 成为关键瓶颈；而大模型 GPU 训练受限于 scale-out 网络。

---

#### Claim 3: 框架间最优配置不可迁移，差距可达 3×

在 Llama-3.1-8B 训练（WL4）上使用 CCL-Search 自动调优：

| 框架 | 最优配置 | Step Time | 若套用另一框架最优配置的表现 |
|------|----------|-----------|-------------------------------|
| **TorchTitan** | TP=1, DP=4, PP=4 | **1.50s** | 在 Megatron-LM 上运行得 **1.30s**（仍慢 3×） |
| **Megatron-LM** | TP=4, DP=1, PP=4 | **0.44s** | 在 TorchTitan 上运行得 **1.32s** |

> 📌 **结论**：并行策略高度依赖框架实现细节，跨框架直接迁移配置会严重损失性能。

---

### 消融实验结果（Ablation Studies）

- **通信库影响**（图9）：
  - MSCCL++ 相比 NCCL 在 MoE 推理中降低通信占比 **53.7% → 46.4%**，TPOT 从 81.2ms → 76.0ms。
- **推理引擎影响**（图10）：
  - vLLM 在 dense 模型上表现更好（更高 MFU），SGLang 在 MoE 上更优（更高效 AllToAll）。
- **TPU 张量并行扫描**（图11）：
  - TP 从 1→4 将 step time 从 222ms → 85ms；但 TP=8 回升至 98ms，表明过度并行有害。

---

## 4. 关键结论和发现

### 主要发现

1. 🔍 **Overlap ≠ 性能**：更高的 compute-communication overlap 可能伴随更大的 collective traffic volume，反而增加 step time。
2. ⚙️ **硬件投资回报差异显著**：
   - 对 TPU：提升 ICI 带宽性价比极高（尤其小中模型）。
   - 对 GPU：大规模训练更受益于扩大 scale-up domain（如全连接拓扑）。
3. 🔄 **配置不可移植**：最佳并行策略严重依赖框架实现，盲目迁移会导致高达 **3× 的性能损失**。
4. 🤖 **自动化调优可行且必要**：CCL-Search 能在 15 轮内将 step time 降低 **8–19×**，并生成可验证的优化路径。

---

### 方法的局限性

- **覆盖率有限**：目前仅支持开源模型、GPU/TPU、训练与批量推理，尚未覆盖在线服务、量化、稀疏化等场景。
- **存储成本高**：单次多卡 trace 可达数十 GB，需引入压缩或采样格式。
- **隐私与敏感信息**：trace 可能暴露集群拓扑、内部调度逻辑，限制企业公开分享意愿。
- **模拟精度依赖 trace 质量**：Astra-Sim 等模拟器的效果取决于 trace 是否准确反映真实行为。

---

### 未来工作方向

- 扩展支持更多硬件（如 Trainium、MTIA）、更大模型、在线推理、多模态任务。
- 构建 trace 压缩与匿名化工具，促进企业参与。
- 开发更多下游插件：如成本-延迟权衡分析、绿色 AI（能耗建模）、安全推理审计。
- 推动 trace 标准化成为行业共识，替代“黑箱榜单”。

---

> ✅ **项目地址**：  
> - GitHub: [https://github.com/cornell-sysphotonics/ccl-bench](https://github.com/cornell-sysphotonics/ccl-bench)  
> - 官网: [https://cclbench.ai/](https://cclbench.ai/)

</details>

---

### 5. [Event-Causal RAG: A Retrieval-Augmented Generation Framework for Long Video Reasoning in Complex Scenarios](https://arxiv.org/abs/2605.06185)

**Authors**: Peizheng Yan, Yu Zhao, Liang Xie, Juntong Qi, Mingming Wang, Erwei Yin  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.06185v1  

#### Abstract
Recent large vision-language models have achieved strong performance on short- and medium-length video understanding, yet they remain inadequate for ultra-long or even infinite video reasoning, where models must preserve coherent memory over extended durations and infer causal dependencies across te...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Event-Causal RAG**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前主流的 **Large Vision-Language Models (LVLMs)** 在短至中等长度视频理解上表现优异，但在**超长或无限时长视频推理**（infinite long-video reasoning）任务中面临三大核心挑战：
- **计算瓶颈**：Transformer 架构的 self-attention 复杂度为 $O(n^2)$，随视频长度增长迅速导致内存溢出（OOM）和推理不稳定。
- **语义碎片化**：传统 clip-based RAG 方法将视频按固定时间窗口切片，破坏了事件的语义完整性，导致因果链被割裂。
- **因果建模弱**：现有检索机制依赖语义相似性匹配，缺乏对**时间顺序**和**因果关系**的显式建模，难以支持多跳推理。

此外，存储开销和在线推理延迟也限制了其在真实场景（如监控、医疗监护）中的部署。

---

### **提出的新方法与创新思路**
本文提出 **Event-Causal RAG (EC-RAG)** ——一种轻量级、面向无限长视频推理的检索增强生成框架，核心创新如下：

#### ✅ **1. 事件驱动的语义分段（Dual-Sentinel Event Segmentation）**
- 不再使用固定时长切片，而是通过**视觉哨兵**（SigLIP 提取帧间相似性）和**音频哨兵**（ASR 转录语音）联合检测事件边界。
- 利用“中心向外扩展”策略捕获完整事件周期（从状态变化前到恢复稳定），确保每个片段是一个**语义完整的事件单元**。

#### ✅ **2. State-Event-State (SES) 图结构记忆**
- 将每个事件解析为结构化的三元组：  
  `Pre-State → Event → Post-State`  
  显式建模事件引发的状态变迁。
- 引入 `temporal_order` 属性保证时间逻辑一致性，防止因果幻觉。

#### ✅ **3. 双存储记忆系统（Dual-Store Memory）**
- **Vector DB**（如 Milvus）：用于节点语义嵌入与相似性检索。
- **Graph DB**（如 Neo4j）：维护事件间的拓扑连接（如 `:TEMPORAL_NEXT` 边），支持因果路径遍历。
- 支持基于语义和因果结构的双向检索。

#### ✅ **4. 双向图检索算法（Bidirectional Retrieval）**
- 先在 Vector DB 中定位语义锚点（entry anchoring）。
- 在 Graph DB 中进行双向 BFS 遍历（通常 N=2 hops），同时获取原因与结果。
- 最后通过**语义去重机制**（deduplication with $T_{dup}=0.85$）压缩冗余描述，提升信息密度。

#### ✅ **5. 轻量化与可部署性设计**
- 所有模块无需端到端训练，可即插即用集成于现有 VLM。
- 单块 RTX 5090（32GB VRAM）即可处理 24 小时连续流媒体。

---

### **相比现有方法的优势**
| 维度 | 传统 Clip-Based RAG / Full Context VLM | EC-RAG |
|------|----------------------------------------|-------|
| **记忆单位** | 固定时长 clip（易断裂事件） | 完整语义事件（保留因果结构） |
| **检索能力** | 仅语义匹配 | 支持因果链检索（topological + semantic） |
| **存储效率** | 高密度向量存储，成本高 | 结构化图存储，支持合并与压缩 |
| **推理能力** | 单跳或浅层推理 | 多跳因果推理（multi-event integration） |
| **硬件需求** | 多卡并行，KV-cache 爆炸 | 单卡运行，VRAM 消耗恒定 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
构建了一个多层次、跨尺度的评估体系，覆盖从秒级到天级的视频理解任务：

| 数据集 | 视频时长 | 特点 |
|--------|----------|------|
| **NExT-QA** | 5–180 秒 | 因果与时序动作推理，标准短/中视频 QA 基准 |
| **EventBench** | 60–1800 秒 | 多层次事件理解，强调上下文、情节性与反直觉推理 |
| **Video-MME Long** | 1800–3600 秒（30–60 分钟） | 极长视频，测试动态识别与动作推理能力 |
| **24-Hour Surveillance Stream** | >24 小时 | 真实工业级监控流，验证无限流处理能力 |

---

### **实验设置与评估指标**
- **主干模型**：Qwen3-VL-8B、VideoLlama3-7B、InternVL3.5-8B 等开源 VLM。
- **嵌入模型**：Qwen3-Embedding-4B。
- **图数据库**：Neo4j；向量数据库：ChromaDB/Milvus。
- **评估方式**：
  - **Accuracy (%)**：标准分类准确率。
  - **Strict Accuracy**（24小时测试）：要求满足四维匹配——无幻觉、时间窗完整、实体特征正确、位置准确。
- **硬件平台**：单张 NVIDIA RTX 5090（32GB VRAM）。

---

### **基线方法对比**
- **Clip-based RAG 方法**：如 Video-RAG [29]，基于固定 clip 向量检索。
- **长上下文 VLM**：如 LongVLM、Video-XL，直接扩展 context window。
- **闭源强基线**：GPT-4o、Gemini 1.5 Pro。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **Table 1: Video-MME Long (1800–3600s) 子集结果**
| 任务 | Qwen3-VL-8B (Baseline) | EC-RAG (Ours) | △ 提升 |
|------|------------------------|--------------|--------|
| Action Reasoning | 43.89% | 46.67% | **+2.78%** |
| Action Recognition | 35.00% | 47.50% | **+12.50%** ✅ |

> 💡 表明 EC-RAG 在极长视频下显著缓解注意力分散问题，有效捕捉关键动态。

---

#### 🔹 **Table 2: EventBench 与 NExT-QA 综合表现**
| 方法 | NExT-QA | EventBench (Overall) | vs GPT-4o |
|------|---------|------------------------|-----------|
| GPT-4o (closed) | 76.71% | 53.33% | — |
| Qwen3-VL-8B + EC-RAG | **75.54%** (+2.46) | **55.31%** (+2.92) | ✅ 超越 |
| VideoLlama3-7B + EC-RAG | **78.48%** (+1.79) | **55.97%** (+4.23) | ✅ 超越 |
| InternVL3.5-8B + EC-RAG | **80.54%** (+3.00) | **55.97%** (+2.07) | ✅ 超越 |

> ✅ 所有三个开源 backbone 均**达到甚至超越 GPT-4o 整体水平**，证明 EC-RAG 的强大增益。

---

#### 🔹 **细粒度分析（EventBench 拆解）**
| 类型 | 提升幅度 |
|------|----------|
| Contextual Reasoning | +4.30% ~ +8.61% |
| Episodic Reasoning | +1.67% ~ +7.67% |
| Counter-intuitive | 微降（因压制模糊艺术表达） |

> ⚠️ EC-RAG 在客观、时序性强的任务上优势明显，但在主观/审美类问题上略有下降，因其优先保留可观测事实。

---

### **消融实验结果（Ablation Study on EventBench）**

| 模型变体 | 准确率 | 下降 |
|--------|--------|------|
| EC-RAG（完整系统） | **49.37%** | — |
| w/o Elastic Sentinel（退化为固定切片） | 46.84% | -2.53% |
| w/o SES 抽象（退化为文本摘要） | 43.04% | -6.33% |
| w/o Dual-store merging（仅向量匹配） | 38.99% | **-10.08%** ❗ |
| w/o Semantic Deduplication | 46.33% | -3.04% |

> 📌 结论：所有组件均不可或缺，尤其是**双存储拓扑合并**对长程逻辑保持至关重要。

---

### **24小时无限流处理测试**
- 输入：连续 24 小时未剪辑监控视频。
- 输出：自动提取 **2514 个事件片段**，总耗时 11h14m。
- **Strict Accuracy 达 90.57%**（2277/2514 完全匹配）。
- 峰值 VRAM 消耗仅 **~17.6 GB**，远低于 32GB 上限。

> ✅ 实现了在消费级 GPU 上对无限流视频的高效、稳定处理。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **事件级记忆优于 clip 级记忆**：以语义完整事件为单位组织记忆，能有效避免“spatiotemporal fragmentation”，提升因果推理能力。
2. ✅ **显式因果结构建模至关重要**：SES 图 + 双存储系统使模型能够检索跨越长时间间隔的因果链，解决“lost-in-the-middle”问题。
3. ✅ **轻量级 RAG 可替代昂贵的长上下文建模**：无需扩展 context window 或 KV-cache，即可实现更强的 long-horizon reasoning。
4. ✅ **可在单卡实现工业级部署**：EC-RAG 成功打破传统 VLM 对极端上下文的硬件依赖，在标准 GPU 上实现 24 小时级流处理。

---

### **方法的局限性**
- ❌ **对主观/抽象问题适应性较差**：由于强制转换为事实性 SES 结构，可能抑制艺术性、情感性或隐喻性线索，在“Counter-intuitive”类问题上表现略逊。
- ❌ **依赖高质量 ASR 和视觉编码器**：若音频质量差或视觉信号模糊，事件边界检测可能失效。
- ❌ **目前评测资源有限**：尚缺乏专门针对“超长视频因果推理”的标准化 benchmark。

---

### **未来工作方向**
- 🔄 构建面向 ultra-long video reasoning 的专用评测数据集。
- 🧠 探索融合主观感知与客观因果的混合表示方法，提升对抽象问题的理解。
- 🌐 扩展至多摄像头协同推理、跨视角事件关联等更复杂场景。
- ⚙️ 进一步优化实时性，支持更高帧率或更多并发流处理。

---

> **总结一句话**：  
> **Event-Causal RAG 通过“事件分段 + SES 图 + 双存储检索”范式，实现了高效、精准、可扩展的长视频因果推理，在性能与效率之间取得了突破性平衡。**

</details>

---

### 6. [LLM-Enhanced Deep Reinforcement Learning for Task Offloading in Collaborative Edge Computing](https://arxiv.org/abs/2605.05727)

**Authors**: Hao Guo, Kaixiang Xv, Ziwu Ge, Lei Yang  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.05727v1  

#### Abstract
Collaborative edge computing uses edge nodes in different locations to execute tasks, necessitating dynamic task offloading decisions to maintain low latency and high reliability, especially under unpredictable node failures. Although deep reinforcement learning (DRL) and large language models (LLMs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM-Enhanced Deep Reinforcement Learning for Task Offloading in Collaborative Edge Computing

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**协作边缘计算（Collaborative Edge Computing, CEC）**环境中，任务卸载面临以下挑战：
- **动态性和不确定性高**：节点故障、链路中断、拓扑变化频繁，导致传统静态策略失效。
- **DRL 的局限性**：样本效率低、收敛慢，在大规模网络中易陷入局部最优。
- **LLM 的实时性瓶颈**：虽然具备强大的语义推理能力，但其高延迟和非确定性输出难以满足边缘场景的实时性要求。

### 🚀 提出的新方法：LeDRL
提出了一种**混合决策框架 LeDRL**，将轻量级 Large Language Model（LLM）与基于 self-attention 的 Deep Reinforcement Learning（DRL）相结合，实现高效且鲁棒的任务卸载。

#### 核心创新点：
1. **LLM 引导的语义先验生成**
   - 构建结构化 prompt，融合节点状态、任务语义和链路动态，由 LLM 输出高层策略先验（high-level strategy priors）。
2. **自注意力对齐模块（self-attention-based alignment module）**
   - 将 LLM 生成的语义意图与本地观测进行选择性融合，提升 DRL 策略优化的上下文感知能力。
3. **反射式评估器（Reflective Evaluator）**
   - 从历史轨迹中提炼语义反馈，用于指导后续 prompt 设计，增强 LLM 查询的信息量与时序泛化能力。
4. **轻量化 LLM 集成于在线决策环**
   - 使用小型 LLM（如 Qwen3-4B），平衡语义引导能力与推理延迟，适用于资源受限的边缘设备部署。

### 🔍 相比现有方法的优势
| 维度 | 传统 DRL 方法 | 纯 LLM 方法 | LeDRL |
|------|----------------|--------------|--------|
| **样本效率** | 低 | 不适用 | 显著提高（利用 LLM 先验加速探索） |
| **收敛速度** | 慢 | 快但不稳定 | 更快且更稳定 |
| **实时性** | 高 | 低（大模型延迟高） | 中等偏高（轻量 LLM + 缓存机制） |
| **适应性** | 依赖训练环境 | 泛化强但不可控 | 动态环境下更强鲁棒性 |
| **可部署性** | 可行 | 困难 | 已在 Jetson 设备上验证可行 |

---

## 2. 核心实验方法和设置

### 📊 数据集与仿真环境
- **无真实公开数据集**，采用**自定义模拟器**构建动态边缘系统。
- 时间划分为 100 个时隙（time slots），每个时隙内任务随机到达。
- 节点数量：10 ~ 20 个异构边缘节点（Jetson 类型）。
- 网络拓扑：稀疏连通图，支持随机图与环形结构（ring topology）。
- 任务参数：
  - 输入大小：[2,000, 4,000] KB
  - 计算强度：[800, 2,400] cycles/bit
  - 截止时间：固定为 4 秒
- 故障建模：软件/硬件/传输失败均按泊松过程建模，节点故障率设为 0.01，出现概率 0.1。

### 🧪 实验设置
- **平台**：AMD Ryzen Threadripper 3990X + NVIDIA RTX 4090 + 256GB RAM
- **模型实现**：PyTorch 框架，集成 Qwen3-4B 作为 LLM 模块
- **DRL 架构**：基于 MAPPO 的 Actor-Critic 结构，MLP 隐藏层大小 64
- **注意力模块**：embedding dimension=8，最大长度 512，dropout=0.1
- **训练参数**：学习率 0.0004，折扣因子 γ=0.99，每 4 轮评估一次

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Task Success Rate (%)** | 在截止时间内完成且满足可靠性要求的任务比例（主指标） |
| **Convergence Speed** | 达到稳定性能所需的训练轮次或 episode 数 |
| **Inference Latency (s)** | 单次决策推理耗时 |
| **Robustness** | 在任务规模、计算复杂度、故障率扰动下的性能稳定性 |
| **Scalability** | 不同网络规模（10 vs 20 节点）、不同拓扑结构下的表现 |

### ⚔️ 基线方法对比
共比较六种方法：
1. **DRL 方法**：
   - VDN-TO：基于值分解的多智能体 RL
   - MAPPO-TO：基于策略梯度的去中心化 PPO
   - MASAC-TO：适配离散动作空间的 SAC 变体
2. **启发式方法**：
   - RATC：基于截止时间和可靠性的轻量规则
   - AGSP：结合遗传算法与模拟退火的混合优化引擎
3. **LLM 方法**：
   - Reflexion：通过迭代自我反思优化 LLM 决策

所有方法均进行 10 次随机运行取平均值以确保可复现性。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table I 和 Fig. 4–8）

| Model | 10节点 成功率↑ | 推理延迟↓ | 20节点 成功率↑ | 推理延迟↓ |
|-------|------------------|------------|------------------|------------|
| RATC | 45.48 ± 1.24% | 0.0004s | 59.98 ± 1.45% | 0.0008s |
| AGSP | 50.01 ± 1.63% | 0.0005s | 39.48 ± 1.72% | 0.0018s |
| VDN-TO | 44.47 ± 1.92% | 0.0028s | 51.27 ± 1.82% | 0.0026s |
| MASAC-TO | 51.81 ± 2.88% | 0.0025s | 51.80 ± 1.77% | 0.0024s |
| MAPPO-TO | 52.68 ± 3.40% | 0.0035s | 60.68 ± 2.24% | 0.0043s |
| Reflexion | 48.34 ± 1.97% | 1.5786s | 60.86 ± 1.39% | 3.0128s |
| **LeDRL (Ours)** | **59.46 ± 1.42%** | **0.7046s** | **63.78 ± 1.68%** | **0.7379s** |

> ✅ **最高成功率**：相比最佳基线（MAPPO-TO），LeDRL 在 10 节点下提升 **+12.87%**，在 20 节点下提升 **+5.1%**  
> ✅ **优于纯 LLM 方法**：Reflexion 成功率接近但延迟高出 **4 倍以上**

### 🔁 与基线方法的对比结果
- **训练性能（Fig. 4）**：
  - LeDRL 收敛更快，早期方差更低，最终成功率最高。
  - MAPPO-TO 初期上升快但后期震荡；MASAC-TO 在 20 节点下趋于停滞。
- **鲁棒性测试（Fig. 5）**：
  - 随着任务大小、计算复杂度增加，各方法成功率下降，但 LeDRL 下降最平缓。
  - 在执行失败率=0.25 时，LeDRL 比 MAPPO-TO 提升约 **12%**；在传输失败率=0.25 时提升 **17%**。
- **拓扑影响（Fig. 6a）**：
  - 在环形拓扑中，由于路由灵活性差，MAPPO-TO 性能显著下降，而 LeDRL 保持稳定优势。
  - RATC 和 AGSP 对拓扑敏感，波动剧烈。

### 🔍 消融实验结果（Ablation Study, Fig. 6b）
| 变体 | 成功率 |
|------|--------|
| Plain MAPPO | 最低 |
| LLM + MLP + MAPPO | 有所提升，但未充分对齐语义 |
| RATC/AGSP + Self-Attention + MAPPO | 进一步改善，但仍缺乏上下文反馈 |
| **LeDRL（完整版）** | **最高成功率，收敛最快，策略最稳定** |

> 表明：**self-attention 融合机制 + 反射式记忆更新** 是性能跃升的关键。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 可有效提供高质量语义先验**，显著提升 DRL 的样本效率和初始探索质量。
2. **self-attention 模块能有效对齐 LLM 意图与本地观测**，避免“语义漂移”，提升策略稳定性。
3. **反射式评估器实现了闭环学习**：将失败经验转化为结构化反馈，持续优化 LLM 输出质量。
4. **轻量化 LLM 可实现在线部署**：在 Jetson Nano/Xavier 上成功运行 CoEdgeSys 原型系统，端到端延迟可控。
5. **LeDRL 在多种扰动下表现出卓越鲁棒性**：面对节点故障、链路变化、负载波动仍能维持高性能。

### ⚠️ 方法的局限性
1. **仍需远程 LLM 服务支持**：当前 Qwen3-4B 部署在服务器端，存在通信开销，尚未完全去中心化。
2. **prompt 设计依赖人工工程**：虽结构化，但仍需领域知识设计有效 prompt schema。
3. **对极端拓扑变化适应有限**：如全网断连后再恢复，需重新建立记忆上下文。
4. **内存管理成本**：长期/短期 memory 存储与检索带来额外开销，可能限制超大规模扩展。

### 🔮 未来工作方向
1. **解耦 LLM 与在线推理**：探索仅在训练阶段使用 LLM 提供监督信号，部署时去除 LLM 依赖。
2. **自动化 prompt engineering**：引入 meta-learning 或 prompt tuning 技术减少人工干预。
3. **扩展至更大规模网络**：研究如何在百节点级别保持高效协调。
4. **支持复杂 DAG 工作流**：当前处理原子任务，下一步支持有向无环图（DAG）任务调度。
5. **跨边缘域协同**：研究多个独立边缘集群间的联合卸载机制。

---

> 💡 **总结一句话**：  
> **LeDRL 成功融合了 LLM 的语义理解优势与 DRL 的快速响应能力，通过 self-attention 对齐与反射式学习，在动态边缘环境中实现了更高效、更鲁棒、更具可解释性的任务卸载决策，并已在真实 Jetson 测试床上验证可行性。**

</details>

---

### 7. [Relay Buffer Independent Communication over Pooled HBM for Efficient MoE Inference on Ascend](https://arxiv.org/abs/2605.06055)

**Authors**: Tianlun Hu, Tiancheng Hu, Shengsheng Litang, Sheng Wang, Xiaoming Bao, Yuxing Li, Wei Wang, Zhongzhe Hu, Lijun Li, Hongwei Sun, Jingbin Zhou\\  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.06055v1  

#### Abstract
Mixture-of-Experts (MoE) inference requires large-scale token exchange across devices, making dispatch and combine major bottlenecks in both prefill and decode. Beyond network transfer, routing-driven layout transformation, temporary relay, and output restoration can add substantial overhead. Existi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Relay Buffer Independent Communication over Pooled HBM for Efficient MoE Inference on Ascend**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Mixture-of-Experts (MoE) 推理中的 **dispatch 和 combine 阶段** 是跨设备 token 交换的关键瓶颈。传统实现依赖于显式的中间缓冲区（如 IPC relay buffer 和 reorder buffer）进行数据中转和布局转换，导致：
- 额外的 **HBM 内存占用**
- 更高的 **通信延迟**
- 复杂的 **同步开销**
- 在 **decode 阶段** 尤其影响 **latency-sensitive 性能**

这些问题在长上下文（如 128K~1M tokens）和多模态场景下更加严重。

### **提出的新方法与新思路**
本文提出了一种 **relay-buffer-free（零中继缓冲）通信设计**，用于加速 Ascend 平台上的 MoE 推理，核心思想是：

> **绕过传统的“打包-传输-恢复”流程，直接将 token 放置到目标专家窗口（direct placement），并在 combine 阶段直接从远程专家窗口读取输出（direct reading）。**

该设计基于以下关键技术支撑：
- 利用 Ascend 的 **全局池化 HBM（globally pooled HBM）**
- 通过 **symmetric-memory allocation** 和 **shmem-style remote access** 实现跨 rank 内存共享
- 引入轻量级控制状态（counts, offsets, sync metadata），而非大型中间张量

并为 MoE 的两个主要阶段分别设计了调度策略：
- **Prefill Schedule**：保留更丰富的规划状态（count, offset），适用于吞吐优先的大批量场景
- **Decode Schedule**：采用紧凑控制路径，减少协调步骤，优化低延迟推理

### **相比现有方法的优势**
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **内存使用** | 显式使用大量 relay/reorder buffer，增加 HBM footprint | 减少甚至消除中间缓冲，节省 HBM 空间 |
| **数据流路径** | Sender-pack → Relay-buffer → Receiver-unpack | Direct placement & direct reading |
| **执行模式** | 写密集型 staging（不利于 Ascend） | Read-favored execution（契合 Ascend 硬件特性） |
| **控制复杂度** | 依赖通用 collectives（如 A2A/A2AV） | 定制化 MoE 专用通信路径 |
| **适用性** | 通用但低效 | 充分利用 Ascend 的 pooled HBM 能力 |

---

## **2. 核心实验方法和设置**

### **数据集与模型配置**
- **未使用公开 benchmark 数据集**，而是基于实际生产级 MoE 模型进行测试
- 实验覆盖代表性模型：
  - **DeepSeek-V3 / DeepSeek 3.1 / 3.2**
  - **Qwen2.5-1M / Qwen-235B**
- 测试场景涵盖：
  - 不同 **context length**（从短序列到超长上下文）
  - 不同 **batch size**（decode 阶段重点测试 16~144）
  - 是否启用 **quantization（int8）**

### **实验设置**
- **硬件平台**：华为 Ascend AI 处理器集群
- **通信后端**：
  - **Baseline**：HCCL-enabled DeepEP（当前主流实现）
  - **Proposed**：ZeroBufferEP（本文提出的 relay-buffer-free 实现）
- **运行模式**：
  - Prefill：varying token count (1024 ~ 16384)
  - Decode：varying batch size (16 ~ 144)

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Kernel-level Latency** | dispatch / combine 单个操作的耗时（μs） |
| **TTFT (Time to First Token)** | 首个 token 输出时间（ms），反映 prompt 处理效率 |
| **TPOT (Time Per Output Token)** | 每个生成 token 的平均延迟（ms），反映稳态吞吐 |
| **Feasible Scheduling Space** | 在 TTFT < 5000ms 且 TPOT < 60ms 约束下的可调度配置数量 |

### **基线方法对比**
- 主要对比对象：**HCCL-enabled DeepEP baseline**
- 同时参考了其他 MoE 系统工作（如 Tutel, FUSCO, TensorRT-LLM 的 one-sided A2A）作为背景支持

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### ✅ **Prefill 阶段（图5 & 图7）**
- **dispatch latency（非量化）**：
  - 输入 token 数从 1024 增至 16384 时：
    - Baseline：1.1ms → 9.4ms
    - Proposed：1.0ms → **6.8ms**（↓27.7%）
- **combine latency** 同样显著下降，趋势一致
- **量化场景下增益略小**，但仍优于 baseline（因 scale handling 开销占比上升）

#### ✅ **Decode 阶段（图6 & 表2）**
| Operator | Hidden Size | Quant Mode | Avg Speedup |
|--------|-------------|------------|-------------|
| Dispatch | 4096 | non-quant | **20.18%** |
| Dispatch | 4096 | quant | **31.02%** |
| Dispatch | 7168 | non-quant | 14.83% |
| Dispatch | 7168 | quant | **27.72%** |
| Combine | 4096 | non-quant | 22.75% |
| Combine | 4096 | quant | 24.45% |
| Combine | 7168 | non-quant | 22.43% |
| Combine | 7168 | quant | 24.34% |

> 🔹 **Dispatch 加速最明显**，尤其在量化 + 小 batch 场景  
> 🔹 **Combine 改进稳定**，基本维持在 22%~24%

#### ✅ **典型案例研究（图7）**
- **DeepSeek 3.1**：
  - Dispatch ↓23.3%，Combine ↓10.8%
- **Qwen-235B**：
  - Dispatch ↓**52.8%**，Combine ↓18.3%
> ➤ 差异源于不同模型的路由分布和通信敏感度

#### ✅ **端到端服务性能（图8 & 图9）**
- **TTFT**：
  - Baseline: **11197 ms**
  - Proposed: **6793 ms**（↓**39.3%**）
- **TPOT**：
  - Baseline: 30.10 ms
  - Proposed: 31.31 ms（仍在目标阈值内）
- **调度空间扩展（图9）**：
  - 更多配置满足 **TTFT < 5000ms & TPOT < 60ms**
  - 最高 QPS 可行点更接近约束边界，说明系统灵活性提升

---

## **4. 关键结论和发现**

### **主要发现**
1. **MoE 通信瓶颈不仅是网络带宽问题，更是 buffer-centric 执行范式的问题**  
   → 中间缓冲带来的额外 copy、sync、layout transform 成为主要开销源。

2. **利用全局池化 HBM 可以重构 dispatch/combine 数据流**  
   → direct placement + direct reading 显著缩短关键路径，尤其适合 read-favored 架构（如 Ascend）。

3. **同一通信模型可适配不同阶段需求**  
   → 通过统一的 relay-buffer-free 设计，分别实例化为 **prefill（重规划）** 和 **decode（轻量控制）** 调度，兼顾吞吐与延迟。

4. **性能收益贯穿 kernel 到 serving 层面**  
   → 不仅降低 kernel latency，还改善 **TTFT** 并扩大 **可行调度空间**，对实际部署具有重要意义。

### **方法的局限性**
- **高度依赖特定硬件能力**：需要支持 **globally pooled HBM** 和 **symmetric memory**（目前主要在 Ascend 和部分 NVLink 域可用）
- **不适用于所有通信架构**：若无远程直接访问能力，则难以复现优势
- **控制逻辑仍需轻量状态管理**：虽去除了大 buffer，但仍需 counts/offsets/sync metadata 协调
- 当前实验集中在 **kernel 和 end-to-end latency**，缺乏详细的 **memory footprint 测量**

### **未来工作方向**
- 扩展至更多模型架构和规模（更大 MoE、multi-modal）
- 进一步优化 **cached-address path** 以减少 decode 阶段地址握手开销
- 结合 **pipeline parallelism** 和 **chunked prefill** 实现全栈协同优化
- 探索在其他具备 pooled memory 能力的硬件平台（如 NVIDIA GPU over NVLink + SHMEM）上的移植可能性
- 提供更系统的 benchmark 套件，覆盖训练与推理全流程

---

> 📌 **一句话总结**：  
> 本论文提出一种基于 **pooled HBM** 的 **relay-buffer-free MoE 通信机制**，通过 **direct placement** 和 **direct reading** 显著降低了 dispatch/combine 的内存与延迟开销，在 Ascend 上实现了最高 **52.8% 的 dispatch 加速** 和 **近 40% 的 TTFT 下降**，为高效 MoE 推理提供了新的系统设计范式。

</details>

---

### 8. [ResiHP: Taming LLM Training Failures with Dynamic Hybrid](https://arxiv.org/abs/2605.06374)

**Authors**: Tenghui Ma, Jihu Guo, Wei Gao, Sitian Lu, Zhisheng Ye, Hanjing Wang, Dahua Lin  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.06374v1  

#### Abstract
Hybrid parallelism underpins large-scale LLM training across tens of thousands of GPUs. At such scale, hardware failures on individual devices lead to performance skew across devices, diminishing overall training efficiency. Existing resilient systems overlook sequence length variability in datasets...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ResiHP: Taming LLM Training Failures with Dynamic Hybrid Parallelism

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模 LLM 训练中，**fail-stop**（设备完全失效）和 **fail-slow**（设备性能下降）故障频繁发生，导致严重的 **device performance skew**，从而降低训练效率。现有系统存在两大缺陷：
- **误检问题**：由于序列长度变化引起的迭代时间波动被误判为 fail-slow 故障，引发大量虚假检测（false positives），增加验证开销。
- **适应机制低效**：现有方法仅在单一维度（如 PP 或 DP）进行调整，无法协同优化 TP、PP 和 DP，造成资源浪费和负载不均衡。

### 提出的新方法与思路
ResiHP 是一个面向混合并行训练的容错系统，其核心是 **“精准检测 + 渐进式自适应”** 架构：

#### （1）Detector：基于工作负载感知的轻量级故障检测器
- **Fail-stop 检测**：采用分层心跳机制（hierarchical heartbeat），通过节点内聚合减少全局通信开销。
- **Fail-slow 检测**：引入 **execution time predictor** 来预测正常情况下的迭代时间，将实际运行时间与预测值比较，过滤由序列长度变化引起的时间波动，避免误报。

#### （2）Scheduler：渐进式混合并行调度器
- **TP 维度**：动态重构受影响的 TP 组，选择性排除故障设备而非整个组，保留健康设备，并支持异构 TP 度之间的高效 P2P 通信。
- **PP 维度**：重新划分模型层（layer repartition），减轻因 TP 性能下降造成的流水线气泡。
- **DP 维度**：基于进度感知的工作负载迁移（progress-aware workload migration），平衡各 DP replica 的完成时间。

### 相比现有方法的优势
| 方面 | 现有方法（如 ReCycle, Greyhound） | ResiHP |
|------|-------------------------------|--------|
| 故障检测准确性 | 易受 workload 变化干扰，误报率高 | 引入 workload-aware 预测器，显著降低误报 |
| 资源利用率 | 失败时丢弃整组 TP 设备，资源浪费严重 | 选择性排除，最大化利用剩余算力 |
| 自适应粒度 | 单一维度调整，难以应对跨维度传播 | 跨 TP/PP/DP 渐进式联合优化 |
| 通信效率 | 忽视异构 TP 下的冗余传输 | 优化 scatter/gather，减少跨节点通信 |

---

## 2. 核心实验方法和设置

### 数据集与模型
- 使用开源 **GitHub 数据集** 进行训练，具有典型的变长序列特性。
- 模型：LLaMA2 系列（7B, 13B, 30B, 70B）和 Qwen 2.5 系列（7B, 14B, 32B, 72B）
- 序列长度：8K ~ 32K tokens

### 实验设置
- **硬件平台**：256 块 NVIDIA A100 GPU（32 节点 × 8 GPU），通过 NVSwitch 和 200Gbps InfiniBand 互联。
- **并行策略**：多种 (TP, DP, PP) 配置，最大达 (4,4,16)，覆盖 16 到 256 GPU 规模。
- **故障注入方式**：
  - **Fail-stop**：手动终止进程模拟设备宕机。
  - **Fail-slow**：
    - 计算层面：使用 `nvidia-smi` 锁定 SM 频率；
    - 通信层面：启动侧信道任务制造网络拥塞。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Throughput (samples/s)** | 端到端训练吞吐量，主性能指标 |
| **Detection Accuracy** | 故障识别准确率（尤其是 fail-slow） |
| **False Alarm Rate** | 每次训练中的平均误报警次数 |
| **MAPE** | 微批次和迭代时间预测的平均绝对百分比误差 |
| **Convergence Stability** | 损失曲线是否稳定，恢复后是否影响最终精度 |

### 基线方法对比
| 基线 | 类型 | 功能 |
|------|------|------|
| **ReCycle [10]** | Fail-stop 容忍 | 在 PP 层迁移微批次 |
| **Oobleck [21]** | Fail-stop 恢复 | 切换预设 pipeline 模板 |
| **Greyhound [48]** | Fail-slow 检测与缓解 | 在 DP 层重分配 batch |
| **Adaptra [47]** | Fail-slow 缓解 | 优化 PP 层调度 |
| **Strengthened ReCycle/Oobleck** | 混合故障处理 | 结合 Greyhound 的 fail-slow 模块 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 故障检测性能（Table 5）
| 设置 | ResiHP False Alarms | Greyhound False Alarms | ResiHP Overhead / FA | Detection Accuracy |
|------|---------------------|-------------------------|------------------------|--------------------|
| Small, 8K | 0 | 3.7 | 34ms | 100% |
| Medium, 16K | 0.1 | 5.2 | 45ms | 100% |
| Medium, 32K | 0.3 | 8.7 | 49ms | 98% |

> ➤ ResiHP 将 false alarm 数量从 **~5–8 次降至接近 0**，单次误报开销从 **>2s 降至 <50ms**。

#### ✅ 时间预测精度（Table 4）
| 模型 | Micro-batch Time Predictor MAPE | Iteration Time Predictor MAPE |
|------|-------------------------------|------------------------------|
| Qwen 2.5-7B | 1.19% | 2.81% |
| LLaMA2-13B | 1.21% | 4.89% |

> ➤ 高精度预测有效分离 workload 波动与真实性能退化。

#### ✅ 吞吐提升（Table 6 & Figures 9–10）
在不同故障频率下，ResiHP 相比基线实现 **1.04–4.39× 的吞吐提升**：

| 场景 | 吞吐加速比（vs 基线） |
|------|-----------------------|
| Fail-stop only | 1.22–1.82× vs ReCycle<br>1.07–1.51× vs Oobleck |
| Fail-slow only | 1.32–3.31× vs 基准<br>1.22–1.46× vs Greyhound |
| Mixed failures | **1.48–4.39× vs ReCycle**<br>**1.22–4.32× vs Strengthened ReCycle**<br>**1.04–3.57× vs Strengthened Oobleck** |

> ➤ 在最严苛的每 30 分钟一次 fail-stop 的场景下，基线训练中断，而 ResiHP 仍可持续运行。

### 消融实验结果（Figure 11）
对 ResiHP 三个组件进行逐步启用分析（以 ReCycle 为基准归一化）：

| 组件 | 吞吐增益贡献 |
|------|-------------|
| Selective Device Exclusion (TP) | 最大增益（直接拯救计算资源） |
| Layer Repartition (PP) | 中等增益，受限于统一复制结构 |
| Workload Migration (DP) | 细粒度调节，进一步压缩同步延迟 |

> ➤ 三者协同作用显著优于任一单独模块。

---

## 4. 关键结论和发现

### 主要发现
1. **workload variability 是 fail-slow 误检的主要根源**，必须通过建模加以区分。
2. **failure amplification effect** 显著：局部 TP 故障可放大至全局 DP 同步瓶颈，需尽早干预。
3. **渐进式跨维度自适应（TP → PP → DP）** 是高效应对混合故障的关键路径。
4. **细粒度资源回收（如部分 TP 保留）** 可大幅提升资源利用率，尤其在高故障率环境下至关重要。
5. ResiHP 在保持数学训练语义不变的前提下实现了严格收敛（图12显示 loss 曲线几乎重合）。

### 方法的局限性
- 当前未处理 **Silent Data Corruption (SDC)** 类故障，这类错误不影响执行时间或心跳信号。
- 动态重构依赖于运行时状态迁移，在极端频繁故障下可能累积一定 overhead。
- 对非 Transformer 架构的支持尚未验证。

### 未来工作方向
- 扩展 SDC 检测模块，结合 loss spike、gradient anomaly 等信号构建多模态检测器。
- 探索更智能的在线 parallelism 搜索算法，自动寻找最优 TP/PP/DP 配置。
- 支持弹性扩展（scaling out/in）与故障恢复的统一框架。
- 将 ResiHP 思想移植到其他大规模分布式训练场景（如 RLHF、MoE）。

---

> 🔚 **总结一句话**：  
> ResiHP 通过 **workload-aware 故障检测** 与 **跨维度渐进式自适应调度**，首次实现了对 fail-stop 与 fail-slow 故障的高效、低开销、高吞吐容错训练，在真实规模集群上取得高达 **4.39× 的性能提升**，为超大规模 LLM 训练提供了坚实可靠的系统支撑。

</details>

---

### 9. [LLMSpace: Carbon Footprint Modeling for Large Language Model Inference on LEO Satellites](https://arxiv.org/abs/2605.05615)

**Authors**: Lei Jiang, Adrian Ildefonso, Daniel Loveless, Fan Chen  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.05615v1  

#### Abstract
Large language models (LLMs) impose rapidly growing energy demands, creating an emerging energy and carbon crisis driven by large-scale inference. Solar-powered, AI-enabled low Earth orbit (LEO) satellites have been proposed to mitigate terrestrial electricity consumption, but their lifecycle carbon...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLMSpace: Carbon Footprint Modeling for Large Language Model Inference on LEO Satellites

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）的大规模推理带来了急剧增长的能源消耗和碳排放问题。虽然已有研究提出将 LLM 推理迁移到低地球轨道（LEO）卫星上，利用太阳能减少地面电力依赖，但**空间计算系统的全生命周期碳足迹建模仍不完善**。现有模型存在以下三大缺陷：
- 忽略关键外围设备（如散热板、电池等）的 **embodied carbon**；
- 未考虑太空环境中必需的 **radiation-hardened 硬件** 及其高碳代价；
- 缺乏对 LLM 推理特有行为（如 prefill-decode 阶段差异、prompt 长度影响）的建模。

这些限制导致无法准确评估“空间 AI”是否真正环保。

### 提出的新方法与思路
本文提出了 **LLMSpace** ——首个面向 LEO 卫星上 LLM 推理任务的端到端碳足迹建模框架。该框架联合建模了：
- **operational carbon**（运行能耗）
- **embodied carbon**（制造、发射、硬件生产中的隐含碳）
- **LLM workload 特征**
- **radiation-hardened 加速器与内存设计**
- **关键外围子系统配置**

### 相比现有方法的优势
| 维度 | 传统方法（如 EIR [7], NE [9]） | LLMSpace |
|------|-------------------------------|----------|
| 外围设备建模 | 不完整或聚合估算（如忽略 radiative cooling panel） | 支持多种 solar array、battery、cooling panel 技术选型，提供细粒度分解 |
| 辐射加固硬件支持 | 假设使用 COTS 商用芯片，忽略可靠性问题 | 显式建模 FD-SOI 工艺、TMR、ECC 等技术带来的面积与碳开销 |
| LLM 工作负载感知 | 针对图像处理等通用负载，无 prefill/decode 区分 | 支持不同 prompt 长度、生成 token 数量的任务级碳分析 |
| 准确性验证 | 缺乏真实参考值对比 | 在 Starlink-V1 + DGX H100 场景下验证，误差降低达 **27.8%** |

> ✅ **核心创新**：首次实现 **“任务-硬件-环境”三位一体的碳建模体系**，为可持续的空间 AI 架构设计提供量化工具。

---

## 2. 核心实验方法和设置

### 数据集与基准测试
- 使用 **HELM（Holistic Evaluation of Language Models）** 中的 11 个代表性任务进行 workload 分析：
  - `bank`, `bcode`, `cfresol`, `macalc`, `mareas`, `paragen`, `finan`, `ifeval`, `ensum`, `mmlup`, `omath`
- 执行平台：**NVIDIA H100 GPU**，精度为 bfloat16，batch size = 1
- 调用 **Vidur-Energy 模型** [46] 进行推理能耗预测

### 实验设置
#### 主要场景配置
| 配置类型 | 参数说明 |
|--------|---------|
| **COTS 配置** | 商用 DGX H100（4nm），寿命约 2 年，适用于短期任务 |
| **Rad-hard 配置** | 采用 28nm FD-SOI 工艺 + MRAM 内存 + ECC/TMR，支持 10 年寿命 |
| **卫星平台** | Starlink-V1 类型，搭载 DGX H100 或 Jetson Nano 系统 |
| **电源系统** | Si/GaAs/Multi-junction solar arrays + NMC/LFP batteries |
| **热控系统** | Passive body-mounted / Honeycomb / Heat-pipe radiators |

#### 发射模型
- 使用 **Falcon-9** 发射器参数：
  - 总排放：3.3×10⁵ kgCO₂e/次
  - 有效载荷能力：22,800 kg → 碳强度 $ I_L = 14.5 \, \text{kgCO}_2\text{e}/\text{kg} $

#### 评估指标
- **总生命周期碳足迹（Total Embodied Carbon）**
- **年均化碳排放（Annualized Carbon Emissions）**
- **推理延迟（TTFT, TBT, E2E Latency）**
- **推理能耗（Inference Energy Consumption）**
- **估计准确性提升（vs. EIR/NE）**

#### 基线方法对比
- **EIR [7]**：仅建模部分外围，忽略冷却面板
- **NE [9]**：将多个组件聚合为单一卫星碳值，缺乏可扩展性

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）碳足迹建模准确性显著提升
在 **Starlink-V1 + DGX H100** 场景下的 embodied carbon 估算结果如下（单位：tCO₂e）：

| 组件 | EIR [7] | NE [9] | LLMSpace (COTS) | LLMSpace (rad-hard) | 实际值 |
|------|--------|--------|------------------|----------------------|--------|
| Solar Array | 0.68 | – | 0.68 | 4.19 | – |
| Battery | 0.54 | – | 0.54 | 0.64 | – |
| Cooling Panel | – | 1.76 | 1.76 | 2.55 | – |
| Computing HW | 0.96 | 0.96 | 0.96 | 1.86 | – |
| Net+Satellite | 1.63 | 2.94 | 1.63 | 1.63 | – |
| **Total** | 12.25 | 12.68 | **16.22** | **19.84** | **18.3 / 22.5** |

> 🔍 **结论**：LLMSpace 的预测偏差仅为 **-11.4%（COTS）** 和 **-13.4%（rad-hard）**，相比 EIR 和 NE 提升了 **32.2% 和 27.8% 的准确性**。

---

#### （2）轨道 vs 地面数据中心碳排放比较
- **DGX H100 场景**：
  - 当运行时间 > 5 年时，**rad-hard LEO 卫星的年均碳排放介于“清洁电网”与“高碳电网”之间**，具备一定环保潜力。
  - COTS 配置因辐射失效只能维持 ~2 年，长期不可持续。
- **Jetson Nano 小型 GPU 场景**：
  - 所有轨道部署（COTS/rad-hard）的年均碳排放均高于地面清洁/高碳电网场景。
  - 原因：小型设备的碳成本以 **manufacturing embodied carbon 为主**，而发射和辐射加固进一步放大碳开销。

> 📌 **发现**：**大规模 GPU（如 H100）更适合空间部署；小型移动 GPU（如 Jetson）则更应在地面使用。**

---

#### （3）LLM 工作负载特性分析
- **解码阶段主导能耗**：尽管长 prompt 增加 prefill 成本，但总体能量仍由生成 token 数决定。
- **通信能耗可忽略**：传输能耗约为 0.5 pJ/bit，远低于推理本身（高出 10⁴–10¹⁰ 倍）。
- **适合空间执行的任务**：**生成大量 tokens 的任务（如 summarization, code generation）** 更能发挥太阳能优势，缓解地面压力。

---

#### （4）延迟-碳权衡分析（Latency-Carbon Tradeoff）
构建 **A100 替代方案**（替换 H100）：
- 功耗更低（6.8kW vs 12kW），所需 solar array/battery/cooling 更小 → **总 embodied carbon 下降 ~30%**
- 但 HBM 带宽和算力下降 → **TTFT ↑85%，TBT ↑43%，E2E ↑47%**
- 能耗平均下降 **~10%**，尤其利于长输出任务

> ✅ **结论**：可通过牺牲延迟换取更低碳足迹，适用于非实时应用场景。

---

#### （5）优化配置探索（rad-opt）
通过升级外围设备（multi-junction solar + rad-hard battery + honeycomb radiator）：
- 尽管单体制造碳更高，但由于质量更轻 → **launch carbon ↓**
- 最终实现 total embodied carbon **再降 8%**

> ⚙️ LLMSpace 支持此类多维度设计空间搜索，是其独特优势。

---

## 4. 关键结论和发现

### 主要发现
1. **embodied carbon 主导 LEO 卫星碳足迹**，尤其是 **launch、solar array、battery 和 radiative cooling panel**。
2. **radiation-hardened 设计虽增加初期碳成本，但延长寿命后可摊薄年均排放**，是长期可持续的关键。
3. **外围系统的设计选择对整体碳影响巨大**，必须精细化建模（如 multi-junction solar 可减重降发射碳）。
4. **并非所有 GPU 都适合上天**：
   - 大规模系统（DGX H100）在长周期任务中可能优于地面高碳电网；
   - 小型系统（Jetson Nano）由于制造碳占比高，空间部署反而更不环保。
5. **LLM 推理任务的选择至关重要**：
   - 生成 token 多的任务更适合空间执行；
   - 短交互式聊天类任务收益有限。

---

### 方法的局限性
- **依赖公开参数估算**：部分 radiation-hardened 技术（如 MRAM CPA）缺乏实测数据，依赖保守假设。
- **未建模动态网络拓扑变化**：星座内 ISL 切换、路由变化对能耗的影响未纳入。
- **未考虑退役与再入大气层的环境影响**（如铝颗粒释放）。
- **当前模型静态，尚未集成任务调度与资源分配策略优化**。

---

### 未来工作方向
1. 扩展至 **training 场景建模**（需解决星上训练可行性问题）
2. 引入 **dynamic workload scheduling** 与 **energy-aware routing** 联合优化
3. 结合 **climate impact评估模型**（如臭氧层扰动、颗粒物排放）
4. 开发 **LLMSpace GUI 工具链**，支持工程师进行绿色空间 AI 架构探索
5. 探索 **hybrid 架构**：地面预处理 + 星上推理 + 星地协同 offloading

---

> 💡 **最终结论**：  
> **LLMSpace 为评估“空间 AI 是否真正绿色”提供了首个科学、系统、可量化的建模框架**。研究表明，在合理设计下（大算力、长寿命、高利用率），LEO 卫星上的 LLM 推理有望成为缓解地面能源危机的一种可持续路径，但必须全面考量 **embodied carbon、radiation hardening 和 workload 特性**，避免陷入“伪低碳”陷阱。

</details>

---

### 10. [LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG](https://arxiv.org/abs/2605.06285)

**Authors**: Yijia Zheng, Marcel Worring  
**Category**: cs.CL  
**Published**: 2026-05-08  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.06285v1  

#### Abstract
Single-step retrieval-augmented generation (RAG) provides an efficient way to incorporate external information for simple question answering tasks but struggles with complex questions. Agentic RAG extends this paradigm by replacing single-step retrieval with a multi-step process, in which the large ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **agentic RAG** 方法通过多步推理（如 chain-of-thought）生成自然语言形式的中间思考（thoughts）和子查询（subqueries），以迭代检索外部知识。然而，这种显式生成方式存在显著的**延迟瓶颈**，因为每个 thought 和 subquery 都需通过自回归（autoregressive）逐词生成，导致推理时间大幅增加（通常是 naive RAG 的 15–20 倍）。此外，该过程还阻碍了端到端优化，且在资源受限场景下效率低下。

### 🚀 提出的新方法：LatentRAG
本文提出 **LatentRAG**，一种将推理与检索从离散语言空间转移到连续**潜在空间**（latent space）的新型高效 agentic RAG 框架。其核心思想是：
- 不再显式生成自然语言的 thoughts 和 subqueries；
- 而是在 LLM 的隐藏状态中直接生成对应的**潜意识令牌**（latent tokens），作为语义表示；
- 利用这些 latent tokens 进行后续的检索操作，并支持端到端联合训练。

#### 主要创新点包括：
1. **Latent Reasoning + Latent Retrieval 统一框架**
   - 在单次前向传播中并行生成 latent thought 和 latent subquery tokens，避免了自回归解码带来的高延迟。
   - 将 latent subquery tokens 投影至检索模型输入空间，用于稠密检索（dense retrieval）。

2. **可微分的端到端联合优化**
   - 引入基于 **KL 散度**的对齐目标函数，使 LLM 生成的 latent subquery embeddings 逼近参考检索模型产生的 natural language subquery embeddings 分布。
   - 实现 LLM 与 retrieval model 的联合 fine-tuning，提升检索质量。

3. **可选的并行隐式解码机制（Parallel Latent Decoding）**
   - 添加一个轻量级 projector + LLM 解码路径，将 latent tokens 可逆地还原为自然语言，提高决策透明性。
   - 所有步骤的解码可**并行执行**，相比传统串行生成仍具显著效率优势。

### 🔍 相比现有方法的优势
| 维度 | 传统 Agentic RAG（如 Search-R1, AutoRefine） | LatentRAG |
|------|---------------------------------------------|---------|
| 推理效率 | 自回归生成，延迟极高（~5s） | 单次前向 + 并行处理，延迟降低约 **90%** |
| 优化能力 | 无法反向传播至 retrieval 模块 | 支持 **end-to-end joint optimization** |
| 透明性 | 中间步骤清晰可见 | 可选解码恢复自然语言，实现**效率与透明性的权衡** |
| 训练兼容性 | 依赖大量标注轨迹 | 可利用已有方法生成的轨迹进行监督微调（SFT） |

---

## 2. 核心实验方法和设置

### 📚 数据集
在 **7 个标准问答基准**上进行评估，涵盖通用 QA 与多跳推理任务：

| 类型 | 数据集 |
|------|--------|
| 通用 QA | **NQ**, **TriviaQA**, **PopQA** |
| 多跳 QA | **HotpotQA**, **2wiki**, **Musique**, **Bamboogle** |

所有实验均使用 **2018 Wikipedia dump** 作为统一检索语料库（共约 2100 万文档块），确保公平性和挑战性。

### ⚙️ 实验设置
- **LLM**: 默认使用 `Qwen2.5-7B`，部分实验扩展至 3B / 14B 版本。
- **Retriever**: 测试多种轻量级 dense retriever，包括：
  - `Qwen3-Embedding-0.6B`
  - `e5-base-v2`
  - `jina-embeddings-v5-text-nano`
  - `harrier-oss-v1-270m`
  - `F2LLM-v2-330M`
- **硬件平台**：默认在单张 NVIDIA H100 GPU 上测量延迟；缩放实验使用三张 H100 部署检索索引。
- **训练方式**：采用 **Supervised Fine-Tuning (SFT)**，基于 Search-R1 和 AutoRefine 生成的交互轨迹构建训练样本。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **EM (%)** | Exact Match 准确率，衡量预测答案是否完全匹配真实答案 |
| **Average Latency (ms)** | 每个问题的平均端到端响应时间（含 prefill、generation、retrieval） |
| **Stage-wise Latency Breakdown** | 各阶段耗时分析（thought gen., subquery gen., retrieval, etc.） |
| **Retrieval Success Rate & Overlap** | 衡量检索有效性及与教师模型的一致性（消融实验中使用） |

### 🆚 基线方法对比
| 类别 | 方法 |
|------|------|
| 直接推理 | Direct Infer |
| 单步 RAG | Naive RAG |
| Prompt-based Agentic RAG | Iter-RetGen, Search-o1 |
| Training-based Agentic RAG | DeepRAG, Search-R1, AutoRefine, ZeroSearch |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1）
- **LatentRAG** 在多个数据集上实现了与最强基线（如 Search-R1、AutoRefine）**相当甚至略优的 EM 性能**，相对差距小于 **5%**。
- **平均延迟下降约 90%**，接近 naive single-step RAG 的水平。
  - 例如，在使用 `Qwen3-Embedding-0.6B` 时：
    - Search-R1 平均延迟：**5372 ms**
    - LatentRAG（无解码）：**593 ms**（↓89.0%）
    - LatentRAG（带解码）：**1970 ms**（↓63.3%）

> 💡 图 1 显示，LatentRAG 极大减少了 thought 和 subquery 生成阶段的时间开销，这是传统方法的主要瓶颈。

### 🔁 与基线方法的对比结果
| 方法 | 相对于 Baseline 的 EM 差距 | 延迟减少比例 |
|------|--------------------------|--------------|
| LatentRAG① (vs. Search-R1) | ±0 ~ +2.3% | ↓86–91% |
| LatentRAGA (vs. AutoRefine) | ±0 ~ +2.5% | ↓87–91% |

- 在 **multi-hop QA 数据集**（如 HotpotQA, Musique）上表现尤为突出，说明 latent 推理能有效捕捉复杂推理结构。
- 即便启用 latent decoding 提升透明性，**整体延迟仍比基线低 47–63%**。

### 🔍 消融实验结果（见 Table 3）
| 变体 | EM (%) | Success (%) | Overlap (%) | 结论 |
|------|--------|------------|------------|-------|
| LatentRAG (完整版) | **43.46** | **61.27** | 59.41 | 最佳综合性能 |
| - 替换为 Cosine Loss | 42.55 | 60.76 | **68.31** | 对齐更紧但性能更低 → 过度模仿限制泛化 |
| - 替换为 InfoNCE Loss | 41.86 | 58.60 | 47.08 | 不适合小规模 noisy 数据 |
| - 移除预训练 retriever | 41.85 | 59.07 | 50.92 | 验证了预训练先验的重要性 |
| - 移除 latent decoding loss | 40.61 | 60.64 | 57.38 | 解码损失有助于学习更好的 latent 表示 |

> ✅ **KL divergence 目标函数**和**latent decoding 辅助任务**均被证明对性能有正向作用。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Latent Space 是高效 agentic RAG 的可行路径**
   - 将 reasoning 与 retrieval 统一于 latent space，可在几乎不牺牲性能的前提下，将推理延迟压缩近 **90%**。
   
2. **Latent Tokens 能有效编码语义意图**
   - LogitLens 分析表明，latent tokens 自然聚集在相关词汇区域（如 “Vincent Novello”、“William Goldman”），具备良好的语义一致性。
   - 单个 latent token 可表达完整概念（如 “Christianity Today”），优于 subword 分割。

3. **并行解码机制实现效率与透明性的平衡**
   - 解码过程可跨步骤并行，即使开启也仅带来约 4–5× 的延迟增长，仍远优于基线。

4. **更大的教师模型能生成更高质量训练轨迹**
   - 使用 7B/14B LLM 生成的轨迹训练 3B LatentRAG，EM 提升达 **15%**，验证了“teacher quality 决定 student performance”。

### ⚠️ 局限性
1. **依赖高质量训练轨迹**
   - 当前采用 SFT 范式，性能受限于教师模型的能力，难以超越 teacher。
   - 未来可通过强化学习探索更优策略。

2. **对 retrieval model 的几何特性敏感**
   - 如 `e5-base-v2` 存在严重各向异性（anisotropy），导致 embedding 对齐困难，性能下降明显。

3. **exact match 下易因细微拼写错误失败**
   - 如 Failure Case 中输出 “Sir John Chilcott” 或 “Montmoreiras”，虽推理正确但仍被判错，反映 latent 表示在精确 token 输出上的不足。

### 🔮 未来工作方向
- 探索 **Reinforcement Learning** 框架下的 latent policy learning，摆脱对 teacher 轨迹的依赖。
- 设计更鲁棒的 **latent-to-text generation head**，提升最终答案的准确性。
- 构建面向 agent 的 **embedding-based search engine**，而非复用人类导向的文本搜索引擎。
- 扩展至其他工具调用场景（tool-use），如代码执行、数据库查询等。

---

> 📌 **一句话总结**：  
> **LatentRAG 成功将 agentic RAG 的推理与检索过程从“语言空间”迁移至“潜意识空间”，在保持竞争力性能的同时，将延迟降低一个数量级，为构建高效、可扩展的智能代理系统提供了新范式。**

</details>

---

### 11. [FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking](https://arxiv.org/abs/2605.05482)

**Authors**: Denys Katerenchuk, Pablo Duboue, Keelan Evanini, David Gondek, Nithin Govindugari, Olivier Allauzen, Joshua Baptiste, David J More, Joshua Schechter  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.05482v1  

#### Abstract
Large language models (LLMs) are rapidly being adopted across various domains. However, their adoption in banking industry faces resistance due to demands for high accuracy, regulatory compliance, and the need for verifiable and grounded responses. We present a unified, data-efficient framework for ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在银行业等**受监管领域**的应用面临重大挑战，主要包括：
- **幻觉（hallucination）**：生成看似合理但不准确的回答；
- **过度顺从（sycophancy / over-compliance）**：即使缺乏支持证据也强行作答；
- **响应不可追溯**：无法提供引用来源，难以满足合规要求；
- **高延迟与成本**：依赖如 GPT-4 这类闭源 API 导致部署成本高昂。

因此，亟需一种**可落地、可靠、高效且可验证**的 LLM 解决方案用于银行场景下的检索增强生成（RAG）问答系统。

---

### 🚀 提出的新方法与创新思路

FinRAG-12B 是一个专为金融领域设计的 **12B 参数 LLM**，其核心是一个**统一、数据高效的训练框架**，旨在联合优化以下三方面能力：

#### （1）高质量、可溯源的回答生成（Grounded QA）
- 构建多阶段数据管道，结合 **LLM-as-a-Judge 过滤**、**显式引用标注** 和 **课程学习（curriculum learning）**。
- 引入合成 QA 数据（基于 SEC filings），并控制难度、风格、长度以贴近真实用户查询分布。
- 在训练中注入 **distractor passages** 并随机化相关段落位置，缓解 RAG 中的**位置偏见（positional bias）**。

#### （2）校准的拒绝机制（Calibrated Refusal）
- 显式引入 **22% 不可回答样本（unanswerable examples）**，监督模型输出 `"I don't know"`。
- 实现“**恰到好处的拒绝**”——既避免 base model 的**拒绝不足**（under-refusal），又防止 GPT-4 的**过度拒绝**（over-refusal）。

#### （3）端到端生产级部署流程
- 从数据清洗 → 模型微调（LoRA/DoRA）→ 量化推理（W4A16 SmoothQuant）形成完整闭环。
- 支持单 GPU 部署，显著降低延迟与成本。

---

### 🔍 相比现有方法的优势

| 维度 | FinRAG-12B 的优势 |
|------|------------------|
| **性能** | JudgeLM 得分 **6.21**，超过 GPT-4.1（5.72）和基座模型 Gemma 3 12B-IT（5.70） |
| **引用质量** | Citation Quality 达 **73.1**，优于 GPT-4.1（70.8） |
| **拒绝行为校准** | “I don’t know”率 **12.0%**，优于 base model（4.3%，太低）和 GPT-4.1（20.2%，太高） |
| **效率与成本** | 推理速度比 GPT-4.1 快 **3–5×**，每 query 成本低 **20–50×** |
| **可复现性** | Stage 1 完全使用开源数据（如 RAG-v1），便于社区复现 |

> 💡 特别强调：该模型是目前少有的经过 **40+ 金融机构实际部署验证** 的 LLM。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据源 | 类型 | 样本数 | 描述 |
|-------|------|--------|------|
| **RAG-v1 (Open)** | 开源合成 QA | 43,581 | 来自 glaiveai/RAG-v1，经 JudgeLM 过滤（score ≥ 5） |
| **SEC Reports (Synthetic QA)** | 合成 QA | 16,773 | 基于 10-K/10-Q 文件生成，五步流程确保真实性与多样性 |
| **CommonCrawl (Financial)** | 真实用户查询 | 20,499 | 抽取无 PII 的高频金融问题，匹配上下文 |
| **Refusal Calibration (Proprietary)** | 私有对话数据 | 17,795 | 包含“上下文相关但不足以回答”的案例，用于训练拒绝行为 |

> 总计：**98,648 样本，143M tokens**

---

### ⚙️ 实验设置

#### 模型架构
- **Base Model**: Gemma 3 12B-IT（128K 上下文窗口，商业许可友好）
- **微调方式**: LoRA（`r=64`, `alpha=256`, dropout=0.05），应用于所有 attention 和 MLP 层
- **优化器**: 8-bit AdamW，学习率 `2e-5`
- **Batch Size**: per-device 4 × gradient accumulation 4 = effective 16
- **序列长度**: 最大 65,536 tokens
- **训练耗时**: ~1,400 步，约 **360 GPU 小时（8×RTX A6000）**

#### 课程学习（Curriculum Learning）
| 阶段 | 数据 | 学习率 | 目标 |
|------|------|--------|-----|
| Stage 1 | RAG-v1 + SEC Synthetic QA | 1e-6（cosine decay） | 建立基础的 grounded generation 能力 |
| Stage 2 | CC + Proprietary Banking Data | 5e-6（linear decay） | 注入真实世界分布与拒绝行为 |

> ❗关键发现：直接混合所有数据会导致性能崩溃（JudgeLM ↓至 3.28）

---

### 🎯 评估指标

| 指标 | 定义 |
|------|------|
| **JudgeLM Score (1–10)** | 使用 JudgeLM（GPT-4.1 或 7B）对答案进行评分，涵盖正确性、完整性、连贯性 |
| **Citation Quality (0–100)** | 复合指标：faithfulness、source relevance、information synthesis、source usage |
| **QA F1** | 精确率与召回率的调和平均，衡量可回答问题上的表现 |
| **IDK%（Abstention Rate）** | 输出 `"I don't know"` 的比例 |
| **True Negative Refusal Rate (TN%)** | 所有拒绝中，“确实不应回答”的占比 |
| **TTFT / TTC** | Time to First Token 和 Total Time to Completion，反映延迟 |
| **Cost per Query** | 单次推理成本估算 |

---

### 🆚 基线方法对比

| 模型 | 类型 |
|------|------|
| **Gemma 3 12B-IT (no fine-tuning)** | 基座模型 |
| **GPT-4.1 (API)** | 商业闭源模型（作为强基准） |
| **BloombergGPT** | 未公开权重，无法比较 |
| **FinGPT** | 无 citation 支持，不适用于 RAG 场景 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 4 & 5）

| Model | JudgeLM | Citation Q. | QA F1 | IDK% |
|-------|---------|-------------|--------|-------|
| Gemma 3 12B (base) | 5.70 | 80.2 | 0.964 | 4.3% |
| GPT-4.1 (API) | 5.72 | 70.8 | 0.900 | 20.2% |
| **FinRAG-12B** | **6.21** ↑ | **73.1** ↑ | **0.936** ↑ | **12.0%** ✅ |

> ✅ FinRAG-12B 在 **JudgeLM 得分、引用质量和 QA F1** 上全面领先，同时实现更合理的拒绝率。

---

### 🌐 Public Benchmark 结果（FinanceBench）

| Model | FinanceBench F1 | Citation Rate |
|-------|------------------|---------------|
| Gemma 3 12B (base) | 0.249 | — |
| GPT-4.1 | 0.238 | — |
| **FinRAG-12B** | **0.284** ↑ | 97.3% |

> 表明 FinRAG-12B 具备良好的泛化能力，在未见过的 SEC 文档问答任务上仍保持领先。

---

### 🔬 消融实验结果（Ablation Study, Table 6）

| 数据策略 | JudgeLM | QA F1 | Cit. Q | IDK% | TN% |
|--------|--------|--------|--------|--------|--------|
| External only | 5.72 | 0.972 | 76.1 | 0.4% | 0% |
| Internal only | 5.62 | 0.913 | 69.2 | 17.4% | 53% |
| Combined (all mixed) | 3.28 | 0.706 | 51.2 | 46.5% | 39% |
| **Curriculum (staged)** | **5.91** | **0.938** | **74.7** | **13.2%** | **56%** |

> 🔍 发现：
> - 单独用外部数据 → 几乎不拒绝（危险！）
> - 单独用内部数据 → 拒绝过多但准确率一般
> - 直接混合 → 性能崩塌（灾难性干扰）
> - **课程学习（先外后内）是最优解**

---

### ⏱️ 推理延迟与成本（Figure 1）

| 指标 | FinRAG-12B | GPT-4.1 |
|------|------------|---------|
| TTFT | 0.14s | 0.57s |
| TTC | 0.57s | 1.77s |
| 成本/Query | ~$0.001 | $0.02–$0.05 |

> ✅ **速度快 3–5×，成本低 20–50×**

---

### 🏢 生产环境影响（Production Impact, Table 7）

在一家美国信用合作社部署 7 个月（3,297 查询）的结果：

| 指标 | 旧模型 | FinRAG-12B | 差值 |
|------|--------|-----------|------|
| Resolution Rate | 77.4% | **84.5%** | **+7.1pp***（p<0.001） |
| Unresolved Rate | 20.7% | **13.7%** | -7.0pp |
| User Satisfaction | 59.5% | 62.9% | +3.4pp（不显著） |

> 💡 关键洞察：**满意度提升主要源于更多问题被成功解决**，而非单条回复质量提高。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **数据质量 > 数据规模**  
   仅用 **143M tokens** 就超越 GPT-4.1，证明精心构建的小规模高质量数据足以支撑高性能 LLM。

2. **课程学习至关重要**  
   分阶段训练（先通用再专用）能有效避免灾难性遗忘与行为冲突，尤其在融合开放与私有数据时。

3. **拒绝行为必须显式校准**  
   加入 **22% 不可回答样本** 可实现最佳平衡：既减少幻觉，又不过度保守。

4. **分辨率（Resolution Rate）是用户体验的关键驱动因素**  
   即使单条回复质量相近，更高的解决率也能显著改善整体满意度。

5. **量化不影响 grounded generation 性能**  
   W4A16 量化后仍保留 >99% 的 citation quality，适合生产部署。

---

### ⚠️ 局限性（Limitations）

1. **领域局限性**  
   当前评估集中于零售与商业银行服务，尚未覆盖交易、保险、投顾等领域。

2. **长尾问题覆盖不足**  
   测试集偏向常见问题，罕见边缘情况（edge cases）探索有限。

3. **私有数据不可共享**  
   Stage 2 使用的 proprietary banking data 因隐私与合规原因无法发布。

4. **拒绝表达形式受限**  
   评估仅统计明确 `"I don't know"` 形式，可能忽略委婉或部分不确定性表达（如 "It might..."）。

---

### 🔮 未来工作方向

1. **扩展至其他金融子领域**  
   如投资银行、财富管理、反洗钱（AML）咨询等。

2. **动态更新机制研究**  
   如何持续将最新政策、利率变动融入模型知识体系。

3. **多模态 RAG 支持**  
   结合表格、PDF、图像等形式的金融文档进行推理。

4. **更细粒度的不确定性建模**  
   支持概率性回答、置信度提示、渐进式披露等交互模式。

5. **开放 Stage 1 训练代码与配置**  
   推动社区复现与改进，促进金融 LLM 的透明发展。

---

> 📌 **一句话总结**：  
> **FinRAG-12B 通过高质量数据工程、课程学习与拒绝校准，在小参数量下实现了超越 GPT-4 的 grounded QA 能力，并已在 40+ 金融机构成功落地，验证了“数据优于规模”、“可控优于强大”的实用主义 LLM 路径。**

</details>

---

### 12. [Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key](https://arxiv.org/abs/2605.06638)

**Authors**: Tianle Wang, Zhaoyang Wang, Guangchen Lan, Xinpeng Wei, Sipeng Zhang, Guanwen Qiu, Abulhair Saparov  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.06638v1  

#### Abstract
Reinforcement learning (RL) has been applied to improve large language model (LLM) reasoning, yet the systematic study of how training scales with task difficulty has been hampered by the lack of controlled, scalable environments. We introduce ScaleLogic, a synthetic logical reasoning framework that...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Can RL Teach Long-Horizon Reasoning to LLMs? Expressiveness Is Key

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在使用 **Reinforcement Learning (RL)** 对 **Large Language Models (LLMs)** 进行推理能力后训练时，缺乏一个**可控且可扩展的环境**来系统研究训练如何随任务复杂度（尤其是长链推理）变化。现有的推理任务（如数学、编程）虽然具有可验证性，但在以下方面存在不足：
- 高质量数据难以大规模获取；
- 推理深度（horizon）和逻辑表达力（expressiveness）无法独立控制；
- 难以进行系统性的 scaling law 分析。

这导致我们无法回答核心问题：**RL 是否能有效教会 LLM 进行长程推理？其训练成本如何随难度增长？**

### 提出了什么新方法或新思路
作者提出了 **SCALELOGIC** —— 一种**合成的、可控的逻辑推理框架**，用于研究 RL 在长程推理中的缩放行为。

#### 核心创新：
- **双轴难度控制**：
  - **推理深度（Reasoning Depth, D）**：控制证明所需的步骤数。
  - **逻辑表达力（Logical Expressiveness）**：从简单的 `→`（implication-only）逐步扩展到包含 `∧`（conjunction）、`¬`（negation）、`∨`（disjunction）和 `∀`（universal quantification）的一阶逻辑。
- **自动可生成与可验证**：所有问题均可通过算法自动生成，并通过形式化验证器（如 Z3）确保标签正确。
- **多选题形式**：每个实例有 $ B $ 个候选结论，仅有一个可通过给定公理推导出，其余通过“破坏”一条公理使其不可证，迫使模型进行完整推理而非依赖启发式。

### 相比现有方法的优势
| 方法 | Verifiable | Scalable | Controllable Horizon | Controllable Expressiveness |
|------|------------|----------|------------------------|----------------------------|
| 数学/编程任务 | ✅ | ❌ | ❌ | ❌ |
| 自演化管道（Self-Evolving） | ✅ | ✅ | ❌ | ❌ |
| SAT/Knights & Knaves | ✅ | ✅ | ❌ | ❌ |
| **SCALELOGIC（本文）** | ✅ | ✅ | ✅ | ✅ |

SCALELOGIC 是首个同时满足**可验证性、可扩展性、推理深度可控性和逻辑表达力可控性**的 RL 训练环境。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **SCALELOGIC**：完全由算法生成的合成逻辑推理数据集。
  - 包含 5 种逻辑表达力层级：
    1. Implication-only (`→`)
    2. + Conjunction (`→`, `∧`)
    3. + Negation (`→`, `∧`, `¬`)
    4. + Disjunction (`→`, `∧`, `¬`, `∨`)
    5. + Quantification (`→`, `∧`, `¬`, `∨`, `∀`)
  - 每个任务为单选题，$ B=4 $ 个候选结论，默认训练深度 $ D \in \{1,\dots,D_{\text{target}}\} $。

### 实验设置和评估指标

#### 模型
- 主要使用 **Qwen3-4B** 进行 RL 后训练。
- 部分实验复现于 **Qwen3-8B** 以验证跨规模一致性。

#### RL 框架
- 使用 **DAPO**（基于 GRPO 的改进版 RL 算法）。
- 奖励设计：二值奖励（binary reward），仅当最终答案格式正确且匹配真值时 $ R=1 $，否则 $ R=0 $。
- Prompt 模板强制模型输出 `<answer>...</answer>`。

#### 主要评估指标
- **训练计算量 $ T $**：达到 **90% 验证准确率**所需的 **RL 训练步数**。
- **下游迁移性能**：在 8 个真实世界推理基准上的平均准确率（Avg@8）：
  - 数学类：AIME 2024/2025, AMC2023, MATH-500, Minerva
  - 综合推理类：OlympiadBench (text-only), GPQA-Diamond, MMLU-Pro (STEM subset)

#### 基线方法对比
- 不同逻辑表达力设置之间的横向比较（如 Implication-only vs. +Quantification）
- 不同训练分布策略：
  - Uniform sampling（默认）
  - Curriculum learning（渐进增加深度）
  - Difficult-only（只用最大深度样本）
- 多种 RL 算法对比：DAPO vs. GRPO vs. GSPO

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔹 RQ1: 训练成本随深度呈幂律增长
- 在所有逻辑设置下，训练步数 $ T $ 与推理深度 $ D $ 满足幂律关系：
  $$
  T \propto D^\gamma, \quad R^2 > 0.99
  $$
- 幂指数 $ \gamma $ 随逻辑表达力单调递增：
  | 逻辑设置 | $\gamma$ |
  |---------|---------|
  | Implication-only | 1.04 ± 0.03 |
  | + Conjunction | 1.72 ± 0.08 |
  | + Negation | 1.81 ± 0.05 |
  | + Disjunction | 2.11 ± 0.09 |
  | + Quantification | **2.60 ± 0.06** |

> 💡 **含义**：最复杂的逻辑设置下，深度翻倍会导致训练成本增长约 $ 2^{2.6} \approx 6\times $，远高于简单逻辑下的 $ 2^{1.04} \approx 2\times $。

#### 🔹 RQ2: 下游迁移性能显著提升
- 所有 RL 训练均优于 base model（49.39%）。
- 最强设置（+Quantification）在 8 个下游基准上平均准确率达到 **60.05%**，绝对提升 **+10.66 个百分点**。
- 更重要的是：**更富表达力的训练带来更强、更持续的迁移增益**，而简单逻辑很快饱和。

#### 🔹 RQ3: Curriculum 显著提高训练效率
- 在 +Conjunction 设置下：
  - Curriculum: $ \gamma = 1.33 $
  - Uniform: $ \gamma = 1.70 $
  - Difficult-only: $ \gamma = 2.36 $
- Curriculum 将训练成本指数降低近 40%，并显著减少方差。

#### 🔹 RQ4: 跨算法鲁棒性
- 在 +Conjunction 设置下测试三种 RL 算法：
  - DAPO: $ \gamma = 1.70 $
  - GSPO: $ \gamma = 1.65 $
  - GRPO: $ \gamma = 2.05 $
- 所有算法均呈现幂律缩放，说明该现象是**普遍规律**，非特定优化器产物。

#### 🔹 RQ5: OOD 泛化有限
- 即使训练到深度 $ D=14 $，模型在测试深度 $ D_{\text{test}} > 3 \times D_{\text{train}} $ 时性能仍会下降至随机水平。
- 表明：**更深训练可线性扩展有效推理范围，但无法消除 horizon limit**。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **RL 训练成本遵循幂律缩放**：$ T \propto D^\gamma $，且 $ \gamma $ 随逻辑表达力单调上升。
2. ✅ **表达力决定迁移效果**：训练数据的**逻辑丰富程度**比训练量更重要；越复杂的逻辑带来越强、越高效的下游迁移。
3. ✅ **Curriculum 提升训练效率**：渐进式训练能显著降低缩放指数，加速 long-CoT（chain-of-thought）行为的出现。
4. ✅ **缩放规律具有一般性**：该幂律关系在不同 RL 算法间保持一致，表明是任务本身的结构性质所致。
5. ✅ **OOD 泛化边界明确**：模型最多只能泛化到约 $ 3\times $ 训练深度，说明存在根本性推理瓶颈。

### 方法的局限性

- **模型规模限制**：主实验基于 Qwen3-4B，虽在 Qwen3-8B 上部分复现，但仍需更大模型验证 scaling law 是否延续。
- **表达力覆盖有限**：未涵盖等式（equality）、高阶逻辑、非单调推理等更复杂结构。
- **构造假设**：保证唯一证明路径的设计可能弱化现实中的歧义搜索场景。
- **单种子任务**：尽管逻辑多样，但仍属于同一类符号推理任务，外推到其他领域需谨慎。

### 未来工作方向

1. **探索更大模型下的 scaling behavior**：是否 $ \gamma $ 会随模型增大而减小？
2. **引入更丰富的逻辑片段**：如 equality、higher-order functions、non-monotonic reasoning。
3. **理论建模**：为何不同逻辑操作符会导致不同的 $ \gamma $？能否建立形式化解释？
4. **更真实的 multi-entity 结构**：构建更具挑战的关系推理环境。
5. **结合 test-time scaling**：研究 training-time 与 test-time scaling 的交互效应。

---

> 📌 **一句话总结**：  
> **SCALELOGIC 揭示了 RL 教会 LLM 长程推理的关键不是“训得多”，而是“训得对”——训练数据的逻辑表达力直接决定了训练效率和迁移能力，且这一过程遵循清晰的幂律规律。**

</details>

---

### 13. [A Privacy-Preserving Machine Learning Framework for Edge Intelligence: An Empirical Analysis](https://arxiv.org/abs/2605.05751)

**Authors**: Quoc Lap Trieu, Bahman Javadi, Jim Basilakis  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.05751v1  

#### Abstract
As Edge Intelligence (EI) becomes increasingly prevalent in domains such as smart healthcare, manufacturing, and critical infrastructure, ensuring data privacy while maintaining system efficiency is a growing challenge. This paper presents a new privacy-preserving machine learning (PPML) framework t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题**: *A Privacy-Preserving Machine Learning Framework for Edge Intelligence: An Empirical Analysis*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对 **Edge Intelligence (EI)** 场景中日益突出的隐私保护挑战，系统性地评估了在资源受限边缘设备上部署 **Privacy-Preserving Machine Learning (PPML)** 技术时面临的性能与隐私权衡问题。具体聚焦于以下挑战：
- 如何在保证低延迟、高能效的同时实现强数据隐私保护；
- 不同 PPML 技术（DP、SMC、FHE）在真实边缘环境中的实际开销与适用场景缺乏全面实证分析。

### 提出了什么新方法或新思路
提出了一种专为 EI 应用定制的 **四层 PPML 框架架构**，并结合真实实现与 trace-based simulation 进行综合评估：
- **分层架构设计**：将系统划分为 **Cloud、Edge Server、Edge Device 和 Sensor** 四个层级，明确各层职责与隐私模块部署位置。
- **统一评估框架**：首次在同一实验环境下对 **Differential Privacy (DP)、Secure Multi-party Computation (SMC) 和 Fully Homomorphic Encryption (FHE)** 三种主流技术进行端到端比较。
- **双阶段评估方法**：采用“硬件实验 + trace-based simulation”方式，既获取真实性能 trace，又支持大规模仿真以模拟复杂负载。

### 相比现有方法的优势
- **更贴近现实**：不同于多数研究仅关注云环境或训练阶段，本工作聚焦 EI 中最常见的 **inference 任务**，并在异构边缘设备上测试。
- **多维度量化分析**：不仅评估 accuracy，还深入测量 **response time、energy consumption、communication overhead** 等关键指标。
- **安全与提取攻击分析**：引入 black-box model stealing 攻击模型，分析不同 PPML 方法对模型可提取性的抵抗能力。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验基于 **UEA & UCR Time Series Classification Archive** 中的真实时间序列数据集：
| 数据集 | 领域 | 类别数 | 输入长度 |
|-------|------|--------|----------|
| **FordA** | Critical manufacturing | 2 | 500 |
| **ElectricDevices** | Energy | 7 | 96 |
| **ECG5000** | Public health | 5 | 140 |

这些数据集分别代表智能医疗、智能制造和智能电网等典型 EI 应用场景。

### 实验设置和评估指标

#### 深度学习模型
- **LeNet-5**：轻量级 CNN，适合资源受限设备
- **SqueezeNet**：紧凑结构，参数少但精度接近 AlexNet
- **AlexNet**：较复杂的 CNN，用于对比高计算需求下的表现

#### 边缘硬件配置
| 设备 | 类型 | CPU | 内存 |
|------|------|-----|------|
| EC2 r6i.4xlarge | Cloud Server | 16核, 3.7GHz Xeon | 128 GB |
| Desktop | Edge Server | 16核, 3.7GHz Xeon | 32 GB |
| Intel NUC | Edge Device | 8核 i7 | 8 GB |
| Jetson AGX | Edge Device | 8核 ARMv8 | 32 GB |
| Raspberry Pi 5 | Edge Device | 4核 Cortex-A76 | 4 GB |

#### 仿真工具与参数
- 工具：**EdgeSimPy**（Python-based edge computing simulator）
- 设置：4 个 Edge Server，12 个 Edge Device，模拟 30–70 并发用户请求
- 带宽测试：SMC 在 250 Mbps 与 500 Mbps 下的表现

#### 评估指标
- **Accuracy (%)**
- **Response Time / Inference Latency (ms/s)**
- **Energy Consumption (Wh/inference)**
- **Communication Overhead (bytes, rounds)**
- **Privacy Budget (ε, δ) for DP**
- **Model Extraction Resistance (△Acc, △F1)**

### 基线方法对比
- **Baseline**：无任何隐私保护的原始模型推理
- **DP**：使用 TensorFlow Privacy 实现 DP-SGD，噪声乘子 σ ∈ [0.1, 0.7]
- **SMC**：使用 CrypTen 实现 2-party 与 3-party 协议
- **FHE**：使用 Concrete-ML 编译模型，量化位宽 q ∈ {4,5,6}，精度位宽 p=5

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 准确率影响（Accuracy）
| 方法 | 模型 | 数据集 | 准确率下降幅度 |
|------|------|--------|----------------|
| **DP** | AlexNet | FordA | ↓35% |
| **DP** | LeNet-5 | FordA | ↓18% |
| **FHE** | AlexNet | ECG5000 | 接近 baseline（损失 < 2% @ 6-bit） |
| **SMC** | 所有 | 所有 | 几乎无损失（数值误差可控） |

> ✅ **DP 对复杂模型更敏感**；**FHE 可保留高准确率但依赖量化调优**；**SMC 准确率基本不变**

#### 响应时间（Response Time）
| 方法 | 相对于 Baseline 的延迟增长 |
|------|----------------------------|
| **DP** | ≈ +10%（几乎无额外开销） |
| **SMC** | ×2–10（取决于通信带宽与参与方数量） |
| **FHE** | **×1000**（最高达数千秒级别） |

- **FHE 延迟极高**：AlexNet 在 FordA 上达到 **~10,000 秒**（70 用户并发）
- **SMC 显著受带宽影响**：从 250 Mbps 提升至 500 Mbps，AlexNet 推理延迟降低约 **30%**
- **DP 最高效**：响应时间接近明文推理

#### 能耗表现（Energy Consumption per Inference）
| 方法 | AlexNet (ECG5000) | LeNet-5 (ECG5000) |
|------|-------------------|--------------------|
| **DP** | 0.0011 Wh | 0.0009 Wh |
| **SMC-2P@500Mbps** | 0.3093 Wh | 0.0088 Wh |
| **FHE** | **20.0167 Wh** | 1.1168 Wh |

> 🔋 **FHE 是最耗能的技术**，尤其对复杂模型；**DP 能效最优**

#### 模型窃取攻击抵抗力（Model Stealing Resistance）
通过 black-box 查询构建替代模型（surrogate），衡量性能差距 △Acc：
| DP 噪声乘子 σ | △Acc (q=0.3) |
|---------------|-------------|
| 0.1 | 0.097 |
| 0.5 | 0.052 |
| 2.0 | 0.024 |

- **DP 显著提升模型抗提取能力**：随着 σ 增大（隐私增强），攻击者构建的 surrogate 性能显著落后于目标模型
- **SMC 与 FHE 不直接阻止输出泄露**：若不加输出控制（如限流、扰动），其预测结果仍可用于有效模型窃取

---

## 4. 关键结论和发现

### 论文的主要发现
1. **DP 是效率最高的选择**：
   - 响应时间与能耗极低，适合实时性要求高的 EI 场景；
   - 代价是准确性下降，尤其对复杂模型（如 AlexNet）；
   - 能有效增加模型提取难度，形成 **privacy-utility-extractability trade-off**。

2. **SMC 性能由通信主导**：
   - 计算本身开销较小，但 **round complexity 与 bandwidth 成瓶颈**；
   - 更适合局域网内低延迟网络环境；
   - 增加参与方会线性甚至超线性增加延迟。

3. **FHE 开销巨大但安全性最强**：
   - 完全隐藏输入与中间状态，提供最强的运行时保密性；
   - **计算开销高达 1000×**，必须配合轻量模型（如 LeNet-5）、低位宽量化（4–6 bit）和高性能边缘服务器；
   - 适用于对延迟容忍度较高但对数据机密性要求极高的场景。

4. **模型复杂度是决定性因素**：
   - AlexNet 在所有 PPML 方案下均表现出最差性能；
   - LeNet-5 和 SqueezeNet 更适合作为边缘加密推理的基础模型。

5. **隐私 ≠ 自动防模型窃取**：
   - SMC 与 FHE 保护了推理过程，但一旦输出暴露，仍可能被用于模型蒸馏；
   - 必须结合 **output perturbation、access control 或 query throttling** 才能全面防御 model stealing。

### 方法的局限性
- **未考虑 FL**：虽然 FL 是常见 PPML 范式，但本文聚焦 inference，故未纳入比较。
- **FHE 当前版本限制**：Concrete-ML v1.7 最大仅支持 8-bit 整数运算，严重制约模型表达能力。
- **SMC 假设半诚实敌手**：默认参与者遵守协议（semi-honest），未考虑恶意篡改行为。
- **仿真简化了调度逻辑**：使用固定启发式调度器，未探索动态负载均衡的影响。

### 未来工作方向
- **优化 PPML 系统设计**：开发自适应量化策略、混合隐私机制（如 DP+FHE）、跨层协同优化算法。
- **改进任务调度机制**：设计面向 PPML 的新型 task scheduling 算法，平衡隐私强度与服务质量（QoS）。
- **加强输出层面防护**：研究如何在不影响可用性的前提下限制模型 API 输出的信息泄露。
- **扩展至更多应用场景**：将该框架应用于视频流处理、联邦推理、多模态感知等更复杂的 EI 场景。

--- 

> 📌 **总结一句话**：  
> 在 EI 场景中，**DP 最适合追求低延迟的轻量级应用，SMC 适用于可信多方协作且网络良好的环境，而 FHE 则是高安全需求下的终极选择——但需付出巨大的性能代价**。选择何种 PPML 技术应基于具体的应用需求、资源约束与威胁模型进行权衡。

</details>

---

### 14. [Scene-Adaptive Continual Learning for CSI-based Human Activity Recognition with Mixture of Experts](https://arxiv.org/abs/2605.06447)

**Authors**: Wenhan Zheng, Yuyi Mao, Ivan Wang-Hei Ho  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.06447v1  

#### Abstract
Channel state information (CSI)-based human activity recognition (HAR) is vulnerable to performance degradation under domain shifts across varying physical environments. Continual learning (CL) offers a principled way to learn new domains sequentially while preserving past knowledge, but existing CL...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Scene-Adaptive Continual Learning for CSI-based Human Activity Recognition with Mixture of Experts

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **CSI-based Human Activity Recognition (HAR)** 中的**跨域性能退化**问题，尤其是在物理环境变化导致的**domain shift**下模型泛化能力下降的问题。传统 Continual Learning (CL) 方法在处理此类任务时面临以下挑战：
- **灾难性遗忘**（Catastrophic Forgetting）：模型在学习新场景时会覆盖旧知识；
- **推理开销线性增长**：如 MEMO 等架构扩展方法需激活所有专家模块，导致计算成本随领域数量增加而上升；
- **依赖大规模回放缓冲区**（Replay Buffer），难以部署于资源受限的边缘设备。

### 提出的新方法：SAMoE-C
作者提出 **Scene-Adaptive Mixture of Experts with Clustered Specialists (SAMoE-C)**，其核心思想是将跨域 CSI-HAR 建模为一个基于 **Mixture of Experts (MoE)** 的系统，并引入**语义路由器**（Semantic Router）实现稀疏激活。

#### 创新点：
1. **模块化 MoE 架构设计**：
   - 共享的 **Shared Backbone** 提取通用 CSI 特征；
   - 多个独立的 **Specialist Networks** 针对不同场景进行优化；
   - 通过 **Semantic Router** 动态选择最合适的专家网络进行推理。

2. **基于注意力机制的语义路由器**：
   - 使用 Temporal Attention 生成上下文向量 `c`；
   - 路由器 `G(c)` 输出概率分布并选择最优专家 `k* = argmax(p_j)`；
   - 实现**单专家激活**，显著降低推理开销。

3. **轻量级训练协议**：
   - **仅需极小回放缓冲区**（p=0.05）用于稳定路由器的域判别能力；
   - 采用三阶段解耦训练策略：
     - 初始域训练（冻结 Backbone）
     - 增量学习（训练新 Specialist）
     - 路由器更新（结合当前数据与历史缓存）

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **推理效率** | 推理成本恒定 ~199 MFLOPS/sample，远低于 MEMO 的 797.8 MFLOPS/sample |
| **抗遗忘能力** | 有效缓解 Catastrophic Forgetting，过往域平均准确率保持在较高水平 |
| **存储开销低** | 仅需 5% 历史数据作为 Replay Buffer，适合边缘部署 |
| **可扩展性强** | 支持持续新增场景而不引发参数爆炸或推理延迟累积 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **MM-Fi dataset** [13]，包含来自 **4 个不同物理环境** 的 CSI 数据：
  - D1, D2：两个 Living Rooms
  - D3, D4：两个 Meeting Rooms
- 活动类别数：**K = 27 种 human activities**
- 输入格式：CSI Tensor 形状为 `(3, 10, 114)`，经预处理后送入模型

### 实验设置
- **Continual Learning 序列**：默认顺序 `D1 → D2 → D3 → D4`，另测试替代顺序 `D3 → D1 → D2 → D4`
- **模型架构**：见 Table I，包含：
  - Shared Backbone（CNN-based）
  - Attention-based Router（ResGateNet）
  - Specialist Blocks（CNN + Bi-GRU）
- **优化器**：AdamW
- **超参数**：
  - Batch Size: 64
  - Learning Rate: 3×10⁻⁴
  - Replay Ratio: p = 0.05（即每域保留 5% 数据进 Buffer）

### 评估指标
| 指标 | 描述 |
|------|------|
| **Final Avg. HAR Acc. (%)** | 在平衡测试集上的最终平均识别准确率 |
| **Inference Cost (MFLOPS/sample)** | 单样本推理所需浮点运算量 |
| **Avg. Past Acc.** | 对已学习领域的平均保持能力（衡量抗遗忘） |
| **New Acc.** | 当前新域的学习性能 |
| **Router Val. Accuracy** | 路由器正确识别输入所属场景的能力 |

### 基线方法对比
| 方法 | 简介 |
|------|------|
| **Basic CL** | 单一模型在所有域上增量训练，无专家分离，易发生灾难性遗忘 |
| **MEMO** [8] | 多专家架构，但**所有专家同时激活**，输出融合决策，推理开销线性增长 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table III）

| Method | Final Avg. HAR Acc. (%) | Inference Cost (MFLOPS/sample) |
|--------|--------------------------|-------------------------------|
| Basic CL | 29.56 | 198.4 |
| MEMO | 87.69 | 797.8 |
| **SAMoE-C** | **81.66** | **199.1** |
| SAMoE-C (Alt. Order) | 81.24 | 199.1 |

> 注：SAMoE-C 在精度上接近 SOTA 的 MEMO（仅低约 6%），但推理成本仅为后者的 **~25%**

### 与基线方法对比结果
- **vs Basic CL**：
  - 准确率提升 **+52.1% 绝对增益**（81.66 vs 29.56）
  - 显著抑制 Catastrophic Forgetting（见 Table II：Basic CL 的 Avg. Past Acc. 下降至 3.74%，而 SAMoE-C 保持在 ~88–94%）
- **vs MEMO**：
  - 尽管 MEMO 精度更高（87.69%），但其推理开销高达 **797.8 MFLOPS**，不适合边缘部署；
  - SAMoE-C 以轻微精度损失换取 **4 倍以上的能效提升**

### 消融实验结果（Ablation Study on Replay Buffer Size）
- **p = 0**（无回放）：路由准确率仅 **25.00%**，严重偏向最新域 D4
- **p = 0.01**：路由准确率跃升至 **91.86%**
- **p = 0.05**：达到 **98.05%**，接近饱和
- **p > 0.05**：增益边际递减（如 p=0.25 达 99.32%）

👉 结论：**极小回放缓冲区即可极大提升路由稳定性**，验证了训练协议的有效性和实用性。

---

## 4. 关键结论和发现

### 主要发现
1. **SAMoE-C 成功实现了高效且可扩展的跨域 CSI-HAR**：
   - 通过 MoE 架构与语义路由器结合，兼顾高精度与低推理成本；
   - 实现了“**constant inference cost**”特性，适用于长期演进的实际场景。

2. **稀疏激活机制至关重要**：
   - 仅激活一个 Specialist 显著降低了计算负担；
   - 注意力引导的 Router 可精准识别输入场景，判别准确率达 98%+。

3. **轻量回放足以维持路由稳定性**：
   - 仅需 5% 历史数据即可防止 Catastrophic Forgetting in routing；
   - 为边缘设备提供了可行的部署方案。

### 方法的局限性
- **硬路由选择**（Hard Routing）可能导致次优决策，未探索 Soft Routing 或 ensemble 策略；
- Backbone 在初始域固定，可能限制后续域的特征表达能力；
- 当前实验仅限 4 个场景，更大规模连续学习下的表现有待验证。

### 未来工作方向
- **预训练更强大的 Shared Backbone**，提升跨域泛化能力；
- 探索 **Soft Routing Mechanism**，允许多个专家协同参与预测；
- 扩展至更多动态场景（如移动用户、多人交互等）；
- 进一步压缩模型尺寸，适配更低功耗 IoT 设备。

---

> ✅ 总结：  
> **SAMoE-C 是一种面向实际部署的、高效可扩展的 CSI-based HAR 框架**。它通过 **MoE 架构 + 语义路由器 + 轻量训练协议** 的联合设计，在保持高识别精度的同时，实现了恒定推理成本与强抗遗忘能力，为无线感知系统的持续自适应演化提供了新范式。

</details>

---

### 15. [FalconGEMM: Surpassing Hardware Peaks with Lower-Complexity Matrix Multiplication](https://arxiv.org/abs/2605.06057)

**Authors**: Honglin Zhu, Jiaping Cao, Jiang Shao, Siyuan Feng, Qian Qiu, Peng Chen, Xu Zhang, Yixian Zhou, Man Lung Yiu, Guang Ji, Minwen Deng, Wenxi Zhu, Jintao Meng  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.06057v1  

#### Abstract
Peak breaking Matrix Multiplication is a promising technique to improve the performance of DL, especially in LLM training and inference. We present FalconGEMM, a cross-platform framework that automates the deployment, optimization, and selection of Lower-Complexity Matrix Multiplication Algorithms (...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FalconGEMM: Surpassing Hardware Peaks with Lower-Complexity Matrix Multiplication

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **GEMM**（General Matrix Multiplication）在现代深度学习（DL）尤其是大语言模型（LLM）训练与推理中占据主导地位，但其 $O(N^3)$ 的计算复杂度已成为性能瓶颈。尽管如 **cuBLAS**、**Intel MKL** 等库已将标准 GEMM 性能推向硬件峰值，进一步提升空间有限。

**Lower-Complexity Matrix Multiplication Algorithms**（LCMAs），如 Strassen 算法、AlphaTensor 发现的算法等，理论上可通过减少乘法次数降低计算复杂度（如降至 $O(N^{\log_2 7})$），但在实际部署中面临三大挑战：
1. **跨平台可移植性差**：LCMAs 数据依赖复杂，难以高效适配多种硬件（GPU/CPU、不同架构）。
2. **内存开销高**：中间张量需写入片外内存，带宽消耗抵消了算力节省。
3. **决策困难**：LCMA 并非总是更快，其优势取决于矩阵形状、硬件带宽与算力比，缺乏轻量级选择机制。

### 提出的新方法与创新
作者提出 **FalconGEMM** —— 一个跨平台框架，系统性解决上述问题，实现“突破硬件峰值”的 GEMM 加速。其核心是三大模块：

#### （1）Deployment Module（部署模块）
- 引入四阶段 LCMA 流程（Combine A/B → GEMM → Combine H）。
- 采用 **自动化代码生成**（基于 TVM、Triton、TileLang 等编译器），将算法逻辑与硬件解耦，支持自动适配不同硬件（NVIDIA、AMD、Intel、ARM）、数据类型（FP32/BF16/FP16/FP8）和分块策略，极大提升可移植性。

#### （2）Execution Module（执行模块）
- 提出 **Group-Parallel Optimization**：
  - 将计算按“组”（Group）组织，利用组内数据局部性，在片上内存完成 Combine A/B 和 GEMM+Combine H 的融合，避免中间结果落盘。
- 进一步优化：
  - **Split-Group Parallelism**：通过持久化 Kernel 技术实现细粒度调度，缓解粗粒度 Group 调度导致的资源浪费。
  - **Cache-Aware Scheduling**：重排计算顺序，提高 L2 缓存命中率，避免缓存抖动。

#### （3）Decision Module（决策模块）
- 构建轻量级 **理论性能模型**，基于 **算术强度分析**（Arithmetic Intensity Analysis）预测 LCMA 是否优于标准 GEMM。
- 输入矩阵形状 $(M,N,K)$ 和硬件参数（FLOPS、带宽 $\beta$），自动选择最优 LCMA 或回退到标准 GEMM。

### 相比现有方法的优势
- **首次实现跨平台、高性能、自动化的 LCMA 部署**，解决了以往实现碎片化、难移植的问题。
- **真正实现“Peak Breaking”**：通过算法级优化突破硬件算力上限，而非仅靠工程优化逼近峰值。
- **端到端稳定加速**：在真实 LLM 推理中表现出色，尤其在小序列长度下仍能有效加速，克服了 AlphaTensor 等方法在小规模输入下的性能退化问题。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 从三个开源 LLM 中提取线性层的权重形状：
  - **DeepSeek-R1**
  - **Qwen3.5-397B**
  - **HunyuanVideo**
- 生成 960 组测试形状 $(M, N, K)$，其中 $M$（序列长度）从 512 到 20480，步长为 512。

### 实验设置和评估指标
- **硬件平台**：
  - **GPU**: NVIDIA H20 (Hopper), A100 (Ampere)
  - **CPU**: Intel Xeon Platinum 8255C (x86), AMD EPYC 9K84 (x86), AWS EC2 M7g (ARM Neoverse-V1)
- **数据类型**：FP32, BF16, FP16, FP8（支持 block-wise scaling）
- **评估指标**：
  - **Effective TFLOPS/GFLOPS**：以标准 GEMM 的浮点操作数 $2MNK$ 计算，即使使用 LCMA 也能公平比较，允许性能“超过硬件峰值”。

### 基线方法对比
- **标准 GEMM 库**：
  - GPU: **cuBLAS**, **CUTLASS**
  - Intel CPU: **Intel MKL**
  - AMD CPU: **OpenBLAS**
  - ARM CPU: **ACL** (Arm Compute Library)
- **LCMA 对手**：
  - **AlphaTensor**（基于 JAX 实现）
  - 其他专用实现（如支持 FP8 的 CUTLASS）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- FalconGEMM 在多个平台和数据类型上均 **超越硬件理论峰值**，实现“Peak Breaking”。
- 在 **960 个 LLM 层形状** 上平均表现显著优于所有基线。

### 与基线方法的对比结果
| 对比项 | 性能增益 |
|--------|----------|
| **vs. 最佳 GEMM 库**（cuBLAS/MKL/CUTLASS 等） | **+7.59% ~ +17.85%** |
| **vs. AlphaTensor**（唯一 LCMA 对手） | **+12.41% ~ +55.61%** |

具体案例：
- 在 H20 (FP8) 上，相比最佳 GEMM 库平均快 **17.85%**。
- 在 ARM CPU 上，相比 AlphaTensor 快 **55.61%**。

### 消融实验结果
#### （1）Step-wise Evaluation（执行模块有效性验证）
在 H20 (BF16) 上使用 Strassen 算法进行逐步优化：
- **Algorithm 1**（基础实现）：+5.32%
- **+ Group-Parallel**：初步融合，但存在负载不均衡
- **+ Split-Group**：解决调度问题，小矩阵更优
- **+ Cache-Aware**：最终优化，**额外带来最多 7.18% 的加速**，整体较 cuBLAS 提升 **3.07% ~ 17.13%**

#### （2）End-to-End LLM 推理加速
在 PyTorch 中替换 GEMM 后端：
- **预填充阶段**（prefill stage）平均加速：
  - A100 (FP32): **+18.12%**
  - H20 (BF16): **+12.24%**
  - H20 (FP8): **+11.46%**
- 在 FP8 小 $M$ 场景下，通过融合量化操作，性能提升 **高达 46%**。

#### （3）Roofline 分析
- 验证了 **Decision Module** 的有效性：在高算术强度区域选择高 R 的 LCMA，在低 AI 区域回退到标准 GEMM。
- 相比标准 GEMM，FalconGEMM 实现 **约 19.31% 的性能增益**；相比 Strassen 算法，增益为 **11.37%**。

---

## 4. 关键结论和发现

### 主要发现
1. **LCMA 可实用化**：通过系统性的部署、执行和决策优化，LCMA 的理论优势可在生产环境中稳定释放。
2. **算法创新可突破硬件极限**：FalconGEMM 证明，通过降低算法复杂度，可以真正“超越硬件峰值”，为 DL 提供新的性能增长路径。
3. **端到端稳定性强**：在真实 LLM 推理中，即使在小批量或小序列长度下，仍能保持显著加速，解决了以往 LCMA 框架的“冷启动”问题。
4. **数值精度更高**：由于中间结果保留在高精度片上内存中，FalconGEMM 的相对误差比 AlphaTensor **低约 17.2%**。

### 方法的局限性
- **仅适用于稠密矩阵乘法**：未处理稀疏矩阵场景。
- **对极低算术强度任务无效**：当内存带宽严重受限时，LCMA 的额外访存反而成为瓶颈。
- **实现复杂度高**：依赖先进的编译器技术（TVM/Triton），对开发者门槛较高。

### 未来工作方向
- 扩展至 **稀疏-稠密 GEMM** 和 **注意力机制中的特殊矩阵运算**。
- 支持更多 **新型数据格式**（如 INT4、NF4）和 **量化感知训练**（QAT）场景。
- 探索 **动态自适应调度**，在运行时根据系统负载调整 LCMA 策略。
- 开源代码，推动社区共建 LCMA 算法库。

--- 

> **总结**：FalconGEMM 是首个将 LCMA 从理论带入大规模生产实践的系统，通过 **Deployment + Execution + Decision** 三位一体的设计，实现了跨平台、高性能、自动化的“Peak-Breaking GEMM”，为下一代 LLM 加速提供了全新的算法级解决方案。

</details>

---

### 16. [Expert Routing for Communication-Efficient MoE via Finite Expert Banks](https://arxiv.org/abs/2605.05278)

**Authors**: Mohammad Reza Deylam Salehi, Ali Khalesi  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.05278v1  

#### Abstract
Resource-efficient machine learning increasingly uses sparse Mixture-of-Experts (MoE) architectures, where the gate acts as both a learning component and a routing interface controlling computation, communication, and accuracy. Motivated by finite-rate interpretations of MoE gating, we treat the gat...

---

### 17. [HCInfer: An Efficient Inference System via Error Compensation for Resource-Constrained Devices](https://arxiv.org/abs/2605.05819)

**Authors**: Shen Xu, Xiangwen Zhuge, Zhe Xu, Yingkun Hu, Zheng Yang, Yunhao Liu  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.05819v1  

#### Abstract
LLMs often struggle with memory-constrained deployment on consumer-grade hardware due to their massive parameter sizes. While existing solutions such as model compression and offloading improve deployment feasibility, they often suffer from substantial accuracy degradation or severe throughput bottl...

---

### 18. [VisMMOE: Exploiting Visual-Expert Affinity for Efficient Visual-Language MoE Offloading](https://arxiv.org/abs/2605.05899)

**Authors**: Cheng Xu, Xiaofeng Hou, Jiacheng Liu, Chao Li  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.05899v1  

#### Abstract
Large-scale vision-language mixture-of-experts (VL-MoE) models provide strong multimodal capability, but efficient deployment on memory-constrained platforms remains difficult. Existing MoE offloading systems are largely designed for text-centric workloads and become much less effective for visual-h...

---

### 19. [Tackling the Data-Parallel Load Balancing Bottleneck in LLM Serving: Practical Online Routing at Scale](https://arxiv.org/abs/2605.06113)

**Authors**: Tianci Bu, Yuan Lyu, Zixi Chen, Chendong Song, Hong Liang, Tsepten Gurung, Yuwei Fan, Yinyu Ye, Zijie Zhou  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.06113v1  

#### Abstract
Data-parallel (DP) load balancing has emerged as a first-order bottleneck in large-scale LLM serving. When a model is sharded across devices via tensor parallelism (TP) or expert parallelism (EP) and replicated across many DP workers, every decode step ends in a synchronization barrier whose latency...

---

### 20. [Towards Scalable One-Step Generative Modeling for Autoregressive Dynamical System Forecasting](https://arxiv.org/abs/2605.05540)

**Authors**: Tianyue Yang, Xiao Xue  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.05540v1  

#### Abstract
Fast surrogate modeling for high-dimensional physical dynamics requires more than low short-term error: useful models must roll out efficiently while preserving the statistical structure of long trajectories. Neural operators provide inexpensive autoregressive forecasts but can drift in turbulent re...

---

### 21. [Towards Generation-Efficient Uncertainty Estimation in Large Language Models](https://arxiv.org/abs/2605.06053)

**Authors**: Mingcheng Zhu, Yu Liu, Tingting Zhu  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.06053v1  

#### Abstract
Uncertainty estimation is important for deploying LLMs in high-stakes applications such as healthcare and finance, where hallucinations can appear fluent and plausible while being factually incorrect, making it difficult for users to judge whether an output should be trusted. Existing methods requir...

---

### 22. [SDFlow: Similarity-Driven Flow Matching for Time Series Generation](https://arxiv.org/abs/2605.05736)

**Authors**: Wei Li, Shibo Feng, Pengcheng Wu, Min Wu, Peilin Zhao  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.05736v1  

#### Abstract
Vector quantization (VQ) with autoregressive (AR) token modeling is a widely adopted and highly competitive paradigm for time-series generation. However, such models are fundamentally limited by exposure bias: during inference, errors can accumulate across sequential predictions, leading to pronounc...

---

### 23. [The Cost of Context: Mitigating Textual Bias in Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2605.05594)

**Authors**: Hoin Jung, Xiaoqian Wang  
**Category**: cs.CL  
**Published**: 2026-05-08  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.05594v1  

#### Abstract
While Multimodal Large Language Models (MLLMs) are increasingly integrated with Retrieval-Augmented Generation (RAG) to mitigate hallucinations, the introduction of external documents can conceal severe failure modes at the instance level. We identify and formalize the phenomenon of recorruption, wh...

---

### 24. [A Unified Pair-GRPO Family: From Implicit to Explicit Preference Constraints for Stable and General RL Alignment](https://arxiv.org/abs/2605.06375)

**Authors**: Hao Yu  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.06375v1  

#### Abstract
Large language model (LLM) alignment via reinforcement learning from human preferences (RLHF) suffers from unstable policy updates, ambiguous gradient directions, poor interpretability, and high gradient variance in mainstream pairwise preference learning paradigms. To systematically address these l...

---

### 25. [FedFrozen: Two-Stage Federated Optimization via Attention Kernel Freezing](https://arxiv.org/abs/2605.06446)

**Authors**: Junye Du, Zhenghao Li, Yushi Feng, Long Feng  
**Category**: cs.LG  
**Published**: 2026-05-08  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.06446v1  

#### Abstract
Federated learning with heterogeneous clients remains a significant challenge for deep learning, primarily due to client drift arising from inconsistent local updates. Existing federated optimization methods typically address this issue through objective-level regularization or update-correction mec...

---

### 26. [Causal Probing for Internal Visual Representations in Multimodal Large Language Models](https://arxiv.org/abs/2605.05593)

**Authors**: Zehao Deng, Tianjie Ju, Zheng Wu, Liangbo He, Jun Lan, Huijia Zhu, Weiqiang Wang, Zhuosheng Zhang  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.05593v1  

#### Abstract
Despite the remarkable success of Multimodal Large Language Models (MLLMs) across diverse tasks, the internal mechanisms governing how they encode and ground distinct visual concepts remain poorly understood. To bridge this gap, we propose a causal framework based on activation steering to actively ...

---

### 27. [Knee Osteoarthritis Severity Grading Using Optimized Deep Learning and LLM-Driven Intelligent AI on Computationally Limited Systems](https://arxiv.org/abs/2605.05731)

**Authors**: Dayam Nadeem,  Neha, Safdar Mustafa, Adnan Alvi, Mohd Hussain  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.05731v1  

#### Abstract
Knee osteoarthritis (KOA) is among the musculoskeletal disorders that considerably restrict joint mobility, cause severe chronic pain and impact negatively on quality life. It is one of the persistent health issues worldwide. Generally, subjectivity and inter-observer variability undermine conventio...

---

### 28. [SANEmerg: An Emergent Communication Framework for Semantic-aware Agentic AI Networking](https://arxiv.org/abs/2605.05861)

**Authors**: Yong Xiao, Haoran Zhou, Yujie Zhou, Marwan Krunz  
**Category**: cs.AI  
**Published**: 2026-05-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.05861v1  

#### Abstract
Future networking systems are envisioned to become part of an agentic AI-native ecosystem in which a vast number of heterogeneous and specialized AI agents cooperate seamlessly to fulfill complex user requirements in real time. However, traditional networking paradigms are characterized by a rigid d...

---

### 29. [Long Context Pre-Training with Lighthouse Attention](https://arxiv.org/abs/2605.06554)

**Authors**: Bowen Peng, Subho Ghosh, Jeffrey Quesnelle  
**Category**: cs.CL  
**Published**: 2026-05-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.06554v1  

#### Abstract
Training causal transformers at extreme sequence lengths is bottlenecked by the quadratic time and memory of scaled dot-product attention (SDPA). In this work, we propose Lighthouse Attention, a training-only symmetrical selection-based hierarchical attention algorithm that wraps around ordinary SDP...

---

### 30. [ADELIA: Automatic Differentiation for Efficient Laplace Inference Approximations](https://arxiv.org/abs/2605.06392)

**Authors**: Afif Boudaoud, Lisa Gaedke-Merzh\"auser, Alexandros Nikolaos Ziogas, Vincent Maillou, Alexandru Calotoiu, Marcin Copik, H{\aa}vard Rue, Mathieu Luisier, Torsten Hoefler  
**Category**: cs.DC  
**Published**: 2026-05-08  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.06392v1  

#### Abstract
Spatio-temporal Bayesian inference drives environmental and health sciences using latent Gaussian models. Integrated Nested Laplace Approximations (INLA) enable inference for these models at HPC scale but rely on derivative-based optimization over $d$ hyperparameters. State-of-the-art INLA implement...

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
