# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-20 08:47:28 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Understanding Inference Scaling for LLMs: Bottlenecks, Trade-offs, and Performance Principles](https://arxiv.org/abs/2605.19775)

**Authors**: Moiz Arif, Avinash Maurya, Sudharshan Vazhkudai, Bogdan Nicolae  
**Category**: cs.DC  
**Published**: 2026-05-20  
**Score**: 17.0  
**Type**: new  
**ArXiv ID**: 2605.19775v1  

#### Abstract
The transition from standard generative AI to \emph{reasoning-centric architectures}, exemplified by models capable of extensive Chain-of-Thought~(CoT) processing, marks a fundamental paradigm shift in system requirements. Unlike traditional workloads dominated by compute-bound prefill, reasoning wo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Understanding Inference Scaling for LLMs: Bottlenecks, Trade-offs, and Performance Principles

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本论文系统性地研究了**推理密集型大语言模型**（reasoning-centric LLMs）在大规模部署中的**推理扩展瓶颈**（inference scaling bottlenecks）。传统 LLM 推理以短文本生成为主，计算瓶颈集中在 **prefill 阶段**（TTFT），而现代推理模型（如 DeepSeek-R1、OpenAI o1）依赖 **Chain-of-Thought (CoT)** 生成数千甚至上万 token 的中间推理链，导致资源需求发生根本转变——从 **compute-bound 转向 capacity-bound**。

具体问题包括：
- **KV-cache 容量墙**（Capacity Wall）：长输出序列导致 Key-Value 缓存呈线性增长，迅速耗尽 GPU 的 High Bandwidth Memory (HBM)，引发调度器频繁 preemption 和 re-computation。
- **标准并行策略失效**：传统的 **Data Parallelism (DP)** 在小模型上有效，但在长上下文推理中因 **KV-cache 分片** 导致“**stranded capacity**”（内存碎片化浪费）。
- **缺乏统一的并行决策框架**：不同架构（dense vs. MoE）对并行策略敏感，最优配置不再通用。

### 提出了什么新方法或新思路
论文并未提出新的算法，而是提出了一个**基于系统级性能特征的推理扩展决策框架**，其核心思想是：
- **识别“并行性转换点”**（Parallelism Transition Point）：确定在何种模型规模和序列长度下，应从 DP 转向 **Tensor Parallelism (TP)** 或 **Hybrid Parallelism**。
- **量化“推理鸿沟”**（Reasoning Gap）：揭示推理任务中，瓶颈已从 prefill 的 TTFT 转移至 decode 的 TPOT，即 **memory capacity 和 bandwidth 成为决定性因素**。
- **提出架构感知的并行策略选择原则**：
  - 对于 **dense 模型**（如 Llama-405B）：高 degree TP 更优，因其聚合 memory capacity 和 bandwidth。
  - 对于 **sparse MoE 模型**（如 DeepSeek-R1-671B）：低 degree TP + Pipeline Parallelism (PP) 的混合策略更佳，以降低同步开销。

### 相比现有方法的优势
| 维度 | 现有方法 | 本文优势 |
|------|--------|--------|
| **分析视角** | 关注 kernel 优化、KV 压缩等微观层面 | 从系统级容量、带宽、调度动态出发，揭示宏观瓶颈 |
| **并行策略指导** | 通常推荐固定策略（如纯 DP 或纯 TP） | 提出**动态、模型感知的并行选择框架**，适应不同规模与架构 |
| **适用场景** | 主要针对标准生成任务 | 明确聚焦 **reasoning-heavy 工作负载**，填补该领域空白 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **Meta's Natural Reasoning dataset**：包含 115 万个需要多跳推理和常识推断的样本。
  - 特征：输入序列较短（77% < 150 tokens），但输出极长（45% > 5000 tokens），完美模拟推理任务的“**输入短、输出长**”特性。

### 实验设置和评估指标
#### 硬件平台
- **节点配置**：8× NVIDIA H200 GPU（SXM5），每卡 141GB HBM3e，峰值带宽 4.8 TB/s。
- **互联**：第四代 NVLink，单向 900 GB/s，支持高效 all-reduce。
- **主机**：双路 Intel Xeon Platinum 8558P，2TB DDR5。

#### 软件与引擎
- **推理引擎**：vLLM v1，启用 **PagedAttention** 以减少 KV-cache 内部碎片。
- **调度策略**：默认 FCFS，调整 `max_num_batched_tokens` 和 `max_num_seqs` 控制并发。

#### 模型范围
| 模型类型 | 具体模型 | 参数量 | 架构特点 |
|--------|--------|-------|---------|
| 小规模推理模型 | DeepSeek-R1-Distill-Llama/Qwen | 8B–70B | Dense + GQA |
| 前沿稠密模型 | Llama-3.1-405B | 405B | Dense + GQA，KV footprint ~1.05 MB/token |
| 前沿稀疏模型 | DeepSeek-R1-671B | 671B | MoE + **MLA**（Multi-Head Latent Attention），KV 压缩显著 |

#### 评估指标
| 指标 | 含义 | 用途 |
|------|------|------|
| **TTFT** (Time-To-First-Token) | 请求到首 token 的延迟 | 反映 prefill 阶段效率 |
| **TPOT** (Time-Per-Output-Token) | 每个输出 token 的平均延迟 | 反映 decode 阶段 memory bandwidth 效率 |
| **Generation Throughput** | 每秒生成 token 数 | 系统级吞吐能力 |
| **E2E Latency** | 请求总延迟 | 综合服务质量 |
| **KV-Cache Utilization** | KV-cache 占用率 | 判断是否接近“推理悬崖”（Reasoning Cliff） |
| **HBM Bandwidth Utilization** | 内存带宽利用率 | 判断是否 bandwidth-bound |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 最优并行策略 | E2E Latency (2k batch) | 性能增益 |
|------|-------------|---------------------|--------|
| DeepSeek-8B | DP=8 | 332s | — |
| DeepSeek-32B | DP=4 + TP=2 | 484s | 比纯 DP (857s) 快 **1.77×**，比纯 TP (686s) 快 **1.42×** |
| Llama-405B | TP=8 | 986s | 比 PP=8 (7537s) 快 **7.6×** |
| DeepSeek-R1-671B | PP=4 + TP=2 | 1663s | 比纯 TP=8 (2047s) 快 **1.23×** |

### 与基线方法的对比结果
- **DP vs. TP 在 32B 模型上的拐点**：
  - DP 扩展效率下降，8 GPU 仅实现 4.9× 加速。
  - TP 实现 6.15× 加速，主因是 **释放了 per-GPU 的 HBM 容量**（从 ~77GB 可用于 KV 提升至 ~133GB），避免 preemption。
- **PP 在 dense 模型上表现灾难性**：
  - Llama-405B 使用 PP=8 时，E2E 延迟高达 7537s，远差于 TP=8 的 986s，原因是 **KV 容量不足导致 pipeline bubbles 无法隐藏**。
- **MoE 模型对同步开销敏感**：
  - DeepSeek-R1-671B 使用 TP=8 时，all-reduce 开销占比过高，而 PP+TP 混合策略通过减少 TP degree 降低了通信瓶颈。

### 消融实验结果
- **Concurrency-Capacity Trade-off**：
  - 在 DeepSeek-8B 上，将 `max_num_seqs` 从 1K 增加到 10K，初期 throughput 提升，但当 KV 利用率达 100% 后，scheduler 开始 preemption，导致 **throughput 崩溃**。
  - 存在一个 **concurrency sweet spot (~2K seqs)**，此时 E2E latency 最低，平衡了排队延迟（TTFT）与执行延迟（TPOT）。
- **Batch Size Scaling with DP**：
  - 即使使用 8 GPU DP，aggregate HBM 达 1.1TB，但每个 GPU 仍独立管理 KV-cache，导致 **per-replica 容量墙依然存在**，无法缓解长序列压力。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **推理任务已进入 Capacity-Bound 时代**：
   - 长推理链导致 KV-cache 成为主要内存消耗者，**HBM 容量而非 FLOPs 成为推理吞吐的决定性因素**。
   
2. **Data Parallelism 存在“容量陷阱”**（Capacity Trap）：
   - DP 不聚合 memory，每个 replica 独立受限于本地 HBM，易因 KV 分片导致 **stranded capacity** 和 **preemption thrashing**。

3. **存在“并行性转换点”**：
   - 小模型（如 8B–14B）适合 DP。
   - 中等模型（如 32B）进入 **TP 优势区**，因其释放的 HBM 容量可抵消通信开销。
   - 大模型需根据架构选择：
     - **Dense 模型** → 高 degree TP（聚合 bandwidth + capacity）
     - **Sparse MoE 模型** → Hybrid PP+TP（降低 TP degree 以减少同步开销）

4. **MLA 架构具有显著优势**：
   - DeepSeek-R1 使用 **Multi-Head Latent Attention**，大幅压缩 KV-cache，使其在相同参数量下内存占用远低于 dense 模型。

5. **Prefill 与 Decode 资源需求严重失衡**：
   - Prefill 是 compute-bound，Decode 是 memory-bound。
   - 系统大部分时间处于 decode 阶段，导致 **compute units 利用率低下**，即使 latency 很高。

### 方法的局限性
- **未考虑 KV offload / tiered memory**：分析假设 KV-cache 完全驻留 HBM，未引入 CPU memory 或 CXL/NVMe offload，可能低估实际系统的弹性。
- **单节点分析为主**：实验集中在 8-GPU NVLink 节点内，跨节点扩展（如 DP across racks）的影响未深入探讨。
- **静态批处理假设**：未模拟动态到达请求下的自适应调度策略。

### 未来工作方向
- **硬件-软件协同设计**：
  - 推动 **disaggregated architecture**，将 prefill 与 decode 分离到不同类型的加速器上（如 high-FLOP vs. high-bandwidth）。
- **智能调度与 Admission Control**：
  - 开发 **KV-aware scheduler**，在 admission 时预测未来 KV 增长，预留 capacity。
- **新型内存架构**：
  - 探索 **3D-stacked memory**、**CXL 扩展内存池**、**HBF**（High Bandwidth Flash）等技术，缓解 capacity 与 bandwidth 压力。
- **Agentic AI 的系统挑战**：
  - 支持多步、状态化 agent 推理，需跨 GPU、CPU、存储层级进行 **KV state migration 与 tiering**。

--- 

> **总结一句话**：  
> 随着 LLM 进入推理时代，**内存容量与带宽已成为比算力更重要的瓶颈**；未来的推理系统必须从“算力中心”转向“**内存中心**”设计，并采用**架构感知的混合并行策略**，才能有效应对“推理悬崖”。

</details>

---

### 2. [TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload](https://arxiv.org/abs/2605.20179)

**Authors**: Zhiben Chen, Youpeng Zhao, Yang Sui, Jun Wang, Yuzhang Shang  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 14.5  
**Type**: new  
**ArXiv ID**: 2605.20179v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have emerged as a competitive alternative to autoregressive (AR) models, offering better hardware utilization and bidirectional context through parallel block-level decoding. However, as dLLMs continue to scale up with mixture-of-experts (MoE) architectures, t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着 **Diffusion Large Language Models (dLLMs)** 规模扩大并采用 **Mixture-of-Experts (MoE)** 架构，其在资源受限设备（如边缘计算平台）上的高效部署面临挑战。现有方法存在以下瓶颈：
- **频繁专家迁移** 导致巨大的 **GPU-CPU I/O 开销**；
- **将 token 路由到 CPU 专家** 执行会导致严重的 **CPU 计算瓶颈**，GPU 利用率低下。

这些问题使得 MoE-dLLM 在单 GPU-CPU 系统中推理效率严重受限。

---

### 🚀 提出的新方法：TIDE
作者提出 **TIDE** —— 一种 **I/O-aware、无需训练、无损的 MoE-dLLM 推理优化系统**，其核心思想是：

#### 创新洞察（Key Insight）
在 dLLM 的块级去噪过程中，相邻步骤之间的 **expert routing 具有高度时间稳定性**（temporal stability），即连续 denoising 步骤激活的专家集合高度相似（cosine similarity > 0.95 即使相隔 5 步仍成立）。  
👉 这意味着可以**复用专家放置策略**，避免每步都进行昂贵的专家迁移。

#### 核心机制
- **Interval-based Expert Refresh Strategy**：仅每隔 `T` 个步骤执行一次专家刷新（refresh step），其余为“跳过步骤”（skipped steps），期间复用当前 GPU 上的专家布局。
- **智能专家调度**：在 refresh step 中，基于 token hit count 动态选择最常被调用的 CPU 专家迁移到 GPU。
- **异步执行流水线**：当 token 被路由到 CPU 专家时，GPU 不阻塞，实现计算重叠（overlap CPU computation with GPU execution）。

#### 数学建模优化
将最优刷新间隔 $ T $ 的选择形式化为一个 **约束数学规划问题**（constrained Mathematical Programming, MP）：
$$
\min_T \left[ C_{IO} \cdot B \cdot T \cdot (1 - (1-d)^T) + C_{CPU} \cdot T \cdot B \cdot f(T) \right]
$$
其中：
- 第一项表示 I/O 迁移开销（随 $ T $ 增大而减小）
- 第二项表示 CPU 计算延迟（随 $ T $ 增大而增加）
通过硬件性能分析 + 贪心搜索求解最优 $ T $

---

### 🔍 相比现有方法的优势
| 特性 | TIDE | Prior Methods (e.g., Mixtral-Offload, Fiddler) |
|------|------|-----------------------------------------------|
| 是否需要模型训练 | ❌ 否（lossless） | ❌ 否 |
| 是否引入精度损失 | ❌ 否 | ❌ 否 |
| I/O 开销控制 | ✅ 显著降低（interval-based refresh） | ❌ 每步迁移或静态放置导致高开销 |
| GPU 利用率 | ✅ 高（减少空闲等待） | ⚠️ 受限于 CPU 计算或 I/O 瓶颈 |
| 自适应能力 | ✅ 动态调整专家分布 | ⚠️ 固定策略，无法利用 routing 局部性 |

> 💡 **本质优势**：TIDE 提供了一种“免费午餐”式的加速方案（free-lunch acceleration），不牺牲准确性即可提升吞吐量。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **sanitized MBPP**（Mostly Basic Python Problems）数据集进行评估。
- 来源于 `1m_eval_harness` 库，适用于代码生成任务。

### 💻 实验平台
| 组件 | 配置 |
|------|------|
| GPU | NVIDIA A100 40GB（用于 LLaDA2.0-mini）、H100 80GB（用于 LLaDA2.0-flash） |
| CPU | 48-core Intel CPU |
| 内存 | 1024 GB DDR4 主机内存 |
| 软件栈 | PyTorch 2.9, CUDA 12.8, HuggingFace Transformers, dInfer |

### 🧪 模型配置
- **LLaDA2.0-mini**: 16B 参数 + 1B 激活参数（top-k=8），共 256 个 FFN experts
- **LLaDA2.0-flash**: 100B 参数 + 6B 激活参数，同样 256 个 experts
- Block size: 默认 32，部分实验扩展至 64/128
- Generation length: 256 / 1024 tokens

### 🎯 评估指标
- **Throughput (token/s)**：每秒解码的 token 数量（越高越好）
- **GPU Expert Hit Rate**：在 GPU 上命中所需专家的比例
- **I/O Traffic** 和 **CPU Computation Latency** 分解分析

### 🔁 基线方法对比
| 基线 | 方法描述 |
|------|----------|
| **Mixtral-Offload** [Eliseev and Mazur, 2023] | 每个 denoising step 都进行专家迁移（full refresh）→ 高 I/O 开销 |
| **Fiddler** [Kamahori et al., 2024] | 静态专家放置，所有未驻留专家均交由 CPU 处理 → 容易成为 CPU 瓶颈 |

> ⚠️ 注意：这些基线原本针对 AR-MoE 设计，并非专为 dLLM-MoE 优化。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Model | GPU Expert Budget | TIDE Throughput (token/s) | Best Baseline (Mixtral/Fiddler) | Speedup |
|-------|-------------------|----------------------------|-------------------------------|--------|
| LLaDA2.0-mini | 128 experts (18GB) | **2.44** | 1.91 (Mixtral) | **1.28×** |
| LLaDA2.0-mini | 64 experts (10GB) | **2.11** | 1.81 (Fiddler) | **1.17×** |
| LLaDA2.0-flash | 64 experts (55GB) | **1.73** | 1.35 (Mixtral) | **1.28×** |
| LLaDA2.0-flash | 32 experts (30GB) | **1.24** | 1.01 (Mixtral) | **1.23×** |

> ✅ **最高达 1.4× ~ 1.5× 吞吐提升**

---

### 🔬 消融实验与敏感性分析（Ablation Study）

#### ✅ 最优刷新间隔 $ T $ 的影响（Table 2 & Figure 4）
- **$ T=1 $**（即 Mixtral-Offload）：I/O 开销最大，但 CPU 计算最小
- **随机 $ T $**：性能波动大，平均低于 TIDE
- **Optimal $ T $**（由 MP 求解）：平衡 I/O 与 CPU 成本，获得最佳总延迟
- 结果显示：TIDE 在不同 block size（32/64）、GPU budget（32~128）下始终优于其他策略，**最多提速 1.4×**

#### ✅ 敏感性研究（Figure 5）
- **Block Size 增加** → TIDE 性能增益更明显（因并行度更高，routing 更稳定）
- **GPU Memory Constraint 放宽** → TIDE 与 Mixtral-offload 均受益，但 Fiddler 因 CPU 瓶颈难以扩展
- **Confidence Threshold 变化**（0.7~0.95）→ TIDE 始终保持约 **1.4× 平均加速**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Expert Routing 具有强时间局部性**：dLLM 在 block 内的多个 denoising steps 中 expert 激活模式高度相似（>0.95 cosine sim），支持跨步复用。
2. **Interval-based refresh 显著降低 I/O 开销**：相比每步迁移，周期性更新可节省大量 GPU-CPU 数据传输。
3. **TIDE 实现 lossless 加速**：不修改模型结构、router 或权重，输出完全一致，真正实现“零成本”性能提升。
4. **在资源受限场景下表现优异**：尤其在 GPU memory 紧张时，TIDE 的智能调度优势更加突出。

---

### ⚠️ 局限性（Limitations）
1. **仅考虑块内相似性**：未探索跨 block 的 routing 相似性，可能进一步优化长期调度。
2. **实验平台有限**：目前只在 NVIDIA GPU + Intel CPU 上验证，尚未测试 AMD GPU 或 ARM 架构（如移动 NPU）。
3. **局限于单节点系统**：未扩展到 multi-GPU 或分布式 expert parallelism 场景。

---

### 🔮 未来工作方向
1. **跨 block routing 分析**：挖掘更长时间尺度下的稳定性，设计跨块专家缓存机制。
2. **多 GPU / 多节点扩展**：将 TIDE 思想应用于分布式 MoE 推理，结合 tensor parallelism 和 expert parallelism。
3. **支持更多硬件架构**：适配 AMD Instinct GPU、Apple Silicon（M系列芯片）、Qualcomm NPU 等边缘设备。
4. **动态自适应 $ T $ 控制**：在线学习 routing drift rate，实时调整 refresh interval。

---

## ✅ 总结一句话
> **TIDE 利用 dLLM-MoE 中 expert routing 的时间稳定性，提出一种无需训练、无损且高效的 interval-based 专家调度机制，通过数学建模优化 I/O 与计算的权衡，在单 GPU-CPU 系统上实现了高达 1.5× 的吞吐提升，为资源受限环境下的 MoE-dLLM 部署提供了实用解决方案。**

</details>

---

### 3. [FlexDraft: Flexible Speculative Decoding via Attention Tuning and Bonus-Guided Calibration](https://arxiv.org/abs/2605.20022)

**Authors**: Yaojie Zhang, Jianuo Huang, Junlong Ke, Yuhang Han, Yongji Long, Tianchen Zhao, Biqing Qi, Linfeng Zhang  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2605.20022v1  

#### Abstract
Speculative decoding accelerates memory-bound LLM inference without quality degradation by using a fast drafter to propose multiple candidate tokens and the target model to verify them in parallel. However, conventional sequential speculative decoding suffers from mutual waiting between drafting and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FlexDraft: Flexible Speculative Decoding via Attention Tuning and Bonus-Guided Calibration**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统的 **Speculative Decoding**（推测解码）在提升大语言模型（LLM）推理速度方面表现出色，但仍存在以下瓶颈：

- **串行执行导致互等（mutual waiting）**：传统方法采用“先 draft 后 verify”的串行流程，drafter 和 target model 需要等待彼此完成，造成计算资源浪费。
- **并行推测解码的不确定性问题**：
  - **Bonus token uncertainty**：未来 draft 的生成无法获知当前验证中产生的 bonus token（修正 token），导致 draft 与 verification 路径不一致。
  - **Acceptance length uncertainty**：在验证完成前，drafter 不知道能接受多少个 draft token，因此必须为所有可能的接受长度准备候选分支，引入大量冗余计算，尤其在大 batch size 下开销显著增长（O(N²)）。

这些问题限制了并行 Speculative Decoding 在高吞吐场景下的有效性。

---

### **提出的新方法与创新思路**
作者提出了 **FlexDraft**，一种**无损（lossless）且灵活适应不同 batch size 的 Speculative Decoding 框架**，其三大核心设计如下：

#### **(1) Attention Tuning（注意力调优）**
- **轻量级 block diffusion drafting**：仅对目标模型最后几层的 **attention projectors** 进行微调，用于预测 mask token，而保持其余参数（尤其是 FFN 层）冻结。
- **优势**：
  - 参数开销极小（约增加 6% 可训练参数）。
  - 完全保留目标模型的 autoregressive 分布，确保无损生成质量。
  - 支持单次 forward 并行生成多个 draft token。

#### **(2) Bonus-guided Calibration（基于 bonus token 的校准）**
- 引入一个轻量级 **2-layer MLP**，以 resolved bonus token 的嵌入和 mask token 的隐藏状态为输入，输出对 draft logits 的校准偏置。
- **作用**：使 draft 分布与已确定的 bonus token 对齐，缓解因 bonus token 不可见导致的 draft-verification 不匹配问题。

#### **(3) Flex Decoding（动态解码策略）**
- **动态切换执行模式**：
  - 小 batch size → 使用 **parallel draft & verify**，利用重叠执行减少等待时间。
  - 大 batch size → 切换到 **sequential draft then verify**，避免冗余分支带来的计算爆炸。
- **自适应剪枝**：
  - 在 parallel 模式下，基于 draft token 的置信度进行 **Selective Verification**，只验证高概率接受长度的前缀，减少冗余 target forward。

---

### **相比现有方法的优势**
| 特性 | FlexDraft | 其他方法（如 BiTA, Apple MTP, DFlash） |
|------|----------|----------------------------|
| 是否 lossless | ✅ 是 | 多数是，但部分需持续预训练可能引入偏差 |
| 参数开销 | 极低（仅调 attention projectors） | 较高（常需 full model adaptation） |
| 支持动态 batch 适配 | ✅ 是（Flex Decoding） | ❌ 否（固定执行模式） |
| 缓解 bonus token 不确定性 | ✅ Bonus-guided Calibration | ❌ 通常忽略 |
| 控制冗余计算 | ✅ Selective Verification + Decoupled Execution | ❌ 接受长度不确定导致 O(N²) 开销 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在 **Qwen3 系列模型** 上进行评估，涵盖多种任务：
- **数学推理**：`GSM8K`, `MATH`
- **代码生成**：`HumanEval`, `MBPP`
- **通用对话能力**：`MT-Bench`

---

### **实验设置与评估指标**
- **硬件平台**：NVIDIA A100 GPU
- **训练数据**：从 `mlabonne/open-perfectblend2` 中采样的 300K 样本
- **评估指标**：
  - **Average Acceptance Length (T)**：每次验证平均接受的 draft token 数量。
  - **Speedup**：相对于标准 autoregressive decoding 的加速比。
  - **Per-step Latency**：单步 draft & verify 的执行时间。

---

### **基线方法对比**
- **Parallel Speculative Decoding**：
  - `BiTA`：双向调优实现并行 draft/verify
  - `Apple MTP`：多位置推测
  - `DART`：扩散启发式推测
- **Strong Baselines**：
  - `EAGLE-3`：基于特征融合的高质量 draft
  - `DFlash`：基于 block diffusion 的高效 draft

> 注：未将 `TiDAR` 作为主基线，因其为 hybrid diffusion-autoregressive 方法，不具备严格 lossless 性质。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1 & 2）**
在 **Qwen3-8B** 上的结果汇总如下：

| Method | GSM8K Speedup | MATH Speedup | HumanEval Speedup | MT-Bench Speedup |
|--------|----------------|---------------|--------------------|------------------|
| BiTA | 1.34× | 1.21× | 1.19× | 1.15× |
| Apple MTP | 2.77× | 2.09× | 2.04× | 1.63× |
| EAGLE-3 | 3.40× | 2.47× | 2.45× | 1.75× |
| DFlash | 3.47× | 2.28× | 2.33× | 1.64× |
| **FlexDraft** | **4.57×** | **3.25×** | **3.04×** | **2.13×** |

> 在扩展训练设置下（Table 2），FlexDraft 在 `GSM8K` 上达到 **5.88×** 加速，在 `MATH` 上达 **5.79×**，全面超越 DFlash。

---

### **与基线方法的对比结果**
- **平均提速**：FlexDraft 在 Qwen3-8B 上实现了 **平均 4.59×** 的加速（无质量损失）。
- **接受长度更长**：T 值普遍高于基线（如在 GSM8K 上达 7.98 vs DFlash 的 6.41），说明 draft 质量更高。
- **跨模型尺度稳定领先**：在 Qwen3-1.7B 到 8B 所有规模上均优于 EAGLE-3 和 DFlash。

---

### **消融实验结果（Ablation Study）**
#### **组件有效性分析（Figure 7）**
- **仅 Attn Tuning**：基础 draft 能力，speedup ~3.0×
- **+ Bonus-guided Calibration**：提升至 ~3.3×，证明校准机制有效对齐 draft 与 bonus token
- **+ Selective Verification**：进一步提升至 ~3.5×，减少冗余 target forward

#### **其他关键发现**
- **层数选择**（Figure 6）：使用 **10 层** draft depth 在速度与开销间取得最佳平衡（13 层虽略快但参数过多）
- **目标知识复用优势**（Table 3）：
  - 相比 full-parameter 微调，Attn Tuning 利用 frozen FFN 继承目标模型知识，T 提升 30–50%，speedup 显著更高

#### **Batch Size 自适应效果（Figure 4）**
- 小 batch（≤2）：parallel 模式占优，最大 speedup >5×
- 大 batch（≥4）：sequential 模式更高效，避免 O(N²) 冗余
- **Flex Decoding 动态切换策略** 成功防止大 batch 下的 speedup 崩溃

---

## **4. 关键结论和发现**

### **主要发现**
1. **并行 Speculative Decoding 的根本瓶颈在于两种不确定性**：
   - Bonus token uncertainty 导致 draft-verification 不一致
   - Acceptance length uncertainty 引发冗余计算爆炸
2. **轻量调优即可激活 block diffusion 能力**：
   - 仅调 attention projectors + 冻结 FFN，既能保留目标分布，又能实现高质量并行 drafting
3. **动态执行策略至关重要**：
   - 单一模式无法兼顾小 batch 与大 batch 场景，Flex Decoding 的 batch-aware 切换是维持高吞吐的关键
4. **Bonus-guided Calibration 显著提升一致性**：
   - 利用已知 bonus token 校准 draft 分布，有效缓解 mismatch 问题

---

### **方法的局限性**
- **依赖 mask token 设计**：需要 careful attention masking 和 position ID 控制，实现复杂度较高。
- **训练数据敏感性**：虽然未使用 DFlash 的专有训练数据，但在完全对等数据下可能仍有优化空间。
- **未支持可变长度生成块**：当前 draft block 固定长度，未来可探索动态 block sizing。

---

### **未来工作方向**
- **更细粒度的 confidence estimation**：结合 token-level 不确定性进行 adaptive draft length 控制。
- **扩展到多模态 LLM**：将 FlexDraft 应用于 vision-language 模型的 speculative generation。
- **硬件感知优化**：结合 GPU kernel 优化，进一步降低 mask token 的 memory overhead。
- **zero-shot domain adaptation**：研究如何在未见领域上直接部署 FlexDraft 而无需重新微调。

---

> ✅ **总结一句话**：  
> **FlexDraft 通过 Attention Tuning + Bonus-guided Calibration + Flex Decoding 三重创新，在不牺牲生成质量的前提下，实现了高达 4.59× 的平均加速，并首次系统性解决了并行 Speculative Decoding 在大 batch 下的性能崩溃问题。**

</details>

---

### 4. [FedADAS: Communication-Efficient Federated Distillation for On-Device Driver Yawn Recognition in Vehicular Networks](https://arxiv.org/abs/2605.19480)

**Authors**: Ahmed Mujtaba, Gleb Radchenko, Marc Masana, Radu Prodan  
**Category**: cs.DC  
**Published**: 2026-05-20  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.19480v1  

#### Abstract
Driver fatigue is a critical safety concern in advanced driver assistance systems. Driver monitoring models trained off-site on static datasets adapt poorly to real-world conditions, while standard federated learning imposes high communication overhead, assumes homogeneous architectures, and struggl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FedADAS: Communication-Efficient Federated Distillation for On-Device Driver Yawn Recognition in Vehicular Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Federated Learning (FL)** 在车载网络中的应用面临三大挑战：
- **设备异构性**（Device Heterogeneity）：不同车辆的边缘设备计算能力差异大，难以统一部署相同模型架构。
- **通信开销高**：标准 FL 需频繁传输完整的模型参数，对带宽受限的车载环境不友好。
- **非独立同分布数据**（Non-IID）：驾驶员行为、摄像头视角、光照条件等导致本地数据高度个性化，影响全局模型收敛。

此外，现有方法大多未验证在真实边缘硬件上的训练可行性，限制了实际部署。

---

### 🚀 提出的新方法与创新
作者提出 **FedADAS** —— 一种基于 **Federated Distillation (FD)** 的新型协作学习框架，其核心创新如下：

#### （1）**完全模型异构支持**
- 不再要求所有客户端使用相同模型结构。
- 每辆车可根据自身硬件资源运行定制化模型（如 ME-Net 或 PE-Net），实现真正的 **full model heterogeneity**。

#### （2）**低通信成本的知识蒸馏机制**
- 客户端仅向服务器上传在共享公共数据集 $D_{pub}$ 上生成的 **soft logits**（而非完整模型参数）。
- 服务器聚合这些 soft logits 形成“软标签”并广播回各客户端用于 **Knowledge Distillation (KD)**。
- 显著降低通信量（最高达 **9974× 减少**）。

#### （3）**端到端的边缘可训练 DMS 流程**
- 设计两个轻量化 yawn 分类架构：
  - **Memory-Efficient Net (ME-Net)**：仅 0.6 MB，适合内存受限设备。
  - **Performance-Efficient Net (PE-Net)**：99.7 MB，推理延迟低至 **1.99ms on Jetson NANO**。
- 支持在 **Jetson AGX Orin 和 Jetson NANO** 上完成训练与推理，无需量化或剪枝。

#### （4）首次在大规模异构车队中验证 FD 框架
- 实验规模高达 **115 个边缘客户端**，是目前最大的 FD 车载系统实证研究之一。

---

### 🔍 相比现有方法的优势

| 特性 | FedADAS | 传统 FL / Hybrid FL-KD 方法 |
|------|--------|----------------------------|
| 模型异构支持 | ✅ 全面支持 | ❌ 多数需统一架构（仅 DB-EPFD 支持部分异构） |
| 通信内容 | soft logits (~0.02 MB) | model parameters (>1–200 MB) |
| 边缘训练支持 | ✅ 在 Jetson 平台完成训练 | ❌ 多数仅测试推理 |
| 数据隐私风险 | 较低（无梯度泄露） | 高（参数/梯度暴露） |
| 可扩展性 | 高（线性于类别数） | 受限于通信负担 |

> 💡 FedADAS 是首个在系统层面实现 **FD + on-device training + large-scale evaluation** 的 vehicular AI 框架。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **YawDD+ [21]**：作者发布的增强版 yawn 检测数据集，提供帧级标注，适用于静态图像分类任务。
- **YawDD [1]**：原始数据集，作为对比基准。

> 所有模型均采用 ImageNet 预训练权重初始化，并进行二分类微调（“yawn” vs “no-yawn”）。

---

### ⚙️ 实验设置

#### （1）硬件平台
- **服务器端**：NVIDIA A100 GPU
- **边缘设备**：
  - NVIDIA Jetson AGX Orin
  - NVIDIA Jetson NANO

#### （2）客户端配置
- 客户端数量 $N \in \{3, 10, 25, 115\}$
- 每个客户端对应一位司机的私有数据（模拟极端 Non-IID 场景）
- 当 $N=115$ 时引入 **视角偏移**（从 dashboard-view 切换为 mirror-view），模拟真实世界中的协变量漂移。

#### （3）公共数据集构建
- 每个客户端贡献约 10% 的本地数据构成共享无标签数据集 $D_{pub}$，用于 KD 过程中的样本对齐。

#### （4）训练流程（遵循 Algorithm 1）
1. **本地训练阶段**：每个客户端在其私有数据上训练 E_local 轮。
2. **Logits 上报**：在 $D_{pub}$ 子集上生成 soft logits 并上传。
3. **服务器聚合**：平均所有客户端的 logits 得到 ensemble soft labels。
4. **知识蒸馏**：客户端下载 soft labels，在 $D_{pub}$ 上执行 KD（温度 $T=1.0$，KL 散度损失）。

#### （5）评估指标
| 指标 | 描述 |
|------|------|
| **Personalization** | 模型在本地数据上的准确率（反映个性化能力） |
| **Generalization** | 模型跨车辆测试集上的准确率（反映泛化能力） |
| **BAM (Balanced Accuracy Metric)** | Personalization 与 Generalization 的几何平均，衡量综合表现 |
| **Communication Cost** | 每轮通信的数据量（MB） |
| **Inference Time** | 单帧推理耗时（ms） |
| **Epoch Training Time** | 每轮训练时间（分钟） |
| **Efficiency Metrics** | 定义复合效率指标：<br> $\eta_{\text{inference}} = \frac{\text{FPS} \times \text{Accuracy}}{\text{Model Size}}$<br> $\eta_{\text{training}} = \frac{\text{Accuracy}}{\text{Epoch Time} \times \text{Model Size}}$ |

---

### 🆚 基线方法对比
- **FedAvg [20]**：经典联邦学习算法，直接聚合模型参数。
- 对比模型包括：DenseNet121, EfficientNet, MobileNetv3, ShuffleNetv2 等主流 CNN 架构。
- 同时比较了其他 FL-KD 混合方法（见 Table 1）。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 指标 | FedADAS (PE-Net) | FedADAS (ME-Net) | FedAvg (PE-Net) |
|------|------------------|------------------|-----------------|
| 最高 Accuracy | **99.39%** | 99.30% | 99.78% (N=3) ↓ 76.35% (N=115) |
| F1-Score | **98.25%** | 97.99% | ≤98.88% |
| 推理延迟 (Jetson NANO) | **1.99 ms** | 3.81 ms | ~14–20 ms |
| 模型大小 | 99.7 MB | **0.6 MB** | 同左 |
| 单 epoch 训练时间 (AGX) | 7.96 min | **6.12 min** | 类似 |
| 每轮通信成本 | **0.02 MB** | 0.02 MB | 1.2 MB (ME-Net) / **199.4 MB** (PE-Net) |

> ✅ **通信成本减少高达 9974×**（FedADAS vs FedAvg with PE-Net）

---

### 📈 与基线方法对比结果（Table 4）

| Client 数量 | 方法 | Personalization | Generalization | BAM |
|-----------|-------|------------------|------------------|-----|
| 3 | FedAvg | 99.78 | 99.78 | 99.78 |
| 3 | FedADAS | 99.50 | 99.15 | 99.33 |
| 115 | FedAvg | 76.35 | 76.35 | 76.35 |
| 115 | FedADAS (PE-Net) | **98.23** | **77.58** | **87.18** |

> 🔺 在 $N=115$ 极端 Non-IID 条件下：
- FedADAS 实现 **21.88% 的 personalization 提升**
- BAM 提升超过 **10 个百分点**

---

### 🔬 消融分析与关键发现

#### （1）模型容量决定 KD 效果
- PE-Net（大模型）始终优于 ME-Net（小模型）在 generalization 表现上（差距达 10–15%）。
- 原因：**学生模型必须有足够的 representational capacity** 来吸收教师模型编码的类间相似性知识（Hinton et al. [10]）。
- ME-Net（0.6 MB）无法有效学习来自高置信度模型的 sharp logits，尤其在 domain shift 下表现下降明显。

#### （2）轻量模型的泛化能力随客户端增加先升后降
- ME-Net generalization：
  - $N=3$: 62.89%
  - $N=10$: 69.06%
  - $N=25$: 83.97% ✅
  - $N=115$: 67.64% ❌
- 结论：当 $D_{pub}$ 引入视角偏差（mirror-view）后，尽管数据量增大，但 **数据质量（representativeness）比数量更重要**。

#### （3）效率指标揭示设计权衡
- **ME-Net** 在 **training efficiency** 上全面领先（Fig. 3），因其采用 depth-wise separable convolutions 和 AdaptiveAvgPool 压缩特征图。
- **PE-Net** 虽然推理最快，但由于全连接层膨胀（50,176 activations），反向传播内存压力大，训练效率较低。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **FedADAS 显著优于传统 FL**：
   - 在高客户端参与度下（尤其是 $N > 10$），FedADAS 在 personalization 和 generalization 之间取得更优平衡。
   - 在 $N=115$ 时 BAM 达 **87.18**，远超 FedAvg 的 76.35。

2. **通信成本极低且可扩展性强**：
   - 每轮仅传输 **0.02 MB soft logits**，相比 FedAvg 减少 **~60–10,000×**。
   - 通信开销仅与类别数线性相关，适合多类感知任务。

3. **边缘训练切实可行**：
   - ME-Net 在 Jetson NANO 上每轮训练仅需 **~10 秒内**，支持实时 on-device collaborative learning。
   - 验证了 **on-device training + FD** 在真实车载场景中的工程可行性。

4. **紧凑模型不一定高效训练**：
   - 尽管某些模型推理快（如 ResNet18），但若结构设计不当（如参数密集连接），仍会导致训练缓慢。
   - **复合效率指标** 更能反映边缘部署的真实性能。

---

### ⚠️ 局限性

1. **依赖共享公共数据集 $D_{pub}$**：
   - 存在潜在隐私泄露风险，特别是当 $D_{pub}$ 来源于客户端子集时。
   - 若 $D_{pub}$ 缺乏代表性（如视角偏移），会损害小模型的泛化能力。

2. **小模型容量瓶颈**：
   - ME-Net 等极轻量模型难以充分吸收 ensemble knowledge，限制了 KD 效益。

3. **服务器信任假设较强**：
   - 当前设定为 semi-honest server，未考虑恶意聚合者攻击。

4. **尚未集成差分隐私或加密机制**：
   - 实际部署中可能需要进一步加强隐私保护。

---

### 🔮 未来工作方向

1. **提升 $D_{pub}$ 的代表性和安全性**：
   - 设计 **server-side auditing 机制** 和 **安全采样协议**，确保数据多样性与隐私合规。

2. **探索无数据蒸馏（data-free KD）**：
   - 使用生成模型合成 proxy data 替代真实 $D_{pub}$，避免数据共享带来的隐私问题。

3. **动态模型选择机制**：
   - 根据设备能力自动推荐最优模型结构（如 ME-Net vs PE-Net）。

4. **引入个性化蒸馏策略**：
   - 允许不同客户端使用不同的温度 $T$ 或加权聚合方式，增强个性化表达。

5. **扩展至更多 DMS 任务**：
   - 如眼动检测、头部姿态估计、注意力预测等 multi-class perception tasks。

---

> 🔗 **开源地址**：https://opensource.silicon-austria.com/mujtabaa/fedadas  
> 📌 **代码已公开，便于复现与社区拓展**。

</details>

---

### 5. [Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption](https://arxiv.org/abs/2605.19593)

**Authors**: Mert Yildiz, Pietro Spadaccino, Alexey Rolich, Francesca Cuomo, Andrea Baiocchi  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.19593v1  

#### Abstract
Modern deployments of Large Language Models (LLMs) increasingly require serving multiple models with diverse architectures, sizes, and specialization on shared, heterogeneous hardware. This setting introduces new challenges for resource allocation, dispatching, and scheduling, particularly under GPU...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文聚焦于**多模型 LLM 推理调度**在资源受限环境下的挑战，特别是当多个异构 LLM 共享有限 GPU 资源时的性能瓶颈。现有系统（如 vLLM）主要针对单一模型优化吞吐量，而对**跨模型的动态资源分配、层卸载（layer offloading）和抢占（preemption）机制**缺乏深入研究。

具体而言，论文探讨了两个关键操作的成本特性：
- **Partial CPU-GPU Offloading**：将部分模型层卸载到 CPU 以缓解 GPU 内存压力。
- **Job-level Preemption**：暂停一个推理任务，腾出 GPU 给更高优先级任务，之后恢复原任务。

这些问题在实际部署中至关重要，尤其是在服务多样化、专业化 LLM 的场景下。

---

### 🚀 提出的新方法与新思路
本论文并非提出一个新的调度器，而是通过**系统的实证研究**揭示了以下关键现象，并为下一代多模型 LLM 调度器设计提供了指导原则：

#### 创新性观察与建模建议：
1. **Offloading 导致非线性吞吐下降**  
   - 小模型对 GPU 层减少极为敏感，即使少量 offloading 也会导致显著性能退化；大模型则表现更平滑。
   - 吞吐不随 GPU 层比例线性变化，传统线性假设失效。

2. **Preemption 开销是“固定成本”而非“可变成本”**  
   - 抢占总开销几乎与中断时机无关（即无论生成 100 还是 5000 token 后被中断），因为主导因素是 **model reload**，而不是 KV cache 迁移。
   - 因此，preemption 成本可以建模为**每个模型-硬件组合的常量**，极大简化调度决策。

3. **KV Cache Transfer 成本极低**  
   - 即使序列长度增长至 5000 tokens，KV cache 大小达近 1GB，其迁移时间仍不足总开销的 1.5%，远低于模型权重重载时间。

4. **硬件差异显著影响 offloading 效果**  
   - 在高性能 GPU（如 RTX A6000）上进行 CPU offloading 的相对惩罚更大，因其 CPU-GPU 性能差距更大。
   - 更平衡的硬件配置（如较弱 GPU + 强 CPU）更适合 hybrid 执行。

这些发现共同构成了一个**面向多模型调度的设计特征集（feature set）**，强调调度器必须具备：
- **Model-awareness**（模型敏感度）
- **Workload-awareness**（输出长度感知）
- **Hardware-awareness**（平台依赖性建模）

---

### 🔍 相比现有方法的优势
| 方面 | 现有工作局限 | 本文优势 |
|------|----------------|----------|
| **Offloading 分析粒度** | 多数仅比较全 GPU / 全 CPU / 固定配置 | 首次系统扫描从 0% 到 100% GPU 层占比的连续性能曲线 |
| **Preemption 成本分解** | 多假设开销存在但未测量细节 | 实测并量化各阶段耗时（unload, reload, KV transfer） |
| **多模型视角** | 多数系统绑定单模型实例 | 明确指出 multi-model 是未来趋势，需新型调度范式 |
| **实用性指导** | 多提框架无实证支撑 | 提供可直接用于调度策略设计的经验法则 |

> 💡 **一句话概括创新点**：  
> 本文首次通过细粒度实验揭示了 LLM 层卸载与抢占行为中的**非线性吞吐衰减规律**和**固定式抢占成本结构**，提出了构建下一代 multi-model LLM 调度器所需的关键输入特征。

---

## 2. 核心实验方法和设置

### 📚 使用的模型（Dataset / Models）
使用多个主流 LLM，在不同精度格式下测试：

| 实验类型 | 模型列表 | 参数范围 | 精度格式 |
|--------|---------|--------|--------|
| **Offloading 实验** | Llama 3 8B, Qwen3-32B, Llama 2 70B | 8B–70B | Q4 量化 |
| **Preemption 实验** | Qwen2.5-3B, Qwen3-8B, Qwen2.5-14B | 3B–14B | FP16 |

> 注：选择不同规模模型以分析 size 对 offloading/preemption 的影响。

---

### ⚙️ 实验设置

#### 硬件平台
两个服务器共享相同 CPU 子系统（AMD Threadripper PRO 5995WX, 64 cores, DDR4），但配备不同 GPU：
- **Server 1**: 2× NVIDIA RTX 5000 Ada Generation (32GB VRAM, PCIe Gen4 x16)
- **Server 2**: 2× NVIDIA RTX A6000 (48GB VRAM, PCIe Gen4 x16)

> 用于对比不同 GPU 架构对 offloading 和 preemption 行为的影响。

#### 软件栈
- Offloading 实验：使用 **Ollama v0.17.7** 控制 layer placement
- Preemption 实验：使用 **HuggingFace Transformers** 直接加载 FP16 模型，手动控制 KV cache 迁移

#### 工作负载设计

| 实验 | 设置详情 |
|-----|---------|
| **Layer Offloading** | 扫描 GPU 层占比：0%, 10%, ..., 90%, 100%，并在接近 100% 时增加精细点（92%, 94%, 96%, 98%）<br>输出长度：50, 150, 300, 500, 1000, 5000 tokens<br>每组配置重复 3 次 |
| **Preemption** | 主任务 Job A：生成 7000 tokens（greedy decoding）<br>在 N ∈ {100, 200, ..., 5000} 处中断，运行 Job B（500 tokens）<br>四组 model pairing：<br>- Qwen2.5-3B → Qwen3-8B<br>- Qwen3-8B → Qwen3-8B<br>- Qwen3-8B → Qwen2.5-3B<br>- Qwen2.5-14B → Qwen3-8B |

> 所有实验均强制释放 GPU memory 并验证清理完成后再继续，确保状态干净。

---

### 📊 评估指标

| 实验 | 主要指标 |
|------|--------|
| **Offloading** | - Decode Throughput (tok/s)<br>- Normalized Throughput（相对于 100% GPU 居住） |
| **Preemption** | - Total Preemption Overhead (s)<br>- 分解步骤耗时：<br> • KV GPU→CPU transfer<br> • Model A unload<br> • Model B load & execute<br> • Model A reload<br> • KV CPU→GPU restore<br>- KV cache size (MB)<br>- PCIe bandwidth utilization |

---

### 🔀 基线方法对比
本文没有直接对比多个调度算法，而是以以下方式建立基准：
- **Fully GPU-resident** 作为 offloading 的性能上限
- **No preemption**（连续执行）作为 preemption 开销的参照
- 不同硬件平台之间的横向比较（RTX 5000 vs A6000）

> 实质上是在构建“经验基线”，用于揭示真实世界的行为模式。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ Layer Offloading 结果
| 发现 | 数据支持 |
|------|--------|
| 吞吐随 prompt length 增加而下降 | KV cache 增大会加剧内存带宽压力 |
| 小模型（Llama3:8B）在接近 100% GPU 时吞吐急剧上升 | 表明对 full GPU residency 极度敏感 |
| 大模型（Llama2:70B）吞吐改善更线性 | 对 partial offloading 更鲁棒 |
| RTX 5000 上 normalized throughput 更高 | 因其 GPU-CPU 性能差较小，offloading 惩罚更低 |

> 图 4 显示：Llama3:8B 在 90% 层放 GPU 时仅达到 ~60% 全 GPU 性能，而 Llama2:70B 可达 ~80%

---

#### ✅ Preemption 实验结果（Table II & Figure 5）

| 模型（Job A） | GPU | 总开销 (avg.) | Model Swap 占比 | KV Transfer 占比 |
|-------------|-----|---------------|------------------|------------------|
| Qwen2.5-3B (5.9GB) | RTX 5000 | 2.98 s | 99.4% | 0.6% |
| Qwen3-8B (15.6GB) | RTX 5000 | 5.16 s | 99.0% | 1.0% |
| Qwen2.5-14B (28.3GB) | RTX 5000 | 7.31 s | 99.1% | 0.9% |
| Qwen2.5-3B | RTX A6000 | 2.62 s | 99.2% | 0.8% |
| Qwen3-8B | RTX A6000 | 4.06 s | 98.5% | 1.5% |
| Qwen2.5-14B | RTX A6000 | 5.73 s | 98.8% | 1.2% |

> 🔹 **Reload 时间占主导**（>98%），且与中断位置无关  
> 🔹 **KV transfer 最多几十毫秒**，即使 cache 达 951MB（Qwen2.5-14B @5000 tokens）

#### 🔁 Preemption Overhead vs Baseline Completion Time
| 模型 | RTX 5000 | RTX A6000 |
|------|----------|----------|
| Qwen2.5-3B | +2.04% | +1.72% |
| Qwen3-8B | +2.11% | +1.86% |
| Qwen2.5-14B | +1.75% | +1.61% |

> 单次抢占带来的延迟增加不足 3%，对长任务影响微乎其微。

---

### 🔍 消融实验与额外发现
虽然未明确称为“消融实验”，但以下分析具有同等价值：

| 分析维度 | 发现 |
|--------|------|
| **中断时机（preemption point）** | 开销基本恒定（flat curve in Fig.5），说明无需根据进度调整策略 |
| **抢占者模型大小（Job B footprint）** | 几乎不影响 Job A 的恢复开销 → 成本只取决于被抢占模型自身 |
| **KV cache transfer growth rate** | 线性增长，速率由 model dimension 决定：<br>- 0.035 MB/token (3B)<br>- 0.14 MB/token (8B)<br>- 0.19 MB/token (14B) |
| **PCIe 实际带宽** | GPU→CPU: ~10–12 GB/s<br>CPU→GPU: ~13–16 GB/s<br>远低于理论峰值 31.5 GB/s（PCIe Gen4 x16），因 PyTorch per-tensor copy 开销高 |

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **Decode throughput 与 GPU 层占比呈强非线性关系**  
   - 小模型对 offloading 极其敏感，轻微卸载即造成大幅性能下降。
   - 调度器不能采用统一比例分配策略，需考虑模型特异性。

2. **Preemption 开销主要是 model reload，且为固定值**  
   - 可建模为 `cost(model_A, hardware)` 的常量函数。
   - 与 KV cache 大小、中断时机无关 → **简化调度逻辑**。

3. **KV cache migration 成本极低（<1.5%）**  
   - 即使在长上下文场景下也不构成瓶颈。
   - 当前优化重点应放在模型加载速度而非 KV 传输效率。

4. **硬件平台显著影响 offloading 效益**  
   - 高端 GPU 上 offloading 惩罚更大 → 更适合全 GPU 或轻度卸载。
   - 调度器必须感知底层硬件能力差异。

5. **Sequence length 放大 offloading 负面效应**  
   - 长输出任务中，decode 阶段反复经历 CPU-GPU 切换，累积延迟显著。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **单请求实验设定** | 所有测试均为 single active request，未涉及 batched inference 或 continuous batching 场景 |
| **未测试频繁 preemption** | 仅分析一次抢占，未研究高频切换下的累积开销 |
| **基于 PCIe Gen4** | 结果可能不适用于 NVLink、CXL 或未来高速互连架构 |
| **未涵盖 MoE 架构** | 如 Mixtral、DeepSeek-MoE 等稀疏激活模型未纳入测试 |
| **忽略冷启动成本** | 实验前已预加载模型文件，未计入磁盘读取时间 |

---

### 🔮 未来工作方向

1. **扩展至 continuous batching 场景**  
   研究在 vLLM 类系统中，multi-model + offloading + preemption 的综合影响。

2. **建模重复 preemption 下的 aggregate cost**  
   探索不同调度策略（如 SJF、priority-based）在多次抢占下的总体收益/损失。

3. **开发基于本文发现的 prototype scheduler**  
   设计一个集成 model-specific offloading curve 和 preemption cost table 的调度器原型。

4. **探索 CXL/NVMe-offloaded weights 的影响**  
   若模型权重存储在 CXL 内存或快速 SSD 上，reload 时间有望降低一个数量级，届时 KV transfer 可能成为瓶颈。

5. **引入 workload trace 驱动仿真**  
   使用真实请求到达模式评估 preemption 是否真正提升系统整体 Goodput。

---

> 📌 **最终启示**：  
> 未来的 LLM serving 系统不能再局限于“单模型最大化吞吐”的思维。面对日益增长的 multi-model、multi-task、resource-constrained 部署需求，必须转向**精细化、感知化、动态化的调度架构**——而这需要建立在对 offloading 与 preemption 的深刻实证理解之上。本文正是这一方向的重要奠基之作。

</details>

---

### 6. [CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs](https://arxiv.org/abs/2605.19269)

**Authors**: Han Guo, Jack Zhang, Arjun Menon, Driss Guessous, Vijay Thakkar, Yoon Kim, Tri Dao  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.19269v1  

#### Abstract
Transformer training systems are built around dense linear algebra, yet a nontrivial fraction of end-to-end time is spent on surrounding memory-bound operators. Normalization, activations, residual updates, reductions, and related computations repeatedly move large intermediate tensors through globa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代 **Transformer** 训练系统虽然在 **GEMM（矩阵乘法）** 上高度优化，但大量时间仍消耗在围绕 GEMM 的**内存密集型操作**上，如：
- **Normalization**（如 RMSNorm）
- **Activation 函数**（如 SwiGLU）
- **Residual 更新**
- **Reduction 操作**（如 log-sum-exp）

这些操作反复将中间张量写入全局内存，造成显著的 **data movement bottleneck**，尤其在 FP8/FP4 等低精度训练加速背景下，该瓶颈愈发严重。

---

### 🚀 提出的新方法：CODA
作者提出 **CODA** —— 一种基于 **GEMM-plus-epilogue** 范式的 GPU kernel 抽象，其核心思想是：

> 将原本独立执行的内存密集型操作，**重参数化为 GEMM 输出仍在片上时即可完成的 epilogue 操作**，从而避免中间张量落地到全局内存。

#### 主要创新点：
1. **GEMM-Epilogue 编程范式重构**  
   固定高性能 GEMM 主循环（mainloop），仅开放一个受限但可组合的 **epilogue 接口**，用于融合以下操作：
   - 元素级/成对变换（Elementwise & Pairwise Maps）
   - 向量加载/广播（Vector Loads）
   - 张量块加载/存储（Tile Loads/Stores）
   - 分块归约（Tile-wise Reductions）
   - 状态保持变换（Stateful Transforms）

2. **代数重参数化（Algebraic Reparameterization）**  
   通过数学等价变换，将原本跨模块的操作（如 `GEMM → Residual → RMSNorm → GEMM`）合并为单个 GEMM 的 epilogue，打破传统模块边界。

3. **支持 LLM 辅助编程**  
   提供高层抽象，使人类或 **LLM** 可以组合预定义的 epilogue primitives 来生成高效 CUDA 内核，无需从零编写底层代码。

---

### 🔍 相比现有方法的优势

| 对比维度 | 传统框架（PyTorch） | 高性能定制内核（Liger, FlashInfer） | **CODA** |
|--------|------------------|-------------------------------|---------|
| **开发效率** | 高（自动微分） | 低（需手写 CUDA） | 中高（可由 LLM 生成） |
| **性能** | 低（频繁内存读写） | 高 | **接近最优** |
| **通用性** | 高 | 低（针对特定模型） | 中（覆盖主流 Transformer 结构） |
| **融合能力** | 弱（算子边界即内存边界） | 强（但需手动设计） | **强且结构化** |

> ✅ **优势总结**：在不牺牲 GEMM 性能的前提下，**系统性地融合了几乎所有非 attention 的 memory-bound 操作**，实现了“**框架级易用性 + 硬件级效率**”的平衡。

---

## 2. 核心实验方法和设置

### 📊 实验设置

- **硬件平台**：单块 **H100 GPU**
- **软件环境**：
  - PyTorch 2.10.0
  - CuTeDSL 4.4.2
  - Liger Kernels 0.8.0
  - FlashInfer 0.6.10.post1
  - QuACK 0.4.1
- **数据规模**：模拟典型 LLaMA 架构，隐藏维度 $ d \in \{2048, 4096, 8192\} $，分别对应 ~1B, 7B, 70B 模型
- **序列长度**：批量处理 16K tokens

---

### 🎯 评估指标

| 指标 | 描述 |
|------|------|
| **Speedup** | 相对于 `cuBLAS + torch.compile` 的端到端运行时间加速比 |
| **Throughput** | 单位时间内处理的 token 数或 GFLOPS |
| **Relative Error** | 与 FP32 精度参考值相比的数值误差（验证重参数化不影响收敛） |
| **Kernel-level vs Block-level** | 单个算子 vs 完整 Transformer 子层的性能对比 |

---

### 🆚 基线方法对比

| 基线 | 说明 |
|------|------|
| `cuBLAS + PyTorch` | 标准实现，未融合 |
| `cuBLAS + torch.compile` | 经过图优化后的 baseline |
| **Liger Kernels** | 高度优化的 LLM 专用 Triton 内核 |
| **FlashInfer** | 面向推理的高效 attention 引擎（部分组件用于对比） |
| **QuACK / Raw GEMM** | 仅执行 GEMM 的理论上限（无 epilogue 开销） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Figures 8, 10, 11）

#### 🔹 Kernel-level 加速（图 8）
| 操作 | CODA (LLM) 相对 `cuBLAS+torch.compile` 加速比 |
|------|---------------------------------------------|
| GEMM + RoPE | **~1.15x** |
| GEMM + SwiGLU | **~1.1x** |
| GEMM + Cross-Entropy | **~1.05–1.1x** |

> ✅ 所有基本 epilogue 操作均实现 **10%-15% 性能提升**

---

#### 🔹 Reparameterized Kernel 加速（图 10）
| 操作 | 加速比（vs `cuBLAS+torch.compile`） |
|------|-------------------------------|
| GEMM-Residual-RMS-GEMM | **~1.15x** |
| GEMM-RMS-SwiGLU | **~1.3x** |
| GEMM-RMS-RoPE | **~1.6x** |
| GEMM-RMS-CrossEntropy | **~1.2x** |

> ⚠️ 注意：这些是**重参数化后的新结构**，传统库无法直接实现，因此对比更具挑战性。

> ✅ **CODA (LLM)** 和 **CODA (Human)** 性能接近，表明 LLM 可有效参与高性能内核开发。

---

#### 🔹 Block-level 端到端加速（图 11）
完整 Transformer 层（含前向+反向传播）性能对比：

| 场景 | CODA (LLM) 加速比 |
|------|------------------|
| Layer Forward | **~1.1–1.2x** |
| Layer Backward | **~1.4–1.8x** |
| Full Layer (F+B) | **~1.1–1.3x** |

> ✅ 在反向传播中收益更高，因 RMSNorm backward 的 reduction 被有效融合。

---

#### 🔹 数值精度（图 6）
- **相对误差**（相对于标准 PyTorch 流程）：
  - CODA 比 QuACK 更精确
  - 重参数化未引入显著数值偏差
- 支持在 **BF16** 下稳定训练

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **GEMM epilogue 是融合 memory-bound 操作的理想场所**  
   利用输出 tile 仍在片上的时机，可高效执行 normalization、activation、residual 等操作，**避免多次全局内存访问**。

2. **代数重参数化可打破模块边界**  
   如 `GEMM → Residual → RMSNorm → GEMM` 可被整体视为一个 GEMM 序列的 epilogue，实现跨层融合。

3. **反向传播同样可保持 GEMM-epilogue 结构**  
   定理 1 证明：前向 pass 中的 tile-local epilogue，其 backward 也可表示为 GEMM + epilogue 形式，仅方向相反。

4. **LLM 可有效参与高性能内核开发**  
   CODA 提供的抽象足够高层，使得 LLM（如 Claude Code）可在少量示例下生成接近人工编写的高性能 CUDA 内核。

---

### ⚠️ 局限性（Limitations）

1. **架构依赖性强**  
   当前重参数化主要针对 **pre-normalized Transformer**（如 LLaMA），扩展到其他架构（如 post-norm, encoder-decoder）需进一步研究。

2. **分布式训练未覆盖**  
   目前聚焦于 **single-GPU kernel**，尚未解决 tensor parallelism 或 pipeline parallelism 下的融合问题。

3. **语义模糊性**  
   过度融合可能**掩盖模块边界和算法语义**，不利于调试和与高层框架（如 PyTorch）集成。

4. **输入预处理开销**  
   如 RoPE 表需提前广播，增加输入流量（见脚注 4），虽提升计算效率，但带来额外通信成本。

---

### 🔮 未来工作方向

1. **扩展至更多模型结构**  
   支持 Vision Transformer、MoE、Diffusion Models 等。

2. **支持分布式融合**  
   在 tensor parallel 场景下，探索跨设备的 epilogue fusion 策略。

3. **自动化 reparameterization 工具链**  
   构建从 PyTorch 模型到 CODA kernel 的自动转换 pipeline。

4. **与编译器栈集成**  
   将 CODA primitives 集成进 TorchInductor、TVM 等，实现自动融合。

---

## 总结

> **CODA 成功将 Transformer 中大量 memory-bound 操作“塞进”GEMM 的 epilogue 中，在几乎不损失数值精度的前提下，实现了 1.1x–1.8x 的性能提升，且支持由 LLM 自动生成高效内核，为“高效 + 易用”的深度学习系统提供了新范式。**

🎯 **一句话总结**：  
**CODA 通过 GEMM-epilogue 重参数化，把“搬砖”的时间省下来“砌墙”，让 Transformer 训练既快又聪明。**

</details>

---

### 7. [Fast Tensorization of Neural Networks via Slice-wise Feature Distillation](https://arxiv.org/abs/2605.19842)

**Authors**: Safa Hamreras, Sukhbinder Singh, Rom\'an Or\'us  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.19842v1  

#### Abstract
We propose a scalable tensorization framework for neural network compression based on slice-wise feature distillation. Unlike conventional tensor decomposition methods that rely on costly global finetuning, our approach decomposes the network into slices consisting of either individual layers or blo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Fast Tensorization of Neural Networks via Slice-wise Feature Distillation**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

传统的 **Tensorization**（张量化）方法虽然能有效压缩神经网络参数，但通常依赖于全局的端到端微调（end-to-end fine-tuning）来恢复模型性能。这种方法存在以下问题：

- **计算成本高**：需要在整个模型上进行反向传播，内存开销大。
- **优化效率低**：对大规模模型（如 LLMs）难以扩展。
- **权重重建误差 ≠ 功能保持**：仅最小化权重矩阵的重构误差（如通过 SVD）不能保证中间特征或最终输出的功能一致性。

### 🚀 提出的新方法：Slice-wise Feature Distillation（切片式特征蒸馏）

作者提出一种**模块化的张量化框架**，其核心思想是：

- 将预训练模型划分为多个独立的 **slice**（可以是单层、MLP block 或连续几层）；
- 对每个 slice 进行独立的 **Tensor Decomposition**（如 Tucker 或 MPO 分解）；
- 不进行全局微调，而是采用 **feature distillation** 的方式，在每个 slice 上局部优化，使其输出激活（output features）逼近原始模型对应 slice 的输出；
- 使用 **MSE 损失函数**监督中间表示匹配。

> 公式定义：  
> $$
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N} \| S^{\text{tensor}}_i(X) - S^{\text{pre}}_i(X) \|^2
$$
其中 $ S^{\text{pre}} $ 是原模型 slice 输出，$ S^{\text{tensor}} $ 是张量化后 slice 的输出。

### 🔍 相比现有方法的优势

| 优势 | 说明 |
|------|------|
| **更强的性能恢复能力** | 利用中间层特征作为监督信号，比仅靠任务级损失（如分类准确率）更丰富、更稳定。 |
| **更高的优化效率与可扩展性** | 各 slice 可并行异步优化，无需同步梯度，适合分布式训练。 |
| **更低的数据需求** | 即使使用少量样本（如 10k），也能实现接近全数据训练的效果，具备良好数据效率。 |
| **自然支持混合策略** | 可先做局部蒸馏初始化，再接全局微调，提升高压缩率下的表现。 |

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集

| 模型 | 数据集 | 用途 |
|------|--------|------|
| ResNet-34 | **CIFAR-10**, **CIFAR-100** | 图像分类任务，评估 CNN 场景下的压缩效果 |
| GPT-2 XL | **OpenWebText**（子集） | 大语言模型场景，测试方法在 LLM 上的可扩展性 |

### ⚙️ 实验设置

#### **ResNet-34 实验**
- **压缩方式**：对所有 3×3 卷积层应用 **Tucker decomposition**，排除 4 个敏感层（见附录图6）。
- **压缩率（Compression Rate, CR）**：
  - Moderate: `CR = 0.5`
  - Aggressive: `CR = 0.7`
- **优化器**：
  - Local tensorization: Adam (`lr=0.001`, `bs=8`)
  - Global tensorization: Adam (`lr=0.0005`, `bs=16` 或 `64`)
- **评估指标**：
  - Top-1 / Top-5 Accuracy
  - Optimization time (分钟)
  - 数据效率（使用 10k/30k/50k 子集）

#### **GPT-2 XL 实验**
- **压缩目标**：只对 **MLP blocks** 进行 MPO 张量化（因其占大部分参数）。
- **压缩率**：整体 `CR = 0.3`，每层均匀压缩 `0.48`。
- **MPO 设置**：两站点分解（two-site factorization），平衡输入输出维度。
- **训练配置**：
  - Batch size: 8
  - Sequence length: 1024
  - Epochs: 1
  - Learning rate: 5e-5
- **评估指标**：
  - Perplexity（WikiText, C4）
  - Accuracy（LAMBADA）
  - Optimization time（单卡 vs 多卡并行）

### 🆚 基线方法对比

| 方法 | 类型 | 描述 |
|------|------|------|
| **Global tensorization** | 基线 | 标准流程：张量化后全局端到端微调，使用任务监督（如交叉熵） |
| **Local tensorization**（本文） | 提出方法 | 张量化后按 slice 独立蒸馏中间特征，无全局依赖 |
| **Hybrid (Local + Global)** | 改进策略 | 先局部蒸馏 5 轮，再全局微调，用于高压缩率场景 |

此外还与多种压缩方法横向比较（表5）：
- **Pruning-based**: APSSF, Edropout
- **Structured pruning**: RL-based pruning
- **Low-rank methods**: NC-CTD, LJSVD

---

## 3. **主要实验结果和性能指标**

### ✅ ResNet-34 on CIFAR-10 & CIFAR-100

#### **CIFAR-10 @ CR=0.5**
| 方法 | Top-1 Acc (%) | 相对原模型下降 | 优化时间 (min) |
|------|----------------|------------------|----------------|
| 原始模型 | 95.04 | — | — |
| Local (50k) | **94.70** | ↓0.34 | **51.38** |
| Global (50k) | 89.47 | ↓5.57 | 120.88 |

> - 局部蒸馏精度高出 **+5.23%**
> - 优化速度快 **2.35×**

#### **CIFAR-10 @ CR=0.7**
| 方法 | Top-1 Acc (%) | 优化时间 (min) |
|------|----------------|----------------|
| Local | **89.19** | **61.11** |
| Global | 88.46 | 103.81 |

> - 性能仍略优，速度提升 **1.7×**

#### **数据效率测试（CIFAR-10）**
| 训练样本数 | Top-1 Acc (%) |
|------------|----------------|
| 50k        | 94.70          |
| 30k        | 94.69          |
| 10k        | 94.61          |

> 表明该方法对训练数据不敏感，极强的数据效率。

#### **CIFAR-100 @ CR=0.5**
| 方法 | Top-1 Acc (%) |
|------|----------------|
| 原始模型 | 79.79 |
| Local (50k) | **78.81** |
| Global (50k) | 68.19 |

> - 准确率恢复达 **98.8%**，远超全局微调（+10.62%）

#### **CIFAR-100 @ CR=0.7：引入 Hybrid 策略**
| 方法 | Top-1 Acc (%) | 优化时间 (min) |
|------|----------------|----------------|
| Global only | 65.12 | 144.08 |
| Local only | 61.12 | 124.12 |
| **Local → Global (5 epochs)** | **74.22** | **106.01** |

> - Hybrid 策略显著优于纯全局微调（↑9.1%），且耗时更少。

---

### ✅ GPT-2 XL 实验结果

#### **性能对比（Table 6）**

| Benchmark | Dense Model | Local Tensorized | Global Tensorized |
|-----------|-------------|------------------|--------------------|
| LAMBADA (acc) | 51.21% | **42.38%** | 35.51% |
| LAMBADA (ppl) | 10.63 | **25.16** | 35.59 |
| WikiText (ppl) | 20.38 | 45.51 | **40.34** |
| C4 (ppl) | 50.03 | 121.12 | **100.70** |

> - 在 LAMBADA 上，local 表现更好；
> - 在 WikiText 和 C4 上，global 更优；
> - 整体表明：**local 在某些任务中更具竞争力**。

#### **优化时间对比（Table 7）**

| 方法 | 单卡时间 (min) | 并行（48 GPU）时间 (min) |
|------|----------------|----------------------------|
| Local tensorization | 531.35 | **13.4** |
| Global tensorization | **110.25** | — |

> - 单设备下 global 更快（因 GPU 利用率更高）；
> - 但在分布式环境下，local 可实现 **近 40× 加速**！

#### **图示分析（Figure 3）**
- 并行 local tensorization 在早期即达到 competitive performance；
- 随着 GPU 数量增加（2→48），性能随时间快速上升；
- 显示出极佳的 **scalability** 和 **parallelism** 特性。

---

### 🔍 与其他压缩方法对比（Table 5）

| 方法 | CIFAR-10 ΔAcc (%) | CIFAR-100 ΔAcc (%) |
|------|--------------------|---------------------|
| Ours (CR=0.5) | **-0.34** | **-0.98** |
| APSSF | +0.02 | -5.17 |
| NC-CTD | +1.77 | +11.97 |
| LJSVD | -1.14 | -1.42 |

> - 虽然部分剪枝方法在准确率保留上略优，但本方法专注于改进 **tensorization 流程本身**；
> - 在 moderate compression 下已接近“近无损”压缩水平。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **Slice-wise feature distillation 显著优于传统全局微调**：
   - 更快收敛、更高精度恢复、更强数据效率。
   
2. **模块化设计带来天然并行优势**：
   - 各 slice 可独立优化，非常适合分布式系统；
   - 在多 GPU 环境下可实现数十倍加速。

3. **适用于不同架构（CNN & LLM）和分解方式（Tucker & MPO）**：
   - 方法通用性强，已在 ResNet-34 和 GPT-2 XL 上验证。

4. **Hybrid 策略有效应对高压缩挑战**：
   - “Local 初始化 + Global 微调”组合可在极端压缩下取得最佳平衡。

5. **Tensorization 可与其他压缩技术结合**：
   - 如 pruning、quantization，未来可用于构建混合压缩 pipeline。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **单设备利用率低** | 小规模 slice 导致 GPU 利用不足，尤其对大模型不利 |
| **依赖中间特征存储** | 需预先缓存激活值，带来一定 I/O 开销（虽可摊销） |
| **未探索动态 slice 划分** | 当前 slice 固定，未根据 layer importance 自适应调整 |
| **Loss 函数较简单** | 当前仅用 MSE，cosine similarity 等可能更优 |

---

### 🔮 未来工作方向

1. **自适应 slice grouping**：
   - 根据层重要性动态合并相邻层为更大 slice，提高硬件利用率。

2. **探索更优蒸馏损失函数**：
   - 如 Cosine Similarity、KL 散度等，增强特征对齐能力。

3. **扩展至更大 Transformer 架构**：
   - 如 Llama、OPT 等百亿级以上模型。

4. **融合其他压缩技术**：
   - 与 pruning、quantization 结合，构建统一的高效压缩框架。

5. **部署级优化**：
   - 探索如何将 MPO/Tucker 层编译为高效推理 kernel。

---

> **一句话总结**：  
> 本文提出的 **slice-wise feature distillation** 为神经网络张量化提供了一种**高效、可扩展、模块化**的新范式，在保持高性能的同时极大提升了优化效率，特别适合在**分布式环境**中压缩大规模模型。

</details>

---

### 8. [EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design](https://arxiv.org/abs/2605.19743)

**Authors**: Gioele Molinari, Florian Felten, Soheyl Massoudi, Mark Fuge  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.19743v1  

#### Abstract
Large Language Model (LLM) agents are increasingly applied to engineering design tasks, yet existing evaluation frameworks do not adequately address multi-agent systems that combine simulation, retrieval, and manufacturing preparation. We introduce a benchmark suite with three evaluation dimensions:...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前的 **Large Language Model (LLM)** 在工程设计任务中展现出潜力，但现有的评估框架主要针对单一任务或简单工具调用场景，**无法有效评估多智能体系统在复杂工程流程中的表现**。具体而言，现有方法缺乏对以下三个关键维度的综合考量：
- **多步骤、多工具协同的工作流执行能力**
- **基于检索的参数选择（Retrieval-Augmented Generation, RAG）的有效性**
- **高性能计算（HPC）训练流水线的端到端编排能力**

这些问题限制了LLM在真实工程场景（如拓扑优化、仿真分析、制造准备）中的可靠部署。

---

### **提出了什么新方法或新思路**
本文提出两个核心贡献：

#### **(1) ENGIAI：一个基于 LangGraph 的 Multi-Agent System (MAS) 参考实现**
- 构建了一个分层的 **supervisor 架构**，由一个中央 supervisor agent 路由请求至七个专业化 agent。
- 各 agent 分别负责不同领域任务：
  - **Engineering Agent**：驱动拓扑优化、仿真与 STL 导出（通过 EngiBench/EngiOpt）
  - **RAG / ArXiv / Search Agents**：提供文档问答、论文检索与网络搜索
  - **HPC / CLI / Prusa Agents**：管理远程集群作业、本地命令执行与 3D 打印机控制
- 支持自然语言交互，并可通过添加新工具 API 扩展功能。

#### **(2) 一套全新的 Benchmark Suite**
该套件涵盖三大评估维度，填补了现有基准的空白：

| 维度 | 内容 |
|------|------|
| **Workflow Benchmark** | 设计七种提示风格（prompt styles），测试不同认知需求下的任务完成率：<br>• `FULL`（直接参数输入）<br>• `NATURAL`（定性描述）<br>• `W-RAND`, `W-DERIVED`, `W-DISTRACT`, `W-COND`, `W-MULTI`（分别测试数值保真度、语义消歧、条件分支、工作记忆等） |
| **RAG Benchmark** | 引入**门控评分机制（gated scoring）**，仅当 agent 显式调用 `search_documents` 工具时才为参数选择打分，从而隔离检索的真实贡献，防止模型依赖先验知识“猜对”答案。 |
| **HPC Benchmark** | 首次评估 LLM agent 是否能自主完成完整的 ML 模型训练流水线：<br>生成 SLURM 脚本 → 提交作业 → 监控进度 → 下载并评估模型 |

---

### **相比现有方法的优势**
| 对比维度 | 现有方法局限 | 本文优势 |
|---------|-------------|--------|
| **覆盖范围** | 多数仅关注单一环节（如拓扑优化或 G-code 生成） | 统一整合设计、仿真、检索、HPC 编排与制造输出 |
| **评估深度** | 工具调用准确率为主，忽略决策逻辑与状态追踪 | 引入条件分支、多导出、派生计算等挑战性任务 |
| **RAG 有效性验证** | 多数未区分检索 vs. 参数化记忆 | 采用门控评分，明确量化 RAG 的增量价值 |
| **HPC 自动化** | 尚无工作评估 LLM 在真实 SLURM 集群上的长期任务编排能力 | 实现从脚本生成到结果提取的全流程自动化 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Beams2D**：二维悬臂梁拓扑优化问题，目标是最小化 compliance，约束体积分数（volfrac）、力作用位置（forcedist）、滤波半径（rmin）。有标准解，适合定量评估。
- **Photonics2D**：二维光子器件逆向设计问题，最大化电磁场重叠，用于跨物理域泛化能力测试。

以上均来自 **EngiBench** 框架提供的统一 Python API。

---

### **实验设置**
#### **LLM 后端（4 种）**
| 类型 | 模型名称 |
|------|--------|
| Proprietary Cloud Models | `GPT-5-mini`, `Gemini-3-Flash` |
| Open-Source 4B-Parameter Models (via Ollama) | `Qwen3-4B`, `Qwen3.5-4B` |

所有调用设置 `temperature=0` 并固定随机种子以提高可复现性。

#### **评估指标**
##### **Workflow 总体得分（Composite Overall Score, CO）**
$$
S_{\text{workflow}} = 0.65 \cdot S_{\text{design}} + 0.20 \cdot S_{\text{tool}} + 0.15 \cdot S_{\text{completion}}
$$
其中：
- $S_{\text{design}}$：设计质量（65%），加权平均 IoU、像素精度、目标匹配、约束满足、连通性、水密性
- $S_{\text{tool}}$：工具调用效率（20%），正确调用次数 / 最优或实际调用总数
- $S_{\text{completion}}$：任务完成率（15%），是否成功调用所有必需工具且参数符合要求

##### **RAG 评分机制（Gated Scoring）**
- 仅当 agent 调用了 `search_documents` 工具后，其参数准确性才计入得分
- 若未调用或索引为空（Empty RAG），即使参数正确也不给分

##### **HPC 流水线评分**
$$
S_{\text{HPC}} = 0.70 \cdot S_{\text{step}} + 0.15 \cdot S_{\text{config}} + 0.15 \cdot S_{\text{eval}}
$$
- $S_{\text{step}}$：四个阶段（生成、提交、监控、评估）的完成比例
- $S_{\text{config}}$：配置是否正确
- $S_{\text{eval}}$：能否成功提取评价指标（MMD, DPP, RVC, IOG/COG/FOG）

---

### **基线方法对比**
本文未直接对比其他完整系统，而是将自身结果与已有研究进行横向比较：
- **DSG-MAS**, **MechAgents**, **FEAGPT** 等多智能体系统：功能不全，缺少 RAG 或 HPC 支持
- **FDM-Bench**, **ToolSandbox**, **AgentBench** 等基准：未涉及多 agent 协同、长周期 HPC 编排或制造导出

因此，ENGIAI 是首个同时覆盖这六大能力维度的系统（见 Table 1）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **Workflow Performance on Beams2D（Table 5）**
| Prompt Style | GPT-5-mini (TC) | Gemini-3-Flash (TC) | Qwen3-4B (TC) | Qwen3.5-4B (TC) |
|------------|------------------|----------------------|---------------|------------------|
| FULL       | 1.00             | 0.93                 | 0.00          | 0.73             |
| NATURAL    | 0.87             | 1.00                 | 0.00          | 0.33             |
| W-RAND     | 1.00             | 1.00                 | 1.00          | 1.00             |
| W-DERIVED  | 1.00             | 1.00                 | 0.47          | 0.85             |
| W-DISTRACT | 1.00             | 1.00                 | 1.00          | 0.93             |
| **W-COND** | **0.93**         | **0.87**             | **0.40**      | **0.60**         |
| W-MULTI    | 0.93             | 1.00                 | 1.00          | 1.00             |
| **Average TC** | **0.96**     | **0.97**             | **0.55**      | **0.78**         |

> ✅ **结论**：专有模型接近完美；开源模型存在显著差距，但 Qwen3.5-4B 相比前代明显提升。

#### **Photonics2D 泛化能力（Table 6）**
| Prompt Style | Best Model (TC) |
|------------|------------------|
| W-RAND     | 1.00             |
| W-DISTRACT | 1.00             |
| **W-COND** | **0.53 (Gemini-3-Flash)** |

> ⚠️ **发现**：即使是最佳模型，在 Photonics2D 上的条件推理成功率也大幅下降（Beams2D 为 87%，此处仅 53%），说明**领域熟悉度影响巨大**。

---

#### **RAG Evaluation（Figure 6）**
- **RAG-on**：得分接近 **1.0**（几乎所有参数都正确获取）
- **RAG-off / Empty RAG**：得分趋近于 **0**，除非是常见默认值（如 P0 中 volfrac=0.35 可能被记住）
- **验证了门控评分的有效性**：只有通过检索才能获得高分，证明 RAG 对工程参数选择至关重要。

---

#### **HPC Orchestration（Figure 7）**
| Prompt Style | Gemini-3-Flash | GPT-5-mini |
|------------|------------------|-------------|
| EXPLICIT   | 100% 完成        | 70% 完成（最终评估失败） |
| NATURAL    | 100% 完成        | 50% 完成（更早中断） |

> 🔍 **根因分析**：GPT-5-mini 在长流程中出现**多步指令退化（multi-step instruction degradation）**，常遗漏最后一步 `evaluate_model` 调用，而 Gemini 更鲁棒。

---

### **消融实验结果**
- **Empty RAG 实验**：证实 agent 准确性依赖检索内容而非工具存在本身
- **JSON Parsing Error 分析**：少量 Qwen3.5-4B 运行失败源于输出格式错误，非能力缺陷
- **Tool-Calling Heatmap（Figure 3）**：显示 Qwen3-4B 存在冗余调用（如重复 `optimize_design`），降低效率得分

---

## **4. 关键结论和发现**

### **主要发现**
1. **RQ1 (Workflow Performance)**  
   - 专有模型（GPT-5-mini, Gemini-3-Flash）在 Beams2D 上平均任务完成率达 **96–97%**，表现出色。
   - 开源 4B 模型仍有较大差距（Qwen3-4B: 55%, Qwen3.5-4B: 78%），但**代际改进显著**。
   - **条件分支（Conditional Branching）是最难的认知任务**，尤其在陌生领域（Photonics2D）下失败率高达 47–80%。

2. **RQ2 (Model Robustness)**  
   - 两大专有模型性能相当，稳定性高。
   - 开源模型虽有进步，但在复杂 prompt 下仍易失败。

3. **RQ3 (Tool Usage Efficiency)**  
   - 多余工具调用会显著拉低总体得分（Figure 5），尽管不影响设计质量。
   - Qwen3.5-4B 在多个任务上实现最优调用次数，表明其具备更好的行为控制能力。

4. **RQ4 (RAG Improvement)**  
   - RAG-on 得分接近 1.0，RAG-off 接近 0，**验证了检索对参数选择的关键作用**。
   - 门控评分机制成功隔离了先验知识的影响。

5. **RQ5 (HPC Orchestration)**  
   - Gemini-3-Flash 可 **100% 成功编排 HPC 训练流水线**。
   - GPT-5-mini 在自然语言提示下失败率高达 50%，暴露了**长周期任务中的指令遵循退化问题**。

---

### **方法的局限性**
1. **问题覆盖有限**：目前仅测试 Beams2D 和 Photonics2D，尚未扩展至更多 EngiBench 问题。
2. **模型数量受限**：仅评测 4 个 LLM，且未包含更大规模开源模型（如 70B）。
3. **HPC 实验成本高**：仅限两个云模型参与，开源模型因推理慢未纳入。
4. **缺乏人类干预实验**：未引入工程师实时反馈与纠正机制。
5. **未进行统计显著性检验**：结果基于均值与标准差，缺乏假设检验支持。

---

### **未来工作方向**
1. **扩大评估范围**  
   - 增加更多物理问题、更大模型家族（Llama, Mistral）、更高参数量模型
   - 开展敏感性分析（温度、prompt 表述、工具描述长度）

2. **优化工具使用效率**  
   - 引入 few-shot 示例或约束解码减少冗余调用
   - 设计工具调用奖励机制用于微调小型模型

3. **增强条件推理能力**  
   - 引入结构化 Chain-of-Thought，强制显式写出判断依据
   - 加强对模拟结果的理解与阈值比较能力

4. **提升 RAG 鲁棒性**  
   - 在更大、噪声更多的文献集合中测试检索精度
   - 引入对抗性设置（多个冲突来源）测试证据融合能力

5. **深化 HPC 自主性**  
   - 从“编排预定义脚本”升级为“自动生成并迭代修改代码”
   - 引入显式状态跟踪机制（checkpoint）防止长流程遗忘

6. **探索工具生态系统扩展**  
   - 结合 MCP 协议接入更多工程工具
   - 研究 agent 性能在工具数量增长时的可扩展性

--- 

> 📌 **总结一句话**：  
> **ENGIAI 是首个将多智能体架构、RAG 增强决策与 HPC 全流程编排集成于一体的 LLM 工程设计框架，并配套提出了一套全面、可量化的三维评估体系，揭示了当前 LLM 在复杂工程任务中的能力边界与关键瓶颈。**

</details>

---

### 9. [Projecting Latent RL Actions: Towards Generalizable and Scalable Graph Combinatorial Optimization](https://arxiv.org/abs/2605.19721)

**Authors**: Franco Terranova (UL, LORIA, Inria), Guillermo Bernardez (UC Santa Barbara), Albert Cabellos-Aparicio (UPC), Nina Miolane (UC Santa Barbara), Abdelkader Lahmadi (LORIA, UL, Inria)  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.19721v1  

#### Abstract
Graph combinatorial optimization (GCO) has attracted growing interest, as many NP-hard problems naturally admit graph formulations, yet their combinatorial explosion renders exact methods computationally intractable. Recent advances in Reinforcement Learning (RL) combined with Graph Neural Networks ...

---

### 10. [D$^3$-Subsidy: Online and Sequential Driver Subsidy Decision-Making for Large-Scale Ride-Hailing Market](https://arxiv.org/abs/2605.20036)

**Authors**: Taijie Chen, Rui Su, Siyuan Feng, Laoming Zhang, Hongyang Zhang, Haijiao Wang, Zhaofeng Ma, Jintao Ke  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.20036v1  

#### Abstract
Ride-hailing platforms like DiDi Chuxing operate in highly dynamic environments where balancing driver supply and passenger demand is critical. Although driver-side subsidies serve as a primary lever to align these forces and improve key KPIs like completed rides (\texttt{Rides}) and gross merchandi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# D³-Subsidy: Online and Sequential Driver Subsidy Decision-Making for Large-Scale Ride-Hailing Market  
**——核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对大规模网约车平台（如 DiDi）中的 **driver-side subsidy 决策优化**问题展开研究。在动态且非平稳的供需环境中，平台需通过补贴激励司机接单以提升关键业务指标（如 **Rides** 和 **GMV**），同时必须满足以下三大约束：
- **响应性**：快速应对随机冲击；
- **预算约束**：严格遵守全局补贴率上限（subsidy-rate cap）；
- **低延迟执行**：在城市级别实现可扩展的实时决策。

传统逐订单优化（per-order optimization）计算成本过高，而标准 **Reinforcement Learning (RL)** 方法因探索风险高、训练不稳定，难以在生产环境中安全部署。

---

### 提出的新方法：D³-Subsidy
作者提出 **D³-Subsidy (Dynamic Driver-side Diffusion-based Subsidy)** ——一种基于扩散模型（diffusion model）的离线控制框架，用于城市级 driver subsidy 的序列化决策。

#### 核心创新点：
1. ✅ **Prefix-Conditional Diffusion Model**
   - 在训练和推理中均固定已观测的历史状态前缀（prefix），仅对未来的轨迹后缀进行生成与去噪。
   - 有效弥合了“训练-推理”之间的不一致性（train-inference gap），确保在线部署时策略的一致性和可行性。

2. ✅ **Constraint-Aware Score Objective**
   - 引入带惩罚项的 score 函数，在补贴超限时施加平滑乘法惩罚，显式促进预算可行性。
   - 支持灵活权衡 KPI 提升与 cap compliance。

3. ✅ **Context-Conditioned Inverse Dynamics Decoder**
   - 不直接生成 action，而是从生成的轨迹中反向解码出城市级控制信号 $ \lambda_t $。
   - 通过上下文（context）条件控制目标偏好（如 Rides vs. GMV 权衡）、预算紧度等，增强策略的可控性与适应性。

4. ✅ **Two-Stage Training with PEFT**
   - 多城市预训练 + 参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）：
     - 预训练阶段学习通用的城市动力学先验；
     - 微调阶段仅更新 inverse dynamics 模块，保持主干冻结，提升冷启动迁移能力。

5. ✅ **Lagrangian-Dual-Derived Mapping**
   - 将城市级 $ \lambda_t $ 映射为细粒度订单-司机对的补贴 $ b_{ij} $，无需迭代优化，具备解析透明性与低延迟优势。

---

### 相比现有方法的优势
| 维度 | D³-Subsidy | 传统方法（如 RL 或 BC） |
|------|------------|------------------------|
| **安全性** | 完全离线训练，无试错探索风险 | RL 探索可能导致预算突破或运营违规 |
| **部署一致性** | Prefix-conditioning 保证历史不可变 | 自回归模型易偏离真实路径 |
| **可扩展性** | 城市级标量控制 → 细粒度补贴映射 | 逐订单建模计算开销大 |
| **可控性** | 上下文调节实现目标导向调控 | 固定策略难适应不同运营目标 |
| **泛化性** | 多城市预训练 + PEFT 支持跨城迁移 | 单城市训练泛化差 |

---

## 2. 核心实验方法和设置

### 数据集
- **来源**：来自 DiDi 平台的真实广播日志（broadcast logs）
- **覆盖范围**：巴西 133 个城市，共 28 天
- **形式**：城市-天级别的轨迹数据（city-day trajectories）
- **时间窗口粒度**：提供三种聚合粒度版本：
  - 2分钟 / 5分钟 / 10分钟
- **字段包括**：
  - 城市级供需状态 $ s_t $
  - 实际补贴率 $ p_t $
  - 日志动作 $ \lambda_t $
  - KPIs：Rides, GMV, DRV（Driver Revenue）

> ⚠️ 注：测试集保留 3 个完整城市（City A/B/C），每城 7 天，共计 21 条轨迹；其余用于训练。

---

### 实验设置与评估方式

#### 评估方式：**闭环仿真回放（Closed-loop Rollout）**
- 使用 DiDi 高保真生产级 simulator 进行模拟。
- 每个时间步输入当前 state，policy 输出 $ \lambda_t $，simulator 执行并推进系统至下一状态。
- 最终输出整日轨迹 $ \xi $，用于计算各项指标。

#### 主要评估指标：
| 指标 | 定义 | 说明 |
|------|------|------|
| **Score($\xi$)** | 结合 Rides 与 subsidy constraint violation 的综合得分（见公式 21） | 主要评价指标 |
| **Rides** | 完成订单数 | 正向越高越好 |
| **GMV** | 总交易额 | 正向越高越好 |
| **DRV** | 司机收入 | 衡量公平性与激励效果 |
| **UnderGap($\xi$)** | $ \max(0, C - C_{\text{real}}(\xi)) $ | 衡量预算利用率，越小越好 |
| **Violation** | 是否 $ C_{\text{real}} > C + \delta $ | 判断是否违反 cap |

---

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Online** | 生产策略（Predict-then-optimize） | 当前线上运行策略，作为基准 |
| **BC (Behavior Cloning)** | 监督学习 | 学习历史动作分布 |
| **BCQ**, **CQL**, **IQL**, **TD3+BC** | Offline RL | 各类保守离线强化学习算法 |
| **DT (Decision Transformer)** | 序列建模范式 | 基于 Transformer 的轨迹建模 |
| **DD (Decision Diffuser)** | 扩散策略模型 | 基于扩散的动作生成方法 |

> 所有 offline RL 方法均以最大化 `Score` 为目标训练，已隐含软约束处理。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（表1 & 表6）

#### 📊 离线仿真结果（平均 Score 对比）
| 方法 | Overall Average Score |
|------|------------------------|
| **D³-Subsidy (Ours)** | **4845.89** ✅ |
| Online | 4645.44 |
| DT (最强基线) | 4780.81 |
| DD | 4742.29 |

👉 **D³-Subsidy 超出最强基线 DT 达 65 分以上，显著领先所有 baseline。**

---

#### 🔬 在线 A/B 测试结果（真实市场验证）
在某城市开展为期 7 天的线上 A/B 测试（各分配 50% 订单流量）：

| Metrics | Lift vs. Online |
|--------|------------------|
| **Rides** | **+1.59%** ✅ |
| **GMV** | **+2.06%** ✅ |
| **DRV** | **+2.31%** ✅ |
| **Inference Time** | +20ms（可接受） |

✅ **所有 KPI 均显著正向提升，且未发生任何 subsidy-rate violation！**

---

### 消融实验结果（Ablation Study）

在 City C 上进行消融分析（5-min 设置）：

| 模型变体 | Score | 相对下降 |
|---------|-------|----------|
| **D³-Subsidy (Full)** | 1792.48 | — |
| -C（移除 context conditioning） | 1737.58 | ↓3.1% |
| -M（移除 multi-city pretraining） | 1704.91 | ↓4.9% |
| -P（移除 PEFT 微调） | 1715.78 | ↓4.3% |

📌 **结论**：所有组件均有贡献，其中多城市预训练影响最大，证明跨域知识迁移至关重要。

---

### 其他重要实证发现

#### ✅ 统计显著性检验（paired t-test）
- 对比所有 baseline，D³-Subsidy 的提升均具有统计显著性（p < 0.05），尤其对 Online 策略达到 p < 0.001。
- 平均增益达 +225 分以上，置信区间稳定。

#### ✅ 时间维度表现分析（图3-4）
- **KPI 提升是持续累积的**，并非依赖短期爆发；
- **补贴率控制更精准**：D³-Subsidy 更接近 cap 上限运行（under-gap 更小），但始终不越界（no overshoot）；
- 相比之下，Online 策略存在明显 under-utilization。

#### ✅ 控制性分析（图5）
- 通过调整 context 中的目标 KPI 输入（scaling factor γ ∈ [0.2, 2.0]），可连续调节实际产出的 Rides/GMV/Score。
- 表明模型具备良好的 **deployment-time steerability**。

#### ✅ 冷启动迁移能力（Cold-Start Transfer）
| 方法 | Average Score（3 新城市） |
|------|----------------------------|
| **D³-Subsidy** | **656.80** ✅ |
| Online | 650.60 |
| DT | 643.66 |

👉 即使没有目标城市的训练数据，D³-Subsidy 仍能凭借多城市先验实现最优表现。

---

## 4. 关键结论和发现

### 主要结论
1. ✅ **D³-Subsidy 是首个将 diffusion model 成功应用于 ride-hailing 自动化 driver subsidy 控制的工作**，实现了安全、高效、可控的序列决策。
2. ✅ 通过 **prefix-conditioned diffusion + inverse dynamics decoding** 架构，解决了 train-inference gap 与部署一致性难题。
3. ✅ **多城市预训练 + PEFT 微调** 显著提升了跨区域泛化能力和冷启动性能。
4. ✅ 在真实数据与线上 A/B 测试中均取得一致且显著的 KPI 提升（Rides +1.59%，GMV +2.06%），同时严格满足预算约束。

---

### 方法局限性
1. ❗ **依赖高质量历史日志数据**：若日志中缺乏多样性或存在偏差，可能限制生成模型的表现。
2. ❗ **context 设计敏感**：性能依赖于 context 编码的有效性（如 budget regime、target preference 的表达能力）。
3. ❗ **逆动力学模块需精确校准**：decoder 若未能准确捕捉轨迹与 action 的关系，会影响最终控制精度。

---

### 未来工作方向
1. ➕ 探索更高效的 **diffusion sampling 策略**（如蒸馏加速），进一步降低 inference latency。
2. ➕ 引入 **multi-agent diffusion modeling**，考虑司机间的竞争与博弈行为。
3. ➕ 结合 **causal modeling**，识别补贴的真实因果效应，避免伪相关干扰。
4. ➕ 扩展至 **spatio-temporal zone-level subsidy control**，支持更精细化的空间调度。

--- 

> 💡 **一句话总结**：  
> D³-Subsidy 利用 diffusion model 的强大生成能力与 offline learning 的安全性，构建了一个可部署、可控制、可泛化的城市级 driver subsidy 决策系统，在真实场景中实现了 KPI 与预算合规性的双赢。

</details>

---

### 11. [From SGD to Muon: Adaptive Optimization via Schatten-p Norms](https://arxiv.org/abs/2605.19781)

**Authors**: Thomas Massena (IRIT, DTIPG - SNCF, UT3), Corentin Friedrich (IRIT), Mathieu Serrurier (IRIT)  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.19781v1  

#### Abstract
Modern optimizers, like Muon, impose matrix-wise geometry constraints on their updates. These matrix-wise constraints can be unified under Linear Minimization Oracle (LMO) theory. However, all current methods impose fixed LMO geometries for the update rules, chosen by-design or empirically, which ar...

---

### 12. [From Simple to Complex: Curriculum-Guided Physics-Informed Neural Networks via Gaussian Mixture Models](https://arxiv.org/abs/2605.19263)

**Authors**: Jianan Yang, Yiran Wang, Shuai Li, Fujun Cao, Xuefei Yan, Junmin Liu  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.19263v1  

#### Abstract
Physics-informed neural networks (PINNs) offer a mesh-free framework for solving partial differential equations (PDEs), yet training often suffers from gradient pathologies, spectral bias, and poor convergence, especially for problems with strong nonlinearity, sharp gradients, or multiscale features...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Simple to Complex: Curriculum-Guided Physics-Informed Neural Networks via Gaussian Mixture Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
**Physics-informed Neural Networks (PINNs)** 在求解偏微分方程（PDEs）时面临以下挑战：
- **梯度病理（gradient pathologies）** 和 **谱偏差（spectral bias）** 导致训练不稳定；
- 对于具有强非线性、尖锐梯度或多尺度特征的PDE，收敛困难；
- 传统方法采用静态采样和固定损失权重，无法自适应地应对空间上学习难度的差异。

现有 **Curriculum Learning (CL)** 方法多依赖人工预设的固定步长调度，缺乏对局部残差分布动态变化的量化建模。

---

### 🚀 提出的新方法：CGMPINN
作者提出 **Curriculum-Guided Gaussian Mixture PINN (CGMPINN)**，将 **Gaussian Mixture Model (GMM)** 与 **动态课程学习（dynamic curriculum learning）** 结合，构建一个统一框架。

#### 核心创新点：
1. **基于GMM的残差分布建模**  
   - 定期对PDE残差进行GMM拟合，识别出多个具有不同均值和方差的子区域（clusters），从而实现对“学习难度”的空间量化。
   - 每个cluster的学习难度由其加权平均残差大小决定。

2. **双通道课程调度机制（Dual Curriculum）**
   - **难度维度**：通过共享的课程参数 $ T(k) \in [0,1] $ 控制从“易”到“难”区域的渐进聚焦。
   - **可靠性维度**：引入基于精度（inverse variance）的调制机制，在早期抑制高方差（即不确定性大）的cluster，避免优化震荡。

3. **端到端可微且无需修改网络结构**
   - 方法为非侵入式（non-intrusive），不改变原始PDE形式或NN架构，适用于正向与反问题。

4. **支持与自适应损失平衡结合**
   - 可集成 **ReLoBRaLo** 等自适应损失平衡机制，进一步缓解不同loss项间的梯度不平衡。

---

### 🔍 相比现有方法的优势
| 方面 | CGMPINN优势 |
|------|-------------|
| **学习策略** | 动态、数据驱动的课程调度 vs 手动/固定调度 |
| **空间感知能力** | 利用GMM捕捉多模态、异质性的残差分布 |
| **鲁棒性** | 早期避开高不确定性区域，提升训练稳定性 |
| **通用性** | 适用于多种类型PDE（椭圆、抛物、双曲、对流主导、非线性反应扩散等） |

---

## 2. 核心实验方法和设置

### 📚 使用的基准PDE数据集（共6个）
| 类型 | PDE名称 | 特点 |
|------|--------|------|
| 椭圆型 | 1D & 2D Poisson 方程 | 多频振荡 + 尖锐过渡层（如tanh） |
| 抛物型 | Heat Equation（热传导） | 时间演化 + 局部陡峭梯度 |
| 双曲型 | Damped Wave Equation（阻尼波动） | 衰减振荡行为 |
| 对流主导 | Advection-Diffusion Equation | 尖锐行进波前，易出现谱偏差 |
| 非线性 | Fisher-KPP Equation | 非线性行波解，强耦合反应-扩散项 |

所有问题均提供解析解用于生成源项、边界条件和初始条件，并在测试点上计算误差。

---

### ⚙️ 实验设置
- **网络结构**：全连接前馈神经网络（4层×50或80 neurons）；
- **优化器**：两阶段 **Adam → L-BFGS**（先Adam预训练，后L-BFGS精细收敛）；
- **GMM更新频率**：每 `k_upd` 次迭代重新拟合一次GMM；
- **课程参数调度**：$ T(k) = \min\left(1, \frac{k}{K_{\text{max}} \cdot C_{\text{sat}}} \right) $

---

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| `eLoss` | 最终总训练损失（composite loss） |
| `e₂` | 绝对L2误差：$\|u - \hat{u}\|_2$ |
| `Relative e₂` | 相对L2误差：$\frac{\|u - \hat{u}\|_2}{\|u\|_2}$ |
| `e∞` | 最大绝对误差：$\|u - \hat{u}\|_\infty$ |
| `CPU(s)` | 总训练耗时（秒） |

---

### 🆚 基线方法对比
比较了五种代表性PINN变体：
1. **Standard PINN** [Raissi et al., 2019]
2. **lbPINN** [Xiang et al., 2022]：基于ReLoBRaLo的自适应损失平衡
3. **gPINN** [Yu et al., 2022]：引入PDE导数信息的梯度增强型
4. **LNN-PINN** [Tao et al., 2025]：液态残差门控结构
5. **STAR-PINN** [Dodge et al., 2025]：堆叠自适应残差架构

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（以相对L2误差为主）

| 方法 | 1D Poisson | 2D Poisson | Heat Eq | Damped Wave | Adv-Diff | Fisher-KPP |
|------|-----------|-----------|---------|------------|----------|-----------|
| PINN | 8.15e-3 | 1.35e-3 | 1.28e-3 | 2.40e-3 | 1.06e-3 | 2.14e-3 |
| lbPINN | 4.22e+0 ↑ | 5.11e-3 ↑ | 2.83e-3 ↑ | 9.76e-4 ↓ | 6.60e-4 ↓ | 1.48e-3 ↓ |
| gPINN | 1.74e-2 ↑ | 2.01e-3 ↑ | 2.73e-3 ↑ | 1.99e-3 ↑ | 8.44e-4 ↑ | 2.24e-3 ↑ |
| LNN-PINN | 4.15e-4 ↓ | 1.02e-3 ↓ | 8.39e-4 ↓ | 9.79e-4 ↓ | 1.18e-3 ↑ | 6.06e-3 ↑ |
| STAR-PINN | 1.14e-3 ↓ | 6.51e-4 ↓ | 1.43e-3 ↑ | 5.31e-3 ↑ | 6.55e-4 ↓ | 1.43e-3 ↓ |
| **CGMPINN** | **1.81e-4** ✅ | **5.83e-4** ✅ | **3.54e-4** ✅ | **7.07e-4** ✅ | **3.82e-4** ✅ | **9.94e-4** ✅ |

> ✅ 表示在该任务中取得最优结果；↑ 表示表现更差；↓ 表示优于标准PINN但仍劣于CGMPINN

---

### 🔁 与标准PINN相比的性能提升
- **最大相对L2误差降低达 97.8%**（在1D Poisson任务上）
- 在所有6个基准任务中，**CGMPINN均取得最低的e₂和e∞误差**
- 多数情况下训练时间低于或接近其他先进方法（尤其显著优于gPINN和LNN-PINN）

例如：
- **1D Poisson**: e₂ 从 8.83e-3 降至 **1.96e-4**（↓97.8%）
- **Heat Equation**: e₂ 从 1.46e-3 降至 **4.04e-4**（↓72.3%）
- **Fisher-KPP**: e₂ 从 1.65e-3 降至 **7.64e-4**（↓53.7%）

---

### 🔍 消融实验结果（Ablation Study）

比较三种变体：
- **GMMPINN**：仅使用GMM加权，无课程调度
- **CLPINN**：仅使用逐样本残差的课程学习，无GMM聚类
- **CGMPINN**：完整方法

#### 发现：
| 任务 | 关键观察 |
|------|--------|
| **1D Poisson** | GMMPINN崩溃（e₂=2.17），说明静态强调高残差区会导致发散；CLPINN有效但远不如CGMPINN（↓94%） |
| **Heat Eq** | CGMPINN比单独组件提升约4倍精度，表明多尺度动态需联合建模 |
| **Advection-Diffusion** | CLPINN已有明显改进，但加入GMM后进一步↓24%，体现结构化聚类价值 |
| **Fisher-KPP** | CLPINN比GMMPINN↓59%，CGMPINN再↓39%，显示两者互补性强 |

> ✅ **结论**：GMM提供结构性难度识别，课程调度防止过早关注困难区域，二者缺一不可。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **CGMPINN显著提升了PINN在复杂PDE上的求解精度与稳定性**，尤其在存在尖锐梯度、非线性或对流主导的情形下效果突出。
2. **GMM能有效识别残差分布中的多模态结构**，为课程学习提供了可靠的空间难度度量。
3. **双通道课程机制（难度 + 可靠性）同步推进训练进程**，既遵循“从简到繁”，又规避早期不确定性干扰。
4. **理论保障充分**：
   - 权重损失与原PDE损失之间具有**一致等价性（uniform equivalence）**
   - 时间变化总损失满足**次线性收敛（sublinear convergence）**
   - 推导出带显式偏差项的**泛化界（generalization bound）**

---

### ⚠️ 方法的局限性
1. **当前实验限于低维（1D/2D）问题**，高维PDE中GMM的聚类效率和可扩展性尚待验证。
2. **超参数选择仍需一定调优**，如GMM组分数 $ K $、更新频率 $ k_{\text{upd}} $、饱和比例 $ C_{\text{sat}} $。
3. **未提供完全迭代级的收敛分析**，特别是针对Adam→L-BFGS混合优化器的动力学特性。
4. **冻结权重假设下的泛化界不能完全刻画全程自适应重加权过程**。

---

### 🔮 未来工作方向
1. **拓展至高维PDE**，研究GMM在高维空间中的近似能力和降维策略（如流形学习）。
2. **自动化GMM配置**：利用信息准则（如BIC/AIC）自动选择最优 $ K $ 和更新周期。
3. **结合域分解（domain decomposition）与逆问题建模**，处理稀疏噪声数据场景。
4. **发展完整的迭代级收敛理论**，覆盖实际使用的混合优化流程。
5. **探索与其他增强机制融合**，如因果训练（causal training）、贝叶斯推断等。

---

## 📎 总结一句话
> **CGMPINN通过GMM驱动的动态课程学习，实现了从“简单区域”到“复杂区域”的智能训练引导，在多种挑战性PDE上实现了高达97.8%的误差下降，同时具备坚实的理论基础和良好的计算效率。**

</details>

---

### 13. [MMoA: An AI-Agent framework with recurrence for Memoried Mixure-of-Agent](https://arxiv.org/abs/2605.19194)

**Authors**: Rui Chu  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.19194v1  

#### Abstract
The Mixture-of-Agents (MoA) framework has shown promise in improving large language model (LLM) performance by aggregating outputs from multiple agents. However, existing MoA systems often rely on static routers that do not fully capture temporal and contextual dependencies across aggregation layers...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MMoA: An AI-Agent framework with recurrence for Memoried Mixure-of-Agent

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **Mixture-of-Agents (MoA)** 框架在聚合多个 LLM 代理输出时，采用**静态路由机制（static router）**，忽略了不同聚合层之间的**时间依赖性和上下文连续性**。这种设计限制了模型对历史决策信息的利用，导致效率低下且缺乏动态适应能力。

### 🚀 提出的新方法
作者提出 **MMoA（Memoried Mixture-of-Agent）**，一种引入**循环结构（recurrence）** 的新型 MoA 架构，其核心创新在于：
- 引入基于 **LSTM/RNN 的 gating 模块**作为**递归路由器（recurrence router）**
- 路由器不仅考虑当前输入，还结合**前一层的历史隐藏状态**来动态调整各 agent 的贡献权重
- 实现了**上下文感知、时序敏感的 agent 选择与融合机制**

### ⭐ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **效率提升** | 动态激活更少的 agents，显著降低推理开销 |
| **结构新颖性** | 首次将 MoE 中的 recurrent routing 思想引入 MoA 框架 |
| **可扩展性** | 支持多层迭代聚合，并通过记忆机制增强一致性 |
| **训练友好** | 设计了专门的 **router loss function**，包含任务损失 + 熵正则化 + 可选负载均衡项 |

> 🔍 这是**首个在 MoA 框架中引入 router 进行 agent selection 的工作**，实现了从“固定融合”到“语言模型式调优”的转变。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验在多个主流指令跟随基准上进行评估：
- **AlpacaEval 2.0**：805 条真实场景指令，使用长度控制（Length-Controlled, LC）胜率作为主要指标
- **MT-Bench**：多轮对话评测，由 GPT-4 打分（0–10 分），评估一、二轮表现
- **Arena-Hard**：500 条高难度查询，涵盖编程、数学、逻辑等复杂任务
- **FLASK**：细粒度技能维度分析（共12维）

### 🧪 实验设置与评估指标
| 设置项 | 内容 |
|-------|------|
| **Agent 数量** | 测试了 $ n = 1, 2, 3, 6 $ 种配置 |
| **层数（Layers）** | 多层聚合结构（通常为 3–4 层） |
| **评估指标** |  
- **Win Rate（LC & overall）**（AlpacaEval）
- **Average Score / Turn-wise Score**（MT-Bench）
- **Relative Inference Time**（运行时效率对比）
- **Agent Activation Count**（激活代理数）

### 🔁 基线方法对比
| 基线模型 | 描述 |
|--------|------|
| **Baseline MoA** | 标准 MoA，静态聚合，无历史信息利用 |
| **Ablated Model** | 替换 LSTM 为简单线性层，移除时序依赖 |
| **MoA-Lite** | 轻量版 MoA，用于效率比较 |
| **GPT-4 系列模型** | 包括 GPT-4 Turbo、Omni、Preview 等作为强基线 |
| **开源大模型** | 如 Llama 3 70B、Qwen1.5 110B、Mixtral 8×22B 等 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

#### ✅ AlpacaEval 2.0 结果
| Model | LC Win Rate | Overall Win Rate |
|-------|-------------|------------------|
| MoA | 65.1±0.6% | 59.8±0.3% |
| **MMoA** | **61.5±0.4%** | **58.0±0.5%** |
| MoA-Lite | 59.3±0.2% | 57.0±0.7% |

> ➤ MMoA 在 LC 胜率上略低于标准 MoA（↓3.6%），但整体胜率仅下降 **1.8%**，仍优于多数基线。

#### ✅ MT-Bench 结果
| Model | Avg. Score | 1st Turn | 2nd Turn |
|-------|----------|---------|----------|
| MoA | 9.25±0.10 | 9.44 | 9.07 |
| **MMoA** | **9.20±0.08** | 9.42 | 9.05 |

> ➤ 平均得分仅下降 **0.05**，几乎持平，说明语义质量保持良好。

### ⏱️ 时间复杂度与效率对比（Table 3）

| Agent 数量 $n$ | MMoA 推理时间占比（vs 单提议者） | Baseline MoA | 绝对节省 |
|---------------|-------------------------------|--------------|-----------|
| 6             | 56.7%                        | 61.3%        | **4.6%**  |
| 3             | 56.1%                        | 58.0%        | 1.9%      |
| 2             | 54.5%                        | 58.8%        | **4.3%**  |
| 1             | 47.8%                        | 47.8%        | 0%（持平）|

> ✅ MMoA 将有效时间复杂度从 $O(nL)$ 降至约 $O(n + L)$，通过复用 LSTM 隐藏状态减少重复计算。

### 🔍 消融实验结果（隐含于分析中）
- **移除 recurrence 结构（ablated model）**：性能下降明显，验证了历史信息的重要性
- **早期层即实现高准确率**：如 Figure 2 显示，MMoA 在 Layer 2 即达到 ~42% LC 胜率，远超原始 MoA
- **更少 agent 激活带来更高性价比**：尽管最终胜率微降，但计算成本大幅降低，形成优良 trade-off

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **首次成功将 recurrent routing 引入 MoA 框架**，实现 agent selection 的动态化与记忆化。
2. **MMoA 在精度损失极小的前提下显著提升运行效率**，最高节省 **4.6% 推理时间**。
3. **早期聚合即可获得高质量输出**，适合需要快速响应的应用场景。
4. **提出的 router loss 函数有效引导 agent 分工与探索行为**，支持端到端训练。

### ⚠️ 方法的局限性
1. **评测范围有限**：仅在 instruction-following 任务上验证，未覆盖长文本推理、工具调用、多语言或安全敏感场景。
2. **架构选择受限**：仅使用 LSTM 作为 recurrence 模块，未探索 Transformer-based 或 RL-based router。
3. **敏感性未充分分析**：对 agent pool 构成、训练数据分布、聚合深度等因素的影响尚不明确。
4. **缺乏定性分析**：未深入研究失败案例、校准性（calibration）或鲁棒性问题。

### 🔮 未来工作方向（见 Appendix）
- 探索 **Reinforcement Learning for Human Feedback (RLHF)** 技术优化 router 训练
- 引入 **token-level routing** 或 **multi-head gating** 机制
- 扩展至 **tool-augmented agents** 和 **multi-modal settings**
- 开展大规模 **real-world deployment study** 验证实用性

---

## ✅ 总结一句话
**MMoA 是首个引入 recurrence router 的 MoA 框架，在几乎不牺牲性能的情况下显著提升了多智能体系统的推理效率，为构建高效、自适应的 LLM 协作系统提供了新范式。**

</details>

---

### 14. [SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models](https://arxiv.org/abs/2605.19357)

**Authors**: Yiyang Gu, Junwei Yang, Junyu Luo, Ye Yuan, Bin Feng, Yingce Xia, Shufang Xie, Kaili Liu, Bohan Wu, Qi Shi, Haoran Li, Beier Xiao, Zhiping Xiao, Xiao Luo, Weizhi Zhang, Philip S. Yu, Zequn Liu, Ming Zhang  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.19357v1  

#### Abstract
Large language models (LLMs) are increasingly applied to scientific research, yet existing evaluations often fail to reflect the fine-grained capabilities required in practice. Most benchmarks are manually curated or domain-generic, limiting scalability and alignment with real scientific use cases. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 LLM 科学能力评测基准（如 MMLU、GPQA）通常是**通用型、静态且手动构建**的，难以反映特定科学应用场景下的细粒度能力需求。这些基准在实际科研任务中表现与真实性能脱节（**低对齐性**），且无法扩展到新兴或高度专业化领域（如“周环反应”）。此外，依赖专家标注或合成问题生成的方法成本高、不可持续。

### 提出的新方法与思路
作者提出 **SciCustom** ——一个基于本体论（ontology）驱动的自动化框架，用于定制化构建面向具体科学应用的评测基准。其核心思想是：
- 将复杂科学任务分解为可复用的**细粒度知识单元（knowledge units）**；
- 利用大规模真实科学语料库进行数据锚定，确保评估的**事实基础性（grounded validity）**；
- 支持按需动态组合知识单元，实现**高效、可扩展的定制化 benchmark 构建**。

### 相比现有方法的优势
| 维度 | 传统方法 | SciCustom |
|------|--------|---------|
| **可扩展性** | 手动标注，难以扩展 | 自动化流程，支持任意新需求 |
| **对齐性** | 通用 benchmark 与特定任务脱节 | 高度匹配目标应用场景 |
| **数据来源** | 合成数据或小规模标注 | 基于大规模真实 QA 对 |
| **复用性** | 每个任务从零开始 | 知识单元可跨任务复用 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **源数据 Corpus D**：聚合了多个高质量科学 QA 数据集，总计约 **200万条实例**，包括：
  - `SciRIFF`, `SciInstruct`, `Mol-Instruct`, `MultiMedQA`, `SciEval`, `MMLU-Pro`, `GPQA`, `IfBench`, `SimpleQA` 等。
- **Ground-truth 基准**（用于验证对齐性）：
  - 化学领域：来自 **ChemBench** 的6个子任务（如有机化学、物理化学等）
  - 医疗健康领域：来自 **MMLU-Pro 的 health subset** 的5个子任务（如病毒学、营养学等）

### 实验设置与评估指标
- **目标**：衡量 SciCustom 构建的 benchmark 所产生的模型排名是否与专家 ground-truth 排名一致。
- **评估协议**：
  1. 在 ground-truth benchmark 上评估 10 个主流 LLM（如 GPT-4o、Claude-opus、Gemini 等），得到参考排名；
  2. 使用各 baseline 方法（含 SciCustom）构建 benchmark，并在同一组 LLM 上测试，获得替代排名；
  3. 计算两个排名之间的 **Spearman 和 Kendall 相关系数**作为一致性指标。

### 基线方法对比
| 类型 | 基线名称 | 描述 |
|------|--------|------|
| **通用 benchmark** | IfBench, SimpleQA | 测试指令遵循能力 |
| **科学通用 benchmark** | GPQA, MMLU | 覆盖广泛科学常识 |
| **领域专用 benchmark** | MedQA (医疗), MMLU-Pro 子集 (化学) | 专家构建，但粒度粗 |
| **替代构建方法** | GPT-5（全合成）、Embedding（k-NN 检索） | 验证 grounded data 与 ontology 设计的重要性 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Spearman 相关性）

#### 表1：化学任务上的 ranking 一致性（Spearman）
| 方法 | 分析化学 | 无机化学 | 材料科学 | 有机化学 | 物理化学 | 技术化学 |
|------|----------|----------|-----------|------------|-------------|-------------|
| GPQA | 0.61 | 0.52 | 0.21 | 0.72 | 0.21 | 0.03 |
| MMLU | 0.21 | 0.27 | -0.61 | 0.21 | 0.52 | 0.31 |
| **SciCustom** | **0.86** | **0.67** | **0.42** | **0.89** | **0.74** | **0.86** |

> ✅ 在6项中有5项显著领先，在“技术化学”上相关性高达 **0.86**，远超 GPQA 的 0.03。

#### 表2：医疗任务上的 ranking 一致性（Spearman）
| 方法 | 病毒学 | 人类衰老 | 医学遗传学 | 解剖学 | 营养学 |
|------|--------|----------|-------------|--------|--------|
| MedQA | 0.44 | 0.62 | 0.35 | -0.19 | 0.45 |
| **SciCustom** | **0.55** | **0.49** | **0.42** | **0.62** | **0.78** |

> ✅ 在全部5项任务中均取得最高相关性，平均提升明显。

### 与基线方法的对比结果
- **相比通用 benchmark（如 GPQA）**：SciCustom 显著更贴近专家判断，尤其在专业性强的任务上（如 Technical Chemistry）差异巨大（0.03 vs 0.86）。
- **相比全合成方法（GPT-5）**：说明仅靠 LLM 生成问题缺乏 grounding，导致评估失真。
- **相比 embedding-based retrieval（Embedding）**：证明单纯语义相似检索不如基于 ontology 结构的知识映射精准。

### 消融实验结果（Ablation Study）
作者进行了以下消融分析（见 Figure 4）：
| 变体 | 描述 | 性能影响 |
|------|------|--------|
| **w/o cutoff** | 移除 binary search 相关性过滤 | 排名一致性下降明显 → 说明 cutoff 提升聚焦性 |
| **w/o selection** | 替换为代表性采样为随机抽样 | 性能大幅下滑 → 证明 proxy subset selection 至关重要 |

> 🔍 结果表明：**relevance-aware filtering** 和 **representative subset selection** 是保障高效且准确评估的关键组件。

---

## 4. 关键结论和发现

### 主要发现
1. **SciCustom 能有效捕捉细粒度科学能力差异**：其构建的 benchmark 与专家 ground-truth 排名高度一致（Spearman 最高达 0.89），远优于现有通用或领域 benchmark。
2. **无需人工标注即可实现高质量评估**：通过 ontology-driven tagging + 多模型投票机制，实现了完全自动化的 benchmark 构建流程。
3. **适用于无先验 benchmark 的新颖场景**：成功为“Pericyclic Reaction”这一尚无标准评测的任务构建了高质量 MCQ benchmark，展示了极强的泛化能力。
4. **知识单元的结构性组织优于扁平化检索**：相比纯 embedding 检索，基于 ontology 的层级结构能更好理解复杂科学概念间的关联。

### 方法的局限性
1. **覆盖范围受限于当前本体**：目前主要涵盖生物医学与化学领域（源自 OBO、BioPortal 等），尚未包括数学、理论物理等抽象学科。
2. **依赖源语料库的覆盖率**：部分知识单元存在数据稀疏问题，可能影响评估稳定性。
3. **tagger 性能有限**：尽管 Macro F1 达 75.2%，但仍有一定误差，可能引入噪声。

### 未来工作方向
- 扩展至更多科学领域（如数学、地球科学），集成更广泛的科学 taxonomy；
- 动态监控低资源 knowledge units 并主动补充数据；
- 引入反馈机制优化 tagger 与 selection 策略；
- 探索多模态科学数据（图表、公式）的支持。

---

> 📌 **总结一句话**：  
> **SciCustom 提供了一个可扩展、应用感知、数据锚定的 LLM 科学能力评测新范式，解决了传统 benchmark “不贴合、难扩展、缺根基”的三大痛点。**

GitHub 开源地址：[https://github.com/yjwtheonly/SciCustom](https://github.com/yjwtheonly/SciCustom)

</details>

---

### 15. [ClinSeekAgent: Automating Multimodal Evidence Seeking for Agentic Clinical Reasoning](https://arxiv.org/abs/2605.20176)

**Authors**: Juncheng Wu, Letian Zhang, Yuhan Wang, Haoqin Tu, Hardy Chen, Zijun Wang, Cihang Xie, Yuyin Zhou  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.20176v1  

#### Abstract
Large language models (LLMs) and agentic systems have shown promise for clinical decision support, but existing works largely assume that evidence has already been curated and handed to the model. Real-world clinical workflows instead require agents to actively seek, iteratively plan, and synthesize...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ClinSeekAgent: Automating Multimodal Evidence Seeking for Agentic Clinical Reasoning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的临床决策支持系统大多依赖于**预先整理好的、静态的证据包**（curated evidence），例如人工筛选的患者病历摘要或结构化的小型病例描述。这种范式在真实临床场景中存在严重局限：
- 忽略了从原始、异构数据源（如原始EHR表、医学影像、外部知识库）中主动检索证据的过程；
- 无法模拟医生“提出问题 → 寻找证据 → 综合推理”的动态决策流程；
- 在风险预测、多模态任务中容易遗漏稀疏但关键的纵向信号。

### **提出的新方法与思路**
本文提出了 **ClinSeekAgent**，一个用于**动态多模态证据搜寻**的自动化智能体框架，其核心思想是：
- 将临床推理从“被动消费证据”转变为“主动获取证据”；
- 构建统一的工具空间（tool space），使LLM能够通过调用工具来访问三种互补的信息源：
  - **EHR retrieval**：查询原始电子健康记录（EHR）数据库；
  - **Web search**：浏览器搜索外部医学知识；
  - **Medical imaging tools**：对DICOM影像进行分类、报告生成、短语定位等分析。
- 支持**迭代式、闭环的 agentic workflow**：模型可基于已有观察继续提问、调用新工具，直到形成最终判断。

### **相比现有方法的优势**
| 方面 | 传统方法 | ClinSeekAgent |
|------|--------|---------------|
| **证据来源** | 静态、预选上下文 | 动态、自主检索原始数据 |
| **模态支持** | 多为文本或有限图像 | 融合EHR + 影像 + 外部知识 |
| **灵活性** | 固定输入格式 | 开放式工具交互协议 |
| **应用场景** | 单步推理 | 多跳、长视野证据整合 |
| **训练潜力** | 仅用于推理 | 可作为训练管道蒸馏轨迹 |

此外，作者还展示了该框架不仅可用于**inference-time agenting**，也可作为**training-time pipeline**，将强教师模型的探索轨迹蒸馏到小型开源模型中。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
构建了两个评估基准：

#### **(1) ClinSeek-Bench**（本文提出）
- **Text-only EHR tasks**：源自 **EHR-Bench** [12]，包含45个子任务，涵盖：
  - 决策制定（Decision Making）
  - 风险预测（Risk Prediction）
  - 涉及诊断、实验室检查、转移等多个维度
- **Multimodal tasks**：融合 **MIMIC-IV EHR** 和 **MIMIC-CXR** 数据，改编自：
  - **EHRXQA** [13]
  - **MedMod** [14]
  - 包含6个任务组：
    - CXR finding presence / enumeration / temporal change comparison
    - 24小时去代偿预测
    - 住院死亡率预测
    - 表型预测（Phenotype prediction）

每个样本都配对设计为两种模式：
- **Curated Input**：保留原基准提供的结构化上下文（如最近24小时事件）；
- **Automated Evidence-Seeking**：仅提供患者ID和时间戳，要求模型使用ClinSeekAgent工具自行检索所有必要信息。

### **实验设置与评估指标**
- **Primary Metric**：Sample-wise F1 (%)，按任务组平均后计算总体得分。
- **Evaluated Models**（共12个）：
  - 闭源模型：Claude Opus 4.6, Claude Sonnet 4.6, GLM-4.7, MiniMax M2.5, Kimi K2.5 等
  - 开源模型：Qwen3.5-35B-A3B, Gemma-4-26B-A4B-it, gpt-oss-120B 等
- **交互机制**：
  - 最大交互轮次：200
  - 工具调用接口标准化（见Appendix C）
  - 图像最长边缩放至≤1568像素
- **推理平台**：
  - 闭源模型通过API调用（如AWS Bedrock）
  - 开源模型使用vLLM部署并兼容OpenAI API

### **基线方法对比**
- **Curated Input**：直接基于预提取的上下文作答，代表当前主流做法；
- **ClinSeekAgent (Ours)**：在无上下文情况下，通过工具调用自主收集证据并推理。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **Text-only EHR Tasks 结果（Table 1）**
| Model | Curated Input (Overall F1) | ClinSeekAgent (Overall F1) | Δ |
|-------|----------------------------|---------------------------|----|
| **Claude Opus 4.6** | 60.0 | **63.2** | **+3.2** |
| **MiniMax M2.5** | 43.1 | **47.3** | **+4.2** |
| **Claude Sonnet 4.6** | 56.6 | 57.5 | +0.9 |
| **Qwen3.5-35B-A3B** | 46.8 | 47.0 | +0.2 |
| **Kimi K2.5** | 49.2 | 37.9 | -11.3 |

> 🔹 **优势集中在 Risk Prediction 子任务**：7/9 模型取得正增益  
> 🔹 **最强提升来自 Opus 和 MiniMax**，说明高阶规划能力是关键驱动因素

#### ✅ **Multimodal Tasks 结果（Table 2）**
| Model | Curated Input (Overall F1) | ClinSeekAgent (Overall F1) | Δ |
|-------|----------------------------|---------------------------|----|
| **Claude Opus 4.6** | 47.5 | **62.6** | **+15.1** |
| **Claude Sonnet 4.6** | 48.0 | 54.9 | +6.9 |
| **Qwen3-VL-235B** | 43.9 | 49.8 | +5.9 |
| **Gemma-4-26B-A4B-it** | 38.2 | 44.9 | +6.7 |

> 🔹 所有具备多模态能力的模型均受益，且**越强的agent提升越大**  
> 🔹 在 **CXR相关任务** 和 **Phenotype Prediction** 上提升显著（如Opus在Phenotype上+34.0）

#### ✅ **训练时验证：ClinSeek-35B-A3B 蒸馏效果（Table 3）**
以 **Claude Opus 4.6** 为teacher，在 **AgentEHR-Bench** 上蒸馏出 **ClinSeek-35B-A3B**

| Model | Average F1 | Diagnoses | Labs | Microbio | Procedures | Transfers |
|-------|------------|----------|------|----------|------------|-----------|
| **Qwen3.5-35B-A3B (Base)** | 22.1 | 36.6 | 17.7 | 16.2 | 21.9 | 18.1 |
| **ClinSeek-35B-A3B (Ours)** | **34.0** | **55.4** | **38.5** | **27.6** | **31.7** | 16.7 |
| **Δ** | **+11.9** | +18.8 | +20.8 | +11.4 | +9.8 | -1.4 |

> 🔹 显著超越所有开源基线，接近 **Claude Opus 4.6**（36.0）  
> 🔹 特别擅长 **Diagnosis** 和 **Lab Event Prediction**

---

## **4. 关键结论和发现**

### **主要发现**
1. **主动证据搜寻优于被动上下文消费**：
   - 当模型具备足够工具使用和长期规划能力时，ClinSeekAgent能有效挖掘稀疏、分布式的临床信号（如长期趋势、影像细节、外部定义），从而提升性能。
   
2. **收益在特定任务中更明显**：
   - **Risk Prediction** > Decision Making：因前者依赖罕见但决定性的证据（如一次低血压发作），而后者需精准识别高频动作模式。
   - **Multimodal Tasks** 提升最大：组合EHR + CXR + 外部知识带来质变，尤其在表型分类中恢复了黄金标签所需的本体词汇（如Harutyunyan-2019 taxonomy）。

3. **可迁移性强**：
   - 强大的teacher模型生成的探索轨迹可以成功蒸馏到较小的student模型中，实现**开放模型的 agentic capability 迁移**。
   - 学生模型学会了更灵活地使用 `ehr.run_sql_query`（SQL调用占比从2.0% → 12.5%），表明学到了程序化数据库操作策略，而非简单模仿答案。

4. **失败案例揭示挑战**：
   - 在部分 **decision-making 任务** 中表现下降，主因是检索过程可能引入噪声或错过关键上下文；
   - 一些轨迹冗余（redundant tool calls），影响效率和上下文窗口利用率。

---

### **局限性（Limitations）**
1. **当前多模态任务仍较简单**：
   - 多数任务可通过少量工具调用解决，未充分测试复杂跨模态迭代推理。
2. **Teacher轨迹质量参差**：
   - 当前依赖监督微调（SFT），但teacher本身可能产生低效或冗余行为，污染训练数据。
3. **缺乏强化学习优化**：
   - 未对“成功且简洁”的探索路径进行显式奖励，未来可通过RLHF/RFT改进。

---

### **未来工作方向**
1. **构建更具挑战性的多模态临床基准**：
   - 设计需要长时间跨度、多跳、跨模态回溯的任务。
2. **优化教师轨迹质量**：
   - 引入过滤、压缩、精炼机制，去除冗余步骤。
3. **引入强化学习**：
   - 使用 reward modeling 或 RL 来优化工具使用效率与推理路径长度。
4. **扩展至更多模态与工具**：
   - 如病理切片、基因组数据、语音记录等。
5. **推进临床落地验证**：
   - 在真实医院环境中测试系统的安全性、可解释性和医生接受度。

---

> 📌 **一句话总结**：  
> **ClinSeekAgent 推动了临床AI从“读题答题”向“主动查资料+综合判断”的转变，证明了 agentic evidence seeking 在真实医疗决策中的巨大潜力，并为开源社区提供了可复现、可训练的高质量临床智能体构建范式。**

</details>

---

### 16. [DAG-Based QoS-Aware Dynamic Task Placement for Networked Multi-Stage Control Pipelines](https://arxiv.org/abs/2605.19887)

**Authors**: Thien Tran, Jonathan Kua, Thuong Hoang, Minh Tran, Yuemin Ding, Jiong Jin  
**Category**: cs.DC  
**Published**: 2026-05-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.19887v1  

#### Abstract
Current Physical AI (PAI) relies heavily on closed-loop visual-servoing pipelines, whose perception and planning stages may become computationally intensive onboard due to complex models embedded on robots. In practice, offloading the perception task to on-site edges statically is inappropriate for ...

---

### 17. [Multi-Pedestrian Safety Warning at Urban Intersections Use Case of Digital Twin](https://arxiv.org/abs/2605.18823)

**Authors**: Yongjie Fu, Qi Gao, Mahshid Ghasemi Dehkordi, Gil Zussman, Xuan Di  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.18823v1  

#### Abstract
Digital twins (DTs) for urban transportation systems have gained increasing attention; however, their systematic evaluation in safety-critical scenarios remains limited. This paper presents a multi-pedestrian safety warning system at urban intersections enabled by a tightly coupled physical-digital ...

---

### 18. [Memory-Augmented Reinforcement Learning Agent for CAD Generation](https://arxiv.org/abs/2605.19748)

**Authors**: Yin Xiaolong, Liu Yu, Shen Jiahang, Lu Xingyu, Ni Jingzhe, Fan Fengxiao, Sang Fan  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19748v1  

#### Abstract
Automatic generation of computer-aided design (CAD) models is a core technology for enabling intelligence in advanced manufacturing. Existing generation methods based on large language models (LLMs) often fall short when handling complex CAD models characterized by long operation sequences, diverse ...

---

### 19. [CogScale: Scalable Benchmark for Sequence Processing](https://arxiv.org/abs/2605.19758)

**Authors**: Yannis Bendi-Ouis (Mnemosyne), Romain de Coudenhove (ENS-PSL), Xavier Hinaut (Mnemosyne)  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19758v1  

#### Abstract
The ability to maintain and manipulate information over time is a fundamental aspect of living beings and Artificial Intelligence. While modern models have achieved remarkable success in tasks like natural language processing, evaluating the capacity of novel architectures to process sequential info...

---

### 20. [OpenComputer: Verifiable Software Worlds for Computer-Use Agents](https://arxiv.org/abs/2605.19769)

**Authors**: Jinbiao Wei, Qianran Ma, Yilun Zhao, Xiao Zhou, Kangqi Ni, Guo Gan, Arman Cohan  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19769v1  

#### Abstract
We present OpenComputer, a verifier-grounded framework for constructing verifiable software worlds for computer-use agents. OpenComputer integrates four components: (1) app-specific state verifiers that expose structured inspection endpoints over real applications, (2) a self-evolving verification l...

---

### 21. [Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization](https://arxiv.org/abs/2605.19782)

**Authors**: Dmitry Redko (Applied AI Institute), Albert Fazlyev (AI Talent Hub, ITMO University), Konstantin Sozykin (Applied AI Institute), Maria Ivanova (YSDA, Applied AI Institute), Evgeny Burnaev (Applied AI Institute), Egor Shvetsov (Applied AI Institute)  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19782v1  

#### Abstract
LLM discovery and optimization systems are increasingly applied across domains, implementing a common propose-evaluate-revise loop. Such optimization or discovery progresses via context conditioning on received feedback from an environment. However, as modern LLM agents are increasingly complex in t...

---

### 22. [Cross-Paradigm Knowledge Distillation: A Comprehensive Study of Bidirectional Transfer Between Random Forests and Deep Neural Networks for Big Data Applications](https://arxiv.org/abs/2605.19299)

**Authors**: Mahdi Naser Moghadasi  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19299v1  

#### Abstract
The exponential growth of big data has intensified the need for efficient and interpretable machine learning models that can handle diverse data characteristics while maintaining computational efficiency. Knowledge distillation has primarily focused on neural network-to-neural network transfer, leav...

---

### 23. [An Objective Performance Evaluation of the LSTM Networks in Time Series Classification](https://arxiv.org/abs/2605.19311)

**Authors**: Sooraj Sunil, Balakumar Balasingam  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19311v1  

#### Abstract
The rapid adoption of deep learning has increasingly led to data-driven models replacing classical model-based algorithms, even in domains governed by well-understood physical laws. While data-driven models, such as long short-term memory (LSTM) networks, have become a popular choice for time-series...

---

### 24. [Accurate, Efficient, and Explainable Deep Learning Approaches for Environmental Science Problems](https://arxiv.org/abs/2605.19366)

**Authors**: Jimeng Shi  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19366v1  

#### Abstract
Environmental science plays a pivotal role in safeguarding ecosystems, a domain driven by large-scale, heterogeneous data. In the big data era, artificial intelligence (AI) has emerged as a transformative tool for learning patterns and supporting decision-making. This dissertation develops AI-based ...

---

### 25. [PRISM: A Benchmark for Programmatic Spatial-Temporal Reasoning](https://arxiv.org/abs/2605.19382)

**Authors**: Qiran Zhang, Yuheng Wang, Runde Yang, Lin Wu, Jingru Fan, Shu Yao, Jie Zhang, Tianle Zhou, Huatao Li, Ruijie Shi, Yihan Li, Chen Qian  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.19382v1  

#### Abstract
Programmatic video generation through code offers geometric precision and temporal coherence beyond pixel-level diffusion models, yet rigorously evaluating whether language models can produce spatially correct animated outputs remains an open problem. We introduce PRISM, a large-scale benchmark of 1...

---

### 26. [What and When to Distill: Selective Hindsight Distillation for Multi-Turn Agents](https://arxiv.org/abs/2605.19447)

**Authors**: Xiaozhe Li, Tianyi Lyu, Yang Li, Yichuan Ma, Peiji Li, Linyang Li, Qipeng Guo, Dahua Lin, Kai Chen  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.19447v1  

#### Abstract
Reinforcement learning can train LLM agents from sparse task rewards, but long-horizon credit assignment remains challenging: a single success-or-failure signal must be distributed across many actions. Existing methods rely on trajectory-level rewards or proxy signals, without fully leveraging per-s...

---

### 27. [When Tabular Foundation Models Meet Strategic Tabular Data: A Prior Alignment Approach](https://arxiv.org/abs/2605.19662)

**Authors**: Xinpeng Lv, Yunxin Mao, Renzhe Xu, Chunyuan Zheng, Yikai Chen, Haoxuan Li, Jinxuan Yang, Kun Kuang, Yuanlong Chen, Mingyang Geng, Wanrong Huang, Shixuan Liu, Shaowu Yang, Wenjing Yang, Zhouchen Lin, Haotian Wang  
**Category**: cs.AI  
**Published**: 2026-05-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.19662v1  

#### Abstract
Tabular foundation models based on pretrained prior-data fitted networks~(PFNs) have shown strong generalization on diverse tabular tasks, but they are typically designed for \emph{non-strategic} settings where data distributions are independent of deployed classifiers. In many real-world decision s...

---

### 28. [Language models struggle with compartmentalization](https://arxiv.org/abs/2605.19284)

**Authors**: Thomas Vincent Howe, David Wingate  
**Category**: cs.CL  
**Published**: 2026-05-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.19284v1  

#### Abstract
In the training data used by large language models (LLMs), the same latent concept is often presented in multiple distinct ways: the same facts appear in English and Swahili; many functions can be expressed in both Python and Haskell; we can express propositions in both formal and natural language. ...

---

### 29. [Quantum-Enhanced Distributed Sensor Fusion: Lower Bounds on Aggregation from Projection Noise to Heisenberg-Limited Byzantine-Tolerant Networks](https://arxiv.org/abs/2605.19327)

**Authors**: Vasanth Iyer, S. S. Iyengar  
**Category**: cs.DC  
**Published**: 2026-05-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.19327v1  

#### Abstract
We derive unified lower bounds on the mean squared error (MSE) of distributed quantum sensor fusion under Byzantine faults and decoherence. Building on the classical Brooks-Iyengar overlap function and its vector extension, the predictive outlier model for virtual sensor tracking, and SPOTLESS spati...

---

### 30. [TabQL: In-Context Q-Learning with Tabular Foundation Models](https://arxiv.org/abs/2605.18979)

**Authors**: Qisai Liu, Zhanhong Jiang, Timilehin Ayanlade, Ashutosh Kumar Nirala, Yang Li, Aditya Balu, Soumik Sarkar  
**Category**: cs.LG  
**Published**: 2026-05-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.18979v1  

#### Abstract
We propose Tabular Q-Learning (TabQL), a reinforcement learning framework that replaces the conventional parametric Q-network in Deep Q-Learning (DQN) with a tabular foundation model endowed with in-context learning capabilities. The key idea is to represent Q-values through a sequence-to-sequence f...

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
