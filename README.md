# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-22 07:18:58 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training](https://arxiv.org/abs/2604.19241)

**Authors**: Size Zheng, Xuegui Zheng, Li-wen Chang, Jidong Zhai  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2604.19241v1  

#### Abstract
The exponential growth in Large Language Model (LLM) parameters has transformed model training into an increasingly resource-intensive endeavor. With the stagnation of Moore's Law and the widening disparity between computation throughput and communication bandwidth, expert parallelism (EP) has emerg...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在大规模语言模型（LLM）训练中，**Mixture-of-Experts (MoE)** 架构通过扩展专家数量来提升模型容量，但随之而来的是显著的通信开销。尤其是在 **Expert Parallelism (EP)** 设置下，GPU之间需要频繁执行 All-to-All 或 AllGather 操作进行 token 路由，导致通信成为瓶颈。

现有优化方案存在以下问题：
- **数值不稳定性**：多数通信-计算重叠（overlap）策略依赖微批次拆分（micro-batch splitting），破坏了浮点运算的结合律，导致梯度累积顺序变化，影响训练可复现性（bitwise reproducibility）。
- **缺乏统一抽象**：不同通信模式（如 AllGather vs AllToAll）需维护多个专用 kernel，增加调优复杂度。
- **系统级开销大**：多 CUDA stream 管理依赖 CPU 同步，引入延迟和调度气泡（synchronization bubbles）。

---

### 提出了什么新方法或新思路

本文提出 **UNIEP**，一个面向 MoE 训练的统一专家并行 MegaKernel 系统，其核心思想是：

#### ✅ **MegaKernel Fusion**
将 MoE 中的 `Dispatch` → `GroupGEMM` → `Combine` 子图融合为单个 CUDA kernel，实现细粒度的 **communication-computation overlap**，无需多 stream 协调。

#### ✅ **确定性 Token 排序机制（Deterministic Token Ordering）**
设计全局 token 映射算法（Algorithm 1），确保无论并行调度如何，所有 token 的接收和累加顺序严格一致，从而保证 **bitwise numerical equivalence** 与串行执行等价。

#### ✅ **统一通信抽象与参数化搜索空间**
通过引入 **token remapping** 和 **parameterized communication primitive**，UNIEP 可自适应选择最优通信路径（AllGather/AllToAll），无需显式切换 kernel。整个优化空间被建模为一组可调参数（如 SM 分配、warp 数量等）。

#### ✅ **基于 SM 的动态调度与 Scoreboard 同步**
利用 GPU 内部的 Streaming Multiprocessors (SMs) 进行动态角色分配（Comm-Worker / Comp-Worker / Relay-Worker），并通过全局 **scoreboard** 在设备端完成同步，消除 CPU 干预。

#### ✅ **带宽优化：Relay Worker 多播**
当多个目标专家位于同一 GPU 上时，仅传输一次 token，并由目标端的 Relay Worker 在本地 HBM 内复制，大幅减少 NVLink 流量（理论节省 ~34%）。

---

### 相比现有方法的优势

| 维度 | UNIEP | 现有方法（如 COMET、DeepEP） |
|------|-------|-------------------------------|
| **数值一致性** | ✅ 保证 bitwise reproducibility | ❌ 微批次拆分导致非确定性 |
| **通信效率** | ✅ 内部多播降低物理流量 | ❌ 每个路由独立发送 |
| **系统开销** | ✅ 单 stream + 设备端同步 | ❌ 多 stream + CPU 控制 |
| **配置灵活性** | ✅ 自动调优统一参数空间 | ❌ 手动切换 kernel 与启发式规则 |

---

## 2. 核心实验方法和设置

### 使用的模型与工作负载

实验基于真实生产级 MoE 模型，涵盖：
- **DeepSeek** 系列（如 DeepSeek-V3）
- **Qwen3** 系列（如 Qwen3-235B-A22B）
- **Kimi** 系列（如 Kimi-K2）

共 **12 种 MoE 配置**（见 Table 4），覆盖：
- 专家数：64–512
- Top-k：6–10
- Hidden dimension：1024–7168
- 序列长度：8k, 32k, 128k, 512k

> ⚠️ 注意：这些是 **模型架构配置**，而非传统意义上的“数据集”。

---

### 实验设置和评估指标

#### 硬件环境
在两个 NVIDIA Hopper GPU 集群上测试：
| 集群 | NVLink 带宽（单向） | GPU 数量 | 特点 |
|------|---------------------|----------|------|
| Cluster 1 | 200 GB/s | 8 GPUs/node | 带宽受限 |
| Cluster 2 | 400 GB/s | 8 GPUs/node | 高带宽 |

均使用 **BF16** 精度。

#### 评估指标
- **端到端延迟**（forward/backward pass）
- **吞吐量**（tokens/day）
- **加速比**（speedup over baselines）
- **数值精度差异**（max absolute diff, % non-bitwise-equal）
- **消融分析**（ablation on bandwidth opt, auto-tuning）

#### 自动调优机制
- 搜索空间大小约 **10⁵** 配置
- 使用 C++ + OpenMP 实现快速求解器（~144ms 完成搜索）
- 引入 **bucketing memoization**：按序列长度假设离散化缓存最优配置

---

### 基线方法对比

| 基线 | 描述 |
|------|------|
| **Serial (DeepEP + TransformerEngine)** | 非重叠 baseline，使用当前最优 kernel，无 overlap |
| **COMET [46]** | 当前最先进的重叠方案，采用双 stream 实现 fine-grained overlap，通信用 AllGather，计算用 CUTLASS |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔹 Kernel-level 性能（Figure 3）
在 Cluster 1（带宽受限）上：
- **Dispatch+GroupGEMM**：
  - 相比 Serial：**18.4×**（8k seq） / **11.2×**（32k seq）
  - 相比 COMET：**1.32×**（8k） / **1.30×**（32k）
- **GroupGEMM+Combine**：
  - 相比 COMET：**1.31×–1.87×**

> 💡 更高增益出现在低带宽环境下，说明 UNIEP 对通信瓶颈缓解更有效。

#### 🔹 Layer-level 性能（Figure 4）
在 Cluster 1 上（8k seq）：
- **Forward Pass**：比 COMET 快 **1.08×**
- **Backward Pass**：快 **1.03×**
- 随着序列增长至 32k，优势扩大至 **1.22× forward**

#### 🔹 长上下文性能（128k seq, Table 8）
- Cluster 1 上：
  - 相比 Serial：**6.90×**（forward）
  - 相比 COMET：**1.28×**
- Cluster 2 上：
  - 相比 COMET：**1.33×**

> 即使在极端长文本场景下仍保持显著优势。

#### 🔹 端到端训练吞吐（Section 6.7）
在 **128 GPU 集群**上运行实际训练任务（seq_len=512k）：
- 吞吐从 **127B tokens/day → 138B tokens/day**
- 实现 **1.09× speedup**

---

### 与基线方法的对比结果

| 方面 | UNIEP vs COMET |
|------|----------------|
| **性能** | 全面优于 COMET，kernel-level 达 **1.38×** 加速 |
| **数值一致性** | ✅ bitwise identical（Table 6）<br>❌ COMET 最大误差达 0.25，22%–29% 输出元素不一致 |
| **支持广度** | 支持更多 Hdim/Top-k 组合<br>COMET 因硬编码限制无法运行 MoE-2/MoE-12 |

---

### 消融实验结果（Table 9）

在 Cluster 1 上对 MoE-1 至 MoE-12 进行逐步增强验证：

| 阶段 | 描述 | 相比前一阶段提速 | 最终相比 COMET |
|------|------|------------------|----------------|
| **O** | 基础 MegaKernel（含 overlap + 动态调度） | — | 1.15×–1.36× |
| **B** | + Relay Worker 带宽优化 | **1.06×–1.36×** | — |
| **A**（完整 UNIEP） | + Auto-tuner + 优先级调度 | **1.15×–1.68×** | **1.24×–1.73×** |

> 表明 **auto-tuning** 是最大性能来源之一。

---

## 4. 关键结论和发现

### 论文的主要发现

1. **MegaKernel 可用于训练场景**  
   首次成功将 MegaKernel 技术应用于 MoE **训练**，而非仅限于推理，实现了高性能且精确的通信-计算重叠。

2. **数值一致性可以与高性能共存**  
   通过 **deterministic token mapping** 和 **pre-ordering**，UNIEP 在不牺牲精度的前提下实现 aggressive overlap，解决了长期存在的 trade-off。

3. **Relay Worker 显著降低通信量**  
   利用 intra-rank multicast 可减少高达 **34%** 的 NVLink 流量，在高 Top-k 场景下尤为有效。

4. **Analytical Performance Model 高效准确**  
   模型预测延迟误差仅 **0.5%–6.5%**，平均 3.8%，可在毫秒级内找到近似最优配置。

5. **实际部署带来可观收益**  
   在 128 GPU 规模下实现 **1.09× 吞吐提升**，对于千卡集群意味着每年节省数百万美元算力成本。

---

### 方法的局限性

- **实现复杂度高**：依赖 Triton-Distributed 编程框架，原生 CUDA 实现难度极大。
- **硬件依赖性强**：目前针对 Hopper 架构优化，迁移至其他平台（如 AMD）需重新验证。
- **未启用 TMA**：当前 GroupGEMM 未使用 Tensor Memory Accelerator，仍有进一步优化空间。
- **静态配置假设**：虽然支持自动调优，但仍假设 workload 特征稳定，动态变化场景适应性待验证。

---

### 未来工作方向

1. **扩展至其他通信模式**  
   将 UNIEP 抽象推广至 DP/TP/SP 等并行范式，构建通用分布式 MegaKernel 框架。

2. **集成 TMA 和异步拷贝**  
   利用 Hopper 的 TMA 引擎进一步提升内存访问效率。

3. **支持动态负载均衡**  
   结合 MegaBlocks 等技术，在 token 分布剧烈偏斜时动态调整资源分配。

4. **跨数据中心扩展**  
   探索在 HybridEP 或 Cross-Datacenter 场景下的应用潜力（参考 HybridEP [43]）。

5. **编译器自动化支持**  
   开发 DSL 或编译器 pass，自动识别 MoE 子图并生成 UNIEP-style MegaKernel。

---

> 📌 **总结一句话**：  
> UNIEP 通过 **统一的 MegaKernel 架构 + 确定性排序 + 自动调优**，首次实现了 **高性能、高精度、高可复现性** 的 MoE 训练通信优化，为下一代 LLM 训练系统提供了可靠基础设施。

</details>

---

### 2. [Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs](https://arxiv.org/abs/2604.18788)

**Authors**: Afsara Benazir, Felix Xiaozhu Lin  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2604.18788v1  

#### Abstract
Apple Neural Engine (ANE) is a dedicated neural processing unit (NPU) present in every Apple Silicon chip. Mixture-of-Experts (MoE) LLMs improve inference efficiency via sparse activation but are challenging for NPUs in three ways: expert routing is unpredictable and introduces dynamic tensor shapes...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **Mixture-of-Experts (MoE) 大语言模型在 Apple Silicon 的 Neural Processing Unit (NPU)** 上进行高效推理所面临的三大挑战：

1. **动态性与 NPU 静态执行约束的冲突**：MoE 的专家路由（expert routing）是动态的，导致输入张量形状不固定，而 NPU 要求静态、预编译的 compute graph。
2. **NPU 不支持关键动态算子**：如 `top-k`、`scatter/gather`、动态索引等操作无法在 NPU 上高效运行，必须回退到 CPU。
3. **频繁的小核调度开销大**：为每个小专家启动独立的 NPU kernel 会带来严重的 dispatch 和同步开销，降低利用率。

### 提出了什么新方法或新思路
作者提出了 **NPUMoE** —— 一个专为 Apple Silicon NPU 设计的 MoE LLM 推理引擎，其核心思想是：  
> **将密集、静态的计算卸载到 NPU，同时保留 CPU/GPU 作为动态操作的 fallback 路径**。

为此，设计了三项关键技术：

- **Static Tiers for Expert Capacity (静态专家容量分级)**  
  通过离线校准（offline calibration）估计各层中每个专家的“流行度”（popularity），并据此分配不同容量等级（tier）。热门专家获得更大容量，冷门专家则使用较小容量，从而在避免过度填充（padding）的同时控制溢出（overflow）风险。

- **Grouped Expert Execution (分组专家执行)**  
  将多个专家合并到一个统一的、静态形状的 FFN compute graph 中，一次调用即可处理多个专家，显著减少 NPU 的 kernel launch 次数和调度开销。

- **Load-Aware Expert Compute Graph Residency (负载感知的计算图驻留策略)**  
  决定哪些专家组应驻留在 NPU 上执行，哪些应回退到 CPU。只有那些能有效摊销 launch 开销的“热”专家组才保留在 NPU，而“冷”或细粒度分组则交由 CPU 处理，避免不必要的同步延迟。

### 相比现有方法的优势
- **首次系统性地将 MoE 动态推理适配于移动 NPU 架构**，填补了该领域的空白。
- 在保持高精度的前提下，显著提升了 NPU 利用率，实现了 **更低延迟、更高能效、更少 CPU 占用**。
- 相比直接使用 Core ML 或 ANEMLL 的默认调度策略，NPUMoE 能更好地协调 CPU-NPU 协同，避免资源浪费。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **BoolQ**：二分类问答任务，few-shot 设置（2-shot）
- **HellaSwag**：常识推理任务，few-shot 设置（10-shot）
- **RULER**：长上下文检索与聚合任务，zero-shot 设置

这些任务均以 **long-context prefill** 为主导，符合研究重点。

### 实验设置和评估指标

#### 硬件平台
- **Apple M2 Max**（64GB RAM, 12-core CPU, 16-core ANE）
- **Apple M2 Ultra**（192GB RAM, 24-core CPU, 32-core ANE）

#### 模型
| Model | #Layers | Total Params | #Experts | Sparsity |
|-------|--------|-------------|----------|----------|
| Phi-3.5-MoE-Instruct | 32 | 42.6B | 8 | 83.75 GB |
| Phi-tiny-MoE-Instruct | 32 | 3.8B | 8 | 7.51 GB |
| Qwen3-30B-A3B | 48 | 30.5B | 128 | 28.64 GB |

#### 评估指标
- **Runtime Latency**：Time-To-First-Token (TTFT)，衡量 prefill 阶段延迟
- **Energy Efficiency**：Energy Per Token (EPT)，单位为 mJ/token
- **CPU Usage**：CPU cycles 消耗，反映 NPU 卸载效果
- **Accuracy Degradation**：与 FP16 基线相比的准确率下降

#### 基线方法对比
| Baseline | Attention | Expert FFN |
|---------|-----------|------------|
| Core ML (CPU only) | CPU | CPU |
| Core ML (naive) | CPU | NPU（默认调度） |
| ANEMLL | NPU | CPU |
| **Ours (NPUMoE)** | **NPU** | **NPU（优化后）** |

所有方法均基于 Core ML 框架实现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（在 M2 Ultra 上，PhiMoE 模型）

| 指标 | 提升倍数（vs. Baselines） |
|------|--------------------------|
| **Prefill Latency (TTFT)** | **1.32x – 5.55x 更低** |
| **Energy Consumption (EPT)** | **1.81x – 7.37x 更优** |
| **CPU Cycle Usage** | **1.78x – 5.54x 减少** |

例如，在 `(prompt_len=1024, chunk_size=256)` 下：
- 相比 CoreML(naive)，延迟降低 **5.55x**
- 相比 CoreML(CPU)，能耗降低 **7.37x**

### 与基线方法的对比结果
- **CoreML(naive)**：虽然尝试将 FFN 卸载至 NPU，但由于缺乏 grouping 和 tiering，dispatch 开销巨大，实际性能反而最差。
- **ANEMLL**：擅长 dense transformer，但在 MoE 场景下未能优化专家调度，性能提升有限。
- **NPUMoE** 在所有配置下均达到 **Pareto 最优前沿**，兼顾低延迟、低功耗、低 CPU 占用。

### 消融实验结果（Ablation Study）

| 方法变体 | 描述 | 性能表现 |
|--------|------|----------|
| **Ours-base** | 仅基础算子划分 + 单专家图 | 性能最差，验证了调度开销的重要性 |
| **Ours-T** | + Static Tiers | 显著改善 padding 和 overflow 平衡 |
| **Ours-TG** | + Grouped Execution | 能效最佳，launch 开销大幅降低 |
| **Ours-all** | + Load-Aware Residency | **延迟最低，比 Ours-base 快 2.89x** |

> 结果表明：**Grouped Execution 是最大贡献者**，减少了约 54% 的 MoE block 运行时间。

此外：
- 平均 **35.35% 的 token 被 zero-padding**，说明盲目扩容不可取。
- **token drop rate 控制在 10–22%**，但对 accuracy 影响极小（<1.1% 降级），因仅丢弃低重要性 token。

---

## 4. 关键结论和发现

### 主要发现
1. **尽管 MoE 具有高度动态性，仍可通过系统级协同设计在 NPU 上高效运行**。
2. **静态化 + 分组 + 负载感知调度** 是解锁 NPU 效率的关键。
3. **prefill 阶段主导端到端延迟和能耗**，加速 prefill 对整体体验至关重要。
4. **Apple Silicon 的统一内存架构（Unified Memory）使完整 MoE 模型可驻留设备端**，减少数据搬移。
5. **专家激活具有强偏斜性和跨数据集稳定性**，使得离线校准（calibration）具有实用价值。

### 方法的局限性
- 当前优化主要针对 **long-context prefill** 工作负载，**decode 阶段未专门优化**。
- decode 是单 token 生成，难以摊销 CPU-NPU 同步成本，目前沿用相同 pipeline 效果不佳。
- compute graph 大小受限（如 >1.2GB 可能触发 fallback），限制了最大分组规模。
- 所有优化依赖于 **离线校准**，若部署场景与校准数据差异过大，可能影响效果。

### 未来工作方向
- **Decode 阶段专用优化**：探索 speculative decoding 或更轻量的 NPU 协同机制。
- **在线自适应校准**：动态调整 tier 和 grouping 策略以应对输入分布变化。
- **扩展至其他 NPU 平台**：如 Qualcomm Hexagon、AMD NPU 等，验证通用性。
- **支持动态 tensor 分区**：突破当前静态图限制，实现真正的混合设备执行。

---

> ✅ **总结一句话**：  
> **NPUMoE 成功将动态 MoE 推理“静态化”，通过 tiering、grouping 和 load-aware residency 三重技术，在 Apple Silicon NPU 上实现了高达 5.55x 的速度提升和 7.37x 的能效增益，是首个系统性解决 MoE-on-NPU 挑战的工作。**

</details>

---

### 3. [Are Large Language Models Economically Viable for Industry Deployment?](https://arxiv.org/abs/2604.19342)

**Authors**: Abdullah Mohammad, Sushant Kumar Ray, Pushkar Arora, Rafiq Ali, Ebad Shabbir, Gautam Siddharth Kashyap, Jiechao Gao, Usman Naseem  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.19342v1  

#### Abstract
Generative AI-powered by Large Language Models (LLMs)-is increasingly deployed in industry across healthcare decision support, financial analytics, enterprise retrieval, and conversational automation, where reliability, efficiency, and cost control are critical. In such settings, models must satisfy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Are Large Language Models Economically Viable for Industry Deployment?

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的LLM评估体系严重依赖**accuracy-centric**（以准确率为中心）的基准测试，忽视了工业部署中至关重要的经济性、能效、延迟和硬件利用率等操作层面的实际约束。这种现象被作者称为 **Deployment-Evaluation Gap**（部署-评估鸿沟），导致模型在实验室表现优异但在真实场景中难以盈利或可持续运行。

### 提出的新方法与新思路
为填补这一鸿沟，论文提出了 **EDGE-EVAL** ——一个面向工业部署生命周期的综合性基准框架。该框架不再仅关注模型精度，而是从全生命周期角度评估LLM的经济可行性。

#### 创新之处：
- **首次系统引入五个新型部署指标**，涵盖经济回报、能源效率、硬件密度、冷启动成本和量化保真度：
  - **Economic Break-Even (Nbreak)**：本地化适配回本所需的请求量
  - **Intelligence-Per-Watt (IPW)**：单位能耗下的任务性能输出
  - **System Density (psys)**：每GB VRAM支持的吞吐量（tokens/s/GB）
  - **Cold-Start Tax (Ctax)**：模型加载能耗相对于推理能耗的比例
  - **Quantization Fidelity (Qret)**：INT4量化后相对于FP16的任务性能保留率

- 强调“**生命周期评估**”（lifecycle assessment），覆盖 **adaptation → compression → inference** 全流程，反映真实部署路径。

### 相比现有方法的优势
| 维度 | 现有方法（如MLPerf、HuggingFace benchmarks） | EDGE-EVAL |
|------|---------------------------------------------|---------|
| 评估目标 | Accuracy / Throughput为主 | Operationally viable + Economically sustainable |
| 硬件环境 | 多样化、前沿设备 | 统一控制于**legacy hardware**（Tesla T4）模拟现实限制 |
| 成本考量 | 忽略训练/适配能耗与API替代成本 | 显式建模**adaptation cost**, **carbon footprint**, **ROI velocity** |
| 指标设计 | 单一维度（如latency） | 多维综合（Nbreak, IPW, psys, Ctax, Qret） |

> ✅ **优势总结**：EDGE-EVAL 更贴近中小企业和边缘部署的真实条件，提供更具决策指导意义的评估标准。

---

## 2. 核心实验方法和设置

### 使用的数据集
三个代表性工业任务对应不同数据集：
- **Summarization**: [XSum](https://huggingface.co/datasets/xsum)（新闻摘要，长文本压缩）
- **Retrieval-Augmented Generation (RAG)**: [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/)（基于检索的知识推理）
- **Conversational Agents**: [UltraChat](https://huggingface.co/datasets/Open-Orca/UltraChat)（多轮对话生成）

> 所有任务采用 70/15/15 划分，fine-tuning样本控制在5K–10K之间，避免过拟合且符合实际资源限制。

### 实验设置
- **硬件平台**：双卡 **NVIDIA Tesla T4 GPU**（每卡16GB VRAM），代表广泛使用的旧款工业级加速器。
- **模型家族**：
  - **LLaMA**（Grouped-Query Attention架构）：1B, 3B, 8B 参数版本
  - **Qwen-2.5**（Dense Transformer）：1.5B, 3B, 7B 参数版本
- **适配策略（PEFT）**：
  - LoRA-FP16
  - LoRA-INT8
  - LoRA-INT4（Post-Training Quantization, PTQ）
  - QLoRA-INT4（Quantization-Aware Training）
- **推理引擎**：使用 **vLLM**（v0.6.3）启用paged attention，batch size=1 模拟低并发工业负载。
- **测量工具**：通过 `pynvml` 采集GPU功耗（100ms粒度），实现细粒度能量分析。

### 评估指标
#### 主要部署指标定义如下：
| 指标 | 定义公式 | 含义 |
|------|--------|------|
| **Nbreak** | $ \frac{C_{\text{train}} + C_{\text{setup}}}{C_{\text{api}} - C_{\text{infer}}} $ | 回本所需请求数越小越好 |
| **IPW** | $ \frac{S_{\text{task}} \cdot \alpha}{E_{\text{req}}} $ | 单位能耗下智能产出越高越好 |
| **psys** | $ \frac{\text{Tput}}{M_{\text{VRAM}}} $ | 每GB显存服务容量，衡量硬件利用效率 |
| **Ctax** | $ \frac{E_{\text{load}}}{E_{\text{infer}}} $ | 冷启动开销倍数，越低越适合serverless |
| **Qret** | $ \frac{S_{\text{INT4}}}{S_{\text{FP16}}} \times 100\% $ | 4-bit量化后的性能保持率 |

> 此外还报告任务特定指标：ROUGE-L（Summ）、NLI Entailment（RAG）、LLM-as-a-Judge评分（Chat）。

### 基线方法对比
- 不同参数规模模型之间的横向比较：<2B vs 3B vs 7B+
- 不同量化策略对比：LoRA-FP16 vs LoRA-INT4 vs QLoRA-INT4
- 架构差异影响：LLaMA（GQA）vs Qwen（Dense）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 3）

| 模型 | Nbreak (reqs) | IPW | psys (tok/s/GB) | Qret (%) | Ctax (×) |
|------|----------------|-----|------------------|----------|----------|
| **LLaMA-3.2-1B (INT4)** | **14** | **0.45** | **6,930** | 100.6% | 183× |
| LLaMA-3B | 33 | 0.27 | 1,336 | 99.8% | 184× |
| LLaMA-7B | 43 | 0.15 | 387 | 100.3% | 230× |
| **Qwen-1.5B (INT4)** | 21 | **0.48** | **6,942** | 99.6% | 179× |

> 🔍 **观察**：所有<2B模型在 **Nbreak、IPW、psys** 上全面超越更大模型。

### 与基线方法的对比结果

#### （1）效率前沿（Efficiency Frontier）
- **<2B模型形成明显优势边界**：
  - LLaMA-1B 在 **14次请求内即可收回本地部署成本**（median）
  - 能效（IPW）是7B模型的 **3倍以上**
  - 系统密度达 **6,900 tokens/s/GB**，是7B模型的 **17×**

#### （2）量化推理收益显著（Table 3）
| 模型 | Precision | Throughput (tok/s) | Speedup | Energy/request | 节省 |
|------|-----------|--------------------|---------|----------------|------|
| LLaMA-1B | FP16 | 2,235 | 1.0x | 6.45 J | — |
| LLaMA-1B | INT4 | **4,331** | **1.94x** | **2.50 J** | **61%↓** |
| Qwen-7B | INT4 | 1,723 | 1.82x | ↓57% | |

✅ 结论：**INT4量化带来近2倍吞吐提升与超57%能耗下降**，是有效的“硬件乘数”。

#### （3）冷启动税极高（Figure 4d）
- 所有模型的 **Ctax > 179×**，意味着加载一次模型的能量相当于执行数百次推理。
- 对低流量应用而言，“scale-to-zero”模式不经济，建议常驻部署。

### 消融实验结果

#### （1）QLoRA 的“内存-能源悖论”（Table 2）
尽管QLoRA将VRAM占用减少约60%，但其训练能耗反而大幅上升：

| 模型 | 方法 | Median Energy (kWh) | 相对增幅 |
|------|------|---------------------|----------|
| LLaMA-1B | LoRA-FP16 | 0.039 | 1.0× |
| LLaMA-1B | **QLoRA-INT4** | **0.251** | **↑6.4×** |
| Qwen-7B | QLoRA-INT4 | 0.563 | ↑2.3× |

> ❗ 发现：**QLoRA虽节省内存，却显著增加adaptation阶段的碳足迹**，挑战了“内存高效即部署高效”的普遍假设。

#### （2）量化保真度分析（Table 4 & 5）
- **LLaMA系列**：INT4下性能保留率达 **98.6%~100.4%**，方差稳定
- **Qwen系列**：部分任务出现退化，如Qwen-3B在Chat任务上Helpfulness评分从7.46降至6.75（↓9.5%），且标准差上升45.6%

> 表明：**量化稳定性具有architecture-dependent特性**，不能一概而论。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **<2B参数模型在经济性和生态效率上全面优于大模型**
   - 在老旧硬件（Tesla T4）上构成明确的“效率前沿”
   - 更快实现ROI（Nbreak最小仅14次请求）
   - 更高IPW与psys，更适合资源受限场景

2. ✅ **INT4量化是高效的“硬件乘数”**
   - 推理吞吐提升1.8–1.9×，能耗降低57–61%
   - 性能保留接近无损（多数>99%），尤其对LLaMA架构友好

3. ⚠️ **QLoRA存在严重的“适应性能源异常”**
   - 尽管降低显存，但fine-tuning能耗最高可达普通LoRA的 **7.2×**
   - 揭示“memory efficiency ≠ energy/carbon efficiency”

4. ⚠️ **Cold-Start Tax过高使serverless部署不可行**
   - 加载能耗远高于推理，不适合频繁启停的函数计算场景

5. 🔄 **小型模型更适合边缘与本地化部署**
   - 功耗稳定在~35W，支持fanless部署，提高可靠性

### 局限性
- 实验集中在 **Tesla T4** 和 **low-batch** 场景，结果可能不适用于Hopper/Ampere高端GPU或大规模云服务
- 仅评估 **LLaMA 和 Qwen** 两个家族，未覆盖更多架构（如Mistral、Phi等）
- 能耗测量基于GPU telemetry，未包含CPU、内存、网络等系统级开销
- 经济参数（如API价格、碳强度因子）为当前估算值，随市场变化会影响绝对阈值

### 未来工作方向
- 扩展至更多模型架构与任务类型（如代码生成、数学推理）
- 支持现代GPU（如H100）与高并发场景下的评估
- 开发完整的端到端生命周期碳核算工具链
- 探索轻量级PEFT方法以缓解QLoRA的高能耗问题
- 构建公开可复现的 **EDGE-EVAL Benchmark Hub**

> GitHub地址：[https://github.com/Abdullah4152/EDGE-EVAL](https://github.com/Abdullah4152/EDGE-EVAL)

---

## 总结一句话
> **在现实工业条件下，小型化（<2B）LLM结合INT4量化可在经济性、能效和硬件利用率上全面超越大型模型；而流行的QLoRA技术虽省显存却不节能，揭示出“部署效率”需重新定义。**

</details>

---

### 4. [FEPLB: Exploiting Copy Engines for Nearly Free MoE Load Balancing in Distributed Training](https://arxiv.org/abs/2604.19654)

**Authors**: Shuyao Qi, Haoyuan Liu, Shizhen Zhao  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.19654v1  

#### Abstract
Fine-grained, per-micro-batch load balancing is essential for efficient Mixture-of-Experts (MoE) training, yet every prior dynamic scheduling scheme pays for it with extra communication that is hard to hide. Especially on modern bulk-transfer backends such as DeepEP. We make a simple but consequenti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FEPLB: Exploiting Copy Engines for Nearly Free MoE Load Balancing in Distributed Training

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大规模 **Mixture-of-Experts (MoE)** 模型的分布式训练中，由于路由机制（router）动态分配 token 到不同专家（expert），导致各 GPU 上的计算负载不均衡（load imbalance）。这种不平衡会导致严重的 **straggler 问题**——即部分设备成为瓶颈，拖慢整体训练速度。

传统解决方案如辅助平衡损失（auxiliary balancing loss）会限制模型表达能力，而动态调度方案（如 FasterMoE、Tutel）则引入额外通信开销或降低 GPU 利用率，尤其在现代高性能通信后端（如 DeepEP）下难以隐藏其代价。

---

### 🚀 提出的新方法与创新思路

FEPLB 提出了一种**资源正交的动态并行维度**（orthogonal dynamic parallelism），利用 NVIDIA Hopper 架构中的 **NVLink Copy Engine (CE)** 和 CPU 资源，在不影响原有 EP（Expert Parallelism）和 PP（Pipeline Parallelism）的前提下实现近乎零成本的负载均衡。

#### 核心创新点：

1. **Two-Phase Dispatch（两阶段分发）**
   - **Phase 1（跨节点分发）**：通过标准 EP 后端（如 DeepEP）将静态专家的 token 正常路由到目标设备；同时将动态专家的 token 收集至本地 NVLink 域。
   - **Phase 2（节点内重平衡）**：使用 **NVLink Copy Engine** 在节点内部重新分配动态专家的 token 和权重，全程不消耗 SM（Streaming Multiprocessor）周期，真正实现“免费”通信。

2. **资源级正交性设计原则**
   - FEPLB 使用的硬件资源与现有并行策略完全解耦：
     | 维度 | 通信资源 | 计算资源 |
     |------|----------|---------|
     | EP / PP | RDMA / NVLink | GPU SMs |
     | **FEPLB** | **NVLink Copy Engine** | **CPU** |
   - 因此可无缝集成进现有系统，无需重构并行配置。

3. **CPU侧轻量级调度器**
   - 在 GPU 执行静态专家计算的同时，CPU 并发运行负载均衡算法，决定哪些动态专家需要迁移。
   - 采用贪心策略：从最繁忙的设备复制整个专家（含 token 和权重）到最空闲设备，避免拆分 batch 导致 GEMM 性能下降。

4. **保持 MoE 语义一致性**
   - 仅移动专家权重副本，每个 token 仍由原定 expert 处理，保证数学等价性，支持无辅助损失训练。

---

### 🔍 相比现有方法的优势

| 方法 | 主要缺点 | FEPLB 如何改进 |
|------|--------|---------------|
| **FasterMoE** | 需预测路由、复制“影子专家”，破坏 Grouped GEMM 连续性，且 pipelining 在 DeepEP 上增加通信体积 | 不依赖预测，反应式调度；使用 CE 实现 SM-free 通信，无额外通信开销 |
| **Triton Distributed** | 融合通信与计算，占用 SM 资源，降低计算效率 | 完全脱离 SM，通信在 Copy Engine 上独立执行 |
| **Tutel / SmartMoE** | 切换并行模式需重新划分权重，带来显著通信开销 | 仅在节点内移动少量动态专家，通信量小且路径专用 |

> ✅ **核心优势总结**：  
> - 实现了 **fine-grained per-micro-batch load balancing**（细粒度每微批次负载均衡）
> - 开销几乎为零（"nearly free"），不干扰主计算流
> - 与主流 MoE 通信库（如 DeepEP）兼容，易于部署

---

## 2. 核心实验方法和设置

### 📊 数据集与模型
- **模型**：基于 **GLM-5** 的 MoE 层简化版本（18层，保留原始 MoE 结构）
  - 128 个 routed experts
  - top-k 路由，**无辅助平衡损失**（no auxiliary loss）
- **任务**：模拟真实 MoE 分布式训练场景下的前向/反向传播性能

### 💻 硬件环境
- **GPU**：NVIDIA H100 SXM5（80GB HBM3）
- **互联**：
  - 节点内：NVLink 4.0（900 GB/s 双向）
  - 节点间：400 Gbps InfiniBand
- **平台规模**：最多使用 16 张 H100 GPU

### ⚙️ 软件框架
- 基线：Megatron-LM + **DeepEP**（高效 EP 通信库）+ Transformer Engine（混合精度）+ cuBLAS Grouped GEMM
- FEPLB 在此基础上叠加实现，共享所有底层 kernel

### 🧪 实验配置（PP/EP 组合）
| 配置 | PP | EP | GPU 数量 | 每设备专家数 |
|-----|----|----|----------|-------------|
| A   | 4  | 2  | 8        | 64          |
| B   | 4  | 4  | 16       | 32          |
| C   | 2  | 8  | 16       | 16          |

### 📈 评估指标
| 指标 | 定义 | 意义 |
|------|------|------|
| **Token Straggler** | $\max_d(T_d) - \bar{T}$ | 最大 token 数与平均值之差，反映 token 分布不均程度 |
| **GEMM Straggler** | $\max_d(G_d) - \bar{G}$ | 最慢设备的 GEMM 时间超出均值的部分，直接衡量等待时间浪费 |
| **Per-layer Execution Time** | MoE 层前后向耗时（ms） | 端到端性能指标 |
| **EP Communication Overhead** | Dispatch & Combine 阶段耗时变化 | 检验是否影响主通信流程 |

### 🆚 基线方法对比
1. **Before LB**：标准 EP，无任何负载均衡
2. **FasterMoE**（pipe=1 和 pipe=2）：带 shadow expert 的复制机制
3. **Triton Distributed**：融合通信与计算的 TP-MoE 方案
4. **Tutel**：自适应切换 EP/DP 模式的调度器
5. **FEPLB**（本文方法）

> 注：作者对 FasterMoE 进行了优化重实现，使用 NVLink CE 和 DeepEP，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📉 关键性能数据汇总

#### 表1：不同配置下的 **Token Straggler Reduction**

| PP/EP | Before LB | FasterMoE | FEPLB | FEPLB 减少比例 |
|-------|-----------|-----------|--------|----------------|
| 4/2   | 2,278     | 1,014 (-55%) | 1,107 (-51%) | ↓51% |
| 4/4   | 4,649     | 2,471 (-47%) | 1,697 (-63%) | ↓63% |
| 2/8   | 6,666     | 4,036 (-39%) | 2,021 (-70%) | ↓70% |

> ✅ **FEPLB 在高 EP 场景下优势显著，最高减少 70% token straggler**

#### 表2：**GEMM Straggler Reduction**（单位：ms）

| PP/EP | Before LB | FasterMoE | FEPLB | FEPLB 减少比例 |
|-------|-----------|-----------|--------|----------------|
| 4/2   | 0.316     | 0.170 (-46%) | 0.157 (-50%) | ↓50% |
| 4/4   | 0.652     | 0.380 (-42%) | 0.247 (-62%) | ↓62% |
| 2/8   | 1.110     | 0.625 (-44%) | 0.352 (-68%) | ↓68% |

> ✅ **GEMM 等待时间减少达 68%，有效提升 GPU 利用率**

#### 表3：每层执行时间（ms）——以 PP=2, EP=8 为例

| 方法 | Forward | Backward |
|------|--------|---------|
| Before LB | 6.9 | 12.5 |
| FasterMoE | 6.3 | 11.1 |
| **FEPLB** | **6.0** | **10.6** |

> ✅ **前向提速 13%，反向提速 15%**

---

### 🔬 与基线方法对比结果

| 对比项 | FEPLB vs Baselines |
|--------|--------------------|
| **vs FasterMoE** | - 在 EP=8 时，token straggler 低 **2倍**（2,021 vs 4,036）<br>- GEMM straggler 低 **1.8倍**<br>- 更重要的是：**不依赖预测，适应性强** |
| **vs Triton Distributed** | - 前向快 **2.3–3.3x**（因后者占用 SM）<br>- 通信融合反而拖累性能 |
| **vs Tutel** | - 前向相当，但反向少 15–16% 开销<br>- 不改变 EP 拓扑，更稳定 |

#### 图：EP 通信开销对比（Figure 4）
- **FasterMoE (pipe=2)**：因分阶段通信，在 DeepEP 上增加高达 **46.8% dispatch 开销**
- **FEPLB**：通信开销 **<1%**，几乎不可测

> ✅ **验证了“正交性”设计的有效性：不干扰主通信路径**

---

### 🔍 消融实验结果（Sensitivity to `dyn`）

参数 `dyn` 控制每设备可用于动态迁移的专家数量（如 dyn=4 表示 4 个动态专家，其余为静态）。

| dyn | Token Straggler Reduction（EP=8） |
|-----|-------------------------------|
| 2   | ~65%                          |
| 4   | ~68%                          |
| 8   | ~70%                          |

> ✅ 即使 `dyn=2` 也能取得大部分收益，`dyn=4` 是性价比最优选择，进一步增加收益递减。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Copy Engine 是被忽视的宝贵资源**
   - NVLink Copy Engine 可提供高达 900 GB/s 的 SM-free 通信带宽，非常适合用于细粒度负载均衡。
   - 当前多数 MoE 系统未充分利用这一硬件特性。

2. **动态负载均衡可以作为一个新的并行维度**
   - FEPLB 成功构建了一个与 EP、PP 正交的新维度，实现了“插入即用”的动态调度能力。
   - 设计哲学：“**use idle resources, don’t compete**” —— 使用闲置资源，而非争夺已有资源。

3. **高 EP 场景下 FEPLB 优势愈发明显**
   - 随着 EP degree 增加，路由不确定性上升，FasterMoE 等预测方法失效，而 FEPLB 的反应式调度更具鲁棒性。

4. **无需牺牲模型质量即可实现高效均衡**
   - 支持 **完全去除 auxiliary balancing loss**，释放模型表达力，同时提升训练效率。

---

### ⚠️ 方法的局限性

1. **仅限于节点内平衡（intra-node only）**
   - 当前受限于 NVLink 拓扑（通常单节点 8 卡），无法跨节点 rebalance。
   - 但在 GB200 NVL72 等全连接 SuperPod 架构中可扩展至全局。

2. **整专家迁移限制了调度粒度**
   - 必须迁移整个 expert，不能按 token 拆分，因此在低 EP（每设备专家多）时调节不够精细。
   - 但这是为了保护 Grouped GEMM 性能所作的必要妥协。

3. **依赖特定硬件架构（Hopper + NVLink CE）**
   - 不适用于旧代 GPU 或缺乏 Copy Engine 的平台。

---

### 🔮 未来工作方向

1. **扩展至跨节点 Copy Engine 调度**
   - 在未来支持跨节点 NVLink 全互连架构（如 GB200）中，将 Phase 2 扩展至整个 EP 组。

2. **结合预测机制进行 hybrid 调度**
   - 将历史路由统计用于指导 CPU 调度器，提前预加载可能热点的专家。

3. **支持 token-level partial migration**
   - 探索在满足 GEMM 性能阈值前提下，允许部分 token 拆分迁移，提高灵活性。

4. **推广至其他条件计算场景**
   - 如稀疏 Attention、Dynamic Networks 等同样存在动态负载问题的模型结构。

---

## ✅ 总结一句话

> **FEPLB 创造性地利用 NVIDIA Hopper 架构中的 NVLink Copy Engine 和 CPU 资源，在不干扰现有 EP/PP 并行体系的前提下，实现了近乎零成本的 MoE 动态负载均衡，显著降低了 token 和 GEMM straggler（最高达 70%），并在高 EP 场景下大幅超越 FasterMoE 等主流方案，展示了“用闲置硬件解决动态开销”的全新设计范式。**

</details>

---

### 5. [DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](https://arxiv.org/abs/2604.19351)

**Authors**: Jinyu Guo, Zhihan Zhang, Yutong Li, Jiehui Xie, Md. Tamim Iqbal, Dongshen Han, Lik-Hang Lee, Sung-Ho Bae, Jie Zou, Yang Yang, Chaoning Zhang  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.19351v1  

#### Abstract
The quadratic computational complexity of the standard attention mechanism constitutes a fundamental bottleneck for large language models in long-context inference. While existing KV cache compression methods alleviate memory pressure, they often sacrifice generation quality and fail to address the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **标准 Attention 机制的二次计算复杂度** $O(N^2)$ 构成了大语言模型（LLM）在长上下文推理中的根本瓶颈。
- 现有的 KV Cache 压缩方法（如量化、选择性驱逐、结构化共享）虽然缓解了内存压力，但存在以下缺陷：
  - **量化方法**（如 KIVI, Atom）：低比特位宽导致生成质量显著下降，且需反量化带来额外开销。
  - **选择性驱逐策略**（如 H2O, SnapKV）：永久丢失历史信息，损害长距离依赖建模能力。
  - **结构化共享技术**（如 GQA）：忽略不同注意力头或层之间的异质性，缺乏数据驱动适应性。
- 更重要的是，这些方法**未改变高精度浮点运算的底层计算范式**，无法解决计算效率的根本瓶颈。

### 提出了什么新方法或新思路
本文提出 **DASH-KV**（Deep Asymmetric KV Cache Hashing），首次将深度哈希检索技术系统性地集成到 Transformer 的 Attention 机制中，重构 KV Cache 的计算范式。其核心思想是：
> 将传统的高维浮点向量相似度计算（dot product）转化为 **汉明空间（Hamming Space）中的二进制码匹配**，从而用高效的位运算替代昂贵的矩阵乘法。

#### 核心创新模块：
1. **Attention-Oriented Asymmetric Deep Hashing（非对称深度哈希）**
   - 针对 Query 和 Key 在语义动态性和重用特性上的差异，设计了不同的编码策略：
     - **Query Encoder**：采用轻量级 MLP 进行深度哈希，以最大化语义保真度，确保高精度检索。
     - **Key Encoder**：使用直接线性投影进行快速编码和紧凑存储，适配大规模 KV 缓存的持久性需求。
   - 实现了“查询精检、键值快存”的差异化映射，兼顾效率与准确性。

2. **Dynamic Importance-Based Mixed-Precision Attention（动态混合精度注意力）**
   - 引入一个轻量级的重要性预测器，区分关键 token（如逻辑连接词、命名实体）与普通 token。
   - 对于关键 token，保留 Full-Precision Attention；对于普通 token，启用高速哈希注意力。
   - 支持细粒度、实例级别的精度-效率权衡控制。

3. **多视角校准机制（Cross-Head Consensus & Cross-Layer Momentum）**
   - 利用 Transformer 的多头和深层结构先验，提升哈希检索的判别力：
     - **跨头共识**：通过“多数投票”机制修正个别头的误判。
     - **跨层动量**：利用前一层的注意力分布作为先验，优先关注持续被注意的 Key。

---

### 相比现有方法的优势
| 维度 | DASH-KV | 现有方法（Quantization/Eviction/Structured Sharing） |
|------|--------|----------------------------------------------------|
| **计算复杂度** | 从 $O(N^2)$ 降至线性 $O(N)$ | 仍基于浮点运算，复杂度不变 |
| **信息保留** | 不丢弃任何 KV，仅跳过计算（masking） | 驱逐类方法永久丢失信息 |
| **精度损失** | 可控，通过残差补偿和混合精度缓解 | 量化误差大，尤其在 1–2bit 下严重退化 |
| **通用性** | 可插拔部署于主流 LLM，无需重新预训练 | 多数结构化方法需在训练阶段集成 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 所有实验均在 **LongBench** 基准上进行，涵盖六项代表性长文本任务：
  - **NarrativeQA**：单文档摘要理解
  - **HotpotQA**：多跳推理
  - **Qasper**：学术论文问答
  - **MultiNews**：多文档摘要
  - **GovReport**：政府报告摘要
  - **TriviaQA**：事实型问答

### 实验设置和评估指标
- **主干模型**：
  - `Qwen2-7B-Instruct`（7B）
  - `Llama-3.1-8B-Instruct`（8B）
  - `Qwen2.5-14B-Instruct`（14B）
- **训练配置**：
  - 训练序列长度：3k tokens
  - 推理序列长度：32k tokens（体现“短训长推”范式）
- **评估指标**：
  - 主要：各任务的 F1 / EM 分数，以及平均得分（Avg）
  - 效率：每 token 推理延迟（ms）、理论内存占用（基于 1-bit 存储模型）

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Full Attention** | Dense | 全精度注意力，性能上限基准 |
| **StreamingLLM** | Eviction | 利用 Attention Sink 实现无限长文本流式推理 |
| **H2O** | Eviction | 动态驱逐低贡献 KV 对，维持固定缓存大小 |
| **SnapKV** | Retrieval | 在观察窗口内识别重要提示特征以压缩 KV Cache |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | Method | NQ | HQ | QM | MN | GR | TQ | **Avg** |
|-------|--------|----|----|----|----|----|----|---------|
| Qwen2-7B | Full Attn | 25.13 | 44.04 | 46.13 | 15.42 | 18.06 | 83.47 | **38.71** |
| | DASH-KV | 24.65 | 44.50 | 45.34 | 15.40 | 19.13 | 83.33 | **38.73** ✅ |
| Llama-3.1-8B | Full Attn | 28.11 | 57.43 | 45.29 | 15.20 | 19.97 | 90.22 | **42.70** |
| | DASH-KV | 27.61 | 58.01 | 45.35 | 14.90 | 19.25 | 89.47 | **42.43** ✅ |
| Qwen2.5-14B | Full Attn | 28.20 | 61.98 | 45.52 | 14.19 | 16.78 | 86.91 | **42.26** |
| | DASH-KV | 29.52 | 61.83 | 45.79 | 14.44 | 18.06 | 87.88 | **42.92** ✅ |

> ✅ **DASH-KV 在所有三个模型上均达到甚至略微超越 Full Attention 的性能水平**

### 与基线方法的对比结果
- 相比 **H2O/SnapKV/StreamLLM** 等 SOTA 方法，DASH-KV 平均得分高出 **6–10 个百分点**。
- 在 **HotpotQA**（多跳推理）等依赖长程依赖的任务上，DASH-KV 显著优于驱逐类方法（如 H2O 得分仅为 ~16，而 DASH-KV 达到 ~58）。
- 表明 DASH-KV 成功避免了因信息永久丢失而导致的推理断裂问题。

### 消融实验结果（Ablation Studies）

#### （1）组件有效性分析（Table 2）
| Variant | Recall@100 | KL(Ph\|Pf) | Latency (ms) |
|--------|------------|-----------|-------------|
| DASH-KV-Naive (LSH) | 8.91 | 3.2000 | 28 |
| DASH-KV-Sym | 68.28 | 1.3200 | 38 |
| **DASH-KV (Ours)** | **86.06** | **0.4054** | **22** |

> 结论：非对称哈希 + 深度学习编码显著优于随机投影（LSH）和对称哈希，在召回率和分布对齐方面表现最佳，同时延迟最低。

#### （2）残差学习的有效性（Figure 3）
- **Pure Hash**（无残差补偿）：F1 下降明显
- **DASH-KV-MSE**（MSE 损失训练）：F1 为 10.92，低于 Full-Precision（11.31）
- **DASH-KV-Distill**（蒸馏训练）：F1 达 **11.44**，**超过全精度基线**
> 结论：基于 KL 散度的知识蒸馏目标更鲁棒，能有效泛化至长序列场景，避免 MSE 导致的尺度偏移问题。

#### （3）哈希码长度的影响（Figure 4）
- 随着哈希码长度增加（64 → 256 bits），准确率持续提升，但边际收益递减。
- 推荐使用 **128-bit** 编码，在精度与效率间取得良好平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **Attention 即检索**：将 Attention 视为大规模近似最近邻（ANN）检索任务是可行且高效的，为 LLM 加速提供了全新视角。
2. **非对称哈希至关重要**：Query 和 Key 的分布特性差异显著（见 Figure 9），统一编码会损害性能；分别优化编码器可实现最优协同。
3. **动态混合精度机制有效**：结合重要性预测与分层处理（强相关/中等相关/弱相关），可在不牺牲关键信息的前提下大幅提升效率。
4. **层敏感性呈 U 型曲线**（Figure 7）：
   - **浅层与深层**（Layer 0–4, 27–32）：对压缩极为敏感，替换后性能崩溃。
   - **中间层**（Layer 5–27）：具有高度冗余性，适合应用 DASH-KV。
   > 因此提出 **Sandwich Deployment Strategy**：首尾层保持 Full Attention，中间层使用 DASH-KV，兼顾稳定性与加速效果。

### 方法的局限性
- 当前实验使用 FP16 模拟 1-bit 哈希运算，尚未实现真正的硬件级位操作（bitwise XOR + POPCNT）。
- 报告的存储节省和速度提升基于理论 1-bit 打包模型，实际部署需定制 GPU kernel 或专用指令支持。
- 虽然性能媲美 Full Attention，但在极少数极端长序列（>128k）下的稳定性仍需进一步验证。

### 未来工作方向
- 开发针对低精度位运算优化的 **Custom GPU Kernels**，充分发挥 I/O 和计算优势。
- 探索 **端到端的硬件-算法联合设计**，例如利用 FPGA 或 ASIC 实现高效 Hamming 距离计算。
- 将 DASH-KV 思路扩展至 **多模态模型**（如 LVMs）和 **Agent Memory Systems** 中的长期记忆管理。

--- 

> 🔚 **总结一句话**：  
> DASH-KV 通过引入**非对称深度哈希 + 动态混合精度机制**，成功将 LLM 的 Attention 计算从浮点密集型转变为位运算主导型，在**几乎无损性能的前提下实现了线性推理加速**，打破了传统 KV Cache 优化中“效率-精度”不可兼得的僵局。

</details>

---

### 6. [LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](https://arxiv.org/abs/2604.18913)

**Authors**: He Cheng, Yifu Wu, Saksham Khatwani, Maya Kruse, Dmitriy Dligach, Timothy A. Miller, Majid Afshar, Yanjun Gao  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.18913v1  

#### Abstract
Knowledge graphs (KGs) are increasingly integrated with large language models (LLMs) to provide structured, verifiable reasoning. A core operation in this integration is multi-hop retrieval, yet existing systems struggle to balance efficiency, scalability, and interpretability. We introduce LogosKG,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的知识图谱（KG）多跳检索系统在**效率、可扩展性和可解释性**之间难以平衡，尤其是在处理大规模、高连通性的图谱时面临以下挑战：
- **计算开销大**：传统图遍历算法（如 BFS/DFS）的时间复杂度为 $O(|V| + |E|)$，随着跳数增加呈指数级增长。
- **内存瓶颈**：大型 KG（如 PubMedKG 含 8650 万条边）无法完全加载到内存中。
- **缺乏路径重建能力**：基于矩阵运算的方法虽高效，但常丢失边的溯源信息，导致推理过程不可解释。

这些问题严重限制了 KG 在与 LLM 集成中的应用，特别是在需要深度推理的医疗诊断等高风险领域。

---

### **提出了什么新方法或新思路**
作者提出 **LogosKG**，一个**硬件对齐、可扩展且可解释的多跳检索框架**，其核心思想是将符号化 KG 推理转化为硬件高效的稀疏矩阵操作，并通过系统级优化实现大规模部署。

#### **关键技术组件**：
1. **Symbolic KG Formulation with Sparse Matrices**
   - 将 KG 分解为三个稀疏关联矩阵：
     - `SUB`（主体矩阵）
     - `OBJ`（客体矩阵）
     - `REL`（关系矩阵）
   - 多跳检索被形式化为连续的稀疏矩阵乘法：  
     $$
     q^{(k)} = q^{(0)} \cdot (SUB \cdot OBJ)^k
     $$

2. **Degree-Aware Partitioning**
   - 图按实体出度进行均衡划分，避免高连接节点集中于单一分区造成负载不均。
   - 所有以同一主体开头的三元组保留在同一子图中，维持局部拓扑完整性。

3. **Cross-Graph Routing & On-Demand Caching**
   - 查询按分区路由，仅加载所需子图至缓存（LRU 策略），其余驻留磁盘。
   - 支持跨子图合并结果，实现全局一致性检索。

4. **Path Reconstruction**
   - 利用中间激活的三元组索引恢复完整推理路径，支持可解释的链式推理。

5. **Multi-Backend Support**
   - 实现 Numba、SciPy 和 Torch 三种后端，支持 CPU/GPU 统一执行。

---

### **相比现有方法的优势**

| 特性 | LogosKG | 其他系统（如 Neo4j, DGL, PyG） |
|------|--------|-------------------------------|
| **Matrix-Based** | ✅ 支持 | 部分支持 |
| **Scalability** | ✅ 可处理十亿级边图 | 多依赖分布式架构 |
| **Path Reconstruction** | ✅ 完整路径输出 | ❌ 或需额外处理 |
| **Single-Device Efficiency** | ✅ 单机即可运行 | ❌ 常需集群 |
| **Deterministic Retrieval** | ✅ 结果一致 | ⚠️ 可能因并行策略不同而异 |

> ✅ 表示具备该特性，❌ 表示不具备，⚠️ 表示不稳定

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **UMLS**：医学本体，40.7K 节点，340 万边
- **PubMedKG (PKG)**：文献引用网络，5440 万节点，8650 万边
- **PrimeKG**：整合 20 个生物医学资源，约 400 万关系

### **临床任务相关数据集**
- **ProbSum**：脱敏电子病历笔记，用于实体抽取与诊断预测
- **DDXPlus**：症状-诊断配对数据集，用于评估 LLM-KG 交互效果

### **实验设置**
- **硬件环境**：双 AMD EPYC 9454 CPU（192 线程），256GB RAM，2×NVIDIA H100 GPU（共 94GB VRAM）
- **查询方式**：从 ProbSum 中提取实体作为初始查询集，执行 1~5 跳检索
- **缓存配置**：固定大小缓存（默认 n=16），采用 LRU 替换策略
- **批处理优化**：按子图需求重排序查询以提升缓存命中率

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Query Time (QT)** | 平均每查询耗时（ms） |
| **Timeout Rate (TR)** | 超过时限未完成的比例（时限随跳数递增） |
| **Jaccard Similarity** | 检索结果与基准方法的集合重合度（衡量准确性） |
| **Precision / Recall / F1** | 下游诊断任务中预测与金标准的匹配程度 |
| **PDSQI-9** | 医疗诊断质量九维评分体系（含准确性、简洁性、逻辑性等） |

### **基线方法对比**
涵盖五大类主流系统：
| 类型 | 代表方法 |
|------|---------|
| 图数据库 | Neo4j, TigerGraph |
| 矩阵计算库 | GraphBLAS |
| 图分析工具 | NetworkX, igraph, graph-tool, SNAP |
| GPU 框架 | cuGraph, DGL, PyG |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表 2：多跳检索效率对比（UMLS + ProbSum）**

| 方法 | Hop 1 QT/ms | Hop 2 QT/ms | Hop 3 QT/ms | Hop 4 QT/ms | Hop 5 QT/ms |
|------|-------------|-------------|-------------|-------------|-------------|
| **LogosKG (Torch-GPU)** | **6.00** | **14.40** | **43.07** | **77.73** | **101.05** |
| LogosKG (Numba) | 12.28 | 28.72 | 77.65 | 140.07 | 204.25 |
| PyG | 249.66 | 271.46 | 365.90 | 646.96 | 735.34 |
| Neo4j | >923.86 | >1946.43 | >5739.02 | >8000 | >10000 |
| NetworkX | 0.21 | 5.47 | 93.92 | 621.95 | 1511.28 |

> 注：超时阈值分别为 2000/4000/6000/8000/10000 ms。**加粗为最优**，下划线为次优。

- **LogosKG Torch-GPU 在所有跳数上均最快**，且无超时。
- 传统工具（如 NetworkX）浅层快但深层崩溃；GPU 框架（如 DGL）未能发挥优势。

#### **表 3：LogosKG-Large 在 PKG 上的可扩展性测试**

| 设置 | QT/ms | Loads | Evicts |
|------|-------|-------|--------|
| Batch Size=1 | 100912.09 | 12 | 0 |
| Batch Size=50 | 1499.39 | 16 | 0 |
| Cache Size=1 | 441870.19 | 3010 | 3009 |
| Cache Size=16 | 4037.69 | 16 | 0 |
| Backend=Numba | 4143.16 | 16 | 0 |
| Backend=Torch-GPU | 6409.32 | 16 | 0 |

- **批量处理显著降低延迟**（>67倍加速）
- **缓存大小对性能影响巨大**：小缓存引发频繁磁盘 I/O，延迟飙升百倍
- **Numba 和 SciPy 后端表现最佳**

---

### **消融实验结果**

#### **路径重建与检索保真度验证（Appendix A.1）**
- 所有版本 LogosKG 与其他基线方法的 **Jaccard 相似度均为 1.00**，证明其检索结果完全一致、确定性高。

#### **缓存与批处理敏感性分析（Table 3）**
- 缓存命中率从 <5%（size=1）提升至 100%（size=16），直接带来两个数量级的速度提升。
- 批量越大，缓存复用越高，I/O 开销越低。

---

## **4. 关键结论和发现**

### **主要发现**
1. **LogosKG 实现了前所未有的单设备多跳检索效率**：
   - 在十亿边级图上实现亚秒级 5 跳检索。
   - 相比 CPU/GPU 基线平均提速 10–100 倍，且无精度损失。

2. **KG 拓扑结构深刻影响 LLM 推理行为**：
   - 在 **DDXPlus + PrimeKG** 场景中，随着跳数增加（k=1→5），**F1 分数持续上升**，表明远距离语义关联对诊断至关重要。
   - 而在 UMLS 上收益较小，因其多数概念已在 1–2 跳内可达（见 Figure 2）。

3. **两轮 KG-LLM 交互机制有效改善诊断质量**：
   - **Round 1（Filtering）**：利用 KG 过滤掉 LLM 的幻觉诊断，提高 precision。
   - **Round 2（Enhancement）**：提供长尾候选疾病供 LLM 选择，提升 recall。
   - 最终输出结合两者，在 **PDSQI-9 评分中多个维度显著优于 baseline**，尤其在 *accuracy extractive*, *succinctness*, *synthesis* 上。

4. **KG 增强更适合零样本/少样本场景**：
   - 对 fine-tuned 模型，KG filtering 可能误删已学知识（“knowledge mismatch”），反而损害性能。
   - 而对 zero-shot 模型，KG 提供了不可或缺的外部知识来源。

---

### **方法的局限性**
1. **依赖 KG 完备性**：
   - 若真实病因不在 KG 中，则无法检索到，限制了上限。
2. **LLM 再选择能力受限**：
   - Round 2 依赖 LLM 正确识别相关候选，若模型本身推理弱，则增强无效。
3. **高跳检索产生大量噪声**：
   - 如在 PrimeKG 中 5 跳可返回超 4 万个实体，需后续排序或剪枝。
4. **当前聚焦检索，非端到端训练**：
   - 不直接参与模型参数更新，属于外部模块。

---

### **未来工作方向**
1. **集成更智能的排序机制**：
   - 结合 SapBERT 等语义相似度模型进行 Top-N 重排（Appendix A.6 显示 Top-N 比阈值法更稳健）。
2. **探索 KG-augmented fine-tuning 新范式**：
   - 当前尝试（Table 7）显示简单拼接检索结果会引入过多噪声，未来应研究迭代式、agent-based 的精炼流程。
3. **扩展至动态图与时序推理**：
   - 支持随时间演化的医学知识更新。
4. **构建通用 KG-LLM 推理引擎**：
   - 将 LogosKG 作为底层检索 backbone，支撑更复杂的 multi-agent 医疗诊断系统。

---

> 🔗 **开源信息**：  
> - GitHub: [https://github.com/LARK-NLP-Lab/LogosKG](https://github.com/LARK-NLP-Lab/LogosKG)  
> - Online Demo: [https://lark-nlp-lab-logoskg.hf.space/](https://lark-nlp-lab-logoskg.hf.space/)

</details>

---

### 7. [POLAR-PIC: A Holistic Framework for Matrixized PIC with Co-Designed Compute, Layout, and Communication](https://arxiv.org/abs/2604.19337)

**Authors**: Yizhuo Rao, Xingjian Cui, Shangzhi Pang, Jiabin Xie, Guangnan Feng, Jinhui Wei, Ziyan Zhang, Languang Gao, Zhenyu Wang, Zhiguang Chen, Yutong Lu  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.19337v1  

#### Abstract
Particle-in-Cell (PIC) simulations are fundamental to plasma physics but often suffer from limited scalability due to particle-grid interaction bottlenecks and particle redistribution costs. Specifically, the particle-grid interaction computations have not taken full advantage of the emerging Matrix...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：POLAR-PIC: A Holistic Framework for Matrixized PIC with Co-Designed Compute, Layout, and Communication

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统的 **Particle-in-Cell (PIC)** 模拟在大规模并行计算中面临三大瓶颈：
- **粒子-网格交互**（Particle-Grid Interaction）效率低下，尤其是 **Field Interpolation** 和 **Deposition** 阶段；
- 粒子运动导致内存访问不规则，破坏数据局部性；
- 粒子重分布（Redistribution）采用 **Bulk-Synchronous Parallel (BSP)** 模型，引入全局同步开销，限制扩展性。

尽管已有框架如 **WarpX** 和 **Matrix-PIC** 在 GPU 或矩阵架构上进行了优化，但仍存在以下不足：
- Matrix-PIC 虽将 Deposition 改写为 MPU 友好的外积形式，但未解决 Field Interpolation 的内积特性与 MPU 外积机制之间的**结构性失配**；
- 现有排序策略（如周期性全局排序）带来高开销，且无法维持物理连续性；
- 通信与计算无法有效重叠，导致重分布成为性能瓶颈。

---

### 提出了什么新方法或新思路
作者提出 **POLAR-PIC**，一个面向下一代矩阵化 HPC 架构的协同设计框架，从 **Compute、Layout、Communication** 三个层面进行一体化优化：

#### （1）**Matrixized Field Interpolation（计算层）**
- 将原本是 **inner-product reduction** 的 Field Interpolation 转换为 **outer-product accumulation** 形式；
- 通过 **cell-centric batching** 策略，将多个粒子的权重向量堆叠成矩阵，利用 **Matrix Outer-Product Accumulate (MOPA)** 指令实现高效并行计算；
- 成功弥合了 Interpolation 与 MPU 硬件原语之间的结构性鸿沟。

#### （2）**Sort-on-Write (SoW) 物理有序布局维护（数据层）**
- 引入 **inline Sort-on-Write (SoW)** 机制，在粒子更新写回时动态分类：
  - 居住粒子（Resident）写入有序区域（Ordered Region），保持物理连续；
  - 迁移粒子（Migrating）追加到无序尾部（Disordered Tail）；
- 利用双缓冲和线程局部缓存避免频繁分配；
- 实现了 **O(1) 分摊开销** 下的持续物理连续性，保障 MPU 流水线饱和。

#### （3）**RMA-overlapped Communication（通信层）**
- 设计细粒度通信重叠策略：
  - 在 Field Interpolation 写回阶段即完成迁移粒子的预打包（pre-packing）；
  - 使用 **UNR (Unified Notifiable RMA)** 库发起非阻塞 one-sided 传输；
  - 将通信隐藏在 Deposition 阶段之后，实现高比例重叠；
- 避免了传统 BSP 模型中的显式同步代价。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **性能** | 显著加速整个粒子处理阶段（最高达 10.9×）； |
| **可扩展性** | 弱扩展效率在超 200 万核下仍达 67.5%； |
| **稳定性** | 减少长尾延迟和性能抖动，适合动态场景； |
| **硬件利用率** | 达到理论峰值的 13.2%，远高于基线； |
| **通用性** | 建立在 WarpX 基础上，兼容性强，可移植至其他平台。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Uniform Plasma**：均匀等离子体微基准测试，用于评估不同粒子密度（PPC）和热速度（uth）下的性能；
- **Laser-Ion Acceleration (LIA)**：真实世界激光离子加速模拟，具有强非均匀性和高迁移率，验证复杂场景下的鲁棒性。

---

### 实验设置
| 项目 | 描述 |
|------|------|
| **硬件平台** | LS 先导系统，每节点配备两颗 **LX2 CPU**，支持 VPU 和 MPU（8×8 MOPA），集成 RDMA 支持； |
| **软件环境** | 基于 **WarpX v24.07** 开发，编译器为 Clang/Flang (-O3 -flto)，使用 OpenMPI 和定制 UNR 库； |
| **并行模型** | MPI+OpenMP 混合并行，每 NUMA 域一个 MPI rank，每个 rank 启动 32 个 OpenMP 线程； |
| **弱扩展规模** | 最高达 4,096 节点（>2 million cores）； |
| **关闭 I/O** | 所有实验禁用 I/O 以聚焦计算性能。 |

---

### 评估指标
| 指标 | 定义 |
|------|------|
| `T_particle` | 粒子相总时间 = `T_interpolation + T_deposit + T_redistribute` |
| `T_steps` | 平均每步粒子相耗时 |
| **Speedup** | 相对于 WarpX 基线的加速比 |
| **PPS** | Particles per Second，衡量吞吐量 |
| **CPP** | Cycles per Particle，归一化到 1.3GHz |
| **Overlap Ratio** | 通信被掩盖的比例：`1 - (T_overlap_issue_wait / T_baseline)` |
| **Peak Efficiency (%)** | `(Particle FLOPs / (T_steps × P_theoretical)) × 100%` |
| **FOM_node** | Node-level Figure of Merit，综合考虑网格和粒子数 |

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **WarpX-Native (Baseline)** | 原生 WarpX 流水线，使用 VPU 自动向量化，无特殊排序或通信重叠； |
| **Matrix-PIC** | 当前最优矩阵化方案，仅对 Deposition 进行外积重构，依赖逻辑索引排序； |
| **POLAR-PIC (Ours)** | 本文提出的完整协同设计框架。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **总体加速比**
| 场景 | 相比 WarpX 加速比 | 相比 Matrix-PIC 加速比 |
|------|------------------|-----------------------|
| **Uniform Plasma** | **10.9×** | **4.7×** |
| **Laser-Ion Acceleration (LIA)** | **4.4×** | **3.8×** |

> 即使在高迁移强度下（uth=0.2），仍能维持 **7.3×** 的稳定加速。

---

#### ✅ **各阶段性能分解（消融实验）**
| 模块 | 加速比 | 说明 |
|------|--------|------|
| **Interpolation** | **8.0×** | 得益于外积重构 + SoW 数据供给 |
| **Deposition** | **13.2×** | 物理连续布局极大缓解原子冲突 |
| **Communication Overlap** | **99.1% 重叠率** | UNR 实现近乎完全的通信隐藏 |

> 表明三项技术协同作用显著，缺一不可。

---

#### ✅ **跨平台峰值效率对比**
| 平台 | 方法 | Peak Efficiency (%) |
|------|------|--------------------|
| **LX2 CPU** | WarpX (Native) | 1.0% |
| **LX2 CPU** | Matrix-PIC | 5.5% |
| **LX2 CPU** | **POLAR-PIC (Ours)** | **13.2%** |
| **NVIDIA A800 GPU** | WarpX (CUDA) | 9.6% |

> POLAR-PIC 在 CPU 上达到的效率超过当前主流 GPU 实现（9.6%），体现其在矩阵化 CPU 架构上的巨大潜力。

---

#### ✅ **弱扩展性表现**
- 在 **4,096 节点（>2 million cores）** 规模下：
  - **POLAR-PIC** 弱扩展效率：**67.5%**
  - **WarpX 基线**：仅 **42.5%**

> 主要得益于通信重叠有效掩盖了大规模互联延迟。

---

#### ✅ **稳定性与长尾效应**
- 在 LIA 场景中：
  - Matrix-PIC 因频繁重建索引导致严重性能抖动（max-mean 差异大）；
  - POLAR-PIC 将长尾延迟降低 **34.6%~38.3%**，提供更可预测的时间到解能力。

---

## 4. 关键结论和发现

### 主要发现
1. **结构性失配可通过算法重构解决**  
   Field Interpolation 的 inner-product 本质可通过 batching + tensor stacking 转换为 MPU 友好的 outer-product，打破 Amdahl 定律瓶颈。

2. **物理连续性是 MPU 高效运行的前提**  
   单纯的逻辑排序（如 Matrix-PIC）不足以支撑高性能；**SoW 机制**实现了低开销、高稳定性的物理连续数据供给。

3. **通信重叠必须与数据路径融合**  
   传统后置扫描打包方式无法消除冗余遍历；**SoW + UNR** 的融合设计实现了真正的零额外打包开销。

4. **协同设计优于孤立优化**  
   Compute、Layout、Communication 三者相互依赖，只有整体协同才能释放最大性能。

---

### 方法的局限性
1. **目前仅针对 CPU 架构优化**  
   虽然可在其他平台移植（见 Artifact Appendix），但在 GPU 或 AI 加速器上需重新映射 MPU/VPU 指令；
2. **Field Solver 成为新瓶颈**  
   在粒子处理大幅加速后，Maxwell 求解器成为主导开销，需后续优化；
3. **依赖特定通信库（UNR）**  
   需要支持 notifiable RMA 的底层网络栈，通用性受限于硬件生态。

---

### 未来工作方向
1. **优化 Field Solver 通信路径**  
   将类似重叠思想应用于 Halo Exchange，进一步提升端到端扩展性；
2. **扩展至多物理场耦合场景**  
   如 QED（Quantum Electrodynamics）、自适应网格细化（AMR）等；
3. **支持动态负载均衡**  
   在非均匀网格和粒子分布下自动调整域分解；
4. **跨平台移植与泛化**  
   探索在 Apple M-series、国产加速芯片等架构上的适配路径。

---

> 🔚 **总结一句话**：  
> **POLAR-PIC 通过“算子重构 + 数据布局维护 + 通信重叠”三位一体的协同设计，首次实现了 PIC 模拟在矩阵化 CPU 架构上的全链路高效执行，为下一代 Exascale 科学计算提供了新的范式。**

</details>

---

### 8. [ReaLB: Real-Time Load Balancing for Multimodal MoE Inference](https://arxiv.org/abs/2604.19503)

**Authors**: Yingping Wang, Yi Wu, Xiangyu Wu, Junwei Cui, Weilin Cai, Zhijiang Guo, Jiayi Huang  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.19503v1  

#### Abstract
Mixture-of-Experts (MoE) architectures are widely used in modern large language models and multimodal models. However, inference efficiency is often limited by highly dynamic and skewed expert workloads across different modalities. During the prefill stage with large batch sizes, vision tokens frequ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ReaLB: Real-Time Load Balancing for Multimodal MoE Inference 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在多模态 Mixture-of-Experts (MMoE) 模型的推理过程中，由于不同模态（如文本与视觉）输入的动态性和不均衡性，导致专家负载高度倾斜。尤其是在 **prefill 阶段**，高分辨率图像生成大量 vision tokens，集中路由到少数设备上，在 Expert Parallelism (EP) 架构下引发严重的 **device-level 负载不平衡**，形成计算瓶颈（stragglers），显著降低系统吞吐量。

传统基于历史预测的负载均衡方法（如 EPLB）难以应对 MMoE 推理中快速变化的路由模式，且引入额外通信开销和内存占用。

---

### **提出了什么新方法或新思路**

作者提出 **ReaLB**（**Real-time Load Balancing**），一种面向多模态 MoE 推理的实时负载均衡方法，其核心思想是：

- **动态调整 MoE 专家的计算精度**：在运行时，对负载过重且以 vision-heavy 专家为主的 EP rank，将其 MoE GEMM 运算切换为低精度（如 FP4），利用硬件加速（如 FP4 Tensor Cores）提升执行效率。
- **零调度开销设计**：通过流水线编排（pipeline orchestration），将精度转换和调度决策完全隐藏在 All-to-All 通信阶段内，**不增加关键路径延迟**。
- **无需冗余专家或额外内存**：所有专家仍以高精度（如 BF16）存储权重，仅在执行前在线量化至目标精度，避免多份权重副本带来的内存膨胀。

---

### **相比现有方法的优势**

| 特性 | ReaLB | EPLB / FasterMoE |
|------|--------|------------------|
| **响应速度** | 实时感知当前路由，即时调整 | 依赖滑动窗口历史统计，滞后于实际负载 |
| **通信开销** | 无专家迁移，通信量不变 | 需跨设备复制/迁移专家，通信成本高 |
| **内存开销** | 无冗余专家，仅临时量化 | 存储冗余专家副本，内存占用上升 |
| **适用场景** | 特别适合高动态、短生命周期的推理任务 | 更适用于训练等具有时间局部性的场景 |
| **实现复杂度** | 轻量级调度 + 精度切换 | 复杂的专家放置与迁移逻辑 |

> ✅ **核心优势**：ReaLB 在几乎零额外开销的前提下，实现了对瞬时负载热点的精准加速。

---

## 2. 核心实验方法和设置

### **使用的模型与数据集**

#### **模型**
- **Kimi-VL-A3B-Instruct**（modality-fused MMoE）
- **Qwen3-VL-30B-A3B-Instruct**（modality-fused）
- **ERNIE-4.5-VL-27B-A3B**（modality-isolated MMoE）

#### **数据集（通过 `lmms-eval` 测试）**
| 数据集 | 任务特点 |
|--------|----------|
| **MMMU** | 多图像理解与推理，跨模态整合要求高 |
| **RealWorldQA** | 文本为主，少量视觉内容 |
| **AI2D** | 单图图表理解 |
| **InfoVQA / TextVQA** | 图像中的文本识别与问答 |
| **MathVista** | 视觉数学推理 |
| **MMBench** | 综合感知、常识与推理能力测试 |

这些数据集覆盖了从“文本主导”到“视觉密集”的多种多模态场景。

---

### **实验设置**

- **硬件平台**：8×NVIDIA RTX 5090（32GB），采用 Expert Parallelism (EP=8)
- **软件框架**：基于 **vLLM v0.13.0** 实现，集成 LLM Compressor 和 FlashInfer 的 NVFP4 GEMM 内核
- **并行策略**：Data Parallelism + Expert Parallelism
- **批处理机制**：连续批处理（continuous batching），混合 prefill 与 decode 请求
- **触发条件**：当全局 token 数超过 `batch_threshold=2048` 时启用 ReaLB（进入 compute-bound 状态）

---

### **评估指标**

| 指标类别 | 具体指标 |
|---------|--------|
| **效率** | MoE 层延迟（CUDA events）、端到端吞吐量（tokens/s） |
| **准确性** | 多个 vision-language benchmark 上的准确率（vs. BF16 baseline） |
| **公平比较** | 控制变量下的 speedup 与 accuracy trade-off 分析 |

---

### **基线方法对比**

| 方法 | 描述 |
|------|------|
| **Baseline** | 所有 MoE 层使用 W16A16 GEMM，无负载均衡 |
| **FP4-All** | 所有 MoE 层统一使用 W4A4 FP4 GEMM |
| **EPLB** | 基于历史负载预测进行专家复制（window=100, interval=100, 8 redundant experts） |
| **Async EPLB** | 异步执行专家迁移以减少阻塞 |
| **ReaLB-seq** | ReaLB 的串行版本（关闭流水线重叠，用于验证开销隐藏效果） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **MoE 层级加速**
- 在典型配置下（如 Qwen-VL, batch=8K, 2 images）：
  - **平均 MoE 层速度提升达 1.29×**
  - 最高可达 **1.36×**（随图像数量增加而略有上升，见 Table 4）

#### ✅ **端到端吞吐量提升**
- 对 5000 个多模态请求的端到端测试显示：
  - **吞吐量最高提升 1.53×**（Kimi-VL）
  - 平均提升范围：**1.06× ~ 1.53×**

> 💡 注：实验中使用 H20 + NVLink 的通信延迟替代 RTX 5090 的 PCIe 带宽，以更真实反映服务器级部署下的负载失衡影响。

---

### **与基线方法的对比结果**

| 方法 | MoE 层 Speedup | E2E Throughput | Accuracy Loss |
|------|----------------|----------------|---------------|
| **Baseline** | 1.00× | 1.00× | 0.0 pt |
| **EPLB** | ≈0.9–1.0× | ≈0.91–0.96× | 0.0 pt |
| **FP4-All** | ~1.40× | ~1.39× | **↓ up to 8.19 pts** |
| **ReaLB** | **1.29×** | **1.53×** | **< 1.2 pts** |

- **EPLB 表现不佳**：因路由高度动态，历史预测失效，频繁迁移反而引入负优化；
- **FP4-All 效率高但精度损失严重**：尤其在 MMMU、DynaMath 等数值敏感任务上；
- **ReaLB 实现最佳权衡**：接近 FP4-All 的性能，但精度损失极小。

---

### **消融实验结果**

#### 🔹 **ReaLB-seq vs. ReaLB**
- **ReaLB-seq**（关闭流水线重叠）性能增益远低于完整版 ReaLB；
- 证明了 **pipeline overlapping 是隐藏调度与量化开销的关键**。

#### 🔹 **不同模态阈值 $M_d$ 的敏感性分析**（Table 2）

| $M_d$ 设置 | E2E Speedup | Avg Acc Loss |
|-----------|------------|-------------|
| 0.0（无模态控制） | 1.29–1.30× | ↓2.40 pts |
| **0.7（推荐）** | **1.29×** | **↓0.3–1.1 pts** |
| 0.9 | 1.28–1.29× | ↓0.2–0.66 pts |

- 结论：适度提高 $M_d$ 可更好保护 text-heavy 设备的精度，同时保持高效。

#### 🔹 **Batch Size 与 Image Number 影响**（Table 4）

| Batch Size | #Images | Speedup |
|-----------|--------|--------|
| 4K | 2 | 1.29× |
| 4K | 3 | 1.32× |
| 8K | 4~7 | **1.35–1.36×** |

- 图像越多 → vision token 越多 → 负载越倾斜 → ReaLB 加速空间越大。

---

## 4. 关键结论和发现

### **主要发现**

1. **MMoE 推理负载高度动态且不可预测**  
   - 与训练不同，推理中专家激活缺乏时间局部性，历史统计无法有效指导调度。

2. **vision tokens 是负载不均的主要来源**  
   - 高分辨率图像产生大量 vision tokens，集中路由造成部分设备成为 straggler。

3. **低精度执行可用于主动负载均衡**  
   - 利用 FP4 Tensor Cores 对 vision-heavy 专家加速，是一种有效的“软扩容”手段。

4. **在线精度切换可做到零开销**  
   - 通过与 All-to-All 通信重叠，完全隐藏量化与调度开销。

5. **ReaLB 显著提升吞吐且几乎无损精度**  
   - 实现 **1.29× MoE 层加速** 与 **最高 1.53× 端到端吞吐提升**，平均精度损失 <1.2 pts。

---

### **方法的局限性**

- **依赖硬件支持 FP4 或更低精度 GEMM**：若无 Tensor Core 支持，则加速效果受限。
- **目前仅针对 prefill 阶段优化**：decode 阶段负载较轻，未作为重点。
- **阈值 $M_d$ 需手动设定**：虽有一定鲁棒性，但尚未实现全自动调参。
- **假设 vision tokens 更易压缩**：该假设在极端文本密集任务中可能不成立。

---

### **未来工作方向**

1. **自适应阈值调节机制**  
   - 如 AIMD（Additive Increase Multiplicative Decrease）策略，根据运行时信号动态调整 $M_d$。

2. **扩展至更多模态**  
   - 应用于视频、音频等多模态 MoE 系统，探索跨模态负载特性。

3. **支持多级精度混合调度**  
   - 不局限于 BF16 ↔ FP4，可在不同 rank 上分配 W8A8、W4A4 等多种精度，进一步细粒度平衡。

4. **部署于 disaggregated PD 架构**  
   - 将 ReaLB 集成至独立的 prefill worker 中，最大化其在生产环境中的收益。

5. **探索 compile-time 与 runtime 协同优化**  
   - 结合静态分析与动态反馈，构建更智能的 MoE 推理引擎。

---

> 📌 **总体评价**：  
> ReaLB 提出了一种新颖的视角——**将 mixed-precision computation 视为一种系统级负载均衡工具**，而非单纯的模型压缩技术。它打破了传统“通过数据重分布来平衡负载”的范式，转而通过“差异化执行速率”实现均衡，为大规模 MMoE 推理系统的高效部署提供了极具前景的新路径。

</details>

---

### 9. [HardNet++: Nonlinear Constraint Enforcement in Neural Networks](https://arxiv.org/abs/2604.19669)

**Authors**: Andrea Goertzen, Kaveh Alim, Navid Azizan  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.19669v1  

#### Abstract
Enforcing constraint satisfaction in neural network outputs is critical for safety, reliability, and physical fidelity in many control and decision-making applications. While soft-constrained methods penalize constraint violations during training, they do not guarantee constraint adherence during in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**HardNet++: Nonlinear Constraint Enforcement in Neural Networks**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
在科学机器学习（scientific machine learning）和基于神经网络的优化求解器（learned optimization solvers）中，模型输出必须满足领域特定的物理约束或安全边界，例如非线性等式与不等式约束。然而，标准神经网络无法保证输出满足这些约束，导致预测结果可能不可行或违反物理规律。

现有方法存在以下局限：
- **Soft-constrained 方法**（如损失函数中加入惩罚项）不能保证推理阶段的可行性。
- **Hard-constrained 方法**（如参数化或投影层）大多仅适用于线性约束或特定形式（如凸二次约束），难以推广到一般非线性情形。
- 迭代投影方法缺乏对不等式约束的支持或理论收敛保证。

### 🔧 提出的新方法：HardNet++
作者提出 **HardNet++**，一种可微分的、用于强制执行**非线性等式与不等式约束**的神经网络输出后处理框架。其核心思想是：
- 在网络输出上应用一个**可微的迭代投影层**，通过局部线性化（local linearizations）逐步将原始输出 $ y_0 $ 映射到可行集 $ \mathcal{Y} = \{ y \mid h(y)=0,~b'\leq g(y)\leq b''\} $。
- 每次迭代采用**阻尼闭式投影**（damped closed-form projection）更新：
  $$
  y_{k+1} = y_k - J_c(y_k)^T (J_c(y_k) J_c(y_k)^T + \epsilon I)^{-1} r(y_k)
  $$
  其中 $ r(y) $ 是 constraint residual，$ J_c $ 是堆叠约束的 Jacobian，$ \epsilon > 0 $ 为阻尼项。

该过程完全可微，支持端到端训练（end-to-end training），即约束满足层参与梯度传播。

### 🆚 相比现有方法的优势
| 特性 | HardNet++ | HardNet | ENFORCE | PiNet/LMI-Net | FSNet |
|------|----------|--------|---------|---------------|--------|
| 支持非线性等式 | ✅ | ❌ | ✅ | ❌ | ✅ |
| 支持非线性不等式 | ✅ | ✅（仅线性） | ❌ | ❌ | ✅ |
| 同时处理两类约束 | ✅ | ✅（仅线性） | ❌ | ❌ | ✅ |
| 可微且支持端到端训练 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 有理论收敛保证 | ✅ | ❌ | 部分 | ❌ | ❌ |
| 不依赖单一闭式投影 | ✅ | ✅ | ✅ | ✅ | ✅ |

> ✅ **创新亮点总结**：
> - 首个能同时处理**通用非线性等式与不等式约束**并具备**理论收敛性证明**的方法。
> - 引入**阻尼机制**克服 vanilla HardNet 在非线性场景下的“平行投影失效”问题（见 Fig. 2）。
> - 与 Levenberg-Marquardt 类似但几何不同：使用 constraint Jacobian 而非 residual Jacobian，保留对已满足不等式的敏感性。

---

## 2. **核心实验方法和设置**

### 📊 实验任务：Model Predictive Control (MPC)
- **目标**：用神经网络近似 MPC 求解器，从初始状态 $ x_{\text{in}} $ 映射到整个时间域上的最优控制序列 $ u $ 和状态轨迹 $ x $。
- **优化问题**：
  $$
  \min_{x,u} \sum_{k=0}^{N-1} \|x_k - x_{\text{target}}\|^2 + \|u_k\|^3 \\
  \text{s.t.}~~x_{k+1} = A x_k + B u_k,~~\|u_k\|_\infty \leq u_b,~~x_k^T Q x_k \leq x_b
  $$
  - 包含线性动力学约束（equality）、控制器 box constraints（inequality）和**非线性状态约束** $ x^T Q x \leq x_b $。
  - $ N=10 $, $ x \in \mathbb{R}^5 $, $ u \in \mathbb{R}^4 $

### 🧪 数据集与训练设置
- **数据生成方式**：随机采样 800 个可行初始条件 $ x_{\text{in}} $ 作为训练集，100 个用于验证，100 个用于测试。
- **网络结构**：前馈神经网络（feed-forward NN），2 层隐藏层，每层 200 神经元。
- **约束满足层配置**：
  - 阻尼系数 $ \epsilon = 0.3 $
  - 迭代次数：500 次 local linearization 步骤
  - 投影层在训练和推理中均激活（active during training）
- **Loss Function**：直接最小化 MPC 目标函数值（即 $ \mathcal{L} = Z(x,u) $）

### 📈 评估指标
1. **Suboptimality（次优性）**：
   $$
   S = \max\left(0, \frac{Z(\hat{x},\hat{u}) - Z(x^*,u^*)}{Z(x^*,u^*)}\right)
   $$
   报告平均与最大次优性。
2. **Constraint Violation（约束违反程度）**：
   - Dynamics constraint violation
   - $ u $ box constraint violation
   - $ x^T Q x \leq x_b $ quadratic constraint violation
   - 使用 constraint residual $ r(y) $ 衡量，报告平均与最大残差。

### ⚔️ 基线方法对比
- **CVXPY Solver**：精确求解器，作为“oracle”基准。
- 本文方法无直接 baselines 对比（因无其他方法支持相同约束类型），但隐含对比了 soft-constrained NN 和 vanilla HardNet。

---

## 3. **主要实验结果和性能指标**

### 📋 定量结果（Table I）
| 指标 | 平均值 | 最大值 |
|------|--------|--------|
| **Suboptimality** | 5.6e-3 | 0.016 |
| **Dynamics constraint violation** | 3.8e-5 | 3.6e-3 |
| **u box constraint violation** | 4.6e-5 | 4.1e-3 |
| **x quadratic constraint violation** | 5.7e-9 | 9.5e-7 |

> ✅ 所有约束均被满足至极高精度（接近数值零），且次优性极低。

### 📈 图形可视化（Fig. 3 & Fig. 4）
- **Fig. 3**：控制变量 $ u $ 的轨迹对比显示，HardNet++ 输出与 CVXPY solver 几乎完全重合，且始终在 box constraints 内。
- **Fig. 4**：状态轨迹在多个二维平面上投影，表明模型成功逼近最优路径，并严格遵守非线性约束区域（红色边界内）。

### 🔍 关键观察
- 即使目标状态 $ x_{\text{target}} $ 在可行域外，模型仍能学习将状态驱动至边界附近——体现与真实 MPC 相同的行为模式。
- 尽管进行了大量迭代投影，性能未显著下降，说明约束满足未牺牲最优性。

> ❗ 注：文中未提供消融实验（ablation study），如不同 $ \epsilon $、迭代次数的影响，也未与其他 constraint enforcement 方法进行横向比较。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **HardNet++ 能有效强制满足非线性等式与不等式约束**，且在整个输入空间上保持高可行性。
2. **理论保障成立**：在 L-smoothness、PL condition 和 Lipschitz constraints 假设下，约束残差能量 $ V(y) = \|r(y)\|^2 $ 以指数速度收敛至零。
3. **实用性强**：在 MPC 任务中实现了接近最优的性能（平均次优性 < 0.6%），同时约束违反几乎为零。
4. **端到端训练至关重要**：若只在推理时施加约束，可能导致 unconstrained output 偏离最优方向；而本方法在训练时即优化 constrained output，提升整体性能。

### ⚠️ 方法局限性
- **计算开销较高**：需多次迭代和雅可比计算，尤其当输出维度高时可能影响实时性。
- **依赖局部线性化**：对于高度非凸或病态约束，可能陷入局部不可行区域。
- **需要自动微分支持**：Jacobian 计算要求网络和约束函数均可导。
- **缺乏鲁棒性分析**：对噪声输入或分布偏移下的表现尚未验证。

### 🔮 未来工作方向
- 扩展至更大规模控制系统（large-scale control）和 PDE-constrained learning。
- 探索 adaptive iteration termination 或 early stopping 以加速推理。
- 结合 uncertainty quantification，增强对不确定性输入的鲁棒性。
- 应用于更多科学机器学习场景，如流体力学、材料建模中的守恒律约束。

---

## ✅ 总结一句话
> **HardNet++ 是首个兼具理论收敛保证与实用性的通用非线性约束强制框架，能够在不牺牲最优性的前提下，实现神经网络输出对复杂非线性等式与不等式约束的高度精确满足。**

</details>

---

### 10. [DT2IT-MRM: Debiased Preference Construction and Iterative Training for Multimodal Reward Modeling](https://arxiv.org/abs/2604.19544)

**Authors**: Zhihong Zhang, Jie Zhao, Xiaojian Huang, Jin Xu, Zhuodong Luo, Xin Liu, Jiansheng Wei, Xuejin Chen  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.19544v1  

#### Abstract
Multimodal reward models (MRMs) play a crucial role in aligning Multimodal Large Language Models (MLLMs) with human preferences. Training a good MRM requires high-quality multimodal preference data. However, existing preference datasets face three key challenges: lack of granularity in preference st...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DT2IT-MRM: Debiased Preference Construction and Iterative Training for Multimodal Reward Modeling》核心总结

---

## 1. 主要贡献和创新点

### 解决的问题
现有的 **Multimodal Reward Models (MRMs)** 在训练过程中面临三大挑战：
- **文本风格偏差 (textual style bias)**：偏好数据中选择的响应通常来自高性能模型，导致 MRM 学习到的是“语言风格”而非“内容质量”。
- **偏好强度缺乏多样性 (lack of granularity in preference strength)**：多数数据集中偏好信号过于单一（强偏好为主），导致模型过拟合且泛化能力差。
- **偏好信号不可靠 (unreliable preference signals)**：依赖闭源模型（如 GPT-4）进行偏好标注时存在位置偏差（positional bias）和推理错误，影响标签可靠性。

此外，现有开源多模态偏好数据噪声大，但缺乏高效、可扩展的数据清洗方法。

---

### 提出的新方法与思路
作者提出 **DT2IT-MRM**，一个集成以下三个核心模块的框架：

#### （1）Debiased Preference Distillation Pipeline（去偏见偏好蒸馏管道）
- 使用**同一 MLLM 池生成多个候选响应**，确保语言风格一致，缓解文本风格偏差。
- 引入**多样性增强机制**：若所有响应质量过高或过低，则通过注入噪声或引入外部参考答案来调整，提升偏好强度分布的多样性。
- 结合 **listwise scoring** 和 **pointwise scoring** 来减少位置偏差，并提供 ground-truth 参考答案以提高评分可靠性。

#### （2）Text-to-Image (T2I) Preference Reformulation（文本到图像偏好的重构）
- 将大规模人类标注的 **text-to-image 生成偏好数据**（如 EvalMuse、HPDv2）转化为适用于 MRM 训练的格式。
- 新设计输入为 `(chosen image, rejected image, text prompt, evaluation prompt)`，输出为“哪张图更好”的判断，更符合 MLLM 的图像理解任务范式，优于 Omni-Reward 的简单顺序交换法。

#### （3）Iterative Training Framework（迭代训练框架）
- 利用当前 MRM 对开源噪声数据进行**一致性过滤与投票修正**，逐步净化数据。
- 数据净化后重新训练 MRM，形成“**模型训练 → 数据清洗 → 模型再训练**”的闭环迭代过程。
- 显著降低对昂贵闭源模型（如 GPT-4o）标注的依赖，实现低成本高质量数据构建。

---

### 相比现有方法的优势
| 维度 | DT2IT-MRM | 现有方法（如 BaseReward, Skywork-VL Reward） |
|------|-----------|---------------------------------------------|
| 文本风格偏差 | ✅ 显著缓解（同模型生成） | ❌ 易受不同模型风格影响 |
| 偏好强度多样性 | ✅ 多样化增强机制保障 | ❌ 多数为强偏好，缺乏梯度 |
| 标注可靠性 | ✅ 提供 GT 答案 + 结构化打分 | ❌ 黑箱模型打分，易错 |
| 数据效率 | ✅ 仅用 ~35% 数据超越 SOTA | ❌ 需更大规模数据 |
| 成本可控性 | ✅ 迭代清洗减少 MLLM 调用 | ❌ 依赖 GPT-4o 全量重标 |

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 构建初始数据：
- **单图像偏好数据**：通过 debiased pipeline 自动构造 **337K preference pairs**。
- **多图像偏好数据**：将 EvalMuse 和 HPDv2 中的 T2I 数据重构为 MRM 可用格式，获得 **133K pairs**。
- 总计初始数据 $ D_0 = 470K $。

#### 开源数据清洗：
在第二阶段迭代中，清洗并整合五个主流开源数据集：
- RLAIF-V
- VLFeedback
- POVID
- WildVision-Battle
- MM-RLHF

最终训练集达 **929K preference pairs**，并进行了 benchmark 去重处理。

---

### 实验设置与评估指标

#### 模型架构
- **Backbone**：基于 Qwen3-VL-8B-Instruct 和 Qwen2.5-VL-7B-Instruct。
- **Head**：添加 linear reward head。
- **Loss Function**：Bradley-Terry loss（Eq. (1) 和 Eq. (3)），无辅助损失。

#### 训练策略
- 学习率：1e-5
- Batch Size：512
- Epochs：1
- Scheduler：cosine with warmup ratio 0.1
- 平台：64 × Ascend 910B3 NPUs
- 工具：LLaMA-Factory 框架

---

### 评估基准与指标

| Benchmark | 样本数 | 主要指标 |
|---------|--------|----------|
| **VL-RewardBench** | 1,247 | Overall Accuracy, Macro Average Accuracy |
| **Multimodal RewardBench** | 4,711 | Holistic evaluation across 6 aspects: General Correctness, Preference, Knowledge, Reasoning, Safety, VQA |
| **MM-RLHF-RewardBench** | 170 | Traditional Accuracy (Acc), Acc+（衡量弱偏好排序能力） |

---

### 基线方法对比
涵盖三类主流 MRM：
- **Generative MRMs**：GPT-5.2, Claude-3.7-Sonnet, LLaVA-Critic, UnifiedReward, MR.Judge
- **Semi-scalar MRMs**：MM-RLHF-Reward
- **Discriminative MRMs**：IXC-2.5-Reward, Skywork-VL Reward, BaseReward, Omni-RewardModel-BT

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）

| Model | VL-RewardBench (Overall Acc) | Multimodal RewardBench (Overall Acc) | MM-RLHF-RewardBench (Acc) | **Mean Overall Acc** |
|-------|-------------------------------|----------------------------------------|------------------------------|------------------------|
| GPT-5.2 | 70.2% | 75.3% | 68.2% | 71.2% |
| Skywork-VL Reward | 73.1% | 74.4% | 65.9% | 71.1% |
| BaseReward (SOTA) | 82.2% | 72.8% | 91.8% | **75.2%** |
| **DT2IT-MRM (Ours)** | **83.5%** | **79.3%** | 89.4% | **80.5%** ✅ |

> ⭐ **总体准确率提升 5.3%**，达到新的 SOTA。

---

### 分项表现亮点

#### 在 VL-RewardBench 上：
- **83.5% Overall Acc**，超过 BaseReward（82.2%）和 GPT-5.2（70.2%）。
- 特别在 **Reasoning 和 Hallucination 抑制方面表现优异**（见 Table 2）。

#### 在 Multimodal RewardBench 上：
- **79.3% Overall Acc**，相比 BaseReward（72.8%）提升 **~9%**。
- 在 **General Correctness、Preference、VQA 等维度均排名第一或第二**（Table 3）。
- 尽管训练数据量仅为 BaseReward 的约 **33%**（929K vs 2.8M），仍实现更高性能，体现极强**数据效率**。

#### 在 MM-RLHF-RewardBench 上：
- 虽未第一（89.4% vs BaseReward 91.8%），但在 **Short 子集上达到 100% 准确率**。
- Acc+ 指标仅次于最优，整体表现均衡。

---

### 消融实验结果（Ablation Studies）

#### （1）Debiased Preference Distillation Pipeline（Table 7）
| 方法 | VL-RewardBench | Multimodal RewardBench | MM-RLHF-RewardBench | Mean Overall Acc |
|------|----------------|--------------------------|------------------------|--------------------|
| Listwise Scoring Only | 74.7% | 76.4% | 77.1% | 76.1% |
| + Diversity Enhancement | 75.6% | 77.2% | 79.4% | 76.9% |
| + Pointwise Scoring | **77.7%** | **78.8%** | **81.2%** | **78.7%** |

✅ 表明两个模块均带来稳定增益，尤其是 pointwise scoring 缓解了 positional bias。

#### （2）T2I Preference Reformulation（vs Omni-Reward）
| 方法 | VL-RewardBench | Multimodal RewardBench | MM-RLHF-RewardBench |
|------|----------------|--------------------------|------------------------|
| Omni-Reward (Baseline) | 46.0% | 55.0% | 47.6% |
| **DT2IT-MRM (Ours)** | **56.5%** (+10.5) | **65.0%** (+10.0) | **59.4%** (+11.8) |

✅ 改进后的 T2I 数据转化方式显著提升各任务性能。

#### （3）Iterative Training Framework（Table 7）
| 方法 | 数据量 | Mean Overall Acc |
|------|--------|------------------|
| 初始数据 $ D_0 $ | 470K | 79.4% |
| $ D_0 $ + 未清洗数据（Baseline） | 942K | 75.5% ↓ |
| 第一次迭代（curated 3 datasets） | 869K | 80.0% ↑ |
| 第二次迭代（全部清洗） | 929K | **80.5%** ✅ |

⚠️ 关键发现：直接混合原始噪声数据会导致性能下降（从 79.4% → 75.5%），证明开源数据含大量噪声。  
✅ 迭代清洗有效提升数据质量和模型性能。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **高质量偏好数据需同时满足：无偏性、多样性、高可靠性** —— DT2IT-MRM 通过系统性设计实现了这三点。
2. ✅ **T2I 偏好数据可通过合理重构用于 discriminative MRM 训练**，极大拓展可用数据来源。
3. ✅ **迭代式训练 + 数据清洗框架能联合优化模型与数据质量**，无需依赖闭源模型即可持续提升性能。
4. ✅ **数据效率极高**：仅用约 1/3 数据量即超越 BaseReward，说明数据质量远胜于数量堆叠。

---

### 方法的局限性
- 当前方法仍部分依赖 GPT-5.2 进行初始打分（虽已尽量减少调用次数）。
- 在 **Knowledge 和 Code 相关任务上表现较弱**（Table 3），因训练数据中缺乏相关偏好对。
- T2I 数据重构依赖人工设计 evaluation prompt，通用性有待进一步验证。

---

### 未来工作方向
- 探索完全基于开源 MLLM 的端到端去偏见数据构建流程。
- 扩展至视频、音频等更多模态的 Reward Modeling。
- 将 DT2IT-MRM 应用于 online RLHF 场景，实现实时反馈优化。
- 构建更具挑战性的细粒度偏好强度 benchmark。

--- 

> 🔚 **总结**：DT2IT-MRM 是一个多维度创新的 MRM 训练框架，在数据构建、表示学习和训练范式上均有突破，不仅刷新了多项 benchmark 的 SOTA，更为未来高质量、低成本、可扩展的多模态奖励建模提供了新范式。

</details>

---

### 11. [DPC: A Distributed Page Cache over CXL](https://arxiv.org/abs/2604.19494)

**Authors**: Shai Bergman, Zhe Yang, Julien Eudine, Giorgio Negro, Onur Mutlu, Arash Tavakkol, Ji Zhang  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.19494v1  

#### Abstract
Modern distributed file systems rely on uncoordinated, per node page caches that replicate hot data locally across the cluster. While ensuring fast local access, this architecture underutilizes aggregate cluster DRAM capacity through massive data redundancy and incurs prohibitive coherence overhead ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DPC: A Distributed Page Cache over CXL》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现代分布式文件系统普遍采用**独立的、每节点（per-node）page cache**架构，这种设计存在两个根本性缺陷：

- **内存利用率低**：热点数据在多个节点上被重复缓存，造成大量**DRAM 冗余复制**，浪费集群整体内存资源。
- **一致性开销高**：为维护缓存一致性，需依赖重量级的软件锁机制或分布式锁管理器（如 LDLM），导致**控制平面开销巨大**，尤其在大规模集群中易引发“coherence storm”（一致性风暴）。

因此，传统架构无法有效将集群的聚合 DRAM 视为一个统一的缓存池，也无法实现高效的一致性管理。

---

### 提出了什么新方法或新思路

本文提出 **DPC (Distributed Page Cache)** —— 一种基于 **CXL 3.0** 的操作系统级分布式页缓存系统，其核心思想是：

- **单副本不变性（Single-copy invariant）**：每个文件页在整个集群中**仅有一个拥有者节点（Owner）持有其唯一驻留副本**，其他节点通过 CXL 远程映射访问该页，而非本地复制。
- **利用 CXL 3.0 的硬件一致性能力**：借助 CXL.mem 协议提供的**跨主机、字节可寻址、硬件管理的缓存一致性**，将远程页访问转化为类似 NUMA 的低延迟 load/store 操作，避免传统 RPC 开销。
- **轻量级软件目录协议**：引入一个集中式的 **DPC Directory**（逻辑上位于存储服务器），负责管理页粒度的所有权、状态迁移和回收协调，解决语义鸿沟与回收冲突问题。

---

### 相比现有方法的优势

| 维度 | 传统方案（如 NFS, VirtioFS） | DPC |
|------|-------------------------------|-----|
| 缓存模式 | 多副本、本地独占 | 单副本、远程共享 |
| 一致性机制 | 软件锁、lease、callback | CXL 硬件行级一致性 + 软件页级协议 |
| 内存效率 | 低（冗余高） | 高（去重） |
| 远程命中延迟 | 高（需网络 RPC + 文件服务处理） | 低（CXL load/store，接近 NUMA 延迟） |
| 控制面开销 | 高（频繁 invalidation） | 低（批量、异步 invalidation） |
| 接口兼容性 | POSIX 兼容 | 完全透明，保留标准接口（POSIX, mmap） |

> ✅ **优势总结**：DPC 在不改变应用和文件系统接口的前提下，实现了**全局视角下的高效缓存利用**和**低开销一致性控制**。

---

## 2. 核心实验方法和设置

### 实验平台与环境

由于目前尚无广泛可用的多主机 CXL 3.0 硬件，作者构建了一个 **QEMU + KVM 的 CXL 仿真框架**：

- **物理主机**：双路 ARMv8.2 服务器，共 128 核，256 GiB DDR4 内存，4×480GB SAS SSD 组成 RAID-0 作为后端存储。
- **虚拟化配置**：使用 QEMU 模拟多个 VM（最多 4 个），每个 VM 有 16 vCPU 和 32 GB RAM，绑定到不同 NUMA 节点以模拟真实拓扑。
- **CXL 模拟**：通过 `ivshmem` 和自研 `dpc_dax` 驱动，将各 VM 的内存暴露为 CXL.mem 设备，由 `ZONE_DEVICE` 管理，实现**硬件一致性共享内存视图**。

### 基线方法对比

评估了以下系统配置（均基于相同 RAID-0 后端）：

| 配置 | 描述 |
|------|------|
| **Virtiofs** | 基线，未修改的 Virtiofs，代表现代虚拟化文件系统 |
| **NFS** | NFSv4.1，广泛使用的网络文件系统 |
| **JuiceFS** | 云原生分布式 POSIX 文件系统，带客户端缓存 |
| **DPC_SC** | DPC 强一致性模式（Strong Consistency） |
| **DPC** | DPC 放松一致性模式（Relaxed Consistency，类 NFS 语义） |

---

### 数据集与工作负载

#### 微基准测试（Microbenchmarks）
使用 `fio` 工具进行：
- **读写延迟、带宽、IOPS**
- 测试三种缓存状态：
  - **CM (Cache Miss)**：目标页不在任何节点缓存
  - **CM-R (Cache Miss-Remote)**：本地未命中，但远程节点已缓存
  - **CH-R (Cache Hit-Remote)**：本地已有远程映射，直接访问

支持两种引擎：
- `libaio`：系统调用路径（read/write）
- `mmap`：内存映射路径（load/store）

#### 应用级基准测试（Application Benchmarks）

| 应用 | 类型 | 工作集大小 | 指标 |
|------|------|------------|------|
| **RocksDB** | KV 存储 | 60 GB | QPS（queries per second） |
| **DeepSeek** | CPU 推理（LLM） | 30 GB | TPS（tokens per second） |
| **DiskANN** | 近似最近邻搜索 | 40 GB | QPS |
| **Webserver** | 静态内容服务 | 32 GB | 几何平均吞吐 |
| **Fileserver** | 文件操作混合负载 | ~32 GB | 几何平均吞吐 |

测试多节点并发场景（1/2/4 节点），衡量扩展性和控制面压力。

---

### 评估指标

- 延迟（Latency）
- 吞吐量（Throughput / Bandwidth）
- IOPS
- 加速比（Speedup）
- 几何平均加速比（Geomean Speedup）
- 回收开销（Reclamation Overhead）

---

## 3. 主要实验结果和性能指标

### 微基准测试结果

#### 读性能（libaio）

| 场景 | DPC vs Virtiofs 加速比 |
|------|------------------------|
| CM-R（首次远程命中） | **2.6× 更低延迟** |
| CH-R（后续远程命中） | **~23× 更低延迟**，受限于 syscall 和 page fault |
| 带宽（CH-R） | **4.5× 提升** |
| IOPS（CH-R） | **高达 72.8× 提升** |

> 💡 原因：DPC 将远程访问从“网络 I/O + 文件服务”降级为“CXL load”，大幅缩短路径。

#### 写性能（libaio）

- **DPC_SC 写延迟略高**（约 195μs vs Virtiofs 110μs），因其需执行两阶段协议确保强一致性。
- 但在 CM-R 场景下，可通过复用远程页减少本地分配，部分抵消开销。
- 大块写入时通过批处理隐藏目录延迟，缩小差距。

#### mmap 性能

- 所有场景下趋势一致，DPC 在 CH-R 中表现尤为突出：
  - **延迟降低最高达 23.3×**
  - **带宽提升 3.7×**
  - **IOPS 提升 18.3×**

> 📌 结论：DPC 对内存映射密集型应用收益最大。

---

### 应用级性能结果（见 Figure 10）

| 应用 | 最大加速比（vs 单节点 Virtiofs） | 几何平均加速比 |
|------|-------------------------------|----------------|
| **DeepSeek** | **12.4×** | — |
| **Webserver** | **16.2×** | — |
| **Fileserver** | **15.2×** | — |
| **总体 Geomean** | — | **5.6×** |

> ✅ **关键观察**：
> - 多节点下性能**不下降**，说明 DPC 控制面无瓶颈。
> - 当总缓存容量足以容纳工作集时，性能飞跃明显（如 DeepSeek 30GB 模型被共享）。
> - DPC_SC 略低于 DPC（2.5× vs 2.8× geomean），体现强一致性代价，但仍显著优于基线。

---

### 消融实验结果

#### 页面回收开销（Page Reclamation Overhead）

- **同步单页回收延迟**：
  - Virtiofs：11 μs（纯本地）
  - DPC：99.7 μs（需目录协调 + 远程 invalidation）
- **但实际影响极小**：DPC 采用**异步批量回收机制**，与页面扫描并行执行，在内存压力测试中未成为瓶颈。

> 🔍 结论：虽然单次操作更重，但批量设计有效掩盖了开销。

---

## 4. 关键结论和发现

### 主要发现

1. **集群 DRAM 可作为统一缓存池**：通过 DPC 的单副本策略，可消除冗余，显著提升内存利用率。
2. **CXL 是构建分布式缓存的理想底座**：CXL 3.0 的硬件一致性能力使得远程页访问可以像本地内存一样高效，是实现透明分布式 page cache 的关键技术使能。
3. **软硬协同设计至关重要**：
   - 硬件提供 cache-line 级一致性；
   - 软件实现 page-level 所有权管理和生命周期控制；
   - 目录服务作为轻量控制平面，协调全局状态。
4. **性能增益显著**：在真实应用场景中，DPC 实现了 **最高 12.4×、几何平均 5.6× 的性能提升**，尤其对低计算强度、高 I/O 密度的应用（如推理、Web 服务）效果最明显。

---

### 方法的局限性

1. **依赖 CXL 3.0 硬件**：当前硬件尚未普及，部署受限。
2. **集中式目录潜在瓶颈**：虽然实验未显现问题，但在超大规模集群中可能成为单点瓶颈，未来可探索分布式目录。
3. **故障容忍性有限**：
   - 节点失效可能导致脏页丢失（与传统 write-back cache 相同）；
   - 虽有 liveness 检测机制，但未提供完全持久化保障。
4. **安全性依赖硬件支持**：需 CXL 适配器具备页粒度权限控制（如 IOMMU 类机制）以防越权访问。

---

### 未来工作方向

1. **分布式目录设计**：将 DPC Directory 分布化以支持更大规模集群。
2. **智能预取与迁移策略**：结合 workload 特征动态迁移热点页，缓解跨节点访问延迟。
3. **与 tiered memory 协同优化**：结合 CXL 池化内存与本地 DRAM 构建多层缓存体系。
4. **更强的容错机制**：探索日志、复制等手段增强数据可靠性。
5. **集成到主流内核**：推动 DPC 或类似机制进入 Linux 主线 kernel。

---

> ✅ **最终结论**：  
> DPC 展示了 **CXL 作为 OS 级集群内存抽象基础设施的巨大潜力**。它不仅是 page cache 的改进，更是迈向“**以内存为中心的分布式系统架构**”的重要一步。

</details>

---

### 12. [Collaborative Contextual Bayesian Optimization](https://arxiv.org/abs/2604.18912)

**Authors**: Chih-Yu Chang, Qiyuan Chen, Tianhan Gao, David Fenning, Chinedum Okwudire, Neil Dasgupta, Wei Lu, Raed Al Kontar  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.18912v1  

#### Abstract
Discovering optimal designs through sequential data collection is essential in many real-world applications. While Bayesian Optimization (BO) has achieved remarkable success in this setting, growing attention has recently turned to context-specific optimal design, formalized as Contextual Bayesian O...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Collaborative Contextual Bayesian Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文聚焦于**Contextual Bayesian Optimization (CBO)** 在多客户端协作场景下的效率提升问题。传统 CBO 需要为每个上下文 $ c \in \mathcal{C} $ 学习最优设计 $ x^*(c) = \arg\max_x f(x, c) $，这在单个客户端上进行时样本效率低、收敛慢。尤其当多个相关但异构（heterogeneous）的客户端各自面临相似任务时，独立优化会浪费潜在的共享知识。

本文提出，在**可控上下文（controllable context）设定下**，多个客户端可以通过协作加速对映射 $ x^*(c) $ 的学习过程，并支持隐私保护通信机制。

---

### **提出了什么新方法或新思路**
作者提出了 **CCBO (Collaborative Contextual Bayesian Optimization)** 框架，其核心创新包括：

- ✅ **统一的协作式 CBO 框架**  
  支持多个客户端联合学习上下文相关的最优设计函数 $ x^*(c) $，适用于在线协作与离线初始化两种模式。
  
- ✅ **基于分歧驱动的决策机制（Disagreement-driven Decision Making）**  
  客户端比较本地推荐的设计 $ x_k^{local}(c) $ 和全局平均模型推荐的设计 $ x^*_{global}(c) $，选择差异最大的上下文进行采样，从而高效纠正局部信念偏差。

- ✅ **自适应切换策略（Adaptive Switching Strategy）**  
  引入概率门控机制 $ p_t $，动态平衡“协作探索”与“独立 Thompson Sampling”，早期依赖协作获取先验，后期转向个性化精调。

- ✅ **隐私保护的后验均值共享机制（Privacy-Preserving Communication）**  
  利用 **Random Fourier Features (RFF)** 对 GP 后验均值进行压缩表示，仅交换低维系数向量 $ \mathbf{w}_{k,t} $，避免原始数据或完整函数泄露。

---

### **相比现有方法的优势**
| 维度 | CCBO 的优势 |
|------|-------------|
| **效率** | 显著减少达到高性能所需的实验次数，尤其在初期阶段通过跨客户端信息共享快速定位优质区域 |
| **通用性** | 支持异构客户端（client heterogeneity），不假设所有客户端具有完全相同的响应函数 |
| **灵活性** | 支持在线协作训练和离线历史知识迁移（offline initialization） |
| **隐私性** | 可选 RFF 压缩通信，降低信息暴露风险，适合工业联盟等敏感环境 |
| **理论保障** | 提供 **sublinear regret guarantee**，证明算法长期无悔（no-regret） |

> ⚠️ 注：这是首个将 CBO 推广到分布式协作范式的框架。

---

## **2. 核心实验方法和设置**

### **使用的数据集 / 测试函数**
实验采用标准黑盒优化基准函数，输入空间划分为上下文 $ c \in \mathcal{C} $ 和设计变量 $ x \in \mathcal{X} $：

- **Ackley 函数族**：`ackley 2-1`, `2-2`, `1-3` （用于同质 setting）
- **Levy 函数族**：`levy 2-1`, `2-2`, `1-3` （用于异质 setting）
- **Hartmann 函数**：`hartmann2-2`

所有函数输入归一化至单位超立方体 $[0,1]^{D_c + D_x}$，并通过负号转换为最大化问题。

此外，还进行了一个真实的 **Hot Rolling 工艺仿真应用**，目标是优化轧制参数以控制晶粒尺寸 $ Z $，其中最终厚度 $ h_f $ 作为上下文。

---

### **实验设置**
- **客户端数量**：$ K = 10 $
- **每轮候选集**：随机采样 100 个候选上下文和 100 个候选设计
- **迭代次数**：$ T = 20(D + D_c) $ 或固定 $ T=50 $（hot rolling）
- **初始数据量**：$ T_0 = 5(D + D_c) $
- **噪声设置**：添加高斯噪声 $ \epsilon \sim \mathcal{N}(0, 0.1\sigma_f) $，$\sigma_f$ 由 1000 次无噪模拟估计
- **探索概率衰减**：$ p_t = 1/\sqrt{t} $

---

### **评估指标**
使用 **log-scale overall simple regret** 衡量性能：
$$
G_t = \frac{1}{K} \sum_{k=1}^K \frac{\int_\mathcal{C} \left[f_k(x^*(c),c) - f_k(\hat{x}_k(c),c)\right] dc}{\int_\mathcal{C} \max_{x_1,x_2} \left[f_k(x_1,c) - f_k(x_2,c)\right] dc}
$$
该指标衡量当前估计设计相对于真实最优设计的平均差距，经上下文空间积分并标准化。

实际计算中通过均匀采样 250 个 $(x,c)$ 对近似。

---

### **基线方法对比**
| 方法 | 简称 | 描述 |
|------|------|------|
| 随机采样 | RS | 完全随机选择上下文-设计对 |
| 多任务 Thompson Sampling | MTS | 单客户端 CBO 的代表方法（Char et al., 2019） |
| 联邦 Thompson Sampling | FTS | 联邦 BO 方法，协作选择设计但随机选择上下文（Dai et al., 2020） |
| CCBO（本文） | —— | 所提协作式 CBO 框架 |
| CCBO + RFF | —— | 使用 RFF 压缩通信的变体 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据与对比结果**

#### ✅ **同质设置下（Homogeneous Setting）**
- 使用 Ackley 函数族测试，结果见 Fig. 3。
- **CCBO 在所有设置下显著优于 FTS、MTS 和 RS**。
- 尤其在前 20–40 次迭代中，**regret 下降速度最快**，表明其能更高效地识别高质量上下文-设计组合。
- 例如，在 `ackley2-1` 上，CCBO 在约 30 次迭代后即接近收敛，而 MTS 仍处于缓慢下降阶段。

#### ✅ **异质设置下（Heterogeneous Setting）**
- 客户端响应函数被施加随机偏移：$ f_k(x,c) = f(x+\delta_x, c+\delta_c) $，$\delta \sim U(-0.05, 0.05)$
- 使用 Levy 函数族测试，结果见 Fig. 4。
- **即使存在异质性，CCBO 依然稳定领先于 MTS 和 FTS**。
- FTS 性能提升有限，因其未主动探索上下文空间，仅协作选择设计。
- CCBO 利用“分歧驱动”机制有效利用共性结构，同时保留个性化能力。

#### ✅ **协作规模的影响（Number of Clients）**
- 使用 Hartmann2-2 函数测试不同 $ K $ 值（Fig. 5）。
- 当 $ K=2 $ 或 $ 5 $ 时，性能较差且波动大；
- 当 $ K=10 $ 或 $ 15 $ 时，性能明显提升，说明**足够的协作伙伴有助于构建更可靠的全局模型**；
- $ K=15 $ 与 $ K=10 $ 差距不大，暗示收益递减。

#### ✅ **RFF 压缩通信的效果（Privacy-Preserving Variant）**
- 结果见 Fig. 6（Levy1-3，J=50）
- **CCBO with RFF 略逊于原版 CCBO，但在后期几乎追平**；
- 相比之下，仍显著优于独立 MTS；
- 表明 RFF 是一种有效的权衡：**牺牲少量早期性能换取更强的隐私性和更低的通信开销**。

#### ✅ **真实制造应用：Hot Rolling 晶粒尺寸优化（Fig. 8）**
- 设计变量：角速度 $ w $、轴向力 $ F_r $、热通量 $ q $
- 上下文：终厚 $ h_f $
- 客户端异质性来源：滚轮半径 $ R_k \sim \mathcal{N}(0.5, 0.1) $
- **CCBO 在整个过程中持续领先**，最终实现最低 regret；
- FTS 与 MTS 表现相近，说明单纯协作设计选择不足以应对复杂上下文依赖；
- 再次验证 CCBO 在现实物理系统中的有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **协作能显著提升 CBO 的样本效率**，尤其是在早期探索阶段。
2. ✅ **“分歧驱动”的上下文选择机制** 比随机或纯局部探索更能发现信息丰富的上下文。
3. ✅ **自适应切换机制** 成功实现了从“协作学习”到“个性化优化”的平滑过渡。
4. ✅ **即使在客户端异质性强的情况下，CCBO 仍保持优越性能**，显示其鲁棒性。
5. ✅ **RFF 压缩通信是一种实用的隐私保护手段**，性能损失小，通信成本亚线性增长。

---

### **方法的局限性**
- 🔹 **理论分析基于离散化假设**（Assumption 1），连续空间下的严格 regret 分析仍具挑战。
- 🔹 **RFF 近似的精度受基函数维度 $ J $ 影响**，过小可能导致信息丢失。
- 🔹 **全局模型 $ \mu_{t-1} $ 是简单平均**，未建模客户端间关系（如图结构、相似度矩阵）。
- 🔹 **未考虑非独立同分布（non-IID）上下文流**，假设上下文可自由选择。

---

### **未来工作方向**
- 🔄 开发更智能的加权聚合机制（如 attention-based fusion）替代简单平均。
- 📈 研究最优 regret rate 下界，为 CBO 提供理论基准。
- 🔐 结合差分隐私（differential privacy）进一步增强安全性。
- 🌐 扩展至非稳态环境（non-stationary contexts）和边缘设备部署。
- 🤝 探索更复杂的协作拓扑（如去中心化 P2P 架构）。

---

> 💡 **总结一句话**：  
> **CCBO 是首个将 Contextual Bayesian Optimization 成功推广至多客户端协作场景的统一框架，兼具高效性、灵活性与隐私性，在仿真与真实制造任务中均展现出显著优势。**

</details>

---

### 13. [TRN-R1-Zero: Text-rich Network Reasoning via LLMs with Reinforcement Learning Only](https://arxiv.org/abs/2604.19070)

**Authors**: Yilun Liu, Ruihong Qiu, Zi Huang  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.19070v1  

#### Abstract
Zero-shot reasoning on text-rich networks (TRNs) remains a challenging frontier, as models must integrate textual semantics with relational structure without task-specific supervision. While graph neural networks rely on fixed label spaces and supervised objectives, recent large language model (LLM)...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TRN-R1-Zero: Text-rich Network Reasoning via LLMs with Reinforcement Learning Only

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **text-rich networks (TRNs)** 上进行零样本节点分类（zero-shot node classification）是一个具有挑战性的任务，因为模型必须在没有任务特定监督的情况下，同时整合文本语义和图结构关系。现有的方法存在以下局限：
- **Graph Neural Networks (GNNs)** 依赖固定的标签空间和有监督目标，泛化能力差。
- **基于 LLM 的方法** 要么忽略图上下文，要么依赖从更大推理模型（如 GPT-4）蒸馏出的 chain-of-thought (CoT) 数据，限制了可扩展性和通用性。

### 🚀 提出的新方法：TRN-R1-Zero
提出了一种**仅通过强化学习（Reinforcement Learning, RL）进行后训练**的框架 TRN-R1-Zero，用于在 TRNs 上实现显式推理。其核心是：
- **Neighbour-aware Group Relative Policy Optimisation (GRPO) Objective**：一种新的策略优化目标，动态调整奖励，以强调邻居信息对分类决策的影响。
- **Margin Gain 指标**：量化邻居聚合前后分类边界的移动程度，作为衡量邻居信息“信息量”的依据，并用于加权奖励。

### 🔍 相比现有方法的优势
| 特性 | TRN-R1-Zero | 其他方法（如 Graph-R1、GraphWiz） |
|------|-------------|-------------------------------|
| 是否需要 SFT 或 CoT 数据 | ❌ 不需要 | ✅ 需要（依赖 LRM 如 GPT-4/Qwen-R1 生成） |
| 是否依赖外部 LRM 进行监督 | ❌ 完全自主训练 | ✅ 是 |
| 是否仅使用 RL | ✅ 是 | ❌ 多为 SFT + RL 或蒸馏 |
| 泛化能力 | ✅ 支持跨域、跨任务零样本推理（edge/graph-level） | ⚠️ 通常局限于训练任务 |
| 推理效率 | ✅ 更短的 response length，更小模型（7B）胜过 14B 模型 | ❌ 响应长，计算开销大 |

> ✅ **核心创新**：首次实现了**无需任何监督微调（SFT）、无需外部 CoT 数据、仅靠 RL 就能激发 LLM 在 TRNs 上的显式关系推理能力**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集（共9个，涵盖4类关系）
| 类型 | 数据集 | 描述 |
|------|--------|------|
| **Citation** | Cora, Citeseer | 学术论文及其引用关系，节点为论文摘要 |
| **Hyperlink** | WikiCS | Wikipedia 页面间的超链接，节点为文章内容 |
| **Social** | Instagram | 用户社交关注关系，节点为用户 bio 或帖子 |
| **Co-purchase** | Photo, History | 亚马逊商品共购关系，节点为评论或描述 |
| **Commonsense / Edge-level** | Expla-Graph, WikiCS-Link, Instagram-Link | 构造用于图级和边级预测任务 |

> ⚠️ **训练仅用两个数据集**：`Citeseer`（citation） 和 `History`（co-purchase），其余全部用于**零样本跨域/跨任务评估**。

### 🧪 实验设置与评估指标
- **任务类型**：
  - 主任务：Zero-shot node classification
  - 扩展任务：Graph-level reasoning, Link prediction（zero-shot）
- **评估指标**：
  - Accuracy (%)
  - Macro-F1 (%)
- **训练细节**：
  - 基座模型：Qwen2.5-7B-Instruct（主），也测试了 Llama 和 Qwen-14B
  - 使用 LoRA 进行高效微调（rank=64）
  - 强化学习算法：Dr.GRPO + KL 正则化
  - Margin Gain 温度参数：α = 10
  - 硬件：单张 AMD MI300X GPU

### 🆚 基线方法对比
| 类别 | 方法 | 简介 |
|------|------|------|
| **LLMs** | GPT-4o, Llama-3.1-8B, Qwen系列 | 直接提示原始 LLM 进行分类 |
| **Graph Foundation Models (GFMs)** | ZeroG, LLaGA | 利用 LLM 编码 + 图结构聚合 |
| **Reasoning LLMs** | Graph-R1 (14B) | 使用 CoT 蒸馏 + SFT + RL，当前 SOTA |

> 💡 TRN-R1-Zero 与 Graph-R1 对比尤为关键：后者使用更强的 14B 模型并依赖外部 CoT 数据，而 TRN-R1-Zero 仅用 7B 模型且完全自驱训练。

---

## 3. 主要实验结果和性能指标

### 📊 表2：Zero-shot Node Classification 性能汇总（Accuracy / Macro-F1）

| Method | Cora ↑ | WikiCS ↑ | Instagram ↑ | Photo ↑ | **Avg.** ↑ |
|--------|--------|----------|--------------|---------|-----------|
| GPT-4o | 70.30 / 71.44 | 69.69 / 64.51 | 42.42 / 39.79 | 69.93 / 68.55 | **63.09 / 61.07** |
| Qwen2.5-14B-it | 67.22 / 68.26 | 73.03 / 70.78 | 55.60 / 52.94 | 58.51 / 61.45 | **63.59 / 63.36** |
| Graph-R1 (14B) | 68.15 / 67.34 | 73.25 / 70.11 | 52.03 / 52.06 | — | — |
| **TRN-R1-Zero (7B)** | **72.59 / 70.33** | **73.63 / 70.30** | **54.76 / 52.54** | **65.12 / 64.22** | **66.53 / 64.35** ✅ |

> ✅ **TRN-R1-Zero 在所有数据集上均取得最佳平均性能**，即使使用更小的 7B 模型也超越了 GPT-4o 和 14B 的 Graph-R1。

### 🔍 关键发现：
- 在 `WikiCS` 上达到 **73.63% 准确率**，显著优于其他方法。
- 即使在噪声较大的 `Instagram` 和 `Photo` 上仍保持稳健表现。
- **未参与训练的领域（如 hyperlink 和 social）也能良好泛化**，证明真正的 zero-shot 能力。

### 🔬 消融实验与分析

#### （1）**Margin Gain 的有效性（Figure 5）**
- 使用 `exp(α|△|)` 加权奖励后：
  - 训练更稳定，准确率持续上升。
  - 响应长度增加 → 推理更深。
  - “neighbour” 出现频率上升 → 显式利用邻域信息。
- 说明该机制成功引导模型关注高影响邻居样本。

#### （2）**跨任务零样本迁移（Table 3）**
| 任务 | 模型 | Expla-Graph | WikiCS-Link | Instagram-Link |
|------|------|------------|------------|----------------|
| Graph Reasoning | Base Qwen (14B) | 89.89 | 72.10 | 71.80 |
| | Graph-R1 (14B) | 89.71 | 48.90 | 56.40 |
| | **TRN-R1-Zero (14B)** | **90.25 (+0.36)** | **73.90 (+1.80)** | **74.20 (+2.40)** ✅ |
| Link Prediction | TRN-R1-Zero (7B) | +3.06 | **+16.10** | +1.90 |

> ✅ **尽管只在 node-level 任务上训练，TRN-R1-Zero 在 edge 和 graph-level 任务上依然表现出色，甚至反超专门为这些任务设计的 Graph-R1**。

#### （3）不同 LLM Backbones 的效果（Figure 6）
| 模型 | 平均增益（Accuracy） |
|------|--------------------|
| Llama-3.2-3B-Instruct | +14.4 |
| Llama-3.1-8B-Instruct | +9.0 |
| Qwen2.5-14B-Instruct | +4.7 |

> ✅ 方法具有良好的**架构通用性**，尤其在小模型上提升显著。

#### （4）监督设置下的表现（Table 4）
| 方法 | Citeseer | History |
|------|---------|--------|
| GCN | 76.45 | 84.23 |
| LLaGA | 76.73 | 85.56 |
| **TRN-R1-Zero** | **77.74** | **86.71** ✅ |

> ✅ 即使在有标签可用的监督场景下，TRN-R1-Zero 依然优于传统 GNN 和 GFM 方法。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **TRN-R1-Zero 成功实现了纯 RL 驱动的 TRN 推理**，无需 SFT 或外部 CoT 数据，打破了对大型推理模型（LRM）的依赖。
2. **Neighbour-aware GRPO + Margin Gain 机制有效激活了 LLM 的关系推理能力**，使模型学会动态利用邻居信息。
3. **具备强大的 zero-shot 泛化能力**：不仅跨域（cross-domain），还能跨任务（cross-task）推广到 edge 和 graph-level 任务。
4. **高效且轻量**：使用 7B 模型即可超越 14B 模型，响应更简洁（见 Box2 vs Box3），适合部署。

### ⚠️ 局限性
- **依赖预训练 LLM 的领域知识**：若 base LLM 缺乏相关背景知识（如法律、医学），RL 很难弥补这一缺陷。
- **Margin Gain 计算依赖冻结编码器**：目前使用 SGC + 冻结 LLM 编码器估算 margin，可能无法完全反映真实推理路径。
- **对噪声文本敏感**：TRNs 中常存在简短、重复或无意义文本（如“good product”），影响推理质量（见 Appendix A）。

### 🔮 未来工作方向
- 将 TRN-R1-Zero 扩展至多模态网络（text + image/graph）。
- 结合检索增强（Retrieval-Augmented）机制缓解噪声问题。
- 探索更精细的 neighborhood contribution 评估方式，替代当前基于 embedding margin 的 proxy metric。
- 应用于动态图或流式图学习场景。

---

> 🔗 **代码已开源**：[https://github.com/superallen13/TRN-R1-Zero](https://github.com/superallen13/TRN-R1-Zero)

</details>

---

### 14. [CROWDio: A Practical Mobile Crowd Computing Framework with Developer-Oriented Design, Adaptive Scheduling, and Fault Resilience](https://arxiv.org/abs/2604.19363)

**Authors**: Lakshani Manamperi, Disumi Pathirana, Thiwanka Pathirana, Nipun Premarathna, Kutila Gunasekara  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.19363v1  

#### Abstract
Mobile Crowd Computing (MCdC) leverages the idle computational capacity of consumer smartphones to enable distributed task processing at scale; however, widespread real-world adoption remains constrained by the absence of developer-oriented frameworks capable of transparently managing device heterog...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《CROWDio: A Practical Mobile Crowd Computing Framework with Developer-Oriented Design, Adaptive Scheduling, and Fault Resilience》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前 **Mobile Crowd Computing (MCdC)** 虽然理论上可行，但由于缺乏面向开发者的框架，难以在实际中广泛应用。主要挑战包括：
- 设备异构性（heterogeneous devices）导致任务调度低效；
- 移动设备频繁断连、电池耗尽等引发的**容错难题**；
- 开发者需手动处理分布式编程复杂性（如并行控制、状态恢复），门槛高。

现有系统（如 Hyrax、Misco、Honeybee）多为研究原型，缺少生产级 **SDK** 和自动化调度机制。

### 提出的新方法与创新思路
CROWDio 是一个**集中式 MCdC 平台**，具备三大核心子系统：

#### （1）声明式 Developer SDK（开发者导向设计）
- 通过单个函数注解（function annotation）将串行代码自动转为分布式任务；
- 自动实现 `map` 类型的数据并行操作，隐藏任务分发与结果聚合逻辑；
- 极大降低开发者的认知负担，接近云函数服务（cloud function service）的易用性。

#### （2）基于实时遥测的可插拔 MCDM 调度架构（Adaptive Scheduling）
- 引入 **Multi-Criteria Decision Making (MCDM)** 框架进行能力感知调度；
- 利用实时设备指标（CPU利用率、内存、电量、网络延迟、温控状态）构建决策矩阵；
- 支持多种策略动态切换：**WRR、FIFO、EDAS、ARAS、MABAC**，无需修改调度核心；
- 使用 **Shannon Entropy Weighting** 自动计算各指标权重，避免主观赋权。

#### （3）分层 Checkpointing 机制（Fault Resilience）
- 提出三级检查点模型：
  - **BASE**：完整状态快照；
  - **DELTA**：仅记录自上次以来的变化变量，减少传输开销；
  - **COMPACTED**：定期合并 DELTA 链条，限制恢复复杂度为 O(k)，默认 k=50；
- 在源码级别进行自动插桩（instrumentation），实现透明的状态保存与恢复；
- 断线后可在其他设备上从最近有效检查点继续执行，保障容错性。

### 相比现有方法的优势
| 维度 | CROWDio | 其他系统（Hyrax/Misco/Honeybee/BOINC） |
|------|--------|---------------------------------------|
| **SDK 支持** | ✅ 完整声明式接口 | ❌ 无或部分支持 |
| **调度智能性** | ✅ 基于 MCDM 的自适应调度 | ❌ 静态或简单轮询 |
| **容错机制** | ✅ 分层 checkpointing | ❌ 无或应用层自行实现 |
| **开发者友好性** | ✅ 单注解即可分布式化 | ❌ 手动管理并发与故障 |

---

## 2. 核心实验方法和设置

### 使用的数据集与工作负载
共评估三种典型任务类型，在六台异构 Android 设备上运行：

| 工作负载 | 描述 | 数据来源 |
|--------|------|---------|
| **Monte Carlo π 估计** | CPU 密集型基准测试，迭代次数从 1M 到 100M | 自定义实现 |
| **Sentiment Analysis** | AI/NLP 推理任务，对客服工单文本情感分类 | [Help desk tickets (Mendeley Data)](https://doi.org/10.17632/btm76zndnt.2) |
| **Tile-based Image Processing** | 数据并行图像处理任务 | [Weather Image Recognition Dataset (Kaggle)](https://www.kaggle.com/datasets/jkanthony/weather-images) |

### 实验设置
- **设备集群**：6 台不同配置的 Android 手机（见 Table 2），涵盖 RAM（3.3–7.4GB）、频率（1.74–2.00 GHz）差异；
- **重复次数**：每组实验重复 10 次，报告均值 ± 标准差；
- **通信机制**：基于 WebSocket 的持久连接，协调器（Coordinator）集中管理任务分发与状态；
- **持久化**：所有检查点与元数据存储于中央 SQLite 数据库。

### 评估指标
| 指标 | 含义 |
|-----|------|
| **Execution Time** | 总任务完成时间 |
| **Speedup** | 相对于最佳单设备的加速比 |
| **Improvement (%)** | 相较基线调度算法的时间节省百分比 |
| **Checkpoint Overhead** | 启用检查点带来的额外耗时 |
| **Jain’s Fairness Index (J)** | 衡量任务分配公平性（理想值为 1） |

### 基线方法对比
- **调度对比**：
  - FIFO（先进先出）
  - WRR（Weighted Round Robin）
- **执行模式对比**：
  - 单设备执行（最优/最差设备）
  - CROWDio 分布式执行（启用/禁用 checkpointing）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）相比单设备执行的显著加速
| 迭代数 | 最佳单设备耗时（s） | CROWDio WRR 耗时（s） | 加速比（Speedup） |
|-------|------------------|--------------------|----------------|
| 1M    | 2.06             | 1.20               | ×1.7           |
| 10M   | 19.2             | 6.13               | ×3.1           |
| 100M  | 192              | 37.9               | **×5.1**       |

> 图3显示随着任务规模增大，CROWDio 的优势持续扩大，体现出良好的**线性可扩展性**。

#### （2）调度算法对比：WRR 显著优于 FIFO
| 迭代数 | FIFO 耗时（s） | WRR 耗时（s） | 性能提升 |
|-------|--------------|-------------|----------|
| 1M    | 2.0          | 1.2         | +40.0%   |
| 10M   | 6.2          | 5.5         | +11.3%   |
| 100M  | 113.4        | 48.9        | **+56.9%** |

> 在大规模任务下，FIFO 因忽略设备能力差异造成“头阻塞”（head-of-line blocking），而 WRR 基于能力加权调度，大幅提升效率。

#### （3）Checkpointing 开销可控且稳定
| 检查点模式 | 平均耗时（1M trials） | 额外开销 |
|----------|------------------|--------|
| Disabled | 2.06 s           | —      |
| 5s interval | 4.14 s         | +2.08 s |
| 2s interval | 4.90 s         | +2.84 s |
| 0.5s interval | 4.94 s       | +2.88 s |

> 结果表明：无论检查点频率如何，**额外开销始终被限制在 2–3 秒之间**，主要来自序列化、压缩与网络传输，具有**有界性（bounded overhead）**。

#### （4）任务分配高度公平
- **Jain’s Fairness Index J = 0.889**（标准差 < 0.01）
- 表明即使在显著异构环境下，任务仍能稳定、均衡地分布到各类设备上。

#### （5）协调开销说明
- 对于小粒度任务（如图像处理），当**单任务计算时间 ≈ 协调开销**时，分布式收益消失；
- 只有当 **per-task computation : coordination cost > ~10:1** 时，分布式才具优势（类比 Amdahl’s Law）。

---

## 4. 关键结论和发现

### 主要发现
1. **开发者抽象至关重要**：通过声明式 SDK 将分布式编程简化至“单注解”级别，极大提升可用性；
2. **能力感知调度是关键**：在异构移动集群中，**naive 调度（如 FIFO）会导致严重瓶颈**，而 MCDM 驱动的自适应调度可带来最高达 **56.9% 的性能提升**；
3. **容错必须轻量高效**：分层 checkpointing 在保证恢复能力的同时，仅引入 **2–3 秒的固定开销**，适合资源受限的移动环境；
4. **公平性可保障**：系统能在异构设备间维持稳定的负载均衡（J = 0.889），避免低端设备过载或高端设备闲置。

### 局限性
- 实验仅在 **6 台设备的小规模集群** 上进行，未验证更大规模（如 50–100 台）下的可扩展性；
- 实验环境为受控实验室，未模拟真实场景中的**用户中断、设备移动、自愿退出**等行为；
- 多数 MCDM 策略（EDAS/ARAS/MABAC）尚未全面比较，目前重点验证了 WRR；
- 缺乏对 **energy consumption** 的测量与分析，无法评估能耗代价。

### 未来工作方向
- 扩展至 **multi-stage DNN inference** 场景，支持移动端上的协作式 AI 推理；
- 引入 **pipeline scheduling** 以优化深度学习任务的流水线执行；
- 探索更复杂的容错策略，如预测性重调度（predictive rescheduling）；
- 在真实用户环境中部署，研究参与激励机制与长期稳定性。

--- 

> ✅ **总体评价**：CROWDio 成功弥合了 MCdC 理论研究与实际部署之间的鸿沟，首次提供了一个集 **developer-friendly SDK、adaptive scheduling、fault-tolerant checkpointing** 于一体的完整解决方案，为移动设备协同计算走向实用化迈出关键一步。

</details>

---

### 15. [AC-SINDy: Compositional Sparse Identification of Nonlinear Dynamics](https://arxiv.org/abs/2604.18889)

**Authors**: Peter Racioppo  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.18889v1  

#### Abstract
We present AC-SINDy, a compositional extension of the Sparse Identification of Nonlinear Dynamics (SINDy) framework that replaces explicit feature libraries with a structured representation based on arithmetic circuits. Rather than enumerating candidate basis functions, the proposed approach constru...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AC-SINDy: Compositional Sparse Identification of Nonlinear Dynamics

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **SINDy**（Sparse Identification of Nonlinear Dynamics）框架依赖于预定义的显式特征库（如多项式项），存在以下问题：
- **组合爆炸**：候选函数库随状态维度 $d$ 和交互阶数 $p$ 组合增长（$\mathcal{O}(d^p)$），导致高维系统难以扩展。
- **噪声敏感**：对观测噪声鲁棒性差，尤其在导数估计阶段。
- **表示与结构不匹配**：真实动力学常具有稀疏、低阶交互结构，但 SINDy 强制将其展开为大量独立基函数。

### 🚀 提出的新方法与思路
作者提出 **AC-SINDy**（Arithmetic Circuit-based SINDy），一种基于**算术电路**（arithmetic circuit）的组合式 SINDy 扩展方法，核心思想包括：

| 创新点 | 内容 |
|--------|------|
| **Compositional Representation** | 放弃显式枚举基函数，转而通过**线性变换与乘法交互的递归组合**构建非线性特征，形成紧凑的计算图（computational graph）。 |
| **Sparsity over Structure** | 将稀疏性施加于**计算图的结构**上（即边权重剪枝），而非固定基下的系数选择，实现“结构稀疏”而非“系数稀疏”。 |
| **Latent State + Multi-step Supervision** | 引入编码器 $\mathcal{E}$ 学习去噪后的隐状态 $s_t$，并用共享参数的动力学模型 $f_0$ 进行多步前向传播，增强时间一致性。 |
| **Feature Normalization** | 提出一种尺度不变的归一化方式，解决因乘法层导致的参数不可识别问题，使系数大小能反映功能重要性。 |
| **Gradient-based Pruning** | 使用梯度估计（一阶梯度近似）衡量参数重要性，进行迭代剪枝以学习稀疏结构。 |

### ⚖️ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可扩展性** | 参数数量从 $\mathcal{O}(d^p)$ 降至 $\mathcal{O}(mpd)$，其中 $m$ 是有效组合项数；当 $m \ll d^{p-1}$ 时显著更优。 |
| **表达能力** | 可高效表示具有低秩张量结构（low-rank tensor structure）或低复杂度算术电路结构的函数，即使其显式展开项很多。 |
| **抗噪能力** | 多步预测监督 + 隐状态建模，隐式完成去噪，无需显式数值微分或预处理。 |
| **可解释性保留** | 剪枝后可回溯符号表达式，仍保持物理意义明确的 governing equations 形式。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
实验集中在典型非线性与混沌动力系统上：
1. **2D Nonlinear System**  
   $$
   \dot{x} = -0.1x + y,\quad \dot{y} = -2x - 0.1y - 0.5xy - 0.025y^2
   $$
2. **Lorenz System**（经典混沌系统）  
   $$
   \dot{x} = \sigma(y - x),\quad \dot{y} = x(\rho - z) - y,\quad \dot{z} = xy - \beta z,\quad (\sigma,\rho,\beta)=(10,28,8/3)
   $$
3. **Forced Lorenz System**  
   在 $x$ 方程中加入 $0.1\sin(x)$ 外力项。

此外，在 **含噪声场景** 下测试鲁棒性：对 2D 系统添加高斯噪声 $\epsilon \sim \mathcal{N}(0, 0.05^2)$。

### 🔧 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| **实现平台** | PyTorch |
| **训练目标** | 最小化多步预测误差：$\mathcal{L} = \sum_{k=1}^K \|s_{t+k} - x_{t+k}\|^2$ |
| **优化流程** | 迭代剪枝 + 微调（prune-and-finetune loop），基于验证损失选择最终模型 |
| **剪枝策略** | 梯度重要性估计（$\Delta \approx \partial \mathcal{L}/\partial w \cdot w$） |
| **归一化** | Feature Normalization（使用运行标准差，stop-gradient） |
| **评估维度** | - 预测轨迹准确性（MSE）<br>- 恢复 governing equations 的结构正确性<br>- 参数效率与缩放行为 |

### 🆚 基线方法对比
尽管文中未直接列出与其他 SINDy 变体的量化对比表格，但明确指出是相对于以下方法的改进：
- **Standard SINDy**：显式库 + 稀疏回归
- **SINDy-PI**：处理隐式/有理动态
- **DeepMoD / Neural ODEs**：结合神经网络提升鲁棒性
- **SINDy Autoencoder**：学习稀疏坐标的尝试

AC-SINDy 的设计融合了深度学习灵活性与 SINDy 的可解释性，定位为结构性改进而非单纯性能超越。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）模型恢复精度（Qualitative Recovery）
| 系统 | 结果描述 |
|------|----------|
| **2D Nonlinear System** | 成功恢复所有主导项：<br>$\dot{x}: -0.11x + 1.00y$<br>$\dot{y}: -1.93x -0.11y -0.50xy -0.029y^2$<br>系数接近真值，结构完全一致。 |
| **Lorenz System** | 主导交互被识别：<br>虽未精确还原原始形式（如出现 $xy$, $x$, $y^2$ 等项），但关键耦合关系（如 $x \to y$, $xy \to z$）被捕获，相空间轨迹形态高度相似。 |

#### （2）噪声鲁棒性测试（Noisy Observations）
- 输入含噪声轨迹（$\sigma=0.05$），模型仍能拟合平滑的动力学路径。
- 恢复方程包含主要线性和双线性项（如 $x$, $y$, $xy$），遗漏弱项 $y^2$，引入轻微伪项（如 $x^2$），但整体结构合理。

#### （3）参数效率对比（Scaling Behavior）
下图展示了不同状态维度 $d$ 下，标准 SINDy 与 AC-SINDy 的参数增长趋势：

![Figure 2](#)  
- 当 $p=2$：两者均为 $\mathcal{O}(d)$，SINDy 常数更小；
- 当 $p=3,4$：SINDy 组合爆炸（$\mathcal{O}(d^3), \mathcal{O}(d^4)$），而 AC-SINDy 保持近似二次增长（$\mathcal{O}(d^2)$），优势明显。

> 示例：$d=15, p=4$ 时，SINDy 需 ~4000 项，AC-SINDy 仅需数百参数即可逼近。

#### （4）训练动态与剪枝效果
- 图3显示：每次剪枝引起短暂 loss 上升，随后通过微调恢复；
- 初始剪枝可去冗余、防过拟合，提升泛化；
- 最终模型选在验证 loss 最低点，体现“稀疏-表达力”权衡。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **组合式表示优于显式枚举**：许多物理系统虽需大量单项式展开，但仍可通过低复杂度 arithmetic circuit 表示，AC-SINDy 能有效捕捉此类结构。
2. **结构稀疏 > 系数稀疏**：将稀疏性作用于计算图结构，比在固定基中选系数更具表达效率和可扩展性。
3. **多步一致性提升鲁棒性**：联合学习隐状态与共享动力学模型，并施加 multi-step supervision，可在不牺牲可解释性的前提下有效抑制噪声影响。
4. **Feature Normalization 至关重要**：解决了 compositional models 中因 scale redistribution 导致的参数不可识别问题，确保剪枝依据可靠。

### ⚠️ 局限性
- 当前实验集中于**低维系统**（$d \leq 3$），尚未验证在高维 PDE 或大规模系统中的表现。
- 恢复的 governing equations 形式可能不同于原始解析形式（如 Lorenz 中未还原标准形式），需后处理简化。
- 剪枝过程依赖梯度估计，可能存在局部最优风险，且计算开销高于一次性稀疏回归。

### 🔮 未来工作方向
- 扩展至更高维系统（如流体力学、反应扩散方程等）；
- 探索更复杂的非线性原语（beyond linear/multiplicative layers）；
- 结合 symbolic regression 工具自动简化恢复的表达式；
- 应用于真实世界传感器数据（partial observation, irregular sampling）。

---

## 总结一句话
> **AC-SINDy 通过将 SINDy 从“稀疏基选择”升级为“稀疏结构学习”，利用 arithmetic circuit 构建紧凑、可解释且可扩展的动力学模型，在保持 interpretability 的同时显著提升了对噪声和高维系统的适应能力。**

</details>

---

### 16. [SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving](https://arxiv.org/abs/2604.19157)

**Authors**: Jinda Jia, Jisen Li, Zhongzhu Zhou, Jung Hwan Heo, Jue Wang, Tri Dao, Shuaiwen Leon Song, Ben Athiwaratkun, Chenfeng Xu, Tianyi Zhang, Xiaoxia Wu  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.19157v1  

#### Abstract
KV-cache memory is a major bottleneck in real-world LLM serving, where systems must simultaneously support latency-sensitive small-batch requests and high-throughput concurrent workloads. Although many KV-cache compression methods improve offline accuracy or compression ratio, they often violate pra...

---

### 17. [From Natural Language to Executable Narsese: A Neuro-Symbolic Benchmark and Pipeline for Reasoning with NARS](https://arxiv.org/abs/2604.18873)

**Authors**: Mina Gabriel, Pei Wang  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.18873v1  

#### Abstract
Large language models (LLMs) are highly capable at language generation, but they remain unreliable when reasoning requires explicit symbolic structure, multi-step inference, and interpretable uncertainty. This paper presents a neuro-symbolic framework for translating natural-language reasoning probl...

---

### 18. [Towards Scalable Lifelong Knowledge Editing with Selective Knowledge Suppression](https://arxiv.org/abs/2604.19089)

**Authors**: Dahyun Jung, Jaewook Lee, Heuiseok Lim  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.19089v1  

#### Abstract
Large language models (LLMs) require frequent knowledge updates to reflect changing facts and mitigate hallucinations. To meet this demand, lifelong knowledge editing has emerged as a continual approach to modify specific pieces of knowledge without retraining the entire model. Existing parameter ed...

---

### 19. [Towards Energy Impact on AI-Powered 6G IoT Networks: Centralized vs. Decentralized](https://arxiv.org/abs/2604.19377)

**Authors**: Anjie Qiu, Donglin Wang, Sanket Partani, Andreas Weinand, Hans D. Schotten  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.19377v1  

#### Abstract
The emergence of sixth-generation (6G) technologies has introduced new challenges and opportunities for machine learning (ML) applications in Internet of Things (IoT) networks, particularly concerning energy efficiency. As model training and data transmission contribute significantly to energy consu...

---

### 20. [A Mechanism and Optimization Study on the Impact of Information Density on User-Generated Content Named Entity Recognition](https://arxiv.org/abs/2604.18944)

**Authors**: Jiang Xiaobo, Dinghong Lai, Song Qiu, Yadong Deng, Xinkai Zhan  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.18944v1  

#### Abstract
Named Entity Recognition (NER) models trained on clean, high-resource corpora exhibit catastrophic performance collapse when deployed on noisy, sparse User-Generated Content (UGC), such as social media. Prior research has predominantly focused on point-wise symptom remediation -- employing customize...

---

### 21. [GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](https://arxiv.org/abs/2604.19398)

**Authors**: Ziyang Wang, Jiangfeng Xiao, Chuan Xiao, Ruoxiang Li, Rui Mao, Jianbin Qin  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.19398v1  

#### Abstract
Large language models (LLMs) are expensive to serve because model parameters, attention computation, and KV caches impose substantial memory and latency costs. We present GRASPrune, a structured pruning framework applied after pretraining that jointly prunes FFN channels and KV head groups under a s...

---

### 22. [STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](https://arxiv.org/abs/2604.18976)

**Authors**: MinJae Jung, YongTaek Lim, Chaeyun Kim, Junghwan Kim, Kihyun Kim, Minwoo Kim  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.18976v1  

#### Abstract
While Large Language Models (LLMs) are widely used, they remain susceptible to jailbreak prompts that can elicit harmful or inappropriate responses. This paper introduces STAR-Teaming, a novel black-box framework for automated red teaming that effectively generates such prompts. STAR-Teaming integra...

---

### 23. [Formally Verified Patent Analysis via Dependent Type Theory: Machine-Checkable Certificates from a Hybrid AI + Lean 4 Pipeline](https://arxiv.org/abs/2604.18882)

**Authors**: George Koomullil  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.18882v1  

#### Abstract
We present a formally verified framework for patent analysis as a hybrid AI + Lean 4 pipeline. The DAG-coverage core (Algorithm 1b) is fully machine-verified once bounded match scores are fixed. Freedom-to-operate, claim-construction sensitivity, cross-claim consistency, and doctrine-of-equivalents ...

---

### 24. [OLLM: Options-based Large Language Models](https://arxiv.org/abs/2604.19087)

**Authors**: Shashank Sharma, Janina Hoffmann, Vinay Namboodiri  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19087v1  

#### Abstract
We introduce Options LLM (OLLM), a simple, general method that replaces the single next-token prediction of standard LLMs with a \textit{set of learned options} for the next token, indexed by a discrete latent variable. Instead of relying on temperature or sampling heuristics to induce diversity, OL...

---

### 25. [UAF: A Unified Audio Front-end LLM for Full-Duplex Speech Interaction](https://arxiv.org/abs/2604.19221)

**Authors**: Yadong Li, Guoxin Wu, Haiping Hou, Biye Li  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19221v1  

#### Abstract
Full-duplex speech interaction, as the most natural and intuitive mode of human communication, is driving artificial intelligence toward more human-like conversational systems. Traditional cascaded speech processing pipelines suffer from critical limitations, including accumulated latency, informati...

---

### 26. [From Experience to Skill: Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](https://arxiv.org/abs/2604.19516)

**Authors**: Beining Wu, Fuyou Mao, Jiong Lin, Cheng Yang, Jiaxuan Lu, Yifu Guo, Siyu Zhang, Yifan Wu, Ying Huang, Fu Li  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19516v1  

#### Abstract
Generative engines (GEs) are reshaping information access by replacing ranked links with citation-grounded answers, yet current Generative Engine Optimization (GEO) methods optimize each instance in isolation, unable to accumulate or transfer effective strategies across tasks and engines. We reframe...

---

### 27. [Multi-modal Reasoning with LLMs for Visual Semantic Arithmetic](https://arxiv.org/abs/2604.19567)

**Authors**: Chuou Xu, Liya Ji, Qifeng Chen  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19567v1  

#### Abstract
Reinforcement learning (RL) as post-training is crucial for enhancing the reasoning ability of large language models (LLMs) in coding and math. However, their capacity for visual semantic arithmetic, inferring relationships from images, remains underexplored. The classic text analogy "king"-"man"+"w...

---

### 28. [Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](https://arxiv.org/abs/2604.18729)

**Authors**: Shubin Kim, Yejin Son, Junyeong Park, Keummin Ka, Seungbeen Lee, Jaeyoung Lee, Hyeju Jang, Alice Oh, Youngjae Yu  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.18729v1  

#### Abstract
Humor holds up a mirror to social perception: what we find funny often reflects who we are and how we judge others. When language models engage with humor, their reactions expose the social assumptions they have internalized from training data. In this paper, we investigate counterfactual unfairness...

---

### 29. [FG$^2$-GDN: Enhancing Long-Context Gated Delta Networks with Doubly Fine-Grained Control](https://arxiv.org/abs/2604.19021)

**Authors**: Pingwei Sun, Yuxuan Hu, Jianchao Tan, Xue Wang, Jiaqi Zhang, Yifan Lu, Yerui Sun, Yuchen Xie, Xunliang Cai  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19021v1  

#### Abstract
Linear attention mechanisms have emerged as promising alternatives to softmax attention, offering linear-time complexity during inference. Recent advances such as Gated DeltaNet (GDN) and Kimi Delta Attention (KDA) have demonstrated that the delta rule, an online gradient descent update, enables sup...

---

### 30. [Accelerating Optimization and Machine Learning through Decentralization](https://arxiv.org/abs/2604.19518)

**Authors**: Ziqin Chen, Zuang Wang, Yongqiang Wang  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19518v1  

#### Abstract
Decentralized optimization enables multiple devices to learn a global machine learning model while each individual device only has access to its local dataset. By avoiding the need for training data to leave individual users' devices, it enhances privacy and scalability compared to conventional cent...

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
