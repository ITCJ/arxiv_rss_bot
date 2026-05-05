# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-05 07:39:23 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Cross-Layer Energy Analysis of Multimodal Training on Grace Hopper Superchips](https://arxiv.org/abs/2605.01938)

**Authors**: Mahmoud Ahmed, Sameh Abdulah, Olatunji Ruwase, Sam Ade Jacobs, Mathis Bode, Mohamed Elhoseiny, David E. Keyes  
**Category**: cs.DC  
**Published**: 2026-05-05  
**Score**: 14.5  
**Type**: new  
**ArXiv ID**: 2605.01938v1  

#### Abstract
Multimodal deep learning models enable joint learning across heterogeneous data sources, including text, images, and video, but their rapid scaling introduces significant memory and communication bottlenecks. As model sizes and sequence lengths increase, training performance becomes increasingly imp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Cross-Layer Energy Analysis of Multimodal Training on Grace Hopper Superchips**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文聚焦于**多模态大模型训练中的能效瓶颈**，尤其是在 **NVIDIA Grace Hopper (GH200)** 这类紧耦合异构架构上的系统级能耗行为。随着模型规模和序列长度的增长，训练性能越来越受限于**数据移动（data movement）** 而非计算本身。尽管已有框架如 **DeepSpeed** 通过 CPU offload、activation checkpointing 和 sequence parallelism 缓解内存压力，但这些优化对系统能耗的影响尚不明确。

现有研究多关注吞吐量（throughput）和内存效率，而忽略了 **runtime 策略与硬件交互带来的能耗变化**，尤其在跨层（application–runtime–hardware）视角下缺乏系统性分析。

### **提出了什么新方法或新思路**
本文提出了一种 **cross-layer energy analysis framework**，从应用、运行时和硬件三层联合分析多模态训练的能耗与性能权衡。其核心思想是：
- 利用 GH200 的高带宽 CPU-GPU 互联（NVLink-C2C）和统一内存特性，量化不同 DeepSpeed 优化策略（offloading, checkpointing, sequence parallelism）对能耗的实际影响。
- 强调“**数据移动重于原始算力利用**”这一关键洞察，并据此提炼出面向能效的实践指南。

### **相比现有方法的优势**
- **首次在 GH200 上进行细粒度的跨层能耗分析**，揭示了传统性能指标（如 FLOP/s）无法反映的真实能耗动态。
- 发现 **异步执行（asynchronous execution）** 在能效上显著优于同步方式，即使时间开销相近。
- 提供了可操作的指导原则，帮助开发者在性能与能耗之间取得平衡，而非盲目追求高吞吐。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 使用 **LLaVA-Video-178K** 数据集的一个子集进行实验。
- 输入为视频帧序列与文本指令对，模拟典型的视频-语言多模态训练场景。

### **实验设置**
- **平台**：德国 Jülich 超算中心的 **JUPITER 超级计算机（Booster 模块）**，基于 **NVIDIA GH200 Superchip 架构**。
  - 每节点含 4 个 GH200 芯片，每个芯片集成 Grace CPU 与 Hopper GPU，通过 **NVLink-C2C（450 GB/s 单向）** 互连。
  - CPU 使用 LPDDR5 内存（~500 GB/s），GPU 使用 HBM3（~4 TB/s）。
  - 单芯片功耗上限为 **680W**（默认 CPU 功耗限制为 100W）。
- **模型配置**：
  - 基于 **Qwen2.5** 的语言骨干（LLM）
  - 视觉编码器采用 **SigLIP**
  - 投影网络为 MLP，支持 **3D RoPE** 处理时空位置
  - 模型参数规模：**7B、32B、72B**
- **训练框架**：**DeepSpeed**，启用以下优化技术：
  - **ZeRO-Offload / ZeRO-Infinity / SuperOffload**（异步优化器卸载）
  - **Activation Checkpointing**（激活检查点）
  - **Sequence Parallelism (SP=1,2,4)**（序列并行）
  - **Grouped Query Attention (GQA)** 减少 KV cache 开销

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Training Time (Wall Time)** | 总训练耗时（分钟） |
| **Throughput (TFLOP/s)** | 每秒浮点运算量，衡量硬件利用率 |
| **Average Power (W)** | 执行期间平均功耗（来自 hwmon 和 NVML） |
| **Energy (kJ)** | 总能耗 = ∫P(t)dt |
| **Energy Efficiency (TFLOP/kJ)** | 单位能量完成的有效计算量 |
| **Power Breakdown** | 分析 CPU、GPU、HBM/LPDDR、NVLink 等组件功耗分布 |

### **基线方法对比**
- **Baseline**：无任何 offloading 或 checkpointing
- **Synchronous vs Asynchronous Offloading**：
  - 同步：GPU 等待 CPU 完成优化器更新
  - 异步（SuperOffload）：利用 NVLink-C2C 重叠传输与计算
- **Sequence Parallelism Levels**：SP=1（无）、SP=2、SP=4
- **Scaling Modes**：
  - Fixed-GPU：固定 GPU 数量，比较不同策略
  - Scaled-GPU：增加 GPU 数量以提升 batch size 或降低负载

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **A. Optimizer Offloading（优化器卸载）**
| 模型 | 配置 | Energy Reduction | Wall Time ↓ | Throughput ↑ |
|------|-------|------------------|-------------|---------------|
| 7B | Async Offload | **12.7%** | 20.8 → 18.9 min | 428 → 449 TFLOP/s |
| 32B | Async Offload | **13.45%** | 19.6 → 16.8 min | 478 → 512 TFLOP/s |
| 72B | Async Offload | **9.76%** | 27.6 → 21.8 min | 465 → 495 TFLOP/s |

- **异步卸载比同步更优**：得益于更好的计算与通信重叠，功率曲线更平稳，idle 时间减少。
- **模块总功耗上升**，但因效率提高，**总能耗下降**。

#### **B. Activation Checkpointing（激活检查点）**
| 模型 | 配置 | Energy Reduction | Wall Time ↓ | Throughput ↑ |
|------|-------|------------------|-------------|---------------|
| 7B | Async Ckpt | 4.7% | 20.5 → 19.3 min | +2.8% |
| 32B | Async Ckpt | 10.9% | 19.4 → 18.0 min | ≈持平 |
| 72B | Async Ckpt | **13.2%** | 25.6 → 22.9 min | +7.0% |

- 能效增益随模型增大而增强，尤其在 **72B 模型上达到 16.5% 的 energy efficiency 提升**（见 Table I）。
- GPU 功耗基本不变，说明收益来自减少内存访问和 stall。

#### **C. Sequence Parallelism（序列并行）**
| 模型 | 配置 | Node Energy ↓ | Wall Time ↓ | Notes |
|------|--------|----------------|--------------|-------|
| 7B | SP=4 (scaled) | 3052 → 2568 kJ | 20.0 → 6.9 min | 通信开销小，收益明显 |
| 32B | SP=4 (scaled) | — | 18.8 → 5.9 min | TFLOP/s 下降（495→460），通信成本显现 |
| 72B | SP=4 (scaled) | 3434 → **939.5 kJ** | 25.4 → 8.3 min | 最大节能效果，适合长序列 |

- **结合 GPU scaling 后节能显著**，尤其对大模型。
- 但 **per-device 效率下降**，属于强扩展（strong scaling）策略。

#### **D. ZeRO Partitioning**
- 在 7B 模型上测试 ZeRO-1/2/3：
  - 小规模（4–16 GPUs）：浅层分区（ZeRO-1）能耗更低（通信少）
  - 大规模（32–64 GPUs）：ZeRO-3 优势显现，通信开销被摊薄
  - **64 GPUs 时各阶段能耗差异 <6%**，表明深度分区在大规模下可行

#### **E. Energy Efficiency（综合能效）**
| 方法 | 最佳能效增益（相对 baseline） |
|------|-------------------------------|
| **Activation Checkpointing (Async)** | **+16.5% @72B** |
| **Sequence Parallelism (SP=4, Scaled)** | **+32.4% @7B**, +5.3% @72B |
| **Optimizer Offloading (Async)** | +7.5% @7B, -4.8% @32B（部分负增益） |

> ✅ **结论**：**activation checkpointing + async execution 是最稳定高效的组合**。

---

## **4. 关键结论和发现**

### **主要发现**
1. 🔋 **能耗主要由数据移动决定，而非计算强度**  
   - 即使 GPU 利用率不高，频繁的数据搬运也会导致高能耗。
   - 内存层级间（HBM ↔ LPDDR）和 CPU-GPU 间的数据流动是主要能耗来源。

2. ⚙️ **异步执行（asynchronous execution）显著改善能效**  
   - 无论是 optimizer 还是 activation offloading，**异步模式均优于同步**。
   - 通过 **overlap computation with data transfer**，减少了 idle 周期，提升了 energy-to-solution。

3. 📈 **配置最优性能 ≠ 最优能效**  
   - 某些配置虽提升 throughput，却因额外通信或内存访问导致能耗上升（如 ZeRO-3 在中小规模）。
   - 必须进行 **cross-layer co-design**，兼顾 runtime 与硬件行为。

4. 💡 **offloading 是一种延迟隐藏机制（latency hiding），而非降功耗手段**  
   - offloading 不降低瞬时功耗，反而可能提高 GPU 和模块总功耗。
   - 其价值在于释放 GPU 内存、允许更大 batch，从而提升整体利用率。

5. 🔄 **sequence parallelism 对大模型极具价值，但需配合足够 GPU 资源**  
   - 在 scaled-GPU 设置下，SP 可大幅降低 wall time 和总能耗。
   - 但过度 partitioning（如 SP=4 on small models）会引入通信瓶颈。

6. 🖥️ **单纯增加 GPU 数量不能保证节能**  
   - 更多 GPU 会带来更高的通信开销和平台级功耗（interconnect, cooling）。
   - “最小化 time-to-solution” 不等于 “最小化 energy-to-solution”。

### **方法的局限性**
- 实验仅限于 **单一超算平台（JUPITER/GH200）**，结果在其他架构（如 AMD + Instinct）上是否普适有待验证。
- 未考虑 **端到端训练周期**（仅测量单 step 或短迭代），长期稳定性未评估。
- 没有深入建模 **coherence traffic** 和 **cache behavior** 对能耗的具体贡献。

### **未来工作方向**
- 扩展至更多异构架构（如 Intel Ponte Vecchio + CPU, Apple Ultra Fusion）进行对比分析。
- 构建 **能耗预测模型**，用于自动选择最优 runtime 配置。
- 探索 **compiler-level 优化**，进一步融合计算与通信调度。
- 研究 **绿色 AI 调度策略**，将能耗纳入分布式训练的任务编排中。

---

> ✅ **一句话总结**：  
> 在 GH200 等紧耦合异构系统上，**多模态训练的能效瓶颈不在算力而在数据移动**；通过 **异步 offloading、activation checkpointing 和适度的 sequence parallelism**，可在不牺牲性能的前提下实现高达 **13–16% 的能耗节省**，且最佳配置需根据模型规模动态调整。

</details>

---

### 2. [AAFLOW: Scalable Patterns for Agentic AI Workflows](https://arxiv.org/abs/2605.02162)

**Authors**: Arup Kumar Sarker, Mills Staylor, Aymen Alsaadi, Gregor von Laszewski, Shantenu Jha, Geoffrey Fox  
**Category**: cs.DC  
**Published**: 2026-05-05  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2605.02162v1  

#### Abstract
Agentic workflows in large language model systems integrate retrieval, reasoning, and memory, but existing frameworks suffer from scalability and reproducibility limitations due to fragmented data orchestration, serialization overhead, and non-deterministic execution. Although these frameworks incre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AAFLOW: Scalable Patterns for Agentic AI Workflows 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前的 **Agentic AI Workflows**（如基于 LLM 的检索增强生成 RAG 系统）虽然在灵活性和推理能力方面取得了进展，但在大规模部署时面临以下系统级瓶颈：

- **数据编排碎片化**：预处理、嵌入、检索、记忆管理等阶段依赖多次序列化和对象传递，导致显著的 I/O 和通信开销。
- **执行非确定性**：由 LLM 驱动的动态控制流使得执行路径难以复现、优化和分析，影响 **Reproducibility** 和 **可扩展性**。
- **缺乏统一执行模型**：现有框架（如 LangChain、AutoGen）将工作流视为黑盒调度任务，未与高性能计算（HPC）原则对齐。

这些问题限制了 Agentic 工作流在科学计算和生产环境中的高效、可靠运行。

---

### 提出了什么新方法或新思路

作者提出 **AAFLOW** —— 一个面向 Agentic AI 的统一分布式运行时系统，其核心思想是：

> 将 Agentic 工作流形式化为一组可组合的 **Operator 抽象**，并编译为高效的、通信感知的执行图（Execution DAG）。

#### 主要创新点包括：

1. ✅ **Agentic Operator Abstraction（代理操作符抽象）**
   - 定义五类标准 Operator：
     - `Opembed`（嵌入）
     - `Opretrieve`（检索）
     - `Opreason`（推理）
     - `Opmemory`（记忆）
     - `Opupsert`（向量索引更新）
   - 每个 Operator 映射到特定的分布式通信模式（如 Broadcast、Shuffle-Reduce、Embarrassingly Parallel），从而实现系统化的性能建模与优化。

2. ✅ **Unified Distributed Runtime（统一分布式运行时）**
   - 构建基于 **Apache Arrow** 和 **Cylon** 的零拷贝（Zero-Copy）数据平面，消除中间序列化开销。
   - 所有阶段共享内存格式（Arrow 表格），支持跨预处理、嵌入、检索的直接互操作。

3. ✅ **Resource-Deterministic Execution Model（资源确定性执行模型）**
   - 分离“逻辑行为”（agent 决策）与“物理执行”（资源调度），确保执行计划可预测、可复现。
   - 支持异步批处理（Asynchronous Batching）和细粒度资源控制。

4. ✅ **多层架构设计**
   - 包括 Operator Runtime、Zero-Copy Data Plane、Memory-Aware Retrieval Path 和 Asynchronous Batched Execution Engine 四个紧密耦合层次，共同支撑高并发、低延迟的端到端流程。

---

### 相比现有方法的优势

| 维度 | 传统框架（LangChain / Ray / Dask） | AAFLOW |
|------|-------------------------------|--------|
| 数据传输 | 多次序列化 + 对象存储中转 | Zero-Copy 内存共享（Arrow） |
| 执行模型 | 动态、非确定性控制流 | 编译为确定性 DAG |
| 调度机制 | 框架级任务调度（高协调成本） | Operator 级调度 + 异步批处理 |
| 可扩展性 | 强同步屏障导致扩展性差 | 流水线重叠 + 无阻塞执行 |
| 性能瓶颈 | 协调与 I/O 开销主导 | 接近计算极限 |

> 🔍 **核心优势总结**：AAFLOW 不加速 LLM 推理本身，而是通过优化 **数据流动、通信效率和执行调度** 来提升整体吞吐和响应速度。

---

## 2. 核心实验方法和设置

### 使用的数据集

- 使用合成数据集进行基准测试：
  - **256 份文档**，共生成 **32,768 tokens**
  - 更大规模实验使用 **10 million chunks**（来自 wikitext2_train 合成语料，约 92GB）
- 文件数量：4096 个输入文件，模拟真实场景下的小文件批量加载。

> 注：所有实验均不依赖真实 LLM 输出，而是用轻量级代理模型（如 `distilgpt2` 和 `LocalHashEmbedder`）替代，以排除 GPU 推理时间干扰，聚焦于 **数据管道性能**。

---

### 实验设置

- **硬件平台**：大型学术 HPC 集群，每节点 40 CPU 核心，最多使用 1024 workers（128–1024 并行度）。
- **向量数据库后端**：FAISS（用于知识索引和记忆索引）。
- **对比框架**：
  - **Agentic 框架**：LangChain、LangGraph、CrewAI、AutoGen
  - **分布式数据系统**：RayScalableRAG、DaskScalableRAG、AsyncParallelOnly、HigressRAG
- **评估维度**：
  - 端到端延迟（Total Latency）
  - 各阶段耗时（Load / Transform / Embed / Upsert）
  - Token Throughput（TPS）
  - 强缩放（Strong Scaling）与弱缩放（Weak Scaling）表现
  - 检索延迟与响应质量

---

### 评估指标

| 指标 | 描述 |
|------|------|
| `Total Runtime` | 整体流水线执行时间 |
| `Embed Time`, `Upsert Time` | 嵌入生成与向量写入耗时 |
| `TPS (Tokens/s)` | 系统级吞吐量 |
| `Speedup` | 相对于基线的加速比 |
| `Latency Reduction` | 检索与响应延迟下降百分比 |
| `Scaling Efficiency` | 增加 worker 数量时的性能增长趋势 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📊 表 I：端到端 RAG 流水线对比（256 文档）

| Framework | Total (s) | TPS (tokens/s) | Embed (s) | Upsert (s) |
|----------|-----------|----------------|-----------|------------|
| LangChain | 1.6447 | 94,823 | 1.1489 | 0.1403 |
| LangGraph | 1.6142 | 93,712 | 1.1396 | 0.1364 |
| CrewAI   | 1.6255 | 93,955 | 1.1453 | 0.1326 |
| AutoGen  | 1.6135 | 94,579 | 1.1362 | 0.1352 |
| **AAFLOW** | **0.8748** | **96,556** | **0.4856** | **0.0488** |

✅ **结论**：
- AAFLOW 实现 **1.88× 端到端加速**（vs. LangChain）
- **Embed 阶段提速 2.36×**
- **Upsert 阶段提速 2.8×**
- TPS 几乎持平 → 说明性能提升来自 **pipeline 优化而非模型加速**

---

#### 📊 表 II：大规模摄入流水线对比（10M chunks）

| Configuration | Total (s) | AAFLOW Boost |
|---------------|-----------|--------------|
| RayScalableRAG | 84.136 | 24.12× slower |
| AsyncParallelOnly | 11.641 | 3.33× slower |
| DaskScalableRAG | 16.188 | 4.64× slower |
| HigressRAG | 4.439 | 1.28× slower |
| **AAFLOW** | **3.487** | **1× (baseline)** |

✅ **结论**：
- 在强负载下仍保持领先，最高实现 **24.12× 加速**（vs. Ray）
- **Transform、Embed、Upsert 全面优于基线**

---

#### 🔬 检索与响应性能对比（Table III）

| Scenario | Engine | Total Latency (ms) | Improvement |
|---------|--------|--------------------|-------------|
| LLMG | HigressRAG | 68.23 | — |
|      | **AAFLOW** | **28.12** | ↓ **58.8%** |
| NCCQ | HigressRAG | 70.31 | — |
|      | **AAFLOW** | **30.18** | ↓ **57.1%** |
| HR   | HigressRAG | 21.45 | — |
|      | **AAFLOW** | **1.33** | ↓ **93.8%** |

✅ **原因分析**：
- 检索路径采用显式分区路由 + 减少冗余数据传输
- Zero-Copy 数据平面避免中间序列化和反序列化
- 缓存命中一致，表明收益来自执行效率而非缓存策略

---

#### 📈 可扩展性分析（Fig. 6–8）

- **强缩放**（固定数据量，增加 worker）：
  - AAFLOW 从 30.944s 降至 4.505s（>6.8×）
  - 接近线性扩展，远超其他框架
- **弱缩放**（按比例扩大数据与 worker）：
  - 延迟仅从 3.064s 升至 5.185s（温和上升）
  - 退化曲线更平缓，体现良好弹性

---

### 消融实验（隐含分析）

尽管未明确列出消融表，文中通过公式建模揭示各因素贡献：

$$
T \sim \frac{N\alpha}{bP} + \frac{N\beta}{P} + \Omega
$$

其中：
- $ \alpha $: 固定请求开销
- $ \beta $: 单项处理成本
- $ \Omega $: 框架额外开销（序列化、协调等）

AAFLOW 成功降低：
- $ \alpha $：通过持久 Worker 和微批次（micro-batch）
- $ \Omega $：通过 Zero-Copy + 显式通信原语
- 提升 $ b $ 和 $ P $：有效利用批处理与并行

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Agentic Workflow 的性能瓶颈不在 LLM 推理，而在数据编排与通信**  
   > “The main bottlenecks in large-scale agentic RAG systems are data orchestration, connectivity, and pipeline execution.”

2. ✅ **Operator 抽象 + 编译型执行模型可显著提升可复现性与性能**  
   - 将动态 agent 流程转化为确定性 DAG 是可行且高效的。

3. ✅ **Zero-Copy 数据平面是关键使能技术**  
   - Apache Arrow + Cylon 实现跨组件无缝数据流动，减少高达 90% 的序列化开销。

4. ✅ **异步批处理与流水线重叠极大掩盖 I/O 延迟**  
   - AAFLOW 总耗时甚至小于各阶段之和 → 证明阶段间完全重叠。

5. ✅ **AAFLOW 在不改变 LLM 的前提下实现高达 4.64× 的端到端加速**

---

### 方法的局限性

1. ❗ 当前评估集中在 CPU-bound 阶段，尚未集成 GPU 上的真实 LLM 推理（作者承诺未来补充）。
2. ❗ 依赖 FAISS 和 Cylon 生态，在非 Arrow 兼容系统中迁移成本较高。
3. ❗ Operator 抽象目前覆盖主流场景，但复杂工具调用或多模态扩展尚待验证。
4. ❗ 实验基于合成负载，真实业务场景下的稳定性需进一步测试。

---

### 未来工作方向（Future Experiments）

1. **Operator 级微基准测试**：分离每个 Operator 的性能贡献，验证通信模式有效性。
2. **多后端支持评估**：测试 ChromaDB、Pinecone 等不同 Vector DB 的影响。
3. **Memory Operator 影响分析**：量化记忆管理带来的开销与收益。
4. **通信开销剖析**：直接测量序列化、网络传输、协调延迟，精确定位 $ \Omega $。
5. **重复性与确定性评估**：分析多次运行间的方差，验证执行一致性。
6. **GPU 集群实验**：引入真实 LLM 推理，全面评估端到端性能。

---

## 总结

> 🔚 **AAFLOW 的本质突破在于：它不是另一个 Agentic 框架，而是一个将 Agentic 工作流“系统化”、“工程化”、“HPC 化”的基础设施重构。**

通过引入 **Operator 抽象**、**Zero-Copy 数据平面** 和 **资源确定性调度**，AAFLOW 成功将原本松散、不可控的 agent 流程转变为高效、可复现、可扩展的分布式程序，为构建下一代高性能 Agentic AI 系统提供了坚实基础。

</details>

---

### 3. [SplitZip: Ultra Fast Lossless KV Compression for Disaggregated LLM Serving](https://arxiv.org/abs/2605.01708)

**Authors**: Yipin Guo, Siddharth Joshi  
**Category**: cs.DC  
**Published**: 2026-05-05  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2605.01708v1  

#### Abstract
Contemporary systems serving large language models (LLMs) have adopted prefill-decode disaggregation to better load-balance between the compute-bound prefill phase and the memory-bound decode phase. Under this design, prefill workers generate a KV cache that must be transferred to decode workers bef...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《SplitZip: Ultra Fast Lossless KV Compression for Disaggregated LLM Serving》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在 **Prefill-Decoding (PD) disaggregation** 架构中，LLM 推理被划分为计算密集型的 **prefill 阶段** 和内存带宽受限的 **decode 阶段**。这两个阶段通常部署在不同物理节点上，因此需要将 prefill 生成的 **KV Cache** 跨节点传输至 decode worker。

然而，KV Cache 数据量巨大（尤其在长上下文场景下），其跨节点传输成为系统瓶颈。现有 **lossless 压缩方法**（如 Huffman 编码）虽然压缩率高，但编码/解码效率低，难以满足在线、低延迟的传输需求。

### **提出了什么新方法或新思路**
提出 **SplitZip** —— 一种专为 **disaggregated LLM serving** 设计的 **GPU-friendly、超高速 lossless KV Cache 压缩器**，其核心思想是：

- 利用 **BF16 浮点数中 exponent 字段的高度冗余性**：实验发现，KV Cache 中约 99% 的 exponent 值来自前 16 个高频值。
- 采用 **固定长度编码（fixed-length coding）** 对 top-16 exponent 进行 4-bit 编码，两个编码打包成一个字节，实现高效并行处理。
- 将非 top-16 的“稀有 exponent”作为 **escape 值**，单独记录其位置和原始值，形成稀疏修正流（sparse escape correction）。
- 整个流程分为 **dense path（主路径）** 和 **sparse correction（逃逸修正）**，前者高度规则、适合 GPU 并行执行，后者开销极小。

### **相比现有方法的优势**
| 维度 | SplitZip | 现有方法（如 Huffman、nvCOMP） |
|------|----------|-------------------------------|
| **压缩速度** | 极高（613.3 GB/s） | 低，CPU 侧或串行依赖强 |
| **解压速度** | 极高（2181.8 GB/s） | 受限于 variable-length coding 的串行解码 |
| **GPU 友好性** | 完全适配 GPU 并行架构 | 多为 CPU 实现或未优化 GPU 执行 |
| **是否 lossless** | 是，bit-exact 恢复 | 多数是，但 SplitZip 更快 |
| **适用场景** | 在线、低延迟 KV 传输 | 多用于离线权重压缩 |

> ✅ **核心优势**：**在保持 lossless 的前提下，极大提升了 codec 吞吐量，特别适合 latency-critical 的 KV 传输路径**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 主要使用真实 **BF16 格式的 KV Cache 激活张量**，来源于多个主流 LLM：
  - **Qwen3-32B**, **Qwen3-30B-A3B**, **Llama-3-8B**, **Llama-3.1-70B**, **Phi-2** 等。
- 校准（calibration）阶段使用 **WikiText-2** 训练集构建 exponent codebook。
- 评估覆盖多种输入分布：**WikiText-2**, **HumanEval**, **GSM8K**, **MMLU**, **PTB**，验证 codebook 泛化能力。

### **实验设置和评估指标**
- **硬件平台**：
  - NVIDIA H200 GPU
  - Intel Xeon Platinum 8468 CPU
  - Mellanox ConnectX-7 InfiniBand / RoCE 网络
- **评估指标**：
  - **Compression/Decompression Throughput (GB/s)**：衡量 codec 性能。
  - **Compression Ratio (CR)**：压缩后大小 / 原始大小。
  - **End-to-end Transfer Time**：KV Cache 从 encode → transfer → decode 的总耗时。
  - **TTFT (Time to First Token)** 和 **Request Throughput**：端到端服务性能。
- **集成框架**：在 **SGLang** 和 **Mooncake** 中集成 SplitZip，进行真实部署测试。

### **基线方法对比**
| 方法 | 类型 | 是否 GPU 友好 | 是否 lossless |
|------|------|----------------|---------------|
| **nvCOMP LZ4 / Cascaded / Bitcomp** | 通用 GPU 压缩库 | 是 | 是 |
| **DFloat11** | BF16 权重压缩（Huffman） | 部分 | 是 |
| **ZipNN** | Huffman-based 权重压缩 | 否（CPU） | 是 |
| **ZipServ** | 多级 fixed-length 编码 | 是（Kernel 版） | 是 |
| **Falcon** | GPU 优化压缩框架 | 是 | 是 |
| **SplitZip (Ours)** | 固定长度 + escape | ✅ 高度 GPU 友好 | ✅ 是 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | SplitZip 结果 |
|------|----------------|
| **Compression Throughput** | **613.3 ± 2.6 GB/s** |
| **Decompression Throughput** | **2181.8 ± 38.5 GB/s** |
| **平均 Compression Ratio (BF16)** | **1.32×**（最高达 1.33×） |
| **Escape Rate** | < 0.2%（top-16 覆盖率 > 99.8%） |

### **与基线方法的对比结果**
#### **Codec 吞吐量对比（Table 2）**
| 方法 | Encode (GB/s) | Decode (GB/s) | Ratio |
|------|----------------|----------------|--------|
| nvCOMP Bitcomp | 341.5 | 147.7 | 1.056× |
| DFloat11 | 0.004 | 468.2 | 1.423× |
| ZipServ-Kernel | N/A | 1260.9 | 1.236× |
| **SplitZip** | **613.3** | **2181.8** | **1.324×** |

- **编码速度**：比最快 nvCOMP 方法（Bitcomp）快 **1.8×**，比 ZipServ 快 **>6000×**（CPU 版）。
- **解码速度**：比 ZipServ GPU Kernel 快 **1.7×**，比 nvCOMP Cascaded 快 **14.1×**。
- **压缩比**：优于 ZipServ (+7.1%)，略低于 DFloat11 (-7.0%)，但吞吐量远胜。

#### **端到端传输加速**
- 在 **Mooncake** 框架下，KV 传输时间减少：
  - **Llama-3-8B**：最高 **1.32×** 加速
  - **Qwen3-30B-A3B**：最高 **1.32×** 加速
- 在 **SGLang** 中端到端测试：
  - **TTFT 加速最高达 1.30×**
  - **Request Throughput 提升最高达 1.23×**

> 📌 **结论**：SplitZip 的 codec 优势成功转化为 **end-to-end 性能提升**，尤其在长序列场景下效果显著。

### **消融实验结果**

#### **(1) Top-8 vs Top-16 编码（Table 3）**
| 指标 | Top-8 (3-bit) | Top-16 (4-bit) |
|------|----------------|-----------------|
| 覆盖率 | 92.11% | 99.84% |
| Escape Rate | 7.89% | 0.16% |
| Compression Ratio | 1.038× | **1.324×** |
| Decode Throughput | 710.5 GB/s | **2181.8 GB/s** |

> ❌ Top-8 虽节省 bit 数，但 escape 开销剧增，且 3-bit 不对齐字节，导致性能大幅下降。

#### **(2) 显式位置 vs Sentinel 逃逸标记（Table 6）**
| 指标 | Top-16 + Pos | Top-15 + Sentinel |
|------|----------------|--------------------|
| Decode Throughput | **2181.8 GB/s** | 620.8 GB/s |
| 原因 | 统一 dense 解码 | 引入控制流分支，破坏并行性 |

> ✅ SplitZip 选择 **显式存储 escape 位置**，以微小 metadata 开销换取 **3.5× 解码加速**。

#### **(3) 预校准 vs 动态校准（Table 7）**
| 指标 | 预校准 | 动态校准 |
|------|--------|----------|
| Encode Throughput | **613.3 GB/s** | 80.7 GB/s |
| 原因 | 无需在线 histogram | 需动态统计 top-16，引入瓶颈 |

> ✅ **预校准 codebook 泛化性好**，无需每轮重建。

#### **(4) 分层覆盖率分析（Figure 5）**
- 固定 codebook 在所有 64 层中：
  - **K Cache**：全部 > 99.0%，61 层 > 99.8%
  - **V Cache**：最差层 98.77%（escape rate 1.23%）
- 表明 **单个全局 codebook 足够稳定**，无需 per-layer 或 per-token 校准。

---

## **4. 关键结论和发现**

### **主要发现**
1. **KV Cache 的 BF16 exponent 具有高度集中性**：top-16 exponent 覆盖率普遍 > 99%，为 fixed-length 编码提供理论基础。
2. **fixed-length coding + sparse escape 是 GPU 上最优路径**：相比 Huffman 等 variable-length 方法，更适合并行执行，吞吐量更高。
3. **SplitZip 实现了 lossless 与 ultra-fast 的统一**：在不损失任何精度的前提下，显著加速 KV 传输。
4. **端到端收益明显**：在真实 serving 框架中，带来 **1.3× 左右的 TTFT 和吞吐提升**。

### **方法的局限性**
- **压缩比受限于 exponent 冗余度**：最大压缩比约为 1.33×，无法进一步突破。
- **仅适用于 exponent 冗余明显的格式**：如 BF16、FP8-E5M2；对 E4M3 等紧凑格式提升有限（见 Appendix B）。
- **短序列场景增益有限**：当传输本身不是瓶颈时，codec 开销可能抵消收益。

### **未来工作方向**
- **结合 lossy 方法**：在 SplitZip 基础上叠加 quantization 或 pruning，实现更高压缩比。
- **支持更多数据类型**：扩展至 FP8、INT8 等新兴低精度格式。
- **更智能的 escape 编码**：对 escape 流进一步压缩（如差分编码）。
- **跨模型通用 codebook**：探索是否可训练一个通用 exponent codebook，避免 per-model 校准。

---

> ✅ **总体评价**：SplitZip 是首个专为 **disaggregated LLM serving** 中 **KV Cache 传输** 优化的 **GPU-native lossless 压缩器**，通过简洁而高效的 **fixed-length exponent coding + escape correction** 设计，在保持 bit-exact 精度的同时实现了前所未有的 codec 吞吐量，具有很强的实用价值和集成潜力。

</details>

---

### 4. [PipeMax: Enhancing Offline LLM Inference on Commodity GPU Servers](https://arxiv.org/abs/2605.02189)

**Authors**: Hongbin Zhang, Taosheng Wei, Jiazhi Jiang, Hui Yan, Jiangsu Du, Zhiguang Chen  
**Category**: cs.DC  
**Published**: 2026-05-05  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.02189v1  

#### Abstract
Offline LLM inference seeks to maximize request processing under fixed budgets, making commodity GPU servers a promising choice. However, prior work typically considers offloading and parallelism in isolation, resulting in suboptimal performance. In this paper, we propose PipeMax, a high-throughput ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PipeMax: Enhancing Offline LLM Inference on Commodity GPU Servers

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在**离线大语言模型（LLM）推理**场景中，目标是在固定预算下最大化请求处理吞吐量。然而，当前主流系统在消费级 GPU 服务器（如无 NVLink 的 RTX 5090）上面临两大瓶颈：
- **内存限制**：KV Cache 占用巨大显存，限制了 batch size。
- **通信开销高**：Tensor Parallelism 在低带宽互连设备上因频繁 all-reduce 操作导致性能低下。

现有方法通常孤立地使用 **offloading** 或 **parallelism**，未能充分利用硬件资源，尤其在 decode 阶段存在严重的“反局部性”（anti-locality）——多个 decode batch 的 KV Cache 被保留在 GPU 显存中但长期闲置，造成有效显存利用率低下。

---

### 提出了什么新方法或新思路
本文提出 **PipeMax**，一个面向**商品化 GPU 服务器**的高吞吐 LLM 推理系统，其核心思想是：

> **将 Pipeline Parallelism 与 Offloading 深度结合，在 decode 阶段主动卸载非活跃 batch 的 KV Cache 到 CPU 内存，从而扩展有效的 GPU 显存容量，并维持大 batch 执行。**

#### 主要创新机制包括：
- **Decode 阶段的 KV Cache 动态卸载与预取**  
  利用 pipeline parallelism 天然特性：每个 GPU 同时只执行一个 decode batch。因此可将其他 inactive batch 的 KV Cache offload 至 CPU，仅在需要时 prefetch 回 GPU。
  
- **计算与数据移动的协同调度（Compute-Prefetch Overlap）**  
  设计了一个**集中式引擎**，包含：
  - **Decode 执行时间估计器**：基于公式 $ T = \alpha \cdot b + \beta \cdot L + \delta $ 离线建模预测每轮 decode 时间。
  - **Prefetch-aware 调度器**：动态决定从 CPU 预取哪些请求，以实现计算与 prefetch 的高效重叠，并保持各 batch 间负载均衡。

- **传输高效的 KV Cache 引擎**
  - **Block-first 布局**：改变传统 PagedAttention 的 layer-first 存储方式，使同一 block 内所有层的 KV Cache 连续存储，提升 prefetch 时的 PCIe 带宽利用率。
  - **异步 CPU 辅助 offloading**：在 QKV 投影后立即异步将原始 KV tensor 传至 CPU，并在 CPU 端重组为 block-first 格式。
  - **优先级感知的数据调度（Demand-Priority Transfer Orchestration）**：通过多优先级队列确保关键路径上的 activation transfer 不受 prefetch/offload 干扰。

---

### 相比现有方法的优势
| 方法 | 局限性 | PipeMax 如何改进 |
|------|--------|------------------|
| **vLLM (TP)** | All-reduce 开销大，不适合低带宽节点 | 改用 PP 减少通信 |
| **vLLM (PP)** | 缺乏负载均衡，decode 阶段严重不平衡 | 加入 work stealing + prefetch-aware 调度 |
| **TD-Pipe** | 保留多个 decode batch 的完整 KV Cache，浪费显存 | 卸载 inactive batch 的 KV Cache，显著提升有效显存 |
| **Seesaw** | decode 使用 TP，依赖 high-bandwidth interconnect | 完全避免 decode 中的 all-reduce |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **ShareGPT**：输入输出长度较均衡，模拟通用对话任务。
- **LongBench**：长输入、短输出，测试对长上下文的处理能力。

| 数据集 | 输入平均长度 | 输出平均长度 |
|-------|-------------|------------|
| ShareGPT | 343.76 | 237.20 |
| LongBench | 2686.89 | 101.78 |

---

### 实验设置和评估指标

#### 硬件平台
| 节点 | GPU 配置 | PCIe 版本 | 是否有 NVLink |
|------|---------|----------|--------------|
| Node 1 | 8×RTX 5090 | PCIe 5.0 (64 GB/s) | ❌ |
| Node 2 | 8×L20 | PCIe 4.0 (32 GB/s) | ❌ |
| Node 3 | 8×H100 | PCIe 5.0 + NVLink | ✅（用于对比参考） |

#### 模型配置
| 节点 | 模型 |
|------|------|
| RTX 5090 / L20 | LLaMA2-70B, Mixtral-8×7B |
| H100 | Qwen3-235B-A22B |

---

#### 评估指标
- **Overall Throughput**：总吞吐量（tokens/sec），为主要性能指标。
- **Normalized Throughput**：相对于 vLLM (TP) 的加速比。
- **PCIe Bandwidth Utilization**：衡量 KV Cache 传输效率。
- **Execution Time Breakdown**：分析计算、prefetch、offload 的时间占比。

---

#### 基线方法对比
| 基线 | 描述 |
|------|------|
| **vLLM (TP)** | 张量并行，广泛使用的基准 |
| **vLLM (PP)** | 流水线并行，作为基础对照 |
| **TD-Pipe** | 当前最先进的 PP 优化系统，采用时间解耦和 work stealing |
| **Seesaw** | Prefill 用 PP，decode 用 TP 的 re-sharding 方法 |

所有基线均基于 vLLM v0.7.3 实现，保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **8-GPU 商品化服务器**上，PipeMax 取得了显著的吞吐提升：

| 对比对象 | 最高加速比 |
|--------|-----------|
| vLLM (TP) | **2.45×** |
| vLLM (PP) | **2.51×** |
| TD-Pipe | **1.42×** |
| Seesaw | **1.38×** |

> 📌 **特别说明**：即使在高端 H100 + NVLink 平台上，PipeMax 仍优于 Seesaw 和 TD-Pipe，表明其设计也适用于数据中心环境。

---

### 与基线方法的对比结果
- **vs vLLM (TP)**：由于避免了 all-reduce，通信开销大幅降低，尤其在 PCIe 4.0 的 L20 上优势更明显。
- **vs vLLM (PP)**：解决了 decode 阶段严重的 inter-batch imbalance 问题，通过 prefetch-aware 调度实现稳定的大 batch 执行。
- **vs TD-Pipe**：虽然都使用 PP，但 PipeMax 通过 offloading inactive KV Cache 显著提升了有效显存容量，允许更大规模的并发 decode。
- **vs Seesaw**：Seesaw 在 decode 阶段依赖 all-reduce，在带宽受限环境下成为瓶颈；而 PipeMax 全程无需跨 GPU 参数同步。

---

### 消融实验结果

#### （1）Decode 执行时间估计器准确性
- 在连续 100 步 decode 中，预测时间与实际时间误差：
  - **超过 90% 的样本误差 < 5%**
  - 最坏情况偏差 < 8%
- 表明该估计器足够准确，可用于指导 prefetch 决策。

#### （2）Prefetch-aware 调度器有效性
- 与静态 prefetch 比较（固定比例预取 5%-25% GPU 显存）：
  - PipeMax 在所有场景下均优于静态策略。
  - 动态调度能更好适应运行时变化，避免 prefetch 不足或阻塞执行。

#### （3）Block-first KV Cache 布局效果
- **PCIe 带宽利用率**：
  - Layer-first：约 **30%**
  - **Block-first（PipeMax）**：接近 **90%**
- 显著提升 prefetch 效率，支持更大规模的 KV Cache 加载。

#### （4）异步 Offloading 重叠效果
- 在 prefill 和 decode 阶段，KV Cache 的 offloading 完全被 attention 和 FFN 计算掩盖。
- 即使输入长度达 256 tokens，offload 时间也被完全隐藏，不增加端到端延迟。

#### （5）运行时动态行为
- 图 18 显示 decode 阶段经历 warm-up 后进入 steady phase：
  - 执行时间趋于稳定
  - 每个 batch 的 **prefetched KV Cache 占据大量 GPU 显存**
  - 总体 KV Cache 使用量远超单卡容量 → 成功实现了显存扩展

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Pipeline Parallelism 是商品化 GPU 上最优选择**：相比 Tensor Parallelism，其通信模式更适合低带宽环境。
2. **Decode 阶段存在巨大显存浪费**：inactive batch 的 KV Cache 长期驻留 GPU，是性能瓶颈根源。
3. **Offloading + PP 可深度协同**：利用 PP 的阶段性激活特性，精准卸载 inactive KV Cache，实现“按需加载”，极大提升有效显存。
4. **动态 prefetch-aware 调度至关重要**：必须联合考虑执行时间和 prefetch 可行性，才能实现稳定的高吞吐。
5. **存储布局严重影响传输效率**：block-first layout 可大幅提升 PCIe 利用率，是高性能的前提。

---

### 方法的局限性
- **依赖较强的 CPU-GPU 带宽**：若 PCIe 带宽极低（如 PCIe 3.0），prefetch 可能不能及时完成，影响性能。
- **调度复杂度较高**：集中式控制器需实时监控状态并做出决策，可能引入额外控制开销。
- **未考虑异构设备混合部署**：目前假设所有 GPU 规格一致。

---

### 未来工作方向
- 将 PipeMax 思想推广至 **MoE 模型**推理。
- 探索 **更细粒度的 offload 策略**，例如按 token 或 layer 级别动态迁移。
- 结合 **prefix sharing** 等技术进一步压缩 KV Cache。
- 支持 **异构 GPU 集群**下的分布式调度。

---

> ✅ **总结一句话**：  
> **PipeMax 通过将 Pipeline Parallelism 与智能 KV Cache Offloading 相结合，在无需高端互连的商品化 GPU 服务器上实现了高达 2.51× 的吞吐提升，为低成本、高吞吐 LLM 推理提供了新的可行路径。**

</details>

---

### 5. [Towards Systematic Generalization for Power Grid Optimization Problems](https://arxiv.org/abs/2605.02026)

**Authors**: Zeeshan Memon, Yijiang Li, Hongwei Jin, Kibaek Kim, Liang Zhao  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.02026v1  

#### Abstract
AC Optimal Power Flow (ACOPF) and Security-Constrained Unit Commitment (SCUC) are fundamental optimization problems in power system operations. ACOPF serves as the physical backbone of grid simulation and real-time operation, enforcing nonlinear power flow feasibility and network limits, while SCUC ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Towards Systematic Generalization for Power Grid Optimization Problems**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
电力系统运行中存在多种异构优化问题，如 **AC Optimal Power Flow (ACOPF)** 和 **Security-Constrained Unit Commitment (SCUC)**。尽管它们共享相同的电网拓扑和物理规律，但传统学习方法通常将它们**孤立建模**，导致模型无法跨任务迁移、表示冗余且难以推广到耦合问题（如 UC-ACOPF）。

此外，在实际应用中，需要在**未见过的电网拓扑**上快速生成可行解，并处理混合整数非线性耦合约束，这对模型的**系统性泛化能力**提出了挑战。

---

### **提出的新方法与新思路**
本文提出一种**联合学习框架**，通过以下设计实现系统性泛化：

- **共享图编码器（Shared Graph-based Encoder）**  
  构建一个基于 **Heterogeneous Graph Transformer (HGT)** 的共享空间编码器，统一建模电网的拓扑结构和物理交互，捕捉跨任务不变的空间特征。

- **任务特定解码器（Task-Specific Decoders）**  
  - 对于 **ACOPF**：静态连续决策 → 解码电压、相角、有功/无功功率。
  - 对于 **SCUC**：时序混合整数决策 → 结合时间嵌入和 Temporal Transformer 解码启停状态和调度轨迹。

- **多任务训练目标**  
  联合优化监督损失（来自求解器）与**物理信息正则项**（physics-informed penalties），强制满足 AC 功率平衡、线路热限、爬坡约束等。

- **无监督适配到 UC-ACOPF**  
  在冻结共享编码器的前提下，仅用**无监督物理约束目标**和**共识机制（consensus mechanism）** 微调解码器，适应更复杂的 **UC-ACOPF** 耦合问题。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **建模方式** | 单独训练独立模型 | 联合训练，共享表示 |
| **可迁移性** | 难以迁移到新任务或新网络 | 支持跨案例零样本迁移和系统性泛化 |
| **耦合问题处理** | 需重新标注、重训练 | 冻结主干，仅微调解码器即可适应 UC-ACOPF |
| **可行性保障** | 依赖监督标签或后处理 | 物理正则 + 共识机制直接优化可行性 |
| **推理速度** | 快于求解器但精度受限 | 推理快 2–3 个数量级，接近最优 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ACOPF**: 使用 **OPFData** 数据集，包含多个标准电网（Case-14, 30, 57, 118），每个提供约 300K 个由工业级求解器生成的可行运行点。
- **SCUC**: 使用 **UnitCommitment.jl** 提供的标准测试实例，源自 MATPOWER 测试网络，覆盖全年 365 天、每 36 小时滚动规划，含完整机组特性与时间约束。
- **UC-ACOPF**: 自行构建，基于 IEEE 标准网络生成 36 小时负荷曲线，使用历史 ISO-NE 数据缩放需求，并乘以折扣因子 0.7 以避免全机组上线。

---

### **实验设置与评估指标**

#### **评估场景**
1. **单任务性能**：在 Case-118 上比较 ACOPF 和 SCUC 性能。
2. **跨案例泛化（Zero-shot Transfer）**：在小网（如 14→30）上训练，在大网（如 118）上测试。
3. **UC-ACOPF 系统性泛化**：冻结编码器，仅微调解码器，评估无监督适配效果。
4. **消融实验**：验证共享投影层与异质图建模的作用。

#### **评估指标**

| 任务 | 指标 | 含义 |
|------|------|------|
| **ACOPF** | `MSEBus`, `MSEGen` | 总线/发电机预测误差 |
|         | `PF Viol.` | 功率平衡 RMSE |
|         | `Viol. Norm` | 归一化总约束违反量 |
|         | `Opt. Gap` | 成本相对于最优解的差距 |
| **SCUC** | `Acc` | 启停决策准确率 |
|         | `RMSE(Pg)` | 发电调度均方根误差 |
|         | `%Viol.` | UC 约束违反比例 |
|         | `Opt. Gap` | 成本间隙 |
| **通用** | `Inference Time` / `Speedup` | 推理耗时及相对于求解器的加速比 |

---

### **基线方法对比**
- **ACOPF Baselines**:
  - CANOS [28]
  - GAT-based model [18]
  - Heterogeneous Graph Transformer (HGT)
- **SCUC Baselines**:
  - STModel [31]: GCN + LSTM
  - MSTT [20]: 多尺度时空 Transformer
- **对比策略**：控制变量实验，比较“单任务训练” vs “联合训练”。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **表 1 & 表 2：Case-118 上的 ACOPF 与 SCUC 性能**

| 方法 | SCUC Acc ↑ | RMSE(Pg) ↓ | %Viol. ↓ | Opt. Gap ↓ |
|------|------------|-----------|----------|------------|
| STModel | 85.19% | 0.28 | 0.50% | 10.72% |
| MSTT | 80.40% | 0.16 | 0.03% | 11.98% |
| **Ours (Joint)** | **88.88% (+14.3%)** | **0.11 (-31.3%)** | 0.07% | **8.86% (-17.3%)** |

| 方法 | ACOPF MSEBus ↓ | MSEGen ↓ | PF Viol. ↓ | Opt. Gap ↓ |
|------|----------------|----------|-----------|-------------|
| CANOS | 0.010 | 0.030 | 0.25 | 0.90% |
| GAT | 0.005 | 0.020 | 0.15 | 2.30% |
| HGT | 0.003 | 0.020 | 0.04 | 1.90% |
| **Ours (Joint)** | **0.002 (-33.3%)** | 0.020 | 0.09 | **0.80%** |

> 📌 **结论**：联合训练显著提升预测精度与经济性，尤其在 SCUC 准确率和 ACOPF 总线误差方面领先。

---

#### ✅ **表 3：跨案例零样本泛化能力（Zero-shot Generalization）**

| 训练→测试 | SCUC Acc | %Viol. | ACOPF MSEBus | Opt. Gap |
|----------|----------|--------|---------------|----------|
| 14 → 30 | 84.28% | 0.11% | 0.06673 | -2.98% |
| 14+30 → 57 | 83.52% | 0.10% | 0.04098 | -4.44% |
| 14+30+57 → 118 | 81.80% | 0.53% | 0.00635 | +0.26% |

> 📌 **结论**：即使未在目标网络上训练，模型仍保持较高准确性与低约束违反，表明共享编码器具备良好的拓扑不变性。

---

#### ✅ **表 4：UC-ACOPF 上的无监督微调 vs 从头训练**

| Case | 设置 | PF Viol. ↓ | RMSE(P) ↓ | %UC Viol. ↓ | Cost Gap ↓ |
|------|------|------------|------------|--------------|-------------|
| 118 | Scratch (50ep) | 1.55 | 0.83 | 1.9% | 5.2% |
| 118 | Finetuned (10ep) | **0.18** | **0.15** | **1.3%** | **4.9%** |

> 📌 **结论**：仅用 1/5 的训练轮次，冻结编码器的微调即大幅降低物理约束违反，说明预训练表示已蕴含 UC-ACOPF 所需结构信息。

---

#### ✅ **图 3：推理效率对比**

| 模型 | Case-118 推理时间 | 相对于 Juniper 求解器的加速比 |
|------|--------------------|-------------------------------|
| Juniper (MINLP Solver) | ~2400 秒 | 1× |
| **Ours (Proposed)** | **~6 秒** | **400×** |

> 📌 **结论**：神经模型实现超实时推理，适用于大规模或紧急调度场景。

---

#### ✅ **表 5：消融实验（Ablation Study）**

| 变体 | SCUC Acc | %Viol. | PF Viol. | Opt. Gap (SCUC) | Opt. Gap (ACOPF) |
|------|----------|--------|----------|------------------|-------------------|
| Ours (Full) | 88.88% | 0.07% | 0.09 | 8.86% | 0.80% |
| - Shared Projection | 86.83% | 0.21% | 0.10 | 8.84% | **4.11%** ❗ |
| - Heterogeneous Modeling | 85.40% | 0.50% | **0.29** ❗ | 9.12% | 2.20% |

> 📌 **结论**：
> - **共享投影层**对 ACOPF 成本敏感，缺失会导致严重退化；
> - **异质图建模**对所有任务均有重要影响，忽略节点/边类型会削弱表达力。

---

## **4. 关键结论和发现**

### **主要发现**
1. **共享图编码器有效支持系统性泛化**  
   通过联合建模 ACOPF 与 SCUC，模型学习到了电网的通用物理结构表示，可在不同任务、不同规模网络间迁移。

2. **无监督适配即可应对复杂耦合问题**  
   在不访问 UC-ACOPF 最优解的情况下，仅通过物理约束与共识机制微调解码器，即可获得高质量近似解，验证了“先学结构、再适配任务”的可行性。

3. **联合训练优于单任务训练**  
   不仅提升性能，还增强鲁棒性和泛化能力，尤其体现在跨网络迁移和约束满足上。

4. **推理速度快，适合工程部署**  
   达成数百倍于传统求解器的速度，为实时调度、应急响应等场景提供了实用替代方案。

---

### **方法的局限性**
- **适用范围有限**：当前框架聚焦于共享同一电网结构的问题（如 ACOPF/SCUC/UC-ACOPF），扩展至完全不同结构的任务（如拓扑控制、故障恢复）尚需研究。
- **依赖高质量训练数据**：虽然 UC-ACOPF 微调是无监督的，但初始多任务训练仍需大量求解器生成的标签。
- **理论保证较弱**：尽管有理论分析（如 Theorem 3.2），但仍为近似可行性，不能完全替代严格验证。

---

### **未来工作方向**
1. **构建真正的 Grid Foundation Model**  
   引入自监督预训练（如掩码重建、对比学习），减少对监督信号的依赖。
2. **扩展至更多任务**  
   如状态估计（SE）、暂态稳定评估（TSA）、拓扑优化等，形成统一的 AI4Grid 平台。
3. **引入不确定性建模**  
   融合概率输出或贝叶斯神经网络，处理新能源波动下的鲁棒调度。
4. **硬件协同部署**  
   探索边缘设备上的轻量化部署与在线增量学习机制。

---

> 🔚 **总结一句话**：  
> 本文首次实现了 **ACOPF 与 SCUC 的联合建模与系统性泛化**，通过**共享图编码器 + 任务专用解码器 + 无监督适配**，展示了深度学习在复杂电力优化问题中的强大迁移潜力，为迈向“电网基础模型”迈出关键一步。

</details>

---

### 6. [Efficient Accelerated Graph Edit Distance Computation on GPU](https://arxiv.org/abs/2605.00830)

**Authors**: Adel Dabah, Andreas Herten  
**Category**: cs.DC  
**Published**: 2026-05-05  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.00830v1  

#### Abstract
Graph representation is a powerful abstraction of real-world objects and relations. Computing the Graph Edit Distance (GED) between graphs is critical in domains such as bioinformatics, machine learning, and pattern recognition. GED measures the minimum number of edit operations required to transfor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Efficient Accelerated Graph Edit Distance Computation on GPU》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
图编辑距离（Graph Edit Distance, GED）是衡量两个图之间差异的关键度量，在生物信息学、模式识别和机器学习等领域有广泛应用。然而，由于 GED 问题是 NP-hard，最优算法（如 A-Star、DFS）在大规模图上计算成本极高，而近似方法（如 Beam Search、K-Best）则面临**准确性与可扩展性之间的权衡**。传统 CPU 实现难以满足实际应用对速度和精度的双重需求。

### **提出了什么新方法或新思路**
本文提出 **FAST-GED**——一个基于 GPU 加速的开源框架，用于高效且可扩展地计算 GED。其核心思想是将经典的 K-Best 搜索策略重构为适合 GPU 并行架构的算法设计，具体创新包括：

- **GPU 友好的搜索树遍历机制**：采用类广度优先搜索（BFS-like）方式逐层扩展搜索树，每层保留最优的 K 个节点。
- **三阶段 GPU 内核流水线**：
  1. **Branching Kernel**：每个 GPU block 扩展一个父节点，生成所有子节点并计算 Partial Edit Distance (PED)。
  2. **Ranking Kernel**：通过两级排序（块内局部排序 + 原子操作全局筛选）快速选出 Top-K 节点，避免全排序开销。
  3. **Update Kernel**：在设备端更新数据结构，完全避免 Host-Device 数据传输，提升可扩展性。
- **隐式边操作处理优化**：利用共享内存缓存已编辑顶点信息，高效计算由顶点映射引发的边替换/插入/删除代价。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **性能** | 相比 CPU 版本实现最高达 **300× 加速**；相比 NetworkX 达到 **26–55× 速度提升** |
| **准确性** | 在小图上与 NetworkX 最优解偏差 <1%，在多数情况下达到最优或近最优 |
| **可扩展性** | 支持百万级候选节点保留（K > 700,000），远超 CPU 方法内存限制 |
| **实用性** | 开源实现支持多种 GPU 架构（V100/A100/H100），适用于真实世界应用场景 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **合成数据集**：随机生成的小规模图（10 个顶点），密度从 0.1 到 0.9 不等，用于验证精度。
- **真实世界中等规模图数据集**：
  - **CMU**：卡内基梅隆大学图像匹配数据集
  - **GREC**：图形结构识别数据库（IAM 图形数据库子集）
  - **MUTA**：致突变性化合物分类数据集（Mutagenicity dataset）

### **实验设置**
- **硬件平台**：JUWELS Booster 节点，配备 **NVIDIA A100/V100/H100 GPU** 和双路 AMD EPYC 7702（共 48 核 CPU）
- **参数设置**：
  - 默认 `K = 700,000`（保留节点数）
  - 顶点操作成本：sub=2, del=ins=4
  - 边操作成本：sub=1, del=ins=2
- **评估指标**：
  - 平均 GED 值（越低越好）
  - 与最优解的**偏差百分比**（Deviation %）
  - **运行时间 / Speedup**
  - 分类任务中的**准确率**

### **基线方法对比**
| 方法 | 类型 | 描述 |
|------|------|------|
| **NetworkX** | 精确（启发式） | 使用 DFS + 二分图启发式估计未来代价，求解小图最优 GED |
| **Beam Search (BS10)** | 近似 | 仅保留前 10 个最佳路径，牺牲精度换取效率 |
| **DFS-1** | 近似 | 可扩展的深度优先搜索变体，用于中大规模图比较 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ 小图精度与加速比（vs NetworkX）
| 密度 | 偏差 (%) | Speedup |
|------|----------|---------|
| 0.1  | 0.55%    | 26×     |
| 0.3  | 0.71%    | 38×     |
| 0.5  | 0.65%    | 48×     |
| 0.7  | 0.71%    | 50×     |
| 0.9  | 1.00%    | 55×     |

> ⚡️ **结论**：FAST-GED 在几乎无损精度的前提下实现数量级加速。

#### ✅ 中等图性能对比（CMU/GREC/MUTA）
| 数据集 | FAST-GED GED | BS10 GED | DFS-1 GED | FAST-GED 时间(s) |
|--------|---------------|-----------|------------|------------------|
| CMU (30) | 95.9         | 132.1     | 171.5      | 1.0              |
| MUTA (70) | 73.6         | 113.5     | —          | 1.0              |

> 📉 **GED 更低**表明更优的近似质量；⏱️ **时间仅为 1 秒以内**，显著优于 BS（高达 600s）

#### ✅ CPU vs GPU 性能对比
- **单线程 CPU**：随 K 增大呈指数增长
- **48 核 CPU 并行版**：相对提速约 4.5×，但受限于内存，K < 800,000
- **A100 GPU 版**：**最高达 300× 加速**，可在 1 秒内处理 K=1M 的搜索空间

#### ✅ 不同 GPU 架构性能
| GPU   | 相对于 V100 的性能增益 |
|-------|------------------------|
| A100  | +40%                   |
| H100  | +55%                   |

> 💡 性能提升源于更高带宽、更多 SM 和改进的 Tensor Core 架构。

### **消融实验结果**
#### 🔍 参数 K 对准确性的影响（Fig. 2c）
- 当 `K` 从 10 增加到 1000 时，归一化 GED 显著下降（精度快速提升）
- 当 `K > 1000` 后趋于收敛，接近理论最优值
> ✅ 验证了 **K 是控制“精度-效率”权衡的有效参数**

#### 🔍 可扩展性测试（Fig. 2d）
- 图大小从 100 到 600 顶点，FAST-GED 保持近线性增长趋势（符合 $O(n^2)$ 复杂度预测）
- DFS-1 出现明显拐点，因串行处理瓶颈导致性能急剧恶化

#### 🔍 优化前后对比（Fig. 2a）
- 引入 **coalesced memory copy kernel** 后，原占 40% 运行时间的数据复制阶段降至 5%
- 整体性能提升 **2×**

---

## **4. 关键结论和发现**

### **主要发现**
1. **GPU 能有效打破 GED 中“精度 vs 速度”的困境**：通过高度并行化 K-Best 搜索，FAST-GED 实现了高精度与高速度的统一。
2. **无需主机交互即可完成整个 GED 计算流程**：全设备端执行极大减少了通信延迟，提升了可扩展性。
3. **更大的 K 值直接带来更高的准确性**：GPU 的轻量级线程模型使得探索更大搜索空间成为可能。
4. **在真实应用中展现巨大潜力**：
   - **Graph Classification**：结合 KNN+GED 在 Mutagenicity 数据集上达到 **75% 准确率**，媲美 GNN 模型。
   - **Neural Architecture Search (NAS)**：生成新架构时相比 NetworkX 实现 **最高 10³× 加速**，误差 <10%，使 GED-based crossover 成为可行策略。

### **方法的局限性**
- 当前版本仍依赖同步 barrier 和原子操作进行 Top-K 筛选，在极端大 K 场景下可能成为瓶颈。
- 对超大规模图（> 数万节点）的支持尚未验证，未来需引入分块或采样策略。
- 目前主要针对无向简单图，对有向图或多标签图的支持有待扩展。

### **未来工作方向**
1. **支持超大规模图**：扩展至数十万顶点级别的图处理。
2. **异构系统协同计算**：结合 CPU 预处理与 GPU 主计算，进一步优化资源利用。
3. **动态调整 K 策略**：根据搜索层级自动调节 K 值以平衡早期探索与后期收敛。
4. **集成到端到端 GNN 或 NAS 流程中**：作为可微模块或检索组件嵌入训练 pipeline。

---

> ✅ **总结一句话**：  
> **FAST-GED 成功将 GED 从“昂贵但精确”的学术工具转变为“快速、准确、可扩展”的工业级解决方案，推动其在图分类、神经架构搜索等领域的广泛应用。**

</details>

---

### 7. [FPTC: A Fast Parallel Transform-based Codec for Efficient Asymmetric Signal Compression](https://arxiv.org/abs/2605.01086)

**Authors**: Ben Mechels, Ryan Billmeyer, Alexander Chen, Shiyang Li, Caiwen Ding  
**Category**: cs.DC  
**Published**: 2026-05-05  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.01086v1  

#### Abstract
Modern high-performance computing and Internet-of-Things deployments increasingly generate large volumes of signal data that must be compressed efficiently on resource-constrained acquisition devices and decompressed at scale on centralized servers. Lossy compression is widely adopted to minimize st...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FPTC: A Fast Parallel Transform-based Codec for Efficient Asymmetric Signal Compression

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代高性能计算（HPC）和物联网（IoT）系统在资源受限的嵌入式设备上持续采集大量信号数据，这些设备面临**带宽、内存和能耗限制**，难以直接传输或存储原始数据。因此需要高效的**有损压缩**来降低开销。

然而，现有压缩方法通常存在以下问题：
- **编码端复杂度高**，不适合低功耗设备；
- **解码吞吐量不足**，无法满足服务器端大规模并行解压需求；
- 多数方法未针对**跨领域信号**进行通用化设计，泛化能力差；
- 缺乏对**重建质量（fidelity）** 和**解压吞吐量**的同时优化。

FPTC 正是为了解决这一“**不对称压缩场景**”中的综合挑战而提出的。

---

### 提出的新方法与创新点

FPTC 是一种**基于变换的快速并行编解码器（Fast Parallel Transform-based Codec）**，专为**编码轻量、解码高吞吐**的异构环境设计。其核心创新包括：

#### ✅ 轻量级顺序编码器 + 高吞吐 GPU 并行解码器
- **编码端**运行于资源受限设备（如可穿戴医疗设备），采用单通路、低复杂度流水线；
- **解码端**部署于 GPU 服务器，支持批量、并行、高速重建，适用于集中式数据分析、归档和 ML 流水线。

#### ✅ 基于窗口 DCT 的频域稀疏性利用
- 使用 **windowed DCT-II** 将时域信号转换到频域；
- 利用大多数信号的能量集中在低频系数上的特性，进行**谱截断（spectral truncation）**，仅保留前 $ E $ 个低频系数，显著减少熵编码输入规模。

#### ✅ 混合三区量化策略（Hybrid Three-Zone Quantization）
将 DCT 系数划分为三个区域，分别采用不同量化方式：
1. **Zone 0（低频）**: 使用 μ-law companding，精细保留大动态范围的关键能量；
2. **Zone 1（中频）**: 线性死区量化（linear deadzone），提升零值比例以利于熵编码；
3. **Zone 2（高频）**: 完全置零（aggressive zeroing），进一步压缩冗余信息。

该策略在保持高质量重建的同时提升了压缩效率。

#### ✅ 基于 SymLen 的高效 Huffman 编码格式
- 引入 **SymLen bitstream 格式**：将 Huffman 编码后的变长码字打包成固定长度的 64-bit 字，并记录每个字中包含的符号数量；
- 支持 GPU 上**无同步、独立线程解码**，避免线程间依赖；
- 结合 **length-limited canonical Huffman coding**，确保查找表大小可控（$ O(2^{L_{\text{max}}}) $），适合硬件实现。

#### ✅ 双融合 GPU 解压流水线（Dual-Fused Kernel Design）
- 第一阶段 kernel：融合 **Huffman 解码 + 变长缓冲区紧缩（variable buffer compaction）**；
- 第二阶段 kernel：融合 **反量化 + IDCT 重建**；
- 充分利用 GPU 并行性和内存访问优化，最大化吞吐量。

#### ✅ 离线训练结构用于在线轻量编码
- 在特定信号域内使用代表性数据集预训练 **量化表（quantization table）** 和 **Huffman 编码树**；
- 这些结构可在同类设备上复用，极大降低实时编码负担。

---

### 相比现有方法的优势

| 维度 | FPTC | 现有主流方法（如 cuSZp3, FZ-GPU, cuZFP） |
|------|------|----------------------------------------|
| **编码复杂度** | 极低，适合嵌入式设备 | 通常较高，不适用于资源受限场景 |
| **解码吞吐量** | 高，GPU 并行优化良好 | 虽然也面向 GPU，但未专门针对信号结构优化 |
| **压缩率（CR）** | 显著更高（尤其平滑信号） | 一般低于 FPTC |
| **重建质量（PRD）** | 更优，在相同 PRD 下 CR 更高 | 多数预测型压缩器易产生块状伪影 |
| **通用性** | 支持多领域信号（生物医学、气象、电力等） | 多为通用浮点压缩，未针对信号统计建模 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共评估 **10 个数据集**，覆盖 **4 类信号域**：

| 数据集 | 领域 | 原始大小（MB） | 描述 |
|-------|------|----------------|------|
| MIT-BIH, ECG-ARTH, EEG-MAT | Biomedical | 238–499 MB | 心电图（ECG）、脑电图（EEG），要求高保真 |
| Seismic | Seismic Traces | 14.98 MB | 地震反射数据，容忍一定失真 |
| Wind/Solar/Load Power | Power/Energy | ~396 MB | 电网负载与发电数据，高度平滑 |
| Temperature, Irradiance, Wind Speed | Meteorological | ~396 MB | 气象观测数据，趋势重要 |

> 所有数据集均被复制至超过 1GB 以公平比较各压缩器达到峰值吞吐所需的最小数据量。

---

### 实验设置与平台
- **硬件平台**：NVIDIA RTX PRO 6000 Blackwell Workstation Edition（96GB 显存）
- **软件环境**：Ubuntu 22.04 + CUDA Toolkit 13
- **编译选项**：`nvcc -O3`
- **测量方式**：
  - 解压吞吐量 = 输出数据量 / 解压时间（不含 Host-Device 传输）
  - 时间取五次运行平均值

---

### 评估指标
| 指标 | 定义 | 目标 |
|------|------|------|
| **Compression Ratio (CR)** | $ \frac{S_{\text{orig}}}{S_{\text{comp}}} $ | 越高越好 |
| **Percentage RMS Difference (PRD)** | $ 100 \times \sqrt{\frac{\sum(x_i - \hat{x}_i)^2}{\sum x_i^2}} $ | 控制在应用允许范围内（通常 <5%） |
| **Throughput (GB/s)** | 解压输出速率 | 越高越好，体现 GPU 利用效率 |

---

### 基线方法对比
选择当前最先进的 GPU 加速有损压缩器作为 baseline：
- **cuSZp3**：基于预测的误差有界压缩器
- **FZ-GPU**：快速高比率科学数据压缩
- **PFPL**：保证误差边界的 CPU/GPU 压缩框架
- **cuZFP**：CUDA 版本的 ZFP，固定率浮点压缩

> 所有方法均调整参数以匹配目标 PRD 水平进行比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 领域 | FPTC 相对最佳基线的 **CR 提升倍数** | PRD 范围 | 吞吐量排名 |
|------|-------------------------------|----------|------------|
| Power/Energy | **3.6x** | ~0.8–1.0% | Top 2 |
| Meteorological | **3.1x** | ~1.0–1.5% | Top 2 |
| Biomedical | **1.5x** | ~2.0–4.0% | 中上游 |
| Seismic | **1.2x** | ~1.5–2.0% | 中游 |

> 示例：在 Load Power 数据上，FPTC 达到 **CR=100.4x @ PRD=0.855%**，而 cuSZp3 仅为 **4.9x @ 1.001%**

---

### 与基线方法的对比结果

#### 📈 压缩率 vs. 重建质量（Rate-Distortion 曲线）
- 图 8 显示，在 PRD ∈ [1%, 6%] 区间内，FPTC 的 **Pareto 前沿全面领先**；
- 在平滑信号（如 Load Power）上优势最明显，CR 可达基线的数十倍；
- 即使在较难压缩的 ECG/EEG 上，仍能提供更优的 CR-PRD 权衡。

#### ⚡ 解压吞吐量表现
- 如图 12 所示，在多个 PRD 分段下，FPTC 吞吐量稳定在 **~260 GB/s 左右**；
- 仅次于 cuZFP（~664 GB/s），但后者重建质量极差（严重失真）；
- 在 **Power 和 Meteorological 领域**，FPTC 吞吐量接近最优水平；
- 表现出良好的稳定性（见 Table 3），标准差远小于 cuSZp3。

#### 🔍 定性重建效果对比（图 10）
- FPTC 重建波形连续自然，保留了原始信号的趋势和局部波动；
- 而 cuSZp3 等预测类压缩器出现明显的**块状伪影（block artifacts）**；
- 表明 FPTC 不仅 CR 更高，且**局部特征保持更好**。

---

### 消融实验与参数分析（Ablation Study）

#### 参数敏感性分析（Table 1 & 图 14）
- **DCT_SIZE (N)**：典型值为 32；过大增加计算负担，过小损失能量集中效果；
- **ENCODED_COEFFS (E)**：直接影响 CR 和吞吐量；越小则 CR 越高、解压越快；
- **Throughput ∝ 1/E**：保留系数越少，Huffman 解码和 IDCT 工作量越小 → 吞吐越高；
- 最佳性能出现在 **N=32, E=16** 附近，形成“甜点”。

#### 数据集相关性分析（图 11）
- 同一领域内的最优参数高度相关（如 ECG 与 EEG 的 r ≥ 0.92）；
- 不同领域差异显著（如 Seismic 与 Load Power 参数完全不同）；
- 支持“**使用代表数据预训练结构具有泛化能力**”的设计理念。

#### 运行时分解（图 13）
- 在非平滑信号（如 MIT-BIH）上，**lossless kernel 占主导（60%）**；
- 在平滑信号（如 Wind Speed）上，**lossy kernel 占主导（80%）**；
- 表明双融合 kernel 设计能适应不同工作负载分布，具备跨域鲁棒性。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **FPTC 成功实现了编码轻量与解码高吞吐的统一**：
   - 编码器可在嵌入式设备上实时运行；
   - 解码器在 GPU 上实现百 GB/s 级吞吐，适合数据中心批量处理。

2. ✅ **基于 DCT 的变换压缩在多领域信号中依然有效**：
   - 尤其对平滑、强频域能量聚集的信号（如电力、气象）极具优势；
   - 结合谱截断和混合量化，可在极高压缩比下维持可用质量。

3. ✅ **SymLen + Dual-Fused Kernel 架构显著提升 GPU 效率**：
   - 无需线程同步即可完成 Huffman 解码；
   - 内存访问高度合并，避免写冲突；
   - 是构建高性能并行解码器的有效范式。

4. ✅ **最优压缩参数具有领域聚类性**：
   - 同类信号共享相似最优配置；
   - 支持离线训练、在线部署模式，增强实用性。

5. ✅ **FPTC 在压缩率上大幅超越现有框架**：
   - 在 Power 和 Meteorological 领域实现 **3.6x 和 3.1x 的 CR 提升**；
   - 即使在严苛的 ECG 应用中也能提供 **1.5x 以上的增益**。

---

### 方法的局限性

1. ❗ 对**非平稳、剧烈变化、弱相关性信号**（如部分地震数据）压缩增益有限；
2. ❗ 当前设计依赖**离线训练的量化表和 Huffman 树**，若信号分布漂移需重新校准；
3. ❗ **极端高保真场景（PRD < 1%）** 下，压缩优势减弱，因必须保留更多系数；
4. ❗ 目前仅支持**单变量时间序列**，尚未扩展至多通道或多维信号。

---

### 未来工作方向

1. ➕ 扩展至 **multi-channel 和 streaming 场景**，支持动态参数调整；
2. ➕ 探索 **adaptive DCT window size selection** 机制，根据局部平滑度自适应切换；
3. ➕ 引入 **learned transform 或轻量神经网络** 替代手工 DCT，进一步挖掘稀疏性；
4. ➕ 开发 **自动参数调优工具链**，基于少量样本自动推导最优 $ N, E, B1, B2 $ 等；
5. ➕ 将 FPTC 集成进实际 IoT/边缘计算系统，验证端到端延迟与能耗收益。

---

## 总结

FPTC 提出了一种面向**异构资源环境**的新型不对称信号压缩架构，通过**变换编码 + 混合量化 + 高效熵编码 + GPU 并行解码优化**，在**压缩率、重建质量和解压吞吐量之间取得了卓越平衡**。实验证明其在多种真实世界信号中均优于现有先进方法，尤其在电力与气象等平滑信号领域表现突出，为未来智能传感系统的规模化部署提供了强有力的底层支撑。

</details>

---

### 8. [Component-Aware Self-Speculative Decoding in Hybrid Language Models](https://arxiv.org/abs/2605.01106)

**Authors**: Hector Borobia, Elies Segu\'i-Mas, Guillermina Tormo-Carb\'o  
**Category**: cs.CL  
**Published**: 2026-05-05  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.01106v1  

#### Abstract
Speculative decoding accelerates autoregressive inference by drafting candidate tokens with a fast model and verifying them in parallel with the target. Self-speculative methods avoid the need for an external drafter but have been studied exclusively in homogeneous Transformer architectures. We intr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Component-Aware Self-Speculative Decoding in Hybrid Language Models**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
- **Autoregressive 推理瓶颈**：大型语言模型（LLMs）在生成文本时逐个 token 进行 autoregressive 解码，导致推理延迟高。
- **现有 Speculative Decoding 的局限性**：传统 speculative decoding 需要一个外部的“drafter”模型来起草候选 token，带来额外存储、对齐和训练成本；self-speculative 方法（如 LayerSkip）虽避免外部模型，但仅适用于同质化架构（如纯 Transformer），无法利用混合模型中的内部异构性。

### **提出了什么新方法或新思路**
- **Component-Aware Self-Speculative Decoding**：
  - 首次提出利用 **hybrid language models** 内部的架构异构性进行 self-speculative decoding。
  - 将模型中非 attention 路径（如 SSM 或 linear attention）作为零成本的内部 draft 模型，通过抑制 attention 分支生成 draft token。
  - 不需要额外参数、训练或外部模型，完全从目标模型中提取。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **效率** | 利用 SSM 路径的 O(1) 序列长度内存特性（无 KV cache），降低 drafting 成本。 |
| **通用性** | 专为 hybrid 架构设计，填补了现有 self-speculative 方法在异构模型上的空白。 |
| **零成本** | 无需额外模型或训练，仅修改 forward pass 即可实现。 |
| **理论保障** | 保留 speculative decoding 的 lossless 性质，输出分布与原模型一致。 |

---

## 2. **核心实验方法和设置**

### **使用的模型与数据集**
#### **模型**
| 模型 | 类型 | 参数量 | 架构特点 |
|------|------|--------|----------|
| **Falcon-H1-0.5B / 3B** | 平行混合模型（Parallel Hybrid） | 0.5B / 3B | 每层并行运行 Mamba-2 SSM 和 attention，输出相加 |
| **Qwen3.5-0.8B** | 串行混合模型（Sequential Hybrid） | 0.8B | 交替堆叠 linear attention 层与 softmax attention 层（18:6） |
| **Qwen2.5-0.5B** | 纯 Transformer 控制组 | 0.5B | 全为 attention 层，用于 LayerSkip 对比 |

#### **数据集**
- 主要评估：**WikiText-2 validation split**（200 prompts，截断至 512 tokens）
- 任务评估：**MMLU**（知识问答）、**GSM8K**（数学推理）、**Alpaca-style instruction prompts**

### **实验设置**
- **Draft Length $k$**：{2, 4, 8}
- **Temperature**：{0.0（greedy）, 0.6（sampling）}
- **硬件**：单张 NVIDIA L4 24GB GPU（RunPod），PyTorch + HuggingFace Transformers
- **实现方式**：通过 forward hooks 抑制 attention 输出或跳过其计算

### **评估指标**
| 指标 | 定义 |
|------|------|
| **All-token acceptance rate $\alpha(k)$** | 所有 $k$ 个 draft token 均被接受的比例（主指标） |
| **Total Variation Distance $D_{TV}$** | Draft 与完整模型输出分布之间的差异 |
| **Wall-clock speedup** | 实际推理时间加速比（受实现影响较大） |
| **Output match rate** | speculative 与 autoregressive 解码输出的一致性（验证 lossless） |

### **基线方法对比**
- **Component-Aware Self-Speculation**（本文方法）：抑制 attention 分支，使用 SSM/linear-only 子图作为 drafter
- **LayerSkip**：跳过部分层（33%），通用 self-speculative 方法
- **Early-exit**：仅使用前 50% 层进行 drafting

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **平行混合模型（Falcon-H1）表现优异**
| 模型 | $k=2$ ($T=0$) | $k=4$ ($T=0$) | $k=8$ ($T=0$) |
|------|---------------|---------------|---------------|
| **Falcon-H1-0.5B** | **0.680** | **0.370** | **0.186** |
| **Falcon-H1-3B** | 0.590 | 0.351 | 0.186 |

> - 在 $k=2$ 下达到 **68% 的全 token 接受率**，显著高于其他模型。
> - 表现出 **scale invariance**：0.5B 与 3B 模型接受率几乎相同，说明效果由架构决定而非规模。

#### ❌ **串行混合模型（Qwen3.5）表现极差**
| 模型 | $k=2$ ($T=0$) | $k=4$ ($T=0$) | $k=8$ ($T=0$) |
|------|---------------|---------------|---------------|
| **Qwen3.5-0.8B** | **0.038** | **0.019** | **0.009** |

> - 接受率仅为 Falcon-H1 的 **1/18**，表明 component-aware 自推测在串行架构中不可行。

#### 🔁 **温度采样下的结果趋势一致**
| 模型 | $k=2$ ($T=0.6$) |
|------|----------------|
| Falcon-H1-0.5B | 0.560 |
| Qwen3.5-0.8B | 0.073 |

> - 温度放松略微提升接受率，但 **架构差距依然巨大**。

### **与基线方法的对比结果**

#### 📊 **Qwen3.5 上不同策略比较（$k=4, T=0$）**
| 策略 | 接受率 $\alpha(4)$ |
|------|--------------------|
| **Linear-only (component-aware)** | 0.019 |
| **LayerSkip 33%** | **0.233**（↑12.3×） |
| **Early-exit 50%** | 0.000 |

> - 在串行混合模型中，**generic LayerSkip 明显优于 component-aware 方法**。
> - 说明保持组件交错结构比隔离单一路径更重要。

#### 📈 **总变异距离（Distribution Divergence）**
| 模型 | $D_{TV}$（平均） | Top-1 Agreement |
|------|------------------|------------------|
| Falcon-H1-0.5B | **0.302** | **65.8%** |
| Qwen3.5-0.8B | 0.803 | 20.3% |
| Qwen2.5-0.5B (LayerSkip) | 0.473 | 49.6% |

> - Falcon-H1 的 draft 分布更接近完整模型，支持其高接受率。

### **消融实验结果**
#### 🔍 **功能组件消融预测 speculative viability**
| 模型 | PPL ratio (no-attn / baseline) | $\alpha(k=4)$ |
|------|-------------------------------|--------------|
| **Falcon-H1-0.5B** | ×3.15 | **0.370** |
| **Qwen3.5-0.8B** | ×81.96 | 0.019 |

> - **完美逆相关**：perplexity 增幅越小 → attention 依赖越低 → component-aware self-speculation 越可行。
> - 可作为廉价诊断工具：只需一次 ablation 测试即可预测 speculative 效果。

#### ⏱️ **实际速度 vs 理论速度**
| 模型 | 实测 wall-clock speedup ($k=2$) | 理论 speedup |
|------|-------------------------------|-------------|
| Falcon-H1-0.5B | 0.342×（慢） | 0.92× |
| Qwen2.5-0.5B (LayerSkip) | 0.496× | ~1.0×（估计） |

> - 当前实现因 Python hook、无 KV cache 共享等造成严重开销。
> - **理论分析表明优化后可达近线速加速**，尤其在长上下文下 SSM 的 O(1) 内存优势将显现。

---

## 4. **关键结论和发现**

### **主要发现**
1. **架构模式决定 speculative viability**：
   - **Parallel hybrids**（如 Falcon-H1）：SSM 与 attention 并行叠加 → SSM 子图可独立承担 drafting 功能 → 接受率高达 **68% @k=2**。
   - **Sequential hybrids**（如 Qwen3.5）：attention 是信息流的关键环节 → 移除后破坏训练分布 → 接受率仅 **3.8% @k=2**。

2. **Scale invariance**：
   - Falcon-H1 在 0.5B 与 3B 规模下接受率基本一致，说明该性质是 **architecture-level property**，不受模型大小影响。

3. **Functional ablation 可预测 speculative 效果**：
   - **PPL ratio < 5×** → component-aware self-speculation 可能有效；
   - **PPL ratio > 20×** → 几乎无效，应改用 LayerSkip 等通用策略。

4. **策略选择建议**：
   - 对 **parallel hybrids**：优先使用 **component-aware self-speculation**（更高接受率 + 更好利用 SSM 特性）。
   - 对 **sequential hybrids**：使用 **LayerSkip** 等通用方法（保留组件交错结构）。

### **方法的局限性**
| 局限 | 描述 |
|------|------|
| **当前实现未达实用加速** | Python 级别 hook 导致严重 overhead，wall-clock 仍慢于 autoregressive。 |
| **仅覆盖两类 hybrid 架构** | 未测试 Jamba（MoE + Mamba）、Samba（滑窗 attention）等更复杂结构。 |
| **缺乏大规模验证** | 最大仅到 3B，尚不确定 7B+ 是否维持相同规律。 |
| **固定 draft length** | 未探索 adaptive $k$，可能限制实际吞吐。 |

### **未来工作方向**
1. **开发生产级实现**：
   - 使用 CUDA kernel fusion、batched verification、KV cache sharing 实现真正加速。
2. **扩展至更多 hybrid 架构**：
   - 如 Jamba、Samba 等，验证 component-aware 方法的泛化能力。
3. **结合 tree-based speculative decoding**：
   - 如 STree，提升有效接受率。
4. **adaptive draft length selection**：
   - 借鉴 SWIFT 思路，动态调整 $k$ 以最大化吞吐。
5. **指导 hybrid 模型设计**：
   - 为支持高效 inference，推荐采用 **parallel integration** 设计，使 SSM 路径本身具备较强语言建模能力。

---

> ✅ **一句话总结**：  
> 本文首次提出 **component-aware self-speculative decoding**，揭示 **hybrid model 的架构集成方式（parallel vs sequential）决定了其是否适合基于内部组件的自推测加速**，并提供了一套可预测、可复现的评估框架，为高效混合模型推理系统的设计提供了重要指导。

</details>

---

### 9. [Differentiable Kernel Ridge Regression for Deep Learning Pipelines](https://arxiv.org/abs/2605.02313)

**Authors**: Jean-Marc Mercier, Gabriele Santin  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.02313v1  

#### Abstract
Deep neural networks dominate modern machine learning, while alternative function approximators remain comparatively underexplored at scale. In this work, we revisit kernel methods as drop-in components for standard deep learning pipelines. We introduce \emph{Sparse Kernels} (SKs), a differentiable,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Differentiable Kernel Ridge Regression for Deep Learning Pipelines*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **kernel methods**（如 Kernel Ridge Regression, KRR）虽然在理论上具有良好的泛化性和可解释性，但由于其 **O(N³)** 的计算复杂度和非端到端可微性，难以集成进现代 **deep learning pipelines** 中。此外，现有的深度学习模型通常依赖于参数化的读出层（如 linear 或 MLP head），限制了对中间表示的灵活利用。

本文旨在弥合这一鸿沟，探索如何将 kernel 方法以**可微分、模块化、高效**的方式嵌入标准的深度学习框架中。

---

### 🚀 提出的新方法：Sparse Kernels (SKs)

作者提出了一种名为 **Sparse Kernels (SKs)** 的新方法，它是 **localized kernel ridge regression (KRR)** 的一种**可微分、懒惰式（lazy）、稀疏变体**，具备以下特点：

- **Lazy Evaluation**：训练被推迟到推理时进行，仅需求解局部小规模线性系统（size M << N），显著降低计算开销。
- **Differentiable**：支持对 feature representations、target values 和 evaluation points 的梯度传播，可无缝接入 PyTorch 等自动微分框架。
- **Modular Design**：作为独立 layer 集成进任意网络架构（CNN、ViT、RL agent），保持 end-to-end 可训练性。
- **Three-way Parameterization**：
  - Feature representations（特征）
  - Target values（标签）
  - Evaluation points（查询点）
  
  这三者均可固定或学习，极大拓展了设计空间，支持多种训练范式。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | 本文方法（SKs） |
|------|--------|----------------|
| **可扩展性** | 全局 KRR 为 O(N³)，不可扩展 | 局部 M-NN + 小系统求解，成本线性于 N |
| **灵活性** | 固定 kernel 或黑箱使用 | 显式暴露三个参数集，支持自由组合 |
| **训练效率** | 需要完整训练 readout head | Lazy 模式下无需训练即可 transfer（zero-shot probing） |
| **表达能力** | Linear probe 仅限线性 | Kernel probe 提供非线性、数据自适应 readout |
| **集成方式** | 多用于分析工具（如 probing） | 可作为增强组件直接提升模型性能 |

> 💡 **核心思想突破**：KRR 不再是替代神经网络的整体方案，而是作为**即插即用的模块化组件**，与深度学习共存而非对立。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 实验任务 | 数据集 |
|--------|-------|
| Transfer Learning & Probing | **CIFAR-10**（基于 ImageNet-pretrained ResNet-18/VGG-19/ViT 提取特征） |
| Reinforcement Learning | **LunarLander-v3**（来自 Gymnasium 环境） |

所有实验均使用相同的特征提取器输出，确保公平比较。

---

### ⚙️ 实验设置与评估指标

#### （1）Transfer Learning（图1）
- **Backbone**：冻结的 ResNet-18（ImageNet 预训练）
- **Readout 对比**：
  - Linear Probe（512→10）
  - MLP(512-512-ReLU-10)
  - SK(100)：M=100 的 discontinuous lazy KRR
  - HAN-SK(100)：连续版本的 hierarchical sparse KRR
- **评估指标**：
  - Test Accuracy（随 labeled sample 数量变化）
  - Wall-clock execution time（训练 + 推理时间）

#### （2）Probing（图2）
- **模型**：VGG-19 和 Vision Transformer（ViT）
- **协议**：逐层移除顶部 block，在不同 depth 的 intermediate representation 上应用 readout
- **目标**：分析哪一层的表示更适合 kernel-based probing
- **readout 类型**：SK(100)

#### （3）Learning with Probes（图3）
- 设置同上，但允许 kernel readout 的参数通过 AdamW 进行 end-to-end 训练
- 验证是否可通过训练进一步提升性能

#### （4）Reinforcement Learning（图4）
- **基础算法**：Double DQN
- **改进**：在第一层和最后一层加入 kernel perturbation：
  $$
  Q_k(s,a;\theta) = \text{ReLU}\left([\text{ReLU}(s\theta^1) + P_k(x_1,y_1)(s)]\theta^2\right)\theta^3 + P_k(x_3,y_3)(s)
  $$
- **learnable 参数**：support sets $(x_i, y_i)$
- **评估指标**：
  - Average Reward over 50 games
  - TD Loss during training
- **运行配置**：5 runs × 500 episodes，结果取均值与 min-max 范围

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据与对比

#### （1）Transfer Learning（图1）
| 方法 | 准确率趋势 | 训练成本 |
|-----|----------|---------|
| Linear Probe | 基线水平，表现尚可 | 低 |
| MLP Head | 略有提升，尤其在大数据量下 | 中等（需训练） |
| **SK(100)** | **接近甚至超过 MLP，且在小样本下优势明显** | **极低（lazy 模式无训练）** |
| HAN-SK(100) | 在中小样本下有竞争力，大样本无明显增益 | 略高于 SK（需少量训练） |

> ✅ **结论**：**lazy KRR readout 在无需训练的情况下达到甚至超越 fully trained MLP head 的性能**。

---

#### （2）Probing（图2）
- **VGG-19**：final layer 表现尚可（因 head 是 MLP），但 intermediate layers 更优。
- **Vision Transformer**：**final layer 性能下降明显**，而中间层（如去掉最后1~2个 block）达到峰值准确率。
- **SK-based probing 比 linear probe 更敏感地揭示 representation quality 差异**。

> ✅ **发现**：**final layers 可能 over-specialize 到原始任务 head，丢失通用信息；intermediate representations 更适合 transfer 和 probing**。

---

#### （3）Learning with Probes（图3）
- 趋势与 probing 实验一致：**在 intermediate layers 上训练 kernel readout 效果更好**。
- 最佳性能仍出现在移除顶部1~2层后的位置。
- 表明 kernel 方法不仅可用于分析，也可用于实际训练，并从中受益。

---

#### （4）Reinforcement Learning（图4）
| 指标 | DQN_Agent | DQK_Agent（kernel-augmented） |
|------|-----------|-------------------------------|
| 收敛速度 | 较慢 | **更快达到高奖励** |
| 最终平均奖励 | ~220 | **~240+（提升约 10%）** |
| TD Loss | 较低初期波动 | **探索阶段更高，后期稳定** |

> ✅ **解释**：kernel 组件提供了更强的拟合能力，在探索期“不过拟合”TD target，从而获得更鲁棒的策略。

---

### 🔬 消融实验（隐含于设计中）

尽管未明确列出消融表，但从多个实验可推断出关键因素的影响：

| 因素 | 影响 |
|------|------|
| **M（neighborhood size）** | M=100 表现良好；更大 M 提升精度但增加延迟 |
| **Kernel Continuity** | discontinuous SK 已足够有效；continuous variant（HAN-SK）未带来显著收益 |
| **Evaluation Depth** | intermediate > final layer，验证了 representation selection 的重要性 |
| **Trainable Targets / Points** | 在 RL 实验中证明 learnable $(x_i, y_i)$ 可带来性能增益 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Nonparametric Readouts Are Powerful**  
   lazy KRR（即 SK）作为一种非参数读出机制，能够在**无需训练**的情况下，**媲美甚至超越 fully trained MLP heads**，尤其适用于 low-data transfer 场景。

2. **Intermediate Representations Are Superior for Kernels**  
   无论是 probing 还是训练，**中间层表示比最终层更适合 kernel-based readouts**，说明 final layers 存在 over-specialization 问题。

3. **Kernels Can Augment Existing Models**  
   即使在强化学习这种非监督场景中，将 SK 作为 additive component 插入 DQN 架构，也能**显著提升性能**，表明 kernel 模块具有广泛适用性。

4. **Three-way Parameterization Enables New Paradigms**  
   显式分离 feature、target、evaluation point 三大要素，支持：
   - Training-free transfer（fix all）
   - Nonlinear probing（fix features & targets）
   - Hybrid training（learn evaluation points or targets）

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|-------|------|
| **Scalability on Large-scale Tasks** | 当前实验集中在 CIFAR-10 和 LunarLander，尚未在 ImageNet、LLM 或 Diffusion Models 上验证 |
| **GPU Batch Efficiency** | 虽然每 query 成本为 O(M³)，但批量 M×M 系统求解和 M-NN 查找的 GPU 并行效率有待优化 |
| **Discontinuity Issue** | 默认 SK 构造是全局不连续的，可能影响需要 smooth gradients 的下游任务（如 gradient-based planning） |
| **Hyperparameter Sensitivity** | M 和 kernel function $p(\cdot)$ 为超参，缺乏自动化选择机制 |
| **Regularization for Learnable Centers** | 若将 evaluation points 设为 learnable，可能需额外正则化（如 OT-based penalty）防止 collapse |

---

### 🔮 未来工作方向

1. **Large-scale Evaluation**：在 ImageNet、大规模语言建模等任务上测试 SK 的有效性。
2. **Efficient GPU Implementation**：优化 batched M-NN search 与 local KRR solve 的并行化。
3. **Auto-tuning of M and Kernel**：开发基于数据驱动的方法自动选择 bandwidth $M$ 和 kernel 类型。
4. **Continuous & Smooth Variants**：推广 Appendix A.5 中的 hierarchical construction，构建 globally smooth predictors。
5. **End-to-end Kernel Center Learning**：深入研究 learnable $(x_i, y_i)$ 的训练动态与正则化策略。

---

## ✅ 总结

该论文成功将 **kernel ridge regression** 从理论工具转变为**实用、可微、模块化的 deep learning 组件**。通过引入 **Sparse Kernels (SKs)**，实现了：

- **高性能 zero-shot transfer**（无需训练 readout）
- **更强的 probing 能力**（捕捉非线性结构）
- **模型增强能力**（如在 DQN 中插入 kernel 模块提升性能）

> 🌟 **一句话总结**：  
> **Once made differentiable and scalable, kernel methods are not competitors to deep learning — they are collaborators.**

</details>

---

### 10. [Revisiting Semantic Role Labeling: Efficient Structured Inference with Dependency-Informed Analysis](https://arxiv.org/abs/2605.02505)

**Authors**: Sangpil Youm, Leah Jones, Bonnie J. Dorr  
**Category**: cs.CL  
**Published**: 2026-05-05  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.02505v1  

#### Abstract
Semantic Role Labeling (SRL) provides an explicit representation of predicate-argument structure, capturing linguistically grounded relations such as who did what to whom. While recent NLP progress has been dominated by large language models (LLMs), these systems often rely on implicit semantic repr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Revisiting Semantic Role Labeling: Efficient Structured Inference with Dependency-Informed Analysis*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **AllenNLP SRL 框架已进入维护模式**（maintenance mode since Dec 2022），导致其与现代 encoder 架构（如 RoBERTa、DeBERTa）及推理需求不兼容。
- 传统 SRL 推理对每个 predicate 都重复进行 sentence-level 编码，造成严重的计算冗余，影响效率。
- 当前 LLM-based 方法虽强大，但缺乏显式的 **predicate-argument 结构约束**，在跨语言迁移、可解释性和结构稳定性方面存在不足。

### 🚀 提出的新方法与创新
1. **现代化的 Encoder-Based SRL 框架**
   - 引入 **sentence-level encoding reuse** 机制：仅对句子进行一次编码，并在所有 predicates 上复用该表示，避免重复计算。
   - 显著提升推理速度（达 **10× 加速**），同时保持预测性能。

2. **Dependency-Informed 诊断分析框架**
   - 提出一种基于 dependency parsing 的 span-level 错误检测方法，用于识别和修复：
     - 同一语义角色的重复分段（repeated spans）
     - 不合理的 span 边界划分（如 BIO tagging violation）
   - 支持自动合并 `same_head`, `pp_attach`, `subtree_attach` 类型的错误 span。

3. **支持多语言 SRL 投影（Multilingual SRL Projection）**
   - 将该结构化框架应用于 cross-lingual SRL transfer，利用 dependency-aware 分析减少源端英文中的标注错误传播。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 推理速度快 **10×**（1.32min vs 13.18min on CoNLL-2012 test set） |
| **兼容性** | 兼容任意 modern encoder（BERT/RoBERTa/DeBERTa），不再依赖过时的 AllenNLP |
| **结构一致性** | 显式建模 predicate-argument 结构，增强模型可解释性与可控性 |
| **下游应用支持** | 可无缝集成到 multilingual SRL projection pipeline 中 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **CoNLL-2012 Shared Task dataset**（基于 OntoNotes 5.0）
  - 英文部分，标准 train/dev/test 划分
  - 包含约 32,617 个 clause-level predictions 和 877,873 tokens
  - 使用 PropBank 风格的 span-based 语义角色标注（如 ARG0, ARG1, ARGM-TMP 等）

### ⚙️ 实验设置
- **Encoder Backbones 对比**：
  - BERT-base-cased（主比较）
  - RoBERTa-large
  - DeBERTa-v3-large
- **训练配置**：
  - 复现 AllenNLP + Shi & Lin (2019) 的设定
  - LSTM 和 MLP 层维度分别为 768 和 300
  - 学习率：5e-5
  - 硬件：AMD EPYC CPU + NVIDIA DGX B200 GPU
- **输入格式**：
  ```
  [CLS] |sent| [SEP] |pred| [SEP]
  ```

### 📊 评估指标
- **Phrase-level F1-score**（主要指标）
- Precision, Recall
- 推理时间（Time in minutes）
- 输出一致性分析（与 AllenNLP 输出的 token-level 匹配率）

### 🆚 基线方法
- **AllenNLP SRL 模型**（Gardner et al., 2018）作为主要 baseline
- 所有实验均确保与之具有相同的训练范式，仅在 inference 阶段优化

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）

| Model | P (%) | R (%) | **F1 (%)** | Time (min) |
|-------|--------|--------|------------|-------------|
| **Ours (BERT)** | 85.98 | 86.32 | **86.15** | **1.36** |
| **Ours (RoBERTa)** | 86.19 | 87.97 | **87.07** | 5.36 |
| **Ours (DeBERTa)** | 86.78 | 88.22 | **87.49** | 7.57 |
| **AllenNLP (BERT)** | 86.81 | 86.63 | **86.72** | 13.18 |

> ✅ **结论**：我们的 BERT 模型达到与 AllenNLP 几乎相同的 F1（仅低 0.57%），但推理快 **10 倍以上**

### 🔁 输出一致性分析
- 在 877,873 个 tokens 上，**95.2% 的预测结果与 AllenNLP 完全一致**
- 差异部分中：
  - AllenNLP 正确而 ours 错误：2.11%
  - Ours 正确而 AllenNLP 错误：1.41%
  - 两者皆错：1.28%

> 表明新框架高度还原原模型行为，且部分错误更少

### 🔍 消融实验与错误分析（见 Section 5）
- **Span Violation 分析**：
  - AllenNLP 有 19,131 个 span 错误，本模型有 17,387 个
  - 最常见错误类型为 **ARGM-ADV** 的重复分段
- **Dependency-Aware Error Detection**：
  - 在 AllenNLP 正确而 ours 错误的 18,539 个 token 中：
    - **498 个（2.68%）可被自动修复**（归类为 `same_head`, `pp_attach`, `subtree_attach`）
    - **933 个需人工审查**
  - 说明 dependency 结构可用于系统性地发现并修正结构不一致问题

### 💬 LLM 实验结果（Section 6）
- 在 zero-shot 设置下测试 gpt-oss-120b：
  
| 方法 | Precision | Recall | **F1** |
|------|----------|--------|--------|
| 无 dependency 提示 | 46.58 | 35.58 | **42.17** |
| **带 dependency 提示** | 49.03 | 40.44 | **44.30** |

> ✅ **dependency-informed prompts 提升 F1 2.13 pts**，主要改善的是 **span boundary 稳定性与短语完整性**，而非分类准确率本身

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **高效推理无需牺牲性能**：
   - 通过 sentence-level encoding reuse 实现 **10× 推理加速**，F1 与 AllenNLP 相当。
2. **结构信号显著提升稳定性**：
   - dependency cues 能有效缓解 span fragmentation 和 duplicate role assignment，提高 argument grouping 的一致性。
3. **显式结构优于隐式生成**：
   - 即使是强大的 LLM，在没有 structural guidance 的情况下也难以稳定输出正确的 BIO tagging 结构；加入 dependency 提示后表现明显改善。
4. **框架天然适配 multilingual transfer**：
   - 在 cross-lingual SRL projection 中，能有效减少因 English-side 标注错误导致的跨语言传播问题（如 `after` 被错误标记为 I-ARGM-TMP 而非 B-ARGM-TMP）。

### ⚠️ 方法的局限性
- **自动错误修复能力有限**：
  - 当前 dependency-aware analyzer 仅能处理 ~2.7% 的错误（fixable cases），大多数仍需人工干预。
- **依赖高质量 dependency parser**：
  - 若 dependency parsing 出错，可能误导 span 合并判断。
- **训练数据偏差**：
  - 模型基于 OntoNotes 训练，可能存在 domain bias 和 annotation bias（如某些 ARGM 类型覆盖不足）。

### 🔮 未来工作方向
1. **将结构化 SRL 思路扩展至 decoder-only LLMs**，实现端到端结构化生成。
2. **构建更大规模的 multilingual SRL dataset**，结合本框架进行高质量跨语言投影。
3. **进一步优化 dependency-aware error detection rule system**，降低人工审核成本。
4. **探索 SRL 作为中间表示在 reasoning、QA、summarization 中的应用价值**。

---

> 📌 **总体评价**：本文成功复兴了 structured SRL 建模范式，在保留显式 predicate-argument 结构的同时，实现了现代 encoder 兼容性与极高推理效率，并通过 dependency-informed analysis 提供了新的可解释性工具，为 SRL 在 LLM 时代的持续发展提供了坚实路径。

</details>

---

### 11. [Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics](https://arxiv.org/abs/2605.01280)

**Authors**: Zijie Zhou  
**Category**: cs.DC  
**Published**: 2026-05-05  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.01280v1  

#### Abstract
This position paper argues that LLM inference serving has outgrown generic heuristics and now demands mathematical optimization and algorithmic foundations. Despite rapid advances in serving systems such as vLLM and SGLang, their algorithmic cores remain largely unchanged from classical distributed ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Position: LLM Serving Needs Mathematical Optimization and Algorithmic Foundations, Not Just Heuristics*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

该论文指出，当前主流的 **LLM Serving 系统**（如 vLLM、SGLang）虽然在架构上取得了显著进展（如 Continuous Batching、PagedAttention），但其**核心调度与资源管理策略仍依赖于经典分布式系统中的通用启发式方法**（heuristics），例如：

- 请求路由：Round-Robin、Join-Shortest-Queue
- 调度策略：FIFO（First-Come-First-Serve）
- KV Cache Eviction：LRU（Least Recently Used）

这些通用策略**忽略了 LLM 推理特有的结构性特征**，包括：

- **Prefill-Decode 阶段异构性**（compute-bound vs. memory-bandwidth-bound）
- **KV Cache 动态增长**（内存占用随生成过程递增）
- **输出长度未知**（job duration 不可预测）
- **Continuous Batching 引入的请求耦合**

因此，现有启发式方法在面对负载突变、长尾请求或复杂多模态场景时可能表现不稳定甚至失效。

---

### ✅ 提出了什么新方法或新思路

本文并非提出单一算法，而是**倡导一种范式转变**：将 LLM Serving 中的关键决策问题（如负载均衡、调度、缓存、容量规划）从“经验调参”转向**基于数学优化与算法理论的系统化设计**。

核心思想是：

> **为 LLM Serving 构建形式化的数学模型，并在此基础上设计具有可证明性能保证的算法**。

具体提出的创新路径包括：

| 问题 | 新思路 |
|------|--------|
| **Expert Routing in MoE** | 将 token 分配建模为 **Linear Program (LP)**，以最小化最大 GPU 负载（min-max load balancing） |
| **Data Parallelism Load Balancing** | 提出 **Online Integer Optimization** 框架，在 sticky assignment 和 barrier synchronization 下进行短视界（lookahead）负载优化 |
| **Worker-Level Scheduling** | 利用 **Queueing Theory** 推导稳定性条件，实现**先验容量规划**（proactive capacity planning）而非反应式扩缩容 |
| **Multimodal Caching** | 设计 **Cost-Aware Eviction Policy**（如 Least Expected Cost, LEC），综合考虑对象大小、访问概率与重计算代价 |

此外，作者强调：**理论的价值不仅在于直接部署求解器，更在于揭示最优策略的结构**（如对偶变量、阈值规则），从而指导高效启发式的设计。

---

### ✅ 相比现有方法的优势

| 维度 | 启发式方法（Heuristics） | 本文倡导的优化方法 |
|------|--------------------------|--------------------|
| **鲁棒性** | 在特定 trace 上有效，但在异常负载下易崩溃 | 具备 **worst-case guarantees**（如 competitive ratio），保障极端情况下的性能下限 |
| **可解释性** | 黑箱行为，难以调试与调优 | 显式目标函数与约束，提供**算法结构洞察** |
| **容量规划能力** | 依赖试错与监控，响应滞后 | 可推导**闭式稳定性条件**，支持部署前容量估算 |
| **工程指导性** | 局部优化，缺乏统一框架 | 提供**设计蓝图**，帮助构建更稳健的系统架构 |
| **长期适应性** | 模型/硬件变更需重新调参 | 理论洞察具有**跨代迁移性**，适用于未来系统演进 |

---

## 2. **核心实验方法和设置**

> ⚠️ 注意：本论文是一篇 **position paper（立场论文）**，**并未报告作者自身的端到端实验**，而是综述并引用近年来多个前沿研究工作来支撑其观点。

### ✅ 使用的研究案例与数据来源

论文通过分析以下四类代表性工作的实证结果来论证其主张：

| 示例 | 来源文献 | 方法类型 |
|------|---------|----------|
| **MoE 负载均衡** | DeepSeek (2025b) 的 LPLB 系统 | LP-based real-time token redistribution |
| **DP 负载均衡** | Chen et al. (2026) | Online integer optimization with lookahead |
| **Worker 内调度与容量规划** | Anonymous (2025), Ao et al. (2025), Jaillet et al. (2025) | Queueing models, fluid approximations, hindsight optimal benchmarks |
| **多模态缓存优化** | Zhu et al. (2023) | Cost-aware caching with optimal regret |

所用 workload 包括：

- **真实 LLM trace**：ShareGPT、ChatGPT 日志等
- **合成负载**：用于测试 adversarial arrival patterns
- **多模态输入分布**：图像分辨率、视频长度差异带来的 embedding 大小变化

---

### ✅ 实验设置与评估指标

尽管无统一实验平台，各被引研究共享以下评估维度：

#### 🔹 主要评估指标

| 指标 | 描述 |
|------|------|
| **Throughput (req/sec)** | 单位时间内完成的请求数量 |
| **Latency (TTFT / TPOT)** | Time-To-First-Token, Time-Per-Output-Token |
| **GPU Utilization** | 计算与内存带宽利用率 |
| **Load Imbalance** | 最大与平均 worker 负载之比 |
| **Cache Hit Rate / Cost Reduction** | 缓存命中率及总推理成本降低 |
| **Stability Region** | 系统在给定资源配置下能否保持队列稳定 |

#### 🔹 基线方法对比

| 类别 | 常见 Baseline |
|------|--------------|
| **Routing** | Round-Robin, Power-of-Two-Choices, Consistent Hashing |
| **Scheduling** | FCFS, Shortest-Job-First (SJF, oracle-assisted) |
| **Eviction** | LRU, LFU |
| **Load Balancing** | GShard-style auxiliary loss, random noise injection |

---

## 3. **主要实验结果和性能指标**

以下是论文中引用的关键研究成果及其性能提升：

### ✅ Example 1: LP-Based MoE Load Balancing (LPLB)

- **方法**：每 batch 求解一个最小化最大负载的 LP，利用冗余专家副本动态重分配 token 流量。
- **性能结果**：
  - 实现接近最优的负载均衡（within 5% of theoretical lower bound）
  - GPU idle time 减少 >40%
  - **求解延迟仅 ~100μs**，可在 decode step（通常 30–100μs）内完成，具备实时可行性
- **对比基线**：优于 GShard 的 auxiliary loss 方法，避免梯度干扰且无需超参调节

---

### ✅ Example 2: Online Optimization for DP Load Balancing (Chen et al., 2026)

- **方法**：基于短期完成预测，求解 lookahead H 步内的整数优化问题，最小化累计不平衡。
- **理论保证**：
  - 证明在 adversarial 请求序列下，平均不平衡降低因子为 **Ω(√B log G)**  
    （B = per-worker batch size, G = worker count）
- **实际意义**：集群规模越大，优化收益越显著
- **对比基线**：相比 Power-of-Two-Choices 或 Random，在高负载下吞吐提升达 20–30%

---

### ✅ Example 3: Queueing-Theoretic Stability Analysis

- **方法**：建立融合 compute 与 memory 约束的 queueing model，推导系统稳定的充要条件。
- **结果**：
  - 可提前计算最小所需 GPU 数量，防止运行时内存溢出或队列爆炸
  - 在真实系统中预测准确率 >90%（vs. reactive autoscaling 的事后发现）
- **价值**：实现 **proactive capacity planning**，减少 over-provisioning 成本

---

### ✅ Example 4: Cost-Aware Caching (Zhu et al., 2023)

- **方法**：提出 **Least Expected Cost (LEC)** 策略，eviction score = (recomputation cost / size) × access probability
- **理论贡献**：实现 **optimal regret bound** —— 学习速度达到信息论极限
- **实验结果**：
  - 当昂贵操作（如高清视频编码）与便宜操作共存时，**总成本下降高达 50×**
  - 在真实 LLM 多模态 workload 上：
    - FLOPs 减少 **4.3×**
    - 端到端延迟降低 **1.8×**

---

## 4. **关键结论和发现**

### ✅ 主要结论

1. **LLM Serving 已超越启发式的适用边界**，必须引入**数学优化与算法理论**作为基础支撑。
2. **现有通用策略（RR、FIFO、LRU）无法捕捉 LLM 推理的独特结构**，导致效率损失与鲁棒性不足。
3. **形式化建模能带来四大不可替代优势**：
   - Worst-case robustness（对抗性负载下的性能保障）
   - Fundamental limits（指导容量规划）
   - Algorithmic structure（启发工程实践）
   - Optimality baselines（避免过度优化）
4. **历史先例表明此路径可行**：航空收益管理（Airline Revenue Management）通过 LP 对偶导出 bid-price 控制策略，实现了十亿美元级收益，且运行时仅为 O(1) 规则。
5. **新兴研究已验证该范式的有效性**：在线优化、排队论、成本感知缓存等方法已在真实系统中展现显著性能增益。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **信息获取挑战** | 如 decode length prediction 不准，影响优化效果（但可通过 robust optimization 缓解） |
| **计算开销顾虑** | 实时求解 LP/IP 是否可行？文中指出现代 GPU solver 可在 100μs 内完成小型 LP，已满足要求 |
| **系统集成复杂度** | 将理论算法嵌入生产系统需要跨学科协作（OR + Systems） |
| **非稳态环境适应性** | workload drift、模型更新可能导致模型失效，需持续学习机制 |

---

### 🔮 未来研究方向（作者建议）

1. **Scheduling with Predictions under Uncertainty**  
   如何联合设计 prediction model 与 scheduling algorithm？如何平衡 consistency 与 robustness？

2. **Multi-Objective Optimization**  
   如何权衡 TTFT、TPOT、throughput、energy、fairness？构建 Pareto frontier 分析框架。

3. **Theoretical Foundations for Disaggregation**  
   Prefill-Decode disaggregation 的理论优势边界是什么？何时应拆分？资源比例如何配置？

4. **Algorithmic Foundations for Agentic Inference**  
   Agent 场景下的分支、暂停、子任务依赖等新型 workload，亟需新的调度模型。

5. **Cross-Disciplinary Collaboration**  
   鼓励 **optimization 研究者深入理解系统瓶颈**，同时 **systems 研究者学习算法理论中的结构洞察**。

---

## ✅ 总结

> **这不是一篇提出新算法的论文，而是一份呼吁变革的宣言**。

它明确指出：随着 LLM Serving 进入大规模工业化阶段，**靠“拍脑袋”的启发式调优已走到尽头**。未来的突破将来自 **Operations Research、Queueing Theory、Online Optimization 与 ML Systems 的深度融合**。

通过引入**数学建模 → 结构分析 → 可证明算法 → 工程近似**这一科学闭环，我们不仅能做出更快的系统，更能做出**更可靠、更可解释、更具前瞻性的智能基础设施**。

</details>

---

### 12. [On Stable Long-Form Generation: Benchmarking and Mitigating Length Volatility](https://arxiv.org/abs/2605.01357)

**Authors**: Zhitao He, Haolin Yang, Rui Min, Zeyu Qin, Yi R. Fung  
**Category**: cs.CL  
**Published**: 2026-05-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.01357v1  

#### Abstract
Large Language Models (LLMs) excel at long-context understanding but exhibit significant limitations in long-form generation. Existing studies primarily focus on single-generation quality, generally overlooking the volatility of the output. This volatility not only leads to significant computational...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：On Stable Long-Form Generation: Benchmarking and Mitigating Length Volatility

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文聚焦于 **Large Language Models (LLMs)** 在长文本生成（long-form generation）中的一个被忽视但严重的问题：**输出长度波动性（length volatility）**。尽管当前模型在长上下文理解（long-context understanding）方面取得了显著进展，但在生成长文本时，其输出长度存在极大的不稳定性。这种波动不仅导致计算资源浪费，也严重影响了模型在实际应用中的可靠性和可控性。

现有研究大多关注单次生成的质量，而忽略了多次运行下的输出一致性。例如，当要求模型生成一篇10万字的文章时，不同次运行的结果可能从几千字到几万字不等，且常伴有提前终止、重复循环或结构跳过等问题。

### 提出的新方法与新思路
为系统性地解决此问题，作者提出了一个三阶段的研究框架：**Benchmarking（基准测试）、Probing（探因分析）和 Mitigating（缓解策略）**。

#### 主要贡献：
1. **VOlatility in Long-form Text Benchmark (VOLTBench)**  
   - 这是一个全新的、异构任务（heterogeneous-task）的基准，旨在**量化长文本生成中的长度波动性**。
   - 与以往仅关注单一任务（如故事生成）的基准不同，VOLTBench 覆盖了多种维度：
     - **任务类型**：包括非结构化任务（如故事、日记、对话）和结构化任务（如代码库、公司信息、数学公式）。
     - **语言**：支持中英文双语评估。
     - **指令复杂度**：从简单提示到包含细粒度约束的复杂指令。
     - **长度规模**：可扩展至最多500章，总长度达10万词以上。
   - 首次将 **multiple sampling** 和 **stability evaluation** 作为核心评估范式，以衡量模型的可靠性而非偶然表现。

2. **Stable Generation via Logits Boosting (GLoBo)**  
   - 一种轻量级、无需额外训练的解码阶段优化策略，用于抑制长文本生成中的不稳定行为。
   - GLoBo 的核心思想是通过动态调整 **logits** 来引导生成过程，具体包含两个机制：
     - **Hybrid Structural Enforcement**：结合“软等待”和“硬强制”两种模式，在满足自然断句条件时提升下一章节标题的生成概率；若模型迟迟不结束当前段落，则在达到上限后强制切换。
     - **Proactive Failure Prevention**：主动抑制与失败模式相关的 token，如对话填充语（"I hope these..."）或过早的 `</s>` 结束符，防止模型提前退出或陷入聊天模式。

### 相比现有方法的优势
| 方面 | 现有方法 | 本文方法 |
|------|--------|---------|
| **目标** | 提高生成质量或长度控制 | 同时提高**长度准确性**与**输出稳定性** |
| **是否需要训练** | 多数需微调或强化学习（如 LongWriter, LongDPO） | **无需训练**，纯解码干预 |
| **评估维度** | 单次生成质量 | 多轮采样下的分布稳定性 |
| **适用范围** | 特定任务或格式 | 可泛化至自由形式（free-form）任务 |

---

## 2. 核心实验方法和设置

### 数据集
- **VOLTBench** 是本文构建的核心基准，包含以下任务类别：
  - **Unstructured Tasks**：Story（小说）、Diary（日记）、Dialogue（对话）、Architecture（建筑描述）
  - **Structured Tasks (GenData)**：Code Function（Python函数库）、User Info（虚拟用户档案）、Company Info（公司简介）、Math LaTeX Formula（数学公式）
- 所有任务均设计为可变长度（5–500节），并引入**细粒度约束**（fine-grained constraints）进行精细化评估：
  - 字符级模式约束（首字母限定）
  - 关键词必须出现
  - 指定主题一致性

### 实验设置
- **模型选择**：
  - 商用闭源模型：`GPT-4o mini`, `Claude-3.5-Sonnet`
  - 开源模型：`Qwen2.5-1.5B/7B`, `Llama3.1-8B`, `Deepseek-V3/R1`, `Mamba-7B`
  - 专门长文本模型：`LongWriter-8B`（基于 Llama 微调）
- **评估方式**：
  - 对每个任务进行 **5次独立采样**（N=5），以统计输出长度的标准差和变异系数。
  - 使用 `temperature=0.7`, `top_p=0.9` 等标准参数。
- **对比基线**：
  - 原始模型（Baseline）
  - 常见解码策略：
    - Repetition Penalty
    - Entropy-Based Stopping
    - Length Constraint
    - Lookahead Decoding

### 评估指标
| 类别 | 指标 | 定义 |
|------|------|------|
| **长度波动性** | LSD (Length Standard Deviation) | 输出长度的标准差 |
| | LVC (Length Variation Coefficient) | LSD / 平均长度，用于跨尺度比较 |
| | MLA (Mean Length Accuracy) | 平均输出长度与目标长度的匹配程度 |
| **生成质量** | FSD (Format Standard Deviation) | 生成节数的标准差 |
| | SCA (Structured Content Accuracy) | 结构化任务中正确章节的比例（执行验证） |
| | UCA (Unstructured Content Accuracy) | 非结构化任务中由 LLM-as-a-Judge 给出的内容评分 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **100节生成任务** 上，基线模型普遍表现不佳：
  - `LongWriter-8B` 平均生成 6,320 词，但 **LVC 高达 45.4%**，表明极不稳定。
  - 多数模型无法完成全部100节，平均仅生成不到一半。
- **GLoBo 显著改善性能**：
  - **平均输出长度提升 148%**：从基线约 6,320 词提升至 **15,651 词**。
  - **长度波动降低 69%**：LVC 从 45.4% 下降至 **14.02%**。
  - **MLA 提升超过两倍**：从 31.6% 提升至 **78.25%**。
  - **SCA 达到 100%**：所有生成的章节均符合结构要求，远超 LongWriter 的 32.6%。

### 与基线方法的对比结果
| 模型 | LVC ↓ | MLA ↑ | 平均长度 ↑ | SCA ↑ |
|------|-------|--------|------------|--------|
| LongWriter-8B | 45.4% | 31.6% | 6,320 | 32.6% |
| + GLoBo (Ours) | **14.02%** | **78.25%** | **15,651** | **100%** |
| Deepseek-V3 | 2.2% | 9.3% | 1,854 | 48.6% |
| Qwen2.5-7B | 17.0% | 2.2% | 445 | 99.8% |

> ✅ GLoBo 不仅提升了长度和稳定性，还保持甚至提高了生成质量（UCA ≈ 86.7%）。

### 消融实验与泛化能力
- **跨模型泛化**：在 `Llama-3.1-8B`, `Mamba-7B`, `Qwen2.5-14B` 上均观察到一致改进（见 Appendix Table 6）。
- **跨任务泛化**：
  - 在外部基准 **LongBench-Write** 上取得 SOTA 总分 **85.3**，优于 Claude 3.5 Sonnet (80.7) 和 LongWriter-9B-DPO (84.0)。
  - 在 **WritingBench** 上平均得分为 **8.43**，接近顶级商用模型。
- **运行效率**：
  - 吞吐量从 20.4 tok/s 降至 18.2 tok/s，**开销仅 10.8%**，远低于 Lookahead Decoding 的 28.9%。

---

## 4. 关键结论和发现

### 主要发现
1. **主流 LLMs 存在严重的长文本生成不稳定性**：即使是最先进的模型，在多次运行下也无法稳定达到目标长度，波动性高达数十个百分点。
2. **结构化任务更易控制**：相比自由创作类任务，结构化任务（如代码、JSON）由于格式明确，反而更容易实现稳定的长文本生成。
3. **Attention Collapse 是根本原因**：通过分析 attention trace 发现，当模型对输入约束的关注度（constraint attention）衰减至零时，会触发提前终止或重复循环，称为 **Attention Collapse** 或 **Attention Instability**。
4. **GLoBo 成功缓解内部崩溃**：通过周期性地 boost 章节标题的 logits，GLoBo 能有效“重同步”模型状态，维持 attention focus，从而避免 representational drift。

### 方法的局限性
- 当前方法依赖于显式的章节结构（如 "Chapter X"）。对于完全无结构的自由写作任务，需通过自动划分里程碑（milestones）来适配。
- 在极端长度（如 500 节）下，虽然 GLoBo 是唯一能生成大量内容的方法，但仍未能完全达到目标（平均 327 节 vs 目标 500 节）。
- 对某些特定领域（如对话）的建模能力仍有待加强，所有模型在此类任务上表现均较差。

### 未来工作方向
- 将 GLoBo 与训练阶段的方法（如 RLHF 或 DPO）结合，探索联合优化路径。
- 探索更智能的动态 milestone 划分机制，适应不同文体节奏。
- 扩展至多模态长序列生成（如图文报告、视频脚本）。
- 构建更大规模的人类评估体系，进一步验证感知质量与自动指标的一致性。

---

> **总结一句话**：本文首次系统揭示了 LLM 长文本生成中的**长度波动性**问题，提出 **VOLTBench** 作为首个稳定性评估基准，并设计了无需训练的轻量级解码策略 **GLoBo**，实现了生成长度、稳定性和质量的全面突破。

</details>

---

### 13. [SURGE: SuperBatch Unified Resource-efficient GPU Encoding for Heterogeneous Partitioned Data](https://arxiv.org/abs/2605.01060)

**Authors**: Shashank Kapadia, Deep Narayan Mishra, Sujal Reddy Alugubelli, Ajay Kumar, Swapnil Yadav, Rishi Bhatia  
**Category**: cs.DC  
**Published**: 2026-05-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.01060v1  

#### Abstract
We present SURGE, a streaming GPU encoding system deployed in production to generate embeddings for over 800 million texts across 40,000 logical partitions. Production embedding pipelines face a tension between logical data partitioning and efficient GPU utilization: processing each partition indepe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SURGE: SuperBatch Unified Resource-efficient GPU Encoding for Heterogeneous Partitioned Data

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代生产级文本嵌入系统面临**逻辑数据分区**（logical data partitioning）与**高效 GPU 利用率**之间的根本矛盾：
- **Partition-by-Partition (PBP)** 方法独立处理每个分区，导致大量 `P` 次进程间通信（IPC），在小模型下 IPC 开销可占总耗时近一半，严重限制吞吐量。
- **Fixed-Size Batching (FSB)** 忽略分区边界进行固定大小批处理，虽能摊销 IPC 成本，但需要 $O(N)$ 峰值内存（如 10M 文本需 32.7GB），且无输出直到全部编码完成，缺乏容错能力。

### 提出的新方法：SURGE
提出 **SURGE** —— 一种流式 GPU 编码系统，通过 **SuperBatch 聚合机制** 统一解决效率与资源消耗问题。

#### 核心创新点：
1. ✅ **成本模型（Theorem 1）**  
   建立了基于 IPC 和计算开销的解析模型，预测任意批处理策略的吞吐量，误差小于 2%，适用于跨参数范围（15×）的多种 encoder。

2. ✅ **内存安全边界（Lemma 3）**  
   提出 **双阈值流式聚合策略**（two-threshold streaming policy），实现峰值内存为 $O(B_{\text{min}} + n_{\text{max}})$，而非 $O(N)$，显著降低内存占用。

3. ✅ **决策框架（δ/CV 决策框架，Section 7）**  
   提出一个通用判断准则：当 IPC 占主导比例 δ 较高 或 分区大小变异系数 CV 较大时，SURGE 模式具有显著优势。

4. ✅ 工程实现三大技术（非核心贡献但支撑设计）：
   - **零拷贝 Arrow 序列化**：避免创建 $O(Nd)$ Python 对象，速度提升 22–25×。
   - **异步 I/O 流水线**：重叠序列化与上传操作，最高减少 93% I/O 阻塞。
   - **SuperBatch 聚合器**：按分区累积至阈值后统一编码，支持增量刷盘。

### 相比现有方法的优势
| 维度 | PBP | FSB | SURGE |
|------|-----|-----|-------|
| 吞吐量 | 低（IPC 多） | 高（摊销 IPC） | ✅ 高（同 FSB） |
| 峰值内存 | $O(n_{\text{max}})$ | $O(N)$ | ✅ $O(B_{\text{min}} + n_{\text{max}})$ |
| Time-to-First-Output (TTFO) | 快（逐个输出） | 极慢（全完才出） | ✅ 快（增量刷盘） |
| 容错性 | 弱（无恢复粒度） | 无（全丢） | ✅ 支持 SuperBatch 粒度恢复 |

---

## 2. 核心实验方法和设置

### 数据集
- **合成数据集**：10M 文本，分布在 4,000 个逻辑分区中，分区大小服从对数正态分布（$\mu=9.03, \sigma=1.72$），平均文本长度 47 字节。
- **扩展测试**：扩展到 50M 文本、最多 20,000 分区以验证可扩展性。
- **真实部署**：已在沃尔玛生产环境处理 **超过 8 亿条文本**，涵盖 40,000 个逻辑分区。

### 实验设置
- **硬件**：Google Cloud g2-standard-48 节点，配备 **4× NVIDIA L4 GPU**（每块 24GB VRAM），48 vCPU，192GB RAM。
- **模型**：
  - `all-MiniLM-L6-v2`（22M 参数，d=384）
  - `bge-base-en-v1.5`（109M 参数，d=768）
  - `E5-large`（335M 参数，d=1024）
- **存储后端**：模拟 GCS 存储延迟（基础延迟 10ms，带宽 200MB/s）。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Throughput (t/s)** | 每秒处理的文本数量 |
| **TTFO (Time-to-First-Output)** | 第一个分区输出的时间 |
| **Peak Memory (RSS)** | 进程常驻内存峰值（不含 GPU 显存） |
| **GPU Utilization (%)** | `nvidia-smi` 报告的 GPU 利用率 |
| **Encode Duty Cycle (δ)** | 编码调用时间占总墙钟时间的比例 |
| **Cost ($/M texts)** | 每百万文本处理成本 |

### 基线方法对比
1. **PBP**：逐分区独立编码。
2. **Fixed-Batch (FSB)**：忽略分区，固定大小批处理 + 最终按标签重组。
3. **PB-PBP-LB**：更强基线，离线排序分区并打包成大批次（First-Fit Decreasing），但仍需预知所有分区大小。
4. **SURGE (sync)**：无异步 I/O 的版本。
5. **SURGE + AsyncIO**：完整系统。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（MiniLM-L6-v2, 10M 文本）

| 方法 | 吞吐量 (t/s) | 峰值内存 (GB) | TTFO (s) | 成本 ($/M) |
|------|-------------|---------------|----------|------------|
| PBP | 13,766 | 2.5 | 0.5 | 0.15 |
| FB-100K | 27,074 | 32.7 | 245.5 | 0.07 |
| **SURGE (ours)** | **26,413** | **2.6** | **3.6** | **0.08** |

✅ **关键结论**：
- SURGE 吞吐量达到 FSB 水平（仅低 2.4%），但内存仅为 **1/12.6**（2.6GB vs 32.7GB）。
- TTFO 仅 **3.6 秒**，比 FSB 快 **68×**。
- 成本与 FSB 相当（$0.08/M），相比 PBP 节省 **47%**。

### 与其他模型的泛化性验证（Table 4）

| 模型 | 方法 | 吞吐量 (t/s) | GPU% | 峰值内存 (GB) | TTFO (s) |
|------|------|--------------|--------|----------------|-----------|
| bge-base | PBP | 7,154 | 31.7 | 3.2 | 0.23 |
| bge-base | FB-100K | 9,282 | 41.7 | 63.4 | 835 |
| bge-base | SURGE | 9,250 | 42.1 | 3.3 | 10.7 |

- SURGE 在更大模型上仍保持与 FSB 接近的吞吐量。
- 内存优势进一步放大：**19.2× 更少内存**（63.4GB → 3.3GB）。
- TTFO 优势达 **78×**。

### 消融实验结果（Table 3）

| 配置 | 吞吐量 (t/s) | 相对下降 |
|------|-------------|-----------|
| 完整系统 | 25,930 | — |
| w/o SURGE (PBP) | 13,828 | ↓46.7% |
| w/o AsyncIO | 21,689 | ↓16.4% |
| w/o Zero-copy | 14,832 | ↓42.8% |
| w/o Multi-GPU | 10,536 | ↓59.4% |

📌 发现：
- **SuperBatch 聚合** 是最大贡献者（↑46.7%）。
- **零拷贝序列化** 影响巨大，因其避免了 GC 压力和内存膨胀。
- **异步 I/O** 在高延迟存储下至关重要（见下表）。

### 异步 I/O 效益随存储延迟增加而上升（Table 6）

| 存储场景 | Sync 吞吐 (t/s) | Async 吞吐 (t/s) | 提升 |
|---------|------------------|-------------------|------|
| Null（无 I/O） | 25,910 | 25,739 | -0.7% |
| Cross-region | 13,641 | 26,235 | **+92.3%** |

✅ 异步流水线在高延迟环境下几乎完全消除 I/O 阻塞。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **IPC 开销是轻量模型吞吐瓶颈的关键因素**，即使只有 23% 的分区本身处于 IPC 主导区，其累计开销仍占 PBP 总时间的 **48%**。
2. 📉 **SURGE 实现了 FSB 的吞吐上限，同时具备 PBP 的部署友好特性**：低内存、快速首出、容错恢复。
3. 📊 **内存优势随数据规模增长而扩大**：在 50M 文本时，SURGE 内存为 8.7GB，而 FSB 达 162.7GB（**18.8× 差距**），后者已接近 192GB 节点极限。
4. ⏱️ **TTFO 几乎恒定（~3.6s）**，不随数据总量增长，适合监控与故障排查；而 FSB 的 TTFO 线性增长，在 50M 时达 **20 分钟以上**。
5. 🧪 **理论模型高度准确**：Theorem 1 预测误差始终 <2%，可用于实际 workload 规划。

### 方法的局限性
- ❗ **依赖输入有序性**：要求输入按 partition key 排序，否则需前置排序步骤（$O(N \log N)$）。
- ❗ **阈值需调优**：$B_{\text{min}}$ 需根据实际 workload 调整，过小无法摊销 IPC，过大则 TTFO 上升。
- ❗ **单节点设计**：当前为单节点多 GPU 架构，未解决跨节点调度问题（作者指出可分解为 per-node 应用）。
- ❗ **长期运行内存碎片风险**：若未启用 PyTorch 的 `expandable_segments`，长时间运行可能导致 GPU OOM。

### 未来工作方向
1. **自适应阈值选择**：根据实时观察的分区统计动态调整 $B_{\text{min}}$ 和 $B_{\text{max}}$。
2. **优化极端分区数下的管理开销**：在 $P > 10,000$ 时出现吞吐下降（Table 9），需优化元数据跟踪与文件路径生成。
3. **扩展至千亿参数模型**：验证在 compute-intensive 极端下的有效性（目前最大测试为 335M 参数的 E5-large）。
4. **多节点扩展与负载均衡**：研究如何将分区智能路由到不同节点以平衡各节点的 $\alpha$（IPC-to-compute ratio）。

---

> 💡 **总结一句话**：  
> **SURGE 不是一个简单的工程优化，而是提出了一个“流式聚合 + 内存有界 + IPC 摊销”的通用范式，使大规模异构分区数据的 GPU 编码既高效又可部署。**

</details>

---

### 14. [FedPLT: Scalable, Resource-Efficient, and Heterogeneity-Aware Federated Learning via Partial Layer Training](https://arxiv.org/abs/2605.02337)

**Authors**: Ahmad Dabaja, Rachid El-Azouzi  
**Category**: cs.DC  
**Published**: 2026-05-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.02337v1  

#### Abstract
Federated Learning (FL) has gained significant attention in distributed machine learning by enabling collaborative model training across decentralized system while preserving data privacy. Although extensive research has addressed statistical data heterogeneity, FL still faces several challenges, in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FedPLT: Scalable, Resource-Efficient, and Heterogeneity-Aware Federated Learning via Partial Layer Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
联邦学习（Federated Learning, FL）在实际部署中面临三大挑战：
- **高通信与计算开销**：全模型训练需要大量带宽和算力，对边缘设备不友好。
- **严重的系统异构性**：客户端设备能力差异大（如 IoT 设备 vs 智能手机），导致“straggler”问题。
- **统计异构性（Non-IID 数据）**：各客户端数据分布不同，易引发 client drift 和收敛不稳定。

现有方法如 **sub-model training**（如 HeteroFL, FedDrop）和 **partial parameter training**（如 FedPMT）虽尝试缓解上述问题，但存在以下缺陷：
- 参数更新分布不均，造成优化偏差；
- 随机掩码引入高方差，影响收敛稳定性；
- 无法保证全局模型所有部分都被充分训练。

---

### 提出的新方法：FedPLT
作者提出 **FedPLT (Federated Learning with Partial Layer Training)**，一种**结构化的部分层训练框架**，其核心思想是：

- 将每一层划分为多个等大小的 **sub-layer（子层）**；
- 根据客户端的通信与计算能力，为其分配一组可训练的 sub-layer；
- 所有客户端仍接收完整的全局模型进行前向传播，仅更新指定的 sub-layer 参数；
- 采用**固定且轮转式的分配策略**，确保每个 sub-layer 在全局范围内被均衡地更新。

该方法实现了：
- **Fine-grained 资源适配**：灵活匹配不同能力的设备；
- **Balanced parameter exposure**：避免某些参数长期未被更新；
- **Preserved full-model semantics**：前向过程完整，损失函数计算准确。

---

### 相比现有方法的优势
| 特性 | FedPLT | FedPMT | HeteroFL | FedDrop | FedRolex |
|------|--------|--------|---------|---------|----------|
| 是否保持全模型前向 | ✅ 是 | ✅ 是 | ❌ 否 | ❌ 否 | ❌ 否 |
| 更新是否结构化 | ✅ 固定分块 | ❌ 冻结浅层 | ❌ 截断神经元 | ❌ 随机丢弃 | ⚠️ 滚动掩码 |
| 参数暴露均衡性 | ✅ 高（轮转机制） | ❌ 浅层少更新 | ❌ 弱设备参数欠训练 | ⚠️ 时间平均后均衡 | ⚠️ 需多轮旋转 |
| 收敛稳定性 | ✅ 高 | ⚠️ 中等 | ⚠️ 存在偏置 | ❌ 波动大 | ⚠️ 易遗忘早期学习 |
| 可扩展至资源极度受限系统 | ✅ 支持 | ❌ 依赖强设备 | ❌ 依赖强设备 | ✅ 支持 | ✅ 支持 |

> ✅ 表示优势明显，⚠️ 表示一般，❌ 表示劣势。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Fashion-MNIST**：用于 FCN 模型测试，模拟轻量级图像分类任务。
- **CIFAR-10**：用于两种模型架构验证：
  - 自定义 **Fully Connected Network (FCN)**
  - **ResNet-8**（简化版 ResNet）

所有数据通过 **Dirichlet 分布（α=0.2 或 0.1）** 划分到客户端，以模拟 **Non-IID 场景**。

---

### 实验设置
#### 客户端配置
共设计三种实验场景：

| 场景 | 描述 |
|------|------|
| **Homogeneous Low-Resource** | 所有客户端只能训练小比例参数（如 18%-29%），模拟 IoT 网络 |
| **Heterogeneous System** | 客户端能力三级分化：<br>• 10% 高能力（rk=1.0）<br>• 30% 中等能力（rk≈0.23–0.29）<br>• 60% 低能力（rk=0.06） |
| **FedPLT + OCS** | 结合改进的 Optimal Client Sampling，在严苛通信预算下评估性能 |

#### 评估指标
- **主指标**：最终验证集 **classification accuracy (%)**
- **辅助指标**：
  - 收敛速度（accuracy vs round）
  - 方差（多次运行的标准差）
  - 通信成本（GB）
  - 计算成本（TFLOPs）
  - 轮次时间（mitigate straggler 效果）

#### 基线方法对比
- **FedAvg**：标准全模型训练基准
- **FedPMT**：冻结浅层，只训练深层
- **HeteroFL**：按设备能力裁剪模型宽度
- **FedDrop**：每轮随机丢弃神经元
- **FedRolex**：滚动掩码更新不同区域

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 方法 | Fashion-MNIST + FCN | CIFAR-10 + FCN | CIFAR-10 + ResNet-8 |
|------|---------------------|----------------|-----------------------|
| **FedAvg** | 74.33% ±9.53 | 47.42% ±4.12 | 79.61% ±1.10 |
| **FedPLT (最优平衡)** | **78.08% ±4.65** | 46.56% ±0.71 | 78.21% ±1.57 |
| **FedPMT** | 65.81% ±5.80 | 39.16% ±3.47 | 75.57% ±3.04 |
| **FedRolex** | 74.32% ±3.85 | 37.65% ±1.32 | 9.96% ±1.48 |
| **FedDrop** | 52.88% ±8.35 | 26.60% ±2.94 | 14.53% ±0.81 |
| **HeteroFL** | 16.56% ±19.70 | 34.81% ±8.31 | 46.22% ±1.58 |

> 💡 **观察**：FedPLT 在多数情况下达到甚至超过 FedAvg 性能，同时显著优于其他 partial training 方法。

---

### 与基线方法的对比结果（提升幅度）

在 **CIFAR-10 + FCN** 上，FedPLT 相比其他方法的精度提升为：
- **vs FedPMT**: +7.40%
- **vs FedRolex**: +8.91%
- **vs HeteroFL**: +11.75%
- **vs FedDrop**: +19.96%

在 **CIFAR-10 + ResNet-8** 上：
- **vs FedPMT**: +2.64%
- **vs HeteroFL**: +31.99%
- **vs FedRolex/FedDrop**: 提升超 60%，且后者几乎完全失败（<15%）

> 📈 **结论**：FedPLT 在高度异构环境下表现最稳健，尤其在复杂模型上优势显著。

---

### 消融实验结果

#### （1）不同参数分配策略的影响（平衡性分析）
使用 **imbalance error ε(X)** 度量分配不平衡程度，结果显示：
- **越接近“most-balanced”配置，性能越好**
- 在 Fashion-MNIST + FCN 上：
  - 最平衡配置：78.08%
  - 大幅不平衡配置：76.32%（↓1.76%）
- 标准差也随不平衡增加而上升 → **验证了平衡分配的重要性**

#### （2）资源效率增益
FedPLT 将每客户端可训练参数减少 **71%–82%**，仍能达到与 FedAvg 相当或更优性能。

具体节省：
- **通信成本降低**：uplink 减少 `1−rk`，总通信量减少最多达 **48.5%**
- **计算成本降低**：backward 阶段节省 `(1−rk)×βTP`，最高达 **64.7%**
- **缓解 straggler**：全局轮次时间从 17.65s 缩短至 4.35s（↓75.35%）

#### （3）结合 FedPLT-aware OCS 的效果
将传统 OCS 扩展为考虑 `rk` 的通信成本约束后：
- 在低资源系统中（90% 客户端 rk=0.2）：
  - 原始 OCS + FedPLT：accuracy ↓1.11%
  - **FedPLT-aware OCS**：accuracy ↑**+1.29%**
- 在中等资源系统中（rk=0.5）：accuracy ↑**+3.84%**

> 🔍 **原因**：新采样策略允许更多低 `rk` 客户端参与，提升数据多样性与参数覆盖。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Partial Layer Training 可媲美全模型训练**  
   FedPLT 通过细粒度 sub-layer 分配，在仅训练 **18%-29% 参数**的情况下，实现与 FedAvg 相当甚至更优的性能。

2. ✅ **结构化分配优于随机或粗粒度策略**  
   固定+轮转的 sub-layer 分配机制有效缓解了参数偏置和梯度冲突，提升了训练稳定性和泛化能力。

3. ✅ **显著降低通信与计算开销**  
   - 通信成本下降 **40%-50%**
   - 计算负载下降 **50%-65%**
   - 轮次时间缩短 **75%+**，有效缓解 straggler

4. ✅ **适用于极端资源受限环境**  
   即使 60% 客户端只能训练 6% 参数，FedPLT 依然能稳定收敛，适合大规模 IoT 部署。

5. ✅ **与 OCS 协同增效**  
   提出的 **FedPLT-aware OCS** 进一步优化了客户端选择策略，在相同通信预算下获得更高精度。

---

### 方法的局限性
- **依赖预设的 sub-layer 划分方式**：如何自适应地决定每层划分数量尚无理论指导。
- **对模型结构敏感**：需针对 CNN、Transformer 等结构定制分块策略（如 ResNet 要尊重 residual block）。
- **聚合复杂度略高**：服务器需维护每个 sub-layer 的独立聚合权重。
- **未解决 Non-IID 下的语义漂移问题**：虽然缓解了 client drift，但未引入显式正则化机制。

---

### 未来工作方向
1. **自动化 sub-layer 划分策略**：基于梯度重要性或参数敏感度动态调整分块粒度。
2. **支持动态能力变化的客户端**：当前为静态分配，未来可研究运行时 re-assignment 机制。
3. **扩展至多模态与大模型**：应用于 Vision Transformer、LLM 等更大规模模型的 FL 场景。
4. **结合差分隐私与安全聚合**：探索在 FedPLT 框架下的隐私保护机制。
5. **跨设备迁移学习**：利用 sub-layer 结构实现模块化知识共享。

---

> 🏁 **总结一句话**：  
> **FedPLT 是一种高效、鲁棒且可扩展的联邦学习框架，它通过结构化的 partial layer training 实现了资源节约与性能保持的统一，特别适用于异构性强、资源受限的真实世界边缘计算系统。**

</details>

---

### 15. [Hybrid Quantum Reinforcement Learning with QAOA for Improved Vehicle Routing Optimization](https://arxiv.org/abs/2605.01574)

**Authors**: T. Satyanarayana Murthy, B. Swathi Sowmya, Santhosh Voruganti, Sai Varshini Giridi, Chaitanyya Pratap Agarwal, Vanteddu Akshitha  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.01574v1  

#### Abstract
Vehicle Routing Problem (VRP) is one of the most complex NP-hard combinatorial optimization problem in transportation and logistics that requires a dynamic solution approach. In this paper we present a new hybrid approach that combines the Quantum Approximate Optimization Algorithm (QAOA) into the Q...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Hybrid Quantum Reinforcement Learning with QAOA for Improved Vehicle Routing Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **Vehicle Routing Problem (VRP)** 是一个经典的 NP-hard 组合优化问题，在物流、运输等领域具有重要应用。随着问题规模增大（如客户数量增加），传统方法（如分支定界、元启发式算法）在时效性和可扩展性上面临挑战。
- 现有的量子方法（如 **QAOA** 和 **Grover Adaptive Search (GAS)**）受限于 **NISQ 设备** 的硬件限制（如 qubit 数量少、噪声大、电路深度有限），难以处理大规模 VRP 实例。
- 传统的 **Quantum Reinforcement Learning (QRL)** 虽然具备动态适应能力，但其使用的通用变分量子电路（PQC）缺乏对问题结构的编码，导致训练缓慢、易陷入“贫瘠高原”（barren plateaus）。

### **提出的新方法与新思路**
- 提出了一种名为 **HQRL-QAOA**（Hybrid Quantum Reinforcement Learning with QAOA）的新型混合框架：
  - 将 **QAOA 的结构化 ansatz**（即 cost Hamiltonian 和 mixing Hamiltonian 层）嵌入到 QRL 的策略网络中，替代传统的通用旋转门层。
  - 引入 **QAOA Warm-Start**：利用 QAOA 在子图上预优化参数，作为 PQC 初始权重，避免随机初始化带来的训练困难。
  - 采用 **三阶段训练流程**：① QAOA 预热 → ② 小规模预训练 → ③ 大规模微调，实现跨问题规模的知识迁移。

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **收敛速度** | 得益于 QAOA 初始化，显著加快训练收敛，减少所需 episode 数量 |
| **可扩展性** | 固定 qubit 数（仅需 4 qubits）和固定电路深度（depth=18），不受问题规模影响 |
| **内存效率** | 内存消耗呈线性增长，远低于 GAS 和标准 QAOA 的指数级增长 |
| **抗噪性** | 浅层电路设计更兼容 NISQ 设备，降低噪声干扰风险 |
| **泛化能力** | 支持 fine-tuning 迁移到更大规模实例，无需从头训练 |

---

## **2. 核心实验方法和设置**

### **数据集与问题实例**
- 自定义生成的 VRP 实例，城市数从 5 到 25 不等，车辆数为 2–3。
- 所有节点坐标在二维平面上随机分布，包含一个中心 depot。
- 实验重点测试了以下规模组合：
  - 小规模：8 cities, 2 vehicles（用于预训练）
  - 大规模：12 cities, 3 vehicles（用于 fine-tuning）
  - 可扩展性测试：5–25 cities

### **实验设置**
- **模拟器平台**：使用 PennyLane 的 `default.qubit` 模拟器，在 Google Colab 上运行。
- **硬件配置**：NQUBITS = 4, NLAYERS = 2, QAOA 层数 $ p = 2 $
- **训练细节**：
  - 总训练 episode 数：250（主实验）
  - 预训练阶段：60 episodes（8 城市）
  - 微调阶段：40 episodes（12 城市）
  - 优化器：COBYLA（用于 QAOA 参数优化），Adam（用于策略更新）
  - 折扣因子 $ \gamma = 0.99 $

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Normalized Route Cost** | 相对于经典最优解（LKH-3）的路径总成本比率 |
| **Peak Memory Usage (MB)** | 训练过程中峰值内存占用 |
| **Episodes to Convergence** | 达到特定奖励阈值所需的 episode 数 |
| **Circuit Depth & Qubit Count** | 衡量量子资源需求的关键指标 |
| **Route Validity & Visualization** | 是否满足约束（无重复访问、返回 depot）、路线合理性 |

### **基线方法对比**
- **Vanilla QRL**：传统 QRL 架构，使用通用 PQC 策略网络
- **QAOA ($p=2$)**：独立运行的标准 QAOA 算法
- **GAS (Grover Adaptive Search)**：基于振幅放大的量子搜索方法
- **Classical (LKH-3)**：经典局部搜索启发式算法，作为性能上限参考
- **Random Policy**：随机选择动作的基准下限

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### ✅ **表1：归一化路径成本对比（越接近 1.00 越好）**

| Method | 5 cities | 8 cities | 10 cities | 15 cities | 25 cities |
|--------|----------|----------|-----------|-----------|-----------|
| **HQRL-QAOA (ours)** | **1.04** | **1.07** | **1.11** | **1.16** | **1.19** |
| Vanilla QRL | 1.12 | 1.18 | 1.23 | 1.31 | 1.42 |
| QAOA($p=2$) | 1.08 | 1.15 | 1.21 | 1.34 | OOM |
| GAS | 1.03 | 1.09 | OOM | OOM | OOM |
| Classical (LKH-3) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

> 注：OOM = Out of Memory

#### ✅ **表2：峰值内存消耗（单位：MB）**

| Method | 5 | 8 | 10 | 12 | 15 | 20 | 25 |
|--------|----|-----|------|------|-------|-------|-------|
| **HQRL-QAOA** | 45 | 52 | 58 | 64 | 71 | 83 | 98 |
| GAS | 32 | 512 | 4096 | OOM | OOM | OOM | OOM |
| QAOA | 28 | 95 | 280 | 850 | OOM | OOM | OOM |

> HQRL-QAOA 内存增长近乎线性，而 GAS 呈指数爆炸式增长。

#### ✅ **电路复杂度对比**
- **Circuit Depth**：
  - HQRL-QAOA：恒定 **18**
  - QAOA：随问题规模线性增长（如 25 cities → depth=200）
  - GAS：指数增长（如 10 cities → depth=4096）
- **Qubit Count**：
  - HQRL-QAOA：始终 **4 qubits**
  - QAOA：$ O(N) $
  - GAS：$ O(N \times K) $

---

### **与基线方法的对比结果**
- **HQRL-QAOA 在所有可运行规模上均优于其他量子方法**：
  - 在 25 城市问题中仍能运行并取得 1.19 成本比，而 GAS 和 QAOA 已因内存不足失败。
  - 收敛速度快于 Vanilla QRL 约 **2–3 倍**，且起始奖励更高。
- **QAOA Warm-Start 显著提升训练稳定性**：
  - 随机初始化模型在前 ~60 episodes 几乎停滞（barren plateau），而 warm-start 模型从第一轮就开始学习。
- **fine-tuning 有效迁移知识**：
  - 在 12 城市任务中，经过预训练 + 微调的 agent 初始奖励即达 -40，而从零开始训练者初始仅为 -95。

---

### **消融实验结果（Ablation Study）**

#### ✅ **表3：组件移除对性能的影响（归一化成本）**

| Configuration | 5 cities | 8 cities | 10 cities | 15 cities |
|---------------|----------|----------|-----------|-----------|
| **Full HQRL-QAOA** | **1.04** | **1.07** | **1.11** | **1.16** |
| w/o QAOA Warm-Start | 1.11 | 1.16 | 1.21 | 1.28 |
| w/o Value Baseline | 1.07 | 1.12 | 1.18 | 1.25 |
| w/o Fine-Tuning | 1.04 | 1.07 | 1.19 | 1.34 |

> 结论：
> - **QAOA Warm-Start 最关键**：缺失时性能下降最大（尤其在小规模）
> - **Value Baseline 提升训练稳定性**
> - **Fine-Tuning 对大规模泛化至关重要**

---

## **4. 关键结论和发现**

### **主要发现**
1. **QAOA Warm-Start 有效规避 barren plateaus**  
   利用 QAOA 预优化参数可使策略网络在训练初期就处于高质量区域，大幅提升学习效率。

2. **固定结构 PQC 实现真正意义上的可扩展性**  
   HQRL-QAOA 使用固定 4-qubit 和 depth=18 的电路，完全摆脱了传统量子算法对问题规模的依赖，是面向 NISQ 设备的理想架构。

3. **性能退化更平缓，内存开销极低**  
   即使在 25 城市问题上，HQRL-QAOA 的成本仅上升至 1.19，而 Vanilla QRL 达 1.42；内存消耗不到 GAS 的 2%（98MB vs >4GB）。

4. **支持跨规模迁移学习（Transfer Learning）**  
   通过预训练 + fine-tuning 策略，可在不同规模 VRP 之间高效迁移策略，减少重复训练成本。

5. **所学策略具备空间感知与负载均衡能力**  
   可视化显示车辆自动形成地理聚类，路径不交叉，体现策略已学会利用空间相关性进行高效分配。

---

### **方法的局限性**
- **依赖子图构建的 QAOA 初始化质量**：若子图代表性不足，warm-start 效果可能打折扣。
- **当前仍在模拟器上验证**：尚未在真实量子设备上部署，实际噪声环境下的表现有待验证。
- **固定 qubit 架构可能限制表达能力**：虽然提升了可扩展性，但在极端复杂场景下可能存在表达瓶颈。

---

### **未来工作方向**
1. **在真实 NISQ 设备上部署 HQRL-QAOA**，验证其在含噪环境中的鲁棒性。
2. **探索更复杂的 VRP 变体**：如带时间窗（VRPTW）、容量限制（CVRP）、动态请求等。
3. **结合 classical pre/post-processing**：例如用经典启发式生成初始解再由量子策略优化。
4. **研究自适应 QAOA 层数选择机制**：根据问题难度动态调整 warm-start 的精度。
5. **扩展至其他组合优化问题**：如 Job Shop Scheduling、TSP、Portfolio Optimization 等。

---

> **总结一句话**：  
> 本文提出的 **HQRL-QAOA** 框架通过将 **QAOA 的结构先验融入 QRL 策略网络**，实现了**快速收敛、低资源消耗、高可扩展性**的量子增强 VRP 求解方案，为 NISQ 时代下的组合优化提供了可行且高效的混合范式。

</details>

---

### 16. [Complex Diffusion Maps with $\omega$-Parameterized Kernels Revealing Inherent Harmonic Representations](https://arxiv.org/abs/2605.01691)

**Authors**: Tongzhen Dang, Weiyang Ding, Michael K. Ng  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.01691v1  

#### Abstract
In this paper, we propose Complex Diffusion Maps (CDM), a novel diffusion mapping framework that aims to reveal the dominant complex harmonics of high-dimensional data. Inspired by the local Gaussian kernel relevant to the heat equation and the nonlocal Schr\"odinger kernel relevant to the Schr\"odi...

---

### 17. [Efficient Preference Poisoning Attack on Offline RLHF](https://arxiv.org/abs/2605.02495)

**Authors**: Chenye Yang, Weiyu Xu, Lifeng Lai  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.02495v1  

#### Abstract
Offline Reinforcement Learning from Human Feedback (RLHF) pipelines such as Direct Preference Optimization (DPO) train on a pre-collected preference dataset, which makes them vulnerable to preference poisoning attack. We study label flip attacks against log-linear DPO. We first illustrate that flipp...

---

### 18. [Focus on the Core: Empowering Diffusion Large Language Models by Self-Contrast](https://arxiv.org/abs/2605.01373)

**Authors**: Jinyuan Feng, Xin Yu, Yiqun Chen, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Zhiqiang Pu  
**Category**: cs.CL  
**Published**: 2026-05-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.01373v1  

#### Abstract
The iterative denoising paradigm of Diffusion Large Language Models (DLMs) endows them with a distinct advantage in global context modeling. However, current decoding strategies fail to leverage this capability, typically exhibiting a local preference that overlooks the heterogeneous information den...

---

### 19. [GhostServe: A Lightweight Checkpointing System in the Shadow for Fault-Tolerant LLM Serving](https://arxiv.org/abs/2605.00831)

**Authors**: Shakya Jayakody, Youpeng Zhao, Chinmay Dhanraj Nehate, Jun Wang  
**Category**: cs.DC  
**Published**: 2026-05-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.00831v1  

#### Abstract
The rise of million-token, agent-based applications has placed unprecedented demands on large language model (LLM) inference services. The long-running nature of these tasks increases their susceptibility to hardware and software faults, leading to costly job failures, wasted resources, and degraded...

---

### 20. [Activation Compression in LLMs: Theoretical Analysis and Efficient Algorithm](https://arxiv.org/abs/2605.01255)

**Authors**: Wen-Da Wei, Han-Bin Fang, Yang-Di Liu, Jiang-Xin Shi, James Kwok, Yu-Feng Li  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.01255v1  

#### Abstract
Training large language models (LLMs) is highly memory-intensive, as training must store not only weights and optimizer states but also intermediate activations for backpropagation. While existing memory-efficient methods largely focus on gradients and optimizer states, activation compression is les...

---

### 21. [Protein-Conditioned Multi-Objective Reinforcement Learning for Full-Length mRNA Design](https://arxiv.org/abs/2605.01513)

**Authors**: Zixi Shao, Tao Wang, Yibei Xiao, Tianyi Huang  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.01513v1  

#### Abstract
Designing therapeutic messenger RNA (mRNA) requires creating full-length transcripts that carefully balance stability, translation efficiency, and immune safety. To address this challenge, we propose ProMORNA, a multi-objective generation framework that produces complete mRNA transcripts \textit{de ...

---

### 22. [Mitigating Multimodal LLMs Hallucinations via Relevance Propagation at Inference Time](https://arxiv.org/abs/2605.01766)

**Authors**: Itai Allouche, Joseph Keshet  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.01766v1  

#### Abstract
Multimodal large language models (MLLMs) have revolutionized the landscape of AI, demonstrating impressive capabilities in tackling complex vision and audio-language tasks. However, a critical challenge remains: these models often suffer from hallucinations, generating outputs that diverge from the ...

---

### 23. [DBLP: Phase-Aware Bounded-Loss Transport for Burst-Resilient Distributed ML Training](https://arxiv.org/abs/2605.01989)

**Authors**: Zechen Ma, Zixi Qu, Jinyan Yi, David Lin, Yashar Ganjali  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.01989v1  

#### Abstract
Distributed machine learning (ML) training has become a necessity with the prevalence of billion to trillion-parameter-scale models. While prior work has improved training efficiency from the ML perspective at the application layer, it often fails to address transient congestion events at the networ...

---

### 24. [Bringing Order to Asynchronous SGD: Towards Optimality under Data-Dependent Delays with Momentum](https://arxiv.org/abs/2605.02043)

**Authors**: Tehila Dahan, Roie Reshef, Sharon Goldstein, Kfir Y. Levy  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.02043v1  

#### Abstract
Asynchronous stochastic gradient descent (SGD) enables scalable distributed training but suffers from gradient staleness. Existing mitigation strategies, such as delay-adaptive learning rates and staleness-aware filtering, typically attenuate or discard delayed gradients, introducing systematic bias...

---

### 25. [Projection-Free Transformers via Gaussian Kernel Attention](https://arxiv.org/abs/2605.02144)

**Authors**: Debarshi Kundu, Archisman Ghosh, Swaroop Ghosh, Vasant Honavar  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.02144v1  

#### Abstract
Self-attention in Transformers is typically implemented as $\mathrm{softmax}(QK^\top/\sqrt{d})V$, where $Q=XW_Q$, $K=XW_K$, and $V=XW_V$ are learned linear projections of the input $X$. We ask whether these learned projections are necessary, or whether they can be replaced by a simpler similarity-ba...

---

### 26. [Trust, but Verify: Peeling Low-Bit Transformer Networks for Training Monitoring](https://arxiv.org/abs/2605.02853)

**Authors**: Arian Eamaz, Farhang Yeganegi, Mojtaba Soltanalian  
**Category**: cs.LG  
**Published**: 2026-05-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.02853v1  

#### Abstract
Understanding whether deep neural networks are effectively optimized remains challenging, as training occurs in highly nonconvex landscapes and standard metrics provide limited visibility into layer-wise learning quality. This challenge is particularly acute for transformer-based language models, wh...

---

### 27. [Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression](https://arxiv.org/abs/2605.01402)

**Authors**: Yao Du, Shanshan Li, Xiaomeng Li  
**Category**: cs.CL  
**Published**: 2026-05-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.01402v1  

#### Abstract
Multimodal large language models (MLLMs) struggle with numerical regression under long-tailed target distributions. Token-level supervised fine-tuning (SFT) and point-wise regression rewards bias learning toward high-density regions, leading to regression-to-the-mean behavior and poor tail performan...

---

### 28. [FT-RAG: A Fine-grained Retrieval-Augmented Generation Framework for Complex Table Reasoning](https://arxiv.org/abs/2605.01495)

**Authors**: Zebin Guo, Weidong Geng, Ruichen Mao  
**Category**: cs.CL  
**Published**: 2026-05-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.01495v1  

#### Abstract
Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding responses in external knowledge during inference. However, conventiona RAG systems under-perform on structured tabular data, largely due to coarse retrieval granularity and insufficient table semantic comprehensi...

---

### 29. [Only Say What You Know: Calibration-Aware Generation for Long-Form Factuality](https://arxiv.org/abs/2605.01749)

**Authors**: Wen Luo, Guangyue Peng, Liang Wang, Nan Yang, Wei Li, Yuhan Song, Shaohang Wei, Feifan Song, Furu Wei, Houfeng Wang  
**Category**: cs.CL  
**Published**: 2026-05-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.01749v1  

#### Abstract
Large Reasoning Models achieve strong performance on complex tasks but remain prone to hallucinations, particularly in long-form generation where errors compound across reasoning steps. Existing approaches to improving factuality, including abstention and factuality-driven optimization, follow a \em...

---

### 30. [Do Large Language Models Plan Answer Positions? Position Bias in Multiple-Choice Question Generation](https://arxiv.org/abs/2605.01846)

**Authors**: Xuemei Tang, Xufeng Duan, Zhenguang G. Cai  
**Category**: cs.CL  
**Published**: 2026-05-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.01846v1  

#### Abstract
Large language models (LLMs) are increasingly used to generate multiple-choice questions (MCQs), where correct answers should ideally be uniformly distributed across options. However, we observe that LLMs exhibit systematic position biases during generation. Through extensive experiments with 10 LLM...

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
