# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-22 07:19:56 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training](https://arxiv.org/abs/2604.19241)

**Authors**: Size Zheng, Xuegui Zheng, Li-wen Chang, Jidong Zhai  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2604.19241v1  

#### Abstract
The exponential growth in Large Language Model (LLM) parameters has transformed model training into an increasingly resource-intensive endeavor. With the stagnation of Moore's Law and the widening disparity between computation throughput and communication bandwidth, expert parallelism (EP) has emerg...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大规模 **Mixture-of-Experts (MoE)** 模型训练中，**Expert Parallelism (EP)** 已成为主流并行策略，但其面临两大挑战：
- **通信瓶颈**：MoE 层中的 All-to-All 或 AllGather 通信开销巨大，尤其在 GPU 计算吞吐增长远超互联带宽（如 NVLink）的背景下。
- **数值不稳定性**：现有优化方法（如计算-通信重叠）常通过微批拆分（micro-batch splitting）实现，但由于浮点数非结合性（non-associativity），导致梯度累积顺序改变，破坏训练的**比特级可复现性（bitwise reproducibility）**。

此外，当前系统缺乏统一抽象，开发者需手动维护多种通信原语（AllGather vs AllToAll）和复杂调优逻辑。

---

### 🚀 提出的新方法与创新点

**UniEP** 是一个面向 MoE 训练的统一系统，其核心是将 MoE 的通信与计算融合为 **MegaKernel**，并在单个 CUDA Stream 中实现细粒度重叠。

#### 主要创新如下：

1. **MegaKernel 架构用于 MoE 训练（首次应用）**
   - 将 `Dispatch + GroupGEMM` 和 `GroupGEMM + Combine` 子图融合为两个 MegaKernel。
   - 利用 GPU 的 Streaming Multiprocessors (SMs) 进行动态角色分配（Comm-Worker / Comp-Worker / Relay-Worker），实现**块级调度**，避免 CPU 干预。

2. **确定性 Token 排序机制（Deterministic Token Ordering）**
   - 提出全局 Token 映射算法（Algorithm 1），确保无论并行执行顺序如何，Token 在目标专家缓冲区中的写入顺序始终一致。
   - 保证了反向传播中 Transposed GroupGEMM 的梯度累积顺序与串行执行完全相同，从而**严格保持数值一致性**。

3. **统一的通信抽象与参数化搜索空间**
   - 抽象出统一的优化配置空间（如 SM 分配、warp 数量、tile 大小等），无需显式切换 AllGather / AllToAll 内核。
   - 提供基于硬件性能模型的自动调优器（AutoTuner），自动选择最优配置。

4. **带宽优化：Relay Worker 实现 intra-rank 多播**
   - 当多个专家位于同一 GPU 上时，仅传输一次 Token，由 Relay Worker 在本地 HBM 中复制，减少 NVLink 流量。
   - 理论分析表明，在 Top-8 路由下平均只需发送到 5.25 个不同 rank，节省约 34% 通信量。

---

### 🔍 相比现有方法的优势

| 维度 | 现有方法（如 COMET、DeepEP） | UniEP |
|------|-------------------------------|-------|
| **通信-计算重叠粒度** | 双流或多流粗粒度重叠（multi-stream） | 单流内细粒度重叠（fine-grained, SM-level） |
| **数值一致性** | 不保证（因微批拆分导致梯度差异） | 严格保证比特级等价 |
| **通信效率** | 无 intra-rank 多播优化 | 通过 Relay Worker 减少冗余传输 |
| **系统复杂性** | 需维护多套 kernel 与 heuristic | 统一抽象 + 自动调优，降低开发负担 |
| **CPU 开销** | 依赖 host 同步管理 stream | 完全设备端调度，消除 CPU 干扰 |

---

## 2. 核心实验方法和设置

### 📊 数据集与工作负载
未使用传统 NLP 数据集，而是基于真实生产级 MoE 模型进行端到端评估，涵盖以下系列：
- **DeepSeek-MoE** 系列（64–512 专家）
- **Qwen3** 系列（从 30B 到 480B 参数）
- **Kimi** 系列（如 Kimi-K2）

共测试 **12 种 MoE 配置**（见 Table 4），覆盖广泛隐藏维度（Hdim）、中间维度（Hinter）、专家数（Nexp）和 Top-k 设置。

---

### ⚙️ 实验设置

| 项目 | 描述 |
|------|------|
| **硬件平台** | 两种 NVIDIA Hopper GPU 集群：<br>- **Cluster 1**: 8 GPUs/node, 200 GB/s/direction NVLink（带宽受限）<br>- **Cluster 2**: 8 GPUs/node, 400 GB/s/direction NVLink（高带宽） |
| **序列长度** | 8k, 32k, 128k（长上下文场景） |
| **评估层级** | - Kernel-level（Dispatch+GEMM, GEMM+Combine）<br>- Layer-level（前向+反向）<br>- End-to-end（128 GPU 训练吞吐） |
| **主要指标** | - Latency (ms)<br>- Speedup<br>- Throughput (tokens/day)<br>- Numerical precision（max_diff, %non-bw） |

---

### 🆚 基线方法对比

1. **Serial Baseline**  
   - 使用 DeepEP（NVSHMEM 加速通信） + TransformerEngine（优化计算）
   - 无任何重叠，作为功能正确性的参考基准。

2. **COMET**  
   - 当前最先进的重叠方案，采用双流架构（CUDA Streams）实现 AllGather 与 GroupGEMM 重叠。
   - 使用 DMA 引擎进行通信，CUTLASS 实现计算。
   - 代表工业界 SOTA 水平。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ Kernel-Level 性能（Figure 3）
- 在 **Cluster 1（低带宽）** 上：
  - `Dispatch+GroupGEMM`：相比 Serial 提升 **11.20×–18.40×**，相比 COMET 提升 **1.30×–1.32×**
  - `GroupGEMM+Combine`：相比 Serial 提升 **8.23×–11.56×**，相比 COMET 提升 **1.31×–1.87×**
- 在 **Cluster 2（高带宽）** 上增益略低但仍显著，说明 UniEP 在通信受限环境下优势更明显。

#### ✅ Layer-Level 性能（Figure 4）
- **Cluster 1, 8k 序列长度**：
  - 前向：比 COMET 快 **1.08×**
  - 反向：比 COMET 快 **1.03×**
- **Cluster 1, 32k 序列长度**：
  - 前向提速扩大至 **1.22×**，验证了 Relay Worker 的带宽节省随 token 数增加而增强。

#### ✅ 长上下文性能（128k，Table 8）
- **Cluster 1**：前向比 COMET 快 **1.28×**
- **Cluster 2**：前向比 COMET 快 **1.33×**
- 即使在极端长序列下仍保持显著优势。

#### ✅ 端到端训练吞吐（Section 6.7）
- 在 **128 GPU 集群上训练 512k 序列长度任务**：
  - 吞吐从 **127B tokens/day → 138B tokens/day**
  - 实现 **1.09× 端到端加速**
  - **且全程保持比特级可复现性**

---

### 🔢 数值精度对比（Table 6）

| 方法 | 最大绝对误差 (max_diff) | 非比特相等元素占比 (%non-bw) |
|------|--------------------------|------------------------------|
| **UniEP (Ours)** | **0** | **0%** |
| **COMET** | 高达 0.25 | 21.69% – 29.31% |

✅ **结论**：UniEP 在所有配置下均实现与串行基线完全一致的结果；而 COMET 存在显著数值偏差。

---

### 🔍 消融实验（Ablation Study, Table 9）

评估三个逐步增强版本：

| 版本 | 内容 | 相对于 COMET 的平均加速 |
|------|------|------------------------|
| **O** | 基础 MegaKernel + 动态调度 | — |
| **B** | + Relay Worker 带宽优化 | **1.06×–1.36×** |
| **A (Full UniEP)** | + AutoTuner + 优先级调度 | **1.24×–1.73×** |

- **Relay Worker** 对高 Top-k 场景收益更大（如 MoE-6, MoE-9）。
- **AutoTuner** 贡献额外 **1.15×–1.68×** 加速，证明其有效探索 ~10⁵ 规模的配置空间。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **细粒度设备端调度优于多流主机控制**
   - MegaKernel 在单流内完成通信与计算的动态协调，彻底消除 CPU 同步开销和 stream bubbles。

2. **数值一致性可以与高性能共存**
   - 通过确定性 Token 映射和禁止微批拆分，UniEP 在不牺牲精度的前提下实现了最佳性能。

3. **通信模式可统一抽象**
   - AllGather 与 AllToAll 的选择可通过参数化配置自动处理，无需硬编码分支。

4. **Relay Worker 显著降低物理通信量**
   - 利用 intra-rank 多播，理论节省高达 34%，实际性能提升可达 1.36×。

5. **自动调优至关重要**
   - 手动调参难以应对多样化的 MoE 架构，Analytical Performance Model 可高效定位最优配置（误差仅 3.8%）。

---

### ⚠️ 局限性

1. **对 Triton-distributed 框架依赖较强**
   - 当前实现基于 Triton-distributed，虽便于开发（代码量减少 5–10×），但在最新 GPU 上可能略逊于手工 CUDA 优化内核。
   - 作者指出可通过移植至 CUDA 克服此问题（已有先例如 DeepGEMM）。

2. **暂未支持异构集群或跨数据中心扩展**
   - 当前聚焦同构单节点内优化，未来可结合 HeterMoE 或 HybridEP 扩展。

3. **MegaKernel 编程模型学习成本较高**
   - 尽管 Triton 降低了门槛，但仍需要理解 warp-level 控制、device-side signaling 等底层机制。

---

### 🔮 未来工作方向

1. **扩展至其他稀疏架构**
   - 如 Dynamic Sparsity、Conditional Computation 等同样涉及动态路由的场景。

2. **支持更多通信原语融合**
   - 将 FSDP、TP 等其他并行范式的通信也纳入 MegaKernel 统一调度。

3. **编译器级集成**
   - 将 UniEP 的调度思想融入 LLM 编译器栈（如 TorchDynamo + Inductor），实现全自动优化。

4. **迁移至 AMD 或国产 GPU 平台**
   - 利用 Triton 的可移植性，推动跨厂商高性能 MoE 支持。

---

> **总结一句话**：  
> **UniEP 通过统一的 MegaKernel 架构，在不牺牲数值精度的前提下，实现了 MoE 训练中通信与计算的极致重叠，是迈向高效、稳定、可复现的大规模 MoE 训练的重要一步。**

</details>

---

### 2. [FEPLB: Exploiting Copy Engines for Nearly Free MoE Load Balancing in Distributed Training](https://arxiv.org/abs/2604.19654)

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
在 **Mixture-of-Experts (MoE)** 模型的分布式训练中，由于路由机制（router）动态分配 tokens 到不同专家（experts），导致每个设备上的计算负载不均衡（load imbalance）。这种不平衡会导致严重的 **straggler 问题**——即部分 GPU 成为瓶颈，造成大量 GPU 时间浪费（论文指出平均浪费达 **18.6%**）。

传统解决方案如辅助平衡损失（auxiliary balancing loss）会限制模型表达能力，而动态调度方案（如 FasterMoE、Tutel）则引入额外通信开销，难以隐藏，尤其在现代高性能通信后端（如 DeepEP）下反而增加延迟。

---

### 🚀 提出的新方法与创新思路
FEPLB 提出了一种 **资源正交的新型并行维度** —— **动态负载均衡并行（Dynamic Parallelism）**，其核心思想是：

- 充分利用 **NVIDIA Hopper 架构中的 NVLink Copy Engine（CE）** 和 **CPU 资源**，这些资源在当前 EP（Expert Parallelism）和 PP（Pipeline Parallelism）中处于闲置状态。
- 设计 **Two-Phase Dispatch（两阶段调度）**：
  1. **Phase 1（跨节点调度）**：通过标准 EP 后端（如 DeepEP）将静态专家 tokens 正常路由，同时将动态专家 tokens 收集到本地 NVLink 域内。
  2. **Phase 2（节点内重平衡）**：利用 **Copy Engine 在 NVLink 上以 ~900 GB/s 的速度进行 SM-free 的 token 和 expert weight 重分布**，完全不占用 GPU SM 资源。
- 引入 **轻量级 CPU 调度器**，在静态专家计算期间并发运行，决定哪些动态专家需要迁移。

> 🔑 **设计原则：正交性（Orthogonality）**
>
> FEPLB 不干扰现有的 EP/PP 并行策略，也不改变通信模式或消耗 SM 资源，因此可无缝集成进现有系统。

---

### ⭐ 相比现有方法的优势
| 方法 | 主要缺点 | FEPLB 的优势 |
|------|--------|-------------|
| **FasterMoE** | 使用 shadow expert 和 pipelining，破坏 Grouped GEMM 性能；预测式调度在高 EP 下失效 | 反应式调度更鲁棒；保持完整 GEMM 批大小；无 SM 开销 |
| **Triton Distributed** | 通信-计算融合占用 SM，降低计算效率 | SM-free 通信，真正实现“零成本”重平衡 |
| **Tutel / SmartMoE** | 需要重新配置并行策略，引入额外通信 | 无需改动 EP/PP，兼容性强 |
| **DeepEP/FUSCO 类库** | 不支持分阶段通信，破坏 pipelining 假设 | FEPLB 在其之上运行，不修改底层协议 |

---

## 2. 核心实验方法和设置

### 📊 数据集与模型
- 使用 **GLM-5 的 MoE 层简化版本**（18 层，原为 78 层），保留原始 MoE 结构：
  - **128 个 routed experts**
  - **Top-k routing**
  - **无辅助平衡损失（no auxiliary loss）**
- 因 FEPLB 在每层独立操作，减少层数不影响 per-layer 分析有效性。

---

### ⚙️ 实验设置
- **硬件平台**：
  - **NVIDIA H100 SXM5 GPUs**（80GB HBM3）
  - 节点内：**NVLink 4.0**（900 GB/s 双向）
  - 节点间：**400 Gbps InfiniBand**
- **软件框架**：
  - 基于 **Megatron-LM + DeepEP** 实现
  - 使用 **Transformer Engine** 进行混合精度训练
  - **cuBLAS 多流 Grouped GEMM**

### 🧪 评估指标
| 指标 | 定义 | 意义 |
|------|------|------|
| **Token Straggler** | $\max_d(T_d) - \bar{T}$，其中 $T_d$ 是每 GPU token 数 | 衡量 token 分配的不均衡程度 |
| **GEMM Straggler** | $\max_d(G_d) - \bar{G}$，其中 $G_d$ 是每 GPU 的 GEMM 执行时间 | 衡量实际计算等待时间浪费 |
| **EP Communication Overhead** | Dispatch 和 Combine 阶段的时间变化 | 验证是否影响主干通信性能 |

### 🔁 对比基线方法
1. **Before LB**：标准 EP，无负载均衡
2. **FasterMoE**（pipe=1 和 pipe=2）：带 shadow expert 的复制机制
3. **Triton Distributed**：TP 并行下的通信-计算融合
4. **Tutel**：自适应切换 EP/DP 模式
5. **FEPLB（本文方法）**

测试三种 PP/EP 配置：
- PP=4, EP=2 → 8 GPUs
- PP=4, EP=4 → 16 GPUs
- PP=2, EP=8 → 16 GPUs

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ▶ 表 2：每 MoE 层执行时间（ms）
| PP/EP | Before LB | FasterMoE | Triton Dist. | Tutel | **FEPLB** |
|-------|-----------|------------|----------------|--------|----------|
| 4/2   | 8.2 / 14.9 | 7.9 / 14.0 | 13.1 / 22.8 | 8.0 / 17.1 | **7.9 / 14.4** |
| 4/4   | 7.3 / 13.2 | 6.9 / 12.2 | 15.3 / 24.0 | 7.2 / 15.2 | **6.8 / 12.1** |
| 2/8   | 6.9 / 12.5 | 6.3 / 11.1 | 22.8 / 30.0 | 6.8 / 14.5 | **6.0 / 10.6** |

✅ FEPLB 在所有配置下均达到最低或接近最低延迟，尤其在 **EP 较高时优势显著**。

---

#### ▶ 图 5 & 表 3/4：负载均衡质量提升

##### Token Straggler Reduction（表 3）
| PP/EP | Before LB | FasterMoE | **FEPLB** |
|-------|-----------|------------|----------|
| 4/2   | 2,278     | 1,014 (-55%) | **1,107 (-51%)** |
| 4/4   | 4,649     | 2,471 (-47%) | **1,697 (-63%)** |
| 2/8   | 6,666     | 4,036 (-39%) | **2,021 (-70%)** |

> ✅ **最高实现 70% 的 token straggler 降低**

##### GEMM Straggler Reduction（表 4）
| PP/EP | Before LB | FasterMoE | **FEPLB** |
|-------|-----------|------------|----------|
| 4/2   | 0.316     | 0.170 (-46%) | **0.157 (-50%)** |
| 4/4   | 0.652     | 0.380 (-42%) | **0.247 (-62%)** |
| 2/8   | 1.110     | 0.625 (-44%) | **0.352 (-68%)** |

> ✅ **最高实现 68% 的 GEMM straggler 降低**

---

#### ▶ 与 FasterMoE 的对比（EP=8 时）
- **Token straggler**：FEPLB 比 FasterMoE **低 2 倍**（2,021 vs. 4,036）
- **GEMM straggler**：FEPLB 比 FasterMoE **低 1.8 倍**
- **通信开销**：FasterMoE (pipe=2) 引入高达 **46.8% 的 dispatch 开销**，而 FEPLB <1%

> 💥 **FEPLB 的优势随 EP 增加而扩大**，因为高 EP 下路由更稀疏、不可预测，FasterMoE 的预测机制失效，而 FEPLB 的反应式调度更具鲁棒性。

---

#### 🔍 消融实验：动态专家数量（`dyn`）的影响（图 6）
- 控制参数 `dyn`：每个设备允许迁移的动态专家数（如 dyn=4 表示 4 个可迁移）
- 实验结果：
  - `dyn=2` 已能显著改善负载均衡（因通常只有少数热点专家）
  - `dyn=4` 达到性价比最优
  - `dyn=8` 提升有限，呈现**收益递减**

> ✅ 推荐默认设置：**dyn = 4**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Copy Engine 是实现“近零成本”负载均衡的关键硬件资源**：
   - NVLink CE 提供高达 900 GB/s 的 SM-free 通信带宽，完美适配节点内重平衡需求。
2. **动态负载均衡可以作为新的正交并行维度（Orthogonal Parallel Dimension）**：
   - 利用 CPU + CE + 静态计算窗口，完全避开 EP/PP 资源竞争。
3. **Two-Phase Dispatch 实现了精确语义保持的重平衡**：
   - 仅在节点内移动 tokens 和 weights，不改变最终输出，支持 **完全去除 auxiliary loss**。
4. **FEPLB 与现代 MoE 通信库（如 DeepEP）兼容性极佳**：
   - 不依赖 staged communication，避免了 FasterMoE 等方法在高效 backend 上的性能退化。

---

### ⚠️ 方法的局限性
1. **受限于当前 NVLink 拓扑**：
   - 当前仅能在单个 NVLink 域（如同一节点）内进行重平衡；跨节点需依赖未来架构（如 GB200 NVL72 的全连接 NVLink）。
2. **整专家迁移（whole-expert migration）限制了低 EP 下的粒度**：
   - 无法对 token batch 进行拆分，否则会破坏 Grouped GEMM 的 roofline 性能。
3. **内存开销虽小但存在**：
   - 每设备需预留 buffer 存放迁移来的 expert weights（例如 max_num_dyn=8 → 576 MiB），约占 HBM 的 <0.7%。

---

### 🔮 未来工作方向
1. **扩展至跨节点 Copy Engine 支持的 SuperPod 架构**（如 GB200）：
   - 实现全局范围内的 free load balancing。
2. **结合预测机制优化调度决策延迟**：
   - 在 CPU 调度器中引入轻量预测模型，进一步提升反应速度。
3. **探索 token-level partial migration 的可行性**：
   - 若能设计出对小 batch GEMM 更友好的 kernel，可能突破整专家迁移限制。
4. **集成到自动并行框架（Auto-parallelization）中**：
   - 将 FEPLB 作为一种 runtime 动态优化选项，纳入 Alpa、Galvatron 等系统。

---

## ✅ 总结
FEPLB 提出了一种 **基于硬件空闲资源（NVLink Copy Engine + CPU）的全新动态并行范式**，实现了 **几乎零代价的 MoE 负载均衡**。它不仅大幅降低了 token 和 GEMM straggler（**51–70%**），而且 **完全兼容现有 EP/PP 并行体系和高性能通信库**，代表了 MoE 分布式训练优化的一个重要方向：**从“掩盖开销”转向“消除开销”**。

</details>

---

### 3. [Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs](https://arxiv.org/abs/2604.18788)

**Authors**: Afsara Benazir, Felix Xiaozhu Lin  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.18788v1  

#### Abstract
Apple Neural Engine (ANE) is a dedicated neural processing unit (NPU) present in every Apple Silicon chip. Mixture-of-Experts (MoE) LLMs improve inference efficiency via sparse activation but are challenging for NPUs in three ways: expert routing is unpredictable and introduces dynamic tensor shapes...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Mixture-of-Experts (MoE) LLMs 虽然通过稀疏激活提升了推理效率，但在 **Apple Neural Engine (ANE)** 等移动 NPU 上部署面临三大挑战：
1. **动态路由导致动态张量形状**：专家选择不可预测，违反 NPU 对静态图和固定形状的要求。
2. **不支持不规则算子**：如 `top-k`、`scatter/gather`、动态索引等操作无法在 ANE 上高效执行。
3. **频繁小核调度开销大**：每个专家单独调度会引发大量 CPU-NPU 同步和启动开销，降低利用率。

目标是实现 **MoE 推理中尽可能多的计算卸载到 NPU**，尤其是在 **prefill 阶段**（占端到端延迟 80%），同时保持系统响应性和能效。

---

### **提出的新方法：NPUMoE**
设计了一个运行时推理引擎 **NPUMoE**，结合离线校准与运行时优化，在 NPU 和 CPU/GPU 之间智能划分计算任务。

#### **三大核心技术**
| 技术 | 核心思想 | 解决的问题 |
|------|--------|----------|
| **Static Tiers for Expert Capacity (S4.1)** | 将专家容量划分为若干静态层级（如高/中/低），热门专家分配更大容量，冷门专家更小容量 | 应对动态路由带来的形状变化，适配 NPU 静态图约束 |
| **Grouped Expert Execution (S4.2)** | 将多个专家合并为一个密集的 FFN 计算图，单次调用执行多个专家 | 减少小图调度次数，提升 NPU 并发利用率 |
| **Load-Aware Expert Compute Graph Residency (S4.3)** | 根据专家热度决定是否驻留于 NPU：热专家组留在 NPU，冷专家回退至 CPU | 避免小负载因同步开销反而拖慢整体性能 |

> ✅ **总体策略**：  
> - **NPU 执行**：Attention、FFN 等密集且可静态化的操作  
> - **CPU/GPU 回退路径**：Top-k、路由、gather/scatter 等动态控制流操作  

---

### **相比现有方法的优势**
- **首次系统性地将 MoE 推理适配到移动 NPU**（尤其是 Apple ANE）
- 不依赖全模型静态化或牺牲精度换取速度
- 在真实长上下文场景下显著优于纯 CPU 或 naïve NPU 卸载方案
- 实现了 **高能效、低延迟、低 CPU 占用** 的三重优化

---

## **2. 核心实验方法和设置**

### **使用的模型**
| Model | #Layers | Total Params | Active Params | #Experts | Sparsity |
|-------|--------|--------------|----------------|----------|----------|
| **Phi-3.5-MoE-Instruct** | 32 | 42.6B | 6.6B | 8 | 2/8 |
| **Phi-tiny-MoE-Instruct** | 32 | 3.8B | 1.1B | 8 | 2/8 |
| **Qwen3-30B-A3B** | 48 | 30.5B | 3.3B | 128 | 8/128 |

> 所有模型均采用 FP16 精度，部分使用 INT8 量化后反量化为 FP16 运行。

---

### **硬件平台**
- **M2 Max**：64GB RAM, 12-core CPU, 16-core ANE
- **M2 Ultra**：192GB RAM, 24-core CPU, 32-core ANE  
> 利用统一内存架构（Unified Memory）使整个 MoE 模型可驻留设备内存

---

### **数据集与工作负载**
针对 **长上下文理解任务**，主要包括：
- **HellaSwag**：常识推理，多选补全
- **BoolQ**：是非问答
- **RULER**：长上下文检索与聚合任务

> 使用标准 few-shot prompting 方式生成输入，重点关注 **prefill 阶段性能**

---

### **评估指标**
| 指标 | 定义 |
|------|------|
| **TTFT (Time-To-First-Token)** | Prefill 延迟，反映首 token 生成时间 |
| **Energy per Token (EPT)** | 总能耗 / 生成 token 数，衡量能效 |
| **CPU Cycles Consumption** | CPU 循环数，作为 NPU 卸载程度的代理指标 |
| **Accuracy Degradation** | 相比 FP16 全精度模型的准确率下降 |

---

### **基线方法对比**
| Baseline | Attention | Expert FFN | 特点 |
|---------|-----------|------------|------|
| **Core ML (CPU only)** | CPU | CPU | 默认 CPU 执行 |
| **Core ML (naive)** | CPU | NPU（自动调度） | Core ML 自动决策是否卸载专家 |
| **ANEMLL** | NPU | CPU | 加速 dense transformer，但未优化 MoE |
| **Ours (NPUMoE)** | NPU | NPU（经优化分组） | 本文方法，全部启用三项技术 |

---

## **3. 主要实验结果和性能指标**

### **关键性能提升（M2 Ultra 上 Phi-3.5-MoE 结果）**

| 指标 | 提升幅度（vs. Baselines） |
|------|--------------------------|
| **Prefill Latency (TTFT)** | **1.32x – 5.55x 更快** |
| **Energy Efficiency (EPT)** | **1.81x – 7.37x 更优** |
| **CPU Cycle Usage** | **1.78x – 5.54x 更少** |
| **End-to-End Speedup** | 最高达 **3.86x**（vs. CoreML naive） |

> 🔋 例如，在 `(C=512, P=1024)` 设置下，CoreML(naive) 能耗是 NPUMoE 的 **7.37x**

---

### **详细性能分解（Table 4 & 5）**
- **MoE Block 中 Expert FFN 占比 >86% 延迟** → 优化专家执行至关重要
- NPUMoE 将 Expert FFN 时间从 3369.86ms（naive）降至 **530.91ms**
- Scatter 写回时间从 896.76ms 降至 **775.52ms**
- 综合 MoE block 延迟下降约 **5–10%**，系统级收益来自并行性和调度优化

---

### **消融实验结果（Ablation Study）**
逐步添加三项技术的效果如下：

| 方法 | 相比 Ours-base 的提速 | 能效优势 |
|------|------------------------|----------|
| **Ours-base**（仅算子划分） | 1.00x | 1.00x |
| **+ Static Tiers (Ours-T)** | 1.48x | — |
| **+ Grouping (Ours-TG)** | 2.11x | **节能 3.74x** |
| **+ Load-aware Residency (Ours-all)** | **2.89x** | 仍保持高能效 |

> ✅ **Grouped Execution 是最大贡献者**，大幅摊销调度开销  
> ✅ **Load-aware Residency 进一步避免冷专家浪费资源**

---

### **准确性表现（Table 3）**
| 数据集 | 平均 Token Drop Rate | 准确率下降 |
|-------|------------------------|------------|
| BoolQ | 14.66% | 0.0% |
| Hellaswag | 10.63% | -1.07% |
| RULER (multi-key) | 21.00% | 0.0% |

> ❗ 尽管有 **~10–20% token 被丢弃**（基于 saliency score 剪枝），但准确率损失极小（<1.1%），说明方法鲁棒性强。

---

## **4. 关键结论和发现**

### **主要发现**
1. **MoE 动态性虽与 NPU 静态性冲突，但可通过系统级协同设计解决**
2. **静态容量分级 + 分组执行 + 负载感知驻留** 可有效桥接动态 MoE 与静态 NPU
3. **NPU 卸载不仅能加速，更能释放 CPU/GPU 资源用于交互任务**
4. **即使有 token 剪枝，模型鲁棒性依然良好**，尤其在 long-context retrieval 类任务中

---

### **局限性**
1. **主要针对 prefill 阶段优化**，decode 阶段单 token 生成难以摊销 NPU 开销
2. **当前原型 decode 复用 prefill 流程**，未专门优化自回归生成
3. **依赖离线校准**，若 workload 分布剧烈偏移可能影响容量估计准确性
4. **不支持跨设备动态张量切分**，受限于 Core ML 编译时绑定机制

---

### **未来工作方向**
- 优化 **decode 阶段的 NPU 利用率**，探索 speculative decoding 与 NPU 协同
- 引入 **在线反馈机制** 动态调整容量 tier 与分组策略
- 扩展至其他厂商 NPU（如 Qualcomm Hexagon, AMD NPU）
- 支持 **多模态 MoE 模型** 的异构推理调度

---

> 📌 **一句话总结**：  
> **NPUMoE 成功将动态 MoE 推理适配到 Apple Silicon ANE，通过“静态化+分组+负载感知”三重设计，在真实设备上实现了最高 5.55x 加速、7.37x 能效提升，且几乎无精度损失，为移动端稀疏大模型落地提供了新范式。**

</details>

---

### 4. [Are Large Language Models Economically Viable for Industry Deployment?](https://arxiv.org/abs/2604.19342)

**Authors**: Abdullah Mohammad, Sushant Kumar Ray, Pushkar Arora, Rafiq Ali, Ebad Shabbir, Gautam Siddharth Kashyap, Jiechao Gao, Usman Naseem  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.19342v1  

#### Abstract
Generative AI-powered by Large Language Models (LLMs)-is increasingly deployed in industry across healthcare decision support, financial analytics, enterprise retrieval, and conversational automation, where reliability, efficiency, and cost control are critical. In such settings, models must satisfy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Are Large Language Models Economically Viable for Industry Deployment?》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **Large Language Models (LLMs)** 评估体系严重依赖以准确性为核心的基准测试（accuracy-centric benchmarks），忽视了工业部署中至关重要的**经济性、能效、延迟、硬件利用率和生命周期成本**等操作维度。这种偏差导致了“**Deployment-Evaluation Gap**”——即模型在实验室表现优异，但在真实产业环境中可能不具备经济可行性。

该论文指出，仅关注 accuracy 无法反映实际部署中的关键权衡，例如：
- 小模型是否能在 ROI（投资回报）上超越大模型？
- 内存压缩技术（如 QLoRA）是否真的节能？

### 提出的新方法与新思路
作者提出了 **EDGE-EVAL** ——一个面向工业部署的全生命周期 LLM 评估框架，首次系统性地将以下五个**部署导向的度量指标**纳入统一评估体系：

| 指标 | 全称 | 含义 |
|------|------|------|
| **Nbreak** | Economic Break-Even | 达到本地部署成本低于 API 调用所需的请求数量，衡量 ROI 速度 |
| **IPW** | Intelligence-Per-Watt | 单位能耗下的任务性能输出，衡量绿色 AI 效率 |
| **psys** | System Density | 每 GB VRAM 支持的吞吐量（tokens/s/GB），反映硬件利用密度 |
| **Ctax** | Cold-Start Tax | 模型加载能耗相对于推理能耗的比例，影响 serverless 部署可行性 |
| **Qret** | Quantization Fidelity | 4-bit 量化后任务性能保留率，衡量压缩安全性 |

> ✅ **创新性**：从“accuracy-only”转向“lifecycle-aware”，覆盖 adaptation → compression → inference 完整流程。

### 相比现有方法的优势
| 维度 | 传统方法 | EDGE-EVAL |
|------|--------|-----------|
| 评估目标 | 准确率最大化 | 经济可持续性 + 生态效率 |
| 硬件假设 | 最新高端 GPU | **Legacy hardware (Tesla T4)**，更贴近中小企业现实 |
| 成本考量 | 忽略训练/加载能耗 | 显式建模 Ctrain, Capi, Cinfer, Eload |
| 量化分析 | 仅看精度损失 | 引入 Qret 和 energy/performance trade-off 分析 |
| 可扩展性 | 不考虑 psys | 明确评估系统密度对服务容量的影响 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在三个典型工业任务上进行验证：

| 任务 | 数据集 | 描述 |
|------|-------|------|
| **Summarization** | XSum | 新闻文章摘要生成，模拟长文本压缩场景 |
| **Retrieval-Augmented Generation (RAG)** | SQuAD v1.1 | 结合检索的知识问答，代表企业级知识推理 |
| **Conversational Agents** | UltraChat | 多轮对话数据，测试低延迟交互能力 |

所有任务采用 **70/15/15 划分**，微调样本控制在 5K–10K，保留完整验证/测试集用于评估。

### 实验设置
- **硬件平台**：双卡 **NVIDIA Tesla T4 (16GB VRAM each)**  
  > ⚠️ 注重 legacy 硬件，反映广泛存在的旧有基础设施
- **模型家族**：
  - **LLaMA-3.x**：1B, 3B, 8B 参数版本（Grouped-Query Attention）
  - **Qwen-2.5**：1.5B, 3B, 7B 参数版本（Dense Transformer）
- **适配策略 (Adaptation)**：
  - LoRA-FP16
  - LoRA-INT8
  - LoRA-INT4 (Post-Training Quantization, PTQ)
  - QLoRA-INT4 (Quantization-Aware Training)
- **推理引擎**：**vLLM (v0.6.3)** + paged attention，batch size = 1（模拟低并发工业场景）
- **测量工具**：`pynvml` 采集 GPU 功耗（100ms 间隔），实现细粒度 energy profiling

### 评估指标
除了传统的 task-specific metric：
- **RAG**: NLI Entailment + ROUGE-L
- **Summarization**: NLI Non-Contradiction + ROUGE-L
- **Conversation**: LLM-as-a-Judge (GPT-4o) on Helpfulness & Safety (1–10)

还引入本文提出的五大部署指标：
- **Nbreak**, **IPW**, **psys**, **Ctax**, **Qret**

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | Nbreak (Req) | IPW | psys (Tok/s/GB) | Qret (%) | Ctax (×) |
|-------|--------------|-----|------------------|----------|---------|
| **LLaMA-3.2-1B (INT4)** | **14** | 0.45 | **6,930** | 100.6% | 183× |
| LLaMA-3.2-3B | 33 | 0.27 | 1,336 | 99.8% | 184× |
| LLaMA-3.1-8B | 43 | 0.15 | 387 | 100.3% | 230× |
| **Qwen-2.5-1.5B** | 21 | **0.48** | **6,942** | 99.6% | 179× |

> 🔍 **结论**：<2B 模型在几乎所有部署维度上均优于更大模型。

### 与基线方法的对比结果

#### （1）ROI 与经济效益
- **LLaMA-1B** 仅需 **14 次请求**即可收回本地化部署成本，远快于 7B 模型的 43 次。
- 在低流量场景下，小模型具有显著的 **ROI velocity** 优势。

#### （2）能效表现（IPW）
- <2B 模型提供高达 **3× 更高的 Intelligence-Per-Watt**，意味着单位能源可完成更多有效推理任务。
- 尤其是 Qwen-1.5B 达到 **0.48 IPW**，为所有配置最高。

#### （3）系统密度（psys）
- 1B 模型在 INT4 下达到 **>6,900 tokens/s/GB**，相较 7B 模型提升达 **17×**。
- 这使得老旧 T4 显卡也能作为高密度推理节点使用。

#### （4）冷启动税（Ctax）
- 所有模型的 Ctax 均超过 **179×**，表明频繁 load/unload（如 serverless）极不经济。
- 大模型更高（最高达 237×），进一步削弱其弹性部署潜力。

### 消融实验结果

#### QLoRA 的“效率悖论”（见 Table 2）
尽管 QLoRA 显著降低 VRAM 占用（约 60%↓），但其训练能耗反而大幅上升：

| Model | 方法 | 中位训练能量 (kWh) | 相对 LoRA 增幅 |
|-------|------|--------------------|---------------|
| LLaMA-1B | LoRA-FP16 | 0.039 | 1.0× |
| LLaMA-1B | QLoRA-INT4 | 0.251 | **6.4×↑** |
| Qwen-7B | QLoRA-INT4 | 0.563 | 2.3×↑ |

> ❗ 发现：**Memory efficiency ≠ Energy efficiency**，尤其对小模型而言，QLoRA 导致高达 **7× 的适应能量增加**。

#### 量化推理收益（Table 3）
INT4 推理带来显著性能增益：

| Model | Precision | Throughput | Speedup | Energy/request | 节省 |
|-------|-----------|------------|---------|----------------|------|
| LLaMA-1B | FP16 → INT4 | 2.2k → 4.3k tok/s | **1.94×** | 6.45J → 2.50J | **61%↓** |

> ✅ 表明：**INT4 是有效的 hardware multiplier**，尤其适合资源受限环境。

#### 量化保真度（Qret）分析（Table 4 & 5）
- 多数 <2B 模型在 INT4 下保持 **>99% 性能保留率**，部分甚至略有提升（如 LLaMA RAG +0.4%）。
- Qwen 在对话任务中出现 **最大 -9.5% 的 helpfulness 下降**，显示架构相关敏感性。
- 输出方差增大（STD △ up to +65.9%），提示稳定性风险。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **<2B 模型构成“效率前沿”（Efficiency Frontier）**
   - 在 ROI、能效、系统密度、延迟等方面全面优于 3B/7B 模型
   - 特别是在 legacy hardware 上更具部署优势

2. ❌ **QLoRA 存在“适应能耗异常”（Adaptation Energy Anomaly）**
   - 尽管节省内存，但训练阶段能耗激增（最多 **7×**）
   - 挑战了“量化一定环保”的普遍认知，揭示 memory-energy trade-off

3. ✅ **INT4 推理是性价比极高的优化手段**
   - 提供近 **2× 吞吐加速** 和 **>57% 能耗下降**
   - 对多数任务几乎无损（Qret ≈ 100%）

4. ⚠️ **Cold-Start Tax 极高，Serverless 不现实**
   - 加载能耗是单次推理的百倍以上，不适合 scale-to-zero 架构

5. 💡 **EDGE-EVAL 揭示 Accuracy-Centric Benchmark 的盲区**
   - 高 accuracy ≠ 高部署价值
   - 必须结合 Nbreak, IPW, psys 等综合判断

### 局限性
- 实验集中在 **Tesla T4** 和 **低批量场景**，结果在 Hopper 架构或大规模云服务中可能不同
- 仅评估 **LLaMA 和 Qwen** 两个 family，未涵盖其他结构（如 Mamba、MoE）
- 能耗测量基于 GPU telemetry，未包含 CPU、网络、存储等系统级开销
- 经济参数（API 定价、碳强度因子）为当前估计值，随时间变化会影响绝对 break-even 点

### 未来工作方向
- 扩展 EDGE-EVAL 至更多硬件平台（如 Jetson、TPU Edge）
- 引入多模态 LMMs 的生命周期评估
- 构建动态定价模型，支持 real-time ROI monitoring
- 探索 cold-start 优化技术（如 persistent warm pools、weight streaming）
- 开发兼顾 memory 与 energy 效率的新一代 PEFT 方法

---

> 📌 **一句话总结**：  
> 在真实工业约束下，**小型语言模型（<2B）通过 INT4 量化部署，在经济性和生态效率上全面超越大型模型**；而当前流行的 QLoRA 技术虽省显存，却可能付出高昂的训练能耗代价——这正是 accuracy-centric benchmark 所无法捕捉的关键现实挑战。

</details>

---

### 5. [ReaLB: Real-Time Load Balancing for Multimodal MoE Inference](https://arxiv.org/abs/2604.19503)

**Authors**: Yingping Wang, Yi Wu, Xiangyu Wu, Junwei Cui, Weilin Cai, Zhijiang Guo, Jiayi Huang  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.19503v1  

#### Abstract
Mixture-of-Experts (MoE) architectures are widely used in modern large language models and multimodal models. However, inference efficiency is often limited by highly dynamic and skewed expert workloads across different modalities. During the prefill stage with large batch sizes, vision tokens frequ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ReaLB: Real-Time Load Balancing for Multimodal MoE Inference 论文总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代多模态大模型（如 MLLMs）广泛采用 **Mixture-of-Experts (MoE)** 架构以实现参数高效扩展。然而，在多模态推理过程中，由于不同模态输入（尤其是视觉 token 和文本 token）导致的专家负载高度动态且严重倾斜，造成 **Expert Parallelism (EP)** 下的设备级负载不均衡。这种不平衡在预填充阶段（prefill stage）尤为显著，部分 GPU 成为“straggler”，严重影响系统吞吐量。

传统基于历史预测的负载均衡方法（如 EPLB）难以应对多模态推理中快速变化的路由模式，且引入额外通信开销和内存占用。

---

### 提出了什么新方法或新思路
作者提出 **ReaLB**（Real-Time Load Balancing），一种面向多模态 MoE 推理的实时负载均衡方法，其核心思想是：

- **模态感知的精度自适应调度**（Modality-aware Precision Adaptation）：  
  观察到视觉 token 具有更高的冗余性，对计算精度更鲁棒；而文本 token 对精度敏感。因此，ReaLB 在运行时动态识别由“视觉密集型专家”主导的过载设备（hot device），并将其 MoE 计算切换至低精度（如 FP4）执行，利用硬件加速单元（如 FP4 Tensor Cores）提升执行效率。
  
- **零调度开销的流水线设计**（Overlapped Pipeline Orchestration）：  
  将精度转换操作（如 BF16 → FP4）与 All-to-All 通信阶段重叠，完全隐藏转换延迟，确保不增加关键路径时间。

- **无需冗余专家或额外内存**：  
  不复制专家权重，仅在线进行权重量化，并保留原始高精度副本，避免内存膨胀。

---

### 相比现有方法的优势
| 维度 | 传统方法（如 EPLB） | ReaLB |
|------|------------------------|--------|
| **响应速度** | 依赖历史窗口预测，滞后于实际负载 | 实时感知当前路由决策，即时调整 |
| **开销** | 需要专家迁移，带来通信与内存开销 | 无专家复制，通信开销为零 |
| **适用性** | 对动态路由适应差 | 特别适合高动态的多模态推理场景 |
| **精度控制** | 无区分 | 模态感知，保护文本 token 精度 |

> ✅ **核心优势**：**零额外调度开销 + 实时响应 + 高效利用硬件低精度能力**

---

## 2. 核心实验方法和设置

### 使用的数据集
使用 `lmms-eval` 工具包评估以下六个主流多模态基准：
- **RealWorldQA**：文本为主的问题回答
- **AI2D**：单图图表理解
- **InfoVQA**：信息图推理
- **TextVQA**：OCR 文本理解
- **MMMU**：多图跨图像推理（高难度）
- **MMBench**：综合多模态问答任务

这些数据集覆盖了从文本主导到视觉主导的不同模态组合，验证方法的泛化能力。

---

### 实验设置和评估指标

#### 测试平台
- **硬件**：8×NVIDIA RTX 5090 GPU（受限于资源，通过 H20 + NVLink 数据模拟高带宽通信环境）
- **软件栈**：基于 **vLLM v0.13.0** 实现，集成 LLM Compressor 和 FlashInfer 的 NVFP4 GEMM 内核
- **部署模式**：Expert Parallelism (EP=8)，结合 Data Parallelism

#### 模型
评估三种开源多模态 MoE 模型：
- **Kimi-VL-A3B-Instruct**
- **Qwen3-VL-30B-A3B-Instruct**
- **ERNIE-4.5-VL-27B-A3B**

其中 ERNIE-VL 为 **modality-isolated MMoE** 架构，验证 ReaLB 对不同架构的兼容性。

#### 输入配置
- 每请求：256 token 文本 prompt + 2 张 1024×728 图像
- 批处理大小：全局 token 数 > 2048 时启用 ReaLB（进入 compute-bound 状态）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **MoE-layer latency** | 单个 MoE 层前向传播耗时（不含通信） |
| **End-to-end throughput** | 整体推理吞吐（tokens/sec） |
| **Accuracy** | 多项 benchmark 平均得分（vs. BF16 baseline） |
| **Speedup** | 相对于 Baseline 的加速比 |

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Baseline** | 标准 W16A16 GEMM，无负载均衡 |
| **FP4-All** | 所有 MoE 层统一使用 W4A4 FP4 GEMM |
| **EPLB** | 历史窗口预测 + 专家复制（8 个冗余专家，窗口=100，间隔=100） |
| **Async EPLB** | 异步专家迁移版本 |
| **ReaLB-seq** | ReaLB 的串行版本（无流水线重叠） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ MoE 层级加速
- **平均 MoE 层加速达 1.29×**
- 在 Qwen-VL 上稳定达到 **1.29× speedup**
- 在 Kimi-VL 和 ERNIE-VL 上分别达到 **1.14×** 和 **1.26×**

> 图 10 显示 ReaLB 在所有迭代中保持低延迟，波动小，优于 EPLB 的不稳定表现。

#### ✅ 端到端吞吐提升
- **最高端到端吞吐提升达 1.53×**
- 三模型平均提升：**1.20× ~ 1.53×**
- ReaLB-seq 提升有限（未重叠开销），说明流水线设计至关重要

> 图 12 表明 ReaLB 显著优于 EPLB 和 Async EPLB，后者几乎无增益。

#### ✅ 负载均衡效果
- 图 11 显示 ReaLB 成功消除设备间最大延迟差异：
  - 在 Qwen-VL 中将最慢 rank 从 Rank7 转移至 Rank4
  - 多个 rank 同时被加速（如 ERNIE-VL 的 Rank 2,5,6,7）

---

### 与基线方法的对比结果

| 方法 | MoE Speedup | E2E Throughput | Accuracy Loss |
|------|-------------|----------------|---------------|
| **Baseline** | 1.00× | 1.00× | 0.0 pt |
| **FP4-All** | ~1.40× | ~1.39× | **↓ 2–8 pts**（尤其 MMMU/DynaMath） |
| **EPLB** | ≤0.92× | ≤0.96× | 0 pt（不影响精度） |
| **ReaLB** | **1.29×** | **1.53×** | **≤1.2 pt** |

> ⚠️ EPLB 反而降低性能：因频繁专家迁移开销超过收益。

---

### 消融实验结果（Ablation Study）

#### 不同模态阈值 $ M_d $ 的影响（表 2）
| $ M_d $ | E2E Speedup | Avg Acc Loss |
|---------|--------------|--------------|
| 0.0（关闭模态感知） | 1.29–1.30× | ↓2.0–2.4 pts |
| **0.7（推荐）** | **1.28–1.29×** | **↓0.0–1.1 pts** |
| 0.9 | 1.28–1.29× | ↓0.2–0.66 pts |

- 结论：适度提高 $ M_d $ 可更好保护文本密集型设备的精度，同时保持高性能。
- **$ M_d = 0.7 $ 是最佳平衡点**：兼顾效率与精度。

#### ReaLB-seq vs ReaLB
- ReaLB-seq 加速极弱，证明 **流水线重叠机制是性能增益的关键**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **多模态 MoE 推理中的负载不均衡具有强动态性和模态驱动特性**：视觉 token 主导导致负载剧烈波动，传统预测方法失效。
2. **低精度计算可用于主动“调节”执行时间**：通过对视觉密集型设备降精度来加速 straggler，是一种有效的负载均衡手段。
3. **实时性 + 零开销 是实用负载均衡的前提**：ReaLB 通过在线精度变换 + 流水线隐藏开销，实现了真正轻量高效的实时均衡。
4. **模态感知控制至关重要**：统一降精度（FP4-All）会严重损害精度；而选择性降精度可在几乎无损下获得显著加速。

---

### 方法的局限性
1. **依赖硬件支持低精度 GEMM**：目前基于 FP4 Tensor Cores，若硬件不支持则无法生效。
2. **阈值需手动调优**：当前 $ M_d $ 为固定值，虽有一定鲁棒性，但未实现自动适配。
3. **主要适用于 prefill 阶段**：decode 阶段负载较轻，收益有限。
4. **假设视觉 token 更鲁棒**：该假设在极端数值推理任务中可能不成立。

---

### 未来工作方向
1. **自适应阈值调节机制**：如采用 AIMD（Additive Increase Multiplicative Decrease）等策略动态调整 $ M_d $。
2. **扩展至更多模态**：应用于视频、音频等异构更强的多模态 MoE 系统。
3. **探索混合精度层级调度**：不止 FP4/BF16，而是多级精度分配以精细平衡延迟。
4. **跨节点部署优化**：在分布式集群中进一步优化通信与精度切换协同。
5. **与 PD disaggregation 架构深度整合**：专用于 prefill worker，最大化生产环境收益。

---

## 总结
ReaLB 提出了一种全新的视角：**将“混合精度计算”的延迟差异作为一种系统级调控工具**，而非单纯的压缩技术。它通过 **modality-aware + real-time + zero-overhead** 的设计，在不牺牲精度的前提下显著提升了多模态 MoE 推理的吞吐量，为大规模 MLLM 部署提供了高效实用的解决方案。

</details>

---

### 6. [DT2IT-MRM: Debiased Preference Construction and Iterative Training for Multimodal Reward Modeling](https://arxiv.org/abs/2604.19544)

**Authors**: Zhihong Zhang, Jie Zhao, Xiaojian Huang, Jin Xu, Zhuodong Luo, Xin Liu, Jiansheng Wei, Xuejin Chen  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.19544v1  

#### Abstract
Multimodal reward models (MRMs) play a crucial role in aligning Multimodal Large Language Models (MLLMs) with human preferences. Training a good MRM requires high-quality multimodal preference data. However, existing preference datasets face three key challenges: lack of granularity in preference st...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DT2IT-MRM: Debiased Preference Construction and Iterative Training for Multimodal Reward Modeling》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前的 **Multimodal Reward Models (MRMs)** 在训练过程中严重依赖高质量的多模态偏好数据（multimodal preference data），但现有的开源偏好数据集存在三大关键缺陷：

1. **文本风格偏见（Textual Style Bias）**：  
   偏好数据中的“被选中”响应通常来自高性能模型，导致 MRM 学习到的是对特定文本风格的偏好，而非内容质量本身。

2. **偏好强度缺乏多样性（Lack of Diversity in Preference Strength）**：  
   多数数据集中偏好信号过于“强”，即“好 vs 差”的区分明显，缺乏弱偏好（如“略好 vs 略差”）样本，导致模型过拟合且泛化能力差。

3. **偏好信号不可靠（Unreliable Preference Signals）**：  
   许多方法使用闭源模型（如 GPT-4o）进行自动标注，但若这些模型自身无法正确回答问题，则其生成的偏好标签是错误的。此外，还存在**位置偏见（positional bias）**等问题。

同时，现有数据清洗方法（如用 GPT-4o 重新蒸馏所有数据）成本高昂，难以扩展。

---

### 提出了什么新方法或新思路

作者提出 **DT2IT-MRM**，一个集成三大模块的框架，系统性解决上述问题：

#### （1）**Debiased Preference Distillation Pipeline（去偏置偏好蒸馏管道）**

- **单一模型生成候选响应**：使用同一个 MLLM 为同一 prompt 生成多个响应，确保文本风格一致，缓解**文本风格偏见**。
- **Listwise + Pointwise 双重打分机制**：
  - 先由教师模型（GPT-5.2）基于 ground-truth 答案对 K 个响应进行 listwise 打分；
  - 再独立进行 pointwise 打分，以消除**位置偏见**。
  - 仅保留两种打分结果一致的偏好对。
- **多样性增强模块**：若所有响应都高分或低分，则通过注入噪声或引入更强模型生成更优响应，提升**偏好强度多样性**。

#### （2）**Text-to-Image (T2I) Preference Reformulation（文本到图像偏好重构）**

- 将大规模人类标注的 **text-to-image 偏好数据**（如 EvalMuse、HPDv2）转化为适用于 MRM 训练的格式。
- 原始形式：`(prompt, I_c, I_r)` → 不适合 MRM 输入输出结构。
- 新形式：输入为 `(I_c, I_r, prompt, E)`，输出为 `"Image 1 is better"` 或 `"Image 2 is better"`，作为 `chosen/rejected response`。
- 更符合 MLLM 的图像理解范式，提升判别能力。

#### （3）**Iterative Training Framework（迭代训练框架）**

- 利用初步训练的 MRM 对开源噪声数据进行清洗，再用清洗后的数据重新训练 MRM，形成正向循环。
- 清洗流程：
  1. **Multi-MRM 投票**估计偏好强度，翻转负强度样本的标签；
  2. **一致性过滤**：保留与当前 MRM 判断一致的数据；
  3. **MLLM 重标注**：对不一致数据，使用多个开源 MLLM 进行成对评估并投票决定最终标签。
- 显著减少对昂贵闭源模型的依赖，实现高效、可扩展的数据净化。

---

### 相比现有方法的优势

| 维度 | DT2IT-MRM 的优势 |
|------|------------------|
| **数据质量** | 显著缓解文本风格偏见、位置偏见，提升偏好强度多样性与可靠性 |
| **数据效率** | 仅用约 35% 的数据量（929K vs BaseReward 的 2.8M）达到更优性能 |
| **可扩展性** | 迭代训练框架无需依赖闭源模型即可持续提升数据与模型质量 |
| **通用性** | 支持单图与多图输入，兼容多种下游任务 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

#### 构建初始数据 $ D_0 $：
- **单图偏好数据**：337K 对，通过 debiased 蒸馏管道构建。
- **多图偏好数据**：133K 对，通过 T2I 偏好重构方法从 EvalMuse 和 HPDv2 转换而来。
- 总计：**470K 初始偏好对**。

#### 开源数据清洗：
在第二阶段，清洗以下五个开源数据集：
- RLAIF-V
- VLFeedback
- POVID
- WildVision-Battle
- MM-RLHF

最终训练集：**929K 偏好对**（含清洗后数据）。

---

### 实验设置和评估指标

#### 模型架构
- **Discriminative MRM**：基于 Qwen3-VL-8B-Instruct 或 Qwen2.5-VL-7B-Instruct，附加线性 reward head。
- 损失函数：Bradley-Terry 风格损失（Eq. (1) 和 (3)）。

#### 训练策略
- 学习率：1e-5
- Batch size：512
- Epochs：1
- 优化器：cosine scheduler，warmup ratio = 0.1
- 硬件：64 × Ascend 910B3 NPUs
- 框架：LLaMA-Factory

#### 评估基准与指标

| Benchmark | 样本数 | 主要指标 |
|---------|--------|----------|
| **VL-RewardBench** | 1,247 | Overall Accuracy, Macro Average Accuracy |
| **Multimodal RewardBench** | 4,711 | Holistic evaluation across 6 aspects: General Correctness, Preference, Knowledge, Reasoning, Safety, VQA |
| **MM-RLHF-RewardBench** | 170 | Traditional Accuracy (**Acc**), **Acc+**（衡量弱偏好判断能力） |

---

### 基线方法对比

涵盖三类主流 MRM：

| 类别 | 代表模型 |
|------|--------|
| **Generative MRMs** | GPT-5.2, Claude-3.7-Sonnet, Gemini-1.5-Pro, LLaVA-Critic, UnifiedReward, MR.Judge-7B-SFT-RL |
| **Semi-scalar MRMs** | MM-RLHF-Reward |
| **Discriminative MRMs** | IXC-2.5-Reward, Skywork-VL Reward, BaseReward, Omni-RewardModel-BT |

特别强调与当前 SOTA **BaseReward** 的对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| Model | VL-RewardBench (Overall Acc) | Multimodal RewardBench (Overall Acc) | MM-RLHF-RewardBench (Acc) | **Mean Overall Acc** |
|-------|-------------------------------|----------------------------------------|----------------------------|------------------------|
| BaseReward | 82.2% | 72.8% | 91.8% | **75.2%** |
| **DT2IT-MRM (Ours)** | **83.5%** | **79.3%** | 89.4% | **80.5%** ✅ |

> ✅ **新 SOTA**：相比 BaseReward 提升 **5.3 个百分点**，且仅使用约 35% 的训练数据量。

---

### 与基线方法的对比结果

#### 在 VL-RewardBench 上：
- DT2IT-MRM 达到 **83.5%**，超越 BaseReward（82.2%）和最强闭源模型 GPT-5.2（70.2%）。
- 在 **Reasoning** 子任务上表现尤为突出。

#### 在 Multimodal RewardBench 上：
- 实现 **79.3%** 准确率，相较 BaseReward（72.8%）提升 **6.5 个百分点**，相对提升达 **8.9%**。
- 在 **General Correctness, Preference, Reasoning, VQA** 等维度均排名第一。

#### 在 MM-RLHF-RewardBench 上：
- 虽未第一（BaseReward 91.8% vs DT2IT-MRM 89.4%），但在 **Short** 子集达到 **100% 准确率**，显示对短序列强偏好判断的优越性。

> ⭐ 整体表现最均衡，在三大 benchmark 上综合性能最强。

---

### 消融实验结果

#### （1）Debiased Preference Distillation Pipeline 消融

| 方法 | VL-RewardBench | Multimodal RewardBench | MM-RLHF-RewardBench | Overall Acc |
|------|----------------|------------------------|---------------------|-------------|
| Listwise Scoring | 74.7 | 76.4 | 77.1 | 76.1 |
| + Diversity Enhancement | 75.6 | 77.2 | 79.4 | 76.9 |
| + Pointwise Scoring | **77.7** | **78.8** | **81.2** | **78.7** |

✅ **Pointwise 打分贡献最大**，有效缓解位置偏见。

#### （2）T2I 偏好重构方法对比（50K 数据）

| 方法 | VL-RewardBench | Multimodal RewardBench | MM-RLHF-RewardBench |
|------|----------------|------------------------|---------------------|
| Omni-Reward（Baseline） | 46.0 | 55.0 | 47.6 |
| **DT2IT-MRM（Ours）** | **56.5** (+10.5) | **65.0** (+10.0) | **59.4** (+11.8) |

✅ 新重构方式显著提升学习效率。

#### （3）Iterative Training 框架消融

| 方法 | Overall Acc |
|------|-------------|
| 仅用初始数据 $ D_0 $ | 79.4% |
| 直接混合原始开源数据（未清洗） | 75.5% ↓ |
| 经过两轮迭代清洗后 | **80.5%** ✅ |

⚠️ 直接加入噪声数据导致性能下降，验证了清洗必要性。

---

## 4. 关键结论和发现

### 主要发现

1. **偏好数据的质量远比数量重要**：  
   DT2IT-MRM 仅用 929K 数据，性能全面超越使用 2.8M 数据的 BaseReward，证明高质量、去偏置、多样化的数据更具价值。

2. **双重打分机制（listwise + pointwise）能有效缓解位置偏见**，提升标签可靠性。

3. **迭代训练框架实现了“模型提升 → 数据清洗 → 模型再提升”的正反馈循环**，是一种低成本、高效益的数据净化范式。

4. **T2I 偏好重构方法使 text-to-image 数据可用于 discriminative MRM 训练**，极大拓展了可用数据来源。

---

### 方法的局限性

1. **仍依赖少量闭源教师模型（GPT-5.2）进行初始打分**，尚未完全实现全开源闭环。
2. **在 Knowledge 和 Coding 维度表现相对较弱**，因训练数据中相关偏好对不足。
3. **对长视频或多步推理任务支持有限**，未来需扩展至更复杂场景。

---

### 未来工作方向

1. 完全使用开源 MLLM 替代闭源教师模型，实现端到端开源训练流程。
2. 引入主动学习策略，动态选择最有价值的样本进行标注。
3. 扩展至更多模态（如音频、3D）和更复杂的交互式偏好建模。
4. 探索将 DT2IT-MRM 应用于 MLLM 的在线 RLHF 流程中。

--- 

> ✅ **总结一句话**：  
> **DT2IT-MRM 通过“去偏置构造 + 数据重构 + 迭代清洗”三重创新，在更少数据下实现了多模态奖励模型的新 SOTA，兼具高性能、高效率与强泛化能力。**

</details>

---

### 7. [LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](https://arxiv.org/abs/2604.18913)

**Authors**: He Cheng, Yifu Wu, Saksham Khatwani, Maya Kruse, Dmitriy Dligach, Timothy A. Miller, Majid Afshar, Yanjun Gao  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 7.0  
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
现有的 **Knowledge Graph (KG)** 多跳检索系统在效率、可扩展性和可解释性之间难以平衡。传统图遍历算法（如 DFS/BFS）在大规模图上计算成本高，内存消耗大，尤其在生物医学领域，图可达数千万节点和上亿边，导致多跳检索不可行。此外，许多系统无法同时支持高效检索和路径重建（path reconstruction），限制了其在需要可解释推理场景（如医疗诊断）中的应用。

### **提出了什么新方法或新思路**
本文提出 **LOGOSKG**，一个硬件对齐（hardware-aligned）、可扩展且可解释的多跳 KG 检索框架。其核心思想是：
- 将 KG 分解为三个稀疏矩阵：**Subject (SUB)**、**Object (OBJ)** 和 **Relation (REL)**，将图遍历转化为高效的稀疏矩阵运算。
- 采用 **degree-aware partitioning** 对大规模图进行分块，结合 **cross-graph routing** 和 **on-demand caching** 实现跨子图的高效检索。
- 支持完整的路径重建，保留中间实体和关系，实现可解释的推理链。

### **相比现有方法的优势**
| 维度 | LOGOSKG | 现有方法（如 Neo4j, GraphBLAS, DGL, PyG） |
|------|---------|------------------------------------------|
| **效率** | 利用稀疏矩阵运算，支持 CPU/GPU 并行加速 | 指针遍历或密集张量操作，扩展性差 |
| **可扩展性** | 支持十亿级边的大图，通过分区和缓存管理内存 | 多依赖分布式架构，单机性能受限 |
| **可解释性** | 支持完整路径重建，提供推理证据链 | 多数仅返回结果集，无路径信息 |
| **硬件适配** | 单设备即可运行，优化数据布局和内核并行 | 部分需复杂基础设施 |

LOGOSKG 是首个在单设备上实现高效、可扩展、可解释多跳检索的统一框架。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **UMLS** (Unified Medical Language System)：407K 节点，3.4M 边，标准生物医学本体。
- **PubMedKG (PKG)**：54.4M 节点，86.5M 边，大规模文献引用网络。
- **PrimeKG**：整合 20 个高质量资源，描述 17,080 种疾病，约 4M 关系。

### **临床任务数据集**
- **ProbSum**：去标识化临床笔记，含症状和诊断，用于检索评估。
- **DDXPlus**：大规模症状-诊断对，用于 KG-LLM 交互研究。

### **实验设置**
- **硬件环境**：双 AMD EPYC 9454 CPU（192 线程），256GB RAM，两块 NVIDIA H100 NVL GPU（各 94GB VRAM）。
- **后端实现**：支持 Numba、SciPy、Torch（CPU/GPU）三种计算后端。
- **查询设置**：合成查询包含 1–20 个随机实体，执行 1–5 跳检索。
- **缓存策略**：LRU 缓存，固定容量 `n=16` 子图。

### **评估指标**
- **准确性**：Jaccard 相似度（与基线比较结果一致性）。
- **效率**：平均查询时间（Query Time, QT）和超时率（Timeout Rate, TR）。
- **可扩展性**：在不同跳数、批大小、缓存大小下的 QT、加载（loads）和驱逐（evicts）次数。
- **下游任务性能**：F1-score、Precision、Recall、PDSQI-9 评分（临床质量评估）。

### **基线方法对比**
涵盖四类主流系统：
- **数据库引擎**：Neo4j, TigerGraph
- **图分析工具**：igraph, NetworkX, graph-tool, SNAP
- **矩阵计算库**：GraphBLAS
- **GPU 框架**：cuGraph, DGL, PyG

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **表 2：多跳检索效率对比（UMLS + ProbSum）**
| 方法 | Hop 1 QT (ms) | Hop 5 QT (ms) | Hop 5 TR (%) |
|------|---------------|----------------|----------------|
| **LOGOSKG (Torch-GPU)** | **6.00** | **101.05** | **0.00** |
| LOGOSKG (Numba) | 12.28 | 204.25 | 0.00 |
| PyG | 249.66 | 735.34 | 0.00 |
| GraphBLAS | 3.03 | 415.43 | 0.00 |
| Neo4j | >923.86 | >10000.00 | 100.00 |

- **LOGOSKG (Torch-GPU)** 在所有跳数下均最快，Hop 5 查询时间仅 **101ms**，远低于其他方法。
- 传统方法（如 NetworkX, igraph）随跳数指数增长，而 LOGOSKG 呈近线性增长。

#### **表 3：LOGOSKG-Large 在 PKG 上的可扩展性（Numba 后端）**
| 设置 | Hop 2 QT (ms) | Loads | Evicts |
|------|----------------|--------|--------|
| Batch Size=1 | 100,912.09 | 12 | 0 |
| Batch Size=50 | 1,499.39 | 16 | 0 |
| Cache Size=1 | 441,870.19 | 3010 | 3009 |
| Cache Size=16 | 4,037.69 | 16 | 0 |

- **批量处理显著提升效率**：Batch=50 比 Batch=1 快 **67倍**。
- **缓存大小至关重要**：Cache=1 时因频繁磁盘 I/O 导致延迟极高，Cache=16 时性能恢复。

### **与基线方法的对比结果**
- 所有方法在 **Jaccard 相似度** 上均为 **1.00**（见附录 Table 4），证明 LOGOSKG 检索结果完全准确。
- 在 **Hop 5** 场景下，LOGOSKG (Torch-GPU) 比 PyG 快 **7倍**，比 Neo4j 快 **>99倍**。
- 在大规模图（PKG）上，LOGOSKG 成功完成 5 跳检索，而多数基线在 2–3 跳即超时。

### **消融实验结果**
- **缓存机制**：小缓存导致大量磁盘 I/O，延迟急剧上升；大缓存显著降低负载。
- **批处理优化**：批量查询共享子图访问，提高缓存命中率，减少重复加载。
- **后端选择**：Numba 和 Torch-GPU 表现最佳，Torch-CPU 因缺乏优化表现较差。

---

## **4. 关键结论和发现**

### **主要发现**
1. **LOGOSKG 实现了高效、可扩展、可解释的多跳检索**：
   - 通过矩阵分解和硬件优化，在单设备上支持十亿级边图的 5 跳检索。
   - 支持路径重建，为 LLM 提供可验证的推理证据。

2. **KG 结构深刻影响 LLM 推理行为**：
   - 在 **DDXPlus + PrimeKG** 上，随着跳数增加，KG 增强效果显著提升（见 Figure 3）。
   - **Round 1（过滤）** 提升精度，抑制幻觉；**Round 2（增强）** 提升召回，补充遗漏诊断。
   - 最终 F1-score 显著优于基线，尤其在长尾分布的 KG 中。

3. **临床诊断质量显著提升**：
   - 使用 **PDSQI-9** 评估，LOGOSKG 在 **accuracy extractive**, **organization**, **comprehensibility**, **succinctness**, **synthesis** 等维度均显著优于基线（见 Figure 4, Table 5）。
   - 说明 KG 不仅提升正确性，还改善诊断的结构化和表达质量。

4. **KG 增强在零样本场景下最有效**：
   - 对于 **fine-tuned LLMs**，KG 过滤可能误删模型已学知识，导致性能下降（见 Appendix A.5）。
   - 而在 **zero-shot** 场景中，KG 提供外部知识，显著弥补模型知识盲区。

### **方法的局限性**
- **依赖 KG 完整性**：若 KG 缺少某症状-诊断关联，则无法检索到相关证据。
- **LLM 重选能力有限**：Round 2 依赖 LLM 从候选集中选择，若模型判断错误，仍可能导致错误输出。
- **高跳检索产生噪声**：5 跳可能返回数千候选实体，需后续排序或过滤策略。

### **未来工作方向**
- 扩展至更多图算法（如最短路径、社区发现）。
- 集成更智能的排序/过滤机制（如基于语义相似度的 Top-N 选择）。
- 探索 **multi-agent + KG** 框架，实现迭代式、自修正的推理流程。
- 将 LOGOSKG 作为 **KG-LLM 训练** 的检索 backbone，支持证据增强的监督微调（见 Appendix A.7）。

---

> **源码与在线演示**  
> - GitHub: [https://github.com/LARK-NLP-Lab/LogosKG](https://github.com/LARK-NLP-Lab/LogosKG)  
> - Online Demo: [https://lark-nlp-lab-logoskg.hf.space/](https://lark-nlp-lab-logoskg.hf.space/)

</details>

---

### 8. [TRN-R1-Zero: Text-rich Network Reasoning via LLMs with Reinforcement Learning Only](https://arxiv.org/abs/2604.19070)

**Authors**: Yilun Liu, Ruihong Qiu, Zi Huang  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.19070v1  

#### Abstract
Zero-shot reasoning on text-rich networks (TRNs) remains a challenging frontier, as models must integrate textual semantics with relational structure without task-specific supervision. While graph neural networks rely on fixed label spaces and supervised objectives, recent large language model (LLM)...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TRN-R1-Zero: Text-rich Network Reasoning via LLMs with Reinforcement Learning Only

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **text-rich network (TRN)** 上的零样本节点分类方法存在以下局限：
- **Encoder-based 方法**（如 ZeroG、LLaGA）将 LLM 仅作为文本编码器，未能激发其显式的推理能力。
- **Generative 方法** 通常依赖于从更大模型（如 GPT-4 或 DeepSeek-R1）蒸馏出的 **Chain-of-Thought (CoT)** 数据，限制了泛化性和可扩展性。
- 多数方法需要任务特定监督或外部大型推理模型（LRMs）提供标注，难以实现真正意义上的“零样本”跨域迁移。

本文旨在解决：**如何在无监督、无 CoT 蒸馏、不依赖外部 LRM 的前提下，让基础 LLM 在 TRN 上具备显式的关系推理能力？**

---

### 提出的新方法与创新思路
作者提出 **TRN-R1-Zero** —— 一种**仅通过强化学习（Reinforcement Learning, RL）进行后训练**的框架，直接激活基础 LLM 在 TRN 上的推理能力。

#### 核心创新点：
1. ✅ **RL-only 训练范式**  
   - 不依赖任何监督微调（SFT）、CoT 数据蒸馏或外部 LRM。
   - 直接在基础 LLM 上应用 RL 进行优化，实现端到端的推理能力习得。

2. ✅ **Neighbour-aware Group Relative Policy Optimisation (GRPO) 目标函数**
   - 引入 **margin gain** 度量机制，量化邻居节点对目标节点分类决策边界的提升程度。
   - 动态调整奖励权重：当邻居信息显著改善预测 margin 时，赋予更高 reward，引导模型关注有价值的图结构上下文。

3. ✅ **无需结构编码器的 prompt 设计**
   - 将图结构自然语言化（natural language description），通过采样邻域构建输入 prompt。
   - 支持不同宽度（width）和深度（depth）的子图采样，增强鲁棒性并缓解长序列问题。

4. ✅ **跨层级零样本迁移能力**
   - 仅在 node-level 任务上训练，却能 zero-shot 推广至 edge-level 和 graph-level 任务，展现出强大的泛化潜力。

---

### 相比现有方法的优势
| 特性 | GraphWiz / Graph-NPH-R1 | Graph-R1 | TRN-R1-Zero |
|------|--------------------------|-----------|--------------|
| 是否依赖 LRM 提供 CoT | ✅ 是 | ✅ 是 | ❌ 否 |
| 是否需要 SFT 或蒸馏 | ✅ 是 | ✅ 是 | ❌ 否 |
| 是否使用 RL-only | ❌ 否 | ❌ 否 | ✅ 是 |
| 是否激活内在推理能力 | ❌ 外部模仿 | ❌ 外部监督 | ✅ 内生学习 |
| 可否跨任务 zero-shot | 有限 | 有限 | ✅ 支持 edge/graph 任务 |

> 🔍 TRN-R1-Zero 是首个完全摆脱对外部推理资源依赖，在 TRN 上实现纯 RL 驱动推理的方法。

---

## 2. 核心实验方法和设置

### 使用的数据集
共使用 **9 个 TRN 数据集**，涵盖四种关系类型：

| 类型 | 数据集 | 描述 |
|------|--------|------|
| **Citation** | Cora, Citeseer | 学术论文引用网络，节点为 paper segment |
| **Hyperlink** | WikiCS | Wikipedia 页面超链接图 |
| **Social** | Instagram | 用户社交关注关系，节点为 bio 或 post 文本 |
| **Co-purchase** | Photo, History | Amazon 商品共购关系，节点为评论或描述 |
| **Commonsense** | Expla-Graph | 常识概念支持/反驳关系图 |
| **Edge-level 构造数据** | WikiCS-Link, Instagram-Link | 由原始图构造的链接预测任务 |

> 📌 **训练集**：仅使用 `Citeseer`（citation）和 `History`（co-purchase）
>
> 📌 **测试集**：其余所有数据集用于 cross-domain 和 cross-task zero-shot 评估

---

### 实验设置与评估指标

| 设置项 | 说明 |
|-------|------|
| **Base Model** | Qwen2.5-7B-Instruct（主干），也验证了 Llama 和 Qwen 更大版本 |
| **RL 算法** | Dr.GRPO with KL 正则化（稳定训练） |
| **Prompt 设计** | 包含 `<think>` 和 `<answer>` 结构，强制生成中间推理过程 |
| **Neighborhood Sampling** | 固定 width-depth 策略采样子图，控制输入长度 |
| **Margin Gain 计算** | 使用 SGC(K=1) 聚合邻居嵌入，比较聚合前后 margin 变化 |
| **Reward Rescaling** | $ R = \exp(\alpha |\Delta_i|) \cdot (S_{\text{format}} + S_{\text{acc}}) $，$\alpha=10$ |

#### 评估指标
- **Accuracy (%)**
- **Macro-F1 (%)**
- 所有结果基于 zero-shot 设置报告（即模型未见过目标 domain）

---

### 基线方法对比
分为三类：

1. **LLMs**：
   - GPT-4o（强 baseline）
   - Llama-3.1-8B, Qwen2.5-1.5B/7B/14B-Instruct

2. **Graph Foundation Models (GFMs)**：
   - ZeroG：基于 LoRA 微调 Sentence-BERT 编码器
   - LLaGA：learnable soft embedding 对齐 GNN 与 LLM

3. **Reasoning LLMs**：
   - Graph-R1(14B)：使用 DeepSeek-v3 生成 CoT 并 SFT + RL 微调

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 2）
在 zero-shot 设置下的平均表现：

| Method | Avg. Accuracy (%) | Avg. Macro-F1 (%) |
|--------|--------------------|---------------------|
| GPT-4o | 63.09 | 61.07 |
| Qwen2.5-14B-it | 63.59 | 63.36 |
| Graph-R1 (14B) | – | – |
| **TRN-R1-Zero (7B)** | **66.53** | **64.35** |

> 💡 TRN-R1-Zero（7B）超越了更大的 Qwen2.5-14B 和 Graph-R1，在 **accuracy 和 F1 上均取得最佳结果**！

#### 分数据集表现亮点：
- **Cora**: 72.59 Acc → 比 Graph-R1 高 +4.44
- **WikiCS**: 73.63 Acc → 接近最优
- **Instagram**: 54.76 Acc → 显著优于其他方法
- **Photo**: 65.12 Acc → 远超 LLaMA 和 LLaGA

---

### 与基线方法的对比结果
- ✅ **显著优于 GFMs**：ZeroG 和 LLaGA 泛化差，尤其 LLaGA 在跨域下崩溃。
- ✅ **优于同规模 LLMs**：Qwen2.5-7B 经 RL 后性能大幅提升（+~10 pts）。
- ✅ **优于更强推理模型 Graph-R1**：尽管后者用了 14B 模型 + CoT 蒸馏，TRN-R1-Zero 仍全面领先，且更轻量高效。

---

### 消融实验结果

#### （1）Neighbour-aware Reward 的有效性（Figure 5）
- 使用 margin gain 加权 reward 后：
  - 训练更稳定（accuracy 曲线平滑上升）
  - 推理深度增加（response length ↑）
  - 输出中 “neighbour” 出现频率 ↑
  - 熵值保持较高 → 鼓励探索而非过早收敛

> 表明该设计有效促进了模型利用邻域信息进行深入推理。

#### （2）不同 backbone 的泛化性（Figure 6）
在多个模型家族上验证 TRN-R1-Zero 的通用性：

| Backbone | Average Gain (Acc) |
|---------|---------------------|
| Llama-3.2-3B | +14.4 |
| Llama-3.1-8B | +9.0 |
| Qwen2.5-14B | +4.7 |

> 即使是小模型也能获得巨大增益，证明该训练范式具有强适配性。

#### （3）cross-task zero-shot 性能（Table 3）
| Task | Model | Expla-Graph | WikiCS-Link | Insta-Link |
|------|-------|-------------|-------------|------------|
| Graph Reasoning | Base Qwen (14B) | 89.89 | 72.10 | 71.80 |
| | TRN-R1-Zero | **90.25** | **73.90** | **74.20** |
| Link Prediction | Base Qwen (7B) | 84.12 | 52.10 | 64.90 |
| | TRN-R1-Zero | **87.18** (+3.06) | **68.20** (+16.10) | **66.80** (+1.90) |

> ✅ 仅在 node 任务上训练，即可在 edge 和 graph 任务上实现显著增益，体现强大迁移能力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **强化学习足以独立激发 LLM 在 TRN 上的推理能力**，无需依赖 CoT 蒸馏或外部 LRM。
2. ✅ **Neighbour-aware reward shaping 是关键**：通过 margin gain 自适应强调高价值样本，显著提升推理质量与稳定性。
3. ✅ **TRN-R1-Zero 实现了真正的 zero-shot 跨域、跨任务泛化**，即使在未见的关系类型和任务形式上也表现优异。
4. ✅ **效率更高**：相比 Graph-R1，TRN-R1-Zero 使用更小模型（7B vs 14B）、生成更短响应（150 vs >900 tokens），推理更快、内存更低。

---

### 方法的局限性
- ❗ **依赖预训练阶段的知识覆盖度**：若 base LLM 缺乏某领域的常识（如法律、医学），RL 很难弥补知识鸿沟。
- ❗ **对噪声文本敏感**：TRN 中常存在简短、重复或无意义文本（如 "good product"），影响推理可靠性（见 Appendix A）。
- ❗ **margin gain 计算依赖冻结编码器**：当前使用 SGC + frozen encoder 计算 Δ，可能无法完全反映 LLM 内部表征变化。

---

### 未来工作方向
- 🔄 探索动态 margin gain 计算方式，结合 LLM 自身注意力机制。
- 🧠 引入 memory 或 retrieval 机制以应对知识缺失场景。
- 🌐 扩展到动态图、异构图或多模态 TRN 场景。
- ⚙️ 将该 RL-only 范式推广至更多 structured reasoning 任务（如 KG completion, program synthesis）。

---

## 总结
TRN-R1-Zero 成功证明了：**仅靠强化学习就能让基础 LLM 在 text-rich networks 上学会显式地融合语义与结构信息进行推理**，无需任何监督微调或外部“老师模型”。它不仅性能领先，而且更加简洁、高效、可泛化，为构建真正自主的图感知语言智能系统提供了新路径。

</details>

---

### 9. [POLAR-PIC: A Holistic Framework for Matrixized PIC with Co-Designed Compute, Layout, and Communication](https://arxiv.org/abs/2604.19337)

**Authors**: Yizhuo Rao, Xingjian Cui, Shangzhi Pang, Jiabin Xie, Guangnan Feng, Jinhui Wei, Ziyan Zhang, Languang Gao, Zhenyu Wang, Zhiguang Chen, Yutong Lu  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.19337v1  

#### Abstract
Particle-in-Cell (PIC) simulations are fundamental to plasma physics but often suffer from limited scalability due to particle-grid interaction bottlenecks and particle redistribution costs. Specifically, the particle-grid interaction computations have not taken full advantage of the emerging Matrix...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：POLAR-PIC: A Holistic Framework for Matrixized PIC with Co-Designed Compute, Layout, and Communication

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **Particle-in-Cell (PIC)** 模拟在大规模并行计算中面临三大瓶颈：
- **粒子-网格交互**（如 Field Interpolation 和 Deposition）受限于不规则内存访问模式，难以利用现代 **Matrix Processing Units (MPUs)** 的高算力。
- **粒子重排序**（Sorting）开销大，周期性全局排序导致显著运行时抖动（performance jitter）。
- **粒子通信**采用 **Bulk-Synchronous Parallel (BSP)** 模型，在每步末尾进行阻塞式重分布，严重破坏数据局部性和扩展性。

这些问题共同导致 PIC 模拟的可扩展性受限，尤其在 **Exascale 超算平台**上，粒子处理阶段常占总运行时间的 80% 以上。

---

### 提出的新方法与创新思路
作者提出 **POLAR-PIC**，一个面向下一代矩阵化 HPC 架构的协同设计框架，从 **Compute、Layout、Communication** 三个层面进行系统级优化：

#### ✅ 创新点 1：将 Field Interpolation 改造为 MPU 友好的外积形式（Outer-Product Reformulation）
- **问题**：Field Interpolation 是内积规约（inner-product reduction），而 MPU 原生支持的是外积累加（MOPA），存在结构性不匹配。
- **解决**：通过 **cell-centric 批处理策略**，将多个粒子的插值操作合并为一个矩阵乘法 $ F = W \times G $，从而适配 MPU 的外积执行模型。
- **优势**：首次实现 Interpolation 在 MPU 上的高效加速，避免 Amdahl 定律下的瓶颈转移。

#### ✅ 创新点 2：提出 **Sort-on-Write (SoW)** 机制维持物理连续布局
- **问题**：传统逻辑排序（如 Matrix-PIC）无法保证物理内存连续，导致 MPU 流水线利用率低；全量物理排序代价高昂。
- **解决**：利用粒子更新中的 **read-modify-write 路径**，在写回过程中动态分离“驻留粒子”和“迁移粒子”，前者保持物理连续，后者暂存于尾部无序区。
- **优势**：
  - 避免周期性全局排序，摊销开销接近 $ O(1) $。
  - 维持高密度 SoA 数据供给，保障 MPU 吞吐。

#### ✅ 创新点 3：基于 **UNR** 的细粒度通信重叠策略
- **问题**：BSP 模型下通信成为同步屏障，尤其在高动态场景（如 Laser-Ion Acceleration）中通信开销随规模增长。
- **解决**：
  - 将迁移粒子的打包（packing）融合进 Interpolation 写回路径。
  - 使用 **UNR (Unified Notifiable RMA)** 库发起非阻塞传输，与 Deposition 计算重叠。
- **优势**：有效隐藏通信延迟，实现高达 **99.1% 的重叠率**。

---

### 相比现有方法的优势
| 方面 | WarpX / VPIC | Matrix-PIC | **POLAR-PIC (本文)** |
|------|--------------|-----------|------------------------|
| **Interpolation 加速** | ❌ 仅 VPU | ❌ 未优化 | ✅ MPU 外积重构 |
| **数据布局维护** | 周期性全局排序 | 逻辑索引排序 | ✅ SoW 物理连续 |
| **通信模型** | BSP 阻塞同步 | BSP | ✅ UNR 异步重叠 |
| **整体协同性** | 低 | 中等 | ✅ 全流程 co-design |

---

## 2. 核心实验方法和设置

### 使用的数据集与工作负载
- **Uniform Plasma**：均匀等离子体微基准测试，用于评估不同粒子密度（PPC=1~512）和热速度（$ u_{th} = 0 \sim 0.2 $）下的性能。
- **Laser-Ion Acceleration (LIA)**：真实世界激光离子加速模拟，具有强非均匀粒子分布和高迁移率，验证复杂场景鲁棒性。

### 实验平台
- **主平台**：LS pilot system，搭载 **LX2 CPU**（双 Die，共 >256 核/节点），支持 **VPU 和 MPU**（8×8 MOPA 指令）。
- **对比平台**：配备 **NVIDIA A800 GPU** 的异构系统，运行 CUDA 版 WarpX。
- **互连**：支持 RDMA 的 LXLink 网络，带宽达 48 GB/s。

### 并行配置
- 使用 **MPI+OpenMP** 混合并行模型。
- 每节点 16 个 MPI rank（每个 NUMA 域一个），每个 rank 启用 32 个 OpenMP 线程。
- 弱扩展实验扩展至 **4,096 节点（超 200 万核）**。

### 评估指标
| 指标 | 定义 |
|------|------|
| **T_particle** | 粒子处理阶段总时间 = $ T_{Interpolation} + T_{Deposition} + T_{Redistribute} $ |
| **Speedup** | 相对于 WarpX 基线的加速比 |
| **PPS (Particles per Second)** | 每秒处理的宏粒子数 |
| **CPP (Cycles per Particle)** | 每个粒子消耗的 CPU 周期数（归一化到 1.3GHz） |
| **Overlap Ratio** | 通信被成功掩盖的比例 |
| **Peak Efficiency (%)** | 实际 FLOPs 占理论峰值比例 |
| **FOM_node** | 每节点性能指标：$ (\alpha N_c + \beta N_p) / (T_{steps} \cdot N_{nodes}) $ |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **WarpX-Native** | 原始 WarpX v24.07，基于 BSP，使用 VPU 自动向量化 |
| **Matrix-PIC** | 当前 SOTA，将 Deposition 改造成 MPU 外积，但 Interpolation 仍为 VPU，使用逻辑排序 |
| **POLAR-PIC** | 本文完整方案：MPU Interpolation + SoW + UNR 重叠 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🚀 整体加速比
| 场景 | 相比 WarpX | 相比 Matrix-PIC |
|------|------------|------------------|
| **Uniform Plasma**（高密度） | **10.9×** | **4.7×** |
| **Laser-Ion Acceleration** | **4.4×** | **3.8×** |

> 注：加速集中在粒子处理阶段，Field Solver 未优化。

#### 🔍 分项加速（消融实验）
| 操作 | 加速比 | 来源 |
|------|--------|------|
| **Field Interpolation** | **8.0×** | MPU 外积重构 + SoW 连续布局 |
| **Deposition** | **13.2×** | MPU 外积 + SoW 布局复用 |
| **Redistribution** | **3.0×** | 通信重叠 + 打包融合 |

#### ⏱️ 通信重叠效率
- **Overlap Ratio**：高达 **99.1%**
- **最大等待时间**：相比 MPI 非阻塞方案降低 **58×**

#### 💡 峰值效率对比（跨平台）
| 平台 | 方法 | Peak Efficiency |
|------|------|------------------|
| **LX2 CPU** | WarpX-Native | 1.0% |
| **LX2 CPU** | Matrix-PIC | 5.5% |
| **LX2 CPU** | **POLAR-PIC** | **13.2%** |
| **NVIDIA A800 GPU** | WarpX | 9.6% |

> POLAR-PIC 在 CPU 上达到甚至超过主流 GPU 实现的效率。

#### 📈 弱扩展性（Weak Scaling）
- 在 **4,096 节点（>200 万核）** 下：
  - **POLAR-PIC**：维持 **67.5%** 弱扩展效率
  - **WarpX-Native**：下降至 **42.5%**
- 粒子处理组件几乎实现近理想扩展（~100%），通信开销被有效掩盖。

---

### 消融实验结果

#### Ablation 1：Interpolation 优化路径
| 变体 | 描述 | 相对加速 |
|------|------|----------|
| G0 | 原始 WarpX（VPU） | 1.0× |
| G2 | 逻辑排序（Matrix-PIC） | 4.7× |
| G7 (**POLAR-PIC**) | MPU + SoW 物理连续 | **7.96×** |

> 表明 **物理连续性** 是释放 MPU 性能的关键。

#### Ablation 2：Deposition 布局复用
| 变体 | 描述 | CPP |
|------|------|-----|
| D0 | 原始原子操作 | 0.522 |
| D1 | MPU + 逻辑索引 | 0.310 |
| D3 (**POLAR-PIC**) | MPU + SoW 布局复用 + VPU 尾部回退 | **0.039** |

> SoW 布局复用使 Deposition 成本降低 **13.2×**。

#### Ablation 3：通信重叠策略
| 变体 | 描述 | 最大等待时间 |
|------|------|---------------|
| C0 | BSP 阻塞 | 0.58s |
| C1 | MPI 非阻塞 | 0.58s |
| C2 (**POLAR-PIC**) | UNR + SoW 预打包 | **0.01s** |

> UNR 硬件卸载 + SoW 融合打包是实现高重叠率的核心。

---

## 4. 关键结论和发现

### 主要发现
1. **协同设计至关重要**：单独优化 Deposition 或 Interpolation 不足以突破 Amdahl 瓶颈，必须从 **Compute、Layout、Communication** 三方面联合优化。
2. **物理连续性优于逻辑排序**：SoW 机制以极低开销维持物理内存连续，是释放 MPU 高吞吐能力的前提。
3. **通信重叠需硬件支持**：仅靠 MPI 非阻塞无法实现高重叠率，**UNR + RDMA** 是实现稳定异步通信的关键。
4. **CPU 上 PIC 性能可达 GPU 水平**：通过矩阵化重构，LX2 CPU 上的 POLAR-PIC 达到 **13.2% 峰值效率**，超越 A800 GPU 上的 WarpX（9.6%）。

---

### 方法的局限性
- **依赖特定硬件**：需要支持 MOPA 指令的 MPU（如 Intel AMX、Arm SME）和 RDMA 网络。
- **Field Solver 成为新瓶颈**：在粒子处理大幅加速后，**Field Solve** 成为主要耗时阶段（见弱扩展图），限制端到端加速潜力。
- **当前实现基于 WarpX**：虽具通用性，但移植到其他 PIC 框架需适配 AMReX 等底层模块。

---

### 未来工作方向
1. **优化 Field Solver**：将 Yee 求解器等也进行矩阵化改造，匹配加速后的粒子核。
2. **支持更复杂物理模型**：扩展至 **Quantum Electrodynamics (QED)**、自适应网格细化（AMR）等多物理场耦合场景。
3. **动态负载均衡**：结合 SoW 机制实现非均匀网格下的高效粒子迁移与调度。
4. **跨架构可移植性增强**：提供统一接口，便于在 Apple M 系列、国产芯片等平台上部署。

---

> **总结**：POLAR-PIC 通过 **operator reformulation + data layout + communication scheduling** 的三位一体协同设计，首次实现了 PIC 模拟全流程在矩阵化 CPU 架构上的高效执行，为下一代 HPC 上的大规模科学计算提供了新范式。

</details>

---

### 10. [Towards Scalable Lifelong Knowledge Editing with Selective Knowledge Suppression](https://arxiv.org/abs/2604.19089)

**Authors**: Dahyun Jung, Jaewook Lee, Heuiseok Lim  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19089v1  

#### Abstract
Large language models (LLMs) require frequent knowledge updates to reflect changing facts and mitigate hallucinations. To meet this demand, lifelong knowledge editing has emerged as a continual approach to modify specific pieces of knowledge without retraining the entire model. Existing parameter ed...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Towards Scalable Lifelong Knowledge Editing with Selective Knowledge Suppression 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **lifelong knowledge editing** 方法面临两大挑战：
- **Catastrophic forgetting**：在连续编辑过程中，新知识覆盖旧知识，导致先前编辑的知识被遗忘。
- **高训练成本与泛化能力差**：基于 retrieval-augmented 的方法（如 LTE、RECIPE）虽然提升了稳定性，但需要针对特定数据集进行微调，导致训练开销大且难以迁移到新数据集。

此外，许多参数修改方法（如 ROME、MEMIT）在序列编辑中容易引发 **model collapse** 或 **overfitting**，影响模型原始能力。

### 提出的新方法与思路
本文提出 **LightEdit**，一种轻量级、无需重新训练的 lifelong knowledge editing 框架，其核心思想是：
- **Selective Knowledge Suppression**：通过选择性地抑制模型原有的知识分布来实现高效编辑，而非直接修改模型参数。
- 包含两个关键组件：
  1. **Edit-Aware Selector**：一个基于 cross-encoder 的模块，判断检索到的知识是否与当前查询相关，仅保留必要信息。
  2. **In-Context Decoding**：在推理阶段动态调整生成概率，显式降低原始对象首词的 log probability，引导模型采纳新知识。

该方法完全 **training-free**（除 selector 外），实现了高效的 on-the-fly 编辑。

### 相比现有方法的优势
| 维度 | LightEdit | 现有方法（如 LTE, RECIPE） |
|------|---------|--------------------------|
| **训练成本** | 极低（仅需少量样本训练 selector） | 高（需 full fine-tuning 或 p-tuning） |
| **可扩展性** | 强，适用于未见数据集（zero-shot transfer） | 弱，依赖特定数据集优化 |
| **稳定性** | 在多轮编辑下保持高可靠性与局部性 | 易出现性能退化 |
| **参数更新** | 不修改原模型参数 | 修改或引入额外参数 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个主流 knowledge editing benchmark 上进行：
- **ZSRE**（Zero-Shot Relation Extraction）：用于零样本关系抽取任务。
- **Counterfact**：包含反事实知识编辑样本，测试模型对低概率陈述的修正能力。
- **RIPE**：评估知识注入后的“涟漪效应”（ripple effects）及泛化能力。

此外，在附录中还使用了：
- **UniEdit**：统一的大规模知识编辑基准（out-of-distribution 测试）
- **CSQA, MMLU, ANLI, TriviaQA**：评估编辑后模型通用能力是否受损

### 实验设置与评估指标
#### 评估场景
- 进行 **1,000 轮顺序知识编辑**，模拟真实 lifelong 场景。
- 查询分为两类：
  - **In-scope**：涉及已编辑知识
  - **Out-of-scope**：无关查询，用于测试 **locality**

#### 三大核心评估指标
| 指标 | 定义 |
|------|------|
| **Reliability** | 编辑后能否正确回答目标问题（准确性） |
| **Generality** | 对 paraphrased 查询仍能输出正确答案（泛化性） |
| **Locality** | 未编辑知识的回答不受干扰（稳定性） |

最终报告三者平均值（AVG）及各自得分。

### 基线方法对比
| 方法 | 类型 | 是否需训练 |
|------|------|----------|
| **BASE** | 原始模型 | 否 |
| **FT** | 全量微调 | 是 |
| **ROME / MEMIT** | 参数定位修改 | 是 |
| **GRACE / R-ROME** | 内存增强 + 权重冻结 | 是 |
| **AlphaEdit** | Null-space 约束编辑 | 是 |
| **LTE / RECIPE** | Retrieval-based 微调 | 是 |
| **LightEdit (Ours)** | Non-parametric + 概率控制 | 否（仅 selector 微调） |

所有方法均在 **LLaMA-3 (8B)** 和 **GPT-J (6B)** 上测试。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1, LLaMA-3 结果）
| Method | ZSRE (AVG) | CF (AVG) | RIPE (AVG) | 最高分 |
|--------|------------|----------|------------|-------|
| BASE | 0.5182 | 0.3377 | 0.5222 | — |
| FT | 0.0916 | 0.0203 | 0.0246 | ❌ |
| ROME | 0.0363 | 0.0300 | 0.0030 | ❌ |
| RECIPE | 0.9380 | 0.7664 | 0.5473 | ✅ (ZSRE) |
| **LightEdit** | **0.9664** | **0.9296** | **0.9818** | ✅✅✅ |

> ✅ LightEdit 在所有 benchmark 上均取得 **最高平均性能**

#### 分项表现亮点：
- **ZSRE**: Reliability 达 **0.9543**, Locality 高达 **0.9966**
- **Counterfact**: Generality 显著优于 RECIPE（0.8141 vs 0.7995）
- **RIPE**: 完全碾压其他方法，Locality 接近完美（0.9981）

### 与基线方法的对比结果
- **vs. LTE**: 更好地平衡了 reliability 与 locality，避免过度牺牲原有知识。
- **vs. RECIPE**: 尽管 RECIPE 在 ZSRE 表现强劲，但在 RIPE 和 Counterfact 上波动明显；而 LightEdit 表现更稳定。
- **vs. GRACE**: GRACE 虽然 locality 几乎完美，但 generality 极弱（~0.27），实用性受限。
- **vs. FT/ROME/MEMIT**: 所有参数修改类方法在长期编辑中迅速退化。

> 🔍 图3显示：随着编辑步数增加，几乎所有基线方法性能急剧下降，而 **LightEdit 保持高度稳定**。

### 消融实验结果（Ablation Study, Table 3）
| 设置 | AVG | Reliability | Generality | Locality |
|------|-----|-----------|------------|----------|
| Full LightEdit | 0.9664 | 0.9543 | 0.9483 | 0.9966 |
| - EAS (无 Edit-Aware Selector) | 0.7541 | 0.8898 | 0.9112 | 0.4612 |
| - ICD (无 In-Context Decoding) | 0.7615 | 0.9019 | 0.9223 | 0.4603 |
| 仅 Base + Retrieval | ~0.76 | ~0.90 | ~0.92 | ~0.46 |

> 📌 发现：
- 移除任一组件都会导致 **locality 急剧下降**，说明两者协同作用至关重要。
- **EAS 负责精准过滤**，防止噪声干扰；
- **ICD 负责有效引导生成**，提升 reliability 与 generality。

### 其他重要实验结果
#### 控制变量分析（α 值选择, Table 2）
- 当 `α = 0.2` 时性能最优，过高会损害 generality，过低则 suppression 不足。
- 验证了 **适度抑制 prior knowledge** 是成功关键。

#### 推理效率对比（Table 4）
| Method | Total Time (s) |
|--------|----------------|
| MEMIT | 11.9438 |
| AlphaEdit | 14.0371 |
| **LightEdit** | **0.2024** |

> ⏱️ LightEdit 的编辑时间接近于零（仅内存存储），总耗时最低。

#### 跨模型能力保持测试（Table 5）
| Method | CSQA | MMLU | ANLI | TriviaQA | AVG |
|--------|------|------|------|----------|-----|
| BASE | 0.7715 | 0.6810 | 0.4661 | 0.5183 | 0.6092 |
| **LightEdit** | **0.7715** | **0.6801** | **0.4738** | **0.5179** | **0.6108** |

> ✅ 编辑后模型在通用 NLP 任务上性能几乎不变，甚至略有提升。

---

## 4. 关键结论和发现

### 主要发现
1. **无需参数修改也能实现高效 lifelong editing**  
   通过 **in-context decoding** 抑制旧知识概率，即可完成准确编辑，打破“必须改参”的范式。

2. **Selective filtering + probabilistic control 是稳定编辑的关键**  
   - Edit-Aware Selector 实现精准知识匹配
   - In-Context Decoding 实现软性知识替换

3. **LightEdit 具备卓越的 scalability 与 zero-shot 泛化能力**  
   - 在未参与训练的 UniEdit 数据集上依然表现最佳（Table 14）
   - 支持高达 5,000 次编辑无显著性能衰减（Table 12）

4. **优于 ICL-based 方法（如 IKE）**  
   - IKE 受限于上下文长度，在多次编辑后性能下降明显（Table 13）
   - LightEdit 通过结构化机制实现可持续编辑

### 方法的局限性
1. **Edit-Aware Selector 仍需监督训练**  
   虽然训练成本低，但在真正 zero-shot 场景下仍需标注数据。

2. **输入长度随编辑次数增长而增加**  
   所有编辑知识都保留在 memory 中，可能带来推理延迟风险。

3. **超参数敏感性**  
   抑制系数 `α` 需要调优，不同领域或模型可能需要不同设置。

4. **依赖高质量 retrieval system**  
   若初始检索不准确，会影响后续选择与编辑效果。

### 未来工作方向
- 设计 **fully zero-shot selector**，利用 LLM 自身推理能力判断相关性
- 引入 **knowledge pruning / summarization** 机制，减少冗余信息积累
- 探索 **adaptive α 控制策略**，根据冲突强度自动调节抑制力度
- 加强伦理考量：研究如何防止恶意知识注入与偏见放大

---

> 💡 **总结一句话**：  
> **LightEdit 通过“选择性知识抑制”实现了高效、稳定、可扩展的 lifelong knowledge editing，在性能、效率与泛化之间取得了前所未有的平衡，为大规模语言模型的持续维护提供了实用解决方案。**

</details>

---

### 11. [A Mechanism and Optimization Study on the Impact of Information Density on User-Generated Content Named Entity Recognition](https://arxiv.org/abs/2604.18944)

**Authors**: Jiang Xiaobo, Dinghong Lai, Song Qiu, Yadong Deng, Xinkai Zhan  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.18944v1  

#### Abstract
Named Entity Recognition (NER) models trained on clean, high-resource corpora exhibit catastrophic performance collapse when deployed on noisy, sparse User-Generated Content (UGC), such as social media. Prior research has predominantly focused on point-wise symptom remediation -- employing customize...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Mechanism and Optimization Study on the Impact of Information Density on User-Generated Content Named Entity Recognition

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于 **User-Generated Content (UGC)** 场景下的 **Named Entity Recognition (NER)** 性能退化问题。尽管在标准文本上表现优异的 NER 模型（如 BERT、RoBERTa）在部署到社交媒体等噪声大、稀疏性强的 UGC 数据时，性能急剧下降。

传统研究多从**表面症状**入手（如拼写变异、新词、长尾实体），采用定制化微调策略进行点对点修复，但这些方法缺乏泛化能力，且忽略了 UGC 中更深层的结构性缺陷。

### 🔍 提出的新方法与新思路

#### （1）提出“信息密度”（Information Density, ID）为核心驱动因素
- 首次系统性地将 UGC 中多种噪声现象归因于一个统一的根本原因：**低信息密度（Low Information Density）**。
- 定义 ID 为：**局部上下文中有效语境线索对实体信号的支持强度**，而非简单的实体比例。
- 引入修正项以考虑句子长度影响，公式如下：
  $$
  \text{NED} = \frac{\text{ET}}{\text{TT}_s} \cdot \left(1 + \log(\text{TT}_s)\right)^{-\lambda}
  $$
  其中 ET 是实体 token 数，TT_s 是含实体句子的总 token 数，λ 是结构因子。

#### （2）机制剖析：揭示低 ID 如何损害模型性能
- 提出 **Attention Spectrum Analysis (ASA)**，一种基于频域分析的诊断工具，用于量化注意力机制中的“注意力钝化”（attention blunting）现象。
- 揭示两条核心路径：
  1. **统计保守偏置（Statistical Conservative Bias）**：低 ID 导致非实体标签（'O' 类）占绝对主导，使模型优化过程偏向预测背景 token，牺牲召回率。
  2. **注意力钝化（Attention Blunting）**：低 ID 下上下文语义同质化，导致 Query-Key 相似度趋同，Softmax 后注意力分布趋于均匀，无法聚焦关键边界。

#### （3）提出 **Window-aware Optimization Module (WOM)**
- 一种 **LLM 驱动、model-agnostic** 的数据级增强框架。
- 核心思想：通过滑动窗口扫描识别信息稀疏区域，并仅对其中含有实体的句子执行选择性回译（back-translation），从而定向提升语义密度。
- 包含三大组件：
  - **Window Segmentation**：按句切分文本流
  - **Window Detection Engine**：计算窗口内信息密度，低于阈值 T 则标记为稀疏区
  - **Back-translation Enhancement**：利用 LLM 进行高质量回译（如英→中→英），保留实体边界与类别，增加表达多样性

### 🆚 相比现有方法的优势
| 维度 | 传统方法 | 本文 WOM |
|------|--------|---------|
| 分析视角 | 表面症状修复（如拼写纠正） | 结构性根源挖掘（ID 驱动） |
| 方法通用性 | 特定任务/架构依赖 | Model-agnostic，适用于任意 NER 架构 |
| 干预层级 | 模型结构修改或训练策略调整 | 数据层面干预，不改变原模型 |
| 增强方式 | 全局数据增强（易引入噪声） | 局部选择性增强（精准控制） |
| 可解释性 | 黑箱式改进 | 有明确机制支撑（ASA + 统计偏置） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **WNUT2017**：主实验数据集，最具挑战性的 UGC NER benchmark，包含大量新兴实体和非规范表达。
- **Twitter-NER**：另一广泛使用的推特 NER 数据集，验证跨分布泛化能力。
- **WNUT2016**：作为历史版本，用于检验方法普适性。

### ⚙️ 实验设置与评估指标
- **评估指标**：采用标准 **F1-score**（精确率与召回率的调和平均）
- **模型架构**：
  - 主要分析模型：`RoBERTa-BiLSTM-CRF` 和 `SpanNER`
  - 扩展对比模型：BERT、MINER、Context(RoBERTa) 等多个 SOTA 基线
- **训练策略**：在原始训练集上应用 WOM 进行数据增强后，再训练各模型，在测试集上报 F1。
- **控制变量**：通过分层重采样构建不同 ID 水平的子集，排除实体稀有性和标注一致性干扰。

### 🔁 基线方法对比
| 方法 | 类型 |
|------|------|
| BERT / RoBERTa 系列 | 主流 PLM 微调 |
| SpanNER | Span-based NER 范式代表 |
| MINER | 针对 OOV 实体设计的信息论方法 |
| Context(RoBERTa) | 引入补充特征的序列标注模型 |
| CL-KL | 外部上下文检索与协同学习方法 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（F1-score）

| Model | WNUT2017 | Twitter-NER | WNUT2016 |
|-------|----------|-------------|----------|
| SpanNER (BERT-large) | 51.20 | 74.93 | 53.17 |
| → +WOM | **55.77 (+4.57%)** | **75.94** | **55.36** |
| RoBERTa-BiLSTM-CRF | 59.54 | 77.85 | 59.03 |
| → +WOM | **61.39 (+1.85%)** | **78.52** | **60.35** |
| MINER + WOM | — | — | **56.67** |
| Context + WOM | — | — | **60.69** |

> ✅ WOM 在多个架构上均带来 **1.0% ~ 4.5% 的绝对 F1 提升**，并在 **WNUT2017 上达到新的 SOTA**。

### 🔍 消融实验结果（Ablation Study）

使用 `RoBERTa(base)-BiLSTM` 模型在 WNUT2017 上进行消融：

| 设置 | F1-score | 说明 |
|------|---------|------|
| Baseline（无增强） | 57.35 | 原始训练集 |
| Global Augmentation（全局回译） | 56.35 ↓ | 性能下降！说明盲目增强会引入噪声干扰 |
| WOM（完整模块） | **59.25 ↑** | 显著优于 baseline 和 GA，证明选择性增强的有效性 |

> 💡 结论：**不是所有数据都适合增强**；只有针对信息稀疏区域的选择性增强才能真正提升性能。

### 🔬 超参数敏感性分析
- **窗口大小 W**：最优值约为 30。过小导致估计不稳定，过大削弱局部检测灵敏度。
- **密度阈值 T**：最优值约为 0.07。过高则过度增强（引入噪声），过低则覆盖不足。
- WOM 在合理范围内表现稳健，表明其具备较强实用性。

---

## 4. 关键结论和发现

### 🎯 主要发现
1. **信息密度（ID）是影响 UGC-NER 性能的关键结构性因素**，其相关性远高于其他特征（如实体不平衡、冗余、多义性等）。
2. 低 ID 通过两大机制损害模型：
   - **统计保守偏置**：'O' 类样本主导训练，导致高 FN（漏检），降低 Recall。
   - **注意力钝化**：ASA 指标证实低 ID 导致注意力频谱高频成分减少，模型难以捕捉细粒度边界。
3. WOM 成功实现了“机制驱动”的优化：通过定位低 ID 区域并选择性增强，有效打破上述两个负反馈循环。
4. WOM 是 **model-agnostic** 且 **无需修改模型结构** 的轻量级插件式模块，易于部署。

### ⚠️ 方法的局限性
- **依赖 LLM 回译质量**：若 LLM 未能准确还原实体或产生语义漂移，可能引入错误样本。
- **阈值需调参**：虽然鲁棒，但仍需根据数据集调整 T 和 W，尚未完全自动化。
- **未处理极端短文本**：极短 tweet（如 <5 tokens）可能无法形成有意义的上下文窗口。
- **计算开销增加**：回译过程增加了预处理时间，尤其在大规模数据上。

### 🔮 未来工作方向
1. **Learnable Causal ID Estimator**：开发可学习的因果 ID 估计器，超越当前代理指标。
2. **ASA 正则化嵌入**：将 ASA 作为训练时的正则项，主动抑制低 ID 下的注意力钝化。
3. **Dynamic WOM Optimization**：结合检索机制实现动态窗口划分，支持在线流式增强。
4. **扩展至其他 IE 任务**：将 ID 框架推广至 Relation Extraction、Entity Linking 等任务，探索其通用性。

---

## ✅ 总结

本论文开创性地将 **Information Density** 提升为理解 UGC-NER 性能瓶颈的核心概念，提出了兼具理论深度与工程实用性的解决方案 WOM。它不仅取得了显著的性能突破（最高 +4.5% F1），更重要的是推动了 NLP 社区从“经验调参”向“机制驱动”的研究范式转变，倡导将 **ID 作为 UGC 数据集的标准属性进行报告**，具有重要的学术价值与实践意义。

</details>

---

### 12. [DPC: A Distributed Page Cache over CXL](https://arxiv.org/abs/2604.19494)

**Authors**: Shai Bergman, Zhe Yang, Julien Eudine, Giorgio Negro, Onur Mutlu, Arash Tavakkol, Ji Zhang  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19494v1  

#### Abstract
Modern distributed file systems rely on uncoordinated, per node page caches that replicate hot data locally across the cluster. While ensuring fast local access, this architecture underutilizes aggregate cluster DRAM capacity through massive data redundancy and incurs prohibitive coherence overhead ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DPC: A Distributed Page Cache over CXL

## 1. 论文的主要贡献和创新点

### 解决的问题
现代分布式文件系统采用**独立的、每节点（per-node）page cache**架构，导致两个根本性问题：
- **内存冗余严重**：热点数据在多个节点上被重复缓存，浪费大量集群 DRAM 容量。
- **一致性开销高昂**：为保证缓存一致性，需依赖重量级的锁机制或分布式锁管理器（如 LDLM），带来显著的控制平面开销。

传统方案无法将整个集群的主存视为一个统一的缓存池，限制了全局局部性（global locality）的利用。

---

### 提出的新方法与创新思路
论文提出 **DPC (Distributed Page Cache)** ——一种基于 **CXL 3.0** 的操作系统级分布式 page cache，其核心思想是：
- 将整个集群的 DRAM 视为单一逻辑缓存池。
- 强制执行 **单副本不变式（single-copy invariant）**：每个文件页最多只有一个“拥有者”节点持有其物理副本，其他节点通过远程映射访问该副本。
- 利用 **CXL.mem 协议**实现低延迟、硬件管理的缓存一致性（cache-line granularity），避免软件 RPC 开销。

#### 核心组件设计：
- **DPC Directory**（位于存储服务器）：
  - 维护每个缓存页的状态机（Invalid, Exclusive, Owner, Shared, To-Be-Invalidated）。
  - 协调跨节点的页所有权转移、远程映射与回收。
- **DPC Client**（运行于计算节点）：
  - 与内核 VFS 集成，透明支持标准接口（如 POSIX I/O 和 `mmap`）。
  - 通过 `dpc_dax` 驱动将远程 CXL 内存注册为 `ZONE_DEVICE`，使远程页对内核表现为本地页。

---

### 相比现有方法的优势
| 方面 | 传统方案（如 NFS, VirtioFS） | DPC |
|------|-------------------------------|-----|
| 缓存粒度 | 多副本、无协调 | 单副本、全局协调 |
| 远程访问方式 | 软件 RPC / 文件服务请求 | CXL 直接 load/store（硬件一致性） |
| 控制开销 | 高（锁、租约、回调） | 低（目录查询 + 批量失效） |
| 接口兼容性 | 通常需应用修改或特定客户端 | 完全透明，无需改动应用 |
| 性能潜力 | 受限于网络栈和文件协议 | 接近远程内存访问延迟 |

> ✅ **优势总结**：DPC 在保持标准语义的同时，实现了接近本地内存访问速度的远程缓存命中，并显著减少内存浪费和一致性开销。

---

## 2. 核心实验方法和设置

### 实验平台与模拟环境
由于尚无广泛可用的多主机 CXL 3.0 硬件，作者构建了一个基于 **QEMU/KVM 的 CXL 仿真框架**：
- **硬件配置**：双路 ARMv8.2 服务器，共 128 核，256 GiB DDR4 内存，4×480GB SAS SSD 组成 RAID-0。
- **虚拟化设置**：使用 QEMU 创建多个 VM 作为“计算节点”，每个 VM 分配 16 vCPU 和 32 GiB RAM。
- **CXL 模拟**：通过 `ivshmem` 共享内存机制模拟 CXL.mem 行为，所有 VM 共享宿主机 DRAM 并由底层 NUMA 提供硬件一致性模型。
- **存储后端**：由 `virtiofsd` 提供 XFS 文件系统服务。

---

### 基线方法对比
评估中对比了以下系统：
- **Virtiofs**：原始未修改的共享文件系统（基线）
- **NFSv4.1**：主流网络文件系统
- **JuiceFS**：云原生分布式 POSIX 文件系统
- **DPC_SC**：DPC 启用强一致性模式（Strong Consistency）
- **DPC**：DPC 启用宽松一致性模式（类似 NFS）

---

### 数据集与工作负载
使用多种真实世界和代表性数据共享场景进行测试：

| 应用 | 描述 | 工作集大小 | 指标 |
|------|------|------------|-------|
| **RocksDB** | KV 存储引擎，随机读基准 | 60 GB | Queries Per Second (QPS) |
| **DeepSeek** | CPU 推理模型加载（30GB 权重文件 `mmap`） | 30 GB | Tokens Per Second (TPS) |
| **DiskANN** | 磁盘近邻搜索索引 | 40 GB | QPS |
| **Webserver** | Filebench 模拟静态内容服务 | 32 GB | 几何平均吞吐量 |
| **Fileserver** | 多用户并发读写共享目录 | ~32 GB | 几何平均吞吐量 |

此外还进行了微基准测试（microbenchmarks）：
- 使用 `fio` 测试不同缓存状态下的读写延迟、带宽、IOPS。
- 场景包括：
  - **CM (Cache Miss)**：目标页未缓存
  - **CM-R (Cache Miss-Remote)**：目标页在远程节点已缓存
  - **CH-R (Cache Hit-Remote)**：本节点已有远程映射

---

### 评估指标
- **延迟（Latency）**：I/O 请求响应时间
- **带宽（Bandwidth）**：顺序读写吞吐量
- **IOPS**：随机小 I/O 吞吐能力
- **Speedup**：相对于单节点 Virtiofs 的加速比
- **Geomean Speedup**：跨工作负载的几何平均加速比

---

## 3. 主要实验结果和性能指标

### 微基准测试结果（`fio`）
#### 🔹 读操作（libaio & mmap）
| 场景 | 性能提升 |
|------|--------|
| **CM-R**（首次远程命中） | 延迟降低 **2.6×**，IOPS 最高提升 **72.8×** |
| **CH-R**（后续远程命中） | 延迟主导为 remote memory access，相比 Virtiofs 达到 **23.3× 更低延迟**，带宽达 **4.5× 提升** |

> 💡 说明：一旦建立远程映射，性能远超从 SSD 加载。

#### 🔹 写操作（libaio & mmap）
- **DPC_SC（强一致）** 在写路径引入额外控制开销（获取独占权），导致 CM 场景下写延迟略高（~195μs vs ~100μs）。
- 但在 **CM-R** 场景下可通过复用远程页避免本地分配，部分抵消开销。
- 大块写入时可通过批量处理隐藏目录延迟。

---

### 应用级性能结果（图 10）
| 工作负载 | DPC 最大加速比 | DPC_SC 最大加速比 | Geomean（跨负载） |
|---------|----------------|--------------------|------------------|
| **DeepSeek** | **12.4×** | 9.5× | — |
| **Webserver** | **16.2×** | 15.2× | — |
| **Fileserver** | **15.2×** | 14.0× | — |
| **综合几何平均** | — | — | **5.6×** |

> ✅ **关键观察**：
> - 当多个节点共享工作集时，DPC 显著受益于跨节点缓存复用。
> - 即使增加节点数（1→4），每节点性能不下降，表明控制平面可扩展。
> - DPC_SC 因强一致性略慢于 DPC，但差距有限（2-node 下 geomean 2.5× vs 2.8×）。

---

### 消融实验结果
#### 页面回收开销分析（§6.2.5）
- **同步单页失效延迟**：
  - Virtiofs：11 μs（仅本地）
  - DPC：99.7 μs（需通知远程并等待 ACK）
- **实际影响**：
  - 在内存压力下（2GB VM 跑大文件读），DPC 通过异步批处理机制（batched invalidation）完全掩盖此开销。
  - 最终带宽与 Virtiofs 相当 → 表明 **批处理有效防止了回收成为瓶颈**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **集群内存可作为统一缓存池**：通过 DPC 的单副本策略，聚合 DRAM 容量得到有效利用，避免了传统 per-node cache 的严重冗余。
2. ✅ **CXL 是实现高效分布式缓存的理想底座**：CXL 3.0 的硬件一致性机制使得远程访问如同本地内存，极大降低了远程缓存命中的延迟和软件复杂度。
3. ✅ **性能提升显著且稳定**：在真实应用中，DPC 实现高达 **12.4×** 的加速，几何平均达 **5.6×**，尤其在低计算强度、高 I/O 密集型负载中表现突出。
4. ✅ **控制平面可扩展**：目录服务和跨节点协调未成为瓶颈，支持多节点并发访问。

---

### 局限性
1. ❌ **依赖 CXL 3.0 硬件普及**：当前仍处于早期部署阶段，缺乏大规模商用验证。
2. ❌ **故障容忍有限**：
   - 节点崩溃会导致其拥有的 dirty page 丢失（同传统 write-back cache）。
   - 虽有 liveness 检测机制，但未提供完整容错恢复能力（如副本重建）。
3. ❌ **安全性依赖硬件隔离**：需 CXL 适配器具备 per-page 权限控制（如 IOMMU 类机制），否则存在越权访问风险。
4. ❌ **强一致性模式有性能代价**：频繁写入场景下，DPC_SC 的两阶段协议会引入可观测延迟。

---

### 未来工作方向
1. **增强容错机制**：探索轻量级复制或日志机制，在不牺牲太多性能的前提下提高 durability。
2. **智能预取与迁移**：结合 workload prediction，主动迁移热点页以减少远程访问。
3. **与 tiered memory 协同优化**：将 DPC 与 CXL 池化内存（pooled memory）结合，形成层次化全局缓存体系。
4. **更细粒度权限与安全模型**：研究如何在 CXL fabric 上实现动态访问控制和加密传输。
5. **部署策略优化**：研究何时允许 dirty page 共享 vs 保守锁定，以平衡性能与一致性。

---

## 总结
DPC 是首个将 **CXL 3.0 的硬件一致性能力** 与 **操作系统 page cache 抽象** 深度融合的工作，提出了一个真正意义上的、透明的、高性能的分布式 page cache 架构。它不仅解决了长期存在的缓存冗余与一致性难题，也为未来基于 CXL 的 **cluster-wide memory management** 提供了重要范式。

</details>

---

### 13. [Collaborative Contextual Bayesian Optimization](https://arxiv.org/abs/2604.18912)

**Authors**: Chih-Yu Chang, Qiyuan Chen, Tianhan Gao, David Fenning, Chinedum Okwudire, Neil Dasgupta, Wei Lu, Raed Al Kontar  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 6.5  
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
本文针对**Contextual Bayesian Optimization (CBO)** 在多客户端场景下的效率瓶颈问题。传统 CBO 需要为每个上下文（context）寻找最优设计（design），学习目标是整个函数映射 $x^*(c) = \arg\max_x f(x, c)$，这比标准 BO 更加资源密集。当多个相关但异构的客户端各自独立执行 CBO 时，样本效率低下。

现实应用中（如制造业、医疗），多个客户可能面临相似的任务（例如不同设备优化热轧参数），若能协作共享知识，可显著加速学习过程。然而，现有方法缺乏对**多客户端协同 CBO** 的系统建模与理论支持。

---

### **提出了什么新方法或新思路**
作者提出 **CCBO (Collaborative Contextual Bayesian Optimization)**，一个统一的分布式 CBO 框架，其核心创新包括：

- ✅ **跨客户端协作机制**：允许多个客户端联合学习上下文到最优设计的映射 $x^*(c)$，通过共享后验均值（posterior mean）构建全局操作模型（operational model）。
- ✅ **基于分歧的决策驱动（Disagreement-driven switching）**：识别本地推荐与全局推荐差异最大的 context，优先采样这些“不一致”区域，从而高效纠正局部模型偏差。
- ✅ **自适应切换策略（Adaptive switching）**：引入概率门控 $p_t$ 控制协作强度，早期高协作以利用群体智慧，后期逐渐转向独立探索，避免因客户端异质性导致的负迁移。
- ✅ **支持离线初始化与隐私保护通信**：
  - 支持单个活跃客户端从其他已有历史数据的客户端中受益（offline collaborative initialization）；
  - 利用 **Random Fourier Features (RFF)** 近似并压缩后验均值，仅传输低维系数向量，实现轻量级、隐私友好的通信。

---

### **相比现有方法的优势**
| 对比维度 | 现有方法（如 FTS、MTS） | CCBO |
|--------|----------------------|------|
| 协作目标 | 寻找单一最优设计 per client | 学习完整的 context-dependent 映射 $x^*(c)$ |
| 探索机制 | 独立探索或随机上下文选择 | 主动选择“本地-全局推荐差异大”的 context |
| 客户端异质性处理 | 易受负迁移影响 | 自适应衰减协作频率，动态平衡协作与个性化 |
| 隐私与通信开销 | 可能需共享原始数据或完整模型 | 仅共享 RFF 压缩后的权重，降低暴露风险 |
| 理论保障 | 多数无针对 CBO 的 regret 分析 | 提供 sublinear regret 上界与通信成本分析 |

> 🔍 **一句话概括优势**：  
> CCBO 是首个专为多客户端 CBO 设计的协作框架，在保证理论收敛性和隐私安全的前提下，显著提升样本效率，尤其适用于存在历史数据或边缘设备受限的工业场景。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验分为两部分：

#### （1）**仿真基准测试（Simulation Studies）**
采用经典黑盒优化函数作为响应函数 $f(x, c)$，输入空间归一化至 $[0,1]^{D_c + D_x}$：
- **Ackley 函数**（同质设定）
- **Levy 函数**（异质设定）
- **Hartmann 函数**（用于分析协作规模效应）

上下文维度 $D_c$ 和设计维度 $D_x$ 设置多样（如 2-1, 2-2, 1-3 表示 $D_c$-$D_x$ 组合）。

#### （2）**真实世界案例研究：热轧工艺优化（Hot Rolling）**
- **任务**：优化金属板材晶粒尺寸 $Z$
- **设计变量 $x$**：角速度 $\omega$、轴向力 $F_r$、热通量 $q$
- **上下文 $c$**：最终厚度 $h_f$
- **客户端异质性来源**：滚轮半径 $R_k \sim \mathcal{N}(0.5, 0.1)$
- **模拟器**：COMSOL Multiphysics 耦合热力学与晶粒演化模型，训练 NN 替代模型用于快速评估

---

### **实验设置和评估指标**

#### **通用设置**
- 客户端数量 $K = 10$
- 总迭代次数 $T = 20(D_c + D_x)$ 或固定为 50（热轧）
- 每次迭代从 100 个候选 context 和 100 个候选 design 中选择
- 初始数据量 $T_0 = 5(D_c + D_x)$
- 观测噪声：$y = f(x,c) + \epsilon$, $\epsilon \sim \mathcal{N}(0, 0.1\sigma_f)$
- 协作概率调度：$p_t = 1/\sqrt{t}$

#### **评估指标**
使用 **对数尺度的整体简单遗憾（log-scale overall simple regret）**：
$$
G_t = \frac{1}{K} \sum_{k=1}^K \frac{\int_C \left[f_k(x^*(c),c) - f_k(\hat{x}_k(c),c)\right] dc}{\int_C \max_{x_1,x_2} \left[f_k(x_1,c) - f_k(x_2,c)\right] dc}
$$
该指标衡量当前估计的设计相对于真实最优设计的平均差距，并进行了归一化以便跨函数比较。

---

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **RS (Random Sampling)** | 随机选择 context-design 对 |
| **MTS (Multi-task Thompson Sampling)** | 独立客户端执行 CBO，无协作（Char et al., 2019） |
| **FTS (Federated Thompson Sampling)** | 联邦 TS，但 context 随机选择，仅 design 协作（Dai et al., 2020） |
| **CCBO (Ours)** | 所提方法，支持 context-aware 协作与自适应切换 |
| **CCBO + RFF** | 使用 RFF 压缩通信版本 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据与对比结果**

#### ✅ **同质环境（Homogeneous Setting）**
- 使用 Ackley 函数族进行测试（Fig. 3）
- **结果**：CCBO 在所有阶段均显著优于 FTS、MTS 和 RS
- **早期性能提升明显**：在前 20 次迭代内，$G_t$ 下降速度最快，表明协作有效加速高质量区域探索

#### ✅ **异质环境（Heterogeneous Setting）**
- 各客户端响应函数加入随机偏移：$f_k(x,c) = f(x+\delta_x, c+\delta_c)$, $\delta \sim U(-0.05, 0.05)$
- 使用 Levy 函数族测试（Fig. 4）
- **结果**：CCBO 仍显著优于 MTS 和 FTS，即使在强异质下也保持稳定增益
- **原因**：自适应切换机制防止过度依赖错误的全局模型

#### ✅ **协作规模效应（Number of Clients）**
- 使用 Hartmann 2-2 函数，改变 $K = 2, 5, 10, 15$（Fig. 5）
- **发现**：
  - $K=2$: 性能差且方差大（信息不足）
  - $K=10$: 显著提升，收敛快且稳定
  - $K=15$: 略好于 $K=10$，边际收益递减
- **结论**：至少需要中等规模客户端群 ($K \geq 10$) 才能充分发挥协作优势

#### ✅ **RFF 隐私通信效果**
- 使用 Levy 1-3 异质设置，J=50（Fig. 6）
- **结果**：
  - CCBO with RFF 略逊于原版 CCBO（初期）
  - 但随时间推移，两者性能趋同
  - 均远优于独立 MTS
- **意义**：RFF 在几乎不牺牲最终性能的前提下，实现了更安全、高效的通信

#### ✅ **真实应用场景：热轧工艺优化（Fig. 8）**
- $K=10$ 异质客户端，$T=50$
- **结果**：
  - RS 完全无法收敛
  - FTS 与 MTS 表现相近
  - **CCBO 显著更快降低 regret，达到最低稳态误差**
- **验证了方法在复杂物理系统中的实用性**

---

### **消融实验（隐含分析）**
虽然未明确命名“ablation”，但以下对比构成实质上的消融分析：
- **是否启用协作？** → CCBO vs MTS：协作带来显著增益
- **是否使用自适应切换？** → 固定 $p_t=1$ 将导致后期性能下降（文中虽未展示，但理论支持此观点）
- **是否使用 RFF 压缩？** → CCBO vs CCBO+RFF：轻微性能折损换取隐私与效率

---

## **4. 关键结论和发现**

### **主要发现**
1. 🎯 **协作能显著提升 CBO 效率**：尤其是在数据稀缺初期，跨客户端信息共享可快速校准 context-space 探索方向。
2. 🔁 **“分歧驱动”策略优于随机协作**：主动识别本地信念不可靠的 context，比均匀协作更高效。
3. ⚖️ **自适应切换至关重要**：在异质环境中，必须动态减少协作频率以避免偏差累积。
4. 🔒 **RFF 可实现高效隐私保护协作**：压缩通信带来的性能损失极小，适合实际部署。
5. 🏭 **方法在真实制造场景中有效**：热轧实验验证了 CCBO 在复杂、非线性、带噪声的真实系统中的鲁棒性。

---

### **方法的局限性**
1. **假设上下文可控（Controllable Context）**：仅适用于 OCBO 场景，不能直接应用于外部给定 context 的在线场景。
2. **理论分析基于离散化假设**：Theorem 1 假设 context/design 空间有限，连续域下的严格 regret bound 仍具挑战。
3. **RFF 近似精度依赖基函数数量 $J$**：过小会导致信息丢失，过大则削弱通信优势。
4. **未考虑通信延迟或失败**：理想化网络假设，尚未适配极端边缘计算环境。

---

### **未来工作方向**
1. **扩展至不可控上下文场景**：发展适用于在线 CBO 的协作机制。
2. **更强的异质性建模**：引入个性化 GP kernels 或 meta-learning 结构增强适应能力。
3. **动态调整 $p_t$ 而非预设衰减**：基于不确定性或性能反馈自动调节协作强度。
4. **结合联邦学习中的公平性机制**：确保弱势客户端不被主导模型淹没。
5. **部署于真实工业闭环控制系统**：实现全自动、分布式的智能制造优化平台。

---

> 💡 **总体评价**：  
> 本论文填补了 **Collaborative CBO** 领域的空白，提出了一套兼具理论严谨性、工程实用性和隐私意识的完整解决方案。其实验充分、逻辑严密，是将 Bayesian Optimization 推向分布式智能系统的重要一步。

</details>

---

### 14. [SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving](https://arxiv.org/abs/2604.19157)

**Authors**: Jinda Jia, Jisen Li, Zhongzhu Zhou, Jung Hwan Heo, Jue Wang, Tri Dao, Shuaiwen Leon Song, Ben Athiwaratkun, Chenfeng Xu, Tianyi Zhang, Xiaoxia Wu  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19157v1  

#### Abstract
KV-cache memory is a major bottleneck in real-world LLM serving, where systems must simultaneously support latency-sensitive small-batch requests and high-throughput concurrent workloads. Although many KV-cache compression methods improve offline accuracy or compression ratio, they often violate pra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在现实世界的 **LLM serving** 场景中，**KV-cache 内存占用** 是主要瓶颈之一。随着模型支持的上下文长度增长至数百万甚至上千万 tokens（如 Llama 4 Scout），KV-cache 所需内存远超模型参数本身（例如 1.8 TiB vs 218 GB）。尽管已有大量 KV-cache 压缩研究（如量化、向量量化、低秩分解等），但这些方法往往忽略了实际部署系统的关键约束：

- **PagedAttention** 要求固定大小、统一类型的内存块；
- **FlashAttention** 等融合内核要求规则的内存访问模式；
- 生产系统对延迟极其敏感，额外计算或不规则访存会显著降低吞吐。

因此，许多离线表现优异的方法在真实服务栈中反而导致 **端到端性能下降**。

### 提出了什么新方法或新思路
本文提出了一种 **系统感知（system-aware）的 4-bit KV-cache 量化框架 SAW-INT4**，其核心是：
> **Token-wise INT4 量化 + Block-Diagonal Hadamard Rotation (BDR)**

具体设计包括：
- **Block-Diagonal Rotation (BDR)**：将完整的 Hadamard 变换分块为多个独立的小块变换，在保留去相关性和平滑 outlier 效果的同时大幅降低计算开销。
- **Fused Rotation-Quantization Kernel**：在 Triton 中实现旋转与量化的融合内核，直接集成进 Paged KV-cache 和 FlashAttention 流程，避免额外内存读写。
- **仅对 Key 应用旋转**：实验证明只对 Key 进行 BDR 即可获得几乎全部收益，进一步减少计算负担。

### 相比现有方法的优势
| 维度 | SAW-INT4 (BDR) | 其他方法（如 KIVI、Kitty、Vector Quantization） |
|------|----------------|---------------------------------------------|
| **系统兼容性** | ✅ 完全兼容 PagedAttention 和 FlashAttention | ❌ 多精度缓存、codebook 查找破坏内存布局和访存连续性 |
| **端到端性能** | ✅ 吞吐与 plain INT4 相当，显著优于 BF16 | ❌ 高复杂度带来可观测延迟增加 |
| **准确性恢复能力** | ✅ 几乎完全恢复 naive INT4 损失的精度 | ⚠️ 更复杂的 Hessian-aware 或 VQ 改进有限 |
| **实现复杂度** | ✅ 极简设计，无需训练或复杂校准 | ❌ 需要 codebook 训练、残差缓冲、通道级缩放等 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
论文在以下五个具有挑战性的推理与编码任务上进行评估：
- **GPQA-Diamond**：高难度科学问答，测试深层推理能力
- **HumanEval**：代码生成任务
- **LiveCodeBench (v6)**：真实场景下的编程问题
- **AIME25**：数学竞赛题（American Invitational Mathematics Examination）
- **MATH500**：复杂数学问题基准

所有实验均以 `thinking mode` 运行，强调长链推理。

### 实验设置和评估指标
#### 模型
- Qwen3-4B-Thinking-2507
- Qwen3-8B
- Qwen3-32B
- GLM-4.7-FP8 (358B)

#### 硬件
- 主要使用 2×H100 GPUs（tp=2）或 8×H100（tp=8）
- 使用 SGLang 作为 serving 引擎，启用 FA3 prefill 和 Triton decode

#### 评估指标
| 指标 | 描述 |
|------|------|
| **TPS/User (TPSreq)** | 每用户的输出 token 数 / 解码时间，反映客户端体验 |
| **TPS/GPU (TPSsys/NGPU)** | 每 GPU 总输出 token 数 / 墙钟时间，衡量系统效率 |
| **TTFT (Time to First Token)** | 请求提交到首个 token 输出的时间，反映响应性 |
| **Accuracy (Mean Score)** | 上述五项任务得分平均值（μ±σ over 5 runs） |

#### 序列长度
- 最大序列长度：32k
- 输入长度：~8,192 tokens
- 输出长度：~1,024 tokens

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **BF16** | 全精度 KV-cache，基准上限 |
| **INT4** | 简单 token-wise 4-bit 量化 |
| **KMeans (C=256/2048)** | 向量量化，不同 codebook 大小 |
| **Hessian+BDR** | 基于查询统计学习旋转矩阵 |
| **KIVI / Kitty** | 当前先进混合精度 KV 量化方案（用于外部比较） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Method | Qwen3-4B Acc. | Qwen3-8B Acc. | TPS/GPU (Qwen3-4B) | TPS/GPU (Qwen3-8B) |
|--------|---------------|--------------|--------------------|--------------------|
| BF16 | 75.64 | 70.84 | 1030.5 | 859.1 |
| INT4 | 0.00 | 0.00 | 1217.5 | 962.8 |
| KMeans (C=256) | 71.64 | 68.91 | 365.7 | 347.9 |
| Hessian+BDR | 65.52 | 70.59 | 1051.0 | 887.2 |
| **BDR (Ours)** | **73.78** | **69.86** | **1242.2** | **986.3** |

> ✅ **SAW-INT4 在保持接近 BF16 精度的同时，实现了比 BF16 高约 45% 的吞吐，并且与朴素 INT4 吞吐基本持平。**

### 与基线方法的对比结果
- **Naive INT4**：准确率崩溃至 0，说明无预处理的 4-bit 量化不可行。
- **KMeans VQ**：虽然能部分恢复精度（最高达 72.81），但因 codebook 查找引入不规则访存，**TPS 下降超过 60%**，得不偿失。
- **Hessian-aware Quantization**：依赖离线校准，精度仍低于 BDR（73.11 vs 65.52），且增加部署复杂性。
- **BDR 显著胜出**：在 Qwen3-4B 上达到 73.78 分（距 BF16 仅差 1.86），同时维持最高吞吐。

### 消融实验结果（Table 2 & Table 8）

#### 不同 BDR 块大小的影响（Qwen3-4B）
| BDR Block Size | Mean Accuracy | Drop from BF16 |
|----------------|---------------|----------------|
| 16 | 54.83 | -20.81 |
| 64 | 72.29 | -3.35 |
| 128 | 73.11 | -2.53 |
| 128 (Key-only) | 73.78 | **-1.86** |

> 🔍 **Block size ≥64 即可显著提升效果，128 达到饱和；Key-only 旋转即可取得最佳性价比。**

#### 是否需要更复杂方法？（组合实验）
| 方法 | Qwen3-4B Mean |
|------|----------------|
| BDR-128 | 73.11 |
| KM (C=2048) + BDR | 73.35 |
| Hessian + BDR | 65.52 |

> 📉 **加入 KMeans 或 Hessian 并未带来实质性增益，反而增加复杂度。BDR 已经捕获了主要误差来源。**

#### 不同模型上的泛化性（Table 3）
- **Qwen3-8B**：BDR 将 accuracy 从 0 提升至 69.97（vs BF16 的 70.84）
- **GLM-4.7**：即使 naive INT4 表现良好（77.21 vs 77.89），BDR 仍可进一步微调至 77.95

> ✅ **BDR 对敏感模型（如 Qwen3）至关重要，对鲁棒模型也无负面影响。**

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **最有效的 KV-cache 压缩不是最复杂的，而是最系统友好的**：  
   在真实 serving 条件下，**token-wise INT4 + BDR** 是唯一能在精度、吞吐、兼容性之间取得平衡的设计。

2. ✅ **Block-Diagonal Hadamard Rotation 是关键使能技术**：  
   它有效缓解 channel-wise outliers 导致的量化误差，且可通过分块控制计算成本。

3. ✅ **融合内核设计消除了旋转开销**：  
   尽管单独看 fused rotation 内核稍慢，但在端到端系统中，其开销被掩盖，**TPS 与 plain INT4 一致**。

4. ✅ **更复杂的量化方法收益递减**：  
   Vector Quantization、Hessian-aware Calibration 等带来的精度提升非常有限，不足以抵消其实现和运行时成本。

5. ✅ **系统指标比局部指标更重要**：  
   微基准（microbenchmark）中的带宽优势不能代表真实吞吐；**TPSsys** 才是衡量 serving 效率的正确指标。

### 方法的局限性
- **依赖 Hadamard 结构**：虽然高效，但仍是固定变换，无法像可学习旋转那样自适应每层分布。
- **对极端 outlier 模型可能不足**：若某些模型 KV 激活极度稀疏或重尾，可能仍需混合精度策略。
- **目前仅支持 INT4**：未探索 2-bit 或非均匀量化空间。

### 未来工作方向
- 探索 **learned lightweight rotations**（如在 Stiefel 流形上优化小规模正交矩阵）以进一步逼近理论极限。
- 将 BDR 思想扩展至 **weight-only quantization** 或 **activation quantization** 中。
- 结合 **Grouped-Query Attention (GQA)** 和 **Multi-Head Latent Attention (MLA)** 架构进一步压缩。
- 开发自动选择 block size 的机制，根据模型敏感度动态调整。

---

> 💡 **一句话总结**：  
> 在真实的 LLM serving 系统中，**最好的 KV-cache 量化方法是最简单的——Token-wise INT4 加上一个轻量级的 Block-Diagonal Hadamard Rotation**，它在几乎零代价的情况下恢复了几乎全部精度损失，证明了“系统协同设计”比“算法极致复杂”更重要。

</details>

---

### 15. [Accelerating Optimization and Machine Learning through Decentralization](https://arxiv.org/abs/2604.19518)

**Authors**: Ziqin Chen, Zuang Wang, Yongqiang Wang  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19518v1  

#### Abstract
Decentralized optimization enables multiple devices to learn a global machine learning model while each individual device only has access to its local dataset. By avoiding the need for training data to leave individual users' devices, it enhances privacy and scalability compared to conventional cent...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerating Optimization and Machine Learning through Decentralization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
传统观点认为，**decentralized optimization**（去中心化优化）是一种在通信受限或隐私要求下不得已而为之的妥协方案，其收敛速度通常慢于或最多与**centralized optimization**（集中式优化）持平。  
本文挑战了这一长期共识，提出并证明：**在适当设计下，decentralization 不仅可行，反而可以加速优化过程**。

### ✅ 提出了什么新方法或新思路
作者提出了 **Algorithm 1** ——一种基于服务器辅助的去中心化梯度下降算法，其核心创新在于：
- **本地自适应步长（local step size）**：每个设备根据其本地数据的 **smoothness constant $L_i$** 设置独立的步长 $\alpha_i = 1/L_i$，从而利用局部几何结构加快早期收敛。
- **动态切换机制（switching mechanism）**：当收敛趋于平缓时，切换到统一全局步长 $\alpha = 1/L$（其中 $L = \frac{1}{N}\sum L_i$），以消除因异构步长导致的稳态误差（steady-state error），确保最终收敛到精确最优解。

该方法结合了去中心化的灵活性与集中式方法的收敛保证。

### ✅ 相比现有方法的优势
| 对比维度 | 传统去中心化方法 | 本文 Algorithm 1 |
|--------|------------------|------------------|
| 收敛速度 | 通常更慢或相当 | **显著更快**（迭代次数减少） |
| 步长策略 | 统一或固定 | **异构 + 动态切换** |
| 收敛性保证 | 可能存在稳态误差 | **可证收敛至全局最优** |
| 性能分析工具 | 渐近上界分析（保守） | **PEP 框架下的精确最坏情况分析** |

> 💡 **核心洞见**：数据分布越异构（non-IID），本地 $L_i$ 差异越大，Algorithm 1 的加速效果越明显。

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集
论文在多个标准 benchmark 数据集上验证方法有效性：

| 数据集 | 任务类型 | 规模 | 分布划分方式 |
|-------|--------|------|-------------|
| **W8A** | Logistic Regression 分类 | ~50k 样本 | 按 label / feature norm / eigenvalue 划分 |
| **CIFAR-10** | 图像分类（CNN） | 60k 彩色图像 | 按 label 分配给 10 个设备 |
| **SST-2** | 文本情感分析（BERT） | ~67k 句子 | Dirichlet 分布 ($\alpha=0.1$) 生成高度非 IID 数据 |
| **MNIST**（附录） | 手写数字识别（CNN） | 60k 黑白图像 | 多种 Dirichlet 参数控制异构程度 |

### ✅ 实验设置和评估指标
- **优化器**：使用梯度下降（GD）及其变体，不引入动量等额外机制以保持公平比较。
- **计算对等性假设**：中央服务器的总算力等于所有分布式设备之和（即每轮处理相同数量样本）。
- **评估指标**：
  - 训练损失（Training Loss）
  - 训练准确率（Training Accuracy）
  - 测试/验证准确率（Test/Validation Accuracy）
  - 达到目标精度所需的 **迭代次数 / epoch 数**

### ✅ 基线方法对比
- **Centralized GD**：标准集中式梯度下降，作为主要 baseline。
- **Decentralized GD (DGD)**：传统去中心化梯度法，使用统一或本地步长。
- **Gradient Tracking** [51]
- **FedOpt variants**：包括 FedAdam, FedYogi, FedAdagrad [52]
- **FedAvgM** [53]

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据与对比结果

#### 🔹 W8A 数据集（图2）
- 在三种异构划分下（by labels, norms, eigenvalues），**Algorithm 1 均显著快于 Centralized GD**。
- 例如，在“by labels”划分中，达到相同训练损失所需迭代数减少约 **30–50%**。
- 局部 smoothness constants 明显不同（如 $L_1=0.4$, $L_2=1.6$），验证了异构性是加速来源。

#### 🔹 CIFAR-10（图3）
- 使用 mini-batch（每轮共处理10个样本）进行公平比较。
- **Algorithm 1 在训练损失下降、训练/测试准确率提升方面均领先于 Centralized GD**。
- 即使 smoothness constant 估计有波动（多次实验），加速效果依然稳定。

#### 🔹 SST-2（图4）
- 使用 full-batch fine-tuning BERT-base 模型。
- **Algorithm 1 迅速收敛至更高准确率，且训练损失下降更快**。
- 验证了该方法在 NLP 和 Transformer 架构上的适用性。

#### 🔹 MNIST（附录图5–10）
- **消融实验 B.2（图6）**：Dirichlet 参数 $\alpha$ 越小（数据越异构），加速越明显 → 表明 **异构性驱动加速**。
- **网络规模实验 B.3（图7）**：随着设备数从 2 增加到 25，Algorithm 1 的优势持续增强。
- **与其他去中心化方法对比（图8–9）**：
  - 超出 **Decentralized GD** 和 **Gradient Tracking**
  - 超出 **FedAdam/FedYogi/FedAdagrad/FedAvgM** 等主流 Federated Learning 方法

### ✅ 消融实验结果
| 实验 | 发现 |
|-----|------|
| **B.2：不同 $\alpha$ 的 Dirichlet 划分** | $\alpha \downarrow$ → 异构性 ↑ → 加速效果 ↑ |
| **B.3：不同设备数量 $N$** | $N \uparrow$ → 加速效果增强（尤其在高异构场景） |
| **B.6：部分参与（Partial Participation）** | 即使参与概率低至 70%，Algorithm 1 仍保持鲁棒性和加速能力 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **去中心化不仅是隐私保护手段，更是性能加速器**：
   - 合理利用本地数据的几何特性（如不同的 $L_i$），可在早期实现比集中式更快的收敛。
   
2. **异构性（Heterogeneity）不是障碍，而是资源**：
   - 数据分布越非 IID，Algorithm 1 的加速潜力越大。

3. **动态步长切换机制至关重要**：
   - 仅用本地步长会导致稳态误差；加入后期切换可兼顾加速与收敛性。

4. **理论与实证一致支持加速现象**：
   - 基于 **Performance Estimation Problem (PEP)** 的严格数学分析表明，在 worst-case 下 Algorithm 1 的优化误差严格小于 Centralized GD（见图1b，ratio < 1）。

### ⚠️ 方法的局限性
- **未考虑通信开销**：当前分析假设通信成本为零或可忽略，但在实际系统中可能成为瓶颈。
- **smoothness constant 估计依赖采样**：对于复杂模型（如神经网络），$L_i$ 需通过数值方法估计，存在一定误差。
- **目前聚焦 first-order methods**：未扩展至 Adam、Newton-type 等高阶优化器。

### 🔮 未来工作方向
- 将 PEP 框架扩展至含通信延迟和压缩的去中心化设定。
- 探索在异步、动态拓扑网络中的应用。
- 结合 adaptive optimizers（如 Adam）设计去中心化加速版本。
- 在更大规模真实系统（如边缘计算集群）中部署验证。

---

> 📌 **一句话总结**：  
> 本文颠覆了“去中心化必然牺牲效率”的传统认知，提出了一种能**主动利用数据异构性来加速学习**的新范式，揭示了 **decentralization 本身即可作为一种战略性的优化加速手段**。

</details>

---

### 16. [HardNet++: Nonlinear Constraint Enforcement in Neural Networks](https://arxiv.org/abs/2604.19669)

**Authors**: Andrea Goertzen, Kaveh Alim, Navid Azizan  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19669v1  

#### Abstract
Enforcing constraint satisfaction in neural network outputs is critical for safety, reliability, and physical fidelity in many control and decision-making applications. While soft-constrained methods penalize constraint violations during training, they do not guarantee constraint adherence during in...

---

### 17. [DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](https://arxiv.org/abs/2604.19351)

**Authors**: Jinyu Guo, Zhihan Zhang, Yutong Li, Jiehui Xie, Md. Tamim Iqbal, Dongshen Han, Lik-Hang Lee, Sung-Ho Bae, Jie Zou, Yang Yang, Chaoning Zhang  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.19351v1  

#### Abstract
The quadratic computational complexity of the standard attention mechanism constitutes a fundamental bottleneck for large language models in long-context inference. While existing KV cache compression methods alleviate memory pressure, they often sacrifice generation quality and fail to address the ...

---

### 18. [Ocean: Fast Estimation-Based Sparse General Matrix-Matrix Multiplication on GPU](https://arxiv.org/abs/2604.19004)

**Authors**: Yifan Li, Giulia Guidi  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.19004v1  

#### Abstract
In computational science and data analytics, many workloads involve irregular and sparse computations that are inherently difficult to optimize for modern hardware. A key kernel is Sparse General Matrix-Matrix Multiplication (SpGEMM), which underpins simulations, graph analytics, and machine learnin...

---

### 19. [OLLM: Options-based Large Language Models](https://arxiv.org/abs/2604.19087)

**Authors**: Shashank Sharma, Janina Hoffmann, Vinay Namboodiri  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19087v1  

#### Abstract
We introduce Options LLM (OLLM), a simple, general method that replaces the single next-token prediction of standard LLMs with a \textit{set of learned options} for the next token, indexed by a discrete latent variable. Instead of relying on temperature or sampling heuristics to induce diversity, OL...

---

### 20. [Towards Energy Impact on AI-Powered 6G IoT Networks: Centralized vs. Decentralized](https://arxiv.org/abs/2604.19377)

**Authors**: Anjie Qiu, Donglin Wang, Sanket Partani, Andreas Weinand, Hans D. Schotten  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19377v1  

#### Abstract
The emergence of sixth-generation (6G) technologies has introduced new challenges and opportunities for machine learning (ML) applications in Internet of Things (IoT) networks, particularly concerning energy efficiency. As model training and data transmission contribute significantly to energy consu...

---

### 21. [Multi-modal Reasoning with LLMs for Visual Semantic Arithmetic](https://arxiv.org/abs/2604.19567)

**Authors**: Chuou Xu, Liya Ji, Qifeng Chen  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19567v1  

#### Abstract
Reinforcement learning (RL) as post-training is crucial for enhancing the reasoning ability of large language models (LLMs) in coding and math. However, their capacity for visual semantic arithmetic, inferring relationships from images, remains underexplored. The classic text analogy "king"-"man"+"w...

---

### 22. [Detoxification for LLM: From Dataset Itself](https://arxiv.org/abs/2604.19124)

**Authors**: Wei Shao, Yihang Wang, Gaoyu Zhu, Ziqiang Cheng, Lei Yu, Jiafeng Guo, Xueqi Cheng  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19124v1  

#### Abstract
Existing detoxification methods for large language models mainly focus on post-training stage or inference time, while few tackle the source of toxicity, namely, the dataset itself. Such training-based or controllable decoding approaches cannot completely suppress the model's inherent toxicity, wher...

---

### 23. [A Simple Communication Scheme for Distributed Fast Multipole Methods](https://arxiv.org/abs/2604.19243)

**Authors**: Srinath Kailasa  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19243v1  

#### Abstract
We present a simple hierarchical communication scheme for distributed Fast Multipole Methods (FMMs) based on MPI neighborhood collectives and uniform trees. The method targets the common case of extending an existing high-performance shared-memory uniform-tree FMM implementation to distributed memor...

---

### 24. [CROWDio: A Practical Mobile Crowd Computing Framework with Developer-Oriented Design, Adaptive Scheduling, and Fault Resilience](https://arxiv.org/abs/2604.19363)

**Authors**: Lakshani Manamperi, Disumi Pathirana, Thiwanka Pathirana, Nipun Premarathna, Kutila Gunasekara  
**Category**: cs.DC  
**Published**: 2026-04-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19363v1  

#### Abstract
Mobile Crowd Computing (MCdC) leverages the idle computational capacity of consumer smartphones to enable distributed task processing at scale; however, widespread real-world adoption remains constrained by the absence of developer-oriented frameworks capable of transparently managing device heterog...

---

### 25. [FG$^2$-GDN: Enhancing Long-Context Gated Delta Networks with Doubly Fine-Grained Control](https://arxiv.org/abs/2604.19021)

**Authors**: Pingwei Sun, Yuxuan Hu, Jianchao Tan, Xue Wang, Jiaqi Zhang, Yifan Lu, Yerui Sun, Yuchen Xie, Xunliang Cai  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19021v1  

#### Abstract
Linear attention mechanisms have emerged as promising alternatives to softmax attention, offering linear-time complexity during inference. Recent advances such as Gated DeltaNet (GDN) and Kimi Delta Attention (KDA) have demonstrated that the delta rule, an online gradient descent update, enables sup...

---

### 26. [LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying Circuit](https://arxiv.org/abs/2604.19117)

**Authors**: Manav Pandey  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19117v1  

#### Abstract
When a language model agrees with a user's false belief, is it failing to detect the error, or noticing and agreeing anyway? We show the latter. Across twelve open-weight models from five labs, spanning small to frontier scale, the same small set of attention heads carries a "this statement is wrong...

---

### 27. [GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models](https://arxiv.org/abs/2604.19398)

**Authors**: Ziyang Wang, Jiangfeng Xiao, Chuan Xiao, Ruoxiang Li, Rui Mao, Jianbin Qin  
**Category**: cs.AI  
**Published**: 2026-04-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.19398v1  

#### Abstract
Large language models (LLMs) are expensive to serve because model parameters, attention computation, and KV caches impose substantial memory and latency costs. We present GRASPrune, a structured pruning framework applied after pretraining that jointly prunes FFN channels and KV head groups under a s...

---

### 28. [STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](https://arxiv.org/abs/2604.18976)

**Authors**: MinJae Jung, YongTaek Lim, Chaeyun Kim, Junghwan Kim, Kihyun Kim, Minwoo Kim  
**Category**: cs.CL  
**Published**: 2026-04-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.18976v1  

#### Abstract
While Large Language Models (LLMs) are widely used, they remain susceptible to jailbreak prompts that can elicit harmful or inappropriate responses. This paper introduces STAR-Teaming, a novel black-box framework for automated red teaming that effectively generates such prompts. STAR-Teaming integra...

---

### 29. [Learning Posterior Predictive Distributions for Node Classification from Synthetic Graph Priors](https://arxiv.org/abs/2604.19028)

**Authors**: Jeongwhan Choi, Jongwoo Kim, Woosung Kang, Noseong Park  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.19028v1  

#### Abstract
One of the most challenging problems in graph machine learning is generalizing across graphs with diverse properties. Graph neural networks (GNNs) face a fundamental limitation: they require separate training for each new graph, preventing universal generalization across diverse graph datasets. A cr...

---

### 30. [Calibrating Scientific Foundation Models with Inference-Time Stochastic Attention](https://arxiv.org/abs/2604.19530)

**Authors**: Akash Yadav, Taiwo A. Adebiyi, Ruda Zhang  
**Category**: cs.LG  
**Published**: 2026-04-22  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.19530v1  

#### Abstract
Transformer-based scientific foundation models are increasingly deployed in high-stakes settings, but current architectures give deterministic outputs and provide limited support for calibrated predictive uncertainty. We propose Stochastic Attention, a lightweight inference-time modification that ra...

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
