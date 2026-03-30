# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-30 07:14:33 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [ParaQAOA: Efficient Parallel Divide-and-Conquer QAOA for Large-Scale Max-Cut Problems Beyond 10,000 Vertices](https://arxiv.org/abs/2603.26232)

**Authors**: Po-Hsuan Huang, Xie-Ru Li, Chi Chuang, Chia-Heng Tu, Shih-Hao Hung  
**Category**: cs.DC  
**Published**: 2026-03-30  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.26232v1  

#### Abstract
Quantum Approximate Optimization Algorithm (QAOA) has emerged as a promising solution for combinatorial optimization problems using a hybrid quantum-classical framework. Among combinatorial optimization problems, the Maximum Cut (Max-Cut) problem is particularly important due to its broad applicabil...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ParaQAOA: Efficient Parallel Divide-and-Conquer QAOA for Large-Scale Max-Cut Problems Beyond 10,000 Vertices

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **QAOA**（Quantum Approximate Optimization Algorithm）在解决大规模 **Max-Cut** 问题时面临严重瓶颈：  
- 随着图规模增大，量子电路深度和计算复杂度急剧上升；
- 现有的 divide-and-conquer 方法虽然能降低子问题规模，但通常以牺牲执行效率为代价，导致总运行时间过长；
- 缺乏对 **solution quality**（解质量）与 **execution efficiency**（执行效率）之间权衡的有效控制机制。

这些问题使得传统方法难以应用于超过数千顶点的大规模 Max-Cut 实例，尤其是在有严格时间约束的实际场景中。

### 提出的新方法或新思路
作者提出 **ParaQAOA** —— 一种基于并行化、分治策略的 QAOA 框架，其核心创新包括：

1. **线性时间图划分算法（Connectivity-Preserving Partitioning）**  
   - 将大图划分为多个大小均衡的子图，每个相邻子图共享一个节点，确保连通性；
   - 时间复杂度为 $O(|V| + |E|)$，显著优于传统二次及以上复杂度的划分方法。

2. **完全并行化的执行流水线（Parallelized QAOA Execution Pipeline）**  
   - 利用多 GPU 架构，并行执行各个子图上的 QAOA 求解；
   - 引入 **Selective Distribution Exploration Strategy**，保留 Top-K 高概率 bitstrings，平衡资源消耗与解质量。

3. **层级感知的并行合并机制（Level-Aware Parallel Merge）**  
   - 在合并阶段采用并行深度优先遍历 Cartesian product 空间；
   - 支持从任意层级开始合并（参数 `L`），提升并行粒度，充分利用多核 CPU 资源。

4. **统一性能评估指标：Performance Efficiency Index (PEI)**  
   - 定义为：$ \text{PEI} = \text{AR} \times \text{EF} \times 100 $
     - **AR**（Approximation Ratio）衡量解质量；
     - **EF**（Efficiency Factor）使用 sigmoid 函数归一化运行时间，反映相对加速比；
   - 实现了解质量与效率的联合量化比较。

### 相比现有方法的优势
| 维度 | ParaQAOA | 现有方法（如 DC-QAOA、QAOA2、Coupling QAOA） |
|------|---------|---------------------------------------------|
| 可扩展性 | ✅ 支持超万级顶点（16,000+） | ❌ 多数仅支持数百顶点 |
| 执行效率 | ⚡ 最高达 **1,600× 加速** | ⏳ 运行时间长达数小时甚至数天 |
| 解质量保持 | ✅ 逼近最优解（AR 损失 < 2%） | ✅ 注重精度但牺牲速度 |
| 并行能力 | ✅ 全流程并行设计 | ❌ 多为串行或局部并行 |
| 参数可控性 | ✅ 用户可调 K 和 L 控制质量-效率权衡 | ❌ 固定配置，灵活性差 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Erdős–Rényi 随机图**（`gen_erdos_renyi_graph` from NetworkX）
- 图规模覆盖：
  - 小规模：20–30 个顶点
  - 中等规模：100–400 个顶点
  - 大规模：1,000–16,000 个顶点
- 边密度（edge probability）：0.1, 0.3, 0.5, 0.8
- 每种配置生成 10 个实例，使用不同随机种子保证可复现性。

### 实验设置
- **硬件平台**：
  - CPU: AMD Ryzen Threadripper 7960X (24-core)
  - GPU: 2× NVIDIA RTX 4090 (24GB GDDR6X each)
  - 内存: 256GB DDR5
  - CUDA 12.5, Python 3.12
- **软件依赖**：
  - 使用 `numba` 加速 QAOA GPU 内核（基于 Lu et al. [31]）
  - NetworkX 用于图生成

### 评估指标
| 指标 | 描述 |
|------|------|
| **Approximation Ratio (AR)** | $ \text{AR} = \frac{\text{CutVal}_{\text{ALG}}}{\text{CutVal}_{\text{OPT}}} $，越高越好 |
| **Execution Time** | 端到端求解时间（秒） |
| **Speedup** | 相对于基线方法的加速比 |
| **Performance Efficiency Index (PEI)** | 综合 AR 与 EF 的单一评分指标 |

### 基线方法对比
| 方法 | 简介 |
|------|------|
| **Goemans-Williamson (GW)** | 经典近似算法，理论下界 AR ≥ 0.878 |
| **QAOA2** [46] | 分层嵌套 QAOA，适用于小量子设备，但运行慢 |
| **Coupling QAOA (CQ)** [31] | 通过耦合项连接子图，限制于二分图且无法扩展至大规模 |
| **DC-QAOA** [29] | 使用复杂图划分，预处理开销大 |

> ⚠️ 注意：由于 CQ 和 DC-QAOA 不适用于大规模图，仅在小图上进行对比；中大规模实验主要对比 QAOA2。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 中等规模图（400 顶点）
| 方法 | 平均运行时间 | 最高速度提升 | AR（相对于 GW） |
|------|--------------|---------------|------------------|
| **ParaQAOA** | **8.7–10.3 秒** | — | 94.0%–97.8% |
| **QAOA2** | 2,158.8–17,001.0 秒 | — | 95.5%–98.4% |
| **Speedup** | — | **最高达 1,652.2×** | 差距 < 2% |

> 🔥 在边密度为 0.8 的 400-vertex 图上实现 **1,652× 加速**，同时 AR 仅下降约 1.4%。

#### ✅ 大规模图（16,000 顶点）
| 方法 | 预计运行时间 |
|------|----------------|
| **ParaQAOA** | **19 分钟** |
| **QAOA2**（外推） | **>13.6 天**（约 327 小时） |
| **Speedup** | **≈1,000× 以上**

> 💡 ParaQAOA 是首个能在 **分钟级别** 解决 **16,000 顶点 Max-Cut** 问题的 QAOA 框架。

### 与基线方法的对比结果

| 对比维度 | 结果摘要 |
|--------|----------|
| **运行时间** | ParaQAOA 比 QAOA2 快 **300× ~ 2,000×**，且随图规模增长优势更明显 |
| **AR 表现** | 在稀疏图（p=0.1）上 AR 略低（~87% vs ~94%），但在稠密图上接近甚至反超 |
| **可扩展性** | QAOA2 在 4,000 顶点后已需数天，而 ParaQAOA 仍可在 20 分钟内完成 16k 顶点任务 |
| **资源利用率** | 多 GPU + 多 CPU 核心实现近线性加速，负载均衡良好 |

### 消融实验结果（Ablation Study）

#### 参数 `K`（Top-K bitstrings 数量）的影响（Fig. 9）
- `K=1`: 运行最快，AR 略低；
- `K≥3`: AR 显著提升，但边际收益递减；
- **建议值：K=2**，在精度与效率间取得最佳平衡。

#### 参数 `L`（合并起始层级）的影响（Fig. 10）
- `L=1`: 启动 4 个进程；
- `L=2`: 启动 8 个进程；
- `L=3`: 启动 16 个进程；
- **结果**：每增加一级，运行时间约减少一半，体现良好的并行扩展性。

> 推荐设置：`2^K ≈ 物理 CPU 核心数`，以最大化硬件利用率。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **ParaQAOA 成功实现了大规模 Max-Cut 问题的高效求解**，首次将 QAOA 应用于 **超过 10,000 顶点** 的实例；
2. ✅ 通过 **线性时间图划分 + 并行 QAOA 执行 + 层级合并**，实现了数量级的速度提升；
3. ✅ 引入 **PEI 指标**，为未来优化算法提供了一个标准化的质量-效率综合评价体系；
4. ✅ 用户可通过调节 `K` 和 `L` 灵活控制 **accuracy-efficiency trade-off**，适应不同应用场景需求；
5. ✅ 实验证明，在大多数情况下，ParaQAOA 的 AR 损失小于 2%，却带来数百至上千倍的加速。

### 方法的局限性
1. **图划分方式较简单**：当前使用的是基于顺序索引的均匀划分，未考虑图结构特征（如社区结构），可能影响稀疏图上的表现；
2. **依赖经典模拟器**：目前所有 QAOA 计算均在 GPU 上模拟，尚未部署到真实量子硬件；
3. **不适用于高度非均匀图**：对于具有极端度分布或模块化的图，固定大小划分可能导致负载不均；
4. **内存占用随 K 指数增长**：当 `K` 过大时，合并阶段的候选空间爆炸（$2^{KM}$），限制了极端高精度场景的应用。

### 未来工作方向
1. **自适应图划分（Adaptive Partitioning）**：结合图聚类算法（如 METIS）进行更智能的子图分割；
2. **拓展至其他 QUBO 问题**：如 TSP、Graph Coloring、Set Cover 等；
3. **集成噪声模型与纠错机制**：面向 NISQ 设备的真实部署；
4. **异构硬件优化**：支持 CPU/GPU/FPGA 协同计算；
5. **动态参数调整策略**：根据中间结果自动调节 `K` 和 `L`，实现智能化运行时优化。

---

> 📌 **一句话总结**：  
> **ParaQAOA 是首个实现“万级顶点 Max-Cut 分钟级求解”的并行 QAOA 框架，通过全流程并行化设计，在几乎不损失解质量的前提下，达成高达 1,600× 的加速，为大规模组合优化问题提供了实用化的量子启发解决方案。**

</details>

---

### 2. [Optimization Trade-offs in Asynchronous Federated Learning: A Stochastic Networks Approach](https://arxiv.org/abs/2603.26231)

**Authors**: Abdelkrim Alahyane (LAAS-SARA), C\'eline Comte (CNRS, LAAS-SARA), Matthieu Jonckheere (CNRS, LAAS-SARA)  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.26231v1  

#### Abstract
Synchronous federated learning scales poorly due to the straggler effect. Asynchronous algorithms increase the update throughput by processing updates upon arrival, but they introduce two fundamental challenges: gradient staleness, which degrades convergence, and bias toward faster clients under het...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimization Trade-offs in Asynchronous Federated Learning: A Stochastic Networks Approach

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Asynchronous Federated Learning (AsyncFL)** 中存在的两个根本性挑战：
1.  **Gradient Staleness (梯度陈旧性)**：由于客户端计算和通信时间的异步性，服务器应用的梯度可能基于过时的全局模型，导致收敛速度变慢甚至不稳定。
2.  **Bias Toward Faster Clients (对快速客户端的偏见)**：在非独立同分布（Non-IID）数据下，计算速度快的客户端会提交更多更新，导致学习过程偏向其本地数据分布。

现有研究大多在 **communication rounds** 层面分析收敛性，忽略了真实的 **wall-clock time** 性能。此外，许多分析忽略了底层的排队动态（queueing dynamics），无法提供闭式表达来精确刻画吞吐量（throughput）和梯度陈旧性之间的权衡。

### 提出的新方法和新思路
作者提出了一种全新的 **stochastic queueing-network framework** 来建模 **Generalized AsyncSGD** 算法。其核心创新在于：
- **统一的随机网络模型**：将整个 AsyncFL 系统（包括客户端的计算、上行/下行通信延迟、中央服务器处理以及客户端侧队列）建模为一个 **closed Jackson network**。
- **闭式性能分析**：利用 **product-form network theory** 和 **Buzen's recursive algorithm**，首次推导出了以下关键性能指标的**闭式表达式**：
  - 平均相对延迟 `E[D]`（衡量梯度陈旧性）
  - 更新吞吐量 `λ(p, m)`（衡量更新频率）
  - 达到 `ε`-stationary point 所需的通信轮数 `K_c(p, m)`
  - 所需的期望 **wall-clock time** `E[T]`
  - 所需的期望 **energy consumption** `E[E]`
- **联合优化框架**：基于上述闭式表达式，提出了一个可微的、基于梯度的优化策略，可以同时优化 **routing probabilities (p)** 和 **concurrency level (m)**，以在准确性、训练时间和能耗之间进行权衡。

### 相比现有方法的优势
- **更真实、更全面**：显式地建模了计算、通信和服务器处理的随机性，而不仅仅是假设固定或有界延迟。
- **可量化、可优化**：提供了精确的闭式公式，使得性能分析不再是黑箱，而是可以通过梯度下降等方法直接优化系统参数。
- **揭示了根本权衡**：形式化地揭示了 `gradient staleness` 与 `wall-clock convergence speed` 之间的根本权衡，并进一步发现了 `convergence speed` 与 `energy efficiency` 之间的额外权衡。
- **超越单一目标**：提出的联合优化框架可以导航多目标帕累托前沿（Pareto frontier），而不仅仅是优化单一指标。

## 2. 核心实验方法和设置

### 数据集
- **EMNIST**：用于主实验，模拟手写字符识别任务。
- **KMNIST**：用于验证能量效率的实验。
- **CIFAR-100**：用于额外的验证实验，以证明方法的泛化能力。

### 实验设置
- **客户端环境**：模拟了一个由 **100个异构客户端** 组成的网络，分为5种类型（A-E），代表从高性能工作站到资源受限的“straggler”设备。
- **服务速率**：每种类型的客户端具有不同的计算（`μ^c`）、上行（`μ^u`）和下行（`μ^d`）服务速率。
- **数据分布**：
  - **Homogeneous (IID)**：数据均匀分布在所有客户端上。
  - **Heterogeneous (Non-IID)**：使用 **Dirichlet 分布**（浓度参数 `α=0.2`）模拟特征和标签的异质性。
- **时间分布**：为了验证鲁棒性，计算和通信时间分别被设定为 **指数分布 (Exponential)**、**确定性分布 (Deterministic)** 和 **对数正态分布 (Lognormal)**。

### 评估指标
- **主要指标**：达到特定测试准确率（如 0.6 或 0.75）所需的 **wall-clock time**。
- **次要指标**：达到目标精度所需的 **energy consumption**。
- **消融指标**：通信轮数 `K_c`、更新吞吐量 `λ`、平均相对延迟 `E[D]`。

### 基线方法对比
1.  **Standard Baseline (AsyncSGD)**：经典的 AsyncSGD，采用均匀路由 (`p_i = 1/n`) 和全并发 (`m=n`)。
2.  **Round-Optimized Generalized AsyncSGD**：优化路由 `p` 以最小化通信轮数 `K_c`，保持 `m=n`。
3.  **Max-Throughput Generalized AsyncSGD**：优化路由 `p` 以最大化更新吞吐量 `λ`，保持 `m=n`。
4.  **Time-Optimized Generalized AsyncSGD (Proposed)**：本文提出的方法，联合优化 `p` 和 `m` 以最小化期望 `E[T]`。
5.  **Joint Time-Energy Co-Optimized**：通过调整权衡参数 `ρ`，在 `E[T]` 和 `E[E]` 之间进行联合优化。

## 3. 主要实验结果和性能指标

### 关键性能数据
- **最优并发度**：在 EMNIST 实验中，联合优化得到的最优并发度 `m*T = 91`，小于总客户端数 `n=100`，这挑战了文献中通常假设 `m=n` 的惯例。
- **最优路由**：提出的路由策略 `p*T` 在保证快速客户端（如 Type E）高更新频率的同时，也为慢速客户端（如 Type D）保留了不可忽略的选择概率，实现了平衡。

### 与基线方法的对比结果
- **相对于 AsyncSGD (Baseline)**：
  - **收敛时间**：在多种设置下，**减少了 29% 到 46%** 的 wall-clock time。
  - **能量消耗**：在多种设置下，**减少了 36% 到 49%** 的能量消耗。
- **相对于 Max-Throughput**：
  - 虽然最大吞吐量策略的更新次数最多（超过 60,000 次），但其最终精度极低（在某些设置下比优化方法低 60%），且学习过程极不稳定。
- **相对于 Round-Optimized**：
  - 尽管轮次优化策略的通信轮数最少，但其更新频率极低（仅约 1800 次更新），导致 wall-clock time 远长于本文提出的时间优化方法。

### 消融实验结果
- **并发度 `m` 的影响**：实验表明，`E[T]` 与 `m` 之间存在一个明显的 U 形关系。当 `m` 太小时，系统利用率不足；当 `m` 太大时，梯度陈旧性急剧增加，抵消了并行化的收益。这证实了存在一个最优的 `m` 值。
- **路由策略的影响**：实验验证了极端策略（只追求吞吐量或只减少轮数）都会导致整体 wall-clock 性能下降，凸显了联合优化的必要性。
- **分布鲁棒性**：所提方法在指数、确定性和对数正态三种时间分布下都表现出显著优势，证明了其鲁棒性超出了理论假设。

## 4. 关键结论和发现

### 主要发现
1.  **根本权衡的存在**：在 AsyncFL 中，**降低梯度陈旧性**（提高更新质量）与**提高更新频率**（提高吞吐量）是相互冲突的目标。单纯优化其中一个会损害另一个，从而恶化真实的 wall-clock 时间性能。
2.  **并发度 `m` 是关键参数**：并发度 `m` 不应简单地设为客户端总数 `n`。存在一个最优的 `m` 值，可以在并行效率和梯度陈旧性之间取得最佳平衡。
3.  **联合优化至关重要**：通过将 `routing` 和 `concurrency` 作为可优化变量，可以系统性地导航性能帕累托前沿，实现比任何单一目标优化方法更好的综合性能。
4.  **能量-延迟权衡**：除了时间-准确性权衡，还存在一个 `energy efficiency` 与 `training latency` 之间的权衡。最小化能耗需要串行执行 (`m=1`)，但这会导致训练时间过长。

### 方法的局限性
- **模型假设**：理论分析依赖于 **指数分布的服务时间** 和 **Jackson 网络** 的产品形式解。尽管实验显示在其他分布下也有效，但严格的理论保证仍基于这些假设。
- **静态环境**：模型假设客户端集合、服务速率和网络条件是静态的。对于动态加入/退出的客户端，模型需要扩展。
- **中心服务器瓶颈**：虽然论文在第7节扩展了模型以包含服务器端缓冲区，但其分析仍然是基于理想化的排队模型。

### 未来工作方向
- **扩展到动态环境**：将框架推广到支持动态客户端参与和变化的网络条件。
- **考虑更多现实约束**：纳入通信带宽限制、客户端可用性（on/off）模式等。
- **应用于其他算法**：将此随机网络框架应用于分析和优化其他先进的 FL 算法，如基于动量的异步方法或压缩通信方法。
- **在线自适应优化**：开发能够在线监测系统状态并实时调整 `p` 和 `m` 的自适应控制策略。

</details>

---

### 3. [Hardware-Agnostic and Insightful Efficiency Metrics for Accelerated Systems: Definition and Implementation within TALP](https://arxiv.org/abs/2603.26576)

**Authors**: Ghazal Rahimi, Victor Lopez, Marc Clasc\`a, Joan Vinyals Ylla Catal\`a, Jesus Labarta, Marta Garcia-Gasulla  
**Category**: cs.DC  
**Published**: 2026-03-30  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.26576v1  

#### Abstract
The increasing adoption of heterogeneous platforms that combine CPUs with accelerators such as GPUs in high-performance computing (HPC) introduces new challenges for performance analysis and optimization. Traditional efficiency metrics, such as those proposed by the Performance Optimization and Prod...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Hardware-Agnostic and Insightful Efficiency Metrics for Accelerated Systems: Definition and Implementation within TALP

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 HPC（High Performance Computing）性能分析中的效率度量（如 POP Parallel Efficiency）主要针对同构的 CPU 架构设计，无法有效捕捉**异构系统中 CPU 与加速器（如 GPU）之间的复杂交互行为**。随着 GPU 在 TOP500 超算中广泛应用（9/10 最强系统含加速分区），开发者面临以下挑战：
- 难以量化**主机（host）与设备（device）间的负载不均衡、通信开销和调度低效**；
- 现有工具多为厂商专用（如 NVIDIA Nsight），缺乏跨平台可移植性和统一视角；
- 性能数据过于底层，非专家用户难以解读瓶颈。

### 🚀 提出的新方法与新思路
本文提出了一套**硬件无关（hardware-agnostic）且具有洞察力的效率度量体系**，扩展了经典的 POP 效率框架，适用于 CPU+加速器的异构架构，并集成到轻量级监控工具 **TALP** 中。

#### 主要创新点包括：
1. **双层级效率度量树（Dual Metric Hierarchies）**
   - 将效率度量分为两个独立分支：
     - **Host-side Metrics**：评估主机资源利用效率，新增 `Host Hybrid Parallel Efficiency` 和 `Device Offload Efficiency`。
     - **Device-side Metrics**：首次系统化定义设备端效率，提出 `Device Parallel Efficiency` 及其子项：
       - `Load Balance`（设备间计算负载均衡）
       - `Communication Efficiency`（内存传输开销）
       - `Orchestration Efficiency`（调度协调效率）

2. **数学形式化定义**
   - 所有新指标均基于时间状态分类进行严格建模：
     - Host 三态：Useful / Device Offloading / MPI calls
     - Device 三态：Kernel Computation / Memory Operation / Idle
   - 支持在单次运行中直接计算效率，无需多规模基准对比。

3. **TALP 框架实现**
   - 在 DLB 库的 TALP 模块中实现了上述指标，支持：
     - 运行时（online）与事后（post-mortem）分析
     - 输出文本 + JSON 格式，便于自动化处理
     - 插件式架构，兼容 NVIDIA（CUDA/CUPTI）、AMD（HIP/rocprofiler）平台

### 🔍 相比现有方法的优势
| 特性 | 传统工具（Nsight, TAU, HPCToolkit） | 本文方法（TALP + 新指标） |
|------|-------------------------------|-----------------------------|
| 平台依赖性 | 厂商绑定（如仅支持 NVIDIA） | ✅ 跨厂商（NVIDIA & AMD） |
| 抽象层次 | 原始事件计数（kernel time, memcpy） | ✅ 高层效率解释（可操作洞察） |
| 易用性 | 数据繁杂，需专家解读 | ✅ 自动聚合为结构化效率树 |
| 编程模型透明性 | 多需代码插桩 | ✅ 通过 PMPI/OMPT 无侵入集成 |

---

## 2. 核心实验方法和设置

### 🧪 实验设置
所有实验在 **MareNostrum5-Acc** 节点上执行，配置如下：
- CPU：2× Intel Sapphire Rapids 8460Y
- GPU：4× NVIDIA H100
- 编程模型：MPI + OpenACC / CUDA

### 📊 评估方法与指标
采用两类测试场景验证所提指标的有效性：

#### （1）合成基准测试：PILS（Portable Instrumented Load Simulator）
- 设计 7 种典型执行模式，控制变量包括：
  - 计算负载分布（平衡 vs 不平衡）
  - CPU/GPU 工作比例
  - 数据搬移频率
  - 是否重叠 CPU 与 GPU 执行
- 使用 Paraver 可视化 trace，与 TALP 输出对比验证指标准确性。

#### （2）真实科学应用（Production HPC Codes）
| 应用 | 功能 | 编程语言 | 并行方式 |
|------|------|----------|---------|
| **SOD2D** | 流体力学模拟（LES/DNS） | Fortran | MPI + OpenACC |
| **FALL3D** | 大气输运与沉降模型 | Fortran | MPI + CUDA |
| **XSHELLS** | 球壳内 Navier-Stokes 方程求解 | C++ | MPI + CUDA |

> 所有应用均从 1 到 8 节点进行弱扩展测试，结合 TALP 输出与 Nsight Systems / Paraver trace 对比分析。

### ⚖️ 基线对比
- **无显式基线算法对比**，而是将 TALP 指标输出与可视化 trace 进行定性一致性验证。
- 强调与原始 POP 指标的延续性，突出其对异构系统的增强能力。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ 合成测试（PILS）关键发现：
| Use Case | 场景描述 | 关键指标表现 |
|--------|--------|------------|
| UC1 | GPU 满载，CPU 卸载充分 | `Device Offload Eff.` = 0.16（低），`Orchestration Eff.` = 0.82 → 表明 CPU 成为瓶颈 |
| UC2 | CPU 主导，GPU 几乎闲置 | `Device Parallel Eff.` = 0.05 → 明确揭示 GPU 未被充分利用 |
| UC3 | GPU 负载严重失衡 | `Device LB` = 0.55，`Host MPI PE` = 0.63 → 指出设备与主机双重不平衡 |
| UC6 | 存在大量 Host-Device 数据移动 | `Comm. Eff.` = 0.36 → 准确定位通信瓶颈 |
| UC7 | 对比是否重叠 CPU/GPU 计算 | 重叠后 `Device Offload Eff.` 从 0.64 → 0.97（↑33%），体现优化效果 |

> 所有案例中，TALP 指标与 trace 视图高度一致，证明其有效性。

#### ✅ 生产应用实测结果（节选 8 节点数据）

| 应用 | Host PE | Device PE | 关键瓶颈 |
|-----|--------|----------|--------|
| **SOD2D** | 0.04 | 0.59 | `Device Offload Eff.` ≈ 0.06 → CPU 几乎只用于卸载任务，利用率极低；`Orchestration Eff.` 下降至 0.60，表明调度不足 |
| **FALL3D** | 0.07 | 0.03 | `Load Balance` = 0.12（host），`Orchestration Eff.` = 0.04 → 主机负载极不均，GPU 长期空闲 |
| **XSHELLS** | 0.15 | 0.10 | `Comm. Eff.`（host）从 0.91→0.27 → 随节点增加，MPI 开销剧增导致无法及时向 GPU 卸载任务 |

> 所有应用均显示：**随着规模扩大，Host-side 通信效率下降成为限制 Device Utilization 的关键因素**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **异构系统必须分离 host 与 device 效率分析**
   - 统一度量会掩盖真实瓶颈（例如高 Device PE 可能伴随极低 Device Offload Eff.）。
   - 分离后能精准定位是“GPU 没活干”还是“GPU 活太多”。

2. **Device Offload Efficiency 是连接主从的关键桥梁**
   - 它反映 CPU 是否有效利用自身资源来驱动 GPU；
   - 若该值低，说明 CPU 频繁阻塞于同步、数据拷贝等操作。

3. **Orchestration Efficiency 揭示调度潜力**
   - 即使 Load Balance 和 Communication 高，若 Orchestration 低，仍表示 GPU 存在大量 idle 时间；
   - 是指导“重叠计算与通信”的重要依据。

4. **真实应用普遍存在“主机拖累设备”现象**
   - 多数情况下 GPU 计算能力强，但受限于主机侧的 MPI 通信或任务分发延迟；
   - 提升主机通信效率可显著改善整体性能。

### ⚠️ 方法的局限性
- 当前仅实现 **Device Parallel Efficiency** 分支，尚未完成 **Device Computational Efficiency**（如 IPC、Occupancy 等微架构层面度量）；
- 假设每个 GPU 作为一个整体资源，未考虑 stream、SM 利用率差异；
- 当前插件主要支持 NVIDIA，AMD HIP 支持仍在发展中。

### 🔮 未来工作方向
1. 完善 Device Computational Efficiency 分支，引入类似 IPC Scalability 的度量；
2. 扩展至更多加速器平台（Intel GPU, FPGA）；
3. 支持动态反馈机制，将效率指标用于 runtime 自适应调度（如结合 DLB 的 DROM 模块）；
4. 探索机器学习辅助的自动瓶颈识别与优化建议生成。

---

## 总结
本文提出的 **TALP 扩展效率度量体系**，成功将经典 POP 方法推广至异构加速系统，提供了**标准化、可解释、跨平台的性能洞察工具**。实验表明，这些指标不仅能准确反映合成负载的行为特征，也能在真实 HPC 应用中揭示深层次的协同效率问题，为开发者提供明确的优化路径。该工作推动了 HPC 性能分析从“可观测”走向“可理解”、“可行动”。

</details>

---

### 4. [Constitutive parameterized deep energy method for solid mechanics problems with random material parameters](https://arxiv.org/abs/2603.26030)

**Authors**: Zhangyong Liang, Huanhuan Gao  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.26030v1  

#### Abstract
In practical structural design and solid mechanics simulations, material properties inherently exhibit random variations within bounded intervals. However, evaluating mechanical responses under continuous material uncertainty remains a persistent challenge. Traditional numerical approaches, such as ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Constitutive Parameterized Deep Energy Method for Solid Mechanics Problems with Random Material Parameters

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在实际工程结构设计和固体力学仿真中，材料参数（如 Young's modulus $E$ 和 Poisson's ratio $\nu$）通常具有随机性和不确定性，表现为在一定区间内的连续变化。传统数值方法（如 FEM）在处理这类**连续多参数不确定性**时面临巨大挑战：
- 需对每个参数组合重复进行网格划分、刚度矩阵组装和求解，计算成本极高；
- 数据驱动的代理模型依赖大量高保真模拟数据，生成成本高昂；
- 物理信息神经网络（如标准 DEM 或 PINNs）在材料参数改变时需从头训练，无法实现快速推理。

### 提出了什么新方法或新思路
本文提出了一种全新的纯物理驱动深度学习框架——**Constitutive Parameterized Deep Energy Method (CPDEM)**，其核心思想是：
- 将材料本构参数（如 $E, \nu$）作为显式输入嵌入到神经网络中，通过一个独立的 **material encoder** 将其编码为隐空间表示 $h_{\text{param}}$；
- 构建参数化能量泛函，使网络输出的位移场 $u(X; \eta)$ 成为材料参数 $\eta = [E, \nu]$ 和空间坐标 $X$ 的联合函数；
- 在预训练阶段通过最小化期望能量 $\mathbb{E}_{\eta \sim p(\eta)}[\Pi(u(X;\eta);\eta)]$ 学习整个参数空间上的解流形（solution manifold）；
- 实现“零样本”（zero-shot）推理：对于任意未知但位于训练区间的材料参数，无需重新训练即可实时预测位移场。

### 相比现有方法的优势
| 方法 | 缺陷 | CPDEM 的优势 |
|------|------|----------------|
| **FEM** | 每个参数配置需重新离散化与求解，计算复杂度为 $O(N_q \cdot T_{\text{FEM}})$ | 仅一次预训练，查询时间仅为前向传播 $T_{\text{infer}} \ll T_{\text{FEM}}$ |
| **Standard DEM / PINNs** | 材料参数固化于损失函数中，每次变更需完全重训 | 参数化架构支持跨参数泛化，避免重复训练 |
| **Neural Operators (e.g., DeepONet, FNO)** | 依赖大规模高保真数据集训练，数据获取成本高 | 完全无监督、无需任何标签数据，仅依赖物理定律 |
| **Data-driven Surrogates** | 外推能力差，面对分布外（OOD）参数易失效 | 结合 fast fine-tuning 可高效适应 OOD 参数 |

> ✅ **核心创新**：首次实现了**纯物理驱动下对连续多参数不确定性的统一建模与高效推理**，突破了传统方法在效率与泛化之间的瓶颈。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
本研究为**无监督物理驱动方法**，未使用任何真实或仿真数据集作为训练标签。所有实验均基于以下典型固体力学问题构建：
- **1D 线弹性杆**（Analytical solution available）
- **2D 弹性悬臂梁**（Reference: FEM with 150×50 mesh）
- **1D/2D/3D Neo-Hookean 超弹性梁**
- **Mooney-Rivlin 超弹性模型**
- **大变形无摩擦接触问题（Ironing Problem）**

材料参数被设定为服从均匀分布或截断高斯分布的随机变量，例如：
- $E \sim \mathcal{U}[0.5, 3.0]$ （1D 杆）
- $(E, \nu) \in [800, 1200] \times [0.25, 0.35]$ （2D 超弹性梁）

### 实验设置和评估指标

#### 网络架构
- 输入：空间坐标 $X$ + 材料参数 $\eta$
- 三模块结构：
  - `g_coord`: 坐标编码器（FC 网络）
  - `g_param`: 材料参数编码器（FC 网络）
  - `g_manifold`: 流形网络（FC 网络），融合两者输出得到位移预测
- 激活函数：ReLU / Tanh
- 输出形式：满足边界条件的试函数构造（如乘子法）

#### 优化策略
- **两阶段训练**：
  1. Adam 优化器（初值学习率 0.5）进行快速下降
  2. 切换至 L-BFGS 进行精细收敛
- 损失函数：参数化总势能 $\mathcal{L}_{\text{CPDEM}} = U - W$

#### 评估指标
- **相对 $L_2$ 误差**：$\frac{\|u_{\text{pred}} - u_{\text{ref}}\|_2}{\|u_{\text{ref}}\|_2}$
- **逐点绝对误差图**
- **训练损失收敛曲线**（total loss, energy loss, boundary loss）
- **计算复杂度分析**（scaling analysis）

### 基线方法对比
| 方法 | 是否需要数据 | 是否需重训 | 查询效率 | 泛化能力 |
|------|---------------|------------|-----------|-----------|
| **FEM** | 否 | 是（每组参数） | 低（$O(T_{\text{solve}})$） | 强（但慢） |
| **Standard DEM** | 否 | 是（每组参数） | 中等（训练耗时） | 差（单参数） |
| **Neural Operator (e.g., DeepONet)** | 是（大量 FEM 数据） | 否（once trained） | 极高 | 分布内强，分布外弱 |
| **CPDEM (Ours)** | 否 | 否（仅预训练一次） | 极高（毫秒级前向） | 强（支持 zero-shot + fine-tuning） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 任务 | 参数范围 | 最大相对 $L_2$ 误差 | 典型误差量级 |
|------|----------|------------------------|--------------|
| 1D 线弹性杆 | $E \in [0.5, 3.0]$ | < 0.45% | $10^{-3} \sim 10^{-4}$ |
| 2D 弹性梁 | $(E,\nu) \in [100,500]\times[0.25,0.35]$ | 视觉无差异 | 绝对误差 ~ $10^{-2}$ |
| 1D Neo-Hookean 杆 | $E \in [0.5,3.0]$ | 2.737e-2 ($E=3.0$) | 随刚度增加略有上升 |
| 2D 超弹性梁 | $E \in [800,1200], \nu \in [0.25,0.35]$ | ~$10^{-2}$ | 与 FEM 几乎一致 |
| 3D 扭转梁 | 参数化三维问题 | 成功建模 | 图像验证有效 |

> 🔍 **Figure 4 显示**：在整个 $E \in [0.5,3.0]$ 区间内，CPDEM 对 1D 杆的预测误差始终保持在 0.15% 以下，最大不超过 0.45%，表明良好的参数内泛化能力。

### 与基线方法的对比结果
- **相比 FEM**：
  - 在 $N_q = 1000$ 次查询场景下，FEM 总耗时呈线性增长 $O(N_q)$；
  - CPDEM 总耗时为 $T_{\text{pre-train}} + O(N_q \cdot T_{\text{infer}})$，其中 $T_{\text{infer}} \approx \mathcal{O}(1)\,\text{ms}$，**摊销后每查询成本趋近于零**。
- **相比 Standard DEM**：
  - 每次参数变更需耗时数分钟至数十分钟重训；
  - CPDEM 仅需一次预训练（约几十分钟），后续任意参数即时响应。
- **相比 Neural Operators**：
  - 后者需数千组 FEM 模拟数据用于训练，生成成本极高；
  - CPDEM 完全免数据，直接从物理方程出发。

### 消融实验结果（如有）
虽然文中未明确列出“ablation study”章节，但通过多个案例验证了不同组件的有效性：
- **Encoder 设计必要性**：分离的 `g_coord` 与 `g_param` 编码器有助于学习更鲁棒的特征表示；
- **Fast Fine-tuning 效果显著**：
  - 对于超出训练区间的 OOD 参数（如 $E = 1300 > 1200$），zero-shot 推理误差增大；
  - 仅用 **<50 步 L-BFGS 微调**，即可将误差降至与 ID 参数相当水平，节省 >95% 时间。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **CPDEM 成功构建了一个可参数化的物理驱动代理模型**，能够在单一网络中同时处理连续变化的材料参数。
2. ✅ **实现了真正的 zero-shot 推理**：对于训练区间内任意未见的 $(E, \nu)$ 组合，均可实时输出高精度位移场。
3. ✅ **兼具高精度与高效率**：在多种线性/非线性、小/大变形问题上均达到与 FEM 相当的精度，且推理速度极快。
4. ✅ **具备良好外推潜力**：结合 fast fine-tuning 策略，可快速适配分布外参数，适用于实际工程中的异常材料检测与数字孪生更新。

### 方法的局限性
- 当前假设材料参数为**空间均匀的随机变量**，尚未考虑空间变异的随机场（如 KL 展开描述的功能梯度材料）；
- 对极端非凸能量 landscape（如高度非线性损伤或塑性问题）的稳定性仍待进一步验证；
- 网络容量与参数空间维度之间存在权衡，高维参数空间可能需要更深/更宽网络。

### 未来工作方向
1. **扩展至空间随机场建模**：结合 Karhunen-Loève (KL) expansion 建立空间-参数联合编码器；
2. **引入路径相关本构关系**：拓展至 elastoplasticity、damage mechanics 等历史依赖问题；
3. **集成不确定性量化（UQ）**：结合贝叶斯神经网络或 dropout 实现概率输出；
4. **面向数字孪生与实时监测应用**：将 CPDEM 部署为边缘设备上的轻量级求解器，支持在线健康监测与反演识别；
5. **多物理场耦合扩展**：应用于热-力、电-机（electromechanical coupling）等问题。

---

> 📌 **总结一句话**：  
> **CPDEM 是首个纯物理驱动、支持连续多参数不确定性的统一深度学习求解器，在无需数据、免重训的前提下实现了高效、精确、可推广的力学响应预测，为下一代智能 CAE 工具提供了全新范式。**

</details>

---

### 5. [Are LLM-Enhanced Graph Neural Networks Robust against Poisoning Attacks?](https://arxiv.org/abs/2603.26105)

**Authors**: Yuhang Ma, Jie Wang, Zheng Yan  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.26105v1  

#### Abstract
Large Language Models (LLMs) have advanced Graph Neural Networks (GNNs) by enriching node representations with semantic features, giving rise to LLM-enhanced GNNs that achieve notable performance gains. However, the robustness of these models against poisoning attacks, which manipulate both graph st...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Are LLM-Enhanced Graph Neural Networks Robust against Poisoning Attacks?

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统性地研究了**LLM-enhanced GNNs**在**Poisoning Attacks**下的鲁棒性问题，填补了当前研究中的三大空白：
- **L1**: 现有研究对受害模型（victim models）和攻击类型的覆盖不全；
- **L2**: 缺乏对“后LLM时代”数据集（post-LLM datasets）的评估，存在潜在的**ground truth leakage**风险；
- **L3**: 缺少从攻防双视角出发的未来研究方向指导。

### 提出了什么新方法或新思路
作者提出了一个**全面的鲁棒性评估框架**（robustness assessment framework），其核心创新包括：
- **多维度评估设计**：结合8种LLM/LM特征增强器与3种GNN骨干网络（GCN、GAT、GraphSAGE），构建24种受害模型组合。
- **多样化攻击策略**：涵盖6种结构攻击（3种针对性 + 3种非针对性）和3种文本攻击（字符级、词级、句子级），实现跨模态攻击覆盖。
- **引入后LLM数据集**：首次引入**Tape-arxiv23**这一2023年后发布的数据集，避免因LLM预训练语料重叠导致的**ground truth leakage**，确保评估公平性。
- **提出新型联合攻击与防御机制**：设计了一种结合结构与文本扰动的**combined attack**，并提出基于LLM生成嵌入的**graph purification defense**。

### 相比现有方法的优势
- **更全面**：是目前对LLM-enhanced GNN鲁棒性评估中**覆盖最广**的研究，涉及最多模型与攻击类型。
- **更真实**：采用灰盒攻击设定（gray-box setting），更贴近实际场景。
- **更可靠**：通过引入Tape-arxiv23数据集，有效缓解了ground truth leakage问题，提升了结果可信度。
- **更具前瞻性**：不仅评估现状，还从攻防两端提出未来研究方向，推动领域发展。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
共使用4个真实世界的**Text-Attributed Graphs (TAGs)** 数据集：

| 数据集 | 节点数 | 边数 | 类别数 | 领域 | 特征提取方式 |
|--------|--------|------|--------|------|--------------|
| **Cora** | 2,708 | 5,429 | 7 | 引用网络 | BoW |
| **Pubmed** | 19,717 | 44,338 | 3 | 引用网络 | TF-IDF |
| **Ogbn-products (subset)** | 12,011 | 21,987 | 47 | 电商网络 | BoW |
| **Tape-arxiv23 (subset)** | 13,167 | 23,735 | 40 | 引用网络（arXiv论文） | Word2Vec |

> ✅ **特别说明**：Tape-arxiv23发布于LLM广泛应用之后，极大降低了ground truth leakage风险。

---

### 实验设置和评估指标

#### 攻击设置
- **结构攻击**（Structural Poisoning）：
  - **针对性攻击**（Targeted）：Nettack、SGA、NAG-R
  - **非针对性攻击**（Non-targeted）：Mettack、DICE、PGA
- **文本攻击**（Textual Poisoning）：
  - **字符级**：DeepWordBug (DWord)
  - **词级**：Bert-Attack (BertAtk)
  - **句子级**：MAYA

所有攻击均在**灰盒设定**下进行：攻击者知晓图结构、节点属性和部分标签，但不知晓模型架构与参数。

#### 评估指标
- **Accuracy (ACC)**：测试集上的分类准确率，衡量绝对鲁棒性。
- **Relative Drop in Accuracy (RDA)**：  
  $$
  \text{RDA} = \frac{\text{ACC}_{\text{clean}} - \text{ACC}_{\text{attack}}}{\text{ACC}_{\text{clean}}}
  $$  
  衡量相对性能下降，值越低表示鲁棒性越强。

#### 嵌入质量分析维度
为深入理解鲁棒性来源，作者还分析了以下嵌入特性：
- **Embedding Visualization**（t-SNE可视化）
- **Embedding Separability**（DBI、Silhouette Score）
- **Label Information Preservation**（Embedding Homophily, ELMI）
- **Structural Information Preservation**（ESMI, Neighbor Consistency）

---

### 基线方法对比
- **基线模型**：使用传统浅层嵌入（shallow embeddings）作为基线，如 BoW、TF-IDF、Word2Vec。
- **对比对象**：将上述基线与24种LLM-enhanced GNN模型进行对比，涵盖三类集成范式：
  - **LLM-Explanation**：TAPE、KEA
  - **LLM-Embedding**：LLaMA、TE3L、Linq
  - **LM-Embedding**：SimTeG、E5-Large、ModernBert

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在高扰动下的结构攻击表现（以Mettack为例）
| 模型 | Cora (ACC/RDA) | Pubmed (ACC/RDA) | Ogbn-products (ACC/RDA) | Tape-arxiv23 (ACC/RDA) |
|------|----------------|------------------|--------------------------|-------------------------|
| **Shallow emb.** | 74.60 / 7.92% | 74.12 / 14.02% | 63.78 / 3.29% | 51.03 / 11.59% |
| **TAPE (best)** | 78.02 / 4.55% | 84.91 / 8.82% | 84.26 / 1.84% | 71.33 / 4.22% |
| **E5-Large (best)** | 80.60 / 4.51% | 84.71 / 8.97% | 83.23 / 1.91% | 70.04 / 4.16% |

> ✅ **观察**：LLM-enhanced模型在所有数据集上均显著优于基线，尤其在Ogbn-products和Tape-arxiv23等大规模数据集上优势明显。

#### 文本攻击下的表现（以MAYA为例）
| 模型 | Cora (ACC/RDA) | Pubmed (ACC/RDA) | Ogbn-products (ACC/RDA) | Tape-arxiv23 (ACC/RDA) |
|------|----------------|------------------|--------------------------|-------------------------|
| **Shallow emb.** | 80.54 / 0.59% | 85.54 / 0.78% | 65.03 / 1.39% | 53.14 / 7.93% |
| **TAPE** | 82.55 / -0.99% | 88.96 / 1.47% | 85.48 / 0.42% | 71.33 / 4.22% |
| **E5-Large** | 83.38 / 1.22% | 86.58 / 4.65% | 84.03 / 1.09% | 69.69 / 4.64% |

> ✅ **观察**：多数LLM-enhanced模型在文本攻击下仍保持高ACC，部分甚至出现**负RDA**（性能提升），表明扰动可能起到正则化作用。

---

### 与基线方法的对比结果
- **总体趋势**：LLM-enhanced GNNs在各类攻击下均表现出**更高的ACC**和**更低的RDA**，尤其是在高扰动率下优势更加显著。
- **鲁棒性排序**：  
  `E5-Large ≈ TAPE > TE3L > LLaMA > 其他LM > 浅层嵌入`
- **GNN骨干影响**：GraphSAGE整体表现最优，因其保留自表示（self-representation）和均值聚合机制增强了抗干扰能力。

---

### 消融实验结果（关键发现支持）

#### （1）Ground Truth Leakage 并非主因
- 在Tape-arxiv23上，尽管LLMs未见过该数据，LLM-enhanced模型依然表现优异。
- 控制训练-测试数据重叠率实验显示：即使在0% overlap下，LLM-enhanced模型仍保持显著优势 → **鲁棒性源于架构整合而非数据泄露**。

#### （2）嵌入质量决定鲁棒性
- **Embedding Visualization** 显示：TAPE 和 E5-Large 的嵌入聚类更清晰。
- **Embedding Separability**（DBI↓, Sil↑）：LLM/LM生成的嵌入具有更强的可分性。
- **Label & Structural Info Preservation**：LLM-enhanced嵌入在扰动后仍能较好保留标签同质性（homophily）和邻居一致性（neighbor consistency）。

#### （3）联合攻击更有效
- **Combined Attack**（结构 + 文本）效果强于单一攻击，尤其是**句子级文本攻击 + 结构攻击**组合最具破坏力（见Table 7）。

#### （4）提出的防御机制有效
- **Graph Purification Defense**（基于嵌入相似度过滤边）显著提升模型性能，在某些情况下甚至**超过干净图上的表现**（见Table 8）。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **LLM-enhanced GNNs 对Poisoning Attacks具有更强鲁棒性**，尤其在高扰动下表现稳定。
2. ✅ 鲁棒性提升**并非来自ground truth leakage**，而是源于LLM/LM生成的高质量嵌入对标签与结构信息的有效编码。
3. ✅ **LLM-Explanation**（如TAPE）和**fine-tuned LM-Embedding**（如E5-Large）表现最佳，表明任务适配的微调至关重要。
4. ✅ **GraphSAGE**作为GNN骨干展现出最强鲁棒性，因其聚合机制更抗局部扰动。
5. ✅ **文本攻击有效性较低**，而**结构攻击仍是主要威胁**；但二者结合可显著增强攻击效果。
6. ✅ 提出的**graph purification defense**利用LLM嵌入进行边过滤，是一种简单且高效的防御手段。

---

### 方法的局限性
- **计算开销大**：LLM推理成本较高，尤其在大规模图上部署受限。
- **依赖外部API**：部分LLM（如TE3L）需调用OpenAI接口，存在隐私与可用性风险。
- **攻击模拟简化**：当前攻击未考虑动态演化图或多跳协同攻击。
- **防御阈值敏感**：graph purification依赖人工设定相似度阈值，缺乏自适应机制。

---

### 未来工作方向

#### 攻击端（Offensive Perspectives）
- 利用LLM作为**link predictor**补全缺失结构信息，实现半监督攻击。
- 设计**更隐蔽的结构攻击**，保留图的关键统计属性（如聚类系数）。
- 开发**高效文本攻击**，降低对LLM作为代理模型的依赖。
- 探索**协调式联合攻击**，动态选择最优扰动策略。

#### 防御端（Defensive Perspectives）
- 构建**天然鲁棒的模型架构**，如结合fine-tuned LM与GraphSAGE。
- 发展基于LLM嵌入的**图净化预处理技术**，提升通用性。
- 将对抗扰动视为**数据增强**，用于训练更泛化的GNN模型。

---

### 总结
该论文首次系统评估了LLM-enhanced GNNs在多种Poisoning Attacks下的鲁棒性，揭示了其优越性能背后的本质原因，并提出了新的攻击与防御思路。研究成果不仅验证了LLM在提升GNN安全性方面的潜力，也为未来构建更安全、可靠的图学习系统提供了理论基础与实践工具。

> 🔗 **开源代码**：作者已将完整框架开源 → [GitHub链接](https://github.com/CyberAlSec/LLMEGNNRP)

</details>

---

### 6. [AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents](https://arxiv.org/abs/2603.26034)

**Authors**: Wenbo Gao, Renxi Liu, Xian Wang, Fang Guo, Shuai Yang, Xi Chen, Hui-Ling Zhen, Hanting Chen, Weizhe Lin, Xiaosong Li, Yaoyuan Wang  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.26034v1  

#### Abstract
Autonomous agents powered by large language models (LLMs) perform complex tasks through long-horizon reasoning and tool interaction, where a fundamental trade-off arises between execution efficiency and reasoning robustness. Models at different capability-cost levels offer complementary advantages: ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）驱动的自主智能体在执行复杂、多步骤任务（如网页搜索、数学推理、长文本生成）时面临一个根本性的权衡：**推理鲁棒性**与**执行效率**之间的矛盾。

- 小型模型（small model）推理速度快、成本低，但在困难的推理环节容易出错，导致后续步骤累积错误。
- 大型模型（large model）推理能力强，但每一步延迟高，整体执行时间长。

传统方法通常采用固定模型或外部路由机制（如 RouteLLM、FrugalGPT），难以在长周期（long-horizon）任务中实现端到端的效率优化。

---

### 提出的新方法与新思路
本文提出 **AgentCollab**，一种**自评估驱动的协作推理框架**，其核心思想是：

- **无需外部路由器或训练策略**，而是利用智能体自身的**自我反思信号**（self-reflection signal）来判断当前推理轨迹是否陷入停滞。
- 当检测到进展停滞（stagnation）时，动态将控制权**临时升级**（escalate）给更强的大型模型进行干预；一旦恢复进展，再降级回小型模型继续执行。
- 引入**难度感知的累积预算分配策略**（difficulty-aware cumulative budget allocation）：连续失败次数越多，赋予大型模型的干预步数越长，从而实现更深度的纠正。

---

### 相比现有方法的优势
| 方面 | AgentCollab | 传统方法（如 RouteLLM / FrugalGPT） |
|------|-------------|-------------------------------|
| 路由决策依据 | 模型自身对**整个推理轨迹**的自省 | 外部评分模型或预训练判别器 |
| 决策粒度 | 轨迹级别（trajectory-level） | 单步级别（step-level） |
| 是否需要额外训练 | 否 | 是（需偏好数据或质量标签） |
| 是否引入额外延迟 | 否（无额外模块） | 是（路由模型推理开销） |
| 长周期稳定性 | 更强（避免频繁切换） | 较弱（可能震荡） |

> ✅ **核心优势**：通过内部自省实现高效协作，在保持高准确率的同时显著提升端到端推理速度。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖三大类多步智能体任务，均具有挑战性和现实意义：

| 数据集 | 任务类型 | 描述 |
|--------|----------|------|
| **BrowseComp_zh** | 深度网页研究 | 中文环境下基于网络的信息检索与综合分析任务（289题） |
| **HLE-math** | 数学推理 | 复杂数学问题求解，要求多步逻辑推导（866题） |
| **WritingBench** | 长文本生成 | 开放式长篇写作能力评测（1000题） |

---

### 实验设置
- **基础框架**：集成至两个开源智能体系统：
  - **DDV2**（7B/38B 模型）
  - **WebSailor**（3B/7B/32B 模型）
- **最大交互轮次**：40 步
- **预算策略**：采用线性增长形式，$ B_0 = 2 $, $ k = 2 $
- **硬件平台**：Ascend 910B3 NPU 上使用 vLLM-Ascend 进行本地推理

---

### 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy / Score** | 任务完成正确率（BrowseComp_zh/HLE-math）或生成质量得分（WritingBench），由 GPT-4o 或基准自带评分器评估 |
| **Speedup** | 相对于纯 large model 的端到端延迟加速比（越大越好） |
| **#Steps** | 平均每个问题所需的推理迭代步数 |
| **Switching Ratio** | 模型切换频率，衡量计算缓存复用效率 |

---

### 基线方法对比
| 基线方法 | 类型说明 |
|---------|----------|
| **Small-only** | 全程使用小型模型 |
| **Large-only** | 全程使用大型模型（性能上限） |
| **Random** | 随机在 small/large 之间切换 |
| **RouteLLM** | 基于质量预测的单步路由机制 |
| **FrugalGPT** | 基于胜率预测的经济型路由机制 |

> 所有基线均适配为 multi-turn setting 下运行。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Agent | Method | BrowseComp_zh Acc.(%) | Speedup | HLE-math Acc.(%) | Speedup | WritingBench Score | Speedup |
|-------|--------|------------------------|---------|------------------|---------|--------------------|---------|
| DDV2 | Small | 18.3 | 1.54× | 8.0 | 3.38× | 4.4 | 3.20× |
| DDV2 | Large | 34.6 | 1.00× | 23.3 | 1.00× | 5.1 | 1.00× |
| DDV2 | RouteLLM | 28.2 | 1.09× | 15.2 | 1.62× | 4.7 | 2.21× |
| DDV2 | FrugalGPT | 27.2 | 1.29× | 18.2 | 1.97× | 4.5 | 1.94× |
| DDV2 | **Ours (AgentCollab)** | **33.9** | **1.36×** | **21.1** | **2.31×** | **5.0** | **2.43×** |

> 🔺 在 DDV2 上，AgentCollab 接近 large-only 的精度（33.9 vs 34.6），同时获得 **1.36× 的端到端加速**，远超其他切换策略。

---

### 与其他方法对比的关键发现
- **显著优于随机切换和静态路由方法**：
  - 在 BrowseComp_zh 上比 RouteLLM 提升 +5.7% 准确率且更快（1.36× vs 1.09×）。
  - 表明**轨迹级自省信号比独立步决策更有效**。
- **接近 large-only 性能，但效率大幅提升**：
  - 在 DDV2 上达到 large model 98% 的准确率，却节省约 30% 时间。
- **减少无效切换，提高 cache 复用**：
  - 动态预算策略将 switching ratio 从 49.64（静态）降至 45.07，降低震荡风险。

---

### 消融实验结果（Table 2 & Figure 5）

#### （1）角色分配影响（Planner vs Executor）
| 配置 | Accuracy | Speedup |
|------|----------|---------|
| Large Planner + Small Executor | 24.6% | 1.39× |
| Small Planner + Large Executor | 27.3% | 1.24× |

> 📌 发现：**执行阶段（Executor）更依赖强模型**，因其直接与环境交互、获取证据，错误影响更大。

#### （2）预算策略对比
| 策略 | Accuracy | Speedup |
|------|----------|---------|
| Static Budget | 32.5% | 1.32× |
| Dynamic Budget (**Ours**) | **33.9%** | **1.36×** |

> ✅ 动态预算策略进一步提升了性能与效率。

#### （3）参数 $k$ 敏感性分析（Figure 5）
- $k=2$ 时取得最佳质量-效率帕累托前沿（Pareto frontier）。
- $k=0$（即静态）效果最差，验证了**随难度递增干预强度**的重要性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **自省信号可作为有效的协作触发机制**：LLM 自身即可判断推理是否停滞，无需外部监督或训练。
2. ✅ **动态协作优于固定模型或简单路由**：仅在必要时刻调用大模型，能集中资源解决关键瓶颈。
3. ✅ **难度感知预算提升稳定性**：连续失败后延长接管时间，有助于打破局部循环，减少反复切换。
4. ✅ **端到端效率取决于轨迹质量而非单步速度**：小模型虽快，但易走弯路；适度引入大模型反而缩短总耗时。

> 💡 “It’s not about going fast at every step, but staying on the right path.”

---

### 方法的局限性
1. **依赖模型的自省可靠性**：
   - 如 WebSailor 中 false-positive 判断较多（7%），导致升级不及时，性能提升有限。
2. **目前仅限同构模型协作**：
   - 所有实验基于同一架构不同规模的模型（如 7B ↔ 32B），未探索异构模型（如专精不同领域的模型）协作。
3. **未涵盖闭源模型 API 场景**：
   - 因 API 延迟不稳定，无法精确测量效率，故未测试 GPT/Gemini 等商业模型。

---

### 未来工作方向
1. **扩展至异构模型协作**：
   - 探索功能互补模型间的协同机制（如一个擅长搜索，另一个擅长逻辑）。
2. **增强自省信号的鲁棒性**：
   - 设计更可靠的 progress checker，防止误判导致错过升级时机。
3. **应用于更复杂的 multi-agent 系统**：
   - 将 AgentCollab 思想推广至多个 agent 之间的动态分工与协作。
4. **结合强化学习优化升级策略**：
   - 在保留自省主干的基础上，引入轻量 RL 微调升级阈值。

---

## 总结
> **AgentCollab 提供了一种简洁而强大的范式转变：让 LLM 智能体“知道自己何时需要帮助”，并通过内部信号驱动模型协作，实现了效率与质量的双赢。**

该方法无需额外训练、不增加系统复杂度，已在多种多步任务上验证其有效性，代表了高效 LLM agent 设计的一个重要前进方向。

</details>

---

### 7. [MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference](https://arxiv.org/abs/2603.26557)

**Authors**: Joris K\"oster, Zixuan Liu, Siavash Khajavi, Zizhan Zheng  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.26557v1  

#### Abstract
Large Language Models (LLMs) deliver strong performance but incur high inference cost in real-world services, especially under workloads with repeated or near-duplicate queries across users and sessions. In this work, we propose MemBoost, a memory-boosted LLM serving framework that enables a lightwe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在现实世界的大语言模型（LLM）服务中，存在大量重复或语义近似的查询请求。直接对每个请求调用高性能、高成本的 **Large-LLM** 会导致严重的计算资源浪费，显著增加推理成本（如 GPU 时间、能耗、费用）。标准的 **Retrieval-Augmented Generation (RAG)** 虽能提升单次响应质量，但未针对交互式、重复性高的场景进行优化。

### 🚀 提出的新方法：MemBoost
MemBoost 是一种**内存增强型 LLM 推理框架**，通过引入“检索优先、必要时升级”的机制，在保证高质量输出的同时大幅降低推理成本。其核心由三个组件构成：

1. **Associative Memory Engine (AME)**  
   - 外部语义记忆库，存储历史问答对 `(x, y)` 及元数据（类别、时间戳等）。
   - 支持基于嵌入的快速相似性检索（使用 `all-MiniLM-L6-v2` 和 FAISS）。
   - 支持写回机制（write-back），将高质量新答案存入内存供后续复用。

2. **Large-LLM Oracle**  
   - 高能力但高成本的大型模型（如 Qwen3-14B-FP8-dynamic），作为“兜底”模型处理无法从内存获取可靠答案的复杂请求。

3. **Meta-Controller (MC)**  
   - 轻量级小模型（如 Qwen3.5-2B），负责决策流程：
     - 检索 AME 中的相关条目；
     - 判断是否可直接复用或组合生成答案；
     - 若不可靠，则路由至 Oracle；
     - 决定是否将 Oracle 新生成的答案写回 AME。

该架构实现了 **“retrieve → decide → escalate (if needed) → write-back (if needed)”** 的闭环流程。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | MemBoost 的改进 |
|------|--------|------------------|
| 标准 RAG | 仅用于单轮知识增强，不支持跨会话答案复用 | 显式支持语义缓存与持续写回 |
| Semantic Caching | 缓存命中依赖静态策略，缺乏智能路由 | 引入 MC 动态判断是否复用或升级 |
| Multi-LLM Routing | 通常不利用历史生成结果 | 结合内存复用与模型路由，双重降本 |

> ✅ **核心创新**：首次将 **semantic caching**、**retrieval-augmented generation** 和 **cost-aware routing** 统一在一个系统框架下，实现质量与成本的联合优化。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **MMLU-Pro** (Wang et al., 2024)：一个更具挑战性的多任务语言理解基准，涵盖多个学科领域。
- 实验聚焦于其中的 **Business 类别**（共 768 条样本），足够支撑长序列模拟请求。

### ⚙️ 实验设置
- **请求流生成**：采用 **Zipf 分布采样** 构建查询流，模拟真实场景中的“长尾访问模式”：
  - 少数问题高频出现，多数低频。
  - 控制参数 `α ∈ {0.8, 1.1, 1.4}`，越大表示重复率越高（越偏斜）。
- **总步数**：5,000 步连续请求。
- **部署环境**：
  - 所有模型使用 **vLLM** 部署以优化吞吐与延迟。
  - MC 与 Oracle 各运行在独立的 **NVIDIA A100 80GB GPU** 上。
  - 温度设为 0.0 保证输出确定性。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 回答正确率，对比 ground truth |
| **Memory-use Rate $I_t$** | 使用 AME 成功响应的比例，反映节省的 Oracle 调用次数 |
| **Total Inference Cost** | 基于 $C_{\text{total}} = \sum (C_M + C_R) + (1-I_t)C_O$ 的成本模型 |
| **Response Latency** | 平均响应时间（滑动窗口 100 步） |

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| Small LLMs (e.g., Qwen3.5-2B) | 单独运行小模型，无任何增强 |
| Large-LLM Oracle (Qwen3-14B) | 单独运行大模型，作为性能上限参考 |
| Standard RAG / Caching 方法 | 文中虽未直接实现，但通过设计对比突显 MemBoost 的集成优势 |

---

## 3. 主要实验结果和性能指标

### 📊 表格结果（Table 1）：准确率对比（%）
| Model | Zipf 0.8 | Zipf 1.1 | Zipf 1.4 |
|-------|----------|----------|---------|
| Qwen3.5-2B (baseline) | 50.0 | 43.5 | 37.1 |
| Qwen3-14B (Oracle) | 76.4 | 79.9 | 85.0 |
| **MemBoost (Qwen3.5-2B)** | **76.7** | **81.8** | **87.4** |
| MemBoost (Ministral-3-3B) | 76.2 | 79.7 | 85.0 |
| MemBoost (Qwen3-4B) | 76.1 | 79.8 | 85.0 |

> 💡 **关键发现**：
> - 所有 MemBoost 配置均显著超越对应的小模型基线。
> - **MemBoost + Qwen3.5-2B** 不仅追平甚至**超过 Oracle 的准确率**（最高达 87.4% vs 85.0%）。
> - 性能提升归因于：一旦难题被 Oracle 正确解答并写回，后续重复请求即可直接命中高质量缓存，避免错误累积。

### 📈 图表分析
#### Figure 2: Memory-use Rate 随时间增长
- 所有配置下，$I_t$（内存使用率）随时间稳步上升。
- 在 Zipf α=1.4（最偏斜）条件下，最终接近 **90%+ 查询由 AME 直接服务**。
- 表明 AME 通过 write-back 机制不断积累有效知识，系统越来越“聪明”。

#### Figure 3: 响应延迟下降趋势
- MemBoost 的平均响应时间随时间推移持续下降。
- 最终稳定在远低于 Oracle-only baseline 的水平（因 AME 检索远快于 LLM 推理）。

> ✅ **结论**：MemBoost 在高重复负载下，既能保持高质量，又能显著降低延迟与成本。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MemBoost 可显著减少昂贵的 Large-LLM 调用**：
   - 在高重复性工作负载下，超过 90% 的请求可通过 AME 快速响应。
2. **答案质量不仅不下降，反而可能提升**：
   - 因正确答案被缓存复用，避免了每次生成的随机误差。
3. **系统具备自进化能力**：
   - 随着运行时间延长，AME 持续增长，效率越来越高。
4. **轻量 MC 即可胜任控制任务**：
   - 即使使用仅 2B 参数的 Qwen3.5-2B 作为 MC，也能实现接近 Oracle 的表现。

### ⚠️ 局限性
1. **评估范围有限**：
   - 当前实验仅在 MMLU-Pro 的固定选择题上验证，尚未覆盖开放域 QA、代码生成等更复杂任务。
2. **仿真工作负载简化**：
   - 使用 Zipf 抽样固定数据集，未充分测试语义相近但非完全重复的查询（如 paraphrasing）。
3. **潜在 false hit 风险**：
   - 若两个问题语义相似但答案不同，可能导致错误缓存命中，影响可靠性。
4. **写回策略未深入优化**：
   - MC 决定是否写回的标准较简单，缺乏对答案价值的精细评估机制。

### 🔮 未来工作方向
1. **扩展到开放域与长文本任务**：
   - 如对话系统、文档摘要、编程助手等更复杂的交互场景。
2. **引入更鲁棒的检索与匹配机制**：
   - 加入上下文感知、意图识别模块，防止 false hits。
3. **动态缓存管理与淘汰策略**：
   - 研究基于访问频率、时效性、领域重要性的缓存更新机制。
4. **结合 MoE 或 Switch Transformers 架构**：
   - 进一步融合系统级路由与模型内部稀疏激活，实现多层次效率优化。

---

## ✅ 总结
**MemBoost** 提出了一种面向实际部署场景的高效 LLM 推理范式，通过整合 **语义记忆（AME）**、**轻量控制器（MC）** 和 **强模型兜底（Oracle）**，实现了“**用小模型的成本，获得大模型的质量**”。其实验结果表明，在重复性强的工作负载中，MemBoost 能将昂贵模型调用减少 90% 以上，同时维持甚至超越原始大模型的准确性，是迈向低成本、高响应、可持续 LLM 服务的重要一步。

</details>

---

### 8. [CADSmith: Multi-Agent CAD Generation with Programmatic Geometric Validation](https://arxiv.org/abs/2603.26512)

**Authors**: Jesse Barkley, Rumi Loghmani, Amir Barati Farimani  
**Category**: cs.AI  
**Published**: 2026-03-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.26512v1  

#### Abstract
Existing methods for text-to-CAD generation either operate in a single pass with no geometric verification or rely on lossy visual feedback that cannot resolve dimensional errors. We present CADSmith, a multi-agent pipeline that generates CadQuery code from natural language. It then undergoes an ite...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*CADSmith: Multi-Agent CAD Generation with Programmatic Geometric Validation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **text-to-CAD** 生成方法存在两大缺陷：
- **单次生成无验证**：如 *Text-to-CadQuery* 虽能生成高质量代码，但缺乏迭代纠错机制，无法保证几何正确性。
- **依赖视觉反馈但精度不足**：如 *Query2CAD* 使用 BLIP-2 提供视觉反馈进行自修正，但其基于图像的表示是“有损”的，无法捕捉毫米级的维度误差或拓扑错误。

因此，这些方法难以满足工程中对 **尺寸精确性、几何有效性（watertight）和可制造性** 的严格要求。

---

### 提出的新方法与创新思路
作者提出 **CADSmith** —— 一个基于多智能体（multi-agent）架构的闭环 CAD 生成系统，核心创新如下：

#### ✅ 多智能体架构（Multi-agent Architecture）
将任务分解为五个专业化 Agent：
- **Planner**：将自然语言解析为结构化设计规范（JSON），明确组件、尺寸、约束等。
- **Coder**：基于检索增强生成（RAG）从 API 文档中获取上下文，生成 CadQuery 脚本。
- **Executor**：在沙箱环境中执行脚本，提取 OpenCASCADE 内核实测数据。
- **Validator**：结合内核实测 + 视觉语言模型 Judge 进行双重验证。
- **Refiner**：接收反馈并迭代修改代码。

> 避免了单一 LLM 的“自我确认偏差”（self-confirmation bias）。

#### ✅ 双重嵌套修正循环（Dual-loop Programmatic Geometric Validation）
- **内层循环（Inner Loop）**：处理代码执行错误（如语法错误、API 调用失败），最多尝试 3 次。
- **外层循环（Outer Loop）**：处理几何不匹配问题，通过程序化度量驱动迭代优化，最多 5 次。

> 实现从“能运行”到“正确”的跨越。

#### ✅ 独立的 VLM-as-Judge 架构
使用更强的 **Claude Opus** 作为独立 Judge 模型，输入包括：
- 原始 prompt
- 生成的 CadQuery 代码
- OpenCASCADE 内核指标（bounding box、volume、face count、solid validity）
- 三视图渲染图像（isometric, rear-top, front-profile）

Judge 综合判断是否通过，并提供分析性反馈。

> 兼顾数值精度与整体形状感知，避免仅靠数字导致的“假收敛”。

#### ✅ 基于 RAG 的动态知识更新机制
不依赖 fine-tuning，而是通过关键词匹配检索以下两个知识库：
- **KB1: API Docs**（155 条 CadQuery 方法说明 + 示例）
- **KB2: Error-Solution Patterns**（25 类常见错误模式及修复方案）

> 支持随着 CadQuery 库演进而持续保持最新，无需重新训练。

---

### 相比现有方法的优势
| 方面 | CADSmith | Query2CAD | Text-to-CadQuery |
|------|----------|-----------|------------------|
| 是否闭环 | ✅ 是（双循环） | ✅ 是（视觉反馈） | ❌ 否（单次生成） |
| 几何验证方式 | ✅ 程序化内核实测 + VLM 视觉评估 | ⚠️ 图像编码反馈（有损） | ❌ 无验证 |
| 尺寸精度保障 | ✅ 显式毫米级测量 | ❌ 不可靠 | ❌ 依赖训练数据分布 |
| 可扩展性 | ✅ RAG 支持动态更新 | ❌ 固定模型能力 | ❌ 需大量标注数据 fine-tune |
| 自我纠正偏差 | ✅ 使用更强 Judge 模型 | ⚠️ 同一模型自评 | ❌ 无 |

---

## 2. 核心实验方法和设置

### 数据集（Benchmark Dataset）
构建了一个包含 **100 个自然语言提示** 的新基准测试集，分为三个难度层级：
- **T1: Basic Primitives**（50 项）  
  单一基本形体（box, cylinder, cone 等），1–3 步操作。
- **T2: Engineering Parts**（25 项）  
  工程零件（支架、齿轮、带孔板等），需布尔运算，3–8 步。
- **T3: Complex Parts**（25 项）  
  复杂部件（含 loft、sweep、shell、多体合并等），5–15 步。

所有参考脚本均为人工编写、执行验证有效且视觉检查无误。  
提示统一格式：明确 mm 单位、坐标系（Z-up）、原点居中、特征位置标注。

---

### 实验设置
比较三种配置：
1. **Zero-shot Baseline**  
   - 单次调用 Claude Sonnet，无 planner、无 RAG、无 refinement、无 vision。
2. **No-vision Ablation**  
   - 完整 pipeline，但 Judge 不接收三视图图像。
3. **Full Pipeline (with vision)**  
   - 完整 CADSmith 流程，包含双循环 + RAG + VLM Judge。

所有输出均经过 ICP 对齐后计算指标，空间单位为绝对毫米（非归一化），以保留真实尺度信息。

---

### 评估指标（Evaluation Metrics）
沿用 *Text-to-CadQuery* 的标准，便于横向对比：
- **Chamfer Distance (CD)**：双向最近邻距离均值（越低越好）
- **F1 Score**：在 1.0mm 阈值下的 precision 和 recall 的调和平均（越高越好）
- **Volumetric IoU**：1mm 分辨率体素网格的交并比（越高越好）

> 所有指标在 **绝对毫米空间** 中计算，强调实际工程精度。

---

## 3. 主要实验结果和性能指标

### 总体性能对比（Table I）

| Configuration       | Exec % | CD<sub>med</sub> | CD<sub>mean</sub> | F1<sub>med</sub> | IoU<sub>med</sub> |
|---------------------|--------|------------------|-------------------|------------------|-------------------|
| Zero-shot           | 95     | 0.55             | 28.37             | 0.9707           | 0.8085            |
| No-vision           | 99     | 0.48             | 18.19             | 0.9792           | 0.9563            |
| **Full (vision)**   | **100**| **0.48**         | **0.74**          | **0.9846**       | **0.9629**        |

> 🔺 **Mean CD 下降 38×**（28.37 → 0.74），表明大幅减少了灾难性失败；  
> 🔺 **执行成功率提升至 100%**；  
> 🔺 **中位数 F1 提升至 0.9846**，接近完美重建。

---

### 分难度层级表现（Table II）

| Tier         | N  | CD<sub>med</sub> | CD<sub>mean</sub> | F1<sub>med</sub> | IoU<sub>med</sub> |
|--------------|----|------------------|-------------------|------------------|-------------------|
| T1 (Primitives)    | 50 | 0.32             | 0.47              | 0.9985           | 0.9834            |
| T2 (Engineering)   | 25 | 0.32             | 0.58              | 0.9979           | 0.7661            |
| T3 (Complex)       | 25 | 0.96             | 1.42              | 0.8859           | 0.9582            |

- T1/T2 接近完美；
- T3 表现略低但仍达较高水平，体现复杂几何挑战。

> 💡 **消融显示：移除视觉输入后，T3 的 Mean CD 暴增至 49.68（+35×），F1 下降至 0.74**，证明视觉反馈对防止“假收敛”至关重要。

---

### 最大改进案例
- **T3_023**: F1 从 0.037（zero-shot）→ 0.943（CADSmith）
- **T1_021**: F1 从 0.168 → 0.995
- **T2_005**: F1 从 0.095 → 0.867

> 这些案例中，zero-shot 模型生成了看似合理但结构完全错误的几何体，而 CADSmith 通过迭代反馈成功修正。

---

## 4. 关键结论和发现

### 主要发现
1. **闭环程序化几何验证显著提升 CAD 生成质量与可靠性**  
   结合 OpenCASCADE 内核实测（精确维度、拓扑）与 VLM 视觉评估，实现了高保真、可制造的 CAD 输出。

2. **双重反馈机制缺一不可**  
   - 数值反馈指导 LLM 修改参数；
   - 视觉反馈识别结构性错误（如漏特征、错布局）；
   > 二者协同才能避免“数字对但形状错”或“看起来像但不能用”。

3. **Vision 是复杂零件的关键**  
   在 T3 层级，去除视觉输入会导致性能断崖式下降，说明仅靠 kernel metrics 不足以捕捉高级语义错误。

4. **RAG 替代 Fine-tuning 更具可持续性**  
   无需大规模标注数据集即可适应 API 更新，更适合工业软件快速迭代场景。

---

### 方法局限性
1. **固定视角可能遗漏局部细节**  
   如 **T3_019（四轴飞行器框架）** 成功通过所有验证，但臂与中心毂之间存在微小间隙（见 Fig. 6），因渲染分辨率和视角限制未被检测到。

2. **仍存在“近似成功”但不可制造的情况**  
   几何足够接近以至于 metric 和 Judge 均判定为通过，但在实际 CNC 或 3D 打印中会失败。

3. **当前仅支持单个零件（part-level）**  
   尚未扩展至装配体（assembly-level）建模。

---

### 未来工作方向
- **自适应视图选择**：根据几何复杂度自动聚焦关键区域（如接缝、倒角）进行高分辨率检查。
- **不确定性引导的多视角渲染**：让 Judge 主动请求特定角度图像以澄清模糊特征。
- **支持 multi-part assembly 生成**。
- **扩大 benchmark 规模**，覆盖更多制造相关几何类型。
- **跨多个 frontier LLM 进行泛化性评测**。

---

## 总结
> **CADSmith 是首个将程序化几何验证与独立 VLM Judge 深度集成于闭环生成流程中的 text-to-CAD 系统**。它不仅提升了生成精度（Mean CD ↓38×），更确保了工程可用性。该工作为实现“自然语言驱动的自动化工程设计”提供了可靠路径，具有重要的研究价值与工业应用前景。

</details>

---

### 9. [GLU: Global-Local-Uncertainty Fusion for Scalable Spatiotemporal Reconstruction and Forecasting](https://arxiv.org/abs/2603.26023)

**Authors**: Linzheng Wang, Jason Chen, Nicolas Tricard, Zituo Chen, Sili Deng  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.26023v1  

#### Abstract
Digital twins of complex physical systems are expected to infer unobserved states from sparse measurements and predict their evolution in time, yet these two functions are typically treated as separate tasks. Here we present GLU, a Global-Local-Uncertainty framework that formulates sparse reconstruc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GLU: Global-Local-Uncertainty Fusion for Scalable Spatiotemporal Reconstruction and Forecasting

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **digital twin** 系统在处理复杂物理系统时面临两个核心挑战：
- **Sparse reconstruction**：从稀疏、不规则分布的传感器测量中推断完整的物理场状态；
- **Dynamic forecasting**：基于这些不完整初始状态预测系统的长期演化。

现有方法通常将这两个任务**割裂处理**：
- 重建模型注重快照精度，但未考虑时间演化稳定性；
- 预测模型依赖高质量全字段初始化，在实际稀疏观测下表现差。

这导致误差传播严重，尤其在非线性动力学系统中，微小的初始重建误差会迅速放大，造成预测失稳。

---

### 🚀 提出的新方法与创新思路
作者提出 **GLU (Global-Local-Uncertainty)** 框架，首次将稀疏重建与动态预测统一为一个**共享的状态表示学习问题**。其核心创新在于构建一种**结构化潜在状态（structured latent state）**，融合三个互补流：

| 组件 | 功能 |
|------|------|
| **Global stream** | 单个 CLS token 编码系统级全局模式和长程相关性（如大尺度涡旋结构） |
| **Local stream** | N 个 sensor tokens 锚定于真实传感器位置，保留局部高保真细节 |
| **Uncertainty stream** | 学习得到的空间重要性分数（importance score），引导信息加权融合 |

#### 核心机制设计：
- **Global-local-aware encoder**：双向注意力机制实现全局上下文注入到本地 token，避免传统编码中的信息瓶颈。
- **Importance-aware adaptive neighborhood selection**：基于“欧氏距离 + learned importance”的加权度量进行软邻域选择，提升复杂区域的重建鲁棒性。
- **Leader-Follower-Dynamics (LFD)**：层级化动态建模结构，仅通过 leader token 驱动 follower 更新，显著降低计算复杂度并增强长期稳定性。

---

### 🔍 相比现有方法的优势
| 方面 | GLU 优势 |
|------|---------|
| **Representation expressiveness** | 同时保留多尺度结构（大尺度组织 + 小尺度湍流），克服 ROM/FNO 的低通滤波效应和 CNN/GNN 的短视问题 |
| **Scalability** | 内存增长远低于 Transformer 类模型（接近线性 vs 二次），支持大规模传感器部署 |
| **Stability in forecasting** | LFD 结构维持正确的吸引子几何形态，延迟误差累积，尤其对混沌系统有效 |
| **Physical consistency** | 成功保持跨通道热化学耦合关系（如温度-CO浓度关联），符合真实燃烧物理规律 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
共测试七类具有不同物理复杂性的 benchmark：

| 数据集 | 特征 |
|-------|------|
| **Cylinder flow (Re=100)** | 二维层流圆柱绕流，周期性涡脱落 |
| **Collinear plate flow (Re=40/100)** | 平板尾迹流，分别对应准周期与强混沌动力学 |
| **NOAA Sea Surface Temperature (SST)** | 全球海温场，含边界流、季节变化等地理异质性 |
| **Turbulent channel flow (Reτ=180)** | 高雷诺数湍流，多尺度间歇性波动 |
| **FitzHugh-Nagumo reaction-diffusion** | 反应扩散系统，模拟神经脉冲或化学波传播 |
| **Turbulent combustion (methane-air flame)** | 自研 LES 数据集，包含薄反应锋面、强耦合多物理场（速度-温度-CO） |

所有数据标准化为 `[ncases, nt, np, nc]` 张量格式，空间坐标归一化至 `[-1,1]`。

---

### 🧪 实验设置与评估指标

#### 评估任务
- **Spatial Reconstruction**：给定稀疏传感器输入，恢复任意查询点的全场状态。
- **Temporal Forecasting**：基于过去 16 步稀疏观测，自回归滚动预测未来轨迹。

#### 主要评估指标
| 指标 | 描述 |
|------|------|
| **Relative L² Error** | $\frac{\|u_{pred} - u_{true}\|_2}{\|u_{true}\|_2}$，用于量化重建与预测误差 |
| **Log-Spectral Distance (LSD)** | $\sqrt{\frac{1}{K}\sum_k (\log P(k) - \log \hat{P}(k))^2}$，衡量能谱保真度，反映多尺度纹理恢复能力 |
| **Jensen-Shannon Divergence (JSD)** | 衡量联合概率密度函数相似性，验证跨变量物理耦合是否被保留 |
| **Spatial/Temporal Correlation Length ($\xi_r$, $\xi_t$)** | 判断结构尺度和动力学时间尺度是否守恒 |

#### 对比基线方法
| 类别 | 基线模型 |
|------|--------|
| **Reduced-order modeling** | POD-GPR |
| **Convolutional models** | MLP-CNN |
| **Neural operators** | RecFNO |
| **Attention-based set models** | Senseiver |
| **Forecasting models** | Causal Transformer (Trans), FNO |

混合管道也进行了比较（如 GLU-LFD vs GLU-Trans vs Senseiver-Trans）。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 任务 | 数据集 | GLU 性能 | 最优基线 | 提升幅度 |
|------|--------|----------|-----------|------------|
| **Reconstruction** | Turbulent channel flow | ~0.04 (L²) | Senseiver (~0.07) | ↓ ~40% |
| **Reconstruction** | NOAA SST (64 sensors) | ~0.06 (L²) | FNO (~0.12) | ↓ ~50% |
| **Forecasting** | Collinear Re100 (chaotic) | Error < 0.1 at 240 steps | Trans > 0.5 | 显著延迟发散 |
| **Spectral fidelity** | Turbulent combustion | LSD ↓ 30–50% vs baselines | FNO/CNN | 更好恢复高频湍流 |
| **Cross-channel coupling** | Combustion (CO-Temp-Vel) | JSD ↓ 40% vs Senseiver | Senseiver | 更准确捕捉热释放机制 |

> 注：具体数值来自图2d/e、图5b/c及正文描述。

---

### ⚖️ 与基线方法的对比结果

#### ✅ 重建方面
- 在 **所有六个基准** 上，GLU 均取得最低相对 L² 误差（见 Fig. 2d）；
- 特别是在 **高场复杂度区域**（如剪切层、火焰锋面），uniform k-NN 方法误差急剧上升，而 GLU 的 adaptive selection 显示平坦误差曲线；
- 能谱分析显示 GLU 最接近 ground truth，尤其在 **高 wavenumber 区域**（细小涡旋）能量损失最小。

#### ✅ 预测方面
- 在 **Collinear Re40**（周期流）中，GLU-LFD 完美保持闭合轨道；Transformer 出现相位漂移；
- 在 **Collinear Re100**（混沌流）中，GLU-LFD 轨迹始终约束在正确吸引子内，而 Trans/FNO 快速偏离；
- 在 **reaction-diffusion** 中，GLU-LFD 抑制了前沿误差放大，保持波形清晰，时空相关长度更接近真实值（Fig. 4b）。

---

### 🔬 消融实验结果（Ablation Studies）

| 模型变体 | 性能影响 | 发现 |
|---------|--------|------|
| **GLU (Global-only)** | 丢失小尺度结构，误差↑↑ | 证明 local tokens 不可替代 |
| **GLU (Uniform neighborhood)** | 复杂区误差显著升高 | 证明 importance-driven selection 至关重要 |
| **GLU-Trans (no LFD)** | 长期预测快速发散 | 证明 LFD 结构是稳定性的关键 |
| **Fixed importance (heuristic)** | 重建质量下降 | 证明 uncertainty-driven learning 的有效性 |

> 图3b–c 和图4a 提供了直观可视化证据。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **统一状态表示至关重要**：将 reconstruction 与 forecasting 构建在同一 latent space 下，可有效阻断误差传递链。
2. **Global + Local + Uncertainty 三元融合优于单一范式**：
   - Global 提供一致性；
   - Local 保障细节；
   - Uncertainty 实现智能资源分配。
3. **Learned importance scores 具有物理可解释性**：
   - 在分离流中聚焦 wake 区域；
   - 在 SST 中关注海岸线与洋流锋面；
   - 在均匀系统中趋于平滑分布。
4. **LFD 架构兼具效率与稳定性**：
   - 计算复杂度近似线性于传感器数量；
   - 显著延缓误差积累，适用于工业级 long-horizon rollout。
5. **GLU 是 physics-consistent 的 inference engine**：
   - 不仅还原单个场，还保持跨变量的 thermodynamic coupling（如燃烧中的放热反馈）。

---

### ⚠️ 方法的局限性
- **依赖预定义传感器位置**：目前假设传感器位置固定且已知，尚未扩展至主动感知或传感器布局优化；
- **实时性未充分验证**：尽管内存占用更低，但端到端推理延迟未在嵌入式平台测试；
- **泛化能力限于同构系统**：跨工况迁移（如不同 Re 数之间）需重新训练或微调；
- **生成不确定性仍为 aleatoric**：当前 uncertainty modeling 主要反映数据噪声，缺乏对模型不确定性的显式建模（epistemic uncertainty）。

---

### 🔮 未来工作方向
1. **结合 data assimilation**：将 GLU 作为 observation operator 或 state prior，集成进 Kalman filter 或 variational DA 流程；
2. **引入 generative modeling**：结合 diffusion models 或 flow matching，实现带置信区间的概率预测；
3. **支持 online adaptation**：开发增量学习策略，使 GLU 能适应缓慢演化的系统特性（如设备老化）；
4. **面向控制闭环应用**：探索如何利用 GLU 的 structured latent space 进行 policy learning 与 feedback control；
5. **硬件协同优化**：针对边缘部署进一步压缩模型，支持 real-time digital twin on IoT devices。

---

## 总结

GLU 提出了一种新颖且实用的 **spatiotemporal digital twin 范式**，通过 **Global-Local-Uncertainty fusion** 实现了：
- 高保真稀疏重建，
- 稳定长程预测，
- 多尺度与多物理耦合保持，
- 可扩展的计算效率。

它不仅在多个 challenging benchmark 上实现了 SOTA 性能，更重要的是提供了一个**统一、结构化、可解释、可扩展**的状态表示框架，为下一代智能工业系统（如能源、制造、交通）中的 real-time monitoring 与 proactive control 奠定了坚实基础。

</details>

---

### 10. [Knowledge Distillation for Efficient Transformer-Based Reinforcement Learning in Hardware-Constrained Energy Management Systems](https://arxiv.org/abs/2603.26249)

**Authors**: Pascal Henrich, Jonas Sievers, Maximilian Beichter, Thomas Blank, Ralf Mikut, Veit Hagenmeyer  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.26249v1  

#### Abstract
Transformer-based reinforcement learning has emerged as a strong candidate for sequential control in residential energy management. In particular, the Decision Transformer can learn effective battery dispatch policies from historical data, thereby increasing photovoltaic self-consumption and reducin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Knowledge Distillation for Efficient Transformer-Based Reinforcement Learning in Hardware-Constrained Energy Management Systems*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文针对**住宅级能源管理系统**（HEMS）中，基于 **Decision Transformer**（DT）的强化学习（RL）策略因模型过大而难以部署在**资源受限的嵌入式硬件**上的问题。尽管 DT 在电池调度等任务中表现出色，但其高计算开销（参数量、内存占用、推理延迟）限制了实际应用。

### ✅ 提出的新方法与创新
提出了一种**响应式知识蒸馏**（Response-Based Knowledge Distillation, KD）框架，用于压缩大型 DT 教师模型（teacher）到小型学生模型（student），同时保留甚至提升控制性能。

#### 主要创新点包括：
- **首次系统性地将 KD 应用于 DT 架构的电池调度任务**，填补了该领域研究空白。
- 设计适用于**连续动作空间**的 KD 框架，直接匹配教师输出的连续动作 logits（而非分类概率），采用 Smooth L1 Loss 进行训练。
- 探索了**离线自蒸馏**（offline self-distillation）作为正则化手段，在不减小模型尺寸的情况下提升泛化能力。
- 验证了 KD 不仅是压缩工具，更是一种**性能增强机制**，能超越原始教师模型的表现。

### ✅ 相比现有方法的优势
| 对比维度 | 传统方法（如 DDPG、Rule-based） | 本文方法（KD + DT） |
|--------|-------------------------------|------------------|
| 控制稳定性 | 易受“致命三元组”影响（bootstrapping + function approximation + off-policy） | 通过序列建模避免显式值函数估计，更稳定 |
| 数据效率 | 通常需在线交互或高质量行为策略 | 支持完全**offline RL**，利用历史轨迹即可训练 |
| 可部署性 | 小模型易部署但性能有限 | 大幅压缩后仍保持高性能，适合边缘设备 |
| 性能上限 | 受限于策略复杂度 | 利用大模型先验知识，小模型也能逼近最优 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **Ausgrid Solar Home Electricity Dataset**：包含20个住宅建筑的高分辨率用电负荷与光伏（PV）发电数据。
- **SMARD 数据库**：提供德国/奥地利/卢森堡地区（DE/AT/LU）的实时电价数据。
- 所有实验基于这20栋建筑进行跨建筑评估，确保结果具有代表性。

### ⚙️ 实验设置
- **环境建模**：将电池调度问题建模为有限时域 MDP，目标是最小化电网购电成本。
- **状态空间** $ S_t $：包含当前电池 SoE、未来24步（12小时）的负荷与价格预测。
- **动作空间** $ A_t $：连续变量，表示充放电功率（受限于电池额定功率）。
- **奖励函数**：经济导向，综合考虑电价、馈电补贴及违反约束的惩罚项。

### 🔍 蒸馏流程
1. 先训练三种不同规模的 DT 教师模型（Small, Medium, Large）。
2. 固定教师模型参数，用其生成的动作 logits 作为监督信号训练更小的学生模型（Tiny, Mini, Small）。
3. 自蒸馏实验：Small 教师 → Small 学生，验证正则化效果。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Mean Electricity Cost (€)** | 主要性能指标，越低越好 |
| **Total Parameters** | 模型大小，衡量可部署性 |
| **Memory Footprint (MB)** | 推理时内存占用 |
| **Inference Time (ms)** | 单步推理延迟 |
| **Performance Gain (%)** | 相对于教师的成本降低幅度 |

### 🆚 基线方法对比
| 基线 | 类型 | 说明 |
|-----|------|------|
| **DDPG** | 强化学习基准 | 使用相同 offline 轨迹训练的行为克隆策略 |
| **Without Battery (WO Battery)** | 上界（最差情况） | 无储能系统的电费支出 |
| **MILP (Mixed Integer Linear Programming)** | 下界（理论最优） | 假设完美预测下的全局最优解，不可实时实现 |
| **Rule-based Controller** | 传统控制策略 | 如基于电价阈值的简单充放电规则 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 模型配置 | 平均电费 (€) | 参数量减少 | 内存减少 | 推理时间减少 |
|---------|--------------|------------|-----------|---------------|
| **Medium DT (Teacher)** | 201.30 | — | — | — |
| **KD (Medium → Small)** | **200.28** | ↓78% | ↓64% | ↓38% |
| **KD (Medium → Tiny)** | 200.70 | ↓**96%** | ↓**90%** | ↓**63%** |
| **DDPG (Baseline)** | 202.43 | — | — | — |
| **MILP (Optimal)** | 168.08 | — | — | — |
| **WO Battery** | 220.25 | — | — | — |

> ✅ **最佳配置**：Medium 教师 → Small 学生，平均成本最低（200.28 €），优于所有其他学习方法。

### 🔁 与基线方法对比结果
- **相比 DDPG**：KD 模型平均节省约 **2.15 €**（提升 ~1.06%），且在 **60% 的建筑中表现最优**（见 Figure 9）。
- **相比未压缩 DT**：KD 模型不仅更轻量，还实现了 **~1% 的性能增益**，表明蒸馏过程具有正则化作用。
- **极端压缩下仍有效**：Tiny 学生（仅38万参数）在 **55% 的建筑中优于 DDPG**，证明高度压缩后的模型依然可靠。

### 🔬 消融实验结果
#### （1）模型大小对 DT 性能的影响（Figure 7）
- 存在**非单调关系**：Medium 模型表现最好（201.30 €），Large 模型反而更差（203.88 €），说明存在过拟合风险。
- 表明：更大 ≠ 更好，适配任务复杂度的中等容量模型更具优势。

#### （2）不同教师对学生性能的影响（Figure 8）
- **Medium 教师**蒸馏效果最佳：
  - → Small 学生：成本下降 **1.02 €**
  - → Mini 学生：下降 **0.68 €**
  - → Tiny 学生：下降 **0.60 €**
- **Large 教师**虽能力强，但因其自身性能较差，导致蒸馏收益有限。
- **Self-Distillation**（Small → Small）也带来 **0.57 € 成本下降**，证实 KD 具备独立于压缩的正则化价值。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KD 能显著压缩 DT 模型而不损失性能**：
   - 最高达 **96% 参数减少**、**90% 内存压缩**、**63% 推理加速**。
   - 特别适用于资源受限的家庭控制器（如 Raspberry Pi、ARM MCU）。

2. **蒸馏模型常优于教师模型**：
   - 在多数配置下，学生模型的平均电费低于对应教师，最高提升达 **1%**。
   - 原因可能是 KD 缓解了原始训练轨迹中的噪声或次优行为，提升了泛化能力。

3. **Medium 规模 DT 是最佳教师选择**：
   - Large 模型因过拟合导致性能下降，不适合作为教师。
   - Medium 模型兼具表达力与泛化性，是理想的蒸馏源。

4. **自蒸馏也是一种有效的正则化技术**：
   - 即使不压缩模型，KD 也能提升性能，说明其不仅是迁移工具，更是优化策略。

### ⚠️ 方法的局限性
- **依赖高质量教师模型**：若教师本身性能不佳（如 Large DT），蒸馏效果受限。
- **仅考虑动作模仿**：未引入中间层特征匹配（feature imitation），可能遗漏潜在知识。
- **静态蒸馏**：教师固定，无法动态调整监督信号强度。
- **未测试多设备协同场景**：目前仅聚焦单电池调度，扩展至 HVAC、EV 等多柔性负载尚待研究。

### 🔮 未来工作方向
1. **探索多阶段蒸馏策略**（progressive distillation）或渐进式架构搜索。
2. 引入**中间层知识迁移**（hint-based KD）以进一步提升小模型表现。
3. 结合**联邦学习**框架，在保护隐私的前提下实现跨用户知识共享与蒸馏。
4. 扩展至**多能系统**（multi-energy systems）与**多智能体 DT** 架构。
5. 研究**在线蒸馏**机制，允许教师与学生共同演化。

---

## ✅ 总结
本论文成功展示了 **Knowledge Distillation 是连接高性能 DT 控制器与资源受限硬件之间的关键桥梁**。通过系统性的实验验证，作者证明了：
> **KD 不仅能实现极致的模型压缩（96% 参数削减），还能反向提升控制性能，使小型模型在真实家庭能源管理场景中具备实用性和竞争力。**

这项工作为将先进 AI 技术落地于边缘侧能源系统提供了坚实的技术路径和实证支持。

</details>

---

### 11. [AutoB2G: A Large Language Model-Driven Agentic Framework For Automated Building-Grid Co-Simulation](https://arxiv.org/abs/2603.26005)

**Authors**: Borui Zhang, Nariman Mahdavi, Subbu Sethuvenkatraman, Shuang Ao, Flora Salim  
**Category**: cs.AI  
**Published**: 2026-03-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.26005v1  

#### Abstract
The growing availability of building operational data motivates the use of reinforcement learning (RL), which can learn control policies directly from data and cope with the complexity and uncertainty of large-scale building clusters. However, most existing simulation environments prioritize buildin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AutoB2G: A Large Language Model-Driven Agentic Framework For Automated Building-Grid Co-Simulation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前建筑能源系统仿真环境（如 CityLearn、Sinergym）主要关注**建筑侧性能指标**（如能耗、成本），缺乏对电力系统层面影响的系统性评估。同时，现有仿真流程依赖大量手动配置和编程，技术门槛高，限制了其广泛应用。

此外，虽然已有研究尝试引入 LLM 来自动化仿真任务（如 Eplus-LLM、OpenCEM），但在复杂、多模块耦合的 **building-grid co-simulation** 场景中，仍面临以下挑战：
- LLM 缺乏对模拟函数实现上下文的先验知识；
- 复杂工作流难以保证模块间的依赖正确性和执行顺序；
- 自然语言到可执行代码的转换可靠性低。

---

### 🚀 提出的新方法与创新思路

本文提出 **AutoB2G** —— 一个基于大语言模型（LLM）驱动的智能体框架，支持从自然语言任务描述端到端自动生成并执行 building-grid 协同仿真。

#### 主要创新点包括：

1. **构建了支持 B2G interaction 的协同仿真环境**
   - 在 CityLearn V2 基础上集成 Pandapower 构建电网模型，支持电压控制、N-1 resilience、thermal loading 等 grid-side metrics 的量化分析。
   - 支持灵活配置 reward 函数（如包含电压偏差惩罚项），实现 grid-aware 控制策略训练。

2. **提出 DAG-based Agentic Retrieval 机制**
   - 将仿真功能模块组织为有向无环图（DAG），显式编码模块间输入输出依赖关系和执行顺序。
   - 利用 LLM 驱动的 agent 进行结构化检索，确保生成的工作流满足拓扑排序约束。
   - 引入验证器（Validator）检测缺失依赖，并通过反馈迭代修复，提升工作流完整性与结构性一致性。

3. **采用 SOCIA 框架实现闭环自动优化**
   - 使用 SOCIA（Simulation Orchestration for Computational Intelligence with Agents）多智能体框架，将仿真程序视为优化变量。
   - 通过 **Textual-Gradient Descent (TGD)** 机制实现迭代式代码生成、执行、评估与修复：
     - 当前版本代码执行失败 → 提取错误日志 → 转换为自然语言反馈（textual gradient）→ 指导下一轮代码修补。
   - 实现无需人工干预的全自动仿真流水线构建。

---

### 🔍 相比现有方法的优势

| 维度 | 现有方法（如 GridLearn, CityLearn） | AutoB2G |
|------|-------------------------------|--------|
| 配置方式 | 手动编码，需深厚编程能力 | **自然语言驱动**，降低使用门槛 |
| 电网评估 | 功能有限或版本不兼容（如 GridLearn 不支持 CityLearn V2） | 完整支持多种 grid-side metrics（电压、负载、短路电流、N-1 安全性） |
| 工作流自动化 | 无 | 全流程自动化：从任务理解 → 模块选择 → 代码生成 → 执行 → 反馈修正 |
| 错误处理能力 | 静态脚本，出错需人工调试 | 多轮迭代修复（via TGD），具备自我纠正能力 |
| 可扩展性 | 固定架构 | 模块化设计，易于接入新工具（如其他 building/power simulators） |

---

## 2. 核心实验方法和设置

### 📚 数据集与平台
- **Building 模拟**：基于 **CityLearn V2**，使用 End-Use Load Profiles for the U.S. Building Stock 数据集生成定制化住宅负荷数据。
- **Grid 模拟**：采用标准 **IEEE 33-bus distribution network**（来自 [6]），用于测试电压调节与安全性。
- **控制算法**：使用 Soft Actor-Critic (SAC) 等 RL 算法进行控制器训练。
- **仿真引擎组合**：
  - EnergyPlus：高保真建筑热力学建模
  - CityLearn：建筑集群控制与 reward 设计
  - Pandapower：潮流计算与 grid analysis

---

### ⚙️ 实验设置

#### 任务复杂度分级（Three Levels）
| 类型 | 描述 | 示例 |
|------|------|------|
| **Simple Tasks** | 单阶段流程，仅涉及基本仿真运行 | “使用规则控制器在默认数据集上运行仿真并报告KPI” |
| **Medium Tasks** | 多阶段流程，含模型训练与初步评估 | “训练 SAC 模型 10 轮，在 IEEE 33-bus 上评估，保存电压与线路负载” |
| **Complex Tasks** | 多模块、多目标比较任务 | “对比集中式与分布式控制策略，加入无功补偿元件，执行 N-1 分析，输出图表与 CSV” |

#### 基线方法对比
共比较四种配置：
1. **LLM**：直接调用 LLM，提供完整 codebase 输入
2. **LLM + AR**：LLM + Agentic Retrieval（基于 DAG 检索相关模块）
3. **SOCIA**：SOCIA 框架但无 retrieval
4. **SOCIA + AR**：所提完整方法（本文核心）

所有实验均基于 **OpenAI GPT-5 API**，保持一致 prompt engineering 和参数设置。

---

### 📊 评估指标

| 指标 | 定义 | 说明 |
|------|------|------|
| **Task Success Rate** | 成功完成任务的比例（布尔判断） | 主要衡量是否能产出可执行且符合要求的结果 |
| **Code Score** | $ \frac{N_{\text{correct}}}{N_{\text{total}} + N_{\text{extra}}} $ | 衡量生成代码的质量：<br>- 正确实现的必需组件数 / （总需求数 + 多余组件数）<br>- 数值越高表示更准确、冗余更少 |

---

## 3. 主要实验结果和性能指标

### ✅ 任务成功率对比（Table 3）

| Method | Simple | Medium | Complex |
|--------|--------|--------|---------|
| LLM | 0.90 ± 0.08 | 0.77 ± 0.12 | 0.53 ± 0.19 |
| SOCIA | 0.93 ± 0.09 | 0.83 ± 0.12 | 0.73 ± 0.09 |
| LLM + AR | 0.97 ± 0.05 | 0.80 ± 0.08 | 0.67 ± 0.05 |
| **SOCIA + AR** | **1.00 ± 0.00** | **0.93 ± 0.09** | **0.83 ± 0.05** |

> 💡 结论：随着任务复杂度上升，**SOCIA + AR 显著优于其他方法**，尤其在复杂任务中领先 LLM 达 30 个百分点。

---

### ✅ 代码质量评分对比（Table 4）

| Method | Simple | Medium | Complex |
|--------|--------|--------|---------|
| LLM | 0.69 ± 0.08 | 0.66 ± 0.05 | 0.44 ± 0.08 |
| SOCIA | 0.82 ± 0.14 | 0.78 ± 0.05 | 0.67 ± 0.09 |
| LLM + AR | 0.72 ± 0.08 | 0.74 ± 0.07 | 0.73 ± 0.13 |
| **SOCIA + AR** | **1.00 ± 0.00** | **0.84 ± 0.07** | **0.88 ± 0.09** |

> 💡 结论：
- 即使 LLM 成功完成任务，也常因引入**冗余模块**导致 code score 下降；
- **Agentic Retrieval 显著减少无关信息干扰**，提高配置准确性；
- **SOCIA 的迭代修复机制有效提升最终代码质量**。

---

### 🔍 消融实验与关键发现

#### （1）DAG-based Retrieval 的作用
- 移除 retrieval 后，LLM 更容易遗漏关键模块或违反执行顺序（如先执行仿真再配置网络）；
- 加入 retrieval 后，模块选取准确率提升约 25%，尤其是在 complex tasks 中。

#### （2）TGD 机制的有效性（Figure 3）
- 图显示：经过 **2–3 轮迭代后，success rate 快速上升**；
- 第一次迭代修复主要解决语法错误和接口不匹配；
- 后续迭代聚焦于逻辑缺陷（如错误的 shunt placement、missing assertions）；
- 最终收敛至稳定可执行状态。

#### （3）Failure Case 分析
失败原因主要包括：
- **跨模块强耦合导致细微配置错误即引发崩溃**（如 bus index 写错一位）；
- **自然语言指令存在歧义**，模型需推断隐含条件（如“inject reactive power”未指明具体 bus）；
- 少数情况下 agent 错误地移除了必要模块（如误判 feature flag）。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **AutoB2G 实现了真正的自然语言驱动仿真自动化**  
   用户只需输入文本任务描述，即可自动生成完整的 building-grid co-simulation 流程，涵盖数据准备、控制器训练、电网仿真与多维度评估。

2. **DAG-based retrieval + SOCIA + TGD 形成可靠闭环**  
   - DAG 提供结构约束，防止无效组合；
   - SOCIA 多 agent 协同分工，提升鲁棒性；
   - TGD 实现类梯度式的程序优化，显著提高成功率。

3. **显著改善 grid-side performance**
   - 如 Figure 4 和 5 所示，RL 控制器能有效平抑电压波动，使电压分布更接近 1.0 p.u.；
   - 在过压场景下主动增加负荷以吸收多余功率，在欠压时削减负荷，体现 adaptive load modulation 能力。

4. **相比传统方法大幅降低技术门槛**
   - 研究人员可专注于科学问题而非编码细节；
   - 平台开发者可通过自然语言接口扩大用户群体。

---

### ⚠️ 局限性

1. **平台依赖性强**  
   当前仅在 CityLearn + Pandapower 上验证，尚未拓展至 EnergyPlus、OpenDSS 或 MATLAB/Simulink 等异构平台。

2. **对高度模糊或不完整指令容忍度有限**  
   若用户未明确指定关键参数（如 bus ID、reactive power 值），可能导致错误假设。

3. **计算开销较高**  
   多轮迭代机制虽提升成功率，但也增加了 LLM 调用次数和响应延迟。

4. **模块库需预先构建与维护**  
   DAG codebase 的覆盖范围直接影响系统能力，新增功能需人工开发并注册进图谱。

---

### 🔮 未来工作方向

1. **扩展平台兼容性**  
   接入更多 building 和 power system simulators（如 OpenStudio、DIgSILENT），增强通用性。

2. **增强语义理解能力**  
   引入对话机制，允许 LLM 主动澄清模糊需求，形成 human-in-the-loop 的交互式配置。

3. **构建可复用的功能模块库**  
   开发标准化接口规范，支持社区共享模块（类似 plugin system），促进生态发展。

4. **探索 zero-shot 跨域迁移能力**  
   研究如何让框架在未见过的 grid topology 或 building archetype 上仍能可靠工作。

5. **部署轻量化本地版本**  
   探索使用开源 LLM（如 Llama3、Qwen）替代闭源 API，降低成本并保护隐私。

---

> 📌 总结：AutoB2G 是迈向 **AI-native simulation infrastructure** 的重要一步，展示了 LLM-driven agentic system 在复杂工程系统自动化中的巨大潜力，特别是在 building-energy-grid 协同优化这一交叉领域具有广泛应用前景。

</details>

---

### 12. [Automatic feature identification in least-squares policy iteration using the Koopman operator framework](https://arxiv.org/abs/2603.26464)

**Authors**: Christian Mugisho Zagabe, Sebastian Petiz  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.26464v1  

#### Abstract
In this paper, we present a Koopman autoencoder-based least-squares policy iteration (KAE-LSPI) algorithm in reinforcement learning (RL). The KAE-LSPI algorithm is based on reformulating the so-called least-squares fixed-point approximation method in terms of extended dynamic mode decomposition (EDM...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Automatic feature identification in least-squares policy iteration using the Koopman operator framework*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在传统的 **Least-Squares Policy Iteration (LSPI)** 和 **Kernel-based LSPI (KLSPI)** 方法中，**特征函数（basis functions）或核函数（kernel）需要人工预先设定**，这带来了以下挑战：
- 缺乏系统性的选择标准；
- 手动设计的特征可能无法充分捕捉环境动态；
- 在不同任务中泛化能力差。

此外，KLSPI 虽然通过 ALD 方法实现字典稀疏化，但仍依赖于固定核函数，并且学习到的特征数量不可控、算法为 on-policy，导致计算开销大。

### 🚀 提出的新方法：KAE-LSPI
本文提出了一种新的强化学习算法——**Koopman Autoencoder-based Least-Squares Policy Iteration (KAE-LSPI)**，其核心思想是：
- 将 LSPI 中的值函数近似问题重新表述为基于 **Koopman operator 框架**的形式；
- 利用 **Koopman Autoencoder (KAE)** 自动从数据中学习一组最优的低维特征字典（dictionary of basis functions），无需手动设计；
- 将学习到的特征用于经典的 LSPI 框架中进行策略迭代。

该方法实现了 **自动特征识别（automatic feature identification）**，解决了传统方法对先验知识的依赖。

### 🔍 相比现有方法的优势
| 方面 | LSPI | KLSPI | KAE-LSPI（本文） |
|------|------|--------|------------------|
| 特征是否需预设 | 是（如多项式/RBF） | 否（但需预设核函数） | ❌ 完全无需预设 |
| 核函数是否需预设 | 不适用 | 是（如 RBF 核） | ❌ 无需 |
| 学习特征数量可控性 | 固定 | 动态增长，难以控制 | 可控（由网络结构决定） |
| 是否支持离线训练（off-policy） | 是 | 否（on-policy） | 是（初始数据来自随机策略） |
| 特征表达能力 | 有限 | 较强 | 更强（神经网络非线性映射） |

> ✅ **核心优势**：**避免了手工特征工程，同时保持了 LSPI 的高效性和收敛性。**

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与任务
实验在两类典型 MDP 任务上进行：

#### （1）Chain Walk Problem（离散状态空间）
- 状态数：20 或 50 个线性排列的状态
- 动作：左（L）、右（R）
- 转移概率：目标方向成功概率 0.9，反向失败概率 0.1
- 奖励函数：
  - Case 1 (n=20): $ r(1)=r(20)=1 $
  - Case 2 (n=50): $ r(10)=r(41)=1 $
- 数据生成：运行 1000 回合 × 20 步，共 20,000 条样本（随机策略采集）

#### （2）Inverted Pendulum Control（连续状态空间）
- 状态：角度 $\theta$ 和角速度 $\dot{\theta}$
- 动作：$\{-50, 0, +50\}$ 施加力（含噪声）
- 奖励：当 $|\theta| \leq \pi/2$ 时为 0，否则为 -1
- 目标：尽可能长时间维持倒立平衡（最长 3000 步）
- 动力学模型：使用 RK4 数值积分求解非线性微分方程
- 训练回合数：10 ~ 1000 不等，每轮最多 20 步
- 测试：200 个靠近平衡点的初始状态，最大步长 3000

---

### 🧪 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| **Baseline 方法** | - Classical LSPI [6]<br>- Kernel-based LSPI (KLSPI) [14] |
| **特征设置** | - LSPI：固定多项式或 RBF 特征<br>- KLSPI：RBF 核 + ALD 字典压缩（阈值 0.001）<br>- KAE-LSPI：通过 KAE 学习特征（数量可调） |
| **KAE 结构** | Encoder: 如 [256,128,64] → Latent dim=k → Decoder 对称结构<br>激活函数：ReLU（末层前两层用 tanh） |
| **损失函数** | $ \mathcal{L}_{\text{tot}} = \lambda_{\text{rec}}\mathcal{L}_{\text{rec}} + \lambda_{\text{pred}}\mathcal{L}_{\text{pred}} + \lambda_{\text{dyn}}\mathcal{L}_{\text{dyn}} $<br>权重 $(\lambda_{\text{rec}}, \lambda_{\text{pred}}, \lambda_{\text{dyn}}) = (1,1,0.1)$ |
| **优化器** | Adam，学习率 $10^{-4}$，batch size=256 |
| **评估指标** | - 收敛至最优/近优策略所需的迭代次数<br>- 控制性能（平均维持步数）<br>- 学习到的特征数量对比 |

---

## 3. 主要实验结果和性能指标

### 📊 Chain Walk 实验结果

#### （1）n = 20 状态
| 方法 | 固定/学习特征数 | 收敛所需迭代数 | 是否达到最优策略 |
|------|------------------|------------------|--------------------|
| LSPI | 8（多项式） | 3 | ✅ 是 |
| KLSPI | 40（RBF） | 1 | ✅ 是 |
| **KAE-LSPI** | **15（学习得到）** | **3** | ✅ 是 |

> ⚠️ 尽管 KLSPI 收敛更快，但使用了更多特征；KAE-LSPI 以合理特征数量实现相同性能。

#### （2）n = 50 状态
| 方法 | 特征数 | 迭代次数 | 性能 |
|------|--------|----------|------|
| LSPI | 22（RBF） | 4 | 达到近优策略 |
| KLSPI | 100（学习得到） | 2 | 达到近优策略 |
| **KAE-LSPI** | **45（学习得到）** | **4** | **达到近优策略** |

> ✅ KAE-LSPI 在仅使用约一半特征的情况下，达到了与其他方法相当的性能。

---

### 📈 Inverted Pendulum 控制性能

| 方法 | 学习特征数 | 平均平衡步数（测试阶段） | 性能趋势 |
|------|------------|----------------------------|---------|
| LSPI | 30（RBF） | 随训练 episode 增加而上升 | 中等 |
| **KAE-LSPI** | **46（学习得到）** | **与 LSPI 相当甚至略优** | 更稳定 |

- 图 4 显示：随着训练 episode 增加，KAE-LSPI 的平均控制步数迅速提升并趋于稳定，表现与 LSPI 相当。
- **关键发现**：尽管没有人为设计特征，KAE-LSPI 仍能学到足够有效的表示来完成复杂控制任务。

---

### 🔍 消融实验与分析（隐含）
虽然未明确列出消融实验，但从设计中可推断以下结论：
- **特征学习的有效性**：KAE 成功提取出可用于 LSPI 的低维线性可预测特征；
- **Koopman 动态约束的重要性**：引入 $ \mathcal{L}_{\text{dyn}} $ 损失确保特征空间中的演化接近线性，这对后续 LSPI 收敛至关重要；
- **off-policy 应用可行性**：即使特征由随机策略数据训练而来，也能迁移到策略改进过程中，体现泛化能力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KAE 可有效用于自动特征学习**：无需任何先验知识即可从原始状态-动作对中学习适合值函数逼近的特征；
2. **KAE-LSPI 具备与经典方法相当的收敛性能**：在离散与连续任务中均能达到最优或近优策略；
3. **特征数量更可控且合理**：相比 KLSPI 动态膨胀的字典规模，KAE-LSPI 的特征维度由网络结构决定，更具实用性；
4. **框架统一性强**：将 LSPI 与 Koopman operator 框架结合，建立了 RL 与动力系统理论之间的桥梁。

---

### ⚠️ 局限性
1. **特征数量仍需预设**：虽然不再需要设计具体形式，但 latent 维度 $k$ 需手动指定；
2. **训练成本较高**：KAE 需要额外训练时间（尤其在高维状态空间）；
3. **缺乏理论收敛保证**：目前尚无关于 KAE 学习误差如何影响 LSPI 收敛性的理论分析；
4. **当前为 offline/off-policy 架构**：未实现实时更新机制，难以应对动态变化环境。

---

### 🔮 未来工作方向
1. **自动化特征维度选择**：探索自适应 latent dimension selection 方法（如变分方法或稀疏正则化）；
2. **Online & On-policy 版本**：借鉴 KLSPI 思路，在每次策略更新后重新采样并学习新特征；
3. **理论分析**：建立 KAE 重建误差、预测误差与 LSPI 收敛性之间的误差传播边界；
4. **Regret Bound 推导**：基于 Koopman 框架已有的误差界，发展 RL 中的 regret bounds；
5. **扩展至高维视觉输入**：将 KAE 替换为 CNN/ViT 架构，应用于图像观测任务。

---

## ✅ 总结
本文提出的 **KAE-LSPI** 是一种将 **Koopman operator 理论** 与 **深度表示学习** 相结合的新型 **value function approximation** 方法。它成功实现了 **自动特征识别**，克服了传统 LSPI 类方法对手工特征的依赖，在多个基准任务上表现出与经典方法相当甚至更优的性能，具有良好的应用前景和发展潜力。

</details>

---

### 13. [LLM Benchmark-User Need Misalignment for Climate Change](https://arxiv.org/abs/2603.26106)

**Authors**: Oucheng Liu, Lexing Xie, Jing Jiang  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.26106v1  

#### Abstract
Climate change is a major socio-scientific issue shapes public decision-making and policy discussions. As large language models (LLMs) increasingly serve as an interface for accessing climate knowledge, whether existing benchmarks reflect user needs is critical for evaluating LLM in real-world setti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LLM Benchmark-User Need Misalignment for Climate Change 论文总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对当前 **LLM 在气候变化领域评估基准（benchmark）与真实用户需求之间存在严重错位**的问题展开研究。尽管 LLM 越来越多地被公众用于获取气候知识，但现有的 LLM 评估基准大多基于教科书式的科学事实问答，未能反映现实世界中用户多样化的信息需求，如政策建议、行动指南、写作支持等。

### 提出了什么新方法或新思路
论文提出了两个核心创新：

1. **Proactive Knowledge Behaviors Framework（主动知识行为框架）**  
   该框架将知识交互分为两类行为：**知识寻求（Asking）** 和 **知识提供（Guiding/Informing）**，并区分了三类主体：人类知识寻求者（如公众）、人类知识提供者（如科学家、记者）和 AI 知识提供者（即 LLM）。通过这一框架，作者系统性地比较了不同主体间的行为模式，为从人类知识交互中寻找高质量参考数据提供了理论依据。

2. **Topic-Intent-Form 三维分类法（Taxonomy）**  
   作者构建了一个细粒度的三维分类体系，用于量化分析知识需求：
   - **Topic（主题）**：涵盖五大类共 25 个子主题，如 `A1. Atmospheric Science`, `E1. Climate Policy`, `E2. Energy Transition` 等。
   - **Intent（意图）**：8 大类 29 子类，包括 `INTENT_1a. Fact Lookup`, `INTENT_3a. General Advice`, `INTENT_6a. Operational Writing` 等。
   - **Form（形式）**：期望的回答形式，如 `FORM_2a. Concise Paragraph`, `FORM_3a. Item List`, `FORM_7a. Multiple Choice` 等。  
   此外，每个意图还映射到扩展的 **Bloom’s Taxonomy**（Factual, Conceptual, Procedural, Metacognitive），揭示所需的知识层级。

### 相比现有方法的优势
- **更贴近真实场景**：相比传统以“事实核查”或“选择题”为主的 benchmark（如 ClimaQA），本方法直接从真实用户查询中挖掘需求分布，更具现实代表性。
- **可迁移性强**：提出的框架和分类法具有通用性，可推广至其他社会科学研究领域（如公共卫生、能源转型）。
- **指导性强**：不仅发现问题，还提出解决方案（如利用 IPCC 报告作为高质量知识源），为 LLM 开发、RAG 系统设计和训练数据构建提供 actionable guidance。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
论文整合了 **11 个数据集**，分为三大类：

| 类别 | 数据集 | 描述 |
|------|-------|------|
| **Human-to-AI Queries** | WildChat, LMSYS-Chat-1M, ClimateQ&A | 来自真实用户与 LLM 的对话日志，代表真实用户需求。总计约 6,000 条气候相关查询。 |
| **Human-to-AI Guidance Knowledge** | ClimaQA-Gold, ClimaQA-Silver | 当前主流的 LLM 气候知识评测基准，由 LLM 基于教科书生成，代表现有评估标准。 |
| **Human-to-Human Knowledge Provision** | Reddit（4 个子版块）, SciDCC（新闻）, IPCC AR6（报告） | 人类之间的知识交互数据，用于验证是否可作为 LLM 评估的参考来源。 |

> 注：所有数据均经过清洗、去重，并使用 LLM 进行 Topic-Intent-Form 标注。

### 实验设置和评估指标
- **标注方法**：使用 LLM（如 GPT-4.1-mini, GPT-5-mini）进行自动标注，结合人工校验确保质量（human verification 显示 Jaccard 相似度达 0.7+）。
- **表示方法**：每条数据表示为 **加权向量**（Topic: 25维, Intent/Form: 各38维），权重按标签排序分配（如第1标签权重最高）。
- **评估指标**：使用 **余弦相似度（cosine similarity）** 比较不同数据集在 Topic、Intent、Form 上的分布一致性。

### 基线方法对比
- **主要对比对象**：将 **真实用户查询（Real-World Group）** 与 **现有 benchmark（LLM Benchmark Group）** 进行对比。
- **次要对比**：比较人类对人提问（Reddit）与人类对 AI 提问（WildChat/LMSYS）的相似性，以及人类知识输出（IPCC）与用户需求的一致性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）Topic 分布对比
- **真实用户查询** 的主题分布更广，尤其关注：
  - `E1. Climate Policy, Governance & Finance Mechanism`
  - `E2. Energy Transition`
  - `D4. Public Awareness, Communication & Community Engagement`
- **现有 benchmark** 高度集中于：
  - `A1. Atmospheric Science & Climate Processes`（占比高出真实需求约 **63%**）
- **Topic 余弦相似度**：
  - Real-World 内部相似度：**0.94–0.98**
  - Benchmark 内部相似度：**≈1.00**
  - Real vs Benchmark 相似度：**<0.4** → 表明显著错位。

#### （2）Intent 与 Form 对比
| 维度 | Real-World 主导 | Benchmark 主导 | 差异幅度 |
|------|----------------|----------------|----------|
| **Intent** | `INTENT_3a. General Advice`, `INTENT_6a. Operational Writing` | `INTENT_1a. Fact Lookup`（占 **60%**） | 差距达 **~40%** |
| **Form** | `FORM_2a. Concise Paragraph`, `FORM_3a. Item List` | `FORM_1a. Concise Value(s)/Entity(ies)`, `FORM_7a. Multiple Choice` | 最大差距 **37%** |

> 即使在相同主题（如 A1）下，Intent 和 Form 的差异依然显著（见 Figure 6），说明错位并非仅由主题偏差导致。

#### （3）人类-人类 vs 人类-AI 知识需求对比
- **Topic 相似度**：人类对 LLM 提问 vs 人类对 Reddit 提问 ≈ **0.85–0.98**
- **Intent/Form 相似度**：同样高度一致
- **结论**：人类在向人类或 AI 寻求知识时，其需求模式基本一致。

#### （4）IPCC 报告与用户需求匹配度
- **IPCC AR6** 与各用户数据集的 Topic 相似度高达 **0.84–0.88**
- **SciDCC（新闻）** 与用户需求相似度较低（~0.5）
- **结论**：权威科学报告（IPCC）比媒体报道更能反映公众真实关切。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **存在系统性错位（Misalignment）**：  
   当前 LLM 气候知识评估基准严重偏向基础科学事实（A1 类），忽视了用户对政策、转型、行动建议等应用型知识的需求。

2. **人类-AI 与人类-人类知识行为高度相似**：  
   用户在向 LLM 或人类专家提问时，表现出一致的主题偏好和回答形式期待，这为从人类知识交互中构建 LLM 评估基准提供了合法性。

3. **IPCC 报告是高质量知识源**：  
   尽管 IPCC 报告常被视为“过于专业”，但其主题分布与公众真实需求高度一致，适合作为 RAG 系统的检索语料或 benchmark 的采样来源。

4. **用户需要高阶认知支持**：  
   真实用户不仅需要 Factual 和 Conceptual 知识，更依赖 **Procedural**（如何做）和 **Metacognitive**（如何规划、决策）知识，这对 LLM 的能力提出了更高要求。

### 方法的局限性
- **数据来源有限**：人类-人类提问仅来自 Reddit，缺乏来自社交媒体（如 Twitter/X）或其他平台的数据。
- **语言偏倚**：数据以英语为主，可能无法代表非英语用户的气候信息需求。
- **主观性风险**：Topic Taxonomy 的构建涉及人工干预，可能存在主观判断影响。
- **被动行为未覆盖**：研究聚焦“主动行为”（如提问、撰写报告），未分析被动传播（如转发、评论）中的知识流动。

### 未来工作方向
- **构建新型 benchmark**：基于 Topic-Intent-Form 分布，设计更贴近真实需求的气候 LLM 评测集。
- **优化 RAG 系统**：以 IPCC 报告为核心构建最小化知识库，并补充 `D4. Public Awareness` 等薄弱领域的外部数据。
- **改进 LLM 训练**：根据真实需求分布调整训练数据配比，增强模型在 `Advice`, `Writing`, `Planning` 等任务上的表现。
- **跨领域推广**：将 Proactive Knowledge Behaviors Framework 应用于其他社会科技议题（如人工智能伦理、公共卫生危机）的研究。

> **代码开源**：https://github.com/OuchengLiu/LLM-Misalign-Climate-Change

</details>

---

### 14. [In-Context Molecular Property Prediction with LLMs: A Blinding Study on Memorization and Knowledge Conflicts](https://arxiv.org/abs/2603.25857)

**Authors**: Matthias Busch, Marius Tacke, Sviatlana V. Lamaka, Mikhail L. Zheludkevich, Christian J. Cyron, Christian Feiler, Roland C. Aydin  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.25857v1  

#### Abstract
The capabilities of large language models (LLMs) have expanded beyond natural language processing to scientific prediction tasks, including molecular property prediction. However, their effectiveness in in-context learning remains ambiguous, particularly given the potential for training data contami...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*In-Context Molecular Property Prediction with LLMs: A Blinding Study on Memorization and Knowledge Conflicts*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对当前 **Large Language Models (LLMs)** 在分子性质预测任务中表现优异的现象，提出了一个根本性质疑：  
这些“成功”是源于模型真正掌握了**结构-性质关系**并进行**in-context learning (ICL)**，还是仅仅因为训练数据中已经包含了这些基准数据集（如 MoleculeNet），从而导致了**数据污染（data contamination）** 和**记忆化（memorization）**？

这一问题在科学领域尤为关键，因为若模型只是“记住”了答案，而非学习推理，则其在新化学空间中的泛化能力将受到严重质疑。

### 🚀 提出的新方法与新思路
作者提出了一套系统的 **六级盲化框架（six-level blinding framework）**，通过逐步移除或变换输入信息来隔离 LLM 的不同能力来源：

| 盲化等级 | 关键操作 | 目的 |
|--------|--------|------|
| Level 1 (Specific) | 完整化学上下文（属性名、原始值、原始 SMILES） | 测试综合性能（含记忆+知识+ICL） |
| Level 2 (Specific-Transformed) | 属性值数学变换（反转+归一化） | 阻止直接记忆，保留排序关系 |
| Level 3 (Generic) | 属性名称抽象为“molecular property” | 移除特定先验知识 |
| Level 4 (Generic-Transformed) | 同上 + 值变换 | 进一步削弱记忆可能 |
| Level 5 (Agnostic) | 输入术语去化学化：“molecule”→“sample”，“SMILES”→“structure string” | 消除领域语义引导 |
| Level 6 (Agnostic-Transformed) | 上述全部 + SMILES 字符替换（保持语法结构不变） | 最大程度盲化，仅依赖模式识别 |

此外，结合 **0-shot、60-shot、1000-shot** 设置，系统控制 in-context 示例数量，分析信息量对性能的影响。

### 🔍 相比现有方法的优势
- **首次系统区分 LLM 能力层级**：明确划分出 *direct memorization*, *learned relationships*, *chemical ICL*, *general ICL* 四种能力。
- **对抗记忆化的强鲁棒性设计**：不仅变换标签值，还变换 SMILES 表示本身，防止模型通过熟悉结构“认出”分子。
- **揭示 prior knowledge 的双刃剑效应**：发现先验知识有时会干扰 in-context 学习，尤其当样本数不足时。
- **提供可复现的评估范式**：开源代码与数据，推荐将“盲化测试”作为科学任务部署前的“sanity check”。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
三个来自 **MoleculeNet** 的标准分子性质预测数据集：

| 数据集 | 性质 | 样本数 | 描述 |
|-------|------|--------|------|
| **Delaney** | 水溶性（aqueous solubility, logS） | 1,128 | 实验测量，常用作溶解度基准 |
| **Lipophilicity** | 脂水分配系数（logD at pH 7.4） | 4,200 | 实验测得，反映亲脂性 |
| **QM7** | 原子化能（atomization energy, kcal/mol） | ~6,834 | DFT 计算得到，量子力学性质 |

所有输入均为 **SMILES string**，目标为连续数值。

### ⚙️ 实验设置
- **LLM 模型族**：共评测 **9 个变体**，覆盖三大主流家族：
  - **GPT-4.1**（nano, mini, base）
  - **GPT-5**（nano, mini, base）
  - **Gemini 2.5**（flash-lite, flash, pro）
- **In-context 设置**：
  - **0-shot**：无训练样例 → 测模型预训练知识
  - **60-shot**：少量样例 → 测试典型 few-shot 场景
  - **1000-shot**：大量样例 → 探索长上下文 ICL 能力
- **提示策略**：采用两阶段 **pre-analysis prompting**：
  1. 分析阶段：让模型识别功能团、相似分子对等
  2. 预测阶段：基于加权平均输出结果
- **重复次数**：每配置运行 2 次，每次使用 150 测试样本以控制成本

### 📊 评估指标
- 主要指标：**Pearson correlation coefficient (r)**  
  （因 MAE/RMSE 受变换尺度影响，而相关性不受线性变换影响）
- 辅助分析：**cumulative error distribution**（用于检测是否出现大量零误差 → 判断记忆化）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Figure 2 & 4–5）

#### ✅ 0-shot 性能差异显著
| 数据集 | 平均 0-shot r | 解释 |
|------|-------------|------|
| **Delaney (solubility)** | ~85–90%（最大模型） | 表明 solubility 是训练语料中最常见的性质 |
| **Lipophilicity** | ~30–40% | 中等先验知识 |
| **QM7** | ~0%（除 Gemini 2.5 Pro 达 53%） | 多数模型未见过此数据集，说明非普遍记忆 |

> 💡 结论：0-shot 性能高低反映了数据集在 LLM 训练集中的曝光程度。

#### ✅ In-context learning 效果随样本数变化复杂
- **1000-shot > 0-shot**：大多数大模型在 1000-shot 下优于 0-shot，表明具备一定 ICL 能力。
- **60-shot < 0-shot**：约一半情况下，加入 60 个样例后性能反而下降！
  - 尤其在 Gemini Flash/Lite 上更明显
  - 表明 **prior knowledge 与 in-context 信息存在冲突**

#### ✅ 盲化实验揭示能力本质（Figure 4–5）
| 数据集 | 观察现象 | 推论 |
|------|--------|------|
| **Delaney** | 即使完全盲化（Level 6），性能仅轻微下降（96% → 92%） | 模型能通过 ICL 学会 solubility 映射，结构-性质关系较简单 |
| **Lipophilicity** | 盲化后性能大幅下滑（72% → 35%） | 严重依赖 prior knowledge，ICL 能力弱 |
| **QM7** | 盲化后性能不降反升，且 transformed 值预测更好 | 原始负值范围（-500~-2500）不利于 LLM 数值处理；无记忆依赖，靠通用 ICL |

#### ❌ 消融实验关键发现
- **SMILES 变换（Level 5–6）** 成功阻止了结构识别，但高性能仍可维持 → 支持“模式学习”而非“记忆”
- **值变换（Level 2,4,6）** 导致部分模型（如 GPT-5）剧烈波动 → 对数值范围敏感，存在“错误先验”
- **小模型（Gemini Flash/Lite）无法融合 prior 与 context** → 加入更多样例反而性能下降

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLMs 不依赖直接记忆（direct memorization）**
   - 0-shot 错误分布连续，极少精确匹配（Appendix A 显示高精度匹配率 < 0.1%，接近随机）
   - 即使在 Delaney 上表现好，也不是“背答案”，而是学到了 SMILES 与 solubility 的关联

2. **Prior knowledge 是一把“双刃剑”**
   - 当 prior 正确且任务常见时（如 solubility），有助于提升性能
   - 但当 prior 与 in-context 示例冲突，或样本不足时（如 60-shot），会导致性能下降
   - **盲化有时能提效**：移除误导性 prior 后，模型更专注于真实数据模式

3. **In-context learning 确实存在，但能力有限**
   - 所有模型都能利用 1000 个样例提升性能
   - 但在复杂性质（如 Lipophilicity）上 ICL 能力弱，表明难以从例子中归纳全局结构规律

4. **不同模型家族行为差异显著**
   - **GPT-5** 更擅长结合 prior 与 context，在 Specific/Generic 条件下表现最优
   - **Gemini 2.5 Pro** 在 Fully Blinded 下最强，体现更强的通用 ICL 能力
   - **Gemini Flash/Lite** 完全无法协调两种信息源，性能随样例增加而恶化

5. **标准 benchmark 可能被污染，需谨慎解读高分**
   - 高 benchmark score 可能源于数据泄露，而非真正的推理能力
   - 推荐使用 **blind evaluation** 作为验证模型泛化性的必要步骤

---

### ⚠️ 方法的局限性

1. **仅使用 SMILES 表示**
   - 忽略了 3D 结构、图表示等更丰富的分子编码方式
   - 可能低估了 LLM 在多模态输入下的潜力

2. **统计效力受限**
   - 每配置仅运行 2 次（due to cost），虽有 150 测试样本，但仍有一定方差

3. **仅覆盖三个数据集**
   - 缺乏对更多类型性质（如毒性、反应性）的验证
   - 泛化性有待进一步检验

4. **未探索其他 prompting 技巧**
   - 如 self-consistency、tool augmentation 等增强方法未纳入比较

---

### 🔮 未来工作方向

1. **扩展至更多科学领域**
   - 将 blind framework 应用于材料、生物序列、物理模拟等任务

2. **结合结构感知输入**
   - 探索 LLM + Graph Representation 或 3D Coordinates 的 hybrid 方法

3. **开发“去偏”机制**
   - 设计自动检测并抑制误导性 prior 的 prompting 策略

4. **构建抗污染 benchmark**
   - 创建专用于测试 ICL 能力的新数据集，避免已有数据泄露风险

5. **研究 LLM 内部机制**
   - 利用 probing 或 interpretability 工具，定位模型在不同盲化级别下的激活路径

---

> 🔚 **总结一句话**：  
> 本文证明，LLMs 在分子性质预测中并非简单“记忆”答案，而是综合利用 **pre-trained domain knowledge** 与 **in-context learning**；然而，这种结合并不总是有益的——**盲目信任先验可能导致失败，而适当“失明”反而促进学习**。因此，未来的科学应用必须审慎评估模型的真实泛化能力，而非仅看标准 benchmark 分数。

</details>

---

### 15. [TinyML for Acoustic Anomaly Detection in IoT Sensor Networks](https://arxiv.org/abs/2603.26135)

**Authors**: Amar Almaini, Jakob Folz, Ghadeer Ashour  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.26135v1  

#### Abstract
Tiny Machine Learning enables real-time, energy-efficient data processing directly on microcontrollers, making it ideal for Internet of Things sensor networks. This paper presents a compact TinyML pipeline for detecting anomalies in environmental sound within IoT sensor networks. Acoustic monitoring...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《TinyML for Acoustic Anomaly Detection in IoT Sensor Networks》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文针对 **IoT sensor networks** 中基于音频的异常检测面临的三大挑战：
- **高延迟**：依赖云端处理导致响应不及时；
- **高功耗与带宽消耗**：持续传输原始音频数据不可行；
- **隐私泄露风险**：原始声音数据外传存在安全隐患。

这些问题在资源受限、分布广泛的边缘设备中尤为突出。

### 🚀 提出的新方法与思路
提出了一种 **端到端的 TinyML pipeline**，用于在微控制器（MCU）上实现嵌入式环境下的 **acoustic anomaly detection**，其核心设计包括：
- 将多类 UrbanSound 分类任务转化为 **binary classification** 问题：  
  - `Normal`（如空调声、引擎怠速）
  - `Anomalous`（如警笛、枪声、电锯等紧急/干扰性声音）
- 采用 **MFCC 特征提取 + 轻量级全连接神经网络** 构建模型；
- 完整流程支持 **量化为 int8 模型并转换为 TensorFlow Lite Micro (TFLM)** 格式，适配 MCU 部署；
- 强调 **可复现性（reproducible）、硬件无关性（hardware-agnostic）和通用性（generalizable）**。

### 🔍 相比现有方法的优势
| 方面 | 本文优势 |
|------|---------|
| **部署兼容性** | 兼容标准 TFLM 工具链，无需定制硬件或专用平台 |
| **性能平衡** | 在保持 91% 准确率的同时，模型仅占 ~60kB 内存，适合低功耗设备 |
| **评估完整性** | 在公开数据集 UrbanSound8K 上进行全面评估，并报告量化前后性能差异 |
| **实用性导向** | 明确面向 real-world IoT 场景，强调隐私保护与实时响应能力 |

相比如 [2][4] 中的方法，本工作更具通用性和可推广性，且性能更优。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **UrbanSound8K**：
  - 包含 8,732 条标注音频片段；
  - 覆盖 10 种城市声音类别（如空调、儿童玩耍、警报、枪击等）；
  - 所有样本重采样至 **16kHz**，统一长度；
  - 使用官方划分的 10-fold 交叉验证策略，本文采用 **stratified 80/20 train-test split**。

### ⚙️ 实验设置
| 组件 | 设置详情 |
|------|----------|
| **特征提取** | 提取 **13 维 MFCCs**，帧长 40ms，重叠 50%，生成 13×32 的时频图，展平为 1D 向量输入 |
| **模型架构** | Fully Connected NN：<br>`Input → [128 ReLU → Dropout(0.2)] → [64 ReLU → Dropout(0.2)] → Sigmoid Output` |
| **训练配置** | - Optimizer: Adam (`lr=0.001`) <br> - Loss: Binary Cross-Entropy <br> - Batch Size: 64 <br> - Max Epochs: 50，Early Stopping（patience=5）<br> - 实际训练 36 轮后停止 |
| **模型优化** | 应用 **8-bit quantization** 并转为 `.tflite` 格式供 TFLM 使用 |
| **运行环境** | Python 3.10, TensorFlow 2.15, Librosa 0.10.1；无 GPU 加速 |

### 📊 评估指标
- **Accuracy**
- **F1-Score（macro）**
- **Precision & Recall（per class）**
- **ROC AUC**
- **Average Precision**
- **Confusion Matrix**
- 训练过程中的 **loss 和 accuracy 曲线**

### 🔁 基线方法对比
虽然未直接与其他模型进行端到端比较，但通过引用相关研究间接对比：
- Hammad et al. [2]：基于 LSTM Autoencoder 的无监督方法，在 ESP32 上部署，但依赖重建误差，难以复制；
- Rani [4]：类似 MFCC + 小模型，但在 EFR32 平台实现，准确率较低（文中指出“modest”），评估有限；
- 本文在相同任务下达到更高精度（91% vs 文献中普遍低于90%），且完全兼容主流 TinyML 流程。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（量化后模型）

| 指标 | 数值 |
|------|------|
| **Test Accuracy** | **91%** |
| **F1-Score (macro)** | **0.91** |
| **ROC AUC** | **0.970** |
| **Average Precision** | **0.970** |

> 注：原始 float32 模型性能更高（Accuracy=95%, F1=0.95），量化带来约 4% 性能下降，但换来显著压缩与部署可行性。

### 📊 类别级表现（Quantized Model）

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal (0) | 0.88 | 0.94 | 0.91 | 3,996 |
| Anomalous (1) | 0.94 | 0.89 | 0.91 | 4,283 |

✅ 表现出良好的 **类别均衡性**，无明显偏向某一类，对安全应用至关重要。

### 🔄 与基线方法对比结果
| 方法 | 准确率 | 是否支持 MCU | 是否使用公开数据集 | 可复现性 |
|------|--------|----------------|--------------------|------------|
| Hammad et al. [2] | 未明确 | 是（ESP32-S3） | 否（自采集） | 较低 |
| Rani [4] | 较低（文中称 modest） | 是（EFR32） | 是（部分 UrbanSound） | 中等 |
| 本文方法 | **91%** | **是（TFLM 兼容）** | **是（完整 UrbanSound8K）** | **高（代码与参数全公开）** |

👉 本文在准确性、通用性、可复现性方面均优于已有嵌入式方案。

### ❌ 消融实验（Ablation Study）
论文**未提供显式的消融实验**（如移除 dropout、调整层数、不同特征对比等）。  
但通过以下方式体现设计合理性：
- 展示训练曲线（Fig. 5–6）显示无过拟合，验证了 dropout 和 early stopping 的有效性；
- 对比量化前后的性能损失，说明量化可行；
- 使用标准 MFCC 特征而非复杂 CNN，突出了轻量化优先的设计哲学。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **TinyML 完全可用于高效的 acoustic anomaly detection**：
   - 即使是简单的 FCNN 结构，也能在边缘设备上实现高达 91% 的检测准确率；
   - MFCC 特征在资源受限场景下依然高效可靠。

2. **量化对性能影响可控**：
   - 从 float32 到 int8 仅造成约 4% 的精度下降，换来了模型体积大幅缩小（~60kB），非常适合 MCU 存储限制。

3. **平衡的分类性能**：
   - 正常与异常类别的 precision 和 recall 均超过 0.88，避免了误报过多或漏检严重的问题，适用于实际监控系统。

4. **完整的部署准备路径**：
   - 模型已成功转换为 TFLM 支持格式，具备实际部署潜力。

### ⚠️ 方法的局限性
- **尚未完成真实硬件部署测试**：
  - 当前仅模拟推理，缺乏在真实 MCU（如 STM32、ESP32）上的 **inference latency、memory footprint、power consumption** 实测数据。
- **静态分类框架**：
  - 采用固定时长音频块分类，未考虑连续流式音频处理（streaming inference）；
  - 不支持事件定位（when did the anomaly occur?）。
- **依赖预定义标签**：
  - 将 anomaly detection 视为有监督 binary classification，不同于真正的“未知异常”检测（unsupervised OOD detection）；
  - 对未见类型的异常可能失效。

### 🔮 未来工作方向
1. **真实硬件部署与性能测量**：
   - 在典型 MCU（如 ARM Cortex-M 系列）上测试 **real-time inference speed、RAM/Flash 占用、能耗**。
2. **进一步模型压缩**：
   - 探索 **pruning、knowledge distillation、sparsity** 等技术以降低计算开销。
3. **扩展至 multi-class 分类**：
   - 支持识别多种具体异常类型（如区分“siren” vs “gunshot”）。
4. **支持 streaming inference**：
   - 实现对连续音频流的实时滑动窗口检测，提升实用性。
5. **真实场景验证**：
   - 在真实 IoT sensor network 中收集数据，评估模型鲁棒性（robustness under noise, distance, background interference）。

---

## ✅ 总结一句话
该论文展示了如何利用 **TinyML + MFCC + 轻量NN** 在微控制器上构建一个高效、可复现、隐私友好的 **acoustic anomaly detection 系统**，在 UrbanSound8K 上实现了 **91% accuracy 和良好类别平衡**，为边缘智能在智慧城市与安全监测中的落地提供了坚实基础。

</details>

---

### 16. [Automatic Laplace Collapsed Sampling: Scalable Marginalisation of Latent Parameters via Automatic Differentiation](https://arxiv.org/abs/2603.26644)

**Authors**: Toby Lovick, David Yallup, Will Handley  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.26644v1  

#### Abstract
We present Automatic Laplace Collapsed Sampling (ALCS), a general framework for marginalising latent parameters in Bayesian models using automatic differentiation, which we combine with nested sampling to explore the hyperparameter space in a robust and efficient manner. At each nested sampling like...

---

### 17. [ClimateCheck 2026: Scientific Fact-Checking and Disinformation Narrative Classification of Climate-related Claims](https://arxiv.org/abs/2603.26449)

**Authors**: Raia Abu Ahmad, Max Upravitelev, Aida Usmanova, Veronika Solopova, Georg Rehm  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.26449v1  

#### Abstract
Automatically verifying climate-related claims against scientific literature is a challenging task, complicated by the specialised nature of scholarly evidence and the diversity of rhetorical strategies underlying climate disinformation. ClimateCheck 2026 is the second iteration of a shared task add...

---

### 18. [Adversarial-Robust Multivariate Time-Series Anomaly Detection via Joint Information Retention](https://arxiv.org/abs/2603.25956)

**Authors**: Hadi Hojjati, Narges Armanfard  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.25956v1  

#### Abstract
Time-series anomaly detection (TSAD) is a critical component in monitoring complex systems, yet modern deep learning-based detectors are often highly sensitive to localized input corruptions and structured noise. We propose ARTA (Adversarially Robust multivariate Time-series Anomaly detection via jo...

---

### 19. [Topology-Aware Graph Reinforcement Learning for Energy Storage Systems Optimal Dispatch in Distribution Networks](https://arxiv.org/abs/2603.26264)

**Authors**: Shuyi Gao, Stavros Orfanoudakis, Shengren Hou, Peter Palensky, Pedro P. Vergara  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.26264v1  

#### Abstract
Optimal dispatch of energy storage systems (ESSs) in distribution networks involves jointly improving operating economy and voltage security under time-varying conditions and possible topology changes. To support fast online decision making, we develop a topology-aware Reinforcement Learning archite...

---

### 20. [Shapley meets Rawls: an integrated framework for measuring and explaining unfairness](https://arxiv.org/abs/2603.26476)

**Authors**: Fadoua Amri-Jouidel, Emmanuel Kemel, St\'ephane Mussard  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.26476v1  

#### Abstract
Explainability and fairness have mainly been considered separately, with recent exceptions trying the explain the sources of unfairness. This paper shows that the Shapley value can be used to both define and explain unfairness, under standard group fairness criteria. This offers an integrated framew...

---

### 21. [An LP-based Sampling Policy for Multi-Armed Bandits with Side-Observations and Stochastic Availability](https://arxiv.org/abs/2603.26647)

**Authors**: Ashutosh Soni, Peizhong Ju, Atilla Eryilmaz, Ness B. Shroff  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.26647v1  

#### Abstract
We study the stochastic multi-armed bandit (MAB) problem where an underlying network structure enables side-observations across related actions. We use a bipartite graph to link actions to a set of unknowns, such that selecting an action reveals observations for all the unknowns it is connected to. ...

---

### 22. [DRiffusion: Draft-and-Refine Process Parallelizes Diffusion Models with Ease](https://arxiv.org/abs/2603.25872)

**Authors**: Runsheng Bai, Chengyu Zhang, Yangdong Deng  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.25872v1  

#### Abstract
Diffusion models have achieved remarkable success in generating high-fidelity content but suffer from slow, iterative sampling, resulting in high latency that limits their use in interactive applications. We introduce DRiffusion, a parallel sampling framework that parallelizes diffusion inference th...

---

### 23. [EcoFair: Trustworthy and Energy-Aware Routing for Privacy-Preserving Vertically Partitioned Medical Inference](https://arxiv.org/abs/2603.26483)

**Authors**: Mostafa Anoosha, Dhavalkumar Thakker, Kuniko Paxton, Koorosh Aslansefat, Bhupesh Kumar Mishra, Baseer Ahmad, Rameez Raja Kureshi  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.26483v1  

#### Abstract
Privacy-preserving medical inference must balance data locality, diagnostic reliability, and deployment efficiency. This paper presents EcoFair, a simulated vertically partitioned inference framework for dermatological diagnosis in which raw image and tabular data remain local and only modality-spec...

---

### 24. [BeSafe-Bench: Unveiling Behavioral Safety Risks of Situated Agents in Functional Environments](https://arxiv.org/abs/2603.25747)

**Authors**: Yuxuan Li, Yi Lin, Peng Wang, Shiming Liu, Xuetao Wei  
**Category**: cs.AI  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.25747v1  

#### Abstract
The rapid evolution of Large Multimodal Models (LMMs) has enabled agents to perform complex digital and physical tasks, yet their deployment as autonomous decision-makers introduces substantial unintentional behavioral safety risks. However, the absence of a comprehensive safety benchmark remains a ...

---

### 25. [ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory](https://arxiv.org/abs/2603.26182)

**Authors**: Zhuohan Ge, Haoyang Li, Yubo Wang, Nicole Hu, Chen Jason Zhang, Qing Li  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.26182v1  

#### Abstract
While Large Language Models (LLMs) have demonstrated potential in healthcare, they often struggle with the complex, non-linear reasoning required for accurate clinical diagnosis. Existing methods typically rely on static, linear mappings from symptoms to diagnoses, failing to capture the iterative, ...

---

### 26. [Automating Clinical Information Retrieval from Finnish Electronic Health Records Using Large Language Models](https://arxiv.org/abs/2603.26434)

**Authors**: Mikko Saukkoriipi, Nicole Hernandez, Jaakko Sahlsten, Kimmo Kaski, Otso Arponen  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.26434v1  

#### Abstract
Clinicians often need to retrieve patient-specific information from electronic health records (EHRs), a task that is time-consuming and error-prone. We present a locally deployable Clinical Contextual Question Answering (CCQA) framework that answers clinical questions directly from EHRs without exte...

---

### 27. [Preventing Data Leakage in EEG-Based Survival Prediction: A Two-Stage Embedding and Transformer Framework](https://arxiv.org/abs/2603.25923)

**Authors**: Yixin Zhou, Zhixiang Liu, Vladimir I. Zadorozhny, Jonathan Elmer  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.25923v1  

#### Abstract
Deep learning models have shown promise in EEG-based outcome prediction for comatose patients after cardiac arrest, but their reliability is often compromised by subtle forms of data leakage. In particular, when long EEG recordings are segmented into short windows and reused across multiple training...

---

### 28. [AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](https://arxiv.org/abs/2603.26096)

**Authors**: Hyeongyu Kim, Geonhui Han, Dosik Hwang  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.26096v1  

#### Abstract
Test-time adaptation (TTA) aims to mitigate performance degradation under distribution shifts by updating model parameters during inference. Existing approaches have primarily framed adaptation around affine modulation, focusing on recalibrating normalization layers. This perspective, while effectiv...

---

### 29. [PEANUT: Perturbations by Eigenvalue Alignment for Attacking GNNs Under Topology-Driven Message Passing](https://arxiv.org/abs/2603.26136)

**Authors**: Bhavya Kohli, Biplab Sikdar  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.26136v1  

#### Abstract
Graph Neural Networks (GNNs) have achieved remarkable performance on tasks involving relational data. However, small perturbations to the graph structure can significantly alter GNN outputs, raising concerns about their robustness in real-world deployments. In this work, we explore the core vulnerab...

---

### 30. [PruneFuse: Efficient Data Selection via Weight Pruning and Network Fusion](https://arxiv.org/abs/2603.26138)

**Authors**: Humaira Kousar, Hasnain Irshad Bhatti, Jaekyun Moon  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.26138v1  

#### Abstract
Efficient data selection is crucial for enhancing the training efficiency of deep neural networks and minimizing annotation requirements. Traditional methods often face high computational costs, limiting their scalability and practical use. We introduce PruneFuse, a novel strategy that leverages pru...

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
