# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-30 07:15:17 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [ParaQAOA: Efficient Parallel Divide-and-Conquer QAOA for Large-Scale Max-Cut Problems Beyond 10,000 Vertices](https://arxiv.org/abs/2603.26232)

**Authors**: Po-Hsuan Huang, Xie-Ru Li, Chi Chuang, Chia-Heng Tu, Shih-Hao Hung  
**Category**: cs.DC  
**Published**: 2026-03-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.26232v1  

#### Abstract
Quantum Approximate Optimization Algorithm (QAOA) has emerged as a promising solution for combinatorial optimization problems using a hybrid quantum-classical framework. Among combinatorial optimization problems, the Maximum Cut (Max-Cut) problem is particularly important due to its broad applicabil...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ParaQAOA: Efficient Parallel Divide-and-Conquer QAOA for Large-Scale Max-Cut Problems Beyond 10,000 Vertices

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **QAOA** 的 **Max-Cut** 求解器在处理大规模图时面临严重效率瓶颈。尽管已有 **divide-and-conquer** 类方法（如 DC-QAOA、QAOA2、Coupling QAOA）通过图划分降低子问题规模以提升可扩展性，但这些方法普遍**牺牲执行效率**，导致总运行时间显著增加，难以满足实际应用中的时效性需求。

### 🚀 提出的新方法：ParaQAOA
作者提出 **ParaQAOA** —— 一种**并行化的分治 QAOA 框架**，专为高效求解超大规模 Max-Cut 问题而设计。其核心思想是利用现代并行计算硬件（如多 GPU 架构）对传统分治流程进行端到端加速。

### 🔧 创新点
1. **线性时间图划分算法（Connectivity-Preserving Partitioning）**
   - 时间复杂度为 $O(|V| + |E|)$，远优于传统 $O(|V|^2)$ 或更高复杂度的划分方法。
   - 保证相邻子图共享一个节点，保留连通性信息，便于后续合并。

2. **完全并行化的执行流水线**
   - 子图上的 QAOA 执行阶段在多个 GPU 上并行运行。
   - 合并阶段采用 **Level-Aware Parallel Merge**，支持从任意层级开始并行深度优先搜索，最大化 CPU 资源利用率。

3. **可调参数化设计**
   - 引入两个关键可调参数：
     - **K**：每个子图保留的 top-K 高概率 bitstrings 数量，控制解空间多样性与计算开销。
     - **L**：合并过程的起始层级，控制并行粒度。
   - 用户可根据硬件资源和质量-效率权衡需求灵活配置。

4. **统一评估指标：Performance Efficiency Index (PEI)**
   - 定义为：  
     $$
     \text{PEI} = \text{AR} \times \text{EF} \times 100
     $$
     其中 AR 是 Approximation Ratio，EF 是基于 Sigmoid 函数归一化的 Efficiency Factor。
   - 综合衡量**解的质量**与**计算效率**，适用于跨方法公平比较。

### ⚖️ 相比现有方法的优势
| 方面 | 现有方法（如 QAOA2） | ParaQAOA |
|------|------------------------|----------|
| 图划分 | 高复杂度或随机划分 | 线性时间、保连通性 |
| 并行性 | 有限或无 | 全流程高度并行（GPU + CPU） |
| 可扩展性 | ~数千顶点 | 支持 **>16,000 顶点** |
| 效率 | 指数级增长 | 近线性扩展，受控于子图求解器性能 |
| 控制能力 | 固定流程 | 参数化调节质量-效率权衡 |

---

## 2. 核心实验方法和设置

### 📊 数据集
使用 **Erdős–Rényi 随机图**，覆盖多种规模与密度：
- **小规模**：20–30 个顶点
- **中等规模**：100–400 个顶点
- **大规模**：1,000–16,000 个顶点
- 边概率 $p$ 设置为 0.1, 0.3, 0.5, 0.8，生成稀疏至密集图。

每组配置生成 10 个图（不同随机种子），确保结果可复现。

### ⚙️ 实验设置
- **硬件平台**：
  - CPU: AMD Ryzen Threadripper 7960X (24-core)
  - GPU: 2× NVIDIA RTX 4090 (24GB GDDR6X each)
  - 内存: 256GB DDR5
  - 软件: CUDA 12.5, Python 3.12, Numba 加速 QAOA 核心

- **参数配置**：
  - 每个 QAOA Solver 最多处理 26 qubits → 单个子图最多 26 个顶点
  - 并行 Solver 数量：24（双 GPU × 12 实例/GPU）
  - K 值测试范围：1–8
  - L 值测试范围：1–3

### 📈 评估指标
1. **Approximation Ratio (AR)**：$\frac{\text{CutVal}_{\text{ALG}}}{\text{CutVal}_{\text{OPT/GW}}}$
2. **Execution Time**：端到端运行时间
3. **Speedup**：相对于基线方法的加速比
4. **Performance Efficiency Index (PEI)**：综合评价指标

### 🔁 基线方法对比
- **Goemans-Williamson (GW)**：经典近似算法，保证 AR ≥ 0.878
- **Coupling QAOA (CQ)**：基于耦合项的分治 QAOA
- **QAOA-in-QAOA (QAOA2)**：当前最先进的分治 QAOA 方法

> 注：由于 CQ 在 >30 顶点时耗时过长（>8 小时），仅用于小规模对比；中大规模主要对比 QAOA2。

---

## 3. 主要实验结果和性能指标

### 📉 性能对比（中等规模图，|V|=400）

| 方法 | 平均运行时间 (s) | Speedup vs QAOA2 | AR (%) | PEI |
|------|------------------|------------------|--------|-----|
| QAOA2 | 17,001.0 | 1× | ~97.8 | ~52 |
| **ParaQAOA** | **10.3** | **1,652×** | **≥95.8%**（差距 ≤2%） | **~95** |

- **最高达 1,652× 的速度提升**，同时保持 AR 在最优解的 98% 以内。
- 在边密度较高（p=0.8）时优势更明显，因 QAOA2 的枚举代价呈指数增长，而 ParaQAOA 不敏感。

### 📈 大规模图表现（|V|=16,000）

| 方法 | 预计运行时间 | 实际运行时间 | 是否可行 |
|------|--------------|--------------|----------|
| QAOA2 | **>13.6 天**（外推值） | 未完成 | ❌ 不实用 |
| **ParaQAOA** | — | **19 分钟** | ✅ 可行 |

> 这是首次在经典计算机上实现 **>10,000 顶点 Max-Cut 问题在分钟级内求解**。

### 🔍 消融实验结果

#### 参数 K 对性能的影响（|V|=200）
- **K=1~2**：即可达到接近 QAOA2 的 AR，且运行时间极低。
- **K↑**：AR 缓慢提升，但运行时间线性增长 → 显示明确的质量-效率权衡。

#### 参数 L 对并行效率的影响（|V|=600, K=2）
- **L=1** → 4 个进程
- **L=2** → 8 个进程
- **L=3** → 16 个进程
- 结果显示：**L 每增加 1，运行时间约减半**，验证了 Level-Aware 设计的有效性和可扩展性。

#### PEI 综合评估
- 在所有测试配置下，**ParaQAOA 的 PEI 均显著高于 QAOA2**。
- 特别是在高复杂度图上（大 |V| 或高 p），优势更加突出。
- 多数情况下 PEI > 90，表明其在质量和效率之间实现了卓越平衡。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **ParaQAOA 成功打破了“高质量 vs 高效率”的固有矛盾**，首次实现了在超大规模 Max-Cut 问题上兼具高精度与高速度。
2. **全流程并行化设计是关键**：从图划分、QAOA 执行到合并阶段，均实现高效并行，充分利用现代异构硬件。
3. **线性时间图划分 + Level-Aware 合并** 构成了可扩展性的基石。
4. **PEI 是有效的综合评估工具**，能够客观反映不同算法在现实场景下的实用性。

### ⚠️ 局限性
1. 当前图划分策略为**随机顺序划分**，可能不适用于具有强社区结构或非均匀分布的图。
2. 依赖于**经典模拟器**执行 QAOA，尚未部署在真实量子设备上（受限于 NISQ 时代硬件规模）。
3. 合并阶段仍存在组合爆炸风险，尽管通过 K 和 L 控制，但在极端情况下仍可能成为瓶颈。

### 🔮 未来工作方向
1. 探索**自适应图划分策略**（如基于图聚类或社区检测），提升结构化图上的性能。
2. 将框架扩展至其他 **QUBO 问题**，如 TSP、Graph Coloring、Portfolio Optimization 等。
3. 结合 **noise-aware circuit design** 和 **error mitigation** 技术，适配真实量子硬件。
4. 开发自动调参机制，根据输入图特征动态优化 K 和 L。

---

## 总结
**ParaQAOA 是首个将并行计算思想系统引入分治 QAOA 的框架**，通过**线性划分、全并行执行、参数化控制和统一评估指标 PEI**，实现了对大规模 Max-Cut 问题的高效求解。其实验结果证明，在 **16,000 顶点图上仅需 19 分钟**，相比现有方法提速 **超过 1,600 倍**，同时保持近似比在 98% 以上，标志着 QAOA 在走向实际应用道路上的重要突破。

</details>

---

### 2. [Optimization Trade-offs in Asynchronous Federated Learning: A Stochastic Networks Approach](https://arxiv.org/abs/2603.26231)

**Authors**: Abdelkrim Alahyane (LAAS-SARA), C\'eline Comte (CNRS, LAAS-SARA), Matthieu Jonckheere (CNRS, LAAS-SARA)  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.26231v1  

#### Abstract
Synchronous federated learning scales poorly due to the straggler effect. Asynchronous algorithms increase the update throughput by processing updates upon arrival, but they introduce two fundamental challenges: gradient staleness, which degrades convergence, and bias toward faster clients under het...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimization Trade-offs in Asynchronous Federated Learning: A Stochastic Networks Approach

---

## 1. 主要贡献和创新点

### **解决了什么问题**

异步联邦学习（Asynchronous Federated Learning, AsyncFL）虽然能缓解同步系统中的 **straggler effect**，但在实际应用中面临两大核心挑战：

1. **Gradient Staleness**：由于客户端计算和通信延迟不同，模型更新基于过时的全局参数，导致梯度陈旧，影响收敛速度和精度。
2. **Bias Toward Faster Clients**：在非独立同分布（Non-IID）数据下，快速客户端贡献更多更新，导致模型偏向其数据分布。

现有研究大多忽略底层的 **queueing dynamics**（排队动态），仅以通信轮次（communication rounds）作为收敛分析指标，忽略了真实的 **wall-clock time**（挂钟时间）。此外，缺乏对能量消耗的联合建模与优化。

---

### **提出了什么新方法或新思路**

本文提出了一种基于 **stochastic queueing-network**（随机排队网络）的统一框架，用于分析和优化 **Generalized AsyncSGD** 算法。主要创新点如下：

#### ✅ **1. 统一的随机排队网络模型**
- 将客户端的 **computation**、**uplink/downlink communication** 和中央服务器（CS）的处理过程统一建模为一个 **closed Jackson network**。
- 显式引入 **随机计算时间**（指数分布）、**通信延迟** 和 **CS 处理能力**，更贴近真实边缘环境。

#### ✅ **2. 闭式表达式（Closed-form Expressions）**
- 利用 **product-form network theory** 推导出以下关键性能指标的闭式解：
  - 平均相对延迟（Average Relative Delay）
  - 更新吞吐量（Update Throughput）
  - 收敛到 ε-平稳点所需的通信轮数
  - 所需的期望 **wall-clock time**
  - 能耗（Energy Consumption）

#### ✅ **3. 揭示并量化核心权衡（Trade-offs）**
- **Staleness vs. Throughput**：并发任务数 `m` 增加可提升吞吐量，但加剧梯度陈旧；反之则降低吞吐。
- **Time vs. Energy**：最小化能耗需串行执行（`m=1`），但训练时间极长；增加并发可加速训练但增加能耗。

#### ✅ **4. 梯度优化策略**
- 提出基于梯度下降的联合优化方法，同时优化：
  - **Routing Probabilities** `p`：控制任务分配给各客户端的概率。
  - **Concurrency Level** `m`：系统中并发的任务数量。
- 目标函数可为最小化 `E[T]`（时间）、`E[E]`（能量）或两者的加权组合。

---

### **相比现有方法的优势**

| 方面 | 传统方法 | 本文方法 |
|------|--------|--------|
| **分析维度** | 仅基于通信轮次（round-based） | 基于 **wall-clock time** 和 **energy** |
| **系统建模** | 忽略通信延迟或假设确定性延迟 | 显式建模 **随机计算与通信延迟** |
| **优化变量** | 固定 `m=n`, 均匀路由 | 联合优化 `p` 和 `m` |
| **理论工具** | 渐近近似或最坏情况分析 | **闭式表达式**，支持精确梯度优化 |
| **适用性** | 理想化假设 | 适用于 **异构硬件、网络、数据分布** |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **EMNIST**：手写字符识别，47类，131,600张图像。
- **KMNIST**：日本手写假名，10类，70,000张图像。
- **CIFAR-100**：彩色图像分类，100类，60,000张图像。

### **实验设置**

- **客户端数量**：`n = 100`
- **客户端集群**：分为5类（A-E），模拟异构设备（从高性能工作站到资源受限的 straggler）。
- **服务速率**：计算、上行、下行速率各异（见 Table 1）。
- **数据分布**：
  - **IID**：数据均匀划分。
  - **Non-IID**：通过 Dirichlet 分布（α=0.2）模拟标签异质性。
- **延迟分布**：测试了 **Exponential**、**Deterministic** 和 **Lognormal** 三种分布，验证鲁棒性。

### **评估指标**

- **Wall-clock time to target accuracy**（达到目标准确率所需的真实时间）
- **Total energy consumption**（总能耗）
- **Test accuracy / Loss over time**
- **Time and energy reduction (%)** 相对于基线

### **基线方法对比**

1. **AsyncSGD (Baseline)**  
   - 均匀路由 `p_i = 1/n`，全并发 `m = n`

2. **Round-Optimized Generalized AsyncSGD**  
   - 固定 `m = n`，优化 `p` 以最小化通信轮数 `K_c`

3. **Max-Throughput Generalized AsyncSGD**  
   - 固定 `m = n`，优化 `p` 以最大化吞吐量 `λ`

4. **Time-Optimized (Proposed)**  
   - 联合优化 `(p*, m*)` 以最小化 `E[T]`

5. **Joint Time-Energy Co-Optimized**  
   - 最小化归一化的加权目标：`p * (E[E]/E*) + (1-p) * (E[T]/T*)`

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **时间优化策略（Time-Optimized）**
- 在 EMNIST 上，相比 AsyncSGD，**收敛时间减少 29%–46%**。
- 在 Non-IID 设置下，最大提速达 **79.3%**（Exponential 延迟）。
- 最优并发数 `m* = 91`（小于 `n=100`），表明 **全并发并非最优**。

#### ✅ **联合时间-能量优化策略**
- 设定 `p=0.1`（轻微偏好节能），相比 AsyncSGD：
  - **能耗降低 36%–49%**
  - **收敛时间仍加快 3.9%–18.95%**（除个别场景外）
  - 在 EMNIST Deterministic Non-IID 下，以 **3.15% 时间代价换取 36.19% 能耗节省**

#### ✅ **消融实验与对比分析**
- **Max-Throughput**：虽吞吐量高（152 vs 7.4 updates/unit time），但因梯度陈旧严重，**最终准确率低 60%**，且不稳定。
- **Round-Optimized**：通信轮数最少，但吞吐量极低（4.5），**wall-clock time 更长**。
- **Time-Optimized**：在吞吐量（18.7）和陈旧度之间取得平衡，实现最快真实时间收敛。

> 表格摘要（来自 Table 3 & 5）：
>
> | 场景 | 相比 AsyncSGD 的时间减少 | 相比 AsyncSGD 的能耗减少 |
> |------|------------------------|--------------------------|
> | EMNIST (Exp, Non-IID) | 35.6% | — |
> | EMNIST (Det, Non-IID) | -3.15% | 36.19% |
> | KMNIST (LogN, Non-IID) | 8.63% | 40.90% |

---

## 4. 关键结论和发现

### **主要发现**

1. **Wall-clock time 是更合理的评估指标**  
   单纯优化通信轮数（如 Round-Optimized）会牺牲吞吐量，反而延长真实训练时间。

2. **并发数 `m` 存在最优值**  
   `m` 过小导致资源未充分利用，过大则加剧 **gradient staleness**。最优 `m* < n`，挑战了“越多越好”的直觉。

3. **联合优化 `p` 和 `m` 至关重要**  
   仅优化路由概率无法解决并发带来的陈旧问题；仅调整 `m` 也无法补偿路由偏差。

4. **存在根本的 Time-Energy Trade-off**  
   最小化能耗需 `m=1`，但训练时间不可接受；实用方案需在两者间权衡。

5. **所提方法具有强鲁棒性**  
   在 **Exponential、Deterministic、Lognormal** 延迟下均表现稳定，说明模型不依赖特定分布假设。

---

### **方法的局限性**

- **假设指数服务时间**：理论推导依赖指数分布以获得 product-form 解，尽管实验显示其他分布也有效。
- **静态系统假设**：未考虑客户端动态加入/退出（dynamic participation）。
- **集中式 CS 模型**：假设单一中央服务器，未扩展至去中心化或多服务器场景。
- **简化能耗模型**：虽考虑了计算、上下行功耗，但未建模更复杂的电源管理策略（如 DVFS 动态调频）。

---

### **未来工作方向**

1. **扩展至非指数服务时间模型**，如 Phase-type 分布，保持解析可处理性。
2. **支持动态客户端参与**，研究在线优化策略。
3. **多服务器异步 FL**，研究分布式缓冲与聚合机制。
4. **结合模型压缩与通信效率**，进一步降低端到端成本。
5. **在真实边缘平台部署验证**，如 Raspberry Pi 或移动设备集群。

---

> **总结**：本文通过将 **stochastic network theory** 引入异步联邦学习，首次实现了对 **wall-clock time** 和 **energy** 的闭式建模与联合优化，揭示了关键的系统级权衡，并通过实验验证了其在真实异构环境下的显著优势。该工作为构建高效、可持续的边缘智能系统提供了坚实的理论基础和优化工具。

</details>

---

### 3. [AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents](https://arxiv.org/abs/2603.26034)

**Authors**: Wenbo Gao, Renxi Liu, Xian Wang, Fang Guo, Shuai Yang, Xi Chen, Hui-Ling Zhen, Hanting Chen, Weizhe Lin, Xiaosong Li, Yaoyuan Wang  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.26034v1  

#### Abstract
Autonomous agents powered by large language models (LLMs) perform complex tasks through long-horizon reasoning and tool interaction, where a fundamental trade-off arises between execution efficiency and reasoning robustness. Models at different capability-cost levels offer complementary advantages: ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在基于 **Large Language Models (LLMs)** 的自主智能体（autonomous agents）中，存在一个根本性的权衡：**推理能力越强的模型（如 large models）虽然推理更鲁棒，但单步延迟高、成本大；而轻量级模型（small models）执行速度快，但在复杂推理任务上容易出错并导致后续步骤失效**。这种错误会随着多轮交互（multi-turn reasoning）不断累积，最终严重影响整体效率。

传统方法通常采用固定模型或外部路由模块（external router）来选择模型，难以有效平衡长程任务中的 **端到端效率（end-to-end latency）** 和 **推理质量（reasoning quality）**。

---

### 提出了什么新方法或新思路
本文提出 **AgentCollab**，一种**自驱动协作推理框架（self-evaluation-driven collaboration paradigm）**，其核心思想是：

- **无需外部路由模块或训练策略**，而是利用智能体自身的 **自我反思信号（self-reflection signal）** 来判断当前推理轨迹是否陷入停滞（stagnation）。
- 当检测到停滞时，才将控制权临时转移给更强的 **large model** 进行干预，解决困难片段后即返回小模型继续高效执行。
- 引入 **难度感知的累计升级策略（difficulty-aware cumulative escalation strategy）**：根据连续失败次数动态调整 large model 的干预时长（intervention budget），实现渐进式更强纠正。

该方法实现了 **计算资源的按需分配**，将高成本推理集中在真正需要的时刻。

---

### 相比现有方法的优势
| 对比维度 | AgentCollab | 传统方法（如 RouteLLM / FrugalGPT） |
|--------|------------|-----------------------------|
| 路由机制 | 内生式（self-reflection） | 外部路由模型或学习策略 |
| 决策依据 | 推理轨迹级进展评估 | 单步输出质量预测 |
| 长程适应性 | 显式建模长期动态 | 忽略历史状态，独立决策每一步 |
| 开销 | 无额外训练或路由模型开销 | 需要额外监督或评分模型 |
| 效率-质量权衡 | 更优的 Pareto frontier | 性能提升有限 |

> ✅ **优势总结**：AgentCollab 在不引入额外模块的前提下，通过内部信号实现更智能、更高效的模型协作，在保持高性能的同时显著降低端到端延迟。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖三大类多步智能体任务，涵盖不同语言和领域：

| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **BrowseComp_zh** | Web-based research | 中文网页浏览与信息检索任务，共 289 题 |
| **HLE-math** | Mathematical reasoning | 数学推理挑战题，共 866 题 |
| **WritingBench** | Long-form generation | 长文本生成任务，共 1000 题 |

这些任务均要求 **Think-Act-Observe 循环**，适合测试长程推理能力。

---

### 实验设置和评估指标

#### 模型配置
集成到两个开源 agent 框架：
- **DDV2**：使用 7B / 38B 参数模型对
- **WebSailor**：使用 3B / 7B / 32B 参数模型组合

#### 关键设置
- 最大交互轮数：40
- 初始 large model 规划阶段预算 $ K_L = 2 $
- 基础干预预算 $ B_0 = 2 $，增长系数 $ k = 2 $，最大预算 $ B_{\text{max}} = 6 $
- 所有推理本地运行于 Ascend 910B3 NPU 上，使用 vLLM-Ascend

#### 评估指标
| 指标 | 定义 |
|-----|------|
| **Accuracy / Score** | GPT-4o 或 benchmark 自带评分器打分 |
| **Speedup** | 相对于 pure large model 的端到端延迟加速比 |
| **#Steps** | 平均每个问题所需的推理迭代次数 |
| **Switching Ratio** | 模型切换频率（衡量稳定性） |

---

### 基线方法对比
| 基线方法 | 类型 | 说明 |
|--------|------|------|
| **Small-only** | 小模型全程执行 | 高速低质 |
| **Large-only** | 大模型全程执行 | 高质低速（基准） |
| **Random** | 随机切换 | 控制变量 |
| **RouteLLM** | 学习型单步路由 | 基于输出质量阈值触发 large model |
| **FrugalGPT** | 成本感知路由 | 使用 win-prediction 模型决定是否升级 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Agent | Method | BrowseComp_zh Acc.(%) | Speedup | HLE-math Acc.(%) | Speedup | WritingBench Score | Speedup |
|------|--------|------------------------|---------|------------------|---------|--------------------|---------|
| DDV2 | Small | 18.3 | 1.54× | 8.0 | 3.38× | 4.4 | 3.20× |
| DDV2 | Large | 34.6 | 1.00× | 23.3 | 1.00× | 5.1 | 1.00× |
| DDV2 | **Ours (AgentCollab)** | **33.9** | **1.36×** | **21.1** | **2.31×** | **5.0** | **2.43×** |
| WebSailor | Small | 14.2 | 2.03× | 11.3 | 2.48× | – | – |
| WebSailor | Large | 25.5 | 1.00× | 14.0 | 1.00× | – | – |
| WebSailor | **Ours (AgentCollab)** | **22.5** | **1.50×** | **13.3** | **1.29×** | – | – |

> 📌 **结论**：AgentCollab 在几乎所有任务上都接近甚至逼近 large model 的性能，同时获得显著的速度提升（最高达 3.38×），远优于其他切换策略。

---

### 与基线方法的对比结果
- 在 DDV2 + BrowseComp_zh 上：
  - AgentCollab 准确率 **33.9%**，远超 Small (18.3%)、RouteLLM (28.2%) 和 FrugalGPT (27.2%)
  - 速度仍快于 large model **1.36×**
- 在 HLE-math 上：
  - 准确率从 8.0%（Small）提升至 21.1%，接近 Large 的 23.3%
  - 速度为 large model 的 **2.31×**

> 🔍 图 3 显示，AgentCollab 明显改善了 accuracy-speedup 的 **Pareto frontier**，优于所有 baseline。

---

### 消融实验结果（Table 2 & Figure 5）

#### （1）角色分配影响（Planner vs Executor）
| 配置 | 准确率 (%) | Speedup |
|------|-----------|--------|
| Large Planner + Small Executor | 24.6 | 1.39× |
| Small Planner + **Large Executor** | **27.3** | 1.24× |

> 💡 发现：**执行阶段（executor）比规划阶段（planner）更关键**，因为其直接与环境交互、获取证据，错误影响更大。

#### （2）静态 vs 动态预算策略
| 方法 | 准确率 (%) | Speedup | Switching Ratio |
|------|-----------|--------|----------------|
| AgentCollab (Static) | 32.5 | 1.32× | 49.64 |
| AgentCollab (**Dynamic**) | **33.9** | **1.36×** | **45.07** |

> ✅ 动态策略不仅准确率更高，且切换更少 → 更利于 **prefill caching 复用**，减少中断。

#### （3）参数 $ k $ 影响（图 5）
- $ k=2 $ 时达到最佳 trade-off
- $ k=0 $（静态）性能最差
- $ k>2 $ 可能导致过度干预，收益递减

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **自我反思可作为有效的内生路由信号**：无需外部控制器，LLM 自身即可判断何时需要更强推理支持。
2. ✅ **协作优于单一模型**：结合 small/large model 的优势，可在几乎不失性能的情况下大幅提升效率。
3. ✅ **执行阶段比规划更重要**：在 agent 架构中，allocate stronger model to executor 更有效。
4. ✅ **动态预算优于固定预算**：根据历史失败信号调节干预强度，能更好应对持续困难任务。
5. ✅ **端到端效率取决于轨迹质量而非单步速度**：即使小模型每步更快，若频繁走偏，总耗时反而更高。

---

### 方法的局限性
1. **依赖高质量 self-evaluation 能力**：
   - 如 WebSailor 因 false-positive 判断较多（7% vs DDV2 的 3%），效果较差。
2. **仅验证同构模型协作**：
   - 所有实验基于同一架构不同规模的模型（如 Qwen 系列），未探索异构模型（如擅长搜索 vs 擅长数学）之间的协作。
3. **未包含闭源模型 API 测试**：
   - 由于 API 延迟不稳定，无法精确测量效率，限制了实际部署场景的泛化性。

---

### 未来工作方向
1. **异构模型协作（Heterogeneous Collaboration）**：
   - 探索具有互补能力的模型如何协同工作（例如：一个专精 web search，另一个擅长逻辑推理）。
2. **改进 self-evaluation 可靠性**：
   - 设计更鲁棒的 progress-check prompt 或辅助机制，降低误判率。
3. **扩展至更多 agent 架构与任务类型**：
   - 应用于 multi-agent system、robotics planning、code generation 等复杂场景。
4. **在线自适应预算调整**：
   - 引入强化学习或其他机制实现完全动态的 budget 分配。

---

> 🏁 **总体评价**：  
> AgentCollab 提出了一种简洁而强大的范式——让 LLM 自己决定“什么时候该动用大脑”，实现了 **高效、自适应、无需额外训练** 的智能体协作机制，为构建下一代高效 LLM agents 提供了重要思路。

</details>

---

### 4. [Hardware-Agnostic and Insightful Efficiency Metrics for Accelerated Systems: Definition and Implementation within TALP](https://arxiv.org/abs/2603.26576)

**Authors**: Ghazal Rahimi, Victor Lopez, Marc Clasc\`a, Joan Vinyals Ylla Catal\`a, Jesus Labarta, Marta Garcia-Gasulla  
**Category**: cs.DC  
**Published**: 2026-03-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.26576v1  

#### Abstract
The increasing adoption of heterogeneous platforms that combine CPUs with accelerators such as GPUs in high-performance computing (HPC) introduces new challenges for performance analysis and optimization. Traditional efficiency metrics, such as those proposed by the Performance Optimization and Prod...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Hardware-Agnostic and Insightful Efficiency Metrics for Accelerated Systems: Definition and Implementation within TALP

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
随着异构计算系统（如 CPU + GPU）在高性能计算（HPC）中的广泛应用，传统的性能分析工具和效率度量（如 POP Parallel Efficiency）已无法有效捕捉**主机（host）与设备（device）之间的复杂交互行为**。现有工具存在以下问题：
- 多数为厂商专用（vendor-specific），如 NVIDIA Nsight、AMD ROCProfiler，缺乏跨平台通用性；
- 工具输出信息过于底层且繁杂，非专家用户难以解读；
- 缺乏统一、可解释的效率度量框架来量化异构系统中不同层次的性能损失。

### 🚀 提出的新方法与新思路
本文提出了一套**硬件无关（hardware-agnostic）且具有洞察力的效率度量体系**，扩展了经典的 POP 效率模型至异构加速系统，并集成于轻量级监控工具 **TALP** 中。主要创新包括：

#### （1）双层级效率度量框架
将原仅适用于 CPU 的 POP 框架拆分为两个独立的度量树：
- **Host-side Metrics Tree**：评估主机端资源利用效率
  - 新增 `Host Hybrid Parallel Efficiency` 和 `Device Offload Efficiency (OE_host)`
  - 区分 Useful、Offloading、MPI 三种状态，更精确反映因 GPU 卸载导致的 CPU 阻塞开销
- **Device-side Metrics Tree**：首次系统化定义设备端效率
  - 引入 `Device Parallel Efficiency` 及其子项：
    - `Load Balance (LB_device)`：衡量 GPU 间负载均衡性
    - `Communication Efficiency (CE_device)`：反映内存传输开销
    - `Orchestration Efficiency (OE_device)`：刻画调度空闲时间（idle time）

> 公式示例：
> $$
> \text{PE}_{\text{host}} = \frac{\sum D_u}{E \cdot n}, \quad
> \text{OE}_{\text{host}} = \frac{\sum D_u}{\sum (D_u + D_w)}, \quad
> \text{OE}_{\text{device}} = \frac{\min(E - D_k - D_m)}{E}
> $$

#### （2）实现于 TALP 框架
- 在 **DLB 库的 TALP 模块**中实现了上述指标
- 支持运行时（online）与事后（post-mortem）两种模式
- 输出格式支持文本与 JSON，便于自动化分析

### ⭐ 相比现有方法的优势
| 特性 | 传统工具（Nsight, TAU, HPCToolkit） | 本文方法（TALP 扩展） |
|------|-------------------------------|------------------------|
| 跨平台支持 | ❌ 通常绑定特定厂商 | ✅ 支持 NVIDIA（CUDA/OpenACC）、AMD（HIP） |
| 易用性 | ❌ 输出原始事件，需专家解读 | ✅ 提供高层语义化的效率指标 |
| 架构抽象 | ❌ 关注低层计数器/事件 | ✅ 抽象为“有用 vs 无用”执行状态 |
| 统一度量 | ❌ 缺乏标准化异构效率模型 | ✅ 统一 host/device 度量框架 |
| 开源透明 | 部分闭源 | ✅ 完全开源（GitHub） |

---

## 2. 核心实验方法和设置

### 🧪 实验设计
通过两类场景验证所提度量的有效性和实用性：

#### （1）合成基准测试：PILS 微基准
- 功能：模拟多种典型的异构执行模式（负载不均、通信瓶颈、重叠不足等）
- 配置：
  - 使用 2 个 MPI ranks，各绑定 1 CPU core + 1 GPU
  - 控制变量构造 7 种 use case
- 目标：验证度量对典型性能问题的敏感性和准确性

#### （2）真实科学应用案例
选取三个生产级 HPC 应用进行实测：
| 应用 | 领域 | 编程模型 |
|------|------|----------|
| **SOD2D** | 流体力学（LES/DNS） | Fortran + MPI + OpenACC |
| **FALL3D** | 大气输运模拟 | Fortran + MPI + CUDA |
| **XSHELLS** | 地球物理流体动力学 | C++ + MPI + CUDA |

- 平台：MareNostrum5 加速分区（MN5-Acc）
  - 节点配置：2× Intel Sapphire Rapids + 4× NVIDIA H100 GPU
- 规模：从 1 到 8 节点横向扩展，观察可扩展性趋势

### 📊 评估指标
所有实验均报告以下核心效率指标：
- **Host-side**:
  - `Parallel Efficiency`, `MPI PE`, `Load Balance`, `Device Offload Efficiency`
- **Device-side**:
  - `Device PE`, `LB_device`, `CE_device`, `Orchestration Efficiency`

### 🔍 基线对比方式
- 不直接与其他工具比较数值结果（因度量维度不同）
- 采用 **trace 级别可视化验证**（使用 Paraver + Nsight Systems 转换 trace）
- 对比 TALP 输出是否与 trace 中观察到的行为一致 → 验证**正确性与洞察力**

---

## 3. 主要实验结果和性能指标

### 📈 合成基准（PILS）关键发现

| Use Case | 关键现象 | TALP 指标响应 |
|---------|--------|-------------|
| UC1: GPU 负载高，CPU 闲置 | CPU 主要用于卸载任务 | `OE_host = 0.16` 极低，`OE_device = 0.82` 较高 |
| UC2: CPU 负载高，GPU 未充分利用 | GPU 几乎空转 | `Device PE = 0.05`，揭示严重资源浪费 |
| UC3: GPU 负载不平衡 | 一个 GPU 工作远多于另一个 | `LB_device = 0.55` 准确反映不均衡 |
| UC6: 大量 Host-Device 数据移动 | 单节点大量 H2D/D2H 传输 | `OE_host = 0.09`, `CE_device = 0.36` 明显下降 |
| UC7: 是否重叠 CPU/GPU 计算 | 有无异步重叠 | `OE_host` 从 0.64 提升至 0.97（↑33%），体现优化效果 |

> ✅ 结论：TALP 指标能精准定位各类异构性能瓶颈，且变化趋势符合预期。

---

### 📊 真实应用性能数据汇总

#### 表 1：SOD2D（1–8 nodes）

| 指标 | 1 node | 8 nodes | 趋势分析 |
|------|--------|--------|--------|
| Host PE | 0.06 → 0.04 | ↓ | 设备卸载效率极低 |
| Device Offload Eff. | ~0.06 | ↔ | CPU 几乎只做卸载，无实际计算 |
| Device PE | 0.87 → 0.59 | ↓ | 随规模增大，GPU 有效利用率下降 |
| Orchestration Eff. | 0.88 → 0.60 | ↓ | 主因是 host 侧通信延迟影响任务下发 |

> 🔍 发现：尽管 GPU 利用率高，但 CPU 成为瓶颈；MPI 通信增多限制了 GPU 工作调度。

---

#### 表 2：FALL3D（1–8 nodes）

| 指标 | 1 node | 8 nodes |
|------|--------|--------|
| Host PE | 0.26 → 0.07 | ↓↓ |
| Load Balance (host) | 0.52 → 0.12 | ↓↓ |
| Device Offload Eff. | 0.59 → 0.64 | ↑ |
| Orchestration Eff. | 0.19 → 0.04 | ↓↓ |

> 🔍 发现：
- `Load Balance` 急剧恶化 → 表明并行划分不合理
- `Orchestration Eff.` 接近 0 → GPU 长期空闲等待工作
- 尽管 `Device Offload Eff.` 上升，说明 CPU 更多地参与计算，但整体协同效率极差

---

#### 表 3：XSHELLS（1–8 nodes）

| 指标 | 1 node | 8 nodes |
|------|--------|--------|
| Host PE | 0.36 → 0.15 | ↓↓ |
| Comm. Eff. (host) | 0.91 → 0.27 | ↓↓ |
| Device Offload Eff. | 0.40 → 0.60 | ↑ |
| Orchestration Eff. | 0.54 → 0.10 | ↓↓ |

> 🔍 发现：
- 初始化阶段 MPI 通信密集且不可伸缩（trace 验证）
- GPU 利用受限于主机无法及时提交任务
- 虽然单节点尚可，但扩展性极差

---

### ✅ 消融实验（隐含在 use case 分析中）
虽然未明确命名“ablation study”，但通过控制变量的多个 PILS use case 实现了类似功能：
- 固定负载分布，改变数据传输量 → 观察 `CE_device` 变化
- 固定通信开销，引入重叠 → 观察 `OE_host` 提升
- 控制负载平衡 → 验证 `LB_device` 敏感性

→ 所有指标均表现出良好的**归因能力与分离性**

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **异构系统的性能瓶颈不能仅靠传统 CPU 指标识别**  
   → 必须同时监测 host 与 device 两端的效率。

2. **所提出的度量体系能够提供可操作的洞察（actionable insights）**  
   → 如低 `Orchestration Efficiency` 明确指向“GPU 等待任务”，提示应改进任务调度或增加并发粒度。

3. **TALP 实现具备轻量、透明、跨平台优势**  
   → 基于 PMPI/OMPT/CUPTI/rocprofiler，无需修改代码即可部署。

4. **真实应用普遍存在“GPU 利用率高 ≠ 系统高效”的误区**  
   → 多个案例显示即使 `Device PE` 较高，整体效率仍受制于 host 侧通信或调度延迟。

---

### ⚠️ 局限性
1. 当前仅实现 **Device Parallel Efficiency** 分支  
   → 尚未包含 `Device Computational Efficiency`（如 IPC、Occupancy 等微架构层面度量）

2. GPU 内部 stream 重叠尚未建模精细  
   → 当前将 kernel 重叠合并处理，可能低估某些隐藏潜力

3. 多设备间拓扑感知不足  
   → 未考虑 NVLink/P2P 带宽差异对 `Communication Efficiency` 的影响

4. 仅支持 CUDA/HIP，暂未覆盖其他加速器（如 Intel GPU/Xe）

---

### 🔮 未来工作方向
1. **完善 Device Computational Efficiency 分支**
   - 引入基于硬件计数器的微观效率度量（如 SM Utilization, Memory Bandwidth Saturation）

2. **增强对编程模型的支持**
   - 扩展至 SYCL、OneAPI、OpenMP Target 等跨厂商卸载模型

3. **支持 runtime 自适应反馈机制**
   - 利用 TALP 实时指标驱动动态调优（如自动调整 batch size 或启用异步传输）

4. **构建可视化仪表盘**
   - 将多维效率指标整合为 dashboard，辅助开发者快速诊断

5. **推广至 AI 训练场景**
   - 验证在大规模分布式训练（如 PyTorch + DDP + CUDA）中的适用性

---

> 💡 **总体评价**：该论文成功地将经典 POP 方法论推广至现代异构 HPC 系统，提出了首个结构清晰、语义明确、工程可行的 host-device 联合效率度量框架，填补了当前性能分析领域的空白，具有重要的理论价值与实践意义。

</details>

---

### 5. [Constitutive parameterized deep energy method for solid mechanics problems with random material parameters](https://arxiv.org/abs/2603.26030)

**Authors**: Zhangyong Liang, Huanhuan Gao  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 6.0  
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
在实际工程结构分析中，材料参数（如 Young's modulus $E$ 和 Poisson's ratio $\nu$）存在**随机性和不确定性**，通常表现为在一个区间内连续变化。传统数值方法（如 FEM）和现有的深度学习方法（如 DEM、PINNs、Neural Operators）在处理这类**连续多参数不确定性**时面临以下挑战：

- **FEM**：每次材料参数改变都需要重新进行网格划分、刚度矩阵组装和求解，计算成本极高。
- **标准 DEM / PINNs**：模型针对固定材料参数训练，参数一旦变化就必须从头开始重新训练。
- **Neural Operators（如 DeepONet、FNO）**：依赖大量高保真仿真数据生成训练集，数据获取成本巨大。

因此，如何实现对**未知材料参数组合的零样本（zero-shot）实时推理**，同时避免重复计算或数据依赖，是当前固体力学模拟中的关键瓶颈。

---

### 提出了什么新方法或新思路
本文提出了一种全新的纯物理驱动深度学习框架——**Constitutive Parameterized Deep Energy Method (CPDEM)**，其核心思想如下：

- 将材料本构参数（如 $E, \nu$）作为**显式输入变量**嵌入神经网络架构中，而非将其视为常数嵌入能量泛函。
- 设计了一个**参数化网络结构**，由三个模块组成：
  - `g_coord`：空间坐标编码器
  - `g_param`：材料参数编码器
  - `g_manifold`：流形子网，融合前两者输出以预测位移场 $u(X; \eta)$
- 在训练阶段通过最小化**期望势能（expected potential energy）** 来学习一个覆盖整个参数空间的**解流形（solution manifold）**：
  $$
  \mathcal{J}(u) = \mathbb{E}_{\eta \sim p(\eta)}[\Pi(u(X;\eta);\eta)]
  $$

该方法实现了“一次预训练，任意参数推理”的能力。

---

### 相比现有方法的优势

| 方法 | 是否需要重训练 | 是否依赖数据 | 是否支持连续参数变化 | 推理效率 |
|------|----------------|--------------|------------------------|----------|
| **FEM** | 是（每次重新求解） | 否 | 是 | 低（需矩阵求逆） |
| **Standard DEM/PINNs** | 是（每组参数重训） | 否 | 否 | 中等（训练耗时） |
| **Neural Operators** | 否（但需训练集） | 是（大量仿真数据） | 是 | 高（推理快） |
| **CPDEM（本文）** | ❌（仅一次预训练） | ❌（无监督） | ✅（完全支持） | ⚡️（毫秒级前向传播） |

> ✅ **三大优势总结**：
> 1. **无需数据**：纯物理驱动，不依赖任何标注或仿真数据；
> 2. **免重训练**：单次训练即可泛化至参数空间内任意新材料配置；
> 3. **高效可扩展**：支持大规模参数扫描与 UQ 分析，计算复杂度远低于 FEM。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **无真实实验数据集**，所有案例均为**学术基准问题（academic benchmarks）**，包括：
  - 1D 弹性杆（Linear Elastic Bar）
  - 2D/3D 悬臂梁（Cantilever Beam）
  - 超弹性体（Neo-Hookean, Mooney-Rivlin）
  - 大变形无摩擦接触问题（Ironing Problem）

- 所有“ground truth”均来自：
  - 解析解（Analytical solution，如 1D 杆）
  - 高分辨率 FEM 数值解（如 2D/3D 梁，采用 150×50 或更高网格）

---

### 实验设置和评估指标

#### 输入参数范围
- **随机材料参数建模为有界区间上的分布**：
  - $E \in [0.5, 3.0]$（1D 杆）
  - $E \in [800, 1200], \nu \in [0.25, 0.35]$（2D 超弹性梁）
  - 分布形式：均匀分布 $\mathcal{U}$ 或截断高斯分布 $N_{\text{trunc}}$

#### 网络结构
- 使用全连接神经网络（FNN）
- 架构示例：`[input_dim, 20, 20, output_dim]`
- 激活函数：ReLU / tanh
- 边界条件通过构造试函数嵌入（ansatz construction），确保强满足 Dirichlet 条件

#### 优化策略（两阶段训练）
1. **Adam 优化器**：初始快速下降，学习率 0.5
2. **L-BFGS**：后期精细收敛，提升物理一致性

#### 评估指标
- **相对 $L_2$ 误差**：
  $$
  \text{Rel } L_2 = \frac{\|u_{\text{pred}} - u_{\text{ref}}\|_2}{\|u_{\text{ref}}\|_2}
  $$
- **逐点绝对误差图**
- **损失函数收敛曲线**（total loss, energy loss, boundary loss）

---

### 基线方法对比
虽然未直接列出与其他模型的端到端性能比较表，但文中明确指出 CPDEM 相较于以下方法具有显著优势：

- **FEM**：需重复求解 → 计算昂贵
- **Standard DEM**：每组参数需独立训练 → 不适用于参数变化场景
- **Neural Operators（如 DeepONet）**：依赖大规模数据集 → 数据生成成本高

> 因此，CPDEM 的对比更多体现在**方法论层面的根本性改进**，而非单纯精度超越。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 问题类型 | 参数组合 | 相对 $L_2$ 误差 | 备注 |
|---------|----------|------------------|------|
| 1D 线弹性杆 | $E = 0.5, 1.5, 3.0$ | $6.289\times10^{-3}, 2.096\times10^{-2}, 3.081\times10^{-2}$ | 图3 |
| 2D 线弹性悬臂梁 | $(E,\nu)=(100,0.25)\sim(500,0.35)$ | 视觉无差异，误差量级 $10^{-2}$ | 图6 |
| 1D 超弹性杆（Neo-Hookean） | $E = 0.5, 1.5, 3.0$ | $4.196\times10^{-3}, 1.448\times10^{-2}, 2.737\times10^{-2}$ | 图7 |
| 2D 超弹性梁 | $(E,\nu)=(800,0.25)\sim(1200,0.35)$ | 与 FEM 几乎一致，局部误差 $<10^{-2}$ | 图9 |
| 3D 扭转梁 | 参数化 $E, \nu$ | 成功预测三维位移场 | 图10 |
| 接触问题（Ironing） | 变形+滑动加载步 | 收敛稳定，符合物理行为 | 图11 |

> ✅ 所有案例中，**单一 CPDEM 模型**成功预测了不同材料参数下的力学响应，且误差保持在较低水平。

---

### 与基线方法的对比结果

#### 计算效率对比（Scaling Analysis）
- **FEM / Standard DEM**：总时间 $\propto N_q \cdot T_{\text{solve/training}}$，随查询次数线性增长
- **CPDEM**：总时间 $T_{\text{pre-train}} + N_q \cdot T_{\text{infer}}$，其中 $T_{\text{infer}} \ll T_{\text{solve}}$（GPU 上为毫秒级）

> 当 $N_q \to \infty$ 时，CPDEM 的**摊销成本趋近于零**，特别适合大规模参数扫描、不确定性量化（UQ）、数字孪生等应用。

#### 泛化能力验证
- **In-Distribution Zero-Shot Inference**：对训练范围内未见参数组合仍能准确预测（误差与训练集相当）
- **Out-of-Distribution (OOD) Extrapolation**：
  - 直接推理误差增大，但仍保持物理合理性（无非物理解）
  - 结合 **Fast Fine-Tuning**（冻结编码器，仅微调流形网络）可在 <50 步 L-BFGS 内恢复高精度，节省 >95% 时间

---

### 消融实验结果（如有）

尽管未设专门消融章节，但以下设计体现了关键组件的有效性：

- **参数编码器 $g_{\text{param}}$ 的必要性**：若将 $\eta$ 直接拼接到 $X$ 中（即 $(X,\eta)$ 作为联合输入），会导致表示能力下降，无法有效捕捉材料影响的非线性映射。
- **双编码器结构 vs 单网络**：分离空间与材料特征有助于学习更鲁棒的解流形。
- **Fast Fine-Tuning 的有效性**：实验证明少量迭代即可适配特定参数，显著优于从头训练。

---

## 4. 关键结论和发现

### 论文的主要发现

1. ✅ **CPDEM 是首个纯物理驱动、支持连续多参数不确定性的深度学习求解器**，能够在没有数据的情况下学习参数化解流形。
2. ✅ 实现了真正的 **zero-shot inference**：对于训练区间内的任意新材料参数，无需再训练即可实时预测位移场。
3. ✅ 方法在 **线弹性、有限应变超弹性、接触力学** 等多种非线性问题上均表现出色，验证了其广泛适用性。
4. ✅ 提出的 **Fast Fine-Tuning 策略** 可进一步提升特定工况下的精度，兼顾通用性与精确性。
5. ✅ 计算效率远超传统方法，在大规模参数分析任务中展现出巨大潜力。

---

### 方法的局限性

1. **目前假设材料参数为全局均匀随机变量**，尚未考虑**空间变异的随机场**（如随机场 $E(X)$）；
2. **边界条件仍需手动构造**以满足 Dirichlet 约束，自动化程度有待提高；
3. 对极端 OOD 参数（如超出训练范围 20% 以上）的直接外推能力有限，仍需微调；
4. 当前实现基于规则采样点，对复杂几何和非结构网格的支持尚不充分。

---

### 未来工作方向（Prospects）

1. **扩展至空间相关随机材料场**：
   - 结合 Karhunen-Loève (KL) 展开建模功能梯度材料或多相复合材料；
2. **引入路径依赖本构关系**：
   - 推广至 elastoplasticity、damage mechanics 等不可逆过程；
3. **集成实时不确定性量化（UQ）与数字孪生系统**：
   - 利用 CPDEM + Fast Fine-Tuning 实现在线自适应更新；
4. **结合多尺度建模框架**：
   - 在微观结构层面引入随机性，向上游传递至宏观响应；
5. **开发开源工具包**，推动物理驱动 AI 在工业 CAE 中的应用落地。

---

> 🔚 **总结一句话**：  
> **CPDEM 开辟了一条全新的路径——将材料参数“编程化”进入神经网络，使深度学习求解器真正具备应对现实世界材料不确定性的能力，为下一代智能仿真引擎奠定了基础。**

</details>

---

### 6. [Are LLM-Enhanced Graph Neural Networks Robust against Poisoning Attacks?](https://arxiv.org/abs/2603.26105)

**Authors**: Yuhang Ma, Jie Wang, Zheng Yan  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.26105v1  

#### Abstract
Large Language Models (LLMs) have advanced Graph Neural Networks (GNNs) by enriching node representations with semantic features, giving rise to LLM-enhanced GNNs that achieve notable performance gains. However, the robustness of these models against poisoning attacks, which manipulate both graph st...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Are LLM-Enhanced Graph Neural Networks Robust against Poisoning Attacks?

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文系统性地研究了**LLM-enhanced GNNs**在**Poisoning Attacks**下的鲁棒性问题。尽管LLM增强的GNN在性能上表现出色，但其在同时遭受图结构和文本属性污染攻击时的安全性尚未被深入探索。

现有研究存在三大局限（L1–L3）：
- **L1：模型与攻击覆盖不全**：仅评估少数LLM/LM与GNN组合，且攻击类型单一。
- **L2：数据泄露风险**：使用传统数据集（如Cora、Pubmed），这些数据可能已被LLM预训练所包含，导致“ground truth leakage”，高估鲁棒性。
- **L3：缺乏攻防视角的未来指导**：缺少对联合攻击与防御机制的讨论。

### 提出了什么新方法或新思路
作者提出了一套**全面的鲁棒性评估框架**（Robustness Assessment Framework），具有以下创新点：

- **多维度攻击评估**：首次同时评估**六种结构攻击**（3种Targeted + 3种Non-targeted）和**三种文本攻击**（Character/Word/Sentence-level）。
- **引入Post-LLM数据集**：采用**Tape-arxiv23**这一2023年后发布的数据集，避免因LLM预训练数据重叠导致的评估偏差。
- **提出新型联合攻击与防御机制**：
  - **Combined Attack**：结合结构与文本攻击，验证更强攻击效果。
  - **Graph Purification Defense**：基于LLM生成的嵌入进行边过滤，提升鲁棒性。

### 相比现有方法的优势
- **更全面**：评估了**24种victim model组合**（8种feature enhancer × 3种GNN backbone）。
- **更公平**：通过引入post-LLM数据集，减少ground truth leakage影响。
- **更具前瞻性**：从攻防双重视角提出未来研究方向，推动领域发展。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
| 数据集 | 类型 | 节点数 | 边数 | 类别数 | 特征方法 | 领域 |
|--------|------|--------|------|--------|----------|------|
| **Cora** | Citation | 2,708 | 5,429 | 7 | BoW | 引用网络 |
| **Pubmed** | Citation | 19,717 | 44,338 | 3 | TF-IDF | 医学文献 |
| **Ogbn-products (subset)** | E-commerce | 12,011 | 21,987 | 47 | BoW | 商品共购 |
| **Tape-arxiv23 (subset)** | Citation | 13,167 | 23,735 | 40 | Word2Vec | 新兴arXiv论文 |

> ✅ **关键设计**：Tape-arxiv23发布于LLM兴起之后，极大降低了预训练数据泄露风险。

---

### 实验设置和评估指标

#### 攻击设置
- **结构攻击（Structural Poisoning）**：
  - **Targeted**：`Nettack`, `SGA`, `NAG-R`
  - **Non-targeted**：`Mettack`, `DICE`, `PGA`
- **文本攻击（Textual Poisoning）**：
  - **Character-level**：`DeepWordBug (DWord)`
  - **Word-level**：`Bert-Attack (BertAtk)`
  - **Sentence-level**：`MAYA`

所有攻击在**灰盒设定**（gray-box）下进行：攻击者知晓图结构和节点文本，但不知晓victim model架构与参数。

#### 评估指标
| 指标 | 公式 | 含义 |
|------|------|------|
| **Accuracy (ACC)** | $\frac{\text{正确预测数}}{\text{总数}}$ | 绝对性能，越高越鲁棒 |
| **Relative Drop in Accuracy (RDA)** | $\frac{ACC_{\text{clean}} - ACC_{\text{attack}}}{ACC_{\text{clean}}}$ | 相对下降，越低越鲁棒 |

此外还分析了：
- **Embedding Visualization**（t-SNE）
- **Embedding Separability**（DBI, Silhouette Score）
- **Label & Structural Information Preservation**（Homophily, Mutual Information）

---

### 基线方法对比
- **Baseline**：浅层嵌入方法（Shallow emb.），包括 BoW、TF-IDF、Word2Vec。
- **Victim Models**：
  - **Feature Enhancers (8)**：
    - LLM-Explanation: `TAPE`, `KEA`
    - LLM-Embedding: `LLaMA`, `TE3L`, `Linq`
    - LM-Embedding: `SimTeg`, `E5-Large`, `ModernBert`
  - **GNN Backbones (3)**：`GCN`, `GAT`, `GraphSAGE`

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在结构攻击下的表现（以Mettack高扰动为例）
| 模型 | Cora (ACC) | Pubmed (ACC) | Ogbn-products (ACC) | Tape-arxiv23 (ACC) |
|------|------------|--------------|---------------------|--------------------|
| **Shallow emb.** | 74.60 | 74.12 | 63.78 | 51.03 |
| **TAPE** | 78.02 (+3.42) | 84.91 (+10.79) | 84.26 (+20.48) | 71.33 (+20.30) |
| **E5-Large** | 80.60 (+6.00) | 84.71 (+10.59) | 83.23 (+19.45) | 70.04 (+19.01) |

> ✅ **观察**：LLM-enhanced模型显著优于baseline，尤其在大规模数据集上优势更大。

#### RDA 对比（越低越好）
| 模型 | Cora (RDA↓) | Pubmed (RDA↓) | Ogbn-products (RDA↓) | Tape-arxiv23 (RDA↓) |
|------|-------------|---------------|-----------------------|----------------------|
| **Shallow emb.** | 7.92 | 14.02 | 3.29 | 11.59 |
| **TAPE** | 4.55 | 8.82 | 1.84 | 4.22 |
| **E5-Large** | 4.51 | 8.97 | 2.04 | 4.16 |

> ✅ **RDA更低** → 更强相对鲁棒性。

---

### 与基线方法的对比结果
- **LLM-enhanced GNNs 在所有攻击场景下均显著优于 shallow embedding baseline**。
- **在高扰动率下，性能差距进一步拉大**，说明LLM增强的嵌入更具抗干扰能力。
- **部分LLM-enhanced模型甚至出现负RDA**（即攻击后准确率更高），表明扰动可能起到正则化作用，提升泛化。

#### 文本攻击结果（MAYA最严重）
| 模型 | Cora (ACC) | Pubmed (ACC) | Ogbn-products (ACC) | Tape-arxiv23 (ACC) |
|------|------------|--------------|---------------------|--------------------|
| **Shallow emb.** | 80.54 | 85.54 | 65.03 | 53.14 |
| **TAPE** | 82.55 | 88.96 | 85.48 | 71.33 |
| **E5-Large** | 83.38 | 86.58 | 84.03 | 69.69 |

> ✅ 文本攻击整体效果弱于结构攻击，但LLM-enhanced仍更鲁棒。

---

### 消融实验结果（关键分析维度）

| 分析维度 | 发现 |
|--------|------|
| **Embedding Visualization (t-SNE)** | LLM/LM生成的嵌入聚类更清晰，类别可分性更强（如TAPE、E5-Large）。 |
| **Embedding Separability (DBI, Silhouette)** | LLM-enhanced嵌入DBI更低、Silhouette更高，表示聚类质量更好。 |
| **Label Info Preservation (Homophily, ELMI)** | LLM嵌入保留更多标签语义信息，homophily高达90%以上。 |
| **Structural Info Preservation (ESMI, NCon)** | 即使在攻击后，LLM嵌入仍能较好保持邻居一致性（NCon）。 |

> 🔍 **根本原因**：LLM/LM生成的嵌入编码了更丰富的**label语义**与**局部结构信息**，使其在图结构被破坏时仍能维持判别能力。

---

## 4. 关键结论和发现

### 论文的主要发现

✅ **Key Takeaway 1**：  
**LLM-enhanced GNNs 对结构与文本Poisoning攻击均表现出显著更强的鲁棒性**，尤其在高扰动下优势更明显。

✅ **Key Takeaway 2**：  
鲁棒性提升**并非源于ground truth leakage**，而是来自LLM/LM在架构层面的集成优势。

✅ **Key Takeaway 3**：  
**LLM/LM生成的嵌入质量更高**，能有效编码label和结构信息，是鲁棒性的关键来源。

✅ **Key Takeaway 4**：  
**GraphSAGE**作为GNN backbone展现出最强鲁棒性，因其保留自表示并使用均值聚合，降低邻居扰动影响。

✅ **Key Takeaway 5**：  
**联合攻击（Combined Attack）比单一攻击更有效**，尤其是结构+句子级文本攻击。

✅ **Key Takeaway 6**：  
提出的**Graph Purification Defense**基于LLM嵌入相似度过滤可疑边，在多个数据集上显著提升鲁棒性，甚至超过clean graph表现。

---

### 方法的局限性

- **计算成本高**：LLM推理与微调开销较大，限制在资源受限场景的应用。
- **防御依赖阈值选择**：Graph Purification需手动设定相似度阈值，缺乏自适应机制。
- **未考虑动态图**：当前评估基于静态图，无法反映真实世界中持续更新的图结构。
- **攻击假设较强**：灰盒设定虽合理，但仍假设攻击者掌握完整图结构，实际中可能受限。

---

### 未来工作方向

#### Offensive Perspectives（攻击侧）
1. **LLM作为知识增强器**：利用LLM补全缺失链接（link prediction），辅助部分结构知识下的攻击。
2. **更隐蔽的结构攻击**：设计保留图结构性质（如clustering coefficient）的攻击，提高隐蔽性。
3. **更高效的文本攻击**：开发轻量级替代方案，降低使用LLM作为代理模型的成本。
4. **协调式联合攻击**：动态选择最优结构或文本扰动，最大化损失函数。

#### Defensive Perspectives（防御侧）
1. **鲁棒架构设计**：探索更适合LLM-GNN融合的GNN结构，如强化自监督学习。
2. **基于LLM的图净化**：扩展本文提出的purification方法，构建端到端防御流程。
3. **对抗扰动作为数据增强**：将可控的对抗扰动用于训练，提升泛化与鲁棒性。

---

### 开源贡献
作者已将整个评估框架开源：  
👉 [GitHub: https://github.com/CyberAlSec/LLMEGNNRP](https://github.com/CyberAlSec/LLMEGNNRP)

为后续研究提供了标准化、可复现的LLM-enhanced GNN鲁棒性评测平台。

</details>

---

### 7. [MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference](https://arxiv.org/abs/2603.26557)

**Authors**: Joris K\"oster, Zixuan Liu, Siavash Khajavi, Zizhan Zheng  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.26557v1  

#### Abstract
Large Language Models (LLMs) deliver strong performance but incur high inference cost in real-world services, especially under workloads with repeated or near-duplicate queries across users and sessions. In this work, we propose MemBoost, a memory-boosted LLM serving framework that enables a lightwe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在现实世界的 **LLM 服务**中，存在大量重复或语义近似的查询（如多用户反复提问相似问题），直接对每个请求调用大型模型（Large LLM）会造成严重的计算资源浪费，导致高昂的 **inference cost**（推理成本）。尽管已有 **RAG**（Retrieval-Augmented Generation）和 **semantic caching** 等技术尝试缓解该问题，但它们通常：
- 仅用于单次问答的知识增强，不支持跨会话的答案复用；
- 缺乏动态写回机制以持续扩展记忆；
- 无法智能地将复杂任务路由到更强模型。

因此，亟需一个系统级框架，在保证高质量输出的同时，显著降低对昂贵大模型的依赖。

---

### 🚀 提出的新方法：MemBoost
作者提出 **MemBoost** —— 一种内存增强型 LLM 推理框架，其核心思想是将推理过程转化为“**retrieve-or-escalate**”决策流程。它由三个组件构成：

| 组件 | 功能 |
|------|------|
| **Associative Memory Engine (AME)** | 外部语义记忆库，存储历史问答对（question-answer pairs），支持快速向量检索与写回更新 |
| **Large-LLM Oracle** | 高能力、高成本的大模型，作为 fallback 用于处理难以通过检索解决的复杂/不确定查询 |
| **Meta Controller (MC)** | 轻量级小模型（如 Qwen-3.5-2B），负责协调整个流程：<br>① 决定是否从 AME 中复用答案；<br>② 若需升级，则调用 Oracle；<br>③ 判断是否将新生成的优质答案写入 AME |

> 💡 **创新亮点**：
> - 支持 **answer reuse** 和 **continual memory growth**，实现长期成本优化；
> - 引入 **cost-aware routing**，结合质量与开销进行动态调度；
> - 区别于标准 RAG，MemBoost 更适用于 **interactive, repeat-heavy workloads**（交互式高频重复场景）。

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | MemBoost 的改进 |
|------|--------|------------------|
| Standard RAG | 主要用于单轮知识增强，不支持跨请求答案复用 | 显式设计用于多轮、跨用户的语义缓存与复用 |
| Semantic Caching | 仅缓存输出，缺乏智能路由机制 | 加入 Meta Controller 实现“检索→决策→升级→写回”闭环 |
| Multi-LLM Routing | 仅在不同模型间切换，无记忆复用机制 | 将 memory hit 作为最廉价的“零代模式”，优先于任何模型调用 |

> ✅ 总体优势：**在保持接近 Oracle 模型质量的前提下，大幅减少对昂贵大模型的调用次数，从而显著降低总推理成本和延迟**。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **MMLU-Pro**（Wang et al., 2024）：一个更具挑战性的多任务语言理解基准，涵盖多个学科领域，用于评估 LLM 的综合能力。
- 使用其中的 **Business 类别**（共 768 条样本）构建测试流，因其足够覆盖模拟请求总量。

---

### ⚙️ 实验设置
- **查询流生成**：采用 **Zipf 分布采样** 构建具有重尾访问特征的 workload，模拟真实场景中的高频重复查询行为。
  - 控制参数：Zipf exponent $ \alpha \in \{0.8, 1.1, 1.4\} $
  - $ \alpha $ 越大，表示少数问题被频繁访问，重复率更高
- **总请求数**：5,000 步（steps）
- **Embedding 模型**：`all-MiniLM-L6-v2` 用于编码 QA 对
- **检索引擎**：基于 **FAISS** 的近似最近邻搜索（cosine similarity），top-k=3
- **部署环境**：
  - 所有模型使用 **vLLM** 部署
  - MC 与 Oracle 各运行在一个独立的 **NVIDIA A100 80GB GPU** 上
  - 温度设为 0.0 以确保响应确定性

---

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 回答正确率，对比 ground truth label |
| **Memory-use Rate ($ I_t $)** | 在滑动窗口内，使用 AME 成功返回答案的比例（即未调用 Oracle 的比例），反映成本节约程度 |
| **Response Latency** | 平均响应时间（过去 100 步的均值） |
| **Total Inference Cost ($ C_r $)** | 综合考虑 Oracle、MC 和 retrieval 的开销，其中 $ c_o \gg c_M + c_R $ |

---

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| Small LLM Baselines | 直接使用 Qwen3.5-2B、Ministral-3-3B、Qwen3-4B 等小模型单独服务所有请求 |
| Oracle-only | 全部请求均由 Qwen3-14B-FP8-dynamic 处理（高质量但高成本） |
| MemBoost variants | 不同 MC 模型组合下的 MemBoost 版本（如 MemBoost + Qwen3.5-2B） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Model | Zipf 0.8 | Zipf 1.1 | Zipf 1.4 |
|-------|----------|----------|----------|
| Qwen3.5-2B (baseline) | 50.0% | 43.5% | 37.1% |
| Qwen3-4B-Instruct (baseline) | 74.5% | 75.6% | 80.5% |
| **Qwen3-14B (Oracle)** | **76.4%** | **79.9%** | **85.0%** |
| **MemBoost (Qwen3.5-2B)** | **76.7%** | **81.8%** | **87.4%** |
| **MemBoost (Ministral-3-3B)** | 76.2% | 79.7% | 85.0% |
| **MemBoost (Qwen3-4B)** | 76.1% | 79.8% | 85.0% |

> ✅ **关键观察**：
> - 所有 MemBoost 变体均 **显著超越对应的小模型 baseline**
> - 即使使用最小的 **Qwen3.5-2B 作为 MC**，MemBoost 也能达到甚至 **超过 Oracle 的准确率**
> - 准确率提升归因于：一旦某个问题被 Oracle 正确解答并写回 AME，后续重复请求可直接命中高质量缓存，避免生成错误

---

### 📉 成本与效率表现（Figure 2 & 3）

#### ▶ Memory-use Rate（图 2）
- 随着时间推移，$ I_t $ 持续上升 → 表明 AME 中积累的有效缓存越来越多
- 在 Zipf 1.4（最高重复率）下，最终 memory-use rate 接近 **90%+**
- 结论：**重复率越高，MemBoost 节省的成本越多**

#### ▶ Response Latency（图 3）
- MemBoost 的平均延迟随时间下降，并稳定低于 Oracle-only 基线
- 因为越来越多的回答来自低延迟的 AME 检索，而非耗时的 LLM 生成

> ✅ **综合结论**：MemBoost 实现了“**越用越快、越用越便宜**”的效果。

---

### ❌ 消融实验（文中未明确列出，但从机制分析可得）
虽然没有显式的 ablation study 表格，但以下逻辑成立：
- 若关闭 write-back 机制 → 记忆无法增长 → memory hit 率不会随时间上升 → 成本节省有限
- 若移除 Meta Controller → 退化为纯 semantic cache 或固定路由策略 → 无法灵活应对不确定性查询
- 若仅用 RAG without reuse → 每次仍需生成，无法实现“零代”节省

> 因此，**三组件协同运作是 MemBoost 成功的关键**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MemBoost 能有效减少对昂贵 Large LLM 的调用频率**，在高重复 workload 下，超过 90% 的请求可通过 AME 快速响应。
2. **答案质量不仅不下降，反而可能提升**：得益于高质量历史答案的复用，避免了重复生成带来的随机误差。
3. **系统具备自进化能力**：随着服务时间延长，AME 不断积累优质知识，整体效率持续优化。
4. **尤其适合 repeat-heavy 场景**：当 Zipf 参数增大（即查询更集中）时，MemBoost 的优势更加明显。

---

### ⚠️ 局限性
1. **尚未在开放域任务上验证**：当前实验集中在 MMLU-Pro 这类封闭、短答案的 QA 任务，未涉及长文本生成、代码生成等更复杂输出形式。
2. **依赖语义相似性匹配可靠性**：若 AME 返回错误但看似相关的答案（false positive），可能导致误导性输出。
3. **仿真 workload 简化现实复杂性**：
   - 当前使用的是从固定数据集中按 Zipf 抽样的问题，而实际用户查询更多为 paraphrased 或语义重叠的形式；
   - 缺乏上下文感知的 multi-turn 对话压力测试。

---

### 🔮 未来工作方向
1. **增强 AME 的鲁棒性**：引入置信度评分、多跳验证机制，防止 false hits 导致错误传播。
2. **扩展至多模态与长输出任务**：探索如何高效缓存图像生成、代码片段等非结构化输出。
3. **动态缓存管理策略**：研究基于访问频率、时效性、类别重要性的智能 eviction policy。
4. **在线学习与反馈闭环**：结合用户反馈调整 write-back 策略和 routing 决策，实现持续优化。

---

## ✅ 总结
**MemBoost** 是首个将 **semantic caching**、**retrieval-augmented generation** 与 **multi-LLM routing** 深度融合的系统级框架。它通过“**检索→决策→升级→写回**”的闭环机制，在交互式、高重复的 LLM 服务场景中实现了：
- ✅ **接近 Oracle 的回答质量**
- ✅ **远低于 Oracle 的推理成本**
- ✅ **持续降低的响应延迟**

> 它为下一代高效、经济、可持续的 LLM 应用提供了重要的架构范式。

</details>

---

### 8. [TinyML for Acoustic Anomaly Detection in IoT Sensor Networks](https://arxiv.org/abs/2603.26135)

**Authors**: Amar Almaini, Jakob Folz, Ghadeer Ashour  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.26135v1  

#### Abstract
Tiny Machine Learning enables real-time, energy-efficient data processing directly on microcontrollers, making it ideal for Internet of Things sensor networks. This paper presents a compact TinyML pipeline for detecting anomalies in environmental sound within IoT sensor networks. Acoustic monitoring...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TinyML for Acoustic Anomaly Detection in IoT Sensor Networks**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
本论文针对 **IoT sensor networks** 中基于声音的异常检测面临的三大挑战：
- **高延迟**：依赖云端处理音频数据导致响应延迟；
- **高功耗与带宽消耗**：持续传输原始音频数据对资源受限设备不现实；
- **隐私风险**：将敏感环境音频上传至云端存在泄露风险。

这些问题在分布式、低功耗的 IoT 部署中尤为突出。

### 🚀 **提出了什么新方法或新思路**
提出了一种端到端的 **TinyML pipeline**，用于在微控制器（MCU）上实现嵌入式 **acoustic anomaly detection**，其核心流程包括：
1. 使用 **MFCC**（Mel Frequency Cepstral Coefficients）进行高效音频特征提取；
2. 构建一个轻量级全连接神经网络分类器；
3. 对模型进行 **8-bit quantization** 并转换为 **TensorFlow Lite Micro (TFLM)** 格式，适配边缘部署；
4. 将 UrbanSound8K 数据集重构为二分类任务（normal vs. anomalous sounds），以支持关键事件识别。

> 🔍 创新视角：将传统 anomaly detection 转换为 **有监督的 binary classification**，利用已知“紧急”声学事件（如警笛、枪声）进行目标化检测，提升实用性。

### ⭐ **相比现有方法的优势**
| 方面 | 本文优势 |
|------|--------|
| **可复现性** | 完全基于公开数据集（UrbanSound8K）和标准工具链（TensorFlow + librosa），无需定制硬件即可复现； |
| **通用性** | 不绑定特定传感器平台，适用于多种 MCU（如 ARM Cortex-M 系列）； |
| **性能与效率平衡** | 在仅占用 ~60kB 内存的情况下达到 91% 准确率，优于多数同类嵌入式方案； |
| **部署兼容性** | 支持 TFLM，便于实际部署于资源受限设备； |

相较于 Hammad et al. [2] 的无监督 LSTM autoencoder 或 Rani [4] 的小型分类器，本文方法在准确性和实用性之间取得了更好平衡。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
- **UrbanSound8K**：
  - 包含 8,732 条标注的城市环境音频片段；
  - 覆盖 10 类城市声音（如空调、引擎、警笛、枪声等）；
  - 按照标准 fold 划分训练/测试集，确保可比性。

> 数据预处理：所有音频重采样至 **16kHz**，裁剪或填充至固定长度；使用 **40ms 帧长、50% 重叠** 提取 **13 维 MFCC** 特征，生成 13×32 的二维特征图并展平为输入向量。

### ⚙️ **实验设置**
- **任务形式**：二分类问题  
  - `Normal`：背景音（如空调、儿童玩耍、街道交谈）  
  - `Anomalous`：高显著性/紧急声音（如警笛、枪声、电锯、狗吠）
- **模型架构**：
  - 全连接网络：`[128, 64]` 隐藏层（ReLU 激活）
  - Dropout rate = 0.2（防过拟合）
  - 输出层：Sigmoid + Binary Cross-Entropy Loss
- **优化器**：Adam（初始学习率 0.001），配合 ReduceLROnPlateau 和 Early Stopping（patience=5）
- **量化与转换**：
  - 浮点模型 → 8-bit int quantized model
  - 转换为 **TensorFlow Lite for Microcontrollers (TFLM)** 格式
- **训练环境**：Python 3.10, TensorFlow 2.15, Librosa 0.10.1；普通工作站（i7, 16GB RAM, 无 GPU）

### 📊 **评估指标**
| 指标 | 描述 |
|------|------|
| **Accuracy** | 整体分类正确率 |
| **F1-Score** | 精确率与召回率的调和平均（尤其关注类别均衡性） |
| **ROC AUC** | 分类器判别能力的整体衡量 |
| **Average Precision (AP)** | Precision-Recall 曲线下面积，适合不平衡场景 |
| **Class-wise Metrics** | 分析 normal / anomalous 各类别的 precision、recall、support |

> ❗未进行真实 MCU 上的实时推理测试（留待未来工作），但在设计阶段充分考虑了内存和计算限制。

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据（Quantized Model）**

| 模型 | Accuracy | F1-Score | ROC AUC | Avg. Precision |
|------|----------|-----------|---------|----------------|
| **Original (float32)** | 0.95 | 0.95 | 0.991 | 0.992 |
| **Quantized (int8)** | **0.91** | **0.91** | **0.970** | **0.970** |

> ✅ 量化后仅损失约 4% 性能，但模型大小从 ~250kB 下降至 **~60kB**，更适合嵌入式部署。

### 📋 **类别级评估结果（Quantized Model）**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal (0) | 0.88 | 0.94 | 0.91 | 3,996 |
| Anomalous (1) | 0.94 | 0.89 | 0.91 | 4,283 |
| **Macro Average** | — | — | **0.91** | — |

> ✅ 表现出良好的类别平衡性，无明显偏向某一类，说明模型在安全监控等应用中具有实用价值（避免漏报或误报过多）。

### 🔍 **与基线方法对比**
| 方法 | 准确率 | 是否支持 MCU | 是否使用公开数据集 | 备注 |
|------|--------|---------------|--------------------|------|
| Hammad et al. [2] (LSTM Autoencoder) | ~85–90%* | 是 | 否（需实测数据） | 无监督，依赖重建误差，难以复现 |
| Rani [4]（嵌入式分类） | ~80–85% | 是 | 是（UrbanSound） | 报告精度较低，部署细节有限 |
| **本文方法** | **91%** | **是（TFLM兼容）** | **是** | 可复现、高性能、轻量化 |

> *注：原文未直接比较，依据文献描述估算。

### 🔁 **消融实验（隐含分析）**
虽然未明确列出消融实验表格，但文中通过以下方式体现设计选择的有效性：
- **量化影响分析**：显示量化带来可控性能下降（-4% Acc），换取显著压缩；
- **训练曲线分析**（Fig. 5 & 6）：验证无过拟合，收敛稳定；
- **混淆矩阵分析**：常见错误集中在频谱相似的声音间（如街边音乐 ↔ 警笛），表明特征表达仍有改进空间。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **TinyML 完全可行用于 acoustics-based anomaly detection**：即使在极小模型（62k 参数）下也能实现高达 91% 的准确率；
2. **MFCC + Dense Network 是高效的嵌入式组合**：在资源受限条件下仍能保持良好性能；
3. **量化对性能影响可控**：8-bit quantization 仅造成轻微下降，却极大提升部署可行性；
4. **模型具备良好泛化能力和类别平衡性**：precision 和 recall 在两类间均衡，适合实际部署；
5. **系统完全可复现且硬件无关**：为后续研究提供了标准化 baseline。

### ⚠️ **局限性**
- **尚未在真实 MCU 上运行**：当前仅为模拟和格式转换，缺乏真实的 inference latency、memory usage 和 power consumption 数据；
- **静态音频片段假设**：未处理连续音频流（streaming inference）；
- **依赖预定义异常类别**：无法检测未知类型的异常声音（非开放集检测）；
- **特征工程较传统**：仅用 MFCC，未探索更先进的时频表示（如 log-melspectrogram、learned features）；

### 🔮 **未来工作方向**
1. **真实硬件部署与性能测量**：
   - 在 ESP32、RP2040 等典型 MCU 上测试 **inference time**、**RAM usage** 和 **power consumption**；
2. **进一步模型压缩**：
   - 探索 **pruning**、**knowledge distillation** 或 **binary neural networks** 进一步降低开销；
3. **扩展至多类别与连续检测**：
   - 支持更多 urban sound 类别；
   - 实现 **real-time streaming inference**，适应长期监测需求；
4. **结合上下文信息**：
   - 融合时间模式、地理位置或其他传感器数据（如 motion、light）提升判断准确性；
5. **迁移至真实 IoT 网络验证**：
   - 在真实城市环境中收集数据，评估鲁棒性与噪声干扰下的表现。

---

## ✅ 总结一句话：
> 本文成功构建了一个**高效、可复现、适用于 MCU 的 TinyML pipeline**，实现了在边缘设备上对城市声音中的异常事件进行高精度检测（**91% accuracy, 0.91 F1**），为构建隐私保护、低延迟、节能的智能 IoT 声学监控系统提供了坚实基础。

</details>

---

### 9. [Automatic Laplace Collapsed Sampling: Scalable Marginalisation of Latent Parameters via Automatic Differentiation](https://arxiv.org/abs/2603.26644)

**Authors**: Toby Lovick, David Yallup, Will Handley  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.26644v1  

#### Abstract
We present Automatic Laplace Collapsed Sampling (ALCS), a general framework for marginalising latent parameters in Bayesian models using automatic differentiation, which we combine with nested sampling to explore the hyperparameter space in a robust and efficient manner. At each nested sampling like...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Automatic Laplace Collapsed Sampling: Scalable Marginalisation of Latent Parameters via Automatic Differentiation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在高维贝叶斯模型中，**计算 Bayesian evidence（边际似然）** 是一个核心但极具挑战性的任务。传统的 **nested sampling** 虽能直接估计证据 $ Z $，但其计算成本随参数维度 $ D $ 呈立方增长（$ \mathcal{O}(D^3) $），难以应用于大规模问题。

许多实际应用（如天体物理中的超新星宇宙学、引力波群体推断等）具有典型的结构：  
- 少量感兴趣的参数 $ \theta \in \mathbb{R}^{d_\theta} $
- 大量潜变量 $ z \in \mathbb{R}^{d_z} $（如每个观测对象的噪声、校准偏移等）

传统做法要么忽略潜变量不确定性，要么牺牲效率进行全联合采样。

---

### 提出的新方法：ALCS（Automatic Laplace Collapsed Sampling）
ALCS 是一种通用框架，用于在贝叶斯模型中高效地边缘化（marginalise）潜变量 $ z $，从而显著降低有效维度。

#### 核心思想：
在每次外层 sampler（如 nested sampling）对 $ \theta $ 进行 likelihood 评估时：
1. **Step 1: MAP Optimisation**  
   固定 $ \theta $，通过自动微分（autodiff）优化求解条件后验 $ p(z|\theta, D) $ 的最大后验估计（MAP）：
   $$
   \hat{z}(\theta) = \arg\max_z \log p(D, z|\theta)
   $$

2. **Step 2: Laplace Approximation**  
   在 MAP 点处计算负 Hessian $ H(\theta) = -\nabla^2_z \log p(D,z|\theta)|_{z=\hat{z}} $，并用 Laplace 近似得到边缘似然：
   $$
   \log \mathcal{L}_{\text{ALCS}}(\theta) = \log p(D,\hat{z}|\theta) + \frac{1}{2}\log(2\pi)^{d_z} - \frac{1}{2}\log\det H(\theta)
   $$

该边缘似然被作为外层 sampler 的输入，使得整个采样过程仅在低维 $ \theta $ 空间进行。

---

### 相比现有方法的优势

| 方法 | 缺陷 | ALCS 改进 |
|------|------|-----------|
| Full nested sampling | 维度灾难，$ \mathcal{O}((d_\theta + d_z)^3) $ | 降为 $ \mathcal{O}(d_\theta^3 + d_z^2) $，仅探索 $ d_\theta $ 空间 |
| Type-II ML / Evidence Approximation | 只优化 $ \theta $，不积分，无法获得证据分布 | 使用 nested sampling 对 $ \theta $ 完整积分，输出完整 posterior 和 evidence |
| INLA | 需要在网格上固定 $ \theta $，受限于 $ d_\theta \leq 5 $ | 支持任意 $ d_\theta $，且支持非高斯扩展 |
| 手工推导梯度/Hessian | 不通用，工程复杂 | **全自动**：只需提供可微 log-joint，无需手动推导任何导数 |

✅ **“Automatic” 是关键词**：完全依赖 JAX 自动微分，无需手写梯度或 Hessian 表达式。  
✅ **GPU 并行化**：利用 `jax.vmap` 实现跨 live points 和 latent blocks 的并行计算，适合现代硬件加速。

---

## 2. 核心实验方法和设置

### 数据集与模型
实验涵盖三类典型场景：

| 类型 | 模型 | 特点 |
|------|------|------|
| 合成基准 | **Supernova Cosmology**（ACDM/wCDM） | 潜变量为每颗超新星的 stretch/color，$ d_z $ 最高达 **25,600**；有解析解作 ground truth |
| 控制非高斯性 | **Student-t Latent Prior** | 构造重尾潜变量后验，测试 Laplace 的失效与修正能力 |
| 已知困难案例 | **Tanh Funnel** | 非线性饱和导致潜变量后验出现平坦肩部，Laplace 近似失效 |
| 标准测试套件 | **Inference Gym** 中的六个模型 | 包括 Eight Schools、Radon、Brownian Motion、LGCP、Stochastic Volatility、IRT |

---

### 实验设置
- **外层 Sampler**：BlackJAX 实现的 **nested slice sampling**，$ m=500 $ live points
- **终止条件**：当 $ \log Z_{\text{live}} - \log 2 < -3 $
- **硬件平台**：NVIDIA H200 GPU，利用 JAX 的 `vmap`, `jit`, `hessian` 等原语
- **重复次数**：所有实验运行 $ R=5 $ 次不同随机种子，报告均值 ± 标准差

---

### 评估指标
| 指标 | 描述 |
|------|------|
| $ \Delta \log Z $ | ALCS 与 ground truth（full NS 或解析解）之间的证据误差（单位：nats） |
| Wall Time | 总运行时间（秒） |
| Speedup | 相对于 full joint NUTS 或 NS 的加速比 |
| ESS/K | Importance Sampling Effective Sample Size / K，用于诊断 Laplace 近似的有效性（越高越好，理想为 1） |

---

### 基线方法对比
| 基线 | 说明 |
|------|------|
| Full Joint Nested Sampling | 在 $ (\theta, z) $ 全空间运行 nested sampling，作为黄金标准（但高维下不可行） |
| Full Joint NUTS | 使用 HMC/NUTS 采样全联合 posterior，用于 posterior 验证 |
| Analytic Marginalisation | 当存在闭式解时（如高斯共轭模型），作为精确参考 |
| Gaussian ALCS vs Student-t ALCS | 测试更高阶近似的改进效果 |

---

## 3. 主要实验结果和性能指标

### ✅ 成功案例：高维可扩展性验证（Supernova Cosmology）

| 测试 | 设置 | 结果 |
|------|------|------|
| Test 1: 扩展 $ N $（固定 $ d_{z,\text{block}}=2 $） | $ N \in \{64, ..., 2048\}, d_z \leq 4096 $ | - $ \Delta \log Z < 0.25 $ nats<br>- 单次调用耗时恒定 ~1.2ms（GPU 并行饱和）<br>- 总时间从 12s 到 45s |
| Test 2: 扩展单个潜变量维度 | $ d_{z,\text{block}} \in \{2,...,256\}, d_z \leq 25,600 $ | - $ \Delta \log Z < 0.06 $ nats<br>- 总时间从 9s 到 693s（约 12 分钟）<br>- Full NS 投影需 **37年** |

> ⚡️ **结论**：在潜变量服从高斯分布时，ALCS 几乎达到解析精度，且 wall time 几乎独立于 $ N $，仅受 $ D_{KL} $ 影响。

---

### ✅ 改进案例：Student-t 扩展（处理重尾分布）

| $ N_{\text{obj}} $ | Gaussian ALCS Error | Student-t ALCS Error | 改进幅度 |
|---------------------|------------------------|-------------------------|----------|
| 50 | -1.00 nats | -0.17 nats | ↑83% |
| 100 | -1.86 nats | +0.12 nats | ↑接近消除偏差 |

- **机制**：通过第四阶导数估计 excess kurtosis，并拟合 Student-t 分布自由度 $ \nu_j $
- **代价增加小**：每方向额外两次 autodiff 调用，局部无采样
- **ESS/K 提升**：Student-t proposal 的 IS ESS/K 显著高于 Gaussian（如 0.49 vs 0.38 @ $ N=50 $）

> ✅ **证明**：ALCS 框架可自然扩展至更丰富的局部近似族（如 Student-t），提升对非高斯潜变量的适应性。

---

### ❌ 失败案例：Tanh Funnel（几何病态）

| 指标 | 数值 |
|------|------|
| True $ \log Z $ | -15.94 ± 0.05 |
| ALCS $ \log Z $ | -16.68 ± 0.04 |
| $ \Delta \log Z $ | **-0.74 nats**（严重低估） |

- **原因**：当 $ \theta > 0 $ 时，$ \tanh(z) $ 饱和，导致潜变量后验出现“平坦肩部”，而 Laplace 仍为二次下降
- **局部导数无法捕捉**：MAP 附近看似高斯，远端结构不可见
- **Student-t 无效**：因为失败源于非局部几何，而非简单重尾

> 🔍 **亮点**：提出的 **IS/ESS diagnostic** 成功定位失败区域 —— 在 $ \theta > 0 $ 时 ESS/K << 1，而在 $ \theta < 0 $ 时 ESS/K ≈ 1，实现事后故障定位。

---

### ✅ Inference Gym 六大模型综合表现

| Model | $ d_\theta $ | $ d_z $ | $ \Delta \log Z $ | ESS/K | 说明 |
|-------|---------------|----------|--------------------|--------|------|
| Eight Schools | 2 | 8 | +0.08±0.04 | 1.00 | 精确高斯，ALCS 正确 |
| Radon | 4 | 85 | 0.00±0.16 | 1.00 | 同上 |
| Brownian Motion | 1 | 50 | +0.06±0.07 | 1.00 | Kalman 参考，一致 |
| LGCP | 2 | 100 | +0.12±0.07 | 0.71 | Poisson 接近高斯，轻微偏误 |
| Stochastic Volatility | 3 | 100 | -0.32±0.18 | 0.67 | AR(1) 结构，tridiagonal Hessian 加速 |
| IRT | 1 | 500 | — | 0.10 | 积累性小误差，但 marginal posterior 无偏 |

> 💡 尽管 IRT 中 ESS/K 很低（0.10），但恢复的 $ \mu_{\text{ability}} $ posterior 与 Stan MCMC 一致，表明 **marginal posterior 可靠，即使整体近似有系统偏差**

---

## 4. 关键结论和发现

### 主要发现
1. **ALCS 在潜变量近似高斯时极为高效准确**  
   - 可扩展至 $ d_z = 25,600 $，误差 < 0.06 nats
   - wall time 主要由 $ D_{KL}(\theta) $ 决定，而非算法本身开销

2. **自动微分使 Laplace 边缘化真正“即插即用”**  
   - 无需手工推导梯度/Hessian
   - 支持 JAX 生态下的各类物理前向模型（如 oLux）

3. **框架可扩展至非高斯近似**  
   - 第四阶导数可用于拟合 Student-t，改善重尾情况下的 evidence 估计

4. **提出实用的 post-hoc diagnostic：IS/ESS**  
   - 可在不重新运行 full NS 的前提下，定位 ALCS 失效的 $ \theta $ 区域
   - 为可信度评估提供工具

5. **Structured Hessian 支持大幅提升效率**  
   - 对 tridiagonal/banded Hessian 使用 graph colouring + HVPs，内存从 $ \mathcal{O}(d_z^2) $ 降至 $ \mathcal{O}(d_z) $
   - 实际速度提升可达 **54×**，峰值内存减少 **51×**

---

### 局限性
| 限制 | 说明 |
|------|------|
| **要求潜变量后验单峰且近高斯** | 若 $ p(z|\theta,D) $ 多峰或严重偏斜，MAP 可能陷入局部最优，Hessian 错误 |
| **无法处理非局部结构病态** | 如 tanh funnel 中的平坦肩部，局部导数无法反映全局形状 |
| **Hessian 计算成本高** | 密集 Hessian 成本为 $ \mathcal{O}(d_z^2 \cdot T_c) $，在 $ d_z > 10^4 $ 时可能成为瓶颈 |
| **log-determinant 计算障碍** | 场级推断（$ d_z \sim 10^6 $）需要随机迹估计等进一步近似 |

---

### 未来工作方向
1. **集成 stochastic HVP 和 Hutchinson trace estimator**  
   以支持场级 latent space（如宇宙学初始条件重建）

2. **主动 refinement 策略**  
   当 IS/ESS diagnostic 检测到低质量区域时，触发局部 MCMC refinement

3. **更多局部近似族探索**  
   如 skew-normal、gamma-family，匹配三阶及以上导数

4. **与 NS-SWiG 等方法结合**  
   ALCS 侧重维度压缩，NS-SWiG 侧重 likelihood factorization，二者互补

5. **部署于真实天文 pipeline**  
   如 Pantheon+、LSST 弱引力透镜分析，替代静态协方差矩阵假设

---

> 📌 **总结一句话**：  
> **ALCS 将 automatic differentiation 与 Laplace marginalisation 相结合，实现了在高维 latent space 下快速、准确、自动化的 Bayesian evidence 计算，是连接现代可微建模与 robust model comparison 的关键桥梁。**

</details>

---

### 10. [PEANUT: Perturbations by Eigenvalue Alignment for Attacking GNNs Under Topology-Driven Message Passing](https://arxiv.org/abs/2603.26136)

**Authors**: Bhavya Kohli, Biplab Sikdar  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.26136v1  

#### Abstract
Graph Neural Networks (GNNs) have achieved remarkable performance on tasks involving relational data. However, small perturbations to the graph structure can significantly alter GNN outputs, raising concerns about their robustness in real-world deployments. In this work, we explore the core vulnerab...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PEANUT: Perturbations by Eigenvector Alignment for Attacking GNNs Under Topology-Driven Message Passing

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **Graph Neural Networks (GNNs)** 在现实部署中面临的**对抗鲁棒性问题**，特别是其对图结构扰动的敏感性。现有研究多集中在 **graph modification attacks**（修改原始边或节点特征），但在许多真实场景（如社交网络、推荐系统）中，直接修改已有结构是不现实的。

此外，多数现有 **graph injection attacks** 存在以下问题：
- 需要昂贵的迭代优化或强化学习；
- 依赖训练 surrogate model（代理模型），存在泛化差异；
- 多数仅适用于 node-level 任务，缺乏在 graph-level 任务（尤其是 graph regression）上的验证。

### 🚀 提出的新方法：PEANUT
作者提出了一种名为 **PEANUT**（Perturbation by Eigenvector Alignment for Attacking GNNs under Topology-driven message passing）的新型攻击方法，其核心思想是：
> 利用 GNN 架构显式使用邻接矩阵 $ A $ 或拉普拉斯矩阵 $ L $ 进行消息传递这一特性，通过注入虚拟节点并精心设计连接权重，来最大化干净图与扰动图之间节点表示的差异。

#### 核心机制
- **Eigenvector Alignment**：基于干净图的最终节点表示 $ Z $，计算 $ ZZ^T $ 的主特征向量 $ u_1 $，并将该方向用于构造虚拟节点的连接模式。
- **Gradient-free & Surrogate-free**：无需梯度信息或训练替代模型，仅需观察推理阶段的节点级输出（如嵌入或 logits）。
- **Injection-based Evasion Attack**：属于 evasion 攻击，在测试时注入虚拟节点，不修改原图结构，更贴近实际威胁模型。

### ⭐ 相比现有方法的优势
| 特性 | PEANUT | 现有方法（如 TDGIA, AGIA, G2A2C） |
|------|--------|-------------------------------|
| 是否需要梯度 | ❌ 否 | ✅ 是（部分） |
| 是否需训练 surrogate model | ❌ 否 | ✅ 是 |
| 是否支持 graph-level regression | ✅ 是 | ❌ 极少涉及 |
| 是否为 injection-based | ✅ 是 | ✅ 多数是 |
| 计算开销 | ⬇️ 极低（无迭代优化） | ⬆️ 高（RL / 优化循环） |
| 实际可部署性 | ✅ 强（即插即用） | ❌ 弱（延迟高） |

> ✅ **首次系统地将 injection attack 应用于 graph regression 任务，并证明其有效性。**

---

## 2. 核心实验方法和设置

### 📚 数据集

#### （1）Node Classification (NC)
- **Cora**, **Citeseer**, **Pubmed**：标准引文网络数据集，用于评估节点分类任务下的攻击效果。

#### （2）Graph Classification (GC) 和 Graph Regression (GR)
- **Graph Regression (GR)**：
  - FreeSolv, ESOL, Lipophilicity, ZINC, AQSOL — 分子性质预测任务（如溶解度、亲脂性）
- **Graph Classification (GC)**：
  - MUTAG, PROTEINS, ENZYMES, IMDB-BINARY — 来自 TUDataSet
  - BBBP, BACE — 来自 MoleculeNet

> 表格汇总见原文 Table 1 和 Table 3。

### 🧪 实验设置

| 设置项 | 描述 |
|-------|------|
| **攻击预算** | 定义为 $ r \in \{0.001, 0.01, 0.05, 0.1\} $<br>- 对于 NC：$ n_v = r|V| $, $ \Delta = r|V|\cdot \text{deg}(G) $<br>- 对于 GC/GR：$ \Delta = r|\mathcal{E}| $，每图独立设定 |
| **防御模型** | - NC：SGC, GCN, GIN, SAGE<br>- GC/GR：以 2-layer GIN 为主，辅以 GCN 验证趋势一致性 |
| **攻击类型** | 黑盒 evasion attack，仅访问最终节点表示 $ Z $，不可见参数、梯度或训练数据 |
| **注入策略** | 注入 $ n_v $ 个虚拟节点，连接由 $ S_o = \Delta \cdot u_1 v^T $ 控制（经 ReLU 截断保证非负） |

### 📊 评估指标

| 任务类型 | 主要指标 |
|---------|----------|
| **Node Classification** | Accuracy (%)、Macro-F1 Score (%) |
| **Graph Classification** | Accuracy (%)、Macro-F1 Score (%) |
| **Graph Regression** | RMSE、MAE（越低越好） |

所有结果均为 **10次运行平均值**，部分含标准差。

### 🔁 基线方法对比
由于 graph injection 在 GC/GR 上研究较少，主要在 NC 上与以下基线比较：
- **TDGIA**：基于拓扑缺陷选择 + 平滑对抗优化
- **AGIA**：引入同质性约束，提升隐蔽性
- **ATDGIA**：TDGIA 的改进变体

> 所有基线均在其原始设定下复现，并统一应用相同的攻击预算以便公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）Node Classification 结果（Table 2 & Table 4）

| Dataset | Method | $ r=0.05 $ Acc↓ | $ r=0.1 $ Acc↓ |
|--------|--------|------------------|----------------|
| Cora   | AGIA   | -8.96%           | -12.44%        |
|        | TDGIA  | -9.81%           | -14.52%        |
|        | **PEANUT (Ours)** | **-36.12%**      | **-51.65%**    |
| Citeseer | PEANUT | -15.16%          | -37.64%        |
| Pubmed | PEANUT | -34.24%          | -41.78%        |

> 💥 在 $ r=0.001 $ 极小预算下，PEANUT 仍能造成显著下降（如 Cora 下降超 52% Accuracy）。

#### （2）Graph Regression 结果（Figure 2, 4, 6）

- 图中显示随着虚拟节点数量增加，RMSE 和 MAE 显著上升。
- 即使在简单 GCN 模型上，也能有效破坏回归性能（Figure 6）。
- **PEA-D**（离散化版本）表现优于随机连接（Rand-D），说明特征向量对齐策略有效。

#### （3）Graph Classification 结果（Figure 3, 8）

- 多数数据集上准确率随注入节点增多而下降（如 PROTEINS, BBBP）。
- 少数数据集（如 IMDB-BINARY）未持续恶化，归因于 PEA 最大化的是表示范数差异而非类别置信度，可能偶然增强正确类得分。

---

### 🔍 消融实验与变体分析

| 变体 | 描述 | 发现 |
|------|------|------|
| **PEA-W**（White-box） | 已知模型参数 $ H = XO $，理论最优解 | 在 Figure 5 中接近理想性能 |
| **PEA**（Black-box） | 使用 $ Z $ 替代 $ H $，近似实现 | 性能略低于 PEA-W，但仍远超随机扰动 |
| **Rand / Rand-D** | 随机生成连接 | 性能几乎无影响，凸显 eigenvector alignment 的重要性 |
| **PEA-D** | 离散化处理（top-k 边） | 效果弱于连续版，但仍优于随机，适用于仅接受二值邻接的 GNN（如 GIN） |

> ✅ 实验表明：即使没有特征输入（全零）、无需梯度、无需 surrogate model，仅靠结构扰动即可严重削弱 GNN 性能。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **GNN 的拓扑依赖性是根本弱点**  
   显式使用 $ A $、$ S $ 或 $ L $ 作为消息传递算子的 GNN 架构极易受到基于结构的扰动攻击，即使只注入零特征节点。

2. **Eigenvector Alignment 是高效扰动策略**  
   利用 $ ZZ^T $ 的主特征向量指导连接设计，可在黑盒条件下逼近白盒攻击效果。

3. **Graph Regression 更易受攻击**  
   回归任务对表示空间的小扰动极为敏感，即使是轻微的 embedding 偏移也会导致预测值大幅偏离。

4. **无需复杂学习机制也可实现强攻击**  
   PEANUT 不依赖 RL、梯度或 surrogate model，却能达到甚至超越复杂方法的效果，挑战了“必须用复杂策略才能成功攻击”的共识。

---

### ⚠️ 方法的局限性

1. **对某些架构适应性有限**  
   - 若 GNN 内部对邻接矩阵强制非负处理（如 ReLU），需额外调整符号方向（文中已处理）；
   - 对不直接使用 $ A $ 的聚合方式（如 mean/max pooling in SAGE）效果较弱（Table 4 中 SAGE 下降最小）。

2. **离散化后性能下降**  
   PEA-D 虽可用于 binary adjacency 场景，但受限于连接数，攻击强度不如连续版本。

3. **未显式优化攻击目标类**  
   攻击目标是最大化表示差异，而非诱导特定误分类，因此在某些情况下可能导致性能“反弹”（如 IMDB-BINARY）。

---

### 🔮 未来工作方向（原文提及）

1. **改进离散化策略**  
   设计更高效的 top-k 边选择机制，提升 PEA-D 在实际系统中的实用性。

2. **扩展至更复杂的 GNN 架构**  
   探索在带有 attention、jumping knowledge 或 hierarchical pooling 的现代 GNN 上的应用。

3. **结合隐蔽性优化**  
   当前未考虑检测问题，未来可融合 homophily 或 smoothness 约束，提高攻击不可察觉性。

4. **防御机制探索**  
   基于本攻击揭示的漏洞，设计新的鲁棒训练方法或结构正则化技术。

---

## ✅ 总结一句话

> **PEANUT 提出了一种极简但极其有效的黑盒图注入攻击方法，仅通过特征向量对齐注入零特征虚拟节点，即可在无需梯度、无需代理模型的情况下，显著破坏包括 graph regression 在内的多种 GNN 任务性能，揭示了当前 GNN 架构在拓扑利用上的深层脆弱性。**

</details>

---

### 11. [PruneFuse: Efficient Data Selection via Weight Pruning and Network Fusion](https://arxiv.org/abs/2603.26138)

**Authors**: Humaira Kousar, Hasnain Irshad Bhatti, Jaekyun Moon  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.26138v1  

#### Abstract
Efficient data selection is crucial for enhancing the training efficiency of deep neural networks and minimizing annotation requirements. Traditional methods often face high computational costs, limiting their scalability and practical use. We introduce PruneFuse, a novel strategy that leverages pru...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PruneFuse: Efficient Data Selection via Weight Pruning and Network Fusion

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 **Active Learning (AL)** 在进行数据选择时，通常需要在大规模未标注数据池上反复训练或推理一个大型模型，导致极高的计算开销，严重限制了其在资源受限环境下的可扩展性和实用性。

### 提出了什么新方法或新思路
本文提出了一种名为 **PruneFuse** 的新型高效数据选择策略，其核心思想是：
1. **利用结构化剪枝（Structured Pruning）** 在初始化阶段从目标网络中生成一个小规模的 **pruned network**，并用它作为“数据选择器”来挑选最具信息量的样本。
2. 将训练好的小规模剪枝网络通过 **weight-aligned fusion** 技术无缝融合回原始未训练的目标网络，形成一个“热启动”的融合模型（fused model），从而加速最终模型的收敛。

该方法将 **model pruning** 和 **network fusion** 创新性地结合到 AL 流程中，实现了“一次训练，双重收益”。

### 相比现有方法的优势
- **显著降低计算成本**：使用小型剪枝网络代替大模型进行数据选择，大幅减少 FLOPs 和训练时间。
- **保持甚至提升性能**：相比直接在全量数据上训练的 AL 基线，PruneFuse 在多个数据集上达到更高或相当的准确率。
- **结构对齐优势**：与使用独立小代理模型（如 SVP）的方法不同，PruneFuse 的剪枝网络与原网络具有 **channel-aligned 结构**，使得权重可以直接映射融合，无需复杂的蒸馏过程。
- **加速训练收敛**：融合后的模型因继承了剪枝网络的知识，表现出更快的收敛速度和更好的泛化能力。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖了图像、文本和分布外（OOD）场景，验证了方法的广泛适用性：
- **图像分类**：`CIFAR-10`, `CIFAR-100`, `Tiny-ImageNet-200`, `ImageNet-1K`
- **文本分类**：`Amazon Review Polarity`, `Amazon Review Full`
- **OOD 基准**：`CIFAR-10-C`

### 实验设置和评估指标
- **模型架构**：`ResNet-50/56/110/164`, `Wide-ResNet`, `VDCNN`, `Vision Transformer (ViT)`
- **剪枝比例**：`p=0.5, 0.6, 0.7, 0.8`（即保留 50%-80% 的通道）
- **标签预算（labeling budget b）**：从 10% 到 50%，逐步增加标注数据
- **评估指标**：
  - 主要：**Top-1 Test Accuracy (%)**
  - 效率：**FLOPs**（用于数据选择阶段）、**训练时间（分钟）**

### 基线方法对比
- **Baseline AL**：标准主动学习流程，每次迭代使用完整模型选择数据。
- **SVP (Selection via Proxy)**：使用独立的小型代理模型（如 ResNet-20）进行选择。
- **BALD**：基于贝叶斯不确定性采样。
- **ALSE (Snapshot Ensembles)**：使用多个快照集成进行选择。
- **Coreset Selection**：基于遗忘事件（Forgetting Events）、Moderate、CSS 等先进指标。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 表格 2 摘要（以 CIFAR-10 和 ImageNet-1K 为例）

| 方法 | 数据集 | 标签预算 | 准确率 (%) | 计算成本 (×10¹⁶ FLOPs) |
|------|--------|----------|------------|------------------------|
| Baseline AL | CIFAR-10 | 50% | 93.00 ± 0.11 | 14.66 |
| PruneFuse (p=0.7) | CIFAR-10 | 50% | **93.40 ± 0.11** | **1.32** |
| Baseline AL | ImageNet-1K | 50% | 73.56 ± 0.16 | 33.53 |
| PruneFuse (p=0.8) | ImageNet-1K | 50% | **73.64 ± 0.52** | **0.58** |

> ✅ **观察**：PruneFuse 在 **ImageNet-1K 上节省了超过 95% 的计算成本**，同时达到了略高的准确率。

### 与基线方法的对比结果
- **vs SVP**：
  - 在相同标签预算下，PruneFuse 的 **target model 准确率更高**（如 93.69% vs 92.95% @ b=50%）。
  - PruneFuse 的 **data selector 参数更少**（0.21M vs 0.26M），且无需额外设计代理架构。
- **vs BALD & ALSE**：
  - PruneFuse 可与这些方法结合（如 PruneFuse + BALD），进一步提升性能。
  - 单独使用时，PruneFuse 在低预算下表现更优，且计算效率远超 BALD。
- **跨架构一致性**：
  - 在 `ResNet`, `Wide-ResNet`, `ViT`, `MobileNet`, `VDCNN` 上均一致优于基线。

### 消融实验结果
#### 融合（Fusion）的作用（图 4 & 表 18）
- **Fusion 是关键**：即使不使用知识蒸馏（KD），仅靠 fusion 就能显著加速收敛并提升最终性能。
- **初始化方式影响**：
  - 随机重初始化 > 保留初始权重 > 零初始化
  - 最佳组合为 **fusion + random re-initialization**。

#### 同步间隔 $T_{\text{sync}}$ 的影响（图 3）
- 设置 $T_{\text{sync}}=1$（每轮更新剪枝网络）能获得最佳精度-效率权衡。
- 即使 $T_{\text{sync}}=0$（固定剪枝网络），仍能大幅节省计算。

#### 剪枝策略的影响（表 7 & 表 28）
- **静态剪枝（Static Pruning）** 效果优于动态剪枝。
- 不同重要性评分准则（Magnitude, GroupNorm, LAMP）下性能稳定，表明方法鲁棒。

#### 知识蒸馏（KD）的作用（表 19）
- KD 提供额外增益，但非必需；**fusion 本身已带来主要收益**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **剪枝网络可有效充当高质量数据选择器**：尽管规模小，但由于结构对齐，其选择行为与原网络高度一致。
2. **网络融合是一种高效的 warm-start 机制**：将剪枝网络的训练成果直接注入原网络，显著提升训练效率和最终性能。
3. **PruneFuse 实现了“双赢”**：既降低了数据选择的计算负担，又提升了最终模型的表现。
4. **方法通用性强**：适用于多种网络架构、数据类型和选择策略（LC, Entropy, Greedy k-centers 等）。

### 方法的局限性
- 当前依赖于 **structured pruning**，对 unstructured pruning 支持有限。
- 极端高剪枝率（如 p=0.9）可能导致选择质量下降（见补充材料）。
- 融合过程假设层间结构完全对齐，在某些复杂架构中可能需调整。

### 未来工作方向
- 探索 **自动化剪枝率调度**（adaptive pruning ratio）。
- 将 PruneFuse 扩展至 **自监督学习** 和 **预训练语言模型** 的数据筛选任务。
- 研究 **非结构化剪枝 + 映射融合** 的可行性。
- 结合 **early stopping** 进一步压缩总训练时间（已在补充材料中初步验证有效）。

---

> 📌 **总结一句话**：  
> **PruneFuse 通过“剪枝选数 + 融合提速”的两阶段范式，为高效数据选择提供了一个兼具高性能、低开销和强通用性的新解决方案。**

</details>

---

### 12. [Knowledge Distillation for Efficient Transformer-Based Reinforcement Learning in Hardware-Constrained Energy Management Systems](https://arxiv.org/abs/2603.26249)

**Authors**: Pascal Henrich, Jonas Sievers, Maximilian Beichter, Thomas Blank, Ralf Mikut, Veit Hagenmeyer  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.26249v1  

#### Abstract
Transformer-based reinforcement learning has emerged as a strong candidate for sequential control in residential energy management. In particular, the Decision Transformer can learn effective battery dispatch policies from historical data, thereby increasing photovoltaic self-consumption and reducin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Knowledge Distillation for Efficient Transformer-Based Reinforcement Learning in Hardware-Constrained Energy Management Systems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本论文针对**住宅级能源管理系统（HEMS）中基于 Transformer 的强化学习模型难以部署在资源受限硬件上的问题**。尽管 Decision Transformer（DT）在电池调度等序列控制任务中表现出色，但其高参数量、内存占用和推理延迟使其不适用于嵌入式控制器。

### 提出了什么新方法或新思路
提出了一种**基于响应的知识蒸馏（Response-Based Knowledge Distillation, KD）框架**，用于压缩大型 DT 教师模型为小型学生模型：
- 将高容量 DT 作为教师模型，在离线多建筑数据上训练；
- 设计紧凑的学生模型，通过匹配教师的动作输出（logits）进行行为模仿；
- 探索了**自蒸馏（self-distillation）** 作为一种正则化手段，提升泛化能力。

### 相比现有方法的优势
- **显著降低计算开销**：实现高达 96% 的参数减少、90% 的内存节省和 63% 的推理时间缩短；
- **保持甚至提升控制性能**：蒸馏后的学生模型在多数场景下表现优于原始教师模型，最高提升达 1%；
- **无需在线交互**：完全基于离线数据训练，适合现实世界中无法频繁试错的应用环境；
- **通用性强**：适用于连续动作空间的回归型控制任务，突破传统分类任务中温度软化蒸馏的限制。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Ausgrid Solar Home Electricity Dataset**：包含 20 个住宅建筑的高分辨率用电负荷与光伏发电数据；
- **SMARD 数据库中的 DE/AT/LU 批发电价数据**，用于构建经济激励信号；
- 数据覆盖三年，选取典型住宅负载模式（如图5所示）。

### 实验设置和评估指标
#### 模型架构
采用不同规模的 DT 架构（Tiny 到 Large），具体配置见表1：

| 名称 | 层数 | 注意力头数 | 模型维度 |
|------|------|------------|----------|
| Tiny | 1    | 1          | 64       |
| Mini | 2    | 2          | 128      |
| Small| 3    | 1          | 128      |
| Medium|8    | 2          | 256      |
| Large|12   | 4          | 512      |

#### 蒸馏策略
- **教师模型**：Small、Medium、Large 规模的 DT；
- **学生模型**：Tiny、Mini、Small；
- **损失函数**：Smooth L1 Loss，直接回归连续动作值；
- **训练方式**：离线蒸馏，固定教师模型权重。

#### 评估指标
- **平均电费成本（Mean Electricity Cost）**：主要性能指标；
- **模型大小（Parameters）**；
- **推理内存占用（Memory Footprint）**；
- **推理延迟（Inference Time）**；
- **跨建筑泛化能力**：在 20 个独立建筑上的表现分布。

### 基线方法对比
| 方法 | 类型 | 描述 |
|------|------|------|
| **Without Battery (WO Battery)** | 上界 | 不使用储能系统，仅依赖电网购电 |
| **Rule-based Controller** | 上界 | 固定规则控制（未详细建模） |
| **DDPG** | 强化学习基线 | 在线 RL 方法，用作轨迹生成器 |
| **MILP (Mixed Integer Linear Programming)** | 下界（理论最优） | 全局最优解，假设有完美预测能力，不可实时实现 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 总体成本比较（图6）
| 方法 | 平均成本（€） |
|------|----------------|
| MILP（理论最优） | 168.08 |
| **KD（Medium → Small）** | **200.28** ✅ 最佳学习方法 |
| DT（Medium） | 201.30 |
| DDPG | 202.43 |
| 无电池（WO Battery） | 220.25 |

> **KD 模型超越了原始 DT 和 DDPG，接近理论极限**

#### 不同 DT 模型的成本表现（图7）
- **非单调关系**：更大模型不一定更好；
- **Medium DT 表现最佳**（201.30 €），Large 反而更差（203.88 €），表明存在过拟合风险；
- Medium 成为后续蒸馏的理想教师。

#### 蒸馏效果（图8）
| 教师 → 学生 | 成本下降（€） |
|-----------|-------------|
| Small → Small（自蒸馏） | +0.57 |
| Medium → Small | **+1.02** ✅ 最大增益 |
| Medium → Tiny | +0.60 |
| Large → Mini | **+1.10** |

> 多数蒸馏配置均带来性能提升，说明 KD 具有**正则化效应**

#### 模型压缩效率（表2）
以 Medium → Tiny 蒸馏为例：
| 指标 | 压缩比例 |
|------|---------|
| 参数数量 | ↓ **96%** |
| 推理内存 | ↓ **90%** |
| 推理时间 | ↓ **63%** |

即使极端压缩后，Tiny 学生仍在 **55% 的建筑中取得最低成本**

#### 建筑级性能分布（图9 & 图10）
- KD（Medium → Small）在 **60% 的建筑中优于 DDPG 和 DT**；
- KD（Medium → Tiny）在 **55% 的建筑中仍为最优**，验证其鲁棒性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **知识蒸馏不仅是压缩工具，更是性能增强机制**：
   - 蒸馏后的学生模型普遍优于教师，尤其在 Medium → Small 配置中；
   - 自蒸馏也有效果，说明 KD 起到平滑决策边界、抑制噪声的作用。

2. **模型容量并非越大越好**：
   - Large DT 表现劣于 Medium，可能因过拟合特定建筑特征；
   - Medium 模型在表达能力和泛化之间达到平衡。

3. **KD 显著促进 Transformer 在边缘设备的落地**：
   - 实现两个数量级的模型压缩，同时维持甚至提升控制质量；
   - 使 DT 控制器可在低功耗嵌入式 HEMS 中运行成为可能。

4. **DT 能克服“致命三元组”问题**：
   - 相比传统 off-policy RL（如 DDPG），DT 以序列建模方式避免 bootstrapping 带来的误差累积；
   - 结合 KD 后进一步提升了稳定性和实用性。

### 方法的局限性
- **依赖高质量离线数据集**：若 DDPG 生成的轨迹本身次优，则教师模型上限受限；
- **未考虑动态环境变化**：所有测试基于历史数据回放，缺乏对季节漂移或设备老化的适应性分析；
- **仅聚焦单一灵活资源（BESS）**：未扩展至 HVAC、EV 等多设备协同控制；
- **未探索其他蒸馏目标**：如中间层表示蒸馏、价值函数对齐等。

### 未来工作方向
1. **探索更多蒸馏目标函数**：结合隐状态匹配、注意力图对齐等方式；
2. **研究轨迹质量对 KD 的影响**：如何从低质量数据中提取可靠知识；
3. **拓展至多设备联合控制场景**：将 KD 应用于更复杂的 residential energy system；
4. **在线微调机制设计**：允许轻量学生模型在部署后持续适应本地环境；
5. **硬件实测验证**：在真实嵌入式平台（如 Raspberry Pi 或 STM32）上部署并测量实际能耗与延迟。

---

> **总结一句话**：  
> 本文证明了 **Knowledge Distillation 是连接高性能 Decision Transformer 与资源受限 HEMS 硬件之间的关键桥梁**——不仅实现了极致压缩，还带来了意外的性能增益，推动了智能能源控制向边缘侧的大规模落地。

</details>

---

### 13. [SPECTRA: An Efficient Spectral-Informed Neural Network for Sensor-Based Activity Recognition](https://arxiv.org/abs/2603.26482)

**Authors**: Deepika Gurung, Lala Shakti Swarup Ray, Mengxi Liu, Bo Zhou, Paul Lukowicz  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.26482v1  

#### Abstract
Real time sensor based applications in pervasive computing require edge deployable models to ensure low latency privacy and efficient interaction. A prime example is sensor based human activity recognition where models must balance accuracy with stringent resource constraints. Yet many deep learning...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SPECTRA: An Efficient Spectral-Informed Neural Network for Sensor-Based Activity Recognition》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于传感器的人类活动识别（HAR）模型面临以下挑战：
- 多数深度学习模型将IMU信号视为“黑盒”时间序列，忽略了**频谱-时序结构**（spectral-temporal structure），导致模型需要更深更复杂的结构来捕捉周期性和节奏特征。
- 主流模型（如CNN、LSTM、Transformer）计算量大、参数多，难以部署在资源受限的边缘设备（如智能手机、MCU）上。
- 云端推理存在**隐私泄露、延迟高、依赖网络连接**等问题。

### 🚀 提出的新方法与创新思路
作者提出 **SPECTRA** —— 一种面向边缘部署的、**共设计**（co-designed）的轻量级神经网络架构，其核心思想是：
> **将经典信号处理（STFT）与轻量神经模块结合，显式注入频谱先验知识，降低下游模型的学习负担。**

具体创新点包括：
- **STFT前端特征提取**：使用短时傅里叶变换（STFT）生成时频图（spectrogram），显式暴露运动中的节律性模式（如步态频率、谐波结构）。
- **Depthwise Separable Convolution**：在时频平面上进行局部模式提取，显著减少参数和MACs。
- **Channel-wise Self-Attention**：建模不同IMU轴之间的耦合关系（如加速度计与陀螺仪间的相关性），增强跨通道信息融合能力。
- **Compact Bi-GRU + Attention Pooling**：用轻量双向GRU建模帧间动态，并通过注意力池化选择关键时间片段，避免使用高成本的Transformer或深层LSTM。

### 🔍 相比现有方法的优势
| 维度 | SPECTRA优势 |
|------|-------------|
| **准确性** | 在多个数据集上接近甚至超过大型CNN/LSTM/Transformer模型 |
| **效率** | 参数减少97–99%，MACs减少85–99%，内存占用最小 |
| **部署性** | 成功部署于Google Pixel 9（ExecuTorch）和STM32L4 MCU（TFLM），支持实时、低功耗、端到端推理 |
| **数据效率** | 在仅25%训练数据下仍表现领先，显示强泛化能力 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在五个公开IMU-based HAR数据集上进行全面评估：
- **WISDM**  
- **USC-HAD**  
- **UCI HAR**  
- **DSADS**  
- **PAMAP2**

所有数据统一预处理为：
- 采样率：50 Hz
- 窗口长度：T = 2秒，重叠50%
- 输入维度：`(B, 1, T, C)`，其中 `C=6`（3轴加速度 + 3轴角速度）

### ⚙️ 实验设置
- **精度变体测试**：SPECTRA在三种精度下运行：
  - FP32（浮点32）
  - FP16（半精度）
  - INT8（整型量化）
- **部署平台**：
  - **PC工作站**：NVIDIA RTX A6000，用于基准训练与仿真
  - **智能手机**：Google Pixel 9（Tensor G4，ExecuTorch + NNAPI）
  - **微控制器**：STM32L4S5VIT6（ARM Cortex-M4，ONNX → TFLite → TFLM + CMSIS-NN）
- **评估指标**：
  - 准确性：Macro F1-score、Accuracy
  - 效率：Params (K)、MACs
  - 部署性能：Latency (ms)、Throughput (samples/s)、Peak Memory (MB/KB)、Energy (J or mJ/uJ)

### 🆚 基线方法对比
| 模型 | 类型 | 特点 |
|------|------|------|
| **DeepConvLSTM** | Accuracy-focused | CNN + LSTM混合结构，准确但庞大 |
| **TinyHAR** | Efficiency-focused | 轻量CNN，专为MCU优化 |
| **TinierHAR** | Ultra-lightweight | 更小版本，部分操作不支持TFLM |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（以FP32为例）

| 指标 | 表现 |
|------|------|
| **参数量** | 仅8.9K–22.8K（相比DeepConvLSTM减少 >97%） |
| **MACs** | 28.5K–481.6K（减少 >99% vs DeepConvLSTM；比TinyHAR低85–95%） |
| **PC端延迟** | 1.48–3.21 ms（含STFT），吞吐达353–671 samples/s |
| **Pixel 9延迟** | 1.7–3.4 ms，最高达840 samples/s（INT8） |
| **STM32延迟** | 1.07–9.33 ms（FP32），INT8进一步降至0.8–7.0 ms |
| **能量消耗** | 最低至 **0.118 μJ/sample**（STM32上） |
| **内存占用** | 峰值RAM仅 **3.29 KB**（STM32），Flash最低 **2.79 KB**（INT8） |

> ✅ 所有平台上，SPECTRA均实现**实时推理**（<10ms/window），且远优于基线。

### 📈 与基线方法对比结果

#### 在PC上的综合表现（Table I）：
- **F1-score**：在USC-HAD和DSADS上**排名第一**，其余接近最优。
- **能效比**：Energy per sample 比DeepConvLSTM低 **96–99%**，比TinyHAR低 **70–90%**。
- **内存**：峰值内存仅为DeepConvLSTM的 **1/3 到 1/4**。

#### 在Pixel 9上的表现（Table II）：
- **延迟**：比TinyHAR快 **3.1–15.3倍**，比DeepConvLSTM快 **6.7–43.8倍**
- **能耗**：每样本能耗 **低72–92% vs TinyHAR**
- **内存**：峰值内存约 **27–121 KB**，约为TinyHAR的一半

#### 在STM32上的表现（Table III）：
- **成功部署**：唯一能在所有数据集上运行的模型（DeepConvLSTM因Flash不足OOM）
- **速度**：比TinyHAR快 **2.6–5.4倍**
- **能耗**：每样本能耗 **低 >99.9% vs TinyHAR**
- **存储**：Flash占用减少 **64–93%**，SRAM恒定为 **3.29 KB**

### 🔬 消融实验结果（Table IV）

| 变体 | F1变化 | 参数/MACs变化 | 结论 |
|------|--------|----------------|------|
| **w/o STFT** | ↓3.7–9.6 pts | ↑12–15× MACs | STFT对准确性和效率至关重要 |
| **w/ FFT instead of STFT** | ↓1.2–4.4 pts | 参数略降 | 缺乏时间定位能力，性能下降明显 |
| **w/ CWT** | 接近或略优 | ↑10×以上MACs | 不适合边缘部署 |
| **w/ MFCC** | ↓5–10 pts | 参数更低 | 丢失过多频谱细节 |
| **w/o Separable Conv** | ↓1.2–2.6 pts | ↓MACs但F1损失更大 | 分离卷积性价比极高 |
| **w/ Standard Conv** | F1略升 | ↑2–4×参数/MACs | 成本过高，得不偿失 |
| **w/o Channel Attention** | ↓0.1–3.0 pts | ↓4–5% MACs | 尤其在多传感器数据集（PAMAP2）影响大 |
| **w/ Temporal Attention** | 略差 | 更轻 | 说明跨轴耦合比时间重加权更重要 |
| **w/o Bi-GRU** | ↓1.7–5.9 pts | ↓56–67% MACs | 显示短期时序建模仍有必要 |
| **w/ Bi-LSTM** | F1持平或略差 | ↑参数和计算 | GRU已足够，无需更复杂结构 |

> ✅ 完整SPECTRA在所有数据集上达到最佳或次佳F1，同时保持最小开销。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **显式引入频谱先验（STFT）可大幅提升数据效率和模型效率**：
   - 避免了深层网络从原始信号中隐式学习频谱结构的过程，极大降低了表示学习难度。
2. **轻量组件组合优于单一重型模块**：
   - Separable Conv + Channel Attention + Bi-GRU 的组合实现了高效的空间、通道、时间建模。
3. **SPECTRA始终位于准确-效率帕累托前沿**：
   - 在五大数据集、三大硬件平台上均优于或匹配各类基线，在精度、延迟、能耗、内存之间取得最优平衡。
4. **具备卓越的数据效率**：
   - 在仅25%训练数据下，SPECTRA在3/5数据集上排名第一，表明其具有更强的泛化能力和更低的样本复杂度。

### ⚠️ 局限性
1. **STFT超参数固定**：
   - `n_fft` 和 `hop` 固定，无法自适应不同活动的时间尺度（如慢速姿势变化 vs 快速手势）。
2. **未考虑域偏移与校准问题**：
   - 对传感器位置漂移、个体差异等现实场景下的鲁棒性尚未验证。
3. **缺乏长期野外实测**：
   - 当前评估集中在标准数据集和实验室环境，缺少真实世界长期使用的反馈。

### 🔮 未来工作方向
1. **自适应或多分辨率前端**：
   - 设计可学习或动态调度的STFT参数，覆盖多种时间尺度。
2. **多模态融合**：
   - 引入其他传感模态（如ALS、PPG）提升相似动作区分能力。
3. **能量感知调度机制**：
   - 动态调整窗口大小或模型深度，基于置信度或上下文优化Energy-Delay Product。
4. **量化感知训练（QAT）与混合精度推理**：
   - 进一步压缩模型并提升嵌入式部署稳定性。

---

## 总结
> **SPECTRA的核心洞见是：将经典信号处理（STFT）作为神经网络的“前置编码器”，可以显著降低后续深度模型的学习难度和计算负担，从而构建出既准确又高效的边缘AI系统。**

该工作不仅提出了一个高性能HAR模型，更重要的是展示了**“信号知情”（signal-informed）架构设计范式**在边缘智能中的巨大潜力，为未来轻量级、可部署、高鲁棒性的普适计算系统提供了重要参考。

</details>

---

### 14. [AutoB2G: A Large Language Model-Driven Agentic Framework For Automated Building-Grid Co-Simulation](https://arxiv.org/abs/2603.26005)

**Authors**: Borui Zhang, Nariman Mahdavi, Subbu Sethuvenkatraman, Shuang Ao, Flora Salim  
**Category**: cs.AI  
**Published**: 2026-03-30  
**Score**: 4.5  
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
当前建筑能源系统仿真环境（如 CityLearn、Sinergym）主要关注**building-side performance metrics**（如能耗、成本），缺乏对电网侧影响的系统性评估。同时，现有仿真流程依赖大量手动配置和编程，技术门槛高，限制了研究者对科学问题的关注。

此外，虽然 LLM 在代码生成方面展现出潜力，但在复杂、多模块、强依赖的仿真任务中，直接使用 LLM 难以保证生成代码的**结构正确性**和**执行可行性**。

### 🚀 提出的新方法与创新思路
本文提出 **AutoB2G** ——一个基于大语言模型（LLM）驱动的**自动化建筑-电网协同仿真框架**，其核心创新包括：

1. **构建支持 B2G（Building-to-Grid）交互的仿真环境**
   - 扩展了最新的 **CityLearn V2**，集成 **Pandapower** 构建电网模型，实现建筑控制与电网动态的双向耦合。
   - 支持电压调节、N-1 resilience、thermal loading 等 grid-side metrics 的量化评估。

2. **提出 DAG-based Agentic Retrieval 机制**
   - 将仿真功能模块组织为有向无环图（DAG），显式编码模块间的依赖关系与执行顺序。
   - 利用 LLM agent 从 DAG 中检索并组合所需模块，确保生成路径符合拓扑约束。
   - 引入验证器（Validator）检测缺失依赖，并通过反馈迭代修复，提升工作流完整性。

3. **采用 SOCIA 多智能体框架进行自动构建与优化**
   - 使用 **SOCIA (Simulation Orchestration for Computational Intelligence with Agents)** 框架，实现模拟器的端到端自动化构造。
   - 多个专业化 agent 分工协作：Code Generator、Simulation Executor、Result Evaluator、Feedback Generator 等。
   - 引入 **Textual-Gradient Descent (TGD)** 反馈机制，将运行错误转化为自然语言修复指令，驱动代码逐步收敛至可执行状态。

### 🔍 相比现有方法的优势
| 维度 | 现有方法（如 GridLearn） | AutoB2G |
|------|--------------------------|--------|
| 配置方式 | 手动编程 | 自然语言描述驱动 |
| 兼容性 | 不兼容 CityLearn V2 新特性 | 基于最新 CityLearn V2 扩展 |
| 电网评估 | 功能有限或缺失 | 支持多种 grid-side metrics（电压、负载、短路电流、N-1 安全性） |
| 自动化程度 | 低，需专家介入 | 高，全流程自动化 |
| 错误处理能力 | 无自修复机制 | 支持 TGD 迭代纠错 |

---

## 2. 核心实验方法和设置

### 📊 数据集与平台
- **建筑侧模拟**：基于 **CityLearn V2** 和 **End-Use Load Profiles for the U.S. Building Stock dataset** 生成定制化住宅负荷数据。
- **电网模型**：采用标准 **IEEE 33-bus distribution network**（来自 [6]），用于测试电压控制与安全性分析。
- **仿真工具链**：
  - 建筑热力学建模：**EnergyPlus + LSTM**
  - 控制策略训练：**RL algorithms (e.g., SAC)**
  - 电网潮流计算：**Pandapower**

### ⚙️ 实验设置
- **任务复杂度分级**：
  - **Simple Tasks**：单阶段任务，如“使用规则控制器运行仿真并报告KPI”。
  - **Medium Tasks**：多阶段任务，如“训练 SAC 模型并在 IEEE 33 节点系统上评估性能”。
  - **Complex Tasks**：复合任务，涉及多策略比较、自定义奖励函数、N-1 分析等。

- **输入形式**：仅提供自然语言任务描述（no code, no config file）。

- **LLM 平台**：使用 OpenAI GPT-5 API（模拟未来版本）作为 backbone。

### 📈 评估指标
1. **Task Success Rate**  
   表示任务是否成功完成（程序可编译、运行、输出预期结果）。

2. **Code Score**  
   衡量生成代码的质量：
   $$
   \text{Code Score} = \frac{N_{\text{correct}}}{N_{\text{total}} + N_{\text{extra}}}
   $$
   - $N_{\text{correct}}$: 正确实现的必需组件数
   - $N_{\text{total}}$: 总必需组件数
   - $N_{\text{extra}}$: 多余或无关组件数

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **LLM** | 单一 LLM，直接访问完整 codebase |
| **LLM + AR** | LLM + Agentic Retrieval（基于 DAG 检索相关模块） |
| **SOCIA** | SOCIA 框架，无 retrieval 辅助 |
| **SOCIA + AR** | 本文完整方法（SOCIA + DAG-based retrieval） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Tables 3 & 4）

#### ✅ Task Success Rate 对比

| Method | Simple | Medium | Complex |
|--------|--------|--------|---------|
| LLM | 0.90 ± 0.08 | 0.77 ± 0.12 | 0.53 ± 0.19 |
| SOCIA | 0.93 ± 0.09 | 0.83 ± 0.12 | 0.73 ± 0.09 |
| LLM + AR | 0.97 ± 0.05 | 0.80 ± 0.08 | 0.67 ± 0.05 |
| **SOCIA + AR** | **1.00 ± 0.00** | **0.93 ± 0.09** | **0.83 ± 0.05** |

> 💡 在所有任务级别中表现最优，尤其在复杂任务中显著领先。

#### ✅ Code Score 对比

| Method | Simple | Medium | Complex |
|--------|--------|--------|---------|
| LLM | 0.69 ± 0.08 | 0.66 ± 0.05 | 0.44 ± 0.08 |
| SOCIA | 0.82 ± 0.14 | 0.78 ± 0.05 | 0.67 ± 0.09 |
| LLM + AR | 0.72 ± 0.08 | 0.74 ± 0.07 | 0.73 ± 0.13 |
| **SOCIA + AR** | **1.00 ± 0.00** | **0.84 ± 0.07** | **0.88 ± 0.09** |

> 💡 显示出更强的模块选择准确性和更少冗余代码，说明 **agentic retrieval + TGD** 有效提升了代码质量。

### 🔍 消融实验分析
- **引入 Agentic Retrieval** 显著减少无关信息干扰，提高模块选取精度。
- **SOCIA 的 TGD 机制** 在第一次迭代后即大幅提升成功率（见 Figure 3），表明其具备强大纠错能力。
- **DAG 结构引导** 有助于避免依赖冲突和执行顺序错误，是保障复杂流程可行性的关键。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **AutoB2G 成功实现了从自然语言到可执行仿真程序的全自动转换**，极大降低了用户的技术门槛。
2. **DAG-based retrieval + SOCIA + TGD 的组合显著优于单一 LLM 或简单 retrieval 方法**，特别是在中高复杂度任务中优势明显。
3. 框架能够协调 building-level 控制策略与 grid-level 安全目标，在 IEEE 33-bus 系统中有效改善电压稳定性（见 Figure 4 & 5）：
   - RL 控制下电压波动范围缩小（baseline: ~±0.4 p.u. → RL: ~±0.05 p.u.）
   - 电压分布更集中于额定值 1.0 p.u.
4. 控制策略能根据电网条件自适应调整负荷行为：
   - 过压时增加用电以抑制电压上升；
   - 欠压时主动降低负荷缓解电压跌落（见 Table 6）。

### ⚠️ 局限性
1. 当前框架主要在 **CityLearn + Pandapower** 生态内验证，尚未扩展至其他 building 或 power system simulators（如 EnergyPlus standalone, OpenDSS）。
2. 自然语言指令可能存在歧义，导致模型误解用户意图，引入不必要的模块或忽略隐含需求。
3. 跨模块高度耦合的任务仍可能出现细微配置错误（如参数单位不一致、接口错位），影响最终执行。

### 🔮 未来工作方向
1. **平台扩展性增强**：集成更多 building 和 power system simulation 工具，提升框架通用性。
2. **支持更丰富的 configuration 接口**：允许用户定义 custom scenario templates、reusable modules 和 evaluation metrics。
3. **改进自然语言理解能力**：结合对话式交互澄清模糊指令，提升任务解析准确性。
4. **探索 LLM 在 real-time control 与 online adaptation 中的应用潜力**。

---

> ✅ **总体评价**：  
> AutoB2G 是首个将 LLM-driven agentic framework 应用于 building-grid co-simulation 的系统性尝试，不仅推动了仿真自动化的发展，也为未来智能城市能源系统的快速原型设计提供了强有力的支持。

</details>

---

### 15. [LLM Benchmark-User Need Misalignment for Climate Change](https://arxiv.org/abs/2603.26106)

**Authors**: Oucheng Liu, Lexing Xie, Jing Jiang  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.26106v1  

#### Abstract
Climate change is a major socio-scientific issue shapes public decision-making and policy discussions. As large language models (LLMs) increasingly serve as an interface for accessing climate knowledge, whether existing benchmarks reflect user needs is critical for evaluating LLM in real-world setti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM Benchmark-User Need Misalignment for Climate Change

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对当前 **LLM 在气候变化领域评估基准（benchmark）与真实用户需求之间存在显著错位（misalignment）** 的问题展开研究。尽管 LLM 越来越多地被公众用于获取气候知识，但现有的 LLM 评估基准大多基于教科书式的科学事实问答，未能反映现实世界中用户多样化的信息需求。

具体而言，现有基准在以下方面存在偏差：
- **Topic（主题）偏差**：过度集中于“大气科学”等基础气候科学，而忽视“气候政策”、“能源转型”等应用型议题。
- **Intent（意图）偏差**：侧重“事实查找”（Fact Lookup），缺乏对“建议咨询”（General Advice）、“操作写作”（Operational Writing）等高阶认知任务的支持。
- **Form（形式）偏差**：答案形式多为简短实体或选择题，而真实用户更期望获得段落解释或列表形式的结构化输出。

### 提出了什么新方法或新思路
论文提出了两个核心创新：

#### （1）Proactive Knowledge Behaviors Framework（主动知识行为框架）
该框架将知识交互分为两类主动行为：
- **知识寻求（Asking）**：人类向人类（Human-to-Human）或人类向 AI（Human-to-AI）提问。
- **知识提供（Guiding/Informing）**：人类指导 AI（如设计 benchmark）或人类向人类提供知识（如撰写报告）。

通过这一框架，作者系统性地比较了不同知识行为之间的分布一致性，从而识别出可迁移的高质量知识源。

#### （2）Topic-Intent-Form 三维分类法（Taxonomy）
构建了一个细粒度的三维标注体系：
- **Topic Taxonomy**：五类 25 个子主题（如 A1. Atmospheric Science, E1. Climate Policy）。
- **Intent Taxonomy**：八类 29 种意图（如 INTENT_1a. Fact Lookup, INTENT_3a. General Advice）。
- **Form Taxonomy**：八类 29 种输出形式（如 FORM_2a. Concise Paragraph, FORM_3a. Item List）。

该分类法结合了扩展版的 **Bloom’s Taxonomy**，将用户意图映射到所需的知识类型（Factual, Conceptual, Procedural, Metacognitive），增强了分析深度。

### 相比现有方法的优势
- **从被动评估转向主动需求建模**：不再仅关注模型能否答对已有问题，而是探究“用户真正需要什么样的知识”。
- **提出可操作的数据来源替代方案**：当直接使用用户查询难以构建高质量 benchmark 时，提出可利用 **Human-to-Human 问答** 和 **IPCC 报告** 作为可靠代理。
- **方法具有可迁移性**：所提出的框架和分类法可推广至其他社会科学研究领域（如公共卫生、可持续发展）。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
共使用 **11 个数据集**，其中 8 个为核心数据集，3 个为辅助语料库：

| 类别 | 数据集 | 描述 | 样本数 |
|------|-------|------|--------|
| **Human-to-AI Queries** | WildChat, LMSYS-Chat-1M, ClimateQ&A | 用户与 LLM 的对话日志，提取首条查询 | 1.7K, 1.3K, 3.0K |
| **Human-to-Human Questions** | Reddit (4 subreddits) | 气候相关论坛中的用户提问 | 1.0K |
| **Human-to-AI Guidance Knowledge** | ClimaQA-Gold, ClimaQA-Silver | 当前主流气候 QA benchmark | 540, 2.5K |
| **Human-to-Human Knowledge Provision** | SciDCC, IPCC AR6 | 新闻报道与权威科学报告 | 7.5K, 18.9K |
| **Auxiliary Corpora** | Climate-FEVER, Environmental Claims, ClimSight | 支持主题建模的辅助数据 | ~1.3K each |

### 实验设置和评估指标
- **主题建模流程**：
  1. **初始生成**：使用 LLM（GPT-4.1-mini）为每条文本生成自由格式的主题标签。
  2. **迭代合并**：基于嵌入相似性 + LLM 判断，合并语义重叠的主题。
  3. **重新分配**：使用最终分类体系对所有样本进行统一标注。
- **意图与形式分类**：使用 LLM（GPT-5-mini）进行多标签分类，按优先级排序。
- **量化分析方式**：
  - 将每个样本表示为 **加权向量**（Topic/Intent/Form 维度分别为 25/38/38）。
  - 使用 **余弦相似度（cosine similarity）** 比较不同数据集间的分布一致性。

### 基线方法对比
- **主要对比对象**：
  - **Real-World Group**：由 WildChat、LMSYS-Chat-1M、ClimateQ&A 合并而成，代表真实用户需求。
  - **LLM Benchmark Group**：由 ClimaQA-Gold/Silver 合并而成，代表当前主流评估标准。
- 对比维度包括：
  - Topic 分布差异
  - Intent 分布差异
  - Form 分布差异
  - 三者联合空间下的整体对齐程度

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）Benchmark 与用户需求严重错位（Misalignment）
| 维度 | Real-World Group | LLM Benchmark Group | 差异 |
|------|------------------|---------------------|------|
| **Topic** | 分布广泛，覆盖政策、适应、减缓等领域 | 高度集中于 A1. Atmospheric Science（占比超 70%） | +63% 偏差 |
| **Intent** | 多样化，含 Advice、Writing 等高阶意图 | 超 60% 为 Fact Lookup | -40% 偏差 |
| **Form** | 偏好段落（FORM_2a/b）和列表（FORM_3a） | 集中于简短值（FORM_1a）、陈述（FORM_1b）和选择题（FORM_7a） | 最大差距达 37% |

> 即使在相同主题（如 A1）下，Intent 和 Form 的差异依然显著（见 Figure 6），说明错位是系统性的，而非单纯主题分布不均所致。

#### （2）Human-to-AI 与 Human-to-Human 需求高度一致（Similarity）
- **Topic 分布相似性**：Human-to-AI 查询与 Reddit 问题之间的余弦相似度高达 **0.85–0.98**（Figure 7a），表明用户在向人或 AI 提问时关注的议题基本一致。
- **Intent 和 Form 分布也高度相似**：尤其在 FORM_2a（Concise Paragraph）和 FORM_3a（Item List）上偏好一致（Figure 9）。

#### （3）IPCC AR6 是最匹配的高质量知识源
- **Topic 对齐度最高**：IPCC AR6 与各 Human-to-AI/Human-to-Human 数据集的 Topic 相似度为 **0.84–0.88**（Figure 10），远高于新闻数据 SciDCC（~0.55）。
- **揭示潜在缺口**：尽管 IPCC 总体匹配度高，但在 D4. Public Awareness, Communication & Community Engagement 上关注度不足（Figure 11），提示需补充外部资源。

### 与基线方法的对比结果
- 当前主流 benchmark（ClimaQA）与真实用户需求（Real-World Group）在 Topic、Intent、Form 上的余弦相似度均低于 **0.4**，而不同 Human-to-AI 数据集之间的内部一致性超过 **0.94**，凸显其代表性更强。
- Human-to-Human 问答（Reddit）与 Human-to-AI 查询的相似性（>0.85）支持了“以人人为镜，优化人机交互”的可行性。

### 消融实验结果（如有）
- 进行了 **6 种不同配置的主题合并实验**（S1–S6），涉及不同嵌入模型（all-MiniLM-L6-v2 vs Qwen3-Embedding-4B）、是否使用解释文本、不同 LLM 合并器等。
- 结果显示，不同配置下最终主题数量从 170 到 722 不等，但 **核心结论保持稳健**（Appendix C.3），验证了方法鲁棒性。
- 手动整合多个合并路径的结果，进一步提升了分类体系的全面性和一致性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **存在系统性错位**：当前 LLM 气候评估基准严重偏离真实用户的信息需求，集中在低阶事实记忆任务，忽略高阶应用型知识服务。
2. ✅ **人-AI 与人-人间知识需求高度相似**：用户在向人类专家或 LLM 提问时表现出几乎相同的主题偏好和输出期待，说明可借鉴成熟的人类知识交互模式。
3. ✅ **IPCC AR6 是理想的最小知识基底（minimal knowledge base）**：其主题分布与真实需求高度一致，适合作为 benchmark 构建或 RAG 检索语料库的基础。
4. ✅ **SciDCC 等媒体数据偏向生态影响，但公共传播类话题覆盖不足**：需额外补充社交媒体或教育材料以弥补 D4 类主题缺口。

### 方法的局限性
- **领域特定性**：研究聚焦于气候变化，结论在其他领域的泛化能力有待验证。
- **数据来源限制**：
  - Human-to-Human 问题仅来自 Reddit，可能无法代表所有人群（如非英语用户、边缘群体）。
  - Human-to-AI Guidance Knowledge 仅依赖 ClimaQA，缺乏更多 benchmark 对照。
- **主观性风险**：主题分类体系的构建涉及人工决策，虽通过多轮合并与验证缓解，但仍可能存在偏见。
- **标注复杂度高**：细粒度分类增加了人工验证难度，且依赖高成本 LLM 标注。

### 未来工作方向
- 将 **Proactive Knowledge Behaviors Framework** 推广至其他社会技术议题（如人工智能伦理、公共卫生危机）。
- 开发 **动态自适应 benchmark**，能随公众关注热点演变而自动更新。
- 构建 **多模态气候知识接口**，支持图像、图表、视频等形式的回答。
- 探索 **个性化知识推荐机制**，根据不同用户角色（学生、政策制定者、企业主）定制输出风格与深度。
- 发布 **Climate-Need-Bench**：基于本文发现构建一个更贴近真实需求的新一代 LLM 气候评估基准。

> 🔗 代码与数据已开源：[https://github.com/OuchengLiu/LLM-Misalign-Climate-Change](https://github.com/OuchengLiu/LLM-Misalign-Climate-Change)

</details>

---

### 16. [Switch Attention: Towards Dynamic and Fine-grained Hybrid Transformers](https://arxiv.org/abs/2603.26380)

**Authors**: Yusheng Zhao, Hourun Li, Bohan Wu, Jingyang Yuan, Meng Zhang, Yichun Yin, Lifeng Shang, Ming Zhang  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.26380v1  

#### Abstract
The attention mechanism has been the core component in modern transformer architectures. However, the computation of standard full attention scales quadratically with the sequence length, serving as a major bottleneck in long-context language modeling. Sliding window attention restricts the context ...

---

### 17. [In-Context Molecular Property Prediction with LLMs: A Blinding Study on Memorization and Knowledge Conflicts](https://arxiv.org/abs/2603.25857)

**Authors**: Matthias Busch, Marius Tacke, Sviatlana V. Lamaka, Mikhail L. Zheludkevich, Christian J. Cyron, Christian Feiler, Roland C. Aydin  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.25857v1  

#### Abstract
The capabilities of large language models (LLMs) have expanded beyond natural language processing to scientific prediction tasks, including molecular property prediction. However, their effectiveness in in-context learning remains ambiguous, particularly given the potential for training data contami...

---

### 18. [GLU: Global-Local-Uncertainty Fusion for Scalable Spatiotemporal Reconstruction and Forecasting](https://arxiv.org/abs/2603.26023)

**Authors**: Linzheng Wang, Jason Chen, Nicolas Tricard, Zituo Chen, Sili Deng  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.26023v1  

#### Abstract
Digital twins of complex physical systems are expected to infer unobserved states from sparse measurements and predict their evolution in time, yet these two functions are typically treated as separate tasks. Here we present GLU, a Global-Local-Uncertainty framework that formulates sparse reconstruc...

---

### 19. [Topology-Aware Graph Reinforcement Learning for Energy Storage Systems Optimal Dispatch in Distribution Networks](https://arxiv.org/abs/2603.26264)

**Authors**: Shuyi Gao, Stavros Orfanoudakis, Shengren Hou, Peter Palensky, Pedro P. Vergara  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.26264v1  

#### Abstract
Optimal dispatch of energy storage systems (ESSs) in distribution networks involves jointly improving operating economy and voltage security under time-varying conditions and possible topology changes. To support fast online decision making, we develop a topology-aware Reinforcement Learning archite...

---

### 20. [EcoFair: Trustworthy and Energy-Aware Routing for Privacy-Preserving Vertically Partitioned Medical Inference](https://arxiv.org/abs/2603.26483)

**Authors**: Mostafa Anoosha, Dhavalkumar Thakker, Kuniko Paxton, Koorosh Aslansefat, Bhupesh Kumar Mishra, Baseer Ahmad, Rameez Raja Kureshi  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.26483v1  

#### Abstract
Privacy-preserving medical inference must balance data locality, diagnostic reliability, and deployment efficiency. This paper presents EcoFair, a simulated vertically partitioned inference framework for dermatological diagnosis in which raw image and tabular data remain local and only modality-spec...

---

### 21. [PQuantML: A Tool for End-to-End Hardware-aware Model Compression](https://arxiv.org/abs/2603.26595)

**Authors**: Roope Niemi, Anastasiia Petrovych, Arghya Ranjan Das, Enrico Lupi, Chang Sun, Dimitrios Danopoulos, Marlon Joshua Helbing, Mia Liu, Sebastian Dittmeier, Michael Kagan, Vladimir Loncar, Maurizio Pierini  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.26595v1  

#### Abstract
PQuantML is a new open-source, hardware-aware neural network model compression library tailored to end-to-end workflows. Motivated by the need to deploy performant models to environments with strict latency constraints, PQuantML simplifies training of compressed models by providing a unified interfa...

---

### 22. [CADSmith: Multi-Agent CAD Generation with Programmatic Geometric Validation](https://arxiv.org/abs/2603.26512)

**Authors**: Jesse Barkley, Rumi Loghmani, Amir Barati Farimani  
**Category**: cs.AI  
**Published**: 2026-03-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.26512v1  

#### Abstract
Existing methods for text-to-CAD generation either operate in a single pass with no geometric verification or rely on lossy visual feedback that cannot resolve dimensional errors. We present CADSmith, a multi-agent pipeline that generates CadQuery code from natural language. It then undergoes an ite...

---

### 23. [ClimateCheck 2026: Scientific Fact-Checking and Disinformation Narrative Classification of Climate-related Claims](https://arxiv.org/abs/2603.26449)

**Authors**: Raia Abu Ahmad, Max Upravitelev, Aida Usmanova, Veronika Solopova, Georg Rehm  
**Category**: cs.CL  
**Published**: 2026-03-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.26449v1  

#### Abstract
Automatically verifying climate-related claims against scientific literature is a challenging task, complicated by the specialised nature of scholarly evidence and the diversity of rhetorical strategies underlying climate disinformation. ClimateCheck 2026 is the second iteration of a shared task add...

---

### 24. [Adversarial-Robust Multivariate Time-Series Anomaly Detection via Joint Information Retention](https://arxiv.org/abs/2603.25956)

**Authors**: Hadi Hojjati, Narges Armanfard  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.25956v1  

#### Abstract
Time-series anomaly detection (TSAD) is a critical component in monitoring complex systems, yet modern deep learning-based detectors are often highly sensitive to localized input corruptions and structured noise. We propose ARTA (Adversarially Robust multivariate Time-series Anomaly detection via jo...

---

### 25. [AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](https://arxiv.org/abs/2603.26096)

**Authors**: Hyeongyu Kim, Geonhui Han, Dosik Hwang  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.26096v1  

#### Abstract
Test-time adaptation (TTA) aims to mitigate performance degradation under distribution shifts by updating model parameters during inference. Existing approaches have primarily framed adaptation around affine modulation, focusing on recalibrating normalization layers. This perspective, while effectiv...

---

### 26. [Machine Unlearning under Retain-Forget Entanglement](https://arxiv.org/abs/2603.26569)

**Authors**: Jingpu Cheng, Ping Liu, Qianxiao Li, Chi Zhang  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.26569v1  

#### Abstract
Forgetting a subset in machine unlearning is rarely an isolated task. Often, retained samples that are closely related to the forget set can be unintentionally affected, particularly when they share correlated features from pretraining or exhibit strong semantic similarities. To address this challen...

---

### 27. [Hardware-Aware Tensor Networks for Real-Time Quantum-Inspired Anomaly Detection at Particle Colliders](https://arxiv.org/abs/2603.26604)

**Authors**: Sagar Addepalli, Prajita Bhattarai, Abhilasha Dave, Julia Gonski  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.26604v1  

#### Abstract
Quantum machine learning offers the ability to capture complex correlations in high-dimensional feature spaces, crucial for the challenge of detecting beyond the Standard Model physics in collider events, along with the potential for unprecedented computational efficiency in future quantum processor...

---

### 28. [DRiffusion: Draft-and-Refine Process Parallelizes Diffusion Models with Ease](https://arxiv.org/abs/2603.25872)

**Authors**: Runsheng Bai, Chengyu Zhang, Yangdong Deng  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2603.25872v1  

#### Abstract
Diffusion models have achieved remarkable success in generating high-fidelity content but suffer from slow, iterative sampling, resulting in high latency that limits their use in interactive applications. We introduce DRiffusion, a parallel sampling framework that parallelizes diffusion inference th...

---

### 29. [Preventing Data Leakage in EEG-Based Survival Prediction: A Two-Stage Embedding and Transformer Framework](https://arxiv.org/abs/2603.25923)

**Authors**: Yixin Zhou, Zhixiang Liu, Vladimir I. Zadorozhny, Jonathan Elmer  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2603.25923v1  

#### Abstract
Deep learning models have shown promise in EEG-based outcome prediction for comatose patients after cardiac arrest, but their reliability is often compromised by subtle forms of data leakage. In particular, when long EEG recordings are segmented into short windows and reused across multiple training...

---

### 30. [Geometric Evolution Graph Convolutional Networks: Enhancing Graph Representation Learning via Ricci Flow](https://arxiv.org/abs/2603.26178)

**Authors**: Jicheng Ma, Yunyan Yang, Juan Zhao, Liang Zhao  
**Category**: cs.LG  
**Published**: 2026-03-30  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2603.26178v1  

#### Abstract
We introduce the Geometric Evolution Graph Convolutional Network (GEGCN), a novel framework that enhances graph representation learning by modeling geometric evolution on graphs. Specifically, GEGCN employs a Long Short-Term Memory to model the structural sequence generated by discrete Ricci flow, a...

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
