# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-15 07:15:55 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving](https://arxiv.org/abs/2604.12171)

**Authors**: Xu Bai, Muhammed Tawfiqul Islam, Chen Wang, Adel N. Toosi  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2604.12171v1  

#### Abstract
Pipeline parallelism (PP) is widely used to partition layers of large language models (LLMs) across GPUs, enabling scalable inference for large models. However, existing systems rely on static PP configurations that fail to adapt to dynamic settings, such as serverless platforms and heterogeneous GP...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

当前的 **Pipeline Parallelism (PP)** 在 LLM 推理中被广泛用于跨 GPU 分割模型层，以支持大规模模型的推理。然而，**现有的系统依赖静态 PP 配置**，无法适应动态变化的工作负载（如 serverless 平台、异构 GPU 环境）。重新配置 PP 通常需要停止服务并重启，导致分钟级的停机时间，严重影响用户体验。

此外，**在运行时进行 PP 重配置面临三大挑战**：
1. **GPU 内存饱和**：模型权重和 KV Cache 已占满内存，难以为新加载的层腾出空间。
2. **KV Cache 动态调整困难**：现有系统（如 vLLM）采用连续内存预分配，不支持运行时动态扩容或缩容。
3. **KV 状态一致性维护难**：在不停止推理的前提下迁移 KV Cache 容易导致状态不一致或引入长暂停。

---

### ✅ **提出了什么新方法或新思路**

作者提出 **PIPELIVE**，一个支持 **高效、低干扰的实时原地 PP 重配置（Live In-place PP Reconfiguration）** 的 LLM 服务系统。其核心创新包括：

#### **1. 统一的 KV Cache 管理机制**
- 扩展 **PageAttention** 支持 **非连续 KV 块访问**，实现基于块粒度的动态 KV Cache 分配。
- 引入 **Layer Stacking 技术**：将多个层的 KV 块打包到同一个物理 GPU 内存块中，对齐 CUDA 最小分配粒度（如 2MB），显著减少内部碎片化。

#### **2. 增量式 KV Patches 同步机制**
- 受 **虚拟机热迁移（Live VM Migration）** 启发，设计 **KV Patching** 机制，在推理过程中持续同步源与目标配置间的 KV 状态。
- 通过监控“已调度 token”与“已同步 token”的差距，确定安全切换点，最终仅需 **<10ms 的短暂停顿**即可完成切换。

#### **3. 协调协议与运行时控制架构**
- 设计 **Reconfiguration Coordinator** 中心控制器，协调所有 GPU 上的重配置操作。
- 提出五阶段重配置协议（可行性评估 → KV 缩容 → 权重加载 + KV 迁移 → 收敛监测 → 原子提交），确保正确性和最小中断。

---

### ✅ **相比现有方法的优势**

| 方面 | 现有方法（如 vLLM） | PIPELIVE |
|------|------------------------|----------|
| **PP 配置灵活性** | 静态部署，不可变 | 支持运行时动态切换 |
| **KV Cache 调整能力** | 固定大小，无法动态伸缩 | 支持运行时 block-level 扩缩容 |
| **内存利用率** | 存在严重内部碎片 | Layer Stacking 显著降低碎片 |
| **服务中断时间** | 重启导致分钟级停机 | <10ms 微秒级暂停 |
| **适用场景** | 均匀负载、同构环境 | 动态负载、异构 GPU 环境 |

---

## 2. **核心实验方法和设置**

### ✅ **使用的模型与硬件平台**

- **模型**：
  - `Llama3-70B`
  - `Qwen3-30B`
- **测试平台（Heterogeneous Testbed）**：
  - 1 × **NVIDIA A100 (80GB)**：高内存带宽（2039 GB/s）
  - 1 × **NVIDIA L40S (48GB)**：强计算能力（FP16/BF16 达 733 TFLOPS）
  - 两卡跨节点通过 **InfiniBand（约 100 Gbps）** 连接，使用 NCCL 进行通信

> 注：该异构组合自然形成不同工作负载下的最优 PP 配置差异。

---

### ✅ **实验设置与评估指标**

#### **工作负载设计（Pattern-Shifting Benchmark）**
- 交替模拟两种典型负载：
  - **Prefill-heavy**：输入 512 tokens，输出 16 tokens
  - **Decode-heavy**：输入 128 tokens，输出 512 tokens
- 请求速率从 1 到 5 req/s 不等，共 200 个请求

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **TTFT (Time-to-First-Token)** | 请求延迟，反映首响应速度 |
| **TPOT (Time-per-Output-Token)** | 解码延迟，衡量生成效率 |
| **Throughput (tokens/s)** | 总吞吐量，体现系统容量 |
| **Composite Score** | 对 TTFT、TPOT、Throughput 归一化后加权平均，综合评价性能 |

---

### ✅ **基线方法对比**

| 基线 | 描述 |
|------|------|
| **Prefill-Optimal** | 在 prefill-heavy 场景下表现最佳的固定 PP 配置 |
| **Decode-Optimal** | 在 decode-heavy 场景下最优的配置 |
| **Balanced** | 全局折中的静态配置 |
| **PIPELIVE w/o KV Resize** | 禁用 KV 动态调整 |
| **PIPELIVE w/o KV Patch** | 禁用增量同步机制 |
| **PIPELIVE w/o Async Loading** | 同步加载权重 |

---

## 3. **主要实验结果和性能指标**

### ✅ **端到端性能提升**

| 模型 | 方法 | 性能增益（Composite Score） |
|------|------|----------------------------|
| Llama3-70B | PIPELIVE vs. Balanced | **+36%** |
| Qwen3-30B | PIPELIVE vs. Balanced | **+33%** |

> 表明 PIPELIVE 能有效结合 prefill 和 decode 最优配置的优点，在动态负载下始终接近最优。

---

### ✅ **关键性能指标对比**

#### **Llama3-70B 结果**
- **TTFT 改善最多达 45%**
- **TPOT 改善最多达 61%**
- 在 decode-heavy 阶段避免了因 KV Cache 溢出导致的性能崩溃

#### **Qwen3-30B 结果**
- 尽管 TTFT 略差于 balanced（-7%），但：
  - **TPOT 提升 13%**
  - **Throughput 提升 25.7%**
- 显示其更擅长利用 decode-optimal 配置提升整体吞吐

---

### ✅ **消融实验结果**

#### **KV Resize 的影响（图 10）**
- **禁用 KV Resize**：
  - 在 request rate > 1 时即出现 **KV Cache Overflow**
  - TTFT 急剧上升（>2×）
- **启用 KV Resize**：
  - 成功维持 KV 容量匹配目标配置
  - 高负载下 **Throughput 提升超 45%**

> ➤ **结论：KV Resize 是实现稳定重配置的前提**

---

#### **Layer Stacking 的影响（图 11–12）**
- **无 stacking（k=1）**：
  - KV 利用率仅 **56%**，近半内存浪费于内部碎片
  - TTFT 比 k=4 时高 **51%**
- **k=4**：
  - KV 利用率提升至 **~93%**
  - 性能达到峰值
- **k>4（如 k=8）**：
  - 重配置粒度变粗，丧失灵活性，性能下降

> ➤ **结论：k=4 是内存效率与重配置灵活性的最佳平衡点**

---

#### **KV Patch 与异步加载的影响（图 13–14）**
| 设置 | Stop Time | TTFT 改善 | TPOT 改善 |
|------|-----------|------------|------------|
| Baseline（全阻塞） | 数秒级 | - | - |
| Only Async Load | ~100ms | +~30% | +~15% |
| + KV Patching | **<10ms** | **+49.7%** | **+29.5%** |
| Full PIPELIVE | **~10ms** | **+72.4%** | **+26.7%** |

> ➤ **KV Patching 是实现微秒级中断的核心技术**

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **动态 PP 重配置可带来显著性能收益**：
   - 在异构 GPU 环境中，不同工作负载对应不同的最优 PP 配置。
   - 静态配置无法兼顾，而 **PIPELIVE 可动态切换，综合性能提升 33–36%**。

2. **KV Cache 必须支持动态伸缩**：
   - 固定分配会导致 KV 溢出或资源浪费。
   - **扩展 PageAttention 支持 block-level 非连续访问是可行且高效的方案**。

3. **Layer Stacking 极大缓解内存碎片问题**：
   - 将逻辑 KV 块与物理分配单位对齐，使动态管理成为可能。

4. **KV Patching 实现近乎无缝切换**：
   - 类似 VM 热迁移的思想成功迁移到 LLM 推理场景。
   - 将服务中断从“秒级”压缩到“毫秒级”，真正实现 **in-place live reconfiguration**。

---

### ⚠️ **方法的局限性**

1. **未解决“何时触发重配置”问题**：
   - 当前由外部手动指定源/目标配置，缺乏自动决策机制。
2. **依赖高性能互联网络（如 InfiniBand）**：
   - 若带宽不足，KV 同步耗时增加，可能影响收敛速度。
3. **目前仅支持 PP，未整合 TP 或 DP**：
   - 多维度并行联合优化尚未探索。
4. **Layer Stacking 限制了重配置粒度**：
   - 层数必须是 stacking factor 的倍数，牺牲部分灵活性。

---

### 🔮 **未来工作方向**

1. **开发智能重配置策略引擎**：
   - 基于实时负载特征（prefill/decode 比例、请求率）自动选择最优 PP 配置。
2. **联合优化多种并行范式（PP + TP + DP）**：
   - 实现更细粒度的资源适配。
3. **支持多租户动态调度**：
   - 在共享集群中为不同用户提供个性化 PP 配置。
4. **进一步降低 KV Patch 开销**：
   - 探索稀疏更新、差分编码等优化手段。

---

## ✅ **总结**

**PIPELIVE 是首个实现高效、低干扰、原地 PP 重配置的 LLM 服务系统**。它通过三项核心技术——**动态 KV Cache 管理、Layer Stacking、KV Patching**——解决了长期存在的“重配置即停机”难题。实验表明，其可在异构环境中实现 **33–36% 的综合性能提升**，同时将服务中断控制在 **10ms 以内**，为构建弹性、自适应的下一代 LLM 推理系统奠定了坚实基础。

</details>

---

### 2. [Accelerating Microswimmer Simulations via a Heterogeneous Pipelined Parallel-in-Time Framework](https://arxiv.org/abs/2604.12083)

**Authors**: Ruixiang Huang, Weifan Liu  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2604.12083v1  

#### Abstract
Simulating large-scale microswimmer dynamics in viscous fluid poses significant challenges due to the coupled high spatial and temporal complexity. Conventional high-performance computing (HPC) methods often address these two dimensions in isolation, leaving a critical gap for synergistic accelerati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Accelerating Microswimmer Simulations via a Heterogeneous Pipelined Parallel-in-Time Framework*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
微泳者（microswimmer）在粘性流体中的大规模、长时间动力学模拟面临极高的计算复杂度，主要源于两个方面：
- **空间复杂度**：基于拉格朗日框架的流固耦合（FSI）模型（如 Method of Regularized Stokeslets, MRS）需要计算 $O(N^2)$ 的成对相互作用，导致空间计算成本高昂。
- **时间复杂度**：Kirchhoff杆模型具有强刚性（stiff），传统显式积分器需极小时间步长以维持稳定性，导致长期模拟需数百万次迭代。

现有高性能计算（HPC）方法通常将空间并行与时间并行解耦处理，未能协同优化时空双重复杂性。

---

### 提出的新方法与创新思路
本文提出一种**异构CPU-GPU流水线并行时域框架**（heterogeneous pipelined parallel-in-time framework），实现时空协同加速，主要贡献如下：

#### ✅ 可扩展的多GPU流水线Parareal架构
- 将传统的 **Parareal算法** 改造为**流水线调度模式**（pipelined Parareal），使粗粒度求解器（coarse solver）在完成一个子区间后立即向下游传递结果，允许细粒度求解器（fine solver）提前启动。
- 利用MPI-GPU分布式架构，在多个GPU设备上重叠执行粗/细求解器，显著减少GPU空闲时间（idle time），提升硬件利用率。

#### ✅ GPU优化的高密度并行核函数
- 针对MRS中主导计算开销的线性和角速度计算，设计了高度并行化的CUDA核函数，充分利用GPU的SIMT架构。
- 提出一种**专用于3×3旋转矩阵平方根**的GPU优化数值例程，替代通用`scipy.linalg.sqrtm`，避免复杂控制流，提高稳定性和执行效率。

#### ✅ 异构协同的两级并行化策略
- **空间层面**：利用GPU并行处理$O(N^2)$的Stokeslet相互作用；
- **时间层面**：通过流水线Parareal实现时间维度上的并行；
- 二者结合形成“**时空混合并行**”（space-time hybrid parallelism），突破传统串行时间推进瓶颈。

---

### 相比现有方法的优势
| 维度 | 传统方法 | 本工作 |
|------|--------|-------|
| 时间并行 | 标准Parareal，粗求解器必须全局串行完成 | 流水线Parareal，粗求解器边计算边传递，消除等待 |
| 硬件利用 | 单一CPU或GPU加速，资源利用率低 | 多GPU协同，有效重叠计算与通信 |
| 数值稳定性 | 通用矩阵运算库不适合GPU | 定制化3×3矩阵平方根算法，兼顾精度与性能 |
| 扩展性 | 时空并行解耦，难以协同优化 | 统一框架支持弱/强扩展，适用于大规模系统 |

---

## 2. 核心实验方法和设置

### 数据集与物理模型
- **模拟对象**：细丝状微泳者（如精子、细菌鞭毛），建模为Kirchhoff弹性杆。
- **流体环境**：低雷诺数下的Stokes流，采用**Method of Regularized Stokeslets (MRS)** 求解。
- **边界条件**：半无限流体域，底部存在无滑移平面壁（no-slip wall）。
- **运动驱动**：施加正弦波形的预设曲率（preferred strain-twist vector）模拟鞭毛波动。

### 实验设置
- **离散参数**：
  - 杆长度 $L$ 分为 $M=51$ 个点；
  - 正则化参数 $\epsilon = 4\Delta s$；
  - 时间步长（细求解器）$\Delta t = 10^{-6}$。
- **硬件平台**：
  - **GPU**：NVIDIA A100 PCIe 40GB；
  - **CPU**：Kunpeng-920（3.0GHz）与AMD 7H12（2.6GHz, 128核）；
  - 平台运行Kylin Linux。
- **软件实现**：Python + `numba.cuda` 编写GPU核函数，MPI用于跨节点通信。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Speedup** | $ S = T_{\text{CPU}} / T_{\text{GPU}} $，衡量GPU加速比 |
| **Relative Error** | 与串行精细求解器结果的相对误差，验证收敛性 |
| **Relative Increment** | 连续Parareal迭代间的差值，作为收敛判据 |
| **Weak/Strong Scaling** | 衡量系统可扩展性 |
| **GPU Idle Time** | 分析资源利用率的关键理论与实证指标 |

### 基线方法对比
- **CPU-only串行求解器**：作为基准性能参考；
- **标准Parareal方法**（non-pipelined）：用于对比流水线版本的时间并行效率；
- **scipy.linalg.sqrtm**：作为矩阵平方根计算的CPU基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📊 表I：单GPU下各组件的CPU/GPU性能对比（以25根杆为例）
| 组件 | CPU时间 (s) | GPU时间 (s) | 加速比 |
|------|-------------|------------|--------|
| 初始化 | 0.276 | 0.000687 | **402×** |
| 速度计算 | 0.888 | 0.000386 | **2302×** |
| 正交标架更新 | 0.152 | 0.000638 | **239×** |
| **总计** | **1.316** | **0.001711** | **769×** |

> ⚡️ 结论：GPU在所有阶段均实现数百至上千倍加速，尤其在主导项“速度计算”中表现突出。

---

#### 📊 表III：流水线 vs 标准Parareal运行时间对比（单位：秒）
| Rods | r=2 (2 GPUs) | r=5 (2 GPUs) | r=4 (4 GPUs) | r=10 (4 GPUs) |
|------|---------------|--------------|--------------|----------------|
| 1 | 815.7 / **552.9** | 508.1 / **408.6** | 810.2 / **548.2** | 501.0 / **421.0** |
| 25 | 3461.9 / **2819.4** | 2756.3 / **2381.4** | 2333.8 / **1749.9** | 1623.6 / **1294.1** |

> 🔽 **平均提速约20–30%**，当 $r$ 较小时优势更明显（即粗求解器较慢时，流水线缓解等待效果更强）。

---

#### 📊 表V：强扩展性测试（固定问题规模，增加GPU数量）
| GPU数 | 总时间 (s) | 加速比 | 并行效率 |
|-------|------------|--------|----------|
| 1 | 9597.82 | 1.00 | 100% |
| 2 | 4834.20 | 1.99 | 99.5% |
| 4 | 2492.66 | 3.85 | 96.3% |
| 8 | 1555.91 | 6.17 | **77.1%** |

> ✅ 接近线性加速至4 GPU；8 GPU时因跨节点通信开销略有下降。

---

#### 📊 弱扩展性（表IV）
| T | GPU数 | 流水线时间 (s) | 理想时间增长倍数 | 实际增长倍数 |
|----|--------|------------------|--------------------|----------------|
| 0.5 → 4 | 1 → 8 | 4283 → 9231 | 8× | ~2.15× |

> ✅ 问题规模扩大8倍，运行时间仅增约2.15倍，显示良好弱扩展性。

---

#### 🔍 消融实验与分析
- **流水线调度 vs 标准调度**：
  - 实验验证了理论预测：GPU空闲时间差 $\Delta W \propto (m-1)(ln - l(l-1)/2)/r$；
  - 图7显示 $T_{\text{reg}} - T_{\text{pipe}} \propto 1/r$，与理论一致；
  - 更多GPU（$m=4$）时性能差距更大，说明流水线在多设备场景更具优势。

- **矩阵平方根性能对比**：
  - GPU定制算法相比`scipy.sqrtm`平均提速 **2.14×**；
  - 数值误差相当甚至更优（见表II），且更适合SIMT执行。

- **收敛性验证**（图5）：
  - Parareal算法在约4次迭代内达到 $10^{-12}$ 精度；
  - 不同杆数（1–25）下均稳定收敛，表明方法鲁棒性强。

---

## 4. 关键结论和发现

### 主要发现
1. **流水线Parareal显著降低GPU空闲时间**，尤其在粗求解器较慢（$r$ 小）或多GPU（$m$ 大）情况下优势突出。
2. **GPU空间并行带来数百至上千倍加速**，特别是对$O(N^2)$速度计算部分。
3. **时空混合并行框架具备优良的弱/强扩展性**，适合大规模生物流体系统的长期演化研究。
4. **定制化3×3矩阵平方根算法在GPU上兼具高效与稳定**，优于通用库函数。

---

### 方法的局限性
1. **资源利用率仍受限于任务依赖与负载不均**：随着模拟时间 $T$ 和GPU数量增加，部分GPU可能处于空闲状态。
2. **Python实现限制峰值性能**：当前基于`numba.cuda`的实现虽灵活，但不如C++/CUDA原生代码高效。
3. **参数$r$需权衡效率与收敛性**：过大的$r$可能导致粗求解器精度不足，影响Parareal收敛，尤其在初始杆间距较近、系统较刚时。

---

### 未来工作方向
1. **优化GPU调度策略**：引入动态负载均衡机制，进一步减少空闲时间。
2. **增强框架对刚性问题的鲁棒性**：探索更高阶或隐式的粗/细求解器组合。
3. **迁移至低级语言**：使用C++/CUDA重构核心模块，释放更高性能潜力。
4. **拓展至多尺度/多物理场应用**：如群体自组织、微血管血流等复杂生物系统模拟。

---

> ✅ **总体评价**：该工作成功构建了一个面向生物流体仿真的高效异构并行框架，首次将**流水线Parareal**与**GPU空间加速**深度融合，为大规模微泳者动力学模拟提供了可扩展、高性能的解决方案。

</details>

---

### 3. [Three Birds, One Stone: Solving the Communication-Memory-Privacy Trilemma in LLM Fine-tuning Over Wireless Networks with Zeroth-Order Optimization](https://arxiv.org/abs/2604.12401)

**Authors**: Zhijie Cai, Yuhao Zheng, Haolong Chen, Dongzhu Liu, Bin Wang, Guangxu Zhu  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.12401v1  

#### Abstract
Federated Learning (FL) offers a promising pathway for collaboratively fine-tuning Large Language Models (LLMs) at the edge; however, this paradigm faces a critical bottleneck: the prohibitive communication and memory overheads incurred by exchanging high-dimensional gradients. Furthermore, recent s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Three Birds, One Stone: Solving the Communication-Memory-Privacy Trilemma in LLM Fine-tuning Over Wireless Networks with Zeroth-Order Optimization*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对在无线边缘网络中对 **Large Language Models (LLMs)** 进行 **Federated Learning (FL)** 微调时面临的“三难困境”（trilemma）：
- **Communication Bottleneck**：传统 FL 需要传输高维梯度，通信开销巨大。
- **Memory Wall**：标准反向传播（Backpropagation, BP）需要存储中间激活值，超出边缘设备内存容量。
- **Privacy Leakage**：即使不共享原始数据，攻击者仍可通过本地梯度重构用户隐私数据（如通过梯度反演攻击）。

现有方法通常单独解决其中一个问题，难以兼顾效率与安全。

### 提出的新方法：pAirZero 和 Sign-pAirZero
作者提出 **pAirZero**，一个将 **Zeroth-Order (ZO) Optimization** 与 **Over-the-Air (OTA) Computation** 相结合的新型联邦微调框架，并进一步设计其数字版本 **Sign-pAirZero**。

#### 核心创新点：
- **一体化设计**：首次将 ZO 优化与 OTA 聚合协同设计，同时缓解通信、内存和隐私三大挑战。
- **Bit-Level Communication**：利用 ZO 仅需传输梯度在随机方向上的投影（scalar），结合 PRNG（伪随机数生成器）同步方向，实现每轮迭代 **1-bit 或 16-bit** 的极低通信负载。
- **Inference-Level Memory**：避免反向传播，仅需前向推理即可估计梯度，内存消耗降至推理级别（约降低 75%）。
- **Privacy-by-Design**：利用无线信道噪声和人工注入噪声，在 OTA 信号叠加过程中自然实现 **Differential Privacy (DP)**，无需额外加噪步骤。
- **自适应功率控制**：建立优化模型，动态调整发射功率和噪声强度，在保证收敛性的前提下最小化最优性差距并满足 DP 约束。

### 相比现有方法的优势
| 维度 | 传统方法（如 FO-SGD + FL） | pAirZero |
|------|----------------------------|---------|
| **通信开销** | 与模型维度成正比（~238MB） | 与设备数量无关，固定为 bit-level（1~16 bits） |
| **内存消耗** | 高（需存储激活值） | 极低（仅需推理内存） |
| **隐私保护** | 需显式加噪（影响精度） | “空中”自然加噪，隐私与通信共存 |
| **同步要求** | 严格时间/相位同步 | 放宽同步要求，更适用于实际系统 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **OPT-125M**：轻量级预训练语言模型，作为微调对象。
- **下游任务**：
  - **SST-2**：二分类情感分析任务（句子级情感判断）。
  - **SQuAD v1.1**：问答任务（从段落中抽取答案跨度）。

### 实验设置
- **客户端数量**：$ K = 5 $
- **本地数据量**：每个客户端 1000 个样本
- **训练轮数**：$ T = 8000 $
- **扰动尺度**：$ \mu = 0.001 $
- **隐私预算**：$ (\epsilon, \delta) = (5, 0.01) $
- **硬件平台**：AMD EPYC CPU + 四张 NVIDIA A100 GPU
- **学习率**：通过网格搜索确定（见 Table I）
- **信道模型**：块衰落（block fading），考虑 AWGN 噪声
- **最大信噪比（SNR_max）**：用于评估不同信道条件下的性能

### 评估指标
- **模型性能**：
  - SST-2：Accuracy
  - SQuAD：F1 Score
- **资源效率**：
  - 内存占用（Memory Cost）
  - 每轮上传通信量（Per-iteration Upload）
- **消融实验**：比较不同功率分配策略的效果

### 基线方法对比
- **Perfect**：理想无噪声聚合，作为性能上界
- **Static**：静态功率分配（非自适应）
- **Reversed**：反转自适应趋势（验证优化方向有效性）
- **FO-SGD / FO-Adam**：基于一阶优化的传统联邦学习方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table II 和 Fig. 2）
| 方法 | Memory Cost | Per-iteration Upload |
|------|-------------|-----------------------|
| **FO-SGD** | ~600 MB | 238.88 MB |
| **FO-Adam** | 955.58 MB | 238.88 MB |
| **pAirZero** | ~250 MB | **16 bits** |
| **Sign-pAirZero** | ~250 MB | **1 bit** |

> ✅ **通信开销降低数个数量级**，**内存成本减少约 75%**

### 与基线方法的对比结果
- 在 **SST-2** 和 **SQuAD** 上，**pAirZero** 和 **Sign-pAirZero** 的最终性能接近 **Perfect** 情况，显著优于传统方法在相同资源限制下的表现。
- 尽管引入了噪声以满足 DP 要求，但模型准确率/F1 分数下降有限，表明该方法能在强隐私保护下保持良好效用。
- **Sign-pAirZero** 在不同 SNR 下表现更稳定，尤其在低 SNR 区域优于模拟版本（pAirZero），因其对噪声鲁棒性更强。

### 消融实验结果（见 Fig. 3）
- **Solution-based Power Allocation（基于优化的功率分配）** 明显优于 **Static** 和 **Reversed** 策略。
- 使用静态功率会导致模型性能严重下降，尤其是在训练后期需要更高信道增益时。
- 自适应功率控制能有效平衡隐私、通信质量和收敛速度。

---

## 4. 关键结论和发现

### 主要发现
1. **ZO + OTA 是解决 LLM 边缘微调三难困境的有效路径**：pAirZero 成功实现了通信、内存、隐私三方面的协同优化。
2. **Bit-level Communication 可行且高效**：通过 ZO 投影压缩，可将通信负载从 GB 级降至 bit 级，极大提升实用性。
3. **无线信道噪声可用于构建隐私机制**：结合人工噪声，可在物理层实现 **privacy-by-design**，无需牺牲过多模型性能。
4. **自适应功率控制至关重要**：动态调整参数可显著缩小最优性差距，提升收敛速度和稳定性。

### 方法的局限性
- 当前理论分析依赖于一些理想假设（如梯度 Lipschitz 连续、PL 条件等），在复杂真实场景中可能不完全成立。
- 实验基于较小规模模型（OPT-125M），扩展到更大模型（如 Llama-3）的实际部署仍需验证。
- 对 PRNG 同步机制的安全性和容错能力未深入探讨。
- 数字调制（Sign-pAirZero）虽鲁棒性强，但引入了符号翻转误差，可能影响收敛。

### 未来工作方向
- 扩展至多模态大模型的联邦微调。
- 探索更高效的 ZO 估计器以减少采样方差。
- 结合 **Parameter-Efficient Fine-Tuning (PEFT)** 方法（如 LoRA）进一步降低计算负担。
- 在真实无线环境中进行原型验证（testbed implementation）。
- 研究异构客户端（non-IID 数据、不同计算能力）下的鲁棒性优化。

--- 

> **总结**：pAirZero 开创性地将 ZO 优化与 OTA 计算深度融合，提出了一种面向边缘 LLM 微调的高效、低耗、隐私安全的新范式，为未来去中心化 AI 提供了重要技术路径。

</details>

---

### 4. [Fast and principled equation discovery from chaos to climate](https://arxiv.org/abs/2604.11929)

**Authors**: Yuzheng Zhang, Weizhen Li, Rui Carvalho  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.11929v1  

#### Abstract
Our ability to predict, control, and ultimately understand complex systems rests on discovering the equations that govern their dynamics. Identifying these equations directly from noisy, limited observations has therefore become a central challenge in data-driven science, yet existing library-based ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast and principled equation discovery from chaos to climate

## 1. 论文的主要贡献和创新点

### 解决的问题
本文致力于解决**方程发现**（equation discovery）中的一个核心挑战：如何在**自动化程度、统计严谨性**（statistical rigor）和**计算效率**之间取得平衡。现有的基于库的稀疏回归方法（library-based sparse regression）通常只能满足其中两个目标，难以同时兼顾三者。

- **自动化**：减少人工调参需求；
- **统计严谨性**：提供模型选择的理论依据和不确定性量化；
- **计算效率**：适用于大规模或高维系统。

### 提出的新方法
作者提出了 **Bayesian-ARGOS**，一种混合框架，结合了**频繁主义筛选**（frequentist screening）和**贝叶斯推断**（Bayesian inference）的优点，实现“粗到精”（coarse-to-fine）的方程发现流程。

#### 方法核心思想
1. **两阶段筛选（Frequentist Screening）**：
   - 第一阶段使用 **Adaptive LASSO**（带岭回归权重）进行初步变量选择，降低维度；
   - 第二阶段使用 **OLS 权重**的 Adaptive LASSO 进行二次筛选，并通过 **BIC** 选择最优模型；
   - 设计矩阵在两阶段间动态优化，防止过度正则化。
2. **贝叶斯后验推断（Bayesian Inference）**：
   - 在筛选后的精简设计矩阵上，使用 **Hamiltonian Monte Carlo (HMC)** 进行贝叶斯采样；
   - 利用后验分布构建**可信区间**（credible intervals），保留不包含零的项作为最终模型；
   - 实现**不确定性量化**（uncertainty quantification）。

### 相比现有方法的优势
| 维度 | Bayesian-ARGOS | SINDy | ARGOS |
|------|----------------|-------|--------|
| **自动化** | 高（自动参数选择） | 低（依赖手动阈值） | 中高（需Bootstrap） |
| **统计严谨性** | 高（贝叶斯后验 + 不确定性） | 低（无不确定性） | 中（Bootstrap置信区间） |
| **计算效率** | 高（约比ARGOS快100倍） | 极高（确定性算法） | 低（多次Bootstrap重拟合） |
| **数据效率** | 最优（多数系统所需样本最少） | 较差 | 居中 |
| **噪声鲁棒性** | 最强（6/7系统优于SINDy） | 弱 | 中等 |

---

## 2. 核心实验方法和设置

### 数据集
1. **七种混沌系统**（chaotic systems）作为基准测试：
   - Lorenz, Thomas, Rössler, Dadras, Aizawa, Sprott, Halvorsen
   - 所有系统均为三维非线性 ODE，具有不同复杂度（如三角函数、高阶多项式）
2. **真实世界高维时空数据**：
   - NOAA 海表温度（SST）数据（180×360 网格，1400周时间序列）
   - 用于测试与深度学习框架 **SINDy-SHRED** 的集成能力

### 实验设置
- **数据生成**：
  - 每个系统生成100条轨迹，随机初始条件；
  - 添加高斯噪声，控制 **SNR**（信噪比）从1 dB到61 dB；
  - 观测数 $ n $ 从 $10^2$ 到 $10^5$ 变化。
- **候选库构造**：
  - 多项式项最高至5次；
  - 对含三角函数的系统（如Thomas）加入 $\sin(x_i)$, $\cos(x_i)$；
  - 总共56–62个候选项。

### 评估指标
1. **成功率**（Success Rate）：
   $$
   \text{Success Rate} = \frac{\text{正确识别所有真实项的试验次数}}{100}
   $$
   - 要求：无假阳性和假阴性；
   - 阈值：80%（Aizawa系统为70%）。
2. **计算时间**（Runtime）：
   - 记录单次运行时间，比较算法效率。
3. **预测误差**（用于SST任务）：
   - Latent MSE（潜空间均方误差）
   - Reconstructed RMSE（重建场根均方误差）

### 基线方法对比
- **SINDy**（Sparse Identification of Nonlinear Dynamics）：
  - 使用 PySINDy 实现，STLSQ优化器，阈值0.1；
  - 快速但缺乏不确定性量化。
- **ARGOS**（Automated Regression for Governing Equations）：
  - 当前最先进的自动化稀疏回归方法；
  - 结合 Adaptive LASSO + Bootstrap + BIC；
  - 准确但计算成本高。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）在混沌系统上的表现（Fig. 2, 3, S1）
| 系统 | 数据效率（最小 $n$） | 噪声鲁棒性（最低 SNR） | 成功率峰值 |
|------|------------------------|--------------------------|------------|
| **Lorenz** | < $10^{3.2}$ | 27 dB | 100% |
| **Thomas** | < $10^{3.2}$ | 17 dB | 100% |
| **Rössler** | ~ $10^{2.5}$ | 27 dB | 100% |
| **Aizawa** | ~ $10^{3.7}$ | 27 dB | ~70% |
| **Dadras** | ~ $10^{2.7}$ | 17 dB | 100% |

- **Bayesian-ARGOS 在5/7系统中比ARGOS更省数据，在6/7系统中比SINDy更抗噪**。
- 对于复杂系统（如Thomas、Aizawa），优势尤为明显。

#### （2）计算效率（Fig. 6）
- 在 $n=10^5$ 时：
  - **ARGOS**: > $10^{4.7}$ 秒（约3小时）
  - **Bayesian-ARGOS**: < $10^{2.5}$ 秒（约300秒）
  - **加速约100倍**
- SINDy最快（<10秒），但无不确定性输出。

#### （3）与SINDy-SHRED集成在SST任务中的表现（Fig. 8）
| 指标 | Bayesian-ARGOS | SINDy |
|------|----------------|--------|
| **有效方程率**（Valid ID Rate） | **77%** (82/107) | 60% (64/107) |
| **平均Latent MSE** | **0.263** | 0.334 |
| **平均Reconstruction RMSE** | **1.055** | 1.282 |
| **长期预测稳定性** | 显著更好（误差增长慢） | 快速发散 |

> 示例发现的潜动力学为**仿射线性系统**：
> $$
> \dot{z} = Az + b
> $$
> 特征分析显示存在周期约 **1.01年** 的弱阻尼振荡模式（对应季节循环）和衰减时间约 **1.25年** 的快速模态。

### 消融实验与机制分析
- **失败模式诊断**（Fig. 3–4）：
  - **Aizawa系统在大数据量下性能下降**：由严重**多重共线性**（multicollinearity）导致，VIF > $10^4$；
  - **Dadras系统在大 $n$ 下性能下降**：因**影响点**（influential observations）过多（PSIS-LOO $k > 0.7$）；
  - **Rössler/Sprott在无噪声下性能下降**：因**异方差性**（heteroscedasticity）破坏同方差假设，导致过选择。

这些分析表明：**更多数据或更低噪声并不总是有益**，而 Bayesian-ARGOS 的概率框架能揭示这些“反直觉”失败机制。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Bayesian-ARGOS 成功调和了自动化、统计严谨性和计算效率之间的矛盾**：
   - 通过“先频繁筛选，再贝叶斯推断”的分工策略，实现了三者的协同而非妥协。
2. ✅ **在大多数混沌系统中，Bayesian-ARGOS 在数据效率和噪声鲁棒性上优于 SINDy 和 ARGOS**：
   - 尤其对含三角函数和高阶非线性的系统（如 Thomas, Aizawa）优势显著。
3. ✅ **计算效率提升两个数量级**：
   - 相比 ARGOS 加速约100倍，使其可用于大规模实际应用。
4. ✅ **不确定性量化带来额外价值**：
   - 支持标准统计诊断（PSIS-LOO, VIF, 残差分析），可识别模型误设、影响点和共线性等问题。
5. ✅ **模块化设计支持高维扩展**：
   - 与 SINDy-SHRED 集成后，在海温重建任务中显著提高有效方程发现率（+17%）和长期预测稳定性。

### 方法的局限性
1. ⚠️ **仍受限于候选库的表达能力**：
   - 若真实项不在库中（如未知函数形式），无法发现；
   - 依赖领域知识构建合理的 $\Phi(X)$。
2. ⚠️ **贝叶斯推断仅作用于筛选后空间**：
   - 未对筛选过程本身进行不确定性建模（post-selection inference 仍是开放问题）。
3. ⚠️ **在极低信噪比下可能略逊于 ARGOS**：
   - 如 Sprott 和 Halvorsen 系统在 SNR < 20 dB 时成功率稍低，反映其正则化权衡策略的取舍。

### 未来工作方向
1. 🔮 **扩展至 PDE 发现**：
   - 结合 mesh-free 或 symbolic regression 技术处理偏微分方程。
2. 🔮 **引入物理先验**（physics-informed priors）：
   - 在贝叶斯框架中融入守恒律、对称性等约束。
3. 🔮 **自适应库构建**：
   - 动态生成候选函数（如通过 symbolic regression 或 transformer）。
4. 🔮 **在线/增量式方程发现**：
   - 适用于实时监测和控制系统。

---

> **总结**：  
> **Bayesian-ARGOS** 是一种兼具**原则性**（principled）、**自动化**和**高效性**的方程发现框架。它不仅提升了性能，更重要的是提供了**可解释的失败诊断路径**，推动了从“黑箱算法”向“透明科学发现工具”的转变，为从气候到神经科学等领域的复杂系统建模提供了实用且可靠的解决方案。

</details>

---

### 5. [Evolution of Optimization Methods: Algorithms, Scenarios, and Evaluations](https://arxiv.org/abs/2604.12968)

**Authors**: Tong Zhang, Jiangning Zhang, Zhucun Xue, Juntao Jiang, Yicheng Xu, Chengming Xu, Teng Hu, Xingyu Xie, Xiaobin Hu, Yabiao Wang, Yong Liu, Shuicheng Yan  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.12968v1  

#### Abstract
Balancing convergence speed, generalization capability, and computational efficiency remains a core challenge in deep learning optimization. First-order gradient descent methods, epitomized by stochastic gradient descent (SGD) and Adam, serve as the cornerstone of modern training pipelines. However,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Evolution of Optimization Methods: Algorithms, Scenarios, and Evaluations

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文系统地回顾并分析了深度学习优化算法的演进历程，旨在解决当前领域内存在的三大关键缺陷：
1.  **研究范围狭窄**：现有综述大多局限于传统的 **First-Order (FO)** 和 **Second-Order (SO)** 方法，忽视了快速发展的 **Zeroth-Order (ZO)** 优化和面向特定场景（如分布式、隐私保护）的框架。
2.  **缺乏统一分类法**：现有工作缺少一个严谨、统一的数学分类体系，导致对不同方法之间内在联系和演化逻辑的理解是碎片化和不一致的。
3.  **缺乏标准化基准**：没有一个公平、大规模的实证基准来评估现代优化器在不同架构上的表现，导致文献中的结论相互矛盾且不可复现。

### 提出的新方法/新思路
论文提出了一个全面的“三位一体”解决方案：
1.  **统一的分类法 (Unified Taxonomy)**：
    *   建立了一个严谨的数学框架，将优化算法分为四大类：**First-Order (FO)**, **Second-Order (SO)**, **Zeroth-Order (ZO)** 和 **Scenario-Oriented Paradigms**。
    *   提出了一个通用的离散时间动力学系统公式 `0t+1=Pe(0-mM.-1m-m入0)`，通过解耦四个维度（梯度估计器 `E`、预处理器 `M`、场景变换 `Tscenario`、结构投影 `Pe`）来揭示不同算法间的内在联系和演化逻辑。

2.  **面向场景的分析 (Scenario-Oriented Analysis)**：
    *   论文强调，现代优化已从纯粹的算法设计转变为一种**系统感知的工程解决方案**。它深入分析了基础算法（FO/SO/ZO）如何被重新架构以应对物理瓶颈，如分布式通信障碍和严格的差分隐私约束。

3.  **标准化的评估框架 (Standardized Evaluation)**：
    *   设计了一个严格控制的评估协议，分离了算法性能与大规模工程优化的影响。
    *   开发了一个标准化测试平台，在多种架构代理（CNN 和 Transformer）上评估了23种主流优化器。

### 相比现有方法的优势
*   **全面性**：首次将 **ZO** 和 **Scenario-Oriented** 范式纳入主流优化综述，填补了研究空白。
*   **系统性**：提出的统一分类法为理解算法的演化提供了清晰的理论框架，超越了零散的描述。
*   **可操作性**：提供的标准化基准和实证结果为研究人员和工程师选择和设计优化器提供了可靠的指导。

---

## 2. 核心实验方法和设置

### 数据集
实验在三个广泛采用的基准数据集上进行：
*   **视觉任务 (Vision Tasks)**：
    *   **ImageNet-1K**：用于图像分类。
    *   **模型**：**ResNet-50** (代表CNN) 和 **ViT-Small (ViT-S)** (代表Vision Transformer)。
*   **语言任务 (Language Tasks)**：
    *   **WikiText-103**：用于因果语言建模。
    *   **模型**：一个 **60M参数的Llama** 架构模型。作者明确指出，此配置并非为了复制大模型的涌现行为，而是作为一个计算上可行的、具有代表性的Transformer架构代理，用于评估优化器的跨架构泛化能力。

### 实验设置和评估指标
*   **超参数设置**：为了公平比较，所有优化器仅调整一个共同的超参数——**学习率 (Learning Rate)**。通过网格搜索（在默认值基础上缩放0.1, 0.2, 1.0, 5.0, 10.0倍）找到最优学习率，并将其应用于所有模型。
*   **评估指标**：
    *   **视觉任务**：使用 **Top-1 Accuracy** 作为主要指标。
    *   **语言任务**：使用 **Perplexity (PPL)** 作为主要指标。
*   **训练设置**：
    *   **视觉任务**：在DeiT和A2/A3等标准训练设置下，分别进行100轮和300轮训练，以评估长期可扩展性。
    *   **语言任务**：遵循Semenov et al. (2025)的配置，使用256的批量大小和512的序列长度。
*   **基线方法对比**：实验对比了23种不同的优化器，涵盖了FO、SO、ZO及其混合变体，包括经典的 **SGD**, **Adam**, **AdamW**，以及新兴的 **Lion**, **Muon**, **MARS**, **MeZO** 等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **学习率敏感性 (Learning Rate Sensitivity)**：
    *   **SGD** 对过大的学习率表现出惊人的鲁棒性，尽管其峰值准确率较低。
    *   **Adam** 家族的方法（如Adam, AdamW）对过大学习率非常敏感，容易发散。
    *   **Muon** 表现出最强的超参数不敏感性，在0.2x到5.0x的学习率范围内都保持了超过75%的高精度，这得益于其通过矩阵正交化实现的内在正则化。

2.  **跨架构泛化能力 (Cross-Architecture Generalization)**：
    *   在 **Llama** 模型上，许多在视觉任务上表现良好的自适应优化器（如 **RMSprop**, **Adam**, **MADGRAD**, **AdaBelief**）出现了严重的训练崩溃或收敛困难。
    *   **Muon** 和 **MARS** 在跨架构泛化上表现最为出色。它们在Llama模型上即使在极端学习率缩放下也能保持稳定且低的PPL（~12-14）。
    *   **SGD** 系列在Llama上遭遇了灾难性的训练崩溃，凸显了其在处理大语言模型高度各向异性的损失景观时的不足。

3.  **长期训练可扩展性 (Long-term Training Scalability)**：
    *   **SGD** 系列在长期训练中表现出强大的可扩展性，随着训练轮数增加，准确率持续提升。
    *   **先进优化器**（如 **Muon**, **Lion**, **Kron**, **AdamW**）在早期就能快速达到高精度，但在300轮训练后增益有限，表明它们已经充分挖掘了网络的表征能力。

4.  **综合性能排名**：
    *   综合考量收敛速度、准确性、内存效率、泛化能力和超参数鲁棒性，**Muon** 获得了最佳的整体性能评价。

### 消融实验结果
*   论文通过分析不同优化器的更新规则（如图4、5、6、7所示），间接进行了消融分析。
*   例如，分析表明 **Lion** 的符号更新机制是其高下界稳定性的关键；**MARS** 的梯度校正能有效抵消由复杂架构和数据分布引入的随机噪声。

---

## 4. 关键结论和发现

### 主要发现
1.  **范式转变**：现代优化已从追求纯算法改进，转变为**系统感知的工程设计**。优化器必须在收敛速度、泛化能力、计算效率、内存占用、通信开销和隐私保护之间进行复杂的权衡。
2.  **ZO优化的重要性**：**Zeroth-Order** 方法不仅是理论上的升级，更是一种根本性的范式转变，它通过牺牲几何精度来换取极致的物理可行性（如内存效率），对于大模型微调至关重要。
3.  **超参数鲁棒性是关键**：在跨架构和长周期训练中，**超参数鲁棒性**比单纯的峰值性能更重要。**Muon** 和 **MARS** 等方法的成功证明了这一点。
4.  **传统方法的局限性**：经典 **SGD** 在大语言模型上会失败，而标准 **Adam** 对学习率过于敏感。这表明需要新的、更稳健的设计原则。

### 方法的局限性
*   **计算成本高昂**：文中提到，完整的实证评估消耗了超过 **1073个单卡A100小时**，这对于大多数研究者来说是不可承受的，突显了现代优化器基准测试的根本性计算壁垒。
*   **理论与实践的差距**：尽管提出了统一的框架，但许多先进的混合方法（尤其是基于 **ZO** 的）仍然缺乏严格的理论收敛保证。
*   **规模限制**：实验使用的Llama模型（60M参数）远小于实际的大语言模型（LLMs），因此结论可能无法完全外推到千亿级模型。

### 未来工作方向
1.  **自动化优化器设计**：利用程序化搜索、神经控制器或元学习（meta-learning）来自动生成针对特定架构和任务的优化器。
2.  **硬件-算法协同设计 (Hardware-Algorithm Co-design)**：优化器设计应与硬件特性（如低精度算术、内存层次结构）紧密结合。
3.  **精确的噪声管理**：开发能够精确抵消由网络架构和数据分布引入的固有随机噪声的框架。
4.  **面向场景的理论保障**：为分布式和隐私保护框架建立严格的理论下界，确保在通信效率和隐私效用之间的权衡是可预测和可控的。

</details>

---

### 6. [Enhancing Clustering: An Explainable Approach via Filtered Patterns](https://arxiv.org/abs/2604.12460)

**Authors**: Motaz Ben Hassine (CRIL), Sa\"id Jabbour (CRIL)  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.12460v1  

#### Abstract
Machine learning has become a central research area, with increasing attention devoted to explainable clustering, also known as conceptual clustering, which is a knowledge-driven unsupervised learning paradigm that partitions data into $\theta$ disjoint clusters, where each cluster is described by a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Enhancing Clustering: An Explainable Approach via Filtered Patterns*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **explainable clustering**（即可解释聚类，又称概念聚类）中的一个关键瓶颈：  
在基于 **k-Relaxed Frequent Patterns (k-RFPs)** 的聚类框架中，多个不同的 k-RFPs 可能诱导出相同的 **k-cover**，导致大量冗余的 symbolic representations（符号化表示）。这种冗余不仅扩大了搜索空间，还显著增加了后续 **Integer Linear Programming (ILP)** 求解器的计算复杂度和运行时间。

### 🚀 提出的新方法与创新思路
作者提出了一种名为 **Optimized Conceptual Clustering Method (OCCM)** 的新框架，其核心思想是通过**模式过滤策略**消除冗余的 k-RFPs。具体贡献如下：

1. **理论分析冗余成因**  
   形式化地刻画了“不同 k-RFPs 诱导相同 k-cover”的条件（见 *Proposition 2*），为检测和去除冗余提供了理论基础。

2. **设计高效的过滤算法**  
   提出一种 **Pattern Filtering Algorithm**，对所有生成的 k-RFPs 进行处理，仅保留每个唯一 k-cover 对应的一个代表性 pattern。选择标准是：**保留最大（largest）itemset**，以增强可解释性。

3. **引入新的可解释性评估指标**  
   提出两个用于衡量 selected pattern 对其 cluster 表达能力的指标：
   - **Shapley Value Variance (SVV)**：量化 pattern 内部各 item 贡献的不均衡程度。
   - **Average Cluster Stability (ACS)**：衡量当移除单个 item 后，cluster 是否保持稳定。

### ⚖️ 相比现有方法的优势
- **效率提升**：显著减少 ILP 输入的候选 patterns 数量，从而大幅缩短求解时间。
- **质量不降反升**：在部分数据集上（如 Mushroom、Primary-Tumor），聚类质量（F1-score）反而更高。
- **可解释性更强**：优先选择更大的 itemset，提供更丰富、更具代表性的 cluster 描述。
- **无损去冗余**：保留了所有唯一的 k-cover，确保语义表达能力不受影响。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验在六个真实世界 transactional datasets 上进行，基本信息如下表所示：

| Dataset         | \|D\|   | \|U\| | Density (%) |
|----------------|--------|-------|-------------|
| Lymph          | 148    | 68    | 40          |
| Mushroom       | 8124   | 119   | 18          |
| Primary-Tumor  | 336    | 31    | 48          |
| Soybean        | 630    | 50    | 32          |
| Tic-tac-toe    | 958    | 27    | 33          |
| Vote           | 435    | 48    | 33          |

> 注：所有数据集均具有明确的二分类 ground-truth，因此聚类任务被视为 binary classification 问题。

### ⚙️ 实验设置
- **k 参数固定为 1**：即允许最多缺失 1 个 item 仍视为被覆盖（k-cover）。
- **聚类数 O = 2**：与 ground-truth 一致。
- **最小支持度阈值 σ**：从 10% 到 40% 不等，在不同阶段调整。
- **SAT Solver**：用于枚举 k-RFPs，沿用 Hassine et al. (2024) 的修改版 SAT 求解器。
- **ILP 求解器**：解决聚类优化问题。
- **最大运行时间限制**：1 小时；超时则视为未找到最优解。

### 📊 评估指标
1. **模式数量对比**：比较过滤前后 k-RFPs 的数量，评估冗余程度。
2. **ILP 求解时间 (CPU time)**：衡量计算效率。
3. **聚类质量 (F1-score)**：与 ground-truth 比较，评估聚类准确性。
4. **可解释性分析指标**：
   - **SVV**（越低表示 item 贡献越均衡）
   - **ACS**（越高表示 cluster 越稳定）

### 🔁 基线方法对比
- **CCA-k-RFP-M1**：由 Hassine et al. (2024) 提出的原始方法，直接使用全部 k-RFPs 输入 ILP。
- **OCCM**（本文方法）：在输入 ILP 前先进行 pattern filtering。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ Phase I：模式过滤效果（冗余消除）
| Dataset       | 支持度 σ | △ 最大降幅 (%) |
|--------------|---------|----------------|
| Tic-tac-toe  | 40%     | **26.67%**     |
| Lymph        | 40%     | 8.30%          |
| Mushroom     | 40%     | 11.19%         |
| Vote         | 40%     | 3.61%          |

> 所有数据集均观察到 \|Af\| < \|A\|，验证了冗余普遍存在，且过滤有效。

#### ✅ Phase II：ILP 性能与聚类质量对比（σ 选优后）

| Dataset       | 方法             | \|A\| / \|Af\| | F1-score | CPU time (s) |
|--------------|------------------|---------------|-----------|---------------|
| Lymph        | CCA-k-RFP-M1     | 91,888        | 0.71      | 49.66         |
|              | **OCCM**         | **85,470**    | **0.71**  | **29.74**     |
| Mushroom     | CCA-k-RFP-M1     | 19,712        | 0.34      | 3133.34       |
|              | **OCCM**         | **18,176**    | **0.73**  | **1144.46**   |
| Primary-Tumor| CCA-k-RFP-M1     | 45,465        | 0.25      | 55.88         |
|              | **OCCM**         | **45,250**    | **0.33**  | **55.71**     |
| Soybean      | 两者             | ~11k          | 0.29      | ~17–18        |
| Vote         | CCA-k-RFP-M1     | 280,386       | 0.51      | 1206.17       |
|              | **OCCM**         | **280,179**   | **0.51**  | **234.30**    |
| Tic-tac-toe  | 两者             | ~800          | —         | — (timeout)   |

> **结论**：
> - 所有可成功求解的数据集上，**OCCM 显著降低 CPU time**（最高提速约 5 倍）。
> - 在 Mushroom 和 Primary-Tumor 上，**F1-score 明显提升**，说明去冗余有助于选出更优 pattern。
> - Tic-tac-toe 因组合爆炸未能在时限内求解，凸显了高效预处理的重要性。

#### ✅ Phase III：可解释性分析（SVV 与 ACS）

| Dataset       | Pattern Size | SVV       | ACS     |
|--------------|--------------|-----------|---------|
| Lymph        | 4, 13        | 48.89, 2.22 | 0.84, 0.95 |
| Mushroom     | 5, 2         | 146486, 27367 | 0.84, 0.86 |
| Primary-Tumor| 2, 4         | 12.5, 319.02 | 0.53, 0.75 |
| Soybean      | 3, 9         | 4742, 6.26 | 0.70, 0.97 |
| Vote         | 2, 3         | 5724, 1196 | 0.53, 0.58 |

> **关键发现**：
> - 存在明显的 **负相关趋势**：**SVV 越低，ACS 越高** → item 贡献越均衡，cluster 越稳定。
> - **Pattern size 与 ACS 正相关**：更大的 pattern 更稳定（见 Figure 4），支持了“保留最大 itemset”的合理性。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **冗余普遍存在**：多个 distinct k-RFPs 可共享相同 k-cover，造成不必要的计算开销。
2. **过滤策略高效可行**：通过保留每类 k-cover 中最大的 pattern，可在不损失语义的前提下显著压缩搜索空间。
3. **性能全面提升**：
   - **计算效率**：ILP 求解时间平均下降数十至数百秒，部分场景提速超过 3 倍。
   - **聚类质量**：在多个数据集上 F1-score 提升，表明去冗余有助于模型聚焦高质量 pattern。
4. **可解释性增强**：
   - 更大的 pattern 具有更高的 **ACS**，表明其描述更稳健。
   - **SVV 与 ACS 负相关**，提示均衡贡献的 item 结构更有助于构建稳定的 cluster。

### ⚠️ 局限性
- **依赖两阶段流程**：先 SAT 枚举再 ILP 选择，无法完全避免初始冗余生成。
- **SAT 阶段未优化**：当前过滤为 post-processing，若能在 SAT 枚举过程中直接避免冗余生成会更高效。
- **目标函数单一**：ILP 仅最大化 pattern size，尚未整合 SVV 或 ACS 等可解释性指标作为优化目标。

### 🔮 未来工作方向
1. **将 redundancy-awareness 集成进 SAT 枚举过程**，设计紧凑约束防止重复生成相同 k-cover 的 patterns。
2. **扩展 ILP 目标函数**，引入 SVV、ACS 等指标作为正则项或多目标优化项，主动追求“既准确又可解释”的聚类结果。
3. **探索动态 k 值选择机制**，根据不同 cluster 特征自适应调整松弛参数 k。
4. **应用于更大规模或高维数据**，验证方法的可扩展性。

---

> **总结一句话**：  
> 本文提出的 **OCCM** 框架通过**理论驱动的 pattern filtering**，有效解决了 k-RFP-based clustering 中的冗余问题，在**不牺牲甚至提升聚类质量的前提下，显著提高了计算效率与可解释性**，为 explainable AI 下的知识发现提供了实用而坚实的工具。

</details>

---

### 7. [BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining](https://arxiv.org/abs/2604.12241)

**Authors**: Haojie Ye, Arjun Laxman, Yichao Yuan, Krisztian Flautner, Nishil Talati  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.12241v1  

#### Abstract
Money laundering detection faces challenges due to excessive false positives and inadequate adaptation to sophisticated multi-stage schemes that exploit modern financial networks. Graph analytics and AI are promising tools, but they struggle with the fuzziness of laundering patterns, which exhibit s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 Anti-Money Laundering (AML) 系统面临以下挑战：
- **高误报率**：基于规则的方法难以适应复杂的洗钱行为。
- **无法有效检测多阶段洗钱模式**：如 scatter-gather、multi-hop cycles 等，这些模式具有**结构性模糊性**（structural fuzziness）和**时间模糊性**（temporal fuzziness）。
- **图挖掘效率低下**：现有系统在处理大规模交易图时吞吐量低，且难以表达模糊模式。

### 提出的新方法与创新思路
作者提出 **BlazingAML**，一个可扩展的 AML 系统，其核心是：
- **多阶段框架（Multi-Stage Framework）**  
  将复杂洗钱模式分解为一系列逻辑阶段（logical stages），每个阶段通过基本图操作（如邻居扩展、集合交集）连接。该框架能统一表达多种模式（如 scatter-gather、cycle），并自然捕捉结构和时间上的模糊性。
  
- **领域专用编译器（Domain-Specific Compiler）**  
  将高层模式描述自动编译为高性能的 C++ 和 CUDA 内核代码，支持 CPU 和 GPU 后端。编译器负责：
  - 并行化优化（OpenMP / CUDA）
  - 内存访问优化（power-law-aware）
  - 工作负载均衡（degree-based）
  - 流水线执行（pipelined CPU-GPU）

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **表达能力** | 支持模糊模式（无需枚举所有变体），优于固定模板匹配方法（如 Cypher） |
| **性能** | 显著高于现有系统（GFP），实现高达 333× 加速 |
| **易用性** | 分析师只需定义逻辑结构，无需编写底层并行代码 |
| **灵活性与可扩展性** | 新模式可通过修改 stage 定义快速集成，无需重写算法 |

---

## 2. 核心实验方法和设置

### 数据集
使用 IBM Research 发布的合成金融交易数据集 [1]，包含不同规模和欺诈密度的图：
| 类别 | 节点数 | 边数 |
|------|--------|-------|
| LI-Small ~ LI-Large | ~70万–207万 | ~690万–1.76亿 |
| HI-Small ~ HI-Large | ~51万–211万 | ~500万–1.8亿 |

其中：
- **LI**: Low Illicit（低欺诈密度）
- **HI**: High Illicit（高欺诈密度）

此外，在可扩展性测试中使用 **Trovares** 生成的从 10K 到 100M 边的合成图。

---

### 实验设置与评估指标

#### 评估指标
- **F1 Score**：用于衡量分类性能（因数据高度不平衡，F1 更合理）
- **Throughput (edges/sec)**：衡量图模式挖掘吞吐量
- **Speedup**：相对于基线的加速比

#### 基线方法对比
| 基线 | 描述 |
|------|------|
| **GFP [4]** | 当前最先进的 AML 系统，采用子图特征提取 + XGBoost 分类器，作为主要对比对象 |
| **FraudGT [19]** | 基于 Graph Transformer 的端到端深度学习模型，代表最新 AI 方法 |

#### 硬件平台
- **CPU**: 双路 Intel Xeon Platinum 8380（共 80 线程）
- **GPU**: 单张 NVIDIA A40（48GB GDDR6）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ F1 Score 对比（表 2 & 图 11）
- BlazingAML **完全复现了 GFP 的特征输出**，因此在相同下游分类器（XGBoost）下达到**相同的 F1 Score**。
- 添加更多图模式特征（Fan → Degree → Cycle → Scatter-Gather）显著提升 F1：
  - 在 HI-Large 上，F1 从 20.2（仅账户ID）提升至 **58.1**
- 验证了图模式特征对 AML 检测的有效性。

#### ✅ 吞吐量与加速比（图 6–9）
| 模式 | CPU 加速比（vs GFP 64线程） | GPU 加速比 |
|------|-------------------------------|------------|
| **Scatter-Gather** | **210×**（平均） | **333×** |
| **Cycle** | 最高 159× | — |
| **Fan-in/out** | ~11.4×（32线程） | 进一步优化可达更高 |
| **Stack** | 最高 25.8×（64线程） | **33.5×** |

> ⚠️ 注：即使单线程 CPU 版本也已接近 GFP 的 64 线程性能，说明编译器生成代码极其高效。

#### ✅ 可扩展性研究（图 10）
在 Trovares 生成的 10K–100M 边图上测试 scatter-gather 模式：
- 随着图规模增大，BlazingAML 的优势持续扩大
- 在 100M 边图上，**平均加速达 27.5×（64线程 CPU）**
- GPU 实现相比 GFP 达到 **24.4× 加速**

#### ✅ 与 FraudGT 对比（图 12 & 表 4）
| 指标 | 结果 |
|------|------|
| **F1 Score** | FraudGT 更高（例如 HI-Medium: 62.3 vs 51.1）——因其使用更强的 Transformer 模型 |
| **吞吐量** | BlazingAML（128线程）平均处理速度是 FraudGT 的 **4.9×** |
| **实际意义** | BlazingAML 更适合实时、大规模部署场景 |

---

## 4. 关键结论和发现

### 主要发现
1. **模糊模式可以被统一建模**：通过 multi-stage 抽象，scatter-gather、cycle 等多样模式可用相同原语表达，极大简化开发。
2. **编译器能自动生成高性能代码**：无需专家手动调优，即可实现远超人工优化系统的性能。
3. **图特征增强 + 轻量级分类器 是高效路径**：相比端到端深度学习（如 FraudGT），BlazingAML 的“图挖掘 + XGBoost”方案在保持精度的同时获得数量级性能提升。
4. **系统具备良好可扩展性**：在千万级边图上仍维持线性加速趋势。

---

### 方法的局限性
- **依赖预定义模式**：仍需领域专家设计 pattern stages，不能完全自动化发现新型未知洗钱模式。
- **未整合更先进 AI 模型**：当前 pipeline 使用 XGBoost，若替换为 GNN 或 Graph Transformer 可能进一步提效，但会牺牲部分速度。
- **合成数据验证**：虽然 IBM 数据集被认为是 state-of-the-art 合成数据，但仍非真实银行流水，泛化能力有待实证。

---

### 未来工作方向
1. **支持动态模式发现**：结合 unsupervised 或 self-supervised 子图发现技术，减少对人工规则的依赖。
2. **集成 GNN 编译优化**：将 BlazingAML 的编译思想扩展至 GNN 推理阶段，构建全栈优化 AML pipeline。
3. **流式增量更新机制深化**：当前支持 streaming，但可进一步优化状态维护与窗口管理。
4. **跨机构联合建模隐私保护机制**：探索联邦学习 + 图挖掘的结合方式，应对数据孤岛问题。

---

> 📌 **总结一句话**：  
> **BlazingAML 通过“多阶段抽象 + 领域编译器”的设计，在不损失检测精度的前提下，实现了高达 333× 的图模式挖掘加速，为大规模、实时反洗钱系统提供了实用且高效的解决方案。**

</details>

---

### 8. [SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling](https://arxiv.org/abs/2604.12110)

**Authors**: Zikun Liu, Liang Luo, Qianru Li, Zhengyu Zhang, Wei Ling, Jingyi Shen, Zeliang Chen, Yaning Huang, Jingxian Huang, Abdallah Aboelela, Chonglin Sun, Feifan Gu, Fenggang Wu, Hang Qu, Huayu Li, Jill Pan, Kaidi Pei, Laming Chen, Longhao Jin, Qin Huang, Tongyi Tang, Varna Puvvada, Wenlin Chen, Xiaohan Wei, Xu Cao, Yantao Yao, Yuan Jin, Yunchen Pu, Yuxin Chen, Zijian Shen, Zhengkai Zhang, Dong Liang, Ellie Wen  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.12110v1  

#### Abstract
Recent advances in recommendation scaling laws have led to foundation models of unprecedented complexity. While these models offer superior performance, their computational demands make real-time serving impractical, often forcing practitioners to rely on knowledge distillation-compromising serving ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代推荐系统中，**Foundation Models (FMs)** 虽然在建模能力和个性化方面表现出色，但由于其计算复杂度高，难以直接用于实时推理（real-time serving）。因此，工业界通常采用 **Knowledge Distillation (KD)** 将 FM 的知识迁移到轻量级的 **Vertical Models (VMs)** 中。然而，传统 KD 存在以下瓶颈：

- **Transfer Ratio 低**：基于 soft-label 的蒸馏仅能实现约 20–25% 的知识迁移效率；
- **知识传递局限于训练阶段**：无法在 inference time 进行动态知识共享；
- **任务耦合性强**：生成的 soft-label 难以泛化到多个下游任务。

### 提出了什么新方法或新思路
作者提出 **SOLARIS**（Speculative Offloading of Latent-bAsed Representation for Inference Scaling），一种面向大规模推荐系统的高效、可扩展的知识共享框架，核心思想是：

> **将昂贵的 FM 推理过程从延迟敏感的服务路径中解耦，通过异步预计算用户-项目交互 embedding，并在 inference time 注入 VMs，实现“推理时知识蒸馏”**。

### 相比现有方法的优势
相比传统的 soft-label 知识蒸馏，SOLARIS 具有三大创新优势：

| 创新点 | 描述 | 优势 |
|--------|------|-------|
| **Direct Embedding-based Transfer** | 直接迁移 FM 中间层输出的 user-item interaction embedding，而非 soft-label | 提供更丰富、更具通用性的表示，显著提升 transfer ratio |
| **Speculative Embedding Precomputation** | 受 LLM 中 speculative decoding 启发，预测未来可能请求的 user-item 对并提前生成 embedding | 实现 inference-time knowledge distillation，避免在线计算开销 |
| **Hierarchical Feature Enrichment** | 当目标 user-item 缺失 embedding 时，通过聚合用户历史 embedding 或利用相似用户进行补全 | 显著提升覆盖率（coverage），保障长尾场景下的性能 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 实验部署于 **Meta 的广告系统**，覆盖全球数十亿日活用户的实际线上流量；
- 数据为真实世界中的 `(user, ad)` 请求对，涵盖 Facebook Feed、Facebook Reels、Instagram 等多个产品表面；
- 模型预测目标为 **CTR (Click-Through Rate)** 和 **CVR (Conversion Rate)**。

### 实验设置和评估指标
- **部署环境**：生产级多阶段排序系统（multi-stage ranking pipeline），SOLARIS 应用于 final-stage ranking 模型；
- **输入增强方式**：将预计算的 FM user-ad embedding 作为额外特征输入至 VMs；
- **缓存机制**：使用分布式缓存存储 `<user, item>` 对应的 embedding，TTL 设为数小时以保证新鲜度；
- **评估指标**：
  - 主要指标：**Relative Binary Cross-Entropy (BCE) Loss Reduction (%)**
  - 辅助指标：**Feature Coverage (%)**, **Revenue Impact**

### 基线方法对比
- **Baseline**：无 SOLARIS 的现有生产系统，各 VM 独立训练，不引入 FM embedding；
- **对比维度**：
  - 是否使用 embedding 输入
  - 不同 coverage 下的性能变化
  - 不同类型 embedding（原始、聚合、相似用户）的效果差异

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 表 1：SOLARIS 在多种任务上的 BCE Loss 改进（相对下降 %）

| Task Category | Sub-task | Relative BCE Loss Reduction (%) |
|-------------|---------|-------------------------------|
| CTR         | Facebook Feed | 0.09% |
| CTR         | Facebook Reels | 0.08% |
| CTR         | Instagram | 0.05% |
| CTR         | Instagram Link Click | 0.05% |
| CVR         | Facebook Feed + Reels (Offsite Conversion) | 0.10% |
| CVR         | Offsite Conversion | 0.05% |

> 所有子任务均取得一致且显著的损失下降，表明 embedding 共享具有良好的跨任务泛化能力。

#### ✅ Transfer Ratio 提升
- SOLARIS 在 Instagram 上达到 **42% transfer ratio**，Facebook 上达 **44%**；
- 相比传统 soft-label 蒸馏（~20–25%）**提升超过 2 倍**。

#### ✅ 商业收益
- **全局广告收入提升 0.67%**，相当于每年增加约 **$100M 收入**；
- 用户-广告 embedding 覆盖率达 **40%**（异步预计算）；
- 引入 **aggregated user-only embedding** 后，覆盖率提升至 **~90%**，带来额外 **0.03% BCE 改进**；
- 使用 **similarity-based embedding**（KNN 补全），覆盖率提升至 **70%**，离线 BCE 改善 **0.02%**。

### 与基线方法的对比结果
| 方法 | BCE 改进 | Coverage | Revenue Gain |
|------|--------|----------|--------------|
| Baseline (no SOLARIS) | 0% | — | 0% |
| SOLARIS (full) | up to 0.13% | 60–90% | **+0.67%** |
| Soft-label KD (prior art) | ~0.05% | <25% transfer | N/A |

> SOLARIS 在性能、覆盖率和商业价值上全面超越传统 KD。

### 消融实验结果（Ablation Study）
- **Coverage vs Performance**（见 Table 2）：
  - 当 feature coverage 从 20% 提升至 100%，Instagram CTR 的 BCE loss 改善从 0.05% 升至 0.3%；
  - 表明 **更高覆盖率可解锁更大性能潜力**。
- **Hierarchical Enrichment 贡献**：
  - Aggregated user embedding：+0.03% BCE 改进；
  - Similarity-based embedding：+0.02% 改进（约为理论上限的 40%）；
  - 验证了两种补全策略的有效性，尽管 neighbor embedding 质量略低。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Embedding-level transfer 比 label-level 更高效**：直接迁移 FM 的 user-item interaction embedding 可实现高达 **44% 的 transfer ratio**，远超传统 KD；
2. **Inference-time knowledge distillation 是可行且高效的**：通过 speculative precomputation 解耦训练与服务路径，可在零延迟代价下提升 VM 性能；
3. **Hierarchical enrichment 显著提升覆盖率**：结合 aggregated 和 similarity-based 方法，embedding 可用性从 40% 提升至 90%，极大增强了系统鲁棒性；
4. **大规模部署产生显著商业价值**：在全球广告系统中实现 **0.67% 广告收入增长**，验证了技术的实际影响力。

### 方法的局限性
1. **Coverage 仍有提升空间**：当前异步预计算受限于资源成本，仅覆盖约 40% 的 user-ad 对；
2. **早期排序阶段尚未应用**：由于候选集规模过大（early-stage 有 100x 更多 ads），目前仅适用于 final-stage ranking；
3. **聚类方法较简单**：当前 U2U（user-to-user）相似性基于 KNN 和 cosine similarity，未探索 A2A（ad-to-ad）或 hybrid 策略；
4. **公司特定依赖强**：部分设计受 Meta 内部架构和业务需求驱动，通用性受限。

### 未来工作方向
- **扩展 coverage 提升策略**：
  - 探索 **A2A（ad-to-ad）clustering** 和 **hybrid (U2U + A2A)** 方法；
  - 引入动态 prioritization 机制优化预计算优先级；
- **向 early-stage ranking 延伸**：
  - 设计更轻量化的 embedding 提取与缓存策略，支持更大规模候选集；
- **改进 similarity modeling**：
  - 使用 graph-based 或 contrastive learning 方法构建更精准的用户/广告邻域；
- **支持更多模态与任务**：
  - 将 SOLARIS 框架推广至视频推荐、搜索排序等其他场景。

---

> 🔚 **总结一句话**：  
> **SOLARIS 成功实现了“把大模型的知识搬到小模型的服务路径上”，通过 speculative precomputation + embedding sharing，在不增加延迟的前提下，释放了 Foundation Model 的全部潜力，带来了可观的技术与商业回报。**

</details>

---

### 9. [Interpretable Relational Inference with LLM-Guided Symbolic Dynamics Modeling](https://arxiv.org/abs/2604.12806)

**Authors**: Xiaoxiao Liang, Juyuan Zhang, Liming Pan, Linyuan L\"u  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.12806v1  

#### Abstract
Inferring latent interaction structures from observed dynamics is a fundamental inverse problem in many-body interacting systems. Most neural approaches rely on black-box surrogates over trainable graphs, achieving accuracy at the expense of mechanistic interpretability. Symbolic regression offers e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Interpretable Relational Inference with LLM-Guided Symbolic Dynamics Modeling

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**从观测到的动力学轨迹中推断潜在交互结构**（relational inference）这一逆问题。在许多复杂系统（如物理、生物、流行病学等）中，实体间的交互图结构通常是不可观测的，而只能获得节点状态的时间序列数据。现有方法面临以下挑战：
- **神经网络方法**（如 NRI）虽然能联合学习结构与动力学，但其“黑箱”特性导致缺乏**机制可解释性**（mechanistic interpretability）。
- **符号回归**（Symbolic Regression, SR）虽能生成显式的动力学方程，但通常假设已知图结构，且依赖固定的函数库（fixed library），难以适应未知拓扑和复杂非线性。

### 提出的新方法：COSINE
作者提出了 **COSINE**（Co-Optimization of Symbolic Interactions and Network Edges），一个端到端可微分的框架，用于**联合发现交互图结构和稀疏符号动力学方程**。

#### 核心创新点：
- **联合优化结构与机制**：将动力学分解为 **message-passing** 和 **update** 两个模块，分别建模节点间相互作用与自身状态演化，并通过共享的符号函数库进行稀疏线性组合。
- **稀疏符号消息传递**（Sparse Symbolic Message Passing）：引入稀疏回归约束，防止过参数化模型拟合虚假边，从而增强结构可识别性。
- **LLM引导的库演化机制**（LLM-Guided Library Evolution）：
  - 外层循环使用 **Large Language Model**（LLM）作为“符号监督器”，根据内层训练反馈（损失、残差模式、系数权重）动态地**剪枝冗余项**并**增补新的候选函数**。
  - 脱离了对预定义、领域特定模板的依赖，实现了开放世界的符号空间探索。
- **双层闭环优化架构**：
  - 内层：基于 Gumbel-Softmax 的可微图生成 + 稀疏符号回归联合优化。
  - 外层：LLM 驱动的符号库迭代进化。

### 相比现有方法的优势
| 方法类型 | 局限性 | COSINE 的优势 |
|--------|-------|-------------|
| 统计方法（GC, MI, TE） | 忽略显式动力学建模，难以处理非线性 | 显式建模动力学，具备更强表达能力 |
| 黑箱神经模型（NRI, GDP） | 缺乏可解释性，易过拟合虚假边 | 输出人类可读的符号表达式，机制透明 |
| 固定库符号回归（SINDy） | 闭世界假设，无法应对未知非线性 | 动态扩展函数库，适应复杂系统 |

---

## 2. 核心实验方法和设置

### 数据集
#### 合成数据（Synthetic Data）
在三种标准图结构上生成六类代表性动力学系统：
- **图结构**：Erdos-Rényi (ER), Barabasi-Albert (BA), Watts-Strogatz (WS)，每种 $N=50$ 节点。
- **动力学系统**：
  1. **Michaelis-Menten (MM)**：生化反应动力学
  2. **Diffusion (Diff)**：扩散过程
  3. **Spring (Spr)**：弹簧网络力学耦合
  4. **Kuramoto (Kura)**：振子同步模型
  5. **Friedkin-Johnsen (FJ)**：意见动力学
  6. **Coupled Map Network (CMN)**：混沌映射网络

#### 真实世界数据
- **COVID-19 流行病数据**：来自美国四个州（Arizona, Connecticut, Illinois, Michigan）的县级每日确诊病例数。
- 包含地理邻接关系与人口流动信息作为参考图。

### 实验设置与评估指标
#### 主要任务
- **Relational Inference**：预测真实存在的边（binary classification）
- **Mechanism Discovery**：恢复正确的动力学方程形式

#### 评估指标
- **AUC**：用于衡量图结构推理性能（主要指标）
- **Term Accuracy**：前 $K=3$ 个绝对系数最大的项中，覆盖真实机制原语的比例（primitive coverage）
- **PCC**（Pearson Correlation Coefficient）：在真实数据中用于评估预测轨迹的相关性

#### 基线方法对比
| 类别 | 方法 |
|------|------|
| 统计方法 | Granger Causality (GC), Mutual Information (MI), Transfer Entropy (TE) |
| 神经关系推理 | Neural Relational Inference (NRI), Graph Dynamics Prior (GDP) |
| 注意力模型 | RIVA (Relational Inference via Variational Attention) |

所有基线均调优超参以确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| DYN | GRAPH | GC | MI | TE | NRI | GDP | RIVA | **COSINE** |
|-----|-------|----|----|----|-----|-----|------|------------|
| MM | ER-50 | 60.24 | 77.84 | 53.66 | 96.25 | 98.31 | 52.17 | **99.63** |
| Diff | WS-50 | 73.63 | 76.39 | 90.04 | 99.31 | 56.68 | 53.72 | **100.00** |
| Spr | ER-50 | 50.61 | 72.24 | 76.05 | 99.84 | 99.99 | 52.82 | **100.00** |
| Kura | WS-50 | 59.27 | 100.00 | 96.63 | 50.17 | 100.00 | 98.19 | **100.00** |
| FJ | WS-50 | 99.81 | 66.14 | 95.37 | 68.76 | 89.51 | 52.89 | **100.00** |

> ✅ **结论**：COSINE 在绝大多数场景下达到 **state-of-the-art 性能**，尤其在复杂非线性系统（如 MM, CMN）上显著优于其他方法。

### 机制发现结果（见 Table 2）
- COSINE 成功恢复了多个系统的**真实机制原语**：
  - **Kuramoto**：正确识别出 `sin(xj - xi)` 作为主要交互项。
  - **Diffusion**：发现 `xj - xi` 差分项。
  - **FJ 模型**：识别出度归一化更新项 `k_i / (k_i + 1)`。
- **Term Accuracy 达到 1.0** 表明其能精准捕捉物理一致的机制。

### 与基线方法的对比结果
- **统计方法**（GC/MI/TE）在高度非线性系统中表现接近随机猜测。
- **NRI/GDP** 虽然在部分系统中表现尚可，但在 Kuramoto 等系统上严重退化（如 BA 图上 AUC < 70）。
- **RIVA** 作为注意力模型，在多数情况下不如 COSINE。
- **COSINE 始终保持高鲁棒性**，不受图类型或动力学复杂度影响。

### 消融实验结果（Ablation Study）

#### （1）LLM 引导库演化的必要性（Figure 4）
- **w/o update**（固定库）：性能大幅下降，尤其在 MM 和 CMN 上。
- **阈值剪枝策略**（threshold-based pruning）：效果不稳定。
- **✅ 结论**：LLM 的主动推理能力对于突破“闭世界”限制至关重要。

#### （2）LLM 尺寸的影响（Table 3）
| LLM 模型 | 参数量 | MM (AUC) | CMN (AUC) |
|---------|--------|----------|-----------|
| Deepseek-R1 | 8B | 0.53 | 0.51 |
| GPT-OSS | 20B | 1.00 | 1.00 |
| Qwen2.5 | 72B | 1.00 | 1.00 |

> ✅ **结论**：对于简单系统（如 Spring），小模型即可胜任；但对于高阶非线性（如 MM, CMN），需要更大规模 LLM 才能稳定发现复杂原语。

#### （3）超参数敏感性分析（Figure 5）
- 对学习率有一定容忍范围，存在“性能高原区”。
- 对正则化参数 $\lambda_w$ 极其鲁棒，即使高达 2.0 仍保持高 AUC。
- 表明框架具有良好的优化稳定性。

---

## 4. 关键结论和发现

### 主要发现
1. **COSINE 实现了可解释的关系推理**：不仅能准确重建图结构（AUC 接近 1.0），还能输出**紧凑、物理一致的符号表达式**，揭示系统内在机制。
2. **LLM 可作为有效的符号假设生成器**：通过反馈驱动的方式，LLM 能有效指导符号空间搜索，避免穷举或盲目演化。
3. **稀疏性是结构可识别性的关键**：相比黑箱神经模型，稀疏符号回归更难吸收虚假边，提升了因果发现的可靠性。
4. **在低数据场景下更具鲁棒性**（Table 4）：当样本仅 10×10 时，COSINE 仍显著优于基线，表明其强归纳偏置带来的**高数据效率**。

### 方法的局限性
- **对 LLM 能力有依赖**：极端复杂的动力学可能超出当前 LLM 的符号构造能力。
- **稀疏线性回归的表达能力有限**：虽然结合 LLM 扩展库，但仍受限于线性组合形式，难以表示深层嵌套结构。
- **符号发现可能存在非唯一性**：在噪声或有限数据下，不同表达式可能拟合相似，导致机制歧义。
- **计算成本较高**：外层 LLM 查询带来额外延迟，不适合实时应用。

### 未来工作方向
- 探索更高效的符号搜索策略（如 MCTS + LLM）。
- 将框架扩展至连续时间系统与潜变量建模。
- 引入物理先验（如守恒律）进一步约束符号空间。
- 应用于更多领域：气候建模、金融网络、脑科学等。

---

> 🔗 **代码开源地址**：https://anonymous.4open.science/r/COSINE-6D43  
> 📚 **一句话总结**：COSINE 通过 **LLM 引导的动态符号库 + 可微稀疏回归**，实现了**高精度、高可解释性、高鲁棒性**的联合关系与机制发现，为复杂系统建模提供了新范式。

</details>

---

### 10. [TCL: Enabling Fast and Efficient Cross-Hardware Tensor Program Optimization via Continual Learning](https://arxiv.org/abs/2604.12891)

**Authors**: Chaoyao Shen, Linfeng Jiang, Yixian Shen, Tao Xu, Guoqing Li, Anuj Pathania, Andy D. Pimentel, Meng Zhang  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.12891v1  

#### Abstract
Deep learning (DL) compilers rely on cost models and auto-tuning to optimize tensor programs for target hardware. However, existing approaches depend on large offline datasets, incurring high collection costs and offering suboptimal transferability across platforms. In this paper, we introduce TCL, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TCL: Enabling Fast and Efficient Cross-Hardware Tensor Program Optimization via Continual Learning —— 核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **offline-trained cost model** 的深度学习（DL）编译器在跨硬件平台迁移时面临三大挑战：
1. **高数据收集成本**：为每个新硬件平台采集大规模 tensor program 性能数据耗时极长（如 CPU 需 40 天，GPU 超过 60 天）。
2. **cost model 架构效率低**：Transformer 类模型虽能捕捉长程依赖，但计算复杂度为 $O(n^2)$，训练和推理开销大；LSTM 并行性差、收敛慢。
3. **跨平台泛化能力弱**：现有 transfer learning 多为单源到单目标的一对一迁移，而 multi-task learning（如 MTL-TLP）存在参数爆炸和需同时访问多平台数据的问题。

### 提出的新方法与创新
作者提出 **TCL**（Tensor Compiler via Continual Learning），一个高效、可转移的编译器框架，包含三个核心组件：

#### ✅ **RDU Sampler**（Data-Efficient Active Learning）
- **思想**：通过联合优化 **Representativeness（代表性）、Diversity（多样性）、Uncertainty（不确定性）** 来主动选择最具信息量的 tensor programs 进行性能测量。
- **优势**：仅用 **10% 的数据** 即可达到接近全量数据训练的模型精度，大幅降低数据采集成本。

#### ✅ **Mamba-based Cost Model**（Efficient Sequence Modeling）
- **架构**：采用 **Mamba Block** 替代传统的 Transformer 或 LSTM。
- **优势**：
  - 利用 **Structured State Space Model (SSM)** 实现 $O(n)$ 时间复杂度，显著优于 Transformer 的 $O(n^2)$。
  - 支持输入感知的动态过滤机制，更有效地识别影响延迟的关键 schedule primitives。
  - 参数更少、训练更快、预测更准。

#### ✅ **Continual Knowledge Distillation (CKD) Framework**（Scalable Cross-Hardware Transfer）
- **设计**：由共享的 **Hardware Knowledge Base (KB)** 和可插拔的 **Hardware Active Column (AC)** 组成。
- **流程**：
  - 在源设备上训练后，通过 **Knowledge Distillation (KD)** 将知识存入 KB；
  - 在目标设备上训练 AC 时，复用 KB 中的知识；
  - 最终将 AC 学到的新知识蒸馏回 KB，实现渐进式知识积累。
- **优势**：
  - 避免 multi-task learning 的参数膨胀问题（模型大小恒定 ~0.7MB）；
  - 不需要同时访问多个平台的数据；
  - 支持从多个源设备持续学习，提升跨平台适应能力。

---

## 2. 核心实验方法和设置

### 数据集
- 基于开源 **Tenset dataset**，并补充自采数据。
- 包含两个自建平台上的大规模数据：
  - **Intel i7-12700F CPU**
  - **NVIDIA GeForce RTX 3080Ti GPU**
- 每个平台约 **8.4M~8.6M 样本**，涵盖多种主流 DNN 模型（如 ResNet, MobileNet, BERT, DCGAN 等）。
- 测试集包含五个代表性模型：ResNet-50, MobileNet-V2, ResNeXt-50, BERT-tiny, BERT-base。

### 实验设置
- **训练环境**：
  - 服务器配置：Intel Xeon Silver 4214 + 256GB RAM + 4×V100 GPU
  - 框架：PyTorch 1.7.0, CUDA 10.1
- **调优集成环境**：
  - CPU: i7-12700F
  - GPU: RTX 3080Ti
  - 集成至 **TVM v0.8.0**

### 评估指标
#### ✅ Dataset-based Evaluation
- **Top-k Score**：衡量 cost model 排序高性价比 tensor programs 的能力，越高越好。
  $$
  \text{Top-k} = \frac{\sum_{m,s} \text{min\_latency}_{m,s} \cdot \text{weight}_{m,s}}{\sum_{m,s} \sum_{i=1}^{k} \text{p\_latency}_{m,s,i} \cdot \text{weight}_{m,s}}
  $$

#### ✅ End-to-end Evaluation
- **Tuning Time**：达到相同推理延迟所需的自动调优轮次（Trials）数量，越少越好。
- **Inference Latency**：最终优化后的模型在目标硬件上的实际推理延迟，越低越好。

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Tenset-MLP** | Fully-supervised | 使用 MLP 作为 cost model，依赖大量离线数据 |
| **MTL-TLP** | Multi-task Learning | 同时学习多个平台任务，参数随平台数增长 |
| **Ansor** | Zero-shot | 无预训练 cost model，在线搜索 |
| **Felix** | Few-shot | 基于梯度下降的轻量级在线学习，仅支持 GPU |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（vs. Tenset-MLP）

| 指标 | CPU 平台 | GPU 平台 |
|------|--------|--------|
| **平均调优加速比**（Tuning Speedup） | **16.8×** | **12.48×** |
| **推理延迟降低倍数**（Lower Latency） | **1.20×** | **1.13×** |

> 即 TCL 只需 Tenset-MLP **1/16.8 的时间**即可达到其最终性能，并且还能进一步将延迟降低 **20%**（CPU）和 **13%**（GPU）。

### 与其他方法对比（Table 6 & Fig. 10–13）

| 方法 | Top-1 Score (i7-12700F) | Top-1 Score (3080Ti) |
|------|------------------------|-----------------------|
| Fine-tuning | 0.8378 | 0.7918 |
| MTL-TLP | 0.8744 | 0.8228 |
| **TCL (Ours)** | **0.8938** | **0.8426** |

- TCL 在 **CPU 和 GPU 上均超越所有 baseline**。
- 相比 MTL-TLP，TCL 在不增加参数规模的前提下实现了更高的 Top-1 准确率。

### 消融实验结果（Table 7）

| 模块组合 | i7-12700F Top-1 | 3080Ti Top-1 |
|----------|------------------|-------------|
| Baseline (None) | 0.7414 | 0.6614 |
| + RDU Sampler | 0.9020 | 0.8349 |
| + Mamba Model | 0.9116 | 0.8533 |
| **Full TCL (All)** | **0.9319** | **0.8675** |

- 所有三个模块均有显著增益，**组合使用效果最佳**。
- RDU Sampler 贡献最大，说明数据质量对性能至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **数据效率是关键瓶颈**：传统方法依赖海量标注数据，而 RDU Sampler 证明只需 **10% 高价值样本**即可实现优越性能。
2. **Mamba 架构优于 Transformer/LSTM**：在保持 $O(n)$ 复杂度的同时，准确捕捉 schedule 序列中的长期依赖关系。
3. **CKD 实现可持续知识积累**：相比 multi-task learning，CKD 框架更具可扩展性和实用性，适合部署在资源受限场景。
4. **同厂商硬件间迁移效果更好**：实验表明 Intel CPU 之间、NVIDIA GPU 之间的知识迁移正向增强，而跨厂商可能引入干扰。

### 局限性
- 当前未在 **CPU 和 GPU 之间进行跨架构迁移**（文中明确指出不做此尝试）。
- 对 **异构程度高的平台**（如 ARM vs x86），知识迁移仍可能存在负迁移风险。
- Mamba 模型对超参较敏感，需谨慎调整 `dconv`, `expand` 等参数（见 Table 2）。

### 未来工作方向
- 设计更细粒度的 **cost model** 以进一步提升预测精度。
- 探索 **跨架构（如 CPU↔GPU）的知识迁移机制**。
- 将 TCL 扩展至更多硬件类型（如 TPU、NPU、FPGA）。
- 结合 **Meta-Learning** 或 **Few-shot Learning** 进一步减少冷启动开销。

---

> 📌 **总结一句话**：  
> TCL 通过 **RDU 主动采样 + Mamba 高效建模 + CKD 渐进蒸馏**，构建了一个 **快速、轻量、可扩展** 的跨硬件 tensor program 优化框架，在仅用 10% 数据的情况下，实现了高达 **16.8× 的调优加速** 和 **1.2× 的延迟降低**，为下一代 DL 编译器提供了新的范式。

</details>

---

### 11. [Think Through Uncertainty: Improving Long-Form Generation Factuality via Reasoning Calibration](https://arxiv.org/abs/2604.12046)

**Authors**: Xin Liu, Lu Wang  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.12046v1  

#### Abstract
Large language models (LLMs) often hallucinate in long-form generation. Existing approaches mainly improve factuality through post-hoc revision or reinforcement learning (RL) with correctness-based rewards, but they do not teach the model to estimate which parts of its generation are reliable. As a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Think Through Uncertainty: Improving Long-Form Generation Factuality via Reasoning Calibration**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLMs）在长文本生成中普遍存在**幻觉（hallucination）**问题，即模型会以高置信度输出错误事实。现有方法如**post-hoc revision**（事后修正）或基于正确性的**Reinforcement Learning (RL)** 虽能提升事实准确性，但未能教会模型对自身不确定性进行建模，导致其仍可能“自信地犯错”。

此外，尽管已有研究将**calibration**（校准）引入 RL，但这些方法通常只提供整个响应的**单一标量置信度**，无法捕捉长文本中不同声明（claim）之间的不确定性差异。

---

### **提出了什么新方法或新思路**
本文提出 **CURE**（Claim-level Uncertainty-aware Query REasoning），一个通过**推理校准**来提升长文本生成事实性的框架，其核心创新在于：

- **Claim-aware reasoning protocol**：  
  将模型输出结构化为原子化的、可独立验证的**声明（claim）**，并为每个声明显式附上**置信度估计**（confidence estimate）。  
  输出格式分为两个阶段：
  - `<think>`：模型进行带置信度推理的过程。
  - `<decompose>`：将推理分解为 `(claim, confidence)` 对。

- **多阶段训练流水线（multi-stage training pipeline）**：  
  明确解耦 **calibration**（置信度校准）与 **factuality optimization**（事实性优化），避免联合优化中的干扰：
  1. **Stage 1: Feasibility Induction**  
     通过监督微调（SFT）和 Group Relative Policy Optimization (GRPO) 建立一个**可行的推理空间**，确保输出格式正确、相关、可验证且忠实于推理过程。
  2. **Stage 2: Calibration Optimization**  
     使用 **Direct Preference Optimization (DPO)** 构造偏好对，使模型的置信度预测与实际正确性对齐。
  3. **Stage 3: Factuality Optimization**  
     在保持校准的前提下，使用 GRPO 和**token-masked rewards** 优化声明级事实准确性。

- **推理时的选择性预测（selective prediction）**：  
  利用校准后的置信度，在推理时通过设定阈值 `t` 过滤低置信度声明，仅保留高可信声明，从而提升输出的整体可靠性。

---

### **相比现有方法的优势**
| 维度 | CURE | 现有方法 |
|------|------|----------|
| **不确定性建模粒度** | **声明级（claim-level）** | 全局或响应级 |
| **校准与事实性关系** | **明确解耦**，避免相互干扰 | 通常联合优化，易导致过拟合 |
| **优化机制** | DPO 用于校准，GRPO 用于事实性 | 多用 GRPO 联合优化 |
| **推理控制** | 支持**选择性预测**，动态权衡准确率与召回率 | 不支持或依赖后处理 |

> ✅ **优势总结**：CURE 首次实现了**细粒度、内在一致的声明级不确定性建模**，不仅提升了事实准确性，还增强了模型透明性和可控性。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
在四个长文本事实性基准上进行评估：

| 数据集 | 描述 |
|--------|------|
| **FactBench** (Bayat et al., 2025) | 包含易产生幻觉的查询，用于压力测试事实鲁棒性 |
| **LongFact** (Wei et al., 2024) | 开放式问题，需生成多段落详细响应 |
| **Biography** (Min et al., 2023) | 要求生成人物生平描述，强调准确性和完整性 |
| **FactRBench** (Liu et al., 2025) | 提供参考声明集，可用于评估**事实召回率（factual recall）** |

---

### **实验设置和评估指标**

#### **模型架构**
- 基座模型：`Llama3.1-8B-Instruct` 和 `Qwen3-4B`
- 训练流程遵循 Chen et al. (2025) 的 SFT 和 RL 提示模板

#### **评估指标**
| 指标 | 含义 |
|------|------|
| **Acc. (Accuracy)** | 声明级事实准确性（使用 VeriScore 验证） |
| **ECE (Expected Calibration Error)** | 校准误差，越低越好 |
| **Brier Score** | 概率预测误差，越低越好 |
| **AUROC** | 区分正确与错误声明的能力，越高越好（本文更重视此指标） |
| **Recall** | 在 FactRBench 上评估模型覆盖真实事实的能力 |

---

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Base LLM** | 未经后训练的原始模型 |
| **L2RF** (Chen et al., 2025) | 强大的 RL 基线，结合事实精度、细节和相关性奖励 |
| **LitCab** (Liu et al., 2024) | 轻量级校准方法，基于 token 概率调整置信度 |
| **SFT** | 监督微调基线 |
| **SFT + F-RL + CO** | CURE 的中间变体，不含最终事实性优化 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（Table 1）**
在 `Llama3.1-8B-Instruct` 上的结果：

| 方法 | FactBench Acc. | LongFact Acc. | Biography Acc. | FactBench AUROC |
|------|----------------|---------------|----------------|----------------|
| **L2RF** | 77.1 | 79.4 | 49.2 | — |
| **SFT** | 82.0 | 85.1 | 51.5 | 0.563 |
| **SFT + F-RL + CO** | 82.5 | 89.8 | 64.7 | 0.648 |
| **CURE (Ours)** | **84.4** | **90.2** | **65.9** | **0.667** |

> 🔺 **相对提升**：
> - 在 **Biography** 上提升高达 **39.9%**（从 L2RF 的 49.2 → 65.9）
> - 在 **FactBench** 上 AUROC 提升 **16.0%**（从 SFT 的 0.563 → CURE 的 0.667）

---

### **与基线方法的对比结果**
- CURE 在所有数据集上均取得**最高事实准确性**和**最佳校准质量**（尤其 AUROC）。
- 相比 L2RF，CURE 不仅提升准确率，还能维持甚至提升**事实召回率**（见 Figure 5），说明其未以牺牲覆盖率换取精度。
- 在 `Qwen3-4B` 上也取得一致提升（Table 3），表明方法具有良好的**泛化性**。

---

### **消融实验结果（Table 2）**
验证了多阶段设计的有效性：

| 方法 | Acc. | AUROC |
|------|------|-------|
| **SFT + F-RL** | 78.1 | 0.605 |
| **+ Joint GRPO (n=16)** | 83.9 | 0.620 |
| **+ CO (DPO)** | 82.5 | 0.648 |
| **+ FO (CURE)** | **84.4** | **0.667** |

> 🔍 **关键发现**：
> - **联合优化（Joint GRPO）** 虽能提升准确率，但**校准能力提升有限**（AUROC 仅微增）。
> - 使用 **DPO 进行校准** 比 GRPO 更有效，显著提升 AUROC。
> - **解耦设计** 是成功关键：先校准再优化事实性，避免了“越优化越自信”的退化现象。

---

## 4. **关键结论和发现**

### **主要发现**
1. **声明级不确定性建模至关重要**：  
   全局置信度不足以应对长文本中复杂的事实分布，**细粒度校准**是提升事实性的关键。

2. **校准与事实性应解耦优化**：  
   联合优化会导致模型通过“全给高置信”来最大化奖励，破坏校准。**分阶段训练**能有效避免这一问题。

3. **DPO 比 GRPO 更适合校准任务**：  
   GRPO 的 rollout-level 优势估计受内容变化主导，难以捕捉细粒度校准信号；而 DPO 可构造干净的偏好对，专注于置信度对齐。

4. **选择性预测增强系统可靠性**：  
   推理时通过置信度阈值过滤，可在不重新训练的情况下动态控制输出精度，提升用户信任。

---

### **方法的局限性**
- **依赖结构化输出格式**：需要模型严格遵循 `<think>` 和 `<decompose>` 格式，对格式鲁棒性有一定要求。
- **人工标注成本较高**：虽然使用 LLM 自动修正，但仍需高质量提示和验证流程。
- **校准依赖外部验证器**：VeriScore 等工具本身可能存在误差，影响校准信号质量。

---

### **未来工作方向**
- 将 CURE 扩展到更多任务，如问答、摘要、对话等。
- 探索**无需外部验证器的自监督校准**方法。
- 研究如何让模型在**生成过程中主动识别不确定性来源**（如知识缺失 vs 表述模糊）。
- 结合**检索增强生成（RAG）**，实现动态知识获取与不确定性表达的协同。

---

> ✅ **总体评价**：  
> CURE 提出了一种**系统性解决长文本幻觉问题的新范式**——不是简单地“纠正错误”，而是教会模型“知道自己不知道”。该方法在多个维度上超越现有技术，为构建**可信赖、可解释、可控的 LLM** 提供了重要路径。

</details>

---

### 12. [Latent-Condensed Transformer for Efficient Long Context Modeling](https://arxiv.org/abs/2604.12452)

**Authors**: Zeng You, Yaofo Chen, Qiuwu Chen, Ying Sun, Shuhai Zhang, Yingjian Li, Yaowei Wang, Mingkui Tan  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.12452v1  

#### Abstract
Large language models (LLMs) face significant challenges in processing long contexts due to the linear growth of the key-value (KV) cache and quadratic complexity of self-attention. Existing approaches address these bottlenecks separately: Multi-head Latent Attention (MLA) reduces the KV cache by pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题**: *Latent-Condensed Transformer for Efficient Long Context Modeling*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在处理长上下文时面临两大瓶颈：
- **KV Cache 的线性增长**：解码过程中需要缓存所有 token 的 key 和 value，导致内存占用随序列长度线性上升。
- **Self-Attention 的二次计算复杂度**：注意力机制的时间复杂度为 $O(L^2)$，限制了超长序列的推理效率。

现有方法通常分别解决这两个问题：
- Multi-head Latent Attention (MLA) 通过将 token 投影到低维潜在空间来压缩 KV Cache。
- Sparse Attention 方法减少注意力计算量，但需在原始高维空间操作。

然而，这些方法无法协同优化：稀疏化不能直接作用于 MLA 的压缩表示，必须先重建完整 KV 状态，从而丧失了进一步压缩的空间。

### 提出了什么新方法或新思路
本文提出 **Latent-Condensed Attention (LCA)** ——一种可在 MLA 的**潜在空间内原生执行上下文压缩**的高效注意力机制。

#### 核心设计思想：
- 在 MLA 的 latent space 中对历史上下文进行分组并压缩为代表性向量。
- 对语义信息（`CKV`）和位置信息（`K'`）采用**解耦策略**：
  - **Semantic Component (`CKV`)**：使用 query-aware weighted pooling 进行加权聚合，保留重要语义。
  - **Positional Component (`K'`)**：通过 max-selection 选择每组中重要性最高的 token 作为 anchor，保持精确的位置信号。

此外，保留最近 `w` 个 token 的完整信息以保障局部细节建模能力。

### 相比现有方法的优势
| 维度 | LCA | 其他方法（如 FlexPrefill, MInference） |
|------|-----|-------------------------------|
| 是否引入额外参数 | ❌ 否 | 通常不引入 |
| 是否能直接在 latent space 工作 | ✅ 是 | ❌ 必须重建 full-dimensional KV |
| 是否同时降低计算与内存开销 | ✅ 是 | ⚠️ 多数仅降低计算或缓存之一 |
| 是否保持 competitive 性能 | ✅ 是 | ❌ 动态稀疏常丢弃关键信息导致性能下降 |

> ✅ **优势总结**：LCA 是首个能在 MLA 的 latent space 内实现“原生压缩”的方法，实现了 **KV Cache 减少 + Attention 计算降低 + 无性能损失** 的三重收益。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Long-context 任务**：
  - **LongBench-E** (Bai et al., 2023)：多语言、多任务长文本理解基准，涵盖问答、摘要、代码等 21 项任务，支持至 128K 上下文。
  - **RULER** (Hsieh et al., 2024)：合成型长上下文评测集，测试检索、多跳推理等能力，精细评估不同长度下的表现。
- **Short-context 任务**（验证通用性）：
  - **MMLU**：知识广度与推理。
  - **GSM8K**：小学数学题，需多步推理。
  - **MBPP**：编程生成任务。

### 实验设置和评估指标
| 设置项 | 描述 |
|--------|------|
| 模型基础 | DeepSeek-V2-Lite (16B)，基于 MLA 架构 |
| 替换模块 | 将原 MLA 层替换为 LCA |
| 实现方式 | 使用 Triton 编写高度优化 kernel，提升运行效率 |
| 微调设置 | 在 SlimPajama 数据集上继续微调 1000 步，序列长度 32K |
| 超参设置 | group size $g=16$，local window $w=1024$ |
| 硬件平台 | 8×H200 GPUs |

#### 评估指标：
- **性能指标**：各 benchmark 的准确率 / 分数（如 LongBench avg. score）
- **效率指标**：
  - Prefilling 阶段延迟（First-Token Latency, FTL）
  - Decoding 阶段每 token 延迟（Interval-token Latency）
  - KV Cache 显存占用（GPU Memory Footprint）

### 基线方法对比
| 方法 | 类型 | 是否适配 MLA | 主要机制 |
|------|------|---------------|----------|
| **DeepSeek-V2-Lite (MLA)** | 原始模型 | ✅ | Latent Attention，压缩 KV |
| **MInference** (Jiang et al., 2024) | 动态稀疏 | ✅（重建后应用） | 动态跳过无关 attention block |
| **FlexPrefill** (Lai et al., 2025) | 动态稀疏 | ✅（重建后应用） | 自适应稀疏模式选择 |
| **KDA** (Kimi et al., 2025b) | Gated Linear Attention | ✅（微调集成） | 控制记忆衰减与位置感知 |

> 所有对比方法均被适配用于 MLA 输出的重建 KV 上，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📊 表格汇总：在 **LongBench-E @64K** 上的表现
| 方法 | 平均得分 | FTL (s) | 加速比 |
|------|---------|--------|-------|
| MLA (原版) | 29.51 | 3.20 | 1.0× |
| MInference | 19.71 | 2.99 | 1.1× |
| FlexPrefill | 21.05 | 2.51 | 1.3× |
| KDA | 27.15 | 2.47 | 1.3× |
| **LCA (ours)** | **29.09** | **1.80** | **1.8×** |

✅ **结论**：LCA 几乎保持原始性能（仅降 0.42），同时获得 **1.8× 预填充加速**。

#### 📈 在 **RULER @128K** 上的结果
| 方法 | 得分 | FTL (s) | 加速比 |
|------|-----|--------|-------|
| MLA | 23.96 | 10.78 | 1.0× |
| MInference | 4.34 | 5.66 | 1.9× |
| FlexPrefill | 7.19 | 5.38 | 2.0× |
| KDA | 22.22 | 4.96 | 2.2× |
| **LCA (ours)** | **24.38** | **4.40** | **2.5×** |

✅ **结论**：LCA 不仅未退化，反而**小幅超越 MLA**，且达到 **2.5× 预填充加速**。

#### 💾 KV Cache 压缩效果（@128K）
| 方法 | KV Cache 占用 | 压缩率 |
|------|----------------|--------|
| MLA | ~10.13 GB | — |
| **LCA (ours)** | **~0.71 GB** | **93%↓** |

> 🔽 达到 **90% 以上的 KV Cache 减少**

#### ⏱️ 解码阶段效率
- **Per-token latency**：在 128K 上降低 **1.8×**
- **Cache size during decoding**：稳定维持在 $O(m + w)$，显著低于 $O(L)$

---

### 消融实验结果（Ablation Studies）

#### ✅ 不同 pooling 策略组合对比（Table 5）
| Semantic Pooling \ Positional Selection | MaxPool | MeanPool | **WeightedPool** | MaxSel |
|----------------------------------------|--------|-----------|------------------|--------|
| MaxPool                                | 27.44  | 27.01     | 27.43            | 28.89  |
| MeanPool                               | 27.38  | 26.39     | 27.67            | 28.84  |
| **WeightedPool**                       | 27.93  | 27.04     | **27.83**        | **29.09** |
| MaxSel                                 | 26.94  | 27.65     | 27.79            | 28.93  |

➡️ 最佳组合：**Weighted Pooling + Max Selection** → 验证了解耦设计的有效性。

#### ✅ Group Size $g$ 影响（Figure 4）
- $g=4$: 更高精度但更高延迟
- $g=16$: 默认设定，在性能与效率间取得良好平衡

#### ✅ Window Size $w$ 影响（Figure 5）
- $w=512$: 性能略低
- $w=1024$: 推荐默认值
- $w=2048$: 性能更好但成本上升

#### ✅ Summary Query 数量影响（Table 8）
- 使用最后 16 个 query 效果最佳，过多或过少均无明显增益。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LCA 可在 latent space 内实现高效压缩**，无需重建 full-dimensional KV。
2. ✅ **解耦语义与位置处理策略有效**：weighted pooling 聚合语义，max-selection 保护位置信号。
3. ✅ **理论保证误差有界**：Theorem 1 证明近似误差与上下文长度无关，仅依赖于组内偏差。
4. ✅ **实测性能几乎无损**：在 LongBench 和 RULER 上接近甚至超过原始 MLA。
5. ✅ **效率大幅提升**：
   - **预填充速度最高提升 2.5×**
   - **KV Cache 减少达 90% 以上**
   - 支持扩展至 1M token 序列（Triton kernel 实现）

### 方法的局限性
1. **依赖定制 kernel 实现高效**：当前高性能依赖 Triton 编写的底层 kernel，部署门槛较高。
2. **对极细粒度检索任务略有退化**：在 multi-document QA 等需精确定位的任务上存在轻微性能下降（如从 11.15 → 8.90），因压缩可能削弱集中信号。
3. **未探索低精度量化兼容性**：目前仅在 bfloat16/float16 下验证，尚未支持 int8 等量化格式。
4. **超参数需手动调节**：group size $g$ 和 window size $w$ 影响性能，缺乏自适应机制。

### 未来工作方向
1. 设计 **adaptive group/window sizing** 机制，根据输入动态调整压缩强度。
2. 探索 LCA 与 **quantization、pruning** 等其他压缩技术的联合优化。
3. 将 LCA 应用于 **training from scratch** 场景，探索其在训练阶段的潜力（已有初步实验显示收敛更快）。
4. 开发更通用的 **framework-level support**，降低 Triton 等底层实现的工程负担。
5. 扩展至更多 attention variant（如 MQA, ALiBi），验证其 architecture-agnostic 特性。

---

> ✅ **总体评价**：LCA 是一个简洁而强大的设计，首次打通了“latent compression”与“efficient attention”的壁垒，为构建真正高效的超长上下文 LLM 提供了一条可行路径。其“**decoupled condensation**”思想具有广泛启发意义。

</details>

---

### 13. [Predictive Bayesian Arbitration: A Scalable Noisy-OR Model with Service Criticality Awareness](https://arxiv.org/abs/2604.11989)

**Authors**: Anil Jangam, Ganesh Karthick Rajendran, Roy Kantharajah  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.11989v1  

#### Abstract
Geographically High-Available (Geo-HA) cluster systems are essential for service continuity in distributed cloud-native environments. However, traditional arbitration mechanisms, which are often predicated on deterministic node-level heartbeats, are resource-intensive and inherently reactive. This n...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Predictive Bayesian Arbitration: A Scalable Noisy-OR Model with Service Criticality Awareness*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Geographically High-Available (Geo-HA)** 集群系统的仲裁机制存在以下关键缺陷：
- **资源密集**：每个部署需独立的专用仲裁器（dedicated arbiter），导致基础设施开销随规模线性增长。
- **反应式（Reactive）决策**：依赖确定性心跳检测，仅在故障发生后触发切换（switchover），造成不可避免的停机时间。
- **缺乏预测能力**：无法识别服务间的级联故障（cascade failure）模式，导致“durability gap”——即在性能与数据持久性之间被迫权衡。

### 🚀 提出的新方法与创新思路
本文提出一种**基于贝叶斯网络的预测性仲裁框架（Predictive Bayesian Arbitration Framework）**，核心创新包括：

#### （1）共享微服务架构（Shared Microservices Architecture）
- 引入 **Shared Arbitration Service (SAS)**，支持多域、多集群的集中管理。
- 通过 **Arbiter Persona 多路复用** 实现高密度资源利用，显著降低总体资源占用。

#### （2）自适应在线学习机制（Adaptive Online Learning）
- 基于 **Bayesian Noisy-OR 模型** 构建动态概率推理引擎。
- 自动从运行时故障数据中发现 **Temporal Cascade Dependencies**（如 Service A → Service B）。
- 利用 **专家先验（expert-informed priors）** 克服冷启动问题，并通过运行时反馈持续优化模型。

#### （3）服务关键性感知（Service Criticality Awareness）
- 引入 **Critical Service Groups (CSGs)** 概念，结合 Kubernetes 中的 `CSG_LABEL` 进行监控。
- 将应用层健康状态（如数据库复制延迟 replication lag）纳入仲裁逻辑，提升上下文感知能力。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | 本论文方法 |
|------|--------|-----------|
| 架构可扩展性 | 每部署一个专用仲裁器，资源开销大 | 共享 SAS 支持数百个域，资源利用率高 |
| 决策模式 | 反应式（failure-after） | **预测式（pre-failure）**，实现主动切换 |
| 模型适应性 | 静态 CPTs，需手动配置 | 动态学习 CPTs，自动发现未知依赖 |
| 可解释性 | 黑箱模型（如 DL/LSTM） | 贝叶斯模型提供清晰因果路径 |
| 计算复杂度 | 高（如深度学习） | **O(n)** 线性复杂度，适合实时系统 |

---

## 2. 核心实验方法和设置

### 📊 数据来源与环境
- 实验基于**模拟的分布式云原生环境**，包含多个 Geo-HA 集群。
- 故障场景由仿真生成，涵盖多种典型事件序列：
  - 正常运行（Normal Operation）
  - 已确认故障（Confirmed Malfunctions）
  - 误报（False Alarms）
  - 用户触发切换（User-triggered Switchovers）
  - 时间级联故障（Temporal Cascades）

> 注：未使用公开真实日志数据集，而是基于生产系统行为建模生成训练与测试轨迹。

### ⚙️ 实验设置
- **评估周期**：跨三个连续故障事件（Event 1–3）进行纵向分析。
- **监控层级**：四层遥测数据融合
  - Node-level health
  - Network path dynamics
  - Application-tier performance（如 replication lag）
  - CSG-level metrics

### 🎯 评估指标
| 指标 | 定义 |
|------|------|
| **Mean Time to Failure Detection (MTTFD)** | 从退化开始到被检测出的时间 |
| **Total Switchover Time** | 从退化开始至备用集群完全接管的总耗时 |
| **Predictive Lead Time** | 在实际故障前启动切换的时间提前量 |
| **Computational Complexity** | 推理过程为 O(n)，确保可扩展性 |

### 🆚 基线方法对比
共比较四种仲裁策略：
1. **Reactive (15s)**：心跳超时15秒
2. **Reactive (5s)**：心跳超时5秒
3. **Static Bayesian**：静态贝叶斯模型，固定 CPTs
4. **Adaptive Bayesian**（本文方法）：动态学习 CPTs 的 Noisy-OR 模型

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table I 与 Fig. 2–5）

| 方法 | 检测时间 (s) | 总切换时间 (s) | 相对改进 |
|------|-------------|----------------|----------|
| Reactive (15s) | +15 | 45 | Baseline |
| Reactive (5s) | +5 | 35 | +22.2% |
| Static Bayesian | -5 | 25 | +44.4% |
| **Adaptive Bayesian** | **-20** | **10** | **+77.8%** |

> ✅ “负值”表示**预测性提前**，即在故障发生前就开始执行切换。

### 🔬 核心发现
- **60% MTTFD 下降**：相比传统反应式方法，平均故障检测时间减少 60%。
- **77.8% 总切换时间缩短**：得益于预测性执行，总停机时间降至仅 10 秒。
- **学习曲线明显**：随着观察到更多级联事件（如 Event 1 → Event 3），模型准确率快速上升，无需人工干预。
- **O(n) 复杂度验证**：推理效率保持线性增长，适用于大规模微服务架构。

### 🔁 消融实验（隐含分析）
虽然未明确列出消融实验表格，但从图示可推断：
- **无自适应学习（Static Bayesian）**：无法捕捉新兴依赖关系，在 Event 3 表现停滞。
- **无级联检测机制**：将失败视为孤立事件，错过早期预警信号。
- **无 hysteresis 控制**：易出现 switchover flapping（震荡切换）。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **预测优于反应**：通过引入 **Bayesian Noisy-OR + 自适应学习**，系统可在硬故障发生前发起切换，有效打破“durability gap”。
2. **资源共享可行且高效**：SAS 架构证明单一仲裁服务可安全托管多个租户域，兼顾隔离性与成本效益。
3. **自动化优于人工建模**：系统能自主发现隐藏的服务依赖（如共享瓶颈、超时链式传播），超越“Day 0”专家图谱。
4. **轻量级模型也能高性能**：相比 DL 方法，该框架以极低计算开销实现了更高可解释性和实用性。

### ⚠️ 局限性
- **依赖历史故障数据积累**：尽管有专家先验缓解冷启动，但在极少故障场景下初期表现受限。
- **因果推断仍具挑战**：当前级联检测基于时间相关性（temporal correlation），尚未完全解决因果混淆问题。
- **边缘环境适配待优化**：SAS 当前设计面向中心云，对 **edge computing** 场景的支持仍在规划中。

### 🔮 未来工作方向（Future Work）
- **自动拓扑发现**：采用 **Causal Structure Discovery** 算法从多元时间序列中提取 CSG 依赖图。
- **联邦自适应学习（Federated Adaptive Learning）**：跨多个 Geo-HA 部署共享知识，同时保证数据本地性。
- **强化学习调参**：引入 **Reinforcement Learning (RL)** 动态调整代价敏感阈值（cost-sensitive thresholds）。
- **长时序预测增强**：集成 **RNN / Transformer** 提升长期趋势预测能力。
- **可解释 AI 接口（XAI）**：为运维人员提供 human-readable 的切换理由报告。
- **边缘轻量化部署**：优化 SAS 以适应资源受限、网络不稳定的边缘节点。

---

## 总结
本文提出的 **Predictive Bayesian Arbitration** 框架代表了 Geo-HA 系统从“被动响应”向“主动预防”的范式转变。它不仅在性能上大幅超越传统方法（**77.8% 切换加速**），更通过 **共享架构 + 自学习贝叶斯模型** 实现了可扩展、可解释、可持续进化的智能仲裁体系，为现代云原生系统的高可用设计提供了新范例。

</details>

---

### 14. [A Periodic Space of Distributed Computing: Vision & Framework](https://arxiv.org/abs/2604.12259)

**Authors**: Mohsen Amini Salehi, Adel N. Tousi, Hai Duc Nguyen, Murtaza Rangwala, Omar Rana, Tevfik Kosar, Valeria Cardellini, Rajkumar Buyya  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.12259v1  

#### Abstract
Advances in networking and computing technologies throughout the early decades of the 21st century have transformed long-standing dreams of pervasive communication and computation into reality. These technologies now form a rapidly evolving and increasingly complex global infrastructure that will un...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Periodic Space of Distributed Computing: Vision & Framework

## 1. 论文的主要贡献和创新点

### 解决的问题
随着分布式计算系统在规模、复杂性和异构性上的迅速增长，现有的分类方式（如基于架构或服务模型的 taxonomies）已难以提供统一、可扩展且具有预测能力的视角来理解整个生态系统。该论文旨在解决以下核心问题：
- 如何系统化地组织日益复杂的分布式计算技术栈？
- 如何揭示不同解决方案之间的设计权衡（trade-offs）和行为模式？
- 如何预测未来技术演进的方向？

### 提出的新方法与新思路
作者提出了一种名为 **“Periodic Space”**（周期空间）的框架，受化学元素周期表启发，构建了一个二维连续空间用于刻画分布式计算生态系统的结构：

- **横向维度（Tiers Continuum）**：从本地设备（Local Device）到边缘（Edge）、雾层（Fog）、云（Cloud），直至“天空层”（Sky），代表物理部署层级的延展。
- **纵向维度（Abstraction Levels）**：从底层硬件抽象逐步上升至高层编程模型和服务，最终达到 **Agent 抽象层**，即由自主或半自主的 agentic AI 驱动的服务。

这一框架不仅对现有系统进行定位，还能通过趋势外推预测新兴技术和系统的行为特征。

### 相比现有方法的优势
| 维度 | 传统方法局限 | 本论文优势 |
|------|--------------|------------|
| **结构性** | 多为离散分类，缺乏内在逻辑关联 | 具备连续、可扩展的空间结构，支持演化推理 |
| **解释力** | 描述性强但难以揭示共性规律 | 可识别跨层级的系统属性趋势（如 responsiveness、elasticity） |
| **预测性** | 仅适用于已有系统 | 支持对未来技术（如量子计算、太空数据中心）的位置与影响进行合理推测 |
| **指导性** | 缺乏设计原则引导 | 可辅助研究人员识别空白区域、优化设计选择 |

此外，作者还开发了交互式网页工具（[https://hpcclab.github.io/periodic-table](https://hpcclab.github.io/periodic-table)）以可视化系统属性在周期空间中的分布。

---

## 2. 核心实验方法和设置

> ⚠️ 注意：本文是一篇 **vision paper**（愿景型论文），并非以实证实验为主导的研究，因此没有传统意义上的“实验设置”、“数据集”或“基线对比”。

### 方法论性质
- 属于 **概念建模与理论分析** 类研究。
- 主要采用 **归纳法** 和 **专家洞察整合法**，结合文献综述与来自全球领先研究者的观点（如 Carnegie Mellon、Ohio State、VU Amsterdam 等机构反馈）。

### 分析手段
1. **系统属性映射**  
   将多个关键系统属性（System Properties）沿两个维度进行趋势标注：
   - **Simple Properties**：如 Responsiveness、Capacity、Elasticity
   - **Compound Properties**：如 Reliability、Governance、Sustainability

2. **趋势推演与案例说明**
   - 使用典型示例说明属性变化规律（例如容器从 Edge 迁移到 Cloud 导致延迟增加）
   - 结合当前技术发展路径（如 LLM、serverless、space-based computing）进行合理性论证

3. **专家共识整合**
   - 引入多位领域专家的观点作为支撑，增强框架的可信度与前瞻性

---

## 3. 主要实验结果和性能指标

> ❌ 无传统意义上的“实验结果”或数值性能指标（如准确率、吞吐量等）

### 关键观察与趋势发现（定性“结果”）
尽管无量化实验，论文提炼出若干重要趋势，这些是其核心“输出”：

#### 跨 Tier 的趋势（→ 从左到右：Local → Sky）
| 属性 | 趋势 |
|------|------|
| **Responsiveness** | ↓ 随 tier 上升而降低（距离终端更远） |
| **Capacity / Scalability** | ↑ 显著提升 |
| **Reliability** | ↑ 更高 tier 具备更强容错机制 |
| **Mobility Support** | ↓ 下降（Sky 层不适合移动节点） |
| **Infrastructure Cost (CapEx)** | ? 初期高，长期可能下降（尤其太空数据中心） |
| **AI-Native Support** | ↑ Sky 层将成为 AI 超级计算主力 |

#### 跨 Abstraction Level 的趋势（↑ 从下到上：Hardware → Agent）
| 属性 | 趋势 |
|------|------|
| **Democratization / Ease-of-Use** | ↑ 开发门槛显著下降 |
| **Controllability** | ↓ 对底层控制减弱 |
| **Elasticity** | ↑ 自动扩缩容能力增强 |
| **Security & Trustworthiness** | ? 存在矛盾：标准化提升安全，但 agent 自主性带来新风险 |
| **Operational Cost (OpEx)** | ↓ 平台托管降低运维负担 |

#### 新兴方向预测
- **Sky Tier** 将融合 multi-cloud、LEO satellites、quantum computing，形成 planetary-scale infrastructure
- **Agent Abstraction** 将成为下一代主流接口，开发者只需表达 intent，由 LLM 驱动的 agents 自动生成 workflow 并执行
- **Sustainability** 将不再是附加目标，而是核心设计原则，需实现 carbon-aware resource scheduling

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **分布式计算正走向“高阶抽象 + 高层 tier”的收敛区**  
   未来的创新将集中在 **Top-Right 区域** —— 即 Sky Tier 上运行的 Agent-based AI Services。

2. ✅ **系统属性呈现可预测的趋势**  
   在周期空间中，多数属性随坐标变化表现出一致的方向性，可用于指导系统设计。

3. ✅ **Failure 将成为常态而非异常**  
   在 Sky Tier 规模下，故障不可避免，系统必须默认“fail-operational”，依赖 resilience 而非 fault avoidance。

4. ✅ **Fluidity 是未来关键能力**  
   计算应像数据包一样自由迁移（vertical/horizontal fluidity），实现真正的 service liquidity。

5. ✅ **可持续性必须内生于系统设计**  
   能效优化不能仅靠硬件改进（Jevons’ Paradox），还需软件栈层面的 grid-aware 与 carbon-aware 控制。

6. ✅ **Agentic AI 带来便利的同时引入新型安全挑战**  
   如 prompt injection、tool misuse、privilege escalation 等，需要新的 security paradigms（如 runtime monitoring、sandboxed execution、explainable decision-making）。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **主观性较强** | 周期空间中系统的定位和属性趋势依赖作者判断，尚未建立形式化度量标准 |
| **缺乏量化验证** | 所有趋势均为定性描述，未通过大规模实证数据分析验证 |
| **动态演化机制不明确** | 框架能描述状态，但未建模“如何推动系统向某个方向演进”的动力学过程 |
| **文化/政策因素忽略** | 未考虑地缘政治、监管政策、数字主权等非技术因素的影响 |

### 未来工作方向
1. 🔄 **构建动态版本的 Periodic Space**  
   引入时间轴，追踪技术在空间中的移动轨迹，建立演化模型。

2. 📊 **开发自动化工具支持系统定位与趋势分析**  
   利用 NLP 技术自动提取论文/产品文档中的特征，并映射到周期空间。

3. 🔍 **开展大规模实证研究验证属性趋势**  
   收集真实系统（如 AWS Lambda、Azure Functions、Kubernetes clusters）的数据，检验 responsiveness、elasticity 等是否符合预期趋势。

4. 🛡️ **深化 Agent Abstraction 安全治理机制研究**  
   探索 multi-level value alignment、runtime behavioral auditing、fine-grained permission models。

5. 🌍 **推动碳感知（carbon-aware）调度算法落地**  
   在 Cloud-to-Sky 架构中实现基于可再生能源波动的任务调度。

6. 🚀 **探索 Extra-planetary Computing 的工程可行性**  
   与航天企业合作研究 lunar/LEO 数据中心的热管理、通信延迟与经济模型。

---

## 总结

该论文提出了一个极具前瞻性的 **“Periodic Space” 框架**，将纷繁复杂的分布式计算生态系统组织成一个具有解释力与预测力的概念空间。虽然缺乏传统实验验证，但其思想深刻、结构清晰，为未来十年的分布式系统研究提供了强有力的思维工具和战略指引。它不仅是分类器，更是**通往智能时代基础设施设计的导航图**。

</details>

---

### 15. [LLM-HYPER: Generative CTR Modeling for Cold-Start Ad Personalization via LLM-Based Hypernetworks](https://arxiv.org/abs/2604.12096)

**Authors**: Luyi Ma, Wanjia Sherry Zhang, Zezhong Fan, Shubham Thakur, Kai Zhao, Kehui Yao, Ayush Agarwal, Rahul Iyer, Jason Cho, Jianpeng Xu, Evren Korpeoglu, Sushant Kumar, Kannan Achan  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.12096v1  

#### Abstract
On online advertising platforms, newly introduced promotional ads face the cold-start problem, as they lack sufficient user feedback for model training. In this work, we propose LLM-HYPER, a novel framework that treats large language models (LLMs) as hypernetworks to directly generate the parameters...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM-HYPER: Generative CTR Modeling for Cold-Start Ad Personalization via LLM-Based Hypernetworks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在在线广告平台中，新上线的促销广告（cold-start ads）面临**冷启动问题**：由于缺乏足够的用户点击反馈（click-through feedback），传统基于监督学习的 CTR 预估模型难以有效训练，导致初期推荐效果差、影响用户体验和平台收入。

### 🚀 提出的新方法：LLM-HYPER
提出 **LLM-HYPER** —— 一种将 **Large Language Models (LLMs)** 作为 **hypernetworks** 来直接生成 CTR 预估模型参数的全新框架，实现无需训练标签的冷启动个性化。

#### 创新点：
- **Training-free 参数生成**：利用 LLM 对多模态广告内容（文本 + 图像）进行推理，直接生成线性 CTR 模型的 feature-wise 权重，绕过传统训练过程。
- **Few-shot Chain-of-Thought (CoT) Prompting**：通过 CLIP 检索语义相似的历史广告，构建包含图文内容和其已知权重的 few-shot 示例，引导 LLM 进行推理。
- **Decoupled 架构设计**：将耗时的 LLM 推理置于离线阶段，线上仅执行轻量级线性计算，满足工业级低延迟要求（p99 < 1ms）。
- **Normalization & Calibration**：引入归一化和截距校准机制，确保生成的权重数值稳定，并与生产环境中已有模型的概率分布对齐。

### 🔍 相比现有方法的优势
| 方法 | 局限性 | LLM-HYPER 的优势 |
|------|--------|------------------|
| 传统 LR 模型（如 LRcold） | 冷启动下依赖启发式权重（如中位数），表达能力弱 | 利用 LLM 强大的跨模态语义理解能力动态生成更合理的初始权重 |
| Embedding-based 方法（如 EmbT5） | 仅基于向量相似度，无法建模复杂特征交互 | 显式输出可解释的 feature weights，支持业务策略干预 |
| 端到端 LLM 推荐器（如 LLM-R / LLM-TR） | 实时调用 LLM 成本高、延迟大、易产生幻觉 | 将 LLM 用于离线“元生成”，解耦推理与服务，兼顾性能与效率 |

---

## 2. 核心实验方法和设置

### 📊 数据集描述
- 使用美国某头部电商平台的真实广告交互数据。
- 包含 **675 个 warm ads** 和 **1,000,000 用户** 的历史行为。
- 时间划分模拟冷启动场景：
  - **Retired ad set**（历史广告）：455 个早期广告及其训练好的权重（用于检索 few-shot 示例）
  - **Active ad set**（待测广告）：120 个近期广告，进一步划分为：
    - 训练集：800,000 用户 → 用于 warm-start 基线训练
    - 测试集：200,000 用户 → 用于评估

> ⚠️ 数据为专有数据，细节未公开。

### 🧪 实验设置与评估指标

#### 评估维度
| 类型 | 指标 | 说明 |
|------|------|------|
| **离线评估** | AUC, NDCG@5, NDCG@10 | 衡量排序质量 |
| | L (ms) | 平均 CTR 预测延迟，反映服务效率 |
| **在线 A/B 测试** | Relative CTR score | 相对于 warm-start 模型的点击率表现 |
| **可解释性** | HitRate@5 (HR@5), Coverage@5, Consistency Rate | 衡量生成权重是否符合人工标注的重要特征及逻辑一致性 |
| **鲁棒性** | Counterfactual Accuracy | 在对抗性语义扰动下的权重调整正确率 |

#### 基线方法对比
| 类别 | 方法 | 描述 |
|------|------|------|
| Warm-start | `LRwarm` | 在 active ad 上训练的传统线性模型（理想上界） |
| Cold-start | `LRcold` | 所有冷启广告使用历史广告权重的中位数（启发式） |
| | `EmbT5` | 基于 Sentence-T5 的语义匹配推荐 |
| | `LLM-R`, `LLM-TR` | 端到端 LLM 推荐系统（zero-shot ranking） |
| **LLM-HYPER (Ours)** | `LH2.5p`, `LH2.5F`, `LH4o`, `LH5.1` | 基于不同 backbone LLM 的变体（Gemini/GPT 系列），采用 5-shot CoT 提示 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（离线测试）

| Model | AUC | NDCG@5 | NDCG@10 | L (ms) |
|-------|-----|--------|---------|--------|
| `LRwarm` (upper bound) | 0.7005 | 0.0639 | 0.0871 | 0.151 |
| `LRcold` (best baseline) | 0.5593 | 0.0328 | 0.0508 | 0.147 |
| **`LH2.5p` (ours)** | **0.6722** | **0.0456** | **0.0792** | **0.157** |
| ↑ vs `LRcold` | +20.2% | +39.0% | **+55.9%** | ≈ |

> ✅ **LLM-HYPER 显著优于所有冷启动基线（p ≤ 0.05）**

#### 延迟表现
- 所有 LLM-HYPER 变体平均延迟在 **0.14–0.17 ms**，与传统线性模型相当。
- 而 `LLM-R` 和 `LLM-TR` 因需实时调用 LLM，延迟高达 **2–3 秒**，不适用于线上实时排序。

---

### 🔍 消融实验结果（Ablation Study）

#### (1) CoT 与视觉输入的影响（Table 2）

| Model | NDCG@5 | NDCG@10 |
|-------|--------|---------|
| `LH2.5p`, zero-shot | 0.0376 | 0.0677 |
| `LH2.5p`, zero-shot, -img | 0.0165 | 0.0352 |
| `LH2.5p`, 3-shot | 0.0421 | 0.0798 |
| `LH2.5p`, 5-shot | 0.0456 | 0.0792 |
| `LH2.5p`, 5-shot, -img | 0.0292 | 0.0506 |

> 🔎 发现：
- 即使是 **zero-shot + 多模态输入** 也能提升 NDCG@10 达 **33.3%**
- 加入 **图像信息** 至关重要，移除后性能大幅下降（↓ >37%）
- **3~5 个 few-shot 示例** 效果最佳，过多可能引入噪声

#### (2) 可解释性分析（Figure 3）
| Metric | Zero-shot | 3-shot | 5-shot |
|--------|-----------|--------|--------|
| HR@5 | 0.78 | **0.81** | 0.80 |
| Coverage@5 | 0.317 | 0.351 | 0.351 |
| Consistency Rate | >0.95 | >0.95 | >0.95 |

> ✅ CoT 示例显著提升与人类专家判断的一致性，且推理与数值输出高度一致。

#### (3) 鲁棒性测试（Counterfactual Robustness, Figure 4）
在三种语义扰动下评估权重变化方向是否合理：

| Setting | Accuracy |
|--------|----------|
| Enhanced | ~0.87 |
| Diminished | ~0.88 (GPT-5.1 最佳) |
| Neutralized | ~0.68 |

> ✅ LLM-HYPER 能够根据语义修改合理调整权重，表现出良好的因果推理能力和部署鲁棒性。

---

### 🌐 在线 A/B 测试结果（Table 3）

| Model | Relative CTR Score |
|-------|--------------------|
| `LRwarm` (control) | 100% |
| **`LLM-HYPER` (variant)** | **92%** |

> ✅ 在真实首页广告场景中运行 30 天，LLM-HYPER 达到 warm-start 模型 **92% 的 CTR 表现**，且差异无统计显著性（p=0.62），表明其实际效果极具竞争力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLMs 可作为 hypernetworks 直接生成高质量 CTR 模型参数**，无需任何训练标签，成功解决严格冷启动问题。
2. **多模态 few-shot CoT 提示** 是关键：结合 CLIP 检索 + 图文内容 + 历史权重示例，能有效引导 LLM 学习“如何打分”。
3. **解耦架构实现高效部署**：离线生成权重 + 线上轻量推理，完美平衡 LLM 的强大推理能力与工业系统的低延迟需求。
4. **生成结果具备高可解释性与可控性**：权重可审计、推理过程透明，满足电商等高风险场景的合规要求。
5. **已在 Walmart 生产环境成功落地**，应用于 Homepage Ads 冷启动排序，验证了工程可行性。

### ⚠️ 方法的局限性
- **依赖高质量 few-shot 示例**：若历史广告库不足或分布偏移，检索效果会下降。
- **LLM 生成成本较高**：虽然不影响线上延迟，但批量生成权重仍涉及可观的 LLM API 开销（见 Appendix A：Gemini-2.5-Pro 单广告生成耗时约 103 秒）。
- **提示工程敏感**：batch size、prompt design、temperature 等超参需精细调优。
- **仅适用于线性层权重生成**：当前聚焦于 linear CTR predictor，扩展至深层结构仍需研究。

### 🔮 未来工作方向
- 探索 **smaller 或 distilled LLMs** 替代通用大模型，降低成本。
- 引入 **feedback loop**，利用冷启动后的初步反馈迭代优化生成权重。
- 扩展至 **multi-task 或 deep model initialization**，例如生成 DeepFM 或 DIN 的部分参数。
- 结合 **diffusion-based hypernetworks** 或 **meta-learning** 进一步提升泛化能力。

---

> 💡 **一句话总结**：  
> **LLM-HYPER 开创性地将 LLM 视为 hypernetwork，通过多模态 CoT 提示离线生成可解释、高性能的 CTR 模型权重，在不解耦业务需求的前提下解决了工业级广告冷启动难题，并已成功落地生产。**

</details>

---

### 16. [Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization](https://arxiv.org/abs/2604.12290)

**Authors**: Yizhe Chi, Deyao Hong, Dapeng Jiang, Tianwei Luo, Kaisen Yang, Boshi Zhang, Zhe Cao, Xiaoyan Fan, Bingxiang He, Han Hao, Weiyang Jin, Dianqiao Lei, Qingle Liu, Houde Qian, Bowen Wang, Situ Wang, Youjie Zheng, Yifan Zhou, Calvin Xiao, Eren Cai, Qinhuai Na  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.12290v1  

#### Abstract
Current LLM agent benchmarks, which predominantly focus on binary pass/fail tasks such as code generation or search-based question answering, often neglect the value of real-world engineering that is often captured through the iterative optimization of feasible designs. To this end, we introduce Fro...

---

### 17. [DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](https://arxiv.org/abs/2604.12812)

**Authors**: Hao Yan, Yuliang Liu, Xingchen Liu, Yuyi Zhang, Minghui Liao, Jihao Wu, Wei Chen, Xiang Bai  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.12812v1  

#### Abstract
Existing Multimodal Large Language Models (MLLMs) suffer from significant performance degradation on the long document understanding task as document length increases. This stems from two fundamental challenges: 1) a low Signal-to-Noise Ratio (SNR), with crucial evidence buried in irrelevant pages; ...

---

### 18. [RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair](https://arxiv.org/abs/2604.12820)

**Authors**: Jagadeesh Rachapudi, Pranav Singh, Ritali Vatsi, Praful Hambarde, Amit Shukla  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.12820v1  

#### Abstract
Large language models (LLMs) inherently absorb harmful knowledge, misinformation, and personal data during pretraining on large-scale web corpora, with no native mechanism for selective removal. While machine unlearning offers a principled solution, existing approaches are provider-centric, requirin...

---

### 19. [Is Sliding Window All You Need? An Open Framework for Long-Sequence Recommendation](https://arxiv.org/abs/2604.12372)

**Authors**: Sayak Chakrabarty, Souradip Pal  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.12372v1  

#### Abstract
Long interaction histories are central to modern recommender systems, yet training with long sequences is often dismissed as impractical under realistic memory and latency budgets. This work demonstrates that it is not only practical but also effective-at academic scale. We release a complete, end-t...

---

### 20. [Memory as Metabolism: A Design for Companion Knowledge Systems](https://arxiv.org/abs/2604.12034)

**Authors**: Stefan Miteski  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12034v1  

#### Abstract
Retrieval-Augmented Generation remains the dominant pattern for giving LLMs persistent memory, but a visible cluster of personal wiki-style memory architectures emerged in April 2026 -- design proposals from Karpathy, MemPalace, and LLM Wiki v2 that compile knowledge into an interlinked artifact for...

---

### 21. [RPRA: Predicting an LLM-Judge for Efficient but Performant Inference](https://arxiv.org/abs/2604.12634)

**Authors**: Dylan R. Ashley, Ga\"el Le Lan, Changsheng Zhao, Naina Dhingra, Zhipeng Cai, Ernie Chang, Mingchen Zhuge, Yangyang Shi, Vikas Chandra, J\"urgen Schmidhuber  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12634v1  

#### Abstract
Large language models (LLMs) face a fundamental trade-off between computational efficiency (e.g., number of parameters) and output quality, especially when deployed on computationally limited devices such as phones or laptops. One way to address this challenge is by following the example of humans a...

---

### 22. [BEAM: Bi-level Memory-adaptive Algorithmic Evolution for LLM-Powered Heuristic Design](https://arxiv.org/abs/2604.12898)

**Authors**: Chuyang Xiang, Yichen Wei, Jiale Ma, Handing Wang, Junchi Yan  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12898v1  

#### Abstract
Large Language Model-based Hyper Heuristic (LHH) has recently emerged as an efficient way for automatic heuristic design. However, most existing LHHs just perform well in optimizing a single function within a pre-defined solver. Their single-layer evolution makes them not effective enough to write a...

---

### 23. [Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score](https://arxiv.org/abs/2604.12196)

**Authors**: Manh Nguyen, Sunil Gupta, Hung Le  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12196v1  

#### Abstract
Large language models (LLMs) frequently generate multiple candidate responses for a given prompt, yet selecting the most reliable one remains challenging, especially when correctness diverges from surface-level majority agreement. Existing approaches, such as self-consistency, rely on discrete votin...

---

### 24. [Transforming External Knowledge into Triplets for Enhanced Retrieval in RAG of LLMs](https://arxiv.org/abs/2604.12610)

**Authors**: Xudong Wang, Chaoning Zhang, Qigan Sun, Zhenzhen Huang, Chang Lu, Sheng Zheng, Zeyu Ma, Caiyan Qin, Yang Yang, Hengtao Shen  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12610v1  

#### Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucination in large language models (LLMs) by incorporating external knowledge during generation. However, the effectiveness of RAG depends not only on the design of the retriever and the capacity of the underlying model, but also on how retrieved ev...

---

### 25. [Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Sequence-Level Likelihood](https://arxiv.org/abs/2604.12736)

**Authors**: Xingyu Lin, Yilin Wen, Du Su, Jinchang Hou, En Wang, Wenbin Liu, Chenfu Bao, Zhonghou Lv  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12736v1  

#### Abstract
Group Relative Policy Optimization (GRPO) has significantly advanced the reasoning ability of large language models (LLMs), particularly in their mathemat ical reasoning performance. However, GRPO and related entropy regularization methods still struggle with token-level sparse-rewards, which is an ...

---

### 26. [PubSwap: Public-Data Off-Policy Coordination for Federated RLVR](https://arxiv.org/abs/2604.12160)

**Authors**: Anupam Nayak, Baris Askin, Muhammed Ustaomeroglu, Carlee Joe-Wong, Gauri Joshi  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12160v1  

#### Abstract
Reasoning post-training with reinforcement learning from verifiable rewards (RLVR) is typically studied in centralized settings, yet many realistic applications involve decentralized private data distributed across organizations. Federated training is a natural solution, but scaling RLVR in this reg...

---

### 27. [SubFlow: Sub-mode Conditioned Flow Matching for Diverse One-Step Generation](https://arxiv.org/abs/2604.12273)

**Authors**: Yexiong Lin, Jia Shi, Shanshan Ye, Wanyu Wang, Yu Yao, Tongliang Liu  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12273v1  

#### Abstract
Flow matching has emerged as a powerful generative framework, with recent few-step methods achieving remarkable inference acceleration. However, we identify a critical yet overlooked limitation: these models suffer from severe diversity degradation, concentrating samples on dominant modes while negl...

---

### 28. [PrivEraserVerify: Efficient, Private, and Verifiable Federated Unlearning](https://arxiv.org/abs/2604.12348)

**Authors**: Parthaw Goswami, Md Khairul Islam, Ashfak Yeafi  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12348v1  

#### Abstract
Federated learning (FL) enables collaborative model training without sharing raw data, offering a promising path toward privacy preserving artificial intelligence. However, FL models may still memorize sensitive information from participants, conflicting with the right to be forgotten (RTBF). To mee...

---

### 29. [Towards Platonic Representation for Table Reasoning: A Foundation for Permutation-Invariant Retrieval](https://arxiv.org/abs/2604.12133)

**Authors**: Willy Carlos Tchuitcheu, Tan Lu, Ann Dooms  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.12133v1  

#### Abstract
Historical approaches to Table Representation Learning (TRL) have largely adopted the sequential paradigms of Natural Language Processing (NLP). We argue that this linearization of tables discards their essential geometric and relational structure, creating representations that are brittle to layout...

---

### 30. [TRUST Agents: A Collaborative Multi-Agent Framework for Fake News Detection, Explainable Verification, and Logic-Aware Claim Reasoning](https://arxiv.org/abs/2604.12184)

**Authors**: Gautama Shastry Bulusu Venkata, Santhosh Kakarla, Maheedhar Omtri Mohan, Aishwarya Gaddam  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.12184v1  

#### Abstract
TRUST Agents is a collaborative multi-agent framework for explainable fact verification and fake news detection. Rather than treating verification as a simple true-or-false classification task, the system identifies verifiable claims, retrieves relevant evidence, compares claims against that evidenc...

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
