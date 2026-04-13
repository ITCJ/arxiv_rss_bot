# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-13 07:34:58 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Modality-Aware Zero-Shot Pruning and Sparse Attention for Efficient Multimodal Edge Inference](https://arxiv.org/abs/2604.08971)

**Authors**: Yueyuan Sui, Payal Mohapatra, Do\u{g}a\c{c} Eldenk, Haodong Yang, Yiting Zhang, Haoyan Zhang, Qi Zhu, Stephen Xia  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.08971v1  

#### Abstract
Edge devices increasingly run multimodal sensing pipelines that must remain accurate despite fluctuating power budgets and unpredictable sensor dropout. Existing pruning methods fail under these conditions: they generally require fine-tuning after compression, consuming over $10\times$ the deploymen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Modality-Aware Zero-Shot Pruning and Sparse Attention for Efficient Multimodal Edge Inference*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前多模态边缘推理面临三大挑战：
1. **资源受限**：边缘设备计算、内存和能耗有限，难以部署大型多模态模型（如基于Transformer的架构）。
2. **模态缺失（Sensor Dropout）**：传感器故障、通信中断或功耗限制导致部分模态在推理时不可用，影响模型鲁棒性。
3. **依赖微调（Fine-tuning）的剪枝方法不实用**：现有剪枝方法通常需要压缩后进行微调以恢复精度，但在边缘设备上执行反向传播不可行。

### 提出的新方法
作者提出 **SentryFuse** 框架，包含两个核心组件：

#### ✅ **SentryGate**：模态感知的零样本剪枝（Modality-Aware Zero-Shot Pruning）
- 引入一个轻量级的 **Dynamic Task-Aware Gating** 模块，在训练阶段通过监督学习（first-order saliency）为每个结构单元（如注意力头、FFN通道）生成**模态条件化的重要性分数**。
- 在部署时，无需微调，直接根据当前可用的模态掩码（modality mask）查询这些预学得的分数，实现动态、零样本的结构剪枝。

#### ✅ **SentryAttend**：稀疏分组查询注意力（Sparse Grouped-Query Attention）
- 替换标准的密集自注意力机制，结合两种技术：
  1. **Grouped-Query Attention (GQA)**：多个查询共享一组Key/Value投影，减少参数冗余。
  2. **稀疏查询选择**：仅保留注意力得分最高的少量查询（top-U queries），降低时间复杂度。
- 显著减少GFLOPs，缓解Transformer中的自注意力瓶颈。

### 相比现有方法的优势
| 维度 | 现有方法 | SentryFuse |
|------|--------|-----------|
| **剪枝方式** | 静态重要性评分，忽略模态变化 | 动态、模态感知的重要性评分 |
| **是否需微调** | 是（消耗大量能量） | 否（Zero-shot） |
| **应对模态缺失** | 差（静态剪枝在缺失下性能骤降） | 强（动态调整子网络） |
| **注意力效率** | 密集注意力（O(T²D)） | 稀疏+分组，显著降低GFLOPs |

---

## 2. 核心实验方法和设置

### 数据集
在三个多模态时间序列基准上进行评估：
- **WESAD**：可穿戴压力检测，10种生理信号，三分类任务。
- **DaliaHAR**：人体活动识别，多种传感器信号。
- **DSADS**：日常活动识别，高采样率传感器数据。

### 多模态骨干网络
使用三种代表性多模态Transformer架构：
- **MAESTRO**
- **FlexMoE**
- **FuseMoE**

### 基线方法对比
#### 剪枝基线（Structured Pruning）：
- **Random**：随机移除
- **Magnitude**：按权重大小剪枝
- **SynFlow**：无数据的突触流评分（data-free saliency）

#### 注意力机制基线：
- 原始的 **Dense Self-Attention**

### 评估指标
- **Accuracy**：测试集准确率
- **GFLOPs**：前向推理浮点运算量
- **Memory**：模型序列化大小（MB）
- **Latency**：端到端推理延迟（ms）
- **Energy Consumption**：总能耗（kJ）

### 硬件平台
在异构硬件上验证实用性：
- **服务器**：NVIDIA L40 GPU
- **嵌入式**：Jetson TX2
- **移动设备**：iPhone 13 Pro, Google Pixel 8
- **CPU平台**

---

## 3. 主要实验结果和性能指标

### 与基线方法的对比结果

#### ✅ 模态感知剪枝（SentryGate）显著优于静态剪枝
- 在 **144 种组合**（3 backbone × 3 dataset × 4 pruning ratio × 3 missingness level）中，SentryGate 在 **133 种情况**下优于最强基线（SynFlow）。
- **平均准确率提升 12.7%**（0.68 vs 0.60）。
- 在 **模态缺失严重时（4个模态丢失）**，提升高达 **18%**。

> 例如在 DaliaHAR 上，MAESTRO + 23% 剪枝 + 4模态缺失：
> - SynFlow：0.61 → **SentryGate：0.79**

#### ✅ SentryAttend 显著提升注意力效率
- 平均减少 **15% GFLOPs**，最高达 **29%**。
- 在多个模型上甚至**提升准确率**，表明其设计有效且非破坏性。

| 模型 | 数据集 | GFLOPs (原) → (SentryAttend) | Acc 变化 |
|------|-------|-----------------------------|---------|
| Transformer | WESAD | 22.25 → 15.81 (-29%) | 0.67 → 0.65 |
| MAESTRO | DSADS | 1.53 → 1.30 (-15%) | 0.98 → 0.88 |

#### ✅ SentryFuse 整体框架性能卓越
| 指标 | 原始 MAESTRO | MAESTRO + SentryGate | **SentryFuse** |
|------|------------|---------------------|---------------|
| Memory (MB) | 6.07 | 4.80 | **4.36** |
| GFLOPs | 6.83 | 5.07 | **4.41** |
| Accuracy (WESAD) | 0.75 | 0.75 | **0.77** |
| Latency (Pixel 8) | 256.28 ms | 169.96 ms | **157.43 ms** |

- 内存减少 **28.2%**，GFLOPs 减少 **35.4%**，延迟降低 **1.63×**，同时**提升准确率**。

### 消融实验结果

#### 🔹 SentryGate 能有效逼近梯度教师（Taylor-based Saliency）
- 在 DaliaHAR 上，SentryFuse 与 Taylor 教师性能接近：
  - 完整模态下，53% 剪枝：**0.78 vs 0.79**
  - 4模态缺失下，46% 剪枝：**0.71 vs 0.72**
- 表明其轻量前向门控能有效“蒸馏”梯度信息，避免部署时计算梯度。

#### 🔹 SentryAttend 中 GQA 分组数的影响
- 测试 1、2、8 个 Key/Value 组：
  - **2组** 在精度与效率间取得最佳平衡。
  - 1组共享过多，灵活性差；8组接近原始MHA，效率低。
- 最终采用 **2组** 作为默认配置。

---

## 4. 关键结论和发现

### 主要发现
1. **模态感知剪枝是必要的**：静态剪枝在模态缺失下性能急剧下降，而 SentryGate 能根据当前可用模态动态调整子网络，保持鲁棒性。
2. **零样本剪枝可行且高效**：通过在训练阶段学习模态条件化的重要性函数，可在部署时实现无需微调的即时压缩。
3. **稀疏注意力可显著提升效率**：SentryAttend 通过 GQA 和稀疏查询选择，有效缓解自注意力瓶颈，平均降低 15% GFLOPs。
4. **SentryFuse 实现“一次训练，多次部署”范式**：适用于不同功率预算和传感器配置的异构边缘设备。

### 方法的局限性
- **依赖模态掩码输入**：模型需显式编码哪些模态缺失，对掩码预测错误敏感。
- **SentryGate 增加少量参数开销**：约增加 4–9% 参数量，虽不影响主干训练，但仍需权衡。
- **未探索更复杂的稀疏模式**：如动态稀疏度控制或任务自适应稀疏。

### 未来工作方向
- 将 SentryFuse 扩展至 **多模态大语言模型（MLLMs）** 的边缘部署。
- 结合 **动态分辨率输入** 或 **早期退出机制** 进一步优化推理效率。
- 探索 **无监督或自监督训练** 下的模态感知剪枝，减少对标签的依赖。

---

> **总结**：  
> SentryFuse 通过 **SentryGate** 和 **SentryAttend** 两大创新，实现了**无需微调、模态感知、高效稀疏**的多模态边缘推理框架，在多个真实场景和硬件平台上验证了其优越性，为资源受限下的鲁棒多模态智能提供了实用路径。

</details>

---

### 2. [Sensor Placement for Tsunami Early Warning via Large-Scale Bayesian Optimal Experimental Design](https://arxiv.org/abs/2604.08812)

**Authors**: Sreeram Venkat, Stefan Henneking, Omar Ghattas  
**Category**: cs.DC  
**Published**: 2026-04-13  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.08812v1  

#### Abstract
Real-time tsunami early warning relies on distributed sensor networks to infer seismic sources and seafloor motion. Optimizing these networks via Bayesian optimal experimental design (OED) is exceptionally challenging for systems governed by hyperbolic partial differential equations, which lack the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Sensor Placement for Tsunami Early Warning via Large-Scale Bayesian Optimal Experimental Design

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文致力于解决**海啸早期预警系统中传感器网络的最优部署问题**，即在有限预算下，如何选择海底压力传感器的位置以最大化对地震引发的海床运动（seafloor motion）推断的精度。这一问题本质上是**大规模贝叶斯最优实验设计**（Bayesian Optimal Experimental Design, OED），尤其针对由**双曲型偏微分方程**（hyperbolic PDEs）控制的动力系统。

传统方法在处理此类高维、非光滑系统的OED时面临巨大挑战：
- 贝叶斯反演本身计算昂贵；
- OED需多次求解反问题，形成“外层优化环”；
- 双曲系统缺乏低秩结构，无法使用标准的低秩近似加速。

### 🚀 提出的新方法与创新思路

作者提出了一套**可扩展的大规模贝叶斯D-optimal OED框架**，其核心创新如下：

#### （1）**算法层面：基于Schur补更新的贪婪算法**
- 将OED目标函数从参数空间转移到**数据空间**（data space），利用Sherman-Morrison-Woodbury恒等式重构后端协方差。
- 引入**块状Schur补更新机制**（block Schur complement update），避免每次候选传感器评估时重新进行完整的Cholesky分解。
- 显著降低单次迭代复杂度：从 $O(k^3)$ 降至 $O(k^2)$，总复杂度从 $O(|C|B^4N_t)$ 降为 $O(|C|B^3N_t)$。

#### （2）**系统实现：多GPU并行与I/O流水线优化**
- 设计了一个**完全重叠I/O与计算的双缓冲异步执行管道**（double-buffered pipelined execution）：
  - CPU读取下一个候选矩阵块的同时，GPU正在处理当前候选；
  - 使用CUDA/HIP事件同步，消除空闲等待。
- 采用**零分配内存管理策略**（zero-allocation in-place memory management）减少GPU内存碎片。
- 利用**独立POSIX I/O**访问HDF5格式的稠密Hessian矩阵 $K$，优于MPI-IO在稀疏读取场景下的表现。

#### （3）**应用突破：首次在$O(10^9)$维度参数场上求解PDE约束OED**
- 成功应用于Cascadia数字孪生模型（Gordon Bell Prize 2025获奖工作），实现了对**超过10亿自由度的参数场**的OED优化。
- 无需任何降阶建模（ROM）、代理模型或简化假设，直接基于高保真物理模型求解。

### 🔍 相比现有方法的优势

| 方面 | 本文方法 | 传统方法 |
|------|--------|---------|
| **适用系统类型** | 支持LTI双曲系统（如声重波传播） | 主要适用于椭圆/抛物系统（具低秩特性） |
| **计算效率** | Schur更新使每轮评估快几个数量级 | 需重复因子化，成本极高 |
| **可扩展性** | 支持数百至千GPU强/弱扩展 | 多数仅限单机或小规模集群 |
| **精度保障** | 基于完整高保真PDE模型 | 常依赖简化模型或低秩近似 |

---

## 2. 核心实验方法和设置

### 📊 数据集与物理背景
- **应用场景**：Cascadia俯冲带（CSZ）海啸预警数字孪生系统。
- **前序工作基础**：基于Henneking et al. [7] 构建的实时贝叶斯反演框架，能从海底压力观测中推断时空连续的海床位移场。
- **候选传感器集合**：共 **600个候选位置**，分布于北加州至不列颠哥伦比亚沿岸（见图1）。
- **目标参数场**：代表地震引起的海床垂直位移，离散化后具有 **超10亿自由度**（spatiotemporal field）。

### ⚙️ 实验设置
- **OED任务**：从600个候选位置中选出 **B=175个最优传感器**，实现最大信息增益（D-optimal design）。
- **OED目标函数**：
  $$
  \Phi(S) = -\log \det(K_S), \quad K = \Gamma_{\text{noise}} + F \Gamma_{\text{prior}} F^*
  $$
  其中 $K$ 是数据空间Hessian矩阵，可通过FFT加速的Hessian-vector乘法高效构建。
- **算法实现**：
  - 使用 **PyTorch + mpi4py** 实现跨平台多GPU支持（NVIDIA A100 / AMD MI250X）。
  - $K$ 存储为 **chunked HDF5 2D dataset**（464 GB, float64），计算阶段转为float32提升吞吐。
- **并行策略**：
  - 每个GPU负责一部分候选传感器的得分评估（embarrassingly parallel）。
  - 每轮通过 `MPI_Allreduce` 同步全局最优传感器索引。

### 📈 评估指标
- **运行时间**（wall-clock time）：衡量端到端OED求解速度。
- **强/弱可扩展性**：测试不同GPU数量下的性能提升。
- **内存占用峰值**：评估算法内存效率。
- **目标函数值比较**：对比随机配置 vs. 贪婪优化配置的信息增益。
- **不确定性场演化可视化**：展示随传感器增加，后验标准差的空间收敛过程。

### 🆚 基线方法对比
虽然未与其他OED算法直接对比（因多数无法扩展至此规模），但文中设置了两种内部对照：
1. **Naive greedy algorithm**：每次对测试矩阵从头进行Cholesky分解（无Schur更新）。
2. **Random sensor selection**：作为性能下界基准。

---

## 3. 主要实验结果和性能指标

### 📉 单GPU性能对比（图2）
| 架构 | 方法 | 最大支持传感器数（B） | 相对速度 |
|------|------|---------------------|----------|
| NVIDIA A100 (80GB) | Naive | ~100 | 1× |
| | **Schur（本文）** | **340** | **>100× faster** |
| AMD MI250X (64GB) | Naive | ~60 | — |
| | **Schur** | **300** | 数量级加速 |
| NVIDIA GH200 (96GB) | Schur | **370** | 内存更优 |

> 结果显示Schur方法不仅更快，且显著提升了可处理的最大预算 $B$。

### 🔁 并行可扩展性（图3）
在 **Perlmutter** 和 **Frontier** 超算上测试：
- **强扩展**（固定问题规模，增加GPU）：
  - 使用8192候选传感器，在1–512 GPU上接近理想线性加速。
  - 效率维持在97%-101%，表明I/O与计算完美重叠。
- **弱扩展**（按比例扩大问题）：
  - 每GPU处理256候选，总候选达13万（Perlmutter）和26万（Frontier）。
  - 仍保持近理想扩展，证明框架具备极强横向扩展能力。

> **关键洞察**：正是由于**流水线I/O与计算重叠**，才避免了I/O成为瓶颈，否则早期版本无法达到此效率。

### 🎯 应用于CSZ数字孪生的结果
- **输入**：600候选位置对应的 $K \in \mathbb{R}^{252,000 \times 252,000}$（$N_d=600, N_t=420$）。
- **硬件资源**：16台NVIDIA A100 GPU（Perlmutter）。
- **耗时**：**仅1.5小时**完成175轮贪婪选择。
- **结果质量**（图4）：
  - 对比100组随机选取的175传感器配置，本文方法的目标函数值远优于所有随机配置。
  - 表明贪婪算法有效捕捉到了高信息增益的空间模式。

### 🖼️ 不确定性场演化（图5）
随着传感器逐步加入（10 → 175）：
- 后验标准差在整个CSZ区域均匀下降；
- 特别是在俯冲带浅部（靠近海岸）不确定性降低明显；
- 动画显示信息增益呈现“前沿推进”特征，反映波传播动力学影响。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **大规模OED首次在双曲系统上实现**：
   - 首次成功在**无降阶模型前提下**，解决由hyperbolic PDE驱动的、参数维度超十亿的贝叶斯OED问题。

2. **Schur补更新是关键算法突破**：
   - 消除冗余矩阵分解，将复杂度降低一个量级，使得贪婪搜索在实际时间内可行。

3. **系统级优化决定可扩展性上限**：
   - 即便算法高效，若I/O不能与计算重叠，也无法发挥GPU潜力；
   - **双缓冲+异步传输+独立I/O** 是实现高性能的关键。

4. **贪婪算法在实践中足够有效**：
   - 尽管是近似算法，但由于D-optimal目标函数满足**子模性**（submodularity），理论保证至少达到全局最优的 $1 - 1/e \approx 63\%$；
   - 实验结果显示其远超随机选择，具备实用价值。

### ⚠️ 局限性
- **仅适用于LTI系统**：当前框架依赖于shift-invariance性质（block-Toeplitz结构），难以推广至非线性或时变系统。
- **贪婪策略非全局最优**：虽有理论下界，但仍可能陷入局部最优。
- **静态设计**：所选传感器位置为一次性离线设计，未考虑动态自适应部署。

### 🔮 未来工作方向
1. **扩展至非线性系统**：结合线性化或增量更新策略，应用于更复杂的地球物理模型。
2. **在线/自适应OED**：根据实时地震信号动态调整后续观测策略。
3. **融合多源数据**：联合GNSS、海底压力、岸基潮位等多种观测设计综合OED。
4. **成本感知与工程约束集成**：进一步引入地形深度、通信链路、维护难度等现实因素加权优化。

---

> **总结一句话**：  
> 本论文通过**算法创新（Schur更新） + 系统协同设计（I/O与GPU流水线）**，首次实现了在极端尺度下对海啸预警系统的科学化、数学严谨的传感器布局优化，为下一代海洋监测网络提供了强有力的决策工具。

</details>

---

### 3. [SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks](https://arxiv.org/abs/2604.08865)

**Authors**: Tianyi Wang, Yixia Li, Long Li, Yibiao Chen, Shaohan Huang, Yun Chen, Peng Li, Yang Liu, Guanhua Chen  
**Category**: cs.AI  
**Published**: 2026-04-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.08865v1  

#### Abstract
Proximal Policy Optimization (PPO) is central to aligning Large Language Models (LLMs) in reasoning tasks with verifiable rewards. However, standard token-level PPO struggles in this setting due to the instability of temporal credit assignment over long Chain-of-Thought (CoT) horizons and the prohib...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在长思维链（Chain-of-Thought, CoT）推理任务中，传统的 **token-level PPO** 存在两个核心问题：

- **Temporal Credit Assignment 不稳定**：由于奖励稀疏（仅在最终答案处给出），标准 PPO 依赖 **Generalized Advantage Estimation (GAE)** 将终端奖励反向传播到数千个中间 token。这导致高偏差（high bias），且价值模型（Critic）容易“过拟合”序列末尾的语义线索（即“Tail Effect”），使得优势信号在关键推理步骤处消失。
  
- **计算开销大**：虽然 **critic-free 方法如 GRPO** 通过多采样（multi-sampling）构建组内统计基线来避免学习 Critic，但其需要对每个提示生成多个响应（如 $N=8$），显著降低了训练吞吐量。

### 提出了什么新方法或新思路

本文提出 **SPPO (Sequence-Level PPO)**，其核心思想是将推理过程从 **token-level MDP** 重构为 **Sequence-Level Contextual Bandit** 问题：

- **Prompt 作为上下文（Context）**，整个推理链作为 **原子动作（atomic action）**。
- 引入一个 **解耦的标量价值函数 $V(s_p)$**，预测给定提示 $s_p$ 下模型成功的概率（即问题可解性），而非逐 token 预测未来回报。
- 优势函数定义为 $A = R - V(s_p)$，其中 $R \in \{0,1\}$ 是二元奖励。该优势信号被 **广播（broadcast）** 到序列中的所有 token 上。

### 相比现有方法的优势

| 特性 | Standard PPO | GRPO | SPPO |
|------|-------------|------|------|
| **Credit Assignment** | 高偏差（GAE 传播困难） | 无（直接 outcome-based） | 低偏差（标量值函数） |
| **Advantage Variance** | 低（Critic 平滑） | 高（依赖多采样方差） | 低（学习型标量基线） |
| **样本效率** | 高（单样本更新） | 低（需 $N>1$ 采样） | 高（单样本更新，$N=1$） |
| **训练速度** | 快 | 慢（受限于多采样） | 极快（5.9× GRPO） |
| **内存占用** | 高（Critic 与 Policy 同规模） | 中等 | 可降低（支持小规模 Critic） |

> ✅ **SPPO 成功调和了 PPO 的样本效率与 GRPO 的稳定性，在保持单样本高效的同时实现了更优性能。**

---

## 2. 核心实验方法和设置

### 使用的数据集

- **训练数据集**：
  - `DeepScaleR`：用于微调 1.5B 模型。
  - `DAPO-17K`：高质量数学推理子集，用于微调 7B 模型。
- **评估基准（held-out benchmarks）**：
  - AIME24, AIME25
  - AMC23
  - MATH500
  - Minerva Math

### 实验设置和评估指标

- **模型规模**：
  - `DeepSeek-R1-Distill-Qwen-1.5B`
  - `DeepSeek-R1-Distill-Qwen-7B`
- **评估方式**：
  - 使用 **Average@16 准确率**（Average@16 accuracy）进行评估。
- **硬件配置**：
  - 1.5B 模型：4 × A100 GPUs
  - 7B 模型：4 × H100 GPUs
- **实现框架**：基于 `verl` 框架实现所有算法。

### 基线方法对比

- **Base Model**：未经 RL 微调的初始模型。
- **Standard PPO**：传统 token-level PPO，使用 GAE 和 Critic。
- **ReMax**, **RLOO**：其他 sequence-level RL 方法。
- **GRPO ($N=8$)**：group-based 方法，每提示采样 8 条响应以构建基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 1.5B 模型上的平均准确率（Avg）

| Method | Avg Score |
|--------|-----------|
| Base Model | 44.96 |
| PPO | 44.06 |
| ReMax | 46.74 |
| RLOO | 46.15 |
| GRPO ($N=8$) | 47.08 |
| **SPPO (Ours)** | **48.06** ✅ |

> 🔺 SPPO 超越 GRPO，提升 **+0.98** 绝对准确率。

#### 在 7B 模型上的平均准确率（Avg）

| Method | Avg Score |
|--------|-----------|
| Base Model | 52.49 |
| PPO | 56.44 |
| ReMax | 57.09 |
| RLOO | 57.02 |
| GRPO ($N=8$) | 57.44 |
| **SPPO (Ours)** | **58.11** ✅ |
| **SPPO + Small Critic (1.5B Critic)** | **58.56** ✅✅ |

> 🔺 SPPO 再次超越 GRPO（+0.67），且使用 **轻量级 Critic（1.5B）对齐 7B Policy** 进一步提升至 **58.56**。

### 与基线方法的对比结果

- **优于所有基线**：SPPO 在绝大多数 benchmark 上均取得最高分。
- **训练效率远超 GRPO**：
  - 图5显示，SPPO 达到峰值性能所需时间约为 **22小时**，而 GRPO 需要更长时间且收敛缓慢。
  - **训练速度提升达 5.9×**。
- **内存优化显著**：
  - 使用 **Decoupled Critic**（1.5B Critic 对齐 7B Policy）可减少 **12.8% 的显存占用**（图6）。
  - 支持在消费级硬件上部署大规模 RL 微调。

### 消融实验结果

#### 控制变量：PPO + BCE 损失函数（图4）

- 实验设计：将 SPPO 中使用的 **Binary Cross-Entropy (BCE)** 损失应用于标准 token-level PPO。
- 结果：**性能未改善**，甚至早衰崩溃。
- 结论：SPPO 的成功并非源于损失函数本身，而是 **Sequence-Level Contextual Bandit 的建模范式** 所致。

#### Critic 分析（图7）

- **校准性分析**：Critic 预测的成功概率 $V(s_p)$ 与实际经验成功率（Avg@64）呈正相关。
  - Pearson 相关系数：0.642
  - Spearman 秩相关：0.664
- 表明：标量价值函数能有效捕捉问题难度分布，是一个有效的低方差基线。

---

## 4. 关键结论和发现

### 主要发现

1. **GRPO 成功的本质是隐式地将任务建模为 Sequence-Level Contextual Bandit**，而非多步 MDP。
2. **显式采用此建模范式并引入学习型标量价值函数**，可在不牺牲稳定性的情况下摆脱对多采样的依赖。
3. **SPPO 实现了高吞吐、高稳定、高性能的统一**：
   - 单样本更新（$N=1$）
   - 无需复杂的系统级优化
   - 性能超越计算密集型的 GRPO
4. **Critic 规模可大幅缩减**：轻量级 Critic（1.5B）即可有效指导大 Policy（7B），验证了价值估计任务的相对简单性。

### 方法的局限性

- **依赖可验证奖励（Verifiable Rewards）**：SPPO 假设存在明确的二元正确性判断（如数学题有唯一答案）。对于开放域生成任务（如创意写作、伦理决策）缺乏客观 ground truth，难以应用。
- **假设 Prompt Solvability 可学习**：要求训练数据足够多样化，以便价值模型泛化到新问题的难度估计。

### 未来工作方向

- 探索如何将 SPPO 范式扩展至 **非 verifiable 或弱监督任务**，例如结合不确定性估计或人类反馈。
- 研究更高效的 **value model 架构压缩技术**，进一步降低 RL 训练门槛。
- 将 SPPO 应用于其他长程决策任务，如代码生成、规划、对话策略学习等。

---

> 📌 **一句话总结**：  
> SPPO 通过将推理任务重新建模为 **Sequence-Level Contextual Bandit**，用一个 **标量价值函数** 替代传统 Critic，解决了 long-horizon reasoning 中 credit assignment 不稳定与训练效率低下的矛盾，在数学推理任务上实现了 **性能更强、速度更快、资源更省** 的 RL 微调新范式。

</details>

---

### 4. [Camera Artist: A Multi-Agent Framework for Cinematic Language Storytelling Video Generation](https://arxiv.org/abs/2604.09195)

**Authors**: Haobo Hu, Qi Mao, Yuanhang Li, Libiao Jin  
**Category**: cs.AI  
**Published**: 2026-04-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.09195v1  

#### Abstract
We propose Camera Artist, a multi-agent framework that models a real-world filmmaking workflow to generate narrative videos with explicit cinematic language. While recent multi-agent systems have made substantial progress in automating filmmaking workflows from scripts to videos, they often lack exp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Camera Artist: A Multi-Agent Framework for Cinematic Language Storytelling Video Generation》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 **multi-agent system (MAS)** 的叙事视频生成方法虽然能够从脚本自动生成视频，但仍存在两大核心缺陷：
- **Narrative Drift**：缺乏对相邻镜头之间视觉与叙事连贯性的显式建模，导致镜头切换生硬、故事断裂。
- **Cinematic Expressiveness 不足**：生成的镜头描述多为泛化语言，缺少专业电影语言（如镜头运动、构图、光影等）的精细控制，导致视频“像动画”而非“像电影”。

这些问题使得现有系统难以实现真正具有电影质感的长篇叙事视频生成。

### **提出的新方法与创新思路**
作者提出了 **Camera Artist**，一个面向**电影级叙事表达**的 multi-agent 框架，其核心创新在于两个模块：

#### ✅ **Recursive Shot Generation (RSG)**
- 在镜头生成过程中引入 **递归机制**，每个镜头的生成都依赖于前一个镜头的上下文（via Chain-of-Thought 推理）。
- 定义三种镜头类型：`Scene Start Point`, `Scene Midpoint`, `Scene End Point`，以结构化方式确保场景内部的叙事流动性和逻辑连续性。

#### ✅ **Cinematic Language Injection (CLI)**
- 引入一个经过 **LoRA fine-tuned 的 LLM**，专门用于将普通镜头描述转换为富含专业电影语言的技术性描述。
- 利用 **ShotBench 数据集**中的专业标注（镜头大小、角度、运镜、灯光等），训练模型注入如“高角度俯拍”、“缓慢推近”、“动态航拍环绕”等表达，显著提升视觉表现力。

### **相比现有方法的优势**
| 维度 | 传统方法（如 MovieAgent, VGoT） | Camera Artist |
|------|-------------------------------|-------------|
| **Narrative Coherence** | 仅基于脚本生成镜头，忽略前后镜头关联 | 显式建模镜头间递归依赖，增强叙事流畅性 |
| **Cinematic Quality** | 镜头描述通用，缺乏专业术语 | 注入专业电影语言，提升导演感与艺术性 |
| **视觉动态性** | 多为静态或固定机位 | 支持复杂运镜设计（拉、摇、跟、升格等） |
| **整体观感** | 类似幻灯片拼接 | 更接近真实电影剪辑节奏与美学风格 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **主基准**：`MoviePrompts [11]`  
  包含来自 10 部专业电影的剧情摘要与角色设定，用于评估叙事一致性。
- **扩展基准**：额外构建 8 个原创故事样本，验证泛化能力。
- **训练数据**：`ShotBench [16]` 中精选 580 对 `(普通描述 x, 电影化描述 y)` 样本，用于 CLI 模块的 LoRA 微调。

### **实验设置**
- **LLM 主干**：Qwen3-30B-A3B-Instruct（所有 agent 默认）
- **CLI 模块微调**：Qwen3-4B + LoRA（rank=8, scaling=32, lr=1e-4, 20 epochs）
- **视频生成器**：MAGREF（支持 multi-reference I2V 控制）
- **参考图像生成**：Flux [18]
- **输出规格**：832×480 分辨率，15fps
- **硬件平台**：单张 NVIDIA Tesla A800 80G GPU

### **评估指标**

#### **自动化指标**
| 指标 | 含义 |
|------|------|
| **CLIP-T** | 文本-视频语义相似度（衡量剧本忠实度） |
| **VBench Metrics** | 多维度视频质量评估：<br>- Subj.（主体一致性）<br>- Bg.（背景一致性）<br>- Motion（动作平滑性）<br>- Dyn.（动态程度）<br>- Aesth.（美学评分） |

#### **VLM-based 自动评估**
使用 GPT-4o、Qwen3、Gemini-3 作为 evaluator，对以下四项打分（1–5）：
- **Script Consistency**：是否符合原始剧本
- **Camera-Movement Consistency**：运镜是否匹配描述
- **Video Quality**：画质稳定性与清晰度
- **Real-Movie Similarity**：是否像真实电影

#### **人工用户研究**
采用五点李克特量表（Likert Scale），由人类评委盲评上述四个维度，避免命名偏见。

### **对比的基线方法**
- **VGoT [10]**：VideoGen-of-Thought，强调逐步推理
- **Anim-Director [8]**：专注于可控动画生成
- **MovieAgent [11]**：早期 multi-agent 电影生成框架

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（定量结果）**

#### 📊 **Table I & II：综合性能对比**
| Method | CLIP-T ↑ | Subj. ↑ | Bg. ↑ | Motion ↑ | Dyn. ↑ | Aesth. ↑ | Script Cons. (avg) ↑ | Cam. Cons. (avg) ↑ | Real-Sim. (avg) ↑ |
|--------|----------|---------|-------|----------|--------|----------|---------------------|--------------------|------------------|
| VGoT | 28.15 | 78.58 | 97.93 | 99.27 | 16.67 | 68.73 | 2.83 | 1.33 | 3.50 |
| Anim-Director | 23.86 | 67.79 | 94.15 | 96.54 | 39.78 | 67.24 | 2.98 | 2.30 | 1.61 |
| MovieAgent | 22.25 | 71.01 | 94.52 | 98.00 | 76.27 | 65.63 | 2.19 | 2.03 | 3.13 |
| **Ours** | **29.61** | **79.54** | **96.26** | **99.32** | **80.00** | **69.51** | **3.90** | **3.55** | **4.02** |

✅ **Camera Artist 在所有指标上均取得最优或次优成绩**，尤其在：
- **Dynamic Degree (+3.73 vs MovieAgent)**：说明镜头更具动感
- **Real-Movie Similarity (+0.89 vs best baseline)**：最接近真实电影观感
- **Camera-Movement Consistency (+1.52 vs best baseline)**：运镜精准度显著领先

### **消融实验结果（Ablation Study）**

#### 📉 **Table III：移除 RSG 或 CLI 的影响**
| Method | CLIP-T | Subj. | Dyn. | Aesth. | Script Cons. | Cam. Cons. | Real-Sim. |
|--------|--------|--------|------|--------|--------------|------------|-----------|
| w/o RSG | 28.22 | 74.69 | 78.67 | 67.45 | 3.55 | 3.36 | 3.91 |
| w/o CLI | 29.27 | 73.49 | 74.25 | 67.10 | 3.60 | 2.83 | 3.67 |
| **Full Model** | **29.61** | **79.54** | **80.00** | **69.51** | **3.90** | **3.55** | **4.02** |

🔍 **关键发现**：
- 移除 **RSG** 导致叙事一致性下降（Script Cons. ↓），出现角色突变、情节跳跃。
- 移除 **CLI** 导致运镜质量骤降（Cam. Cons. ↓ 从 3.55 → 2.83），画面趋于静态。
- 二者协同作用，共同提升整体电影感。

### **定性结果（Qualitative Analysis）**

#### 🔹 单镜头表达（Fig. 4）
- 当描述 “Elsa senses magical energy” 时：
  - 基线方法多为中景/近景静态拍摄
  - **Camera Artist** 使用 “high-angle wide shot with smooth zoom-out”，增强神秘氛围与空间张力

#### 🔹 多镜头连贯性（Fig. 5, Fig. 12b）
- 基线方法常出现：
  - 角色身份切换错误（Anna → Elsa 突兀）
  - 场景时空不一致（白天→黑夜无过渡）
- **Camera Artist** 能保持人物与环境稳定，并实现自然推进：“进入森林 → 发现湖泊 → 深入探索”

#### 🔹 用户研究（Fig. 6）
- 在四项主观指标中均获得最高分：
  - **Script Consistency**: 4.28 / 5
  - **Real-Movie Similarity**: 4.12 / 5
- 表明人类观众也认为其输出更连贯、更具电影感

---

## **4. 关键结论和发现**

### **主要发现**
1. **Narrative Continuity 是电影感的基础**：通过 RSG 实现镜头间的递归依赖，有效缓解 narrative drift。
2. **Cinematic Language 是电影感的关键**：CLI 模块成功将抽象情节转化为具象化的专业镜头语言，极大提升视觉表现力。
3. **Multi-Agent 架构可模拟真实制片流程**：Director → Cinematography Shot Agent → Video Generation Agent 的分工，逼近真实剧组协作模式。
4. **自动化评估与人类感知高度一致**：VLM-based 评估能较好反映 human judgment，在大规模测试中具备可行性。

### **方法的局限性**
- **依赖高质量 reference image**：尽管支持纯文本输入（见 Fig. 13），但在复杂角色一致性任务中仍受益于 reference 提供的身份锚定。
- **CLI 模块泛化能力受限于 ShotBench 数据分布**：若遇到极端罕见镜头类型（如子弹时间、鱼眼畸变），可能无法准确建模。
- **计算成本较高**：需多个 LLM 与 I2V 模型协同运行，端到端延迟较大，不适合实时应用。

### **未来工作方向**
- 扩展 CLI 模块至多语言与跨文化电影风格（如 noir, anime, wuxia）
- 引入 audio-agent 实现音画同步（配乐、对白、声效）
- 结合 reinforcement learning 进行 cinematic policy learning
- 探索 zero-shot cinematic transfer（无需 fine-tuning 即可模仿特定导演风格）

---

> ✅ **总结一句话**：  
> **Camera Artist 通过 Recursive Shot Generation 和 Cinematic Language Injection 双轮驱动，在 multi-agent 视频生成框架中首次实现了“讲好故事”与“拍出电影感”的统一，推动 AIGC 向专业影视制作迈进一步。**

</details>

---

### 5. [Efficient RL Training for LLMs with Experience Replay](https://arxiv.org/abs/2604.08706)

**Authors**: Charles Arnal, Vivien Cabannes, Taco Cohen, Julia Kempe, Remi Munos  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.08706v1  

#### Abstract
While Experience Replay - the practice of storing rollouts and reusing them multiple times during training - is a foundational technique in general RL, it remains largely unexplored in LLM post-training due to the prevailing belief that fresh, on-policy data is essential for high performance. In thi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Efficient RL Training for LLMs with Experience Replay》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代大型语言模型（LLMs）在推理任务中广泛采用强化学习（Reinforcement Learning, RL）进行后训练（post-training），但当前主流方法（如 PPO、GRPO）普遍采用**严格 on-policy** 的范式，即生成的轨迹（rollouts）仅用于一次梯度更新后即被丢弃。这种“生成即丢弃”（generate-then-discard）的方式导致极高的推理计算开销，通常占整个训练预算的 **80%以上**。

尽管在传统 RL 中，**Experience Replay**（经验回放）是提升样本效率的核心技术，但在 LLM 训练中，人们普遍认为 off-policy 数据会因策略过时（staleness）而导致性能下降，因此长期被忽视。

本文挑战了这一共识，系统研究了在 LLM 后训练中引入 Experience Replay 的可行性与效益。

---

### **提出了什么新方法或新思路**
作者提出将 **Experience Replay 缓冲区**（replay buffer）集成到异步 LLM RL 训练流水线中，允许轨迹被多次重用，从而显著降低推理计算成本。

- **理论框架**：首次形式化分析了在 LLM RL 中使用 replay buffer 的三重权衡：
  - **Staleness-induced variance**（由策略过时引起的梯度方差）
  - **Sample diversity**（样本多样性损失）
  - **Generation cost**（生成成本）
  
  并推导出最优缓冲区大小 $N$ 和回放比例 $B/R$ 的理论边界，证明当推理成本高昂时，适度的 off-policy 回放是最优选择。

- **实现方式**：
  - 在异步架构中，$W$ 个 inference worker 持续向共享 buffer 添加 rollout；
  - $T$ 个 trainer 从 buffer 中采样构建训练批次；
  - buffer 支持 FIFO 管理和均匀随机采样，不删除已采样项。

---

### **相比现有方法的优势**
| 维度 | 传统方法（On-policy） | 本文方法（Experience Replay） |
|------|------------------------|-------------------------------|
| **计算效率** | 极低，每样本仅用一次 | 显著提高，样本可复用多次 |
| **GPU 利用率** | 易出现空闲（生产消费耦合） | 更稳定，解耦生产与消费 |
| **最终性能** | 可能不稳定或崩溃 | 更稳定，有时更高 |
| **实现复杂度** | 简单但浪费资源 | 轻量改动即可集成 |

> ✅ **核心优势**：以可控的 off-policiness 换取高达 **40% 的推理计算节省**，同时保持甚至提升最终准确率。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **OpenR1-Math-220k**：大规模数学推理数据集，用于主实验。
- **MATH**：标准数学评测集，用于测试泛化能力。
- **miniF2F**：Lean 形式化定理证明任务，用于跨任务验证。
- **Llama3.2 3B on OpenR1-Math-220k**：验证方法在不同模型上的普适性。

---

### **实验设置**
- **模型**：
  - Qwen3-0.6B、Qwen2.5-7B（主实验）
  - Qwen3-8B、Llama3.2 3B（扩展实验）
- **训练算法**：
  - 主要使用 **GRPO**（Group Relative Policy Optimization）
  - 部分实验尝试 **AsymRE** 损失函数
- **硬件配置**：
  - 使用 H100/H200 GPU
  - 异步并行架构：$W$ inference workers + $T$ trainers
- **缓冲区设计**：
  - FIFO 缓冲区，容量 $N \in \{64, 256, ..., 20736\}$
  - 采样策略：默认 uniform sampling
  - 扩展尝试：positive-bias sampling（优先保留正确解答）

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Accuracy on MATH/OpenR1-Math** | 主要性能指标 |
| **Compute spent** | 归一化的总计算量（考虑 inference + training） |
| **Wall-time** | 实际运行时间 |
| **Pass@k** ($k>1$) | 多样性指标，衡量输出分布熵是否保留 |
| **Replay Ratio** | 平均每个样本被使用的次数 |
| **Off-policiness** | 样本生成步数与当前训练步数之差 |

---

### **基线方法对比**
- **Baseline**：无 buffer 的 on-policy 训练（典型异步 RL 架构）
- **对比维度**：
  - 相同 compute 下的 accuracy
  - 达到相同 accuracy 所需 compute
  - 训练稳定性（是否崩溃）
  - 输出多样性（pass@k）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **图1 & 图15：计算效率显著提升**
- 使用 replay buffer（如 $N=84$, $(W,T)=(5,3)$）可在 **减少约 40% compute** 的情况下达到与 on-policy baseline 相同甚至更高的 MATH 准确率。
- 例如，在 Qwen2.5-7B 上，baseline 需 ~35k compute 单位达到 0.76 准确率，而 buffer 方法仅需 ~21k。

#### ✅ **图3：小模型上最高节省 40% compute**
- Qwen3-0.6B 实验显示，最佳 buffer 配置可在相同 accuracy 下节省近 **40% compute**。
- 更高 replay ratio（如 $(W,T)=(4,4)$）带来更大效率增益，但需控制 buffer size 以防性能下降。

#### ✅ **图4：Pareto 前沿优于 baseline**
- 在 learning rate 和 buffer 参数联合调优下，所有 buffer 配置的 **Pareto frontier 完全支配** on-policy baseline。
- 表明 replay buffer 不仅省算力，还能找到更优的性能-效率平衡点。

#### ✅ **图5：改进采样策略进一步提升**
- 使用 **positive-bias sampling**（保留更多正确解答）结合 **AsymRE 损失**，可在相同步数下更快收敛且更稳定。
- 尤其在高 $\delta$（如 0.5）时表现更好，说明对高质量样本的偏好有助于缓解 staleness 问题。

#### ✅ **图10–11：实际 wall-time 加速**
- 在真实系统中，buffer 还能缓解 GPU 等待问题（inference worker 或 trainer stall），使得 wall-time 加速与 compute 节省一致甚至更优。

---

### **消融实验结果**
#### 🔹 **缓冲区大小影响（图3 左）**
- 缓冲区越大 → 平均 off-policiness 越高 → 训练越慢但越稳定
- 中等大小（如 256–768）在效率与稳定性间取得最佳平衡

#### 🔹 **$W/T$ 比例影响（图2 & 表1）**
- $W/T$ 越小 → replay ratio 越高 → compute 成本越低
- 但过高 replay ratio（>5）会导致局部多样性丧失，影响性能

#### 🔹 **off-policiness 的正则化效应（图12）**
- 即使在无 buffer 设置下，人为延迟权重同步也会引入 off-policiness，并观察到**训练更稳定、峰值 accuracy 更高**
- 说明适度的旧策略样本具有**正则化作用**，防止过拟合

#### 🔹 **输出多样性提升（图3 中）**
- 使用 replay buffer 后，**pass@k（k>1）显著提升**
- 表明模型输出多样性得到更好保持，未出现 collapse 到单一模式的现象

---

## **4. 关键结论和发现**

### **主要发现**
1. ❗ **“必须 on-policy” 是误区**：  
   适度使用 off-policy 数据不仅不会损害性能，反而可通过增加训练分布多样性来**稳定训练过程**。

2. ⚖️ **存在最优权衡点**：  
   最优 buffer 设计取决于推理成本 $\mu$、staleness-induced variance $\sigma(\cdot)$ 和样本相关性 $p$。当 $\mu$ 较大时，应增大 $N/R$ 和 $B/R$。

3. 💡 **Experience Replay 是 compute-efficient RL 的关键组件**：  
   它不仅能大幅降低推理开销（up to 40%），还能通过平滑优化路径提升训练稳定性。

4. 🔄 **解耦生产与消费提升系统效率**：  
   buffer 缓解了异步系统中的调度瓶颈（如 queue full/empty），提升了 GPU 利用率。

---

### **方法的局限性**
- 当前分析集中在 **中小规模模型**（<8B），在超大规模模型（如 70B+）上的效果仍需验证。
- 理论假设（如梯度噪声建模）为简化版本，实际中可能存在更复杂的依赖结构。
- 当前采样策略较简单（uniform / positive-bias），尚未探索更高级的 prioritized replay。
- 对极端稀疏奖励任务的有效性尚不明确。

---

### **未来工作方向**
1. **更智能的采样机制**：
   - 结合 Prioritized Experience Replay（PER）
   - 动态调整 buffer 内容基于 reward、uncertainty 或 difficulty

2. **适配更强的 off-policy 算法**：
   - 探索 Retrace、V-trace 等带重要性采样修正的方法
   - 设计专用于 LLM 的 off-policy correction 层

3. **理论深化**：
   - 分析 finite-time 收敛速率下的绝对最优 batch/buffer 大小
   - 建模 token-level 的策略漂移而非 step-level

4. **扩展至其他场景**：
   - Preference Modeling（如 DPO 的 offline 版本）
   - 多轮对话、Agent Planning 等 long-horizon 任务

---

> **一句话总结**：  
> 本文颠覆了 LLM 强化学习中“必须 on-policy”的教条，证明通过精心设计的 **Experience Replay**，可以在 **不牺牲甚至提升性能的前提下，节省高达 40% 的推理计算资源**，为高效、可持续的大模型训练提供了新范式。

</details>

---

### 6. [Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](https://arxiv.org/abs/2604.08926)

**Authors**: Taojie Zhu, Dongyang Xu, Ding Zou, Sen Zhao, Qiaobo Hao, Zhiguo Yang, Yonghong He  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.08926v1  

#### Abstract
Post-training paradigms for Large Language Models (LLMs), primarily Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), face a fundamental dilemma: SFT provides stability (low variance) but suffers from high fitting bias, while RL enables exploration (low bias) but grapples with high gradi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对大型语言模型（LLMs）在**推理任务后训练阶段**面临的根本性困境：  
- **Supervised Fine-Tuning (SFT)** 虽然训练稳定（低方差），但容易陷入高拟合偏差（high fitting bias），限制了模型对新推理路径的探索能力，尤其在 Out-of-Distribution (OOD) 任务上泛化能力弱。  
- **Reinforcement Learning (RL)** 能通过奖励信号驱动探索，降低偏差，但其梯度估计具有高方差（high gradient variance），导致训练不稳定，尤其在稀疏奖励场景下难以收敛。

现有统一优化策略（如简单加权损失）忽略了 SFT 和 RL 梯度信号之间的**统计冲突**，无法有效平衡偏差与方差。

---

### 提出了什么新方法或新思路
作者提出 **DYPO (Dynamic Policy Optimization)**，一个统一的框架，通过结构性设计缓解 SFT 与 RL 之间的偏差-方差冲突。其核心创新在于：

#### （1）Dynamic Difficulty Grading（动态难度分级）
- 基于一组 rollout 结果（group rollout）将查询动态划分为三类：
  - **Easy**：全部成功 → 忽略，不参与训练（节省计算）
  - **Hard**：全部失败 → 使用 SFT 进行知识注入
  - **Mid**：部分成功 → 启动 RL 探索
- 实现**实例级路由**（instance-level routing），根据不同样本的学习信号可靠性分配最优优化路径。

#### （2）Multi-Teacher Distillation（多教师蒸馏）——用于 Hard 样本
- 针对 Hard 样本，引入多个教师模型（如 DeepSeek-R1、Qwen3-235B）生成多样化推理路径。
- 通过聚合不同教师的输出，**线性减少 idiosyncratic bias**（个体偏差），保留系统性知识，避免单一教师的过拟合。

#### （3）Group Alignment Loss (GAL) —— 用于 Mid 样本
- 在 RL 阶段引入一种基于组内对比的损失函数，显式拉近成功轨迹与失败轨迹的距离。
- 相比标准 GRPO，GAL 显著**降低 RL 梯度方差**，提升训练稳定性。

#### （4）Dynamic Exploitation-Exploration Gating
- 动态调节 SFT（利用）与 RL（探索）的权重，基于奖励反馈自适应切换，实现更高效的训练。

---

### 相比现有方法的优势
| 维度 | 传统方法 | DYPO |
|------|--------|-------|
| 优化方式 | 固定加权 / 两阶段流程（SFT→RL） | 动态路由 + 结构性融合 |
| 偏差控制 | 单一教师 SFT → 高偏差 | 多教师蒸馏 → 线性降偏 |
| 方差控制 | GRPO 等 RL 方法 → 高方差 | GAL 对比机制 → 显著降方差 |
| 样本利用率 | 所有样本同等对待 | 按难度分级处理，聚焦“学习前沿” |

DYPO 不是简单的损失加权，而是从**梯度层面结构性解决统计冲突**，实现了更鲁棒、高效的联合优化。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **In-Distribution (ID) 数学推理基准**：
  - AIME 2024 / 2025
  - AMC
  - MATH-500
  - Minerva
- **Out-of-Distribution (OOD) 泛化任务**：
  - ARC-c（科学常识问答）
  - GPQA-Diamond（博士级跨学科问答）
- **训练数据**：基于 `OpenR1-Math-220k` 子集，提示来自 `NuminaMath 1.5`，并使用 DeepSeek-R1 和 Qwen3-235B-A22B 生成多教师推理轨迹。

---

### 实验设置和评估指标
- **基础模型**：
  - Qwen2.5-Math-7B
  - Qwen3-4B-Base
- **训练配置**：
  - 每个提示生成 8 条 rollout（轨迹）
  - 最大响应长度：8192 tokens
  - 学习率：1e-6
  - 使用 `verl` 框架，`vLLM` 加速推理
  - bfloat16 精度
- **评估指标**：
  - AIME/AMC：pass@32
  - 其他任务：pass@1
  - 温度 = 0.6，启用选项打乱防止数据泄露

---

### 基线方法对比
共四类基线：
1. **标准监督基线**：
   - SFT（vanilla）
2. **零样本 RL 方法**：
   - SimpleRL-Zero, OpenReasoner-Zero, PRIME-Zero, Oat-Zero
3. **多阶段优化方法**：
   - SFT→RL
   - SuperRL, LUFFY, ReLIFT, SRFT, CHORD
4. **统一训练方法**：
   - 强调动态混合策略（如 CHORD）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen2.5-Math-7B）

| 模型 | ID 平均 | OOD 平均 | GPQA-D |
|------|--------|---------|--------|
| SFT | 44.1 | 50.0 | 24.7 |
| RL | 45.2 | 61.4 | 40.4 |
| SFT→RL | 47.7 | 48.3 | 24.2 |
| CHORD | 50.2 | 60.8 | 40.4 |
| **DYPO** | **52.5 (+4.8)** | **61.6 (+13.3)** | **41.4** |

> ✅ **平均提升 4.8%（ID）和 13.3%（OOD）**

---

### 与基线方法的对比结果
- **vs SFT**：DYPO 在 Qwen2.5 上平均提升 **+8.4%**，在 Qwen3-4B-Base 上提升 **+18.8%**。
- **vs 零样本 RL**：在 AIME 24/25 上超越 Oat-Zero **+19.4 分**，显示更强稳定性。
- **vs 多阶段管道**：
  - 超越 SRFT（当前 SOTA）**+4.8 分（AIME 25）**
  - 超越 SFT→RL 流程 **+10.8%（Qwen3 结果）**
- **vs 统一方法（CHORD/LUFFY）**：DYPO 在所有任务上均领先，尤其在 OOD 上优势显著。

---

### 消融实验结果（Ablation Study）

| 变体 | AIME 25 (↑) | GPQA-D (↑) | 说明 |
|------|------------|-----------|------|
| +SFT | 22.3 | 24.7 | 基础 |
| +Multi-Teacher | 23.3 | 33.3 | 多教师提升泛化 |
| +RL | 26.6 | 35.4 | 引入探索能力 |
| +Dynamic Grading | 28.7 | 36.4 | 难度分级带来最大增益 |
| **+GAL (DYPO)** | **28.7** | **41.4** | GAL 进一步稳定并提升最终性能 |

> 🔍 **发现**：
> - **Dynamic Difficulty Grading** 是最关键组件，尤其在难任务上提升显著。
> - **GAL** 显著提升 OOD 性能，验证其对梯度稳定性的贡献。
> - 即使使用较弱的 8B 教师模型，DYPO 仍能将 AIME 25 从 22.0 提升至 27.8。

---

## 4. 关键结论和发现

### 主要发现
1. **SFT 与 RL 的梯度存在本质统计冲突**：SFT 低方差高偏差，RL 高方差低偏差，简单加权无法解决。
2. **动态难度分级是高效学习的关键**：只在“Mid”难度样本上进行 RL 探索，可最大化信息利用率，避免在 Hard 样本上无效探索。
3. **多教师蒸馏能线性降低监督偏差**：通过聚合多个教师的推理路径，抵消个体偏差，保留通用解法。
4. **GAL 是有效的方差控制器**：相比 GRPO，GAL 提供有界梯度权重，随训练自动退火，实现更平滑收敛。
5. **DYPO 实现了更优的探索-利用平衡**：训练过程中，Offline Data Ratio 从 1.0 自动降至 ~0.35，表明模型自主减少对教师信号的依赖，但仍保留监督锚点。

---

### 方法的局限性
1. **领域局限性**：目前评估集中在逻辑密集型任务（如数学推理），在开放域任务（如创意写作、闲聊）中的表现尚待验证。
2. **计算开销较高**：
   - 每个提示需生成 8 条 rollout，带来较高的在线采样成本。
   - 相比纯离线方法（如 SFT），样本效率较低。
3. **依赖高质量教师模型**：Multi-Teacher Distillation 的效果受限于教师模型的多样性与质量。

---

### 未来工作方向
- 将 DYPO 扩展到更多模态（如视觉-语言模型）和任务类型（如规划、决策）。
- 探索更高效的 rollout 采样策略，降低计算成本。
- 研究如何在没有强教师的情况下进行自举式多教师构建（self-bootstrapping）。
- 将动态难度分级机制应用于课程学习（curriculum learning）框架中。

---

> 📌 **总结一句话**：  
> DYPO 通过 **Dynamic Difficulty Grading + Multi-Teacher Distillation + Group Alignment Loss** 三重机制，**结构性地解决了 SFT 与 RL 的偏差-方差冲突**，在复杂推理与 OOD 泛化任务上实现了显著且稳定的性能提升，为下一代推理增强型 LLM 提供了一个统一、可扩展的新范式。

🔗 **代码开源地址**：[https://github.com/Tocci-Zhu/DYPO](https://github.com/Tocci-Zhu/DYPO)

</details>

---

### 7. [Integrated electro-optic attention nonlinearities for transformers](https://arxiv.org/abs/2604.09512)

**Authors**: Luis Mickeler, Kai Lion, Alfonso Nardi, Jost Kellner, Pierre Didier, Bhavin J. Shastri, Niao He, Rachel Grange  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.09512v1  

#### Abstract
Transformers have emerged as the dominant neural-network architecture, achieving state-of-the-art performance in language processing and computer vision. At the core of these models lies the attention mechanism, which requires a nonlinear, non-negative mapping using the Softmax function. However, al...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Integrated Electro-Optic Attention Nonlinearities for Transformers*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Transformer 模型中的 **attention 机制**依赖于 **Softmax** 这一非线性函数进行归一化。尽管 Softmax 在总计算量（FLOPs）中占比不足 1%，但由于其在 GPU 上依赖低吞吐的 **Special Function Units (SFUs)** 执行指数运算，导致其成为推理延迟的关键瓶颈。例如，在 GPT-2 中，Softmax 占据了高达 **22% 的执行时间**。

### 提出了什么新方法或新思路
本文提出利用 **薄型铌酸锂（TFLN）Mach-Zehnder 调制器（MZM）** 作为模拟非线性计算单元，实现两种新型电光非线性函数：
- **Optmax**：基于 MZM 的上升沿和下降沿分别逼近 Softmax 的指数项和归一化倒数项。
- **Optmoid**：利用 MZM 的完整正弦响应来逼近 Sigmoid 函数，用于替代 Softmax 的 **Sigmoid attention** 变体。

该方法采用 **混合数字-模拟架构**，将 MZM 集成在传统数字硬件旁（co-packaged），仅将非线性部分交由光学器件处理，其余线性计算仍由电子单元完成。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **延迟** | Optmax 和 Optmoid 的延迟相比现有硬件方案降低 **一个到两个数量级**（见 Table I）。 |
| **能效** | 在 10 GBaud 下，Optmax 和 Optmoid 的能量消耗分别为 10 pJ 和 4.7 pJ，具备高能效潜力。 |
| **量化鲁棒性** | 在 **4-bit 输入输出量化** 下仍保持竞争力，甚至在 GPT-2 中优于数字 Softmax。 |
| **系统复杂性** | 不需要光放大、光查找表或多级光电转换，避免了微环谐振器（micro-ring）等方案的可扩展性问题。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **计算机视觉任务**：
  - **MNIST**, **CIFAR-10**, **SVHN**：用于 Vision Transformer (ViT) 图像分类任务。
- **自然语言处理任务**：
  - **FineWeb-Edu**：用于 GPT-2 的因果语言建模（Causal Language Modeling, CLM）。

### 实验设置和评估指标
| 项目 | 设置 |
|------|------|
| **模型架构** | ViT（6 层，8 头）、GPT-2（124M 参数） |
| **注意力模块替换** | 将标准 Softmax / Sigmoid 替换为 Optmax / Optmoid，其他部分不变 |
| **量化设置** | 模拟 DAC/ADC 量化至 4-bit、8-bit、16-bit，内部模拟计算保持连续 |
| **噪声建模** | 注入加性高斯噪声 $ \mathcal{N}(0, \sigma) $，测试模型鲁棒性 |
| **训练方式** | 使用拟合后的 MZM 响应函数进行可微分前向传播，支持反向传播 |
| **评估指标** |
| - ViT：**测试准确率（Test Accuracy）**<br>- GPT-2：**测试损失（Test Loss）**<br>- 延迟与能耗：基于物理建模估算 |

### 基线方法对比
- **数字基线**：
  - **Softmax**（标准）
  - **Sigmoid**（无归一化的 element-wise 替代）
- **硬件加速基线**（Table I）：
  - nMOS SMA (Analog)
  - Softermax (Digital)
  - SOFTONIC (Photonic)
  - VEXP (RISC-V 扩展)

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 图像分类（ViT on CIFAR-10）
| 方法 | 测试准确率（4-bit） | 测试准确率（FP32） |
|------|---------------------|--------------------|
| Softmax | 76.3% | 79.0% |
| Optmax | **74.6%** | 78.5% |
| Sigmoid | 75.9% | 80.5% |
| Optmoid | **69.9%** | 79.5% |

> Optmax 在 4-bit 下仅比 Softmax 低 1.7%，表现稳健；Optmoid 对量化更敏感。

#### 因果语言建模（GPT-2 on FineWeb-Edu）
| 方法 | 测试损失（4-bit） | 测试损失（FP32） |
|------|------------------|------------------|
| Softmax | 5.97 | 4.07 |
| Optmax | **5.85** | 4.08 |
| Sigmoid | 5.97 | 4.18 |
| Optmoid | **5.89** | 4.22 |

> **Optmax 在 4-bit 下反而优于 Softmax**，表明其对量化具有更强的鲁棒性。

### 与基线方法的对比结果（Table I）
| 架构 | 延迟 (s) | 能量 (J) |
|------|--------|--------|
| Softermax (Digital) | 7.7e-4 | 1.3e-8 |
| SOFTONIC (Photonic) | 1.7e-5 | 4.5e-11 |
| **Optmax (This Work)** | **1.3e-8** | 1.0e-8 |
| **Optmoid (This Work)** | **6.5e-9** | 4.7e-9 |

> Optmax 延迟比 SOFTONIC 快 **约 2600 倍**，比 Softermax 快 **约 6 万倍**。

### 消融实验结果
- **量化敏感性分析**：
  - Optmax 在 4-bit 下性能下降较小，而 Optmoid 因偏置参数 $ b = -4.16 $ 导致大量激活值被截断为零，造成信息丢失。
- **噪声鲁棒性分析**：
  - 在 **无噪声训练、有噪声测试** 场景下，4-bit 模型在 $ \sigma > 0.02 $ 时性能急剧下降。
  - 若在训练中引入噪声（noise-aware training），模型鲁棒性显著提升，甚至出现性能反弹（如 Optmax 在 $ \sigma=0.1 $ 下从 75.6% 提升至 77.8%）。
- **乘性 vs 加性噪声**：
  - 模型对 **乘性噪声** 更鲁棒，因其不会激活本应为零的权重。
  - 加性噪声易使接近零的输出越过量化阈值，破坏稀疏性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **TFLN MZM 可高效实现 Transformer 中的非线性函数**，特别是 Softmax 和 Sigmoid，且无需复杂的全光网络。
2. **Optmax 和 Optmoid 在真实任务中保持高度功能保真度**，即使在 4-bit 量化和高速（10 GBaud）条件下仍具竞争力。
3. **模拟非线性在量化下可能优于数字实现**，尤其是在 GPT-2 中，Optmax 在 4-bit 下实现了更低的测试损失。
4. **延迟可大幅压缩**，相比现有硬件方案降低 1–2 个数量级，具备部署潜力。
5. **噪声是主要挑战**，尤其是加性噪声与量化交互会显著退化性能，但可通过 **noise-aware training** 缓解。

### 方法的局限性
- **动态范围受限**：MZM 的正弦响应有界，无法完全复现指数函数的无限动态范围。
- **噪声敏感**：当前系统在高加性噪声下性能下降快，尤其在低比特量化时。
- **校准依赖性强**：输入范围、偏置参数需针对不同序列长度和任务进行调优。
- **硬件集成尚未实现端到端**：当前为概念验证，DAC/ADC/激光等组件仍为分立器件。

### 未来工作方向
- 开发 **抗噪训练策略**，将噪声建模嵌入训练流程，提升实际部署鲁棒性。
- 探索其他天然非线性器件，如 **electro-absorption modulators** 或 **CMOS subthreshold 特性**。
- 实现 **单片集成** 的 TFLN + 电子电路，减少接口损耗与延迟。
- 扩展至更多非线性函数（如 GeLU）和更大规模模型（如 Llama）的验证。
- 研究 **multiplicative noise-aware 架构设计**，以匹配光子系统的物理特性。

--- 

> ✅ **总结一句话**：  
> 本文提出了一种基于 **TFLN MZM** 的电光非线性计算单元 **Optmax/Optmoid**，可在保持高精度的同时，将 Transformer 注意力机制中的非线性计算延迟降低 **超过两个数量级**，为下一代高速、低功耗 AI 硬件提供了新路径。

</details>

---

### 8. [Attention-Based Sampler for Diffusion Language Models](https://arxiv.org/abs/2604.08564)

**Authors**: Yuyan Zhou, Kai Syun Hou, Weiyu Chen, James Kwok  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.08564v1  

#### Abstract
Auto-regressive models (ARMs) have established a dominant paradigm in language modeling. However, their strictly sequential decoding paradigm imposes fundamental constraints on both inference efficiency and modeling flexibility. To address these limitations, diffusion-based large language models (dL...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Attention-Based Sampler for Diffusion Language Models**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

当前的 **Diffusion Large Language Models (dLLMs)** 虽然支持并行解码，提升了推理效率，但其主流的解码顺序选择策略（如基于置信度、熵、margin 的 token-level greedy search）存在以下问题：

- **缺乏理论支撑**：这些方法基于局部输出概率空间进行贪心选择，未从全局序列似然（log-likelihood）最大化的角度进行建模。
- **次优解码路径**：仅依赖 token 级别的不确定性信号，忽略了序列内部的结构性依赖关系，导致生成质量下降。

因此，本文聚焦于一个核心问题：  
> **如何选择最优的解码顺序以最大化目标序列的 log-likelihood？**

---

### ✅ 提出了什么新方法或新思路

作者提出了一种**基于注意力机制的解码顺序选择算法——Attn-Sampler**，其核心思想是：

- 将解码顺序的选择形式化为一个优化问题，目标是最小化“**Permutation Dependency Gap (PDG)**”——即实际排列因子分解与理想全上下文预测之间的 log-likelihood 差距。
- **理论证明**：在单层 Transformer 模型假设下，该差距的上界与注意力矩阵列和（column sum of attention matrix）直接相关。
- **关键结论**：按 token 的 **总注意力得分（total attention score）降序** 进行解码，可近似最小化 PDG 上界，从而提升生成质量。

由此提出的 **Attn-Sampler** 是一种无需训练的（training-free）解码算法，具有以下创新设计：

1. **Attention-Guided Decoding**：
   - 利用模型自注意力矩阵的列和作为 token 重要性的代理指标。
   - 动态决定每一步应优先解码的 token。

2. **Block Attention Approximation**：
   - 避免完整计算 $n \times n$ 注意力矩阵，采用分块（block-wise）方式计算注意力得分，兼容 FlashAttention 等高效内核。

3. **Dynamic Attention Thresholding（动态注意力阈值）**：
   - 在并行解码中，结合预测概率阈值与动态调整的注意力阈值，筛选出“高置信且高注意力”的 token 同时解码，兼顾速度与准确性。

---

### ✅ 相比现有方法的优势

| 维度 | Attn-Sampler | 现有方法（如 Confidence/Entropy/Margin Sampling） |
|------|---------------|---------------------------------------------|
| **理论基础** | 有明确的 log-likelihood 最大化动机，提供理论保证 | 缺乏与似然优化的直接联系，多为启发式规则 |
| **信息来源** | 利用隐藏层注意力结构（global structure） | 仅使用输出层概率分布（local signal） |
| **灵活性** | 支持串行与并行解码，动态控制并行度 | 多为静态并行策略（如固定 top-k 或阈值） |
| **实现成本** | 完全无需额外训练，即插即用 | 通常也无需训练，但部分需调参 |

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集

实验涵盖两大类典型语言任务基准：

- **数学推理（Mathematical Reasoning）**：
  - **GSM8K**：小学数学应用题
  - **MATH**：复杂数学问题
- **代码生成（Code Generation）**：
  - **HumanEval**：函数级代码生成
  - **MBPP**：面向编程任务的 Python 代码生成

---

### ⚙️ 实验设置和评估指标

- **模型平台**：
  - `Fast-dLLM v2`（1.5B 和 7B 参数）
  - `LLaDA-1.58B`

- **硬件环境**：
  - 单张 NVIDIA A6000 GPU

- **评估指标**：
  - 主要指标：各任务上的准确率（Accuracy）
  - 综合指标：平均得分（Avg.）
  - 效率指标：吞吐量（Throughput, Tokens Per Second, TPS）

- **解码配置**：
  - 并行模式中，概率阈值设为 0.9
  - 块大小（block size）默认为 8

---

### 🔁 基线方法对比

对比了多种先进的 dLLM 解码策略：

| 方法 | 类型 | 特点 |
|------|------|------|
| **Top-1 Confidence** | Token-level Greedy | 选最高预测概率 token |
| **Margin Sampler** | Token-level Greedy | 选 top1 与 top2 概率差最大者 |
| **Entropy Sampler** | Token-level Greedy | 选预测熵最小（最确定）token |
| **Fast-dLLM** | Adaptive Threshold | 基于置信度动态并行 |
| **EB-Sampler** | Entropy-Bounded | 基于熵限制解码集合 |
| **KLASS Sampler** | KL-Divergence Guided | 使用 KL 散度衡量变化程度 |

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（来自 Table 1）

| 模型 | 方法 | GSM8K | MATH | HumanEval | MBPP | **Avg** |
|------|------|--------|-------|------------|--------|---------|
| **Fast-dLLM v2 7B** | Confidence | 82.71 | 50.96 | 54.27 | 30.69 | 54.66 |
| | **Attn-Sampler (Seq)** | **84.00** | **52.50** | **57.93** | **36.24** | **57.67** |
| | **Attn-Sampler (Par)** | 84.23 | 51.88 | 58.54 | 35.98 | **57.66** |
| **LLaDA-1.58B** | Best Baseline (Confidence) | 74.75 | 39.64 | 40.85 | 48.68 | 50.98 |
| | **Attn-Sampler (Seq)** | **74.98** | **39.66** | **43.29** | **53.44** | **52.84** |
| | **Attn-Sampler (Par)** | 74.91 | 39.62 | 42.68 | 53.97 | **52.80** |
| **Fast-dLLM v2 1.5B** | KLASS | 61.64 | 32.22 | 37.20 | 27.78 | 39.71 |
| | **Attn-Sampler (Seq)** | **62.70** | **32.56** | **39.63** | **30.16** | **41.26** |

> ✅ 所有模型规模下，**Attn-Sampler 均显著优于所有 baseline**，平均提升约 **1–3 个百分点**，尤其在 HumanEval 上增益明显（+2.44%）。

---

### ⏱️ 推理速度与精度权衡（Figure 2a）

- **Attn-Sampler 实现更优的 Pareto frontier**：
  - 在相同吞吐量（95 TPS）下，**Attn-Sampler 达到 84.2% 准确率**，而 Fast-dLLM 仅为 82.1%。
  - 可达到 **107 TPS（3.06× 加速）**，仍保持 82.6% 准确率，接近 confidence baseline 的精度，但速度快三倍。
- **KLASS** 虽达 83.2% 精度，但吞吐仅 51 TPS，效率远低于 Attn-Sampler。

> 结论：**Attn-Sampler 成功平衡了高质量生成与高吞吐推理的需求**。

---

### 🔍 消融实验结果

#### （1）Dynamic Attention Thresholding vs. 固定策略（Figure 2b）

| 方法 | 最高吞吐 | 对应精度 | 性能下降趋势 |
|------|----------|-----------|----------------|
| **Top-k (k=2,3,4)** | 124 TPS | 65.35% ↓ | 严重退化（↓15.77%） |
| **Static Threshold (0.8~1.0)** | ~120 TPS | <70% | 明显下降 |
| **Attn-Sampler (动态)** | **118 TPS** | **81.35%** | 几乎无损 |

> ✅ 动态阈值机制能自适应调节并行强度，在高速下仍保留关键语义信息。

#### （2）注意力层数与头数的影响（Figure 3）

- **层数越多越好**：
  - 仅用第1层 → 82.26%
  - 使用全部 28 层均值 → **84.23%**
- **头数越多越好**：
  - 仅用第1个头 → 83.32%
  - 使用全部 28 个头 → **84.23%**

> ✅ **聚合所有层与所有头的注意力信息对性能至关重要**，说明高层语义与分布式表示共同作用。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **注意力矩阵列和是解码顺序的理想指导信号**：
   - 理论上可近似最小化 log-likelihood gap。
   - 实践中显著优于传统 token-level greedy 方法。

2. **Attn-Sampler 是一种高效、通用、无需训练的新范式**：
   - 可无缝集成到现有 dLLM 架构中。
   - 兼顾生成质量和推理效率。

3. **全局结构信息优于局部概率信号**：
   - 注意力机制蕴含的 token 间依赖关系比输出概率更能反映真实重要性。

4. **动态并行策略优于静态策略**：
   - 固定 top-k 或阈值无法适应不同扩散阶段的语义需求。
   - 动态注意力阈值可根据上下文灵活调整。

---

### ⚠️ 方法的局限性

- **依赖注意力矩阵稳定性假设**（Assumption 3.1）：
  - 假设在一个 block 内注意力不变，可能在深层模型或多步去噪中不完全成立。
- **对 multi-head/multi-layer 的处理为简单平均**：
  - 是否存在更优的注意力融合方式有待探索。
- **目前仅适用于 absorbing kernel 类型的 dLLM**：
  - 对 uniform kernel 或其他变体的适用性未验证。

---

### 🔮 未来工作方向

1. **扩展至其他生成任务**：
   - 如图像、音频等领域的 diffusion model。
2. **结合强化学习或可微搜索优化解码路径**：
   - 将 Attn-Sampler 作为初始化，进一步微调顺序策略。
3. **探索更精细的注意力聚合机制**：
   - 如加权融合、门控机制、跨层传播等。
4. **降低额外计算开销**：
   - 设计轻量化 attention estimation 模块，适配更大 block size。

---

> 💡 **一句话总结**：  
> **Attn-Sampler 首次将注意力机制与 dLLM 解码顺序选择建立理论联系，提出了一种无需训练、高效准确的新型采样器，为 diffusion language modeling 提供了新的理论视角与实践标准。**

🔗 代码已开源：[https://github.com/YuyanZhoul/Attn_Sampling](https://github.com/YuyanZhoul/Attn_Sampling)

</details>

---

### 9. [Think Less, Know More: State-Aware Reasoning Compression with Knowledge Guidance for Efficient Reasoning](https://arxiv.org/abs/2604.09150)

**Authors**: Yi Sui, Chaozhuo Li, Dawei Song  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.09150v1  

#### Abstract
Large Reasoning Models (LRMs) achieve strong performance on complex tasks by leveraging long Chain-of-Thought (CoT), but often suffer from overthinking, leading to excessive reasoning steps and high inference latency. Existing CoT compression methods struggle to balance accuracy and efficiency, and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Think Less, Know More: State-Aware Reasoning Compression with Knowledge Guidance for Efficient Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型推理模型（**Large Reasoning Models, LRMs**）虽然通过长链式思维（**Chain-of-Thought, CoT**）在复杂任务上表现出色，但普遍存在“**overthinking**”现象：
- 推理步骤冗余、响应延迟高；
- 在不确定或有偏差的状态下反复验证，导致效率低下；
- 缺乏对不同推理阶段动态适应的细粒度压缩机制。

现有 CoT 压缩方法（如 prompt engineering 或 model optimization）难以在**准确性**与**效率**之间取得良好平衡，且无法针对不同推理状态进行自适应调整。

---

### 🚀 提出的新方法：STACK
作者提出 **State-Aware Reasoning Compression with Knowledge Guidance (STACK)**，一个统一的推理压缩框架，具备以下三大创新：

#### 创新点 1：**状态感知的动态压缩策略切换**
- 通过监控每一步的**local information entropy**检测“犹豫状态”（hesitation state），即模型不确定性高的时刻。
- 动态切换两种压缩策略：
  - **Knowledge-Guided Compression**：在不确定/有偏时引入外部知识引导推理方向；
  - **Self-Prompted Compression**：在过度冗长但置信度高时自动精简表达。

> ✅ 优势：实现**step-level**的细粒度控制，而非仅基于任务级别的粗略压缩。

#### 创新点 2：**知识引导的对比解码 + 答案收敛早停机制**
- 引入 **Knowledge-Guided Contrastive Decoding (KGCD)**：
  - 将检索到的知识作为事实锚点，在生成分布层面进行对比调节；
  - 抑制无依据扩展，提升推理准确性和紧凑性。
- 设计基于**答案分布收敛**的早停机制（answer-convergence-based early stopping）：
  - 当后续推理不再显著改变答案分布时（KL 散度低于阈值），提前终止；
  - 避免无效的重复验证。

> ✅ 优势：既保证推理完整性，又有效抑制冗余输出。

#### 创新点 3：**奖励差驱动的训练范式（MDPO）**
- 结合 **Proximal Policy Optimization (PPO)** 和 **Direct Preference Optimization (DPO)** 的优点；
- 提出 **Reward-Difference-Driven Training**，将奖励差异 $ \Delta R $ 作为动态 margin 融入 DPO 损失中；
- 模型能学习到“何时压缩更优”，并保持训练稳定性。

> ✅ 优势：避免传统 RL 方法的不稳定性，同时增强策略对压缩收益的敏感性。

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | STACK 的改进 |
|------|--------|---------------|
| Prompt-based | 压缩能力有限，易丢失逻辑完整性 | 动态策略选择，保留关键推理 |
| ConCISE / MuTIS | 离线构建数据，分布固定 | 在线构建对比样本，适应性强 |
| TokenSqueeze | 忽视推理偏差传播 | 引入知识纠正，缓解 hallucination |

---

## 2. 核心实验方法和设置

### 📚 数据集
在三个数学推理基准上进行评估，难度递增：
- **GSM8K**：小学数学应用题，相对简单；
- **MATH500**：高中至大学水平数学问题；
- **AIME24**：奥数级别挑战性题目。

> 所有数据均来自 DeepScaleR 数据集（含 AIME、AMC、Omni-Math 等）。

---

### ⚙️ 实验设置
- **基础模型**：
  - `DeepSeek-R1-Distill-Qwen-1.5B`
  - `DeepSeek-R1-Distill-Qwen-7B`
- **训练配置**：
  - 学习率：2e-6
  - Batch size：64
  - 硬件：4×NVIDIA A100-80GB GPU
  - 框架：PyTorch
- **外部知识来源**：
  - 使用 **Bing Web Search API** 获取 top-5 检索结果；
  - 构建 retrieval-augmented 输入用于 KGCD。

---

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **Accuracy (Acc)** | 最终答案正确率 |
| **Response Length (Len)** | 平均推理 token 数量 |
| **Inference Latency (Lat)** | 推理耗时（秒） |
| **Token Efficiency (TE)** | $ \text{Acc} \times 100 / \text{Len} $，衡量单位 token 的推理效益 |

---

### 🆚 基线方法对比
| 方法 | 类型 | 简要说明 |
|------|------|----------|
| **Original** | 原始长 CoT 模型 | 不压缩，作为性能上限 |
| **Prompt** | Prompt Engineering | 注入轻量提示限制长度 |
| **ConCISE** | Model Optimization | 基于置信度和早停构建压缩数据 |
| **MuTIS** | Online Intervention | 多轮干预采样训练 |
| **TokenSqueeze** | Distribution-Aligned Refinement | 自适应深度选择 + 分布对齐微调 |

---

## 3. 主要实验结果和性能指标

### 📈 性能总览（见 Table 1）
| 方法 | 平均 Acc ↑ | 平均 Len ↓ | 响应长度减少 | 平均 Lat ↓ |
|------|------------|-------------|------------------|--------------|
| Original | — | — | — | 16.14s (1.5B), 13.29s (7B) |
| Prompt | +0.77 | -7.35% | 微弱 | ~1s 减少 |
| ConCISE | +0.83 | -51.5% | 显著 | 9.55s |
| MuTIS | +2.07 | -56.7% | 显著 | 8.83s |
| **STACK (Ours)** | **+4.80** | **-59.9%** | **最优** | **7.23s (1.5B), 6.73s (7B)** |

> ✅ **STACK 在所有模型规模下均实现最佳 accuracy-efficiency 权衡**

---

### 🔬 关键发现
- **精度提升显著**：
  - 在 1.5B 模型上平均提升 **4.8 points**；
  - 在 AIME24 上从 29.4 → 36.7（+7.3 pts）；
- **压缩效果最强**：
  - 平均推理长度减少 **59.9%**；
  - 在 MATH500 上从 4876 → 1783 tokens（减少超 63%）；
- **Token Efficiency 最高**：
  - TE 达到 **12.78–17.00**，远高于其他方法（如 ConCISE 仅 9.86）；

---

### 🔍 消融实验（Ablation Study，见 Table 2）

#### （1）早停机制对比
| 方法 | MATH500 Acc | AIME24 Acc | Len |
|------|-------------|-----------|-----|
| End Signal Only | 86.7 | 31.7 | 3374 |
| Answer Consistency | 93.3 | 57.5 | 1980 |
| **STACK (Answer Convergence)** | **93.5** | **57.4** | **1733** |

> ✅ 表明基于答案分布收敛的早停机制在**不牺牲精度的前提下实现更强压缩**。

#### （2）训练范式对比
| 方法 | MATH500 Acc | Len |
|------|-------------|-----|
| SFT（监督微调） | 92.4 | 3016 |
| SFT + DPO | 93.2 | 1895 |
| **SFT + MDPO (STACK)** | **93.5** | **1733** |

> ✅ 验证了**reward-difference-driven training**的有效性，进一步优化压缩与准确性的权衡。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **状态感知是高效推理的关键**：
   - 不同推理阶段存在异构冗余源（如犹豫 vs 过度自信），需差异化处理；
   - STACK 成功实现了**step-level 的动态策略调度**。

2. **知识引导可纠正偏差、提升压缩质量**：
   - 外部知识不仅用于增强事实性，还能在分布层面指导生成路径；
   - KGCD 机制有效缓解了 long CoT 中的 error accumulation 和 hallucination。

3. **在线对比采样优于离线数据构建**：
   - 相比 ConCISE/MuTIS 等依赖静态数据的方法，STACK 的在线 contrastive sampling 更具适应性，随模型进化持续优化。

4. **reward difference 是有效的训练信号**：
   - 将压缩带来的实际收益量化为 reward gap，并融入 DPO，使模型学会“何时该压缩”。

---

### ⚠️ 局限性
1. **计算开销增加**：
   - 在线检索与对比解码带来额外延迟，训练成本较高；
2. **检索质量不可控**：
   - 外部知识可能不准确或无关，影响引导效果；
3. **模态单一**：
   - 当前仅支持文本知识增强，未整合 symbolic solver、structured DB 或 multi-modal 工具。

---

### 🔮 未来工作方向
- **联合优化 retrieval 与 reasoning 模块**：端到端训练 retrieval policy；
- **多模态知识增强**：结合公式解析器、图表理解等工具；
- **轻量化部署方案**：设计低延迟版本以适用于边缘设备；
- **扩展至非数学领域**：应用于代码生成、法律分析等需要高可靠性推理的任务。

---

## 总结
> **STACK** 是首个将 **state-awareness**、**knowledge guidance** 与 **contrastive decoding** 深度融合的 CoT 压缩框架。它不仅大幅提升了推理效率（**-59.9% 长度**），还反向提升了准确性（**+4.8 pts**），为构建“快而准”的智能推理系统提供了新范式。

</details>

---

### 10. [UIPress: Bringing Optical Token Compression to UI-to-Code Generation](https://arxiv.org/abs/2604.09442)

**Authors**: Dasen Dai, Shuoqi Li, Ronghao Chen, Huacan Wang, Biao Wu, Qizhen Lan  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.09442v1  

#### Abstract
UI-to-Code generation requires vision-language models (VLMs) to produce thousands of tokens of structured HTML/CSS from a single screenshot, making visual token efficiency critical. Existing compression methods either select tokens at inference time using task-agnostic heuristics, or zero out low-at...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：UIPRESS: Bringing Optical Token Compression to UI-to-Code Generation

## 1. 论文的主要贡献和创新点

### 解决的问题
UI-to-Code（UI2CODE）任务要求视觉语言模型（VLMs）从单张UI截图生成数千个token的结构化HTML/CSS代码，这对视觉token效率提出了极高要求。现有的视觉token压缩方法存在以下问题：
- **推理时压缩**（如FastV、VisionZip）仅在特征层面进行裁剪或归零，不减少序列长度，无法降低 **prefill latency** 或 **time-to-first-token (TTFT)**。
- **启发式选择策略**（如基于注意力或L2范数）是任务无关的（task-agnostic），未能考虑UI截图中信息密度高度不均的特性（如文本区域重要性远高于背景）。
- **光学压缩**（optical compression）虽在OCR任务中成功（如DeepSeek-OCR），但尚未被引入UI-to-Code领域。

### 提出的新方法：UIPRESS
作者提出 **UIPRESS**（UI Pipeline for Representation-Efficient Screenshot Synthesis），是首个将**编码器侧学习型压缩**（encoder-side learned compression）应用于UI-to-Code任务的方法。

#### 方法核心设计：
- 在冻结的ViT编码器（如Qwen3-VL-8B）和LLM解码器之间插入一个轻量级的 **Optical Compressor**。
- 将原始约6,700个视觉token压缩至固定预算 **K=256**。
- 压缩模块包含三个阶段：
  1. **深度可分离卷积**（depthwise-separable convolutions）：实现空间下采样，保留局部结构先验。
  2. **元素引导的空间重加权**（element-guided spatial reweighting）：利用OmniParser检测UI元素（按钮、文本等），为高信息密度区域分配更高权重。
  3. **Transformer精炼层**（Transformer refinement）：恢复因池化而丢失的长距离依赖关系。

#### 解码器适配机制：
- 引入 **Low-Rank Adaptation (LoRA)** 到LLM解码器的query和value投影层，以桥接压缩后token与原解码器之间的表示差距。
- 整个系统仅增加约 **21.7M** 可训练参数（占8B基础模型的0.26%）。

### 相比现有方法的优势
- **真正减少序列长度**：不同于feature-zeroing方法，UIPRESS实际缩短了输入序列，显著降低TTFT和显存占用。
- **任务感知压缩**：通过UI元素检测实现“智能”压缩，优先保留语义关键区域。
- **端到端联合优化**：压缩器与LoRA共同训练，使压缩过程与代码生成目标对齐，实现“超越无损”（beyond lossless）效果。

---

## 2. 核心实验方法和设置

### 数据集
- **Design2Code**：包含485个真实网页截图及其对应HTML代码，用于主实验和验证。
- **WebSight**：包含823K合成的截图-HTML对，从中选取50K样本用于训练UIPRESS，另用100页子集测试跨域泛化能力。

### 实验设置
- **基础模型**：统一使用 **Qwen3-VL-8B-Instruct** 作为基准VLM。
- **训练细节**：
  - 使用50K WebSight样本训练20轮。
  - Optimizer：AdamW，梯度裁剪1.0，权重衰减0.01。
  - 学习率调度：余弦退火，压缩器学习率 $2\times10^{-4}$，LoRA学习率 $2\times10^{-5}$（10:1比例）。
  - 硬件：6×NVIDIA A40 GPU，有效batch size为48。
- **推理设置**：温度0.1，top-p 0.9，最大生成4096个token。

### 评估指标
- **质量指标**：
  - **CLIP Score**：使用ViT-B/32计算原始截图与生成HTML渲染图之间的余弦相似度，衡量全局视觉保真度。
  - 报告Bootstrap 95%置信区间以评估统计显著性。
- **效率指标**：
  - **Time-to-First-Token (TTFT)**：反映prefill延迟。
  - **端到端延迟**（End-to-end latency）。
  - **峰值VRAM占用**。

### 基线方法对比
| 类型 | 方法 | 特点 |
|------|------|------|
| 无压缩 | Qwen3-VL (native) | 原始分辨率，~6,517 tokens |
| 分辨率缩放 | Resolution Scaling (480px) | 输入降采样，控制token数量 |
| 推理时选择 | VisionZip-256 | 基于L2范数选择主导token |
| 元素感知剪枝 | EfficientUICoder (60%) | 基于边缘检测保留重要区域 |
| 特征归零 | FastV (75%) | 注意力得分低的token设为零，序列长度不变 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Design2Code 50页验证集）
| 方法 | Tokens | CLIP Score | TTFT (ms) | 压缩比 |
|------|--------|------------|-----------|--------|
| 无压缩（baseline） | 6,517 | 0.7563 | 384 | 1.0× |
| Resolution Scaling | 845 | 0.7768 | 69 | 7.7× |
| VisionZip-256 | 256 | 0.7333 | 45 | 25.5× |
| UIPRESS-256 (Ours) | **256** | **0.8127** | **42** | **25.5×** |

### 与基线方法的对比结果
- **相比无压缩基线**：
  - CLIP提升 **+7.5%**（0.8127 vs 0.7563），同时实现 **9.1× TTFT加速**（42ms vs 384ms）。
- **相比最强推理时方法**（Resolution Scaling @845 tokens）：
  - CLIP提升 **+4.6%**，且使用更少token（256 vs 845），TTFT更低（42ms vs 69ms）。
- **相比同token数方法**（VisionZip-256）：
  - CLIP提升 **+10.8%**（0.8127 vs 0.7333），证明学习型压缩优于启发式选择。

### 消融实验结果（Table 2）
| 配置 | CLIP Score | Δ |
|------|------------|-----|
| Full UIPRESS (Conv + Pool + Refine + LoRA) | **0.8127** | — |
| -LoRA | 0.7046 | -0.108 |
| -Transformer Refinement | 0.7940 | -0.019 |
| -Depthwise-Sep Conv → Std Conv | 0.8020 | -0.011 |
| LoRA only (no compressor) | 0.7610 | -0.052 |
| Compressor only (no LoRA) | 0.7046 | -0.108 |

> **结论**：LoRA适配是性能提升的最主要因素（贡献+10.8% CLIP），Transformer精炼和深度可分离卷积均有正向贡献，各组件协同作用显著。

---

## 4. 关键结论和发现

### 主要发现
1. **“超越无损”压缩成为可能**：尽管将视觉token从6,700压缩至256（25.5×），UIPRESS反而将CLIP score提升了7.5%，表明**有损压缩可通过去除冗余噪声（如JPEG伪影、抗锯齿）并聚焦结构信号来提升生成质量**。
2. **编码器侧学习压缩优于推理时方法**：UIPRESS在质量与效率上全面超越所有基线，尤其在极端压缩比下优势明显。
3. **解码器适配至关重要**：LoRA是性能提升的关键，说明压缩后的表示需专门适配才能被冻结解码器正确理解。
4. **非均匀信息密度建模有效**：元素引导重加权机制使得模型能自适应地关注文本和交互元素，提升结构保真度。

### 局限性
1. **依赖额外训练数据**：需要约50K标注的截图-HTML对进行训练，在特定领域（如移动端App、Figma设计）应用受限。
2. **未完全复现EfficientUICoder全流水线**：实验仅实现了其输入侧压缩，未包含输出去重模块，可能低估其完整性能。
3. **评估指标局限**：CLIP score衡量全局相似性，缺乏对文本准确率、CSS属性精度等细粒度指标的分析。
4. **固定token预算**：当前使用固定K=256，未根据页面复杂度动态调整token数量。

### 未来工作方向
- 开发**自适应token分配机制**，根据UI复杂度动态决定压缩强度。
- 构建面向**移动App或设计稿**的专用数据集，拓展方法适用范围。
- 引入**细粒度评估指标**，如字符级文本匹配、CSS属性召回率等。
- 探索**无需外部检测器**（如OmniParser）的端到端可微压缩框架。

> ✅ **总结**：UIPRESS首次将光学压缩范式成功迁移到UI-to-Code任务，通过轻量级学习压缩模块与LoRA适配，在大幅降低延迟的同时反超无压缩基线性能，为高效多模态生成提供了新范式。代码与模型已开源。

</details>

---

### 11. [MATCHA: Efficient Deployment of Deep Neural Networks on Multi-Accelerator Heterogeneous Edge SoCs](https://arxiv.org/abs/2604.09124)

**Authors**: Enrico Russo, Mohamed Amine Hamdi, Alessandro Ottaviano, Francesco Conti, Angelo Garofalo, Daniele Jahier Pagliari, Maurizio Palesi, Luca Benini, Alessio Burrello  
**Category**: cs.DC  
**Published**: 2026-04-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.09124v1  

#### Abstract
Deploying DNNs on System-on-Chips (SoC) with multiple heterogeneous acceleration engines is challenging, and the majority of deployment frameworks cannot fully exploit heterogeneity. We present MATCHA, a unified DNN deployment framework that generates highly concurrent schedules for parallel, hetero...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MATCHA: Efficient Deployment of Deep Neural Networks on Multi-Accelerator Heterogeneous Edge SoCs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代边缘系统芯片（Edge SoC）通常集成多个异构加速器（如 SIMD 单元、向量处理器、GEMM 引擎等），以提升 DNN 推理的能效和性能。然而，现有的 DNN 部署框架（如 TVM、MATCH）大多采用**层粒度映射**和**同步串行执行**策略，导致硬件资源利用率低下——同一时间只有一个加速器处于活跃状态，其余空闲。

此外，传统方法难以充分利用多分支网络（如 ResNet、Transformer）中的并行性，也无法在算子内部进行细粒度任务划分以适配异构设备。

### 🚀 提出的新方法与创新思路
本文提出 **MATCHA**，一个面向无操作系统（OS-less）、多加速器异构边缘 SoC 的统一 DNN 部署框架，其核心创新如下：

- **统一的并发调度架构**  
  支持跨多个异构加速器的**异步、并行执行**，打破传统部署中“单层 → 单设备”的串行模式。

- **基于 Constraint Programming (CP) 的联合优化引擎**  
  将 **pattern matching、tile 分配、device 映射、memory 规划和 scheduling** 统一建模为一个约束优化问题，目标是最小化端到端推理延迟（makespan）。

- **Tile-Centric 异构模式匹配（Tile-Centric Pattern Matching）**  
  允许将单个算子切分为多个 tile，并将不同 tile 分配给最适合处理该部分工作的设备。例如，卷积层可按输出特征图行分块，全连接层可按输出神经元维度分块。

- **支持细粒度内存规划与异步数据传输**  
  利用 DMA 引擎实现 L2/L3 内存间的数据搬运，并将其纳入调度模型；支持静态分配、动态换出（swapping）和按需加载（planned loading）等多种内存策略。

### 🔍 相比现有方法的优势
| 特性 | MATCHA | MATCH / TVM | 说明 |
|------|--------|------------|------|
| 并发执行 | ✅ | ❌ | 多设备可同时运行 |
| 异步调度 | ✅ | ❌ | 不依赖主机轮询控制 |
| Tile-level 并行 | ✅ | ❌ | 支持算子内部分片 |
| 异构设备协同 | ✅ | ⚠️（仅 layer-level） | 可混合使用不同类型加速器 |
| 内存感知调度 | ✅ | ⚠️（有限支持） | 联合优化 memory 和 compute |

> MATCHA 是首个支持 **OS-less 异构 SoC 上 tile-level 并行 + 异步执行** 的开源编译器。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集与模型
- **MLPerf Tiny Benchmark** 中的四个典型轻量级 DNN 模型：
  - AutoEncoder（用于异常检测）
  - DS-CNN（深度可分离卷积为主）
  - MobileNet（MobileNetV1 结构）
  - ResNet-18（CIFAR-10 训练）
- 微基准测试（Microbenchmarks）：
  - ResNet-50 第一个残差块
  - ResNeXt-50 第一个 block
  - Transformer Encoder Layer（hidden size=128）

### 💻 实验平台
- **硬件平台**：开源的 **Carfield HSoC**，部署于 Xilinx VCU118 FPGA，主频 50MHz。
- **SoC 架构组成**：
  - **Host Domain**：双核 RV64GCH RISC-V CPU（Device 0），带 1MiB 动态配置 L2 SPM。
  - **Accelerator Domain**：
    - **PULP Cluster**（Device 1）：8 个 RI5CY 核心，256KiB L1 SPM，支持 FP16。
    - **Spatz Cluster**（Device 2）：2 个 RISC-V 标量核心 + 2 个 RVVU 向量单元，128KiB L1 SPM，支持 FP8–FP64、bfloat16、整数及混合精度运算。
- **通信机制**：通过 PLIC 和 mailbox 实现中断驱动的异步任务通知。

### 🎯 评估指标
- **端到端推理延迟（Latency）**
- **总周期数（Cycles）**
- **FLOPS（每秒浮点操作数）**
- **加速器利用率（Utilization）**
- **内存占用与调度可行性**

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **TVM (Host-only)** | 所有计算在主机 CPU 上执行 |
| **MATCH** | 层粒度映射到最优设备，**串行执行** |
| **MATCHA (No Tiling)** | 关闭 tile 分割，仅启用异步层卸载 |
| **MATCHA (Ours)** | 完整版本：启用 tile 分割 + 异步并行执行 |

所有方法使用相同的 pattern 库和 FP16 数据精度。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2 和 Figure 7）

#### ✅ MLPerf Tiny 性能对比（vs MATCH）

| Model | MATCHA vs MATCH 延迟降低 |
|-------|--------------------------|
| AutoEncoder | **33.3%** ↓ |
| DS-CNN | 0% （无改善） |
| MobileNet | 0% （无改善） |
| ResNet-18 | **28.8%** ↓ |

> - **AutoEncoder**：由全连接层链构成，缺乏图级并行性，但通过 **output-neuron 维度分 tile** 实现负载均衡，获得显著收益。
> - **DS-CNN / MobileNet**：以 depthwise convolution 为主，计算密度低，**slice/concat 开销超过并行增益**，故未提速。

#### ✅ Microbenchmarks 加速比（vs TVM Host-only）

| Block | Speedup |
|-------|---------|
| ResNet-50 Block | **40.34×** |
| ResNeXt-50 Block | **11.04×** |
| Transformer Encoder | **~30×**（文中未精确给出，图示约 3e8 FLOPS） |

#### ✅ 相对于 MATCH 的延迟改进（Microbenchmarks）

| Block | MATCHA vs MATCH 延迟降低 |
|-------|----------------------------|
| ResNet-50 Block | **35.02%** ↓ |
| ResNeXt-50 Block | **17.55%** ↓ |
| Transformer Encoder | **23.65%** ↓ |

> 图 7 显示，MATCHA 在各类构建块上均达到最高 FLOPS，表明其有效提升了硬件吞吐率。

#### ✅ 消融实验结果
- **仅启用异步执行（No Tiling）**：
  - 对 ResNet 类模型（含分支结构）带来约 **13–18%** 的延迟下降（利用 graph-level 并行）。
- **启用 tile-level 并行后进一步优化**：
  - 在 ResNet-18 上额外带来 **15.5%** 的延迟下降（从 13.3% → 28.8%）。
  - 表明 **tile-level 调度对负载平衡至关重要**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Tile-Centric 并行是提升异构 SoC 利用率的关键**  
   即使在网络结构简单（如 AutoEncoder）或无天然并行分支的情况下，也能通过算子内部分 tile 实现高效负载分配。

2. **异步执行显著减少空等时间**  
   通过事件/中断机制解耦主机与加速器，避免轮询等待，实现真正的并发执行。

3. **异构加速器需联合优化 mapping 与 memory**  
   单纯依据峰值性能选择设备（如 MATCH）无法保证最优，必须结合实际 workload 分布、内存容量和数据移动开销进行全局优化。

4. **并非所有网络都能从 tiling 中受益**  
   对于 depthwise conv 等低计算强度操作，辅助操作（slice/concat）的开销可能抵消并行带来的好处，需智能决策是否分 tile。

### ⚠️ 方法的局限性
1. **当前流程拆分为两阶段优化**  
   - 先做高层 pattern matching 和 tile 分配（CP）
   - 再做底层 device-specific mapping（ZigZag）
   - 缺乏端到端联合搜索，可能导致次优解。

2. **Helper Operator 开销未被准确建模**  
   - `slice`、`concat` 等操作的延迟未纳入 CP 成本函数，影响调度准确性。
   - 未来可通过 view-based tiling 或连续地址布局消除这些操作。

3. **DMA 传输未重叠计算**  
   当前模型中 DMA 传输被序列化处理，未能与计算重叠，限制了进一步加速潜力。

4. **未评估能耗表现**  
   虽然强调能效场景，但实验仅报告延迟和 FLOPS，缺少功耗或能量消耗数据。

### 🔮 未来工作方向
1. **构建统一的成本模型**  
   联合优化 pattern matching、per-device mapping 和 low-level tiling，引入硬件闭环校准参数。

2. **优化 slice/concat 等辅助操作**  
   通过内存视图（view-based）或地址规划消除冗余操作，降低运行时开销。

3. **支持 DMA 与计算重叠**  
   修改调度模型，允许 DMA transfer 与 kernel execution 并发，进一步压缩 makespan。

4. **扩展至更多异构设备组合**  
   如 NPU + GPU + FPGA clusters，验证通用性和可扩展性。

5. **加入能耗建模与优化目标**  
   在 CP 求解中引入 energy-aware objective，支持低功耗优先的部署策略。

---

> **总结一句话**：MATCHA 通过 **tile-centric constraint programming 优化框架**，实现了 DNN 在多异构加速器上的**高并发、异步执行**，在真实边缘 SoC 上相较 MATCH 最多降低 **35% 推理延迟**，为未来高效边缘 AI 部署提供了新范式。

</details>

---

### 12. [MT-OSC: Path for LLMs that Get Lost in Multi-Turn Conversation](https://arxiv.org/abs/2604.08782)

**Authors**: Jyotika Singh, Fang Tu, Miguel Ballesteros, Weiyi Sun, Sandip Ghoshal, Michelle Yuan, Yassine Benajiba, Sujith Ravi, Dan Roth  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.08782v1  

#### Abstract
Large language models (LLMs) suffer significant performance degradation when user instructions and context are distributed over multiple conversational turns, yet multi-turn (MT) interactions dominate chat interfaces. The routine approach of appending full chat history to prompts rapidly exhausts co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《MT-OSC: Path for LLMs that Get Lost in Multi-Turn Conversation》核心总结

## 1. 主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在多轮对话（multi-turn conversation）中表现显著下降，主要原因包括：
- **信息碎片化**：用户指令和关键上下文分散在多个对话轮次中，导致模型难以准确回忆和整合重要细节。
- **上下文窗口压力**：常规做法是将完整聊天历史附加到提示（prompt）中，这迅速耗尽 context window，增加延迟、计算成本，并因信息冗余而降低性能。

### 提出的新方法：MT-OSC
作者提出 **MT-OSC (One-off Sequential Condensation)** 框架，一种高效、自动的对话历史压缩机制，其核心思想是在后台动态压缩聊天历史，而非简单拼接全部历史。

#### 创新点
- **任务无关的压缩框架**：MT-OSC 是一个通用、可插拔的模块，无需对 LLM 进行微调或修改模型架构，易于集成到任何基于 LLM 的聊天系统中。
- **Condenser Agent 架构**：
  - **Condenser**：基于 few-shot 推理的 LLM Agent，通过精心设计的示例（exemplars）学习如何保留关键信息（如原始用户指令、数值、否定词等），并生成高质量的压缩摘要。
  - **Decider**：轻量级、可配置的决策组件，通过量化助手回复中的冗余度和新颖性，决定是否对当前对话段执行压缩，避免在信息密集型对话中丢失关键内容。
- **One-off Sequential Application**：采用滑动窗口策略（每 w 轮压缩一次），逐步替换旧的历史记录，确保始终维护一个紧凑且语义丰富的上下文表示。

### 相比现有方法的优势
| 方法 | 局限性 | MT-OSC 的优势 |
|------|--------|----------------|
| **Concatenation (直接拼接)** | 上下文线性增长，不可持续 | 显著减少 token 数量（最高达 72%） |
| **Ad-hoc Summarization (简单摘要)** | 风险遗漏关键细节，缺乏鲁棒性 | Few-shot Condenser 更精准保留关键信息 |
| **Fine-tuning / Memory Compression** | 需要训练，扩展性差 | 无需训练，即插即用 |
| **Retrieval-Augmented Methods** | 增加系统复杂性和延迟 | 无实时检索开销，纯前向处理 |

---

## 2. 核心实验方法和设置

### 数据集
实验覆盖 **10 个多样化、state-of-the-art 的 multi-turn 数据集**，分为两类：

#### Sharded Datasets（分片静态基准）
将单轮任务人为拆分为多轮，模拟真实对话中的信息碎片化：
- **GSM8K**（数学推理）
- **BFCL-V3 Parallel**（函数调用）
- **HumanEval (HEval)**（代码生成）
- **Spider**（Text-to-SQL）
- **ToTTo**（表格到文本生成）
- **Summary of Haystack (SoH)**（长文档摘要）

#### MT-EVAL Datasets（自然对话流）
来自 MT-EVAL 基准，反映真实对话的不同方面：
- **Recollection**（回忆）
- **Refinement**（精炼）
- **Expansion**（扩展）
- **Follow-up**（跟进）

> 所有样本包含 4–12 个用户回合，远超以往研究范围。

### 实验设置
- **主干模型**：默认使用 `Llama-3.3-70B-Instruct` 作为对话模型（chat model）和 Condenser。
- **窗口大小（w）**：测试了 w=2, 3, 4 的效果。
- **运行方式**：MT-OSC 在后台异步运行，不影响用户体验和响应延迟（TTFT）。

### 评估指标
根据不同任务类型采用相应指标：
- **Accuracy**：用于代码与数学类任务（HumanEval, BFCL, GSM, Spider, Recollection+）
- **LLM-as-a-Judge Rating (10分制)**：用于开放式对话任务（Refinement, Follow-up, Expansion+）
- **BLEU Score**：用于 ToTTo
- **Composite Joint-Score F1**：用于 SoH

此外，还引入 **LLM Judge** 对模糊情况（如 GSM 最终答案位置不固定）进行二次验证，提升评估可靠性。

### 基线方法对比
- **MT-baseline**：将所有历史轮次直接传入模型（标准做法）
- **Simple Summarization (Summ)**：使用简化提示进行摘要，无 few-shot 示例（用于消融实验）
- **RECAP / SNOWBALL / CONCAT**：来自 Laban et al. (2025) 的基线方法（文中指出其实际部署不可行）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **平均 token 减少率** | **30.9%**（所有样本）<br>**45.5%**（≥6 轮对话）<br>**最高达 72%**（10 轮对话） |
| **端到端延迟降低** | 平均节省 ~1,789 tokens/轮 → **约 1–1.2 秒延迟减少**（基于 TTFT 与 token 数的线性关系） |
| **总体性能变化** | 性能保持稳定甚至提升，**统计上无显著差异**（Wilcoxon test, p=0.19） |

### 与基线方法对比（MT-OSC w=4 vs MT-baseline）
| Dataset | MT-baseline | MT-OSC | Token Reduction |
|--------|-------------|--------|-----------------|
| BFCLshrd | 81.13% | **86.79%** ↑ | 263 → 134 (**-49%**) |
| GSMshrd | 83.45% | **84.80%** ↑ | 1026 → 879 (-14%) |
| HEvalshrd | 74.67% | **77.33%** ↑ | 332 → 194 (-42%) |
| Spidershrd | 76.95% | **79.44%** ↑ | 167 → 167 (=) |
| recollmte+ | 94.22% | 92.22% (~) | 15312 → 8247 (**-46%**) |
| expanmte+ | 8.61 | **7.90** ↓ | 5169 → 4053 (-22%) |

> ✅ 多数任务性能持平或提升，同时 token 显著下降  
> ❗ SoH 和 ToTTo 未降 token 因 Decider 判断需保留全文

### 消融实验结果

#### (1) Decider 组件作用（有 vs 无）
| Dataset | 性能 (有 Decider) | 性能 (无 Decider) | Token Reduction |
|-------|------------------|------------------|-----------------|
| ToTTOshrd | 0.18 | **0.09** ↓ | 4670 → 177 (**-96%**) |
| SoHshrd | 0.13 | **0.08** ↓ | 16063 → 2360 (**-85%**) |
| Refinmtev | 5.35 | **5.18** ↓ | 15486 → 6662 (**-57%**) |

> 🔍 **发现**：移除 Decider 后 token 压缩更强，但性能大幅下降 → 表明 **Decider 成功防止了信息密集对话中的关键信息丢失**

#### (2) Condenser vs Simple Summarization
- 使用简单摘要提示（无 few-shot 示例）会导致：
  - 忽略关键用户指令（如“以字典格式返回”）
  - 错误合并助手假设为事实（见附录 E 示例）
  - 在所有数据集和窗口大小下，性能均低于 MT-OSC
- **证明**：few-shot 设计的 Condenser 对保留细微语义至关重要

#### (3) 跨 13 个 SOTA LLM 的泛化能力
在包括 `GPT-4o`, `GPT-5`, `Llama-4`, `Grok-3/4` 等在内的 **13 个顶级 LLM 上一致提升性能**：
- GPT-5 在 baseline 下最强（90%），但 **MT-OSC 再提升 3.3% 达 92%**
- 所有模型均受益，表明 MT-OSC 具有良好架构无关性

#### (4) 鲁棒性测试（加入噪声）
在三种人工注入噪声场景下测试：
1. **Repetition Infusion**（重复轮次）
2. **Filler Injection**（填充词如 “um”, “well”）
3. **Contextual Diversion**（添加相关但无关内容）

> 📊 **结果**：MT-OSC 不仅维持性能，还在 **Contextual Diversion 场景下实现高达 7% 的相对增益**，表明其能有效过滤干扰信息。

---

## 4. 关键结论和发现

### 主要发现
- ✅ **MT-OSC 显著缓解 multi-turn 性能衰退问题**：通过智能压缩，不仅未损失性能，反而在多数任务上有所提升。
- ✅ **高效且实用**：token 减少最高达 72%，显著降低延迟与成本，适合工业部署。
- ✅ **鲁棒性强**：对噪声、冗余和无关信息具有天然免疫力，优于直接拼接。
- ✅ **通用性强**：适用于多种任务类型、数据集和主流 LLM 架构，无需定制化调整。

### 局限性
- 当前评估基于 **公开的短程对话数据集**（≤12 轮），尚未验证在极长对话（>50 轮）下的累积误差。
- 数据集中 **多话题切换、复杂 agent 行为（tool use）较少**，未充分测试在 agentic workflow 中的表现。
- 依赖外部 LLM 作为 Condenser，虽不增加用户延迟，但仍产生额外 token 开销（实验证明净收益仍为正）。

### 未来工作方向
- 构建更长、更多样化、多主题的真实世界 multi-turn 对话数据集。
- 探索结合 **topical decomposition** 技术，更好处理复杂、多线索对话。
- 将 MT-OSC 思想融入模型预训练阶段，使 LLM 原生具备更强的 multi-turn 上下文管理能力。
- 研究如何在压缩过程中更好地保留 **时序逻辑与因果关系**，进一步提升推理一致性。

</details>

---

### 13. [Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](https://arxiv.org/abs/2604.08964)

**Authors**: Shun Zou, Yong Wang, Zehui Chen, Lin Chen, Chongyang Tao, Feng Zhao, Xiangxiang Chu  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.08964v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have recently become a promising alternative to autoregressive large language models (ARMs). Semi-autoregressive (Semi-AR) decoding is widely employed in base dLLMs and advanced decoding strategies due to its superior performance. However, our observations rev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **Semi-AR decoding 的块边界延迟问题**：在扩散大语言模型（dLLMs）中，广泛使用的 Semi-autoregressive (Semi-AR) 解码策略将序列划分为多个 block，并按顺序逐块解码。这种机制导致许多“跨块稳定 token”（cross-block stable tokens）必须等待其所在 block 被激活后才能被解码，造成不必要的延迟。
- **现有方法不可靠**：以往基于置信度（confidence）或熵（entropy）的提前解码策略容易受到局部波动影响，导致误判和生成质量下降。

### **提出了什么新方法或新思路**
提出 **Anchor-based History-stable Decoding (AHD)**，一种无需训练、即插即用的动态解码策略，核心思想如下：
- **动态锚点监控历史轨迹**：在每一步解码中，以当前预测分布作为“动态锚点”（dynamic anchor），回溯历史缓冲区中的预测轨迹。
- **定义“绝对稳定性趋势”**：通过计算锚定 KL 散度（anchored KL divergence）并加权聚合，构建一个历史一致性得分 $D_{\text{acs}}$，用于捕捉 token 是否进入稳定的收敛阶段。
- **跨块早解锁机制**：一旦某个 future block 中的 token 被判定为已达到“绝对稳定趋势”，即可提前解锁并参与当前步的并行解码，打破 block 边界限制。

### **相比现有方法的优势**
| 维度 | AHD | 现有方法（如 Fast-dLLM, Saber, KLASS） |
|------|-----|----------------------------|
| **是否利用历史信息** | ✅ 显式建模整个历史轨迹 | ❌ 仅依赖单步置信度或相邻步 KL |
| **对局部波动鲁棒性** | ✅ 高（基于趋势而非瞬时值） | ❌ 低（易受噪声干扰） |
| **能否跨 block 提前解码** | ✅ 支持 future block token 提前释放 | ❌ 通常局限于当前 block 内部加速 |
| **是否需要重新训练** | ✅ 完全 training-free | ✅ 多数也是 training-free |
| **性能 vs 效率权衡** | ✅ 同时提升性能与效率 | ⚠️ 多数加速方法会牺牲性能 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
#### **语言领域**
- **代码生成**：`HumanEval`, `MBPP`
- **通用任务**：`BBH`, `MMLU-Pro`, `TruthfulQA`
- **数学推理**：`Math`, `Asdiv`

#### **多模态领域**
- **视觉-语言**：`MATH-Vision`, `MathVista`, `ScienceQA`, `GQA`, `MME`
- **音频-语言**：`VoiceBench` 上的五个子任务（如 `OpenBookQA`, `AlpacaEval`）

### **实验设置和评估指标**
| 类别 | 设置 |
|------|------|
| **模型** | `LLaDA-8B-Instruct`, `LLaDA-1.5`, `MMaDA-8B-MixCoT`, `DIFFA` |
| **生成长度** | 默认 256，部分长序列实验至 1024 或 2048 |
| **Block 长度** | 默认 32，消融实验测试 16–128 |
| **历史缓冲区长度 H** | 默认 6 |
| **一致性阈值 ε** | 默认 0.01 |
| **评估指标** | <ul><li>**性能**：各任务标准指标（Pass@1, Accuracy 等）</li><li>**效率**：Decoding Steps、Latency / Speed-up、TPF（Time Per Forward）</li></ul> |

### **基线方法对比**
- **Vanilla decoding**：原始 Semi-AR 解码
- **PC-sampler**：基于位置感知校准的方法
- **Fast-dLLM**：固定置信阈值的并行解码
- **KLASS**：基于相邻步 KL 引导的快速推断
- **Saber**：自适应加速 + 回溯重掩码机制

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（以 LLaDA-8B-Instruct 为例）**

| 任务 | 方法 | Score ↑ | Steps ↓ | Reduction |
|------|------|--------|---------|----------|
| **BBH** | Vanilla | 53.11 | 256 | — |
| | Fast-dLLM | 53.17 (+0.06) | 55.85 | -78% |
| | Saber | 52.88 (-0.23) | 87.31 | -66% |
| | **AHD (Ours)** | **56.78 (+3.67)** | **51.48** | **-80%** |
| **HumanEval** | Vanilla | 40.85 | 256 | — |
| | Fast-dLLM | 41.46 (+0.61) | 78.36 | -69% |
| | **AHD (Ours)** | **43.29 (+2.44)** | **77.24** | **-70%** |
| **MMLU-Pro** | Vanilla | 35.57 | 256 | — |
| | Saber | 36.10 (+0.53) | 123.83 | -52% |
| | **AHD (Ours)** | **37.42 (+1.85)** | **133.06** | **-48%** |

> ✅ **亮点**：AHD 在所有 7 个 benchmark 上均实现 **性能提升 + 解码步数减少**，而其他先进方法大多只能在性能下降的前提下提速。

### **多模态与音频领域结果**
| 任务 | 方法 | Score ↑ | Speed-up × |
|------|------|--------|------------|
| **MathVista-mini** | MMaDA | 32.90 | 1.00× |
| | **AHD (Ours)** | **36.00 (+3.10)** | **2.37×** |
| **OpenBookQA (VoiceBench)** | DIFFA | 36.50 | — |
| | **AHD (Ours)** | **38.50 (+2.00)** | **-78% 步数** |

> ✅ 表明 AHD 可泛化到 vision-language 和 audio-language 模型，在保持甚至提升性能的同时显著加速。

### **长序列生成（1024 tokens）**
| 方法 | HumanEval Score | Steps |
|------|------------------|-------|
| Vanilla | 43.90 | 1024 |
| Fast-dLLM | 42.68 | 184.79 |
| **AHD (Ours)** | **48.78 (+4.88)** | **146.48 (-86%)** |

> ✅ 长序列下优势更明显，说明 block boundary 延迟随长度增加而加剧，AHD 缓解效果更强。

### **消融实验结果**
| 变量 | 最优配置 | 发现 |
|------|--------|------|
| **历史缓冲长度 H** | H=6 | 性能最佳；H<4 导致趋势捕获不足，H>6 带来边际收益递减 |
| **一致性阈值 ε** | ε=0.02 | 平衡点：太小 → 过于保守；太大 → 提前释放不稳定 token |
| **生成长度** | 128→512 | AHD 的步数压缩比随长度增加而提高（从 -49% 到 -88%） |
| **block 长度** | 16–128 | AHD 在各种 block 配置下均稳定优于 baseline，具有强鲁棒性 |
| **温度系数** | 0.0–1.0 | AHD 在不同 temperature 下均保持增益，显示良好鲁棒性 |

---

## **4. 关键结论和发现**

### **主要发现**
1. **Cross-block stable tokens 普遍存在且重要**：大量 token 在其 block 被激活前早已稳定，延迟解码不仅浪费计算，还会抑制邻近 token 的收敛（radiative effects）。
2. **单步指标不可靠**：confidence 和 entropy 易受局部波动干扰，不能准确判断 token 是否真正稳定。
3. **历史轨迹是关键**：token 稳定性的本质体现在其预测分布的“收敛趋势”上，而非某一时刻的峰值。
4. **AHD 实现双赢**：首次实现了 **性能提升 + 推理加速** 的协同优化，打破了传统“加速必损性能”的困境。

### **方法的局限性**
1. **参数需微调**：虽然 training-free，但最优的 $H$ 和 $\varepsilon$ 在不同任务间略有差异，增加了部署成本。
2. **未验证超大规模模型**：实验集中在 ~8B 参数模型，尚未在 72B 或 256B 级别验证有效性。
3. **额外开销虽小但仍存在**：计算 anchored KL 和维护历史 buffer 引入少量额外内存和延迟（约 1–2%），尽管远小于节省的成本。

### **未来工作方向**
- 探索 **自动化超参调节机制**，降低适配新任务的成本。
- 将 AHD 扩展至 **更大规模模型**（如 72B+）和 **更长序列生成**（>4096 tokens）。
- 结合 **KV Cache 优化技术**（如 dllm-cache, dkv-cache），进一步提升端到端推理效率。
- 探索 **与其他高级解码策略**（如 Saber 的 backtracking）的融合可能性。

---

> 📌 **总结一句话**：  
> AHD 通过引入“动态锚点+历史轨迹分析”，首次实现了对跨块稳定 token 的可靠识别与提前解码，在无需训练的前提下，同时提升了 dLLMs 的 **生成质量** 与 **推理效率**，为下一代高效扩散语言模型提供了新范式。

</details>

---

### 14. [EthicMind: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](https://arxiv.org/abs/2604.09265)

**Authors**: Jiawen Deng, Wei Li, Wentao Zhang, Ziyun Jiao, Fuji Ren  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.09265v1  

#### Abstract
Intelligent dialogue systems are increasingly deployed in emotionally and ethically sensitive settings, where failures in either emotional attunement or ethical judgment can cause significant harm. Existing dialogue models typically address empathy and ethical safety in isolation, and often fail to ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

当前的对话系统在处理**情感敏感与伦理风险并存**的多轮对话时存在显著缺陷：

- **情感对齐（Empathetic Alignment）** 模型往往忽视伦理规范，可能无意中支持有害信念或行为。
- **伦理安全（Ethical Safety）** 模型则倾向于采用刚性的规则过滤或拒绝机制，导致回应冷漠、缺乏共情，损害用户体验和信任。
- 在**多轮交互中**，用户的**情绪状态**和**潜在伦理风险**是动态演变的，而现有模型通常将情感与伦理作为孤立目标处理，无法实现动态适应。

因此，论文旨在解决：如何在多轮对话中**同时实现伦理责任与情感共鸣之间的动态平衡**。

---

### 🚀 **提出了什么新方法或新思路**

作者提出 **ETHICMIND** —— 一种**风险感知的推理框架（risk-aware alignment framework）**，其核心思想是：

> 将“伦理-情感对齐”建模为一个**显式的回合级决策过程（turn-level decision problem）**，而非直接生成回复。

该框架在推理阶段（inference time）引入三阶段模块化流程：

1. **Joint Risk and Emotion Analyzer (A)**  
   - 分析当前对话历史，输出：
     - `ethical_category`：六类伦理风险标签（如 Serious Illegal Conduct）
     - `emotion`：自由文本描述用户情绪（如 "guilty and apprehensive"）
     - `Rules of Thumb (RoTs)`：上下文相关的轻量级道德准则（例如："It is wrong to flee after an accident."）

2. **Strategy Planner (P)**  
   - 基于分析结果，规划高层响应策略（high-level response strategy），例如：
     - “Firm Correction (due to harmful intent)”
     - “Perspective Diversification (in moral dilemma)”

3. **Response Generator (G)**  
   - 在对话历史、分析结果和策略指导下生成最终回复，确保内容既符合伦理又具共情力。

> 🔑 **无需额外训练**：整个框架通过 prompt engineering 实现，可插拔地应用于任何 LLM。

---

### ⭐ **相比现有方法的优势**

| 维度 | 传统方法 | ETHICMIND |
|------|--------|----------|
| **对齐方式** | 单一维度优化（仅情感 or 仅安全） | 联合建模伦理与情感 |
| **动态性** | 回复静态，难以随对话演进调整 | 每轮重新分析与策略更新 |
| **灵活性** | 刚性规则或泛化拒绝（refusal） | 动态策略选择，避免“一刀切” |
| **可解释性** | 黑箱生成 | 显式输出 RoTs 和策略，便于调试 |

> ✅ 特别适用于高风险、道德模糊场景（如心理支持、法律咨询等），能提供**有温度且负责任**的回应。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**

- 主要基于 **Prosocial Dialogues (ProS.) dataset**（Kim et al., 2022）
- 对原始数据进行**再标注（re-annotation）**，划分出六个伦理风险类别：
  1. Serious Illegal Conduct  
  2. Ethical Violations  
  3. Moral Dilemmas  
  4. Social Misconduct  
  5. Potentially Harmful Behaviors  
  6. Benign Conversations
- 最终构建包含 **298个多轮对话样本** 的评估集，每类约50例。

---

### 🔧 **实验设置**

#### ✅ **用户模拟器（User Simulator）**
- 使用 **context-aware paraphrasing** 技术生成自然变体的用户输入。
- 保持原始意图和伦理风险不变，仅改变表达形式，以测试系统的鲁棒性和一致性。

#### ✅ **评估维度（Evaluation Dimensions）**
四个关键指标，均由 GPT-4o 和 Claude-Sonnet-4 自动评分，并辅以人工偏好判断：

| 指标 | 描述 |
|------|------|
| **Respectful Tone** | 语言是否礼貌、得体、无偏见或攻击性 |
| **Ethical Guidance** | 是否识别并妥善引导伦理风险 |
| **Empathy** | 是否理解并回应用户的情绪状态 |
| **Specificity and Engagement** | 回应是否具体、相关、促进进一步交流 |

> 所有评分均采用结构化 prompt（见附录 B.3），保证一致性。

---

### 🆚 **基线方法对比**

分为两类：

#### （1）任务专用模型（Task-Specific Models）
- **CoSMO-3B**：专为 prosocial 对话微调
- **Emotional-LLaMA-8B**：专注于情感表达

#### （2）通用指令模型（General-Purpose Assistants）
- **Llama-2-7B-chat**
- **Llama-3-8B-Instruct**
- **Llama-3.3-70B-Instruct**
- **GPT-4o**

> 所有 baseline 均使用相同 system prompt：“You are a helpful, emotionally aware, and safety-aligned Chatbot…”

> ETHICMIND 则在其基础上增加 A→P→G 的三步推理链。

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据（来自 Table 2 & 3）**

| Model | Overall Score (GPT-4o) | Ethical Guid. | Empathy | Respect. Tone | Specif. Engage. |
|-------|------------------------|---------------|---------|----------------|------------------|
| CoSMO-3B | 4.54 | 4.37 | 4.01 | 4.55 | 5.24 |
| Llama-3-8B-Instruct | 7.37 | 6.56 | 6.89 | 8.23 | 7.79 |
| **ETHICMIND-Llama3-8B** | **7.53↑** | **6.67↑** | **7.31↑** | 8.24 | **7.92↑** |
| Llama-3.3-70B | 7.68 | 6.84 | 7.08 | 8.54 | 8.26 |
| **ETHICMIND-Llama3.3-70B** | **7.82↑** | **7.03↑** | **7.45↑** | 8.43 | **8.36↑** |
| GPT-4o | 7.60 | 6.83 | 6.99 | 8.46 | 8.11 |
| **ETHICMIND-GPT4o** | **7.90↑** | **7.31↑** | **7.35↑** | 8.58 | **8.34↑** |

> ✅ **所有 backbone 上 ETHICMIND 均带来一致提升**，尤其在 **Empathy** 和 **Ethical Guidance** 上增益明显。

---

### 📈 **跨风险类别的稳定性表现（RQ2）**

- 在 **高风险类别**（如 Serious Illegal Conduct, Ethical Violations）中，ETHICMIND 表现更优。
- 例如，在 `ETHICMIND-GPT-4o` 中：
  - 对比 baseline，**Serious Illegal Conduct** 得分从 7.71 → **7.85**
  - **Ethical Violations** 从 7.53 → **7.89**
- 表明该框架在**最需要谨慎处理的情境下仍能维持高质量对齐**。

---

### 🔍 **消融实验结果（Ablation Study, Table 6）**

移除任一组件均导致性能下降，验证各模块必要性：

| 移除组件 | 影响最大维度 | 典型下降（以 GPT-4o 为例） |
|--------|-------------|----------------------------|
| **w/o Emotion** | Empathy ↓ | 7.35 → 6.98 |
| **w/o RoT** | Ethical Guidance ↓ | 7.31 → 6.82 |
| **w/o Planner** | 多维度全面下降 | Overall: 7.89 → 7.77 |

> 🔥 **Strategy Planner 是最关键组件**，起到整合风险与情绪信号、协调响应立场的作用。

---

### 👥 **人类偏好评估（Human Evaluation, Table 5）**

三人标注员进行成对比较（preference-based study）：

| Backbone | ETHICMIND 获胜率 | 平局率 |
|--------|------------------|-------|
| Llama-3-8B-Instruct | 52.68% | 39.93% |
| Llama-3.3-70B-Instruct | 68.46% | 24.83% |
| GPT-4o | **70.47%** | 19.80% |

> 即使面对最强基线 GPT-4o，**超过七成的人类评估者更偏好 ETHICMIND 的对话质量**。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **伦理与情感必须联合建模**：分离处理会导致系统在高风险情境下要么过于僵硬，要么过度迎合。
2. **回合级决策机制有效**：通过显式分析 + 策略规划，ETHICMIND 能够动态调整语气与指导强度。
3. **无需训练即可增强对齐能力**：基于 prompt 的框架设计使其具有良好的**可迁移性与部署灵活性**。
4. **在道德模糊场景中优势显著**：特别是在用户表现出矛盾心理（如 guilt + defensiveness）时，ETHICMIND 更擅长渐进式引导而非说教。

---

### ⚠️ **局限性**

1. **依赖外部 LLM 进行推理**：Analyzer 和 Planner 本身由 LLM 实现，可能存在幻觉或误判。
2. **评估仍依赖自动打分**：尽管使用交叉验证和人工偏好，但 GPT-4o 自评仍存在潜在偏差。
3. **推理开销较高**：每轮需三次模型调用（vs. 基线一次），延迟增加（见 Table 7）。
   - 解决方案建议：仅在中高风险回合启用完整流程，低风险回落至标准生成。

4. **文化普适性有限**：伦理分类与 RoTs 基于西方文献设定，可能不适用于所有文化背景。

---

### 🔮 **未来工作方向**

- 将 Analyzer 与 Planner **蒸馏为轻量控制模块**，降低推理成本。
- 探索 **多智能体协作机制**，让不同角色分别负责风险检测、情感建模与策略制定。
- 引入 **用户反馈闭环**，实现在线学习与策略优化。
- 扩展至更多领域（如医疗、教育），构建领域特定的伦理知识库。

---

## ✅ 总结一句话

> **ETHICMIND 提供了一种无需训练、可插拔的风险感知框架，首次将伦理-情感对齐建模为显式的回合级决策过程，在多轮对话中实现了更具适应性、责任感与共情力的交互体验。**

</details>

---

### 15. [Synthesizing real-world distributions from high-dimensional Gaussian Noise with Fully Connected Neural Network](https://arxiv.org/abs/2604.09091)

**Authors**: Joanna Komorniczak  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.09091v1  

#### Abstract
The use of synthetic data in machine learning applications and research offers many benefits, including performance improvements through data augmentation, privacy preservation of original samples, and reliable method assessment with fully synthetic data. This work proposes a time-efficient syntheti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Synthesizing real-world distributions from high-dimensional Gaussian Noise with Fully Connected Neural Network*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**合成数据生成**中的三个核心挑战：
- **隐私保护**：传统方法（如 SMOTE）存在隐私泄露风险，原始样本可能被重构。
- **效率低下**：主流深度生成模型（如 CTGAN、TVAE）训练时间长、计算成本高。
- **分布建模能力不足**：部分方法难以准确捕捉真实世界数据的复杂分布特性。

目标是开发一种**高效、安全且能精确模拟真实数据分布**的合成数据生成方法。

### 提出的新方法：DiMSO（Distribution Mapping with Shuffled Optimization）
- **核心思想**：直接从高维标准正态噪声 $ \mathcal{N}(0,1) $ 出发，通过一个**全连接神经网络**（Fully Connected Neural Network）学习映射函数，将其转换为目标真实数据分布。
- **无需编码器-解码器结构**：不同于 VAE 或 GAN，DiMSO 不进行潜在空间编码，而是端到端地将随机噪声映射为合成数据。
- **随机化损失函数（Randomized Loss Function）**：
  - 默认使用 **Randomized Absolute Error (RAE)** 损失，即在每次优化时随机配对合成样本与目标样本并最小化其绝对误差。
  - 其他变体包括基于 Wasserstein Distance (W) 和结合协方差的 Wasserstein + Covariance (WC) 损失。

### 相比现有方法的优势
| 维度 | DiMSO | 传统方法（如 CTGAN/TVAE） |
|------|-------|--------------------------|
| **架构复杂度** | 极简：仅三层 FCN | 复杂：多层网络、对抗训练或变分推断 |
| **训练速度** | 快数个数量级 | 慢，需大量迭代 |
| **内存开销** | 低 | 高 |
| **隐私性** | 更好（因非精确复制） | 存在记忆效应风险 |
| **实现难度** | 简单 | 复杂，调参困难 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **25 个多样化的公开真实世界 tabular 数据集**，涵盖不同规模（163–3772 样本）、特征维度（4–44 特征）和类别不平衡程度（从极度不平衡到平衡）。
- 包括医疗、金融等领域常见任务，确保广泛适用性。
- 所有数据均经过标准化处理。

### 实验设置与评估指标

#### 实验一：分布相似性评估
- **目标**：衡量生成数据与真实数据的分布接近程度。
- **评估指标**：
  - **Wasserstein Distance (WD)**：衡量分布间“搬运”成本。
  - **Maximum Mean Discrepancy (MMD)**：核方法下的分布差异。
  - **Mean Euclidean Distance to Nearest Neighbor (MeanNN)**：反映局部邻近关系。
- **注意**：低值表示更高相似性；MeanNN 过低可能意味着隐私泄露。

#### 实验二：分类性能影响
- **目标**：评估使用合成数据训练分类器的效果。
- **流程**：
  - 在训练集上生成合成数据 → 分别用真实/合成数据训练分类器 → 在相同测试集上比较性能。
- **评估指标**：
  - **Balanced Accuracy (BAC)**：适用于不平衡数据的平均召回率。
- **分类器**：GNB、DT、RFC、SVC、MLP。

#### 实验三：时间复杂度对比
- **目标**：比较生成达到相同 MMD 水平时所需的时间。
- **基准**：以 TVAE 和 CTGAN 达到的目标 MMD 为参考，测量 DiMSO 达到同等水平所需时间。

### 基线方法对比
| 方法 | 类型 | 来源 |
|------|------|------|
| **SMOTE**, **SVMSMOTE** | 经典过采样 |
| **Gaussian Copula (GC)** | 统计模型 |
| **TVAE**, **CTGAN** | 当前 SOTA 深度生成模型 |
| **DiMSO (RAE/W/WC)** | 本文提出 |

所有方法均按类单独生成，并组合成最终合成数据集。

---

## 3. 主要实验结果和性能指标

### 分布相似性（Experiment 1）
- **MMD 表现最佳**：
  - DiMSO 的 **RAE** 和 **SMOTE** 在多数数据集上取得最低 MMD 值（即最高分布相似性）。
  - 在 PCA 降维后场景下，**RAE** 明显优于其他方法。
- **统计检验支持**：
  - Friedman + Nemenyi 检验显示，**SMOTE 和 RAE 在原始和 PCA 空间中均处于领先组**，且两者无显著差异。
- **MeanNN 结果合理**：
  - RAE、SMOTE、SVMSMOTE 得分较高，表明未完全复制原样本，有利于隐私保护。

> ✅ **关键数据**：在多个数据集上，DiMSO(RAE) 的 MMD 比 TVAE/CTGAN 低 30%-70%，且更稳定。

---

### 分类性能提升（Experiment 2）
- **总体趋势**：
  - 使用 DiMSO(RAE) 和 SMOTE 生成的数据训练分类器，**BAC 提升最显著**，尤其在高度不平衡数据上。
  - 对于 RFC、SVC、MLP 等强分类器，RAE 和 WC 明显优于 W、GC、CTGAN。
- **PCA 场景下表现更优**：
  - 在 PCA 降维后生成合成成分再逆变换，**RAE 成为所有分类器中的最优方法**。
  - Critical Difference 图显示 RAE 与其他领先方法（SMOTE、WC）统计相关，稳居榜首。
- **典型提升案例**：
  - 在 `allrep`、`yeast` 等严重不平衡数据上，使用 RAE 合成数据可使 SVC 的 BAC 提升超过 **40%**。

---

### 时间效率碾压级优势（Experiment 3）
- **目标设定**：让 DiMSO 达到 TVAE 或 CTGAN 生成数据所达到的 MMD 水平。
- **结果惊人**：
  - **DiMSO(RAE) 在约 100 轮内即可超越 TVAE 的 MMD 效果**，而 TVAE 需 300 轮。
  - **执行时间对比**：
    - TVAE 平均耗时：**7.38 秒**
    - CTGAN 平均耗时：**20.93 秒**
    - DiMSO(RAE) 平均耗时：**0.056 秒（vs TVAE） / 0.014 秒（vs CTGAN）**

> ⚡️ **结论**：DiMSO 实现相同甚至更好的分布拟合效果，**速度快达数十至数百倍**。

| 方法 | vs TVAE 时间(s) | vs CTGAN 时间(s) |
|------|------------------|-------------------|
| DiMSO (RAE) | 0.056 | 0.014 |
| DiMSO (W)   | 0.598 | 0.019 |
| DiMSO (WC)  | 0.207 | 0.023 |
| TVAE        | 7.382 | — |
| CTGAN       | —     | 20.933 |

---

## 4. 关键结论和发现

### 主要发现
1. **极简架构也能胜出**：  
   尽管 DiMSO 仅使用简单的 FCN 和 RAE 损失，但在分布拟合和下游任务性能上**全面超越或媲美当前 SOTA 深度生成模型**（如 CTGAN、TVAE）。

2. **RAE 损失最为有效**：  
   在三种损失函数中，**Randomized Absolute Error (RAE)** 表现最佳，兼具高分布保真度与良好的分类增益。

3. **PCA 可增强隐私与性能**：  
   在 PCA 降维空间中生成合成数据，不仅能提升隐私保护（模糊原始语义），还能进一步提高分类质量，同时降低生成时间和内存消耗。

4. **时间效率革命性突破**：  
   DiMSO 的生成速度比现代深度生成模型快**两个数量级以上**，使其非常适合资源受限或需要快速迭代的应用场景。

### 方法的局限性
- **假设连续特征为主**：当前实验基于标准化后的数值型数据，未明确处理混合类型（categorical + numerical）特征。
- **缺乏显式隐私机制**：虽然 RAE 天然避免样本复制，但未引入如 Differential Privacy 等形式化隐私保障。
- **对极端稀疏或高维数据适应性未知**：实验集中在中小规模 tabular 数据，未验证在超高维或稀疏场景下的表现。

### 未来工作方向
- 扩展至 **mixed-type data** 支持，加入预处理与后处理模块以保持类别语义。
- 探索 **early stopping 策略**，防止 RAE 因过度训练导致过拟合与隐私下降。
- 研究生成样本数量、类别基数等超参数对 MMD/BAC 的影响。
- 结合 **Differential Privacy** 或 **Federated Learning** 框架，构建更安全的分布式合成数据系统。

--- 

> 📌 **总结一句话**：  
> **DiMSO 是一种简单、快速、高效的合成数据生成新范式，在分布拟合精度、分类效用和运行效率上全面超越主流深度生成模型，为隐私保护与数据增强提供了极具实用价值的新工具。**

</details>

---

### 16. [Event-Driven Temporal Graph Networks for Asynchronous Multi-Agent Cyber Defense in NetForge_RL](https://arxiv.org/abs/2604.09523)

**Authors**: Igor Jankowski  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.09523v1  

#### Abstract
The transition of Multi-Agent Reinforcement Learning (MARL) policies from simulated cyber wargames to operational Security Operations Centers (SOCs) is fundamentally bottlenecked by the Sim2Real gap. Legacy simulators abstract away network protocol physics, rely on synchronous ticks, and provide cle...

---

### 17. [TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](https://arxiv.org/abs/2604.08948)

**Authors**: Gang Hu, Yating Chen, Haiyan Ding, Wang Gao, Jiajia Huang, Min Peng, Qianqian Xie, Kun Yu  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.08948v1  

#### Abstract
While Large Language Models (LLMs) excel in various general domains, they exhibit notable gaps in the highly specialized, knowledge-intensive, and legally regulated Chinese tax domain. Consequently, while tax-related benchmarks are gaining attention, many focus on isolated NLP tasks, neglecting real...

---

### 18. [Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](https://arxiv.org/abs/2604.09377)

**Authors**: Hui Liu, Bin Zou, Kecheng Chen, Jie Liu, Wenya Wang, Haoliang Li  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.09377v1  

#### Abstract
Large language models (LLMs) exhibit substantial variability in performance and computational cost across tasks and queries, motivating routing systems that select models to meet user-specific cost-performance trade-offs. However, existing routers generalize poorly in cold-start scenarios where in-d...

---

### 19. [Wireless Communication Enhanced Value Decomposition for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2604.08728)

**Authors**: Diyi Hu, Bhaskar Krishnamachari  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.08728v1  

#### Abstract
Cooperation in multi-agent reinforcement learning (MARL) benefits from inter-agent communication, yet most approaches assume idealized channels and existing value decomposition methods ignore who successfully shared information with whom. We propose CLOVER, a cooperative MARL framework whose central...

---

### 20. [Finite-Sample Analysis of Nonlinear Independent Component Analysis:Sample Complexity and Identifiability Bounds](https://arxiv.org/abs/2604.08850)

**Authors**: Yuwen Jiang  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.08850v1  

#### Abstract
Independent Component Analysis (ICA) is a fundamental unsupervised learning technique foruncovering latent structure in data by separating mixed signals into their independent sources. While substantial progress has been made in establishing asymptotic identifiability guarantees for nonlinear ICA, t...

---

### 21. [Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](https://arxiv.org/abs/2604.08944)

**Authors**: Benjamin Amoh, Geoffrey Parker, Wesley Marrero  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.08944v1  

#### Abstract
Multi-agent coordination under partial observability requires agents to share complementary private information. While recent methods optimize messages for intermediate objectives (e.g., reconstruction accuracy or mutual information), rather than decision quality, we introduce \textbf{SeqComm-DFL}, ...

---

### 22. [DiffHLS: Differential Learning for High-Level Synthesis QoR Prediction with GNNs and LLM Code Embeddings](https://arxiv.org/abs/2604.09240)

**Authors**: Zedong Peng, Zeju Li, Qiang Xu, Jieru Zhao  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.09240v1  

#### Abstract
High-Level Synthesis (HLS) compiles C/C++ into RTL, but exploring pragma-driven optimization choices remains expensive because each design point requires time-consuming synthesis. We propose \textbf{\DiffHLS}, a differential learning framework for HLS Quality-of-Result (QoR) prediction that learns f...

---

### 23. [Distributed Online Convex Optimization with Compressed Communication: Optimal Regret and Applications](https://arxiv.org/abs/2604.09276)

**Authors**: Sifan Yang, Dan-Yue Li, Lijun Zhang  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.09276v1  

#### Abstract
Distributed online convex optimization (D-OCO) is a powerful paradigm for modeling distributed scenarios with streaming data. However, the communication cost between local learners and the central server is substantial in large-scale applications. To alleviate this bottleneck, we initiate the study ...

---

### 24. [Meta-Learned Basis Adaptation for Parametric Linear PDEs](https://arxiv.org/abs/2604.09289)

**Authors**: Vikas Dwivedi, Monica Sigovan, Bruno Sixou  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.09289v1  

#### Abstract
We propose a hybrid physics-informed framework for solving families of parametric linear partial differential equations (PDEs) by combining a meta-learned predictor with a least-squares corrector. The predictor, termed \textbf{KAPI} (Kernel-Adaptive Physics-Informed meta-learner), is a shallow task-...

---

### 25. [Stochastic-Dimension Frozen Sampled Neural Network for High-Dimensional Gross-Pitaevskii Equations on Unbounded Domains](https://arxiv.org/abs/2604.09361)

**Authors**: Zhangyong Liang  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.09361v1  

#### Abstract
In this paper, we propose a stochastic-dimension frozen sampled neural network (SD-FSNN) for solving a class of high-dimensional Gross-Pitaevskii equations (GPEs) on unbounded domains. SD-FSNN is unbiased across all dimensions, and its computational cost is independent of the dimension, avoiding the...

---

### 26. [E3-TIR: Enhanced Experience Exploitation for Tool-Integrated Reasoning](https://arxiv.org/abs/2604.09455)

**Authors**: Weiyang Guo, Zesheng Shi, Liye Zhao, Jiayuan Ma, Zeen Zhu, Junxian He, Min Zhang, Jing Li  
**Category**: cs.AI  
**Published**: 2026-04-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.09455v1  

#### Abstract
While Large Language Models (LLMs) have demonstrated significant potential in Tool-Integrated Reasoning (TIR), existing training paradigms face significant limitations: Zero-RL suffers from inefficient exploration and mode degradation due to a lack of prior guidance, while SFT-then-RL is limited by ...

---

### 27. [Quantisation Reshapes the Metacognitive Geometry of Language Models](https://arxiv.org/abs/2604.08976)

**Authors**: Jon-Paul Cacioli  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.08976v1  

#### Abstract
We report that model quantisation restructures domain-level metacognitive efficiency in LLMs rather than degrading it uniformly. Evaluating Llama-3-8B-Instruct on the same 3,000 questions at Q5_K_M and f16 precision, we find that M-ratio profiles across four knowledge domains are uncorrelated betwee...

---

### 28. [Facet-Level Tracing of Evidence Uncertainty and Hallucination in RAG](https://arxiv.org/abs/2604.09174)

**Authors**: Passant Elchafei, Monorama Swain, Shahed Masoudian, Markus Schedl  
**Category**: cs.CL  
**Published**: 2026-04-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.09174v1  

#### Abstract
Retrieval-Augmented Generation (RAG) aims to reduce hallucination by grounding answers in retrieved evidence, yet hallucinated answers remain common even when relevant documents are available. Existing evaluations focus on answer-level or passage-level accuracy, offering limited insight into how evi...

---

### 29. [Fully Autonomous Z-Score-Based TinyML Anomaly Detection on Resource-Constrained MCUs Using Power Side-Channel Data](https://arxiv.org/abs/2604.08581)

**Authors**: Abdulrahman Albaiz, Fathi Amsaad  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.08581v1  

#### Abstract
This paper presents a fully autonomous Tiny Machine Learning (TinyML) Z-Score-based anomaly detection system deployed on a low-power microcontroller for real-time monitoring of appliance behavior using power side-channel data. Unlike existing Internet of Things (IoT) anomaly detection approaches tha...

---

### 30. [Adaptive Simulation Experiment for LLM Policy Optimization](https://arxiv.org/abs/2604.08779)

**Authors**: Mingjie Hu, Siyang Gao, Jian-qiang Hu, Enlu Zhou  
**Category**: cs.LG  
**Published**: 2026-04-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.08779v1  

#### Abstract
Large language models (LLMs) have significant potential to improve operational efficiency in operations management. Deploying these models requires specifying a policy that governs response quality, shapes user experience, and influences operational value. In this research, we treat LLMs as stochast...

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
