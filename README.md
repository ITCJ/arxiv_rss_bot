# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-04 06:14:13 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs](https://arxiv.org/abs/2603.02731)

**Authors**: Wuyue Zhang, Chongdong Huang, Chunbo You, Cheng Gu, Fengjuan Wang, Mou Sun  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2603.02731v1  

#### Abstract
Training large-scale Mixture-of-Experts (MoE) models is bottlenecked by activation memory and expert-parallel communication, yet FP4 training remains impractical on Hopper-class GPUs without native MXFP4 or NVFP4 support. In this work, we present a training recipe that enables MXFP4 efficiency for M...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大规模 **Mixture-of-Experts (MoE)** 模型的训练受限于两大瓶颈：
- **激活内存（activation memory）占用高**
- **专家并行通信（expert-parallel communication）开销大**

尽管 **FP4** 量化格式在理论上能显著降低内存和带宽需求，但主流 **Hopper 架构 GPU**（如 H100）缺乏对 FP4 的原生支持（无 FP4 Tensor Core），导致 FP4 训练难以实用化。此外，将 FP8/BF16 数据转换为 FP4 通常需要经过 BF16 中间态（如 FP8 → BF16 → FP4），带来额外的精度损失和性能开销。

---

### 🚀 提出的新方法与创新思路

作者提出了一种**软件模拟的 MXFP4 训练框架**，在无原生 FP4 支持的 Hopper GPU 上实现接近硬件级效率的 FP4 内存与通信压缩。其核心设计是**解耦计算精度与存储/通信精度**：

#### 主要创新点包括：

| 创新点 | 描述 |
|--------|------|
| **FP4 激活缓存与通信压缩** | 将前向传播中的 activation 和 All-to-All 通信压缩为 **MXFP4** 格式，减少 50% 以上的内存和通信体积。 |
| **直接 FP8-to-FP4 转换算法** | 避免传统路径 `FP8 → BF16 → FP4`，提出**位级（bitwise）直接转换**，消除 BF16 中间态带来的延迟与精度损失。 |
| **缩放感知的行列转换（Scaling-aware Row→Col）** | 设计专用 CUDA kernel `FP4RowToFP8Col`，融合反量化与矩阵转置，在 Wgrad 计算中大幅减少内存访问。 |
| **异步精度策略（Asymmetric Precision Flow）** | 前向用 FP4 压缩，后向恢复为标准 FP8 流程，避免梯度反量化开销过大影响性能。 |
| **端到端集成至 Transformer Engine / DeepEP** | 完整支持现有训练栈，兼容 Megatron-LM、Transformer Engine 和 DeepEP 通信库。 |

---

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **无需新硬件** | 在无 FP4 Tensor Core 的 Hopper GPU 上实现 FP4 效益，立即适用于现网集群。 |
| **零收敛损失** | 实验显示 MXFP4 与 FP8/BF16 收敛轨迹一致，相对 BF16 损失偏差仅 +0.61%。 |
| **系统效率提升** | 显著降低峰值内存、提高吞吐量，允许更宽松的重计算策略。 |
| **避免 double quantization error** | 不依赖多次精度转换，保持数值稳定性。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
论文未明确列出具体预训练语料，但指出实验配置复现了当前主流大模型训练流程，上下文长度为 **4096 tokens**，全局 batch size 为 **4800 sequences**，符合典型 LLM 预训练设定。

> 注：重点在于**系统性能评估**而非下游任务表现，因此关注的是训练过程本身的效率指标。

---

### ⚙️ 实验设置

| 项目 | 设置 |
|------|------|
| **硬件平台** | 32 节点集群，共 256 张 **NVIDIA Hopper GPU（80GB HBM3）**，节点内通过 NVSwitch 连接，跨节点使用 InfiniBand。 |
| **模型规模** | - **671B 参数 MoE 模型**（主实验）<br>- **236B 参数 MoE 模型**（用于控制变量对比）<br>架构基于 DeepSeek-V3，采用 MLA（Multi-Head Latent Attention）。 |
| **评估指标** | - **Tokens per GPU per Second (TGS)**：衡量训练吞吐<br>- **Peak Memory Usage (%)**：显存占用率<br>- **Loss Trajectory**：验证收敛性 |
| **重计算策略（Recomputation Scope）** | 多种设置对比：<br>- Checkpoint Attention + LayerNorm + MoE Experts<br>- Checkpoint MLA Up-Projection Only |

---

### 🆚 基线方法对比

| 基线 | 描述 |
|------|------|
| **BF16** | 标准 bfloat16 混合精度训练，作为高精度基准。 |
| **FP8** | 使用 NVIDIA Transformer Engine 的 blockwise FP8 训练方案，针对 Hopper 优化，代表当前先进实践。 |

> 所有基线均基于 Megatron-LM 实现，并启用相同并行策略（tensor parallelism, expert parallelism）。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（671B MoE 模型）

| 重计算范围 | 方法 | TGS | 显存占用 |
|-----------|------|-----|---------|
| Attn + LN + MoE Experts | BF16 | 1122 | 68.29% |
|                              | FP8   | 1157 | 74.20% |
|                              | **Ours (MXFP4)** | **1156** | **59.40%** ✅ |
| MLA Up Proj Only           | BF16 | OOM | — |
|                              | FP8   | OOM | — |
|                              | **Ours (MXFP4)** | **1302** ✅ | **70.11%** |

#### 性能增益总结：
- **峰值激活内存下降 14.8%（11.8 GB）**
- **训练吞吐提升 12.5%**（从 1157 → 1302 TGS）
- 在最激进的重计算策略下仍可运行，而 BF16/FP8 均 OOM

---

### 🔬 控制变量实验（236B 模型）

| 方法 | 相比 FP8 内存降幅 | 相比 BF16 内存降幅 |
|------|------------------|-------------------|
| MXFP4（仅 MLP + Shared Experts） | 6.9% | 11% |
| MXFP4（扩展至全部 MoE 层）     | 7.2% | 11% |

> 表明 FP4 压缩效果具有良好的可扩展性和一致性。

---

### 🔍 消融实验结果

#### （1）Kernel 融合效率（图3）
- 自定义融合 kernel（dequant + cast + transpose）相比非融合实现提速 **1.6x ~ 1.9x**
- 成功消除中间内存写入开销

#### （2）总量化开销分析（图4）
- **标准线性层（Attention 投影等）**：因引入实时量化，延迟略高于 TE FP8 基线（约 0.34x ~ 0.36x 相对速度）
- **MoE 专家层（Group Linear）**：定制 `msplit` kernel 实现 **1.43x ~ 1.53x 加速**
- **总体收益来自 MoE 层主导的 FLOPs 占比高**，足以抵消其他部分的开销

#### （3）收敛性验证（图5）
- 在消耗 160B tokens 后，MXFP4 与 BF16、FP8 的 loss 曲线高度一致
- 相对于 BF16：
  - FP8 损失偏差：+0.29%
  - MXFP4 损失偏差：+0.61%
- 无发散或不稳定现象，证明数值鲁棒性强

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **FP4 级别的内存与通信效率可在无原生支持的 Hopper GPU 上实现**  
   通过软硬协同设计（software-hardware co-design），无需等待 Blackwell 等新一代硬件即可享受 FP4 带来的红利。

2. **解耦“计算精度”与“存储/通信精度”是可行且高效的策略**  
   计算保留在 FP8 以利用 Tensor Core，存储与通信使用 MXFP4 压缩，兼顾性能与稳定。

3. **异步精度流（asymmetric forward/backward）是最优折衷**  
   前向压缩节省内存，后向保留 FP8 避免反量化开销，实现端到端最优吞吐。

4. **专用 kernel 融合对性能至关重要**  
   特别是在 MoE 场景中，grouped memory access 和 layout transformation 必须联合优化。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **仅前向压缩，后向未进一步优化** | 当前未尝试在反向传播中也引入 FP4，可能仍有潜力未挖掘。 |
| **依赖特定 block size 对齐（32/128）** | MXFP4 与 FP8 的 block granularity 不同，需 hierarchical scaling 对齐，增加实现复杂度。 |
| **尚未支持完全端到端 FP4 训练** | 权重更新、optimizer state 仍使用 FP8/BF16，未探索更低比特表示。 |

---

### 🔮 未来工作方向

1. **探索后向传播中的 FP4 应用**  
   结合 stochastic rounding 或 error feedback 技术，在梯度传输中也启用 FP4。

2. **扩展至 NVFP4 或其他 4-bit 格式**  
   验证不同 FP4 编码（如 E4M3 变体）在 MoE 中的表现差异。

3. **结合 sparsity 与量化进行联合压缩**  
   在 token routing 稀疏性的基础上叠加 activation quantization，进一步释放资源。

4. **部署至更大规模模型（Trillion+ 参数）**  
   验证该方法在更高参数量下的可扩展性与稳定性。

5. **开源推动生态适配**  
   已公开代码（GitHub），有望被纳入主流框架（如 Megatron-Core、PyTorch Distributed）成为标准组件。

---

> 💡 **一句话总结**：  
> 本文首次实现了在 **无原生 FP4 支持的 Hopper GPU 上高效训练千亿级 MoE 模型**，通过 **direct FP8-to-FP4 转换 + hybrid precision flow + fused kernel design**，达成 **内存降 14.8%、吞吐升 12.5%、收敛无损** 的卓越效果，为低比特训练提供了实用化的工程范本。

</details>

---

### 2. [MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks](https://arxiv.org/abs/2603.02630)

**Authors**: Zhi Hong, Qian Zhang, Jiahang Sun, Zhiwei Shang, Mingze Kong, Xiangyi Wang, Yao Shu, Zhongxiang Dai  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2603.02630v1  

#### Abstract
Large Language Models (LLMs) have achieved great success in many real-world applications, especially the one serving as the cognitive backbone of Multi-Agent Systems (MAS) to orchestrate complex workflows in practice. Since many deployment scenarios preclude MAS workflow modifications and its perfor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MASPOB: Bandit-Based Prompt Optimization for Multi-Agent Systems with Graph Neural Networks

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Multi-Agent Systems (MAS)** 中的 **Prompt Optimization** 问题，提出了一个在现实部署中极具挑战性的场景下的解决方案。许多实际应用（如医疗SOP、金融审计）中的 MAS 工作流（workflow topology）是经过专家验证且不可修改的，因此无法通过调整系统结构来提升性能。在这种“冻结拓扑”（frozen-topology）的设定下，**优化各智能体（agent）的提示词（prompt）成为提升系统性能的主要手段**。

然而，MAS 的 Prompt 优化面临三大挑战：
1.  **样本效率低下（Sample inefficiency）**：每次评估一个完整的 Prompt 组合都需要执行整个多智能体流程，成本极高，评估预算（evaluation budget）极其有限。
2.  **拓扑诱导耦合（Topology-induced coupling）**：上游智能体的 Prompt 改变会改变其输出，从而影响下游智能体的输入分布，导致目标函数不可分（non-separable），独立优化每个智能体的 Prompt 会导致不稳定。
3.  **组合爆炸（Combinatorial explosion）**：联合 Prompt 空间是所有智能体 Prompt 候选集的笛卡尔积，搜索空间随智能体数量呈指数级增长。

### 提出的新方法和新思路
为解决上述挑战，作者提出了 **MASPOB (Multi-Agent System Prompt Optimization via Bandits)**，这是一个新颖的、高效的框架，其核心创新点在于将三种技术有机结合：

1.  **基于 Bandit 的不确定性引导探索（Uncertainty-driven exploration）**：
    *   将 Prompt 优化问题建模为 **contextual bandit** 问题。
    *   采用 **Upper Confidence Bound (UCB)** 准则来平衡 **利用（exploitation）** 和 **探索（exploration）**。
    *   利用 **information matrix** 在学习到的表示空间中量化认知不确定性（epistemic uncertainty），优先选择那些不仅预测性能好而且信息量大（即能减少不确定性的）的 Prompt 组合，从而在有限的评估次数内实现高效优化。

2.  **拓扑感知的代理模型（Topology-aware surrogate）**：
    *   引入 **Graph Neural Network (GNN)** 作为性能预测的代理模型（surrogate model）。
    *   将 MAS 的工作流建模为有向无环图（DAG），其中节点代表智能体，边代表信息流。
    *   GNN 通过消息传递机制显式地编码工作流拓扑结构，能够捕捉 Prompt 变化如何沿着图结构传播，为优化过程提供了强大的结构归纳偏置（structural inductive bias）。

3.  **可扩展的组合搜索（Scalable combinatorial search）**：
    *   采用 **Coordinate Ascent** 策略来分解全局搜索问题。
    *   在每一轮迭代中，依次固定其他智能体的 Prompt，只优化单个智能体的 Prompt。
    *   这种策略将每轮的搜索复杂度从 $O(\prod|P_i|)$ 降低到 $O(\sum|P_i|)$，实现了线性复杂度，使其能够高效处理大规模的组合空间。

### 相比现有方法的优势
*   **相比单智能体优化器（如 OPRO, PromptBreeder）**：MASPOB 显式考虑了智能体间的拓扑依赖关系，避免了因忽略耦合而导致的次优解。
*   **相比多阶段优化器（如 MIPRO）**：MIPRO 等方法通常隐式地捕获依赖（如通过 TPE），对拓扑结构不敏感。MASPOB 通过 GNN 显式地建模拓扑，能更有效地发现符合 MAS 结构的协调 Prompt 组合。
*   **综合优势**：MASPOB 在严格的评估预算下，实现了更高的样本效率和更好的最终性能。

## 2. 核心实验方法和设置

### 使用的数据集
实验在六个广泛使用的公共基准数据集上进行，覆盖了多种任务：
*   **问答（Question Answering）**: `HotpotQA`, `DROP`
*   **代码生成（Code Generation）**: `HumanEval`, `MBPP`
*   **数学推理（Mathematical Reasoning）**: `GSM8K`, `MATH`

### 实验设置和评估指标
*   **骨干 LLM**：默认使用 `GPT-4o-mini` 作为执行智能体的底层大语言模型。
*   **评估预算**：每个优化方法被给予 **50 次**验证集评估机会来选择最优的 Prompt 组合。
*   **评估指标**：根据任务类型使用标准指标：
    *   数学推理 (`GSM8K`, `MATH`)：**solve rate (%)**
    *   代码生成 (`HumanEval`, `MBPP`)：**Pass@1**
    *   问答 (`HotpotQA`, `DROP`)：**F1 score**

### 基线方法对比
与以下三类基线方法进行了比较：
1.  **无优化的单智能体方法**：`IO` (直接调用), `CoT` (Chain-of-Thought), `ReAct`.
2.  **单智能体 Prompt 优化**：`PromptBreeder`, `Instinct`.
3.  **多智能体系统**：`AFlow` (用于生成固定的 MAS 结构), `MIPRO` (作为多智能体 Prompt 优化的强基线).

## 3. 主要实验结果和性能指标

### 关键性能数据
*   **总体性能**：如 **Table 1** 所示，MASPOB 在所有六个基准测试上均取得了最佳结果，平均得分为 **80.58%**。
*   **与基线对比**：
    *   相比 `IO` 基线，平均提升 **12.02%**。
    *   相比 `AFlow`，平均提升 **2.06%**。
    *   相比 `MIPRO`，平均提升 **1.71%**。
*   **收敛性**：如 **Figure 3** 所示，MASPOB 能够稳定地随着评估次数的增加而提升性能，并在约第 35 轮时趋于稳定，表明其能在较少的评估次数内找到高质量的解。

### 与基线方法的对比结果
*   **在复杂结构上的表现**：在使用 `AFlow` 生成的更复杂的 MAS 结构（更多智能体）上进行测试时（**Table 2**），MASPOB 依然保持领先，而 `MIPRO` 的表现甚至不如 `AFlow`。这表明 MASPOB 的 GNN 模型能更好地泛化到不同的工作流复杂度，而 `MIPRO` 对于显式的拓扑依赖处理不足。
*   **跨模型泛化**：当将骨干 LLM 从 `GPT-4o-mini` 替换为 `Qwen-3-32B` 时（**Table 4**），MASPOB 依然是所有基准上的最佳方法，证明其性能提升并非依赖于特定模型的特性，而是源于更优的 Prompt 协调能力。

### 消融实验结果
*   **GNN 的有效性**：将 GNN 替换为普通的 MLP 后（**Table 3**），平均性能下降了 **2.31%**。这证明了显式地对工作流拓扑进行建模对于捕捉智能体间的耦合至关重要。
*   **Coordinate Ascent 的有效性**：与穷举的全局搜索（global search）相比（**Table 5**, **Figure 4**），Coordinate Ascent 的性能仅有轻微下降（例如在 HotpotQA 上仅低 0.29%），但运行时间减少了 **99.8%**，实现了计算成本和优化质量之间的极佳权衡。

## 4. 关键结论和发现

### 主要发现
1.  **Prompt 优化是提升冻结拓扑 MAS 的有效途径**：即使不改变经过验证的工作流结构，仅仅通过优化 Prompt 也能带来显著的性能提升。
2.  **显式建模拓扑结构至关重要**：忽略智能体间的拓扑依赖会导致次优的优化结果。MASPOB 通过 GNN 显式地编码拓扑信息，是其成功的关键。
3.  **不确定性引导的探索提高了样本效率**：在评估成本高昂的场景下，基于 UCB 的探索策略能够更智能地分配有限的评估资源，快速收敛到高性能区域。
4.  **Coordinate Ascent 是处理组合爆炸的有效策略**：它极大地降低了搜索复杂度，使得在大规模 Prompt 空间中进行优化变得可行。

### 方法的局限性
*   **依赖预定义的 Prompt 域**：MASPOB 优化的是给定的 Prompt 候选集，本身并不生成全新的 Prompt。Prompt 的质量和多样性会影响最终上限。
*   **GNN 和 Bandit 模型的假设**：其性能依赖于 GNN 能否准确捕捉拓扑效应以及 UCB 能否正确估计不确定性。

### 未来工作方向
*   **结合 Prompt 生成与优化**：将 Prompt 的生成（generation）和优化（optimization）过程结合起来，形成一个端到端的闭环。
*   **探索更复杂的不确定性估计**：研究在极低样本条件下更鲁棒的神经网络不确定性估计方法。
*   **应用于更广泛的场景**：将 MASPOB 的思想推广到其他需要在固定结构下进行参数优化的复杂系统。

</details>

---

### 3. [CUCo: An Agentic Framework for Compute and Communication Co-design](https://arxiv.org/abs/2603.02376)

**Authors**: Bodun Hu, Yoga Sri Varshan V, Saurabh Agarwal, Aditya Akella  
**Category**: cs.DC  
**Published**: 2026-03-04  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.02376v1  

#### Abstract
Custom CUDA kernel development is essential for maximizing GPU utilization in large-scale distributed LLM training and inference, yet manually writing kernels that jointly leverage both computation and communication remains a labor-intensive and error-prone process. Prior work on kernel optimization...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CUCo: An Agentic Framework for Compute and Communication Co-design

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在大规模分布式 LLM（Large Language Model）训练与推理中，GPU 利用率最大化依赖于高效地协同优化 **computation**（计算）和 **communication**（通信）。传统上，这两者由 CPU 主导分阶段执行（host-driven），存在以下核心问题：

- **手动编写融合内核（fused kernel）极其困难且易错**：需同时考虑 tile 大小、同步时机、通信粒度等复杂设计决策。
- **现有工具链割裂**：ML 编译器（如 TVM、XLA）假设通信是 host-side 的副作用，无法处理设备端发起的通信（device-initiated communication）。
- **硬件异构性强**：不同网络拓扑（InfiniBand vs. RoCE）、带宽延迟差异大，导致最优配置高度依赖具体环境，难以手动调优。

因此，如何自动化生成高性能、正确性有保障的 **compute-communication fused CUDA kernel** 成为一个关键挑战。

---

### **提出了什么新方法或新思路**

本文提出 **CUCo** —— 一种 **agentic（基于智能体）框架**，用于自动设计和生成融合了计算与通信的 CUDA 内核。其核心创新在于 **两阶段智能体架构 + 结构化搜索空间**：

#### ✅ 三大核心组件

| 组件 | 功能 |
|------|------|
| **Design Space Specification** | 定义了一个受控的、语义正确的优化空间 `C = B × P × S × I × G`，将复杂的 co-design 转化为可搜索的问题：<br>- `B`: Backend（GIN / LSA）<br>- `P`: Placement（重叠策略）<br>- `S`: Sync Scope（同步范围）<br>- `I`: Issuer Granularity（线程粒度）<br>- `G`: Granularity（传输块大小） |
| **Fast-Path Agent** | “正确优先” 的快速路径代理，通过三步流程（分析 → 转换 → 注解）生成一个保守但功能正确的 device-initiated baseline kernel，作为后续优化起点。 |
| **Slow-Path Agent** | “性能优先” 的慢速进化代理，采用 **LLM 驱动的进化搜索（evolutionary search）**，在结构化空间中探索更激进的优化策略（如 split put/wait、multi-stream overlap），利用实测反馈迭代改进。 |

#### ✅ 新思路亮点

- **先保正确性，再求高性能**：避免直接让 LLM 从零生成复杂分布式代码带来的高失败率。
- **指令先行（directive-first）**：要求 agent 必须先输出 `optimization_directive` 才能生成代码，使设计决策可解释、可追溯。
- **动态上下文注入**：运行时注入 NCCL Device API 文档、硬件拓扑、构建环境等 context，弥补 LLM 对新 API 缺乏训练数据的问题。
- **闭环反馈机制**：编译 → 验证 → 测评 → LLM 反馈诊断 → 指导下一轮变异，形成自适应优化循环。

---

### **相比现有方法的优势**

| 方面 | CUCo | 现有方法（如 CUDAForge, STARK, KernelFalcon） |
|------|------|---------------------------------------------|
| **多 GPU 支持** | ✅ 原生支持 | 多数仅单卡 |
| **通信支持** | ✅ 支持 GIN/LSA 设备端通信 | ❌ 通常忽略或仅 host-side NCCL |
| **是否需要训练** | ✅ 无需 fine-tune，zero-shot | ⚠️ 部分需 domain-specific 微调（如 Kevin-32B） |
| **融合能力** | ✅ 显式支持 compute-comm fusion | ⚠️ 多数未涉及或弱支持 |
| **鲁棒性** | ✅ 两阶段设计降低错误传播风险 | ❌ 直接生成易陷入死锁或不一致状态 |

> 如 Table 1 所示，CUCo 是目前唯一同时满足 **Multi-GPU、Comm Support、No Training、Fusion** 四项特性的系统。

---

## 2. 核心实验方法和设置

### **使用的代表性 workload（非标准数据集）**

由于缺乏标准 benchmark 来衡量 compute-communication fused kernel 性能，作者设计了四个典型多 GPU 场景：

| Workload | 描述 |
|--------|------|
| **Flash Attention + Context Parallelism** | 序列并行下的注意力计算，KV 分片环形传递（Ring Attention），强调细粒度流水线重叠。 |
| **DeepSeek-V3 MoE Dispatch & Combine** | MoE 模型中的稀疏路由 AlltoAll 通信，结合专家计算，测试负载不平衡下的性能表现。 |
| **KV-Cache Transfer** | Prefill 阶段后 K/V 投影的跨 GPU 传输，关注 compute-to-send gap 消除。 |
| **GEMM + AllGather** | 局部矩阵乘后聚合广播，隔离 post-compute collective 的优化潜力。 |

---

### **实验设置**

- **硬件平台**：
  - 2 台服务器，每台含 4× NVIDIA A100 (80GB)，通过 NVLink 连接；
  - 服务器间通过 **RoCE** 互联；
  - CPU: Intel Xeon Silver 4314, OS: Ubuntu 22.04, CUDA 13.1, NCCL 2.28.9。
- **LLM 模型**：Claude Sonnet 4.5（用于 fast-path、slow-path、feedback agent）。
- **评估指标**：
  - **End-to-end latency**（主指标）：CUDA event 测量中位值。
  - **Speedup**：相对于 host-driven baseline 的加速比。
  - **Fitness Score**：`fitness(c) = 10000 / (1 + t_ms)`，用于进化搜索的目标函数。

---

### **基线方法对比**

- **Host-driven Baseline**：
  - 使用标准 `ncclSend/Recv`, `ncclAlltoAll` 等 host API；
  - 通过双流（compute stream + comm stream）尝试 overlap；
  - 存在 kernel launch gap、CPU 同步开销等问题。

- **CUCo Evolved Kernel**：
  - 自动生成 device-initiated fused kernel（使用 GIN put/wait 或 LSA direct access）；
  - 实现 per-tile pipelining、split put/wait、信号驱动同步等高级优化。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| Workload | 加速比（Speedup） | 最大端到端延迟降低 |
|---------|------------------|--------------------|
| Flash Attention (SEQ=4096, HD=32) | **1.113× (11.3%)** | 从 1230.7ms → 1091.7ms |
| DeepSeek MoE (2:1 skew) | **1.262× (26.2%)** | 显著优于 padded AlltoAll |
| KV-Cache Transfer | **1.156× (15.6%)** | 消除 compute-send gap |
| GEMM + AllGather | **1.053× ~ 1.262×** | 取决于矩阵规模和连接方式 |

> 整体实现 **最高达 1.57× 的端到端加速**（见 Abstract 和 Section 4）。

---

### **与基线方法的对比结果**

- **主要收益来源**（以 Flash Attention 为例，Table 4）：
  | 优化项 | 延迟节省 | 机制说明 |
  |-------|--------|----------|
  | 消除 pipeline bubble | 63.7ms | host baseline 因 SM 饱和无法调度小信号 kernel，被迫串行等待所有 tile 完成；CUCo 支持 per-tile pipelining。 |
  | 移除 host API 开销 | 37.6ms | 替代 768 次 `ncclSend/Recv` 调用为 device-side `gin.put/flush`。 |
  | 消除 compute stream idle | 37.7ms | host 在通信轮次期间闲置 compute stream。 |
  | **总计** | **139.0ms** | 占总耗时 11.3% |

- **MoE 场景优势明显**：
  - CUCo 使用 **split put/wait + variable-size transfer**，避免了 host baseline 中为对齐而引入的 padding 开销。
  - 在高 skew ratio（如 5:1）下仍保持良好性能，体现其动态适应能力。

---

### **消融实验结果**

#### 🔹 Fast-Path Agent 的必要性（Figure 7 vs 8）

| 配置 | 正确 kernel 出现代数 | 最终得分 | 是否稳定收敛 |
|------|---------------------|----------|--------------|
| Fast + Slow Path | 第 2 代即生成 | 83.95 | ✅ 快速稳定 |
| Only Slow Path（无 fast-path） | 直到第 5 代才出现 | 81.81（↓2.5%） | ❌ 前期浪费大量预算 |

> **结论**：Fast-path 极大提升了样本效率，防止 slow-path 在基础编程模型上“重复试错”。

#### 🔹 Two-Phase Evolution（Explore → Exploit）的重要性（Figure 9 vs 10）

| 配置 | 探索阶段 | 最终得分 | 达到近优速度 |
|------|--------|----------|------------|
| Two-Phase（前 40% 探索） | ✅ 发现 barrier-free overlap、split put/wait 等新结构 | **83.95** | 第 3 代即接近最优 |
| Single-Phase（仅 exploit） | ❌ 陷入局部优化 | 80.36（↓4.3%） | 第 10 代才突破 76 |

> **结论**：前期多样性探索至关重要，否则会过早收敛到低质量局部最优。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Device-initiated communication 是未来方向**：NCCL Device API 和 NVSHMEM 提供了前所未有的控制粒度，使得真正的 compute-communication fusion 成为可能。
2. ✅ **Agentic 搜索优于纯 LLM 生成**：直接让 LLM 生成复杂分布式 kernel 成功率极低；分解任务 + 进化搜索 + 反馈闭环才是可行路径。
3. ✅ **结构化设计空间至关重要**：没有 `C = B×P×S×I×G` 这样的约束，agent 容易产生语义错误或不可编译代码。
4. ✅ **两阶段策略显著提升效率**：Fast-path 提供可靠起点，slow-path 在此基础上精细打磨，二者缺一不可。
5. ✅ **实际收益可达 1.57× 加速**：尤其在通信密集型、tile 小、skew 高的场景下优势更为明显。

---

### **方法的局限性**

- **依赖高质量 LLM**：当前效果基于 Claude Sonnet 4.5，若换成较小模型可能无法理解复杂 API。
- **搜索成本较高**：尽管有 cascade evaluation 控制成本，但完整进化仍需数十代，每次需编译运行。
- **适用范围限于已知模式**：目前聚焦于 AlltoAll、GEMM、KV-transfer 等常见模式，泛化到全新算子仍有挑战。
- **尚未集成进主流框架**：PyTorch/TensorFlow 生态尚未原生支持此类自动 fused kernel 插入。

---

### **未来工作方向**

1. **扩展 design space**：支持更多 collective ops（如 ReduceScatter、AllReduce）、异构设备（GPU+TPU）。
2. **降低 LLM 依赖**：探索轻量级规则引擎 + 小模型辅助的混合方案。
3. **在线自适应优化**：在运行时根据流量变化动态调整 fusion 策略。
4. **与 Triton、XLA 等编译器集成**：将 CUCo 作为 backend pass，实现全自动 kernel 生成。
5. **支持 fault tolerance 和弹性训练**：在动态扩缩容场景下维持 fused kernel 正确性。

---

> 📌 **一句话总结**：  
> CUCo 首次实现了 **无需训练、基于 agent 的 compute-communication fused kernel 自动生成**，通过 **结构化搜索空间 + 两阶段智能体架构**，在真实 workload 上实现了最高 **1.57× 的端到端加速**，为下一代分布式 ML 系统提供了自动化 co-design 的新范式。

</details>

---

### 4. [EdgeFLow: Serverless Federated Learning via Sequential Model Migration in Edge Networks](https://arxiv.org/abs/2603.02562)

**Authors**: Yuchen Shi, Qijun Hou, Pingyi Fan, Khaled B. Letaief  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.02562v1  

#### Abstract
Federated Learning (FL) has emerged as a transformative distributed learning paradigm in the era of Internet of Things (IoT), reconceptualizing data processing methodologies. However, FL systems face significant communication bottlenecks due to inevitable client-server data exchanges and long-distan...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《EdgeFLow: Serverless Federated Learning via Sequential Model Migration in Edge Networks》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 **Federated Learning (FL)** 架构依赖于中心化的 **cloud server** 进行模型聚合，导致以下关键瓶颈：
- **通信开销大**：本地设备需通过多跳传输将模型参数上传至远程云服务器，造成显著延迟和带宽压力；
- **网络拓扑低效**：即使采用 Hierarchical FL（HFL），仍存在边缘节点到云端的长距离通信；
- **非独立同分布（non-IID）数据** 和设备异构性加剧训练收敛缓慢。

### 提出了什么新方法或新思路
作者提出 **EdgeFLow** —— 一种**无服务器（serverless）、基于边缘网络中顺序模型迁移**的新型 FL 框架。其核心思想是：
- **移除中心云服务器**，改由边缘基站（edge base station）之间进行**顺序模型迁移（sequential model migration）**；
- 将客户端划分为固定地理集群（cluster），每轮仅一个集群活跃训练；
- 训练完成后，该集群的聚合模型直接迁移到下一个预定集群，形成“流动式”学习流程。

#### 三个阶段：
1. **Cluster Initialization**：客户端动态分组为 M 个局部集群，每个锚定在一个 edge base station 上；
2. **Intra-Cluster Training**：当前集群内客户协作训练并本地聚合；
3. **Inter-Cluster Model Migration**：更新后的全局模型直接传递给下一集群，无需回传云端。

### 相比现有方法的优势
| 方法 | 缺陷 | EdgeFLow 改进 |
|------|------|----------------|
| **Standard FL (e.g., FedAvg)** | 所有客户端直连云服务器，通信成本高 | 完全绕过云，减少跨区域传输 |
| **Hierarchical FL (HFL)** | 虽在边缘聚合，但仍需上传至云进行全局聚合 | 彻底消除云参与，实现 edge-to-edge 流动 |
| **Sequential FL** | 全 P2P 架构易引发灾难性遗忘（catastrophic forgetting） | 在 cluster 层面迁移，保留一定并行性和稳定性 |

> ✅ **优势总结**：
> - 显著降低通信开销（尤其在复杂网络拓扑下）；
> - 更适合大规模 IoT 和移动边缘计算场景；
> - 维持可证明的收敛性保障。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **FashionMNIST**：10 类服装图像分类任务；
- **CIFAR-10**：10 类自然图像分类任务，更具挑战性。

### 学习模型
- 六层 CNN 架构：
  - 卷积核大小 3×3，含 Batch Normalization；
  - 每两个卷积层后接 2×2 MaxPooling；
  - 最终两层全连接层输出 (128 → 10)，使用 Cross-Entropy Loss + Adam 优化器。

### 实验设置
- 总客户端数 $ N = 100 $，划分为 $ M $ 个集群，每集群约 10 名客户端（$ N_{m(t)} \approx 10 $）；
- 每通信轮次执行 $ K = 5 $ 个本地 epoch，mini-batch size = 64；
- **数据分布配置**：
  1. **IID**：所有客户端均匀随机分配样本；
  2. **NIID A**：
     - 10 客户端 IID；
     - 20 客户端 95%-non-IID（主类别占 95%）；
     - 70 客户端 98%-non-IID；
  3. **NIID B**：
     - 10 客户端 IID；
     - 90 客户端 100%-non-IID（极端偏斜）；

> 可视化见原文 Fig. 2，体现分布与数量双重偏斜。

### 基线方法对比
- **FedAvg**：标准联邦平均算法，作为基准；
- **EdgeFLowRand**：下一集群随机选择；
- **EdgeFLowSeq**：按预定义序列顺序迁移。

### 评估指标
1. **Accuracy (%)**：最终测试准确率；
2. **Communication Efficiency**：
   - 参数上传量（per round）；
   - **Compression Ratio** = EdgeFLow 传输量 / 原始传输量（越小越好）；
3. **Convergence Behavior**：精度随通信轮次的变化曲线；
4. **Hyperparameter Sensitivity**：不同 $ N_m $、$ K $ 下的表现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table I）
| Method        | FashionMNIST (IID) | NIID A | CIFAR-10 (IID) | NIID A | NIID B |
|---------------|--------------------|--------|----------------|--------|--------|
| **FedAvg**       | 90.60              | 86.89  | 88.66          | 77.04  | 71.04  |
| **EdgeFLowRand** | 90.13              | 87.97  | 89.16          | 80.26  | 73.14  |
| **EdgeFLowSeq**  | 90.53              | 87.50  | 88.99          | 81.58  | 73.36  |

> 🔍 **观察**：
> - 在 **IID 场景下性能接近 FedAvg**，差距微小；
> - 在 **non-IID 场景下全面超越 FedAvg**，最大提升达 **+4.54%（CIFAR-10 NIID A）**；
> - **EdgeFLowSeq 表现略优于 Rand 版本**，说明有序调度更稳定。

### 通信效率结果（Fig. 4）
在四种典型边缘网络拓扑中比较压缩比：
1. **Simple**（local-edge-cloud）
2. **Parallel**（广度优先）
3. **Linear**（深度链式）
4. **Hybrid**（混合结构）

| 方法           | Compression Ratio（相对 FedAvg/HFL） |
|----------------|---------------------------------------|
| **EdgeFLow**    | **0.2 ~ 0.5**（即节省 50%~80% 通信量） |
| 尤其在 **Linear & Hybrid** 结构中优势明显 |

> 💡 原因：随着 hop 数增加，传统方法需多次转发至云端；而 EdgeFLow 仅在相邻 edge 间迁移，路径极短。

### 消融实验结果（Fig. 3）
#### (a) 集群规模 $ N_m $ 影响
- $ N_m $ 越大 → 收敛更快、精度更高；
- 符合理论预测（Theorem 1 中 $ N_m(t) $ 出现在分母项，增大有助于减小误差界）。

#### (b) 本地 epoch $ K $ 影响
- $ K $ 增加并不总带来收益；
- 因 $ K $ 同时出现在分子与分母项（如 $ L^2K^3\eta^3G^2 $ 项），导致非单调影响；
- 存在最优 $ K $，需经验调参。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **EdgeFLow 成功实现了 serverless 的 FL 架构设计**，通过 **sequential model migration** 替代 central server，有效规避了云通信瓶颈；
2. ✅ 在 **non-IID 数据下表现优于 FedAvg**，尤其在复杂模型（CIFAR-10）上提升显著；
3. ✅ **通信开销降低 50–80%**，且在网络越深（hop 越多）时增益越大；
4. ✅ 提供了严格的 **convergence guarantee**，适用于非凸目标函数和 non-IID 数据，扩展了经典 FL 理论边界；
5. ✅ 算法对 cluster 内部 heterogeneity 施加约束（Assumption 3），使理论分析更具现实意义。

### 方法的局限性
- ❗ **训练速度可能变慢**：由于每次只有一个 cluster 活跃，缺乏并行性，总体训练时间可能延长；
- ❗ **依赖固定的 cluster 划分**：未考虑动态拓扑变化或移动客户端；
- ❗ **迁移顺序的设计影响性能**：目前仅验证了固定 vs 随机顺序，尚未引入智能调度策略；
- ❗ **未处理安全与隐私问题**：如差分隐私、拜占庭容错等未纳入框架。

### 未来工作方向
- 🔮 **Dynamic Cluster Formation**：根据地理位置、资源状态自适应调整集群划分；
- 🔮 **Wireless-Aware Scheduling**：结合信道条件优化迁移路径与时序；
- 🔮 **Integration with Over-the-Air Aggregation (AirComp)**：进一步提升通信效率；
- 🔮 **Enhanced Personalization & Safety Mechanisms**：支持个性化模型与鲁棒性训练；
- 🔮 **Test on Real-World Edge Testbeds**：部署于实际 IoT 或 6G 网络验证可行性。

---

> 📌 **总结一句话**：  
> **EdgeFLow 开创了一种去中心化、高效通信的新一代 FL 架构范式，为构建可扩展、低延迟的边缘智能系统提供了坚实基础。**

</details>

---

### 5. [Robust Heterogeneous Analog-Digital Computing for Mixture-of-Experts Models with Theoretical Generalization Guarantees](https://arxiv.org/abs/2603.02633)

**Authors**: Mohammed Nowaz Rabbani Chowdhury, Hsinyu Tsai, Geoffrey W. Burr, Kaoutar El Maghraoui, Liu Liu, Meng Wang  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.02633v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) models enable efficient scalability by activating only a small sub-set of experts per input, yet their massive parameter counts lead to substantial memory and energy inefficiency during inference. Analog in-memory computing (AIMC) offers a promising solution by elimin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Robust Heterogeneous Analog-Digital Computing for Mixture-of-Experts Models with Theoretical Generalization Guarantees  
**论文核心总结**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代 **Sparse Mixture-of-Experts (MoE)** 模型虽然通过稀疏激活实现了高效的可扩展性，但由于其巨大的参数量，在推理过程中面临严重的 **内存占用高** 和 **能量效率低下** 的问题。尽管 **Analog In-Memory Computing (AIMC)** 能有效减少数据搬移以提升能效，但其硬件非理想性（如 DAC/ADC 量化噪声、权重编程噪声）会导致模型性能显著下降。

传统缓解方案依赖 **noise-aware retraining**，但对于超大规模 MoE 模型而言，这种再训练在计算成本上是不可行的。

> **核心挑战**：如何在不进行 retraining 的前提下，实现 MoE 模型在 AIMC 上的鲁棒部署？

---

### 提出的新方法与创新思路

作者提出了一种 **无需 retraining 的异构计算框架（Retraining-Free Heterogeneous Computing Framework）**，将对模拟噪声敏感的关键模块分配到数字加速器（digital accelerator），其余部分运行于 AIMC 设备上。

#### 创新点如下：

1. **理论驱动的专家选择机制 —— Maximum Neuron Norm Score (MaxNNScore)**  
   - 定义一个新度量：`MaxNNScore(s)` = 各线性层中最大 neuron norm 的乘积（即 $\text{MaxNNorm}(W^{(s)}_{\text{up}}) \times \text{MaxNNorm}(W^{(s)}_{\text{down}})$，对于 gated-MLP 还包括 gate 层）。
   - **理论证明**：具有高频语义 token（如 "the", "a"）的专业化专家，其神经元具有更大的 $l^2$-norm，因此对 **weight-programming noise** 更敏感。
   - 因此，应优先将这些高 MaxNNScore 的专家保留在数字端执行。

2. **系统性的异构部署策略**
   - 所有 **密集激活模块**（如 MHSA、LM Head、Shared Experts）默认部署在数字端，因其处理所有输入 token，噪声影响被放大。
   - 剩余稀疏专家按 MaxNNScore 排序，前 $I'$ 分数的专家也部署在数字端，其余在 AIMC 上运行。

3. **首次为 MoE 模型提供 AIMC 部署的泛化误差保证**
   - 在简化理论模型下，证明了当部分高 MaxNNScore 专家被移到数字端后，剩余模拟专家所能容忍的编程噪声幅度可提升 $\Omega(1/\gamma)$ 倍（$\gamma$ 是低频 token 专家占比），从而保障泛化能力。

---

### 相比现有方法的优势

| 方面 | 本文方法 | 现有方法 |
|------|--------|---------|
| 是否需要 retraining | ❌ 不需要 | ✅ 通常需要（如 noise-aware FT） |
| 可扩展性 | ✅ 适用于 7B–16B 级 MoE 模型 | ❌ 多用于 <3B 小模型 |
| 决策依据 | ✅ 理论可解释（MaxNNScore） | ⚠️ 经验性（如激活频率、路由权重） |
| 性能-效率平衡 | ✅ 支持灵活调节数字/模拟比例 | ⚠️ 多为全模拟或全数字 |

---

## 2. 核心实验方法和设置

### 使用的模型与任务
- **模型**：
  - **DeepSeekMoE**（16B 参数）
  - **OLMoE**（7B 参数）
- **任务**（共 8 个 LLM benchmark）：
  - PIQA, ARC-e, ARC-c, BoolQ, HellaSwag, WinoGrande, MathQA, MMLU
  - 覆盖常识推理、数学、自然语言推断等多方面能力。

### 实验设置
- **噪声模拟工具**：使用 IBM 的 **AIHWKIT-Lightning** 模拟 AIMC 非理想性。
- **噪声类型**：
  - **DAC-ADC noise**：8-bit 量化，通过调整 `K` 和 `λ` 进行校准。
  - **Weight-programming noise**：基于 PCM 器件实测数据建模（公式见原文 Eq.3），添加与权重相关的高斯噪声。
- **NVM Tile Size**：512 × 512
- **评估指标**：
  - 平均准确率（Average Accuracy）
  - 吞吐量（Throughput, Tokens/s）
  - 能效（Energy Efficiency, Tokens/Watt·s）
  - 泛化误差边界（理论分析）

### 基线方法对比
- **全数字 FP16**：精度上限基准
- **全模拟（All-analog）**：能效上限基准
- **其他专家选择策略**（用于数字端）：
  - **Activation Frequency**：按专家激活频率排序
  - **Activation Weight**：按平均路由权重排序
  - **Router Norm**：按路由矩阵参数范数排序

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Figure 3）

#### DAC-ADC 噪声影响极小（经校准后）
| 模型 | 模块加噪 | Avg Accuracy ↓ |
|------|----------|---------------|
| DeepSeekMoE | 仅 Experts | 61.28 → 61.03 (**↓0.25**) |
| DeepSeekMoE | Experts + Dense | → 55.04 (**↓6.24**) |
| OLMoE | 仅 Experts | 63.17 → 62.46 (**↓0.71**) |
| OLMoE | Experts + Dense | → 61.54 (**↓1.63**) |

> ✅ **结论**：DAC-ADC 噪声可通过校准有效抑制；但一旦涉及 **dense modules**（如 MHSA、LM Head），性能大幅下降，必须保留于数字端。

---

#### Weight-Programming Noise 下的性能表现（Figure 4 & 5）

- 在不同噪声强度下，**MaxNNScore 策略始终优于其他 baseline**。
- 当 **12.5% 或 25% 专家** 被置于数字端时：
  - OLMoE 在噪声强度为 2.5 时，相比全模拟恢复约 **1/3 至 1/2 的性能损失**。
  - 准确率差距随噪声增大而拉大，体现其更强的鲁棒性。

> 📈 示例：在 DeepSeekMoE 中，MaxNNScore 在 25% 数字专家配置下，比 Activation Frequency 高出近 1–2 个百分点。

---

#### 能效与吞吐量权衡（Table 2，以 OLMoE 为例）

| 数字参数比例 | 架构 | Throughput (tok/s) | Energy Eff. (tok/W·s) | Accuracy @ c=1.5 |
|-------------|-------|---------------------|------------------------|------------------|
| 0% (all-analog) | Analog | 768 | **23,949** | 58.45 |
| 100% (FP16) | Digital | 4,220 | 10.55 | 63.17 |
| 5.37% (dense only) | Heterogeneous | 49,781 | 123.92 | 60.73 |
| 28.65% (dense + 25% exp) | Heterogeneous | 14,513 | 36.25 | **61.82** |

> 🔍 **观察**：
> - 异构架构在 **吞吐量** 上远超纯数字（得益于 AIMC 并行性）；
> - 能效虽低于纯模拟，但通过增加数字专家比例，可在 **精度与效率之间灵活折衷**。

---

### 消融实验结果
- **是否将 dense modules 放入模拟？**
  - 即使它们只占 **~5% 参数**，放入 AIMC 也会导致比移动 **>80% 稀疏专家** 更大的性能下降（Figure 3）。
  - ✅ 验证了“dense modules 必须放数字”的设计合理性。
- **MaxNNScore 的语义解释性验证（Figure 6）**
  - 高 MaxNNScore 专家对应高频词（如 "the", "a", "and"）；
  - 低 MaxNNScore 专家对应低频子词（如 "ach", "Ireland"）；
  - ✅ 支持了理论假设：高频 token 专家更易受噪声影响。

---

## 4. 关键结论和发现

### 主要发现
1. **并非所有专家都适合 AIMC**：少数具有大 neuron norm 的专家对 weight-programming noise 极其敏感，应保留在数字端。
2. **MaxNNScore 是有效的理论可解释指标**：它不仅能识别敏感专家，且优于经验性指标（如激活频率）。
3. **dense modules 是性能瓶颈**：尽管参数少，但由于处理所有 token，其噪声敏感度极高，必须用数字计算。
4. **异构架构实现帕累托最优**：可在保持接近全数字精度的同时，获得远高于数字系统的吞吐量与能效。

---

### 方法的局限性
- **依赖预定义的数字/模拟划分比例 $I'$**：需根据目标噪声水平手动设定，缺乏动态自适应机制。
- **理论分析基于简化模型**：假设为二分类、单 MoE 层、expert-choice routing，难以完全覆盖真实复杂场景。
- **未考虑片外通信开销**：异构系统中双设备协同可能引入额外延迟。

---

### 未来工作方向
- 设计 **runtime 动态调度机制**：根据输入 token 或资源预算，动态决定专家在 digital/AIMC 间切换。
- 扩展至 **更多模型架构**：如 dense-only Transformers、Vision MoE 等。
- 探索 **联合硬件-算法优化**：例如定制化 NVM 编程策略以降低特定专家的噪声。
- 构建 **完整的异构推理栈**：包括编译器支持、内存管理、负载均衡等系统级优化。

--- 

> ✅ **总体评价**：该论文提出了首个 **理论可解释、无需 retraining** 的 MoE 异构计算框架，在精度、能效、吞吐量之间取得了良好平衡，为大规模 MoE 模型的绿色高效部署提供了重要路径。

</details>

---

### 6. [Optimizing In-Context Demonstrations for LLM-based Automated Grading](https://arxiv.org/abs/2603.00465)

**Authors**: Yucheng Chu, Hang Li, Kaiqi Yang, Yasemin Copur-Gencturk, Kevin Haudek, Joseph Krajcik, Jiliang Tang  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.00465v1  

#### Abstract
Automated assessment of open-ended student responses is a critical capability for scaling personalized feedback in education. While large language models (LLMs) have shown promise in grading tasks via in-context learning (ICL), their reliability is heavily dependent on the selection of few-shot exem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimizing In-Context Demonstrations for LLM-based Automated Grading

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Models (LLMs)** 的自动评分系统在使用 **In-Context Learning (ICL)** 时面临两个关键挑战：
- **Exemplar selection 不够有效**：传统基于语义相似度（如 KNN + SBERT）的方法虽然能检索“看起来像”的例子，但无法捕捉到细微的 **rubric 边界差异**，导致对边界案例（borderline cases）评分不准。
- **高质量 rationale 缺乏**：专家撰写的 chain-of-thought（CoT）rationale 虽然有助于提升模型理解，但人工编写成本高、难以规模化。

这些问题导致 LLM 在开放性答案评分中可靠性不足，尤其是在相邻分数之间容易出错（如将应得 1 分的回答误判为 0 或 2 分）。

---

### 🚀 提出的新方法：GUIDE
作者提出 **GUIDE (Grading Using Iteratively Designed Exemplars)**，一个以“决策边界”为核心的迭代优化框架，其核心思想是：
> 将 exemplar selection 和 rationale generation 视为一个 **boundary-focused optimization problem**，而非简单的全局准确率最大化任务。

#### 创新点包括：
1. **Contrastive Operators for Boundary Pair Discovery**
   - 引入两种操作符：`Contrastive-Add` 和 `Contrastive-Swap`，主动寻找并插入“语义相近但得分不同”的 **boundary pairs**。
   - 这些 pair 强制模型关注 rubric 中的关键区分特征，从而更精准地划分评分等级。

2. **Discriminative Rationale Generation via Contrastive Infilling**
   - 提示 LLM 生成具有判别性的 rationale，明确解释：
     - 为什么这个回答得此分？
     - 为什么不是低一级或高一级？
   - 示例 prompt：“Explain why this deserves a 1 (not a 0 or 2)” → 显式建模边界条件。

3. **Iterative Optimization Loop**
   - 两阶段循环：
     - **Phase 1: Constrained Bayesian Optimization** —— 从候选池中选出最优 exemplar 子集，目标函数融合 accuracy、sparsity 和 contrastive density。
     - **Phase 2: Teacher-Forced Expansion** —— 使用当前最优 exemplar 集作为上下文，重新生成训练样本的 rationale，持续增强候选池质量。

4. **自动化与可扩展性强**
   - 减少对人工标注 rationale 的依赖，支持从小规模种子集启动（cold start），适用于新课程或动态变化的教学场景。

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | GUIDE 的改进 |
|------|--------|-------------|
| Random / Naive Few-Shot | 随机选择，缺乏针对性 | 主动聚焦边界案例 |
| KNN-SBERT | 只考虑语义相似性，忽略 pedagogical utility | 加入 contrastive density 指标 |
| BRIDGE [18] | 优化全局 accuracy，不强调边界区分 | 显式构造 boundary pairs + discriminative rationale |
| 手工设计 rationale | 成本高，难扩展 | 自动合成高质量 rationale |

> ✅ GUIDE 不仅告诉模型“典型例子长什么样”，更教会它“**分数之间的界限在哪里**”。

---

## 2. 核心实验方法和设置

### 📚 数据集
实验覆盖三个教育领域的真实数据集，涵盖科学教育与教师教育：

| 数据集 | 领域 | 样本数 | 分数等级 | 是否含专家 rationale |
|-------|------|--------|----------|------------------|
| **Dr** | 物理（电相互作用） | 314 | {0,1}（二元） | ❌ 无 |
| **Dc** | 化学（3DLP 框架） | ~163–184 | {0,1,2}（有序） | ✅ 每类 3 条 |
| **DT** | 教师教育（数学教学知识） | ~229–236 | {0,1,2}（有序） | ✅ 每类 3–5 条 |

所有数据集按 **train:valid:test = 3:1:1** 划分。

---

### ⚙️ 实验设置
- **Backbone Model**: GPT-4o-mini（用于 grading、rationale generation 和 inference）
- **Embedding Model**: text-embedding-3-small（计算语义相似度）
- **Optimization Rounds**: T = 5
- **Demonstration Set Size**: [4, 16]
- **Similarity Threshold for Boundary Pairs**: τ = 0.7（cosine similarity）
- **Evaluation Budget per Round**: 32 candidate subsets
- **Max Pool Size**: 512
- **Temperature**: 0.2（保证生成稳定性）

---

### 📊 评估指标
1. **Accuracy (Acc)**：完全匹配真实分数的比例。
2. **Quadratic Weighted Kappa (QWK)**：衡量序数一致性，对跨级错误惩罚更大。
3. **Adjacent Error Rate (AdjErr)**：预测偏移 ±1 的比例（反映边界模糊问题）。
4. **Non-Adjacent Error Rate (NonAdjErr)**：预测跳级（如 0→2），代表严重逻辑错误。

> 特别强调 **AdjErr**，因为这是 rubric adherence 的核心挑战。

---

### 🔁 基线方法对比
| 基线 | 类型 | 描述 |
|------|------|------|
| **Naive** | Static | 固定 few-shot 示例，无优化 |
| **Random** | Static | 随机选取 k 个示例 |
| **KNN-SBERT** | Dynamic | 按语义相似度检索最近邻 |
| **Vote-K** | Static | 最大化多样性，覆盖语义空间 |
| **BRIDGE** [18] | Optimization-based | 通过 optimize-generate 循环提升验证集准确率，但未关注边界 |

所有方法共享相同的 prompt template、instruction 和 rubric。

---

## 3. 主要实验结果和性能指标

### 📈 总体性能（见 Table 2）

| Method | Dr (Acc/QWK) | Dc (Acc/QWK) | DT (Acc/QWK) |
|--------|---------------|---------------|---------------|
| Naive | 0.74 / 0.42 | 0.69 / 0.39 | 0.59 / 0.54 |
| Random | 0.75 / 0.43 | 0.58 / 0.32 | 0.52 / 0.52 |
| KNN-SBERT | 0.78 / 0.44 | 0.62 / 0.38 | 0.60 / 0.58 |
| BRIDGE | 0.90 / 0.57 | 0.76 / 0.53 | 0.66 / 0.65 |
| **GUIDE (Ours)** | **0.92 / 0.62** | **0.80 / 0.59** | **0.71 / 0.67** |

✅ **GUIDE 在所有数据集上均取得最佳 Accuracy 和 QWK**，尤其在复杂任务 DT 上相对 naive 提升约 **20%**。

---

### 🎯 边界错误分析（AdjErr）

| Method | Dr (AdjErr) | Dc (AdjErr) | DT (AdjErr) |
|--------|--------------|--------------|--------------|
| Naive | 0.26 | 0.31 | 0.37 |
| BRIDGE | 0.19 | 0.24 | 0.32 |
| **GUIDE** | **0.08** | **0.20** | **0.28** |

📌 **关键发现**：
- GUIDE 将 **Dr 的 AdjErr 降低至原来的 1/3**（0.26 → 0.08），说明其在边界案例上的强大分辨能力。
- NonAdjErr 始终保持极低水平（≤0.02），表明 GUIDE 并未牺牲整体稳定性来换取边界精度。

---

### 🔬 与 BRIDGE 的直接比较
尽管 BRIDGE 已优于传统方法，但 GUIDE 仍实现进一步突破：
- 在 DT 上，Acc 从 0.66 → 0.71（↑7.6%），AdjErr 从 0.32 → 0.28。
- 表明：**仅优化全局 accuracy 不足以解决 rubric adherence 问题；必须显式建模边界差异**。

---

### 💡 消融实验（Ablation Study，文中隐含）
虽然没有独立表格，但从设计逻辑可推断以下关键组件的作用：
- **Contrastive Operators** 是降低 AdjErr 的主因；
- **Discriminative Rationale Generation** 显著提升了 rationale 的 pedagogical clarity；
- **Iterative Refinement** 允许 rationale 质量随轮次提升，形成正向反馈。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Boundary Cases 是自动化评分的核心难点**  
   多数错误发生在相邻分数之间，而非跨级错误。因此，exemplar selection 应优先考虑 **discriminative utility** 而非 mere representativeness。

2. **Semantic Similarity ≠ Pedagogical Utility**  
   单纯基于语义相似度的 retrieval 方法（如 KNN-SBERT）不足以支撑 rubric-adherent grading。

3. **Explicit Boundary Reasoning 是关键**  
   Discriminative rationale（即解释“为何不是其他分数”）提供了更强的学习信号，显著提升 LLM 对 rubric 的理解和应用能力。

4. **Small but Smart Context Windows Suffice**  
   GUIDE 仅需 4–16 个精心挑选的 exemplar 即可达到高性能，证明了 **quality > quantity** 的原则。

5. **High Scalability and Low Human Effort**  
   支持从少量标注启动，自动合成 rationale，适合实际教育部署。

---

### ⚠️ 局限性
1. **依赖 embedding quality**：boundary pair 发现依赖于文本嵌入的质量，在多模态或公式密集型任务中可能失效。
2. **固定 demonstration set**：推理时使用静态 exemplar set，无法针对每个 query 动态调整（虽提高一致性，但也限制灵活性）。
3. **尚未应用于大规模多题型场景**：目前实验集中在单一题目或多任务但小规模 setting。

---

### 🔮 未来工作方向
1. **扩展至 Multimodal Grading**  
   如 grading diagrams、mathematical derivations，需定义新的“semantic neighbor”度量方式。

2. **Integration with Active Learning**  
   结合 GUIDE 的 boundary detection 能力，指导教师优先审核最不确定的学生回答。

3. **Cross-domain Generalization**  
   探索是否可在某一学科训练的 GUIDE prompt 迁移到另一学科，减少重复优化成本。

4. **Real-time Classroom Deployment**  
   构建端到端系统，支持即时反馈生成，并收集学生反应以进一步优化 exemplar pool。

---

## ✅ 总结
**GUIDE** 是首个将 **automated grading 中的 exemplar optimization** 明确建模为 **boundary-focused contrastive learning problem** 的框架。通过引入 contrastive operators 和 discriminative rationale generation，实现了在物理、化学和教师教育多个真实数据集上的 SOTA 表现，尤其大幅降低了边界案例的评分误差。该工作为构建 **可信、可扩展、符合人类教学标准的 AI grading systems** 提供了重要路径。

</details>

---

### 7. [HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents](https://arxiv.org/abs/2603.00977)

**Authors**: Hongbo Jin, Rongpeng Zhu, Jiayu Ding, Wenhao Zhang, Ge Li  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.00977v1  

#### Abstract
Large language model (LLM) agents have recently demonstrated strong capabilities in interactive decision-making, yet they remain fundamentally limited in long-horizon tasks that require structured planning and reliable execution. Existing approaches predominantly rely on flat autoregressive policies...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HiMAC: Hierarchical Macro-Micro Learning for Long-Horizon LLM Agents

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Large Language Model (LLM)** 的智能体在处理**长视野任务**（long-horizon tasks）时面临三大挑战：
- **指数级探索复杂度**（exponential exploration complexity）
- **延迟奖励分配**（delayed credit assignment）
- **语义漂移**（semantic drift），即推理过程随步骤增加而偏离原始目标

现有方法普遍采用“扁平”（flat）的自回归策略，在单一 token 序列中同时生成高层推理和低层动作，导致错误传播严重、规划效率低下。

---

### 提出了什么新方法或新思路
本文提出 **HiMAC**（Hierarchical Macro-Micro Agentic Control），一种**分层强化学习框架**，将长视野决策分解为两个层级：

- **Macro-Policy（宏观策略）**：作为**规划器**（planner），负责生成结构化的自然语言子目标序列（structured blueprint），实现高层次的战略规划。
- **Micro-Policy（微观策略）**：作为**执行器**（executor），在给定蓝图下逐个完成子任务，生成具体的原子动作。

该架构实现了**战略规划与战术执行的解耦**，从根本上降低探索空间维度并遏制错误传播。

---

### 相比现有方法的优势
1. **结构化归纳偏置**（Structural Inductive Bias）  
   显式引入层次结构，而非依赖模型规模扩展，有效提升长程任务鲁棒性。

2. **无 Critic 的分层优化机制**（Critic-Free Hierarchical Policy Optimization）  
   扩展 **Group Relative Policy Optimization (GRPO)** 至双层级结构，通过层级内比较组（comparison groups）估计相对优势，避免训练不稳定的价值网络（Value Network / Critic）。

3. **迭代共进化训练策略**（Iterative Co-Evolution Training）  
   交替进行：
   - **Macro-Exploration Phase**：固定 executor，更新 planner
   - **Micro-Adaptation Phase**：固定高质量 blueprint，单独训练 executor  
   此设计缓解了双层级联合训练中的**非平稳性**（non-stationarity）问题。

4. **更高的样本效率与更强的泛化能力**  
   在多个复杂环境中达到 SOTA 性能，且收敛更快，尤其适用于稀疏奖励场景。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个具有代表性的长视野基准上进行，涵盖文本与视觉模态：

| 数据集 | 任务描述 | 特点 |
|-------|--------|------|
| **ALFWorld** | 在模拟家庭环境中执行多步操作（如“把蜡烛放进马桶”） | 多模态、具身推理、部分可观测 |
| **WebShop** | 在噪声电商网站中导航并购买指定商品 | 高维观测、真实网页交互、上下文易漂移 |
| **Sokoban** | 推箱子谜题，需精确顺序推理 | 视觉接地、空间规划、强逻辑约束 |

---

### 实验设置和评估指标

#### 模型骨干
- 文本任务（ALFWorld, WebShop）：`Qwen2.5-Instruct`（1.5B 和 7B 参数）
- 视觉任务（Sokoban）：`Qwen2.5-VL`, `Qwen3-VL` 系列 VLMs

#### 超参数配置
- 最大 prompt 长度：2048（ALFWorld）、1024（Sokoban）
- Rollout 组大小：N = 8
- KL 散度系数 β = 0.01
- 每 episode 最大步数：15（WebShop/Sokoban）

#### 评估指标
- **Success Rate (%)**：任务成功完成的比例
- **Score**：综合得分（如 WebShop 的价格折扣得分）
- **Sample Efficiency**：达到特定成功率所需的训练迭代次数

---

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **Prompting 方法** | ReAct, Reflexion |
| **RL 方法（含 Critic）** | PPO |
| **Critic-Free RL 方法** | RLOO, GRPO, GiGPO |
| **闭源模型** | GPT-4o, Gemini-2.5-Pro, Claude Sonnet 4.5 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

| 方法 | ALFWorld (7B) ↑ | WebShop Success (7B) ↑ | Sokoban Success (7B) ↑ |
|------|------------------|-------------------------|------------------------|
| **GiGPO** | 90.8% | 75.2% | 82.8% |
| **HiMAC (Ours)** | **92.1%** (+1.3pp) | **84.1%** (**+8.9pp**) | **87.5%** (**+4.7pp**) |

> 注：在 WebShop 上相比最强 RL 基线 **GiGPO** 提升达 **16% 绝对增益**（从 67.4% → 83.4%，见原文）

---

### 与基线方法的对比结果
- 在所有任务上均超越 prompt-based 和 flat RL 方法。
- 即使使用更小的 1.5B 模型，HiMAC 在 ALFWorld 上的表现（89.9%）已接近闭源的 Gemini-2.5-Pro（60.3%），凸显其结构优势远超单纯扩大模型规模。
- 在 **WebShop** 上表现尤为突出，成功率达 **83.4%（1.5B）** 和 **84.1%（7B）**，显著优于 GiGPO（67.4%），说明其对**上下文漂移**有极强抑制能力。

---

### 消融实验结果（Ablation Study，Table 3）

| 变体 | ALFWorld ↓ | WebShop Score ↓ | WebShop Succ. ↓ |
|------|------------|------------------|------------------|
| **HiMAC (Full)** | 92.1% | 93.8 | 84.1% |
| w/o Hierarchy (Flat GRPO) | 77.6% (-14.5pp) | 79.3 | 66.1% |
| w/o Iterative Co-Evolution | 85.3% (-6.8pp) | 86.7 | 74.8% |
| w/o `<sub_done>` token | 88.2% (-3.9pp) | 90.1 | 79.8% |
| Random Blueprint | 89.7% (-2.4pp) | 91.6 | 81.5% |

#### 消融分析结论：
- **移除层次结构** 导致最大性能下降，验证了分层建模的关键作用。
- **取消迭代共进化** 明显影响稳定性，尤其在 WebShop 上下降 9.3%，表明非平稳性是主要瓶颈。
- **移除 `<sub_done>` 机制** 限制了动态节奏控制，证明自适应终止的重要性。
- **随机选择蓝图训练 executor** 也会造成性能损失，说明高质量训练信号至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **结构化层次优于单纯扩大模型规模**  
   引入 **Macro-Micro 分离** 是提升长视野 LLM Agent 鲁棒性的关键因素，其效果显著超过仅靠参数增长。

2. ✅ **分层信用分配更精准稳定**  
   通过 **Hierarchical Relative Advantage Estimation**，可在无 Critic 的情况下实现高效、低方差的策略更新。

3. ✅ **迭代共进化构建隐式课程学习**  
   Planner 与 Executor 的交替优化自然形成难度递增的学习路径，促进系统协同演化。

4. ✅ **涌现行为可解释性强**  
   宏观策略在后期训练中自发发展出**自我验证机制**（如“确认物品是否放入”），体现高级认知结构的形成。

5. ✅ **样本效率高**（见 Table 4）  
   在 ALFWorld、WebShop、Sokoban 上均以更少训练迭代达到目标阈值（如 WebShop 达 65% 成功率仅需 ~220 迭代 vs GRPO 的 ~380）。

---

### 方法的局限性
- 当前蓝图仍由同一 LLM 生成与执行，未完全实现模块独立性。
- 对蓝图语义一致性缺乏显式约束，可能生成逻辑断裂的子目标链。
- 在极端开放域任务中，如何定义“可执行子目标”仍是挑战。

---

### 未来工作方向
- 将 HiMAC 应用于更开放的现实世界环境（如安卓设备控制、机器人操作）。
- 探索跨任务、跨领域的**蓝图迁移与复用机制**。
- 结合记忆机制（Memory-Augmented Architecture）增强长期状态追踪能力。
- 引入外部工具调用支持更复杂的子目标分解。

---

> 🔍 **总体评价**：HiMAC 不仅是一项技术改进，更是对 LLM Agent 架构范式的重新思考——它强调**结构设计**（structural inductive bias）比**模型规模**（model scale）更能决定长视野智能体的成败。这一思想有望成为下一代 agentic AI 的核心设计理念。

</details>

---

### 8. [Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models](https://arxiv.org/abs/2603.02631)

**Authors**: Shubhangi Upasani, Ravi Shanker Raju, Bo Li, Mengmeing Ji, John Long, Chen Wu, Urmish Thakker, Guangtao Wang  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.02631v1  

#### Abstract
Prompt length is a major bottleneck in agentic large language model (LLM) workloads, where repeated inference steps and multi-call loops incur substantial prefill cost. Recent work on speculative prefill demonstrates that attention-based token importance estimation can enable training-free prompt co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **agentic LLM workloads** 中，输入提示（prompt）长度过长导致 **prefill 阶段成本高昂**，成为推理延迟和吞吐量的主要瓶颈。传统 **Speculative Prefill** 方法依赖于与目标模型（target model）同一家族的小型草稿模型（draft model），且共享 tokenizer。然而，许多前沿模型（如 DeepSeek-V3.1、Kimi-K2）**缺乏可用的 in-family draft model**，同时实际系统中常需组合不同家族的异构模型。

因此，本文提出并验证了一个关键问题：  
> **能否使用来自不同模型家族的轻量级 draft model 来为另一个家族的 target model 进行 prompt 压缩？**

### 🚀 提出的新方法与创新
提出了 **Cross-Family Speculative Prefill**，其核心思想是：
- **沿用原始 Speculative Prefill 的算法机制**（基于 draft model 的 attention 分数估计 token 重要性）
- **打破“draft 和 target 必须同家族”的限制**，允许跨家族（cross-family）使用不同架构、不同 tokenizer 的模型组合
- 在 draft 模型上进行 token 重要性评分与 chunk 选择，在文本层面拼接后重新 tokenize 输入 target 模型
- 引入 **新的连续 position ID** 而非恢复原始位置，避免跨 tokenizer 的位置对齐难题

### 🔍 相比现有方法的优势
| 特性 | 原始 Speculative Prefill | 本文方法（Cross-Family） |
|------|------------------------|--------------------------|
| 是否需要训练 | ❌ 否（training-free） | ❌ 否 |
| 是否支持异构模型 | ❌ 否（必须同家族） | ✅ 是（Qwen ↔ LLaMA ↔ DeepSeek） |
| 是否通用性强 | 有限 | ✅ 极强，适用于任意可用长上下文 draft model |
| 实际部署灵活性 | 低 | ✅ 高（可基于成本/可用性选 draft） |

> ✅ **无需修改算法本身即可实现跨家族迁移，是一种通用的 prompt compression primitive**

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **LongBench v1 & v2**：双语多任务长上下文理解基准，涵盖问答、摘要、代码理解等
- **RULER**：极端长上下文任务，如 needle-in-a-haystack 检索、长文档 QA
- **Code Debug（来自 InfiniteBench）**：大规模代码仓库中的错误定位任务，平均输入长度达 110k–128k tokens

### ⚙️ 实验设置
- **Target Models**：
  - `Qwen-8B`, `LLaMA-3.1-8B-Instruct`, `DeepSeek-R1`, `DeepSeek-V3.1`
- **Draft Models**：
  - 跨家族组合：`Qwen3-0.6B/1.7B/4B`, `LLaMA-3.2-1B-Ins`, `LLaMA-3.1-8B-Ins`
- **关键参数**：
  - **Keep Rate (p)**：保留 token 比例（按 target tokenizer 定义）
  - **Chunk Size**：32 或 128（代码任务更大以匹配结构）
  - **Lookahead Steps N = 8**
  - **Delimiter Tokens**：如 `[...]` 或 `// omitted` 显式标记断点
- **硬件平台**：自研 Reconfigurable Dataflow Unit (RDU)，内存受限（最大支持 32k 上下文）

### 📊 评估指标
- **Accuracy (%)**：下游任务准确率 vs full-prompt baseline
- **Time-to-First-Token (TTFT)**：首 token 延迟，衡量 prefill 效率
- **Compression Ratio / Keep Rate**：控制压缩强度
- **Robustness across draft models**：不同规模 draft 的效果一致性

### 🆚 基线方法对比
- **Full Prompt Baseline**：完整输入不压缩
- **SnapStream KV Cache Compression**（用于 RULER 对比）：通过缓存剪枝优化内存，但会丢失远距离信息
- 不涉及 rewriting-based 方法（如 LLMLingua），因 focus 在 token selection 范式

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 任务 | 方法 | Keep Rate | Accuracy (%) | 相对 Baseline |
|------|------|-----------|---------------|----------------|
| **LongBench v2** | Full Prompt | 100% | 29.2–31.2 | 100% |
| | Cross-Family SP (Qwen→LLaMA) | 25% | 29.6 | ~95–100% |
| | Cross-Family SP (LLaMA→Qwen) | 50% | 30.0 | ~100% |
| | Cross-Family SP (LLaMA→DeepSeek-R1) | 6% | 54.1 | ~93% |
| **Code Debug** | Full Prompt | 100% | 67.51 (V3.1), 74.37 (R1) | 100% |
| | Cross-Family SP (LLaMA→V3.1) | 20% | 64.72 | ~96% |
| | | 15% | 59.13 | ~87.6% |
| | Cross-Family SP (LLaMA→R1) | 30% | 70.30 | ~94.5% |
| | | 15% | 62.44 | ~84.0% |
| **RULER (128k → 16k)** | Full Prompt + SnapStream | 100% | 58.0% | — |
| | Cross-Family SP (Qwen/LLaMA→DeepSeek-V3) | 12.5% | **89.67–93.36%** | ✅ **显著更高** |

### 🔬 与基线方法对比结果
- 在大多数任务中，**cross-family speculative prefill 保持了 90–100% 的 full-prompt 准确率**
- 在 **LongBench v2** 上部分配置甚至 **略高于 full-prompt baseline**，归因于 **denoising effect**（去噪效应）——移除无关上下文反而提升模型聚焦能力
- 在 **RULER** 上，由于 full-prompt 需依赖 SnapStream KV 压缩导致信息损失，而 speculative prefill 提前压缩输入，**性能反超 baseline 达 +30pp 以上**
- **TTFT 大幅降低**：
  - RULER 任务中，将 128k 输入压缩至 16k，**TTFT 从 46 秒降至约 2.5 秒，加速 ~18×**
  - 即使保守压缩到 32k，也降至 4.3 秒

### 🔍 消融实验与鲁棒性分析
- **不同 draft model 的鲁棒性高**：
  - 同一 target 下使用 `Qwen3-0.6B` 与 `Qwen3-1.7B` 表现接近
  - `LLaMA-8B` 和 `Qwen3-4B` 均可有效作为 DeepSeek-R1 的 draft
- **keep rate 可灵活调整**，性能随压缩率下降平滑退化，无剧烈波动
- **position ID 重置策略影响极小**，证明跨 tokenizer 兼容性良好

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Attention-based token importance 具有高度跨家族可迁移性**  
   尽管 draft 与 target 模型在架构、tokenizer、参数规模上有显著差异，attention 得分仍能可靠反映语义重要性。

2. **Speculative Prefill 的有效性源于 task priors 与 semantic structure，而非特定模型内部表示**  
   这表明该方法本质是一种 **generalizable prompt compression primitive**。

3. **跨模型压缩可在真实部署限制下启用长上下文能力**  
   利用具有原生长上下文支持的 draft model（如 LLaMA-8B 支持 128k），可为受限 target model（仅支持 32k）处理 128k 输入，突破硬件限制。

4. **带来显著延迟收益（TTFT ↓ ~18×）的同时几乎不损精度**

### ⚠️ 方法的局限性
- **在代码调试类任务中对极端压缩敏感**：
  - 当 keep rate ≤15%，accuracy 下降明显（保留 ~84–87% baseline）
  - 推测原因：代码错误可能依赖细粒度语法/跨文件依赖，删除低 saliency 区域易破坏局部上下文
- **当前框架为 structure-agnostic**，未考虑代码函数边界、文档章节结构等先验
- 对 highly structured input（如表格、AST）适配性未知

### 🔮 未来工作方向（A.5节指出）
- 引入 **task-aware structural constraints**：
  - 如 dependency-aware span selection
  - 强制覆盖关键函数或模块
- 扩展至更多模态与结构化数据
- 探索更高效的 chunk selection 策略（如动态 chunk size）
- 结合 KV cache management 与 prompt compression 的联合优化

---

## ✅ 总结一句话
> **Cross-family speculative prefill 成功验证了 attention-driven prompt compression 的泛化能力，使得任何具备长上下文能力的小型 draft model 都可作为“通用压缩器”，为无 in-family draft 的大模型提供高效、免训练、高保真的长上下文推理支持，尤其适合 agentic systems 中异构模型栈的实际部署场景。**

</details>

---

### 9. [Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization](https://arxiv.org/abs/2603.02701)

**Authors**: Yueyang Cang, Xiaoteng Zhang, Erlu Zhao, Zehua Ji, Yuhang Liu, Yuchen He, Zhiyuan Ning, Chen Yijun, Wenge Que, Li Shi  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.02701v1  

#### Abstract
Optimizing communication topology is fundamental to the efficiency and effectiveness of Large Language Model (LLM)-based Multi-Agent Systems (MAS). While recent approaches utilize reinforcement learning to dynamically construct task-specific graphs, they typically rely on single-sample policy gradie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Graph-GRPO: Stabilizing Multi-Agent Topology Learning via Group Relative Policy Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Model (LLM)** 的 **Multi-Agent Systems (MAS)** 在优化通信拓扑（communication topology）时，普遍采用标准的 **Reinforcement Learning (RL)** 方法（如 REINFORCE），依赖**单样本策略梯度**和**绝对奖励**（如二元正确性）。这种范式存在两大根本缺陷：

1. **高梯度方差（High Gradient Variance）**  
   - 简单任务中，即使非最优拓扑也能获得正奖励（reward=1），导致模型错误地强化冗余边；
   - 困难任务中，几乎所有拓扑都失败（reward=0），导致梯度消失，无法学习。

2. **信用分配问题（Credit Assignment Problem）**  
   - 成功的图结构中所有边被同等奖励，无法区分关键连接与冗余连接，阻碍精细结构学习。

### 🚀 提出的新方法：Graph-GRPO
提出 **Graph-GRPO**（Graph-based Group Relative Policy Optimization），首次将 **Group Relative Policy Optimization (GRPO)** 引入离散图结构搜索领域，核心思想是：

- **从绝对奖励转向相对优势（relative advantage）**：对每个查询，采样一组多样化的通信图（group sampling），以组内平均表现为基准，计算每条边的相对优势。
- **边缘级信用分配机制（Edge-Level Advantage Estimation）**：
  - 定义每条边 $ e_{ij} $ 的条件成功概率 $ S_{ij} = P(\text{Success} \mid e_{ij} \in g_k) $
  - 归一化为优势值 $ A_{ij} = (S_{ij} - \mu_s)/\sigma_s $，仅当某边显著提升成功率时才进行强化。

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 EIB-LEARNER） | Graph-GRPO |
|------|----------------------------|------------|
| 奖励机制 | 绝对奖励（absolute reward） | 相对优势（group-normalized advantage） |
| 信用分配 | 图级别（all edges rewarded equally） | 边缘级别（fine-grained edge scoring） |
| 方差控制 | 高方差，受任务难度影响大 | 动态归一化，抑制噪声 |
| 是否需要 Critic | 否（REINFORCE）或需（PPO） | ❌ 无 Critic，节省内存，更稳定 |
| 学习信号质量 | 易受“简单胜利”污染 | 过滤非信息性批次，聚焦有效结构 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在六个跨领域的 benchmark 上评估，涵盖推理与代码生成任务：

| 类别 | 数据集 |
|------|--------|
| **通用推理** | MMLU |
| **数学推理** | GSM8K, MultiArith, SVAMP, AQUA |
| **代码生成** | HumanEval |

遵循 EIB-LEARNER 的标准协议，确保可比性。

### ⚙️ 实验设置
- **Backbone LLM**: GPT-3.5-Turbo
- **Policy Network**: 基于 G-Designer 架构，使用 all-MiniLM-L6-v2 编码器 + 3层 GAT
- **Agent 数量**:
  - MMLU: 6 agents
  - HumanEval: 5 agents
  - 数学任务: 4 agents
- **Group Size $ K $**: 16
- **最大通信轮次**: 3
- **优化器**: Adam ($ lr = 1e^{-4} $)，在 NVIDIA A100 GPU 上训练

### 📊 评估指标
- **主指标**：准确率（Accuracy %）
- **辅助分析**：
  - Token 消耗（衡量效率）
  - 收敛稳定性
  - 拓扑稀疏性与语义密度

### 🆚 基线方法分类对比
| 类型 | 方法 |
|------|------|
| **Single-Agent** | Chain-of-Thought (CoT), Self-Consistency (SC) |
| **Fixed Topologies** | Chain, Tree, Complete Graph, LLM-Debate |
| **Topology Optimization Methods** | AgentPrune, AgentDropout, G-Designer, **EIB-LEARNER**（SOTA） |

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（Table 1）
| Method | MMLU | GSM8K | AQuA | MultiArith | SVAMP | HumanEval | Avg. |
|--------|------|-------|------|-----------|--------|-----------|------|
| EIB-LEARNER | 88.90 | 95.20 | 83.49 | 96.83 | 94.70 | 89.15 | **91.38** |
| **Graph-GRPO** | **90.12** | **96.10** | **84.21** | **97.07** | **96.01** | **91.25** | **92.45** |

- **平均准确率提升 +1.07%**，在所有六项任务上均超越 SOTA。
- 在复杂任务上增益更显著：
  - **HumanEval**: +2.1%
  - **GSM8K**: +0.9%

### 🔬 消融实验（Ablation Study）
比较两种优化粒度：

| 方法 | MMLU | GSM8K | HumanEval | Avg. | Δ |
|------|------|-------|----------|-----|----|
| Graph-Level GRPO | 88.54 | 94.40 | 89.07 | 90.67 | -1.82 |
| **Graph-GRPO (Edge-Level)** | **90.12** | **96.10** | **91.25** | **92.49** | — |

> **结论**：边缘级优势估计至关重要。图级奖励会强化“搭便车”边（freeloader edges），导致结构臃肿；而边缘级机制能精准识别因果路径。

### 💡 Token 效率分析（Figure 3）
- Graph-GRPO 实现 **Pareto 最优权衡**：最高准确率 + 低 token 消耗。
- 虽无显式剪枝约束，但自然收敛到**稀疏且语义丰富**的拓扑。
- **Signal-to-Token Ratio** 更高，优于 Complete Graph 和 LLM-Debate。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **任务难度引起的奖励噪声严重干扰拓扑学习**，传统 RL 方法难以稳定优化。
2. **Group-relative 机制有效过滤非信息性样本**（如简单任务中的全胜批次），防止模型学习冗余结构。
3. **边缘级信用分配是实现高效拓扑搜索的关键**，能够解耦“结构有效性”与“任务难度”。
4. Graph-GRPO 在不引入 Critic 网络的前提下，实现了更稳定、高效的训练，并自动发现高价值通信路径。

### ⚠️ 局限性
1. **可扩展性限制**：
   - 当前 GAT 主干具有 $ O(N^2) $ 复杂度，在大规模 agent swarm（如 $ N > 100 $）中可能面临计算瓶颈。
   - 需探索分层或稀疏生成策略。
2. **动态适应性不足**：
   - 当前框架为每个查询生成一个静态拓扑。
   - 对于多轮对话场景，最优通信结构可能随时间变化，缺乏 turn-level 动态调整能力。

### 🔮 未来工作方向
- 扩展至 **heterogeneous agent systems**（异构代理系统）
- 探索 **dynamic, turn-adaptive topology generation**
- 结合 **memory mechanisms** 实现长期协作演化
- 应用于 **open-ended environments**（开放环境）中的自组织 agent swarm

---

## 总结
Graph-GRPO 通过引入 **Group Relative Policy Optimization** 到多智能体拓扑学习中，从根本上解决了传统 RL 方法在离散结构搜索中的**高方差**与**信用分配模糊**问题。其实验表明，该方法不仅在多个 benchmark 上达到 SOTA 性能，还具备出色的训练稳定性与 token 效率，为构建高效、自适应的 LLM-based MAS 提供了新的范式。

</details>

---

### 10. [Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration](https://arxiv.org/abs/2603.02760)

**Authors**: Linhao Zhong, Linyu Wu, Wen Wang, Yuling Xi, Chenchen Jing, Jiaheng Zhang, Hao Chen, Chunhua Shen  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.02760v1  

#### Abstract
Diffusion large language models (dLLMs) have recently attracted significant attention for their ability to enhance diversity, controllability, and parallelism. However, their non-sequential, bidirectionally masked generation makes quality assessment difficult, underscoring the need for effective sel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Self-Evaluation for Diffusion Language Models via Sequence Regeneration**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **dLLMs 的自评估难题**：Diffusion Large Language Models (dLLMs) 采用双向掩码和非顺序生成机制，导致无法像 auto-regressive (AR) 模型那样通过链式法则分解序列概率，难以进行有效的 likelihood-based 自我评估。
- **固定长度生成限制**：传统 dLLMs 需要预设生成长度 $L$，缺乏类似 AR 模型中基于 EOS token 的动态终止机制，限制了生成灵活性。

### **提出的新方法：DiSE**
- **DiSE (Diffusion Self-Evaluation)**：一种简单而高效的 dLLMs 自评估置信度量化方法。
  - 核心思想：将整个生成序列 $X$ 输入回 dLLM，计算模型在完整上下文下“再生”每个 token 的概率。
  - 公式定义为：
    $$
    \text{DiSE}(X) = \frac{1}{|U|} \sum_{i \in U} \log p_\theta(x_i | X)
    $$
    其中 $U$ 是选定的位置集合（如 `full`, `last-10`），$p_\theta(x_i|X)$ 是模型对位置 $i$ 上 token 再生的概率。

### **相比现有方法的优势**
| 方面 | 传统方法（Monte Carlo） | DiSE（本文） |
|------|------------------------|-------------|
| **效率** | 高计算开销（需 $N_{mc}=32$ 次采样） | **仅需一次前向传播**，提速约 **32×** |
| **可靠性** | 估计不稳定，方差大 | 利用完整上下文，更稳定可靠 |
| **可解释性** | 黑箱近似 | 基于 token 再生能力，具有语义一致性基础 |
| **应用扩展** | 仅用于 likelihood 估计 | 支持 **灵活长度生成** 和 **不确定性量化** |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
分为两类任务：

#### **条件 likelihood 估计**
- **ARC-Challenge**：小学科学选择题，强调推理能力。
- **GPQA**：博士级生物、物理、化学难题，挑战人类专家。

#### **条件生成与评估**
- **Countdown**：组合数学任务，从给定数字构造目标值。
- **GSM8K**：小学数学应用题，需多步推理。
- **MATH500**：高中竞赛题子集，考察高级解题能力。
- **SVAMP**：基础算术题，测试语言鲁棒性。

### **实验设置与评估指标**

| 任务 | 评估指标 | 设置说明 |
|------|----------|---------|
| **Conditional Likelihood Estimation** | 准确率（Accuracy） | 选取最高 likelihood 的候选答案作为最终输出 |
| **Uncertainty Quantification** | ROC-AUC | 衡量模型区分正确 vs 错误答案的能力（越高越好） |
| **Flexible-length Generation** | 准确率（Accuracy） | 动态调整生成长度，比较不同基线 |

### **基线方法对比**
- **Monte Carlo Simulation**：$N_{mc}=1$ 和 $N_{mc}=32$，代表低/高成本近似。
- **Auto-regressive LLM**：LLaMA3-Instruct-8B，提供 perplexity 作为 UQ 基线。
- **Fixed-length Baselines**：固定长度 $L$ 或最大长度 $L + M_{\text{max}}$。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### ✅ **Conditional Likelihood Estimation（Table 1）**
| Model | Method | ARC-Challenge | GPQA | #NFE |
|-------|--------|---------------|------|-----|
| LLaDA-Instruct-8B | MC ($N_{mc}=1$) | 0.306 | 0.212 | 1 |
| LLaDA-Instruct-8B | MC ($N_{mc}=32$) | 0.478 | 0.286 | 32 |
| LLaDA-Instruct-8B | **DiSE (ours)** | **0.542** | **0.301** | **1** |
| LLaMA3-Instruct-8B | AR Prob. | 0.530 | 0.304 | 1 |

> 💡 **结论**：DiSE 在 **相同计算成本下显著优于 MC($N_{mc}=1$)**，甚至超越 **32倍成本的 MC**，且接近 AR 模型表现。

---

#### ✅ **Uncertainty Quantification（Table 2）**
| Model | Method | Avg. ROC-AUC |
|-------|--------|--------------|
| LLaDA-Instruct-8B | MC ($N_{mc}=1$) | 0.532 |
| LLaDA-Instruct-8B | MC ($N_{mc}=32$) | 0.573 |
| LLaDA-Instruct-8B | **DiSE (ours)** | **0.637** |
| LLaMA3-Instruct-8B | Perplexity | 0.578 |

> 💡 **提升幅度**：
> - 比 MC($N_{mc}=1$) ↑ **10.5% absolute**
> - 比 MC($N_{mc}=32$) ↑ **6.4% absolute**
> - 比 AR perplexity ↑ **5.9% absolute**

> 🔍 **消融分析（Figure 9）**：使用 `last-10`（最后10个非EOT token）效果最佳，因其靠近答案区域，信息最相关。

---

#### ✅ **Flexible-length Generation（Table 3）**
| Model | Method | Avg. Accuracy |
|-------|--------|----------------|
| LLaDA-Instruct-8B | Baseline ($L$) | 52.24 |
| LLaDA-Instruct-8B | Baseline (Max Len) | 52.38 |
| LLaDA-Instruct-8B | **DiSE-flexible (ours)** | **53.79** |
| LLaDA-1.5-8B | Baseline | 53.37 |
| LLaDA-1.5-8B | **DiSE-flexible (ours)** | **54.92** |

> 📈 **优势**：
> - 显著优于所有固定长度基线
> - 在多个 base length 下均有效
> - 无需额外训练，完全基于 self-evaluation 信号控制生成过程

> 🔧 **算法流程（Algorithm S1）**：
> 1. 初始生成长度 $L$
> 2. 计算当前 DiSE 分数
> 3. 逐步扩展末尾 mask 区域并重新生成
> 4. 若 DiSE 提升则保留；连续 $K$ 次未提升则停止

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **DiSE 是高效可靠的 self-evaluation 机制**：
   - 基于 token 再生概率，充分利用 dLLM 的泛化能力。
   - 实验验证其与 **语义连贯性** 和 **答案准确性** 正相关（Observation I & II）。
2. ✅ **DiSE 可替代昂贵的 Monte Carlo 方法**：
   - 速度提升达 **32×**，同时精度更高。
   - 在 likelihood 估计和 uncertainty quantification 中全面领先。
3. ✅ **实现真正的灵活长度生成**：
   - 首次为 dLLMs 引入无需训练的 adaptive length 控制机制。
   - 利用 DiSE 作为 stopping criterion，实现 principled 的动态终止。

### **方法的局限性**
- ❌ **尚未适配 semi-autoregressive 架构**：当前方法未在结合 AR 和 diffusion 的混合模型上验证。
- ❌ **token selection 策略依赖经验设计**：目前使用 `full` 或 `last-10` 等启发式策略，最优 subset 尚无系统性确定方法。
- ❌ **假设模型具备良好泛化能力**：方法有效性依赖于 dLLM 对扰动输入仍能准确预测 GT token 的能力。

### **未来工作方向**
1. **探索更优的 token selection 策略**：研究如何自动识别最具判别力的 token 子集用于 DiSE 计算。
2. **扩展至 semi-autoregressive 模型**：验证 DiSE 在新型 hybrid 架构中的适用性。
3. **构建端到端可微的 flexible generation 框架**：将 DiSE 信号融入训练过程，进一步优化生成策略。
4. **应用于更多下游任务**：如 hallucination detection、self-correction、active learning 等。

---

> **一句话总结**：  
> 本文提出的 **DiSE** 方法首次为 dLLMs 提供了一种 **高效、可靠、可解释的 self-evaluation 机制**，不仅大幅超越传统 Monte Carlo 方法，在 likelihood 估计和 uncertainty quantification 上取得 SOTA 表现，还催生了首个无需训练的 **flexible-length generation 框架**，推动 dLLMs 向更智能、更灵活的方向发展。

</details>

---

### 11. [MaBERT:A Padding Safe Interleaved Transformer Mamba Hybrid Encoder for Efficient Extended Context Masked Language Modeling](https://arxiv.org/abs/2603.03001)

**Authors**: Jinwoong Kim, Sangjin Park  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.03001v1  

#### Abstract
Self attention encoders such as Bidirectional Encoder Representations from Transformers(BERT) scale quadratically with sequence length, making long context modeling expensive. Linear time state space models, such as Mamba, are efficient; however, they show limitations in modeling global interactions...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MaBERT: A Padding-Safe Interleaved Transformer-Mamba Hybrid Encoder for Efficient Extended-Context Masked Language Modeling

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **长序列建模效率瓶颈**：传统基于 **self-attention** 的编码器（如 BERT）在处理长序列时面临 $O(n^2)$ 的计算复杂度，导致训练和推理成本随序列长度急剧上升。
- **SSM 在 encoder 中的适配问题**：虽然 **Mamba** 等 **State-Space Models (SSMs)** 具备线性时间复杂度 $O(n)$，但在用于 **encoder-style MLM pretraining** 时存在严重缺陷——**padding-induced state contamination**（填充诱导的状态污染）。由于 variable-length batching 引入的 `[PAD]` token 会参与 SSM 的顺序状态更新，导致有效 token 的表示被污染。

### 提出的新方法与创新思路
- **MaBERT 架构设计**：
  - 提出一种新型混合编码器 **MaBERT**，通过在层级别 **interleave（交错）Transformer 和 Mamba 层**，结合两者优势：
    - **Transformer 层**：负责全局上下文建模（global dependency modeling）。
    - **Mamba 层**：实现线性时间复杂度的序列状态累积（sequential state accumulation）。
  - 采用 **MMT (Mamba-Mamba-Transformer)** 的重复模式，在效率与表达能力之间取得平衡。

- **关键技术改进**：
  - **Padding-Safe Masking (PSM)**：
    - 在 Mamba 层中引入两阶段掩码机制：
      1. **Pre-SSM Masking**：在进入 SSM 核心前屏蔽 padding 位置的输入。
      2. **Post-Block Masking**：在残差连接和 FFN 后重新将 padding 位置置零，防止其影响上层。
    - 有效阻断 padding 对内部状态传播的影响。
  - **Mask-Aware Attention Pooling (MAP)**：
    - 替代传统的 `[CLS]` pooling，使用可学习的注意力权重对非 padding token 进行加权聚合。
    - 显式排除 padding token 影响，提升句子表示稳定性。

### 相比现有方法的优势
| 方面 | MaBERT | 传统方法（如 BERT, Longformer, BigBird） |
|------|--------|------------------------------------------|
| **计算复杂度** | 长序列下接近线性增长 | BERT: $O(n^2)$；稀疏注意力仍受限于结构限制 |
| **padding 鲁棒性** | 显式防御 state contamination | 无专门机制，易受 padding 干扰 |
| **表示质量** | 更强的句法和语义捕捉能力（尤其 CoLA） | 受限于局部或稀疏交互 |
| **扩展性** | 支持 4,096 超长上下文高效训练/推理 | 扩展代价高昂 |

---

## 2. 核心实验方法和设置

### 数据集
- **预训练数据**：
  - **BookCorpus** 和 **English Wikipedia**，仅使用 **MLM (Masked Language Modeling)** 目标进行预训练。
- **下游评估任务**：
  - **GLUE benchmark** 的全部 8 项任务：
    - 单句分类：**CoLA**, **SST-2**
    - 句对分类：**MRPC**, **QQP**, **MNLI-m/mm**, **QNLI**, **RTE**

### 实验设置与评估指标
- **预训练协议**：
  - 总步数：1M steps，按 10% / 25% / 50% / 100% 分阶段评估。
  - 序列长度调度：前 90% 步用 128 长度，后 10% 切换至 512。
  - 优化器：Adam，峰值学习率 6e-4，warmup 24k 步。
  - 批大小：256。
- **评估方式**：
  - 报告 **mean ± std** over 5 random seeds。
  - 使用各任务官方指标：
    - CoLA: **Matthews Correlation Coefficient (MCC)**
    - 其余分类任务：**Accuracy 或 F1**

### 基线方法对比
- **对比模型**：
  - **BERT** (标准 Transformer encoder)
  - **ALBERT** (参数共享轻量化版本)
  - **Longformer** (滑动窗口 + 全局注意力)
  - **BigBird** (随机 + 局部 + 全局注意力)
  - **DeBERTa** (解耦内容与位置建模)

> 所有模型使用统一预训练配方（recipe），确保比较公平。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（完整预算下，Table 2）
| Model     | CoLA ↑ | SST-2 ↑ | MRPC ↑ | QQP ↑ | MNLI-m ↑ | MNLI-mm ↑ | QNLI ↑ | RTE ↑ |  
|-----------|--------|---------|--------|-------|----------|-----------|--------|-------|
| BERT      | 0.522  | 0.912   | 0.853  | 0.856 | 0.826    | 0.829     | 0.876  | 0.618 |
| ALBERT    | 0.503  | 0.920   | 0.855  | 0.857 | 0.829    | 0.832     | 0.880  | 0.618 |
| Longformer| 0.534  | 0.924   | 0.863  | 0.858 | 0.830    | 0.831     | 0.882  | 0.626 |
| BigBird   | 0.528  | 0.926   | 0.864  | 0.857 | 0.831    | 0.832     | 0.881  | 0.624 |
| DeBERTa   | 0.617  | 0.934   | 0.862  | 0.868 | 0.838    | 0.842     | 0.886  | 0.648 |
| **MaBERT**| **0.676**| **0.933**| **0.869**| **0.879**| **0.835**  | **0.837**   | **0.893**| **0.654**|

> ✅ MaBERT 在 **8 项任务中的 5 项**（CoLA, MRPC, QQP, QNLI, RTE）取得最佳成绩，尤其在语法接受度（CoLA）上显著领先。

### 与基线方法的对比结果
- **平均 GLUE 得分趋势**（图3）：
  - MaBERT 在所有预训练预算下均表现最优，且随训练量增加持续提升。
- **长序列效率**（图6 & 表 B5–B7）：
  - 当序列长度从 512 扩展到 **4,096** 时：
    - **训练时间减少 2.36×**
    - **推理延迟降低 2.43×**
    - **峰值内存增长更缓慢**，在 4,096 长度下显著优于 DeBERTa 和 BigBird。

### 消融实验结果（Table 3 & 图4/5）
| 模型变体       | CoLA ↓ | 整体趋势 |
|----------------|--------|----------|
| Full (PSM + MAP)| 0.676  | 最优     |
| PSM only        | 0.641  | 下降，说明 MAP 重要 |
| MAP only        | 0.661  | 下降，说明 PSM 必要 |
| None (无防护)   | 0.596  | 显著下降，验证两项技术互补 |

- **Representation Stability 分析**（图4/5）：
  - 无 PSM 时，随着 padding 长度增加，有效 token 的嵌入漂移（cosine distance）明显增大。
  - 使用 **Pre+Post PSM** 组合可最大程度抑制表示漂移，证明其对深层堆叠至关重要。

---

## 4. 关键结论和发现

### 主要发现
1. **混合架构有效性**：
   - 将 **Transformer 的全局建模能力** 与 **Mamba 的线性效率** 结合，可在保持高质量表示的同时大幅提升长序列处理效率。
2. **padding 安全机制必要性**：
   - 在 encoder 中应用 SSM 必须解决 **padding-induced state contamination**，否则会导致表示退化。
   - **PSM + MAP** 是稳定 variable-length 输入的关键。
3. **性能与效率双赢**：
   - MaBERT 不仅在多个 GLUE 任务上超越强大 baseline（如 DeBERTa），还在 **超长上下文（4,096 tokens）场景下实现训练和推理加速超过 2.3×**。

### 方法的局限性
- **评估范围有限**：
  - 实验集中在 **GLUE 分类任务**，未测试真正的 **long-context reasoning**（如文档问答、摘要生成）。
- **硬件依赖性**：
  - 效率优势基于特定配置（禁用 FlashAttention、packing 等），实际部署效果可能因系统优化而异。
- **参数量更高**：
  - MaBERT 参数更多（约 205M vs BERT 109M），虽效率更高，但模型体积更大。

### 未来工作方向
- 在 **long-context understanding 和 generation benchmarks** 上进一步验证 MaBERT。
- 探索针对扩展上下文定制的 **pretraining curriculum**。
- 研究更高效的参数压缩策略以降低模型规模。

--- 

> 📌 **总结一句话**：  
> MaBERT 成功构建了一个既能高效处理超长文本、又能在标准 NLP 任务上达到 SOTA 表现的 **padding-safe hybrid encoder**，为下一代高效 encoder 设计提供了新范式。

</details>

---

### 12. [Efficient Sparse Selective-Update RNNs for Long-Range Sequence Modeling](https://arxiv.org/abs/2603.02226)

**Authors**: Bojian Yin, Shurong Wang, Haoyu Tan, Sander Bohte, Federico Corradi, Guoqi Li  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.02226v1  

#### Abstract
Real-world sequential signals, such as audio or video, contain critical information that is often embedded within long periods of silence or noise. While recurrent neural networks (RNNs) are designed to process such data efficiently, they often suffer from ``memory decay'' due to a rigid update sche...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Sparse Selective-Update RNNs for Long-Range Sequence Modeling**

---

## **1. 主要贡献和创新点**

### **解决的问题**
传统 **RNN** 在处理长序列时面临“**记忆衰减**”（memory decay）问题，其根本原因在于**刚性的更新机制**：即使输入是静止或冗余的（如长时间的空白或噪声），RNN 仍会在每个时间步强制更新隐藏状态，导致历史信息被反复覆盖，梯度难以回传到遥远过去。

此外，尽管 **Transformers** 和现代 **State Space Models (SSMs)** 在长程依赖任务上表现优异，但它们通常以 $O(T^2)$ 或全局卷积方式处理所有时间步，计算成本高，且未区分信息密度高低的时刻。

### **提出的新方法**
本文提出了 **Selective-Update RNNs (suRNNs)**，一种通过**神经元级别的二值门控**实现稀疏更新的新型 RNN 架构。其核心思想是：
- 引入一个**每神经元、每时间步的二值开关 $g_{t,i} \in \{0,1\}$**，控制是否对该神经元进行非线性更新。
- 当 $g_{t,i}=0$ 时，该神经元直接继承前一时刻状态（**exact identity carry**），不进行任何变换。
- 当 $g_{t,i}=1$ 时，才执行标准的 RNN 更新操作。

这种机制实现了：
- **精确的状态保持**：在低信息密度区间，记忆完全不变。
- **稀疏信用分配**（Sparse Credit Assignment）：梯度路径长度由“有效更新次数”决定，而非原始序列长度。

### **相比现有方法的优势**
| 对比维度 | 传统 RNN | LSTM/GRU | Transformers/SSMs | suRNN |
|--------|--------|---------|------------------|------|
| 更新机制 | 全局密集更新 | 连续门控（fractional update） | 全连接注意力或卷积 | **二值选择性更新** |
| 内存效率 | $O(1)$ | $O(1)$ | $O(T)$ | $O(1)$ |
| 推理延迟 | 低 | 低 | 高 | **低 + 可跳过计算** |
| 梯度传播 | 易消失/爆炸 | 缓解但仍有衰减 | 直接连接但计算重 | **恒定梯度路径，仅经激活步** |
| 信息感知 | ❌ | ⚠️（渐进遗忘） | ⚠️（均匀处理） | ✅（动态感知信息密度） |

> ✅ **核心优势**：在保持 RNN 的 $O(1)$ 内存和低延迟优势的同时，达到甚至超越 Transformer 的长程建模能力。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 类型 | 序列长度 | 任务描述 |
|-------|-----|---------|---------|
| **Long Range Arena (LRA)** | 多模态基准 | 1K–4K | 包括文本分类（TEXT）、图像分类（IMAGE）、路径查找（PATHFINDER）等5项任务 |
| **WikiText-103** | 语言模型 | 可变 | 下一个词预测，评估 **Perplexity (PPL)** |
| **Selective Copy** | 合成任务 | T=4096 | 输入流中提取标记符号并延后输出，测试极端长程依赖 |
| **psMNIST / sMNIST** | 图像分类 | 784 | 将 MNIST 图像按像素顺序流式输入，需整合全序列信息 |
| **sCIFAR-10** | 图像分类 | 3072 | CIFAR-10 扫描线顺序输入，更复杂的时间依赖挑战 |
| **Mackey-Glass** | 时间序列预测 | 5000 | 预测混沌系统未来值，检验对动力学结构的学习能力 |

### **实验设置与评估指标**
- **训练方式**：使用 **BPTT**，结合 **Straight-Through Estimator (STE)** 处理二值门不可导问题。
- **门控生成**：采用可学习的正弦节奏模块（rhythmic module）生成 $g_t$，支持不同神经元学习不同的更新节奏。
- **实现优化**：提出 **cuDNN-fused suGRU**，将门控信号作为额外输入通道，在单次 kernel 调用中完成稀疏更新，避免逐步步控带来的性能瓶颈。
- **评估指标**：
  - 分类任务：**Accuracy**
  - 语言模型：**Validation/Test Perplexity**
  - 合成任务：**Exact Match / Accuracy**
  - 效率：**Latency (ms/step), FLOPs, Memory Usage**

### **基线方法对比**
- **RNN 家族**：vanilla RNN, GRU, LSTM, UR-GRU/LSTM, HiPPO-RNN, LMU
- **Transformer 家族**：Vanilla Transformer, Reformer, BigBird, Linear Transformer, Performer, FNet, Nystromformer, LUNA
- **现代 SSMs**：S4, S6, Mamba, Hyena
- **其他稀疏 RNN**：Skip RNN, Clockwork RNN, Phased LSTM

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **Long Range Arena (LRA) 结果（Table 2）**
| Model | LISTOPS | TEXT | RETRIEVAL | IMAGE | PATHFINDER | **Avg** |
|-------|--------|------|-----------|-------|------------|--------|
| Transformer | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | 53.66 |
| S4 | 59.60 | 86.82 | 90.90 | 88.65 | 94.20 | **86.09** |
| **suGRU (binary)** | 44.93 | 64.42 | 77.54 | **80.32** | **84.92** | **70.43** |

> 🔹 suGRU 在严格**单向流式处理**下，平均得分远超多数 Transformer，并在 PATHFINDER 上显著优于 RWKV-v4（58.42%）。
> 🔹 尽管低于 S4，但 S4 使用了双向或全局卷积，而 suGRU 是纯因果模型。

#### **Selective Copy 准确率（Table 3）**
| Model | Accuracy |
|-------|----------|
| S4 | 18.3 |
| H3+S4 | 57.0 |
| S6 | 97.0–99.8 |
| **suGRU + S4 (2L)** | **97.2±0.3** |
| **suGRU + S4 (3L)** | **99.5±0.5** |

> ✅ suGRU 仅用两层即可媲美最强 SSM 组合，验证其强大的长程信用分配能力。

#### **WikiText-103 语言模型（Table 4）**
| Model | Val PPL | Test PPL | Params (M) |
|-------|--------|---------|-----------|
| Transformer | 24.40 | 24.78 | 44.65 |
| Mamba | 22.58 | 23.19 | 44.39 |
| **suGRU** | 19.28 | **19.20** | 44.86 |
| **suGRU (100M)** | – | **18.29** | 88.98 |
| **Hybrid-suGRU** | – | **18.03** | 48.92 |

> ✅ suGRU 在参数相当下已优于 Transformer 和多数 SSM；
> ✅ 混合架构进一步提升至 **18.03 PPL**，接近当前最优水平。

#### **Pixel-Level Classification（Table 5）**
| Model | sMNIST | psMNIST | sCIFAR |
|-------|--------|---------|--------|
| Transformer | 98.9 | 97.9 | 62.2 |
| LSSL (S4 variant) | 99.53 | 98.76 | 84.65 |
| **suGRU** | **99.53** | **98.46** | **87.26** |

> ✅ 在 sCIFAR 上大幅领先第二名 LSSL（+2.61%），显示其卓越的长期信息保留能力。

### **消融实验结果**
- **门控类型对比**：binary gate 显著优于 sigmoid（soft）gate（LRA 平均 70.43 vs 59.44）。
- **门控策略分析**：
  - 固定随机节奏（fixed random rhythm）有一定效果，但不如可学习节奏。
  - 基于输入阈值（x > 0）或固定间隔（every 3 steps）会导致训练崩溃或性能下降。
- **效率实测（Appendix D.0.3）**：
  - 在 seq-MNIST 上实现 **83% sparsity**。
  - 使用 mask-aware C 实现，推理延迟从 **466ms → 88ms**，获得 **5.3× 加速**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **二值选择性更新能有效缓解梯度消失/爆炸**：通过构建“身份直通路径”，使梯度仅穿越实际更新步骤，理论有效深度从 $O(T)$ 降至 $O(pT)$，其中 $p$ 为平均更新率。
2. ✅ **suRNN 在多种长程任务上匹敌甚至超越 Transformer 和 SSMs**，同时保持 RNN 的 $O(1)$ 推理内存和潜在低延迟优势。
3. ✅ **门控学习到了有意义的信息触发模式**：可视化显示更新集中在边缘变化、笔画起始等高信息量区域，背景则保持静默。
4. ✅ **方法具有通用性**：可应用于 RNN、GRU、LSTM、SNN 等多种递归结构（见 Figure A1），形成统一的“选择性更新”范式。

### **局限性**
1. **训练仍依赖 BPTT**：虽然效率提升，但在极长序列（如 $T>10^5$）上仍是瓶颈。
2. **门控机制引入额外复杂性**：需要设计门控生成器（如正弦节奏模块），可能影响泛化性。
3. **硬件支持有限**：当前 PyTorch 不原生支持条件执行，实际加速需定制 kernel 或依赖特定硬件（如 neuromorphic chips）。

### **未来工作方向**
1. **探索事件驱动的反向传播**：开发无需完整展开的训练算法，如 sparse checkpointing 或 event-driven BPTT。
2. **更灵活的门控生成器**：使用 MLP、SSM 或上下文感知模块替代手工设计的节奏函数。
3. **扩展至双向架构**：研究如何在非因果任务中融合 selective update 与全局上下文。
4. **与持续学习结合**：利用“子-RNN”结构分离任务间更新，减少灾难性遗忘。
5. **部署于事件驱动硬件**：充分发挥其 event-triggered computation 特性，适配 neuromorphic computing 平台。

---

> 🎯 **最终结论**：  
> **suRNN 重新定义了 RNN 的潜力边界**——它证明了在保持高效流式处理的前提下，通过**结构性地管理时间信息密度**，RNN 完全可以达到 Transformer 级别的长程建模性能。这为构建下一代高效、可持续、适用于边缘设备的大规模序列模型提供了全新路径。

</details>

---

### 13. [Generalized Discrete Diffusion with Self-Correction](https://arxiv.org/abs/2603.02230)

**Authors**: Linxuan Wang, Ziyi Wang, Yikun Bai, Wei Deng, Guang Lin, Qifan Song  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.02230v1  

#### Abstract
Self-correction is an effective technique for maintaining parallel sampling in discrete diffusion models with minimal performance degradation. Prior work has explored self-correction at inference time or during post-training; however, such approaches often suffer from limited generalization and may ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Generalized Discrete Diffusion with Self-Correction》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **Masked Diffusion Language Models (MDLMs)** 虽然支持并行生成，但在推理过程中缺乏有效的 **self-correction** 机制，导致早期生成错误无法被修正，从而影响生成质量和推理效率。尽管已有工作（如 GIDD）尝试在预训练阶段引入 self-correction，但其基于连续插值的框架存在以下问题：
- **uniform transitions** 和 **absorbing masks** 的交互不透明；
- 超参数调优困难；
- 推理时仍依赖冗余的 **remasking** 步骤（即先掩码再重采样），限制了并行 self-correction 的效率。

### **提出的新方法：Self-Correcting Discrete Diffusion (SCDD)**
本文提出 **SCDD (Self-Correcting Discrete Diffusion)** 模型，通过以下设计实现更高效、更清晰的 self-correction：
- **显式的离散状态转移建模**：在离散时间框架下重新定义前向噪声过程，明确区分 **uniform transitions** 和 **absorbing masks**。
- **解耦的噪声控制机制**：引入两个独立参数 $ p_t $ 和 $ y_t $ 分别控制：
  - $ p_t $：token 保留原始内容的概率（uniform transition SNR）；
  - $ y_t $：token 不被掩码的概率（absorbing mask SNR）。
- **完全消除 remasking**：由于 [MASK] 是吸收态（absorbing state），反向去噪过程中不会出现非 [MASK] → [MASK] 的转换，因此可直接进行 **token-to-token self-correction**，无需中间掩码步骤。

### **相比现有方法的优势**
| 特性 | GIDD | SCDD |
|------|------|------|
| 噪声机制 | 耦合的 uniform + mask 插值 | 解耦的独立控制 |
| remasking | 需要（两步纠错） | 完全消除（一步纠错） |
| 训练损失 | 动态加权，复杂 | 无重加权，理论 ELBO |
| 超参数调优 | 复杂（因插值耦合） | 更简单（参数语义清晰） |
| 并行纠错效率 | 较低 | 最高可达 **2倍提升** |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LM1B**（One Billion Words Dataset）
- **OWT**（OpenWebText）
- **Wikitext-103**（用于消融实验）

### **模型架构与训练细节**
- **骨干网络**：DiT (Diffusion transformer) small 规模，166M 参数
- **Tokenizer**：GPT-2 tokenizer
- **上下文长度**：
  - LM1B: 128
  - OWT / Wikitext-103: 512
- **训练步数与数据量**：
  - LM1B: 500K 步，33B tokens
  - OWT: 1M 步，131B tokens
- **优化器**：AdamW，学习率从 1e-6 线性上升至 5e-4 后余弦衰减
- **精度**：bfloat16 混合精度，EMA decay 0.9999

### **评估指标**
- **验证困惑度 (Val PPL)**：衡量模型拟合能力
- **生成困惑度 (Gen PPL)**：由 GPT2-large 评估生成文本质量
- **单字熵 (Unigram Entropy)**：衡量生成多样性
- **纠错率 (Correction Rate)**：$ \frac{\text{总纠正次数}}{\text{序列长度} \times \text{去噪步数}} $
- **零样本准确率**：在 ARC-E/C, BoolQ, HellaSwag 等 7 个基准上测试

### **基线方法对比**
- **MDLM**：标准 masked diffusion 模型
- **GIDD**：当前最先进的预训练 self-correction 方法（pu ∈ {0.1, 0.2}）
- **ReMDM-cap/conf**：基于推理时 remasking 的后处理 self-correction 方法

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **生成困惑度 (Gen PPL) 对比（越低越好）**
| Model | LM1B (32 steps) | OWT (128 steps) | OWT (1024 steps) |
|-------|------------------|----------------|------------------|
| MDLM | 162.6 | 104.7 | 88.5 |
| GIDD (pu=0.1) | 146.4 | 66.7 | 63.8 |
| **SCDD (pu=0.1)** | **133.5** | **67.6** | **61.3** |
| **SCDD (pu=0.2)** | **130.0** | **60.7** | **55.7** |

> 🔹 在 **32步生成** 下，SCDD 相比 GIDD+ 提升 **9.2%**；相比 ReMDM-cap 提升 **55%**  
> 🔹 在 **1024步生成** 下，SCDD(pu=0.2) 达到 **55.7 Gen PPL**，显著优于所有基线

#### ✅ **纠错能力对比**
| Model (pu=0.2) | Correction Rate @128 | @512 | @1024 |
|----------------|------------------------|------|--------|
| GIDD+ | 0.40 | 0.40 | 0.40 |
| **SCDD (ours)** | **0.71** | **0.73** | **0.75** |

> 🔹 SCDD 的纠错率随步数增长持续提升，而 GIDD 停滞不前  
> 🔹 表明 SCDD 更有效地利用额外去噪步进行渐进式修正

#### ✅ **消融实验结果**

##### （1）**不同 uniform noise 强度的影响**
- 设置最大 uniform noise ratio $ p_u \in \{10^{-6}, 0.05, 0.1, 0.2\} $
- 发现：**$ p_u $ 越大，每步平均纠错率越高**，尤其在少步生成场景下更明显
- 结论：更强的 uniform noise 促使模型学习更积极的 self-correction 策略

##### （2）**噪声峰值时间 $ t_{\text{peak}} $ 的影响**
- 控制 uniform noise 在 $ t_{\text{peak}} = 0.25, 0.5, 0.75 $ 时达到峰值
- 发现：模型会“对齐”训练噪声分布，在 $ t_{\text{peak}} $ 附近集中纠错
  - 若 $ t_{\text{peak}} = 0.75 $，则早期大量纠错（前200步完成近40%）
  - 若 $ t_{\text{peak}} = 0.25 $，则纠错延迟到最后阶段
- 结论：可通过设计噪声调度来调控 self-correction 的动态行为

---

## **4. 关键结论和发现**

### **主要发现**
1. **SCDD 实现了真正高效的并行 self-correction**：
   - 无需 remasking，直接 token-to-token 修正，纠错效率翻倍。
2. **解耦的噪声控制机制提升了可解释性和可调性**：
   - $ p_t $ 和 $ y_t $ 分别控制两种噪声，避免 GIDD 中的参数纠缠。
3. **更强的少步生成能力**：
   - 在 32-step 甚至 16-step 场景下仍保持高质量生成，适用于低延迟应用。
4. **self-correction 行为可被噪声调度引导**：
   - 通过调整 $ p_u $ 和 $ t_{\text{peak}} $，可以主动塑造模型的纠错策略。

### **方法的局限性**
- 当前实验仅在 **GPT-2 small** 规模（166M）上验证，尚未扩展到百亿以上参数模型。
- 在标准零样本任务（如 ARC, BoolQ）上表现略逊于纯 mask 模型，可能因 uniform noise 增加了学习难度。
- Gen PPL 优势显著，但实际下游任务增益需进一步探索。

### **未来工作方向**
1. **Scaling to Billion-Parameter Architectures**：
   - 将 SCDD 扩展到更大模型（如 DiT-large/xlarge），以获得更强泛化能力。
2. **结合强化学习（RL）进一步增强 self-correction**：
   - 如 LLaDA 2.1 所示，RL 可能进一步激发纠错潜力。
3. **探索更智能的 noise scheduling 策略**：
   - 动态调整 $ p_t $ 和 $ y_t $ 以适应不同输入或任务。
4. **应用于 code generation、reasoning 等复杂任务**：
   - 利用 self-correction 提升逻辑一致性与代码正确性。

---

> 📌 **一句话总结**：  
> SCDD 通过 **解耦的离散扩散框架** 和 **吸收态设计**，首次实现了 **免 remasking 的预训练 self-correction**，在保持生成多样性的前提下，显著提升了并行生成的质量与效率，尤其在少步推理场景中优势突出。

</details>

---

### 14. [PrivMedChat: End-to-End Differentially Private RLHF for Medical Dialogue Systems](https://arxiv.org/abs/2603.03054)

**Authors**: Sudip Bhujel  
**Category**: cs.CL  
**Published**: 2026-03-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.03054v1  

#### Abstract
Large language models are increasingly used for patient-facing medical assistance and clinical decision support, but adapting them to clinical dialogue often requires supervision derived from doctor-patient conversations that may contain sensitive information. Conventional supervised fine-tuning and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《PrivMedChat: End-to-End Differentially Private RLHF for Medical Dialogue Systems》核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLMs）在医疗对话系统中的应用日益广泛，但其训练通常依赖于医生-患者的真实对话数据，这些数据包含受保护的健康信息（PHI），如罕见症状组合、个人身份信息等。传统的监督微调（SFT）和基于人类反馈的强化学习（RLHF）容易导致模型**记忆化训练数据**，从而引发以下风险：

- **成员推断攻击（Membership Inference Attacks, MIA）**：攻击者可判断某条患者记录是否存在于训练集中。
- **数据提取攻击**：模型可能生成与训练数据高度相似甚至完全相同的敏感内容。
- **隐私泄露合规风险**：违反 HIPAA、GDPR 等法规。

尽管已有研究将差分隐私（Differential Privacy, DP）应用于预训练或 SFT 阶段，但尚未实现对整个 RLHF 流程（包括奖励建模和策略优化）的端到端 DP 保护。

---

### **提出了什么新方法或新思路**

本文提出 **PrivMedChat**，一个**端到端的差分私有 RLHF 框架**，用于医疗对话系统的安全对齐。其核心创新如下：

#### ✅ 创新点 1：端到端差分私有 RLHF（End-to-End DP-RLHF）
首次将 DP-SGD 应用于 RLHF 全流程三个阶段：
1. **DP-SFT**：在医生-患者对话上进行监督微调时施加 DP。
2. **DP-Reward Modeling**：从偏好对中训练奖励模型时也使用 DP-SGD。
3. **DP-PPO**：在 PPO 策略优化阶段，对 Actor 和 Critic 均应用 DP-SGD，并固定已训练好的 DP 奖励模型以避免额外隐私消耗。

总隐私预算为三阶段之和：  
$$ \epsilon_{\text{total}} = \epsilon_{\text{SFT}} + \epsilon_{\text{RM}} + \epsilon_{\text{PPO}} $$

#### ✅ 创新点 2：无标注偏好评分构建（Annotation-Free Preference Construction）
提出一种无需临床专家标注的偏好数据构造方法：
- **“选择”响应**：真实医生回复（chosen）。
- **“拒绝”响应**：用基础 LLM（如 Llama-3）模拟非专家助手生成的回答（rejected）。
- 通过语义相似度过滤（cosine similarity < 0.9）、质量过滤（长度、重复、拒绝模式）确保偏好信号清晰。

该方法显著降低人工标注成本，支持大规模奖励模型训练。

#### ✅ 创新点 3：三区隔离架构设计（Three-Zone Architecture）
- **Zone 1（私有区）**：所有访问敏感数据的操作均在此执行，强制应用 DP。
- **Zone 2（对齐区）**：仅使用公开/合成提示进行 PPO 对齐。
- **Zone 3（评估与部署区）**：完全脱离私有数据，保障推理阶段安全性。

---

### **相比现有方法的优势**

| 维度 | 现有方法 | PrivMedChat |
|------|--------|------------|
| **隐私覆盖范围** | 仅 DP-SFT 或部分阶段 | ✅ 全流程 DP-RLHF（SFT + RM + PPO） |
| **标注成本** | 依赖昂贵的人工标注 | ✅ 自动构造高质量偏好对 |
| **实用性** | 医疗场景下性能下降严重 | ✅ 在强隐私下仍保持高实用性和安全性 |
| **形式化保证** | 多数无严格 (ε, δ)-DP 保证 | ✅ 提供完整的差分隐私理论保障 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **主数据集**：`OpenMed/MedDialog` —— 大规模去标识化的医生-患者对话数据集。
  - 使用 Presidio 工具移除电子邮件、电话号码等 PII。
  - 按会话 ID 进行组感知划分（group-aware split），防止跨集泄漏。
- **辅助任务数据集**：
  - `PubMedQA`：用于评估医学问答能力（n=500）。

---

### **实验设置**

#### **模型架构**
- 基础模型：`Meta-Llama-3-8B-Instruct`
- 微调方式：全程采用 **LoRA（Low-Rank Adaptation）** 实现参数高效微调。
  - SFT：r=16, α=32
  - Reward Model & PPO：r=8, α=16/32

#### **DP 实现**
- 工具：`Opacus` + Ghost Clipping
- 梯度裁剪范数 $ C = 1.0 $
- 噪声乘子 $ \sigma $ 动态调整以满足目标 ε
- 隐私会计：Rényi Differential Privacy (RDP) 转换为 (ε, δ)-DP，δ = 1e-5

#### **训练流程**
1. 构造偏好对（expert vs. non-expert generation）
2. DP-SFT on MedDialog
3. DP-Reward Modeling on preference pairs
4. DP-PPO with fixed reward model and DP-SGD on actor/critic

---

### **评估指标**

| 类别 | 指标 | 描述 |
|------|------|------|
| **效用（Utility）** | ROUGE-L, BERTScore, Entity F1, Perplexity (PPL) | 衡量生成文本与参考医生回答的匹配程度 |
| **安全性（Safety）** | Hallucination Rate, Harmful Advice Rate, Emergency Escalation | 检测错误诊断、有害建议、紧急情况处理能力 |
| **隐私性（Privacy）** | MIA AUC（6种攻击）、Canary Extraction | 成员推断攻击成功率；能否复现插入的训练数据片段 |
| **综合质量** | LLM-as-a-Judge（G-EVAL） | 使用 Qwen-32B、Mistral-24B、Gemma-27B 组成三人评审团打分（Factuality, Safety, Helpfulness, Empathy 等维度） |

---

### **基线方法对比**

| 模型 | 是否私有 | 是否 RLHF | 说明 |
|------|---------|----------|------|
| **Base Model** | 否 | 否 | 未微调的原始 Llama-3 |
| **SFT** | 否 | 否 | 标准监督微调 |
| **Overfit SFT** | 否 | 否 | 故意过拟合版本，用于压力测试 MIA |
| **PPO** | 否 | 是 | 非私有的完整 RLHF |
| **DP-SFT** | 是 | 否 | 仅在 SFT 阶段应用 DP |
| **PrivMedChat** | ✅ 是 | ✅ 是 | 本文提出的端到端 DP-RLHF 方法 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据汇总（ε=7, δ=1e-5）**

| 模型 | ROUGE-L ↑ | BERTScore ↑ | Entity F1 ↑ | Hallucination ↓ | Harmful Advice ↓ | MIA AUC ↓ | Overall Score ↑ |
|------|-----------|-------------|--------------|------------------|--------------------|------------|------------------|
| SFT | 0.163 | 0.829 | 0.096 | 2.2% | 0.2% | ~0.53–0.55 | 2.40 |
| DP-SFT | 0.154 | 0.833 | 0.094 | 2.6% | 0.2% | 0.510–0.546 | 2.48 |
| **PrivMedChat (ε=7)** | **0.156** | **0.836** | **0.103** | **1.4%** | **0.4%** | **0.510–0.545** | **2.86** |

> 🔍 注：**Bold** 表示所有 DP 模型中的最优值。

---

### **与基线方法的对比结果**

#### ✅ **效用方面**
- PrivMedChat 在 ε=7 下达到 **最高 ROUGE-L（0.156）** 和 **最高 Entity F1（0.103）**，优于所有其他 DP 模型。
- BERTScore 达到 0.836，接近非私有模型水平。
- 在 PubMedQA 上准确率稳定在 54.6%，虽略低于非 DP 模型，但表现出更强的事实一致性。

#### ✅ **安全性方面**
- **幻觉率降至 1.4%**（DP-SFT 平均为 2.3%），表明 RLHF 有效抑制了虚构内容。
- **有害建议率仅为 0.4%**，且在多个预算下均维持低位。
- 紧急转诊行为稳定（~4.4%），说明 DP 噪声未损害关键临床判断。

#### ✅ **隐私性方面**
- 所有 MIA 攻击的 AUC 均在 **0.510–0.555 之间**，接近随机猜测（0.50），远低于非私有模型的风险区间。
- **Lowercase normalization 攻击出现反向信号（AUC < 0.5）**，进一步证明成员与非成员无法区分。
- **Canary Extraction 成功率为 0/25**，无任何明文记忆现象。

#### ✅ **综合质量（LLM-Judge）**
- PrivMedChat 在 **事实性、安全性、共情能力** 上全面领先。
- 总体评分达 **2.86 分（满分？）**，显著高于 DP-SFT（2.48）和标准 SFT（2.40）。

---

### **消融实验结果**

- **DP 噪声不影响奖励模型有效性**：即使奖励模型本身经过 DP 训练，其指导下的 PPO 策略仍能超越非私有 PPO。
- **隐私预算 ε 变化影响有限**：在 ε ∈ {1,3,5,7} 范围内，性能波动极小（ROUGE-L: 0.146–0.156），说明可在不牺牲太多效用的前提下选择更高隐私强度。
- **六类 MIA 攻击表现一致**：所有攻击均无法突破 chance level，验证了隐私防护的鲁棒性。

---

## 4. 关键结论和发现

### **主要发现**

1. **端到端 DP-RLHF 是可行且有效的**：
   - 尽管 DP-SGD 引入噪声，但结合 LoRA 和偏好对齐机制，仍能在强隐私约束下恢复大部分性能损失。
   - RLHF 阶段不仅没有加剧隐私风险，反而提升了安全性和事实准确性。

2. **隐私与效用并非零和博弈**：
   - 在合理配置下（如 ε=7），PrivMedChat 实现了**隐私、效用、安全三者的帕累托最优**。
   - 医疗领域特有的偏好结构有助于缓解 DP 噪声带来的负面影响。

3. **自动化偏好构建具有高性价比**：
   - “专家 vs. 非专家”生成策略可低成本获得高质量训练信号，适用于资源受限的医疗机构。

4. **形式化隐私保障至关重要**：
   - 即使非私有模型在某些攻击下表现“看似安全”，也只有 DP 模型提供严格的 (ε, δ)-DP 保证，抵御未知攻击。

---

### **局限性**

1. **计算开销大**：
   - DP-SGD 需要 per-sample gradients，导致训练时间增加约 2–3 倍。
   - 多阶段隐私会计复杂，需精细调参。

2. **偏好数据仍是代理信号**：
   - 自动生成的“拒绝”响应可能不够多样化或存在偏差，依赖基础模型的能力。

3. **评估仍基于静态基准**：
   - 缺乏真实临床环境下的动态交互测试和长期漂移分析。

4. **多数为 example-level DP**：
   - 当前主要考虑单样本相邻性，user-level 或 conversation-level DP 更具挑战性，尚未完全解决。

---

### **未来工作方向**

1. **阶段自适应隐私预算分配**：
   - 探索不同阶段（SFT/RM/PPO）的敏感度差异，动态分配 ε 以最大化效用。

2. **增强不确定性表达与拒绝机制**：
   - 在 DP 噪声下提升模型识别“我不知道”的能力，减少过度自信输出。

3. **多模态医疗输入扩展**：
   - 将框架推广至影像报告、电子病历等结构化+非结构化混合数据。

4. **对抗性 MIA 攻击研究**：
   - 开发更强大的自适应攻击方法，检验当前防御的边界。

5. **前瞻性临床验证**：
   - 与医院合作开展医生-in-the-loop 实验，评估实际辅助诊疗效果。

---

> 📦 **开源信息**：作者已将代码开源 → [GitHub: sudip-bhujel/privmedchat](https://github.com/sudip-bhujel/privmedchat)

</details>

---

### 15. [Enhancing Physics-Informed Neural Networks with Domain-aware Fourier Features: Towards Improved Performance and Interpretable Results](https://arxiv.org/abs/2603.02948)

**Authors**: Alberto Mi\~no Calero, Luis Salamanca, Konstantinos E. Tatsis  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.02948v1  

#### Abstract
Physics-Informed Neural Networks (PINNs) incorporate physics into neural networks by embedding partial differential equations (PDEs) into their loss function. Despite their success in learning the underlying physics, PINN models remain difficult to train and interpret. In this work, a novel modeling...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Enhancing Physics-Informed Neural Networks with Domain-aware Fourier Features

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **PINNs 的训练困难**：传统 Physics-Informed Neural Networks（PINNs）在训练过程中面临多目标优化难题，即需要同时最小化 PDE 残差、边界条件（BCs）和初始条件（ICs）损失项，导致梯度刚性（gradient stiffness）、收敛慢、难以平衡损失权重。
- **频谱偏差（spectral bias）**：标准 PINNs 倾向于学习低频解，难以捕捉高频率物理现象。
- **可解释性缺失**：尽管嵌入了 PDE 物理先验，PINNs 仍是“黑箱”模型，缺乏对输入特征如何影响预测的可解释机制。

### 🚀 提出的新方法
提出了一种名为 **Domain-aware Fourier Features (DaFFs)** 的新型输入编码方式，并构建了 **PINN-DaFFs** 模型框架。

#### 核心思想：
- **DaFFs 来源于域特定的拉普拉斯算子特征分解**：
  $$
  -\nabla^2 \phi_j(\mathbf{x}) = \lambda_j \phi_j(\mathbf{x}), \quad \text{在 } \Omega \text{ 上}, \quad \text{s.t. } B[\phi_j] = 0 \text{ 在 } \partial\Omega
  $$
  这些特征函数 $\phi_j$ 天然满足齐次 Dirichlet、Neumann 或 Robin 边界条件。
- 将这些 $\phi_j$ 作为网络输入的傅里叶基函数，替代传统的 Random Fourier Features（RFFs）。

### ⭐ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **训练效率** | 不再需要显式建模 BC/IC 损失项，将多目标优化简化为单目标（仅 PDE 残差），无需 loss balancing 技术（如 ReLoBRaLo）。 |
| **性能提升** | 实现 **orders-of-magnitude 更低的误差**，更快收敛速度。 |
| **物理一致性** | DaFFs 内在满足边界条件，使模型输出更符合物理规律。 |
| **可解释性增强** | 结合 **Layer-wise Relevance Propagation (LRP)** 构建 XAI 框架，揭示不同 DaFF 成分对预测的贡献，实现“物理一致”的归因分析。 |
| **输入表示优越性** | 相比 RFFs 的随机性和冗余性，DaFFs 是结构化、领域相关的，具有更强的表达能力和泛化潜力。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集 / 数值问题
论文采用两个经典的偏微分方程（PDE）数值求解任务进行验证：

1. **Kirchhoff-Love Plate Bending Equation**
   - 描述薄板弯曲行为的四阶 PDE。
   - 包含混合边界条件（Dirichlet 和 Neumann 类型）。
   - 域：矩形区域 $\Omega = [0,a] \times [0,b]$。

2. **Helmholtz Equation**
   - 形式为 $\Delta u + k u = f$ 的二阶椭圆型 PDE。
   - 应用于波动问题等。
   - 域：正方形区域 $\Omega = [-1,1]^2$，齐次 Dirichlet 边界条件。

> 注：所有实验均为 **zero-data setting** —— 无真实观测数据，仅通过 collocation points 上的 PDE 残差进行监督训练。

### 🔧 实验设置与评估指标

| 设置项 | 说明 |
|--------|------|
| **Collocation Points** | 总数固定，其中 3/4 分布在域内（interior），1/4 在边界上（boundary）。 |
| **Batch Size** | 512, 1024, 2048 |
| **网络结构** | Fully-connected NN，层数 2–5，每层单元数 64–256 |
| **优化器** | Adam + BFGS（后阶段） |
| **训练轮数** | 50,000 epochs |
| **Early Stopping** | Patience = 2000 |
| **超参搜索** | 两阶段：手动初筛 + 随机搜索（random samples） |

#### 评估指标：
- **Training Loss**：PDE 残差平均损失（对 vanilla PINN 和 PINN-RFFs 还包括 BC/IC 损失）。
- **Validation Loss**：在规则网格上的 **MSE（Mean Squared Error）**，与真实解析解比较。
- **Training Time**：总训练耗时。
- **LRP Attribution**：用于可解释性分析，衡量各 Fourier Feature 对输出的相关性贡献。

### 🆚 基线方法对比
| 方法 | 简介 |
|------|------|
| **Vanilla PINN** | 原始 PINN 架构，直接输入 $(x,t)$，需多个 loss terms 并使用 loss balancing。 |
| **PINN-RFFs** | 使用 Random Fourier Features 编码输入，缓解 spectral bias，但仍需处理边界损失和 loss balancing。 |
| **PINN-DaFFs (Ours)** | 使用 Domain-aware Fourier Features，自动满足边界条件，仅保留 PDE loss。 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 1 & 2）

#### 表 1：Kirchhoff-Love 问题结果
| 指标 | Vanilla PINN | PINN-RFFs | **PINN-DaFFs** |
|------|--------------|-----------|----------------|
| Training Loss | 2.71e-04 | 7.74e-06 | **1.02e-09** |
| Validation Loss | 2.75e-06 | 3.01e-07 | **6.42e-18** |
| Training Time | 1h03m | 44m09s | **45m51s** |

> 💡 **结论**：PINN-DaFFs 的 validation loss 比最优基线（PINN-RFFs）低 **约 11 个数量级**！

#### 表 2：Helmholtz 问题结果
| 指标 | Vanilla PINN | PINN-RFFs | **PINN-DaFFs** |
|------|--------------|-----------|----------------|
| Training Loss | 1.23e-03 | 8.23e-04 | **3.86e-05** |
| Validation Loss | 5.48e-05 | 7.75e-06 | **2.32e-11** |
| Training Time | 2h51m | 1h13m | **31m25s** |

> 💡 **结论**：不仅精度显著提升（validation loss 降低约 5 个数量级以上），且训练时间缩短近 **60%**。

### 🔍 消融实验与关键观察
- **Loss Balancing 是否必要？**
  - Vanilla PINN 和 PINN-RFFs 必须使用 adaptive loss balancing（如 ReLoBRaLo）才能稳定训练。
  - **PINN-DaFFs 完全不需要**，因其天然满足 BCs，只需优化单一 PDE loss。
- **收敛稳定性**：
  - 图 2 和图 3 显示，PINN-DaFFs 虽然 loss 曲线波动较大，但整体下降迅速且最终达到极低水平。
  - 基线方法存在明显的 loss competition 和震荡。
- **LRP 可解释性分析**：
  - **Vanilla PINN**：空间坐标 $x$ 和 $y$ 的贡献分布杂乱、不对称，不符合物理对称性预期。
  - **PINN-RFFs**：RFF 特征的重要性分布随采样随机变化，缺乏一致性，无法指导特征选择。
  - **PINN-DaFFs**：DaFF 成分的 relevance attribution 更加集中且物理合理。例如，在 Helmholtz 问题中，高频模式 $(m=8,n=2)$ 贡献更大，与真实解频率匹配。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DaFFs 显著提升了 PINNs 的性能与效率**：
   - 通过将物理域的几何与边界信息编码进输入特征，实现了 orders-of-magnitude 的误差降低。
   - 单一目标优化极大简化了训练流程，避免了复杂的 loss balancing。

2. **DaFFs 改善了模型的可解释性**：
   - 提出的 LRP-based XAI 框架能有效提取 DaFF 输入的 relevance scores。
   - 发现 PINN-DaFFs 的归因模式更具物理一致性，而 RFFs 和 vanilla PINNs 的归因则分散且不具可重复性。

3. **DaFFs 具备理论合理性与通用性**：
   - 基于 Laplace 算子特征系统构造，自然兼容多种边界类型（Dirichlet, Neumann, Robin）。
   - 可扩展至非齐次边界条件（通过解的分解策略）。

### ⚠️ 局限性
- 当前 DaFFs 构造依赖于对简单几何形状（如矩形、圆形）的特征函数解析或数值求解，在复杂不规则域中计算成本可能上升。
- 所有实验基于 zero-data setting，尚未验证在部分观测数据下的表现。
- LRP 的可靠性依赖于模型本身的性能；低质量模型产生的 attribution 可能误导特征选择。

### 🔮 未来工作方向
1. **推广到非齐次边界条件与更复杂物理系统**（如 Navier-Stokes 方程）。
2. 探索 DaFFs 在 **sparse observation 场景** 下的应用，结合贝叶斯推断等方法。
3. 利用 XAI（如 LRP）进一步诊断模型偏差、改进架构设计。
4. 开发自动化工具生成任意域的 DaFFs（例如结合 FEM/FD 数值求解器）。

---

## 总结
该论文提出的 **PINN-DaFFs** 框架通过引入 **Domain-aware Fourier Features**，从根本上重构了 PINNs 的输入表示方式，成功解决了传统方法中的 **多目标优化困境** 与 **频谱偏差问题**。实验表明其在 **精度、收敛速度、训练稳定性** 上全面超越 vanilla PINN 和 PINN-RFFs。更重要的是，结合 **LRP** 的可解释性分析揭示了该方法具备更强的 **物理一致性**，为发展“可信、可理解”的科学机器学习模型提供了新路径。

</details>

---

### 16. [SWE-Hub: A Unified Production System for Scalable, Executable Software Engineering Tasks](https://arxiv.org/abs/2603.00575)

**Authors**: Yucheng Zeng, Shupeng Li, Daxiang Dong, Ruijie Xu, Zimo Chen, Liwei Zheng, Yuxuan Li, Zhe Zhou, Haotian Zhao, Lun Tian, Heng Xiao, Tianshu Zhu, Longkun Hao, Jianmin Wu  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.00575v1  

#### Abstract
Progress in software-engineering agents is increasingly constrained by the scarcity of executable, scalable, and realistic data for training and evaluation. This scarcity stems from three fundamental challenges in existing pipelines: environments are brittle and difficult to reproduce across languag...

---

### 17. [Fair in Mind, Fair in Action? A Synchronous Benchmark for Understanding and Generation in UMLLMs](https://arxiv.org/abs/2603.00590)

**Authors**: Yiran Zhao, Lu Zhou, Xiaogang Xu, Zhe Liu, Jiafei Wu, Liming Fang  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.00590v1  

#### Abstract
As artificial intelligence (AI) is increasingly deployed across domains, ensuring fairness has become a core challenge. However, the field faces a "Tower of Babel'' dilemma: fairness metrics abound, yet their underlying philosophical assumptions often conflict, hindering unified paradigms-particular...

---

### 18. [TraceSIR: A Multi-Agent Framework for Structured Analysis and Reporting of Agentic Execution Traces](https://arxiv.org/abs/2603.00623)

**Authors**: Shu-Xun Yang, Cunxiang Wang, Haoke Zhang, Wenbo Yu, Lindong Wu, Jiayi Gui, Dayong Yang, Yukuo Cen, Zhuoer Feng, Bosi Wen, Yidong Wang, Lucen Zhong, Jiamin Ren, Linfeng Zhang, Jie Tang  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.00623v1  

#### Abstract
Agentic systems augment large language models with external tools and iterative decision making, enabling complex tasks such as deep research, function calling, and coding. However, their long and intricate execution traces make failure diagnosis and root cause analysis extremely challenging. Manual...

---

### 19. [Harmonizing Dense and Sparse Signals in Multi-turn RL: Dual-Horizon Credit Assignment for Industrial Sales Agents](https://arxiv.org/abs/2603.01481)

**Authors**: Haojin Yang, Ai Jian, Xinyue Huang, Yiwei Wang, Weipeng Zhang, Ke Zeng, Xunliang Cai, Jingqing Ruan  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.01481v1  

#### Abstract
Optimizing large language models for industrial sales requires balancing long-term commercial objectives (e.g., conversion rate) with immediate linguistic constraints such as fluency and compliance. Conventional reinforcement learning often merges these heterogeneous goals into a single reward, caus...

---

### 20. [Graph-Based Self-Healing Tool Routing for Cost-Efficient LLM Agents](https://arxiv.org/abs/2603.01548)

**Authors**: Neeraj Bholani  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.01548v1  

#### Abstract
Tool-using LLM agents face a reliability-cost tradeoff: routing every decision through the LLM improves correctness but incurs high latency and inference cost, while pre-coded workflow graphs reduce cost but become brittle under unanticipated compound tool failures. We present Self-Healing Router, a...

---

### 21. [S5-HES Agent: Society 5.0-driven Agentic Framework to Democratize Smart Home Environment Simulation](https://arxiv.org/abs/2603.01554)

**Authors**: Akila Siriweera, Janani Rangila, Keitaro Naruse, Incheon Paik, Isuru Jayanada  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.01554v1  

#### Abstract
The smart home is a key domain within the Society 5.0 vision for a human-centered society. Smart home technologies rapidly evolve, and research should diversify while remaining aligned with Society 5.0 objectives. Democratizing smart home research would engage a broader community of innovators beyon...

---

### 22. [MuxTune: Efficient Multi-Task LLM Fine-Tuning in Multi-Tenant Datacenters via Spatial-Temporal Backbone Multiplexing](https://arxiv.org/abs/2603.02885)

**Authors**: Chunyu Xue, Yi Pan, Weihao Cui, Quan Chen, Shulai Zhang, Bingsheng He, Minyi Guo  
**Category**: cs.DC  
**Published**: 2026-03-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.02885v1  

#### Abstract
Parameter-Efficient Fine-Tuning (PEFT) is widely applied as the backend of fine-tuning APIs for large language model (LLM) customization in datacenters. Service providers deploy separate instances for individual PEFT tasks, giving rise to prominent resource inefficiencies, including (1) GPU underuti...

---

### 23. [Bridging Diffusion Guidance and Anderson Acceleration via Hopfield Dynamics](https://arxiv.org/abs/2603.02531)

**Authors**: Kwanyoung Kim  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.02531v1  

#### Abstract
Classifier-Free Guidance (CFG) has significantly enhanced the generative quality of diffusion models by extrapolating between conditional and unconditional outputs. However, its high inference cost and limited applicability to distilled or single-step models have shifted research focus toward attent...

---

### 24. [EmCoop: A Framework and Benchmark for Embodied Cooperation Among LLM Agents](https://arxiv.org/abs/2603.00349)

**Authors**: Hanqing Yang, Shiyu Chen, Narjes Nourzad, Marie Siew, Jingdi Chen, Carlee Joe-Wong  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00349v1  

#### Abstract
Real-world scenarios increasingly require multiple embodied agents to collaborate in dynamic environments under embodied constraints, as many tasks exceed the capabilities of any single agent. Recent advances in large language models (LLMs) enable high-level cognitive coordination through reasoning,...

---

### 25. [NeuroHex: Highly-Efficient Hex Coordinate System for Creating World Models to Enable Adaptive AI](https://arxiv.org/abs/2603.00376)

**Authors**: Quinn Jacobson, Joe Luo, Jingfei Xu, Shanmuga Venkatachalam, Kevin Wang, Dingchao Rong, John Paul Shen  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00376v2  

#### Abstract
NeuroHex is a hexagonal coordinate system designed to support highly efficient world models and reference frames for online adaptive AI systems. Inspired by the hexadirectional firing structure of grid cells in the human brain, NeuroHex adopts a cubic isometric hexagonal coordinate formulation that ...

---

### 26. [LOGIGEN: Logic-Driven Generation of Verifiable Agentic Tasks](https://arxiv.org/abs/2603.00540)

**Authors**: Yucheng Zeng, Weipeng Lu, Linyun Liu, Shupeng Li, Zitian Qu, Chenghao Zhu, Shaofei Li, Zhengdong Tan, Mengyue Liu, Haotian Zhao, Zhe Zhou, Jianmin Wu  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00540v1  

#### Abstract
The evolution of Large Language Models (LLMs) from static instruction-followers to autonomous agents necessitates operating within complex, stateful environments to achieve precise state-transition objectives. However, this paradigm is bottlenecked by data scarcity, as existing tool-centric reverse-...

---

### 27. [Advancing Multimodal Judge Models through a Capability-Oriented Benchmark and MCTS-Driven Data Generation](https://arxiv.org/abs/2603.00546)

**Authors**: Zeyu Chen, Huanjin Yao, Ziwang Zhao, Min Yang  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.00546v1  

#### Abstract
Using Multimodal Large Language Models (MLLMs) as judges to achieve precise and consistent evaluations has gradually become an emerging paradigm across various domains. Evaluating the capability and reliability of MLLM-as-a-judge systems is therefore essential for ensuring trustworthy assessment. Ex...

---

### 28. [Agents Learn Their Runtime: Interpreter Persistence as Training-Time Semantics](https://arxiv.org/abs/2603.01209)

**Authors**: Victor May, Aaditya Salgarkar, Yishan Wang, Diganta Misra, Huu Nguyen  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.01209v1  

#### Abstract
Tool-augmented LLMs are increasingly deployed as agents that interleave natural-language reasoning with executable Python actions, as in CodeAct-style frameworks. In deployment, these agents rely on runtime state that persists across steps. By contrast, common training pipelines treat agent traces a...

---

### 29. [Exploring Plan Space through Conversation: An Agentic Framework for LLM-Mediated Explanations in Planning](https://arxiv.org/abs/2603.02070)

**Authors**: Guilhem Fouilh\'e, Rebecca Eifler, Antonin Poch\'e, Sylvie Thi\'ebaux, Nicholas Asher  
**Category**: cs.AI  
**Published**: 2026-03-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.02070v1  

#### Abstract
When automating plan generation for a real-world sequential decision problem, the goal is often not to replace the human planner, but to facilitate an iterative reasoning and elicitation process, where the human's role is to guide the AI planner according to their preferences and expertise. In this ...

---

### 30. [ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](https://arxiv.org/abs/2603.02216)

**Authors**: Ruike Cao, Shaojie Bai, Fugen Yao, Liang Dong, Jian Xu, Li Xiao  
**Category**: cs.LG  
**Published**: 2026-03-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.02216v1  

#### Abstract
Effective information seeking in multi-turn medical dialogues is critical for accurate diagnosis, especially when dealing with incomplete information. Aligning Large Language Models (LLMs) for these interactive scenarios is challenging due to the uncertainty inherent in user-agent interactions, whic...

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
