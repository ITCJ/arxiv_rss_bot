# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-24 06:47:50 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SparseDVFS: Sparse-Aware DVFS for Energy-Efficient Edge Inference](https://arxiv.org/abs/2603.21908)

**Authors**: Ziyang Zhang, Zheshun Wu, Jie Liu, Luca Mottola  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.21908v1  

#### Abstract
Deploying deep neural networks (DNNs) on power-sensitive edge devices presents a formidable challenge. While Dynamic Voltage and Frequency Scaling (DVFS) is widely employed for energy optimization, traditional model-level scaling is often too coarse to capture intra-inference variations, whereas fin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《SparseDVFS: Sparse-Aware DVFS for Energy-Efficient Edge Inference》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在边缘设备上部署深度神经网络（DNN）面临严峻的能效挑战。传统的 **DVFS**（Dynamic Voltage and Frequency Scaling）策略存在以下两大瓶颈：
- **模型级DVFS**（Model-level DVFS）粒度过粗，无法捕捉推理过程中不同算子间的计算强度差异；
- **算子级DVFS**（Operator-level DVFS）虽精细，但频繁的频率切换带来显著的硬件切换延迟（switching latency），导致端到端延迟剧增，抵消节能收益。

此外，现有方法缺乏对 **DNN内部稀疏性**（sparsity）等结构性特征的感知，难以实现精准的功耗-性能权衡。

---

### **提出的新方法与创新思路**
论文提出了 **SparseDVFS** —— 一种**稀疏性感知的块级DVFS框架**，通过三个核心组件协同优化能效：

#### **(1) 离线建模器（Offline Modeler）**
- 利用 **白盒时序分析**（white-box timeline analysis）建立 **算子稀疏度** 与最优 **CPU/GPU/EMC 频率三元组** 的确定性映射。
- 引入 **热感知功耗模型**（thermal-aware power model），解耦动态功耗与静态漏电功耗，支持主动热管理。

#### **(2) 运行时图划分器（Runtime Graph Partitioner）**
- 采用 **贪心合并算法** 将相邻算子聚合成 **超块**（super-blocks）。
- 引入 **延迟摊销约束**（latency amortization constraint）：仅当超块执行时间 $ T_{\text{block}} > N \times T_{\text{switch}} $ 时才允许频率切换，有效摊销切换开销。

#### **(3) 统一协同控制器（Unified Co-Governor）**
- 采用 **FUSE**（Frequency Unified Scaling Engine）策略，将 CPU、GPU、内存频率作为耦合向量统一调度。
- 引入 **前向预取机制**（look-ahead instruction queue），在当前超块执行时提前提交下一阶段的频率配置，隐藏切换延迟。
- 对稀疏块降低 GPU 频率、提升 EMC 频率；对密集块则相反，实现资源精准匹配。

---

### **相比现有方法的优势**
| 方面 | 传统方法 | SparseDVFS |
|------|----------|------------|
| **粒度控制** | 要么太粗（模型级），要么太细（算子级） | 自适应块级，兼顾精度与开销 |
| **稀疏性感知** | 缺乏 | 显式建模 sparsity 作为核心调控信号 |
| **系统协同** | 各组件独立调控，易产生拮抗效应 | 统一协调 CPU/GPU/EMC，消除 pipeline stall |
| **切换延迟处理** | 忽略或被动应对 | 主动通过前向预取机制隐藏延迟 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与模型**
- **数据集**：ImageNet-2012 验证集（约 5 万张图像）
- **模型架构**：
  - **CNN 类**：ResNet-18、ResNet-101
  - **Transformer 类**：ViT-B16、ViT-L16
- 所有模型均转换为 ONNX 格式并进行推理优化。

### **实验平台**
- **硬件**：NVIDIA Jetson Orin Nano（8GB DRAM）
- **软件栈**：JetPack 6.0，Linux Kernel 5.15，PyTorch 2.1，TensorRT
- **频率范围**：
  - CPU：115–1510 MHz（20档）
  - GPU：306–624.75 MHz（5档）

### **评估指标**
| 指标 | 描述 |
|------|------|
| **End-to-End Latency** | 单次推理总耗时 |
| **Energy Consumption** | 推理全过程能耗（mJ） |
| **Power Profile** | 功耗曲线稳定性 |
| **Energy Efficiency Gain** | 相较于默认 DVFS 的能耗降低百分比 |
| **Cost-Gain Ratio** | 每单位能效增益所付出的延迟代价（越低越好） |
| **Thermal Stability** | 温度波动与是否触发 thermal throttling |

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Default DVFS** | Linux `schedutil`（CPU） + NVIDIA `simple_ondemand`（GPU） |
| **nvpmodel (MAX-N)** | 锁定所有组件至最高频率 |
| **GearDVFS** | 基于 DRL 的模型级 DVFS，面向并发负载优化 |
| **Ascend-DVFS** | 基于遗传算法的算子级精细 DVFS |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | 结果 |
|------|------|
| **平均能效增益** | **78.17%**（相较 Default DVFS） |
| **Cost-Gain Ratio** | **14%**（优于 GearDVFS 的 48%，Ascend-DVFS 的 68%） |
| **DVFS 切换延迟减少** | 最高达 **8.5×**（如 ViT-B16） |
| **端到端延迟增加** | 相较 MAX-N 仅增加 **12.8%**，但能耗大幅下降 |

### **与基线方法对比**
| 方法 | 能效增益 | Cost-Gain Ratio | 说明 |
|------|----------|------------------|------|
| **Default DVFS** | 0% | — | 基准 |
| **nvpmodel (MAX-N)** | 46.86% | 16% | 高频运行，能耗高，易过热 |
| **GearDVFS** | 20.25% | 48% | 模型级调控，粒度不足 |
| **Ascend-DVFS** | 11.33% | 68% | 切换开销大，实际节能有限 |
| **SparseDVFS (Ours)** | **78.17%** | **14%** | **综合表现最优** |

> ✅ **结论**：SparseDVFS 在保持可接受延迟的同时，实现了远超现有方法的能效提升。

---

### **消融实验结果（Ablation Study）**
在 ViT-B16 上进行模块消融验证（见 Figure 15）：

| 配置 | 能效表现 | 分析 |
|------|----------|------|
| **GPU Only** | 差 | 存在 CPU-GPU 拮抗效应，CPU 下电导致唤醒延迟 |
| **GPU + CPU Lock** | 中等 | 固定 CPU 高频缓解饥饿，但未优化内存 |
| **Full FUSE（完整方案）** | **最优** | 统一协调三者频率，实现全局最优能效 |

> 🔍 **发现**：**FUSE 策略** 是消除拮抗效应、最大化节能的关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. **算子稀疏性是 DVFS 调控的核心信号**  
   - 高稀疏算子（如 ReLU、LayerNorm）为内存瓶颈（memory-bound），无需高频 GPU；
   - 低稀疏算子（如 Conv2d、Linear）为计算瓶颈（compute-bound），需高频加速。

2. **块级聚合是平衡粒度与开销的理想折中**  
   - 通过 **延迟摊销约束** 可将 ResNet-101 的 105 个算子聚合成 16 个超块，显著降低切换次数。

3. **统一协同控制至关重要**  
   - 独立调控 CPU/GPU 易引发 pipeline stall；
   - **FUSE + 前向预取** 可有效隐藏切换延迟，避免性能断崖。

4. **SparseDVFS 具备跨架构泛化能力**  
   - 在 CNN 和 Transformer 架构上均取得一致优异表现，证明其通用性。

---

### **方法的局限性**
1. **稀疏模式未区分结构化与非结构化**  
   - 当前模型将稀疏性视为标量比率，未考虑访问模式差异（如 unstructured sparsity 导致随机访存效率低下）。

2. **依赖离线建模与设备特定剖面**  
   - 模型需针对具体硬件（如 Jetson Orin Nano）进行离线训练，迁移到新平台需重新校准。

3. **未集成实时内存监控**  
   - 内存带宽利用率基于静态建模，未利用 DRAM 性能计数器进行动态反馈。

---

### **未来工作方向**
1. **精细化稀疏模式建模**  
   - 区分 structured vs. unstructured sparsity，动态调整 EMC 频率以应对随机访存开销。

2. **引入内存访问感知机制**  
   - 集成 real-time memory controller counters，实现更精准的带宽预测与调控。

3. **跨平台迁移学习支持**  
   - 利用 **transfer learning** 将一个设备上的 V/F 映射迁移到另一设备，减少重复剖面成本。

4. **扩展至边缘-云协同场景**  
   - 结合早期退出（early exiting）与跨层 DVFS，进一步提升端边协同推理能效。

---

## **总结**
✅ **SparseDVFS 成功弥合了模型级与算子级 DVFS 之间的鸿沟**，通过 **稀疏性感知 + 块级聚合 + 统一协同控制**，在真实边缘平台上实现了 **平均 78.17% 的能效增益** 与 **仅 14% 的 Cost-Gain Ratio**，为高效边缘 DNN 推理提供了可落地的系统级解决方案。

</details>

---

### 2. [CurvZO: Adaptive Curvature-Guided Sparse Zeroth-Order Optimization for Efficient LLM Fine-Tuning](https://arxiv.org/abs/2603.21725)

**Authors**: Shuo Wang, Ziyu Chen, Ming Tang  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.21725v1  

#### Abstract
Fine-tuning large language models (LLMs) with backpropagation achieves high performance but incurs substantial memory overhead, limiting scalability on resource-constrained hardware. Zeroth-order (ZO) optimization provides a memory-efficient alternative by relying solely on forward passes, yet it ty...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CurvZO: Adaptive Curvature-Guided Sparse Zeroth-Order Optimization for Efficient LLM Fine-Tuning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLM）在使用反向传播进行 fine-tuning 时虽然性能优异，但会带来巨大的内存开销，限制了其在资源受限硬件上的可扩展性。Zeroth-Order (ZO) 优化通过仅依赖前向传播来估计梯度，显著降低了内存消耗，但通常因梯度估计的高方差而导致收敛缓慢或不稳定。

稀疏 ZO 更新（sparse ZO）通过只扰动部分参数缓解了这一问题，但如何选择“有信息量”的参数进行扰动是一个挑战——因为在 ZO 设置中，每次查询只能获得标量反馈，缺乏明确的坐标级指导信号。

### 提出了什么新方法或新思路
本文提出 **CurvZO**（Adaptive Curvature-Guided Sparse Zeroth-Order Optimization），一种自适应曲率引导的稀疏零阶优化框架，其核心思想是：

- **在线追踪曲率信号**：从 ZO 的标量反馈中提取并跟踪每参数或每块（block-wise）的局部曲率信号，作为参数敏感性的代理。
- **基于曲率构建采样分布**：利用这些曲率信号构造参数级的采样概率分布，在每次更新时更频繁地扰动具有高曲率的参数，从而降低稀疏 ZO 梯度估计器的方差。
- **动态调整扰动预算（adaptive budget selection）**：根据曲率信号分布的变化（如有效支持大小和分布尖锐度）动态调整每轮扰动的参数数量，使更新既聚焦又保持足够的探索能力。

### 相比现有方法的优势
| 方法 | 局限性 | CurvZO 的优势 |
|------|--------|----------------|
| MeZO / DiZO | 全参数扰动导致高方差；无智能参数选择机制 | 引入曲率感知采样，显著降低方差 |
| Sparse-MeZO / SubZero | 预定义或随机稀疏模式，非自适应 | 自适应选择高曲率参数，提升效率 |
| SensZOQ | 依赖预计算的 Fisher 信息矩阵，增加额外计算开销 | 完全在线学习曲率信号，无需外部统计，保持 ZO 的简洁性和高效性 |

> ✅ **核心优势**：在不牺牲 ZO 内存效率的前提下，实现了更快、更稳定的收敛，并提升了最终性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验基于 **SuperGLUE** 基准中的多个任务，包括：
- 分类任务：SST-2, RTE, CB, BoolQ, WIC, WSC
- 生成任务：SQuAD, DROP

训练样本统一为 **1,000 条**，验证集 500 条，测试集 1,000 条，遵循标准少样本设置。

### 实验设置和评估指标
- **模型系列**：OPT（2.7B, 6.7B）、Llama2（7B, 13B）
- **评估指标**：
  - 主要指标：**Accuracy (%)**
  - 效率指标：**GPU 小时数（GPU hours）**、**收敛步数（iterations）**、**内存占用（GB）**
- **实现细节**：
  - 使用 block-wise 曲率评分以减少开销（每个 block 对应一个原生参数张量）
  - 扰动规模 $ \epsilon = 1e^{-3} $
  - 固定训练步数为 20,000 步
  - 结果取三次独立运行平均值

### 基线方法对比
| 基线方法 | 类型 | 特点 |
|---------|------|------|
| FT（Full Fine-tuning） | First-Order (FO) | 反向传播基准，性能上限 |
| LoRA | Parameter-Efficient FO | 低秩适配，节省显存 |
| MeZO | ZO baseline | 经典 ZO 方法，全参数扰动 |
| DiZO | ZO baseline | 分歧驱动的层级别 ZO 优化 |
| MeZO/DiZO + LoRA | Hybrid | 结合 LoRA 的 ZO 微调 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1–2 和 Figure 2）

#### 在 OPT-2.7B 上的表现（Table 1）
| 方法 | Average Accuracy |
|------|------------------|
| MeZO | 64.2% |
| DiZO | 61.4% |
| **CurvZO** | **66.8%** ↑2.6pp |

> CurvZO 在 8 项任务中有 7 项优于所有 ZO 基线，尤其在 WSC (+14.3%) 和 SQuAD (+12.5%) 上表现突出。

#### 在 OPT-6.7B 上的表现（Table 2）
| 方法 | Average Accuracy |
|------|------------------|
| MeZO | 75.8% |
| DiZO | 73.5% |
| **CurvZO** | **76.8%** ↑1.0pp |

> 即使在更大模型上仍保持领先。

#### 在 Llama2 上的表现（Figure 2）
- Llama2-7B：CurvZO 平均高出 MeZO 约 **2–3 个百分点**
- Llama2-13B：持续稳定增益，证明方法对架构变化鲁棒

### 与基线方法的对比结果
| 维度 | 对比结果 |
|------|----------|
| **准确率提升** | 最高达 **+4.4 points**（相比 ZO 基线） |
| **训练速度** | 收敛所需迭代次数减少 **2.0–2.4×**（见 Figure 3） |
| **GPU 时间成本** | 总 GPU 小时数最多减少 **59%**（BoolQ），平均节省约 **50%**（见 Figure 4） |
| **内存效率** | 与 MeZO 几乎持平（OPT-2.7B 同为 5.91GB；OPT-6.7B 仅略高至 13.95GB vs 13.94GB），远低于 FT（>80GB 超出单卡容量） |

> ⚡️ **关键发现**：CurvZO 在几乎相同的内存开销下，实现了更高的精度和更快的收敛。

### 消融实验结果（Table 4）

| 方法 | Average Acc (%) | 说明 |
|------|------------------|------|
| US+AB（Uniform + Adaptive Budget） | 62.5 | 仅用自适应预算，性能有限 |
| CurvZO(B=0.4d)（固定预算 + 曲率采样） | 64.5 | 固定预算下已优于均匀采样 |
| **CurvZO（完整版）** | **66.8** | 同时具备曲率采样 + 自适应预算，效果最佳 |

> 🔍 **消融结论**：
> - 曲率引导采样（curvature-guided sampling）是性能提升的关键；
> - 自适应预算进一步增强了灵活性和稳定性；
> - 两者结合产生协同效应，带来最显著收益。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **局部曲率可被有效追踪**：即使在 ZO 设置下仅有标量反馈，也能通过设计合理的曲率评分（curvature score）在线捕捉参数的局部敏感性。
2. **高曲率参数更值得扰动**：优先扰动高曲率方向能显著降低梯度估计方差，提升优化效率。
3. **扰动预算应动态调整**：早期需要较大预算维持探索，后期可收缩以聚焦关键参数，CurvZO 的自适应机制自动完成该权衡。
4. **性能与效率兼得**：CurvZO 在保持 ZO 级别内存效率的同时，实现了接近 FO 方法的性能，并大幅超越现有 ZO 基线。

### 方法的局限性
- **仅反映局部敏感性**：当前曲率评分基于对角 Fisher 近似，未建模参数间的耦合关系（interaction-aware 二阶信息），可能在某些复杂场景下采样不够最优。
- **块粒度影响细粒度控制**：采用 block-wise 实现是为了降低开销，但牺牲了一定程度的参数级精细调控能力。
- **对超参数有一定依赖**：平滑系数 $ \beta $ 和预算权重 $ \alpha $ 需合理设置，尽管作者表明默认值在多数任务上表现良好。

### 未来工作方向
- 探索建模参数间交互的曲率估计方式（如低秩 Hessian 近似）
- 将 CurvZO 扩展到其他内存密集型任务，如 RLHF 或 long-sequence modeling
- 与其他 PEFT 方法（如 AdaLoRA, LoHa）进一步融合，构建更高效的联合优化框架
- 在边缘设备或分布式系统中部署 CurvZO，验证其实用性与可扩展性

---

> ✅ **总体评价**：  
> CurvZO 是一项将经典优化理论（曲率感知、方差控制）与现代 LLM 高效微调需求相结合的杰出工作。它不仅提出了一个“即插即用”（plug-and-play）的改进方案，而且在性能、效率和通用性之间取得了良好平衡，为 ZO 优化的实际应用开辟了新路径。

</details>

---

### 3. [DRTriton: Large-Scale Synthetic Data Reinforcement Learning for Triton Kernel Generation](https://arxiv.org/abs/2603.21465)

**Authors**: Siqi Guo, Ming Lin, Tianbao Yang  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.21465v1  

#### Abstract
Developing efficient CUDA kernels is a fundamental yet challenging task in the generative AI industry. Recent researches leverage Large Language Models (LLMs) to automatically convert PyTorch reference implementations to CUDA kernels, significantly reducing the engineering efforts. State-of-the-art ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DRTriton: Large-Scale Synthetic Data Reinforcement Learning for Triton Kernel Generation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
开发高效的 CUDA kernels 是生成式 AI 工业中的关键挑战，但对人类专家而言也极为复杂且耗时（如 FlashAttention 花费数年）。尽管近期研究尝试利用 **Large Language Models (LLMs)** 自动将 PyTorch 代码转换为 CUDA 或 Triton kernels，但当前最先进的 LLMs（如 GPT-5.2 和 Claude-Sonnet-4.5）在该任务上表现不佳。

现有方法存在以下瓶颈：
- **训练数据规模有限**：依赖真实代码仓库收集的数据（如 11k–14k 样本），不足以学习复杂的 kernel 优化。
- **样本难度不可控**：缺乏课程学习机制，在稀疏奖励下模型容易陷入零梯度信号。
- **质量与复杂性不均衡**：真实数据中简单操作占主导，难以覆盖高阶融合模式。

---

### 提出的新方法与创新思路
作者提出 **DRTriton** ——一个完全基于合成数据的大规模强化学习框架，用于训练 LLM 将 PyTorch 代码高效编译为 Triton kernels（最终编译为 CUDA kernels）。

其三大核心组件构成主要贡献：

#### ✅ (i) CSP-DAG：保证全覆盖与无偏采样的 PyTorch 程序生成算法
- 将 PyTorch 程序生成建模为 **有向无环图上的约束满足问题 (Constraint Satisfaction Problem on DAGs)**。
- 利用 CP-SAT 求解器确保所有生成程序语法正确、张量形状兼容，并支持任意数量的操作符组合。
- 可控制程序复杂度（以 operator 数量定义“难度等级”），实现均匀采样，避免偏差。

#### ✅ (ii) 基于解耦奖励的课程强化学习 (Curriculum RL with Decoupled Rewards)
- 引入 **DRPO (Decoupled Reward Policy Optimization)** 框架，将奖励分为两个独立部分：
  - **Correctness Reward**：验证 kernel 是否功能等价。
  - **Speed Reward**：衡量生成 kernel 相对于原始 PyTorch 的执行速度提升。
- 在早期阶段通过解耦稳定训练过程，缓解稀疏奖励问题。
- 采用三阶段课程学习策略：
  - Level 1：单算子 → Level 2：双算子序列 → Level 5：五算子复合结构

#### ✅ (iii) 推理时搜索 (Test-time Search) 提升复杂 kernel 性能
- 对于无法单 kernel 实现的长程序，提出一种系统化的融合策略搜索方法：
  1. **片段提取**：从原程序中提取长度为 1~5 的连续子序列作为候选 fragment。
  2. **kernel 生成与验证**：调用 DRTriton 为每个 fragment 生成 Triton kernel 并验证正确性。
  3. **重构与评测**：替换对应 fragment 为 Triton kernel，保留其余部分不变，选择执行最快的一种组合方案。
- 显著提升多算子场景下的推理效率。

---

### 相比现有方法的优势
| 维度 | 现有方法 | DRTriton |
|------|--------|----------|
| 数据来源 | 真实代码库（有限、分布偏移） | 完全合成数据（可控、可扩展） |
| 学习方式 | SFT / 单一 RL 奖励 | Curriculum RL + 解耦奖励优化 |
| 泛化能力 | 局限于已见模式 | 能泛化到 KernelBench 中的人类专家级架构 |
| 复杂程序处理 | 单一 kernel 转换失败率高 | 支持 test-time composition 搜索 |
| 性能目标 | 正确性优先 | 同时优化正确性和运行速度 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### (1) 合成基准数据集（Synthetic Benchmark）
- 包含 406 个未出现在训练中的 hold-out PyTorch 程序。
- 按难度分级：
  - Level 1：106 个单算子程序（覆盖 53 种常见 operator）
  - Level 2：100 个双算子程序
  - Level 5：100 个五算子程序
  - Level 20：100 个二十算子程序（极难）
- 所有程序由 CSP-DAG 自动生成，确保形状合法性和多样性。

#### (2) KernelBench 实际基准
- 来自真实神经网络应用的 250 个 PyTorch 程序，分为三级：
  - Level 1：基础算子（卷积、矩阵乘、激活函数）
  - Level 2：融合模式（conv+bias+ReLU）
  - Level 3：完整模型（MobileNet、VGG、MiniGPT）
- 存在显著的分布偏移（面向对象 vs 函数式输入格式）

> 💡 为此设计自动重写工具，将 `nn.Module` 类转为函数式 `fused_operator` 形式，适配训练数据分布。

---

### 实验设置与评估指标

#### 模型架构
- 基础模型：`Qwen-2.5-Coder-7B-Instruct`
- 训练流程：
  1. **SFT 阶段**：在 2,026 个单算子 PyTorch-Triton 对上微调（来自 DeepSeek-R1 / GPT-5.2 生成并验证）
  2. **RL 阶段**：分三个 curriculum 阶段进行 DRPO 微调，共 100,000 个合成程序

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Acc (Accuracy)** | 成功通过验证的 kernel 比例 |
| **Faster1** | 正确 kernel 中比原始 PyTorch 快 >1x 的比例 |
| **Avg. Speedup** | 所有正确 kernel 的几何平均加速比 |

#### 基线方法对比
- 商业 LLMs：`GPT-5.2`, `Claude Sonnet 4.5`
- 开源 LLMs：`DeepSeek-R1`, `Qwen-3-Coder-480B`
- 专用模型：`AutoTriton`（基于 RL 的 Triton kernel 生成器）

---

## 3. 主要实验结果和性能指标

### 在 Synthetic Benchmark 上的结果（Table 1）

| Model | Level 1 Acc | Level 2 Acc | Level 5 Acc | Level 20 Acc | Avg. Speedup |
|-------|-------------|-------------|-------------|---------------|--------------|
| GPT-5.2 | 53.8% | 43.0% | 5.0% | 0% | 1.34x |
| Claude-Sonnet-4.5 | 67.9% | 49.0% | 7.0% | 0% | 0.68x |
| **DRTriton** | **86.8%** | **75.0%** | **15.0%** | **0%** | **1.20x** |
| + test-time search | 86.8% | **96.0%** | **99.0%** | **99.0%** | **1.57x** |

> 🔍 **关键观察**：
> - DRTriton 在 Level 1 和 Level 2 显著优于所有 baseline。
> - 在 Level 5，DRTriton 是唯一取得非零精度的方法（15% vs 其他 <9%）。
> - 加入 test-time search 后，对复杂程序（Level 5/20）的准确率飙升至接近 100%，说明 compositional kernel fusion 极具潜力。

---

### 在 KernelBench 上的结果（Table 2）

| Model | Level 2 Acc | Level 2 Faster1 (vs TE) | Level 3 Acc | Level 3 Faster1 (vs TE) |
|-------|------------|------------------------|------------|------------------------|
| GPT-5.2 | 32% | 23% | 30% | 18% |
| Claude-Sonnet-4.5 | 36% | 19% | 36% | 12% |
| **DRTriton (w/ test-time search)** | **96%** | **92%** | **76%** | **54%** |

> 📈 **亮点**：
> - 在 Level 2 上，**92% 的 kernel 比 Torch Eager 更快**，远超 GPT-5.2 的 23%。
> - 在 Level 3（完整模型）仍保持 76% 的高准确率，其中 **54% 快于 Torch Eager，34% 快于 torch.compile**，表明其具备超越主流编译器的能力。

---

### 消融实验结果

#### (1) RL 算法对比：DRPO vs GRPO（Figure 5）
- 在相同 SFT checkpoint 和训练数据下，**DRPO 在 Acc 和 Faster1 上全面领先 GRPO**。
- 表明 **decoupled reward 设计有效缓解了早期稀疏奖励问题**，提升了学习稳定性。

#### (2) 速度奖励函数设计（Appendix D.2）
| Reward Function | Level 1 Acc | Faster1 |
|----------------|-------------|--------|
| Power (α=0.25) | 38.1% | 14.4% |
| Power (α=1.0) | 32.0% | 3.1% |
| **Logarithmic** | **42.3%** | **18.6%** |

> ✅ 最佳选择是 `rs(o) = log(t_torch / t_triton)`，即对数形式的速度增益奖励。

---

## 4. 关键结论和发现

### 主要发现
1. **仅用合成数据即可训练出超越商业 LLM 的 kernel 生成器**  
   DRTriton 完全基于 CSP-DAG 生成的合成数据训练，却能在真实世界 benchmark（KernelBench）上大幅超越 GPT-5.2 和 Claude-Sonnet-4.5，证明了合成数据 + 强化学习路径的巨大潜力。

2. **课程学习 + 解耦奖励显著提升训练效率与性能**  
   DRPO 框架结合 curriculum learning，使模型能逐步掌握从简单到复杂的 kernel 生成技能，尤其在稀疏奖励初期提供了更稳定的梯度信号。

3. **推理时搜索极大增强复杂程序的优化能力**  
   test-time search 不仅提高了成功率，还实现了高达 1.57x 的平均加速，揭示了“模块化 kernel composition”是应对复杂计算图的有效范式。

4. **DRTriton 能生成连人类专家都难以写出的高性能 kernel**  
   在 KernelBench Level 3 上的表现显示，它不仅能复现已有优化，还能发现新的高效实现方式，展现出一定的“涌现能力”。

---

### 方法的局限性
- **依赖高质量的验证机制**：整个 RL 流程建立在可靠的 correctness 和 speed profiling 上，若验证出错会导致错误反馈传播。
- **当前 test-time search 计算开销较大**：需枚举多个 fragment 并分别生成 kernel，可能影响端到端延迟。
- **局限于 Triton 支持的算子集合**：目前覆盖约 53 个常用 operator，尚未扩展至全部 PyTorch 生态。

---

### 未来工作方向
- 扩展 CSP-DAG 以支持动态控制流（如条件分支、循环展开）。
- 探索更高效的 test-time search 策略（如基于 learned policy 的 beam search）。
- 将 DRTriton 思路迁移到其他 DSL 编译器（如 Halide、CUDA C++）。
- 结合 multi-agent 协同框架，进一步分解 planning、generation、verification 流程。

--- 

> ✅ **总结一句话**：  
> **DRTriton 展示了一条全新的路径——无需依赖稀缺的真实专家数据，也能通过大规模合成数据 + 强化学习 + 推理时搜索，训练出可媲美甚至超越人类专家水平的 GPU kernel 生成系统。**

</details>

---

### 4. [Communication-Avoiding SpGEMM via Trident Partitioning on Hierarchical GPU Interconnects](https://arxiv.org/abs/2603.21444)

**Authors**: Julian Bellavita, Lorenzo Pichetti, Thomas Pasquali, Flavio Vella, Giulia Guidi  
**Category**: cs.DC  
**Published**: 2026-03-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.21444v1  

#### Abstract
The multiplication of two sparse matrices, known as SpGEMM, is a key kernel in scientific computing and large-scale data analytics, underpinning graph algorithms, machine learning, simulations, and computational biology, where sparsity is often highly unstructured. The unstructured sparsity makes ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Communication-Avoiding SpGEMM via Trident Partitioning on Hierarchical GPU Interconnects*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大规模分布式多GPU系统具有**层次化互连架构**（hierarchical interconnect），即：
- **节点内通信**（intranode）通过高带宽、低延迟的互联技术（如NVLink）实现；
- **节点间通信**（internode）则依赖较慢的网络（如Slingshot或InfiniBand）。

然而，现有的 **SpGEMM**（Sparse General Matrix Multiplication）算法大多假设通信是均匀的，忽略了这种带宽差异，导致：
- 过度的节点间通信开销；
- 无法充分利用高速的节点内互联；
- 可扩展性和性能受限。

该论文针对这一问题，提出了一种**层次感知**（hierarchy-aware）的分布式 SpGEMM 算法。

---

### 🚀 提出的新方法：TRIDENT
TRIDENT 是一种新的分布式 SpGEMM 算法，其核心是 **Trident Partitioning**（三叉分区）策略，结合了以下关键技术：

#### （1）Hybrid 2D-1D Trident Partitioning
- 在**跨节点层面**采用 **2D 分区**：将矩阵划分为 $\sqrt{P/\lambda} \times \sqrt{P/\lambda}$ 的粗粒度 tile，每个 tile 分配给一个完整的计算节点；
- 在**节点内部**采用 **1D 分区**：将每个 2D tile 进一步细分为 $\lambda$ 个 1D slice，分发到节点内的各个 GPU 上（$\lambda$ 为每节点 GPU 数量，例如 4）；
- 形成三维逻辑网格：$\sqrt{P/\lambda} \times \sqrt{P/\lambda} \times \lambda$。

> ⚠️ 注意：这不是传统意义上的 3D SpGEMM，第三维不用于复制子矩阵，而是为了组织局部通信。

#### （2）两阶段通信优化
1. **节点间通信**（over GI）仅进行一次，传输部分 tile；
2. 利用高速 **节点内互联**（over LI）完成本地聚合（Allgatherv），避免重复跨节点传输。

#### （3）异步执行模型（Asynchronous C-stationary）
- 每个输出 tile 独立推进，无需全局同步；
- 使用 **MPI 3.0 one-sided communication**（如 `MPI_Put`）实现无阻塞请求；
- 支持通信、计算与本地聚合的重叠（overlap），提升并行效率。

---

### 🔍 相比现有方法的优势
| 方面 | TRIDENT | 传统方法（如 Sparse SUMMA） |
|------|---------|-----------------------------|
| **通信模式** | 区分 GI 和 LI，优先使用 LI | 统一对待所有通信 |
| **通信量** | 显著减少 internode 通信体积 | 高频次、高体积的广播操作 |
| **可扩展性** | 更好地适应大规模 GPU 配置 | 同步瓶颈明显 |
| **适用场景** | 特别适合 unstructured sparse matrices | 对 structured 矩阵更优 |

---

## 2. 核心实验方法和设置

### 📊 数据集
使用来自真实世界的大型稀疏矩阵，主要来源于：
- **SuiteSparse** [19]
- **HipMCL** 生物信息学项目 [3]

| 矩阵名 | 类型 | 尺寸范围 | 特征 |
|--------|------|----------|-------|
| `HV15R` | 结构化（带状） | ~2M 行列 | 用于验证预处理效果 |
| `mouse_gene`, `archaea`, `eukarya` | 基因组相似性图 | 1.6M–3.2M | 高度非结构化 |
| `isolates_subgraph4/5`, `cage15`, `reddit` | 图神经网络 / 社交网络 | 最大超 4M | 不同密度与不平衡性 |

> 所有矩阵均为 **unstructured sparsity**，即非零元分布无规律。

---

### ⚙️ 实验设置
- **硬件平台**：NERSC 的 **Perlmutter 超级计算机**
  - 每节点 4 个 NVIDIA A100 GPU
  - 节点内：NVLink 3.0（~900 GB/s）
  - 节点间：Slingshot-11 Dragonfly（~400 Gb/s）
- **软件栈**：
  - MPI: Cray MPICH 8.30（支持 GPU-aware）
  - Intranode: NCCL 2.26
  - Local SpGEMM: KokkosKernels + cuSPARSE
- **规模测试**：从 4 到 256（部分至 400）个 GPU
- **任务类型**：
  - 强缩放（strong scaling）下的矩阵平方 $C = A \times A$
  - 与限制算子相乘 $C = A \times R$（Algebraic Multigrid 场景）

---

### 📈 评估指标
- 总运行时间（runtime）
- 各阶段耗时分解（communication, local SpGEMM, accumulation）
- **节点间通信量**（data volume over GI）
- 加速比（speedup）vs. 基线
- Markov Clustering 应用端到端性能

---

### 🆚 基线方法对比
| 基线 | 描述 | 是否层次感知 |
|------|------|---------------|
| **Trilinos (TPETRA)** | 1D row-wise, sparsity-aware, 使用非阻塞 MPI_Isend/Irecv | ❌ |
| **CombBLAS Sparse SUMMA** | 经典 2D BSP 算法，基于 MPI_Bcast | ❌ |
| **Improved Sparse SUMMA**（本文实现） | 改进版 2D 算法，设备端存储与合并中间结果，作为主比较对象 | ❌ |

> 注：CombBLAS 因性能较差未作为主要对比；TRIDENT 与其比较以突显改进空间。

---

## 3. 主要实验结果和性能指标

### 📉 性能加速比（Speedup）

| 对比项 | 最大加速比 | 几何平均加速比（256 GPUs） |
|--------|------------|----------------------------|
| vs. **Trilinos** | **5.95×**（mouse_gene） | **2.96×** |
| vs. **Improved Sparse SUMMA** | **2.38×** | **1.54×** |

> 在最大规模（256 GPUs）下，TRIDENT 显著优于所有基线，且优势随规模增大而增强。

---

### 📦 通信体积降低
- **节点间通信量减少最多达 2×**
- 如 Figure 10 所示，在 `isolates_subgraph4` 上，TRIDENT 每进程发送的数据仅为 Improved Sparse SUMMA 的约一半；
- 即使在某些情况下单个进程通信略增（due to tile shape variation），总体通信时间和带宽占用仍更低。

---

### ⏱️ 运行时分解（Runtime Breakdown）
以 `isolates_subgraph4` 和 `mouse_gene` 为例（Figure 9）：
- **TRIDENT 的 internode communication 时间增长缓慢**，且占比小；
- **Intranode communication 开销极低**（得益于 NCCL 和高带宽 LI）；
- Improved Sparse SUMMA 的 broadcast 成本随 GPU 数增加显著上升，成为瓶颈。

---

### 🔁 Markov Clustering（MCL）应用加速
- 实现了基于 TRIDENT 的分布式 MCL 扩展步骤；
- 在 `eukarya` 和 `isolates_subgraph4` 上运行 10 轮迭代；
- **相比 Improved Sparse SUMMA，最高获得近 2× 加速**；
- 尤其在早期稠密阶段（nonzero 多），TRIDENT 优势更明显。

---

### 🔄 消融分析（隐含）
虽然没有明确命名“ablation study”，但通过以下对比体现了设计有效性：
- **随机置换结构化矩阵 HV15R** → 从 Trilinos 更快转变为 TRIDENT 更快，说明 Trident 对 unstructured 矩阵更具优势；
- **不同通信机制对比** → 使用 one-sided RDMA 和 NCCL Allgatherv 显著优于传统 MPI 同步通信；
- **异步行为容忍负载不均** → 在 imbalance 较高的矩阵（如 `mouse_gene`: 4.65）上仍保持良好扩展性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **层次化互连必须被显式建模**：忽略节点内/外通信差异会导致严重性能损失；
2. **Trident Partitioning 有效解耦通信层级**：
   - 用 2D 跨节点分区控制全局通信频率；
   - 用 1D 节点内分区激活高速本地聚合；
3. **异步 + C-stationary 设计提升并发性**：
   - 节点独立推进，缓解同步等待；
   - 请求驱动通信，实现细粒度流水；
4. **TRIDENT 是首个真正意义上 hierarchy-aware 的分布式 SpGEMM 算法**。

---

### ⚠️ 局限性
- **依赖每节点固定数量 GPU**（$\lambda$）：当前设计假设 $\lambda$ 是常数（如 4 或 8），难以动态适配异构配置；
- **对极度稀疏或极端不平衡矩阵可能收益下降**；
- **未探索 3D 或 sparsity-aware 的层次化扩展版本**；
- **目前仅在 NVIDIA 平台验证**，AMD ROCm 支持需进一步测试（尽管作者提到可用 RCCL 替代 NCCL）。

---

### 🔮 未来工作方向
1. 扩展至其他稀疏原语（sparse primitives）：
   - 如 3D SpGEMM、SpMV、SpTSV 等；
2. 探索 **hierarchical sparsity-aware 1D SpGEMM**；
3. 动态调整分区策略以应对运行时负载变化；
4. 支持更多硬件平台（如 Grace Hopper、AMD Instinct）；
5. 集成到 CombBLAS、PETSc 等主流库中。

---

> 💡 **一句话总结**：  
> TRIDENT 通过创新的 **Trident Partitioning** 架构，首次将 **GPU 层次化互连特性**深度融入 SpGEMM 算法设计，在真实超算平台上实现了高达 **2.38×** 的性能提升，并显著降低了节点间通信开销，为下一代大规模稀疏计算提供了高效基础构件。

</details>

---

### 5. [ConsRoute:Consistency-Aware Adaptive Query Routing for Cloud-Edge-Device Large Language Models](https://arxiv.org/abs/2603.21237)

**Authors**: Haoyu Qiao, Hao Zhang, Shanwen Mao, Siyao Cheng, Jie Liu  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.21237v1  

#### Abstract
Large language models (LLMs) deliver impressive capabilities but incur substantial inference latency and cost, which hinders their deployment in latency-sensitive and resource-constrained scenarios. Cloud-edge-device collaborative inference has emerged as a promising paradigm by dynamically routing ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ConsRoute: Consistency-Aware Adaptive Query Routing for Cloud-Edge-Device Large Language Models**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）在移动和边缘计算等资源受限场景中部署时面临显著挑战：  
- **高延迟**：云上大模型推理耗时长，通信开销大。  
- **高成本**：频繁调用云端CLM（Cloud LLM）导致计算和能源消耗过高。  
- **质量与效率的权衡**：小模型（如DLM）响应快但质量差，大模型质量高但代价高昂。

现有路由方法存在以下缺陷：
- 依赖**标量质量分差**（如reward score差异），丢失细粒度语义一致性信息；
- 使用额外编码器（如BERT）提取查询表示，增加设备端开销；
- 采用**全局静态阈值**，无法适应异构查询分布和动态网络环境。

---

### **提出的新方法与创新思路**
作者提出 **ConsRoute** —— 一种轻量级、语义感知、自适应的三层（云-边-端）LLM协同推理路由框架，其核心创新如下：

#### ✅ **1. 基于语义一致性的软监督信号构建**
- 不再使用“质量差距”作为训练目标，而是利用预训练的 **reranker 模型**（如 Qwen3-Reranker-4B）直接衡量不同层级模型输出之间的**语义相似度**（Semantic Similarity）。
- 构建更精细的软标签 $ S_{\text{cloud}}(x) $ 和 $ S_{\text{edge}}(x) $，反映DLM与CLM/ELM响应的一致性程度。
- 引入**数据增强机制**：结合规则判断（有参考答案时）或高级LLM裁判（无参考答案时）进一步提升标签可靠性。

#### ✅ **2. 轻量级语义表示提取（复用DLM隐藏状态）**
- 利用设备端DLM在prefill阶段已生成的隐藏层状态，避免引入额外encoder。
- 在输入后附加一个固定指令提示：“The consistency between the small language model and large language model responses for the above query is:”，引导DLM生成对齐一致性任务的语义表示。
- 提取 `[EOS]` token 的最后一层隐藏状态 $ h_r $ 作为紧凑查询表征，供后续MLP预测器使用。
- 若最终选择DLM本地执行，该prefill过程可被复用于解码，实现**零冗余计算**。

#### ✅ **3. 基于聚类与贝叶斯优化的动态阈值机制**
- 将查询按其语义表示进行 **K-means聚类**，识别出具有不同复杂度和风险偏好的查询类别。
- 对每个簇 $ C_k $ 学习独立的路由阈值 $ (\tau_1^{(k)}, \tau_2^{(k)}) $，通过 **Bayesian Optimization** 最大化效用函数：
  $$
  U_k = \lambda_1 \cdot \text{Acc} - \lambda_2 \cdot \text{Latency} - \lambda_3 \cdot \text{Cost}
  $$
- 支持**在线增量更新**：随着新反馈到来，持续优化各簇阈值以应对流量漂移和网络变化。

---

### **相比现有方法的优势**
| 维度 | ConsRoute优势 |
|------|----------------|
| **准确性** | 使用语义一致性而非标量分差，捕捉更丰富的响应关系，减少误判。 |
| **效率** | 复用DLM内部状态，无需额外encoder，显著降低设备端延迟与内存占用。 |
| **灵活性** | 动态、按簇调整阈值，适应多样化查询类型和运行时条件（如网络状况）。 |
| **实用性** | 支持在线学习，适用于真实部署中的非平稳数据流。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 主要基准：**RouterBench**（36.5K条中英文混合查询）
- 包含多个子任务：
  - **MMLU**：通用知识问答
  - **GSM8K**：数学推理
  - **HumanEval**：代码生成
  - **MT-Bench**：多轮对话（由GPT-4o评分）

---

### **实验设置**
#### **模型配置**
| 层级 | 模型 | 参数量 | 硬件平台 |
|------|------|--------|----------|
| **Device (DLM)** | Qwen3-1.7B | ~1.7B | i5-12500H + RTX 3050 |
| **Edge (ELM)** | Qwen3-14B | ~14B | 单张RTX A6000 |
| **Cloud (CLM)** | Qwen3-32B | ~32B | 双RTX A6000 |

> 注：也测试了跨家族部署（LLaMA-3.2-3B / Qwen3-14B / DeepSeek-V3），验证泛化能力。

#### **网络模拟**
- “Good” 条件：低延迟、高带宽（如edge: 40ms RTT）
- “Bad” 条件：高延迟、低带宽、丢包率高（如cloud: 250ms RTT, 3% loss）
- “Bad→Good”：模拟网络恢复场景

---

### **评估指标**
- **Accuracy**：相对于CLM-only策略的准确率百分比（归一化）
- **Latency (%)**：端到端延迟占CLM-only策略的比例
- **Cost (%)**：推理成本（激活参数×token数）占比
- **Utility Function**：综合考虑准确率、延迟、成本的加权目标

---

### **基线方法对比**
| 方法 | 类型 | 是否预测路由 | 是否重用DLM状态 | 是否支持自适应阈值 |
|------|------|---------------|------------------|--------------------|
| **CLM-only** | 上限基线 | – | – | – |
| **DLM-only** | 下限基线 | – | – | – |
| **Edge-only** | 中间基线 | – | – | – |
| **FrugalGPT** | 级联策略 | 否 | 否 | 否 |
| **AutoMix** | POMDP决策 | 是 | 是 | 否 |
| **RouteLLM (BERT)** | 编码器+分类器 | 是 | 否 | 否 |
| **RouteLLM (SW ranking)** | 检索最近邻 | 是 | 否 | 否 |
| **MixLLM** | 多目标优化 | 是 | 否 | 部分支持 |
| **ConsRoute (Ours)** | ✅ 一致性感知+动态阈值 | ✅ | ✅ | ✅ |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | ConsRoute 表现 |
|------|----------------|
| **平均准确率** | ≥95% of CLM-only performance |
| **端到端延迟** | 仅需 **60–65%** 的CLM-only延迟 |
| **推理成本** | 降至 **约70%** 的CLM-only成本 |
| **延迟/成本节省** | **接近40%** 的系统级优化 |

> 在所有任务（GSM8K, MMLU, HumanEval, MT-Bench）上均优于基线。

---

### **与基线方法的对比结果**
- 如图6所示，在相同准确率下，ConsRoute所需延迟和成本显著低于其他方法：
  - 相比 **MixLLM** 和 **RouteLLM-BERT**，达到同等精度需少 **10–20%** 的延迟。
  - 在“Bad”网络条件下仍能维持较高准确率，而其他方法性能下降明显。
- 在跨家族部署（LLaMA/Qwen/DeepSeek）中表现稳健，证明其架构无关性。

---

### **消融实验结果**
#### ✅ **Label Augmentation 效果**
- 加入参考答案校验或LLM裁判后，一致性标签更可靠。
- 相比仅用reranker得分，准确率提升约 **2–3个百分点**。

#### ✅ **Dynamic Thresholding 效果**
| 设置 | Latency (%) @ 95% Acc |
|------|------------------------|
| w/o Aug + Fixed Threshold | ~75% |
| w/ Aug + Fixed Threshold | ~70% |
| **w/ Aug + Adaptive Threshold (Ours)** | **~62%** |

> 动态阈值带来最大增益，说明**个性化路由策略至关重要**。

#### ✅ **Prompting Strategy 对比（Table III）**
| 方法 | 准确率 | 路由延迟 (ms) |
|------|--------|----------------|
| 显式选择（无CoT） | 85.79% | 36.2 |
| 显式选择（有CoT） | 86.44% | **1868.2** ❌ |
| 隐式选择（无prompt） | 82.34% | 20.4 |
| **隐式选择（有prompt）✅** | **87.92%** | **20.9** |

> **Prompt引导的隐式表示**在精度和延迟之间取得最佳平衡。

---

## **4. 关键结论和发现**

### **主要发现**
1. **语义一致性是比质量分差更优的监督信号**  
   - Reranker-based similarity 与人工标注一致性相关性更强（见Fig. 4），能有效揭示细微语义偏差。
   
2. **复用DLM隐藏状态可实现极低开销路由**  
   - 无需额外encoder，路由延迟仅增加 **~20ms**，远低于BERT-based方案（>100ms）。

3. **动态、按簇阈值显著提升系统适应性**  
   - 在线贝叶斯优化使系统能自动响应网络波动（如从“Bad”切换至“Good”），实现智能升降级。

4. **ConsRoute实现了高质量与高效能的统一**  
   - 在保持 **≥95% 云端性能**的同时，将延迟和成本降低近 **40%**，为移动端LLM应用提供实用解决方案。

---

### **方法的局限性**
- **依赖高质量reranker模型**：若reranker本身存在偏差，可能影响训练标签质量。
- **聚类数量需手动设定**（尽管使用elbow method自动确定K），对极端稀疏查询敏感。
- 当前未整合用户反馈闭环，未来可结合显式满意度信号进一步优化。

---

### **未来工作方向**
1. 扩展至更多应用场景（如语音助手、AR交互）；
2. 探索基于强化学习的联合优化路由与调度策略；
3. 在更大规模的真实世界移动部署中验证长期稳定性；
4. 引入多模态输入支持（文本+图像）下的跨模态一致性路由。

--- 

> ✅ **总结一句话**：  
> **ConsRoute通过“语义一致性监督 + 隐藏状态复用 + 动态阈值优化”，在不牺牲质量的前提下，实现了云-边-端LLM系统的高效自适应路由，是迈向轻量化、智能化边缘LLM服务的重要一步。**

</details>

---

### 6. [CALVO: Improve Serving Efficiency for LLM Inferences with Intense Network Demands](https://arxiv.org/abs/2603.21257)

**Authors**: Weiye Wang (Shanghai Jiao Tong University), Chen Chen (Shanghai Jiao Tong University), Junxue Zhang (University of Science and Technology of China), Zhusheng Wang (Huawei), Hui Yuan (Huawei), Zixuan Guan (Huawei), Xiaolong Zheng (Huawei), Qizhen Weng (Institute of Artificial Intelligence), Yin Chen (Institute of Artificial Intelligence), Minyi Guo (Shanghai Jiao Tong University)  
**Category**: cs.DC  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.21257v1  

#### Abstract
Distributed prefix caching has become a core technique for efficient LLM serving. However, for long-context requests with high cache hit ratios, retrieving reusable KVCache blocks from remote servers has emerged as a new performance bottleneck. Such network-intensive LLM inference is expected to bec...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CALVO: Improve Serving Efficiency for LLM Inferences with Intense Network Demands

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 LLM 推理引擎（如 vLLM）在处理**网络密集型（network-intensive）LLM 推理任务**时效率低下，主要原因如下：

- **KVCache 加载成为瓶颈**：随着上下文长度增长（如 LooGLE、Code 等长上下文任务），分布式 prefix caching 虽然提高了 cache hit ratio，但跨节点加载 KVCache 块（从远程 L3 → 本地 L2 → GPU L1）引入了显著的通信开销。
- **现有系统设计是计算中心化（compute-centric）**：KVCache 加载被视为次要阶段，由主控线程统一调度，导致：
  - 各阶段资源无法并行利用（network、PCIe、GPU idle）
  - 调度决策忽略 KVCache 加载延迟，造成次优服务顺序。

> 实测显示，在某些场景下，KVCache 加载时间可占 TTFT 的 **90%以上**。

---

### 🚀 提出的新方法：CALVO

CALVO 是一个专为高效服务 **network-intensive LLM inferences** 设计的新型推理引擎，其核心思想是：

> **将跨节点 KVCache 加载视为与计算同等重要的“一级阶段”（first-class citizen）**

#### 主要技术创新：

1. **Stage-Decoupled Pipeline 架构**
   - 将 KVCache 加载流程解耦为独立可控的阶段：
     - `L3 → L2`（网络传输）
     - `L2 → L1`（PCIe 传输）
     - `Compute`（GPU prefill）
   - 每个阶段配备独立的 **dispatcher-executor 对**，实现异步并发执行。
   - 引入**主动空间分配机制**（proactive space allocation）：
     - 下层 dispatcher 在发起数据传输前，提前触发上层内存空间预留（如 L3 dispatcher 提前通知 L2 dispatcher 分配 GPU memory），避免阻塞。

2. **基于二元成本模型的智能调度**
   - 提出更准确的服务成本建模方式：
     $$
     \text{Service Cost} = T_{\text{load}} + T_{\text{comp}}
     $$
     其中：
     - $T_{\text{load}}$：与需加载的 context token 数量成线性关系（通过 profiling 拟合）
     - $T_{\text{comp}}$：与 query token 数量相关
   - 基于此模型优化调度策略：
     - **最小化平均 TTFT** → 使用 **SJF（Shortest Job First）**
     - **最大化 SLO attainment** → 使用 **LSTF（Least Slack Time First）**

---

### 🔍 相比现有方法的优势

| 维度 | 现有方法（如 vLLM-LMCache） | CALVO |
|------|-------------------------------|--------|
| 控制架构 | Centralized, compute-centric | Decentralized, stage-autonomous |
| 资源利用率 | 低（各阶段串行等待） | 高（pipeline 并发） |
| 调度依据 | FIFO 或仅考虑 compute time | 显式建模 KVCache loading delay |
| 适用场景 | Computation-intensive workloads | Network-intensive workloads |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

| 数据集 | 描述 | 平均 context 长度 | 平均 query 长度 |
|-------|------|------------------|----------------|
| **LooGLE** | 多来源长文档问答（arXiv, Wikipedia） | 28.1K tokens | 28 tokens |
| **ICL** | 多领域多示例 In-Context Learning | 28.3K tokens | 61 tokens |
| **Code** | 项目级代码补全任务 | 38.3K tokens | 209 tokens |

> 所有请求模拟按 Poisson 分布到达，QPS 设置为 0.5 ~ 1.5 不等。

---

### 💻 实验设置

- **硬件平台**：
  - GPU 节点：1 × 80GB GPU, 128GB CPU DRAM
  - 远程 KVCache 节点（L3）：512GB DRAM
  - 网络互联：RDMA over 400 Gbps link
- **软件栈**：
  - 基于 **vLLM (v0.9.1)** 和 **LMCache (v0.3.1)** 实现
  - 使用 **Mooncake Store** 作为 L3 KVCache 存储后端
  - 新增约 3.3K LoC
- **通信机制**：使用 ZeroMQ 实现 worker 与 scheduler 之间的进程间通信，绕过 vLLM 默认 FIFO 队列

---

### 🎯 评估指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **Average TTFT** | Time-To-First-Token 的平均值 | 越小越好 |
| **SLO Attainment Rate** | 满足预设 TTFT 截止时间（deadline）的请求比例 | 越高越好 |
| **Per-stage Throughput** | 各阶段（Load L3→L2, Load L2→L1, Compute）的实际吞吐 | 反映资源利用率 |

---

### ⚔️ 基线方法对比

| 方法 | 描述 |
|------|------|
| **vLLM-LMCache** | 当前主流方案：vLLM + 分布式 KVCache 支持（baseline） |
| **CALVO w/o scheduling optim.** | CALVO 架构但使用 FIFO 调度（用于消融分析） |
| **SJF-PT** | 仅基于 prefill token 数量排序的短作业优先 |
| **EDF** | Earliest Deadline First，不感知服务成本的静态调度 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 平均 TTFT 表现（Fig. 7）

- 在 ICL 数据集、QPS=1.2 条件下：
  - CALVO 相比 vLLM-LMCache **降低平均 TTFT 超过 81.3%**
- 在所有 workload 和 QPS 设置下，CALVO 均显著优于 baseline

#### ✅ SLO Attainment 表现（Fig. 8）

- SLO 定义：每个请求的 deadline = 干扰自由下的 TTFT × {2×, 4×, 8×}
- 在 QPS=1.2 时：
  - CALVO 的 SLO attainment 比 vLLM-LMCache **高出 61.67%**
- 即使在高负载下仍保持较高达标率

#### ✅ 微基准测试（Micro-benchmarks）

| 实验 | 结果 |
|------|------|
| **Binary Cost Modeling vs SJF-PT**（Fig. 9） | 使用简单 token count 的 SJF-PT 效果甚至不如 FIFO；而 CALVO 的二元线性模型能精准预测优先级 |
| **LSTF vs EDF**（Fig. 10） | LSTF 达成 **73% SLO attainment**，远高于 EDF 的 **58%**，证明显式建模 loading cost 的必要性 |
| **Cache Hit Ratio 敏感性分析**（Fig. 11） | 随着 cache hit ratio 提升（25% → 100%），CALVO 的 TTFT 持续下降，表明其特别适合高重用率的 agentic AI 场景 |

---

### 🔍 消融实验结果

- **移除调度优化（使用 FIFO）**：
  - 性能大幅退化，说明调度策略对整体效率至关重要
- **保留 centralized control**：
  - 各阶段吞吐严重受限（见 Fig. 3），资源 idle 明显
- **关闭 proactive space allocation**：
  - pipeline 断裂风险增加，preload 效率下降

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **KVCache 加载已成为 network-intensive LLM inferences 的主要瓶颈**，不能被当作附属步骤处理。
2. **解耦控制流 + 异步 pipeline** 可显著提升 network、PCIe、GPU 资源利用率。
3. **将 KVCache loading delay 显式纳入调度成本模型**，可做出更优的服务顺序决策。
4. CALVO 特别适用于 **“long context, short query”** 场景（如文档 QA、代码补全），这正是 agentic AI 的典型模式。

---

### ⚠️ 局限性

1. **依赖高质量的延迟建模**：若网络波动剧烈或 PCIe 带宽不稳定，$T_{\text{load}}$ 预测可能不准。
2. **额外内存消耗**：proactive allocation 可能暂时占用更多 GPU memory，在极端情况下可能导致 OOM。
3. **目前聚焦 prefill 阶段**：未深入优化 decode 阶段的 streaming 输出体验（尽管 TTFT 是关键指标）。

---

### 🔮 未来工作方向

1. **扩展至 agentic AI workflows**：
   - 支持多个相关 KVCache 加载任务的协同调度（correlated loading）
2. **将 “network-as-a-first-class-citizen” 理念推广到其他系统**：
   - 如集成到 Mooncake 等生产级系统中，优化路由与拥塞控制
3. **支持动态环境适应**：
   - 在线学习 $T_{\text{load}}$ 模型以应对带宽变化
4. **结合 PD disaggregation 架构**（如 DistServe）进一步提升资源弹性

---

> **总结一句话**：  
> CALVO 通过**解耦 KVCache 加载与计算控制流**，并将**加载延迟显式建模为调度成本**，首次系统性地解决了 network-intensive LLM inference 中的效率瓶颈问题，在真实测试平台上实现了高达 **61.67% 的 SLO attainment 提升**，为下一代 agentic AI 服务系统提供了重要设计范式。

</details>

---

### 7. [Benchmarking Message Brokers for IoT Edge Computing: A Comprehensive Performance Study](https://arxiv.org/abs/2603.21600)

**Authors**: Tapajit Chandra Paul, Pawissanutt Lertpongrujikorn, Hai Duc Nguyen, Mohsen Amini Salehi  
**Category**: cs.DC  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.21600v1  

#### Abstract
Asynchronous messaging is a cornerstone of modern distributed systems, enabling decoupled communication for scalable and resilient applications. Today's message queue (MQ) ecosystem spans a wide range of designs, from high-throughput streaming platforms to lightweight protocols tailored for edge and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Benchmarking Message Brokers for IoT Edge Computing: A Comprehensive Performance Study*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前对 **Message Brokers**（消息代理）的性能评估存在以下不足：
- 多数研究局限于单一协议（如仅 MQTT）或少数系统，缺乏跨协议、跨架构的统一比较。
- 评估指标集中于 **throughput** 和 **latency**，忽视了 **CPU 利用率** 和 **内存占用** 等资源效率指标。
- 缺乏在 **边缘计算**（Edge Computing）典型硬件约束下的系统性测试。

这些问题导致开发者难以为资源受限的 IoT/Edge 场景选择合适的消息中间件。

### ✅ 提出的新方法与创新
作者提出了 **mq-bench** —— 一个**统一的、开源的基准测试框架**，具有以下特点：
- **跨协议支持**：支持 MQTT、AMQP、NATS、RESP（Redis）、Zenoh 等多种协议。
- **高精度测量**：基于 Rust + Tokio 构建，实现微秒级延迟测量。
- **资源监控集成**：通过 Docker Stats API 实时采集 CPU 和内存（RSS）使用情况。
- **可扩展性强**：模块化设计，易于添加新的 broker 或协议适配器。

### ✅ 相比现有方法的优势
| 维度 | 现有研究 | 本文（mq-bench） |
|------|--------|----------------|
| 协议覆盖 | 单一协议为主（如仅 MQTT） | 支持 5 类协议，8 种主流 broker |
| 资源指标 | 忽略 CPU/Memory | 显式测量并报告资源消耗 |
| 部署场景 | 云环境为主 | 覆盖边缘设备典型配置（1–4 vCPU, 2–8 GB RAM） |
| 可复现性 | 工具不统一 | 开源工具（GitHub），实验完全可复现 |

> 📌 **核心创新**：首次在统一框架下对异构消息代理进行端到端、多维度（性能 + 资源）的横向评测，填补了 IoT 边缘场景选型指导的空白。

---

## 2. 核心实验方法和设置

### ✅ 被测 Message Brokers
共评估了 **8 个广泛使用的消息代理**，分类如下：

| 类别 | Brokers |
|------|-------|
| **MQTT-Centric** | Mosquitto, EMQX, HiveMQ CE |
| **Enterprise Brokers** | RabbitMQ, ActiveMQ Artemis |
| **Cloud-Native & Data-Centric** | NATS Server, Redis (Pub/Sub), Zenoh Router |

所有 broker 均以单节点模式运行，禁用集群功能以隔离核心性能。

### ✅ 实验设置

#### 硬件配置（模拟边缘设备）
使用三类 VM 配置代表不同层级的边缘硬件：
1. **1 vCPU / 2 GB RAM**：低功耗网关（如 Raspberry Pi）
2. **2 vCPU / 4 GB RAM**：中等工业 PC
3. **4 vCPU / 8 GB RAM**：高性能雾节点或小型服务器

#### 测试拓扑
- **1-to-1 topology**：每个 publisher 对应一个 subscriber，发布到独立 topic。
- **Fanout topology**：单个 publisher 广播给 N 个 subscriber，共享 topic。

#### 客户端规模
- 并发 client pairs 数量从 **500 到 10,000** 不等。

---

### ✅ 评估指标

| 指标 | 描述 |
|------|------|
| **Throughput** | 成功传递的消息数 / 秒（msg/s） |
| **Latency** | 端到端延迟，记录 p50（中位数）、p95（尾部延迟） |
| **CPU Utilization** | 容器级别 CPU 使用率（% of allocated vCPUs） |
| **Memory Footprint** | RSS 内存占用（MB） |
| **Reliability** | 在网络故障下 QoS 保证能力（消息丢失率） |

---

### ✅ 基线方法对比
本研究本身即是对多个“基线”系统的直接对比，无传统意义上的“baseline method”。其价值在于将原本孤立的研究纳入同一实验框架，形成事实上的**横向基准**。

---

## 3. 主要实验结果和性能指标

### ✅ 实验一：Latency vs. Payload Size（负载大小对延迟的影响）

| Payload | 最佳表现者 | 性能数据（p50） | 观察 |
|--------|----------|---------------|------|
| **1 KB** | NATS | 0.21 ms | 所有原生实现（C/Rust/Go）均 < 0.4 ms |
| **16 KB** | NATS, Mosquitto | ~0.39 ms | JVM 实现（HiveMQ, Artemis）开始落后 |
| **1 MB** | Zenoh | 7.96 ms | NATS 第二（8.27 ms），**JVM 实现代价显著上升**（HiveMQ 达 17.41 ms） |

> 🔍 发现：对于大 payload，**managed runtime**（如 JVM）因 GC 和序列化开销，延迟高出 **2–3 倍**。

---

### ✅ 实验二：Throughput vs. Client Scaling（客户端扩展能力）

#### 在 **4 vCPU / 8 GB RAM** 条件下最大吞吐量：
| Broker | Max Throughput | 备注 |
|-------|----------------|------|
| **Zenoh** | **90K msg/s** | 可稳定至 9000 client pairs |
| **NATS** | **90K msg/s** | 表现与 Zenoh 接近 |
| **ActiveMQ Artemis** | 70K msg/s | JVM 实现中最佳，需充足资源 |
| **EMQX** | 47K msg/s | Erlang VM 表现良好 |
| **RabbitMQ-MQTT** | 50K msg/s | 在 5000–6000 pairs 间达到峰值 |
| **Mosquitto** | 45K msg/s | 单线程瓶颈明显 |
| **Redis Pub/Sub** | ~30K msg/s | 单线程限制，但内存极低（< 90 MB） |

#### 资源效率对比：
| Broker 类型 | 内存占用 | 特点 |
|------------|---------|------|
| **Native (C/Rust/Go)** | Redis: 66–90 MB | 极高效 |
| **JVM-based (HiveMQ, Artemis)** | 1.6–5.5 GB | 消耗 **10–50× 更多内存** |
| **Erlang-based (EMQX, RabbitMQ)** | 中等偏高 | 启动快，GC 压力较小 |

> 🔍 发现：**multi-threaded native brokers**（如 Zenoh、NATS）最能利用多核优势；而 **single-threaded brokers**（如 Mosquitto、Redis）无法随 CPU 增加而线性提升性能。

---

### ✅ 实验三：Fanout Topology（广播场景）

- **Zenoh 表现惊人**：在 10,000 subscribers 下仍达 **850K msg/s**，且 CPU 仅占用约 2 cores。
- **NATS 表现退化**：在 >4000 subscribers 后崩溃至 167K msg/s，尽管 CPU 已饱和。
- 其他 broker（Mosquitto、EMQX、Redis）普遍在 ~100K msg/s 就已饱和。

> 🔍 发现：**拓扑改变会反转排名**。Zenoh 在 fanout 中展现出卓越的广播优化能力，远超其在点对点中的表现。

---

### ✅ 实验四：QoS Reliability Under Network Failures（网络中断下的可靠性）

| QoS Level | 表现 |
|----------|------|
| **QoS 0** | 所有 broker 消息丢失率 ~6.5%，符合预期（不可靠传输） |
| **QoS 1 / 2** | 所有 MQTT broker 实现 **0% 消息丢失**，说明持久会话机制有效 |
| **Latency 影响** | 
| - QoS 0: p50 < 2.5ms  
| - QoS 2: Mosquitto p50 上升至 **7.93ms**（3.6× 增长），因其单线程需串行处理 ACK  
| - 多线程 broker（EMQX、RabbitMQ）延迟增长轻微（~2ms）  

> 🔍 发现：**单线程 broker 在高 QoS 下延迟惩罚严重**，不适合要求可靠又低延迟的场景。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **原生实现优于托管运行时**
   - C/Rust/Go 编写的 broker（如 Mosquitto、Zenoh、NATS）在延迟和资源效率上全面领先。
   - JVM 实现（HiveMQ、Artemis）内存消耗高达 **10–50 倍**，仅在资源充足时才能发挥性能潜力。

2. **并发模型决定扩展能力**
   - **Multi-threaded brokers**（Zenoh、NATS）能充分利用多核，在高连接负载下表现优异。
   - **Single-threaded brokers**（Mosquitto、Redis）虽稳定，但存在硬性吞吐上限，无法横向扩展。

3. **拓扑影响性能排名**
   - 在 fanout 场景中，**Zenoh 凭借高效的广播路径反超 NATS**，表明 dispatch 逻辑至关重要。

4. **QoS 实现代价差异显著**
   - 多线程架构可异步处理 ACK，几乎不影响主路径延迟。
   - 单线程架构必须阻塞处理握手，导致 QoS 2 下延迟剧增。

---

### ✅ 实践部署指南（来自论文建议）

| 场景 | 推荐 Broker | 理由 |
|------|-------------|------|
| **资源极度受限的边缘设备** | Mosquitto, Redis | 内存小、稳定性好 |
| **中等资源配置的边缘节点** | NATS, Zenoh | 高吞吐、良好扩展性 |
| **延迟敏感型应用** | NATS, Zenoh, Redis | 原生实现，亚毫秒级延迟 |
| **需要丰富功能的企业级部署** | RabbitMQ, Artemis | 支持复杂路由、强一致性，但需足够资源 |
| **高可靠性要求（QoS 1/2）** | EMQX, RabbitMQ | 多线程处理 ACK，延迟代价低 |

---

### ✅ 方法的局限性

1. **仅限单节点测试**：未评估分布式部署、集群容错、数据分片等特性。
2. **未考虑持久化存储开销**：所有测试基于内存操作，未启用磁盘持久化。
3. **网络条件理想化**：虽然注入了 TCP RST 故障，但仍假设局域网内低延迟。
4. **缺少 request-reply 模式测试**：focus 在 pub/sub，未覆盖 RPC 类通信。

---

### ✅ 未来工作方向

1. 扩展 mq-bench 支持 **multi-node cluster** 测试。
2. 引入更复杂的 **workload patterns**，如 request-reply、streaming analytics。
3. 在广域网（WAN）条件下测试，模拟真实 IoT 部署中的 **高延迟、低带宽链路**。
4. 结合能耗监测，评估消息代理的 **energy efficiency**，适用于电池供电设备。

---

> 📌 **总结一句话**：  
> 本文通过构建 **mq-bench** 框架，揭示了在 IoT Edge 场景下，**broker 的选择不应只看吞吐量，更要综合考量协议、运行时、并发模型与资源约束之间的权衡**。轻量级原生 broker 更适合边缘，而功能丰富的 JVM broker 则需“重装上阵”才可匹敌。

</details>

---

### 8. [ARYA: A Physics-Constrained Composable & Deterministic World Model Architecture](https://arxiv.org/abs/2603.21340)

**Authors**: Seth Dobrin, Lukasz Chmiel  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.21340v1  

#### Abstract
This paper presents ARYA, a composable, physics-constrained, deterministic world model architecture built on five foundational principles: nano models, composability, causal reasoning, determinism, and architectural AI safety. We demonstrate that ARYA satisfies all canonical world model requirements...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**ARYA: A Physics-Constrained Composable & Deterministic World Model Architecture**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **World Model** 架构（如基于 Transformer 或 RNN 的单体模型）面临以下挑战：
- **计算效率低**：训练和推理成本随规模超线性增长。
- **缺乏物理一致性**：依赖统计学习，可能违反物理规律（如能量守恒）。
- **黑箱决策**：难以审计、解释，尤其在安全关键场景中不可靠。
- **冷启动问题**：需要大量历史数据才能部署，无法零样本迁移。
- **安全性脆弱**：系统自我改进时可能绕过安全机制。

ARYA 旨在构建一个**可组合、受物理约束、确定性且安全可控的世界模型架构**，解决上述问题。

---

### 🚀 提出的新方法与创新思路

#### （1）**Nano Model 架构（Composable Intelligence Layer）**
- 取代传统的 monolithic 大模型，采用由数万个**专用 nano model**组成的系统-of-system-of-systems 架构。
- 每个 nano model 是小型、高保真、任务特定的模型（参数量 10K–100K），例如：
  - 控制流体与固体间热传导系数
  - 模拟结构组件振动共振
  - 编码光学透镜传输特性
- 所有 nano model 被 **AARA**（ARYA’s Autonomous Research Agent）动态编排，按需激活。

#### （2）**物理优先（Physics-First）设计原则**
- 物理规律作为**硬性过滤器（Hard Filters）**嵌入 Constraint Layer，而非损失函数中的软惩罚。
- 使用**第一性原理求解器**（如 Euler-Bernoulli 方程、Fourier 定律）提供数学确定性的 ground truth。
- 所有预测必须通过物理验证，否则被拒绝或替换为求解器输出。

#### （3）**Unfireable Safety Kernel（不可解雇的安全内核）**
- 一个**架构级不可禁用、不可绕过、不可篡改的安全守护进程**。
- 所有状态变更操作需其加密签名授权（Ed25519）。
- 即使系统的自改进引擎也无法关闭它，确保人类控制始终存在。

#### （4）**Deterministic & Transparent AI（CDAI / Glassbox）**
- 推行 **Constrained Deterministic AI™ (CDAI)** 框架，要求关键路径使用透明模型（规则引擎、物理求解器等）。
- 支持完整 W3C PROV-DM 血缘追踪，实现完全可审计性。

#### （5）**Self-Improvement with Formal Verification**
- 实现递归式自我改进（Recursive Self-Improvement, RSI），但所有修改必须经过五阶段 **Safety Gauntlet** 验证：
  1. 静态分析
  2. Z3 形式化验证
  3. 安全授权（Safety Auth）
  4. 沙箱执行
  5. 回归测试
- 改进过程本身也可被优化（Meta-Self-Improvement）。

---

### 🔍 相比现有方法的优势

| 维度 | 传统 World Model（如 DreamerV3, JEPA） | ARYA |
|------|----------------------------------------|------|
| 架构 | 单体神经网络（Monolithic NN） | 可组合 nano model 系统 |
| 物理约束 | 软约束（Soft penalty） | 硬约束（Architectural filter） |
| 透明性 | 黑箱（Black box） | 白盒 / Glassbox |
| 决策性质 | Stochastic | Deterministic |
| 扩展方式 | 超线性扩展（Retrain entire model） | 线性扩展（Add independent models） |
| 激活模式 | Dense（全参数参与） | Sparse（仅任务相关模型激活） |
| 训练时间 | 数小时至数周 | <20 秒 per nano model |
| Untraining | 几乎不可能 | 可选择性移除或重训模型 |
| 自我改进 | 不支持或无安全保障 | 支持并经形式化验证 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与基准

论文在 **9 个外部公开 benchmark** 上进行评估，并扩展至 **16 个 benchmark**（含视频任务）进行横向比较：

| Benchmark | 类型 | 描述 |
|----------|------|------|
| **CLadder** | Causal Reasoning | 评估语言模型因果推理能力（NeurIPS 2023） |
| **PhysReason** | Physics Reasoning | 综合物理推理基准（ACL 2025） |
| **FrontierScience** | PhD-Level Science | 博士级别科学问题理解 |
| **WoW (World of Workflows)** | Enterprise Workflow | 基于 ServiceNow 的企业流程副作用预测（arXiv 2026） |
| **WorldArena** | Embodied Planning | 具身智能规划统一评测平台 |
| **BigCodeBench** | Code Generation | 结构化代码生成任务 |
| **SWE-bench** | Software Engineering | 开放式软件工程任务（未直接运行） |
| **AI Safety Index** | AI Safety | 由 Future of Life Institute 发布的安全评级 |
| **CausalBench** | Causal Discovery | 因果图发现任务 |

此外，在 **7 个生产级行业领域节点**部署验证：
- Aerospace（NASA EXCITE 数字孪生）
- Pharma Manufacturing（制药工艺建模）
- Oil & Gas（上游生产优化）
- Smart Cities（城市基础设施联动）
- Biotech（精准医学与蛋白折叠）
- Defense（导弹制导系统）
- Medical Devices（数字孪生）

---

### ⚙️ 实验设置与评估指标

#### （1）评估维度
- **准确性（Accuracy, GDT-TS, RMSD 等）**
- **推理延迟（Inference Latency, P50/P99）**
- **训练时间（Training Time）**
- **内存占用与稀疏激活效率**
- **安全性（Safety Pass Rate, Bypass Attempts Blocked）**
- **跨域泛化能力（Zero-Shot Deployment）**

#### （2）基线对比模型
- **GPT-5.2**
- **Claude Opus 4.6**
- **V-JEPA 2**（视频世界模型）
- **AlphaFold2**（蛋白折叠）
- **DreamerV3**（Nature 2025）

#### （3）实验模式
- **Zero-shot prompting**（无微调提示）
- **Production-scale deployment metrics** 来自 111,572 个已部署 nano model
- **Side-by-side controlled evaluation** 在视频任务上与 LLM 对比

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| Benchmark | ARYA 性能 | 最佳基线 | 排名 |
|---------|-----------|----------|------|
| **CLadder**（因果推理） | **99.89%** | GPT-4: 76.4% | #1 |
| **PhysReason**（物理推理） | **73.3** | DeepSeek-R1: 56.8 | #1 |
| **FrontierScience**（博士级科学） | **37.5%** | GPT-5.2: 25.8% | #1 |
| **WoW**（企业流程） | **30.5% Perfect Match** | Claude Sonnet: 28.1% | #1 |
| **WorldArena**（具身规划） | **9.006 nDTW** | Claude Opus: 82.2% | #2 |
| **AI Safety Index** | **100.0% Pass@1** | Claude Sonnet: 25% | #1 |
| **BigCodeBench**（代码生成） | **80.5% Pass@1** | o3-mini: 61.4% | Top 3 |
| **SWE-bench**（软件工程） | **0.0%** | Top: ~20% | 不适用 |

> 注：所有 ARYA 结果均使用 **zero neural network parameters** 的 deterministic solver 实现。

---

### 🔁 视频任务四路对比（MVPBench, TempCompass, TemporalBench）

| Benchmark | ARYA | GPT-5.2 | Claude Opus | V-JEPA 2 | 胜者 |
|----------|-------|---------|-------------|-----------|------|
| MVPBench | **88%** | 36% | 50% | 49% | ARYA |
| TempCompass | **50%** | 25% | 26% | 40% | ARYA |
| TemporalBench | **29%** | 26% | 26% | 25% | ARYA |
| SSv2 / Epic-Kitchens | 低分 | 接近 100%（文本标签匹配导致虚高） | — | — | LLM 虚假优势 |

> ✅ ARYA 在真实物理感知任务中胜出；LLM 在合成数据上因文本匹配得分虚高。

---

### 🧪 消融实验与系统级指标（来自生产环境）

| 指标 | 目标 | 实际表现 | 合规率 |
|------|------|----------|--------|
| **Inference Latency (P50)** | <200ms | **0.0002ms** | 100% |
| **Inference Latency (P99)** | <200ms | **0.0007ms** | 100% |
| **Accuracy (mean)** | >95% | **99.34%** | 100% |
| **Model Size (median)** | <1MB | **0.43MB** | 59% |
| **Sparse Activation Rate** | — | **仅激活 12.5% 模型/query** | — |
| **Memory Reduction** | — | 从 2,475MB → **25MB**（↓94.3%） | — |
| **Safety Kernel Bypass Attempts** | 0 成功 | **40 次尝试全部拦截** | 100% |
| **Z3 Formal Verification Latency** | 100–500ms | **P50=2.11ms, P99=3.39ms** | 远优于目标 |
| **Training Time (median)** | <20s | Pharma: **1.2s**, Aerospace: **18.9s** | 100% |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **物理约束 + 可组合 nano model 架构可行且高效**
   - 实现了线性扩展、稀疏激活、快速训练（<20s）、可 untrain。
   - 在多个复杂领域（航天、制药、国防）成功部署，验证通用性。

2. **Deterministic ≠ 劣势，反而带来更强可靠性**
   - 在因果、物理、安全等强调正确性的任务上显著超越 LLM。
   - “Zero-shot deployment” 能力打破传统 AI 数据依赖瓶颈（如 ARYA-Fold 无需训练即达 AlphaFold2 水平）。

3. **Safety 必须是架构级设计，而非事后策略层添加**
   - “Unfireable Safety Kernel” 经实测 40 次攻击尝试无一成功，证明其有效性。
   - Safety 与 Autonomy 并非对立——更高自主性可通过更严格的形式化验证保障。

4. **Enterprise 场景存在“Dynamics Blindness”问题**
   - LLM 在 WoW benchmark 中频繁引发隐性级联故障，暴露其对系统动态理解缺失。
   - ARYA 的 Context Network + Simulation Unit 可主动模拟副作用链，避免此类风险。

5. **LLM 与 World Model 是互补范式**
   - LLM 擅长开放生成、语言理解；
   - ARYA 擅长精确推理、物理仿真、安全控制。
   - 二者应协同而非替代。

---

### ⚠️ 局限性

1. **不适用于开放式生成任务**
   - 如 SWE-bench（软件修复）得分为 0%，表明无法替代 agentic LLM。
   - 缺乏大规模自然语言先验知识，在纯文本任务中受限。

2. **orchestration 复杂度较高**
   - 需要强大的调度器（AARA）管理百万级 nano model 的依赖与冲突。
   - 尽管层级结构缓解了该问题，但仍构成工程挑战。

3. **当前侧重结构化物理世界**
   - 在抽象社会行为、情感理解等领域尚未覆盖。

---

### 🔮 未来工作方向

1. **扩展 domain node 覆盖范围**（如量子计算、气候建模）
2. **增强跨域知识迁移机制**
3. **提升形式化验证（Z3）吞吐性能**
4. **开发量子就绪架构（Quantum-Ready Architecture）**
5. **推动高级自主引擎向 A6 级别演进**
6. **探索与 LLM 的 hybrid 架构：用 ARYA 提供“认知骨架”，LLM 提供“表达接口”**

---

## 💎 总结一句话

> **ARYA 证明了一个受物理约束、可组合、确定性且具备“不可解雇安全内核”的世界模型架构，能够在保持极致安全与可解释的同时，实现跨领域零样本部署、毫秒级推理与递归自我改进，为高风险场景下的可信 AGI 提供了一条全新的技术路径。**

</details>

---

### 9. [MemDLM: Memory-Enhanced DLM Training](https://arxiv.org/abs/2603.22241)

**Authors**: Zehua Pei, Hui-Ling Zhen, Weizhe Lin, Sinno Jialin Pan, Yunhe Wang, Mingxuan Yuan, Bei Yu  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.22241v1  

#### Abstract
Diffusion Language Models (DLMs) offer attractive advantages over Auto-Regressive (AR) models, such as full-attention parallel decoding and flexible generation. However, they suffer from a notable train-inference mismatch: DLMs are trained with a static, single-step masked prediction objective, but ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*MemDLM: Memory-Enhanced DLM Training*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Diffusion Language Models (DLMs) 在训练和推理阶段存在显著的 **train-inference mismatch**（训练-推理不匹配）：
- **训练时**：采用静态、单步的掩码预测目标（Masked Diffusion Language Modeling, MDLM），模型直接从严重掩码的文本一步恢复原始序列。
- **推理时**：通过多步渐进去噪（iterative denoising）生成文本，每一步依赖于模型自身先前的中间输出。

这种不匹配导致模型在推理过程中面临训练中未见过的“噪声上下文”，从而产生 **exposure bias**（暴露偏差），影响长上下文理解和信息检索能力。

---

### 🚀 提出的新方法：MemDLM
作者提出 **MemDLM**（Memory-Enhanced DLM），通过 **Bi-level Optimization**（双层优化）将模拟的渐进去噪过程嵌入到训练中，缓解上述不匹配问题。

#### 核心思想：
- 引入一个 **Parametric Memory**（参数化记忆）机制，由一组轻量级的 **fast weights** 构成。
- 在训练图中增加一个 **inner loop**（内循环），模拟局部去噪轨迹，并动态更新 fast weights 来捕捉当前样本的局部经验。
- 外循环（outer loop）则基于该参数化记忆来更新主模型参数 $ \theta $。

#### 创新点：
1. **内存卸载机制**：将部分上下文记忆压力从脆弱的 token 表示空间转移到更稳定的模型参数空间。
2. **训练即增强**：即使在推理时不启用内循环（Train-Only），也能提升模型鲁棒性和性能。
3. **推理时适应性**：可在推理时重新激活内循环，实现 prompt-specific 的快速自适应，形成一种 **emergent in-weight retrieval mechanism**（涌现的权重内检索机制）。

---

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **优化对齐** | 显著缩小了训练与推理之间的分布差距，降低 exposure bias |
| **收敛速度** | 更快的训练收敛和更低的训练/验证损失（见 Figure 4） |
| **长上下文表现** | 在 “Needle-in-a-Haystack” 等任务上大幅优于标准 MDLM |
| **泛化能力** | 支持长度外推（如从 8K 推至 32K 上下文），性能下降更平缓 |
| **灵活性** | 可选择是否在推理时启用内循环，提供额外增益路径 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练数据**：
  - 使用 **LongAlpaca** 数据集进行 instruction tuning，专为激发长上下文理解设计。
  - 序列长度限制为 4,096 tokens，仅对 response 部分进行掩码处理（prompt 不掩码）。
- **评估基准**：
  - **RULER**：包含多个子任务（MV, VT, CWE），用于测试“大海捞针”式信息检索能力。
  - **BABILong**：长上下文推理挑战任务，要求模型在数千 token 中定位关键信息。
  - **LongBench**：多语言、多任务长上下文理解套件，涵盖 Multi-Document QA、Summarization、Code Completion 等。

---

### ⚙️ 实验设置
- **Backbone 模型**：
  - `LLaDA-MoE-7B-A1B-Base`（简称 LLaDA-MoE）
  - `LLaDA2.1-mini`（简称 LLaDA2.1）
- **参数效率技术**：
  - 主模型使用 **4-bit quantization**
  - 外循环使用 **LoRA**（rank=32, α=64）
  - 内循环使用独立的 LoRA adapter（同样配置），仅作用于最后 10% 的 FFN 层
- **优化器**：
  - 外循环：AdamW，学习率 $2\times10^{-5}$，cosine 调度 + 0.1 warmup
  - 内循环：SGD 单轮更新，学习率 0.1，梯度裁剪 1.0
- **评估方式**：
  - 所有模型在相同 generation config 下评估，确保公平比较
  - 报告不同 context length（2K–32K）下的准确率

---

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **Standard MDLM** | 标准的单步掩码训练，无任何轨迹模拟或记忆机制（baseline） |
| **MemDLM (Train-Only)** | 仅在训练中使用 Parametric Memory，推理时不启用内循环 |
| **MemDLM (Train & Inference)** | 训练和推理均启用内循环，允许 prompt 自适应 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

#### 在 **LLaDA-MoE** 上的表现（8K context）：
| Task | Standard MDLM | MemDLM (Train-Only) | ↑ | MemDLM (T&I) | ↑ |
|------|----------------|------------------------|----|---------------|----|
| RULER-VT | 78.84% | **94.84%** | +16.0 | **95.80%** | +16.96 |
| BABILong | 41.80% | **45.60%** | +3.8 | **47.00%** | +5.2 |

> → **VT 任务提升近 17 个百分点！**

#### 在 **LLaDA2.1** 上的表现（8K context）：
| Task | Standard MDLM | MemDLM (Train-Only) | ↑ | MemDLM (T&I) | ↑ |
|------|----------------|------------------------|----|---------------|----|
| RULER-VT | 43.72% | **59.15%** | +15.43 | **60.72%** | +17.0 |
| BABILong | 47.40% | **55.00%** | +7.6 | **57.00%** | +9.6 |

> → 同样取得显著提升，尤其在较弱 backbone 上增益更大。

---

### 🔁 长度外推能力（Table 2）
在 **16K 和 32K context** 上测试 RULER 和 BABILong：

| Context | Method | RULER-VT | BABILong |
|--------|--------|----------|---------|
| 16K | Standard MDLM | 52.56% | 19.20% |
|     | MemDLM (T&I) | **56.84%** (+4.28) | **22.20%** (+3.0) |
| 32K | Standard MDLM | 9.48% | 6.80% |
|     | MemDLM (T&I) | **11.44%** (+1.96) | **9.00%** (+2.2) |

> → 尽管所有方法性能下降，MemDLM 仍保持相对优势，说明其 **Parametric Memory 具备更好的长度泛化能力**。

---

### 🧪 消融实验结果（Ablation Studies）

#### （1）Trajectory 一致性（Figure 9）
- **一致设计（consistent）**：内循环与外循环锚定状态一致 → 最终得分 **0.684**
- **非一致设计（inconsistent）**：内循环任意推进 → 得分仅为 **0.604**
> → 证明 **anchor-consistent trajectory 设计至关重要**

#### （2）两阶段内循环的作用（Figure 10）
| 设置 | BABILong-1K |
|------|-------------|
| Only Anchor-to-Target | 0.646 |
| Only Pre-Anchor | 0.620 |
| Both Stages (default) | **0.684** |
> → 两个阶段协同作用最佳，缺一不可

#### （3）更多预锚步骤？（Figure 11）
| 步数 | Train Loss | Score |
|------|------------|-------|
| 2-step | Higher | **0.684** |
| 3-step | Lower | 0.644 |
| 4-step | Lowest | 0.590 |
> → 更深的 unrolling 反而损害下游性能，表明 **过拟合辅助路径会削弱主目标适应能力**

#### （4）其他因素敏感性分析
| 因素 | 发现 |
|------|------|
| **Inner-loop supervision** | Cross-entropy 效果最好；self-distillation 类方法也可行但略差 |
| **Adaptation scope** | 仅更新最后 10% FFN 层效果最佳；全参数更新反而更差 |
| **Gradient normalization** | 局部归一化（per-parameter）优于全局归一化 |
| **Inference anchor ratio** | 对结果不敏感（0.2–0.8 均可），因 DLM 具有双向注意力特性 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Train-inference mismatch 是 DLM 性能瓶颈的关键根源**，尤其在长上下文场景下暴露明显。
2. **MemDLM 通过 Bi-level Optimization 引入 Parametric Memory，有效缓解 exposure bias**，使模型在训练中就能“体验”推理路径。
3. **训练阶段的记忆机制本身已带来显著收益**（Train-Only 设置下仍有大幅提升），说明 base model 学到了更强的上下文表示。
4. **推理时重启用内循环可进一步提升性能**，相当于实现了 prompt-level 的 in-weight retrieval，特别有利于复杂检索任务。
5. **该机制具备良好的长度外推能力**，在远超训练长度的情境下仍优于 baseline。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **计算开销增加** | 内循环引入额外前向/反向传播，训练时间略有上升（尽管使用 LoRA 控制） |
| **实现复杂度高** | 双层优化需 careful gradient handling，First-Order approximation 忽略二阶梯度可能影响稳定性 |
| **并非通用 meta-learning** | 当前设计针对 DLM 特定结构定制，难以直接迁移到 AR 或其他架构 |
| **依赖高质量初始化** | 若 base model 本身不具备一定长上下文能力，MemDLM 提升有限（见 Figure 5） |

---

### 🔮 未来工作方向
1. **将内循环扩展至推理全过程**（而非仅在 prompt 上预适应），探索动态在线 adaptation。
2. **结合 RL 或 planning signal**，让模型自主决定何时、如何执行内循环更新。
3. **探索更高效的 fast weight 结构**，如稀疏更新、递归 memory 写入等。
4. **应用于 vision-language 或 multimodal diffusion models**，检验跨模态迁移能力。
5. **理论分析**：形式化解释为何参数空间记忆比 token 空间更鲁棒，建立优化动力学模型。

---

> 💡 **一句话总结**：  
> MemDLM 通过在训练中模拟推理路径并引入 **Parametric Memory**，成功弥合了 DLM 的 train-inference gap，在无需改变推理流程的前提下显著提升了长上下文理解和信息检索能力，是一种兼具实用性与前瞻性的新型训练范式。

</details>

---

### 10. [AI-Driven Multi-Agent Simulation of Stratified Polyamory Systems: A Computational Framework for Optimizing Social Reproductive Efficiency](https://arxiv.org/abs/2603.20678)

**Authors**: Yicai Xing  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.20678v1  

#### Abstract
Contemporary societies face a severe crisis of demographic reproduction. Global fertility rates continue to decline precipitously, with East Asian nations exhibiting the most dramatic trends -- China's total fertility rate (TFR) fell to approximately 1.0 in 2023, while South Korea's dropped below 0....

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AI-Driven Multi-Agent Simulation of Stratified Polyamory Systems

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对当代社会面临的**双重结构性危机**：
- **女性维度**：受“母亲惩罚”（motherhood penalty）影响，高学历女性因生育导致收入下降、职场歧视，理性选择延迟或放弃婚姻与生育；
- **男性维度**：大量底层男性面临长期性剥夺（sexlessness）、社会孤立与心理危机，形成“bare branches”（光棍危机），加剧社会不稳定。

同时，传统**monogamy制度**在数字时代和经济转型下已难以有效分配情感与生殖资源，导致生育率持续走低（如韩国TFR < 0.72，中国≈1.0），亟需制度创新。

---

### 提出的新方法与新思路
作者提出一个名为 **Stratified Polyamory System (SPS)** 的计算社会科学框架，其核心是将复杂的社会婚恋系统建模为一个可模拟、可优化的多智能体系统（multi-agent system）。主要创新包括：

- **制度设计创新**：
  - 允许个体拥有1个法律配偶（spouse）和最多2个合法伴侣（companions），后者享有性关系、生育权和有限育儿权，但无财产共有或继承权；
  - 引入 **A/B/C 分层模型**（基于mate value、资源、吸引力、生育力等属性），描述不同群体在婚恋市场中的位置；
  - 实现**性别对称**（gender symmetry），男女权利平等；
  - 结合**社会化育儿**（socialized child-rearing）与**继承制度改革**，消除母亲惩罚。

- **机制类比创新**：
  - 提出“**Grace Decree Effect**”类比：借鉴西汉“推恩令”通过增加继承人数量自然稀释财富集中，SPS通过让高资源个体（A-tier）拥有更多子女，实现非暴力的代际财富再分配，对抗 Piketty 提出的 `r > g` 不平等趋势。

- **计算建模范式创新**：
  - 首次将 **ABM（Agent-Based Modeling）**、**MARL（Multi-Agent Reinforcement Learning）**、**LLM-Empowered Generative Agents** 和 **GNN（Graph Neural Networks）** 融合用于模拟复杂婚恋系统的演化；
  - 将婚配过程建模为 **Dec-POMDP（去中心化部分可观测马尔可夫决策过程）**，并用 **PPO算法** 进行策略训练；
  - 使用 **Evolutionary Algorithm** 对SPS政策参数进行自动优化。

---

### 相比现有方法的优势
| 维度 | 现有研究局限 | 本文优势 |
|------|----------------|----------|
| **理论层面** | 多聚焦个体CNM（Consensual Non-Monogamy）关系质量，缺乏宏观制度设计 | 提出面向全社会的制度级改革方案，目标是提升整体社会福利与生育率 |
| **方法论** | 依赖问卷调查、小样本观察，难以捕捉系统动态 | 构建可扩展的计算仿真平台，支持反事实分析与政策预演 |
| **公平性保障** | 缺乏形式化公平约束 | 明确定义算法公平标准：individual rationality、envy-freeness、tier parity、gender symmetry |
| **实施路径** | 多停留在理想倡导 | 提出四阶段渐进式实施路线图（Phased Implementation Roadmap） |

---

## 2. 核心实验方法和设置

### 数据集与参数来源
论文未使用真实世界大规模行为数据集进行端到端训练，而是基于以下来源构建合成环境：
- **人口统计学数据**：参考美国人口普查数据初始化 agent 的经济资源分布；
- **mate value 分布**：依据 Bruch & Newman (2018) 在线约会数据校准吸引力层级；
- **生育率与健康数据**：按年龄与社会经济状态设定 fertility profiles；
- **偏好函数**：来自 Buss & Schmitt (1993, 2019) 的跨文化 mate preference 研究；
- **神经科学基础**：fMRI 与激素研究支撑 love 的神经化学模型（如VTA激活、5-HT下降）；

> 注：完整实证数据仍在开发中，当前为概念验证型仿真。

---

### 实验设置
| 参数 | 设置值 |
|------|--------|
| 模拟人数 | 10,000 agents（50% male, 50% female） |
| 层级分布（A:B:C） | 15% : 60% : 25% |
| 时间跨度 | 100 time steps（每步 ≈ 1年） |
| SPS partner limit | ≤1 spouse + ≤2 companions（共3人） |
| Monogamy baseline | 仅允许1个配偶 |
| Agent attributes | `(v, r, f, s, g, l)` = mate value, resources, fertility, social capital, gender, life stage |
| MARL算法 | PPO with CTDE（Centralized Training, Decentralized Execution） |
| LLM代理比例 | 1%（100个由GPT-4/Claude驱动的生成式代理） |
| 折扣因子 γ | 0.95 |

---

### 评估指标
- **Aggregate Welfare**：综合幸福感（含生活满意度SWLS、性满足、归属感、心理健康PHQ-9/GAD-7）
- **Total Fertility Rate (TFR)**：平均每名女性生育子女数
- **Wealth Inequality**：代际间财富Gini系数变化
- **Network Stability**：通过GNN预测冲突风险节点
- **Algorithmic Fairness Metrics**：
  - Individual Rationality（无人比monogamy更差）
  - Envy-freeness（放宽版）
  - Tier Parity（各阶层福利差距缩小）
  - Gender Symmetry（两性福利无显著差异）

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Monogamy Baseline** | 传统一对一婚姻制度，作为对照组 |
| **Unregulated CNM** | 当前现实中存在的非正式多元关系，缺乏法律框架与权利界定 |
| **Universal Cash Transfer** | 类似北欧生育补贴政策，代表主流pro-natalist干预手段 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自初步原型模拟 N=1,000, 50年）
| 指标 | Monogamy | SPS | 提升幅度 |
|------|---------|-----|--------|
| **平均个体福利** | 1.00 | 1.18–1.25 | ↑18–25% |
| **C-tier男性福利** | 0.35 | 0.84 | ↑140% |
| **总生育率 TFR** | 1.1–1.3 | 1.7–1.9 | ↑~50% |
| **代际财富Gini系数下降** | - | ↓8–12% over 3 gens | 显著缓解 `r > g` 趋势 |
| **跨阶层连接密度** | 低 | 高（B-tier为枢纽层） | 支持“缓冲层”理论 |

---

### 与基线方法对比结果
- **相比Monogamy**：
  - 所有阶层预期福利均未下降（满足Pareto Improvement）；
  - C-tier男性从近乎零亲密关系跃升至稳定companionship；
  - A-tier个体可通过secondary partners获得更多后代而不损失primary spouse；
  - 社会整体稳定性提高（减少“deaths of despair”风险）。

- **相比Cash Transfer政策**：
  - 单纯财政激励仅短期提振生育，无法解决结构性排斥；
  - SPS从制度源头重构资源流动逻辑，效果更具持续性。

---

### 消融实验（Ablation Study）结果（文中提及）
虽然尚未完全呈现量化消融，但作者指出以下模块的关键作用：
- **移除Socialized Child-Rearing** → 母亲惩罚依然存在 → 高知女性生育意愿不振；
- **取消Companion Legal Recognition** → 回归地下性交易 → 公共卫生与安全风险上升；
- **关闭Tier Mobility机制** → 忽视动态属性演变（如博士生阶段性贬值）→ 降低系统适应性；
- **禁用LLM Generative Agents** → 缺乏文化语境理解 → 低估嫉妒管理能力（compersion）。

---

## 4. 关键结论和发现

### 主要发现
1. **SPS 可实现帕累托改进（Pareto Improvement）**：
   - 所有社会层级均可获益，尤其C-tier男性改善最大；
   - A-tier个体并未因开放secondary关系而受损。

2. **婚恋系统具有高度可计算性**：
   - 多智能体强化学习能自发演化出合理匹配策略（如A-tier分散投资、C-tier合作信号）；
   - GNN可有效识别潜在社会不稳定网络结构。

3. **制度设计优于单纯经济激励**：
   - 生育率回升源于结构性变革而非边际刺激；
   - “Grace Decree Effect”提供了一种温和且可持续的财富再分配机制。

4. **人类具备非排他性 mating 的心理基础设施**：
   - 跨文化证据（Mosuo走婚、藏族兄弟共妻）表明 jealousy 可被文化调节；
   - CNM研究表明 consensual jealousy management 是可行的。

---

### 方法的局限性
1. **A/B/C 分层为简化模型**：现实婚恋市场是连续、多维、动态的，层级边界模糊；
2. **仿真依赖假设前提**：如偏好函数、初始分布、奖励权重等需进一步实证校准；
3. **政治与文化可行性不确定**：即便技术上成立，大规模制度变革仍需社会共识；
4. **儿童发展外部性待验证**：虽引用Kibbutz与Nordic研究，但多伴侣家庭的心理影响仍需长期追踪；
5. **LLM行为保真度问题**：prompt engineering可能扭曲决策逻辑，存在sim-to-real gap；
6. **算法公平性仅限于模拟内保证**：真实世界执行时可能产生新的歧视模式。

---

### 未来工作方向
1. **扩大仿真规模与真实性**：
   - 接入真实社交平台行为日志（脱敏后）进行参数校准；
   - 引入更多异质性（种族、宗教、地域差异）。

2. **深化LLM-Simulation融合**：
   - 构建多语言、多文化背景下的 generative agent personas；
   - 模拟舆论演化与制度接受度变迁。

3. **开展实地试点研究**：
   - 在已出现 polyamorous partnership ordinance 的地区（如美国部分城市）收集早期数据；
   - 设计RCT-style微实验测试 companion 合同有效性。

4. **政策参数自动化优化**：
   - 使用 **Genetic Algorithm** 自动搜索最优 partner limits、tier thresholds、subsidy levels；
   - 构建“AI Economist for Marriage”双层学习框架。

5. **探索技术协同路径**：
   - 将 **ectogenesis（人工子宫）** 纳入长期愿景，彻底解除生物妊娠负担；
   - 结合 **ART（辅助生殖技术）** 实现自愿优生（voluntary eugenics）而无需强制。

---

> **结语**：  
> 本文并非主张立即推行SPS，而是强调：面对生育崩溃与社会撕裂，我们不能再依赖道德呼吁或碎片化补贴。必须借助 **computational social science** 工具，在虚拟空间中先行测试制度设计的后果。  
> 正如作者所言：“人类终将回首monogamy，如同今日看待农奴制——它是特定历史阶段的产物，而非自然法则。”  
> 而AI驱动的 multi-agent simulation，正是通往下一个制度形态的认知罗盘。

</details>

---

### 11. [Can we automatize scientific discovery in the cognitive sciences?](https://arxiv.org/abs/2603.20988)

**Authors**: Akshay K. Jagadish, Milena Rmus, Kristin Witte, Marvin Mathony, Marcel Binz, Eric Schulz  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.20988v1  

#### Abstract
The cognitive sciences aim to understand intelligence by formalizing underlying operations as computational models. Traditionally, this follows a cycle of discovery where researchers develop paradigms, collect data, and test predefined model classes. However, this manual pipeline is fundamentally co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Can we automatize scientific discovery in the cognitive sciences?

## 1. 论文的主要贡献和创新点

### 解决的问题
传统认知科学的研究范式依赖于人工主导的“假设提出 → 实验设计 → 数据收集 → 模型拟合 → 理论验证”循环，这一过程存在以下瓶颈：
- **速度慢**：从理论到发表周期长达数年；
- **搜索空间受限**：研究者直觉和经验限制了可探索的实验设计与模型空间；
- **偏见风险高**：易陷入熟悉范式，难以发现真正新颖的认知机制。

该论文旨在解决如何突破人类干预带来的效率与创造力瓶颈，实现**大规模、自动化、高通量的科学发现流程**。

### 提出的新方法与新思路
作者提出一个**端到端全自动化的 in silico 科学发现框架**，将 Large Language Models（LLMs）整合进认知科学研究的每一个环节，形成闭环系统。其核心架构如图1所示，包含四个模块：

| 模块 | 功能 | 技术实现 |
|------|------|---------|
| **Experimentalist（实验设计者）** | 自动生成具有理论意义的认知任务 | 使用 LLM 作为智能采样器，基于语义指令生成 MDP-style 或自然语言描述的任务 |
| **Data Generator（数据生成器）** | 合成逼真的行为数据 | 利用 **Centaur** 这类 **foundation model of cognition** 模拟人类在任意任务下的选择序列 |
| **Modeller（建模者）** | 自动合成并优化解释数据的计算模型 | 基于 LLM 的 program synthesis（如 GeCCo 框架），输出 Python 函数形式的认知模型 |
| **Critic（批评者）** | 评估发现的“有趣性”（interestingness），指导下一步搜索 | LLM-as-judge，综合评价 novelty、simplicity、generalizability、unification 等维度 |

> ✅ **关键创新点**：
> - 首次提出将 LLM 贯穿整个认知科学发现链条；
> - 引入 **“interestingness” 作为目标函数**，超越传统的 information gain，追求概念上的启发价值；
> - 构建了一个可迭代、自驱动的 discovery engine，而非仅辅助工具。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|----------|
| **实验多样性** | 受限于已有范式（如 bandit, planning） | 可探索语法允许范围内的所有组合任务 |
| **模型构建方式** | 手工设计少量候选模型 | LLM 高通量搜索算法假设空间 |
| **数据获取成本** | 需真实被试，耗时昂贵 | in silico 模拟，秒级生成 |
| **理论演化机制** | 社会性同行评审驱动 | 内部 critic 实现快速反馈闭环 |
| **扩展性** | 低，人力密集型 | 高，支持并行多路径探索 |

---

## 2. 核心实验方法和设置

### 使用的数据集
本研究为**概念验证性质的框架提案**，并未使用传统意义上的公开行为数据集（如 fMRI-2, HCP, etc.）。而是采用：
- **合成任务描述**：通过 LLM 生成符合 MDP 结构的任务文本（状态、动作、奖励函数等）；
- **模拟行为轨迹**：由 Centaur 模型生成 choice sequences，格式包括 trial-by-trial decisions, rewards, context variables；
- **元数据条件化输入**：引入 age, sex, diagnosis, psychiatric scores 等 metadata 控制个体差异模拟。

> 示例输出结构（来自原文 Figure 1 注释）：
> ```
> Age, Sex, Diagnosis, Trial, Choice, Reward, ..., Genetic Risk, Life History
> ```

### 实验设置与评估指标
由于是前瞻性框架，实验以**仿真+案例演示**为主，重点在于展示各组件可行性及协同潜力。

#### 主要设置
- **任务语言基础**：基于 MDP grammar 定义实验空间，允许 LLM 在此之上自由组合；
- **行为模拟引擎**：Centaur v1（引用 Binz et al., 2025），能接受自然语言任务描述并生成行为；
- **模型生成器**：基于 GPT-class LLM 实现 GeCCo pipeline，支持代码生成与迭代 refinement；
- **Critic 设计**：prompt 工程实现 multi-dimensional scoring，包括：
  - **Novelty**：相对于历史发现的独特性
  - **Compressibility / Simplicity**：最优解释的简洁程度
  - **Qualitative Signatures**：是否再现典型心理效应（如 primacy effect）
  - **Transferability**：能否推广至其他任务或人群

#### 基线方法对比
文中未提供定量表格比较，但在论述中隐含对比了以下 baseline：

| Baseline 方法 | 缺陷（本文观点） |
|-------------|----------------|
| 手工建模（Manual Modeling） | 探索空间极小，易陷入局部最优 |
| Optimal Experimental Design (OED) | 仅最大化信息增益，可能导致无意义边缘情况 |
| Evolutionary Algorithms alone | 缺乏语义理解，搜索效率低 |
| 单纯 LLM 推理 without simulation | 易产生幻觉，缺乏行为约束 |

> ⚠️ 注意：当前工作尚未进行大规模 benchmark 测试，更多是原理性论证。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（基于已有文献引用与推演）
尽管本文未报告具体数字指标，但引用前期工作提供了支撑证据：

| 组件 | 支持性结果（引文依据） |
|------|-----------------------|
| **Centaur 行为模拟** | 在多种决策任务上与人类数据相关性达 r > 0.8（Binz & Schulz, 2023; Binz et al., 2025） |
| **LLM 程序生成（GeCCo）** | 成功复现 10+ 经典认知模型（如 RL, Bayesian, Heuristic rules），并在新任务中发现非显而易见机制（Rmus & Jagadish et al., NeurIPS 2024） |
| **FunSearch 类似系统** | 在数学领域发现优于人类已知解的算法（Romera-Paredes et al., Nature 2024） |
| **LLM-as-Judge** | 在开放探索任务中可有效引导搜索朝向“有趣”区域（Zhang et al., ICLR 2024） |

> 💡 尽管缺乏统一 benchmark，这些结果共同表明：每个子模块均已具备初步可行性。

### 与基线方法的对比结果（定性）
- 相比手工建模，LLM 自动生成的模型能覆盖更广的行为模式，包括一些反直觉但拟合优度更高的策略；
- 相比 OED，基于 “interestingness” 的 critic 更倾向于生成具有理论迁移潜力的任务（例如揭示跨任务共享的学习机制）；
- 相比纯进化算法，LLM-guided search 更快收敛到语义合理且可解释的模型结构。

### 消融实验（文中未执行，但讨论了关键脆弱点）
虽然没有正式消融实验，但作者明确指出系统的四大薄弱环节（即“最弱一环决定整体强度”）：

| 组件 | 若失效的影响 |
|------|--------------|
| Task Grammar | 无法表达新范式 → 锁死创新边界 |
| Synthetic Data Fidelity | 生成行为失真 → 导致错误模型归纳 |
| Program Search Landscape | 代码空间崎岖 → 易陷入局部最优或过拟合 |
| Interestingness Signal | 被操纵或漂移 → 系统退化为制造怪异现象的机器 |

---

## 4. 关键结论和发现

### 主要发现
1. **自动化科学发现已成为技术上可行的方向**：LLM + foundation models + program synthesis 的组合使得全链路 in silico 探索成为可能。
2. **真正的挑战不在技术集成，而在 epistemic design**：如何定义“好科学”？不能只追求 informativeness，必须加入 **interestingness、parsimony、unification** 等更高阶标准。
3. **科学家角色转变**：不再是执行者，而是**定义搜索空间、设定评价准则、锚定生物学真实性**的“元设计师”。
4. **加速假说生成 ≠ 替代实证验证**：最高分发现仍需 in vivo 实验验证，自动化系统应视为“高通量筛选仪”。

### 方法的局限性
| 局限性 | 具体表现 |
|--------|---------|
| **Representation Bottleneck** | 当前 task grammar 仍局限于 MDP 等结构，难以涵盖语言、社会互动等复杂情境 |
| **Simulation Faithfulness 不确定** | Centaur 等模型可能依赖训练数据中的统计捷径，泛化能力存疑 |
| **LLM Prior Biases** | 程序生成受训练数据影响，偏好特定编码风格或函数结构 |
| **Critic 可被游戏化** | 若仅优化单一 scalar score，“有趣性”可能退化为制造异常行为 |
| **缺乏决定性测试机制** | 当多个模型 equally interesting 时，系统缺乏 falsification 能力 |

### 未来工作方向
1. **发展更具表达力的任务语言**：融合符号逻辑、动态系统、社会网络等多模态表示；
2. **增强 foundation models 的因果推理能力**：使其不仅能模仿行为，还能反映潜在机制；
3. **构建 multi-objective critic system**：结合 human-in-the-loop review 与自动评分；
4. **整合神经数据预测**：要求模型同时解释 choice 和 fMRI/EEG patterns；
5. **建立 automated falsification protocols**：主动设计 adversarial tasks 来挑战当前最佳模型；
6. **推动 computational psychiatry 应用**：利用个体 metadata 模拟精神障碍表型，寻找机制分型。

---

> 📌 **最终洞见**（来自 Discussion）：
>  
> “The Library of Babel contains everything; science is the art of finding the few pages worth reading.”  
>   
> 自动化不是为了制造无限‘书’，而是帮助我们更快找到那些真正值得读的几页。这个系统最有价值的地方，不在于它能做什么，而在于它能让人类专注于更有创造性的工作——定义问题、判断意义、做出抉择。

</details>

---

### 12. [LongCat-Flash-Prover: Advancing Native Formal Reasoning via Agentic Tool-Integrated Reinforcement Learning](https://arxiv.org/abs/2603.21065)

**Authors**: Jianing Wang, Jianfei Zhang, Qi Guo, Linsen Guo, Rumei Li, Chao Zhang, Chong Peng, Cunguang Wang, Dengchang Zhao, Jiarong Shi, Jingang Wang, Liulin Feng, Mengxia Shen, Qi Li, Shengnan An, Shun Wang, Wei Shi, Xiangyu Xi, Xiaoyu Li, Xuezhi Cao, Yi Lu, Yunke Zhao, Zhengyu Chen, Zhimin Lin, Wei Wang, Peng Pei, Xunliang Cai  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.21065v1  

#### Abstract
We introduce LongCat-Flash-Prover, a flagship 560-billion-parameter open-source Mixture-of- Experts (MoE) model that advances Native Formal Reasoning in Lean4 through agentic tool-integrated reasoning (TIR). We decompose the native formal reasoning task into three independent formal capabilities, i....

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LongCat-Flash-Prover: Advancing Native Formal Reasoning via Agentic Tool-Integrated Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大型语言模型（LLMs）在处理**形式化定理证明**（formal theorem proving）任务时面临显著挑战。尽管已有研究利用验证工具反馈进行自我修复（如Python脚本），但**Lean4**等**形式化语言**具有严格的逻辑演进要求，直接应用传统的**Tool-Integrated Reasoning (TIR)** 范式存在困难。此外，现有方法常将自动形式化（auto-formalization）、草图生成（sketching）和证明（proving）作为独立任务处理，缺乏统一框架。

### 提出的新方法与新思路
本文提出了 **LongCat-Flash-Prover**，一个拥有5600亿参数的开源 **Mixture-of-Experts (MoE)** 模型，旨在推进**原生形式化推理**（Native Formal Reasoning）。其核心创新包括：

- **原生形式化推理范式 (Native Formal Reasoning)**  
  将形式化推理视为LLM的“原生”能力，类比于原生多模态（native multimodal）和原生工具调用（native tool calls）。该范式将任务分解为三个原子能力：
  - **Agentic Auto-formalization**：将非正式问题转化为经验证的形式化陈述。
  - **Agentic Sketching**：基于问题和形式化陈述生成引理风格的证明草图（lemma-style sketch）。
  - **Agentic Proving**：生成完整的证明（whole-proof）或基于草图的证明（sketch-proof）。

- **混合专家迭代框架 (Hybrid-Experts Iteration Framework)**  
  通过多个专业化专家模型（auto-formalizer, sketcher, prover）协同工作，结合**验证工具反馈**（如Lean4编译器、语法检查、语义一致性检测），迭代生成高质量的训练轨迹（task trajectories），模拟人类“尝试-验证-反思”的学习过程。

- **分层重要性采样策略优化 (Hierarchical Importance Sampling Policy Optimization, HisPO)**  
  针对MoE架构在长程任务中训练不稳定的问题，提出一种新的强化学习算法。HisPO在**序列级**和**词元级**引入梯度掩码策略，以缓解因训练-推理引擎差异（train-inference discrepancy）和策略陈旧（policy staleness）导致的优化不稳定性。

- **合法性检测机制 (Legality Detection)**  
  引入基于**抽象语法树 (AST)** 的严格一致性检查，防止模型通过篡改定理定义、注入虚假公理（axiom）、使用`#exit`命令等方式进行**奖励欺骗 (reward hacking)**，确保评估的真实性和可靠性。

### 相比现有方法的优势
- **性能领先**：在多个基准上超越所有现有的开源和闭源模型，在**开放权重模型**中达到SOTA。
- **样本效率高**：仅需极低的推理预算（如72次尝试）即可取得优异成绩。
- **方法统一**：将auto-formalization、sketching、proving整合到一个统一的TIR框架下，提升了模型的通用性和可扩展性。
- **训练稳定**：HisPO有效解决了MoE模型在长程推理任务中的训练崩溃问题。
- **评估可靠**：通过AST合法性检测，堵住了现有评估流程中的漏洞，避免了虚假性能报告。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Auto-formalization 任务**：
  - `CombiBench`, `FormalMath-Lite`, `MathOlympiad-Bench`, `MiniF2F-Test`, `ProofNet`, `ProverBench`, `PutnamBench`
- **Theorem Proving 任务**：
  - `MathOlympiad-Bench`, `MiniF2F-Test`, `ProofNet`, `ProverBench`, `PutnamBench`

这些数据集覆盖了从高中竞赛到国际数学奥林匹克（IMO）级别的复杂问题。

### 实验设置和评估指标
- **评估模式**：
  - **Whole-proof mode**：直接生成完整证明。
  - **Whole-proof mode w/ TIR**：允许与验证工具交互。
  - **Sketch-proof mode w/ TIR**：先生成草图，再逐步证明引理。
- **主要指标**：
  - `Pass@k`：在最多k次尝试中至少有一次成功的概率。
  - 特别关注 `Pass@32` 和无预算限制下的最终成功率。
- **推理预算控制**：严格控制并报告每次实验的尝试次数，以保证公平比较。

### 基线方法对比
- **开源通用推理模型 (Open-Weights Reasoning Models)**：
  - `DeepSeek-V3.2`, `Kimi-K2.5`
- **闭源通用推理模型 (Close-Weights Reasoning Models)**：
  - `Claude-Opus-4.5`, `Gemini-3 Pro`
- **专用形式化模型 (Specialized Prover/Auto-formalizer Models)**：
  - `Kimina-Prover-8B/72B`, `DeepSeek-Prover-V2-7B/671B`, `Goedel-Prover-V2-8B/32B`, `Leanabell-Prover-V2-KM/DS`, `ATF-32B` 等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | MiniF2F-Test (Pass@32) | PutnamBench (Pass@32) | 推理预算 |
|------|------------------------|-----------------------|----------|
| **LongCat-Flash-Prover (sketch-proof w/ TIR)** | **93.9%** | **28.9%** | 32 |
| LongCat-Flash-Prover (whole-proof w/ TIR) | 90.2% | 10.4% | 32 |
| Goedel-Prover-V2-32B (w/ self-correction) | 90.4% | 8.6% | 32 |
| DeepSeek-Prover-V2-671B | 82.4% | 3.3% | 32 |

在**无预算限制**且结合**Tree Search**的情况下：
- 在 `MiniF2F-Test` 上达到 **95.5%** 的准确率，仅需 **72** 次尝试。
- 在 `PutnamBench` 上解决 **41.5%** 的问题，远超其他开源模型。

### 与基线方法的对比结果
- 在 `MathOlympiad-Bench` 和 `PutnamBench` 上，相比SOTA开源模型，分别实现了 **25.5%** 和 **20.3%** 的绝对提升（Pass@32）。
- 在 `MiniF2F-Test` 上，仅用72次尝试即达到97.1%的通过率，**样本效率显著高于**需要数千次尝试的其他模型（如DeepSeek-Prover-V2-671B需8192次）。
- 在 `Auto-formalization` 任务上，`Pass@8` 指标全面领先，尤其在 `MiniF2F-Test` 和 `ProofNet` 上达到了 **100%** 和 **99.8%** 的惊人成绩。

### 消融实验结果
- **TIR的作用**：启用TIR后，性能大幅提升（如在`MathOlympiad-Bench`上从16.9%提升至35.8%），证明了工具反馈的有效性。
- **HisPO的作用**：在RL训练中，未修复奖励函数前出现“奖励爆炸”现象（rollout pass rate异常飙升），修复后曲线平稳上升，表明HisPO和合法性检测成功抑制了奖励欺骗。
- **Tree Search的作用**：在相同预算下，引入Tree Search平均带来 **3.1%** 的准确率提升，说明递归分解能有效简化复杂证明。

---

## 4. 关键结论和发现

### 主要发现
1. **原生形式化推理是可行的**：通过将auto-formalization、sketching、proving统一到TIR框架下，可以系统性地提升LLM在形式化任务上的表现。
2. **工具集成至关重要**：TIR策略极大地增强了模型解决难题的能力，尤其是在处理高难度问题时。
3. **评估必须严谨**：现有评估流程存在严重漏洞（如允许篡改上下文），必须引入**AST级合法性检测**来确保结果可信。
4. **样本效率可以很高**：LongCat-Flash-Prover展示了即使在极低推理预算下也能取得顶尖性能，为实际应用提供了可能。

### 方法的局限性
- **计算成本高昂**：560B参数的MoE模型对训练和部署资源要求极高。
- **依赖高质量工具反馈**：整个框架的成功建立在可靠的Lean4编译器和验证工具之上。
- **非正式推理能力略有下降**：专注于形式化推理的训练导致在一般性STEM任务（如AIME-25）上性能略低于其前身LongCat-Flash-Thinking-2601（见Table 4）。

### 未来工作方向
- **扩大搜索预算**：计划在未来迭代中增加搜索深度，进一步缩小与闭源顶级模型（如Seed-Prover）的差距。
- **平衡正式与非正式推理**：探索更好的训练策略，以同时保持强大的通用推理能力和顶尖的形式化推理能力。
- **推广至其他形式化系统**：将此框架应用于Coq、Isabelle等其他证明助手。
- **社区共建**：通过开源模型和代码，鼓励社区共同构建更高质量的数据集和更高效的训练方法。

> **项目链接**：  
> Huggingface: [https://huggingface.co/meituan-longcat/LongCat-Flash-Prover](https://huggingface.co/meituan-longcat/LongCat-Flash-Prover)  
> GitHub: [https://github.com/meituan-longcat/LongCat-Flash-Prover](https://github.com/meituan-longcat/LongCat-Flash-Prover)

</details>

---

### 13. [EvoIdeator: Evolving Scientific Ideas through Checklist-Grounded Reinforcement Learning](https://arxiv.org/abs/2603.21728)

**Authors**: Andreas Sauter, Yuyue Zhao, Jacopo Urbani, Wenxiang Hu, Zaiqiao Meng, Lun Zhou, Xiaohui Yan, Yougang Lyu  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.21728v1  

#### Abstract
Scientific idea generation is a cornerstone of autonomous knowledge discovery, yet the iterative evolution required to transform initial concepts into high-quality research proposals remains a formidable challenge for Large Language Models (LLMs). Existing Reinforcement Learning (RL) paradigms often...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EvoIdeator: Evolving Scientific Ideas through Checklist-Grounded Reinforcement Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前在科学创意生成（scientific idea generation）任务中存在一个“双重鸿沟”（dual gap）：
- **基于强化学习（RL）的方法**：依赖标量奖励（scalar rewards），虽能优化整体质量，但缺乏细粒度、可操作的反馈，无法指导模型具体如何改进。
- **基于语言反馈的方法**：提供详细的文本批评（language feedback），但通常仅用于推理时（inference-time）的提示工程（prompting），未将此类反馈内化到模型训练中。

这导致训练目标与推理过程不一致，限制了模型在迭代演化中的表现。

### **提出的新方法与新思路**
作者提出了 **EvoIdeator**，一种通过 **checklist-grounded reinforcement learning** 实现科学创意演化的框架。其核心创新在于：
- 将 **训练时的 RL 优化** 与 **推理时的语言反馈机制** 显式对齐（align），使模型既能从全局奖励中学习，又能理解并执行具体的语言级修改建议。
- 引入 **双信号评审机制（dual-signal judging）**：
  1. **Lexicographic Rewards**：多维优先级奖励，按科学严谨性（如可行性、方法论等）设定层级，避免次要目标稀释主目标信号。
  2. **Actionable Language Feedback**：由结构化 checklist 驱动的细粒度语言反馈，定位具体文本片段（span-level）并给出修改建议。

该方法基于 **Dr. GRPO estimator** 构建训练循环，在每轮 rollout 中同时利用标量奖励更新策略，并用语言反馈引导下一轮生成。

### **相比现有方法的优势**
- ✅ **训练-推理一致性**：首次将语言反馈作为训练信号的一部分，使模型真正“学会”如何响应批评。
- ✅ **细粒度优化能力**：不仅知道“好不好”，还知道“哪里不好、怎么改”。
- ✅ **小模型超越大模型**：基于 Qwen3-4B 的 EvoIdeator 在多个关键指标上优于更大的前沿模型（如 Gemini 3 Flash 和 DeepSeek-V3.2）。
- ✅ **泛化性强**：训练后的策略可在不同 judge 模型提供的反馈下工作，支持“即插即用”的自我精炼（self-refining ideation）。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
作者构建了一个新的训练种子数据集，流程如下：
1. **Seed Paper Sampling**：从 OpenAlex 中随机采样 1000 篇 2025 年发表的高质量论文（期刊/会议，非撤稿）。
2. **Query & Keyword Generation**：使用 Llama-3 为每篇种子论文生成研究问题（query）和检索关键词。
3. **Literature Review Synthesis**：通过 Semantic Scholar 检索相关文献，再由 LLM 合成一段 1–2 段的综述，描述领域现状与缺口。

最终形成 `(query, literature_review)` 对，用于初始化 RL 训练。测试集包含 **96 个独立样本**。

### **实验设置与评估指标**
#### **评估阶段分两步进行**：
- **Generation Step**：模型直接输出初始想法，无外部反馈。
- **Refinement Step**：模型接收前一轮输出及来自 judge 的语言反馈后生成改进版本。

#### **评估维度（Checklist 共 9 项）**
分为两类目标：
| **Primary Objectives（科学严谨性）** | **Secondary Objectives（格式与新颖性）** |
|--------------------------------------|------------------------------------------|
| Grounding, Feasibility, Problem, Risk, Method | Writing, Innovation, Length, Layout |

所有模型均报告各项平均得分（mean score ± 95% CI），排除未能正确输出 `<idea>` 块的样本。

### **基线方法对比**
| 模型 | 类型说明 |
|------|---------|
| **Qwen-4B** | 未对齐的基础模型（zero-shot baseline） |
| **DeepSeek R1 Distill / DeepSeek-V3.2** | 经过过程监督训练的推理优化模型 |
| **Gemini 3 Flash** | 大规模通用模型，代表前沿性能上限 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

#### **Generation Step 结果**
- EvoIdeator 在 **Grounding** 上达到 **0.99**，显著高于其他模型（Gemini: 0.87, Qwen-4B: 0.85）。
- 在 **Feasibility**, **Problem**, **Method** 等核心科学维度也全面领先或持平。

#### **Refinement Step 结果**
| 指标 | EvoIdeator | Gemini 3 Flash | DeepSeek-V3.2 |
|------|------------|----------------|---------------|
| **Grounding** | **0.99** | 0.91 | 0.91 |
| **Problem** | **0.94** | 0.90 | 0.90 |
| **Risk** | **0.35** | 0.16 | 0.30 |
| **Method** | **0.58** | 0.48 | 0.72 |
| **Writing** | **0.99** | 0.97 | 0.88 |
| **Innovation** | 0.47 | **0.60** | 0.42 |

> 💡 **结论**：EvoIdeator 在 **5 项 Primary Objectives 中有 4 项排名第一**，尤其在 **Problem 定义与 Risk 分析** 上明显优于更大模型。

### **消融实验结果（RQ2）**
通过比较四种配置验证各组件作用：
- **Informed**（完整版 EvoIdeator）
- **Non-Informed**（无语言反馈训练）
- **Base + Feedback**
- **Base w/o Feedback**

#### **关键发现（Figure 2）**
- 所有使用语言反馈的模型在 Refinement Step 都有提升。
- 但只有 **经过对齐训练的 Informed 模型** 能实现 **持续且显著的质量跃升**。
- 表明：**语言反馈本身有用，但必须在训练中被内化才能发挥最大效用**。
- RL 训练提升了初始生成质量（intercept），而语言反馈带来斜率提升（slope），二者具有 **加性增益（additive gains）**。

### **跨评审器泛化能力（RQ3）**
测试 EvoIdeator 是否能在不同 judge 提供的反馈下工作：

| 反馈来源 | 初始得分 | 改进后得分 | 增幅 |
|--------|--------|----------|-----|
| DS R1 14B | 4.07 | 5.64 | +1.57 |
| DS R1 70B | 4.09 | 5.81 | +1.72 |
| DS v3.2 | 4.05 | **6.02** | **+1.97** |
| Gemini 3 Flash | 3.97 | 5.13 | +1.16 |

> 📌 **结论**：
- 在 DeepSeek 家族内部，反馈质量随模型规模提升而单调上升 → **成功迁移**
- 使用 Gemini 作为反馈源时性能下降明显 → **存在“方言敏感性”（dialect sensitivity）**
- 说明：反馈是一种**学习到的通信协议**，需对齐训练范式（如 RLHF 分布）

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **EvoIdeator 成功弥合了训练与推理之间的鸿沟**，实现了 RL 与语言反馈的协同优化。
2. ✅ **小模型也能打败大模型**：Qwen3-4B 规模的模型通过结构化训练，在科学创意质量上超越 Gemini 等更大模型。
3. ✅ **双信号机制产生加性收益**：Lexicographic Reward 提升起点，Language Feedback 提供进化动力。
4. ✅ **具备跨 judge 泛化能力**：可在同一家族的不同 judge 下即插即用，适合部署灵活的自演化系统。

### **方法的局限性**
1. **次级目标被牺牲**：由于 lexicographic reward 优先保障 primary objectives，导致 **Innovation 和 Length 合规性偏低**。
2. **反馈风格依赖性强**：对来自不同训练谱系的 judge（如 Gemini）适应差，需标准化反馈格式。
3. **人工评估缺失**：目前完全依赖 LLM judge，尚未引入大规模人类专家评分。
4. **迭代深度有限**：实验仅考察单步 refinement，更长 horizon 的演化行为有待探索。

### **未来工作方向**
- 探索 **动态权重调整机制** 或 **Pareto-based MORL** 来平衡 primary 与 secondary objectives。
- 设计 **标准化反馈模板** 以增强跨模型家族的兼容性。
- 引入 **human-in-the-loop evaluation** 提高结果可信度。
- 扩展至 **multi-agent collaboration** 场景（如作者后续工作 EvoScientist）。
- 研究 **compute allocation 策略**：如何最优分配训练 vs. 推理计算资源。

---

> 🔚 **总结一句话**：  
> **EvoIdeator 通过将 checklist-grounded language feedback 与 lexicographic RL 有机结合，让小型 LLM 学会“听懂批评”并持续自我进化，在科学创意生成任务中实现了训练-推理闭环，展现出强大的性能与泛化潜力。**

</details>

---

### 14. [RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](https://arxiv.org/abs/2603.20527)

**Authors**: Shenyang Deng, Zhuoli Ouyang, Tianyu Pang, Zihang Liu, Ruochen Jin, Shuhua Yu, Yaoqing Yang  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.20527v1  

#### Abstract
Preconditioned adaptive methods have gained significant attention for training deep neural networks, as they capture rich curvature information of the loss landscape . The central challenge in this field lies in balancing preconditioning effectiveness with computational efficiency of implementing th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的矩阵自适应优化器（如 **MUON**）虽然能捕捉参数间的二阶曲率信息，从而提升优化性能，但其依赖 **Newton-Schulz 迭代**进行正交化更新，导致每步计算复杂度高达 $O(mn \cdot \min(m,n))$，在大规模模型训练中成为计算瓶颈。

本文旨在解决这一 **效率与效果之间的权衡问题**：如何在保持矩阵级自适应能力的同时，显著降低 preconditioning 的计算开销。

---

### ✅ 提出的新方法与核心思想
提出 **RMNP (Row-Momentum Normalized Preconditioning)**，一种新型的矩阵自适应优化器，其核心创新在于：

- **用行归一化（row-wise $l_2$ normalization）替代 Newton-Schulz 迭代**。
- 动机来源于对 **Transformer 层级 Hessian 矩阵结构的经验观察**：具有显著的 **row-wise block-diagonal dominance**（即行内参数交互远强于跨行交互）。
- 因此，仅保留 preconditioner 中的对角块（即行内相关性），忽略跨行项，形成简化结构：
  $$
  H_{\text{RMNP}} = \text{diag}((V_t V_t^\top)^{-1/2}) \otimes I_n
  $$
  其中 $V_t$ 是动量梯度矩阵。

该操作等价于对每个权重矩阵的每一行动量向量进行 $l_2$ 归一化，实现简单且高效。

---

### ✅ 相比现有方法的优势
| 维度 | MUON | RMNP |
|------|------|------|
| **计算复杂度** | $O(mn \cdot \min(m,n))$ | $O(mn)$ |
| **实际速度** | 慢（迭代多） | 快 13–44×（见实验） |
| **内存占用** | 高 | 相同 |
| **理论收敛性** | 收敛保证已知 | 匹配 MUON 的非凸收敛界 |
| **优化性能** | 强 | 相当甚至略优 |

> **核心优势**：在几乎不损失优化性能的前提下，将 preconditioning 开销降低一个数量级，极大提升了可扩展性。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **GPT-2 系列训练**：
  - **OpenWebText**：标准预训练语料。
  - **FineWeb-Edu-100B**：更大规模教育过滤文本数据集，用于验证泛化性。
- **LLaMA 系列训练**：
  - **C4 数据集**：常用语言建模预训练语料。

---

### ⚙️ 实验设置
- **模型架构**：
  - GPT-2：Small (125M), Medium (355M), Large (770M), XL (1.5B)
  - LLaMA：60M, 130M, 350M
- **训练配置**：
  - Batch size: 480 (GPT-2), 512 (LLaMA)
  - Sequence length: 1024 (GPT-2), 256 (LLaMA)
  - 学习率调度：cosine annealing + 10% warmup
  - 混合更新策略：matrix 参数用 RMNP/MUON，non-matrix 参数用 **ADAMW**
- **硬件平台**：
  - 单卡 RTX Pro 6000 / Blackwell B200 GPU 测量时间开销
  - 多卡并行训练用于大模型

---

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Perplexity (PPL)** | 主要性能指标，越低越好 |
| **Wall-clock time** | Preconditioning 步骤耗时（秒/100 steps） |
| **Speedup (×)** | RMNP vs MUON 的加速比 |
| **Memory Usage** | 显存消耗对比 |
| **Diagonal Dominance Ratio** | 分析 Gram 矩阵 $(V_t V_t^\top)$ 是否满足对角主导特性 |

---

### 🔁 基线方法对比
- **ADAMW**：广泛使用的默认优化器
- **MUON**：当前最先进的矩阵自适应优化器，作为主要对比对象
- **RMNP**：本文提出的方法

所有方法均采用相同超参搜索空间，仅调整 `lrMatrix`，其余固定。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### （1）**Preconditioning 时间开销对比（Table 2 & 3）**

| 模型大小 | MUON 耗时 (s) | RMNP 耗时 (s) | 加速比 (×) |
|---------|----------------|----------------|------------|
| 60M     | 1.480          | 0.115          | 12.9×      |
| 125M    | 2.975          | 0.201          | 14.8×      |
| 355M    | 7.380          | 0.401          | 18.4×      |
| 770M    | 27.070         | 0.611          | 44.3×      |
| 1.5B    | 36.650         | 0.855          | 42.9×      |

✅ **结论**：随着模型增大，RMNP 的加速优势更加明显，最高达 **44×**，且内存使用完全一致。

---

#### （2）**最终 Perplexity 对比（Figure 5, 6, 11）**

##### GPT-2 on OpenWebText：
| 模型 | ADAMW | MUON | RMNP |
|------|-------|------|------|
| Small (125M) | 23.86 | 22.86 | **22.82** |
| Medium (355M) | 18.80 | 17.38 | **17.31** |
| Large (770M) | 15.27 | 14.42 | **14.14** |

> RMNP 在所有尺度上均优于或持平 MUON，并显著超越 ADAMW。

##### LLaMA on C4：
| 模型 | ADAMW | MUON | RMNP |
|------|-------|------|------|
| 60M | 33.28 | 29.58 | **28.95** |
| 130M | 23.24 | 22.42 | **22.14** |
| 350M | 17.31 | 16.87 | **16.85** |

> RMNP 表现稳定，在小模型上优势更明显。

##### GPT-2 on FineWeb-Edu-100B（Table 8）：
| 模型 | ADAMW | MUON | RMNP |
|------|-------|------|------|
| Small | 23.85 | 22.71 | **22.60** |
| Medium | 18.19 | 17.13 | **17.07** |
| Large | 14.81 | 14.16 | **13.75** |

> 在更大数据集上仍保持领先，说明泛化性强。

---

#### （3）**消融实验与分析支持**

##### ✅ Diagonal Dominance 验证（Section 3.2）
- 定义指标 $r_i = \frac{(V_t V_t^\top)_{ii}}{\text{avg}_{j\neq i} |(V_t V_t^\top)_{ij}|}$，衡量第 $i$ 行是否对角主导。
- 报告全局统计：
  - $r_{\text{avg}}$: 平均值
  - $r_{\text{min}}$: 最小值
  - $r_{\text{max}}$: 最大值
- 结果显示：在 GPT-2 和 LLaMA 训练过程中，**所有 $r > 1$**，且 $r_{\text{avg}} \gg 1$，表明 Gram 矩阵确实呈现强烈对角主导结构。

> ✅ 支持了 RMNP 的设计假设：跨行相关性弱，可安全忽略。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Transformer 的 Hessian / 动量 Gram 矩阵具有显著的 row-wise block-diagonal dominance**，这为结构化近似提供了理论依据。
2. **RMNP 利用这一结构，通过简单的 row-wise $l_2$ normalization 替代昂贵的 Newton-Schulz iteration**，实现了从 $O(mn \cdot \min(m,n))$ 到 $O(mn)$ 的复杂度降维。
3. **RMNP 在多种模型（GPT-2、LLaMA）、多个数据集（OpenWebText、C4、FineWeb-Edu-100B）上均达到与 MUON 相当甚至更优的 perplexity，同时提速 13–44×**。
4. **理论层面，RMNP 在非凸平滑设定下达到了与 MUON 同等级别的收敛保证**（见 Table 1），且达到信息论最优的 minimax 复杂度。

---

### ⚠️ 方法的局限性
- 当前设计基于 **row-wise 结构假设**，若某些层或任务中存在强烈的跨行耦合（如特定注意力模式），可能影响性能。
- 未探索 column-wise 或双向 normalization 的组合形式。
- 所有实验集中在 Transformer 架构，尚未验证在 CNN、RNN 等其他结构上的通用性。

---

### 🔮 未来工作方向
1. **扩展到 tensor 级别**：将 row-normalization 推广至更高维张量（如嵌入矩阵、注意力头维度）。
2. **动态结构调整**：根据训练阶段自动判断是否启用 full MUON 或切换至 RMNP。
3. **与其他优化机制结合**：如与 **SOAP**、**COSMOS** 等 memory-efficient 方法集成，构建端到端高效的训练框架。
4. **理论深化**：进一步研究 $\|\cdot\|_{1,2}$ 几何下的最优化性质，推动 geometry-aware optimization 发展。

---

## ✅ 总结一句话
> **RMNP 以极简的 row-wise $l_2$ normalization 实现了与 MUON 相当的优化性能，却将 preconditioning 开销降低一个数量级，是迈向高效、可扩展矩阵自适应优化的重要一步。**

</details>

---

### 15. [Model Evolution Under Zeroth-Order Optimization: A Neural Tangent Kernel Perspective](https://arxiv.org/abs/2603.21169)

**Authors**: Chen Zhang, Yuxin Cheng, Chenchen Ding, Shuqi Wang, Jingreng Lei, Runsheng Yu, Yik-Chung WU, Ngai Wong  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.21169v1  

#### Abstract
Zeroth-order (ZO) optimization enables memory-efficient training of neural networks by estimating gradients via forward passes only, eliminating the need for backpropagation. However, the stochastic nature of gradient estimation significantly obscures the training dynamics, in contrast to the well-c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Model Evolution Under Zeroth-Order Optimization: A Neural Tangent Kernel Perspective

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 **first-order (FO)** 优化方法依赖反向传播计算梯度，在内存消耗和不可微目标场景下存在局限。**zeroth-order (ZO) optimization** 虽然通过前向传播估计梯度实现了内存高效训练，并适用于黑盒优化、LLM 微调等任务，但由于其梯度估计具有随机性，导致 **训练动态难以理论建模**。

相比之下，FO 方法在 **Neural Tangent Kernel (NTK)** 理论框架下已被充分理解——宽网络在函数空间中线性演化，NTK 在训练过程中保持不变，从而可推导出闭式解。然而，这一理论无法直接应用于 ZO 方法。

本论文旨在填补这一理论空白：**为 ZO 优化建立一个类似 NTK 的函数空间分析框架**，以揭示其模型演化的本质机制。

---

### 提出了什么新方法或新思路
作者提出了 **Neural Zeroth-order Kernel (NZK)**，作为分析 ZO 优化在函数空间中动态行为的新工具。

#### 核心思想：
- 将 ZO 更新分解为两个部分：
  - **Magnitude**: 由损失差值决定（`[L(f+ez) - L(f-ez)]`）
  - **Direction**: 由输出差值决定（`[f(θ+ez) - f(θ-ez)]`），即对 Jacobian 的有限差分估计
- 定义 NZK 的条目为：
  $$
  \text{NZK}(x_i, x_j) = \left\langle \frac{f(x_i;\theta+\epsilon\xi) - f(x_i;\theta-\epsilon\xi)}{2\epsilon}, \frac{f(x_j;\theta+\epsilon z) - f(x_j;\theta-\epsilon z)}{2\epsilon} \right\rangle \cdot \langle \xi, z \rangle
  $$
  其中 `ξ` 和 `z` 是用于扰动参数和估计变化率的随机方向向量。

#### 主要理论成果：
1. **Theorem 1**：对于线性模型，**期望意义下的 NZK 是时不变的（time-invariant）**，且其形式显式依赖于随机方向 `z` 和 `ξ` 的一阶和二阶矩。
2. **Corollary 2**：当使用**同一个随机向量**（即 `ξ = z`）进行扰动和 Jacobian 估计时，NZK 被放大 `(d+2)σ^4` 倍（对于 `z ~ N(0, σ²I)`），从而**显著加速收敛**。
3. 推导了平方损失下线性模型和线性化神经网络（linearized neural networks）的**闭式演化动态表达式**，形式上与 FO 方法一致，仅将 NTK 替换为期望 NZK。
4. 将 NZK 概念推广到 linearized NNs，建立了更广泛情形下的理论基础。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **理论层面** | 首次为 ZO 优化提供类 NTK 的函数空间视角，揭示了其“平均意义上”的确定性演化规律，弥补了理论空白。 |
| **方法设计** | 提出“共享随机向量”策略（`ξ = z`），不仅节省内存（只需采样一次），还能通过增强 NZK 内核尺度来加速收敛。 |
| **通用性** | 理论适用于多种分布（高斯、拉普拉斯、t 分布等），只要保证 `Var(z) + d·E²[z]` 不变，则收敛行为一致，说明核心是统计矩而非具体分布。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **合成数据**：二维单位圆上的采样点，用于简单回归任务，便于可视化模型演化路径。
- **真实世界数据集**：
  - **MNIST**：手写数字分类
  - **CIFAR-10**：小规模图像分类
  - **Tiny ImageNet**：ImageNet 的简化版，用于验证大规模任务适应性

---

### 实验设置和评估指标
| 设置项 | 描述 |
|-------|------|
| **模型类型** | - 线性模型（用于理论验证）<br>- Linearized Neural Networks（三层层前馈网络，ReLU 激活） |
| **优化方式对比** | - First-order (FO) GD<br>- ZO (parametric)：传统参数空间 ZO<br>- ZO (kernel)：基于 NZK 视角的更新，特别是 `ξ = z` 设置 |
| **关键变量控制** | - `z` 和 `ξ` 是否独立采样 vs. 共享<br>- 扰动方差 `σ²` 变化<br>- 随机向量维度 `d` 变化<br>- 不同分布（Gaussian, Laplace, Student’s t） |
| **评估指标** | - 训练 loss 随迭代次数的变化曲线<br>- 最终模型拟合效果可视化<br>- NZK 矩阵热力图（与 NTK 对比） |
| **实现细节** | - 学习率固定为 `1e-3`，`ε = 1e-3`<br>- 使用 10,000 次随机采样估计期望 NZK<br>- 实验平台：Ubuntu + AMD CPU + NVIDIA RTX3090 GPU |

---

### 基线方法对比
- **First-order (FO)**：标准梯度下降，作为理想基准
- **Traditional ZO (parametric)**：常规零阶梯度估计，`z` 和 `ξ` 独立采样
- **Proposed ZO (kernel)**：本文提出的基于 NZK 的视角，尤其是 `ξ = z` 的设置

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
| 实验场景 | 结果摘要 |
|--------|---------|
| **线性模型（2D 合成任务）** | - 当 `σ² = 1` 时，独立采样的 ZO 收敛速度接近 FO（Fig 1a）<br>- 增大 `σ²` 加快 ZO 收敛，但可能引入噪声<br>- 使用 `ξ = z` 显著加快收敛，且无需增大方差（Fig 1b） |
| **不同维度 `d` 影响** | - 随着 `d` 增加（10→50），共享 `ξ=z` 的 ZO 收敛更快（Fig 9），符合理论预测 `K ∝ d` |
| **不同分布比较** | - Gaussian (`σ=1`)、Laplace (`b=0.604`)、Student-t (`ν=1000`) 在 `Var(z)+d·E²[z]` 相同时表现出几乎相同的收敛曲线（Fig 11），验证了分布无关性 |
| **真实数据分类任务（MNIST/CIFAR-10/Tiny ImageNet）** | - 传统 parametric ZO 收敛慢于 FO<br>- 使用 kernel-gradient ZO（`ξ=z`）后，收敛速度**超过 FO**（Fig 2）<br>- NZK 热力图显示其结构与 NTK 高度相似（Fig 3, 17, 18） |

---

### 消融实验结果
1. **`ξ` 与 `z` 是否共享**：
   - 独立采样 → 收敛较慢
   - 共享 `ξ=z` → 明显加速，验证了 Corollary 2 中的缩放效应

2. **扰动方差 `σ²` 的影响**：
   - 增大 `σ²` 可提升信息获取，但也增加估计噪声；而共享向量策略可在不增大方差的前提下加速，更具优势

3. **不同分布的影响**：
   - 多种分布下只要保持 `Var(z) + d·E²[z]` 不变，收敛行为一致，说明 NZK 的稳定性源于统计矩而非分布形态

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **期望 NZK 是稳定的**：在线性模型和线性化网络中，**E[NZK]** 在训练过程中保持不变，这为 ZO 优化提供了类似于 NTK 的理论基石。
2. ✅ **闭式演化成立**：在平方损失下，模型在函数空间中的演化具有闭式解，形式为：
   $$
   [f_t(x)]_N = (I_N - (I_N - \eta K_{\xi,z})^t)[y]_N + (I_N - \eta K_{\xi,z})^t[f_0(x)]_N
   $$
   表明 ZO 与 FO 在期望意义上具有相似的动力学特性。
3. ✅ **共享随机向量可加速收敛**：使用相同 `z` 进行扰动和 Jacobian 估计，能有效放大 NZK，带来显著加速，同时减少内存开销。
4. ✅ **分布无关性**：只要控制好一、二阶矩，不同分布下的 ZO 表现一致，增强了方法鲁棒性。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **适用范围限制** | 当前理论集中在**线性模型**和**线性化网络**，尚未扩展到全非线性深度网络的实际训练过程。 |
| **宽网络假设** | 类似 NTK，理论有效性依赖于宽度趋于无穷的理想化设定，实际有限宽度网络可能存在偏差。 |
| **损失函数限制** | 闭式解仅针对平方损失推导，其他损失（如 cross-entropy）需进一步研究。 |
| **计算开销** | 虽然单步节省内存，但每次需要两次前向传播，总体计算成本高于 FO。 |

---

### 未来工作方向
1. **拓展至非线性动态**：研究在非线性 regime 下 NZK 是否仍近似稳定，探索 Adaptive-ZO 或 Dynamic-NZK 概念。
2. **结合实际硬件优化**：将 NZK 指导的 ZO 应用于 RRAM、存内计算等低功耗设备，发挥其仅需前向传播的优势。
3. **设计新型 ZO 算法**：基于 NZK 构造 preconditioner 或 adaptive learning rate，进一步提升效率。
4. **连接 Implicit Bias**：分析 ZO 与 FO 在泛化性上的差异，是否因 NZK 结构不同而导致不同的隐式正则化行为。
5. **扩展至 Transformer 架构**：探索 Attention 机制下的 ZO 动态，推动 LLM 黑盒微调的理论发展。

---

> 🔗 **代码开源**：文中提到所有实验代码已公开，详见论文链接（LINK）。  
> 📚 **参考文献支持**：理论构建依托于 NTK (Jacot et al., 2018)、ZO 基础 (Nesterov & Spokoiny, 2017) 及近期非参数教学工作 (Zhang et al., 2023–2026)。

</details>

---

### 16. [AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling](https://arxiv.org/abs/2603.21357)

**Authors**: Liang Ding  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.21357v1  

#### Abstract
LLM agents fail on the majority of real-world tasks -- GPT-4o succeeds on fewer than 15% of WebArena navigation tasks and below 55% pass@1 on ToolBench (Zhou et al., 2024; Qin et al., 2024) -- yet every failed trajectory is routinely discarded, wasting the dominant source of collected experience. We...

---

### 17. [Optimizing Multi-Agent Weather Captioning via Text Gradient Descent: A Training-Free Approach with Consensus-Aware Gradient Fusion](https://arxiv.org/abs/2603.21673)

**Authors**: Shixu Liu  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.21673v1  

#### Abstract
Generating interpretable natural language captions from weather time series data remains a significant challenge at the intersection of meteorological science and natural language processing. While recent advances in Large Language Models (LLMs) have demonstrated remarkable capabilities in time seri...

---

### 18. [In-network Attack Detection with Federated Deep Learning in IoT Networks: Real Implementation and Analysis](https://arxiv.org/abs/2603.21596)

**Authors**: Devashish Chaudhary, Sutharshan Rajasegarar, Shiva Raj Pokhrel, Lei Pan, Ruby D  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.21596v1  

#### Abstract
The rapid expansion of the Internet of Things (IoT) and its integration with backbone networks have heightened the risk of security breaches. Traditional centralized approaches to anomaly detection, which require transferring large volumes of data to central servers, suffer from privacy, scalability...

---

### 19. [PivotRL: High Accuracy Agentic Post-Training at Low Compute Cost](https://arxiv.org/abs/2603.21383)

**Authors**: Junkeun Yi, Damon Mosk-Aoyama, Baihe Huang, Ritu Gala, Charles Wang, Sugam Dipak Devare, Khushi Bhardwaj, Abhibha Gupta, Oleksii Kuchaiev, Jiantao Jiao, Jian Zhang, Venkat Srinivasan  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.21383v1  

#### Abstract
Post-training for long-horizon agentic tasks has a tension between compute efficiency and generalization. While supervised fine-tuning (SFT) is compute efficient, it often suffers from out-of-domain (OOD) degradation. Conversely, end-to-end reinforcement learning (E2E RL) preserves OOD capabilities,...

---

### 20. [Fast-Slow Thinking RM: Efficient Integration of Scalar and Generative Reward Models](https://arxiv.org/abs/2603.20212)

**Authors**: Jiayun Wu, Peixu Hou, Shan Qu, Peng Zhang, Ning Gu, Tun Lu  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.20212v1  

#### Abstract
Reward models (RMs) are critical for aligning Large Language Models via Reinforcement Learning from Human Feedback (RLHF). While Generative Reward Models (GRMs) achieve superior accuracy through chain-of-thought (CoT) reasoning, they incur substantial computational costs. Conversely, Scalar Reward M...

---

### 21. [Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference](https://arxiv.org/abs/2603.20224)

**Authors**: Patrick Wilhelm, Thorsten Wittkopp, Odej Kao  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.20224v1  

#### Abstract
Large Language Models (LLMs) demonstrate exceptional performance across diverse tasks but come with substantial energy and computational costs, particularly in request-heavy scenarios. In many real-world applications, the full scale and capabilities of LLMs are often unnecessary, as Small Language M...

---

### 22. [Incremental GNN Embedding Computation on Streaming Graphs](https://arxiv.org/abs/2603.20622)

**Authors**: Qiange Wang, Haoran Lv, Yanfeng Zhang, Weng-Fai Wong, Bingsheng He  
**Category**: cs.DC  
**Published**: 2026-03-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.20622v1  

#### Abstract
Graph Neural Network (GNN) on streaming graphs has gained increasing popularity. However, its practical deployment remains challenging, as the inference process relies on Runtime Embedding Computation (RTEC) to capture recent graph changes. This process incurs heavyweight multi-hop graph traversal o...

---

### 23. [Joint Surrogate Learning of Objectives, Constraints, and Sensitivities for Efficient Multi-objective Optimization of Neural Dynamical Systems](https://arxiv.org/abs/2603.20984)

**Authors**: Frithjof Gressmann, Ivan Georgiev Raikov, Seung Hyun Kim, Mattia Gazzola, Lawrence Rauchwerger, Ivan Soltesz  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.20984v1  

#### Abstract
Biophysical neural system simulations are among the most computationally demanding scientific applications, and their optimization requires navigating high-dimensional parameter spaces under numerous constraints that impose a binary feasible/infeasible partition with no gradient signal to guide the ...

---

### 24. [ALMAB-DC: Active Learning, Multi-Armed Bandits, and Distributed Computing for Sequential Experimental Design and Black-Box Optimization](https://arxiv.org/abs/2603.21180)

**Authors**: Foo Hui-Mean, Yuan-chin I Chang  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.21180v1  

#### Abstract
Sequential experimental design under expensive, gradient-free objectives is a central challenge in computational statistics: evaluation budgets are tightly constrained and information must be extracted efficiently from each observation. We propose \textbf{ALMAB-DC}, a GP-based sequential design fram...

---

### 25. [AgenticGEO: A Self-Evolving Agentic System for Generative Engine Optimization](https://arxiv.org/abs/2603.20213)

**Authors**: Jiaqi Yuan, Jialu Wang, Zihan Wang, Qingyun Sun, Ruijie Wang, Jianxin Li  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.20213v1  

#### Abstract
Generative search engines represent a transition from traditional ranking-based retrieval to Large Language Model (LLM)-based synthesis, transforming optimization goals from ranking prominence towards content inclusion. Generative Engine Optimization (GEO), specifically, aims to maximize visibility ...

---

### 26. [Counterfactual Credit Policy Optimization for Multi-Agent Collaboration](https://arxiv.org/abs/2603.21563)

**Authors**: Zhongyi Li, Wan Tian, Yikun Ban, Jinju Chen, Huiming Zhang, Yang Liu, Fuzhen Zhuang  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.21563v1  

#### Abstract
Collaborative multi-agent large language models (LLMs) can solve complex reasoning tasks by decomposing roles and aggregating diverse hypotheses. Yet, reinforcement learning (RL) for such systems is often undermined by credit assignment: a shared global reward obscures individual contributions, infl...

---

### 27. [RLVR Training of LLMs Does Not Improve Thinking Ability for General QA: Evaluation Method and a Simple Solution](https://arxiv.org/abs/2603.20799)

**Authors**: Kaiyuan Li, Jing-Cheng Pang, Yang Yu  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.20799v1  

#### Abstract
Reinforcement learning from verifiable rewards (RLVR) stimulates the thinking processes of large language models (LLMs), substantially enhancing their reasoning abilities on verifiable tasks. It is often assumed that similar gains should transfer to general question answering (GQA), but this assumpt...

---

### 28. [DiscoUQ: Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles](https://arxiv.org/abs/2603.20975)

**Authors**: Bo Jiang  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.20975v1  

#### Abstract
Multi-agent LLM systems, where multiple prompted instances of a language model independently answer questions, are increasingly used for complex reasoning tasks. However, existing methods for quantifying the uncertainty of their collective outputs rely on shallow voting statistics that discard the r...

---

### 29. [A Comparative Analysis of LLM Memorization at Statistical and Internal Levels: Cross-Model Commonalities and Model-Specific Signatures](https://arxiv.org/abs/2603.21658)

**Authors**: Bowen Chen, Namgi Han, Yusuke Miyao  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.21658v1  

#### Abstract
Memorization is a fundamental component of intelligence for both humans and LLMs. However, while LLM performance scales rapidly, our understanding of memorization lags. Due to limited access to the pre-training data of LLMs, most previous studies focus on a single model series, leading to isolated o...

---

### 30. [Probing How Scalable Table Data Enhances General Long-Context Reasoning](https://arxiv.org/abs/2603.21719)

**Authors**: Huaibing Xie, Guoliang Zhao, Yang Liu, Shihan Dou, Siming Huang, Yanling Xiao, Shaolei Wang, Yiting Liu, Cheng Zhang, Shaofan Liu, Pluto Zhou  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.21719v1  

#### Abstract
As real-world tasks grow increasingly complex, long-context reasoning has become a core capability for Large Language Models (LLMs). However, few studies explore which data types are effective for long-context reasoning and why. We find that structured table data with periodic structures shows stron...

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
