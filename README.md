# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-27 06:54:45 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DFLOP: A Data-driven Framework for Multimodal LLM Training Pipeline Optimization](https://arxiv.org/abs/2603.25120)

**Authors**: Hyeonjun An, Sihyun Kim, Chaerim Lim, Hyunjoon Kim, Rathijit Sen, Sangmin Jung, Hyeonsoo Lee, Dongwook Kim, Takki Yu, Jinkyu Jeong, Youngsok Kim, Kwanghyun Park  
**Category**: cs.DC  
**Published**: 2026-03-27  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2603.25120v1  

#### Abstract
Multimodal Large Language Models (MLLMs) have achieved remarkable advances by integrating text, image, and audio understanding within a unified architecture. However, existing distributed training frameworks remain fundamentally data-blind: they parallelize computation without accounting for variati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DFLOP: A Data-driven Framework for Multimodal LLM Training Pipeline Optimization**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
现代 **Multimodal LLM (MLLM)** 的训练面临严重的**计算负载不均衡**和**输入依赖的吞吐量波动**问题。传统分布式训练框架（如 Megatron-LM、PyTorch）采用“数据盲”（data-agnostic）策略，假设所有 microbatch 的计算成本一致，但在 MLLM 中，由于以下原因，该假设被打破：

- **异构架构**：pipeline 包含不同模块（如 modality encoder 和 LLM），其计算特性差异大。
- **动态输入形状**：图像数量、视频帧数、序列长度等变化导致处理成本高度可变。
- **3D 并行度配置僵化**：现有系统对整个模型使用统一的 Tensor Parallelism (TP)、Pipeline Parallelism (PP)、Data Parallelism (DP)，无法针对不同模块独立优化。

这导致 **GPU 利用率低、同步延迟高、训练效率下降**。

---

### **提出了什么新方法或新思路**
作者提出 **DFLOP** ——一个**数据驱动的 MLLM 训练流水线优化框架**，通过将**数据特征**与**执行规划**结合，实现端到端的性能提升。

#### **核心组件**
1. **Profiling Engine（分析引擎）**
   - **Model Profiler**：在合成数据上测量模型的内存消耗和吞吐量，构建预测性性能模型。
   - **Data Profiler**：分析真实训练数据集中输入形状（如 `Ebatch_size`, `Lseq_len`）的分布。

2. **Data-aware 3D Parallelism Optimizer（数据感知的并行优化器）**
   - 基于 Profiling 引擎的结果，在离线阶段搜索最优的 3D 并行配置（`Etp, Epp, Edp` for encoder；`Ltp, Lpp, Ldp` for LLM）。
   - 目标是最小化**期望 makespan**（训练迭代时间），考虑数据分布而非单一输入。

3. **Online Microbatch Scheduler（在线微批次调度器）**
   - 在运行时动态划分 global batch 为 microbatches，以平衡各 stage 的计算负载。
   - 使用 **ILP 求解器** 或 **LPT 启发式算法** 进行负载均衡，减少 pipeline bubbles。
   - 支持 **Adaptive Correction**：持续监控实际与预测吞吐量偏差，反馈调整调度策略。

4. **Inter-model Communicator（跨模型通信机制）**
   - 支持 encoder 和 LLM 使用不同的 DP 组大小，解决异构并行下的通信不匹配问题（如 encoder DP=4 vs LLM DP=2）。

---

### **相比现有方法的优势**
| 特性 | 传统框架（Megatron-LM, PyTorch） | DFLOP |
|------|-------------------------------|-------|
| 并行策略 | 全局统一的 3D 并行 | **模块级独立优化** |
| 数据感知 | ❌ 数据盲 | ✅ 显式建模数据分布 |
| 调度方式 | 随机分配 microbatch | ✅ 动态负载均衡 |
| 通信支持 | 固定 DP 组 | ✅ 支持异构 DP 组（via Inter-model Communicator） |
| 性能目标 | 最大化理论吞吐 | ✅ 最小化**期望 makespan** |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
构建了一个**混合数据集**以模拟真实 MLLM 训练场景：

| 数据集 | 类型 | 样本数 |
|--------|------|--------|
| LLaVA-Wild, AI2D, Infographic VQA | 单图（Single Image） | ~65k |
| M4-Instruct | 多图（Multiple Images） | 60k |
| LLaVA-Video | 视频（Video） | 60k |

> 数据异质性强，涵盖不同视觉 token 数量和文本长度。

---

### **实验设置和评估指标**

- **硬件平台**：最多 8 个节点，每节点 8×NVIDIA HGX A100 GPU，NVLink + 800Gbps InfiniBand。
- **评估模型**：
  - **LLaVA-OV**：SigLIP encoder + Qwen-2.5 / Llama-3 LLM（7B–72B）
  - **InternVL-2.5**：InternViT encoder + Qwen-2.5（72B）
  - **Qwen2-Audio**：音频语言模型，验证跨模态泛化能力
- **评估指标**：
  - **End-to-end training throughput**（tokens/s 或 samples/s）
  - **Training time reduction**
  - **GPU 利用率 / Idle time**
  - **Stage-wise throughput variance**
  - **Pipeline bubble 分析**

---

### **基线方法对比**
- **PyTorch**：基于 `torch.distributed` 的自定义 3D 并行实现。
- **Megatron-LM**：业界领先的高性能训练框架。
- 所有基线均手动调优至最佳配置。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**
- **端到端吞吐提升**：DFLOP 相比 PyTorch 和 Megatron-LM 实现 **1.2× 至 3.6× 的加速**。
- **训练时间缩短**：在多个 MLLM 配置下，总训练时间减少 **5 至 40 小时**。
- **GPU 利用率显著提高**：
  - **Idle time 减少 82–84%**（相比基线）。
  - **Stage-wise throughput 更高且更均衡**（见 Fig. 14）。

---

### **与基线方法的对比结果**
| 模型 | DFLOP 加速比（vs PyTorch） | DFLOP 加速比（vs Megatron-LM） |
|------|--------------------------|------------------------------|
| LLaVA-OV (Qwen-2.5 7B) | 1.3× | 1.2× |
| LLaVA-OV (Qwen-2.5 32B) | 2.7× | 2.3× |
| LLaVA-OV (Qwen-2.5 72B) | 3.1× | 3.6× |
| InternVL-2.5 (72B) | 3.6× | 3.6× |

> 性能优势随模型规模增大而增强。

---

### **消融实验结果**
#### **组件贡献分析（Fig. 10）**
- **LLaVA-OV (Llama-3 8B)**：主要收益来自 **Data-aware 3D Parallelism Optimizer**（静态优化主导）。
- **LLaVA-OV (Qwen-2.5 32B)**：主要收益来自 **Online Microbatch Scheduler**（动态调度更重要）。
- **InternVL-2.5 (72B)**：两者贡献相当。

> 表明 DFLOP 可根据不同模型的计算不对称性自动适应优化重心。

#### **Adaptive Correction 成本效益分析（Fig. 15）**
- 当异常输入频率低或延迟小（<50% stage duration）时，系统**自动关闭监控**，避免开销。
- 仅当性能增益 > 开销（~4%）时才启用，确保净收益为正。

#### **扩展性测试（Fig. 12）**
- 随着 GPU 节点数增加（1 → 32），DFLOP 的性能优势**持续扩大**。
- 原因：
  1. 更大的 GPU 资源池允许更精细的并行策略搜索。
  2. 在线调度器有效缓解大规模下的 straggler 问题。

---

## 4. **关键结论和发现**

### **主要发现**
1. **数据异质性是 MLLM 训练效率的关键瓶颈**，现有“数据盲”框架无法应对。
2. **模块级独立并行优化**（per-module 3D parallelism）是提升性能的核心，尤其在 encoder 与 LLM 计算负载接近时效果最显著（Fig. 8）。
3. **静态优化 + 动态调度** 的协同设计至关重要：前者提供全局最优配置，后者应对运行时波动。
4. **DFLOP 的实际 idle time 接近理论下限**，而基线系统因负载不均严重偏离理想情况（Fig. 13）。
5. **开放性和可复现性**：作者开源了框架代码，支持灵活的 process group 管理。

---

### **方法的局限性**
- **初始化开销**：Profiling 阶段耗时约 **7–10 分钟**，但占总训练时间比例极低（<2.1%）。
- **当前不兼容 DeepSpeed**：因其缺乏灵活的 pipeline parallelism 支持。
- **Adaptive Correction 依赖边缘案例频率**：若异常极少，可能不会触发。

---

### **未来工作方向**
- 支持更多模态（如 3D point clouds, sensor streams）。
- 扩展至推理阶段的动态批处理优化。
- 结合编译器技术（如 TorchInductor）进一步优化 kernel 级性能。
- 探索在线 re-profiling 机制以适应数据漂移（data drift）。

---

> **总结**：DFLOP 是首个将**数据分布**作为一等公民纳入 MLLM 训练优化的框架，通过**数据感知的静态配置**与**动态运行时调度**的联合设计，实现了高达 **3.6× 的端到端加速**，为大规模多模态模型训练提供了新的范式。

</details>

---

### 2. [S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation](https://arxiv.org/abs/2603.25702)

**Authors**: Ligong Han, Hao Wang, Han Gao, Kai Xu, Akash Srivastava  
**Category**: cs.CL  
**Published**: 2026-03-27  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.25702v1  

#### Abstract
Block-diffusion language models offer a promising path toward faster-than-autoregressive generation by combining block-wise autoregressive decoding with within-block parallel denoising. However, in the few-step regime needed for practical acceleration, standard confidence-thresholded decoding is oft...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **block-diffusion language models** 虽然通过块状自回归（block-wise autoregressive）解码与块内并行去噪相结合，实现了比传统自回归（AR）模型更快的生成速度，但在实际加速所需的“少步”（few-step）解码场景下，其性能仍面临挑战。标准的基于置信度阈值（confidence-thresholded）的解码策略在该场景下表现脆弱：激进的阈值会损害生成质量，而保守的阈值则需要不必要的去噪步骤，导致效率低下。

### 提出的新方法和新思路
本文提出了 **S2D2**（**S**elf-**S**peculative **D**ecoding for **D**iffusion），一种**无需训练的自投机解码**（training-free self-speculative decoding）框架。

其核心创新在于一个关键观察：当将 block-diffusion 模型的**块大小（block size）减小到1时，该模型就退化为一个标准的自回归（AR）模型**。这一特性使得可以复用同一个预训练好的 block-diffusion 模型，扮演两个角色：
- **Drafter (起草者)**：使用标准的大块大小进行 block-diffusion 解码，以实现并行去噪和快速生成。
- **Verifier (验证者)**：使用块大小为1的同一模型进行自回归解码，作为对草案序列的局部序列级评判器。

S2D2 在标准的 block-diffusion 解码流程中插入了一个“投机验证”（speculative verification）步骤。它利用验证者（Verifier）来对起草者（Drafter）提出的 token 进行基于拒绝采样（rejection sampling）的接受测试，从而在不牺牲模型分布的前提下，提高解码的准确性和速度。

### 相比现有方法的优势
- **无需额外训练**：S2D2 完全复用现有的预训练模型，不需要像 EDLM 那样训练辅助的能量模型，也不需要 distillation 或微调。
- **无需额外推理开销**：相比 ASSD 等需要特定架构（如 XLNet-style）的方法，S2D2 是即插即用的（plug-and-play），适用于绝大多数已有的 block-diffusion 模型。
- **更高的准确率-速度权衡**：通过引入一个更强大的局部序列级接受准则（基于验证者的概率比），S2D2 能够在保持甚至提升生成质量的同时，实现比动态置信度阈值（dynamic confidence-thresholding）等强基线更快的解码速度。

## 2. 核心实验方法和设置

### 使用的数据集
实验在以下四个基准数据集上进行：
- **GSM8K**: 数学推理任务。
- **MBPP**: Python 代码生成任务。
- **HumanEval**: Python 代码生成任务。
- **IFEval**: 指令遵循能力评估。

### 实验设置和评估指标
- **模型**：在五个来自三个主流 block-diffusion 家族的模型上进行了评估：
  - **SDAR** (1.7B/4B/8B)
  - **Fast-dLLM v2**
  - **LLaDA2.1-Mini**
- **评估指标**：
  - **准确率**（Accuracy）：在 GSM8K、MBPP、HumanEval 和 IFEval 上的得分。
  - **速度提升**（Speedup）：相对于自回归基线（block size=1）的加速比。
  - **准确率-速度权衡**（Accuracy-Speed Tradeoff）：综合评估模型的效率和效果。

### 基线方法对比
- **自回归基线**（AR）：块大小为1的标准自回归解码。
- **标准 Block-Diffusion**：使用静态（static）和动态（dynamic）置信度阈值的解码方法。
- **其他相关方法**：如 EDLM（需要额外训练）和 ASSD（需要特定架构）。

## 3. 主要实验结果和性能指标

### 关键性能数据
- **在 SDAR-1.7B 上**，S2D2 的 **config-B** 设置达到了 **4.7倍** 的速度提升（相比 AR 基线），同时平均准确率从 48.4 提升到了 **52.9**，实现了“又快又准”。
- **在 SDAR-8B-Chat 上**，S2D2 在多个配置下均优于动态置信度阈值基线。例如，在 `B=32` 的大块设置下，S2D2 将平均准确率从 70.5 提升至 **71.3**，同时速度提升从 2.6x 提高到 **3.7x**。
- **在 Fast-dLLM v2 上**，S2D2 在 `SB=32` 的配置下，相比动态解码基线，速度提升了约 **1.07倍**（3.1x vs 2.9x），且平均准确率大幅提升了 **+4.5** 个百分点。
- **在 LLaDA2.1-Mini 上**，S2D2 与模型内置的 self-correction 机制互补。在一个保守设置下，S2D2 达到了 **2.2倍** 的速度提升和 **79.3** 的平均准确率，而静态基线仅为 0.5x 速度和 79.2 准确率。这表明 S2D2 在此情况下是 **4.4倍更快且准确率略高**。

### 与基线方法的对比结果
S2D2 在所有测试的 block-diffusion 模型家族中，都显著改善了准确率-速度的前沿（accuracy-speed frontier）。它通常能够同时实现**更高的准确率和更低的延迟**，或者在相同速度下提供更高的准确率，尤其是在大块大小（large block size）这种标准 diffusion 解码不稳定的情况下，优势更为明显。

### 消融实验结果
- **路由策略**（Routing Policies）：研究了多种轻量级路由策略（如 Minimum-span, Score-threshold, Hysteresis）来决定何时执行验证。结果表明，这些策略能有效平衡验证带来的收益和额外一次前向传播的成本。
- **接受估计器**（Acceptance Estimators）：比较了不同的接受前缀长度预测器。虽然硬边际阈值（hard margin thresholding）在预测上最准确，但软熵估计器（soft entropy-based estimator）在最终任务准确率上表现更好，因此被选为主方法。
- **拒绝采样比率调整**（Rejection-sampling ratio tempering）：略微增加温度（γ=1.25）可以在某些配置下带来轻微的准确率提升，但默认的 γ=1 已经提供了良好的权衡。

## 4. 关键结论和发现

### 主要发现
- **自回归模式是天然的验证者**：将 block-diffusion 模型的块大小设为1，即可获得一个强大的、同源的（in-family）序列级评判器，用于指导和纠正并行解码过程。
- **投机验证是一种有效的能量校正**：S2D2 的验证过程可被解释为一种**随机的、贪婪的局部自回归能量校正**（stochastic, greedy local AR-guided energy correction）。它通过降低被接受 token 的残差能量（residual energy）来提高生成质量。
- **训练免费的推理优化是可行的**：S2D2 成功地证明了，无需任何额外的训练或复杂的架构修改，仅通过巧妙地重新利用模型的固有模式，就能显著提升 diffusion LLM 的推理效率和效果。

### 方法的局限性
- **额外的计算成本**：验证步骤增加了一次额外的前向传播，因此对于非常短的候选序列，其收益可能无法覆盖成本。
- **仅验证连续片段**：S2D2 当前只对第一个连续的掩码片段进行验证，未能充分利用模型对任意子集进行自回归建模的潜力。
- **依赖于模型的 AR 特性**：该方法的有效性建立在 block-diffusion 模型在块大小为1时能表现出良好 AR 性能的基础上。

### 未来工作方向
- **探索更复杂的路由策略**：开发更智能的、可能基于学习的路由机制，以更精确地判断何时进行验证。
- **扩展验证范围**：尝试验证非连续的 token 子集，或设计更高效的多片段验证方案。
- **理论分析深化**：进一步形式化 S2D2 与能量模型、变分推断等理论框架之间的联系。
- **应用于其他模型**：将 S2D2 的思想推广到其他类型的生成模型或解码范式中。

</details>

---

### 3. [CoordLight: Learning Decentralized Coordination for Network-Wide Traffic Signal Control](https://arxiv.org/abs/2603.24366)

**Authors**: Yifeng Zhang, Harsh Goel, Peizhuo Li, Mehul Damani, Sandeep Chinchali, Guillaume Sartoretti  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.24366v1  

#### Abstract
Adaptive traffic signal control (ATSC) is crucial in alleviating congestion, maximizing throughput and promoting sustainable mobility in ever-expanding cities. Multi-Agent Reinforcement Learning (MARL) has recently shown significant potential in addressing complex traffic dynamics, but the intricaci...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CoordLight: Learning Decentralized Coordination for Network-Wide Traffic Signal Control**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**自适应交通信号控制（ATSC）**中的两个核心挑战：
- **部分可观测性（Partial Observability）**：在去中心化多智能体系统中，单个路口（agent）只能获取局部交通状态，难以全面感知全局交通动态。
- **协调困难（Coordination Difficulty）**：缺乏有效的机制来建模相邻路口之间的动态依赖关系，导致决策短视、自私，影响网络级交通优化。

传统方法如固定时长控制（Fixed-Time）或简单压力控制（MaxPressure）无法应对复杂动态流量；而现有 MARL 方法虽有潜力，但在状态表示和邻居协调方面仍存在不足。

---

### **提出的新方法与新思路**

#### **(1) Queue Dynamic State Encoding (QDSE)**  
一种新颖的**状态表示方法**，基于车辆排队动力学模型构建，包含六个车道特征向量：
- `Q(t)`：停止车辆数（队列长度）
- `Nin(t)`：进入车辆数
- `Nout(t)`：离开车辆数
- `Nr(t)`：移动车辆总数
- `Nfr(t)`：紧随首辆移动车后的车辆数
- `Dfr(t)`：首辆移动车到队尾的距离

> ✅ **优势**：不仅反映当前拥堵情况，还能**预测未来可能形成的拥堵**，提升策略的前瞻性。

#### **(2) Neighbor-aware Policy Optimization (NAPO)**  
一种全新的**去中心化 MARL 算法**，通过注意力机制实现对邻居状态与动作依赖性的识别：
- 引入**空间-时间注意力网络（STN）**作为 Actor，捕捉邻居间的时空依赖。
- 设计**特权局部批评家网络（Privileged Local Critic）**，结合状态编码器与状态-动作解码器，聚合邻居的历史状态-动作信息。
- 利用加权优势函数更新策略，使 agent 能够“关注”对其影响最大的邻居。

> ✅ **优势**：增强协调能力，稳定训练过程，避免次优收敛。

---

### **相比现有方法的优势**
| 维度 | CoordLight 的优势 |
|------|------------------|
| **状态表示** | QDSE 比传统 VC、EP、ATS 更具预测性，优于图像式 DTSE 的简洁性 |
| **协调机制** | 不依赖通信或集中式训练（CTDE），完全去中心化，可扩展性强 |
| **算法设计** | NAPO 显式建模 state-action 依赖，优于仅靠奖励塑形或图注意力的方法（如 CoLight） |
| **实用性** | 在真实城市路网（最大达 196 个交叉口）上验证有效，具备现实部署潜力 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
基于开源仿真平台 **CityFlow**，采用三个来自真实城市的交通网络：
| 数据集 | 规模 | 描述 |
|--------|-----|------|
| **Jinan** | 3×4 = 12 个路口 | 中国济南，中等规模，三种不同流量需求（DJN1–3） |
| **Hangzhou** | 4×4 = 16 个路口 | 中国杭州，两种流量需求（DHz1–2） |
| **New York** | 7×28 = 196 个路口 | 美国纽约曼哈顿部分区域，大规模高密度场景 |

所有路口均为四向标准布局，每方向三条车道。

---

### **实验设置**
- **训练/测试时长**：3600 秒
- **相位持续时间**：固定为 5 秒（含黄灯补偿）
- **相位选择**：从一组无冲突 traffic phases 中选择下一阶段
- **策略共享**：全网使用同一组参数（homogeneous policy）
- **训练平台**：Ubuntu + RTX 3060，使用 PyTorch 实现
- **超参数**：
  - 学习率：Actor 3e-4，Critic 5e-4
  - 批大小：720
  - 折扣因子 γ = 0.98，GAE λ = 0.98
  - PPO 更新轮数：6

---

### **评估指标**
主指标为 **平均旅行时间（Average Travel Time, ↓越小越好）**：
$$
\text{Travel Time} = \frac{1}{N_v} \sum_{i=1}^{N_v} (\min(t_{\text{end}}, 3600) - t_{\text{start}})
$$
其中 $N_v$ 是总车辆数，若未完成行程则以 3600 秒计算。

辅助指标包括：
- 平均队列长度（↓）
- 车辆速度（↑）
- 队列标准差（↓，衡量稳定性）

---

### **基线方法对比**
分为两类：

#### **传统方法**
- **FixedTime**：基于预设周期的定时控制
- **MaxPressure (MP)**：贪心选择最小压力相位
- **Advanced-MP**：考虑有效范围内移动车辆的压力控制

#### **MARL 基线**
| 方法 | 特点 |
|------|------|
| **CoLight / Advanced-CoLight** | 使用 GAT 进行图注意力协作 |
| **MPLight / Advanced-MPLight** | 基于压力的状态与奖励设计 |
| **DenseLight*** | 使用密集反馈与非局部特征提取 |
| **SocialLight** | 分布式估计个体贡献以促进合作 |

> 注：部分结果因无法复现，引用原论文数据（标 *）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table II）**

| 方法 | Jinan DJN(1) | Hangzhou DHz(2) | NYC DNY(2) |
|------|--------------|------------------|------------|
| FixedTime | 346.36 s | 359.44 s | 1660.29 s |
| MaxPressure | 273.96 s | 348.98 s | 1535.77 s |
| Advanced-MP | 253.61 s | 318.67 s | — |
| CoLight | 276.33 s | 297.26 s | 1476.18 s |
| SocialLight | 217.92 s | 288.55 s | 1106.69 s |
| **CoordLight (Ours)** | **199.24 s** | **250.87 s** | **1039.15 s** |

> ✅ **性能提升显著**：
- 在 Jinan 上比 SocialLight 快 **8.57%**
- 在 Hangzhou 高负载下快 **7.87%**
- 在 NYC 大规模网络上快 **6.1%**

---

### **统计显著性检验（Table III）**
进行独立 t-test 对比 CoordLight 与第二好的 SocialLight：
- 所有 7 个实验场景下的 **p-value < 1.1e-8**
- 经 Bonferroni 校正后阈值为 1.57e-4 → **全部显著优于**

---

### **消融实验结果（Ablation Studies）**

#### **(1) 状态表示 QDSE 的有效性（Fig. 7a）**
比较五种 state encoding：
- **VC / GP / EP / ATS / DTSE / QDSE**
- 结果显示：**QDSE 在 travel time、speed、queue length 及其方差上均最优**
- 尤其是 **queue std ↓ 最明显**，说明其能更好平滑交通波动
- QDSE 性能接近甚至略优于更复杂的 DTSE 图像表示 → **性价比更高**

#### **(2) 模块组件分析（Fig. 7b）**
移除关键模块的影响：
| 变体 | 平均旅行时间（DHz2） | 相对下降 |
|------|--------------------|---------|
| CoordLight (完整) | 250.87 s | — |
| w/o QDSE | ~270 s | ↑ ~8% |
| w/o STN（无时空注意力） | ~280 s | ↑ ~12% |
| w/o AD（无状态-动作解码器） | ~275 s | ↑ ~10% |
| Base（仅 FC + IPPO） | >300 s | ↑ >20% |

> 🔍 发现：
- QDSE 和 NAPO 各自带来约 8–10% 改进
- 二者协同作用更强，体现**互补性**

#### **(3) 传感器噪声鲁棒性测试（Fig. 8）**
模拟摄像头定位误差（高斯噪声 σ=10m/20m/30m）：
- 即使在 30 米噪声下，平均旅行时间增加不超过 **2.34%**
- 表明 QDSE 对实际部署中的传感不确定性具有较强鲁棒性

#### **(4) 泛化性验证**
将 QDSE 应用于其他算法（IPPO、SocialLight）也带来性能提升 → **QDSE 具备通用增强潜力**

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **精细的状态表示至关重要**：QDSE 通过建模排队动力学，显著提升了 agent 对未来拥堵的预测能力。
2. ✅ **显式的邻居感知机制优于隐式协作**：NAPO 通过注意力机制主动识别“关键邻居”，实现更有针对性的协调。
3. ✅ **去中心化不等于低效**：CoordLight 在无需全局信息的前提下，在大规模网络上实现了超越 CTDE 类方法的表现。
4. ✅ **训练更稳定、性能更一致**：在多种流量模式下表现稳健，尤其在高负载场景优势明显。

---

### **方法的局限性**
- 当前假设所有路口结构同质（homogeneous），尚未处理异构路口（如 T 型、环岛等）
- 相位切换逻辑简化（固定 5 秒），未联合优化 phase duration
- 依赖高质量检测数据（如车辆位置、速度），极端天气或遮挡可能影响 QDSE 效果
- 尚未集成优先通行（如救护车）、突发事件响应等功能

---

### **未来工作方向**
1. **扩展至异构网络**：支持不同类型路口、车道配置和信号序列
2. **引入动态相位时长控制**：将 phase duration 纳入 action space
3. **处理不完美感知**：研究在缺失或错误观测下的鲁棒学习
4. **融合事件驱动机制**：支持紧急车辆优先、事故绕行等现实功能
5. **跨城市迁移学习**：探索 meta-learning 或 domain adaptation 提升泛化能力

---

> 📌 **总体评价**：  
> **CoordLight 是一个兼具理论深度与工程实用性的 ATSC 框架**。它通过 QDSE 和 NAPO 的双重创新，在状态建模与多智能体协调之间取得了良好平衡，为大规模去中心化交通控制提供了新的范式。代码已开源（GitHub: [marmotlab/CoordLight](https://github.com/marmotlab/CoordLight)），有望推动领域发展。

</details>

---

### 4. [Prune as You Generate: Online Rollout Pruning for Faster and Better RLVR](https://arxiv.org/abs/2603.24840)

**Authors**: Haobo Xu, Sirui Chen, Ruizhong Qiu, Yuchen Yan, Chen Luo, Monica Cheng, Jingrui He, Hanghang Tong  
**Category**: cs.CL  
**Published**: 2026-03-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.24840v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Large Language Models (LLMs). However, methods such as GRPO and DAPO suffer from substantial computational cost, since they rely on sampling many rollouts for each prompt. Moreover, in RLVR...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Prune as You Generate: Online Rollout Pruning for Faster and Better RLVR*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **高计算成本**：在 **Reinforcement Learning with Verifiable Rewards (RLVR)** 中，如 **GRPO** 和 **DAPO** 等方法依赖于为每个 prompt 生成大量 **rollouts**，导致训练过程极其耗时且资源密集。
- **稀疏学习信号**：由于奖励是二元的（0/1），rollouts 经常出现“全对”或“全错”的情况，导致组内奖励方差低，优势估计退化为零，从而产生微弱甚至消失的策略梯度（vanishing policy gradient）。

### 提出了什么新方法或新思路
提出 **ARRoL (Accelerating RLVR via online RoLlout Pruning)**，一种**在线 rollout 剪枝方法**，其核心思想是“边生成边剪枝”：
- **在线质量预测头 (Quality Head)**：在训练过程中动态训练一个轻量级的 **MLP 质量头 (quality head)**，该头基于模型早期生成的 **partial rollout** 隐藏状态，预测其最终成功的概率。
- **早期剪枝决策**：利用质量头的预测分数，在 rollout 生成到某个中间长度（如 `L_detect=512`）时，立即做出剪枝决策，移除那些极可能重复（全是正确或全是错误）的样本。
- **显式平衡控制**：通过设计生存概率函数，主动引导剩余的 rollouts 向 **0.5 的正样本比例**靠拢，从而最大化组内奖励方差，增强学习信号。
- **测试时扩展 (Test-time Scaling)**：将训练好的质量头用于推理阶段，作为候选答案的投票权重，替代简单的多数投票（majority vote），提升最终答案的准确性。

### 相比现有方法的优势
- **效率更高**：相比后处理剪枝（post-rollout pruning）或推测解码（speculative decoding），ARRoL 在生成过程中就移除冗余序列，显著减少总生成时间。
- **信号更强**：通过主动平衡，解决了 RLVR 中奖励稀疏的问题，使得学习信号更稳定、更有效。
- **端到端加速**：不仅减少了生成开销，还因参与后续 log-prob 计算和策略更新的样本数减少，进一步降低了整体训练成本。
- **统一框架**：同一个质量头既可用于训练时剪枝，也可用于测试时加权投票，实现了功能复用。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练数据集**：**Dapo-Math-17K**，一个包含数学问题和可验证答案的大规模数据集。
- **评估数据集**：
  - **Math500**：来自 MATH 数据集的 500 道高中水平数学题。
  - **Minervamath**：272 道来自 MIT 课程的定量推理题。
  - **OlympiadBench**：8,476 道奥赛级别的数学和物理题。
  - **AMC'23**, **AIME'24**, **AIME'25**：美国数学竞赛系列，题目难度递增，用于评估在挑战性任务上的表现。

### 实验设置和评估指标
- **模型**：在 **Qwen-3** (1.7B, 4B, 8B) 和 **LLaMA-3.2** (1B) 系列模型上进行实验。
- **算法**：将 ARRoL 分别集成到 **GRPO** 和 **DAPO** 两种主流 RLVR 算法中。
- **评估指标**：
  - **平均准确率 (Average Accuracy)**：在多个基准上的平均表现。
  - **pass@16**：在 AMC/AIME 等竞赛题上，16 次采样中至少有一次正确的概率。
  - **maj@32**：32 次采样后，通过多数投票得到的准确率。
  - **训练速度 (Speedup)**：以 wall-clock time 衡量的端到端训练加速比。

### 基线方法对比
- **Vanilla GRPO / DAPO**：不进行任何剪枝的标准方法。
- **Random Pruning**：随机剪枝一半的 rollouts，用于验证 ARRoL 剪枝策略的有效性。
- **DeepConf (Fu et al., 2025)**：一种基于 log-probability 的启发式置信度方法，用于与质量头在测试时加权投票的效果进行对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **训练加速**：ARRoL 实现了 **1.6× 到 1.7×** 的端到端训练速度提升。
- **训练精度提升**：
  - 在 GRPO 上，平均准确率提升 **+2.30 至 +2.87**。
  - 在 DAPO 上，平均准确率提升 **+2.99**。
- **测试时扩展增益**：利用质量头作为投票权重，相比 DeepConf 方法，在平均准确率上额外获得了高达 **+8.33** 的提升。

### 与基线方法的对比结果
- **vs Vanilla GRPO/DAPO**：ARRoL 在所有模型和大部分数据集上均显著优于基线，尤其是在更难的 AIME 等数据集上提升更大（例如在 Qwen-3-8B 上 AIME'24 提升 +10.00）。
- **vs Random Pruning**：尽管两者都剪枝了约一半的样本，但 ARRoL 显著优于随机剪枝（见 Table 4），证明了其**智能选择机制**的重要性。
- **vs DeepConf**：在测试时投票中，学习得到的质量头分数比基于 log-prob 的 DeepConf 更能准确反映最终答案的正确性，因此效果更好。

### 消融实验结果
- **不同保留率 (keep ratio K) 的影响**（Table 6）：
  - `K=0.5` 时，在精度和速度之间取得了最佳平衡。
  - `K=0.25` 时，虽然速度更快（2.33×），但性能略有下降，说明过度剪枝会损失信息。
  - `K=1.0` 时即无剪枝，速度最慢。
- **效率分解**（Table 5）：
  - **Rollout Generation**：加速 **1.46×**（因为需先生成到 `L_detect`）。
  - **Log-prob Computation & Model Update**：加速约 **2×**（因为处理的样本数直接减半）。
- **组内平衡性验证**：
  - ARRoL 将组内正样本期望比例 `E[p]` 推向 0.5，并显著提高了 `E[p(1-p)]`（与组内奖励方差成正比），证明了其成功增强了学习信号。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **“少即是多” (Less is More)**：通过智能地减少参与训练的 rollouts 数量，反而可以**提高学习效率和最终性能**，因为保留下来的样本提供了更强、更平衡的学习信号。
2. **在线剪枝是可行的**：在 rollout 生成的早期阶段，一个轻量级的 **quality head** 就能相对准确地预测其最终命运，这为在线干预提供了理论基础。
3. **系统级优化至关重要**：将剪枝逻辑深度集成到推理引擎（如 vLLM）的调度器中，能够及时释放 GPU 资源，是实现端到端加速的关键。
4. **多功能性**：训练出的质量头不仅可以用于训练时剪枝，还可以无缝迁移到测试时，作为更可靠的置信度分数用于投票聚合。

### 方法的局限性
- **检测长度开销**：必须将每个 rollout 生成到 `L_detect`（如 512）才能进行评估，这限制了在**生成阶段**的最大加速潜力。
- **领域依赖性**：研究主要集中在具有**可验证奖励 (verifiable rewards)** 的数学推理任务上。在奖励不可直接验证或更复杂的交互式任务（如 UI 操作）中的泛化能力有待验证。
- **冷启动问题**：需要一个短暂的冷启动期（如 20 步）来初始化质量头和校准器。

### 未来工作方向
- 将 ARRoL 的思想推广到其他类型的 RL 任务，如工具调用（tool-use）、游戏代理等。
- 探索更早的检测点或自适应的 `L_detect` 策略，以进一步降低生成开销。
- 研究如何将此方法应用于非二元、稠密奖励的场景。
- 探索质量头在模型自省（self-reflection）和错误诊断方面的潜在应用。

</details>

---

### 5. [Symbolic--KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning](https://arxiv.org/abs/2603.23854)

**Authors**: Salah A Faroughi, Farinaz Mostajeran, Amirhossein Arzani, Shirko Faroughi  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.23854v1  

#### Abstract
Symbolic discovery of governing equations is a long-standing goal in scientific machine learning, yet a fundamental trade-off persists between interpretability and scalable learning. Classical symbolic regression methods yield explicit analytic expressions but rely on combinatorial search, whereas n...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Symbolic-KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning**

---

## 1. 论文的主要贡献和创新点

### **解决的问题**
科学机器学习（Scientific Machine Learning, SciML）中的一个长期挑战是**模型可解释性与可扩展性之间的权衡**：
- **传统符号回归方法**（如遗传算法、SINDy）能生成显式的解析表达式，但依赖组合搜索，计算成本高，难以扩展到高维问题。
- **标准神经网络**（如MLP、PINN）虽能高效处理大规模数据，但其内部表示是“黑箱”，缺乏可解释性。

本文旨在弥合这一差距，提出一种既能**从数据中自动发现简洁、可读的符号表达式**，又具备**深度神经网络的可扩展性和训练效率**的方法。

### **提出的新方法与创新思路**
作者提出了 **Symbolic Kolmogorov-Arnold Networks (Symbolic-KAN)**，其核心创新包括：

- **嵌入离散符号结构的神经网络架构**：
  - 基于 **Kolmogorov-Arnold 表示定理 (KART)**，将多元函数分解为一元函数的叠加。
  - 与标准KAN不同，Symbolic-KAN引入了一个**由解析基函数库驱动的离散选择机制**。

- **三大核心机制实现符号化**：
  1. **解析基函数库 (Analytic Primitive Library)**：
     - 预定义一组人类可读的函数（如 `x`, `x²`, `sin x`, `cos x`, `log(1+|x|)` 等），作为候选“原子”操作。
  2. **分层门控机制 (Hierarchical Gating)**：
     - **Primitive Gating**：通过 Gumbel-Softmax 机制，让每个边（edge）从库中“软选择”一个基函数，训练后期退火为 one-hot 硬选择。
     - **Edge Selection Mask**：每个单元（unit）从多个投影边中选择一条最相关的路径。
     - **Unit Gating**：决定哪些隐藏单元被保留，实现结构稀疏化。
  3. **符号正则化 (Symbolic Regularization)**：
     - 包括熵正则化（鼓励 one-hot 分布）和非极大抑制（NMS，避免重复选择相同基函数），引导模型向低复杂度、高可解释性的符号形式收敛。

- **无需后处理的端到端符号发现**：
  - 训练结束后，网络自然坍缩为一个紧凑的闭式表达式（closed-form expression），**无需额外的符号拟合步骤**。

### **相比现有方法的优势**
| 方法 | 可解释性 | 可扩展性 | 是否需后处理 | 能否发现新结构 |
|------|----------|-----------|----------------|------------------|
| **Symbolic Regression (e.g., Genetic Programming)** | ✅ 高 | ❌ 低 | ❌ 否 | ✅ 是 |
| **Sparse Regression (e.g., SINDy)** | ✅ 高 | ✅ 中 | ❌ 否 | ❌ 否（依赖预设库） |
| **Standard KAN / PINN** | ❌ 低 | ✅ 高 | ✅ 是（需符号拟合） | ✅ 是 |
| **Symbolic-KAN (本文)** | ✅✅ 高 | ✅ 高 | ❌ 否 | ✅ 是 |

> ✅ **优势总结**：兼具高可解释性与高可扩展性，端到端输出符号表达式，可作为 SINDy 等方法的“前置库优化器”。

---

## 2. 核心实验方法和设置

### **使用的数据集与任务**
实验涵盖三类典型科学学习任务：

1. **Data-driven Regression**：
   - 目标函数：`F(x) = x²` 和 `F(x) = sin(3x)/(1+x²) + 0.4cos(5x)`
   - 数据来源：人工生成的散点数据（无噪声或低噪声）
   - 目的：验证能否从纯数据中恢复正确符号结构。

2. **Dynamical System Identification (Inverse Problem)**：
   - **Van der Pol Oscillator**：
     ```
     dx/dt = a*y
     dy/dt = μ*(1 - x^2.15)*y - c*x
     ```
   - 参数 `a`, `μ`, `c` 未知，需从时间序列数据中识别。
   - 数据：通过 RK45 数值求解生成轨迹，采样 `T=20` 和 `T=50` 两个时间段。

3. **Physics-Informed Learning of PDEs**：
   - **Reaction-Diffusion Equation**（逆问题）：
     ```
     D*u_xx + K*tanh(u) = f(x), u(x) = sin^4(6x)
     ```
     - 目标：估计参数 `K` 并重建解。
   - **Laplace Equation**（正问题）：
     ```
     ∇²u = 0, u(x,y) = sin(πx)*sinh(πy)
     ```
     - 目标：在物理约束下求解，并分析符号结构。

### **实验设置与评估指标**

| 设置项 | 描述 |
|--------|------|
| **网络配置** | 多层 Symbolic-KAN，每层 `K_e` 个单元，每个单元 `E` 条边（默认 `[L, Ke, E] = [4,6,3]`） |
| **基函数库** | `{0, 1, x, x², x³, sin x, cos x, tanh x, exp x, log(1+|x|), ...}` |
| **训练流程** | 两阶段：<br>1. **软训练**：使用 Gumbel-Softmax 温度退火，逐步锐化选择。<br>2. **硬固化**（Hardening）：将门控变为 one-hot，再用 L-BFGS 微调参数。 |
| **损失函数** | `L = λ_data * L_data + L_phys + λ_sel * L_sel + λ_unit * L_unit + λ_bias * L_bias`<br>其中 `L_sel` 包含熵正则和 NMS 正则。 |

### **评估指标**
- **相对误差 (Relative Error)**：
  $$
  \mathcal{E}(F) = \frac{\|F_{\text{true}} - F_{\text{pred}}\|}{\|F_{\text{true}}\|}
  $$
- **参数识别误差**：预测参数与真实值的相对误差。
- **符号结构一致性**：最终选中的基函数是否与真实解的数学形式一致。

### **基线方法对比**
- **cPIKAN**：基于 Chebyshev 多项式的 Physics-Informed KAN。
- **PINN**：标准 Physics-Informed Neural Network（使用 `tanh` 激活函数）。
- **SINDy**：作为符号发现的对比基准（但未直接用于所有实验）。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **1. Data-driven Regression**
- 对 `F(x)=x²`：
  - 相对误差：`1.04×10⁻⁵`
  - 成功识别出 `x` 和 `x²` 原语，冗余项（如 `sin x`）被剪枝。
- 对 `F(x)=sin(3x)/(1+x²)+0.4cos(5x)`：
  - 相对误差：`7.75×10⁻³`
  - 选出 `sin`, `cos`, `lorentz (1/(1+x))` 等关键原语，结构合理。

#### **2. Van der Pol Oscillator**
| 时间范围 | 参数 `a` 误差 | `μ` 误差 | `c` 误差 | 轨迹相对误差 |
|---------|--------------|----------|----------|----------------|
| `T=20` | ~0.01% | ~1% | ~0.02% | `6.02×10⁻⁴` |
| `T=50` | <0.1% | ~7% | <0.1% | `5.87×10⁻³` |

> ✅ 即使存在非整数幂 `x^2.15`，也能准确识别参数和动态结构。

#### **3. Reaction-Diffusion 方程**
| 方法 | 域 `[-2,2]` 误差 | `K` 识别误差 | 域 `[-4,4]` 误差 | `K` 误差 |
|------|------------------|---------------|------------------|-----------|
| **Symbolic-KAN** | `5.93×10⁻⁴` | <0.1% | `9.37×10⁻³` | ~0.2% |
| **cPIKAN** | `8.25×10⁻⁴` | <0.05% | `2.07×10⁻¹` | ~5.6% |
| **PINN** | `1.50×10⁻¹` | ~1.4% | `2.15×10⁻¹` | ~2.7% |

> 🔺 在大域上，Symbolic-KAN 的误差比 cPIKAN 低 **95%**，比 PINN 低 **99%+**。

#### **4. Laplace 方程**
| 方法 | 验证误差 |
|------|----------|
| **Symbolic-KAN** | `1.11×10⁻³` |
| **cPIKAN** | `8.76×10⁻³` |
| **PINN** | `2.71×10⁻³` |

> ✅ Symbolic-KAN 误差比 cPIKAN 低 **87%**，比 PINN 低 **59%**。
> ✅ 最终选出 `sin`, `cos`, `sinh`, `cosh`，与真解 `sin(πx)sinh(πy)` 完全一致。

### **消融实验（隐含）**
虽然未明确列出消融表，但从设计逻辑可见：
- 若移除 **Gumbel-Softmax** 或 **熵正则**，无法实现 one-hot 选择，导致表达式冗长。
- 若禁用 **Unit Gating**，模型无法自动剪枝，结构复杂度上升。
- **基函数库的设计**直接影响发现能力——库越丰富，越可能覆盖真实结构。

---

## 4. 关键结论和发现

### **主要发现**
1. ✅ **Symbolic-KAN 能端到端地从数据或物理约束中发现简洁、可读的符号表达式**，无需后处理。
2. ✅ 所选基函数与真实解的数学结构高度一致（如 `sin`, `tanh`, `lorentz`），表明其具有**机制可解释性**（mechanistic interpretability）。
3. ✅ 在 **PDE 求解、动力系统识别、函数回归**等任务中，均优于或媲美主流基线（PINN, cPIKAN），尤其在**外推性、大域稳定性、参数识别精度**方面表现突出。
4. ✅ 可作为 **SINDy 等稀疏回归方法的“前置库优化器”**，自动发现重要基函数，缓解“库设计偏见”问题。

### **局限性**
1. **仍受限于基函数库的完备性**：若真实解涉及库中不存在的函数（如 `erf(x)`、特殊函数），则无法精确恢复。
2. **非整数指数或复杂复合函数的表示能力有限**：虽然能识别 `x^2.15` 的影响，但无法直接输出该形式，而是通过多项式近似。
3. **训练过程较复杂**：两阶段训练 + 门控机制增加了实现难度和调参成本。
4. **尚未完全解决外推泛化问题**：尽管在某些任务中表现良好，但神经网络固有的外推风险依然存在。

### **未来工作方向**
1. **动态扩展基函数库**：结合符号回归引擎，在训练过程中自动生成新函数加入库中。
2. **应用于更复杂的 PDE 系统**：如 Navier-Stokes、Maxwell 方程等多场耦合问题。
3. **与因果发现结合**：利用符号结构揭示变量间的因果关系。
4. **硬件部署与实时推理**：由于最终模型极简，适合嵌入式或边缘设备应用。
5. **理论分析**：建立 Symbolic-KAN 的泛化误差界、收敛性证明等理论基础。

---

> **总结**：  
> **Symbolic-KAN** 是迈向**可扩展、可解释、机制化学习**的重要一步。它成功将神经网络的强大拟合能力与符号系统的透明性结合起来，不仅提升了模型性能，更重要的是提供了**从数据中“读懂”物理规律**的可能性，为科学发现提供了新的工具范式。

</details>

---

### 6. [Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation](https://arxiv.org/abs/2603.23838)

**Authors**: Han Zheng, Yining Ma, Brandon Araki, Jingkai Chen, Cathy Wu  
**Category**: cs.AI  
**Published**: 2026-03-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23838v1  

#### Abstract
Lifelong Multi-Agent Path Finding (MAPF) is critical for modern warehouse automation, which requires multiple robots to continuously navigate conflict-free paths to optimize the overall system throughput. However, the complexity of warehouse environments and the long-term dynamics of lifelong MAPF o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**Lifelong Multi-Agent Path Finding (MAPF)** 在现代仓库自动化中的应用挑战。传统 one-shot MAPF 方法假设静态任务分配，而现实场景中机器人需持续接收新任务，导致系统面临动态拥堵、长期依赖性和级联低效等问题。现有方法如 **CBS** 和 **PBS** 虽然在小规模下表现良好，但在大规模、高密度环境下计算开销大；而基于学习的方法尚未在复杂动态环境中稳定超越搜索方法。

### 提出的新方法：RL-RH-PP
作者提出 **Reinforcement Learning-guided Rolling Horizon Prioritized Planning (RL-RH-PP)**，是首个将强化学习（RL）与经典搜索式规划器结合用于 lifelong MAPF 的框架。其核心思想是：
- 将动态优先级分配建模为一个 **Partially Observable Markov Decision Process (POMDP)**。
- 使用一个基于 **Transformer** 的神经网络作为 RL 策略，自回归地解码高质量的全局优先级顺序（total priority order）。
- 利用经典的 **Prioritized Planning (PP)** 作为轻量级、高效的路径求解骨干，在滚动时域（rolling horizon）内执行路径规划。

### 相比现有方法的优势
- **效率与质量兼顾**：相比复杂的搜索方法（如 CBS/PBS），PP 具有线性时间复杂度，适合实时大规模部署；通过 RL 引导生成更优的优先级顺序，显著提升了 PP 的解的质量。
- **长时序决策能力**：RL 策略能够捕捉 agent 之间的时空交互模式，并进行前瞻性规划，避免短期贪婪决策引发的拥堵或死锁。
- **可扩展性强**：模型设计支持零样本迁移（zero-shot generalization），能泛化到不同 agent 密度、规划窗口大小及未见地图布局。

---

## 2. 核心实验方法和设置

### 数据集与仿真环境
实验基于两个真实启发的仓库地图构建模拟环境：
- **Amazon fulfillment center dense map**：障碍物密度为 15.3%，具有多条平行通道。
- **Symbotic warehouse map**：障碍物密度高达 56.6%，存在瓶颈区域（如狭窄交叉通道），更具挑战性。

任务生成方式符合实际物流流程，例如 Symbotic 地图中 agent 需要在 inbound、outbound 和 aisle 区域之间循环运输货物。

### 实验设置
- **规划参数**：
  - 规划时域 `w` = 20，
  - 执行时域 `h` = 5，
  - 总仿真步数 `T` = 800。
- **训练配置**：
  - 使用 **Proximal Policy Optimization (PPO)** 进行训练。
  - 编码器采用堆叠的 multi-head attention 层，分别处理时间和空间维度。
  - 在 N = {80, 100, 120} 下分别训练策略。
- **推理阶段**：使用 Top-K 采样机制从 RL 策略中抽取 K 个候选优先级顺序，选择最优者送入 RH-PP 执行。

### 评估指标
- **Throughput per agent (TPA)**：每个 agent 成功完成的任务数量（平均值）。
- **Total throughput**：所有 agent 完成的总任务数（TPA × N）。
- **Solve time**：每次规划步骤所消耗的 CPU/GPU 时间。

### 基线方法对比
- **RH-CBS / RH-PBS**：基于滚动时域的 Conflict-Based Search 和 Priority-Based Search。
- **PIBT**：去中心化的优先继承方法，运行速度快但局部决策可能导致次优。
- **WPPL**：2023 年“Robot Runner”竞赛冠军方案，结合 PIBT 与 LNS 优化。
- **RH-PP (Random)**：随机生成优先级顺序的基准版本。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | Amazon (N=120) TPA | Symbotic (N=120) TPA |
|------|---------------------|------------------------|
| RH-CBS | 2.84 ± 0.29 | 1.50 ± 0.45 |
| RH-PBS | 3.37 ± 0.26 | 1.76 ± 1.10 |
| PIBT | 16.09 ± 0.48 | 2.67 ± 0.49 |
| WPPL | 23.59 ± 0.26 | 10.05 ± 1.33 |
| **RL-RH-PP (Ours)** | **25.56 ± 0.55** | **11.31 ± 2.21** |

> ✅ **RL-RH-PP 在两种地图上均取得最高 throughput**，尤其在高密度、高障碍的 Symbotic 地图上优势明显。

- 在 Amazon 地图上，相比 RH-PP(K=5)，**平均提升约 25% 的 throughput**。
- 在 Symbotic 地图上，随着 agent 数量增加，RL-RH-PP 显著优于其他方法，展现出更强的抗拥堵能力。

### 与基线方法的对比结果
- **优于所有搜索类方法**：尽管 RH-CBS 和 RH-PBS 在理论上更完备，但由于计算资源限制（1s CPU time budget），难以在高密度场景找到可行解。
- **优于强启发式方法 WPPL**：RL-RH-PP 在 Amazon 上高出 ~8.4%，在 Symbotic 上高出 ~12.5%。
- **优于纯随机 PP**：证明 RL 学习到的优先级顺序确实优于随机策略。

### 消融实验结果
#### （1）奖励函数权重分析
- 设置 congestion penalty `K` 和 infeasibility penalty `σ` 均为 1000 时达到最佳性能。
- 若设为 0，则训练不稳定且最终 throughput 下降显著。
- 结论：两项惩罚对引导策略避开拥堵和不可行路径至关重要。

#### （2）编码器结构消融
比较四种变体：
- **Full Model (ours)**：同时含 temporal 和 spatial attention。
- **w/o Temporal**：用 MLP 替代时间注意力。
- **w/o Spatial**：用 MLP 替代空间注意力。
- **Replace with Yan & Wu (2024)**：使用 CNN + intra-path attention 架构。

结果表明：
- 移除任一 attention 模块都会降低性能，尤其是在 Symbotic 地图上。
- 使用 CNN 架构在 Symbotic 上完全无法收敛，说明其难以建模全局 agent 间交互。

#### （3）与其他启发式对比
引入 **Distance Query Heuristic (DQ-RH-PP)**，即按最短路径长度排序。
- 结果显示 DQ-RH-PP 表现远低于 RL-RH-PP，说明固定规则无法适应动态拥堵演化。
- 证明 RL 学习到的是更复杂的、情境感知的优先级策略。

#### （4）上下文 Bandit 对比
将 RL 替换为单步决策（contextual bandit），即不考虑未来影响。
- 虽然初期学习更快，但最终 throughput 明显更低。
- 说明 **long-horizon planning 是实现高性能的关键**。

---

## 4. 关键结论和发现

### 主要发现
1. **RL 可有效指导传统启发式方法**：将 RL 用于生成优先级顺序，而非端到端路径生成，是一种高效且实用的设计范式。
2. **RL-RH-PP 能主动缓解拥堵**：可视化分析显示，该方法会优先调度处于拥堵区域的 agent，并战略性地让边界 agent “后退”，以疏通通道。
3. **具备强大的零样本泛化能力**：
   - 泛化至不同 agent 数量（N ≠ 120）；
   - 不同规划时域（w ≠ 20）；
   - 未见的地图变体（如 aisle 更长/更短、inbound/outbound 位置交换等）。
4. **可恢复由次优策略引起的拥堵状态**：即使前段使用随机 PP 导致严重拥堵，切换至 RL-RH-PP 后仍能快速恢复并提升 throughput。

### 方法的局限性
- **依赖绝对位置嵌入**：当前编码器使用基于坐标的 learnable position embeddings，因此无法直接迁移到尺寸不同的地图（non-zero-shot for map size change）。
- **Top-K 评估串行执行**：目前 Top-K 的路径评估是在 CPU 上串行完成，成为 K 较大时的性能瓶颈。
- **未联合优化任务分配**：仅优化路径规划，未与 task assignment 联合建模。

### 未来工作方向
- **实现全地图无关（map-agnostic）表示**：探索相对坐标或拓扑感知编码，提升跨地图泛化能力。
- **并行化 Top-K 评估**：利用多线程或 GPU 加速多个候选顺序的路径求解过程。
- **扩展至联合任务与路径规划**：让 autoregressive decoder 输出 `(agent, task)` 序列，统一解决 task assignment 与 path planning。
- **增强鲁棒性**：纳入对延迟、传感器噪声等不确定性因素的建模。
- **探索 one-shot MAPF 应用**：将本框架迁移至静态 MAPF 场景，验证其通用性。

> 🔗 开源地址：https://github.com/MikeZheng777/RL-RH-PP

</details>

---

### 7. [Lightweight Fairness for LLM-Based Recommendations via Kernelized Projection and Gated Adapters](https://arxiv.org/abs/2603.23780)

**Authors**: Nan Cui, Wendy Hui Wang, Yue Ning  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23780v1  

#### Abstract
Large Language Models (LLMs) have introduced new capabilities to recommender systems, enabling dynamic, context-aware, and conversational recommendations. However, LLM-based recommender systems inherit and may amplify social biases embedded in their pre-training data, especially when demographic cue...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Lightweight Fairness for LLM-Based Recommendations via Kernelized Projection and Gated Adapters*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **LLM-based Recommender Systems (RecLLMs)** 在推荐任务中表现出色，但由于其预训练数据中嵌入的社会偏见（如性别、年龄、职业等），容易在生成推荐时放大对特定人群的不公平性。
- 现有公平性方法存在以下问题：
  - 需要额外可训练参数（如 UP5 使用对抗训练）；
  - 优化不稳定，对超参数敏感；
  - 多属性去偏时易造成任务性能显著下降。

### 🚀 提出的新方法
本文提出一种**轻量级、无需额外训练周期**的去偏框架，结合两种核心技术：

#### （1）Kernelized Iterative Null-space Projection (RFF-INLP)
- 将传统的线性 INLP 扩展到非线性空间，通过 **Random Fourier Features (RFF)** 显式提升表示空间维度，使非线性偏见信号变得线性可分。
- 引入 **isotropic Gaussian perturbation** 增强投影鲁棒性，防止微调导致的表示漂移。
- 最终得到一个**闭式解（closed-form）的正交投影矩阵 $P$**，直接作用于 LLM 的 sequence-level 表示，去除敏感属性信息。
- **优势**：无梯度传播、无额外可训练参数、计算开销极小。

#### （2）Two-level Gated Mixture-of-Experts (MoE) Adapter
为解决多属性联合去偏带来的性能损失，设计了一个两层门控适配器结构：
- **Level-1 Gate（属性加权门）**：
  - 输入上下文向量，输出每个敏感属性对应的软权重 $\alpha_k$；
  - 动态控制各属性专属投影器的融合强度，实现“按需去偏”。
- **Level-2 Gate（专家恢复门）**：
  - 每个属性关联一个低秩 LoRA 专家（rank-$r$）；
  - 使用 sigmoid 门控 $p_k$ 控制是否注入该专家更新，仅在不引入偏见的前提下修复有用信号。
- 构成“**erase-then-repair**”流程：先用投影擦除偏见，再用门控 MoE 选择性恢复任务能力。

### 🔍 相比现有方法的优势
| 维度 | 本方法 | 现有方法（如 UP5） |
|------|--------|------------------|
| 可训练参数 | 仅 MoE 中少量新增（$O(Kdr)$） | 大量对抗模块参数 |
| 优化稳定性 | 高（投影为闭式解，无对抗训练） | 低（依赖梯度反转，易震荡） |
| 多属性处理 | 支持动态调节，避免过度去偏 | 固定策略，易损 utility |
| 推理延迟 | 几乎无增加（投影为矩阵乘法） | 显著增加 |

---

## 2. 核心实验方法和设置

### 📚 数据集
使用两个真实世界数据集进行验证：

| 数据集 | 类型 | 敏感属性 | 用途 |
|-------|------|----------|------|
| **MovieLens-1M** | 电影评分数据 | Gender (2类), Age (7类), Occupation (21类) | 序贯推荐 & 直接推荐 |
| **Insurance Dataset**（非洲保险公司客户数据） | 保险产品交互 | Marital Status (8类), Age (5类), Occupation (6类) | 直接推荐（缺乏时间戳） |

> 注：所有用户-项目交互被转换为自然语言 prompt 输入 LLM。

### ⚙️ 实验设置
- **Backbone 模型**：冻结的 `Instruct Llama-3.2-1B` 模型作为基础 RecLLM。
- **Fine-tuning 方式**：采用 **LoRA** 进行参数高效微调（rank=32）。
- **去偏组件配置**：
  - RFF 维度 $D = 4096$
  - 投影噪声尺度 $\eta = 0.05$
  - MoE 专家 rank = 8
- **公平性阈值**：当 Counterfactual Leakage Gap (AcL) > $T=0.01$ 时触发投影更新。

### 📊 评估指标
| 指标类型 | 名称 | 定义 |
|--------|------|------|
| **Utility（效用）** | Hit@{1,3,10} | Top-k 推荐命中率（越高越好） |
| **Fairness（公平性）** | AcL (Counterfactual Leakage Gap) | 基于 MLP 探针预测敏感属性的 AUC 与 0.5 的平均偏差（越低越好，理想为 0） |

### 🆚 基线方法
- **LLaRA**：原始 RecLLM 框架，无任何去偏机制。
- **P5**：基于 prompt engineering 的推荐范式。
- **UP5**：当前最先进的公平性增强版本 P5，使用对抗训练保护用户侧公平。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 2 & 3）

#### ✅ MovieLens - 序贯推荐（Table 2）
| 属性组合 | 方法 | Hit@1 | Hit@3 | Hit@10 | AcL ↓ |
|--------|------|-------|-------|--------|--------|
| G+A+O | Ours | **56.08** | **72.28** | **81.67** | **0.17** |
| G+A+O | UP5 | 56.08 | 72.28 | 81.67 | 3.21 |

> ➤ 我们的方法在保持与 UP5 相当甚至更优的效用的同时，将平均 AcL 从 3.21% 降至 **0.17%**！

#### ✅ MovieLens - 直接推荐
| 属性 | 方法 | Hit@1 ↑ | AcL ↓ |
|------|------|---------|--------|
| Gender | Ours | **36.92** | **0.60** |
| Gender | UP5 | 16.38 | 4.19 |
| Age | Ours | **37.14** | **0.00** |
| Age | UP5 | 21.22 | 2.91 |

> ➤ 在单属性上，Hit@1 提升超过 **20个百分点**，同时 AcL 接近零。

#### ✅ Insurance Dataset - 直接推荐
| 属性组合 | 方法 | Hit@1 ↑ | AcL ↓ |
|--------|------|---------|--------|
| M+A+O | Ours | 57.37 | **0.13** |
| M+A+O | UP5 | 81.63 | 0.74 |

> ➤ 虽然 UP5 在 Hit@1 上略高，但其 AcL 是 ours 的 **5.7倍**，说明牺牲了公平性换取精度。

### 🔬 消融实验（文中未显式列出表格，但从分析可推断）
- **RFF 提升效果**：相比标准线性 INLP，RFF 显著降低残余偏见（尤其对非线性泄露）。
- **双门控设计必要性**：
  - 若仅使用 Level-1 门控（软投影），仍会丢失部分任务相关特征；
  - 加入 Level-2 LoRA 专家后，能有效恢复性能而不反弹偏见。
- **投影更新频率**：实验证明每轮训练最多更新 3 次即可收敛，表明方法高效稳定。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Kernelized INLP 可有效去除非线性偏见**：
   - 利用 RFF 将隐含在高阶交互中的偏见显式暴露，并通过闭式投影消除。
   - 无需反向传播，真正实现“zero optimization cost”的去偏。

2. **Gated MoE 实现“精准外科手术式修复”**：
   - Level-1 Gate 实现多属性差异化去偏强度控制；
   - Level-2 Gate 在保证不重新引入偏见的前提下，选择性恢复任务性能。

3. **性能-公平性帕累托前沿突破**：
   - 在多个数据集上实现了 **AcL ≈ 0** 的近乎完美 counterfactual fairness；
   - 同时达到或超越 SOTA 方法的推荐准确率（Hit@k）。

4. **轻量化设计实用性强**：
   - 新增参数极少（仅 MoE 中 LoRA 部分）；
   - 不改变原有训练流程，易于集成进现有 RecLLM 系统。

### ⚠️ 方法的局限性
- **依赖用户交互历史**：去偏基于 sequence-level representation，若用户行为稀疏或缺失，则难以提取有效上下文。
- **仅针对已知敏感属性**：需要预先标注敏感属性用于训练探针，无法应对未知或隐式偏见源。
- **假设线性/核线性可分离**：虽然 RFF 提升了表达力，但仍可能遗漏极端复杂的非线性偏见模式。

### 🔮 未来工作方向
- 探索**不依赖用户历史**的去偏方法（如利用群体统计信息或合成反事实样本）。
- 扩展至**零样本或多模态场景**下的公平性保障。
- 结合 causal reasoning 进一步提升 counterfactual fairness 的理论保证。
- 将该框架推广至其他 LLM 下游任务（如对话系统、文本生成）中的公平性控制。

---

> 💡 **一句话总结**：  
> 本文提出了一种**无需额外训练、闭式求解、支持多属性动态调节**的轻量级去偏方法，通过 **Kernelized INLP + Gated MoE Adapter** 的“擦除-修复”架构，在几乎不牺牲推荐性能的前提下，实现了接近理想的 counterfactual fairness，为构建高效且公正的 RecLLM 提供了新范式。

</details>

---

### 8. [On Gossip Algorithms for Machine Learning with Pairwise Objectives](https://arxiv.org/abs/2603.24128)

**Authors**: Igor Colin (LTCI, S2A, IP Paris), Aur\'elien Bellet (PREMEDICAL), Stephan Cl\'emen\c{c}on (LTCI, IDS, S2A, IP Paris), Joseph Salmon (IROKO, UM)  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.24128v1  

#### Abstract
In the IoT era, information is more and more frequently picked up by connected smart sensors with increasing, though limited, storage, communication and computation abilities. Whether due to privacy constraints or to the structure of the distributed system, the development of statistical learning me...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：On Gossip Algorithms for Machine Learning with Pairwise Objectives

## 1. 论文的主要贡献和创新点

### 解决的问题
本文聚焦于**分布式机器学习中的成对目标函数（pairwise objectives）优化问题**。在物联网（IoT）和边缘计算场景下，数据通常分散在多个节点上，且由于隐私或通信限制无法集中处理。传统的 gossip 算法大多针对可分离目标函数（如均值），即 $ f(\theta; X_1,\dots,X_n) = \frac{1}{n}\sum_{i=1}^n f_i(\theta; X_i) $。然而，许多重要任务（如排序、聚类、度量学习）的目标函数是**成对形式**的，例如 U-statistic of degree two：

$$
U_n(h) = \frac{2}{n(n-1)}\sum_{1\leq i<j\leq n} h(X_i, X_j)
$$

这类函数依赖于所有数据对的交互，因此不能直接应用标准的 gossip 平均算法。

### 提出的新方法与新思路
作者系统地研究并改进了适用于成对目标的 gossip 算法，其核心贡献如下：

- **GoSTA 算法的完整非渐近分析**：首次为 GoSTA（Gossip-based Stochastic Averaging for U-statistics）算法提供了完整的期望和方差界，弥补了之前仅分析偏差的不足。
  
- **去中心化优化的收敛保证**：在基于 dual averaging 的分布式优化框架中，证明了由辅助观测传播引起的梯度偏差（bias）会随着图的混合性质而衰减，并最终消失，从而建立了严格的收敛保证。

- **新颖的下界分析**：扩展了 Scaman et al. (2018) 的下界构造，首次为成对目标的去中心化优化推导出匹配的下界，揭示了网络拓扑中“平均两跳距离”（averaged two-hop distance）$\Delta$ 是影响收敛速度的关键因素。

### 相比现有方法的优势
| 方面 | 本文工作 | 以往工作 |
|------|--------|--------|
| **理论完整性** | 提供了估计与优化的完整非渐近误差界（含方差） | 多数仅提供偏差分析或渐近结果 |
| **收敛性保证** | 严格证明梯度偏差随迭代衰减至零 | 部分工作假设偏差收敛但未证明 |
| **下界匹配** | 给出了与上界匹配的成对优化下界 | 缺乏针对成对目标的下界分析 |

---

## 2. 核心实验方法和设置

### 数据集
实验在 **Breast Cancer Wisconsin (Original)** 数据集上进行，该数据集包含：
- 样本数量 $ n = 699 $
- 特征维度 $ d = 11 $
- 每个节点存储一个样本（single point per node）

### 实验设置
- **任务**：最大化 **AUC (Area Under the ROC Curve)**，这是一个典型的成对目标函数。
- **模型**：线性评分函数 $ x \mapsto x^\top\theta $。
- **损失函数**：使用 logistic pairwise loss 作为 AUC 的凸代理：
  $$
  R_n(\theta) = \frac{1}{n^2}\sum_{i,j} \mathbf{1}\{l_i > l_j\} \log(1 + \exp((x_j - x_i)^\top\theta))
  $$
- **算法**：实现并测试了 **Algorithm 2**（同步）和 **Algorithm 7**（异步）版本的 gossip dual averaging。
- **步长策略**：$ \gamma(t) = 10^{-3}/\sqrt{t} $

### 网络拓扑（Graph Topologies）
为了研究通信结构的影响，设计了三种不同连通性的图：
1. **Complete Graph**：全连接图，理想情况，谱隙最大。
2. **2D Grid**：二维网格，连通性差，直径大。
3. **Watts-Strogatz Graph**：小世界网络，介于规则与随机之间，参数 $ k=5, p=0.3 $。

### 评估指标
- **目标函数演化**：各节点局部目标值的平均与标准差。
- **偏差项（Bias Term）**：$ \| \mathbb{E}[e(t)] \| $，衡量梯度估计的偏差大小。
- **共识损失（Consensus Loss）**：节点间参数差异，反映网络一致性。

### 基线方法对比
文中未明确列出其他成对 gossip 方法作为基线（因相关工作稀少），而是通过在同一框架下比较不同**网络拓扑**的表现来验证方法的有效性和理论预测。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
#### (a) 目标函数演化（异步设置）
- **收敛速度**：`Complete Graph` 和 `Watts-Strogatz` 明显快于 `Grid`。
- **共识性**：高连通图（Complete, WS）的节点间标准差更小，表明更快达成一致。
- **结论**：网络连通性显著影响收敛速度，验证了谱隙（spectral gap）在理论界中的作用。

#### (b) 偏差项演化（Bias Term）
- 所有拓扑下的偏差项 $ \|e(t)\omega(t)\| $ 均**快速收敛至接近零**。
- 在整个优化过程中，偏差项的量级始终比目标函数值低几个数量级。
- **结论**：实证支持了理论分析中“偏差迅速衰减”的结论，解释了算法良好的实际表现。

### 消融实验
本文虽无传统意义上的消融实验，但通过以下方式进行了分析：
- **不同图结构的对比**：本质上是对“网络拓扑”这一关键变量的消融，验证了谱隙和 $\Delta$ 的影响。
- **同步 vs 异步**：附录中给出了异步版本的实现与分析，显示其收敛略慢但依然有效。

---

## 4. 关键结论和发现

### 主要发现
1. **理论完备性**：首次为 gossip-based pairwise estimation 和 optimization 提供了完整的非渐近收敛保证，包括期望、方差及偏差衰减的量化分析。
2. **偏差可忽略**：尽管早期梯度估计存在偏差，但由于图上的随机游走具有快速混合性（fast mixing），该偏差以几何速率衰减，在有限时间内变得可忽略。
3. **图属性决定效率**：网络的**谱隙（spectral gap）** 和 **平均两跳距离 $\Delta$** 是决定算法效率的核心图论属性。更好的连通性（更大的谱隙）带来更快的收敛。
4. **下界匹配**：推导的上界与下界在 $\Delta$ 上匹配，表明当前算法在最坏情况下已达到最优。

### 方法的局限性
- **单点每节点假设**：默认每个节点只持有一个数据点，虽然第6.1节讨论了多点扩展，但未深入实验。
- **凸性假设**：理论分析依赖于目标函数的凸性，对非凸问题（如深度学习）的适用性需进一步研究。
- **通信开销**：虽然 gossip 本身通信高效，但频繁交换辅助观测仍可能在带宽受限场景下成为瓶颈。

### 未来工作方向
1. **多点每节点（Multiple Points per Node）**：将方法扩展到每个节点持有 $k>1$ 个样本的更现实场景，并优化通信策略。
2. **差分隐私（Differential Privacy）**：结合本地差分隐私机制，在交换数据时加入噪声以保护原始数据隐私。
3. **鲁棒性与公平性**：将鲁棒统计和公平性约束（常以 U-statistic 形式表达）融入框架，实现负责任的去中心化学习。
4. **非凸优化**：探索在非凸成对目标（如深度排序模型）上的收敛行为与算法设计。

</details>

---

### 9. [Decentralized Task Scheduling in Distributed Systems: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/2603.24738)

**Authors**: Daniel Benniah John  
**Category**: cs.DC  
**Published**: 2026-03-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.24738v1  

#### Abstract
Efficient task scheduling in large-scale distributed systems presents significant challenges due to dynamic workloads, heterogeneous resources, and competing quality-of-service requirements. Traditional centralized approaches face scalability limitations and single points of failure, while classical...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Decentralized Task Scheduling in Distributed Systems: A Deep Reinforcement Learning Approach

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大规模分布式系统（如云-边协同计算环境）面临**动态负载、资源异构性和多样化QoS需求**带来的任务调度挑战。传统集中式调度存在**可扩展性差、单点故障风险高**等问题；而经典启发式算法（如 FCFS、SJF）缺乏对动态变化的适应能力。此外，现有的基于 DRL 的调度方法多依赖**中心化控制器**和**重型深度学习框架**（如 TensorFlow/PyTorch），难以部署在资源受限的边缘设备上。

本论文旨在解决以下核心问题：
- 如何实现**去中心化、自适应且高效的任务调度机制**？
- 如何在保证性能的同时，使模型轻量化以支持**边缘设备部署**？

---

### 🚀 提出的新方法与创新思路

作者提出了一种 **Decentralized Multi-Agent Deep Reinforcement Learning (DRL-MADRL)** 框架，用于异构分布式系统的任务调度，并做出多项关键创新：

#### （1）**Dec-POMDP 建模**
将任务调度问题形式化为一个 **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**，准确刻画了：
- 局部可观测性（每个节点仅观察本地状态）
- 多智能体并发决策
- 随机状态转移（任务到达、执行时间不确定）
- 无需全局同步或显式通信

#### （2）**轻量级 Actor-Critic 架构（NumPy-only 实现）**
- 完全使用 **NumPy、Matplotlib 和 SciPy** 实现神经网络，**不依赖任何重型 DNN 框架**（如 PyTorch/TensorFlow）
- 采用简单的前馈网络（无 RNN、Attention 等复杂模块）
- 单个 agent 内存占用约 **100 KB**，推理延迟 <10 ms（CPU 上）

> ✅ 这使得该方法可直接部署于 **IoT 网关、嵌入式设备等资源受限平台**

#### （3）**优先级感知的动作选择机制（Priority-aware Action Selection）**
结合 Google Cluster Trace 中的生产任务分类（Production/Batch/Best-effort），设计混合策略：
- 融合 DRL 学习到的偏好
- 显式引入任务优先级评分与资源匹配度
- 确保高优先级任务获得及时调度，提升 SLA 满足率

#### （4）**精细化能量建模与解释**
构建了一个基于线性功耗模型的综合能耗公式：
$$
P_i(t) = P_{idle,i} + P_{dyn,i} \times u_i(t)
$$
并通过实验揭示了“低总能耗 ≠ 高能效”的陷阱——某些基线因拒绝大量任务导致能耗虚低。

#### （5）**开源可复现性保障**
- 所有代码、数据、脚本公开于 GitHub
- 提供确定性随机种子，可在普通笔记本电脑上 **4 分钟内复现实验结果**

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（FCFS/SJF/Min-Min） | 中心化 DRL 方法 | 本文 DRL-MADRL |
|------|-------------------------------|------------------|----------------|
| 可扩展性 | 差（尤其 Min-Min） | 差（需全局状态） | ✅ 强（完全去中心） |
| 故障容忍 | 单点失败即崩溃 | 同左 | ✅ 分布式容错 |
| 自适应性 | 固定规则，无法学习 | 可学习但集中训练 | ✅ 分布式在线学习 |
| 部署成本 | 低 | 高（GPU + 大内存） | ✅ 极低（仅 NumPy） |
| 能效分析 | 忽视实际完成情况 | 缺乏细粒度建模 | ✅ 明确区分“节能”与“拒载” |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用 **Google Cluster Trace v3** 公开数据集进行统计建模
- 并非直接回放 trace，而是提取其分布特征生成合成任务流，确保实验可控且可重复

#### 关键 workload 特征建模如下：
| 属性 | 分布模型 | 参数说明 |
|------|----------|-----------|
| 执行时长 $t_j$ | Pareto 分布 | $\alpha=1.5, t_{min}=5s$（重尾特性） |
| CPU 需求 $cpu_j$ | Log-Normal | $\mu=0.5, \sigma=0.8$ |
| 内存需求 $mem_j$ | Log-Normal | $\mu=2.0, \sigma=1.0$ |
| 到达间隔 | Poisson 过程 | $\lambda=0.5$ task/s |
| 优先级类别 | 三类比例 | Production: 25%, Batch: 60%, Best-effort: 15% |
| 截止时间 $d_j$ | 与优先级相关 | Production: $a_j + 1.5t_j$, Batch: $+3.0t_j$, Best-effort: $+5.0t_j$ |

---

### ⚙️ 实验设置

| 项目 | 设置详情 |
|------|---------|
| 节点规模 | 100 个异构节点（模拟真实云-边混合架构） |
| 节点分层 | High-capacity (20%) / Medium (50%) / Low (30%) |
| 每轮任务数 | 1,000 tasks per episode |
| 总实验次数 | 30 次独立运行（不同随机种子） |
| 评估周期 | 取最后 10 轮（episodes 21–30）平均值（已收敛） |
| 硬件平台 | 普通笔记本（Intel i5, 8GB RAM, 无 GPU） |

---

### 🎯 评估指标

| 指标 | 描述 |
|------|------|
| **ATCT** | Average Task Completion Time（平均任务完成时间） |
| **Total Energy Consumption** | 总能耗（kWh），含 idle 与 dynamic 功耗 |
| **SLA Satisfaction Rate** | 在截止时间内完成的任务占比 |
| **Throughput** | 单位时间完成任务数（tasks/1000s） |
| **Load Balance** | 节点利用率方差的倒数（越高越均衡） |
| **Statistical Significance** | 两样本 t-test，显著性水平 $p < 0.05$ |

---

### 🆚 基线方法对比

| 基线方法 | 简要描述 |
|--------|--------|
| **Random** | 在可行节点中均匀随机分配 |
| **Weighted Round-Robin (W-RR)** | 按节点容量加权循环调度 |
| **Priority-aware Min-Min** | 优先级优先 + 最小负载节点分配（强启发式基线） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（Table I）

| Scheduler | ATCT (s) | Energy (kWh) | SLA (%) | Throughput (tasks/1000s) | Completed Tasks |
|----------|----------|-------------|---------|--------------------------|----------------|
| Random | 36.5 | 878.3 | 75.5 | 407.27 | 998 |
| W-RR | 36.2 | 1007.1 | 76.1 | 405.00 | 996 |
| Priority-MinMin | 36.1 | **155.3** | 47.3 | 105.47 | 280 |
| **DRL-MADRL (Ours)** | **30.8** | **745.2** | **82.3** | **425.15** | **999** |

> 注：Priority-MinMin 的低能耗源于仅完成 28% 的任务，属虚假节能

---

### 📊 对比提升效果

| 指标 | 相比 Random 提升 | 相比 W-RR 提升 |
|------|------------------|----------------|
| **ATCT ↓** | **-15.6%** (36.5 → 30.8s) | -14.9% |
| **Energy ↓** | **-15.2%** (878.3 → 745.2 kWh) | -26.0% |
| **SLA ↑** | **+6.8 pp** (75.5% → 82.3%) | +6.2 pp |
| **Throughput ↑** | +4.4% | +5.0% |

✅ 所有改进均具有高度统计显著性：**p < 0.001**（经 Bonferroni 校正）

---

### 🔍 消融实验与学习动态分析（Ablation & Convergence）

- **学习曲线显示**：从初始 48s 的随机策略，经过 30 轮训练后稳定在 30.8s，**相对自身提升约 35.8%**
- **快速学习阶段**：第 11–20 轮性能跃升，得益于 **Prioritized Experience Replay**
- **稳定性好**：最终 10 轮波动极小，表明策略已收敛
- **消融验证**：移除 priority-aware 模块会导致 SLA 下降超 30 个百分点，证明其必要性

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **DRL-MADRL 显著优于传统启发式方法**  
   在 ATCT、Energy、SLA 等多个目标上实现 **一致且显著的提升**，验证了 MARL 在分布式调度中的有效性。

2. **去中心化 + 轻量化是边缘部署的关键路径**  
   仅用 NumPy 实现的轻量级架构，在保持高性能的同时极大降低了部署门槛，适用于 **IoT/Edge 场景**。

3. **必须联合优化多目标，避免“伪节能”陷阱**  
   实验揭示：单纯追求低能耗可能导致任务拒接率上升。因此应结合 **completion rate、throughput** 来评估真实能效。

4. **优先级机制对 SLA 至关重要**  
   生产级系统中，必须显式建模任务优先级，否则即使平均性能好也可能违反 SLA。

---

### ⚠️ 局限性（Limitations）

| 限制 | 说明 |
|------|------|
| **仿真环境** | 实验基于离散事件模拟，未在真实集群中部署，可能忽略网络抖动、部分失效等现实因素 |
| **合成 workload** | 尽管遵循 Google Trace 统计规律，但仍为采样生成，可能遗漏真实 trace 中的时序依赖模式 |
| **规模有限** | 当前实验为 100 节点，远小于超大规模数据中心（>10k 节点），扩展性有待进一步验证 |
| **忽略任务依赖** | 假设所有任务相互独立，未考虑 DAG 结构、数据局部性或通信开销，限制了在复杂应用中的适用性 |

---

### 🔮 未来研究方向

1. **真实环境部署验证**  
   在 Edge Testbed 或私有云平台上部署原型系统，收集真实 workload 表现。

2. **超大规模系统的层次化协调机制**  
   设计 hierarchical 或 federated coordination 架构，以支持万级节点调度。

3. **支持任务图（Task Graphs）调度**  
   扩展框架以处理具有 precedence constraints、data dependencies 的 DAG 任务。

4. **联邦学习集成（Federated Learning）**  
   多个地理分布集群共享调度经验而不泄露本地数据，兼顾隐私与协作效率。

5. **引入 workload 预测模块**  
   利用 time-series forecasting 实现 proactive scheduling，提前调配资源应对高峰。

---

## 总结

本文提出了一种**去中心化、轻量化、可复现的多智能体深度强化学习调度框架 DRL-MADRL**，成功解决了传统调度方法在可扩展性、适应性和部署成本上的瓶颈。通过严谨建模、精细奖励设计和极致轻量实现，在 100 节点异构系统上实现了 **15.6% 的完成时间缩短、15.2% 的节能增益和 82.3% 的 SLA 满足率**，所有结果均具有高度统计显著性。

更重要的是，该项目完全开源并提供完整复现路径，推动了 AI for Systems 领域的透明化与可验证研究发展。

🔗 开源地址：[https://github.com/danielbenniah/marl-distributed-scheduling](https://github.com/danielbenniah/marl-distributed-scheduling)

</details>

---

### 10. [Implicit Turn-Wise Policy Optimization for Proactive User-LLM Interaction](https://arxiv.org/abs/2603.23550)

**Authors**: Haoyu Wang, Yuxin Chen, Liang Luo, Buyun Zhang, Ellie Dingqiao Wen, Pan Li  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.23550v1  

#### Abstract
Multi-turn human-AI collaboration is fundamental to deploying interactive services such as adaptive tutoring, conversational recommendation, and professional consultation. However, optimizing these interactions via reinforcement learning is hindered by the sparsity of verifiable intermediate rewards...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Implicit Turn-Wise Policy Optimization for Proactive User-LLM Interaction》论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对 **multi-turn User-LLM interaction**（多轮人机交互）中的 **reward sparsity**（奖励稀疏性）问题展开研究。在长周期对话任务中，如数学辅导、医疗咨询等，最终的 outcome reward（结果奖励）通常只在对话结束时才能获得，导致强化学习（RL）训练过程中样本效率低下，且容易收敛到错误策略。

此外，现有的 **Process Reward Models (PRMs)** 多为 token-level 细粒度奖励建模，存在以下问题：
- **高方差（high variance）**：单个 token 的奖励信号不稳定。
- **语义不一致（semantic misalignment）**：难以解释。
- **过拟合风险（overfitting）**：细粒度监督易受噪声影响。

### **提出了什么新方法或新思路**

作者提出了一种名为 **Implicit Turn-wise Policy Optimization (ITPO)** 的新框架，其核心思想是：
- 利用 **implicit PRM** 从 outcome reward 中反向推导出 **turn-wise process rewards**（回合级过程奖励），而非传统的 token-level 奖励。
- 引入 **Normalization 机制（Norm-ITPO）**，将 turn-level 隐式奖励通过 Softmax 归一化后按比例分配全局 outcome reward，从而稳定训练动态。

#### **核心创新点**
1. **Turn-wise 而非 Token-wise 奖励建模**  
   将每个对话回合视为一个语义单元，聚合该回合内所有 token 的隐式奖励，形成更鲁棒、可解释的 turn-level 优势信号。

2. **Normalization 机制提升稳定性**  
   通过 $ R_k = w_k \cdot R $ 分配奖励，其中 $ w_k = \text{Softmax}(R_k^{\text{imp}} / \eta) $，确保隐式奖励与真实 outcome 之间的尺度一致性，防止价值函数漂移。

3. **Bayesian Interpretation**  
   将归一化过程解释为对“关键回合”（pivotal turn）的后验概率估计，赋予奖励分配更强的理论基础。

4. **与主流 RL 算法兼容**  
   可无缝集成于 **PPO, GRPO, RLOO** 等策略优化器中，实现 turn-level 的 clipped surrogate objective 更新。

### **相比现有方法的优势**

| 方法 | 局限性 | ITPO 的改进 |
|------|--------|-------------|
| **Outcome-only (Trajectory-Share)** | 忽略对话内部结构，所有 token 平均分奖 | 显式建模回合重要性，实现信用分配 |
| **Token-level implicit PRM (e.g., PRIME)** | 高方差、难收敛、语义混乱 | 聚合为 turn-level，显著降低噪声 |
| **LLM-as-a-Judge** | 推理延迟高，评估偏见严重 | 在线学习，无需外部模型，成本低 |
| **Monte Carlo Rollouts** | 样本复杂度高（8–10倍） | 仅需 outcome label，在线更新 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

论文在三个典型的多轮协作任务上进行验证：

| 任务 | 数据集 | 描述 |
|------|--------|------|
| **Math Tutoring** | MATH dataset (500 problems) | 学生提问模糊，需要 LLM 主动追问以获取完整信息 |
| **Document Writing** | Medium 文章摘要（500篇） | 用户给出高层摘要，LLM 迭代撰写草稿，目标是生成符合意图的文档 |
| **Medical Recommendation** | MTMedDialog (550 samples) | 模拟医生问诊，患者逐步提供症状，LLM 需诊断并推荐治疗方案 |

> 所有任务均使用 **LLM-based user simulator** 模拟真实用户行为，遵循 POMDP 框架。

### **实验设置和评估指标**

#### **模型配置**
- **Policy Model**: Qwen2.5-3B-Instruct（主实验）、Qwen2.5-7B、Qwen3-4B（泛化性测试）
- **User Simulator & Judge**: Qwen2.5-14B-Instruct
- **Training Framework**: VeRL + vLLM + FSDP + RAY 分布式系统

#### **评估指标**
| 任务 | 主要指标 | 辅助指标 |
|------|----------|-----------|
| Math Tutoring | **Accuracy (ACC)** | Token count ↓ |
| Document Writing | **BLEU Score** | Token count ↓ |
| Medical Recommendation | **Diagnosis Score**（0–10分制） | Token count ↓ |

> 最终得分综合考虑任务表现与长度正则项：$ S = R - \gamma \times N $

#### **优势估计器（Advantage Estimators）**
- **PPO**: 使用 critic model + GAE
- **GRPO**: 基于组内统计标准化奖励
- **RLOO**: Leave-one-out 方差缩减

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Table 1）**

| 方法 | Math Tutoring (ACC↑) | Doc Writing (BLEU↑) | Med Rec (Score↑) |
|------|------------------------|--------------------|------------------|
| **RLOO + Trajectory-Share** | 29.06 | 37.35 | 65.12 |
| **RLOO + LLM-as-Judge** | 28.87 | 42.34 | 66.77 |
| **RLOO + PRIME (token-level)** | 29.75 | 40.95 | 61.42 |
| **RLOO + ITPO (Ours)** | 29.06 | 44.59 | 68.43 |
| **RLOO + Norm-ITPO (Ours)** | **32.50** | **44.83** | **69.24** |

> ✅ **Norm-ITPO 在三项任务上均达到最优性能**

#### **相对提升幅度**
- **vs. Trajectory-Share**: +34.4% (Math), +12.0% (Doc), +8.0% (Med)
- **vs. PRIME**: +9.2% (Math), +9.5% (Doc), +12.8% (Med)

### **与基线方法的对比结果**

- **Norm-ITPO > ITPO > PRIME ≈ LLM-as-Judge > Uniform Decompose > Trajectory-Share**
- **Norm-ITPO 在 PPO 设置下优势最明显**：说明其 normalization 机制有效缓解了 value model 训练中的非平稳目标问题（non-stationary target）。
- **ITPO 在 token-level 方法中表现最佳**，证明 turn-wise aggregation 显著优于 token-wise 监督。

### **消融实验结果**

#### （1）**Turn-wise vs. Token-wise 奖励稳定性分析（Fig. 3 & Fig. 6）**
- **Turn-wise 偏好快速收敛**：Spearman 相关系数在约 100 步内趋于稳定。
- **Token-level 排序波动剧烈**：相同 token 在不同训练阶段奖励差异大（见 Fig. 2 红框）。
- **ITPO 与 Norm-ITPO 高度自洽**：两者 turn-level 偏好结构强相关（>0.9），表明学习到的是内在一致的语义偏好。

#### （2）**Normalization 对尺度稳定性的影响（Fig. 7）**
- 未归一化的 ITPO 其隐式奖励总和 $ R_s(T) $ 与 outcome $ R $ 的映射斜率波动剧烈。
- Norm-ITPO 通过强制 $ \sum w_k = 1 $ 和 $ R_k = w_k \cdot R $，实现了稳定的回归目标，极大提升了 critic model 收敛性。

#### （3）**人类可解释性评测（Semantic Interpretability）**
- 在 64 个决策点（32 best + 32 worst turns）上，由三位专家标注“最佳/最差”回合。
- **Norm-ITPO 匹配 48/64（75%）**，**ITPO 匹配 47/64（73.4%）**
- 显著优于随机猜测（期望 16/64），接近 Gemini-3.0-Pro（58/64）

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Turn-wise 奖励建模优于 Token-wise**  
   在 multi-turn interaction 中，回合是更自然的语义单位，聚合 token-level 信号能显著降低方差、提升可解释性和训练稳定性。

2. ✅ **Normalization 是关键设计**  
   单纯的 turn-wise aggregation（ITPO）虽有改进，但隐式奖励的尺度漂移仍会影响策略优化；引入归一化机制（Norm-ITPO）后，性能进一步跃升，尤其在依赖 critic model 的 PPO 中效果显著。

3. ✅ **ITPO 可泛化至多种 RL 框架**  
   与 PPO, GRPO, RLOO 结合均能带来一致增益，说明其作为“奖励分配模块”的通用性。

4. ✅ **学习到的偏好符合人类判断**  
   轨迹分析显示，ITPO 能正确识别早期澄清性提问的重要性（如 Math Tutoring 中首次追问），与人类标注高度一致。

### **方法的局限性**

- **依赖高质量 outcome reward**：若最终 outcome label 不准确，隐式奖励传播会放大误差。
- **无法处理完全无 outcome 的场景**：必须至少有一个终端反馈信号。
- **turn boundary 定义依赖对话结构**：对于非回合式连续交互可能不适用。
- **计算开销略高于纯 outcome 方法**：需额外训练 implicit PRM。

### **未来工作方向**

1. **扩展至多智能体协作（multi-agent）**：将 ITPO 应用于 agent-to-agent 协同推理。
2. **结合 offline RL**：利用历史对话数据预训练 implicit PRM，减少在线采样需求。
3. **动态 turn 划分机制**：自动识别语义边界，而非固定每轮一次响应。
4. **跨任务迁移能力探索**：是否可在一个任务上学到的 turn-wise 奖励模式迁移到其他领域？

---

> 🔗 **代码开源地址**：https://github.com/Graph-COM/ITPO

</details>

---

### 11. [Residual Attention Physics-Informed Neural Networks for Robust Multiphysics Simulation of Steady-State Electrothermal Energy Systems](https://arxiv.org/abs/2603.23578)

**Authors**: Yuqing Zhou, Ze Tao, Fujun Liu  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.23578v1  

#### Abstract
Efficient thermal management and precise field prediction are critical for the design of advanced energy systems, including electrohydrodynamic transport, microfluidic energy harvesters, and electrically driven thermal regulators. However, the steady-state simulation of these electrothermal coupled ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Residual Attention Physics-Informed Neural Networks for Robust Multiphysics Simulation of Steady-State Electrothermal Energy Systems

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该研究针对**稳态电热耦合多物理场系统**（steady-state electrothermal coupled multiphysics systems）的数值模拟难题，尤其是以下挑战：
- 强非线性场耦合（如速度、压力、电势、温度之间的相互作用）
- 温度依赖的变系数（temperature-dependent variable coefficients）
- 复杂界面动力学（oblique material interfaces）
- 不同物理场间梯度尺度差异大，导致优化失衡

这些问题使得传统 PINN 在处理复杂能量系统（如微流体能量收集器、电驱动热调节器等）时精度下降甚至发散。

---

### 提出的新方法：RA-PINN
作者提出了一种新型框架——**Residual Attention Physics-Informed Neural Network (RA-PINN)**，其核心创新包括：

#### （1）统一五场算子建模（Unified Five-Field Operator Formulation）
将速度 $u$、压力 $p$、电势 $\phi$、温度 $T$ 和连续性约束统一为一个向量化的 PDE 残差算子 $ \mathcal{N}(U) = 0 $，实现对多物理场的联合求解。

#### （2）残差注意力机制（Residual-Attention Mechanism）
结合了：
- **残差连接（Residual Connection）**：稳定深层特征传播，保留全局背景信息；
- **通道调制注意力（Channel-wise Attention Modulation）**：动态增强携带局部强梯度或界面特征的通道响应。

公式体现为：
$$
z^{(l+1)} = z^{(l)} + (1 + m^{(l)}) \odot t^{(l)}
$$
其中 $m^{(l)}$ 是由注意力分支生成的调制因子，用于放大关键通道。

#### （3）自适应残差点采样（Adaptive Residual-Based Collocation Sampling）
根据当前残差大小动态调整训练点分布：
- 高残差点增加采样密度；
- 低残差点减少或移除；
- 特别强化在界面区域和变系数区域的采样。

这提升了模型在“难拟合”区域的学习能力。

---

### 相比现有方法的优势
| 方面 | RA-PINN 优势 |
|------|-------------|
| **表示能力** | 残差注意力机制能更好捕捉局部陡峭梯度与耦合结构 |
| **稳定性** | 残差连接缓解梯度消失，注意力避免优化偏向平滑区域 |
| **泛化性** | 自适应采样使模型更关注物理敏感区，提升鲁棒性 |
| **适用场景** | 可处理常系数、压力规范（gauge）、变系数、斜界面等多种复杂情形 |

---

## 2. 核心实验方法和设置

### 数据集与基准任务
研究未使用真实世界数据集，而是构建了**四个理想化的电热耦合仿真基准案例**（Benchmark Cases），均定义在单位正方形域 $\Omega = [0,1] \times [0,1]$ 上：

| Case | 描述 |
|------|------|
| **Case 1** | 常系数耦合系统（Constant-Coefficient Coupling） |
| **Case 2** | 压力规范约束（Indirect Pressure-Gauge Constraint）$\int_\Omega p\,d\Omega = 0$ |
| **Case 3** | 温度依赖传输系数（Temperature-Dependent Transport Coefficients）<br>$v(T)=v_0(1+\rho_v T),\ \alpha(T)=\alpha_0(1+\rho_\alpha T)$ |
| **Case 4** | 斜界面一致性（Oblique-Interface Consistency）<br>分界线：$x + 0.4y - 0.7 = 0$，两侧材料参数不同 |

每个案例提供精确解（ground truth）用于定量评估。

---

### 实验设置与评估指标

#### 模型输入输出
- 输入：空间坐标 $(x, y)$
- 输出：五维场预测 $[u, v, p, \phi, T]$

#### 评估指标
- **MSE**（Mean Squared Error）
- **RMSE**（Root Mean Squared Error）
- **MAE**（Mean Absolute Error）
- **Relative L2 Error**（相对 L2 范数误差）

所有误差按字段分别计算，并报告平均值（Avg. row）。

#### 基线方法对比
与以下主流 PINN 架构进行比较：
- **Pure-MLP**：标准全连接网络 + PINN
- **LSTM-PINN**：引入 LSTM 结构以增强序列感知能力
- **pLSTM-PINN**：并行 LSTM 变体，旨在提升多场协调性

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Case | Model | Avg. Relative L2 Error |
|------|-------|------------------------|
| Case 1 | **RA-PINN** | **3.235×10⁻³** |
|       | LSTM-PINN     | 5.695×10⁻³ |
|       | pLSTM-PINN    | 1.205×10⁻² |
|       | Pure-MLP      | 5.105×10⁻² |
|  
| Case 2 | **RA-PINN** | **7.660×10⁻³** |
|       | LSTM-PINN     | 1.038×10⁻² |
|       | pLSTM-PINN    | 3.608×10⁻² |
|       | Pure-MLP      | 4.868×10⁻² |
|  
| Case 3 | **RA-PINN** | **5.065×10⁻³** |
|       | LSTM-PINN     | 7.155×10⁻³ |
|       | pLSTM-PINN    | 8.456×10⁻¹ ⚠️（严重失败） |
|       | Pure-MLP      | 3.031×10⁻² |
|  
| Case 4 | **RA-PINN** | **1.377×10⁻³** |
|       | LSTM-PINN     | 1.449×10⁻³ |
|       | pLSTM-PINN    | 3.895×10⁻² |
|       | Pure-MLP      | 1.061×10⁻² |

> ✅ **RA-PINN 在所有四个案例中均取得最低的平均误差（各项指标最小值加粗）**

---

### 与基线方法的对比结果

| 对比维度 | 结果总结 |
|--------|----------|
| **总体精度** | RA-PINN 在所有案例中达到最高精度，尤其在非线性和界面问题上优势显著 |
| **压力规范问题（Case 2）** | RA-PINN 成功恢复无锚定点的压力场，而其他方法出现明显漂移 |
| **变系数问题（Case 3）** | RA-PINN 显著优于所有基线，pLSTM-PINN 几乎完全失效 |
| **斜界面问题（Case 4）** | RA-PINN 与 LSTM-PINN 接近，但仍保持最优 Avg. 性能，且界面过渡更锐利 |

#### 可视化分析支持
- 图2–图5显示 RA-PINN 的预测场最接近真值，背景噪声最少；
- 其他方法普遍存在模糊、振荡或结构性偏差。

---

### 训练效率（Table 2）

| Case | RA-PINN (h) | 最快方法 |
|------|--------------|---------|
| Case 1 | 24.01 | pLSTM-PINN (1.09 h) |
| Case 2 | 24.67 | Pure-MLP (3.17 h) |
| Case 3 | 39.81 | pLSTM-PINN (4.28 h) |
| Case 4 | 38.35 | pLSTM-PINN (9.30 h) |

> ❗ **RA-PINN 训练时间最长，约为基线的 4–10 倍**

说明：高精度是以更高的计算成本换取的。

---

### 消融实验（隐含于设计分析）
虽然文中未明确列出消融表，但从架构组件分析可推断：
- **残差连接** → 提升训练稳定性与深层信息传递
- **注意力调制** → 改善局部结构识别（如界面、边界层）
- **自适应采样** → 加速收敛并聚焦困难区域

这些模块协同作用，共同构成 RA-PINN 的优越表现。

---

## 4. 关键结论和发现

### 主要发现
1. **RA-PINN 是目前最稳健的稳态电热耦合求解器之一**：
   - 在四种典型复杂场景下均取得最佳综合性能；
   - 尤其擅长处理**强非线性、变系数、间接约束**等问题。

2. **残差注意力机制有效解决了 PINN 中的“优化失衡”问题**：
   - 通过通道级调制，让网络自动关注高梯度区域；
   - 避免某些物理场被压制（如压力场在无锚定情况下仍能准确重建）。

3. **自适应采样显著提升界面与异质区域建模能力**：
   - 动态重采样策略使模型持续聚焦于误差大的区域；
   - 对斜界面问题尤其重要。

4. **精度与效率存在权衡**：
   - RA-PINN 精度领先，但训练耗时远高于轻量级模型；
   - 适用于对精度要求极高的工程仿真，而非实时推理。

---

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **高训练成本** | 当前实现需要数十小时 GPU 时间，限制实际部署 |
| **缺乏硬件加速优化** | 未探索稀疏化、量化或分布式训练策略 |
| **仅限稳态问题** | 尚未扩展到瞬态或多尺度时间演化系统 |
| **依赖高质量残差计算** | 对自动微分精度敏感，可能受网格分辨率影响 |

---

### 未来工作方向
1. **训练加速技术集成**：
   - 引入 curriculum learning 或 warm-start 初始化；
   - 探索基于 Jacobian 分析的智能初值选择。

2. **扩展至瞬态与三维系统**：
   - 将 RA-PINN 推广到 time-dependent electrothermal dynamics；
   - 应用于 3D 微能源器件建模。

3. **嵌入数字孪生系统**：
   - 作为 high-fidelity surrogate model 支持在线监测与控制；
   - 结合 uncertainty quantification 实现可靠性评估。

4. **与其他神经算子融合**：
   - 与 Fourier Neural Operator (FNO) 或 DeepONet 结合，提升泛化能力；
   - 构建“一次训练，多次查询”的通用电热求解器。

---

> 🔚 **总结一句话**：  
> RA-PINN 通过 **residual-connected backbone + attention-guided modulation + adaptive residual sampling** 的三重增强，在稳态电热多物理场模拟中实现了前所未有的精度与鲁棒性，为下一代可持续能源系统的高保真建模提供了强有力的工具。

</details>

---

### 12. [MetaKube: An Experience-Aware LLM Framework for Kubernetes Failure Diagnosis](https://arxiv.org/abs/2603.23580)

**Authors**: Wei Sun, Ting Wang, Xinran Tian, Wanshun Lan, Xuhan Feng, Haoyue Li, Fangxin Wang  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.23580v1  

#### Abstract
Existing LLM-based Kubernetes diagnostic systems cannot learn from operational experience, operating on static knowledge bases without improving from past resolutions. We present MetaKube, an experience-aware LLM framework through three synergistic innovations: (1) an Episodic Pattern Memory Network...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《MetaKube: An Experience-Aware LLM Framework for Kubernetes Failure Diagnosis》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Models (LLMs)** 的 Kubernetes 故障诊断系统存在三大根本缺陷：
1. **无法从操作经验中学习**：现有系统依赖静态知识库（如 RAG），每次诊断独立进行，无法积累历史解决经验。
2. **高质量诊断数据稀缺**：Kubernetes 故障知识分散在文档、论坛、GitHub issues 中，缺乏结构化、高质量的训练与检索数据。
3. **企业级数据隐私要求高**：生产环境中敏感的集群日志不能发送到外部 LLM API，而本地部署的大模型（>70B）计算开销过大，小模型（<10B）又缺乏足够的推理能力。

### 提出的新方法与创新
作者提出 **MetaKube** —— 一种具备“经验感知”能力的 LLM 框架，通过三个协同组件实现持续学习与高效诊断：

#### （1）Episodic Pattern Memory Network (EPMN)
- **功能**：抽象并存储历史故障解决中的模式（patterns），支持基于置信度的快速模式匹配与引导式因果探索。
- **机制**：
  - 双粒度记忆架构：同时保存具体事件（episodic memories）和泛化模式（pattern abstractions）。
  - 动态检索策略：结合语义相似性、时间新鲜度、历史成功率等多因素加权检索最优模式。
  - 支持连续学习：新诊断结果自动更新记忆池，形成闭环反馈。

#### （2）Meta-Cognitive Controller
- **功能**：动态决策是否启用直觉路径（intuitive pathway）或分析路径（analytical pathway）。
- **机制**：
  - 基于记忆相似性、一致性、时效性等指标计算置信度 $ C(M^*) $。
  - 若置信度高于阈值 $ t $，走轻量级直觉路径；否则触发深度分析路径。
  - 阈值 $ t $ 可通过元学习自适应调整，优化速度与准确性的权衡。

#### （3）KubeLLM：领域专用 LLM
- **基础模型**：基于可本地部署的 **Qwen3-8B**。
- **增强方式**：在自建的 **Kubernetes Fault Resolution Dataset (KFRD)** 上进行 **Supervised Fine-Tuning (SFT)** 和 **Low-Rank Adaptation (LoRA)**。
- **优势**：兼具高性能与完全数据隐私保障，适合企业内部部署。

### 相比现有方法的优势
| 维度 | 传统方法 | MetaKube |
|------|--------|---------|
| 学习能力 | 静态知识，无反馈学习 | 持续从经验中学习，支持模式复用 |
| 推理效率 | 固定流程，资源浪费 | 动态路由，简单问题快响应，复杂问题深分析 |
| 数据隐私 | 依赖外部 API | 完全本地部署 |
| 性能表现 | 小模型精度低 | 8B 模型达到接近 GPT-4.1 水平 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### （1）**KubeFault**（主测试集）
- **来源**：从 KubeGraph 自动生成 + 运营商工程师人工校验。
- **规模**：共 1,873 个真实世界故障场景。
- **类别分布**：
  - Resource Errors (22.0%)
  - Network Errors (20.7%)
  - Scheduling Errors (15.9%)
  - Image Errors (14.7%)
  - Configuration Errors (16.8%)
  - System Errors (9.9%)
- **内容**：症状描述、环境上下文、日志、根因、解决方案步骤及命令。

#### （2）**Kubernetes Fault Resolution Dataset (KFRD)**
- **用途**：用于 KubeLLM 的 SFT 训练。
- **规模**：7,000 条高质量样本。
- **构成**：
  - 5,000 条用于训练
  - 2,000 条用于验证/测试
- **构建流程**：
  1. 采集 Stack Overflow / GitHub issues 中的真实问题-解法对；
  2. 重构为 “问题-尝试-解法” 结构；
  3. 使用 LLM（Grok-4）生成合成样本；
  4. 语义去重（sentence-transformer）；
  5. 使用 GPT-5 添加 Chain-of-Thought 推理链。

#### （3）**KubeGraph**
- **知识图谱**：使用 GraphRAG 构建，涵盖 44,022 实体和 111,832 关系。
- **知识来源**：
  - 官方文档（Kubernetes.io）
  - 技术博客（StackOverflow, Medium）
  - 专业书籍（如 *Kubernetes in Action*）
- **分类体系**：六大类故障领域（见下表）

| 类别 | 覆盖范围 |
|------|--------|
| Resource Errors | OOMKilled, CPU Throttling, PVC 挂载失败等 |
| Network Errors | DNS 失败、Ingress 配置错误、CNI 问题等 |
| Scheduling Errors | Node Affinity、Taint/Toleration 不匹配等 |
| Image Errors | ImagePullBackOff、私有仓库认证失败等 |
| Configuration Errors | ConfigMap/Secret 挂载、RBAC 权限不足等 |
| System Errors | kubelet 崩溃、etcd 异常、证书过期等 |

---

### 实验设置与评估指标

#### 评估维度（每项满分 10 分，最终换算为 100 分制）
| 指标 | 含义 |
|------|------|
| **Effectiveness (Eff.)** | 是否正确识别根因并提供有效修复方案 |
| **Equivalence (Equ.)** | 解决方案是否与专家推荐一致 |
| **Completeness (Com.)** | 是否覆盖所有必要步骤、边缘情况 |
| **Safety/Accuracy (S/A)** | 是否符合 Kubernetes 最佳实践，避免危险操作 |
| **Average (Avg.)** | 四项平均得分 |

#### 评分方式
- **自动化评估**：由 GPT-5 对输出打分（盲评）
- **人工评估**：三位具有 5 年以上经验的运营商工程师独立盲评，标准差 < 1.5 才接受

#### 基线方法对比
| 方法 | 类型 |
|------|------|
| GPT-4.1 (Zero-shot) | 商业大模型，零样本提示 |
| GPT-4.1-mini (Zero-shot) | 轻量版 GPT-4.1 |
| Qwen3-8B (Zero-shot) | 开源 8B 模型，无微调 |
| GPT-4.1 (GraphRAG) | GPT-4.1 + 图结构检索 |
| GPT-4.1-mini (GraphRAG) | 同上，轻量版本 |
| Qwen3-8B (GraphRAG) | Qwen3-8B + 图检索 |
| **MetaKube (Ours)** | 本文方法（Qwen3-8B + EPMN + KubeGraph + Meta-Control） |

---

## 3. 主要实验结果和性能指标

### 总体性能对比（KubeFault 数据集，100 分制）

| 方法 | Eff. | Equ. | Com. | S/A | **Avg.** |
|------|------|------|------|-----|----------|
| GPT-4.1 (Zero-shot) | 72.1 | 74.3 | 69.8 | 78.9 | **73.8** |
| GPT-4.1-mini (Zero-shot) | 61.5 | 63.8 | 59.2 | 69.7 | **63.6** |
| Qwen3-8B (Zero-shot) | 48.7 | 51.2 | 46.1 | 57.4 | **50.9** |
| GPT-4.1 (GraphRAG) | 89.3 | 92.6 | 91.4 | 94.1 | **91.9** |
| GPT-4.1-mini (GraphRAG) | 79.8 | 81.3 | 78.4 | 85.2 | **81.2** |
| Qwen3-8B (GraphRAG) | 66.7 | 69.1 | 64.8 | 73.3 | **68.5** |
| **MetaKube (Ours)** | **91.2** | **90.8** | **87.3** | **92.5** | **90.5** |

> ✅ **关键结论**：
> - MetaKube 将 Qwen3-8B 的性能从 **50.9 提升至 90.5**，提升达 **+39.6 分**（约 77.8%）。
> - 性能逼近最强基线 GPT-4.1 + GraphRAG（91.9），仅差 **1.4 分**。
> - 在 **安全性/准确性（S/A）** 上表现最佳（92.5），说明其建议更可靠、合规。

---

### 消融实验结果（Ablation Studies）

#### （1）EPMN 消融实验
- **移除 EPMN 后性能下降**：平均降低 **15.3%**
- 各项指标下降幅度：
  - Effectiveness: ↓13.4%
  - Equivalence: ↓15.2%
  - Completeness: ↓15.9%
  - **Safety/Accuracy: ↓16.6%**

> 🔍 表明 EPMN 对捕捉常见错误模式、提高安全性和复用历史经验至关重要。

#### （2）KubeLLM 消融实验（是否进行 SFT）
- 在 KFRD 测试集上，经过 SFT 的 KubeLLM 相比原始 Qwen3-8B：
  - **性能提升 45.5%**
  - 在 Effectiveness、Completeness 等维度显著改善
- 即使是 8B 模型，也能通过领域微调逼近商业 API 表现。

#### （3）KubeGraph 消融实验

| 数据集 | 指标 | w/ KubeGraph | w/o KubeGraph | 提升 |
|-------|------|---------------|----------------|------|
| **KubeFault (in-domain)** | Avg. | **75.2** | 34.6 | **+117.3%** |
| **Telecom Dataset (out-of-domain)** | Avg. | **57.6** | 22.4 | **+157.1%** |

> 🚀 表明 KubeGraph 不仅提升领域内性能，还展现出强大泛化能力，尤其在真实电信场景中效果显著。

---

## 4. 关键结论和发现

### 主要发现
1. **经验学习至关重要**：EPMN 使得系统能够从过往诊断中提炼通用模式，并在新问题中复用，带来 **15.3% 的性能增益**。
2. **双路径设计高效灵活**：Meta-Cognitive Controller 成功实现了“快慢思维”的动态切换，在保证准确率的同时大幅节省计算资源。
3. **小模型也能胜任专业任务**：通过对 Qwen3-8B 进行高质量 SFT + LoRA 微调，可在本地部署条件下实现接近 GPT-4.1 的诊断水平。
4. **结构化知识图谱极大增强泛化能力**：KubeGraph 不仅提升完整性与安全性，还在跨域场景中表现出卓越迁移能力。

### 方法的局限性
- **初始冷启动问题**：EPMN 需要一定数量的历史诊断数据才能发挥价值，初期可能依赖分析路径较多。
- **依赖高质量标注数据**：尽管使用了合成数据增强，但 KFRD 和 KubeGraph 的质量仍受限于原始资料的准确性。
- **实时性挑战**：大规模图遍历（KubeGraph）在极端复杂拓扑下可能存在延迟风险。

### 未来工作方向
- **引入强化学习机制**：让系统根据诊断成功与否自动调整记忆权重与控制策略。
- **支持多模态输入**：融合 Prometheus 指标、Fluentd 日志流、Jaeger 追踪等结构化信号。
- **构建开放社区生态**：鼓励用户共享匿名化诊断记录以持续扩充 EPMN 记忆库。
- **扩展至其他分布式系统**：将 MetaKube 架构推广至 Spark、Flink、Service Mesh 等系统的故障诊断。

---

> ✅ **开源信息**  
> 项目代码、KFRD 数据集、KubeGraph 与 KubeLLM 模型均已公开：  
> 🔗 [https://github.com/MetaKube-LLM-for-Kubernetes-Diagnosis/MetaKube](https://github.com/MetaKube-LLM-for-Kubernetes-Diagnosis/MetaKube)

</details>

---

### 13. [Kronecker-Structured Nonparametric Spatiotemporal Point Processes](https://arxiv.org/abs/2603.23746)

**Authors**: Zhitong Xu, Qiwei Yuan, Yinghao Chen, Yan Sun, Bin Shen, Shandian Zhe  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.23746v1  

#### Abstract
Events in spatiotemporal domains arise in numerous real-world applications, where uncovering event relationships and enabling accurate prediction are central challenges. Classical Poisson and Hawkes processes rely on restrictive parametric assumptions that limit their ability to capture complex inte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Kronecker-Structured Nonparametric Spatiotemporal Point Processes*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **spatiotemporal point process** 模型在建模复杂事件关系时存在以下局限：
- **经典模型**（如 Poisson 和 Hawkes 过程）依赖于参数化假设（如指数型核函数），难以捕捉复杂的时空交互模式，尤其是抑制效应（inhibition）和非单调动态。
- **神经点过程模型**（如 Neural Hawkes、Transformer Hawkes）虽然表达能力强，但通过黑箱方式编码历史事件，缺乏对事件间关系的可解释性。

因此，如何在保持高建模灵活性的同时实现**透明且可解释的事件关系发现**，是一个关键挑战。

### 提出的新方法与创新思路
本文提出 **KSTPP**（Kronecker-Structured Nonparametric Spatiotemporal Point Process），其核心思想是：
- 将条件强度函数建模为背景强度与历史事件影响的叠加：
  $$
  \lambda(t,x,y|\mathcal{H}_t) = \phi\left(g(x,y) + \sum_{t_n < t} f(t-t_n, x-x_n, y-y_n)\right)
  $$
  其中：
  - $g(x,y)$：空间背景强度，由 **spatial Gaussian Process (GP)** 建模；
  - $f(\Delta t, \Delta x, \Delta y)$：时空影响核函数，由 **spatiotemporal GP** 建模；
  - $\phi(\cdot)$：SoftPlus 链接函数，确保强度非负。
- 支持任意符号的影响值：$f > 0$ 表示激发（excitation），$f < 0$ 表示抑制（inhibition），$f \approx 0$ 表示中性。

#### 创新点：
1. **非参数化建模 + 可解释性**：  
   使用 GP 对影响核进行灵活建模，无需预设函数形式，同时直接输出 $f(\Delta t, \Delta x, \Delta y)$，提供直观的事件关系可视化。
   
2. **Kronecker-Structured Inducing Representation**：  
   采用可分离乘积核（separable product kernels）并在结构化网格上定义诱导点，使得协方差矩阵具有 Kronecker 结构（如 $K = K_0 \otimes K_1 \otimes K_2$）。利用 Kronecker 代数显著降低计算复杂度。

3. **高效积分方案**：  
   提出基于 **tensor-product Gauss-Legendre quadrature** 的数值积分方法，用于高效求解似然中的不可解析积分项，兼顾精度与效率。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **建模能力** | 超越传统参数化模型，支持 excitation/inhibition/time-varying effects |
| **可解释性** | 显式学习影响核 $f$，可分析事件间的时空依赖结构 |
| **可扩展性** | 利用 Kronecker 结构将计算从 $O(N^3)$ 降至 $O(\sum m_k^3)$，适用于大规模事件序列 |
| **预测性能** | 在多个真实世界数据集上达到 SOTA 或接近最优水平 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共三类真实世界 benchmark 数据集：
- **Earthquake**：日本1990–2020年地震事件（每月一个序列），时间单位为天。
- **Covid-19**：美国新泽西州2020年每日新冠病例报告（每周一个序列）。
- **Citibike**：纽约市2019年共享单车出行记录（每天一个序列），时间单位为小时。

此外还构造了两个合成数据集（SYN1, SYN2）用于验证模型恢复 ground-truth 强度的能力。

### 实验设置与评估指标
#### 评估任务
- **下一次事件预测**：
  - 时间预测：使用 **RMSE**（Root Mean Square Error）
  - 位置预测：使用 **Euclidean Distance**

#### 模型配置
- 使用 **SE kernel** 或 **Matérn-5/2 kernel**
- SoftPlus 参数 $\beta = 1$
- 每维度使用 8–16 个 Gauss-Legendre quadrature 节点
- Mini-batch 大小为 1，Adam 优化器（lr=1e-3）

### 基线方法对比
| 类别 | 方法 |
|------|------|
| **传统参数模型** | STHP（Spatiotemporal Hawkes Process）、Homogeneous Poisson Process |
| **神经点过程** | NHP（Neural Hawkes Process）、THP（Transformer Hawkes Process）、NSTPP、DeepSTPP、NJSDE |
| **扩散生成模型** | DSTPP（Diffusion Spatiotemporal Point Process）|

所有对比均基于相同训练/验证/测试划分，并尽可能复现原论文最优结果。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

| Dataset | Metric | Best Method | Score | KSTPP Score |
|--------|-------|-------------|-------|-----------|
| **Earthquake** | Time RMSE | DSTPP | 0.375 ± 0.001 | **0.372 ± 0.000** ✅ |
| | Location Dist. | DSTPP / KSTPP | 6.77 / **6.72** | ✅ |
| **Covid-19** | Time RMSE | DSTPP | 0.093 ± 0.000 | 0.100 ± 0.000 |
| | Location Dist. | KSTPP | — | **0.392 ± 0.000** ✅ |
| **Citibike** | Time RMSE | DSTPP | 0.200 ± 0.002 | 0.206 ± 0.000 |
| | Location Dist. | DSTPP / KSTPP | 0.031 / **0.031** ✅ |

> ✅ 表示进入前两名；**加粗**表示最佳

#### 总结：
- KSTPP 在 **location prediction 上全部第一**，在 **time prediction 上几乎全部进入前二**。
- 性能媲美最先进的 **diffusion-based model DSTPP**，远超其他神经点过程模型（如 NSTPP、DeepSTPP）。

### 合成数据上的强度恢复能力（Table 1 & 2）
| 方法 | SYN1（相对 L2 误差） | SYN2 |
|-----|------------------|------|
| STHP | 1.45e-1 | 8.42e-2 |
| NSTPP | 5.57e-2 | 2.99e-2 |
| **KSTPP** | **4.44e-2** ✅ | **2.00e-2** ✅ |

- KSTPP 在 **temporal marginal intensity** 和 **full spatiotemporal intensity** 恢复上均表现最优。
- 特别是在强局部抑制场景（SYN2）中，传统 excitation-only 模型严重失真，而 KSTPP 成功识别出负影响区域。

### 影响核估计结果（Figure 3, 4, 7）
- 在 SYN1 和 SYN2 中，KSTPP 准确恢复了 ground-truth 的 excitation/inhibition 区域分布。
- 在 Earthquake 数据上，学到的 kernel 显示：
  - 短时间滞后（$\Delta t \approx 0$）有强烈近场激发 → 符合 aftershock 现象；
  - 存在局部负值区域 → 暗示“stress shadow”效应，即主震后某些区域地震活动被暂时抑制；
  - 随着 $\Delta t$ 增大，影响迅速衰减 → 符合 Omori-Utsu 定律。

这些发现具有明确地质学意义，且无法被标准 Hawkes 模型捕获。

---

## 4. 关键结论和发现

### 主要发现
1. **KSTPP 实现了灵活性与可解释性的统一**：
   - 通过 GP 非参数建模，摆脱了对固定核函数的依赖；
   - 显式输出影响核 $f(\Delta t, \Delta x, \Delta y)$，可用于科学分析事件机制。

2. **支持抑制效应建模至关重要**：
   - 在地震等实际系统中，抑制现象普遍存在；
   - 允许 $f < 0$ 极大地提升了模型的真实性和拟合能力。

3. **Kronecker 结构带来显著可扩展性提升**：
   - 协方差运算分解到各维度，避免全矩阵操作；
   - 支持 mini-batch 训练，适用于长序列和大规模数据集。

4. **高性能不以牺牲预测能力为代价**：
   - 尽管强调可解释性，KSTPP 的预测精度仍达到甚至超过多数黑箱神经模型。

### 方法的局限性
- **假设可分离性**：模型依赖 kernel 的 separable product structure，可能限制对高度耦合时空动态的建模能力。
- **网格分辨率限制**：诱导点和 quadrature 网格需预先设定范围与密度，在稀疏或极端分布区域可能不够灵活。
- **三维扩展成本增加**：虽然文中提到可推广至 3D 空间，但 Kronecker 维度越多，每维所需计算资源仍会累积。

### 未来工作方向
1. 探索更灵活的 kernel 分解方式（如低秩扰动）以缓解 strict separability 假设；
2. 引入 adaptive mesh refinement 技术，动态调整诱导点布局；
3. 扩展至 marked point processes，联合建模事件类型与时空动态；
4. 应用于更多领域（如神经科学 spike trains、金融交易、社交媒体传播）中的因果关系发现。

--- 

> ✅ **总体评价**：  
> KSTPP 是一种兼具 **high expressiveness**, **interpretability**, 和 **scalability** 的新型 spatiotemporal point process 框架，在理论设计与实证效果之间取得了良好平衡，为事件数据分析提供了新的工具范式。

</details>

---

### 14. [MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning](https://arxiv.org/abs/2603.24044)

**Authors**: Andrea Manzoni  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.24044v1  

#### Abstract
Standard LoRA fine-tuning of Mixture-of-Experts (MoE) models applies adapters to every expert, yet our profiling shows that per-layer expert routing is highly skewed: a small subset of experts handles most tokens in each layer, while many others are rarely activated ("cold"). We propose MoE-Sieve, a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在对 **Mixture-of-Experts (MoE)** 模型进行参数高效微调（如 LoRA）时，标准做法是为**所有专家（experts）都附加适配器（adapter）**。然而，作者通过分析发现：  
- 在每一层中，**专家路由（expert routing）高度集中**，即少数“热”专家处理了大部分 token，而大多数专家很少被激活（称为“冷”专家）。
- 因此，在所有专家上应用 LoRA 是一种资源浪费，尤其是在训练成本和参数量方面。

该问题的本质是：**全局负载均衡（global load balancing）掩盖了局部路由不均（local imbalance）**，导致统一适配策略效率低下。

---

### 提出了什么新方法或新思路
作者提出 **MoE-Sieve**，一个基于路由引导的 LoRA 微调框架，其核心思想是：

> **只对每层中被最多路由的 top-k 专家应用 LoRA，而非全部专家。**

具体流程为三步：
1. **Profile（分析）**：在任务数据上运行一次前向传播，统计每个专家的激活次数。
2. **Select（选择）**：在每层中选择激活次数最多的 top-k 专家（默认 k = 25% 的 routed experts）。
3. **Fine-tune（微调）**：仅在这部分“热”专家上应用 LoRA，其余保持冻结。

此外，**attention 层、router 和 shared experts** 始终参与训练。

---

### 相比现有方法的优势
- **显著降低训练开销**：减少约 **70–73% 的 LoRA 可训练参数** 和 **71–73% 的检查点大小**，训练时间最多减少 **50%**。
- **性能无损**：在多个模型和任务上，仅使用 top-25% 专家的性能与全量 LoRA 相当，平均差异在 ±1 个百分点以内。
- **简单有效**：无需复杂优化或超参搜索，仅需一次轻量级 profiling 即可确定适配目标。
- **优于随机选择**：消融实验证明，基于路由信号的选择比随机选择高约 **2.5 pp**，说明路由信息具有实际意义。
- **优于动态分配策略**：贪婪或覆盖率阈值等更复杂的预算分配方式并未带来增益，表明 **uniform top-k 已足够有效**。

---

## 2. 核心实验方法和设置

### 使用的模型
- **OLMoE-1B-7B**：16 层，64 个 routed experts，无 shared experts，每 token 激活 8 个专家。
- **Qwen1.5-MoE-A2.7B**：24 层，60 个 routed + 4 个 shared experts，top-4 路由。

这两个模型架构差异大，用于验证方法泛化性。

---

### 使用的数据集（任务）
共三个下游任务，覆盖多种类型：
- **Spider**：文本到 SQL 生成（structured generation），使用官方 Test Suite 执行准确率评估。
- **GSM8K**：小学数学题（symbolic reasoning），exact-match 准确率。
- **HellaSwag**：常识推理（commonsense understanding），归一化准确率。

此外，在 profiling 阶段还使用了另外 7 个数据集（如 MMLU、BoolQ、PIQA 等）进行路由模式分析。

---

### 实验设置
- **微调方法**：LoRA（rank=32, α=64, dropout=0.05），应用于 selected modules 的线性投影。
- **训练配置**：AdamW 优化器，学习率 4e-4，3 个 epoch，batch size 64，每条件运行 **8 个 seed** 以评估稳定性。
- **对比条件**：
  - **Full LoRA**：所有 routed experts 均应用 LoRA（baseline）。
  - **Hot-25%-LoRA**：仅 top-25% 最常被路由的专家应用 LoRA。
- **评估指标**：
  - 主要：**平均准确率 ± 标准差**
  - 配对差值（△）、95% 置信区间、TOST 等价性检验（pre-declared ±2 pp margin）
  - 效率指标：可训练参数数、检查点大小、训练耗时

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 3）

| Model       | Task       | Full LoRA     | Hot(25%)      | △ (pp) | 95% CI (pp)       | 等价性 @±2pp |
|-------------|------------|---------------|----------------|--------|--------------------|--------------|
| OLMoE       | Spider     | .396 ± .026   | .399 ± .015    | +0.30  | [-2.04, +2.64]     | ×（因方差大）|
| OLMoE       | GSM8K      | .304 ± .011   | .304 ± .006    | -0.08  | [-1.45, +1.30]     | √            |
| OLMoE       | HellaSwag  | .805 ± .005   | .807 ± .008    | +0.17  | [-0.71, +1.05]     | √            |
| Qwen        | Spider     | .520 ± .014   | .511 ± .005    | -0.93  | [-1.88, +0.03]     | √            |
| Qwen        | GSM8K      | .590 ± .011   | .592 ± .007    | +0.20  | [-0.77, +1.17]     | √            |
| Qwen        | HellaSwag  | .885 ± .002   | .893 ± .001    | +0.73  | [+0.53, +0.93]     | √            |

> ✅ **结论**：所有条件下，性能差异均在 ±1 pp 内；5/6 条件通过 TOST 等价性检验。

---

### 效率提升（Table 4）

| Model   | 参数减少 | 检查点大小减少 | 训练时间减少（示例） |
|---------|----------|----------------|------------------------|
| OLMoE   | 72.7%    | 73.4%          | 1h48m → 54m (**50%**)  |
| Qwen    | 70.3%    | 71.0%          | 3h23m → 1h44m (**49%**) |

> ⏱️ 显著节省存储与计算资源。

---

### 消融实验结果

#### （1）Random Selection 对比
- 在相同预算下（k=16 或 k=8），**random-k 比 routing-guided-k 差 ~2.5 pp**。
- 甚至 **random-k=16 不如 hot-k=8**，说明“选对专家”比“多用专家”更重要。
- 支持了路由信号的有效性和任务特异性。

#### （2）Dynamic Allocation 策略对比
测试两种更复杂策略：
- **Greedy marginal-gain allocation**：按覆盖率增益动态分配每层专家数量。
- **Coverage-threshold allocation**：每层选至覆盖 60% 路由流量为止。

✅ 结果：**均未超越 uniform top-k**，且覆盖率相近但精度无提升。

> 📌 表明：**uniform top-k 是简单且最优的实用选择**。

#### （3）Count vs. Mass Ranking
- 排名依据可以是 **activation count** 或 **routing mass（softmax 权重和）**。
- 在 OLMoE 和 Qwen 上两者高度一致（Jaccard > 0.92），但在 DeepSeek 上差异较大（Jaccard=0.646）。
- 默认推荐使用 **count-based ranking**（更直观、解释性强）。

---

## 4. 关键结论和发现

### 主要发现
1. **Per-layer 路由严重倾斜**：
   - 尽管全局负载均衡，但**每层内部路由极不均匀**。
   - 层内激活系数变异度（Layer CV）是全局 CV 的 **4.0–4.9 倍**。
   - top-25% 专家即可捕获 37–53% 的层内激活量。

2. **MoE-Sieve 高效且有效**：
   - 仅微调 top-25% 被路由专家，即可实现与 Full LoRA **性能相当**。
   - 节省 **70–73% 参数与检查点空间**，训练时间减半。

3. **路由信号至关重要**：
   - 基于路由的选择显著优于随机选择（+2.5 pp），证明了其语义价值。

4. **非单调预算-方差关系**：
   - 增加冷专家可能引入梯度噪声，导致 seed-to-seed 方差上升。
   - 提出“**cold-expert noise hypothesis**”：冷专家微调无助于性能却增加不稳定性。

5. **uniform top-k 已足够好**：
   - 更复杂的动态分配策略无法带来收益，推荐采用“profile → count → pick top-k → fine-tune”的**一揽子实践方案**。

---

### 方法的局限性
- **模型规模有限**：实验仅在 ~7B 和 ~14B 总参数的 MoE 模型上验证，更大模型（如百亿级）是否适用尚不清楚。
- **任务范围受限**：集中在结构化生成、数学、常识任务，未涵盖指令跟随、安全对齐、多语言等场景。
- **静态选择机制**：专家集合在训练前固定，未考虑训练过程中路由分布的变化。
- **25% 阈值的经验性**：当前阈值基于实验经验设定，缺乏理论指导原则。

---

### 未来工作方向
1. **建立理论模型**：从路由 Pareto 分布出发，推导最优专家预算公式，替代经验阈值。
2. **验证 cold-expert noise 假设**：设计控制实验（如添加冷专家填充组）直接检验其影响。
3. **探索动态 re-profiling**：在训练中周期性更新专家选择，适应路由漂移。
4. **扩展至更大 MoE 模型**：验证方法在工业级大规模 MoE 上的有效性。
5. **结合其他 PEFT 技术**：将 MoE-Sieve 与 LoRA+, AdapterDrop, DR-LoRA 等结合，进一步提升效率。

---

> 🔚 **一句话总结**：  
> **MoE-Sieve 揭示了 MoE 模型中“局部路由倾斜”的普遍现象，并提出一种简单高效的 routing-guided LoRA 微调策略——只需关注最活跃的 25% 专家，就能以不到三分之一的成本达到全量微调的性能水平，兼具实用性与理论启发性。**

</details>

---

### 15. [AVO: Agentic Variation Operators for Autonomous Evolutionary Search](https://arxiv.org/abs/2603.24517)

**Authors**: Terry Chen, Zhifan Ye, Bing Xu, Zihao Ye, Timmy Liu, Ali Hassani, Tianqi Chen, Andrew Kerr, Haicheng Wu, Yang Xu, Yu-Jung Chen, Hanfeng Chen, Aditya Kane, Ronny Krashinsky, Ming-Yu Liu, Vinod Grover, Luis Ceze, Roger Bringmann, John Tran, Wei Liu, Fung Xie, Michael Lightstone, Humphrey Shi  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.24517v1  

#### Abstract
Agentic Variation Operators (AVO) are a new family of evolutionary variation operators that replace the fixed mutation, crossover, and hand-designed heuristics of classical evolutionary search with autonomous coding agents. Rather than confining a language model to candidate generation within a pres...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AVO: Agentic Variation Operators for Autonomous Evolutionary Search

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于LLM的进化搜索框架（如FunSearch、AlphaEvolve）将大语言模型（LLM）限制在固定的“生成-评估”流水线中，仅作为**候选生成器**（candidate generator），无法主动查阅资料、调试代码、分析反馈或迭代优化策略。这种设计严重限制了LLM在复杂工程任务中的潜力，尤其是在需要深度硬件级推理和持续调优的场景下（如GPU kernel优化）。

### 提出的新方法：Agentic Variation Operators (AVO)
作者提出 **Agentic Variation Operators (AVO)** ——一种全新的进化变异算子范式，其核心思想是：
> 将LLM从“管道中的一个步骤”升级为**完整的变异操作符本身**，即一个具备自主决策能力的AI代理（AI agent）。

该代理具备以下能力：
- **自主规划与执行**：可自主决定何时查阅历史方案 $P_t$、领域知识库 $K_C$、执行评测函数 $f$。
- **多轮交互循环**：支持“计划 → 实现 → 测试 → 调试 → 修复”的长周期闭环，而非单次生成。
- **持久记忆与工具调用**：拥有对话历史作为长期记忆，并能调用编译器、profiler、shell等开发工具进行自我验证。

### 相比现有方法的优势
| 维度 | 传统方法（如FunSearch） | AVO |
|------|------------------------|-----|
| 角色定位 | LLM仅为Generate模块 | LLM作为完整Vary操作符 |
| 工作流控制 | 固定Pipeline由框架控制 | 自主Agent Loop动态决策 |
| 反馈利用 | 单次输出无修正机制 | 支持失败诊断与策略调整 |
| 探索深度 | 局限于表面代码变换 | 可实现微架构级（micro-architectural）优化 |

> ✅ **本质区别**：AVO将进化过程从“人类设计流程 + LLM填空”转变为“由智能体主导的自主工程探索”。

---

## 2. 核心实验方法和设置

### 实验目标
在NVIDIA Blackwell B200 GPU上对**multi-head attention (MHA)** 和 **grouped-query attention (GQA)** 内核进行全自动性能优化，挑战当前最先进的人工优化实现。

### 数据集 / 配置
- **硬件平台**：NVIDIA B200 GPU
- **软件环境**：CUDA 13.1, PyTorch 2.10.0
- **精度模式**：BF16
- **头维度**：128
- **序列长度**：{4K, 8K, 16K, 32K}
- **总token数固定为32768**，通过调节batch size实现（例如：seq=4K时bs=8；seq=32K时bs=1）
- **MHA配置**：16 query heads
- **GQA配置**：
  - Group=8: 32 query heads, 4 KV heads
  - Group=4: 32 query heads, 8 KV heads

### 评估指标
- **Throughput (TFLOPS)**：前向传播预填充阶段的计算吞吐量
- **Correctness**：数值正确性（相对于参考实现）
- 所有实验运行10次取平均值与标准差，timing脚本复用FlashAttention-4官方仓库。

### 基线方法对比
| 基线 | 类型 | 描述 |
|------|------|------|
| **cuDNN** | 闭源专家优化 | NVIDIA官方库，含Blackwell定制优化（v9.19.1） |
| **FlashAttention-4 (FA4)** | 开源SOTA | 当前最先进的开源attention kernel，针对Blackwell架构优化 |

---

## 3. 主要实验结果和性能指标

### 多头注意力（MHA）性能对比
#### 非因果（non-causal）attention
- AVO最高达到 **1668 TFLOPS**
- 相比 cuDNN 提升 **+1.8% ~ +2.4%**（长序列）
- 在短序列接近测量噪声水平

#### 因果（causal）attention
- AVO全面超越两个基线
- 相比 cuDNN 提升 **+0.4% ~ +3.5%**
- 相比 FA4 提升 **+5.0% ~ +10.5%**

> 🔥 **峰值表现**：在某些配置下，AVO实现了比FA4高超10%的吞吐提升。

### 分组查询注意力（GQA）迁移能力测试
- **无需人工干预**，仅通过prompt让AVO代理将MHA kernel适配为GQA版本
- **耗时约30分钟**完成自动化改造
- 结果显著优于基线：
  - 相比 cuDNN 最高提升 **+7.0%**
  - 相比 FA4 最高提升 **+9.3%**

> 🔄 **关键发现**：AVO发现的优化具有强泛化性，能有效迁移到不同attention变体。

### 消融实验与关键优化分析（见Section 5）
| 优化项 | 版本跨度 | 非因果增益 | 因果增益 | 技术要点 |
|-------|---------|------------|----------|----------|
| **Branchless accumulator rescaling** | v19→v20 | **+8.1%** | +1.6% | 消除条件分支，改用predicated select + 更轻量memory fence |
| **Correction/MMA pipeline overlap** | v29→v30 | +1.1% | +0.4% | 允许correction warp与第二阶段PV GEMM重叠执行 |
| **Register rebalancing across warp groups** | v32→v33 | +2.1% | ~0% | 从softmax组转移寄存器给correction组，缓解local memory spill |

> 💡 这些优化均涉及**跨子系统协同推理**（同步、内存序、流水线调度、寄存器分配），体现真正的硬件级理解。

### 进化轨迹分析
- 总共提交 **40个版本**
- 内部探索超过 **500个优化方向**
- 性能提升呈“阶梯式跃迁 + 平台期精调”
- 后期优化收益递减，符合典型kernel tuning规律

---

## 4. 关键结论和发现

### 主要结论
1. ✅ **AVO成功将AI代理角色从“生成器”升维至“变异操作符”**，实现了真正意义上的自主进化搜索。
2. ✅ 在高度优化的attention kernel上，AVO仍能发现超越专家手工调优的性能突破（最高+10.5% over FA4）。
3. ✅ 发现的优化具备**真实硬件级推理能力**，涵盖register allocation、instruction pipelining、workload balancing等多个层面。
4. ✅ 优化成果具有良好**可迁移性**，可在极短时间内适配到GQA等新结构并保持显著优势。
5. ✅ 整个7天连续演化过程完全无人参与，展示了AI代理在复杂工程任务中的长期自治潜力。

### 方法的局限性
- 当前研究聚焦于**单谱系演化**（single-lineage），未探索种群多样性机制（如island model）。
- 依赖高质量的**domain-specific knowledge base**（如PTX文档、架构手册），在知识缺失场景可能受限。
- 成功依赖于强大的基础agent能力（planning, tool use, memory），对底层LLM要求极高。
- 当前应用集中于高性能计算领域，通用性有待在其他领域验证。

### 未来工作方向
- 将AVO扩展至**多代理协作演化框架**，引入竞争与多样性机制。
- 探索AVO在其他性能敏感系统中的应用，如：
  - Sparse kernels
  - Linear algebra routines (BLAS/GEMM)
  - Database query engines
  - Compiler optimization passes
- 构建更完善的**self-supervision机制**以应对更复杂的停滞与误入歧途情况。
- 推动AVO成为通用AI驱动软件工程（AI-driven SE）基础设施的一部分，支持全栈自动优化。

---

> 🏁 **总结一句话**：  
> **AVO标志着LLM在进化搜索中的角色根本转变——不再是被动的代码填写者，而是主动的、具备工程实践能力的“AI工程师”，能够在无人干预下持续发现超越人类专家的底层系统优化。**

</details>

---

### 16. [DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving](https://arxiv.org/abs/2603.24587)

**Authors**: Pengxuan Yang, Yupeng Zheng, Deheng Qian, Zebin Xing, Qichao Zhang, Linbo Wang, Yichen Zhang, Shaoyu Guo, Zhongpu Xia, Qiang Chen, Junyu Han, Lingyun Xu, Yifeng Pan, Dongbin Zhao  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.24587v1  

#### Abstract
We introduce DreamerAD, the first latent world model framework that enables efficient reinforcement learning for autonomous driving by compressing diffusion sampling from 100 steps to 1 - achieving 80x speedup while maintaining visual interpretability. Training RL policies on real-world driving data...

---

### 17. [LLM-Driven Reasoning for Constraint-Aware Feature Selection in Industrial Systems](https://arxiv.org/abs/2603.24979)

**Authors**: Yuhang Zhou, Zhuokai Zhao, Ke Li, Spilios Evmorfos, G\"okalp Demirci, Mingyi Wang, Qiao Liu, Qifei Wang, Serena Li, Weiwei Li, Tingting Wang, Mingze Gao, Gedi Zhou, Abhishek Kumar, Xiangjun Fan, Lizhu Zhang, Jiayi Liu  
**Category**: cs.CL  
**Published**: 2026-03-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.24979v1  

#### Abstract
Feature selection is a crucial step in large-scale industrial machine learning systems, directly affecting model accuracy, efficiency, and maintainability. Traditional feature selection methods rely on labeled data and statistical heuristics, making them difficult to apply in production environments...

---

### 18. [eBeeMetrics: An eBPF-based Library Framework for Feedback-free Observability of QoS Metrics](https://arxiv.org/abs/2603.25067)

**Authors**: Muntaka Ibnath, Mohammadreza Rezvani, Daniel Wong  
**Category**: cs.DC  
**Published**: 2026-03-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.25067v1  

#### Abstract
Many system management runtimes (SMRs), such as resource management and power management techniques, rely on quality-of-service (QoS) metrics, such as tail latency or throughput, as feedback. These QoS metrics are generally neither observable with hardware performance counters nor directly observabl...

---

### 19. [Unveiling Hidden Convexity in Deep Learning: a Sparse Signal Processing Perspective](https://arxiv.org/abs/2603.23831)

**Authors**: Emi Zeger, Mert Pilanci  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.23831v1  

#### Abstract
Deep neural networks (DNNs), particularly those using Rectified Linear Unit (ReLU) activation functions, have achieved remarkable success across diverse machine learning tasks, including image recognition, audio processing, and language modeling. Despite this success, the non-convex nature of DNN lo...

---

### 20. [Mixed-signal implementation of feedback-control optimizer for single-layer Spiking Neural Networks](https://arxiv.org/abs/2603.24113)

**Authors**: Jonathan Haag, Christian Metzner, Dmitrii Zendrikov, Giacomo Indiveri, Benjamin Grewe, Chiara De Luca, Matteo Saponati  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.24113v1  

#### Abstract
On-chip learning is key to scalable and adaptive neuromorphic systems, yet existing training methods are either difficult to implement in hardware or overly restrictive. However, recent studies show that feedback-control optimizers can enable expressive, on-chip training of neuromorphic devices. In ...

---

### 21. [The DeepXube Software Package for Solving Pathfinding Problems with Learned Heuristic Functions and Search](https://arxiv.org/abs/2603.23873)

**Authors**: Forest Agostinelli  
**Category**: cs.AI  
**Published**: 2026-03-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.23873v1  

#### Abstract
DeepXube is a free and open-source Python package and command-line tool that seeks to automate the solution of pathfinding problems by using machine learning to learn heuristic functions that guide heuristic search algorithms tailored to deep neural networks (DNNs). DeepXube is comprised of the late...

---

### 22. [Rafture: Erasure-coded Raft with Post-Dissemination Pruning](https://arxiv.org/abs/2603.24761)

**Authors**: Rithwik Kerur, Divyakant Agrawal, Michael K. Reiter, Dahlia Malkhi  
**Category**: cs.DC  
**Published**: 2026-03-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.24761v1  

#### Abstract
Spreading and storing erasure-coded data in distributed systems effectively is challenging in real settings. Practical deployments must contend with unpredictable network latencies, particularly when information dispersal is integrated into consensus protocols, a prominent and latency-sensitive use ...

---

### 23. [Stochastic Dimension-Free Zeroth-Order Estimator for High-Dimensional and High-Order PINNs](https://arxiv.org/abs/2603.24002)

**Authors**: Zhangyong Liang, Ji Zhang  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.24002v1  

#### Abstract
Physics-Informed Neural Networks (PINNs) for high-dimensional and high-order partial differential equations (PDEs) are primarily constrained by the $\mathcal{O}(d^k)$ spatial derivative complexity and the $\mathcal{O}(P)$ memory overhead of backpropagation (BP). While randomized spatial estimators s...

---

### 24. [Linear-Nonlinear Fusion Neural Operator for Partial Differential Equations](https://arxiv.org/abs/2603.24143)

**Authors**: Heng Wu, Junjie Wang, Benzhuo Lu  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.24143v1  

#### Abstract
Neural operator learning directly constructs the mapping relationship from the equation parameter space to the solution space, enabling efficient direct inference in practical applications without the need for repeated solution of partial differential equations (PDEs) - an advantage that is difficul...

---

### 25. [TsetlinWiSARD: On-Chip Training of Weightless Neural Networks using Tsetlin Automata on FPGAs](https://arxiv.org/abs/2603.24186)

**Authors**: Shengyu Duan, Marcos L. L. Sartori, Rishad Shafik, Alex Yakovlev  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.24186v1  

#### Abstract
Increasing demands for adaptability, privacy, and security at the edge have persistently pushed the frontiers for a new generation of machine learning (ML) algorithms with training and inference capabilities on-chip. Weightless Neural Network (WNN) is such an algorithm that is principled on lookup t...

---

### 26. [Efficient Benchmarking of AI Agents](https://arxiv.org/abs/2603.23749)

**Authors**: Franck Ndzomga  
**Category**: cs.AI  
**Published**: 2026-03-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.23749v1  

#### Abstract
Evaluating AI agents on comprehensive benchmarks is expensive because each evaluation requires interactive rollouts with tool use and multi-step reasoning. We study whether small task subsets can preserve agent rankings at substantially lower cost. Unlike static language model benchmarks, agent eval...

---

### 27. [Approaches to Analysing Historical Newspapers Using LLMs](https://arxiv.org/abs/2603.25051)

**Authors**: Filip Dobrani\'c, Tina Munda, Oliver Peji\'c, Vojko Gorjanc, Uro\v{s} \v{S}majdek, David Bordon, Jakob Lenardi\v{c}, Tja\v{s}a Konov\v{s}ek, Kristina Pahor de Maiti Tekav\v{c}i\v{c}, Ciril Bohak, Darja Fi\v{s}er  
**Category**: cs.CL  
**Published**: 2026-03-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.25051v1  

#### Abstract
This study presents a computational analysis of the Slovene historical newspapers \textit{Slovenec} and \textit{Slovenski narod} from the sPeriodika corpus, combining topic modelling, large language model (LLM)-based aspect-level sentiment analysis, entity-graph visualisation, and qualitative discou...

---

### 28. [PICon: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency](https://arxiv.org/abs/2603.25620)

**Authors**: Minseo Kim, Sujeong Im, Junseong Choi, Junhee Lee, Chaeeun Shim, Edward Choi  
**Category**: cs.CL  
**Published**: 2026-03-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.25620v1  

#### Abstract
Large language model (LLM)-based persona agents are rapidly being adopted as scalable proxies for human participants across diverse domains. Yet there is no systematic method for verifying whether a persona agent's responses remain free of contradictions and factual inaccuracies throughout an intera...

---

### 29. [Causal Reconstruction of Sentiment Signals from Sparse News Data](https://arxiv.org/abs/2603.23568)

**Authors**: Stefania Stan, Marzio Lunghi, Vito Vargetto, Claudio Ricci, Rolands Repetto, Brayden Leo, Shao-Hong Gan  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.23568v1  

#### Abstract
Sentiment signals derived from sparse news are commonly used in financial analysis and technology monitoring, yet transforming raw article-level observations into reliable temporal series remains a largely unsolved engineering problem. Rather than treating this as a classification challenge, we prop...

---

### 30. [Wireless communication empowers online scheduling of partially-observable transportation multi-robot systems in a smart factory](https://arxiv.org/abs/2603.23967)

**Authors**: Yaxin Liao, Qimei Cui, Kwang-Cheng Chen, Xiong Li, Jinlian Chen, Xiyu Zhao, Xiaofeng Tao, Ping Zhang  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.23967v1  

#### Abstract
Achieving agile and reconfigurable production flows in smart factories depends on online multi-robot task assignment (MRTA), which requires online collision-free and congestion-free route scheduling of transportation multi-robot systems (T-MRS), e.g., collaborative automatic guided vehicles (AGVs). ...

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
