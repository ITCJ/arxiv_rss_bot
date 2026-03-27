# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-27 06:54:23 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DFLOP: A Data-driven Framework for Multimodal LLM Training Pipeline Optimization](https://arxiv.org/abs/2603.25120)

**Authors**: Hyeonjun An, Sihyun Kim, Chaerim Lim, Hyunjoon Kim, Rathijit Sen, Sangmin Jung, Hyeonsoo Lee, Dongwook Kim, Takki Yu, Jinkyu Jeong, Youngsok Kim, Kwanghyun Park  
**Category**: cs.DC  
**Published**: 2026-03-27  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2603.25120v1  

#### Abstract
Multimodal Large Language Models (MLLMs) have achieved remarkable advances by integrating text, image, and audio understanding within a unified architecture. However, existing distributed training frameworks remain fundamentally data-blind: they parallelize computation without accounting for variati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DFLOP: A Data-driven Framework for Multimodal LLM Training Pipeline Optimization

---

## 1. 主要贡献和创新点

### 解决的问题
现有的分布式训练框架（如 Megatron-LM、PyTorch）在训练 **Multimodal LLM (MLLM)** 时存在严重的**数据盲区**（data-blind）。这些框架假设所有 microbatch 的计算负载是均匀的，但在 MLLM 中，输入数据具有高度异质性（如单图、多图、视频等），导致：
- **Pipeline 阶段间负载不均衡**：modality encoder 和 LLM 架构不同，处理成本差异大。
- **输入依赖的吞吐波动**：图像分辨率、序列长度等动态变化，导致 throughput 不稳定。
- **GPU 利用率低、同步延迟高**，最终训练效率严重下降。

### 提出的新方法
作者提出 **DFLOP** —— 一个**数据驱动的 MLLM 训练流水线优化框架**，其核心思想是将**数据特征**显式地纳入并行策略和调度决策中。

#### 三大核心组件：
1. **Profiling Engine**  
   - **Model Profiler**：通过合成数据测量模型在不同输入形状下的内存消耗和 throughput。
   - **Data Profiler**：分析真实训练数据集中输入形状（如视觉 token 数、文本长度）的分布。
   - 输出：构建可预测的性能模型和数据分布统计。

2. **Data-aware 3D Parallelism Optimizer**  
   - 基于 Profiling 结果，在离线阶段搜索最优的 3D 并行配置（TP/PP/DP）。
   - 对 **modality encoder** 和 **LLM** 分别独立配置并行度，以最小化期望的 **makespan**。
   - 引入 **Inter-model Communicator** 抽象，解决不同 DP 组之间的通信问题。

3. **Online Microbatch Scheduler**  
   - 在运行时动态划分 global batch 为 microbatches。
   - 使用 ILP 或 LPT 启发式算法，平衡各 stage 的计算时间，减少 pipeline bubbles。
   - 支持 **Adaptive Correction** 机制，持续监控实际执行时间并修正预测偏差。

### 相比现有方法的优势
| 方面 | 传统方法（Megatron-LM/PyTorch） | DFLOP |
|------|-------------------------------|-------|
| 并行策略 | 全局统一的 3D 并行配置 | **模块级独立配置**（encoder vs LLM） |
| 调度方式 | 随机分配 microbatch | **基于预测的动态负载均衡** |
| 数据感知 | ❌ 完全忽略数据分布 | ✅ 显式建模输入形状影响 |
| 性能目标 | 最大化理论吞吐 | 最小化 **期望 makespan**（考虑数据分布） |

---

## 2. 核心实验方法和设置

### 使用的数据集
构建了一个混合数据集，模拟真实 MLLM 训练场景中的多样性：

| 数据集 | 类型 | 样本数 |
|--------|------|--------|
| LLaVA-Wild, AI2D, Infographic VQA | 单图（Single Image） | ~65k |
| M4-Instruct | 多图（Multiple Images） | 60k |
| LLaVA-Video | 视频（Video） | 60k |

> **总样本量约 225k**，涵盖多种视觉输入模式。

### 实验设置
- **硬件平台**：最多 8 节点，每节点 8×NVIDIA A100（NVLink 连接），共 64 GPUs。
- **网络**：800 Gbps InfiniBand。
- **评估模型**：
  - **LLaVA-OV**：SigLIP + Qwen-2.5 (7B, 32B, 72B) / Llama-3 (8B, 70B)
  - **InternVL-2.5**：InternViT + Qwen-2.5 (72B)
- **评估指标**：
  - **End-to-end training throughput**（samples/sec）
  - **Training time reduction**
  - **GPU 利用率、pipeline idle time**
  - **stage-wise throughput variance**

### 基线方法对比
- **Megatron-LM**：业界领先的 3D 并行框架。
- **Custom PyTorch Baseline**：基于 `torch.distributed` 手动调优的实现。
- 所有 baseline 均采用人工调参达到最佳性能。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **端到端吞吐提升**：相比 Megatron-LM 和 PyTorch，DFLOP 实现 **1.2× ~ 3.6× 的加速比**。
  - 最高可达 **3.6× 更快训练速度**。
- **训练时间显著缩短**：在多个配置下节省 **5~40 小时** 的训练时间。
- **GPU idle time 减少**：相比 baseline 降低 **82%~84%**。
- **Pipeline bubbles 大幅压缩**：实测 idle time 接近理论下限。

### 与基线方法对比结果
| 模型 | DFLOP 加速比（vs Megatron-LM） |
|------|-------------------------------|
| LLaVA-OV (Qwen-2.5 7B) | 1.3× |
| LLaVA-OV (Llama-3 8B) | 2.3× |
| LLaVA-OV (Qwen-2.5 32B) | 2.7× |
| LLaVA-OV (Qwen-2.5 72B) | 3.6× |
| InternVL-2.5 (72B) | 3.1× |

> 图 7 显示，随着模型规模增大，DFLOP 的优势更加明显。

### 消融实验结果
#### （1）组件贡献分析（图 10）
- **LLaVA-OV (Llama-3 8B)**：主要收益来自 **Data-aware 3D Parallelism Optimizer**（静态配置优化）。
- **LLaVA-OV (Qwen-2.5 32B)**：主要收益来自 **Online Microbatch Scheduler**（动态调度应对高异质性）。
- **InternVL-2.5 (72B)**：两者贡献相当，说明复杂模型需协同优化。

#### （2）自适应修正机制（图 15）
- 当异常输入频率 ≥3% 且延迟 >25% 时，**Adaptive Correction** 自动激活，带来正向净增益。
- 否则自动关闭，避免不必要的 profiling 开销（<4%）。

#### （3）跨模态泛化能力（图 9）
- 在 **Qwen2-Audio** 架构上测试，仍取得 **2×~4× 吞吐提升**，验证了 DFLOP 对音频等其他模态的有效性。

#### （4）可扩展性（图 12）
- 随着 GPU 节点从 1 增加到 32，DFLOP 的性能优势**持续扩大**。
- 原因：
  1. 更大的搜索空间允许更优的并行策略；
  2. 动态调度有效缓解大规模 DP 下的 straggler 问题。

---

## 4. 关键结论和发现

### 主要发现
1. **数据异质性是 MLLM 训练效率的关键瓶颈**，传统“数据盲”框架无法应对。
2. **将数据分布作为一等公民纳入优化问题**，可显著提升训练效率。
3. **静态配置 + 动态调度** 的联合设计是解决 MLLM pipeline 不平衡的核心路径。
4. **模块级独立并行化**（per-module 3D parallelism）比全局统一配置更具灵活性和性能优势。
5. DFLOP 的优化效果在**更大模型、更高异质性数据、更大集群**上更为显著。

### 方法的局限性
- **初始化开销**：Profiling 阶段耗时约 **7–10 分钟**，虽占总训练时间 <2.1%，但对短任务不划算。
- **仅支持 PyTorch/Megatron-LM**，目前不兼容 DeepSpeed（因其缺乏灵活的 pipeline parallelism 支持）。
- **Adaptive Correction** 依赖运行时监控，可能引入轻微不确定性。

### 未来工作方向
- 将 DFLOP 扩展至更多模态（如 3D point clouds、sensor fusion）。
- 支持在线 re-profiling，适应训练过程中数据分布漂移。
- 与编译器级优化（如 TorchInductor）结合，进一步提升 kernel 级效率。
- 探索轻量化 profiling，降低冷启动成本。

---

> ✅ **开源声明**：DFLOP 已开源，代码地址：[https://github.com/BDAI-Research/DFLOP](https://github.com/BDAI-Research/DFLOP)

</details>

---

### 2. [S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation](https://arxiv.org/abs/2603.25702)

**Authors**: Ligong Han, Hao Wang, Han Gao, Kai Xu, Akash Srivastava  
**Category**: cs.CL  
**Published**: 2026-03-27  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2603.25702v1  

#### Abstract
Block-diffusion language models offer a promising path toward faster-than-autoregressive generation by combining block-wise autoregressive decoding with within-block parallel denoising. However, in the few-step regime needed for practical acceleration, standard confidence-thresholded decoding is oft...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
块扩散语言模型（block-diffusion LLMs）通过结合**块级自回归生成**（block-wise autoregressive generation）和**块内并行去噪**（within-block parallel denoising），在理论上实现了比传统自回归（AR）模型更快的生成速度。然而，在实际应用中，为了实现加速，通常需要在极少数的去噪步骤（few-step regime）内完成解码。此时，标准的基于置信度阈值（confidence-thresholded decoding）的解码策略变得非常脆弱：
- **激进的阈值**会损害生成质量。
- **保守的阈值**则需要不必要的额外去噪步骤，降低了加速效果。

现有的解决方案要么需要**额外训练**（如引入辅助能量模型），要么增加**测试时计算开销**（如多采样重加权）。本文旨在解决这一矛盾：如何在不进行额外训练、不显著增加推理成本的前提下，提升块扩散模型在少步解码下的准确率-速度权衡（accuracy-speed tradeoff）。

### 提出的新方法：S2D2
论文提出了 **S2D2**（**S**elf-**S**peculative **D**ecoding for **D**iffusion LLMs），一种**无需训练的自推测解码框架**。

其核心思想是利用同一个预训练的块扩散模型，扮演两个角色：
1.  **起草者（Drafter）**：使用标准的块大小（如 B=32）进行块扩散解码，提出候选词元序列。
2.  **验证者（Verifier）**：将同一个模型的块大小设为1（B=1），使其退化为一个**自回归（AR）模型**，用于对起草者提出的词元进行局部验证。

S2D2 在标准的块扩散解码流程中插入了一个“推测性验证”（speculative verification）步骤：
- 起草者提出一组连续的待解码词元。
- 验证者以自回归的方式重新评估这些词元。
- 采用类似**推测性解码**（speculative decoding）的拒绝采样（rejection sampling）机制来决定接受多少个词元。
- 如果验证失败，则回退到标准的置信度阈值解码。

### 相比现有方法的优势
1.  **Training-Free（无需训练）**：这是 S2D2 最大的优势。它完全复用现有的预训练块扩散模型，无需任何微调、蒸馏或引入额外的网络参数。
2.  **Plug-and-Play（即插即用）**：作为一种纯推理时（inference-time）的优化策略，它可以无缝集成到现有的块扩散模型中。
3.  **高效性**：通过轻量级的路由策略（lightweight routing policies），仅在预期收益大于验证成本时才执行验证，避免了无谓的额外前向传播。
4.  **性能优越**：实验证明，S2D2 不仅能提高解码速度，还能同时提升生成质量，打破了传统上“加速必损质”的困境。

## 2. 核心实验方法和设置

### 使用的数据集
实验在以下四个基准数据集上进行，覆盖了数学推理、代码生成和指令遵循任务：
- **GSM8K**：小学数学应用题，评估数学推理能力。
- **MBPP** 和 **HumanEval**：代码生成任务，评估编程能力。
- **IFEval**：指令遵循评估，评估模型遵循复杂指令的能力。

### 实验设置和评估指标
- **模型**：在五个来自三个主流块扩散家族的模型上进行了评估：
  - **SDAR** (1.7B/4B/8B)
  - **Fast-dLLM v2**
  - **LLaDA2.1-Mini**
- **评估指标**：
  - **准确性（Accuracy）**：在各任务上的得分。
  - **加速比（Speedup）**：相对于自回归基线（block size=1）的推理时间加速倍数。
  - **准确率-速度权衡（Accuracy-Speed Tradeoff）**：综合评估模型效率的核心指标。
- **基线方法对比**：
  - **自回归解码（AR）**：块大小为1的标准自回归解码。
  - **标准块扩散解码（Standard Block-Diffusion）**：使用静态或动态置信度阈值的解码方法，作为主要的强基线。

### 路由策略
为了决定何时进行验证，S2D2 设计了多种轻量级路由策略：
- **Minimum-span**：当第一个连续掩码跨度（contiguous mask span）长度超过阈值 `Tspan` 时触发验证。
- **Score-threshold**：当预测的可接受前缀长度分数 `s` 超过阈值 `Tscore` 时触发。
- **Hysteresis**：使用迟滞开关，避免在推测模式和扩散模式之间频繁切换。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
S2D2 在所有测试的模型和任务上均取得了显著优于强基线的结果。

- **在 SDAR-1.7B 上**：
  - **S2D2 (config-B)** 达到了 **4.7倍** 的自回归加速，而**平均准确率提升了 4.5 个百分点**（从 48.4 提升到 52.9）。
  - 这相当于比调优后的动态解码基线快 **1.57倍**，同时准确率更高。

- **在 SDAR-8B-Chat 上**：
  - S2D2 在大块大小（如 B=32）下表现尤为出色，解决了标准扩散解码在此类设置下不稳定的问题。
  - 其准确率-速度曲线（见图3）始终位于基线之上，证明了其优越的权衡能力。

- **在 LLaDA2.1-Mini 上**：
  - S2D2 与模型内置的“编辑”（editing）自校正机制**互补**。
  - 在一个保守设置下，S2D2 比静态基线快 **4.4倍**，且准确率**略高**（79.3% vs 79.2%）。

### 消融实验结果
- **路由策略**：消融实验（表6-10）表明，不同的路由策略都能有效提升性能。例如，`minimum-span` 策略简单却非常有效，而 `score-threshold` 策略在大块大小下增益更明显。
- **验证成本**：实验分析了验证带来的额外计算成本，并证明了路由策略的有效性——只有在预期收益高时才进行验证，从而保证了整体效率。
- **接受估计器**：研究了多种用于预测可接受前缀长度的轻量级估计器（如熵基、边际基），发现虽然硬边际阈值估计最准确，但软熵基估计器在实际路由中效果更好，因为它更鲁棒且易于使用。

## 4. 关键结论和发现

### 主要发现
1.  **自回归模式是天然的序列级批评家**：将块扩散模型的 B=1 模式用作验证者，本质上是利用其自回归特性对扩散解码的输出进行局部、序列级的质量检查。
2.  **S2D2 是一种有效的局部能量修正**：作者从理论角度分析，认为 S2D2 的验证过程可以被解释为一种**随机的、贪婪的局部残差能量修正**（residual energy correction），它倾向于接受那些在自回归视角下能量更低（即更合理）的词元提议。
3.  **无需训练也能实现高质量加速**：S2D2 成功地证明了，通过巧妙地设计推理时的解码流程，可以在不修改模型本身的情况下，同时获得速度和质量的双重提升。

### 方法的局限性
1.  **验证范围有限**：S2D2 当前只验证第一个连续的掩码跨度，而不是整个块，这限制了其全局修正能力。
2.  **依赖于模型架构**：该方法依赖于块扩散模型能够平滑地退化为 B=1 的自回归模式。对于其他类型的扩散模型可能不适用。
3.  **路由策略的设计**：虽然轻量级，但路由策略的超参数（如 `Tspan`, `Tscore`）仍需根据具体模型和任务进行调整。

### 未来工作方向
1.  **扩展验证范围**：探索验证多个非连续跨度或更长的序列片段。
2.  **更智能的路由**：开发更复杂的、甚至可学习的路由策略，以更精准地判断验证的价值。
3.  **与其他技术结合**：将 S2D2 与模型内置的自校正机制（如 LLaDA 的 token editing）或分层批处理（hierarchical batching）进一步结合，追求更极致的性能。
4.  **理论深化**：更深入地建立 S2D2 与能量模型、变分推断等理论框架之间的联系。

</details>

---

### 3. [Prune as You Generate: Online Rollout Pruning for Faster and Better RLVR](https://arxiv.org/abs/2603.24840)

**Authors**: Haobo Xu, Sirui Chen, Ruizhong Qiu, Yuchen Yan, Chen Luo, Monica Cheng, Jingrui He, Hanghang Tong  
**Category**: cs.CL  
**Published**: 2026-03-27  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.24840v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Large Language Models (LLMs). However, methods such as GRPO and DAPO suffer from substantial computational cost, since they rely on sampling many rollouts for each prompt. Moreover, in RLVR...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Prune as You Generate: Online Rollout Pruning for Faster and Better RLVR**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **高计算成本**：在 **Reinforcement Learning with Verifiable Rewards (RLVR)** 中，如 **GRPO** 和 **DAPO** 等方法依赖于对每个提示生成大量 **rollouts**，导致训练过程计算开销巨大。
- **稀疏学习信号**：由于奖励是二元的（0/1），rollout 组内奖励分布常趋于极端（全正确或全错误），导致组内奖励方差低，优势估计退化为零，梯度消失，学习信号弱。

### **提出的新方法：ARRoL（Accelerating RLVR via online RoLlout Pruning）**
- **在线 rollout 剪枝（Online Rollout Pruning）**：
  - 在生成过程中动态剪枝低质量 rollout，而非事后处理。
  - 引入一个轻量级的 **quality head**，实时预测部分 rollout 的成功概率，并据此进行早期剪枝决策。
- **显式控制奖励平衡**：
  - 剪枝策略旨在保留一个 **correctness-balanced** 的 rollout 子集（正负样本比例接近 0.5），以增强组内奖励方差，强化学习信号。
- **系统级优化设计**：
  - 将剪枝逻辑集成到推理引擎（vLLM）中，在后端执行 early pruning，并在前端重新批处理幸存 rollout，实现端到端加速。
- **测试时扩展（Test-time Scaling）复用**：
  - 训练得到的 quality head 可在推理阶段作为投票权重，替代朴素多数投票（majority vote），提升最终答案聚合准确性。

### **相比现有方法的优势**
| 方面 | ARRoL 优势 |
|------|------------|
| **效率** | 实现 **1.6–1.7× 的端到端训练加速**，显著降低 rollout 生成与 log-prob 计算开销 |
| **效果** | 平均准确率提升 **+2.30 至 +2.99**，尤其在难题上增益更明显（如 AIME 上 +10.00） |
| **学习信号质量** | 显式控制组内 reward 平衡，避免稀疏信号问题，提升梯度有效性 |
| **通用性** | 适用于 GRPO 和 DAPO 等多种 RLVR 算法，且在 Qwen-3 和 LLaMA-3.2 系列模型上均有效 |
| **测试时增益** | quality head 可用于 test-time scaling，带来额外 **+8.33 的平均准确率提升** |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：
  - `Dapo-Math-17K`：包含 17K 数学问题，用于 RLVR 训练。
- **评估数据集**：
  - `Math500`：来自 MATH 数据集的 500 道高中水平数学题。
  - `Minervamath`：272 道 MIT 课程级别的定量推理题。
  - `OlympiadBench`：8,476 道奥数级别数学与物理题。
  - `AMC'23`, `AIME'24`, `AIME'25`：美国数学竞赛题，难度递增，答案为整数（0–999）。

### **实验设置**
- **模型系列**：
  - `Qwen-3`（1.7B, 4B, 8B）
  - `LLaMA-3.2`（1B）
- **算法基线**：
  - `GRPO`（Group Relative Policy Optimization）
  - `DAPO`（大规模开源 RLVR 系统）
- **对比变体**：
  - `GRPO + ARRoL`
  - `DAPO + ARRoL`
- **评估指标**：
  - **Average Accuracy**：多个数据集上的平均准确率。
  - **pass@16**：在 AMC/AIME 等小数据集上报告至少一次正确的概率。
  - **maj@32**：32 次采样下的多数投票准确率（test-time scaling 场景）。
  - **Wall-clock time**：实际训练耗时，衡量端到端速度提升。
- **关键超参**：
  - 剪枝检测长度 $L_{\text{detect}} = 512$
  - 目标保留率 $K = 0.5$
  - 目标正样本比例 $p = 0.5$

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **GRPO + ARRoL 结果（表1）**
| 模型 | 方法 | 平均准确率 ↑ | 速度提升 |
|------|------|-------------|--------|
| Qwen-3-1.7B | GRPO | 34.79 | — |
| | +ARRoL | **37.09 (+2.30)** | 1.61× |
| Qwen-3-4B | GRPO | 51.01 | — |
| | +ARRoL | **53.54 (+2.53)** | 1.63× |
| Qwen-3-8B | GRPO | 56.66 | — |
| | +ARRoL | **59.53 (+2.87)** | 1.62× |
| LLaMA-3.2-1B | GRPO | 14.63 | — |
| | +ARRoL | **17.49 (+2.86)** | 1.67× |

> 💡 在 AIME'24 上，Qwen-3-8B 的准确率从 56.67 提升至 **66.67（+10.00）**。

#### ✅ **DAPO + ARRoL 结果（表2）**
| 模型 | 方法 | 平均准确率 ↑ | 速度提升 |
|------|------|-------------|--------|
| Qwen-3-1.7B | DAPO | 36.43 | — |
| | +ARRoL | **39.42 (+2.99)** | 1.70× |

> 表明 ARRoL 对不同 RLVR 算法均有效。

#### ✅ **Test-time Scaling 性能（表3）**
| 模型 | 投票方式 | AIME'24 准确率 |
|------|----------|----------------|
| Qwen-3-1.7B | Majority | 16.7 |
| | DeepConf（trace confidence） | 16.7 |
| | **ARRoL（quality head 权重）** | **23.3 (+6.6)** |
| Qwen-3-8B | Majority | 23.3 |
| | DeepConf | 23.3 |
| | **ARRoL** | **33.3 (+10.0)** |

> ARRoL 在 test-time scaling 中带来高达 **+8.33 的额外增益**，优于基于 log-prob 的 DeepConf。

### **消融实验结果**

#### 🔍 **vs. 随机剪枝（Random Pruning）**
- ARRoL 明显优于随机剪枝（表4）。
- ARRoL 能将组内正样本比例 $E[p]$ 推向 0.5，显著提高 $E[p(1-p)]$（奖励方差代理），说明其确实增强了学习信号。

#### ⚙️ **效率分解（表5）**
| 阶段 | GRPO (s) | ARRoL (s) | 加速比 |
|------|---------|----------|-------|
| Rollout Generation | 106.82 | 72.96 | **1.46×** |
| Log-prob Computation | 18.40 | 10.02 | **1.84×** |
| Model Update | 63.05 | 30.26 | **2.08×** |

> 后两个阶段因 rollout 数量减少而大幅加速，生成阶段受限于需先运行至 $L_{\text{detect}}$。

#### 📊 **保留率 $K$ 的影响（表6）**
| $K$ | 平均准确率 | 速度提升 |
|-----|-----------|--------|
| 0.25 | 32.46 ↓ | 2.33× |
| 0.50 | **33.34** | 1.61× |
| 0.75 | 32.68 ↓ | 1.17× |
| 1.00 | 32.36 ↓ | 1.00× |

> $K=0.5$ 是精度与效率的最佳权衡；过低会导致信息丢失，性能下降。

#### 🕒 **Wall-clock 收敛曲线（图4）**
- ARRoL 在更短时间内达到相同甚至更高的训练奖励，验证了其 **更快的实际收敛速度**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **“Less is More” 范式成立**：
   - 通过智能剪枝减少 rollout 数量，反而能提升训练效率与模型性能。
   - 关键在于保留一个 **reward-balanced** 的子集，从而增强学习信号。
2. **quality head 有效且高效**：
   - 轻量级 head 可在训练中在线学习，准确预测 rollout 成功率（~80% 准确率）。
   - 其得分比传统 trace confidence（如 DeepConf）更具判别力，且与最终正确性更相关。
3. **系统设计实现真实加速**：
   - 在 vLLM 后端集成剪枝，结合前端 re-batching，实现了 **1.6–1.7× 的端到端 wall-clock 加速**。
4. **训练组件可迁移至推理**：
   - quality head 不仅用于训练剪枝，还可作为 test-time voting 权重，进一步提升推理性能。

### **局限性**
1. **任务范围有限**：
   - 当前研究聚焦于具有 **可验证奖励（verifiable rewards）** 的数学推理任务。
   - 是否适用于无明确奖励信号的任务（如对话、创作）尚待验证。
2. **检测延迟限制加速上限**：
   - 必须生成到 $L_{\text{detect}}=512$ 才能剪枝，因此 rollout 生成阶段的加速有限（仅 1.46×）。
3. **依赖 rollout 标签监督**：
   - quality head 依赖最终 reward 作为标签，无法应用于 reward 不可观测的场景。

### **未来工作方向**
- 将 ARRoL 扩展至其他 RL 场景，如 **tool-use agents** 或 **UI interaction**。
- 探索更早的检测机制（如基于前缀 token 的预测）以进一步提升生成阶段加速。
- 研究无监督或自监督方式训练 quality head，以适应 reward 稀疏或不可得的环境。
- 结合 speculative decoding 与 ARRoL，实现双重加速。

---

> **代码已开源**：[https://github.com/Hsu1023/ARRoL](https://github.com/Hsu1023/ARRoL)

</details>

---

### 4. [Residual Attention Physics-Informed Neural Networks for Robust Multiphysics Simulation of Steady-State Electrothermal Energy Systems](https://arxiv.org/abs/2603.23578)

**Authors**: Yuqing Zhou, Ze Tao, Fujun Liu  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 9.5  
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
该研究针对**稳态电热耦合多物理场系统**（steady-state electrothermal coupled multiphysics systems）的数值模拟难题，解决以下挑战：
- 强非线性场耦合（如速度、压力、电势、温度之间的相互作用）
- 温度依赖的变系数（temperature-dependent variable coefficients）
- 复杂界面动力学（oblique interface with jump conditions）
- 传统 PINN 在局部梯度集中区域（如边界层、接口附近）精度不足且训练不稳定

这些问题在微能源系统、电驱动热调节器、液冷电池等可持续能源应用中具有重要工程意义。

---

### 提出的新方法与创新思路
作者提出了一种新型框架：**Residual Attention Physics-Informed Neural Network (RA-PINN)**，其核心创新包括：

#### ✅ 统一五场算子建模（Unified Five-Field Operator Formulation）
- 将速度 $u,v$、压力 $p$、电势 $\phi$ 和温度 $T$ 联立为一个统一的向量场 $U = [u,v,p,\phi,T]$。
- 构造统一的残差算子 $ \mathcal{N}(U) = 0 $，实现对所有物理场的一致性求解。

#### ✅ 残差注意力机制（Residual-Attention Mechanism）
- 结合 **residual-connected 特征传播** 与 **attention-guided 通道调制**：
  - 残差连接保持深层梯度稳定传输；
  - 注意力门控放大携带陡峭梯度或局部耦合结构的信息通道。
- 显著增强模型对**局部强梯度区**（如界面、边界层）的捕捉能力。

#### ✅ 自适应残差点采样（Adaptive Collocation Sampling）
- 动态调整训练点分布：高残差点被保留并增加权重，低残差点可剔除或降权。
- 实现“动态聚焦”于难拟合区域，提升优化效率与精度。

---

### 相比现有方法的优势
| 方面 | RA-PINN 优势 |
|------|--------------|
| **表示能力** | 比 Pure-MLP 更能捕捉复杂耦合结构；比 LSTM-PINN 更适合空间局部特征 |
| **鲁棒性** | 在变系数、间接约束（pressure gauge）、斜界面场景下仍保持高保真度 |
| **精度一致性** | 所有测试案例中均取得最低误差，尤其在强非线性条件下优势显著 |

---

## 2. 核心实验方法和设置

### 数据集与基准任务
论文未使用真实世界数据集，而是构建了四个**人工设计的电热耦合 PDE 基准问题**，定义在单位正方形域 $\Omega = [0,1]\times[0,1]$ 上：

| Case | 描述 |
|------|------|
| **Case 1**: Constant-Coefficient Coupling | 常系数耦合系统，作为基础对照 |
| **Case 2**: Pressure-Gauge Constraint | 用零均值积分条件替代直接压力锚定（$\int_\Omega p\,d\Omega=0$） |
| **Case 3**: Temperature-Dependent Transport | 黏度 $v(T)$ 与热扩散率 $\alpha(T)$ 随温度变化：$v(T)=v_0(1+\beta_v T)$ |
| **Case 4**: Oblique-Interface Consistency | 斜界面分割材料参数，并施加跳跃通量条件 |

每个案例提供参考解（ground truth），用于定量评估。

---

### 实验设置与评估指标

#### ✅ 评估指标
- **MSE**（Mean Squared Error）
- **RMSE**（Root Mean Squared Error）
- **MAE**（Mean Absolute Error）
- **Relative L2 Error**（相对 L2 范数误差）

报告各物理场（$u,v,p,\phi,T$）及平均值（Avg.）。

#### ✅ 基线方法对比
- **Pure-MLP**: 标准全连接前馈网络
- **LSTM-PINN**: 引入时序记忆结构的空间建模
- **pLSTM-PINN**: 并行 LSTM 架构，适用于多场联合预测

#### ✅ 训练细节
- 输入仅为坐标 $(x,y)$（稳态无时间维度）
- 输出为五维场预测 $U(x,y;\theta)$
- 使用 Adam 优化器进行端到端训练
- 损失函数包含 PDE 残差、边界条件、数据项、正则化项及特殊约束（如 pressure gauge、interface jump）

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{res}}\|\mathcal{R}\|^2 + \lambda_b\|\mathcal{R}_b\|^2 + \cdots + \lambda_{\text{gauge}}|\mathcal{R}_g|^2 + \lambda_r\|\mathcal{R}_r\|^2
$$

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Case | 方法 | Avg. MSE | Avg. Relative L2 Error |
|------|------|----------|------------------------|
| Case 1 | **RA-PINN** | **9.083×10⁻⁷** | **3.235×10⁻³** |
|       | LSTM-PINN     | 2.901×10⁻⁶ | 5.695×10⁻³ |
|       | pLSTM-PINN    | 1.164×10⁻⁵ | 1.205×10⁻² |
|       | Pure-MLP      | 2.249×10⁻⁴ | 5.105×10⁻² |
|  
| Case 2 | **RA-PINN** | **2.053×10⁻⁶** | **7.660×10⁻³** |
|       | LSTM-PINN     | 3.956×10⁻⁶ | 1.038×10⁻² |
|       | pLSTM-PINN    | 9.551×10⁻⁵ | 3.608×10⁻² |
|       | Pure-MLP      | 1.642×10⁻⁴ | 4.868×10⁻² |
|  
| Case 3 | **RA-PINN** | **7.119×10⁻⁹** | **5.065×10⁻³** |
|       | LSTM-PINN     | 1.398×10⁻⁸ | 7.155×10⁻³ |
|       | pLSTM-PINN    | 1.719×10⁻⁴ | 8.456×10⁻¹ |
|       | Pure-MLP      | 8.434×10⁻⁷ | 3.031×10⁻² |
|  
| Case 4 | **RA-PINN** | **9.845×10⁻⁸** | **1.377×10⁻³** |
|       | LSTM-PINN     | 1.159×10⁻⁷ | 1.449×10⁻³ |
|       | pLSTM-PINN    | 9.296×10⁻⁵ | 3.895×10⁻² |
|       | Pure-MLP      | 5.948×10⁻⁶ | 1.061×10⁻² |

> 🔍 **观察**：RA-PINN 在所有案例中均达到最小误差，尤其在 Case 3（温度依赖）中表现远超其他方法。

---

### 与基线方法的对比结果

| 对比维度 | 结果 |
|--------|------|
| **总体精度** | RA-PINN 在全部 4 个案例中取得最优的 MSE、RMSE、MAE 和 Relative L2 错误 |
| **稳定性** | 在 interface-dominated 和 variable-coefficient 场景下，传统 PINN 容易失败，而 RA-PINN 保持结构保真度 |
| **视觉质量** | 图像级对比显示 RA-PINN 最接近 ground truth，伪影最少，边缘最清晰 |

---

### 消融实验分析（隐含于案例比较中）
虽然没有显式消融表，但从不同 case 的设计可视为一种**功能模块验证**：
- Case 1 → 验证基本耦合能力
- Case 2 → 验证对间接约束（gauge condition）的处理能力
- Case 3 → 验证对非线性变系数系统的鲁棒性
- Case 4 → 验证对几何复杂界面的适应性

→ RA-PINN 在所有扩展场景中持续领先，说明其架构具备良好的泛化性和模块兼容性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **RA-PINN 是目前最准确的 steady electrothermal PINN 求解器之一**，在多种复杂物理场景下均优于主流基线。
2. ✅ **残差注意力机制有效缓解了多场耦合中的优化不平衡问题**，使网络能同时关注平滑大尺度场与局部尖锐过渡。
3. ✅ **自适应采样显著提升了高残差区域的学习效率**，特别是在斜界面和温度敏感区表现出更强的收敛能力。
4. ✅ 在 **temperature-dependent coefficient** 和 **oblique interface** 这类极具挑战性的场景中，RA-PINN 展现出明显优于 LSTM 类模型的鲁棒性。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| ⚠️ **训练耗时高** | 如 Table 2 所示，RA-PINN 训练时间最长（Case 3 达 39.81 小时），远高于 pLSTM-PINN（~4h）和 Pure-MLP（~4.5h） |
| ⚠️ **计算资源需求大** | 残差注意力与动态重采样增加了内存与计算开销 |
| ⚠️ **尚未推广至瞬态或多尺度系统** | 当前仅验证稳态问题，动态演化与多尺度耦合仍需进一步研究 |

---

### 未来工作方向
1. **加速训练策略**：探索更高效的 adaptive sampling 策略、混合精度训练或分布式并行。
2. **拓展至 transient systems**：将 RA-PINN 推广到时间依赖的电热耦合问题（如热冲击响应）。
3. **硬件集成与数字孪生应用**：结合 real-time sensing 数据，构建面向微能源设备的 **digital twin workflow**。
4. **与其他 neural operator 方法融合**：尝试与 FNO、DeepONet 或 PI-Transformer 结合，提升跨配置泛化能力。

---

## 总结

📌 **RA-PINN 成功地将 residual-connected 架构与 attention 机制引入 PINN 框架，在稳态电热耦合多物理场模拟中实现了前所未有的精度与鲁棒性。**

尽管存在较高的训练成本，但其在复杂工程场景下的卓越表现使其成为下一代高保真 energy system modeling 的有力候选工具。该工作为 **physics-informed machine learning** 在可持续能源系统中的深度应用奠定了坚实基础。

</details>

---

### 5. [Symbolic--KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning](https://arxiv.org/abs/2603.23854)

**Authors**: Salah A Faroughi, Farinaz Mostajeran, Amirhossein Arzani, Shirko Faroughi  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.23854v1  

#### Abstract
Symbolic discovery of governing equations is a long-standing goal in scientific machine learning, yet a fundamental trade-off persists between interpretability and scalable learning. Classical symbolic regression methods yield explicit analytic expressions but rely on combinatorial search, whereas n...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Symbolic-KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Scientific Machine Learning (SciML)** 中存在一个根本性权衡：
- **Symbolic Regression**（如遗传算法、SINDy）能生成可解释的解析表达式，但依赖组合搜索，计算成本高且难以扩展到高维。
- **Neural Networks**（如MLP、PINN）可高效处理大规模、高维数据，但模型“黑箱”，缺乏可解释性。

本文旨在**弥合可解释性与可扩展性之间的鸿沟**，实现从数据中直接学习出紧凑、人类可读的符号化表达式，而无需后处理（post-hoc symbolic fitting）。

### 🚀 提出的新方法：Symbolic-KAN
提出 **Symbolic Kolmogorov-Arnold Network (Symbolic-KAN)**，一种将离散符号结构嵌入可训练深度网络的新架构。

#### 核心思想：
- 基于 **Kolmogorov-Arnold Representation Theorem (KART)**，将多元函数表示为一元函数的叠加。
- 引入三大机制实现**从连续混合到离散符号选择**的过渡：
  1. **Analytic Primitive Library**：预定义一组可解释的基元函数（如 `sin`, `cos`, `x^2`, `log`, `tanh` 等）。
  2. **Hierarchical Gating Mechanism**：
     - **Primitive Selection Gate**：通过 Gumbel-Softmax 逐步将每个边上的基元组合“锐化”为 one-hot 选择。
     - **Edge Selection Mask**：每个神经元只保留一个输入边（即一个投影方向）。
     - **Unit Gate**：控制神经元是否激活，实现结构稀疏化。
  3. **Symbolic Regularization**：
     - **Entropy Loss**：鼓励 primitive gates 收敛到 one-hot 分布。
     - **Non-Maximum Suppression (NMS)**：防止同一单元不同边选择相同基元，提升多样性。

最终，经过训练和“硬化”（hardening），网络退化为一个**紧凑的闭式符号表达式**，每个活跃单元仅对应一个基元和一个投影方向。

### 🔍 相比现有方法的优势
| 方法 | 缺陷 | Symbolic-KAN 的改进 |
|------|------|---------------------|
| **标准 KAN** | 使用 B-spline 或 Chebyshev 多项式等非解析基，最终表达式冗长、不可读 | 使用**解析基元库**，输出形式天然可解释 |
| **SINDy / Genetic SR** | 需要预设候选库，无法发现新组合；搜索空间大，效率低 | 可作为**可微分的前处理模块**，自动发现重要基元，为 SINDy 提供优化后的候选库 |
| **PINN / cPIKAN** | 黑箱模型，参数估计不稳定，外推能力差 | 输出结构反映真实物理机制，**泛化性和稳定性更强** |

---

## 2. 核心实验方法和设置

### 📊 数据集与任务
在三类典型 SciML 场景下进行验证：

| 任务 | 描述 |
|------|------|
| **Data-driven Regression** | 学习人工构造的多元函数（如 $F(x)=x^2$, $F(x)=\frac{\sin(3x)}{1+x^2} + 0.4\cos(5x)$） |
| **Inverse Dynamical Systems** | 从轨迹数据中识别 Van der Pol 振子的未知参数（含非整数幂次 $x^{2.15}$） |
| **Physics-informed PDE Learning** | 正向求解与反演两类 PDE：<br>- **Reaction-Diffusion Equation**（反演反应系数 $k$）<br>- **Laplace Equation**（正向求解） |

### ⚙️ 实验设置
- **网络结构**：多层 Symbolic-KAN，每层 $K_e$ 个单元，每个单元有 $E$ 条边。
- **训练流程两阶段**：
  1. **Stage I**：软选择训练，使用 Gumbel-Softmax 和温度退火（annealing）逐步锐化 gates。
  2. **Stage II**：硬选择微调，固定 one-hot 结构，用 L-BFGS 微调剩余参数。
- **损失函数**：
  $$
  \mathcal{L} = \lambda_{\text{data}}\mathcal{L}_{\text{data}} + \mathcal{L}_{\text{phys}} + \lambda_{\text{sel}}(t)\mathcal{L}_{\text{sel}} + \lambda_{\text{unit}}\mathcal{L}_{\text{unit}} + \lambda_{\text{bias}}\mathcal{L}_{\text{bias}}
  $$
  其中 $\mathcal{L}_{\text{sel}}$ 包含熵正则和 NMS 惩罚。

### 📈 评估指标
- **相对误差 (Relative Error)**：
  $$
  \mathcal{E}(F) = \frac{\|F_{\text{true}} - F_{\text{pred}}\|}{\|F_{\text{true}}\|}
  $$
- **参数识别误差**：比较预测参数与真值的相对偏差。
- **符号结构一致性**：检查最终选择的 primitive 是否与真实解的数学结构一致（如是否选中 `sin`, `sinh` 等）。

### 🆚 基线方法对比
| 基线 | 说明 |
|------|------|
| **PINN** | 标准 Physics-Informed Neural Network，使用 `tanh` 激活 |
| **cPIKAN** | Chebyshev-based Physics-Informed KAN，使用多项式基 |
| **SINDy** | Sparse Identification of Nonlinear Dynamics（文中未直接对比，但作为动机提及） |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### （1）Van der Pol 振子参数识别（逆问题）
| 时间区间 | 方法 | 参数 $a$ | 参数 $\mu$ | 参数 $c$ | 轨迹误差 $\mathcal{E}(u)$ |
|--------|------|---------|----------|---------|------------------------|
| $[0,20]$ | Symbolic-KAN | 1.0000 | 0.0099 | 0.9998 | $6.02\times10^{-4}$ |
| $[0,50]$ | Symbolic-KAN | 0.9999 | 0.0093 | 0.9992 | $5.87\times10^{-3}$ |

> ✅ 所有参数均被高精度恢复，即使在更长时域下仍保持稳定。

#### （2）Reaction-Diffusion 方程（反演 $k$）
| 域 | 方法 | 验证误差 $\mathcal{E}(u)$ | 识别 $k$ | 误差 |
|----|------|------------------------|--------|------|
| $[-2,2]$ | Symbolic-KAN | $5.93\times10^{-4}$ | 0.6994 | <0.1% |
| $[-4,4]$ | Symbolic-KAN | $9.37\times10^{-3}$ | 0.6985 | ~0.2% |

> ✅ 在更大域上仍保持高精度，远优于基线。

#### （3）Laplace 方程（正向求解）
| 方法 | 验证误差 $\mathcal{E}(u)$ |
|------|------------------------|
| **Symbolic-KAN** | $1.11\times10^{-3}$ |
| **cPIKAN** | $8.76\times10^{-3}$ |
| **PINN** | $2.71\times10^{-3}$ |

> ✅ 误差比 cPIKAN 降低 **87%**，比 PINN 降低 **59%**。

### 📉 与基线方法对比结果
| 对比项 | Symbolic-KAN 表现 |
|-------|------------------|
| **数值精度** | 显著优于 PINN 和 cPIKAN，尤其在大域、强非线性场景 |
| **参数识别** | 在 Van der Pol 和 Reaction-Diffusion 中均实现高精度反演 |
| **符号结构发现** | 自动选出与真实解一致的基元（如 `sin`, `sinh`, `cos`），体现**机制可解释性** |
| **外推能力** | 在 $F(x)=x^2$ 实验中，外推区域仍保持正确趋势，残差极小 |

### 🔬 消融实验（隐含分析）
虽然未明确列出消融表，但以下设计体现了关键组件作用：
- **Gumbel-Softmax + 温度退火**：确保训练稳定，避免早熟收敛。
- **Entropy + NMS 正则**：引导模型选择简洁、多样化的基元组合。
- **Unit Gating**：支持结构稀疏化，生成紧凑模型。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Symbolic-KAN 成功实现了可解释性与可扩展性的统一**：
   - 不再是“先拟合后解释”，而是**原生地学习出符号表达式**。
2. **所学符号结构反映真实物理机制**：
   - 在 Laplace 方程中自动选出 `sin` 和 `sinh`，与解析解 $u(x,y)=\sin(\pi x)\sinh(\pi y)$ 完全一致。
3. **兼具高精度与强泛化能力**：
   - 在插值与外推区域均表现优异，残差极小。
4. **可作为“符号发现引擎”**：
   - 发现的关键基元可用于构建更高效的 SINDy 候选库，形成“Symbolic-KAN + SINDy”协同框架。

### ⚠️ 局限性
- **仍未完全解决外推问题**：尽管表现优于基线，但神经网络固有的外推风险依然存在。
- **依赖基元库的设计**：若真实解涉及库外函数（如特殊函数），可能无法准确捕捉。
- **训练复杂度较高**：双阶段训练 + 多种正则项，调参难度大于标准 PINN。

### 🔮 未来工作方向
1. **动态扩展基元库**：允许模型在训练中“发明”新的函数组合。
2. **结合贝叶斯推理**：为符号表达式提供不确定性量化。
3. **应用于更复杂的 PDE 系统**：如 Navier-Stokes、Maxwell 方程等。
4. **与 symbolic reasoning 系统集成**：实现从数据到理论的闭环科学发现。

---

## 总结
**Symbolic-KAN 是迈向“机制可解释机器学习”的重要一步**。它不仅提升了模型的透明度，更通过**可微分符号发现**机制，为科学发现提供了新的工具范式。其在多种 SciML 任务中的卓越表现，证明了将 **KAN 架构 + 符号先验 + 结构正则化** 相结合的巨大潜力。

</details>

---

### 6. [CoordLight: Learning Decentralized Coordination for Network-Wide Traffic Signal Control](https://arxiv.org/abs/2603.24366)

**Authors**: Yifeng Zhang, Harsh Goel, Peizhuo Li, Mehul Damani, Sandeep Chinchali, Guillaume Sartoretti  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 9.5  
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
- **部分可观测性（Partial Observability）**：在去中心化多智能体系统中，每个路口（agent）仅能获取局部交通状态，难以全面感知全局交通流，导致决策短视（myopic behavior）。
- **协调困难（Coordination Difficulty）**：缺乏有效的机制来建模相邻路口之间的动态依赖关系，影响网络级协同优化。

现有方法如独立学习（Independent Learning）或基于图注意力的方法（如CoLight）往往忽视了对**队列动态演化**和**邻居状态-动作依赖**的精细建模，限制了其在大规模复杂路网中的扩展性和稳定性。

---

### **提出的新方法与创新思路**

#### **(1) Queue Dynamic State Encoding (QDSE)**  
一种新颖的**状态表示方法**，基于车辆排队动力学模型构建：
- 包含六个车道级特征向量：
  - `Q(t)`：停止车辆数（队列长度）
  - `Nin(t)`：进入车辆数
  - `Nout(t)`：离开车辆数
  - `Nr(t)`：移动车辆总数
  - `Nfr(t)`：紧随首车后的移动车辆数
  - `Dfr(t)`：首辆移动车与队尾的距离
- **优势**：不仅反映当前拥堵状态，还能**预测未来潜在拥堵**（通过估计即将加入队列的车辆），使策略更具前瞻性而非被动响应。

#### **(2) Neighbor-aware Policy Optimization (NAPO)**  
一个完全去中心化的**MARL算法框架**，旨在提升训练稳定性和协调能力：
- 引入**注意力机制**（Attention Mechanism）识别“有影响力的邻居”（influential neighbors），实现**自适应加权协作**。
- 设计**增强型Actor-Critic网络架构**：
  - **Actor Network**：采用时空注意力网络（STN），融合空间（Spatial Aggregation Unit）和时间（Temporal Aggregation Unit）维度信息。
  - **Critic Network**：引入**特权本地批评器**（Privileged Local Critic），包含状态编码器和状态-动作解码器（State-action Decoder），用于聚合邻居的状态-动作历史依赖，提升价值函数估计准确性。
- 利用改进的**优势函数计算方式**，将邻居的影响纳入策略更新，提高信用分配（credit assignment）精度。

---

### **相比现有方法的优势**
| 维度 | CoordLight 的优势 |
|------|------------------|
| **状态表示** | 超越传统 `vehicle count` 或 `pressure`，提供更丰富、可预测的动态信息 |
| **协调机制** | 不依赖通信或集中式训练（CTDE），实现高效、可扩展的去中心化协调 |
| **泛化性与鲁棒性** | 在不同规模、流量强度下均表现优异；对传感器噪声具有较强容忍度 |
| **实用性** | 平衡了模型复杂度与性能，适合实际部署 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
基于开源仿真平台 **CityFlow**，使用三个真实城市路网：
- **Jinan (China)**：3×4 = 12 个交叉口
- **Hangzhou (China)**：4×4 = 16 个交叉口
- **New York (USA)**：7×28 = 196 个交叉口（最大规模）

每组数据包含多个不同交通需求场景（共7种流量模式），涵盖低、中、高流量条件。

| 数据集 | 类型 | 车辆总量 | 最大到达率 (veh/min) |
|--------|------|----------|-----------------------|
| Jinan | DJN(1)-(3) | ~4k–6k | 136 |
| Hangzhou | DHZ(1), PHZ(2) | ~3k, ~7k | 230 |
| New York | DNY(1)-(2) | ~10k–16k | 320 |

---

### **实验设置**
- **仿真时长**：3600 秒
- **相位持续时间**：固定为 5 秒（黄灯占 2 秒）
- **训练方式**：同质策略共享（homogeneous policy sharing）
- **硬件环境**：Ubuntu + RTX 3060 GPU
- **超参数**：
  - 学习率：Actor 0.0003，Critic 0.0005
  - 批大小：720
  - 折扣因子 γ = 0.98，GAE λ = 0.98
  - PPO 更新轮次：6 epochs

---

### **评估指标**
- 主要指标：**平均旅行时间（Average Travel Time, ↓越好）**
- 辅助指标：
  - 队列长度均值与标准差
  - 车辆速度
  - 协调一致性（各路口间旅行时间方差）

---

### **基线方法对比**
分为两类：

#### **传统方法**
- **FixedTime**：固定周期配时
- **MaxPressure (MP)**：基于上下游压力差选择相位
- **Advanced-MP**：考虑有效范围内的移动车辆

#### **MARL 方法**
- **CoLight**, **Advanced-CoLight**：基于GAT的图注意力方法
- **MPLight**, **Advanced-MPLight**：基于压力的状态与奖励设计
- **DenseLight***：利用密集反馈机制
- **SocialLight**：分布式合作学习，衡量个体贡献

> 注：部分结果来自原论文（标 *）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table II）**

| 方法 | Jinan DJN(1) | Hangzhou PHZ(2) | NYC DNY(2) |
|------|--------------|------------------|------------|
| Fixed-Time | 346.36 s | 359.44 s | 1660.29 s |
| MaxPressure | 273.96 s | 348.98 s | 1535.77 s |
| Advanced-MP | 253.61 s | 318.67 s | — |
| CoLight | 276.33 s | 297.26 s | 1476.18 s |
| Advanced-CoLight | 253.95 s | 308.62 s | 1025.47 s |
| SocialLight | 217.92 s | 288.55 s | 1106.69 s |
| **CoordLight (Ours)** | **199.24 s** | **250.87 s** | **1039.15 s** |

> ✅ **全部7个测试场景中，CoordLight 均取得最优性能**

---

### **与最佳基线 SocialLight 的对比提升**
- **Jinan**：提升 **6.39% ~ 9.23%**
- **Hangzhou**：在高负载下提升 **7.87%**
- **New York (196 intersections)**：
  - DNY(1)：优于 SocialLight 4.3%
  - DNY(2)：优于 SocialLight 6.1%

> 📌 **尤其在大规模、高流量场景下优势显著，说明其良好的可扩展性**

---

### **消融实验结果（Ablation Studies）**

#### **(1) QDSE 状态表示的有效性（vs 其他 state encoding）**
比较五种状态定义：
- Vehicle Counts (VC)
- General Pressure (GP)
- Efficient Pressure (EP)
- Advanced Traffic State (ATS)
- Discrete Traffic State Encoding (DTSE, 图像式)

✅ **结果**：
- QDSE 显著降低平均旅行时间和队列长度
- 性能接近甚至略优于复杂的 DTSE，但计算开销更低
- 表明 QDSE 成功捕捉关键动态特征，在**复杂性与准确性之间取得良好平衡**

#### **(2) 各组件作用分析（CoordLight 变体）**
| 变体 | 描述 | 影响 |
|------|------|------|
| w/o QDSE | 替换为 VC 状态 | 性能明显下降，验证 QDSE 必要性 |
| w/o STN | 移除时空注意力网络 | 无法捕获时空依赖，收敛至次优解 |
| w/o AD | 移除状态-动作解码器 | 缺乏动作历史建模，训练不稳定 |
| Base (IPPO) | 仅全连接层 + IPPO | 所有指标最差 |

✅ **结论**：NAPO 中的注意力结构和状态-动作建模对性能至关重要。

#### **(3) 对传感器噪声的鲁棒性测试**
- 在 QDSE 输入中添加高斯噪声（σ = 10m, 20m, 30m）
- 结果显示：即使在最大噪声下，平均旅行时间仅增加 **约2.34%**
- 表明 QDSE 对现实世界中的摄像头定位误差具备较强鲁棒性

---

## **4. 关键结论和发现**

### **主要发现**
1. **QDSE 是一种高效且具预测性的状态表示**，能够帮助 RL agent 更好地理解并预判交通动态，从而做出更主动的控制决策。
2. **NAPO 实现了高效的去中心化协调**，通过注意力机制识别关键邻居，并结合状态-动作依赖建模，提升了策略学习的稳定性和协调效率。
3. **CoordLight 在从小规模到超大规模（196个路口）的真实路网中均表现出色**，尤其在高流量条件下仍保持稳定，展现出强大的可扩展性。
4. **所提方法不依赖全局信息或通信机制**，更适合在现实中部署于分布式控制系统。

---

### **方法的局限性**
- 当前研究基于**同质化路口布局**（规则四路交叉口），未考虑异构结构（T型、环岛等）。
- 相位切换逻辑简化（固定5秒），未涉及动态相位时长调整。
- 仿真实验假设理想检测设备，虽做了噪声测试，但尚未集成真实摄像头误检、遮挡等问题。
- 尚未处理紧急车辆优先、事故响应等特殊场景。

---

### **未来工作方向**
1. **拓展至异构网络**：支持多样化路口结构与非对称车道配置。
2. **联合优化相位序列与时长**：从“选相位”升级为“定周期+选相位”。
3. **增强现实适应性**：
   - 处理不完整或延迟的观测数据
   - 支持在线迁移学习以应对突发交通事件
4. **引入优先通行机制**：支持公交优先、应急车辆绿波通行。
5. **探索与V2X系统的融合**：结合车载感知信息进一步提升状态估计精度。

---

> 🔚 **总结**：  
> **CoordLight** 提出了一套完整的去中心化 ATSC 框架，通过 **QDSE + NAPO** 的双重创新，在状态建模与多智能体协调方面实现了突破。实验证明其在多个真实城市路网上显著优于现有 SOTA 方法，为大规模智能交通系统提供了兼具性能、稳定性与实用性的解决方案。

</details>

---

### 7. [Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation](https://arxiv.org/abs/2603.23838)

**Authors**: Han Zheng, Yining Ma, Brandon Araki, Jingkai Chen, Cathy Wu  
**Category**: cs.AI  
**Published**: 2026-03-27  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.23838v1  

#### Abstract
Lifelong Multi-Agent Path Finding (MAPF) is critical for modern warehouse automation, which requires multiple robots to continuously navigate conflict-free paths to optimize the overall system throughput. However, the complexity of warehouse environments and the long-term dynamics of lifelong MAPF o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**Lifelong Multi-Agent Path Finding (MAPF)** 在现代仓库自动化中的应用挑战展开研究。传统 one-shot MAPF 假设任务静态且一次性完成，而现实中的仓储系统要求机器人持续执行动态分配的任务，导致以下长期规划难题：
- 代理（agents）不断进入和离开系统，需持续协调；
- 拥堵模式随时间演化，需要前瞻性决策；
- 局部最优可能导致级联低效甚至死锁；
- 需在有限时间内找到高质量路径。

现有方法如 **CBS**, **PBS**, **PIBT** 等虽有良好理论性质或实时性，但在高密度、复杂布局下难以兼顾效率与全局优化能力。机器学习方法虽被探索，但其相对于搜索方法的优势尚不明确。

---

### 提出的新方法：RL-RH-PP
作者提出 **Reinforcement Learning-guided Rolling Horizon Prioritized Planning (RL-RH-PP)**，是首个将强化学习（RL）与基于搜索的规划器结合用于 lifelong MAPF 的框架。

#### 核心思想
- 将 **Prioritized Planning (PP)** 作为基础规划骨架，因其简单、高效、可扩展性强。
- 引入 **Rolling Horizon Prioritized Planning (RH-PP)** 框架，实现周期性重规划以适应动态任务流。
- 利用 **Reinforcement Learning** 动态生成优先级顺序（priority order），将优先级分配建模为一个 **Partially Observable Markov Decision Process (POMDP)**。
- 使用 **Transformer-style 神经网络** 编码多智能体路径信息，并通过自回归解码生成总优先级序列。

#### 技术亮点
- **学习引导的搜索空间缩减**：RL 政策聚焦于采样高质量的 priority order，显著缩小 RH-PP 的搜索空间。
- **时空注意力机制**：编码器采用交替的 temporal 和 spatial attention，分别捕捉单个 agent 路径的时间依赖性和 agent 间的空间交互。
- **端到端训练闭环**：RL 政策与 RH-PP 规划器协同训练，奖励函数设计鼓励减少拥堵、提高吞吐量。

---

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|--------|
| **性能** | 显著优于各类 baselines，在高密度场景下提升平均 **25% 吞吐量**。 |
| **泛化能力** | 展现出强大的 zero-shot 泛化能力，能跨 agent 密度、规划窗口大小和未见地图布局有效迁移。 |
| **实用性** | 保持与非学习方法相当的推理时间，适合实际部署；GPU 推理 + CPU 规划架构平衡效率与质量。 |
| **可解释性** | 可视化分析表明 RL 学会主动优先处理拥堵区域 agent 并策略性“后退”以疏通交通。 |

---

## 2. 核心实验方法和设置

### 数据集与仿真环境
构建了两个受真实启发的大规模 warehouse simulation 环境：
1. **Amazon Fulfillment Center Dense Map**
   - 基于公开基准修改，障碍物密度为 **15.3%**
   - 多条平行通道，窄走廊
2. **Symbotic Warehouse Map**
   - 首次引入该类布局至 lifelong MAPF 研究
   - 高障碍密度 **56.6%**，存在瓶颈（bottlenecks）、交叉口等典型拥堵点
   - 分为 inbound、outbound、deck、aisles 四个功能区

> 所有实验均在上述两种地图上进行，任务随机生成并动态分配。

---

### 实验设置
- **模拟时长**：`T = 800` 时间步
- **规划窗口**（planning horizon）：默认 `w = 20`
- **执行窗口**（execution horizon）：`h = 5`
- **agent 数量**：`N ∈ {80, 100, 120}`，形成高密度测试场景
- **训练平台**：单块 NVIDIA RTX 6000 Ada GPU，CPU 多进程并行 rollout
- **RL 算法**：使用 **Proximal Policy Optimization (PPO)** 进行训练

---

### 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput per Agent (TPA)** | 单个 agent 在整个仿真周期内成功完成的任务数 |
| **Total Throughput** | 所有 agent 成功完成任务总数，即 `TPA × N` |
| **Solve Time** | 每次规划步骤的平均耗时（仅 CPU wall time 或含 GPU 推理时间） |

---

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **RH-CBS** | 搜索-based | 最优但计算开销大，难扩展 |
| **RH-PBS** | 搜索-based | 使用部分优先级，效率高于 CBS |
| **PIBT** | 分布式贪心 | 实时性强，但局部决策易引发拥堵 |
| **WPPL** | 混合方法 | 2023 League of Robot Runner 冠军方案，结合 PIBT 与 LNS 优化 |
| **RH-PP (Random)** | 非学习基线 | 使用随机 priority order 的滚动地平线 PP |
| **DQ-RH-PP** | 规则启发式 | 基于最短路径长度排序（distance-query heuristic） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **Symbotic 地图** 上，`N=120` 时：
  - RL-RH-PP 达到 **TPA ≈ 11.31**
  - 比 RH-PP 提升约 **90%**
  - 比 WPPL 提升约 **12.5%**
- 在 **Amazon 地图** 上，`N=120` 时：
  - RL-RH-PP 达到 **TPA ≈ 25.56**
  - 比 RH-PP 提升约 **17.6%**
  - 比 WPPL 提升约 **8.4%**

> 总体平均提升达 **25% 吞吐量**，尤其在高密度、高障碍环境下优势更明显。

---

### 与基线方法的对比结果
| 方法 | Amazon (N=120) TPA | Symbotic (N=120) TPA |
|------|---------------------|-----------------------|
| RH-CBS | 2.84 ± 0.29 | 1.50 ± 0.45 |
| RH-PBS | 3.37 ± 0.26 | 1.76 ± 1.10 |
| PIBT | 16.09 ± 0.48 | 2.67 ± 0.49 |
| WPPL | 23.59 ± 0.26 | 10.05 ± 1.33 |
| RH-PP (K=5) | 21.74 ± 2.25 | 8.21 ± 2.24 |
| **RL-RH-PP (ours)** | **25.56 ± 0.55** | **11.31 ± 2.21** |

✅ **RL-RH-PP 在所有设置下均取得最高或接近最高的 TPA**  
✅ 在 Symbotic 地图上远超其他方法，体现对复杂约束环境的强大适应力  
✅ 推理时间与其他方法在同一量级（约 1 秒以内）

---

### 消融实验结果

#### （1）奖励函数权重影响（ablation on K 和 σ）
- 设置 `K=1000`, `σ=1000` 时达到最佳性能
- 若 `K=0`（无拥塞惩罚），收敛慢且最终性能下降
- 若 `σ=0`（无不可行性惩罚），初始阶段失败率高，学习不稳定
> ✅ 表明两项惩罚项对稳定训练和提升吞吐至关重要

#### （2）编码器结构消融
| 编码器变体 | Amazon TPA | Symbotic TPA |
|-----------|------------|---------------|
| Full Model (ours) | 31.80 | 18.38 |
| w/o Temporal Attention | ↓ 明显下降 | ↓ 显著下降 |
| w/o Spatial Attention | ↓ 小幅下降 | ↓ 中等下降 |
| 替换为 Yan & Wu (2024) 架构 | ↓ 下降 | ❌ 几乎无法学习 |

> ✅ 证明 **temporal attention 对缓解 aisle 拥堵尤为关键**  
> ✅ **spatial attention 实现全局 agent 交互建模，优于局部卷积**

#### （3）不同 priority assignment 策略比较
| 方法 | Amazon TPA (N=120) | Symbotic TPA (N=120) |
|------|--------------------|-----------------------|
| DQ-RH-PP (距离启发式) | 17.66 | 9.88 |
| **RL-RH-PP (ours)** | **25.56** | **11.31** |

> ✅ 固定规则启发式无法应对动态拥堵，RL 学习到更灵活、前瞻性的优先级策略

#### （4）Contextual Bandit vs RL
- 将 RL 替换为一步决策（contextual bandit）
- 结果：初期收敛快，但最终性能低于完整 RL 框架
> ✅ 证明 **long-horizon planning 是提升性能的关键**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **RL 可有效学习高质量的全局优先级策略**，显著超越随机和固定规则方法。
2. ✅ **RL-RH-PP 能主动识别并缓解拥堵**：通过 heatmap 分析发现，RL 会优先调度处于拥堵区域的 agent。
3. ✅ **具备强零样本泛化能力**：训练于特定配置（如 `N=120`, `w=20`）的模型可在不同 agent 数量、规划窗口、甚至未见地图布局上直接迁移使用。
4. ✅ **能从次优状态中恢复**：当切换至 RL-RH-PP 时，即使系统已由 RH-PP 引发严重拥堵，也能快速恢复通行效率。
5. ✅ **学习与搜索的协同优于纯学习或纯搜索方法**：RL 提供战略指导，PP 提供高效执行，形成“best of both worlds”。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **绝对位置嵌入限制跨尺寸迁移** | 当前 encoder 使用基于坐标的 learnable embedding，无法直接迁移到不同尺寸的地图。 |
| **Top-K 评估串行执行影响效率** | 当前 `K` 增大会线性增加 CPU 时间，缺乏并行化支持。 |
| **未联合优化任务分配** | 当前框架假设任务已分配，未来可扩展至 joint task assignment + path planning。 |
| **未考虑不确定性** | 如延迟、传感器噪声等现实扰动尚未建模。 |

---

### 未来工作方向
1. **全地图无关表示**（map-agnostic representation）：开发不依赖绝对坐标的 state encoder，支持任意大小和拓扑的地图迁移。
2. **并行化 Top-K 评估**：利用多线程或分布式计算加速候选 order 的评估过程，提升大规模系统的实时性。
3. **联合任务与路径规划**：让 autoregressive decoder 输出 `(agent, task)` 序列，统一解决 task assignment 与 path planning。
4. **增强鲁棒性**：引入对模型不确定性、动态障碍、通信延迟的建模，提升现实适用性。
5. **扩展至其他领域**：将 RL-guided optimization 框架推广至机场物流、自动驾驶车队协调等 long-horizon 多智能体决策问题。

---

> 🔗 **开源声明**：作者承诺将 RL-RH-PP 框架与训练流程开源，项目地址为：[https://github.com/MikeZheng777/RL-RH-PP](https://github.com/MikeZheng777/RL-RH-PP)

</details>

---

### 8. [Decentralized Task Scheduling in Distributed Systems: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/2603.24738)

**Authors**: Daniel Benniah John  
**Category**: cs.DC  
**Published**: 2026-03-27  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.24738v1  

#### Abstract
Efficient task scheduling in large-scale distributed systems presents significant challenges due to dynamic workloads, heterogeneous resources, and competing quality-of-service requirements. Traditional centralized approaches face scalability limitations and single points of failure, while classical...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Decentralized Task Scheduling in Distributed Systems: A Deep Reinforcement Learning Approach*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大规模分布式系统（如云-边协同计算、IoT）面临**动态负载、资源异构性和多样化服务质量（QoS）需求**带来的任务调度挑战。传统集中式调度存在**可扩展性差、单点故障风险高**等问题；而经典启发式算法（如 FCFS、SJF）缺乏对动态环境的适应能力。此外，现有的基于 DRL 的调度方法多依赖**中心化控制架构**和**重型深度学习框架**（如 TensorFlow/PyTorch），难以在资源受限的边缘设备上部署。

### 提出的新方法与创新点
本文提出了一种**去中心化的多智能体深度强化学习框架（DRL-MADRL）**，用于异构分布式系统的任务调度，其核心创新包括：

- ✅ **Dec-POMDP 建模**：将任务调度建模为 **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**，每个计算节点作为独立 Agent，仅基于局部观测进行决策，无需全局状态同步或中央协调器，提升了系统的**可扩展性、容错性与鲁棒性**。
  
- ✅ **轻量级神经网络架构**：设计了一个仅使用 **NumPy** 实现的轻量级 Actor-Critic 架构，摒弃了 RNN、Attention 等复杂组件，采用简单的前馈网络（ReLU 激活）。每 Agent 内存占用约 **100 KB**，推理延迟低于 **10 ms**，可在无 GPU 的边缘设备运行。

- ✅ **优先级感知的动作选择机制**：结合 Google Cluster Trace 中的任务优先级分类（Production/Batch/Best-effort），引入显式的优先级评分函数，确保高优先级任务获得及时调度，同时兼顾系统整体效率。

- ✅ **精细化的能量消耗模型**：提出了一个准确反映异构节点能耗特性的线性功率模型：
  $$
  P_i(t) = P_{idle,i} + P_{dyn,i} \times u_i(t)
  $$
  并通过积分计算总能耗，避免因低吞吐率导致的“虚假节能”误解。

- ✅ **完整的开源实现与可复现性保障**：提供完整源码、实验脚本与数据集（MIT 协议），任何研究者均可在普通笔记本电脑上 **4 分钟内复现实验结果**。

### 相比现有方法的优势
| 维度 | 传统方法（FCFS/SJF） | 中心化 DRL 方法 | 本文 DRL-MADRL |
|------|------------------------|------------------|----------------|
| 可扩展性 | 差（集中式瓶颈） | 差（需全局状态） | ✅ 高（完全去中心化） |
| 容错性 | 差（单点故障） | 差 | ✅ 高 |
| 自适应能力 | 无 | 强（但集中训练） | ✅ 强（分布式学习） |
| 资源开销 | 低 | 高（GPU + 大内存） | ✅ 极低（仅 NumPy） |
| 部署可行性 | 高 | 低（难于边缘端） | ✅ 高（适用于 IoT/边缘网关） |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **Google Cluster Trace v3** 数据集中的统计特性生成合成任务负载，以保证真实性和可比性。
- 关键分布建模如下：
  - **执行时间**：服从 Pareto 分布（α=1.5, t_min=5s），体现重尾特征。
  - **CPU/内存需求**：分别服从 LogNormal(μ=0.5, σ=0.8) 和 LogNormal(μ=2.0, σ=1.0)。
  - **到达过程**：泊松过程（λ=0.5 tasks/s）。
  - **任务优先级**：Production (25%)、Batch (60%)、Best-effort (15%)。
  - **截止时间**：按优先级设定倍数（Production: 1.5×, Batch: 3.0×, Best-effort: 5.0×）。

### 实验设置
- **系统规模**：100 个异构节点，分为三类：
  - High-capacity（20%）：24–32 cores, 96–128 GB RAM
  - Medium-capacity（50%）：8–16 cores, 32–64 GB RAM
  - Low-capacity（30%）：2–8 cores, 8–32 GB RAM
- **任务数量**：每轮 Episode 处理 1,000 个任务。
- **训练与评估**：共运行 30 轮独立实验，取最后 10 轮平均值作为最终性能指标。
- **硬件平台**：在普通笔记本（Intel i5-8265U, 8GB RAM, 无 GPU）上完成全部实验。

### 评估指标
| 指标 | 描述 |
|------|------|
| **ATCT** | Average Task Completion Time（平均任务完成时间） |
| **E_total** | 总能量消耗（kWh） |
| **SLA Satisfaction Rate** | 在截止时间前完成的任务占比 |
| **Throughput** | 单位时间内完成的任务数（tasks/1000s） |
| **Load Balance** | 利用率方差的倒数，衡量负载均衡程度 |
| **p-value** | 使用两样本 t-test 进行显著性检验（α=0.05） |

### 基线方法对比
1. **Random**：从可行节点中均匀随机选择。
2. **Weighted Round-Robin (W-RR)**：按节点容量比例循环分配。
3. **Priority-aware Min-Min**：优先处理高优先级任务，并分配给最早能完成的可用节点。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（DRL-MADRL）
| 指标 | 数值 |
|------|------|
| **Average Task Completion Time** | **30.8 秒** |
| **Total Energy Consumption** | **745.2 kWh** |
| **SLA Satisfaction Rate** | **82.3%** |
| **Throughput** | **425.15 tasks / 1000s** |
| **Completed Tasks** | 999 / 1000 |
| **Decision Latency** | **8.2 ms** |
| **Memory per Agent** | ~97 KB |

### 与基线方法的对比结果
| 对比项 | vs Random | vs W-RR | vs Priority-MinMin |
|-------|-----------|---------|--------------------|
| **ATCT 改进** | ↓15.6% (36.5s → 30.8s) | ↓14.9% (36.2s → 30.8s) | 显著更优 |
| **Energy Reduction** | ↓15.2% (878.3kWh → 745.2kWh) | ↓26.0% (1007.1kWh → 745.2kWh) | 更高效且完成更多任务 |
| **SLA Satisfaction ↑** | +6.8 pp (75.5% → 82.3%) | +6.2 pp (76.1% → 82.3%) | +35.0 pp (47.3% → 82.3%) |
| **Throughput ↑** | +4.4% | +5.0% | +303% |
| **p-value** | < 0.001 (**高度显著**) | < 0.001 | < 0.001 |

> ⚠️ 特别说明：Priority-MinMin 的总能耗仅为 155.3 kWh，看似最低，实则因其仅完成了 **280 个任务（28% 完成率）**，大量任务未被调度。若归一化到每任务能耗，其实际为 **0.554 kWh/task**，高于 DRL-MADRL 的 **0.746 kWh/task**，表明其“节能”是假象。

### 消融实验结果（Ablation Study）
虽然文中未列出详细表格，但在讨论部分明确指出以下模块的关键作用：
- **优先级感知机制**：对 SLA 满足率提升至关重要，移除后 SLA 下降超 20 个百分点。
- **自适应奖励塑形（adaptive reward shaping）**：平衡多个目标（completion time, energy, SLA, balance），显著加快收敛速度。
- **优先经验回放（prioritized experience replay）**：聚焦高 TD-error 转移，使学习效率提升约 40%，在第 20 轮即趋于稳定。
- **Dec-POMDP 设计**：验证了即使没有全局信息，Agent 仍可通过局部交互隐式协作达成系统级优化。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **去中心化 MADRL 可有效解决大规模异构系统调度问题**，在保持低通信开销的同时实现高性能。
2. ✅ **轻量化 DRL 是可行的**：仅用 NumPy 实现的简单 FFN 就能达到甚至超越传统复杂模型的效果，打破了“必须用 PyTorch/TensorFlow”的固有认知。
3. ✅ **多目标联合优化优于单一目标优化**：单独最小化能耗可能导致拒绝调度（如 Priority-MinMin），必须综合考虑 completion time、SLA、throughput 和 energy。
4. ✅ **优先级机制显著提升 QoS**：针对 Production 类任务的紧迫性设计调度策略，是提高 SLA 满足率的关键。
5. ✅ **仿真结果具有强统计显著性**：所有改进均通过 p < 0.001 的假设检验，Type I 错误概率低于 0.1%。

### 方法的局限性
1. **基于仿真而非真实部署**：实验在离散事件模拟器中进行，未考虑网络拥塞、Byzantine 故障等现实因素。
2. **任务间无依赖关系**：假设所有任务相互独立，无法直接应用于 DAG 结构的任务图调度场景。
3. **规模仍有限**：100 节点系统虽具代表性，但远小于超大规模数据中心（>10,000 节点），扩展性有待进一步验证。
4. **静态拓扑假设**：未考虑节点动态加入/退出、链路变化等动态网络行为。

### 未来工作方向
- 🔄 **真实环境部署验证**：在边缘计算测试床（如 Raspberry Pi 集群）上部署并评估实际性能。
- 🔺 **支持任务图调度（Task Graph Scheduling）**：扩展框架以处理带前驱约束和数据依赖的应用。
- 🏗️ **构建分层协调机制（Hierarchical Coordination）**：应对万级节点的超大规模系统，实现区域自治 + 全局协调。
- 🤝 **集成联邦学习（Federated Learning）**：允许多个集群共享调度策略而不泄露本地数据，增强隐私保护。
- 📈 **引入工作负载预测**：利用时间序列模型预测未来负载趋势，实现前瞻性资源预配置与调度。

---

> 🔗 **代码与数据公开地址**：[https://github.com/danielbenniah/marl-distributed-scheduling](https://github.com/danielbenniah/marl-distributed-scheduling)  
> 所有实验均可在普通笔记本上 **4 分钟内复现**，极大促进了后续研究的可重复性与透明度。

</details>

---

### 9. [On Gossip Algorithms for Machine Learning with Pairwise Objectives](https://arxiv.org/abs/2603.24128)

**Authors**: Igor Colin (LTCI, S2A, IP Paris), Aur\'elien Bellet (PREMEDICAL), Stephan Cl\'emen\c{c}on (LTCI, IDS, S2A, IP Paris), Joseph Salmon (IROKO, UM)  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.24128v1  

#### Abstract
In the IoT era, information is more and more frequently picked up by connected smart sensors with increasing, though limited, storage, communication and computation abilities. Whether due to privacy constraints or to the structure of the distributed system, the development of statistical learning me...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：On Gossip Algorithms for Machine Learning with Pairwise Objectives

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于**分布式机器学习中的成对目标函数（pairwise objectives）优化问题**，即当目标函数是数据点对之间的统计量（如 U-statistic of degree two）时，如何在去中心化网络中高效地进行估计与优化。

这类问题广泛存在于许多重要任务中，例如：
- **Ranking**（排序）：最大化 AUC（Area Under the ROC Curve）
- **Metric/Similarity Learning**（度量/相似性学习）
- **Clustering**（聚类）
- **Graph Reconstruction**（图重建）

传统 gossip 算法主要针对可分解为单个样本加和的目标函数（如均值），而无法直接处理依赖于所有样本对组合的成对目标。因此，**缺乏理论完备且高效的去中心化算法来处理此类非局部交互问题**。

---

### 提出了什么新方法或新思路

作者提出并系统分析了适用于成对目标的 gossip 算法框架，主要包括两个核心部分：

#### （1）**GoSTA 算法（Gossip-based Stochastic Averaging）**
- 用于**去中心化估计 U-statistic**（如 $\frac{2}{n(n-1)}\sum_{i<j} h(x_i, x_j)$）。
- 结合了**辅助观测传播（auxiliary observation propagation）** 和 **标准 gossip 平均机制**。
- 每个节点维护一个“辅助观测” $y_k$，通过与其他节点交换 $y_k$ 来逐步获取全局配对信息。

#### （2）**Gossip Dual Averaging for Pairwise Functions**
- 将 Duchi et al. (2012a) 的分布式对偶平均方法扩展到成对目标。
- 在每次迭代中，每个节点基于其本地数据与其当前拥有的辅助观测计算偏梯度（biased subgradient estimate）。
- 利用**ergodic dual averaging 分析框架**证明梯度偏差会随图混合过程指数衰减。

---

### 相比现有方法的优势

| 方面 | 本文贡献 vs. 现有工作 |
|------|------------------------|
| **理论完整性** | 首次提供了 GoSTA 算法的完整非渐近收敛保证，包括期望误差和方差上界；此前仅分析了偏差（Colin et al., 2015）。 |
| **优化收敛性证明** | 明确证明了成对优化中的梯度偏差项最终消失，解决了 Colin et al. (2016) 中遗留的收敛性疑问。 |
| **下界分析** | 推导了首个针对成对目标的去中心化优化下界，揭示了图结构中“平均两跳距离”（averaged two-hop distance）的关键作用。 |
| **通用性** | 统一处理同步与异步设置，并支持多种图拓扑（complete, grid, Watts-Strogatz）。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Breast Cancer Wisconsin (Original)**  
  - 样本数 $n = 699$
  - 特征维度 $d = 11$
  - 每个节点存储一个样本（single point per node）

### 实验设置
- **任务**：最大化 AUC（ROC 曲线下面积），使用 logistic pairwise loss 作为代理损失：
  $$
  R_n(\theta) = \frac{1}{n^2}\sum_{i,j} \ell(l_i > l_j)\log(1+\exp((x_j - x_i)^T\theta))
  $$
- **网络拓扑结构对比三种图**：
  1. **Complete Graph**：全连接图，理想通信条件
  2. **2D Grid**：四邻域网格，低连通性，大直径
  3. **Watts-Strogatz Graph**：小世界网络（$k=5$, $p=0.3$），介于规则与随机之间

- **算法实现**：
  - 同步与异步版本的 Gossip Dual Averaging（Algorithm 2 & 7）
  - 步长策略：$\gamma(t) = 10^{-3}/\sqrt{t}$

- **评估指标**：
  1. **目标函数值演化**（loss evolution）
  2. **共识损失**（consensus loss）：衡量各节点参数一致性
  3. **偏差项（bias term）**：$\|\mathbb{E}[g(t)] - \nabla F(\theta(t))\|$，反映梯度估计质量

- **重复次数**：50 次独立运行取均值 ± 标准差

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

| 图类型 | 收敛速度 | 最终目标值 | 偏差衰减速率 |
|--------|----------|------------|--------------|
| Complete Graph | ⭐ 最快 | 最优 | 极快趋零 |
| Watts-Strogatz | 中等 | 接近最优 | 快速衰减 |
| 2D Grid | ❌ 最慢 | 较高残差 | 衰减缓慢但仍趋于零 |

#### 图表观察（Figure 1）：
- **(a) Loss Evolution**：
  - 完整图和 Watts-Strogatz 表现显著优于 Grid。
  - 所有设置下目标函数单调下降，表明算法稳定收敛。
- **(b) Bias Term**：
  - 所有网络中 bias term 迅速下降至接近零（远小于目标函数值）。
  - 实证验证了理论预测：**即使初始梯度估计有偏，其影响在有限时间内可忽略**。

### 消融实验结果（隐含分析）
虽然未明确列出消融实验表格，但以下分析体现了关键变量的影响：

| 因素 | 影响分析 |
|------|---------|
| **图连通性（spectral gap $\lambda_{n-1}/|E|$）** | 谱隙越大（如 complete graph），收敛越快，与理论一致。 |
| **通信成本** | Grid 需更多轮次达到相同精度，体现拓扑瓶颈。 |
| **偏差动态** | bias 在前几百轮内快速衰减，说明辅助观测已充分混合。 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **GoSTA 算法具有 $O(1/t)$ 的期望收敛速率**，方差控制良好，首次给出完整非渐近分析。
2. ✅ **成对优化中的梯度偏差确实会随图混合过程指数衰减**，无需额外假设即可保证收敛。
3. ✅ **网络拓扑的谱性质（spectral gap）直接影响收敛效率**：连通性越好，收敛越快。
4. ✅ **提出的 lower bound 揭示了成对问题的本质复杂度**：不仅取决于图直径，还涉及“平均两跳路径长度”（△），反映了信息需经中间节点传递的特点。
5. ✅ **数值实验验证了理论预测**：bias 快速消失，算法可靠收敛，尤其在高连通图中表现优异。

---

### 方法的局限性
1. **每节点仅存一个样本**：假设较理想化，实际场景常为多点分布。尽管第6节讨论了扩展方案，但仍需进一步验证。
2. **凸性要求**：理论分析依赖目标函数的凸性和 Lipschitz 连续性，在非凸任务中可能不适用。
3. **通信开销**：频繁交换辅助观测可能导致带宽压力，尤其在大规模稀疏图中。
4. **异步设置收敛更慢**：由于时间估计器引入额外方差，整体收敛速率低于同步情形。

---

### 未来工作方向
1. **Multiple Points per Node**：将方法推广至每个节点持有多个样本的情形，结合虚拟节点建模。
2. **Differential Privacy Integration**：结合本地差分隐私（Local DP）机制，在保护原始数据的同时完成成对统计量估计。
3. **Robustness and Fairness**：将鲁棒学习与公平性约束（本身可表达为 U-statistics）融入该框架，构建负责任的去中心化学习体系。
4. **非凸与深度学习应用**：探索在联邦学习或多智能体强化学习中使用 gossip-based pairwise 更新的可能性。
5. **压缩与量化技术**：引入梯度压缩（gradient compression）、量化（quantization）以降低通信负担。

---

> **总结一句话**：  
> 本文填补了 gossip 算法在成对目标学习上的理论空白，提出了兼具实践有效性与理论严谨性的解决方案，为去中心化环境下处理 ranking、clustering、metric learning 等任务奠定了坚实基础。

</details>

---

### 10. [DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving](https://arxiv.org/abs/2603.24587)

**Authors**: Pengxuan Yang, Yupeng Zheng, Deheng Qian, Zebin Xing, Qichao Zhang, Linbo Wang, Yichen Zhang, Shaoyu Guo, Zhongpu Xia, Qiang Chen, Junyu Han, Lingyun Xu, Yifeng Pan, Dongbin Zhao  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.24587v1  

#### Abstract
We introduce DreamerAD, the first latent world model framework that enables efficient reinforcement learning for autonomous driving by compressing diffusion sampling from 100 steps to 1 - achieving 80x speedup while maintaining visual interpretability. Training RL policies on real-world driving data...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **diffusion-based world model** 的自动驾驶模拟器虽然能实现安全的“想象训练”（imagination-based training），但在实际用于 **Reinforcement Learning (RL)** 时面临两大瓶颈：
- **高推理延迟**：标准扩散模型需要 100 步采样，导致单帧生成耗时约 2 秒，无法支持高频 RL 交互。
- **像素级目标不适用于驾驶任务**：过度关注视觉保真度，而忽视对空间结构、动态演化等驾驶安全至关重要的语义理解。

此外，传统 RL 中的探索机制（如随机高斯采样）容易产生物理上不合理或动态不连续的轨迹，引发 world model 的幻觉（hallucination）。

### 提出了什么新方法或新思路
本文提出 **DreamerAD**，是首个在 **latent space** 内完成高效强化学习的自动驾驶框架，其三大核心创新为：

#### （1）Shortcut Forcing World Model (SF-WM)
- 利用 **shortcut forcing** 技术将扩散采样从 100 步压缩至 **1–4 步**，实现 **80× 推理加速**。
- 在 rectified flow 框架下引入多分辨率步长调度机制，通过 teacher-student 蒸馏保留预测质量，在一步推理下仍保持清晰的场景重建能力。

#### （2）Autoregressive Dense Reward Model (AD-RM)
- 直接在 **latent feature** 上构建奖励模型，无需显式标注。
- 引入跨时间步的密集奖励信号（dense temporal rewards），覆盖多个预测时域（0–4s，每 0.5s 一个阶段），实现细粒度信用分配（credit assignment）。
- 奖励维度包括 NC、DAC、TTC 等共 8 项，并采用对数融合机制优先保障安全性。

#### （3）Gaussian Vocabulary Sampling for GRPO
- 构建高质量轨迹词典（vocabulary of trajectories），仅保留与人类驾驶接近的安全候选路径。
- 使用以当前策略输出为中心的 **Gaussian 分布进行邻域采样**，确保探索轨迹具有物理合理性和动态平滑性。
- 避免了传统随机探索带来的不稳定性与幻觉风险。

### 相比现有方法的优势
| 维度 | DreamerAD | 现有方法（如 Epona、WorldRFT） |
|------|-----------|-------------------------------|
| 推理速度 | 单步采样，延迟低至 **0.03s/frame** | 多步扩散，延迟高达 2s/frame |
| RL 效率 | 支持高频交互训练 | 因延迟难以集成到 RL loop |
| 可解释性 | latent 可无损解码为 RGB 视频用于事故分析 | 像素级生成易失真 |
| 探索质量 | 基于 vocabulary 的高斯采样保证轨迹合理性 | 随机采样易导致动态断裂或危险行为 |
| 奖励建模 | latent-level autoregressive 密集奖励，泛化强 | 依赖全轨迹打分或外部标注 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **NavSim Dataset**：基于 nuPlan 构建的大规模自动驾驶仿真数据集。
  - 包含 8 个环视摄像头图像（surround-view images）
  - 高质量 LiDAR 点云
  - 共计 1,192 个训练场景 + 136 个测试场景
  - 排除了静态和匀速场景，聚焦于复杂、高挑战性驾驶情境

### 实验设置
- **基础模型**：以 **Epona**（基于 flow matching 的 autoregressive diffusion model）作为 backbone world model
- **输入**：历史观测 $O \in \mathbb{R}^{B×P×H×W×3}$ 和动作序列 $A$
- **训练平台**：32 × NVIDIA H20 GPU
- **关键参数**：
  - 图像尺寸：512×1024
  - VisDiT 采样步数：1；TrajDiT 采样步数：20
  - 批大小与学习率随模块调整（见原文 Section 4.2）

### 评估指标
#### 主要指标：
- **EPDMS (Extended Predictive Driver Model Score)**：NavSim v2 的综合评价指标，包含以下子项：
  - 安全类：No Collision (NC), Drivable Area Compliance (DAC), Time-to-Collision (TTC), Traffic Light Compliance (TLC)
  - 性能类：Lane Keeping (LK), History Comfort (HC), Extended Comfort (EC), Ego Progress (EP)
- **PDMS**：NavSim v1 使用的传统版本，涵盖 NC、DAC、TTC、Comfort、EP

#### 辅助分析工具：
- PCA 可视化 latent features 的空间一致性
- BEV 地图展示轨迹规划结果
- RGB 解码用于事故回放与可解释性验证

### 基线方法对比
参与比较的方法包括：
- **TransFuser**, **Hydra-MDP++, DriveSupreme**, **iPad**, **DiffusionDrive**
- **World-model-based baselines**：Epona (base), World4Drive, WorldRFT, DriveVLA-W0
- **VLA 类先进方法**：AutoVLA, RecogDrive

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | 数据集 | EPDMS / PDMS | 是否使用 latent RL |
|------|--------|---------------|---------------------|
| **DreamerAD (Ours)** | **NavSim v2** | **87.7** | ✅ |
| Epona (Base) | NavSim v2 | 85.1 | ❌ |
| DriveVLA-W0 | NavSim v2 | 86.1 | ❌ |
| **DreamerAD (Ours)** | **NavSim v1** | **88.7** | ✅ |
| Epona (Base) | NavSim v1 | 86.2 | ❌ |

> ✅ **State-of-the-art 成绩**：DreamerAD 在两个版本中均达到 SOTA。

### 与基线方法的对比结果（相对 Epona 提升）
在 **NavSim v2** 上的关键指标提升：
- **EPDMS ↑2.6 pts**
- **NC ↑0.9**（更少碰撞）
- **DAC ↑1.5**（更强可行驶区域遵守）
- **TTC ↑1.1**（更高的前向安全性）
- **LK ↑0.5, HC ↑4.6, EC ↑?**（显著舒适性改善）
- **EP ↓0.8**：主动牺牲激进性换取更高安全性（design choice）

> 💡 表明：**imagination-based trial-and-error learning 显著增强了避障鲁棒性与多维驾驶能力**

在 **NavSim v1** 上的表现：
- **PDMS 达到 88.7（+2.5 pts）**
- DAC ↑2.1, TTC ↑0.5 → 安全性全面提升
- 尽管低于 AutoVLA (89.1) 和 RecogDrive (90.8)，但后者依赖更强的 VLA encoder 与监督信号，而 DreamerAD 仅使用无监督预训练 encoder

### 消融实验结果（Ablation Studies）

#### （1）Shortcut Forcing 的影响（Table 3 & 4）
| 采样步数 | Latency/frame (s) | EPDMS |
|---------|--------------------|-------|
| 16      | 0.40               | 87.7  |
| 4       | 0.10               | 87.8  |
| **1**   | **0.03**           | **87.7** |

✅ 结论：**单步推理即可达到完整性能，实现超低延迟部署**

#### （2）Autoregressive Dense Reward Model 的有效性
- 对比 ID 3（无 AD-RM）与 ID 4（完整方法）：EPDMS 从 87.0 → 87.7
- 数据效率测试（Table 5）显示：
  - 使用 **仅 20% 训练数据** 即可达 87.5 EPDMS
  - 表明 reward model 泛化能力强，**小样本即可捕捉好坏驾驶差异**

#### （3）Vocabulary Sampling 方法对比
| 方法 | EPDMS | 动态连续性 | 物理合理性 |
|------|-------|------------|-------------|
| Ours (Gaussian vocab sampling) | **87.7** | ✅ 平滑流畅 | ✅ 高 |
| WorldRFT [31] | 86.6 | ❌ 断裂明显 | ❌ 易幻觉 |
| Flow-GRPO [23] | 87.0 | ⚠️ 锯齿状轨迹 | ⚠️ 存在扰动 |

✅ 结论：**基于 vocabulary 的高斯采样最适配 autoregressive world model，提供稳定高效的探索机制**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Latent-space RL 是可行且高效的**：首次证明可在 video generation model 的 latent space 中完成端到端 RL，兼顾效率与性能。
2. ✅ **Imagination-based training 显著提升安全性**：通过大量虚拟试错，模型学会规避潜在碰撞，获得超越模仿学习的行为理解。
3. ✅ **Step compression 不损害下游任务表现**：SF-WM 实现 80× 加速的同时未损失预测精度，打破“快则不准”的权衡。
4. ✅ **latent-level reward modeling 具备强泛化性**：AD-RM 能从小量数据中学习复杂驾驶规范，适合现实场景中标注稀缺的情况。
5. ✅ **exploration 应受物理约束引导**：Gaussian vocabulary sampling 比纯随机探索更有效，防止 world model 进入不可靠区域。

### 方法的局限性
- **依赖高质量预训练 world model**：若 SF-WM 本身存在系统偏差，则会影响整个 RL pipeline。
- **trajectory vocabulary 设计需谨慎**：过滤阈值（如 lateral offset ≤5m）可能限制极端但合法的 maneuver（例如紧急变道）。
- **尚未完全脱离 expert data**：reward labeling 仍基于 near-expert demonstrations，难以应对全新城市环境中的零样本迁移。
- **实时性仍受限于整体 pipeline**：尽管 world model 快速，但 reward model 与 policy rollout 仍有优化空间。

### 未来工作方向
- 将 DreamerAD 扩展至 **multi-agent setting**，模拟车-车交互。
- 探索 **fully offline RL setup**，进一步减少对 simulator 的依赖。
- 引入 **uncertainty-aware latent rollout**，提升对长尾事件的鲁棒性。
- 结合 **language instruction** 实现意图驱动的 trajectory planning。
- 推动该框架在真实车辆上的 **closed-loop deployment** 验证。

--- 

> 📌 **总结一句话**：  
> DreamerAD 成功实现了 **高速、安全、可解释** 的 latent-space 强化学习闭环，为大规模自动驾驶策略优化提供了全新的工程范式。

</details>

---

### 11. [MetaKube: An Experience-Aware LLM Framework for Kubernetes Failure Diagnosis](https://arxiv.org/abs/2603.23580)

**Authors**: Wei Sun, Ting Wang, Xinran Tian, Wanshun Lan, Xuhan Feng, Haoyue Li, Fangxin Wang  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.23580v1  

#### Abstract
Existing LLM-based Kubernetes diagnostic systems cannot learn from operational experience, operating on static knowledge bases without improving from past resolutions. We present MetaKube, an experience-aware LLM framework through three synergistic innovations: (1) an Episodic Pattern Memory Network...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MetaKube: An Experience-Aware LLM Framework for Kubernetes Failure Diagnosis

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Models (LLMs)** 的 Kubernetes 故障诊断系统存在三大根本缺陷：
1. **无法从运维经验中学习**：现有系统依赖静态知识库（如 RAG），不支持反馈机制，每次诊断独立进行，忽略历史解决方案中的宝贵经验。
2. **高质量诊断数据极度稀缺**：Kubernetes 故障知识分散在文档、GitHub Issue 和内部 Runbook 中，缺乏结构化、高质量的训练与检索数据。
3. **企业级部署的数据隐私要求高**：生产环境中敏感的集群日志不能发送至外部 LLM API，而本地部署的大模型（如 70B+ 参数）计算开销过大，小模型又能力不足。

---

### 🚀 提出的新方法与创新
为解决上述问题，作者提出 **MetaKube** —— 一种具备“经验感知”能力的 LLM 框架，其核心由三个协同组件构成：

#### （1）Episodic Pattern Memory Network (EPMN)
- **功能**：抽象并持续更新历史故障处理模式，形成可复用的“经验记忆”。
- **机制**：
  - 双粒度记忆架构：存储具体事件（episodic memories）和泛化模式（pattern abstractions）。
  - 置信度校准检索：结合语义相似性、时间新鲜度、历史成功率等多因素动态选择最优匹配。
  - 支持快速模式识别（intuitive pathway）与因果推理引导（analytical pathway）。

#### （2）Meta-Cognitive Controller
- **功能**：模拟人类双过程认知理论，动态决策使用直觉路径还是分析路径。
- **机制**：
  - 基于 EPMN 返回的记忆置信度 $ C(M^*) $ 进行路由：
    - 若 $ C(M^*) > t $，走轻量级 **intuitive pathway**（速度快）
    - 否则走深度 **analytical pathway**（精度高）
  - 阈值 $ t $ 可通过元学习自适应调整，实现速度与准确性的智能权衡。

#### （3）KubeLLM：领域专用的小型 LLM
- **基础模型**：基于开源的 **Qwen3-8B** 构建。
- **增强方式**：
  - 在自建的 **Kubernetes Fault Resolution Dataset (KFRD)** 上进行 **Supervised Fine-Tuning (SFT)**。
  - 使用 **LoRA** 技术微调，保持高效且避免灾难性遗忘。
- **优势**：可在本地部署，保障数据隐私，同时达到接近 GPT-4 的诊断性能。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | MetaKube |
|------|--------|----------|
| 学习能力 | 静态知识，无反馈学习 | 动态积累经验，持续优化 |
| 数据利用 | 被动检索 | 主动抽象模式，支持跨场景迁移 |
| 推理策略 | 固定流程或单一路径 | 双路径自适应切换 |
| 部署可行性 | 外部API有隐私风险 / 本地大模型成本高 | 本地部署 8B 模型，兼顾性能与安全 |
| 准确率 | 有限 | 显著提升，逼近 GPT-4 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### （1）KubeFault（主测试集）
- **来源**：从 KubeGraph 自动生成 + 人工验证。
- **规模**：共 **1,873 个真实世界故障场景**。
- **类别分布**：
  - Resource Errors (22.0%)
  - Network Errors (20.7%)
  - Scheduling Errors (15.9%)
  - Image Errors (14.7%)
  - Configuration Errors (16.8%)
  - System Errors (9.9%)
- **内容包含**：症状描述、环境上下文、日志、根因、修复步骤（含 `kubectl` 命令）。

#### （2）Kubernetes Fault Resolution Dataset (KFRD)
- **用途**：用于 KubeLLM 的 SFT 训练。
- **规模**：**7,000 条高质量样本**。
- **构建流程**：
  1. 采集 Stack Overflow / GitHub 真实问题；
  2. 结构化为 “问题-尝试-解决方案” 格式；
  3. 使用 LLM 扩充合成数据；
  4. 去重 + Chain-of-Thought 增强推理链；
  5. 划分：5,000 用于训练，2,000 用于测试。

#### （3）Telecom Dataset（外域测试集）
- 来自某电信公司的实际 Kubernetes 故障案例，用于评估泛化能力。

---

### 🧪 实验设置与评估指标

#### 评估维度（每项满分 10 分）
| 指标 | 定义 |
|------|------|
| **Effectiveness (Eff.)** | 是否正确识别根因、提供有效解决方案及预防建议 |
| **Equivalence (Equ.)** | 方法是否与专家推荐一致，逻辑是否合理 |
| **Completeness (Com.)** | 是否覆盖所有必要操作、命令和边界情况 |
| **Safety/Accuracy (S/A)** | 是否符合 Kubernetes 最佳实践，无危险操作 |
| **Average (Avg.)** | 四项平均得分 |

#### 评分方式
- **GPT-5 自动评估**：大规模自动化打分。
- **盲审专家评估**：三位具有 5 年以上经验的运维工程师独立评分，标准差 < 1.5 才有效。

#### 基线方法对比
| 类型 | 方法列表 |
|------|---------|
| Zero-shot | GPT-4.1, GPT-4.1-mini, Qwen3-8B |
| GraphRAG 增强 | GPT-4.1 (GraphRAG), GPT-4.1-mini (GraphRAG), Qwen3-8B (GraphRAG) |
| 本文方法 | **MetaKube (Ours)** |

> 注：所有 GraphRAG 方法均使用相同图结构知识库，但缺少 EPMN 和 KubeLLM 的经验学习机制。

---

## 3. 主要实验结果和性能指标

### 📊 性能总览（GPT-5 评估，100 分制）

| 方法 | Eff. | Equ. | Com. | S/A | **Avg.** |
|------|------|------|------|-----|----------|
| GPT-4.1 (Zero-shot) | 72.1 | 74.3 | 69.8 | 78.9 | **73.8** |
| Qwen3-8B (Zero-shot) | 48.7 | 51.2 | 46.1 | 57.4 | **50.9** |
| GPT-4.1 (GraphRAG) | 89.3 | 92.6 | 91.4 | 94.1 | **91.9** |
| Qwen3-8B (GraphRAG) | 66.7 | 69.1 | 64.8 | 73.3 | **68.5** |
| **MetaKube (Ours)** | **91.2** | **90.8** | **87.3** | **92.5** | **90.5** |

> ✅ **MetaKube 将 Qwen3-8B 的性能从 50.9 提升到 90.5，提升了整整 40.6 分！**
>
> ✅ 性能已非常接近最强基线 GPT-4.1 + GraphRAG（仅低 1.4 分），但完全本地运行，保护数据隐私。

---

### 👁️‍🗨️ 人类专家评估结果（更贴近实战）

| 方法 | Eff. | Equ. | Com. | S/A | **Avg.** |
|------|------|------|------|-----|----------|
| GPT-4.1 (GraphRAG) | 73.8 | 77.8 | 71.2 | 79.4 | **75.6** |
| **MetaKube (Ours)** | **75.6** | **74.2** | **69.8** | **81.2** | **75.2** |

> 💡 在专家眼中，MetaKube 在 **Effectiveness** 和 **Safety/Accuracy** 上表现最佳，说明其输出更具实用性和可靠性。

---

### 🔬 消融实验（Ablation Studies）

#### （1）EPMN 消融实验（Figure 3）
| 指标 | 有 EPMN | 无 EPMN | 提升幅度 |
|------|--------|--------|---------|
| Effectiveness | 91.2 → | 80.1 | +13.8% |
| Equivalence | 90.8 → | 78.7 | +15.4% |
| Safety/Accuracy | 92.5 → | 78.5 | **+16.6%** |
| **Overall** | 90.5 → | 78.5 | **+15.3%** |

> ✅ **EPMN 贡献了约 15.3% 的性能增益**，尤其在安全性方面显著提升，表明经验记忆对规避错误方案至关重要。

#### （2）KubeLLM SFT 消融（Figure 4）
- 在 KFRD 测试集上，经过 SFT 微调后，KubeLLM 相比原始 Qwen3-8B 实现了 **45.5% 的性能提升**。
- 表明领域特定训练对诊断准确性具有决定性作用。

#### （3）KubeGraph 消融（Table 2）
| 场景 | 有 KubeGraph | 无 KubeGraph | 提升 |
|------|-------------|--------------|-------|
| KubeFault（内域） | 75.2 | 34.6 | **+117.3%** |
| Telecom Dataset（外域） | 57.6 | 22.4 | **+157.1%** |

> ✅ KubeGraph 不仅大幅提升内域性能，还展现出极强的**跨域泛化能力**，证明其结构化因果知识的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **经验学习是关键**：引入 EPMN 后，系统能够从过往成功/失败案例中提炼通用模式，显著提高诊断效率与准确性。
2. **双路径机制更智能**：Meta-Cognitive Controller 成功实现了“简单问题快响应，复杂问题深分析”的自适应行为。
3. **小型模型也能胜任**：通过对 Qwen3-8B 进行高质量 SFT + LoRA 微调，可在本地部署下实现接近 GPT-4 的诊断水平。
4. **结构化知识图谱价值巨大**：KubeGraph 提供可靠的因果推理基础，尤其在外域任务中表现出色。
5. **端到端闭环可行**：MetaKube 实现了“诊断 → 反馈 → 学习 → 优化”的完整闭环，是迈向真正自主运维 AI 的重要一步。

---

### ⚠️ 局限性
1. **初始冷启动问题**：系统初期缺乏历史经验（memory pool 为空），需一定数量的历史故障注入才能发挥 EPMN 优势。
2. **依赖高质量标注数据**：虽然构建了 KFRD，但在某些罕见故障类型上仍可能存在覆盖不足。
3. **动态环境适应性待验证**：实验主要基于静态故障注入，对长期运行、持续演化的集群适应性还需进一步研究。
4. **硬件资源需求较高**：尽管使用 8B 模型，但仍需多块 A100/A6000 GPU 支持，中小企业可能难以负担。

---

### 🔮 未来工作方向
1. **在线增量学习机制**：让 EPMN 支持实时写入新诊断结果，实现真正的持续学习（continual learning）。
2. **多模态输入支持**：集成 Prometheus 指标、Fluentd 日志流、Jaeger 调用链等多源信号作为输入。
3. **自动化执行与验证**：将诊断建议转化为 Ansible Playbook 或 Argo Workflow，实现自动修复闭环。
4. **轻量化版本开发**：探索蒸馏版 KubeLLM 或量化技术，降低部署门槛。
5. **开放生态共建**：推动 KFRD 和 KubeGraph 成为社区标准数据集与知识库，促进经验共享。

---

## 📦 开源信息
- **代码仓库**：[https://github.com/MetaKube-LLM-for-Kubernetes-Diagnosis/MetaKube](https://github.com/MetaKube-LLM-for-Kubernetes-Diagnosis/MetaKube)
- **发布内容**：MetaKube 框架、KFRD 数据集、KubeGraph 知识图谱、KubeLLM 模型权重与训练脚本。

> 该项目已全面开源，为构建下一代经验感知型 AIOps 系统提供了坚实基础。

</details>

---

### 12. [LLM-Driven Reasoning for Constraint-Aware Feature Selection in Industrial Systems](https://arxiv.org/abs/2603.24979)

**Authors**: Yuhang Zhou, Zhuokai Zhao, Ke Li, Spilios Evmorfos, G\"okalp Demirci, Mingyi Wang, Qiao Liu, Qifei Wang, Serena Li, Weiwei Li, Tingting Wang, Mingze Gao, Gedi Zhou, Abhishek Kumar, Xiangjun Fan, Lizhu Zhang, Jiayi Liu  
**Category**: cs.CL  
**Published**: 2026-03-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.24979v1  

#### Abstract
Feature selection is a crucial step in large-scale industrial machine learning systems, directly affecting model accuracy, efficiency, and maintainability. Traditional feature selection methods rely on labeled data and statistical heuristics, making them difficult to apply in production environments...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM-Driven Reasoning for Constraint-Aware Feature Selection in Industrial Systems

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 Feature Selection 方法在大规模工业 ML 系统中面临以下挑战：
- 严重依赖 labeled data，在标注数据稀缺的生产环境中难以应用；
- 忽略业务和操作约束（如 feature group 复杂度、维护成本）；
- 仅基于统计相关性选择特征，无法利用语义信息和 metadata（如 feature 类型、所属团队、重要性分数等）。

### 提出的新方法：Model Feature Agent (MoFA)
MoFA 是一个由 **Large Language Model (LLM)** 驱动的、支持**约束感知**的特征选择框架，其核心创新包括：

- **将 Feature Selection 视为 LLM 的推理任务**  
  利用 LLM 对自然语言的理解能力，结合 feature 的语义描述（name, desc）与定量 metadata（importance, correlation, group），进行可解释的、上下文感知的选择决策。

- **引入“Divide-and-Conquer”策略以实现可扩展性**  
  针对工业系统中成千上万的候选特征（超出 LLM 上下文窗口），采用两阶段机制：
  1. **Phase 1: Bucketed Parallel Selection** —— 将特征划分为多个 bucket，并行地在每个 bucket 中选出 top-K' 候选；
  2. **Phase 2: Global Synthesis and Refinement** —— 合并所有候选进入全局池，再执行一次全局顺序选择，缓解跨 bucket 冗余与共线性问题。

- **支持多维度 operational constraints 的集成**  
  在 prompt 中显式编码约束条件（如最小化 feature group 数量、避免特定类型特征组合），使 LLM 能够在优化预测性能的同时满足工程实践需求。

### 相比现有方法的优势
| 维度 | 传统方法（Filter/Wrapper/Embedded） | LLM-based 方法（如 LLM-Rank） | MoFA |
|------|----------------------------------|-------------------------------|------|
| 数据依赖 | 强依赖 labeled data | 通常仍需离线训练数据 | 可在低监督甚至无监督场景运行 |
| 语义理解 | ❌ 仅处理数值信号 | ✅ 支持文本描述推理 | ✅ + 更强的上下文整合能力 |
| 约束建模 | ❌ 难以表达复杂业务规则 | ❌ 多数忽略 operational constraints | ✅ 显式支持 group、capacity、fairness 等约束 |
| 可解释性 | ❌ 黑箱或弱解释性 | ⭕ 输出理由但缺乏一致性 | ✅ 每步输出 `Selected Feature: ..., Reason: ...`，全程可追溯 |

---

## 2. 核心实验方法和设置

### 使用的数据集与应用场景
论文在三个真实世界的工业推荐系统任务中验证 MoFA：

#### （1）True Interest and Time-Worthiness Prediction (TI & WT)
- **目标**：预测用户是否对内容有“真实兴趣”（True Interest）以及“是否值得花时间”（Time-Worthiness）
- **数据来源**：通过 in-product survey 获取用户反馈标签（binary labels）
- **样本量**：71,754 训练实例，17,939 测试实例
- **特征空间**：1,030 个 user-level 和 content-level 特征，按 owning team 分为多个 **feature groups**

#### （2）Signal Pair Selection for Value Model Enhancement
- **目标**：从约 400 个行为信号对中识别高价值的 **higher-order interaction terms**（如 `p_like × p_profile_tap`），用于增强 value model
- **上线方式**：在线 A/B testing，逐个加入 top-ranked 特征对

#### （3）Notification Behavior Prediction
- **目标**：预测用户是否会点击或忽略通知（multi-task learning）
- **模型架构**：包含 MLP、gating network 的复杂子网络结构
- **特征规模**：从 8,169 个原始特征中选择 4,000 个
- **特征类型**：sparse / dense / float，具有明确的 modality metadata

---

### 实验设置与评估指标

| 任务 | 主要评估指标 | 特征数量 K | MoFA 设置 | Baseline 方法 |
|------|-------------|------------|-----------|----------------|
| TI & WT | AUC on test set<br>Selected feature group count | K ∈ {20, 100, 200, 500} | Llama4-Maverick<br>B=5 buckets<br>K'=1.5×(K/B) | Lasso（有数据时）<br>Random selection（无数据时） |
| Value Model | Daily app session lift (%) | Top-10 interaction pairs | 全局排序（无需分桶） | 人工经验 + 枚举尝试 |
| Notification Prediction | Normalized Entropy (NE) Loss<br>relative win rate | 4,000 features from 8,169 | B=10 buckets<br>K'=450 per bucket | Random sampling of 4,000 features |

---

## 3. 主要实验结果和性能指标

### （1）True Interest & Time-Worthiness Prediction

| K | 方法 | AUC (Test) | Feature Groups |
|----|--------|------------|---------------|
| 20 | Lasso | 71.51 | 9 |
|    | MoFA  | **72.14** | **13** |
| 500 | Lasso | 75.22 | 187 |
|     | MoFA  | 74.81 | **85** |

- **预测性能**：在小预算下（K=20），MoFA 在 TI 任务上显著优于 Lasso（+0.63% AUC）；在大预算下两者接近（差距 <0.5% AUC）。
- **group 效率**：MoFA 实现了显著的 **feature group 压缩**：
  - 在 K=500 时，TI 任务减少 **54%** 的 groups（85 vs 187）；
  - WT 任务减少 **69%** 的 groups（96 vs 313）；
- 这意味着更低的跨团队协作成本和更高的可维护性。

### （2）Value Model Enhancement（在线实验）

MoFA 推荐的前 3 个 interaction pairs 上线后带来的 session 提升如下：

| Rank | Interaction Term | Session Lift (%) |
|------|------------------|------------------|
| 1 | `p_external_share × p_profile_tap` | **+0.055%** |
| 2 | `p_reshare_button_tap × p_profile_tap` | +0.048% |
| 3 | `p_reshare_button_tap × p_comment` | +0.018% |

- 所有提升均具有统计显著性；
- 表明 MoFA 能有效发现 human-hard-to-capture 的高阶协同效应。

### （3）Notification Behavior Prediction

| 模型配置 | P_click NE Win | P_dismiss NE Win |
|----------|----------------|------------------|
| MoFA (4k) vs Production (>7.5k) | +0.065% | +0.198% |
| MoFA (4k) vs Random (4k) | +0.19% | +0.332% |

- MoFA 不仅优于随机选择，在仅使用 **一半不到的特征数** 下还超越了当前生产模型；
- 显示其在高维异构特征空间中的高效筛选能力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM-based reasoning 可作为工业级 feature selection 的可行范式**  
   MoFA 在多个真实场景中证明了其有效性，尤其适用于 labeled data 稀缺、metadata 丰富、且存在多重 operational constraints 的环境。

2. ✅ **语义 + 定量信息融合带来更优权衡**  
   MoFA 能同时考虑 feature 的预测能力和工程影响（如 group 归属），实现 accuracy 与 maintainability 的更好平衡。

3. ✅ **自动发现高阶交互项具备实际业务价值**  
   在 value model 场景中，MoFA 自动识别出的人机难察的 signal pairs 显著提升了用户活跃度。

4. ✅ **Divide-and-Conquer 策略保障可扩展性**  
   成功应用于超千维特征空间（>8k features），解决了 LLM 上下文长度限制问题。

---

### 局限性（Limitations）
1. ⚠️ **计算延迟较高**：由于是 sequential selection，总耗时随 K 线性增长，不适合实时 feature selection；
2. ⚠️ **Token 消耗大**：当候选池极大时，prompt 构造可能逼近甚至超过 LLM 的最大输入长度；
3. ⚠️ **贪婪搜索缺陷**：sequential greedy selection 可能错过需要联合出现才有效的 feature 组合（synergy effect）；
4. ⚠️ **LLM 幻觉风险**：尽管 prompt 设计强调事实依据，但仍存在 reasoning 不一致或虚构理由的风险。

---

### 未来工作方向
- 引入 **beam search 或 MCTS** 替代 greedy selection，探索非局部最优组合；
- 开发 **feedback loop 机制**，利用 downstream model performance 反馈迭代优化 selection policy；
- 探索 **smaller specialized agents** 替代通用 LLM，降低推理成本与延迟；
- 结合 **causal reasoning** 模块，进一步提升所选特征的可解释性与鲁棒性。

--- 

> **总结一句话**：  
> MoFA 成功将 LLM 的语义推理能力引入工业系统的 feature selection 流程，在保持 competitive predictive performance 的同时，显著提升了 feature 子集的可维护性、可解释性和业务对齐能力，为下一代智能 ML pipeline 提供了新的基础设施思路。

</details>

---

### 13. [eBeeMetrics: An eBPF-based Library Framework for Feedback-free Observability of QoS Metrics](https://arxiv.org/abs/2603.25067)

**Authors**: Muntaka Ibnath, Mohammadreza Rezvani, Daniel Wong  
**Category**: cs.DC  
**Published**: 2026-03-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.25067v1  

#### Abstract
Many system management runtimes (SMRs), such as resource management and power management techniques, rely on quality-of-service (QoS) metrics, such as tail latency or throughput, as feedback. These QoS metrics are generally neither observable with hardware performance counters nor directly observabl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：eBeeMetrics: An eBPF-based Library Framework for Feedback-free Observability of QoS Metrics

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

现代数据中心中的 **System Management Runtimes (SMRs)**（如资源管理、功耗管理、调度器等）严重依赖 **Quality-of-Service (QoS) metrics**（如尾延迟 tail latency 和吞吐量 throughput）作为反馈信号来做出决策。然而，这些指标通常面临以下挑战：

- **不可观测性**：硬件性能计数器（hardware performance counters）无法直接捕获应用层 QoS 指标。
- **侵入性高**：需要对应用程序进行插桩（instrumentation），增加开发和维护成本。
- **开销大**：通过 `sysfs` 或系统调用上报指标会引入显著延迟，难以支持实时控制回路。
- **粒度不足**：传统 eBPF 方法只能提供离线的、代理性质的吞吐量估计，缺乏对 **延迟指标** 的支持。

因此，如何在 **无需应用插桩、无反馈机制** 的前提下，实现对服务器端 QoS 指标的 **准确、实时、低开销可观测性** 是一个关键难题。

---

### 🚀 **提出了什么新方法或新思路**

本文提出 **eBeeMetrics** —— 一种基于 **eBPF** 的库框架，用于实现 **feedback-free** 的 QoS 指标可观测性。

#### 核心创新点：

1. **完全在线的实时分析框架**
   - 与先前工作仅能进行 **offline 分析** 不同，eBeeMetrics 支持 **real-time streaming** 处理 eBPF 事件，满足 SMR 对实时反馈的需求。

2. **精准请求边界识别（Request Disambiguation）**
   - 利用 **syscall 元数据**（如 `accept4` 返回的文件描述符 fd）或 **用户态函数钩子**（uprobes on gRPC 库函数）来唯一标识每个请求。
   - 通过匹配请求开始（如 `accept4`）和结束（如 `close` 或 `trailing_metadata_completion`）事件，重建每请求的生命周期。

3. **协议感知的设计（Protocol-aware）**
   - 支持多种主流协议：
     - **HTTP/1.1**：利用 `accept4` 和 `close` syscall 配对。
     - **gRPC/HTTP/2**：使用 uprobes 捕获 `chttp2_stream` 构造和元数据完成事件。
   - 能处理持久连接（persistent connections）和流复用（multiplexing）等复杂场景。

4. **轻量级架构设计**
   - 内核中只收集最小化的请求级元数据（start/end 时间戳），计算任务下沉到用户空间。
   - 使用 **eBPF ring buffer** 实现高效内核-用户通信，避免阻塞和高开销上下文切换。

5. **解耦 SMR 与应用**
   - 提供简洁 API 接口（如 `get_RPS()`、`get_latency_percentile()`），使 SMR 可直接获取指标而无需依赖客户端上报。

---

### 🔍 **相比现有方法的优势**

| 维度 | Prior Work [13] | eBeeMetrics |
|------|------------------|------------|
| 是否在线 | ❌ Offline 分析 | ✅ Fully online |
| 支持延迟指标 | ❌ 仅吞吐量代理 | ✅ 支持 per-request latency |
| 请求边界识别 | ❌ 无法精确划分 | ✅ 基于 fd/stream ID 精确匹配 |
| 应用侵入性 | ⚠️ 需 trace 日志 | ✅ 完全无插桩 |
| 实时性 | ❌ 后处理 | ✅ 流式实时输出 |
| 协议通用性 | ❌ 仅适用于简单模式 | ✅ 支持 HTTP/1.1, gRPC, 持久连接等 |

> ✅ **eBeeMetrics 实现了从“相关代理”到“准确测量”的跨越**。

---

## 2. 核心实验方法和设置

### 🧪 **使用的数据集与工作负载**

在一台双路 AMD EPYC 7302 服务器上评估，运行 Linux 5.15.0，Ubuntu 20.04。

共测试 **10 个真实世界的延迟敏感型工作负载**，涵盖多个基准套件：

- **vSwarm suite**（微服务场景）：
  - Hotel Reservation（搜索、预订、评分）
  - Online Shop（推荐、广告、购物车）
- **CloudSuite**：
  - Data Caching（Memcached）
- **NVIDIA Triton Inference Server**：
  - 支持 HTTP 和 gRPC 两种接口，代表 AI 推理服务典型负载

这些工作负载具有多样化的线程模型、并发行为和协议使用方式，验证了 eBeeMetrics 的通用性。

---

### 🛠️ **实验设置与评估指标**

#### 实验目标：
- 验证 eBeeMetrics 报告的 QoS 指标是否与客户端实际测量值高度一致。
- 评估其对系统性能的影响（overhead）。
- 展示其与真实 SMR（PARTIES）集成的能力。

#### 主要评估指标：

| 指标类型 | 具体指标 |
|--------|---------|
| **准确性** | 客户端报告 vs eBeeMetrics 报告的：<br>• 尾延迟（95th/99th percentile latency）<br>• 吞吐量（RPS） |
| **相关性** | R² 相关系数、趋势一致性 |
| **开销** | 延迟增加百分比、CPU 开销 |
| **功能性** | 是否成功替代 PARTIES 中的客户端反馈机制 |

#### 基线方法对比：
- **客户端直接上报**（Client-reported）：黄金标准（ground truth）
- **Prior work [13]**：基于 `epoll_wait` 时长推断的吞吐量代理指标

---

## 3. 主要实验结果和性能指标

### 📊 **关键性能数据**

#### ✅ **尾延迟跟踪精度高（图7）**

- 在所有 10 个工作负载中，eBeeMetrics 报告的 **95th 百分位延迟** 与客户端测量值高度一致。
- 差异主要来自 **网络延迟**（network latency），而 eBeeMetrics 仅测量 **服务器侧延迟**（server-side latency），这反而成为优势：
  - 可帮助区分是 **网络拥塞** 还是 **服务器过载** 导致延迟上升。
- 例如：
  - vSwarm 平均网络延迟 ~0.75ms
  - Triton HTTP: ~20ms, gRPC: ~3ms
  - Memcached：请求处理时间仅 0.01ms，大部分延迟在网络

> 💡 **发现**：eBeeMetrics 提供的是“干净”的服务器内部视图，有助于根因定位。

---

#### ✅ **吞吐量高度一致（图8）**

- eBeeMetrics 报告的 RPS 与客户端测量值几乎完全重合（R² ≈ 1.0）。
- 在高负载下仍保持强一致性，表明其能应对高并发场景。
- 个别偏差出现在网络饱和区域，属于正常现象。

> 📈 图9(b) 显示：prior work [13] 的代理指标在高 RPS 下明显偏离理想线（y=x），而 eBeeMetrics 始终紧贴。

---

#### ✅ **与 Prior Work [13] 对比（图9）**

| 指标 | Prior Work [13] | eBeeMetrics |
|------|------------------|------------|
| 吞吐量相关性（R²） | < 0.93（随负载下降） | > 0.99（接近 1.0） |
| 是否反映真实延迟 | ❌ `epoll_wait` 是 slack 指标，非服务时间 | ✅ 精确 per-request 延迟 |
| 是否可用于实时控制 | ❌ 仅适合离线分析 | ✅ 支持实时反馈 |

> 🔥 **结论**：[13] 提供的是间接代理信号；eBeeMetrics 提供的是可操作的真实指标。

---

#### ⚖️ **性能开销极低（图10）**

- **平均延迟开销 ≤ 8μs**
- **最大性能影响 < 3%**（vSwarm 工作负载）
- Triton 服务器几乎无可见开销
- 负开销情况由运行间波动引起，说明开销在误差范围内

> ✅ **满足生产环境部署要求**：低延迟、低资源消耗。

---

#### 🔌 **与 SMR 成功集成（图11）**

将 eBeeMetrics 集成进 **PARTIES** 资源管理器，替换原有的客户端反馈路径。

##### 结果：
- 资源分配趋势与原系统基本一致。
- **更早检测到负载缓解**：由于不包含网络排队延迟，eBeeMetrics 更快判断出服务器已恢复，从而更快地降频/缩核，节省更多能量。
- **更适合服务器侧调控决策**：因为频率和核心分配只影响 server-side processing，不应被 network queueing 干扰。

> ✅ **证明了 eBeeMetrics 可作为 plug-in replacement 无缝接入现有 SMR 框架**。

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **eBPF 可以实现高精度的应用层 QoS 观测**  
   无需修改应用代码，仅通过观察 syscall 和用户态函数即可重建请求生命周期。

2. **请求边界可通过轻量元数据可靠识别**  
   文件描述符（fd）、stream ID 等可作为 request identifier，在并发环境下依然有效。

3. **服务器侧延迟是更优的反馈信号**  
   相比客户端总延迟，server-side latency 更能反映真实系统状态，尤其适用于动态资源调控。

4. **eBeeMetrics 具备实用性和通用性**  
   已支持主流协议（HTTP/1.1, gRPC），并可在不同线程模型下稳定运行。

---

### ⚠️ **局限性**

1. **依赖稳定的请求生命周期暴露**
   - 若应用未暴露清晰的 start/end hook（如自定义协议、无标准库），需额外适配。
   - 当前假设每个请求有可追踪的生命周期。

2. **对加密流量有限制**
   - 无法解析 TLS 加密 payload，但不影响基于 fd/stream 的边界识别。

3. **尚未覆盖所有 RPC 框架**
   - 当前聚焦 gRPC 和 HTTP，扩展至其他框架（如 Thrift、Dubbo）需新增探针逻辑。

---

### 🔮 **未来工作方向**

1. **支持更多协议和框架**
   - 扩展至 Kafka、Redis 协议、WebSocket 等常见中间件。

2. **增强自动探测能力**
   - 自动识别应用使用的协议栈，动态加载对应 probe 策略。

3. **结合机器学习进行异常检测**
   - 利用 eBeeMetrics 输出的 rich telemetry 数据训练 anomaly detection 模型。

4. **部署于边缘和 Serverless 环境**
   - 在资源受限设备上验证其轻量化特性。

5. **开源生态整合**
   - 与 Prometheus、OpenTelemetry 等监控系统对接，形成统一可观测性平台。

---

## ✅ 总结

| 项目 | 内容 |
|------|------|
| **核心思想** | 利用 eBPF 实现无插桩、实时、准确的 QoS 指标观测 |
| **关键技术** | 请求边界识别 + 协议感知探针 + 内核-用户协同架构 |
| **最大亮点** | 同时支持 **throughput** 和 **latency** 的在线估计，且精度逼近 ground truth |
| **适用场景** | 动态资源管理、功耗优化、SLA 监控、根因分析 |
| **代码开源** | ✅ [https://github.com/Ibnathism/eBeeMetrics](https://github.com/Ibnathism/eBeeMetrics) （MIT 许可） |

> 🏁 **一句话总结**：  
> **eBeeMetrics 成功打通了 eBPF 从底层事件到高层 QoS 指标的“最后一公里”，为构建智能、自治的数据中心管理系统提供了坚实可观测性基础**。

</details>

---

### 14. [Lightweight Fairness for LLM-Based Recommendations via Kernelized Projection and Gated Adapters](https://arxiv.org/abs/2603.23780)

**Authors**: Nan Cui, Wendy Hui Wang, Yue Ning  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 8.0  
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
- **LLM-based Recommender Systems (RecLLMs)** 虽然具备强大的语义理解和生成能力，但在推荐过程中会**继承并放大预训练数据中的社会偏见**（如性别、年龄、职业等敏感属性）。
- 现有公平性方法存在以下问题：
  - **基于对抗训练的方法**（如 UP5）引入大量可训练参数，优化不稳定；
  - **重排序或曝光控制类方法** 主要关注 item-side 公平，忽视 user-side 差异；
  - 多数方法需要额外训练周期或复杂模块，增加计算开销。

### 🚀 提出的新方法
作者提出一种**轻量级、可扩展的去偏框架**，结合两种核心技术：

#### （1）Kernelized Iterative Null-space Projection (RFF-INLP)
- 将原始 INLP 方法升级为**核化版本**，通过 **Random Fourier Features (RFF)** 显式捕捉 LLM 表征中非线性的敏感信息。
- 引入 **isotropic Gaussian perturbation** 增强投影鲁棒性，防止因微调导致表征漂移而失效。
- 投影矩阵以闭式解（closed-form）计算，**无需梯度更新，不引入任何可训练参数**。

#### （2）Two-level Gated Mixture-of-Experts (MoE) Adapter
- 在每个 Transformer 层后插入一个双门控适配器，实现“**擦除-修复**”（erase-then-repair）机制：
  - **Level-1 Gate**：软混合多个属性专属的投影算子，动态调节各敏感属性的去偏强度；
  - **Level-2 Gate**：基于低秩 LoRA 专家网络，选择性恢复对任务有用但未引入偏见的信息。

### ⭐ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 无额外训练损失函数，投影部分完全冻结，仅 MoE 中少量参数需微调 |
| **稳定性** | 避免对抗训练带来的优化震荡 |
| **灵活性** | 支持多敏感属性联合处理，避免叠加投影造成过度信息丢失 |
| **轻量化** | 总增参数量仅为 $O(k)$（$k$ 为敏感属性数），适合部署 |

---

## 2. 核心实验方法和设置

### 📚 数据集
使用两个真实世界数据集进行验证：

| 数据集 | 类型 | 敏感属性 | 任务形式 |
|--------|------|----------|---------|
| **MovieLens-1M** | 电影评分数据 | Gender (2类), Age (7类), Occupation (21类) | Sequential & Direct 推荐 |
| **Insurance Dataset**（非洲保险公司客户数据） | 保险产品交互 | Marital Status (8类), Age (5类), Occupation (6类) | Direct 推荐（缺乏时间戳） |

> 注：所有用户交互按时间划分训练/验证/测试集。

### 🧪 实验设置与评估指标

#### ✅ 模型架构
- **Backbone**: Instruct Llama-3.2-1B（冻结）
- **Adapter**: LLaRA-style LoRA（rank=32） + 本文提出的 RFF-INLP + Gated MoE（expert rank=8）
- **RFF 参数**: $D=4096$, 噪声尺度 $\eta=0.05$

#### 🔍 评估指标
| 指标类型 | 指标名称 | 描述 |
|--------|--------|------|
| **Utility** | Hit@1 / Hit@3 / Hit@10 | 判断目标 item 是否在 Top-k 推荐列表中 |
| **Fairness** | **Counterfactual Leakage Gap (AcL)** ↓ | 冻结表征后训练 MLP 预测敏感属性的 AUC 与 0.5 的平均偏差；越接近 0 越公平 |

#### 🆚 基线方法
| 方法 | 类型 | 特点 |
|-----|------|------|
| **LLaRA** [24] | 基础模型 | 不含任何去偏机制 |
| **P5** [13] | Prompt-based | 通过提示工程增强推荐 |
| **UP5** [21] | 对抗去偏 | 使用 gradient reversal 和 discriminator 控制公平性 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Tables 2 & 3）

#### ✅ MovieLens - Sequential Recommendation
| 方法 | Hit@1 ↑ | Hit@3 ↑ | Hit@10 ↑ | AcL ↓（G+A+O） |
|------|--------|--------|---------|----------------|
| UP5 | 56.08 | 72.28 | 81.67 | 3.21% |
| **Ours** | **56.08** → **72.28** | **72.28** → **81.67** | **81.67** → **81.67** | **0.17%** |

> ✅ 在保持甚至提升准确率的同时，将多属性平均 AcL 从 3.21% 降至 **0.17%**

#### ✅ MovieLens - Direct Recommendation
| 方法 | Hit@1 ↑（G） | AcL ↓（G） |
|------|-------------|-----------|
| UP5 | 16.38% | 4.19% |
| **Ours** | **36.92%** (+20.5pt) | **0.60%** |

> ✅ 单一属性下，Hit@1 提升超 20 个百分点，AcL 下降近 90%

#### ✅ Insurance Dataset - Direct Recommendation
| 方法 | Hit@1 ↑（M） | AcL ↓（M+A+O avg） |
|------|------------|------------------|
| UP5 | 81.63% | 0.74% |
| **Ours** | 57.37% | **0.13%** |

> ⚠️ 虽然 Hit@1 下降，但 AcL 减少约 **82%**，且多数其他指标持平或领先（如 Hit@3/10）

| 方法 | Hit@3 ↑（Ins.） | Hit@10 ↑（Ins.） | AcL ↓ |
|------|----------------|-----------------|-------|
| UP5 | 91.52% | 97.37% | 0.74% |
| **Ours** | **91.47%** | **99.35%** | **0.13%** |

> ✅ 在绝大多数 utility 指标上持平或反超，同时显著提升公平性

### 🔍 消融实验分析（隐含于文中设计逻辑）
虽然未单独列出消融表，但从方法设计可推断关键组件作用：

| 组件 | 功能 | 验证方式 |
|------|------|----------|
| **RFF-Lift** | 捕获非线性偏见信号 | 若仅用线性 INLP，则残余 AcL 更高（见 Related Work） |
| **Isotropic Perturbation** | 提高投影鲁棒性 | 训练中重复检测 AcL > T 才更新 P，表明其具有自适应能力 |
| **Level-1 Gate (Soft Mixing)** | 动态控制不同属性去偏强度 | 避免“一刀切”式投影破坏有用信息 |
| **Level-2 Gate (LoRA Repair)** | 恢复任务相关特征 | 实验显示 utility 不降反升，证明 repair 有效 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Kernelized INLP 可高效去除多维敏感信息**  
   - 通过 RFF 将非线性偏见显式暴露，并以闭式投影消除，**无需反向传播**，实现真正 lightweight 去偏。

2. **“Erase-then-Repair” 架构优于端到端对抗训练**  
   - 先清除偏见方向，再用门控 LoRA 选择性恢复有用信号，在 **utility 与 fairness 之间取得更好平衡**。

3. **所提方法在多个数据集上显著降低 AcL（趋近于 0）**  
   - 多属性联合去偏时，AcL 平均下降 **80–95%**，达到近乎 counterfactually fair 的水平。

4. **性能不依赖全模型微调**  
   - 仅微调 adapter 参数即可完成修复，backbone 完全冻结，**适合大规模部署**。

### ⚠️ 方法的局限性
- **高度依赖用户交互历史**：若缺乏足够上下文（如冷启动用户），序列级表征难以构建，影响去偏效果。
- **假设敏感属性已知且可标注**：实际应用中可能无法获取用户的 demographic labels。
- **当前仅针对静态属性**：未考虑随时间变化的动态敏感特征。

### 🔮 未来工作方向
- 探索**无需用户历史依赖的去偏机制**；
- 扩展至**零样本或弱监督场景下的敏感属性识别与缓解**；
- 将该框架推广至其他 LLM 下游任务（如对话系统、搜索排序）中的 fairness 保障。

---

> 💡 **一句话总结**：  
> 本文提出了一种基于 **kernelized INLP + gated MoE adapter** 的轻量级去偏框架，能够在**几乎不牺牲推荐准确性的情况下，显著降低 LLM-based 推荐系统中多种敏感属性的信息泄露（AcL ↓80–95%）**，是迈向高效、稳定、实用的公平推荐系统的有力一步。

</details>

---

### 15. [MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning](https://arxiv.org/abs/2603.24044)

**Authors**: Andrea Manzoni  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.24044v1  

#### Abstract
Standard LoRA fine-tuning of Mixture-of-Experts (MoE) models applies adapters to every expert, yet our profiling shows that per-layer expert routing is highly skewed: a small subset of experts handles most tokens in each layer, while many others are rarely activated ("cold"). We propose MoE-Sieve, a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Mixture-of-Experts (MoE)** 模型中，标准的 **LoRA** 微调方法通常将适配器（adapter）应用于所有专家（expert），然而作者通过分析发现：  
- 在每一层中，**专家路由（routing）高度集中**，少数“热”专家处理大部分 token，而多数“冷”专家极少被激活；
- 因此，在所有专家上应用 LoRA 是一种资源浪费，既增加了训练参数量，也可能引入不必要的梯度噪声。

该问题的本质是：**微调时的参数分配未与实际的路由模式对齐**。

---

### 提出了什么新方法或新思路
提出 **MoE-Sieve**，一个基于路由引导的 LoRA 微调框架，其核心流程为三步：

1. **Profile（分析）**：在任务数据上进行一次前向传播，统计每层中每个专家的激活次数；
2. **Select（选择）**：在每层中选择激活次数最多的 top-k 专家；
3. **Fine-tune（微调）**：仅在这些“热”专家上应用 LoRA，其余专家不加适配器。

> ✅ 默认策略：每层选择 **top-25%** 路由专家（如 OLMoE 中 64 个专家选 16 个）。

该方法无需额外超参搜索，仅需一次轻量级分析即可确定适配器部署位置。

---

### 相比现有方法的优势
- **高效性**：大幅减少可训练参数（↓70–73%）、检查点大小（↓71–73%）和训练时间（↓~50%）；
- **有效性**：性能与全专家 LoRA 相当，平均差异在 ±1 pp 内；
- **通用性**：在不同架构（OLMoE、Qwen）和任务（Spider、GSM8K、HellaSwag）上均有效；
- **简单实用**：无需复杂优化，仅需 `profile → count → pick top-k → fine-tune` 一行式流程。

---

## 2. 核心实验方法和设置

### 使用的数据集
用于 **profiling 分析** 和 **微调实验** 的数据集包括：

| 数据集 | 任务类型 |
|-------|--------|
| **Spider** | 文本到 SQL（text-to-SQL），结构化生成 |
| **GSM8K** | 小学数学题（math word problems），符号推理 |
| **HellaSwag** | 常识推理（commonsense reasoning） |

此外，profiling 阶段还使用了另外 7 个数据集（如 MMLU、BoolQ、PIQA、ARC-Challenge、CodeAlpaca、Wiktionary、MBPP）以验证路由模式的普遍性。

---

### 实验设置和评估指标

#### 模型
- **OLMoE-1B-7B**：16 层，64 路由专家/层，无共享专家（shared expert）
- **Qwen1.5-MoE-A2.7B**：24 层，60 路由 + 4 共享专家/层

两者均为激活 8 个专家/ token 的 MoE 架构，但专家粒度和共享机制不同。

#### 微调配置
- **LoRA 设置**：rank=32, α=64, dropout=0.05，应用于选定模块的线性层
- **优化器**：AdamW，学习率 4e-4，训练 3 轮，batch size=64
- **随机种子**：每条件运行 8 个 seed，确保结果稳定

#### 评估指标
- **主指标**：任务准确率（如 execution accuracy for Spider, exact-match for GSM8K）
- **效率指标**：
  - 可训练参数量（Trainable Parameters）
  - Adapter Checkpoint Size
  - Wall-clock Training Time
- **稳定性指标**：seed-to-seed 标准差、置信区间、TOST 等价性检验

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Full LoRA** | 所有路由专家均添加 LoRA 适配器（标准做法） |
| **Hot-25%-LoRA** | MoE-Sieve 方法：仅 top-25% 最常路由专家添加 LoRA |
| **Random-k LoRA** | 随机选择相同数量专家添加 LoRA（控制变量） |
| **Greedy Allocation** | 动态分配预算，最大化覆盖率（更复杂策略） |
| **Coverage-threshold** | 每层选择足够专家以覆盖 60% 路由流量 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

| Model | Task | Full LoRA | Hot-25% | Δ (pp) | 95% CI |
|-------|------|----------|---------|--------|--------|
| OLMoE | Spider | .396±.026 | .399±.015 | +0.30 | [-2.04, +2.64] |
| OLMoE | GSM8K | .304±.011 | .304±.006 | -0.08 | [-1.45, +1.30] |
| OLMoE | HellaSwag | .805±.005 | .807±.008 | +0.17 | [-0.71, +1.05] |
| Qwen | Spider | .520±.014 | .511±.005 | -0.93 | [-1.88, +0.03] |
| Qwen | GSM8K | .590±.011 | .592±.007 | +0.20 | [-0.77, +1.17] |
| Qwen | HellaSwag | .885±.002 | .893±.001 | +0.73 | [+0.53, +0.93] |

> ✅ 所有条件下，**Hot-25% 与 Full LoRA 的平均差异均在 ±1 pp 内**，且多数达到统计等价性（TOST @ ±2pp margin）。

---

### 效率提升（Table 4）

| Model | 参数减少 | Checkpoint 减少 | 训练时间减少 |
|-------|----------|----------------|-------------|
| OLMoE | ↓72.7% (311.5M → 85.0M) | ↓73.4% (1.25GB → 340MB) | ↓50% (1h48m → 54m) |
| Qwen | ↓70.3% (509.7M → 151.3M) | ↓71.0% (2.04GB → 606MB) | ↓49% (3h23m → 1h44m) |

> ⚡ 仅用约 **1/4 的专家适配器**，即可实现几乎相同的性能。

---

### 消融实验结果

#### 🔹 Random vs. Routing-Guided Selection（§6.1）
在 OLMoE × GSM8K 上比较：
- **Hot-k=16**：0.304±0.006
- **Random-k=16**：0.279±0.008 （↓2.5 pp）
- **Hot-k=8**：0.291 > Random-k=16

> ❗ 即使 budget 更高，**随机选择仍不如路由引导的小预算方案**，说明 **routing 信号具有任务特异性价值**。

#### 🔹 Greedy vs. Uniform Top-k（§6.2）
- Greedy 分配试图最大化路由覆盖率；
- 结果显示：**accuracy 无显著差异**，coverage 也几乎一致；
- 表明 uniform top-k 已足够捕捉关键专家。

#### 🔹 Counts vs. Mass Ranking（§6.3）
- 使用 **activation count** 或 **routing mass**（softmax 权重和）排序专家；
- 对 OLMoE 和 Qwen，两者 top-k 选择高度一致（Jaccard > 0.92）；
- 对 DeepSeek 类细粒度专家模型，mass 更优（因低权重大量存在）；
- 推荐默认使用 **count-based ranking**（更简单直观）。

---

## 4. 关键结论和发现

### 主要发现
1. **MoE 模型存在“全局均衡、局部失衡”现象**：
   - 全局负载均衡（global load balancing）掩盖了每层内部严重的路由倾斜；
   - 层内专家激活的 **CV（变异系数）是全局 CV 的 4.0–4.9 倍**。

2. **仅微调 top-25% 热门专家即可媲美全专家 LoRA**：
   - 性能无显著损失（Δ < ±1 pp）；
   - 参数、存储、时间成本下降 70%+。

3. **路由信号至关重要**：
   - 随机选择专家导致性能下降 ~2.5 pp；
   - 证明 **hot experts 承载了任务相关知识**。

4. **更复杂的动态分配策略并无增益**：
   - Greedy 或 coverage-threshold 策略未超越 uniform top-k；
   - 支持采用简单、统一的 25% 规则。

5. **冷专家可能引入梯度噪声**：
   - 完整 LoRA 方案 seed variance 更高（尤其在 OLMoE × Spider）；
   - 支持“**cold expert noise hypothesis**”：冷专家更新稀疏，增加训练不稳定性。

---

### 方法的局限性
- **仅测试两种 MoE 架构**（OLMoE、Qwen），更大规模 MoE（如 DeepSeek-MoE-16B）未验证；
- **固定静态选择**：专家集合在训练前确定，未考虑训练过程中路由变化；
- **阈值经验性**：25% 为经验设定，缺乏理论指导；
- **任务范围有限**：未涵盖安全对齐、指令遵循、多语言等场景。

---

### 未来工作方向
1. **建立理论模型**：从 routing 的 Pareto 分布出发，推导最优专家预算公式；
2. **验证 cold expert noise 假设**：设计对照实验（如 hot + cold filler）分离噪声影响；
3. **探索动态 re-profiling**：在训练中周期性更新专家选择；
4. **扩展至更多架构与任务**：验证方法在更大 MoE 模型和多样化下游任务中的泛化能力；
5. **结合其他 PEFT 技术**：如与 LoRA rank adaptation 或 prompt tuning 结合。

---

## ✅ 总结一句话
> **MoE-Sieve 通过简单的 routing-guided 专家选择，在仅微调 25% 专家的情况下实现了与全 LoRA 相当的性能，同时节省超过 70% 的训练成本，是一种高效、实用、可推广的 MoE 微调新范式。**

</details>

---

### 16. [SCoOP: Semantic Consistent Opinion Pooling for Uncertainty Quantification in Multiple Vision-Language Model Systems](https://arxiv.org/abs/2603.23853)

**Authors**: Chung-En Johnny Yu, Brian Jalaian, Nathaniel D. Bastian  
**Category**: cs.AI  
**Published**: 2026-03-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.23853v1  

#### Abstract
Combining multiple Vision-Language Models (VLMs) can enhance multimodal reasoning and robustness, but aggregating heterogeneous models' outputs amplifies uncertainty and increases the risk of hallucinations. We propose SCoOP (Semantic-Consistent Opinion Pooling), a training-free uncertainty quantifi...

---

### 17. [Mixed-signal implementation of feedback-control optimizer for single-layer Spiking Neural Networks](https://arxiv.org/abs/2603.24113)

**Authors**: Jonathan Haag, Christian Metzner, Dmitrii Zendrikov, Giacomo Indiveri, Benjamin Grewe, Chiara De Luca, Matteo Saponati  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.24113v1  

#### Abstract
On-chip learning is key to scalable and adaptive neuromorphic systems, yet existing training methods are either difficult to implement in hardware or overly restrictive. However, recent studies show that feedback-control optimizers can enable expressive, on-chip training of neuromorphic devices. In ...

---

### 18. [Linear-Nonlinear Fusion Neural Operator for Partial Differential Equations](https://arxiv.org/abs/2603.24143)

**Authors**: Heng Wu, Junjie Wang, Benzhuo Lu  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.24143v1  

#### Abstract
Neural operator learning directly constructs the mapping relationship from the equation parameter space to the solution space, enabling efficient direct inference in practical applications without the need for repeated solution of partial differential equations (PDEs) - an advantage that is difficul...

---

### 19. [Evaluating a Multi-Agent Voice-Enabled Smart Speaker for Care Homes: A Safety-Focused Framework](https://arxiv.org/abs/2603.23625)

**Authors**: Zeinab Dehghani, Rameez Raja Kureshi, Koorosh Aslansefat, Faezeh Alsadat Abedi, Dhavalkumar Thakker, Lisa Greaves, Bhupesh Kumar Mishra, Baseer Ahmad, Tanaya Maslekar  
**Category**: cs.AI  
**Published**: 2026-03-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23625v1  

#### Abstract
Artificial intelligence (AI) is increasingly being explored in health and social care to reduce administrative workload and allow staff to spend more time on patient care. This paper evaluates a voice-enabled Care Home Smart Speaker designed to support everyday activities in residential care homes, ...

---

### 20. [DUPLEX: Agentic Dual-System Planning via LLM-Driven Information Extraction](https://arxiv.org/abs/2603.23909)

**Authors**: Keru Hua, Ding Wang, Yaoying Gu, Xiaoguang Ma  
**Category**: cs.AI  
**Published**: 2026-03-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23909v1  

#### Abstract
While Large Language Models (LLMs) provide semantic flexibility for robotic task planning, their susceptibility to hallucination and logical inconsistency limits their reliability in long-horizon domains. To bridge the gap between unstructured environments and rigorous plan synthesis, we propose DUP...

---

### 21. [PICon: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency](https://arxiv.org/abs/2603.25620)

**Authors**: Minseo Kim, Sujeong Im, Junseong Choi, Junhee Lee, Chaeeun Shim, Edward Choi  
**Category**: cs.CL  
**Published**: 2026-03-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.25620v1  

#### Abstract
Large language model (LLM)-based persona agents are rapidly being adopted as scalable proxies for human participants across diverse domains. Yet there is no systematic method for verifying whether a persona agent's responses remain free of contradictions and factual inaccuracies throughout an intera...

---

### 22. [Kronecker-Structured Nonparametric Spatiotemporal Point Processes](https://arxiv.org/abs/2603.23746)

**Authors**: Zhitong Xu, Qiwei Yuan, Yinghao Chen, Yan Sun, Bin Shen, Shandian Zhe  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23746v1  

#### Abstract
Events in spatiotemporal domains arise in numerous real-world applications, where uncovering event relationships and enabling accurate prediction are central challenges. Classical Poisson and Hawkes processes rely on restrictive parametric assumptions that limit their ability to capture complex inte...

---

### 23. [Wireless communication empowers online scheduling of partially-observable transportation multi-robot systems in a smart factory](https://arxiv.org/abs/2603.23967)

**Authors**: Yaxin Liao, Qimei Cui, Kwang-Cheng Chen, Xiong Li, Jinlian Chen, Xiyu Zhao, Xiaofeng Tao, Ping Zhang  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23967v1  

#### Abstract
Achieving agile and reconfigurable production flows in smart factories depends on online multi-robot task assignment (MRTA), which requires online collision-free and congestion-free route scheduling of transportation multi-robot systems (T-MRS), e.g., collaborative automatic guided vehicles (AGVs). ...

---

### 24. [Lagrangian Relaxation Score-based Generation for Mixed Integer linear Programming](https://arxiv.org/abs/2603.24033)

**Authors**: Ruobing Wang, Xin Li, Yujie Fang, Mingzhong Wang  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.24033v1  

#### Abstract
Predict-and-search (PaS) methods have shown promise for accelerating mixed-integer linear programming (MILP) solving. However, existing approaches typically assume variable independence and rely on deterministic single-point predictions, which limits solution diversityand often necessitates extensiv...

---

### 25. [AVO: Agentic Variation Operators for Autonomous Evolutionary Search](https://arxiv.org/abs/2603.24517)

**Authors**: Terry Chen, Zhifan Ye, Bing Xu, Zihao Ye, Timmy Liu, Ali Hassani, Tianqi Chen, Andrew Kerr, Haicheng Wu, Yang Xu, Yu-Jung Chen, Hanfeng Chen, Aditya Kane, Ronny Krashinsky, Ming-Yu Liu, Vinod Grover, Luis Ceze, Roger Bringmann, John Tran, Wei Liu, Fung Xie, Michael Lightstone, Humphrey Shi  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.24517v1  

#### Abstract
Agentic Variation Operators (AVO) are a new family of evolutionary variation operators that replace the fixed mutation, crossover, and hand-designed heuristics of classical evolutionary search with autonomous coding agents. Rather than confining a language model to candidate generation within a pres...

---

### 26. [AnalogAgent: Self-Improving Analog Circuit Design Automation with LLM Agents](https://arxiv.org/abs/2603.23910)

**Authors**: Zhixuan Bao, Zhuoyi Lin, Jiageng Wang, Jinhai Hu, Yuan Gao, Yaoxin Wu, Xiaoli Li, Xun Xu  
**Category**: cs.AI  
**Published**: 2026-03-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.23910v1  

#### Abstract
Recent advances in large language models (LLMs) suggest strong potential for automating analog circuit design. Yet most LLM-based approaches rely on a single-model loop of generation, diagnosis, and correction, which favors succinct summaries over domain-specific insight and suffers from context att...

---

### 27. [Steering Code LLMs with Activation Directions for Language and Library Control](https://arxiv.org/abs/2603.23629)

**Authors**: Md Mahbubur Rahman, Arjun Guha, Harshitha Menon  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.23629v1  

#### Abstract
Code LLMs often default to particular programming languages and libraries under neutral prompts. We investigate whether these preferences are encoded as approximately linear directions in activation space that can be manipulated at inference time. Using a difference-in-means method, we estimate laye...

---

### 28. [Unveiling Hidden Convexity in Deep Learning: a Sparse Signal Processing Perspective](https://arxiv.org/abs/2603.23831)

**Authors**: Emi Zeger, Mert Pilanci  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.23831v1  

#### Abstract
Deep neural networks (DNNs), particularly those using Rectified Linear Unit (ReLU) activation functions, have achieved remarkable success across diverse machine learning tasks, including image recognition, audio processing, and language modeling. Despite this success, the non-convex nature of DNN lo...

---

### 29. [Efficient Controller Learning from Human Preferences and Numerical Data Via Multi-Modal Surrogate Models](https://arxiv.org/abs/2603.24138)

**Authors**: Lukas Theiner, Maik Pfefferkorn, Yongpeng Zhao, Sebastian Hirt, Rolf Findeisen  
**Category**: cs.LG  
**Published**: 2026-03-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.24138v1  

#### Abstract
Tuning control policies manually to meet high-level objectives is often time-consuming. Bayesian optimization provides a data-efficient framework for automating this process using numerical evaluations of an objective function. However, many systems, particularly those involving humans, require opti...

---

### 30. [The DeepXube Software Package for Solving Pathfinding Problems with Learned Heuristic Functions and Search](https://arxiv.org/abs/2603.23873)

**Authors**: Forest Agostinelli  
**Category**: cs.AI  
**Published**: 2026-03-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.23873v1  

#### Abstract
DeepXube is a free and open-source Python package and command-line tool that seeks to automate the solution of pathfinding problems by using machine learning to learn heuristic functions that guide heuristic search algorithms tailored to deep neural networks (DNNs). DeepXube is comprised of the late...

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
