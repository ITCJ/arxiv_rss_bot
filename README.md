# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-20 06:38:20 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference](https://arxiv.org/abs/2603.19133)

**Authors**: Yida Zhang, Zhiyong Gao, Shuaibing Yue, Jie Li, Rui Wang  
**Category**: cs.DC  
**Published**: 2026-03-20  
**Score**: 14.5  
**Type**: new  
**ArXiv ID**: 2603.19133v1  

#### Abstract
Recent advancements and widespread adoption of Large Language Models (LLMs) in both industry and academia have catalyzed significant demand for LLM serving. However, traditional cloud services incur high costs, while on-device inference alone faces challenges due to limited resources. Edge-cloud col...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*PicoSpec: A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **LLM** 推理在资源受限的 **edge device** 上难以部署，而完全依赖 **cloud** 服务则带来高昂成本和高延迟。虽然 **edge-cloud collaborative inference** 是一个有前景的方向，但现有方案面临以下挑战：
- **Communication-Computation Mismatch**：网络 **RTT** 远高于计算时间，导致流水线中出现大量“气泡”（bubbles）。
- **Stop-and-Wait 架构**：边缘设备必须等待云端验证结果后才能继续生成，严重受制于网络延迟。
- **通信开销大**：传输完整的 **vocabulary distribution** 数据消耗大量带宽。

### 🚀 提出的新方法：**PicoSpec**
PicoSpec 是一种**无需训练**、通用的 **speculative decoding** 框架，专为 **edge-cloud 协同推理** 设计，核心创新如下：

#### （1）**异步协同推测解码流水线（Asynchronous Pipeline）**
- 引入 **Parallel Drafting** 和 **Fast Verification**，将边缘起草（drafting）与云端验证（verification）解耦。
- 边缘设备在等待验证的同时继续“预推测”后续 token，有效隐藏 **RTT** 延迟。
- 云端可在部分 draft token 到达时即启动 **Pre-Verify**，进一步压缩空闲时间。

#### （2）**分离式拒绝采样算法（Separate Rejection Sampling with Sparse Compression）**
- 边缘仅上传 draft token 对应的概率值（如 `q1...qy`），而非整个 **vocabulary distribution**，上行负载从 KB 级降至 <50 bytes。
- 仅当 token 被拒绝时，云端才回传稀疏压缩后的 Top-K 概率分布（`Psent`），下行负载从 ~500KB 降至 <100 bytes。
- 本地执行 **re-sampling**，减少通信频率和数据量。

#### （3）**零拷贝通信与动态截断机制**
- 使用 **Zero-Copy Communicator** 减少序列化开销。
- 实时监控 drafting 时间，若边缘过载，则提前截断并发送当前 draft，避免云端空等。

### 🔍 相比现有方法的优势
| 特性 | PicoSpec | 现有方法（如 DSD、DSSD、SLED） |
|------|--------|-------------------------------|
| 是否需要模型微调 | ❌ 否（Plug-and-Play） | ✅ 多数需修改模型或 fine-tune |
| 是否异步 | ✅ 完全异步 | ❌ 多为串行 stop-and-wait |
| 通信效率 | ✅ 极低（稀疏压缩 + 分离采样） | ❌ 高频或大数据量传输 |
| 通用性 | ✅ 支持任意标准 LLM/SLM | ❌ 往往依赖特定架构 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **GSM8K**：数学推理任务，测试逻辑推导能力。
- **HumanEval**：代码生成任务，评估编程能力。

### ⚙️ 实验设置
| 组件 | 配置 |
|------|------|
| **Edge Device** | NVIDIA Jetson AGX（代表资源受限边缘设备） |
| **Cloud Server** | 配备 NVIDIA A100 (40GB) GPU 的高性能服务器 |
| **网络环境** | 模拟高延迟、低带宽的 **WAN** 环境 |
| **模型对** | 
| - Qwen 3: 0.6B (edge) & 32B (cloud) |
| - Llama 3: 1B (edge) & 70B (cloud) |
| **Speculative Step Size** | 默认设为 4 |

### 📊 评估指标
- **Throughput**（吞吐量）：tokens/s
- **Speedup**：相对于 **Autoregressive (AR)** 基线的加速比
- **Time Per Output Token (TPOT)**：输出每个 token 的平均耗时
- **Token Acceptance Rate / Length**：衡量 draft model 与 target model 的对齐程度
- **Tdraft**, **Tverify**, **TRTT**：各阶段耗时分析

### 🆚 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Autoregressive (AR)** | 云端逐 token 自回归生成 |
| **Vanilla Speculative Decoding** | 传统 speculative decoding，但应用于 edge-cloud 场景 |
| **Split Inference** | 将部分模型层切分到边缘，其余在云端执行 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）
| 方法 | Qwen-GSM8K | Qwen-HumanEval | Llama-GSM8K | Llama-HumanEval |
|------|------------|----------------|-------------|------------------|
| **AR Baseline** | 1.00x (13.89) | 1.00x (14.18) | 1.00x (6.87) | 1.00x (6.86) |
| **Vanilla Spec.** | 0.58x | 0.44x | 1.35x | 1.45x |
| **Split Inf.** | 0.54x | 0.53x | 0.63x | 0.66x |
| **PicoSpec (Ours)** | **1.45x** | **1.13x** | **2.51x** | **2.90x** |

> ✅ **最高实现 2.9× 加速**（Llama-70B + HumanEval）

### 🔍 重要观察
- **Vanilla Speculative Decoding 表现更差**：由于 stop-and-wait 机制，网络延迟完全抵消了推测增益，甚至不如 AR。
- **Split Inference 受限于 RTT**：即使优化切分，仍无法突破网络瓶颈。
- **PicoSpec 在大模型上优势更明显**：Llama-70B 的 `Tverify` 更大，PicoSpec 能更好隐藏该延迟，体现“**Latency Immunity**”。

### 🔬 消融实验结果（Ablation Study）

#### 表：移除关键组件的影响（以 Llama-GSM8K 为例）
| 配置 | Throughput (tokens/s) | TPOT (ms) | Tverify (ms) |
|------|------------------------|-----------|-------------|
| **Full PicoSpec** | **17.22** | **272.89** | **166.46** |
| w/o Parallel Drafting | 12.51 | 390.35 | 194.68 |
| w/o Fast Verification | 16.64 | 287.74 | 189.32 |
| w/o Separate Rejection Sampling | 12.12 | 389.76 | 283.79 |

#### 结论：
- **Parallel Drafting** 是性能提升主因（+37.6% 吞吐），解决了核心依赖问题。
- **Separate Rejection Sampling** 显著降低 `Tverify`（↓41%），缓解通信压力。
- **Fast Verification** 进一步优化流水线连续性，减少 bubble。

### 📊 参数敏感性分析（Draft Length `n`）
| Draft Length | Throughput | Tdraft (ms) |
|-------------|-----------|------------|
| 3 | 19.06 | 73.41 |
| **4** | **20.19** | **97.46** |
| 5 | 18.12 | 121.40 |
| 6 | 17.12 | 145.40 |

> ✅ 最优 `n=4`，此时 `Tdraft ≈ TRTT + Tverify`，实现最佳重叠；过大反而增加本地负担。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **PicoSpec 实现真正的 latency hiding**：
   - 通过异步流水线设计，使系统吞吐不再受 **RTT** 限制，逼近本地 draft model 的物理极限。
2. **通信效率是分布式 speculative decoding 成败的关键**：
   - 传统的 full distribution 传输不可行，必须采用 **sparse compression** 和 **separate sampling**。
3. **无需模型修改即可实现高效协同**：
   - PicoSpec 是纯系统级优化，兼容现有 **LLM** 和 **SLM**，具备强通用性和可部署性。
4. **性能与模型对齐度正相关**：
   - token acceptance length 越高，pipeline hit 率越高，加速效果越显著。

### ⚠️ 局限性
- **依赖 draft model 与 target model 的语义一致性**：若 alignment 差（acceptance rate 低），性能会退化至接近同步 baseline（理论下界 S→1）。
- **边缘设备算力有限**：若 `Tdraft` 过长，仍可能导致 pipeline stall。
- 当前未考虑多客户端场景下的资源竞争与调度问题。

### 🔮 未来工作方向
- 扩展至 **multi-client** 和 **multi-edge node** 协同场景。
- 动态调整 speculative window size 以适应不同输入长度和网络状态。
- 探索轻量化 **draft model** 的自动选择策略，提升跨任务泛化能力。
- 结合 **quantization** 或 **pruning** 技术进一步降低边缘端开销。

---

> **总结一句话**：  
> **PicoSpec 通过异步流水线 + 分离式稀疏采样，在不修改模型的前提下，首次实现了高效的 edge-cloud speculative decoding，最高可达 2.9× 加速，展现出强大的“延迟免疫”能力。**

</details>

---

### 2. [AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models](https://arxiv.org/abs/2603.18464)

**Authors**: Chengxuan Lu, Shukuan Wang, Yanjie Li, Wei Liu, Shiji Jin, Fuyuan Qian, Peiming Li, Baigui Sun, Yang Liu  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2603.18464v1  

#### Abstract
Reinforcement learning (RL) for large-scale Vision-Language-Action (VLA) models faces significant challenges in computational efficiency and data acquisition. We propose AcceRL, a fully asynchronous and decoupled RL framework designed to eliminate synchronization barriers by physically isolating tra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Vision-Language-Action (VLA) 模型在具身智能（Embodied Intelligence）任务中展现出巨大潜力，但将其应用于 **大规模强化学习**（Reinforcement Learning, RL）时面临两大瓶颈：

1. **系统级效率低下**：传统同步 RL 框架存在严重的“长尾延迟”（long-tail latency）和 GPU 利用率低的问题，因训练、推理和 rollout 必须同步进行，导致高性能 GPU 频繁空闲等待慢速物理模拟器。
2. **样本效率极低**：依赖真实环境交互获取经验，采样速度受限于模拟器步进频率和模型推理延迟，难以满足多亿参数 VLA 模型对海量数据的需求。

### 提出了什么新方法或新思路
作者提出 **AcceRL** —— 一种**完全异步、解耦的分布式 RL 框架**，并首次将可训练的 **World Model** 集成到 VLA 的异步训练流程中，实现“在想象中学习”（learning in imagination）。

其核心创新包括：

- ✅ **全异步架构设计（Fully Asynchronous Architecture）**  
  物理隔离训练（Trainer）、推理（Inference Pool）和 rollout（Rollout Worker），消除全局同步屏障，支持持续并行执行。

- ✅ **双层级异步机制（Dual-Level Asynchrony）**
  - **Macro-Asynchrony**：训练与 rollout 解耦，通过 replay buffer 流式传输轨迹，避免集体阻塞。
  - **Micro-Asynchrony**：环境交互与模型推理解耦，采用“推理即服务”（Inference-as-a-Service）模式，由专用 GPU 推理池提供批量响应。

- ✅ **集成可插拔的 World Model（Plug-and-Play Trainable World Model）**  
  引入基于扩散模型的 **observation model Mobs** 和奖励模型 **Mreward**，生成高保真虚拟经验（imaginary rollouts），大幅减少对真实环境的依赖。

- ✅ **算法级优化机制**
  - **Value Re-computation**：重计算过时轨迹的价值目标，缓解策略滞后（policy lag）带来的偏差。
  - **Global Advantage Normalization**：跨节点归一化优势值，提升训练稳定性。
  - **GIPO（Gaussian Importance sampling Policy Optimization）**：替代 PPO 的硬裁剪机制，软化重要性权重，增强对陈旧数据的鲁棒性。
  - **Dynamic Weighted Resampling (DWR)**：动态调整任务采样概率，聚焦困难任务，防止灾难性遗忘。

### 相比现有方法的优势
| 维度 | AcceRL | 现有方法（如 IMPALA、RLinf-VLA、SimpleVLA-RL） |
|------|--------|---------------------------------------------|
| 架构同步性 | 完全异步 | 同步或部分异步 |
| GPU 利用率 | >94%，接近满载 | 显著受制于 straggler effect |
| 扩展性 | 超线性扩展（super-linear scaling） | 受通信开销限制，扩展性差 |
| 样本效率 | 提升 **200×**（20,000%） | 严重依赖真实 rollout |
| World Model 集成 | 支持在线联合训练与推理 | 多为离线使用或未集成 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LIBERO benchmark**：一个面向终身机器人学习的知识迁移基准，包含多个子任务套件：
  - **LIBERO-Spatial**
  - **LIBERO-Object**
  - **LIBERO-Goal**
  - **LIBERO-Long**（长视野任务）
- 所有实验均在 **MuJoCo** 物理引擎中运行，使用 **OSMesa** 进行无头渲染，支持大规模并行采样。

### 实验设置和评估指标
- **主干模型**：基于 **OpenVLA-OFT**（Llama-2-7B 架构）构建 VLA 策略模型。
- **World Model 架构**：
  - **Mobs**：基于 DIAMOND 的扩散模型，用于生成像素级下一帧图像。
  - **Mreward**：基于 OpenVLA-OFT 微调的二分类器，预测成功概率。
- **训练流程**：
  - 真实 rollout 用于训练 World Model 并初始化想象起点。
  - 想象 rollout 在 World Model 中生成，存入 imaginary replay buffer 用于策略更新。
- **评估指标**：
  - **Success Rate (%)**：各 LIBERO 子任务上的平均成功率。
  - **Samples Per Second (SPS)**：衡量系统吞吐量。
  - **GPU Utilization (%)**：硬件利用率。
  - **Training Steps vs. Environment Steps**：评估样本效率与收敛速度。

### 基线方法对比
- **OpenVLA-OFT**：监督微调基线（Behavior Cloning）。
- **SimpleVLA-RL**：基于 PPO 的同步 RL 微调框架。
- **RLinf-VLA**：混合细粒度流水线的异步框架。
- **AcceRL (w/o WM)**：消融版本，不使用 World Model。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📈 系统扩展性与吞吐量（Table 1 & Figure 4）
| Trainer GPUs | Throughput (SPS) | GPU Utilization |
|--------------|------------------|-----------------|
| 1            | 14.13            | 96.45%          |
| 2            | 28.82            | 97.17%          |
| 3            | 42.42            | 94.22%          |
| 4            | 60.33            | 98.36%          |
| 5            | 75.95            | 96.72%          |
| 6            | 90.78            | 96.63%          |
| **7**        | **104.22**       | **95.07%**      |

> ✅ 实现 **超线性扩展**（super-linear scaling），得益于 ZeRO-2 分布式优化技术降低内存占用，允许更大 batch size，提升 Tensor Core 利用率。

#### 🏆 任务性能（Table 2）
| Method             | Spatial (%) | Object (%) | Goal (%) | **Long (%)** | **Average (%)** |
|--------------------|-------------|------------|----------|---------------|------------------|
| OpenVLA-OFT        | 96.2        | 98.3       | 96.2     | 90.7          | 95.35            |
| SimpleVLA-RL       | 99.4        | 99.8       | 99.2     | 98.5          | 99.23            |
| RLinf-VLA          | 99.4        | 99.8       | 98.8     | 94.0          | 98.00            |
| **Ours (AcceRL)**  | **99.6**    | **100.0**  | **98.8** | **99.1**      | **99.38**        |

> ✅ 在所有类别上达到 **State-of-the-Art (SOTA)** 性能，尤其在 **长视野任务（Long）** 上显著优于其他方法（+0.6% vs best prior）。

#### 🔬 World Model 效果（Figure 5）
- **样本效率提升 200×（20,000%）**：
  - 仅用 **10,000 真实环境步数** 即突破 0.8 平均回报阈值。
  - 收敛至近最优性能仅需 **< 400 训练步**，远快于传统 RL。
- **想象 rollouts 提供密集反馈**：潜在奖励结构（potential-based reward）加速学习过程。

### 消融实验结果

#### ✅ **Value Re-computation 消融（Figure 6）**
- 移除该机制后，训练出现明显波动和性能下降。
- 说明其对缓解陈旧价值估计、维持训练稳定至关重要。

#### ✅ **GIPO vs PPO 对比（Figure 7）**
- 使用标准 PPO 出现剧烈震荡，收敛缓慢。
- **GIPO 在 ~8,000 步达到的性能，PPO 需要 ~60,000 步**，表明其在异步环境下具有 **约 7.5 倍的样本效率优势**。
- GIPO 的平滑信任权重有效抑制极端重要性比率，避免梯度消失。

---

## 4. 关键结论和发现

### 主要发现
1. **异步解耦是提升 VLA-RL 效率的关键路径**：通过分离训练、推理与 rollout，AcceRL 成功打破物理模拟器的速度瓶颈，实现 **>94% GPU 利用率** 和 **超线性扩展能力**。
2. **World Model 可极大提升样本效率**：首次实现 World Model 与 VLA 的端到端异步联合训练，使在线样本效率提升 **200×**，验证了“在想象中学习”的可行性与强大潜力。
3. **算法设计必须适配异步架构**：传统的 PPO 在异步场景下表现不佳，而 **GIPO + Value Re-computation + Global Advantage Normalization** 共同构成了稳定高效的异步优化方案。
4. **AcceRL 在复杂控制任务中表现出卓越鲁棒性**：尤其在长视野、易累积误差的任务中，RL 机制显著优于纯模仿学习。

### 方法的局限性
- 当前框架尚未支持 **大语言模型的后训练（post-training）对齐**，无法直接用于通用对话或指令跟随任务。
- World Model 依赖高质量离线数据预训练（文中使用 1,000 条轨迹），冷启动阶段仍需一定真实交互。
- 扩散模型生成虽高保真，但本身计算成本较高，需依赖专用推理池以维持低延迟。

### 未来工作方向
- 将 AcceRL 架构扩展至支持 **全规模语言模型对齐训练**，实现从感知到语言再到动作的统一高效训练。
- 探索更轻量化的 World Model 架构，在保证生成质量的同时进一步降低推理延迟。
- 结合离线 RL 与在线 fine-tuning，构建更强大的混合学习范式。

--- 

> 💡 **一句话总结**：AcceRL 通过“**完全异步架构 + 可训练 World Model**”的双重创新，实现了 VLA 模型在具身控制任务中的 **高吞吐、高样本效率、高稳定性** 强化学习，为大规模基础模型的现实部署提供了可行的技术路径。

</details>

---

### 3. [DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge](https://arxiv.org/abs/2603.19172)

**Authors**: Yuegui Huang, Zhiyuan Fang, Weiqi Luo, Ruoyu Wu, Wuhui Chen, Zibin Zheng  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.19172v1  

#### Abstract
Despite the computational efficiency of MoE models, the excessive memory footprint and I/O overhead inherent in multi-expert architectures pose formidable challenges for real-time inference on resource-constrained edge platforms. While existing static methods struggle with a rigid latency-accuracy t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge》总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在边缘设备上部署 **Mixture-of-Experts (MoE)** 模型面临两大挑战：
- **巨大的内存占用**：尽管 MoE 模型具有稀疏激活特性（仅使用部分专家），但其总参数量远超边缘设备的 VRAM 容量（如 RTX 3090 仅有 24GB）。
- **严重的 I/O 开销**：为缓解内存压力，通常将非活跃专家 **offload 到主机内存或 SSD**，但在推理时按需加载会导致显著的 `Wait-for-Weight` 延迟，严重拖慢推理速度。

现有静态压缩方法（如 uniform quantization、static mixed-precision）无法适应输入动态变化，导致精度损失大或优化不足。

---

### **提出了什么新方法或新思路**

作者提出 **DyMoE** —— 一种面向边缘计算的 **算法-系统协同设计框架**，通过 **动态混合精度量化（Dynamic Mixed-Precision Quantization）** 实现高效 MoE 推理。

其核心思想基于三个关键观察：

1. **Dynamic Skewness（动态重要性偏斜）**  
   少数“重击令牌”（heavy-hitter tokens）决定了专家的重要性，且这种分布随输入动态变化，因此应动态识别关键专家。

2. **Depth-Dependent Sensitivity（深度依赖敏感性）**  
   浅层对量化噪声极为敏感，而深层具备更强的鲁棒性，允许更激进的低比特表示（如 Int2 或跳过）。

3. **Inter-layer Predictability（层间可预测性）**  
   相邻 Transformer 层的激活状态高度相似，可用于提前预测下一层的关键专家，实现 **look-ahead prefetching**。

基于以上洞察，DyMoE 引入三大机制：

#### ✅ 动态专家重要性分类（Phase-Adaptive Expert Importance Estimator）
- **Prefill 阶段**：基于 attention score 聚合 token 重要性，统计每个专家处理的“重击 token”数量作为其重要性评分。
- **Decode 阶段**：直接使用 **gating score** 作为专家重要性的实时代理。

#### ✅ 深度感知的精度调度器（Depth-Aware Precision Scheduler）
- 采用余弦衰减策略分配各层的高精度专家比例 $ r(l) $，确保浅层保留更多高精度专家，深层逐步过渡到低精度。
- 支持多种配置，如 `"4/2"`（关键专家用 4-bit，次关键用 2-bit）、`"4/0"`（非关键专家完全跳过）。

#### ✅ 动态专家编排引擎（Dynamic Expert Orchestration Engine）
- **Look-ahead Prefetcher**：利用当前层隐藏状态预测下一 layer 的 gating 分布，提前预取关键专家权重，掩盖 I/O 延迟。
- **Mixed-Precision Cache Manager**：扩展 LRU 缓存支持多精度版本管理，并制定三条规则避免冗余与精度降级。

---

### **相比现有方法的优势**

| 维度 | 现有方法局限 | DyMoE 改进 |
|------|--------------|-----------|
| **量化方式** | 静态、统一比特宽度（Uniform PTQ） | 动态、混合精度、运行时决策 |
| **专家选择** | 固定保留比例或随机剪枝 | 基于 token/gate 的动态优先级排序 |
| **I/O 优化** | 被动加载或简单预取 | 主动 lookahead 预取，最大化计算-I/O 重叠 |
| **适用场景** | 需要离线校准或再训练 | **Plug-and-play**，无需 retraining 或 calibration |

> ✅ **优势总结**：DyMoE 在保持模型准确率的同时，显著降低 TTFT 和 TPOT，特别适合资源受限、延迟敏感的边缘应用场景。

---

## 2. 核心实验方法和设置

### **使用的模型与数据集**

#### 🧠 模型
- **Mixtral-8×7B**：粗粒度 MoE 架构（低稀疏性）
- **Qwen3-30B-A3B**：细粒度 MoE 架构（高稀疏性）

#### 📊 数据集
- **ShareGPT**：用于生成真实对话序列，模拟实际用户请求。
- **Benchmark Suite**（评估准确性）：
  - **MMLU**：多任务语言理解
  - **CMMLU**：中文多任务理解
  - **GSM8K**：数学推理题

---

### **实验设置**

| 项目 | 设置说明 |
|------|----------|
| **硬件平台** | AMD EPYC 7542 CPU + NVIDIA RTX 3090 (24GB)，通过软件限制模拟 12–16GB 边缘环境 |
| **批大小** | Batch Size = 1（模拟单用户连续交互） |
| **量化方案** | 使用 GPTQ 进行后训练量化；支持 `"4/2"` 和 `"4/0"` 配置 |
| **缓存机制** | 自定义混合精度 LRU 缓存，支持不同精度版本共存与升级 |

---

### **评估指标**

| 指标 | 含义 |
|------|------|
| **TTFT (Time-to-First-Token)** | 首个输出 token 的延迟，反映响应速度 |
| **TPOT (Time-Per-Output-Token)** | 平均每生成一个 token 所需时间，决定流式输出流畅度 |
| **Accuracy** | 在 MMLU、CMMLU、GSM8K 上的得分，衡量语义保真能力 |

---

### **基线方法对比**

| 基线 | 简介 |
|------|------|
| **Accelerate (acc)** | HuggingFace 提供的通用异构设备支持框架，支持 offloading 和量化 |
| **Mixtral-Offloading** | MoE 专用 offloading 框架，使用 LRU 缓存和混合精度 |
| **MoE-Infinity** | 支持 activation-aware prefetching 和细粒度缓存 |
| **Fiddler** | CPU-GPU 协同执行框架，动态卸载计算任务 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔥 端到端推理加速（vs. 最优基线）

| 模型 & 场景 | 指标 | 加速倍数 |
|------------|------|---------|
| **Mixtral-8×7B @24GB** | TTFT ↓ | **22.7×** faster than Fiddler |
| **Mixtral-8×7B @16GB** | TPOT ↓ | **14.58×** speedup vs. Fiddler |
| **Qwen3-30B-A3B @12GB** | TTFT ↓ | **3.44×** reduction vs. Accelerate |

> 💡 表明 DyMoE 在极端内存受限条件下仍能实现数量级级别的延迟下降。

---

### **与基线方法的对比结果**

| 方法 | TTFT (↓越好) | TPOT (↓越好) | 准确率 (↑越好) |
|------|-------------|-------------|----------------|
| **Fiddler** | 高 | 极高 | 中等 |
| **Accelerate** | 中高 | 高 | 接近原始模型 |
| **Mixtral-Offloading** | 中 | 中 | 受限于静态策略 |
| **DyMoE (4/0)** | **极低** | **极低** | **几乎无损** |

- DyMoE 在所有配置下均优于基线，尤其在 **decode 阶段**表现突出，因其有效掩盖了 I/O stall。
- 在 **Qwen3-30B-A3B** 上甚至出现“反直觉增益”：`"4/0"` 在 GSM8K 上得分 **超过 Int4 全量模型**（91.74% vs 89.08%），可能因低精度起到正则化作用。

---

### **消融实验结果（Ablation Study）**

在 Mixtral-8×7B 上进行组件增量测试（@16GB/24GB）：

| 配置 | TPOT (s) | 相对提升 |
|------|--------|---------|
| 1. Load-on-Demand | 0.2795 | 基线 |
| 2. +Cache | 0.1489 | **1.88×** |
| 3. +Prefetch | 0.1315 | 再提速 1.13× |
| 4. +Dyquant (4/2) | 0.1307 | 显著减少 I/O 体积 |
| 5. Full DyMoE (4/2) | 0.1150 | 总计 **2.43×** 加速 |
| 6. Full DyMoE (4/0) | **0.1048** | 最高达 **2.67×** 解码加速 |

> ✅ 结论：**动态量化 + 预取 + 缓存** 三者协同带来显著叠加收益。

---

## 4. 关键结论和发现

### **主要发现**

1. **专家重要性是动态且可预测的**  
   重击 token 和 gating score 是判断专家重要性的可靠指标，支持运行时动态决策。

2. **深层网络对量化更具鲁棒性**  
   允许在深层使用 Int2 甚至完全跳过非关键专家，而不显著影响最终输出质量。

3. **lookahead prefetching 可有效掩盖 I/O 延迟**  
   利用相邻层激活相似性，提前加载下一层所需专家，实现计算与传输的高度并行。

4. **DyMoE 实现了灵活的精度-资源权衡**  
   用户可通过调节 `retention ratio r` 动态平衡延迟与精度，适用于不同负载场景。

---

### **方法的局限性**

| 局限 | 说明 |
|------|------|
| **依赖 gating score 的可靠性** | 若 gating 网络不稳定，可能导致错误跳过关键专家 |
| **对新型 MoE 架构泛化性待验证** | 当前实验集中在标准 Top-2 routing MoE，是否适用于 Switch Transformer 或 HyperMixer 类结构尚不明确 |
| **PCIe 带宽仍是瓶颈** | 尽管优化 I/O，但在更低带宽链路（如 USB-C 或嵌入式平台）中效果可能打折扣 |

---

### **未来工作方向**

1. **结合 KV Cache 压缩技术**（如 PyramidKV）进一步降低内存压力。
2. **探索硬件协同设计**：定制 FPGA/ASIC 支持动态精度切换与快速 dequantization。
3. **引入轻量微调机制**：在边缘端进行局部 adapter tuning，以补偿极端量化带来的误差。
4. **支持多模态 MoE 模型**：拓展至 vision-language 模型中的 MoE 推理优化。

---

## ✅ 总结

**DyMoE 是首个将动态专家重要性感知、深度自适应量化与 lookahead 预取相结合的 MoE 推理框架**。它打破了传统静态压缩的僵局，在无需 retraining 的前提下实现了：

- **高达 22.7× 的 TTFT 降低**
- **最高 14.58× 的 TPOT 加速**
- **几乎无损的模型准确率**

> 🚀 特别适用于 **边缘侧部署大规模 MoE 模型** 的场景，为“本地化、低延迟、高精度”的 LLM 应用提供了可行路径。

</details>

---

### 4. [Accurate and Efficient Multi-Channel Time Series Forecasting via Sparse Attention Mechanism](https://arxiv.org/abs/2603.18712)

**Authors**: Lei Gao, Hengda Bao, Jingfei Fang, Guangzheng Wu, Weihua Zhou, Yun Zhou  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.18712v1  

#### Abstract
The task of multi-channel time series forecasting is ubiquitous in numerous fields such as finance, supply chain management, and energy planning. It is critical to effectively capture complex dynamic dependencies within and between channels for accurate predictions. However, traditional method paid ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Accurate and Efficient Multi-Channel Time Series Forecasting via Sparse Attention Mechanism*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**多通道时间序列预测**（multivariate time series forecasting）中的两个核心挑战：
- **计算效率低下**：传统基于 self-attention 的模型在长序列上存在 $O(L^2)$ 的计算复杂度，难以扩展。
- **多模态信息融合困难**：现实场景中常伴随丰富的辅助信息（如日期、商品类别、门店ID等），如何有效融合这些异构的 multimodal 信息以提升预测准确性仍是一个挑战。

### 提出的新方法与创新思路
作者提出了一种名为 **Li-Net** 的新型架构，其核心创新包括：

#### ✅ **Top-K Sparse Attention + 多尺度投影框架**
- 设计了一个动态可学习的 **Top-K Softmax attention** 机制，在时间和通道两个维度上进行稀疏化注意力计算。
- 通过 Top-K 近似实现特征压缩与聚焦，显著降低内存占用和计算开销，同时保留捕捉长程依赖的能力。

#### ✅ **统一的多模态信息融合框架**
- 构建了一个端到端的 **multimodal embedding module**，将原始时间序列、日期嵌入（date embedding）、概念嵌入（concept embedding）、门店嵌入（store embedding）以及实时销售状态等异构信息联合编码。
- 利用这些嵌入向量引导稀疏注意力过程，使其更准确地识别对预测最关键的 time steps 和 feature channels。

#### ✅ **灵活高效的通用预测框架**
- 采用 **Encoder-Decoder 结构**，集成一个可配置的非线性模块（non-linear module），支持替换为 **MLP 或 Transformer** 编码层。
- 实现了表达能力与计算效率之间的灵活权衡，增强了模型的通用性和实用性。

### 相比现有方法的优势
| 维度 | Li-Net 的优势 |
|------|---------------|
| **准确性** | 在多个真实世界数据集上达到 SOTA 或接近 SOTA 的预测精度（MAE/MSE 最低）。 |
| **效率** | 显著更低的训练/推理时间、内存占用和模型大小（平均仅 0.5MB）。 |
| **可解释性** | 稀疏注意力机制使模型关注的关键时间点和通道更具可解释性。 |
| **灵活性** | 支持多种 backbone（MLP/Transformer）和多模态输入，适用于不同任务需求。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在五个公开且广泛认可的时间序列基准数据集上进行，覆盖多个领域：

| 数据集 | 描述 |
|--------|------|
| **ETT** | 电力变压器负载与油温数据，含每小时（ETTh2）和每15分钟（ETTm2）采样子集，测试长期依赖建模能力。 |
| **Electricity** | 321个用户的逐时用电量（kWh），具有强周期性和跨用户相关性。 |
| **Weather** | 德国气象站记录的21项气象指标（每10分钟一次），强调变量间复杂的非线性交互。 |
| **Traffic** | 旧金山高速公路862条车道的逐时占有率，典型时空预测问题。 |
| **M5** | Walmart 提供的大规模分层销售预测数据集，包含3049种产品在10个门店的日销量，需处理多尺度季节性和聚合一致性。 |

### 实验设置与评估指标
- **划分比例**：训练集:验证集:测试集 = 60% : 20% : 20%
- **Batch Size**: 16
- **优化器**：AdamW
- **硬件平台**：NVIDIA L40S GPU
- **实现框架**：PyTorch 2.6.0 + CUDA 12.4

#### 评估指标从三个维度综合衡量：
| 类别 | 指标 | 说明 |
|------|------|------|
| **预测准确性** | MAE, MSE | 衡量预测偏差与误差方差，越低越好 |
| **计算效率** | Training/Testing Time | 反映训练和推理速度 |
| **部署可行性** | Memory Usage, Model Size | 决定是否可在资源受限设备部署 |

### 基线方法对比
选取了代表性的先进模型作为 baseline，涵盖多种架构类型：

| 类型 | 基线模型 |
|------|---------|
| **Transformer 变体** | iTransformer, PatchTST |
| **MLP-based 模型** | TimeMixer, TSMixer |
| **专用预测模型** | TFT (Temporal Fusion Transformer) |
| **卷积网络** | TCN (Temporal Convolutional Network) |

所有 baseline 均来自公认的 **Tsinghua Time Series Library**，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（部分代表性结果）
> 注：以下数值为 MSE 或 MAE，越小越好；完整结果见原文 Table I。

| Dataset & Horizon | Li-Net (MAE) | Best Baseline (e.g., PatchTST/iTransformer) | 提升幅度 |
|-------------------|-------------|-------------------------------------------|----------|
| ETTh2 (96-step)   | **0.282**   | 0.3406 (TimeMixer)                         | ↓ ~17%   |
| ETTm2 (96-step)   | **0.2284**  | 0.2610 (TimeMixer)                         | ↓ ~12.5% |
| Electricity (96)  | **0.275**   | 0.2762 (TimeMixer)                         | 接近持平但更高效 |
| Traffic (96)      | **0.3171**  | 0.3337 (TimeMixer)                         | ↓ ~5%    |
| M5 (96)           | **0.2051**  | 0.2085 (TimeMixer)                         | 小幅领先 |

- **总体表现**：在 **32 个实验设置中，Li-Net 在 23 项中排名第一或第二**。
- **平均指标**：
  - 平均 MAE: **0.3443**
  - 平均 MSE: **0.2993**
  - 均优于 iTransformer（MAE 0.4064）、PatchTST（MAE 0.3673）等主流模型。

### 与基线方法的对比结果
| 指标 | Li-Net 表现 |
|------|------------|
| **预测精度** | 优于或媲美当前 SOTA 模型（如 PatchTST, iTransformer） |
| **训练时间** | 在 ETTh2 上仅为 38–60 秒，远低于 TFT/PatchTST（常超百秒） |
| **测试延迟** | 仅 **0.4–0.56 秒**，而其他模型可达数秒甚至数十秒（如 Traffic 上 TFT 超 10s） |
| **内存使用** | 训练内存最低仅 **27.71MB**（Electricity），远低于 iTransformer (>400MB) |
| **模型大小** | 平均仅 **0.5MB**，而 TFT 达 **26.8MB**，M5 上甚至接近 **1GB** |

> ✅ **结论**：Li-Net 实现了“高精度 + 高效率 + 小体积”的理想平衡，特别适合工业级实时预测系统部署。

### 消融实验结果（Ablation Study）

设计了三种变体来验证各组件的有效性：

| 变体 | 修改内容 | 性能变化 |
|------|----------|----------|
| **Li-Net-Softmax** | 用标准 Softmax 替换 Top-K Softmax | MAE/MSE 上升，证明稀疏注意力在保持精度的同时显著提效 |
| **Li-Net-Primitive** | 移除所有 multimodal embeddings | 性能大幅下降，尤其在 M5 和 Electricity 上，说明多模态融合至关重要 |
| **Li-Net-MLP** | 移除 encoder-decoder 架构，直接用三层 MLP 映射 | 性能崩溃，表明 multi-scale projection 和 feature reconstruction 是核心机制 |

> 🔍 发现：**multi-scale 投影结构**和**多模态融合**是性能基石，而**稀疏注意力**是实现高效的关键。

---

## 4. 关键结论和发现

### 主要发现
1. **稀疏注意力机制可以有效替代全注意力**，在不牺牲预测精度的前提下，极大降低计算成本。
2. **多模态信息融合显著提升预测性能**，尤其是静态元信息（如 store/item ID）和上下文信号（如节假日）。
3. **encoder-decoder + multi-scale projection 架构**对于高维多通道时间序列的信息压缩与重建至关重要。
4. **Li-Net 在精度与效率之间实现了最优平衡**，是目前最适合实际部署的多通道时间序列预测方案之一。

### 方法的局限性
- 当前框架主要面向离线批量预测，尚未探索在线学习（online learning）或增量更新能力。
- 对极端稀疏或缺失严重的数据未做专门处理。
- 多模态 embedding 的设计依赖预定义类别字段，对纯文本描述的支持有限。

### 未来工作方向
1. 扩展至 **online learning 场景**，支持流式数据持续更新。
2. 探索 **multi-granularity 时间序列预测**（如同时预测小时级与日级）。
3. 研究 **cross-domain transfer learning**，将在某一领域训练的 Li-Net 应用于其他领域。
4. 引入 **causal inference** 或 **counterfactual reasoning** 能力，增强决策支持功能。

---

> 📌 **总结一句话**：  
> **Li-Net 通过 Top-K Sparse Attention 与多模态融合，在保持 SOTA 预测精度的同时，实现了前所未有的计算效率与部署友好性，为工业级多通道时间序列预测提供了新的范式。**

</details>

---

### 5. [Bridging Network Fragmentation: A Semantic-Augmented DRL Framework for UAV-aided VANETs](https://arxiv.org/abs/2603.18871)

**Authors**: Gaoxiang Cao, Wenke Yuan, Huasen He, Yunpeng Hou, Xiaofeng Jiang, Shuangwu Chen, Jian Yang  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.18871v1  

#### Abstract
Vehicular Ad-hoc Networks (VANETs) are the digital cornerstone of autonomous driving, yet they suffer from severe network fragmentation in urban environments due to physical obstructions. Unmanned Aerial Vehicles (UAVs), with their high mobility, have emerged as a vital solution to bridge these conn...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Bridging Network Fragmentation: A Semantic-Augmented DRL Framework for UAV-aided VANETs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在城市环境中，由于建筑物遮挡和车辆高移动性，**VANETs**（Vehicular Ad-hoc Networks）常出现严重的**网络碎片化**（network fragmentation），导致通信中断。传统基于 **DRL** 的 **UAV** 部署策略缺乏对道路拓扑的语义理解，容易陷入盲目探索，训练效率低、泛化能力差。

此外，尽管 **LLM** 具备强大的推理能力，但其在控制任务中的直接应用仍面临挑战，如输出不稳定、难以与 DRL 有效融合等。

### 提出的新方法与新思路
本文提出了一种全新的 **Semantic-Augmented DRL (SA-DRL)** 框架，核心创新如下：

- **图论建模量化碎片化程度**  
  提出 **Road Topology Graph (RTG)** 和 **Dual Connected Graph (DCG)** 来精确建模城市道路结构和车辆连通状态，将网络碎片化问题形式化为“动态双图连通性最大化”问题。

- **四阶段 SA-DRL 框架**  
  构建了一个从环境经验采集到 LLM 知识对齐再到 DRL 决策增强的完整流程：
  1. **Experience Collection**：通过轻量级 PPO 探索收集代表性状态；
  2. **Semantic Prior Construction**：将图状态序列化为文本，构建监督微调数据集；
  3. **Knowledge Alignment**：使用 **LoRA** 对通用 **LLM** 进行参数高效微调，使其成为道路拓扑专家；
  4. **Semantic-Augmented Training & Execution**：提出 **SA-PPO** 算法，通过 **Logit Fusion** 将 LLM 的语义先验注入策略网络。

- **Logit Fusion 机制**  
  在策略生成阶段，将 LLM 输出的动作得分（作为先验分布）与 DRL 网络的 logits 进行加权融合，并引入 **KL 正则项** 约束策略偏离语义先验，实现“高层语义引导 + 底层数据驱动”的协同决策。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **训练效率** | 仅需基线方法 26.6% 的训练 episode 即可收敛 |
| **性能表现** | 显著提升两个关键连通性指标（+13.2%, +23.5%） |
| **能耗控制** | 能耗降至基线的 28.2%，飞行距离大幅降低 |
| **泛化能力** | 在不同交通流分布下表现出更强鲁棒性和自适应性 |
| **避免模式崩溃** | 成功缓解 SAC 中的 mode collapse 问题 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用来自文献 [37] 的真实城市轨迹数据，涵盖中国两个城市的交通监控记录。
- 构建了一个小规模子集用于仿真实验：
  - 包含 **47 个节点**（交叉口）、**88 条边**（路段）
  - 共 **5,000 条车辆轨迹记录**
  - 时间跨度覆盖多个时段（08:00–18:00）

### 实验设置
- **仿真平台**：基于 Python 自研高保真模拟器
- **UAV 模型**：
  - 固定飞行高度 $ H $
  - 移动目标为交叉口（intersection）
  - 采用 hover-and-transmit 模式以保证通信质量
- **通信模型**：采用概率性 **Air-to-Ground (A2G)** 信道模型，考虑 LoS/NLoS 切换
- **训练配置**：
  - 硬件：Intel i7-14700KF + NVIDIA RTX 4080 Super
  - 框架：PyTorch（SA-PPO）、LLaMA Factory（微调）、vLLM（推理加速）

### 评估指标
| 类别 | 指标 |
|------|------|
| **连通性** | - `K(t)`：连通组件数量（越小越好）<br>- `C(t)`：平均每个组件内的车辆数（越大越好） |
| **能耗** | - 总能量消耗<br>- 平均飞行距离 |
| **训练效率** | 收敛所需 episode 数 |
| **语义对齐质量** | - BLEU-4, ROUGE-L<br>- Kendall’s τ（排序相关性）<br>- Top-k Hit Rate (HRk)<br>- JSON Parsing Success Rate (PSR) |

### 基线方法对比
- **SAC**（Soft Actor-Critic）：Off-policy 强化学习算法，强调熵最大化探索
- **Vanilla PPO**：标准 PPO，无任何结构或语义增强
- **GAT-PPO**：使用 Graph Attention Network 替代 MLP 作为特征提取器的 PPO 变体
- **SA-PPO (Ours)**：本文提出的语义增强方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Fig. 10 与正文）
| 方法 | 平均连通块内车辆数 (C) | 平均飞行距离 (m) | 能耗占比（相对基线） |
|------|------------------------|------------------|---------------------|
| **SAC** | 35.7 | 45.7 | >100% |
| **Vanilla PPO** | 38.2 | 954.0 | 100% |
| **GAT-PPO** | 36.1 | 1158.2 | >150% |
| **SA-PPO (Ours)** | **41.9** (+13.2%↑) | **223.7** | **28.2%** |

> ✅ **SA-PPO 在提升连通性的同时，显著降低了飞行距离和能耗**

### 与基线方法的对比结果
- **收敛速度**：
  - SA-PPO 在约 **2,500 个 episode** 后即稳定收敛
  - 其他方法需要超过 **9,400 个 episode**
  - ➜ **仅需 26.6% 的训练量即可达到同等性能**

- **连通性优势**：
  - 相比 Vanilla PPO，`C(t)` 提升 **13.2%**
  - 相比 GAT-PPO，提升 **23.5%**

- **能耗优势**：
  - 飞行距离仅为 Vanilla PPO 的 **23.4%**
  - 总能耗降至基线的 **28.2%**

- **鲁棒性测试（跨时间段泛化）**：
  - 在稀疏流量场景（如 12:00）中，SA-PPO 连通性优于 PPO **14% 以上**
  - 在密集流量场景（如 16:00）中，SA-PPO 主动减少飞行（471.3m vs 954.0m），实现节能与连通性的智能权衡

### 消融实验结果（Ablation Study）
| 方法 | C(t) | 飞行距离 (m) | 说明 |
|------|------|--------------|------|
| **w/o LLM (Vanilla PPO)** | 38.2 | 954.0 | 盲目探索，效率低下 |
| **w/o Logit Fusion (Pure Semantic Policy)** | 40.1 | 980.6 | LLM 具备良好先验，但无法约束能耗行为 |
| **SA-PPO (Full)** | **41.9** | **223.7** | LLM 与 DRL 协同，实现最优平衡 |

> 🔍 发现：即使不经过强化学习，仅靠微调后的 LLM 就能超越 Vanilla PPO，证明了**语义先验的有效性**；而 **Logit Fusion** 是实现低能耗的关键机制。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 可被有效转化为领域专家**  
   通过 LoRA 微调，通用 LLM（如 Qwen2.5-3B）可以学会理解城市道路拓扑结构，识别关键连接点（cut vertices），并输出合理的动作评分。

2. **语义先验显著提升 DRL 效率与性能**  
   将 LLM 的“常识推理”能力以 **Logit Fusion** 形式注入 PPO，可有效引导 agent 避免无效探索，快速聚焦于拓扑关键节点。

3. **简单 MLP 优于复杂 GNN**  
   实验表明，GAT-PPO 因过度敏感于局部交通波动而导致频繁抖动和高能耗，反而不如 Vanilla PPO。这说明**结构复杂性不等于性能优越**，而 SA-PPO 通过引入稳定语义先验解决了该问题。

4. **SA-PPO 实现多目标优化平衡**  
   在连通性最大化与能耗最小化之间取得优异平衡，展现出“战略驻留”（strategic stationing）行为——即精准定位关键交叉口并长期悬停，而非盲目巡逻。

### 方法的局限性
- **依赖高质量地图信息**：需已知完整的道路拓扑图（RTG），在未知或部分可观测环境下适用性受限。
- **LLM 推理延迟**：尽管采用批量推理（batch inference）和 vLLM 加速，LLM 的实时响应仍是系统瓶颈之一。
- **单 UAV 设计**：当前框架基于单无人机场景，扩展至多 UAV 协同需进一步研究。
- **静态道路假设**：未考虑施工、事故等临时拓扑变化的影响。

### 未来工作方向
- 扩展至 **multi-UAV coordination** 场景，研究分布式 SA-DRL 架构
- 结合 **online map updating** 机制，应对动态道路环境
- 探索 **end-to-end vision-based input**，减少对先验地图的依赖
- 研究更高效的 **LLM distillation** 或 **prompting strategies**，降低部署成本
- 将框架推广至其他 **mobile relay deployment** 场景（如 disaster recovery, rural connectivity）

--- 

> 📌 **总结一句话**：  
> 本论文开创性地将 **LLM 的语义推理能力** 与 **DRL 的实时决策能力** 相结合，提出了 **SA-DRL 框架** 与 **SA-PPO 算法**，在真实轨迹驱动的仿真中实现了**更高连通性、更低能耗、更快收敛、更强泛化**的 UAV 辅助 VANET 部署，为智能交通系统的空地一体化组网提供了新范式。

</details>

---

### 6. [Behavioral Fingerprints for LLM Endpoint Stability and Identity](https://arxiv.org/abs/2603.19022)

**Authors**: Jonah Leshin, Manish Shah, Ian Timmis, Daniel Kang  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.19022v1  

#### Abstract
The consistency of AI-native applications depends on the behavioral consistency of the model endpoints that power them. Traditional reliability metrics such as uptime, latency and throughput do not capture behavioral change, and an endpoint can remain "healthy" while its effective model identity cha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Behavioral Fingerprints for LLM Endpoint Stability and Identity 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对 **LLM 服务端点（endpoint）的行为不稳定性** 问题展开研究。传统可靠性指标（如 uptime、latency、throughput）无法捕捉模型输出行为的变化。即使接口“健康”，其背后的模型可能因权重更新、tokenizer 变更、量化策略调整、推理引擎升级或硬件变更而发生显著行为漂移（behavioral drift），从而破坏下游应用（如 agent 工作流、安全护栏、解析器）。

作者指出：“**Reliability is not stability**”——系统可用且快速，并不代表其行为一致。

### 提出了什么新方法或新思路
提出两个核心组件：

- **Stability Monitor**：一个黑盒（black-box）稳定性监控系统，通过周期性地对 LLM endpoint 进行“指纹采集”（fingerprinting），检测其输出分布是否发生变化。
- **Stability Arena**：一个可视化 Web 应用（https://arena.projectvail.com），发布并展示跨 provider 的稳定性数据。

#### 核心技术流程：
1. **Fingerprint 构造**：使用一组固定的 prompts，从 endpoint 多次采样响应，将响应嵌入为向量，形成 fingerprint。
2. **Pairwise 比较**：利用 **energy distance** 统计量衡量不同时间点 fingerprints 之间的分布差异。
3. **Change Detection**：结合 **permutation test** 得到 p-value，并通过基于 **e-values** 的序列证据累积机制实现连续监测与变化事件检测。
4. **Change Event 触发**：当累积证据超过阈值时，判定为 change event，重置 baseline fingerprint。

### 相比现有方法的优势
| 方面 | 本工作 | 现有方法（如 B3IT[3]） |
|------|--------|------------------------|
| **Prompt 设计** | 固定、通用、跨模型 prompt 集 | 依赖初始化阶段发现的“border inputs”，需重新探测 |
| **适用场景** | 支持持续监控、无需重新校准 | change 后边界可能失效，需重新学习 probe |
| **统计框架** | 使用 e-values 支持可选停止（optional stopping），适合流式检测 | 通常固定样本假设检验 |
| **访问模式** | 完全 black-box，仅需 API 调用 | 类似，但 probe 更具针对性 |
| **成本与频率** | 轻量级（每次 ~800 次 inference 请求），支持高频率监测 | 可能需要更多定制化调优 |

> ✅ **优势总结**：更轻量、更鲁棒、更适合生产环境中的长期、自动化、跨 provider 的行为一致性监控。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 并未使用标准 benchmark 数据集（如 MMLU、GSM8K），而是构建了一个 **固定 prompt set** 用于 fingerprint 生成。
- Prompts 是自然语言形式的问题/指令，覆盖多样主题和任务类型，具体未公开细节，但强调其 **model-agnostic** 特性。

### 实验设置
#### 控制实验（Controlled Validation）
- 在本地部署模型 endpoint，运行 Stability Monitor（每小时生成一次 fingerprint）。
- 手动引入五类明确干预（见 Table 1）：
  - Model family change（Qwen → Llama）
  - Version upgrade（Qwen2.5-0.5B → Qwen3-0.6B）
  - Inference stack change（vLLM → Transformers）
  - Quantization change（BF16 → INT8）
  - Temperature change（0.7 → 0.6）

> 每次只改变一个变量，其余保持不变。

#### 真实世界部署实验
- 对多个 provider 提供的同一名义模型（如 Kimi-K2-0905-Instruct）进行持续监控。
- 时间跨度：2025年11月至12月。
- 监控频率：通常每几小时一次。

### 评估指标
- **Change Event Detection Latency**：变化发生后多久被检测到。
- **False Positive Rate**：在无变化期间是否误报 change event。
- **Energy Distance**：作为分布偏移的度量。
- **Permutation Test p-value**：判断分布是否相同的统计证据。
- **Divergence Ratio**：某 provider 相对于其他 provider 群体行为的偏离程度（归一化于中位数距离）。

### 基线方法对比
文中未直接与其他 change detection 方法进行端到端性能比较（如准确率/F1），但讨论了与 **B3IT[3]** 的设计差异：

- B3IT 使用 low-probability border inputs 来放大微小变化，敏感但脆弱；
- 本文方法使用常规 prompts，牺牲部分灵敏度换取更强泛化性和免维护性。

此外，引用 Chen et al.[4] 表明 GPT-3.5/GPT-4 存在随时间的行为漂移，验证了问题的存在性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 变化类型 | 是否触发 change event | 检测延迟 |
|---------|------------------------|----------|
| Model family change (Qwen→Llama) | ✅ 是 | 下一个 fingerprint 即触发 |
| Version upgrade | ✅ 是 | 下一个 fingerprint 即触发 |
| Inference stack change (vLLM→Transformers) | ✅ 是 | 下一个 fingerprint 即触发 |
| Quantization change (BF16→INT8) | ✅ 是 | 下一个 fingerprint 即触发 |
| Temperature change (0.7→0.6) | ✅ 是（弱变化） | **18 个 fingerprint 后触发**（约半天） |

> ✔️ 所有重大变更均被成功捕获  
> ⚠️ 小幅度 temperature 调整检测较慢，说明对细微参数变化敏感度有限

### 与基线方法的对比结果
- **相比 B3IT**：
  - 不需要 per-endpoint 初始化或 probe 发现过程。
  - 更易于扩展至大规模多 endpoint 监控。
  - 对 inference stack、quantization 等非功能参数变化同样有效。
- **相比 domain-specific evals**（如 Margin Research 的 Claude tracker）：
  - 成本更低、频率更高（daily-bench 等通常是每日一次 heavy eval）。
  - 更关注“行为一致性”而非特定能力得分趋势。

### 消融实验结果（文中未提供）
论文未进行显式的消融实验（ablation study），例如：
- 不同 prompt sets 的影响
- 不同 embedding model 的选择
- energy distance vs MMD 的实际表现差异
- e-values 阈值设定的影响

但通过 controlled 实验间接验证了整体 pipeline 的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **LLM endpoints 即使“正常运行”，也可能经历显著行为变化**，这些变化源于模型本身以外的因素（inference engine、quantization、caching、routing、hardware）。
2. 📊 **不同 provider 提供的“相同模型”表现出显著行为差异**：
   - 如在 Kimi-K2 模型上，**Moonshot**（原厂）表现出 **100% 稳定性**；
   - 而 **DeepInfra** 几乎每次 fingerprint 都触发 change event，极不稳定。
3. 🛠️ **Stability Monitor 成功识别出真实世界中的变更事件**：
   - 例如 Parasail 在 2025 年 12 月的一次 change event 被确认为因物理节点故障导致的硬件提供商切换。
4. 🌐 **可通过 fingerprint heatmap 实现 provider identification**：
   - 各 provider 自身 fingerprints 最相似（对角线最小），可用于反向识别未知 endpoint 的来源 provider。
5. 📈 **提出了“vs the pack” divergence ratio 指标**，可量化单个 provider 相对于群体共识的偏离程度，有助于识别 outlier。

### 方法的局限性
- **基础设施引起的随机性难以区分**：
  - 即使没有主动变更，batch size、caching、load balancing 等可能导致输出波动，模糊“change event”与“固有不稳定性”的界限。
- **对微小参数变化不够敏感**：
  - 如 temperature 从 0.7→0.6 需要 18 次才检测到，可能漏检轻微 tuning。
- **embedding-based 方法依赖语义空间质量**：
  - 若 embedding model 不能很好捕捉行为差异（如格式、风格），可能降低检测能力。
- **未解决因果归因问题**：
  - 检测到 change event 后，无法自动判断是哪个因素（weights? tokenizer? kernel?）引起。

### 未来工作方向
- 探索更高效的 fingerprint 构造方式（减少 inference 请求次数）。
- 引入 interpretable diagnostics 工具，帮助定位 change 的根本原因（root cause analysis）。
- 扩展到多模态模型 endpoint 的稳定性监控。
- 结合 domain-specific evaluation signals，构建混合监控体系。
- 开放 fingerprint dataset 和 prompt set，促进社区复现与比较。

---

> 💡 **总体评价**：该论文提出了一个实用、可落地的 LLM endpoint 行为稳定性监控框架，填补了当前 AI 系统可观测性（observability）领域的重要空白。其核心思想——用 **behavioral fingerprint + distribution shift detection** 来保障模型身份一致性——具有广泛的应用前景，尤其适用于生产级 AI agent、合规审计、benchmark 公平性分析等场景。

</details>

---

### 7. [A Family of Adaptive Activation Functions for Mitigating Failure Modes in Physics-Informed Neural Networks](https://arxiv.org/abs/2603.18328)

**Authors**: Krishna Murari  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.18328v1  

#### Abstract
Physics-Informed Neural Networks(PINNs) are a powerful and flexible learning framework that has gained significant attention in recent years. It has demonstrated strong performance across a wide range of scientific and engineering problems. In parallel, wavelets have been extensively used as efficie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Family of Adaptive Activation Functions for Mitigating Failure Modes in Physics-Informed Neural Networks

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **Physics-Informed Neural Networks (PINNs)** 在求解偏微分方程（PDEs）时常见的**失败模式**（failure modes），如：
- 对**高频振荡、多尺度特征和强对流问题**建模能力差；
- 训练过程中梯度不平衡导致优化困难；
- 长时间演化下预测精度下降（“early-time accuracy, late-time failure”现象）。

这些问题使得传统激活函数（如 `tanh`）在复杂物理系统中表现不佳。

---

### 提出了什么新方法或新思路
作者提出了一类**基于小波（wavelet-based）的自适应激活函数家族**，将可训练的小波或类小波函数与 `tanh` 或 `softplus` 结合，构建新型激活函数。具体包括五种新设计：

| 激活函数 | 组成成分 |
|--------|--------|
| `SoftMexTanh` | Mexican hat wavelet + `tanh` + `softplus` 参数化 |
| `SoftMorTanh` | Morlet wavelet + `tanh` |
| `SoftGaussTanh` | Gaussian wavelet + `tanh` |
| `SoftGaborTanh` | Gabor wavelet + `tanh` |
| `SoftHerTanh` | Hermite wavelet + `tanh` |

所有参数均通过 `softplus` 映射为正数，并作为**可学习变量（trainable parameters）** 在训练中动态更新，实现真正的**自适应非线性表达**。

此外，还引入了其消融变体（ablation variants），记作 `Soft*X*TanhW`，其中 `tanh` 的缩放因子固定，仅小波部分参数可训练。

---

### 相比现有方法的优势
- ✅ **更强的表达能力**：小波固有的局部化和多频特性使其更适合捕捉PDE中的振荡和多尺度行为。
- ✅ **更高的训练稳定性**：结合 `softplus` 可避免负参数带来的数值不稳定。
- ✅ **端到端可训练**：所有激活参数参与反向传播，模型能自动调整激活形状以适应任务需求。
- ✅ **通用性强**：在多种类型PDE上验证有效，且无需改变网络架构。
- ✅ **优于Transformer等复杂结构**：相比 PINNsFormer、PINN-Mamba 和 ML-PINN，在准确性和效率之间取得更好平衡。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集（PDE测试用例）
研究选取四类具有挑战性的典型PDE作为基准问题，涵盖不同物理机制：

| 方程 | 类型 | 是否有解析解 |
|------|-----|-------------|
| 1D Reaction Equation | 非线性双曲型 | 是 |
| 1D Wave Equation | 双曲型 | 是 |
| 1D Convection Equation | 对流主导双曲型（高频率） | 是 |
| 2D Navier-Stokes Equations | 不可压缩流体动力学（非线性耦合系统） | 否（采用文献提供的数值解作为ground truth） |

这些方程被选为标准PINNs容易失败的案例，用于检验所提方法的有效性。

---

### 实验设置和评估指标

#### 网络结构与训练配置
- **网络架构**：前馈神经网络（MLP）
  - 隐藏层数：4
  - 隐藏层宽度：512
- **优化器**：L-BFGS（强Wolfe线搜索），迭代1000次
- **损失权重**：$ \lambda_R = \lambda_B = \lambda_I = 1 $
- **随机种子**：固定为5，确保可复现性
- **硬件平台**：单块NVIDIA A100 GPU，32GB内存

#### 评估指标
- **Relative MAE (rMAE)**：相对平均绝对误差
- **Relative RMSE (rRMSE)**：相对均方根误差
- **Training Loss**：总PINNs损失值（PDE残差 + 边界条件 + 初始条件）

$$
\text{rMAE} = \frac{\sum_n |\hat{u}(x_n,t_n) - u(x_n,t_n)|}{\sum_n |u(x_n,t_n)|}, \quad
\text{rRMSE} = \sqrt{ \frac{\sum_n (\hat{u} - u)^2 }{ \sum_n u^2 } }
$$

---

### 基线方法对比
- **Baseline PINN**：使用标准 `tanh` 激活函数 [Raissi et al., 2019]
- **其他深度学习模型**：
  - `PINNsFormer` [Zhao et al., 2024]：基于Transformer的PINN框架
  - `ML-PINN` [Gao et al., 2025]：融合Mamba-LSTM的记忆高效网络
  - `PINN-Mamba` [Xu et al., 2025]
  - `FLS`（First Layer Sine）[Wong et al., 2022]
  - `QRes`（Quadratic Residual Networks）[Bu & Karpatne, 2021]

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Tables 2–6）

#### 📊 1D Reaction Equation（Table 2）
| 方法 | rMAE ↓ | rRMSE ↓ | Loss ↓ |
|------|-------|--------|--------|
| Tanh [37] | 0.975 | — | 0.199 |
| **SoftGaborTanh** | **0.0021** | — | **0.0010** |
| SoftMexTanh | 0.023 | — | 0.011 |

> **提升显著**：`SoftGaborTanh` 的 rMAE 比标准 `tanh` 降低超过 **460倍**

---

#### 📊 1D Wave Equation（Table 3）
| 方法 | rRMSE ↓ | Loss ↓ |
|------|--------|--------|
| Tanh [37] | 0.223 | 0.214 |
| **SoftHer2Tanh** | **0.019** | **0.018** |
| SoftGaborTanh | 0.031 | 0.029 |

> 所有新激活函数均大幅优于基线，最小误差仅为原方法的 **~8.5%**

---

#### 📊 1D Convection Equation（Table 4）
此问题因高对流系数（β=50）导致剧烈振荡，标准PINN难以处理。

| 方法 | rRMSE ↓ | rMAE ↓ |
|------|--------|--------|
| Tanh [37] | 0.796 | 0.724 |
| **SoftGaborTanhW** | **0.0068** | **0.00606** |

> 使用**固定β版本**（即 `*TanhW`）反而效果更佳，说明某些场景下简化参数空间有助于稳定训练。

---

#### 📊 2D Navier-Stokes（Table 5 & 6）
无解析解，依赖数值参考解。

| 方法 | rRMSE ↓ | Loss ↓ |
|------|--------|--------|
| Tanh [37] | 12.35 | 1.101e-4 |
| **SoftGaborTanh** | **1.084** | **1.04e-6** |
| **SoftGaborTanhW** | **0.590** | 6.17e-6 |

> `SoftGaborTanhW` 在rRMSE上达到 **5.9倍提升**，表明固定`tanh`缩放有时更有利。

---

### 与先进模型的横向比较（Remark 6.2）
| 方法 | rRMSE (Navier-Stokes) | 训练速度 (it/s) | 内存消耗 |
|------|------------------------|------------------|----------|
| PINN-Tanh | 12.35 | 14.03 | 低 |
| **Proposed (SoftGaborTanh)** | **1.084** | **9.01** | 低 |
| PINNsFormer [57] | ~8.03 | 6.54 | 高 |
| ML-PINN [14] | ~8.03 | < PINN | 高 |

> 尽管训练稍慢于原始PINN，但远快于PINNsFormer，同时精度大幅提升。

---

### 消融实验结果
- **是否让 `tanh` 缩放参数可训练？**
  - 在大多数情况下（如Reaction、Wave），允许β可训练效果更好；
  - 但在**Convection和Navier-Stokes**中，固定β（即使用`*TanhW`）反而获得更低误差 → 表明**并非越多可调参数越好**，需根据问题调节灵活性。
- **不同小波的选择影响明显**：
  - `Gabor` 和 `Mexican hat` 在多数任务中表现最优；
  - `Hermite` 高阶项有助于捕捉复杂振荡模式。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **小波启发的激活函数能显著缓解PINNs的失败模式**，尤其在高频、多尺度、强对流等问题中表现出卓越鲁棒性。
2. ✅ **自适应参数机制有效**：通过将小波参数设为可学习量，模型可在训练中自动调整激活形态以匹配目标函数特征。
3. ✅ **简单改进胜过复杂架构**：提出的激活函数仅修改非线性层，不增加模型复杂度，却在多个指标上超越Transformer类（如PINNsFormer）和循环结构（如ML-PINN）。
4. ✅ **softplus约束保障训练稳定性**：防止参数发散，提升收敛可靠性。
5. ⚠️ **并非所有参数都应可训练**：在某些复杂系统中（如Navier-Stokes），限制部分参数反而提升泛化性能。

---

### 方法的局限性
- ❗ 当前方法仍基于MLP架构，未探索CNN/GNN等几何归纳偏置结构；
- ❗ 小波函数形式为预定义，尚未实现完全由数据驱动生成；
- ❗ 超参数初始化敏感（如Gabor的 $ \omega_0 $ 设置为3或5），缺乏自动化选择机制；
- ❗ 所有实验集中在中小规模PDE，尚未验证在超高维或大规模工业仿真中的扩展性。

---

### 未来工作方向
- 🔁 探索混合优化策略：结合Adam预热 + L-BFGS微调，进一步加速收敛；
- 💻 开发适用于CPU集群的大规模并行版本，推动实际科学计算部署；
- 🧠 引入元学习或NAS技术自动选择最佳激活组合；
- 🔄 将该思想推广至其他神经算子（Neural Operators）框架，如DeepONet、Fourier Neural Operator；
- 📈 延长训练epoch，研究长期时间外推能力（long-time integration）的表现。

---

## 总结
本文提出了一套**基于小波的可学习激活函数家族**，成功提升了PINNs在多种典型PDE上的建模能力和鲁棒性。实验证明，这种轻量级、模块化的改进方式在保持算法简洁的同时，实现了对主流先进模型的全面超越，是推动**Scientific Machine Learning**走向实用化的重要一步。

</details>

---

### 8. [Thinking with Constructions: A Benchmark and Policy Optimization for Visual-Text Interleaved Geometric Reasoning](https://arxiv.org/abs/2603.18662)

**Authors**: Haokun Zhao, Wanshi Xu, Haidong Yuan, Songjun Cao, Long Ma, Yanghua Xiao  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.18662v1  

#### Abstract
Geometric reasoning inherently requires "thinking with constructions" -- the dynamic manipulation of visual aids to bridge the gap between problem conditions and solutions. However, existing Multimodal Large Language Models (MLLMs) are largely confined to passive inference with static diagrams, lack...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Thinking with Constructions: A Benchmark and Policy Optimization for Visual-Text Interleaved Geometric Reasoning**

---

## **1. 主要贡献和创新点**

### **解决的问题**
现有的 **Multimodal Large Language Models (MLLMs)** 在几何推理任务中主要依赖静态图表进行被动推理，缺乏“主动构造”视觉辅助（如添加辅助线）的能力。这种能力是人类专家解决复杂几何问题的关键——即“**thinking with constructions**”。然而，当前模型无法有效决策何时、如何以及为何构造这些辅助元素。

具体挑战包括：
- 缺乏对**视觉-文本交错推理**（Visual-Text Interleaved Reasoning）的支持。
- 现有方法在生成精确的几何图示时存在**像素级幻觉**（pixel-level hallucinations）。
- 没有系统性的框架来学习**辅助构造的战略适用性**（strategic applicability）。

---

### **提出的新方法与新思路**

#### ✅ **1. GeoAux-Bench：首个对齐文本构造指令与真实视觉更新的基准**
- 包含 **4,334 个几何问题** 和 **8,470 张图表**。
- 明确将每个文本中的辅助构造步骤 `<aux>...</aux>` 与其对应的视觉更新 `Iaux` 进行配对。
- 支持细粒度监督，用于训练模型理解“构造”的动态过程。

#### ✅ **2. 视觉-文本交错思维链（Visual-Text Interleaved Chain-of-Thought）**
- 提出一种新的推理范式：模型在推理过程中交替使用文本指令和视觉反馈。
- 构造不仅是描述，更是可执行的认知操作，推动推理状态转移。

#### ✅ **3. Action Applicability Policy Optimization (A2PO)**
一种基于强化学习的策略优化框架，旨在教会模型**战略性地使用视觉辅助**：
- **Tri-Partition Sampling**：通过三种采样路径构建反事实对比：
  - **Mandatory (O+)**：强制插入辅助线；
  - **Prohibited (O−)**：禁止任何辅助构造；
  - **Natural (O)**：自由生成（目标策略）。
- **Adaptive Reward Shaping**：设计复合奖励函数，包含：
  - **Timing Reward (`rtime`)**：判断是否“有必要”画辅助线（仅当带来性能增益时才鼓励）；
  - **Quality Reward (`rqual`)**：衡量构造是否真正简化了推理路径（以降低 **Perplexity (PPL)** 为依据）。
- **Visual Re-prompting**：在检测到正确构造后，注入真实的 `Taux` 和 `Iaux`，实现高质量视觉反馈闭环。

---

### **相比现有方法的优势**

| 维度 | 现有方法 | 本文方法 |
|------|--------|---------|
| 推理模式 | 静态观察 | 动态交互式构造 |
| 辅助线处理 | 被动接受或忽略 | 主动决策“是否/如何”构造 |
| 学习机制 | 监督微调（SFT）为主 | 强化学习 + 自适应奖励塑形 |
| 视觉反馈质量 | 易产生结构幻觉 | 使用 ground-truth 图表注入（via retrieval）保证精度 |
| 性能提升来源 | 更强预训练 | 更优的推理控制策略 |

> 💡 **核心优势**：A2PO 不是简单地“多画图”，而是学会“聪明地画图”——只在必要且有效的时刻引入简洁、高信息量的视觉辅助。

---

## **2. 核心实验方法和设置**

### **使用的数据集**

| 数据集 | 描述 |
|-------|------|
| **GeoAux-Bench** | 本文提出的主基准，分为：<br>• **GeoAux-Core**：1,679 人工标注题，涵盖课程与奥赛难度；<br>• **GeoAux-Canvas**：2,655 扩展题，来自 MathCanvas-Bench 并重新标注。 |
| **GeomVerse** | 外部几何推理基准，用于跨域泛化验证。 |
| **Geometry3k** | 经典几何数据集，进一步测试通用性。 |

---

### **实验设置与评估指标**

#### 🔹 **模型架构**
- 主干模型：`Qwen2.5-VL-7B-Instruct` 及其变体。
- 视觉编码器冻结，仅微调语言部分。

#### 🔹 **训练流程**
1. **SFT Warm-up**：混合提示训练，包含允许/禁止辅助线的样本。
2. **A2PO 训练阶段**：采用 GRPO 框架，结合三路采样与自适应奖励。

#### 🔹 **评估指标**
| 指标 | 含义 |
|------|------|
| **Accuracy (%)** | 最终答案正确率（主要指标） |
| **Perplexity (PPL) ↓** | 推理过程的语言不确定性，越低表示逻辑更清晰 |
| **Token Count** | 生成长度，反映效率 |

#### 🔹 **基线方法对比**
| 基线 | 简介 |
|-----|------|
| **SFT** | 监督微调，基础起点 |
| **GRPO** | 分组相对策略优化，标准 RL 基线 |
| **ToRL** | 工具集成强化学习 |
| **GeometryZero** | 当前最优几何 RL 方法，基于 group contrastive learning |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 📊 表格：在多个数据集上的平均准确率（Table 3）

| 方法 | GeoAux (%) | Geometry3k (%) | Avg. Acc (%) | PPL ↓ |
|------|------------|----------------|---------------|--------|
| SFT | 23.09 | 39.40 | 39.56 | 1.1389 |
| GRPO | 31.22 | 50.72 | 47.01 | 1.1550 |
| ToRL | 28.68 | 47.06 | 44.71 | 1.1558 |
| GeometryZero | 29.33 | 52.72 | 46.35 | 1.1535 |
| **A2PO (Ours)** | **33.20** | **53.05** | **48.22** | **1.1534** |

> ✅ **A2PO 在 3B 模型上超越所有基线，平均提升达 3.51%**。

#### 📈 在 7B 模型上的表现更为显著：

| 方法 | GeoAux (%) | Geometry3k (%) | Avg. Acc (%) | PPL ↓ |
|------|------------|----------------|---------------|--------|
| GRPO | 39.77 | 53.50 | 52.92 | 1.0941 |
| GeometryZero | 40.18 | 53.72 | 54.07 | 1.0945 |
| **A2PO (Ours)** | **42.97** | **53.61** | **55.76** | **1.0869** |

> ✅ **A2PO 实现最高准确率的同时保持最低 PPL**，说明其推理路径更简洁、确定性强。

---

### **消融实验结果（Ablation Study）**

| 方法 | Acc (%) | 说明 |
|------|--------|------|
| GRPO | 39.28 | 基线 |
| w/o Length Reward | 39.52 | 移除长度偏好 → 更精简推理 |
| + Timing Reward (TR) | 40.18 | 学会“何时该画” → 超越 ToRL |
| + Quality Reward (QR) | 41.17 | 学会“画得好” → 减少冗余构造 |
| + Visual Re-prompting | **42.97** | 注入真实图像 → 最大跃升（+1.8%） |

> 🔍 **关键发现**：
> - 文本描述不足以替代实际视觉呈现（`Visual Re-prompting` 贡献最大）；
> - “好”的构造应降低 PPL，而非仅仅语法合法；
> - 战略性决策（timing + quality）比盲目鼓励构造更有效。

---

## **4. 关键结论和发现**

### **主要发现**

1. ✅ **交错模态优于单模态**  
   - 实验表明，同时提供 `<aux>` 文本指令和对应 `Iaux` 图像，比单独任一模态高出最多 **1.97%**。
   - 单靠文字无法完全传递空间关系，单靠图像易因感知模糊导致失败。

2. ✅ **有效构造是“熵减器”（Entropy Reducer）**  
   - 正确的辅助线显著降低推理 **Perplexity (PPL)**，模拟人类“顿悟”时刻。
   - PPL 与 Accuracy 呈强负相关，可用于指导奖励设计。

3. ✅ **视觉显著性（Visual Saliency）至关重要**  
   - 将辅助线加粗/染红后，准确率提升高达 **+1.10%**。
   - 表明当前模型对细微视觉变化敏感，需增强特征可检测性。

4. ✅ **分析捷径（Analytic Shortcut）普遍存在**  
   - 多数 MLLMs 倾向于建立坐标系暴力求解，而非运用纯几何定理。
   - 导致在 junior 几何题上表现差，在 senior 题上反而更好（见 Table 2）。

5. ✅ **记忆化现象严重**  
   - 如 `Qwen3-VL-Thk` 在 Olympiad 题上得分极高但在基础题上崩盘，暗示可能 memorize 了特定竞赛题。

---

### **局限性**

1. ❌ **依赖检索式视觉注入（Retrieval-based Injection）**  
   - 当前 `Visual Re-prompting` 使用的是 **ground-truth diagram**，非模型自主生成。
   - 尚未实现端到端的“画图-看图-再推理”闭环。

2. ❌ **统一多模态模型仍不成熟**  
   - 原生支持图文交错生成的模型（如 MathCanvas-7B）表现最差（<13%），因其常出现**视觉逻辑错配**（Visual-Logic Mismatch），如曲线代替直线、交点丢失等。

3. ❌ **推理延迟高**  
   - 自回归图像生成耗时长，阻碍 RL 中的大规模探索。

---

### **未来工作方向**

1. 🔮 开发具备**精细几何编辑能力**的生成模型，支持原子级操作（如“连接两点”、“作垂线”）。
2. 🔁 构建真正的**动态采样-反馈循环**，让模型能自我生成、感知并修正其构造。
3. 🧠 探索**几何概念与像素操作之间的对齐机制**，打通高层语义与底层视觉。
4. 🤖 结合 agent 架构，赋予模型“工具调用”能力，调用 CAD-style 工具进行精确绘图。

---

> 🎯 **总结一句话**：  
> 本论文首次系统性地将“**thinking with constructions**”形式化为一个可学习的策略问题，提出了 **GeoAux-Bench** 和 **A2PO** 框架，证明了**战略性视觉辅助构造**不仅能提分，更能使推理过程更清晰、更接近人类认知本质。

</details>

---

### 9. [D5P4: Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding](https://arxiv.org/abs/2603.19146)

**Authors**: Jonathan Lys, Vincent Gripon, Bastien Pasdeloup, Axel Marmoret, Lukas Mauch, Fabien Cardinaux, Ghouthi Boukli Hacene  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.19146v1  

#### Abstract
Discrete diffusion models are promising alternatives to autoregressive approaches for text generation, yet their decoding methods remain under-studied. Standard decoding methods for autoregressive models, such as beam search, do not directly apply to iterative denoising, and existing diffusion decod...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：D5P4: Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **离散扩散模型（Discrete Diffusion Models）** 在文本生成任务中展现出潜力，但其解码机制仍不成熟。
- 现有方法如 **beam search** 不适用于迭代去噪过程，且缺乏对 **in-batch diversity** 的有效控制。
- 强制引导（Classifier-Free Guidance, CFG）虽提升保真度，却导致 **mode collapse** 和输出多样性下降。

### 🚀 提出的新方法：D5P4
- 提出 **D5P4**（Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding），一种专为离散扩散模型设计的并行束搜索框架。
- 将每一步候选选择建模为 **Determinantal Point Process (DPP)** 上的最大后验（MAP）推断问题，实现质量与多样性的显式权衡。
- 引入 **partition constraint** 防止“lineage collapse”——即所有候选序列退化为单一祖先路径的现象。

### 🔍 创新点与优势
| 特性 | 描述 |
|------|------|
| **Set-level selection** | 不再独立评分候选序列，而是基于集合进行联合选择，建模假设间的交互关系。 |
| **DPP + Partition Constraint** | 结合 DPP 的多样性偏好与分组约束，确保每个父节点至少保留一个子代，防止多样性过早崩溃。 |
| **高效贪婪求解器** | 使用可扩展的 **greedy MAP solver**，在 GPU 上并行执行，计算开销几乎为零，保持多 GPU 兼容性。 |
| **无需外部评分器** | 所需信号（entropy 表示质量，hidden states 表示语义相似性）均来自扩散模型自身，避免依赖 GPT-2 等外部 evaluator。 |

> 💡 **相比现有方法的优势**：
> - 相比传统 beam search 和 temperature scaling，D5P4 提供更平滑、可控的质量-多样性权衡；
> - 相比 diverse beam search（如 Transversal MMR），D5P4 显式建模成对排斥，效果更强；
> - 相比标准 DPP sampling，引入 partition constraint 更适合 beam 结构，防止祖先路径单一化。

---

## 2. 核心实验方法和设置

### 📚 数据集
| 任务 | 数据集 |
|------|--------|
| **Open-ended Generation** | FineWeb（用于 MDLM） |
| **Question Answering** | TruthfulQA、CommonSenseQA（用于 LLaDA） |

### ⚙️ 实验设置
- **模型**：使用两种主流离散扩散语言模型
  - **MDLM**（Masked Diffusion Language Model）
  - **LLADA**（Large Language Diffusion Model）
- **解码配置**：
  - 并行生成 `k × w` 个候选（例如 8 组 × 4 分支）
  - 每步通过 D5P4 进行 selection，保留 `k` 个 beam
- **硬件兼容性**：支持跨 GPU 分布式执行，利用并行性维持高吞吐

### 📊 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **Quality** | PPL（Perplexity）、MAUVE、BLEU、F1-score、Wasserstein Distance | 衡量生成质量和与参考文本的分布对齐程度 |
| **Diversity** | COS（Jina embedding cosine similarity）、Self-BLEU、Distinct-n、EAD | 衡量语义和词汇层面的多样性 |
| **Alignment** | CKA（Centered Kernel Alignment） | 衡量内部表示与外部模型（如 GPT-2）之间的对齐程度 |

### 🆚 基线方法对比
| 类型 | 方法 | 说明 |
|------|------|------|
| **Quality-centric** | Best-of-n（oversample）、Standard Beam Search | 基于得分选择最优 k 个样本或每组取最高分 |
| **Diversity-promoting** | Diverse Beam Search (Transversal MMR)、Standard DPP Sampling（无 partition） | 引入多样性惩罚或纯 DPP 抽样，但后者无法保证跨组选择 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ 开放式生成（Open-ended Generation）
- **Pareto Front 对比（图2）**：
  - D5P4（尤其是 multiplicative variant D5P4×）在低 PPL 下实现了更低的 cosine similarity（更高多样性）。
  - 温度调节（CAT）在多样性增加时出现“突变式崩溃”，而 D5P4 衰退更平稳。
  - D5P4+ 在高多样性区域表现最佳，延迟了 PPL 的爆炸性增长。

| 方法 | 参数 | PPL 相关性 | COS 相关性 |
|------|------|------------|------------|
| CAT | log temp | 0.940 | -0.438 |
| DivBS | log α_div | 0.950 | -0.846 |
| D5P4+ | log β | 0.902 | -0.875 |
| D5P4× | log β | 0.959 | -0.628 |

> 👉 表明 D5P4 对 diversity 控制更敏感，能实现更强的 trade-off。

#### ✅ 问答任务（Question Answering）
- **对抗 CFG 导致的多样性坍塌**（图4）：
  - 随着 CFG 强度上升，baseline 方法（independent sampling）的 EAD（期望调整后的 distinct）显著下降。
  - **D5P4 即使在高 CFG 下仍维持较高 EAD 和 distinct-n 值**，有效缓解 mode collapse。
  - 同时保持相近甚至更好的 F1、BLEU 和 PPL。

| 方法 | Setting | Perplexity ↓ | F1-score ↑ | EAD ↑ | Avg COS ↓ |
|------|---------|-------------|------------|-------|-----------|
| Best-of-k | TruthfulQA | 17.446 | 0.212 | 0.363 | 0.963 |
| D5P4+ | TruthfulQA | **15.725** | 0.184 | **0.385** | **0.946** |
| D5P4+ (w/ P-CFG) | TruthfulQA | 15.015 | 0.195 | **0.389** | **0.918** |

> ✅ D5P4 在降低 PPL 的同时提升了多样性指标。

### 🔍 消融实验结果

#### （1）多样性估计方式对比（表4）
| Pooling 方法 | MDLM CKA | LLADA CKA |
|-------------|----------|----------|
| Mean | 0.777 | 0.482 |
| Non-masked | 0.660 | 0.536 |
| Masked | 0.710 | 0.435 |
| **Flatten（本文采用）** | **0.821** | **0.667** |

> ✔️ **Flatten pooling** 在表示对齐方面表现最优，验证了设计合理性。

#### （2）质量评分函数对比（图9）
| 方法 | 与 GPT-2 PPL 的相关性 |
|------|------------------------|
| Entropy | **-0.776** |
| Self-certainty | -0.290 |

> ✔️ **Entropy-based scoring** 与外部质量信号更强负相关，更适合用作质量代理。

#### （3）选择算法效率与性能（表6）
| 方法 | 子行列式值（↑） | 时间（秒）↓ |
|------|------------------|------------|
| Random | -0.9074 | 0.0001 |
| DPP（CPU） | -0.8752 | 0.5478 |
| Diverse Beam Search | 0.6645 | 0.0295 |
| **Greedy MAP（ours）** | **1.0214** | **0.0023** |

> ⚡ 我们的方法不仅目标值最高，而且速度比 Diverse Beam Search 快 **10倍以上**，接近随机采样开销。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **D5P4 实现了当前最优的质量-多样性帕累托前沿**，尤其在高多样性需求场景下优于所有基线。
2. **无需外部评分器即可可靠驱动 DPP 内核**：扩散模型自身的 entropy 和 hidden states 已足够反映质量和多样性。
3. **partition constraint 是防止 lineage collapse 的关键机制**，保障了解码过程中多个祖先路径的持续存在。
4. **greedy MAP solver 极具效率**，可在多 GPU 环境中以极低开销运行，不影响并行解码的整体吞吐。

### ⚠️ 局限性
- 当前方法依赖于固定的 `k` 大小 beam，未探索动态 beam 数量。
- DPP kernel 设计较为通用，尚未针对特定任务（如推理链生成）定制语义距离度量。
- 虽然避免了外部 scorer，但在某些安全敏感任务中仍需结合 alignment filtering 使用。

### 🔮 未来工作方向
- 探索 **task-aware kernel design**，例如在数学推理中引入逻辑一致性度量。
- 将 D5P4 扩展至 **continuous diffusion models** 或其他模态（图像、音频）。
- 结合 **test-time compute scaling** 思路，构建“generate-many, select-smartly”范式，进一步提升覆盖率。
- 研究如何将 D5P4 与 **self-correction** 或 **verification modules** 联动，实现高质量+高多样性+高安全性三重目标。

---

## 总结一句话
> **D5P4 通过将 beam selection 建模为带 partition 约束的 DPP MAP 推断，在几乎零额外开销下实现了对离散扩散模型生成多样性的精细控制，显著优于现有解码策略。**

</details>

---

### 10. [AutoScreen-FW: An LLM-based Framework for Resume Screening](https://arxiv.org/abs/2603.18390)

**Authors**: Zhelin Xu, Shuhei Yamamoto, Atsuyuki Morishima  
**Category**: cs.CL  
**Published**: 2026-03-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.18390v1  

#### Abstract
Corporate recruiters often need to screen many resumes within a limited time, which increases their burden and may cause suitable candidates to be overlooked. To address these challenges, prior work has explored LLM-based automated resume screening. However, some methods rely on commercial LLMs, whi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AutoScreen-FW: An LLM-based Framework for Resume Screening 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
企业招聘中，HR 需在有限时间内筛选大量简历，导致工作负担重、易遗漏优秀候选人。传统 AI 简历筛选方法存在以下问题：
- 依赖商业 **LLM**（如 GPT），存在**数据隐私风险**；
- 缺乏公开标注的简历数据集，难以确定有效的 **in-context learning** 示例；
- 现有方法多关注“简历-职位描述匹配”，不适用于日本等国家流行的“潜力导向型招聘”（potential-based hiring），即通过开放性问题评估成长潜力、组织适配度等。

### 提出了什么新方法或新思路
提出 **AutoScreen-FW** —— 一个基于开源 LLM 的本地化、自动化简历筛选框架，其核心设计包括：
- **Sample Selection**：采用三种策略从简历集中选取代表性样本用于 **in-context learning**：
  - **Diversity-based**：最大化样本多样性；
  - **Similarity-based**：选择最接近整体分布的典型样本；
  - **Clustering-based**（本文提出）：基于聚类中心选择最具代表性的样本。
- **LLM Persona 设计**：定义 LLM 角色为“日本应届生招聘简历评估专家”，增强其领域判断能力。
- **Evaluation Criteria 显式建模**：结合日本求职文化，制定涵盖 **Content、Structure、Language** 三个维度的细粒度评分标准。
- 支持 **few-shot in-context learning**，无需微调即可适配不同公司评价偏好。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **隐私安全** | 使用本地部署的开源 LLM（如 Qwen、Llama），避免敏感信息外泄 |
| **适用场景** | 专为日本潜力导向型招聘设计，能有效评估开放性问题的回答质量 |
| **灵活性** | 可灵活调整提示中的样本、标准和角色，适应不同企业需求 |
| **效率** | 推理速度显著快于商业 GPT 模型 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- 数据来源：日本求职支持平台 **One Career** 上公开的 1,655 份简历。
- 数据预处理：
  - 仅保留 “Resume Content” 和 “Applied Position” 字段；
  - 过滤长度 ≤100 字符的子项（视为非实质性内容）；
  - 匿名化文件名中的公司名称。

### 实验设置和评估指标
#### 评估任务
对每份简历进行二分类判断：“This resume is of high quality.” 或 “This resume is of low quality.”

#### Ground Truth 构建
由于缺乏真实 HR 判定标签，采用三个高性能 GPT 模型生成判定作为 **ground truth**：
- GPT-5.2-2025-12-11
- GPT-5.1-2025-11-13
- GPT-o3-2025-04-16  
每个模型的输出被视为一种独立的 ground truth，以模拟现实中不同招聘官的主观差异。

#### 评估指标
- **Accuracy**：预测结果与 ground truth 的一致率。
- **Per-resume judgment time**：单份简历的平均推理耗时。

#### 消融变量
在 AutoScreen-FW 中测试多种组合：
- **Sampling Strategy**：diversity / similarity / clustering
- **Number of Shots**：3, 5, 10, 15, 20
- **Sample Type**：仅高质量 / 高+低质量混合（低质占30%）
- **Attribute Type**：overall judgment / dimension scores（content, structure, language）

### 基线方法对比
| 模型 | 类型 | 是否使用 in-context learning |
|------|------|----------------------------|
| GPT-5-mini | 商业闭源 LLM | Zero-shot |
| GPT-5-nano | 商业闭源 LLM | Zero-shot |
| Qwen3-8B | 开源 LLM | Zero-shot + Few-shot（AutoScreen-FW） |
| Llama-3.1-8B-Instruct | 开源 LLM | Zero-shot + Few-shot（AutoScreen-FW） |

所有模型使用相同 prompt（含 persona、evaluation criteria、instruction），确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（取各模型最优配置下的 Accuracy）

| Model | GPT-5.2 GT | GPT-5.1 GT | GPT-o3 GT |
|-------|------------|-----------|----------|
| GPT-5-mini | 0.6917 | **0.8445** | 0.7387 |
| GPT-5-nano | 0.6091 | 0.8374 | 0.6617 |
| Qwen3-8B (zero-shot) | 0.6063 | 0.8552 | 0.6744 |
| Qwen3-8B (**few-shot**) | **0.7218** | **0.8682** | **0.7218** |
| Llama-3.1-8B (few-shot) | 0.6101 | 0.8485 | 0.6333 |

> ✅ 注：最佳结果来自 AutoScreen-FW 下的 few-shot 设置。

### 与基线方法的对比结果
- **vs GPT-5-nano**：
  - Qwen3-8B 在所有三种 ground truth 下均**显著超越** GPT-5-nano，最高提升达 **10.8%**。
- **vs GPT-5-mini**：
  - 当以 GPT-5.1 为 ground truth 时，Qwen3-8B 达到 **0.8682**，**超过 GPT-5-mini (0.8445)**，增益 **+2.8%**；
  - 在其他两种设置下略低于 GPT-5-mini，但差距很小（<1%）。
- **零样本 vs 少样本**：
  - 引入 in-context learning 后，Qwen3-8B 最高提升 **11.3%**（GPT-5.2 场景），表明 sample selection 有效提升了判断一致性。

### 效率对比（见 Table II）
| Model | Time per Resume (s) |
|--------|---------------------|
| GPT-5-mini | 5.09 ± 0.22 |
| GPT-5-nano | 7.48 ± 0.45 |
| **Qwen3-8B (few-shot)** | **3.84 ± 0.26** |
| **Llama-3.1-8B (few-shot)** | **2.10 ± 0.24** |

- Qwen3-8B 比 GPT-5-mini 快 **24.6%**，比 GPT-5-nano 快 **48.7%**；
- 结合文献中 HR 平均审阅时间为 **7.4 秒**，使用 Qwen 可减少高达 **48%** 的人工时间。

### 消融实验结果
- **Sampling Strategy 影响**：
  - **Llama** 更适合使用多样性强的样本（如 diversity/clustering-based）；
  - **Qwen** 表现更依赖 ground truth 特性，需针对性选择样本策略。
- **Sample Type**：
  - 混合使用高/低质量样本能缓解“锚定效应”，提高泛化能力。
- **Attribute Type**：
  - 提供维度得分（dimension scores）有助于 LLM 更精细地校准评分行为。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **开源 LLM 完全可以在简历筛选任务上媲美甚至超越商业 GPT 模型**，尤其是在合理设计 in-context learning 的前提下。
2. ✅ **AutoScreen-FW 能有效提升开源 LLM 的判断准确性与稳定性**，且对不同 ground truth 具有良好适应性。
3. ✅ **few-shot in-context learning 显著改善了维度评分的一致性**，使输出更具解释性，可为 HR 提供细粒度反馈。
4. ⚡ **推理效率远高于商业模型**，适合大规模部署于企业内部系统。
5. 🔄 **不同 LLM 对 sample selection 策略敏感度不同**，说明需根据模型特性定制提示工程。

### 方法的局限性
1. ❗ **最优参数需手动调优**：当前框架无法自动识别最佳 sampling strategy、shot 数等配置，当评价标准变化时需重新调试。
2. ❗ **Ground Truth 来自 LLM 而非真实 HR**：尽管使用多个 GPT 模拟主观差异，但仍可能偏离实际企业招聘标准。
3. ❗ **未验证跨文化迁移能力**：目前实验聚焦日本求职场景，是否适用于欧美等经验导向型招聘尚待研究。

### 未来工作方向
1. 🔧 **开发自动化 sample selector 模块**：基于目标 ground truth 自动推荐最优 sample selection 策略与示例集合。
2. 👥 **开展真实企业合作验证**：收集真实 HR 的筛选结果，评估 AutoScreen-FW 与人类判断的一致性。
3. 🌐 **扩展至多语言、多文化场景**：探索该框架在其他国家招聘体系中的适用性。
4. 🤖 **集成 feedback loop 机制**：允许 HR 对 LLM 输出进行修正，并用于动态优化后续判断。

--- 

> **总结一句话**：  
> AutoScreen-FW 成功构建了一个**高效、安全、可解释**的本地化简历筛选框架，在保持高性能的同时大幅降低推理成本，展现出在企业级招聘自动化中的巨大应用潜力。

</details>

---

### 11. [From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning](https://arxiv.org/abs/2603.18383)

**Authors**: Grant Wilkins, Fiodar Kazhamiaka, Ram Rajagopal  
**Category**: cs.DC  
**Published**: 2026-03-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.18383v1  

#### Abstract
Datacenter operators and electrical utilities rely on power traces at different spatiotemporal scales. Operators use fine-grained traces for provisioning, facility management, and scheduling, while utilities use site-level load profiles for capacity and interconnection planning. Existing datacenter ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大型语言模型（LLM）推理工作负载具有高度动态的功耗特性，在 **prefill**（计算密集）、**decode**（内存密集）和 **idle** 状态之间快速切换。这种动态性导致数据中心设施级电力需求在亚秒级时间尺度上剧烈波动。

然而，现有的数据中心功耗建模方法存在以下不足：
- **基于 TDP（Thermal Design Power）的静态假设**：高估能耗，过于保守，无法支持精细化容量规划。
- **基于相位的查找表（LUT-based）方法**：如 Splitwise，将功耗离散为几个固定阶段，忽略了连续批处理（continuous batching）下混合状态的中间功耗水平。
- **回放式（replay-based）方法**：仅能复现已观测到的流量模式，无法泛化到新的请求到达过程、模型配置或部署规模。

因此，缺乏一种能够**跨不同流量条件、硬件平台和服务配置进行泛化**，并生成**从 GPU 到站点（site）多粒度功耗轨迹**的方法。

---

### 🚀 提出的新方法与创新思路
作者提出了一种**组合式（compositional）功耗轨迹生成框架**，其核心思想是将 LLM 推理功耗分解为两个独立但可组合的部分：

1. **工作负载驱动的状态转移**  
   使用请求调度特征（如并发请求数 $A_t$ 和其变化量 $\Delta A_t$）来预测系统在不同运行状态间的转移。

2. **配置相关的状态内功耗分布**  
   每个状态下，功耗由特定于硬件、模型和并行策略（如 tensor parallelism）的概率分布建模。

该框架通过三个阶段实现：
- **离线训练**：从实测轨迹中学习状态分类器（BiGRU）和状态条件下的功耗模型（Gaussian Mixture Model + AR(1) for MoE）。
- **轨迹合成**：给定新的请求到达流，先用轻量级吞吐代理（throughput surrogate）估算 $A_t$ 和 $\Delta A_t$，再通过 BiGRU 预测状态序列，最后采样生成服务器级功耗轨迹。
- **层级聚合**：将服务器级轨迹按机架（rack）→ 行（row）→ 数据中心（site）逐层聚合，生成适用于电网规划的设施级负载曲线。

---

### ⚖️ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **泛化能力** | 可推广至未见过的请求到达率、模型大小、GPU 类型和并行配置，无需重新测量整个端到端场景。 |
| **保真度** | 更好地保留了功耗的时间自相关结构（temporal autocorrelation），避免人工跳跃或平滑失真。 |
| **实用性** | 支持下游分析任务，如电力超售（oversubscription）、功率调制、电网互连研究等。 |
| **灵活性** | 开源实现允许外部使用，同时保护内部服务细节不被泄露，便于数据中心与电网公司协作。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **硬件平台**：Microsoft Azure 的 NVIDIA DGX 服务器，配备 A100 (80GB) 和 H100 (80GB) GPU。
- **模型范围**：
  - **Dense 模型**：Llama-3.1 系列（8B, 70B, 405B）、DeepSeek-R1-Distill（8B, 70B）
  - **MoE 模型**：gpt-oss（20B, 120B）
- **服务引擎**：vLLM（version 0.10.0），启用 continuous batching。
- **请求数据集**：ShareGPT、InstructCoder、AIMO-Validation-AIME、Edit-10K-Char。
- **真实流量验证**：使用 Microsoft 发布的一天 Azure LLM 推理请求日志（2024年5月16日）进行站点级模拟。

---

### 🔧 实验设置
- **采样频率**：250ms（足够捕捉 prefill/decode 转换）。
- **请求速率**：7 个等级（0.125 ~ 4 req/s），每种重复 5 次，总约 10 分钟轨迹。
- **训练/验证/测试划分**：按轨迹级别 70%/15%/15% 划分，跨不同到达率混合。

---

### 🎯 评估指标
| 指标 | 含义 |
|------|------|
| **ACF R²** | 合成与实测轨迹的自相关函数拟合优度，衡量时间结构保真度 |
| **ΔEnergy (%)** | 总能量误差（相对误差），中位数报告 |
| **KS Statistic** | Kolmogorov-Smirnov 检验值，衡量功率分布相似性 |
| **NRMSE** | 归一化均方根误差，点对点误差标准化到功率范围 |

---

### 🆚 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **TDP (Nameplate)** | 所有服务器始终以满额定功率运行 |
| **Mean Power** | 所有服务器恒定输出训练集平均功耗 |
| **LUT-based (Splitwise-inspired)** | 基于 batch token 数量查表决定 prefill/decode/mixed/idle 功耗比例 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）

| Model | KS ↓ | ACF R² ↑ | NRMSE ↓ | ΔEnergy (%) ↓ |
|-------|------|----------|---------|----------------|
| Llama-3.1(8B) | 0.18±0.04 | 1.00 | 0.33±0.03 | **0.6±0.3** |
| Llama-3.1(70B) | 0.22±0.09 | 0.97 | 0.43±0.09 | **4.0±2.7** |
| DeepSeek-R1-Distill (8B) | 0.21±0.08 | 0.99 | 0.32±0.05 | **1.0±0.4** |
| gpt-oss(120B) | 0.51±0.11 | 0.58 | 0.33±0.04 | **10.8±3.5** |

> ✅ **结论**：对于大多数 dense 模型配置，**中位绝对能量误差 < 5%**，且 ACF R² > 0.96，表明时间结构高度一致；MoE 模型因专家路由引入额外变异性，性能略低但仍优于基线。

---

### 🔍 与基线方法对比（Table 2 & Figure 8）

#### 服务器级对比（Llama-3.1-70B, A100, TP=4/8）
| Method | KS ↓ | ACF R² ↑ | NRMSE ↓ | ΔE (%) ↓ |
|--------|------|----------|---------|-----------|
| TDP | 1.00 | – | 1.66 | **243.6** |
| Mean | 0.69 | – | 0.32 | 17.35 |
| LUT-based | 0.64 | 0.56 | 0.27 | 13.71 |
| **Ours** | **0.12** | **0.99** | 0.27 | **6.09** |

> ✅ 我们的方法显著优于所有基线，尤其在能量误差和时间结构保持方面。

---

#### 设施级案例研究（240 台服务器，PUE=1.3，Azure 流量驱动）
| Metric | TDP | Mean | LUT-Based | **Ours** |
|--------|-----|------|------------|---------|
| Peak facility power (MW) | 1.19 | 0.85 | 0.82 | **0.75** |
| Avg. facility power (MW) | 1.19 | 0.85 | 0.76 | **0.63** |
| Peak-to-average ratio | 1.00 | 1.00 | 1.09 | **1.19** |
| Max ramp rate (MW/15min) | 0.00 | 0.00 | 0.07 | **0.11** |

> ✅ **TDP 过高估计容量需求达 60%**；我们的方法揭示了真实的峰值、波动性和爬坡能力，这对电网规划至关重要。

---

#### 超售（Oversubscription）分析
- 假设单行配电限制为 600 kW。
- TDP 方案最多部署 **23 个机架**（保守）。
- 使用我们生成的轨迹，在相同安全边界下可部署 **57 个机架**（超过两倍密度）。
- LUT-based 和 Mean 分别达到 52 和 42 个机架。

> ✅ 显示了**静态假设浪费大量可用电力头寸（headroom）**，而动态建模可释放潜力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 推理功耗具有强结构性**：主要由并发请求数 $A_t$ 和是否进入 prefill（$\Delta A_t > 0$）决定。
2. **组合式建模有效分离关注点**：工作负载动态 vs. 配置依赖功耗，提升泛化性和可迁移性。
3. **高保真轨迹改变基础设施决策**：
   - TDP 导致过度 provisioning；
   - 平均功耗忽略瞬态行为；
   - 我们的模型揭示了真实峰值、爬坡速率和超售空间。
4. **聚合效应平滑短时波动**：随着从 server → rack → row → site 聚合，变异系数（CV）从 0.583 降至 0.127，解释了为何局部峰值不会直接转化为站点级危机。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **MoE 模型建模不足** | 专家路由导致的状态内变化无法完全由 $A_t$ 和 $\Delta A_t$ 捕获，影响 AR(1) 模型精度。 |
| **恒定 PUE 与非 GPU 负载** | 忽略冷却系统随温度变化的影响，适用于规划但不适合实时控制。 |
| **到达过程假设** | 在 Poisson 和 Azure 日志上验证，但极端相关或突发流量可能挑战模型鲁棒性。 |
| **调度器抽象** | 使用 FIFO + 吞吐代理近似，未建模复杂调度策略（如抢占、优先级）。 |

---

### 🔮 未来工作方向
1. **扩展至 agentive workloads**：支持工具调用、多轮推理等自相关请求流。
2. **支持异构混合部署**：在同一数据中心内混合多种模型、硬件和 SLO。
3. **集成负载灵活性分析**：量化延迟容忍任务的削峰填谷潜力，支持电网互动（grid-interactive）。
4. **增强 MoE 建模**：引入专家激活模式作为额外特征，改进状态内变异建模。
5. **长期年度负载预测**：结合业务增长预测，生成全年负载形状用于资源充足性研究。

---

> 💡 **总体评价**：本文提供了一个**实用、开放、可扩展的框架**，填补了 LLM 推理建模与数据中心/电网规划之间的鸿沟，推动了 AI 基础设施向更高效、更可持续的方向发展。代码与模型已开源（GitHub: `grantwilkins/powertrace-sim`），具备强应用前景。

</details>

---

### 12. [Path-Constrained Mixture-of-Experts](https://arxiv.org/abs/2603.18297)

**Authors**: Zijin Gu, Tatiana Likhomanenko, Vimal Thilak, Jason Ramapuram, Navdeep Jaitly  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.18297v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) architectures enable efficient scaling by activating only a subset of parameters for each input. However, conventional MoE routing selects each layer's experts independently, creating N^L possible expert paths -- for N experts across L layers. This far exceeds typical...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Path-Constrained Mixture-of-Experts》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统的 **Mixture-of-Experts (MoE)** 架构在每一层独立进行 **routing** 决策，导致一个输入 token 在 $L$ 层网络中可能经历 $N^L$ 条不同的专家路径（expert paths），其中 $N$ 是每层的专家数。这种组合爆炸带来了严重的**统计低效性**（statistical inefficiency）：

- 实际训练数据量远小于可能的路径数量（例如，24层、16专家 → ~$10^{29}$ 路径，而典型训练仅约 $10^{11}$ tokens）；
- 大多数路径从未被充分学习，模型难以建立有意义的跨层结构。

此外，传统 MoE 需要引入 **auxiliary load balancing loss** 来防止专家负载不均，增加了超参数调优负担。

---

### **提出了什么新方法或新思路**
作者提出 **PathMoE**，一种通过**块状共享路由器参数**来约束专家路径空间的新架构设计：

- 将连续的 MoE 层划分为若干 **block**（如每 4 层为一个 block）；
- 同一 block 内的所有层**共享相同的 router 参数**（即 $W_b$），但仍然允许每个 token 在每层选择不同专家；
- 这种设计鼓励 token 在 block 内遵循**一致且连贯的路径**，从而自然地减少有效路径空间。

该方法处于“完全独立路由”与“全网络共享决策”的中间地带，既保持灵活性，又增强路径一致性。

---

### **相比现有方法的优势**
- ✅ **更高的样本效率**：路径更集中，相同路径获得更强的学习信号；
- ✅ **无需辅助负载均衡损失**（no auxiliary load balancing loss needed），简化训练；
- ✅ **更强的跨层协调性**（cross-layer consistency）和专家专业化；
- ✅ **更好的鲁棒性**：对 routing 扰动更具抵抗力；
- ✅ **性能提升**：在语言建模和下游任务上全面优于 baseline。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **0.9B 模型**：在 **FineWeb-100B** 数据集上训练 400k 步；
- **16B 模型**：在 **DCLM-Pro** 数据集上训练 200k 步；
- 使用 **Llama 2 tokenizer**（0.9B） 和 **GPT-NeoX-20B tokenizer**（16B）；
- 序列长度统一为 4096。

---

### **实验设置和评估指标**

#### **模型配置**
| 参数 | 0.9B 模型 | 16B 模型 |
|------|----------|---------|
| 总参数量 | 0.9B | 16.2B |
| 激活参数量 | 0.37B | 2.13B |
| MoE 层数 | 24 | 28 |
| 专家数 | 16 | 64 |
| Top-k routing | 4 | 6 |

#### **评估任务**
- **下游任务**（8项）：
  - ARC-Easy, BoolQ, HellaSwag, LAMBADA, OpenBookQA, PIQA, SocialIQA, WinoGrande
- **大规模评估**（16B 模型新增）：
  - MMLU, ARC-Challenge, CommonsenseQA, TriviaQA（5-shot exact match）
- **主要指标**：
  - 准确率（Accuracy %）
  - Perplexity（PPL）
  - Throughput（ktok/s/GPU）
  - Peak GPU Memory（GiB）

---

### **基线方法对比**

| 类别 | 方法 | 描述 |
|------|------|------|
| **Independent Routing** | Indep-MoE | 每层独立 router（标准 MoE） |
| | Rand-MoE | 固定随机 router（下界） |
| | X-MoE | 基于归一化表示防 collapse |
| **Recurrent Routing** | Recurrent-MoE | GRU 共享历史状态 |
| **Path-Constrained Routing** | LowRank-MoE | 共享 base router + 低秩扰动 |
| | MonoB8-MoE | 强制 block 内使用同一专家 |
| | **PathMoE**（本文） | **共享 router 参数，鼓励路径一致** |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **0.9B 模型结果（FineWeb-100B）**
| 方法 | 平均准确率 (%) | 最佳 PPL | 是否需 LBL |
|------|----------------|----------|-----------|
| Indep-MoE | 47.53 | 12.91 | 是 ($\alpha=0.01$) |
| PathB4-MoE (**PathMoE**) | **49.62** | **12.29** | ❌（无需） |

- **PathB4-MoE** 在平均准确率上 **+2.09%**，PPL 下降至 **12.29**，显著领先；
- 吞吐量与内存开销与 Indep-MoE 相当（55.67 vs 55.21 ktok/s；66.83 vs 66.94 GiB），无额外计算成本。

#### **16B 模型结果（DCLM-Pro）**
| 方法 | 平均准确率 (%) | 关键单项增益 |
|------|----------------|--------------|
| Indep-MoE | 48.39 | — |
| **PathMoE** | **50.34** | **+1.95%** |

- 在 12 项任务中赢下 **10 项**，尤其在：
  - **CommonsenseQA**: +5.73%
  - **ARC-Easy**: +5.09%
  - **OpenBookQA**: +3.80%

---

### **与基线方法的对比结果**

| 对比维度 | 结果 |
|--------|------|
| **vs. Indep-MoE** | 显著提升 accuracy 与 PPL，无需 LBL，训练更稳定 |
| **vs. Recurrent-MoE** | 性能相当但更高效（54.65 vs 55.71 ktok/s） |
| **vs. MonoB8-MoE** | **参数共享 > 决策强制**：PathMoE 更灵活且表现更好 |
| **vs. LowRank-MoE** | 更简单有效，无复杂矩阵分解 |

> 💡 **重要发现**：PathMoE 与 X-MoE 可结合（`PathB4X-MoE`），平均准确率从 48.44% → **48.86%**，说明其收益正交于表示优化。

---

### **消融实验结果**

#### **(1) Top-k Routing 分析**
- **Top-2 效果最佳**（平均 49.99%），平衡容量与专业化；
- Top-1 不稳定，Top-16 导致稀释。

#### **(2) Block Size 影响**
- **B=4 表现最优**（平均 49.62%）；
- 过小（B=1）缺乏协调，过大（B=24）过度约束；
- 不同任务偏好不同 block size：
  - 语言/常识类：B=4
  - 物理推理：B=8
  - 知识密集任务：B=24

#### **(3) 路径限制实验（Top-K Paths Only）**
| 限制路径数 | 平均准确率 |
|-----------|------------|
| 10 | 46.22% |
| 100 | 48.18% |
| 500 | 48.86% |
| All（无限制） | 48.71% |

→ 表明模型**自然收敛到数百条高频率路径**，验证路径集中现象合理。

---

## **4. 关键结论和发现**

### **主要发现**

1. **专家路径具有可解释的语言学结构**：
   - 相同路径的 token 自然聚类于特定语言功能：
     - 标点符号（punctuation）
     - 人名（person names）
     - 时间词（temporal words）
     - 动词（speech verbs）
   - PathMoE 比独立路由产生**更集中的聚类**。

2. **PathMoE 显著提升路由一致性**：
   - **跨层路径一致性**（Jaccard 相似度）：
     - PathB4-MoE: **~79%**
     - Indep-MoE: **~48%**
   - **持续参与率**（sustained engagement ≥4 层）：
     - PathMoE 更高，表明 token 更倾向于长期使用相同专家。

3. **更强的专业化与鲁棒性**：
   - **Routing Entropy** 降低 **11%**（更专注）；
   - **对抗专家索引打乱时**，PathMoE 的 PPL 增幅仅为 **237%**，而 Indep-MoE 高达 **5328%**；
   - 即使更专业化，也更鲁棒 → 说明依赖的是**结构化路径而非脆弱模式**。

4. **无需 load balancing loss**：
   - PathB4-MoE 在移除 LBL 后仍稳定训练并取得最佳性能；
   - 而 Indep-MoE 移除后出现震荡甚至崩溃。

---

### **方法的局限性**

- 当前设计假设为 **token-choice routing**（token 选专家）；
- 在 **expert-choice routing**（专家选 token）场景中未观察到收益，因其已有内在稳定性；
- **block size 需手动设定**，尚未实现自适应划分；
- 假设残差连接带来平滑表征演化，若结构变化剧烈可能影响效果。

---

### **未来工作方向**

1. **联合学习路径预测器**：让模型动态决定如何分块或调整共享策略；
2. **基于路径结构的模型压缩**：识别高频路径，合并冗余专家；
3. **探索非均匀 block 划分**：根据语义深度自动调整 block 大小；
4. **应用于多模态 MoE**：检验视觉/语音 token 是否也形成结构性路径。

---

> 📌 **总结一句话**：  
> **PathMoE 通过块状共享 router 参数，有效约束了 MoE 中爆炸性的专家路径空间，在无需辅助损失的情况下实现了更强的一致性、专业化、鲁棒性和性能提升，同时揭示了 MoE 路由中天然存在的语言学结构。**

</details>

---

### 13. [DriftGuard: Mitigating Asynchronous Data Drift in Federated Learning](https://arxiv.org/abs/2603.18872)

**Authors**: Yizhou Han, Di Wu, Blesson Varghese  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.18872v1  

#### Abstract
In real-world Federated Learning (FL) deployments, data distributions on devices that participate in training evolve over time. This leads to asynchronous data drift, where different devices shift at different times and toward different distributions. Mitigating such drift is challenging: frequent r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DriftGuard: Mitigating Asynchronous Data Drift in Federated Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在现实世界的 **Federated Learning (FL)** 部署中，设备上的数据分布会随时间动态变化，导致 **asynchronous data drift** —— 不同设备在不同时间以不同方式发生数据漂移。传统方法面临以下挑战：
- **频繁全局重训练（global retraining）** 导致资源受限设备计算开销巨大；
- **稀疏重训练** 又会导致部分设备性能长期下降；
- 现有 **clustering-based FCL** 方法虽降低参与设备数量，但将各组视为独立单元，无法共享全局可迁移知识，影响模型准确性。

### **提出的新方法：DriftGuard**
DriftGuard 是一种新型的 **Federated Continual Learning (FCL)** 框架，其核心创新在于结合 **Mixture-of-Experts (MoE)** 架构与双层级重训练机制，实现对异步数据漂移的高效适应。

#### **关键技术点：**
1. **MoE 启发的参数分离架构**：
   - 将模型分为 **shared parameters**（捕获跨设备通用知识）和 **local parameters**（适配特定设备组的数据分布）。
   - 引入 **branch-level soft gating** 和 **layer-level hard gating**，分别控制共享分支与本地分支的激活。

2. **基于 MoE gating 输出的无数据设备聚类**：
   - 利用设备上报的 **aggregated gating matrix**（层级别门控输出的统计），在服务器端进行 **agglomerative hierarchical clustering**，无需传输原始数据即可识别具有相似数据分布的设备群组。

3. **双层级重训练机制**：
   - **Global Retraining**：当全局平均准确率低于阈值时触发，仅更新 **shared parameters**，所有设备参与。
   - **Group Retraining**：当某组设备平均准确率下降时触发，仅更新该组的 **local parameters**，显著减少计算负载。

### **相比现有方法的优势**
| 维度 | 传统 FCL | Clustering-based FCL | DriftGuard |
|------|----------|------------------------|------------|
| 是否支持异步漂移 | ❌（假设同步漂移） | ✅ | ✅ |
| 全局与局部更新解耦 | ❌ | ❌ | ✅ |
| 支持轻量级自适应 | ❌ | ❌ | ✅ |
| 聚类无需原始数据 | ❌ | ❌ | ✅ |

> ✅ DriftGuard 在保持高精度的同时，大幅降低重训练成本，实现了更优的 **accuracy-cost trade-off**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验在三个多域图像分类数据集上进行，模拟 domain shift 类型的数据漂移：
- **DG5**：5个手写数字域（MNIST, MNIST-M, SVHN, SYN, USPS）
- **PACS**：4个视觉风格域（Photo, Art Painting, Cartoon, Sketch）
- **DomainNet**：6个域（Real, Clipart, Painting, Sketch, Infograph, Quickdraw）

### **模型架构**
基于两种主流结构构建 MoE 变体：
- **ResNet** → cResNet-S / cResNet-M
- **ViT** → cViT-S / cViT-M  
共四种模型配置用于评估。

### **实验设置**
- **设备数**：20 个设备
- **时间步**：30 步，每步模拟一次潜在漂移事件
- **漂移模式**：
  - **Instantaneous drift**：立即切换到新域
  - **Incremental drift**：在最多 4 个时间步内逐步过渡
- **漂移频率**：每步随机选择 10%-15% 设备发生漂移

### **评估指标**
- **Accuracy per retraining cost (ℰ)**：核心优化目标，定义为  
  $$
  ℰ = \frac{\text{Average Accuracy } (A)}{\text{Total Computational Cost } (TC)}
  $$
- **Total Cost (TC)**：累计 FLOPs 或真实世界中的 wall-clock time
- **Accuracy Preservation**：在参考准确率之上的时间步数

### **基线方法对比**
| 方法 | 触发条件 | 参与设备 | 更新参数 |
|------|----------|----------|----------|
| **FCL-AveTrig** | 全局平均准确率低 | 所有设备 | 全部参数 |
| **FCL-perDevice** | 单设备准确率低 | 准确率低的设备 | 全部参数 |
| **PFL-AveTrig** | 全局准确率低 | 所有设备 | shared + local（仅 shared 全局聚合） |
| **PFL-perDevice** | 单设备准确率低 | 准确率低的设备 | shared + local |
| **Cluster-based** | 组平均准确率低 | 组内设备 | 全部参数 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) Accuracy per Retraining Cost (ℰ)**
| 数据集-模型 | DriftGuard ℰ | 最强基线 ℰ | 提升倍数 |
|-------------|---------------|--------------|-----------|
| PACS-cResNet-M | **3.75** | 1.62 (PFL-perDevice) | **2.3×** |
| PACS-cViT-M | **3.47** | 2.68 (PFL-AveTrig) | **1.3×** |
| DomainNet-cResNet-M | **3.81** | 2.44 (PFL-AveTrig) | **1.6×** |
| DomainNet-cViT-M | **8.57** | 3.75 (PFL-AveTrig) | **2.3×** |
| DG5-cResNet-S | **12.14** | 9.56 (PFL-perDevice) | **1.3×** |

> ✅ DriftGuard 在 **6 个配置中的 5 个达到最高 ℰ**，最高提升达 **2.3×**

#### **(2) 总重训练成本降低**
- 相比最强基线，**总 retraining cost 降低 22%–83%**
- 成本节省主要来自：
  - 更少的 **global retraining 触发次数**（如 PACS 上从 13 次降至 1 次）
  - 更小的 **每次 group retraining 参数更新量**

#### **(3) 准确率保持能力**
| 数据集-模型 | DriftGuard > 参考线步数 | 最强基线步数 |
|-------------|--------------------------|----------------|
| PACS-cResNet-M | 16 | 18 (FCL-perDevice) |
| PACS-cViT-M | 21 | 22 (PFL-AveTrig) |
| DomainNet-cResNet-M | **17** | 17 (PFL-AveTrig) |
| DomainNet-cViT-M | **21** | 23 (PFL-AveTrig) |

> ✅ DriftGuard 在大多数情况下保持与最优基线相当甚至更高的稳定性。

### **消融实验结果（Hyperparameter 影响）**

#### **(1) Global Retraining Trigger Threshold**
- 当 `T_global ≤ 0.54` 时影响有限；
- 当 `T_global > 0.57` 时，触发过于频繁，cost 显著上升（从 0.08→0.21），行为趋近于 FCL-AveTrig。

#### **(2) Clustering Distance Threshold**
- 阈值在 **0.2–0.5** 区间内表现稳定（accuracy: 0.59–0.60, ℰ: 8.43–8.57）；
- 若设为极端值（如 0.1），产生过多小群组，数据不足导致重训练频繁，性能下降。

---

## **4. 关键结论和发现**

### **主要发现**
1. **异步数据漂移需差异化响应**：统一的 global retraining 浪费资源，而完全独立的 group modeling 忽视知识共享。
2. **DriftGuard 实现了最优权衡**：
   - 通过 **shared/local 参数解耦**，既保留全局知识迁移能力，又支持局部快速适应；
   - 通过 **gating matrix 聚类**，实现隐私保护下的设备分组；
   - 通过 **两级触发机制**，按需调度重训练，显著降低成本。
3. **在理论与真实硬件上均有效**：
   - 在模拟环境中，**retraining cost 最多降低 83%**，**ℰ 提升至 2.3×**；
   - 在 **Raspberry Pi 4 物理原型** 上，**retraining time 减少最多 20%**，**ℰ 提升至 1.2×**。

### **方法的局限性**
- **依赖 MoE 架构设计**：需要 careful tuning of expert 数量、top-k 设置等；
- **对极小群组敏感**：若聚类产生过小群组（<3台设备），local retraining 效果不佳；
- **未考虑通信开销建模**：当前 focus 在计算成本，未来可扩展至通信-计算联合优化。

### **未来工作方向**
1. 扩展至 **non-vision 任务**（如语音、文本）；
2. 引入 **adaptive clustering threshold** 动态调整群组粒度；
3. 探索 **asynchronous parameter aggregation** 进一步提升效率；
4. 结合 **model compression** 技术部署于更低功耗 IoT 设备。

---

> 🔗 **代码开源**：DriftGuard 已发布于 GitHub：[https://github.com/blessonvar/DriftGuard](https://github.com/blessonvar/DriftGuard)

</details>

---

### 14. [Communication-Efficient and Robust Multi-Modal Federated Learning via Latent-Space Consensus](https://arxiv.org/abs/2603.19067)

**Authors**: Mohamed Badi, Chaouki Ben Issaid, Mehdi Bennis  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.19067v1  

#### Abstract
Federated learning (FL) enables collaborative model training across distributed devices without sharing raw data, but applying FL to multi-modal settings introduces significant challenges. Clients typically possess heterogeneous modalities and model architectures, making it difficult to align featur...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Communication-Efficient and Robust Multi-Modal Federated Learning via Latent-Space Consensus**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对**多模态联邦学习（Multi-Modal Federated Learning, MM-FL）中的异构性挑战**，具体包括：
- 客户端拥有不同的**数据模态**（如加速度计ACC、陀螺仪GYR、mmWave、LiDAR等）；
- 客户端采用不同的**模型架构**，导致特征空间不一致；
- 传统FL依赖梯度或参数平均（如FedAvg），在跨模态、跨架构场景下无法对齐语义信息；
- 多数现有方法依赖**公共数据集**（如FedMD）或仅允许相同模态子集协作（如Harmony），限制了泛化能力和实用性。

### **提出了什么新方法或新思路**
作者提出了一种名为 **CoMFed**（Communication-Efficient Multi-Modal Federated Learning via Latent-Space Consensus）的新框架，其核心思想是：
- 引入可学习的**投影矩阵（projection matrices）** $ P_i \in \mathbb{R}^{d \times d_i} $，将各客户端高维中间特征映射到一个共享的低维**潜在空间（latent space）** $\mathbb{R}^d$（$d < d_i$）；
- 在该共享空间中交换**类别级均值潜表示（class-wise mean latent representations）** $ \mathbf{u}_{i,m} = P_i \cdot \mathbb{E}[\mathbf{v}_i(\mathbf{x})] $，实现跨客户端的知识对齐；
- 设计基于**几何中位数共识（geometric-median consensus）** 的正则项，增强对异常值和拜占庭客户端的鲁棒性；
- 整个协作过程不依赖任何公共数据集，且可在任意通信拓扑（centralized/decentralized）上运行。

### **相比现有方法的优势**
| 特性 | CoMFed | FedMD | Harmony (UniFL) | FedIoT |
|------|--------|-------|------------------|--------|
| 是否需要公共数据集 | ❌ 否 | ✅ 是 | ❌ 否 | ✅ 是 |
| 支持异构架构 | ✅ 是 | ✅ 是 | ✅ 是 | ✅ 是 |
| 支持不同模态组合 | ✅ 是 | ✅ 是 | ⚠️ 仅同类模态协作 | ✅ 是 |
| 通信开销 | $O(|\mathcal{M}|d)$ | $O(|\mathcal{D}_{\text{public}}|\cdot|\mathcal{M}|)$ | $O(|\mathbf{w}|)$ | $O(|\mathbf{w}|)$ |
| 鲁棒性机制 | ✅ 几何中位数正则 | ❌ 无 | ❌ 无 | ❌ 无 |
| 单阶段训练 | ✅ 是 | ⚠️ 多阶段蒸馏 | ✅ 是 | ⚠️ 多阶段 |

> ✅ 显著优势：**无需公共数据、通信高效、单阶段训练、支持完全异构环境、具备鲁棒性**

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **USC-HAD Dataset**
   - 来源：14个可穿戴设备采集的惯性传感器数据（ACC + GYR）
   - 模态分布：3个客户端仅用ACC，3个仅用GYR，8个使用双模态
   - 分类任务：人体活动识别（6类动作）
   - 局部模型：单模态客户端使用CNN+GRU；多模态客户端使用并行CNN分支融合后接GRU

2. **DeepSense Blockage-Duration Prediction Dataset**
   - 来源：真实世界mmWave雷达与LiDAR测量数据
   - 客户端划分：2个mmWave-only，2个LiDAR-only，2个双模态
   - 分类任务：障碍持续时间分类（4类）
   - 挑战：更强的模态差异性和客户端异构性

### **实验设置和评估指标**
- **总训练轮次**：250轮
- **学习率**：$10^{-3}$
- **CoMFed超参**：正则权重 $\lambda = 0.4$，每轮进行10次投影矩阵更新
- **评估指标**：
  - 平均测试准确率（Average Test Accuracy）
  - 收敛速度（Training Curves）
  - 累积通信开销（Cumulative Communication Overhead）
  - 对拜占庭攻击的鲁棒性（Byzantine Resilience）
  - 不同投影维度 $d$ 的影响（Ablation Study）

### **基线方法对比**
| 方法 | 类型 | 关键假设 |
|------|------|----------|
| **Harmony (UniFL)** | Modality-wise FL | 只有相同模态组合的客户端之间协作 |
| **FedMD** | Distillation-based FL | 需要共享公共数据集用于logits蒸馏 |
| **FedIoT** | Autoencoder + Global Classifier | 客户端训练自编码器，服务器用公共标签训练全局分类器 |
| **Centralized Training** | 上限基准 | 所有数据集中训练单一模型（理想情况） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **USC-HAD 数据集**
| 方法 | 达到40%准确率所需通信量 | 达到50% | 达到58% |
|------|-------------------------|--------|--------|
| **CoMFed** | **269 KB** | **410 KB** | **768 KB** |
| Harmony (UniFL) | 4.39 GB | 9.57 GB | NA |
| FedMD | 723 KB | 1.73 MB | NA |
| FedIoT | NA | NA | NA |

> 🔥 **CoMFed通信开销仅为Harmony的约1/16000，为FedMD的约1/6**

#### **DeepSense 数据集**
- CoMFed在所有模态子集上均取得最高平均准确率（见图4）
- 尤其在mmWave-only和LiDAR-only客户端表现优于FedMD和Harmony

### **与基线方法的对比结果**
- **收敛速度更快**：CoMFed在两个数据集上均比其他方法更早达到高精度（见Fig. 1 & 2）
- **平均准确率更高**：
  - USC-HAD：CoMFed ~58%，FedMD ~54%，Harmony ~53%
  - DeepSense：CoMFed ~69.8%，FedMD ~67.1%，Harmony ~63.4%
- **避免公共数据依赖**：FedMD因公共数据与私有数据分布不匹配导致训练波动（尤其在DeepSense上），而CoMFed稳定收敛

### **消融实验结果**
#### （1）**投影维度 $d$ 的影响（Fig. 6）**
- 随着 $d$ 增大（8→16→32→64），准确率先升后降
- 最佳性能出现在 $d=32$ 左右，过大维度反而引入噪声并增加通信成本
- 通信开销随 $d$ 线性增长，验证了轻量化设计的有效性

#### （2）**鲁棒性对比：算术平均 vs 几何中位数（Fig. 5）**
- 当存在拜占庭客户端时（$\gamma=1,2$），使用**几何中位数共识**的CoMFed变体显著优于使用算术平均的版本
- 表明所提正则器能有效抑制异常值干扰，提升系统鲁棒性

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **潜在空间共识是一种高效的跨模态协作机制**：通过低维投影实现语义对齐，无需共享原始数据或公共数据集。
2. ✅ **通信效率极高**：仅传输类别级潜表示（$O(|\mathcal{M}|d)$），远低于参数级传输（$O(|\mathbf{w}|)$）。
3. ✅ **单阶段端到端训练可行**：投影矩阵与本地模型联合优化，简化部署流程。
4. ✅ **鲁棒性强**：几何中位数共识机制有效缓解拜占庭行为和稀疏类估计误差。
5. ✅ **适用于任意网络拓扑**：既可用于去中心化图结构，也可适配中心化Parameter Server模式。

### **方法的局限性**
- 投影维度 $d$ 需要调优，过大会降低性能；
- 当某些类别样本极少时，类别均值估计可能不稳定（尽管几何中位数有一定缓解作用）；
- 当前未提供严格的**差分隐私（DP）保证**，虽然潜在空间天然适合加入噪声保护；
- 实验集中在中小规模客户端（≤14），大规模扩展性有待验证。

### **未来工作方向**
- 结合**差分隐私机制**，在潜表示中添加噪声以实现形式化隐私保护；
- 探索非线性投影函数（如小型MLP）替代线性$P_i$，进一步提升表达能力；
- 扩展至**异步训练**和**动态拓扑**场景；
- 应用于更多现实场景，如智能医疗、自动驾驶中的多传感器协同学习。

---

> 📌 **总结一句话**：  
> **CoMFed通过引入可学习的潜空间投影与几何中位数共识机制，在无需公共数据的前提下实现了高效、鲁棒、通信友好的多模态联邦学习，为复杂边缘环境下的异构协作提供了新范式。**

</details>

---

### 15. [From Inference Efficiency to Embodied Efficiency: Revisiting Efficiency Metrics for Vision-Language-Action Models](https://arxiv.org/abs/2603.19131)

**Authors**: Zhuofan Li (Celine), Hongkun Yang (Celine), Zhenyang Chen (Celine), Yangxuan Chen (Celine),  Yingyan (Celine),  Lin, Chaojian Li  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.19131v1  

#### Abstract
Vision-Language-Action (VLA) models have recently enabled embodied agents to perform increasingly complex tasks by jointly reasoning over visual, linguistic, and motor modalities. However, we find that the prevailing notion of ``efficiency'' in current VLA research, characterized by parameters, FLOP...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Inference Efficiency to Embodied Efficiency: Revisiting Efficiency Metrics for Vision-Language-Action Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 Vision-Language-Action (VLA) 模型的研究普遍采用传统的 **inference efficiency**（推理效率）指标（如参数量 #Parameters、FLOPs、解码吞吐量 Tokens/s）来衡量模型效率。然而，这些指标仅关注模型前向计算阶段的效率，**忽略了机器人实际执行任务时的物理行为成本**（如运动时间、能耗、轨迹平滑性等）。这导致一个看似“高效”的模型在真实部署中可能表现低效。

本文指出：  
> **“Inference-efficient” 不等于 “Embodied-efficient”**。

### ✅ 提出的新方法与新思路
- **提出“Embodied Efficiency”（具身效率）作为新的评估范式**，强调从系统级闭环行为角度评估 VLA 模型的实际执行效率。
- 定义了一套全面的 **embodied efficiency metrics**，涵盖：
  - **任务持续时间类**：`Task Completion Time`, `End-effector Path Length`, `Joint-space Path Length`
  - **运动平滑性类**：`Average Jerk L2 Norm`, `Average Action Rate`
- 将传统 AI 效率优化技术（如 weight pruning、quantization、token sparsification、action compression）置于具身环境中进行重新评估，揭示其对最终机器人行为的影响。

### ✅ 相比现有方法的优势
- **更贴近现实部署需求**：考虑了机器人平台的功耗、机械磨损、执行时间等实际因素。
- **暴露隐藏差异**：多个成功率达标的 VLA 模型在传统指标下表现相似，但在具身效率上存在显著差异。
- **指导模型设计与压缩策略**：为开发者提供更合理的权衡依据，避免“牺牲动作质量换推理速度”的陷阱。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **Libero 系列任务套件**（共四个子集）：
  - `Libero-Spatial`
  - `Libero-Object`
  - `Libero-Goal`
  - `Libero-10`
- **Bridge Benchmark**（更具挑战性的操作任务）
- 所有实验均基于仿真环境，遵循标准评估协议（每任务运行 N=50 轮或直到获得 10 次成功），结果取成功轨迹的平均值。

### ⚙️ 实验设置
- **测试模型**：
  - `To` [6]
  - `To.5` [1]
  - `MolmoAct` [40]
- **压缩技术分类研究**：
  1. **Model Compression**：weight pruning（幅度剪枝）、post-training quantization（int8/int4）
  2. **Token Compression**：visual token pruning [25]
  3. **Action Compression**：使用 FAST tokenizer 替换原 action head [26]
- **适应性策略探索**：
  - **In-context Learning**：通过 prompt 注入效率目标（如“最小化能量消耗”、“降低 action rate”）
  - **Supervised Fine-tuning**：引入辅助损失函数（jerk loss 或 action rate loss）

### 🎯 评估指标
| 类别 | 指标 | 含义 |
|------|------|------|
| **任务完成效率** | `Success Rate (%)` | 任务成功率 |
| | `Completion Time T` | 总执行时间（越小越好） |
| | `End-effector Path Length Lee` | 末端执行器路径长度 |
| | `Joint-space Path Length Ljoint` | 关节空间总移动距离 |
| **运动平滑性** | `Average Jerk L2 Norm J` | 加加速度范数，反映抖动程度（越小越平滑） |
| | `Average Action Rate R` | 高层动作变化频率（越小越稳定） |

> 所有指标均以 baseline 为 100%，报告相对变化百分比。

### 🔁 基线方法对比
- 主要对比对象为未压缩的原始 VLA 模型（如 `To`, `To.5`）。
- 对比不同压缩比例下的变体（如 `To w/ 5% pruning`, `To w/ int4 quantization`）。
- 引入 FAST 作为 action tokenizer 的替代方案进行横向比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ 表格 I & II：Weight Pruning 与 Quantization 结果（Libero 平均）
| 方法 | 成功率 | 完成时间↑ | 路径长度↑ | 抖动（Jerk）↑ |
|------|--------|-----------|------------|----------------|
| Weight Pruning (5–20%) | ±0–3% | +0.8–3.0% | +0–5.6% | **+2.6–11.0%** |
| Quantization (int4) | ±0–2.3% | +0.7–2.8% | +0–1.1% | **+3.5–19.5%** |

> ➤ 即使成功率不变，**剪枝和量化显著增加 jerk 和 completion time**。

#### ✅ 图 4：Bridge 数据集上的 weight pruning 影响
- 仅剪枝 5% 权重 →
  - 成功率下降 0.2%
  - **End-effector path length 增加 46.2%**
  - **Completion time 增加 13.6%**

> ➤ 在复杂任务中，轻微压缩即可引发严重的行为退化。

#### ✅ 图 5：Visual Token Pruning 影响
- 随着 token pruning ratio 提高（12% → 78%），
  - `Jerk` 可达 baseline 的 **3.75 倍**
  - 动作剧烈震荡，轨迹不连贯

> ➤ 视觉 token 压缩严重影响感知-动作一致性。

#### ✅ 表 III：Action Compression with FAST
| 指标 | 变化趋势 |
|------|---------|
| `Completion Time` | ↓ **1.5–5.6%**（更快） |
| `Jerk L2 Norm` | ↑ **28.0–50.6%**（更抖） |
| `Action Rate` | 微升或微降 |
| `Success Rate` | 下降最多达 -3.8% |

> ➤ **以牺牲平滑性和稳定性换取少量提速**。

#### ✅ 表 IV & V：Adaptation Strategies 效果
| 方法 | 指标改善 | 代价 |
|------|----------|------|
| **In-context Prompting** | `Jerk ↓ 最多 25.8%`<br>`Action Rate ↓ 最多 9.5%` | `Completion Time ↑ 最多 11.2%` |
| **Supervised Fine-tuning (Auxiliary Loss)** | `Jerk ↓ 13.2–24.0%` | `Completion Time ↑ 最多 3.8%` |

> ➤ 可定向优化特定指标，但存在明显 trade-off。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **推理效率 ≠ 具身效率**  
   减少 FLOPs 或参数量的方法（如 pruning、quantization）常导致：
   - 更长的任务完成时间
   - 更曲折的运动轨迹
   - 更高的 jerk 和 energy 消耗  
   → 实际系统级效率反而下降。

2. **传统指标掩盖重要行为差异**  
   多个模型成功率相同，但在 `completion time`、`jerk` 上差异显著，说明 **accuracy 和 throughput 无法反映 policy 质量**。

3. **embodied efficiency metrics 揭示隐藏 trade-off**  
   如 FAST 虽略快，但 jerk 显著上升；fine-tuning 可降 jerk，但拖慢执行。

4. **常见适配方法效果有限且有代价**  
   in-context learning 和 supervised fine-tuning 能部分提升具身效率，但改进是 **metric-specific 且伴随其他指标恶化**。

### ⚠️ 方法的局限性
- 当前实验主要在 **simulator 中进行**，缺乏真实机器人上的能耗测量（受限于模拟器对动力学建模的不足）。
- 所提 metrics 是 **kinematic proxies**（运动学代理），非直接能量测量。
- 未系统探索 multi-objective optimization（如同时优化 time + jerk）。

### 🔮 未来工作方向
- 开发支持 **end-to-end embodied efficiency optimization** 的训练框架。
- 设计兼顾 inference 与 actuation 成本的 **联合压缩算法**。
- 构建更真实的 **energy-aware simulator** 或开展真实硬件验证。
- 探索将 embodied efficiency metrics 纳入预训练或强化学习目标中。

---

## 🔚 总结
该论文从根本上质疑了当前 VLA 领域对“效率”的狭隘定义，呼吁社区从 **inference-centric evaluation** 转向 **system-level embodied efficiency assessment**。通过大量实证研究表明，许多流行的压缩与加速技术在提升推理效率的同时，会损害机器人的实际执行表现。作者提出的具身效率指标体系为未来 VLA 模型的设计、比较与部署提供了更全面、更实用的评估视角。

</details>

---

### 16. [A Computationally Efficient Learning of Artificial Intelligence System Reliability Considering Error Propagation](https://arxiv.org/abs/2603.18201)

**Authors**: Fenglian Pan, Yinwei Zhang, Yili Hong, Larry Head, Jian Liu  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18201v1  

#### Abstract
Artificial Intelligence (AI) systems are increasingly prominent in emerging smart cities, yet their reliability remains a critical concern. These systems typically operate through a sequence of interconnected functional stages, where upstream errors may propagate to downstream stages, ultimately aff...

---

### 17. [Retrieval-Augmented LLM Agents: Learning to Learn from Experience](https://arxiv.org/abs/2603.18272)

**Authors**: Thomas Palmeira Ferraz, Romain Deffayet, Vassilina Nikoulina, Herv\'e D\'ejean, St\'ephane Clinchant  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18272v1  

#### Abstract
While large language models (LLMs) have advanced the development of general-purpose agents, achieving robust generalization to unseen tasks remains a significant challenge. Current approaches typically rely on either fine-tuning or training-free memory-augmented generation using retrieved experience...

---

### 18. [AS2 -- Attention-Based Soft Answer Sets: An End-to-End Differentiable Neuro-Soft-Symbolic Reasoning Architecture](https://arxiv.org/abs/2603.18436)

**Authors**: Wael AbdAlmageed  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18436v1  

#### Abstract
Neuro-symbolic artificial intelligence (AI) systems typically couple a neural perception module to a discrete symbolic solver through a non-differentiable boundary, preventing constraint-satisfaction feedback from reaching the perception encoder during training. We introduce AS2 (Attention-Based Sof...

---

### 19. [ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents](https://arxiv.org/abs/2603.18815)

**Authors**: Hao Zhang, Mingjie Liu, Shaokun Zhang, Songyang Han, Jian Hu, Zhenghui Jin, Yuchi Zhang, Shizhe Diao, Ximing Lu, Binfeng Xu, Zhiding Yu, Jan Kautz, Yi Dong  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18815v1  

#### Abstract
Multi-turn LLM agents are increasingly important for solving complex, interactive tasks, and reinforcement learning (RL) is a key ingredient for improving their long-horizon behavior. However, RL training requires generating large numbers of sandboxed rollout trajectories, and existing infrastructur...

---

### 20. [RewardFlow: Topology-Aware Reward Propagation on State Graphs for Agentic RL with Large Language Models](https://arxiv.org/abs/2603.18859)

**Authors**: Xiao Feng, Bo Han, Zhanke Zhou, Jiaqi Fan, Jiangchao Yao, Ka Ho Li, Dahai Yu, Michael Kwok-Po Ng  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18859v1  

#### Abstract
Reinforcement learning (RL) holds significant promise for enhancing the agentic reasoning capabilities of large language models (LLMs) with external environments. However, the inherent sparsity of terminal rewards hinders fine-grained, state-level optimization. Although process reward modeling offer...

---

### 21. [Unmasking Algorithmic Bias in Predictive Policing: A GAN-Based Simulation Framework with Multi-City Temporal Analysis](https://arxiv.org/abs/2603.18987)

**Authors**: Pronob Kumar Barman, Pronoy Kumar Barman  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18987v1  

#### Abstract
Predictive policing systems that direct patrol resources based on algorithmically generated crime forecasts have been widely deployed across US cities, yet their tendency to encode and amplify racial disparities remains poorly understood in quantitative terms. We present a reproducible simulation fr...

---

### 22. [CWoMP: Morpheme Representation Learning for Interlinear Glossing](https://arxiv.org/abs/2603.18184)

**Authors**: Morris Alper, Enora Rice, Bhargav Shandilya, Alexis Palmer, Lori Levin  
**Category**: cs.CL  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18184v1  

#### Abstract
Interlinear glossed text (IGT) is a standard notation for language documentation which is linguistically rich but laborious to produce manually. Recent automated IGT methods treat glosses as character sequences, neglecting their compositional structure. We propose CWoMP (Contrastive Word-Morpheme Pr...

---

### 23. [EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models](https://arxiv.org/abs/2603.18489)

**Authors**: Minsoo Cheong, Donghyun Son, Woosang Lim, Sungjoo Yoo  
**Category**: cs.CL  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18489v1  

#### Abstract
Diffusion-based large language models (dLLMs) rely on bidirectional attention, which prevents lossless KV caching and requires a full forward pass at every denoising step. Existing approximate KV caching methods reduce this cost by selectively updating cached states, but their decision overhead scal...

---

### 24. [Detecting Basic Values in A Noisy Russian Social Media Text Data: A Multi-Stage Classification Framework](https://arxiv.org/abs/2603.18822)

**Authors**: Maria Milkova, Maksim Rudnev  
**Category**: cs.CL  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18822v1  

#### Abstract
This study presents a multi-stage classification framework for detecting human values in noisy Russian language social media, validated on a random sample of 7.5 million public text posts. Drawing on Schwartz's theory of basic human values, we design a multi-stage pipeline that includes spam and non...

---

### 25. [RE-SAC: Disentangling aleatoric and epistemic risks in bus fleet control: A stable and robust ensemble DRL approach](https://arxiv.org/abs/2603.18396)

**Authors**: Yifan Zhang, Liang Zheng  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18396v1  

#### Abstract
Bus holding control is challenging due to stochastic traffic and passenger demand. While deep reinforcement learning (DRL) shows promise, standard actor-critic algorithms suffer from Q-value instability in volatile environments. A key source of this instability is the conflation of two distinct unce...

---

### 26. [GAPSL: A Gradient-Aligned Parallel Split Learning on Heterogeneous Data](https://arxiv.org/abs/2603.18540)

**Authors**: Zheng Lin, Ons Aouedi, Wei Ni, Symeon Chatzinotas, Xianhao Chen  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18540v1  

#### Abstract
The increasing complexity of neural networks poses significant challenges for democratizing FL on resource?constrained client devices. Parallel split learning (PSL) has emerged as a promising solution by offloading substantial computing workload to a server via model partitioning, shrinking client-s...

---

### 27. [Book your room in the Turing Hotel! A symmetric and distributed Turing Test with multiple AIs and humans](https://arxiv.org/abs/2603.18981)

**Authors**: Christian Di Maio, Tommaso Guidi, Luigi Quarantiello, Jack Bell, Marco Gori, Stefano Melacci, Vincenzo Lomonaco  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.18981v1  

#### Abstract
In this paper, we report our experience with ``TuringHotel'', a novel extension of the Turing Test based on interactions within mixed communities of Large Language Models (LLMs) and human participants. The classical one-to-one interaction of the Turing Test is reinterpreted in a group setting, where...

---

### 28. [Efficient Dense Crowd Trajectory Prediction Via Dynamic Clustering](https://arxiv.org/abs/2603.18166)

**Authors**: Antonius Bima Murti Wijaya, Paul Henderson, Marwa Mahmoud  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.18166v1  

#### Abstract
Crowd trajectory prediction plays a crucial role in public safety and management, where it can help prevent disasters such as stampedes. Recent works address the problem by predicting individual trajectories and considering surrounding objects based on manually annotated data. However, these approac...

---

### 29. [From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents](https://arxiv.org/abs/2603.18382)

**Authors**: Myeongseob Ko, Jihyun Jeong, Sumiran Singh Thakur, Gyuhak Kim, Ruoxi Jia  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.18382v1  

#### Abstract
Anonymization is widely treated as a practical safeguard because re-identifying anonymous records was historically costly, requiring domain expertise, tailored algorithms, and manual corroboration. We study a growing privacy risk that may weaken this barrier: LLM-based agents can autonomously recons...

---

### 30. [AlignMamba-2: Enhancing Multimodal Fusion and Sentiment Analysis with Modality-Aware Mamba](https://arxiv.org/abs/2603.18462)

**Authors**: Yan Li, Yifei Xing, Xiangyuan Lan, Xin Li, Haifeng Chen, Dongmei Jiang  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.18462v1  

#### Abstract
In the era of large-scale pre-trained models, effectively adapting general knowledge to specific affective computing tasks remains a challenge, particularly regarding computational efficiency and multimodal heterogeneity. While Transformer-based methods have excelled at modeling inter-modal dependen...

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
