# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-20 06:38:03 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference](https://arxiv.org/abs/2603.19133)

**Authors**: Yida Zhang, Zhiyong Gao, Shuaibing Yue, Jie Li, Rui Wang  
**Category**: cs.DC  
**Published**: 2026-03-20  
**Score**: 16.0  
**Type**: new  
**ArXiv ID**: 2603.19133v1  

#### Abstract
Recent advancements and widespread adoption of Large Language Models (LLMs) in both industry and academia have catalyzed significant demand for LLM serving. However, traditional cloud services incur high costs, while on-device inference alone faces challenges due to limited resources. Edge-cloud col...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PicoSpec: A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前 Large Language Models (LLMs) 在边缘设备上部署面临**计算资源受限**的问题，而完全依赖云端推理则带来**高延迟和带宽成本**。虽然 Edge-Cloud 协同推理被提出作为折中方案，但传统 speculative decoding 方法在分布式环境下存在以下瓶颈：

- **通信-计算不匹配（Communication-Computation Mismatch）**：由于 WAN 环境下高 Round-Trip Time (RTT)，传统的“停等式”（stop-and-wait）机制导致大量 pipeline bubbles，严重限制吞吐量。
- **高通信开销**：频繁传输完整的 vocabulary distribution 导致带宽压力巨大。
- **灵活性差**：现有方法如 early-exit 或隐藏状态交换需要模型重训练或微调，缺乏通用性。

### **提出的新方法与思路**
本文提出了 **PicoSpec** —— 一种**无需训练、支持异步流水线的协同 speculative decoding 框架**，用于高效的 Edge-Cloud LLM 推理。

#### **核心创新点：**
1. ✅ **异步协同 speculative decoding 流水线（Asynchronous Pipeline）**
   - 引入 **Parallel Drafting** 和 **Fast Verification** 机制，解耦边端 draft 生成与云端验证过程。
   - 边缘设备可在等待前一批 token 验证的同时继续推测后续 token，有效掩盖网络延迟。

2. ✅ **分离式拒绝采样算法（Separate Rejection Sampling with Sparse Compression）**
   - 将 rejection sampling 分布到边端执行，仅需上传 draft token 对应的概率 $ q_i $，而非完整分布。
   - 下行链路采用 **Top-K 稀疏压缩**，将目标模型输出 $ P(x) $ 中 top-K 概率及其索引发送回边端，大幅降低通信负载（从 O(V) 降至 O(K)）。

3. ✅ **零拷贝通信与动态截断机制**
   - 使用 Zero-Copy Communicator 减少序列化开销。
   - 引入 **latency-aware truncation**，当 drafting 时间过长时自动提前提交，防止云侧空闲。

### **相比现有方法的优势**
| 特性 | PicoSpec | 其他方法（如 DSD、SLED、HAT） |
|------|--------|-------------------------------|
| 是否需要模型修改 | ❌ 否（Plug-and-play） | ✅ 是（需 fine-tune / early exit） |
| 是否支持异步流水线 | ✅ 是 | ❌ 多为 stop-and-wait |
| 通信效率 | ✅ 极高（稀疏压缩 + 局部 resampling） | ⚠️ 高开销（全分布传输） |
| 通用性 | ✅ 支持任意标准 LLM/SLM 组合 | ⚠️ 受限于特定架构 |

> 💡 **优势总结**：PicoSpec 实现了真正的并行化、低通信开销、无需训练的协同推理框架，在高延迟广域网环境中仍能保持高性能。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **GSM8K**：数学推理任务，测试逻辑推导能力。
- **HumanEval**：代码生成任务，评估编程能力。

### **实验设置**
- **边端设备**：NVIDIA Jetson AGX（代表资源受限边缘硬件）
- **云端服务器**：配备 NVIDIA A100 (40GB) GPU 的高性能集群
- **网络环境**：Wide Area Network (WAN)，具有高 RTT 和有限带宽
- **模型组合**：
  - Qwen3 0.6B (edge) + Qwen3 32B (cloud)
  - Llama3.1 1B (edge) + Llama3.1 70B (cloud)
- **Speculative Step Size**：默认设为 4

### **评估指标**
- **Throughput (tokens/s)**：每秒生成 token 数量
- **Speedup**：相对于 Autoregressive 基线的加速比
- **Average Acceptance Length**：每个 speculation 步骤平均接受的 token 数
- **Time Per Output Token (TPOT)**：输出每个 token 所需时间
- **End-to-End Latency**
- **Ablation Study**：分析各模块对性能的影响

### **基线方法对比**
| 基线方法 | 描述 |
|--------|------|
| **Autoregressive (AR)** | 云端逐 token 自回归生成 |
| **Vanilla Speculative Decoding** | 传统 speculative decoding，应用于边云场景（stop-and-wait） |
| **Split Inference** | 模型切分，前几层运行在边端，其余在云端 |
| **DSD [Yu et al., 2025]** | 分布式 speculative decoding，使用 Adaptive Window Control |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| 方法 | Qwen-GSM8K | Qwen-HumanEval | Llama-GSM8K | Llama-HumanEval |
|------|------------|----------------|-------------|------------------|
| Autoregressive | 13.89 t/s (1.00x) | 14.18 t/s (1.00x) | 6.87 t/s (1.00x) | 6.86 t/s (1.00x) |
| Vanilla Spec. | 8.04 t/s (0.58x) | 6.24 t/s (0.44x) | 9.27 t/s (1.35x) | 9.95 t/s (1.45x) |
| Split Inference | 7.51 t/s (0.54x) | 7.52 t/s (0.53x) | 4.33 t/s (0.63x) | 4.53 t/s (0.66x) |
| **PicoSpec (Ours)** | **20.19 t/s (1.45x)** | **16.04 t/s (1.13x)** | **17.22 t/s (2.51x)** | **19.88 t/s (2.90x)** |

> 🔥 **最高达 2.9× 加速！**

### **与基线方法对比结果**
- **相较于 AR**：PicoSpec 在 Llama 组合上实现 **2.5–2.9× 吞吐提升**，尤其在大模型（70B）上增益更显著。
- **优于 Vanilla Speculative**：后者因 stop-and-wait 开销过大，在 Qwen 上甚至**慢于 AR**（仅 0.44x），而 PicoSpec 成功逆转劣势。
- **远超 Split Inference**：Split 方法受限于 RTT，无法突破物理带宽瓶颈；PicoSpec 则通过异步 pipeline 实现“**latency immunity**”。

### **消融实验结果（Ablation Study）**

#### **Table 2 & 3 关键发现：**

| 模块移除 | 影响说明 |
|---------|--------|
| **w/o Parallel Drafting** | 吞吐下降最严重（如 Llama-GSM8K 从 17.22 → 12.51 t/s），证明异步 pipeline 是核心驱动力 |
| **w/o Fast Verification** | TPOT 明显上升，验证准备延迟增加，pipeline bubble 扩大 |
| **w/o Separate Rejection Sampling** | $ T_{verify} $ 显著升高（如 Llama 场景从 166ms → 283ms），因需处理完整概率分布，带宽和 CPU 开销剧增 |

> ✅ 三者协同作用，缺一不可。

### **参数敏感性分析**
- 最优 **draft length = 4** 时达到峰值吞吐（20.19 t/s）。
- 当 $ n > 4 $，$ T_{draft} $ 上升导致本地开销超过收益，系统退出非阻塞状态。
- 表明 PicoSpec 对参数不敏感，在合理范围内稳定高效。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **PicoSpec 实现了“Latency Immunity”**
   - 即使在网络 RTT 较高的情况下，也能通过异步 pipeline 将通信延迟隐藏在计算过程中，使整体速度逼近本地 draft 模型极限。

2. ✅ **大模型受益更多**
   - 对 Llama-70B 这类 $ T_{verify} $ 更大的模型，PicoSpec 能更好地掩盖验证延迟，获得更高加速比（最高 2.9×）。

3. ✅ **Acceptance Rate 决定性能上限**
   - 性能与平均 acceptance length 正相关。High-alignment 模型对（如 Llama-HumanEval 达 4.9 tokens）可维持 pipeline 满载运行。

4. ✅ **真正实现了边云协同的并行化**
   - 不再是“边发-云验-边等”的串行模式，而是边端持续 speculative generation，云端并行验证，形成高效流水线。

### **方法的局限性**
- **依赖 draft model 与 target model 的 alignment**
  - 若小模型生成质量差（acceptance rate 低），会导致频繁 rollback，削弱异步优势。
- **不适合极低带宽极端场景**
  - 尽管已压缩通信，但在极低带宽下仍可能成为瓶颈。
- **rollback 开销未完全消除**
  - token 被拒后需恢复 KV Cache 状态，虽有优化但仍有一定代价。

### **未来工作方向**
- 探索 **adaptive draft length 控制策略**，根据实时 acceptance rate 动态调整 speculation 窗口。
- 结合 **multi-token prediction 或 tree-based speculation** 进一步提升 draft 效率。
- 扩展至 **multi-edge client** 场景下的调度与资源竞争管理。
- 研究 **轻量化 draft model 设计原则**，以更好适配 PicoSpec 框架。

---

> 📌 **总结一句话**：  
> **PicoSpec 是首个无需训练、支持异步流水线的边云协同 speculative decoding 框架，通过 Parallel Drafting + Fast Verification + Separate Rejection Sampling 三大技术，在真实高延迟环境下实现了高达 2.9× 的推理加速，显著提升了 Edge-Cloud LLM 推理效率。**

</details>

---

### 2. [AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models](https://arxiv.org/abs/2603.18464)

**Authors**: Chengxuan Lu, Shukuan Wang, Yanjie Li, Wei Liu, Shiji Jin, Fuyuan Qian, Peiming Li, Baigui Sun, Yang Liu  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 15.0  
**Type**: new  
**ArXiv ID**: 2603.18464v1  

#### Abstract
Reinforcement learning (RL) for large-scale Vision-Language-Action (VLA) models faces significant challenges in computational efficiency and data acquisition. We propose AcceRL, a fully asynchronous and decoupled RL framework designed to eliminate synchronization barriers by physically isolating tra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AcceRL: A Distributed Asynchronous Reinforcement Learning and World Model Framework for Vision-Language-Action Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Vision-Language-Action (VLA) 模型在具身智能（Embodied Intelligence）任务中展现出巨大潜力，但将其应用于 **大规模强化学习**（Reinforcement Learning, RL）时面临两大瓶颈：

1. **系统效率低下**：传统同步 RL 框架存在严重的“长尾延迟”（long-tail latency）和 GPU 空转问题，因物理仿真器（physical simulator）速度远慢于模型推理，导致训练吞吐量受限。
2. **样本效率极低**：依赖真实环境交互获取经验，采样成本高昂，尤其对多模态、高维输入的 VLA 模型而言，数据饥渴严重。

### 🚀 提出的新方法与创新
作者提出 **AcceRL** —— 一种**完全异步、解耦的分布式 RL 框架**，并首次将可训练的 **World Model** 集成到 VLA 的异步训练流程中。

#### 主要创新点：
- **全异步架构（Fully Asynchronous Architecture）**
  - 实现 **宏观异步**（Macro-Asynchrony）：解耦 Rollout 与 Training，消除全局同步屏障。
  - 实现 **微观异步**（Micro-Asynchrony）：将环境交互与模型推理分离，采用 “Inference-as-a-Service” 范式，通过中央 Inference Pool 动态批处理请求，最大化 GPU 利用率。

- **首个支持可插拔 World Model 的分布式异步 RL 框架**
  - 引入基于扩散模型（Diffusion-based）的 **Observation Model $M_{\text{obs}}$** 和二分类 **Reward Model $M_{\text{reward}}$**，构建虚拟环境进行“想象中的 rollout”（imagination rollouts）。
  - 支持“零卸载”（zero-offload）执行：策略和世界模型常驻专用 GPU 推理池，避免频繁内存交换，显著降低通信开销。

- **算法级优化增强稳定性与效率**
  - **Value Re-computation**：每次训练前重新计算价值目标，缓解异步带来的策略滞后（policy lag）。
  - **Global Advantage Normalization**：跨节点归一化优势值，提升训练稳定性。
  - **GIPO（Gaussian Importance sampling Policy Optimization）**：替代 PPO 的硬裁剪机制，软性控制策略更新幅度，提高鲁棒性。
  - **Dynamic Weighted Resampling (DWR)**：根据任务成功率动态调整采样权重，聚焦困难任务，防止灾难性遗忘。

### 🔍 相比现有方法的优势
| 维度 | AcceRL | 现有框架（如 IMPALA、Ray RLlib、RLinf-VLA） |
|------|--------|---------------------------------------------|
| 架构同步性 | 完全异步 | 同步或部分异步 |
| 系统吞吐 | 超线性扩展（super-linear scaling） | 受限于最慢 worker（straggler effect） |
| 样本效率 | 提升 **200×**（得益于 World Model） | 依赖真实环境采样，效率低 |
| 内存管理 | 零卸载设计，VRAM 利用高效 | 存在频繁 CPU-GPU 参数搬运 |
| 可扩展性 | 支持大规模集群部署 | 扩展性差，通信开销大 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **LIBERO Benchmark**：一个面向终身机器人学习的挑战性基准，包含四类子任务：
  - **LIBERO-Spatial**：空间布局相关任务
  - **LIBERO-Object**：物体操作任务
  - **LIBERO-Goal**：目标条件任务
  - **LIBERO-Long**：长视野任务（最长可达 200 步）

> 所有实验均在 **MuJoCo 物理引擎** 中运行，使用 **OSMesa 进行无头渲染**，支持大规模并行 rollout。

### ⚙️ 实验设置与评估指标

| 项目 | 设置 |
|------|------|
| **主干模型** | OpenVLA-OFT（基于 Llama-2-7B） |
| **World Model 结构** | <br>- $M_{\text{obs}}$: DIAMOND 架构（diffusion-based 观测预测）<br>- $M_{\text{reward}}$: 微调后的 OpenVLA-OFT 分类器（判断成功概率） |
| **训练方式** | 分布式异步 RL，ZeRO-2 优化梯度分区 |
| **评估指标** | <br>- **Success Rate (%)**：各 LIBERO 子集的任务成功率<br>- **Samples Per Second (SPS)**：训练吞吐量<br>- **硬件利用率（GPU %）**<br>- **学习曲线（Average Return vs. Env Steps / Training Steps）** |

### 🆚 基线方法对比
- **OpenVLA-OFT**：监督微调基线（Behavior Cloning）
- **SimpleVLA-RL**：基于 PPO 的同步 RL 微调框架
- **RLinf-VLA**：混合细粒度流水线的异步框架
- **AcceRL (w/o WM)**：消融版，不使用 World Model

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）系统吞吐与可扩展性（Table 1 & Figure 4）
| GPU 数量 | Throughput (SPS) | GPU 利用率 |
|---------|------------------|-----------|
| 1       | 14.13            | 96.45%    |
| 2       | 28.82            | 97.17%    |
| 4       | 60.33            | 98.36%    |
| 7       | **104.22**       | 95.07%    |

✅ **结论**：AcceRL 在训练器 GPU 上实现了 **超线性扩展**（super-linear scaling），且 GPU 利用率始终 >94%，表明其卓越的硬件利用能力和系统效率。

#### （2）任务性能（Table 2）
| 方法 | Spatial (%) | Object (%) | Goal (%) | Long (%) |
|------|-------------|------------|----------|----------|
| OpenVLA-OFT | 96.2 | 98.3 | 96.2 | 90.7 |
| SimpleVLA-RL | 99.4 | 99.8 | 99.2 | 98.5 |
| RLinf-VLA | 99.4 | 99.8 | 98.8 | 94.0 |
| **Ours (AcceRL)** | **99.6** | **100.0** | **98.8** | **99.1** |

✅ **结论**：AcceRL 在所有类别上达到 **SOTA 性能**，尤其在 **Long-Horizon 任务** 上表现突出（**99.1%**），远超监督学习基线（90.7%），证明其在复杂长期任务中的强大泛化与纠错能力。

#### （3）样本效率提升（Figure 5）
- **AcceRL with WM** 仅需约 **10,000 真实环境步数** 即突破 0.8 平均回报阈值。
- 相比之下，纯在线 RL 需要数十万步才能达到类似水平。
- **World Model 贡献：在线样本效率提升 200×（即 20,000%）**

#### （4）收敛速度
- 借助来自可微分 World Model 的密集梯度信号，策略在 **少于 400 次训练更新** 后即收敛至近最优性能。

---

### 🔬 消融实验结果

#### （1）Value Re-computation 消融（Figure 6）
- **移除 Value Re-computation** 导致性能大幅下降、方差增大。
- 表明该机制对缓解异步环境下的 **数据陈旧性（data staleness）** 至关重要，是稳定训练的关键。

#### （2）GIPO vs PPO 对比（Figure 7）
- 使用标准 **PPO** 出现剧烈震荡，收敛缓慢，在 ~60k 步才达到 GIPO 在 ~8k 步的效果。
- **GIPO** 显著提升稳定性与样本效率，实现约 **7.5 倍加速**。
- 证明其在异步、高延迟场景下优于传统 PPO。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **异步解耦架构可极大提升 VLA 模型的训练效率**  
   通过彻底分离 rollout、inference 与 training，AcceRL 成功打破物理仿真的频率瓶颈，实现 **超线性吞吐扩展** 和 **接近饱和的 GPU 利用率**。

2. **集成 World Model 是突破样本效率瓶颈的关键**  
   首次实现 **plug-and-play 式可训练 World Model** 在分布式异步 RL 中的应用，使模型能在“梦境”中学习，**在线样本效率提升 200×**。

3. **算法设计必须适配异步系统特性**  
   传统的 PPO 在异步环境下失效，而 **GIPO + Value Re-computation + Global Norm** 等组合有效解决了策略滞后与梯度噪声问题，保障了训练稳定性。

4. **AcceRL 实现在复杂控制任务上的 SOTA 表现**  
   在 LIBERO 基准上全面超越现有方法，尤其在长视野任务中展现卓越鲁棒性。

---

### ⚠️ 局限性
- 当前框架尚未支持 **大规模语言模型的后训练对齐**（post-training alignment）。
- World Model 依赖高质量离线轨迹预训练（文中使用 1,000 条），冷启动阶段仍需一定真实数据。
- 扩散模型生成虽保真度高，但推理延迟仍高于轻量级 latent world models。

---

### 🔮 未来工作方向
- 将 AcceRL 架构扩展至支持 **完整规模的语言模型对齐训练**。
- 探索更高效的 **world model 架构**（如 latent + diffusion hybrid）以进一步降低延迟。
- 引入 **自适应 horizon 控制** 机制，动态调整 imagination rollout 长度，平衡误差累积与探索深度。
- 推广至真实机器人部署，验证其在真实世界中的迁移能力。

--- 

> **总结一句话**：  
> **AcceRL 通过“全异步 + 可插拔 World Model”的双重革新，为大规模 VLA 模型的强化学习提供了一条高效、稳定、可扩展的新路径，实现了系统吞吐与样本效率的双重飞跃。**

</details>

---

### 3. [DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge](https://arxiv.org/abs/2603.19172)

**Authors**: Yuegui Huang, Zhiyuan Fang, Weiqi Luo, Ruoyu Wu, Wuhui Chen, Zibin Zheng  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2603.19172v1  

#### Abstract
Despite the computational efficiency of MoE models, the excessive memory footprint and I/O overhead inherent in multi-expert architectures pose formidable challenges for real-time inference on resource-constrained edge platforms. While existing static methods struggle with a rigid latency-accuracy t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DyMoE: Dynamic Expert Orchestration with Mixed-Precision Quantization for Efficient MoE Inference on Edge》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
尽管 **MoE**（Mixture-of-Experts）模型在计算上具有稀疏激活特性，理论上适合边缘设备部署，但其庞大的参数量导致严重的**内存占用**和**I/O开销**。尤其是在资源受限的边缘硬件（如消费级笔记本、嵌入式AI加速器）上，即使采用量化或专家卸载（offloading），仍面临以下挑战：
- **静态量化策略**无法适应输入动态变化，导致精度损失严重；
- **按需加载专家**引发显著的 `Wait-for-Weight` 延迟；
- **现有系统缺乏对专家重要性差异和层间敏感度差异的细粒度建模**。

### 提出的新方法与创新思路
作者提出 **DyMoE** —— 一种算法-系统协同设计的动态混合精度推理框架，核心思想是：
> 利用 **专家重要性的动态偏斜性** 和 **深度依赖的量化鲁棒性**，实现运行时自适应的专家调度与精度分配。

具体创新点包括：

#### （1）**Dynamic Expert Importance Classification Method**
- 在 **prefill 阶段**：基于 token 的注意力权重聚合定义“重击者 token”（heavy-hitter tokens），并以处理这些 token 的数量作为专家重要性指标。
- 在 **decode 阶段**：直接利用 **gating score** 作为专家重要性的实时代理。
- 引入 **depth-aware retention ratio**，浅层保留更多高精度专家（因更敏感），深层允许更高压缩率。

#### （2）**High-Performance Inference System**
构建了一个完整的运行时调度引擎，包含两个关键组件：
- **Look-ahead Prefetching Engine**：利用相邻层之间的激活相似性（high inter-layer cosine similarity），通过当前层隐藏状态预测下一层的路由分布，提前预取关键专家。
- **Mixed-Precision Cache Manager**：支持不同精度（如 INT8/INT4/INT2/"0-bit"）的专家缓存管理，并制定三条规则防止冗余与精度降级：
  - No Duplication
  - Precision Promotion（低精缓存 → 请求高精 = 缓存未命中）
  - Conservative Reuse（高精缓存 → 请求低精 = 可复用）

#### （3）端到端 Plug-and-Play 解决方案
无需重新训练或校准，兼容主流 PTQ 技术（如 GPTQ、AWQ），可无缝集成进现有推理流程。

### 相比现有方法的优势
| 维度 | 现有方法局限 | DyMoE 改进 |
|------|---------------|-----------|
| **量化策略** | 静态统一或分层固定比例（如 EdgeMoE） | 动态、输入感知、混合精度 |
| **专家选择** | 固定保留集或随机跳过 | 基于 token/gate 的重要性动态识别 |
| **I/O优化** | 被动加载或简单预取 | 主动 lookahead 预取，有效掩盖延迟 |
| **适用场景** | 批处理友好，不适合单 token 流式生成 | 特别优化 **batch size = 1** 的边缘交互式场景 |

---

## 2. 核心实验方法和设置

### 使用的模型与数据集
- **模型**：
  - `Mixtral-8×7B`：粗粒度 MoE 架构（每层8专家，激活2）
  - `Qwen3-30B-A3B`：高稀疏性 MoE 架构（每层大量专家，仅激活少数）
- **数据集**：
  - **推理负载**：从 `ShareGPT` 数据集中采样真实对话序列，模拟实际用户请求。
  - **准确性评估**：使用标准评测基准：
    - `MMLU`（多任务理解）
    - `CMMLU`（中文多任务理解）
    - `GSM8K`（数学推理）

### 实验设置
- **硬件平台**：
  - 主机：AMD EPYC 7542 CPU + NVIDIA RTX 3090（24GB VRAM）
  - 模拟边缘环境：通过软件限制 VRAM 至 **12GB / 16GB / 24GB**
  - 接口：PCIe Gen3 ×16（带宽受限）
- **配置对比**：
  - DyMoE 两种模式：
    - `"4/2"`：关键专家 INT4，次关键专家 INT2
    - `"4/0"`：关键专家 INT4，非关键专家完全跳过（0-bit）
  - 控制变量：专家保留比例 `r ∈ [0.75, 0.9, 1.0]`

### 评估指标
| 指标 | 含义 | 应用阶段 |
|------|------|---------|
| **TTFT**（Time-to-First-Token） | 首个输出 token 的延迟 | Prefill 阶段 |
| **TPOT**（Time-Per-Output-Token） | 每个解码 token 的平均延迟 | Decode 阶段 |
| **Accuracy** | 多项基准得分 | 模型保真度验证 |

### 基线方法对比
1. **Accelerate**（acc）：通用异构设备支持框架，支持量化与分区
2. **Mixtral-Offloading**：专为 MoE 设计，使用 LRU 缓存 + 混合精度
3. **MoE-Infinity**：基于激活感知的预取与细粒度缓存
4. **Fiddler**：CPU-GPU 协同执行，动态卸载计算

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Figure 10 与 Table 2）

| 指标 | 模型 | 场景 | DyMoE 表现 | 提升倍数 |
|------|------|-------|------------|----------|
| **TTFT** | Mixtral-8×7B | 24GB | **↓ 22.7× vs Fiddler** | 最高达 22.7× 加速 |
| **TTFT** | Qwen3-30B-A3B | 12GB | **↓ 3.44× vs Accelerate** | 显著降低首 token 延迟 |
| **TPOT** | Mixtral-8×7B | 16GB | **↓ 14.58× vs Fiddler** | 解码速度大幅提升 |
| **TPOT** | Qwen3-30B-A3B | 12GB | **↑ 2.86× faster** | 高稀疏模型同样受益 |

> ✅ 所有加速均在 **几乎无损精度** 下实现。

### 与基线方法的对比结果
- 在所有配置下，DyMoE 均显著优于四大基线：
  - 相比 **Accelerate**：TTFT 提升 7.2×–8.3×，TPOT 提升 8.3×
  - 相比 **Mixtral-Offloading**：最高达 **14.58× TPOT 优势**
  - 相比 **Fiddler**：虽为 co-execution 设计，但在 MoE 上反而因 dequantization 成为瓶颈

### 消融实验结果（Table 3）
在 `Mixtral-8×7B` 上进行增量消融（16GB/24GB 设置）：

| 配置 | TPOT (16GB) | 提升倍数（vs baseline） | 关键改进 |
|------|-------------|------------------------|----------|
| 1. Load-on-Demand | 0.2795s | 1.0× | 基线 |
| 2. + Expert Cache | 0.1489s | 1.88× | 减少重复传输 |
| 3. + Prefetching | 0.1315s | 2.13× | 重叠 I/O 与计算 |
| 4. + DyQuant (4/2) | 0.1307s | 2.14× | 减小 I/O 数据量 |
| 5. 全部组合 | **0.1048s** | **2.67×** | 协同效应明显 |
| 6. + DyQuant (4/0) | **0.0656s** | **4.26×** | 完全跳过非关键专家效果最强 |

> 🔍 结论：**DyQuant + Prefetching + Cache** 三者协同带来最大收益，其中 `"4/0"` 策略在减少 I/O 方面最为激进且高效。

---

## 4. 关键结论和发现

### 主要发现
1. **专家重要性高度偏斜且动态变化**  
   少数“重击者 token”主导模型表现，且其对应的专家随输入内容动态转移 → 支持动态识别而非静态划分。

2. **深度依赖的量化敏感性存在显著差异**  
   - 浅层对量化噪声极度敏感（Int2 导致 accuracy 断崖式下降）
   - 深层具备强鲁棒性，可承受极端压缩（如 Int2 或跳过）
   → 支持 **depth-aware precision scheduling**

3. **相邻层激活高度相关，支持 lookahead 预测**  
   当前层隐藏状态可近似预测下一 layer 的 gate 分布 → 实现精准 prefetching，提升缓存命中率。

4. **动态混合精度优于静态压缩**  
   - `"4/0"` 在 Qwen3-30B-A3B 上甚至 **反超 Int4 基线精度**（如 GSM8K 达 91.74% vs 89.08%），可能起到正则化作用。
   - 用户可通过调节 `retention ratio r` 实现 **runtime 可调的精度-延迟权衡**。

### 方法的局限性
- **依赖 MoE 架构本身**：不适用于 Dense LLM。
- **预测误差风险**：lookahead 预测若不准可能导致预取浪费或缓存污染。
- **硬件依赖性强**：性能增益在 PCIe 带宽受限环境下最显著，在 NVLink 或片上存储架构中可能减弱。
- **未探索更多 bit-width 组合**：目前仅测试 INT4/INT2/"0-bit"，未涉及 FP8、NF4 等新兴格式。

### 未来工作方向
- 扩展至 **vision-language models** 中的 MoE 结构；
- 结合 **KV Cache 压缩技术**（如 PyramidInfer）进一步优化内存；
- 探索 **learnable gating + dynamic quantization 联合优化**；
- 在真实移动端芯片（如 Jetson Orin NX、手机 NPU）上部署验证能效比。

--- 

> 📌 **总体评价**：  
> DyMoE 是首个将 **动态专家重要性识别**、**深度感知混合精度量化** 与 **前瞻性预取机制** 深度融合的 MoE 边缘推理系统，在保持高精度的同时实现了数量级的延迟降低，为大模型边缘化提供了极具前景的技术路径。

</details>

---

### 4. [Accurate and Efficient Multi-Channel Time Series Forecasting via Sparse Attention Mechanism](https://arxiv.org/abs/2603.18712)

**Authors**: Lei Gao, Hengda Bao, Jingfei Fang, Guangzheng Wu, Weihua Zhou, Yun Zhou  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2603.18712v1  

#### Abstract
The task of multi-channel time series forecasting is ubiquitous in numerous fields such as finance, supply chain management, and energy planning. It is critical to effectively capture complex dynamic dependencies within and between channels for accurate predictions. However, traditional method paid ...

---

### 5. [Bridging Network Fragmentation: A Semantic-Augmented DRL Framework for UAV-aided VANETs](https://arxiv.org/abs/2603.18871)

**Authors**: Gaoxiang Cao, Wenke Yuan, Huasen He, Yunpeng Hou, Xiaofeng Jiang, Shuangwu Chen, Jian Yang  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.18871v1  

#### Abstract
Vehicular Ad-hoc Networks (VANETs) are the digital cornerstone of autonomous driving, yet they suffer from severe network fragmentation in urban environments due to physical obstructions. Unmanned Aerial Vehicles (UAVs), with their high mobility, have emerged as a vital solution to bridge these conn...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Bridging Network Fragmentation: A Semantic-Augmented DRL Framework for UAV-aided VANETs  
**——核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在城市环境中，由于建筑物遮挡和车辆高移动性，**VANETs**（Vehicular Ad-hoc Networks）常出现严重的**网络碎片化**（network fragmentation），导致通信中断、服务连续性差。传统基于**DRL**（Deep Reinforcement Learning）的**UAV部署策略**缺乏对道路拓扑结构的语义理解，导致探索盲目、样本效率低、泛化能力弱。

此外，现有方法多将城市区域建模为欧氏空间或栅格地图，忽略了车辆只能沿道路行驶这一**拓扑约束**，无法准确刻画网络连通状态。

---

### 🚀 提出的新方法与新思路

作者提出了一种全新的 **Semantic-Augmented DRL (SA-DRL)** 框架，其核心创新如下：

#### （1）**图论建模量化网络碎片**
- 构造 **Road Topology Graph (RTG)** 和 **Dual Connected Graph (DCG)**，将网络碎片化问题转化为**双图连通性优化问题**。
- 利用 DCG 的连通分量数量 $K(t)$ 和平均节点权重 $C(t)$ 来量化网络连通性，更符合实际中“边级连通”的特性。

#### （2）**四阶段 SA-DRL 框架**
构建了一个从环境经验到语义增强决策的完整流程：
1. **Experience Collection**：通过轻量级 PPO 收集代表性状态。
2. **Semantic Prior Construction**：将图状态序列化为文本，并利用即时奖励生成动作评分标签，构建监督微调（SFT）数据集。
3. **Knowledge Alignment**：采用 **LoRA** 对通用 **LLM** 进行参数高效微调，使其成为具备道路拓扑推理能力的“领域专家”。
4. **Semantic-Augmented Training & Execution**：提出 **SA-PPO** 算法，引入 **Logit Fusion** 机制，在策略输出层融合 LLM 的语义先验与 DRL 的实时决策。

#### （3）**Logit Fusion 机制**
- 将微调后的 LLM 输出作为 **semantic prior distribution**，与 PPO 的 logits 加权融合后生成最终策略。
- 实现了 LLM 高层语义推理与 DRL 数据驱动学习的深度协同，避免了仅用 LLM 做高层规划时的控制失配问题。

---

### 🔍 相比现有方法的优势

| 维度 | 传统 DRL 方法 | SA-DRL |
|------|----------------|--------|
| 探索效率 | 盲目探索，收敛慢 | 由 LLM 引导，聚焦关键路口 |
| 语义理解 | 无拓扑常识 | 注入道路拓扑“常识” |
| 泛化能力 | 对交通流变化敏感 | 能适应不同时段流量分布 |
| 能耗控制 | 易陷入局部最优或模式崩溃（Mode Collapse） | 实现“战略驻留”，节能高效 |
| 模型集成方式 | 多为串行（LLM → 规划 → 执行） | 并行双流融合，端到端训练 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用来自文献 [37] 的真实城市轨迹数据集，涵盖中国两个城市的交通监控视频提取的车辆轨迹。
- 实验选取深圳某区域子集：**47个节点（交叉口）**、**88条道路段**、**5,000条轨迹记录**，时间跨度为一天内多个时段。

### ⚙️ 实验设置
- **仿真平台**：自研 Python 模拟器，基于 IDM 模型模拟车辆移动。
- **UAV模型**：单架无人机，飞行高度固定，目标位置限定于交叉口上方。
- **通信模型**：采用概率 A2G 信道模型，考虑 LoS/NLoS 路径损耗。
- **任务周期**：离散时间槽 $T$，每个时隙执行一次决策。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Average Connected Block Size ($C(t)$)** | 每个连通子网中平均车辆数，越大越好 |
| **Number of Connected Components ($K(t)$)** | 网络碎片数量，越小越好 |
| **Average UAV Flight Distance** | 总飞行距离均值，反映能耗，越小越好 |
| **Training Episodes to Convergence** | 达到稳定性能所需训练轮次，越少越好 |
| **Generalization across Traffic Distributions** | 在未见时间段测试的鲁棒性 |

### 🆚 基线方法对比
- **Vanilla PPO**：标准 PPO，无任何结构增强
- **GAT-PPO**：使用 Graph Attention Network 提取图特征的 PPO 变体
- **SAC**（Soft Actor-Critic）：主流 Off-Policy 算法，强调熵最大化探索
- **SA-PPO (Ours)**：本文提出的方法

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（综合表现）

| 方法 | Avg. Connected Block Size | UAV 平均飞行距离 (m) | 训练收敛所需 episode 数 |
|------|----------------------------|------------------------|--------------------------|
| Vanilla PPO | 37.0 | 660.1 | 100% (基准) |
| GAT-PPO | 36.1 | 1158.2 | >100% |
| SAC | 45.7 | 45.7 | 未完全收敛（Mode Collapse） |
| **SA-PPO (ours)** | **41.9** (+13.2% ~ +23.5%) | **223.7** (**仅 28.2%**) | **26.6%** |

> ✅ **SA-PPO 在仅用 26.6% 的训练回合下即达到甚至超越基线性能，且能耗仅为 Vanilla PPO 的 28.2%。**

---

### 🔬 消融实验结果（Ablation Study）

比较以下变体：
- **w/o LLM (Vanilla PPO)**：移除语义分支
- **w/o Logit Fusion (Pure Semantic Policy)**：仅使用 LLM 输出决策，无 RL 微调

| 方法 | Connected Block Size | UAV 飞行距离 (m) |
|------|------------------------|--------------------|
| w/o LLM (Vanilla PPO) | 37.0 | 660.1 |
| w/o Logit Fusion | 39.8 | 980.6 |
| **SA-PPO (full)** | **41.9** | **223.7** |

> 🔍 发现：
> - 即使不经过 RL 训练，**纯 LLM 策略**也优于盲探 DRL，说明 LLM 成功学到了拓扑常识。
> - 但 LLM 容易“过度活跃”，频繁切换位置导致能耗极高。
> - **Logit Fusion 机制有效约束了 LLM 的行为**，实现“识别关键节点 + 战略驻留”的智能决策。

---

### 🔄 泛化性与鲁棒性测试

在不同时间段（08:00–18:00）进行跨分布测试：

| 场景 | SA-PPO 表现 |
|------|-------------|
| **稀疏流量（如 12:00）** | 凭借拓扑先验仍保持高连通性（50.3 vs Vanilla PPO 44.0） |
| **密集流量（如 16:00）** | 合理折衷，连接性能略低但能耗降低约 50%（471.3m vs 954.0m） |
| **动态调整能力** | 可根据需求自动调节飞行距离（152.1m ~ 471.3m） |

> ✅ 表明 SA-PPO 具备良好的**场景自适应能力**，能应对未知交通分布变化。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 可被有效转化为领域拓扑专家**：通过 LoRA 微调，通用 LLM 能掌握城市道路的结构性知识（如割点、枢纽节点的重要性）。
2. **语义先验显著提升 DRL 效率**：SA-PPO 仅需 **26.6% 的训练回合**即可收敛，极大缓解了样本低效问题。
3. **Logit Fusion 是高效融合范式**：相比奖励塑形或高层规划，直接在 logits 层注入 prior 更稳定、可控，避免 Reward Hacking。
4. **简单 MLP 优于复杂 GNN**：GAT-PPO 因对局部噪声敏感而表现更差，凸显了“常识引导”比“结构建模”更重要。
5. **实现节能与高性能的平衡**：SA-PPO 采取“战略驻留”而非盲目巡逻，大幅降低能耗的同时提升连通性。

---

### ⚠️ 方法的局限性
1. **依赖高质量状态序列化设计**：如何将图状态有效转换为 LLM 可理解的文本提示，仍需人工工程。
2. **LLM 推理延迟影响训练吞吐**：尽管采用批量并行系统缓解，LLM 推理仍是瓶颈，限制大规模部署。
3. **当前为单 UAV 设计**：虽可扩展至多机场景，但协作机制尚未验证。
4. **未考虑动态障碍物与空域冲突**：假设 UAV 可自由飞至任意交叉口，未涉及避障或空管约束。

---

### 🔮 未来工作方向
1. **自动化提示工程**：研究如何自动生成最优 instruction 和 state serialization 方式。
2. **端到端 LLM-DRL 联合训练**：探索 joint training 而非两阶段 pipeline。
3. **多 UAV 协同扩展**：将 SA-DRL 框架推广至 multi-agent setting。
4. **边缘部署优化**：压缩 LLM 或使用蒸馏技术，实现在 UAV 本地运行轻量语义模块。
5. **结合视觉感知输入**：从摄像头图像直接生成语义先验，减少对精确定位的依赖。

---

## 总结

> **SA-DRL 框架首次实现了 LLM 的语义推理能力与 DRL 决策能力在 UAV 辅助 VANET 中的深度融合。它不仅解决了传统方法“盲探低效、泛化差”的痛点，还通过 Logit Fusion 机制实现了节能高效的“战略部署”。实验表明，该方法在真实轨迹数据上取得了 SOTA 性能，是迈向智能、可持续空中通信中继的重要一步。**

</details>

---

### 6. [Retrieval-Augmented LLM Agents: Learning to Learn from Experience](https://arxiv.org/abs/2603.18272)

**Authors**: Thomas Palmeira Ferraz, Romain Deffayet, Vassilina Nikoulina, Herv\'e D\'ejean, St\'ephane Clinchant  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.18272v1  

#### Abstract
While large language models (LLMs) have advanced the development of general-purpose agents, achieving robust generalization to unseen tasks remains a significant challenge. Current approaches typically rely on either fine-tuning or training-free memory-augmented generation using retrieved experience...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Retrieval-Augmented LLM Agents: Learning to Learn from Experience

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在构建通用智能体方面取得了显著进展，但在面对训练期间未见过的新任务时，其泛化能力仍然有限。当前主流方法存在以下局限：
- **Fine-tuning**：虽然能提升在分布内任务上的表现，但难以外推到新任务，且容易过拟合。
- **Training-free memory-augmented generation**（如 RAG）：通过检索经验进行推理，但通常性能不如监督微调方法。

本文旨在解决如何让 LLM 智能体有效利用过往经验轨迹（experience trajectories），实现对**未见任务**的强泛化能力。

---

### 提出的新方法：ExPRAG 和 ExPRAG-LoRA

#### （1）ExPRAG（Experience Retrieval-Augmented Generation）
一种基于检索的经验增强生成框架，其核心思想是将智能体的历史交互轨迹作为“外部记忆”存储，并在推理时动态检索最相关的过去经历，注入提示（prompt）中以指导决策。

- **Experience Bank**：离线构建一个由智能体历史轨迹组成的索引库，每条轨迹被编码为嵌入向量（key embedding）。
- **Retrieval**：在推理阶段，将当前任务描述和交互历史编码为查询向量，从经验库中检索 Top-K 最相似的轨迹。
- **In-context Injection**：将检索到的轨迹格式化为记忆块（memory block），插入系统提示中，供 LLM policy 使用。

#### （2）ExPRAG-LoRA
提出将检索机制**集成到参数高效微调（PEFT）过程中**，即在 LoRA 微调时也使用检索增强的上下文进行训练。

- 在训练样本中加入从经验库中检索到的相关轨迹。
- 使模型学会在有检索支持的情况下做出决策，从而在测试时更有效地利用检索信息。

---

### 相比现有方法的优势
| 方法 | 优势 |
|------|------|
| **ExPRAG（仅推理时检索）** | 无需重新训练即可提升零样本性能；简单有效，优于许多复杂的 memory-augmented 方法 |
| **ExPRAG-LoRA（训练+推理都检索）** | 显著提升对未见任务（OOD）的泛化能力；避免标准 LoRA 的“分布外崩溃”现象；在多个基准上达到最优性能 |

> ✅ **核心洞见**：*仅仅在推理时使用检索不足以最大化其潜力，必须在训练阶段就教会模型如何使用检索到的经验。*

---

## 2. 核心实验方法和设置

### 数据集
- **ALFWorld**：模拟家庭环境中物体操作的任务（如“把蜡烛放进抽屉”），输出为二元成功/失败。
- **ScienceWorld**：科学实验类任务，对应小学科学课程，提供密集评分（[-1,1]），最终转换为二元成功/失败用于评估。

> ⚠️ 特别地，作者为了测试更强的**分布外泛化能力**，将任务分为：
> - **Easy tasks**：用于训练和构建经验库
> - **Hard tasks**：完全保留用于测试，确保真正的“未见任务”

---

### 实验设置
- **Backbone Models**：
  - `Ministral 3-8B`
  - `Gemma 3-4B`
  - `Qwen 2.5-7B` 和 `Qwen 2.5-7B-1M`（长上下文版本）
- **微调方法**：LoRA（低秩适配），目标模块为 `q_proj`, `v_proj`, `k_proj`, `output_proj`，rank=8, α=16
- **解码策略**：贪婪解码（greedy, temperature=0）
- **检索配置**：
  - 编码器：`Qwen3-Embedding-0.6B`
  - 相似度：点积（dot product）
  - Top-K：1, 2, 4
  - 检索模式：静态（仅在 t=0 检索一次） vs 动态（每步重检索）

---

### 评估指标
- **Success Rate**（成功率）：主要指标
- 分别报告：
  - **In-distribution (Easy tasks)**：已见任务
  - **Out-of-distribution (Hard tasks)**：未见任务
  - **All tasks**：综合表现

---

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **Zero-shot / Prompting** | Zero-shot, ReAct, ITP |
| **Training-Free Memory-Augmented** | Mem0, A-MEM, AgeMem-noRL, Reflexion |
| **Supervised Fine-tuned** | NAT+ReAct, IWM, ETO+ReAct, SAND, ITPR |
| **本工作提出的基线** | LoRA（强微调基线）、ExPRAG（仅推理检索） |

> 📌 表格 1 显示：作者提出的 LoRA 基线在 ALFWorld 上达到 **94.1%** 成功率，远超已有方法，说明建立强基线的重要性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

| Backbone | Method | ALFWorld (Hard, OOD) | ScienceWorld (Hard, OOD) |
|----------|--------|-----------------------|----------------------------|
| **Ministral 3-8B** | LoRA (no ExPRAG) | 34.4% | 15.6% |
| | LoRA + Inference-time ExPRAG | 67.2% | 28.1% |
| | **ExPRAG-LoRA** | **88.5%** | **42.2%** |
| **Qwen 2.5-7B** | LoRA (no ExPRAG) | 21.3% | 7.8% |
| | LoRA + Inference-time ExPRAG | 70.5% | 23.4% |
| | **ExPRAG-LoRA** | **90.2%** | **29.7%** |
| **Qwen 2.5-7B-1M** | LoRA (no ExPRAG) | 23.0% | 12.5% |
| | LoRA + Inference-time ExPRAG | 68.9% | 12.5% |
| | **ExPRAG-LoRA** | **91.8%** | **29.7%** |

> 🔥 **结论**：ExPRAG-LoRA 在所有模型上均显著优于其他方法，尤其在 OOD 任务上提升巨大（例如从 ~20% 提升至 >90%）。

---

### 与基线方法的对比结果
- 在 ALFWorld 上，**ExPRAG baseline（仅推理检索）** 达到 **83.6%**，已超过大多数已有 memory-augmented 方法（如 Reflexion: 42.7%，Memory Bank: 40.3%）。
- **ExPRAG-LoRA** 进一步将性能推向接近饱和水平（>90%），甚至超越规则专家（Rule-based Expert: 89.6%）。

---

### 消融实验结果

#### （1）Top-K 数量的影响（Table 2）
- 更多检索轨迹（K=4）通常更好，但存在“context rot”风险：
  - ALFWorld：K=4 明显优于 K=1
  - ScienceWorld：K=2 后增益趋于饱和，K=4 反而可能下降（尤其在 index 不匹配时）
- 长上下文模型（如 Qwen-1M）更能受益于更大的 K。

#### （2）检索模式：静态 vs 动态
- **静态检索**（Static）：更稳定，适合大多数场景
- **动态检索**（Dynamic）：在 index 匹配时可提升性能，但若 index 不匹配则不稳定，可能导致性能下降（因频繁更换上下文造成“context churn”）

#### （3）Index Composition
- 使用“all” index（包含 easy 和 hard 轨迹）通常是最佳折衷
- “easy-only” index 对 hard 任务帮助较小
- “hard-only” index 在某些情况下表现良好，但依赖高质量 hard 轨迹

#### （4）轨迹格式化方式（Table 16）
- **chat JSON** 是较优默认选择，尤其在 K 较大时更鲁棒
- 文本格式（textual）在部分小模型上尚可，但在 Qwen 上随 K 增加急剧退化
- 结论：**格式设计需与 backbone 匹配**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **简单的 episodic retrieval 是非常强大的基线**：无需复杂 memory controller 或 reflection 机制，仅靠检索完整轨迹即可大幅提升性能。
2. ✅ **训练时引入检索至关重要**：ExPRAG-LoRA 显著优于“先微调再检索”的两阶段方法，证明模型需要在训练中学习如何使用检索。
3. ✅ **延迟泛化现象**：即使验证损失开始上升，智能体在 OOD 任务上的性能仍可持续提升达 50 个 epoch，挑战传统早停准则。
4. ✅ **长上下文模型更能发挥多轨迹检索优势**：Qwen-1M 在 K=4 时表现更优，表明 context capacity 是瓶颈之一。

---

### 方法的局限性
1. **依赖高质量经验库**：
   - 当前使用脚本生成的“专家轨迹”，不反映真实 LLM 错误模式。
   - 若无相关轨迹可用，ExPRAG-LoRA 性能会大幅下降（Table 4）。
2. **计算开销增加**：
   - 检索和处理更多轨迹带来更高的延迟和内存消耗（Table 17）。
3. **固定只读记忆**：
   - 未支持在线更新、压缩或抽象记忆，限制长期适应能力。

---

### 未来工作方向
1. 探索基于 LLM 自身 rollouts 构建经验库，研究噪声轨迹下的鲁棒性。
2. 设计更高效的 memory management 机制，如摘要、技能提取、选择性写入等。
3. 研究如何在无匹配轨迹时仍保持稳健性能（zero-shot retrieval generalization）。
4. 将该范式扩展至多模态、具身智能体（embodied agents）等更复杂场景。

---

> 💡 **总体评价**：本文提出了一个简洁而强大的范式——**ExPRAG-LoRA**，强调“在训练中教模型学会使用经验”，为构建真正具备持续学习能力的 LLM 智能体提供了坚实基础。同时呼吁社区重视强基线建设，避免因弱基线导致虚假进步。

</details>

---

### 7. [Behavioral Fingerprints for LLM Endpoint Stability and Identity](https://arxiv.org/abs/2603.19022)

**Authors**: Jonah Leshin, Manish Shah, Ian Timmis, Daniel Kang  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.19022v1  

#### Abstract
The consistency of AI-native applications depends on the behavioral consistency of the model endpoints that power them. Traditional reliability metrics such as uptime, latency and throughput do not capture behavioral change, and an endpoint can remain "healthy" while its effective model identity cha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Behavioral Fingerprints for LLM Endpoint Stability and Identity 论文总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **LLM 服务端点（endpoint）的行为不稳定性** 问题展开研究。传统可靠性指标（如 uptime、latency、throughput）无法捕捉模型输出行为的变化，而实际中，即使接口“正常运行”，其背后的模型权重、tokenizer、推理引擎、量化策略、缓存机制或硬件等可能已发生变更，导致输出分布漂移（distribution shift），从而破坏下游应用（如 agent 工作流、安全护栏、解析器）。

作者指出：“**Reliability is not stability**”——系统可用且响应快，并不代表行为一致。

### 提出了什么新方法或新思路
提出两个核心组件：

- **Stability Monitor**：一个黑盒（black-box）稳定性监控系统，通过周期性地对固定 prompt 集进行采样生成输出指纹（fingerprint），并比较不同时间点的指纹来检测行为变化。
- **Stability Arena**：一个可视化 Web 应用（https://arena.projectvail.com），发布和展示各 LLM 端点的稳定性数据，支持跨提供商对比。

#### 核心技术流程：
1. **Fingerprint 构建**：使用一组固定的自然语言 prompts，每个 prompt 多次采样输出，将输出文本嵌入为向量（embedding vectors），形成“响应集合的集合”作为 fingerprint。
2. **Pairwise 比较**：使用 **energy distance** 统计量衡量两个 fingerprint 之间的分布差异；结合 **permutation test** 计算 p-value，判断是否存在显著变化。
3. **时序变化检测**：采用基于 **e-values** 的顺序证据累积机制，在连续监控中实现可选停（optional stopping）特性，及时识别 change event 并更新 baseline fingerprint。

### 相比现有方法的优势
| 方面 | 本文方法（Stability Monitor） | 现有方法（如 B3IT [3]） |
|------|-------------------------------|--------------------------|
| **Prompt 设计** | 固定、通用、模型无关的 prompt 集，无需初始化阶段 | 使用“border inputs”（边界输入），需在初始化阶段为每个 endpoint 定制探测样本 |
| **适应性** | change event 后仍可用同一 prompt 集继续监测 | change 后决策边界改变，“border inputs”可能失效，需重新发现 |
| **访问模式** | 完全 black-box，仅依赖标准 API 调用 | 同样 black-box，但 probe 更具攻击性 |
| **监控频率与成本** | 轻量级高频监控（每几小时一次），每次约 800 次 inference 请求 | 可能更高效 per-token，但依赖特定 probe 发现机制 |
| **统计框架** | 基于 energy distance + permutation test + e-values 的非参数序列检测框架，适合流式场景 | 类似假设检验框架，但未强调连续监控下的统计稳健性 |

> ✅ **优势总结**：更实用、可持续、免维护的长期行为监控方案，适用于真实生产环境中的多供应商比较与审计追踪。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **无公开 benchmark 数据集**，而是构建了一个 **自定义 prompt set** 用于 fingerprinting。
- Prompt 类型覆盖多样任务（如问答、指令遵循、格式化输出等），确保能激发模型多种行为。
- 每个 fingerprint 包含多个 prompt 上的多次采样结果（共约 800 次 inference 请求）。

### 实验设置和评估指标

#### 控制实验设置（Controlled Validation）
- 在本地部署可控 endpoint，运行 Stability Monitor 进行 hourly fingerprinting。
- 手动引入五类明确干预（见 Table 1），观察是否触发 change event：
  - Model family change （Qwen → Llama）
  - Version upgrade （Qwen2.5-0.5B → Qwen3-0.6B）
  - Inference stack change （vLLM → Transformers）
  - Quantization change （BF16 → INT8）
  - Temperature change （0.7 → 0.6）

#### 真实世界部署实验
- 对多个提供商提供的 **相同名义模型**（如 Kimi-K2-0905-Instruct）进行持续监控。
- 时间跨度：2025年11月至12月。
- 监控粒度：每几小时生成一次 fingerprint。

#### 评估指标
- **Change Event Detection Latency**：从变更发生到被检测出的时间（以 fingerprint 数量计）。
- **False Positive Rate / Stability Period Length**：在无变更期间是否误报 change event。
- **Energy Distance Matrix**：用于跨 provider 行为相似性分析。
- **Divergence Ratio**：单个 provider 相对于其他所有 provider 聚合分布的偏离程度（归一化后）。

#### 基线方法对比
- 主要与 **B3IT [3]** 对比设计理念与适用性，而非直接性能数字对比（因目标与设置不同）。
- 强调本方法无需 per-endpoint 初始化、更具普适性和可持续性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 变更类型 | 是否检测到 change event？ | 检测延迟（fingerprint 数） |
|--------|--------------------------|-----------------------------|
| Model family change | 是 | 1（下一轮即触发） |
| Version upgrade | 是 | 1 |
| Inference stack change | 是 | 1 |
| Quantization change | 是 | 1 |
| Small temp change (0.7→0.6) | 是 | 18 |

> ⚠️ 注：除微小 temperature 调整外，其余所有变更均在下一 fingerprint 周期立即被检测到。

- **Change Event 特性**：每次干预只触发 **单一 change event**，之后行为稳定（相对于新的 baseline），说明系统具备良好的 reset 和持续监控能力。

### 与基线方法的对比结果
- 未提供定量性能对比表，但从设计角度论证了相比 B3IT 的优势：
  - 不需要定制 probe，避免 change 后失效问题；
  - 更易于部署于大规模多 endpoint 场景；
  - 使用自然语言 prompts，更贴近真实用户行为。

### 消融实验结果（如有）
- 文中未报告显式的消融实验（ablation study），但通过控制变量实验验证了各类变更的影响独立性（each change was evaluated on its own）。

---

## 4. 关键结论和发现

### 论文的主要发现
1. 🔍 **LLM 端点行为会频繁变化**，即使“健康检查”通过，其输出分布也可能已发生显著偏移。
2. 🔄 **Change events 可被有效检测**：Stability Monitor 成功识别了模型家族、版本、推理栈、量化方式等多种变更。
3. 💥 **不同提供商间存在巨大稳定性差异**：
   - 同一名义模型（如 Kimi-K2）由不同 provider 提供时，行为显著不同；
   - 某些 provider（如 DeepInfra）极不稳定，几乎每次 fingerprint 都触发 change event；
   - 模型创建者自身托管的 endpoint（如 Moonshot）则表现出 100% 稳定性。
4. 📊 **“Same model, different behavior” 是现实运营常态**：由于底层 infra 差异（hardware、caching、routing、batching 等），即使是同一模型也会呈现不同行为特征。
5. 🛠️ **真实事件验证**：2025年12月，Parasail 的 change event 被确认是由物理节点故障引发的硬件提供商切换所致，证明系统具有实际诊断价值。

### 方法的局限性
- **基础设施引起的随机性难以区分**：某些系统级因素（如 batch size 变化、non-batch-invariant kernels）会导致持续的轻微行为波动，使得“change event”与“固有不稳定性”之间界限模糊。
- **依赖 embedding quality**：fingerprint 效果受限于文本 embedding 的表达能力和语义敏感度。
- **prompt set 的代表性**：当前 prompt 集虽多样化，但仍可能遗漏某些行为维度。
- **无法定位变更原因**：只能检测“是否变化”，不能自动判断是 weight 更新还是 tokenizer 修改所致。

### 未来工作方向
- 扩展 fingerprint 方法至多模态模型（multimodal models）。
- 探索更细粒度的行为分解（如按功能模块 fingerprint）。
- 结合 domain-specific evaluation（如 code generation accuracy）增强解释性。
- 开发自动化根因分析（root cause analysis）工具，辅助判断 change event 的来源。
- 推动行业建立 **behavioral SLA**（行为服务水平协议），将稳定性纳入 API 合同条款。

---

> ✅ **总体评价**：  
> 本文提出了一个实用、轻量、可扩展的 LLM 端点行为稳定性监控框架，填补了传统运维指标与实际行为一致性之间的鸿沟。其最大的价值在于揭示了“同一个模型在不同 provider 下行为迥异”的普遍现象，对 AI 应用开发、安全合规、基准测试等领域具有重要警示意义。

</details>

---

### 8. [From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning](https://arxiv.org/abs/2603.18383)

**Authors**: Grant Wilkins, Fiodar Kazhamiaka, Ram Rajagopal  
**Category**: cs.DC  
**Published**: 2026-03-20  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.18383v1  

#### Abstract
Datacenter operators and electrical utilities rely on power traces at different spatiotemporal scales. Operators use fine-grained traces for provisioning, facility management, and scheduling, while utilities use site-level load profiles for capacity and interconnection planning. Existing datacenter ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代大型语言模型（LLM）推理工作负载具有高度动态的功耗特性，在 **prefill**（计算密集）、**decode**（内存密集）和 **idle** 状态之间快速切换。这种动态性导致数据中心设施级电力需求在亚秒级时间尺度上剧烈波动。

然而，现有的数据中心电力建模方法存在以下不足：
- **基于 TDP（nameplate）的静态假设**：高估功耗，导致过度规划。
- **基于平均功率的平滑模型**：忽略瞬态峰值和突发行为，低估实际负载波动。
- **相位查表法（Phase-based LUT）**：无法捕捉连续批处理（continuous batching）下混合状态的中间功耗水平。
- **实测轨迹回放**：无法泛化到未见过的流量模式或部署配置。

这些问题使得传统模型难以支持从服务器级调度到电网级互联规划的多层级基础设施决策。

### 提出了什么新方法或新思路
本文提出了一种**组合式（compositional）电力轨迹生成框架**，其核心思想是将 LLM 推理功耗分解为两个独立但可组合的部分：

1. **工作负载驱动的状态转移**  
   使用请求并发数 $A_t$ 和其变化量 $\Delta A_t$ 作为特征，通过一个 **BiGRU 分类器**预测每个时刻的隐含操作状态（如 idle, prefill-heavy, decode-dominant 等）。

2. **配置相关的状态内功耗分布**  
   对每个硬件-模型-并行度配置 $(H, M, TP)$，使用 **Gaussian Mixture Model (GMM)** 学习不同状态下 GPU 功耗的概率分布；对于 MoE 模型，进一步引入 **AR(1) 模型**以保留时间相关性。

该框架允许：
- 从少量测量数据中学习模型参数；
- 合成任意新流量条件下的电力轨迹；
- 从 GPU 服务器逐层聚合至机架（rack）、行（row）、站点（site）级别的负载曲线。

### 相比现有方法的优势
| 方法 | 缺陷 | 本方法优势 |
|------|------|------------|
| TDP/nameplate | 过于保守，严重高估功耗 | 显著降低峰值估计，释放规划空间 |
| 平均功率模型 | 忽略所有时间动态 | 完整保留 burst、ramp、autocorrelation 结构 |
| Phase-based LUT | 固定相位划分，无法表示中间状态 | 支持细粒度、连续变化的功耗建模 |
| 轨迹回放 | 无法外推至新场景 | 可泛化至未见流量、新配置 |

此外，该方法实现了 **planner-facing interface**，支持下游的数据中心运营分析与电网互联研究。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **硬件平台**：Microsoft Azure 上的 NVIDIA DGX 服务器，配备 A100 (80GB) 和 H100 (80GB) GPU。
- **LLM 模型**：
  - Dense models: Llama-3.1 (8B, 70B, 405B), DeepSeek-R1-Distill (8B, 70B)
  - MoE models: gpt-oss (20B, 120B)
- **服务系统**：vLLM（版本 0.10.0），启用 continuous batching。
- **测量频率**：`nvidia-smi` 采样间隔为 **250ms**。
- **训练流量**：7 种到达率（0.125 ~ 4 req/s），每种重复 5 次，共约 10 分钟/次。
- **验证流量**：来自 Microsoft 发布的生产级 LLM 推理 trace（Azure coding activity trace, 5/16/24）。

### 实验设置和评估指标

#### 输入配置
- **设施拓扑**：支持自定义 rows → racks → servers 层级结构。
- **服务器配置**：GPU 类型、模型大小、tensor parallelism (TP) 度。
- **工作负载场景**：请求到达过程（如 Poisson）、prompt/output 长度分布。
- **站点假设**：非 GPU IT 功耗（默认 1kW/server）、PUE（默认 1.3）。

#### 输出
- 每台服务器的 **250ms 粒度电力轨迹**，可聚合至更高层级。

#### 评估指标
| 指标 | 描述 |
|------|------|
| **ACF R²** | 自相关函数拟合优度，衡量时间结构保真度 |
| **△Energy (%)** | 总能耗相对误差（中位数） |
| **NRMSE** | 归一化均方根误差 |
| **KS Statistic** | Kolmogorov-Smirnov 检验，比较功率分布相似性 |

测试集为 held-out 测量轨迹，每个生成 5 条合成轨迹取中位性能。

### 基线方法对比
1. **TDP (Nameplate)**：始终使用满额定功率。
2. **Mean Power**：使用训练集平均功率。
3. **LUT-based (Splitwise-inspired)**：基于 batch token 数判断 prefill/decode/mixed phase，并查表赋值功耗比例。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）
对多种 dense 和 MoE 模型进行评估，结果如下：

| Model | △Energy ↓ | ACF R² ↑ | NRMSE ↓ | KS ↓ |
|-------|-----------|----------|---------|------|
| Llama-3.1 (8B) | **0.6±0.3%** | **1.00** | 0.33 | 0.18 |
| Llama-3.1 (70B) | 4.0±2.7% | 0.97 | 0.43 | 0.22 |
| DeepSeek-R1-Distill (8B) | 1.0±0.4% | 0.99 | 0.32 | 0.21 |
| gpt-oss (120B) | 10.8±3.5% | 0.58 | 0.33 | 0.51 |

> ✅ **结论**：对于 dense 模型，**中位能耗误差 <5%**，且 **ACF R² > 0.96**，表明时间结构高度还原；MoE 模型因专家路由引入额外变异性，性能稍低但仍优于基线。

### 与基线方法的对比（见 Table 2 & Figure 8）

#### 服务器级对比（Llama-3.1-70B, A100, TP=8）
| Method | △Energy | NRMSE | ACF R² |
|--------|--------|--------|--------|
| TDP | +243.6% | 1.66 | — |
| Mean | +17.35% | 0.32 | — |
| LUT-based | +13.71% | 0.27 | 0.56 |
| **Ours** | **+6.09%** | **0.27** | **0.99** |

> ✅ **我们的方法显著优于所有基线**，尤其在时间结构建模方面远超 LUT 方法。

#### 设施级对比（24小时模拟，240 servers）
| Metric | TDP | Mean | LUT-Based | **Ours** |
|--------|-----|------|-----------|--------|
| Peak facility power (MW) | 1.19 | 0.85 | 0.82 | **0.75** |
| Avg. power (MW) | 1.19 | 0.85 | 0.76 | **0.63** |
| Peak-to-average ratio | 1.00 | 1.00 | 1.09 | **1.19** |
| Max ramp rate (MW/15min) | 0.00 | 0.00 | 0.07 | **0.11** |

> ✅ **TDP 高估容量需求达 60%**；而我们的方法能准确捕获 **ramping 行为** 和 **peak coincidence**，这对电网规划至关重要。

### Oversubscription 分析（见 Figure 11）
在一个 600kW 行配电限制下：
- **TDP 规划**：仅允许部署 **23 个机架**
- **我们的方法**：可安全部署 **57 个机架**（超过 **2.5 倍密度**）

原因在于服务器级负载峰不同时发生，聚合后平滑。这揭示了巨大的 **oversubscription headroom**，而静态模型完全无法利用。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **LLM 推理功耗具有强结构性**，可通过 $A_t$ 和 $\Delta A_t$ 有效建模。
2. ✅ **组合式建模实现高保真轨迹生成**：dense 模型能量误差 <5%，时间结构高度一致。
3. ✅ **真实轨迹显著影响基础设施决策**：
   - TDP 导致 **过度规划约 60%**
   - 可实现 **2.5 倍以上的机架过订用（oversubscription）**
   - 准确刻画 **ramp rate** 和 **peak coincidence** 对电网接入至关重要
4. ✅ **支持反事实分析（counterfactual analysis）**：可在部署前评估不同流量、模型混合、硬件升级的影响。

### 方法的局限性
1. **对 arrival process 的依赖**：目前主要验证于 Poisson 和生产 trace，未覆盖所有分布类型。
2. **恒定 PUE 和非 GPU 开销**：未建模温度依赖的冷却效率变化或动态辅助系统功耗。
3. **MoE 模型建模挑战**：专家路由导致的状态内变异难以仅由 $A_t$ 特征解释，AR(1) 仅部分缓解。
4. **调度细节抽象化**：使用简化吞吐代理模型（throughput surrogate），未考虑内存感知批处理、抢占等复杂机制。
5. **跨 serving engine 泛化性有限**：分类器基于 vLLM 训练，其他引擎可能需重新校准。

### 未来工作方向
1. **扩展至 agentive workloads**：支持工具调用、多轮推理等自相关请求流。
2. **长期负荷预测**：结合年尺度增长趋势，生成年度 load shape 用于资源充足性研究。
3. **负载灵活性量化**：评估通过排队、路由、延迟容忍任务调度实现的 demand shaping 潜力。
4. **隐私保护接口设计**：探索如何在不泄露内部 telemetry 的前提下向电网共享合成负载特征。
5. **集成更精细的热-PUE 模型**：联合建模功耗与散热动态，提升能效分析精度。

---

> 🔚 **总结**：本文提出的 compositional power trace generation 框架填补了 LLM 推理动态性与基础设施规划之间的鸿沟，提供了一个**开放、可复现、可扩展**的工具链，使数据中心运营商和电网规划者能够在真实动态下做出更精准、更高效的决策。

</details>

---

### 9. [A Family of Adaptive Activation Functions for Mitigating Failure Modes in Physics-Informed Neural Networks](https://arxiv.org/abs/2603.18328)

**Authors**: Krishna Murari  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 8.5  
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
Physics-Informed Neural Networks (PINNs) 在求解偏微分方程（PDEs）时面临多种**失败模式（failure modes）**，尤其是在处理具有**高频振荡、多尺度特征或强对流项**的问题时表现不佳。传统激活函数（如 `tanh`、ReLU）在梯度传播、表达能力和训练稳定性方面存在局限，导致模型难以准确捕捉复杂物理行为。

此外，标准 PINNs 经常出现**梯度不平衡**、**收敛缓慢**以及**时间演化过程中精度下降**等问题。

### 提出了什么新方法或新思路
本文提出了一类**基于小波（wavelet-based）的自适应激活函数家族**，通过将可训练的小波函数与非线性激活函数结合，显著提升 PINNs 的建模能力。具体设计如下：

- 构造了五种新型激活函数：
  - `SoftMexTanh`
  - `SoftMorTanh`
  - `SoftGaussTanh`
  - `SoftGaborTanh`
  - `SoftHerTanh`

这些函数由三部分组成：
1. **可学习参数化的小波或类小波函数**（如 Mexican hat、Morlet、Gabor 等），用于捕捉局部频率和空间特征；
2. **双曲正切函数 `tanh(βx)`**，作为主非线性组件；
3. **softplus 函数** 对所有可调参数进行约束，确保其为正值且可导。

所有参数均为**可训练（trainable）**，在优化过程中动态调整，实现真正的“自适应”。

还引入了对应的**消融变体**（ablation variants）——`SoftMexTanhW` 等，其中 `tanh` 的缩放系数 β 固定，仅小波部分参数可学。

### 相比现有方法的优势
- ✅ 显著提升了 PINNs 的**表达能力与训练稳定性**，尤其适用于高频率、快速变化的 PDE 解；
- ✅ 小波结构天然具备**多尺度分析能力**，更契合物理系统的内在特性；
- ✅ 自适应机制增强了模型对不同问题的泛化能力；
- ✅ 相较于 Transformer 类架构（如 PINNsFormer）、Mamba 结构（如 PINN-Mamba）等复杂模型，该方法保持了 PINN 的**简洁性和低计算开销**；
- ✅ 实验表明，在多个典型 PDE 上均优于标准 PINNs 及其他先进方法（如 QRes、FLS、ML-PINN）。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
研究并未使用真实世界数据集，而是选取四类典型的**基准 PDE 问题**作为测试平台，涵盖不同物理性质：

| 方程类型 | 特点 |
|--------|------|
| **1D Reaction Equation** | 非线性反应项，周期边界条件 |
| **1D Wave Equation** | 双曲型，高频振荡解 |
| **1D Convection Equation** | 强对流主导，高波数 β'=50，挑战性强 |
| **2D Navier-Stokes Equations** | 不可压缩流体系统，无解析解，参考数值模拟结果 |

> 所有实验设置均参照已有文献 [14, 24, 37, 57]，保证公平比较。

### 实验设置和评估指标

#### 模型配置
- **网络结构**：4 层隐藏层，每层宽度 512；
- **优化器**：L-BFGS（强 Wolfe 线搜索），迭代 1000 次；
- **损失权重**：λ_R = λ_B = λ_I = 1；
- **随机种子固定为 5**，确保可复现性；
- **硬件环境**：单块 NVIDIA A100 GPU，内存 32GB。

#### 评估指标
采用两个相对误差指标衡量预测精度：
- **rMAE**（Relative Mean Absolute Error）
- **rRMSE**（Relative Root Mean Square Error）

同时报告总训练 loss 和可视化误差图（absolute error maps）。

### 基线方法对比
- **Baseline PINNs**：使用原始 `tanh` 激活函数 [Raissi et al., 2019]
- **PINNsFormer** [Zhao et al., 2024]：基于 Transformer 的 PINN 架构
- **QRes** [Bu & Karpatne, 2021] 和 **FLS** [Wong et al., 2022]：改进型神经网络结构
- **ML-PINN** [Gao et al., 2025] 和 **PINN-Mamba** [Xu et al., 2025]：基于 Mamba/LSTM 的高效架构

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 2–6）

#### 📊 1D Reaction Equation
| 方法 | rMAE ↓ | rRMSE ↓ | Loss ↓ |
|------|-------|--------|--------|
| Tanh [37] | 0.975 | — | 0.199 |
| **SoftGaborTanh** | **0.0021** | **0.0021** | **7.00e-7** |
| SoftHer3Tanh | 0.010 | — | 9.16e-6 |
| SoftGaussTanh | 0.032 | — | 3.92e-6 |

> ✅ `SoftGaborTanh` 错误降低超过 **99%**！

#### 📊 1D Wave Equation
| 方法 | rRMSE ↓ | Loss ↓ |
|------|--------|--------|
| Tanh | 0.223 | 0.214 |
| **SoftHer2Tanh** | **0.019** | **0.018** |
| SoftGaborTanh | 0.031 | 0.029 |

> ✅ 最佳方法将 rRMSE 从 ~0.22 降至 ~0.019，提升约 **91%**

#### 📊 1D Convection Equation (β=50)
| 方法 | rMAE ↓ | rRMSE ↓ |
|------|--------|--------|
| Tanh | 0.724 | 0.796 |
| **SoftGaborTanhW** | **0.00606** | **0.0068** |

> ⚠️ 注意此处使用的是带后缀 `W` 的版本（即 tanh 缩放固定），说明某些任务中固定 β 更稳定。

#### 📊 2D Navier-Stokes Equation
| 方法 | rMAE ↓ | rRMSE ↓ | Loss ↓ |
|------|--------|--------|--------|
| Tanh | 17.80 | 12.35 | 1.101e-4 |
| **SoftGaborTanh** | **1.547** | **1.084** | **1.04e-6** |
| **SoftGaborTanhW** | **0.835** | **0.590** | 6.17e-6 |

> ✅ `SoftGaborTanhW` 表现最佳，rRMSE 下降超 **95%**

### 与基线方法的对比结果
- 在所有四个 PDE 测试中，提出的激活函数均**显著优于标准 PINNs (tanh)**；
- 性能全面超越 **QRes、FLS、PINNsFormer、ML-PINN、PINN-Mamba** 等先进方法；
- 虽然 PINNsFormer 和 ML-PINN 具备更强建模能力，但其**训练速度慢、显存占用高**；
- 本方法在 **训练效率上更具优势**：
  - 标准 PINN-Tanh：14.03 it/s
  - PINNsFormer：6.54 it/s
  - **PINN-SoftGaborTanh**：**9.01 it/s**

> 💡 在精度与效率之间取得更好平衡。

### 消融实验结果
- **是否让 `tanh` 参数可训练？**
  - 多数情况下允许 `β` 可学效果更好；
  - 但在 **convection equation** 中，固定 `β`（即使用 `...TanhW`）反而更鲁棒（见 Remark 5.7）；
- **不同小波形式的影响**：
  - `Gabor` 和 `Mexican hat` 在多数任务中表现优异；
  - `Hermite` 高阶多项式适合特定频谱结构；
- **bar plot 分析**（Figures 31–32）直观显示：
  - 所有新激活函数均明显优于 `tanh`；
  - `SoftGaborTanh` 和 `SoftMexTanh` 是最稳定的赢家。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **传统激活函数是限制 PINNs 性能的关键瓶颈之一**，特别是在高频或多尺度问题中；
2. **引入可训练的小波激活函数能有效缓解 PINNs 的失败模式**，极大提升其逼近能力和鲁棒性；
3. **softplus 参数化策略保障了训练稳定性**，避免参数发散；
4. **所提方法简单、通用、无需修改网络结构**，易于集成到现有 PINN 框架中；
5. **在多个经典 PDE 上实现了数量级级别的误差下降**，验证了其有效性与普适性。

### 方法的局限性
- 当前仅在中小规模 PDE 上验证，尚未扩展至三维或极端大规模问题；
- 小波函数的选择仍依赖经验，缺乏自动选择机制；
- 对于某些问题（如 convection），需手动切换是否启用 `tanh` 参数学习；
- 所有实验基于 L-BFGS，未充分探索 Adam 或混合优化策略的效果。

### 未来工作方向
- 探索更多类型的 wavelet-inspired 激活函数；
- 引入 **hybrid Adam-L-BFGS 优化策略** 加速收敛；
- 开发适用于 CPU 架构的轻量化版本，推动实际工程部署；
- 将该思想推广至 **Neural Operators、Transformer-based PINNs** 等更复杂框架中；
- 研究如何自动化选择最优激活函数组合（activation routing）。

---

> ✅ **总体评价**：这是一篇极具实践价值的工作，通过巧妙地融合小波理论与深度学习中的 activation design，为解决 PINNs 的根本性缺陷提供了新路径。其方法简洁而强大，有望成为未来 PINN 改进的标准组件之一。

</details>

---

### 10. [A Computationally Efficient Learning of Artificial Intelligence System Reliability Considering Error Propagation](https://arxiv.org/abs/2603.18201)

**Authors**: Fenglian Pan, Yinwei Zhang, Yili Hong, Larry Head, Jian Liu  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.18201v1  

#### Abstract
Artificial Intelligence (AI) systems are increasingly prominent in emerging smart cities, yet their reliability remains a critical concern. These systems typically operate through a sequence of interconnected functional stages, where upstream errors may propagate to downstream stages, ultimately aff...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Computationally Efficient Learning of Artificial Intelligence System Reliability Considering Error Propagation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **AI系统可靠性建模中的三大挑战** 展开研究：
1. **Data Challenge（数据挑战）**：真实世界中AI系统的可靠性数据稀缺，尤其是罕见故障事件（如自动驾驶车辆disengagement），难以通过现实测试大规模获取。
2. **Model Challenge（模型挑战）**：传统模型忽略 **Error Propagation (EP)** ——即上游模块的错误会传播至下游模块，导致系统级失效。这种传播是**隐式的（latent）、概率性的（probabilistic）且跨阶段依赖的**，违反了统计独立性假设。
3. **Inference Challenge（推断挑战）**：在大规模、高频率的递归错误事件数据下，传统的 **Expectation-Maximization (EM) 算法计算复杂度高（O(N²)）**，难以高效估计参数。

### 提出的新方法与创新思路
为应对上述挑战，论文提出了一套完整的框架，其核心创新包括：

#### （1）基于物理仿真的系统性数据生成框架
- 构建了一个**可扩展的、基于物理的自动驾驶仿真平台**（如CARLA等），并集成**可控的Error Injector (EI)**。
- EI可在指定时间戳以指定概率向特定模块注入错误（如图像加噪、点云稀疏化），从而**系统性地生成带有EP标签的高质量可靠性数据**。
- 该框架桥接了仿真与现实之间的鸿沟，提升了数据的真实性和可解释性。

#### （2）强度分解（Intensity-Decomposition）的EP建模方法
- 明确区分两类错误：
  - **Primary Error**：由模块自身缺陷引起的错误。
  - **Propagated Error**：由上游错误输入引发的错误。
- 将模块的总错误强度（intensity）分解为两部分：
  $$
  \lambda_{ms}(t) = \lambda^0_{ms} + \sum_{m_{s-1}} \alpha_{ms,m_{s-1}} \cdot \exp(-\beta_{ms,m_{s-1}}(t - t_{m_{s-1}}))
  $$
  - $\lambda^0_{ms}$：主错误强度（常数）
  - 第二项：来自前一阶段各模块的传播错误强度，采用**指数衰减核函数**建模传播效应随时间减弱的特性。

#### （3）复合似然EM算法（Composite Likelihood EM, CLEM）
- 针对传统EM算法计算效率低的问题，提出**CLEM算法**。
- 将观测窗口划分为多个子窗口（sub-windows），在每个子窗口内构建**局部复合似然（composite likelihood）**，仅考虑窗口内的EP关系。
- 显著降低E-step的计算复杂度，从 $O(N^2)$ 降至 $O(\sum_k N_k^2)$，其中 $N_k \ll N$。
- 引入**逐步Friedman检验**自动选择最优子窗口数量 $K^*$，平衡精度与效率。

### 相比现有方法的优势
| 维度 | 现有方法局限 | 本文方法优势 |
|------|---------------|-------------|
| **数据** | 依赖真实测试或孤立模块数据，缺乏EP信息 | 仿真+EI生成带EP标签的系统级数据 |
| **模型** | 忽略EP或假设独立性（如NHPP/HPP） | 显式建模EP作为可量化的风险强度成分 |
| **推断** | MLE/EM计算昂贵，难处理大规模数据 | CLEM显著提升计算效率，具理论保证（ascent property, consistency） |

---

## 2. 核心实验方法和设置

### 数据集
- **非公开仿真数据**：基于物理的自动驾驶仿真平台（如CARLA）生成。
- **自定义错误注入场景**：模拟不同天气条件下的感知系统行为。
  - **Setting I（持续注入）**：在整个200秒内持续注入错误，模拟持续恶劣天气（晴、雪、雨、雾）。
  - **Setting II（间歇注入）**：仅在 `[50,100)` 和 `[150,200)` 时间段注入错误，模拟突发恶劣天气。

### 实验设置
- **采集数据类型**：2-D检测错误、3-D检测错误、定位错误的时间序列。
- **采样频率**：20 Hz，总时长200秒。
- **重复次数**：每种场景重复30次（R=30）。
- **训练/预测划分**：使用前180秒数据训练模型，预测 `[180, 200)` 区间的错误数量（ΔT=20）。

### 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **MRRMSE**（Mean Relative Root Mean Square Error） | 衡量参数估计准确性 | 用于数值仿真中评估CLEM与EM的估计偏差 |
| **MAE**（Mean Absolute Error） | $ \text{MAE} = \frac{1}{R}\sum_{r=1}^R |\hat{N} - N| $ | 评估未来错误数量的预测准确性 |
| **Computational Time** | 平均运行时间 | 评估算法计算效率 |

### 基线方法对比
| 基线模型 | 类型 | 参数形式 |
|---------|------|----------|
| **NHPP-Musa-Okumoto (MO)** | 非齐次泊松过程 | $\lambda(t) = \theta_2(1+\theta_2\theta_1 t)^{-1}$ |
| **NHPP-Gompertz** | 非齐次泊松过程 | $\lambda(t) = \theta_1 \theta_2^{\theta_3 t}$ |
| **HPP-Poisson** | 齐次泊松过程 | $\lambda(t) = \theta_1$（恒定强度） |

> 所有基线模型均未显式建模EP，仅拟合整体错误发生率。

---

## 3. 主要实验结果和性能指标

### 数值仿真结果（Numerical Case Study）

#### （1）估计准确性（MRRMSE）
- 当子窗口数 $K \leq 100$ 时，**CLEM的MRRMSE与标准EM几乎一致**，表明其在合理分块下能保持高估计精度。
- 当 $K=250$ 或 $500$ 时，MRRMSE显著上升，说明过度分割会导致信息丢失和估计偏移。

#### （2）计算效率
- **CLEM相比EM显著减少计算时间**，尤其在大数据规模下（T=5000）：
  - EM耗时约300秒；
  - CLEM ($K=50$) 耗时约50秒，**提速约6倍**。
- 计算时间随 $K$ 增大而下降，但存在精度-效率权衡。

#### （3）最优 $K$ 选择
- 使用**逐步Friedman检验**自动选择最优 $K^*$：
  - $T=500$: $K^*=10$
  - $T=1000$: $K^*=20$
  - $T=2500$: $K^*=50$
  - $T=5000$: $K^*=100$
- 发现：**最优子窗口长度 $d = T/K^* \approx 50$ 秒保持恒定**，表明性能更依赖于局部时间尺度而非总时长。

---

### 物理仿真案例研究（Physics-based Simulation）

#### （1）预测准确性（MAE）
- 在 **Setting I（持续注入）** 下，本文方法与基线模型表现相近，因错误模式稳定，传统模型也能捕捉趋势。
- 在 **Setting II（间歇注入）** 下，本文方法**显著优于所有基线模型**：
  - MAE平均降低 **30%-50%**。
  - 例如，在“间歇雾天”场景下，本文方法MAE ≈ 1.2，而MO/Gompertz/Poisson分别为≈2.0/2.3/2.5。

#### （2）消融分析（隐含）
- 通过比较是否建模EP的效果，验证了**显式建模EP对提升预测精度至关重要**，尤其是在动态、非平稳环境下。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Error Propagation 是影响AI系统可靠性的关键因素**，必须被显式建模。
2. ✅ **CLEM算法在保持估计精度的同时，大幅提升了计算效率**，适用于大规模实时可靠性监控。
3. ✅ **最优子窗口长度约为50秒**，这一发现为实际部署提供了指导原则。
4. ✅ 所提方法在**间歇性、突发性错误场景下优势明显**，更适合真实世界的复杂驾驶环境。

### 方法的局限性
1. **依赖仿真数据**：尽管EI机制增强了真实性，但仍需在**真实道路测试数据上进一步验证**泛化能力。
2. **传播假设简化**：
   - 假设错误仅从前一阶段传播（Assumption 1）；
   - 平行模块间错误独立（Assumption 2）；
   - 实际中可能存在多跳传播或模块间耦合。
3. **未量化不确定性**：当前框架未提供参数或预测的置信区间。

### 未来工作方向
1. **拓展至真实运营数据**：将框架应用于真实AV车队的日志数据，验证其在多样化环境下的鲁棒性。
2. **引入不确定性量化与自适应学习**：结合Bayesian方法实现在线更新与可信度评估。
3. **推广至其他安全关键领域**：如医疗AI、工业自动化、能源系统等，构建通用的AI系统可靠性分析框架。
4. **建模更复杂的EP机制**：支持多跳传播、非指数衰减核、模块间交互效应等。

--- 

> **总结**：本文提出了一种**兼顾准确性与计算效率的AI系统可靠性建模新范式**，通过**仿真驱动的数据生成 + 强度分解的EP建模 + CLEM高效推断**，有效解决了EP建模中的三大瓶颈，为安全关键AI系统的可信评估提供了有力工具。

</details>

---

### 11. [Unmasking Algorithmic Bias in Predictive Policing: A GAN-Based Simulation Framework with Multi-City Temporal Analysis](https://arxiv.org/abs/2603.18987)

**Authors**: Pronob Kumar Barman, Pronoy Kumar Barman  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.18987v1  

#### Abstract
Predictive policing systems that direct patrol resources based on algorithmically generated crime forecasts have been widely deployed across US cities, yet their tendency to encode and amplify racial disparities remains poorly understood in quantitative terms. We present a reproducible simulation fr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Unmasking Algorithmic Bias in Predictive Policing: A GAN-Based Simulation Framework with Multi-City Temporal Analysis*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**预测性警务系统中算法偏见的量化与传播机制**。尽管预测性警务在全美多个城市广泛应用，但其如何通过历史犯罪数据编码并放大种族不平等，仍缺乏可复现、多城市、纵向的定量分析框架。

现有研究通常局限于单一城市、单一年份或仅使用聚合统计数据，难以建模从“犯罪发生”到“警察接触”的完整执法流程。本文填补了这一空白。

### 🚀 提出的新方法与创新思路
作者提出了一个**基于生成对抗网络（GAN）的仿真框架**，结合**Noisy-OR 检测模型**，用于模拟和测量算法偏见在整个执法管道中的传播过程。具体创新包括：

1. **GAN-based spatial patrol model**  
   使用 **Conditional Tabular GAN (CTGAN)** 学习真实犯罪事件的空间分布，并生成反映历史偏见模式的巡逻部署位置，从而再现现实世界中“过度执法区域”的集中趋势。

2. **Longitudinal, multi-city bias audit**  
   在 **Baltimore (2017–2019)** 和 **Chicago (2022)** 跨越 **264 个 city-year-mode 观测单元**进行月度级偏见审计，首次实现跨时间和城市的动态偏见追踪。

3. **集成多种公平性指标（fairness metrics）**  
   同时计算四种可解释的偏见度量：
   - **Disparate Impact Ratio (DIR)**
   - **Demographic Parity Gap**
   - **Gini Coefficient**
   - **Bias Amplification Score (BAS)**

4. **引入 debiasing 与政策干预联动分析**  
   探索了 CTGAN 数据再平衡策略的效果，并结合社会经济回归分析揭示结构性不平等的影响。

### 🔍 相比现有方法的优势
| 方面 | 传统方法局限 | 本论文优势 |
|------|---------------|------------|
| 时间维度 | 多为横截面分析 | 支持多年纵向比较（如 Baltimore 三年变化） |
| 空间建模 | 静态分配或简单聚类 | 使用 GAN 动态学习空间分布，更贴近实际巡逻模式 |
| 偏见传播路径 | 仅关注预测输出 | 建模“犯罪 → 巡逻 → 检测”全过程 |
| 可复现性 | 数据封闭、代码未公开 | 所有代码与数据公开，支持完全复现 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

| 数据类型 | 来源 | 描述 |
|--------|------|------|
| **Baltimore Part 1 Crime Data (2017–2019)** | Baltimore City Open Data Portal | 共计 145,823 条记录，包含 GPS 坐标、时间、案件类型等 |
| **Chicago Crime Data (2022)** | Chicago Data Portal | 共计 233,456 条记录，含社区区域标识（community area） |
| **U.S. Census ACS Demographic Data** | American Community Survey | 获取各 census tract 的种族构成（%Black, %White）、收入中位数、贫困率等 |
| **Citizen Reporting Rate** | Pew Research Center | 设定基础报案概率为 52.1% |

> 所有犯罪事件通过 point-in-polygon 匹配至对应 census tract，以获取人口统计协变量。

### ⚙️ 实验设置

#### 模拟架构
- **Generator**: 五层全连接网络，输入噪声向量 $ z \sim N(0,I) $，输出合成巡逻点坐标
- **Discriminator**: 四层网络判断位置是否来自真实数据
- **训练目标**: 标准 Minimax GAN loss
- **每步生成巡逻警力数量**: $ N_{officers} = 60 $
- **检测半径**: $ r = 700 $ ft
- **单警员检测概率**: $ p_j = 0.85 $

#### 两种模拟模式（simulation modes）
| 模式 | 描述 |
|------|------|
| **Detected Mode** | 巡逻点由 GAN 生成 → 表示算法驱动的资源分配 |
| **Reported Mode** | 巡逻响应公民报警（独立以 52.1% 概率触发）→ 表示基于公众报告的真实反馈 |

#### 总体实验设计
- 每月重新训练 GAN 并运行一次模拟
- 时间范围：每年 Feb–Dec（共 11 个月）
- 实验总数：$ 3 \text{年} \times 2 \text{模式} \times 11 \text{月} + 1 \text{年} \times 2 \text{模式} \times 11 \text{月} = 264 $ 次观测

### 📈 评估指标

| 指标 | 定义 | 判断标准 |
|------|------|----------|
| **Disparate Impact Ratio (DIR)** | $ \frac{P(\text{detected}|\text{Black})}{P(\text{detected}|\text{White})} $ | < 0.8 表示对 Black 居民欠检测；>1 表示过检测 |
| **Demographic Parity Gap** | $ P(\text{detected}|\text{Black}) - P(\text{detected}|\text{White}) $ | 正值表示 Black 更可能被检测 |
| **Gini Coefficient** | 检测率在不同群体间的不平等程度 | 越高表示不平等越严重（0.4以上即为高度不均） |
| **Bias Amplification Score (BAS)** | $ \text{Parity Gap} \times \text{Gini} $ | 综合衡量方向性和总体不公 |

### 🔁 基线方法对比
- **Detected Mode vs. Reported Mode**：作为内在对照组，验证算法驱动 vs. 社会反馈的不同偏见水平
- **Raw Training Data vs. CTGAN-debiased Data**：用于评估 debiasing 效果
- 无传统 ML 分类器基线，因研究重点在于系统性偏见而非预测准确率

---

## 3. 主要实验结果和性能指标

### 📌 关键性能数据汇总（见 Table 1）

| City | Year | Mode | Avg. DIR | Max. DIR | Avg. PG | Avg. Gini |
|------|------|------|---------|---------|--------|----------|
| Baltimore | 2017 | detected | 0.952 | 2.013 | -0.031 | 0.425 |
| Baltimore | 2018 | detected | 0.079 | 0.522 | -0.142 | 0.618 |
| Baltimore | 2019 | detected | **15,714** | 35,582 | **+0.016** | 0.553 |
| Chicago | 2022 | detected | 0.220 | 1.201 | -0.073 | 0.567 |
| Baltimore | 2019 | reported | 0.653 | 1.655 | -0.029 | 0.361 |
| Chicago | 2022 | reported | **1.218** | 2.694 | ~0.000 | 0.213 |

> 注：DIR > 10,000 是由于 White 居民检测率趋近于零所致

### 🔍 核心发现

#### （1）极端且年变的算法偏见
- **Baltimore 2019 detected mode** 出现 **平均 DIR 达 15,714**，是历史上罕见的高偏见值，主因是 GAN 将巡逻几乎全部集中在 Black-majority 区域，导致 White 区域检测率接近零。
- 相比之下，2017 年 DIR ≈ 0.95（轻微过检 Black），2018 年骤降至 0.079（严重欠检 Black），显示偏见具有显著**年度波动性**。

#### （2）Reported Mode 明显缓解偏见
- 在所有条件下，**reported mode 的 DIR 更稳定（0.61–1.22）且 Gini 更低（0.12–0.36）**
- 这表明公民报案提供了“ground truth”校正机制，打破算法自我强化的反馈循环

#### （3）Gini 系数持续高位
- 所有 detected mode 下的 Gini 系数均在 **0.43–0.62** 之间，属于高度不平等区间
- 即使 DIR 方向改变（如 Baltimore 2018 vs 2019），整体检测不平等依然顽固存在

#### （4）CTGAN Debiasing 实验结果（消融实验）

| 条件 | DIR | Black Det. Rate | White Det. Rate | Parity Gap |
|------|-----|------------------|------------------|-------------|
| Biased (原始) | 0.513 | 3.44% | 6.70% | -0.033 |
| Debiased (CTGAN) | **3.106** | **4.93%** | **1.59%** | **+0.033** |

> 结果显示：CTGAN 再平衡后，Black 检测率上升，但 White 检测率大幅下降，**偏见方向反转而非消除**

👉 **结论**：在固定警力资源下（60 名警官），任何提升某一族群检测率的做法都会牺牲另一族群——这是一种**零和博弈（zero-sum allocation）**

#### （5）敏感性分析（Sensitivity Analysis）

| 参数调整 | 对 DIR 影响 |
|--------|-----------|
| **Officer Count ↓ (60→30)** | DIR ↑↑↑（从 0.084 → 7.71） |
| **Patrol Radius ↑ (400→1500 ft)** | DIR ↑（单调递增） |
| **Reporting Probability** | 非单调关系，影响较小 |

> **最关键因素是 officer count**：警力越少，空间集中效应越强，偏见越剧烈

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **预测性警务系统会显著放大种族偏见**，尤其是在使用 GAN 类模型学习历史数据时，会导致极端 DIR 值（如 Baltimore 2019 的 15,714），远超“四分之五规则”阈值（0.8）。
   
2. **偏见具有强烈的时间变异性和城市特异性**：
   - Baltimore 三年间 DIR 从 0.079 到 15,714，说明不能假设模型偏见稳定
   - Chicago 2022 的 DIR = 0.22，表现为系统性欠检测 Black 社区，与 Baltimore 2019 完全相反

3. **公民报告机制能有效抑制算法反馈循环**，reported mode 下的偏见明显更低且更稳定，建议加强社区参与渠道建设。

4. **纯数据层面的 debiasing（如 CTGAN）无法根除结构性不平等**，反而可能导致偏见转移（bias shifting），在资源受限环境下形成零和博弈。

5. **社会经济因素与检测率强相关**：
   - Pearson 相关系数达 $ r = +0.83 $（%White）和 $ r = -0.81 $（%Black）
   - OLS 回归显示：社区 Black 人口比例每增加 1%，检测率下降约 0.097%

### ⚠️ 方法的局限性

1. **种族归属为区域级代理变量**：未使用个体数据，而是基于 census tract 的人口比例随机赋值，可能存在生态谬误（ecological fallacy）
2. **每月重训 GAN 不符合实际部署周期**：现实中模型更新频率较低，可能低估长期稳定性
3. **Noisy-OR 模型假设警员独立行动**：忽略协同巡逻行为，可能高估或低估检测概率
4. **未建模犯罪转移（crime displacement）**：集中巡逻可能导致犯罪迁移到邻近区域，影响长期反馈
5. **CTGAN debiasing 仅测试于 Baltimore 2019**：泛化能力有待验证

### 🔮 未来工作方向

1. 引入**因果推断模型**（如 counterfactual fairness, causal graphs）来区分相关性与因果性
2. 构建**多轮反馈模拟**，研究长期动态演化下的偏见累积机制
3. 探索**资源扩展与政策干预组合方案**（如增加总警力 + 公平约束优化）
4. 扩展至更多城市和国家，构建通用偏见审计平台
5. 结合**参与式治理框架**（participatory governance），将社区意见纳入模型设计

---

> 💡 **一句话总结**：  
> 本文揭示了 GAN 驱动的预测性警务不仅继承而且剧烈放大历史执法偏见，尤其在警力有限时形成零和博弈；唯有结合制度性改革、资源投入与公众监督，才能真正实现算法公平。

</details>

---

### 12. [D5P4: Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding](https://arxiv.org/abs/2603.19146)

**Authors**: Jonathan Lys, Vincent Gripon, Bastien Pasdeloup, Axel Marmoret, Lukas Mauch, Fabien Cardinaux, Ghouthi Boukli Hacene  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 8.0  
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
- **离散扩散模型**（Discrete Diffusion Models）在文本生成任务中展现出潜力，但其**解码机制仍不成熟**，缺乏对输出多样性的有效控制。
- 现有方法如 **beam search** 不适用于扩散模型的并行去噪过程；而强 **classifier-free guidance**（CFG）虽提升保真度，却导致**模式崩溃**（mode collapse），降低多样性。
- 当前的多样性促进策略多为启发式设计，缺乏理论支撑，且难以平衡**生成质量与多样性**。

### 🚀 提出的新方法：D5P4
- 提出 **D5P4**（Partition Determinantal Point Process for Diversity in Parallel Discrete Diffusion Decoding），一种面向离散扩散模型的新型解码框架。
- 将每一步候选序列的选择建模为 **Determinantal Point Process**（DPP）上的 **MAP 推断**，实现集合级别的选择而非独立打分。
- 引入 **partition constraint**，确保每个保留的 beam 来自不同祖先路径，防止“lineage collapse”。
- 构造两种形式的 DPP kernel：
  - **Additive**: `L = diag(Q) + βK`
  - **Multiplicative**: `L = diag(e^{Q/β}) · K · diag(e^{Q/β})`
  其中 `Q` 是质量得分（基于熵或 self-certainty），`K` 是语义相似性矩阵（基于隐藏状态嵌入）。

### 🔍 相比现有方法的优势
| 特性 | D5P4 | 传统方法（如 beam search、temperature sampling） |
|------|------|---------------------------------------------|
| 多样性控制 | 显式建模候选间交互，支持细粒度 trade-off | 隐式或无控制，易出现重复输出 |
| 质量保持 | 利用模型内部信号（entropy, embeddings），无需外部评分器 | 依赖外部 verifier 或简单缩放 |
| 可扩展性 | 使用 **greedy MAP solver**，GPU 并行友好，开销接近零 | 如 Particle Gibbs 等 MCMC 方法计算昂贵 |
| 结构兼容性 | 专为并行去噪设计，天然适配 MDLM / LLaDA 类架构 | 多为 AR 模型设计，无法直接迁移 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Open-ended Generation**:
  - **FineWeb**（Penedo et al., 2024）：用于 MDLM 模型的自由生成实验。
- **Question Answering**:
  - **TruthfulQA**（Lin et al., 2021）：测试事实性和抗幻觉能力。
  - **CommonSenseQA**（Talmor et al., 2019）：评估常识推理能力。

### ⚙️ 实验设置
- **模型**：
  - **MDLM**：基于掩码的离散扩散语言模型，固定长度生成。
  - **LLaDA**：大语言扩散模型（Large Language Diffusion Model），支持指令跟随。
- **解码参数**：
  - Beam 数量：`k=8`（open-ended）、`k=3`（QA）
  - 分支因子（w）：4 → 总候选池大小 `n=k×w`
  - 使用 **linear noise schedule**，从全掩码开始逐步去噪。
- **硬件**：多 GPU 设置下验证可扩展性。

### 📊 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **Quality** | PPL（Perplexity） | 使用 GPT-2 / LLaMA-3 作为外部评估器 |
|           | MAUVE / MAUVE* | 衡量生成分布与真实文本的相似性 |
|           | BLEU, F1-score | QA 任务中的准确率 |
|           | Wasserstein Distance | 度量答案簇间的语义偏移 |
| **Diversity** | COS（In-batch cosine similarity） | 基于 Jina embeddings 的语义多样性 |
|             | Self-BLEU | 跨样本的 n-gram 重叠程度 |
|             | Distinct-n, EAD | n-gram 唯一性及长度归一化版本 |

### 🆚 基线方法对比
#### 质量导向基线：
- **Best-of-n (Oversample)**：暴力采样后选最优
- **Standard Beam Search**：按组取最高分，忽略冗余

#### 多样性导向基线：
- **Diverse Beam Search (DivBS)**：基于 transversal MMR 的贪心选择
- **Standard DPP Sampling**：使用 DPPy 库进行无约束 DPP 抽样（仅作参考）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### （1）开放域生成（MDLM + FineWeb）
| 方法 | PPL ↓ | COS ↑（多样性） | MAUVE ↑ |
|------|-------|------------------|---------|
| Baseline（独立采样） | ~35 | 0.80 | ~0.85 |
| CAT（温度调节） | ~20 | 0.78 → 0.82（突变） | — |
| DivBS | ~22 | 0.76–0.79 | — |
| **D5P4+（additive）** | **~18** | **0.72–0.78（平滑过渡）** | **0.90** |
| **D5P4×（multiplicative）** | **~17** | **0.73–0.77** | **0.91** |

> ✅ **D5P4 在更低 PPL 下实现了更优的多样性-质量帕累托前沿**

#### （2）问答任务（LLaDA + TruthfulQA/CommonSenseQA）
| 方法 | PPL ↓ | F1-score ↔ | EAD ↑（多样性） | Self-BLEU ↓ |
|------|--------|------------|------------------|--------------|
| Best-of-k | 17.446 | 0.212 | 0.363 | 47.102 |
| D5P4+ | **15.725** | 0.184 | **0.385** | **40.404** |
| D5P4+ + P-CFG | 15.015 | 0.195 | **0.389** | 42.78 |

> ✅ 即使在高 CFG 设置下，D5P4 也能显著缓解多样性坍塌（见 Figure 4），同时维持竞争力的质量。

#### （3）子行列式最大化算法效率对比（Table 6）
| 方法 | 归一化目标值 ↑ | 运行时间（秒） ↓ |
|------|----------------|------------------|
| Random | -0.9074 | 0.0001 |
| DPP（CPU） | -0.8752 | 0.5478 |
| Diverse Beam Search | 0.6645 | 0.0295 |
| **Greedy MAP（ours）** | **1.0214** | **0.0023** |

> ✅ 我们的方法不仅目标值更高，而且速度是 Diverse Beam Search 的 **10 倍以上**，几乎无额外延迟。

---

### 🔍 消融实验结果

#### （1）质量估计方式对比（Figure 9）
| 方法 | 与 GPT-2 PPL 的相关性（负相关越好） |
|------|-------------------------------|
| Entropy-based scoring | **-0.776** |
| Self-certainty | -0.290 |

> ✅ **基于熵的质量评分** 更好地反映了外部评估器的判断。

#### （2）多样性表示构建方式（Table 4）
| Pooling 方法 | MDLM CKA ↑ | LLADA CKA ↑ |
|-------------|------------|------------|
| Mean | 0.777 | 0.482 |
| Non-masked | 0.660 | 0.536 |
| Masked | 0.710 | 0.435 |
| **Flatten（全文展平）** | **0.821** | **0.667** |

> ✅ **flatten embedding** 最佳保留了原始语义结构，被选为最终方案。

#### （3）多样性控制参数敏感性（Table 2）
| 方法 | 参数 | PPL 相关性 ↑ | COS 相关性 ↓ |
|------|------|---------------|---------------|
| CAT | log temp | 0.940 | -0.438 |
| DivBS | log α_div | 0.950 | -0.846 |
| D5P4+ | log β | 0.902 | **-0.875** |

> ✅ D5P4 对多样性控制更敏感且可控性强，失败模式更平缓。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **D5P4 实现了高质量与高多样性的统一**：
   - 在多个任务上优于标准 beam search 和多样性增强方法（如 DivBS）。
   - 支持通过单一参数 `β` 显式调节质量-多样性权衡。

2. **模型内部信号足以驱动多样性选择**：
   - 使用 **sequence-level entropy** 和 **hidden-state embeddings** 即可获得与外部评估器高度一致的结果（CKA > 0.8），无需引入额外 verifier。

3. **高效且可扩展**：
   - 所提出的 **greedy MAP solver** 在 GPU 上运行极快（毫秒级），适合大规模部署。
   - 支持跨 GPU 并行执行，契合现代训练基础设施。

4. **有效对抗 CFG 导致的多样性坍塌**：
   - 即使在强引导条件下，D5P4 仍能维持较高 lexical 和 semantic 多样性。

---

### ⚠️ 局限性
- **未解决根本性偏差问题**：增加多样性可能暴露更多训练数据中的偏见或有害内容，需配合安全过滤机制。
- **kernel 设计仍有改进空间**：当前使用简单的 cosine/RBF kernel，未来可探索任务感知 kernel。
- **仅限于 discrete diffusion models**：尚未验证在 continuous diffusion 或其他 modalities 上的表现。

---

### 🔮 未来工作方向
1. **扩展至多模态扩散模型**（如图像、音频）中的多样化生成。
2. **结合 test-time compute scaling**，进一步放大“generate-many-then-select”范式的效果。
3. **开发任务定制化 kernel**，例如在数学推理中鼓励逻辑路径多样性。
4. **集成 alignment 技术**，在提升多样性的同时保证安全性与可靠性。

---

> 💬 **总体评价**：  
> D5P4 是首个将 **DPP** 成功应用于 **discrete diffusion decoding** 的工作，提出了一种**理论严谨、工程高效、效果优越**的多样性解码框架。它不仅填补了该领域的算法空白，也为未来 diffusion LMs 的 test-time scaling 提供了实用路径。

</details>

---

### 13. [An Agentic System for Schema Aware NL2SQL Generation](https://arxiv.org/abs/2603.18018)

**Authors**: David Onyango, Naseef Mansoor  
**Category**: cs.CL  
**Published**: 2026-03-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.18018v1  

#### Abstract
The natural language to SQL (NL2SQL) task plays a pivotal role in democratizing data access by enabling non-expert users to interact with relational databases through intuitive language. While recent frameworks have enhanced translation accuracy via task specialization, their reliance on Large Langu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《An Agentic System for Schema Aware NL2SQL Generation》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **NL2SQL** 系统严重依赖 **Large Language Models (LLMs)**，导致以下三大挑战：
- **高计算成本**：LLMs 推理开销大，难以在资源受限环境中部署。
- **数据隐私风险**：企业敏感数据库需本地化处理，而云端 LLM 存在数据泄露隐患。
- **复杂 Schema 处理困难**：面对多表连接、领域约束等复杂查询时，易出现语义错位（如错误 JOIN、值不匹配）。

### 🚀 提出的新方法与创新思路
本文提出一种 **基于 SLM 的 Schema-Aware Agentic 系统**，其核心思想是：
- **以 Small Language Models (SLMs)** 作为主推理引擎，承担大部分常规查询生成任务；
- 引入 **智能 LLM 回退机制 (intelligent LLM fallback)**：仅当 SLM 输出失败或验证不通过时，才调用 LLM 进行纠错重试（最多三次）；
- 整个系统采用 **模块化多智能体架构**，将 NL2SQL 流程分解为四个协同代理（Agent）：
  1. **Extractor Agent**：结合 RAG 和向量检索提取 Schema 上下文；
  2. **Decomposer Agent**：进行实体识别、条件分析与子查询规划；
  3. **Generator Agent**：使用 SLM 初步生成 SQL，失败后触发 LLM 修复；
  4. **Validator & Executor Agent**：执行语法、语义及运行时验证，并反馈错误用于修正。

该设计实现了 **“SLM 为主 + LLM 按需调用”** 的混合范式。

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **成本效率** | 平均每查询成本从 $0.094（纯 LLM）降至 $0.0085，**降低超 90%**；约 67% 查询由本地 SLM 完全解决，接近零运营成本。 |
| **隐私安全** | 支持完全本地部署，敏感数据无需外传至云 LLM，满足企业合规要求。 |
| **可扩展性** | 减少对昂贵 LLM 的依赖，更适合大规模生产环境部署。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **BIRD Benchmark**：
  - 包含 12,751 条 text-to-SQL 配对样本；
  - 覆盖 95 个真实世界数据库，涉及体育、医疗、政治等 37 个专业领域；
  - 数据库规模大、Schema 复杂，强调外部知识映射与高效 SQL 生成。
- **本研究使用开发集**：共 1,534 条查询，在 11 个数据库上进行评估。

### ⚙️ 实验设置
- **实现框架**：基于 `LangGraph` 构建工作流，`LangChain` 管理通信与状态。
- **模型配置**：
  - **Embedding Model**：all-MiniLM-L6-v2（用于 RAG 检索）
  - **SLMs**：
    - Extractor / Decomposer：Mistral-7B
    - Generator：Llama-3.1-8B
  - **LLM Fallback**：GPT-4o（仅用于失败情况下的重生成）
- **存储系统**：Chroma DB 存储预计算的向量嵌入，提升检索速度。

### 📊 评估指标
| 指标 | 定义与意义 |
|------|-----------|
| **Execution Accuracy (EX)** | 执行结果是否与标准 SQL 完全一致。衡量功能正确性，允许结构不同但语义等价的 SQL。公式：<br>$$ \text{EX} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(V_i, \hat{V}_i) $$ |
| **Validation Efficiency Score (VES)** | 衡量有效查询的执行效率，综合考虑准确性和响应时间。<br>$$ \text{VES} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(V_i, \hat{V}_i) \cdot \sqrt{\frac{E(Y_i)}{E(\hat{Y}_i)}} $$<br>其中 $E(\cdot)$ 是实际执行耗时。 |

### 🔁 对比的基线方法
| 方法 | 特点 |
|------|------|
| **MAC-SQL** | 多智能体协作框架，执行精度最高（59.59%），全程依赖 GPT-4 |
| **DAIL-SQL** | 基于示例驱动的分解提示策略，利用密集检索引导生成 |
| **DIN-SQL** | 引入自校正机制，通过执行反馈迭代优化查询 |

所有基线均为 **LLM-centric** 设计，无 SLM 参与。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2 & 3）

| 方法 | Execution Accuracy (%) | Validation Efficiency Score (%) | Avg Cost per Query ($) |
|------|------------------------|-------------------------------|-------------------------|
| MAC-SQL | **59.59** | **67.68** | ~0.094 |
| DAIL-SQL | 57.41 | 61.95 | ~0.094 |
| DIN-SQL | 55.90 | 59.44 | ~0.094 |
| **Proposed Agentic System** | **47.78** | **51.05** | **~0.0085** |

### 🔍 对比结果分析
- **准确性方面**：提出的系统略低于纯 LLM 方法（低约 10–12 个百分点），但在 **成本控制上取得显著突破**。
- **成本效率方面**：
  - 成本仅为 LLM-only 系统的 **~9%**，实现 **>90% 成本削减**；
  - **约 67% 的查询由本地 SLM 成功处理**，无需调用 LLM，真正实现“近零成本”本地推理。
- **效率权衡合理**：尽管 EX 和 VES 较低，但考虑到成本下降两个数量级，这种 trade-off 在多数企业场景中具有高度实用性。

### ❌ 消融实验（隐含分析）
虽然未明确列出消融实验表格，文中通过对比揭示了关键组件作用：
- **移除 LLM 回退机制**（即仅用 SLM）会导致部分复杂查询无法恢复，影响最终准确率；
- **Extractor Agent 的 RAG + 向量检索** 显著减少幻觉，提高 Schema 对齐能力；
- **Validator Agent 的四阶段验证机制**（值校验、语法、执行、语义）有效拦截错误输出并提供诊断信息用于修复。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **SLM 可作为 NL2SQL 的主力模型**：对于大多数常见查询，SLM 已具备足够表达能力完成高质量 SQL 生成。
2. **Hybrid Agentic 架构可行且高效**：“SLM 主 + LLM 按需回退”的设计在保持较高可用性的同时极大降低成本。
3. **支持本地化部署与数据隐私保护**：适用于金融、医疗等对数据安全要求高的行业。
4. **成本效益显著优于现有方案**：平均每查询成本仅 $0.0085，适合大规模商用部署。

### ⚠️ 局限性
| 问题 | 描述 |
|------|------|
| **复杂推理能力不足** | SLM 在嵌套子查询、多跳 JOIN 或时间逻辑推理任务中表现下降，仍需依赖 LLM 回退。 |
| **尾部延迟较高** | 当需要多次 LLM 重试时，响应时间明显增加，影响实时用户体验。 |
| **级联错误传播** | 若 Extractor 或 Decomposer 出错，后续模块即使正常也无法纠正，影响整体成功率。 |
| **依赖 Schema 命名规范** | 在老旧系统中若列名晦涩或文档缺失，会影响 RAG 检索质量，加剧幻觉风险。 |

### 🔮 未来工作方向
1. **细粒度 Token 使用分析**：按查询复杂度分类，量化不同层级的成本-性能曲线。
2. **端到端延迟建模**：深入研究 fallback 场景下的响应时间分布，优化调度策略。
3. **增强隐私保护机制**：探索在发送至 LLM 前对敏感字段进行自动脱敏或掩码的技术。
4. **引入动态路由机制**：根据查询难度预测是否直接启用 LLM，进一步提升效率。

---

## 总结（一句话概括）
> 本文提出了一种 **Schema-aware、基于 SLM 的多智能体 NL2SQL 系统**，通过 **智能 LLM 回退机制** 实现了 **接近 LLM 级别的可用性** 与 **超过 90% 的成本节约**，为低成本、高安全性、可落地的企业级 NL2SQL 应用提供了新范式。

</details>

---

### 14. [GRAFITE: Generative Regression Analysis Framework for Issue Tracking and Evaluation](https://arxiv.org/abs/2603.18173)

**Authors**: Ja Young Lee, M\'irian Silva, Mohamed Nasr, Shonda Witherspoon, Enzo Bozzani, Veronique Demers, Radha Ratnaparkhi, Hui Wu, Sara Rosenthal  
**Category**: cs.CL  
**Published**: 2026-03-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.18173v1  

#### Abstract
Large language models (LLMs) are largely motivated by their performance on popular topics and benchmarks at the time of their release. However, over time, contamination occurs due to significant exposure of benchmark data during training. This poses a risk of model performance inflation if testing i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GRAFITE: Generative Regression Analysis Framework for Issue Tracking and Evaluation

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在发布时通常基于流行基准（benchmarks）进行评估，但随着时间推移，这些基准数据可能在训练过程中被污染（data contamination），导致模型性能被高估。此外，现有评估框架多为一次性、通用型测试，缺乏对**特定领域问题的持续追踪能力**，难以检测模型版本迭代中的**性能回归（regression）**。

### 提出的新方法与思路
作者提出 **GRAFITE** —— 一个面向 LLM 的**生成式回归分析框架**，用于系统化地跟踪和评估模型问题。其核心思想是构建一个由用户反馈驱动的、可复用的“问题-测试”仓库，并通过 LLM-as-a-judge 和人工协同的方式实现自动化、可持续的模型评估。

### 相比现有方法的优势
| 维度 | GRAFITE 的优势 |
|------|----------------|
| **持续性** | 支持跨模型版本的长期性能追踪，识别 regression 或 improvement |
| **领域针对性** | 聚焦 domain-specific 场景，弥补通用 benchmark 忽略实际用例的不足 |
| **协作机制** | 引入 end-user → subject matter expert → model developer 的闭环反馈流程 |
| **评估可靠性** | 采用 ensemble of LLM judges + human-in-the-loop annotation 减少偏见 |
| **易用性** | 提供图形化界面（GUI），降低技术门槛，支持非 CLI 用户 |

---

## 2. 核心实验方法和设置

### 数据集
- **样本数据来源**：
  - 人类生成的实例（human-generated instances）
  - 来自 [Chatbot Arena](https://chat.lmsys.org/) 的精选 benchmark 数据
- **任务分布**：
  - 共涵盖 **10 个 domain**，包括：Code, Creative, Factual, Instruction Following, Math, Multilingual, Reasoning, Summarization, Table Calculation, Underspecified
  - 包含 **20 个 issues** 和 **110 个 QA tests**
- 所有测试样例均开源：[GitHub 链接](https://github.com/IBM/grafite)

### 实验设置
- **评估对象**：Meta 的四个 Llama 系列模型：
  - `Llama-3.1-8B-Instruct`
  - `Llama-3.2-3B-Instruct`
  - `Llama-3.3-70B-Instruct`
  - `Llama-4-Maverick-17B-128E-Instruct`
- **Judge 模型（Ensemble）**：
  - `Llama-3.3-70B-Instruct`
  - `Phi-4`（Microsoft）
  - `Gpt-oss-120b`（OpenAI）
- **评估方式**：
  - 使用 LLM-as-a-judge 对每个 test 输出进行二元评分（0/1）
  - 判断标准基于预设的 prompt template 和 domain-specific guideline
  - Ensemble 决策：平均得分 > 0.5 视为通过
- **报告形式**：
  - 单模型测试报告（test report）
  - 多模型趋势分析（trend analysis），支持 comparison mode 和 individual mode

### 评估指标
- 主要指标：**Pass Rate (%)** 按 issue domain 聚合
- 辅助指标：
  - 各 domain 的 failure rate
  - 不同模型间的 performance disparity
  - 用户研究中的 Likert 量表评分（5-point scale）

### 基线方法对比
本文未直接与传统 benchmark 工具（如 HELM、LM Evaluation Harness）进行性能数值对比，而是从**功能维度**强调差异：

| 功能 | GRAFITE | 其他工具（如 HELM） |
|------|--------|--------------------|
| 持续追踪 regression | ✅ 支持 | ❌ 仅支持单次快照 |
| 用户反馈集成 | ✅ 支持 thumbs-down 反馈触发 issue 创建 | ❌ 无此机制 |
| GUI 支持 | ✅ Web-based 平台 | ❌ 多为 CLI 工具 |
| 领域定制化测试 | ✅ 支持手动创建 domain-specific issue/test | ⚠️ 依赖已有 dataset |
| 人机协同评估 | ✅ Human-in-the-loop override | ❌ 多为全自动 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 3）
| Issue Domain (n tests) | L3.1 (8B) | L3.2 (3B) | L3.3 (70B) | L4 (17B) |
|------------------------|-----------|-----------|------------|----------|
| **Total (110)** | 42.1% | 26.3% | 57.9% | **63.2%** |
| Code (17) | 76.5% | 64.7% | 76.5% | **100%** |
| Creative (6) | 100% | 83.3% | 100% | 83.3% |
| Factual (17) | 5.9% | 5.9% | 17.7% | **29.4%** |
| Instruction Following (12) | 41.7% | 25.0% | 25.0% | **58.3%** |
| Math (20) | 40.0% | 25.0% | 55.0% | **60.0%** |
| Multilingual (10) | 40.0% | 20.0% | 40.0% | 40.0% |
| Reasoning (6) | 66.7% | 0.0% | **83.3%** | 33.3% |
| Summarization (6) | 83.3% | 83.3% | 66.7% | 83.3% |
| Table (11) | 36.4% | 44.5% | 36.4% | **54.5%** |
| Underspecified (5) | 0.0% | 16.7% | 0.0% | 0.0% |

> 注：所有值为多个 judge 模型评分的平均值

### 与基线方法的对比结果
- **相比一次性 benchmark**：
  - GRAFITE 成功识别出 `Llama-3.2` 在 reasoning 和 factual 知识上的显著退步（regression），而这类细粒度变化在整体 benchmark 中容易被掩盖。
- **相比纯自动评估工具**：
  - 支持 human override，提升了边缘 case 和主观判断场景下的准确性。
- **可视化分析优势**：
  - Trend analysis dashboard 支持跨模型、跨 domain 的直观比较，便于快速定位性能瓶颈。

### 消融实验（Ablation Study）
文中未提供正式消融实验，但在 **User Study** 中间接验证了关键组件的有效性：
- 用户高度认可以下特性（Likert 平均分 ≥ 4.0）：
  - Human-in-the-loop annotation (**μ = 4.13**)
  - Ensemble evaluation with multiple LLM judges (**μ = 4.13**)
  - Pass/fail metrics (**μ = 4.13**)
  - Visual analytics & charts (**μ = 4.33**)
  - Regression test tracking (**μ = 4.13**)

表明这些设计对平台实用性具有显著正向影响。

---

## 4. 关键结论和发现

### 主要发现
1. **存在明显的性能 regression**：
   - `Llama-3.2` 尽管参数更小，但在 reasoning 和 factual 任务上表现远差于前代，说明不能仅以参数规模衡量进步。
2. **不同模型在特定 domain 表现分化明显**：
   - `Llama-4` 在 code generation 和 instruction following 上表现最优；
   - `Llama-3.3` 在 reasoning 上领先，但在 instruction following 上落后。
3. **“Underspecified” 类型任务普遍失败**：
   - 所有模型在面对信息不全的问题时，倾向于自行补全而非请求澄清，存在幻觉风险。
4. **用户反馈可有效转化为可执行测试**：
   - 通过 end-user thumbs-down → expert triaging → test creation 流程，实现了真实世界问题到 QA test 的闭环转化。

### 方法的局限性
- **LLM-as-a-judge 的固有偏见**：
  - 即使使用 ensemble，仍可能存在偏好泄露（preference leakage）或文化偏见。
  - 特别当 judge 模型与被测模型同源时（如都用 Llama 系列），评估结果可能失真。
- **GUI 易用性有待提升**：
  - 用户调查显示 interface navigation 和 issue organization 得分较低（μ ≈ 3.0），限制了大规模推广。
- **依赖人工标注成本**：
  - 虽然支持自动化评估，但高质量 issue 和 test 的创建仍需专家参与。

### 未来工作方向
1. **增强 issue 分类自动化**：
   - 开发更智能的 issue clustering 和标签推荐工具，减少人工整理负担。
2. **自动化重复评估任务**：
   - 探索通过 synthetic data generation 扩展 test sets，实现动态压力测试。
3. **改进 GUI 体验**：
   - 增加 performance heatmap、baseline model 对照、自定义 dashboard 等功能。
4. **引入更多评估范式**：
   - 如 pairwise comparison、reward modeling 等，丰富评估维度。

---

> **项目地址**：[https://github.com/IBM/grafite](https://github.com/IBM/grafite)  
> **演示视频**：[YouTube 链接](https://www.youtube.com/watch?v=XFZyoleN56k)

</details>

---

### 15. [Detecting Basic Values in A Noisy Russian Social Media Text Data: A Multi-Stage Classification Framework](https://arxiv.org/abs/2603.18822)

**Authors**: Maria Milkova, Maksim Rudnev  
**Category**: cs.CL  
**Published**: 2026-03-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.18822v1  

#### Abstract
This study presents a multi-stage classification framework for detecting human values in noisy Russian language social media, validated on a random sample of 7.5 million public text posts. Drawing on Schwartz's theory of basic human values, we design a multi-stage pipeline that includes spam and non...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Detecting Basic Values in A Noisy Russian Social Media Text Data: A Multi-Stage Classification Framework*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本研究旨在解决在**非英语、高噪声的社会媒体文本中检测人类基本价值观（basic human values）** 的挑战。具体包括：
- 社交媒体数据中价值表达稀疏且隐含（implicit），难以有效识别。
- 现有方法多基于英文平台（如Twitter、Reddit），缺乏对俄语等非西方语言环境的研究。
- 传统标注依赖人工专家或众包工人，成本高、可扩展性差，且存在主观偏差。

### 🚀 提出的新方法与创新思路
提出了一种**多阶段分类框架（multi-stage classification framework）**，结合 **LLM辅助软标签（soft labeling）** 和 **transformer-based 多标签分类模型**，实现大规模、文化敏感的价值检测。

#### 主要创新点：
1. **采用 GPT-4 进行多轮标注并生成 soft labels**  
   - 对每条文本进行5次独立的 GPT-4 标注，根据一致性程度生成连续型软标签（soft labels）：
     - 4–5 次同意 → label = 1.0（强共识）
     - 3 次同意 → label = 0.6（中等共识）
     - ≤2 次同意 → label = 0.0（无共识）
   - 避免将 LLM 输出视为“ground truth”，而是作为反映解释多样性的信号源。

2. **不将专家标注视为绝对标准，而是一种“解释性基准”（interpretative benchmark）**  
   - 承认专家之间也存在显著分歧（inter-rater agreement 较低），因此不追求完全匹配专家标签，而是分析模型与专家判断之间的系统性差异。

3. **构建端到端的四阶段处理流程**：
   1. 垃圾内容过滤（spam filtering）
   2. 识别是否为价值表达性文本（value-expressive post detection）
   3. 政治相关性检测（politically-oriented post detection）
   4. 十类基本价值的多标签分类（multi-label value classification）

4. **发布开源资源**  
   - 公开了所有代码、标注指南、API脚本和预训练模型，促进后续研究复现与拓展。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|----------|---------|
| 数据来源 | 英文平台为主 | 俄语VKontakte，填补非西方语境空白 |
| 标注方式 | 人工标注（专家/众包） | GPT-4 + 软标签聚合，高效可扩展 |
| 模型目标 | 分类决策 | 学习不确定性模式，输出概率化预测 |
| 文化适应性 | 忽视本地表达习惯 | 显式建模文化嵌入性（如Universalism与Conservation共现） |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **平台**：俄罗斯最大社交网络 **VKontakte (VK)**  
- **采样方式**：随机抽取100万个用户ID，收集其公开文本帖子
- **时间跨度**：2007年 – 2025年1月（覆盖俄乌战争爆发后时期）
- **最终可用数据量**：
  - 初始文本数：7,498,657 条
  - 经过 Cyrillic 字符 & 至少两词过滤：5,561,547 条
  - 去除垃圾与非个人内容后：约 358万 条
  - 最终用于价值/政治表达分析的文本：**1,105,085 条**

> ⚠️ 注：未公开原始数据集以保护用户隐私，仅提供处理脚本和工具。

### 🧪 实验设置与评估指标

#### 多标签分类任务设定
- **理论基础**：Schwartz’s theory of basic human values
- **十类价值类型**：
  - **Openness to Change**: Self-Direction, Stimulation, Hedonism
  - **Self-Enhancement**: Achievement, Power
  - **Conservation**: Security, Conformity, Tradition
  - **Self-Transcendence**: Benevolence, Universalism

- **任务形式**：multi-label classification（单个post可表达多个value）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **F1 / F1-macro** | 主要性能指标，尤其适用于类别不平衡场景 |
| **Fleiss’ Kappa** | 衡量多人标注间的一致性（>0.6 表示 substantial agreement） |
| **ICC (Intra-class Correlation)** | 评估连续变量（如预测概率）的一致性（>0.75 为良好） |
| **Value Circle Distance** | 将价值配置投影至Schwartz圆环，计算欧氏距离（0~1），衡量整体结构相似性 |
| **Procrustes Rotation + Congruence Coefficient (rc)** | 比较实证结构与理论结构的匹配度（rc > 0.9 为极佳） |

#### 基线方法对比
- 比较了多种 **transformer-based 模型** 在相同训练设置下的表现：
  - RuBERT-tiny2, RuBERT-base
  - RuRoberta-large
  - DeBERTa-v3-large
  - Multilingual e5-large
  - **XLM-RoBERTa-large**（最佳模型）
  - BERTA

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 模型 | F1 (test) | F1-macro (test) |
|------|-----------|------------------|
| XLM-RoBERTa-large | **0.71** | **0.83** ✅（SOTA） |
| RuRoberta-large | 0.69 | 0.82 |
| BERTA | 0.69 | 0.82 |
| RuBERT-base | 0.68 | 0.81 |

> ✅ **XLM-RoBERTa-large 是表现最好的模型**，尽管它并非专门针对俄语训练，但在部分微调后表现出强大的跨语言迁移能力。

#### 各价值类别的F1得分（XLM-RoBERTa-large, test set）
| Value Type | F1 |
|-----------|-----|
| Benevolence | 0.82 |
| Self-direction | 0.77 |
| Stimulation | 0.73 |
| Achievement | 0.69 |
| Power | 0.64 |
| Hedonism | 0.61 |
| Universalism | 0.60 |
| Security | 0.63 |
| Tradition | 0.63 |
| Conformity | 0.52 ❗最低 |

> 💡 发现：**Benevolence 和 Self-direction 最易识别**；**Conformity 最难**，可能因其表达更隐晦或语境依赖性强。

### 🔁 与基线方法的对比结果
- **XLM-RoBERTa-large 显著优于其他模型**，尤其是在 F1-macro 上领先。
- 多语言模型（如 XLM-RoBERTa）优于纯俄语专用模型（如 RuBERT），说明**跨语言预训练提供了更强的语义泛化能力**。
- 使用 soft labels 的模型比使用硬标签的版本性能更高，验证了建模标注不确定性的有效性。

### 🔍 消融实验与关键验证

#### （1）GPT vs. 专家标注一致性
- GPT 内部一致性（Fleiss’ Kappa）平均为 **0.649**，高于专家间的 **0.301**
- 但在 multi-label 设置下，GPT 与专家多数投票的 F1 仅为 **0.53**
- 在**高阶价值域（higher-order domains）** 层面一致性提升：
  - Self-Transcendence: F1 = 0.75
  - Openness to Change: F1 = 0.60
  - Conservation: F1 = 0.52 ❗最低

#### （2）模型 vs. 专家判断
- 模型与专家多数标签的整体 F1 = **0.53**（与 GPT–expert 水平相当）
- 模型预测概率与专家一致比例（consistency score）的 Spearman ρ = **0.45**（中等相关）
  - Benevolence 达到 **0.75**，Power 仅为 **0.33**

#### （3）关键偏差分析
- 模型和 GPT 均表现出**系统性偏向 Openness to Change**：
  - 当专家标注为 Conservation 时，GPT 将 **9.9%** 的案例判为 Openness to Change
  - 模型在 Conservation → Openness to Change 方向出现 **16%** 的误判
- 原因推测：
  - 健康生活、内心挣扎、情感倾诉等内容被解读为“自我成长”而非“维持稳定”

#### （4）结构效度检验（MDS + Procrustes）
- 实证价值结构与理论 Schwartz 圆环高度吻合：
  - **Congruence coefficient rc = 0.927**（>0.9，极佳）
  - Rank correlation τ = 0.556（中等）
- 微小偏离：
  - Hedonism 与 Self-direction 位置互换（仍在 Openness to Change 内部）
  - Universalism 更接近 Conservation（反映本土化表达：和平诉求常与传统、家庭绑定）

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLM 可作为高效、可靠的价值标注代理（proxy annotator）**
   - GPT-4 的内部一致性高于人类专家，且能捕捉到合理的解释路径。
   - 结合多轮运行与 soft labeling，可在保持效率的同时保留不确定性信息。

2. **XLM-RoBERTa-large 在俄语价值检测任务上表现最优**
   - 跨语言预训练模型具备良好的文化迁移潜力，无需专有俄语语料即可取得优异效果。

3. **模型整体与人类判断一致，但存在系统性偏见**
   - 特别是倾向于将复杂情绪或反思性表达归类为 Openness to Change。
   - 这种偏差本身具有社会意义，反映了不同“阅读视角”的合理性。

4. **俄语社交媒体中的价值表达模式**
   - 最常见的是 **Self-direction**, **Benevolence**, **Stimulation**
   - 最少见的是 **Power**, **Conformity**
   - **Universalism 与 Security/Tradition 正相关**，不同于西方语境，表明其通过“集体安全”而非“全球包容”来体现。

5. **价值共现结构符合 Schwartz 理论预期**
   - 相邻价值（如 Security–Tradition, Self-direction–Stimulation）正相关
   - 对立价值（如 Self-direction–Security, Benevolence–Power）负相关
   - MDS 分析显示结构高度还原理论模型（rc = 0.927）

---

### ⚠️ 方法的局限性

1. **数据代表性受限**
   - 仅使用公开账户数据，可能遗漏因审查压力转为私密的用户群体。
   - 无法排除 bot 或组织化账号的影响。

2. **未涵盖评论与互动数据**
   - 仅分析用户主动发布的原创帖文，忽略了对话情境中的价值协商过程。

3. **忽略多模态内容**
   - 图像、视频等非文本形式的价值表达未被纳入。

4. **GPT 标注可能存在 prompt bias**
   - 尽管控制温度与随机种子，但仍受提示工程影响，需谨慎推广至其他文化背景。

5. **soft labels 并非完全概率化**
   - 仅有三个离散等级（1.0, 0.6, 0.0），未能充分利用 full probabilistic annotation 的潜力。

---

### 🔮 未来工作方向

1. **探索 annotator 自身价值观对标注行为的影响**
   - 是否高开放性的人更倾向识别 Openness to Change？

2. **引入交互数据分析**
   - 分析点赞、转发等行为如何调节价值传播与可见性。

3. **扩展至多模态价值检测**
   - 融合图像 caption、表情符号、音频线索进行 multimodal value classification。

4. **动态追踪价值演变**
   - 利用该框架监测重大事件（如战争、选举）前后价值表达的变化趋势。

5. **应用于政策传播、舆论引导等领域**
   - 探索哪些价值组合更容易获得公众共鸣或引发争议。

---

> 📌 **总结一句话**：  
> 本研究通过融合 **LLM软标注 + 多阶段分类 + 文化敏感建模**，成功实现了在嘈杂俄语社交媒体中自动检测十类基本人类价值的任务，不仅达到了 SOTA 性能（F1=0.71, F1-macro=0.83），还揭示了数字环境中价值表达的文化特异性与解释多样性，为跨文化计算社会科学提供了可复现、可扩展的新范式。

</details>

---

### 16. [DriftGuard: Mitigating Asynchronous Data Drift in Federated Learning](https://arxiv.org/abs/2603.18872)

**Authors**: Yizhou Han, Di Wu, Blesson Varghese  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.18872v1  

#### Abstract
In real-world Federated Learning (FL) deployments, data distributions on devices that participate in training evolve over time. This leads to asynchronous data drift, where different devices shift at different times and toward different distributions. Mitigating such drift is challenging: frequent r...

---

### 17. [An Optimised Greedy-Weighted Ensemble Framework for Financial Loan Default Prediction](https://arxiv.org/abs/2603.18927)

**Authors**: Ezekiel Nii Noye Nortey, Jones Asante-Koranteng, Marcellin Atemkeng, Theophilus Ansah-Narh, David Mensah, Rebecca Davis, Ravenhill Adjetey Laryea  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.18927v1  

#### Abstract
Accurate prediction of loan defaults is a central challenge in credit risk management, particularly in modern financial datasets characterised by nonlinear relationships, class imbalance, and evolving borrower behaviour. Traditional statistical models and static ensemble methods often struggle to ma...

---

### 18. [Reflection in the Dark: Exposing and Escaping the Black Box in Reflective Prompt Optimization](https://arxiv.org/abs/2603.18388)

**Authors**: Shiyan Liu, Qifeng Xia, Qiyun Xia, Yisheng Liu, Xinyu Yu, Rui Qu  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.18388v1  

#### Abstract
Automatic prompt optimization (APO) has emerged as a powerful paradigm for improving LLM performance without manual prompt engineering. Reflective APO methods such as GEPA iteratively refine prompts by diagnosing failure cases, but the optimization process remains black-box and label-free, leading t...

---

### 19. [LuMamba: Latent Unified Mamba for Electrode Topology-Invariant and Efficient EEG Modeling](https://arxiv.org/abs/2603.19100)

**Authors**: Dana\'e Broustail, Anna Tegon, Thorir Mar Ingolfsson, Yawei Li, Luca Benini  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.19100v1  

#### Abstract
Electroencephalography (EEG) enables non-invasive monitoring of brain activity across clinical and neurotechnology applications, yet building foundation models for EEG remains challenging due to \emph{differing electrode topologies} and \emph{computational scalability}, as Transformer architectures ...

---

### 20. [AutoScreen-FW: An LLM-based Framework for Resume Screening](https://arxiv.org/abs/2603.18390)

**Authors**: Zhelin Xu, Shuhei Yamamoto, Atsuyuki Morishima  
**Category**: cs.CL  
**Published**: 2026-03-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.18390v1  

#### Abstract
Corporate recruiters often need to screen many resumes within a limited time, which increases their burden and may cause suitable candidates to be overlooked. To address these challenges, prior work has explored LLM-based automated resume screening. However, some methods rely on commercial LLMs, whi...

---

### 21. [GAPSL: A Gradient-Aligned Parallel Split Learning on Heterogeneous Data](https://arxiv.org/abs/2603.18540)

**Authors**: Zheng Lin, Ons Aouedi, Wei Ni, Symeon Chatzinotas, Xianhao Chen  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.18540v1  

#### Abstract
The increasing complexity of neural networks poses significant challenges for democratizing FL on resource?constrained client devices. Parallel split learning (PSL) has emerged as a promising solution by offloading substantial computing workload to a server via model partitioning, shrinking client-s...

---

### 22. [Agentic Flow Steering and Parallel Rollout Search for Spatially Grounded Text-to-Image Generation](https://arxiv.org/abs/2603.18627)

**Authors**: Ping Chen, Daoxuan Zhang, Xiangming Wang, Yungeng Liu, Haijin Zeng, Yongyong Chen  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.18627v1  

#### Abstract
Precise Text-to-Image (T2I) generation has achieved great success but is hindered by the limited relational reasoning of static text encoders and the error accumulation in open-loop sampling. Without real-time feedback, initial semantic ambiguities during the Ordinary Differential Equation trajector...

---

### 23. [Thinking with Constructions: A Benchmark and Policy Optimization for Visual-Text Interleaved Geometric Reasoning](https://arxiv.org/abs/2603.18662)

**Authors**: Haokun Zhao, Wanshi Xu, Haidong Yuan, Songjun Cao, Long Ma, Yanghua Xiao  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.18662v1  

#### Abstract
Geometric reasoning inherently requires "thinking with constructions" -- the dynamic manipulation of visual aids to bridge the gap between problem conditions and solutions. However, existing Multimodal Large Language Models (MLLMs) are largely confined to passive inference with static diagrams, lack...

---

### 24. [CWoMP: Morpheme Representation Learning for Interlinear Glossing](https://arxiv.org/abs/2603.18184)

**Authors**: Morris Alper, Enora Rice, Bhargav Shandilya, Alexis Palmer, Lori Levin  
**Category**: cs.CL  
**Published**: 2026-03-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.18184v1  

#### Abstract
Interlinear glossed text (IGT) is a standard notation for language documentation which is linguistically rich but laborious to produce manually. Recent automated IGT methods treat glosses as character sequences, neglecting their compositional structure. We propose CWoMP (Contrastive Word-Morpheme Pr...

---

### 25. [TopoChunker: Topology-Aware Agentic Document Chunking Framework](https://arxiv.org/abs/2603.18409)

**Authors**: Xiaoyu Liu  
**Category**: cs.CL  
**Published**: 2026-03-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.18409v1  

#### Abstract
Current document chunking methods for Retrieval-Augmented Generation (RAG) typically linearize text. This forced linearization strips away intrinsic topological hierarchies, creating ``semantic fragmentation'' that degrades downstream retrieval quality. In this paper, we propose TopoChunker, an agen...

---

### 26. [Literature Study on Operational Data Analytics Frameworks in Large-scale Computing Infrastructures](https://arxiv.org/abs/2603.19016)

**Authors**: Shekhar Suman, Xiaoyu Chu, Alexandru Iosup  
**Category**: cs.DC  
**Published**: 2026-03-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.19016v1  

#### Abstract
By 2025, there are zettabytes of data generated every year. The size and complexity of modern large-scale computing infrastructures like High-Performance Computing (HPC) systems continue to evolve and become complex, leaving us wondering about their manageability and sustainability concerns. Because...

---

### 27. [Conflict-Free Policy Languages for Probabilistic ML Predicates: A Framework and Case Study with the Semantic Router DSL](https://arxiv.org/abs/2603.18174)

**Authors**: Xunzhuo Liu, Hao Wu, Huamin Chen, Bowei He, Xue Liu  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.18174v1  

#### Abstract
Conflict detection in policy languages is a solved problem -- as long as every rule condition is a crisp Boolean predicate. BDDs, SMT solvers, and NetKAT all exploit that assumption. But a growing class of routing and access-control systems base their decisions on probabilistic ML signals: embedding...

---

### 28. [Communication-Efficient and Robust Multi-Modal Federated Learning via Latent-Space Consensus](https://arxiv.org/abs/2603.19067)

**Authors**: Mohamed Badi, Chaouki Ben Issaid, Mehdi Bennis  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.19067v1  

#### Abstract
Federated learning (FL) enables collaborative model training across distributed devices without sharing raw data, but applying FL to multi-modal settings introduces significant challenges. Clients typically possess heterogeneous modalities and model architectures, making it difficult to align featur...

---

### 29. [From Inference Efficiency to Embodied Efficiency: Revisiting Efficiency Metrics for Vision-Language-Action Models](https://arxiv.org/abs/2603.19131)

**Authors**: Zhuofan Li (Celine), Hongkun Yang (Celine), Zhenyang Chen (Celine), Yangxuan Chen (Celine),  Yingyan (Celine),  Lin, Chaojian Li  
**Category**: cs.LG  
**Published**: 2026-03-20  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.19131v1  

#### Abstract
Vision-Language-Action (VLA) models have recently enabled embodied agents to perform increasingly complex tasks by jointly reasoning over visual, linguistic, and motor modalities. However, we find that the prevailing notion of ``efficiency'' in current VLA research, characterized by parameters, FLOP...

---

### 30. [Adaptive Domain Models: Bayesian Evolution, Warm Rotation, and Principled Training for Geometric and Neuromorphic AI](https://arxiv.org/abs/2603.18104)

**Authors**: Houston Haynes  
**Category**: cs.AI  
**Published**: 2026-03-20  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.18104v1  

#### Abstract
Prevailing AI training infrastructure assumes reverse-mode automatic differentiation over IEEE-754 arithmetic. The memory overhead of training relative to inference, optimizer complexity, and structural degradation of geometric properties through training are consequences of this arithmetic substrat...

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
