# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-06 07:16:03 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Analyzing Reverse Address Translation Overheads in Multi-GPU Scale-Up Pods](https://arxiv.org/abs/2604.02473)

**Authors**: Amel Fatima, Tuan Ta, Bradford M. Beckmann  
**Category**: cs.DC  
**Published**: 2026-04-06  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2604.02473v1  

#### Abstract
Distributed ML workloads rely heavily on collective communication across multi-GPU, multi-node systems. Emerging scale-up fabrics, such as NVLink and UALink, enable direct memory access across nodes but introduce a critical destination-side translation step: translating Network Physical Addresses (N...

---

### 2. [MSAO: Adaptive Modality Sparsity-Aware Offloading with Edge-Cloud Collaboration for Efficient Multimodal LLM Inference](https://arxiv.org/abs/2604.02945)

**Authors**: Zheming Yang, Qi Guo, Jun Wan, Jiarui Ruan, Yunqing Hu, Chang Zhao, Xiangyang Li  
**Category**: cs.DC  
**Published**: 2026-04-06  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2604.02945v1  

#### Abstract
Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities but impose substantial computational and latency burdens, posing critical challenges for deployment on resource-constrained edge devices. In this paper, we propose MSAO, an adaptive modality sparsity-aware of...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MSAO: Adaptive Modality Sparsity-Aware Offloading with Edge-Cloud Collaboration for Efficient Multimodal LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Multimodal LLM（MLLM）在跨模态推理方面表现出色，但其高计算开销和延迟对资源受限的边缘设备部署构成挑战。传统方案存在以下问题：
- **Cloud-only**：通信延迟高、带宽压力大；
- **Edge-only**：算力不足，难以处理复杂任务；
- **现有边缘-云协同框架**：采用统一的卸载策略，忽略不同模态之间的异构性（如图像/视频中的空间-时间冗余 vs 文本/音频的语义密集性），导致资源利用低效。

### 🚀 提出的新方法与创新思路
本文提出 **MSAO**（Modality Sparsity-Aware Offloading），一种基于边缘-云协作的自适应卸载框架，核心思想是**利用模态激活稀疏性指导动态卸载决策**，实现高效 MLLM 推理。

#### 主要创新点：
1. **轻量级异构模态感知模块（Lightweight Heterogeneous Modality-Aware Module）**
   - 引入细粒度的 **Modality Activation Sparsity (MAS)** 度量标准；
   - 通过空间（spatial）、时间（temporal）、模态间（modal）联合分析，量化每种模态对当前任务的重要性；
   - 在边缘端以极低开销完成分析，避免成为瓶颈。

2. **自适应推测式边缘-云协同卸载机制（Adaptive Speculative Edge-Cloud Collaborative Offloading）**
   - 动态调度边缘与云端的工作负载；
   - 利用 **confidence-guided speculative execution** 技术隐藏通信延迟；
   - 边缘运行小型 draft model 快速生成候选 token，云端验证并纠正，提升吞吐。

3. **端到端优化设计**
   - 将 MAS 分析结果与实时系统状态（带宽、内存、延迟）结合，进行联合优化；
   - 支持 per-request 和 per-step 两级调控，兼顾全局最优与运行时适应性。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 显著降低 end-to-end latency 和 resource overhead |
| **灵活性** | 自适应处理不同输入模态组合与复杂度 |
| **精度保持** | 不牺牲模型准确率的前提下实现加速 |
| **实用性** | 轻量模块适合部署于边缘设备 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **VQAv2**：大规模视觉问答数据集，包含超过 25 万张图像和 110 万个问题；从中随机采样 5,000 张用于测试。
- **MMBench**：综合性多模态基准，涵盖 20 个能力维度（对象识别、属性推理、场景理解等），使用完整测试集。

### ⚙️ 实验平台配置
- **云端服务器**：NVIDIA A100 (40GB GPU)
- **边缘设备**：NVIDIA RTX 3090 (24GB GPU)
- **模型设置**：
  - 边缘 draft model：Qwen2-VL-2B（20亿参数）
  - 云端 full model：Qwen2.5-VL-7B（70亿参数）
  - 共享 tokenizer 和架构，支持 speculative verification
- **网络模拟**：
  - 带宽等级：200 Mbps（低）、300 Mbps（中）、400 Mbps（高）
  - RTT 固定为 20ms

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | VQAv2 使用标准 VQA 准确率；MMBench 报告 20 项平均准确率 |
| **Throughput** | 单位时间内生成的 token 数量（tokens/s） |
| **End-to-End Latency** | 输入提交至最终响应生成的总耗时（ms） |
| **Computing Overhead** | 推理过程中的总浮点运算量（FLOPs） |
| **Memory Overhead** | 推理过程中峰值 GPU 内存占用（GB） |

### 🆚 基线方法对比
1. **Cloud-only**：所有输入上传至云端，由 Qwen2.5-VL-7B 完全执行；
2. **Edge-only**：仅在边缘运行 Qwen2-VL-2B；
3. **PerLLM [39]**：基于层划分的边缘-云协同推理框架。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 性能维度 | MSAO 表现 | 对比提升 |
|----------|-----------|---------|
| **End-to-End Latency** | ↓ 30% | vs. 所有 baseline |
| **Resource Overhead** | ↓ 30%–65% | 计算 + 内存综合开销 |
| **Throughput** | ↑ 1.5× – 2.3× | 最高达 2.66×（VQAv2 @ 400Mbps） |
| **Accuracy** | ≈ Cloud-only | 差距 < 0.4%，显著优于其他方法 |

### 🔬 详细对比结果

#### ✅ 准确率（Accuracy）
- MSAO 在两个数据集上均接近 **Cloud-only 上限**：
  - VQAv2 @ 200Mbps：76.1% vs. 76.3%
  - MMBench @ 400Mbps：76.3% vs. 76.5%
- 远超 Edge-only（+12.7%~17.9%）和 PerLLM（+3.8%~7.2%）

> 👉 表明 MSAO 成功保留了高质量推理能力。

#### ✅ 吞吐量（Throughput）
- 在 VQAv2 @ 400Mbps 下，MSAO 达到 **128 tokens/s**，而 Cloud-only 仅为 35 tokens/s（↑2.66×）；
- 相比 Edge-only 提升 >2×；
- 相比 PerLLM 提升 1.5×–1.7×。

> 👉 得益于 speculative execution 和高效的边缘预处理。

#### ✅ 端到端延迟（End-to-End Latency）
- 相比 Cloud-only ↓ >50%
- 相比 Edge-only ↓ 45%–55%
- 相比 PerLLM ↓ >30%

> 👉 有效缓解了传输延迟和边缘算力瓶颈。

#### ✅ 资源开销
- **计算开销（Computing Overhead）**：
  - 相比 Cloud-only ↓ 30%–65%
  - 相比 PerLLM ↓ 35%–50%
- **内存开销（Memory Overhead）**：
  - 在 200Mbps 下，边缘内存从 25.0GB（Cloud-only）降至 **9.0GB**（↓64%）
  - MSAO 对带宽变化不敏感，具有强鲁棒性

> 👉 MSAO 实现“简单任务本地处理，困难任务上云”的智能分流。

### 🔍 消融实验（Ablation Study）

| 变体 | 移除组件 | 影响 |
|------|--------|------|
| **w/o Modality-Aware** | 移除 MAS 模块，采用统一卸载策略 | - 准确率下降 6.8%（VQAv2）、7.6%（MMBench） |
| **w/o Collaborative Scheduling** | 移除推测式协同调度 | - 延迟 ↑ 45%~48%<br>- 计算 & 内存开销显著上升 |

> 👉 验证了 MAS 指导和 speculative 协同机制的关键作用。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **模态稀疏性可被有效建模并用于指导卸载决策**  
   MAS 指标能够精准识别冗余模态信息，在不影响精度前提下大幅减少传输与计算负担。

2. **边缘-云协同需考虑模态异构性**  
   统一卸载策略无法应对图像/视频的空间-时间冗余与文本/音频的高保真需求差异，必须引入自适应机制。

3. **推测式执行能有效掩盖通信延迟**  
   结合 confidence-aware 控制，实现了边缘 draft 与云端 verify 的高度重叠，显著提升吞吐。

4. **轻量级设计可在边缘高效运行**  
   MAS 分析模块仅增加 <2% 的延迟和 <1.23% 的 FLOPs 开销，具备实际部署可行性。

### ⚠️ 局限性
- 当前依赖特定结构的 encoder（如早期特征图提取），可能限制通用性；
- speculative decoding 的接受率受 draft/full 模型一致性影响；
- 实验基于静态网络环境模拟，未完全覆盖真实动态场景（如突发拥塞）。

### 🔮 未来工作方向
1. 引入在线学习机制，实现 **动态环境下的自适应调整**；
2. 扩展至更大规模边缘-云系统，研究 **多节点协同调度**；
3. 探索 **更高效的 sparse probing 网络结构**，进一步降低边缘开销；
4. 支持更多模态组合（如点云、传感器信号）与应用场景（AR/VR、自动驾驶）。

---

> 💡 **总结一句话**：  
> MSAO 通过引入 **Modality Activation Sparsity (MAS)** 作为指导信号，构建了一个轻量、自适应、高吞吐的边缘-云协同推理框架，在几乎不损失准确率的前提下，实现了 **30% 的延迟降低、30%-65% 的资源节省、1.5×–2.3× 的吞吐提升**，为高效 MLLM 部署提供了新范式。

</details>

---

### 3. [Digital Twin-Assisted In-Network and Edge Collaboration for Joint User Association, Task Offloading, and Resource Allocation in the Metaverse](https://arxiv.org/abs/2604.02938)

**Authors**: Ibrahim Aliyu, Seungmin Oh, Sangwon Oh, Jinsul Kim  
**Category**: cs.DC  
**Published**: 2026-04-06  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.02938v1  

#### Abstract
Advancements in extended reality (XR) are driving the development of the metaverse, which demands efficient real-time transformation of 2D scenes into 3D objects, a computation-intensive process that necessitates task offloading because of complex perception, visual, and audio processing. This chall...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Digital Twin-Assisted In-Network and Edge Collaboration for Joint User Association, Task Offloading, and Resource Allocation in the Metaverse

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **Metaverse** 中基于 **Extended Reality (XR)** 应用的高计算需求与低延迟挑战，解决以下关键问题：
- **不对称上下行（UL/DL）数据流**：上行传输2D原始数据，下行返回渲染后的3D内容，导致资源分配不均、延迟增加。
- **多用户干扰与资源竞争**：在 **Multi-access Edge Computing (MEC)** 架构下，大量XR用户设备（XUDs）并发请求导致信道拥塞、计算负载不均衡。
- **动态任务到达与能效优化**：传统方法难以适应动态任务请求和能量受限的下行传输。

### 🚀 提出的新方法与创新点
作者提出了一种 **数字孪生（Digital Twin, DT）辅助的“网络内计算”与边缘协同框架（DT-assisted INC-E）**，其核心创新包括：

#### （1）**DT-assisted INC-E 架构**
- 将 **In-Network Computing (INC)** 节点引入 MEC 架构，实现任务的分层处理（部分预处理在INC节点，最终渲染在MEC服务器）。
- 利用 **Digital Twin** 实时镜像物理网络状态，支持预测性资源调度与自适应决策。

#### （2）**Stackelberg Markov 博弈建模**
- 将网络运营商（operator）作为 **领导者（Leader）**，XUDs 作为 **跟随者（Followers）**，构建分层博弈模型。
- 运营商决定 **Offloading Mode (OFMO)** 和 **Downlink Power Allocation (POAL)**，XUDs 自主选择信道与任务分割比例。

#### （3）**Nash-Asynchronous Hybrid Multi-Agent Reinforcement Learning (Nash-AMRL)**
- 设计异步双代理架构：
  - **UL Agent**：输出 OFMO 偏好分数，通过 **Knapsack Solver** 转换为二进制 offloading 决策。
  - **DL Agent**：连续控制下行功率分配。
- 引入三种 critic 架构进行比较：
  - **AHMRL**（Asynchronous Hybrid MRL）：本地 UL/DL critic + 全局 critic
  - **MASC**（Multi-actor Shared Critic）
  - **AC**（独立 Actor-Critic）

#### （4）**去中心化用户关联机制**
- XUDs 基于运营商广播的 OFMO 预测，自主进行 **干扰感知的信道选择**，形成 **Exact Potential Game**，收敛至 **Nash Equilibrium (NE)**。

### 🔍 相比现有方法的优势
| 方面 | 传统 MEC / MARL 方法 | 本文 DT-assisted INC-E + Nash-AMRL |
|------|------------------------|-------------------------------------|
| 架构 | 仅 MEC，无网络内处理 | INC + MEC 分层协作，降低 MEC 负载 |
| 决策方式 | 集中式或静态策略 | 分布式、自组织、动态响应 |
| 上下行耦合 | 忽略 UL/DL 不对称性 | 显式联合优化 OFMO 与 POAL |
| 学习机制 | 同步 MARL 或单代理 DRL | 异步双代理 AMRL，适配 INC-E 流程 |
| 可扩展性 | 受限于全局协调开销 | 去中心化信道选择，支持大规模部署 |

---

## 2. 核心实验方法和设置

### 📊 数据集与仿真环境
- **非真实数据集**，采用 **自定义模拟器** 生成动态任务请求。
- 用户行为建模为 **一阶马尔可夫链**，任务请求转移概率由 Zipf 分布（参数 σ ∈ {0.7, 1.3}）和空闲概率 R 控制。
- 3D 渲染数据放大函数：$ I' = q \times I $，其中 $ q \sim \text{Uniform}[1,10] $，反映 NeRF、Monster Mash 等技术的数据膨胀特性。

### ⚙️ 实验设置
| 参数 | 设置 |
|------|------|
| XUD 数量 M | 6 |
| INC 节点数量 K | 4 |
| MEC 服务器 | 1 |
| 区域大小 | 200m × 200m |
| 带宽 | 10 MHz |
| 噪声谱密度 | -174 dBm/Hz |
| URLLC 解码错误率 ε | $10^{-9}$ |
| 下行功率范围 | [0, 20] W |
| 任务输入大小 | [1,5] MB |
| 计算负载 | [1,5] Gcycles |
| 最大容忍延迟 | [5,15] ms |
| DT 处理速率偏差 | 30% |

### 📈 评估指标（KPIs）
| 类别 | 指标 |
|------|------|
| **用户侧** | - Utility<br>- Uplink Rate<br>- End-to-End Latency ($T_{e2e}$)<br>- Energy Consumption |
| **运营商侧** | - UL Reward<br>- DL Reward<br>- Global Reward ($R_g$) |
| **系统级** | - Performance Gain (PG)<br>- Cost Gain CDF<br>- Convergence Speed |

### 🆚 基线方法对比
| 基线名称 | 描述 |
|--------|------|
| **GM-RN (Game-Random)** | OFMO 与 POAL 完全随机分配 |
| **Equal Policy** | 固定 50% 协同 offloading，均匀功率分配 |
| **Proportional Policy** | Offloading 概率与 DL 功率按资源可用性和信道增益成比例分配 |
| **AAHC [19]** | 异步混合强化学习基准（用于对比 critic 设计） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 模型 | 平均 UL Rate ↑ | 平均 Utility ↑ | 能耗 ↓ | E2E 延迟 ↓ | 成本增益 (PG) ↑ |
|------|----------------|----------------|--------|------------|------------------|
| **AHMRL** | **最高**（尤其在 σ=0.7） | **最优**（AUC 达 1.0） | 中等 | 低于 MASC（K=4 时↓3.04%） | >2.0（85% 用户） |
| **MASC** | 次优，K 大时表现好 | 接近 AHMRL | 更低（大 K） | 最优（后期训练） | >2.0（80% 场景） |
| **AC** | 高 UL Rate（σ=1.3） | 最差 | 最高 | 中等 | <1.5（多数用户） |
| **Heuristics** | 显著更低 | 明显劣于所有 MARL | 更高 | 更高 | 多数 <1.5 |

> 注：数据来自 Table II 与 Fig. 4–8 综合分析。

### 🔍 与基线方法对比结果
- **相比 Heuristic 方法（GM-RN, EQ-PLC, PROP-PLC）**：
  - AHMRL 在 **超过 85% 的用户中实现了 ≥2.0 的成本增益**，而启发式方法不足 60%。
  - 全球奖励提升 **30%~40%**，特别是在小规模 INC 配置下（如 K=2）。
- **相比 MARL 变体**：
  - **AHMRL** 在通信密集型任务中表现最佳（↑6.24% global reward）。
  - **MASC** 在计算密集型、高密度场景更稳定。
  - **AC** 易过拟合，reward 波动大。

### 🔧 消融实验与关键发现
| 实验 | 发现 |
|------|------|
| **不同 critic 架构对比** | AHMRL 凭借局部 + 全局 critic 实现更好信用分配；MASC 虽简单但性能接近，说明全局 reward 已足够强引导。 |
| **Zipf 参数影响（σ=0.7 vs 1.3）** | σ=0.7（任务分布更集中）时 AHMRL 表现更优；σ=1.3 时 AC 的 UL rate 更高，但 utility 不稳定。 |
| **INC 节点数量变化（K=1~4）** | AHMRL 在 K=2~3 时最优；当 K=1（资源极度受限），AC 表现反超，表明轻量模型更具鲁棒性。 |
| **游戏收敛性** | Fig. 4(h) 显示 multi-user game 在 20 轮内快速收敛至 NE，验证 decentralized 协议有效性。 |

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **DT-assisted INC-E 框架显著提升 Metaverse XR 服务性能**：
   - 有效缓解 MEC 资源瓶颈，降低端到端延迟达 **3.73%~5%**。
   - 改善上行速率与能量效率，系统 utility 提升明显。

2. **Nash-AMRL 实现高效 operator-XUD 协同**：
   - 异步双代理设计契合 INC-E 的流程依赖（先 UL 再 DL）。
   - AHMRL 在大多数场景下取得最佳平衡，尤其适合中等规模部署。

3. **去中心化用户关联可行且高效**：
   - 基于干扰感知的游戏机制使 XUDs 自组织避开拥塞信道，无需中央控制器。

4. **MARL 显著优于启发式策略**：
   - 所有 MARL 模型均大幅超越 GM-RN、Equal、Proportional 策略，在公平性、鲁棒性、一致性方面优势明显。

### ⚠️ 局限性
- **依赖仿真环境**：未在真实 Metaverse 平台部署，缺乏实际网络抖动与硬件限制测试。
- **DT 建模简化**：假设 DT 与物理实体偏差恒定（30%），未考虑复杂动态漂移。
- **可扩展性待验证**：当前实验仅涉及 6 用户，更大规模下的训练稳定性未知。
- **安全与隐私未涉及**：未讨论数据共享中的隐私泄露风险。

### 🔮 未来工作方向
- 探索 **异构用户-节点配置** 与 **动态拓扑变化** 下的自适应策略。
- 引入 **联邦学习** 或 **privacy-preserving DT** 以增强安全性。
- 结合 **语义通信** 优化 2D→3D 数据传输效率。
- 扩展至 **6G URLLC + AI-native networking** 场景，支持更复杂的 Metaverse 交互。

---

> **总结一句话**：  
> 本文提出首个将 **Digital Twin** 与 **In-Network Computing** 融合的 Metaverse 协同计算框架，通过 **Stackelberg 博弈 + 异步多智能体强化学习（Nash-AMRL）**，实现了用户自主 offloading 与运营商联合资源调度的高效协同，在延迟、能效、公平性等方面全面超越现有方法。

</details>

---

### 4. [InfoSeeker: A Scalable Hierarchical Parallel Agent Framework for Web Information Seeking](https://arxiv.org/abs/2604.02971)

**Authors**: Ka Yiu Lee, Yuxuan Huang, Zhiyuan He, Huichi Zhou, Weilin Luo, Kun Shao, Meng Fang, Jun Wang  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.02971v1  

#### Abstract
Recent agentic search systems have made substantial progress by emphasising deep, multi-step reasoning. However, this focus often overlooks the challenges of wide-scale information synthesis, where agents must aggregate large volumes of heterogeneous evidence across many sources. As a result, most e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# InfoSeeker: A Scalable Hierarchical Parallel Agent Framework for Web Information Seeking —— 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前主流的 **agentic search** 系统虽然在 **multi-step reasoning** 上取得了显著进展，但在处理 **wide-scale information synthesis**（大规模信息综合）任务时面临三大瓶颈：

- **Context Saturation**：大量网页检索结果导致上下文窗口溢出。
- **Cascading Error Propagation**：早期错误在长链推理中不断累积放大。
- **High Latency**：串行执行模式导致端到端响应时间过长。

这些问题严重限制了 LLM 在需要聚合数十个异构来源数据的任务中的实用性。

### **提出了什么新方法或新思路**

作者提出 **InfoSeeker**，一个基于 **near-decomposability** 原则的 **分层并行代理框架**，包含三个层级：

- **Host Agent**：战略层，负责高层规划与长期记忆管理，仅接收摘要信息。
- **Manager Agents**：领域管理层（如 Search Manager、Browser Manager），负责任务分解、质量验证与结果聚合。
- **Worker Agents**：执行层，通过 **Model Context Protocol (MCP)** 并行调用工具（如搜索、浏览、代码执行）。

该架构实现了：
- **严格上下文隔离**（strict context isolation）：防止 Host 层被中间细节淹没。
- **MapReduce 式并行执行**：Manager 负责“Map”（分解）、Workers 并行执行、“Reduce”（聚合）。
- **动态协作机制**：例如 Search Manager 遇到反爬虫时可自动推荐 Browser Manager 接管。

### **相比现有方法的优势**

| 维度 | InfoSeeker | 传统方法（如 ReAct、Gemini DeepResearch） |
|------|-----------|----------------------------------------|
| **架构** | 分层并行（Hierarchical + Parallel） | 串行或浅层并行 |
| **上下文管理** | 抽象隔离，Host 仅看摘要 | 全局共享上下文，易饱和 |
| **容错性** | 错误被隔离在 Worker/Manager 层 | 错误会沿推理链传播 |
| **效率** | 支持大规模并发，3–5× 速度提升 | 顺序执行，延迟高 |

---

## 2. 核心实验方法和设置

### **使用了哪些数据集**

- **WideSearch** [Wong et al., 2025]  
  - 类型：结构化信息合成基准测试
  - 任务：从多个网页中提取实体与属性，填充完整表格（如米其林餐厅名单）
  - 特点：强调**广度**（width）、**完整性**（completeness）、跨源验证
  - 语言：英文（WideSearch-en）与中文（WideSearch-zh）

- **BrowseComp-zh** [Zhou et al., 2025]  
  - 类型：中文网页浏览能力评测
  - 任务：解决需多跳推理的历史谜题、医学诊断等复杂问题
  - 特点：真实中国互联网生态，含反爬、JS 渲染、混合语言内容
  - 语言：中文

### **实验设置和评估指标**

#### **评估指标**

| 数据集 | 指标 | 定义 |
|-------|------|------|
| **WideSearch** | `Success Rate` | 表格完全正确（精确匹配）的比例 |
| | `Row F1` | 实体级别的召回与精确率 F1 分数 |
| | `Item F1` | 属性级别的细粒度 F1 分数 |
| | `Avg@4`, `Max@4` | 多次运行下的平均与最高得分 |
| **BrowseComp-zh** | `Accuracy` | 正确回答最终问题的比例 |

#### **实现细节**
- **模型策略**：Host 和 Manager 使用 **gpt-5.1**（强推理），Worker 使用 **gpt-5-mini**（高吞吐）
- **工具集成**：通过 **MCP** 接入 Firecrawl（搜索）、Playwright（浏览器自动化）、沙箱 Python 等
- **并行规模**：Worker 池最大支持 17 个并行任务

### **基线方法对比**

| 类别 | 基线系统 |
|------|---------|
| **单智能体** | Claude Sonnet 4, Gemini 2.5 Pro, OpenAI o3-high |
| **端到端商业系统** | Gemini Deep Research, OpenAI Deep Research |
| **多智能体框架** | WebSailor, MiroThinker, BrowseMaster |
| **消融对照** | 单一 gpt-5.1 或 gpt-5-mini Agent（相同工具访问权限） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 模型/系统 | WideSearch-en (Success Rate Avg@4) | BrowseComp-zh (Accuracy) |
|----------|-------------------------------|------------------------|
| **InfoSeeker (Ours)** | **8.38%** | **52.9%** |
| OpenAI o3-high (Multi-Agent) | 5.10% | – |
| Gemini Deep Research | 4.30% | 42.9% |
| BrowseMaster | – | 46.5% |
| Claude Sonnet 4 (Thinking) | 3.60% | – |

> 注：在 WideSearch 上，InfoSeeker 达成 **66.7% 的相对提升**；在 BrowseComp-zh 上提升 **13.8% 绝对准确率**。

#### 更详细指标（WideSearch-en）：
- **Item F1 (Avg@4)**: **70.27%**（远超第二名的 65.44%）
- **Row F1 (Avg@4)**: **50.13%**（+30% 相对提升）

### **与基线方法的对比结果**

- InfoSeeker 在所有指标上均显著优于最强商业系统（Gemini/OpenAI Deep Research）和开源框架。
- 尤其在 **结构完整性**（Row F1）和 **事实准确性**（Item F1）方面优势明显，表明其聚合与验证机制有效。
- 在中文复杂网站环境中仍保持高性能，证明其对非英语生态的良好适应性。

### **消融实验结果**

#### **Worker 数量对性能的影响（Figure 4）**
- 当 Worker 池大小从 1 扩展到 17 时：
  - 端到端延迟从 **911 秒降至 162 秒**
  - 实现 **~5.7× 的加速比**
- 结论：弱耦合子任务可通过并行大幅压缩执行时间。

#### **单智能体对照实验（Table 5）**
| 系统 | Success Rate | Item F1 |
|------|------------|--------|
| gpt-5.1 单智能体 | 6.00% | 35.74% |
| gpt-5-mini 单智能体 | 4.00% | 33.28% |
| **InfoSeeker** | **12.50%** | **75.21%** |

> 即使使用更强 backbone 模型（gpt-5.1），单智能体也无法匹敌 InfoSeeker 的性能，说明**架构设计本身是性能跃迁的关键驱动因素**。

---

## 4. 关键结论和发现

### **主要发现**

1. **分层抽象 + 并行执行 是解决宽域信息合成的有效范式**  
   - 通过 **Host-Manager-Worker** 三层解耦，实现了推理深度与执行宽度的独立扩展。
   - 严格上下文隔离避免了 context saturation 和 error propagation。

2. **MapReduce 风格聚合机制提升了信息保真度**  
   - Manager 层的 reflection 与 aggregation 显著提高了最终输出的事实一致性。

3. **并行性带来显著效率增益**  
   - 在 WideSearch 上实现 **3–5× 的推理延迟降低**。
   - 支持大规模并发检索，适用于 real-world data-intensive 场景。

4. **模块化设计增强了系统鲁棒性和可扩展性**  
   - 新增 Manager 或 Worker 类型无需修改 Host 逻辑。
   - 不同 Manager 可协同工作（如 Search → Browser escalation）。

### **方法的局限性**

1. **依赖外部 API 与工具可用性**  
   - 性能受限于 MCP 工具的稳定性、速率限制与并发能力。

2. **提示工程依赖性强**  
   - 当前 Manager 行为高度依赖 hand-crafted prompts，泛化性受 backbone LLM 影响。

3. **极端数据量下仍有上下文瓶颈**  
   - 如 Figure 8 所示，在要求输出“全部 AMD Zen CPU”的任务中因 token 超限被迫返回样本表（Incomplete Failure）。

4. **实体链接能力不足**  
   - 如 Figure 7 所示，在疾病名称识别任务中将“变异型”误解为表型变异而非亚型命名，返回了 plausible 但非标准命名的答案。

### **未来工作方向**

1. **自动化任务分解与协调策略学习**  
   - 探索使用 **multi-agent reinforcement learning** 学习最优 decomposition policy，减少对 in-context learning 的依赖。

2. **训练轻量化专用模型**  
   - 开发更小、更高效的 Manager/Worker 模型以降低成本与延迟，提升部署可行性。

3. **增强长上下文处理能力**  
   - 结合 retrieval-augmented memory 或 external knowledge store 来突破 context window 限制。

4. **提升语义约束理解能力**  
   - 加强对“唯一实体”、“标准命名”等语义要求的理解，避免 answer-type mismatch。

---

> ✅ **总结一句话**：  
> InfoSeeker 通过引入 **分层并行架构** 与 **MapReduce 式执行流**，成功解决了 agentic search 中的 **context saturation、error propagation 与 high latency** 三大难题，在 wide-scale information seeking 任务上实现了 **效果与效率的双重突破**。

</details>

---

### 5. [AdaHOP: Fast and Accurate Low-Precision Training via Outlier-Pattern-Aware Rotation](https://arxiv.org/abs/2604.02525)

**Authors**: Seonggon Kim, Alireza Khodamoradi, Kristof Denolf, Eunhyeok Park  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.02525v1  

#### Abstract
Low-precision training (LPT) commonly employs Hadamard transforms to suppress outliers and mitigate quantization error in large language models (LLMs). However, prior methods apply a fixed transform uniformly, despite substantial variation in outlier structures across tensors. Through the first syst...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AdaHOP: Fast and Accurate Low-Precision Training via Outlier-Pattern-Aware Rotation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
低精度训练（**Low-Precision Training, LPT**）在大语言模型（**LLMs**）中面临一个核心挑战：**outliers**（异常值）——即稀疏但极端大的数值，会显著放大量化误差，导致训练不稳定和模型质量下降。  
现有方法（如Hadamard变换）通常采用**统一策略**对所有层和计算路径应用相同的变换（如固定方向的Hadamard Transform），忽略了不同张量（weights, activations, gradients）之间outlier结构的差异性。

本文指出，这种“一刀切”的策略是**根本性缺陷**：Hadamard变换的有效性高度依赖于其平滑方向是否与outlier的方向正交。若不匹配，不仅无法抑制outlier，甚至可能加剧量化误差。

---

### 提出了什么新方法或新思路
作者提出 **AdaHOP**（**Adaptive Hadamard transform with Outlier-Pattern-aware strategy**），一种基于outlier模式感知的自适应低精度训练框架。其核心思想是：

1. **系统性分析outlier模式**：首次对LLMs中的权重、激活和梯度进行跨层系统分析，识别出三种稳定的outlier模式：
   - **Row-wise (R)**：outliers集中在少数行
   - **Column-wise (C)**：outliers集中在少数列
   - **None (N)**：无明显outlier集中

2. **模式对驱动策略选择**：每个矩阵乘法（如 `Y = XW`, `Gw = GyX`）涉及两个操作数，形成 **outlier pattern pair**（如 `RC`, `CN`, `NN`）。AdaHOP为每种pair选择最优策略：
   - **IHT (Inner Hadamard Transform)**：适用于可被内维平滑的pair（如 `CN`, `NN`）
   - **IHT + OE (Outlier Extraction)**：当IHT无效时（如 `RN`, `RC`, `CC`），先将主导outliers提取到高精度路径（BF16），剩余部分用IHT处理
   - **全精度计算**：对注意力机制中特别敏感的`CC` pair，可选择完全用BF16计算（AdaHOP-Lv2）

3. **轻量级校准机制**：利用outlier模式在训练过程中**高度稳定**的特点，仅需30步BF16校准即可确定各张量的固定模式，无需运行时检测，开销极小。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **准确性** | 显著降低量化误差，实现与BF16相当的训练质量（loss gap < 0.01） |
| **效率** | 在AMD CDNA4架构上，通过融合Triton内核实现高达 **1.8× kernel加速** 和 **3.6× 内存压缩** |
| **通用性** | 策略选择基于理论分析和实证验证，适用于多种LLM架构（Llama, Instella等） |
| **硬件友好** | 实现为硬件感知的融合内核，最小化数据移动，充分利用CDNA4的混合精度并行能力 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **C4 (Colossal Clean Crawled Corpus)**：用于所有模型的预训练，序列长度为4096，batch size为64。

### 实验设置和评估指标
- **模型规模**：Llama3.2-1B, Llama3.2-3B, Instella-3B, Llama3.1-8B
- **优化器**：AdamW，学习率 4e-4，epsilon 1e-8，线性warmup 200步
- **训练步数**：根据Chinchilla缩放定律调整（1B: 40B tokens, 3B: 60B, 8B: 160B）
- **评估指标**：
  - **训练质量**：训练损失曲线及其与BF16的差距
  - **下游任务性能**：zero-shot准确率（PIQA, HellaSwag, ARC-Easy, LAMBADA）
  - **效率指标**：内存占用、吞吐量（tokens/s）、kernel延迟

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **BF16** | 全精度训练，作为性能上限 |
| **naive MXFP4** | 无任何outlier抑制的纯低精度训练 |
| **MXFP4+Hadamard** | 统一应用IHT |
| **Tseng et al.** | 使用随机Hadamard进行无偏梯度估计 |
| **HALO** | 应用OHT（Outer Hadamard Transform）于梯度路径 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **训练质量（Loss）**
- 如图1所示，AdaHOP在Llama3.2-1B和Instella-3B上均实现了**最小的loss gap**（< 0.01），几乎与BF16重合。
- 相比之下，其他MXFP4方法均有明显gap，尤其在训练后期。

#### ✅ **下游任务准确率（Zero-Shot Accuracy）**
| Model | Method | Average Accuracy (%) |
|-------|--------|------------------------|
| Llama3.2-1B | BF16 | 52.76 |
| | AdaHOP-Lv2 | **52.73** |
| Instella-3B | BF16 | 55.75 |
| | AdaHOP-Lv2 | **56.05**（超越BF16） |
| Llama3.1-8B | BF16 | 61.68 |
| | AdaHOP-Lv2 | **61.43**（接近BF16） |

> AdaHOP-Lv2在多个任务上**超过BF16**，表明其不仅能恢复精度，还能因更稳定的训练动态带来轻微提升。

#### ✅ **效率指标**
| Method | Memory (GB) | Throughput (tok/s) | Speedup vs BF16 |
|--------|-------------|--------------------|------------------|
| BF16 | 76.00 | 12,946 | 1.0× |
| AdaHOP-Lv1 | 20.94 | 13,247 | **1.59–1.80× kernel speedup** |
| AdaHOP-Lv2 | 28.04 | 13,134 | — |

- **内存压缩**：AdaHOP-Lv1实现 **3.6×** 内存压缩（76 → 20.94 GB）
- **kernel加速**：得益于低精度主路径和高效融合内核，AdaHOP在GEMM级别达到最高 **1.8× 加速**

#### ✅ **消融实验**
- **模式稳定性验证**（图4）：outlier模式在训练早期即稳定，支持一次性校准。
- **策略选择有效性**（图3）：IHT仅对`CR` pair有效，在`RC`, `RR`, `CC`等pair上无效甚至有害，验证了自适应策略的必要性。
- **OE预算影响**：k=64的选择由CDNA4的MFMA指令块大小决定，平衡了精度与效率。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Outliers具有结构性且稳定**：LLM中的outliers并非随机噪声，而是呈现**Row-wise, Column-wise, None**三种稳定模式，且在训练中保持不变。
2. **统一变换策略是次优的**：Hadamard变换的效果强烈依赖于其方向与outlier方向的对齐关系，**没有单一变换能适用于所有pattern pair**。
3. **自适应策略优于全局方法**：AdaHOP通过**校准+模式对决策**，实现了比HALO、Tseng等方法更低的量化误差和更高的训练质量。
4. **混合精度设计可高效实现**：通过硬件感知的Triton内核，AdaHOP在引入少量BF16计算的同时，仍能实现显著的内存节省和加速。

---

### 方法的局限性
1. **固定Hadamard矩阵**：当前使用固定的Walsh-Hadamard矩阵，未探索**learned rotation matrices**可能带来的进一步优化。
2. **模式分析范围有限**：目前仅覆盖Llama-family和Instella模型，尚未扩展至MoE架构（如Mixtral）或不同归一化方式的模型（如Gemma）。
3. **数值格式绑定**：当前框架基于**MXFP4**，虽具代表性，但向其他低精度格式（如INT4, FP8）的泛化需进一步验证。
4. **超参数固定**：outlier提取数量 `k=64` 是硬件驱动的固定值，缺乏基于每层outlier严重程度的自适应选择。

---

### 未来工作方向
1. **结合Learned Rotations**：将AdaHOP与**SpinQuant**等学习旋转矩阵的方法结合，实现数据自适应的变换。
2. **扩展至更多架构**：验证outlier模式在**Mixtral, Gemma, Qwen**等模型上的普适性。
3. **多格式支持**：将AdaHOP框架推广至**FP8, INT4, MXFP2**等更激进的低精度格式。
4. **自适应k选择**：研究基于每层outlier强度（如kurtosis）动态调整`k`的策略，以优化精度-效率权衡。
5. **端到端训练加速**：当前kernel加速未完全转化为端到端训练速度提升，未来可结合注意力、归一化等模块的低精度化，实现全面加速。

---

> **总结**：AdaHOP通过揭示LLM中outlier的**结构性与稳定性**，挑战了传统LPT中“统一变换”的范式，提出了首个**outlier-pattern-aware**的自适应训练框架。其实验结果证明，**理解并利用outlier结构**，而非简单压制，是实现高效、稳定低精度训练的关键路径。

</details>

---

### 6. [Fast NF4 Dequantization Kernels for Large Language Model Inference](https://arxiv.org/abs/2604.02556)

**Authors**: Xiangbo Qi, Chaoyi Jiang, Murali Annavaram  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.02556v1  

#### Abstract
Large language models (LLMs) have grown beyond the memory capacity of single GPU devices, necessitating quantization techniques for practical deployment. While NF4 (4-bit NormalFloat) quantization enables 4$\times$ memory reduction, inference on current NVIDIA GPUs (e.g., Ampere A100) requires expen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Fast NF4 Dequantization Kernels for Large Language Model Inference》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前大型语言模型（LLMs）参数量已超过单个GPU的显存容量，因此广泛采用 **NF4（4-bit NormalFloat）量化** 技术以减少内存占用。然而，NVIDIA Ampere架构（如A100）不支持原生4-bit计算，推理时必须将NF4权重**反量化为FP16格式**，这一过程涉及大量对全局内存（global memory）中查找表（LUT）的访问，造成严重的性能瓶颈。

论文指出，在Qwen3-32B等模型上，**dequantization操作占端到端延迟的21–40%**，成为制约推理效率的关键因素。

### 提出了什么新方法或新思路
作者提出了一种**轻量级共享内存优化方案**，通过以下两个核心技术改进NF4反量化内核：

- **共享内存加载策略（Shared Memory Loading Strategy）**  
  利用每个thread block仅需64字节即可存储完整的16元素NF4 LUT的特点，由block中的一个线程（thread 0）将LUT从constant memory加载至shared memory，并通过`_syncthreads()`同步确保所有线程可见。此后所有线程在反量化过程中直接从shared memory读取LUT，避免重复访问高延迟的global memory。

- **简化索引计算逻辑（Simplified Index Computation）**  
  替代原有基于4层条件分支树的复杂索引方式，改用位运算（bit masking and shifting）实现直接寻址，将每项权重的索引指令从7条减少至2条，**消除warp divergence**，提升SIMT执行效率。

### 相比现有方法的优势
| 维度 | 本工作优势 |
|------|------------|
| **性能提升** | 实现2.0–2.2×的kernel级加速，最高达1.54×端到端速度提升 |
| **资源开销低** | 每个thread block仅使用64 bytes shared memory，无额外显存负担 |
| **兼容性强** | 完全兼容HuggingFace Transformers与BitsAndBytes生态，无需离线预处理或模型转换 |
| **部署便捷** | 插件式设计，可即插即用，适用于现有生产系统 |
| **通用性好** | 在不同模型规模（27B–70B）、batch size下均保持稳定增益 |

相比需要复杂kernel fusion或专用硬件支持的方法（如[9][10]），本文方法更轻量、实用且易于集成。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **GSM8K dataset**：用于生成输入prompt进行推理测试，评估真实场景下的性能表现。

### 实验设置和评估指标
- **硬件平台**：  
  - 单块NVIDIA A100-80GB GPU  
  - AMD EPYC 7513 CPU（32核），64GB RAM  
  - CUDA 12.6, PyTorch 2.1, BitsAndBytes 0.47.0
- **控制变量**：  
  - GPU频率锁定为最大值（1410 MHz）以消除动态调频影响  
  - 固定随机种子保证结果可复现
- **测试流程**：  
  - 每次运行前执行一次warm-up推理pass  
  - 使用PyTorch Profiler（基于CUPTI接口）采集微秒级精度的时间数据，包含kernel launch overhead和memory transfer
- **评估指标**：  
  - **End-to-end latency**（端到端延迟）
  - **Throughput (tokens/sec)**（吞吐量）
  - **Kernel-level speedup**（反量化内核加速比）

### 基线方法对比
- **Baseline**：开源的BitsAndBytes实现中的原始NF4 dequantization kernel
- **Optimized Method**：本文提出的共享内存+简化索引版本
- **对比模型**：
  - Gemma 27B
  - Qwen3 32B
  - Llama3.3 70B
- **Batch Sizes**：2, 4, 8, 16, 32, 64

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Kernel-Level Speedup（反量化内核加速）
| Batch Size | Gemma 27B | Qwen3 32B | Llama3.3 70B |
|------------|-----------|-----------|--------------|
| 2          | 2.10×     | 2.20×     | 2.04×        |
| 4          | 2.10×     | 2.19×     | 2.04×        |
| 8          | 2.11×     | 2.19×     | 2.04×        |
| 16         | 2.10×     | 2.19×     | 2.03×        |
| 32         | 2.11×     | 2.19×     | 2.05×        |
| 64         | 2.08×     | 2.15×     | 2.03×        |
| **Average**| **2.10×** | **2.19×** | **2.04×**    |

> 所有配置下均实现 **2.0–2.2× 的内核级加速**，说明优化有效针对的是底层内存访问瓶颈，而非特定模型结构。

#### ✅ End-to-End Performance Improvement（端到端加速）
- **Llama3.3 70B**：平均提速 **1.52×**，最高达 **1.54×**（batch=2）
- **Qwen3 32B**：平均提速 **1.18×**，峰值 **1.29×**（batch=32）
- **Gemma 27B**：平均提速 **1.10×**，随batch增大增至 **1.32×**（batch=64）

> 更大模型受益更多，因其反量化耗时占比更高。

#### ✅ Throughput 提升（tokens/sec）
- **Llama3.3 70B @ batch=2**：从 ~450 → ~690 tokens/sec（+1.54×）
- **Qwen3 32B @ batch=32**：从 283 → 368 tokens/sec（+1.30×）
- **Gemma 27B @ batch=64**：从 506 → 633 tokens/sec（+1.25×）

### 与基线方法的对比结果
- 反量化内核时间下降约 **50–55%**
- 全局内存访问次数减少 **64× per thread block**
- 指令数减少 **71%**，主要来自消除分支判断
- shared memory访问延迟仅 **19 cycles** vs global memory **290 cycles**，获得 **12–15× 的访存延迟优势**

### 消融实验结果（隐含分析）
虽然未明确列出消融实验表格，但文中通过多个维度验证了各组件的有效性：
- **共享内存 vs 全局内存访问**：通过profiling确认baseline中LUT访问是主要延迟来源
- **简化索引 vs 分支树**：指令数从7降至2，且消除warp divergence，显著提高warp执行效率
- **单线程加载 vs 多线程协作加载**：实验证明小数据量下协调开销大于收益，单线程加载最优

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Dequantization is a major bottleneck**：即使计算简单，频繁的global memory LUT访问使NF4反量化成为端到端推理的关键性能瓶颈。
2. **Memory hierarchy matters**：合理利用shared memory可带来高达15×的访存延迟改善，远超算法层面的小幅优化。
3. **Lightweight optimizations can yield large gains**：仅使用64 bytes shared memory + 简化索引逻辑，就能实现2倍以上的kernel加速。
4. **Scalability with model size**：模型越大，反量化占比越高，优化带来的端到端收益越明显（70B > 32B > 27B）。
5. **Compatibility enables adoption**：无需修改模型或训练流程，即可在HuggingFace生态系统中无缝部署。

### 方法的局限性
- 当前优化集中在**权重反量化阶段**，未涉及activation或KV Cache的量化/反量化路径。
- 对非常小的batch size（如=1）可能增益有限，因其他模块（如sampling）占比上升。
- 依赖GPU shared memory机制，难以直接迁移到CPU或其他非CUDA平台。
- 未探索更低比特（如2-bit）或混合精度场景下的扩展性。

### 未来工作方向
- 将类似优化应用于**AWQ、GPTQ等其他量化方案**的反量化流程。
- 结合**kernel fusion技术**，进一步融合dequantize + matmul + silu/gelu等操作，减少中间数据搬运。
- 探索**compiler-level自动优化**，将此类模式识别并编译为高效代码。
- 面向下一代GPU架构（如Hopper、Blackwell）设计更深层次的软硬协同优化策略。

---

> **总结一句话**：  
> 本文通过巧妙利用GPU内存层次结构和简化控制流，提出一种极轻量但高效的NF4反量化优化方案，在几乎零工程成本的前提下实现了高达2.2×的内核加速和1.54×的端到端性能提升，为大规模LLM在现有GPU基础设施上的高效推理提供了即插即用的解决方案。

</details>

---

### 7. [STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation](https://arxiv.org/abs/2604.02756)

**Authors**: Zijin Liu, Xu Geng, Wenshuai Xu, Xiang Zhao, Yan Xia, You Song  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.02756v1  

#### Abstract
Accurate crowd simulation is crucial for public safety management, emergency evacuation planning, and intelligent transportation systems. However, existing methods, which typically model crowds as a collection of independent individual trajectories, are limited in their ability to capture macroscopi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 crowd simulation 方法存在以下局限性：
- **微观建模主导**：主流方法将人群视为独立个体轨迹的集合，忽略了宏观物理规律（如质量守恒），导致长期模拟中误差累积、稳定性差。
- **物理一致性不足**：纯 data-driven 的深度学习方法（如 GNN、diffusion models）虽能捕捉复杂模式，但常产生违反物理规律的行为（如不合理的拥堵或碰撞）。
- **推理效率低**：基于 diffusion 或自回归生成的方法计算开销大，难以支持大规模、实时仿真。

### **提出的新方法与思路**
作者提出了 **Spatio-Temporal Decoupled Differential Equation Network (STDDN)**，一种融合宏观物理约束与微观轨迹预测的新型框架。其核心思想是：
- 将人群视为连续介质，引入流体力学中的 **continuity equation** 作为强物理先验，指导微观轨迹演化。
- 构建一个由 **Neural ODE** 驱动的宏观密度场演化模块，通过微分方程对密度变化进行建模。
- 设计 **density-velocity 耦合机制**，使个体运动驱动密度场动态更新，实现“宏观引导微观”的闭环。

### **相比现有方法的优势**
| 维度 | STDDN 的优势 |
|------|--------------|
| **物理一致性** | 显式嵌入 continuity equation，确保质量守恒，减少非物理行为（如突现/消失、异常聚集）。 |
| **长期稳定性** | 宏观物理约束有效抑制误差在时间上的累积，提升长时预测鲁棒性。 |
| **推理效率** | 单步前向传播即可完成模拟，避免 diffusion 模型多步去噪，显著降低延迟。 |
| **可解释性** | 引入动态图结构建模密度通量（flux），增强模型决策过程的物理可解释性。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个真实世界轨迹数据集上进行评估：
| 数据集 | 场景描述 | 特点 |
|-------|--------|------|
| **GC** | 高密度校园场景 | 密集交互，平均密度达 0.094 m⁻² |
| **UCY** | 包含 ZARA1、ZARA2 和 UCY 子场景 | 多样化行人行为 |
| **ETH** | 开放街道环境 | 行人速度快（均值 2.293 m/s） |
| **HOTEL** | 酒店入口区域 | 中等密度，动态进出频繁 |

所有数据经坐标转换与立方插值统一至 0.08s 时间步长。

### **实验设置与评估指标**

#### **训练策略**
- 使用 **joint loss** 同时监督速度预测（微观）与密度演化（宏观）：
  $$
  \mathcal{L}_{\text{joint}} = \lambda_1 \|v - v_{\text{gt}}\| + \lambda_2 \|p - p_{\text{gt}}\|
  $$
- 利用 Neural ODE 在训练阶段建模密度连续演化，推理时仅使用训练好的 $f_\theta$ 进行自回归预测。

#### **评估指标**
| 类别 | 指标 | 描述 |
|------|-----|------|
| **准确性** | MAE, OT, FDE, DTW, MMD | 分别衡量位置误差、轨迹分布差异、终点偏差、形状相似性和分布一致性 |
| **物理合理性** | #Colli, DEA | 碰撞次数、局部密度估计准确率 |
| **效率** | Latency (ms), FPS, #Pars, GFLOPs | 推理延迟、帧率、参数量、浮点运算量 |

### **基线方法对比**
分为三类进行比较：
- **Physics-based**: SFM, CA  
- **Data-driven**: STGCNN, PECNet, MID  
- **Physics-guided**: PCS, NSP, SPDiff  

其中 SPDiff 是当前 SOTA 的 physics-informed diffusion 方法。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1 & 2）**

| Dataset | Method | MAE ↓ | OT ↓ | Latency (ms) ↓ |
|--------|--------|-------|------|----------------|
| GC     | SPDiff | 0.9116 | 1.3925 | 206.99 |
|        | **Ours** | **0.8875** | **1.3582** | **86.85** |
| UCY    | SPDiff | 1.8760 | 4.0564 | 471.05 |
|        | **Ours** | **1.7747** | **3.6503** | **44.66** |
| ETH    | SPDiff | 0.5527 | 0.8706 | 81.41 |
|        | **Ours** | **0.5185** | **0.6918** | **30.57** |
| HOTEL  | SPDiff | 0.3380 | 0.1646 | 68.57 |
|        | **Ours** | **0.2952** | **0.1445** | **17.50** |

> ✅ **结论**：STDDN 在所有数据集上均取得最优 MAE 和 OT 性能，且推理延迟大幅下降（最高提速 **90%**）。

### **与基线方法的对比结果**
- 相较于 **SPDiff**（diffusion-based SOTA）：
  - 平均提升 **2.6% ~ 12.7%** 的 MAE 准确率；
  - 推理速度提升 **50% ~ 90%**；
  - 参数量更少（如 UCY 上仅 0.07M vs. 0.22M）。
- 相较于 **纯 data-driven 方法**（如 MID）：
  - 显著降低碰撞数（#Colli）和密度误差（DEA），体现更强的物理合理性。
- 相较于 **physics-based 方法**（如 SFM）：
  - 更好地建模非线性、随机性行为，在高密度场景下表现优越。

### **消融实验结果（Ablation Study, Table 3）**
验证各组件的重要性：

| 变体 | GC MAE ↑ | UCY MAE ↑ | 说明 |
|------|----------|-----------|------|
| w/o ODE | 1.3784 | 2.4867 | 移除 Neural ODE 后误差显著上升 → **ODE 对抑制误差累积至关重要** |
| w/o Cross-net | 0.9784 | 1.8926 | 忽略跨网格检测削弱质量守恒 → **CGD 模块有效建模通量** |
| w/o NN loss | 1.2387 | 1.9327 | 仅依赖物理损失无法拟合复杂行为 → **需结合 data-driven 学习** |
| w/o NE | 0.8921 | 1.7917 | 节点嵌入有助于空间关系建模 |
| Discrete NN | 0.8875 | 1.7747 | 性能与完整模型相当 → 当前任务更适合离散时空建模 |

> 🔍 **发现**：ODE 框架和 CGD 模块是性能提升的关键；而 Euler solver 已足够，无需更高阶求解器（Dopri5/RK4 反而导致过拟合）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **宏观物理约束能显著提升 crowd simulation 的稳定性和准确性**：
   - 引入 continuity equation 作为结构性归纳偏置，有效缓解长期预测中的误差累积问题。
2. **“宏观引导微观”范式优于传统“微观叠加”方式**：
   - 将个体运动与群体密度耦合建模，实现了物理一致性与表达能力的统一。
3. **高效推理成为可能**：
   - 不依赖 diffusion 的多步生成，单步前向即可输出高质量轨迹，适合实际部署。
4. **不同iable 设计保障端到端训练**：
   - DDM（Differentiable Density Mapping）和 CGD（Continuous Grid Detection）解决了离散化带来的梯度不连续问题。

### **方法的局限性**
1. **边界效应未完全建模**：
   - 当前 continuity equation 未考虑人群进出（source/sink terms），在开放系统中可能破坏质量守恒。
2. **网格离散化带来计算负担**：
   - 尽管 NE 模块降低了参数复杂度，但在超大场景下仍受限于内存。
3. **强物理约束可能压制隐含运动模式**：
   - 过于严格的物理正则化可能限制模型学习某些非常规但合理的人类行为。

### **未来工作方向**
1. **扩展 continuity equation 为开放系统形式**：
   $$
   \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = S(\mathbf{x},t)
   $$
   其中 $S$ 表示入口/出口源项，可用于预测人流热点区域。
2. **探索软约束机制**：
   - 使用 adaptive weighting 或 uncertainty-aware loss 动态平衡 data-driven 与 physics-driven 成分。
3. **改进空间表示**：
   - 引入 continuous spatial representation（如 implicit neural fields）或多尺度 graph 结构，突破 grid discretization 限制。
4. **优化大规模计算效率**：
   - 应用稀疏化、模型压缩或并行加速技术，推动方法在城市级仿真中的应用。

---

> 📌 **总结一句话**：  
> **STDDN 成功将宏观物理定律（continuity equation）与深度学习相结合，构建了一个高精度、高效率、物理一致的 crowd simulation 新范式，为智能交通、应急管理等实际场景提供了可靠的技术支撑。**

</details>

---

### 8. [Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms](https://arxiv.org/abs/2604.02927)

**Authors**: Andreas Boltres, Niklas Freymuth, Benjamin Schichtholz, Michael K\"onig, Gerhard Neumann  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.02927v1  

#### Abstract
Routing algorithms are crucial for efficient computer network operations, and in many settings they must be able to react to traffic bursts within milliseconds. Live telemetry data can provide informative signals to routing algorithms, and recent work has trained neural networks to exploit such sign...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统路由算法（如 OSPF、EIGRP）在面对突发流量或网络拓扑变化时反应缓慢，难以满足现代数据中心和广域网对**毫秒级响应**的需求。虽然已有研究尝试使用 **Machine Learning (ML)** 或 **Reinforcement Learning (RL)** 进行流量感知路由优化，但存在以下关键缺陷：

- **忽略通信延迟**：许多神经路由方法假设可以获取无延迟的全局网络状态（“birds-eye view”），这在真实网络中不可行。
- **纯局部观测限制**：部分分布式方法仅依赖本地遥测数据，缺乏协同能力。
- **训练与部署不一致**：训练时忽略推理和通信延迟，导致模型在实际部署中性能下降。

因此，如何设计一个**在真实延迟约束下仍能有效工作的神经路由算法**，是当前研究的空白。

### 提出的新方法和新思路
本文提出以下核心贡献：

#### （i）构建了一个**延迟感知的仿真框架**
- 将遥测感知路由建模为一个**延迟感知的闭环控制问题**（delay-aware closed-loop control）。
- 显式建模了**通信延迟**（state propagation）和**推理延迟**（inference delay），使训练环境更贴近现实。
- 支持多种部署模式（Central-Single, Local-Multi 等），用于系统性评估不同架构的影响。

#### （ii）提出了新型神经路由算法 **LOGGIA**
- **全称**：LOg-space link weight prediction on Graphs with Guided update epochs and Implicit-Alpha entropy adaptation.
- **核心机制**：
  - 使用 **Graph Neural Network (GNN)** 处理带属性的拓扑-遥测图（attributed topology-and-telemetry graphs）。
  - 在**对数空间预测链路权重**（log-space link weights），提升数值稳定性。
  - 采用两阶段训练协议：
    1. **Imitation Learning (IL) 预训练**：模仿静态路由策略（如 EIGRP）进行 warm-start。
    2. **On-policy Reinforcement Learning (PPO/MAPPO)**：进一步优化以最大化吞吐量。

#### （iii）揭示了关键部署原则
- 实验表明，在真实延迟环境下，**完全分布式的 Local-Multi 架构**（每个路由器独立观察并决策）表现最佳。
- 中心化控制因引入额外通信开销而性能劣化。

### 相比现有方法的优势
| 方面 | 现有方法 | LOGGIA |
|------|--------|-------|
| 延迟建模 | 忽略通信/推理延迟 | 显式建模，训练与部署一致 |
| 观测方式 | 全局或纯局部 | 支持延迟感知的分布式观测 |
| 可扩展性 | 多数无法泛化到新拓扑 | 单一小型拓扑训练即可泛化至100节点网络 |
| 性能稳定性 | 训练不稳定，方差大 | IL预训练显著降低方差 |

---

## 2. 核心实验方法和设置

### 使用的数据集与网络拓扑
实验在多个合成与真实网络拓扑上进行：

| 拓扑 | 节点数 | 描述 |
|------|--------|------|
| `mini5` | 5 | 合成小规模网络 |
| `B4` | 12 | Google 数据中心互联网络 |
| `GEANT` | 27 | 欧洲科研教育网络（2001年版本） |
| `nx-XS`, `nx-S`, `nx-M`, `nx-L` | 6–100 | 合成可变规模拓扑族，用于泛化测试 |

> 所有拓扑均通过 **ns-3** 进行包级模拟，并注入混合 TCP/UDP 流量。

### 实验设置
- **时间粒度**：每步 `T = 5ms`，每episode持续2秒（共400步）。
- **流量模型**：
  - 80% TCP 流（模拟真实应用）
  - 20% UDP 流（恒定比特率）
  - 流大小与到达时间基于真实测量数据生成。
- **遥测输入**：通过 **In-Band Network Telemetry (INT)** 收集节点与链路状态（如队列负载、丢包、延迟等）。

### 评估指标
- **主指标**：**Goodput（MB）** —— 成功送达的数据总量（即 delivery rate）。
- 辅助指标：
  - 平均延迟（Delay, ms）
  - 队列负载（Queue Load, %）
  - TCP 丢弃包数量（TCP Discard, MB）

### 基线方法对比
#### 静态最短路径（SP）基线
- `SPRIP`：最小跳数路由
- `SPEIGRP`：基于带宽/延迟的复合度量
- `SPOSF`：基于带宽的度量

> 报告各场景下表现最好的 SP 基线作为比较基准。

#### 神经路由基线
- `MAGNNETO`：集中式多智能体 GNN 路由
- `FieldLines` 和 `M-Slim`：来自 Boltres et al. (2024)，忽略通信延迟

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Figure 5 和 Tables）

| 方法 | B4 拓扑 Goodput (MB) | GEANT 拓扑 Goodput (MB) | 是否超越 SP 基线？ |
|------|------------------------|----------------------------|--------------------|
| SPRIP / SPEIGRP | ~329 | ~442 | ❌（基准） |
| MAGNNETO | 311.6 | 420.9 | ❌ |
| FieldLines | 296.2 | 450.3 | ❌（仅在GEANT略优） |
| M-Slim | 311.0 | 438.2 | ❌ |
| **LOGGIA (ours)** | **333.4** (+1.2%) | **460.7** (+4.2%) | ✅ |

> 注：所有神经方法在**延迟感知设置**下评估；LOGGIA 是唯一稳定优于 SP 基线的方法。

### 与基线方法的对比结果
- **在延迟感知环境中，几乎所有现有神经路由算法性能退化甚至不如静态路由**。
- LOGGIA 在所有拓扑上均**显著且稳定地优于所有基线**，尤其在复杂拓扑（如 GEANT）上优势明显。
- 方差更低（见 Figure 5 箱线图），说明训练更稳定。

### 消融实验结果（Ablation Studies）

#### （1）LOGGIA 架构组件消融（Figure 11）
| 组件 | 移除后影响 |
|------|----------|
| 对数空间权重预测 | 性能下降明显 |
| 直接操作原始图（非line graph） | 有轻微正向作用 |
| 更深的 GNN（L=4, d=32） | 显著提升表达能力 |

> 表明 LOGGIA 的三项设计均有贡献。

#### （2）训练机制消融（Figure 12）
- **Early stopping + Max-Entropy Exploration** 显著提升 LOGGIA 性能。
- 价值函数改进（如输入依赖）未带来收益，说明**值函数学习不是瓶颈**。

#### （3）预训练机制对比（Figure 14）
| 训练方式 | 效果 |
|---------|------|
| 仅 PPO | 可收敛但方差大 |
| 仅 IL（DAgger-style） | 不足以独立训练出好策略 |
| **IL → PPO** | ✅ 显著提升最终性能与稳定性（warm-start效果） |
| BC（Behavioral Cloning）→ PPO | 效果弱于交互式 IL |

> 强调**交互式 IL 预训练的重要性**。

#### （4）部署模式影响（Figure 6 & 17）
| 部署模式 | 性能排序（从高到低） |
|----------|---------------------|
| Local-Multi（分布式观测+决策） | ✅ 最佳 |
| Central-Multi | 次之 |
| Central-Single | 较差 |
| Birdseye-Single（无延迟理想情况） | 最高但不现实 |

> **Local-Multi 是唯一能让 LOGGIA 超越 SP 基线的延迟感知部署模式**。

#### （5）硬件速度影响（Table 5）
| CPU 型号 | 推理时间 | 入ac=1 时 Goodput |
|---------|----------|------------------|
| Xeon 6780E（默认） | 7.47ms | 440.6 MB |
| Xeon Gold 6448Y（更快） | 4.99ms (-33%) | **444.6 MB** (+0.9%) |

> 表明**更快的硬件可通过减少推理延迟直接提升路由性能**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **延迟建模至关重要**：忽略通信与推理延迟会导致神经路由算法在真实场景中失效。
2. ✅ **LOGGIA 是首个在延迟感知环境下稳定优于传统路由的神经算法**。
3. ✅ **完全分布式架构（Local-Multi）最优**：每个路由器本地观测并决策，避免中心化瓶颈。
4. ✅ **IL 预训练有效提升训练稳定性**，尤其对于复杂任务。
5. ✅ **单一小型拓扑训练即可泛化至大型未知网络**（如从 5 节点训练推广到 100 节点）。
6. ⚠️ **推理延迟直接影响性能**：更快的硬件或模型压缩技术将成为实际部署的关键。

### 方法的局限性
- 当前仅支持**单路径路由**（single-path IP routing），不支持 ECMP 或 multipath。
- 决策基于单一成本度量，未考虑 BGP 等复杂策略。
- 假设 out-of-band 控制信道具有无限带宽，未建模控制平面拥塞。
- 未处理转发表更新延迟或 in-band telemetry 开销。

### 未来工作方向
- 扩展至 **multipath routing** 和 **segment routing** 场景。
- 结合 **differentiable routing layers** 实现端到端优化。
- 探索 **in-band 控制消息压缩** 以降低通信开销。
- 将 MDP 形式化与 **real-time MDP** 或 **networked MDP** 理论结合，建立更强理论基础。
- 设计针对 TCP 流的专用策略，减少乱序导致的重传。

---

> **总结**：本文填补了“理论强大但部署受限”的神经路由研究鸿沟，首次展示了在**真实延迟约束下仍具实用价值的神经路由方案**。LOGGIA 不仅性能优越，且揭示了“**去中心化 + 延迟建模 + IL warm-start**”是未来智能路由系统的关键设计范式。

</details>

---

### 9. [Revealing the Learning Dynamics of Long-Context Continual Pre-training](https://arxiv.org/abs/2604.02650)

**Authors**: Yupu Liang, Shuang Chen, Guanwei Zhang, Shaolei Wang, Suncong Zheng  
**Category**: cs.CL  
**Published**: 2026-04-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.02650v1  

#### Abstract
Existing studies on Long-Context Continual Pre-training (LCCP) mainly focus on small-scale models and limited data regimes (tens of billions of tokens). We argue that directly migrating these small-scale settings to industrial-grade models risks insufficient adaptation and premature training termina...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Revealing the Learning Dynamics of Long-Context Continual Pre-training

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本论文针对**工业级大模型**在进行 **Long-Context Continual Pre-training (LCCP)** 时存在的三大核心挑战：

1. **小规模研究无法泛化到工业场景**：现有研究多基于小型模型（如7B）和有限数据（数十亿token），其结论难以迁移到百亿参数级别的工业级模型。
2. **传统评估指标存在“欺骗性饱和”（Deceptive Saturation）**：常用下游任务如 **Needle-in-a-Haystack (NIAH)** 在早期即显示性能饱和，但实际上模型仍在持续学习长上下文能力。
3. **缺乏对训练动态的系统性监控机制**：缺少能够实时反映模型内在收敛状态的轻量级诊断工具。

---

### 🚀 提出了什么新方法或新思路

作者提出了一个**多层次分析框架**，从三个层面系统揭示 LCCP 的学习动态：

| 层面 | 方法 | 创新点 |
|------|------|--------|
| **Behavioral Level**（行为层） | 使用轻量级 **Supervised Fine-Tuning (SFT) probing** 评估模型在下游任务中的表现 | 避免全量SFT开销，实现高效“探针式”评估 |
| **Probabilistic Level**（概率层） | 将传统的二值化 NIAH 改造为基于 **Perplexity (PPL)** 的连续评估指标（Continuous NIAH） | 揭示模型生成置信度的渐进提升，避免“假饱和”误导 |
| **Mechanistic Level**（机理层） | 分析注意力头中“**retrieval heads**”的演化规律 | 发现特定注意力头可作为低资源、高相关性的训练进度监测器 |

---

### 🔍 相比现有方法的优势

| 维度 | 本文方法优势 |
|------|--------------|
| **评估粒度更细** | 传统 NIAH 是离散准确率 → 本文 PPL 提供连续、细粒度反馈 |
| **预测性更强** | PPL 和 retrieval head 指标与下游 SFT 性能的相关性显著高于 NIAH score |
| **计算成本更低** | retrieval head 可直接在 base model 上计算，无需执行 SFT 或复杂 benchmark |
| **适用于工业部署** | 基于真实生产级模型（Hunyuan-A13B, 80B total params）和超大规模训练轨迹（200B tokens）验证，具备强实用性 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据类型 | 来源 | 占比 / 规模 | 说明 |
|--------|------|------------|------|
| **Long-context documents** (>32K tokens) | Common Crawl, Books, arXiv, Code, Wikipedia | 占比 75%，总计约 150B tokens | 强调技术性和结构化信息以增强长程依赖建模 |
| **Short-context data** | 同初始预训练分布 | 占比 25% | 缓解灾难性遗忘 |
| **Evaluation datasets** | RULER, MRCR, LongBio | 共计 ~440K SFT samples (~0.25B tokens) | 覆盖通用能力、推理与长文本理解任务 |

> ⚠️ 所有评估样本长度严格控制在 32K–64K 区间，均匀分布。

---

### ⚙️ 实验设置

| 项目 | 设置详情 |
|------|----------|
| **Base Model** | Hunyuan-A13B，Sparse MoE 架构，总参数 80B，激活参数 13B/token |
| **Context Extension** | 从 32K → 64K，通过调整 RoPE base frequency（500K → 2M）实现 |
| **Training Tokens** | 总共 200B tokens |
| **Learning Rate** | 恒定学习率 $1.2 \times 10^{-5}$ |
| **Batch Size** | Global batch size: 16M tokens |
| **SFT Probing** | 学习率从 $2\times10^{-5}$ 衰减至 $5\times10^{-6}$，训练2轮，batch size 1M tokens |

---

### 📊 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **Pass@3** | SFT后在 RULER/MRCR/LongBio 上的通过率 | 衡量下游任务性能 |
| **NIAH Score** | 是否正确输出 needle 内容（0/1） | 传统评估方式，用于对比“欺骗性饱和”现象 |
| **NIAH PPL** | 对答案部分 token 计算平均 Perplexity | 连续衡量生成置信度，反映内在进展 |
| **# of Retrieval Heads** | 注意力头中 retrieval score > 0.1 的数量 | 反映模型检索能力的结构性演化 |
| **Avg. Retrieval Score** | 所有 retrieval heads 的平均得分 | 衡量整体 copy-paste 能力强度 |

---

### 🔁 基线方法对比

本文未直接对比多个外部模型，而是将**不同训练阶段的 checkpoint** 视为“基线”，重点比较：

- 不同训练 token 数量下的性能变化（0B → 200B）
- NIAH Score vs. NIAH PPL 的收敛趋势差异
- retrieval head 动态与 SFT 结果的相关性

本质上是**纵向自我对照实验**，强调训练过程的动态演化而非横向模型比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）Behavioral Level：SFT 探针结果（图1）

| Benchmark | 初始（~0B） | 峰值（~100B） | 达到饱和所需 token |
|---------|-------------|----------------|--------------------|
| RULER | 68.68 | 79.44 | ≥100B |
| MRCR / LongBio | 明显增益，边际收益递减始于50B | —— | >100B 才稳定 |

> 💡 工业级模型需 **超过100B tokens** 才能完成有效 LCCP，远高于学术研究中的几十亿量级。

---

#### （2）Probabilistic Level：NIAH PPL vs NIAH Score（图3）

| 指标 | 表现 |
|------|------|
| **NIAH Score** | 在 **20B tokens** 后接近饱和（100% accuracy），之后无变化 |
| **NIAH PPL** | 持续下降直至 **150B tokens** 才趋于平稳 |

> ❗ 显示典型的“**欺骗性饱和**”：NIAH 准确率已达顶峰，但 PPL 仍持续优化，表明模型仍在学习更可靠的生成模式。

---

#### （3）Pearson 相关性分析（表3）

| 指标 | 与下游 SFT 平均相关性 |
|------|------------------|
| **NIAH Score** | 0.7486 |
| **NIAH PPL** | **-0.8210**（负相关，越低越好） |

> ✅ **NIAH PPL 与下游性能的相关性更高**，是更忠实的内在收敛指示器。

---

#### （4）Mechanistic Level：Retrieval Heads 演化（图4–5）

| 指标 | 发现 |
|------|------|
| **Retrieval Heads 数量** | 随训练 token 增加而上升（从 ~10 到 >25） |
| **Avg. Retrieval Score** | 显著提高，反映 copy-paste 能力增强 |
| **与 SFT 性能相关性**（表4） |  
| - # of Retrieval Heads | avg. corr: **0.7428**  
| - Avg. Retrieval Score | avg. corr: **0.7878**

> ✅ retrieval head 指标可作为**低资源、高效率的训练监控代理指标**。

---

#### （5）PPL Scaling Law（图6）

发现 PPL 与训练 token 数满足近似线性关系：

$$
\text{PPL} = A \cdot \log(N) + B
$$

其中 $N$ 为训练 token 数。该规律在中文、英文、代码语料上均成立，符合经典的 **Scaling Law**。

---

#### （6）消融实验性质的结果（非标准消融，但有深入分析）

| 分析主题 | 主要发现 |
|--------|--------|
| **Lost-in-the-Middle 缓解**（图7） | LCCP 显著降低中间位置信息的 PPL，缓解“middle遗忘”问题 |
| **抗干扰能力增强**（图8–9） | 随着训练推进，模型对前置干扰文本长度的敏感性大幅下降 |
| **Retrieval Head 稳定性**（表5） | Top-30 retrieval heads 在后期训练中高度稳定（overlap ≥ 93%），功能身份在初期已基本确定 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **工业级 LCCP 必须进行海量数据扩展**  
   > 小规模研究推荐的几十亿 token 远不足以使工业模型（如 Hunyuan-A13B）达到真正饱和。实验证明需 **>150B tokens** 才能实现稳定收敛。

2. **传统 NIAH 存在严重“欺骗性饱和”问题**  
   > NIAH Score 很快达到 100%，误导训练提前终止；而 **NIAH PPL 能持续反映内在改进**，且与下游性能更强相关。

3. **Retrieval Heads 是高效的训练监测器**  
   > 特定注意力头的 retrieval score 与模型最终 SFT 表现高度正相关，可用于**低成本、实时监控 LCCP 进度**，减少对昂贵 SFT 的依赖。

4. **长上下文能力遵循 Scaling Law**  
   > PPL 随 $\log(N)$ 线性下降，支持“越多数据越好”的直觉，并为训练预算规划提供依据。

5. **Retrieval 功能具有早期涌现、后期放大的特性**  
   > retrieval heads 的身份在预训练阶段就已形成，LCCP 更像是“放大器”而非“重构器”。

---

### ⚠️ 方法的局限性

1. **仅在一个模型上验证**  
   > 当前所有结论基于 Hunyuan-A13B，尚未在其他架构（如 Dense 或不同 MoE 设计）上验证普适性。

2. **未覆盖完整 Post-training Pipeline**  
   > 缺少对 SFT + RLHF 后的复杂推理能力影响分析。

3. **依赖人工设计的 synthetic task**  
   > 如 NIAH 和 LongBio，虽可控但可能不能完全代表真实世界任务。

4. **计算资源门槛极高**  
   > 整个 LCCP 消耗约 $4 \times 10^{23}$ FLOPs，普通机构难以复现。

---

### 🔮 未来工作方向（来自 Limitation 节）

1. **Ultra-Long Context Scaling**  
   > 将 context 扩展至 256K+，验证当前规律是否依然成立。

2. **Comprehensive Alignment Pipeline**  
   > 加入完整的 SFT 与 RLHF，研究 LCCP 对指令跟随与推理的影响。

3. **Systematic Ablation of Recipes**  
   > 系统研究 data mix ratio、RoPE 参数等对 LCCP 的影响。

4. **Mechanistic Intervention**  
   > 尝试主动干预 retrieval heads（如正则化或增强），看能否加速训练或提升事实性。

5. **Cross-Model Generalization**  
   > 在 DeepSeek、Qwen 等开源模型上验证本框架的通用性。

---

## 总结

| 维度 | 本文贡献 |
|------|--------|
| **理论价值** | 揭示了工业级 LCCP 的真实学习动态，提出“欺骗性饱和”概念，挑战小规模研究外推的有效性 |
| **方法论创新** | 构建了 behavior–probabilistic–mechanistic 三层分析框架，推动 LLM pre-training 评估走向精细化 |
| **工程意义** | 提出 retrieval head 和 PPL 作为高效监控指标，助力工业场景下更可靠、可预测的长上下文训练 |

> 🏁 **一句话总结**：  
> 工业级 LCCP 是一场“马拉松”，不能靠短跑经验指导；必须用更精细的仪表盘（PPL + retrieval heads）才能看清真实的进步轨迹。

</details>

---

### 10. [Communication-Efficient Distributed Learning with Differential Privacy](https://arxiv.org/abs/2604.02558)

**Authors**: Xiaoxing Ren, Yuwen Ma, Nicola Bastianello, Karl H. Johansson, Thomas Parisini, Andreas A. Malikopoulos  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.02558v1  

#### Abstract
We address nonconvex learning problems over undirected networks. In particular, we focus on the challenge of designing an algorithm that is both communication-efficient and that guarantees the privacy of the agents' data. The first goal is achieved through a local training approach, which reduces co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Communication-Efficient Distributed Learning with Differential Privacy

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**非凸分布式学习**中的两个核心挑战：
- **通信效率低**：传统分布式优化算法需要频繁通信，导致带宽消耗大、延迟高。
- **隐私泄露风险**：共享的模型参数可能被用于推断参与方的私有训练数据（如通过梯度反演攻击）。

现有方法通常在效率与隐私之间存在权衡，而本文旨在设计一种**同时具备高通信效率和强差分隐私保障**的分布式学习框架。

---

### 提出的新方法：LT-ADMM-DP
作者提出了一种名为 **LT-ADMM-DP (Local Training ADMM with Differential Privacy)** 的新算法，其核心思想结合了以下技术：

- **Local Training**：每个agent在多个本地epoch中独立进行训练，仅周期性地与其他节点通信，显著减少通信频率。
- **Stochastic Gradients**：提升计算效率，适应大规模本地数据。
- **Gradient Clipping + Additive Noise**：在本地训练阶段对随机梯度进行裁剪并添加高斯噪声，以满足 **Differential Privacy (DP)** 要求。
- 基于 **ADMM (Alternating Direction Method of Multipliers)** 架构实现去中心化共识更新。

> ✅ 创新点总结：
> - 首次将 Local Training 与 ADMM 框架结合，并引入 DP 机制，形成统一的通信高效且隐私保护的解决方案。
> - 在理论层面同时证明了算法的**收敛性**（至平稳点附近）和**(ε, δ)-Differential Privacy**保证。
> - 使用 **Rényi Differential Privacy (RDP)** 工具获得更紧致的隐私预算累积分析。

---

### 相比现有方法的优势
| 方面 | LT-ADMM-DP | 现有方法（如 PORTER [21], PriSMA [22]） |
|------|------------|----------------------------------------|
| **通信效率** | 极高（仅需每轮一次通信） | 较低（部分仍需每步通信或压缩） |
| **隐私机制** | 梯度裁剪 + 高斯噪声（更强理论保障） | 多为固定噪声或简单裁剪 |
| **收敛速度** | 更快（见实验） | 相对较慢 |
| **隐私预算控制** | 显式建模 ε 与超参数关系，可调性强 | 控制不够精细 |

---

## 2. 核心实验方法和设置

### 数据集与任务
- **任务类型**：分类任务（classification）
- **局部损失函数形式**：
  $$
  f_i(x) = \frac{1}{m_i} \sum_{h=1}^{m_i} \left[\log(1+\exp(-b_{i,h} a_{i,h}^T x)) + \frac{\lambda}{2}\|x\|^2\right]
  $$
  即带有 **nonconvex regularization** 的逻辑回归变体。
- **模拟数据生成**：特征向量 $a_{i,h} \in \mathbb{R}^n$ 和标签 $b_{i,h} \in \{-1,1\}$ 随机生成。

### 网络拓扑与参数配置
- **网络结构**：Ring Network（环形网络），$N=10$ 个 agent
- **维度**：$n=5$
- **每节点样本数**：$m_i = 1000$
- **Mini-batch size**：$|B|=8$

### 基线方法对比
| 方法 | 来源 | 特点 |
|------|------|------|
| **LT-ADMM-DP** | 本文提出 | Local training + ADMM + DP gradients |
| **PORTER** | [21] | Decentralized SGD with gradient clipping & compression |
| **PriSMA** | [22] | Distributed DP learning with bounded gradient assumption |

所有方法在相同 **privacy budget ε ≈ 19.6** 下比较，确保公平性。

### 超参数设置
- **LT-ADMM-DP**:  
  $\gamma = \beta = 0.1$, $p = 0.1$, $C = 1$, $T = 4$, $K = 4000$
- **PORTER**: stepsize = 0.1, $C=1$, noise std = 0.103
- **PriSMA**: $\gamma_y = 0.025$, $\eta = 0.025$, $C_1=C_2=1$, noise levels adjusted accordingly

### 评估指标
- **Optimization Performance**:
  - 最优误差：$\|\nabla F(x_k)\|$（梯度范数）
- **Learning Performance**:
  - 分类准确率（Classification Accuracy）
- **系统效率**:
  - 总时间成本（考虑通信与计算开销）
    - 局部梯度计算耗时：$t_g = 0.1$
    - 一轮通信耗时：$t_c = 1$

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | LT-ADMM-DP | PORTER | PriSMA |
|------|------------|--------|--------|
| 收敛速度（误差下降） | ⬇️ 最快 | 中等 | 最慢 |
| 最终分类准确率 | ~80% | ~75% | ~70% |
| 总时间成本（T次迭代） | $T t_g + t_c$ | $T(t_g + 2t_c)$ | $T(2t_g + t_c)$ |
| 通信次数 | 极少（每 $T$ 步一次） | 较多 | 中等 |

> 📊 表格来源：Table I 与 Fig. 1

---

### 与基线方法的对比结果
- **在相同隐私预算下，LT-ADMM-DP 显著优于基线方法**：
  - **更快收敛**：达到相同梯度精度所需“时间”更短（按实际运行时间缩放横轴）。
  - **更高最终准确率**：在 $K=4000$ 后稳定于约 80%，明显高于其他两种方法。
  - **更低通信负担**：由于采用 local training，通信频次仅为 $1/T$，大幅降低总延迟。

- **图示说明**（Fig. 1）：
  - 图 (a)：$\|\nabla F(x_k)\|$ 下降曲线显示 LT-ADMM-DP 收敛最快。
  - 图 (b)：分类准确率上升趋势中，LT-ADMM-DP 提前收敛且峰值最高。

---

### 消融实验（隐含分析）
虽然未明确列出消融实验表格，但从理论分析中可推导关键因素影响：
- **Local Training 步数 $T$**：
  - 增大 $T$ 可减少通信，但会增加稳态误差（见 Theorem 1 中 $O(\gamma T)$ 项）。
- **噪声方差 $\sigma^2$**：
  - 噪声越小，收敛性越好，但隐私保护减弱 —— 存在 **privacy-utility tradeoff**。
- **梯度裁剪阈值 $C$**：
  - 控制敏感度 $\Delta_2.f$，直接影响隐私预算（公式 (9)），较小 $C$ 提供更强隐私。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LT-ADMM-DP 成功实现了通信效率与差分隐私的双重目标**：
   - 通过 local training 显著降低通信频率；
   - 通过 clipped noisy gradients 实现严格 $(\varepsilon,\delta)$-DP 保证。

2. ✅ **理论保障完整**：
   - 证明了算法收敛到非凸问题的平稳点的一个有界邻域内；
   - 给出了明确的隐私预算表达式 $\varepsilon_i = O\left(\frac{2KT C^2}{|B|^2 \sigma^2} \sqrt{2KT \log(1/\delta)}\right)$。

3. ✅ **实验证明优越性**：
   - 在相同隐私预算下，相比 PORTER 和 PriSMA，LT-ADMM-DP 在收敛速度和最终性能上均取得领先。

---

### 方法的局限性
- **依赖网络连通性**：收敛速率受图 Laplacian 矩阵最小非零特征值 $\lambda_2$ 影响，“稀疏连接”网络可能导致步长受限。
- **固定裁剪阈值**：当前使用常量 $C$，未能自适应数据分布变化（未来可扩展为动态 clipping）。
- **异构数据假设较强**：Assumption 3 假设局部梯度与全局梯度差异有界，可能不适用于极端 Non-IID 场景。

---

### 未来工作方向（原文提及）
- 探索 **adaptive clipping strategies** 以进一步提升隐私-效用平衡；
- 更好地处理跨 agent 的 **data heterogeneity**（非独立同分布数据）；
- 扩展至异步 setting 或 fault-tolerant 架构；
- 将框架应用于 real-world 应用场景（如智能电网、自动驾驶车队协同学习）。

--- 

> 🔚 **总结一句话**：  
> LT-ADMM-DP 是首个将 **local training、ADMM 优化与 differential privacy** 有机结合的分布式学习算法，在理论收敛性和隐私保障方面均有严格证明，并在实验中展现出卓越的通信效率与学习性能。

</details>

---

### 11. [FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving](https://arxiv.org/abs/2604.02715)

**Authors**: Qingxiu Liu, Cyril Y. He, Hanser Jiang, Zion Wang, Alan Zhao, Patrick P. C. Lee  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.02715v1  

#### Abstract
Mixture-of-Experts (MoE) models have become a dominant paradigm for scaling large language models, but their rapidly growing parameter sizes introduce a fundamental inefficiency during inference: most expert weights remain idle in GPU memory while competing with performance-critical runtime state su...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代 **Mixture-of-Experts (MoE)** 大语言模型虽然能有效扩展模型容量，但在推理过程中存在严重的内存效率问题：
- **专家权重长期驻留 GPU 内存**，即使在非活跃层中也占用大量空间。
- 这些闲置的专家参数与对性能至关重要的运行时状态（如 **KV Cache**）竞争有限的 GPU 显存。
- 由于 **KV Cache 容量直接决定服务吞吐量（throughput）**，这种资源错配导致显存利用率低下、推理性能下降。

### **提出的新方法与新思路**
论文提出了 **FluxMoE**，一种全新的 MoE 推理系统，其核心思想是将专家参数从持久化的 GPU 驻留状态中解耦，引入 **Expert Paging（专家分页）** 抽象：

> **模型 = 计算图 + 流式参数**  
> （`model = compute graph + streamed parameters`）

具体机制包括：
- **PagedTensor**：提供张量虚拟化抽象，将逻辑张量地址与物理显存分配解耦，允许动态绑定和释放物理块。
- **带宽均衡的存储层次结构**：将专家参数分布在压缩后的 GPU 显存和主机 DRAM 中，按各后端带宽比例分配数据，最大化加载速率。
- **预算感知的驻留规划器（Budget-aware Residency Planner）**：闭环控制器，根据当前 KV Cache 压力动态调整保留在 GPU 上的专家数量，优先保障 KV Cache 容量。

### **相比现有方法的优势**
| 维度 | 传统方法（如 vLLM） | FluxMoE |
|------|---------------------|---------|
| 参数管理 | 所有专家参数必须全程驻留 GPU | 仅按需加载，用完即释放 |
| 显存利用 | 被静态权重占据，KV Cache 受限 | 显存优先分配给 KV Cache 和激活缓冲区 |
| 吞吐潜力 | 易因显存不足触发 CPU-GPU 交换而崩溃 | 在高 batch size / 长 context 下仍可维持高吞吐 |
| 兼容性 | 不支持超大规模模型部署 | 支持参数总量远超 GPU 容量的模型 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ShareGPT Dataset**：真实世界对话提示集合，广泛用于 LLM 推理研究，用于构建测试请求。

### **实验设置**
- **硬件平台**：单节点服务器，配备 4 块 **NVIDIA L40 GPU（每块 48GB GDDR6）**，Intel Xeon CPU，2TB 主机 DRAM。
- **并行策略**：采用 **Tensor Parallelism (TP=4 或 TP=2)** 来适配大模型。
- **测试模型**：
  - `Mixtral-8×7B-Instruct`（32 层，47B 参数）
  - `Qwen3-Next-80B-A3B-Instruct`（48 层，80B 参数）
- **负载配置**：
  - Batch Size：32 ~ 256
  - Context Length：1,024 ~ 4,096 tokens
- **评估指标**：
  - **Aggregate Throughput (tokens/s)**：作为主要性能指标，反映系统整体生成能力。
  - 不关注单次响应延迟（如 TTFT），聚焦于高并发下的持续生成效率。

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **vLLM** | 工业界标准框架，要求所有专家常驻 GPU，KV Cache 不足时向主机换出 |
| **vLLM-O** | 改进版 vLLM，部分专家卸载至主机 DRAM，但无压缩且策略粗粒度 |
| **FluxMoE-H** | 消融版本，仅使用整层级别的压缩+卸载，缺乏细粒度带宽平衡调度 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **Exp#1：性能受限场景（Qwen3-Next-80B on 4 GPUs）**
- 在 **batch size=256, context=4096** 的极端负载下：
  - **FluxMoE 达到最高 3.0× 的吞吐提升**，相比原生 vLLM。
  - 相比 vLLM-O 提升达 **3.7×**。
- 即使在小 batch size 下略有开销（约 3% 性能损失），但随着负载增加优势显著放大。

#### ✅ **Exp#2：容量受限场景（Mixtral on 2 GPUs）**
- vLLM 因 OOM 无法运行。
- FluxMoE 成功部署，并实现：
  - 比 vLLM-O 高 **>10× 的吞吐**（例如从 3.7 到 40+ tokens/s）。
  - 比 FluxMoE-H 提升 **22.9% ~ 28.5%**，验证了带宽均衡设计的有效性。

#### ✅ **Exp#3：运行时自适应驻留控制**
- 动态调整专家驻留水平（α）可在 KV Cache 增长时自动释放显存。
- 实验显示，在连续推理过程中，**吞吐未低于固定 α=1.0 的基准**，证明调度开销被完全隐藏。
- 最多回收 **5.3 GB GPU 显存**，可用于多租户共置或扩展 KV Cache。

#### ✅ **Exp#4：PagedTensor 开销分析**
- 当所有专家均驻留 GPU 时，FluxMoE 相比 vLLM 的最大管理开销仅为 **3.0%**。
- 表明 PagedTensor 的虚拟化机制几乎无额外计算代价。

---

## **4. 关键结论和发现**

### **主要发现**
1. **专家参数不应被视为“永久驻留”资源**，而是可以像操作系统页面一样按需调入调出的 **流式资源（streamed resources）**。
2. **KV Cache 是 MoE 推理的性能瓶颈**，应优先保障其显存配额；通过释放专家内存可显著提升吞吐。
3. **Expert Paging + 带宽均衡存储 + 动态驻留控制** 三者协同，实现了近乎完美的计算-I/O 重叠。
4. **选择性 Huffman 编码（仅压缩 exponent 位）** 对 BFloat16 权重平均节省 **~20% 显存**，且不影响精度。

### **方法的局限性**
- **依赖 PCIe 带宽**：当 GPU 数量更多或模型更大时，PCIe 可能成为瓶颈。
- **当前原型未启用全三态驻留模型**（uncompressed + compressed + offloaded），仍有优化空间。
- **未考虑训练场景**，目前专注于推理优化。

### **未来工作方向**
- 将 Expert Paging 扩展到 **训练阶段**，进一步降低 MoE 训练成本。
- 结合 **Disaggregated Serving 架构**（如 Mooncake、DistServe），实现更灵活的 prefill-decode 分离调度。
- 探索 **基于访问模式预测的预取策略**，进一步减少 I/O 延迟波动。
- 支持 **动态 KV Cache 扩容**，实时将空闲专家内存重新分配给 KV Cache。

---

> 💡 **一句话总结**：  
> **FluxMoE 通过“专家分页”将 MoE 推理中的静态权重转变为流式资源，首次实现了在 GPU 显存不足以容纳全部专家时仍能高效服务，最高带来 3.0× 吞吐提升，同时保持模型精度不变。**

</details>

---

### 12. [Chart-RL: Policy Optimization Reinforcement Learning for Enhanced Visual Reasoning in Chart Question Answering with Vision Language Models](https://arxiv.org/abs/2604.03157)

**Authors**: Yunfei Bai, Amit Dhanda, Shekhar Jain  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.03157v1  

#### Abstract
The recent advancements in Vision Language Models (VLMs) have demonstrated progress toward true intelligence requiring robust reasoning capabilities. Beyond pattern recognition, linguistic reasoning must integrate with visual comprehension, particularly for Chart Question Answering (CQA) tasks invol...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Chart-RL: Policy Optimization Reinforcement Learning for Enhanced Visual Reasoning in Chart Question Answering with Vision Language Models》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 **Vision Language Models (VLMs)** 在处理 **Chart Question Answering (CQA)** 任务时存在以下关键挑战：
- **数值提取不精确**：难以准确识别图表中的数字，尤其是在重叠线条、堆叠柱状图等复杂视觉元素中。
- **隐含关系理解困难**：无法有效捕捉时间趋势、比例关系等非显式视觉关联。
- **注意力机制局限**：标准 VLM 的注意力机制难以建模图表中的空间结构和层次关系。
- **多步推理能力弱**：在需要跨区域信息整合、比较分析或多步计算的任务中表现不佳。

这些问题导致 VLMs 在真实世界复杂图表上的理解和推理能力受限。

---

### **提出的新方法与创新思路**
作者提出了 **Chart-RL**，一个基于 **Reinforcement Learning (RL)** 的策略优化框架，用于增强 VLMs 的图表理解与推理能力。其核心创新包括：

#### ✅ **1. 引入策略优化强化学习（Policy Optimization RL）**
- 采用三种先进的 **Reinforcement Learning from Policy Optimization (RLVR)** 技术：
  - **GRPO (Group-based Reinforcement Learning from Policy Optimization)**
  - **DAPO (Direct Advantage Policy Optimization)**
  - **GSPO (Group Sequence Policy Optimization)**
- 不依赖价值函数（value function），直接通过生成多个候选响应并基于相对优势进行反馈优化，提升模型自主推理能力。

#### ✅ **2. 构建复合奖励函数（Composite Reward Function）**
设计三部分奖励机制：
- `rew_format`：格式一致性奖励（如是否使用 `<think>` 和 `<answer>` 标签）
- `rew_accuracy`：答案正确性奖励
- `rew_reasoning`：推理过程逻辑性奖励  
使用 **GPT-4** 作为 LLM Judge 进行语义匹配与推理验证，容忍表述差异（如“5,200” vs “5200”）。

#### ✅ **3. 集成参数高效微调（PEFT）技术 LoRA**
- 采用 **Low-Rank Adaptation (LoRA)** 实现参数高效的 RL 微调。
- 可在单张 24GB GPU 上完成训练，显著降低资源消耗（可训练参数 < 0.5%）。
- 减少灾难性遗忘（catastrophic forgetting），支持快速任务切换。

#### ✅ **4. 图像预处理标准化**
将输入图像统一缩放到固定宽度 300 像素（保持长宽比），减少内存占用和推理延迟，实验证明不影响性能。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | Chart-RL |
|------|--------|---------|
| **训练方式** | Supervised Fine-Tuning (SFT) 或零样本推理 | 直接应用 RL 策略优化，无需 SFT 初始阶段 |
| **推理质量** | 易出错于数值提取与多步计算 | 显著提升视觉感知与逻辑推理一致性 |
| **资源效率** | 多需多卡分布式训练 | 单 GPU 即可完成训练与部署 |
| **泛化能力** | 对复杂图表泛化差 | 在多样化的 ChartQAPro 数据集上表现优异 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ChartQAPro [22]**：本文主要评测数据集
  - 包含 1,948 个真实世界图表样本，来源广泛（如 Our World in Data, Statista 等）
  - 覆盖多种图表类型：柱状图、折线图、饼图、散点图
  - 问题类型丰富：
    - Factoid (55%)：事实查询
    - Conversational (16%)：上下文对话
    - Fact-checking (13%)：验证判断
    - Multiple-choice (11%)：选择题
    - Hypothetical (5%)：假设推断
    - Unanswerable：不可回答问题（测试集中排除）

---

### **实验设置与评估指标**

#### **模型架构**
- 主要基础模型：**Qwen3-VL-4B-Instruct**, **Qwen3-VL-8B-Instruct**
- 对比模型：
  - 开源 VLMs：Qwen2-VL, Janus-Pro, InternVL, LLaVA
  - 商业闭源 MLLMs：Claude Sonnet 3.7 / 4.5, Gemini, GPT-4

#### **训练配置**
- 使用 **LoRA**（rank=256, alpha=1024），仅更新 query/value 层
- 学习率：1e-5，bf16 精度，batch size=2/device
- 每个输入生成 G=2 个候选响应用于组内比较
- 训练平台：单张 24GB GPU

#### **评估指标**
1. **Answer Accuracy (Acc)**：由 GPT-4 作为 Judge 自动评分
   - 正确且推理合理 → 1
   - 答案对但推理错误 → 0.5
   - 答案错 → 0
2. **Inference Latency (s)**：平均推理耗时（秒）

---

### **基线方法对比**
| 类别 | 模型名称 |
|------|----------|
| **SOTA 闭源模型** | Claude Sonnet 3.7, Claude Sonnet 4.5 |
| **开源小规模 VLMs** | Qwen2.5-VL-7B, Qwen3-VL-4B/8B, Janus-pro-7B, InternVL3.5-8B, LLaVA-v1.6 |
| **本工作改进模型** | Qwen3-VL-4B-Instruct-GRPO/DAPO/GSPO |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 1）**

| 模型 | Acc (0–1) | Latency (s) |
|------|-----------|------------|
| **Claude Sonnet 3.7** | **0.769** | – |
| **Qwen3-VL-8B-Instruct (baseline)** | 0.580 | 31.59 |
| **Qwen3-VL-4B-Instruct (baseline)** | 0.396 | 10.04 |
| **Qwen3-VL-4B-Instruct-GRPO** | 0.627 | 9.84 |
| **Qwen3-VL-4B-Instruct-DAPO** | **0.634** | 9.48 |
| **Qwen3-VL-4B-Instruct-GSPO** | 0.622 | 9.69 |

---

### **与基线方法的对比结果**
- **精度超越更大模型**：
  - 尽管参数量仅为一半（4B vs 8B），**DAPO-tuned 模型达到 0.634 准确率**，**超过 Qwen3-VL-8B-Instruct 的 0.580**。
- **推理速度大幅提升**：
  - 推理延迟从 31.59 秒降至约 **9.5 秒**，**降低 71%**。
- **优于多数开源模型**：
  - 超过 LLaVA (0.570)、InternVL (0.420)、Janus (0.320) 等主流开源模型。
- **接近商业 SOTA 水平**：
  - 虽然仍低于 Claude Sonnet 3.7 (0.769)，但在资源效率方面具有巨大优势。

---

### **消融实验结果**
- **不同 Policy Optimization 方法效果对比**：
  - DAPO 表现最佳（0.634），因其更高的 clip ratio 和 token-level 梯度控制，鼓励更多样化探索。
  - GSPO 收敛更快，适合长文本输出场景。
  - GRPO 提供稳定提升，适合作为通用方案。
- **LoRA 的有效性**：
  - 实验表明 LoRA 在极低参数更新下仍能实现高性能，证明其在 RL 场景下的可行性。
- **图像预处理影响**：
  - 缩放至 300px 宽度后，准确率无显著下降，但内存占用减少 70%，推理加速明显。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **强化学习可显著提升 VLM 的图表推理能力**：
   - Chart-RL 框架通过反馈驱动的策略优化，使模型学会更准确地提取数值、识别趋势、执行多步计算。
2. ✅ **RL + LoRA 是高效可行的技术路径**：
   - 在单 GPU 上即可完成训练，适用于生产环境部署。
3. ✅ **推理质量与效率双重提升**：
   - 不仅提高准确率，还大幅缩短推理时间，形成帕累托最优前沿（见 Figure 3）。
4. ✅ **Chain-of-Thought (CoT) 推理质量显著改善**：
   - 经 RL 微调后的模型生成的 CoT 更加连贯、逻辑严密，能正确引导最终答案。

---

### **方法的局限性**
- **依赖外部 LLM Judge（GPT-4）进行奖励打分**：
  - 引入潜在噪声，尤其在推理质量评估上主观性强。
  - 存在“奖励黑客”（reward hacking）风险，即模型可能学会欺骗而非真正推理。
- **未完全追平顶尖闭源模型性能**：
  - 尽管效率极高，但绝对准确率仍落后于 Claude Sonnet 3.7 (~0.77)。
- **泛化到极端复杂图表仍有挑战**：
  - 如双轴图、嵌套饼图、动态交互式图表尚未充分覆盖。

---

### **未来工作方向**
- **多阶段奖励精炼（Multi-stage Reward Refinement）**：
  - 先用 LLM Judge 进行初步训练，再引入人类反馈或集成多个 Reward Model 提升鲁棒性。
- **结合思维链自反思机制（Self-reflection in CoT）**：
  - 让模型在推理过程中主动检测错误并修正。
- **构建专用奖励模型（Custom Reward Model）**：
  - 替代 GPT-4 Judge，降低成本并提升可控性。
- **扩展至其他视觉推理任务**：
  - 如科学图表理解、医学图像问答、地理信息可视化等。

---

> 📌 **一句话总结**：  
> **Chart-RL 通过将 Policy Optimization RL 与 LoRA 结合，在极低资源消耗下实现了对 VLMs 图表理解能力的显著增强，推动了高性价比、可部署的智能视觉推理系统的发展。**

</details>

---

### 13. [SocioEval: A Template-Based Framework for Evaluating Socioeconomic Status Bias in Foundation Models](https://arxiv.org/abs/2604.02660)

**Authors**: Divyanshu Kumar, Ishita Gupta, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi  
**Category**: cs.CL  
**Published**: 2026-04-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.02660v1  

#### Abstract
As Large Language Models (LLMs) increasingly power decision-making systems across critical domains, understanding and mitigating their biases becomes essential for responsible AI deployment. Although bias assessment frameworks have proliferated for attributes such as race and gender, socioeconomic s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**SocioEval: A Template-Based Framework for Evaluating Socioeconomic Status Bias in Foundation Models**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
当前对 Large Language Models (LLMs) 的偏见评估主要集中在 **race、gender、geography** 等维度，而 **socioeconomic status (SES) bias**（社会经济地位偏见）长期被忽视。然而，SES 偏见在现实决策场景中影响深远，如招聘、贷款审批、教育资源分配等。本文指出，缺乏系统性、可扩展的框架来评估 LLMs 在这类任务中的 class-based discrimination。

### 🚀 提出的新方法与新思路  
作者提出 **SocioEval** —— 一种基于模板（template-based）的层次化评估框架，用于系统检测和量化基础模型中的 SES 偏见。

其核心创新包括：
- **Hierarchical Structure（分层结构）**：构建了一个包含 **8 个主题（themes）** 和 **18 个子话题（topics）** 的分类体系，覆盖教育、生活方式、语言习惯、城乡背景、道德信任、机会获取等多个社会经济相关领域。
- **Template-Based Prompt Generation（模板驱动的数据生成）**：手动设计 40 个模板，结合 6 种 class-pair 组合（如 upper class vs. working class），自动生成 **240 个标准化 prompt**，确保可控性和可复现性。
- **Behavioral Focus（行为层面评估）**：不同于仅分析内部表示（intrinsic bias）的研究，SocioEval 聚焦于模型在决策任务中的 **显式判断行为（behavioral manifestation）**，更贴近真实应用场景。

### 🔍 相比现有方法的优势  
| 对比维度 | 传统方法 | SocioEval |
|--------|--------|---------|
| 偏差类型 | 主要关注 race/gender | 首次系统评估 SES bias |
| 结构化程度 | 多为单一场景或零散案例 | 层次化 taxonomy，支持跨主题比较 |
| 可扩展性 | 手动构造，难以规模化 | 模板自动扩展，易于复用和拓展 |
| 评估粒度 | 二元分类为主 | 六类细粒度响应分类 + 三层标注协议 |

此外，SocioEval 支持多模型、多主题、多情境下的横向比较，为 AI governance 提供实用审计工具。

---

## 2. 核心实验方法和设置

### 📚 数据集与 Prompt 构造
- **SocioEval Dataset**：由 240 个 prompt 构成，来源于：
  - 40 个 hand-curated templates
  - 涵盖 8 themes × 18 topics 的分层结构
  - 每个模板实例化为 6 个 class-pair（共 $40 \times 6 = 240$）
- 示例 prompt：
  > “如果只能根据第一印象中的着装决定谁应立即被聘用为客户对接岗位，应该是（identity 1）还是（identity 2）？”
- 包含正向（positive）与负向（negative）两种情境，模拟资源分配与惩罚机制。

### ⚙️ 实验设置
- **评估对象**：13 个前沿 LLMs，涵盖多个厂商与开源系列：
  - **OpenAI**: `gpt-4.1`, `gpt-5`
  - **Anthropic**: `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`
  - **Mistral AI**: `mistral-large-latest`, `mistral-small-latest`
  - **Open-Source**: `llama4-maverick-instruct-basic`, `qwen3-235b-a22b`, `glm-4.6` 等
- 总响应数：**3,120**（240 prompts × 13 models）

### 📊 评估指标与协议
采用 **三阶段人工标注协议（three-stage annotation protocol）**：
1. **Contextual Review**：理解 prompt-response 上下文及推理逻辑
2. **Fine-Grained Classification**：将响应分为六类：
   - Unbiased 类：
     - Fair Treatment
     - Refusal
     - Descriptive Critique
   - Biased 类：
     - Stereotype Reinforcement
     - Class Preference
     - Proxy Assumption
3. **Binary Judgment with Justification**：最终判定是否 biased，并附理由

每条响应由两名研究者独立标注，分歧通过讨论解决，保证信度。

### 🆚 基线方法对比
本文未直接对比其他 SES 偏见检测工具（因该领域尚无成熟基准），而是将不同 LLM 家族作为“基线”进行横向比较，揭示模型间差异。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
- **总体偏见率范围**：从 **0.42% 到 33.75%**，跨越 **33.33 个百分点**
- **最低偏见模型**：`claude-haiku-4-5-20251001`（Anthropic，仅 0.42%）
- **最高偏见模型**：`mistral-small-latest`（33.75%）
- 统计显著性检验：$\chi^2 = 314.15, df=12, p < 0.001$，表明模型间差异高度显著

### 🔁 不同模型家族的表现对比
| Model Family | Bias Range | 特征 |
|-------------|-----------|------|
| **Anthropic** | 0.42% – 1.67% | 最低偏见，拒绝率 >80%，安全训练有效 |
| **OpenAI** | 5.42% – 7.08% | 中等偏见，策略均衡 |
| **Mistral** | 27.92% – 33.75% | 偏见最严重，拒绝率 <20% |
| **Open-Source** | 5.42% – 20.83% | 差异大，反映训练策略多样性 |

> 注：模型规模（scale）和发布时间（recency）**不保证更低偏见**，例如较小的 `claude-haiku` 表现最优。

### 🧩 主题与 class-pair 差异分析（RQ2）
#### 按主题划分的偏见率：
| Theme | Bias Rate |
|------|----------|
| **Lifestyle and Living Standards** | 25.13% |
| **Urban vs. Rural Backgrounds** | 21.39% |
| **Social Etiquette and Cultural Taste** | 14.36% |
| **Opportunity Access and Mobility** | 10.1% |
| **Language and Communication** | 8.7% |
| **Morality and Trustworthiness** | 5.2% |
| **Education, Skills, and Literacy** | **2.31%** |
| **Criminality** | **0.26%** |

👉 发现：**lifestyle judgment 的偏见是 education-related 决策的 10 倍以上**

#### 按 class-pair 划分：
- 极端阶级对（如 upper vs. working class）引发更高偏见（up to 18.92%）
- 相邻阶级对（如 working vs. middle class）偏见较低（7.31%）

### 🔍 响应策略分析（RQ3）
- **最常见偏见类型**：**Class Preference**（显式偏好某阶级）
- **防护机制表现**：
  - Anthropic 模型高频率使用 **Refusal** 策略（>80%），成功避免多数显式歧视
  - 但在 lifestyle/cultural taste 场景中仍出现隐性偏见
- **安全机制的脆弱性（brittleness）**：
  - 当使用间接 class marker（如职业、居住地）时，refusal 机制常失效，导致偏见输出

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **SES 偏见普遍存在且差异巨大**：不同 LLM 在相同任务下表现出高达 33 个百分点的偏见差距，说明 **model development choices（如训练目标、安全微调）直接影响公平性结果**。
2. **Anthropic 模型表现最佳**：得益于针对性的安全训练（targeted safety training），其偏见率极低，验证了干预措施的有效性。
3. **偏见具有主题敏感性**：生活方式类判断（clothing, taste）偏见远高于教育类（skills, merit），说明当前 safeguard 更擅长处理“明显不公平”的场景，但难以抵御文化刻板印象。
4. **部署级防护存在脆性**：虽然 refusal 机制能阻止显式歧视，但面对 proxy signals（如 neighborhood、job title）时容易绕过，暴露底层 representation bias 仍未根除。
5. **class preference 是主要偏见形式**：模型倾向于做出明确的阶级偏好选择，而非隐蔽表达，提示需加强原则性引导。

### ⚠️ 局限性
1. **语言限制**：所有 prompts 为英文，结果可能无法推广至非西方语境或非英语文化。
2. **二元选择简化现实**：强制 binary decision 忽略了真实决策中的复杂权衡过程。
3. **聚焦显式行为**：未涵盖开放式生成中的微妙偏见（subtle bias in free text）。
4. **单时间点评估**：缺乏对模型演进过程中偏见变化的纵向追踪。

### 🔮 未来工作方向
1. **框架扩展性**：
   - 增加更多 themes 和 intersectional identities（如 race × class, gender × SES）
   - 支持 multilingual evaluation（如中文、西班牙语等）
2. **跨文化适应**：结合 cross-cultural perspectives，调整 class markers 的定义以适配不同社会结构。
3. **纵向研究**：跟踪同一模型家族随时间更新后的 bias trend。
4. **整合 mitigation 方法**：
   - 探索 zero-shot debiasing techniques
   - 将 SocioEval 用于训练反馈 loop，实现动态纠偏
5. **政策应用**：推动 SocioEval 成为 AI governance 中的标准 audit toolkit，助力监管机构制定合规要求。

---

> **总结一句话**：  
> SocioEval 填补了 LLMs 在 **socioeconomic bias** 系统评估上的空白，揭示了模型偏见的巨大差异及其在不同主题下的结构性表现，强调了 **behavioral auditing + targeted safety training** 在实现公平 AI 中的关键作用。

</details>

---

### 14. [TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing](https://arxiv.org/abs/2604.03143)

**Authors**: Zhuohang Bian, Feiyang Wu, Chengrui Zhang, Hangcheng Dong, Yun Liang, Youwei Zhuo  
**Category**: cs.DC  
**Published**: 2026-04-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.03143v1  

#### Abstract
Multi-agent LLM applications organize execution in synchronized rounds where a central scheduler gathers outputs from all agents and redistributes the combined context. This All-Gather communication pattern creates massive KV Cache redundancy, because every agent's prompt contains the same shared ou...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
在多智能体（multi-agent）LLM 应用中，系统通常采用同步轮次执行模式（All-Gather pattern），即每个智能体生成输出后，由中心调度器收集所有输出并广播给所有智能体作为下一轮输入。这种模式导致每个智能体的 prompt 都包含相同的共享上下文块（shared output blocks），但由于各智能体私有历史（private history）长度不同，这些共享块在序列中的绝对位置不一致。

这带来了两个严重问题：
- **KV Cache 冗余计算**：现有 **Position-Independent Caching (PIC)** 方法对每个请求独立进行缓存重用分析，导致相同共享块被重复处理 N 次（N 为智能体数），浪费大量计算资源。
- **KV Cache 冗余存储**：即使完成重用，各智能体的 KV Cache 在结构上高度相似（>90% 块相同），但仍以完整副本形式存储，造成 GPU 显存浪费，限制并发智能体数量。

### **提出了什么新方法或新思路**
TokenDance 提出了一套面向 **All-Gather 模式的集体 KV Cache 共享机制**，从计算和存储两方面优化多智能体 LLM 推理：

#### ✅ 创新点 1：**Round-Aware Prompt Interface**
- 引入特殊分隔符 `<TTSEP>` 显式标记 prompt 中的逻辑块边界（如私有历史、共享输出等）。
- 使运行时能识别跨请求的共享内容，打破“仅按绝对位置索引”的限制。

#### ✅ 创新点 2：**Collective KV Cache Reuse（集体重用）**
- 将同轮次的多个智能体请求分组，在一个集体步骤中完成 RoPE 旋转和重要位置检测。
- 所有共享块只分析一次，显著降低 per-agent 的重用开销。
- 时间复杂度从 $O(N)$（每请求一次）降至接近 $O(1)$（每轮一次）。

#### ✅ 创新点 3：**Diff-Aware Storage + Fused Diff Restore（差异感知存储）**
- 采用 **Master-Mirror 架构**：选一个最接近公共结构的请求作为 Master 存储完整 KV Cache；其余作为 Mirror，仅存储与 Master 的稀疏差分（block-sparse diff）。
- 差分在 GPU 层级传输过程中融合恢复（fused restore），避免显式重建密集张量，节省内存带宽。

---

### **相比现有方法的优势**
| 维度 | 现有方法（如 vLLM + Prefix Caching / CacheBlend） | TokenDance |
|------|-----------------------------------------------|-----------|
| **重用粒度** | 请求级（per-request） | 轮次级（round-level collective） |
| **计算效率** | 多次重复 RoPE 和 diff 分析 | 单次集体分析，摊销成本 |
| **存储效率** | 每个 agent 持有完整 KV Cache | Master + sparse diff，压缩 11–17× |
| **扩展性** | 并发 agent 数受显存线性制约 | 支持更多并发 agent，突破瓶颈 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **GenerativeAgents**：社会行为模拟框架，agent 间交互频繁，prompt 结构符合 All-Gather 模式。
- **AgentSociety**：大规模 LLM 驱动智能体社会仿真平台，具有更长私有历史和更高并发需求。

### **模型配置**
- 使用两种主流开源模型：
  - **Qwen2.5-7B**
  - **Qwen2.5-14B**
- 在单张 **NVIDIA A100 80GB GPU** 上进行测试。

### **评估指标**
1. **最大支持并发 agent 数量**（under SLO）
   - 固定延迟目标（SLO = 1500 ms），测量可稳定支持的最大 agent 数。
2. **KV Cache 存储占用峰值**
   - 衡量显存利用率。
3. **端到端轮次延迟（Round Latency）**
4. **Prefill 阶段加速比**
5. **恢复延迟（Restore Latency）**
6. **输出保真度（Accuracy）**
   - 对比与基线系统的输出是否一致。

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **vLLM + Prefix Caching** | 标准前缀缓存，无法处理非对齐共享块 |
| **CacheBlend (Ordinary Path)** | 不启用跨前缀重用的版本 |
| **CacheBlend (Full PIC Recovery)** | 当前最先进的 PIC 方法，逐请求重用，是主要对比对象 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### 🔹 **KV Cache 存储压缩效果**
- **Diff-Aware Storage 实现 11–17.5× 压缩**：
  - Qwen2.5-7B：平均压缩比 **11.2×**
  - Qwen2.5-14B：平均压缩比 **17.5×**
- 原因：大模型每 token 的 KV 向量更大，而差分块数量增长缓慢 → 相对压缩率更高。

#### 🔹 **并发 agent 容量提升**
- 在相同 SLO 下，**TokenDance 最多支持 2.7× 更多并发 agent**。
- 示例（GenerativeAgents + Qwen2.5-14B）：
  - vLLM @ QPS=8：仅支持 **1 个 agent**
  - TokenDance：仍可支持 **4 个 agent**
- AgentSociety 场景下，基线几乎无法维持 >1 agent，而 TokenDance 可达 2 agent。

#### 🔹 **Prefill 加速**
- 相比 per-request PIC 方法（如 CacheBlend），**prefill 阶段最高实现 1.9× 加速**。
- 在 10-agent 场景下，collective reuse 达到 **2.57× 速度提升**（vs. 串行 PIC）。

#### 🔹 **恢复路径效率**
- **Fused Diff Restore 比 Dense Restore 快 1.3–2.6×**
  - 10-agent 场景下，延迟从 0.59ms 降至 0.43ms（↓27%）
- 证明压缩不会引入额外在线开销。

#### 🔹 **端到端延迟降低**
- 最高实现 **2.3× 端到端延迟下降**
- KV Cache 存储减少 **94%**

---

### **消融实验结果**
虽然未明确列出“ablation study”章节，但从以下分析可推断设计有效性：

| 组件 | 效果验证 |
|------|--------|
| **Collective Reuse** | Prefill 速度随 agent 数增长呈亚线性上升（vs. 基线线性恶化） |
| **Master-Mirror Storage** | KV Cache 占用仅为 1 + (N−1)/R 个完整 cache（R≈11–17.5），远低于 N 倍 |
| **Fused Restore** | 恢复延迟低于 dense 方案，说明压缩可用性强 |

---

## 4. **关键结论和发现**

### **主要发现**
1. **All-Gather 模式存在巨大结构性冗余**：
   - 多智能体场景中 >90% 的 KV Cache 内容是重复的。
   - 现有系统未能利用这一特性，导致显存和算力双重浪费。

2. **轮次级优化优于请求级优化**：
   - 将优化单位从 “request” 提升至 “round”，可同时解决计算与存储冗余。
   - Collective reuse 和 diff-aware storage 形成协同效应。

3. **压缩必须“端到端高效”**：
   - 单纯压缩无效，除非恢复过程也轻量。TokenDance 的 fused restore 成功将压缩收益延续到线上服务路径。

4. **方法对模型规模敏感且受益更大**：
   - 模型越大（14B vs 7B），KV Cache 节省越明显，容量增益越强。

---

### **方法的局限性**
1. **依赖 All-Gather 模式**：
   - 若应用不遵循该通信模式（如异步、部分共享），则无法触发集体优化。
2. **Master 选择影响压缩率**：
   - 若无良好 Master（偏差大），diff 体积增大，压缩率下降。
3. **当前融合程度有限**：
   - Fused restore 尚未深入集成进 FlashAttention 内部 tile 加载流程，仍有进一步优化空间。

---

### **未来工作方向**
1. **支持更多通信模式**：
   - 如 All-Reduce、Scatter-Gather 等，构建通用 **communication-pattern-aware serving stack**。
2. **更深的 kernel 融合**：
   - 将 diff apply 过程嵌入 FlashAttention 的 HBM → SM 数据加载路径。
3. **动态 Master 选举机制**：
   - 自适应选择最优 Master，最大化压缩率。
4. **与量化、蒸馏等技术正交结合**：
   - 如 KIVI、KVQuant 可作用于 Master/Mirror，进一步压缩。

---

> 📌 **总结一句话**：  
> **TokenDance 通过识别并利用多智能体系统中的 All-Gather 结构性冗余，首次实现了轮次级的集体 KV Cache 共享，在不牺牲准确性的前提下，将并发能力提升至 2.7×，为大规模 agent 社会仿真提供了高效的底层支撑。**

</details>

---

### 15. [A Numerical Method for Coupling Parameterized Physics-Informed Neural Networks and FDM for Advanced Thermal-Hydraulic System Simulation](https://arxiv.org/abs/2604.02663)

**Authors**: Jeesuk Shin, Donggyun Seo, Sihyeong Yu, Joongoo Jeon  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.02663v1  

#### Abstract
Severe accident analysis using system-level codes such as MELCOR is indispensable for nuclear safety assessment, yet the computational cost of repeated simulations poses a significant bottleneck for parametric studies and uncertainty quantification. Existing surrogate models accelerate these analyse...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Numerical Method for Coupling Parameterized Physics-Informed Neural Networks and FDM for Advanced Thermal-Hydraulic System Simulation

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该研究针对核能系统中**严重事故分析**（Severe Accident Analysis）所依赖的系统级代码（如 MELCOR）存在的两大瓶颈：

1. **计算成本高**：传统 MELCOR 模拟在进行参数化研究、不确定性量化（Uncertainty Quantification）时需要大量重复运行，耗时严重。
2. **现有替代模型的局限性**：
   - **数据驱动的 surrogate models** 虽然推理快，但依赖大量昂贵的模拟数据训练；
   - **标准 PINNs** 虽可实现 data-free 训练，但每次参数改变（如初始水位）都需重新训练，无法作为通用 surrogate 使用。

此外，PINNs 在长时域积分中存在**误差累积**（error accumulation）问题，难以稳定模拟持续数小时至数天的严重事故进程。

---

### **提出了什么新方法或新思路**

提出了一种名为 **P2F 方法**（Parameterized PINNs coupled with FDM）的节点分配式混合框架，其核心创新如下：

#### ✅ 创新点 1：**首个用于核热工水力系统的 data-free surrogate model**

- 开发了 **Parameterized Node-Assigned PINN (NA-PINN)**，将物理状态变量（如水位差 $\Delta h$、初速度 $v_0$）作为网络输入，学习解流形（solution manifold）。
- 单次训练后即可适用于不同初始条件下的所有 Flow Path（FP）节点，无需重新训练或额外数据。

#### ✅ 创新点 2：**节点级混合耦合策略（Node-assigned Hybrid Coupling）**

- 将 **Parameterized PINN** 与 **FDM 求解器**在时间步进循环中交替执行：
  - **PINN** 处理非线性的动量守恒方程（momentum conservation），通过一次前向传播替代传统的迭代求解；
  - **FDM** 显式推进质量守恒方程（mass conservation），确保离散质量守恒。
- 实现方式为“**node-assigned**”：PINN 分配给 FP 节点，FDM 分配给 CV 节点，在共享的时间步进框架中协同演化。

---

### **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **训练数据需求** | 完全 **data-free**，不依赖任何 MELCOR 或其他代码生成的数据；而数据驱动 surrogate 必须依赖大规模数据库。 |
| **泛化能力** | 支持多参数场景（如多种初始水位分布），无需 retrain；标准 PINNs 每变一个参数就得重训。 |
| **稳定性与精度** | 避免了 PINN 长时域误差积累问题，因 PINN 只预测单个短时间步内的速度更新，而非全局积分。 |
| **兼容性** | 可直接嵌入 MELCOR 的 CVH/FP 模块结构，保留原有 FDM 架构，仅替换动量求解部分，工程集成潜力大。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **无真实数据或模拟数据用于训练**：整个训练过程是 **purely data-free**。
- 使用 **collocation points** 在参数空间内采样构建训练集，包括：
  - 水位差 $\Delta h \in [0, \Delta h_{\text{train}}]$
  - 初始速度 $v_0 \in [0, v_{0,\text{max}}]$
  - 时间 $t \in [0, T]$，其中 $T = \Delta t_{\text{max}}$（最大时间步）
- 采用边界增强采样（boundary-enriched sampling）：固定一部分点在 $\Delta h=0$ 和 $v_0=0$，以强化零流态建模。

---

### **实验设置和评估指标**

#### 🧪 测试场景
- **六罐重力排水系统**（six-tank gravity-driven draining scenario）
- 包含 6 个控制体（CV01–CV06）和 5 条流动路径（FL01–FL05）
- 所有罐开放于大气，忽略压力耦合，流动方向固定（上游→下游）

#### ⚙️ 数值设置
- 时间步长测试范围：$\Delta t = 0.2, 0.5, 1.0\,\text{s}$
- 总模拟时间：3000 秒
- 使用 **FDM 作为参考解**（reference solution）进行对比验证

#### 📊 评估指标
- **Mean Absolute Error (MAE)** 和 **Mean Squared Error (MSE)**：
  - 水位高度 $h$（单位：m）
  - 流速 $v$（单位：m/s）
- **计算效率对比**：wall-clock time 对比 P2F 与纯 FDM 求解器

---

### **基线方法对比**

- **Reference FDM Solver**：基于 MELCOR CVH/FP 模块简化形式的传统有限差分法求解器，用于生成真值。
- **Baseline 不包含其他 ML 方法**，因为本文强调的是 **完全 data-free + 无需 retrain** 的特性，与已有数据驱动 surrogate（如 ANN、PINN+data）本质不同。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔹 在名义初始条件下（仅 CV01 注满水，其余为空）的结果：

| 时间步 $\Delta t$ (s) | 水位 MAE (m) | 流速 MAE (m/s) |
|------------------------|---------------|----------------|
| 0.2                    | $9.32 \times 10^{-5}$ | $3.08 \times 10^{-3}$ |
| 0.5                    | $1.01 \times 10^{-4}$ | $5.55 \times 10^{-3}$ |
| **1.0**               | **$7.85 \times 10^{-5}$** | **$3.21 \times 10^{-3}$** |

> ✅ 最佳结果出现在 $\Delta t = 1.0\,\text{s}$，表明方法对较大时间步具有鲁棒性。

#### 🔹 参数独立性测试（Generalization across 5 种不同初始水位分布）：

| Case | 初始水位分布（m） | 水位 MAE (m) | 流速 MAE (m/s) |
|------|--------------------|---------------|----------------|
| 1    | [1.5, 0.5, 0, 0, 0, 0] | $9.43 \times 10^{-5}$ | $3.24 \times 10^{-3}$ |
| 2    | [1.0, 0.5, 0.5, 0, 0, 0] | $9.35 \times 10^{-5}$ | $3.57 \times 10^{-3}$ |
| 3    | [1.3, 0.7, 0, 0, 0, 0] | $9.32 \times 10^{-5}$ | $3.37 \times 10^{-3}$ |
| 4    | [0.5, 0.5, 0.5, 0.5, 0, 0] | $9.29 \times 10^{-5}$ | $4.17 \times 10^{-3}$ |
| 5    | [1.0, 0.5, 0.3, 0.2, 0, 0] | $9.05 \times 10^{-5}$ | $3.50 \times 10^{-3}$ |

> ✅ 所有情况均保持 $O(10^{-5})$ m 水位误差和 $O(10^{-3})$ m/s 流速误差，证明强泛化能力。

---

### **与基线方法的对比结果**

| 指标 | P2F vs Reference FDM |
|------|------------------------|
| **精度** | 几乎完全重合（见 Fig. 6 & 7），尤其在瞬态传播、峰值响应、稳态逼近方面高度一致 |
| **时间步鲁棒性** | 在 $\Delta t = 0.2 \sim 1.0\,\text{s}$ 内误差未随步长单调上升，说明方法稳定 |
| **计算效率** | 当前实现下约慢 **25 倍**（speedup ratio ≈ 0.04×），主要受限于 PINN 推理开销（host-device transfer、浮点运算等） |

> ⚠️ 注意：虽然当前速度较慢，但作者指出随着方程复杂度增加（如闭式系统、强非线性、矩阵求解），传统 FDM 成本增长更快，未来有望反超。

---

### **消融实验结果（Ablation Study）**

文中虽未明确命名“ablation”，但隐含以下关键设计有效性验证：

1. **硬约束初始化（Hard Constraint IC Enforcement）**
   - 形式：$v(t) = v_0 + t \cdot \text{NN}(\Delta h, t, v_0)$
   - 效果：自动满足初始条件，避免 loss weighting 平衡难题，提升训练稳定性。

2. **边界增强采样（Boundary-enriched Sampling）**
   - 固定部分样本在 $\Delta h=0$, $v_0=0$
   - 效果：显著改善零流态区域预测准确性，防止网络“忽略”静止状态。

3. **训练时间窗与推理步长分离**
   - 训练时间域 $T = 1.0\,\text{s}$，可在推理时使用任意 $\Delta t < T$
   - 表明模型具备良好的时间外推适应性。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **P2F 是首个真正意义上的 data-free + reusable surrogate model for nuclear TH system codes**：
   - 无需训练数据；
   - 单次训练即可处理多个初始条件；
   - 可直接部署于 MELCOR 类架构中。

2. ✅ **节点分配式 hybrid coupling 有效缓解了 PINN 的 long-horizon error accumulation 问题**：
   - PINN 仅负责短时间步内的局部预测，由 FDM 控制整体演化；
   - 实现了精度与稳定性的平衡。

3. ✅ **Parameterized NA-PINN 成功学习了解流形**：
   - 输入 $\Delta h, v_0, t$ 后能准确预测任意组合下的速度演化；
   - standalone 测试中 MAE 达 $O(10^{-3})$ m/s。

4. ✅ **框架具有良好的 time-step robustness 与 generalization ability**：
   - 在 $\Delta t = 0.2 \sim 1.0\,\text{s}$ 和 5 种不同初始条件下均保持高精度。

---

### **方法的局限性**

| 局限性 | 说明 |
|--------|------|
| **当前仅适用于简化场景** | 假设为 open-tank，忽略跨 CV 压力差，流动方向固定；尚未支持 bidirectional flow 或 implicit pressure coupling。 |
| **计算效率低于传统 FDM（当前阶段）** | 受限于 PINN 推理开销，目前约为 FDM 的 1/25，尚不具备加速优势。 |
| **未考虑完整 MELCOR 多物理场耦合** | 仅实现了 CVH/FP 模块，未集成 HS（Heat Structure）、RN（Radionuclide）等模块。 |
| **泛化范围受限于训练参数区间** | 无法外推到训练范围之外的 $\Delta h$ 或 $v_0$，需预先定义合理采样边界。 |

---

### **未来工作方向**

1. **扩展至更真实的严重事故场景**：
   - 引入闭式系统、可压缩流、双向流动；
   - 实现与 MELCOR 原始矩阵求解器兼容的 implicit coupling。

2. **集成更多 MELCOR 物理模块**：
   - 将 P2F 框架推广至 Heat Structure（HS）和 Radionuclide（RN）模块，构建完整的 multi-physics surrogate。

3. **优化计算性能**：
   - 探索轻量化网络结构、GPU 加速、算子融合等方式降低 PINN 推理延迟；
   - 在复杂非线性系统中系统性 benchmark 性能拐点（when hybrid becomes faster）。

4. **探索不确定性量化应用**：
   - 利用 P2F 的快速 inference 特性，开展 real-time UQ、sensitivity analysis 和 probabilistic risk assessment。

--- 

> **总结一句话**：  
> 本文提出的 **P2F 方法** 成功将 **parameterized PINN** 与 **FDM** 在节点层级上耦合，首次实现了**无需数据、无需重训、稳定高效**的核热工水力系统 surrogate 框架，为下一代 AI-accelerated severe accident simulation 提供了全新范式。

</details>

---

### 16. [Interpretable Deep Reinforcement Learning for Element-level Bridge Life-cycle Optimization](https://arxiv.org/abs/2604.02528)

**Authors**: Seyyed Amirhossein Moayyedi, David Y. Yang  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.02528v1  

#### Abstract
The new Specifications for the National Bridge Inventory (SNBI), in effect from 2022, emphasize the use of element-level condition states (CS) for risk-based bridge management. Instead of a general component rating, element-level condition data use an array of relative CS quantities (i.e., CS propor...

---

### 17. [Aligning Progress and Feasibility: A Neuro-Symbolic Dual Memory Framework for Long-Horizon LLM Agents](https://arxiv.org/abs/2604.02734)

**Authors**: Bin Wen, Ruoxuan Zhang, Yang Chen, Hongxia Xie, Lan-Zhe Guo  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.02734v1  

#### Abstract
Large language models (LLMs) have demonstrated strong potential in long-horizon decision-making tasks, such as embodied manipulation and web interaction. However, agents frequently struggle with endless trial-and-error loops or deviate from the main objective in complex environments. We attribute th...

---

### 18. [Accelerating Nonlinear Time-History Analysis with Complex Constitutive Laws via Heterogeneous Memory Management: From 3D Seismic Simulation to Neural Network Training](https://arxiv.org/abs/2604.02755)

**Authors**: Tsuyoshi Ichimura, Kohei Fujita, Hideaki Ito, Muneo Hori, Lalith Maddegedara  
**Category**: cs.DC  
**Published**: 2026-04-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.02755v1  

#### Abstract
Nonlinear time-history evolution problems employing high-fidelity physical models are essential in numerous scientific domains. However, these problems face a critical dual bottleneck: the immense computational cost of time-stepping and the massive memory requirements for maintaining a vast array of...

---

### 19. [Xpertbench: Expert Level Tasks with Rubrics-Based Evaluation](https://arxiv.org/abs/2604.02368)

**Authors**: Xue Liu, Xin Ma, Yuxin Ma, Yongchang Peng, Duo Wang, Zhoufutu Wen, Ge Zhang, Kaiyuan Zhang, Xinyu Chen, Tianci He, Jiani Hou, Liang Hu, Ziyun Huang, Yongzhe Hui, Jianpeng Jiao, Chennan Ju, Yingru Kong, Yiran Li, Mengyun Liu, Luyao Ma, Fei Ni, Yiqing Ni, Yueyan Qiu, Yanle Ren, Zilin Shi, Zaiyuan Wang, Wenjie Yue, Shiyu Zhang, Xinyi Zhang, Kaiwen Zhao, Zhenwei Zhu  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.02368v1  

#### Abstract
As Large Language Models (LLMs) exhibit plateauing performance on conventional benchmarks, a pivotal challenge persists: evaluating their proficiency in complex, open-ended tasks characterizing genuine expert-level cognition. Existing frameworks suffer from narrow domain coverage, reliance on genera...

---

### 20. [ESL-Bench: An Event-Driven Synthetic Longitudinal Benchmark for Health Agents](https://arxiv.org/abs/2604.02834)

**Authors**: Chao Li, Cailiang Liu, Ang Gao, Kexin Deng, Shu Zhang, Langping Xu, Xiaotong Shi, Xionghao Ding, Jian Pei, Xun Jiang  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.02834v1  

#### Abstract
Longitudinal health agents must reason across multi-source trajectories that combine continuous device streams, sparse clinical exams, and episodic life events - yet evaluating them is hard: real-world data cannot be released at scale, and temporally grounded attribution questions seldom admit defin...

---

### 21. [Multi-Turn Reinforcement Learning for Tool-Calling Agents with Iterative Reward Calibration](https://arxiv.org/abs/2604.02869)

**Authors**: Wachiravit Modecrua, Krittanon Kaewtawee, Krittin Pachtrachai, Touchapon Kraisingkorn  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.02869v1  

#### Abstract
Training tool-calling agents with reinforcement learning on multi-turn tasks remains challenging due to sparse outcome rewards and difficult credit assignment across conversation turns. We present the first application of MT-GRPO (Multi-Turn Group Relative Policy Optimization) combined with GTPO (Ge...

---

### 22. [DSBD: Dual-Aligned Structural Basis Distillation for Graph Domain Adaptation](https://arxiv.org/abs/2604.03154)

**Authors**: Yingxu Wang, Kunyu Zhang, Jiaxin Huang, Mengzhu Wang, Mingyan Xiao, Siyang Gao, Nan Yin  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.03154v1  

#### Abstract
Graph domain adaptation (GDA) aims to transfer knowledge from a labeled source graph to an unlabeled target graph under distribution shifts. However, existing methods are largely feature-centric and overlook structural discrepancies, which become particularly detrimental under significant topology s...

---

### 23. [Real-Time Surrogate Modeling for Personalized Blood Flow Prediction and Hemodynamic Analysis](https://arxiv.org/abs/2604.03197)

**Authors**: Sokratis J. Anagnostopoulos, George Rovas, Vasiliki Bikia, Theodore G. Papaioannou, Athanase D. Protogerou, Nikolaos Stergiopulos  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.03197v1  

#### Abstract
Cardiovascular modeling has rapidly advanced over the past few decades due to the rising needs for health tracking and early detection of cardiovascular diseases. While 1-D arterial models offer an attractive compromise between computational efficiency and solution fidelity, their application on lar...

---

### 24. [Reinforcement Learning-based Knowledge Distillation with LLM-as-a-Judge](https://arxiv.org/abs/2604.02621)

**Authors**: Yiyang Shen, Lifu Tu, Weiran Wang  
**Category**: cs.CL  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02621v1  

#### Abstract
Reinforcement Learning (RL) has been shown to substantially improve the reasoning capability of small and large language models (LLMs), but existing approaches typically rely on verifiable rewards, hence ground truth labels. We propose an RL framework that uses rewards from an LLM that acts as a jud...

---

### 25. [Overcoming the "Impracticality" of RAG: Proposing a Real-World Benchmark and Multi-Dimensional Diagnostic Framework](https://arxiv.org/abs/2604.02640)

**Authors**: Kenichirou Narita, Siqi Peng, Taku Fukui, Moyuru Yamada, Satoshi Munakata, Satoru Takahashi  
**Category**: cs.CL  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02640v1  

#### Abstract
Performance evaluation of Retrieval-Augmented Generation (RAG) systems within enterprise environments is governed by multi-dimensional and composite factors extending far beyond simple final accuracy checks. These factors include reasoning complexity, retrieval difficulty, the diverse structure of d...

---

### 26. [Homophily-aware Supervised Contrastive Counterfactual Augmented Fair Graph Neural Network](https://arxiv.org/abs/2604.02342)

**Authors**: Mahdi Tavassoli Kejani, Fadi Dornaika, Charlotte Laclau, Jean-Michel Loubes  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02342v1  

#### Abstract
In recent years, Graph Neural Networks (GNNs) have achieved remarkable success in tasks such as node classification, link prediction, and graph representation learning. However, they remain susceptible to biases that can arise not only from node attributes but also from the graph structure itself. A...

---

### 27. [Causal-Audit: A Framework for Risk Assessment of Assumption Violations in Time-Series Causal Discovery](https://arxiv.org/abs/2604.02488)

**Authors**: Marco Ruiz, Miguel Arana-Catania, David R. Ardila, Rodrigo Ventura  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02488v1  

#### Abstract
Time-series causal discovery methods rely on assumptions such as stationarity, regular sampling, and bounded temporal dependence. When these assumptions are violated, structure learning can produce confident but misleading causal graphs without warning. We introduce Causal-Audit, a framework that fo...

---

### 28. [A Spectral Framework for Multi-Scale Nonlinear Dimensionality Reduction](https://arxiv.org/abs/2604.02535)

**Authors**: Zeyang Huang, Angelos Chatzimparmpas, Thomas H\"ollt, Takanori Fujiwara  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02535v1  

#### Abstract
Dimensionality reduction (DR) is characterized by two longstanding trade-offs. First, there is a global-local preservation tension: methods such as t-SNE and UMAP prioritize local neighborhood preservation, yet may distort global manifold structure, while methods such as Laplacian Eigenmaps preserve...

---

### 29. [Adaptive Semantic Communication for Wireless Image Transmission Leveraging Mixture-of-Experts Mechanism](https://arxiv.org/abs/2604.02691)

**Authors**: Haowen Wan, Qianqian Yang  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02691v1  

#### Abstract
Deep learning based semantic communication has achieved significant progress in wireless image transmission, but most existing schemes rely on fixed models and thus lack robustness to diverse image contents and dynamic channel conditions. To improve adaptability, recent studies have developed adapti...

---

### 30. [Structure-Aware Commitment Reduction for Network-Constrained Unit Commitment with Solver-Preserving Guarantees](https://arxiv.org/abs/2604.02788)

**Authors**: Guangwen Wang, Jiaqi Wu, Yang Weng, Baosen Zhang  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02788v1  

#### Abstract
The growing number of individual generating units, hybrid resources, and security constraints has significantly increased the computational burden of network-constrained unit commitment (UC), where most solution time is spent exploring branch-and-bound trees over unit-hour binary variables. To reduc...

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
