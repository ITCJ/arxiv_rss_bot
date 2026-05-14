# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-14 08:18:13 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](https://arxiv.org/abs/2605.13319)

**Authors**: Yunhe Han, Yunqi Gao, Bing Hu, Mahdi Boloursaz Mashhadi, Yitong Duan, Pei Xiao, Yanfeng Zhang  
**Category**: cs.DC  
**Published**: 2026-05-14  
**Score**: 17.5  
**Type**: new  
**ArXiv ID**: 2605.13319v1  

#### Abstract
Speculative decoding can significantly accelerate LLM inference, especially given that its cloud-edge collaborative deployment offers cloud workload offloading, offline robustness, and privacy enhancement. However, existing collaborative inference frameworks with speculative decoding are constrained...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **cloud-edge collaborative inference** 框架在结合 **speculative decoding** 时面临两大瓶颈：
1. **Sequential Execution Bottleneck**：传统的框架采用“先生成所有 draft tokens → 再通信 → 最后验证”的串行模式，导致计算与通信资源利用率低。
2. **Inflexible NAV Triggering**：非自回归验证（NAV）触发机制不灵活，如固定长度或单一置信度阈值，容易造成过早验证（premature verification）或过度推测（costly rollbacks）。

### 🚀 提出的新方法
作者提出 **PipeSD** —— 一种高效的云边协同流水线推理框架，其核心创新包括：

#### （1）Token-Batch Pipeline Scheduling Mechanism
- 将 draft token 的生成与传输进行重叠（overlap），通过动态规划（Dynamic Programming, DP）优化 token 分批策略。
- 数学建模为最小化总延迟的优化问题，并用 DP 高效求解最优分批边界。

#### （2）Dual-Threshold NAV Triggering Mechanism
- 引入双重阈值机制，综合考虑：
  - **Single-Token Confidence**：单个 token 的预测概率；
  - **Cumulative Sequence Confidence**：整个序列的累积置信度（各 token 概率乘积）。
- 当任一条件低于设定阈值 $ R_1 $ 或 $ R_2 $ 时触发 NAV，提升验证灵活性。

#### （3）Lightweight Bayesian Optimization (BO) Autotuner
- 设计轻量级 BO 自动调优器，动态调整双阈值 $ (R_1, R_2) $，适应不同任务复杂度和运行环境变化。

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **资源利用** | 通过 pipeline 调度显著提高带宽与计算资源利用率 |
| **延迟降低** | 减少空闲等待时间，隐藏通信开销 |
| **验证灵活性** | 双阈值机制避免误判，减少 rollback 开销 |
| **自适应能力** | BO autotuner 实现参数自动调节，无需人工干预 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Programming Task**: `HumanEval`  
  - Draft Model: `DeepSeek-Coder-1.3B`  
  - Target Model: `DeepSeek-Coder-6.7B`
- **Mathematical Reasoning Task**: `GSM8K`  
  - Draft Model: `TinyLlama-1.1B-Chat-v1.0`  
  - Target Model: `Llama-2-7B`

### ⚙️ 实验设置
- **测试平台**：真实城域网环境
  - **Edge Device**：Lenovo ThinkBook 16+（Intel Ultra 9 CPU）
  - **Cloud Server**：天翼云 A800 GPU 实例
- **网络配置**：
  - 上行/下行带宽：20 Mbps / 200 Mbps（静态场景）
  - 动态带宽波动范围：上行 [10,80] Mbps，下行 [150,280] Mbps（Scenario 4）
- **模拟设备**：
  - 场景2：模拟手机（2.5GHz CPU）
  - 场景3：模拟 IoT 设备（1.2GHz CPU）

### 🎯 评估指标
| 指标 | 含义 |
|------|------|
| **TPT (Time Per Token)** | 平均每接受一个 token 所需的时间（ms），衡量推理速度 |
| **ECS (Energy Consumption on Server)** | 每接受 100 个 token 的云端能耗（J），反映能效 |

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **Vanilla** | 固定长度 speculative decoding（N=6 编程 / N=4 数学） |
| **HSL** | 单 token 置信度低于阈值时触发 NAV |
| **EdgeLLM** | 基于累积序列置信度触发 NAV，支持边缘持续生成 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（来自 Table 1 & 2）

| 场景 | 数据集 | TPT (PipeSD) | 相对 Vanilla 加速 | ECS 降低 |
|------|--------|---------------|--------------------|-----------|
| 1 | HumanEval | 129 ms | **1.50×** | 25.3% |
| 1 | GSM8K | 145 ms | **1.33×** | 14.3% |
| 2–3 | HumanEval/GSM8K | ↓ 至 134–152 ms / 168–186 ms | **最高达 2.16×** | 最高 25.3% |
| 4（动态带宽） | HumanEval/GSM8K | 108 / 139 ms | **1.48× / 1.68×** | — |

> ✅ **总体表现**：PipeSD 在所有场景下均优于基线，实现 **1.16× ~ 2.16× 的加速**，并降低 **14.3% ~ 25.3% 的云端能耗**。

### 🔁 与基线方法对比结果
| 对比项 | PipeSD 表现 |
|-------|------------|
| vs **Vanilla** | 显著减少等待时间，尤其在低算力边缘设备中增益更大 |
| vs **HSL** | 更合理的验证时机，避免频繁小批量验证 |
| vs **EdgeLLM** | 更高的 acceptance rate 和更长的有效 draft 序列 |

> 💡 示例：在 Scenario 1 的 HumanEval 上，PipeSD 的平均 draft length 达到 4.96，acceptance rate 高达 96.16%，远超 HSL（3.18 / 91.48%）和 EdgeLLM（4.74 / 89.17%）。

### 🔍 消融实验结果（Ablation Study, Table 6）
| 方法变体 | TPT (ms) | 相对 Full PipeSD 速度下降 |
|---------|----------|--------------------------|
| **PipeSD w/o Pipeline** | 147 ms | ↓ 1.12× |
| **PipeSD + Fixed-length NAV** | 164 ms | ↓ 1.25× |
| **PipeSD + Token-level only** | 137 ms | ↓ 1.05× |
| **PipeSD + Sequence-level only** | 139 ms | ↓ 1.06× |
| **PipeSD (Full)** | **129 ms** | ✅ 最佳 |

> ✅ 结论：
> - **Pipeline 调度** 贡献约 1.12× 加速；
> - **Dual-threshold NAV** 比单一机制更优；
> - 二者结合带来最大收益。

此外，DP-based batching 策略相比 greedy/immediate-send/no-early-upload 等策略仍可获得 **1.02×~2.06× 的额外增益**（见 Appendix F），证明其必要性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Pipeline 是必要的**：将 token 生成与通信重叠可有效隐藏延迟，尤其在网络带宽受限时效果显著。
2. **双阈值机制优于单信号触发**：同时监控 token-level 与 sequence-level 置信度，能更准确判断何时该验证，避免误判。
3. **自动化调参可行且高效**：仅需约 **16 次采样**，BO autotuner 即可在几分钟内收敛至近似最优阈值组合。
4. **系统具备良好鲁棒性**：在动态带宽、多客户端并发等现实条件下依然保持高性能。

### ⚠️ 局限性
- **依赖稳定通信模型假设**：当前通信延迟建模基于线性假设，在极端网络抖动下可能失效。
- **未支持 tree-based speculative decoding**：虽然文中讨论了 tree-based 方法（如 SpecInfer），但由于其高带宽消耗，暂未集成进 PipeSD。
- **边缘能耗分析为理论估算**：由于难以精确测量通用 CPU 上的边缘能耗，相关分析基于增量功率模型推导。

### 🔮 未来工作方向
1. 支持更复杂的 speculative decoding 架构（如 **tree-based drafting**）；
2. 探索在 **异构硬件平台** 和 **多样化任务类型** 下的泛化能力；
3. 进一步优化 BO autotuner 的收敛速度与采样效率；
4. 扩展至 **multi-edge 多设备协作场景**，支持负载均衡与资源调度。

---

## 总结

PipeSD 是首个将 **pipeline 调度** 与 **双阈值自适应验证触发** 相结合的云边协同 speculative decoding 框架。它通过 **DP 优化分批策略** 和 **BO 自动调参机制**，实现了高达 **2.16× 的端到端加速** 与 **超过 25% 的能耗节省**，为 LLM 在资源受限边缘设备上的高效部署提供了实用解决方案。

</details>

---

### 2. [D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models](https://arxiv.org/abs/2605.13276)

**Authors**: Yucheng Guo, Yongjian Guo, Zhong Guan, Wen Huang, Haoran Sun, Haodong Yue, Xiaolong Xiang, Shuai Di, Zhen Sun, Luqiao Wang, Junwu Xiong, Yicheng Gong  
**Category**: cs.AI  
**Published**: 2026-05-14  
**Score**: 15.5  
**Type**: new  
**ArXiv ID**: 2605.13276v1  

#### Abstract
The rapid evolution of Embodied AI has enabled Vision-Language-Action (VLA) models to excel in multimodal perception and task execution. However, applying Reinforcement Learning (RL) to these massive models in large-scale distributed environments faces severe systemic bottlenecks, primarily due to t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前在大规模 **Vision-Language-Action (VLA)** 模型中应用 **Reinforcement Learning (RL)** 面临严重的系统瓶颈，主要源于以下矛盾：
- **高保真物理仿真**（如机器人环境模拟）需要高频、碎片化的计算资源占用；
- **深度学习训练** 则对 GPU 显存（VRAM）容量和通信带宽有极高要求。

这两类任务在传统框架中耦合严重，导致：
- 内存碎片化（memory fragmentation）
- 数据传输开销大（serialization overhead）
- GPU 利用率低（GPU bubbles/idle time）
- 整体吞吐量受限于最慢环节（如物理步进或同步延迟）

### 提出的新方法与创新思路
为解决上述问题，论文提出 **D-VLA** —— 一种高并发、低延迟的分布式异步 RL 框架，其核心创新包括：

#### ✅ **Plane Decoupling 架构设计**
- 将执行平面划分为两个独立通道：
  - **Data Plane（数据平面）**：处理高频的数据采样（rollout）、观测与动作交互；
  - **Control Plane（控制平面）**：负责低频的模型权重更新与分发。
- 物理隔离避免了 simulation 与 training 之间的资源争用。

#### ✅ **四线程异步“Swimlane”流水线**
构建完全重叠的并行执行流程：
1. **Sampling Thread**：环境采样生成轨迹数据；
2. **Inference Thread**：策略推理；
3. **Gradient Training Thread**：梯度计算；
4. **Parameter Distribution Thread**：参数广播。
通过轻量级信号量同步，实现计算与通信的全重叠，显著提升硬件利用率。

#### ✅ **Dual-Pool VRAM 管理模型 + Zero-Copy 数据交换**
- 显存划分为两个池：
  - **Model Computation Pool**：由 PyTorch 管理，用于权重与梯度；
  - **Environment Auxiliary Pool**：专供物理引擎临时对象（如接触点），防止内存碎片。
- 在共置部署下启用 **zero-copy shared memory**，直接共享 observation 数据，减少序列化开销。

#### ✅ **拓扑感知复制（Topology-Aware Replication）与通信优化**
- 在节点内构建闭环的 sampling-inference 单元，并跨集群复制该拓扑；
- 高频张量流动限制在本地高速互联（如 NVLink / InfiniBand）；
- 权重广播使用 CPU 后端（Gloo）异步进行，避免与 CUDA stream 冲突。

---

### 相比现有方法的优势
| 方面 | 现有方法（如 RLinf-VLA, RL-VLA3） | D-VLA |
|------|-------------------------------|--------|
| 资源调度 | 混合或粗粒度分离，仍存在干扰 | 平面级解耦，彻底消除冲突 |
| 流水线设计 | 两阶段或三阶段异步 | 四线程完全异步，最大化重叠 |
| 显存管理 | 统一显存池，易受碎片影响 | 双池隔离，稳定性强 |
| 通信效率 | GPU 主导 all-reduce，易阻塞 | 控制平面 offload 至 CPU，降低 contention |
| 扩展性 | 多节点扩展时通信成为瓶颈 | 拓扑复制 + 局部聚合，支持线性加速 |

> 🔥 总结：D-VLA 从**系统架构层面重构了 VLA 训练范式**，实现了 simulation 与 learning 的“无感并行”。

---

## 2. 核心实验方法和设置

### 使用的数据集与环境
- **Simulation Environment**: `ManiSkill`  
  - 基于 GPU 加速的物理仿真框架（PhysX），支持高并发渲染与并行环境实例；
  - 区别于传统 CPU-bound 环境（如 Gym/MuJoCo），更贴近真实 embodied AI 场景。
- **Benchmark Task**: `LIBERO`（用于验证迁移能力与泛化性）

### 模型架构
测试两种主流 VLA 范式：
1. **T0.5**：基于扩散过程（diffusion-based）的动作预测模型；
2. **OpenVLA-OFT**：自回归 Transformer 架构，采用 PEFT 微调。

> 两者均采用 action chunking 预测方式，即一次输出多个未来动作，以提高交互效率。

### 基线方法对比
在 16-GPU 集群上对比以下框架及其典型部署模式：

| 基线方法 | 部署策略 | 描述 |
|---------|----------|------|
| **RLinf-VLA** | Colocated (co), Disaggregated (dis), Hybrid (hyper) | 支持多种资源划分，但同步机制导致 GPU bubbles |
| **RL-VLA3** | Disaggregated (2:4 GPU 分配) | 全异步三阶段流水线，已较先进但仍存在通信阻塞 |
| **D-VLA (Ours)** | Hybrid Asynchronous | 4 GPU 共享 rollout/environment，4 GPU 专用 actor，plane decoupling 设计 |

### 评估指标
- **Throughput（吞吐量）**：单位时间内处理的状态转移数（steps/sec），为核心指标；
- **Step Time / Rollout Time / Actor Time**：各阶段耗时分解；
- **Hardware Utilization**：GPU 利用率、通信等待时间占比；
- **Training Convergence**：在 ManiSkill 上的成功率曲线（success rate vs. training steps）；
- **Scalability**：随环境数量增加的吞吐变化趋势。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Figure 4–5）

#### 📈 在 T0.5 模型上的表现（3:1 资源分配）
| 方法 | Throughput (steps/s) | 提升幅度 |
|------|------------------------|--------|
| RLinf-co | 175.29 | — |
| RL-VLA3 | 250.77 | +43% |
| **D-VLA** | **376.00** | **+86.26% vs. RLinf-co**, +50% vs. RL-VLA3 |

#### 📈 在 OpenVLA-OFT 模型上的表现（1:1 分配）
| 方法 | Throughput (steps/s) | 提升幅度 |
|------|------------------------|--------|
| RLinf-co | 87.20 | — |
| RL-VLA3 | 170.48 | +95.5% |
| **D-VLA** | **250.90** | **+188% vs. RLinf-co**, +47% vs. RL-VLA3 |

> 💡 注：D-VLA 在参数量更大的 OpenVLA-OFT 上优势更为明显，说明其对大模型更具适应性。

### 与其他方法的关键对比结果
- **延迟控制优异**：
  - T0.5 任务中，D-VLA 的 step time 仅为 **566.41 μs**，相比 RLinf-dis（1006.8 μs）下降 **50.43%**；
  - OpenVLA-OFT 中也仅需 **520.3 μs**，远低于其他基线。
- **GPU 利用率接近饱和**：
  - 异步流水线有效掩盖了 inference 延迟，使 rollout 与 actor 几乎无空等。
- **多节点可扩展性强**：
  - 在 16-GPU 多节点环境下，D-VLA 保持近似线性加速，未被跨节点通信拖累。

### 消融实验与分析（隐含于实验部分）
虽然未明确列出消融表，但从以下分析可推断关键组件作用：

| 组件 | 贡献体现 |
|------|--------|
| **Plane Decoupling** | 是实现低延迟的基础，避免了 PhysX 与 CUDA stream 冲突 |
| **Swimlane Pipeline** | 实现了 rollout 与 training 的完全重叠，压缩单步周期 |
| **Dual-Pool Memory** | 防止物理引擎频繁 malloc/free 导致 OOM 或卡顿 |
| **Topology-Aware Replication** | 保证大规模扩展时不退化为全局同步模式 |

> ⚠️ 实验还发现：当 actor 推理负载过重时（如 OpenVLA-OFT 在 3:1 设置下），会成为新瓶颈 → 表明 **动态资源调整** 的必要性。

---

## 4. 关键结论和发现

### 主要发现
1. **simulation 与 learning 的强耦合是当前 VLA 训练的主要性能瓶颈**；
2. **Plane Decoupling 架构能从根本上缓解资源争用问题**，是实现高并发的关键；
3. **四线程 Swimlane 流水线可实现近乎完美的计算-通信重叠**，将 GPU bubbles 最小化；
4. **D-VLA 在 billion-parameter 级 VLA 模型上实现高达 86% 的吞吐提升**，且不影响收敛质量；
5. **在 trillion-parameter 规模测试中表现出良好的稳定性和线性加速潜力**。

### 方法的局限性
- **依赖硬件拓扑配置**：最优性能需配合高速互联（如 InfiniBand/NVLink）才能发挥；
- **静态资源划分可能不适应动态负载**：目前采用固定比例（如 3:1），尚未集成实时负载感知的弹性调度；
- **主要面向单智能体场景**：尚未验证在 multi-agent 或复杂 sim-to-real 迁移中的表现。

### 未来工作方向
1. **动态负载感知的资源再分配机制**：根据实时 latency 自动调整 rollout/actor GPU 数量；
2. **扩展至 multi-agent 协同训练场景**；
3. **集成更多异构平台支持**（如边缘设备、不同机器人形态）；
4. **探索更高效的 reward shaping 与 GRPO 变体**，进一步提升样本效率。

---

> ✅ **总体评价**：  
> D-VLA 不仅是一个高性能系统框架，更是为下一代 **general-purpose embodied agents** 提供了可扩展、高效率的训练基础设施蓝图。它标志着 VLA 训练正从“模仿学习主导”向“在线强化学习驱动”的范式转变迈出关键一步。

</details>

---

### 3. [Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models](https://arxiv.org/abs/2605.11854)

**Authors**: Kecheng Chen, Ziru Liu, Xijia Tao, Hui Liu, Yibing Liu, Xinyu Fu, Shi Wu, Suiyun Zhang, Dandan Tu, Lingpeng Kong, Rui Liu, Haoliang Li  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2605.11854v1  

#### Abstract
Diffusion Language Models (DLMs) have recently emerged as a promising alternative to autoregressive language models, offering stronger global awareness and highly parallel generation. However, post-training DLMs with standard Negative Evidence Lower Bound (NELBO)-based supervised fine-tuning remains...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **Diffusion Language Models (DLMs)** 在后训练阶段存在的 **训练-推理不一致**（training-inference discrepancy）问题。具体表现为：
- **训练过程**：采用标准的 **Negative Evidence Lower Bound (NELBO)** 目标，通过随机掩码、单步重建进行监督微调（SFT），隐含地引入了“均匀归纳偏置”（uniform inductive bias），即所有被掩码 token 被同等对待。
- **推理过程**：实际解码遵循基于置信度或熵引导的多步“由易到难”（easy-to-hard）去噪路径，优先恢复高置信度（低熵）token。

这种不一致性导致即使使用 **self-distilled trajectories**（自蒸馏轨迹）进行训练，模型也难以充分吸收其中蕴含的结构化生成知识，仅能实现有限的推理加速，而无法显著提升性能，甚至在全步长解码下出现性能下降。

### 提出了什么新方法或新思路
作者提出了一种名为 **Trajectory-Aligned optimization via Boltzmann Modeling (TABOM)** 的新型后训练框架，其核心思想是：
- 将推理过程中观察到的“由易到难”解码偏好形式化为一个 **Boltzmann 分布**，该分布以每个 token 的理想预测熵（ideal predictive entropy）作为能量函数。
- 通过最小化模型预测的 unmasked 分布与目标 Boltzmann 分布之间的 **KL 散度** 来对齐训练与推理行为。

由于直接优化 KL 散度不可行（涉及不可计算的配分函数 $Z$ 和全局采样），作者进一步设计了一个 **可处理的替代目标**：
- 引入 **Pairwise Ranking Loss**，强制模型对更容易（先解码）的 token 赋予更低的预测熵。
- 在局部时间窗口内构建 token 对 $(r, s)$，若 $r$ 在轨迹中早于 $s$ 被解码，则要求 $h_o(r; T) < h_o(s; T)$，并通过 hinge loss 进行优化。

### 相比现有方法的优势
| 方法 | 主要目标 | 是否提升生成质量 | 是否缓解灾难性遗忘 |
|------|----------|------------------|--------------------|
| **SFT-GT** | 注入外部 GT 数据 | ✅（有限） | ❌（严重遗忘） |
| **SFT-SD / dInfer / T3D** | 利用 self-distilled 轨迹压缩采样步数 | ❌（主要用于加速） | ✅（防止遗忘） |
| **TABOM (Ours)** | **利用轨迹进行知识获取与能力提升** | ✅✅（显著提升） | ✅✅（完全缓解） |

**TABOM 的优势在于**：
- 不仅保留了 self-distillation 防止灾难性遗忘的优点；
- 更进一步将轨迹视为高质量的教学示范，从中学习“何时解码哪个 token”的策略，从而真正提升了模型的知识边界和生成能力。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **数学推理**：`MixChain-Z-PRM12K`（12K queries）
- **代码生成**：`Ling-Coder-SFT`（18K queries）
- 所有 self-distilled 数据均由 base model 自主生成，确保分布一致性。

### 实验设置和评估指标
- **基础模型**：
  - `Dream-7B-Instruct`
  - `LLaDA-8B-Instruct`
- **训练配置**：
  - 使用 **LoRA**（rank=16）进行参数高效微调；
  - batch size: 32（8 GPUs × 4）；
  - 学习率：2e-5，cosine decay，warm-up 50 steps；
  - 训练轮数：5 epochs；
  - TABOM 窗口大小 $W=32$，margin $\gamma \in \{0.1,0.2,0.3\}$，ranking weight $\lambda \in \{1,2\}$。
- **评估任务**：
  - **数学推理**：GSM8K、MATH500
  - **代码生成**：HumanEval、MBPP
  - **指令跟随**：IFEval
- **评估方式**：
  - 报告 in-domain 和 out-of-distribution (OOD) 性能；
  - 使用官方默认推理超参（见附录 Table 8）；
  - 所有结果取最佳 checkpoint。

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **No-SFT** | 原始 DLM，无任何微调 |
| **SFT-GT** | 使用离线 ground-truth 数据的标准 SFT |
| **SFT-SD** | 使用 self-distilled 轨迹的标准 SFT |
| **dInfer** | 学习从后期状态跳跃到早期状态的压缩转换 |
| **T3D** | 结合 direct discriminative optimization 和路径一致性加权 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Dream-7B-Instruct）

#### 表：代码生成任务（平均提升 +5.15%，无灾难性遗忘）
| Method | HumanEval↑ | MBPP↑ | Avg.↑ | GSM8K↓ | MATH500↓ | IFEval↓ | OOD Avg.↑ |
|--------|------------|-------|--------|--------|----------|---------|-----------|
| No-SFT | 52.66      | 58.00 | 55.33  | 81.41  | 39.80    | 56.56   | 59.26     |
| SFT-GT | **61.55** (+8.89) | 58.00 (+0.00) | **59.78** (+4.45) | 52.33 (-29.08) | 32.40 (-7.40) | 46.21 (-10.35) | 43.65 (-15.61) |
| SFT-SD | 53.66 (+1.00) | 59.20 (+1.20) | 56.43 (+1.10) | 81.81 (+0.40) | 41.60 (+1.80) | 57.10 (+0.54) | 60.17 (+0.91) |
| TABOM  | **60.36** (+7.70) | **60.60** (+2.60) | **60.48** (+5.15) | 81.73 (+0.32) | **42.40** (+2.60) | 55.45 (-1.11) | **59.86** (+0.60) |

#### 表：数学推理任务（平均提升 +2.10%，OOD 正向增益）
| Method | GSM8K↑ | MATH500↑ | Avg.↑ | HumanEval↑ | MBPP↑ | IFEval↑ | OOD Avg.↑ |
|--------|--------|----------|--------|-------------|--------|---------|-----------|
| No-SFT | 81.41  | 39.80    | 60.61  | 52.66       | 58.00  | 56.56   | 55.74     |
| SFT-GT | 80.12 (-1.29) | 37.40 (-2.40) | 58.76 (-1.85) | 46.34 (-6.32) | 58.00 (+0.00) | 53.23 (-3.33) | 52.52 (-3.22) |
| SFT-SD | 81.95 (+0.54) | 39.80 (+0.00) | 60.88 (+0.27) | 57.92 (+5.26) | 58.60 (+0.60) | 56.01 (-0.55) | 57.51 (+1.77) |
| TABOM  | **84.31** (+2.90) | **41.10** (+1.30) | **62.71** (+2.10) | **58.54** (+5.88) | **59.20** (+1.20) | **56.19** (-0.37) | **57.98** (+2.24) |

> ✅ **TABOM 实现了“鱼与熊掌兼得”**：既大幅提升 in-domain 性能，又保持甚至增强 OOD 能力。

### 消融实验结果（Ablation Study on Dream）

| 方法 | GSM8K | MATH500 | HumanEval | MBPP |
|------|--------|----------|-------------|--------|
| Base (SFT-SD) | 81.95 | 39.80 | 57.92 | 58.60 |
| + Traj. Masking only | 82.18 | 41.20 | 56.45 | 58.70 |
| + Pairwise Ranking (Global) | 83.10 | 40.20 | 57.50 | 58.20 |
| + Pairwise Ranking (**Local, W=32**) | **84.31** | **41.10** | **58.54** | **59.20** |

- **关键发现**：
  - 仅使用 trajectory-aware masking 提升有限；
  - 加入 pairwise ranking 显著提升性能；
  - **局部窗口（local window）优于全局比较**，避免跨阶段噪声干扰。

### 并行解码鲁棒性（Parallel Decoding Robustness）
在固定 2-token 并行解码下，TABOM 仍保持高性能，而 SFT-GT 性能大幅下降，说明 TABOM 更适应多 token 同时生成场景。

---

## 4. 关键结论和发现

### 主要发现
1. **Self-distilled trajectories 本身不足以解决训练-推理不一致**：尽管提供了更低的优化壁垒，但直接用 NELBO 训练只能带来边际收益。
2. **必须显式建模“由易到难”的推理偏好**：通过将推理 unmasked 分布建模为 Boltzmann 分布，并使用 pairwise ranking 实现可处理优化，才能真正释放轨迹中的知识潜力。
3. **TABOM 成功解决了 SFT 中的“遗忘 vs. 增益”困境**：实现了 in-domain 性能飞跃的同时，完全避免了灾难性遗忘，甚至扩展了模型的知识边界。
4. **Trajectory Discrimination Score (TDS)** 是有效诊断工具：TABOM 显著提高了 TDS 值（如 Dream 上 HumanEval 从 0.035 → 0.711），表明其成功塑造了更具区分性的熵景观。

### 方法的局限性
- 当前方法依赖于高质量的 self-distilled 轨迹，若 base model 本身存在系统性错误，可能传播偏差；
- Pairwise ranking 的效果受限于局部窗口大小 $W$，过大或过小均影响性能（实验显示 $W=32$ 最优）；
- 目前未探索与其他对齐技术（如 RLHF）的结合。

### 未来工作方向
- 探索如何将 TABOM 与强化学习（RL）结合，实现更细粒度的决策优化；
- 将该范式推广至其他生成模型（如图像 diffusion models）；
- 研究如何动态调整 ranking margin 或 window size 以适应不同任务复杂度；
- 构建更高效的 trajectory 采样与存储机制，支持更大规模训练。

---

> **总结一句话**：  
> **TABOM 重新定义了 self-distillation 的用途——不再只是“提速器”，而是“知识放大器”**。它通过能量排序机制，让 DLM 在训练中学会模仿自己的最优生成路径，从而桥接训练与推理鸿沟，在不牺牲泛化能力的前提下实现性能跃迁。

</details>

---

### 4. [TurboGR: An Accelerated Training System for Large-Scale Generative Recommendation](https://arxiv.org/abs/2605.13433)

**Authors**: Huichao Chai, Zhixin Wu, Xuemiao Li, Shiqing Fan, Hengfeng Wang, Maojun Peng, Lu Xu, Yaoyuan Wang, Yibo Jin, Wei Guo, Yongxiang Feng  
**Category**: cs.DC  
**Published**: 2026-05-14  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2605.13433v1  

#### Abstract
Generative recommendation (GR) has emerged as a promising paradigm that replaces fragmented, scenario-specific architectures with unified Transformer-based models, exhibiting scaling-law behavior where recommendation quality improves systematically with increased model capacity and training data. Ho...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TurboGR: An Accelerated Training System for Large-Scale Generative Recommendation 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

该论文针对在 **Ascend NPU** 上部署大规模 **Generative Recommendation (GR)** 模型时面临的三大系统级瓶颈：

1. **Jagged Variable-Length Sequences（锯齿状变长序列）**  
   - 用户行为序列长度不一，导致大量填充（padding），引发冗余计算和内存浪费，使 **MFU（Model FLOPs Utilization）低于10%**。

2. **Sparse-Dense Communication Bottleneck（稀疏-稠密通信瓶颈）**  
   - 稀疏的 Embedding Table 与稠密的 Transformer 主干耦合，导致 All-to-All 通信开销巨大，分布式扩展性差（scalability < 0.6）。

3. **Memory-Intensive Negative Sampling（负采样内存消耗大）**  
   - 长序列下的 token-level 负采样需要存储大量负样本嵌入，占用过多 HBM（High Bandwidth Memory），限制了大规模召回训练的可行性。

此外，Ascend NPU 缺乏对 jagged 操作和稀疏原语的高性能支持，加剧了上述问题。

---

### 提出的新方法与创新点

为解决上述挑战，作者提出了 **TurboGR** —— 一种专为 Ascend NPU 设计的、具有硬件亲和性的 GR 训练系统，包含三大核心技术：

#### (1) Ascend-affinity Jagged Acceleration（锯齿加速）
- **Jagged Fusion Operators**：将 Attention 和 RAB（Relative Attention Bias）操作融合为单一内核，消除密集-锯齿格式转换，减少内存访问和内核启动开销。
- **Jagged Embedding Lookup Acceleration**：基于 KeyedJaggedTensor（KJT）优化索引方式，并采用表级数据重组与核分区策略，提升缓存命中率。
- **Dynamic Load Balancing**：
  - **Token-Aware Dynamic Batch Scaling**：按 token 数而非样本数动态调整 batch size。
  - **Global Token Reallocation**：跨设备重分配样本以平衡负载。

✅ 效果：**内存减少 70%，延迟降低 2.2×，设备间负载不平衡从 47% 降至 2.4%**。

#### (2) Distributed Communication Optimization（分布式通信优化）
- **Hierarchical Sparse Parallelism (HSP)**：将设备分组，每组内部进行模型并行，组间数据并行；通过局部 All-to-All 减少通信规模。
- **Semi-Asynchronous Training**：解除稀疏前向与后向之间的依赖，实现一个 batch 的稀疏前向可提前执行，掩盖通信延迟。
- **Fine-Grained Pipeline Orchestration**：设计六阶段细粒度流水线（如 DataLoader → Feature All-to-All → Embedding Forward → Dense Module → Embedding Backward），最大化 CPU/NPU 并发。

✅ 效果：**All-to-All 延迟下降 75.9%，整体通信延迟下降 39.1%，NPU 利用率达 94%**。

#### (3) Negative Sampling Optimization（负采样优化）
- **Asynchronous Offloading**：将负样本嵌入异步卸载至 CPU 内存，按段加载回 NPU，避免全程驻留 HBM。
- **Jaggedness-aware FP16 Quantization**：仅对负样本嵌入使用 FP16 存储和查找，正样本保持精度。
- **Intra-batch Logit Sharing**：复用同一批次中其他 token 的负样本 logits 作为辅助负样本，扩大有效负空间而不增加查表开销。

✅ 效果：**HBM 使用最多减少 24.59%，推荐质量无显著损失**。

---

### 相比现有方法的优势

| 维度 | 现有方案（如 TorchRec + Megatron） | TurboGR |
|------|-------------------------------|--------|
| **硬件适配性** | 主要面向 GPU，未针对 Ascend 优化 | 全栈适配 Ascend NPU 架构特性 |
| **Jagged 处理** | 使用 Padding + Dense Tensor，冗余高 | 原生支持 Jagged Tensor，消除 padding |
| **通信效率** | 全局 All-to-All，扩展性差 | HSP 分层并行 + Semi-Async 掩盖延迟 |
| **内存管理** | 负样本全驻留 HBM | 异步卸载 + FP16 + Logit 共享 |
| **系统集成** | 多组件拼接，协调复杂 | 统一 GR-Engine，模块化设计 |

> ✅ **TurboGR 是首个开源的、专为 Ascend NPU 构建的大规模 GR 训练系统**。

---

## 2. 核心实验方法和设置

### 数据集

- **KuaiRand-27K**：当前最大公开短视推荐数据集之一。
  - 包含超过 27,000 名用户的交互日志。
  - 单用户平均历史更长，适合评估长序列建模能力。
  - 序列长度支持到 4096，远超常规基准。

> ✅ 特别适用于测试 GR 模型在 **长尾分布、变长输入、大规模召回** 场景下的表现。

---

### 实验设置

- **硬件平台**：
  - Ascend 910B1 NPU（64GB HBM），集群规模从 32 到 128 NPUs（4–16 节点）。
  - CPU：Kunpeng-920 ARM。
- **模型架构**：
  - HSTU [12] 和 FuXi [14]：基于 Transformer 的生成式推荐模型。
  - 多种尺寸：tiny, small, medium, large, long（参数量最高达 0.2B）。
- **训练配置**：
  - 优化器：AdamW（lr = 4e-3），TF32 精度。
  - 负采样：128 negatives，默认使用 Sampled Softmax。
  - 异步更新 Embedding。
- **评估指标**：
  - **MFU（Model FLOPs Utilization）**：衡量硬件利用率。
  - **Throughput (samples/sec)**：吞吐量。
  - **Scalability（线性扩展性）**：多卡加速比。
  - **HR@K, NDCG@K**：推荐准确性指标。
  - **End-to-End Latency, Communication Overhead, HBM Usage**：系统性能指标。

---

### 基线方法对比

- **Baseline**：标准 TorchRec + Megatron 实现，直接迁移到 Ascend。
- **对比维度**：
  - 是否启用 Jagged Fusion
  - 是否使用 HSP 替代全局 All-to-All
  - 是否开启 Semi-Async 和 Pipeline Orchestration
  - 是否应用 Offloading / FP16 / Logit Sharing

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Model | 参数量 (M) | Seq Len | Comp. Complexity (TFLOPs/step) | Throughput (sample/s) | **MFU (%)** | **Scalability** |
|-------|-----------|---------|-------------------------------|------------------------|-------------|------------------|
| HSTU-large | 83.97 | 2048 | 4.33 | 1616.83 | 24.74 | 0.93 |
| HSTU-long | 83.97 | 4096 | 7.15 | 770.79 | **34.08** | **0.97** |
| FuXi-large | 201.55 | 2048 | 8.25 | 1156.26 | **39.34** | 0.94 |
| **FuXi-long** | **201.55** | **4096** | **11.54** | **574.39** | **54.71** | **0.97** |

> 🔥 **TurboGR 在 0.2B 参数级别达到 54.71% MFU，接近线性扩展性（0.97）**。

---

### 与基线方法的对比结果

#### (1) Jagged Fusion vs Baseline（Figure 2）
- 序列长度 8k 时：
  - **端到端延迟从 961.21ms → 431.13ms（↓55%）**
  - **内存占用从 47.77GB → 14.31GB（↓70%）**

#### (2) Jagged Embedding Lookup（Table 2）
- 百万 ID 查询（50.43% 填充）：
  - **前向延迟 ↓6×（18ms → 3ms）**
  - **反向延迟 ↓4×（36ms → 9ms）**

#### (3) 动态负载均衡（Table 3）
- KuaiRand-27K 上：
  - 最大 token 差异从 **10,726 → 559**
  - 同步等待时间占比从 **47.01% → 2.40%**

#### (4) HSP 通信优化（Table 4）
- All-to-All 延迟 ↓**75.9%**（498ms → 120ms）
- 总通信延迟 ↓**39.1%**

#### (5) 负采样异步卸载（Table 7）
- 使用 128 个负样本时：
  - **HBM 占用从 50.39GB → 34.27GB（↓24.59%）**

#### (6) FP16 Quantization 对准确率影响（Figure 12）
- HR@2000 差异 < 0.01%
- NDCG@1000 差异 < 0.05%
> ✅ **精度损失可忽略**

---

### 消融实验结果

#### (1) 动态负载均衡（Table 3）
- Amazon-all（短序列）：
  - 固定 batch → 最大 token 差 623
  - Token-Aware → 差值降至 31，通信延迟占比 ↓至 1.48%

#### (2) Intra-batch Logit Sharing（Table 8）
- FuXi-large 使用 64 negatives + k=4 扩展至等效 256：
  - NDCG@200 达 **0.0360**，优于 baseline（0.0347）
  - 实现“用更少查表换更强判别力”

#### (3) 扩展因子建议（Table 9）
- 高维模型需更高 k 值补偿 batch 缩小带来的信息不足：
  - 如 FuXi-large（dim > 1024）需 k=4 才能匹配 128-negative baseline

---

## 4. 关键结论和发现

### 主要发现

1. **GR 模型具备强扩展性（Scaling Law）**，但其潜力受限于底层系统效率。
2. **Ascend NPU 不适合直接运行传统 DLRM 或 GPU 优化框架**，必须重构算法与系统协同设计。
3. **Jagged Tensor 是提升 GR 效率的关键路径**，原生支持可大幅降低内存与延迟。
4. **通信不再是不可逾越的障碍**：通过 HSP + Semi-Async + Pipeline 可实现近线性扩展。
5. **负采样可通过“共享+卸载+量化”组合拳突破内存墙**，无需牺牲准确性。

---

### 方法的局限性

1. **目前仅支持单一流水线结构**，尚未引入 MoE 或动态路由机制。
2. **未探索自动并行策略搜索**，仍需人工调优 HSP 分组大小、pipeline 深度等超参。
3. **长序列处理上限仍在 8k 级别**，尚未验证 16K–32K 超长上下文场景。
4. **依赖特定数据预处理流程**（如 KJT 格式），通用性有一定门槛。

---

### 未来工作方向（原文 Conclusion）

1. **支持 Mixture-of-Experts (MoE)** 架构，带来更大容量的同时需解决专家负载均衡问题。
2. **开发面向 Ascend 的稀疏注意力机制**（Sparse Attention），进一步延长上下文窗口。
3. **集成自动并行搜索系统**，联合优化 Hybrid Parallelism 策略，提升跨集群可移植性。
4. **拓展至多模态生成式推荐**（如图文、视频内容生成），推动 GR 成为统一 AI Agent 推荐引擎。

---

> 📌 **总结一句话**：  
> **TurboGR 通过软硬协同设计，在 Ascend NPU 上实现了高达 54.71% MFU 和 0.97 扩展性的大规模生成式推荐训练，是迈向工业级 GR 系统的重要一步**。

</details>

---

### 5. [BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion](https://arxiv.org/abs/2605.11577)

**Authors**: Shaobin Zhuang, Yuang Ai, Jiaming Han, Xiaohui Li, Huaibo Huang, Xiangyu Yue, Xuefeng Hu, Kun Xu, Yali Wang, Hao Chen  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.11577v1  

#### Abstract
Autoregressive language models generate text one token at a time, yet natural language is inherently structured in multi-token units, including phrases, n-grams, and collocations that carry meaning jointly. This one-token bottleneck limits both the expressiveness of the model during pre-training and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **autoregressive language models (AR LLMs)** 采用“逐 token 生成”的范式，即每一步通过 **vocabulary softmax** 预测下一个 token。这种模式存在两个根本瓶颈：
- **表达能力受限**：自然语言的基本单位是短语、n-gram 或搭配，而非孤立的 token。
- **推理效率低下**：生成过程本质上是串行的，限制了吞吐量。

尽管已有如 speculative decoding、multi-token prediction 等加速方法，但它们通常：
- 不改变底层的“分类决策”接口；
- 或牺牲了语言建模所需的因果结构（causal structure）。

---

### 🚀 提出的新方法与核心思想

**BitLM** 提出了一种全新的语言生成范式，其核心创新在于：

#### （1）将 token 生成重构为 **bitwise 连续扩散（bitwise continuous diffusion）**
- 每个 token 被表示为一个固定长度的 **binary code**（例如 18-bit），映射到 $\{-1, +1\}^B$ 的超立方体顶点。
- 将传统的 **vocabulary softmax 分类任务** 替换为在连续二进制空间中的 **denoising diffusion 任务**。
- 生成不再是“选择哪个 token”，而是“逐步去噪恢复一组比特”。

#### （2）引入 **block-causal 因果结构 + 扩散头（diffusion head）**
- 使用标准的 **causal LLM backbone**（如 Qwen 架构）进行上下文建模。
- 引入轻量级 **diffusion head**，对多个未来 token 的 binary codes 进行 **并行联合去噪**。
- 注意力掩码从全因果变为 **block-causal**：块内可双向交互，块间保持因果依赖。

#### （3）实现原生的 **multi-token 并行生成**
- 每次 backbone 推理后，diffusion head 可一次性生成一个 block（如 4 个 token）。
- 并行性来自模型本身的输出接口设计，而非后处理技巧（如 speculative decoding）。

---

### 🔍 相比现有方法的优势

| 方法类型 | 局限性 | BitLM 的改进 |
|--------|------|-------------|
| Autoregressive (Softmax) | 串行生成，速度慢 | 支持 block-level 并行生成 |
| Speculative/Multi-token Decoding | 仍基于 softmax，需额外 proposal 模型 | 原生支持多 token 联合生成，无需外部机制 |
| Diffusion-based LM | 多在 embedding 或 mask 空间操作，难以保持因果性 | 在 binary token 空间操作，保留 backbone 因果结构 |
| Non-AR/Semi-AR | 易破坏语言流畅性和一致性 | 保持左到右推理链，仅在局部块内并行 |

> 💡 **核心洞见**：大型词汇表 softmax 是历史选择，而非必要终点；改变符号输出空间的几何结构（geometry），可以自然解锁新的解码范式。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **预训练（Pretraining）**：
  - 使用 **FineWeb** 子集，共 **350B tokens**。
- **微调与评估（Fine-tuning & Evaluation）**：
  - 在 **XSum** 数据集上进行监督微调（supervised fine-tuning），用于摘要任务评估。

### ⚙️ 实验设置
- **模型架构**：
  - Backbone：基于 **Qwen-3** 架构。
  - Diffusion Head：借鉴 **BitDance** 设计，轻量级 MLP 结构。
- **Binary Code 设置**：
  - Code length $B = 18$，足以覆盖主流 tokenizer 的 vocab size（$2^{18} \approx 260k$）。
- **Block Size**：
  - $m = 4$，即每次生成 4 个 token 组成的 block。
- **训练细节**：
  - Optimizer: AdamW ($lr=1e^{-4}, \beta_1=0.9, \beta_2=0.95$)
  - 序列长度：16384 tokens / packed sequence
  - 输入：将原始文本 tokenize 后转为 binary codes，再经 MLP 投影至 hidden dim。
- **推理设置**：
  - Denoising Steps: $K = 15$
  - 使用 **ODE solver** 和 **classifier-free guidance (CFG=9.0)** 提升生成质量。

### 📊 评估指标
- 主要使用 **ROUGE** 系列指标：
  - ROUGE-1 (R1)
  - ROUGE-2 (R2)
  - ROUGE-L (RL)
- 对比不同配置下的性能变化（消融实验）。

### 🆚 基线方法对比
- **经典摘要模型**：
  - ILead-3, PTGen, PTGen+Cov
- **不同头部结构的 BitLM 变体**：
  - BitLM w/ LM Head（softmax 输出）
  - BitLM w/ Diffusion Head（本文提出）
- 所有模型均为 **8B 参数规模**，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（XSum 测试集）

| Method | R1 | R2 | RL |
|-------|----|----|----|
| ILead-3 | 16.30 | 1.60 | 11.95 |
| PTGen (See et al., 2017) | 29.70 | 9.21 | 23.24 |
| **BitLM 8B w/ LM Head (FT)** | 23.20 | 4.45 | 18.04 |
| **BitLM 8B w/ Diff. Head (FT)** | **26.05** | **6.44** | **20.12** |

> ✅ 微调后的 BitLM（diffusion head）显著优于自身 softmax 版本，且接近经典指针生成器模型水平。

---

### 🔬 消融实验结果

#### （1）**Denoising Steps 与 CFG 影响**（图 4）
- 最优配置出现在：
  - **Denoising Steps = 15**
  - **CFG = 9.0**
- 性能随步数增加先升后稳，表明足够迭代对高质量生成至关重要。

#### （2）**模型可扩展性**（图 3）
- 在 0.6B → 8B 规模下预训练 loss 持续下降，说明：
  - BitLM 具备良好的 **scalability**。
  - 无需特殊结构即可稳定训练。

#### （3）**不同头部结构对比**
- 使用 diffusion head 的版本在所有指标上均优于使用传统 LM head 的版本。
- 表明 **binary diffusion 接口本身具有建模优势**，不仅仅是加速手段。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **“一次一个 token”不是必须的**：
   - 语言生成可以被重新定义为在紧凑 binary 空间中的迭代去噪过程。
   - 改变输出空间的几何结构（从 simplex 到 hypercube）能自然引出更高效的生成方式。

2. **block-causal 并行是原生能力**：
   - BitLM 的并行性源于其生成接口设计，而非外部加速策略。
   - 实现了 **可靠性（causal backbone）与效率（parallel realization）的统一**。

3. **vocabulary softmax 可被替代**：
   - 实验证明，用 diffusion 替代 softmax 仍能有效完成下游任务（如摘要），具备可行性。

4. **高效训练与快速推理潜力**：
   - 由于避免了大 vocab softmax 的计算开销，BitLM 在训练和推理阶段均有潜在加速空间。

---

### ⚠️ 方法的局限性
1. **当前性能尚未超越 SOTA**：
   - 在 XSum 上虽表现合理，但仍落后于 PTGen 等经典模型，说明当前实现尚不充分优化。
2. **fixed binary code 缺乏语义结构**：
   - 当前 binary code 是静态映射（integer → bits），未学习语义化编码。
3. **固定 block size 限制灵活性**：
   - $m=4$ 是超参，可能不适合所有场景；动态 block 大小有待探索。
4. **推理延迟受 denoising steps 影响**：
   - 虽然每次生成多个 token，但需多次 diffusion 步骤，实际加速比需进一步测量。

---

### 🔮 未来工作方向
1. **Learned Binary Codes**：
   - 探索可学习的、语义感知的 binary token 表示。
2. **Adaptive Block Sizes**：
   - 根据内容复杂度动态调整 block 长度。
3. **Hybrid Architectures**：
   - 结合 softmax 与 diffusion 的优点，例如在 coarse level 用 diffusion，在 fine level 用 autoregressive refinement。
4. **端到端推理加速评测**：
   - 测量真实场景下的 **throughput 提升** 和 **latency 降低**。
5. **扩展至多模态**：
   - 利用统一 binary interface 实现 text-image-audio 的联合生成（参考 UniWeTok）。

---

## 📌 总结

> **BitLM 提出了一种颠覆性的语言生成视角：把 token 生成看作 binary code 的连续去噪过程。它不仅挑战了“vocabulary softmax 是唯一出路”的默认假设，还为高效、并行的语言模型设计开辟了全新路径。**

虽然目前性能仍有提升空间，但其实验验证了“改变输出空间几何结构”这一设计维度的巨大潜力，有望推动下一代 LLM 架构的发展。

</details>

---

### 6. [Efficient LLM-based Advertising via Model Compression and Parallel Verification](https://arxiv.org/abs/2605.11582)

**Authors**: Wenxin Dong, Chang Gao, Guanghui Yu, Xuewu Jiao, Mingqing Hu, Qiang Fu, Peng Xu, Penghui Wei, Hui Xu, Yue Xing, Shuanglong Li, Lin Liu  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.11582v1  

#### Abstract
Large language models (LLMs) have shown remarkable potential in advertising scenarios such as ad creative generation and targeted advertising. However, deploying LLMs in real-time advertising systems poses significant challenges due to their high inference latency and computational cost. In this pap...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Efficient LLM-based Advertising via Model Compression and Parallel Verification》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于 Large Language Models（LLMs）的在线广告系统面临两大挑战：
- **高推理延迟（high inference latency）**：LLMs 参数量大，导致实时生成广告内容时响应速度慢。
- **高计算成本（high computational cost）**：FP16 全精度模型在大规模部署中资源消耗巨大。

这些问题严重制约了 LLM 在时间敏感型场景（如实时广告投放）中的应用。

---

### 🚀 提出的新方法与创新思路

作者提出了一套完整的 **高效生成式广告框架（Efficient Generative Targeting Framework）**，结合以下两个核心技术：

#### （1）**Model Compression（模型压缩）**
- **Adaptive Group-Wise Quantization（自适应分组量化）**  
  将模型层按参数敏感度划分为“敏感”与“非敏感”层，对前者采用细粒度（更多组）、后者粗粒度（更少组）进行 INT4 量化，实现精度损失最小化下的高效压缩。
  
- **Layer-wise Semi-Structured Sparsity（逐层半结构化稀疏）**  
  基于重要性分析，在不同 Transformer 层施加不同的 N:M 稀疏比例（如关键层保留 2:4 密度，次要层用 1:4），兼顾效率与质量。

- **Index-Compressed 2bit-CSR 数据结构**  
  改进传统的 Compressed Sparse Row（CSR）格式，将索引压缩至原大小的 30%，显著降低内存带宽开销。

- **自研 SparseGemv 加速内核（Kernel）**  
  支持 INT4 权重量化 + 半结构化稀疏矩阵乘法，填补了 NVIDIA cuSparse/cuSparseLT 对 GEMV 类操作支持不足的空白。

#### （2）**Prefix Tree-based Parallel Verification（前缀树并行验证）**
- 构建语义驱动的 **Prefix Tree（Trie 结构）**，通过层次聚类算法组织广告候选文本（如品牌名“Taobao”、“Baidu”），形成顶部宽、底部窄的搜索路径。
- 设计 **动态触发机制（Parallel Verification Trigger）**：实时评估生成剩余 token 所需时间 vs 验证时间，选择最优时机启动并行解码。
- 引入 **Tree-Based Parallel Verification**：结合 Beam Search 和树形注意力约束，一次性并行验证多个候选序列，大幅减少解码步数。

> 🔥 创新亮点：据作者称，这是**首个将 Prefix Tree 约束解码与 Beam Search 完整结合应用于广告生成任务的工作**。

---

### ⚖️ 相比现有方法的优势

| 维度 | 本文方法优势 |
|------|-------------|
| **效率优化** | 同时从模型压缩（量化+稀疏）和解码策略（并行验证）双路径提升推理速度 |
| **硬件适配性** | 自定义 SparseGemv 内核专为工业级部署优化，优于通用库 |
| **业务定制化** | 前缀树构建针对广告实体设计，符合实际商业需求 |
| **灵活性与精度平衡** | 自适应量化避免一刀切，保护敏感层精度 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 场景 | 数据集 | 描述 |
|------|--------|------|
| **Targeted Advertising（定向广告）** | 公司内部私有数据集 | 包含百度平台收集的真实商业流量数据（未公开） |
| **Ad Creative Generation（广告创意生成）** | CSL [12] | 中文学术文献摘要数据集，用于模拟广告文案重写与关键词提取任务 |

> 注：由于涉及商业隐私，定向广告数据不可公开。

---

### ⚙️ 实验设置

| 项目 | 设置详情 |
|------|----------|
| **主干模型** | ERNIE 1.5B（百度自研 LLM） |
| **深度学习框架** | PaddlePaddle |
| **硬件平台** | <br>• 广告创意生成：NVIDIA A10 GPU，beam size = 1<br>• 定向广告：NVIDIA A30 GPU，beam size = 20（因需输出多候选） |
| **评估指标** | <br>• 推理延迟（Per-token latency, ms）<br>• Recall（召回率，衡量推荐准确性）<br>• BLEU / METEOR（生成质量评分）<br>• Speedup（加速比） |

---

### 🔁 基线方法对比

| 基线配置 | 描述 |
|--------|------|
| **Baseline (FP16)** | 原始全精度模型，无任何压缩或优化 |
| **Quantization Only** | 仅使用 INT4 量化（含自适应分组） |
| **Sparsity Only (2:4 或 1:4)** | 仅应用不同程度的半结构化剪枝 |
| **Sparse + Quant** | 量化 + 不同稀疏策略组合 |
| **+PTPV** | 加入 Prefix Tree Parallel Verification 优化 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（见 Table 1 及 Figure 2–3）

#### ✅ 总体加速效果
- 最终完整方案（Sparse(Mix)+Quant + PTPV）在真实环境中实现 **超过 1.8× 的端到端推理加速**。
- 消融实验显示最高可达 **1.89× speedup**（Sparse(1:4)+Quant），同时保持可接受的质量下降。

#### 📊 定向广告场景（Targeted Advertising）
| 方法 | Recall | Per-Token Latency (ms) | 相对提速 |
|------|-------|------------------------|---------|
| Baseline (FP16) | 60.0% | 1.9 | ×1.00 |
| Quantization | 60.0% | 1.2 | ×1.58 |
| Sparse + Quant | 60.0% | 1.1 | ×1.73 |
| +PTPV（完整方案） | 60.0% | **0.8** | **>×2.3** |

> 💡 结论：引入 PTPV 后延迟进一步下降近 30%，且未牺牲 Recall。

#### 📝 广告创意生成（Ad Creative Generation）
| 方法 | BLEU | METEOR | Avg Length | Latency (ms/token) |
|------|------|--------|------------|--------------------|
| Baseline (FP16) | 0.4247 | 0.6345 | 17.5 | 6.6 |
| Quantization | 0.4178 | 0.6283 | 17.6 | 4.8 |
| Sparse(2:4)+Quant | 0.4103 | 0.6195 | 17.5 | 4.0 |
| **Sparse(Mix)+Quant** | **0.4038** | **0.6127** | **17.5** | **3.7** |

> ✅ 质量影响可控：混合稀疏+量化下 BLEU 下降约 4.9%，METEOR 下降约 3.4%，但延迟降低 **43.9%**。

---

### 🔍 消融实验结果（Ablation Study）

| 技术组合 | Speedup | BLEU | METEOR | 观察结论 |
|--------|--------|------|--------|----------|
| Baseline (FP16) | ×1.00 | 0.4247 | 0.6345 | — |
| Quantization | ×1.37 | 0.4178 | 0.6283 | 显著降延迟，轻微损精度 |
| Sparsity (2:4) | ×1.25 | 0.4161 | 0.6260 | 效率增益有限 |
| Sparsity (1:4) | ×1.43 | 0.3476 | 0.5549 | 速度提升明显，但质量骤降 |
| **Sparse(2:4)+Quant** | **×1.65** | **0.4103** | **0.6195** | 最佳折中点之一 |
| **Sparse(1:4)+Quant** | ×1.89 | 0.3369 | 0.5446 | 极致加速，适合低质容忍场景 |
| **Sparse(Mix)+Quant** | **×1.78** | **0.4038** | **0.6127** | **综合最优：兼顾速度与质量** |

> 🔎 发现：**混合稀疏策略（Mix）优于单一固定比例**，能更好匹配各层的重要性分布。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **模型压缩 + 解码优化协同增效**：单独使用量化或稀疏虽有效，但二者结合 + 并行验证才能达到最佳加速比（>1.8×）。
2. **自适应量化优于统一量化**：根据不同层敏感度调整量化粒度，可在几乎不损失精度的前提下大幅提升效率。
3. **Prefix Tree 并行验证显著减少解码步数**：通过动态触发机制，在合适时机切换为并行验证，极大缩短长序列生成耗时。
4. **工业级可行性已验证**：该系统已在 **百度广告平台上线部署**，服务于大规模实时流量，证明其工程实用性。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **领域依赖性强** | 当前优化策略（尤其是前缀树构建）高度依赖广告业务语料，迁移到其他推荐场景可能需要重构 |
| **稀疏化质量代价明显** | 过度稀疏（如 1:4）会导致生成质量显著下降，限制其在高质量要求场景的应用 |
| **缺乏跨模型泛化验证** | 实验仅基于 ERNIE 1.5B，是否适用于更大或更小规模 LLM 尚待验证 |
| **未开放代码与数据** | 私有数据集和生产系统细节未公开，复现难度较高 |

---

### 🔮 未来工作方向

1. **引入强化学习与自适应算法**：动态调节量化粒度、稀疏比例及并行触发阈值，以应对多样化的输入请求。
2. **拓展至多模态广告生成**：将文本生成扩展到图文联合生成场景，探索视觉-语言联合压缩技术。
3. **增强跨域迁移能力**：研究通用性更强的前缀树构建方法，使其适用于新闻推荐、电商搜索等其他场景。
4. **探索训练-推理联合优化**：结合 retraining-based 方法，在微调阶段同步优化稀疏结构与量化稳定性。

---

> ✅ **总结一句话**：  
本论文提出了一种面向工业级 LLM 广告系统的高效推理框架，通过 **自适应量化 + 分层稀疏 + 前缀树并行验证** 的三重优化，在保证推荐精度的同时实现了 **超 1.8× 的实际加速**，并在百度广告平台成功落地，具有重要的实践价值。

</details>

---

### 7. [Training-Inference Consistent Segmented Execution for Long-Context LLMs](https://arxiv.org/abs/2605.11744)

**Authors**: Xianpeng Shang, Jiang Li, Zehua Duo, Qianyi Cai, Xiangdong Su  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.11744v1  

#### Abstract
Transformer-based large language models face severe scalability challenges in long-context generation due to the computational and memory costs of full-context attention. Under practical computation and memory constraints, many inference-efficient long-context methods improve efficiency by adopting ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Training-Inference Consistent Segmented Execution for Long-Context LLMs*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Transformer** 的大语言模型（LLMs）在处理长上下文（long-context）生成任务时面临严重的可扩展性挑战，其根源在于 **full-context attention** 的计算和内存开销随上下文长度呈二次增长。为提升推理效率，许多方法在 **inference 阶段**采用受限执行策略（如 bounded-context、chunked attention），但在 **training 阶段仍使用 full-context attention**，导致训练与推理之间存在 **execution mismatch** 和 **state-transition semantics 不一致**。

这种不一致性使得模型在训练中依赖的信息在推理时不可用，从而损害长上下文场景下的稳定性与泛化能力。

### 提出的新方法
本文提出一种 **训练-推理一致的分段执行框架（Training-Inference Consistent Segmented Execution）**，其核心思想是将分段执行（segmented execution）作为训练和推理共享的建模假设，而非仅在推理阶段应用的优化手段。

#### 主要创新点：
- **统一的前向执行语义**：训练和推理均按相同方式将序列划分为非重叠 segment，并通过两个跨段输入进行条件建模：
  - **Carried KV state**：一个固定大小的 KV 尾部，作为唯一可微的跨段状态接口，在训练中通过 **Truncated Backpropagation Through Time (TBPTT)** 控制梯度传播深度（最多 K 段）。
  - **Retrieved KV prefix**：从“只读”历史 KV 池中检索的前缀，用于提供远距离上下文信息，但以 **forward-only** 方式使用，不参与梯度传播。
- **理论保证的一致性**：证明在该设定下，TBPTT 可精确计算一个与推理目标一致的损失函数的梯度（即非近似），从而理论上确保训练与推理对齐。
- **Head- and Layer-Sparse 架构设计**：并非所有层和头都启用长程检索，而是选择性地在部分层（`L_long`）的部分头（`H_long`）上启用 retrieval，其余保持局部状态传递，兼顾效率与性能。

### 相比现有方法的优势
| 维度 | 传统方法（如 StreamingLLM, MInference） | 本文方法（Ours） |
|------|----------------------------------------|------------------|
| **训练-推理一致性** | ❌ 存在 mismatch | ✅ 完全对齐 |
| **梯度传播机制** | 全局 full-context 或无控制截断 | 明确限制为最多 K 段的 TBPTT |
| **长程信息访问** | 通常受限于窗口或压缩 | 支持 forward-only retrieval，保留远距离依赖 |
| **可扩展性** | 在极长上下文下内存压力大 | 内存占用与总长度无关，显著降低峰值内存 |

---

## 2. 核心实验方法和设置

### 数据集
- **PG19**：用于评估不同上下文长度下的语言建模困惑度（PPL）。
- **LongBench**（及子集 LongBench-E）：多任务长上下文理解基准，涵盖问答（QA）、摘要、代码、合成任务等。
- **RULER**：系统性测试长度外推能力的任务（如 CWE、FWE），支持从 4K 到 64K 的上下文扩展。

### 实验设置与评估指标
- **模型主干**：LLaMA2-7B-32K / 80K，以及 LLaMA3.1-8B-Instruct（用于验证泛化性）。
- **分段设置**：固定 segment 长度 $ S = 4096 $，携带 KV 长度 $ M = 512 $，检索前缀长度 $ R = 512 $。
- **训练配置**：fine-tune 1000 步，micro-batch size=1，gradient accumulation=8，使用 SlimPajama 数据集。
- **评估指标**：
  - **Perplexity (PPL)**：衡量语言建模稳定性。
  - **Task Accuracy / Score**：LongBench 各任务得分及平均分。
  - **Prefill Latency**：首 token 时间（TTFT）。
  - **Peak GPU Memory**：prefill 阶段最大显存占用。

### 基线方法对比
| 方法 | 类型 | 是否训练-推理一致 |
|------|------|------------------|
| **Vanilla Self-Attention** | Full-context | ❌（仅限短上下文可行） |
| **StreamingLLM** | Sliding window + sink | ❌ |
| **MInference** | Inference-time sparse attention | ❌ |
| **CCA (Core Context Aware)** | Compression-based alignment | ✅ |
| **DuoAttention** | Head separation for retrieval/streaming | ❌（训练未对齐） |
| **Ours** | Segment-level consistent execution | ✅ |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）语言建模困惑度（PG19）
- 在 LLaMA2-32K 和 80K 上，随着 context length 增加至 64K（超出训练长度），本文方法表现出更平滑的增长趋势，而其他方法出现剧烈波动或崩溃。
- **Figure 4** 显示，ours 在 64K 下仍保持稳定 PPL，而多数 baseline 性能骤降。

#### （2）下游任务表现（LongBench-E）
| 方法 | Avg Score (32K) | Avg Score (80K) | Peak Mem (GB, 32K) | TTFT (s, 32K) |
|------|----------------|----------------|--------------------|---------------|
| Vanilla SA | 23.13 | 23.38 | 23.61 | 1.62 |
| CCA | 21.12 | 21.98 | 28.08 | 1.79 |
| StreamingLLM | 21.90 | 21.56 | 22.19 | 1.59 |
| DuoAttention | 23.00 | 22.94 | 18.15 | 1.53 |
| **Ours** | **23.24** | **24.17** | **18.56** | **1.70** |

> ✅ 在 32K 和 80K 下均取得最高平均分，尤其在 **Summarization** 和 **Multi-document QA** 上优势明显。

#### （3）长度外推能力（RULER）
| 方法 | CWE Avg* (4K–32K) | FWE Avg* | CWE @64K | FWE @64K |
|------|-------------------|----------|----------|---------|
| Vanilla SA | 32.94 | 41.33 | – | – |
| StreamingLLM | 27.78 | 41.37 | – | – |
| CCA | 24.90 | 31.96 | – | – |
| **Ours** | **46.39** | **43.88** | 2.00 | 34.17 |

> ✅ 在有效范围内性能最优，且在 **64K**（超出训练范围）仍能保持非零准确率，体现更强的鲁棒性和外推能力。

#### （4）效率表现（Prefill 阶段）
- **Figure 5 & 6** 显示，在长上下文（如 128K）下：
  - **峰值内存**：相比 full attention，ours 在 128K 下实现约 **6× 更低的 peak prefill memory**。
  - **延迟-内存权衡**：在 64K 下，ours 在较低内存下维持合理延迟，优于其他方法（如 MInference 虽快但内存高，StreamingLLM 内存低但性能差）。

### 消融实验结果

#### （1）训练-推理一致性影响（Table 3）
| 方法 | Avg Score |
|------|-----------|
| **Aligned (TBPTT=1)** | **24.17** |
| Misaligned（训练 full-context，推理 segmented） | 11.91 |

> ❗ 移除一致性导致性能腰斩，验证了对齐的重要性。

#### （2）TBPTT 截断深度 $ K $
| 方法 | Avg Score |
|------|-----------|
| Aligned (K=1) | 24.17 |
| Aligned (K=2) | 24.07 |

> ✅ $ K=1 $ 已足够且最优，更深的回传并未带来收益，说明严格限制跨段信用分配即可达到最佳效果。

#### （3）局部状态容量（Local KV Size）
| Size | PPL (Avg) | LongBench-E Avg |
|------|-----------|------------------|
| 0 | 7.16 | 23.27 |
| 512 | 7.10 | 24.17 |
| 1024 | 7.07 | 24.19 |

> ✅ 引入有限大小的 carried KV state 即可显著提升性能，进一步增大收益递减。

#### （4）长程模块位置与头分组
- 使用更多 long-range layers 提升下游任务表现（尤其是需要跨段推理的任务），但不影响 PPL。
- **Prior-based head grouping**（基于先验选择具有 retrieval 行为的 head）优于 contiguous 或 interleaved 分组，表明 head 功能异质性重要。

---

## 4. 关键结论和发现

### 主要发现
1. **训练-推理一致性至关重要**：在受限执行范式下，若训练与推理语义不一致，会导致严重性能退化；显式对齐可显著提升长上下文稳定性与泛化能力。
2. **TBPTT 可精确而非近似地优化推理目标**：在受控的跨段接口下，即使只允许梯度回传 1 段（$ K=1 $），也能精确计算 inference-consistent objective 的梯度。
3. **分离短程连续性与长程检索通道有效**：通过 carried KV 实现局部状态延续，通过 forward-only retrieval 获取远距离证据，既能保持高效又能增强建模能力。
4. **稀疏启用 long-range heads/layer 是高效设计**：无需全局启用 retrieval，选择性激活即可获得大部分增益。

### 方法的局限性
- **依赖 fine-tuning**：需对预训练模型进行 alignment fine-tuning，不能直接部署于原始 checkpoint。
- **forward-only retrieval 不更新历史状态**：虽然避免了梯度复杂性，但也意味着无法通过训练优化 retrieval 策略本身。
- **segment 划分可能破坏语义边界**：固定长度分段可能割裂句子或段落，影响某些任务的理解质量（尽管实验中未观察到明显负面影响）。

### 未来工作方向
- 探索动态 segment 划分机制（如基于语义边界）。
- 设计可学习的 retrieval policy 或 memory 更新机制，在保持一致性的同时增强长期记忆能力。
- 将该框架扩展至 encoder-decoder 架构或多模态长上下文建模。
- 进一步研究 head-level 功能分工的机理，指导更优的 sparse 架构设计。

--- 

> 📌 **总结一句话**：  
> 本文提出了首个在 **segmented execution** 范式下实现 **训练-推理完全一致** 的长上下文 LLM 框架，通过 **controlled state propagation + forward-only retrieval** 的设计，在保持高效的同时实现了卓越的长上下文建模能力与外推鲁棒性，且被理论证明为精确梯度优化。

</details>

---

### 8. [Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference](https://arxiv.org/abs/2605.11581)

**Authors**: Wenxin Dong, Mingqing Hu, Guanghui Yu, Qiang Fu, Peng Xu, Hui Xu, Yue Xing, Xuewu Jiao, Shuanglong Li, Lin Liu  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.11581v1  

#### Abstract
When large language models (LLMs) serve real-time inference in commercial online advertising systems, end-to-end latency must be strictly bounded to the millisecond range. Yet every token generated during the decode phase triggers thousands of kernel launches, and kernel launch overhead alone can ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在商业在线广告系统中，大语言模型（LLM）推理需要满足**严格的端到端延迟约束（毫秒级）**。然而，在解码阶段（Decode），每个token生成会触发数千次 Kernel Launch，导致 **Kernel Launch Overhead 占据端到端推理时间的约14.6%**。此外，频繁的 HBM 访问也带来显著延迟。

传统 MegaKernel 虽能通过算子融合消除 Launch 开销，但在资源受限的 GPU 架构（如 NVIDIA Ada L20）上面临两大挑战：
- **共享内存（Shared Memory）严重不足**（仅128KB，约为 H100 的一半），限制了流水线深度；
- 手动调优方案缺乏可移植性，而自动编译方案引入运行时分支判断，影响指令发射效率。

### 提出的新方法与创新思路
作者提出 **Ada-MK**，一种面向资源受限 GPU 的自适应 MegaKernel 优化框架，其三大核心贡献如下：

#### （1）自适应共享内存管理（Adaptive Shared Memory Management）
- 构建了一个**三维共享内存约束模型**，综合考虑硬件规格、模型架构和动态工作负载；
- 引入 **K-dimension 细粒度分裂**，将每轮迭代所需权重子块加载，**峰值共享内存需求降低50%**；
- 实现跨算子的**共享内存页复用机制**，包括：
  - **Activation-Weight Page Reuse**：激活数据加载进寄存器后，释放其共享内存用于权重存储；
  - **Activation-Output Page Reuse**：激活内存空间复用于 MMA 输出缓存。

#### （2）基于 DAG 的细粒度自动搜索（Fine-grained DAG-based Automatic Search）
- 利用 **MLIR Lowering 技术** 将高级 IR 分解为 PTX 级依赖图（DAG），实现更精细的并行机会挖掘；
- 在离线阶段进行 **DAG 级搜索与最优执行路径固化（Execution Path Solidification）**，完全消除运行时动态决策开销；
- 支持 Load Balancing、Tiling 参数探索、Gap Filling、Address Permutation 等调度策略。

#### （3）异构混合推理引擎（Heterogeneous Hybrid Inference Engine）
- 将 MegaKernel 作为插件嵌入 **TensorRT-LLM**，构建混合执行模式：
  - **Prefill 阶段**：使用 TensorRT-LLM 原生高性能融合算子，保证高吞吐；
  - **Decode 阶段**：切换至 MegaKernel 引擎，实现低延迟、免 Launch Overhead 推理；
- 复用 TensorRT-LLM 已有业务能力（如 prefix-tree decoding），避免工程迁移成本。

### 相比现有方法的优势
| 方法 | 局限性 | Ada-MK 的改进 |
|------|--------|----------------|
| Stanford MegaKernel [9] | 仅支持 Hopper/Blackwell 架构，硬编码特定模型（如 Llama-1B），不支持 Qwen；无 Prefill 支持 | 支持 Ada 架构，通用性强，适配多种 Qwen 模型 |
| Mirage MPK [1] | 运行时通过 `if-else` 动态判断内存状态，引入分支惩罚 | **离线搜索 + 路径固化**，彻底消除运行时分支 |
| vLLM / SGLang | 未实现算子级深度融合，Launch Overhead 显著 | 在 Decode 阶段实现全链路融合，显著降低延迟 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **固定短序列任务**：输入长度 64 tokens，输出长度 12 tokens（`in64/out12`），模拟低延迟短文本生成场景；
- **真实任务数据集**：
  - **CSL 数据集**：中等上下文长度（~200–1000 tokens），反映学术文献生成任务；
  - **Human-eval 数据集**：代码生成任务，测试复杂逻辑下的性能稳定性。

### 实验设置
- **硬件平台**：单台服务器，配备 **NVIDIA L20 GPU**（Ada 架构，48GB GDDR6，SMem=128KB）；
- **软件环境**：Linux 5.10, CUDA 12.2, Docker 隔离容器；
- **测试模式**：**离线批处理模式（offline batch mode）**，控制并发请求数量，排除调度器干扰；
- **批量大小（Batch Size）**：1, 2, 4, 8, 16；
- **量化配置**：GPTQ-W4A16（4-bit 权重，16-bit 激活）；
- **测试模型**：
  - Qwen3-1.7B
  - Qwen2.5-1.5B

### 评估指标
- **生成吞吐量（Generation Throughput）**：单位为 **tokens/s**，越高越好；
- 对比指标：相对于 baseline 的 **Speedup Ratio**。

### 基线方法对比
| 框架 | 版本 | 特点 |
|------|------|------|
| **vLLM** [13] | v0.19.0 | 高吞吐，高效 KV Cache 管理 |
| **SGLang** [33] | v0.5.10 | 结构化生成优化，高性能服务 |
| **vanilla TensorRT-LLM** [21] | v1.1.0rc5 | NVIDIA 官方推理框架，Baseline |
| **Ada-MK (Ours)** | —— | TensorRT-LLM + MegaKernel 插件 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总
| 场景 | 最佳表现 | 相对提升 |
|------|---------|----------|
| 固定短序列 (`in64/out12`)，BS=1，Qwen3-1.7B | **Ada-MK 达 3.5K tokens/s** | 相比 TRT-LLM ↑23.6%，相比 vLLM ↑50.2% |
| CSL 数据集，BS=8 | Ada-MK 吞吐最高 | 相比 TRT-LLM ↑11.8% |
| Human-eval 数据集，BS=16 | Ada-MK 仍保持领先 | 相比 TRT-LLM ↑19.5%，略超 vLLM 1.6% |

### 与基线方法的对比结果
#### （1）在所有场景下均优于 vanilla TensorRT-LLM
- **最小增益**：4.0%（长序列高 batch）
- **最大增益**：**23.6%**（短序列小 batch）
- 所有测试配置中均取得正向收益，验证了优化的普适性。

#### （2）在小批量、短序列场景优势最显著
- 在 `in64/out12`, BS=1 下：
  - 相比 **vLLM** 提升 **50.2%**
  - 相比 **SGLang** 提升 **71.9%**
- 表明 Ada-MK 特别适合 **低延迟、交互式推理任务**。

#### （3）在高并发长上下文场景下优势收窄
- 在 CSL 数据集，BS=16 时，**vLLM 反超 Ada-MK 3.5%**
- 原因分析：vLLM/SGLang 在 **请求调度、KV Cache 共享、并行扩展性** 上具备系统级优势，弥补了算子融合的不足。

#### （4）跨模型一致性验证
- 在 Qwen3-1.7B 和 Qwen2.5-1.5B 上均观察到一致趋势：
  - Ada-MK 在所有 batch size 下均取得最高吞吐；
  - 增益范围稳定在 6.7%–15.6%（相对 TRT-LLM）；
- 说明优化不依赖特定参数量或模型版本，具有良好的泛化能力。

### 消融实验结果（隐含于文中分析）
虽然未单独列出消融表格，但文中通过模块化分析揭示各组件贡献：
- **K-dimension splitting + Page Reuse**：共享内存峰值下降 50%，使流水线从 2 stage 提升至 4 stage；
- **DAG 级搜索优化**：
  - 消除伪依赖（如 RMS Norm 加载提前）；
  - 实现跨路径流式 Reduce（SwiGLU 中 Up/Gate 并行处理）；
- **Warp 分配优化**：Consumer Warps 从 16 减至 8，配合 stage 扩展至 4，有效缓解角色间吞吐失配，减少停顿。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **MegaKernel 可成功部署于资源受限的 Ada 架构 GPU**，通过自适应共享内存管理和离线 DAG 搜索克服硬件瓶颈；
2. ✅ **“离线搜索 + 路径固化”范式** 可彻底消除运行时分支开销，实现确定性低延迟推理；
3. ✅ **异构混合引擎设计** 成功结合 TensorRT-LLM 的 Prefill 高吞吐与 MegaKernel 的 Decode 低延迟，兼顾性能与工程可行性；
4. ✅ 在 **小批量、短序列、低延迟场景** 下，Ada-MK 显著优于主流框架（最高提升 50.2% @ vLLM）；
5. ✅ 优化效果具有**跨模型一致性**，适用于不同规模的 Qwen 系列模型。

### 方法的局限性
- **在高 batch size + 长序列场景下优势减弱**：系统级调度（如 vLLM 的 PagedAttention）可能超越算子级融合带来的收益；
- **当前主要聚焦 Decode 阶段**：Prefill 阶段尚未启用 MegaKernel，仍有进一步融合空间；
- **依赖 MLIR 编译基础设施**：对编译栈的支持要求较高，可能增加部署复杂性；
- **目前仅支持 GPTQ 量化**，其他量化格式需额外适配。

### 未来工作方向
- 探索 Ada-MK 向更大规模模型（如 10B+）的迁移与适配；
- 尝试将 MegaKernel 应用于 **Prefill 阶段**，实现全流程融合；
- 支持下一代 **Blackwell 架构 GPU**，利用其更强的硬件特性（如更大的 SMem、原生 TMA）；
- 扩展量化支持（如 AWQ、Marlin），提升兼容性与灵活性。

---

> 📌 **总结一句话**：  
> **Ada-MK 是首个在商业在线广告系统中成功落地的 MegaKernel 方案，通过“三维共享内存建模 + DAG 级离线搜索 + 异构混合引擎”，在 NVIDIA Ada 架构上实现了低延迟、高吞吐的 LLM 推理突破，尤其在小批量短文本场景下性能遥遥领先。**

</details>

---

### 9. [EMO: Frustratingly Easy Progressive Training of Extendable MoE](https://arxiv.org/abs/2605.13247)

**Authors**: Linghao Jin, Chufan Shi, Huijuan Wang, Nuan Wen, Zhengzhong Liu, Eric Xing, Xuezhe Ma  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.13247v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) models offer a powerful way to scale model size without increasing compute, as per-token FLOPs depend only on k active experts rather than the total pool of E experts. Yet, this asymmetry creates an MoE efficiency paradox in practice: adding more experts balloons memo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**EMO: Frustratingly Easy Progressive Training of Extendable MoE**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题：**MoE 效率悖论（MoE Efficiency Paradox）**

尽管 **Sparse Mixture-of-Experts (MoE)** 在理论上实现了参数增长与 per-token FLOPs 的解耦（即计算量仅依赖激活专家数 $k$，而非总专家数 $E$），但在实际训练中，随着 $E$ 增大，以下系统开销显著上升：

- **All-to-all 通信成本**
- **Optimizer state 内存占用**
- **小规模 GEMM 导致 GPU 利用率低**

这导致 **wall-clock time 随 $E$ 增长而增加**，违背了“高效扩展”的初衷 —— 即所谓的 **MoE 效率悖论**。

> 如图1所示，在固定激活参数下，将 $E$ 从8增至128，理论 FLOPs 不变，但 step time 上升达 **1.72×**。

---

### 🚀 提出的新方法：**EMO（Extendable Mixture-of-Experts）**

提出一种**渐进式 MoE 扩展训练框架**，其核心思想是：

> 将 MoE 容量视为可扩展的**参数化内存（parametric memory）**，在训练过程中逐步扩大专家池 $E$。

#### 主要创新点：

1. **多阶段渐进扩展（Progressive Expansion）**
   - 起始于小型 MoE 或稠密模型（如 $E=8$）
   - 分多个阶段逐步扩展至大 MoE（如最终 $E=128$）
   - 每次扩展后继续训练新增数据

2. **基于稀疏性感知的 Token 分配策略（Sparsity-aware Token Allocation）**
   - 引入统一 MoE 缩放定律（Unified MoE Scaling Law, Ludziejewski et al., 2025）
   - 显式建模专家数量 $E$ 对损失函数的影响：
     $$
     L(N_{\text{act}}, E, D) = m(E) N_{\text{act}}^{-p(E)} + n(E) D^{-v(E)} + c
     $$
   - 推导出各阶段最优 token 预算 $d_s$，实现 compute-optimal 数据分配

3. **无需复杂架构修改或辅助目标**
   - 仅需标准 MoE 层 + 路由机制
   - 不依赖特殊路由设计或 load balancing loss

---

### 🔍 相比现有方法的优势

| 方面 | 现有方法 | EMO |
|------|--------|-----|
| **训练效率** | 固定 $E$ 全程高开销 | 早期小 $E$ 快速训练，后期才引入大 $E$ |
| **资源利用率** | 初始即承担全量通信/内存压力 | 渐进承担，降低峰值资源需求 |
| **性能保留** | — | 最终性能接近 fixed-large-$E$ 基线 |
| **灵活性** | 固定结构 | 支持任意扩展路径（如 8→16→32→64→128） |

> ✅ **核心优势**：以更低的 GPU 小时数和 wall-clock 时间，达到与固定大规模 MoE 相当的性能。

---

## 2. 核心实验方法和设置

### 📚 数据集

- **预训练数据**：混合语料，包含
  - Web 文本（Slimpajama, C4, Dolma）
  - Code（GitHub）
  - 数学（MathPile）
  - 多语言文本
  - 社交媒体（Reddit, Twitter）
- **总 token 预算**：**1.92T tokens**
- **上下文长度**：8192

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| 模型架构 | Decoder-only Transformer |
| 总层数 | 16 |
| Hidden dim | 2048 |
| MoE 层 | 每层含共享专家 + 路由专家池 |
| Top-k | 8 |
| 专家隐藏维度 | 768 |
| Router | Sigmoid score + bias + entropy-based load balancing |
| Batch size | 8M tokens |
| Optimizer | AdamW ($\beta_1=0.9$, $\beta_2=0.95$) |
| Peak LR | $9 \times 10^{-4}$ |
| LR Schedule | Warm-Stable-Decay（前90%稳定，后10%线性衰减） |
| 硬件 | 32×NVIDIA H200 GPUs |

### 🔄 EMO 扩展策略（五阶段）

| 阶段 | 专家数 $E$ | Token 分配比例（来自缩放律拟合） |
|------|------------|-------------------------------|
| Stage 1 | 8 | 23.5% |
| Stage 2 | 16 | 9.5% |
| Stage 3 | 32 | 16.5% |
| Stage 4 | 64 | 21.8% |
| Stage 5 | 128 | 28.6% |

> 每次扩展时：
> - 新专家 Gaussian 初始化
> - Router bias 重置为0
> - Optimizer states 重置
> - 学习率 warmup 500 步

---

### 📊 评估指标

- **上游任务**：
  - Pretraining loss（验证集）
  - Validation perplexity（多个 domain-specific splits）
- **下游任务**（共8个基准）：
  - **知识类**：MMLU, TriviaQA, NQ
  - **推理类**：GSM8K
  - **常识类**：HellaSwag, PIQA, SIQA, Winograd
  - **多选题**：ARC-C/E, COPA, RACE
- **效率指标**：
  - Wall-clock time
  - GPU hours
  - Communication overhead

---

### 🆚 基线方法对比

| 基线名称 | 描述 |
|---------|------|
| **FIXED_E=16** | 固定 16 专家，全程训练 |
| **FIXED_E=32** | 固定 32 专家 |
| **FIXED_E=128** | 固定 128 专家（上限基线） |

所有基线使用相同超参、数据、token budget，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 模型 | Final Pretrain Loss | GPU Hours 节省 | 下游综合表现 |
|------|---------------------|----------------|--------------|
| **FIXED_E=128** | 0.994 | 基准 | 最优（部分任务领先） |
| **EMO (progressive)** | **1.017** | **节省 10%** | **接近 FIXED_E=128，显著优于 FIXED_E=32/16** |
| **FIXED_E=64** | 1.065 | — | 明显落后 |

> 💡 尽管 EMO 最终 loss 略高于 FIXED_E=128（相对差距 2.3%），但其训练成本显著更低。

---

### 📊 下游任务表现（见 Table 2）

| 任务 | EMO vs FIXED_E=128 | EMO vs FIXED_E=32 |
|------|--------------------|-------------------|
| **MMLU** | 略低（56.34 vs 57.40） | 显著胜出 |
| **GSM8K** | 落后（57.09 vs 64.73） | 显著胜出 |
| **HellaSwag / ARC / PIQA** | 接近持平 | 明确胜出 |
| **TriviaQA / NQ** | 明显优于 FIXED_E=32，接近 FIXED_E=128 | — |

> ✅ **总体趋势**：EMO 在大多数任务上**媲美 FIXED_E=128**，且**全面超越小规模固定 MoE**。

---

### 🔬 消融实验结果

#### （1）**扩展时机验证（Expanding at 25%, 50%, 75%）**

- **早扩（25%）**：质量最高（loss=1.069），但 wall-clock 成本高
- **晚扩（75%）**：速度快，但质量下降明显（loss=1.076）
- **EMO 推荐点（~45%）**：位于“平坦区”，平衡质量与效率

> ✅ 结论：缩放律预测的分配策略处于**最佳权衡区域**

#### （2）**初始化策略对比**

| 初始化方式 | 特点 |
|----------|------|
| Gaussian init | 简单，收敛快 |
| Bias reset | spike 更大，但不影响最终性能 |
| Copy from old ckpt | spike 最大，恢复最快 |

> ✅ **发现**：EMO 对初始化鲁棒，不同策略均能收敛到相似 loss，仅影响 transient spike 大小。

#### （3）**Optimizer State 处理**

- 保留旧状态 vs 重置：差异在约 500 步内消失
- 最终选择**重置**以简化实现

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **MoE 容量不必从一开始就拉满**
   - 早期数据不足以利用大量专家
   - 小 $E$ 阶段更高效地构建基础表示

2. **渐进扩展可逼近固定大 MoE 性能**
   - EMO 达到与 FIXED_E=128 相当的 loss 和 downstream 表现
   - 同时节省 **10% GPU hours**

3. **稀疏性感知缩放律可用于指导训练调度**
   - 可预测每阶段应分配多少数据
   - 自动实现 compute-optimal token 分配

4. **EMO 具有良好的稳定性与鲁棒性**
   - 扩展后 loss spike 可在 ~10K 步内恢复
   - 对初始化、optimizer 处理不敏感

5. **专家利用均衡性良好**
   - Gini coefficient ≈ 0.5（baseline 为 0.44），无 collapse
   - 中间层略不平衡，但整体分布合理

---

### ⚠️ 方法的局限性

1. **缩放律未显式建模优化器超参**
   - 当前拟合未考虑 LR、batch size 等动态变化的影响
   - 若训练策略非固定，可能需重新校准

2. **尚未应用于最前沿超大规模 MoE**
   - 实验最大为 9.6B total / 1.1B activated
   - 在 trillion-parameter 级别是否仍有效待验证

3. **某些任务仍偏好全程大专家池**
   - 如 **GSM8K** 上 FIXED_E=128 明显更强
   - 推测：数学推理能力需要跨专家长期协同演化

---

### 🔮 未来工作方向

1. **将 EMO 扩展至更大规模模型**
   - 应用于 >100B 参数 MoE 系统
   - 验证其在极致 scale 下的性价比优势

2. **动态调整扩展策略**
   - 基于在线 loss 曲线或数据难度自适应决定何时扩展

3. **结合 upcycling 与 EMO**
   - 先 dense pretrain → upcycle to MoE → progressive expand
   - 进一步压缩训练成本

4. **探索非均匀扩展策略**
   - 不同层采用不同扩展节奏（如浅层早扩，深层晚扩）

---

## ✅ 总结

**EMO** 提出了一种“简单却极其有效”的 MoE 渐进训练范式：

> “与其一开始就把所有专家都拉上来承受高昂开销，不如像搭积木一样，边训边加。”

它通过 **sparsity-aware scaling law 指导下的渐进扩展机制**，成功缓解了 MoE 效率悖论，在几乎不牺牲性能的前提下，显著降低了训练时间和 GPU 成本。该方法简洁、通用、易于集成，为未来大规模 MoE 训练提供了新的工程实践路径。

</details>

---

### 10. [SOMA: Efficient Multi-turn LLM Serving via Small Language Model](https://arxiv.org/abs/2605.11317)

**Authors**: Xueqi Cheng, Qiong Wu, Zhengyi Zhou, Xugui Zhou, Tyler Derr, Yushun Dong  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.11317v1  

#### Abstract
Large Language Models (LLMs) are increasingly deployed in multi-turn dialogue settings where preserving conversational context across turns is essential. A standard serving practice concatenates the full dialogue history at every turn, which reliably maintains coherence but incurs substantial cost i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SOMA: Efficient Multi-turn LLM Serving via Small Language Model

## 1. 论文的主要贡献和创新点

### 解决的问题
在多轮对话场景中，传统的 **Large Language Model (LLM)** 服务通常需要将完整的对话历史拼接并传入模型以生成回复。这种做法虽然能保持上下文连贯性，但随着对话轮次增加，会导致：
- **高延迟**（latency）
- **高内存消耗**
- **高昂的API调用成本**

现有方法难以在响应质量与推理效率之间取得良好平衡。

---

### 提出的新方法：SOMA
作者提出 **SOMA (Soft-prompts for lOcal Manifold Approximation)**，一种高效的多轮LLM服务框架，其核心思想是：
> 利用对话早期轮次建立“局部推理流形”（local reasoning manifold），然后训练一个小型语言模型（surrogate small language model）来近似大模型在此局部区域的行为，后续轮次由小模型接管。

#### 三阶段流程：
1. **Soft Prompt Tuning（软提示挖掘）**
   - 在小模型上学习一组 soft prompts，最大化其与大模型输出之间的语义差异。
   - 引入 **expectation-weighted semantic divergence loss** 和 **anti-degeneration regularizer** 来稳定训练并防止退化。

2. **Localized Fine-tuning（局部微调）**
   - 将挖掘出的“最难对齐”的样本用于对小模型进行 **LoRA 微调**，使其在当前对话的局部上下文中逼近大模型行为。
   - 微调后丢弃 soft prompts，实现无提示推理。

3. **Efficiency Inference（高效推理 + 回滚机制）**
   - 使用一个简单的 **cosine gate** 决定是否切换到小模型。
   - 在线监控语义漂移（semantic drift），一旦检测到话题偏移，则回滚至原始大模型重新初始化状态。

---

### 相比现有方法的优势
| 方法类型 | 缺陷 | SOMA 的改进 |
|--------|------|-------------|
| 单模型压缩/摘要 | 仍依赖大模型每轮推理，成本高 | 后续轮次完全由小模型处理 |
| 多模型路由（如 RouteLLM） | 小模型泛化能力差，切换开销大 | 动态适配小模型至当前上下文，提升一致性 |
| 注意力复用/缓存 | 需要底层支持，且可能截断长程依赖 | 不依赖特定架构，通用性强 |

✅ **核心优势**：  
- 显著降低 token 成本和延迟；
- 保持高质量的上下文感知能力；
- 支持自动回滚，保障服务质量。

---

## 2. 核心实验方法和设置

### 数据集
在六个多轮对话基准上进行评估：
- **ShareGPT**：开放域人机聊天
- **ReMeDi**：医生-患者医疗咨询
- **Craigslist**：买卖双方谈判
- **Multi-Char**：多角色扮演对话
- **MATH**：数学推理题（带逐步解答）
- **MT-Bench**：多任务质量评测

> 所有数据均经过过滤，仅保留 **context-dependent** 对话（即后续轮次强依赖前期上下文）。

---

### 实验设置
#### 模型配置
测试两个主流模型族：
| 模型家族 | 大模型 F | 小模型 G |
|---------|----------|----------|
| LLaMA | LLaMA-3.1-70B | LLaMA-2-7B |
| Qwen | Qwen-3-8B | Qwen-3-0.6B |

#### 基线方法对比
- **Original**：始终使用大模型 + 完整历史
- **Surrogate**：始终使用小模型 + 完整历史
- **History-Prefix**：小模型输入完整历史，不微调
- **History-FT**：基于历史数据对小模型做 LoRA 微调
- **LLMLingua-2**：压缩历史后再输入小模型
- **RouteLLM**：动态路由选择大小模型
- **Random-FT**：随机选取局部样本微调（用于消融）

---

### 评估指标
| 类别 | 指标 |
|------|------|
| **响应保真度** | Response similarity（通过 GPT-OSS、DeepSeek-V3、Gemma2-27B 作为 judge 打分） |
| **任务质量** | MATH 数据集上的 Exact Match (EM) 准确率 |
| **效率指标** | 总 token 数、吞吐量（throughput）、端到端延迟 |
| **可靠性分析** | 漂移检测准确率、误回滚率（false rollback） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 表 1：LLaMA 家族下的响应相似度（越高越好）
| 方法 | Avg. Similarity |
|------|----------------|
| Surrogate | 75.1 ± 5.98 |
| History-Prefix | 84.8 ± 2.94 |
| History-FT | 90.8 ± 2.18 |
| RouteLLM | 92.2 ± 1.78 |
| **SOMA** | **93.1 ± 1.99** ✅ |

> SOMA 在所有六项数据集中均取得最高相似度，尤其在 MATH 和 Multi-Char 上增益显著。

#### ✅ 表 2：MATH 数据集 EM 准确率
| 方法 | LLaMA 家族 | Qwen 家族 |
|------|------------|-----------|
| Original | 48.34 | 36.48 |
| Surrogate | 19.20 | 11.73 |
| History-FT | 31.46 | 22.57 |
| RouteLLM | 33.88 | 25.08 |
| **SOMA** | **41.62** ✅ | **31.14** ✅ |

> SOMA 接近原始大模型表现，远超其他小模型方案，说明其不仅模仿表面输出，也保留了深层推理能力。

---

### 与基线方法的对比结果
- **相比 Surrogate**：平均提升约 **18个百分点** 的响应相似度。
- **相比 History-FT**：通过智能采样 hard cases，进一步提升约 **2.3 pts**。
- **相比 RouteLLM**：在相似度和任务准确率上全面超越。
- **相比 LLMLingua-2**：虽压缩更激进，但 SOMA 更注重语义对齐而非单纯删减。

---

### 消融实验结果（Ablation Study）

#### 图 2b：组件重要性分析（LLaMA 家族）
- 移除 **anti-degeneration loss** → 性能下降明显（prompt collapse 风险）
- 移除 **expectation-weighted divergence** → 对语义影子（semantic shadowing）捕捉不足
- 同时移除两者 → 下降幅度最大，验证联合设计的有效性

> 结论：**anti-degeneration 正则项** 和 **期望加权损失** 是稳定 prompt mining 的关键。

#### 表 4b：漂移回滚机制有效性
| 数据集 | 无回滚（No RB） | 使用 SOMA 回滚 |
|-------|----------------|----------------|
| ShareGPT | 88.5 ± 2.03 | **94.1 ± 1.47** |
| ReMeDi | 85.7 ± 1.92 | **91.1 ± 1.34** |
| Craigslist | 83.1 ± 2.11 | **89.3 ± 1.68** |

> 回滚机制可恢复大部分因话题跳跃导致的质量损失，且误触发率 < 5%。

---

## 4. 关键结论和发现

### 主要发现
1. **多轮对话存在“长尾模式”**：
   - 前几轮承载大量上下文信息（token 多），后续轮次变短但仍高度依赖前期状态。
   - 这为“前段用大模型建模，后段用小模型代理”提供了理论基础。

2. **局部流形近似可行**：
   - 小模型可通过 soft prompt 挖掘 + LoRA 微调，在局部上下文中有效逼近大模型行为。

3. **效率收益可摊销**：
   - 虽然引入 warm-up 开销（soft prompt probing + LoRA training），但在中长对话中（>8轮）即可实现净节省。
   - 如 Table 3 所示，在 13+ 轮对话中 token 节省达 **29.1%~37.2%**。

4. **自适应切换优于固定策略**：
   - 基于语义门控的切换机制能有效识别何时可以安全使用小模型。

---

### 方法的局限性
1. **小模型容量限制**：
   - 当大小模型差距过大时（如 Qwen-3-8B vs Qwen-3-0.6B），局部拟合能力受限。

2. **需访问嵌入空间**：
   - soft prompt tuning 要求能操作小模型的 embedding 层，在纯黑盒 API 场景下不可行。

3. **适用于局部连贯对话**：
   - 若对话频繁跳题或 paraphrase 差异大，局部流形假设失效，需频繁回滚。

4. **一次性启动成本**：
   - 不适合极短对话（<5轮），无法覆盖 warm-start 开销。

---

### 未来工作方向
- 设计更强的 **drift detector** 提高回滚鲁棒性；
- 探索 **approximate mining 方法**，减少对内部参数的依赖；
- 支持 **multi-region adaptation**，应对多主题对话；
- 引入 **privacy-preserving mining**，保护敏感对话数据；
- 将框架扩展至 **multimodal serving** 场景。

---

> 🔗 **开源代码地址**：[https://github.com/LabRAI/SOMA](https://github.com/LabRAI/SOMA)

</details>

---

### 11. [F-GRPO: Factorized Group-Relative Policy Optimization for Unified Candidate Generation and Ranking](https://arxiv.org/abs/2605.12995)

**Authors**: Rohan Surana, Gagan Mundada, Junda Wu, Xintong Li, Yizhu Jiao, Bowen Jin, Sizhe Zhou, Tong Yu, Ritwik Sinha, Jiawei Han, Jingbo Shang, Julian McAuley  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.12995v1  

#### Abstract
Traditional retrieval pipelines optimize utility through stages of candidate retrieval and reranking, where ranking operates over a predefined candidate set. Large Language Models (LLMs) broaden this into a generative process: given a candidate pool, an LLM can generate a subset and order it within ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：F-GRPO: Factorized Group-Relative Policy Optimization for Unified Candidate Generation and Ranking

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

传统检索系统通常采用多阶段流水线（multi-stage pipeline），先进行**候选生成**（candidate retrieval），再对固定候选集进行**重排序**（reranking）。这种解耦方式存在两个核心缺陷：

- **分布不匹配**（Distribution Mismatch）：排名器（ranker）在训练时看到的是“黄金”候选集，但在推理时却必须处理由生成器产生的非完美候选集，导致性能下降。
- **信用分配模糊**（Credit Assignment Gap）：当使用端到端的 LLM 进行生成并排序时，单一的序列级奖励（sequence-level reward）无法区分是“没生成好”还是“没排好”，导致优化不稳定且样本效率低。

此外，现有基于 LLM 的方法要么将 LLM 视为黑箱重排序器（black-box reranker），要么将生成与排序分离成多个模块，均未能实现真正统一的联合优化。

### 提出了什么新方法或新思路

本文提出 **F-GRPO**（**Factorized Group-Relative Policy Optimization**），一种用于统一候选生成与排序的端到端强化学习框架，其核心创新如下：

- **统一的两阶段生成流程**：在一个 autoregressive rollout 中，模型首先生成候选列表（`<SLATE>` 阶段），然后对这些候选进行排序（`<RANK>` 阶段），整个过程由同一个 LLM 完成。
  
- **因子化策略建模**（Factorized Policy）：
  $$
  \pi(\tau, \sigma|x) = \pi_{\text{slate}}(\tau|x) \cdot \pi_{\text{rank}}(\sigma|x,\tau)
  $$
  将策略分解为 slate 生成和 ranking 两个子策略，共享 LLM 主干，但目标不同。

- **相位特定的奖励机制**：
  - `Rslate`：顺序无关的覆盖奖励（如 Recall 或 F1），衡量是否生成了相关项。
  - `Rrank`：位置敏感的效用奖励（如 NDCG@k），衡量排序质量。

- **因子化信用分配**（Factorized Credit Assignment）：
  - 分别计算 `slate` 和 `rank` 阶段的 **group-relative advantage**：
    $$
    A^{(i)}_{\text{slate}} = R^{(i)}_{\text{slate}} - \bar{R}_{\text{slate}}, \quad A^{(i)}_{\text{rank}} = R^{(i)}_{\text{rank}} - \bar{R}_{\text{rank}}
    $$
  - 在反向传播中，每个阶段仅使用对应的优势信号更新参数，避免梯度污染。

### 相比现有方法的优势

| 对比维度 | F-GRPO | 传统解耦方法 | 单一奖励 GRPO |
|--------|-------|------------|-------------|
| 架构复杂度 | 单一 LLM | 多个模型（generator + ranker） | 单一 LLM |
| 推理一致性 | ✅ 无分布偏移 | ❌ 存在分布偏移 | ✅ 一致 |
| 信用分配 | ✅ 明确分离生成与排序责任 | N/A（监督训练） | ❌ 模糊混杂 |
| 优化稳定性 | ✅ 更高 | 取决于训练方式 | ❌ 较低 |

F-GRPO 实现了**架构简洁性**、**训练有效性**与**推理鲁棒性**的统一。

---

## 2. 核心实验方法和设置

### 使用的数据集

| 任务 | 数据集 | 描述 |
|-----|--------|------|
| **Sequential Recommendation** | MovieLens, LastFM | 用户历史 → 预测下一个可能喜欢的 item（从 20 个候选中选 1 个正例） |
| **Multi-hop Question Answering** | HotpotQA, MuSiQue | 多跳问题 → 从 20 个 passage 中选出 2–4 个支持证据并排序 |

所有任务均构造为从固定候选池中选择并排序的问题。

### 实验设置和评估指标

- **模型**：
  - Qwen3-4B-Instruct-2507
  - Qwen3.5-2B
- **训练流程**：
  - 所有 RL 方法均以 SFT 模型为起点。
  - 使用 Dr. GRPO 设置，每轮采样 G=8 个 rollout。
  - 推理时使用 greedy decoding（temperature=0）。
- **评估指标**：
  - **Recall@k**, **Precision@k**, **Hit@k**
  - **NDCG@k**（Normalized Discounted Cumulative Gain）
- **输出格式控制**：
  - 使用 `<SLATE>` 和 `<RANK>` XML 标签明确分隔两个阶段。
  - 最大 slate 大小为 10，最大 rank 输出为 5。

### 基线方法对比

| 类型 | 基线方法 | 说明 |
|------|----------|------|
| **传统模型** | Random, Popularity, BM25, GRU4Rec, UniSRec | 经典推荐/检索模型 |
| **零样本 LLM** | Zero-shot (two-step), Zero-shot (single-step) | 不经微调直接提示 |
| **监督微调** | SFT, Decoupled SFT | 监督训练生成或排序行为 |
| **强化学习基线** | GRPO | 同样使用 rollout，但使用单一联合奖励 $ R_{\text{joint}} = R_{\text{slate}} + \lambda R_{\text{rank}} $ |

其中 **Decoupled SFT** 是最强的竞争者之一，它分别训练一个 selector 和一个 ranker，并在推理时串联调用。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1 & 2）

#### ✅ **Sequential Recommendation (LastFM, Qwen3-4B)**

| Method | Recall@3 | Recall@5 | NDCG@3 | NDCG@5 |
|--------|----------|----------|--------|--------|
| SFT | 47.1 | 53.8 | 44.0 | 46.8 |
| GRPO | 55.1 | 73.9 | 50.8 | 58.6 |
| **F-GRPO (Ours)** | **72.4** | **81.7** | **61.7** | **65.5** |

> ➤ 相比 GRPO，Recall@3 提升 **+31.4%**，Recall@5 提升 **+10.6%**

#### ✅ **Multi-hop QA (MuSiQue, Qwen3-4B)**

| Method | Recall@3 | Recall@5 | NDCG@3 | NDCG@5 |
|--------|----------|----------|--------|--------|--------|
| GRPO | 63.0 | 63.0 | 71.3 | 69.2 |
| **F-GRPO (Ours)** | **71.3** | **71.8** | **78.5** | **76.5** |

> ➤ Recall@3 提升 **+13.2%**，表明在覆盖成为瓶颈的任务中优势更明显。

### 与基线方法的对比结果

- **显著优于所有监督方法**（SFT、Decoupled SFT）：
  - Decoupled SFT 虽然有两个独立模型，但由于 ranker 训练时依赖“黄金 slate”，实际表现反而不如单模型 F-GRPO。
- **优于标准 GRPO**：
  - 表明**因子化信用分配**确实缓解了梯度干扰问题。
  - 改进在较高 k 值（如 Recall@5）更为显著，说明 slate 覆盖能力增强。
- **媲美甚至超越专用零样本重排序器**（如 RankZephyr, MonoT5）：
  - 尽管这些 reranker 在 MS MARCO 上专门训练，而 F-GRPO 是 in-domain 训练，仍能取得竞争力。

### 消融实验结果（Ablation Studies）

#### （1）奖励权重 $\lambda$ 敏感性分析（Fig 4a）

- 设定 $\lambda = 1.0$ 时性能最优。
- 若 $\lambda = 0.5$（削弱 ranking 损失），则 NDCG@1 和 Recall@1 显著下降，说明 ranking 需要足够强的学习信号。

#### （2）slate 大小 $n$ 影响（Fig 4b）

- $n=10$ 时达到最佳平衡：
  - $n=5$：限制了 coverage，Recall 下降；
  - $n=15$：引入过多噪声项，Precision 和 NDCG 受损。

#### （3）训练动态分析（Fig 2b）

- **slate generator 先收敛**，约在第 150 步达到峰值 recall 的 90%；
- **ranker 后收敛**，需等到第 200 步才稳定；
- 验证了条件依赖结构的有效性：ranker 必须等待 slate generator 提供有效输入才能学习排序。

---

## 4. 关键结论和发现

### 主要发现

1. **因子化信用分配至关重要**：
   - 单一奖励会导致“排序好就奖励生成”、“生成差也惩罚排序”的错误梯度方向。
   - F-GRPO 通过分离优势信号，实现了**第一阶可分离性**（first-order separability），使两个阶段各司其职。

2. **端到端联合训练优于解耦流水线**：
   - 尽管 decoupled 方法直观合理，但因训练与推理间的分布差异，实际效果受限。
   - F-GRPO 中的 ranker 在训练时就看到 generator 的输出，自然适应其行为模式。

3. **coverage 是性能瓶颈的关键因素**：
   - 在 MuSiQue 等需要广泛检索的任务中，F-GRPO 提升最大。
   - 分析显示 error 来源均衡分布在 **slate miss**（未生成）和 **rank drop**（未排前）两类，证明两个阶段都需优化。

4. **无需架构修改即可部署**：
   - 推理时只需普通自回归生成，无需额外模块或流程变更，具备良好实用性。

### 方法的局限性

- **依赖高质量候选池**：F-GRPO 假设候选池已给定，未解决 cold-start 或长尾 item 的召回问题。
- **格式错误风险**：若模型未正确输出 `<SLATE>` 或 `<RANK>` 标签，会触发 format penalty，影响训练稳定性（尽管 SFT 后基本可控）。
- **扩展性挑战**：当前 slate size 限制为 10，难以应用于超大规模候选场景。

### 未来工作方向

- 将 F-GRPO 思想扩展至 **generative retrieval** 场景，即直接从语料库中生成文档 ID。
- 引入 **adaptive slate size** 机制，根据 query 动态决定生成多少候选。
- 结合 **retrieval-augmented generation**（RAG），让 slate 生成更具解释性和可控性。
- 探索 **offline RL** 版本，利用人类偏好数据而非显式 reward 函数进行训练。

---

> 📌 **一句话总结**：  
> F-GRPO 通过在单个 LLM rollout 内部实现**因子化的生成-排序策略**与**相位特定的 group-relative 优势估计**，解决了传统方法中的信用分配模糊与分布不匹配问题，在多个 ranking 任务上实现了优于 GRPO、解耦 SFT 和零样本 reranker 的性能，且无需改变推理架构。

</details>

---

### 12. [An Agentic LLM-Based Framework for Population-Scale Mental Health Screening](https://arxiv.org/abs/2605.13046)

**Authors**: Giuliano Lorenzoni, Paulo Alencar, Donald Cowan  
**Category**: cs.AI  
**Published**: 2026-05-14  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.13046v1  

#### Abstract
Mental health disorders affect millions worldwide, and healthcare systems are increasingly overwhelmed by the volume of clinical data generated from electronic records, telemedicine platforms, and population-level screening programs. At the same time, the emergence of novel AI-based approaches in he...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：An Agentic LLM-Based Framework for Population-Scale Mental Health Screening

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前心理健康筛查面临以下挑战：
- **数据量大且非结构化**：来自电子健康记录、远程医疗平台和群体筛查项目的临床文本数据快速增长，传统人工分析难以扩展。
- **模型配置脆弱**：现有的 LLM 和 RAG 系统在超参数选择、检索策略等方面缺乏系统性管理，导致结果不可复现、易退化（regression）。
- **适应性差**：面对不同患者、临床场景或部署环境时，现有方法难以动态调整并保证性能稳定。

该研究旨在构建一个**可信赖、可复现、自适应的 LLM-based 框架**，用于支持大规模人群的心理健康筛查任务（如抑郁检测），同时解决配置管理、成本控制与非回归保障等问题。

---

### 提出的新方法与思路
作者提出了一种 **Agentic LLM-Based Framework**，其核心是将整个 NLP 流水线分解为多个由 **LangChain Agent** 封装的模块化组件，并通过一个中央 **Orchestrator Agent** 进行协调。每个 Agent 负责特定阶段的任务，并遵循明确的策略进行“冻结”（freeze）、跳过或回滚（rollback）。

#### 主要创新点包括：
1. **Agentic Pipeline 架构设计**
   - 将文本分类流水线划分为 8 个独立 Agent 阶段：
     - Preprocessing（嵌入与截断）
     - Similarity Metric（相似度度量）
     - Selection（Top-k / RAG 类型）
     - Diversity（MMR 多样性增强）
     - Post-filters（去重/元数据过滤）
     - Data Expansion（训练集扩展）
     - Threshold Optimization（阈值调优）
     - Decoding & Gold Validation（解码参数优化与最终验证）
   - 每个 Agent 具有明确定义的角色、责任和决策策略。

2. **Proxy-Guided Exploration + Non-Regression Enforcement**
   - 引入多种低成本 **proxy metrics**（代理指标）来快速筛选无效配置，避免昂贵的 gold evaluation（如 GPT-4 判断）资源浪费。
   - 只有通过 proxy 筛选的候选才进入高成本评估。
   - 所有配置一旦被锁定（frozen），后续更改必须证明有显著提升（>0.01–0.02 macro-F1 增益），否则触发 rollback，确保 **non-regression guarantee**。

3. **Configuration Locking Mechanism**
   - 支持增量式配置固化：各阶段逐步验证后“上锁”，防止后期修改破坏已有成果。
   - 实现了对 variability space 的有效探索与控制。

4. **Population-Scale Readiness**
   - 框架设计不依赖 FAISS 等重型向量数据库，可在 CPU 上运行（Intel i7, 16GB RAM），具备良好的可移植性和部署潜力，适用于大规模数字健康基础设施。

---

### 相比现有方法的优势
| 维度 | 传统 RAG / ML 方法 | 本文提出的 Agentic 框架 |
|------|------------------|------------------------|
| **可复现性** | 配置常为硬编码或脚本化，难追踪变化 | 明确的日志与冻结机制保障可审计性 |
| **鲁棒性** | 参数微调可能导致性能下降 | 非回归策略阻止劣化配置覆盖 |
| **成本效率** | 全面搜索耗资巨大 | Proxy 先筛，大幅降低 GPT-4 调用次数 |
| **模块化与可维护性** | 单体流程，耦合性强 | 分离关注点，便于迭代升级 |
| **适应性** | 固定流程，难以应对新需求 | 支持运行时注入新策略，具备演化能力 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **DAIC-WOZ (Distress Analysis Interview Corpus - Wizard-of-Oz)**  
  - 包含 189 场临床访谈的多模态数据（转录文本、音频、视频等）。
  - 以 PHQ-8 问卷为基础标注抑郁症状态（binary label）：
    - `PHQ8_Binary`: 0 表示抑郁，1 表示非抑郁
    - `PHQ8_Score > 10` 视为阳性病例
  - 数据划分为训练、开发与测试集。
  - 注：因隐私限制，原始数据需申请获取，不能直接分发。

---

### 实验设置
- **任务**：基于访谈转录文本的二分类抑郁检测（transcript-based depression detection）
- **框架实现工具**：
  - Python 3.12, PyTorch 2.8
  - HuggingFace Transformers 4.55, Sentence-Transformers 5.1
  - **LangChain + LangGraph** 实现 Agent 编排与调度
- **Embedding 模型**：`e5-base-v2`（SentenceTransformer）
- **LLM Judge**：`gpt-4o-mini`（temperature=0.0, top_p=0.9, n=1）
- **检索方式**：Cosine similarity over NumPy arrays（无需 FAISS）
- **Pipeline 结构**：
  - 离线构建案例库（embedding store）
  - 在线执行 RAG-style 分类流程：
    ```
    Query Transcript → Embedding → Retrieval (cosine ≥ thr 或 fallback TopK) 
                   → Prompt Construction（带示例） 
                   → LLM Judge → Parser → Binary Label
    ```

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 整体准确率 |
| **Macro-F1** | 正负类平衡下的 F1 分数（主指标） |
| **Precision / Recall** | 特别关注 recall（避免漏诊） |
| **Confusion Matrix** | 分析误判模式 |
| **Proxy Metrics** | 用于前置筛选：<br>- Semantic-retrieval（hit@k, coverage）<br>- Statistical-text（长度、停用词比例）<br>- Ranking-stability（Kendall-T）<br>- Confidence heuristics（投票熵）<br>- Cheap LLM judge（mini-judge 一致性） |

---

### 基线方法对比
文中未直接列出与其他完整系统的端到端对比（如传统 ML 模型或标准 RAG），而是采用 **ablation-style 自比较**，即以逐步锁定的配置作为 baseline，检验新增策略是否带来增益。

- **Frozen Baseline Configuration**（来自 Steps 1–6）：
  - Top-k = 5
  - Similarity Threshold T ≈ 0.75
  - Dynamic RAG selection
  - MMR = OFF
  - Post-filters = OFF
  - PRF = OFF
  - Decoding: temp=0.0, top_p=1.0, n=1

所有后续步骤均以此为基准进行 non-regression 检查。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table III）

| 步骤 | 变动参数 | 最佳结果 | 决策 |
|------|--------|---------|-------|
| Steps 1–4 | K ∈ {2,3,5}, Thr ∈ {0.75,0.78,0.82}, RAG 类型, MMR | K=5, T=0.75, dynamic RAG, MMR OFF | ✅ 锁定配置 |
| Step 5 | Post-filters（去重/低置信过滤） | 无增益 | ❌ 保持关闭 |
| Step 6 | PRF（伪相关反馈） | 宏观 F1 下降，不稳定 | ❌ 拒绝启用 |
| Step 7 | 阈值扫描 T ∈ [0.70, 0.80] | T=0.78 时 acc=0.821, macroF1=0.789 < baseline | ❌ 回滚至 T=0.75 |
| Step 8 | 温度（temp）、top_p、采样数（n） | 所有变体 macroF1 均低于 baseline | ❌ 保留默认 |

> 🔑 **最终锁定配置性能（baseline on VAL set）**：
> - **Accuracy**: 0.857
> - **Macro-F1**: 0.825
> - **Recall**: 0.875

---

### 与基线方法的对比结果
- 所有尝试的改进（如开启 MMR、添加 post-filter、使用 PRF、调整 decoding 参数）**均未能超越初始锁定配置**。
- 特别地：
  - T=0.78 虽然在某些 proxy 上表现更好，但在 gold evaluation 中 macro-F1 下降到 0.789，**违反 non-regression policy**，故被拒绝。
  - 各种 decoding 设置（如 temp=0.1, top_p=0.9）也未改善结果，说明当前任务更适合 deterministic 输出。

---

### 消融实验结果
虽然没有显式的“消融表”，但每一步都相当于一次消融分析：

| 组件 | 是否有益？ | 原因 |
|------|-----------|------|
| **Dynamic RAG** | ✅ 是 | 相比静态检索，在 recall 和稳定性上有优势 |
| **MMR 多样性重排序** | ❌ 否 | 导致 minority class recall 下降，macro-F1 不增反降 |
| **Post-filters** | ❌ 否 | 未观察到一致收益，且可能引入偏差 |
| **PRF（伪相关反馈）** | ❌ 否 | 性能波动大，存在泄露风险，proxy 与 gold 不一致 |
| **Threshold > 0.75** | ❌ 否 | 提高阈值减少召回，损害敏感性 |
| **Stochastic Decoding** | ❌ 否 | 所有随机生成设置均劣于 greedy decoding（temp=0.0） |

这些结果表明：**简单而稳定的配置往往优于复杂但脆弱的策略**，尤其是在医疗场景中。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Agentic 架构可有效支持可信的大规模心理健康筛查**
   - 通过模块化 Agent 设计，实现了对 LLM pipeline 的精细化控制与治理。
   - 框架能够在不牺牲性能的前提下，显著降低评估成本（proxy 先筛机制）。

2. ✅ **Non-regression Policy 是保障系统稳健性的关键**
   - 一旦某个配置被验证为最优，后续任何变更都不能轻易推翻它。
   - 这种“只进不退”的机制极大提升了系统的可靠性与可维护性。

3. ✅ **稳定配置优于过度优化**
   - 实验发现，最简单的配置（cosine similarity + dynamic Top-k=5 + threshold=0.75）已经达到了最佳性能。
   - 更复杂的机制（如 MMR、PRF、动态温度）反而容易引起性能波动甚至退化。

4. ✅ **Proxy Metrics 可有效指导探索**
   - 多种轻量级 proxy（语义覆盖率、排名稳定性、cheap judge）能够较准确预测最终性能趋势，大幅减少 GPT-4 调用次数。

5. 🌐 **框架具备向 population-scale 扩展的潜力**
   - 不依赖专用硬件或重型索引系统，适合集成进电子病历（EHR）、远程医疗平台等真实世界系统。

---

### 方法的局限性
1. **实验范围有限**
   - 当前仅在一个数据集（DAIC-WOZ）上进行了验证，尚未在跨机构、跨语言或多疾病场景下测试泛化能力。

2. **依赖外部 LLM（GPT-4）作为裁判**
   - “gold judge”本身具有不确定性，尽管设置了 deterministic 参数（temp=0.0），仍可能存在 prompt 敏感性问题。

3. **自动化程度仍有提升空间**
   - 当前 Orchestrator 的决策逻辑仍部分基于规则，未来可引入强化学习或贝叶斯优化进一步智能化。

4. **伦理与隐私考量不足**
   - 文中未深入讨论如何处理敏感心理数据的安全存储、访问控制与合规性问题。

---

### 未来工作方向
1. **扩展 variability space**
   - 探索更多 truncation 策略（如 salience-based summarization）、embedding 模型、RAG 架构（hybrid retrieval）等。

2. **引入统计测试机制**
   - 替代单次运行判断，采用多次重复实验 + 显著性检验（如 paired t-test）来决定是否接受新配置，更好地应对 LLM 随机性。

3. **支持在线学习与持续演进**
   - 允许在生产环境中安全地注入新策略，并自动评估其影响。

4. **应用于其他临床任务**
   - 如焦虑、PTSD、认知障碍等疾病的早期识别，以及个性化干预推荐。

5. **构建开源 Agentic SE 工具链**
   - 将此框架抽象为通用的 **Agentic Software Engineering Toolkit**，服务于代码分析、需求工程等领域。

---

## 总结

该论文提出了一个开创性的 **Agentic LLM-Based Framework**，将软件工程中的模块化、配置管理与非回归保障理念引入 LLM 应用开发，特别针对心理健康筛查这一高风险、高价值场景。其实验验证了：
> **“稳健优于炫技”** —— 在医疗 AI 中，一个经过严格验证、可解释、不会退化的简单系统，远胜于频繁变动却不可靠的复杂模型。

这项工作不仅推动了 **LLM in Healthcare** 的可信落地，也为 **Agentic Software Engineering** 提供了一个强有力的实践范例。

</details>

---

### 13. [PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory for Long-Horizon Agents](https://arxiv.org/abs/2605.12260)

**Authors**: Jingyi Peng, Zhongwei Wan, Weiting Liu, Qiuzhuang Sun  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.12260v1  

#### Abstract
Long-horizon language agents accumulate conversation history far faster than any fixed context window can hold, making memory management critical to both answer accuracy and serving cost. Existing approaches either expand the context window without addressing what is retrieved, perform heavy ingesti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory for Long-Horizon Agents — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**长周期语言智能体（long-horizon agents）** 中，对话历史迅速积累，远超任何固定长度的上下文窗口（context window）。这导致两个核心挑战：
- **答案准确性**：如何从海量记忆中检索出真正相关的证据；
- **服务成本**：如何控制输入到 LLM 的上下文 token 数量。

现有方法存在以下不足：
- 扩展上下文窗口的方法（如 long-context LLMs）不解决“检索什么”的问题；
- 在写入时进行重事实提取（heavy ingestion-time fact extraction）代价高昂；
- 启发式图遍历（heuristic graph traversal）效率和精度均不高。

PRISM 正是为了解决这一**准确率-上下文成本权衡（accuracy-context-cost trade-off）** 的根本矛盾而提出的。

---

### ✅ 提出的新方法与创新思路

PRISM 是一个**无需训练、纯推理侧（training-free, retrieval-side）** 的框架，将长周期记忆管理建模为一个联合的**检索-压缩问题（joint retrieval-and-compression problem）**，运行在一个**结构化图记忆（graph-structured memory）** 上。

其核心由四个正交的推理时模块组成：

| 模块 | 功能 |
|------|------|
| **N1: Hierarchical Bundle Search** | 在**带类型的路径模板（typed relation paths）** 上进行分层束搜索，优先发现多跳、因果等复杂关系链路 |
| **N2: Query-Sensitive Edge Costing** | 根据检测到的查询意图动态调整不同类型边的遍历成本，使检索更符合语义意图 |
| **N3: Evidence Compression** | 使用单次 LLM 调用对候选证据包进行重排序与压缩，生成紧凑的上下文 |
| **N4: Adaptive Intent Routing** | 通过三级级联路由（关键词匹配 → 原型嵌入 → LLM分类），尽可能避免每次查询都调用 LLM 进行意图识别 |

> 🔑 **关键创新**：首次将**基于类型路径的图检索**与**LLM端证据压缩**结合，且**无需训练任何策略模型**，可即插即用（plug-in）于已有结构化记忆系统。

---

### ✅ 相比现有方法的优势

| 维度 | PRISM 的优势 |
|------|-------------|
| **性能** | 在显著更低的上下文预算下实现更高的准确率，占据“高准确-低开销”空白区域 |
| **效率** | 平均每查询仅需约 2K tokens，比 full-context 方法减少 **13倍** |
| **通用性** | 不依赖特定训练，兼容任意后端结构化记忆系统 |
| **可解释性** | 检索路径可追踪，支持最小成本路径回溯（min-cost path tracing） |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **LoCoMo** [14]：一个专为评估长周期对话记忆设计的基准测试。
  - 包含 10 场多轮对话，共 1,540 个 QA 对；
  - 覆盖四类任务：
    - 单跳（Single-hop）
    - 多跳（Multi-hop）
    - 时间推理（Temporal）
    - 开放域（Open-domain）

> ❗排除第5类（对抗性拒绝），因不涉及检索能力。

---

### ⚙️ 实验设置

| 设置项 | 配置说明 |
|-------|--------|
| **主干模型** | `gpt-4o-mini`（answer & judge model），temperature=0.0 |
| **评估协议** | “Answer-then-judge”：先让模型作答，再由另一个 LLM 判断是否正确 |
| **Tokenizer** | 固定 tokenizer，确保 token 计数一致 |
| **上下文预算** | 测量每查询平均使用的 context tokens 数量 |
| **随机种子** | 固定为 42，保证可复现性 |

---

### 📊 评估指标

| 指标 | 定义 |
|------|-----|
| **LLM-judge score** | `(CORRECT / (CORRECT + WRONG))`，越高越好 |
| **Context tokens per query** | 检索并送入 answer model 的平均 token 数，越低越好 |
| **Per-1K Efficiency** | judge score / (context_tokens / 1000)，衡量单位 token 效率 |
| **Evidence Recall@K (ER@K)** | 前 K 个检索结果中包含黄金证据的比例（用于消融分析） |

---

### 🔁 基线方法对比

| 类别 | 方法 |
|------|------|
| **Full-context baseline** | 将全部 ~26K tokens 输入模型 |
| **Graph-based retrieval** | MAGMA [7]：基于多图结构的启发式 beam search |
| **Fact-extraction + retrieval** | Mem0, Mem09 [3]：在 ingestion 阶段提取结构化事实 |
| **Commercial platform** | Mem0 platform：商业托管版本，非开源 |
| **Other retrieval systems** | M-Flow [6]：近期先进方法，使用更强模型 |

> 所有 same-protocol 方法统一使用 `gpt-4o-mini` 和相同 prompt 设计，公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 总体性能表现（Table 1）

| 方法 | Judge Score | Context Tokens/Query | Per-1K Eff. |
|------|------------|------------------------|--------------|
| Full Context | 0.481 | 26,031 | 0.018 |
| MAGMA | 0.688 | 3,370 | 0.204 |
| Mem0 | 0.669 | 1,764 | 0.379 |
| Mem09 | 0.684 | 3,616 | 0.189 |
| **PRISM (ours)** | **0.831** | **2,023** | **0.411** |

> ✅ **PRISM 在所有 same-protocol 方法中全面领先**：
- 准确率提升 **+35.2pp** vs Full Context
- 提升 **+14.2pp** vs Mem09
- 单位 token 效率最高（0.411）

---

### 🔍 关键对比发现

- **上下文大小 ≠ 更好效果**：  
  Full Context 使用 26K tokens 但得分仅为 0.481，低于所有检索方法（最低的 Mem0 也有 0.669），说明**检索质量比原始长度更重要**。

- **PRISM 实现数量级压缩**：  
  相比 full-context，**token 数减少 13 倍**（~26K → ~2K），同时准确率大幅提升。

- **不同协议参考对比**：
  - M-Flow 使用更强模型（gpt-5-mini）仍落后 PRISM（0.818 vs 0.831）
  - PRISM + gpt-5.5 可达 **0.891**，表明残差错误部分来自 answer model 能力限制
  - Mem0 商业平台虽达 0.916，但消耗 ~7K tokens，效率仅为 PRISM 的 1/3

> 💡 PRISM 成功占据了“高准确-低开销”前沿上的**此前空白角落**。

---

### 🔬 消融实验结果（Ablation Study）

#### 表格摘要（Table 2）

| 配置 | Judge Score | ER@5 | Context Tokens |
|------|------------|--------|----------------|
| PRISM (完整) | 0.831 | 0.694 | 2,023 |
| -N1 (无 relation paths) | 0.831 | 0.694 | 2,024 |
| -N2 (无 edge cost 调整) | 0.831 | 0.694 | 2,020 |
| **-N3 (无 LLM 重排)** | **0.825** | **0.627↓** | **4,108↑** |
| +N4 (启用 intent routing) | 0.833 | 0.694 | 2,023 |

#### 关键发现：

- **N3（Evidence Compression）是主导组件**：
  - 移除后 ER@5 下降 **6.8pp**
  - 上下文膨胀至 **4,108 tokens**（翻倍）
  - 是实现**高效压缩**的关键机制

- **N1/N2 在 LoCoMo 上影响微弱**：
  - 因为该数据集主要是“锚点可发现”（anchor-discoverable），多数问题可通过表面相似性解决
  - 仅有约 **3% 的样本需要真正的多跳桥接结构**
  - 预期在 MuSiQue 或 HotpotQA 等更难的数据集上会显现价值

- **N4 显著降低 LLM 调用次数**：
  - 42.3% 查询通过零 LLM 路径处理（关键词或原型匹配）
  - 其中 temporal 查询高达 82.6% 被免调用
  - **不牺牲准确率的前提下节省推理成本**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **长周期记忆应视为 joint retrieval-compression 问题**：  
   单独优化检索或压缩都无法达到最优平衡；PRISM 通过两者的协同实现了帕累托前沿突破。

2. **结构化图记忆 + 意图感知检索 = 高效精准**：  
   利用 typed relation paths 和 intent-conditioned edge costing，能有效捕捉复杂语义关联。

3. **LLM-side compression 是压缩主力**：  
   N3 模块不仅提升精度（过滤无关项），更是将上下文压缩至 2K 的核心技术。

4. **无需训练即可超越强基线**：  
   PRISM 完全无需 fine-tuning 或 policy learning，即可在严格上下文预算下击败多种先进方法。

5. **LoCoMo 主要是“锚点可发现”基准**：  
   当前主流数据集对多跳推理挑战不足，未来需更具挑战性的 benchmark。

---

### ⚠️ 局限性（Limitations）

- 当前聚焦于**对话记忆**，尚未扩展到包含动作、工具调用、反馈等更丰富的 agent 轨迹；
- **SEMANTIC 边被禁用**：因在 LoCoMo 上引入噪声过多，未来可在概念密集文本中重新激活；
- 图构建依赖 ingestion pipeline，若上游存在偏见或隐私信息，PRISM 会忠实传播；
- 在当前 benchmark 上，N1/N2 的优势未充分体现，需在更难的多跳任务中验证。

---

### 🔮 未来工作方向

- 扩展至 **action-aware memory graphs**，支持完整 agent 轨迹管理；
- 探索 **dynamic edge activation**，根据领域自动启用 SEMANTIC 或其他关系类型；
- 构建更具挑战性的 **multi-hop reasoning benchmarks**，推动图检索能力发展；
- 结合 PRISM 与 KV-cache 优化技术，进一步降低推理延迟；
- 引入 ingestion-time filtering，防止敏感/偏见内容进入 memory graph。

---

> 🧩 **一句话总结**：  
> PRISM 通过“意图感知的图路径检索 + LLM端证据压缩”，在无需训练的前提下，实现了**高准确率与极低上下文成本的完美平衡**，为构建可扩展、低成本的长周期语言智能体提供了新范式。

</details>

---

### 14. [A Resampling-Based Framework for Network Structure Learning in High-Dimensional Data](https://arxiv.org/abs/2605.12706)

**Authors**: Ziwei Huang, Zeyuan Song, Paola Sebastiani, Stefano Monti  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.12706v1  

#### Abstract
RSNet is an open-source R package that provides a resampling-based framework for robust and interpretable network inference, designed to address the limited-sample-size challenges common in high-dimensional data. It supports both the estimation of partial correlation networks modeled as Gaussian net...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Resampling-Based Framework for Network Structure Learning in High-Dimensional Data

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对高维数据（如‘omics’数据）中常见的“小样本、大变量”（small *n*, large *p*）问题，解决传统网络推断方法在**样本量有限时网络结构可靠性差**、**缺乏统计稳健性**以及**难以处理相关观测（如家族数据）** 的挑战。

此外，现有工具在**高阶拓扑分析**（如 graphlet 分析）方面存在效率瓶颈，尤其是对**带符号图（signed networks）** 的支持不足。

---

### 提出的新方法与创新
作者开发了 **RSNet** —— 一个开源的 R 包，提出了一种基于**重采样（resampling-based）框架**的网络结构学习方法，其核心创新包括：

- ✅ **统一的重采样框架**  
  支持多种重采样策略（bootstrap、subsampling、cluster-based resampling），用于量化边级不确定性，并构建鲁棒的共识网络（consensus network），提升在小样本下的稳定性。

- ✅ **支持混合数据类型的条件高斯贝叶斯网络（CGBN）**  
  可同时建模连续变量与离散变量，适用于更广泛的生物医学数据场景。

- ✅ **首次实现近常数时间构造 signed GDVM（Graphlet Degree Vector Matrix）**  
  利用先进的 graphlet 计数算法（如 ORCA 的改进版本）结合并行计算，在稀疏网络上以 $O(|d|)$ 时间复杂度高效生成带符号的图基特征矩阵，显著提升了高阶拓扑分析的可扩展性。

- ✅ **集成下游分析模块**  
  提供从网络推断到 centrality 分析、community detection、differential connectivity 和 graphlet-based topology analysis 的端到端流程。

---

### 相比现有方法的优势

| 功能/特性 | RSNet | glasso / huge | SILGGM | BDgraph | RHugin | igraph / ORCA |
|---------|-------|---------------|--------|--------|--------|----------------|
| 支持重采样评估稳定性 | ✅ | ❌ | ❌ | ⚠️（Bayesian 后验概率） | ❌ | ❌ |
| 提供经验置信区间与调整 p 值 | ✅ | ❌ | ✅（渐近法） | ❌ | ❌ | ❌ |
| 支持 cluster-based resampling（家族数据） | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 支持 CGBN（混合数据） | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| 构造 signed GDVM（高效） | ✅（近常数时间） | ❌ | ❌ | ❌ | ❌ | ❌（仅 unsigned） |
| 集成 graphlet 分析于统一框架 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

> 🔍 **关键优势总结**：RSNet 是首个将**重采样稳健性评估**、**混合数据建模能力**与**高效 signed graphlet 分析**整合于一体的开源工具，填补了当前 R 生态系统中的多项空白。

---

## 2. 核心实验方法和设置

### 使用的数据集
论文展示了 RSNet 在多个真实世界生物学数据集上的应用，主要包括：

- 🧬 **衰老与长寿研究队列**：
  - New England Centenarian Study (NECS)
  - Long Life Family Study (LLFS)
  - Integrative Longevity Omics (ILO)

- 🧠 **神经退行性疾病数据**：
  - Late-Onset Alzheimer’s Disease (LOAD)

- 🧫 **癌症组学数据**：
  - The Cancer Genome Atlas (TCGA) 多癌种数据

这些数据具有典型的高维特征（*p* >> *n*）、复杂的变量类型组合及潜在的相关观测结构（如家系成员间相关性）。

---

### 实验设置
- **输入**：sample-by-feature 矩阵（如基因表达、代谢物水平等）
- **输出**：共识网络（weighted 或 binary adjacency matrices）
- **重采样策略**：
  - 对 Gaussian Network：使用 `ensemble_ggm()` 函数，支持 unstratified/stratified bootstrap/subsampling，以及 cluster/fractional cluster bootstrap。
  - 对 CGBN：使用 `ensemble_cgbn()`，基于 RHugin 实现结构学习。
- **并行化设计**：所有重采样与 GDVM 构造均支持多线程加速。
- **下游分析**：基于共识网络进行：
  - Centrality analysis
  - Community detection
  - Graphlet-based topology analysis
  - Differential connectivity analysis

---

### 评估指标
- 边选择频率（edge-selection frequency）
- 经验置信区间（empirical confidence intervals）
- 调整后的 p 值（adjusted p-values）
- 共识网络的稳定性与稀疏性
- 高阶拓扑模式变化（通过 signed GDVM 捕获）

未报告传统机器学习意义上的 AUC 或 F1-score，而是强调**统计可靠性和解释性增强**。

---

### 基线方法对比
虽然没有直接列出定量性能比较表格，但文中明确指出以下工具为对比对象，并说明其局限性：

- **glasso / huge**：仅提供单一网络估计，无统计推断支持。
- **SILGGM**：虽提供边缘置信区间，但依赖渐近正态假设，且不支持重采样。
- **BDgraph**：基于贝叶斯后验采样，无法给出频率主义意义下的稳定性和显著性。
- **RHugin**：支持 CGBN 学习但缺少重采样机制。
- **igraph / ORCA**：支持 graphlet 分析但不支持 signed graphlets，也不集成于网络推断流程。

---

## 3. 主要实验结果和性能指标

### 关键性能表现
- ✅ **高效构建 signed GDVM**：在稀疏网络中达到 $O(|d|)$ 时间复杂度，接近常数时间，远优于暴力枚举的 $O(p^3)$。
- ✅ **成功应用于多个真实高维数据集**：在 NECS、LLFS、TCGA 等数据中稳定构建出具有生物学意义的共识网络。
- ✅ **揭示高阶拓扑差异**：通过 signed graphlet 分析识别出特定条件下（如疾病 vs 正常）节点局部连接模式的变化，超越传统 centrality 指标。
- ✅ **支持家族数据建模**：cluster bootstrap 成功保留簇内依赖关系，避免标准方法导致的假阳性膨胀。

---

### 与基线方法对比结果
- 在相同数据下，RSNet 相比 single-network 方法（如 glasso）能有效过滤不稳定边，提高网络可信度。
- 相比 SILGGM 的渐近推断，RSNet 的重采样方法更适合小样本场景，无需分布假设。
- 相比 BDgraph 的贝叶斯边缘包含概率，RSNet 提供的是频率主义视角下的边选择频率和经验 CI，更具可解释性。
- ORCA 无法处理符号信息，而 RSNet 支持 signed graphlet degree signature，增强了对调控方向（激活/抑制）的捕捉能力。

---

### 消融实验（文中未明确列出）
论文未进行形式化的消融实验（ablation study），但通过模块化设计隐含验证了各组件价值：

- 若关闭重采样 → 失去不确定性评估与稳定性保障
- 若移除 cluster-based resampling → 家族数据中可能低估方差、增加假阳性
- 若不使用并行化 → GDVM 构造速度大幅下降，难以扩展至大规模网络

因此，**重采样 + 并行化 + signed graphlet 分析**三者共同构成 RSNet 的核心技术支柱。

---

## 4. 关键结论和发现

### 主要发现
1. **重采样是提升高维网络推断鲁棒性的关键**：通过集成多个重采样网络，RSNet 显著提高了边估计的稳定性与可重复性。
2. **consensus network 更具生物学可解释性**：相比单一网络，共识网络更能反映稳定的分子交互模式。
3. **signed graphlet 分析能揭示传统方法忽略的高阶结构变化**：例如某些 hub 节点在不同条件下虽度数不变，但其邻域符号构型发生改变，提示功能切换。
4. **RSNet 可灵活适配独立与相关数据结构**：特别适合家系、纵向或批量效应存在的数据。

---

### 方法的局限性
- 当前主要依赖 SILGGM 和 RHugin 的底层算法，自身未提出新的网络结构学习算法。
- 尽管优化了 graphlet 计算，但在极稠密网络中仍可能面临计算压力。
- 对超大规模网络（如 >10,000 节点）的应用尚未充分测试。
- 不支持动态或时空网络建模。

---

### 未来工作方向
- 扩展至更多类型的混合图模型（如 hybrid Bayesian networks）。
- 引入深度学习辅助的 network prior 或 regularization mechanism。
- 开发可视化工具以直观展示 signed graphlet signatures 的变化。
- 推广至单细胞 omics 数据中的 cell-type-specific network inference。

---

## 总结

📌 **RSNet 是一个面向高维数据、强调统计稳健性与结构可解释性的网络推断平台**。它不是单纯追求预测精度的黑箱模型，而是致力于为科研人员提供一套**透明、可复现、可解释**的分析流水线。

🎯 其最大亮点在于：
> **将 resampling 的稳定性控制、CGBN 的灵活性与 signed graphlet 的高阶表征能力融为一体，并以高效并行实现落地于 R 工具包**。

🔗 项目地址：[github.com/montilab/RSNet](https://github.com/montilab/RSNet)  
📦 适用领域：生物信息学、系统生物学、精准医学、复杂系统建模等需要从高维数据中挖掘稳健网络结构的研究方向。

</details>

---

### 15. [On Predicting the Post-training Potential of Pre-trained LLMs](https://arxiv.org/abs/2605.11978)

**Authors**: Xiaoyuan Li, Yubo Ma, Kexin Yang, Moxin Li, Keqin Bao, Wenie Wang, Fuli Feng, Dayiheng Liu  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.11978v1  

#### Abstract
The performance of Large Language Models (LLMs) on downstream tasks is fundamentally constrained by the capabilities acquired during pre-training. However, traditional benchmarks like MMLU often fail to reflect a base model's plasticity in complex open-ended scenarios, leading to inefficient model s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《On Predicting the Post-training Potential of Pre-trained LLMs》核心总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对当前大语言模型（LLMs）开发流程中的一个关键瓶颈：如何在**预训练阶段**有效预测一个基础模型（pre-trained base model）在经过指令微调（instruction tuning）、强化学习（RL）等**后训练（post-training）** 后的最终性能。

传统做法依赖于在多个选择上进行昂贵的后训练试错，效率低下。现有的评估基准（如 MMLU、C-Eval）主要衡量静态知识回忆能力，与后训练所需的复杂指令遵循、对齐人类意图等动态能力相关性较弱，无法准确预测模型的“后训练潜力”。

### 提出了什么新方法或新思路
为解决上述问题，论文提出了以下核心创新：

1.  **新任务定义**：正式引入了“**预测后训练潜力（predicting post-training potential）**”这一新任务，旨在通过仅对预训练模型进行推理评估，来预测其在下游开放域任务上的最终生成性能。
2.  **新框架 RuDE**：提出 **RuDE (Rubric-based Discriminative Evaluation)** 框架。该框架的核心思想是利用“**生成-评估一致性假设（Generation-Evaluation Consistency hypothesis）**”，即一个模型若能很好地区分高质量和低质量的响应，则它也更有可能学会生成高质量的响应。
3.  **新评估范式**：采用**判别式评估（discriminative evaluation）** 来规避预训练模型本身生成能力不足的问题。具体做法是构建成对的对比样本（contrastive pairs），让模型判断哪个响应更好。
4.  **系统化评估体系**：提出 **4C Taxonomy**，将复杂的对齐能力解耦为四个维度：
    *   **Competence**（能力）：事实准确性、逻辑推理。
    *   **Content**（内容）：完整性、连贯性、相关性。
    *   **Control**（控制）：格式、长度、范围等非语义约束。
    *   **Compliance**（合规）：安全性、角色扮演、实用性。
5.  **高质量数据构造**：设计了一个**生成器-验证器迭代精炼管道（generator-verifier iterative refinement pipeline）**，通过受控降级（controlled degradation）精确地构造出高质量的“硬负例”（hard negatives），确保正负样本之间的差异仅在于特定的评分标准（rubric）违反。

### 相比现有方法的优势
*   **高预测性**：相比传统基于知识问答（multiple-choice）的基准，RuDE 与后训练性能的相关性显著更高（>90%）。
*   **成本效益**：无需执行完整的后训练过程，即可高效筛选出最有潜力的基础模型，大幅节省计算资源。
*   **细粒度诊断**：4C Taxonomy 提供了模型潜力的多维分析，有助于理解模型的优势和短板。
*   **通用性强**：框架适用于多样化的开放域任务，如医疗、法律金融、复杂指令遵循和创意写作。

## 2. 核心实验方法和设置

### 使用了哪些数据集
研究者从四个不同领域的现有基准中改编并构建了评估数据集：
*   **HealthBench**：医学咨询领域，侧重安全性和专业规范。
*   **PRBench**：法律与金融专业报告，考察逻辑一致性和术语准确性。
*   **AdvancedIF**：复杂指令遵循，包含嵌套的格式和逻辑约束。
*   **WritingBench**：创意写作，评估风格迁移和修辞技巧。

最终评估数据集共包含 **28,683** 个样本，并通过控制违反的评分标准数量（`||`）来调节难度。

### 实验设置和评估指标
*   **评估对象**：涵盖多种架构（Dense 和 MoE）和参数规模（4B 到 1T）的 state-of-the-art 预训练基础模型，如 Qwen3、DeepSeek-V3.1、GLM-4.5、Gemma3 等。
*   **评估方式**：在判别任务中，给定一个查询（query）和两个响应（y+ 和 y-），要求模型判断哪个响应更优。位置（A/B）随机分配以消除偏差。
*   **评估指标**：
    *   **主要指标**：**准确率（Accuracy, Acc）**，即模型正确地将更高概率赋予正样本的比例。
    *   **核心验证指标**：**皮尔逊相关系数（Pearson correlation, corr）**，用于衡量基础模型在 RuDE 上的得分与其对应的指令微调版本在标准生成基准上得分之间的相关性。

### 基线方法对比
论文没有直接对比其他具体的“预测潜力”的基线方法，而是将 RuDE 的预测能力与**传统的预训练模型评估指标**进行了对比，证明了它们的不足：
*   **传统基准**：如 **MMLU**, **C-Eval**, **SuperGPQA** 等知识问答准确率。
*   **困惑度（Perplexity, PPL）**：衡量语言建模能力的传统指标。
*   **结果**：图1和附录N显示，这些传统指标与后训练性能的相关性很弱（例如，MMLU 的 corr ≈ 0.56），而 RuDE 的相关性则超过 0.9。

## 3. 主要实验结果和性能指标

### 关键性能数据
*   **强相关性**：在所有四个领域，RuDE 得分与后训练生成性能之间均表现出极强的皮尔逊相关性，其中 **AdvancedIF 达到 corr=0.91 (p<0.001)**，PRBench 为 corr=0.80，HealthBench 为 corr=0.67，WritingBench 为 corr=0.62。
*   **SOTA 模型表现**：在 RuDE 上表现最好的模型是 **DeepSeek-V3.1**（平均准确率 78.8%），其次是 **GLM-4.5** 和 **Kimi-K2**。
*   **领域专长**：**Kimi-K2** 在 WritingBench 上表现最佳，而 **GLM-4.5** 在 PRBench 上优势明显，体现了模型的领域特异性。

### 与基线方法的对比结果
*   **压倒性优势**：RuDE 与后训练性能的相关性（>0.9）远高于传统知识基准（通常 <0.6）和困惑度（PPL）的相关性（在某些领域甚至不显著）。
*   **实际指导意义**：在 Reinforcement Learning (RL) 实验中，RuDE 预测潜力更高的较小模型（**Qwen3-4B-Base**）在经过相同后训练后，确实**超越了**更大的模型（**Qwen2.5-7B-Base**），这直接验证了 RuDE 的实用价值。

### 消融实验结果
*   **必要性验证**：论文通过消融实验证明了其“受控降级”（Controlled Degradation）管道的必要性。
    *   **自然采样（Rejection Sampling）**：直接从模型生成中采样最好和最差的响应作为对比对，导致任务过于困难，所有模型都难以区分，失去了判别力。
    *   **简单改写（Locate-and-Rewrite）**：通过定位并改写文本来制造错误，会产生明显的“拼接痕迹”（stitching artifacts），使得模型可以轻易通过表面特征识别负样本，导致任务过于简单，无法区分不同模型的真实潜力。
*   **结论**：RuDE 的“受控降级”管道成功地在“太难”和“太易”之间找到了平衡点，生成了具有适当难度的高质量对比对，从而能够有效地区分模型的后训练潜力。

## 4. 关键结论和发现

### 论文的主要发现
1.  **核心假设成立**：“生成-评估一致性”假设在预测后训练潜力上是有效的。一个预训练模型的**判别能力**是其**生成潜力**的强有力预测指标。
2.  **RuDE 高效可靠**：RuDE 框架能够以极高的相关性（>90%）预测后训练性能，是一种计算高效的模型选择机制。
3.  **小模型潜力巨大**：通过更好的预训练方法（如 Qwen3 相比 Qwen2.5），即使是较小的模型（如 4B）也能展现出超越更大模型（如 7B）的后训练潜力，打破了单纯依靠模型规模的局限。
4.  **能力可解耦**：4C Taxonomy 成功揭示了不同模型在不同对齐维度上的优势和劣势，例如小模型在 **Control**（控制）维度上普遍较弱。

### 方法的局限性
*   **判别-生成差距**：GD-Potential 假设可能并非在所有情况下都成立。一个模型可能具备很强的“理论家”（theoretician）般的判别能力，但缺乏流畅生成的“执行者”（motor control）能力。
*   **依赖生成器质量**：RuDE 构造的黄金标准响应依赖于强大的生成器（如 Gemini-3-Pro），因此其上限受限于该“教师模型”的能力。
*   **静态评估**：RuDE 评估的是模型的静态准备状态，无法预测后训练过程中可能出现的动态问题，如灾难性遗忘（catastrophic forgetting）或奖励黑客攻击（reward hacking）。

### 未来工作方向
*   探索如何进一步缩小判别能力和生成能力之间的差距。
*   研究如何降低对顶级生成器的依赖，使评估更加自洽。
*   将 RuDE 的理念应用于更广泛的模型开发场景，如数据选择、课程学习等。
*   进一步探索和优化 4C Taxonomy，使其能覆盖更全面的对齐维度。

</details>

---

### 16. [Continual Fine-Tuning of Large Language Models via Program Memory](https://arxiv.org/abs/2605.13162)

**Authors**: Hung Le, Svetha Venkatesh  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.13162v1  

#### Abstract
Parameter-Efficient Fine-Tuning (PEFT), particularly Low-Rank Adaptation (LoRA), has become a standard approach for adapting Large Language Models (LLMs) under limited compute. However, in continual settings where models are updated sequentially with small datasets, conventional LoRA updates struggl...

---

### 17. [Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding](https://arxiv.org/abs/2605.12185)

**Authors**: Yigeng Zhou, Wu Li, Yifan Lu, Yequan Wang, Xuebo Liu, Wenya Wang, Jun Yu, Min Zhang, Jing Li  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.12185v1  

#### Abstract
Large language models accumulate extensive parametric knowledge through pre-training. However, knowledge conflicts occur when outdated or incorrect parametric knowledge conflicts with external knowledge in the context. Existing methods address knowledge conflicts through contrastive decoding, but in...

---

### 18. [Efficient and Portable Support for Overdecomposition on Distributed Memory GPGPU Platforms](https://arxiv.org/abs/2605.12734)

**Authors**: Aditya Bhosale, Anant Jain, Shourya Goel, Ritvik Rao, Peddoju Sateesh Kumar, Laxmikant Kale  
**Category**: cs.DC  
**Published**: 2026-05-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.12734v1  

#### Abstract
Overdecomposition has emerged as a powerful and sometimes essential technique in parallel programming. Many application domains or frameworks, including those based on adaptive mesh refinements, or tree codes use it. Charm++ is a parallel programming system which has demonstrated the utility of over...

---

### 19. [Learning When to Act: Communication-Efficient Reinforcement Learning via Run-Time Assurance](https://arxiv.org/abs/2605.12561)

**Authors**: Adam Haroon, Erick J. Rodr\'iguez-Seda, Cody Fleming, Tristan Schuler  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.12561v1  

#### Abstract
Safe reinforcement learning (RL) typically asks $\textit{what}$ an agent should do. We ask $\textit{when}$ it needs to act, and show that a single policy can jointly learn control inputs and communication-efficient timing decisions under a pointwise Lyapunov safety shield. We focus on stabilization ...

---

### 20. [Byzantine-Robust Distributed Sparse Learning Revisited](https://arxiv.org/abs/2605.13283)

**Authors**: Yuxuan Wang, Lixin Zhang, Kangqiang Li  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.13283v1  

#### Abstract
We revisit Byzantine robust distributed estimation for high-dimensional sparse linear models. By combining local $\ell_1$-regularized robust estimation with robust aggregation at the server, the framework applies to pseudo-Huber regression, quantile regression, and sparse SVM. We show that the resul...

---

### 21. [Uncertainty-Aware Prediction of Lung Tumor Growth from Sparse Longitudinal CT Data via Bayesian Physics-Informed Neural Networks](https://arxiv.org/abs/2605.13560)

**Authors**: Lingfei Kong, Haoran Ma  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.13560v1  

#### Abstract
This work studies lung tumor growth prediction from sparse and irregular longitudinal computed tomography (CT) observations with measurement variability. A Bayesian physics-informed neural network is developed by combining Gompertz growth dynamics with low-dimensional Bayesian inference in the log-v...

---

### 22. [Force-Aware Neural Tangent Kernels for Scalable and Robust Active Learning of MLIPs](https://arxiv.org/abs/2605.13788)

**Authors**: Eszter Varga-Umbrich, Zachary Weller-Davies, Paul Duckworth, Jules Tilly, Olivier Peltre, Shikha Surana  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.13788v1  

#### Abstract
Active learning for machine-learning interatomic potentials (MLIPs) must address several challenges to be practical: scaling to large candidate pools, leveraging energy-force supervision, and maintaining robustness when candidate pools are biased relative to the target distribution. In this work, we...

---

### 23. [Sampling More, Getting Less: Calibration is the Diversity Bottleneck in LLMs](https://arxiv.org/abs/2605.11128)

**Authors**: Amin Banayeeanzade, Qingchuan Yang, Dhruv Tarsadiya, Fatemeh Bahrani, Leonardo Blas, Alfy Samuel, Robin Jia, Meisam Razaviyayn, Sai Praneeth Karimireddy  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.11128v1  

#### Abstract
Diversity is essential for language-model applications ranging from creative generation to scientific discovery, yet modern LLMs often collapse into a narrow subset of plausible outputs. While prior work has developed benchmarks for measuring this lack of diversity, less is known about how the step-...

---

### 24. [MedHopQA: A Disease-Centered Multi-Hop Reasoning Benchmark and Evaluation Framework for LLM-Based Biomedical Question Answering](https://arxiv.org/abs/2605.12361)

**Authors**: Rezarta Islamaj, Robert Leaman, Joey Chan, Nicholas Wan, Qiao Jin, Natalie Xie, John Wilbur, Shubo Tian, Lana Yeganova, Po-Ting Lai, Chih-Hsuan Wei, Yifan Yang, Yao Ge, Qingqing Zhu, Zhizheng Wang, Zhiyong Lu  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.12361v1  

#### Abstract
Evaluating large language models (LLMs) in the biomedical domain requires benchmarks that can distinguish reasoning from pattern matching and remain discriminative as model capabilities improve. Existing biomedical question answering (QA) benchmarks are limited in this respect. Multiple-choice forma...

---

### 25. [MARLIN: Multi-Agent Game-Theoretic Reinforcement Learning for Sustainable LLM Inference in Cloud Datacenters](https://arxiv.org/abs/2605.13496)

**Authors**: H. Moore, S. Qi, D. Milojicic, C. Bash, S. Pasricha  
**Category**: cs.DC  
**Published**: 2026-05-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.13496v1  

#### Abstract
Large Language Models (LLMs) have become increasingly prevalent in cloud-based platforms, propelled by the introduction of AI-based consumer and enterprise services. LLM inference requests in particular account for up to 90% of total LLM lifecycle energy use, dwarfing training energy costs. The risi...

---

### 26. [Reinforced Collaboration in Multi-Agent Flow Networks](https://arxiv.org/abs/2605.12943)

**Authors**: Zheng Wang, Yuang Liu, Yangkai Ding  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.12943v1  

#### Abstract
Multi-agent systems provide a powerful way to extend large language models (LLMs) by decomposing a complex task into specialized subtasks handled by different agents. However, their performance is often hindered by error propagation, arising from suboptimal workflow design or inaccurate agent output...

---

### 27. [Attention Once Is All You Need: Efficient Streaming Inference with Stateful Transformers](https://arxiv.org/abs/2605.13784)

**Authors**: Victor Norgren  
**Category**: cs.LG  
**Published**: 2026-05-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.13784v1  

#### Abstract
Conventional transformer inference engines are request-driven, paying an O(n) prefill cost on every query. In streaming workloads, where data arrives continuously and queries probe an ever-growing context, this cost is prohibitive. We introduce a data-driven computational model centred on stateful s...

---

### 28. [Think Twice, Act Once: Verifier-Guided Action Selection For Embodied Agents](https://arxiv.org/abs/2605.12620)

**Authors**: Nishad Singhi, Christian Bialas, Snehal Jauhri, Vignesh Prasad, Georgia Chalvatzaki, Marcus Rohrbach, Anna Rohrbach  
**Category**: cs.AI  
**Published**: 2026-05-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.12620v1  

#### Abstract
Building generalist embodied agents capable of solving complex real-world tasks remains a fundamental challenge in AI. Multimodal Large Language Models (MLLMs) have significantly advanced the reasoning capabilities of such agents through strong vision-language knowledge and chain-of-thought (CoT) re...

---

### 29. [An Agentic AI Framework with Large Language Models and Chain-of-Thought for UAV-Assisted Logistics Scheduling with Mobile Edge Computing](https://arxiv.org/abs/2605.13221)

**Authors**: Hanwen Zhang, Dusit Niyato, Wei Zhang, Xin Lou, Malcolm Yoke Hean Low  
**Category**: cs.AI  
**Published**: 2026-05-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.13221v1  

#### Abstract
In cloud manufacturing, unmanned aerial vehicles (UAVs) can support both product collection and mobile edge computing (MEC). This joint operation forms a hybrid scheduling problem, where physical logistics decisions are coupled with computational task scheduling. In this paper, UAVs collect finished...

---

### 30. [How Does Differential Privacy Affect Social Bias in LLMs? A Systematic Evaluation](https://arxiv.org/abs/2605.11195)

**Authors**: Eduardo Tenorio, Karuna Bhaila, Xintao Wu  
**Category**: cs.CL  
**Published**: 2026-05-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.11195v1  

#### Abstract
Large language models (LLMs) trained on web-scale corpora can memorize sensitive training data, posing significant privacy risks. Differential privacy (DP) has emerged as a principled framework that limits the influence of individual data points during training, yet the relationship between differen...

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
