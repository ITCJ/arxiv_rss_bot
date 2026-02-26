# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-26 06:42:26 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DHP: Efficient Scaling of MLLM Training with Dynamic Hybrid Parallelism](https://arxiv.org/abs/2602.21788)

**Authors**: Yifan Niu, Han Xiao, Dongyi Liu, Wei Zhou, Jia Li  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 15.0  
**Type**: new  
**ArXiv ID**: 2602.21788v1  

#### Abstract
Scaling long-context capabilities is crucial for Multimodal Large Language Models (MLLMs). However, real-world multimodal datasets are extremely heterogeneous. Existing training frameworks predominantly rely on static parallelism strategies, which suffer from severe load imbalance, redundant communi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DHP: Efficient Scaling of MLLM Training with Dynamic Hybrid Parallelism

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前多模态大语言模型（**MLLMs**）在训练过程中面临严重的**负载不均衡**、**冗余通信**和**硬件利用率低下**的问题，尤其是在处理异构性强的多模态数据（如不同长度的视频、高分辨率图像）时。传统的静态并行策略（如 **Megatron-LM** 和 **DeepSpeed** 中的 4D 并行）采用固定的并行度划分，无法适应序列长度高度不均的数据分布，导致计算资源浪费。

### 🚀 提出的新方法：Dynamic Hybrid Parallelism (DHP)
本文提出 **Dynamic Hybrid Parallelism (DHP)** ——一种高效的动态混合并行框架，其核心思想是：
- 在每个微批次（micro-batch）中**动态重构通信组**和**调整并行度**（parallelism degree），以适配实际数据分布。
- 支持**非2的幂次方**（non-power-of-two）的并行度，突破传统 **Sequence Parallelism (SP)** 或 **Context Parallelism (CP)** 对并行度的限制。
- 设计了一个**两阶段近似算法**来高效求解该 NP-hard 调度问题，确保调度开销极低。

### 🔍 相比现有方法的优势
| 维度 | 传统方法（Megatron-LM / DeepSpeed） | DHP |
|------|-------------------------------|-----|
| 并行策略 | 静态并行（Static Mesh） | 动态并行（Dynamic Mesh） |
| 并行度灵活性 | 通常为 2 的幂（如 2, 4, 8） | 支持任意整数（如 3, 5, 6） |
| 负载均衡能力 | 差（长尾数据下严重失衡） | 强（自适应分配） |
| 通信效率 | 存在冗余通信（短序列被过度分片） | 最小化冗余通信 |
| 调度开销 | 无调度或高开销（如 FlexSP） | ≤86ms（毫秒级，可隐藏） |

> ✅ **核心优势总结**：DHP 实现了在极端数据异构性下的高硬件利用率，同时保持极低调度延迟。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **InternVid**：1000万视频片段 + 自动生成高质量字幕，数据分布呈长尾特性。
- **OpenVid**：高审美质量视频，最小分辨率为 512×512，用于高质量图文生成任务。
- **MSRVTT**：1万个视频 + 20万自然语言描述，涵盖20个类别，相对均匀但仍具多样性。

> 这些数据集共同体现了真实世界中多模态数据的**高度异质性和长尾分布**。

### ⚙️ 实验设置
- **硬件环境**：8节点集群，每节点8块 Ascend910B NPU（共64块），节点内通过 HCCS 互联，节点间使用 100Gbps InfiniBand。
- **模型规模**：从 2B 到 8B 参数的 **InternVL3** 和 **Qwen3VL** 系列 MLLM。
- **全局批大小（Global Batch Size, GBS）**：固定为 512。
- **并行配置**：TP 和 PP 固定为静态配置；DHP 动态优化 CP 分组与分配。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **端到端训练迭代时间**（End-to-end Iteration Time） | 单次完整前向+反向传播耗时 |
| **每设备吞吐量**（Token Throughput per-device） | 单位时间内处理的 token 数量（k tokens/s） |
| **扩展效率**（Scaling Efficiency） | 随 NPU 数量增加的吞吐增长趋势 |

### 🆚 基线方法对比
- **Megatron-LM**：支持 4D 并行（TP/PP/DP/CP），使用 Ring-style CP。
- **DeepSpeed**：基于 Ulysses-style SP，依赖 All-to-All 通信。
- 两者均采用**单一静态并行策略**，需手动调优并行参数。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据
#### （1）端到端加速比（Speedup）
DHP 在所有测试配置下均显著优于基线：

| 模型 | 数据集 | 加速比（vs. 最佳基线） |
|------|--------|------------------|
| InternVL3-8B | OpenVid | **1.35×** |
| InternVL3-8B | MSRVTT | 1.14× |
| Qwen3VL-8B | OpenVid | 1.32× |
| Qwen3VL-2B | OpenVid | 1.22× |

> 💡 在 **更大模型（8B）** 和 **更复杂数据集（OpenVid）** 上提升最为明显。

#### （2）吞吐量与可扩展性
- 当从 8 NPU 扩展到 64 NPU 时：
  - DHP 吞吐量下降最少，并呈现轻微上升趋势；
  - DeepSpeed 吞吐从 ~2.0k 下降到 ~1.7k tokens/s；
  - Megatron-LM 从 ~1.8k 下降到 ~1.52k tokens/s；
- DHP 相对 DeepSpeed 的加速比从 **1.02× 提升至 1.16×**，表明其具备**卓越的扩展性**。

#### （3）调度开销极低
| 设置 | Solver 时间 | 总调度时间 | 计算时间 |
|------|-----------|------------|----------|
| GBS=512 | ≤86ms | ≤921ms | 7.32s |
| NPU=64 | ≤86ms | ≤921ms | 7.32s |

> ✅ 调度时间始终远小于单个 GBS 的计算时间（<13%），可完全被计算掩盖，**不影响训练流水线**。

#### （4）成本估计误差低
| 模型 | 2B | 4B | 8B |
|------|----|----|----|
| Qwen3VL | 7.93% | 6.71% | 4.27% |
| InternVL3 | 7.48% | 6.54% | 4.12% |

> 成本预测误差低于 8%，且随模型增大而降低，说明 Profiler 具有良好的泛化能力。

### 🔍 消融实验与案例分析（Case Study）
| 场景 | 特点 | DHP 行为 | 加速比 |
|------|------|---------|--------|
| Case 1 (OpenVid) | 极端长尾分布 | 使用多种 CP 度（8×1, 6×2, 4×1, 2×2, 1×4） | **1.17×** |
| Case 2 (MSRVTT) | 分布较均匀 | CP 度更集中（4×2, 3×4, 2×6） | 1.14× |

> DHP 能根据数据分布自动选择最优分组策略，避免短序列的“过度并行”带来的通信开销。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **动态并行优于静态并行**：在异构多模态数据上，静态并行不可避免地造成负载失衡和资源浪费，而 DHP 可实现精细化资源匹配。
2. **非2幂并行度至关重要**：允许任意整数并行度使系统能更灵活应对短序列，减少通信冗余。
3. **调度开销可控且可隐藏**：通过两阶段算法（BFD + 2D-DP），可在毫秒级完成调度决策，不影响训练效率。
4. **高性能与强扩展性兼得**：DHP 不仅提速最高达 **1.36×**，还能在大规模集群上维持近乎线性的扩展效率。

### ⚠️ 方法的局限性
- **未动态调整 TP/PP**：由于权重重分布代价过高，TP 和 PP 仍为静态配置，限制了全栈动态优化的可能性。
- **依赖预建通信池**：虽然通过 group pooling 缓解了 HCCL 创建开销，但在极端动态场景下可能仍存在缓存压力。
- **目前仅集成于 Megatron-LM**：尚未验证在其他训练框架（如 DeepSpeed Zero-Infinity）中的通用性。

### 🔮 未来工作方向
1. 探索 **TP/PP 的轻量化动态重构机制**，进一步释放并行灵活性。
2. 将 DHP 思路推广至 **推理阶段**，实现训练-推理一体化动态调度。
3. 结合 **在线学习** 或 **强化学习** 实现更智能的并行策略搜索。
4. 支持跨模态异步加载与计算重叠，进一步压缩端到端延迟。

---

## ✅ 总结
**DHP** 是首个将**动态混合并行**成功应用于 MLLM 训练的工作，它通过引入灵活的非2幂并行度和高效的两阶段调度算法，在不增加训练延迟的前提下，显著提升了训练吞吐和硬件利用率。其实验结果证明了其在真实异构多模态数据上的强大适应能力和卓越性能，为下一代大规模 MLLM 训练系统提供了重要范式参考。

</details>

---

### 2. [Multi-Layer Scheduling for MoE-Based LLM Reasoning](https://arxiv.org/abs/2602.21626)

**Authors**: Yifan Sun, Gholamreza Haffar, Minxian Xu, Rajkumar Buyya, Adel N. Toosi  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2602.21626v1  

#### Abstract
Large Language Models (LLMs) have achieved remarkable success across a wide range of tasks, but serving them efficiently at scale remains a critical challenge due to their substantial computational and latency demands. While most existing inference frameworks rely on simple scheduling strategies suc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Multi-Layer Scheduling for MoE-Based LLM Reasoning**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
当前主流的 LLM 推理框架（如 vLLM、SGLang）在服务 **Mixture-of-Experts (MoE)** 模型时存在以下三大瓶颈：
- **Engine-Level**：采用简单的 Round-Robin (RR) 调度策略，忽略各引擎的实时负载（如 KV Cache 使用率、前缀 token 数量），导致负载不均衡。
- **Request-Level**：使用 First-Come-First-Serve (FCFS)，易引发长请求阻塞短请求（Head-of-Line Blocking），增加尾延迟。
- **Expert-Level**：未考虑专家之间的跨层依赖关系（Inter-Layer Expert Affinity）和热点分布（Hotspot），造成 GPU 间计算负载失衡。

这些问题限制了 MoE 模型的推理效率与资源利用率。

---

### 🚀 提出的新方法：**Gimbal**
提出一个**多层调度框架 Gimbal**，从三个层级协同优化调度决策：

| 层级 | 方法 |
|------|------|
| **Request-Level** | 引入 **Shortest-Job-First (SJF)** + **aging 机制** 的调度器，优先处理预填充 token 较少的请求，并防止大请求饥饿。 |
| **Engine-Level** | 设计 **负载感知的 DP Engine Load Balancer**，综合考虑 KV Cache 利用率、运行负载（RunningLoad）、用户粘性（User Stickiness）进行请求分发。 |
| **Expert-Level** | 构建 **Expert Dynamic Replacement Module**，结合专家激活频率与跨层依赖关系，动态调整专家在 GPU 上的位置，缓解热点并减少跨 GPU 通信开销。 |

---

### 🔍 相比现有方法的优势
- **非模型绑定设计**：仅依赖通用 MoE 路由信号，适用于 Mixtral、Switch Transformer、DeepSeek 等多种 MoE 架构。
- **系统级协同优化**：首次实现 request、engine、expert 三层联合调度，打破传统单层优化局限。
- **无需输出长度预测**：SJF 使用输入 token 数作为代价估计，避免不可靠的输出长度预测。
- **提升缓存命中率**：通过 user affinity 提高 prefix cache 复用，降低重复计算。

---

## 2. **核心实验方法和设置**

### 📊 数据集
- **BurstGPT Dataset**：用于主性能测试，包含真实世界 LLM 请求分布特征。
  - 构造五种合成分布以评估鲁棒性：
    - Random
    - Central
    - Descending
    - Two-end
    - Average
- **ShareGPT Dataset**：用于专门评估 **prefix cache hit rate** 和 user affinity 效果。

> 注：BurstGPT 不含用户 ID，故 ShareGPT 用于补充用户行为建模。

---

### ⚙️ 实验设置
- **硬件平台**：
  - 2×Intel Xeon Gold 6326 CPU @2.90GHz
  - 2×NVIDIA A100 80GB GPU（NVLink 连接）
  - 1TB 内存
- **模型**：
  - **Qwen3-30B-A3B**：典型的中小规模 MoE 模型，共 48 层，每层激活 3 个专家。
- **框架基础**：
  - 基于 **vLLM (0.9.1)** 修改实现，启用 expert parallelism 和 pplx-kernels 支持 all-to-all 通信。
- **阈值设定**：
  - `θ_k = 0.9`：KV Cache 超过 90% 视为饱和
  - `θ_diff = 10%`：引擎间 KV 差异容忍度
  - `θ_load = 3000 tokens`：运行负载差异阈值
  - `θ_age = 5s`：老化提权时间（低于 P99 TTFT）
  - `T = 3000 steps`：专家重定位周期

---

### 🎯 评估指标
| 指标 | 含义 |
|------|------|
| **TTFT** | Time To First Token，首 token 延迟，反映 prefll 阶段效率 |
| **TPOT** | Time Per Output Token，每个输出 token 的平均解码延迟 |
| **Throughput** | 系统吞吐量（tokens/s） |
| **Prefix Cache Block Hit Count** | 全局 prefix cache 命中次数 |
| **Prefix Cache Hit Rate (Global)** | 缓存块命中数 / 总探查数 |

---

### 🆚 基线方法对比
- **vLLM**：当前最先进的开源推理框架，使用 RR + FCFS。
- **Ablation Variants**：
  - **DPLB**：仅启用 DP Engine Load Balancer
  - **SJFS**：仅启用 SJF Scheduler
  - **EDR**：仅启用 Expert Dynamic Replacement

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（在 1.4 RPS 高负载下）

| 指标 | 结果 |
|------|------|
| **TTFT Reduction** | 平均降低 **17.76%**（最高达 18.7% in Central 分布） |
| **TPOT Reduction** | 平均降低 **13.34%**（最高达 17.6% in Central） |
| **Throughput** | 与 vLLM **相当**，无明显下降，说明性能提升非牺牲吞吐换得 |
| **Prefix Cache Hit Count** | 平均提升 **3%**（18,451 → 18,992） |
| **Prefix Cache Hit Rate** | 平均从 3.64% 提升至 **3.80%**（+4.4%） |

> 所有结果基于超过 100 次实验，在五种 workload 分布下稳定复现。

---

### 🔬 消融实验分析（Ablation Study）
- **DPLB 贡献最大**：对 TTFT 改善起主导作用，因其有效平衡了引擎间的 KV Cache 和运行负载。
- **SJFS 显著改善尾延迟**：尤其在混合长短请求场景中，减少 HoL blocking。
- **EDR 对 TPOT 更敏感**：通过缓解专家热点和优化通信局部性，降低解码阶段延迟。
- **三者协同增益明显**：Gimbal 综合三模块后效果最优，表明多层调度具有互补性。

> 示例：在 Two-end 分布下，Gimbal 比单独 DPLB 再降 3.2% TTFT。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **多层协同调度显著优于单层优化**：
   - 单独优化任一层无法充分释放 MoE 模型潜力。
   - 只有联合优化 request、engine、expert 三层才能全面降低 TTFT 与 TPOT。

2. **负载感知调度至关重要**：
   - 忽略 KV Cache 或 RunningLoad 会导致严重负载倾斜，影响整体性能。

3. **用户粘性可有效提升缓存效率**：
   - 将同一用户的请求路由到相同 engine，能显著提高 prefix cache hit rate，减少冗余计算。

4. **跨层专家依赖是真实存在的且可利用的**：
   - 实验验证 Qwen3 模型中存在强 inter-layer expert affinity。
   - 利用该特性进行 co-location 可减少跨 GPU 通信，提升推理效率。

---

### ⚠️ 局限性
1. **实验环境受限**：
   - 仅在 2×A100 单机环境下测试，未扩展至大规模分布式集群。
   - 硬件同质性强，缺乏异构设备支持验证。

2. **专家重定位开销较高**：
   - 当前 EDR 模块依赖离线构建的 affinity matrix，需额外数据收集与迁移成本。
   - 动态在线学习 affinity 的机制尚未实现。

3. **未支持更复杂 workload 类型**：
   - 如 streaming generation、function calling、tool use 等 API-augmented 场景未覆盖。

---

### 🔮 未来工作方向
1. **扩展至更大规模集群**：支持跨节点 expert parallelism 与分布式调度。
2. **轻量化在线 affinity 学习**：设计低开销的实时专家依赖检测机制。
3. **自适应阈值调节**：根据 workload 动态调整 `θ_k`, `θ_diff`, `θ_age` 等参数。
4. **融合 preemption 与 phase-aware 调度**：进一步区分 prefill 与 decode 阶段资源需求。
5. **探索 RL-based 多目标调度器**：联合优化延迟、吞吐、能耗等多重目标。

---

## ✅ 总结
本文提出的 **Gimbal** 是首个面向 MoE-based LLM 的**多层协同调度框架**，通过在 request、engine、expert 三个层级引入精细化调度策略，实现了：
- 最高 **17.8% 的 TTFT 下降**
- **13.3% 的 TPOT 降低**
- 同时保持与 vLLM 相当的 throughput
- 显著提升 prefix cache 利用率

实验证明，**协调多层调度决策** 是提升 MoE 模型推理效率的关键路径，为未来 LLM Serving 系统设计提供了重要范式参考。

</details>

---

### 3. [C$^{2}$TC: A Training-Free Framework for Efficient Tabular Data Condensation](https://arxiv.org/abs/2602.21717)

**Authors**: Sijia Xu, Fan Li, Xiaoyang Wang, Zhengyi Yang, Xuemin Lin  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2602.21717v1  

#### Abstract
Tabular data is the primary data format in industrial relational databases, underpinning modern data analytics and decision-making. However, the increasing scale of tabular data poses significant computational and storage challenges to learning-based analytical systems. This highlights the need for ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：C²TC: A Training-Free Framework for Efficient Tabular Data Condensation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Dataset Condensation (DC)** 方法在处理**表格数据 (tabular data)** 时面临三大挑战：
- **高计算成本 (High computational cost)**：依赖复杂的梯度优化过程（如参数匹配或分布匹配），导致训练开销巨大，难以扩展到大规模工业级表格数据。
- **类别不平衡下的效用损失 (Utility loss under class imbalance)**：传统标签分配策略（如比例保留 Ratio 或每类固定样本数 FIPC）无法有效平衡多数类与少数类的表现，导致 Macro-F1 性能下降。
- **异构特征的信息丢失 (Information loss in heterogeneous features)**：现有方法忽视了对分类变量（categorical features）的有效编码，导致语义信息丢失。

### 提出的新方法与思路
作者提出了 **C²TC (Class-Adaptive Clustering for Tabular Condensation)** ——首个**无需训练 (training-free)** 且支持**自适应标签分配 (label-adaptive)** 的表格数据压缩框架。其核心思想包括：

- **将 DC 重构为组合优化问题**：通过理论分析统一现有 DC 方法的目标函数，并将其转化为一个**类自适应聚类分配问题 (Class-adaptive Cluster Allocation Problem, CCAP)**，从而摆脱对模型训练和梯度计算的依赖。
- **提出 HFILS 算法求解 NP-hard 的 CCAP**：设计了一种启发式的首次改进局部搜索算法（Heuristic First-Improvement Local Search, HFILS），交替进行软分配与类内聚类，高效逼近高质量解。
- **引入混合分类特征编码 (HCFE)**：针对字符串型和整数型分类变量分别采用相似性编码（similarity encoding）和平滑目标编码（smoothed target encoding），实现语义保持的数值化表示，提升聚类质量。

### 相比现有方法的优势
| 维度 | C²TC | 现有 DC 方法（如 DM/GM/MTT） |
|------|------|-------------------------------|
| **效率** | 至少快两个数量级，无 OOM/OTT 问题 | 高内存/时间消耗，常因 OOM 或超时失败 |
| **通用性** | 不依赖特定模型架构，跨模型泛化能力强 | 通常依赖 relay model，存在架构偏差 |
| **公平性** | 动态调整各类别压缩规模，显著提升 Macro-F1 | 固定分配策略，加剧类别不平衡 |
| **可扩展性** | 支持百万级样本和千维特征的大表 | 在大表上不可行 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共使用 **10 个真实世界表格数据集**，来自 OpenML benchmark，涵盖多种领域和规模：
- **Energy**: `Electricity` (EL)
- **Income**: `Adult` (AD)
- **Medical**: `Diabetes130US` (DA), `Covertype` (CO)
- **Flight Delay**: `Airlines` (AI)
- **Physics**: `Higgs` (HI)
- **Synthetic/Large-scale**: `Epsilon` (EP), `Microsoft` (MI)

这些数据集具有以下特点：
- 样本量从 $10^4$ 到 $>10^6$
- 特征维度从几十到两千
- 包含数值型（numerical）和分类型（categorical）混合特征
- 存在不同程度的类别不平衡（如 `JA`, `CO`, `MI`）

### 实验设置与评估指标
#### 数据划分
所有数据集按 80%/10%/10% 划分为训练/验证/测试集。

#### 压缩比率 (Condensation Ratio)
测试多个压缩率：`1%`, `0.1%`, `0.01%`（即原始数据的 1‰）

#### 评估流程
1. **Condensation Stage**：在训练集上应用 DC 方法生成小规模合成数据。
2. **Evaluation Stage**：用合成数据从零开始训练深度表格模型，在原始测试集上评估性能。

#### 评估指标
- **Accuracy (Acc)**：整体预测准确率
- **Macro-F1 (MF1)**：各类别 F1 分数的平均值，反映类别均衡能力

#### 深度表格模型（用于下游任务）
- **MLP-based**: MLP, MLP_PLR, RealMLP  
- **Transformer-based**: FT-Transformer (FT_T), TabNet, TabR

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table II 和 III）

| 数据集 | 方法 | Acc (%) @0.1% | MF1 (%) @0.1% | 是否成功运行 |
|-------|------|----------------|----------------|-------------|
| AD | C²TC | **81.5** | **70.8** | ✅ |
| AD | MTT | 76.2 | 56.0 | ✅ |
| EP | C²TC | **75.2** | **74.8** | ✅ (<72s) |
| EP | GM/MTT | ❌ (OOM/OTT) | ❌ | ⛔ |
| CO | C²TC | **69.6** | **46.4** | ✅ |
| CO | MTT | 66.7 | 38.3 | ✅ |
| MI | C²TC | **53.6** | **21.5** | ✅ (39.2s) |
| MI | MTT | 53.4 | 19.9 | ⚠️ (78h+) |

> 注：C²TC 在 **8/10 数据集上取得最高 Accuracy**，在 **29/30 实验场景下获得最佳 Macro-F1**

### 与基线方法的对比结果

#### 下游性能全面领先
- 平均 Accuracy 超过最强基线 MTT 达 **3.1%**（AD 上）
- 在高度不平衡数据（如 CO, JA）上，Macro-F1 提升达 **9.8%**
- 即使在极低压缩率（0.01%）下，仍能恢复超过 86% 的全数据性能（7/10 数据集）

#### 效率优势极其显著
- **速度提升 ≥ 100 倍**（见 Figure 3 & 4）
  - 在最大数据集 MI 上：C²TC 仅需 **39.2 秒**，而 MTT 超过 **78 小时**
  - 在高维数据 EP 上：C²TC < **72 秒**，其他方法全部 OOM/OTT
- 时间复杂度为 $O(T \cdot N \cdot F)$，随数据规模线性增长，具备良好可扩展性

#### 跨架构泛化能力强（Table III）
- 在不同模型（MLP / Transformer）上均表现稳定
- C²TC 在 EP 上使 FT-Transformer 准确率从 50% 提升至 **83.7%**
- 显著优于 Random、DM、GM 等基线，证明其生成数据具有强迁移性

### 消融实验结果（Ablation Study）

#### 不同标签分配策略比较（Table IV）
| 策略 | Acc ↑ | MF1 ↑ |
|------|--------|--------|
| Ratio (比例保留) | 80.4 (AD) | 69.7 |
| FIPC (每类等量) | 74.2 | **70.7** |
| **C²TC (自适应)** | **81.5** | **70.8** |

👉 结论：C²TC 同时兼顾整体精度与类别公平性，优于任何静态策略。

#### 模块消融（Table V）
移除关键组件后性能大幅下降：
- **w/o Scale Factor**：Macro-F1 下降最多达 **10.6%**（CO 上），说明重加权机制对缓解类别偏倚至关重要。
- **w/o Soft Step Size**：Accuracy 下降明显（如 DA 上从 51.8 → 36.8），表明随机步长有助于跳出局部最优。
- **w/o Soft Allocation**：多类数据（CO, MI）性能显著降低，说明多目标再分配更利于全局探索。

#### 参数敏感性分析（Figure 6）
- **权重指数 $\gamma$**：
  - $\gamma=0.25$：推荐默认值，平衡 Acc 与 MF1
  - $\gamma=1.0$：适用于极端不平衡场景，优先保护少数类
- **步长衰减因子 $l$**：
  - 推荐 $l=0.5$，在探索与利用间取得平衡
  - 大规模多类数据可适当增大以避免早收敛

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **训练自由是可行且高效的**：通过将 DC 重新建模为聚类分配问题，完全去除梯度更新，实现了前所未有的效率提升。
2. ✅ **动态标签分配至关重要**：固定的 Ratio 或 FIPC 策略无法应对现实中的类别不平衡；C²TC 的自适应机制能自动向信息增益更高的类别倾斜资源。
3. ✅ **异构特征需专门处理**：HCFE 编码策略有效保留了分类变量的语义信息，实验证明其优于 Label Encoding、One-hot + PCA 和 Target Encoding。
4. ✅ **极高压缩比下仍具实用性**：即使只保留 0.01% 的数据，C²TC 也能恢复大部分原始性能，特别适合资源受限场景。

### 方法的局限性
- **依赖 K-means 聚类假设**：假设类内数据呈凸形分布，可能不适用于高度非线性结构的数据。
- **线性编码器近似**：虽然提升了效率和解释性，但在某些复杂模式识别任务中可能牺牲部分表达能力。
- **未考虑结构关系**：仅适用于独立同分布（i.i.d.）表格数据，不能直接推广到图结构或序列数据。

### 未来工作方向
- 扩展至 **continual learning** 场景，支持增量式数据压缩
- 探索 **non-Euclidean 聚类方法**（如谱聚类）以捕捉更复杂的类内结构
- 将 C²TC 与 **neural architecture search (NAS)** 结合，构建端到端高效 AutoML 流程
- 研究如何将该框架应用于 **tabular-textual multimodal data** 的联合压缩

--- 

> 📌 **一句话总结**：  
> C²TC 是首个真正意义上的**免训练、高效、可扩展且公平**的表格数据压缩框架，通过将 DC 转化为类自适应聚类问题，在效率上超越现有方法两个数量级以上，同时在下游任务中取得更优且更均衡的性能表现。

</details>

---

### 4. [ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning](https://arxiv.org/abs/2602.21534)

**Authors**: Xiaoxuan Wang, Han Zhang, Haixin Wang, Yidan Shi, Ruoyan Li, Kaiqiao Han, Chenyi Tong, Haoran Deng, Renliang Sun, Alexander Taylor, Yanqiao Zhu, Jason Cong, Yizhou Sun, Wei Wang  
**Category**: cs.AI  
**Published**: 2026-02-26  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2602.21534v1  

#### Abstract
Agentic reinforcement learning (ARL) has rapidly gained attention as a promising paradigm for training agents to solve complex, multi-step interactive tasks. Despite encouraging early results, ARL remains highly unstable, often leading to training collapse. This instability limits scalability to lar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ARLArena: A Unified Framework for Stable Agentic Reinforcement Learning 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

Agentic Reinforcement Learning (ARL) 是训练大语言模型（LLM）作为自主代理以完成复杂、多步交互任务（如网页导航、具身环境操作、数学推理等）的关键范式。然而，现有的 ARL 训练过程**高度不稳定**，常出现训练崩溃（training collapse），表现为性能骤降、梯度爆炸、格式错误激增等问题。这种不稳定性严重限制了 ARL 在更大环境和更长交互序列中的扩展，并阻碍了对算法设计选择的系统性探索。

### **提出了什么新方法或新思路**

为解决上述问题，本文提出 **ARLArena** —— 一个用于稳定 ARL 训练的统一框架，其核心创新包括：

- **标准化测试平台（Standardized Testbed）**：通过行为克隆（Behavior Cloning）、格式惩罚（Format Penalty）、KL 正则化（KL Regularization）和超参数搜索，构建了一个干净、可复现的基准环境，有效消除了早期训练噪声。
  
- **四维策略梯度分解（Four-Dimensional Policy Gradient Decomposition）**：将基于 PPO 的 ARL 方法解耦为四个正交的设计维度进行独立分析：
  1. **Loss Aggregation**（损失聚合）
  2. **Importance Sampling (IS) Clipping**（重要性采样裁剪）
  3. **Trajectory Filtering and Resampling**（轨迹过滤与重采样）
  4. **Advantage Design**（优势函数设计）

- **提出 SAMPO（Stable Agentic Multi-turn Policy Optimization）**：一种新的 PO 方法，整合了在各维度中被验证最稳定的组件，形成统一的稳定训练方案。

### **相比现有方法的优势**

- **高稳定性**：SAMPO 实现了单调、无崩溃的训练过程，在多个任务上均表现出色。
- **强泛化性**：在 ALFWorld、WebShop、Sokoban 和 TIR Math 四个差异显著的任务上均取得最优性能。
- **可复现性**：提供了一套完整的“清洁训练配方”（clean training recipe），极大提升了 ARL 实验的可复现性。
- **理论指导意义**：通过系统性分析，提炼出关于 ARL 稳定性的普适性原则，为后续研究提供了明确的设计指南。

---

## 2. 核心实验方法和设置

### **使用的数据集**

实验在四个代表性的 ARL 任务上进行：

| 数据集 | 任务描述 |
|--------|---------|
| **ALFWorld** | 文本型具身环境，代理需在虚拟家庭环境中执行多步任务（如“把冷却的鸡蛋放进微波炉”）。 |
| **WebShop** | 模拟电商购物环境，代理需根据用户需求搜索并购买商品。 |
| **Sokoban** | 经典推箱子游戏，视觉输入，代理需规划路径推动所有箱子到目标点。 |
| **TIR Math** | 数学推理任务，代理可通过调用 Python 工具进行计算，评估其逐步推理能力。 |

### **实验设置和评估指标**

- **模型基础**：主要使用 Qwen3-4B-SFT 模型进行微调，部分实验验证了 Qwen3-8B 的可扩展性。
- **训练框架**：基于 verl RL 框架实现，采用多轮交互架构（agentic-loop）。
- **评估指标**：
  - **Success Rate**：任务成功完成率（ALFWorld, WebShop, Sokoban）。
  - **Score / Pass@k**：任务得分或准确率（TIR Math）。
  - **Training Dynamics**：监控成功率、KL 散度、梯度范数、有效格式比例等动态指标。

### **基线方法对比**

对比了多种主流 PO 方法，按设计维度分类：

| 方法 | 所属维度 | 特点 |
|------|----------|------|
| GRPO | 基线 | 标准 PPO 风格，token-level IS clipping |
| GRPOsT | Loss Agg | sequence-mean-token-mean 损失聚合 |
| SAPO | IS Clipping | Soft adaptive clipping |
| CISPO | IS Clipping | Stop-gradient clipping |
| GSPO | IS Clipping | **Sequence-level IS clipping** |
| GIGPO | Advantage Design | Hierarchical advantage (episode + step-level) |
| EMPG | Advantage Design | Entropy-modulated advantage |
| DAPO | Dynamic Sampling | 动态过滤零梯度轨迹 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

在 **Qwen3-4B** 上的平均性能（Table 3）：

| 方法 | Avg Score | Avg Improvement vs GRPO |
|------|----------|------------------------|
| **GRPO** | 46.16 | — |
| **GSPO** | 52.28 | +13.3% |
| **GIGPO** | 49.71 | +7.7% |
| **DAPO+GIGPO** | 53.36 | +15.6% |
| **SAMPO (Ours)** | **60.21** | **+25.2%** |

在 **ALFWorld** 上的成功率（最高达 **92.72%**），远超闭源模型如 GPT-5.2 (51.56%) 和 o3-based MAS (56.25%)。

### **与基线方法的对比结果**

- **SAMPO 全面领先**：在所有任务上均取得最高分，且训练曲线平滑稳定（见 Figure 2）。
- **GSPO 显著优于其他 IS 方法**：证明 **sequence-level clipping** 比 token-level clipping 更稳定。
- **DAPO 效果依赖 Advantage Design**：仅当与 GIGPO 结合时才显著提升，单独使用可能因丢失格式信号而退化。

### **消融实验结果**

- **IS Clipping 分析**（Figure 3 & 4）：
  - **Tolerant Clipping**（如 SAPO, CISPO）初期上升快，但很快因负优势低 IS 比率样本积累导致崩溃。
  - **Sequence-level Clipping**（GSPO）能有效抑制有害轨迹，保持 KL 和梯度稳定。
- **Stabilization Strategies for SAPO/CISPO**（Table 4）：
  - 增加 KL 系数或增大 batch size 效果有限。
  - **Sequence Masking**（屏蔽负优势且低 IS 的序列）可将 SAPO 成功率从 25.16% 提升至 **76.92%**，接近 GSPO。
- **Loss Aggregation**：
  - `seq-mean-token-mean`（GRPOsT）在长度方差大的任务（如 TIR Math）上表现差，因其对长短序列加权不均。
- **Off-policy Staleness**（Table 5）：
  - 高离策略程度（high off-policy staleness）显著降低性能，表明 ARL 对数据新鲜度敏感。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **IS Clipping 是决定性因素**：
   - **Tolerant clipping** 导致快速训练崩溃。
   - **Sequence-level clipping** 是实现长期稳定训练的关键。

2. ✅ **Advantage Design 提供稳定增益**：
   - 引入环境级细粒度信息（如 GIGPO 的 step-level advantage）能改善信用分配，缓解稀疏奖励问题。

3. ✅ **Dynamic Filtering 需与强 Advantage 配合**：
   - 单独使用可能削弱格式学习信号，但与 GIGPO 结合可进一步提升性能。

4. ❌ **Loss Aggregation 影响有限**：
   - 在所测场景下，不同聚合方式对最终性能影响较小。

5. 🔍 **训练崩溃的根本原因**：
   - 主要由 **负优势且低 IS 比率的序列** 驱动，这些样本导致梯度爆炸和策略漂移。

### **方法的局限性**

- **依赖高质量 SFT 初始化**：行为克隆阶段需要大量高质量的人类或模型生成轨迹。
- **未解决根本的探索效率问题**：虽然 SAMPO 稳定了训练，但在极端稀疏奖励环境下仍可能难以探索到成功轨迹。
- **计算成本较高**：完整的 ARLArena 流程（含超参搜索）需要大量 GPU 资源。

### **未来工作方向**

1. **Clean Training Recipes 为核心**：应将初始化和早期训练配方视为算法核心，而非辅助技巧。
2. **IS Clipping 是高风险高回报方向**：微小改动可能导致巨大稳定性差异，需谨慎设计。
3. **Advantage Design 是稳健增益来源**：更适合追求可预测性能提升的研究。
4. **稳定 ARL 支持长视野扩展**：一旦训练稳定，可探索更长交互序列和更大环境空间，类似监督预训练的 scaling law。
5. **探索更高效的离策略修正机制**：以应对批量 rollout 带来的 off-policy staleness 问题。

---

> **总结**：ARLArena 通过系统性解构和实证分析，揭示了 ARL 不稳定的核心根源，并提出了 **SAMPO** 这一融合了 sequence-level clipping、fine-grained advantage 和 dynamic filtering 的统一稳定方案。该工作不仅提供了当前最优的 ARL 训练方法，更重要的是建立了一套可复现、可分析的科学框架，为未来 ARL 研究奠定了坚实基础。

</details>

---

### 5. [Energy Efficient Federated Learning with Hyperdimensional Computing over Wireless Communication Networks](https://arxiv.org/abs/2602.21949)

**Authors**: Yahao Ding, Yinchao Yang, Jiaxiang Wang, Zhaohui Yang, Dusit Niyato, Zhu Han, Mohammad Shikh-Bahaei  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.21949v1  

#### Abstract
In this paper, we investigate a problem of minimizing total energy consumption for secure federated learning (FL) over wireless edge networks. To address the high computational cost and privacy challenges in conventional FL with neural networks (NN) for resource-constrained users, we propose a novel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Energy Efficient Federated Learning with Hyperdimensional Computing over Wireless Communication Networks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**无线边缘网络中联邦学习（Federated Learning, FL）面临的两大核心挑战**：
- **高能耗问题**：资源受限的边缘设备在执行传统基于神经网络（NN）的FL时，本地计算和无线通信开销巨大，导致电池快速耗尽。
- **隐私泄露风险**：尽管FL不共享原始数据，但模型更新仍可能被用于梯度反演等攻击，暴露敏感信息。

现有方法通常分别优化资源分配或引入隐私机制，缺乏对**模型结构、隐私保护与系统资源**的联合建模与协同优化。

---

### 提出的新方法与新思路
作者提出了一种全新的框架——**FL-HDC-DP**（Federated Learning with Hyperdimensional Computing and Differential Privacy），其核心创新如下：

#### ✅ 新型计算范式：Hyperdimensional Computing (HDC)
- 用**超维计算**替代传统的深度神经网络进行本地训练。
- HDC通过简单的**hypervector操作**（如bundling, binding）实现快速编码与推理，显著降低CPU周期消耗。
- 特别适合低功耗、实时性要求高的边缘设备。

#### ✅ 隐私增强机制：Differential Privacy (DP)
- 在客户端上传前向本地AM（Associative Memory）添加符合zCDP（zero-Concentrated DP）标准的高斯噪声。
- 提供严格的数学可证明隐私保障，抵御服务器端和窃听者的推理攻击。

#### ✅ 联合优化框架：多变量协同设计
首次将以下五个维度纳入统一优化问题：
- HDC维度 $d$
- 传输时间 $t_i$
- 带宽分配 $b_i$
- 发射功率 $p_i$
- CPU频率 $f_i$

目标是在满足**延迟约束**和**$(\epsilon, \delta)$-DP隐私预算**的前提下，最小化所有用户的**总能量消耗**。

#### ✅ 创新的收敛轮次建模
提出一个**sigmoid-variant函数**来拟合HDC维度$d$与达到目标精度所需收敛轮数$J_d$之间的非线性关系：
$$
J_d(d) = \mu + \frac{\nu}{1 + e^{\beta(\log d - \alpha)}}
$$
该模型能准确反映“维度增加 → 收敛加快 → 达到饱和”的物理特性。

#### ✅ 可行初始化策略
为确保交替优化算法有合法起点，提出一种**逐轮传输时间最小化**的方法构造初始可行解，并给出闭式表达式。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **能效性** | 总能耗相比基线最多降低 **83.3%** |
| **收敛速度** | 达到相同准确率所需的通信轮次仅为NN基线的约 **1/3.5** |
| **硬件友好性** | HDC操作简单、并行度高，更适合嵌入式部署 |
| **隐私保障** | 引入严格DP机制，优于仅依赖模型脱敏的传统FL |

---

## 2. 核心实验方法和设置

### 数据集
- 主要使用 **MNIST 手写数字图像数据集**
- 考虑两种数据分布场景：
  - **IID**：60,000张图像均匀分给50个用户（每人1200张）
  - **Non-IID**：按标签排序后划分为150个碎片（shards），每个用户随机分配3个，造成严重的类别偏斜

---

### 实验设置
| 参数 | 设置值 |
|------|--------|
| 用户数量 $U$ | 50 |
| 网络拓扑 | 单小区，BS位于中心，用户分布在半径500m圆内 |
| 信道模型 | 大尺度路径损耗 $L(d) = 128.1 + 37.6 \log_{10}(d)$ |
| 上行接入方式 | FDMA（频分多址） |
| 总带宽 $B$ | 10 MHz |
| 最大发射功率 $P_{\text{max}}$ | 1 mW |
| CPU最大频率 $f_{\text{max}}$ | 2.3 GHz |
| 时间预算 $T$ | 30 秒 |
| 隐私参数 $(\epsilon, \delta)$ | $(20, 10^{-5})$ 或 $(25, 10^{-5})$ |
| HDC维度范围 | 3000 ~ 10000（步长1000） |

---

### 评估指标
- **测试准确率（Accuracy）**
- **总能量消耗 $E$**（含计算 + 通信）
- **收敛轮次 / Epoch 数**
- **资源分配结果**：带宽、功率、CPU频率、传输时间

---

### 基线方法对比
| 基线名称 | 描述 |
|---------|------|
| **FL-NN-DP** | 传统神经网络 + 差分隐私，作为主要对比基准 |
| **Fixed-d Baseline** | 固定HDC维度（如 d=3000 或 d=5000），其他资源优化 |
| **Fixed-p Baseline** | 固定发射功率 $p = P_{\text{max}}$，其余变量优化 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **最终准确率** | FL-HDC-DP 达到约 **90%**（IID下） |
| **收敛轮次减少比例** | 比 FL-NN-DP 少 **约3.5倍** 的通信轮次 |
| **最大节能效果** | 相比固定维度基线，总能耗降低 **83.3%** |
| **最优HDC维度** | 实验中发现最佳维度为 **d=4000**（非越大越好） |

---

### 与基线方法的对比结果
#### 📈 准确率 vs. 通信轮次（图4）
- 在 $\epsilon=20$ 下：
  - FL-HDC-DP 在 **40轮左右达到90%准确率**
  - FL-NN-DP 需要 **超过140轮才能达到相同水平**
- 在 Non-IID 场景下，两者性能均下降，但 FL-HDC-DP 依然领先。

#### ⚡ 能量 vs. 发射功率（图7）
- 当 $P_{\text{max}}$ 较小时，能量随功率提升迅速下降；
- 提出的方法始终优于两个固定基线，在 $P_{\text{max}}=10^{-3}$W 时节能达 **40%~46%**

#### 📊 能量 vs. 总带宽（图8）
- 增加带宽可有效降低能量，尤其在低带宽区域（<2MHz）改善明显；
- 提出的联合优化方案在各种带宽下均表现最优，最高节能 **83.3%**

---

### 消融实验结果（隐含分析）
虽然未明确列出消融表，但从实验设计中可推断以下结论：
- **HDC维度选择影响最大**：优化 $d$ 同时影响计算与通信成本，是节能的关键；
- **资源联合优化带来增量收益**：单独优化带宽或功率只能缓解瓶颈，无法根本改变能效曲线；
- **DP噪声的影响**：加入DP后准确率略有下降且收敛更震荡，但更高维度仍有助于缓解这一影响（见图3b）；

---

## 4. 关键结论和发现

### 主要发现
1. **HDC是轻量化FL的理想候选者**  
   其基于简单向量运算的机制天然具备**低计算复杂度、高鲁棒性和快速收敛能力**，非常适合边缘设备。

2. **维度存在“黄金平衡点”**  
   并非维度越高越好。虽然大维度加速收敛，但也增加了每轮的计算与通信负载。实验表明 **d=4000 是当前设置下的最优折衷点**。

3. **联合优化显著优于单点优化**  
   单独调整带宽或功率虽能改善部分性能，但只有同时优化模型维度与系统资源，才能实现全局能效最大化。

4. **远距离用户更“耗能”**  
   由于信道质量差，远离基站的用户需要更多带宽和更高功率补偿SNR损失，同时压缩本地计算时间，迫使提高CPU频率，从而整体能耗更高（见图9）。

---

### 方法的局限性
| 局限性 | 说明 |
|-------|------|
| **适用任务有限** | HDC目前主要用于分类任务（如MNIST），在复杂视觉、自然语言处理任务上的表现尚待验证 |
| **模型容量限制** | 超维向量虽高效，但在处理大规模、高语义抽象任务时可能不如深度模型灵活 |
| **静态环境假设** | 实验假设信道稳定、用户位置不变，动态移动场景下的适应性需进一步研究 |
| **初始化依赖** | 算法需要可行初始解，若初始维度不可行则需重新尝试，可能影响实用性 |

---

### 未来工作方向
1. **扩展至更复杂的HDC架构**  
   探索支持序列建模、注意力机制的HDC变体，以应对语音识别、时间序列预测等任务。

2. **动态资源调度机制**  
   设计在线自适应算法，根据实时信道状态和设备电量动态调整维度与资源分配。

3. **跨层联合优化**  
   将HDC编码方式、DP噪声注入策略也纳入优化空间，形成端到端的自动化设计流程。

4. **硬件协同设计**  
   开发专用HDC处理器（如存内计算芯片），进一步释放其能效潜力。

5. **应用于6G与Metaverse场景**  
   如文中引用[17]所述，探索HDC与大模型融合，构建面向未来 **Integrated Learning and Communication (ILAC)** 架构。

---

> ✅ **一句话总结**：  
> 本文提出的 **FL-HDC-DP** 框架通过结合**超维计算的高效性**与**差分隐私的安全性**，并首次实现了对**模型维度与通信/计算资源的联合优化**，在保证高准确率的同时，将联邦学习的总能耗降低了**高达83.3%**，为绿色、安全、可持续的边缘智能提供了全新路径。

</details>

---

### 6. [NGDB-Zoo: Towards Efficient and Scalable Neural Graph Databases Training](https://arxiv.org/abs/2602.21597)

**Authors**: Zhongwei Xie, Jiaxin Bai, Shujie Liu, Haoyu Huang, Yufei Li, Yisen Gao, Hong Ting Tsang, Yangqiu Song  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.21597v1  

#### Abstract
Neural Graph Databases (NGDBs) facilitate complex logical reasoning over incomplete knowledge structures, yet their training efficiency and expressivity are constrained by rigid query-level batching and structure-exclusive embeddings. We present NGDB-Zoo, a unified framework that resolves these bott...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NGDB-Zoo: Towards Efficient and Scalable Neural Graph Databases Training

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

Neural Graph Databases (NGDBs) 虽然在复杂逻辑推理任务中展现出潜力，但在实际应用中面临两大瓶颈：

- **计算效率与拓扑刚性**：传统基于 query-level batching 的训练方式要求同一批次内的查询具有相同的拓扑结构（如 2p、3i），导致真实世界中多样化查询混合时硬件利用率低下，尤其在大规模稀疏图上收敛困难。
- **表示摩擦（Representation Friction）**：将 Pre-trained Text Encoders (PTEs) 的高维语义向量集成到结构嵌入中虽能提升泛化能力，但会引发严重的 I/O 阻塞和内存溢出，造成训练延迟。

### 提出了什么新方法或新思路

作者提出 **NGDB-Zoo**，一个统一的高效可扩展神经图数据库训练框架，其核心创新如下：

#### ✅ Contribution 1: Operator-Level Training Paradigm（操作符级训练范式）
- 将复杂查询分解为由原子操作符（如 Project、Intersect、Union）构成的有向无环图（DAG）。
- 引入 **operator-level batching**，动态调度不同类型的操作符进行批量执行，打破 query-level batching 的拓扑隔离限制。
- 采用 **Max-Fillness 调度策略**，优先执行负载最高的操作符池，最大化 GPU 利用率。

#### ✅ Contribution 2: Decoupled Semantic Integration Architecture（解耦语义集成架构）
- 提出 **离线预编码 + GPU 内存驻留缓存** 策略：使用冻结的 PTE（如 Qwen3-Embedding、BGE）预先生成实体文本嵌入，并将其作为只读缓冲区加载至 GPU 的 High Bandwidth Memory (HBM)。
- 在训练过程中，语义特征融合退化为高效的 `Gather` 操作，实现“推理自由”（inference-free）训练。

#### ✅ Contribution 3: Scalable Neuro-Symbolic Benchmarking
- 在六个标准 benchmark 上进行了全面评估，涵盖从中小规模（FB15k）到超大规模图（ATLAS-Wiki-Triple-4M，含 400 万实体）。
- 展示了框架在多跳逻辑查询（multi-hop logical queries）上的高吞吐与强表达力平衡能力。

### 相比现有方法的优势

| 维度 | NGDB-Zoo | 传统方法 |
|------|---------|--------|
| 批处理粒度 | Operator-level | Query-level |
| 硬件利用率 | 高（接近峰值） | 低（碎片化严重） |
| 语义集成开销 | 极低（仅内存查找） | 高（实时 PTE 推理） |
| 可扩展性 | 支持千万级边图 | 受限于 batch 多样性 |
| 吞吐量 | 提升 1.8×–6.8× | 基线水平 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

共使用 **6 个 benchmark 数据集**，覆盖不同规模与密度：

| 数据集 | 实体数 | 关系数 | 总边数 | 特点 |
|-------|--------|--------|--------|------|
| FB15k | 14.9K | 1,345 | ~592K | 早期标准数据集 |
| FB15k-237 | 14.5K | 237 | ~310K | 更难，去反向关系 |
| NELL995 | 63.4K | 200 | ~142K | 归纳学习场景 |
| FB400k | 410K | 918 | ~2.15M | 大规模稀疏图 |
| ogbl-wikikg2 | 2.5M | 535 | ~17.1M | OGB 大图基准 |
| ATLAS-Wiki-Triple-4M | 4.04M | 512K | ~28.8M | 超大规模工业级图 |

> 注：部分数据集通过 degree-weighted edge sampling 构建子图用于测试。

### 实验设置和评估指标

#### 模型后端（Backbone Models）
- **GQE**, **Q2P**: 向量空间表示
- **Q2B**, **BetaE**: Beta 分布建模不确定性
- **FuzzQE**: 模糊逻辑推理

#### 语义编码器（PTEs）
- **Qwen3-Embedding-0.6B**
- **BGE-Base-En-v1.5**

#### 评估指标
- **MRR (%)**：Mean Reciprocal Rank，衡量排序质量
- **Throughput (Queries/sec)**：每秒处理的查询数量
- **Peak GPU Memory (GB)**：显存占用峰值
- **Multi-GPU Scalability**：加速比

#### 训练配置
- 框架：PyTorch 2.0 + Python 3.10
- 硬件：NVIDIA A6000（48GB VRAM），40 核 CPU
- Batch Size：512 queries
- Optimizer：Adam (lr=1e-4)

### 基线方法对比
- **SQE** (Bai et al., 2023)
- **SMORE** (Ren et al., 2022)
- **DGL-KE** (Zheng et al., 2020)
- **PyTorch-BigGraph (PBG)** (Lerer et al., 2019)
- **Marius** (Mohoney et al., 2021)

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔥 吞吐量显著提升（Table 3 & Figure 7）

| 场景 | 加速比 |
|------|--------|
| FB15k + BetaE vs. SQE | **7.0×** |
| 平均训练吞吐（vs. SQE） | **5.0×** |
| 多跳推理整体加速 | **1.8× – 6.8×** |

> 在 ATLAS-Wiki-Triple-4M 上仍保持 >17K queries/sec 的高吞吐。

#### 📈 多 GPU 可扩展性强（Figure 7）
- 在 ogbl-wikikg2 和 ATLAS-Wiki 上，1→8 GPU 几乎呈 **线性加速**。
- NGDB-Zoo 在 8 GPU 下达到 **~25K queries/sec**，远超其他系统。

#### ⚙️ 单跳链接预测速度领先（Table 2）
| 系统 | 1-GPU Epoch Time (s) |
|------|---------------------|
| PBG | 3060 |
| SMORE | 760 |
| **NGDB-Zoo** | **628** |

> 显示底层优化有效，适合大规模 KG completion。

### 与基线方法的对比结果（Table 3）

| 指标 | NGDB-Zoo 表现 |
|------|---------------|
| **MRR (%)** | 匹配或优于基线（如 FB15k 上 BetaE 达 43.04%） |
| **Training Throughput** | 平均 **5086 queries/sec**，是 SMORE 的 ~1.8×，SQE 的 ~3.9× |
| **GPU Memory** | 略高于 SMORE，但低于 SQE（因缓存机制更优） |

> 尤其在 **Q2P** 和 **FuzzQE** 模型上，吞吐提升最为明显（最高达 3.5×）。

### 消融实验结果（Ablation Studies）

#### ✅ Operator-Level Batching 效果（Table 6）
| Operator | Speedup |
|--------|---------|
| EmbedE | 2.88× |
| Project | 3.74× |
| **Intersect** | **13.11×** |
| **Union** | **12.22×** |

> Intersect/Union 因输入可变且计算密集，批处理收益最大。

#### ✅ Decoupled Semantic Integration（Figure 8 & Table 8）
| 指标 | 提升效果 |
|------|----------|
| **Throughput** | **5×–7× 加速**（移除 PTE 实时推理） |
| **Memory Usage** | **下降**（卸载 PTE 参数，仅保留 embedding buffer） |
| **MRR** | **平均 +4.74%**（利用语言先验增强稀疏图推理） |

> 证明语义增强不仅不拖慢训练，反而大幅提升效率与性能。

#### ✅ Adaptive Online Sampling（Figure 9）
- 引入动态采样策略，在非平稳查询分布下（突发难题）MRR 提升 **21.5%**。
- 支持 curriculum learning，主动聚焦复杂多跳路径。

---

## 4. 关键结论和发现

### 主要发现

1. **Operator-level batching 是突破 NGDB 训练瓶颈的关键**：
   - 打破 query-level batching 的拓扑刚性，实现跨查询的操作符融合。
   - 动态调度（Max-Fillness）使 GPU 利用率接近饱和。

2. **语义增强无需牺牲效率**：
   - 通过 **decoupled encoding + GPU-resident caching**，可在不增加训练延迟的前提下引入高质量语言先验。
   - 对稀疏图（如 NELL995）尤其有益，缓解 representation friction。

3. **高吞吐与强表达力可以兼得**：
   - NGDB-Zoo 在维持高 MRR 的同时，实现 1.8×–6.8× 吞吐提升。
   - 支持多种 backbone model（BetaE、Q2B 等）无缝接入。

4. **在线采样优于静态数据集**：
   - 零存储开销、无限多样性、支持自适应难度调节，更适合 agentic reasoning 场景。

### 方法的局限性

- **依赖冻结的 PTE**：当前语义编码器不可微调，无法实现 end-to-end 的语义-结构联合优化。
- **PCIe 带宽瓶颈**：对于更大图（>4M 节点），仍需 CPU offload，受限于 PCIe 带宽。
- **静态操作符池设计**：尚未支持 streaming KG 中动态演化的拓扑结构。

### 未来工作方向

1. **轻量级 co-optimization**：探索对 PTE 进行微调的小参数适配器（LoRA-like）以保留灵活性。
2. **全 GPU-resident 架构**：进一步减少主机内存依赖，构建纯 GPU 流水线。
3. **流式 Operator DAG 更新**：支持实时更新的知识图谱，实现真正的 autonomous reasoning。
4. **异构硬件支持**：扩展至 TPU 或多节点分布式训练，应对百亿级图挑战。

---

> **总结一句话**：  
> NGDB-Zoo 通过 **operator-level training + decoupled semantic integration**，首次实现了高效、可扩展且语义丰富的 Neural Graph Database 训练框架，为下一代 neuro-symbolic reasoning 系统提供了坚实基础。

</details>

---

### 7. [SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference](https://arxiv.org/abs/2602.22136)

**Authors**: Qunyou Liu, Pengbo Yu, Marina Zapater, David Atienza  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.22136v1  

#### Abstract
Deep neural networks (DNNs) are essential for performing advanced tasks on edge or mobile devices, yet their deployment is often hindered by severe resource constraints, including limited memory, energy, and computational power. While uniform quantization provides a straightforward approach to compr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SigmaQuant: Hardware-Aware Heterogeneous Quantization Method for Edge DNN Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前边缘设备上的 DNN 推理面临严重的资源限制（内存、能耗、算力），而现有的量化方法存在以下不足：
- **Uniform Quantization**（如 INT8）对所有层采用相同 bitwidth，无法适应不同层对量化噪声的敏感度差异，导致精度损失或资源利用不充分。
- **现有 Heterogeneous Quantization 方法** 要么依赖耗时的 brute-force 搜索（如 RL-based HAQ）、要么缺乏对硬件约束（memory, energy, latency）的自适应能力，难以在多样化边缘场景中灵活部署。

### 🚀 提出的新方法：SigmaQuant
提出一种**硬件感知的分层异构量化框架 SigmaQuant**，其核心思想是：
- 基于每层权重的 **标准差（Standard Deviation, σ）** 和 **KL 散度（Kullback-Leibler Divergence）** 来指导 bitwidth 分配。
- 设计两阶段策略：
  1. **Phase 1：基于 σ 的聚类初始化**  
     利用 adaptive k-means 将层按 σ 分组，并分配初始 bitwidth（2/4/6/8-bit），快速进入满足至少一个约束（accuracy 或 size）的区域。
  2. **Phase 2：基于 KL 散度的迭代优化**  
     对少数关键层进行微调，通过 sensitivity score（结合 σ 和 KL）动态增减 bitwidth，逐步逼近目标区域。

该方法无需全局搜索，显著降低设计空间探索成本。

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **效率高** | 避免强化学习或 Hessian 分析等昂贵计算，仅需少量 QAT 循环即可收敛。 |
| **自适应强** | 可根据用户指定的 memory constraint 与 accuracy target 自动调整，适用于不同边缘平台。 |
| **硬件友好** | 显式考虑 shift-add-based MAC 架构特性，优化 latency 与 energy 消耗。 |
| **理论支撑强** | 从分布拟合视角出发，以 KL 散度衡量量化前后 weight distribution 失真，提升保精度能力。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **ImageNet**：用于主实验验证，在 ResNet-50、InceptionV3 上测试 Top-1 Accuracy。
- **CIFAR-100**：用于分析模型大小与准确率之间的 trade-off 趋势，在 ResNet-18 至 ResNet-152 系列上进行消融研究。

### ⚙️ 实验设置
- **模型架构**：ResNet 系列（18/34/50/101/152）、MobileNet、InceptionV3。
- **量化粒度**：Layer-wise heterogeneous quantization，支持 weights 为 {2,4,6,8}-bit，activations 固定为 8-bit（除非特别针对 BOPs 优化）。
- **训练流程**：
  - 先 calibration → 再短周期 Quantization-Aware Training (QAT)。
  - Phase 1 最多运行 3 次迭代，每次 4 epochs；Phase 2 最多 40 步 refinement，每步调整 2 层。
- **硬件仿真环境**：
  - 使用 TSMC 28nm 工艺实现 **shift-add-based MAC unit** 进行 post-synthesis 仿真。
  - 对比对象包括 FP32、FP16、BF16、INT8 和 shift-add 实现。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Model Size (MB)** | 所有权重量化后总存储占用（仅 weights）。 |
| **Top-1 Accuracy (%)** | 在测试集上的分类准确率。 |
| **Energy Consumption** | 推理过程中的功耗（归一化至 INT8）。 |
| **Latency / Cycle Count** | MAC 操作所需周期数（反映延迟）。 |
| **Area (μm²)** | 硬件实现面积开销。 |
| **PPA Trade-off** | Power, Performance, Area 综合权衡表现。 |

### 🔁 基线方法对比
- **Uniform Quantization**：A8W8, A8W6, A8W4, A8W2
- **State-of-the-art Heterogeneous Methods**：
  - **HAQ**（Reinforcement Learning）
  - **HAWQ-V3**（Hessian-based sensitivity）
  - **UNIQ**（Noise injection for non-uniform quantization）
  - **CLADO**（Integer Quadratic Programming）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### （1）算法层面：精度 vs. 模型大小
| 场景 | 结果 |
|------|------|
| **相同比特预算下** | SigmaQuant 比 uniform quantization **最高提升 4% Top-1 Accuracy**。 |
| **相同精度要求下** | 模型大小减少 **最多达 40%**（即仅需 60% memory budget 达到 uniform 的精度）。 |
| **相比 SOTA 异构方法** | 在 ResNet-50 上，达到 **76.86% Top-1 准确率，模型仅 12.02MB**，优于 HAWQ-V3（18.7MB @ 76.73%）、CLADO（13.42MB @ 73.10%）。 |

> 图 4(b) 显示 SigmaQuant 的回归曲线始终高于 uniform 方法，误差带无重叠，统计显著。

#### （2）硬件层面：能效与面积优化（vs. INT8）
| 指标 | 提升幅度 |
|------|---------|
| **Energy Reduction** | 最高达 **20.6%**（ResNet-101/152） |
| **Area Savings** | 最高 **22.3%** 更小芯片面积 |
| **Latency Overhead** | 略有增加（约 17.5%），但仍优于多数低比特 uniform 方案 |
| **Accuracy Loss** | 控制在 **< 3%**，远低于 A8W2 uniform（>8.5%） |

> 如图 5 所示，SigmaQuant 的数据点更靠近左上角（低能耗 + 高精度），表现出更优的 Pareto 前沿。

#### （3）激活量化扩展（BOPs 目标）
当目标切换为降低 BOPs（compute-aware）时，同时优化 weights 和 activations：
| 模型 | Accuracy Drop | BOPs Reduction |
|------|-------------|----------------|
| ResNet-18 | ≤1% | -32.9% |
| ResNet-34 | ≤1% | -49.4% |
| ResNet-50 | ≤1% | -32.0% |

表明 SigmaQuant 可灵活适配多种优化目标。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **σ 是有效的 sensitivity indicator**：权重的标准差可作为量化敏感性的第一阶代理指标，简单高效。
2. **KL 散度有助于控制分布失真**：结合 KL 散度进行 refinement，能有效避免因过度压缩导致的精度崩溃。
3. **两阶段策略高效且鲁棒**：Phase 1 快速定位可行区，Phase 2 精细调节，避免盲目搜索，wall-clock 时间可控（ResNet-18 ~2h, ResNet-152 ~30h on A100）。
4. **硬件收益显著**：在通用 shift-add 架构上，SigmaQuant 不仅节省内存，还带来明显的 energy 与 area 改善，适合边缘部署。

### ⚠️ 方法的局限性
- **依赖 QAT**：虽比 RL/Hessian 方法轻量，但仍需要多次 QAT 循环，不适合完全 zero-shot PTQ 场景。
- **bitwidth 选择受限**：目前仅支持 {2,4,6,8}，未探索非整数或 channel-wise 更细粒度分配。
- **未集成 pruning**：未与稀疏化联合优化，未来可拓展为 joint compression framework。

### 🔮 未来工作方向
- 扩展至 **dynamic bitwidth selection at runtime**（类似 AdaBits），增强跨设备泛化能力。
- 探索 **co-design with pruning and sparsity**，进一步提升压缩率。
- 应用于 **Transformer 类模型**（如 ViT、BERT），验证在 attention 结构中的有效性。
- 集成到 **full-stack compiler flow** 中，实现端到端自动化部署。

---

## 总结
SigmaQuant 提出了一种**轻量级、硬件感知、自适应的异构量化方法**，通过 **σ + KL 散度驱动的两阶段策略**，实现了在严格资源约束下的高性能 DNN 部署。实验证明其在 **accuracy、model size、energy、area** 等多个维度均优于 uniform 与主流 heterogeneous 方法，是面向边缘 AI 推理的理想解决方案。

</details>

---

### 8. [Multi-dimensional Assessment and Explainable Feedback for Counselor Responses to Client Resistance in Text-based Counseling with LLMs](https://arxiv.org/abs/2602.21638)

**Authors**: Anqi Li, Ruihan Wang, Zhaoming Chen, Yuqian Chen, Yu Lu, Yi Zhu, Yuan Xie, Zhenzhong Lan  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.21638v1  

#### Abstract
Effectively addressing client resistance is a sophisticated clinical skill in psychological counseling, yet practitioners often lack timely and scalable supervisory feedback to refine their approaches. Although current NLP research has examined overall counseling quality and general therapeutic skil...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Multi-dimensional Assessment and Explainable Feedback for Counselor Responses to Client Resistance in Text-based Counseling with LLMs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **临床心理咨询中对 client resistance（客户阻抗）应对能力缺乏及时、可扩展的反馈机制**。传统督导依赖专家人工评估，成本高、延迟大，尤其在 text-based counseling（基于文本的心理咨询）场景下更难实施。
- 现有 NLP 研究多关注整体咨询质量或单一技能（如 empathy），**缺乏针对“高风险时刻”——即客户表现出阻抗时，咨询师回应质量的细粒度评估框架**。

### 🚀 提出的新方法与创新
1. **提出四维理论驱动的评估框架（Four-dimensional Framework）**  
   将咨询师回应分解为四个核心 communication mechanisms：
   - **Respect for Autonomy**（尊重自主性）
   - **Stance Alignment**（立场一致性）
   - **Emotional Resonance**（情感共鸣）
   - **Conversational Orientation**（对话导向）  
   每个维度分为三个表达等级：No / Weak / Strong，实现**多维度、可操作化的精细评估**。

2. **构建首个专家标注的阻抗回应评估数据集**
   - 基于真实咨询对话，使用 RECAP 工具识别 client resistance utterances，并提取对应的 counselor response。
   - 数据集包含 **3,836 个样本**，由持证心理咨询师进行多维标注 + 自然语言解释（explanatory rationales），确保高质量与信效度（Cohen’s κ = 0.74–0.77）。

3. **开发端到端的 LLM 评估与反馈生成模型**
   - 在 **Llama-3.1-8B-Instruct** 上进行 full-parameter instruction tuning，同时完成两个任务：
     - 分类：判断每个 communication mechanism 的强度等级
     - 生成：输出解释性反馈（explainable feedback）
   - 引入 **explanation-augmented training**，利用人类撰写的 rationale 提供更强监督信号。

### 🔍 相比现有方法的优势
| 维度 | 本文方法 | 现有方法 |
|------|--------|---------|
| **评估粒度** | 针对 client resistance 场景下的具体干预策略 | 宏观质量评分或孤立技能评估 |
| **理论基础** | 明确的心理学理论支撑（MI, CBT等） | 多为数据驱动，缺乏理论可解释性 |
| **反馈形式** | 可生成高质量、理论一致的自然语言解释 | 多为打分或标签，缺乏行动指导 |
| **实用性** | 支持实时、可扩展的 AI 辅助训练系统 | 依赖专家人力，难以规模化 |

---

## 2. 核心实验方法和设置

### 📚 数据来源
- 来自两个公开研究用途的心理咨询对话数据集：
  - **ClientBehavior** [18]
  - **ObserverWAI** [6]
- 使用 **RECAP** 模型自动检测 client resistance utterances（准确率 91.41%），并截取其后紧接的 counselor response 构成样本对。

### ⚙️ 实验设置
- **模型架构**：基于 **Llama-3.1-8B-Instruct** 进行 full-parameter fine-tuning
- **训练策略**：
  - 5-fold cross-validation
  - Stratified sampling + 随机过采样缓解类别不平衡
  - Early stopping based on validation loss
  - 学习率：1e-5，训练 3 轮
- **推理配置**：deterministic decoding（temperature=0, top_p=1.0），保证结果可复现

### 📊 评估指标
#### （1）分类任务（Communication Mechanisms Identification）
- **Macro-F1 Score**（主指标）
- Accuracy

#### （2）解释生成任务（Explanation Generation）
- **自动评估**：
  - BLEU-1/BLEU-2
  - ROUGE-1/ROUGE-2/ROUGE-L
- **人工评估**（三维度，3点李克特量表）：
  - **Framework Consistency**（框架一致性）
  - **Evidence Anchoring**（证据锚定性）
  - **Clarity & Specificity**（清晰性与具体性）

### 🆚 基线方法对比
| 类型 | 模型列表 |
|------|----------|
| Closed-source LLMs | GPT-4o, Claude-3.5-Sonnet（zero-shot） |
| Open-source LLMs | Qwen2.5 系列（7B/14B/32B/72B）、Llama-3.1 系列（8B/70B） |

---

## 3. 主要实验结果和性能指标

### 📈 分类任务表现（Table II）

| Model | Respect for Autonomy (F1) | Stance Alignment (F1) | Emotional Resonance (F1) | Conversational Orientation (F1) |
|-------|----------------------------|--------------------------|----------------------------|----------------------------------|
| **Claude-3.5-Sonnet** | 41.61 | 52.59 | 55.59 | 45.36 |
| **GPT-4o** | 45.37 | 58.61 | 53.16 | 41.72 |
| **Our Model** | **80.92** | **77.56** | **77.34** | **77.87** |

> ✅ **相对最强 baseline 提升超过 20+ F1 points，最高达 +35.5 F1（Respect for Autonomy）**

- 所有维度均显著优于所有零样本 LLM，说明 task-specific fine-tuning 的巨大价值。
- 特别在识别“No Expression”类别上 recall 显著提升，这对临床预警具有重要意义。

### 💬 解释生成任务表现（Table III）

| Model | BLEU-1 | Framework Consistency (Human) | Evidence Anchoring (Human) | Clarity & Specificity (Human) |
|-------|--------|-------------------------------|-----------------------------|------------------------------|
| **Claude-3.5-Sonnet** | 32.45 | 1.94 | 1.94 | 2.18 |
| **GPT-4o** | 24.87 | 2.04 | 2.00 | 2.40 |
| **Our Model** | **60.34** | **2.78** | **2.73** | **2.88** |

> ✅ **BLEU-1 接近翻倍（0.60 vs 0.32）；人工评分接近天花板水平（~2.8/3.0）**

- 表明模型不仅能正确分类，还能生成**高度符合专业标准、有理有据、清晰具体的反馈建议**。

### 🔍 消融实验（Ablation Study）
- 移除训练中的 explanation supervision 后（仅用 label 训练）：
  - Macro-F1 下降约 **4 F1 points**（如从 80.92 → 73.24）
- 结论：**显式的 rationale 监督提供了额外的学习信号，有助于模型理解底层沟通原则**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **提出的四维框架有效捕捉了应对 client resistance 的关键沟通机制**，具备良好的理论与实践一致性。
2. **基于专家标注数据微调的 LLM 能够显著超越通用大模型（如 GPT-4o/Claude-3.5）在该任务上的表现**，验证了 domain-specific adaptation 的必要性。
3. **引入 explanation-augmented training 可进一步提升模型性能与可解释性**，证明“为什么这样评”比“怎么评”更重要。
4. **AI 生成的反馈能切实帮助咨询师改进回应质量**：
   - 控制实验证明：接受 AI 反馈的实验组在 post-test 中各维度得分显著提升（p < 0.001）
   - 半结构化访谈显示：参与者认为反馈提升了自我觉察、明确了改进方向、增强了处理阻抗的信心。

### ⚠️ 局限性
- 数据来源于中国在线心理平台，文化背景可能影响泛化性。
- 当前模型未考虑 long-term dialogue context，仅分析局部交互片段。
- 尚未集成 response suggestion 功能（仅有 feedback，无 alternative response 示例）。
- 依赖高质量人工标注，扩展成本较高。

### 🔮 未来工作方向
- 引入 **multi-turn context modeling** 以更好理解对话动态演变。
- 开发 **generative feedback with exemplars**，提供替代回应建议（participants 明确表达了此需求）。
- 探索 **few-shot 或参数高效微调方法**（如 LoRA），降低部署门槛。
- 将系统整合进 real-time counseling support tools，支持即时辅助决策。

---

> 🌟 **总体评价**：本研究成功将心理学理论、高质量标注数据与 LLM 技术深度融合，提出了一个**可解释、可操作、经实证有效的 AI 辅助心理咨询技能发展框架**，为心理健康领域的 NLP 应用树立了新标杆。

</details>

---

### 9. [Training-free Composition of Pre-trained GFlowNets for Multi-Objective Generation](https://arxiv.org/abs/2602.21565)

**Authors**: Seokwon Yoon, Youngbin Choi, Seunghyuk Cho, Seungbeom Lee, MoonJeong Park, Dongwoo Kim  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.21565v1  

#### Abstract
Generative Flow Networks (GFlowNets) learn to sample diverse candidates in proportion to a reward function, making them well-suited for scientific discovery, where exploring multiple promising solutions is crucial. Further extending GFlowNets to multi-objective settings has attracted growing interes...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Training-free Composition of Pre-trained GFlowNets for Multi-Objective Generation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **GFlowNets** 在多目标生成任务中面临两大挑战：
- **需要额外训练**：无论是基于 **scalarization**（如 MOGFN、HN-GFN）还是 **logical operators**（如 compositional sculpting），都需要为每组新的目标组合重新训练模型或训练辅助分类器，计算开销大且缺乏灵活性。
- **框架不统一**：现有方法通常只支持单一类型的组合方式（线性加权或逻辑操作），无法灵活处理多样化的奖励组合需求。

### 提出的新方法
本文提出一种**无需训练**（training-free）的混合策略，在推理阶段直接组合多个预训练好的 GFlowNets 的前向策略（forward policy），实现多目标生成。其核心思想是：
- 利用每个 GFlowNet 的 **reaching probability** $ u_i(s) $ 作为权重，对各模型在状态 $ s $ 处的动作概率进行加权混合。
- 定义混合策略如下：
  $$
  p_M.F(s'|s) = \frac{G\left(u_1(s)p_{1,F}(s'|s), \dots, u_k(s)p_{k,F}(s'|s)\right)}{N_M(s)}
  $$
  其中 $ G $ 是组合函数（如加权和、调和平均等），$ N_M(s) $ 是归一化常数。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **无需训练** | 可直接复用已有的单目标 GFlowNets，无需为新目标组合重新训练或微调。 |
| **统一框架** | 支持多种组合形式，包括线性 scalarization 和非线性逻辑操作（如 conjunction、subtraction）。 |
| **理论保证** | 对于线性 scalarization（$\beta=1$），证明该方法能**精确恢复目标分布**。 |
| **高效推理** | 相比 classifier-guided 方法，避免了每步枚举后继状态的高成本，显著提升推理速度。 |

---

## 2. 核心实验方法和设置

### 数据集
1. **合成任务**：  
   - **2D Grid Domain (32×32)**：用于验证方法在可控环境下的表现，ground truth 分布可计算。
   - 使用 5 个标准基准奖励函数（Shubert, Diagonal, Currin, Sphere, Branin）和 3 个圆形区域定义的合成奖励（Circle rewards）。

2. **真实世界任务**：  
   - **Fragment-based molecule generation**：从 72 个分子片段构建分子，使用三个奖励函数：
     - **SEH**（结合亲和力）
     - **SA**（合成可及性）
     - **QED**（类药性）
   - **Atom-based molecule generation (QM9)**：逐原子/键生成分子图，使用：
     - **GAP**（HOMO-LUMO 能隙）
     - **SA**
     - **QED**

### 实验设置与评估指标
| 任务类型 | 设置 | 评估指标 |
|--------|------|----------|
| **Scalarization** | 使用不同偏好向量 $ \mathbf{w} $ 进行加权组合（2~3个目标） | - 合成任务：**L1 error**（与真实分布的距离）<br>- 真实任务：top-10样本的**平均奖励** + **多样性**（1-Tanimoto相似度） |
| **Logical Operators** | 测试调和平均（⑧，conjunction）和对比算子（①，subtraction） | - 合成任务：**L1 error**<br>- 真实任务：落入“目标 bin”的样本百分比（如所有目标均高 / 仅第一个目标高） |
| **消融实验** | 移除 reaching probability 权重（即简单 ensemble） | 对比 L1 error 或目标 bin 百分比 |

### 基线方法对比
| 类型 | 基线方法 |
|------|---------|
| **Scalarization Baselines** | - **MOGFN**（条件偏好 GFlowNet）<br>- **HN-GFN**（超网络 GFlowNet） |
| **Logical Operator Baseline** | - **Classifier guidance**（compositional sculpting 中的方法） |
| **消融基线** | - **Simple Ensemble**：不使用权重 $ u_i(s) $ 的朴素混合策略 |

---

## 3. 主要实验结果和性能指标

### 合成任务（2D Grid）

#### ✅ Scalarization 结果（Table 1）
| 方法 | 2 Obj | 3 Obj | 4 Obj | 5 Obj |
|------|-------|-------|-------|-------|
| MOGFN | 0.021 | 0.027 | 0.042 | 0.048 |
| HN-GFN | 0.017 | 0.021 | 0.032 | 0.035 |
| Ensemble | 0.117 | 0.098 | 0.113 | 0.111 |
| **Ours** | **0.003** | **0.003** | **0.003** | **0.003** |

- **结论**：我们的方法在所有目标数量下均显著优于基线，且误差几乎恒定，而 MOGFN/HN-GFN 随目标增加性能下降。
- **消融实验**：简单 ensemble 表现极差，说明 **reaching probability 加权至关重要**。

#### ✅ Logical Operators 结果（Table 2）
| 操作 | Classifier Guidance | Ensemble | **Ours** |
|------|---------------------|----------|---------|
| Harmonic Mean (⑧) | 0.142–0.397 | 0.180–0.453 | **0.136–0.229** |
| Contrast (①) | 0.150–0.245 | 0.158–0.390 | **0.106–0.231** |

- **结论**：我们的方法在多数情况下优于或媲美需要训练的 classifier guidance，且无需任何额外训练。

---

### 真实世界任务（Molecule Generation）

#### ✅ Scalarization 性能（Table 3）
| 任务 | 指标 | MOGFN | HN-GFN | **Ours** |
|------|------|--------|--------|--------|
| Fragment-based (SEH-QED) | Reward | 0.759 | 0.772 | **0.773** |
| QM9 (GAP-SA) | Reward | 0.816 | 0.805 | **0.876** |
| QM9 (ALL) | Reward | 0.727 | 0.625 | **0.734** |

- **结论**：我们的方法在多个任务上达到甚至超过需训练基线的样本质量，**无需重训练**。

#### ✅ Logical Operators 性能（Table 4）
| 操作 | Classifier Guidance (%) | **Ours (%)** |
|------|------------------------|-------------|
| Harmonic Mean (⑧) | 57–85 | **66–93** |
| Contrast (①) | 60–86 | **67–95** |

- **结论**：我们的方法在目标 bin 的命中率上**全面超越 classifier guidance**，尤其在复杂组合中优势明显。

#### ⚙️ 推理效率对比（Table 5）
| 方法 | Fragment-based (ms) | QM9 (ms) |
|------|---------------------|----------|
| MOGFN | ~16 | ~24 |
| Classifier Guidance | **~1789** | **~1956** |
| **Ours** | **~25** | **~46** |

- **结论**：我们的方法比 classifier guidance 快 **40–70倍**，接近单模型推理速度。

#### ✅ 分子有效性（Table 6）
| 组合 | Classifier Guidance (%) | **Ours (%)** |
|------|------------------------|------------|
| PGAP×PsA | 86.8 | **99.6** |
| PGAP×PsA×PQED | 54.8 | **99.8** |

- **结论**：classifier guidance 显著降低生成分子的有效性，而我们的方法保持接近 100%，说明其更尊重原始生成过程。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **首次实现无需训练的 GFlowNet 组合**：通过在推理时混合前向策略，即可灵活应对任意目标组合。
2. ✅ **理论精确性**：对于线性 scalarization（$\beta=1$），所提方法**严格恢复目标分布**（Proposition 4.1）。
3. ✅ **近似质量高**：对于非线性操作，虽然存在 distortion factor $\delta(x)$，但在高密度区域（即重要采样区域）近似良好（Fig. 3）。
4. ✅ **实际性能优越**：在合成与真实任务上，性能媲美或超越需训练基线，且推理速度快数十倍。
5. ✅ **保持生成质量**：相比 classifier guidance，不会破坏分子结构的有效性。

### 局限性
- 当前理论分析集中在 $\beta=1$ 的 scalarization 场景，对一般 $\beta$ 或复杂非线性组合的误差边界仍待深入研究。
- 依赖于 GFlowNet 输出的 **state flow** 或 **partition function**，因此要求训练目标显式建模这些量（如 FM、DB、SubTB），不适用于 TB（Trajectory Balance）等隐式方法。
- 在高度相关的目标之间执行 contrast 操作时效果受限（见 Table A6 和 Fig. A7）。

### 未来工作方向
- 扩展到更复杂的组合函数（如条件生成、层次化组合）。
- 探索如何在 TB 等不显式学习 flow 的 GFlowNet 上估计 reaching probability。
- 将该框架应用于其他基于轨迹的生成模型（如强化学习中的 policy composition）。
- 结合主动学习，在线选择最优目标组合以加速科学发现。

---

> **一句话总结**：本文提出了一种无需训练即可组合预训练 GFlowNets 的通用框架，实现了灵活、高效、高质量的多目标生成，在理论和实验上均展现出显著优势。

</details>

---

### 10. [Structured Prompt Language: Declarative Context Management for LLMs](https://arxiv.org/abs/2602.21257)

**Authors**: Wen G. Gong  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.21257v1  

#### Abstract
We present SPL (Structured Prompt Language), a declarative SQL-inspired language that treats large language models as generative knowledge bases and their context windows as constrained resources. SPL provides explicit WITH BUDGET/LIMIT token management, an automatic query optimizer, EXPLAIN transpa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Structured Prompt Language: Declarative Context Management for LLMs

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **prompt engineering** 实践存在以下核心问题：
- **手动 token 计数**：开发者需反复试错以确保提示在 context window 内。
- **缺乏优化机制**：当超出 token 限制时，通常采用粗暴截断，无系统性压缩策略。
- **不可见性**：无法预知 token 在各部分提示中的分配情况。
- **平台锁定**（Provider lock-in）：提示代码依赖特定 LLM API，迁移成本高。
- **不可组合性**：提示多为单体结构，难以模块化复用。

这些问题类似于数据库发展早期在 SQL 出现前的混乱状态。

### 提出的新方法：SPL (Structured Prompt Language)
作者提出 **SPL**，一种受 SQL 启发的声明式语言，将 LLM 视为 **generative knowledge base**，其 context window 被视为受限资源进行管理。

#### 核心创新点：
1. **声明式语法**：开发者只需声明“需要什么”，而非“如何获取”。
2. **显式资源管理**：
   - `WITH BUDGET` 和 `LIMIT` 子句明确管理 token 预算。
   - 自动优化器负责 token 分配与压缩。
3. **EXPLAIN 机制**：提供类似 SQL 的执行计划，可在执行前查看 token 分配、成本估算等。
4. **集成 RAG 与持久化内存**：原生支持 `rag.query()` 和 `memory.get()`。
5. **可移植性**：同一 `.spl` 脚本可在不同提供商（如 OpenRouter, Ollama）上运行。

### 相比现有方法的优势
| 特性 | SPL | Prompty | DSPy | LMQL |
| :--- | :--- | :--- | :--- | :--- |
| **声明式查询语言** | ✅ | ❌（模板语言） | ❌（Python 嵌入式） | ⚠️（部分） |
| **全局 token 预算管理** | ✅ | ❌ | ❌ | ❌（仅变量级） |
| **EXPLAIN 执行计划** | ✅ | ❌ | ❌ | ❌ |
| **内置 RAG & Memory** | ✅ | ❌ | ❌ | ❌ |
| **提供商无关性** | ✅ | ❌（Azure 锁定） | ✅ | ✅ |

## 2. 核心实验方法和设置

### 实验设置
所有实验均在不调用实际 LLM 的情况下完成，仅测试 SPL 的解析、分析、优化和 `EXPLAIN` 流程，证明其在“零 token 成本”下即可提供价值。

### 评估指标
1. **开发者体验**：代码行数（LoC）、手动 token 操作次数。
2. **预算可见性**：是否支持 `EXPLAIN` 查看 token 分配。
3. **静态验证**：是否支持语法和预算的静态检查。
4. **成本估算**：基于目标模型定价预估执行成本。
5. **功能验证**：自动化测试覆盖所有核心特性。

### 基线方法对比
- **Imperative Python**：使用 `tiktoken` 手动计数、截断和缓存的典型实现。
- **Prompty**：微软的 YAML 提示格式。
- **DSPy**：用于优化提示内容的 Python 框架。
- **LMQL**：用于约束生成的查询语言。

## 3. 主要实验结果和性能指标

### 开发者体验提升
在五个基准任务上的平均结果：

| 任务 | SPL LoC | Python LoC | **减少 65%** |
| :--- | :---: | :---: | :---: |
| 简单问答 | 9 | 20 | 55.0% |
| RAG 增强问答 | 17 | 51 | 66.7% |
| 多步 CTE | 24 | 63 | 61.9% |
| 函数复用 | 16 | 43 | 62.8% |
| 缓存重复 | 9 | 42 | 78.6% |
| **平均** | **15** | **44** | **65.0%** |

- **消除 35 次**手动 token 操作。
- 所有 SPL 查询均支持 `EXPLAIN` 和静态验证。

### 成本差异显著
对同一查询在不同模型上运行 `EXPLAIN`，结果显示成本差异高达 **68 倍**：

| 模型 | 预估成本 (USD) | 相对成本 |
| :--- | :---: | :---: |
| GPT-4 (Legacy) | $0.2316 | 67.5x |
| Claude Opus 4.6 | $0.2058 | 60.0x |
| Claude Sonnet 4.5 | $0.0412 | 12.0x |
| GPT-4o | $0.0293 | 8.5x |
| GPT-3.5 Turbo | $0.0049 | 1.4x |
| **Claude Haiku 4.5** | **$0.0034** | **1.0x** |

此信息在执行前即可获得，有助于模型选择。

### 功能验证结果
20 项自动化测试全部通过，验证了：
- 完整的声明式语法解析。
- `WITH BUDGET`, `LIMIT`, `OUTPUT BUDGET` 工作正常。
- `EXPLAIN` 输出树状结构、百分比和成本。
- RAG (`rag.query`) 和 Memory (`memory.get`) 功能正常。
- CTE 和函数支持。
- 多种提供商适配器（Ollama, OpenRouter）可用。
- 自动压缩在超预算时触发。
- 解析器无外部依赖。

## 4. 关键结论和发现

### 主要发现
1. **SPL 将 prompt engineering 从“手工艺”转变为“工程学科”**，通过声明式、可优化、可组合的框架提升了开发效率和可靠性。
2. **token 预算是一个头等公民**：显式管理 context window 可避免运行时错误，并实现成本优化。
3. **EXPLAIN 是关键**：在执行前洞察 token 分配和成本，是实现高效 LLM 应用的基础。
4. **同一 `.spl` 脚本可跨平台运行**：可在云端（OpenRouter）或本地（Ollama）以不同成本执行，无需修改。
5. **逻辑分块**（Logical Chunking）结合 **MoM 路由** 可能替代单纯扩大 context window 的路径，通过将大任务分解并路由到专家小模型，实现更高效的计算。

### 局限性
1. **轻微超预算仍使用截断**：对于小规模超限，目前仍采用简单截断，未来计划引入语义压缩（如摘要）。
2. **类型系统有限**：尚未支持上下文源的类型注解。
3. **MoM 路由基于关键词**：当前的模型路由使用规则匹配，对模糊或多领域查询较脆弱。
4. **多轮对话管理不足**：虽支持 `memory.get("history")`，但缺少正式的长对话修剪策略（如摘要、按相关性保留）。
5. **评估范围有限**：当前基准集中在问答类任务，需扩展至代码生成、复杂推理等。

### 未来工作方向
1. **学习型优化**：
   - 基于历史数据学习最优的 token 分配。
   - 学习最佳的文档分块策略。
   - 动态 MoM 路由，根据性能反馈更新路由表。
2. **生态系统集成**：
   - 与 **DSPy** 结合：用 DSPy 优化 SPL 中每个 `SELECT` 子句的内容。
   - 与 **LangChain / LlamaIndex** 结合：作为其之上的声明式编排层。
   - 在 **IDE** 中集成：提供语法高亮、自动补全和内联 `EXPLAIN`。
3. **安全与合规**：
   - 内容过滤：扫描 `system_role` 和 `GENERATE` 字符串字面量。
   - 模型白名单：限制 `USING MODEL` 的可选范围。
   - RAG 注入防护：对检索结果进行净化处理。
4. **形式化验证**：自动生成解析器并验证其与 EBNF 语法的一致性。

</details>

---

### 11. [Geometric Priors for Generalizable World Models via Vector Symbolic Architecture](https://arxiv.org/abs/2602.21467)

**Authors**: William Youngwoo Chung, Calvin Yeung, Hansen Jin Lillemark, Zhuowen Zou, Xiangjian Liu, Mohsen Imani  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.21467v1  

#### Abstract
A key challenge in artificial intelligence and neuroscience is understanding how neural systems learn representations that capture the underlying dynamics of the world. Most world models represent the transition function with unstructured neural networks, limiting interpretability, sample efficiency...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Geometric Priors for Generalizable World Models via Vector Symbolic Architecture*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前主流的 **World Models** 多依赖于非结构化的神经网络（如MLP）来建模状态转移函数 $T: S \times A \rightarrow S$，这种黑箱建模方式存在以下缺陷：
- **样本效率低**（poor sample efficiency）
- **泛化能力弱**，难以推广到未见过的状态-动作对（zero-shot generalization）
- **长时程rollout误差累积严重**（compounding rollout errors）
- **潜空间缺乏几何解释性**，无法进行符号级操作

这些问题限制了模型在真实世界规划与推理任务中的应用。

### 提出了什么新方法或新思路
本文提出一种基于 **Vector Symbolic Architecture (VSA)** 的可泛化世界模型框架，其核心思想是将环境动力学建模为具有**群作用**（group action）的代数结构，并利用 **Fourier Holographic Reduced Representation (FHRR)** 构造具备几何先验的潜表示。

#### 主要创新点包括：
- **使用FHRR编码器** 将状态 $s$ 和动作 $a$ 映射为单位复向量（unitary complex vectors），即 $\phi_S(s), \phi_A(a) \in \mathbb{C}^D$，其中每个分量位于单位圆上。
- **状态转移通过element-wise complex multiplication实现绑定操作（binding）**：
  $$
  \phi_S(s_{t+1}) = \phi_S(s_t) \odot \phi_A(a_t)
  $$
  这种操作天然满足组合性和近似逆性。
- 引入**等变潜表示**（equivariant latent representations）：确保动作在潜空间中的作用对应于群操作，从而保留环境对称性。
- 利用 **cleanup机制** 实现噪声鲁棒性和误差纠正：通过与状态codebook进行相似度搜索，恢复最接近的真实状态嵌入。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **泛化能力** | 在零样本（zero-shot）场景下表现优异，能处理训练中未见的状态-动作对 |
| **长时程稳定性** | 多步rollout中误差不指数增长，结合cleanup显著提升长期预测准确性 |
| **可解释性** | 潜空间具有明确的代数结构（如群同态、可逆性），支持符号推理 |
| **鲁棒性** | 对输入噪声高度容忍，在高斯噪声干扰下仍保持高性能 |
| **计算效率** | 所有VSA操作均为逐元素运算，训练和推理均为线性时间复杂度 $O(D)$ |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GridWorld Environment**：一个 $10 \times 10$ 的离散网格世界，共100个状态，4个确定性动作（上下左右）。
- 动作遵循确定性转移函数 $T(s, a)$。
- 数据集中80%的 $(s, a)$ 对用于训练，20%被hold-out用于**零样本测试**。

### 实验设置和评估指标
#### 模型配置
- **VSA-FHRR模型**：
  - 状态/动作嵌入维度 $D = 512$
  - 使用learnable FHRR编码器
  - 训练目标包括：binding loss、invertibility regularizer、orthogonality regularizer
- **MLP基线模型**：
  - MLP-Small（2层，128隐单元）
  - MLP-Medium（4层，256隐单元）
  - MLP-Large（6层，512隐单元）
  - 输入为拼接后的 $(s, a)$ 向量，输出下一状态预测

#### 评估指标
| 指标 | 描述 |
|------|------|
| **1-step Accuracy** | 单步状态转移预测准确率 |
| **Zero-shot Accuracy** | 在未参与训练的 $(s,a)$ 对上的预测准确率 |
| **Cosine Similarity** | 预测状态与真实状态嵌入之间的余弦相似度 |
| **Rollout Accuracy** | 多步（5、20、100步）潜空间 rollout 的最终状态预测准确率 |
| **Rollout + Clean** | 每隔2步执行一次cleanup后的rollout性能 |
| **Robustness to Noise** | 添加不同强度高斯噪声后的一阶动态预测准确率 |
| **t-SNE可视化** | 展示潜空间是否保留原始环境的空间结构 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1）

| 任务 | FHRR (Ours) | MLP-Small | MLP-Medium | MLP-Large |
|------|------------|-----------|------------|-----------|
| **1-step Accuracy** | **96.3%** | 80.0% | 80.0% | 80.25% |
| **1-step Zero-Shot Accuracy** | **87.5%** | 0.0% | 0.0% | 1.25% |
| **Cosine Similarity** | **83.0** | 79.5 | 79.9 | 80.6 |
| **Cosine Similarity (Zero-Shot)** | **80.5** | 0.9 | 0.15 | 3.1 |
| **Rollout (5 steps)** | **74.6%** | 39.8% | 38.0% | 40.8% |
| **Rollout (20 steps)** | **34.6%** | 2.0% | 4.0% | 6.2% |
| **Rollout (20 steps + Clean)** | **61.4%** | 5.4% | 7.8% | 8.4% |
| **Rollout (100 steps)** | 1.8% | 0.8% | 1.8% | 2.0% |
| **Rollout (100 steps + Clean)** | **38.6%** | 2.8% | 4.0% | 3.2% |

> ✅ **亮点总结**：
> - 在**零样本设置下**，FHRR达到 **87.5%** 准确率，而所有MLP基线几乎失效（≤1.25%）
> - 在**20步rollout + cleanup**中，FHRR比最大MLP高出 **~8×**
> - 即使在100步长rollout中，FHRR仍维持 **38.6%** 准确率，远超MLP（~3%）
> - FHRR参数量仅为 **53K**，与MLP-Small相当（41K），远小于MLP-Large（1.4M）

### 与其他方法的对比结果
- **泛化性**：随着zero-shot比例增加（图3），FHRR性能呈线性下降，而MLP性能指数崩溃，在仅训练90%数据时已低于10%。
- **鲁棒性**（图5a）：在标准差高达5的高斯噪声下，FHRR仍保持 >80% 准确率；MLP-Medium迅速降至 <20%。
- **潜空间结构**（图4）：t-SNE显示FHRR的state embedding清晰反映GridWorld的空间拓扑结构，而MLP完全混乱。
- **相似性核函数**（图5b）：FHRR展现出平滑且尖锐的动作局部性核，表明其学习到了结构化的几何关系。

### 消融实验结果（图6）
- **维度影响**：提高embedding dimension $D$ 显著增强robustness，验证“高维空间中随机向量趋于正交”的理论基础。
- **正交正则项**（$C_{\text{ortho}}$）：即使较小权重也能有效促进状态分离，进一步增大收益有限。
- **invertibility约束**：有助于动作表示形成近似群结构，提升多步组合能力。
- **MLP参数规模**：增加参数并未改善robustness，反而可能加剧过拟合。

---

## 4. 关键结论和发现

### 主要发现
1. **引入VSA作为几何先验可显著提升世界模型的泛化能力和鲁棒性**：
   - 通过将动作建模为群操作，实现了**结构感知的表示学习**。
2. **FHRR的binding机制天然支持多步组合与逆向推理**：
   - 支持直接在潜空间进行 $k$ 步 rollout：  
     $$
     \phi_S(s_{t+k}) = \phi_S(s_t) \odot \prod_{i=1}^k \phi_A(a_{t+i-1})
     $$
3. **cleanup机制是缓解误差积累的关键组件**：
   - 类似nearest neighbor decoding，在每一步提供“纠错反馈”，防止语义漂移。
4. **高维空间中的准正交性保障了cleanup的有效性**：
   - 不同状态嵌入之间具有大间隔边界（margin ~ $1 - O(1/\sqrt{D})$），使得噪声不易跨越决策边界。

### 方法的局限性
- 当前仅适用于**小规模离散状态空间**（如GridWorld），尚未扩展至连续、部分可观测或随机环境。
- FHRR要求状态和动作为离散符号，需额外设计编码策略以处理图像或连续控制输入。
- 虽然参数量小，但显式维护state codebook在大规模环境中可能导致内存开销上升。

### 未来工作方向
- 将该VSA-based world model集成到 **Model-Based RL** 框架中，用于真实场景的规划与决策。
- 探索如何将FHRR与CNN/RNN结合，处理视觉输入并构建端到端可微系统。
- 扩展至**连续动作空间**，例如通过将动作离散化或使用GHRR（Generalized HRR）中的矩阵表示。
- 结合**neuromorphic computing**硬件，发挥VSA在边缘设备上的高效推理潜力。

---

> 📌 **总体评价**：  
> 本论文提出了一个**原则性强、结构清晰、极具解释性的世界模型架构**，通过引入 **VSA + FHRR + cleanup** 的组合，成功地将符号AI的结构化优势与深度学习的表示能力相结合，在样本效率、泛化性、鲁棒性和可解释性方面全面超越传统MLP模型，为构建下一代通用智能体提供了新的范式路径。

</details>

---

### 12. [Mamba Meets Scheduling: Learning to Solve Flexible Job Shop Scheduling with Efficient Sequence Modeling](https://arxiv.org/abs/2602.21546)

**Authors**: Zhi Cao, Cong Zhang, Yaoxin Wu, Yaqing Hou, Hongwei Ge  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.21546v1  

#### Abstract
The Flexible Job Shop Problem (FJSP) is a well-studied combinatorial optimization problem with extensive applications for manufacturing and production scheduling. It involves assigning jobs to various machines to optimize criteria, such as minimizing total completion time. Current learning-based met...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Mamba Meets Scheduling: Learning to Solve Flexible Job Shop Scheduling with Efficient Sequence Modeling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **Flexible Job Shop Problem (FJSP)** 这一经典的组合优化问题展开研究。FJSP 是制造与生产调度中的核心挑战，其目标是在异构机器上为作业分配操作并安排顺序，以最小化 **makespan**（总完成时间）等目标。

传统基于图神经网络（GNN）或图注意力机制（Graph Attention）的方法存在以下问题：
- **局部特征提取限制**：仅在邻域内传播信息，难以捕捉全局依赖关系；
- **计算复杂度高**：图注意力机制具有 $O(n^2)$ 复杂度，对大规模实例效率低下；
- **图结构设计复杂**：需要精心构造 disjunctive graph，增加了建模难度。

### 🚀 提出的新方法与创新思路
作者提出了一种名为 **Mamba-CrossAttention (M-CA)** 的新型神经架构，首次将 **Mamba 模型**引入到制造调度领域，用于高效序列建模。

#### 主要创新点如下：

| 创新点 | 具体描述 |
|--------|----------|
| **1. 引入 Mamba 模型进行序列建模** | 首次将 Mamba（一种线性复杂度的 State-Space Model, SSM）应用于 FJSP，替代传统的 GNN 或 Transformer 架构，实现对完整操作和机器序列的全局建模。 |
| **2. 双分支 Mamba 编码器（Dual Mamba Encoder, DME）** | 分别用两个独立的 Mamba block 提取 operation 和 machine 的特征，避免模态干扰，提升表示学习能力。 |
| **3. 轻量级交叉注意力解码器（Cross-Attention Decoder）** | 设计了一个高效的 cross-attention 结构，融合 operation 和 machine 的交互信息，复杂度仅为 $O(|O| \times |M|)$，远低于 self-attention 的 $O(|O|^2)$。 |
| **4. 端到端强化学习框架** | 基于 PPO 算法训练策略网络，直接输出最优 operation-machine 对的选择概率，实现 end-to-end 的调度求解。 |

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|---------|
| **性能更强** | 在多个 benchmark 上超越当前最先进的学习型方法（如 DAN、HGNN），甚至优于精确求解器 OR-Tools。 |
| **速度更快** | Mamba 的线性时间复杂度显著提升了推理速度，尤其在大规模问题中表现突出。 |
| **泛化能力好** | 在未见过的大规模实例（如 1000×10, 10000×10）上仍能保持高性能，展现出极强的可扩展性。 |
| **无需复杂图结构** | 完全摆脱了对 disjunctive graph 的依赖，简化了状态表示设计。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验涵盖了多种公开 benchmark 和合成数据：

| 数据集类型 | 名称 | 规模范围 | 来源 |
|----------|------|--------|------|
| **公开基准** | Brandimarte | 10×6 ~ 20×15 | [2] |
| | Hurink (rdata/edata/vdata) | 10×5 ~ 30×10 | [12] |
| | Barnes | 10×11 ~ 15×17 | [1] |
| | Dauzere | 10×5 ~ 20×10 | [6] |
| **合成数据** | FJSP NxM | 10×5, 20×5, ..., 40×10 | [27] 中生成方式 |
| **超大规模测试** | FJSP 100×10 ~ 10000×10 | 用于验证泛化能力 | 自定义生成 |

> 注：所有处理时间从 $U(1,20)$ 或 $U(1,99)$ 均匀采样。

### 🧪 实验设置与评估指标

#### 模型配置
- **编码器**：Dual Mamba blocks（各1层）
- **解码器**：两层 Cross-Attention
- **决策网络**：3层 MLP（actor/critic）
- **训练算法**：PPO + GAE
- **硬件**：单张 RTX 3090 GPU
- **训练迭代数**：10,000 次，每轮 20 个实例

#### 评估策略
- **Greedy Strategy (-G)**：每步选择最高概率的动作，仅一条轨迹。
- **Sampling Strategy (-S)**：并行采样 100 条轨迹，取最优结果，增强探索。

#### 评估指标
| 指标 | 定义 |
|------|------|
| **Objective Value (Obj)** | 最终 makespan |
| **Gap (%)** | $(C / C_{\text{best}} - 1) \times 100\%$，其中 $C_{\text{best}}$ 为已知最优或 OR-Tools 解 |
| **Time (s)** | 生成一个解所需的时间（秒） |

### ⚔️ 基线方法对比
| 类别 | 方法 | 描述 |
|------|------|------|
| **Exact Solver** | OR-Tools | Google 开源求解器，设 1800 秒时间上限 |
| **Heuristic Rules (PDRs)** | FIFO, MOPNR, SPT, MWKR | 工业常用启发式规则 |
| **Learning-based Methods** | HGNN-G/S | 图神经网络 + 强化学习 [27] |
| | DAN-G/S | 图注意力网络 [31] |
| | MLP* | 轻量级多层感知机 [32] |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（来自 Table 1–3）

#### 在标准 benchmark 上的表现（Table 1）
| 方法 | 平均 Gap (%) | 推理时间 (s) | 是否最优 |
|------|-------------|--------------|---------|
| **M-CA-G** | **最低或第二低** | **~0.6–1.2** | ✅ 多项第一 |
| **DAN-G** | 较高 | ~0.97–1.27 | ❌ |
| **MLP*** | 更高 | ~0.39 | ❌ |
| **OR-Tools** | 有 Gap（非最优） | >879s | ⚠️ 超时 |

> 💡 图表 Figure 3 显示 M-CA-G 在多数 benchmark 上取得最小平均 gap。

#### 在合成数据上的表现（Table 2）
| 方法 | FJSP 20×10 (Gap / Time) |
|------|------------------------|
| OR-Tools | 0.00% / 1805s |
| DAN-S | -0.4% / 4.97s |
| **M-CA-S** | **-3.11% / 4.61s** ✅ |

> ✅ **M-CA-S 不仅速度快，且质量更高，甚至优于 OR-Tools！**

#### 在超大规模实例上的泛化能力（Table 3 & 8）
| 方法 | FJSP 1000×10 (Obj / Time) | FJSP 10000×10 |
|------|----------------------------|--------------|
| OR-Tools | 10283.9 / 3603s | ❌ OOM-CPU（内存溢出） |
| HGNN | 10287.1 / 1019.86s | ❌ OOM-GPU |
| **M-CA** | **9038.1 / 120.44s** ✅ | **90200 / 3.34h** ✅ |

> ✅ M-CA 成功求解 **10000×10** 实例（训练规模仅 20×10），展现惊人泛化能力！

### 🔬 消融实验结果（Table 4）

| 模型变体 | FJSP 20×10 Gap | Time(s) | 分析 |
|---------|---------------|--------|------|
| DAN | 1.95% | 1.31 | 当前 SOTA |
| DME (only Mamba) | -1.17% | 0.62 | 已优于 DAN |
| CA (only cross-attn) | -0.65% | 0.61 | 小规模有效，大问题退化 |
| **M-CA (DME + CA)** | **-1.03%** | **0.80** | ✅ 最佳平衡 |
| M2-CA / M3-CA | > -0.35% | ↑ 时间增加 | ❌ 多层无益，反而下降 |

> ✅ 单层 Mamba + Cross-Attention 是最优组合，兼顾性能与效率。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Mamba 模型适用于离散组合优化问题**  
   首次证明 Mamba 可成功应用于 FJSP，打破了其主要用于 NLP 的认知边界。

2. **序列建模优于图建模**  
   全局序列视角能更好地捕捉 operation 与 machine 之间的长程依赖，突破了图注意力的局部性瓶颈。

3. **高效架构带来显著加速**  
   M-CA 的推理速度比 DAN 快 **约 2 倍以上**，特别适合实时调度场景。

4. **强大的跨规模泛化能力**  
   在训练于 20×10 的模型上，成功求解高达 **10000×10** 的工业级问题，gap 仍显著低于启发式方法。

5. **端到端学习可超越精确求解器**  
   在部分任务中，M-CA-S 得到的解质量 **优于 OR-Tools**，同时耗时不到其百分之一。

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖强化学习训练稳定性** | PPO 训练过程较长，需大量调参；reward sparse 问题依然存在。 |
| **自回归生成导致延迟累积** | 虽然模型快，但 autoregressive 与环境交互仍是瓶颈（见 Table 6）。 |
| **极端稀疏问题可能失效** | 若 operation-machine 兼容性极低，action space 过小，策略学习困难。 |
| **无法保证理论最优性** | 作为启发式方法，不能提供最优性证明。 |

### 🔮 未来工作方向
1. **进一步优化环境交互机制**  
   减少 autoregressive 步骤开销，探索并行化或非自回归生成策略。

2. **跨问题迁移学习**  
   探索 FJSP → JSSP 或 Flow Shop 的通用调度模型（文中已初步验证在 JSSP 上也优于 L2D）。

3. **结合搜索机制（如 Beam Search）**  
   提升 sampling 效率，在有限时间内找到更优解。

4. **部署至真实工厂系统**  
   在实际产线中测试鲁棒性和响应速度，推动工业落地。

5. **探索其他 SSM 架构**  
   如是否可用 Hyena、S4 等替代 Mamba，进一步压缩成本。

---

> 🎯 **总结一句话**：  
> 本论文开创性地将 **Mamba 模型**引入 **FJSP 调度问题**，提出了 **Mamba-CrossAttention** 架构，在保持线性复杂度的同时实现了更强的全局建模能力，实验表明其在多个维度上全面超越现有方法，是迈向 **高效、可扩展、端到端智能调度系统** 的重要一步。

</details>

---

### 13. [MERRY: Semantically Decoupled Evaluation of Multimodal Emotional and Role Consistencies of Role-Playing Agents](https://arxiv.org/abs/2602.21941)

**Authors**: Zhenyu Wang, Xiaofen Xing, Yirong Chen, Xiangmin Xu  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.21941v1  

#### Abstract
Multimodal Role-Playing Agents (MRPAs) are attracting increasing attention due to their ability to deliver more immersive multimodal emotional interactions. However, existing studies still rely on pure textual benchmarks to evaluate the text responses of MRPAs, while delegating the assessment of the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MERRY: Semantically Decoupled Evaluation of Multimodal Emotional and Role Consistencies of Role-Playing Agents

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前对 **Multimodal Role-Playing Agents (MRPAs)** 的评估存在两大缺陷：
1. **语义评估与模态生成耦合**：传统方法将文本响应评估与多模态表达合成质量混在一起，导致错误归因模糊（例如，是LLM理解出错还是Talker模块合成失败？）。
2. **依赖人类主观判断**：现有自动评估指标（如SSIM、WER等）仅关注低级信号保真度，无法衡量跨模态情感一致性（cross-modal emotional consistency）、角色风格一致性等高层语义维度。

### 🚀 提出的新方法与创新思路
作者提出 **MERRY** —— 一个**语义解耦的评估框架**（semantically decoupled evaluation framework），用于系统评估 MRPAs 在 **Emotional Consistency (EC)** 和 **Role Consistency (RC)** 上的表现。

#### 主要创新点包括：
- **五项精细化 EC 指标 + 三项 RC 指标**：
  - **EC 指标**：  
    - `MEC`（Multimodal Emotional Consistency）：综合文本、面部、身体、语音的情感匹配程度。  
    - `CEC`（Cross-modal Emotional Consistency）：衡量不同模态间情绪是否一致。  
    - `EDD`（Emotional Distribution Divergence）：比较模型生成的情绪转移分布与真实数据的差异。  
    - `RCD`（Relative Character Discrepancy）：评估角色间情感转移差异是否合理。  
    - `ED`（Emotional Discrepancy）：反映专家在识别情绪时的一致性，间接体现表达清晰度。
  - **RC 指标**：基于改进的 LLM-as-Judge 范式，引入 **双向证据查找任务**（bidirectional-evidence-finding task），提升评分可靠性。

- **新型评估范式转变**：  
  将传统的“主观打分”转化为“寻找支持/反驳证据”的客观任务，显著提高 LLM 作为裁判时的人类一致性（human agreement）。

- **高质量数据集 MERRY-Data 构建**：  
  基于真实电视剧视频数据 CPED 构建，包含：
  - 角色画像（Profile）
  - 多轮对话历史
  - 面部表情、肢体动作、语音语调的**细粒度语义描述**
  - 支持训练和评估中的语义解耦

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | MERRY |
|------|--------|-------|
| 评估粒度 | 端到端整体评价 | 语义与模态分离，可定位问题来源 |
| 自动化水平 | 严重依赖人工标注 | 引入高一致性 LLM 评估流程 |
| 数据真实性 | 多为合成数据 | 基于真实影视数据，情感动态更自然 |
| 指标设计 | 缺乏跨模态一致性度量 | 明确量化跨模态与角色内情感连贯性 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **主数据集**：**MERRY-Data**（本文构建）
  - 来源：从 **CPED**（Chinese Personalized and Emotional Dialogue dataset）中提取
  - 内容：来自40部中文电视剧的真实多轮对话视频片段
  - 特征：
    - 包含角色 Profile、图像、关系网络
    - 每个响应配有详细的 facial expression、body movement、speech prompt 的自然语言描述
    - 标注了13种情绪类别和对话行为
  - 划分：Train (23,817样本), Test (1,690样本)

- **对比训练数据集**：
  - `OmniCharacter-10K`：合成数据集，基于文本生成语音和动作
  - `MMRole`：基于小说改编的角色扮演数据集，部分合成
  - `MERRY-Data`（ours）：真实数据驱动

### ⚙️ 实验设置
#### 模型类型
- **Training-free 方法**（零样本提示）：
  - Closed-source: `Doubao-1.5v`, `GPT-5-chat`, `Gemini-2.5-pro`
  - Open-source: `Qwen2.5-Omni(7B)`, `MiniCPM-o-2.6(8B)`
- **Training 方法**（LoRA 微调）：
  - 统一使用 `MiniCPM-o-2.6(8B)` 作为基础模型
  - 在不同数据集上进行微调对比：`None`, `OmniCharacter`, `MMRole`, `MERRY-Data`

#### 输入配置类型
| 类型 | 含义 |
|------|------|
| `All` | 提供 Profile + Previous Info |
| `Prof` | 仅提供 Profile |
| `Prev` | 仅提供 Previous Info |
| `None` | 不提供任何额外信息 |

#### 评估指标（共8项）
| 类别 | 指标 | 方向 |
|------|------|------|
| **EC** | MEC_lower / upper | ↑ |
|       | CEC_lower / upper / intra | ↑ |
|       | EDD_inter / intra | ↓ |
|       | RCD_inter / intra | → 0 |
|       | ED_all / fac / bod / spe | ↓ |
| **RC** | Exp (Experience) | ↑ |
|       | Cha (Characteristic) | ↑ |
|       | Rel (Relationship) | ↑ |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Test Set 上平均表现）

#### ✅ Training-free 方法结果（Table VII）
- **最佳整体表现**：`Gemini-2.5-pro`（omni-modal foundation model）
  - `MEC_upper`: **0.608**（最高）
  - `CEC_upper`: **0.797**
  - `RC_Exp`: 4.923
- **Vision-only 模型劣势明显**：`Doubao-1.5v` 和 `GPT-5-chat` 在多数 EC 指标上落后于 Gemini
- **开放模型表现较弱**：`MiniCPM-o-2.6(8B)` 的 `MEC_upper=0.536`，低于闭源模型

> 💡 发现：具备完整多模态输入能力的 foundation model 更能准确捕捉情感动态。

#### ✅ 训练方法对比（Table VIII）
| 训练数据 | MEC_upper | CEC_upper | RCD_intra | RC_Cha |
|---------|----------|-----------|------------|--------|
| None (无训练) | 0.536 | 0.652 | -0.017 | 4.117 |
| OmniCharacter (synthetic) | 0.534 | 0.514 | -0.001 | 4.276 |
| MMRole (semi-synthetic) | 0.517 | 0.586 | -0.061 | 4.274 |
| **MERRY-Data (real)** | **0.541** | **0.962** | **-0.021** | 4.167 |

> ✅ 结论：**在真实数据 MERRY-Data 上训练显著提升了 EC 表现**，尤其是 `CEC_upper` 达到 **0.962**（远超其他），说明跨模态一致性更好。

> ❌ 反常现象：尽管 EC 提升，但 RC 提升有限，甚至不如在合成数据上微调的结果，表明 **simple fine-tuning 存在角色泛化差的问题**。

#### 🔍 消融分析发现
- **Profile 对性格相关指标（Cha）最重要**
- **Previous Info 对经历（Exp）和关系（Rel）最关键**
- **All ≠ 最优**：当同时提供 Profile 和 Prev 时，强模型（如 GPT-5）反而性能下降，呈现 “1+1 < 2” 效应；而弱模型受益，呈 “1+1 > 2”，说明提示长度可能成为强模型的约束。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **训练数据来源决定情感一致性**：
   > ✅ **在真实世界数据上训练能显著提升 Emotional Consistency（EC）**  
   > ❌ **在合成数据上训练会降低 EC，尤其损害跨模态一致性**

2. **现有模型普遍存在以下问题**：
   - **Positive Bias**：过度使用 happy、grateful 等正面情绪，缺乏情境适配性
   - **Emotional Templatization**：用 anger/sadness 替代 depress/fear 等复杂负面情绪
   - **Fine-grained Negative Emotion Bottleneck**：对 depress, fear, disgust, astonished 等情绪识别与生成能力极弱（precision & recall 均低）

3. **提示工程的有效性受限于模型强度**：
   - **Simple prompting** 对弱模型有增强作用，但会**限制强模型发挥**
   - **Simple fine-tuning** 方法面临严重的 **poor role generalization** 问题，难以适应新角色

4. **Intra-turn 情感转移更重要且更稳定**：
   - `EDD_intra` 和 `RCD_intra` 更受 Profile 影响，说明角色内在性格主导单轮内的细微情绪变化
   - Inter-turn 转移更多依赖上下文，波动更大

### ⚠️ 方法局限性
- 当前研究聚焦于**中文场景**，语言文化特异性较强
- 数据构造仍需大量人工验证（如 Profile 校正、描述质检）
- ERC（Emotion Recognition in Conversation）依赖多个 LLM 专家投票，成本较高
- 尚未完全解决角色泛化问题（role generalization）

### 🔮 未来工作方向
- 开源代码与数据管道，推动多语言复现
- 探索更高效的自动化语义描述生成方式
- 设计专门针对角色泛化的训练策略（如 meta-learning、prompt tuning）
- 扩展至更多模态（如手势、眼神、空间位置）

---

> 📌 总结一句话：  
> **MERRY 通过语义解耦 + 真实数据 + 新型评估任务，揭示了当前 MRPAs 在情感表达上的模板化、正向偏见与负向瓶颈，并指出“真实数据训练 + 解耦评估”是通往拟人化交互的关键路径。**

</details>

---

### 14. [PASTA: A Modular Program Analysis Tool Framework for Accelerators](https://arxiv.org/abs/2602.22103)

**Authors**: Mao Lin, Hyeran Jeon, Keren Zhou  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.22103v1  

#### Abstract
The increasing complexity and diversity of hardware accelerators in modern computing systems demand flexible, low-overhead program analysis tools. We present PASTA, a low-overhead and modular Program AnalysiS Tool Framework for Accelerators. PASTA abstracts over low-level profiling APIs and diverse ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PASTA: A Modular Program Analysis Tool Framework for Accelerators

## 1. 论文的主要贡献和创新点

### 解决的问题
现代计算系统中硬件加速器（如 GPU、TPU）日益复杂且多样化，传统的性能分析工具存在以下问题：
- **灵活性差**：厂商提供的工具（如 NVIDIA Nsight Systems、AMD ROCm Profiler）功能固定，难以支持定制化分析需求。
- **缺乏跨框架集成能力**：无法有效关联低层硬件事件与高层深度学习（DL）框架语义（如 PyTorch/TensorFlow 中的算子执行、张量分配）。
- **高开销**：传统基于 CPU 的分析方式在处理大规模并行任务时引入显著运行时开销，影响程序行为的真实性。

### 提出的新方法与思路
作者提出 **PASTA**（Program AnalysiS Tool Framework for Accelerators），一个**模块化、低开销、可扩展的程序分析工具框架**，其核心设计思想包括：

- **统一抽象层**：通过 `PASTA Event Handler` 抽象不同厂商（NVIDIA/AMD）的底层 profiling API 和 DL 框架回调机制，提供一致的事件接口。
- **多粒度事件捕获**：同时支持粗粒度（kernel launch, memory copy）和细粒度（thread-level memory access）事件，并融合 DL 框架级事件（operator start/end, tensor allocation）。
- **GPU 加速的数据预处理**：引入 `PASTA Event Processor` 在 GPU 上进行 in-situ 数据预处理和初步分析，避免大量 trace 数据传输到 CPU 导致的瓶颈。
- **即插即用的工具模板**：开发者可通过继承 `PASTA Tool Collection` 模板快速构建自定义分析工具，仅需实现少量函数即可完成复杂分析逻辑。

### 相比现有方法的优势
| 特性 | PASTA | 厂商工具（Nsight/ROCProfiler） | DL 框架内置 Profiler（PyTorch/TF） |
|------|-------|-------------------------------|----------------------------------|
| 跨厂商支持 | ✅ | ❌（仅限自家） | ❌ |
| 支持 DL 框架语义 | ✅ | ❌ | ✅ |
| 支持低层硬件细节 | ✅ | ✅ | ❌ |
| 可扩展性 | 高（模块化设计） | 低 | 低 |
| 分析开销 | 极低（GPU-accelerated） | 中高 | 中 |
| 开源开放 | ✅（MIT License） | ❌ | ✅ |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验基于六种广泛使用的深度学习模型进行评估，涵盖 CNN 和 Transformer 架构：

| 模型 | 类型 | 批大小（Batch Size） | 缩写 |
|------|------|------------------|------|
| AlexNet | CNN | 128 | AN |
| ResNet18 | CNN | 32 | RN-18 |
| ResNet34 | CNN | 32 | RN-34 |
| GPT-2 | Transformer (Decoder) | 8 | GPT-2 |
| BERT | Transformer (Encoder) | 16 | BERT |
| Whisper (small) | Transformer (En/Dec) | 16 | Whisper |

这些模型用于推理和训练场景下的性能分析。

### 实验设置和评估指标
#### 硬件平台
| 机器 | CPU | GPU | 驱动版本 | 工具链 |
|------|-----|-----|----------|--------|
| A | Xeon Gold 5320 | 2×NVIDIA A100 (80GB) | Driver 570.86.10 | CUDA 12.1 |
| B | Ryzen 7 5800X | NVIDIA RTX 3060 | Driver 560.28.03 | CUDA 12.1 |
| C | Xeon Platinum 8568Y | AMD MI300X | ROCm 6.12.12 | ROCm 6.4 |

#### 评估指标
- **分析开销（Overhead）**：以执行时间延长倍数衡量，越低越好。
- **分析速度（Speedup）**：相比基线方法的加速比。
- **内存工作集大小（Working Set Size, WS）**：单个 kernel 执行期间的最大内存占用。
- **UVM 预取效率**：通过执行时间变化评估 object-level vs. tensor-level prefetching 效果。
- **跨平台一致性**：比较 NVIDIA 与 AMD 平台上的内存行为差异。

### 基线方法对比
- **Nsight Systems / ROCProfiler**：代表厂商级通用分析工具。
- **PyTorch Profiler / TensorFlow Profiler**：代表框架级分析工具。
- **Compute Sanitizer MemoryTracker** 和 **NVBit MemTrace**：作为传统 CPU-based 分析模式的代表，用于与 PASTA 的 GPU-accelerated 模式对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### （1）分析开销极低，GPU 加速显著
- 在 A100 上，PASTA 的 GPU-accelerated 分析比基于 CPU 的 Compute Sanitizer 快 **941×**，比 NVBit 快 **13,006×**。
- 在 RTX 3060 上，平均加速分别为 **627×**（vs. Compute Sanitizer）和 **7,353×**（vs. NVBit）。
- 图 10 显示，CPU-based 方法的“分析时间”占主导，可达数小时甚至数天；而 PASTA 将收集与分析融合于 GPU 内部，几乎无 stall。

#### （2）内存工作集远小于总内存足迹
- 表 V 显示，在训练阶段，平均内存足迹是工作集的 **3.79×**，说明大量内存未被充分利用。
- 大多数 kernel 的 median 和 90th percentile 工作集较小，表明存在优化空间（如 swapping 或 offloading）。

#### （3）UVM 预取策略效果因粒度和负载而异
- **非超订场景**（图 11）：
  - Object-level prefetching 平均提速 37–39%。
  - Tensor-level prefetching 提速约 26–30%。
- **3× 内存超订场景**（图 12）：
  - Object-level prefetching 导致严重 page thrashing，平均慢 **2.35–2.91×**。
  - Tensor-level 更精准，表现更稳定，尤其适合内存受限环境。
- **例外情况**：GPT-2 因其工作集小，在超订下仍受益于 object-level 预取。

#### （4）跨厂商行为对比揭示后端差异
- 图 14 显示，相同 GPT-2 模型在 NVIDIA 与 AMD 上内存使用趋势相似（三阶段模式），但：
  - NVIDIA 发出更少的分配/释放事件，峰值内存略高。
  - 推测原因：CUDA/cuDNN 与 HIP/MIOpen 的 kernel fusion 策略不同。

#### （5）多 GPU 场景准确反映并行语义
- 图 15 展示 Megatron-LM 在 Data Parallelism (DP)、Tensor Parallelism (TP)、Pipeline Parallelism (PP) 下的 per-GPU 内存使用：
  - DP：两 GPU 完全对称。
  - TP：内存峰值约为 DP 的一半，符合模型分片预期。
  - PP：GPU1 尾部负载更高（logits 计算），体现语义正确性。

---

## 4. 关键结论和发现

### 主要发现
1. **PASTA 实现了真正的跨层、跨厂商统一分析**：首次将 vendor-level profiling 与 DL framework-level semantics 结合，填补了传统工具的语义鸿沟。
2. **GPU-accelerated analysis 是降低开销的关键**：利用 GPU 并行性进行 in-situ 数据处理，使原本不可行的大规模 trace 分析变得高效可行。
3. **现代 DL 工作负载具有独特的内存访问模式**：pool-based memory management 使得传统的 object-level UVM 优化不再适用，需要感知 tensor 边界的精细化策略。
4. **细粒度分析能揭示深层性能瓶颈**：例如通过 cross-layer call stack（图 4）可直接定位到最热 kernel `at::cuda::blas::gemm_and_bias` 及其 Python 源码路径。

### 方法的局限性
- 当前主要针对 GPU，虽声称可推广至其他加速器（如 TPU），但尚未验证。
- 对某些极端密集的指令级 trace（如全 SASS instrumentation），即使 GPU 加速也可能带来可观开销。
- 依赖 vendor 提供的 profiling 接口（如 Compute Sanitizer/NVBit），若接口变更可能需同步更新。

### 未来工作方向
- **支持更多加速器架构**：如 Google TPU、Intel Gaudi、Apple Silicon GPU。
- **自动化分析建议生成**：结合 ML 模型从 trace 数据中自动识别瓶颈并推荐优化策略。
- **实时反馈闭环优化**：将分析结果反馈给 runtime 系统（如 PyTorch Dispatcher）实现动态调优。
- **增强可视化与交互式调试支持**：构建 GUI 工具提升用户体验。

---

> ✅ **补充信息**：PASTA 已完全开源（MIT License），代码地址为 [https://github.com/AccelProf/AccelProf](https://github.com/AccelProf/AccelProf)，并附有详细文档和复现实验脚本，确保研究成果可复现、可扩展。

</details>

---

### 15. [Interleaved Head Attention](https://arxiv.org/abs/2602.21371)

**Authors**: Sai Surya Duvvuri, Chanakya Ekbote, Rachit Bansal, Rishabh Tiwari, Devvrit Khatri, David Brandfonbrener, Paul Liang, Inderjit Dhillon, Manzil Zaheer  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.21371v1  

#### Abstract
Multi-Head Attention (MHA) is the core computational primitive underlying modern Large Language Models (LLMs). However, MHA suffers from a fundamental linear scaling limitation: $H$ attention heads produce exactly $H$ independent attention matrices, with no communication between heads during attenti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Interleaved Head Attention**

## **1. 主要贡献和创新点**

### **解决的问题**
- **Multi-Head Attention (MHA)** 存在**线性扩展瓶颈**：每个 attention head 独立计算，产生一个独立的 attention 矩阵，无法在不同 head 之间进行交互。
- 这种隔离限制了模型在**多步推理**（multi-step reasoning）任务中的表现，例如需要组合多个中间关系才能得出答案的场景（如“《霍比特人》的作者出生在哪里？”需先推断作者，再查其出生地）。

### **提出的新方法：Interleaved Head Attention (IHA)**
- **核心思想**：打破 MHA 中 head 之间的隔离，引入 **pseudo-heads**（伪头）机制，在 attention 计算前实现跨 head 的信息混合。
- **具体实现**：
  - 对于每个原始 head $ h $，通过可学习的线性变换生成 $ P $ 个 **pseudo-query**, **pseudo-key**, 和 **pseudo-value**（通常 $ P = H $，即 head 数量）。
  - 这些 pseudo-heads 在序列维度上进行 **interleaving**（交错），形成长度为 $ NP $ 的扩展序列。
  - 在此扩展序列上执行标准的 attention 操作，使得单个 head 内部可以产生多达 $ P^2 $ 种不同的 attention 模式。
- **兼容性**：IHA 保留了标准的 attention 操作符，因此与 **FlashAttention** 等高效 kernel 兼容。

### **相比现有方法的优势**
- **更高的表达能力**：IHA 严格泛化了 MHA（$ \text{MHA} \subset \text{IHA} $），能够表示 MHA 无法捕捉的非线性函数。
- **更优的参数效率**：在理论任务上，IHA 实现了**二次方级别的参数节省**。
- **更少的 head 需求**：完成相同复杂度的任务，所需 head 数量显著减少。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **合成任务（用于理论验证）**：
   - **Polynomial Filter**：模拟 k-hop 信息聚合，验证模型对多步依赖的建模能力。
   - **Count Permutation Match-3 (CPM-3)**：测试模型对**顺序敏感**的组合与计数能力，模拟多跳问答中的有序事实链。
2. **真实世界基准测试**：
   - **RULER**：长上下文建模基准，重点测试 **Multi-Key Retrieval** 任务。
   - **GSM8K**：小学数学应用题，评估推理能力。
   - **MATH-500**：高中数学竞赛题，评估复杂推理能力。
   - **MBPP** 和 **HumanEval**：代码生成任务。

### **实验设置和评估指标**
- **模型架构**：2.4B 参数的 decoder-only Transformer，26 层，$ H=20 $ 个 attention heads。
- **训练**：预训练 240B tokens，使用相同的超参数和训练预算，确保公平比较。
- **FLOP 匹配**：由于 IHA 会增加计算量（序列长度从 $ N $ 扩展到 $ NP $），实验采用 **hybrid local-global schedule**（4层滑动窗口 IHA + 1层全局 attention）来匹配计算成本。
- **评估指标**：
  - **RULER**：Exact Match (EM)，特别是 Multi-Key Retrieval 准确率。
  - **GSM8K/MATH-500**：Pass@1 (P@1) 和 Majority Vote @16 (Maj@16)。
  - **MBPP/HumanEval**：Pass@1 和 Pass@10。

### **基线方法对比**
- **Global Attention**：标准的 full attention MHA。
- **Global+Local**：交替使用全局和局部滑动窗口 attention。
- **Talking Heads**：在 softmax 前后混合 head 信息。
- **Diff Transformer**：使用两个 softmax 注意力图的差值。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
1. **长上下文检索（RULER）**：
   - 在 **Multi-Key Retrieval** 任务上，IHA 相比 Global Attention 取得了 **10-20% 的相对提升**（在 4k 到 16k 上下文长度下）。
   - 在 16k 长度上，提升高达 **112%**。
   - RULER 整体平均 EM 达到 **44.0%**，优于所有基线。

2. **推理能力（Fine-tuned on OpenThoughts）**：
   - **GSM8K**：Maj@16 达到 **54.2%**，比 Global Attention 提升 **+5.8%**。
   - **MATH-500**：Maj@16 达到 **18.4%**，比 Global Attention 提升 **+2.8%**。
   - 平均排名 (Avg. Rank) 为 **1.5**，排名第一。

3. **预训练阶段推理能力（Zero-shot/5-shot）**：
   - 即使在未微调时，IHA 在 GSM8K 和 MATH-500 上也表现出色，表明其更强的内在推理能力。

### **与基线方法的对比结果**
- IHA 在所有推理任务上均**显著优于** Global Attention、Global+Local 和 Diff Transformer。
- Talking Heads 在代码生成任务（MBPP）上表现最好，而 IHA 在逻辑推理任务上优势明显，说明不同方法各有侧重。

### **消融实验结果**
- 虽然论文未明确列出消融表，但其理论分析和设计本身构成了强有力的消融：
  - **理论证明**：Thm. 2 证明了 IHA 严格包含 MHA，且当 $ P \geq 2 $ 时是严格子集，证明了额外参数的有效性。
  - **参数效率分析**：在 Polynomial Filter 任务上，MHA 需要 $ O(kn^2) $ 参数，而 IHA 仅需 $ O(\sqrt{k}n^2) $，证明了其效率优势。

---

## **4. 关键结论和发现**

### **主要发现**
1. **MHA 的根本瓶颈在于 head 隔离**，这限制了其在组合性、多步推理任务上的扩展能力。
2. **IHA 通过引入 pseudo-heads 和 interleaving 机制，成功打破了这一瓶颈**，实现了 head 间的有效通信。
3. **IHA 不仅在理论上更强大**（更高的表达能力和参数效率），**在实践中也带来了显著的性能提升**，尤其是在长上下文检索和数学推理任务上。

### **方法的局限性**
- **计算开销**：全局 IHA 的计算复杂度为 $ O(P^2N^2d) $，远高于标准 MHA。虽然通过 hybrid schedule 缓解，但仍是一个挑战。
- **固定 P**：当前方法中 $ P $ 是一个超参数，未来可能探索自适应分配 pseudo-heads 的方法。

### **未来工作方向**
- 设计更高效的计算策略以支持全局 IHA。
- 探索自适应的 pseudo-head 分配机制。
- 将 IHA 应用到 **encoder-decoder** 架构和 **vision** 模型中。

</details>

---

### 16. [Learning Unknown Interdependencies for Decentralized Root Cause Analysis in Nonlinear Dynamical Systems](https://arxiv.org/abs/2602.21928)

**Authors**: Ayush Mohanty, Paritosh Ramanan, Nagi Gebraeel  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.21928v1  

#### Abstract
Root cause analysis (RCA) in networked industrial systems, such as supply chains and power networks, is notoriously difficult due to unknown and dynamically evolving interdependencies among geographically distributed clients. These clients represent heterogeneous physical processes and industrial as...

---

### 17. [SymTorch: A Framework for Symbolic Distillation of Deep Neural Networks](https://arxiv.org/abs/2602.21307)

**Authors**: Elizabeth S. Z. Tan, Adil Soubki, Miles Cranmer  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.21307v1  

#### Abstract
Symbolic distillation replaces neural networks, or components thereof, with interpretable, closed-form mathematical expressions. This approach has shown promise in discovering physical laws and mathematical relationships directly from trained deep learning models, yet adoption remains limited due to...

---

### 18. [HiPPO Zoo: Explicit Memory Mechanisms for Interpretable State Space Models](https://arxiv.org/abs/2602.21340)

**Authors**: Jack Goffinet, Casey Hanks, David E. Carlson  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.21340v1  

#### Abstract
Representing the past in a compressed, efficient, and informative manner is a central problem for systems trained on sequential data. The HiPPO framework, originally proposed by Gu & Dao et al., provides a principled approach to sequential compression by projecting signals onto orthogonal polynomial...

---

### 19. [RuCL: Stratified Rubric-Based Curriculum Learning for Multimodal Large Language Model Reasoning](https://arxiv.org/abs/2602.21628)

**Authors**: Yukun Chen, Jiaming Li, Longze Chen, Ze Gong, Jingpeng Li, Zhen Qin, Hengyu Chang, Ancheng Xu, Zhihao Yang, Hamid Alinejad-Rokny, Qiang Qu, Bo Zheng, Min Yang  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.21628v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a prevailing paradigm for enhancing reasoning in Multimodal Large Language Models (MLLMs). However, relying solely on outcome supervision risks reward hacking, where models learn spurious reasoning patterns to satisfy final answer ...

---

### 20. [DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference](https://arxiv.org/abs/2602.21548)

**Authors**: Yongtong Wu, Shaoyuan Chen, Yinmin Zhong, Rilin Huang, Yixuan Tan, Wentao Zhang, Liyue Zhang, Shangyan Zhou, Yuxuan Liu, Shunfeng Zhou, Mingxing Zhang, Xin Jin, Panpan Huang  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.21548v1  

#### Abstract
The performance of multi-turn, agentic LLM inference is increasingly dominated by KV-Cache storage I/O rather than computation. In prevalent disaggregated architectures, loading the massive KV-Cache from external storage creates a fundamental imbalance: storage NICs on prefill engines become bandwid...

---

### 21. [LLMTailor: A Layer-wise Tailoring Tool for Efficient Checkpointing of Large Language Models](https://arxiv.org/abs/2602.22158)

**Authors**: Minqiu Sun, Xin Huang, Luanzheng Guo, Nathan R. Tallent, Kento Sato, Dong Dai  
**Category**: cs.DC  
**Published**: 2026-02-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.22158v1  

#### Abstract
Checkpointing is essential for fault tolerance in training large language models (LLMs). However, existing methods, regardless of their I/O strategies, periodically store the entire model and optimizer states, incurring substantial storage overhead and resource contention. Recent studies reveal that...

---

### 22. [AgentLTV: An Agent-Based Unified Search-and-Evolution Framework for Automated Lifetime Value Prediction](https://arxiv.org/abs/2602.21634)

**Authors**: Chaowei Wu, Huazhu Chen, Congde Yuan, Qirui Yang, Guoqing Song, Yue Gao, Li Luo, Frank Youhua Chen, Mengzhuo Guo  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.21634v1  

#### Abstract
Lifetime Value (LTV) prediction is critical in advertising, recommender systems, and e-commerce. In practice, LTV data patterns vary across decision scenarios. As a result, practitioners often build complex, scenario-specific pipelines and iterate over feature processing, objective design, and tunin...

---

### 23. [VecGlypher: Unified Vector Glyph Generation with Language Models](https://arxiv.org/abs/2602.21461)

**Authors**: Xiaoke Huang, Bhavul Gauri, Kam Woh Ng, Tony Ng, Mengmeng Xu, Zhiheng Liu, Weiming Ren, Zhaochong An, Zijian Zhou, Haonan Qiu, Yuyin Zhou, Sen He, Ziheng Wang, Tao Xiang, Xiao Han  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21461v1  

#### Abstract
Vector glyphs are the atomic units of digital typography, yet most learning-based pipelines still depend on carefully curated exemplar sheets and raster-to-vector postprocessing, which limits accessibility and editability. We introduce VecGlypher, a single multimodal language model that generates hi...

---

### 24. [Improving Implicit Discourse Relation Recognition with Natural Language Explanations from LLMs](https://arxiv.org/abs/2602.21763)

**Authors**: Heng Wang, Changxing Wu  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21763v1  

#### Abstract
Implicit Discourse Relation Recognition (IDRR) remains a challenging task due to the requirement for deep semantic understanding in the absence of explicit discourse markers. A further limitation is that existing methods only predict relations without providing any supporting explanations. Recent ad...

---

### 25. [D-COT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in Small Language Models](https://arxiv.org/abs/2602.21786)

**Authors**: Shunsuke Ubukata  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21786v1  

#### Abstract
Chain-of-Thought (CoT) distillation from Large Language Models (LLMs) often induces "overthinking" in Small Language Models (SLMs), leading to performance degradation and excessive token consumption. In this study, we propose Disciplined Chain-of-Thought (D-CoT), a novel framework that enforces a st...

---

### 26. [Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data](https://arxiv.org/abs/2602.21320)

**Authors**: Emre Can Acikgoz, Cheng Qian, Jonas H\"ubotter, Heng Ji, Dilek Hakkani-T\"ur, Gokhan Tur  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21320v1  

#### Abstract
Large language models (LLMs) are becoming the foundation for autonomous agents that can use tools to solve complex tasks. Reinforcement learning (RL) has emerged as a common approach for injecting such agentic capabilities, but typically under tightly controlled training setups. It often depends on ...

---

### 27. [DocDjinn: Controllable Synthetic Document Generation with VLMs and Handwriting Diffusion](https://arxiv.org/abs/2602.21824)

**Authors**: Marcel Lamott, Saifullah Saifullah, Nauman Riaz, Yves-Noel Weweler, Tobias Alt-Veit, Ahmad Sarmad Ali, Muhammad Armaghan Shakir, Adrian Kalwa, Momina Moetesum, Andreas Dengel, Sheraz Ahmed, Faisal Shafait, Ulrich Schwanecke, Adrian Ulges  
**Category**: cs.LG  
**Published**: 2026-02-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.21824v1  

#### Abstract
Effective document intelligence models rely on large amounts of annotated training data. However, procuring sufficient and high-quality data poses significant challenges due to the labor-intensive and costly nature of data acquisition. Additionally, leveraging language models to annotate real docume...

---

### 28. [fEDM+: A Risk-Based Fuzzy Ethical Decision Making Framework with Principle-Level Explainability and Pluralistic Validation](https://arxiv.org/abs/2602.21746)

**Authors**: Abeer Dyoub, Francesca A. Lisi  
**Category**: cs.AI  
**Published**: 2026-02-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.21746v1  

#### Abstract
In a previous work, we introduced the fuzzy Ethical Decision-Making framework (fEDM), a risk-based ethical reasoning architecture grounded in fuzzy logic. The original model combined a fuzzy Ethical Risk Assessment module (fERA) with ethical decision rules, enabled formal structural verification thr...

---

### 29. [Explore-on-Graph: Incentivizing Autonomous Exploration of Large Language Models on Knowledge Graphs with Path-refined Reward Modeling](https://arxiv.org/abs/2602.21728)

**Authors**: Shiqi Yan, Yubo Chen, Ruiqi Zhou, Zhengxi Yao, Shuai Chen, Tianyi Zhang, Shijie Zhang, Wei Qiang Zhang, Yongfeng Huang, Haixin Duan, Yunqi Zhang  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.21728v1  

#### Abstract
The reasoning process of Large Language Models (LLMs) is often plagued by hallucinations and missing facts in question-answering tasks. A promising solution is to ground LLMs' answers in verifiable knowledge sources, such as Knowledge Graphs (KGs). Prevailing KG-enhanced methods typically constraine...

---

### 30. [Personalized Graph-Empowered Large Language Model for Proactive Information Access](https://arxiv.org/abs/2602.21862)

**Authors**: Chia Cheng Chang, An-Zi Yen, Hen-Hsen Huang, Hsin-Hsi Chen  
**Category**: cs.CL  
**Published**: 2026-02-26  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.21862v1  

#### Abstract
Since individuals may struggle to recall all life details and often confuse events, establishing a system to assist users in recalling forgotten experiences is essential. While numerous studies have proposed memory recall systems, these primarily rely on deep learning techniques that require extensi...

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
