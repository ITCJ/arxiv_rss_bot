# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-05 06:15:23 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [HyperParallel: A Supernode-Affinity AI Framework](https://arxiv.org/abs/2603.03731)

**Authors**: Xin Zhang, Beilei Sun, Teng Su, Qinghua Zhang, Chong Bao, Lei Chen, Xuefeng Jin  
**Category**: cs.DC  
**Published**: 2026-03-05  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2603.03731v1  

#### Abstract
The emergence of large-scale, sparse, multimodal, and agentic AI models has coincided with a shift in hardware toward supernode architectures that integrate hundreds to thousands of accelerators with ultra-low-latency interconnects and unified memory pools. However, existing AI frameworks are not de...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《HyperParallel: A Supernode-Affinity AI Framework》核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

随着大模型向**trillion-scale 参数**发展，现代 AI 模型呈现出以下趋势：
- **架构异构化**：如 Mixture-of-Experts (MoE)、omni-modal（多模态融合）、agentic RL（智能体强化学习）等；
- **计算负载不均衡**：不同模块间计算强度差异大，导致传统并行策略出现 pipeline bubbles 和通信瓶颈；
- **内存压力剧增**：激活值（activations）、KV caches、参数等中间状态规模爆炸；
- **硬件演进为 supernode 架构**：集成数百至数千加速器，具备统一内存池（unified memory pool）和超低延迟互联（如 Huawei Atlas 900），但现有 AI 框架无法有效利用其潜力。

现有框架（如 PyTorch + DeepSpeed/Megatron）面临三大挑战：
1. **编程复杂度高**：需手动设计复杂的并行策略（DP/TP/PP/EP），难以适应动态拓扑；
2. **缺乏对 MPMD 支持**：主流采用 SPMD 范式，难以处理异构任务并发；
3. **内存管理负担重**：开发者需手动管理 HBM 与 DRAM 间的 offload/prefetch，效率低下。

---

### **提出了什么新方法或新思路**

本文提出一种**supernode-affinity AI framework**，将整个 supernode 视为一个“逻辑上的巨型计算机”，并在 MindSpore 中实现了 **HyperParallel 架构**，包含三个核心技术组件：

#### ✅ **HyperOffload**
- **功能**：实现自动化的分层内存管理（automated hierarchical memory management）
- **机制**：利用 supernode 的统一内存池，将模型状态（weights, activations, KV caches）按需从 DRAM 预取到 HBM，作为高速缓存使用
- **优势**：解耦 computation 与 state 存储，缓解 HBM 内存墙问题

#### ✅ **HyperMPMD**
- **功能**：支持细粒度的 Multiple Program, Multiple Data 并行
- **机制**：突破传统 SPMD 限制，在三种维度上实现灵活调度：
  - **Intra-sub-model**：子模型内部多核并发（AICube/AIVector）
  - **Inter-sub-model**：跨子模块负载均衡（如 video/audio encoder vs LLM）
  - **Cross-model/task**：多任务协同调度（如 actor/learner/critic 在 RL 中共存）
- **优势**：显著提升硬件利用率，消除 pipeline bubbles 和 straggler 效应

#### ✅ **HyperShard**
- **功能**：提供声明式并行编程接口（declarative parallel programming）
- **机制**：通过 `Layout(device_matrix, alias_name, tensor_map)` 接口抽象设备拓扑，自动推导并插入通信算子
- **优势**：彻底解耦算法逻辑与并行策略，降低开发门槛

> 🔁 总体设计理念：**以 supernode 为中心进行软硬协同抽象**，把系统级优化内建于框架中，而非依赖外部扩展层。

---

### **相比现有方法的优势**

| 维度 | 传统框架（PyTorch + DeepSpeed/Megatron） | HyperParallel |
|------|----------------------------------------|---------------|
| 并行范式 | 主要基于 SPMD，静态划分 | 支持动态 MPMD，细粒度调度 |
| 内存管理 | 手动 offload 或 ZeRO 分片 | 自动化统一内存池调度（HyperOffload） |
| 编程模式 | 命令式（imperative），紧耦合 | 声明式（declarative），完全解耦 |
| 拓扑感知 | 弱，需人工调优 | 内建 topology-aware 调度 |
| 可移植性 | 差，代码绑定特定集群配置 | 高，策略可复用 |

---

## 2. 核心实验方法和设置

### **实验平台**
- **硬件**：基于华为 **Matrix384（Atlas 900）supernode** 架构
  - 包含 384 个 Ascend 910C NPU 和 192 个 Kunpeng CPU
  - 支持统一内存寻址、P2P 互联、4D 全连接拓扑
  - 通信带宽提升 15×，单跳延迟降至 200ns

### **测试模型与场景**
| 场景 | 模型示例 | 特征 |
|------|----------|------|
| **训练** | Llama-8B, DeepSeek-V3 (671B), Qwen-3 | MoE、长序列、多模态 |
| **推理** | Llama-8B, DeepSeek-V3 | 高吞吐、低延迟要求 |
| **强化学习** | Agent-based RL workflows | Actor/Learner/Critic 多任务并发 |

### **评估指标**
- **训练性能**：每步迭代时间（iteration time per step）
- **推理能力**：最大支持 sequence length（在相同延迟约束下）
- **资源利用率**：
  - Pipeline bubble 比例
  - Communication masking ratio（通信隐藏率）
  - Cluster-wide MFU（Model Flops Utilization）
- **开发效率**：新算法并行化所需时间、调优周期

### **基线方法对比**
- **DeepSpeed-ZeRO3 / ZeRO-Offload**
- **Megatron-LM**（TP/PP/DP 组合）
- **Pathways**（Google 的异步执行框架）
- **原生 PyTorch + 手动 SPMD 实现**

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **HyperOffload 实验结果**
| 场景 | 指标 | 结果 |
|------|------|------|
| **Llama-8B 训练** | Iteration time per step | 从 5.2s → **4.08s**（↑ ~21.5%） |
| **Llama-8B 推理** | 最大 sequence length | 从 71K → **123K**（↑ ~73%） |

> 💡 表明通过自动内存交换大幅提升了内存可用性，允许更长上下文处理。

#### ✅ **HyperMPMD 实验结果**
| 场景 | 指标 | 结果 |
|------|------|------|
| **Omni-modal 模型训练** | Pipeline bubble 减少 | 减少 10%–40%，整体性能 ↑ **~15%** |
| **MoE 模型（DeepSeek-V3）** | Communication masking ratio | 从 61% → **90%**（接近理想值） |
| **RL 多任务调度** | 集群资源利用率 | 提升 **15%**，消除 straggler 效应 |

> 💡 显示 MPMD 在异构负载下的优越调度能力。

#### ✅ **HyperShard 开发效率提升**
| 指标 | 传统方式 | HyperShard |
|------|---------|-----------|
| 新模型并行化时间 | 1–2 周（资深工程师） | **<1 天** |
| 并行策略调优周期 | 数天 | **数小时** |

> 💡 极大降低开发门槛，提升研发敏捷性。

---

### **消融实验（Ablation Study）**
虽然文中未明确列出表格形式的消融实验，但从多个案例分析中可归纳出以下结论：

| 组件移除 | 影响 |
|--------|------|
| 移除 HyperOffload | 必须启用复杂 ND-SPMD，并行通信开销上升，内存受限 |
| 移除 HyperMPMD | 回归 SPMD 后 pipeline bubbles 显著增加，MoE 通信无法有效隐藏 |
| 移除 HyperShard | 需手动编写通信逻辑，代码不可移植，调试成本飙升 |

> ➕ 三者协同作用明显：**HyperShard 定义策略 → HyperMPMD 执行调度 → HyperOffload 管理状态**

---

## 4. 关键结论和发现

### **主要发现**

1. **Supernode Affinity 是下一代 AI 框架的核心设计原则**
   - 必须将 supernode 视为单一逻辑计算机，才能充分发挥其统一内存、超低延迟互联的优势。

2. **声明式 + 自动化是解决复杂性的关键路径**
   - HyperShard 的 declarative 接口使开发者无需关心底层拓扑，极大简化编程模型。

3. **MPMD 比 SPMD 更适合未来异构 workload**
   - 对 MoE、多模态、RL 等非均匀负载，MPMD 可实现更精细的任务调度与资源分配。

4. **统一内存池必须由框架自动管理**
   - 手动控制 HBM ↔ DRAM 数据迁移已不可行；HyperOffload 的自动预取机制能高效隐藏内存访问延迟。

---

### **方法的局限性**

1. **当前实现依赖特定硬件生态**
   - 主要在华为 Ascend 系列 NPU 上验证，对 CUDA 生态兼容性尚未展示。

2. **HyperMPMD 的运行时调度开销未量化**
   - 尽管声称高效，但大规模 task graph 下的调度延迟仍可能成为瓶颈（类似 Pathways 的 central coordinator 问题）。

3. **暂未支持跨数据中心训练**
   - 当前聚焦于单 supernode 内部优化，未涉及 geo-distributed 场景。

4. **缺乏对小型模型或边缘场景的支持**
   - 设计重心在 trillion-scale LLM，轻量级应用可能过度工程化。

---

### **未来工作方向**

1. **扩展至多 supernode 协同训练**
   - 实现跨 supernode 的 dynamic scaling 与 fault tolerance

2. **增强异构任务调度能力**
   - 支持更多类型的 agent-based workflow（如 multi-agent simulation）

3. **构建通用 execution layer**
   - 类似 Pathways 的愿景，打造统一的 AI workload runtime

4. **开放生态与跨平台支持**
   - 向 CUDA/GPU 生态迁移，推动标准化 adoption

5. **引入 AI-driven 自动调优**
   - 利用 ML 模型预测最优 parallel strategy 和 memory layout

---

> 📌 **总结一句话**：  
> **HyperParallel 通过“声明式编程 + 细粒度 MPMD + 自动内存管理”三位一体的设计，首次系统性地实现了面向 supernode 的 AI 框架抽象，为 trillion-scale 模型的高效训练与推理提供了新的软件基础设施范式。**

</details>

---

### 2. [Entropic-Time Inference: Self-Organizing Large Language Model Decoding Beyond Attention](https://arxiv.org/abs/2603.03310)

**Authors**: Andrew Kiruluta  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.03310v1  

#### Abstract
Modern large language model (LLM) inference engines optimize throughput and latency under fixed decoding rules, treating generation as a linear progression in token time. We propose a fundamentally different paradigm: entropic\-time inference, where decoding is governed by the flow of uncertainty ra...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Entropic-Time Inference: Self-Organizing Large Language Model Decoding Beyond Attention*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前大语言模型（LLM）推理系统将生成过程视为**基于 token 步骤的线性执行任务**，所有解码步骤在计算资源分配上被视为等价。然而，不同 token 步骤对不确定性（entropy）的减少程度差异巨大：
- 有些步骤是语义上的关键决策，熵迅速下降；
- 有些则是语法填充、重复或长尾模糊，熵保持高位。

这种“固定时间步”范式导致大量计算资源被浪费在对输出质量影响甚微的步骤上，尤其是在长上下文或高并发场景中。

### 提出了什么新方法或新思路
论文提出 **Entropic-Time Inference（熵时推理）** ——一种全新的 LLM 推理控制框架，其核心思想是：
> 将推理中的“时间”重新定义为**不可逆熵流（irreversible entropy flow）**，而非 token 索引。

在此基础上，构建了一个**自组织（self-organizing）的推理架构**，通过统一的熵控制目标，联合调节以下三个系统级组件：
1. **Entropy-Aware Scheduling（熵感知调度）**  
   调度器优先处理预期熵减高的序列，避免为已收敛序列浪费算力。
2. **Entropic Attention Pruning（熵驱动注意力剪枝）**  
   动态稀疏化 paged attention 中的 KV 块，仅保留信息贡献大的内存区域。
3. **Entropy-Stabilized Sampling（熵稳定采样）**  
   自适应调整 sampling temperature，使预测分布熵稳定在目标值附近，防止过早坍缩或持续震荡。

该方法不依赖新模型结构，而是将**熵作为第一类系统原语（first-class control signal）**，实现推理引擎层面的动态资源调配。

### 相比现有方法的优势
| 维度 | 现有方法（如 vLLM、FlashAttention） | Entropic-Time Inference |
|------|-------------------------------|--------------------------|
| 时间定义 | Token-indexed time（索引时间） | **Entropic time（熵时）** |
| 控制信号 | 固定参数、公平调度 | **全局熵反馈信号** |
| 注意力机制 | 全连接或静态稀疏 | **动态、信息导向的稀疏化** |
| 随机性控制 | 固定 temperature/nucleus | **闭环温度调控** |
| 系统视角 | 分离优化模块 | **耦合自组织系统** |

**优势总结**：
- 更高效地利用计算资源，提升吞吐量与能效；
- 实现“按需计算”，在不确定性高时多投入，在低时少投入；
- 支持与 Speculative Decoding、MoE 等技术正交集成，具备良好兼容性。

---

## 2. 核心实验方法和设置

### 数据集与任务
- 使用**异构混合提示集**进行评估，涵盖：
  - Instruction-following（指令遵循）
  - Long-context reasoning（长程推理）
  - Free-form generation（自由生成）
- 提示来自真实应用场景分布，确保多样性。

### 实验设置
- **模型**：预训练的 decoder-only Transformer（具体未指明，假设为类似 LLaMA 架构）
- **后端**：基于 vLLM 扩展实现 entropic-time 控制逻辑
- **硬件环境一致**，保证比较公平
- 所有序列运行相同解码步数（除非提前终止）

### 评估指标
| 指标类别 | 具体指标 |
|--------|---------|
| 效率 | Latency（延迟）、Throughput（吞吐量）、FLOPs/token、KV-cache bandwidth |
| 信息效率 | $ \frac{dT}{dC} $（单位资源下的熵减率） |
| 动态行为 | Entropy collapse rate、Entropy variance |
| 输出质量 | ROUGE、BLEU、人工抽样评估（human eval） |

### 基线方法对比
- **Baseline**：标准推理（公平调度 + 密集注意力 + 固定 temperature）
- 各消融配置：
  - Micro-only：仅启用 entropy-stabilized sampling
  - Macro-only：仅启用 entropy-aware scheduling
  - Meso-only：仅启用 entropic attention pruning
- 最终对比完整 entropic-time loop 与 baseline 的综合表现

---

## 3. 主要实验结果和性能指标

### 关键性能数据（相对于 Baseline = 1.00）

| Configuration | Latency | Throughput | $ dT/dC $ | Quality |
|--------------|--------|-----------|------------|--------|
| Baseline     | 1.00   | 1.00      | 1.00       | 1.00   |
| Sampling only | 0.98   | 1.02      | 1.08       | 1.00   |
| Scheduling only | 0.88 | 1.15      | 1.12       | 1.00   |
| Attention only | 0.85  | 1.20      | 1.25       | 0.98   |
| **Full system** | **0.70** | **1.40** | **1.55** | **1.00** |

> ✅ 表格来源：原文 Table 1

### 与基线方法的对比结果
- **端到端延迟降低 25–35%**
- **吞吐量提升 30–45%**
- **每单位计算的熵减提升 40–60%**
- **输出质量保持稳定甚至略有改善**

表明该方法不仅提升了效率，还增强了生成稳定性。

### 消融实验结果
#### （1）Micro-Scale Only（熵稳定采样）
- 减少熵震荡达 15–20%
- 加速高熵状态向确定性的过渡
- 对整体效率提升有限（因 attention/scheduling 成本不变）

#### （2）Macro-Scale Only（熵感知调度）
- 平均延迟 ↓10–15%，吞吐 ↑12–18%
- 显著改善批处理利用率，尤其在混合负载下
- 不改变单步计算量，但优化资源分配

#### （3）Meso-Scale Only（熵驱动注意力剪枝）
- Attention FLOPs ↓20–30%，KV-cache 带宽 ↓15–25%
- 在涉及长距离依赖的任务中出现轻微质量下降（↓0.02 BLEU/ROUGE）
- 表明**孤立剪枝存在风险，需全局熵协调**

#### （4）Full System（三者协同）
- 性能增益呈**超可加性（super-additive）**，远超各部分之和
- 展现出**自组织行为**：调度聚焦未解序列，剪枝仅在低熵时激进，采样维持稳态
- 即使在资源受限下也能优雅退化（graceful degradation）

---

## 4. 关键结论和发现

### 主要发现
1. **熵可以作为有效的全局控制信号**  
   Shannon entropy 是衡量生成进展的自然尺度，适合作为推理系统的“操作时间”基础。

2. **自组织推理是可行且高效的**  
   无需集中优化，局部熵反馈即可促成全局协调，形成稳定、高效的动态平衡。

3. **效率提升源于跨层级耦合**  
   单独优化任一模块收益有限；真正突破来自 scheduling、attention、sampling 的**联合熵控制闭环**。

4. **与主流加速技术正交**  
   该框架可无缝集成至 vLLM、Speculative Decoding、MoE 等系统，提供更高层次的资源治理能力。

### 方法的局限性
- **依赖熵估计准确性**：若模型严重过自信（overconfident），会导致熵低估，引发过度剪枝或调度偏差。
- **引入额外开销**：尽管使用 top-k 和 tail-corrected 估算降低了成本，但仍有一定 overhead。
- **短文本增益有限**：在极短生成或高度确定性任务中，控制开销可能超过收益。
- **未改进模型本身**：仅作用于 inference-time，不影响训练质量或校准性。

### 未来工作方向
- **训练时集成熵控制**：探索在训练阶段引入 entropy regularization，提升推理时控制鲁棒性。
- **非自回归扩展**：将 entropic-time 思想推广至 parallel decoding 或 speculative generation 框架。
- **硬件协同设计**：开发支持 entropy-aware memory 访问和动态功耗管理的专用 inference chip。
- **不确定性建模增强**：结合 Bayesian NN 或 ensemble 方法，提供更可靠的 epistemic uncertainty 估计。

---

> **总结一句话**：  
> *Entropic-Time Inference 将 LLM 推理从“机械执行”转变为“智能热力学过程”，以熵为指挥棒，实现了计算资源的按需分配与系统级自组织，为下一代高效、自适应的 inference engine 提供了全新范式。*

</details>

---

### 3. [Combating data scarcity in recommendation services: Integrating cognitive types of VARK and neural network technologies (LLM)](https://arxiv.org/abs/2603.03309)

**Authors**: Nikita Zmanovskii  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.03309v1  

#### Abstract
Cold start scenarios present fundamental obstacles to effective recommendation generation, particularly when dealing with users lacking interaction history or items with sparse metadata. This research proposes an innovative hybrid framework that leverages Large Language Models (LLMs) for content sem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Combating data scarcity in recommendation services: Integrating cognitive types of VARK and neural network technologies (LLM)*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于推荐系统中的 **cold start problem**（冷启动问题），即在以下场景中难以生成有效推荐：
- 新用户缺乏交互历史（user-side cold start）
- 新物品元数据稀疏或缺失（item-side cold start）
- 平台初始阶段无累积数据

此外，作者指出当前系统普遍忽视两个关键维度：
- **个体认知差异**（如信息接收偏好：视觉、听觉、读写、动觉等）
- **动态心理状态变化**（如疲劳程度、注意力水平、可用时间等）

### 🚀 提出的新方法与创新思路
提出了一种融合 **Large Language Models (LLMs)**、**Knowledge Graphs (KGs)** 和 **VARK 认知模型** 的混合架构，共包含六个模块：

| 模块 | 功能 |
|------|------|
| **Semantic Metadata Enhancement** | 利用 LLM 对稀疏/非结构化物品描述进行语义增强，提取实体、关系、难度等级、先修要求、目标受众及 VARK 对齐性 |
| **Dynamic Graph Construction** | 构建多关系知识图谱（multi-relational KG），支持基于图神经网络的推理 |
| **VARK-Based Profiling** | 基于 VARK 学习风格框架建立用户认知画像，捕捉个体信息处理偏好 |
| **Mental State Estimation** | 结合上下文信号（时间、设备、会话行为）估计用户的当前认知负荷、注意力跨度和复杂度接受能力 |
| **Graph-Enhanced Retrieval + LLM-Powered Ranking** | 多策略候选生成 + LLM 驱动排序，结合相关性、多样性、惊喜度等标准 |
| **Adaptive Interface Design with Iterative Learning** | 动态调整推荐呈现方式（图文比例、交互元素等），并持续从反馈中学习优化 |

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **语义理解深度** | 超越传统 content-based 方法仅依赖关键词匹配，利用 LLM 实现深层语义解析 |
| **个性化粒度提升** | 不仅考虑兴趣偏好，还纳入 **VARK 认知类型** 和 **实时心理状态**，实现更精细适配 |
| **可解释性增强** | 推荐附带自然语言解释，提高透明度与用户信任感 |
| **跨域适应性强** | 架构为 domain-agnostic，适用于教育、电商、健康等多个领域 |
| **端到端冷启动应对** | 同时解决 user-side 与 item-side 冷启动，并引入迭代学习机制逐步完善模型 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- 主要使用 **MovieLens-1M** 数据集：
  - 包含约 6,040 名用户对 3,706 部电影的 1,000,209 条评分（1–5 分）
  - 提供用户人口统计信息（年龄、性别、职业）和电影元数据（标题、类型、年份）

#### 冷启动模拟设计：
- 将用户随机划分为训练集（80%）和“冷启动”测试集（20%）
- 测试集中用户视为“全新用户”，无任何历史交互
- 使用 **GPT-3.5-turbo** 自动生成每部电影的丰富语义描述
- 用户 VARK 偏好通过抽样真实分布随机分配；电影 VARK 对齐性由类型推断得出

### 🧪 实验设置
- **任务**：为冷启动用户预测其未评分电影的偏好，生成 Top-K 推荐列表
- **输出形式**：
  - 排序后的推荐列表 $ R $
  - 自适应呈现格式 $ F $
  - LLM 生成的自然语言解释 $ E $

### 📈 评估指标
| 指标 | 定义与用途 |
|------|-----------|
| **HR@K (Hit Rate@K)** | 至少有一个相关项出现在 Top-K 中的比例，衡量覆盖率 |
| **nDCG@K (normalized Discounted Cumulative Gain@K)** | 考虑排名位置的相关性加权得分，反映排序质量 |
| **Recall@K** | 在 Top-K 中捕获的相关项目比例 |
| **Unique Top-1** | 不同用户 Top-1 推荐项的数量，用于衡量多样性与个性化程度 |

### ⚖️ 基线方法对比
| 方法 | 描述 |
|------|------|
| **Random** | 随机打乱推荐项（下界） |
| **Popularity** | 推荐最受欢迎的 K 个物品（强基线） |
| **Embedding Cosine** | 基于 Sentence Transformer 编码标题/类型，计算与用户伪画像的余弦相似度 |
| **Candidates Only** | 仅使用候选生成模块（无 LLM 排序） |
| **Ours (CE Rerank)** | 完整系统，采用 cross-encoder 进行高效重排序代替完整 LLM 排序 |

---

## 3. 主要实验结果和性能指标

### 📉 关键性能数据（Table 1）

| Model | HR@10 | nDCG@10 | Recall@50 | Recall@200 | Recall@1000 |
|-------|--------|---------|------------|-------------|---------------|
| Random | 0.005 ± 0.006 | 0.002 ± 0.003 | ~0.002 | ~0.003 | ~0.004 |
| Popularity | **0.268 ± 0.018** | **0.224 ± 0.014** | ~0.002 | ~0.003 | ~0.004 |
| Embedding Cosine | 0.101 ± 0.021 | 0.050 ± 0.011 | ~0.002 | ~0.003 | ~0.004 |
| Candidates Only | 0.011 ± 0.003 | 0.004 ± 0.001 | ~0.000 | ~0.001 | ~0.001 |
| **Ours (CE Rerank)** | 0.008 ± 0.005 | 0.005 ± 0.002 | ~0.000 | ~0.001 | ~0.001 |

> 注：所有 Recall@K 数值极低，表明候选池中真正相关的物品极少被召回。

### 🔍 与基线方法对比分析
- **Popularity 表现最强**：HR@10 达到 0.268，显著优于其他所有方法（p < 0.001）
  - 原因：MovieLens 数据存在强烈流行度偏差，新用户倾向于观看热门电影
- **Embedding Cosine 次之**：HR@10 = 0.101，显示基于语义匹配有一定作用，但仍远低于流行度法
- **本文方法绝对性能较低**：HR@10 = 0.008，nDCG@10 = 0.005
  - 但相比 “Candidates Only” 有轻微提升（nDCG 从 0.004 → 0.005），说明 LLM 排序模块仍具一定价值

### 🌀 多样性与个性化表现（Table 2）

| Model | Unique Top-1 |
|-------|--------------|
| Random | 1.0 |
| Popularity | 1.0 |
| Embedding Cosine | 4.0 |
| Candidates Only | 4.0 |
| **Ours (CE Rerank)** | **3.0** |

- 尽管准确率低，**本方法实现了适度个性化**（Unique Top-1 = 3），优于完全统一推荐的 Popularity 方法
- 显示系统确实在尝试根据 VARK 和上下文定制推荐

### ❌ 消融实验启示
虽然未明确列出消融实验表格，但从模块分析可知：
- **候选生成阶段是瓶颈**：Recall@K 极低 → 即使后续排序再优也无法弥补
- **LLM 排序带来有限增益**：nDCG 微幅上升，受限于候选池质量
- **VARK 过滤可能过于激进**：可能导致部分潜在相关项被提前排除

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Popularity 在极端冷启动中极具竞争力**  
   在缺乏交互数据的情况下，推荐流行内容是一种稳健且有效的策略，尤其在像 MovieLens 这类具有强流行偏好的数据集中。

2. **提出的架构具备高质量定性能力**  
   - 成功生成 **语义丰富的物品画像**
   - 输出 **个性化、连贯且合理的自然语言解释**
   - 实现 **基于 VARK 和上下文的动态展示适配**

3. **认知建模具有潜力但需更强信号支撑**  
   - 当前 VARK 偏好为随机模拟，若能通过实际行为或轻量问卷获取真实偏好，可能显著提升效果

4. **系统更适合高价值个性化场景**  
   如教育、医疗、职业发展等领域，其中认知适配直接影响学习成效或决策质量，而非单纯点击率。

### ⚠️ 方法的局限性
| 局限 | 具体说明 |
|------|----------|
| **候选召回率过低** | 知识图谱构建依赖有限元数据，导致相关物品未能进入候选池 |
| **计算开销大** | 多次调用 LLM、图查询、向量检索影响实时性，不利于大规模部署 |
| **VARK 评估负担重** | 16题问卷增加用户摩擦，需探索隐式推断机制 |
| **评估基准不匹配** | MovieLens 缺乏真实的 VARK 或认知状态标签，无法全面衡量心理适配效果 |
| **流行度偏差掩盖个性化收益** | 现有指标偏向预测已知偏好，难以体现“发现”与“引导”的长期价值 |

### 🔮 未来工作方向
| 方向 | 具体建议 |
|------|----------|
| **改进候选生成** | 引入多视图检索（dense + sparse + graph walk）、主动学习选择最具信息量的初始推荐 |
| **高级认知建模** | 结合情感检测、分心识别、情绪波动建模，进一步细化心理状态估计 |
| **隐式 VARK 推断** | 通过早期交互模式（阅读时长、视频播放、互动频率）自动推测学习风格 |
| **强化惊喜性（Serendipity）** | 完整实施 SerenEva 框架，平衡利用与探索，促进多样化发现 |
| **多模态知识图谱扩展** | 加入图像特征（CLIP）、音频嵌入、文本细粒度表示，提升语义覆盖 |
| **隐私保护机制** | 支持联邦学习、差分隐私、本地化认知建模，适应敏感场景 |
| **领域专项验证** | 在教育、健康、专业培训等场景开展用户研究，测量学习成果、满意度、留存率等真实指标 |
| **综合评估体系** | 设计涵盖 relevance、diversity、serendipity、explainability、trust 的多维评测框架 |

---

## 总结

尽管在 MovieLens-1M 上的定量表现不及简单 **Popularity** 基线，但该论文提出了一个**面向未来的认知智能型推荐系统范式**。其核心价值不在于短期预测精度，而在于：
- 将 **心理学原理（VARK）** 与 **前沿 AI 技术（LLM + KG）** 深度融合
- 实现从“推荐什么”到“如何推荐”的转变 —— 即不仅关注内容相关性，也重视**呈现方式的认知适配性**
- 提供 **可解释、可信赖、可持续进化** 的推荐体验

> 💡 **一句话总结**：  
> 本文虽未在传统指标上取胜，却为下一代 **cognitively-aware recommender systems** 奠定了重要基础 —— 在冷启动困境中，不仅要“猜你喜欢”，更要“懂你怎么学”。

</details>

---

### 4. [$V_1$: Unifying Generation and Self-Verification for Parallel Reasoners](https://arxiv.org/abs/2603.04304)

**Authors**: Harman Singh, Xiuyu Li, Kusha Sareen, Monishwaran Maheswaran, Sijun Tan, Xiaoxia Wu, Junxiong Wang, Alpay Ariyak, Qingyang Wu, Samir Khaki, Rishabh Tiwari, Long Lian, Yucheng Lu, Boyi Li, Alane Suhr, Ben Athiwaratkun, Kurt Keutzer  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.04304v1  

#### Abstract
Test-time scaling for complex reasoning tasks shows that leveraging inference-time compute, by methods such as independently sampling and aggregating multiple solutions, results in significantly better task outcomes. However, a critical bottleneck is verification: sampling is only effective if corre...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*$V_1$: Unifying Generation and Self-Verification for Parallel Reasoners*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在复杂推理任务中，**并行推理**（parallel reasoning）通过生成多个独立的推理路径（chains of thought）来探索多样化的解决方案，是提升模型性能的有效手段。然而，其效果高度依赖于能否从候选解中准确识别出正确答案。现有方法通常采用**逐点验证**（pointwise verification），即为每个解独立打分，但这种方法存在以下问题：

- **校准崩溃**（Calibration Collapse）：缺乏比较参照，导致绝对分数不可靠、跨上下文不一致。
- **自我偏见**（Self-bias）：模型倾向于高估自己生成的样本，即使它们是错误的。
- **多样性坍缩**（Diversity Collapse）：基于聚合的方法（如RSA）在迭代过程中可能丢弃正确的“异常”解。

### 提出的新方法与新思路
本文提出 **V1** 框架，统一了生成（generation）与自验证（self-verification）过程，核心创新如下：

#### ✅ **V1-Infer**：不确定性引导的成对验证算法
- 将传统的逐点评分改为**成对比较**（pairwise comparison），利用模型更强的相对判断能力。
- 设计了一种**瑞士制锦标赛精炼策略**（Swiss-system tournament refinement），动态分配验证计算资源：
  - **阶段一：拓扑覆盖**（Topology Coverage）——确保所有解至少参与一定次数的比较，防止“孤儿”节点。
  - **阶段二：瑞士精炼**（Swiss Refinement）——优先比较当前得分相近的解对，最大化信息增益。
- 引入**加权聚合机制**，根据评分差异大小（置信度）加权统计胜率，得到最终排名。

#### ✅ **V1-PairRL**：联合训练生成器与成对自验证器的强化学习框架
- 在RL训练阶段，**同时优化生成能力和成对自验证能力**，形成一个共进化的单一模型。
- 使用在线、协同演化的训练目标，确保验证器始终在当前生成器分布的数据上训练，避免推理时的分布偏移。
- 设计了防奖励黑客攻击（reward hacking）的机制：
  - **稀疏奖励阈值**：仅当预测分数接近真实标签时才给予正向奖励，防止“安全赌注”（safe bet）行为。
  - **严格配对策略**：只在能构成“正确-错误”或“正确-正确”对时触发验证训练，防止生成器退化为空输出。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **验证方式** | 成对比较比逐点评分更鲁棒、更可校准。 |
| **效率** | 动态配对策略显著减少无效验证调用，比全连接或随机配对更高效。 |
| **多样性保持** | 不会像聚合方法那样丢失正确解，保留了并行采样的多样性价值。 |
| **训练一致性** | 验证器与生成器同步演化，保证推理时验证能力匹配生成风格。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **代码生成**（Code Generation）
  - `LiveCodeBench-v5`, `LiveCodeBench-v6`：竞赛级编程题，无污染。
  - `CodeContests`：来自Codeforces等平台的真实编程挑战。
  - `SWE-bench Lite`：GitHub上的真实软件工程问题（补丁修复）。
- **数学推理**（Math Reasoning）
  - `AIME'25`, `HMMT'25`：高中数学竞赛题目。

### 实验设置与评估指标
- **生成设置**：
  - 温度 $T=0.6$（代码）、$T=1.0$（数学），top-p=0.95。
  - 每个问题生成 $N=8, 16, 32$ 个候选解。
- **验证预算**：
  - V1-Infer 支持灵活控制验证调用数，设为 $1\times$, $2\times$, $3\times N$。
- **评估指标**：
  - **Pass@1**：选出的最佳解是否通过所有测试用例。
  - **Pass@N**：原始 $N$ 个候选解中是否存在正确解（理论上限）。
  - 对比总计算量（生成 + 验证调用总数）下的性能曲线。

### 基线方法对比
| 类型 | 基线方法 |
|------|--------|
| **逐点验证** | 使用相同1-10评分系统的独立打分，选最高分者。 |
| **聚合方法** | `Recursive Self-Aggregation (RSA)`：通过多轮自我聚合提炼解。 |
| **RL训练基线** | `Standard RL`：仅优化生成目标。 |
| | `V1-PointRL`：联合训练生成器与逐点验证器。 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📌 V1-Infer 性能提升（vs. Pointwise Verification）
| 数据集 | 模型 | Pass@1 提升 |
|-------|------|-------------|
| CodeContests | GPT-OSS-20B | **+7.3%** |
| LiveCodeBench-v5 | GPT-OSS-20B | **+8.6%** |
| LiveCodeBench-v6 | GPT-OSS-20B | **+10.0%** |
| HMMT | GPT-OSS-20B | **+10.0%** |

> ✅ 在同等预算下，V1-Infer 显著优于逐点验证，并且随着验证预算增加持续提升性能。

#### 📌 与 RSA 的对比
| 方法 | LiveCodeBench-v6 (GPT-OSS-20B, N=16) | 所需验证步数 |
|------|------------------------------------|--------------|
| RSA 最大 Pass@1 | ~72% | >100 步 |
| **V1-Infer** | **76%** | **仅需 48 步** |

> ✅ V1-Infer 以更少的模型调用达到更高准确率，效率远超 RSA。

#### 📌 SWE-bench Lite 结果（真实软件工程）
| 方法 | Resolve Rate |
|------|--------------|
| Vanilla（首个解） | 26.3% |
| Pointwise | 28.3% |
| **Pairwise (V1-Infer)** | **33.3%** |

> ✅ 成对比较能更好地区分“表面修改”与“根本原因修复”。

---

### 消融实验结果

#### 🔍 V1-PairRL 的训练优势
| 指标 | 提升幅度 |
|------|----------|
| 测试时扩展收益（test-time scaling gain） | **+7–9%** 超过标准 RL 和 V1-PointRL |
| 基础 Pass@1（无扩展） | 在 CodeContests 上比标准 RL **+8.7%** |
| 推理时配合 V1-Infer | 即使使用相同验证算法，仍比 RL 基线高 **+3.6%~8.9%** |

> 表明联合训练不仅提升了验证能力，也反哺了生成质量。

#### 🔍 共进化训练的重要性（Co-evolving vs. Non-co-evolving）
| 设置 | LiveCodeBench-v5 | CodeContests |
|------|------------------|------------|
| 非共进化（offline data） | 50.7% | 45.2% |
| **共进化（V1-PairRL）** | **53.9%** | **52.5%** |

> ✅ 在线共进化训练至关重要，离线数据无法捕捉生成分布的动态变化。

#### 🔍 困难问题上的增益更大
在 `LCB-v6` 上，对于原本 Pass@1 只有 40.2% 的难题，V1-Infer ($3\times$ 预算) 达到 **63.9%**，**绝对提升 +23.7%**。

---

## 4. 关键结论和发现

### 主要发现
1. **成对自验证远优于逐点验证**  
   模型在相对判断（A vs. B）上比绝对打分更可靠，解决了校准问题。

2. **动态不确定性引导的配对策略高效且有效**  
   “瑞士精炼”机制将有限的验证资源集中在最模糊的对决上，实现高性价比提升。

3. **生成与验证应协同演进**  
   V1-PairRL 证明，在训练阶段就让模型学会如何自我评判，能显著增强其推理能力和测试时扩展潜力。

4. **成对验证可作为通用增强模块**  
   它不仅能独立使用，还能与聚合方法（如 RSA）结合，提供可靠的 fitness signal，加速收敛。

5. **方法在多种任务上泛化良好**  
   从数学、编程到真实软件工程（SWE-bench），均取得一致提升。

---

### 局限性
- **依赖可验证任务**：目前仅适用于有明确执行反馈的任务（如代码运行、数学答案匹配）。
- **验证成本仍存在**：尽管已优化，但成对比较仍需额外 LLM 调用，对小模型或低延迟场景可能不适用。
- **极端相似错误难以区分**：当所有候选解都犯类似错误时，成对比较可能放大无关紧要的风格差异。

---

### 未来工作方向
- 将 V1 框架扩展至非可验证领域（如创意写作、对话）。
- 探索轻量化验证器或缓存机制，进一步降低推理开销。
- 研究如何将成对偏好信号融入预训练或 SFT 阶段。
- 构建端到端可微的近似版本，替代显式的多次 LLM 调用。

---

> 💡 **一句话总结**：  
> **V1** 通过引入**成对自验证**这一更本质、更鲁棒的判断范式，并设计**动态资源分配算法**与**共进化训练框架**，实现了生成与验证能力的统一跃迁，在多项复杂推理任务上实现了显著且高效的性能突破。

</details>

---

### 5. [Hierarchical Inference and Closure Learning via Adaptive Surrogates for ODEs and PDEs](https://arxiv.org/abs/2603.03922)

**Authors**: Pengyu Zhang, Arnaud Vadeboncoeur, Alex Glyn-Davies, Mark Girolami  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.03922v1  

#### Abstract
Inverse problems are the task of calibrating models to match data. They play a pivotal role in diverse engineering applications by allowing practitioners to align models with reality. In many applications, engineers and scientists do not have a complete picture of i) the detailed properties of a sys...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Hierarchical Inference and Closure Learning via Adaptive Surrogates for ODEs and PDEs*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**部分已知物理系统的逆问题**（inverse problems），即在常微分方程（ODEs）和偏微分方程（PDEs）建模中，系统参数未知且控制方程中存在未建模的非线性项（如摩擦、阻尼、湍流模型等）。传统方法通常假设方程形式完全已知，而本文关注的是“**部分已知物理 + 未知闭合项（closure）**”这一更贴近现实的场景。

此外，由于数值求解器重复调用导致计算成本高昂，以及单个系统数据稀疏时参数估计不稳定，这些问题也亟待解决。

---

### 提出的新方法与新思路

作者提出了一种**双层优化框架**（bilevel optimization framework），结合以下三个关键技术：

#### ✅ **C1. 联合概率参数推断与确定性闭合学习**  
- **Hierarchical Bayesian Inference**：对多个相关物理系统进行联合建模，引入群体级超参数（hyperparameters）来共享统计信息，提升小样本下的推断稳定性。
- **Deterministic Closure Learning**：将未知的非线性闭合函数 $ f(u) $ 表示为一个可训练的神经网络（MLP），通过最大边缘似然（maximum marginal likelihood）进行学习，避免高维函数空间采样的困难。

#### ✅ **C2. 迭代式层次化推理与闭合更新机制**  
采用交替优化策略：
1. 使用 **ensemble Metropolis-Adjusted Langevin Algorithm (MALA)** 对系统参数和超参数进行后验采样；
2. 利用采样结果近似梯度，更新闭合模型 $ f^\theta $；
3. 在收敛后固定闭合模型，进行最终的后验采样以获得可靠的不确定性量化（UQ）。

#### ✅ **C3. 基于代理模型加速的贝叶斯反演**  
引入**可微分前向代理模型**（surrogate model）替代昂贵的数值求解器（如FEM、Runge-Kutta），构建双层优化结构：
- **上层目标**（Upper-level）：最大化观测数据的边缘似然，用于学习闭合模型 $ f^\theta $；
- **下层目标**（Lower-level）：训练代理模型 $ \mathcal{F}^\beta $ 来逼近真实求解器输出。

该框架实现了在线联合训练，确保代理模型在当前探索的参数区域保持高精度。

#### ✅ **C4. 多类ODE/PDE问题验证**
在三类典型问题上验证方法有效性：
- 非线性质量-弹簧-阻尼系统（nonlinear mass-spring-damper, ODE）
- 非线性达西流（nonlinear Darcy flow, PDE）
- 广义伯格斯方程（generalized Burgers’ equation, PDE）

---

### 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **建模能力** | 支持同时估计系统特异性参数和共享的非线性闭合项，适用于工程中常见的“已知主干 + 缺失细节”场景 |
| **不确定性量化** | 采用 hierarchical Bayes 提供稳健的参数后验分布，优于点估计方法 |
| **计算效率** | 引入 surrogate model 显著降低 MALA 中反复调用求解器的成本，尤其适合时间依赖或迭代型PDE求解 |
| **泛化性** | 可灵活集成多种 surrogate 架构（FNO、PINNs等），适应不同维度和复杂度的问题 |

---

## 2. 核心实验方法和设置

### 使用的数据集（模拟生成）
所有实验均基于**合成数据集**，由真实物理方程加噪声生成，共三组：

| 实验 | 物理系统 | 参数数量 $ K $ | 观测方式 |
|------|--------|------------------|---------|
| Exp 1 | Nonlinear Mass-Damper (ODE) | $ K = 5 \sim 100 $ | 稀疏时间点位移观测，含噪声 |
| Exp 2 | Nonlinear 2D Darcy Flow (PDE) | $ K = 10, 20, 30 $ | 固定空间点观测，含噪声 |
| Exp 3 | Generalized Burgers’ Equation (PDE) | $ K = 20 $ | 随机时空点观测，含噪声 |

每个系统具有不同的未知参数，但共享相同的闭合函数 $ f(\cdot) $。

---

### 实验设置与评估指标

#### 🧪 主要对比模型
| 模型 | 描述 |
|------|------|
| **Solver (Baseline)** | 直接使用数值求解器（Leapfrog/FEM/RK4）进行推理，无代理模型 |
| **FNO (Supervised)** | 使用监督损失训练 FNO，输入为参数+坐标，输出为全场解 |
| **FNO (Physics-based)** | 使用PDE残差作为损失函数（弱形式） |
| **PINNs** | 输入参数+坐标，通过自动微分强制满足PDE |
| **Non-Hierarchical** | 移除层级结构，独立推断各系统参数 |

#### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **Parameter Inference Mean MSE** | 所有系统后验均值与真值之间的平均均方误差 |
| **Coverage (%)** | 真实参数落在后验±2标准差范围内的比例（衡量UQ可靠性） |
| **Closure MSE** | 学习到的闭合函数 $ f^\theta $ 与真实 $ f $ 在关键区间上的MSE |
| **Surrogate MSE** | 代理模型预测解与真实数值解之间的MSE |
| **Computational Time per Epoch** | 单轮训练耗时（秒） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（代表性 $ K=20 $ 设置）

| 模型 | Param MSE ↓ | Coverage ↑ | Closure MSE ↓ | Surrogate MSE ↓ | Time (s) ↓ |
|------|-------------|------------|---------------|------------------|------------|
| **Solver** | $ 2.70\times10^{-2} $ | 96.7% | 0.50 | — | 0.201 |
| **FNO (Supervised)** | $ 3.13\times10^{-2} $ | 93.3% | 0.88 | $ 8.44\times10^{-4} $ | 0.712 |
| **FNO (Physics-based)** | $ 6.77\times10^{-2} $ | 86.7% | 1.38 | $ 4.58\times10^{-3} $ | 0.124 |
| **PINNs** | $ 6.37\times10^{-2} $ | 90.0% | 1.27 | $ 3.97\times10^{-3} $ | **0.040** |
| **PINNs (Non-Hierarchical)** | $ 6.02\times10^{-2} $ | 86.7% | 1.44 | $ 5.41\times10^{-3} $ | 0.045 |

> 注：Exp 2 和 Exp 3 结果趋势一致。

---

### 与基线方法的对比结果

#### 🔹 参数推断
- **Solver 和 Supervised FNO 最准确**，得益于高质量参考轨迹；
- **PINNs 性能接近**，尤其在 Exp 2 中表现良好；
- **Physics-based FNO 表现较差**，尤其在小 $ K $ 下（$ K=5,10 $）无法稳定收敛。

#### 🔹 闭合函数学习
- **Supervised FNO 和 Solver 准确捕捉非线性结构**（如立方项）；
- **PINNs 和 Physics-based FNO 存在偏差**，尤其在低密度数据区域外推能力弱；
- 所有方法在训练数据密集区（如速度集中在 [-2,2]）表现更好，体现数据驱动特性。

#### 🔹 代理模型精度
- **Supervised FNO 最优**，其 Surrogate MSE 比其他低一个数量级；
- **PINNs 次之**，但仍优于 Physics-based FNO；
- **Physics-based 方法在边界处误差大**，因软约束难以精确满足BC。

#### 🔹 计算效率
- **PINNs 最快**，且运行时间几乎不随 $ K $ 增长（高度可扩展）；
- **Supervised FNO 最慢**，因其需在线调用数值求解器生成标签；
- **传统求解器在 $ K>30 $ 时内存溢出**，不可行。

---

### 消融实验结果

#### ✅ 层次化 vs 非层次化 Bayes（见 Table 2 & 5）
| 指标 | Hierarchical | Non-Hierarchical |
|------|--------------|------------------|
| Param MSE ($ K=5 $) | $ 4.19\times10^{-2} $ | $ 4.79\times10^{-1} $ ❌ |
| Coverage ($ K=5 $) | 86.7% | 26.7% ❌ |
| Closure MSE ($ K=5 $) | 1.18 | 36.89 ❌ |

➡️ **结论**：Hierarchical Bayes 显著提升小样本下的估计准确性与不确定性校准能力，并加快收敛速度（见 Figure 9）。

#### ✅ 不同 surrogate 架构比较
- **Supervised FNO**：精度最高，但代价是计算开销；
- **PINNs**：效率最优，适合大规模系统族分析；
- **Physics-based FNO**：稳定性差，尤其在数据稀疏时易发散。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **层级贝叶斯框架显著增强参数推断鲁棒性**，特别是在小样本情况下，通过群体统计实现“信息共享”，有效缓解过拟合与不确定性膨胀问题。

2. ✅ **双层优化 + 自适应代理模型** 是解决高成本逆问题的有效路径。代理模型不仅加速推理，还能在关键参数区域动态精炼，提高整体效率。

3. ✅ **监督式 FNO 在精度上领先**，尤其在闭合学习和代理建模方面；而 **PINNs 在效率上占优**，更适合大规模、实时应用场景。

4. ✅ **纯物理引导的 surrogate（如 physics-based FNO）在复杂PDE中不够稳定**，尤其是在观测稀疏、非线性强的情况下，容易陷入局部极小或欠拟合。

5. ✅ **所提框架成功实现了从稀疏、含噪数据中联合恢复系统参数与未知动力学**，展示了在真实工程系统建模中的潜力（如材料识别、流体建模等）。

---

### 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **依赖合成数据训练与验证** | 尚未在真实实验数据上测试，实际噪声结构可能更复杂 |
| **监督式 FNO 需要调用真实求解器** | 在线生成标签带来额外开销，限制其在完全黑箱系统中的应用 |
| **闭合函数表示受限于MLP容量** | 对极端非光滑或高频行为建模可能存在瓶颈 |
| **目前仅支持静态或稳态参数** | 无法处理时变参数或状态依赖参数 |

---

### 未来工作方向

1. **扩展至动态系统联合状态-参数估计**：结合 Kalman filtering 或 sequential MCMC 实现在线 inference。
2. **引入更强先验结构**：如将物理约束嵌入 closure network 设计中（physics-encoded MLP）。
3. **探索免标签代理训练**：发展完全无需真实求解器参与的自洽训练机制（如对抗训练、能量匹配）。
4. **应用于真实世界系统**：如结构健康监测、气候模型校正、生物力学建模等。
5. **融合多保真度数据**：整合低精度仿真与高精度实验数据，构建 multi-fidelity hierarchical inference pipeline。

--- 

> 💡 **总体评价**：本文提出了一套完整、灵活且高效的框架，将 hierarchical Bayes、neural closure learning 与 surrogate modeling 有机结合，在理论严谨性和工程实用性之间取得了良好平衡，为复杂物理系统的数据驱动建模提供了新的范式。

</details>

---

### 6. [Specification-Driven Generation and Evaluation of Discrete-Event World Models via the DEVS Formalism](https://arxiv.org/abs/2603.03784)

**Authors**: Zheyu Chen, Zhuohuan Li, Chuanhao Li  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.03784v1  

#### Abstract
World models are essential for planning and evaluation in agentic systems, yet existing approaches lie at two extremes: hand-engineered simulators that offer consistency and reproducibility but are costly to adapt, and implicit neural models that are flexible but difficult to constrain, verify, and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Specification-Driven Generation and Evaluation of Discrete-Event World Models via the DEVS Formalism**

---

## **1. 主要贡献和创新点**

### **解决的问题**
现有 **World Models** 在 **agentic systems** 中的应用面临两极分化：
- **显式模拟器（hand-engineered simulators）**：一致性高、可复现，但适应成本高，难以在线调整。
- **隐式神经模型（implicit neural models）**：灵活、可通过 prompting 在线适应，但长期 rollout 下不可靠，难以约束、验证和调试。

本文旨在为 **离散事件主导的环境**（如排队系统、服务流程、多智能体协调等）构建一种兼具 **可靠性与灵活性** 的中间路线：**可执行、可验证、可按需生成的离散事件世界模型（Discrete-Event World Models）**。

---

### **提出的新方法与思路**

#### **核心框架：基于 DEVS 形式化语言的规范驱动生成与评估**
1. **DEVS 形式化建模**  
   采用 **DEVS（Discrete Event System Specification）** 作为世界模型的形式化表示，将系统分解为 **Atomic Models** 和 **Coupled Models**，明确状态、事件、时间推进和交互语义。

2. **分阶段 LLM 生成流水线（Staged LLM-based Generation Pipeline）**
   - **第一阶段：结构合成（Structural Synthesis）**  
     由 LLM 推断组件层次结构、端口接口和交互图，形成 **PlanTree**，作为后续生成的“架构蓝图”。
   - **第二阶段：行为合成（Behavioral Synthesis）**  
     并行生成各 Atomic Model 的状态转移逻辑和时序行为，最后自底向上组装成完整耦合模型。

3. **基于事件轨迹的规范驱动评估（Trace-Based, Specification-Driven Evaluation）**
   - 生成的模拟器输出标准化的 **JSONL 格式事件轨迹（event trace）**。
   - 验证轨迹是否满足从自然语言规范中提取的 **时序、因果和语义约束**（如先序关系、响应约束、定时边界等）。
   - 支持无唯一真值下的可复现验证，并提供 **局部诊断（localized diagnostics）** 定位违规实体和变量。

---

### **相比现有方法的优势**

| 维度 | 显式模拟器 | 隐式神经模型 | 本文方法（DEVS-Gen） |
|------|-----------|-------------|------------------|
| **一致性** | ✅ 高 | ❌ 低（误差累积） | ✅ 高（结构化状态演化） |
| **可验证性** | ✅ 可调试 | ❌ 黑箱 | ✅ 轨迹可验证 |
| **灵活性** | ❌ 难修改 | ✅ 可 prompting | ✅ 可按需生成 |
| **在线适应性** | ❌ 困难 | ✅ 可能 | ✅ 支持 on-demand synthesis |
| **生成效率** | —— | —— | ✅ 并行化、模块化解耦 |

---

## **2. 核心实验方法和设置**

### **数据集**
构建了一个包含 **7 个真实场景** 的基准数据集，覆盖多个领域和动态特性：

| 场景 | 领域 | 关键动态 | 规模 |
|------|------|--------|-----|
| IOBS | 银行业务 | 流水线 + 概率路由 | M |
| OTrain | 交通 | 调度驱动延迟 | M |
| SEIRD | 生物数学 | 连续动态（ODE） | S |
| FileTransfer | 网络 | 双循环 FSM | L |
| ABP | 网络协议 | Stop-and-Wait | S |
| StratAirlift | 物流 | 主动 reneging | L |
| Barbershop | 服务 | 阻塞与信号握手 | S |

所有场景均通过 **逆向工程高质量开源 DEVS 模型** 构建，确保逻辑真实性，并提取自然语言规范和验证规则。

---

### **实验设置与评估指标**

#### **评估维度**
1. **操作成功性（Operational Success Score, OSS）**
   - 编译运行成功
   - 输出符合 I/O 合同（JSONL 格式）
   - $ \text{OSS} = \frac{1}{m} \sum_{i=1}^m v_i $，其中 $ v_i $ 表示第 $ i $ 个测试用例是否有效。

2. **行为一致性（Behavioral Conformance Score, BCS）**
   - 轨迹是否满足规范中的逻辑与时序约束。
   - $ \text{BCS} = \frac{1}{m} \sum_{i=1}^m \left(1 - \frac{\text{违反规则数}}{\text{总规则数}}\right) $

#### **基线方法**
- **OpenHands** 和 **SWE-Agent**：主流开源代码代理。
- 每种方法分为两个变体：
  - **Standard（Iterative）**：允许执行反馈、自我修复。
  - **Lite（Non-Iterative）**：限制交互轮次，无执行工具，用于公平比较单次生成能力。

#### **LLM 模型**
- **Large Models**：GPT-5.2, Gemini-3-Pro, GLM-4.7, Claude-3.5-Sonnet
- **Small Models**：Llama-4-17B, GLM-4.7-Flash, Gemini-3-Flash, Qwen3-Coder-30B

#### **效率指标**
- **耗时（Wall-clock time）**
- **Token 消耗量（log₁₀ scale）**

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 2 & 3）**

#### **有效性（Effectiveness）**
| 方法 | OSS（平均） | BCS（平均） |
|------|------------|------------|
| **DEVS-Gen（Large）** | **0.84** | **5.35** |
| OpenHands（Large） | 0.91 | 5.67 |
| SWE-Agent（Large） | 0.58 | 6.17 |
| OpenHands-Lite | 0.75 | 5.51 |
| SWE-Agent-Lite | 0.31 | 5.77 |

- **DEVS-Gen 在不使用执行反馈的情况下，表现接近甚至优于部分迭代式代理**。
- 在 **小模型上优势更明显**：DEVS-Gen 的 OSS 是 SWE-Agent-Lite 的 **10 倍以上**。

#### **效率（Efficiency）**
| 方法 | 平均耗时（s） | Token 消耗（log₁₀） |
|------|--------------|--------------------|
| **DEVS-Gen（Large）** | **753.3** | **5.35** |
| OpenHands | 701.6 | 5.67 |
| SWE-Agent | 4020.1 | 6.17 |
| **DEVS-Gen（Small）** | **1575.1** | **5.19** |
| OpenHands（Small） | 5364.9 | 6.25 |
| SWE-Agent（Small） | 5028.4 | 6.13 |

- **DEVS-Gen 减少约一个数量级的 token 消耗**（~6–10×），尤其在小模型上避免“doom loops”导致的超时。
- 即使在大模型上，也保持与优化代理相当的时间效率。

---

### **消融实验（Ablation Study）**

#### **并行化加速效果（Figure 3）**
- **规划阶段（Planning）**：加速有限（因层级浅），串行 vs 并行提升不显著。
- **生成阶段（Generation）**：实现 **~4.7× 加速**，验证了原子模型并行生成的有效性。

#### **模块化解耦的价值**
- 将复杂系统分解为独立组件后，每个组件可在局部上下文中生成，降低 LLM 上下文负担。
- 即使某个组件生成失败，也不影响其他组件，支持 **细粒度修复**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **DEVS 形式化提供了“正确即构造”（correct-by-construction）的生成路径**  
   通过结构化接口和契约，显著提升了 LLM 生成代码的 **首次通过率**，减少了对试错循环的依赖。

2. **模块化 + 并行生成是高效合成大规模世界模型的关键**  
   时间复杂度从 $ O(N) $ 降至 $ O(\log N) $，支持可扩展的在线生成。

3. **基于轨迹的规范驱动评估是可行且必要的替代方案**  
   在无唯一真值实现的场景下，通过 **可观测行为** 进行验证，实现了黑盒可测试性。

4. **小模型也能胜任复杂建模任务**  
   得益于任务解耦，即使是弱 LLM 也能稳定生成可运行组件，展现出更强的鲁棒性。

---

### **局限性**
1. **依赖 LLM 对 DEVS 语义的理解**  
   当前方法假设 LLM 能正确理解 `deltint`, `lambdaf`, `hold_in` 等 DEVS 核心概念，否则会出现 **状态-输出不同步** 等语义错误。

2. **仅适用于离散事件主导的系统**  
   不适合连续动态强主导的环境（如物理仿真、视觉预测）。

3. **验证规则仍需人工编写**  
   虽然轨迹可自动收集，但约束规则目前是手动提取，尚未完全自动化。

---

### **未来工作方向**
1. **混合建模（Hybrid DEVS Systems）**  
   将 LLM 嵌入 DEVS 组件内部，作为 **事件生成器或决策模块**，构建 LLM-driven 多智能体系统。

2. **自动化验证规则提取**  
   利用 LLM 从自然语言规范中自动推导出可检查的时序逻辑公式（如 LTL）。

3. **支持增量更新与在线修正**  
   在运行过程中根据观测反馈动态调整模型结构或参数。

4. **扩展至更复杂的交互模式**  
   如支持不确定性传播、学习型组件与符号组件的协同推理。

---

> **总结一句话**：  
> 本文提出了一个 **规范驱动、基于 DEVS 的离散事件世界模型生成与评估框架**，通过 **结构化分解 + 并行生成 + 轨迹验证**，实现了 **高一致性、可验证、可扩展** 的世界模型构建，在效率、稳定性与可维护性上显著优于现有方法。

</details>

---

### 7. [A framework to reason about consistency and atomicity guarantees in a sparsely-connected, partially-replicated peer-to-peer system](https://arxiv.org/abs/2603.03899)

**Authors**: Sreeja S. Nair, Nicholas E. Marino, Nick Pascucci, Russell Brown, Arthur P. R. Silva, Tim Cummings, Connor M. Power  
**Category**: cs.DC  
**Published**: 2026-03-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.03899v1  

#### Abstract
For an offline-first collaborative application to operate in true peer-to-peer fashion, its collaborative features must function even in environments where internet connectivity is limited or unavailable. Each peer may only be interested in a subset of the application data relevant to its workload, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A framework to reason about consistency and atomicity guarantees in a sparsely-connected, partially-replicated peer-to-peer system

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

本文针对**离线优先（offline-first）协作型应用**在**稀疏连接、部分复制的对等网络（peer-to-peer, P2P）系统**中的一致性和原子性保障难题展开研究。这类系统面临以下挑战：

- 节点仅对全局数据的一个子集感兴趣（partial replication），即具有不同的 **interest set**（订阅与权限的交集）；
- 网络连接不稳定或稀疏，无法依赖中心化协调服务；
- 多个事务可能跨多个数据项更新，且需要满足 **atomicity（原子性）** 和 **causal consistency（因果一致性）** 等强一致性语义；
- 在传统模型如 TCC+（Transactional Causal+ Consistency）下，当节点只接收事务的部分操作时，难以保证这些性质。

> ❗现有模型（如 TCC+）假设所有节点持有完整数据副本，不适用于部分复制场景；而现有 CRDT 实现仍常依赖中心化协调，限制了真正的去中心化协作能力。

---

### 🆕 提出的新方法或新思路

作者提出了两个新的形式化模型，用于在部分复制环境下推理一致性和原子性：

#### **1. INTERSECTIONATOMICITY**

- 定义：一个事务 $ T $ 满足 INTERSECTIONATOMICITY，当且仅当对于任意节点 $ N $，若其兴趣集合为 $ S_N $，且该事务中有任一操作作用于 $ S_N $ 并被观察到，则该事务中所有作用于 $ S_N $ 的其他操作也必须被观察到。
- 形式化表达：
  $$
  \forall o_1,o_2 \in T.\ o_1 \downarrow S_N \land o_1 \xrightarrow{vis} N \Rightarrow o_2 \xrightarrow{vis} N \lor o_2 \circ S_N
  $$
- 即：只要节点看到了事务中的某个与其兴趣相关的更新，就必须看到该事务中所有与其兴趣相关的更新。

#### **2. INTERSECTIONCC**

- 定义：一个操作满足 INTERSECTIONCC，当且仅当如果它被某节点观察到，则其整个因果历史中属于该节点兴趣集的部分也都应被观察到。
- 形式化表达：
  $$
  \forall o_1,o_2.\ o_1 \xrightarrow{hb} o_2 \land o_2 \xrightarrow{vis} N \Rightarrow o_1 \xrightarrow{vis} N \lor o_1 \circ S_N
  $$
- 保证因果链在局部视图中不中断。

这两个模型共同构成了在**局部数据视图下维持全局 TCC+ 保证的基础框架**。

---

### 🔍 相比现有方法的优势

| 方面 | 本文优势 |
|------|--------|
| **适用性更强** | 支持稀疏连接、移动性强、带宽受限的边缘环境（如蓝牙/Wi-Fi Direct） |
| **无需全量复制** | 允许每个节点仅同步其 `interest set` 内的数据，节省存储与通信开销 |
| **兼容 CRDTs** | 可结合 △-CRDTs 等现代技术实现高效同步，避免传统 state-based 或 op-based CRDT 的缺陷 |
| **支持事务性更新** | 能处理跨多个对象的多操作事务，并在部分复制下保持原子性和因果序 |
| **指导设计实践** | 提供明确的设计准则（如拓扑结构选择、interest set 配置建议），帮助开发者构建可靠系统 |

> 💡 创新本质：将传统的全局一致性概念“投影”到局部兴趣集上，提出可验证的局部条件，从而确保系统整体仍能满足 TCC+。

---

## 2. 核心实验方法和设置

⚠️ **注意：本论文为理论建模与形式化分析类工作，未包含传统意义上的“实验”（如性能测试、真实数据集跑分）。**

### 🧩 方法论类型

- 属于**系统抽象建模 + 形式化验证 + 场景推演分析**的研究范式。
- 使用执行模型（execution model）$ A = (H, vis, ar) $ 来描述节点的历史、可见性关系与仲裁顺序。
- 基于已有理论基础（Viotti & Vukolic [16], Burckhardt et al. [4]）扩展定义一致性属性。

### 📚 数据集

- **无实际数据集使用**。
- 所有分析基于虚构但典型的工业场景（如飞机维修团队协作）进行逻辑推导。

### ⚙️ 分析设置与评估方式

| 维度 | 描述 |
|------|------|
| **系统模型** | 节点具备 connectivity、local storage、processing power；通过 P2P 连接交换数据；每个节点有明确的 subscription 与 permission，构成 interest set |
| **评估目标** | 是否能在各种 interest set 关系和网络拓扑下维持 atomicity、causal consistency 和 convergence |
| **评估指标（隐式）** | <ul><li>是否满足 INTERSECTIONATOMICITY</li><li>是否满足 INTERSECTIONCC</li><li>是否最终达成 TCC+</li></ul> |
| **对比方式** | 与标准 TCC+、Causal+ Consistency、Strong Convergence 等经典模型进行逻辑对比 |

### 🔀 基线方法对比（逻辑层面）

| 基线模型 | 局限性 | 本文改进 |
|--------|-------|---------|
| **TCC+ [Akkoorath et al.]** | 假设数据中心内全复制架构，不适合边缘设备 | 支持部分复制，适配稀疏连接 |
| **Hierarchical Edge Models [Toumlilt et al.]** | 强制森林状拓扑（forest-like），灵活性差 | 提供多种拓扑容忍方案 |
| **Standard CRDTs** | 不支持多对象事务级原子性 | 引入 INTERSECTIONATOMICITY 补充事务语义 |
| **Operational Transformation (OT)** | 依赖中心服务器协调冲突 | 完全去中心化，支持离线协作 |

---

## 3. 主要实验结果和性能指标

📌 **再次强调：本文无量化实验或性能基准测试。所有“结果”均为形式化证明与逻辑推演所得结论。**

### ✅ 关键理论结果

#### （1）INTERSECTIONATOMICITY 与 INTERSECTIONCC 成立的条件总结如下表（原文 Table 1）：

| 保证类型 | LSET ⊇ RSET | LSET ⊆ RSET | LSET = RSET | LSET ∩ RSET ≠ ∅ |
|----------|-------------|-------------|--------------|------------------|
| **INTERSECTIONATOMICITY（本地事务）** | ✅ 全部 | ✅ 交集 | ✅ 全部 | ✅ 交集 |
| **INTERSECTIONATOMICITY（远程事务）** | ✅ 全部 | ❌ 仅交集 | ✅ 全部 | ✅ 交集 |
| **INTERSECTIONCC（单对象）** | ✅ | ✅ | ✅ | ✅ |
| **INTERSECTIONCC（多对象）** | ✅ | ❌ 仅交集 | ✅ | ✅ 交集 |
| **Convergence** | ✅ | ✅ | ✅ | ✅ |

> 注：LSET = Local peer’s Interest Set, RSET = Remote peer’s Interest Set

#### （2）关键发现：

- 当发送方的兴趣集是接收方的超集（LSET ⊇ RSET）时，所有保证均可维持；
- 否则，只能保证**交集部分**的数据满足 INTERSECTIONATOMICITY 和 INTERSECTIONCC；
- 若使用 CRDTs，则 **convergence 自动成立**；
- 若所有 P2P 链接均满足上述局部保证，则整个系统可达 **全局 TCC+**。

---

### 🔍 消融实验（Ablation Study）

- **无传统消融实验**。
- 但文中通过多个反例分析展示了违反条件会导致的问题，相当于逻辑上的“消融”论证：

#### 示例：中间节点兴趣集过小导致原子性破坏

- 设节点 N1（兴趣 s1∪s2）、N2（兴趣 s2）、N3（兴趣 s1∪s2）
- 若事务 T 同时修改 s1 和 s2，在 N1→N2 传输时仅保留 s2 部分；
- N2→N3 传递时，N3 只收到部分更新 → **破坏原子性**
- ➜ 解决方案：避免路径中出现兴趣集小于端点的中继节点

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **在部分复制 P2P 系统中，传统 TCC+ 无法直接应用**，因为节点仅能看到事务的一部分。
2. **INTERSECTIONATOMICITY 和 INTERSECTIONCC 是维持全局 TCC+ 的充分条件**，只要每条 P2P 同步链路都遵守这些局部规则。
3. **CRDTs 可天然保证 convergence**，但需额外机制保障事务级原子性与因果序。
4. **网络拓扑设计至关重要**：
   - 推荐采用**层次化结构**（hierarchical/forest-like），底层节点兴趣小，顶层节点兴趣大；
   - 避免“瓶颈节点”（interest set 小于上下游）出现在同步路径中。
5. **可通过配置 interest set 或广播元数据**来增强一致性保障。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **缺乏实证验证** | 所有结论基于形式化推导，尚未在真实系统中部署验证 |
| **依赖 CRDTs 特性** | 要求数据类型本身支持 convergent merge semantics |
| **事务粒度假设** | 默认事务内操作可被拆分传播，未考虑复杂依赖或锁机制 |
| **安全性假设较强** | 假设节点诚实、消息不被篡改，未讨论拜占庭容错 |
| **动态 interest set 变更处理粗略** | 文中将 interest set 变更视为“旧节点退出、新节点加入”，未深入探讨迁移过程 |

---

### 🔮 未来工作方向

1. **构建原型系统验证模型有效性**  
   > 如基于 Ditto Live SDK 实现 INTERSECTIONATOMICITY 控制逻辑。

2. **支持动态 interest set 更新下的平滑过渡机制**  
   > 当前模型将 interest set 更改视为节点重启，未来可研究增量同步策略。

3. **优化元数据传播策略**  
   > 探索仅复制 metadata（如版本向量、因果依赖图）以降低带宽消耗。

4. **扩展至弱信任环境（如公共网络）**  
   > 引入签名、验证机制防止恶意节点注入虚假更新。

5. **与现有数据库协议集成**  
   > 将该框架应用于 SQLite、Realm、Firebase 等移动端数据库的 P2P 同步模块设计中。

---

## 总结

| 维度 | 结论 |
|------|------|
| **核心价值** | 提出首个专为稀疏连接、部分复制 P2P 系统设计的一致性与原子性推理框架 |
| **关键技术** | INTERSECTIONATOMICITY + INTERSECTIONCC 形式化模型 |
| **适用场景** | 工业维护、野外作业、军事通信、IoT 边缘协作等离线优先环境 |
| **现实意义** | 为开发真正去中心化的协作 App 提供理论支撑与设计指南 |

> 🎯 **一句话总结**：  
> 在无法保证全网连通与全量复制的前提下，只要每个节点在其 `interest set` 上维护好 **INTERSECTIONATOMICITY** 与 **INTERSECTIONCC**，整个系统依然可以达到接近最强的一致性水平——**Transaction Causal+ Consistency (TCC+)**。

</details>

---

### 8. [When Small Variations Become Big Failures: Reliability Challenges in Compute-in-Memory Neural Accelerators](https://arxiv.org/abs/2603.03491)

**Authors**: Yifan Qin, Jiahao Zheng, Zheyu Yan, Wujie Wen, Xiaobo Sharon Hu, Yiyu Shi  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.03491v1  

#### Abstract
Compute-in-memory (CiM) architectures promise significant improvements in energy efficiency and throughput for deep neural network acceleration by alleviating the von Neumann bottleneck. However, their reliance on emerging non-volatile memory devices introduces device-level non-idealities-such as wr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When Small Variations Become Big Failures: Reliability Challenges in Compute-in-Memory Neural Accelerators*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于 **Compute-in-Memory (CiM)** 神经网络加速器在实际部署中面临的**可靠性挑战**，尤其是由非易失性存储器（NVM）设备级非理想性（如写入变异性、电导漂移、随机噪声）引发的小幅权重扰动可能导致推理任务中出现**灾难性失败**的问题。传统研究多关注平均情况下的精度表现（average-case accuracy），而忽视了安全关键场景下更应关注的**尾部风险（tail failures）**。

### 提出的新方法与新思路
论文提出了跨层协同设计（cross-layer co-design）框架，结合硬件与软件手段应对可靠性问题，主要包括两个核心技术：

- **SWIM (Selective Write-Verify Mechanism)**  
  一种硬件层面的选择性写验证机制，在有限预算下仅对最关键权重执行 write-verify 操作，以最小化写入开销的同时显著提升系统最坏情况下的鲁棒性。

- **TRICE (Training with Right-Censored Gaussian Noise)**  
  一种软件层面的训练策略，在训练过程中注入**右截断高斯噪声（right-censored Gaussian noise）**，使模型学习适应硬件引入的实际变异分布，从而优化现实中最坏情况的表现（如 KPP 指标）。

### 相比现有方法的优势
| 方面 | 传统方法局限 | 本文优势 |
|------|-------------|---------|
| **评估方式** | 依赖 Monte Carlo 采样估计平均性能，难以捕捉尾部失败 | 引入 worst-case 优化视角和 KPP 指标，更贴近安全关键系统需求 |
| **缓解策略** | 平均性能导向的容错技术（如冗余、映射优化）对尾部失效无效 | 针对“影响最大”的权重进行保护（SWIM）或训练中显式建模真实噪声（TRICE） |
| **效率与开销平衡** | 全量 write-verify 开销巨大，破坏 CiM 能效优势 | SWIM 实现精准干预，大幅降低写操作次数；TRICE 无需额外硬件开销 |

---

## 2. 核心实验方法和设置

### 使用的数据集
论文未明确列出具体数据集名称，但从上下文推断使用的是典型 DNN 推理基准任务，可能包括：
- **CIFAR-10 / ImageNet**（常见于 CiM 加速器研究）
- 安全关键型 workload（如自动驾驶感知、医疗诊断等模拟任务）

模型方面涉及主流 CNN 架构（如 ResNet、VGG 等），用于评估不同网络结构下的可靠性行为。

### 实验设置和评估指标
#### 实验设置：
- 模拟 NVM 设备的写入变异性（write variation）为有界但非独立同分布的噪声 △W。
- 在 write-verify 约束下设定噪声上限 $th_b$。
- 对比不同 write-verify 范围（全量 vs. 选择性）、不同训练策略（标准训练 vs. TRICE）下的推理稳定性。

#### 主要评估指标：
| 指标 | 描述 |
|------|------|
| **Average Accuracy** | 多次蒙特卡洛仿真后的平均推理准确率 |
| **Worst-case Accuracy** | 最差配置下的准确率（通过优化搜索得到） |
| **K-th Percentile Performance (KPP)** | 如 1st percentile 准确率，衡量现实中最坏但可实现的情况 |
| **Normalized Write Cycles** | 写操作总量相对于全量 write-verify 的比例，反映硬件开耗 |

### 基线方法对比
- **Monte Carlo Sampling**：常规随机采样评估平均性能
- **Full Write-Verify**：所有权重都执行 write-verify，作为高可靠但低效基线
- **Naive Heuristics for Verification**：按权重大小、层级顺序选择验证对象
- **Standard Training without Noise Injection**：无任何硬件感知训练
- **Uncensored Gaussian Noise Training**：传统噪声注入训练方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据
1. **小幅度变化导致极端失败**：
   - 单个设备变异虽小（<5% 权重偏移），但在联合最坏配置下可导致**接近 100% 错误率**。
   - 而 Monte Carlo 方法即使运行 100K 次仍无法捕获此类尾部事件。

2. **SWIM 效果显著**：
   - 在仅验证 **~20% 的关键权重**的情况下，即可将 worst-case accuracy 从 <10% 提升至 >85%，接近 full write-verify 效果。
   - 相比 full write-verify，**normalized write cycles 下降约 60–80%**，保留了 CiM 的能效优势。

3. **TRICE 显著改善 KPP**：
   - 在多种模型和变异强度下，TRICE 将 **1st percentile accuracy (KPP)** 提升达 **15–25个百分点**。
   - 相比 uncensored Gaussian noise 注入，TRICE 更有效提升尾部性能，且不损害平均精度。

4. **敏感度排序优于启发式方法**：
   - 基于 loss sensitivity（泰勒展开近似）的权重排序比 magnitude-based 或 layer-wise selection 在防止尾部失败上**高出 20% 以上的恢复能力**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **小变异 ≠ 小影响**：  
   CiM 中看似微弱的设备级变异可通过非线性放大效应引发**灾难性推理失败**，凸显传统平均性能评估的不足。

2. ✅ **尾部风险必须被专门建模与优化**：  
   安全关键系统不能仅依赖平均表现，需采用 **worst-case 分析** 和 **KPP 类指标** 进行评估。

3. ✅ **跨层协同是解决路径**：  
   单纯硬件加固或算法改进不足以应对复杂可靠性问题，必须结合 **device-aware architecture design** 与 **learning algorithm co-design**。

4. ✅ **选择性干预高效可行**：  
   SWIM 表明，并非所有权重都需要保护，**识别并验证最具影响力的关键权重**可在极低开销下大幅提升系统稳健性。

5. ✅ **训练需匹配硬件噪声特性**：  
   TRICE 验证了使用 **right-censored noise** 比标准高斯噪声更能反映真实硬件扰动，从而实现更有效的鲁棒训练。

### 方法的局限性
- **Worst-case search 成本高**：寻找真正最坏配置是一个 NP-hard 优化问题，目前依赖近似方法（如 sensitivity ranking）。
- **假设噪声模型已知**：TRICE 依赖对 write-verify 后噪声分布的先验知识，若实际偏差较大则效果下降。
- **尚未完全覆盖所有 NVM 非理想性**：当前工作主要针对 write variation，conductance drift 和 read noise 的长期影响有待进一步整合。

### 未来工作方向
- 扩展到动态时序任务（如 RNN、Transformer）中的可靠性分析。
- 结合在线监控与自适应 write-verify 策略，实现 runtime 可靠性调节。
- 探索更多硬件-算法接口（hardware-algorithm interface）设计，支持自动化的可靠性-能效权衡。
- 将方法推广至其他类脑计算架构（如 analog AI chips、spintronic devices）。

--- 

> 📌 **总结一句话**：  
> 本文揭示了 CiM 加速器中“小变异引发大失败”的本质风险，并提出 SWIM + TRICE 的跨层解决方案，在几乎不牺牲效率的前提下显著提升了现实中最坏情况下的推理可靠性，为安全关键 AI 系统的部署提供了坚实基础。

</details>

---

### 9. [SENTINEL: Stagewise Integrity Verification for Pipeline Parallel Decentralized Training](https://arxiv.org/abs/2603.03592)

**Authors**: Hadi Mohaghegh Dolatabadi, Thalaiyasingam Ajanthan, Sameera Ramasinghe, Chamin P Hewa Koneputugodage, Gil Avraham, Yan Zuo, Violetta Shevchenko, Alexander Long  
**Category**: cs.DC  
**Published**: 2026-03-05  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.03592v1  

#### Abstract
Decentralized training introduces critical security risks when executed across untrusted, geographically distributed nodes. While existing Byzantine-tolerant literature addresses data parallel (DP) training through robust aggregation methods, pipeline parallelism (PP) presents fundamentally distinct...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SENTINEL: Stagewise Integrity Verification for Pipeline Parallel Decentralized Training 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**去中心化训练**（decentralized training）中的一个关键安全风险——**流水线并行**（pipeline parallelism, PP）架构下的完整性验证问题。在跨不可信、地理上分散的节点进行训练时，恶意参与者可以通过篡改层间传输的**激活值**（activations）和**梯度**（gradients）来破坏训练过程。

传统的拜占庭容错（Byzantine-tolerant）方法主要针对**数据并行**（data parallelism, DP）场景，通过鲁棒聚合（robust aggregation）来防御参数梯度中毒攻击。然而，在PP架构中，模型被分割到不同阶段，通信的是中间信号而非最终梯度，因此传统方法无法直接应用。

### 提出的新方法和新思路
为解决此问题，论文提出了 **SENTINEL**，一种用于PP训练的轻量级、无计算复制的验证机制。

其核心创新点在于：
*   **动量基础的异常检测**（Momentum-based Anomaly Detection）：SENTINEL不依赖于计算复制，而是利用**指数移动平均**（Exponential Moving Averages, EMAs）作为统计基准点。它在每个阶段之间部署可信的“验证者节点”（verifier nodes），这些节点持续监控前向传播的激活值和反向传播的梯度。
*   **多维度距离度量**（Multi-dimensional Distance Measures）：为了鲁棒地检测各种攻击，SENTINEL采用了一组互补的距离度量标准，包括：
    *   **均值绝对差**（Mean Absolute Difference, L1）
    *   **归一化欧氏距离**（Normalized Euclidean Distance, L2）
    *   **符号翻转率**（Sign Flip Ratio, SFR）
    *   **切片Wasserstein距离**（Sliced Wasserstein Distance, SW）
*   **自适应阈值设定**（Adaptive Thresholding）：使用基于**四分位距**（Inter-Quartile Range, IQR）的Tukey fences方法动态调整检测阈值，使系统能自动适应训练过程中数据分布的自然变化。

### 相比现有方法的优势
*   **高效且轻量**：相比需要将一半算力用于计算复制的冗余方案（如Lu et al., 2024），SENTINEL仅需少量CPU资源的验证者节点，计算开销极小，几乎不影响训练吞吐量。
*   **针对性强**：专门设计用于解决PP架构中特有的层间通信攻击，填补了现有拜占庭容错文献在PP领域的空白。
*   **理论保证**：提供了理论收敛性分析，证明未被检测到的恶意行为对最终模型收敛的影响是有限的，且影响程度与检测阈值成正比。

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个大规模文本语料库上进行：
*   **Common Crawl** (C4)
*   **FineWeb** (FW)
*   **OpenWebText** (OW)

### 实验设置和评估指标
*   **模型**：主要使用0.6B参数的Llama-3模型，并扩展到1.2B、4B参数的Llama-3以及Llama-4-0.4B和DeepSeek-V3-1B等MoE模型。
*   **分布式架构**：在8×16、16×16等数据-流水线并行网格上进行，最多使用176个worker节点。
*   **恶意设置**：每个阶段随机指定25%-37.5%的worker为恶意节点，模拟多种攻击（常数攻击、随机值攻击、缩放攻击、延迟攻击、隐形噪声攻击等）。
*   **评估指标**：
    *   **精确率**（Precision）、**召回率**（Recall）、**F1分数**：衡量检测恶意worker的准确性。
    *   **检测速度**（Detection Speed）：从攻击开始到worker被封禁的平均迭代次数。
    *   **验证损失**（Validation Loss）：衡量模型收敛性和性能。

### 基线方法对比
*   **无验证**（No Verification）：作为基线，展示在没有防护的情况下，恶意攻击如何导致训练发散。
*   **计算复制**（Computation Duplication）：引用Lu et al. (2024)的方法作为现有解决方案的代表，虽然其F1分数可达100%，但会将训练吞吐量减半。

## 3. 主要实验结果和性能指标

### 关键性能数据
*   **高F1分数**：在多种攻击场景下，SENTINEL实现了**>90%的F1分数**。例如，在C4数据集上对0.6B模型的常数攻击、随机值攻击等，F1分数达到100%；对于更隐蔽的“隐形噪声”（Invisible Noise）攻击，F1分数也稳定在85.7%以上。
*   **快速检测**：大多数攻击能在**6-7次迭代内**被检测到并封禁，有效遏制了恶意行为的扩散。
*   **维持模型性能**：使用SENTINEL的模型最终验证损失与无攻击的基线模型（vanilla）非常接近。例如，在4B模型的大规模实验中，有攻击但有SENTINEL保护的模型与无攻击基线的验证损失差异仅为0.04。

### 与基线方法的对比结果
*   **VS 无验证**：在没有SENTINEL的情况下，即使是简单的攻击也会迅速导致验证损失飙升（例如，从3.8升至超过11），训练完全失败。而SENTINEL能有效阻止这种发散。
*   **VS 计算复制**：SENTINEL在检测精度上虽略低于计算复制（后者理论上可达到100%），但其优势在于**训练速度是后者的两倍**，因为它不需要牺牲一半的算力。

### 消融实验结果
*   **EMA的重要性**：消融实验证明，如果用瞬时平均值代替EMA，检测性能急剧下降（F1从100%降至37.5%），凸显了EMA捕捉时间动态模式的关键作用。
*   **距离度量的组合效果**：单独使用任何一种距离度量（如SFR或L2）都无法应对所有攻击类型。只有当所有度量（ALL）组合使用时，才能在混合攻击下取得最优性能（F1 87.8% vs. 单独使用最高为79.0%）。
*   **EMA衰减率的敏感性**：实验表明，SENTINEL对EMA的衰减率（β）不敏感，在β=0.6到0.99的范围内性能保持稳定。

## 4. 关键结论和发现

### 主要发现
1.  **PP中的激活攻击与梯度攻击同等危险**：论文首次系统性地指出，在流水线并行中，对中间激活值的攻击与对梯度的攻击一样具有破坏性，但这一风险在以往研究中被忽视。
2.  **轻量级验证是可行的**：SENTINEL证明了无需计算复制，仅通过轻量级的动量监控即可实现对PP训练的有效保护。
3.  **“弱攻击”可能存活但无害**：一些影响微小的“弱攻击”可能因未超过检测阈值而未被发现，但根据理论分析和实验验证（图1），这类攻击对模型收敛的影响可以忽略不计，这符合“安全即服务”的理念——只阻止真正有害的行为。
4.  **与现有方法正交互补**：SENTINEL保护的是PP轴上的层间通信，而传统的拜占庭容错方法（如Krum, Bulyan）保护的是DP轴上的梯度聚合。两者可以结合使用，共同构建更全面的安全体系。

### 方法的局限性
*   **假设诚实多数**（Honest Majority）：该方法依赖于每个阶段恶意worker的比例小于50%（ys<1/2）。如果恶意节点比例过高，系统安全性将无法保证。
*   **对新型自适应攻击的泛化能力未知**：虽然对文中提出的自适应EMA攻击有效，但可能无法抵御所有未来的高级对抗策略。
*   **未覆盖其他威胁**：该工作专注于防御训练中断攻击（training-interruption attacks），但去中心化训练还面临后门攻击（backdoor attacks）、隐私泄露（privacy inference）等其他威胁。

### 未来工作方向
*   探索更智能的自适应检测机制，减少手动调参。
*   研究使用神经网络等更强大的模型来进行异常检测。
*   将SENTINEL的框架扩展到防御后门攻击和隐私泄露等其他安全威胁。
*   进一步优化在异步和高度动态的去中心化环境中的性能。

</details>

---

### 10. [AI4S-SDS: A Neuro-Symbolic Solvent Design System via Sparse MCTS and Differentiable Physics Alignment](https://arxiv.org/abs/2603.03686)

**Authors**: Jiangyu Chen  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.03686v1  

#### Abstract
Automated design of chemical formulations is a cornerstone of materials science, yet it requires navigating a high-dimensional combinatorial space involving discrete compositional choices and continuous geometric constraints. Existing Large Language Model (LLM) agents face significant challenges in ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对**化学配方自动化设计**中的三大挑战：
- **Context Overflow**：长周期推理过程中，LLM 的上下文窗口受限，导致历史逻辑链断裂；
- **Path Dependence 与 Mode Collapse**：基于语言先验的 LLM 容易陷入局部最优，缺乏全局探索多样性；
- **Discrete-Continuous Gap**：LLM 擅长提出定性配方（拓扑结构），但在连续比例优化（几何参数）上常产生违反物理约束的“数值幻觉”（numerical hallucination）。

这些问题使得现有 LLM 代理在高维组合空间（如溶剂配方搜索）中难以实现有效且可靠的科学发现。

---

### 提出的新方法与新思路
作者提出了 **AI4S-SDS** —— 一个闭环的 **neuro-symbolic** 框架，融合多智能体协作与定制化的 **Monte Carlo Tree Search (MCTS)** 引擎，核心创新如下：

#### （1）**Sparse State Storage + Dynamic Path Reconstruction**
- **机制**：节点仅存储轻量级元组 `(action, value, visit count, Q)`，丢弃冗余对话日志。
- **优势**：解耦推理历史与上下文长度，支持在固定 token 预算下进行任意深度的探索，避免 context overflow。

#### （2）**Global-Local Search Strategy**
- **Global Planning Module**：基于向量数据库中的历史反馈动态生成“全局计划”，作为 MCTS 的根节点，引导搜索方向（如 EHS 优先、创新导向等）。
- **Sibling-Aware Expansion**：在节点扩展时显式考虑兄弟节点的动作摘要，作为负向约束，促进正交探索，缓解 mode collapse。

#### （3）**Differentiable Physics Engine + Hybrid Normalized Loss**
- 将符号推理与物理可行性桥接：
  - 使用 **Hansen Solubility Parameters (HSP)** 构建可微分的物理引擎（PyTorch 实现）；
  - 设计 **Hybrid Normalized Loss** 平衡相对选择性（selectivity）与绝对溶解距离（solubility）；
  - 引入 **Audit Mode**，通过 L1 正则化（sparsity-inducing）自动剪枝非必要组分，体现 Occam’s Razor 原则。

---

### 相比现有方法的优势
| 维度 | AI4S-SDS | 现有方法（如 CoSCIENTIST, ChemCrow, ToT/GoT） |
|------|----------|---------------------------------------------|
| 上下文管理 | ✅ Sparse 存储 + 动态重建 | ❌ 文本级记忆检索，无法维持搜索树完整性 |
| 探索多样性 | ✅ 全局规划 + 兄弟感知机制 | ❌ 易受语言先验主导，路径依赖严重 |
| 物理一致性 | ✅ 可微物理层 + 约束优化 | ❌ 依赖工具调用或无连续优化能力 |
| 闭环能力 | ✅ 完整闭环：生成→优化→评估→反馈 | ⚠️ 多数为开环或弱闭环 |

> ✅ 表明显著改进；⚠️ 表示部分支持；❌ 表示缺失

---

## 2. 核心实验方法和设置

### 数据集
- **候选溶剂池**：共 $ N = 50 $ 种商用溶剂（来自 Chemical DB）；
- **目标任务**：设计用于光刻胶（photoresist）开发的新型显影液配方；
- **物理模型基础**：采用 **Hansen Solubility Parameters (HSP)** 理论构建混合物性质预测模型。

---

### 实验设置
- **搜索空间**：组合数超过百万级（2–5 组分混合），每种组合需优化连续体积分数 $\phi \in \mathbb{R}^{|M|}$，满足 simplex constraint；
- **MCTS 设置**：
  - 每轮迭代执行一定次数 rollout；
  - 使用 MoE 架构的 Generator 进行多角色协同生成；
  - Physics Engine 对每个候选配方执行梯度下降优化比例；
- **Token 预算限制**：模拟真实场景下的上下文长度限制（如 32K tokens）。

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **Physical Validity (PV)** | 是否满足热力学约束（如沸点顺序、闪点安全、HSP 距离） |
| **Top-10 Score** | 最优前 10 个配方的平均得分（综合性能） |
| **Exploration Diversity ($\mathcal{H}$)** | 基于唯一溶剂组成的 **Shannon Entropy**，衡量探索广度 |
| **Safety Zone Classification** | 根据 $D(\text{mix}, \text{protect})$ 判断是否进入 CRITICAL DANGER / BUFFER ZONE / ROBUST SAFE 区域 |

---

### 基线方法对比
| 方法 | 类型 | 是否具备 Agentic | Search | Sparse Memory | Physics-based Opt. |
|------|------|------------------|--------|---------------|--------------------|
| ReAct-Critic (Baseline) | LLM Agent | ✅ | ❌ | ❌ | ❌ |
| ChemCrow [4] | Tool-Augmented LLM | ✅ | ❌ | ❌ | ✅（工具） |
| CoSCIENTIST [3] | Lab Automation Agent | ✅ | Limited | ❌ | ❌ |
| ToT/GoT [31] | Structured Reasoning | ✅ | ✅（Tree/Graph） | ❌ | ❌ |
| RAP [17] | MCTS + LLM | ✅ | ✅（MCTS） | ❌ | ❌ |
| **AI4S-SDS (Ours)** | **Neuro-Symbolic MCTS** | ✅ | ✅（Sparse MCTS） | ✅ | ✅（Differentiable Physics） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 4）
| 方法 | PV | Top-10 Score | $\mathcal{H}$ (Entropy) | 主要失败模式 |
|------|----|------------|-------------------------|--------------|
| 1. ReAct-Critic Baseline | Low | 83.5 | 3.59 | Numerical Hallucination |
| 2. Naive MCTS + Physics | 100% | 86.5 | 3.53 | Mode Collapse |
| 3. MCTS + Sibling-Aware | 100% | 85.8 | 3.73 | — |
| **4. AI4S-SDS (Ours)** | **100%** | **81.17** | **4.37** | Redundant Exploration |

> 注：虽然 Top-10 Score 略降，但 **entropy 提升 23.7%**，表明探索更广泛、更均匀。

---

### 与基线方法的对比结果
- **物理有效性**：所有引入 Physics Engine 的方法均达到 **100% PV**，而纯 LLM 方法因数值幻觉导致大量无效方案；
- **探索多样性**：
  - AI4S-SDS 的 Shannon Entropy 达到 **4.37**，远高于 baseline 的 3.59 和 naive MCTS 的 3.53；
  - 图 2 显示其发现的**独特溶剂组合数量最多**，分布最均匀；
- **安全性保障**：
  - 所有最终推荐配方均落在 **ROBUST SAFE** 区域（$D_p \geq \text{Threshold}$）；
  - Audit Mode 成功剪枝掉 <1% 贡献的组分，提升工程简洁性。

---

### 消融实验结果
通过逐步添加模块验证各组件作用：
1. **Phase 1：加入 Physics Engine**
   - PV 从低提升至 100%，证明 LLM 无法可靠处理连续优化；
2. **Phase 2：引入 Sibling-Aware Expansion**
   - Entropy 从 3.53 → 3.73，减少局部重复探索；
3. **Phase 3：加入 Global Planning Module**
   - Entropy 大幅跃升至 **4.37**，Top-5 溶剂模板使用集中度显著下降（图 2c），说明成功打破全局模式锁定。

> 结论：**Global Planning 是提升探索多样性的最关键因素**。

---

## 4. 关键结论和发现

### 主要发现
1. **Diversity-aware search 比 score maximization 更适合科学发现任务**：
   - 单纯追求高分会导致 overfitting 到 evaluator bias；
   - AI4S-SDS 主动探索 evaluator 未覆盖的可行区域，发现结构新颖但仍物理有效的替代方案。

2. **Symbolic + Differentiable Coupling 是解决 discrete-continuous 优化的关键**：
   - LLM 负责拓扑生成，Physics Engine 负责几何优化，二者分工明确、互为补充；
   - Hybrid Loss + Audit Mode 实现了理论最优与工程最小之间的平衡。

3. **Memory-driven planning 改变了搜索动力学**：
   - 不再是盲搜或贪婪扩展，而是基于历史经验的战略重定向；
   - “Golden Features”、“Death List” 等抽象知识提升了搜索效率。

---

### 方法的局限性
1. **物理模型简化**：
   - 当前 Physics Engine 基于 HSP 线性混合规则，未完全捕捉非理想行为或长期材料退化效应；
2. **计算成本较高**：
   - MCTS + Gradient Descent 导致单次 rollout 开销较大，在超大规模搜索中可能受限；
3. **结果随机性**：
   - 由于 MCTS 的随机性，不同运行间候选集可能存在差异，需专家聚合判断；
4. **领域依赖性强**：
   - Global Plan 的有效性依赖高质量的历史 action-value 数据，冷启动阶段表现可能受限。

---

### 未来工作方向
1. **增强物理保真度**：
   - 引入更复杂的 thermodynamic models（如 UNIFAC）或结合分子动力学模拟；
2. **跨任务迁移能力**：
   - 探索该框架在其他配方设计任务（如电池电解质、药物制剂）中的泛化能力；
3. **人机协同接口优化**：
   - 构建可视化界面，辅助科学家理解 AI 推荐背后的“设计原理”；
4. **在线学习机制**：
   - 将实验室实测反馈直接纳入 Memory Module，实现真正的 lifelong discovery loop。

---

> ✅ **总结一句话**：  
> AI4S-SDS 通过 **Sparse MCTS + Differentiable Physics Alignment** 实现了**深度、多样且物理可信**的溶剂配方探索，在保持 100% 物理有效性的同时大幅提升搜索多样性，为 LLM 在复杂科学发现任务中的应用提供了新的范式。

</details>

---

### 11. [A Rubric-Supervised Critic from Sparse Real-World Outcomes](https://arxiv.org/abs/2603.03800)

**Authors**: Xingyao Wang, Valerie Chen, Heng Ji, Graham Neubig  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.03800v1  

#### Abstract
Academic benchmarks for coding agents tend to reward autonomous task completion, measured by verifiable rewards such as unit-test success. In contrast, real-world coding agents operate with humans in the loop, where success signals are typically noisy, delayed, and sparse. How can we bridge this gap...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Rubric-Supervised Critic from Sparse Real-World Outcomes

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **coding agent** 评测多依赖于学术基准（如 SWE-bench），这些基准以可验证的奖励信号（如单元测试通过率）作为成功标准。然而，在真实世界中，agent 是与人类协同工作的，成功信号往往是**稀疏、延迟且噪声大**的（例如 PR 是否被合并、代码是否长期存活）。这种“benchmark-success”与“real-world-success”的不一致性导致在 benchmark 上表现好的 agent 在实际应用中可能效果不佳。

本文旨在解决如何从**稀疏的真实世界交互数据**中学习一个有效的 **critic 模型**，用于评估 agent 行为，并支持训练和推理时的优化。

---

### 提出的新方法与创新点

#### ✅ **Critic Rubrics：基于行为特征的评分框架**
- 提出了一套包含 **24 个可观察的行为特征（rubrics）** 的监督框架，这些特征完全从 **human-agent interaction trace** 中提取，无需依赖最终结果。
- 特征分为三类：
  - **Agent Behavioral Issues**（如 `misunderstood intention`, `insufficient testing`）
  - **User Follow-Up Patterns**（如 `correction`, `frustration`, `reversion request`）
  - **Infrastructure Issues**（外部平台故障 vs agent 引发的故障）
- 这些 rubrics 提供了**密集的过程性反馈**，即使没有 outcome 标签也能进行监督。

#### ✅ **半监督训练机制（Semi-Supervised Multi-Task Learning）**
- 设计了一个 multi-task critic 模型，同时预测：
  - 24 个 rubric 特征（dense supervision）
  - 最终任务成功概率（sparse supervision，仅约 4–6% 数据有标签）
- 利用 rubric 预测提供**密集梯度信号**，将原本无标签的 96% 数据变为有效训练样本。

#### ✅ **细粒度 outcome proxy：Code Survival**
- 提出比 PR Merge 更精细的成功代理指标 —— **Code Survival**：
  > $$ \text{survival}(s) = \frac{\sum_{c \in C_s} \text{lines\_in\_final}(c)}{\sum_{c \in C_s} \text{lines\_total}(c)} $$
- 衡量 agent 编写的代码有多少比例保留在最终合并的 diff 中，能更准确地归因 credit。

---

### 相比现有方法的优势

| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 监督信号 | 仅依赖 outcome（稀疏） | 结合 dense rubrics + sparse outcome |
| 泛化能力 | 在 benchmark 上过拟合 | 能泛化到 real-world 场景 |
| 推理效率 | 使用 LLM 注释 rubrics 成本高 | 训练轻量 critic 模型实现高速打分（16×加速） |
| 应用场景 | 仅用于评估 | 支持 Best-of-K、early stopping、SFT 数据筛选 |

---

## 2. 核心实验方法和设置

### 使用的数据集

| 数据来源 | 类型 | 规模 | 说明 |
|--------|-----|------|------|
| **Real-World User-Agent Interactions** | 生产环境轨迹 | 38,241 次对话，151,837 个 segments | 来自 OpenHands 平台的实际开发者交互 |
| **SWE-Gym** | 学术 benchmark | 4,238 轨迹 | 用于训练和迁移测试 |
| **SWE-bench Verified** | 测试集 | 1,000 实例 | 主要用于下游任务性能评估 |

> 所有数据都被划分为 **segments**：从用户请求开始 → agent 完成（`finish` 动作）为止的最小工作单元。

---

### 实验设置与评估指标

#### ✅ 模型架构
- 基座模型：`Qwen3-4B-Instruct`
- 多任务头：联合预测 24 个 rubric + success 概率
- 输入格式：完整 segment trace（平均 38K tokens，最长截断至 64K）

#### ✅ 训练目标对比
| 方法 | 描述 |
|------|------|
| **Success-Only** | 仅预测 outcome（PR Merge 或 Code Survival） |
| **Success+Rubrics** | 联合预测 rubrics 和 outcome（本文提出） |

#### ✅ Outcome Proxy 对比
| Proxy | 定义 | 标注覆盖率 |
|-------|------|-----------|
| **PR Merge** | PR 是否被合并 | 6% segments |
| **Code Survival** | agent 代码保留比例（连续值） | 4% segments（但更细粒度） |

#### ✅ 评估指标
| 场景 | 指标 |
|------|------|
| **Intrinsic Critic Quality** | AUC, F1, Precision, Recall（对 outcome 的预测） |
| **Downstream Agent Performance** | Best@K（Best@8）、MRR、Early Stopping 效果 |
| **Cross-Backbone Generalization** | 在不同 LLM（Sonnet vs Opus）上的 ranking 性能 |
| **Training-Time Utility** | SFT 数据选择后的 solve rate 提升 |

#### ✅ 基线方法
- **No Real-World Data**：仅用 SWE-Gym 数据训练 critic
- **Success-Only**：无 rubric 辅助任务
- **Random Selection**：随机选择轨迹作为 baseline

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

#### 📊 **Best-of-8 性能提升（SWE-bench Mixed Subset）**

| 模型 | Best@8 | 相比 Random@8（57.9%） |
|------|--------|------------------|
| Success-Only (Survival) | 63.6% | +5.7 pts |
| **Success+Rubrics (BCE-floor)** | **73.8%** | **+15.9 pts** ✅ |

> ⬆️ **相对提升高达 15.9 个百分点**，显著优于仅使用 outcome 监督的方法。

#### ⏱️ **Early Stopping 效果**
- 使用 `Success+Rubrics (BCE-floor)`：
  - 平均尝试次数：**1.35 次**
  - 相比穷举 8 次：**节省 83% 计算资源**
  - 性能提升：**+17.7 pts over random**

> 表明 critic 可靠识别高质量轨迹，支持高效推理调度。

#### 🔍 **跨 backbone 泛化能力（Table 3）**

| 模型 | Sonnet 上 MRR | Opus 上 MRR | Combined MRR |
|------|---------------|-------------|--------------|
| Success-Only | 0.77 | 0.72 | 0.78 |
| **Success+Rubrics (BCE-floor)** | 0.76 | **0.74** | **0.83** ✅ |

> Success-Only 在 Opus 上表现下降，显示其**对特定 LLM 过拟合**；而 rubric-supervised critic 具备更强的**跨模型鲁棒性**。

#### 🧪 **消融实验结果**

| 发现 | 内容 |
|------|------|
| ❌ **仅用 benchmark 数据训练无效** | 在 real-world 上 AUC ≈ 0.45–0.48（低于随机） |
| ✅ **Code Survival > PR Merge** | 尽管标签更少，但提供更细粒度 credit 分配，AUC 达 **0.69 vs 0.58** |
| ✅ **Rubric supervision 提升 learnability** | critic 在 rubric 预测上达到平均 AUC **0.78**，关键特征达 **0.81**（见 Appendix E） |
| 🔁 **训练动态稳定** | 性能在 step 4000–6000 达峰，后期轻微退化（Appendix F） |

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **真实世界监督是必要的**  
   仅在 benchmark 上训练的 critic 无法迁移到 real-world 场景（AUC < 0.5），甚至会损害下游性能。

2. ✅ **Code Survival 是比 PR Merge 更优的 outcome proxy**  
   尽管标注更稀疏，但由于其**细粒度、可归因性强**，能更好指导 critic 学习。

3. ✅ **Rubric supervision 显著提升 critic 的泛化性和实用性**  
   - 支持跨 LLM backbone 的可靠 ranking
   - 实现高效的 early stopping 和 Best-of-K selection
   - 可用于训练时数据筛选（data curation）

4. ✅ **Critic 不仅可用于评估，还可赋能 agent 全流程优化**  
   - 推理时：Best-of-K、early stopping
   - 训练时：筛选高质量 trajectory 用于 SFT

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **Rubric 定义可能存在偏见** | 当前 24 个 rubrics 基于专家设计和 LLM 分析，可能未覆盖所有 failure mode |
| **Outcome proxy 仍是近似信号** | Code Survival ≠ 正确性，仍受人为修改、协作影响 |
| **依赖 segment 划分质量** | segment 边界检测错误会影响 credit 归属 |
| **未完全端到端集成到 RL pipeline** | 当前主要用于 SFT 数据筛选，尚未验证在 RLHF 中的效果 |

---

### 未来工作方向

1. **扩展 rubric taxonomy**  
   引入更多维度（如安全性、可维护性、文档完整性）以支持更全面评估。

2. **构建动态 rubric 更新机制**  
   根据用户反馈持续迭代 rubric 定义，适应不同团队或项目风格。

3. **将 critic 集成进 RL 训练循环**  
   作为 reward model 直接用于 PPO 或 DPO 训练，探索其在 online learning 中的作用。

4. **支持多模态输入**  
   结合 IDE 截图、语音指令等 richer context 提升评估准确性。

5. **开放生态建设**  
   已开源 [critic-rubrics](https://github.com/OpenHands/critic-rubrics) 和 [model](https://huggingface.co/OpenHands/openhands-critic-4b-v1.0)，鼓励社区共建通用 critic 标准。

---

> 💡 **一句话总结**：  
> 本文提出 **Critic Rubrics** 框架，通过引入 **24 个 trace-observable 行为特征**，实现了从**稀疏真实世界反馈中学习强泛化 critic 模型**，不仅提升了 agent 评估质量，还支持 **Best-of-K、early stopping 和 SFT 数据筛选**，为 bridging benchmark-to-reality gap 提供了实用路径。

</details>

---

### 12. [ErrorLLM: Modeling SQL Errors for Text-to-SQL Refinement](https://arxiv.org/abs/2603.03742)

**Authors**: Zijin Hong, Hao Chen, Zheng Yuan, Qinggang Zhang, Luyao Zhuang, Qing Liao, Feiran Huang, Yangqiu Song, Xiao Huang  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.03742v1  

#### Abstract
Despite the remarkable performance of large language models (LLMs) in text-to-SQL (SQL generation), correctly producing SQL queries remains challenging during initial generation. The SQL refinement task is subsequently introduced to correct syntactic and semantic errors in generated SQL queries. How...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ERRORLLM: Modeling SQL Errors for Text-to-SQL Refinement

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Model (LLM)** 的 **Text-to-SQL** 系统在初始生成阶段常产生语法或语义错误。尽管已有 **self-debugging** 和 **self-correction** 等 **refinement** 方法，但仍存在两大局限：
- **Self-debugging** 依赖执行反馈，但现代 LLM 很少产生显式执行错误（仅占错误的 3%），导致该方法覆盖范围极小。
- **Self-correction** 缺乏对错误的显式建模，检测精度低，且容易对正确 SQL 进行“过度修正”（hallucination），造成 **corruption**（将正确的 SQL 改错）。

### 提出的新方法：ERRORLLM
作者提出 **ERRORLLM**，一个显式建模 SQL 错误的框架，用于提升 **text-to-SQL refinement** 效果。其核心创新包括：

#### （1）显式错误建模（Explicit Error Modeling）
- 引入专用的 **error tokens**（如 `[ERR]1`, `[ERR]2`...）扩展 LLM 的词表，每个 token 对应一种预定义的语义错误类型（共 12 类）。
- 通过训练使 LLM 能够预测这些 error tokens，从而实现对复杂隐式错误的精准检测。

#### （2）两阶段错误检测机制
- **静态表面检测（Static Superficial Detection）**：基于规则检测执行失败和结构不匹配。
- **LLM 语义检测（LLM-based Semantic Detection）**：利用微调后的 ERRORLLM 检测无法通过规则捕捉的深层语义错误。

#### （3）错误引导的精细化修复流程
- **错误定位与分析（Error Localization and Analysis）**：使用专门的 LLM（LocLLM）精确定位 AST 中的错误节点及涉及的 schema 元素。
- **优先级排序的修复（Priority-Ordered Refinement）**：按错误严重性排序，引导 LLM（REFLLM）进行有针对性的修复，避免迭代修正中的幻觉累积。

### 相比现有方法的优势
- **高检测精度**：结合规则与语义模型，显著提升错误召回率（Recall）和 F1 分数。
- **低 corruption 率**：避免对正确 SQL 的无效修改，保护原始正确结果。
- **强泛化能力**：在不同 backbone LLM（如 GPT-4o, OpenSearch-SQL）上均能稳定提升性能。
- **可扩展性**：预留 error token 槽位（`[ERR]13` 到 `[ERR]N`），便于未来扩展新错误类型。

---

## 2. 核心实验方法和设置

### 数据集
- **主基准测试集**：
  - **BIRD**：大规模、复杂的 text-to-SQL 基准，使用官方开发集（1,534 样本）。
  - **SPIDER**：标准跨域 text-to-SQL 数据集，使用官方开发集（1,034 样本）。
- **变体测试集（SPIDER VARIANTS）**：
  - SPIDER-REALISTIC、SPIDER-SYN、SPIDER-DK，用于测试泛化能力。
- **专项错误检测基准**：
  - **NL2SQL-BuGs**：专家标注的 SQL 语义错误数据集（2,018 样本），用于细粒度错误类型检测评估。

### 实验设置
- **Backbone LLMs**：GPT-4o 和 OpenSearch-SQL。
- **ERRORLLM 模型**：基于 **CodeS-7B** 微调，使用 **LoRA** 适配器。
- **LocLLM & REFLLM**：默认使用 GPT-4o，也测试了 DeepSeek-V3。
- **训练数据合成**：通过规则扰动（rule-based perturbation）和 LLM 辅助注入（LLM-assisted injection）从 BIRD 训练集生成带标签的错误 SQL。

### 评估指标
| 指标 | 定义 |
|------|------|
| **EX (Execution Accuracy)** | 执行准确率，主要端到端评估指标 |
| **D-Accuracy** | 二分类错误检测准确率 |
| **D-F1** | 错误检测的 F1 分数（Precision & Recall 的调和平均） |
| **Fixed Rate (FR)** | 成功修复的真实错误比例 |
| **Corruption Rate (CR)** | 被错误修改的正确 SQL 比例 |
| **Type-specific Accuracy (TSA)** | 细粒度错误类型的检测准确率 |

### 基线方法对比
| 类别 | 基线方法 |
|------|--------|
| **Self-correction** | Vanilla self-correction [19], DIN-SQL [37] |
| **Self-debugging** | Self-debugging [8] |
| **Self-consistency** | C3 [11], PET-SQL [29], MCS-SQL [21] |
| **Dedicated Refinement** | Refiner [47], SQLFixAgent [6], MAGIC [1], SHARE [40] |

---

## 3. 主要实验结果和性能指标

### 端到端执行准确率（EX）提升显著
在 **BIRD** 和 **SPIDER** 上，ERRORLLM 取得了最显著的提升：

| 方法 | BIRD (Overall) | SPIDER (Overall) |
|------|----------------|------------------|
| GPT-4o (Baseline) | 55.87% | 75.44% |
| + ERRORLLM (GPT-4o) | **66.23% (+18.54%)** | **86.94% (+15.24%)** |
| + ERRORLLM (DeepSeek-V3) | 64.99% | 86.36% |

> ✅ **在更强的 backbone OpenSearch-SQL 上，ERRORLLM 仍能带来正向增益（+1.27% on BIRD, +2.39% on SPIDER）**，而其他方法多出现性能下降（因 corruption）。

### 错误检测性能领先
在 **BIRD 开发集** 上的错误检测任务中，ERRORLLM 显著优于基线：

| 方法 | Precision | Recall | D-F1 |
|------|-----------|--------|-------|
| Self-correction [37] | 61.93% | 30.28% | 40.67% |
| SQLFixAgent [6] | 76.73% | 70.61% | 73.54% |
| **ERRORLLM (Ours)** | **80.12%** | **76.22%** | **78.12%** |

> ✅ **ERRORLLM 在保持高精度的同时，召回率远超基于执行反馈的方法（如 Refiner 仅 2.95%）**。

### 消融实验（Ablation Study）
在 BIRD 开发集上的消融研究验证了各组件的重要性：

| 移除组件 | D-F1 ↓ | EX ↓ | 结论 |
|----------|--------|------|------|
| w/o LLM-based detection | 65.91 | 7.82 | 语义检测是核心机制 |
| w/o LLM-assisted injection | 10.57 | 4.10 | 学习真实 LLM 错误行为至关重要 |
| w/o Rule-based perturbation | 1.77 | 1.04 | 规则扰动作用较小 |
| w/o Question-schema structure | 5.31 | 3.52 | 结构化表示对检测和修复均有放大效应 |
| w/o Error localization | — | 4.04 | 定位模块对修复效果影响最大 |

> 🔍 发现：**D-F1 下降与 EX 下降呈正相关**，证明“高质量的错误检测是有效修复的前提”。

---

## 4. 关键结论和发现

### 主要发现
1. **错误检测质量决定修复效果**：现有方法因检测不准而导致高 corruption，限制了整体提升。
2. **显式错误建模是关键**：通过引入 **error tokens**，ERRORLLM 能够精准识别复杂语义错误，为后续修复提供可靠信号。
3. **ERRORLLM 是唯一能在强 backbone 上持续提升的方法**：在 OpenSearch-SQL 上仍能增益，证明其鲁棒性和实用性。
4. **细粒度错误检测可行**：即使作为 7B 模型，ERRORLLM 在多数错误类别上的 TSA 与 GPT-4o 等大模型相当。

### 局限性
- 当前错误类型仅定义了 12 类，未覆盖 **subquery-related errors** 和 **operator-related errors**（如逻辑算符错误）。
- 依赖外部 schema linking 或需运行 RGAT 编码器，增加系统复杂性。
- 修复过程仍依赖较强的 LLM（如 GPT-4o）作为 REFLLM。

### 未来工作方向
- 扩展错误类型覆盖范围，特别是子查询和操作符相关错误。
- 探索更轻量化的定位与修复模块，降低对 proprietary LLM 的依赖。
- 将 ERRORLLM 的思想应用于其他代码生成任务的错误修正。

---

> 📌 **总结**：ERRORLLM 通过**显式建模 SQL 错误**，构建了一个**检测-定位-修复**的闭环流程，在多个 benchmark 上实现了 SOTA 的 text-to-SQL refinement 效果，为解决 LLM 生成错误的“黑箱修正”问题提供了新范式。

</details>

---

### 13. [Position: Vector Prompt Interfaces Should Be Exposed to Enable Customization of Large Language Models](https://arxiv.org/abs/2603.04292)

**Authors**: Liangwei Yang, Shiyu Wang, Haolin Chen, Rithesh Murthy, Ming Zhu, Jielin Qiu, Zixiang Chen, Juntao Tan, Jianguo Zhang, Zhiwei Liu, Wenting Zhao, Silvio Savarese, Caiming Xiong, Huan Wang, Shelby Heinecke  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.04292v1  

#### Abstract
As large language models (LLMs) transition from research prototypes to real-world systems, customization has emerged as a central bottleneck. While text prompts can already customize LLM behavior, we argue that text-only prompting does not constitute a suitable control interface for scalable, stable...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Position: Vector Prompt Interfaces Should Be Exposed to Enable Customization of Large Language Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大型语言模型（LLMs）在企业级部署中面临**定制化瓶颈**。尽管可通过 **text-based prompting** 或 **fine-tuning** 进行行为调整，但二者均存在显著缺陷：
- **Text prompts** 虽灵活易用，但在系统级定制中表现**不稳定、难以迭代优化、扩展性差**；
- **Fine-tuning** 需要梯度访问和训练资源，不适用于大多数下游用户（尤其在黑盒 API 场景下），且更新成本高。

因此，论文指出：**现有的文本提示接口不适合作为可扩展、稳定、仅推理（inference-only）场景下的定制控制接口**。

---

### 🚀 提出的新方法/新思路
提出将 **vector prompt inputs** 作为 LLM 定制的**公开接口**之一，与 text prompt 并列，形成“双通道”控制机制。

- 将 **vector prompt** 视为一种**接口抽象（interface abstraction）**，而非特定优化技术；
- 主张模型提供商应暴露一个允许用户注入、复用和管理固定长度 **continuous control vectors** 的 API 接口；
- 强调该接口可在 **black-box setting** 下运行，无需访问模型权重、梯度或内部激活。

> 🔍 核心思想：**Customization 应被视为“控制接口设计问题”，而不仅仅是“prompt engineering”或“参数微调”问题。**

---

### ⚖️ 相比现有方法的优势

| 维度 | Text Prompt | Fine-tuning | Vector Prompt（本文主张） |
|------|-----------|------------|--------------------------|
| 是否需要训练 | 否 | 是 | 否 |
| 是否支持 inference-only | 是 | 否 | 是 |
| 控制稳定性 | 差（语义敏感） | 好 | 好 |
| 可学习性（learnability） | 低（离散空间难优化） | 高 | 高（连续空间） |
| 扩展性 | 低（易饱和） | 中等 | 高 |
| 部署灵活性 | 高 | 低 | 高 |
| 上下文效率 | 低（长 prompt 导致 dilution） | — | 高（少量向量即可编码复杂行为） |

> ✅ **Vector prompts 在保持 inference-only 兼容的同时，提供了更高的控制效率、更强的监督吸收能力和更稳定的迭代优化路径。**

---

## 2. 核心实验方法和设置

### 📚 数据集
- **SST-5**：情感分类任务（5类电影评论），用于分析不同 prompt 接口在增加监督数据时的表现。
- 使用 Hugging Face 提供的标准版本：`SetFit/sst5`

---

### ⚙️ 实验设置
- **Backbone 模型**：LLaMA3-8B-Instruct（AI@Meta, 2024）
- **目标**：比较不同 prompt 接口对监督信号的吸收能力（scaling behavior）
- **变量控制**：
  - 固定模型架构与任务；
  - 仅改变训练样本数量（从少到多）；
  - 对比三种 prompt 类型的表现：
    1. **Human-written text prompt**
    2. **Optimized text prompt**（使用 TextGrad 优化）
    3. **Optimized vector prompt**（通过 prompt tuning 学习）

> ⚠️ 注意：gradient-based prompt tuning 仅作为**诊断工具**（diagnostic upper bound），不代表实际部署方式。

---

### 📊 评估指标
- **准确率（Accuracy）**：在验证集上的分类性能；
- **Scaling curve**：随着训练数据量增加，各方法性能的变化趋势；
- **Attention pattern visualization**：通过可视化注意力分布，分析 prompt tokens 如何被 task tokens 利用。

---

### 🔁 基线方法对比
| 方法类别 | 代表技术 | 是否需梯度 | 是否 inference-only |
|--------|---------|-----------|---------------------|
| Manual Prompting | Hand-crafted text prompts | × | ✓ |
| Text Prompt Optimization | TextGrad, RLPrompt | △（部分） | ✓ |
| Prompt Tuning | Prompt tuning, Prefix tuning | ✓ | × |
| Black-box Vector Optimization | BBT, ZOO | × | ✓ |

> 表格来源：Table 1（原文）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Figure 2）

| 方法 | 性能随数据增长趋势 |
|------|------------------|
| **Human-written text prompt** | 几乎无提升，性能稳定但上限低 |
| **Optimized text prompt** | 小样本有提升，但很快饱和（~200 samples 后趋于平缓） |
| **Optimized vector prompt** | 持续受益于更多监督数据，未见饱和迹象 |

> 💡 结论：**vector prompt 接口具有更强的信息吸收能力，其性能随监督增强而持续上升；text prompt 存在明显的接口级瓶颈。**

---

### 🔍 注意力模式分析（Figure 3）

#### Text Prompt 特征：
- 注意力稀疏（sparse utilization）
- Prompt tokens 与普通输入 token 无明显区别
- 多数 attention 集中在初始 BOS token（attention sink 现象）
- 不同层之间无一致模式，影响短暂

#### Vector Prompt 特征：
- 注意力密集且全局可寻址（dense and globally attended）
- 多个 learned vector tokens 被反复引用
- attention sink 效应减弱，注意力更均匀分布
- 跨层稳定，表明其作为**持久控制锚点（persistent control anchors）**

> ✅ 发现：**vector prompts 改变了模型内部的信息流动机制，更像是“内置控制器模块”，而非“临时指令”。**

---

### ❌ 消融实验（间接体现）
虽然没有显式的消融表，但以下对比构成逻辑上的消融分析：
- **相同优化算法 + 不同接口 → 性能差异显著** ⇒ 差异源于接口本身，而非优化过程；
- **vector prompt 在 mid-layer 和 late-layer 均有效** ⇒ 控制作用贯穿整个前向传播；
- **text prompt 即使经过优化仍快速饱和** ⇒ 离散语言表示是根本限制。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Text prompts 存在接口级瓶颈**：
   - 其离散性和语义依赖导致难以系统化吸收监督；
   - 易出现 prompt 爆炸（prompt explosion）、维护困难、性能早饱和。

2. **Vector prompts 是更优的控制接口抽象**：
   - 提供连续、高效、稳定的控制信号；
   - 支持 inference-only 场景下的数据驱动优化；
   - 可实现高控制效率（few vectors 控制复杂行为）。

3. **接口设计应独立于优化方法**：
   - Vector prompt 是一种接口形式，可用 gradient-based、black-box search（如 BBT/ZOO）等多种方式填充；
   - 当前 black-box 方法虽弱于 full tuning，但已有成效，未来有望逼近上限。

4. **安全性可控**：
   - 在标准 black-box 威胁模型下，暴露 vector prompt 输入不会引入新的可观测攻击面；
   - 输出通道容量不变，信息泄露风险与 text prompt 等价。

---

### ⚠️ 局限性
- **并非否定 text prompts 的有效性**：对于简单、静态任务，text prompts 依然适用；
- **不解决所有安全问题**：仅论证风险未加剧，不代表完全安全；
- **依赖 black-box 优化进展**：目前 black-box vector optimization 效率仍低于 gradient-based 方法；
- **尚未大规模工程验证**：API 设计、版本管理、兼容性等问题有待实践检验。

---

### 🔮 未来工作方向
1. **开发高效的 inference-only vector optimization 方法**：
   - 如基于进化算法、零阶优化（ZOO）、扩散模型等；
2. **构建统一的 vector prompt 管理框架**：
   - 支持存储、共享、组合、迁移多个 control vectors；
3. **建立 prompt interface 评估基准**：
   - 包括 control efficiency、scalability under supervision、behavioral stability 等维度；
4. **推动工业界采纳 vector prompt 接口标准**：
   - 鼓励 LLM 提供商开放 constrained vector prompt API；
5. **探索 hybrid interfaces**：
   - 结合 text + vector prompts，兼顾可读性与控制精度。

---

## 📣 总结一句话
> 本论文呼吁将 **vector prompt** 从一种训练技巧升格为 LLM 的**一级定制接口**，以解决当前 text-only prompting 在可扩展性、稳定性与推理兼容性方面的根本局限，为现实世界中的企业级 LLM 应用提供更强大的控制能力。

</details>

---

### 14. [Lang2Str: Two-Stage Crystal Structure Generation with LLMs and Continuous Flow Models](https://arxiv.org/abs/2603.03946)

**Authors**: Cong Liu, Chengyue Gong, Zhenyu Liu, Jiale Zhao, Yuxuan Zhang  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.03946v1  

#### Abstract
Generative models hold great promise for accelerating material discovery but are often limited by their inflexible single-stage generative process in designing valid and diverse materials. To address this, we propose a two-stage generative framework, Lang2Str, that combines the strengths of large la...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Lang2Str: Two-Stage Crystal Structure Generation with LLMs and Continuous Flow Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统材料生成模型（如 VAE、Diffusion、Flow-based models）在晶体结构生成中面临以下挑战：
- **数值精度不足**：LLMs 在直接生成 CIF 文件中的连续数值（如原子坐标、晶格参数）时表现差，存在数值理解与精度问题。
- **离散变量建模困难**：原子类型、空间群（space group）等离散且具有复杂拓扑结构的变量难以被统一建模。
- **缺乏可控性和可解释性**：单阶段生成框架难以实现细粒度控制，且生成过程不透明。

### 🚀 提出的新方法：Lang2Str
提出一种**两阶段生成框架 Lang2Str**，结合 **Large Language Models (LLMs)** 和 **Continuous Flow Models** 的优势：
1. **第一阶段（语言描述生成）**：
   - 使用微调后的 LLM（如 LLaMA2）生成关于晶体结构的**自然语言描述**（natural language description），包括几何布局、空间群、成键关系等。
   - 描述基于化学组成 $A$ 和预测的空间群 $S$ 条件生成，即建模 $p(T|A,S)$。
2. **第二阶段（结构解码）**：
   - 使用 **text-conditioned flow matching model** 将文本描述 $T$ 解码为精确的原子分数坐标 $F$ 和晶格参数 $L$，即建模 $p(M|A,T)$。
   - 利用 MatSciBERT 编码文本，并通过 cross-attention 层融合文本与结构表示。

### 🔍 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **分解式建模** | 显式分离 $p(S\|A)$, $p(T\|A,S)$, $p(M\|A,T)$，提升可控性与可解释性 |
| **避免数值生成瓶颈** | LLM 不直接生成数值，而是输出语义丰富的文本描述，规避其对数字的弱项 |
| **更强的条件引导** | 文本描述提供比单一空间群更丰富的几何先验信息 |
| **模块化设计** | 可灵活替换 LLM 或 flow model 组件，便于扩展 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **MP-20**：Materials Project 子集，包含少于 20 个原子的稳定晶体结构，用于 *ab initio generation* 和 *crystal structure prediction (CSP)*。
- **MPTS-52**：更具挑战性的数据集，每个 unit cell 最多含 52 个原子，测试模型泛化能力。
- 数据划分：采用标准的 60%-20%-20% 训练/验证/测试分割。

### ⚙️ 实验设置
- **LLM 模型**：LLaMA2-7B，使用 Robocrystallographer 自动生成的文本进行微调。
- **Flow 模型架构**：基于 CrystalFlow 架构，使用 6 层 periodic equivariant message passing network，每层后接 cross-attention 层以融合 MatSciBERT 文本嵌入。
- **训练细节**：
  - LLM 微调：64 V100 GPU，学习率 $5\times10^{-4}$，batch size 64，最多 30 轮。
  - Flow 模型训练：3000 轮，学习率 $1\times10^{-5}$，使用 Euler sampler 进行 100 步采样。

### 📊 评估指标
#### 对于 *Ab Initio Generation*：
- **有效性**：Structural Validity (%)、Compositional Validity (%)
- **覆盖率**：Recall (%)、Precision (%)
- **稳定性代理指标**：
  - Match Rate (MR%)：CHGNet 优化前后结构相似度
  - Δ-Energy：原始 vs 优化后能量差
  - RMSD：坐标偏差
- **真实稳定性**：DFT 计算能量高于凸包（energy above hull）< 0 的比例
- **新颖性**：S.U.N. rate（Stable, Unique, Novel）

#### 对于 *Crystal Structure Prediction*：
- Match Rate (%)：预测结构与真实结构匹配的比例
- RMSE：晶格参数与原子坐标的均方根误差

### 🆚 基线方法对比
| 方法 | 类型 |
|------|------|
| CDVAE | VAE-based |
| DiffCSP | Diffusion-based |
| FlowMM / CrystalFlow | Flow matching-based |
| FlowLLM | LLM + Flow refinement |
| UniGenX / Uni-3DAR | Autoregressive generation |

---

## 3. 主要实验结果和性能指标

### 📈 Ab Initio Generation 结果（见 Table 2 & 3）
| 指标 | Lang2Str 表现 | 对比优势 |
|------|----------------|-----------|
| **Compositional Validity** | **91.63%** | 超过所有基线（最高为 FlowLLM 的 90.84%） |
| **Structural Validity** | 99.59% | 接近 SOTA |
| **Precision Coverage** | **99.97%** | 当前最优 |
| **Match Rate (CHGNet)** | **96.2%** | 显著优于 DiffCSP (88.0%) 和 FlowLLM (94.9%) |
| **Δ-Energy** | **0.055 eV/atom** | 所有方法中最低，表明生成结构已非常接近基态 |
| **S.U.N. Rate** | **3.2%**（无策略）→ **5.8%**（加入 rejection sampling） | 明显优于 DiffCSP 的 2.7%，显示更强探索能力 |

> 💡 **Rejection Sampling 策略**：丢弃训练集中已存在的化学式样本，显著提升新颖性。

### 🧱 Crystal Structure Prediction 结果（见 Table 4）
| 数据集 | 方法 | Match Rate (%) | RMSE ↓ |
|--------|------|----------------|--------|
| MP-20 | Lang2Str | **63.92** | 0.076 |
| MPTS-52 | Lang2Str | **28.36** | 0.1424 |

- 在两个数据集上均优于 DiffCSP、FlowMM、CrystalFlow 等 flow/diffusion 基线。
- 尽管 Uni-3DAR 在 MPTS-52 上 RMSE 更低（0.0684），但 Lang2Str 在文本条件生成路径下仍具竞争力。

### 🔍 消融实验结果（Ablation Studies）

#### （1）LLM 是否只是记忆训练数据？
- 在 **全新化学式集合**（来自 DiffCSP 且不在 MP 中）上测试 CSP 性能：
  - **Match Rate 达到 55.2%**，RMSE = 0.048
  - 表明 LLM 具备良好泛化能力，非简单记忆。

#### （2）Flow 模型是否仅依赖空间群信息？
- 设计对照实验：将 CrystalFlow 改为仅以 one-hot space group 为条件。
- 结果（Table 5）：
  - Lang2Str（使用完整文本） > CrystalFlow + SP encoding
  - 即使使用检索到的空间群，**文本描述提供了超越空间群的额外几何信息**

#### （3）采样步数影响（Table 7）
- 增加 flow sampling 步数 → RMSE 下降（精度提高），但 Match Rate 略降
- 说明当前 Match Rate 指标可能存在“宽松匹配”问题，需谨慎解读

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Lang2Str 实现了高保真、高有效性和高新颖性的晶体生成**：
   - 生成结构在几何和能量层面均更接近真实基态。
2. **自然语言是有效的中间表示**：
   - LLM 生成的文本描述作为 high-level condition，显著优于仅使用空间群或化学式的粗粒度条件。
3. **两阶段框架增强可控性与可解释性**：
   - 可独立调节 LLM 输出风格或 flow 模型行为，支持定制化材料设计。
4. **LLM 并未过拟合或记忆数据**：
   - 能成功泛化至未见过的化学系统，证明其具备推理能力。

### ⚠️ 方法的局限性
1. **两阶段流程引入延迟**：
   - 需先运行 LLM 再运行 flow model，推理速度慢于端到端模型。
2. **LLM 对空间群预测不准**：
   - 若仅给化学式，LLM 常生成错误空间群，影响下游性能；目前依赖外部工具（CSPML）辅助预测。
3. **文本描述质量依赖训练数据分布**：
   - Robocrystallographer 生成的文本风格可能限制 LLM 表达多样性。

### 🔮 未来工作方向
1. **构建端到端联合优化框架**：
   - 探索 jointly train LLM 和 flow model，减少阶段间误差传播。
2. **改进空间群预测模块**：
   - 引入更强的 graph-based predictor 替代 CSPML。
3. **拓展至其他材料类型**：
   - 如分子晶体、二维材料、合金等。
4. **引入人类反馈机制**：
   - 构建 human-in-the-loop 的交互式材料设计系统。

---

> ✅ **总体评价**：  
> Lang2Str 提出了一种新颖且有效的**语言中介生成范式**（language-mediated generation），巧妙规避了 LLM 数值生成缺陷，同时充分发挥其结构化推理能力，在 ab initio 生成与结构预测任务中达到 SOTA 水平，为多模态材料发现提供了坚实基础。

</details>

---

### 15. [A Multi-Agent Framework for Interpreting Multivariate Physiological Time Series](https://arxiv.org/abs/2603.04142)

**Authors**: Davide Gabrielli, Paola Velardi, Stefano Faralli, Bardh Prenkaj  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.04142v1  

#### Abstract
Continuous physiological monitoring is central to emergency care, yet deploying trustworthy AI is challenging. While LLMs can translate complex physiological signals into clinical narratives, it is unclear how agentic systems perform relative to zero-shot inference. To address these questions, we pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Multi-Agent Framework for Interpreting Multivariate Physiological Time Series

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对**安全关键医疗场景**（如急诊医学）中部署可信 AI 系统的挑战，特别是以下三个未被充分探索的问题：
- **(C1) 临床工作流适配性**：如何将复杂的生理时间序列转化为符合临床推理流程（如分诊、假设生成、同行评审）的解释。
- **(C2) Agentic 推理的有效性**：多智能体（multi-agent）系统是否总能提升解释质量？尤其对于具备“思考能力”的大模型（thinking models），其效果尚不明确。
- **(C3) 医生期望对齐**：解释必须简洁、情境化，并符合医生熟悉的临床表达方式，否则可能降低信任度。

### 提出的新方法与思路
作者提出了 **Vivaldi** —— 一个**角色结构化的多智能体框架**（role-structured multi-agent system），用于解释多变量生理时间序列。其核心创新在于：
- **模拟真实急诊团队协作流程**：将解释生成过程建模为五个角色代理协同完成：
  1. **Triage Agent**：计算安全指标（如 Shock Index, qSOFA）并设定个性化阈值。
  2. **Doctor Agent**：基于趋势迭代形成和修正诊断假设。
  3. **Consultant Agent**：提供同行评审，指出盲点和替代诊断。
  4. **Coder Agent**：执行定量分析和可视化任务（通过生成可执行 Python 代码）。
  5. **Synthesizer Agent**：整合所有证据生成最终临床报告。
- **Orchestrator 协调机制**：维护共享临床状态（Shared Memory Buffer, SMB），管理代理间通信与迭代流程。

### 相比现有方法的优势
- **超越单一模型零样本推理**（zero-shot inference）：通过外部化计算和结构化推理，提升了小模型和医学专用模型的表现。
- **显式工具调用保障数值精度**：关键临床指标由独立的 `CoderAgent` 通过确定性代码计算，避免 LLM 内部数值推理错误。
- **贴近真实临床决策流程**：多轮迭代、同行咨询、证据筛选等设计更符合实际医生工作模式。
- **可审计性强**：每一步推理、代码、图表均可追溯，增强透明性和可信度。

---

## 2. 核心实验方法和设置

### 数据集
使用 **MC-MED**（Multimodal Clinical Monitoring in the Emergency Department）数据集，包含来自学术医疗中心的大规模匿名急诊就诊记录。研究聚焦于以下信息：
- **生命体征**（Vital Signs）：HR, BP (SBP/DBP), SpO₂, RR, Temp
- **人口统计学**：年龄、性别、种族
- **背景信息**：主诉（Chief Complaint）、既往病史（PMH）、用药情况
- **目标标签**：急诊严重程度指数（ESI）、疼痛评分（Pain Score）、住院时长（LOS）

### 实验设置
- **模型家族对比**：在五类 LLM 上测试 Vivaldi 框架：
  - Thinking Models：Gemini 3 Pro, Claude 4.5 Opus
  - Non-Thinking / 医学微调模型：GPT 5.2, Llama 4 Maverick, MedGemma 27B
- **两种推理策略对比**：
  - **Zero-Shot Baseline**：单次提示输入完整上下文，要求模型自行完成全部分析。
  - **Agentic Pipeline**：运行完整的 Vivaldi 多智能体流程。
- **控制变量**：保持提示逻辑和编排层一致，仅更换底层 LLM，以隔离模型本身的影响。

### 评估指标
通过一项受控的**临床专家评估实验**进行评估，共收集 **109 条匿名专家反馈**，涵盖六个维度：
| 维度 | 描述 |
|------|------|
| **Factuality** | 报告的事实和机制是否正确（No/Partially/Yes） |
| **Justification** | 结论是否有数据支持（No/Partially/Yes） |
| **Relevance** | 信息是否针对具体病例而非泛泛而谈（No/Partially/Yes） |
| **Trust** | 解释是否足够稳健和安全可用于临床决策（No/Partially/Yes） |
| **Chart Comprehensibility** | 图表是否易于理解（5点李克特量表） |
| **Clinical Utility** | 是否有助于临床决策（5点李克特量表） |

此外还评估了：
- **Triagemetrics 的预测性能**（如 ESI F1-score, MAP MAE）
- **计算效率**（延迟、token 消耗）

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）解释质量：模型依赖性强，非普适增益
| 指标 | Thinking Models (Zero-shot → Agentic) | Non-Thinking Models (Zero-shot → Agentic) |
|------|----------------------------------------|------------------------------------------|
| **Relevance** | ↓ -14.5 pts（显著下降） | ↑ +10.4 pts（大幅提升） |
| **Justification** | ↓ -9.9 pts | ↑ +7.3 pts |
| **Factuality** | ↓ -5.3 pts | ↑ +4.0 pts |
| **Trust** | ↓ -7.6 pts | ↑ +2.9 pts |

> ✅ **结论**：Agentic 架构**显著提升非思考型/医学微调模型**的解释质量，但**损害了强思考模型**（如 Gemini, Claude）的表现。

#### （2）临床指标预测：工具调用决定成败
| 临床指标 | Thinking (Zero-shot → Agentic) | Non-Thinking (Zero-shot → Agentic) |
|---------|-------------------------------|------------------------------------|
| **ESI (F1)** | 61.0 → 64.6 | 40.7 → **65.4** |
| **qSOFA (F1)** | 63.6 → **100.0** | 48.1 → **100.0** |
| **MAP (MAE↓)** | 11.8 → **0.0** | 9.7 → **0.0** |
| **SI (MAE↓)** | 0.1 → **0.0** | 0.1 → **0.0** |
| **Pain Score (MAE)** | 1.9 → 1.8 | 2.4 → 2.8（恶化） |
| **LOS (MAE)** | 1.5 → 1.9（恶化） | 1.6 → 2.0（恶化） |

> ✅ **结论**：对于**可编码的确定性指标**（如 MAP, qSOFA），agentic + 显式工具调用带来近乎完美的改进；但对于**主观指标**（如 Pain, LOS），agentic 反而导致性能下降。

#### （3）可视化与实用性权衡
- 所有模型在 **Clinical Utility** 上均有提升（感知到更多帮助）。
- 但在 **Chart Comprehensibility** 上表现分化：
  - **MedGemma 和 GPT 5.2**：在提升实用性的同时维持图表清晰度。
  - **Claude 4.5 Opus**：实用性上升但图表可读性下降，表明其生成复杂但不易理解的可视化。
  - **Llama 4 Maverick**：两项均无明显改善。

#### （4）消融与失败模式分析
- **计算开销巨大**：Agentic 流程比 zero-shot 平均增加 **5–14 倍延迟**，消耗 **13–38 倍 token**。
- **失败主因**：`CoderAgent` 中的语法错误（如使用 ≤ 而非 <=）导致反复重试，尤其在 GPT 5.2 上严重。
- **可靠性税**（Reliability Tax）：即使最终输出正确，自我修复机制也极大拖慢系统响应速度。

---

## 4. 关键结论和发现

### 主要发现
1. **Agentic AI 不是通用性能放大器**  
   > “Agentic AI is best understood as an ensemble that benefits specialized, smaller models, rather than as a general-purpose performance multiplier.”  
   对于已具备强大内部推理能力的 thinking models，引入外部结构反而会分散注意力、降低相关性和信任度。

2. **价值在于选择性外部化计算**  
   Agentic 系统的最大优势体现在：
   - 将**确定性数值计算**（如 MAP, qSOFA）交给专用模块；
   - 补偿**小型或医学专用模型**在抽象推理上的不足。

3. **主观任务不适合过度结构化推理**  
   对疼痛评分、住院时长等主观目标，agentic 分解未能带来一致收益，甚至有害，说明这些任务需要更灵活的判断而非机械流程。

4. **可视化需平衡信息丰富性与惯例遵循**  
   医学微调模型（如 MedGemma）能在提供高价值信息的同时保持图表清晰，而通用大模型倾向于生成新颖但难懂的视觉形式。

### 方法的局限性
- **高计算成本**：多轮交互和代码执行带来显著延迟，难以满足实时监控需求。
- **对 LLM 输出稳定性敏感**：语法错误、格式偏差会导致整个流程中断或低效重试。
- **依赖高质量预处理数据**：实验基于严格清洗的数据集，在真实世界噪声环境下表现未知。
- **评估规模有限**：尽管专家质量高，但样本量较小（109 次评估），主要用于定性洞察而非统计推断。

### 未来工作方向
- **动态代理调度策略**（Dynamic Agent Selection）：根据任务类型和模型能力动态启用或绕过某些代理。
- **紧耦合人机协作机制**：集成医生实时反馈以优化代理行为。
- **前瞻性临床验证**：在真实临床环境中评估系统对医生决策和患者结局的实际影响。
- **轻量化与边缘部署**：探索如何压缩多智能体流程以适应资源受限设备（如 ICU 床旁终端）。

</details>

---

### 16. [Robust Unscented Kalman Filtering via Recurrent Meta-Adaptation of Sigma-Point Weights](https://arxiv.org/abs/2603.04360)

**Authors**: Kenan Majewski, Micha{\l} Modzelewski, Marcin \.Zugaj, Piotr Lichota  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.04360v1  

#### Abstract
The Unscented Kalman Filter (UKF) is a ubiquitous tool for nonlinear state estimation; however, its performance is limited by the static parameterization of the Unscented Transform (UT). Conventional weighting schemes, governed by fixed scaling parameters, assume implicit Gaussianity and fail to ada...

---

### 17. [BeamPERL: Parameter-Efficient RL with Verifiable Rewards Specializes Compact LLMs for Structured Beam Mechanics Reasoning](https://arxiv.org/abs/2603.04124)

**Authors**: Tarjei Paule Hage, Markus J. Buehler  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.04124v1  

#### Abstract
Can reinforcement learning with hard, verifiable rewards teach a compact language model to reason about physics, or does it primarily learn to pattern-match toward correct answers? We study this question by training a 1.5B-parameter reasoning model on beam statics, a classic engineering problem, usi...

---

### 18. [Training-free Dropout Sampling for Semantic Token Acceptance in Speculative Decoding](https://arxiv.org/abs/2603.03333)

**Authors**: Jeongtae Lee, Minjung Jo, Hyunjoon Jeong, Gunho Park, Sunghyeon Woo, Joonghoon Kim, Se Jung Kwon, Dongsoo Lee  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.03333v1  

#### Abstract
Speculative decoding accelerates large language model inference by proposing tokens with a lightweight draft model and selectively accepting them using a target model. This work introduces DropMatch, a novel approach that matches draft tokens to the predictive distribution of the target model via Mo...

---

### 19. [FINEST: Improving LLM Responses to Sensitive Topics Through Fine-Grained Evaluation](https://arxiv.org/abs/2603.04123)

**Authors**: Juhyun Oh, Nayeon Lee, Chani Jung, Jiho Jin, Junho Myung, Jongwon Lee, Taeui Song, Alice Oh  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.04123v1  

#### Abstract
Large Language Models (LLMs) often generate overly cautious and vague responses on sensitive topics, sacrificing helpfulness for safety. Existing evaluation frameworks lack systematic methods to identify and address specific weaknesses in responses to sensitive topics, making it difficult to improve...

---

### 20. [Accelerating OpenPangu Inference on NPU via Speculative Decoding](https://arxiv.org/abs/2603.03383)

**Authors**: Yuntao Dai, Jing Wu, Hang Gu, Teng Wang  
**Category**: cs.DC  
**Published**: 2026-03-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.03383v1  

#### Abstract
To mitigate the Memory Wall bottleneck encountered by Large Language Models (LLMs) during inference on \textbf{NPU} hardware, and addressing the scarcity of native support for mainstream speculative decoding algorithms on domestic infrastructure, this study presents an end-to-end speculative inferen...

---

### 21. [Hybrid Belief Reinforcement Learning for Efficient Coordinated Spatial Exploration](https://arxiv.org/abs/2603.03595)

**Authors**: Danish Rizvi, David Boyle  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.03595v1  

#### Abstract
Coordinating multiple autonomous agents to explore and serve spatially heterogeneous demand requires jointly learning unknown spatial patterns and planning trajectories that maximize task performance. Pure model-based approaches provide structured uncertainty estimates but lack adaptive policy learn...

---

### 22. [Adaptive Sensing of Continuous Physical Systems for Machine Learning](https://arxiv.org/abs/2603.03650)

**Authors**: Felix K\"oster, Atsushi Uchida  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.03650v1  

#### Abstract
Physical dynamical systems can be viewed as natural information processors: their systems preserve, transform, and disperse input information. This perspective motivates learning not only from data generated by such systems, but also how to measure them in a way that extracts the most useful informa...

---

### 23. [Asymmetric Goal Drift in Coding Agents Under Value Conflict](https://arxiv.org/abs/2603.03456)

**Authors**: Magnus Saebo, Spencer Gibson, Tyler Crosse, Achyutha Menon, Eyon Jang, Diogo Cruz  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03456v1  

#### Abstract
Agentic coding agents are increasingly deployed autonomously, at scale, and over long-context horizons. Throughout an agent's lifetime, it must navigate tensions between explicit instructions, learned values, and environmental pressures, often in contexts unseen during training. Prior work on model ...

---

### 24. [RAGNav: A Retrieval-Augmented Topological Reasoning Framework for Multi-Goal Visual-Language Navigation](https://arxiv.org/abs/2603.03745)

**Authors**: Ling Luo, Qiangian Bai  
**Category**: cs.AI  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03745v1  

#### Abstract
Vision-Language Navigation (VLN) is evolving from single-point pathfinding toward the more challenging Multi-Goal VLN. This task requires agents to accurately identify multiple entities while collaboratively reasoning over their spatial-physical constraints and sequential execution order. However, g...

---

### 25. [Benchmarking Motivational Interviewing Competence of Large Language Models](https://arxiv.org/abs/2603.03846)

**Authors**: Aishwariya Jha, Prakrithi Shivaprakash, Lekhansh Shukla, Animesh Mukherjee, Prabhat Chand, Pratima Murthy  
**Category**: cs.CL  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03846v1  

#### Abstract
Motivational interviewing (MI) promotes behavioural change in substance use disorders. Its fidelity is measured using the Motivational Interviewing Treatment Integrity (MITI) framework. While large language models (LLMs) can potentially generate MI-consistent therapist responses, their competence us...

---

### 26. [Exploring Challenges in Developing Edge-Cloud-Native Applications Across Multiple Business Domains](https://arxiv.org/abs/2603.03738)

**Authors**: Pawissanutt Lertpongrujikorn, Hai Duc Nguyen, Juahn Kwon, Mohsen Amini Salehi  
**Category**: cs.DC  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03738v1  

#### Abstract
As the convergence of cloud computing and advanced networking continues to reshape modern software development, edge-cloud-native paradigms have become essential for enabling scalable, resilient, and agile digital services that depend on high-performance, low-latency, and reliable communication. Thi...

---

### 27. [Towards Improved Sentence Representations using Token Graphs](https://arxiv.org/abs/2603.03389)

**Authors**: Krishna Sri Ipsit Mantri, Carola-Bibiane Sch\"onlieb, Zorah L\"ahner, Moshe Eliasof  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03389v1  

#### Abstract
Obtaining a single-vector representation from a Large Language Model's (LLM) token-level outputs is a critical step for nearly all sentence-level tasks. However, standard pooling methods like mean or max aggregation treat tokens as an independent set, discarding the rich relational structure capture...

---

### 28. [Q-Measure-Learning for Continuous State RL: Efficient Implementation and Convergence](https://arxiv.org/abs/2603.03523)

**Authors**: Shengbo Wang  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.03523v1  

#### Abstract
We study reinforcement learning in infinite-horizon discounted Markov decision processes with continuous state spaces, where data are generated online from a single trajectory under a Markovian behavior policy. To avoid maintaining an infinite-dimensional, function-valued estimate, we propose the no...

---

### 29. [FedCova: Robust Federated Covariance Learning Against Noisy Labels](https://arxiv.org/abs/2603.04062)

**Authors**: Xiangyu Zhong, Xiaojun Yuan, Ying-Jun Angela Zhang  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.04062v1  

#### Abstract
Noisy labels in distributed datasets induce severe local overfitting and consequently compromise the global model in federated learning (FL). Most existing solutions rely on selecting clean devices or aligning with public clean datasets, rather than endowing the model itself with robustness. In this...

---

### 30. [LUMINA: Foundation Models for Topology Transferable ACOPF](https://arxiv.org/abs/2603.04300)

**Authors**: Yijiang Li, Zeeshan Memon, Hongwei Jin, Stefano Fenu, Keunju Song, Sunash B Sharma, Parfait Gasana, Hongseok Kim, Liang Zhao, Kibaek Kim  
**Category**: cs.LG  
**Published**: 2026-03-05  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.04300v1  

#### Abstract
Foundation models in general promise to accelerate scientific computation by learning reusable representations across problem instances, yet constrained scientific systems, where predictions must satisfy physical laws and safety limits, pose unique challenges that stress conventional training paradi...

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
