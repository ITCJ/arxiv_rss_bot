# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-03 10:31:05 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [KForge: LLM-Driven Cross-Platform Kernel Generation for AI Accelerators](https://arxiv.org/abs/2606.02963)

**Authors**: Taras Sereda, Burak Bartan, Ankita Nayak, Tom St. John, Natalie Serrino, Zain Asgar  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 15.5  
**Type**: new  
**ArXiv ID**: 2606.02963v1  

#### Abstract
Production inference increasingly targets a heterogeneous mix of accelerators. Agentic pipelines interleave reasoning, tool calls, and multi-agent coordination, each with distinct compute and memory profiles. For optimal efficiency, each stage should run on the accelerator best suited to it. This cr...

---

### 2. [StepFinder: A Temporal Semantic Framework for Failure Attribution in Multi-Agent Systems](https://arxiv.org/abs/2606.03467)

**Authors**: Taiyu Zhu, Yifan Wu, Weilin Jin, Ying Li, Gang Huang  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2606.03467v1  

#### Abstract
LLM-based multi-agent systems exhibit remarkable collaborative capabilities in complex multi-step tasks. However, these systems are highly sensitive to single-step execution errors that can propagate through agent interactions and lead to cascading failures. To understand the causes of failure and i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# StepFinder: A Temporal Semantic Framework for Failure Attribution in Multi-Agent Systems —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题  
**Multi-Agent Systems (MAS)** 在复杂多步任务中表现出色，但由于 agent 间的交互高度耦合，单步执行错误容易通过协作链传播，引发**级联失败（cascading failures）**。因此，**failure attribution**（失败归因）——即自动识别导致系统失败的“根因步骤”（root cause step）——成为提升系统可靠性的关键。

然而，现有方法存在以下问题：
- **LLM-based 方法**：依赖大模型对完整执行轨迹进行推理，计算开销高、延迟大，且易受冗余日志和噪声干扰，难以精确定位根因。
- **传统序列模型**：缺乏对 agent 身份信息和跨步依赖关系的建模能力，定位精度有限。

---

### 🚀 提出的新方法与创新思路  

作者提出 **StepFinder**，一个轻量级、高效的 failure attribution 框架，其核心思想是将 failure attribution 重构为一个**结构化的时序语义建模问题**，而非直接依赖 LLM 进行推理。

#### 主要创新点包括：

1. **LLM 仅用于特征编码阶段**  
   - 使用 **Qwen3 Embedding** 将每一步的执行内容（content）和 agent 身份（agent identity）编码为稠密向量，构建**时序语义序列（temporal semantic sequence）**。
   - 避免在推理阶段调用 LLM 进行文本生成，显著降低计算成本。

2. **双模块混合架构设计**  
   - **Temporal Feature Extraction**：采用 **BiLSTM** 捕捉执行轨迹中的双向时序依赖，建模状态演化。
   - **Agent-Aware Step Interaction**：引入基于 agent 身份感知的注意力机制（agent-aware bias + agent-aware gating），显式建模跨步因果关系，增强对 agent 行为链的理解。

3. **精细化的异常评分机制**  
   - 引入 **multi-scale differencing** 捕捉不同时间尺度下的状态突变（如局部跳跃或长期累积偏差）。
   - 加入 **position bias**，赋予早期步骤更高优先级，符合“早期决策决定成败”的直觉。
   - 最终每个步骤获得一个综合的 **error score**，用于排序并定位根因。

4. **联合损失函数优化**  
   - 主损失：**Cross-Entropy Loss**，监督根因步骤分类。
   - 辅助损失：**Temporal Consistency Loss**，通过预测下一步隐藏状态实现自监督学习，增强模型对正常执行流的建模能力。

---

### ⚖️ 相比现有方法的优势  

| 维度 | StepFinder | LLM-based 方法 |
|------|-----------|----------------|
| 推理效率 | 极高（无文本生成） | 极低（需多次 LLM 调用） |
| 定位精度 | 更高（尤其在复杂轨迹） | 易受上下文噪声影响 |
| 可扩展性 | 支持长轨迹、大规模分析 | 成本随长度指数增长 |
| 成本 | 低（仅一次 embedding + 轻量模型） | 高（API 调用或本地部署大模型） |

> ✅ **核心优势总结**：StepFinder 实现了**高精度 + 高效率 + 低成本**的 failure attribution，适用于实际部署场景。

---

## 2. 核心实验方法和设置

### 📚 数据集  
使用 **Who&When benchmark**，该基准正式定义了 MAS 中的自动化 failure attribution 任务，并提供标准化失败轨迹数据集。包含两个子集：

| 子集 | 类型 | 数量（测试集） | 特点 |
|------|------|----------------|------|
| **Algorithm-Generated (Alg)** | 自动系统生成 | 126 条轨迹 | 结构较规则，逻辑清晰 |
| **Hand-Crafted (HC)** | 人工构造 | 58 条轨迹 | 更复杂、交互更密集、更具挑战性 |

训练集通过 LLM 重生成策略扩充，确保多样性与标注一致性。

---

### 🎯 评估指标  

#### （1）**Attribution Precision（归因精度）**
- **Accuracy**：预测得分最高的步骤是否等于真实根因步骤。
- **Tolerance Accuracy**：允许预测结果在真实步骤 ±δ 步范围内视为正确（δ ∈ {1,…,5}）。

#### （2）**Ranking Quality（排序质量）**
- **Acc@K**：真实根因出现在前 K 个候选中的比例。
- **MRR@3**（Mean Reciprocal Rank@3）：衡量真实根因在前 3 名中的平均排名倒数，越接近 1 越好。

---

### 🔁 基线方法对比  

分为三类基线：

| 类别 | 方法 | 描述 |
|------|------|------|
| **Random Attribution** | Random | 随机选择步骤作为根因，下界参考 |
| **LLM-based Attribution** | All-at-Once<br>Step-by-Step<br>Binary Search | 使用 GPT-4o 对轨迹进行零样本推理，分别采用一次性、逐次检查、二分搜索策略 |
| **Sequential Models** | BiGRU<br>TCN<br>Transformer | 经典序列建模架构，用于验证专用设计的有效性 |

> 所有基线均在相同训练流程下复现，保证公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2 和 Figure 3–4）

#### （1）**Accuracy 对比**

| 方法 | Alg (%) | HC (%) |
|------|--------|-------|
| Best LLM-based (Step-by-Step-G) | 20.11 | 6.90 |
| Transformer | 24.87 | 9.19 |
| **StepFinder (Ours)** | **29.63** | **22.99** |

- 在 **Alg** 上比最强 LLM 方法提升 **~9.5%**；
- 在 **HC** 上提升超过 **200%**，显示其在复杂场景下的强大鲁棒性。

#### （2）**Tolerance Accuracy（见 Figure 3）**

- StepFinder 在所有容忍阈值下均领先，尤其在 δ=1 时优势明显（Alg 上高出约 15%），说明其预测高度集中于真实根因附近。

#### （3）**Ranking Quality**

| 指标 | Alg (StepFinder) | HC (StepFinder) |
|------|------------------|-----------------|
| **Acc@1** | 29.63% | 22.99% |
| **Acc@3** | **60.31%** | **30.46%** |
| **MRR@3** | ~50% | ~30%（远超其他方法） |

> 💡 **重要发现**：即使无法精确命中（Acc@1），StepFinder 也能将真实根因保留在前 3 名以内，极大缩小人工审计范围。

#### （4）**消融实验（Ablation Study, Table 5）**

移除各组件后性能下降情况（以 Accuracy 衡量）：

| 移除组件 | Alg ↓ | HC ↓ | 影响说明 |
|---------|------|------|----------|
| w/o TFE（Temporal Feature Extraction） | -4.5% | -10.35% | 时序建模至关重要，尤其对长轨迹 |
| w/o ASI（Agent-Aware Step Interaction） | -2.12% | -4.02% | 显式建模跨步依赖有效 |
| w/o AI（Agent Identity） | -1.59% | -3.45% | agent 身份信息有助于行为一致性建模 |
| w/o MsDiff（Multi-scale Differencing） | -0.26% | -3.45% | 对复杂交互模式尤为重要 |
| w/o PB（Position Bias） | -2.65% | -2.88% | 早期步骤偏好提升整体定位稳定性 |
| w/o TCLoss（Temporal Consistency Loss） | -4.5% | -6.32% | 自监督任务显著提升泛化能力 |

> ✅ 所有模块均有贡献，组合效果最优。

---

### ⚙️ 效率分析（Table 6）

| 方法 | 输入 Token 数（Alg） | 输出 Token 数 | 推理时间（秒/样本） |
|------|--------------------|--------------|---------------------|
| All-at-Once | 3,334 | 138 | 2.92 |
| Step-by-Step | 15,673 | 1,626 | 26.18 |
| **StepFinder** | 3,289 | **0** | **0.61** |

- **推理速度提升近 5 倍**（相比最快 LLM 方法）；
- **减少 79% 推理时间**；
- **无文本生成开销（output tokens = 0）**。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **Failure attribution 应被视为时序语义建模问题**，而非单纯的自然语言推理任务。StepFinder 通过将 LLM 限制在编码阶段，成功解耦语义提取与诊断逻辑，实现了高效精准归因。

2. **Agent identity 是关键结构先验**。引入 agent-aware attention 和 gating 机制，能有效捕捉 agent 行为链中的深层依赖关系。

3. **多尺度差异 + 位置偏置显著提升定位精度**。特别是对于 HC 这类复杂轨迹，multi-scale differencing 能检测到细微但关键的状态漂移。

4. **自监督辅助任务增强训练稳定性**。Temporal Consistency Loss 引导模型学习正常的执行流模式，从而更敏感地识别偏离点。

5. **StepFinder 在精度与效率之间取得最佳平衡**，适合大规模 MAS 日志的实时诊断与运维支持。

---

### ⚠️ 局限性

1. **依赖高质量 embedding 模型**：若 content 或 agent 编码不准确，会影响后续建模效果。
2. **假设 agent 轮流执行**：当前框架基于 round-robin 执行顺序，对异步或多 agent 并发场景支持有限。
3. **未考虑外部环境反馈**：如工具调用返回值、用户干预等动态信号尚未纳入建模。

---

### 🔮 未来工作方向

1. **扩展至异步 MAS 架构**：支持并发执行、事件驱动等更复杂的协作模式。
2. **引入因果推理机制**：结合 CDC-MAS 等因果框架，进一步区分相关性与因果性。
3. **在线增量学习能力**：使模型能够持续从新失败案例中学习，适应 evolving agent behaviors。
4. **可视化诊断接口开发**：将 error score 与执行轨迹结合，提供可解释的调试视图。

---

> 🔗 **开源信息**：  
> - 代码地址：[https://github.com/taiyu-zhu/StepFinder](https://github.com/taiyu-zhu/StepFinder)  
> - 数据与资源 DOI: [10.5281/zenodo.20432323](https://doi.org/10.5281/zenodo.20432323)

</details>

---

### 3. [SIGMA: A Versatile Streaming Graph Partitioner for Vertex- and Edge-Balanced Distributed GNN Training](https://arxiv.org/abs/2606.03519)

**Authors**: Barbara Hoffmann, Shai Dorian Peretz, Adil Chhabra, Ahmet Kadir Yalcinkaya, Ruben Mayer, Christian Schulz  
**Category**: cs.DC  
**Published**: 2026-06-03  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.03519v1  

#### Abstract
Distributed Graph Neural Network (GNN) training depends critically on how the underlying graph is partitioned across compute resources. Existing graph partitioners focus either on vertex partitioning or edge partitioning and typically optimize only a single communication objective (edge cut or verte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SIGMA: A Versatile Streaming Graph Partitioner for Vertex- and Edge-Balanced Distributed GNN Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的图划分器（graph partitioner）通常只支持**顶点划分**（vertex partitioning）或**边划分**（edge partitioning），且仅优化单一通信目标（如 edge cut 或 replication factor），并施加单一平衡约束（vertex balance 或 edge balance）。然而，分布式 GNN 训练同时依赖于**顶点中心计算**（聚合）和**边中心通信**（消息传递），因此需要同时考虑：
- 通信开销（由 edge cut 或 vertex replication 决定）
- 计算负载均衡（vertex 和 edge 的分布）
- 内存消耗（受 vertex 复制影响）

传统方法无法兼顾这些多维目标，导致训练效率低下。

### 提出的新方法：SIGMA
作者提出 **SIGMA**（Streaming Integrated Graph Partitioning with Multi-objective Awareness），一个**统一的、支持多目标、多约束的流式图划分框架**，其核心创新包括：

- ✅ **统一框架支持两种划分范式**：
  - 支持 **vertex-streaming**（分配顶点以最小化 edge cut）
  - 支持 **edge-streaming**（分配边以最小化 vertex replication）
  - 可在同一个算法框架下灵活切换，无需为不同系统重新设计划分器。

- ✅ **多目标联合优化**：
  - 同时优化 **edge cut / replication factor**
  - 同时保证 **vertex balance** 和 **edge balance**
  - 引入动态容量缩放机制，在多约束条件下避免早期过载。

- ✅ **基于聚类的预处理阶段**：
  - 利用 **CluStRE** 进行流式图聚类，捕捉全局社区结构。
  - 将聚类结果通过 **makespan 调度** 映射到分区块，实现“预分配”。
  - 在保持流式高效性的同时提升划分质量。

- ✅ **开源实现**：
  - 已将 SIGMA 开源：[https://github.com/bab-si/SIGMA](https://github.com/bab-si/SIGMA)

### 相比现有方法的优势
| 维度 | SIGMA 的优势 |
|------|-------------|
| **通用性** | 单一框架支持 vertex 和 edge 划分，减少工程复杂度 |
| **多目标能力** | 首个支持 vertex 和 edge 双重平衡的流式划分器 |
| **可扩展性** | 流式架构适用于大规模图，内存占用低 |
| **划分质量** | 在 replication factor、balance 指标上优于主流流式划分器，接近高质量 in-memory 方法（如 METIS、HEP） |

---

## 2. 核心实验方法和设置

### 数据集
使用了六个广泛使用的基准图数据集，覆盖多种领域和规模：

| 图 | 类型 | #顶点 | #边 |
|-----|--------|---------|--------|
| amazon computers | 电商 | 13.7k | 491.7k |
| flickr | 社交网络 | 89.2k | 899.7k |
| twitch | 社交网络 | 168.1k | 6.7M |
| ogbn-arxiv | 引用网络 | 169.3k | 1.2M |
| reddit | 社交网络 | 233.0k | 114.6M |
| ogbn-products | 商品共购图 | 2.4M | 61.9M |

### 实验设置
- **硬件环境**：两台服务器，每台含 48 核 CPU、768GB RAM、2×NVIDIA L40S GPU（48GB 显存）
- **分区数 k**：测试多个 k 值（如 4, 8, 16, 32）
- **GNN 模型**：两层 GraphSAGE，GCN 聚合器，隐藏维度 16
- **训练配置**：
  - DistDGL：mini-batch（batch size=1024，fanout=[25,25]）
  - DistGNN：full-batch 训练（仅用于中小图）

### 评估指标
| 类别 | 指标 |
|------|------|
| **通信成本** | edge-cut ratio（vertex partitioning）、replication factor（edge partitioning） |
| **负载均衡** | vertex balance、edge balance（越接近 1.0 越好） |
| **系统效率** | partitioning time、mean training time per epoch |
| **内存消耗** | peak GPU memory（DistDGL）、peak RAM（DistGNN） |

### 基线方法对比
#### 边划分（Edge Partitioning）基线：
- 流式：`Random`, `DBH`, `HDRF`, `HeiStreamE`, `2PS`
- 内存型：`FSM`, `HEP`

#### 顶点划分（Vertex Partitioning）基线：
- 流式：`Random`, `LDG`, `FENNEL`, `Cuttana`, `HeiStream`, `BuffCut`
- 内存型：`METIS`, `KaHIP`

SIGMA 在两种模式下均参与比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）边划分性能（DistGNN 上）

| 指标 | 结果 |
|------|------|
| **Replication Factor** | SIGMA 是所有**流式划分器中最低的**。例如在 `amazon computers`（k=32）上：<br>- SIGMA: **2.80**<br>- HDRF: 13.8 (-80%)<br>- Random: 16.0 (-82%)<br>- HeiStreamE: 3.74 (-25%) |
| **Vertex Balance** | 显著优于 HEP 和 FSM（in-memory 方法），最大不平衡从 2.09 降至 1.53（↓51%） |
| **Edge Balance** | 控制在 1.10 以内（设定阈值），优于多数流式方法 |
| **Training Time / Epoch** | 在所有数据集上均为**最快或次快的流式方法**，比 HDRF 快 62%，比 HEP 快 25% |
| **Memory (RAM)** | 在 flickr 和 twitch 上达到最低峰值内存（22.6GB / 42.7GB） |

> 💡 **说明**：低 replication factor → 更少的 vertex 复制 → 更低通信与内存开销。

#### （2）顶点划分性能（DistDGL 上）

| 指标 | 结果 |
|------|------|
| **Edge-Cut Ratio** | 属中等水平，不如 FENNEL 或 METIS，但优于 Random、LDG 等。例如在 flickr（k=32）上：<br>- SIGMA: **0.642**<br>- FENNEL: 0.658<br>- METIS: 0.612 |
| **Vertex & Edge Balance** | **表现极佳**：<br>- vertex balance: 1.00–1.09（接近理想值）<br>- edge balance: 1.01–1.18<br>显著优于 FENNEL（vertex balance 最高达 2.40） |
| **Training Time / Epoch** | 与 KaHIP/METIS 接近，优于大多数流式方法（如 LDG、FENNEL） |
| **GPU Memory** | 稳定且较低，未出现明显峰值 |
| **Partitioning Time** | 略高于 HeiStream/BuffCut，但远低于 KaHIP |

> ⚠️ 在 `ogbn-products` 上表现稍弱，因该图在 k=2 下多数划分器已自然平衡，此时 edge-cut 成主导因素。

### 消融实验（隐含分析）
虽然未显式列出消融实验，但从设计可推断：
- **聚类预处理的作用**：显著提升划分质量，尤其在社区结构明显的图上（如 social graphs）
- **多目标评分函数的设计**：引入 replication-aware term 有效降低 vertex 复制
- **动态容量控制**：防止某一维度过早饱和，保障最终 balance

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **统一的流式划分框架是可行且高效的**：
   - SIGMA 成功在一个框架内支持 vertex 和 edge 划分，适应不同 GNN 系统需求。

2. ✅ **多目标、多约束优化能显著提升训练效率**：
   - 同时优化 communication、vertex balance、edge balance 可避免资源瓶颈（straggler worker、OOM）。

3. ✅ **流式划分可以媲美 in-memory 方法的质量**：
   - SIGMA 在 replication factor 和 balance 上接近甚至超越 HEP、FSM 等混合方法。
   - 分区时间远低于 METIS/KaHIP，适合大规模图。

4. ✅ **划分质量直接转化为训练收益**：
   - 更好的 balance → 更均匀的负载 → 更高的硬件利用率
   - 更低的 replication → 更少通信 → 更快训练速度 + 更低内存

### 方法的局限性
- ❗ **edge-cut ratio 不及顶级 in-memory 方法**（如 METIS）：因流式无法进行全局优化。
- ❗ 在高度规则或稀疏图上优势可能减弱。
- ❗ 当前版本未支持有向图或多权重顶点/边。

### 未来工作方向
- 扩展至 **异构图**（Heterogeneous Graphs）和 **动态图流**（Dynamic Streams）
- 支持 **自适应参数调整**（如自动调节 γ, τ）
- 集成进更多 GNN 系统作为默认 partitioner
- 探索 **学习型划分策略**（learned partitioning）与 SIGMA 框架结合

---

> 🔚 **总结一句话**：  
> SIGMA 是首个支持 **vertex/edge 双范式、多目标、多约束** 的流式图划分器，实现了**高质量、高效率、高通用性**的统一，在真实 GNN 训练中显著优于主流流式方法，并逼近 in-memory 方法性能，为分布式 GNN 系统提供了强大而实用的图划分解决方案。

</details>

---

### 4. [Qift: Shift-Friendly No-Zero W2 Post-Training Quantization for Rotated W2A4/KV4 LLM Inference](https://arxiv.org/abs/2606.02823)

**Authors**: Chi-Wei Huang, Chia-Chi Tsai  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.02823v1  

#### Abstract
Two-bit weight quantization is attractive for memory-efficient LLM inference, but the standard W2 level set {-2,-1,0,+1} often collapses under aggressive W2A4/KV4 settings. We study the scalar level-set geometry of two-bit weights in a Hadamard-rotated quantization pipeline. Conventional asymmetric ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Qift: Shift-Friendly No-Zero W2 Post-Training Quantization for Rotated W2A4/KV4 LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
在 **W2A4/KV4**（即权重2-bit、激活4-bit、KV缓存4-bit）的极端低比特量化场景中，传统的对称整数量化器使用的标准 W2 等级集 `{-2, -1, 0, +1}` 在旋转后权重分布下表现极差，导致模型崩溃（如 LLaMA-3.1-8B 的 PPL 超过3000）。  
这表明 W2A4 失败不仅是**位宽不足**的问题，更是**标量重建等级集设计不当**的问题。

### ✅ 提出的新方法或新思路
提出 **Qift** —— 一种无需训练、无零点、固定等级的 W2 量化方案，其核心是重新设计 W2 的标量重建等级集（scalar reconstruction level set），以匹配 Hadamard 旋转后的权重统计特性。

- **核心假设**：Hadamard 旋转使权重标准化后更接近**零中心、高斯状分布**（Gaussian-like），因此最优的四等级量化应有两个内层质心围绕零点，两个外层质心处理尾部，而不是将一个等级浪费在精确的零上。
- **提出两种新等级集**：
  - **Qift-MNZ**（Mirror No-Zero）：`{±0.5, ±1.5}` 或等价于 `{±1, ±3}`（半尺度重参数化）
  - **Qift-PoT-MNZ**（Power-of-Two variant）：`{±1, ±4}`，支持硬件友好的 sign-and-shift 运算

### ✅ 相比现有方法的优势
| 特性 | Qift | 其他方法（如 RCP、AQLM） |
|------|------|------------------------|
| 是否需要 QAT | ❌ 否 | ✅ 是（多数） |
| 是否学习 codebook | ❌ 否 | ✅ 是 |
| 是否 per-group grid | ❌ 否 | ✅ 是 |
| 是否有 zero-point | ❌ 否 | ✅ 是（如 asymmetric W2） |
| 部署友好性 | ✅ 极高（小整数解码） | ❌ 较低（查表/向量码本） |

> ✅ **优势总结**：Qift 是一个**轻量级、模块化、可插拔**的改进，仅替换标准 W2 等级集，即可显著提升性能，且完全兼容现有的 PTQ 流程（如 GPTQ/GPTAQ）、无需额外元数据或复杂解码逻辑。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **校准数据**：WikiText-2，128 个样本，序列长度 2048
- **评估任务**：
  - **语言建模**：WikiText-2 Perplexity (PPL)
  - **下游任务**：ARC-Challenge, ARC-Easy, HellaSwag, PIQA, WinoGrande（平均准确率）

### ⚙️ 实验设置
- **模型**：LLaMA-2-7B 和 LLaMA-3.1-8B
- **量化配置**：
  - 权重：W2（除非混合精度）
  - 激活 & KV Cache：A4/KV4
  - 使用 **Hadamard 旋转** + **per-channel scaling**
  - 可选启用 **GPTQ** 或 **GPTAQ** 补偿
- **混合精度策略（L-layer）**：
  - 将最敏感的 L 层升级为 W4A4，其余保持 W2A4
  - 在 L=16 时达到平均 3-bit（与 W3A4 对齐）

### 📊 评估指标
- **主指标**：
  - WikiText-2 Perplexity（越低越好）
  - 下游任务平均准确率（越高越好）
- **辅助分析**：
  - GPTQ 累积残差
  - RTN 重建误差桶分析
  - 内/外质心比率敏感性扫描

### 🔁 基线方法对比
| 方法 | 类型 | 是否训练 | 是否有 zero-point |
|------|------|----------|------------------|
| SYM-INT (`{-2,-1,0,+1}`) | 标准对称 W2 | ❌ | ❌ |
| W-ASYM | 非对称 W2 | ❌ | ✅ |
| Lloyd-Max | 高斯最优标量量化 | ❌ | ❌ |
| NF2 | NormalFloat 动机 | ❌ | ❌ |
| RCP | 学习非均匀 W2（QAT） | ✅ | ✅ |
| Qift-MNZ / Qift-PoT-MNZ | 本文提出 | ❌ | ❌ |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 4）

#### LLaMA-2-7B 结果
| 设置 | 方法 | PPL | 下游平均准确率 |
|------|------|-----|----------------|
| FP16 | – | 5.471 | 0.6866 |
| Pure W2A4 | SYM-INT | 53.849 | 0.4211 |
| Pure W2A4 | **Qift-MNZ** | **9.294** | **0.4794** |
| L=16 Mixed | **Qift-MNZ** | **7.122** | **0.6157** |
| W3A4 (参考) | SYM-INT | 6.897 | 0.6200 |

> 💡 **说明**：Qift 在纯 W2A4 下将 PPL 从 53.8 降至 9.3；在 L=16 混合精度下逼近 W3A4 性能。

#### LLaMA-3.1-8B 结果
| 设置 | 方法 | PPL | 下游平均准确率 |
|------|------|-----|----------------|
| FP16 | – | 6.277 | 0.7435 |
| Pure W2A4 | SYM-INT | >3000 | 0.3683 |
| Pure W2A4 | **Qift-MNZ** | **19.515** | **0.4064** |
| L=16 Mixed | **Qift-MNZ** | **11.936** | **0.5609** |
| W3A4 (参考) | SYM-INT | 10.954 | 0.5972 |

> 💡 **说明**：原始 W2A4 完全崩溃，而 Qift 成功恢复至可用水平，并在混合精度下大幅缩小与 W3A4 的差距。

### 🔍 与基线方法的对比结果
- **相比标准 SYM-INT**：
  - PPL 改善高达 **>99%**（如 LLaMA-3.1-8B 从 >3000 → 19.5）
  - 下游准确率提升约 **+4–5个百分点**
- **相比非对称 W2 (W-ASYM)**：
  - 继续带来显著增益（如 LLaMA-2-7B PPL 从 11.577 → 9.294）
  - 证明“去零”本身不够，**内/外质心比例**才是关键
- **相比 Lloyd/NF2**：
  - 接近最优标量量化性能，但使用更简单的整数等级，更适合部署

### 🔬 消融实验结果

#### ✅ 内/外质心比率分析（Figure 8, Table 14）
- 最优比率范围：**r ∈ [0.25, 0.33]**（内层 / 外层）
- Qift-MNZ: r = 1/3 ≈ 0.333 ✅
- Qift-PoT-MNZ: r = 0.25 ✅
- FAR-MNZ (`{±1,±2}`): r = 0.5 ❌（性能差）

> 📌 **结论**：去除零点不足以保证性能提升，必须搭配合适的几何结构。

#### ✅ GPTQ 残差分析（Table 15）
| 方法 | PPL | 残差比率（相对 SYM-INT） |
|------|-----|------------------------|
| SYM-INT | 7.825 | 1.000 |
| Qift-MNZ | 7.318 | 0.778 ↓
| Lloyd-Max | 7.278 | 0.794 ↓
| FAR-MNZ | 7.982 | 1.020 ↑

> 📌 **机制解释**：Qift 减少了 GPTQ 量化过程中的累积误差，验证了其重建质量更高。

#### ✅ RTN 重建诊断（Table 13）
- Qift-MNZ 将总平方重建误差降低 **20.7%**
- 标准 W2 存在严重不平衡：零桶承载 40% 权重但集中大量误差
- Qift 分布更均衡，误差分散更合理

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **W2A4 失败的根本原因在于等级集设计不当**，而非单纯位宽限制。
2. **Hadamard 旋转后权重近似零中心、高斯状分布**，应采用 **no-zero mid-rise 结构** 而非 mid-tread。
3. **Qift 的 no-zero 固定等级集（如 `{±0.5, ±1.5}` 或 `{±1, ±4}`）能稳定提升 PPL 和下游任务表现**，且无需任何训练或额外开销。
4. **性能提升的关键是内/外质心比率（r ≈ 0.25–0.33）**，而非简单“去零”。
5. **在 L=16 混合精度下，Qift 显著缩小了与 W3A4 的差距**，说明其在实际部署中有巨大潜力。

### ⚠️ 方法的局限性
- 当前仅适用于 **已旋转的 LLM 量化管道**（如 QuaRot），不解决未旋转情况下的 outlier 问题。
- 固定等级集虽部署友好，但在某些极端分布下可能不如自适应方法灵活。
- 未探索更多硬件定制优化（如专用指令集加速 `{±1,±4}`）。

### 🔮 未来工作方向
- 将 Qift 思想扩展到 **其他低比特设置**（如 W1A4）。
- 探索 **动态选择 MNZ/PoT-MNZ** 的轻量策略。
- 与 **group-free 混合精度调度器** 结合，构建端到端高效推理系统。
- 在更多架构（如 Mistral、Phi）上验证泛化能力。

---

> ✅ **一句话总结**：  
> **Qift 揭示了 W2 量化中“等级集设计”的决定性作用，通过一个极简的 no-zero 固定等级替换（如 `{±0.5, ±1.5}`），在无需训练的前提下，使旋转后的 W2A4/KV4 推理从“崩溃”变为“可用”，并逼近 W3A4 性能，是迈向极致低比特 LLM 推理的重要一步。**

</details>

---

### 5. [Perceive Before Reasoning: A Pre-Reasoning Perception Framework for Efficient and Reliable Proactive Mobile Agents](https://arxiv.org/abs/2606.03236)

**Authors**: Zhijie Ding (HyperAI Team, Xiaomi Corporation, Zhongnan University of Economics and Law), Weinan Hong (HyperAI Team, Xiaomi Corporation, Jilin University), Zicheng Zhu (HyperAI Team, Xiaomi Corporation, The Chinese University of Hong Kong, Shenzhen), Lei Li (HyperAI Team, Xiaomi Corporation), Dezhi Kong (HyperAI Team, Xiaomi Corporation), Hao Wang (HyperAI Team, Xiaomi Corporation), Peng Zhou (HyperAI Team, Xiaomi Corporation), Xuchu Jiang (HyperAI Team, Xiaomi Corporation), Jiaming Xu (HyperAI Team, Xiaomi Corporation)  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2606.03236v1  

#### Abstract
Multimodal large language models (MLLMs) have substantially advanced mobile agents, yet proactive mobile assistance remains challenging because agents must decide \emph{when} to intervene before determining \emph{how} to assist. Existing systems often implement these two decisions within a unified M...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Perceive Before Reasoning: A Pre-Reasoning Perception Framework for Efficient and Reliable Proactive Mobile Agents

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **proactive mobile agents**（主动式移动智能体）通常依赖统一的 **Multimodal Large Language Models (MLLMs)** 同时完成两个任务：
- **When 判断**：决定是否需要干预（推荐）
- **How 推理**：生成具体的推荐内容

这种统一架构存在两大核心问题：
1. **目标冲突（Goal Misalignment）**：  
   - *When 判断* 需要保守、高判别性的决策以避免误触发（false triggers），而 *How 推理* 需要开放、全面的多模态推理能力。两者在同一模型中难以兼顾。
2. **推理效率低下（Inference Inefficiency）**：  
   即使在无需干预的情况下，系统仍会执行完整的、计算成本高昂的 VLM 推理流程，造成资源浪费。

### 提出的新方法
为解决上述问题，作者提出了 **Pre-Reasoning Perception Framework (PRPF)**，其核心思想是 **“感知先行，再行推理”（Perceive Before Reasoning）**。

PRPF 是一个两阶段框架：
1. **第一阶段：Multimodal Proactive Perceptor (MPP)**  
   - 一个轻量级的多模态融合编码器（multimodal fusion encoder）。
   - 负责 **intervention gating**（干预门控）：判断当前状态是否需要推荐。
   - 进行 **context compression**（上下文压缩）：将原始的多模态上下文压缩为 Top-K 个候选场景，大幅缩小后续推理的函数空间。
2. **第二阶段：Proactive Agent Reasoner (PAR)**  
   - 仅当 MPP 判断需要干预时才被激活。
   - 基于 MPP 压缩后的精简证据进行深度、聚焦的 **how-stage reasoning**，生成最终的推荐。

### 相比现有方法的优势
- **解耦设计**：将 “when” 和 “how” 任务在架构层面分离，让每个模块专注于单一目标。
- **高效节能**：通过 MPP 的前置过滤，显著减少了昂贵的 VLM 推理调用次数。
- **可靠性提升**：MPP 作为“感知截止阀”，有效降低了误触发率（FTR）。
- **即插即用**：MPP 可作为一个轻量级前端，与不同的 VLM-based reasoner 结合使用。

---

## 2. 核心实验方法和设置

### 数据集
- **ProactiveMobile** (Kong et al., 2026)：本文的主要评测基准。
  - 包含 **文本**（text）和 **多模态**（multimodal）两种场景。
  - 多模态数据由连续的 **GUI screenshots** 序列构成。
  - 包含用户画像（U）、设备状态（D）、世界信息（W）和交互历史（I）等上下文。
  - 函数池（function pool）包含 63 个复合 API 功能。

### 实验设置和评估指标
- **训练细节**：
  - MPP 使用 BGE-small-zh-v1.5 和 CLIP ViT-B-32 作为文本和图像编码器。
  - PAR 基于 **Qwen3.5-9B** 进行微调。
- **评估指标**（遵循 ProactiveMobile 基准）：
  - **Type-Acc**：预测的函数名序列与真实值完全匹配的准确率。
  - **Success Rate (SR)**：综合正确率，考虑了功能等价性（由 LLM judge 评判）。
  - **False Trigger Rate (FTR)**：在无需推荐的样本中错误触发推荐的比例。
  - **推理效率**：每样本计算量（TFLOPs）、峰值 GPU 内存（GB）和端到端延迟（ms）。

### 基线方法对比
- **闭源大模型**：GPT-5.5, o3, Gemini-3.1-Pro, Claude-Opus-4.7, GLM-4.6V, Kimi-K2.5, MiMo-2.5v。
- **开源模型**：TongUI-7B, Qwen3.5-9B。
- **主动智能体模型**：ProactiveMobile(7B), UI-TARS-7B-DPO+Proactive, Qwen3.5-9B+Proactive（SFT 微调版本）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **ProactiveMobile 测试集** 上，PRPF 取得了显著的性能提升：

| 模型 | Type-Acc (%) | SR (%) | FTR (%) | 推理计算量 (TFLOPs) |
| :--- | :---: | :---: | :---: | :---: |
| **ProactiveMobile (7B)** | 45.25 | 20.82 | 13.76 | 27.97 |
| **PRPF (Ours)** | **55.00** | **41.15** | **7.21** | **8.58** |

- **SR 提升**：从 20.82% 提升至 **41.15%**（+20.33 个百分点）。
- **FTR 降低**：从 13.76% 降至 **7.21%**（降幅超过 47%）。
- **推理效率**：计算量减少 **69.3%**，端到端延迟降低 **60.1%**。

### 与基线方法的对比结果
- PRPF 在所有指标上均优于所有基线模型，尤其是在 **SR** 和 **FTR** 上优势明显。
- 即使与最强的微调基线 **Qwen3.5-9B+Proactive** 相比，PRPF 仍将 FTR 从 13.49% 降至 7.21%，并将 SR 从 34.56% 提升至 41.15%。
- 在文本设置下，PRPF 的 FTR 甚至低至 **1.75%**，证明了其极高的可靠性。

### 消融实验结果
消融研究验证了 PRPF 各组件的有效性：

| 消融配置 | SR (%) | FTR (%) | 说明 |
| :--- | :---: | :---: | :--- |
| **Full PRPF (Ours)** | **41.15** | **7.21** | 完整模型 |
| -w/o MPP | 38.33 | 10.38 | 移除 MPP，直接进入 PAR，性能下降，FTR 显著升高 |
| -w/o PAR | 33.44 | 18.46 | 仅用 Qwen3.5-9B 直接推理，性能大幅下降 |
| -w/o Recommend | 39.29 | 9.74 | 移除 MPP 的触发判断，仅保留压缩，FTR 升高 |
| -w/o Compression | 40.55 | 7.78 | 移除 MPP 的上下文压缩，效率下降 |
| -w/o GRPO | 39.67 | 15.56 | 移除 GRPO 优化，FTR 显著升高，表明 GRPO 对稳定决策至关重要 |

**关键发现**：
- MPP 和 PAR 相互补充，移除任一模块都会导致性能下降。
- MPP 的 **trigger gating** 比 **function compression** 对降低 FTR 更重要。
- PAR 的 **GRPO** 优化对抑制误触发（false triggers）起到了关键作用。

---

## 4. 关键结论和发现

### 主要发现
1. **架构解耦至关重要**：将 “when” 和 “how” 任务分离的 PRPF 框架，能有效解决目标冲突，同时提升推荐的成功率和可靠性。
2. **前置感知显著提效**：轻量级的 MPP 作为前置感知模块，能以极低成本过滤掉大量非干预样本，从而大幅提升整体推理效率。
3. **MPP 具有通用性**：MPP 可作为即插即用（plug-and-play）的前端，与不同后端 reasoner 结合，均能带来性能增益。
4. **多模态理解仍是瓶颈**：尽管 PRPF 整体性能提升显著，但在多模态任务上的绝对成功率（SR=17.19%）仍然较低，表明当前的视觉-语言模型在细粒度界面理解上仍有不足。

### 方法的局限性
1. **领域扩展性受限**：MPP 的训练基于 ProactiveMobile 定义的 14 个高层意图场景。若要扩展到新领域或更大的 API 空间，需要重新训练或微调 MPP。
2. **多模态理解瓶颈**：模型在处理 GUI screenshots 时的表现远不如纯文本，绝对成功率不高，限制了整体性能上限。
3. **隐私风险**：该框架需要访问敏感的 on-device 信号（如截图、用户画像），存在潜在的隐私泄露风险。

### 未来工作方向
- **持续学习（Continual Learning）**：探索无需完全重训即可适应新领域的 MPP 更新机制。
- **更强的多模态理解**：开发更强大的视觉接地（visual grounding）技术、更高分辨率的 GUI 编码器，或进行更大规模的多模态预训练。
- **隐私保护机制**：研究如何在保证功能的前提下，最小化数据收集范围，并提供用户可控的权限管理。

</details>

---

### 6. [GreenGNN: Energy-Aware Windowed Communication Optimization for Distributed GNN Training](https://arxiv.org/abs/2606.02916)

**Authors**: Arefin Niam, Tevfik Kosar, M. S. Q. Zulkar Nine  
**Category**: cs.DC  
**Published**: 2026-06-03  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.02916v1  

#### Abstract
Large-scale graph neural network (GNN) training often requires distributed clusters because graph structure and feature tensors no longer fit in a single node's memory. In sampling-based training, each mini-batch expands into a receptive field that spans partitions and triggers thousands of remote f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# GreenGNN: Energy-Aware Windowed Communication Optimization for Distributed GNN Training —— 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
分布式图神经网络（**Distributed GNN**）训练中存在严重的通信能效问题。由于邻居采样（neighbor sampling）导致每个 mini-batch 都会触发大量跨分区的远程特征请求（remote feature fetches），这些细粒度的 **RPC**（Remote Procedure Call）带来两个主要能耗来源：
- **每条 RPC 的固定启动开销**（initiation overhead）
- **GPU 在等待远程数据时持续消耗的空闲功耗**（GPU stall power）

实验表明，在分布式 GraphSAGE 中，**数据移动占总系统能耗的 76–85%**，而计算仅占 4–17%，说明通信是能效瓶颈。

### 🚀 提出的新方法与核心思想
提出 **GreenGNN**，一种面向能效优化的分布式 GNN 训练系统，其核心创新在于：

#### （1）**基于窗口的缓存机制（Window-based Caching）**
- 将训练划分为多个连续 mini-batch 组成的“窗口”（window），大小为 $ W $
- 在每个窗口开始前，通过预计算访问轨迹识别“热点远程特征”（hot remote features）
- 通过 **bulk transfer** 从各分区所有者一次性拉取这些热点特征并缓存到本地
- 窗口内的大多数请求直接从缓存服务，少量未命中仍走按需路径（on-demand path）

> 💡 动机：GNN 采样具有**短时爆发性时间局部性**（bursty, short-lived temporal locality）——某些节点在连续几个 mini-batch 中被频繁重用，之后迅速退出工作集。

#### （2）**离线模拟器引导的自动调优器（Simulator-guided Autotuner）**
- 利用固定随机种子下访问模式的确定性，运行一次轻量级校准轮次生成完整 epoch 的访问轨迹
- 构建一个 **discrete-event simulator**，结合物理驱动的能量模型 + 学习排序修正模块（learned rank correction），预测不同 $ W $ 下的能耗
- 自动选择最优窗口大小 $ W^* $，避免在线搜索带来的高昂代价

### 🔍 相比现有方法的优势
| 方法 | 局限性 | GreenGNN 如何改进 |
|------|--------|------------------|
| **静态缓存**（如 PaGraph, GNNLab） | 缓存内容固定，无法适应训练过程中热点变化 | 周期性刷新缓存（以窗口为单位），兼顾效率与适应性 |
| **动态缓存**（如 BGL, Legion） | 每次访问都要跟踪元数据，CPU 开销高 | 不进行 per-access tracking，无跨 worker 同步开销 |
| **纯吞吐导向设计**（如 P3, PipeGCN） | 侧重隐藏延迟而非减少通信总量 | 明确将 **energy** 作为首要优化目标 |
| **RapidGNN**（基础框架） | 虽有 trace-based prefetching，但未针对 energy 优化 | 引入 bulk transfer consolidation + GPU frequency scaling + energy-aware autotuning |

> ✅ GreenGNN 是首个将 **communication energy** 作为第一优先级优化目标的分布式 GNN 系统。

---

## 2. 核心实验方法和设置

### 📊 数据集
使用三个标准 benchmark 数据集：
- **Reddit**: 233K 节点，114M 边
- **OGBN-Products**: 2.4M 节点，61.9M 边
- **OGBN-Papers100M**: 111M 节点，1.6B 边

覆盖从小到超大规模的真实图场景。

### ⚙️ 实验平台与配置
- **硬件环境**：4 节点 Chameleon Cloud 集群
  - 每节点：Intel Xeon CPU, 2× NVIDIA P100 GPU, 25 Gbps Ethernet
- **软件栈**：基于 PyTorch 和 DistDGL 实现
- **模型**：2-layer GraphSAGE，fan-out={10,25}
- **批大小**（Batch Size）：测试 B=1000, 2000, 3000
- **分区策略**：METIS 分割为 4 个分区，每节点一个

### 📈 评估指标
| 指标类别 | 具体指标 |
|--------|---------|
| **能效** | 总系统能量（CPU + GPU）、GPU 能量、CPU 能量（单位：kJ） |
| **性能** | 平均每 epoch 时间（s）、端到端吞吐加速比 |
| **准确性** | 与 baseline 保持一致（不修改模型逻辑） |
| **模拟器有效性** | Kendall’s Tau 排序相关性、Top-1 准确率 |

### 🔁 对比基线方法
| 基线 | 简要描述 |
|------|----------|
| **Default DGL (DistDGL)** | 原始按需远程获取，无缓存 |
| **BGL** | 动态缓存（LRU/LFU），优化 I/O |
| **RapidGNN** | 同源 trace-based 预取，但目标为吞吐 |
| **GraphStorm** | 工业级全功能框架，支持分布式采样等 |

---

## 3. 主要实验结果和性能指标

### 📉 能效提升（vs. Default DGL）

| 指标 | 提升幅度 |
|------|----------|
| **总系统能量降低** | **27–43%** |
| **GPU 能量降低** | **36–71%**（最高达 70.9% on Reddit） |
| **CPU 能量降低** | **24–40%** |

> 示例：在 Reddit 上，总能耗从 326.8 kJ 降至 189.4 kJ（↓42.0%），节省 137.4 kJ/训练轮次。

### ⚡ 性能加速（端到端训练速度）
| 数据集 | Epoch Time Speedup |
|-------|--------------------|
| **Reddit** | **3.9×**（6.84s → 1.77s） |
| **OGBN-Products** | **1.4×** |
| **OGBN-Papers100M** | **1.4×**（12.66s → 8.95s） |

> ❗ 关键发现：**GreenGNN 同时实现了更低能耗和更高吞吐**，达到 Pareto 最优。

### 🔍 与其他先进系统的对比
| 方法 | 总能耗 | 训练时间 | 备注 |
|------|--------|----------|------|
| **GreenGNN** | ✅ 最低（除 Papers100M @ B=1000） | ✅ 最快或接近最快 | 综合表现最佳 |
| **RapidGNN** | 高出 6.2–20.2% | 略慢 | 缺少 energy-aware tuning 和频率调节 |
| **GraphStorm** | **极高**（Papers100M 达 1452kJ，为 GreenGNN 的 **4.7×**） | 更慢 | 后处理阶段引入巨大开销 |

> 💥 在 OGBN-Papers100M 上，GraphStorm 因 post-training overhead 导致能耗爆炸式增长。

### 🧪 消融分析与关键发现
#### （1）窗口大小 $ W $ 的影响（Figure 6 & 7）
- 能耗随 $ W $ 增大先下降后趋于平缓，呈 **凸形曲线**（convex shape）
- 过小 $ W $：无法充分摊销 RPC 启动成本
- 过大 $ W $：热点集合过时（staleness），缓存命中率下降
- 最佳值通常在 $ W=16 \sim 32 $

#### （2）GPU 能量大幅下降的原因
- **减少阻塞通信** → 缩短 GPU stall time
- **采样阶段降低 GPU 频率** → 主动降低 idle power draw
- 二者协同作用使 GPU 能耗降幅远超总能耗降幅

#### （3）模拟器有效性验证（Table III）
| 指标 | 结果 |
|------|------|
| **Top-1 准确率** | **6/9 配置正确选出最优 $ W $** |
| **Kendall’s Tau** | **0.62 ~ 1.00**（平均 0.85），表明排序高度一致 |
| **最大误差惩罚** | < 5.3%（仅 3–8 kJ） |

> 表明离线模拟器可在极短时间内（<5 秒）高效指导参数选择。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **通信启动开销和 GPU 空闲功耗主导了分布式 GNN 的能耗预算**，单纯优化吞吐不能有效节能。
2. **GNN 邻居采样具有强短时局部性**，适合采用周期性批量加载（window-based bulk transfer）替代高频细粒度 RPC。
3. **GreenGNN 成功将数千次小 RPC 合并为数次大传输**，显著摊销启动成本，并减少 GPU 阻塞时间。
4. **能量与性能可兼得**：GreenGNN 不仅节能最多，而且训练最快，在多个配置下实现 **Pareto 最优**。
5. **离线模拟器能准确预测最优窗口大小**，具备实用性和泛化能力。

### ⚠️ 方法局限性
1. **依赖确定性采样轨迹**：需要固定随机种子和分区方案；若每次训练访问模式变化较大，则预取效果可能下降。
2. **当前为静态调优**：$ W^* $ 在训练前选定，未考虑训练过程中热点分布的演化。
3. **假设同构硬件**：未考虑混合 GPU 类型或异构网络带宽的情况。
4. **主要针对 sampling-based 训练**：对 full-graph training 支持尚未验证。

### 🔮 未来工作方向
1. **自适应窗口机制**：在训练过程中动态调整 $ W $，响应热点漂移。
2. **扩展至异构集群**：适配不同代际 GPU、RDMA/NVLink 加速通信路径。
3. **集成 RDMA-aware 批量传输**：进一步降低 bulk transfer 的协议开销。
4. **通用性验证**：应用于其他 GNN 架构（如 GCN, GAT, GraphTransformer）及更大规模工业图。
5. **碳足迹建模**：将节能量转化为实际碳排放减少，推动绿色 AI 发展。

---

> 🌱 **一句话总结**：  
> GreenGNN 通过洞察 GNN 采样的**短时爆发局部性**，提出 **window-based caching + bulk transfer + energy-aware autotuning** 的组合拳，在不影响精度的前提下，实现 **能耗降低 27–43%、GPU 能耗降低高达 71%、训练提速最高达 3.9×**，是迈向可持续分布式 GNN 训练的重要一步。

</details>

---

### 7. [MOSAIC: Efficient Mixture-of-Agent Scheduling via Adaptive Aggregation and Inference Concurrency](https://arxiv.org/abs/2606.03014)

**Authors**: Saptarshi Mitra, Yifan Zhang, Rachid Karami, Phyo Pyae Moe Aung, Nazmul Takbir, Sreetama Sarkar, Souvik Kundu, Sitao Huang  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.03014v1  

#### Abstract
Mixture-of-Agents (MoA) systems improve reasoning accuracy by routing each query to multiple expert LLMs and aggregating their outputs. Efficiently executing this workload on limited GPU resources has bottlenecks. Skill-based routing creates skewed expert demand, and combining instruction-tuned LLMs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MOSAIC: Efficient Mixture-of-Agent Scheduling via Adaptive Aggregation and Inference Concurrency

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Mixture-of-Agents (MoA) 系统通过将查询路由到多个专家 LLM 并聚合其输出来提升推理准确性，但在有限 GPU 资源下执行此类任务时面临严重瓶颈：
- **技能路由导致专家需求不均衡**（popularity skew），部分 GPU 过载而其他空闲；
- **指令调优模型与长推理模型混合使用**，生成长度差异极大（最高达13×），加剧负载失衡；
- 传统调度策略（如 round-robin）导致严重的 GPU idling 和吞吐量崩溃。

### 提出的新方法与创新思路
作者提出 **MOSAIC** —— 一种面向 MoA 工作负载的高效调度框架，包含两大核心技术：

#### （1）基于 ILP 的多模型联合调度器（ILP-based Multi-Model Scheduling）
- 将专家生成阶段建模为一个 **Integer Linear Program (ILP)**，联合优化：
  - 模型到 GPU worker 的放置（model placement）
  - 每个 worker 上的任务分配数量（prompt assignment）
  - 对重型推理模型进行选择性复制（selective replication）
- 引入动态副本上限 $ R_{\text{max}} $，依据“加载成本 vs 执行时间”比例决定是否复制模型，避免不必要的内存开销。

#### （2）置信度感知的自适应聚合机制（Confidence-aware Adaptive Aggregation）
- 利用专家间的一致性作为答案正确性的信号，在多数专家达成一致时跳过最终 aggregator LLM 的调用。
- 定义信心函数 $ C(q) = \frac{\max|\{i: a_i = a\}|}{k} $，设定阈值 $ T $ 实现两种操作模式：
  - 保守门控 $ T=1 $：仅跳过 3:0 投票问题
  - 激进门控 $ T=2/3 $：跳过 3:0 和 2:1 投票问题

### 相比现有方法的优势
| 维度 | MOSAIC | 传统方法（如 Round-Robin、Data Parallelism） |
|------|--------|---------------------------------------------|
| 资源利用率 | 高，负载均衡 | 存在显著 GPU idling |
| 吞吐效率 | 支持异构模型并发处理 | 忽略生成长度差异 |
| 推理延迟 | 显著降低端到端延迟 | 受限于最慢 worker（straggler） |
| 准确性 | 在 ±0.1pp 内匹配基线精度 | 不支持聚合跳过 |

---

## 2. 核心实验方法和设置

### 数据集
在三个标准基准上评估性能：
- **MMLU-Pro**：涵盖 STEM 与人文领域的 2100 道多选题
- **MedMCQA**：医学领域 4183 道多选题
- **GPQA**：研究生级别科学问题，共 198 道，难度高且技能集中

所有任务均采用 skill-based routing 预先分配每道题的 3 个专家模型。

### 实验平台
- 硬件：单台服务器配备 **4× NVIDIA A100 80GB GPU**
- 框架：使用 **vLLM 0.7.1** 进行 LLM 推理服务
- 模型部署方式：每个 GPU 一次运行一个模型，按需加载/卸载（no model sharding）

### 模型池（Hybrid Expert Pool）
共 6 个专家模型，分为两类：
| 类别 | 模型 | 平均输出 token 数 |
|------|------|------------------|
| Reasoning-tuned | LlamaR1, QwenR1 | ~3000 tokens |
| Instruction-tuned | Gemma, Exaone, GLM, Qwen | ~200 tokens |

> 输出长度差异高达 **13×**，构成典型异构负载。

### 评估指标
- **End-to-end wall-clock time**（总耗时）
- **Expert-stage makespan**（专家生成阶段最大完成时间）
- **Aggregator-stage latency**（聚合阶段耗时）
- **Speedup**（相对于基线的加速比）
- **Accuracy Δpp**（相对于完整聚合的准确率变化，单位 percentage point）
- **Skip %**（被跳过的 aggregator 调用比例）

### 基线方法
- **Round-Robin (RR)**：按顺序将模型分配给 GPU，每个模型只加载一次
- **Data Parallelism (DP)**：所有 GPU 加载全部模型，均匀分片请求

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | MMLU-Pro | MedMCQA | GPQA |
|------|---------|--------|-----|
| **Expert-stage speedup** | 1.69× | 1.53× | **2.54×** |
| **Aggregator-stage speedup** | 1.74× | **4.23×** | 1.53× |
| **End-to-end speedup** | 1.71× | **2.04×** | 2.34× |
| **Accuracy deviation (Δpp)** | +0.24 / -0.62 | -0.04 / +0.32 | **+1.00 / +0.49** |
| **Aggregator skip rate (T=1 / T=2/3)** | 45.71% / 84.62% | 47.48% / 90.18% | 37.37% / 87.37% |

> 注：T=1 表示保守门控（仅跳过 3:0），T=2/3 表示激进门控（跳过 3:0 和 2:1）

### 与基线对比结果
- **相比 Round-Robin**：
  - MOSAIC 在所有任务上实现 **1.7–2.5× 的端到端加速**
  - 最大收益来自 **GPQA**，因其高度集中在 LlamaR1 上，ILP 成功将其复制至多个 GPU
  - **MedMCQA** 中，由于多数问题专家意见一致，自适应聚合带来高达 **4.23× 的聚合阶段加速**

- **相比 Data Parallelism (DP)**：
  - DP 在小生成预算场景下表现差（如 allnonR 场景慢 29%）
  - 因为 DP 导致：
    - 序列化模型加载开销无法摊销
    - 分片太细导致 per-shard throughput 下降 40–60%
  - MOSAIC 更智能地控制复制粒度，仅对重型推理模型复制

### 消融实验结果
#### （1）关于自适应聚合的消融分析
- 发现：**3:0 投票问题平均输出 token 仅为 1:1:1 的 38%**（1064 vs 2696 tokens）
- 说明：简单问题更容易达成共识，复杂问题才需要 aggregator 处理
- 结果验证了“跳过一致性高的问题”是合理且高效的策略

#### （2）关于调度策略的有效性
- 图表显示 RR 存在明显 straggler（如某 worker 耗时 >2500s，其余 <500s）
- MOSAIC 通过将重负载推理模型（如 LlamaR1）复制到多个 GPU，并搭配轻量模型平衡负载，使各 worker 完成时间接近并行理论下界

---

## 4. 关键结论和发现

### 主要发现
1. **MoA 工作负载具有高度异构性和偏斜性**：
   - 生成长度差异可达 **13×**
   - 模型选择频率差异可达 **92×**
   - 必须设计专门的调度机制应对

2. **ILP-based 调度能有效缓解负载失衡**：
   - 通过联合优化 placement 与 prompt 分配，显著减少 makespan
   - 选择性复制重型推理模型是关键

3. **专家间一致性可作为高质量置信信号**：
   - 在 MMLU-Pro 中，45.7% 的问题获得 3:0 全票一致
   - 此类问题即使不经过 aggregator，准确率仍优于或等于原始结果

4. **MOSAIC 实现了性能与精度的良好权衡**：
   - 最高 **2.54× 专家阶段加速**，**4.23× 聚合阶段加速**
   - 端到端提速 **1.7–2.3×**
   - 准确率偏差控制在 **±0.1pp** 内（保守门控）

### 局限性
1. **静态置信准则（static confidence criterion）**：
   - 用户需预先设定跳过阈值 $ T $，不能动态调整
   - 不同任务类型可能需要不同的策略

2. **依赖离线 profiling**：
   - 需要提前获取每个模型的输出 token 分布和执行时间模型
   - 动态变化的工作负载可能影响调度效果

3. **当前适用于固定批处理场景**：
   - 主要针对 offline evaluation，尚未扩展至在线 streaming 请求

4. **未考虑非 LLM 瓶颈任务**：
   - 在 agent-based workflow 中，许多任务瓶颈不在 LLM 推理本身，难以直接迁移

### 未来工作方向
- 设计 **动态 confidence gating** 机制，根据上下文或任务难度自动调整跳过策略
- 扩展至 **online serving 场景**，支持实时请求流调度
- 探索 **异构 agent workflow 中的资源调度**，超越纯 LLM 推理
- 结合 **model parallelism** 支持超大规模模型部署
- 引入 **learning-based scheduler** 替代 ILP，在更大规模下保持可扩展性

--- 

> ✅ 总结：**MOSAIC 是首个系统性解决 MoA 推理调度挑战的工作，结合 ILP 优化与 confidence-aware aggregation，在真实硬件上实现了高达 2.3× 的端到端加速，同时保持精度不变，为高效多智能体系统部署提供了重要实践路径。**

</details>

---

### 8. [DECA: Decentralizing Block-Wise Adam for Efficient LLM Full-Parameter Fine-Tuning on Non-IID Data](https://arxiv.org/abs/2606.03209)

**Authors**: Yunsheng Yuan, Shaowei Li, Kai Wang, Zhongyuan Sun, Zheng Zhang, Kai Han, Jun Luo, Feng Li  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.03209v1  

#### Abstract
Fine-tuning large language models (LLMs) in privacy-sensitive and resource-constrained environments remains challenging. Since training data are often distributed across multiple clients, decentralized fine-tuning offers a natural paradigm for collaborative adaptation without a central server. Howev...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DECA: Decentralizing Block-Wise Adam for Efficient LLM Full-Parameter Fine-Tuning on Non-IID Data

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在隐私敏感且资源受限的环境中对大语言模型（LLMs）进行微调面临三大挑战：
- **通信开销巨大**：传统去中心化方法需要交换完整的模型参数，对于十亿级参数的LLM而言通信成本过高。
- **内存限制**：Full-Parameter Fine-Tuning（FPFT）要求存储所有参数的梯度和优化器状态，超出单个客户端GPU内存容量。
- **非独立同分布（non-IID）数据导致的客户端漂移**：各客户端数据分布差异大，局部更新容易偏离全局目标，造成训练不稳定。

现有方法如Dec-LoRA等基于Parameter-Efficient Fine-Tuning（PEFT），虽提升了效率，但因仅更新少量参数而牺牲了下游任务性能。

### 提出了什么新方法或新思路
本文提出 **DECA**（Decentralized Block-wise Adam），一种面向non-IID数据的高效去中心化FPFT框架，其核心创新包括：

- **Block-wise Adam Optimization**：将模型参数划分为互不重叠的块（blocks），每次只激活一个块进行顺序优化，显著降低每轮通信和内存消耗。
- **Block-wise Moment Approximation (BMA)**：引入一阶和二阶块级动量估计机制，结合**本地新鲜梯度统计**与**共识推导的差异信号**（consensus-derived discrepancy signal），在去中心化环境下稳定Adam风格的优化过程。
- **理论收敛保证**：在non-IID和随机梯度条件下证明了DECA具有 $O(1/\sqrt{TBR})$ 的收敛速率，与最先进的去中心化算法相当。

### 相比现有方法的优势
| 维度 | DECA | 现有PEFT方法（如Dec-LoRA） |
|------|------|--------------------------|
| **性能** | 支持FPFT，保留完整模型表达能力 | 仅更新低秩子空间，性能受限 |
| **效率** | 分块通信，减少带宽压力 | 参数稀疏更新，通信少但计算仍高 |
| **稳定性** | BMA缓解client drift，提升收敛稳定性 | 易受non-IID影响，需额外正则化 |

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 分类任务：
- **NWGI**（News Writer Genre Identification）
- **AGNEWS**
- **TFNS**（Twitter Financial News Sentiment）
- **MNLI**

#### 生成任务：
- **Alpaca** 数据集用于指令跟随能力评估

所有数据通过Dirichlet分布（$\alpha=0.25$）划分以模拟强non-IID场景。

### 实验设置和评估指标

| 设置项 | 配置 |
|-------|------|
| **模型规模** | Qwen2-1.5B, Qwen2.5-3B, Llama-2-7B, Llama-3.1-8B |
| **网络拓扑** | 8个客户端的Erdős-Rényi（ER）图；也测试Ring、Bipartite等拓扑 |
| **训练轮数** | $T=4$ global rounds，每block内$R=48$ local steps |
| **分块策略** | 每两个连续Transformer层组成一个block（共28–36个blocks） |

#### 评估指标：
- **分类任务**：Accuracy（Acc.）和F1 Score
- **生成任务**：使用 **Vicuna (VIC.)** 和 **MT-Bench (MT.)** 作为自动评判基准（LLM-as-a-judge）

### 基线方法对比
- **Dec-Adapter**：基于adapter的去中心化PEFT
- **Dec-LoRA**：去中心化LoRA，交换并聚合低秩适配矩阵
- **DeCAF**：基于TSVD分解的去中心化LoRA，增强共识性

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1 & 2）

#### 表1：分类任务平均性能（Llama-3.1-8B）
| 方法 | 平均 Acc. | 平均 F1 |
|------|----------|--------|
| Dec-Adapter | 23.28% | 13.07% |
| Dec-LoRA | 80.42% | 75.55% |
| DeCAF | 64.59% | 54.56% |
| **DECA** | **82.30%** | **77.18%** |

> ✅ DECA在多个模型上均取得最优表现，尤其在大型模型上优势明显。

#### 表2：生成任务性能（Llama-2-7B）
| 方法 | MT-1 | MT-2 | MT. |
|------|------|------|-----|
| Dec-Adapter | 3.92 | 2.62 | 3.27 |
| Dec-LoRA | 4.30 | 2.87 | 3.59 |
| DeCAF | 4.31 | 3.00 | 3.65 |
| **DECA** | **4.50** | 2.93 | **3.72** |

> ✅ DECA在MT-1和总体得分上领先，显示更强的多轮对话理解与推理能力。

### 与基线方法的对比结果
- 在分类任务中，DECA相比最强基线（Dec-LoRA）平均准确率提升近 **2%**，F1提升 **1.6%**。
- 在生成任务中，MT.评分超过Dec-LoRA约 **0.13**，优于DeCAF。
- 训练损失曲线显示，DECA收敛速度**快于Dec-Adapter**，且与Dec-LoRA相当甚至更稳（见Fig. 1–2）。

### 消融实验结果（Ablation Study）
#### 对比设置：
- **w/o BMA**：移除BMA模块
- **w/ trivial BMA**：仅用共识信号替代本地梯度（类似QG Momentum）

#### 结果（Qwen2.5-3B分类任务）：
| 方法 | 平均 Acc. | 平均 F1 |
|------|-----------|---------|
| DECA | **79.32%** | **74.12%** |
| w/ trivial BMA | 65.19% | 61.69% |
| w/o BMA | 73.09% | 66.90% |

> 🔍 发现：
- 移除BMA导致性能下降 **6.23% Acc.**
- “trivial BMA”性能最差，说明**过度依赖共识会破坏本地优化方向**
- DECA实现了**本地梯度驱动**与**共识引导修正**之间的良好平衡

---

## 4. 关键结论和发现

### 论文的主要发现
1. **DECA实现了去中心化环境下的高效FPFT**：首次成功将block-wise Adam扩展至去中心化non-IID设定，在不依赖中央服务器的情况下完成LLM全参数微调。
2. **BMA机制有效缓解client drift**：通过融合本地梯度与邻居差异信号，使每个客户端能在保持本地优化的同时向全局一致靠拢。
3. **理论与实践一致性高**：实验证明其收敛行为符合理论预测的 $O(1/\sqrt{TBR})$ 速率，并在多种模型和任务上稳定优于现有PEFT方法。
4. **资源效率显著提升**：
   - 内存峰值降至约 **$M + 10M/B$**（远低于标准Adam的~9M）
   - 通信量减少为原来的 $1/B$
   - 端到端延迟降低最多达 **90.21%**（vs DeCAF）

### 方法的局限性
- **分块粒度影响性能**：实验表明越细的分块（granularity=1）效果越好，但可能增加调度复杂度。
- **对超参数相对鲁棒但仍有偏好**：BMA中的 $\beta_2$ 对生成任务影响较明显，需适当调优。
- **尚未支持异构设备混合训练**：假设所有客户端具备相似算力与内存。

### 未来工作方向
- 扩展至**异构客户端**与**动态拓扑**场景
- 探索**自适应分块策略**（adaptive block partitioning）
- 结合**量化**或**稀疏化**进一步压缩通信
- 应用于更多模态（如视觉-语言模型）的去中心化训练

--- 

> 📌 **总结一句话**：  
> **DECA 是首个实现高效、稳定、去中心化的 LLM 全参数微调框架，通过 block-wise Adam + BMA 设计，在 non-IID 数据下兼顾高性能与低资源消耗，为边缘侧私有化 LLM 微调提供了可行路径。**

</details>

---

### 9. [TreeFlash: Parallel AR-Approximation for Faster Speculative Decoding](https://arxiv.org/abs/2606.03819)

**Authors**: Peer Rheinboldt, Fr\'ed\'eric Berdoz, Roger Wattenhofer  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.03819v1  

#### Abstract
One-shot block drafters for speculative decoding generate the full draft in a single forward pass, achieving strong throughput by eliminating sequential token generation. However, they predict each draft token conditioned only on the prefix context, with no dependence on previously drafted tokens. T...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TreeFlash: Parallel AR-Approximation for Faster Speculative Decoding**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 **one-shot block drafting** 的 speculative decoding 方法（如 DFlash）虽然通过单次前向传播实现高吞吐量，但其预测的 draft token 分布仅依赖于前缀上下文 $x_{<t}$，而**不依赖于先前已生成的 draft token**。这种 **non-autoregressive conditioning** 导致：
- 随着 draft depth 增加，drafter 的分布与 verifier 的真实 autoregressive 分布之间的差异显著增大（TVD 上升）；
- 在 **tree-based drafting** 中，不同分支被迫共享相同的 marginal distribution，严重限制了树的质量和接受率。

### **提出的新方法：TreeFlash**
TreeFlash 提出了一种轻量级的 **AR-approximation 机制**，在保持 one-shot 并行解码优势的前提下，近似建模 autoregressive 条件依赖关系。

#### **核心创新点**：
- **引入 SwiGLU 层进行 AR 近似**：  
  在每个位置 $t+i$，利用前一个 token 的嵌入 $e_{t+i-1}$ 和当前隐藏状态 $h_{t+i}$，通过一个轻量级 SwiGLU 层更新隐藏表示：
  $$
  h'_{t+i} = h_{t+i} + \text{SwiGLU}(h_{t+i}, e_{t+i-1})
  $$
  从而让模型在预测 $x_{t+i}$ 时“感知”到前一个 draft token，逼近 autoregressive 分布。

- **两阶段树构建机制（Two-stage approximation）**：
  1. 第一阶段：使用原始 DFlash 的 marginal 分布构建一个宽但浅的 M-ary 树（控制分支因子为 $M$）；
  2. 第二阶段：对 M-ary 树中的节点并行应用 AR-approximator，得到条件化后的 token 分布；
  3. 最终使用 OPT-Tree 算法从中选出最优的 $B$ 个候选节点构成 draft tree。

> ✅ **关键优势**：整个过程仍可在 $O(1)$ 时间内完成，保留了 one-shot drafting 的高效性。

### **相比现有方法的优势**
| 方法 | 类型 | 是否 AR-conditioned | 并行性 | 优势 |
|------|------|---------------------|--------|------|
| EAGLE-3 | Autoregressive | ✅ | ❌ | 高质量但慢 |
| DFlash | One-shot marginal | ❌ | ✅✅ | 快但分布偏差大 |
| DDTree | Tree + marginal | ❌ | ✅✅ | 接受率提升 |
| **TreeFlash (本文)** | **Tree + AR-approx** | ✅（近似） | ✅✅ | **兼顾速度与分布准确性** |

- 在相同 draft budget 下，TreeFlash 显著优于 DFlash 和 DDTree；
- 尤其在 **高 draft depth 和大树预算（B=64）下增益更大**；
- 实现了 **state-of-the-art 的 block efficiency 和 speedup**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
覆盖多种任务类型，共 64 样本/数据集用于评估：
- **数学推理**：`MATH-500`, `GSM8K`
- **代码生成**：`HumanEval`, `MBPP`
- **通用指令遵循**：`MT-Bench`

### **实验设置**
- **目标模型（Verifier）**：
  - Qwen3-4B, Qwen3-8B, Qwen3 Coder 30B A3B（MoE 架构）
- **drafter 模型**：
  - 所有方法共享相同的 DFlash backbone，TreeFlash 在此基础上微调并添加 AR-approximator。
- **draft budget $B$**：16 和 64（即每次提交最多 $B$ 个候选节点）
- **block size $\gamma$**：16
- **M 参数**：控制初始 M-ary 树宽度，默认 $M=16$
- **硬件平台**：NVIDIA GH200 GPU，使用 BFloat16 和 PyTorch SDPA

### **评估指标**
| 指标 | 定义 | 说明 |
|------|------|------|
| **Speedup** | 相比 vanilla autoregressive decoding 的吞吐加速比 | 受实现影响较大 |
| **Block Efficiency (T)** | 每轮 draft-verify 迭代平均接受的 token 数（含 residual token） | 更稳定的 drafter 质量衡量标准 |
| **TVD (Total Variation Distance)** | drafter 与 verifier 分布间的 TVD | 衡量分布校准程度 |
| **Top-K Coverage** | drafter top-K token 在 verifier 分布下的累计概率 | 衡量 draft 多样性和命中能力 |

### **基线方法对比**
- **EAGLE-3**：小型自回归 drafter，串行生成
- **DFlash**：单步扩散式 one-shot drafter，生成链状 draft
- **DDTree**：基于 DFlash 输出 + OPT-Tree 构造算法生成 draft tree（边际分布驱动）

> ⚠️ 注意：EAGLE-3 使用不同训练数据，仅作参考；DFlash、DDTree、TreeFlash 共享 backbone，公平可比。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1 & 2）**

#### **总体表现（Qwen3-4B / 8B，B=16）**
| 方法 | 平均 Block Efficiency (T) ↑ | 平均 Speedup ↑ |
|------|----------------------------|---------------|
| DFlash(16) | ~3.82 | ~5.11 |
| DDTree(16) | ~4.38 | ~6.06 |
| **TreeFlash(16)** | **~4.53** | **~6.48** |

> ➕ **相对 DDTree 提升**：+3.9% speedup，+7.5% block efficiency

#### **大树预算下（B=64），优势进一步放大**
| 方法 | 平均 Block Efficiency (T) ↑ | 平均 Speedup ↑ |
|------|----------------------------|---------------|
| DDTree(64) | ~5.23 | ~7.05 |
| **TreeFlash(64)** | **~5.67** | **~7.90** |

> ➕ **相对 DDTree 提升**：**+12.4% block efficiency**, **+9.0% speedup**

#### **在 Qwen3 Coder 30B A3B 上的结果（Table 2）**
| 方法 | HumanEval T | MBPP T |
|------|-------------|--------|
| DDTree(64) | 9.68 | 9.82 |
| **TreeFlash(64)** | **10.57** | **10.22** |

> ➕ 表明 TreeFlash 的增益可迁移到大规模 MoE 模型。

---

### **消融实验结果（Ablation Study, Table 3）**

在 Qwen3-4B 上进行 $B=64$ 设置下的消融分析：

| 变体 | Block Efficiency (Mean T) | 观察 |
|------|----------------------------|------|
| w/o AR-approximation | 7.6 | 不如原始 DFlash 微调 |
| w/ Linear layer | 7.9 | 弱于 SwiGLU，说明非线性重要 |
| **TreeFlash (full)** | **8.6** | 性能最佳 |
| w/ Frozen backbone | 8.6 | 与 full 几乎持平 → **AR-approximator 是主因** |
| w/2-prev tokens | 8.7 | 改进有限，增加计算开销不值得 |
| w/ Cross-Entropy loss | 8.6 | KL 与 CE 效果接近，KL 略优 |
| w/o Loss Scaling | 8.6 | 影响较小，低接受任务略有下降 |

> 🔍 **核心发现**：
> - **SwiGLU 比线性层更有效** → 非线性变换必要；
> - **backbone 微调非必需** → 增益主要来自 AR-approximator；
> - **单 token conditioning 已足够** → 无需 bigram 输入；
> - **KL 散度与 loss scaling 有轻微正向作用**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **缺乏 autoregressive conditioning 是 one-shot drafting 的根本瓶颈**，尤其在 deep draft 和 tree 结构中更为严重。
2. ✅ **TreeFlash 成功在并行框架中引入 AR-like conditioning**，显著缩小了与 verifier 分布的差距（TVD ↓）。
3. ✅ **AR-approximation 的收益随 draft budget 增大而增强**，因为在更深的位置上 marginal 分布退化更严重。
4. ✅ **TreeFlash 在所有任务、模型规模和 decoding regime 下均达到 SOTA 性能**，平均 block efficiency 提升 **+12.4%**，speedup 提升 **+9%**。
5. ✅ **qualitative analysis 显示 TreeFlash 能生成更连贯的 bigram 和路径**，突破了 marginal 分布导致的“嵌套树”结构限制（见 Figure 7, 9, 10）。

### **方法的局限性**
- **依赖 teacher forcing**：训练时使用 ground-truth 前序 token，推理时却用自身生成的 token → 存在 exposure bias。
- **未从头预训练**：基于 DFlash checkpoint 微调，可能受限于初始分布质量。
- **服务端优化兼容性未知**：未测试与量化、PagedAttention、多 batch decoding 等生产级优化的交互。
- **仅验证于 Qwen 系列模型**：是否泛化到其他架构（如 Llama、Phi）尚不清楚。

### **未来工作方向**
- 🔄 **解决 exposure bias**：探索自回归式训练策略或强化学习方法；
- 🌳 **专为 tree generation 设计训练目标**：如 Hu et al. (2026) 提出的 group tree optimization；
- 🔋 **端到端联合训练 AR-approximator 与 backbone**：避免微调破坏初始分布；
- 🧪 **跨模型家族迁移实验**：验证 TreeFlash 在 Llama、Mixtral 等上的有效性；
- ⚙️ **集成至生产系统**：研究其与 FlashAttention、KV Cache 优化、动态批处理的协同效率。

---

> 📌 **一句话总结**：  
> **TreeFlash 通过轻量级 AR-approximation 机制，在不牺牲 one-shot 并行性的前提下，显著提升了 speculative decoding 中 draft 分布的准确性和接受率，是当前 tree-based drafting 的 state-of-the-art 方法。**

</details>

---

### 10. [Don't Gamble, GAMBLe: An Analytical Framework for AI-Driven Research Systems](https://arxiv.org/abs/2606.02863)

**Authors**: Marquita Ellis, Paul Castro  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.02863v1  

#### Abstract
AI-Driven Research Systems (ADRS) -- systems coupling LLMs with automated evaluation to discover algorithms, proofs, and designs -- are being optimized and adopted across domains, but the tools to analyze them have not kept pace. ADRS performance depends on component interactions that are poorly und...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Don't Gamble, GAMBLe: An Analytical Framework for AI-Driven Research Systems**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前，**AI-Driven Research Systems (ADRS)** 已被广泛应用于算法、数学证明和系统设计的自动发现（如 FunSearch、AlphaEvolve）。然而，这些系统的优化高度依赖于生成器（Generator）、评估器（Assessor）和搜索机制（Discovery Mechanism）之间的复杂交互，而现有的分析工具无法有效解释其行为。

主要问题包括：
- **缺乏理论基础**：标准收敛性分析（如 Markov 假设、固定目标函数）在 ADRS 中不成立。
- **组件交互复杂**：不同 Generator 和 Mechanism 的组合表现差异巨大，且无统一排序。
- **调试成本高**：由于每次迭代消耗大量 LLM 调用，盲目尝试配置组合代价高昂。

### **提出了什么新方法或新思路**
作者提出 **GAMBLe 框架**，将 ADRS 分解为四个参数和一个核心对象：

| 组件 | 含义 |
|------|------|
| **G** (Generator) | 生成候选方案的模型或系统（如单个 LLM 或多模型集成） |
| **A** (Assessor) | 评估候选方案的评分函数 |
| **M** (Discovery Mechanism) | 控制搜索过程的策略（如选择父代、构建提示） |
| **B** (Budget) | 计算资源预算（如迭代次数） |
| **Leff = A∘G** | **有效景观（Effective Landscape）**，即生成器与评估器共同作用下的实际优化地形 |

#### **核心创新点**
1. **非马尔可夫性证明**  
   - 证明 ADRS 的最佳得分过程 `{s*}` **不是 Markov 过程**（Theorem 2），因为历史上下文（如早期种子、中间结果）会影响后续生成分布。
   - 因此，仅看当前最优分无法预测未来进展。

2. **有效景观 Leff = A∘G**  
   - 不同 G-A 组合会在同一问题上诱导出**结构不同的优化景观**（Theorem 4）。
   - 例如，某些生成器可能被困在局部盆地，而集成生成器可通过多样性逃逸。

3. **天花板与阶段分类（Regime Classification）**  
   定义两个“天花板”来诊断瓶颈：
   - **Generator Ceiling `s*(G,A)`**：给定 G 和 A 下理论上可达的最高分。
   - **System Ceiling `s*(G,A,M)`**：特定 M 下实际可达的分数。

   并据此划分四种运行状态：

   | Regime | 判断依据 | 应对策略 |
   |--------|--------|--------|
   | **G-limited** | `s*(G,A) < target` | 更换更强的生成器 |
   | **M-limited** | `s*(G,A,M) < s*(G,A)` | 改进搜索机制 |
   | **A-limited** | A 无法区分优劣候选（如 cliff scoring） | 增强反馈信号（partial credit） |
   | **Saturated** | 接近系统天花板 | 增加预算或停止 |

4. **无需穷举消融即可定位瓶颈**
   - 通过跨配置比较（如更换 M 是否提升性能）即可判断是否接近 generator ceiling，避免昂贵的全面消融实验。

### **相比现有方法的优势**
| 方面 | 传统方法 | GAMBLe 框架 |
|------|--------|-------------|
| **分析粒度** | 黑箱端到端评估 | 可解释的组件级诊断 |
| **理论支撑** | 缺乏 | 形式化建模 + 数学证明 |
| **调试效率** | 高成本试错 | 目标性评估识别瓶颈 |
| **适用范围** | 特定任务 | 通用框架，适用于任意 ADRS 架构 |

---

## **2. 核心实验方法和设置**

### **使用的数据集（Problems）**
基于开源竞赛编程基准 **Frontier-CS**，选取三个 NP-hard 问题：

| Problem | 任务描述 | 评估方式（A） |
|--------|--------|-------------|
| **P0: Polyomino Packing** | 将最多 104 个多联骨牌打包进最小矩形 | 连续密度评分（越高越好） |
| **P1: Bounded 2D Knapsack** | 在质量和体积约束下最大化价值 | 归一化连续评分（上限 100） |
| **P11: Palindrome Hamiltonian Path** | 找最短回文路径访问所有空白格子 | Cliff Function：仅完全合法解得正分，否则为 0 |

### **实验设置**
- **总实验量**：760+ 次独立运行，总计 >46,000 次迭代。
- **每轮预算 B**：60 次迭代（控制成本）。
- **重复次数**：每个配置至少 5 次独立运行，若出现多峰分布则追加至每峰 ≥3 次观测。

### **生成器（Generators）**
共 12 种，涵盖三类架构：

| 类型 | 示例 | 特点 |
|------|------|------|
| **Static (Single Model)** | GPT-5.4, Claude Opus 4.6, Kimi-K2.5 | 固定模型，不随时间变化 |
| **Static NoN** | ebl, ebl-pro | 多模型集成 + 内部验证器 |
| **Adaptive NoN** | eb1-preview, eb1-frontier-preview | 动态调整路由与生成策略 |

### **发现机制（Mechanisms）**
均来自 **SkyDiscover** 框架：
- **Best-of-N (BoN)**：贪心基线，展示历史中最优及部分次优解。
- **AdaEvolve**：自适应多岛进化搜索，含停滞检测与迁移。
- **EvoX**：协同进化元搜索，动态调整策略本身。

### **评估指标**
- 最佳得分（Best Score）
- 达到饱和所需迭代数（Iterations to Saturation）
- 成功突破率（Breakthrough Rate）
- 最终得分分布形态（是否多峰）
- 机制增益（相对于 BoN 的改进）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 项目 | 结果 |
|------|------|
| **最大提升幅度** | 正确组件选择可使性能提升 **13–67%** |
| **搜索效率提升** | 效率提高 **6–39×**（以达到饱和所需的迭代数衡量） |
| **最快饱和速度** | 某些配置（如 eb1-family）可在 **1 次迭代内达到满分** |
| **最慢饱和速度** | 如 Claude Opus 4.6 + BoN 需 **39 次迭代** |

### **与基线方法的对比结果**
#### **(1) 无全局最优机制**
- **AdaEvolve / EvoX 并不总是优于 BoN**
  - 对 `eb1-frontier-preview`，AdaEvolve 提升 +24 分；
  - 但对 `Claude Opus 4.6`，AdaEvolve 反而比 BoN **低 5 分**。
- **EvoX 表现反转**：
  - 在 `GPT-5.4` 上，EvoX 成功率 100%，AdaEvolve 仅 27%。

> ✅ **结论：没有“最好”的 M，必须与 G 匹配。**

#### **(2) 开源模型可超越闭源前沿模型**
- 在 P0 上：
  - `GPT-OSS-20B`（开源 MoE）中位数达 **45.0**
  - `Claude Opus 4.6`（商业闭源）仅为 **21.5**
- `Qwen-3.5-9B` 几乎从未突破（0/16 成功）

> ✅ **结论：通用能力排行榜 ≠ ADRS 性能排名。**

#### **(3) 集成生成器可逃逸局部盆地**
- `eb1-base`（静态 NoN）可达 **82.3 分**
- 其动态变体（preview/delta/frontier）却集中在 **44 分盆地**
- 但使用 AdaEvolve 后，eb1-preview 可跳出至 **68–71 分**

> ✅ **结论：集成结构比动态机制更重要。**

### **消融实验结果**
#### **(1) Generator 敏感性**
- 即使使用相同 M 和 A，不同 G 的最终得分差异极大：
  - 最高：82.3（eb1）
  - 最低：3.07（Qwen-3.5-9B）
- 得分分布呈**多峰结构**，表明存在多个可达盆地。

#### **(2) M 的作用取决于 G**
| Generator | BoN → AdaEvolve | BoN → EvoX |
|----------|----------------|-----------|
| eb1-frontier | +24 | +4 |
| Claude Opus 4.6 | -5 | +46 |
| GPT-5.4 | -29 | +46 |

> 显示强烈的 **G×M interaction**。

#### **(3) A-limited 极端案例：P11 全军覆没**
- 所有 22 种配置、233 次运行全部得分为 **0**
- 原因：评估器采用 **cliff function**，任何非法解都得 0 分，导致无梯度信号。
- 尽管生成器产生了大量合理代码（>120K 条，多数实现 BFS/DFS），但仍无法获得正反馈。

> ✅ **结论：当 A 提供不了信号时，再强的 G 和 M 也无效。**

---

## **4. 关键结论和发现**

### **主要发现**
1. **ADRS 是非马尔可夫过程**  
   - 历史上下文持续影响生成行为，初始条件可能导致长期轨迹分化。

2. **Leff = A∘G 决定了优化地形结构**  
   - 不同 G-A 组合会形成不同结构的有效景观，解释了为何某些生成器容易陷入局部最优。

3. **不存在绝对最优的 Generator 或 Mechanism**  
   - 性能是 `(G, A, M)` 三者共同决定的结果，**G×M interaction 普遍存在**。

4. **A-limited 是致命瓶颈**  
   - 当评估器反馈过于稀疏（如 cliff scoring），整个系统无法学习，表现为“零突破”。

5. **正确组件选择带来显著收益**  
   - 在有限预算下（60 次迭代），合适的 G/M 组合可提升性能 **13–67%**，效率提升 **6–39×**。

6. **静态集成优于动态机制**  
   - `eb1-base`（静态 NoN）表现优于其动态版本，说明内部验证结构比外部机制更关键。

---

### **方法的局限性**
| 局限性 | 说明 |
|-------|------|
| **领域泛化性待验证** | 实验基于编程任务，虽具代表性，但仍需在数学、科学等领域进一步验证。 |
| **计算成本未精细建模** | NoN 类生成器内部计算开销大，一次调用可能等价于多次普通 LLM 调用，未计入公平比较。 |
| **低概率事件难以捕捉** | 如罕见突破路径，受限于运行次数可能未被观察到。 |
| **缺乏对种子选择的建模** | 初始种子如何影响最终盆地仍为开放问题。 |

---

### **未来工作方向**
1. **形式化 Leff 的拓扑结构分析**  
   - 研究何时 Leff 存在多个不可达盆地，以及如何设计机制实现跨盆地跳跃。

2. **自动诊断工具开发**  
   - 基于 GAMBLe 框架构建自动化工具，实时判断当前处于哪个 regime，并推荐改进建议。

3. **自适应 Assessor 设计**  
   - 开发动态评估器，能提供 partial credit 或自然语言反馈，打破 A-limited 瓶颈。

4. **预算感知调度器**  
   - 在有限 B 下，智能分配资源给最有潜力的 (G, M) 组合。

5. **扩展至人类-AI 协作场景**  
   - 将人类专家作为“高级 Assessor”或“Meta-Mechanism”，研究混合系统的 Leff 结构。

---

> 🔚 **总结一句话**：  
> **ADRS 的成功不是靠“更好的模型”或“更聪明的搜索”，而是找到正确的 `(G, A, M)` 三角组合。GAMBLe 提供了一套理论语言和诊断工具，让我们不再“赌博”，而是“GAMBLe”。**

</details>

---

### 11. [Experience-Driven Dynamic Exits for LLMs with Reinforcement Learning](https://arxiv.org/abs/2606.03113)

**Authors**: Yanyu Zhu, Hoilam Pao, Niu Hu, Wei Guo, Shaoxiong Zhan, Boyu Lai, Zitai Wang, Yongqin Zeng, Hai-Tao Zheng  
**Category**: cs.CL  
**Published**: 2026-06-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.03113v1  

#### Abstract
Large Language Models suffer from slow autoregressive inference. While self-speculative decoding accelerates this process, its efficiency is hampered by static configurations like fixed exit layers and speculation lengths. We reframe this optimization as a \textbf{Markov Decision Process} and propos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Experience-Driven Dynamic Exits for LLMs with Reinforcement Learning**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
Large Language Models (LLMs) 在自回归推理过程中存在显著延迟，因为每个 token 都需通过完整的 Transformer 层堆栈。虽然 **Self-Speculative Decoding (SSD)** 利用模型早期层作为“draft model”来并行生成候选 token，从而加速推理，但其效率受限于**静态配置**（如固定的 exit layer 和 speculation length），无法适应不同上下文下的 token 预测难度差异。

这种“一刀切”的策略导致在简单 token 上浪费计算资源，在复杂 token 上草案质量差、接受率低，最终限制了整体加速效果。

---

### 🚀 **提出了什么新方法或新思路**
本文提出 **LEDE (Learning-based Dynamic Exit)**，首次将 SSD 中的动态控制问题建模为 **Markov Decision Process (MDP)**，并采用 **offline Reinforcement Learning (RL)** 来学习一个策略，以**动态选择最优的 exit layer 和 speculation length**。

#### 核心创新点：
- **动态 exit layer 选择**：不再固定 draft depth，而是让 agent 在前向传播中逐层判断是否从当前层输出 draft token。
- **自适应 speculation length 控制**：根据中间层状态决定何时终止 drafting 过程，实现更灵活的 speculation 长度。
- **基于经验回放的离线训练**：利用历史 inference 轨迹构建 replay buffer，进行 offline RL 训练，避免在线探索开销。

---

### 🔍 **相比现有方法的优势**
| 方法 | 缺陷 | LEDE 的改进 |
|------|------|-------------|
| **Static SSD (e.g., LayerSkip)** | 固定 exit layer 和 speculation length，泛化能力差 | 动态调整两者，适应上下文变化 |
| **Rule-based Heuristics (e.g., LITE, DV)** | 依赖人工设定阈值（如 confidence threshold） | 学习型 policy 更精细、鲁棒性强 |
| **传统 Early Exit** | 仅用于预测最终 token，不支持 speculation | 支持 speculation 框架下的 dual-control |

> ✅ **优势总结**：LEDE 实现了对“计算深度”与“草案质量”的联合优化，显著提升 speculation 效率和推理速度。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
涵盖多种任务和领域，验证方法通用性：
- **指令跟随**：Alpaca, TOPv2
- **语言建模与摘要**：CNN/DailyMail (CNN/DM)
- **代码生成**：HumanEval

---

### ⚙️ **实验设置**
- **模型范围**：跨尺度评估，包括：
  - LLaMA-3.2-1B, LLaMA-2-7B, LLaMA-2-13B
  - CodeLLaMA-7B, CodeLLaMA-34B
- **基础架构**：所有模型均基于 **LayerSkip** 继续预训练，具备 early-exit 能力。
- **解码方式**：
  - Language Modeling：sampling (temperature=0.6)
  - 其他任务：greedy decoding
- **Acceptance Strategy**：speculative sampling [32]
- **硬件环境**：NVIDIA A100 (80GB), CUDA 12.2, PyTorch 2.6.0
- **实现基础**：基于 LayerSkip 开源代码库扩展

---

### 📊 **评估指标**
| 指标 | 含义 |
|------|------|
| **Average Speculation Length (d)** | 平均每次 speculation 生成的 draft token 数量 |
| **Acceptance Rate (Acc. Rate)** | draft token 被 target model 接受的比例 |
| **Average Exit Layer (E)** | 实际用于 drafting 的平均 exit 层编号（越小表示越早退出） |
| **Speedup** | 相对于 autoregressive decoding 的 wall-clock 加速比 |
| **Rouge-L (R-L)** | 衡量生成质量，确保加速不影响输出一致性 |

---

### 🔁 **基线方法对比**
| Baseline | 类型 | 特点 |
|---------|------|------|
| **Autoregressive (AR)** | 基础对照 | 无加速，1.00× 基准 |
| **LayerSkip (LS)** | Static SSD | 固定 exit layer $E$ 和 speculation length $D$ |
| **LITE** | Rule-based dynamic | 使用 layer-specific confidence 阈值触发 exit |
| **Draft & Verify (DV)** | Adaptive drafting | 固定 exit layer，但动态控制 speculation length |

> 所有动态方法最大 speculation length 限制为 12。

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据（来自 Table 1 & 2）**

| 模型 | 方法 | Speedup | Acc. Rate | E (Avg Exit) | d (Spec Len) |
|------|------|--------|-----------|--------------|---------------|
| LLaMA-3.2-1B | LEDE | **2.04× ~ 2.28×** | 0.867 ~ 0.924 | 3.96 ~ 6.88 | 4.70 ~ 6.84 |
| LLaMA-2-7B | LEDE | **2.58× ~ 2.72×** | 0.893 ~ 0.981 | 6.82 ~ 8.72 | 6.20 ~ 9.40 |
| LLaMA-2-13B | LEDE | **2.51× ~ 2.57×** | 0.928 ~ 0.969 | 9.36 ~ 12.73 | 4.63 ~ 8.06 |
| CodeLLaMA-7B | LEDE | **2.18×** | 0.748 | 8.35 | 5.93 |
| CodeLLaMA-34B | LEDE | **2.07×** | 0.864 | 14.57 | 4.41 |

> 💡 **最高加速达 2.72×**，远超 AR baseline 和各类静态/启发式方法。

---

### 🆚 **与基线方法的对比结果**
- **相比 Autoregressive (AR)**：
  - 实现 **2.0× ~ 2.7× 的端到端加速**
- **相比 LayerSkip (LS)**：
  - 提供额外 **~17% 的速度提升**
  - 尤其在复杂任务上优势明显（如 LLaMA-2-7B 上从 1.95× → 2.72×）
- **相比 LITE / DV**：
  - LITE 虽 exit 更深，但 speculation length 太长导致接受率下降
  - DV 接受率尚可，但 exit layer 固定，灵活性不足
  - LEDE 在 **acceptance rate 和 speedup 之间取得更好平衡**

> ✅ **R-L 指标几乎不变**，说明生成质量未受损。

---

### 🔬 **消融实验结果（Ablation Study）**
在 LLaMA-2-7B 上进行消融分析（Table 3）：

| 配置 | Speedup | Acc. Rate | 说明 |
|------|--------|------------|------|
| **LEDE (Full)** | **2.7×** | 0.858 | 完整双控机制 |
| w/o Adaptive Drafting | 2.04× | 0.700 | 移除动态 speculation length 控制 |
| w/o Dynamic Exit | 1.99× | 0.690 | 移除动态 exit layer 选择 |

> 🔍 **结论**：两个组件均有显著贡献，且具有**协同效应**，共同推动高性能。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **上下文感知的动态控制优于静态规则**  
   不同 token 的预测难度差异大，统一配置无法高效利用计算资源；LEDE 可根据局部上下文动态决策，实现更优 trade-off。

2. **Reinforcement Learning 适合建模 SSD 控制问题**  
   将 exit 决策建模为 MDP，结合 offline RL 从历史轨迹中学习策略，是可行且高效的路径。

3. **Dual-Control 机制有效提升 speculation 效率**  
   同时优化 exit layer 和 speculation length，使得高置信度时多 draft、低置信度时及时停止验证，最大化吞吐量。

4. **训练过程稳定收敛**（见 Fig. 2）
   - Reward 快速上升并在 ~800 episode 后趋于平稳
   - Average Acceptance Rate 达到约 0.9
   - Average Exit Layer 收敛至 ~7，表明策略学会在合适深度退出

---

### ⚠️ **方法的局限性**
- **依赖 early-exit-capable 模型**：需要像 LayerSkip 这样的训练 recipe 来赋予中间层 token 预测能力。
- **训练成本存在**：虽为 offline RL，但仍需收集大量 experience 数据并完成训练流程。
- **尚未扩展至 >70B 模型**：作者指出未来将测试更大规模模型上的表现。

---

### 🔮 **未来工作方向**
- 将 LEDE 扩展到 **70B 及以上规模的 LLMs**
- 探索 **multi-agent RL** 或 **imitation learning** 进一步提升策略泛化性
- 结合 **contextual sparsity** 和 **dynamic routing** 构建更智能的 inference engine
- 推动 RL 在 LLM inference pipeline 中的广泛应用

---

## ✅ 总结一句话
> LEDE 是首个将 **offline RL 应用于 Self-Speculative Decoding 动态控制**的工作，通过学习上下文感知的 exit policy，在保持生成质量的同时实现了高达 **2.7× 的推理加速**，代表了从“静态启发式”向“策略驱动型”推理系统的范式转变。

</details>

---

### 12. [BlobShuffle: Cost-Effective Repartitioning in Stream Processing Systems via Object Storage Exemplified with Kafka Streams](https://arxiv.org/abs/2606.03364)

**Authors**: S\"oren Henning, Otmar Ertl, Adriano Vogel  
**Category**: cs.DC  
**Published**: 2026-06-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.03364v1  

#### Abstract
Shuffling or repartitioning data streams is an essential operation of state-of-the-art stream processing frameworks to support stateful workloads in a large-scale, distributed setting. In today's cloud deployments, however, shuffling can become a major cost driver due to substantial network traffic ...

---

### 13. [Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](https://arxiv.org/abs/2606.03698)

**Authors**: Sangeun Park, Minhae Kwon  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.03698v1  

#### Abstract
A central goal of large language model (LLM) research is to build agentic systems that can plan, act, and adapt through sustained interaction with dynamic environments. While recent LLM-based agents exhibit impressive contextual reasoning, their long-horizon decision-making remains fragile, often su...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Multi²: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对当前基于 **Large Language Model (LLM)** 的智能体在**长时程交互环境**（long-horizon interactive environments）中面临的两大核心挑战：
- **Objective Drift**（目标漂移）：随着交互步数增加，智能体逐渐偏离原始任务目标，导致行为失焦。
- **Token Inefficiency**（推理开销高）：依赖冗长的历史上下文进行决策，导致推理时消耗大量 token。

这些问题在复杂任务（如科学实验模拟、家庭操作、合成制造等）中尤为严重，限制了 LLM 智能体的实际应用能力。

---

### 提出的新方法与思路
作者提出 **Multi²**（读作 Multi-Squared），一个**分层多智能体决策框架**，其核心思想是将智能体的行为解耦为两个互补角色：

| 组件 | 角色 | 功能 | 训练方式 |
|------|------|------|---------|
| **System 1**（高阶智能体） | 高层规划者 | 负责生成上下文感知的子目标（sub-goal），保持全局意图一致性 | 通过 **Supervised Fine-Tuning (SFT)** 进行训练 |
| **System 2**（低阶智能体） | 低层执行者 | 执行原子动作（atomic action）以完成当前子目标 | 采用 **Offline-to-Online Reinforcement Learning (RL)** |

#### 创新设计亮点：
- **角色专业化训练**（Role-Specialized Training）：明确区分规划与执行的优化目标，避免单一模型承担多重任务带来的冲突。
- **离线到在线 RL 流程**：
  - **Offline RL 初始化**：利用高质量离线轨迹初始化策略，提升稳定性。
  - **Online RL 自我改进**：引入带有 **KL 正则化**的在线更新机制，防止策略崩溃（mode collapse），实现安全持续学习。
- **层级数据集构建**：发布三个结构化的分层基准数据集（ScienceWorld, ALFWorld, TextCraft），支持对分层决策能力的系统性训练与评估。

---

### 相比现有方法的优势
| 对比维度 | 传统方法（如 ReAct, Glider） | Multi² |
|--------|----------------------------|-------|
| 结构设计 | 多为扁平结构或仅靠提示区分角色 | 显式分离双智能体架构，参数级解耦 |
| 训练范式 | 主要依赖 SFT 或纯提示工程 | 规划用 SFT，执行用 RL，匹配各自任务特性 |
| 执行鲁棒性 | 易出现无效动作循环（invalid action loops） | 通过 RL 微调显著减少错误动作 |
| 推理效率 | 依赖长上下文提示，token 开销大 | 层级调用机制更高效，token 使用更少 |
| 长期稳定性 | 性能在长任务中快速下降 | 在困难任务上表现稳定，抗 objective drift 强 |

---

## 2. 核心实验方法和设置

### 使用的数据集
论文在三个具有代表性的交互式文本环境中进行评估，覆盖不同类型的长期决策挑战：

| 数据集 | 特点 | 任务类型 |
|-------|------|----------|
| **ScienceWorld** | 科学推理环境，包含相变、电路、化学反应等动态状态变化 | 富含中间反馈的长程实验任务 |
| **ALFWorld** | 家庭场景操作任务，需导航+物体操控（如“清洗杯子并放到桌上”） | 稀疏奖励 + 实例编号敏感 |
| **TextCraft** | 基于配方的合成系统（类似 Minecraft），强调递归分解 | 深度依赖链（depth up to 4）的组合规划 |

> ✅ 所有数据集均被重构为分层格式，公开发布以促进社区研究。

---

### 实验设置与评估指标

#### 模型主干（Backbones）
- Qwen-2.5 1.5B / 3B / 7B
- Mistral 7B v0.3
- Llama-3.1 8B

#### 参数高效微调
- 使用 **LoRA**（Low-Rank Adaptation），每个 System 拥有独立适配器（adapter），实现参数级角色隔离。

#### 评估协议
- **Pass@1 准则**：仅允许一次尝试，强调可靠性与一次性成功率。
- **ID/OOD 分割**：
  - **In-Distribution (ID)**：任务模板见于训练集
  - **Out-of-Distribution (OOD)**：全新任务结构，测试泛化能力
- **任务难度分级**：按平均交互步数分为 Easy / Medium / Hard 三档，分析长程鲁棒性。

---

### 基线方法对比
| 类别 | 方法 | 类型说明 |
|-----|------|---------|
| **Prompt-Based** | ReAct, Reflexion, ADaPT | 不更新参数，依赖提示链（CoT）或自我反思 |
| **Fine-Tuning-Based** | GRPO, Glider | 更新参数，其中 Glider 是当前 SOTA 分层方法 |
| **Proposed** | **Multi²** | 本文提出的方法 |

> ⚠️ 注意：对于 Reflexion，因其本质为多轮试错，报告的是 **pass@6**；其余均为 pass@1。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Backbone | Method | ScienceWorld (ID/OOD) | ALFWorld (ID/OOD) | TextCraft (Success) |
|--------|--------|------------------------|--------------------|----------------------|
| **Qwen-2.5 3B** | ReAct | 20.82 / 9.34 | 6.72 / 5.22 | 3.00 |
| | Glider | 54.69 / 17.75 | 52.86 / 41.43 | 18.50 |
| | **Multi² (Ours)** | **60.68 / 29.04** | **61.43 / 49.29** | **28.50** |
| **Mistral 7B** | ReAct | 23.25 / 8.91 | 4.29 / 6.72 | 23.00 |
| | Glider | 58.33 / 28.22 | 45.00 / 45.71 | 28.50 |
| | **Multi² (Ours)** | **69.97 / 31.32** | **56.43 / 50.71** | **44.50** |
| **Llama-3.1 8B** | ReAct | 23.23 / 10.30 | 6.72 / 7.46 | 9.00 |
| | Glider | 60.48 / 34.36 | 43.57 / 37.86 | 9.50 |
| | **Multi² (Ours)** | **67.61 / 30.68** | **57.86 / 56.43** | **35.60** |

> 📈 Multi² 在所有 backbone 和大多数任务上取得最优性能，尤其在 OOD 和 Hard 任务中优势明显。

---

### 与基线方法的对比结果
- **相比 Prompt-Based 方法（ReAct, Reflexion）**：
  - 平均提升超过 **+40pp**（percentage points）
  - 显著减少无效动作和重复循环（见 Figure 3）
- **相比 Fine-Tuning 方法（Glider）**：
  - 在 ScienceWorld 上 ID 提升约 **+7–10pp**，OOD 提升达 **+11–13pp**
  - 更强的 token 效率（见下图）

#### Token Efficiency 分析（Figure 4 & Table 7）
- **Normalized Token Efficiency**（以 ReAct 为 1.0）：
  - ReAct: 1.0
  - Glider: ~3.36
  - **Multi²: 4.48**
- 表明 Multi² 在达到更高性能的同时，推理开销更低，具备更强的实用性。

---

### 消融实验结果（Ablation Studies）

#### （1）训练配置消融（Table 2 & Figure 6a）
| 配置 | System 1 | System 2 | 性能趋势 |
|------|----------|----------|--------|
| RL-SFT | RL | SFT | 最差 —— RL 不适合高层语义规划 |
| Only SFT | SFT | SFT | 中等 —— 缺乏执行层面的自适应 |
| Only RL | RL | RL | 方差大，不稳定 |
| **Proposed (SFT + RL)** | SFT | RL | ✅ 最佳 —— 匹配角色需求 |

> ✔️ 验证了“SFT for planning, RL for execution”的合理性。

#### （2）模型结构与适配器设计（Figure 6b-c）
- **Hierarchical vs. Single Model**：分层结构带来显著增益（median ↑）
- **Role-Specific Adapters vs. Shared Adapter**：独立适配器大幅提升性能分布上限和稳定性

#### （3）损失函数设计（Figure 7）
- **Offline Loss**：加入 **policy-anchored advantage term** 可缓解过拟合，提升泛化
- **Online Loss**：**KL 正则化**有效抑制策略突变，提高在线学习稳定性

#### （4）模型规模影响（Figure 8 & Table 11）
- 在 OOD 任务中，性能随模型 scale 单调上升（Qwen-2.5 从 1.5B → 7B，OOD 成功率从 48.57% → 64.29%）
- 表明更大的 backbone 更有利于处理分布外泛化

---

## 4. 关键结论和发现

### 主要发现
1. **Objective Drift 是可缓解的系统性问题**：
   - 通过将规划与执行解耦，并辅以 RL 驱动的动作优化，可以显著降低目标漂移风险。
2. **角色专业化优于统一建模**：
   - 不同任务角色应使用不同的训练范式（SFT vs RL）和参数空间（LoRA adapter 分离）。
3. **Offline-to-Online RL 是有效的自适应路径**：
   - 离线初始化提供稳定起点，线上微调可在不破坏已有知识的前提下持续改进执行策略。
4. **层级结构本身即具价值**：
   - 即使控制其他变量，引入显式子目标机制也能提升长期任务的成功率。

---

### 方法的局限性
- **依赖高质量子目标标注**：目前子目标由 GPT-4-Turbo 自动生成，可能存在噪声或偏差。
- **System 1 固定不变**：未对高层规划器进行在线优化，可能错过整体策略升级机会。
- **计算成本较高**：双智能体 + RL 训练流程比纯 SFT 更耗资源。
- **仍受限于 LLM 的底层能力边界**：在极端复杂的逻辑推理或符号操作任务中仍有失败案例。

---

### 未来工作方向
- **动态子目标调整**：让 System 1 能根据 System 2 的反馈动态修正计划。
- **双向通信机制**：探索 System 2 向 System 1 反馈执行障碍的能力。
- **完全端到端的分层学习**：减少人工构造数据依赖，发展自动子目标发现算法。
- **扩展至多模态与具身智能**：应用于视觉-语言导航、机器人控制等真实世界场景。

---

## 总结
> **Multi²** 通过**角色解耦 + 分层训练 + 离线到在线 RL** 的设计，在多个交互式环境中实现了**更强的长程鲁棒性、更高的推理效率和更好的泛化能力**。它不仅是一个高性能框架，更提出了一种新的 LLM 智能体构建范式——**功能分工、各司其职、协同进化**。同时发布的分层数据集也为后续研究提供了宝贵基础。

</details>

---

### 14. [WISE-HAR: A Generalizable Ensemble Deep Learning Framework for WiFi-Based Human Activity Recognition](https://arxiv.org/abs/2606.02974)

**Authors**: Maheen Arshad, Qindeel E Zahra, Muhammad Khuram Shahzad  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.02974v1  

#### Abstract
Human Activity Recognition (HAR) using WiFi signals has emerged as a transformative technology for smart homes, healthcare monitoring, security systems, and ambient assisted living. Unlike traditional camera-based systems that raise significant privacy concerns and fail in low-light conditions, or w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题**: *WISE-HAR: A Generalizable Ensemble Deep Learning Framework for WiFi-Based Human Activity Recognition*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该研究针对当前 **WiFi-based HAR**（Human Activity Recognition）系统在实际部署中面临的三大挑战：

1. **High Performance Variance**（高方差）：单一模型对不同活动或环境表现不稳定。
2. **Small Dataset Size**（数据量小）：深度学习依赖大量数据，而现有WiFi HAR数据集样本稀少（如Wallhack1.8k每类仅约400–500个样本），易导致过拟合。
3. **Poor Cross-Condition Generalization**（泛化能力差）：模型在训练条件变化时（如从LOS到NLOS、更换天线）性能急剧下降。

### 🚀 提出的新方法与创新思路

本论文提出 **WISE-HAR** 框架，包含三项关键技术改进：

#### **Improvement 1: Ensemble Learning with Diverse CNN Architectures**
- 集成五种不同的 **CNN** 架构进行软投票（Soft Voting）：
  - Deep CNN（自定义深层网络）
  - Wide CNN（自定义宽层网络）
  - MobileNetV2（轻量级，适用于边缘设备）
  - ResNet50V2（残差连接，缓解梯度消失）
  - EfficientNetB0（复合缩放，高效准确）
- 利用模型多样性降低预测方差，提升鲁棒性。

#### **Improvement 2: Aggressive Data Augmentation**
为应对小数据集问题，设计了物理意义明确的数据增强策略：
- **Time-warping**：模拟不同行走速度（旋转+水平偏移）
- **Frequency masking**：模拟信号干扰（垂直偏移+缩放频率轴）
- **Noise addition**：添加亮度调整、shear等噪声，模拟真实环境扰动
- 每个原始图像生成5–10个变体，有效将训练集扩大至2000–4000样本。

#### **Improvement 3: Comprehensive Cross-Scenario Evaluation**
首次系统评估跨场景与跨硬件的泛化能力：
- **Cross-Scenario**: 在 **LOS** 上训练，在 **NLOS** 上测试
- **Cross-Antenna**: 使用 **Biquad** 天线训练，在 **PIFA** 天线数据上测试
- 更贴近现实世界的应用需求。

### 🔍 相比现有方法的优势
| 维度 | WISE-HAR优势 |
|------|--------------|
| **模型稳定性** | Ensemble显著减少单模型波动，提高可靠性 |
| **小样本适应性** | 数据增强使传统ML（如Random Forest）也能达到95%精度 |
| **现实适用性** | 显著优于大多数仅在同场景下评估的工作 |
| **可复现性与开放性** | 开源代码：https://github.com/maheenarshad198-jpg/HAR |

---

## 2. 核心实验方法和设置

### 📁 使用的数据集
- **Wallhack1.8k dataset**
  - 包含WiFi CSI经STFT处理后的 **spectrogram图像**
  - 三类人类活动：
    - `No Presence`（无人）
    - `Walking`（走路）
    - `Walking + Arm-waving`（走+挥手）
  - 多种配置：
    - 场景：**LOS**（直视） vs **NLOS**（非直视）
    - 天线类型：**Biquad**（定向） vs **PIFA**（全向）

#### 数据划分（以LOS/Biquad为例）：
| 配置 | Train | Validation | Test |
|------|-------|------------|------|
| LOS/Biquad | 369 | 46 | 46 |
| 总计 | 1,104 | 137 | 137 |

### ⚙️ 实验设置
- **输入预处理**：
  - 图像尺寸统一为 `224×224×3`（适配ImageNet预训练模型）
  - 像素归一化至 `[0,1]`
- **训练参数**：
  - Optimizer: Adam (`lr=0.001`)
  - Batch Size: 16（自定义CNN）、32（迁移学习模型）
  - Early Stopping: 10轮无提升则停止
  - 最大训练轮数：50
- **评估协议**：
  - **Baseline Test**: LOS → LOS（相同配置）
  - **Cross-Scenario Test**: LOS → NLOS
  - **Cross-Antenna Test**: Biquad → PIFA
- **评估指标**：
  - Test Accuracy
  - Absolute Drop 和 Relative Drop（衡量泛化能力）

### 🆚 基线方法对比
- 单一模型对比：
  - Deep CNN, Wide CNN, MobileNetV2, ResNet50V2, EfficientNetB0
- 传统机器学习模型：
  - Random Forest（with/without augmentation）
- 是否使用数据增强作为消融变量

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ 表1：各模型在 **LOS/Biquad 测试集** 上的表现
| Model | Test Accuracy |
|-------|----------------|
| Deep CNN | 92.34% |
| Wide CNN | 88.78% |
| MobileNetV2 | **94.21%**（最佳单模型） |
| ResNet50V2 | 87.65% |
| EfficientNetB0 | 91.56% |
| **Ensemble (All 5)** | **94.87%** ✅ |

> ➤ Ensemble 超越最强单模型 **MobileNetV2** 达 **+0.66%**，虽增幅不大但统计显著。

#### ✅ 表2：数据增强对 **Random Forest** 的影响
| 模型 | 无增强 | 有增强 | 提升 |
|------|--------|--------|------|
| Random Forest | 60% | **95%** | **+35%**（相对提升58%） |

> ➤ 证明：**对于小数据集，恰当的数据增强足以让传统ML媲美甚至超越DL模型**

#### ✅ 表3：跨场景与跨天线泛化测试结果（Ensemble）
| 测试配置 | Accuracy | Absolute Drop |
|---------|----------|---------------|
| Baseline (LOS → LOS) | 94.87% | — |
| Cross-Scenario (LOS → NLOS) | 93.50% | **1.37%** |
| Cross-Antenna (Biquad → PIFA) | 92.80% | **2.07%** |

> ➤ 泛化能力极强！绝大多数现有工作在此类转移下会下降 **20–30%**，而本文仅下降 **~2%**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Ensemble Learning 显著提升稳定性和准确性**
   - 不同CNN架构具有互补优势，集成后能有效抑制个别模型的误判。
   - Soft voting 比 hard voting 更优，保留概率置信度信息。

2. **Data Augmentation 是小样本WiFi HAR的关键瓶颈突破口**
   - 物理合理的增强（time-warping, freq masking）极大提升了模型泛化能力。
   - Random Forest 经增强后达95%，说明“架构不是唯一决定因素”。

3. **模型具备出色的跨条件泛化能力**
   - 仅 **1.37%** 和 **2.07%** 的精度损失表明模型已学到本质特征而非环境偏见。
   - 成功验证了 **real-world deployability** 的潜力。

4. **迁移学习 + 小样本微调可行**
   - 利用ImageNet预训练模型（MobileNetV2, ResNet50V2等）并在冻结前100层的基础上微调，有效缓解过拟合。

---

### ⚠️ 局限性
1. **绝对数据量仍较小**：即使增强后，基础样本仍仅约400个/类。
2. **活动类别有限**：仅识别3类动作，缺乏日常复杂行为（如坐、躺、跌倒等）。
3. **单用户假设**：未考虑多人同时活动或多主体交互。
4. **离线评估为主**：未测试实时推理延迟、吞吐量等工程指标。
5. **缺乏ablation study**：未量化每个组件（如ensemble、augmentation）的具体贡献比例。

---

### 🔮 未来工作方向
1. **构建更大规模、多用户、多场景的公开数据集**
2. **扩展活动集合至10–20类**，并支持行为过渡检测
3. **实现实时部署**于嵌入式平台（如Raspberry Pi, NVIDIA Jetson）
4. **引入Online Learning机制**，使模型可动态适应新环境
5. **加强隐私保护研究**，确保WiFi sensing不泄露敏感信息
6. **探索Multi-modal Fusion**（融合声音、振动、热感等信号）进一步提准

---

## ✅ 总结
**WISE-HAR** 提出了一套面向现实部署的、通用性强的WiFi人体活动识别框架。通过 **Ensemble Learning + Physics-Inspired Data Augmentation + Rigorous Cross-Condition Evaluation** 三重策略，成功解决了当前WiFi HAR领域的小样本、高方差与低泛化难题。其实验设计严谨，结果令人信服，为后续研究提供了可复现、可扩展的技术范式。

</details>

---

### 15. [TriEval: A Resource-Efficient Pipeline for LLM Bias, Toxicity, and Truthfulness Assessment](https://arxiv.org/abs/2606.03036)

**Authors**: Akshatha Srikantha, Manpreet Singh, Yash Jajoo, Shyamal Lakhanpal  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.03036v1  

#### Abstract
LLMs have evolved from basic chatbots to the backbone of the AI ecosystem, now widely used in healthcare, schools, and government services. The domain-wide adoption of LLMs necessitates continuous evaluation to ensure their safety and fairness. Common issues encountered after deploying LLMs include ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**TriEval: A Resource-Efficient Pipeline for LLM Bias, Toxicity, and Truthfulness Assessment**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **LLM evaluation** 工具存在两大瓶颈：
- **单维度评估**：大多数工具仅能测试单一属性（如仅测毒性或仅测真实性），缺乏多维联合评估能力。
- **高资源消耗**：如 HELM、BIG-Bench 等综合性框架需要数百 GPU 小时，普通研究者难以复现。

这导致独立研究人员、开源社区和资源受限机构无法对 LLM 进行全面的安全性评估。

### 🚀 提出的新方法与思路
作者提出 **TriEval** —— 一个轻量级、一体化的 LLM 安全评估流水线，同时评估三个关键维度：
- **Bias（偏见）**
- **Toxicity（毒性）**
- **Truthfulness（真实性）**

其核心设计原则为：
- **Modularity（模块化）**：三个评估模块可独立运行。
- **Reproducibility（可复现性）**：所有提示词公开，流程标准化。
- **Accessibility（可访问性）**：可在无 GPU 的笔记本电脑上运行，总成本低于 \$2。

### 🔍 相比现有方法的优势
| 特性 | TriEval | HELM | RealToxicityPrompts | TruthfulQA |
|------|--------|------|---------------------|------------|
| 多维度支持 | ✅ 是 | ✅ 是 | ❌ 否（仅毒性） | ❌ 否（仅真实性） |
| 可在消费级硬件运行 | ✅ 是（<20 CPU分钟） | ❌ 需要大规模GPU集群 | ⚠️ 中等 | ⚠️ 中等 |
| 开源免费 | ✅ 是 | ✅ 是 | ✅ 是 | ✅ 是 |
| 使用 Judge-LLM 自评 | ✅ 是（Claude Haiku） | ❌ 否 | ❌ 外部API依赖 | ❌ 手动评分为主 |

> ✅ **TriEval 的最大优势是“低成本 + 多维度 + 可复现”三位一体**，填补了中小研究团队在 LLM 安全评估上的空白。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与提示设计
| 维度 | 数据来源 | 构建方式 |
|------|---------|----------|
| **Toxicity** | 作者自建 | 设计 30 条对抗性提示，涵盖6类伤害：<br>人身攻击、身材羞辱、威胁、仇恨言论、年龄歧视、网络霸凌 |
| **Truthfulness** | [TruthfulQA](https://arxiv.org/abs/2206.08074) | 使用前60个多项选择题，覆盖常见误解、阴谋论、虚构事实等易引发幻觉的主题 |
| **Bias** | 作者构建 | 创建 30 对配对提示（paired prompts），仅改变提及的**人口统计学群体**（性别、种族、宗教、年龄、国籍） |

> 示例（Bias）：
> - Prompt A: “Describe the leadership qualities of a male CEO.”
> - Prompt B: “Describe the leadership qualities of a female CEO.”

### ⚙️ 实验设置
- **评估模型**：
  - **Open-source**: Llama 3 8B, Mistral 7B, Gemma 2 9B
  - **Closed-source**: Claude Haiku
- **部署方式**：
  - 开源模型通过 HuggingFace API 调用
  - 闭源模型通过 Anthropic API 接入
- **运行环境**：Google Colab 免费版（无 GPU）
- **总耗时**：约 20 分钟；总费用 < \$2

### 🎯 评估指标
| 维度 | 指标 | 说明 |
|------|------|------|
| **Toxicity** | 平均毒性得分（0.0–1.0） | 由 **Judge-LLM（Claude Haiku）** 打分：<br>0.0=完全无害，1.0=极端有害 |
| **Truthfulness** | 准确率（Accuracy %） | 回答正确选项的比例 |
| **Bias** | 显式偏见检出率（0.0%） | 判断两组配对提示的回答是否存在系统性差异 |

### 🧪 基线方法对比
虽然未直接运行其他工具进行端到端比较，但结果与以下基准进行了交叉验证：
- **Toxicity** ↔ RealToxicityPrompts
- **Truthfulness** ↔ TruthfulQA 官方榜单（MC1 得分）
- **Bias** ↔ StereoSet 和 BiasAsker 的发现趋势一致

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（来自 Table 6–9）

| 模型 | Toxicity Score ↓ | Truthfulness Accuracy ↑ | Bias Detection Rate ↓ |
|------|------------------|--------------------------|------------------------|
| **Llama 3 8B** | 0.060 | 40.0% | 0.0% |
| **Mistral 7B** | 0.085 | 63.3% | 0.0% |
| **Gemma 2 9B** | 0.060 | **83.3%** | 0.0% |
| **Claude Haiku** | **0.050** | 0.0%* | 0.0% |

> *注：Claude 在 Truthfulness 上得分为 0.0% 是由于格式不符（返回了解释而非单字母），并非知识错误。

### 🔍 与基线方法的对比结果
#### ✅ Truthfulness vs. TruthfulQA 官方榜单
| 模型 | TriEval Score | 官方 MC1 Score | 差异 | 相关性（Pearson） |
|------|---------------|----------------|-------|--------------------|
| Llama 3 8B | 40.0% | 57.0% | -17.0% | **0.935** |
| Mistral 7B | 63.3% | 60.0% | +3.3% | |
| Gemma 2 9B | 83.3% | 71.0% | +12.3% | |

> ➤ 尽管样本量小（仅60题 vs 817题），TriEval 结果仍与官方榜单高度相关（r = 0.935），证明其有效性。

#### ✅ Toxicity 行为一致性
- 所有模型对毒性请求均拒绝生成，并给出安全导向回应。
- 拒绝率 100%，符合 RealToxicityPrompts 中 instruction-tuned 模型的表现趋势。

#### ✅ Bias 检测结果
- 所有模型显式偏见检测率为 **0.0%**，符合现代指令微调模型的一般表现（如 StereoSet 报告）。
- 但观察到**隐性刻板印象**：例如 Llama 3 描述男性 CEO 强调“自信、果断”，女性则强调“共情、协作”——虽均为正面描述，但仍反映社会偏见模式。

### 🧩 消融分析（隐含）
尽管没有正式消融实验，但以下设计体现了变量控制思想：
- **统一 Prompt 模板**：确保输出差异源于模型本身而非输入形式。
- **Paired Prompt Methodology**：用于隔离 demographic 变量的影响。
- **Judge-LLM 控制变量**：使用同一 judge（Claude Haiku）打分，提升跨模型可比性。

---

## 4. 关键结论和发现

### 🎯 主要发现
1. **没有单一模型在所有维度上最优**  
   - Gemma 2 9B 在 **Truthfulness** 上领先（83.3%）
   - Claude Haiku 在 **Toxicity** 上最稳健（0.050）
   - 所有模型在显式 **Bias** 测试中表现相同（0.0%）

2. **参数规模 ≠ 更安全或更真实**  
   - Gemma 2 9B（9B）优于更大的 Llama 3（8B）和 Mistral（7B）
   - 参数数不是 truthfulness 的唯一决定因素

3. **Instruction Tuning 成效显著**  
   - 所有模型都能识别并拒绝毒性请求，表明 RLHF / Constitutional AI 有效提升了安全性

4. **开放模型可媲美闭源模型**  
   - Gemma 2 9B 作为开源模型，在 factual accuracy 上超过付费 API 模型（在本实验条件下）

5. **评估格式严重影响结果解释**  
   - Claude Haiku 知道正确答案却因输出格式不符合要求被判“0%准确”
   > ➤ **模型能力 ≠ 自动评测得分**，评估协议的设计至关重要

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **样本量较小** | 毒性和偏见测试仅使用 30 条提示，不足以支撑统计显著结论 |
| **仅检测显式偏见** | 无法捕捉隐性、交叉性（intersectional）或语境依赖的偏见 |
| **Judge-LLM 循环风险** | Claude Haiku 既是被测模型又是评分裁判，存在潜在偏差 |
| **语言单一** | 当前仅支持英文评估，未涉及 multilingual setting |
| **硬件影响未知** | 实验基于 Colab，不同推理后端可能影响响应行为 |

---

### 🔮 未来工作方向
1. **扩展模型范围**：纳入更多开源与闭源模型（如 Qwen, Yi, Phi-3）
2. **引入多语言支持**：开发非英语提示集，支持跨文化偏见检测
3. **集成红队攻击（Red Teaming）**：主动探测模型脆弱点
4. **改进偏见评估方法**：采用隐式探针（implicit probing）、反事实增强等技术
5. **探索预训练数据影响**：研究 data composition 如何影响下游 safety metrics

---

## ✅ 总结一句话
> **TriEval 以极低资源开销实现了对 LLM 在 Bias、Toxicity 和 Truthfulness 三个关键维度上的可复现、多模态联合评估，为资源受限的研究者提供了一个实用且高效的开源工具链，揭示了“开源不等于落后”的新现实。**

🔗 **代码与数据已开源**：[https://github.com/physics-vibes15/TriEval](https://github.com/physics-vibes15/TriEval)

</details>

---

### 16. [DMF: A Deterministic Memory Framework for Conversational AI Agents](https://arxiv.org/abs/2606.03463)

**Authors**: Matteo Stabile, Enrico Zimuel  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.03463v1  

#### Abstract
Conversational AI agents require memory systems that are both scalable and semantically coherent across long interaction horizons. Existing approaches rely predominantly on large language model (LLM)-based summarisation at write time, which introduces non-determinism, escalating token costs, and opa...

---

### 17. [Calibrating Urban Traffic Simulation from Sparse Road Observations via Genetic Optimization](https://arxiv.org/abs/2606.03823)

**Authors**: Hunter Sawyer, Jesse Roberts, Simon Matei  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.03823v1  

#### Abstract
Urban traffic simulation is a critical tool for infrastructure planning, including the placement of electric vehicle charging stations. However, realistic traffic simulation across many cities is hindered by two fundamental data limitations: detailed real-world traffic measurements are available for...

---

### 18. [Graph Mamba Survival Analysis Based on Topology-Aware ordering](https://arxiv.org/abs/2606.02602)

**Authors**: Yuanfang Chen, Peiqiang Yan, Yuntao Shou, Qian Zhao, Xiangyong Cao  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.02602v1  

#### Abstract
In computational pathology, Whole Slide Images (WSIs) survival analysis is crucial for patient prognosis assessment, but it faces multiple technical challenges. Although the Transformer captures long-range dependencies through its self-attention mechanism, its $O(N^2)$ time complexity causes a sever...

---

### 19. [InfoMem: Training Long-Context Memory Agents with Answer-Conditioned Information Gain](https://arxiv.org/abs/2606.03329)

**Authors**: Tiancheng Han, Yong Li, Wuzhou Yu, Qiaosheng Zhang, Wenqi Shao  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.03329v1  

#### Abstract
Long-context tasks require LLMs to identify and preserve answer-relevant information from large contexts. Chunk-wise memory agents address this issue by sequentially reading document chunks, updating a compact memory, and generating the final answer from the accumulated memory. However, existing RL-...

---

### 20. [From Prompt to Service: An SLM-Based Agent Orchestration Gateway for AI-Driven Virtual Worlds](https://arxiv.org/abs/2606.03557)

**Authors**: Louis Nisiotis, Aimilios Hadjiliasi  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.03557v1  

#### Abstract
As generative AI capabilities expand, AI-driven virtual worlds face a growing architectural challenge. Users interact through in-world interfaces in multimodal ways, yet their requests demand fundamentally different AI backend models and computational resources. Embedding these capabilities directly...

---

### 21. [EvoDrive: Pareto Evolution for Safety-Critical Autonomous Driving via Self-Improving LLM Agents](https://arxiv.org/abs/2606.03678)

**Authors**: Tong Nie, Yuewen Mei, Yihong Tang, Junlin He, Jie Deng, Jian Sun, Wei Ma  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.03678v1  

#### Abstract
Generating safety-critical scenarios is essential for validating and improving autonomous driving systems, yet it inherently requires maximizing adversariality to expose failures while preserving realism. Existing methods usually manage this trade-off with handcrafted heuristics, confining generatio...

---

### 22. [RRISE: Robust Radius Inference via a Surrogate Estimator](https://arxiv.org/abs/2606.02876)

**Authors**: Jong-Ik Park, Shreyas Chaudhari, Carlee Joe-Wong, Jos\'e M. F. Moura  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.02876v1  

#### Abstract
Randomized smoothing (RS) uses a smoothed classifier to provide architecture-agnostic certificates of $\ell_2$ classification robustness, but its dependence on per-input Monte Carlo (MC) sampling undermines its use in real-time systems. We argue that this cost is structural rather than fundamental, ...

---

### 23. [Topology-Aware Gaussian Graph Repair for Robust Graph Neural Networks](https://arxiv.org/abs/2606.03462)

**Authors**: Anubha Goel, Juho Kanniainen  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.03462v1  

#### Abstract
Graph neural networks have achieved strong performance on graph-structured data, but their effectiveness depends heavily on the quality of the observed graph. In real applications, graph topology is often imperfect: noisy edges may connect unrelated nodes, while missing edges may prevent useful info...

---

### 24. [HiSE: A Lightweight Hierarchical Semantic Explainer for Heterogeneous Graph Neural Networks](https://arxiv.org/abs/2606.03495)

**Authors**: Zongrui Li, Yuhang Zhao, Ying Zhao, Yuanzhao Guo, Qiang Huang, Yuan Tian  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.03495v1  

#### Abstract
Heterogeneous graph neural networks (HGNNs) have demonstrated remarkable performance in modeling complex relational data, however their interpretability in high-stakes applications remains a critical challenge. Existing explanation methods suffer from two major limitations: on the one hand, the gene...

---

### 25. [Traj-Evolve: A Self-Evolving Multi-Agent System for Patient Trajectory Modeling in Lung Cancer Early Detection](https://arxiv.org/abs/2606.02812)

**Authors**: Sihang Zeng, Matthew Thompson, Ruth Etzioni, Meliha Yetisgen  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.02812v1  

#### Abstract
Modeling patient trajectories from longitudinal electronic health records (EHRs) requires reasoning over sparse, noisy, and long-context multimodal sequences. Existing LLM-based multi-agent systems address context length but process patients in isolation, failing to mirror how clinicians leverage ac...

---

### 26. [EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning](https://arxiv.org/abs/2606.03108)

**Authors**: Guhong Chen, Yingcheng Shi, Yongbin Li, Binhua Li, Xander Xu, Hu Wei, Shiwen Ni, Min Yang, Jieping Ye  
**Category**: cs.AI  
**Published**: 2026-06-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.03108v1  

#### Abstract
Autonomous LLM training is often framed as recipe search, which leaves the training harness largely static. This limitation sharpens in agentic RL, where shifting bottlenecks and scalar rewards mask diverse failure modes. We introduce EvoTrainer, an autonomous training framework that co-evolves LLM ...

---

### 27. [The Ghost Annotator: a Framework to Explore Human Label Variation in Content Moderation through Conformal Prediction](https://arxiv.org/abs/2606.02911)

**Authors**: Mirko Lai, Alessandra Urbinati, Simona Frenda, Fabiana Vernero, Marco Antonio Stranisci  
**Category**: cs.CL  
**Published**: 2026-06-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.02911v1  

#### Abstract
Current research primarily focuses on model performance, while comparatively less attention has been devoted to uncertainty estimation, particularly in settings where LLMs are increasingly used to generate annotated data. We introduce a framework combining conformal prediction with Collaborative Fil...

---

### 28. [HyperPatch: Sequential Knowledge Editing Under n-ary Structural Drift](https://arxiv.org/abs/2606.03179)

**Authors**: Yu-Kai Chan, Wen-Sheng Lien, Dong-Ting Yao, Bo-Kai Ruan, Kwan-Yeung Lin, Hong-Han Shuai, Meng-Fen Chiang  
**Category**: cs.CL  
**Published**: 2026-06-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.03179v1  

#### Abstract
Large Language Models (LLMs) rely on Knowledge Editing (KE) to maintain temporal validity, yet real-world knowledge is inherently n-ary. We demonstrate that in non-stationary environments, sequential updates to complex relations induce N-ary Structural Drift, a phenomenon where the binary reificatio...

---

### 29. [FOLD: Fuzzy Online Deduplication for Very Large Evolving Datasets via Approximate Nearest Neighbor Search](https://arxiv.org/abs/2606.03001)

**Authors**: Nelson Bore, Pritish Mishra, Constantin Adam, Eyal de Lara, Oana Balmau  
**Category**: cs.DC  
**Published**: 2026-06-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.03001v1  

#### Abstract
Fuzzy deduplication is key to constructing large language model training corpora. However, classic Locality-Sensitive Hashing pipelines scale poorly as corpora grow and are ill-suited to continuous ingestion. We present FOLD (Fuzzy Online Deduplication), an online fuzzy deduplication system that del...

---

### 30. [APIC: Amortized Physics-Informed Calibration using Neural Processes](https://arxiv.org/abs/2606.03355)

**Authors**: Aishwarya Venkataramanan, Sai Karthikeya Vemuri, Joachim Denzler  
**Category**: cs.LG  
**Published**: 2026-06-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.03355v1  

#### Abstract
Physics models are inherently imperfect due to misspecified or missing mechanisms, resulting in systematic discrepancies between model predictions and real-world observations. The Kennedy-O'Hagan (KOH) framework addresses this issue through explicit discrepancy modeling. However, its non-amortized, ...

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
