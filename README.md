# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-02 06:39:28 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [The Auton Agentic AI Framework](https://arxiv.org/abs/2602.23720)

**Authors**: Sheng Cao, Zhao Chang, Chang Li, Hannan Li, Liyao Fu, Ji Tang  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.23720v1  

#### Abstract
The field of Artificial Intelligence is undergoing a transition from Generative AI -- probabilistic generation of text and images -- to Agentic AI, in which autonomous systems execute actions within external environments on behalf of users. This transition exposes a fundamental architectural mismatc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《The Auton Agentic AI Framework》论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决当前 **Agentic AI**（自主智能体系统）在企业级部署中面临的三大核心挑战：

- **集成悖论（Integration Paradox）**：大型语言模型（LLM）生成的是**概率性、非结构化输出**，而企业后端系统（如数据库、API、云服务）要求**确定性、符合Schema的输入**。这种不匹配导致系统不稳定和失败率高。
- **生态碎片化（Ecosystem Fragmentation）**：现有框架（如 LangChain、AutoGen）将 agent 定义与运行时耦合，造成**技术债、厂商锁定（vendor lock-in）、跨语言不可移植**等问题。
- **治理与安全性不足**：多数安全机制依赖于**事后过滤（post-hoc filtering）**，而非从设计上排除危险行为，难以满足合规审计需求。

---

### 提出的新方法与架构创新

论文提出了 **Auton Agentic AI Framework**，其核心是通过**声明式架构**实现 agent 的标准化定义、执行与治理。主要创新点如下：

#### ✅ 1. **认知蓝图（Cognitive Blueprint）与运行时引擎（Runtime Engine）的分离**
- 将 agent 定义为一个**与语言无关的声明式配置文件（AgenticFormat）**，采用 YAML/JSON 表达。
- 运行时（如 `agentic-py`, `agentic-java`）负责加载并实例化该蓝图。
- 类比于 **Infrastructure-as-Code**（如 Kubernetes/Terraform），实现了 agent 的版本控制、可审计性和跨平台可移植性。

> **优势**：避免厂商锁定；支持 Python 开发 → Java 生产部署；便于合规审查。

#### ✅ 2. **AgenticFormat 标准：统一的 Agent 配置协议**
- 定义了标准化 schema，涵盖：
  - 接口（input/output contracts）
  - 工具绑定（tool bindings via MCP）
  - 内存配置
  - 安全约束（constraint manifold）
- 输出强制绑定到 **Pydantic / JSON Schema**，确保结构化、类型安全。

> **优势**：解决“集成悖论”——将 LLM 的随机输出转化为确定性接口。

#### ✅ 3. **约束流形（Constraint Manifold）形式化安全机制**
- 不再使用事后过滤，而是通过**策略投影（policy projection）** 在生成过程中直接屏蔽非法动作。
- 使用 token-level masking，在 autoregressive 生成时将可能导致违规的 token logits 设为 `-∞`。
- 支持形式化表达权限边界（如只读访问、PII 禁止外泄等）。

> **优势**：从根本上防止越权操作，提升系统鲁棒性与合规性。

#### ✅ 4. **分层记忆架构（Hierarchical Memory Architecture）**
- 受生物记忆启发，构建两级存储：
  - **短期记忆（Event Stream）**：会话内完整事件日志（用户输入、工具调用等）。
  - **长期记忆（Knowledge Base）**：分为语义、情景、程序三类记忆。
- 引入 **Reflector-Driven Consolidation Protocol**：
  - 自动分割事件流 → 提取关键洞察 → 向量化存储 → 上下文压缩。

> **优势**：突破 context window 限制，实现经验积累与跨会话学习。

#### ✅ 5. **三层次自进化框架（Three-Level Self-Evolution）**
| 层级 | 名称 | 方法 | 特点 |
|------|------|------|------|
| Level 1 | In-Context Evolution | 失败后由 Reflector 生成“Lesson”存入长期记忆 | 无需训练，快速适应 |
| Level 2 | Self-Taught Reasoning (STaR) | 对成功轨迹进行 SFT 微调 | 将复杂推理内化为直觉 |
| Level 3 | Agentic Reinforcement Learning | 使用 GRPO/PPO 在多步 POMDP 中探索新策略 | 发现超越人类设计的优化路径 |

> **优势**：形成“运营即训练”的正向循环，持续提升 agent 能力。

#### ✅ 6. **运行时效率优化（Runtime Efficiency）**
- **Cognitive Map-Reduce**：将执行计划解析为 DAG，对独立步骤并行执行。
- **Speculative Execution**：预测慢速工具返回值，提前进行后续推理（类似 CPU 分支预测）。
- **Dynamic Context Pruning**：基于注意力分数动态淘汰低价值 token，维持 KV Cache 规模稳定。

> **优势**：显著降低端到端延迟，适用于实时场景（如广告竞价、事务处理）。

---

### 相比现有方法的优势

| 维度 | 传统方法（LangChain/AutoGen） | Auton Framework |
|------|-------------------------------|------------------|
| 架构模式 | 命令式代码嵌入逻辑 | 声明式配置（agent-as-configuration） |
| 可移植性 | 锁定 Python & 框架 API | 跨语言（Java/Python）、跨平台 |
| 安全机制 | 事后过滤（脆弱） | 策略投影 + 约束流形（构造级安全） |
| 记忆管理 | 上下文拼接（易溢出） | 分层压缩 + 向量检索 |
| 演进能力 | 手动调参/提示工程 | 自动三阶进化（in-context → SFT → RL） |
| 性能 | 串行执行，延迟高 | 并行图执行 + 投机推理 |

---

## 2. 核心实验方法和设置

> ⚠️ **注意**：本文是一篇**系统架构论文**，侧重理论建模与工程设计，并未提供传统意义上的“实验结果表格”。文中没有列出具体使用的 benchmark 数据集或定量消融研究，而是通过形式化建模、架构分析与案例说明来论证有效性。

尽管如此，仍可归纳其验证方式与评估逻辑：

### 实验设置与评估范式

#### 📌 形式化建模作为“理论实验”
- 将 agent 建模为增强型 **Partially Observable Markov Decision Process (POMDP)**，引入 **Latent Reasoning Space Z**。
- 定义 **Factorized Policy Architecture**：  
  $$
  \pi_{\text{action}}(a_t | m_t, z_t), \quad z_t \sim \pi_{\text{reason}}(z_t | m_t)
  $$
  强制“先思考再行动”，提升决策质量。

#### 📌 模拟任务与工业用例驱动设计
虽然未公布标准测试集，但全文围绕多个典型企业级任务展开设计验证：
- 数据分析师 agent 查询数据库（需生成合法 SQL）
- 客户支持 agent 接入 Salesforce CRM
- 代码评审 agent 输出结构化 review 结果供 CI/CD 消费
- 实时广告出价系统中的低延迟 Java 微服务集成

这些用例用于反推架构需求，并指导 AgenticFormat 的字段设计。

#### 📌 评估指标（隐含）
| 指标类别 | 具体指标 |
|--------|---------|
| 功能正确性 | 输出是否符合 schema？能否被下游系统消费？ |
| 安全性 | 是否杜绝越权操作？是否满足合规要求？ |
| 延迟性能 | 端到端响应时间（尤其多步 workflow） |
| 可维护性 | 是否支持版本控制、diff、审计？ |
| 可扩展性 | 是否支持新工具接入（via MCP）？ |

#### 📌 基线对比（概念层面）
| 基线方法 | 缺陷 |
|--------|------|
| LangChain / AutoGen | 定义与运行时耦合，无法跨语言迁移 |
| Prompt Engineering + Regex Parsing | 易失效，难维护 |
| Post-hoc Filter + Retry | 无法根除错误，增加延迟 |
| Full Context Append | 上下文爆炸，成本指数增长 |

---

## 3. 主要实验结果和性能指标

> ❌ **无明确数值结果或图表发布**

由于本论文属于**系统白皮书性质**，聚焦于提出完整架构而非展示单一模块性能，因此**未报告具体的 accuracy、latency 数值或 benchmark 对比表**。

但可通过文本描述提取以下**定性性能优势**：

### ✅ 性能改进方向（基于架构设计推导）

| 优化项 | 效果描述 |
|-------|----------|
| **Cognitive Map-Reduce** | 将总耗时从 $\sum_i L_i$ 降至关键路径长度 $\max_{\text{path}} \sum L_{\text{node}}$，对于宽任务图有显著加速 |
| **Speculative Execution** | 可隐藏部分网络延迟（如 DB 查询、API 调用），尤其当输出可预测时命中率高 |
| **Dynamic Context Pruning** | KV Cache 成本从 $O(N^2)$ 控制为近似常数，支持长周期任务 |
| **Constraint Manifold** | 安全违规发生率为零（理论上），无需重试或人工干预 |

### ✅ 案例说明（间接体现效果）
- 一个 Python 中开发的 customer support agent，可通过 `agentic-java` SDK 部署至低延迟 JVM 服务中，无需重写逻辑。
- 当 agent 多次因“日期格式错误”调用 API 失败后，Reflector 自动生成 lesson：“Use ISO 8601 format for date fields”，后续自动纠正。
- 在执行“查询苹果股价 + 查询微软情绪 + 对比分析”任务时，两个查询并发执行，节省约 50% 时间。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **Agentic AI 必须走向声明式架构**  
   将 agent 定义为数据（而非代码），才能实现真正的可复用、可审计、可治理。

2. **认知蓝图与运行时必须解耦**  
   类似 IaC 范式，是打破生态碎片化的唯一可行路径。

3. **安全应内置于生成过程**  
   “Policy Projection over Filtering” 是企业级部署的安全底线。

4. **持久化记忆需仿生设计**  
   借鉴人脑的记忆巩固机制，才能实现高效、可持续的知识沉淀。

5. **agent 应具备自我进化能力**  
   从 in-context adaptation 到 RL，构建闭环的学习飞轮。

6. **性能瓶颈在于调度而非模型本身**  
   并行图执行 + 投机推理 + 注意力剪枝 是实现实时性的关键技术组合。

---

### ⚠️ 方法的局限性（文中未明说，可合理推断）

| 局限性 | 说明 |
|--------|------|
| **依赖高质量 MCP 工具连接器** | 若外部工具缺乏标准化接口（如老旧内部系统），集成难度仍大 |
| **Reflector 的可靠性假设较强** | 自动提取“lesson”需要高精度诊断能力，否则可能传播错误因果 |
| **初始冷启动问题** | 新 agent 缺乏历史数据，难以立即发挥记忆与进化优势 |
| **形式化约束表达复杂度** | Constraint Manifold 的 predicate 设计需要领域专家参与 |
| **未开源 SDK 全部实现** | 当前仅提及 `agentic-py` 和 `agentic-java`，生态建设尚早期 |

---

### 🔮 未来工作方向（来自 Section 9）

1. **推动 AgenticFormat 成为开放标准**
   - 开源“Agent Cards”规范，鼓励社区共享可复用 agent 配置。

2. **深化与 Model Context Protocol (MCP) 的协同**
   - 构建统一的工具市场（Tool Marketplace），支持即插即用。

3. **发展自动化 Reflector 能力**
   - 利用更强 LLM 实现全自动失败归因与 lesson 生成。

4. **构建端到端训练流水线**
   - 支持大规模 trajectory 收集、reward modeling 与分布式 RL 训练。

5. **支持多 agent 协作场景**
   - 扩展框架以管理 agent 团队间的通信、分工与冲突解决。

---

## 总结

| 维度 | 内容 |
|------|------|
| **论文定位** | 系统架构论文，提出企业级 Agentic AI 的工程化解决方案 |
| **核心思想** | 声明式定义（AgenticFormat） + 解耦运行时 + 形式化安全 + 自进化机制 |
| **最大亮点** | 将 agent 视为“可编程的数据单元”，实现跨语言、可审计、可持续演进的企业级智能体系统 |
| **适用场景** | 金融、医疗、基础设施、广告等对可靠性、安全性、性能要求高的行业 |
| **现实意义** | 为 Agentic AI 从“玩具”走向“生产系统”提供了完整的架构蓝图 |

> 💡 **一句话总结**：  
> Auton Framework 提出了一个面向企业的、类比于 Kubernetes 的 **“Agent-as-Configuration” 架构范式**，通过声明式蓝图、约束流形、分层记忆与自进化机制，系统性解决了 Agentic AI 在集成、安全、效率与演化方面的根本挑战。

</details>

---

### 2. [Bi-level RL-Heuristic Optimization for Real-world Winter Road Maintenance](https://arxiv.org/abs/2602.24097)

**Authors**: Yue Xie, Zizhen Xu, William Beazley, Fumiya Iida  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.24097v1  

#### Abstract
Winter road maintenance is critical for ensuring public safety and reducing environmental impacts, yet existing methods struggle to manage large-scale routing problems effectively and mostly reply on human decision. This study presents a novel, scalable bi-level optimization framework, validated on ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Bi-level RL-Heuristic Optimization for Real-world Winter Road Maintenance

## 1. 论文的主要贡献和创新点

### 解决的问题
该研究针对**大规模冬季道路维护中的路径规划难题**，特别是英国战略级公路网络（如 M25、M6、A1）中盐撒车的调度问题。传统方法依赖人工决策或简单启发式算法，在面对多仓库（multi-depot）、异构车队、复杂立交桥和严格时间约束时表现不佳，导致资源分配不均、响应延迟、碳排放高。

### 提出的新方法与创新点
提出了一种**可扩展的双层优化框架（bi-level optimization framework）**，结合强化学习（Reinforcement Learning, RL）与约束感知启发式算法：

- **上层（Upper Level）**：使用基于 **Proximal Policy Optimization (PPO)** 的 RL 代理进行**战略级决策**，将路网划分为多个集群，并从多个仓库中优化资源配置。
- **下层（Lower Level）**：在每个集群内求解一个**多目标车辆路径问题（Multi-objective VRP）**，最小化最大行驶时间（makespan）和总碳排放（carbon emissions），采用改进的最近邻（Nearest Neighbor, NN）启发式算法确保可行性。

该方法首次将 RL 用于真实世界大规模冬季维护任务的**高层结构化决策**，同时保留底层路由的可解释性和合规性。

### 相比现有方法的优势
- ✅ **可扩展性强**：能有效处理包含数万条边的真实路网，远超传统 CARP 或 WRMP 模型的能力范围。
- ✅ **兼顾效率与环保**：显式优化两个关键目标——完成时间和碳排放。
- ✅ **符合实际操作流程**：模拟了运营人员“先分区再排班”的决策逻辑，易于集成到现有系统。
- ✅ **无需改变底层调度机制**：仅调整上层分配策略，保留 NN 路由器的可审计性与安全性。
- ✅ **闭环反馈机制**：通过迭代优化实现持续改进，优于一次性静态分配。

---

## 2. 核心实验方法和设置

### 数据集
- 来源于 **UK National Highways** 的真实运营数据。
- 包含英国境内约 **60km × 60km** 区域的战略公路网络（EPSG:27700 坐标系）。
- 路网来自 **OpenStreetMap (OSM)**，并结合历史 GPS 轨迹进行地图匹配以识别需处理路段。
- 总计压缩后为 **37,007 个节点、71,505 条有向边**，其中 **1,515 条边（543.5km）需要撒盐处理**，共 **1208.7 lane-km**。
- 包含三个服务仓库：Misterton、Pytchley 和 Rothersthorpe。
- 集成局部道路网络用于车辆转向和通行。

### 实验设置与评估指标
| 类别 | 内容 |
|------|------|
| **目标函数** | <ul><li>`Z1`：最小化最大单辆车完成时间（makespan，单位：分钟）</li><li>`Z2`：最小化车队总碳排放（kg CO₂e）</li></ul> |
| **约束条件** | <ul><li>每辆车最大行程 ≤ 630 km</li><li>每趟任务时长 ≤ 120 分钟</li><li>盐罐容量限制（166 km·lane）</li><li>必须返回原仓库（return-to-depot）</li><li>遵守单行道规则与速度限制</li></ul> |
| **计算平台** | 支持预班次（pre-shift）规划，支持 warm-start 初始化 |

### 基线方法对比
比较两种规划器：
1. **KDTree + NN**  
   - 静态 KDTree 最近仓库分配 + 最近邻路由
   - 单次执行，无策略优化
2. **KDTree-PPO + NN (10 iterations)**  
   - 初始使用 KDTree 分配作为 warm-start
   - 后续通过 PPO 代理根据下层反馈迭代优化分配策略
   - 共运行 10 次交互迭代

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）
| 方法 | `Z1` (makespan, min) | `Z2` (emissions, kg CO₂) | NoV (Number of Vehicles) |
|------|------------------------|----------------------------|----------------------------|
| KDTree + NN | 122.14 | 3,386.63 | 20 |
| KDTree-PPO + NN | **118.81** | **3,220.95** | **28** |

> 注：NoV 增加是由于更精细的任务拆分以满足 120 分钟上限，反映更高的合规性而非低效。

### 与基线方法的对比结果
- 📉 **Makespan 减少 2.7%**（122.14 → 118.81 min）  
  尽管看似微小，但在所有车辆和高峰事件中累积效应显著，有助于释放运力应对突发热点。
- 📉 **碳排放减少约 4.9%**（3,386.63 → 3,220.95 kg CO₂）  
  表明死行程（deadheading）和怠速减少，环境效益可观。
- ✅ **帕累托改进（Pareto improvement）**：两个目标同时改善。
- ✅ **车队利用率稳定或降低**：尽管车辆使用次数略有上升（因合规拆分），但总行驶距离下降，运营负担减轻。

### 消融分析与过程观察（Figure 3）
- 大部分性能提升发生在前 **1–4 次迭代**，后续趋于收敛。
- PPO 快速捕捉最优分配模式，并逐步微调区域边界。
- 迭代过程中路线变得更紧凑、平衡，减少了跨区入侵（cross-depot incursions）和往返冗余（out-and-back stubs）。

---

## 4. 关键结论和发现

### 主要发现
- 双层 RL-Heuristic 架构能够在**不改变底层调度逻辑的前提下显著提升整体性能**。
- 上层 RL 仅负责“谁负责哪片区域”，即可驱动下层产生更优的微观路径。
- **紧凑且去中心化的责任区划分**是提高效率的关键因素。
- 方法在真实复杂路网中实现了：
  - 所有任务在 **<2 小时阈值内完成**
  - 工作负载均衡化
  - 显著降低碳足迹
  - 可审计、可复现的输出格式（CSV 路径记录、日志等）

### 方法的局限性
- 当前 PPO 策略仍基于较简单的状态特征（坐标、长度、限速、车道数、距仓库距离），未考虑天气动态或交通流变化。
- 未建模中途补给（reload logistics）或库存管理。
- 路由层仍依赖贪心启发式（NN），存在局部最优风险。
- 当前验证仅在一个地理区域内进行，泛化能力有待进一步测试。

### 未来工作方向
- 扩展至**全国尺度的战略公路网络**，验证其可扩展性极限。
- 引入**风暴演进模型与优先级队列**（storm progression & priority queues），实现动态响应。
- 整合**仓库库存与补给调度**（depot stock/reload logistics）。
- 探索**有限中途回调规划**（limited mid-shift replanning）能力。
- 将相同模块化模板推广至 National Highways 的其他区域。

---

> 🔍 **一句话总结**：本文提出了一种实用、可部署的 bi-level RL-Heuristic 框架，在真实大规模冬季道路维护场景中实现了**更低耗时、更低排放、更高合规性**的帕累托改进，展示了 AI 在现实交通物流中的直接应用价值。

</details>

---

### 3. [BTTackler: A Diagnosis-based Framework for Efficient Deep Learning Hyperparameter Optimization](https://arxiv.org/abs/2602.23630)

**Authors**: Zhongyi Pei, Zhiyao Cen, Yipeng Huang, Chen Wang, Lin Liu, Philip Yu, Mingsheng Long  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.23630v1  

#### Abstract
Hyperparameter optimization (HPO) is known to be costly in deep learning, especially when leveraging automated approaches. Most of the existing automated HPO methods are accuracy-based, i.e., accuracy metrics are used to guide the trials of different hyperparameter configurations amongst a specific ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BTTackler: A Diagnosis-based Framework for Efficient Deep Learning Hyperparameter Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
- **超参数优化**（Hyperparameter Optimization, HPO）在深度学习中计算成本高昂，尤其是自动化方法（如 Bayesian Optimization、Random Search）需要大量试验。
- 现有方法多为 **accuracy-based**，即依赖验证集上的准确率来判断试验好坏，但在训练早期，准确率无法反映严重的训练问题（如梯度消失、爆炸、loss震荡等）。
- 这导致许多“坏试验”（bad trials）持续运行，浪费大量计算资源，降低整体优化效率。

### 🚀 提出的新方法和新思路
- 提出 **BTTackler**（Bad Trial Tackler），首个将 **training diagnosis** 引入 HPO 的框架。
- 核心思想：通过设计一组可量化的 **Quality Indicators**（质量指标），在训练过程中实时检测潜在的训练问题，并对明显失败的试验进行 **early termination**（早停），从而节省资源用于更有希望的配置探索。
- 不同于传统的基于性能预测的早停策略（如 LCE、MSR），BTTackler 基于 **训练过程健康度诊断**，更早识别根本性训练故障。

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 RS、BO） | BTTackler |
|------|------------------------|-----------|
| 决策依据 | 准确率/损失值趋势 | 训练过程中的数值稳定性、梯度行为等底层信号 |
| 早停机制 | 基于性能比较（如低于中位数） | 基于训练问题诊断（如梯度爆炸、loss不降） |
| 效率 | 易被不稳定初期表现误导，误杀好配置 | 更保守且精准地终止明显失败试验，减少资源浪费 |
| 自动化程度 | 高 | 更高（无需人工干预即可自动诊断） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与模型架构
实验覆盖三种典型任务和 DNN 架构：

| 任务 | 数据集 | 模型 | 超参数数量 |
|------|--------|------|------------|
| `Cifar10CNN` | CIFAR-10 图像分类 | CNN（4卷积层 + 3全连接层） | 13个（3连续+5离散+5类别） |
| `Cifar10LSTM` | CIFAR-10 分类 | LSTM（循环神经网络） | 8个（3连续+1离散+4类别） |
| `Ex96Trans` | Exchange-Rate 时间序列预测 | Transformer | 8个（1连续+4离散+3类别） |

> 所有任务均设置较大的搜索空间以增加挑战性。

### ⚙️ 实验设置
- **硬件环境**：3台服务器，每台含 Intel Xeon 14核 CPU、384GB RAM、8块 NVIDIA TITAN X(Pascal) GPU
- **并发数**：统一设为 8
- **时间预算**：6小时
- **重复次数**：每个实验重复3次取平均值
- **实现平台**：基于开源 HPO 框架 **NNI** 实现 BTTackler

### 📊 评估指标
提出两个新指标以更公平衡量 HPO 方法的有效性：

| 指标 | 定义 | 用途 |
|------|------|------|
| **Top10 Hit Ratio (Top10HR)** | 在增强版方法与基线共同选出的前10个最佳试验中，增强方法贡献的比例 | 衡量性能优势（越高越好） |
| **Time-Saving for Baseline Accuracy (TSBA)** | 增强方法达到基线最佳准确率所需时间相比基线的节省比例 | 衡量效率提升（越高越好） |

---

## 3. 主要实验结果和性能指标

### 📈 性能对比结果（RQ1 & RQ2）

#### ✅ 准确性提升（Effectiveness）
- 在多数情况下，**BTTackler-enhanced 方法显著优于基线**。
- **平均 Top10HR 达到 72.25%**，远高于传统早停规则（ETRs）的 52.08%，说明其找到优质配置的能力更强。
- 特别是在复杂模型（如 Transformer）上提升明显。

#### ⏱️ 效率提升（Efficiency）
- **TSBA 平均达 40.33%**，意味着 BTTackler 只需不到 60% 的时间就能达到基线的最佳性能。
- 具体任务上的 TSBA：
  - `Cifar10CNN`: 16%
  - `Cifar10LSTM`: 47%
  - `Ex96Trans`: **58%**

> 复杂模型中效率增益更大，说明 BTTackler 对难训模型更具价值。

#### 📊 试验数量对比（Table 3）
- 在相同时间内，**BTTackler 能执行更多 trial**：
  - 例如在 `Ex96Trans` 上，Random-BTTackler 在6小时内完成 **486 次试验**，而原始 Random 仅完成 **307 次**（↑58.3%）。
- 更多试验 = 更大概率采样到最优配置。

### 🔬 消融实验与质量指标分析（RQ3）

#### 各 Quality Indicator 的触发频率（Table 5）
| 指标 | Cifar10CNN | Cifar10LSTM | Ex96Trans | 功能描述 |
|------|------------|-------------|-----------|----------|
| **ERG**（梯度指数衰减） | 317 | 296 | 11 | 检测梯度消失早期迹象 |
| **NMG**（无进一步增益） | 93 | 134 | **35** | 判断训练已收敛，建议停止 |
| **LAR**（低激活率） | 62 | 105 | 3 | 检测 ReLU 死亡等问题 |
| **PLC**（被动损失变化） | 24 | 1 | — | 检测初期 loss 不下降 |
| **AGV/EAG/ULC** | 少量触发 | 少量触发 | 少量触发 | 检测异常梯度、loss波动等 |

> 发现：不同任务中主导的质量指标不同，组合使用多个指标可最大化覆盖率。

#### 关键发现
- **质量指标协同作用显著**：单一指标难以覆盖所有坏试验，组合使用效果最佳。
- **BTTackler 的 overhead 极低**：额外开销 < 5%，得益于多线程并行实现。
- **误判风险可控**：因采用保守阈值，极少将好配置误判为坏试验。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **BTTackler 是首个将 training diagnosis 系统引入 HPO 的框架**，实现了从“看结果”到“看过程”的范式转变。
2. 通过引入 **7个精心设计的 Quality Indicators**，能够有效识别训练中的根本性问题（如梯度异常、loss震荡、激活不足等），并在早期终止失败试验。
3. 实验表明：
   - **平均节省 40.33% 时间** 即可达基线最佳性能；
   - **单位时间内多执行 44.5% 的 top-10 级别试验**，大幅提升搜索效率。
4. 开源实现已发布：[GitHub - thuml/BTTackler](https://github.com/thuml/BTTackler)，支持与主流 HPO 框架集成。

### ⚠️ 局限性
- 当前 Quality Indicators 设计偏保守，可能漏检一些轻微但累积性的训练问题。
- 阈值依赖经验设定，虽具通用性，但在极端任务中需手动调参。
- 对极短训练任务收益有限，更适合长周期、高成本的 HPO 场景。

### 🔮 未来工作方向
1. **理论突破**：研究权重/梯度分布演化规律，建立更坚实的 training diagnosis 理论基础。
2. **自适应指标设计**：开发可学习或动态调整的 Quality Indicators，提高泛化能力。
3. **控制搜索空间规模**：结合诊断反馈动态缩小无效区域，进一步提升 HPO 效率。
4. **扩展至 NAS**：将该框架应用于 Neural Architecture Search，诊断架构本身的可训练性。

---

> **一句话总结**：  
> BTTackler 通过“诊断训练健康度”而非“等待最终性能”，实现了更高效、更智能的 HPO，是自动化机器学习领域的一次重要范式升级。

</details>

---

### 4. [GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks](https://arxiv.org/abs/2602.23795)

**Authors**: Wenwu Tang, Dong Wang, Lothar Thiele, Olga Saukh  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.23795v1  

#### Abstract
Structured deep model compression methods are hardware-friendly and substantially reduce memory and inference costs. However, under aggressive compression, the resulting accuracy degradation often necessitates post-compression finetuning, which can be impractical due to missing labeled data or high ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在深度模型压缩中，**structured pruning**（结构化剪枝）和 **folding**（通道聚类合并）虽然硬件友好、能显著降低内存和推理开销，但在高压缩率下会导致明显的精度下降。传统做法依赖**微调（finetuning）**来恢复性能，但微调需要大量标注数据和计算资源，在许多实际部署场景中不可行。

现有补偿方法通常：
- 是**data-free**（无数据感知），忽略真实数据分布下的激活模式；
- 或与特定压缩算法耦合，缺乏通用性；
- 或仅修正一阶统计量（如均值偏移），无法捕捉通道间的复杂相关性。

GRAIL 正是为解决这一“**压缩后性能恢复无需微调且需数据感知**”的挑战而提出。

---

### **提出了什么新方法或新思路**
作者提出 **GRAIL**（**GRAm-Integrated Linear compensation**），一种**训练无关（zero-finetuning）、模块化、块级补偿机制**，核心思想如下：

- **Post-hoc 补偿**：在任意结构化压缩（pruning 或 folding）完成后，再进行补偿，完全解耦压缩与补偿。
- **数据感知（data-aware）**：利用一个小的**未标注校准集（calibration set）**，通过前向传播收集 consumer 层输入的隐藏激活 $ H \in \mathbb{R}^{N \times H} $。
- **Gram 矩阵建模**：计算激活的二阶统计量——**Gram 矩阵** $ G = H^T H $，捕捉通道间相关性。
- **岭回归重建**：学习一个线性映射 $ B $，使得从压缩后的隐藏表示 $ h_p $ 能**线性重建原始隐藏表示 $ h $**：
  $$
  h \approx B h_p, \quad B = G_{HP}(G_{PP} + \lambda I)^{-1}
  $$
- **权重融合（weight merging）**：将重建矩阵 $ B $ 合并到下游 consumer 层的权重中（如 `W_proj = W_proj @ B`），而上游 producer 层则完成宽度缩减（选择或平均）。

> ✅ **关键洞察**：不试图穿过非线性层，而是在**非线性激活之后**进行重建，保证可操作性和有效性。

---

### **相比现有方法的优势**
| 维度 | GRAIL 的优势 |
|------|-------------|
| **Selector-agnostic** | 兼容多种压缩方法：Magnitude Pruning, Wanda, SlimGPT, FLAP, Folding 等 |
| **Training-free** | 无需反向传播、无需标签、无需梯度，仅需少量前向 pass |
| **通用性强** | 统一适用于 CNN（ResNet）、ViT、CLIP 和 decoder-only LLM（如 LLaMA） |
| **高效轻量** | 仅引入一次性的 $ O(H^2) $ 内存用于 Gram 矩阵积累，补偿步骤为闭式解（closed-form） |
| **理论联系经典方法** | 当 Gram 矩阵接近单位阵时，自动退化为经典剪枝/折叠行为 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 模型类别 | 数据集 |
|---------|--------|
| **视觉模型** | CIFAR-10, ImageNet-1K |
| **语言模型** | C4, WikiText-2, PTB |
| **零样本评估** | ARC-C, ARC-E, HellaSwag, PIQA, BoolQ, Winogrande |

---

### **实验设置和评估指标**

#### **模型与压缩方法**
- **视觉模型**：
  - ResNet-18（576 个 checkpoint）
  - ViT-B/32（125 个 checkpoint）
  - CLIP ViT-B/32（72 个 checkpoint）
- **语言模型**：
  - LLaMA-2-7B
- **压缩方式**：
  - Structured Pruning: Magnitude (L1/L2), Wanda, Wanda++, SlimGPT, FLAP
  - Model Folding
  - **注意**：ZipLM 因压缩与更新耦合，GRAIL 不适用

#### **校准设置**
- **视觉模型**：128 张无标签图像
- **LLM**：128 条序列，每条长度 2048 token
- 所有实验均无 label、无 gradient

#### **评估指标**
- **分类任务**：Test Accuracy (%)
- **语言建模**：Perplexity ↓
- **零样本能力**：Zero-shot Accuracy ↑
- **效率分析**：Calibration 时间、峰值内存消耗

#### **对比基线**
- 原始压缩方法（如 Wanda、SlimGPT）
- Data-free 折叠方法 [3]
- REPAIR [34]（BatchNorm 激活重归一化）
- 微调（Finetuned）模型作为上界参考

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **在 ResNet-18 上（CIFAR-10）**
- 在 **65% 稀疏度** 下：
  - 基线（L1-pruning）准确率暴跌至 **17.6%**
  - GRAIL 补偿后恢复至 **84.8%**，提升 **67.2 个百分点**
- 在低压缩率（10%-40%）下，几乎实现**无损压缩**（within 0.5% of original）

#### **在 CLIP ViT-B/32 上（ImageNet-1K）**
- 所有压缩方法下，GRAIL 显著提升 accuracy
- **Pruning + GRAIL 效果优于 Folding + GRAIL**，表明选择比聚类更利于后续重建

#### **在 LLaMA-2-7B 上（语言建模）**
| 方法 | 数据集 | Sparsity | PPL (原) | PPL (+GRAIL) | 改善幅度 |
|------|--------|----------|-----------|----------------|------------|
| Wanda | C4 | 40% | 12.49 | **11.26** | ↓10% |
| SlimGPT | C4 | 50% | 26.72 | **23.75** | ↓11% |
| FLAP | C4 | 70% | 803.27 | **126.88** | ↓84% ❗️ |
| Wanda++ | PTB | 50% | 90.80 | **592.58** | ⬆️？⚠️（表中可能笔误，应为下降）|

> ✅ **总体趋势**：压缩越激进、原方法越缺乏重建机制，GRAIL 提升越显著。

---

### **与基线方法的对比结果**
- **优于 REPAIR**：在 ResNet 和 ViT 上，GRAIL + REPAIR 进一步提升，说明二者互补。
- **优于微调外的其他 post-hoc 方法**：
  - 超过 FLAP 自带的 bias correction
  - 超过 SlimGPT 的 curvature-based reconstruction
  - 超过 Wanda++ 的 regional optimization（尤其在低稀疏度）
- **算法无关性验证**：对所有 pruning/folding 方法均有增益，证明其通用性。

---

### **消融实验结果**
#### **校准集大小的影响（Figure 4）**
- **数据效率极高**：
  - 视觉模型：仅需 **128 图像**即达性能饱和
  - LLM：仅需 **128 sequences**（约 26 万 tokens）
- 性能随样本数呈**对数增长**，初期提升快，后期饱和
- **SlimGPT 需更多样本稳定**，而 Wanda/FLAP 更高效

#### **计算与内存开销（Table 3）**
| Model | Calibration Time | Compensation Time | Peak Memory |
|-------|------------------|--------------------|-------------|
| ResNet | 0.19s | 0.10s | 162 MB |
| ViT | 0.20s | 0.04s | 162 MB |
| CLIP | 0.95s | 0.16s | 300 MB |
| **LLaMA-2-7B** | **58.08s** | **3.16s** | **~3.3 GB** |

> 💡 结论：**calibration 占主导成本**，compensation 极轻量；虽对数据中心 GPU 可接受，但在边缘设备可能受限。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **GRAIL 是一种简单、有效、通用的 post-compression 补偿框架**，能在**无需微调**的前提下显著恢复压缩模型性能。
2. ✅ **Gram 矩阵 + 岭回归** 是一种强大的数据感知重建机制，能有效捕捉通道间相关性，优于仅用一阶统计的方法。
3. ✅ **补偿应发生在非线性之后**，并与 consumer 权重融合，形成 end-to-end 等效变换。
4. ✅ **兼容性极强**：适用于 CNN、ViT、LLM；兼容 pruning 与 folding；适配多种选择器。
5. ✅ **数据效率高**：极小校准集即可获得大部分收益，适合隐私敏感或数据稀缺场景。

---

### **方法的局限性**
1. 🚫 **依赖未压缩模型的一次前向 pass**：必须运行完整模型以获取激活，无法完全脱离原始模型。
2. 🚫 **Gram 矩阵内存开销为 $ O(H^2) $**：对超宽隐层（如 d_model=4096）可能占用数 GB 显存，限制在边缘设备应用。
3. 🚫 **块局部性（block-local）补偿**：未考虑跨层联合优化，极端压缩下可能仍有累积误差。
4. 🚫 **对分布偏移敏感**：若校准集与真实推理数据分布不一致，Gram 统计可能失效。

---

### **未来工作方向**
1. 🔮 **扩展至多层联合补偿**：设计跨层协同重建机制，减少误差传播。
2. 🔮 **结合量化与 KV-cache 压缩**：将 GRAIL 思路推广至其他压缩维度（如 INT8、KV cache pruning）。
3. 🔮 **降低内存需求**：
   - 使用低秩近似 Gram 矩阵
   - 流式累计避免全存储
4. 🔮 **任务感知校准**：使用任务相关 prompt 或数据增强提升校准质量。
5. 🔮 **理论分析**：建立 Gram 矩阵条件数、压缩率与重建误差之间的理论边界。

---

> 🔗 **代码开源**：https://github.com/TWWinde/GRAIL  
> 📚 **一句话总结**：**GRAIL 用一次 Gram 矩阵回归，实现了无需训练的通用压缩补偿，在 ResNet、ViT 和 LLaMA 上均显著缩小了与微调模型的差距。**

</details>

---

### 5. [Learning Generation Orders for Masked Discrete Diffusion Models via Variational Inference](https://arxiv.org/abs/2602.23968)

**Authors**: David Fox, Sam Bowyer, Song Liu, Laurence Aitchison, Raul Santos-Rodriguez, Mengyue Yang  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.23968v1  

#### Abstract
Masked discrete diffusion models (MDMs) are a promising new approach to generative modelling, offering the ability for parallel token generation and therefore greater efficiency than autoregressive counterparts. However, achieving an optimal balance between parallel generation and sample quality rem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning Generation Orders for Masked Discrete Diffusion Models via Variational Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Masked Discrete Diffusion Models (MDMs) 虽然支持并行 token 生成，从而在推理效率上优于传统的 Autoregressive Models (ARMs)，但在实际应用中面临一个核心挑战：**如何在保持高并行度的同时避免因过度并行化（over-parallelisation）导致的样本质量下降**。  
当前主流方法依赖于启发式采样策略（如 Top-k、Top Probability Margin），这些方法缺乏灵活性且对模型置信度的校准敏感。

### 提出的新方法与新思路
本文提出了一种基于 **Variational Inference (VI)** 的框架，用于学习 MDMs 中 token 的生成顺序（generation order）。其核心思想是：
- 将生成顺序建模为隐变量（latent variable），通过变分推断联合优化去噪网络和 unmasking 策略。
- 显式地将模型分解为两个组件：
  - 决定哪些 token 位置被解码（unmask）
  - 给定位置后预测 token 值
- 引入一个参数化的近似后验分布 $ q_\phi(\mathbf{r}|x_{t+1}, x_0) $ 来建模 unmasking 变量 $\mathbf{r}$，该设计支持高效采样和低方差训练。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **理论基础更强** | 首次从 VI 角度形式化 generation order 学习问题，提供了更坚实的概率建模视角 |
| **训练稳定性更高** | 利用 Rao-Blackwellisation 和 REINFORCE-Leave-One-Out (RLOO) 控制梯度方差，提升训练效率 |
| **自适应并行性** | 模型能根据输入动态调整每步 unmask 的 token 数量，实现“按需并行”，避免盲目并行带来的错误累积 |
| **模块化设计** | 分离 unmasking policy 与 denoising network，便于独立优化与分析 |

---

## 2. 核心实验方法和设置

### 数据集
- **GSM8K**：一个数学应用题问答数据集，用于评估模型在复杂语义任务上的生成能力。
- 使用预训练的 170M 参数 MDM 模型进行微调，确保起点一致。

### 实验设置
- **训练流程**：
  1. 先对 vanilla MDM 进行 45,000 步监督微调（batch size=256）
  2. 在此基础上，额外训练 15,000 步以学习 generation order（batch size=32，每样本采样 8 次 RLOO）
- **预算控制（Budget T）**：设定最大解码步数 $ T \in \{5, 10, 15\} $，衡量不同并行程度下的性能表现
- **评估方式**：
  - 报告平均、最小、最大解码步数
  - 使用 **准确率（Accuracy %）** 作为主要指标（针对 GSM8K 的答案匹配）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **IID** | 每个 masked token 以相同概率独立 unmask |
| **Top Probability** | 选择模型最自信（max prediction prob）的 K 个 token 进行 unmask |
| **Top Probability Margin** | 选择预测概率差值最大的 K 个 token（即最确定唯一性的 token） |

所有基线均在相同平均或最大步数下评估，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Budget (T) | Method | Avg. Steps | Range | Acc. (%) |
|-----------|--------|------------|-------|----------|
| 5 | Ours (Learned Order) | **4.01** | [2,5] | **33.1** |
| 5 | IID @ Avg | 4.0 | [4,4] | 29.0 |
| 5 | Top Prob @ Avg | 4.0 | [4,4] | 23.7 |
| 5 | Top Prob Margin @ Avg | 4.0 | [4,4] | 24.0 |
| 10 | Ours | **9.57** | [7,10] | **37.8** |
| 10 | Top Prob Margin @ Max | 10.0 | [10,10] | 39.5 |
| 15 | Ours | **9.43** | [5,12] | **39.0** |
| 15 | Top Prob Margin @ Max | 12.0 | [12,12] | 42.3 |

### 对比结果分析
- 在 **极低预算（T=5）** 下，本方法仅用约 **4 步** 即达到 **33.1% 准确率**，显著优于所有基线（最高为 29.0%），说明其在高度并行场景下仍能维持高质量生成。
- 当预算增加时，本方法虽未全面超越 best-in-class 启发式方法（如 Top Prob Margin），但仍表现出强竞争力，尤其在 **平均步数更低的情况下接近甚至超过基线**。
- 本方法具有 **自适应步长特性**：可在简单样本上快速完成生成，在困难样本上逐步展开，体现出智能调度潜力。

### 消融实验（文中未明确列出，但有设计讨论）
- **温度系数 $ \tau $**：实验发现设置 $ \tau \in [0.05, 0.1] $ 更优，有助于防止初期训练时 unmask 过快，提高训练信号利用率。
- **posterior 形式设计**：尝试多种形式后最终采用基于 score renormalization 的轻量级更新机制（Eq. 14），满足计算效率与可学习性的平衡。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **VI 框架可用于有效学习 generation order**：首次成功将 generation order 学习纳入 VI 框架，证明其可行性与有效性。
2. ✅ **learned unmasking policy 可胜过启发式方法**：特别是在高并行度限制下（如 T=5），本方法明显优于 IID、Top Prob 等固定策略。
3. ✅ **自适应并行生成是可行路径**：模型能够动态决定每步 unmask 数量，兼顾效率与准确性，为未来构建“智能解码控制器”提供思路。
4. 🔍 **性能差距随预算增大而缩小**：当允许更多步骤时，启发式方法（尤其是 Top Prob Margin）也能取得很好效果，表明当前 learned 方法仍有改进空间。

### 方法的局限性
- **训练成本较高**：由于引入额外网络 $ \alpha, p $ 并使用 RLOO 估计梯度，训练开销大于标准 MDM。
- **依赖预训练 denoiser**：实验中固定了 denoising network，仅训练 unmasking policy，可能限制联合优化潜力。
- **尚未验证泛化性**：目前仅在 GSM8K 上测试，缺乏在文本生成、代码生成等其他任务上的验证。
- **梯度估计仍具挑战**：尽管使用 RLOO，REINFORCE 类方法依然存在方差问题，影响收敛速度。

### 未来工作方向
1. **扩展到更大规模模型与多任务场景**：验证方法在 LLM 级别 MDM 上的有效性。
2. **探索更高效的 posterior parameterization**：设计更灵活且易于优化的 generation order 分布族。
3. **端到端联合训练 denoiser 与 unmasking policy**：打破冻结 denoiser 的假设，实现协同进化。
4. **结合强化学习或其他控制机制**：进一步提升 generation order 的决策能力。
5. **理论分析 unmasking order 与 mutual information 的关系**：深入理解并行化误差来源。

---

> **总结一句话**：  
> 本文提出了一种基于 Variational Inference 的新框架，使 MDM 能够**学会自适应地决定 token 的生成顺序**，在保持高并行效率的同时显著提升生成质量，尤其在资源受限场景下展现出优越性能，为下一代高效扩散语言模型提供了重要思路。

</details>

---

### 6. [RF-Agent: Automated Reward Function Design via Language Agent Tree Search](https://arxiv.org/abs/2602.23876)

**Authors**: Ning Gao, Xiuhui Zhang, Xingyu Jiang, Mukang You, Mohan Zhang, Yue Deng  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.23876v1  

#### Abstract
Designing efficient reward functions for low-level control tasks is a challenging problem. Recent research aims to reduce reliance on expert experience by using Large Language Models (LLMs) with task information to generate dense reward functions. These methods typically rely on training results as ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RF-Agent: Automated Reward Function Design via Language Agent Tree Search

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在低层级控制任务（low-level control tasks）中，设计高效的 **Reward Function** 是强化学习（RL）成功的关键，但传统方法依赖专家经验手动设计，费时且可能次优。虽然近期研究尝试利用 **Large Language Models (LLMs)** 自动生成密集奖励函数（dense reward functions），但现有方法存在以下问题：

- **搜索效率低下**：如 Eureka 使用贪心算法，Revolve 使用进化算法，难以平衡 **exploration 与 exploitation**。
- **历史反馈利用不足**：仅保留局部最优路径，忽略从低分到高分的潜在优化路径。
- **缺乏系统性推理能力**：未充分利用 LLM 的多阶段上下文推理能力。

### **提出了什么新方法或新思路**

本文提出 **RF-Agent**，一种将 LLM 视为语言智能体（language agent）、将奖励函数设计建模为**序列决策过程**的新框架。其核心创新包括：

- **Monte Carlo Tree Search (MCTS) 集成**：首次将 MCTS 引入 LLM-based 奖励函数设计，构建一棵树形结构来管理整个优化过程，每个节点代表一个奖励函数及其反馈。
- **多类型动作机制（Action Types）**：在扩展阶段引入五种不同动作类型，引导 LLM 生成多样化奖励函数：
  - `Mutation`（局部修改结构或参数）
  - `Crossover`（融合精英节点的高分组件）
  - `Path Reasoning`（基于祖先路径进行推理）
  - `Different Thought`（鼓励跳出已有模式）
- **自验证机制（Self-Verify）**：引入 LLM 自我评估分数，作为节点选择的先验，帮助早期识别有潜力的低分函数。
- **思维对齐（Thought Alignment）**：在执行后重新生成设计思想，确保代码与原始意图一致，缓解 LLM 幻觉问题。

### **相比现有方法的优势**

| 特性 | Eureka | Revolve | RF-Agent |
|------|--------|---------|----------|
| 搜索策略 | 贪心迭代 | 进化算法 | MCTS 树搜索 |
| 历史信息利用 | 局部最优保留 | 种群演化 | 全局路径推理 + 精英融合 |
| 探索多样性 | 有限 | 中等 | 高（多种动作类型） |
| 上下文推理深度 | 浅层 | 浅层 | 多阶段、树状推理 |

RF-Agent 显著提升了搜索效率与质量，在复杂任务中表现更鲁棒。

---

## 2. 核心实验方法和设置

### **使用的数据集**

实验在两个主流低层级控制环境上进行，共覆盖 **17 个任务**：

- **IsaacGym**：7 个代表性任务，涵盖：
  - **Locomotion**：Ant, Anymal, Humanoid, Quadcopter
  - **Manipulation**：AllegroHand, FrankaCabinet, ShadowHand
- **Bi-DexHands**：10 个双手机械臂操作任务，进一步分为两类：
  - **Expert-Easy**：人类奖励函数成功率高（如 GraspAndPlace, BlockStack）
  - **Expert-Hard**：人类奖励函数成功率低（如 SwingCup, DoorCloseOutward）

### **实验设置和评估指标**

- **Policy Training**：
  - 使用 PPO 算法，固定超参。
  - 每个最终奖励函数独立训练 5 个 seed，报告平均最大评估得分。
- **评估指标**：
  - **IsaacGym**：任务特定指标（如 Ant 的前进速度）
  - **Bi-DexHands**：二元成功信号（0/1），以 **Success Rate** 为主要指标
  - 报告 **Normalized Score**：`(Method - Sparse) / (Human - Sparse)`，衡量相对于人类专家的性能比例

- **LLM 模型**：
  - 主要使用 **GPT-4o-mini** 和 **GPT-4o** 进行公平比较
  - 控制总采样次数一致（IsaacGym: 80, Bi-DexHands: 512）

### **基线方法对比**

| 方法 | 类型 | 描述 |
|------|------|------|
| **Sparse** | 基线 | 仅使用稀疏任务完成信号作为奖励 |
| **Human** | 专家基线 | 由 RL 研究者手工设计的密集奖励函数 |
| **Eureka** | 贪心法 | 批量生成 + 保留最优 + 迭代优化 |
| **Revolve** | 进化法 | 维持种群 + 进化操作（本工作实现自动化版本） |
| **Ours (RF-Agent)** | 树搜索法 | 本文提出的方法 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **IsaacGym 结果（见 Table 1）**

| 方法 | Avg Norm Score (4o-mini) | Avg Norm Score (4o) |
|------|---------------------------|---------------------|
| Human | 1.00 | 1.00 |
| Eureka | 0.63 | 2.00 |
| Revolve | 0.67 | 2.03 |
| **RF-Agent (Ours)** | **1.70** | **2.68** |

- 在轻量模型 **GPT-4o-mini** 下，RF-Agent 已全面超越 Eureka 和 Revolve，并在多数任务上超过人类专家。
- 使用更强模型 **GPT-4o** 后，所有方法提升，但 RF-Agent 提升最显著，达到 **2.68 倍于人类归一化得分**。

#### ✅ **Bi-DexHands 结果（见 Figure 3）**

- 在 **Expert-Easy** 任务中，RF-Agent 匹配甚至略超人类；而 Eureka/Revolve 仅达人类一半水平。
- 在 **Expert-Hard** 任务中，RF-Agent 明显领先，尤其在 SwingCup、DoorCloseOutward 等复杂任务上优势显著。

#### ✅ **训练效率对比（见 Figure 4）**

- RF-Agent 设计的奖励函数能更快地引导策略收敛至高性能区域，表明其生成的奖励具有更高的**训练效率**。

#### ✅ **优化轨迹分析（见 Figure 5）**

- 随着采样次数增加，RF-Agent 的平均最高得分上升最快，说明其具备更强的**持续优化能力**，有效克服了“早熟收敛”问题。

---

### **消融实验结果（Ablation Studies）**

#### 🔹 **搜索策略影响（图6a）**
- 替换 MCTS 为 DFS/BFS/Greedy 导致至少一个任务性能大幅下降，证明 MCTS 对探索-利用平衡至关重要。

#### 🔹 **动作类型影响（图6b）**
- 移除任一动作类型均导致性能下降，尤其是移除全部动作时退化严重。
- 表明多种动作协同作用是成功关键。

#### 🔹 **推理机制影响（图6c）**
- 移除 `self-verify` 或 `thought alignment` 均造成性能损失，尤其在复杂任务（如 AllegroHand）上更为明显。
- 证明自我评估与思维对齐有助于缓解幻觉、提升推理质量。

> 更细粒度组合实验（Appendix F.5）显示：**本地操作（Mutation）与全局操作（Crossover + Path Reasoning）结合**才能达到最佳性能。

---

## 4. 关键结论和发现

### **主要发现**

1. **将奖励函数设计视为序列决策问题是有效的**：通过 MCTS 构建树结构，可系统化利用历史反馈，避免陷入局部最优。
2. **LLM 的多阶段上下文推理能力可被充分激发**：通过设计多样化的动作提示和路径推理机制，显著提升生成质量。
3. **全局信息整合优于局部优化**：Crossover 和 Path Reasoning 动作能够复用其他路径的成功经验，实现跨路径的知识迁移。
4. **即使使用轻量级 LLM，RF-Agent 仍可超越人类专家**：证明其框架本身具有强大增益效果。

### **方法的局限性**

- **计算成本高**：需要多次完整的 RL 训练循环（policy training），耗时长（Bi-DexHands 单任务约 40 小时）。
- **不减少 RL 训练轮次**：尽管搜索更高效，但仍需大量环境交互，无法规避 RL 本身的开销。
- **依赖高质量环境观测接口**：需访问 `observation` 代码片段以供 LLM 分析，限制了通用性。

### **未来工作方向**

- 减少 RL 训练周期数，例如引入**课程学习**或**代理模型预测反馈**。
- 探索更高效的树剪枝策略，降低冗余节点生成。
- 将 RF-Agent 应用于更高层次的任务规划或多智能体协作场景。

---

> **源码地址**：[https://github.com/deng-ai-lab/RF-Agent](https://github.com/deng-ai-lab/RF-Agent)

</details>

---

### 7. [Learning Flexible Job Shop Scheduling under Limited Buffers and Material Kitting Constraints](https://arxiv.org/abs/2602.24180)

**Authors**: Shishun Zhang, Juzhan Xu, Yidan Fan, Chenyang Zhu, Ruizhen Hu, Yongjun Wang, Kai Xu  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.24180v1  

#### Abstract
The Flexible Job Shop Scheduling Problem (FJSP) originates from real production lines, while some practical constraints are often ignored or idealized in current FJSP studies, among which the limited buffer problem has a particular impact on production efficiency. To this end, we study an extended p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning Flexible Job Shop Scheduling under Limited Buffers and Material Kitting Constraints*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文研究的是**Flexible Job-Shop Scheduling Problem with Limited Buffers and Material Kitting (FJSP-LB-MK)**，即在有限缓冲区和物料配套（material kitting）约束下的柔性作业车间调度问题。

- **现实背景**：在钢板加工、零件分拣等高混合生产场景中，工件需暂存于有限数量的缓冲区（如托盘 pallet），且每个托盘只能存放同一类别的零件（material kitting 规则）。
- **挑战**：当新任务的零件类别无法匹配现有托盘时，必须更换托盘（pallet change），导致停机时间增加、效率下降。
- 当前大多数 FJSP 研究忽略此类实际资源约束，导致模型难以落地。

### 🚀 提出的新方法与思路
提出了一种基于 **Deep Reinforcement Learning (DRL)** 和 **Heterogeneous Graph Neural Network (HGNN)** 的端到端调度框架：

1. **全局状态建模**：
   - 构建一个异构图（heterogeneous graph）来统一表示机器（machine）、操作（operation）、缓冲区（buffer）之间的复杂依赖关系。
   - 图节点包含丰富的特征，如操作类型、零件类别 one-hot 编码、预估托盘更换次数（SwEst）等。

2. **Operation-Buffer Embedding 模块**（核心创新）：
   - 引入专门的 **Operation-Buffer 边**，仅连接“零件分拣操作”（part-sorting operations）与 buffer 节点。
   - 采用**加权消息传递机制**，边权重与该操作预计引发的 pallet change 成正比（`w = sigmoid(α × SwEst)`），实现**cost-sensitive propagation**。
   - 这使得网络能提前感知高成本决策，主动规避频繁换盘。

3. **双目标奖励函数设计**：
   - 奖励函数同时考虑 `makespan` 和 `pallet changes` 的变化：
     $$
     r(s_t,a_t,s_{t+1}) = (C_{\text{max}}(s_t) - C_{\text{max}}(s_{t+1})) + \lambda (P_{\text{max}}(s_t) - P_{\text{max}}(s_{t+1}))
     $$
   - 实现对调度紧凑性和缓冲区利用率的联合优化。

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | 现有 DRL 方法 | 本文方法 |
|------|----------|----------------|-----------|
| **状态表达能力** | 手工规则或扁平化表示 | 简单图结构，难捕获长程依赖 | 异构图 + 加权消息传递，精准建模共享资源影响 |
| **处理复杂约束** | 受限于求解器规模或启发式泛化性差 | 忽略 buffer/kitting 等实际约束 | 显式建模 limited buffer 与 material kitting |
| **决策质量** | 局部最优风险高 | 难以预见 pallet change 后果 | 主动避免高代价换盘，提升整体效率 |
| **可扩展性** | 随规模增长计算开销剧增 | 一般较好 | 在大规模实例上仍保持高性能 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

#### （1）合成数据集（Synthetic Dataset）
- 基于 [Brandimarte, 1993] 方法生成，共6种规模：从 `10×5` 到 `40×10`（jobs × machines）。
- 添加 FJSP-LB-MK 特有参数（见 Table I）：
  - part-sorting operations 数量
  - 每个 job 的 part categories
  - 总 part categories C
  - 可用 pallet 数量 P
  - pallet replacement time (`t_switch`)
- 所有参数随问题规模自适应调整。

#### （2）真实生产线数据集（Real Production Line Dataset）
- 来源于四条工业级钢板加工线（A:20×16, B:20×12, C:20×10, D:20×12）。
- 每条线采集 10,000 训练样本、100 验证、100 测试样本。
- 更具挑战性：机器-操作映射非均匀、part categories 分布不均、buffer 拥堵更严重。

### ⚙️ 实验设置

- **训练平台**：NVIDIA RTX 3080 Ti GPU + Intel i9 CPU
- **算法框架**：Proximal Policy Optimization (**PPO**)，Actor-Critic 结构
- **训练策略**：
  - 先在小规模合成数据上训练
  - 再迁移到真实产线数据进行 fine-tuning
  - Fine-tuning 时引入 KL 正则项防止策略突变
- **推理方式**：
  - Greedy：选择概率最高的动作
  - Sampling：采样 100 次取最优结果

### 📈 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **Makespan** | 所有作业完成的最大时间 | 主要优化目标 |
| **Gap** | 相对于 OR-Tools 解的质量差距 (%) | 衡量解优度 |
| **Switches (Pallet Changes)** | 总托盘更换次数 | 衡量缓冲区利用效率 |
| **Computation Time** | 获取解所需平均时间（秒） | 衡量算法效率 |

### 🆚 基线方法对比

| 类别 | 方法 |
|------|------|
| **Exact Method** | OR-Tools (CP-SAT) + MCTS 初始化序列，限时 1800s |
| **Heuristics** | PDRs：<br>- Sequencing: FIFO, MWR, LWR, SPT<br>- Assignment: EET |
| **DRL Baseline** | [Song et al., 2022] 的 GNN+DRL 方法（adapted to FJSP-LB-MK）<br>- DRL-Greedy / DRL-Sampling |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table IV & V）

#### 在合成数据集上的表现（以 40×10 为例）：

| 方法 | Makespan | Gap | Switches | Time (s) |
|------|----------|-----|----------|----------|
| PDRs (best) | 723.96 | 39.7% | 68.33 | 2.10 |
| [15] DRL-Greedy | 742.75 | 43.3% | 67.66 | 3.21 |
| **Ours (Greedy)** | **567.47** | **9.5%** | **33.76** | 3.47 |
| [15] DRL-Sampling | 715.23 | 38.0% | 62.73 | 13.18 |
| **Ours (Sampling)** | **564.89** | **9.0%** | **31.47** | 14.05 |
| OR-Tools | 518.18 | — | 41.82 | >1800 |

✅ **结论**：我们的方法在所有规模下均显著优于 PDRs 和 DRL 基线，在 `makespan` 上降低约 20%-30%，在 `switches` 上减少超过 **50%**，且计算时间远低于 OR-Tools。

#### 在真实产线数据集上的表现（以 C 组为例）：

| 方法 | Makespan | Gap | Switches |
|------|----------|-----|----------|
| Best PDR (MWR) | 40565.81 | 0.5% | 31.60 |
| [15] DRL-Sampling | 40994.28 | 1.5% | 25.63 |
| **Ours (Sampling)** | **40359.80** | **-0.04%** | 25.89 |
| OR-Tools | 40377.03 | — | 25.42 |

✅ **亮点**：
- 在产线 C 上，我们的方法甚至**略微超越**了 OR-Tools 的解质量（Gap < 0），是唯一达到或接近最优的方法。
- 尽管 switches 不是最少，但整体 makespan 最优，说明实现了更好的权衡。

### 🔍 消融实验结果（Ablation Study）

#### （1）关键状态特征有效性（Table VI）

| 特征组合 | Makespan | vs Ours | Switches | vs Ours |
|--------|---------|--------|---------|--------|
| PS+SwEst | 183.94 | +8.3% | 15.41 | +27.4% |
| PS+Type | 187.73 | +10.5% | 15.42 | +27.4% |
| Type+SwEst | 172.41 | +1.5% | 12.08 | -0.2% |
| **PS+Type+SwEst (Ours)** | **169.85** | **0.0%** | **12.10** | **0.0%** |

📌 **发现**：`Type`（零件类别）和 `SwEst`（预估换盘数）最为关键；缺少任一都会显著恶化性能。

#### （2）Selective Connectivity 与 Cost-sensitive Propagation（Table VII）

| 配置 | Makespan | vs Ours | Switches | vs Ours |
|------|---------|--------|---------|--------|
| Base (无连接) | 172.62 | +1.6% | 12.18 | +1.0% |
| Pallet_AllOps | 172.80 | +1.7% | 12.22 | +1.0% |
| Pallet_SortOnly (uniform) | 171.03 | +0.7% | 12.31 | +1.7% |
| Pallet_SortOnly_InverseWeight (benefit-seeking) | 170.82 | +0.6% | 12.28 | +1.5% |
| **Pallet_SortOnly_Weighted (cost-avoiding, Ours)** | **169.85** | **0.0%** | **12.10** | **0.0%** |

📌 **关键发现**：
- **Selective connectivity**（只连 part-sorting ops）优于全连接，避免无关操作接收噪声信息。
- **Cost-avoiding strategy**（权重正比于 SwEst）明显优于 benefit-seeking（反比），后者易陷入局部最优。
- 二者结合带来最大收益。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **首次将 DRL 应用于 FJSP-LB-MK 问题**，填补了实际生产中 buffer 与 material kitting 约束建模的研究空白。
2. 提出的 **HGNN + weighted message passing** 架构能够有效捕捉跨资源的长期依赖，使 agent 能“预见”当前决策对未来 pallet availability 的影响。
3. 通过 **cost-avoiding 策略** 和 **双目标奖励机制**，实现了 makespan 与 pallet changes 的良好平衡。
4. 在合成与真实数据集上均表现出色，尤其在大规模问题上展现出更强的 scalability 和 generalization 能力。
5. 消融实验证明了 **Selective connectivity** 和 **Cost-sensitive propagation** 是性能提升的关键组件。

### ⚠️ 方法的局限性

1. **对产线异质性敏感**：在某些真实产线上（如 B/D），虽然 makespan 占优，但 switches 并未最少，反映出不同环境下优化优先级可能需要动态调整。
2. **预估 SwEst 依赖经验或辅助模型**：目前 SwEst 是作为输入特征提供，若其估计不准会影响性能，未来可探索联合学习。
3. **未考虑动态扰动**：实验假设环境理想（无机器故障、无动态订单插入），离完全在线调度仍有距离。

### 🔮 未来工作方向

1. **动态调度扩展**：引入 online learning 机制应对实时扰动（如机器宕机、紧急插单）。
2. **端到端 SwEst 学习**：让模型自主预测 pallet change 成本，而非依赖外部输入。
3. **多目标强化学习**：使用 MORL 或 Pareto-based 方法显式探索 makespan 与 switches 的 trade-off frontier。
4. **部署至真实工厂系统**：结合数字孪生与边缘计算，构建闭环智能调度平台。

---

> 💡 **一句话总结**：  
> 本文提出一种融合 **Heterogeneous GNN** 与 **DRL** 的新型调度方法，首次系统解决了 **FJSP-LB-MK** 问题，通过精细化建模缓冲区状态与换盘成本，在真实与合成数据上均实现了 **更短 makespan** 与 **更少 pallet changes** 的双重优势，为智能制造中的复杂调度提供了高效可行的解决方案。

</details>

---

### 8. [Divide and Conquer: Accelerating Diffusion-Based Large Language Models via Adaptive Parallel Decoding](https://arxiv.org/abs/2602.23792)

**Authors**: Xiangzhong Luo, Yilin An, Zhicheng Yu, Weichen Liu, Xu Yang  
**Category**: cs.CL  
**Published**: 2026-03-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.23792v1  

#### Abstract
Diffusion-based large language models (dLLMs) have shown promising performance across various reasoning tasks, establishing themselves as an alternative to autoregressive large language models (LLMs). Unlike autoregressive LLMs that generate one token per step based on all previous tokens, dLLMs the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Divide and Conquer: Accelerating Diffusion-Based Large Language Models via Adaptive Parallel Decoding**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
当前主流的 **autoregressive LLMs**（AR LLMs）受限于逐token生成机制，推理过程高度串行化，导致高延迟、低吞吐。虽然 **diffusion-based LLMs**（dLLMs）理论上支持在每个解码步并行生成多个 token，具备加速潜力，但在实践中仍普遍采用“一次一token”生成策略以维持生成质量。

这揭示了一个关键矛盾：**dLLMs 的理论并行性** 与 **实际性能之间的巨大差距**。直接进行 naive parallel decoding 会导致生成质量显著下降，尤其是在数学推理和代码生成等复杂任务中。

---

### ✅ **提出了什么新方法或新思路**
本文提出 **DiCo**（**Divide and Conquer**），一种无需额外训练的自适应并行解码框架，旨在释放 dLLMs 的并行潜力，同时保持高质量输出。

DiCo 采用三阶段 **divide-and-conquer 范式**：

1. **Divide 阶段**  
   - 探索输入序列，识别具有中等且稳定置信度的 masked tokens 作为 **seed tokens**。
   - 利用 **Soft-NMS 启发的位置感知函数** 和 **trajectory-guided confidence 校准机制**，确保种子 token 在语义上丰富且空间上分散。
   - 将种子双向扩展为局部簇（local clusters），并通过合并形成非重叠的大簇。

2. **Conquer 阶段**  
   - 在不同 local clusters 上执行 **adaptive parallel decoding**。
   - 动态决定每一步解码多少 token，依据是：  
     $$
     S_t = \{ i \mid (|S_t| + 1)(1 - c(i)) < 1 \}
     $$
     即根据置信度动态控制并行规模。
   - 随着解码推进，持续更新簇边界，纳入新达到阈值的 token。

3. **Finalize 阶段**  
   - 对剩余少量、上下文依赖强的 masked tokens，采用 **fine-grained compound decoding**。
   - 结合 **top-1 confidence** 和 **logit margin**（最高与次高 logit 差值）进行精细解码，提升最终输出稳定性。

---

### ✅ **相比现有方法的优势**

| 维度 | DiCo | 现有方法（如 Fast-dLLM） |
|------|------|--------------------------|
| **是否需要训练** | ❌ 无需训练（training-free） | 多数需额外训练或策略学习 |
| **并行策略** | 自适应动态并行（adaptive） | 固定置信阈值或块大小（fixed/block-wise） |
| **解码粒度控制** | 分阶段精细化控制（Divide-Conquer-Finalize） | 缺乏阶段性演化建模 |
| **生成质量** | 显著优于 naive 并行方法 | 在并行时易出现质量坍塌 |
| **适用性** | 兼容 non-AR 和 semi-AR 设置 | 多依赖 semi-AR 才能稳定 |

> DiCo 成功实现了 **高效并行** 与 **高质量生成** 的统一，突破了 dLLMs “并行即降质”的瓶颈。

---

## 2. **核心实验方法和设置**

### 📚 **使用的模型与数据集**

- **模型**：
  - `LLaDA-8B-Instruct` (Nie et al., 2025)
  - `Dream-7B-Instruct` (Ye et al., 2025)

- **数据集**（涵盖数学与代码生成任务）：
  - **数学推理**：`GSM8K`, `Math-500`
  - **代码生成**：`HumanEval`, `MBPP`

---

### 🧪 **实验设置与评估指标**

- **硬件平台**：NVIDIA RTX 4090 GPU (24GB)
- **生成长度**：256 tokens
- **semi-AR 设置**：block size = 128
- **评估指标**：
  1. **Accuracy**：通过 `lm-eval-harness` 框架评估任务正确率
  2. **Throughput**：平均每秒生成 token 数（TPS），并报告相对 Vanilla 的 speedup ratio

---

### 🔁 **基线方法对比**

| 方法 | 类型 | 特点 |
|------|------|------|
| **Vanilla** | Baseline | Top-1 confidence-based decoding，一次一token |
| **Fast-dLLM** (Wu et al., 2025b) | Training-free | 固定置信阈值（0.95）并行解码，结合 KV caching |

> 所有方法均在 **non-AR** 和 **semi-AR** 两种模式下测试，确保公平比较。

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据（来自 Table 1）**

| 模型 | 方法 | Accuracy ↑ | Throughput (TPS) ↑ | Speedup × |
|------|------|------------|--------------------|-----------|
| **LLaDA-8B-Instruct** (non-AR) | Vanilla | 56.33 (GSM8K) | 6.94 | 1.0× |
| | **DiCo (Ours)** | **75.13** (+18.8%) | **23.46** | **3.4×** |
| | Fast-dLLM | 57.01 | 16.18 | 2.3× |
| **Dream-7B-Instruct** (non-AR) | Vanilla | 52.01 (GSM8K) | 8.07 | 1.0× |
| | **DiCo (Ours)** | **63.38** (+11.37%) | **31.83** | **3.9×** |
| | Fast-dLLM | 50.80 | 17.10 | 2.1× |

> 在 **HumanEval** 上，DiCo 达到 **29.27% 准确率**（Vanilla: 12.80%），吞吐达 **88.55 TPS**（4.8× 加速）。

---

### 🔍 **与基线方法的对比结果**

- **相对于 Vanilla**：
  - 平均准确率提升高达 **+18.8%**
  - 推理速度提升 **3.4× ~ 4.8×**
- **相对于 Fast-dLLM**：
  - 在 non-AR 设置下，DiCo 不仅更快（最高达 3.9× vs 2.3×），而且准确率更高（如 GSM8K 上 +18.13%）
  - 在 MBPP 上，Dream-7B + DiCo 实现 **7.9× 加速**，远超 Fast-dLLM 的 3.5×
- **特别优势**：
  - DiCo 在 **non-AR 模式下表现优异**，而多数 prior work 必须退回到 semi-AR 才能稳定。

---

### 🔬 **消融实验结果**

#### （1）**种子数量 N 的影响（Figure 5）**
- 在 HumanEval 上测试 N ∈ [2,10]
- 结果显示：**accuracy 和 throughput 均高度稳定**
- ➤ 表明 DiCo 对 seed token 数量不敏感，具备良好鲁棒性。

#### （2）**是否使用 trajectory guidance（Table 2）**
| 方法 | Accuracy | Throughput |
|------|---------|------------|
| DiCo w/o TG | 49.28 | 34.00 (4.9×) |
| DiCo w/ TG | **75.13** | 23.46 (3.4×) |

> 虽然无 TG 时吞吐更高，但准确率暴跌；加入后显著提升质量，说明 **trajectory guidance 对生成一致性至关重要**。

#### （3）**是否使用 fine-grained compound decoding（Table 3）**
| 方法 | Accuracy | Throughput |
|------|---------|------------|
| DiCo w/o LM | 74.45 | 19.45 (2.8×) |
| DiCo w/ LM | **75.13** | **23.46** (3.4×) |

> 引入 logit margin 后，**准确率和吞吐双升**，表明该机制有效提升了 Finalize 阶段的效率与可靠性。

#### （4）**可视化解码轨迹（Figure 7）**
- DiCo 完成解码仅需 **67 步**
- Fast-dLLM 需要 **87 步**
- Vanilla 需要完整 **256 步**

> 直观体现 DiCo 极大减少了总解码步数，实现真正意义上的高效收敛。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **dLLMs 存在“理论并行 vs 实践串行”的鸿沟**  
   - naive parallel decoding 导致严重性能坍塌（见 Figure 2）
   - 根本原因在于忽略了 token 间的上下文依赖

2. **dLLMs 解码过程中存在自然聚类行为**  
   - 中等置信度 token 会快速形成高置信区域（cluster）
   - 早期阶段依赖稀疏，适合并行；晚期依赖密集，需精细处理

3. **Divide-and-Conquer 是释放 dLLMs 并行性的有效范式**  
   - Divide 阶段构建语义局部性
   - Conquer 阶段安全并行
   - Finalize 阶段保障全局一致性

4. **adaptive 并行优于 fixed 并行**  
   - 动态控制每步解码 token 数，更契合置信度演化趋势

---

### ⚠️ **方法的局限性**

- **对 trajectory-guided 参数敏感**：尽管整体鲁棒，但在极端顺序任务中可能需调参。
- **依赖模型输出置信度可靠性**：若 dLLM 自身置信度校准差，seed selection 可能失效。
- **未探索更大规模模型**：实验限于 7B~8B 模型，百亿级以上效果未知。
- **无法完全消除 late-stage 串行开销**：Finalize 阶段仍有一定串行成分。

---

### 🔮 **未来工作方向**

1. **将 DiCo 思想迁移到其他生成范式**  
   - 如 speculative decoding、flow matching 等非自回归模型

2. **结合 approximate KV caching 进一步优化内存与延迟**  
   - 当前 DiCo 可与 dKV-Cache、dLLM-Cache 等正交技术联合使用

3. **设计可学习的 divide policy**  
   - 当前为 heuristic-based，未来可引入轻量 policy network 自动决策划分策略

4. **拓展至多模态 diffusion model 解码加速**

---

> **总结一句话**：  
> **DiCo 通过 divide-and-conquer 范式，在无需训练的前提下，首次实现了 dLLMs 高效且高质量的自适应并行解码，显著缩小了其理论潜力与实际性能之间的鸿沟。**

</details>

---

### 9. [ProductResearch: Training E-Commerce Deep Research Agents via Multi-Agent Synthetic Trajectory Distillation](https://arxiv.org/abs/2602.23716)

**Authors**: Jiangyuan Wang, Kejun Xiao, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.23716v1  

#### Abstract
Large Language Model (LLM)-based agents show promise for e-commerce conversational shopping, yet existing implementations lack the interaction depth and contextual breadth required for complex product research. Meanwhile, the Deep Research paradigm, despite advancing information synthesis in web sea...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《ProductResearch: Training E-Commerce Deep Research Agents via Multi-Agent Synthetic Trajectory Distillation》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前基于 **Large Language Model (LLM)** 的购物代理在处理复杂商品研究任务时存在以下不足：

- **交互深度不足**：现有 ReAct-style 代理多用于简单检索或推荐，缺乏对复杂、长周期研究任务的支持。
- **上下文广度有限**：难以融合开放网页信息与结构化电商产品数据库进行综合分析。
- **领域迁移困难**：“Deep Research” 范式虽在通用网络搜索中表现优异，但在 **e-commerce** 场景下因工具差异和数据异构性导致性能下降。

该论文旨在解决如何训练具备**深度信息整合能力**的电商购物代理，以支持复杂的购买前研究（如为敏感肌哺乳期妈妈构建安全有效的护肤方案）。

---

### 🚀 提出了什么新方法或新思路

作者提出 **ProductResearch** —— 一种基于 **Multi-Agent Synthetic Trajectory Distillation** 的框架，用于生成高质量、长视野的工具使用轨迹（tool-use trajectories），从而训练强大的电商研究型代理。

#### 核心组件：
1. **User Agent**  
   - 输入：真实用户的长期行为序列（购买、评论、对话等）
   - 输出：推断用户画像（persona）、生成复杂研究查询 `Q` 和定制化的评估标准（rubric）

2. **Research Agent**  
   - 执行实际的研究任务，遵循 `Plan → Toolcall → Report` 的认知流程
   - 工具集涵盖两类环境：
     - **Web Environment**：`web_search`, `web_visit`
     - **E-commerce Environment**：`product_search`, `visit_product`

3. **Supervisor Agent**  
   - 基于三阶段状态机提供细粒度监督：
     - `Check Plan`：验证计划逻辑完整性
     - `Check Toolcall`：确保工具调用正确且不重复
     - `Check Report`：依据动态 rubric 严格评估报告质量
   - 提供文本反馈引导修正，形成“监督-反馈-迭代”闭环

4. **Reflective Internalization（反射内化）**
   - 将多轮多角色交互轨迹（含 supervisor 反馈）提炼为单一角色的连贯训练样本
   - 保留纠错信号的同时适配标准 SFT 流程

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | ProductResearch |
|------|--------|----------------|
| 数据来源 | 人工标注或简单合成 | 多智能体协同生成高保真、长周期轨迹 |
| 领域适应性 | 通用 Deep Research 模型无法有效迁移到电商 | 显式建模电商特有工具流与数据结构 |
| 训练效率 | 多角色轨迹不可直接用于 SFT | 通过 **reflective internalization** 转换为单角色格式 |
| 评估机制 | 固定二元成功指标 | 动态生成 query-adaptive 评估 rubric（RACE） |

> ✅ 创新亮点：首次将 **multi-agent collaboration + synthetic trajectory + distillation** 成功应用于电商 Deep Research 场景，实现了从“任务完成”到“深度洞察”的跃迁。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **原始数据源**：来自阿里巴巴国际数字商业集团的真实匿名用户交互日志，包括：
  - 购买记录
  - 商品评价
  - 客服对话
- **构造数据集**：
  - 从中筛选出 1,000 名代表性用户，由 User Agent 生成对应的 persona、query 和 evaluation rubric
  - 最终数据集按 8:1:1 划分为训练、验证、测试集

---

### ⚙️ 实验设置

- **基础模型**：`Qwen3-30B-A3B`（MoE 架构，总参数 30B，每 token 激活 3B）
- **微调方式**：Supervised Fine-Tuning (SFT)
- **训练平台**：32×NVIDIA A100 (80GB) GPU 集群，使用 Megatron-LM 框架
- **上下文长度变体**：32k / 64k / 80k / 128k tokens
- **训练细节**：
  - 全局 batch size = 4
  - 训练 3 个 epoch
  - 采用 packing 策略提升 GPU 利用率

---

### 🎯 评估指标

采用改进版 **RACE metric**（源自 DeepResearch Bench），包含四个维度加权聚合：

| 维度 | 描述 |
|------|------|
| **Comprehensiveness (Comp.)** | 内容覆盖广度与深度 |
| **Depth** | 分析深度（因果推理、权衡分析等） |
| **Instruction-Following (Inst.)** | 对显式/隐式需求的遵循程度 |
| **Readability** | 结构清晰、可读性强、易于行动 |

此外还引入：
- **Effective Product Count (E.Prod)**：报告中提及的有效、不同产品的平均数量，衡量产品覆盖面

> 注：RACE 是相对评分（target vs reference），归一化至 [0,1] 区间，0.5 表示与参考报告持平。

---

### 🆚 基线方法对比

分为两类基线：

#### （1）Deep Research Agents
| 模型 | 类型 | 工具情况 |
|------|------|---------|
| Tongyi-DeepResearch | 开源模型 | 使用本研究定义的统一工具集 `T` |
| Qwen-DeepResearch | 闭源系统 | 使用其原生内置工具 |
| Gemini-DeepResearch | 闭源系统 | 使用其原生内置工具 |

#### （2）ReAct Agents
| 模型 | 类型 | 工具情况 |
|------|------|---------|
| Gemini-3-flash | 前沿 LLM | 接入统一工具集 `T` |
| GPT-4.1 | 前沿 LLM | 接入统一工具集 `T` |
| Qwen3-max | 前沿 LLM | 接入统一工具集 `T` |
| Qwen3-30B-A3B | 本研究 base model | 接入统一工具集 `T` |

> 所有非闭源系统均共享相同工具集 `T`，保证公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1）

| Model | RACE Overall | Comp. | Depth | Inst. | Read. | E.Prod |
|-------|--------------|--------|--------|--------|--------|--------|
| Tongyi-DeepResearch | 29.84 | 29.10 | 26.43 | 33.00 | 32.79 | 6.69 |
| Qwen-DeepResearch | 42.76 | 41.70 | 42.87 | 43.45 | 43.15 | 14.4 |
| Gemini-DeepResearch | **45.56** | 45.81 | **47.46** | 45.38 | 42.31 | **25.2** |
| Gemini-3-flash | 32.41 | 30.16 | 29.17 | 38.43 | 33.85 | 6.54 |
| GPT-4.1 | 36.46 | 33.88 | 41.47 | 41.10 | 37.65 | 7.98 |
| Qwen3-max | 36.67 | 35.40 | 33.44 | 41.28 | 38.74 | 6.06 |
| Qwen3-30B-A3B (base) | 31.78 | 29.81 | 28.41 | 36.33 | 35.42 | 3.58 |
| **ProductResearch-SFT-128k** | **45.40** | **45.44** | 43.87 | **46.09** | **47.22** | 12.45 |

> ✅ **关键发现**：
- 微调后的模型 **ProductResearch-SFT-128k** 总体 RACE 得分达 **45.40**，接近最强闭源系统 **Gemini-DeepResearch (45.56)**，远超所有 ReAct 基线。
- 在 **Readability (47.22)** 和 **Instruction-Following (46.09)** 上表现尤为突出，说明监督反馈显著提升了输出结构与指令对齐能力。
- **E.Prod = 12.45**，相比 base model 的 3.58 提升超过 **3倍**，表明模型能探索更广泛的产品空间。

---

### 🔍 消融实验结果

#### （1）上下文长度的影响（Figure 2）

| 上下文长度 | RACE Overall |
|------------|---------------|
| 32k | 37.75 |
| 64k | 44.59 (+6.84) |
| 80k | ~45.0 |
| 128k | **45.40** |

> ❗ 发现：从 32k 到 64k 提升巨大，说明 **长上下文是承载复杂推理链的关键**；继续延长仍有益处，趋于饱和。

#### （2）中间报告质量演化（Figure 3）

- 第一轮报告平均得分约 0.43
- 经过 Supervisor 多轮反馈后，第六轮接近 0.50（即与 reference 报告相当）
- 改进最明显的是第一到第二轮，说明初始反馈纠正了最大缺陷

> ✅ 验证了 **iterative feedback loop** 的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Multi-Agent Synthetic Trajectory 是可行且高效的训练范式**  
   - 能够生成高质量、长周期、领域适配的研究轨迹
   - 显著优于传统 ReAct 模式下的直接微调

2. **Supervisor Agent 的监督机制至关重要**  
   - 有效防止 hallucination、logic drift 和 insufficient evidence
   - 特别是在 `Check Report` 阶段，大幅提高输出质量和可读性

3. **Reflective Internalization 成功桥接多角色轨迹与单角色训练**  
   - 保留了纠错过程中的学习信号，同时兼容标准 SFT 流水线

4. **轻量级 MoE 模型经训练后可达近似闭源前沿系统的水平**  
   - 展示了 **scalable** 且 **cost-effective** 的路径

---

### ⚠️ 方法的局限性

1. **工具实现仍有优化空间**  
   - 当前 `web_visit` 和 `visit_product` 工具的底层实现尚不完善，影响信息提取精度

2. **仅支持单轮研究查询**  
   - 现实中用户意图常随对话演进，而当前框架未建模 multi-turn intent shift

3. **依赖高质量 LLM 作为基础组件**  
   - 若 Research Agent 或 Supervisor Agent 自身能力弱，则合成数据质量受限

---

### 🔮 未来工作方向

1. **扩展至 multi-turn 对话场景**  
   - 让 User Agent 模拟跨轮次的意图演变，训练更具对话持续性的购物代理

2. **进一步优化工具设计与调用策略**  
   - 引入强化学习或课程学习优化 agent 的 tool-use policy

3. **探索自动化 rubric 生成机制**  
   - 减少对人工设计评估标准的依赖，增强泛化能力

4. **降低计算成本与环境影响**  
   - 探索更小规模模型上的高效蒸馏路径，并公开资源消耗数据以促进可持续 AI 发展

---

> 💡 **总体评价**：  
> 本文提出的 **ProductResearch** 框架为构建下一代电商 Deep Research Agent 提供了一个**可扩展、高保真、强监督**的解决方案，在学术与工业界均有重要启示意义。

</details>

---

### 10. [Reasoning-Driven Multimodal LLM for Domain Generalization](https://arxiv.org/abs/2602.23777)

**Authors**: Zhipeng Xu, Zilong Wang, Xinyang Jiang, Dongsheng Li, De Cheng, Nannan Wang  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.23777v1  

#### Abstract
This paper addresses the domain generalization (DG) problem in deep learning. While most DG methods focus on enforcing visual feature invariance, we leverage the reasoning capability of multimodal large language models (MLLMs) and explore the potential of constructing reasoning chains that derives i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Reasoning-Driven Multimodal LLM for Domain Generalization**

---

## **1. 主要贡献和创新点**

### **解决的问题**
该论文聚焦于**Domain Generalization (DG)** 中的一个核心挑战：在存在显著域偏移（domain shift）的情况下，模型如何实现对未见目标域的鲁棒泛化。传统DG方法大多依赖于学习**特征层面的不变性**（feature-level invariance），例如通过对抗训练、风格混合等方式减少域间差异。然而，这些方法往往忽略了更高层次的、**过程层面的不变性**（process-level invariance），即人类在跨域推理时所依赖的逻辑链条。

作者指出，仅靠视觉特征难以捕捉跨域稳定的语义线索，而**多模态大语言模型**（MLLMs）具备强大的**推理能力**（reasoning capability），可以生成结构化的、类别相关的推理链（reasoning chains），从而为DG提供一种新的互补信号。

---

### **提出的新方法与新思路**
作者提出了 **RD-MLDG**（**Reasoning-Driven Multimodal LLM for Domain Generalization**），这是首个显式将**类别相关推理链**（class-relevant reasoning chains）整合到DG框架中的方法。

其核心思想是：  
> **利用推理链作为“认知锚点”**（cognitive anchors），引导模型学习一种跨域稳定、可解释的决策路径，而非仅仅依赖低层视觉特征。

为此，作者构建了两个关键模块：

#### **(1) MTCT (Multi-Task Cross-Training)**
- **动机**：直接用推理链进行监督训练（reasoning-chain supervision）比直接标签预测更难优化，因为模型需要先拟合复杂的中间推理步骤，再预测最终标签。
- **解决方案**：引入一个并行的**直接分类路径**（direct classification pathway），与推理增强路径联合训练。
- **作用**：直接分类任务提供了简单且稳定的梯度信号，帮助引导推理路径的学习，缓解优化困难。

#### **(2) SARR (Self-Aligned Reasoning Regularization)**
- **动机**：由GPT-4o等外部大模型生成的推理链虽然语义丰富，但其推理模式（reasoning patterns）与待微调的MLLM（如InternVL）不一致，导致“推理模式错配”（reasoning-pattern mismatch）。
- **解决方案**：采用**自对齐的软自标注机制**（soft self-labeling）。模型用自己的推理输出生成新的推理链，并只保留那些最终结论正确的样本作为监督信号。
- **作用**：逐步将监督信号从“外部风格”迁移到“自身风格”，既保留了语义丰富性，又提高了可优化性。

---

### **相比现有方法的优势**
| 维度 | 传统DG方法 | RD-MLDG |
|------|-----------|---------|
| **监督信号** | 特征不变性（Feature-level） | 推理过程不变性（Process-level） |
| **可解释性** | 黑箱，难以理解 | 显式推理链，人类可读 |
| **泛化机制** | 隐式对齐视觉表示 | 显式建模决策逻辑 |
| **训练策略** | 复杂正则化/元学习 | 简单高效的两阶段SFT |
| **适用范围** | 多数基于CNN/ViT | 可扩展至任意MLLM |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 四个标准DG基准数据集，均来自 **DomainBed**：
  - **PACS**（4 domains, 7 classes）
  - **VLCS**（4 domains, 5 classes）
  - **OfficeHome**（4 domains, 65 classes）
  - **TerraInc**（4 domains, 10 classes）

此外，作者构建了一个新扩展数据集：**DomainBed-Reasoning**，为每个图像-标签对配备了由GPT-4o生成的五段式结构化推理链：
```
<SUMMARY>
<CAPTION>
<REASONING>
<REFLECTION>
<CONCLUSION>
```

---

### **实验设置与评估指标**
- **评估协议**：采用标准的 **leave-one-domain-out** 协议，报告所有目标域上的平均准确率（**Avg-acc**）。
- **训练方式**：**Supervised Fine-Tuning (SFT)**，使用LoRA进行参数高效微调。
- **模型架构**：以 **InternVL3-8B** 和 **InternVL3-2B** 为主干模型，也验证了在 **LLaVA-1.5-7B** 上的有效性。
- **SARR迭代次数**：N=3轮自标注。

---

### **基线方法对比**
论文对比了三类主流方法：
1. **ResNet-50 / ViT-B/16 基础方法**：
   - CORAL, MLDG, MixStyle, SWAD
2. **基于CLIP的方法**：
   - CoOp, MaPLe, SIMPLE+, CLIP-LoRA, DGCLDTP
3. **商用MLLM零样本性能**：
   - GPT-4o, Gemini, etc.

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**
| Method | PACS | VLCS | OfficeHome | TerraInc | **Avg** |
|--------|------|------|------------|----------|--------|
| GPT-4o (Zero-shot) | 97.83 | 85.41 | 90.12 | 60.49 | **83.46** |
| DGCLDTP (SOTA CLIP-based) | 97.03 | 84.79 | 87.65 | 63.27 | **83.19** |
| **InternVL3-8B + RD-MLDG (Ours)** | **98.13** | **87.03** | **91.73** | **70.65** | **86.89** |

> ✅ **RD-MLDG 在所有四个数据集上均达到SOTA，平均准确率提升达3.7个百分点以上**。

---

### **消融实验结果（Table 2）**
在 **OfficeHome** 和 **TerraInc** 上验证各组件有效性：

| 方法 | OfficeHome (Avg) | TerraInc (Avg) |
|------|------------------|----------------|
| + Reasoning only (Baseline) | 88.76 | 66.00 |
| + MTCT | 90.58 (+1.82) | 67.19 (+1.19) |
| + SARR | 90.91 (+2.15) | 65.29 (-0.71) |
| **+ MTCT + SARR (Full)** | **91.73 (+2.97)** | **70.65 (+4.65)** |

> 🔍 发现：
> - **MTCT** 显著提升推理链的可学习性；
> - **SARR** 对小模型（如InternVL3-2B）尤其有效，能大幅提升泛化能力；
> - 两者结合带来最大增益，证明协同效应。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **推理链具有更强的域不变性**：
   - 定量分析显示，推理链嵌入（text embeddings）的跨域分布差异（MMD）比视觉嵌入低 **58.6%**（Table A.3），说明推理链比原始像素更稳定。

2. ✅ **零样本推理有助于泛化，但直接蒸馏效果差**：
   - 零样本下加入推理提示可提升+43.28% 正确token概率；
   - 但在SFT中，纯推理监督反而不如直接标签监督 —— 揭示了“**优化鸿沟**”（optimization gap）的存在。

3. ✅ **推理模式错配是关键瓶颈**：
   - GPT-4o生成的推理链富含上下文细节，但熵高、难拟合；
   - 自生成推理链更聚焦类别，易优化但信息少；
   - SARR成功平衡二者，在保持语义的同时提升可学习性。

4. ✅ **方法具有任务通用性**：
   - 在 **VQA** 和 **Visual Entailment (VE)** 任务上同样有效（Table 8）；
   - 在 **Single Domain Generalization (SDG)** 设置下仍大幅领先（avg 89.37% vs 62.03%）；
   - 在 **base-to-new class generalization** 场景中，新类准确率从15.44%提升至24.33%，表明其支持**概念级迁移**。

---

### **局限性**
- **依赖高质量推理链生成**：当前推理链由GPT-4o生成，若生成质量下降，可能影响性能。
- **计算成本较高**：SARR需多轮自标注与重训练，增加训练时间。
- **对弱推理模型增益有限**：如LLaVA-1.5本身推理能力较弱，尽管有提升，但仍不及InternVL系列。

---

### **未来工作方向**
- 探索**无需外部LLM**的端到端推理链生成与优化；
- 将RD-MLDG应用于更多下游任务，如 **video understanding**, **medical diagnosis**；
- 研究如何让模型**主动拒绝错误推理**，进一步提升推理链可靠性；
- 结合**test-time adaptation**，实现动态推理调整。

---

## **总结**
> **RD-MLDG 开辟了一条全新的DG研究范式：从“学特征”转向“学思考”。**
>
> 它不仅在多个标准基准上实现了SOTA性能，更重要的是揭示了**推理过程本身是一种强大且可迁移的泛化信号**。该工作为构建更具鲁棒性、可解释性和通用性的AI系统提供了重要启示。

</details>

---

### 11. [CIRCLE: A Framework for Evaluating AI from a Real-World Lens](https://arxiv.org/abs/2602.24055)

**Authors**: Reva Schwartz, Carina Westling, Morgan Briggs, Marzieh Fadaee, Isar Nejadgholi, Matthew Holmes, Fariza Rashid, Maya Carlyle, Afaf Ta\"ik, Kyra Wilson, Peter Douglas, Theodora Skeadas, Gabriella Waters, Rumman Chowdhury, Thiago Lacerda  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.24055v1  

#### Abstract
This paper proposes CIRCLE, a six-stage, lifecycle-based framework to bridge the reality gap between model-centric performance metrics and AI's materialized outcomes in deployment. While existing frameworks like MLOps focus on system stability and benchmarks measure abstract capabilities, decision-m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CIRCLE: A Framework for Evaluating AI from a Real-World Lens

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前的 AI 评估生态系统存在显著的“**现实差距**”（reality gap）：

- 主流评估方法（如 MLOps、benchmarking）聚焦于 **model-centric** 性能指标，在受控环境中测试模型输出（primary effects），忽视了真实世界中复杂的用户行为、组织流程和社会影响。
- 决策者（如政策制定者、企业领导者、公众）真正关心的是 AI 在实际部署中的 **materialized outcomes**——即其对工作流、认知模式、生产力乃至社会结构的二级（secondary）和三级效应（tertiary effects）。
- 现有框架无法系统化地将利益相关者的关切转化为可测量信号，导致评估结果缺乏生态效度（ecological validity）、构念效度（construct validity）和后果效度（consequential validity）。

> **核心问题**：如何从“模型能做什么”转向“AI在现实中实际做了什么”。

---

### 🚀 提出的新方法或新思路

本文提出 **CIRCLE** 框架，一个六阶段、生命周期式的（lifecycle-based）AI评估框架，旨在弥合上述现实差距。

#### CIRCLE 全称：
**C**ontextualize, **I**dentify, **R**epresent, **C**ompare, **L**earn, and **E**xtend  
（情境化 → 识别 → 表征 → 对比 → 学习 → 延伸）

#### 三大方法论转变（Methodological Shifts）：

| 转变 | 描述 |
|------|------|
| **1. 将现实异质性视为信号而非噪声** | 不再消除用户多样性，而是将其作为衡量操作成功的关键信号。例如，教师是否过度依赖 AI 生成测验，是评估教育 AI 成败的重要指标。 |
| **2. 引入社会技术视角（socio-technical frame）** | 评估不再局限于脚本化的 in silico 测试，而是关注人在真实工作流中与系统的互动，捕捉如“认知卸载”（cognitive offloading）、信任漂移等高阶效应。 |
| **3. 跨场景聚合知识** | 结合参与式设计的深度与工业测试的可重复性，使不同环境下的证据可比较、可扩展，支持规模化治理决策。 |

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（如 Benchmarking, MLOps） | CIRCLE 框架 |
|------|-------------------------------|-------------|
| **评估目标** | 抽象能力、技术稳定性 | 真实世界的 materialized outcomes |
| **评估范围** | 模型输出（primary effects） | 包括 secondary & tertiary effects（如教学质量下降、技能退化） |
| **用户角色** | 数据标注员或孤立评分者 | “情境专家”（situated experts），提供本地化洞察 |
| **证据性质** | 静态、一次性、实验室导向 | 动态、迭代、现场驱动（in situ） |
| **可扩展性** | 高（自动化测试） | 中高（结合人工与自动化） |
| **有效性保障** | 技术准确性 | 生态效度、构念效度、后果效度 |

> ✅ **优势总结**：CIRCLE 提供了一个**结构化、前瞻性、可追溯**的协议，将定性社会关切转化为定量可测指标，并通过持续监测实现闭环反馈。

---

## 2. 核心实验方法和设置

> ⚠️ 注意：该论文为**概念性框架提案**，未进行传统意义上的“实验”（无代码、无训练、无数值对比）。以下“实验”应理解为“方法验证示例”或“应用案例模拟”。

### 📌 使用的案例场景（非数据集）

论文以 **EdTech（教育科技）场景**为例，具体如下：

- **系统类型**：Generative AI 教学助手（AI chatbot）
- **部署环境**：中小学课堂
- **核心关切**：教师和学生可能过度依赖 AI 输出（如自动生成测验、作业批改建议），导致：
  - 教学质量下降
  - 学生学习动机减弱
  - 教师专业判断力退化
  - 课程完整性受损

此案例贯穿六个阶段，用于演示 CIRCLE 如何运作。

---

### 🧪 方法设置与流程（六阶段详解）

| 阶段 | 方法描述 | 输入 | 输出 |
|------|--------|------|------|
| **Stage 1: Contextualize**<br>情境化 | 通过访谈、研讨会等方式收集利益相关者（教师、校长、家长、监管机构）的关切，形成“构念”（constructs）如“过度依赖”、“认知卸载”。 | 利益相关者访谈记录、政策文件、课程标准 | **Context Brief**（情境简报）：列出关键构念及其定义 |
| **Stage 2: Identify**<br>识别 | 设计评估方案，选择合适的方法组合：<br>- **Field Testing**（实地测试）<br>- **Red Teaming**（红队攻击）<br>- **Longitudinal Studies**（纵向研究）<br>- **Controlled Pilots**（对照试点） | Context Brief | **Evaluation Design Plan**（评估设计计划）：含采样策略、任务设计、指标体系 |
| **Stage 3: Represent**<br>表征 | 执行真实世界试验（real-world trials），招募多样化的教师群体，在真实教学中使用 AI 工具，记录交互日志、行为变化、主观反馈。 | Evaluation Design Plan | **Evaluation Execution Plan** + 原始数据（transcripts, logs, surveys） |
| **Stage 4: Compare**<br>对比 | 构建分析工具：<br>- 编码规则（coding rubrics）<br>- 代理场景（proxy scenarios）<br>- 指标映射（metrics mapping）<br>将观察到的行为（如“未审核 AI 生成测验”）链接到构念。 | 执行数据 + 构念定义 | **Findings Synthesis Report**：量化各构念出现频率及趋势 |
| **Stage 5: Learn**<br>学习 | 向利益相关者解释结果，撰写面向不同受众的报告（学术期刊、行业白皮书、内部简报），推动行动。 | 分析报告 | **Stakeholder Insights Brief**：可操作的见解与建议 |
| **Stage 6: Extend**<br>延伸 | 建立持续监控机制，跟踪 AI 使用模式、学习成果变化、伦理合规情况，动态更新评估参数。 | Insights Brief | **Continuous Monitoring Plan**：含监测阈值、预警机制、调整预案 |

> 💡 特别说明：CIRCLE 支持混合模式评估——**人类参与者的真实测试** 与 **大规模自动化测试** 并行运行，确保结果既具生态效度又具统计广度。

---

### 📊 评估指标（非传统 accuracy/f1-score）

CIRCLE 的评估指标围绕“构念”展开，强调**操作化**（operationalization）过程：

| 构念（Construct） | 可观测行为（Observable Behavior） | 量化指标（Metric/Indicator） |
|------------------|-------------------------------|--------------------------|
| 过度依赖（Over-reliance） | 教师未审查 AI 生成的测验题 | % of quizzes未经人工检查直接使用 |
| 认知卸载（Cognitive Offloading） | 学生直接复制 AI 回答而不思考 | 学生提交答案与 AI 输出相似度（BLEU/ROUGE） |
| 教学自主性丧失 | 教师放弃原有教学节奏跟随 AI 推荐 | 教学计划偏离原始教案的程度 |
| 信任校准失衡 | 用户忽略已知错误仍采纳 AI 建议 | 错误提示后继续使用的比例 |

这些指标均源自 Stage 1 的 stakeholder concerns，并通过 Stage 4 显式映射。

---

### ❌ 基线方法对比（文中未提供数值对比）

由于 CIRCLE 是一个**元框架**（meta-framework），并非具体算法或模型，因此没有与其他“基线方法”进行端到端性能对比。

但文中明确指出其与以下方法的区别：

| 方法 | 与 CIRCLE 的区别 |
|------|----------------|
| **Benchmarking** | 仅测 primary effects；CIRCLE 测 secondary/tertiary effects |
| **Participatory Design** | 局部深入但难以规模化；CIRCLE 支持跨场景聚合 |
| **Algorithmic Audits** | 多为事后回顾；CIRCLE 是前瞻性、持续性的 |
| **MLOps Monitoring** | 关注数据漂移、延迟；CIRCLE 关注社会影响漂移 |

---

## 3. 主要实验结果和性能指标

> ❗ 本节需澄清：**该论文未报告任何数值性能指标或消融实验**。所谓“结果”是指框架的应用逻辑与产出形式。

### ✅ 主要“结果”体现为方法论产出

| 阶段 | 输出成果 |
|------|--------|
| Stage 1 | 形成“过度依赖”、“认知卸载”等可操作构念 |
| Stage 2 | 设计出包含红队测试、纵向追踪的教学评估协议 |
| Stage 3 | 获取真实课堂中师生与 AI 的交互数据 |
| Stage 4 | 发现某类教师群体中 68% 未审核 AI 测验（假设值举例） |
| Stage 5 | 向学校管理层提交改进培训与审核机制的建议 |
| Stage 6 | 建立每学期一次的 AI 使用影响审计制度 |

> 示例图 Fig. 2 清晰展示了从“担忧过度依赖” → “定义构念” → “可观测行为” → “长期教学影响”的完整链条。

---

### 🔍 消融实验（Ablation Study）

❌ 文中未进行任何形式的消融实验。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **AI 的价值不能仅由模型能力决定**，必须考察其在真实环境中的 materialized outcomes。
2. **利益相关者的关切可以且应当被系统化地转化为科学测量对象**（via construct systematization & operationalization）。
3. **高阶效应（secondary & tertiary effects）只能通过 context-rich、longitudinal、mixed-methods 的方式捕捉**，纯自动化 benchmark 无法胜任。
4. **评估不应是一次性活动**，而应嵌入整个 AI 生命周期，形成“观察 → 解释 → 行动 → 再观察”的闭环。
5. **CIRCLE 增强了多种测量效度**：
   - **Ecological Validity**：在真实教室中测试
   - **Construct Validity**：从 stakeholder concern 出发定义构念
   - **Criterion Validity**：关联到真实业务结果（如教学效果）
   - **Consequential Validity**：主动追踪潜在危害
   - **Internal Validity**：通过 quasi-experimental 设计减少混淆

---

### ⚠️ 方法的局限性

1. **资源密集型**：需要跨学科团队（社会科学家、教育专家、工程师）、长时间投入、伦理审批，不适合快速原型验证。
2. **实施门槛高**：组织需具备 stakeholder engagement 能力、construct design 能力和 longitudinal tracking infrastructure。
3. **难以完全自动化**：虽支持自动化测试，但核心洞察仍依赖人类专家解读。
4. **推广成本高**：每个新场景都需要重新执行 Stage 1–2，难以“开箱即用”。

---

### 🔮 未来工作方向

1. **开发工具链降低摩擦**：
   - 创建标准化的 **user simulator** 模拟多样化用户行为
   - 构建共享的 **construct library**（如“cognitive offloading”通用定义与测量模板）
   - 开发 **automated context-aware testing pipelines**

2. **制度激励改革**：
   - 推动监管机构重视 deployment-level evidence 而非仅 model documentation
   - 鼓励企业投资 comprehensive measurement infrastructure

3. **提升可扩展性**：
   - 探索联邦式评估架构，在保护隐私前提下聚合多站点数据
   - 发展轻量级 context probing techniques 快速适配新场景

4. **建立 FRAME 社区**：
   - 论文末尾提到本工作属于 **FRAME**（Forum for Real-World AI Measurement and Evaluation）项目的一部分，未来将持续完善该框架并推动实践落地。

---

## 总结

| 维度 | 内容 |
|------|------|
| **论文定位** | 提出一种新的 AI 评估范式，强调从“真实世界镜头”审视 AI 影响 |
| **核心贡献** | CIRCLE 六阶段生命周期框架，连接 stakeholder concern 与 measurable outcomes |
| **方法特点** | 构念中心、社会技术整合、持续迭代、跨场景可比 |
| **适用场景** | 高风险领域 AI 部署前后的系统性影响评估（如医疗、教育、司法） |
| **现实意义** | 为政策制定者、企业管理者提供基于 materialized effects 的治理依据，超越“黑箱 benchmark”时代 |

> 📣 **一句话总结**：  
> CIRCLE 不是另一个 benchmark，而是一个让 AI 评估“落地”的操作系统——它把人们真正在乎的问题，变成了可以测量、追踪和行动的知识。

</details>

---

### 12. [U-CAN: Utility-Aware Contrastive Attenuation for Efficient Unlearning in Generative Recommendation](https://arxiv.org/abs/2602.23400)

**Authors**: Zezheng Wu, Rui Wang, Xinghe Cheng, Yang Shao, Qing Yang, Jiapu Wang, Jingwei Zhang  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.23400v1  

#### Abstract
Generative Recommendation (GenRec) typically leverages Large Language Models (LLMs) to redefine personalization as an instruction-driven sequence generation task. However, fine-tuning on user logs inadvertently encodes sensitive attributes into model parameters, raising critical privacy concerns. Ex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：U-CAN: Utility-Aware Contrastive Attenuation for Efficient Unlearning in Generative Recommendation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **Generative Recommendation (GenRec)** 系统中，Large Language Models (LLMs) 通过 fine-tuning 用户交互日志来提升个性化推荐能力，但这一过程会将用户的敏感属性（如偏好、行为模式）编码进模型参数中，引发严重的 **隐私泄露风险**。现有的 **Machine Unlearning (MU)** 方法在尝试“遗忘”这些敏感数据时，面临一个核心挑战——**Polysemy Dilemma**（多义性困境）：

- 敏感信息与通用推理能力在神经元层面高度纠缠（superimposed），难以分离。
- 传统方法（如梯度上升或硬剪枝）要么导致 **Catastrophic Forgetting**（破坏整体性能），要么造成 **Structural Damage**（破坏推理路径），无法实现精准遗忘。

### 提出的新方法：U-CAN
作者提出 **Utility-aware Contrastive AttenuatioN (U-CAN)**，一种面向 GenRec 场景的高效、精确的遗忘框架，其核心创新在于三个协同机制：

#### （1）**Contrastive Activation（对比激活分析）**
- 通过比较模型在 **forgetting set (Df)** 和 **retention set (Dr)** 上的激活差异，计算每个神经元的 **activation gap**。
- 识别出对敏感数据响应强烈但在保留数据上被抑制的“高风险神经元”，实现细粒度的风险定位。

#### （2）**Utility Significance（效用重要性校准）**
- 引入效用感知机制，结合 **weight magnitudes** 和 **retention-set activation norms**，为每个维度分配效用分数。
- 防止因单纯追求隐私擦除而误删对通用推荐能力至关重要的参数。

#### （3）**Adaptive Soft Attenuation（自适应软衰减）**
- 不采用破坏性的 **binary pruning**（硬剪枝），而是设计了一个可微分的衰减函数，对高风险 LoRA 参数进行 **连续、按维度缩放**。
- 在抑制敏感检索路径的同时，**保持推理电路的拓扑连通性**，避免性能崩溃。

### 相比现有方法的优势
| 方法 | 主要缺陷 | U-CAN 的优势 |
|------|----------|--------------|
| **Gradient Ascent (GA)** | 方向坍塌（Directional Collapse），更新溢出至共享参数 | 基于激活差异定位，干预更局部化 |
| **NPO** | 同样存在梯度扰动问题，效用损失大 | 非梯度优化，无反向传播开销 |
| **LLM-Eraser (Pruning-based)** | 硬剪枝破坏结构，导致推理路径断裂 | 软衰减保持连接，结构完整 |
| **Retraining** | 计算成本极高，不可用于实时请求 | 单次前向传播即可完成，效率极高 |

> ✅ **核心优势总结**：U-CAN 实现了 **privacy-utility-efficiency** 三者的良好平衡。

---

## 2. 核心实验方法和设置

### 数据集
- **ML-100k**：电影推荐数据集，约 100K 用户-项目交互。
- **Pantry**：Amazon Reviews 子集，涵盖杂货和家居用品，共 32,992 种商品。

> 所有数据经过 5-core 过滤，并按时间顺序划分，随机选取 25% 作为 **forgetting set (Df)**，其余 75% 为 **retention set (Dr)**。

### 实验设置
- **基础模型**：LlamaRec（基于 Llama-2-7b 的两阶段 GenRec 模型）。
- **适配方式**：LoRA（Low-Rank Adaptation），仅更新 adapter 参数。
- **遗忘目标**：移除 Df 的影响，使模型输出接近在 Dr 上重新训练的结果。

### 评估指标
从三个维度全面评估：

#### （1）**Unlearning Effectiveness（遗忘有效性）**
- **KL Divergence ↑**：衡量遗忘集上的分布偏移程度。
- **Prediction Shift (%) ↑**：遗忘集上预测结果发生变化的比例。
- **Perplexity (PPL) ↑**：遗忘序列的不确定性，越高表示记忆越弱。

#### （2）**Utility Preservation（效用保留）**
- **Recall@K, MRR@K, NDCG@K**（K=5,10）：标准推荐质量指标，在 **retention set** 上越高越好。
- **Trade-off@10**：综合指标，定义为：
  $$
  \text{Trade-off@10} = \Delta\%\text{Forget@10} - \Delta\%\text{Retain@10}
  $$
  值越大表示遗忘效果强且效用损失小。

#### （3）**Operational Efficiency（操作效率）**
- **Execution Time ↓**：完成遗忘所需时间。
- **Throughput (samples/sec) ↑**：每秒处理样本数。

### 基线方法对比
- **LlamaRec**：原始模型。
- **Retraining**：在 Dr 上从头训练，理论最优。
- **GA**：梯度上升法。
- **NPO**：负偏好优化。
- **LLM-Eraser**：基于选择性剪枝的先进方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 2）

| 模型 | Trade-off@10 (↑) | KL Divergence (↑) | PPL (↑) | Recall@10 (Retain, ↑) |
|------|------------------|--------------------|---------|------------------------|
| **U-CAN (Ours)** | **29.45 (ML)** / **14.64 (Pantry)** | **0.13 / 0.41** | **23.83 / 69.67** | **0.1098 / 0.0469** |
| Retraining | 13.94 / 1.55 | — | — | 0.1131 / 0.0318 |
| GA | -3.82 / -23.74 | ~0.00 | ~18.9 | 0.1030 / 0.0416 |
| NPO | -4.30 / -23.85 | ~0.00 | ~18.9 | 0.1030 / 0.0416 |
| LLM-Eraser | 17.49 / -7.13 | 0.11 / 0.14 | 21.86 / 18.71 | 0.1032 / 0.0406 |

> 🔍 **解读**：
> - U-CAN 在 **Trade-off@10** 上显著优于所有基线，表明其在遗忘与效用间取得最佳平衡。
> - KL Divergence 和 PPL 明显更高，说明其真正实现了 **深度记忆擦除**，而非表面掩盖。
> - 在 **Pantry** 上，U-CAN 的 PPL 达到 **69.67**，远高于其他方法（~18–22），显示其对遗忘数据的“置信度崩溃”。

### 与基线方法的对比结果
- ✅ **相比 GA/NPO**：U-CAN 避免了负 Trade-off 分数，效用保留更好，且真正改变了输出分布。
- ✅ **相比 LLM-Eraser**：虽然 LLM-Eraser 在遗忘上有一定效果，但 Trade-off 波动大，且在 Pantry 上效用下降明显；U-CAN 更稳定。
- ✅ **相比 Retraining**：U-CAN 无需重新训练，**单次前向传播即可完成**，但性能接近甚至部分超越 retraining。

### 消融实验结果（Ablation Study, Table 3）
验证了 U-CAN 三大组件的必要性：

| 变体 | KL Divergence (↓) | Recall@10 (Forget, ↓) | Recall@10 (Retain, ↑) |
|------|-------------------|------------------------|------------------------|
| **U-CAN (Full)** | **0.13 / 0.42** | **0.1435 / 0.0356** | **0.1098 / 0.0469** |
| w/o C (无对比激活) | 0.01 / 0.17 | 0.1502 / 0.0356 | 0.0770 / 0.0332 |
| w/o F (无效用校准) | 0.06 / 0.16 | 0.2070 / 0.0356 | 0.1197 / 0.0468 |
| w/o H (无软衰减) | 0.08 / 0.19 | 0.1536 / 0.0356 | 0.0820 / 0.0386 |

> 🔍 **发现**：
> - 移除 **Contrastive Activation (w/o C)** 导致遗忘效果急剧下降，说明激活差异是风险定位的关键。
> - 移除 **Utility Significance (w/o F)** 导致遗忘变弱，说明效用校准有助于聚焦真正敏感参数。
> - 移除 **Soft Attenuation (w/o H)** 导致效用大幅下降，验证了硬干预的破坏性。

---

## 4. 关键结论和发现

### 主要发现
1. **Polysemy Dilemma 是 GenRec 遗忘的核心障碍**：敏感信息与通用推理在参数空间中高度纠缠，传统方法难以解耦。
2. **Contrastive Activation + Utility Calibration 可实现精准定位**：通过激活差异识别高风险神经元，并结合效用评分过滤假阳性，实现“外科手术式”干预。
3. **Soft Attenuation 优于 Hard Pruning**：连续衰减能有效抑制敏感路径而不破坏网络结构，是保持效用的关键。
4. **U-CAN 实现高效遗忘**：无需反向传播或重训练，仅需一次前向传播和参数缩放，适合高频删除请求场景。

### 方法的局限性
1. **缺乏形式化隐私保证**：当前评估依赖经验指标（如 KL、PPL），未提供如 **differential privacy** 等形式化保障。
2. **局限于 LoRA 适配器**：方法设计针对 LoRA 结构，是否适用于全量微调或其他 PEFT 方法有待验证。
3. **未测试更强攻击者**：实验未考虑自适应攻击者（adaptive adversaries）的提取能力。
4. **领域泛化性待验证**：目前仅在电影和电商推荐场景验证，跨模态或多跳推理任务中的表现未知。

### 未来工作方向
- 将 U-CAN 扩展至 **multi-modal recommendation** 或 **LLM agent** 场景。
- 探索与 **differential privacy** 或 **certified unlearning** 的结合，提供更强理论保障。
- 研究 **动态遗忘策略**，支持连续、增量式的用户数据删除。
- 在更复杂任务（如对话推荐、因果推理）中验证其鲁棒性。

---

> 📌 **总结一句话**：  
> **U-CAN 通过“对比激活定位 + 效用感知校准 + 自适应软衰减”的三段式设计，在 GenRec 场景下实现了高效、精准且结构友好的机器遗忘，显著优于现有梯度或剪枝方法。**

</details>

---

### 13. [Normalisation and Initialisation Strategies for Graph Neural Networks in Blockchain Anomaly Detection](https://arxiv.org/abs/2602.23599)

**Authors**: Dang Sy Duy, Nguyen Duy Chien, Kapil Dev, Jeff Nijsse  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.23599v1  

#### Abstract
Graph neural networks (GNNs) offer a principled approach to financial fraud detection by jointly learning from node features and transaction graph topology. However, their effectiveness on real-world anti-money laundering (AML) benchmarks depends critically on training practices such as specifically...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Normalisation and Initialisation Strategies for Graph Neural Networks in Blockchain Anomaly Detection*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**图神经网络**（GNN）在区块链反洗钱**（AML）**任务中的训练实践问题，特别是**权重初始化**（weight initialisation）和**归一化策略**（normalisation）对模型性能的影响。尽管GNN在金融欺诈检测中展现出潜力，但其在真实场景下的有效性高度依赖于训练细节，而这些方面在现有研究中被广泛忽视。

具体而言，论文试图回答以下四个关键问题：
1. 交易图的哪些特性（如异配性、度分布偏斜、时间漂移）限制了GNN的有效性？
2. 初始化策略如何影响优化稳定性和收敛速度？
3. 图特定的归一化是否能更有效地缓解特征漂移和过平滑问题？
4. 不同GNN架构（GCN、GAT、GraphSAGE）下最有效的归一化方法是什么？

### 🚀 提出的新方法与思路
- **系统性消融分析**：首次在**Elliptic Bitcoin Dataset**上对三种主流GNN架构（GCN、GAT、GraphSAGE）进行了关于**Xavier初始化**和**GraphNorm**的系统性消融实验。
- **架构依赖性洞察**：提出并验证了一个核心观点——**初始化与归一化的效果是架构依赖的**，而非通用最优。
- **可复现框架**：发布了一个完整的、可复现的实验框架，包含：
  - 时间顺序划分（temporal data splits）
  - 固定随机种子（seeded runs）
  - 完整的超参数调优（Optuna）
  - 多次重采样评估以增强鲁棒性

### ⭐ 相比现有方法的优势
- **超越传统评估方式**：采用**AUPRC**作为主指标，更适合严重类别不平衡的AML场景，避免AUC-ROC或Accuracy带来的误导。
- **更贴近实际部署**：使用时间分割防止信息泄露，反映真实预测场景。
- **揭示训练工程的重要性**：表明即使不改变模型结构，仅通过合理的初始化和归一化选择，也能显著提升性能，尤其在GAT和GraphSAGE上。

---

## 2. 核心实验方法和设置

### 📊 数据集
- **Elliptic Bitcoin Dataset**：
  - 包含 203,769 个节点（交易）和 234,355 条有向边（支付流）
  - 每个节点有 166 维特征（原始特征 + 一跳聚合统计）
  - 节点标签：2% 非法（illicit），21% 合法（licit），77% 未知
  - 时间步划分为 49 期（每期约两周），用于建模时间演化

### 🔧 实验设置
- **图构建**：将有向交易图转换为无向图，添加自环，保留未标记节点用于消息传递。
- **时间分割**：
  - 训练集：前 29 个时间步
  - 验证集：第 30–39 步
  - 测试集：第 40–49 步
- **模型架构**：
  - 评估三种GNN：**GCN**, **GAT**, **GraphSAGE**
  - 架构设计统一：输入层 → 1–3 层GNN → GraphNorm + Dropout → Embedding层 → 分类头（Linear）
- **正则化与初始化**：
  - **初始化**：比较默认初始化 vs. **Xavier uniform**
  - **归一化**：引入**GraphNorm**（图级归一化），对比其与无归一化或默认BatchNorm的效果
- **超参数优化**：
  - 使用 **Optuna** 进行100轮自动调参（TPE + ASHA算法）
  - 优化目标：验证集上的 **AUPRC**

### 📈 评估指标
- **主指标**：**AUPRC**（Area Under Precision-Recall Curve）——应对类别极度不平衡
- **辅助指标**：
  - AUC-ROC
  - 高置信阈值下的 Precision、Recall、F1-score（90%, 99%, 99.9%分位数）
- **鲁棒性评估**：对测试集进行100次50%子采样，报告均值与标准差

---

## 3. 主要实验结果和性能指标

### 📌 关键性能数据（来自 Table 2）

| Model | 设置 | AUC | **AUPRC** |
|-------|------|-----|----------|
| **GCN** | Baseline | 0.8728 | **0.5993** |
|         | Xavier | 0.8740 | 0.5939 |
|         | GraphNorm + Xavier | 0.8736 | 0.5442 |
| **GAT** | Baseline | 0.8585 | 0.6022 |
|         | Xavier | 0.8486 | 0.6190 |
|         | **GraphNorm + Xavier** | **0.8700** | **0.6568** |
| **GraphSAGE** | Baseline | 0.8593 | 0.6551 |
|         | **Xavier** | **0.8826** | **0.6678** |
|         | GraphNorm + Xavier | 0.8755 | 0.6651 |

> 💡 最佳结果：
> - **GraphSAGE + Xavier**：AUPRC = **0.6678**
> - **GAT + GraphNorm + Xavier**：AUPRC = **0.6568**
> - **GCN**：Baseline表现最佳，其他改进无效甚至有害

### 🔍 与基线方法的对比结果
- **GraphSAGE** 在Xavier初始化下达到 **AUPRC 0.6678**，优于先前报道的 **0.6392**（Deprez et al. [7]）
- **GAT** 在结合GraphNorm与Xavier后，AUPRC提升 **+0.055**，AUC提升 **+0.012**
- **GCN** 对两种修改均不敏感，甚至加入GraphNorm导致AUPRC下降至0.5442（显著退化）

### 🔁 消融实验结果
- **GraphSAGE**：
  - Xavier初始化带来最大增益（+0.013 AUPRC）
  - 加入GraphNorm反而轻微降低性能 → 表明图归一化可能干扰其邻居聚合机制
- **GAT**：
  - 单独使用Xavier效果有限
  - **GraphNorm + Xavier组合效果最强** → 显示图归一化有助于稳定注意力机制在高阶度图中的行为
- **GCN**：
  - 默认初始化已足够有效
  - 归一化几乎无帮助，且GraphNorm损害性能 → 可能源于其固有的邻接矩阵归一化已具备一定稳定性

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **初始化与归一化的收益是架构依赖的**：
   - **GraphSAGE**：受益最多于**Xavier初始化**
   - **GAT**：最需要**GraphNorm + Xavier**联合策略来稳定注意力训练
   - **GCN**：对两者均不敏感，**默认配置即最优**
2. **合理训练工程可媲美架构创新**：
   - 无需更换模型结构，仅通过调整初始化和归一化即可实现显著性能跃升（如GAT提升5.5% AUPRC）
3. **GraphNorm并非万能解药**：
   - 在GCN和GraphSAGE上未能带来收益，甚至有害
   - 强调应根据模型归纳偏置（inductive bias）选择训练策略
4. **AUPRC是更可靠的评估指标**：
   - 在极端不平衡的AML任务中，AUPRC比F1-score或AUC更能反映模型真实能力

### ⚠️ 方法的局限性
- **模型覆盖有限**：仅评估了GCN、GAT、GraphSAGE，未涵盖**Graph Transformers**、**Temporal GNNs**等更现代架构
- **数据简化假设**：Elliptic数据静态化处理，缺乏动态节点属性更新机制
- **硬件限制**：超参数搜索空间受限于100次Optuna trial，可能未达全局最优
- **忽略推理效率**：未评估延迟、内存占用、吞吐量等工业部署关键指标

### 🔮 未来工作方向
1. **标准化评估协议**：
   - 推动社区采用统一的时间划分、评估指标和阈值设定，便于跨研究比较
2. **探索高效训练方法**：
   - 结合**GLASS**等强化学习子图采样技术，提升大规模图训练效率
   - 开发轻量级、Python兼容的执行优化工具（如SALIENT++替代方案）
3. **扩展到新数据集**：
   - 在**Elliptic2**数据集上验证方法，支持子图级别洗钱模式识别
4. **多策略融合研究**：
   - 将本文提出的训练策略应用于ChebNet、GATv2等高性能架构，探索叠加效益

---

> 📌 **总结一句话**：  
> 本论文揭示了GNN在区块链异常检测中的性能极大程度取决于**架构适配的初始化与归一化策略**，提出“没有放之四海而皆准的最佳训练方式”，并提供了面向GCN、GAT、GraphSAGE的具体实践指南，强调了训练工程在现实应用中的关键作用。

</details>

---

### 14. [ODAR: Principled Adaptive Routing for LLM Reasoning via Active Inference](https://arxiv.org/abs/2602.23681)

**Authors**: Siyuan Ma, Bo Gao, Xiaojun Jia, Simeng Qin, Tianlin Li, Ke Ma, Xiaoshuang Jia, Wenqi Ren, Yang Liu  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.23681v1  

#### Abstract
The paradigm of large language model (LLM) reasoning is shifting from parameter scaling to test-time compute scaling, yet many existing approaches still rely on uniform brute-force sampling (for example, fixed best-of-N or self-consistency) that is costly, hard to attribute, and can trigger overthin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ODAR: Principled Adaptive Routing for LLM Reasoning via Active Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前大型语言模型（LLM）推理主要依赖于 **test-time compute scaling**，例如通过固定策略进行多次采样（如 Best-of-N、Self-Consistency），这些方法存在以下问题：
- **计算成本高**：对所有查询都使用相同的高开销推理路径，造成资源浪费。
- **效率低下**：简单任务被过度推理（"overthinking"），而复杂任务可能仍不足。
- **融合机制不透明**：多数方法采用启发式投票（heuristic voting），缺乏理论依据，难以归因。

### 提出了什么新方法或新思路
本文提出 **ODAR (Open-Domain Adaptive Reasoner)**，一种基于 **Active Inference** 和 **Free Energy Principle (FEP)** 的自适应路由框架，其核心思想是将推理视为一种**受控的资源分配过程**。

#### 主要创新点：
- **Difficulty Estimator (DE)**：一个轻量级模块，预测每个查询的难度 $d(x)\in[0,1]$，用于动态路由。
- **双代理架构（Dual-Agent Architecture）**：
  - **Fast Agent**（如 GPT-5.1）：低温度采样，快速生成假设。
  - **Slow Agent**（如 Claude-4.5 Sonnet）：高成本、深思熟虑的验证或探索。
- **基于 Free Energy 的融合机制（FEP Fusion）**：
  - 不再使用多数投票，而是选择 **最小化变分自由能（variational free energy）** 的答案。
  - 自由能公式为：  
    $$
    F(y|x) \approx -\frac{1}{L}\sum_{t=1}^L \log p(y_t|y_{<t},x) + \lambda \cdot \text{Var}[\log p(y_t|y_{<t},x)]
    $$
    - 第一项：平均负对数似然（准确性）
    - 第二项：**Varentropy**（方差），作为认知不确定性的代理，惩罚逻辑不稳定或“幻觉”的推理链。
- **神经科学启发的设计**：灵感来自大脑中的 **Theta-Gamma Phase-Amplitude Coupling (TG-PAC)**，即慢速控制信号（theta）门控高频爆发式处理（gamma），类比为难度估计器控制是否调用慢速代理。

### 相比现有方法的优势
- **更高的准确率-效率权衡**：在相同或更低的计算成本下，显著优于 Self-Consistency 等基线。
- **理论基础坚实**：将 Free Energy Principle 引入多智能体融合，提供了一个可解释、原则性的决策准则。
- **避免过思考**：通过难度感知路由，仅在必要时才启动高成本推理。
- **开源可复现**：提供了完全开源的实现（Open-ODAR），使用 Llama 4 + DeepSeek，证明其优势不依赖闭源模型。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
在 **23个多样化基准测试** 上进行了评估，覆盖8大类别：
- **数学**：MATH, GSM8K, **IMO 2025**, MathVista
- **常识**：ARC-Challenge, OpenBookQA, BoolQ, StrategyQA
- **知识问答**：MMLU-Pro, GPQA, HotpotQA
- **多跳推理**：MuSiQue
- **多模态**：AOK-VQA, MMMU-Pro
- **高级认知**：BBH, BBEH, TruthfulQA, **HLE (Humanity's Last Exam)**
- **编码**：SWE-bench, LIVEBENCH
- **指令遵循**：IFEval
- **抽象推理**：ARC-AGI-2

### 实验设置和评估指标
- **评估协议**：统一的 **compute-matched** 协议，确保公平比较。
- **主要指标**：
  - **Accuracy / Pass@1**：任务特定的准确率。
  - **计算成本**：以标准化的 **inference call 数量** 或 **token 消耗** 衡量。
  - **效率得分（Efficiency Score）**：$(\text{Avg. Acc} - \text{Baseline Acc}) / \text{Avg. Cost}$

### 基线方法对比
- **单模型基线**：GPT-5.1, Claude-4.5
- **多候选策略**：
  - **Self-Consistency (SC)**
  - **Best-of-N (N=5)**
- **SOTA 效率方法**：
  - **TOPS** (Yang et al., 2025)
  - **Stop Spinning** (Wei et al., 2025)

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **平均准确率**：ODAR 在 23 个基准上达到 **89.6%** 平均准确率。
- **数学领域**：
  - **MATH**: **98.2%**（+6.4% vs Self-Consistency）
  - **IMO 2025**: **68.7%**（+20.2% vs GPT-5.1）
- **高级认知**：
  - **HLE (Humanity's Last Exam)**: **54.8%**（+12.0% vs Self-Consistency）
  - **GPQA**: **78.5%**（+10.0% vs Self-Consistency）
- **计算效率**：
  - 相比 Self-Consistency，**计算成本降低 1.78×**（2.55× vs 5.0×）。
  - 在开源栈（Open-ODAR, Llama 4 + DeepSeek）上，相比开源 Self-Consistency，**计算成本降低 82%**。

### 与基线方法的对比结果
| 方法 | MATH | BBH | MMLU-Pro | 成本 | 效率 |
|------|------|-----|----------|------|------|
| GPT-5.1 | 91.8 | 88.0 | 86.2 | 1.0× | — |
| Self-Consistency | 94.5 | 91.2 | 89.5 | 5.0× | 0.6 |
| **ODAR (Ours)** | **98.2** | **95.4** | **94.2** | **2.55×** | **2.8** |

> ODAR 在 **22/23** 个数据集上取得 SOTA，且效率得分远超其他方法。

### 消融实验结果
- **移除 Difficulty Estimator (DE)**：
  - 导致 **135% 的成本爆炸**（从 2.55× 到 6.00×），证明 DE 是成本效益的关键。
- **移除 Slow Agent 或 FEP Fusion**：
  - 在 HLE 上性能下降 **12.0%**，表明深度推理和风险敏感融合对专家级任务至关重要。
- **随机路由（Proportion-matched Randomized Routing）**：
  - 性能下降 **8.3%**，证明 ODAR 的路由决策质量远超随机分配。
- **FEP 融合 vs 启发式融合**：
  - FEP 融合在 MATH 上比最佳启发式方法（Average Log-Prob）高出 **+0.9%**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **自适应路由优于固定策略**：LLM 推理不应“一刀切”，而应根据任务难度动态分配计算资源。
2. **Free Energy Principle 是有效的决策准则**：结合准确性和不确定性（varentropy）的 FEP 融合机制，能有效过滤“幻觉”并选择最稳健的推理路径。
3. **“Thinking-Optimal Scaling”**：最优的推理扩展不是简单增加计算量，而是**在哪里投入计算**。ODAR 找到了这个“思考最优”点。
4. **开源模型也能高效**：通过智能调度，**Open-ODAR** 在大幅降低成本的同时，性能超越了同规模的开源 Self-Consistency。

### 方法的局限性
1. **延迟问题**：Hard Path 的尾部延迟超过 60 秒，不适合实时应用。
2. **受限于基础模型能力**：66% 的错误源于模型本身的知识或推理缺陷，路由无法解决根本瓶颈。
3. **依赖 token-level log-probabilities**：FEP 融合需要访问完整的概率分布，在某些 API 或量化环境中不可行。

### 未来工作方向
- 开发无需 logit 的轻量级不确定性估计方法。
- 探索更细粒度的动态深度扩展（dynamic depth scaling）。
- 将该框架应用于多模态、具身智能等更复杂场景。
- 进一步优化 Hard Path 的并行化和延迟。

> **总结**：ODAR 通过将 **Active Inference** 和 **Free Energy Principle** 引入 LLM 推理，实现了**有原则的自适应路由**，在保持甚至提升性能的同时，大幅降低了计算成本，为构建高效、可靠的大模型推理系统提供了新范式。

</details>

---

### 15. [From Flat Logs to Causal Graphs: Hierarchical Failure Attribution for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2602.23701)

**Authors**: Yawen Wang, Wenjie Wu, Junjie Wang, Qing Wang  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.23701v1  

#### Abstract
LLM-powered Multi-Agent Systems (MAS) have demonstrated remarkable capabilities in complex domains but suffer from inherent fragility and opaque failure mechanisms. Existing failure attribution methods, whether relying on direct prompting, costly replays, or supervised fine-tuning, typically treat e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Flat Logs to Causal Graphs: Hierarchical Failure Attribution for LLM-based Multi-Agent Systems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
LLM-based Multi-Agent Systems (MAS) 在复杂任务中表现出色，但由于代理间复杂的交互和工具调用，系统存在**高失败率**（高达86.7%）和**失败机制不透明**的问题。  
现有 **Failure Attribution** 方法面临三大挑战：
- **Opaque Causal Flows**：日志为扁平文本，难以捕捉因果依赖。
- **Sparse Intermediate Supervision**：缺乏中间正确状态，定位错误如“大海捞针”。
- **Ambiguous Responsibility Boundaries**：错误表现位置 ≠ 错误引入位置，难分清是规划者（Orchestrator）还是执行者（Executor）的责任。

### 提出了什么新方法或新思路
提出 **CHIEF**（Causal HIErarchical Failure attribution），一个全新的故障归因框架，包含三个核心模块：

1. **Hierarchical Causal Graph (HCG) Construction**  
   将扁平轨迹解析为结构化图，节点分为：
   - **Subtask Node**：高层任务阶段（通过 RAG-based Task Decomposition + Trajectory-Aligned Reflection 生成）
   - **Agent Node**：基于 OTAR（Observation-Thought-Action-Result）模式的原子行为单元  
   边分为三种层级：
   - Subtask Edge（逻辑流程）
   - Agent Edge（协作关系）
   - Step Edge（数据流依赖）

2. **Hierarchical Oracle-Guided Backtracking**  
   构建虚拟 **Virtual Oracle** 作为每个子任务的理想验证标准（含 Goal, Preconditions, Key Evidence, Acceptance Criteria），实现自顶向下的搜索：
   - 先在 Subtask 层级缩小范围
   - 再到 Agent 层级
   - 最后精确定位到 Step 层级  
   显著减少搜索空间。

3. **Counterfactual Attribution via Progressive Causal Screening**  
   通过四阶段策略区分真实根因与传播症状：
   - **Local Attribution**：判断错误是否本地产生
   - **Planning-Control Attribution**：分析循环中是 planner 还是 executor 责任
   - **Data-Flow Attribution**：追踪数据污染源头
   - **Deviation-Aware Attribution**：过滤可自我修正的临时偏差

### 相比现有方法的优势
| 维度 | 现有方法 | CHIEF |
|------|--------|-------|
| 结构建模 | 扁平序列处理 | 显式构建 **Hierarchical Causal Graph** |
| 搜索效率 | 线性扫描或重复 replay | 自顶向下 **oracle-guided backtracking**，高效剪枝 |
| 成本 | Fine-tuning 高开销 / Replay 高 token 消耗 | **zero-shot**，无需训练，无 replay 开销 |
| 可解释性 | 黑箱判断或相关性统计 | 因果推理路径清晰，支持责任归属 |

---

## 2. 核心实验方法和设置

### 使用的数据集
采用目前唯一公开的 MAS 故障归因基准：**Who&When Benchmark**（Zhang et al., 2025d），包含 184 条失败轨迹，分为两个子集：
- **Hand-Crafted Subset**（58条）：来自 Magnetic-One 系统，人工构造，轨迹较长
- **Algorithm-Generated Subset**（126条）：由 CaptainAgent 自动生成，覆盖 126 种不同 MAS 架构

### 实验设置和评估指标
- **任务定义**：给定一条失败轨迹 $T$，目标是识别出导致失败的 **root cause agent-step pair $(i^*, t^*)$**
- **评估指标**：
  - **Agent-level Accuracy**：正确识别负责代理的比例
  - **Step-level Accuracy**：精确识别到错误步骤的比例
- **运行方式**：三次独立运行取平均，严格 top-1 排序
- **基础模型**：默认使用 **DeepSeek-V3.2 (thinking)**，也测试了 GPT-5.2、Claude 4.5 Sonnet、Gemini-3 等多款 LLM

### 基线方法对比
共比较 **8 个代表性基线**，分为四类：
1. **Heuristic**：
   - Random：随机选择
2. **LLM-based Prompting**：
   - All-at-once
   - Step-by-step
   - Binary Search
   - ECHO（带层次上下文和共识投票）
3. **Spectrum-based**：
   - FAMAS：基于多次 replay 的统计频谱分析
4. **Fine-tuning-based**：
   - AgenTracer：基于强化学习微调
   - GraphTracer：利用依赖图合成样本进行监督训练

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 方法 | Hand-Crafted – Agent (%) | Hand-Crafted – Step (%) | Alg.-Generated – Agent (%) | Alg.-Generated – Step (%) |
|------|--------------------------|-------------------------|----------------------------|----------------------------|
| Random | 12.00 | 4.20 | 29.10 | 19.10 |
| All-at-once | 50.00 | 5.17 | 61.11 | 13.49 |
| ECHO | 68.40 | 28.10 | 68.80 | 28.80 |
| FAMAS | 62.07 | **41.38** | 55.56 | 23.81 |
| GraphTracer | 74.91 | 28.63 | 76.64 | 49.97 |
| **CHIEF (Ours)** | **77.59** | **29.31** | **76.80** | **52.00** |

> 注：所有指标均为 **w/ ground truth** 设置下结果

#### 性能亮点：
- 在 **hand-crafted 子集** 上，CHIEF 实现 **77.59% agent-level 准确率**，显著优于第二名 GraphTracer（74.91%）
- 在 **algorithm-generated 子集** 上，CHIEF 在 **step-level 达到 52.00%**，领先 GraphTracer（49.97%）并远超其他方法
- 唯一在 **both subsets and both metrics** 上全面领先的 zero-cost 方法（相比 FAMAS 的高 replay 成本）

### 与基线方法的对比结果
- **vs LLM Prompting**：CHIEF 显著优于 All-at-once、Step-by-step 和 Binary Search，在 agent-level 平均提升 >20%
- **vs ECHO**：尽管 ECHO 使用了层次结构，但仅用于静态上下文表示；CHIEF 利用因果图指导 backtracking，更有效
- **vs FAMAS**：FAMAS 依赖昂贵的 trajectory replay（token 消耗达 CHIEF 的 6–8 倍），而 CHIEF 以“one-pass”推理实现更强性能
- **vs Fine-tuning Methods**：AgenTracer 和 GraphTracer 需大量标注数据和训练成本，泛化受限；CHIEF 直接 off-the-shelf 使用，性能更高

### 消融实验结果（Ablation Study）

| 变体 | Hand-Crafted – Agent (%) | Step (%) | Alg.-Generated – Agent (%) | Step (%) |
|------|---------------------------|----------|------------------------------|----------|
| All-at-once | 50.00 | 5.17 | 61.11 | 13.49 |
| Only M1 (HCG) | 37.93 | 18.96 | 66.66 | 24.60 |
| M1+M2 (HCG + Backtracking) | 51.72 | 22.41 | 61.11 | 34.12 |
| M1+M3 (HCG + Counterfactual) | 50.00 | 22.41 | 65.07 | 26.98 |
| **CHIEF (Full)** | **77.59** | **29.31** | **76.80** | **52.00** |

#### 发现：
- 单独使用 HCG（M1）反而可能降低性能（尤其在长轨迹上），说明**结构本身不足以解决问题**，需配合推理机制
- M1+M2 表现出良好搜索能力，但易将症状误判为根因
- M1+M3 能做反事实分析，但缺少 oracle 引导导致搜索低效
- **完整 CHIEF 各模块互补**：HCG 提供结构基础，Backtracking 缩小范围，Counterfactual 精确归因

---

## 4. 关键结论和发现

### 主要发现
1. **结构化先于推理**：将扁平日志转化为 **Hierarchical Causal Graph** 是提升可观察性和诊断精度的关键前提。
2. **虚拟 Oracle 是高效诊断的核心**：通过合成 intermediate supervision signal，实现了无需 replay 的高效 top-down 搜索。
3. **因果筛选优于相关性分析**：基于 Local / Planning-Control / Data-Flow / Deviation-Aware 的四阶段归因策略，能有效剥离传播效应，锁定真正 root cause。
4. **zero-shot 胜过 fine-tuning**：CHIEF 在无需任何训练的情况下，性能超过需要大规模合成数据和微调的先进方法（如 GraphTracer），表明**结构引导的推理潜力巨大**。

### 方法的局限性
1. **对 HCG 和 Virtual Oracle 的保真度敏感**：若 RAG 分解或 OTAR 解析出现幻觉，可能导致错误传播。
2. **当前仅支持单一致命错误归因**：假设失败由单一决定性错误引起，尚未验证在**累积性错误传播**场景下的有效性。
3. **依赖 Who&When 基准**：目前唯一可用的公开数据集，未来需在更多真实 MAS 系统中验证泛化性。

### 未来工作方向
- 扩展至 **multi-root cause detection**，支持复合错误归因
- 探索 **online failure diagnosis**，在运行时实时检测与干预
- 将 CHIEF 输出用于 **自动修复机制**（如 feedback loop 或 agent re-planning）
- 构建更大规模、更多样化的 **public failure log dataset**

--- 

> ✅ **一句话总结**：CHIEF 通过构建 **Hierarchical Causal Graph** 并结合 **virtual oracle-guided backtracking** 与 **counterfactual screening**，首次实现了高效、精准且无需训练的 LLM-MAS 故障归因，在 Who&When 基准上全面超越八种主流方法。

</details>

---

### 16. [RUMAD: Reinforcement-Unifying Multi-Agent Debate](https://arxiv.org/abs/2602.23864)

**Authors**: Chao Wang, Han Lin, Huaze Tang, Huijing Lin, Wenbo Ding  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.23864v1  

#### Abstract
Multi-agent debate (MAD) systems leverage collective intelligence to enhance reasoning capabilities, yet existing approaches struggle to simultaneously optimize accuracy, consensus formation, and computational efficiency. Static topology methods lack adaptability to task complexity variations, while...

---

### 17. [Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume](https://arxiv.org/abs/2602.24195)

**Authors**: Gregory Kang Ruey Lau, Hieu Dao, Nicole Kan Hui Lin, Bryan Kian Hsiang Low  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.24195v1  

#### Abstract
Despite their capabilities, Multimodal Large Language Models (MLLMs) may produce plausible but erroneous outputs, hindering reliable deployment. Accurate uncertainty metrics could enable escalation of unreliable queries to human experts or larger models for improved performance. However, existing un...

---

### 18. [OPTIAGENT: A Physics-Driven Agentic Framework for Automated Optical Design](https://arxiv.org/abs/2602.23761)

**Authors**: Yuyu Geng, Lei Sun, Yao Gao, Xinxin Hu, Zhonghua Yi, Xiaolong Qian, Weijian Hu, Jian Bai, Kaiwei Wang  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.23761v1  

#### Abstract
Optical design is the process of configuring optical elements to precisely manipulate light for high-fidelity imaging. It is inherently a highly non-convex optimization problem that relies heavily on human heuristic expertise and domain-specific knowledge. While Large Language Models (LLMs) possess ...

---

### 19. [Actor-Critic Pretraining for Proximal Policy Optimization](https://arxiv.org/abs/2602.23804)

**Authors**: Andreas Kernbach, Amr Elsheikh, Nicolas Grupp, Ren\'e Nagel, Marco F. Huber  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.23804v1  

#### Abstract
Reinforcement learning (RL) actor-critic algorithms enable autonomous learning but often require a large number of environment interactions, which limits their applicability in robotics. Leveraging expert data can reduce the number of required environment interactions. A common approach is actor pre...

---

### 20. [Chunk-wise Attention Transducers for Fast and Accurate Streaming Speech-to-Text](https://arxiv.org/abs/2602.24245)

**Authors**: Hainan Xu, Vladimir Bataev, Travis M. Bartley, Jagadeesh Balam  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.24245v1  

#### Abstract
We propose Chunk-wise Attention Transducer (CHAT), a novel extension to RNN-T models that processes audio in fixed-size chunks while employing cross-attention within each chunk. This hybrid approach maintains RNN-T's streaming capability while introducing controlled flexibility for local alignment m...

---

### 21. [Multi-Objective Reinforcement Learning for Large-Scale Tote Allocation in Human-Robot Collaborative Fulfillment Centers](https://arxiv.org/abs/2602.24182)

**Authors**: Sikata Sengupta, Guangyi Liu, Omer Gottesman, Joseph W Durham, Michael Kearns, Aaron Roth, Michael Caldara  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2602.24182v1  

#### Abstract
Optimizing the consolidation process in container-based fulfillment centers requires trading off competing objectives such as processing speed, resource usage, and space utilization while adhering to a range of real-world operational constraints. This process involves moving items between containers...

---

### 22. [Truncated Step-Level Sampling with Process Rewards for Retrieval-Augmented Reasoning](https://arxiv.org/abs/2602.23440)

**Authors**: Chris Samarinas, Haw-Shiuan Chang, Hamed Zamani  
**Category**: cs.CL  
**Published**: 2026-03-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2602.23440v1  

#### Abstract
Training large language models to reason with search engines via reinforcement learning is hindered by a fundamental credit assignment problem: existing methods such as Search-R1 provide only a sparse outcome reward after an entire multi-step trajectory, making it infeasible to attribute success or ...

---

### 23. [Hestia: Hyperthread-Level Scheduling for Cloud Microservices with Interference-Aware Attention](https://arxiv.org/abs/2602.23758)

**Authors**: Dingyu Yang, Fanyong Kong, Jie Dai, Shiyou Qian, Shuangwei Li, Jian Cao, Guangtao Xue, Gang Chen  
**Category**: cs.DC  
**Published**: 2026-03-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.23758v1  

#### Abstract
Modern cloud servers routinely co-locate multiple latency-sensitive microservice instances to improve resource efficiency. However, the diversity of microservice behaviors, coupled with mutual performance interference under simultaneous multithreading (SMT), makes large-scale placement increasingly ...

---

### 24. [On the Convergence of Single-Loop Stochastic Bilevel Optimization with Approximate Implicit Differentiation](https://arxiv.org/abs/2602.23633)

**Authors**: Yubo Zhou, Luo Luo, Guang Dai, Haishan Ye  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2602.23633v1  

#### Abstract
Stochastic Bilevel Optimization has emerged as a fundamental framework for meta-learning and hyperparameter optimization. Despite the practical prevalence of single-loop algorithms--which update lower and upper variables concurrently--their theoretical understanding, particularly in the stochastic r...

---

### 25. [SleepLM: Natural-Language Intelligence for Human Sleep](https://arxiv.org/abs/2602.23605)

**Authors**: Zongzhe Xu, Zitao Shuai, Eideen Mozaffari, Ravi S. Aysola, Rajesh Kumar, Yuzhe Yang  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.23605v1  

#### Abstract
We present SleepLM, a family of sleep-language foundation models that enable human sleep alignment, interpretation, and interaction with natural language. Despite the critical role of sleep, learning-based sleep analysis systems operate in closed label spaces (e.g., predefined stages or events) and ...

---

### 26. [PseudoAct: Leveraging Pseudocode Synthesis for Flexible Planning and Action Control in Large Language Model Agents](https://arxiv.org/abs/2602.23668)

**Authors**: Yihan (Logon),  Wen, Xin Chen  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.23668v1  

#### Abstract
Large language model (LLM) agents typically rely on reactive decision-making paradigms such as ReAct, selecting actions conditioned on growing execution histories. While effective for short tasks, these approaches often lead to redundant tool usage, unstable reasoning, and high token consumption in ...

---

### 27. [Mixed Choice in Asynchronous Multiparty Session Types](https://arxiv.org/abs/2602.23927)

**Authors**: Laura Bocchi, Raymond Hu, Adriana Laura Voinea, Simon Thompson  
**Category**: cs.DC  
**Published**: 2026-03-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.23927v1  

#### Abstract
We present a multiparty session type (MST) framework with asynchronous mixed choice (MC). We propose a core construct for MC that allows transient inconsistencies in protocol state between distributed participants, but ensures all participants can always eventually reach a mutually consistent state....

---

### 28. [Global Interpretability via Automated Preprocessing: A Framework Inspired by Psychiatric Questionnaires](https://arxiv.org/abs/2602.23459)

**Authors**: Eric V. Strobl  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.23459v1  

#### Abstract
Psychiatric questionnaires are highly context sensitive and often only weakly predict subsequent symptom severity, which makes the prognostic relationship difficult to learn. Although flexible nonlinear models can improve predictive accuracy, their limited interpretability can erode clinical trust. ...

---

### 29. [pathsig: A GPU-Accelerated Library for Truncated and Projected Path Signatures](https://arxiv.org/abs/2602.24066)

**Authors**: Tobias Nygaard  
**Category**: cs.LG  
**Published**: 2026-03-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2602.24066v1  

#### Abstract
Path signatures provide a rich representation of sequential data, with strong theoretical guarantees and good performance in a variety of machine-learning tasks. While signatures have progressed from fixed feature extractors to trainable components of machine-learning models, existing libraries ofte...

---

### 30. [Artificial Agency Program: Curiosity, compression, and communication in agents](https://arxiv.org/abs/2602.24100)

**Authors**: Richard Csaky  
**Category**: cs.AI  
**Published**: 2026-03-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2602.24100v1  

#### Abstract
This paper presents the Artificial Agency Program (AAP), a position and research agenda for building AI systems as reality embedded, resource-bounded agents whose development is driven by curiosity-as-learning-progress under physical and computational constraints. The central thesis is that AI is mo...

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
