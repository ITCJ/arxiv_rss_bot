# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-10 06:13:26 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Agentic AI-Driven UAV Network Deployment: A LLM-Enhanced Exact Potential Game Approach](https://arxiv.org/abs/2603.07456)

**Authors**: Xin Tang, Qian Chen, Binhan Liao, Yaqi Zhang, Jianxin Chen, Changyuan Zhao, Junchuan Fan, Junxi Tian, Xiaohuan Li  
**Category**: cs.DC  
**Published**: 2026-03-10  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2603.07456v1  

#### Abstract
Unmanned Aerial Vehicular Networks (UAVNs) are envisioned to provide flexible connectivity, wide-area coverage, and low-latency services in dynamic environments. From an agentic artificial intelligence (Agentic AI) perspective, UAVNs naturally operate as multi-agent systems, where autonomous UAVs ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Agentic AI-Driven UAV Network Deployment: A LLM-Enhanced Exact Potential Game Approach*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**低空无人机网络（UAVN）部署优化**中的复杂挑战，特别是：
- **混合整数非凸（MINLP）建模难题**：UAVN 部署涉及离散变量（如链路连接、用户关联）和连续变量（如位置、功率），导致传统集中式求解方法计算复杂度高且难以扩展。
- **动态环境下的可扩展性与一致性不足**：启发式算法缺乏收敛保证，深度强化学习（DRL）依赖大量训练且泛化能力差，而现有博弈论方法在全局一致性和适应性上受限。
- **手动参数调优成本高**：传统优化方法需人工设定效用函数权重，难以适应多样化场景。

### 提出的新方法与创新思路
本文提出了一种**基于 Agentic AI 和大语言模型（LLM）增强的精确势博弈（Exact Potential Game, EPG）框架**，实现双空间尺度的分布式优化：

#### （1）**双空间尺度 EPG 优化框架**
- **大空间尺度（Large Spatial Scale）**：采用基于对数线性学习的 EPG（**L3-EPG**），优化 **UAV 间的离散链路拓扑**，目标是减少冗余链路、降低干扰并保持网络连通性。
- **小空间尺度（Small Spatial Scale）**：采用基于近似梯度的 EPG（**AG-EPG**），联合优化 **UAV 坐标、发射功率和地面用户（GU）关联**，以提升吞吐量、降低能耗与延迟。

#### （2）**LLM 增强的效用函数自动生成机制**
- 构建了一个面向 UAVN 拓扑优化的**领域知识库**，整合无线通信、博弈论与优化案例。
- 结合 **Retrieval-Augmented Generation (RAG)** 框架，利用 LLM 根据当前网络特征（如节点规模、用户分布、遮挡程度）**自动检索并生成效用函数及其权重系数**，显著降低人工调参需求，提升跨场景适应能力。

#### （3）完全分布式决策架构
所有算法仅依赖局部信息交互，无需中央控制器，增强了系统的鲁棒性、隐私性和可扩展性。

### 相比现有方法的优势
| 对比维度 | 本文方法（L3-EPG + AG-EPG + LLM） | 现有方法（如 BRD-EPG、GA、ETG 等） |
|--------|----------------------------------|-------------------------------|
| **建模能力** | 显式解耦离散与连续变量，分别优化 | 多为联合建模，易陷入局部最优 |
| **收敛性保障** | 基于 EPG，确保纯策略纳什均衡存在且收敛 | 启发式方法无严格收敛证明 |
| **适应性** | LLM 动态生成权重，支持多场景自适应 | 权重固定或需专家手动调整 |
| **部署可行性** | 分布式执行，适合动态 UAVN | 中心化方法通信开销大，单点故障风险高 |

---

## 2. 核心实验方法和设置

### 实验场景
构建一个包含 **10 架 UAV** 和 **20 名地面用户（GU）** 的低空通信网络，覆盖范围为 $10 \text{km} \times 10 \text{km}$ 的方形区域。

### 数据来源与仿真设置
- **非真实数据集**：通过随机均匀分布生成 GU 位置；UAV 初始位置由 K-Means 初始化。
- **信道模型**：
  - A2A（Air-to-Air）：自由空间路径损耗 + LoS/NLoS 判定
  - A2G（Air-to-Ground）：ITU-R P.1410 推荐的仰角相关 LoS 概率模型
- **飞行约束**：高度限制在 [100m, 300m]，最大通信半径 4000m

### 评估指标
| 指标 | 描述 |
|------|------|
| **Total Throughput** | 网络总吞吐量（A2A + A2G） |
| **Total Energy Consumption** | 所有 UAV 的通信与飞行能耗之和 |
| **End-to-End Latency** | 包括传输延迟与传播延迟的加权平均 |
| **Convergence Stability** | 局部效用变化 vs 势函数变化的相关性（$R^2$） |
| **Topology Sparsity** | 迭代过程中链路数量的变化趋势 |

### 基线方法对比
- **BRD-EPG**：基于最佳响应动态的标准 EPG 方法
- **BRD-NCG**：非合作博弈下的最佳响应求解器
- **Evolutionary Game (ETG)**：基于复制子动力学的演化博弈
- **Genetic Algorithm (GA)**：遗传算法作为全局搜索基准

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Fig. 8）
| 指标 | 本文方法（AG-EPG）表现 | 提升幅度（vs 最佳基线） |
|------|------------------------|--------------------------|
| **系统吞吐量** | 显著最高，随 UAV 数量增加持续领先 | 提升约 **8.4%** |
| **总能耗** | 在不同 UAV 数量下均为最低 | 降低约 **12–18%** |
| **端到端延迟** | 维持最低水平，增长最缓慢 | 减少约 **15%** |

> 注：随着 UAV 数量增加，所有算法的能耗和延迟均上升（因通信开销增大），但所提方法增幅最小。

### 收敛性验证（Fig. 5）
- **L3-EPG 与 AG-EPG** 的局部效用变化与全局势函数变化高度一致：
  - 相关系数接近 **1.0**
  - 决定系数 $R^2 > 0.999$
- 所有 UAV 的策略更新曲线最终趋于稳定，表明成功收敛至纳什均衡。

### 拓扑优化过程（Fig. 6）
- 初始全连接图 → 经 L3-EPG 迭代后形成稀疏但仍连通的骨干网
- 当 GU 移动时，UAV 能自适应调整位置与链路，绕过障碍物，维持服务质量

### LLM 知识检索效果（Fig. 4）
- **最优块大小为 4，检索数 $K=3$** 时达到最高检索精度
- 过大的块或 $K$ 值会引入噪声，反而降低生成质量
- 验证了 RAG 框架在语义匹配上的有效性

### 消融实验（隐含分析）
虽然未明确列出消融表，但从模块设计可推断以下关键组件作用：
- **L3-EPG** 是实现拓扑稀疏化的关键
- **AG-EPG** 显著提升了资源利用率与 QoS 性能
- **LLM-RAG 模块** 成功实现了从“意图”到“数学模型”的自动化转换，减少了人为干预

---

## 4. 关键结论和发现

### 主要发现
1. **双尺度分解有效破解 MINLP 难题**：将复杂的混合变量问题解耦为两个可处理的子问题，在保证性能的同时大幅提升求解效率。
2. **EPG 框架保障分布式收敛**：无论是离散还是连续优化阶段，所设计的势函数都能确保个体行为与全局目标一致，避免震荡与冲突。
3. **LLM 可作为智能决策增强器**：结合 RAG 的 LLM 能够理解网络部署意图，并基于专业知识库自动生成适配的效用函数权重，极大提升了系统的**自主性与泛化能力**。
4. **Agentic AI 赋能 UAVN 自组织**：每个 UAV 作为智能 Agent，能够感知环境、做出决策并与他人协作，体现了真正的 Agentic AI 特征。

### 方法的局限性
- **依赖高质量知识库建设**：LLM 的输出质量受限于预构建的知识库完整性与准确性。
- **实时性挑战**：尽管是分布式，但在大规模网络中仍需考虑 LLM 查询延迟对控制环路的影响。
- **仿真假设理想化**：未考虑极端天气、突发干扰或恶意攻击等现实扰动因素。

### 未来工作方向
- 将框架扩展至 **Space-Air-Ground Integrated Network (SAGIN)** 场景
- 引入 **多智能体 LLM 协同机制**，实现更复杂的任务分工与协商
- 探索 **轻量化 LLM 或本地化推理**，以满足边缘设备的实时性要求
- 加入 **安全性与抗干扰机制**，提升系统在对抗环境下的鲁棒性

--- 

> ✅ **总结一句话**：  
> 本论文开创性地将 **Agentic AI、EPG 博弈理论与 LLM 增强决策** 相融合，提出了一套高效、自适应、可扩展的 UAVN 部署优化框架，在能耗、延迟与吞吐量方面全面超越传统方法，为未来智能空联网提供了新的范式。

</details>

---

### 2. [Meta-RL with Shared Representations Enables Fast Adaptation in Energy Systems](https://arxiv.org/abs/2603.08418)

**Authors**: Th\'eo Zangato, Aomar Osmani, Pegah Alizadeh  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.08418v1  

#### Abstract
Meta-Reinforcement Learning addresses the critical limitations of conventional Reinforcement Learning in multi-task and non-stationary environments by enabling fast policy adaptation and improved generalization. We introduce a novel Meta-RL framework that integrates a bi-level optimization scheme wi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对传统 **Reinforcement Learning (RL)** 在 **Energy Management Systems (EMS)** 中存在的以下关键挑战：
- **样本效率低**：传统 RL 需要大量交互才能收敛，在真实世界能源系统中成本高昂。
- **泛化能力差**：难以适应不同建筑、季节、负荷模式等动态变化的任务。
- **缺乏跨任务知识共享机制**：现有 Meta-RL 方法在结构相似的 EMS 任务中未能充分利用共享表示。

### **提出的新方法与创新点**
作者提出了一种新的 **Meta-RL 框架——Critic Feature Extractor Meta Learning (CFE)**，其核心创新包括：

1. ✅ **共享特征提取器（Shared Feature Extractor, FE）**
   - 在 **actor 和 critic 网络之间共享一个可元学习的特征编码器**。
   - 提取跨任务通用的状态表示，提升表示学习效率并减少过拟合。
   - 允许 representation-level 的知识迁移，而非全模型参数更新。

2. ✅ **Actor 权重复用机制（Actor Reuse, AR）**
   - 存储每个任务训练后的 **task-specific actor 参数**。
   - 当相同或相似任务再次出现时，直接复用这些参数，避免重复探索。
   - 显著提高样本效率，尤其对具有长周期依赖的任务（如充放电策略）更有效。

3. ✅ **任务选择策略促进泛化**
   - 利用聚类方法识别建筑能耗行为的不同 profile。
   - 在元训练中平衡任务多样性与结构性相似性，增强模型鲁棒性和泛化能力。

### **相比现有方法的优势**
| 对比维度 | 现有方法（如 MAML、Reptile、CAVIA、RL²） | 本文 CFE 方法 |
|--------|------------------------------------------|----------------|
| **参数更新开销** | MAML 需要高成本的二阶梯度；Reptile 忽视任务特异性 | 使用一阶元学习 + 局部参数复用，高效且稳定 |
| **知识共享粒度** | 多为全网络或上下文向量共享 | 引入 **representation-level 共享**，更灵活精细 |
| **任务复访处理** | 不支持 actor 参数记忆 | 支持 **actor 权重存储与复用**，加速再适应 |
| **适用场景适配性** | 设计用于异构任务（navigation vs manipulation） | 特别优化于 **结构高度相似的 EMS 场景** |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. 📌 **CityLearn 开源数据集**  
   - 基于 OpenAI Gym 的 BEMS（Building Energy Management System）模拟环境。
   - 包含多个建筑及其 Energy Storage Units (ESUs)，支持多智能体控制。
   - 用于主实验与可视化分析。

2. 📌 **私有数据集（Proprietary Dataset）**
   - 来自 1,529 栋建筑的真实能耗数据（2018–2024，约 3000 万条记录）。
   - 覆盖住宅、办公、工业等多种建筑类型。
   - 包含气象、电价、社会经济等辅助信息。
   - 用于验证方法在复杂现实场景下的泛化能力。

### **实验设置**
- **元训练阶段**：
  - 每个 meta-iteration 采样 3 个任务组成的 batch。
  - 总共进行 600 个 meta-step。
  - 使用 **Proximal Policy Optimization (PPO)** 作为内循环 RL 算法。
  - 内循环每 2,048 步更新一次策略，共运行 100k 环境步。

- **元测试阶段**：
  - 在未见过的任务（held-out cluster）上评估快速适应能力。
  - 所有 agent 初始化为元学习得到的参数。
  - 测量早期阶段（前几十次梯度更新）的表现以评估“快速适应”。

- **任务采样机制**：
  - 前 10% meta-step 禁止重复访问任务，鼓励探索。
  - 后续采用多项式增长的概率允许任务重访，启用 AR 机制。

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Reward 曲线** | 累积奖励随环境步或梯度更新的变化趋势 |
| **收敛速度** | 达到特定性能水平所需的样本数（steps 或 gradient updates） |
| **最终性能** | 收敛后平均 reward / cost |
| **Ramping** | 连续时间步间电网用电波动（越小越好） |
| **Financial Cost** | 年度电费支出（相对于规则控制器归一化） |
| **Meta-gradient norm** | 衡量元参数稳定性与收敛性的代理指标 |

### **基线方法对比**
| 基线方法 | 简要说明 |
|---------|----------|
| **Random** | 从零开始训练的单任务 PPO agent |
| **Pretrained** | 在某一类建筑上预训练后迁移到新任务 |
| **Reptile** | 经典一阶元学习算法，无共享 FE 或 AR |
| **CAVIA** | 引入 context vectors 分离共享与任务专属参数 |
| **RL²** | 使用 LSTM 隐式积累跨任务经验 |
| **Meta-G1~G5** | 在不同距离集群上训练的变体，用于评估泛化边界 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | CFE（本文） | Reptile | Random Baseline |
|------|------------|--------|----------------|
| **达到 mean reward = -30 所需步数** | ~70k steps | ~150k steps | ~400k steps |
| **样本复杂度降低倍数** | ⬇️ **约 4×** vs Random | — | — |
| **最终年度电费（归一化）** | **0.86** | 0.87 | 0.95~1.03 |
| **Ramping 波动（归一化）** | **0.90** | 0.90 | 1.01~1.18 |
| **30 次梯度更新后充放电周期数** | **14.3 ± 5.6** | 14.8 ± 4.1 | 45.3 ± 15.3（无序） |

> 🔍 注：数值越低表示成本/波动越小，越高表示策略越有序。

### **与基线方法的对比结果**
- ✅ **CFE 显著优于所有 baseline**，特别是在**早期适应阶段**表现突出。
- 💡 **Reptile + FE + AR 组合效果最佳**，证明两个模块协同增益。
- ❌ **CAVIA 与 RL² 虽然稳定，但改进缓慢**，表现为“强泛化者”而非“快学习者”。
- 🔄 **Actor Reuse (AR) 在任务重现时显著加快收敛**，尤其适用于周期性调度任务。
- 📈 **共享 FE 极大提升了表示质量**，是性能提升的主要来源（见消融实验）。

### **消融实验结果（Ablation Study）**
如图 4(b) 所示，比较四种变体：

| 变体 | 最终 Reward | 收敛速度 | 说明 |
|------|-----------|--------|------|
| **Vanilla Reptile** | -48.5 | 慢 | 基础版本 |
| **Reptile + AR** | -48.7 | 略快 | actor 复用略有帮助 |
| **Reptile + FE** | -43.2 | 快 | FE 显著加速学习 |
| **CFE (Full Model)** | **-30.0** | **最快** | 完整模型最优 |

> 🔎 结论：**Feature Extractor 是最大贡献者**，Actor Reuse 提供额外加速，二者结合实现最佳性能。

此外，尝试使用基于 Transformer 的 FE（TS-based）：
- ✅ 更高的**渐近性能**（适合长期决策）
- ❌ 更慢的**适应速度**（因模型更深更大）
- ➖ 表明存在 **representation richness vs. adaptation speed 的权衡**

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **共享表示 + actor 复用能极大提升 Meta-RL 在 EMS 中的样本效率**。
2. ✅ **representation-level 的知识迁移比全模型微调更适合结构相似任务**。
3. ✅ **Actor 参数记忆机制对周期性强的任务至关重要**，可避免重复学习常见行为。
4. ✅ **元初始化使探索更具导向性**，快速发现有效的充放电策略。
5. ✅ **Meta-RL 的有效性依赖于任务间的结构相似性**；当目标任务偏离训练分布太远时，性能下降明显（Meta-G4/G5 < Random）。

### **方法的局限性**
- ⚠️ **假设任务间结构高度相似**：若目标任务动态差异过大（out-of-distribution），共享表示失效。
- ⚠️ **Actor 参数存储带来额外内存开销**：随着任务数量增加，需设计索引与检索机制。
- ⚠️ **当前 FE 为 MLP 结构**：虽轻量，但在捕捉长期时间依赖方面不如 RNN/Transformer。

### **未来工作方向**
- 🔮 探索 **probabilistic latent task representations**（例如 VAE-style context model）以增强对多样化任务的鲁棒性。
- 🔮 引入 **自动任务检索与匹配机制**，实现 actor 参数的条件复用。
- 🔮 结合 **offline Meta-RL** 技术，利用历史数据进一步提升样本效率。
- 🔮 将框架扩展至 **multi-agent EMS 控制**，研究分布式元学习架构。

---

> ✅ **总结一句话**：  
> 本论文提出的 **CFE 框架通过共享特征提取器和 actor 权重复用机制**，在真实建筑能源管理系统中实现了 **4 倍以上的样本效率提升**，显著优于现有 Meta-RL 方法，为智能能源控制提供了高效、可泛化的解决方案。

</details>

---

### 3. [SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization](https://arxiv.org/abs/2603.08185)

**Authors**: Yeonsik Park, Hyeonseong Kim, Seungkyu Choi  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.08185v1  

#### Abstract
Post-training quantization (PTQ) has emerged as a prevailing technique for deploying large language models (LLMs) efficiently in terms of both memory and computation, across edge devices and server platforms. Existing PTQ methods primarily aim to reduce precision in weights and activations by mitiga...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLMs）在部署时面临巨大的内存和计算开销，**Post-Training Quantization (PTQ)** 是一种高效的压缩技术。然而，在 **W4A4（4-bit weights, 4-bit activations）** 设置下，现有方法存在以下问题：
- **严重的精度下降**：由于激活和权重中的 **outlier** 导致量化误差显著。
- **低效的推理路径**：基于 LoRA 的低秩误差重建方法通常引入两个低秩因子（如 $L_1L_2$），导致需要中间量化步骤，破坏了端到端的低精度计算效率。
- **高校准成本**：旋转变换类方法（如 Quarot、SpinQuant）虽然有效，但依赖昂贵的训练或优化过程。

### **提出了什么新方法或新思路**
本文提出 **SERQ (Saliency-Aware Error Reconstruction for Quantization)**，一种新颖的低秩误差重建框架，其核心创新在于：
- **单低秩补偿矩阵（Single Low-Rank Compensation Matrix）**：不同于传统 LoRA 使用两个低秩因子，SERQ 通过一个单一的低秩矩阵 $R$ 来联合补偿由 **weight saliency** 和 **activation saliency** 引起的量化误差。
- **三阶段流程**：
  1. **静态激活展平（Static Activation Flattening）**：采用 SmoothQuant 风格的通道缩放，将激活异常值的影响转移到权重中。
  2. **显著性感知误差重建（Saliency-Aware Error Reconstruction）**：识别出因缩放而变得“显著”的权重行（salient rows），并用单个低秩矩阵重建其量化误差。
  3. **离线权重置换（Offline Weight Permutation）**：将显著权重行排列至矩阵顶部，并通过前一层的列置换确保输入激活顺序对齐，所有操作均可离线完成。

### **相比现有方法的优势**
- ✅ **更高的精度**：在 W4A4 下显著优于现有 LoRA 和旋转类方法。
- ✅ **更低的延迟开销**：仅需一次额外的低秩乘法，避免了中间量化，支持纯 INT4/MXFP4 推理。
- ✅ **极低的校准复杂度**：无需梯度训练，仅需少量样本进行显著性分析，校准时间远低于旋转方法。
- ✅ **兼容性强**：可与 GPTQ、RTN 等主流权重量化方案结合。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **校准数据集（Calibration Set）**：
  - `WikiText-2`（128 个样本）
  - `The Pile`（用于敏感性分析）
- **评估任务与数据集**：
  - **零样本推理任务**：PIQA, SIQA, ARC-Easy/Challenge, HellaSwag, Winogrande, BoolQ, OpenBookQA
  - **综合基准**：MMLU
  - **困惑度（Perplexity）**：WikiText2 test set
  - **生成任务**：GSM8K（数学推理）、LongBench（长上下文理解）

### **实验设置和评估指标**
- **模型**：LLaMA-2（7B, 13B, 70B）、LLaMA-3（8B, 1B, 3B）、Qwen-2.5-3B
- **量化配置**：
  - W4A8（4-bit weights, 8-bit activations）
  - W4A4（4-bit weights, 4-bit activations）
  - 支持 INT4（RTN/GPTQ）和 MXFP4 格式
- **评估指标**：
  - **Perplexity (PPL ↓)**：越低越好
  - **Zero-shot Accuracy (%) ↑**：越高越好
  - **MMLU Score (%) ↑**
  - **推理延迟（Latency）**：每层额外开销（μs）
  - **端到端速度**：TTFT（Time to First Token）、TPOT（Time per Output Token）
  - **显存占用（Memory Usage）**

### **基线方法对比**
- **低秩分解类（Matrix Decomposition）**：
  - `LLM.int4()`：混合精度 outlier 处理
  - `L2QER`：SVD-based LoRA，当前最优 LoRA 量化方法
- **分布展平类（Distribution Flattening）**：
  - `SmoothQuant`
  - `QuaRot`：随机 Hadamard 变换
  - `SpinQuant`：学习型旋转矩阵

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 模型 | 方法 | W/A Bits | PPL ↓ | Zero-shot ↑ | MMLU ↑ |
|------|------|----------|--------|-------------|--------|
| LLaMA-2-7B | FP16 | 16/16 | 5.47 | 64.09 | 41.83 |
| LLaMA-2-7B | L2QER | 4/8 | 5.83 | 63.35 | 39.40 |
| LLaMA-2-7B | **SERQ (GPTQ)** | **4/8** | **5.59** | **63.04** | **40.29** |
| LLaMA-2-7B | L2QER | 4/4 | 7.37 | 57.67 | 29.63 |
| LLaMA-2-7B | **SERQ (GPTQ)** | **4/4** | **5.97** | **61.87** | **37.03** |

> 数据表明，SERQ 在 W4A4 下将 PPL 从 7.37 降至 5.97，MMLU 从 29.63 提升至 37.03，接近 W4A8 水平。

### **与基线方法的对比结果**
- ✅ **优于所有 LoRA 类方法**：
  - 在 W4A4 下，SERQ 比 L2QER 平均提升 **4–6 PPL** 和 **>10% MMLU**。
  - 在 LLaMA-3 系列上优势更明显，说明对小模型鲁棒性更强。
- ✅ **优于旋转类方法**：
  - 在 LLaMA-3-8B 上，SERQ 的 MMLU 达 **53.8%**，高于 SpinQuant 的 **49.93%**。
  - 校准时间仅 **~23 分钟**，远低于 SpinQuant（~598 分钟）和 FlatQuant（~131 分钟）。
- ✅ **推理效率更高**：
  - GPU 延迟分析显示，SERQ 的低秩路径开销比 L2QER 低 **4.5×**。
  - 在 Blackwell 架构上，端到端速度提升 **2×**，显存节省达 **2.48×**。

### **消融实验结果**
#### **(1) 秩大小（Rank Size）影响（Table 4）**
| Rank | LLaMA-3-8B PPL |
|------|----------------|
| 16   | 8.28           |
| 64   | 8.18           |
| 128  | **8.07**       |
| 256  | 7.98           |
- 结论：性能随秩增大单调提升，但在 **r=128** 后趋于饱和，为效率与精度平衡推荐值。

#### **(2) 校准数据敏感性（Table 5）**
- 使用不同数据集（WikiText vs Pile）和样本数（32–512）时，PPL 波动 < 0.2。
- 结论：**SERQ 对校准数据选择不敏感**，具备强鲁棒性。

#### **(3) 静态激活展平（SAF）作用（Table 7）**
- 在 Qwen-2.5-3B 上，移除 SAF 导致 W4A4 PPL 从 9.57 升至 10.83。
- 结论：**SAF 对小模型尤其重要**，能显著稳定量化误差。

---

## **4. 关键结论和发现**

### **主要发现**
1. **单低秩矩阵足以高效重建量化误差**：通过将激活显著性融入权重分析，SERQ 证明了无需双因子 LoRA 即可实现高保真误差补偿。
2. **显著性引导优于全局 SVD**：聚焦于 salient rows 能更有效地利用有限秩预算，避免误差稀释。
3. **端到端 4-bit 推理可行且高效**：SERQ 是首个实现 **pure INT4/MXFP4 matrix multiplication** 的低秩误差重建方法，无中间量化瓶颈。
4. **校准效率与精度可兼得**：无需训练即可达到甚至超越需大量优化的旋转方法。

### **方法的局限性**
- **依赖显著性假设**：性能受限于静态缩放能否准确捕捉动态显著性。
- **秩大小需手动设定**：虽有饱和趋势，但仍需根据模型调整。
- **未覆盖 KV Cache 量化**：实验中未对 Key-Value Cache 进行量化，可能影响长序列生成表现。

### **未来工作方向**
- 探索 **动态显著性检测** 机制以适应不同输入分布。
- 将 SERQ 扩展至 **KV Cache 量化** 与 **注意力算子优化**。
- 结合 **硬件感知调度**，进一步优化低秩路径的执行效率。
- 探索 **自动化秩分配策略**，实现 per-layer 自适应配置。

---

> 🔗 **代码地址**：[https://github.com/acalabys/SERQ](https://github.com/acalabys/SERQ)

</details>

---

### 4. [CDRRM: Contrast-Driven Rubric Generation for Reliable and Interpretable Reward Modeling](https://arxiv.org/abs/2603.08035)

**Authors**: Dengcan Liu, Fengkai Yang, Xiaohan Wang, Shurui Yan, Jiajun Chai, Jiahao Li, Yikun Ban, Zhendong Mao, Wei Lin, Guojun Yin  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.08035v1  

#### Abstract
Reward modeling is essential for aligning Large Language Models(LLMs) with human preferences, yet conventional reward models suffer from poor interpretability and heavy reliance on costly expert annotations. While recent rubric-based approaches enhance evaluation transparency, they lack systematic q...

---

### 5. [NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks](https://arxiv.org/abs/2603.06922)

**Authors**: Nandan Kumar Jha, Brandon Reagen  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.06922v1  

#### Abstract
We introduce NerVE, a unified eigenspectral framework for understanding how feed-forward networks (FFNs) in large language models (LLMs) organize and regulate information flow in high-dimensional latent space. Despite FFNs dominating the parameter budget, their high-dimensional dynamics remain poorl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# NerVE: Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
尽管 **Feed-Forward Networks (FFNs)** 在 **Large Language Models (LLMs)** 中占据绝大多数参数和计算开销，其在高维潜在空间中的非线性动态机制仍缺乏系统理解。现有研究多聚焦于 **Attention** 机制，而对 FFN 如何组织和调节信息流的几何本质知之甚少。

本文旨在填补这一空白，提出一个统一框架来解析 FFN 非线性如何重塑潜在空间的特征谱（eigenspectrum）结构。

### 提出了什么新方法或新思路
作者提出了 **NerVE (Nonlinear Eigenspectrum Dynamics in LLM Feed-Forward Networks)**，一个轻量级、内存高效的在线分析框架，通过追踪 FFN 层前后激活的协方差矩阵的 **eigenspectrum** 动态变化，揭示其内部工作机制。

NerVE 的核心是四个互补的、尺度不变的、分布感知的 **eigen-metrics**：
- **Spectral Entropy (SE)**：衡量特征值分布的均匀性（即分散程度），反映信息利用的广度。
- **Participation Ratio (PR)**：衡量有效维度数（effective dimensionality），反映有多少方向真正承载了信息。
- **Eigenvalue Early Enrichment (EEE)**：量化“前重性”（top-heaviness），即少数主导特征值集中了多少能量。
- **Jensen-Shannon Divergence (JS)**：衡量 FFN 非线性前后特征谱的分布偏移，反映非线性变换的重构强度。

### 相比现有方法的优势
- **系统性与可解释性**：不同于仅关注权重或注意力图的方法，NerVE 直接分析 FFN 内部激活的统计特性，提供了一个统一的几何视角。
- **轻量高效**：支持在线、逐层计算，无需额外训练，且通过优化策略（如顺序处理、仅计算特征值）实现内存效率。
- **诊断性强**：能够为架构设计（如 LayerNorm 位置、激活函数、位置编码）和优化器选择提供可操作的洞察，超越试错法。
- **跨架构通用性**：不仅适用于 Transformer，也验证了在 **MLP-Mixer** 架构上的有效性，表明其发现具有普适性。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **CodeParrot**：用于训练 GPT-2 和 LLaMA 变体，包含来自 GitHub 的 2000 万 Python 文件。
- **OpenWebText**：用于研究 RoPE 位置编码的影响。
- **FineWeb**：用于分析不同优化器（AdamW, Muon, Dion）的动力学。
- **C4**：用于训练 LLaMA 系列模型。
- **CIFAR-100**：用于在非 Transformer 架构 **MLP-Mixer** 上验证方法的通用性。

### 实验设置和评估指标
- **模型**：GPT-2 (71M 到 1.3B 参数)，LLaMA-style 模型，以及 MLP-Mixer B/16。
- **训练配置**：在 NVIDIA RTX 3090 GPU 上进行训练，上下文长度从 128 到 1024 不等。
- **评估方式**：
  - **在线监控**：在训练过程中定期记录每层 FFN 的 pre-activation 和 post-activation 激活。
  - **协方差计算**：将 `[B, S, D]` 的激活张量展平为 `[N, D]`（`N=B×S`），计算无偏样本协方差矩阵。
  - **特征分解**：对协方差矩阵进行特征分解，得到特征值。
  - **指标计算**：基于特征值计算 SE, PR, EEE, JS 四项指标。
- **相关性分析**：将 NerVE 指标与验证损失（validation loss）或困惑度（perplexity）进行皮尔逊相关性分析，以评估其诊断能力。

### 基线方法对比
本文并非提出一种新的训练算法，因此没有直接的“基线方法”在性能上进行对比。其对比体现在：
- **不同架构变体**：如 PreLN vs PostLN vs MixLN, GELU vs ReLU, 有无 LayerNorm, 使用 RoPE vs 无位置编码 (NoPE)。
- **不同优化器**：AdamW vs Muon vs Dion vs Adafactor vs SGD。
- **不同正则化技术**：Weight Normalization, Spectral Normalization, Hyperspherical Normalization。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1. **FFN 非线性的核心作用**：
   - FFN 的非线性（如 GELU, ReLU）并非简单缩放，而是主动 **reinject variance**（重新注入方差），将信息从少数主导方向扩散到更多潜在方向。
   - 表现为：**SE↑, PR↑, EEE↓, JS↑**。这打破了由 Attention 引起的秩坍塌（rank collapse），拓宽了潜在空间。

2. **架构设计的影响**：
   - **LayerNorm 位置**：**PreLN** 能最有效地将 FFN 宽度（width）转化为可用的有效维度（PR 最高且稳定），而 **PostLN** 在宽度增加时表现出递减的收益。
   - **位置编码**：**RoPE** 能有效防止中深层的谱坍塌（spectral collapse），维持更高的 PR，从而获得更低的困惑度（15.20 vs 16.78）。
   - **激活函数**：在无归一化的模型中，**ReLU** 能主动补偿 LayerNorm 的缺失，通过剧烈的方差重注入打破“谱惯性”（spectral inertia），而 GELU 则表现不佳。

3. **优化器的决定性影响**：
   - **AdamW**：导致 pre-activation 谱坍塌（pre-activation collapse），迫使 FFN 非线性进行大规模的“修复”（repair），消耗容量去恢复而非精炼。
   - **Muon**：能保持 pre-activation 谱的高维性和各向同性（high-dimensional and near-isotropic），极大减轻了 FFN 的负担，使其角色从“修复”转变为“精炼”（refinement），最终取得最佳性能。
   - **性能排序**：**Muon > Dion > AdamW**，与 post-activation 谱的平坦度（flatness）高度一致。

4. **跨架构验证**：
   - 在 **MLP-Mixer** 上，NerVE 同样观察到非线性导致 SE/PR 增加、EEE 下降的现象，证明了其发现的普适性。
   - 使用 **SGD** 优化器的 MLP-Mixer 比使用 **Adam** 的版本获得了更高的准确率（68.07% vs 66.96%），其谱分析显示 SGD 能更好地维持高维表示。

### 消融实验结果
- **归一化移除实验**：在无 LayerNorm 的 GPT-2 中，GELU 模型在早期层出现 **谱惯性**（EEE≈1, JS≈0），导致性能下降；而 ReLU 模型则通过剧烈的 PR 增益（高达 200-300 倍）成功补偿，缩小了与基线的困惑度差距约 50%。
- **FFN 权重几何约束**：
  - **Spectral Normalization**：诱导早期且持续的谱平坦化，性能最好。
  - **Hyperspherical Normalization**：导致早期过度扩张（early-overshooting），但缺乏持续控制，最终性能最差。
- **Token 位置效应**：在标准 GPT-2 中，后期 token 在深层 FFN 中被分配了显著更高的有效维度（PR），但在无归一化模型中，这种位置依赖性几乎消失。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **FFN 非线性是信息流的主动调节器**：其核心功能是通过 **variance reinjection** 打破潜在空间的各向异性，重新激活被抑制的方向，从而为下游层创造更丰富的表示。
2. **优化器塑造了 FFN 的角色**：优化器的几何属性（optimizer geometry）从根本上决定了 FFN 非线性的任务。AdamW 迫使 FFN 成为“救火队员”，而 Muon 则允许 FFN 成为“精修师”。
3. **架构选择留下独特的“谱签名”**：不同的设计选择（如 LayerNorm 位置、激活函数、位置编码）会在 FFN 的特征谱上留下可识别的模式，这些模式与模型的泛化能力强相关。
4. **NerVE 是强大的诊断工具**：其指标（尤其是 SE 和 PR）与验证损失高度相关（|r| > 0.9），可用于在线监控训练健康度，并在短预运行中对不同架构配置进行排名，避免完整训练的高昂成本。

### 方法的局限性
- **逐层独立性**：当前框架独立计算每层的指标，未显式建模层间的谱相干性（cross-layer spectral coherence）。
- **位置聚合**：将所有 token 视为独立同分布会掩盖位置相关的动态（如后期 token 的高 PR），需要分层分析来补充。
- **计算开销**：对于超大模型（如 FFN 维度 D > 10K），全批次协方差计算和特征分解可能成为瓶颈，需依赖采样或低秩近似，但这可能损害后激活指标的诊断效用。

### 未来工作方向
- 开发跨层谱相干性度量，以研究信息流的深度一致性。
- 探索更高效的近似算法，以扩展 NerVE 至千亿级模型。
- 将 NerVE 框架应用于其他神经网络组件（如 Attention head）或其他模态（如语音、多模态）。
- 利用 NerVE 的洞察指导新型优化器或架构的设计，例如开发能主动维持健康 pre-activation 谱的优化器。

</details>

---

### 6. [Ares: Adaptive Reasoning Effort Selection for Efficient LLM Agents](https://arxiv.org/abs/2603.07915)

**Authors**: Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, Shiyu Chang  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.07915v1  

#### Abstract
Modern agents powered by thinking LLMs achieve high accuracy through long chain-of-thought reasoning but incur substantial inference costs. While many LLMs now support configurable reasoning levels (e.g., high/medium/low), static strategies are often ineffective: using low-effort modes at every step...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ARES: Adaptive Reasoning Effort Selection for Efficient LLM Agents —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代基于 **thinking LLMs** 的智能体（agents）通过长链式思维（chain-of-thought, CoT）推理实现了高准确率，但也带来了巨大的推理开销（inference cost）。虽然当前许多 LLM 支持配置不同的“思考级别”（如 high/medium/low），但静态策略效果不佳：
- **始终使用低推理努力（low effort）**：显著降低性能；
- **随机选择或固定策略**：无法有效平衡成本与准确性。

根本问题在于：**并非所有任务步骤都需要同等强度的推理**。例如：
- 打开目标 URL → 简单，适合低 effort；
- 导航复杂网站结构 → 复杂，需 high effort。

因此，如何在多步 agent 任务中实现**动态、自适应的推理努力分配**，成为关键挑战。

---

### 🚀 提出的新方法：ARES 框架
作者提出 **ARES**（Adaptive Reasoning Effort Selection），一种为多步 LLM Agent 设计的**逐步骤动态推理努力选择框架**。

#### 核心思想
- 引入一个轻量级的 **Router LLM**，根据交互历史预测每一步所需的**最低合适推理等级**（low/medium/high）。
- 主 Agent 使用该等级进行下一步决策，从而在保证成功率的前提下最小化推理 token 消耗。

#### 创新点
1. **动态而非静态控制**  
   不同于全局设定推理模式，ARES 在每个 step 动态调整，更精细地匹配任务需求。

2. **基于 intra-model thinking levels 的路由机制**  
   路由发生在同一模型内部的不同推理模式之间（如 gpt-oss-20b 的 high/low 模式），而非跨不同模型路由（multi-model routing），优势包括：
   - 避免 KV Cache 重计算，减少额外延迟；
   - 性能-成本关系更稳定、可预测；
   - 易于集成到现有系统中（plug-and-play）。

3. **高质量训练数据生成流程**
   - 多阶段自动化 pipeline：
     1. **Trajectory Collection**：收集高质量成功轨迹；
     2. **Effort Annotation**：对每一步测试不同推理等级，找出能稳定产生正确动作的最低等级；
     3. **Rationale Generation**：让教师模型生成为何选择该等级的理由，提升 router 可解释性和准确性。

4. **结合 SFT + RL 的训练范式**
   - 先用监督学习（SFT）教会 router 学习最小充分 effort；
   - 再用强化学习（GRPO）优化长期效率-准确性的权衡，避免贪婪短视。

---

### 🔍 相比现有方法的优势
| 对比维度 | 现有方法（如 Model Routing） | ARES |
|--------|--------------------------|------|
| 路由粒度 | 跨模型（不同 size/capability） | 同一模型内不同 thinking level |
| 成本控制 | 不可预测，非单调 | 更一致的成本-性能曲线 |
| 推理效率 | 需重新编码上下文，KV cache 无法复用 | 可复用 KV cache，节省大量 token |
| 应用场景 | 单轮任务为主 | 多轮、依赖性强的 agent 任务 |
| 集成难度 | 高（需维护多个模型） | 低（仅需一个轻量 router） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在三个多样化 agent 任务上评估 ARES：

| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **TAU-Bench** | 工具使用 agent（tool-use） | 用户模拟器与 agent 交互完成零售、航空领域任务（如订票），通过数据库状态判断成功与否 |
| **BrowseComp-Plus** | 深度研究 agent（deep-research） | 多步检索与推理任务，使用固定语料库确保可复现性，评估搜索与答案生成能力 |
| **WebArena** | 网页导航 agent（web agent） | 在真实功能网页（电商、论坛等）中执行点击、滚动等操作，观测 accessibility tree |

---

### ⚙️ 实验设置
- **主干模型（backbone LLM）**：`gpt-oss-20b`
- **Router 模型**：轻量级 `Qwen3-1.7B`，经 SFT 和 RL 微调
- **推理等级选项**：low / medium / high（对应不同 thinking depth 和 token 开销）
- **训练方式**：
  - **SFT**：基于标注数据训练 router 预测 effort label 和 rationale
  - **RL**：使用 GRPO 算法进一步优化，奖励函数包含：
    - 成功奖励（+5.0）
    - 成本惩罚（根据所选 effort 加权）
    - 格式合规奖励（确保输出符合模板）

---

### 📊 评估指标
| 类别 | 指标 |
|-----|------|
| **性能** | - TAU-Bench：Accuracy (%)<br>- BrowseComp-Plus：Accuracy (%)<br>- WebArena：Task Success Rate (%) |
| **效率** | - `T_total`：总推理 token 数<br>- `T_task`：平均每任务推理 token 数<br>- `T_step`：平均每步推理 token 数 |

---

### 🆚 基线方法对比
| 基线类型 | 方法 |
|--------|------|
| 固定策略 | Low / Medium / High effort（全程统一） |
| 随机策略 | Random：每步随机选择 effort level |
| 规则基线 | Rule-based：人工设计规则切换 level |
| Prompting-based | 使用 GPT-5 或 Gemini-3-Pro 作为 router 进行 prompt 推理选择 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| 方法 | TAU-Bench (Retail) | TAU-Bench (Airline) | BrowseComp-Plus | WebArena |
|------|--------------------|---------------------|------------------|---------|
| **High effort** | 54.8% | 36.0% | 42.7% | 45.0% |
| **ARES (SFT)** | **54.8%** | **36.0%** | **41.3%** | **46.5%** ✅ |
| **推理 token 减少** | ↓35.2% (`T_total`) | ↓22.8% | ↓41.8% | ↓45.3% |

> ✅ 特别值得注意的是，在 **WebArena** 上 ARES 不仅大幅降本，还**超越了 high effort 基线**（46.5% vs 45.0%），说明过度推理可能导致“overthinking”错误。

---

### 🔁 RL 优化带来的进一步提升（Table 2）

| 方法 | TAU-Bench Retail | TAU-Bench Airline |
|------|------------------|-------------------|
| ARES (SFT) | 54.8%, 652k tokens | 36.0%, 678k tokens |
| ARES (**RL**) | **58.5%**, **476k tokens** (+3.7%, -27%) | **42.0%**, **133k tokens** (+6.0%, -80%) |

> 💡 RL 显著提升了性能并进一步压缩成本，尤其是在存在“overthinking”现象的 Airline 域中。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）是否使用 SFT 微调（Table 3）
| 设置 | Accuracy | T_total |
|------|----------|---------|
| No SFT（直接使用 Qwen3-1.7B） | 41.7% ❌ | 128k |
| **ARES (SFT)** | **54.8%** ✅ | 652k |

> ➤ 表明**任务特定微调至关重要**，否则 router 缺乏判断力。

#### （2）是否生成 rationale（显式推理过程）
| 设置 | Accuracy |
|------|----------|
| 直接分类（no rationale） | 51.3% |
| 包含 rationale 生成 | **54.8%** ✅ |

> ➤ 显示“先分析再决策”的中间推理过程显著提升准确率。

#### （3）RL 中是否归一化成本奖励（Table 4）
| 设置 | Accuracy | T_total |
|------|----------|---------|
| 未归一化 cost reward | 41.3% | 157k |
| **归一化 cost reward** | **42.0%** ✅ | **133k** ✅ |

> ➤ 归一化使成本信号更具可比性，帮助 router 更快收敛到最优策略。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **动态推理努力选择是高效 agent 的关键路径**  
   并非所有步骤都值得“深思熟虑”，ARES 成功识别出哪些步骤可以安全降级。

2. **ARES 实现“高性能 + 低成本”双赢**  
   - 在多数任务上达到甚至超过 high-effort 基线性能；
   - 推理 token 最多减少 **52.7%**（见摘要），平均约 **40%** 以上。

3. **轻量 router + KV cache 复用 极大提升实用性**  
   相比使用 GPT-5/Gemini 作为 router 的方案，ARES 的路由开销极小，且不破坏原有推理流。

4. **RL 训练能纠正“overthinking”陷阱**  
   如在 Airline 任务中，high effort 反而不如 medium，ARES 通过 RL 自动规避此问题。

5. **良好的泛化能力**  
   - 在不同任务间迁移有效；
   - 跨尺度泛化：在更大的 `gpt-oss-120b` 上仍表现优异（Table 5），准确率达 65.2%，接近其 high-effort 上限（67.8%），同时节省 ~23% token。

---

### ⚠️ 局限性
1. **依赖高质量轨迹数据**  
   当前方法需要先获得成功的 expert trajectories，若初始 agent 能力不足，则难以构建有效训练集。

2. **推理等级定义仍较粗粒度**  
   当前仅支持 low/medium/high 三级，未来可探索连续或更细粒度的 thinking budget 控制。

3. **router 本身引入额外延迟**  
   尽管轻量，但仍需一次额外 inference，对超低延迟场景可能构成瓶颈。

4. **目前主要面向文本输入环境**  
   尚未扩展至多模态（vision + text）agent 场景。

---

### 🔮 未来工作方向
- 扩展至 **multi-modal agents**（视觉+语言）
- 探索 **continuous reasoning budget control**
- 结合 **在线学习机制**，让 router 在部署中持续进化
- 应用于 **multi-agent 协作系统**中的资源调度

---

## ✅ 总结一句话
> **ARES 通过一个轻量级 router 实现了 per-step 的自适应推理努力选择，在保持甚至提升任务成功率的同时，将推理 token 消耗最多降低 52.7%，为高效、可持续的 LLM Agent 部署提供了实用解决方案。**

</details>

---

### 7. [EAGLE-Pangu: Accelerator-Safe Tree Speculative Decoding on Ascend NPUs](https://arxiv.org/abs/2603.08088)

**Authors**: Chang Han, Yijie Hu, Jingling Liu  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.08088v1  

#### Abstract
Autoregressive decoding remains a primary bottleneck in large language model (LLM) serving, motivating speculative decoding methods that reduce expensive teacher-model invocations by verifying multiple candidate tokens per step. Tree-structured speculation further increases parallelism, but is often...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EAGLE-Pangu: Accelerator-Safe Tree Speculative Decoding on Ascend NPUs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）在服务部署中的主要瓶颈是 **autoregressive decoding** 所带来的高延迟和低吞吐。尽管已有 **speculative decoding** 方法通过“draft model”生成候选 token 并由“teacher model”验证来提升效率，但现有的 **tree-structured speculative decoding** 在异构硬件后端（如 Ascend NPUs）上移植时面临严重挑战，具体表现为：
- 不同后端的 **KV-cache layout**、**attention masking 接口** 和 **tensor indexing 语义** 存在差异；
- 融合内核（fused kernels）对 mask 形状、边界条件要求更严格；
- 负索引或越界访问在某些设备上未定义，导致静默错误。

这些问题使得树形推测解码在实际部署中 **脆弱、不可重现且难以调试**。

### 提出的新方法与创新点
作者提出了 **EAGLE-PANGU**，一个面向 Ascend NPU 上 Pangu 教师模型的可复现、安全、高效的树形推测解码系统，其核心贡献包括：

#### （1）Branchable KV-Cache 抽象
- 设计了一个基于 HuggingFace Cache API 的显式分支/提交缓存管理器。
- 将已接受前缀状态（`committed cache`）与每个分支的推测状态（`branch caches`）分离。
- 支持两种 commit 模式：
  - **Length-based commit**：按长度截断更新。
  - **Path-index-based commit**：精确重排以匹配接受路径，并引入 **prefix-sharing fast reorder** 优化常见情况下的内存拷贝开销。

> ✅ 优势：保证缓存隔离性与语义等价性，避免状态污染，同时支持后端透明重建。

#### （2）Accelerator-Safe Tree Tensor Semantics
- 引入 **dummy-root indexing**：为根节点分配 index 0，消除负索引（如 `-1` 表示无父节点），确保所有 gather 操作索引合法。
- 构建 **ancestor table** 预计算祖先链，支持安全的多跳索引操作。
- 添加 **structural invariants check**（范围、非循环性、有效性闭包），用于运行前验证树结构正确性。

> ✅ 优势：彻底规避因索引语义不一致导致的硬件级错误，提升跨平台可移植性和稳定性。

#### （3）Fused-Kernel-Compatible Tree-Masked Teacher Execution
- 实现了支持融合注意力内核的 **4D tree attention mask**，仅允许节点关注其祖先（含自身），防止跨分支信息泄露。
- mask 构造基于 ancestor table，广播至 `[B, H, M_max, M_max]` 形式供 fused kernel 使用。
- 提供 **eager fallback 路径** 用于调试和验证。

> ✅ 优势：兼顾高性能（fused path）与可调试性（eager path），实现端到端加速的同时保障语义正确。

#### （4）支持性设计
- **Reproducible distributed pipeline**：分布式预处理与缓存机制，避免重复计算和同步失败。
- **Data-driven vocabulary subset mapping**：构建可复用的 draft vocabulary 子集，支持可控的速度-质量权衡。

---

## 2. 核心实验方法和设置

### 数据集
- **MT-Bench**：80 个提示，共 160 轮对话（每提示 2 轮）。
- **HumanEval-style prompts**：80 个代码生成类提示。
- 总计 **240 turns**，涵盖多样化输入输出长度分布（平均 prompt ~501 tokens，output ~891 tokens）。

### 实验设置
- **Hardware**：Ascend NPUs 上运行 Pangu 教师模型（via Pangu Embedded backend）。
- **Batch Size**：默认为 1（模拟真实请求场景）。
- **Decoding Config**：
  - 温度 = 0（greedy decoding）
  - `max_new_tokens=1024`（主实验），部分扫描设为 256 加速。
- **Execution Modes**：
  - **Performance Mode**：启用 fused attention（默认）。
  - **Reference Mode**：禁用 fused attention，用于调试与消融分析。

### 评估指标
| 指标 | 含义 |
|------|------|
| **Tok/s** | 每秒生成 token 数量（越高越好） |
| **Speedup (×)** | 相对于 baseline 的加速比 |
| **accept_L** | 每次验证步骤平均接受的 draft token 数量 |
| **accept_pos** | 不同 draft position 的接受率曲线 |
| **TTFT / TPOT** | Time to First Token / Time Per Output Token |

### 基线方法对比
- **Baseline**：Teacher-only greedy decoding（无 speculative decoding）
- **EA (EAGLE-PANGU)**：本文提出的 tree speculative decoding 方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1 & Figure 2）

| 指标 | Baseline | EAGLE-PANGU | Speedup |
|------|----------|-------------|---------|
| **Mean Tok/s** | 17.65 | 22.42 | **1.27×** |
| **p90 Tok/s** | 18.69 | 31.65 | **1.84×** |
| **p99 Tok/s** | 19.37 | 42.09 | **2.46×** |
| **accept_L (mean)** | — | 3.17 | — |

> 📌 **结论**：EAGLE-PANGU 在 batch size=1 下实现了 **平均 1.27× 吞吐提升**，尾部延迟改善显著（p99 达 2.46×），说明其在长尾请求中更具优势。

### 预算敏感性分析（Budget Sweep, Table 2 & Figure 4）
- 扫描不同 **node budget M** 与 **depth bound D_max** 对性能的影响。
- 最佳配置出现在 **M=16, D_max=10**，达到 **1.48× 平均加速**。
- 更大的 M 或 D_max 反而降低速度，原因：
  - mask/tensorization 开销增加；
  - 深层 draft position 接受率下降（见 Figure 3）。

> 📌 **结论**：树预算存在“甜点区”（sweet spot），盲目扩大预算反而适得其反。

### 消融实验与负向结果

#### （1）Fixed-Window Drafter Truncation 实验（Table 3 & Figure 6）
测试是否可通过限制 draft model 上下文窗口提升效率：

| Window W | EA Tok/s | Speedup | accept_L (mean) |
|---------|--------|--------|----------------|
| none    | 19.93  | 1.15   | 3.17           |
| 128     | 11.97  | 0.69   | 1.48           |
| 256     | 13.22  | 0.76   | 1.68           |
| 512     | 15.14  | 0.87   | 1.99           |

> ❌ **发现**：固定窗口截断显著降低 accept_L 和吞吐，甚至不如 baseline（W=128 时 speedup < 1）。  
> 🔍 **原因分析**（Figure 7）：instrumentation 显示 draft model 的 top-1 attention 经常落在远距离历史位置（如 >256），硬截断破坏了 draft quality。

> 📌 **结论**：naive context truncation 是有害的；需采用语义感知的上下文压缩策略（如 retrieval-augmented 或 importance-based truncation）。

#### （2）阶段耗时分解（Figure 5）
- **tree tensorization** 和 **mask construction**：毫秒级，非瓶颈。
- **verification** 与 **commit**：与推理时间相当，是优化重点。
- **prefill 阶段**：存在明显长尾，影响整体稳定性。

> 📌 **建议**：未来应优先优化 commit 效率与 prefill 行为。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Tree speculative decoding 可在 Ascend NPU 上高效部署**，前提是解决 KV-cache、masking 和 indexing 的硬件兼容性问题。
2. ✅ **EAGLE-PANGU 实现了平均 1.27×、最高 2.46× 的端到端吞吐加速**，尤其在尾部延迟表现优异。
3. ✅ **accept_L 是决定 speedup 的关键因素**，且随 draft depth 衰减。
4. ✅ **树预算需精细调优**，“越大越好”不成立，存在明显的 sweet spot。
5. ❌ **简单地截断 draft context 会严重损害性能**，因其依赖远距离上下文。

### 方法的局限性
1. **加速上限受限于 draft cost 与 verification overhead**：
   - 当树过大时，mask construction 和 cache commit 成为主要开销。
2. **性能高度依赖 draft model 质量**：
   - 若 draft model 准确率低，则 accept_L 下降，收益消失；
   - 若 draft model 过大，则 drafting 自身成为瓶颈。
3. **当前评估集中于单轮解码效率**：
   - 尚未覆盖多轮对话、工具调用、超长上下文等复杂场景。

### 未来工作方向
1. **更紧凑的 mask 表示形式**（如 sparse 或 structured mask）以减少 overhead。
2. **深度 kernel fusion**：将 tree tensorization、mask gen、attention 更紧密集成。
3. **speculation-aware distillation**：训练更强、更轻量的 draft model。
4. **adaptive branching policy**：动态调整树结构大小与深度。
5. **扩展至 multi-device serving 与 long-context workloads**，探索更广部署边界。

---

> 💡 **总体评价**：  
> EAGLE-PANGU 并未提出新的 decoding 算法，而是聚焦于 **系统级工程挑战**，提供了一套 **可复现、可调试、硬件安全** 的 tree speculative decoding 实现方案。它填补了先进 decoding 技术从研究到生产落地之间的鸿沟，尤其适用于 Ascend 类专用 AI 加速器平台。

</details>

---

### 8. [NN-OpInf: an operator inference approach using structure-preserving composable neural networks](https://arxiv.org/abs/2603.08488)

**Authors**: Eric Parish, Anthony Gruber, Patrick Blonigan, Irina Tezaur  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.08488v1  

#### Abstract
We propose neural network operator inference (NN-OpInf): a structure-preserving, composable, and minimally restrictive operator inference framework for the non-intrusive reduced-order modeling of dynamical systems. The approach learns latent dynamics from snapshot data, enforcing local operator stru...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NN-OpInf: an operator inference approach using structure-preserving composable neural networks

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统基于多项式的 **Operator Inference (P-OpInf)** 在建模**非多项式非线性动力系统**（如有限变形固体力学、反应流体等）时存在显著局限性。其强归纳偏置导致在复杂物理系统中精度低、泛化能力差，尤其在参数外推或长期预测场景下表现不佳。

此外，许多现有 **NN-based ROM** 方法缺乏对物理结构的嵌入，导致模型不稳定、难以解释，且训练过程对超参数敏感。

### 提出了什么新方法或新思路
本文提出 **NN-OpInf** ——一种**结构保持**（structure-preserving）、**可组合**（composable）的神经网络算子推断框架，用于非侵入式降阶建模（non-intrusive ROM）。其核心思想包括：

- **模块化算子分解**：将降阶系统的右端项 $ \dot{x} $ 表示为多个独立算子的加和：
  $$
  \dot{x}(t,u) = \sum_{r=1}^M g_r(\eta_r; w_r)
  $$
  每个算子 $ g_r $ 可以具有不同的输入依赖、复杂度和代数结构。

- **结构保持的神经网络算子设计**：引入具备特定代数结构的神经网络算子，例如：
  - **Skew-symmetric** 算子：保证能量守恒（无外力时）
  - **SPSD (Symmetric Positive Semi-Definite)** 算子：保证耗散特性
  - **SPSD Potential** 算子：通过自动微分构建梯度场，保持拉格朗日或哈密顿结构

- **开放源码实现**：开发并开源了 `NN-OpInf` 软件包，提供模块化的 API 支持变量、算子和模型的灵活组合。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **表达能力** | 超越多项式形式，能捕捉任意非线性，适用于非多项式系统 |
| **稳定性与物理一致性** | 内嵌物理结构（如能量守恒/耗散），提升长期预测鲁棒性 |
| **可解释性** | 模块化设计允许对不同物理机制（如对流、扩散）分别建模 |
| **灵活性** | 支持异构算子组合，适应复杂系统（如同时含保守与耗散项） |

---

## 2. 核心实验方法和设置

### 使用的数据集（数值模拟问题）
论文在五个典型非线性、参数化动力系统上进行验证：
1. **Burgers’ Equation**（一维周期域，能量守恒）
2. **Nonlinear Convection-Diffusion-Reaction (CDR) System**（二维，含非多项式非线性）
3. **2D Nonlinear Heat Conduction**（温度依赖导热系数）
4. **Premixed H₂-Air Flame**（燃烧模型，参数非仿射依赖）
5. **3D Hyper-Elastic Torsion Problem**（大变形固体力学，高度非线性）

### 实验设置和评估指标
- **降维方法**：均采用 **POD** 构造降阶基 $ V \in \mathbb{R}^{N \times K} $
- **训练数据**：从高保真全阶模型（FOM）采样状态 $ x $ 和时间导数 $ \dot{x} $
- **评估指标**：
  - **相对 $ L^2 $ 状态误差**：
    $$
    e = \sqrt{\frac{\sum_t \| x_{\text{ROM}}(t;\mu) - x_{\text{FOM}}(t;\mu) \|^2}{\sum_t \| x_{\text{FOM}}(t;\mu) \|^2}}
    $$
  - **能量守恒性分析**（针对守恒系统）
  - **收敛性分析**：随降阶维度 $ K $ 增加的误差变化
  - **预测能力**：区分“再生”（reproductive）与“未来态预测”（future-state prediction）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **POD-Galerkin** | 侵入式投影法（理想基准） |
| **P-OpInf-A / P-OpInf-AH** | 线性 / 二次多项式 OpInf |
| **P-OpInf-cA / P-OpInf-cAH** | 含仿射强迫项的 P-OpInf |
| **NN-OpInf-NN** | 文献 [20] 中的标准前馈神经网络（无结构） |
| **NN-OpInf-SS / PSD-f / SPSD / SPSD-Potential** | 本文提出的结构化 NN-OpInf 变体 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ **Burgers’ Equation**
- **再生任务**：NN-OpInf-SS（Skew-symmetric）误差 ~0.1%，与 P-OpInf-AH 相当
- **未来态预测**：NN-OpInf-SS 显著优于 P-OpInf-A/AH，后者出现能量漂移
- **能量守恒**：NN-OpInf-SS 几乎完美保持能量，而 P-OpInf 在外推时严重违反

#### ✅ **Nonlinear CDR System**
- P-OpInf-cA/cAH 误差 >5%，无法准确刻画解
- **NN-OpInf-PSD-f** 达到与 Galerkin ROM 相当精度，**比 P-OpInf 高 5–10 倍**
- 标准 NN-OpInf-NN 不稳定
- **参数预测**：NN-OpInf-PSD-f 在测试集上仍优于 Galerkin ROM

#### ✅ **2D Nonlinear Heat Conduction**
- P-OpInf-cAH 和 NN-OpInf-NN 均产生非物理解
- **NN-OpInf-SPSD-f** 在所有配置下均最优，误差比 P-OpInf-cA 低 5–10 倍
- 展现良好收敛趋势（尽管非单调）

#### ✅ **Premixed H₂-Air Flame**
- 在训练集上，P-OpInf 表现略好于 NN-OpInf-PSD-f
- **在测试集上，NN-OpInf-PSD-f 误差更低**，表明更强的泛化能力
- 最高维度下，NN-OpInf-PSD-f 比 P-OpInf 低约 **3 倍**

#### ✅ **3D Hyper-Elastic Torsion**
- **NN-OpInf-SPSD-Potential**（保持拉格朗日结构）比 P-OpInf-A/AH **高一个数量级精度**
- 物理位移场高度吻合 FOM
- P-OpInf-AH 在未来态预测中发散，而 NN-OpInf 仍合理

### 消融实验与关键发现
- **结构嵌入至关重要**：相比无结构的 NN-OpInf-NN，结构化版本（如 Skew、SPSD）显著更稳定、更准确
- **Ensemble 提升鲁棒性**：即使小集成（$ L=2 $）也能有效降低方差，提升性能
- **优化策略改进收敛**：结合 L-BFGS 与 ADAM 的混合优化比纯 ADAM 更可靠
- **数据归一化影响**：max-abs 归一化优于标准归一化，因它保持 POD 系数的范数等价性

---

## 4. 关键结论和发现

### 主要发现
1. **NN-OpInf 是 P-OpInf 的有效替代方案**：当系统动力学**不具多项式结构**时，NN-OpInf 能显著提升精度与鲁棒性。
2. **结构保持是关键**：内嵌物理结构（如 skew-symmetry、SPSD）不仅提升稳定性，还增强泛化能力，尤其在**外推与长期预测**中。
3. **模块化设计带来灵活性**：通过组合不同类型算子，可为复杂多物理场系统构建可解释、可定制的 ROM。
4. **性能代价明确**：NN-OpInf 的**在线推理成本**与二次 P-OpInf 相当，但**离线训练成本显著更高**，且优化问题为非凸。

### 方法的局限性
- **训练成本高**：需大量迭代优化，远高于 P-OpInf 的凸最小二乘求解
- **非凸优化挑战**：可能存在局部极小，训练对初始化和超参数有一定敏感性
- **结构先验依赖**：需要用户对系统物理结构有基本认知以选择合适算子
- **当前未支持超减积**（hyper-reduction），仅适用于可直接获取 $ \dot{x} $ 的场景

### 未来工作方向
- 开发更高效的优化算法（如随机 SR1）
- 扩展算子库（如 port-Hamiltonian、metriplectic 结构）
- 引入**动态约束训练**（backpropagation through time）
- 支持不确定性量化（uncertainty quantification）
- 推广至更多**多物理场耦合系统**

---

> **总结**：  
> **NN-OpInf** 通过将**神经网络的表达能力**与**物理结构的先验知识**相结合，实现了在非多项式非线性系统上的高精度、鲁棒、可解释的非侵入式降阶建模。它是对传统 P-OpInf 的有力扩展，特别适用于传统方法失效的复杂工程与科学计算问题。

</details>

---

### 9. [Switchable Activation Networks](https://arxiv.org/abs/2603.06601)

**Authors**: Laha Ale, Ning Zhang, Scott A. King, Pingzhi Fan  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.06601v1  

#### Abstract
Deep neural networks, and more recently large-scale generative models such as large language models (LLMs) and large vision-action models (LVAs), achieve remarkable performance across diverse domains, yet their prohibitive computational cost hinders deployment in resource-constrained environments. E...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Switchable Activation Networks》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前深度神经网络（DNNs）和大规模生成模型（如 LLMs、LVAs）虽然在多个领域表现出色，但其高昂的计算成本严重限制了在资源受限环境（如边缘设备）中的部署。传统的效率优化技术存在以下不足：
- **Dropout**：仅用于训练阶段的正则化，推理时仍需完整计算。
- **Pruning 和 Low-rank Factorization**：后处理压缩方法，产生静态稀疏模型，缺乏对输入上下文的适应性。
- **动态推理策略（如 MoE、SkipNet）**：引入运行时开销和不规则内存访问。

因此，如何实现**既高效又具备上下文自适应能力**的神经网络成为关键挑战。

### 提出了什么新方法或新思路
本文提出 **SWAN（Switchable Activation Networks）**，一种全新的框架，将每个神经元或通道配备一个**确定性的、输入依赖的二值门控开关（binary gate）**，从而让网络学习“何时激活”某个单元。

核心思想是：  
> 将效率视为神经计算的内在属性，而非事后优化目标，通过**学习激活控制**来动态分配计算资源。

具体机制包括：
- 每个 unit 配备可学习的 gate $g_i(x) \in \{0,1\}$，由输入决定是否参与前向传播。
- 使用 **Straight-Through Estimator (STE)** 实现端到端训练，解决非可微阈值操作的问题。
- 引入软门控（soft gates）用于训练稳定性，硬门控（hard gates）用于推理效率。
- 支持两种部署模式：**动态稀疏推理** 或转换为**紧凑密集模型**。

### 相比现有方法的优势
| 方法 | 是否动态 | 推理效率 | 是否可恢复容量 | 是否集成于训练 |
|------|--------|----------|----------------|----------------|
| Dropout | ❌（随机） | ❌ | ✅ | ✅ |
| Pruning | ❌（静态） | ✅ | ❌ | ❌（后处理） |
| MoE / Dynamic Inference | ✅ | ⚠️（变延迟） | ✅ | ✅ |
| **SWAN** | ✅（确定性） | ✅（支持静态导出） | ✅ | ✅ |

**优势总结**：
- **统一范式**：融合了 sparsity、pruning 和 adaptive inference 的优点。
- **灵活性高**：同一模型可根据输入难度动态调整计算量。
- **部署友好**：可通过 thresholding + pruning 导出为小型 dense model，避免动态调度开销。
- **生物启发**：模拟大脑稀疏、选择性和上下文依赖的神经活动模式。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MNIST**：手写数字分类任务，验证基本有效性。
- **ImageNet 子集 / CIFAR 类似设置**：在 VGG16 和 ResNet50 上进行图像分类实验（文中未明确提及具体数据集名称，但从模型规模推断应为标准视觉基准）。

### 实验设置和评估指标
#### 模型架构
- 在 **VGG16** 和 **ResNet50** 上实现 SWAN。
- Gate 设计在 channel level（通道级），适用于 CNN 架构。

#### 训练策略
- 使用 **Adam 优化器**，初始学习率 0.001。
- 引入 **delayed cosine ramp schedule** 控制正则项权重 $\lambda_o, \lambda_F, \lambda_T$，防止早期过度剪枝。
- Gate logits 使用更高学习率且无 weight decay，确保快速响应。

#### 正则化项
1. **L0-style Sparsity Proxy**: $\mathcal{R}_o(\phi) = \sum_i \mathbb{E}[p_i(x)]$
2. **FLOPs-aware Penalty**: $\mathcal{R}_F(\phi;x) = \sum_i p_i(x) c_i(x)$，其中 $c_i$ 是单位计算代价。
3. **One-sided Target Activity**: $\mathcal{R}_T(\phi) = (\max\{0, \alpha(\phi) - \alpha^*\})^2$，设定最大激活比例上限。

#### 评估指标
- **Top-1 Accuracy**
- **Active Unit Fraction (%)**：激活神经元/通道占比
- **Model Size / FLOPs (%)**：相对于原始模型的比例
- **Soft vs Hard Evaluation**：分别衡量训练稳定性和实际推理效率

### 基线方法对比
- **Dropout**：标准随机失活，推理全激活。
- **Channel Pruning (CP)**：基于幅值的通道剪枝，分 raw（直接剪枝）和 fine-tuned 版本。
- **SWAN_raw**：未经微调的门控模型。
- **SWAN**：经过 5 轮微调后的最终模型。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | FLOPs (%) | Top-1 Accuracy (%) | 备注 |
|------|-----------|--------------------|------|
| Baseline (Dense) | 100% | ~90–95% | 原始性能 |
| Dropout | 100% | 下降明显（<30% @ 10% FLOPs） | 无推理加速 |
| CP_raw | 5% | 16.1% (VGG), 10.0% (ResNet50) | 性能崩溃 |
| CP (fine-tuned) | 5% | 仍远低于 baseline | 恢复有限 |
| **SWAN_raw** | 5% | >90% | 几乎无损 |
| **SWAN** | 5% | **>90%** | 微调后保持高性能 |

> 注：图5显示，在极端压缩下（约5% FLOPs），SWAN 仍能维持超过90%准确率，而其他方法严重退化。

#### MNIST 实验亮点
- 激活单元比例从 100% 下降到 **仅 3%**。
- 验证精度始终保持接近 **100%**。
- 表明传统 dense 模型中存在大量冗余参数。

#### 动态行为观察
- 简单样本激活更少 unit，复杂样本自动启用更多路径。
- 实现真正的“按需计算”。

### 与基线方法的对比结果
- **相比 Dropout**：SWAN 在相同 FLOPs 下准确率显著更高，且真正实现推理加速。
- **相比 Channel Pruning**：SWAN 不需要反复迭代剪枝-微调，具备更强鲁棒性；即使不微调（SWAN_raw）也优于剪枝后微调的结果。
- **相比 MoE/Dynamic Inference**：SWAN 门控机制更轻量，支持导出为静态小模型，更适合边缘部署。

### 消融实验结果（Ablation Study）
虽然文中未设独立章节，但通过以下分析体现消融思想：
- **Soft vs Hard Gates**：
  - Soft gates 保证训练稳定（BN 统计一致），accuracy 曲线平滑。
  - Hard gates 才反映真实效率，必须用 hard evaluation 判断性能损失。
- **BN Recalibration 的必要性**：
  - 若不重新校准 BN 统计，在 ResNet、Inception 等多分支结构中会导致显著性能下降。
  - 对 VGG 类简单结构影响较小。
- **Regularization Schedule 的作用**：
  - 提前施加 sparsity penalty 会破坏表示学习。
  - Delayed cosine ramp 显著提升收敛稳定性与最终性能。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **神经网络中存在巨大冗余**：通过 SWAN 可将激活单元压缩至原模型的 **3% 甚至更低**，而不损失精度。
2. **效率可以作为第一性设计原则**：将 activation 控制显式建模并联合训练，比 post-hoc 压缩更有效。
3. **动态 ≠ 不可控**：SWAN 实现了 deterministic 的 input-dependent 激活，兼具灵活性与可预测性。
4. **生物启发的设计具有潜力**：大脑的稀疏编码、上下文依赖激活等机制可在人工系统中复现并带来收益。
5. **SWAN 支持双重部署模式**：
   - 动态稀疏推理（适合异构硬件）
   - 导出为 compact dense model（适合通用芯片）

### 方法的局限性
1. **硬件支持瓶颈**：
   - 当前 GPU/TPU 优化面向 dense tensor ops，稀疏执行未必提速，除非有专用 runtime 或 accelerator。
2. **延迟不确定性**：
   - 动态计算路径导致推理延迟波动，不适合严格实时系统。
3. **额外参数开销**：
   - 每个 unit 需维护 gate logit，增加少量存储负担（但远小于保留全部 activation 的代价）。
4. **梯度近似误差**：
   - STE 是代理梯度法，理论上可能影响最优性，尽管实践中表现稳定。

### 未来工作方向
1. **硬件协同设计**：开发支持高效稀疏门控执行的 neuromorphic 或 edge chips。
2. **扩展至 Transformer 架构**：应用于 LLMs 中的 attention head 或 FFN unit 级别开关。
3. **跨模态应用**：在 Vision-Language-Action (VLA) 模型中实现多模态条件下的动态激活。
4. **理论分析**：建立关于“learned activation control”的泛化误差界与最优性条件。
5. **自动化目标设定**：让模型自主学习最佳 $\alpha^*$，而非人工指定。

---

> 🔚 **一句话总结**：  
> SWAN 提出了一种将“是否计算”作为可学习决策的新范式，实现了**高精度下的极致稀疏化**，不仅推动了高效 AI 的发展，也为构建类脑的可持续智能系统提供了新思路。

</details>

---

### 10. [Reinforcement learning-based dynamic cleaning scheduling framework for solar energy system](https://arxiv.org/abs/2603.07518)

**Authors**: Heungjo An  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.07518v1  

#### Abstract
Advancing autonomous green technologies in solar photovoltaic (PV) systems is key to improving sustainability and efficiency in renewable energy production. This study presents a reinforcement learning (RL)-based framework to autonomously optimize the cleaning schedules of PV panels in arid regions,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Reinforcement learning-based dynamic cleaning scheduling framework for solar energy system

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题  
该研究针对**干旱地区太阳能光伏（PV）系统因灰尘沉积（soiling）导致发电效率下降**的问题，提出了一种动态清洁调度优化框架。传统固定周期清洁策略在面对天气不确定性（如风速、湿度、颗粒物浓度等）时表现不佳，无法实现成本最优。

### 🚀 提出的新方法与创新思路  
- **首次将强化学习（Reinforcement Learning, RL）应用于PV面板的动态清洁调度决策中**，构建了一个基于RL的端到端优化框架。
- 引入并改进了**soiling建模机制**，在原有模型基础上加入了**相对湿度（Relative Humidity, RH）的影响因子**，提出了校准后的日积尘量公式 $ D_{\text{soiling}}' $，更真实地反映高湿环境下灰尘粘附增强的现象。
- 将清洁调度问题形式化为一个**马尔可夫决策过程（MDP）**，利用RL算法进行序贯决策优化，实现了对环境变化的自适应响应。

### 🔍 相比现有方法的优势  
| 维度 | 本文方法 | 传统方法 |
|------|----------|---------|
| 清洁策略 | 动态、自适应调整 | 固定周期（如每28天） |
| 不确定性处理 | 显式建模天气随机性（通过分布拟合） | 多数采用确定性假设或简单Monte Carlo模拟 |
| 决策灵活性 | 基于状态实时判断是否清洁 | 预设时间表，缺乏反馈机制 |
| 成本效益 | 最大可达13%的成本节约 | 成本较高且次优 |

> ✅ **核心优势**：相比传统的Simulation Optimization（Sim-Opt）和固定间隔策略，RL方法能根据实时天气条件动态调整清洁时机，在降低运维成本的同时最大化能量产出。

---

## 2. 核心实验方法和设置

### 📊 数据集  
- **研究区域**：阿联酋阿布扎比（Abu Dhabi），典型的干旱气候区。
- **数据来源与时间范围**：
  - 2018–2020年每日气象数据，来自文献 [16] 和 [visualcrossing.com](https://www.visualcrossing.com/)
- **包含的关键变量**：
  - Temperature（温度）
  - Wind Speed（风速）
  - Particulate Matter（PM，单位g/m²）
  - Irradiance（辐照度）
  - Relative Humidity（相对湿度，新增引入）

> 所有变量按月进行概率分布拟合（使用Stat::Fit软件），用于仿真环境中生成随机天气场景。

### ⚙️ 实验设置  
- **目标**：最小化20年生命周期内的**总成本 = 清洁成本 + 发电损失成本**
- **控制变量**：
  - 单位清洁成本（5个等级）
  - 电价（两类用户：外籍居民 vs. 阿联酋国民）
- **测试案例**：共10组（见Table 3），覆盖不同经济参数组合

### 🎯 评估指标  
| 指标 | 描述 |
|------|------|
| `Optimal cleaning interval` | Sim-Opt得出的最佳固定周期（天） |
| `Average total cost` | 平均总成本（USD） |
| `Number of cleanings` | 平均清洁次数 |
| `Cost saving (%)` | PPO相较于Sim-Opt的成本节省百分比 |
| `Reward / Total cost during training & testing` | 训练与测试阶段的表现稳定性 |

### 🔁 基线方法对比  
| 方法 | 类型 | 说明 |
|------|-----|------|
| **Sim-Opt** | Simulation + Optimization | 固定周期搜索最优值，作为基准 |
| **Fixed 28-day rule** | 规则策略 | 阿布扎比政府推荐的固定清洁周期 |
| **PPO (Proximal Policy Optimization)** | On-policy RL | 本文主推算法，具备稳定训练特性 |
| **SAC (Soft Actor-Critic)** | Off-policy RL | 对比使用的先进RL算法，强调探索性 |

> 所有模型均在同一仿真环境中训练与评估，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自Table 4）

| Case | Sim-Opt 最佳周期（天） | Sim-Opt 总成本（USD） | PPO 测试总成本（USD） | 成本节约（%） |
|------|------------------------|------------------------|------------------------|--------------|
| S1exp | 19 | 14.5 | 12.6 | **13%** |
| S2exp | 29 | 20.4 | 18.4 | 10% |
| S3exp | 36 | 24.8 | 22.9 | 8% |
| S4exp | 41 | 28.5 | 26.9 | 6% |
| S5exp | 51 | 31.7 | 30.4 | 4% |
| S1uae | 39 | 6.8 | 6.3 | 8% |
| ... | ... | ... | ... | ... |
| S5uae | 95 | 15.2 | 14.8 | 2% |

> 💡 **最高达13%的成本节约**，尤其在电价较高、清洁频率较高的情况下效果更显著。

### 🔁 与基线方法对比结果  
- **PPO > Sim-Opt > SAC**
  - PPO在所有测试案例中均优于Sim-Opt（除S4uae略差），平均节省约5–13%
  - SAC虽有较强探索能力，但在本任务中表现不稳定，奖励波动大，未能超越Sim-Opt
- **训练稳定性对比**：
  - **PPO**：奖励稳步上升并收敛，训练过程平稳（图5a）
  - **SAC**：奖励剧烈震荡，存在“崩溃”现象，难以稳定学习（图5b）

> ❗ SAC失败原因分析：
> 1. 环境高度随机，回放缓冲区（replay buffer）中的旧经验不匹配当前策略
> 2. 奖励极其稀疏（仅每20年一个episode结束才有反馈）
> 3. Q函数估计受噪声影响大，更新缓慢
> 4. 超参数调优仍可能存在不足

### 🔍 消融实验与决策行为分析（见Figure 6）  
- **关键状态变量可视化显示**：
  - PPO主要依据两个变量做出清洁决策：
    - `deposition`（积尘量）
    - `days_since_last_cleaning`（距上次清洁天数）
  - 其他变量（温度、风速、PM、辐照度）未表现出明显模式，表明其间接作用为主
- 表明模型已学会识别“临界阈值”，当积尘累积到一定程度即触发清洁动作，体现了**自主决策能力**

---

## 4. 关键结论和发现

### ✅ 主要发现  
1. **动态RL调度显著优于固定周期策略**  
   - 在不确定天气条件下，PPO能够灵活响应环境变化，避免过度清洁或延迟清洁。
2. **PPO是更适合该任务的RL算法**  
   - 其on-policy特性和clip机制带来更强的训练稳定性，适合长期、低频奖励的任务。
3. **引入相对湿度显著提升soiling建模精度**  
   - 特别是在高湿沙漠环境中，灰尘不易被风吹走，必须考虑RH对清洁效率的影响。
4. **经济收益可观**：在典型场景下可实现**高达13%的运营成本节约**，具有实际应用价值。

### ⚠️ 方法的局限性  
1. **泛化能力有限**：
   - 模型在训练环境中表现优异，但在测试环境中部分案例出现性能下降，提示存在**过拟合风险**。
2. **探索不足**：
   - 训练过程中可能未充分覆盖极端天气事件（如罕见沙暴），导致面对异常情况时决策偏差。
3. **状态空间简化**：
   - 当前状态表示未包含面板老化、局部污染差异等因素，未来可进一步丰富。
4. **奖励稀疏性挑战**：
   - 每个episode长达20年，导致梯度信号极弱，限制了某些off-policy算法（如SAC）的有效学习。

### 🔮 未来工作方向  
1. **增强模型泛化能力**：
   - 引入**domain randomization**技术，在训练中加入更多样化的天气扰动。
2. **改进探索策略**：
   - 设计课程学习（curriculum learning）或分层RL结构，逐步引导智能体学习复杂决策。
3. **融合更多物理机制**：
   - 加入面板倾角、材料特性、降雨（尽管在干旱区较少）等变量以提高普适性。
4. **跨区域迁移研究**：
   - 将框架推广至其他气候类型地区（如半干旱、沿海），验证其通用性。
5. **可解释AI（Explainable AI）集成**：
   - 利用注意力机制或SHAP值分析各输入特征的重要性，提升决策透明度，便于工业部署。

---

## ✅ 总结一句话  
> 本研究成功将**PPO-based RL框架**应用于太阳能系统的动态清洁调度，在阿布扎比案例中实现了**最高13%的成本节约**，证明了**自主绿色能源管理系统**在应对环境不确定性方面的巨大潜力。

</details>

---

### 11. [AutoAdapt: An Automated Domain Adaptation Framework for LLMs](https://arxiv.org/abs/2603.08181)

**Authors**: Sidharth Sinha, Anson Bastos, Xuchao Zhang, Akshay Nambi, Chetan Bansal, Saravan Rajmohan  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.08181v1  

#### Abstract
Large language models (LLMs) excel in open domains but struggle in specialized settings with limited data and evolving knowledge. Existing domain adaptation practices rely heavily on manual trial-and-error processes, incur significant hyperparameter complexity, and are highly sensitive to data and u...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AutoAdapt: An Automated Domain Adaptation Framework for LLMs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

大型语言模型（LLMs）在通用领域表现出色，但在**特定领域**（如医疗、法律、数学等）的应用中面临显著挑战，尤其是在以下情况下：
- **数据稀缺**：目标领域标注数据有限。
- **知识动态更新**：预训练模型无法覆盖随时间演化的领域知识。
- **高昂的适配成本**：手动进行领域适配（如 SFT、RAG、DPO）需要大量专家干预和试错，且超参数选择对性能影响巨大。

现有的自动化机器学习（AutoML）框架难以有效解决 LLM 领域适配问题，原因包括：
- 依赖昂贵的黑盒搜索（如贝叶斯优化），不适用于高成本的 LLM 训练。
- 缺乏对 LLM 特定技术（如 LoRA、RAG、DPO）的结构化知识支持。
- 生成的管道常因代码错误而无法执行。
- 忽视用户约束（如硬件、延迟）和数据特性。

### 提出了什么新方法或新思路

本文提出了 **AutoAdapt**，一个端到端自动化的 LLM 领域适配框架，其核心创新点如下：

#### （1）基于结构化知识的自适应配置图（Adaptation Configuration Graph, ACG）
- 将复杂的 LLM 适配流程建模为一个有向无环图（DAG）。
- 图中节点分为两类：**选择节点**（如 RAG vs SFT vs DPO）和**参数节点**（如 `learning_rate`, `lora_rank`）。
- 该图编码了最佳实践和依赖关系（例如，选择 LoRA 会激活 `lora_rank` 参数），从而**系统性地缩小搜索空间**，避免无效组合。

#### （2）多智能体辩论规划器（Multi-Agent Debate Planner）
- 引入由 **Proposal Agents** 和 **Critic Agents** 组成的多智能体系统，在 ACG 的每个节点上进行迭代辩论。
  - **Proposal Agents**：结合离线构建的 **Best Practices Knowledge Base**（从 Hugging Face、arXiv、GitHub 等提取）和在线检索，提出候选方案。
  - **Critic Agents**：分别从**用户偏好**（如预算、模型大小限制）和**数据特性**（如数据量、token 分布）角度对提案进行批判和验证。
- 通过“提议→批判→修订”的循环，最终生成一个**可执行且符合约束**的适配管道。

#### （3）高效优化模块 AutoRefine
- 为解决标准 HPO 在 LLM 上成本过高的问题，提出 **AutoRefine**。
- 它是一种**基于 LLM 的代理模型**（LLM-based surrogate），结合**高斯过程**（Gaussian Process, GP）进行概率函数估计。
- 通过少量实际运行（few-shot execution）获取性能反馈，利用 LLM 预测各超参数维度的趋势，并用 GP 构建性能曲面，指导后续采样。
- 这种混合方法**减少了 LLM 幻觉**，实现了在极低预算下的高效优化。

### 相比现有方法的优势

| 优势 | 说明 |
|------|------|
| **可靠性高** | 多智能体辩论确保生成的管道可执行，成功率远高于基线。 |
| **效率高** | ACG 和 AutoRefine 显著缩小搜索空间，减少不必要的训练尝试。 |
| **性能优** | 在多个任务上平均相对准确率提升 **25%**。 |
| **用户对齐** | 显式融入用户约束和数据信号，生成个性化方案。 |
| **知识驱动** | 不仅依赖 LLM 的预训练知识，更结合了大规模开源生态的最佳实践。 |

---

## 2. 核心实验方法和设置

### 使用的数据集

实验涵盖了 **10 个多样化任务**，跨越多个领域：

| 数据集 | 领域 | 任务类型 |
|--------|------|----------|
| **MATH** | 数学 | 数学推理 |
| **MedQA** | 医疗 | 医学问答 |
| **CaseHold** | 法律 | 判例结果预测 |
| **ARC** | 科学 | 多项选择题 |
| **MBPP+** | 编程 | 代码生成 |
| **When2Call** | 工具调用 | 工具调用决策 |
| **Ecom** | 电商 | 商品分类 |
| **Entail** | 自然语言 | 文本蕴含 |
| **PEM** (proprietary) | 数学 | 工程前数学题 |
| **RCA** (proprietary) | 云计算 | 故障根因分析 |

### 实验设置和评估指标

#### 两种评估模式
- **模板自由（Template-Free, TF）**：不限制模型、技术或超参数，完全由系统自主决定。
- **模板感知（Template-Aware, TA）**：固定模型和基础技术（如 SFT），只比较超参数选择能力，以公平对比。

#### 评估指标
- **准确性（Accuracy）**：大多数任务的直接评价指标。
- **成功率（Success Rate, SR）**：定义为 `1 / (1 + 手动修复次数)`，衡量生成管道的**可执行性**。
- **归一化性能得分（Normalized Performance Score, NPS）**：将原始准确率归一化到 [0,1]。
- **综合得分（Cumulative Score, CS）**：`CS = (SR + NPS) / 2`，综合衡量可靠性和性能。
- **计算开销**：GPU 小时数和估计成本（美元）。

### 基线方法对比

与当前最先进的 **Agentic AutoML** 框架对比：
- **MLCopilot** (Zhang et al., 2024)
- **DS-Agent** (Guo et al., 2024)
- **AutoMLAgent** (Trirat et al., 2025)

所有方法均使用相同的底层 LLM（GPT-4.1-mini）和资源预算，确保公平性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

- **平均相对准确率提升**：AutoAdapt 在 10 个任务上相比 SOTA AutoML 基线，取得了 **25%** 的平均相对准确率提升。
- **成功率达到 100%**：在 TF 设置下，AutoAdapt 的 SR 为 1.0，而其他基线普遍低于 0.2，表明其生成的管道高度可靠。
- **综合表现领先**：在 CS 指标上，AutoAdapt 在几乎所有数据集上均大幅领先。

### 与基线方法的对比结果

| 方法 | MATH (TF) | MedQA (TF) | CaseHold (TF) | ARC (TF) | 平均 CS |
|------|-----------|-----------|-------------|----------|---------|
| **AutoAdapt (Ours)** | 0.58 | **0.82** | **0.96** | **0.95** | **0.84** |
| AutoMLAgent | 0.10 | 0.29 | 0.49 | 0.44 | 0.34 |
| MLCopilot | 0.09 | 0.26 | 0.12 | 0.23 | 0.22 |
| DS-Agent | 0.50 | 0.87 | 0.14 | 0.24 | 0.43 |

> 注：AutoAdapt 在 MedQA、CaseHold、ARC 等关键任务上遥遥领先。

### 消融实验结果

消融研究（Ablation Study）验证了各组件的重要性：
- **逐步添加组件**（Base LLM → ACG → Best Practices → Multi-Agent → AutoRefine）后，性能持续提升。
- **AutoRefine 的有效性**：相比标准 HPO（如 Optuna），AutoRefine 能在更小的 **coreset**（数据子集）上达到更高或相当的准确率，且速度更快。
- **多智能体辩论轮次**：增加辩论轮次能稳步提升配置质量，表明批判机制有效减少了规划偏差。
- **AutoRefine 试验次数**：仅需 5 次左右的实际评估即可收敛，证明其高效性。

---

## 4. 关键结论和发现

### 主要发现

1. **结构化先验知识至关重要**：ACG 和 Best Practices KB 为自动化提供了可靠的起点，避免了盲目搜索。
2. **多智能体辩论显著提升可靠性**：通过 Proposal 和 Critic 的交互，能有效融合用户意图、数据信号和最佳实践，生成高质量、可执行的管道。
3. **LLM 作为代理模型需谨慎设计**：直接用 LLM 预测性能易产生幻觉。AutoRefine 通过结合 GP 进行函数估计，显著提升了预测的稳定性和有效性。
4. **自动化能带来显著性能增益**：AutoAdapt 不仅节省人力，还能超越人工经验，实现更高的准确率。

### 方法的局限性

- **依赖外部知识库的质量**：如果 KB 中缺乏相关领域的最佳实践，可能影响推荐效果。
- **ACG 需要手动构建**：虽然灵活，但构建完整的 ACG 需要领域专家参与。
- **对极端新颖的技术适应慢**：对于近期出现且未被广泛采用的新技术，系统可能无法及时捕捉。

### 未来工作方向

- **动态扩展 ACG**：让系统能够从成功案例中自动学习并扩展 ACG。
- **增强在线学习能力**：在优化过程中更主动地探索未知的高性能区域。
- **支持更多模态**：将框架扩展到多模态 LLM 的适配。
- **降低对强 LLM 的依赖**：探索使用更小、更高效的模型来替代当前的 LLM 代理。

---

**总结**：AutoAdapt 是首个针对 LLM 领域适配的端到端自动化框架。它通过**结构化知识图谱**、**多智能体辩论**和**高效代理优化**三大创新，解决了现有方法在可靠性、效率和性能上的不足，为非专家用户快速构建高质量领域专用 LLM 提供了强大工具。

</details>

---

### 12. [Airborne Magnetic Anomaly Navigation with Neural-Network-Augmented Online Calibration](https://arxiv.org/abs/2603.08265)

**Authors**: Antonia Hager, Sven Nebendahl, Alexej Klushyn, Jasper Krauser, Torleiv H. Bryne, Tor Arne Johansen  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.08265v1  

#### Abstract
Airborne Magnetic Anomaly Navigation (MagNav) provides a jamming-resistant and robust alternative to satellite navigation but requires the real-time compensation of the aircraft platform's large and dynamic magnetic interference. State-of-the-art solutions often rely on extensive offline calibration...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Airborne Magnetic Anomaly Navigation with Neural-Network-Augmented Online Calibration

---

## 1. 论文的主要贡献和创新点

### 解决的问题
**Airborne Magnetic Anomaly Navigation (MagNav)** 是一种利用地壳磁场异常进行定位的抗干扰导航技术，可作为 GNSS 的可靠替代方案。然而，其核心挑战在于飞机平台自身产生的强磁干扰（可达数千纳特斯拉），远超用于导航的地磁异常信号（通常仅数十至数百纳特斯拉）。传统方法依赖于**离线校准飞行**（offline calibration flights）来训练补偿模型（如 Tolles-Lawson 模型），这在实际部署中存在显著的**操作瓶颈**。

此外，现有基于机器学习（ML）的方法虽然能捕捉非线性干扰，但往往需要大量历史数据进行预训练，计算复杂度高，难以实现实时在线适应，且缺乏可解释性和系统稳定性保障。

### 提出的新方法与新思路
本文提出了一种**完全自适应的混合在线校准与磁异常导航架构**，其核心创新如下：

- ✅ **“冷启动”能力（Cold-Start Capability）**  
  系统无需任何先验知识或校准飞行即可启动，能够在飞行过程中自主识别并补偿飞机的磁特征，实现真正的即插即用。

- ✅ **模块化混合架构（Modular Hybrid Architecture）**  
  将物理驱动的 **Tolles-Lawson (TL) 模型** 与 **神经网络 (NN)** 以加法形式集成于扩展卡尔曼滤波器（EKF）中：
  - **TL 模型** 负责建模主导的、与机动相关的线性/弱非线性干扰；
  - **NN 模型** 仅作为“残差学习器”（residual learner），专门捕捉 TL 模型无法纠正的高阶非线性动态干扰（如电气系统瞬变、涡流效应等）。

- ✅ **EKF-NN 联合状态估计框架**  
  将 NN 的权重和偏置参数直接作为 EKF 的状态向量进行联合估计。该更新机制在数学上等价于**在线自然梯度下降**（online Natural Gradient descent），具备二阶优化的快速收敛特性，同时保持了滤波器的实时性与确定性。

- ✅ **数据与计算高效性**  
  不需要大规模训练数据集或复杂的特征工程，仅使用磁力计数据即可完成高精度导航，适用于嵌入式硬件部署。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|--------|
| **初始化要求** | 需要专用校准飞行或历史数据 | 支持零先验“冷启动” |
| **实时适应性** | 多为离线处理，无法动态调整 | 在线持续学习，适应热漂移、振动等变化 |
| **模型可解释性** | TL 模型可解释，纯 NN 为黑盒 | TL 为主干，NN 为残差，保留物理基础 |
| **收敛速度** | 一阶梯度下降较慢 | 自然梯度等效，收敛更快更稳定 |
| **部署可行性** | 高算力需求，难嵌入 | 浅层 NN + EKF，适合机载系统 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用公开的 **MagNav Challenge Dataset** [38]，由 MIT 和 DAF 提供。
- 主要分析 **flight line 1007.06**（时长约 87 分钟），并在附录中验证了另一条长航迹 **flight line 1003**（总时长 4.15 小时）。
- 数据包含多个磁力计传感器（Flux A 及磁力计 1–5），其中部分传感器受平台干扰严重（如 Magnetometer 2 干扰达 ~1250 nT）。

### 实验设置
- **两种初始化模式对比**：
  - **Cold Start (C)**：所有参数（TL、NN、SCB）初始化为零或随机，协方差 $P_0$ 设为大值，表示无先验知识。
  - **Warm Start (W)**：复用前次飞行结束后的最终参数及其协方差矩阵，模拟记忆延续。
- **NN 架构**：浅层全连接网络，1 个隐藏层（2–128 个神经元），激活函数为 `tanh`，输出层无线性偏置项。
- **输入特征集**：仅使用磁力计原始测量值（magnetometer-only feature set），未引入时间导数或其他辅助信号，简化系统设计。
- **噪声建模**：通过过程噪声 $Q$ 和观测噪声 $R$ 控制学习速率动态演化。

### 评估指标
- **定位精度**：水平位置 **DRMS**（Distance Root Mean Square），单位为米：
  $$
  \text{DRMS} = \sqrt{2 \cdot \mathbb{E}\left[(x_{\text{GNSS}} - x_{\text{MagNav}})^2 + (y_{\text{GNSS}} - y_{\text{MagNav}})^2\right]}
  $$
- **校准效果**：磁干扰预测的 **RMSEₘ**（单位：nT）：
  $$
  \text{RMSE}_m = \sqrt{\frac{1}{N}\sum (m - h(\mathbf{x}))^2}
  $$

### 基线方法对比
- **TL-only Online Calibration**：仅使用在线更新的 Tolles-Lawson 模型，无 NN 补偿。
- **Gnadt [8] 的 “Online Model 2c”**：需基于校准飞行和 12 小时数据进行预训练的 TL+NN 方法，代表当前最优离线训练水平。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Cold Start, Nₕ=5）
| 磁力计 | 平均干扰 | DRMS (本文方法) | DRMS (TL-only) | 提升幅度 |
|--------|----------|------------------|----------------|-----------|
| Mag 3  | ~1150 nT | **42 m**         | 46 m           | ~9%       |
| Mag 4  | ~250 nT  | **37 m**         | 58 m           | ~36%      |
| Mag 5  | ~100 nT  | **14 m**         | 15 m           | ~7%       |

> 注：相比需预训练的 Gnadt [8] 方法（Mag 3: 32 m, Mag 4: 37 m, Mag 5: 18 m），本文在无需任何校准飞行的情况下达到了**相当甚至更优**的性能。

### 与基线方法对比结果
- 在所有受干扰严重的磁力计上，**TL+NN 混合模型显著优于 TL-only 方案**，尤其在前 50 km 内表现突出，说明 NN 有效抑制了初始漂移。
- **冷启动下性能接近甚至超越需预训练的方法**，证明了在线自然梯度学习的强大适应能力。
- **Warm Start 进一步提升鲁棒性**，特别是在高干扰传感器（如 Mag 2）上，DRMS 从 51 m（冷启）降至 48 m（暖启）。

### 消融实验结果
- **NN 规模影响**：
  - 增加隐藏层神经元数量可略微降低 RMSEₘ，但对 DRMS 影响不一致。
  - 过大的网络可能导致状态可观测性下降（通过 Cramér-Rao Lower Bound 分析证实），反而不利于导航精度。
  - 推荐使用 **极简网络**（Nₕ ≈ 2–5），避免过拟合并节省算力。
- **输入特征影响**：
  - 仅使用磁力计输入（pm）的表现优于加入速度等额外特征（mv 或 full set），表明特征冗余可能损害学习效率。
- **残差学习有效性**：
  - TL 模型补偿了数千 nT 级别的主干扰；
  - NN 输出稳定在数百 nT 范围内，验证了其作为“精细残差修正”的角色，而非主导建模。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **首次实现了真正意义上的 MagNav 冷启动**：系统可在无任何先验信息条件下完成高精度导航，打破了对校准飞行的依赖。
2. ✅ **EKF-NN 联合状态估计是高效的在线学习范式**：其更新机制等价于自然梯度下降，兼具快速收敛与数值稳定性。
3. ✅ **“物理模型为主 + NN 为辅”的混合架构最具实用价值**：既保证了解释性与安全性，又提升了对复杂非线性干扰的建模能力。
4. ✅ **极简设计即可达到高性能**：浅层 NN 与单一传感器输入足以满足导航级需求，极大增强了工程落地可行性。

### 方法的局限性
- **初始收敛期存在约 50–100 km 的过渡阶段**：在此期间系统尚未完全学习到干扰模式，定位精度较低，需结合 INS 或 GNSS 进行交叉验证。
- **NN 激活函数饱和问题**：实验观察到间歇性的 innovation spikes，推测与 `tanh` 函数在极端输入下的梯度消失有关。
- **缺乏形式化的收敛证明**：尽管实践中表现良好，但从理论角度仍需建立更严格的稳定性与收敛性边界。
- **对地图质量敏感**：若磁异常图存在误差或缺失区域，会影响整体性能。

### 未来工作方向
- 🔄 **采用去饱和激活函数**：尝试 Mish、SiLU 或 GELU 替代 `tanh`，提升对尖锐干扰的学习能力。
- 🔗 **联邦学习与解耦架构**：探索 Partial-Update Schmidt-Kalman Filter 等机制，防止 NN 学习初期震荡影响主导航状态。
- 📈 **移动窗口估计（Moving Horizon Estimation）**：增强对地图伪影的鲁棒性。
- 🛰️ **紧耦合架构与故障检测**：实现传感器失效与模型学习的区分，提升系统完整性。
- 📊 **成熟度指标开发**：定义“校准成熟度”度量，用于判断何时可进入高完整度运行模式，并支持后续飞行的 warm start。
- 🧲 **量子磁力计融合**：探索基于量子传感的更高精度、更低噪声测量，进一步释放 MagNav 潜力。

--- 

> **总结**：本研究提出了一种面向实际部署的、可认证的 MagNav 解决方案，将物理建模与数据驱动学习有机结合，在无需离线训练的前提下实现了媲美顶尖预训练模型的导航精度，为未来抗干扰、自主化航空导航系统提供了关键技术路径。

</details>

---

### 13. [DualFlexKAN: Dual-stage Kolmogorov-Arnold Networks with Independent Function Control](https://arxiv.org/abs/2603.08583)

**Authors**: Andr\'es Ortiz, Nicol\'as J. Gallego-Molina, Carmen Jim\'enez-Mesa, Juan M. G\'orriz, Javier Ram\'irez  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.08583v1  

#### Abstract
Multi-Layer Perceptrons (MLPs) rely on pre-defined, fixed activation functions, imposing a static inductive bias that forces the network to approximate complex topologies solely through increased depth and width. Kolmogorov-Arnold Networks (KANs) address this limitation through edge-centric learnabl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DualFlexKAN: Dual-stage Kolmogorov-Arnold Networks with Independent Function Control**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
传统 **Multi-Layer Perceptrons (MLPs)** 依赖固定的激活函数（如 ReLU），导致其表达能力受限，必须通过增加网络深度和宽度来逼近复杂函数，造成参数冗余和训练低效。  
**Kolmogorov-Arnold Networks (KANs)** 虽然引入了可学习的边函数（edge-centric learnable functions）以提升表达力，但存在以下问题：
- 参数量呈二次增长（$O(n_{\text{in}} \cdot n_{\text{out}} \cdot m)$），难以扩展；
- 架构僵化，缺乏对不同层或位置的灵活性控制；
- 训练不稳定，难以集成标准正则化技术（如 Dropout、BatchNorm）。

### **提出的新方法**
本文提出了 **DualFlexKAN (DFKAN)**，一种具有双阶段机制的灵活架构，核心创新如下：

#### **(1) 双阶段解耦设计（Dual-Stage Mechanism）**
将非线性变换分为两个独立阶段：
- **Pre-linear 输入变换（Input Transformation）**：在权重乘法前对输入进行可学习的非线性处理；
- **Post-linear 输出激活（Output Activation）**：在仿射变换后应用可学习的输出激活函数。

这种解耦实现了对输入和输出非线性的**独立控制**，支持构建混合架构，在表达力与计算成本之间取得更好平衡。

#### **(2) 多级函数共享策略（Hierarchical Function Strategies）**
为输入和输出阶段分别定义多种策略，实现细粒度控制：
- **输入变换 T(·)** 支持 5 种策略：
  - S0: 无变换（Identity）
  - S1: 固定非线性（Fixed Function）
  - S2: 全局共享可学习函数（Global Shared）
  - S3: 每维度独立函数（Per-Dimension）
  - S4: 每连接独立函数（Per-Connection，仅用于输入）
- **输出激活 φ(·)** 支持 4 种策略（S0–S3）

该设计使 DFKAN 能涵盖从纯 MLP 到全 KAN 的整个架构谱系。

#### **(3) 多样化的基函数族与正则化支持**
- 支持多种可学习基函数：**B-splines**, **Legendre/Jacobi/Gegenbauer 多项式**, **Radial Basis Functions (RBF)**, **Sine/Spectral**, **Wavelets**, **Rational Functions** 等。
- 提供灵活的正则化框架，支持 **Dropout** 和 **Batch Normalization** 在 pre-act 或 post-act 位置配置，并可组合使用。

#### **(4) 生物学启发的设计**
- 输入变换模拟神经元树突的局部非线性计算（dendritic computation）；
- 输出激活模拟胞体整合与动作电位生成（somatic integration）；
- 层间策略切换模仿大脑皮层从高可塑性感知区到稳定决策区的层级处理。

---

### **相比现有方法的优势**
| 维度 | DFKAN 优势 |
|------|-----------|
| **参数效率** | 比标准 KAN 减少 **1–2 个数量级**的参数，接近 MLP 规模 |
| **表达能力** | 保留 KAN 式的强表达力，能高效建模物理规律中的乘法、除法、根号等结构 |
| **训练稳定性** | 支持标准正则化，缓解过拟合与噪声敏感问题 |
| **架构灵活性** | 可按需配置每层的函数策略，适应不同任务需求 |
| **可解释性** | 显式学习单变量函数，便于可视化分析与符号回归 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖三大类任务，共 **14 个基准数据集**：

| 类别 | 数据集 | 特点 |
|------|--------|------|
| **物理驱动任务** | Friedman#1, Friedman#2, Franke Function, Feynman I.18.12, Feynman II.6.11 | 包含非线性交互、乘法、$1/r$、$\sqrt{}$ 等数学结构 |
| **复合与高频函数** | Damped Oscillator ($e^{-t}\sin(\omega t)$), Sin_Exp ($\sin(e^x)$), Nested Trig ($\sin(\cos(\sin(x)))$) | 测试频谱偏置（spectral bias）克服能力 |
| **真实世界回归** | UCI 数据集：<br>Yacht Hydrodynamics, Servo, Boston Housing, Auto MPG, Diabetes | 小样本、异构特征、噪声环境下的泛化能力 |

> ✅ 所有数据集样本数均 ≤ 5000，强调小数据场景下的表现。

---

### **实验设置与评估指标**

#### **模型配置**
- 对比模型：
  - **MLP**：优化后的三层网络，使用 ReLU/Tanh 激活
  - **KAN (vanilla)**：原始边函数形式，B-spline 参数化
  - **DFKAN**：采用混合策略（如首层 S4 输入 + 后续层 S2/S3）
- 基函数选择：B-splines、Legendre polynomials 为主
- 优化器：Adam
- 初始化：He 初始化（权重），系数衰减初始化（basis coefficients）

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **MSE / RMSE** | 回归误差主指标 |
| **R² Score** | 决定系数，衡量拟合优度 |
| **Parameter Count** | 总参数量，评估模型规模 |
| **Effective Parameter Count** | 经剪枝后维持 90% 性能所需的最少参数，反映结构稀疏性 |
| **Training Time (s)** | 单次训练耗时，评估计算开销 |
| **Gradient Fidelity** | 学习梯度场与真实梯度的匹配程度（用于 manifold 分析） |

---

### **基线方法对比**
- **MLP**：代表经典深度学习范式
- **Vanilla KAN**：代表当前最先进的可学习激活网络
- **DFKAN**：本文提出的方法，作为两者的中间桥梁

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 数据集 | 最佳方法 | MSE | R² | 参数量（相对） |
|--------|---------|-----|----|----------------|
| **Friedman#2** | DFKAN | **1.2e-4** | **0.998** | ~1/50 of KAN |
| **Feynman I.18.12** | DFKAN | **8.7e-5** | **0.999** | ~1/100 of KAN |
| **Damped Oscillator** | DFKAN | **3.1e-4** | **0.996** | ~1/80 of KAN |
| **Sin_Exp** | DFKAN ≈ KAN | ~2e-4 | ~0.995 | DFKAN 少 2× 参数 |
| **Yacht Hydrodynamics** | MLP | 0.012 | 0.92 | — |
| | DFKAN | 0.014 | 0.90 | **仅为 KAN 的 1%** |
| | KAN | 0.021 | 0.85 | 参数爆炸 |

> 📊 图表显示：**DFKAN 在物理任务上全面超越 MLP 和 KAN；在真实数据上优于 KAN，略逊于最优 MLP，但参数极少。**

---

### **与基线方法的对比结果**

#### **(1) 参数效率碾压 KAN**
- 如图 4 所示，**DFKAN 参数量通常比 vanilla KAN 低 1–2 个数量级**，与 MLP 相当。
- 例如在 Feynman 任务中，KAN 需 25,000+ 参数，而 DFKAN 仅需 ~300。

#### **(2) 结构稀疏性最强**
- 定义 “有效参数”（Effective Parameters）为剪枝后保持 90% 性能所需最小参数。
- **中位有效参数**：
  - MLP: ~6,721
  - KAN: ~281
  - **DFKAN: ~93**
- ➡️ DFKAN 是最紧凑且高效的表示。

#### **(3) 训练速度更快**
- 如图 5 所示，DFKAN 平均训练时间显著低于 KAN，尤其在宽网络中优势明显。

#### **(4) 物理任务全面领先**
- 在 **Friedman#2** 和 **Feynman 方程** 上，DFKAN 均达到最低 MSE，因其能用 Legendre 多项式直接逼近 $\sqrt{}$, $1/x$ 等光滑流形。
- MLP 因 ReLU 的分段线性特性难以精确建模。

#### **(5) 抗噪能力强，支持符号发现**
- 在加噪的 $y=2x^2 - x + 0.5$ 任务中：
  - KAN 过拟合噪声（出现高频振荡）
  - **DFKAN 自动平滑噪声，恢复出干净的二次公式**（见图 10）
- 体现其“奥卡姆剃刀”（Occam’s Razor）性质。

#### **(6) 梯度保真度最高**
- 在 $z = \sin(2x)\cos(2y)$ 的 manifold 学习任务中：
  - MLP：函数值尚可，但梯度模糊（spectral bias）
  - KAN：训练失败（MSE 达 0.25）
  - **DFKAN：MSE = 2.9e-4，准确重建梯度拓扑**
- 表明其适用于 **Physics-Informed Neural Networks (PINNs)**。

---

### **消融实验结果**
虽然未明确列出“ablation study”章节，但从多组配置比较中可推断以下结论：

| 配置变化 | 影响 |
|--------|------|
| 使用 **Per-Connection 输入 (S4)** | 显著提升对高阶交互的捕捉能力（如 Feynman 任务） |
| 后续层使用 **Global Shared 激活 (S2)** | 大幅降低参数量，增强泛化 |
| 引入 **BatchNorm + Dropout** | 提升训练稳定性，防止过拟合 |
| 选用 **Legendre 多项式** vs B-spline | 更适合高频/光滑函数逼近 |
| 不同正则化顺序（BN before Dropout） | 影响梯度流动，需调优 |

---

## **4. 关键结论和发现**

### **主要发现**
1. **DFKAN 成功弥合了 MLP 与 KAN 之间的鸿沟**：
   - 在表达力上媲美 KAN，
   - 在参数效率上接近 MLP，
   - 在可解释性上远超两者。

2. **双阶段解耦是关键突破**：
   - 解放了对输入与输出非线性的独立控制，
   - 实现了真正的“混合架构”设计自由。

3. **结构正则化优于参数正则化**：
   - 通过函数共享策略（如 S2/S3）天然抑制过拟合，
   - 在小样本场景下比 KAN 更鲁棒。

4. **适用于科学发现（AI4Science）**：
   - 可直接提取符号公式；
   - 能恢复物理系统的微分结构；
   - 是理想的 **PINN backbone**。

---

### **方法的局限性**
1. **超参数敏感**：
   - 函数策略、基函数类型、正则化位置等需手动选择，搜索空间大。
2. **真实表格数据上仍略逊于调优 MLP**：
   - 在 UCI 数据集中，MLP 仍常取得最佳 MSE/R²。
3. **牺牲部分可解释性换取效率**：
   - 使用全局共享函数（S2）会丢失 per-connection 的细粒度解释能力。
4. **尚未拓展至 CV/NLP**：
   - 当前验证集中在回归与函数逼近任务。

---

### **未来工作方向**
1. **自动化架构搜索（NAS）**：
   - 开发算法自动选择最优函数策略组合。
2. **理论分析**：
   - 形式化证明 DFKAN 的逼近能力边界。
3. **跨领域扩展**：
   - 探索在 **Computer Vision** 和 **Natural Language Processing** 中的应用。
4. **神经科学交叉研究**：
   - 深入分析学习到的 activation functions 是否符合生物神经元动力学。
5. **轻量化部署**：
   - 推动 DFKAN 在 **Edge AI** 和 **TinyML** 场景中的落地。

---

> 🔗 **代码开源**：DFKAN 库已发布于 GitHub：[https://github.com/BioSIP/dfkan](https://github.com/BioSIP/dfkan)

✅ **总体评价**：  
**DualFlexKAN 是一次兼具理论深度与工程实用性的创新，为下一代可解释、高效、科学友好的神经网络架构提供了坚实基础。**

</details>

---

### 14. [Animating Petascale Time-varying Data on Commodity Hardware with LLM-assisted Scripting](https://arxiv.org/abs/2603.07053)

**Authors**: Ishrat Jahan Eliza, Xuan Huang, Aashish Panta, Alper Sahistan, Zhimin Li, Amy A. Gooch, Valerio Pascucci  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.07053v1  

#### Abstract
Scientists face significant visualization challenges as time-varying datasets grow in speed and volume, often requiring specialized infrastructure and expertise to handle massive datasets. Petascale climate models generated in NASA laboratories require a dedicated group of graphics and media experts...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Animating Petascale Time-varying Data on Commodity Hardware with LLM-assisted Scripting*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
科学领域中，随着传感器分辨率和模拟精度的提升，**time-varying 数据集**（如气候、海洋模型）规模迅速增长至 **petascale 级别（超过 1PB）**。传统可视化流程面临以下挑战：
- 需要高性能计算（HPC）资源和专用基础设施；
- 可视化专家团队介入，耗时且成本高；
- 科学家在普通工作站上难以快速生成高质量动画用于分析或传播。

本论文旨在解决：**如何让非可视化专业的科学家在普通硬件（commodity hardware）上高效创建 petascale 时间序列数据的 3D 动画**。

---

### 🚀 提出的新方法与创新点

作者提出一个**端到端的动画生产框架**，其核心创新包括：

#### （1）Generalized Animation Descriptor (GAD)
- 一种**应用无关（application-independent）的 JSON 类脚本格式**，用于描述复杂动画流程。
- 支持 keyframe-based 抽象，可定义场景边界、相机路径、transfer function、时间范围等。
- 分层设计（Header、Data List、Keyframe Files），便于模块化管理和多分辨率渲染。

> ✅ 优势：解耦动画逻辑与底层渲染工具（如 VTK、OSPRay），实现跨平台兼容性。

#### （2）LLM-assisted Conversational Scripting Interface
- 引入基于 **GPT-4o 的多模态大语言模型（MLLM）** 构建自然语言交互界面。
- 用户可通过自然语言输入（如 “show salinity in the Mediterranean Sea”）自动生成 GAD 脚本。
- 支持迭代反馈机制：AI 分析渲染结果并建议参数优化（如调整区域、添加 streamlines、提高分辨率）。

> ✅ 优势：极大降低使用门槛，无需掌握坐标系统、transfer function 或编程技能。

#### （3）高效的远程数据访问机制
- 基于 **OpenVisus + NSDF 数据抽象层**，直接从云端按需下载指定时空子集（subset）。
- 支持多分辨率 streaming（downsampling），先低分辨率预览再逐步细化。

> ✅ 优势：避免全量下载导致的存储溢出和带宽瓶颈。

#### （4）轻量化本地渲染流水线
- 利用 **OSPRay（CPU 光追） 和 VTK** 作为后端渲染器，通过 Python binding 驱动。
- 内存复用策略：逐帧加载与渲染，显著减少内存占用。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法（如 NASA SVS） | 本文方法 |
|------|------------------------|---------|
| 所需硬件 | HPC + 专业图形集群 | 普通工作站（<128GB RAM） |
| 用户技能要求 | 高（需熟悉 ParaView/VisIt/VTK） | 无（支持自然语言交互） |
| 工作流效率 | 数天至数周（人工调试） | 分钟级原型，小时级成品 |
| 数据管理复杂度 | 高（需本地存储 PB 级数据） | 低（按需远程读取） |
| 可扩展性 | 依赖特定软件栈 | 应用无关 GAD 格式，易于集成 |

---

## 2. **核心实验方法和设置**

### 📊 使用的数据集
- **NASA DYAMOND LLC2160 海洋模拟数据集**
  - 总大小：**1.8 PB**
  - 时间步长：10,269 个 hourly timesteps（约 14 个月）
  - 包含字段：
    - 3D scalar fields：temperature, salinity, velocity (u/v/w)
    - 多个 2D 场
  - 存储格式：IDX（由 OpenVisus 支持）
  - 公开访问地址：[NASA Data Portal](https://portal.nccs.nasa.gov/datashare/G5NR/DYAMONDv2/)

---

### ⚙️ 实验设置

#### 硬件环境
- CPU: 12th Gen Intel® Core™ i7-14700
- RAM: 128 GB
- GPU: Quadro RTX 5000
- 无专用 HPC 或分布式存储

#### 渲染配置
- 输出分辨率：2048×2048 PNG/GIF
- 渲染引擎：OSPRay（默认）、VTK（可选）
- 多分辨率策略：支持 1/256 → 1/64 → 1/16 原始分辨率渐进加载

#### 评估方式
- **定性评估**：通过两个 case study 展示动画质量和科学有效性。
- **定量指标**：
  - **Turnaround time**：从请求到首帧/最终动画完成的时间
  - **内存使用峰值**
  - **数据下载量**
- **对比基线**：隐式对比传统流程（需手动编写脚本、多次试错、大量数据传输）

> ❗ 注：未显式对比其他自动化工具（如 ParaView + Python 脚本），而是强调对“零经验用户”的可用性提升。

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据

| 指标 | 结果 |
|------|------|
| **首次粗略动画生成时间** | **1–5 分钟**（低分辨率 + 快速下载） |
| **完整高质量动画生成时间** | **1–2 小时**（取决于分辨率和帧数） |
| **最大单次数据下载量** | ~几十 GB（仅 ROI 子集） |
| **内存峰值使用** | <100 GB（得益于逐帧处理） |
| **典型动画长度** | 60–90 frames（对应 60–90 天动态演化） |

> 示例：地中海盐度动画（60 天，1/16 分辨率）总耗时约 **75 分钟**（含 3 次 AI 迭代优化）。

---

### 🧪 Case Study 对比结果

#### **Case Study 1: Agulhas Ring 可视化**
- **任务**：追踪印度洋 Agulhas 流回转形成的环流结构（Agulhas Rings）
- **流程**：
  1. 使用交互式 viewer 定义 ROI（南纬 30°–45°, 东经 15°–30°）
  2. 下载 90 个 timestep 的 full-resolution 数据（~20GB × 90）
  3. 生成 GAD 脚本并渲染
- **结果**：
  - 成功捕捉到环流形成与脱落过程（见 Fig. 6 & 7）
  - 总耗时：**~42 分钟**（30min 脚本+下载 + 12min 渲染）
  - 支持 transfer function 调优以增强温跃层可见性

#### **Case Study 2: LLM-assisted 探索地中海与红海盐度**
- **任务**：探索地中海 Meddies 与红海涡旋的动力学特征
- **流程**：
  - 用户输入自然语言：“Mediterranean sea salinity with 60 days”
  - AI 自动推断时空范围、分辨率、是否启用 streamlines
  - 经过 **4 轮迭代**，AI 建议加入 velocity streamlines 并提升分辨率至 1/16
  - 最终动画清晰展示表层淡水流入与深层咸水流出形成的 Meddies（Fig. 8）
- **迁移能力测试**：
  - 请求 “Red Sea Currents with Salinity”，AI 初始定位错误（东南非附近）
  - 经用户引导后成功聚焦 Bab el Mandeb Strait 区域（Fig. 1c）
  - 表明系统具备一定泛化能力，但仍需上下文微调

---

### 🔍 消融实验（Implicit Ablation）

虽然未明确列出消融实验表格，但从流程中可推断：

| 组件 | 是否移除影响显著？ | 说明 |
|------|--------------------|------|
| **LLM-assisted scripting** | 是 | 若仅靠基础 Python 脚本，需用户已知坐标和参数，学习曲线陡峭 |
| **Multi-resolution access** | 是 | 若强制加载 full-res 数据，内存将超限，无法运行 |
| **GAD abstraction** | 是 | 缺少该层则无法统一调度不同 backend（如 OSPRay/VTK） |
| **Remote data streaming** | 是 | 若需预先下载整个 PB 级数据集，则不可行 |

---

## 4. **关键结论和发现**

### ✅ 主要结论

1. **Petascale 动画可在 commodity hardware 上实现**
   - 通过 **progressive refinement workflow**（低分辨率原型 → 高分辨率精修），即使在 128GB RAM 工作站也能处理 >1PB 数据。

2. **LLM 显著降低了科学可视化的准入门槛**
   - 非专家用户可通过自然语言快速获得初步可视化结果，并借助 AI 反馈持续优化。
   - 实现了“idea to animation”的分钟级转化。

3. **GAD 提供了一种灵活、可移植的动画描述标准**
   - 解耦了动画设计与具体渲染工具，为未来构建可视化 DSL（Domain-Specific Language）奠定基础。

4. **科学发现优先于技术细节**
   - 用户能专注于现象本身（如 Meddies 形成机制），而非陷入数据格式转换、内存管理等琐事。

---

### ⚠️ 局限性

1. **LLM 输出存在随机性和不稳定性**
   - GPT-4o 在推理过程中可能产生轻微不同的参数建议，影响重复性。
   - 当前为 rule-based prompting，尚未 fine-tuned 专用于科学动画任务。

2. **地理空间理解有限**
   - 对未训练过的区域（如 Red Sea）初始定位不准，依赖用户纠正。

3. **缺乏自动质量评估指标**
   - 当前依赖人工判断“是否好看”，尚无量化 metric 衡量科学可视化质量。

4. **暂未支持实时交互浏览**
   - 当前为离线批处理模式，不适合探索式 in-situ 分析。

---

### 🔮 未来工作方向

1. **Fine-tune MLLM for scientific animation generation**
   - 构建专门的 training corpus，提升参数预测准确率和一致性。

2. **引入自动化评价体系**
   - 设计科学动画质量 metric（如 feature visibility, motion coherence），推动 fully autonomous pipeline。

3. **扩展 backend 支持**
   - 集成 PyVista、ANARI 等现代渲染框架，提供更丰富的视觉效果选项。

4. **支持更多数据类型**
   - 推广至大气、地质、宇宙学等领域，验证通用性。

5. **构建共享 GAD 脚本库**
   - 社区协作积累常见现象的标准动画模板，进一步加速复现。

---

> 💡 **一句话总结**：  
> 本文通过 **GAD + LLM-assisted scripting + cloud-streaming** 三重创新，实现了在普通电脑上“一句话生成 petascale 科学动画”的愿景，**将高端科学可视化民主化**，为大规模数据驱动科研提供了全新范式。

</details>

---

### 15. [Adaptive Collaboration with Humans: Metacognitive Policy Optimization for Multi-Agent LLMs with Continual Learning](https://arxiv.org/abs/2603.07972)

**Authors**: Wei Yang, Defu Cao, Jiacheng Pang, Muyan Weng, Yan Liu  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.07972v1  

#### Abstract
While scaling individual Large Language Models (LLMs) has delivered remarkable progress, the next frontier lies in scaling collaboration through multi-agent systems (MAS). However, purely autonomous MAS remain ''closed-world'' systems, constrained by the static knowledge horizon of pre-trained model...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Adaptive Collaboration with Humans: Metacognitive Policy Optimization for Multi-Agent LLMs with Continual Learning

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Multi-Agent Systems (MAS)** 尽管在复杂任务上表现出色，但本质上是“闭世界”系统（closed-world systems），其知识边界受限于预训练模型的数据。当面对需要实时信息、领域专业知识或训练数据中未见推理模式的任务时，这些系统容易出现**集体失败**（collective failure）。此外，现有 **Human-in-the-loop** 系统通常将人类视为被动的“oracle”或监督者，缺乏对“何时求助”和“如何从反馈中学习”的智能决策机制。

### 提出的新方法与新思路
本文提出了 **Human-In-the-Loop Multi-Agent Collaboration (HILA)** 框架，并引入 **Dual-Loop Policy Optimization (DLPO)** 训练范式，以实现自适应的人机协作。

#### 核心创新点：
- **Metacognitive Policy（元认知策略）**  
  HILA 赋予每个 agent 一个元认知策略，使其能够动态判断自身能力边界，决定是否应自主解决问题，还是将任务**战略性地委派**（Strategic Deferral）给人类专家。这超越了基于置信度阈值的启发式规则。

- **Dual-Loop Policy Optimization (DLPO)**  
  一种双循环优化框架，分离短期决策与长期能力增长：
  - **Inner Loop（内环）**：使用 **Group Relative Policy Optimization (GRPO)** 和成本感知奖励（cost-aware reward）来优化“何时求助”的决策策略。
  - **Outer Loop（外环）**：通过 **Continual Learning** 将人类专家的反馈转化为高质量的监督信号，持续提升 agent 的底层推理能力。

- **开放式的知识扩展机制**  
  “Defer” 动作不仅是风险规避手段，更是**知识注入通道**。每次向人类求助后获得的高质量解答都会被存储并用于后续的监督微调（SFT），从而突破 LLM 的静态知识天花板。

### 相比现有方法的优势
| 维度 | 传统 MAS / Human-in-the-loop | HILA |
|------|-------------------------------|------|
| **知识获取** | 仅重组已有知识 | 可通过人类反馈持续学习新知识 |
| **求助决策** | 启发式（如低置信度） | 学习型元认知策略，权衡成功率与干预成本 |
| **反馈利用** | 一次性修正 | 持续学习，转化为长期能力提升 |
| **系统演化** | 静态协作 | 动态、自适应、持续进化的协作系统 |

---

## 2. 核心实验方法和设置

### 数据集
实验覆盖多个高难度基准测试，涵盖不同推理类型：
- **数学推理**：
  - **GSM8K**：小学算术应用题
  - **AMC**：美国数学竞赛题（中等难度）
  - **AIME**：美国数学邀请赛题（高难度）
- **程序合成**：
  - **HumanEval**：代码生成任务
- **通用知识与分析推理**：
  - **MMLU**：涵盖57个学科的多项选择题

### 实验设置与评估指标
- **骨干模型（Backbone）**：使用 `LLaMA3-8B`、`LLaMA3-3B`、`Qwen2.5-7B`、`Qwen2.5-3B` 等开源 LLM。
- **人类代理（Human Proxy）**：为控制成本与可复现性，采用强 LLM（如 `GPT-4o-mini`, `GPT-4o`）模拟人类专家提供高质量解答。
- **评估指标**：
  - 数学与程序任务：**Solve Rate**（精确匹配率）
  - 多项选择任务：**Accuracy**
- **协作配置**：默认使用 **3个 agent**，进行 **3轮交互**。
- **训练方式**：采用 **LoRA** 进行参数高效微调，结合 **GRPO** 与 **SFT** 实现双循环优化。

### 基线方法对比
对比了三大类主流方法：
1. **Single-Agent 方法**：
   - Vanilla（直接生成）
   - Chain-of-Thought (CoT)
   - Self-Consistency (SC)

2. **Interactive Multi-Agent 方法**：
   - LLM-Debate（辩论式交互）
   - G-Debate（图结构辩论）
   - DyLAN（动态拓扑控制）
   - G-Swarm（图优化协作）

3. **System-Level Coordination 方法**：
   - A-Prune（通信剪枝）
   - AFlow（自动化工作流）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（基于 LLaMA3-8B）

| 方法 | GSM8K | AMC | AIME | HumanEval | MMLU |
|------|-------|-----|------|-----------|------|
| Vanilla | 72.76 | 8.03 | 2.96 | 47.56 | 57.99 |
| CoT | 74.22 | 11.65 | 3.70 | 51.42 | 61.57 |
| Debate | 83.52 | 19.28 | 5.56 | 57.72 | 67.59 |
| G-Swarm | 84.89 | 15.66 | 5.78 | 59.55 | 69.67 |
| **HILA (Ours)** | **89.86** | **35.83** | **9.37** | **72.15** | **73.62** |

> ✅ **HILA 在所有基准上均显著优于最强基线**，绝对提升达 **3.7 ~ 15.4 个百分点**，尤其在 AMC 和 AIME 等竞赛级数学任务上表现突出。

### 跨骨干泛化能力（GSM8K 上的结果）

| Backbone | Vanilla | HILA | 提升 |
|----------|--------|------|------|
| Qwen2.5-7B | 90.71 | **94.72** | +4.01 |
| Qwen2.5-3B | 83.25 | **91.17** | +7.92 |
| LLaMA3-8B | 72.76 | **89.86** | +17.10 |
| LLaMA3-3B | 45.26 | **83.85** | +38.59 |

> ✅ HILA 的优势在更小、更弱的模型上更为明显，说明其能有效补偿基础模型的能力不足。

### 消融实验结果

#### （1）训练阶段消融（Table 3）
| 方法 | GSM8K | AMC | MMLU |
|------|-------|-----|------|
| HILA (Init Policy) | 88.15 | 33.33 | 68.30 |
| HILA + GRPO（仅内环） | 88.38 | 32.50 | 70.47 |
| HILA + DLPO（完整双环） | **89.86** | **35.83** | **73.62** |

> 🔍 结果表明：仅优化决策策略（GRPO）带来的增益有限；而加入外环的持续学习（SFT）后，性能显著跃升，证明**能力增长**是关键驱动力。

#### （2）backbone 迁移实验（Table 4）
将经过 DLPO 训练后的 backbone 应用于其他推理框架（如 Vanilla、Debate），仍能带来一致性能提升：
- Debate 原始准确率：83.52 → 使用 DLPO 更新 backbone 后：**88.93**
- 表明 DLPO 不仅优化了协作策略，还**实质性提升了 backbone 的通用推理能力**。

#### （3）策略分布变化（Table 5）
随着训练推进，“DEFER”比例持续下降，“EVAL”比例上升：
- 初始策略：DEFER 占 ~24–29%
- 完整 DLPO：DEFER 下降至 **5–17%**

> 📈 说明系统学会了更少依赖外部帮助，更多通过内部协作解决任务，体现了真正的**能力成长**而非单纯的成本规避。

---

## 4. 关键结论和发现

### 主要发现
1. **智能求助优于盲目协作**  
   HILA 通过元认知策略实现了“有选择地求助”，避免了无意义的集体内耗和错误传播。

2. **人类反馈是系统进化的燃料**  
   “Defer” 不是终点，而是起点。每一次人类干预都被转化为监督数据，驱动整个系统持续进化。

3. **双循环设计是成功的关键**  
   内环优化“何时问”，外环解决“问完后怎么变强”。两者协同实现了**战略智能 + 能力增长**的双重目标。

4. **更强的人类专家带来更大收益**  
   实验显示，使用 `GPT-4o` > `GPT-4o-mini` > `GPT-3.5-turbo` 作为 human proxy 时，HILA 性能依次提升，验证了“质量决定上限”。

5. **真实人类参与同样有效**  
   在引入真实 PhD 专家的实验中，HILA 依然表现出色，且人类提供的**完整推理链**比简单提示更能提升系统性能。

### 方法的局限性
- **依赖高质量人类反馈**：若人类专家不可靠，系统可能学到错误知识。
- **计算开销较高**：多 agent + 多轮交互导致 token 消耗大，尤其在大规模部署时需权衡效率。
- **当前依赖强 LLM 作为 human proxy**：尚未完全脱离对先进 LLM 的依赖。
- **元认知信号为轻量级规则构建**：未来可探索端到端学习 metacognitive cues。

### 未来工作方向
- 探索更动态的协作机制（如角色自适应分配）
- 引入记忆机制以长期保存专家知识
- 扩展至具身智能（embodied agents）或多模态场景
- 研究多人类协作下的知识融合与冲突解决
- 构建真实世界中的人类反馈闭环系统

---

> 💡 **总体评价**：HILA 提供了一个**原则性强、可扩展、可持续进化**的多智能体协作新范式，标志着从“封闭式协作”向“开放式人机共智”的重要迈进。其核心思想——**将人类干预制度化为学习机会**——为构建真正具备成长能力的 agentic systems 指明了方向。

</details>

---

### 16. [In-Context Reinforcement Learning for Tool Use in Large Language Models](https://arxiv.org/abs/2603.08068)

**Authors**: Yaoqi Ye, Yiran Zhao, Keyu Duan, Zeyu Zheng, Kenji Kawaguchi, Cihang Xie, Michael Qizhe Shieh  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.08068v1  

#### Abstract
While large language models (LLMs) exhibit strong reasoning abilities, their performance on complex tasks is often constrained by the limitations of their internal knowledge. A compelling approach to overcome this challenge is to augment these models with external tools -- such as Python interpreter...

---

### 17. [CORE-Acu: Structured Reasoning Traces and Knowledge Graph Safety Verification for Acupuncture Clinical Decision Support](https://arxiv.org/abs/2603.08321)

**Authors**: Liuyi Xu, Yun Guo, Ming Chen, Zihan Dun, Yining Qian, An-Yang Lu, Shuang Li, Lijun Liu  
**Category**: cs.AI  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.08321v1  

#### Abstract
Large language models (LLMs) show significant potential for clinical decision support (CDS), yet their black-box nature -- characterized by untraceable reasoning and probabilistic hallucinations -- poses severe challenges in acupuncture, a field demanding rigorous interpretability and safety. To add...

---

### 18. [KohakuRAG: A simple RAG framework with hierarchical document indexing](https://arxiv.org/abs/2603.07612)

**Authors**: Shih-Ying Yeh, Yueh-Feng Ku, Ko-Wei Huang, Buu-Khang Tu  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.07612v1  

#### Abstract
Retrieval-augmented generation (RAG) systems that answer questions from document collections face compounding difficulties when high-precision citations are required: flat chunking strategies sacrifice document structure, single-query formulations miss relevant passages through vocabulary mismatch, ...

---

### 19. [EvoScientist: Towards Multi-Agent Evolving AI Scientists for End-to-End Scientific Discovery](https://arxiv.org/abs/2603.08127)

**Authors**: Yougang Lyu, Xi Zhang, Xinhao Yi, Yuyue Zhao, Shuyu Guo, Wenxiang Hu, Jan Piotrowski, Jakub Kaliski, Jacopo Urbani, Zaiqiao Meng, Lun Zhou, Xiaohui Yan  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.08127v1  

#### Abstract
The increasing adoption of Large Language Models (LLMs) has enabled AI scientists to perform complex end-to-end scientific discovery tasks requiring coordination of specialized roles, including idea generation and experimental execution. However, most state-of-the-art AI scientist systems rely on st...

---

### 20. [COACH meets QUORUM: A Framework and Pipeline for Aligning User, Expert and Developer Perspectives in LLM-generated Health Counselling](https://arxiv.org/abs/2603.08392)

**Authors**: Yee Man Ng, Bram van Dijk, Pieter Beynen, Otto Boekesteijn, Joris Jansen, Gerard van Oortmerssen, Max van Duijn, Marco Spruit  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.08392v1  

#### Abstract
Systems that collect data on sleep, mood, and activities can provide valuable lifestyle counselling to populations affected by chronic disease and its consequences. Such systems are, however, challenging to develop; besides reliably extracting patterns from user-specific data, systems should also co...

---

### 21. [Configurable Runtime Orchestration for Dynamic Data Retrieval in Distributed Systems](https://arxiv.org/abs/2603.06980)

**Authors**: Abhiram Kandiraju  
**Category**: cs.DC  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.06980v1  

#### Abstract
Modern enterprise platforms increasingly depend on distributed microservices, analytical data platforms, and external APIs to construct composite responses for applications. Orchestrating data retrieval across these heterogeneous systems is challenging because many workflow platforms rely on predefi...

---

### 22. [Reward Under Attack: Analyzing the Robustness and Hackability of Process Reward Models](https://arxiv.org/abs/2603.06621)

**Authors**: Rishabh Tiwari, Aditya Tomar, Udbhav Bamba, Monishwaran Maheswaran, Heng Yang, Michael W. Mahoney, Kurt Keutzer, Amir Gholami  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.06621v1  

#### Abstract
Process Reward Models (PRMs) are rapidly becoming the backbone of LLM reasoning pipelines, yet we demonstrate that state-of-the-art PRMs are systematically exploitable under adversarial optimization pressure. To address this, we introduce a three-tiered diagnostic framework that applies increasing a...

---

### 23. [Making LLMs Optimize Multi-Scenario CUDA Kernels Like Experts](https://arxiv.org/abs/2603.07169)

**Authors**: Yuxuan Han, Meng-Hao Guo, Zhengning Liu, Wenguang Chen, Shi-Min Hu  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.07169v1  

#### Abstract
Optimizing GPU kernels manually is a challenging and time-consuming task. With the rapid development of LLMs, automated GPU kernel optimization is gradually becoming a tangible reality. However, current LLM-driven automated optimization methods narrowly focus on machine learning applications, such a...

---

### 24. [wDPO: Winsorized Direct Preference Optimization for Robust LLM Alignment](https://arxiv.org/abs/2603.07211)

**Authors**: Jilong Liu, Yonghui Yang, Pengyang Shao, Haokai Ma, Wei Qin, Richang Hong  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.07211v1  

#### Abstract
Direct Preference Optimization (DPO) aligns large language models by optimizing pairwise preferences and has shown remarkable effectiveness as a simple and scalable alternative to RLHF. However, in practice, preference data are often noisy. Existing robust variants of DPO mainly rely on uniform obje...

---

### 25. [LeJOT-AutoML: LLM-Driven Feature Engineering for Job Execution Time Prediction in Databricks Cost Optimization](https://arxiv.org/abs/2603.07897)

**Authors**: Lizhi Ma, Yi-Xiang Hu, Yihui Ren, Feng Wu, Xiang-Yang Li  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.07897v1  

#### Abstract
Databricks job orchestration systems (e.g., LeJOT) reduce cloud costs by selecting low-priced compute configurations while meeting latency and dependency constraints. Accurate execution-time prediction under heterogeneous instance types and non-stationary runtime conditions is therefore critical. Ex...

---

### 26. [Reforming the Mechanism: Editing Reasoning Patterns in LLMs with Circuit Reshaping](https://arxiv.org/abs/2603.06923)

**Authors**: Zhenyu Lei, Qiong Wu, Jianxiong Dong, Yinhan He, Emily Dodwell, Yushun Dong, Jundong Li  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.06923v1  

#### Abstract
Large language models (LLMs) often exhibit flawed reasoning ability that undermines reliability. Existing approaches to improving reasoning typically treat it as a general and monolithic skill, applying broad training which is inefficient and unable to target specific reasoning errors. We introduce ...

---

### 27. [Lying to Win: Assessing LLM Deception through Human-AI Games and Parallel-World Probing](https://arxiv.org/abs/2603.07202)

**Authors**: Arash Marioriyad, Ali Nouri, Mohammad Hossein Rohban, Mahdieh Soleymani Baghshah  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.07202v1  

#### Abstract
As Large Language Models (LLMs) transition into autonomous agentic roles, the risk of deception-defined behaviorally as the systematic provision of false information to satisfy external incentives-poses a significant challenge to AI safety. Existing benchmarks often focus on unintentional hallucinat...

---

### 28. [RexDrug: Reliable Multi-Drug Combination Extraction through Reasoning-Enhanced LLMs](https://arxiv.org/abs/2603.08166)

**Authors**: Zhijun Wang, Ling Luo, Dinghao Pan, Huan Zhuang, Lejing Yu, Yuanyuan Sun, Hongfei Lin  
**Category**: cs.CL  
**Published**: 2026-03-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.08166v1  

#### Abstract
Automated Drug Combination Extraction (DCE) from large-scale biomedical literature is crucial for advancing precision medicine and pharmacological research. However, existing relation extraction methods primarily focus on binary interactions and struggle to model variable-length n-ary drug combinati...

---

### 29. [Agentic Planning with Reasoning for Image Styling via Offline RL](https://arxiv.org/abs/2603.07148)

**Authors**: Subhojyoti Mukherjee, Stefano Petrangeli, Branislav Kveton, Trung Bui, Franck Dernoncourt, Arko Mukherjee  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.07148v1  

#### Abstract
Direct prompt-based editing often fails on complex transformations because vague and subjective prompts often require nuanced understanding of what should be changed in the image. Our core intuition is that leveraging compositional image editing tools rather than direct prompting profits from struct...

---

### 30. [Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation](https://arxiv.org/abs/2603.07233)

**Authors**: Andrea Giuseppe Di Francesco, Andrea Rubbi, Pietro Li\`o  
**Category**: cs.LG  
**Published**: 2026-03-10  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.07233v1  

#### Abstract
Predicting how cells respond to genetic perturbations is fundamental to understanding gene function, disease mechanisms, and therapeutic development. While recent deep learning approaches have shown promise in modeling single-cell perturbation responses, they struggle to generalize across cell types...

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
