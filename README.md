# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-01 07:09:08 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [A Framework for Hybrid Collective Inference in Distributed Sensor Networks](https://arxiv.org/abs/2603.28778)

**Authors**: Andrew Nash, Dirk Pesch, Krishnendu Guha  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2603.28778v1  

#### Abstract
With the ever-increasing range of applications of Internet in Things (IoT) and sensor networks, challenges are emerging in various categories of classification tasks. Applications such as vehicular networking, UAV swarm coordination and cyber-physical systems require global classification over distr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Framework for Hybrid Collective Inference in Distributed Sensor Networks

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**分布式传感器网络**（Distributed Sensor Networks）中的**集体推理**（Collective Inference）任务，解决在资源受限环境下（如通信带宽、能量消耗、计算能力有限）如何高效地实现全局分类或预测的问题。典型应用场景包括：
- UAV swarms（无人机集群）
- IoT 平台
- 智能交通系统（Intelligent Transportation Systems）

这些场景中，传感器节点通常通过高成本的上行链路（uplink）连接到云/边缘服务器，而彼此之间可通过低成本的对等通信（peer-to-peer, P2P）进行数据交换。

### 提出的新方法与新思路
作者提出了一种**混合式集体推理框架**（Hybrid Collective Inference Framework），其核心创新在于：
- **首次将分布式推理**（Distributed Inference）与**云/边缘分层推理**（Cloud/Edge Hierarchical Inference）**动态结合**，形成一个统一的决策机制。
- 引入**动态通信策略**（Dynamic Communication Strategy），每个传感器节点基于本地观测和通信成本，自主决定以下三种动作之一：
  1. **Early Exit**：若本地置信度 $ P(Y_k|s_i) > \lambda $，则直接输出预测；
  2. **Peer Request**：若请求邻居数据的期望成本低于上传至云端的成本，则向邻居请求数据并联合推理；
  3. **Offload to Cloud**：否则将数据上传至云/边缘服务器进行集中推理。

该策略由公式 (1) 控制：
$$
C_{S_iS_j} + C_{S_jE}(1 - P_i) \leq C_{S_iE}
$$
其中 $ P_i = \max E[P(P(Y_k|s_i,s_j) > \lambda)] $ 表示请求后能成功早退的概率。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **通信效率** | 集中式需全部上传，通信开销大；纯分布式可能冗余通信 | 动态选择最优路径，显著降低平均通信成本 |
| **准确性** | 独立分类器准确率低；集中式准确率高但代价大 | 准确率接近集中式方法，远高于独立分类器 |
| **灵活性与适应性** | 多为静态调度或单一模式 | 支持动态运行时决策，适应不同数据分布和网络条件 |

---

## 2. 核心实验方法和设置

### 数据集
论文未使用真实世界数据集，而是基于**合成高斯数据**（Synthetic Gaussian Data）构建实验环境，考虑以下两种分布设定：
1. **Binomial Y, N=2**：两个传感器，隐藏状态 $ Y \in \{0,1\} $，每个传感器的数据 $ S_i \sim \mathcal{N}(\mu_y, \sigma) $
2. **Multinomial Y, N>2**：多个传感器（$ N=4,8 $），多类隐藏状态 $ Y \in \{0,1,\dots,K-1\} $

参数变化包括：
- 类间均值差 $ \delta_\mu $
- 方差 $ \sigma $
- 置信阈值 $ \lambda $
- 通信成本 $ C_{S_iS_j} = 1J $, $ C_{S_iE} \in [1,5]J $

### 实验设置与评估指标
#### 设置
- 时间同步离散步长
- 所有策略在 Python 中实现，使用 SciPy 和 NumPy 进行模拟
- 每组参数重复采样 10,000 次以获得统计平均值

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 全局预测正确率（多数投票） |
| **Avg. Cost (J)** | 平均总通信能耗 |
| **Avg. # Direct Decisions** | 成功早退的次数 |
| **Avg. # Successful Requests** | 成功通过 P2P 请求完成推理的次数 |
| **Cloud Avg. Cost** | 集中式基线成本：$ N \times C_{S_iE} $ |

### 基线方法对比
| 基线名称 | 描述 |
|--------|------|
| **Cloud/Fog Baseline** | 所有传感器上传数据至云端联合推理，成本高但准确率最高 |
| **Independent Classifier Baseline** | 各传感器独立决策，无通信成本，但准确率较低 |
| **Globally Optimal Baseline** | 理论最优划分（通过回溯求解），用于衡量本方法是否存在冗余通信 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table VI 和 VII）

#### 示例：$ N=2, \delta_\mu=5, \sigma=1.5, \lambda=0.95, C_{S_iE}=4J $
| 方法 | Accuracy (%) | Avg. Cost (J) |
|------|-------------|--------------|
| **本文方法** | 97.6% | 0.75 |
| Cloud Baseline | 99.0% | 8.0 |
| Independent | 95.2% | 0.0 |

> ✅ 在仅 **0.75J** 的通信成本下达到接近集中式的精度（相差 <2%），远优于独立分类器。

#### 更复杂场景：$ N=4, Y \in \{0,1,2,3\}, \delta_\mu=2, \lambda=0.85, C_{S_iE}=4J $
| 方法 | Accuracy (%) | Avg. Cost (J) |
|------|-------------|--------------|
| **本文方法** | 80.7% | 4.71 |
| Cloud Baseline | 82.9% | 16.0 |
| Independent | 74.6% | 0.0 |

> ✅ 节省约 **70% 通信成本**，同时提升 **6% 准确率** vs 独立分类器。

### 与基线方法的对比结果
- 在所有测试场景中，本文方法的**准确率始终介于独立分类器与云端基线之间**，且更接近后者。
- **通信成本显著低于云端基线**，尤其当 $ C_{S_iE} $ 较高时优势更大（见 Fig. 7）。
- 当 $ \delta_\mu $ 很小时（类别难分），行为趋近于云端上传；当 $ \delta_\mu $ 很大时，趋近于独立早退，体现自适应性。

### 消融实验与关键观察
#### （1）置信阈值 $ \lambda $ 的影响
- ↑ $ \lambda $ → ↓ 直接决策数 → ↑ 请求与上传数
- 高 $ \lambda $ 下更多依赖 P2P 请求，验证了“动态协作”机制的有效性

#### （2）传感器数量 $ N $ 的扩展性（Fig. 5）
- 成本随 $ N $ 线性增长，但斜率远小于云端方案
- 准确率随 $ N $ 提升趋于饱和（特征冗余），但仍优于独立分类器

#### （3）类数 $ |dom(Y)| $ 增加的影响（Fig. 8）
- 类越多 → 分类越难 → 准确率下降
- 但在 $ \delta_\mu $ 不足时（如 $ \delta_\mu=1 $），仍可维持一定性能优势

#### （4）启发式近似算法验证（Table VIII）
- 使用高效启发式替代均匀采样后，结果误差 < 3.5%
- 证明该方法具备实际部署可行性

---

## 4. 关键结论和发现

### 主要发现
1. **混合推理框架可在保持高准确率的同时大幅降低通信成本**，尤其适用于 $ C_{S_iE} \gg C_{S_iS_j} $ 的场景。
2. **P2P 请求机制在中等可分性条件下最有效**（即 $ \delta_\mu $ 适中），此时既不能完全早退，也不必全部上传。
3. 框架具有良好的**可扩展性**，随着传感器数量增加仍能保持成本优势。
4. **动态决策优于静态策略**，能够根据实时观测调整行为，避免不必要的通信。

### 方法的局限性
- 目前仅在**高斯假设下推导解析解**，对非高斯或复杂分布需依赖近似方法。
- **未考虑计算能耗**，仅建模通信成本；在深度学习模型中，本地推理计算开销不可忽略。
- **未实测真实通信模块能耗**，依赖文献估算（如 [27] 中 NB-IoT vs LoRaWAN）。
- 存在**潜在冗余通信风险**，因缺乏全局协调可能导致多个节点同时请求同一数据。

### 未来工作方向
1. **更精细的成本建模**：
   - 加入计算能耗（尤其是 TinyML、加速器支持）
   - 考虑延迟、带宽、QoS 等多维指标
2. **支持更复杂的分类器**：
   - 将框架扩展至 DNN、GNN、Ensemble Models
   - 探索使用 Gaussian Mixture Models（GMM）逼近任意分布
3. **真实场景验证**：
   - 在真实 IoT/UAV 网络中部署，使用真实空气污染、交通流量等数据
   - 与现有 SOTA 方法（如 CoEdge、Split Learning）对比
4. **引入强化学习优化策略**：
   - 使用 Multi-Agent RL 学习最优通信策略，超越当前基于规则的决策
5. **目标设备 T 的集成**（Section IV-I）：
   - 支持中间聚合节点（如 Fog Node）作为低代价目标，进一步优化层级结构

---

> 🔚 **总结**：本文提出的 Hybrid Collective Inference Framework 是首个将 P2P 协作与 Cloud Offloading 动态融合的推理架构，在理论分析与仿真实验中均展现出卓越的通信效率与准确性平衡，为未来大规模、低功耗 IoT 应用提供了重要技术路径。

</details>

---

### 2. [HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling](https://arxiv.org/abs/2603.29090)

**Authors**: Jaber Jaber, Osama Jaber  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2603.29090v1  

#### Abstract
World models that predict future states from video remain limited by flat latent representations that entangle objects, ignore causal structure, and collapse temporal dynamics into a single scale. We present HCLSM, a world model architecture that operates on three interconnected principles: object-c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **world model**（如 V-JEPA、DreamerV3）依赖于扁平化的 latent 表示，存在三大缺陷：
- **对象纠缠**：无法将场景中的不同物体（如机器人夹爪、杯子）分离表示；
- **时间尺度单一**：难以同时建模连续运动（如轨迹）、离散事件（如碰撞）和长期目标（如任务规划）；
- **缺乏因果结构**：无法推理“如果夹爪推得更用力会怎样？”这类反事实问题。

这些限制使得模型难以支持高级认知功能，如规划与因果推理。

---

### 🚀 提出的新方法与核心思想

HCLSM 提出一种全新的 **object-centric world model 架构**，融合三个关键设计原则：

#### （1）**Object-Centric 分解（Layer 2）**
- 使用 **Slot Attention** 对视频帧进行对象分解；
- 引入 **Spatial Broadcast Decoder (SBD)**，每个 slot 独立重建图像区域，迫使 slot 学习空间专属权属；
- 以 **frozen ViT 特征** 作为重建目标（借鉴 DINOSAUR），提升语义感知能力。

#### （2）**Hierarchical Temporal Dynamics（Layer 3）**
构建三级时间层次结构：
- **Level 0: Selective SSM**  
  处理每帧之间的连续物理动态（毫秒级），高效建模平滑运动；
- **Level 1: Sparse Transformer**  
  在检测到状态跃迁（如接触、滑动）时触发，仅处理稀疏事件序列，节省计算；
- **Level 2: Compressed Transformer**  
  将事件压缩为抽象目标表示，用于高层推理与规划。

#### （3）**Causal Structure Learning（Layer 4）**
- 利用 **GNN** 建模对象间交互，边权重反映影响强度；
- 引入 **NOTEARS 风格的 DAG 正则化器**，鼓励学习无环因果图。

#### （4）**Two-Stage Training Protocol（关键训练策略）**
> “先学会看，再学会预测”

- **Stage 1（前40%训练步数）**：仅启用 SBD 重建损失，强制 slot 进行空间专业化；
- **Stage 2（后60%）**：激活完整 JEPA-style 动态预测损失，基于已分解的对象学习演化规律。

该机制避免了动态预测损失主导导致的 slot 编码退化为分布式表示。

#### （5）工程优化亮点
- 开发 **自定义 Triton 内核** 加速 Selective SSM 的 scan 操作，实现 **38× 速度提升**；
- 支持 GPU-native Sinkhorn 匹配，消除 CPU-GPU 同步瓶颈；
- 实现 chunked GNN 计算，降低高 slot 数下的内存占用。

---

### 🔍 相比现有方法的优势

| 维度 | HCLSM | 典型 baseline（如 V-JEPA、SlotFormer） |
|------|-------|-------------------------------|
| 对象表示 | ✅ 显式 object slots | ❌ 扁平 latent 或静态 slots |
| 时间建模 | ✅ 三层次（连续+事件+目标） | ❌ 单一尺度（通常为 transformer） |
| 因果结构 | ✅ GNN 边权 + DAG 正则 | ❌ 无显式因果建模 |
| 可解释性 | ✅ slot 分工、事件边界、因果边可可视化 | ❌ 黑箱预测 |
| 效率 | ✅ Sparse & Compressed Transformers + Triton kernel | ❌ 全序列 attention 成本高 |

> HCLSM 是首个统一 **object decomposition**, **hierarchical dynamics**, 和 **causal reasoning** 的端到端可微架构（见 Table 1）。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **PushT**：来自 **Open X-Embodiment** 数据生态的一个机器人操作任务；
- 包含 206 个 episode，共约 25,650 帧；
- 场景：机械臂推动一个 T 形块至目标位置；
- 输入：16帧视频片段（224×224），动作向量（2D位移）；

---

### ⚙️ 实验设置
- **模型规模**：HCLSM-Small（68M 参数）
- **硬件平台**：NVIDIA H100 80GB GPU
- **训练配置**：
  - Batch size: 4
  - Optimizer: AdamW
  - LR: 1.5e-4（cosine 调度，2K warmup）
  - Precision: bfloat16 混合精度
  - 总训练步数：50K（Stage 1 占 20K，Stage 2 占 30K）
- **评估方式**：2 次成功运行取平均结果（共启动 4 次，因 bf16 NaN 导致 2 次失败）

---

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| `Prediction Loss` (MSE) | 下一状态 latent 预测误差 |
| `SBD Loss` | Spatial Broadcast Decoder 重建损失 |
| `Tracking Loss` | slot 跨帧一致性损失 |
| `Diversity Loss↓` | 衡量 slot 是否差异化（越低越好） |
| `Speed (sps)` | 每秒处理样本数 |
| `Event Detection` | 是否能识别接触等关键事件 |
| `Slot Visualization` | alpha mask 是否呈现空间分工 |

---

### 🔀 基线对比
- **HCLSM (no SBD)**：消融版本，不使用两阶段训练与 SBD；
- 其他隐含对比系统（通过 Table 1）：
  - V-JEPA / V-JEPA2
  - DreamerV3
  - SlotFormer
  - DINOSAUR
  - Slot SSM

---

## 3. 主要实验结果和性能指标

### 📈 定量结果（Table 3）

| 方法 | Pred. Loss | Track. Loss | Diversity ↓ | SBD Loss | Total Loss | Speed (sps) |
|------|------------|-----------|-------------|----------|------------|--------------|
| HCLSM (no SBD) | **0.002** | 0.001 | 0.154 | — | **0.100** | 2.3 |
| **HCLSM (two-stage)** | 0.008 | **0.016** | **0.132** | **0.008** | 0.262 | **2.9** |

> 💡 关键观察：
- **虽然预测误差更高，但 SBD 损失显著下降且 diversity 更优** → 表明 slot 成功实现了空间专业化；
- 两阶段训练牺牲了一定预测精度，换来了结构化表示能力；
- 推理速度更快，得益于 Triton 加速与模块化设计。

---

### 🔬 消融实验与分析

#### （1）Two-Stage Training 的必要性
- 若从一开始就联合优化重建与预测，slot 注意力趋于均匀分布，无法形成对象专属；
- Stage 1 的纯重建阶段迫使每个 slot 最小化自身区域误差，促使其“占地盘”；
- 图 2 显示：两阶段训练后，alpha masks 出现明显空间分割模式（不同颜色代表不同 slot 主导区域）；

#### （2）Event Detection 能力
- 图 3 显示 event detector 在状态突变时刻（如夹爪触碰 T-block）准确触发；
- 平均每 16 帧检测出 2–3 个事件，符合人类直觉；
- 使用多尺度差分特征 + 因果膨胀卷积实现鲁棒检测。

#### （3）Latent Dynamics 可视化
- 图 4 展示 slot 状态轨迹经 PCA 投影后的演化路径；
- 不同 slot 轨迹路径各异，说明其捕捉了不同的动态成分；
- 轨迹在事件边界处发生转向，体现层级动力学的有效耦合。

#### （4）Triton Kernel 性能优势（Table 2）
| 配置 | Sequential (ms) | Triton (ms) | Speedup |
|------|------------------|------------|---------|
| Tiny | 6.22 | 0.16 | **39.3×** |
| Base | 69.64 | 1.83 | **38.0×** |

> → SSM scan 从正向传播的主要瓶颈降至仅占 5% 时间。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **结构先于预测（Structure Before Prediction）**  
   两阶段训练是实现有效 object-centric 表示的关键——必须先让模型“看清物体”，才能“理解它们如何运动”。

2. **Hierarchical Dynamics 更贴近真实世界的时间组织方式**  
   将连续物理、离散事件与抽象目标分层处理，既提高了效率，也增强了可解释性。

3. **Spatial Broadcast Decoder 是驱动 slot 分工的核心机制**  
   结合 ViT 特征重建，使 slot 能够学习语义上有意义的空间区域，而非随机编码。

4. **GNN 边权重可作隐式因果信号**  
   尽管显式 DAG 学习尚未成功，但 GNN 中的消息传递权重自然反映了对象间的相互作用强度。

5. **工程优化对大规模 object-centric 模型至关重要**  
   自定义 Triton kernel 和 GPU-native 实现大幅提升了训练效率，支撑了实际部署潜力。

---

### ⚠️ 当前局限性
| 问题 | 描述 |
|------|------|
| **Slot 数固定且冗余** | 使用 32 slots 处理仅含 3 个物体的场景，每个物体被多个 slot 分割；existence head 未能有效关闭空闲 slot； |
| **显式因果图学习失败** | NOTEARS 式 DAG 学习在 bf16 下崩溃，所有边权重归零；需更强正则或 fp32 精度支持； |
| **跨 episode 泛化弱** | event detection 在不同 episode 中表现不稳定；未验证是否学到通用因果机制； |
| **模型规模受限** | 仅完成 Small 版本（68M）训练；Base（262M）及以上因 NaN 无法收敛； |
| **种子敏感性强** | ~40–60% 训练运行因梯度溢出失败，尤其在 slot attention GRU 中； |

---

### 🔮 未来工作方向
1. **引入 adaptive slot 机制**  
   如 Adaptive Slot Attention 或 MetaSlot，动态调整 slot 数量，减少冗余。

2. **预训练 ViT 初始化**  
   使用 V-JEPA2 等大型视觉模型初始化 encoder，提供更丰富的 patch 特征作为 SBD 目标。

3. **扩展到复杂多物体场景**  
   在 ALOHA 双手机械臂任务等包含 5+ 物体的数据上验证 object decomposition 的优势。

4. **闭环控制集成**  
   将 HCLSM 与 CEM/MPPI 规划器结合，实现基于 world model 的 real-time robot control。

5. **解决数值稳定性问题**  
   探索 gradient scaling、critical path 使用 fp32、或改进 SSM 初始化方案，突破大模型训练障碍。

6. **因果评估基准建设**  
   构建具有已知因果结构的仿真环境，使用干预测试（intervention-based metrics）定量评估 learned causal graph。

---

## 📎 总结

> **HCLSM 是迈向结构化世界建模的重要一步**：它不再把世界当作像素流来预测，而是尝试像人一样——看见对象、感知事件、理解因果、分层思考。

尽管目前仍处于“基础原型”阶段（作者称“warts and all”开源），但它提供了完整的代码库（8,478 行 Python，51 模块，171 单元测试）、训练流水线与评估工具，为后续研究奠定了坚实基础。

🔗 **项目地址**：[https://github.com/rightnow-ai/hclsm](https://github.com/rightnow-ai/hclsm)

</details>

---

### 3. [Federated Inference for Heterogeneous LLM Communication and Collaboration](https://arxiv.org/abs/2603.28772)

**Authors**: Zihan Chen, Zeshen Li, Howard H. Yang, Tony Q. S. Quek, Jihong Park  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.28772v1  

#### Abstract
Given the limited performance and efficiency of on-device Large Language Models (LLMs), the collaborations between multiple LLMs enable desirable performance enhancements, in which data, tokens, and model weights could be shared across LLMs. This process is constrained by task-oriented QoS demands, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Federated Inference for Heterogeneous LLM Communication and Collaboration》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前**on-device LLMs**（设备端大语言模型）在推理准确性和速度上远不如云端全规模LLM。虽然可以通过将任务完全卸载到云端来提升性能，但这会带来高延迟、隐私泄露和通信开销等问题。此外，多LLM协作面临以下挑战：
- **Latency（延迟）**：传统文本级通信（T2T）需重复预填充（prefill），导致显著延迟；
- **Privacy（隐私）**：原始输入输出token可能暴露用户敏感信息；
- **Heterogeneity（异构性）**：不同架构的LLM之间难以直接共享知识。

现有协作方式如Federated Learning主要用于训练阶段，而本文聚焦于**推理阶段的联邦协同**（federated inference），填补了该领域的空白。

---

### 🚀 提出的新方法：FedRefine
提出一种新型**联邦推理框架——FedRefine**（Federated Refinement），其核心思想是：
> 利用 **Cache-to-Cache (C2C)** 通信机制，在异构LLM间直接交换KV Cache（而非token），实现高效、低延迟、隐私保护的协同推理。

#### 核心组件：
- **SelfRefine + C2C 结合**：借鉴SelfRefine中的自我迭代优化思想，并通过C2C实现跨模型的知识传递。
- **Bidirectional Co-C2C**：支持双向KV缓存通信（Co-C2C），允许每个设备同时作为发送方和接收方，促进公平协作。
- **Model-Agnostic Fuser**：引入预训练的**C2C fuser网络**（如MLP），用于对齐不同架构间的KV Cache表示，突破异构限制。
- **Rephrased Input for Privacy**：使用语义重写后的输入（rephrased input）进行推理，防止意图泄露，实现隐私保护。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（T2T） | FedRefine（C2C） |
|------|------------------|-------------------|
| **延迟** | 高（需重复prefill） | 极低（跳过prefill） |
| **通信内容** | 明文token（易泄密） | KV Cache（更抽象，配合rephrase增强隐私） |
| **兼容性** | 要求同构或强对齐 | 支持异构模型（通过fuser桥接） |
| **协作模式** | 单向为主 | 双向互惠（mutual refinement） |
| **更新成本** | 可能需要权重传输 | 无需模型参数更新 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练集**：OpenHermes2.5 的前50,000个样本（用于训练C2C fuser）
- **测试集**：OpenBookQA（标准开放书本问答数据集，评估推理能力）

---

### ⚙️ 实验设置
- **系统构成**：
  - **Receiver**：Qwen3-0.6B（主推理模型）
  - **Transmitters（4个）**：
    - Qwen2.5-0.5B
    - Qwen2.5-0.5B-code
    - Qwen2.5-1.5B
    - Llama-3.2-1B
- **Fuser设计**：
  - 层层对齐（bottom-up layer alignment）
  - 每层使用三层MLP作为fuser，投影KV Cache
  - 总共部署4个独立fuser（对应4组配对）
- **隐私机制**：
  - Receiver负责将原始query重写为语义等价但形式不同的“rephrased input”
- **通信协议对比**：
  - **T2T**：传输token序列
  - **C2C-KV**：传输KV Cache（含key/value张量）

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 在OpenBookQA上的问答准确率 |
| **Latency** | 平均推理时间（秒） |
| **Communication Load** | 每token传输的数据量（KB/byte） |
| **Scalability** | 随参与模型数量增加的性能变化趋势 |

---

### 🆚 基线方法对比
| 方法 | 类型 | 是否支持异构 | 是否双向 | 是否隐私保护 |
|------|------|---------------|-----------|----------------|
| Standalone Inference | 单模型 | — | — | 是 |
| T2T Collaboration | Token级通信 | 否（依赖token一致性） | 单向为主 | 否（明文传输） |
| C2C (Unidirectional) | KV Cache通信 | 是（有fuser） | 否 | 弱 |
| **FedRefine (Ours)** | **Bidirectional C2C + Rephrase** | **是** | **是** | **是** |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见Fig. 3）

#### (a) 推理准确性（Accuracy）
- **独立模型 baseline**（Qwen3-0.6B alone）：约 **55%**
- **FedRefine（非隐私版，4个sharer）**：**+21.2%** 提升 → 达到 **~66.8%**
- **FedRefine（隐私保护版，rephrased input）**：仅下降 **3%**，仍达 **~63.8%**
- **vs T2T方案**：C2C在满员参与下**高出约15个百分点**

> ✅ 表明：KV Cache通信能有效融合多模型知识，且隐私处理代价小。

---

#### (b) 通信负载（Communication Load）
- **每token T2T传输**：仅 **16 bytes**
- **每token C2C传输**：高达 **88 KB**

> ⚠️ 缺点凸显：C2C通信资源消耗大，适合高带宽边缘环境或选择性启用。

---

#### (c) 推理延迟（Latency）
- **T2T方案**：延迟最高（因反复prefill）
- **C2C（含rephrase）**：尽管query重写带来额外开销，**总延迟仍显著低于T2T**
- **原因**：C2C避免了重复上下文编码，节省大量计算

> ✅ 优势明显：即使加上rephrase时间，C2C整体更快。

---

#### (d) 模型组合影响（Fig. 3b）
- 不同task下选择最优模型组合可进一步提升性能
- 表明：**动态调度机制具有潜力**

---

#### ❌ 消融实验（隐含分析）
虽然未明确列出消融表，但从实验设计中可推断：
- **移除rephrase** → 准确性略升但牺牲隐私
- **禁用fuser或使用单向C2C** → 性能下降，验证bidirectional与model-agnostic的重要性
- **减少sharer数量** → 准确率呈正相关下降，说明知识聚合效应存在

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KV Cache是比token更高效的协作媒介**  
   > C2C通信大幅降低延迟，优于传统T2T范式。

2. **双向Co-C2C支持更公平、可持续的协作生态**  
   > 小模型也能帮助大模型 refine 输出，打破“强者恒强”假设。

3. **异构LLM可通过fuser实现无缝协作**  
   > 不再要求模型结构一致，极大提升部署灵活性。

4. **隐私保护可通过input rephrasing低成本实现**  
   > 仅损失3%精度即可防止意图泄露，性价比高。

5. **FedRefine具备良好扩展性**  
   > 支持N个LLM灵活接入，形成可扩展的联邦推理网络。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **高通信开销** | KV Cache体积远大于token（88KB vs 16B），限制在低带宽场景应用 |
| **Fuser训练成本** | 每对模型需单独训练fuser，N个模型需O(N²)个fuser，维护复杂 |
| **依赖高质量rephraser** | 输入重写质量直接影响协作效果，目前缺乏自动化优化机制 |
| **尚未验证大规模部署** | 当前实验局限于小规模系统，真实边缘网络表现待验证 |

---

### 🔮 未来工作方向（作者建议）
1. **Iterative Local Refinement**  
   > 设计基于cache/token反馈的多轮本地 refine 流程。

2. **Continuous Global Federation Iterations**  
   > 构建全局迭代优化机制，让协作收益持续累积。

3. **Cache Communication for Multi-modal LLMs**  
   > 扩展至视觉-语言等多模态场景，探索跨模态KV Cache对齐。

4. **Prompt Engineering for Federated Inference**  
   > 开发面向隐私保护协作的prompt策略，指导角色分工与交互流程。

5. **Opportunistic Communication Switching**  
   > 动态切换T2T与C2C，依据QoS需求与网络状态自适应调整。

---

## ✅ 总结一句话
> **FedRefine提出了一种基于双向KV Cache通信的联邦推理新范式，在保证隐私的前提下，实现了异构LLM间的高效、低延迟、可扩展协作，为下一代智能边缘网络提供了重要技术路径。**

</details>

---

### 4. [ARCS: Autoregressive Circuit Synthesis with Topology-Aware Graph Attention and Spec Conditioning](https://arxiv.org/abs/2603.29068)

**Authors**: Tushar Dhananjay Pathak  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.29068v1  

#### Abstract
I present ARCS, a system for amortized analog circuit generation that produces complete, SPICE-simulatable designs (topology and component values) in milliseconds rather than the minutes required by search-based methods. A hybrid pipeline combining two learned generators (a graph VAE and a flow-matc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ARCS: Autoregressive Circuit Synthesis with Topology-Aware Graph Attention and Spec Conditioning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统模拟电路设计依赖人工经验，耗时且效率低。现有机器学习方法存在以下三大局限：
- **仅生成拓扑**（如 AnalogGenie）：不输出元件值，需额外通过遗传算法（GA）进行 sizing，耗时数分钟。
- **基于文本生成**（如 CircuitSynth）：在字符级别操作，缺乏电路语义理解，易产生语法错误。
- **缺乏规格条件控制**（spec conditioning）：无法根据目标参数（如 $ V_{out}=5V $）定向生成电路。

ARCS 针对上述问题，提出一个**端到端可微分、快速、高有效性的模拟电路生成系统**。

---

### 🚀 提出的新方法与创新点

#### （1）**Group Relative Policy Optimization (GRPO)**  
- **问题识别**：标准 REINFORCE 在多拓扑训练中因不同拓扑奖励分布差异大（例如电源转换器 reward 可达 8.0，放大器仅 ~4.0），导致“简单拓扑主导梯度更新”，困难拓扑被放弃。
- **解决方案**：引入 per-topology advantage normalization，即每个拓扑组内独立计算优势函数（advantage）：
  $$
  A^{(i)} = \frac{R_i - \mu_T}{\sigma_T}
  $$
  保证所有拓扑都能获得有意义的梯度信号。
- **效果**：仅用 500 步 RL 训练（相比 REINFORCE 的 5000 步），**simulation validity 提升 +9.6 个百分点**。

#### （2）**Grammar-Constrained Decoding**  
- 引入基于状态机的 token masking 机制，在自回归解码每一步强制结构合法性。
- 定义三级约束：
  - `GRAMMAR`：组件-值交替模式
  - `TOPOLOGY`：限制为当前拓扑所需的组件类型
  - `FULL`：进一步限制物理合理的取值范围
- **结果**：无需 RL 或后处理即可实现 **100% 结构有效性（structural validity）**。

#### （3）**Hybrid Multi-Source Ranking Pipeline**  
- 融合两个互补生成器：
  - **VCG**（graph VAE）
  - **CCFM**（Constrained Circuit Flow Matching）
- 通过 SPICE 仿真选出最优候选。
- **结果**：仅需 **8 次 SPICE 评估** 即达到 **99.9% simulation validity 和 reward 6.43/8.0**，比 GA 少 40× 仿真次数。

#### （4）**Topology-Aware Graph Transformer 架构**
- 注入电路图结构先验：
  - **Topology-aware attention bias**：连接节点间注意力增强
  - **Random-Walk Positional Encoding (RWPE)**：编码组件在网络中的结构性角色（如开关节点 vs 负载电阻）
- 显著提升对复杂信号电路（滤波器、振荡器）的建模能力。

---

### 🔍 相比现有方法的优势

| 方法 | 是否含元件值 | 支持 spec 条件 | 结构有效性 | 仿真有效性 | 推理速度 |
|------|----------------|------------------|--------------|----------------|------------|
| AnalogGenie | ❌ | ❌ | 93.2% | N/A | ~1s |
| CircuitSynth | ✅（字符级） | ❌ | N/A | N/A | ~10s |
| GA / Random Search | ✅ | ✅ | ✅ | ~80% | 数十秒至数百秒 |
| **ARCS (Best-of-3)** | ✅ | ✅ | **100%** | **85%** | **97ms** |
| **ARCS (Hybrid)** | ✅ | ✅ | **100%** | **99.9%** | ~几秒（含 SPICE） |

> ✅ **核心优势总结**：
> - **>1000× 速度快于搜索方法**
> - **单次前向传播生成完整 SPICE 可仿真的网表**
> - **支持规格条件控制，可用于快速原型设计与设计空间探索**

---

## 2. 核心实验方法和设置

### 📚 数据集
- **自动化生成 62,000 个电路样本**，覆盖 **34 种拓扑模板**（32 用于主实验）
  - Tier 1：7 种电源转换器（Buck, Boost 等）
  - Tier 2：9 种信号电路（放大器、滤波器、振荡器）
  - Tier 2b：18 种扩展电路（BJT 放大器、稳压器等）
- 使用 **ngspice** 进行瞬态/AC 仿真，自动提取性能指标（效率、增益、带宽等）
- 经过 **5× component order shuffling** 扩增至约 205,000 序列

---

### ⚙️ 实验设置与评估指标

#### 评估协议
- 测试集：**160 个带规格的目标电路**（每种拓扑 10 个）
- 模型输入：`[START][TOPO][SEP][SPEC_VIN][VAL_x][...]`

#### 主要评估指标
| 指标 | 描述 |
|------|------|
| **Structural Validity** | 生成序列是否符合语法规范（能否正确解析成电路） |
| **Sim. Success** | SPICE 是否收敛 |
| **Sim. Validity** | 仿真结果是否物理合理（如非负效率、合理输出） |
| **Reward** | 综合评分（满分 8.0），包含准确性、效率、质量等 |
| **Wall Time** | 端到端时间（含仿真） |

---

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **Random Search (RS)** | 在参数范围内随机采样 200 组合，选最佳（200 次 SPICE） |
| **Genetic Algorithm (GA)** | BLX-α 交叉 + 高斯变异，种群 30，迭代 20 代（~630 次评估） |
| **Supervised Only (SL)** | 仅监督预训练模型 |
| **REINFORCE** | 标准策略梯度强化学习微调 |

> 💡 注意：ARCS 是 single-shot 生成，而 GA/RS 是 iterative search，因此比较的是“单位计算成本下的性能”。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table II & IV）

| 方法 | Params | Struct % | Sim Valid % | Reward (max 8.0) | Wall Time |
|------|--------|----------|-------------|------------------|-----------|
| Random Search | – | – | 81.2% | 7.28 | 58.8s |
| Genetic Algorithm | – | – | 80.0% | 7.48 | 271s |
| Baseline GPT (SL) | 6.5M | 86.0% | 40.1% | 3.37 | ~20ms |
| Graph Transformer + GRPO (500 steps) | 6.8M | 96.6% | **53.1%** | **4.15** | ~20ms |
| **Best-of-3 (ARCS)** | 6.8M | 100% | **85.0%** | **5.48** | **97ms** |
| **Hybrid (VCG+CCFM)** | – | 100% | **99.9%** | **6.43** | ~几秒（8 SPICE evals） |

> ✅ **亮点结果**：
> - **Best-of-3 ARCS 推理仅需 97ms，比 RS 快 600×，比 GA 快 2800×**
> - **Hybrid 方法以 40× 更少 SPICE 评估达到接近 GA 的 reward（6.43 vs 7.48）**

---

### 🔍 消融实验结果（Table V & VII）

#### 表 V：消融分析（160 samples）
| 变体 | Struct % | Sim Valid % | Reward |
|------|----------|-------------|--------|
| ARCS + RL (full) | 98.1% | 52.5% | 3.49 |
| No RL (supervised only) | 90.6% | 46.9% | 3.24 |
| No spec conditioning | 58.8% | 38.8% | 2.60 |
| Tier 1 only (7 topos) | 100% | 20.6% | 3.77 |

> 发现：
> - **Spec conditioning 至关重要**：移除后 reward 下降 -25.5%
> - **RL 微调带来小幅提升**（+0.25 reward, +5.6pp sim-valid）
> - **扩大拓扑数量会降低平均质量但提升实用性**

#### 表 VII：Grammar-Constrained Decoding（随机初始化模型测试）
| Level | Struct % | Comp Correct % | Time |
|-------|----------|----------------|------|
| NONE | 0.0% | 0.0% | 269ms |
| GRAMMAR | 100% | 0.0% | 125ms |
| TOPOLOGY | 100% | 100% | 27ms |
| FULL | 100% | 100% | 25ms |

> 发现：
> - **约束解码本身就能保证 100% 结构有效性**
> - **反而更快**：因为避免无效长序列生成

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **GRPO 是多拓扑 RL 成功的关键**  
   - 标准 REINFORCE 因跨拓扑 reward 分布不均而失效
   - GRPO 通过 per-topology advantage normalization 解决该问题，**仅 500 步即超越 5000 步 REINFORCE**

2. **Amortized inference 具有巨大实用价值**  
   - 虽然单设计质量不及 GA（5.48 vs 7.48），但其 **>1000× 速度优势**使其成为理想工具：
     - 快速原型设计
     - 设计空间探索
     - **作为 GA 的 warm-start 初始化器**

3. **Warm-start 实验验证互补性**  
   - 使用 ARCS 生成 3 个候选作为 GA 初始种群
   - 结果：达到冷启动 GA **96.6% 的性能，节省 49% 仿真次数和 58% 时间**

4. **Learned Reward Model 可改进 Best-of-N 排序**  
   - 模型置信度（log-prob）与 SPICE reward 不完全对齐
   - 使用轻量级 reward model（666K 参数）重新排序，**在 N=3 时提升 +2.1% reward**

---

### ⚠️ 局限性

| 问题 | 描述 |
|------|------|
| **Per-design quality 不及搜索方法** | Best-of-3 reward 5.48 < GA 的 7.48，仍有差距 |
| **Value tokenizer 精度有限** | 500 个 bin 导致 ~3.5% 相对误差，可能不适合高频 RF 或精密模拟电路 |
| **拓扑仍需预定义** | 不支持拓扑发现，仅在 32 个模板上进行 component sizing |
| **Simulation validity 依赖学习值** | 虽然结构 100% 合法，但电气行为仍取决于模型预测精度 |

---

### 🔮 未来工作方向

1. **模型扩展**：将模型规模扩大至 50–100M 参数，并增加更多训练数据
2. **延长 GRPO 训练 + Curriculum Learning**：按难度逐步训练复杂拓扑
3. **End-to-end CCFM fine-tuning**：解冻 VCG 编码器并联合优化 flow matching 与 SPICE reward
4. **扩展约束解码**：加入基尔霍夫定律等电气约束，提升 simulation validity
5. **Multi-fidelity training**：结合快速解析模型与全 SPICE 仿真进行 RL

---

## 总结

> **ARCS 建立了 amortized analog circuit generation 的新范式**：
> - 通过 **GRPO** 解决多拓扑 RL 中的 reward 分布失配问题；
> - 通过 **grammar-constrained decoding** 实现 100% 结构合法性；
> - 通过 **hybrid ranking** 实现接近 GA 的性能，但使用 40× 更少 SPICE 评估；
> - 通过 **fast inference + Best-of-N** 实现 85% 仿真有效性，推理时间不足 0.1 秒。

尽管尚未完全取代搜索方法，但 ARCS 凭借其**极高的吞吐率和良好的初始设计质量**，已成为**快速原型、设计探索和 warm-start 优化的理想工具**，推动模拟电路设计进入“生成即可用”时代。

</details>

---

### 5. [Meteorology-Driven GPT4AP: A Multi-Task Forecasting LLM for Atmospheric Air Pollution in Data-Scarce Settings](https://arxiv.org/abs/2603.29974)

**Authors**: Prasanjit Dey, Soumyabrata Dev, Bianca Schoen-Phelan  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.29974v1  

#### Abstract
Accurate forecasting of air pollution is important for environmental monitoring and policy support, yet data-driven models often suffer from limited generalization in regions with sparse observations. This paper presents Meteorology-Driven GPT for Air Pollution (GPT4AP), a parameter-efficient multi-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Meteorology-Driven GPT4AP: A Multi-Task Forecasting LLM for Atmospheric Air Pollution in Data-Scarce Settings*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于深度学习的空气污染预测模型通常依赖大量标注数据，在**数据稀缺（data-scarce）** 和**跨区域迁移（cross-domain）** 场景下泛化能力差。这限制了其在监测基础设施薄弱地区（如新兴城市、偏远地区）的应用。

### 提出的新方法与新思路
本文提出了 **GPT4AP (Meteorology-Driven GPT for Air Pollution)**，一个专为大气污染多任务预测设计的参数高效型大语言模型框架，具有以下核心创新：

- ✅ **首个将预训练 LLM 应用于气象驱动的空气污染多任务预测的工作**  
  利用 GPT-2 的强大时序建模能力，结合气象与污染物变量进行联合预测。
  
- ✅ **提出 Gaussian rank-stabilized low-rank adaptation (rsLoRA)**  
  在标准 LoRA 基础上引入高斯初始化与秩相关的缩放机制（$\beta_r = \sigma / \sqrt{r}$），提升低秩适配在不同时间尺度下的稳定性与收敛性。

- ✅ **冻结主干 + 轻量模块微调的架构设计**  
  冻结 GPT-2 的 self-attention 和 feed-forward 层，仅训练轻量级组件：
  - 可学习的位置编码（positional encoding）
  - 输出预测头（prediction head）
  并通过 rsLoRA 进行参数高效适配，显著减少可训练参数数量（< 0.31%）。

### 相比现有方法的优势
| 维度 | GPT4AP 优势 |
|------|-------------|
| **数据效率** | 在仅使用 10% 数据的 few-shot 设置下表现最优，适合低资源场景 |
| **跨站迁移能力** | 在 zero-shot cross-station transfer 中大幅优于所有 baseline |
| **统一框架支持多种任务** | 支持 few-shot、zero-shot、long-term 多种预测模式，无需修改结构 |
| **参数效率高** | 使用 rsLoRA 显著降低训练成本，适合边缘部署 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- 来自中国六个空气质量监测站点（共 6 个）：
  - Aoti Zhongxin (AZ)
  - Dongsi (DS)
  - Shunyicheng (SY)
  - Tiantan (TT)
  - Haidian Wanliu (HW)
  - Wanshou Xigong (WX)
- 时间范围：2013年3月1日 至 2017年2月28日（约 4 年）
- 数据频率：每小时一次
- 输入变量：
  - **6 种污染物**：PM2.5, PM10, SO₂, NO₂, CO, O₃
  - **4 种气象因素**：温度、气压、露点、降雨
- 划分方式：前三年训练，最后一年测试

### 实验设置与评估指标
| 项目 | 描述 |
|------|------|
| **Lookback Window** | 固定为 $T=36$ 小时 |
| **预测目标** | 所有站点未来 6 步（24–60 小时）的 PM2.5 浓度 |
| **三种评估范式** | <ul><li>**Few-shot**：仅用 10% 训练数据</li><li>**Long-term**：使用完整训练数据</li><li>**Zero-shot**：在一个源站点训练，在其他目标站点直接测试，无 fine-tuning</li></ul> |
| **评估指标** | **MSE**（均方误差）、**MAE**（平均绝对误差） |
| **Patch Size** | $P=24$，滑动窗口分块输入序列 |
| **rsLoRA Rank** | 默认 $r=32$，并在消融实验中对比 $r \in \{4,8,16,32,64\}$ |

### 基线方法对比
对比了六种先进的 time-series 预测模型：
- **DLinear**
- **ETSformer**
- **FiLM**
- **Informer**
- **Pyraformer**
- **Standard Transformer**

所有模型在相同数据划分、特征工程和超参条件下进行公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### （1）Few-shot Setting（10% 数据训练）
| 模型 | Average MSE | Average MAE |
|-------|--------------|---------------|
| **GPT4AP (Ours)** | **0.686** | **0.442** |
| DLinear | 0.728 | 0.530 |
| ETSformer | 0.734 | 0.505 |
| FiLM | 0.758 | 0.479 |

✅ **GPT4AP 在所有站点和多数时间跨度上取得最佳性能**，平均 MSE 下降 7.2%（vs DLinear），MAE 下降达 19.5%。

#### （2）Long-term Setting（全量数据训练）
| 模型 | Average MSE | Average MAE |
|-------|--------------|---------------|
| **Informer** | **0.598** | **0.416** |
| Pyraformer | 0.605 | 0.425 |
| **GPT4AP (Ours)** | 0.665 | **0.429** |

🔹 GPT4AP 表现**具有竞争力但不占优**，略逊于专门为长序列优化的 Informer 和 Pyraformer，但在多个站点保持稳定输出。

#### （3）Zero-shot Cross-station Transfer
| 模型 | Average MSE | Average MAE |
|-------|--------------|---------------|
| **GPT4AP (Ours)** | **0.529** | **0.403** |
| Pyraformer | 0.565 | 0.557 |
| DLinear | 0.674 | 0.603 |
| Transformer | 0.644 | 0.592 |

✅ **GPT4AP 在 5/6 个转移方向中排名第一**，平均 MSE 比第二名 Pyraformer 低 6.4%，展现出极强的跨域泛化能力。

---

### 消融实验结果（Ablation Study）

研究了 rsLoRA 的秩 $r$ 对性能的影响：

| Rank $r$ | Trainable Params (%) | Avg MSE (Few-shot) | Avg MSE (Zero-shot) |
|----------|------------------------|---------------------|----------------------|
| 4        | 0.038%                 | 0.722               | 0.552                |
| 8        | 0.077%                 | 0.701               | 0.540                |
| 16       | 0.154%                 | 0.692               | 0.536                |
| **32**   | **0.309%**             | **0.686**           | **0.529**            |
| 64       | 0.617%                 | 0.686               | 0.533                |

📌 发现：
- 性能在 $r=32$ 达到饱和，继续增加秩几乎不再提升效果；
- 参数量随 $r$ 几乎线性增长，从 $r=32$ 到 $r=64$ 参数翻倍（+99.7%），但性能无增益甚至轻微下降；
- **$r=32$ 是精度与效率的最佳平衡点**，达到峰值性能的 99.9%，仅需 0.309% 可训练参数。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **GPT4AP 在数据稀缺和跨域场景下显著优于现有方法**  
   其基于预训练 LLM 的先验知识和参数高效适配机制，使其能有效捕捉气象-污染之间的复杂关系，即使在极少量数据或未见站点也能做出可靠预测。

2. ✅ **rsLoRA 是一种高效的环境时间序列迁移学习策略**  
   引入的高斯初始化与秩归一化缩放提升了训练稳定性，尤其适用于多尺度、异构的时间序列任务。

3. ✅ **统一架构支持多任务预测，具备良好鲁棒性和可扩展性**  
   同一模型可在 few-shot、zero-shot 和 long-term 设置下运行，无需结构调整，便于实际部署。

4. 🔹 **在数据充足时，专用 time-series 架构仍具优势**  
   如 Informer 在 long-term setting 中表现更优，说明 GPT4AP 更适合“小样本 + 强泛化”而非“大数据 + 精细拟合”的场景。

### 方法的局限性
- ❌ 当前模型未超越 specialized time-series models 在 full-data 场景下的性能；
- ❌ 实验仅在中国境内 6 个站点开展，地理多样性有限，泛化能力有待全球验证；
- ❌ 仅预测单一污染物（PM2.5），尚未实现 multi-pollutant joint forecasting；
- ❌ 缺乏不确定性估计（uncertainty quantification）能力。

### 未来工作方向
- 🔄 探索将显式的时序归纳偏置（temporal inductive bias）融入 LLM 框架；
- 🌍 扩展至更多国家和地区，增强地理泛化能力；
- 🧪 开展 multi-pollutant 联合预测与不确定性建模；
- 💡 构建 hybrid 架构，融合物理约束与数据驱动模型以提升解释性与可靠性；
- ☁️ 推进边缘计算部署，利用其参数高效特性实现实时、低成本空气质量预警系统。

--- 

> **总结一句话**：  
> GPT4AP 成功展示了 **foundation model + parameter-efficient tuning** 在环境科学中的巨大潜力，特别是在数据稀缺和跨区域部署的关键挑战中提供了高效、稳健且可扩展的解决方案。

</details>

---

### 6. [AgentFixer: From Failure Detection to Fix Recommendations in LLM Agentic Systems](https://arxiv.org/abs/2603.29848)

**Authors**: Hadar Mulian, Sergey Zeltyn, Ido Levy, Liane Galanti, Avi Yaeli, Segev Shlomov  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.29848v1  

#### Abstract
We introduce a comprehensive validation framework for LLM-based agentic systems that provides systematic diagnosis and improvement of reliability failures. The framework includes fifteen failure-detection tools and two root-cause analysis modules that jointly uncover weaknesses across input handling...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*AgentFixer: From Failure Detection to Fix Recommendations in LLM Agentic Systems*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对 **LLM-based agentic systems** 在生产部署中面临的可靠性挑战，尤其是由 **输出格式错误、输入-提示不一致、计划-行动错位** 等导致的执行中断问题。作者指出，在实际系统中，**解析相关故障（parsing-related incidents）占任务失败的近38%**，成为最主要的执行瓶颈。

现有做法如正则过滤、后期 schema 强制等多为临时补丁，缺乏系统性，难以泛化且易引入技术债。

### 提出了什么新方法或新思路
提出 **AgentFixer** —— 一个端到端的 **agentic system validation framework**，具备以下核心创新：

- **综合性验证工具集**：构建了包含 **15个 failure-detection tools** 和 **2个 root-cause analysis modules** 的完整诊断体系，覆盖从 prompt 设计、输入处理到输出生成的全链路。
- **混合验证策略**：结合 **轻量级规则检查（rule-based）** 与 **LLM-as-a-Judge（LaaJ）语义评估**，兼顾效率与语义深度。
- **跨阶段一致性检测**：不仅检查单阶段合规性，还关注 **prompt-input-output 之间的一致性** 及 **agent handoff 中的状态连贯性**。
- **从诊断到修复的闭环**：框架不仅能发现问题，还能通过 LLM 自省机制生成 **可操作的改进建议（fix recommendations）**，推动系统自我优化。
- **交互式诊断分析**：将诊断结果反馈给 LLM 进行 **self-reflection and prioritization**，实现对话驱动的质量改进流程。

### 相比现有方法的优势
| 维度 | 现有方法局限 | AgentFixer 改进 |
|------|---------------|----------------|
| 范围 | 多聚焦于单次调用或语法层面 | 覆盖 multi-agent workflow 全生命周期 |
| 架构依赖 | 多绑定特定框架（如 LangSmith） | 架构无关（architecture-agnostic），支持跨平台部署 |
| 验证方式 | 多为静态 schema 或后置修复 | 动态集成 deterministic + semantic checks |
| 输出价值 | 仅提供“发生了什么” | 提供“为什么发生”+“如何修复”的 actionable insights |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **AppWorld Benchmark**：包含 24 个任务模板，用于测试 agentic system 在复杂 web 操作中的表现。
- **WebArena Benchmark**：特别是其 GitLab 子集（204 samples），用于评估模型在真实网页环境下的准确性和鲁棒性。

### 实验设置和评估指标
#### 模型配置
使用三种不同规模的 LLM 作为 agent 引擎：
- `gpt-4o`（frontier model）
- `mistral-medium-2505`（mid-sized open-source）
- `llama-4-mav-17b-128e-instruct`（mid-sized open-source）

共记录 **1,940 次 LLM 调用日志**，用于工具分析。

#### 评估指标
- **pass@3 / pass@avg**：衡量任务完成率（WebArena）
- **Issue detection rate (%)**：各工具检测出的问题频率
- **Root cause accuracy**：根因定位准确性（人工验证）
- **Regression rate**：改进后原有成功案例是否退化

#### 工具分类与应用
| 类别 | 工具数量 | 实现方式 |
|------|--------|----------|
| Prompt 分析 | 4 | LLM-as-a-Judge |
| 输入验证 | 3 | LLM-as-a-Judge |
| 输出验证 | 3 | LLM-as-a-Judge |
| Token 异常检测 | 2 | Rule-based |
| Python 代码验证 | 1 | Rule-based (AST) |
| 跨阶段一致性 | 2 | Rule/LaaJ 混合 |

此外还包括两个 **root cause analysis tools**：
- **LLM-RC**：直接分析原始 LLM 输出
- **T-RC**：聚合多个 validation tool 结果进行元分析

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在 AppWorld 上的任务成功率
| Model | Tasks Solved | Rate |
|-------|-------------|------|
| gpt-4o | 14 / 24 | 58.3% |
| mistral-medium | 10 / 24 | 41.7% |
| llama-4-mav | 8 / 24 | 33.3% |

> 注：尽管成功率不高，但暴露大量潜在问题。

#### 各类问题检出频率（图1）
- **Schema non-compliance** 和 **format violations** 是最常见问题
- **Edge case handling** 缺失普遍（尤其在 planning agents）
- **API Planner** 成为问题高发 agent
- **Python syntax error** 在 GPT-4o 中显著低于其他模型

#### Root Cause Analysis 表现
- **LLM-RC** 成功识别出 “缺少必要工具访问权限” 导致任务失败
- **T-RC** 准确定位 “JSON 字段名错误（'applications' vs 'available_apps'）” 引发 schema violation
- 两种工具均能有效减少百兆级日志的手动排查成本

#### Trace Comparison Tool 对比结果（表3）
对 26 对成功/失败轨迹进行分析：
- **46.2%（12/26）** 场景下，**comparison tool 提供更优解释**
- 剩余场景两者质量相当，无一例单轨迹工具更优
- 示例显示 comparison tool 更能精准定位如 “rounding logic 错误” 等深层问题

### 与基线方法的对比结果
未直接对比传统 observability 工具（如 LangSmith），但从设计上体现优势：
- 不依赖特定框架（OTel 兼容）
- 支持 **pre-execution validation** 而非仅事后 trace
- 提供 **repair guidance** 而不仅是 error logging

### 消融实验结果（隐含在 prompt refinement 中）

#### Prompt Refinement 效果（表2）
| Model | Prompt Version | pass@3 | Δ |
|-------|----------------|--------|----|
| GPT-4o | Original | 47% | - |
| → | Plan+QA+Action | 50% | +3pp |
| → | +Decomposition | 52% | +5pp |
| LLaMA-4 | Original | 38% | - |
| → | Full refinement | 46% | +8pp |
| Mistral | Original | 35% | - |
| → | Full refinement | 42% | +7pp |

> 小模型提升幅度更大，接近 frontier model 表现。

#### 任务级改进分解（表4）
| Model | Preserved Success | Improved | Regressed |
|-------|--------------------|---------|----------|
| GPT-4o | 93 | 10 | 1 |
| LLaMA-4 | 72 | 12 | 4 |
| Mistral | 74 | 8 | 2 |

> 显示改进具有稳定性，回归极少。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Parsing-related failures 是生产系统中最主要的失效源**（占比 ~38%），即使小格式偏差也会引发级联故障。
2. **Mid-sized models 完全可以通过 validation-driven refinement 达到接近 frontier model 的可靠性水平**，无需一味追求大模型。
3. **Prompt engineering 的系统性优化（如 schema anchoring、few-shot alignment）可显著提升整体鲁棒性**。
4. **Trace comparison（成功 vs 失败路径）是一种强大的根因定位手段**，优于单一失败路径分析。
5. **Validation 本身可以变得“agentic”**：通过 LLM 自省机制，将诊断报告转化为优先级建议，形成持续改进闭环。

### 方法的局限性
- 当前工具集仍需人工定义规则和 prompt，自动化程度有限。
- LLM-as-a-Judge 存在 **position bias、calibration 问题**，可能影响判断一致性。
- 对超大规模 multi-agent system 的扩展性尚未充分验证。
- 所有实验基于 IBM CUGA 架构，通用性有待在更多框架中验证。

### 未来工作方向
- 将 validation loop 集成进训练/微调过程，实现 **training-time resilience**。
- 开发 **auto-prompt refinement agents**，自动迭代 prompt 内容。
- 探索 **real-time enforcement mechanisms**，在运行时动态拦截非法输出。
- 构建统一的 **agentic failure taxonomy & benchmark**，推动标准化评估。
- 推动 **AgentOps** 成为独立工程学科，融合 DevOps、MLOps 与 agentic 特性。

---

> ✅ **一句话总结**：  
> *AgentFixer 展示了一条通往可靠、低成本、可解释 agentic systems 的工程路径——不是靠更大的模型，而是靠更智能的验证、诊断与自省机制。*

</details>

---

### 7. [Stochastic Dimension Implicit Functional Projections for Exact Integral Conservation in High-Dimensional PINNs](https://arxiv.org/abs/2603.29237)

**Authors**: Zhangyong Liang  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.29237v1  

#### Abstract
Enforcing exact macroscopic conservation laws, such as mass and energy, in neural partial differential equation (PDE) solvers is computationally challenging in high dimensions. Traditional discrete projections rely on deterministic quadrature that scales poorly and restricts mesh-free formulations l...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Stochastic Dimension Implicit Functional Projections for Exact Integral Conservation in High-Dimensional PINNs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在高维物理信息神经网络（Physics-Informed Neural Networks, PINNs）中，**精确满足宏观守恒律**（如质量、能量守恒）是一个长期存在的挑战。传统方法面临以下瓶颈：

- **显式离散投影依赖固定网格**：如 PINN-proj 需要基于均匀网格的确定性数值积分（如 Riemann 和），这破坏了 PINNs 的“无网格”（mesh-free）特性，并引发维度灾难（curse of dimensionality）。
- **随机采样导致过拟合与不稳定性**：若用 Monte Carlo (MC) 随机点替代网格，强制每个小批量满足全局守恒会导致模型过度拟合采样方差，产生高频虚假振荡。
- **非凸约束缺乏收敛保证**：能量守恒对应的是二次型积分约束（如 $\int u^2 dx = C$），其可行域为超椭球面，属于**非凸流形**，通用隐式优化层（如 Inet）无法保证收敛。
- **高阶微分算子带来内存爆炸**：反向模式自动微分（reverse-mode AD）在高维空间计算高阶导数时内存开销巨大。

---

### 提出的新方法：SDIFP 框架

作者提出 **Stochastic Dimension Implicit Functional Projection (SDIFP)** 框架，核心思想是：

> 将约束投影从“对离散空间输出向量的欧氏空间投影”转变为“对连续网络输出的仿射函数空间投影”。

具体实现如下：

#### ✅ 创新点 1：**连续函数空间中的仿射变换投影**
定义最终解为原始网络输出的全局仿射变换：
$$
u(x,t) = \alpha(t) \cdot u_{\text{raw}}(x,t;\theta) + \beta(t)
$$
其中 $\alpha(t), \beta(t)$ 是动态确定的标量参数。该变换将无限维非凸投影问题简化为二维代数求根系统。

#### ✅ 创新点 2：**分离式蒙特卡洛积分（Detached MC Quadrature）**
- 使用大规模低差异序列（如 Sobol’ 序列）进行前向积分估计（如均值、方差）：
  $$
  \mu_1 = \frac{1}{M}\sum_{x_i \in S_{MC}} u_{\text{raw}}(x_i),\quad \mu_2 = \frac{1}{M}\sum u_{\text{raw}}^2(x_i)
  $$
- 关键在于这些积分**脱离自动微分图（detached from AD graph）**，避免反向传播时的内存爆炸。

#### ✅ 创新点 3：**闭式解析求解非凸约束**
通过代数推导得到 $\alpha^*, \beta^*$ 的**闭式解**：
$$
\alpha^* = \sqrt{\frac{c_2 - c_1^2}{\mu_2 - \mu_1^2}},\quad \beta^* = c_1 - \alpha^*\mu_1
$$
此解严格满足线性（质量）和二次（能量）守恒，且规避了非凸迭代求解的发散风险。

#### ✅ 创新点 4：**双随机无偏梯度估计器（DS-UGE）**
为支持可微训练，设计了一种新型梯度估计机制：
- **空间小批量采样**用于估计局部 PDE 残差梯度；
- **维度子集采样**用于高效计算高维线性算子梯度（借鉴 SDGD 思想）；
- 两者正交解耦，显著降低内存复杂度至 $O(N \times |Z|)$ 而非常规的 $O(M \times N_c)$。

---

### 相比现有方法的优势

| 维度 | SDIFP | 显式投影（如 PINN-proj） | 软约束（PINN-SC） | 隐式优化层（如 Inet） |
|------|-------|--------------------------|--------------------|------------------------|
| 是否 mesh-free | ✅ 是 | ❌ 否（依赖网格） | ✅ 是 | ✅ 是 |
| 支持非凸约束 | ✅ 是（有闭式解） | ✅ 是（但仅限均匀权重） | ❌ 否（软约束） | ❌ 否（需凸性） |
| 内存效率 | ✅ 高（DS-UGE） | ❌ 极高（全连接系统反传） | ✅ 中等 | ❌ 高（需解线性系统） |
| 守恒精度 | ✅ 数值机器精度级 | ✅（仅在固定网格下） | ❌ 近似 | ⭕ 取决于收敛性 |
| 推理效率 | ✅ $O(1)$ 点级推理 | ✅ | ✅ | ⭕ |

---

## 2. 核心实验方法和设置

### 使用的 PDE 方程族（作为“数据集”）
实验覆盖四类典型守恒系统，在 1D–1000D 上测试：

| PDE 类型 | 守恒量 | 初始条件与边界 |
|---------|--------|----------------|
| **Advection Equation** | 线性积分 $c_1 = \int u dx$ | 高斯脉冲，Neumann 边界 |
| **Reaction-Diffusion Equation** | $c_1$, $c_2 = \int u^2 dx$ | 扩散+源项，Neumann |
| **Wave Equation** | 动量 $c_1$，能量 $c_2 = \int (\partial_t u)^2 + c^2(\partial_x u)^2 dx$ | 波动方程，Neumann |
| **Korteweg-de Vries (KdV) Equation** | $c_1$, $c_2 = \int u^2 dx$ | 孤立波，Neumann |

所有实验均考虑周期或 Neumann 边界以确保封闭系统的守恒性。

---

### 实验设置

- **网络结构**：4 层 MLP，每层 128 单元；
- **优化器**：Adam，初始学习率 $10^{-3}$，线性衰减至 0，共 10,000 epoch；
- **采样策略**：
  - 固定网格（Fixed Grid） vs. 随机 MC 采样（Random Collocation）
  - SDIFP 使用 $M=10^5$ detached MC 点，$N=100$ 残差点，$|Z|=100$ 维度采样子集；
- **评估指标**：
  - **相对 $L^2$ 误差**：$\|u_{\text{pred}} - u_{\text{true}}\| / \|u_{\text{true}}\|$
  - **绝对守恒误差**：
    - 动量误差：$|\hat{c}_1(t) - c_1(t)|$
    - 能量误差：$|\hat{c}_2(t) - c_2(t)|$

---

### 基线方法对比

| 方法 | 简称 | 特点 |
|------|------|------|
| Vanilla PINN | — | 无任何守恒机制 |
| PINN with Soft Constraints | PINN-SC | 加惩罚项 $\lambda \|c(t) - c_{\text{pred}}(t)\|^2$ |
| PINN-proj | PINN-Proj | 显式离散投影（基于 Riemann 求和） |
| KKT-hPINN | PINN-KTT | 基于 KKT 条件的硬约束，仅适用于线性约束 |
| SDIFP（本文） | — | 本文提出的框架 |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Table 5.1）

| 维度 | 方程 | 方法 | 动量误差 ($c_1$) | 能量误差 ($c_2$) |
|-----|------|------|------------------|------------------|
| 1D | KdV | SDIFP (Random) | $6.94\times10^{-6}$ | $3.62\times10^{-6}$ |
| 1D | KdV | PINN-Proj (Random) | $1.79$ | $0.906$ |
| 3D | Reaction-Diffusion | SDIFP (Random) | $2.57\times10^{-5}$ | $2.33\times10^{-5}$ |
| 3D | Reaction-Diffusion | PINN-SC (Random) | $2.22$ | $1.40$ |
| 1000D | Wave | SDIFP (Random) | ~$10^{-4}$ | ~$10^{-4}$ |
| 1000D | Wave | PINN-SC | ~$1$ | ~$1$ |

> **SDIFP 在所有维度和方程上，守恒误差比基线低 3–7 个数量级。**

---

### ✅ 与基线方法的对比结果

#### （1）在固定网格下：
- SDIFP 与 PINN-proj 表现相当，均能实现高精度守恒（误差达 $10^{-6} \sim 10^{-7}$）；
- 但 SDIFP 不依赖网格结构，更具泛化能力。

#### （2）在随机采样下：
- **PINN-proj 失效**：因权重不均，无法解析求解 Lagrange 乘子，守恒崩溃；
- **PINN-SC 和 PINN-KTT 出现严重漂移或震荡**；
- **SDIFP 保持稳定守恒**，误差几乎不受采样方式影响。

#### （3）高维扩展性（Fig. 5.6–5.8）：
- **内存消耗**：固定网格随维度指数增长，$D \geq 7$ 即 OOM；而 SDIFP 在随机采样下内存平稳；
- **计算时间**：SDIFP 在 $D=1000$ 下仍可运行，相比基线提速 **126×**；
- **守恒误差增长缓慢**：SDIFP 误差随维度近似平缓，而其他方法迅速恶化至接近 100% 错误。

---

### ✅ 消融实验与关键验证

- **分离式 MC 积分有效性**：若将积分嵌入 AD 图，则训练立即 OOM；
- **DS-UGE 的无偏性验证**：通过 Fubini-Tonelli 定理证明期望梯度一致；
- **正交解耦加速机制**：允许 $I=J$ 实现“一次采样”加速，在初期探索阶段有效提升收敛速度；
- **数值稳定性处理**：加入 $\max(\sigma^2, \epsilon)$ 防止除零，$\epsilon=10^{-8}$。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **精确守恒可在完全无网格条件下实现**：SDIFP 成功摆脱了对均匀网格的依赖，首次实现了在任意非结构化点云上的**机器精度级守恒**。
2. **非凸约束可通过函数空间变换闭式求解**：将非凸投影转化为低维代数系统，绕过了通用隐式求解器的凸性限制。
3. **双随机估计器打破双重维度诅咒**：
   - **积分维度诅咒**：由 detached MC 解决；
   - **微分维度诅咒**：由 DS-UGE 解决；
   二者结合使 PINNs 可扩展至 **100,000 维系统**（文中提及）。
4. **解的正则性得以保留**：由于守恒参数与局部梯度更新解耦，避免了高频伪影，提升了预测平滑性。

---

### ❗ 局限性

1. **不兼容 Dirichlet 边界条件**：
   - 当前仿射变换 $u = \alpha u_{\text{raw}} + \beta$ 中的常数项 $\beta(t)$ 会破坏 $u|_{\partial X}=0$；
   - 文中明确指出这是当前框架的数学不兼容问题。
2. **仅适用于全局积分型守恒律**：
   - 对局部约束（如熵不等式）、点态约束难以直接推广；
3. **假设守恒量已知**：
   - 方法要求 $c_1(t), c_2(t)$ 为已知常数或可显式积分得到的时间函数。

---

### 🔮 未来工作方向

1. **扩展至 Dirichlet 约束**：
   - 引入空间依赖的掩码函数 $u(x,t) = \alpha u_{\text{raw}} + \beta \phi(x)$，其中 $\phi(x)|_{\partial X}=0$；
2. **集成到 Neural Operators**：
   - 将 SDIFP 投影层嵌入 FNO、DeepONet 等架构，构建端到端保守算子学习；
3. **处理更复杂的守恒结构**：
   - 如守恒律组（Euler 方程）、Hamiltonian 结构、多不变量子系统；
4. **在线自适应守恒估计**：
   - 结合观测数据动态修正 $c(t)$，用于非理想封闭系统。

---

## 总结

SDIFP 是一项突破性的方法，**从根本上解决了高维 PINNs 中精确守恒的理论与工程难题**。它通过“函数空间仿射投影 + 分离式随机估计”的范式转移，实现了：

- ✅ **无网格**
- ✅ **精确守恒**
- ✅ **非凸兼容**
- ✅ **高维可扩展**

为发展下一代可信、稳定、高效的科学深度学习求解器提供了坚实基础。

</details>

---

### 8. [From Physics to Surrogate Intelligence: A Unified Electro-Thermo-Optimization Framework for TSV Networks](https://arxiv.org/abs/2603.29268)

**Authors**: Mohamed Gharib, Leonid Popryho, Inna Partin-Vaisband  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.29268v1  

#### Abstract
High-density through-substrate vias (TSVs) enable 2.5D/3D heterogeneous integration but introduce significant signal-integrity and thermal-reliability challenges due to electrical coupling, insertion loss, and self-heating. Conventional full-wave finite-element method (FEM) simulations provide high ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**高密度 Through-Substrate Vias (TSVs)** 在 2.5D/3D 异构集成中带来的信号完整性（如反射、插入损耗、串扰）和热可靠性挑战，解决传统全波有限元法（FEM）仿真在大规模设计空间探索时计算成本过高、难以实用化的问题。

现有方法存在以下不足：
- 分析模型多局限于对称小规模结构，无法推广到任意布局的大规模阵列；
- 忽略高频下的寄生电感效应，适用频率受限（通常 < 20 GHz）；
- 机器学习辅助优化仍依赖大量 FEM 数据，扩展性差；
- 缺乏电-热协同建模能力。

---

### 提出的新方法与新思路
本文提出了一种**统一的电-热协同优化框架**，结合物理驱动建模、图神经网络（GNN）代理模型与多目标 Pareto 优化，实现高效且高精度的设计探索。其核心创新点如下：

1. ✅ **物理信息引导的解析建模（Physics-informed Analytical Modeling）**
   - 扩展了文献 [29] 中的金属-绝缘体-半导体（MIM）模型，支持多接地 TSV 结构和非对称布局；
   - 同时计算宽带 S-参数 和 各向异性等效热导率（Effective Thermal Conductivity, ETC），支持电-热耦合分析；
   - 支持任意尺寸、信号/地分配及几何参数配置。

2. ✅ **基于图神经网络的代理模型 TSV-PhGNN**
   - 将 TSV 阵列建模为完全连接图 $ G = (V, E) $，节点表示 TSV，边编码空间距离；
   - 引入 **Feature-wise Linear Modulation (FiLM)** 机制以显式调制频率影响；
   - 使用 **Graph Transformer** 进行消息传递，学习复杂电磁耦合行为；
   - 输出头强制满足互易性（Reciprocity: $ S_{ij} = S_{ji} $）约束。

3. ✅ **Sim-to-Sim 转移学习策略**
   - 先用快速但低保真的解析模型生成大规模训练数据进行预训练；
   - 再用少量高保真 Ansys HFSS 仿真数据微调，实现“低成本+高精度”的平衡。

4. ✅ **多目标 Pareto 优化与自动化验证流程**
   - 支持对 TSV 布局、半径、间距、高度、氧化层厚度等参数进行联合优化；
   - 构建包含反射系数、插入损耗、近端/远端串扰（NEXT/FEXT）、热导率的 Pareto 前沿；
   - 自动通过 PyAEDT 流程导出最优设计并由 Ansys HFSS/Mechanical 完成最终签核（sign-off）验证。

---

### 相比现有方法的优势

| 维度 | 本工作 | 现有典型方法 |
|------|--------|--------------|
| **可扩展性（Scalability）** | 支持高达 22×22 大型阵列 | 多数限于 5×5 或更小 |
| **频率范围** | 支持至 100 GHz 宽带分析 | 多数 < 20 GHz |
| **计算效率** | 千万级配置可在分钟内完成评估 | FEM 每次需小时级 |
| **准确性** | RFE < 2%，接近 FEM 精度 | 解析模型误差常 >10% |
| **电-热协同建模** | 显式建模 Joule 加热与温度反馈 | 多数忽略自加热效应 |

> 图 1 对比显示，本框架在 **scalability、frequency range、computational cost** 上全面优于传统 FEM、解析模型和纯 ML 辅助方法。

---

## 2. 核心实验方法和设置

### 数据集构建
- **解析数据集**：使用提出的物理模型生成 **100,000 个样本**，涵盖 3×3 到 20×20 不同尺寸、随机信号/地分配及几何参数组合；
  - 输入参数包括：TSV 半径（2–6 μm）、间距（20–60 μm）、高度（60–100 μm）、氧化层厚度（0.5–3 μm）；
  - 使用 PrimeSim HSPICE 提取 S-参数作为标签。
- **FEM 数据集**：使用 Ansys HFSS 仿真生成 **10,000 个高保真样本**，仅用于微调阶段，集中在 3×3 至 7×7 小型阵列。

### 实验设置
- **硬件平台**：
  - ML 训练/推理：AMD Ryzen 97950X + NVIDIA RTX 4090 GPU；
  - FEM 仿真：Intel 14-core 工作站运行 Ansys AEDT 2024 R2。
- **软件工具链**：
  - 物理建模：Python + NumPy；
  - 电路仿真：Synopsys PrimeSim HSPICE；
  - ML 框架：PyTorch 2.8.0 + CUDA 12.9；
  - FEM 与热仿真：Ansys HFSS / Mechanical。

### 评估指标
1. **Relative Frobenius Error (RFE)**：复数域全局误差度量
   $$
   \text{RFE} = \frac{\|S_{\text{model}} - S_{\text{HFSS}}\|_F}{\|S_{\text{HFSS}}\|_F}
   $$
   可同时捕捉幅度与相位偏差。
2. **Runtime Comparison**：单次预测耗时对比。
3. **Maximum Steady-State Temperature**：用于热模型验证。
4. **Average Crosstalk / Worst-case Victim Performance**：用于优化效果评估。
5. **Pareto Frontier Quality**：衡量多目标权衡能力。

### 基线方法对比
- **Analytical Model Only**：原始解析模型无 ML 加速；
- **Ansys HFSS**：黄金标准全波仿真；
- **State-of-the-Art (SOTA)**：引用 [22] 中基于遗传算法 + Q2D/FEM 的优化方法；
- **其他 ML 方法**：未明确列出，但隐含比较传统黑箱 ML 模型缺乏物理一致性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | 数值 |
|------|------|
| **解析模型 RFE** | 5% – 10.6% （vs HFSS） |
| **TSV-PhGNN RFE** | **< 1.68%**（跨尺度泛化至 15×15） |
| **推理速度（15×15）** | **1.73 ms**（vs HFSS 的 2,998 s） |
| **加速比** | 达 **~1.7×10⁶ 倍** |
| **热仿真加速** | ETC 模型达 **35× ~ 1300×**，误差 < 5% |
| **设计空间探索能力** | 5×5 阵列共 5.2×10⁶ 种布局，可在 **5 分钟内穷举**；利用 D₄ 对称性后降至 1 分钟内 |

---

### 与基线方法的对比结果

#### 📌 与 [22]（Genetic Algorithm + FEM）对比
| 场景 | 本工作 | [22] | 改进 |
|------|--------|-------|------|
| 12-signal 5×5 阵列最差受害者串扰 | **-45 dB** | -34 dB | ↓ **11 dB**（约 3.5× 幅度下降） |
| 9-signal 5×5 阵列最差受害者串扰 | **-48 dB** | -30 dB | ↓ **18 dB**（约 8× 幅度下降） |
| 探索规模 | 支持 9–15 信号 TSV，累计 ~2.8×10⁷ 配置 | 仅能处理有限迭代 | ✅ 全面覆盖 |

> 图 12 和图 13 展示了优化后的 TSV 布局及其显著降低的 NEXT/FEXT。

#### 📌 多目标 Pareto 优化结果（Table V）
在 15 GHz 下获得四类 Pareto 最优设计：
- **最佳串扰设计**：Worst Xtalk = **-50.71 dB**
- **最佳热导设计**：$ k_z = 149.02 \, \text{W/mK} $
- **最佳插入损耗设计**：$ S_{21} = -0.066 \, \text{dB} $
- **最佳反射设计**：$ \max|S_{11}| = -45.75 \, \text{dB} $

揭示了不同目标间的内在权衡关系。

---

### 消融实验结果（Implicit Ablation Study）

虽然未设独立章节，但从设计选择中可推断关键组件作用：

| 组件 | 作用说明 |
|------|---------|
| **FiLM 调制频率输入** | 显著提升宽频段建模能力，避免将频率作为普通特征导致的过拟合 |
| **Graph Transformer + Attention** | 成功捕获屏蔽效应（如 ground TSV 阻挡耦合路径） |
| **Symmetry Enforcement ($ S_{ij}=S_{ji} $)** | 提升物理一致性，减少无效预测 |
| **Transfer Learning（Analytical → HFSS）** | 减少对昂贵 FEM 数据的依赖，仅需 10k 样本即可达到 sub-2% RFE |
| **Sparsity-aware ETC Model** | 正确建模稀疏阵列下的横向热扩散，避免均匀同质化带来的高估 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **物理先验 + ML 代理是破解“精度-效率”困境的有效路径**：
   - 解析模型提供可解释、可扩展的基础；
   - GNN 代理继承物理规律并在高频细节上逼近 FEM；
   - Sim-to-Sim 微调大幅减少对 FEM 数据的需求。

2. ✅ **图结构天然适配 TSV 阵列建模**：
   - 支持任意大小、形状、布线模式；
   - 可外推至训练之外的更大阵列（如从 7×7 → 15×15）；
   - 注意力机制有效识别关键耦合路径。

3. ✅ **大规模穷举优于启发式搜索**：
   - 利用对称性压缩搜索空间（D₄ symmetry 可减 8×）；
   - 快速代理使百万级枚举成为可能，发现非直观高性能布局；
   - 显著优于 GA、PSO 等受限于仿真次数的传统优化方法。

4. ✅ **电-热协同不可忽视**：
   - Joule 加热影响电阻进而改变 S-参数；
   - 提出的 ETC 模型兼顾精度与速度，适用于稳态热分析。

---

### 方法的局限性
1. ❗ **VRAM 限制大阵列推理**：
   - 图为全连接结构，边数随 $ N^4 $ 增长；
   - 当前最大支持 ~22×22，更大阵列受 GPU 显存限制。

2. ❗ **热模型目前仅支持稳态分析**：
   - 虽然已计算等效体积热容（$ \rho C_p $），但暂未开展瞬态电-热仿真。

3. ❗ **未考虑制造变异性和工艺偏差**：
   - 当前假设理想几何与材料参数；
   - 实际中需引入不确定性建模（UQ）。

4. ❗ **FEM 微调仍需一定人工干预**：
   - HFSS 设置、网格划分、收敛判断尚未完全自动化。

---

### 未来工作方向
1. 🔧 **稀疏图表示与局部性剪枝**：
   - 引入距离阈值或注意力掩码，减少远距离弱耦合边；
   - 降低内存占用，提升超大规模阵列可扩展性。

2. ⏱️ **扩展至瞬态电-热联合仿真**：
   - 利用已有的 $ (\rho C_p)_\text{eq} $ 模型构建动态热响应；
   - 支持周期性负载下的温度波动分析。

3. 🤖 **闭环自主设计系统**：
   - 集成贝叶斯优化、强化学习等主动学习策略；
   - 实现“探索-优化-验证”全自动流程。

4. 🔄 **不确定性量化与鲁棒设计**：
   - 引入随机变量建模工艺波动；
   - 开发稳健优化目标函数。

---

## 总结
本文提出了一套**从物理建模到代理智能的完整 TSV 设计自动化框架**，实现了：
> 🔥 **百万倍加速** + 💯 **近 FEM 精度** + 🎯 **多目标 Pareto 优化** + ✅ **自动签核验证**

不仅解决了传统方法在**效率与精度之间难以兼得**的根本矛盾，还推动 TSV 设计从“试错式”走向“系统化、数据驱动”的新时代，为下一代高性能异构集成芯片提供了强有力的 EDA 支撑。

</details>

---

### 9. [ASI-Evolve: AI Accelerates AI](https://arxiv.org/abs/2603.29640)

**Authors**: Weixian Xu, Tiantian Mi, Yixiu Liu, Yang Nan, Zhimeng Zhou, Lyumanshan Ye, Lin Zhang, Yu Qiao, Pengfei Liu  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.29640v1  

#### Abstract
Can AI accelerate the development of AI itself? While recent agentic systems have shown strong performance on well-scoped tasks with rapid feedback, it remains unclear whether they can tackle the costly, long-horizon, and weakly supervised research loops that drive real AI progress. We present ASI-E...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ASI-Evolve: AI Accelerates AI**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前 AI 研究依赖于人类主导的“假设生成-实现-实验-分析”循环，该过程受限于**多维人类瓶颈**：
- **探索能力有限**：人类难以并行探索大规模假设空间；
- **实验成本高**：模型训练、数据清洗等流程耗时耗力；
- **知识积累困难**：经验难以系统化保存与复用。

因此，论文提出一个核心问题：  
> **能否构建一个统一框架，让 AI 自主加速其自身在数据、架构、算法三大基础组件上的研发进程？**

### **提出了什么新方法或新思路**
作者提出 **ASI-EVOLVE** —— 一种面向 AI-for-AI 研究的**智能体演化框架**，通过闭环的 **Learn-Design-Experiment-Analyze 循环** 实现自主科研。

#### **两大核心创新组件**：
1. **Cognition Base（认知库）**
   - 注入从文献中提取的人类先验知识（如设计原则、常见陷阱），引导搜索方向。
   - 避免重复试错，显著提升冷启动效率。

2. **Dedicated Analyzer（专用分析器）**
   - 将复杂的多维度实验输出（loss 曲线、benchmark 分布、效率日志等）提炼为可重用的洞察报告。
   - 支持跨轮次的知识沉淀，形成“经验数据库”。

该框架不仅演化解决方案，更**演化认知本身**，实现了可持续的长期改进。

### **相比现有方法的优势**
| 维度 | ASI-EVOLVE | 其他系统（如 SciMaster, AI Scientist, AlphaEvolve） |
|------|------------|--------------------------------------------------|
| **任务范围** | 覆盖 AI 开发全栈（data, arch, algo） | 多为单一任务或封闭问题 |
| **反馈复杂性** | 可处理间接、噪声大、多维反馈 | 多依赖标量评分或简单规则 |
| **知识传承** | 显式建模“认知”与“经验”的双通道存储 | 缺乏系统化的知识蒸馏机制 |
| **通用性** | 可迁移至数学、生物医药等领域 | 多数局限于特定领域 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与任务场景**

#### **(1) Neural Architecture Design**
- **任务**：设计优于 DeltaNet 的线性注意力结构。
- **基线模型**：DeltaNet, Gated-DeltaNet, Mamba2
- **评估基准**：Wiki, LAMBADA, PIQA, HellaSwag, ARC-e/c, SIQA, BoolQ 等共 16 个 benchmark。
- **训练规模**：
  - 探索阶段：~20M 参数模型，训练 2000 步；
  - 验证阶段：~340M / ~1.3B 参数，训练 1B / 100B tokens。

#### **(2) Pretraining Data Curation**
- **原始语料**：Nemotron-CC（672B tokens，涵盖数学、CS、医学等 STEM 领域）
- **目标**：自动发现高质量清洗策略，提升预训练数据质量。
- **评估模型**：3B 参数语言模型，训练 500B tokens。
- **测试基准**：BBH, ARC-E/C, MMLU, AGIEval, CSQA, MedQA, MedMCQA, PubMedQA 等 18 项。

#### **(3) Reinforcement Learning Algorithm Design**
- **任务**：优化 GRPO 的优势分配与梯度计算机制。
- **训练框架**：SIIRL，使用 Skywork OR1 数据集。
- **基线**：Group Relative Policy Optimization (GRPO)
- **评估基准**：Math500, AMC32, AIME24/25, OlympiadBench

#### **(4) 跨领域验证：Drug-Target Interaction Prediction**
- **任务**：在生物医学领域验证架构泛化能力。
- **数据集**：BindingDB, Human, BioSNAP, C.elegans
- **评估设置**：random split, unseen drug, unseen protein, cold-start
- **指标**：AUROC, AUPRC, F1, MCC

### **实验设置与评估指标**

| 模块 | 功能说明 |
|------|--------|
| **Researcher** | 基于 LLM 生成新程序 + 自然语言动机，结合上下文节点与认知条目 |
| **Engineer** | 执行实验，返回结构化指标（主分数 + 辅助信号）；支持早停与 LLM Judge |
| **Analyzer** | 解析原始日志，生成决策导向的分析报告 |
| **Database** | 存储历史节点（动机、代码、结果、分析、元数据） |
| **Cognition Store** | 向量化索引的认知条目库，支持语义检索 |

**采样策略对比**：UCB1, Random, MAP-Elites  
**基模型对比**：GPT-5-mini, Qwen3-32B

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) Neural Architecture Design**
- 在 1,773 轮探索中，**发现 105 个超越 DeltaNet 的 SOTA 架构**。
- 最佳模型相对 DeltaNet 提升 **+0.97 点平均准确率**，约为当前人类最先进改进（Mamba2）的 **3 倍增益**。
- 在泛化 benchmark 上也取得稳定提升（最高达 +0.66 点）。

#### **(2) Pretraining Data Curation**
- 使用 ASI-EVOLVE 发现的清洗策略构建 **Nemotron-CCAsI+**（504B tokens）。
- 相比原始数据，平均 benchmark 性能提升 **+3.96 点**。
- 在知识密集型任务上表现尤为突出：
  - **MMLU +18.64 点**
  - **CSQA +18.80 点**
  - **MedQA +13.48 点**

#### **(3) Reinforcement Learning Algorithm Design**
- 发现多个优于 GRPO 的 RL 算法，在数学推理任务上大幅领先：
  - **AMC32: +12.5 pts (67.5 → 80.0)**
  - **AIME24: +11.67 pts (20.00 → 31.67)**
  - **OlympiadBench: +5.04 pts (45.92 → 50.96)**

#### **(4) Circle Packing 基准测试**
- 在经典组合优化任务上进行横向比较：
  - **仅用 17 步即达到 SOTA 水平（2.63597）**
  - 最终得分 **2.635983**，媲美 AlphaEvolve 和 SkyDiscover
- 显著快于 OpenEvolve（460 步）、GEPA（未达 SOTA）

#### **(5) Drug-Target Interaction 预测（跨领域应用）**
- 在 BindingDB-dev 上：
  - AUROC 从 94.15（DrugBAN）提升至 **96.06（+1.91）**
  - F1 从 86.89 提升至 **89.84（+2.95）**
- 在冷启动场景下泛化能力更强：
  - **Unseen Drug: +6.94 AUROC**
  - **Unseen Protein: +3.56 AUROC**
  - **Double Cold-start: +4.36 AUROC**

---

### **与基线方法的对比结果**

| 方法 | AMC32 | AIME24 | OlympiadBench | MMLU | MedQA | DTI AUROC |
|------|-------|--------|---------------|------|-------|----------|
| GRPO | 67.5 | 20.00 | 45.92 | — | — | — |
| ASI-EVOLVE | **80.0 (+12.5)** | **31.67 (+11.67)** | **50.96 (+5.04)** | — | — | — |
| Raw Data | — | — | — | 27.49 | 26.77 | — |
| Nemotron-CCAsI+ | — | — | — | **46.13 (+18.64)** | **40.25 (+13.48)** | — |
| DrugBAN | — | — | — | — | — | 94.15 |
| ASI-EVOLVE-discovered | — | — | — | — | — | **96.06 (+1.91)** |

---

### **消融实验结果**

#### **移除 Analyzer**
- 初始阶段仍能获得较高分（得益于 Cognition 库）；
- 后期陷入平台期，改进缓慢且不可靠；
- 表明 Analyzer 对持续演进至关重要。

#### **移除 Cognition Base**
- 冷启动极慢，早期得分低；
- 经过足够迭代后可追上，但收敛速度明显下降；
- 表明 Cognition 加速探索，但非绝对必要。

#### **不同 Sampling 策略对比**
- **UCB1** > **MAP-Elites** > **Random**
- UCB1 在强认知引导下表现出更快收敛（exploitation 更高效）
- 结合 GPT-5-mini，**仅 17 步达 SOTA**；而 MAP-Elites 需 79 步

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **首次实现统一框架下的 AI 自我加速研发**：
   - 成功在 **data、architecture、algorithm** 三个核心维度实现 AI-driven discovery。
2. 🔍 **认知与分析双通道机制是关键**：
   - Cognition Base 提供“方向感”，Analyzer 实现“反思能力”，二者共同支撑长周期科研闭环。
3. 🚀 **性能远超人类设计与现有自动化系统**：
   - 在多个任务上实现数量级超越已有手动优化成果。
4. 🌍 **具备跨领域泛化潜力**：
   - 在药物靶点预测等非 AI 任务上仍展现强大设计能力，证明其科学普适性。

### **方法的局限性**
1. **无法直接优化底层实现**：
   - 如不能生成 CUDA kernel 级别的硬件优化代码，仅限高层架构/算法设计。
2. **依赖高质量初始认知输入**：
   - 若领域缺乏成熟文献，冷启动效果受限。
3. **计算资源消耗巨大**：
   - 单次实验可能需数十至上百 GPU 小时，限制部署门槛。
4. **对异常反馈敏感**：
   - 若评估脚本出错或评分失真，可能导致错误学习路径。

### **未来工作方向**
1. **扩展至 AI 全栈自进化**：
   - 引入基础设施（infrastructure）、编译器优化、分布式调度等模块。
2. **增强跨模态与跨学科迁移能力**：
   - 探索在物理、化学、材料科学中的应用。
3. **降低资源需求与提高鲁棒性**：
   - 引入 early rejection、meta-learning 初始化等技术减少试错成本。
4. **人机协同范式升级**：
   - 人类角色从“执行者”转向“问题定义者”与“价值校准者”。

---

> 📌 **一句话总结**：  
> **ASI-EVOLVE 是首个实现 AI 在数据、架构、算法三大支柱上自主进化的统一框架，标志着“AI 加速 AI”从愿景走向现实，开启了闭环保证的 AI 自我演化新时代。**

🔗 **开源地址**：[https://github.com/GAIR-NLP/ASI-Evolve](https://github.com/GAIR-NLP/ASI-Evolve)

</details>

---

### 10. [CRAFT: Cost-aware Expert Replica Allocation with Fine-Grained Layerwise Estimations](https://arxiv.org/abs/2603.28768)

**Authors**: Adrian Zhao, Zhenkun Cai, Zhenyu Song, Lingfan Yu, Haozheng Fan, Jun Wu, Yida Wang, Nandita Vijaykumar  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.28768v1  

#### Abstract
Mixture-of-Experts (MoE) has recently emerged as the mainstream architecture for efficiently scaling large language models while maintaining near-constant computational cost. Expert parallelism distributes parameters by partitioning experts across devices, but this introduces token-level load imbala...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**CRAFT: Cost-aware Expert Replica Allocation with Fine-Grained Layerwise Estimations**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在大规模 **Mixture-of-Experts (MoE)** 模型的推理部署中，**Expert Parallelism (EP)** 虽然能有效分布专家参数，但由于动态路由机制导致 **token-level 负载不均衡**（即少数“热”专家承载大量请求），引发 GPU 间计算负载失衡、通信拥塞等问题。

为缓解此问题，业界广泛采用 **Expert Replication**（专家复制）技术——将高负载专家复制到多个设备上以分摊流量。然而，主流方案如 **EPLB** 采用 **每层每设备统一复制一个副本**（uniform replication），造成严重的 **内存浪费** 和 **KV Cache 压缩**，反而可能降低整体吞吐量。

CRAFT 正是针对这一矛盾提出的新框架：  
> **如何在有限 GPU 内存预算下，最大化专家复制带来的负载均衡收益？**

---

### ✅ 提出的新方法与核心思想

CRAFT 是一种 **端到端、成本感知的专家副本分配框架**，其核心创新在于：

#### 🔹 **细粒度、逐层（per-layer）的复制决策**
- 不再对所有 MoE 层采用相同复制策略。
- 提出 **“复制效益”（replication benefit）** 概念：衡量每一层在不同副本数下的 **负载均衡增益**。
- 构建 **cost model** 来估计每层的复制效益曲线。

#### 🔹 **基于动态规划的最优副本分配**
- 将副本分配建模为 **Multiple-Choice Knapsack Problem (MCKP)**：
  - 每层选择一个副本数量（选项）
  - 总副本数受内存预算约束（背包容量）
  - 目标是最大化总均衡增益
- 使用 **动态规划** 高效求解，在实际规模下运行开销极低。

#### 🔹 **容量感知的专家分配策略（Capacity-Aware Assignment）**
- 解决非均匀复制带来的设备间内存不平衡问题。
- 设计 **interleaved assignment + greedy placement** 策略，确保：
  - 所有设备保留相同的专家槽位（保持 KV Cache 一致）
  - 各层内部专家分布尽可能均衡

---

### ✅ 相比现有方法的优势

| 特性 | EPLB（Baseline） | CRAFT（本文） |
|------|------------------|-------------|
| 复制粒度 | Uniform per layer | Fine-grained per layer |
| 内存效率 | 低（固定高开销） | 高（按需分配） |
| 负载均衡能力 | 强但代价高 | 接近最优，代价更低 |
| 可集成性 | 支持主流 Serving Framework | 无缝替换 EPLB，无需训练或模型修改 |
| 自动化程度 | 手动配置 | 支持自动选择最优 `R` |

> 💡 **优势总结**：CRAFT 在显著减少副本数量的前提下，达到甚至超过 EPLB 的负载均衡效果，从而释放更多内存用于 KV Cache 和批处理，提升端到端吞吐。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与模型

#### 模型
- **DeepSeek-R1-671B**：6710亿参数，58个 MoE 层，256 专家，top-8 路由
- **Kimi-K2-1000B**：1万亿参数，60个 MoE 层，384 专家，top-8 路由
- 均使用 **bfloat16** 精度

#### 数据集（长序列输入）
| 名称 | 描述 |
|------|------|
| **FinePDFs** | 包含德语（deu_Latn）、日语（jpn_Jpan）文本，用于测试多语言场景 |
| **Lambada** | 上下文依赖性强的语言建模任务 |
| **RedPajama-Data-1T (arXiv split)** | 学术文献数据，测试专业领域负载特性 |

> 所有输入截断至 **4096 tokens**，输出长度固定为 **256 tokens**

---

### ⚙️ 实验设置

- **硬件平台**：AWS EC2 `p4de.24xlarge` 实例（8×NVIDIA A100 80GB GPU），通过 NVLink 连接
- **通信后端**：NCCL 2.26.2 + EFA（Elastic Fabric Adapter）支持跨节点 P2P
- **推理框架**：基于 **SGLang v0.4.8** 实现，集成 EPLB 作为 baseline
- **并行策略**：DP（数据并行） + TP-Attention（张量并行注意力） + EP（专家并行）

---

### 📊 评估指标

| 指标 | 定义 |
|------|------|
| **Balancedness** | 平均负载 / 最大负载，越接近 1 表示负载越均衡 |
| **Goodput** | 系统可维持的最大稳定吞吐量（req/s），定义为 TTFT 开始急剧上升前的拐点 |
| **Throughput (req/s)** | 请求吞吐率 |
| **TTFT (Time to First Token)** | 首 token 延迟 |
| **ITL (Inter-Token Latency)** | 解码阶段 token 生成延迟 |

---

### 🔁 基线方法对比

| 方法 | 描述 |
|------|------|
| **BASE** | 仅启用 Expert Placement，无复制 |
| **EPLB** | 默认最小复制策略：每层每 GPU 分配 1 个副本（共 `L` 个副本/GPU） |
| **CRAFT (CRA)** | 本文方法，支持灵活设置复制因子 `R`（如 CRA8 表示 `R=8`） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **平均吞吐提升** | **1.14×** vs EPLB（最高达 **1.2×**） |
| **峰值吞吐提升** | **1.17×**（见 Figure 1） |
| **内存节省** | 相比 EPLB 减少 **7.25× ~ 7.5× 副本数** |
| **KV Cache 缩减幅度** | EPLB 最多压缩 **75%**；CRAFT 仅 **6%** |
| **TTFT 改善** | 平均降低 **29%**（最高 **58%**）vs BASE |

---

### 🆚 与基线方法对比结果

#### （1）端到端吞吐（Goodput）

- 在 **8节点集群** 上：
  - 对 **DeepSeek-R1**：CRAFT 平均提升 **1.15×** 吞吐（最高 1.2×）
  - 对 **Kimi-K2**：平均提升 **1.12×**（最高 1.17×）
- 即使在 **小集群（6节点）** 中，EPLB 因过度占用内存导致吞吐 **下降 46%**，而 CRAFT 仍能实现 **1.14× 提升**

> ✅ **原因分析**：CRAFT 用更少副本实现了相近的负载均衡，释放内存用于更大 batch 和 KV Cache。

#### （2）负载均衡效果

- CRAFT 在远低于 EPLB 的副本预算下，即可达到 **接近饱和的 balancedness**
- 如图 5 所示，增加副本超过 16 后，均衡性增益趋于平缓 → **存在明显边际递减效应**

#### （3）不同数据集鲁棒性

| 数据集类型 | BASE Balancedness | CRAFT Goodput Gain | EPLB Goodput Gain |
|----------|-------------------|--------------------|-------------------|
| 高偏斜（E/J） | 低 | **1.42×** | 1.24× |
| 低偏斜（L/A） | 高 | **1.14×** | 仅 1.02× |

> ✅ CRAFT 在各类负载下均表现稳健；而 EPLB 在低偏斜数据中因“过度复制”几乎无收益。

---

### 🔍 消融实验结果

#### （1）复制因子 `R` 的影响（Figure 9 & 11）

- 存在 **最佳 `R` 值**（通常为 8），过高或过低都会损害性能：
  - `R` 太小 → 负载不均严重
  - `R` 太大 → KV Cache 被压缩，batch size 下降，goodput 反而下降
- CRAFT 支持自动选择最优 `R`，避免人工调参

#### （2）逐层差异化复制的有效性

- 图 6 显示不同层达到“收益饱和”的副本数差异巨大：
  - Layer 20（低偏斜）：复制无效
  - Layer 51（高偏斜）：需 16 个副本才饱和
- 统一复制策略必然导致 **某些层复制不足，另一些层复制过剩**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **专家复制存在显著的边际效益递减现象**  
   → 盲目增加副本不仅浪费内存，还会损害系统吞吐。

2. **各 MoE 层对复制的敏感度差异巨大**  
   → 应采用 **fine-grained per-layer** 策略，而非 uniform replication。

3. **CRAFT 可在极低副本预算下逼近最优均衡性**  
   → 实现 **高性价比的负载均衡优化**。

4. **CRAFT 与现有 Serving Framework 完全兼容**  
   → 无需修改模型结构或重新训练，易于部署。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖离线负载分析** | 需预先收集一段时间的推理 trace 来构建负载分布，对快速变化的工作负载适应性有限 |
| **静态复制计划** | 当前为静态配置，未实现实时动态调整（但可通过周期性重分析解决） |
| **假设负载分布稳定** | 若路由模式剧烈变化（如切换任务域），可能需要重新 profiling |

---

### 🔮 未来工作方向

1. **在线自适应复制机制**  
   → 结合运行时监控，动态调整各层副本数。

2. **与 Expert Sharding 或 Grouping 技术结合**  
   → 如与 **GraceMoE** 的 grouping 策略协同，进一步优化通信与内存。

3. **扩展至训练场景**  
   → 当前聚焦推理，未来可探索在分布式训练中的应用。

4. **支持异构设备环境**  
   → 在混合 GPU 类型或内存容量的集群中进行智能副本分配。

---

## ✅ 总结一句话

> **CRAFT 通过细粒度、逐层感知的专家副本分配，在极低内存开销下实现了接近最优的负载均衡，平均提升 MoE 模型推理吞吐 1.14×，且完全兼容现有 Serving 框架，是高效、实用的大规模 MoE 推理优化方案。**

</details>

---

### 11. [Exploration of Energy and Throughput Tradeoffs for Dataflow Networks](https://arxiv.org/abs/2603.29367)

**Authors**: Abrarul Karim, Joachim Falk, J\"urgen Teich  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.29367v1  

#### Abstract
The introduction of dynamic power management strategies such as clock gating and power gating in dataflow networks has been shown to provide significant energy savings when applied during idle times. However, these strategies can also degrade throughput due to shutdown and wake-up delays. Such throu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代信号处理系统（如物联网设备中的图像/视频分析）面临**高计算需求**与**能量受限**之间的矛盾。虽然动态功耗管理（如时钟门控、电源门控）可在空闲期显著节能，但其引入的**唤醒延迟**（wake-up delay）和**关机延迟**（shutdown delay）会降低系统的吞吐量（throughput），这对需要保证实时性的应用尤为不利。

本文旨在解决以下两个核心问题：
1. 如何在允许部分actor进入自供电（self-powered）模式的同时，仍能保证周期性调度的存在性和最大吞吐量？
2. 如何在给定吞吐量约束下，最小化总能耗，并高效探索能量与吞吐量之间的权衡空间？

---

### **提出的新方法与新思路**
论文提出了三个层次的贡献：

#### **(1) 最大吞吐量周期性调度的线性规划（LP）建模**
- **方法**：提出一个**线性规划**（Linear Program, LP）公式 `LP(g, x)`，用于在给定actor运行模式（`x` 向量，0为Always-Active，1为Self-Powered）的情况下，求解该配置下的**最小周期**（即最大吞吐量）。
- **创新点**：首次将自供电数据流网络（self-powering dataflow networks）的执行时间变化（因唤醒/关机延迟）形式化为可优化的数学模型，证明了周期性调度的存在性。

#### **(2) 给定吞吐量下最小能耗的混合整数线性规划（MILP）**
- **方法**：提出一个**混合整数线性规划**（Mixed-Integer-Linear-Program, MILP）公式 `MILP(g, P)`，在给定周期 `P` 下，自动决定每个actor应处于AA还是SP模式，以**最小化每周期总能耗**。
- **创新点**：将能耗优化问题转化为决策问题，识别出“关键actor”（critical actors）——这些actor必须保持AA模式以满足周期约束，其余actor可设为SP以节能。

#### **(3) 高效多目标设计空间探索策略：Hop and Skip (H&S)**
- **方法**：提出一种名为 **Hop and Skip** 的新型多目标设计空间探索（DSE）策略。
  - **Hop**：调用MILP，跳转到当前周期下能耗最低的配置。
  - **Skip**：调用LP，验证该配置能否支持更小的周期，从而“跳过”中间大量无意义的周期点。
- **创新点**：避免了对所有 $2^{|A|}$ 种配置的穷举或对所有整数周期的遍历，极大提升了探索效率。

---

### **相比现有方法的优势**
| 方法 | 时间复杂度 | 是否找到Pareto前沿 | 效率优势 |
|------|------------|---------------------|----------|
| **Decision Variable Sweep (x Sweep)** | $O(2^{|A|})$ | ✅ 是（精确） | 基线，慢 |
| **Period Sweep (P Sweep)** | $O(P_{\text{max}} - P_{\text{min}})$ | ❌ 可能遗漏有理数周期点 | 中等 |
| **Hop and Skip (H&S)** | 远低于前两者 | ✅ 是（几乎总是） | ⭐ **最高，可达上千倍加速** |

H&S 在保证找到完整Pareto前沿的前提下，探索时间远低于其他两种方法。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **真实世界案例研究**（Real-world case study）：
   - **AEC Network**（Acoustic Echo Cancellation）：一个实际的声学回声消除数据流网络，包含6个核心actor。
2. **基准测试集**（SDF3 Benchmarks）：
   - 来自 **SDF3-SDF For Free** 工具套件的7个标准数据流图（DFG），涵盖不同拓扑（循环型、前馈型）。
   - 包括：H.263 Encoder, Modem, MP3 Playback, Satellite, Samplerate, MP3 Decoder (Block/Granule)。
3. **随机生成图**（Random Graphs）：
   - 使用SDF3随机图生成器创建了 **100个SDF图**，每个图约含15个actor，用于统计评估。

> 注：所有SDF图均通过展开（unfolding）转换为Marked Graph以适用本文分析。

---

### **实验设置与评估指标**

#### **参数设置**
- **功耗与延迟模型**：
  - 执行时间 `lexe(a)` 来自综合结果或基准数据。
  - 唤醒延迟 `lwkp(a) = 1` 时钟周期，关机延迟 `lshd(a) = 2` 时钟周期（基于硬件模拟）。
  - 功耗值（`pexe`, `pidle`, `pslp`, `pwkp`, `pshd`）通过归一化与随机缩放生成。

#### **评估指标**
1. **探索时间**（Exploration Time）：完成DSE所需的CPU时间。
2. **速度提升**（Speedup）：H&S 相对于 x Sweep 和 P Sweep 的加速比。
3. **超体积比**（Hypervolume Ratio, HV）：
   - 用于衡量近似Pareto前沿与真实前沿（由x Sweep得到）的接近程度。
   - $ \text{HV} = \frac{\text{hypervolume}(S_{\text{App}})}{\text{hypervolume}(S_{\text{Ref}})} $
   - 若 HV = 1，则说明找到的解集与真实Pareto前沿完全一致。

#### **基线方法对比**
- **x Sweep**（决策变量穷举）：作为**黄金标准**，用于生成真实Pareto前沿。
- **P Sweep**（周期遍历）：作为传统启发式搜索的代表。
- **H&S**（本文提出）：作为高效探索方法。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) AEC网络结果**
| 配置 | 周期 `P` | 能耗 `E` | 节能效果 | 吞吐量损失 |
|------|--------|--------|---------|-----------|
| 全AA模式 | 23 | 182 pJ | 基准 | 0% |
| 混合模式（MILP优化） | 23 | 146 pJ | **↓20%** | 0% |
| 全SP模式 | 27 | 82 pJ | **↓55%** | ↑17% |

> ✅ 在不牺牲吞吐量的前提下实现20%节能；允许17%周期增加可获55%节能。

---

#### **(2) SDF3基准测试（Table I）**
| 网络 | H&S vs x Sweep | H&S vs P Sweep | HV Ratio |
|------|----------------|----------------|----------|
| Satellite | 348.93× | — | 1.0000 |
| Modem | 1452.39× | 8.30× | 1.0000 |
| MP3Decoder (Block) | 1489.11× | 3.65× | 1.0000 |
| Samplerate | 2.67× | 27.83× | 1.0000 |
| **平均加速比** | **~342×** | **~30×** | **1.0000** |

> ✅ H&S在所有SDF3基准上均达到 **HV = 1**，即**完全复现真实Pareto前沿**。
> ✅ 加速比高达 **2300×**（Modem网络），尤其在大网络中优势明显。

---

#### **(3) 100个随机图结果**
- **H&S vs x Sweep**：
  - 当步长 `ε = 1` 时，98/100次成功找到完整Pareto前沿（HV=1）。
  - 当 `ε = 0.1` 时，**100/100次成功**（HV=1）。
- **P Sweep vs x Sweep**：
  - 仅95/100次成功，失败原因在于无法处理有理数周期点。
- **平均加速比**：
  - H&S 相比 P Sweep：**29.95×**
  - H&S 相比 x Sweep：**342.39×**

> ✅ H&S在统计意义上**稳定且高效**地逼近最优解。

---

### **消融实验（隐含）**
- **步长 `ε` 的影响**：实验证明较小的 `ε`（如0.1）可避免错过最优解，确保HV=1。
- **模式选择的影响**：MILP能准确识别关键actor（如AEC中a₂-a₅必须为AA），错误关闭会导致周期不可行。

---

## **4. 关键结论和发现**

### **主要发现**
1. **自供电模式可显著节能**：在AEC案例中，全SP模式节能达55%，混合模式在零吞吐损失下节能20%。
2. **Hop and Skip策略极其高效**：
   - 探索时间比暴力穷举（x Sweep）快数百倍，比周期遍历（P Sweep）快数十倍。
   - 在绝大多数情况下（尤其是配合小步长 `ε`）能**精确找到整个Pareto前沿**。
3. **关键actor的存在性**：某些位于关键路径上的actor（如决定最大环均值的actor）必须保持AA模式，否则无法维持最小周期。

---

### **方法的局限性**
1. **依赖周期性假设**：方法基于周期性调度（periodic schedule），可能不适用于高度动态或非周期性数据流。
2. **固定延迟模型**：假设唤醒/关机延迟为常数，而现实中可能受电压、温度等因素影响。
3. **MILP求解开销**：尽管H&S大幅减少调用次数，但每次MILP求解本身仍是NP-hard问题，对超大规模网络仍有挑战。

---

### **未来工作方向**
1. **扩展至异构功耗管理**：结合Dynamic Voltage Scaling (DVS) 与 DPM，进行多维度优化。
2. **支持动态数据流**：将方法推广至非静态或场景感知的数据流模型（如SADF）。
3. **硬件原型验证**：在FPGA或ASIC上实现H&S调度器，进行真实能效测量。
4. **在线自适应调度**：开发轻量级算法，在运行时根据负载动态调整actor的供电模式。

---

> **总结**：本文系统性地解决了自供电数据流网络中的**能效-吞吐量权衡**问题，提出的 **Hop and Skip** 策略在保证解质量的同时实现了**数量级的效率提升**，为低功耗信号处理系统的自动化设计提供了强有力的工具。

</details>

---

### 12. [Mitigating Temporal Blindness in Kubernetes Autoscaling: An Attention-Double-LSTM Framework](https://arxiv.org/abs/2603.28790)

**Authors**: Faraz Shaikh, Gianluca Reali, Mauro Femminella  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.28790v1  

#### Abstract
In the emerging landscape of edge computing, the stochastic and bursty nature of serverless workloads presents a critical challenge for autonomous resource orchestration. Traditional reactive controllers, such as the Kubernetes Horizontal Pod Autoscaler (HPA), suffer from inherent reaction latency, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Mitigating Temporal Blindness in Kubernetes Autoscaling: An Attention-Double-LSTM Framework*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在边缘计算和 Serverless 架构中，工作负载具有**高度随机性、突发性和非平稳性**，传统的 Kubernetes HPA（Horizontal Pod Autoscaler）采用基于阈值的**反应式控制机制**，存在显著的**反应延迟**（reaction latency），导致：
- 流量激增时出现 **SLO 违规**（如延迟超标）
- 负载下降时资源释放滞后，造成**资源抖动**（flapping）和浪费

此外，现有的深度强化学习（DRL）方法（如 DQN、PPO）虽然具备一定的预测能力，但由于缺乏对长期历史状态的有效建模，普遍存在“**时间盲视**”（temporal blindness）问题，即无法区分短期噪声与真正的流量趋势变化。

---

### 🚀 提出的新方法与创新思路

作者提出了一种**稳定性感知的自适应扩缩容框架**，其核心是将**工作负载预测**与**控制决策**统一在一个端到端的 DRL 框架内，具体创新如下：

1. **Attention-Enhanced Double-Stacked LSTM-PPO 架构**
   - 在 PPO 智能体的策略网络中引入**双层堆叠 LSTM**（Double-Stacked LSTM），以捕捉多尺度的时间依赖关系（短期波动 + 长期趋势）。
   - 结合**软注意力机制**（soft attention），使模型能够动态加权历史隐藏状态，聚焦于关键事件（如并发请求爆发前兆），同时过滤高频噪声。

2. **紧耦合预测-控制架构**
   - 不同于传统“先预测再控制”的分离设计（Separation of Concerns），该方法将预测模块直接嵌入控制策略网络内部，避免误差传播。

3. **多维离散动作空间设计**
   - 控制动作不仅限于副本数调整，还包括：
     - HPA CPU 目标利用率（30%/50%/70%/90%）
     - 吞吐量乘数（API Gateway 限流调节）
     - 增强级别（安全模式开关）
   - 实现更细粒度、更灵活的 Kubernetes 资源调控。

---

### 🔍 相比现有方法的优势

| 对比维度 | 传统 HPA | 单层 LSTM-PPO（如 DRe-SCale） | 本文方法 |
|--------|---------|-------------------------------|----------|
| 控制方式 | 反应式（Reactive） | 准主动式（Proactive） | 主动式（Proactive） |
| 时间建模能力 | 无记忆 | 浅层时序记忆 | 深层+注意力增强记忆 |
| 抗噪能力 | 弱（易受瞬时波动影响） | 中等 | 强（可识别真实趋势） |
| 扩缩容稳定性 | 差（冷却期导致滞后） | 一般（易震荡） | 优（低 churn） |

> ✅ **核心优势**：通过缓解“时间盲视”，实现了**更低延迟、更高稳定性、更少资源震荡**的自动扩缩容。

---

## 2. 核心实验方法和设置

### 📊 数据集
- 使用 **Azure Functions Invocation Trace 2021** 真实生产级函数调用日志。
- 包含多个服务的真实请求到达模式，具有典型的**突发性、非平稳性和长周期特征**。
- 实验选取其中 **7 天数据**，**5 天用于训练，2 天用于测试**。

### ⚙️ 实验环境与平台
- 部署在基于 **MicroK8s** 的异构边缘集群上，包含一个主节点（带 NVIDIA L40S GPU）和一个工作节点。
- 应用负载通过 `hey` 工具回放生成，目标为 CPU 密集型的 `factorizator` Serverless 函数（部署于 OpenFaaS）。
- 控制周期（Control Interval）为 **60 秒**。

### 🎯 评估指标
| 类别 | 指标名称 | 描述 |
|------|--------|------|
| **性能质量** | Avg Latency, P90 Latency | 平均延迟与尾部延迟（用户体验关键） |
| **SLO 合规性** | SLO Compliance @ 20ms / 50ms | 请求延迟低于目标阈值的比例 |
| **资源效率** | Avg CPU Utilization | 集群平均 CPU 利用率（越高越高效） |
| **系统稳定性** | Replica Churn, Oscillation Frequency | 副本变更次数与波动频率（越低越好） |
| **可靠性** | Fraction of Missed Calls | 请求失败或超时比例 |

### 🔁 基线方法对比
| 基线方法 | 类型 | 特点 |
|--------|-----|------|
| **Static HPA (50%)** | 行业标准 | 固定 50% CPU 阈值，反应式控制 |
| **DDQN (Double Deep Q-Network)** | 值函数法 DRL | Stateless，无显式记忆机制 |
| **Single-LSTM PPO** | 消融基线 | 仅单层 LSTM，无 attention，复现 DRe-SCale 思路 |

> 所有 RL 方法共享相同的动作解码逻辑和运行时稳定化规则，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table II）

| Agent | Avg Latency (ms) | P90 Latency (ms) | SLO @20ms | SLO @50ms | Avg CPU (%) | Replica Churn |
|-------|------------------|------------------|-----------|-----------|-------------|--------------|
| Static HPA | 58.82 | >300 | 10.3% | 43.5% | 15.44 | 97 |
| DDQN | 77.33 | ~175 | 9.0% | 24.6% | 87.29 | 0 |
| Single-LSTM | 32.37 | ~110 | 15.6% | 92.0% | 31.00 | 716 |
| **Double-LSTM (Ours)** | **24.11** | **~80** | **46.2%** | **96.9%** | **38.22** | **432** |

---

### 🔬 与基线方法的对比结果

#### ✅ 相比 Single-LSTM PPO（消融基线）：
- **P90 Latency 下降约 29%**（从 ~110ms → ~80ms）
- **Replica Churn 降低 39%**（从 716 → 432）
- SLO @20ms 提升近 **3 倍**（15.6% → 46.2%）
- 更高的 CPU 利用率（31% → 38.22%），说明资源利用更充分且未牺牲稳定性

> 💡 表明：**双层 LSTM + 注意力机制显著提升了预测准确性与控制平滑性**

#### ✅ 相比 Static HPA：
- 尾延迟（P90）改善超过 **70%**
- SLO @50ms 从 43.5% 提升至 96.9%
- 虽然副本变化更多（churn 更高），但均为**必要响应**，而非无效震荡

#### ✅ 相比 DDQN：
- DDQN 几乎不扩缩容（replica churn = 0），采取保守策略保资源成本，但导致严重过载
- 本文方法在保障性能的同时实现合理扩缩容

---

### 🔍 消融实验分析（Ablation Study）

- **移除 Attention 机制**（即退化为 Single-LSTM）：
  - 预测误差分布右偏更重（见 Fig. 8a KDE 图），表明对突发流量预测不准
  - 导致控制滞后，必须等到延迟上升后才被动响应
- **保留 Attention 但使用单层 LSTM**：
  - 仍存在信息瓶颈，难以捕获长期趋势
- **结论**：**双层结构提供深度表征能力，注意力机制实现选择性聚焦，二者缺一不可**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **“时间盲视”是制约 DRL 自动扩缩容可靠性的根本瓶颈**  
   单纯增加记忆单元（如 LSTM）不足以解决复杂边缘环境下的预测问题。

2. **深度注意力机制可有效缓解信息瓶颈**  
   通过让智能体学会“关注重要时刻”，实现了对真实需求趋势的精准识别，从而支持**前瞻性扩缩容**。

3. **预测与控制的深度融合优于分离架构**  
   端到端联合优化避免了预测误差向控制层的传播，提升整体鲁棒性。

4. **稳定性可通过奖励工程与平滑更新共同保障**  
   引入 `Rstab` 稳定性惩罚项 + PPO 的 clipped update，有效抑制了过度反应。

---

### ⚠️ 方法的局限性

1. **推理开销较高**
   - Attention + Double-LSTM 推理延迟约为 **5–10ms**（GPU 加速下可接受）
   - 在无 GPU 的资源受限边缘设备（如 IoT 网关）可能成为瓶颈

2. **实验场景简化**
   - 当前评估基于单一微服务（factorizator），未考虑多服务链中的**级联依赖与背压效应**
   - 虚拟化环境中未完全模拟物理层干扰（如 noisy neighbor）

3. **训练依赖高质量轨迹数据**
   - 模型泛化能力依赖于训练数据覆盖的多样性，在全新业务模式下可能需重新训练

---

### 🔮 未来工作方向

1. **轻量化部署方案**
   - 探索模型量化（quantization）、知识蒸馏（knowledge distillation）以适配嵌入式边缘节点

2. **扩展至多智能体框架（MARL）**
   - 构建 Multi-Agent RL 框架协调上下游服务的协同扩缩容，应对微服务链依赖问题

3. **集成能耗作为优化目标**
   - 将 Energy Consumption 纳入 reward function，构建绿色节能的 autoscaling 策略

4. **真实 6G 物理测试床验证**
   - 在 AI-RAN 架构下评估无线接入网动态对控制环路的影响（如信道波动、移动性）

5. **在线持续学习机制**
   - 支持模型在生产环境中进行安全的在线微调，适应不断演化的业务模式

---

> 🔗 **代码与数据公开**：  
> - 项目仓库：[https://github.com/farazshaikh581/Autoscaling_mitigating-temporal-blindness](https://github.com/farazshaikh581/Autoscaling_mitigating-temporal-blindness)  
> - 工作负载数据来源：[Azure Functions Trace 2021](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md)

</details>

---

### 13. [1.5 Million Messages Per Second on 3 Machines: Benchmarking and Latency Optimization of Apache Pulsar at Enterprise Scale](https://arxiv.org/abs/2603.29113)

**Authors**: Muhamed Ramees Cheriya Mukkolakkal  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.29113v1  

#### Abstract
This paper presents two independent contributions for Apache Pulsar practitioners. First, we validate 1,499,947 msg/s at 3.88 ms median publish latency on just three bare-metal Kubernetes nodes running Pulsar 4.0.8 with Java 21 and ZGC Generational garbage collection, and project a hardware-driven p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*1.5 Million Messages Per Second on 3 Machines: Benchmarking and Latency Optimization of Apache Pulsar at Enterprise Scale*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对 **Apache Pulsar** 在企业级生产环境中出现的高发布延迟（13–18 ms）和间歇性长尾延迟（高达 213 ms）的问题，进行了系统性的根因分析与优化。尽管集群负载较低（仅 700–9,000 msg/s），但性能表现远未达到硬件潜力。

### 提出了什么新方法或新思路
- **基于 JFR（Java Flight Recorder）的线上实时诊断方法**：在不中断流量的前提下，对运行中的 Bookie 节点进行深度性能剖析，识别出三个独立的延迟根源。
- **发现并揭示了一个此前未被记录的 Linux 内核行为**：即多个物理上分离的 NVMe 设备共享内核块层（block layer）、bio 分配池和 IRQ 处理机制，导致 BookKeeper 的 `ForceWriteThread` 中 `fdatasync` 性能从 <1 ms 恶化至 15–22 ms。
- **提出了一套完整的端到端优化方案**，涵盖 JVM GC 策略、操作系统调优、写缓存刷新策略及硬件配置建议。

### 相比现有方法的优势
- **无需外部负载均衡器**：利用 Pulsar 原生的 key-based partition routing 实现水平扩展。
- **全链路可复制性**：所有优化均基于标准 Kubernetes + Bare Metal 架构，无需定制硬件或修改 Pulsar 源码。
- **显著提升吞吐与降低延迟**：实现 1.5M msg/s 吞吐量，中位发布延迟降至 3.88 ms，相比原始环境提升 4.7 倍，并保留 65–82% 非网络资源余量。
- **首次文档化 kernel page cache writeback 对 BookKeeper 的影响**，为社区提供重要实践参考。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **真实生产数据流特征模拟**：使用典型的企业事件流模式（如物理安防设备事件），通过压测工具生成恒定消息速率。
- **无公开数据集引用**，实验基于自建压测环境，消息大小和结构符合实际业务场景。

### 实验设置和评估指标
#### 硬件配置
- **3 台裸金属服务器**，每台配备：
  - 专用 NVMe 日志盘（journal disk），fsync 延迟低至 0.02 ms
  - 10 Gbps NIC
  - Kubernetes 部署，每个节点运行 2 个 bookie（60 Gi RAM）和 2 个 broker（32 Gi RAM）
  - Pulsar 版本：4.0.8，Java 21，启用 ZGC Generational

#### 软件配置
- Topic 设置：128 partitions，E=3/Qw=2/Qa=2（三副本，写入等待两个确认）
- 关键参数调整：
  - `flushInterval=30s`
  - `groupWaitMSec=1`
  - OS 内核调优：`vm.dirty_ratio=2`, `vm.dirty_background_ratio=1` 等
  - JVM GC：ZGC Generational（`-XX:+UseZGC -XX:+ZGenerational`）

#### 评估指标
| 指标 | 描述 |
|------|------|
| Throughput | 每秒处理的消息数（msg/s） |
| Publish Latency P50/P95/P99/P99.9 | 发布延迟的百分位值 |
| End-to-end Latency | 从发布到消费的端到端延迟 |
| Resource Headroom | CPU、内存等资源利用率余量 |
| Failure Count | 运行期间失败次数 |

### 基线方法对比
| 基线配置 | 描述 |
|--------|------|
| 生产基线（Production Baseline） | G1GC + 32GB heap + SSD journal（5.1 ms fsync）+ 默认 flushInterval=60s |
| 不同 GC 配置对比 | G1GC vs ZGC（仅 bookie）vs ZGC（bookie & broker） |
| 不同 flushInterval 对比 | 60s（默认） vs 30s vs 15s |
| 优化前后对比 | 表 V 展示了各组件延迟分解的“Before vs After” |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **峰值吞吐量**：**1,499,947 msg/s**（约 1.5M msg/s）
- **持续时间**：稳定运行 **10 分钟**，零失败
- **发布延迟（Publish Latency）**
  - P50: **3.88 ms**
  - P95: ~5.5 ms
  - P99: **6.5 ms**
  - P99.9: **8 ms**
- **端到端延迟（End-to-end）**
  - P50: 14.5 ms
  - P99: 25 ms
- **网络瓶颈**：达到 **8.4 Gbps**（占 10 Gbps 的 84%），成为唯一瓶颈
- **非网络资源余量**：CPU 和内存仍有 **65–82% headroom**

### 与基线方法的对比结果
| 维度 | 优化前（生产） | 优化后（本文） | 提升幅度 |
|------|----------------|---------------|----------|
| Throughput | 30k msg/s | 1.5M msg/s | **50×** |
| Publish P50 Latency | 18.1 ms | 3.88 ms | **下降 79%** |
| GC Spikes | 最高 213 ms | 完全消除 | ✅ 消除长尾 |
| Journal fsync | 5.1 ms（SSD） | 0.02 ms（NVMe） | 99% 改善 |
| Consumer P99 | 15–197 ms | 14–18 ms | 更稳定 |

### 消融实验结果（Ablation Study）
论文通过多组控制变量实验验证各项优化效果：

#### （1）GC 算法对比（Table III）
| 配置 | GC 类型 | Publish P99 | Consumer P99 |
|------|--------|-------------|--------------|
| A | G1GC（both） | 17–55 ms | 15–197 ms |
| B | ZGC（bookie only） | 5.5–21 ms | 15–36 ms |
| C | ZGC（both） | 7.7–38 ms | **14–18 ms** |

> 结论：ZGC 显著改善消费者延迟稳定性；同时应用于 bookie 和 broker 效果最佳。

#### （2）Flush Interval 对比（Table IV）
| 间隔 | P50 | P99 | P99.9 |
|------|-----|-----|-------|
| 60s（default） | 2.20 ms | 6.09 ms | 118.8 ms |
| 30s（chosen） | **1.42 ms** | 5.91 ms | 104.5 ms |
| 15s | 1.46 ms | **17.0 ms** | **45.3 ms** |

> 结论：30 秒为最优平衡点，P50 下降 35%，而 15 秒反而引发 P99 恶化。

#### （3）综合优化前后对比（Table V）
详细拆解了五大延迟组件的改进：
- Journal fsync：5.1 → 0.02 ms（↓99%）
- groupWaitMSec：2.0 → 1.0 ms（↓50%）
- BookKeeper processing：6.5 → 1.86 ms（↓71%）
- Broker + network：4.5 → 1.0 ms（↓78%）
- 总体 P50：18.1 → 3.88 ms（↓79%）

---

## 4. 关键结论和发现

### 论文的主要发现
1. **三大独立延迟根源被识别并解决**：
   - **G1GC 大堆暂停**（32GB 堆触发 213 ms spike）→ 改用 **ZGC Generational** 彻底消除。
   - **磨损 SSD 上 journal fdatasync 平均 5.1 ms** → 使用专用 NVMe 日志盘可降至 **0.02 ms**。
   - **Linux 内核 page cache writeback 与 BookKeeper ForceWriteThread 的交互问题**：即使 journal 与 ledger 存放于不同 NVMe 盘，仍因共享 kernel block layer 导致 `fdatasync` 恶化至 15–22 ms —— 此为**首次披露的新现象**。

2. **优化后系统具备强大横向扩展能力**：
   - 当前 3 节点已达 1.5M msg/s，受限于 10 Gbps 网络。
   - 升级至 25 Gbps NIC 可轻松达到 **3M msg/s**。
   - 采用 **Partition Federation 架构**（5 个独立 3 节点集群共享一个 128 分区 topic），预计可达 **15M msg/s（~1.3 trillion/day）**。

3. **推荐架构完全去中心化**：
   - 无需外部 LB
   - 利用 Pulsar 原生 Key Shared Subscription 实现 per-account ordering
   - namespace bundles（numBundles=256, auto-split）实现细粒度负载均衡

### 方法的局限性
- **依赖高性能 NVMe 和高速网络**：优化效果高度依赖底层硬件质量，老旧 SSD 或 HDD 场景收益有限。
- **Kernel 层面问题难以绕过**：多个 NVMe 共享 block layer 的限制是操作系统层级设计，无法通过应用层规避。
- **未测试跨地域部署**：当前架构假设集群本地化，未涉及 geo-replication 或 WAN 延迟场景。

### 未来工作方向
- 探索更激进的 OS 层优化，例如使用 `io_uring` 替代传统 syscalls。
- 研究将 journal 放置在持久化内存（PMEM）上的可行性以进一步降低 fsync 开销。
- 扩展 benchmark 至更大规模（>15 节点）验证线性扩展性。
- 将 `ForceWriteThread` 的同步逻辑与 kernel writeback 解耦，提交社区 patch。

---

> **总结一句话**：  
> 本文通过 **JFR + OS + JVM + Hardware** 四维联动优化，在 3 台裸金属机器上实现了 **1.5M msg/s 吞吐 + 3.88 ms 中位延迟**，并揭示了影响 Pulsar 性能的关键隐藏因素 —— **kernel page cache writeback 对 fdatasync 的干扰**，为企业级 Pulsar 部署提供了极具价值的调优指南。

</details>

---

### 14. [Big2Small: A Unifying Neural Network Framework for Model Compression](https://arxiv.org/abs/2603.29768)

**Authors**: Jing-Xiao Liao, Haoran Wang, Tao Li, Daoming Lyu, Yi Zhang, Chengjun Cai, Feng-Lei Fan  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.29768v1  

#### Abstract
With the development of foundational models, model compression has become a critical requirement. Various model compression approaches have been proposed such as low-rank decomposition, pruning, quantization, ergodic dynamic systems, and knowledge distillation, which are based on different heuristic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Big2Small: A Unifying Neural Network Framework for Model Compression**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前模型压缩领域存在多种技术（如低秩分解、剪枝、量化、知识蒸馏等），但这些方法大多基于不同的启发式策略，缺乏统一的理论框架。这导致研究碎片化，难以系统化设计和比较不同压缩算法。

本文旨在解决以下两个核心问题：
- **理论层面**：能否将主流的模型压缩方法统一在一个数学框架下？
- **实践层面**：能否提出一种高效、无需原始训练数据（data-free）的通用压缩框架？

---

### 🚀 提出的新方法与新思路

#### （1）**统一的理论框架：基于测度论的模型压缩理论**
- 提出 **Universal Compressibility Theorem** 和 **Structural Equivalence Theorem**，从数学上证明：
  - 所有主流压缩方法（低秩、量化、剪枝、遍历动态系统、知识蒸馏）本质上都是对参数集合进行“映射-重构”操作，并满足误差约束。
  - 每种压缩方法在数学上等价于一个特定结构的神经网络加上某种正则化项。
- 这是首次为模型压缩建立统一的**形式化数学基础**，推动该领域从“经验驱动”走向“原理驱动”。

#### （2）**提出 Big2Small：基于 INR 的新型压缩范式**
- 将 **Implicit Neural Representation (INR)** 的思想从图像/信号压缩迁移到**神经网络参数压缩**。
- 核心思想：用一个小的 INR 网络来编码大模型的权重张量，在推理时通过 INR 动态重建原始权重。
- 实现“**压缩-解压**”架构（Compression-Decompression），实现真正的存储节省。

#### （3）关键技术增强
- **Outlier-Aware Preprocessing**：识别并单独存储极端权重值（outliers），避免其影响 INR 学习。
- **Frequency-Aware Loss Function**：结合 MSE、Gradient Difference 和 Focal Frequency Loss，提升高频细节重建能力。

---

### 🔍 相比现有方法的优势

| 维度 | Big2Small 的优势 |
|------|------------------|
| **理论统一性** | 首次将五类压缩方法统一于同一数学框架下，提供全新视角 |
| **无需数据** | 完全 data-free，不依赖原始训练数据或合成数据，保护隐私 |
| **灵活性高** | 可与其他压缩方法（如量化、剪枝）正交组合，进一步提升压缩率 |
| **重建保真度高** | 利用 INR 强大的连续函数拟合能力，保持权重分布和高频特征 |
| **可扩展性强** | 同一套框架适用于 CNN、Transformer 等多种架构 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与模型
- **分类任务**：ImageNet 数据集上的主流模型
  - ResNet18, ResNet50
  - Swin-T, Swin-S
- **分割任务**：Carvana Image Masking Challenge 数据集
  - UNet, UNet++, R2UNet

所有预训练模型来自 `timm` 库。

---

### ⚙️ 实验设置与评估指标

#### 主要设置
- **压缩方式**：逐层压缩卷积和线性层（>100KB）
- **INR 结构**：双 MLP 架构（Synthesis + Modulation），使用 SIREN 激活函数 + Positional Encoding
- **训练配置**：
  - 优化器：AdamW
  - 学习率：1e-3，调度器：CosineAnnealingLR
  - 训练轮数：10,000 epochs
  - 可选后处理：对 INR 参数进行量化（如 6bit）

#### 评估指标
| 任务 | 指标 |
|------|------|
| 图像分类 | Top-1 Accuracy (%) |
| 图像分割 | mIOU (%), mACC (%) |
| 权重重建质量 | MAE, Cosine Similarity, PSNR, Q-Q Plot |
| 压缩效率 | Compression Ratio, Model Size (MB) |
| 推理开销 | Inference Latency (throughput) |

---

### 🆚 基线方法对比
- **DFMC（Data-Free Model Compression）方法**：
  - **DSG**, **Squant**, **UDFC**：基于生成样本的量化方法
  - **RieM** [56]：当前最先进的 data-free 压缩方法，使用黎曼动力学建模权重
- **其他 INR 编码器对比**：
  - SIREN, VAE, GAN —— 用于验证所提 INR 结构的有效性

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（ImageNet 分类）

#### ResNet18
| 方法 | Size (MB) | W/A Bits | CR | Top-1 Acc (%) |
|------|-----------|----------|-----|----------------|
| 原始模型 | 44.6 | 32/32 | – | 71.47 |
| DSG | 8.4 | 6/6 | 5.3 | 70.46 |
| Squant | 8.4 | 6/6 | 5.3 | 70.74 |
| UDFC | 8.4 | 6/6 | 5.3 | 72.76 |
| **Big2Small-6bit** | **7.6** | **6/6** | **5.9** | **71.24** |

> ✅ 在更高压缩比下仍保持竞争力，优于多数 baseline。

#### ResNet50
| 方法 | Size (MB) | W/A Bits | CR | Top-1 Acc (%) |
|------|-----------|----------|-----|----------------|
| 原始模型 | 97.5 | 32/32 | – | 77.72 |
| RieM | 12.3 | 4/4 | 7.9 | 73.26 |
| **Big2Small-8bit** | **12.6** | **8/8** | **7.7** | **73.98** |

> ✅ 虽然压缩比略低，但准确率显著领先（+0.72% vs RieM）。

#### Swin-T
| 方法 | Size (MB) | W/A Bits | CR | Top-1 Acc (%) |
|------|-----------|----------|-----|----------------|
| RieM | 14.5 | 4/8 | 8.0 | 76.30 |
| **Big2Small-6bit** | **14.6** | **6/6** | **7.9** | **77.25** |

> ✅ 在相近大小下达到最佳精度。

---

### 🧪 图像分割结果（Carvana 数据集）

以 UNet 为例：

| 方法 | Size (MB) | mIOU (%) |
|------|-----------|---------|
| 原始模型 | 51.5 | 95.87 |
| RieM | 25.8 | 94.67 |
| **Big2Small-32bit** | **22.9** | **95.32** |

> ✅ 更小体积 + 更高性能，尤其在细粒度结构（如天线）恢复上表现更优（见 Fig. 5）。

---

### 🔬 消融实验结果（Ablation Study）

使用 ResNet50 卷积层进行分析：

| 策略 | Rel MAE ↓ | Cosine ↑ | PSNR ↑ |
|------|----------|--------|-------|
| w/o Preprocessing | 0.0832 | 0.953 | 29.32 |
| **With Preprocessing** | **0.0085** | **1.000** | **54.21** |
| MSE Loss | 0.0672 | 0.996 | 34.29 |
| MSE+Grad | 0.0421 | 0.996 | 35.21 |
| MSE+Freq | 0.0351 | 0.996 | 35.52 |
| **Proposed (Ours)** | **0.0085** | **1.000** | **54.21** |

> ✅ Outlier-Aware Preprocessing 和 Frequency-Aware Loss 显著提升重建质量。

---

### 🔗 与其他压缩方法的组合效果（Table V）

| 方法 | Size (MB) | CR | Top-1 Acc (%) |
|------|-----------|----|----------------|
| Big2Small | 30.4 | 1.5 | 71.85 |
| +SVD | 10.32 | 2.9 | 70.32 |
| +Pruning (60%) | 9.12 | 4.9 | 69.82 |
| **+Quantization (6bit)** | **7.59** | **5.9** | **71.24** |

> ✅ 与量化结合效果最好，实现高压缩比且精度损失极小。

---

### ⏱️ 推理延迟（Inference Latency）
- 使用 NVIDIA RTX PRO 6000 GPU 测试
- Big2Small 比原模型慢约 **30%**
- 原因：需运行多个 INR 实例重建权重
- 适用场景：非实时边缘部署（storage-constrained > latency-constrained）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **理论层面突破**：
   - 成功构建首个统一的模型压缩数学框架，揭示了不同压缩方法之间的内在联系。
   - 证明所有压缩方法均可视为某种“带正则化的神经网络映射”，为未来算法设计提供了原则性指导。

2. **实践层面创新**：
   - 提出 **Big2Small**，首次将 INR 应用于神经网络参数压缩，开辟了新的压缩范式。
   - 实现高质量、data-free、灵活可组合的压缩方案，在多个任务和架构上达到 SOTA 性能。

3. **技术有效性验证**：
   - Outlier-Aware Preprocessing 和 Frequency-Aware Loss 显著提升重建保真度。
   - 可无缝集成量化、剪枝等其他方法，具备良好扩展性。

---

### ⚠️ 方法的局限性
1. **推理延迟增加**：由于需要在线重建权重，推理速度下降约 30%，不适合低延迟场景。
2. **训练成本较高**：每层独立训练 INR，总训练时间较长（万级 epoch）。
3. **未覆盖 BN 层**：BatchNorm 参数未被压缩，因其统计特性对推理稳定性至关重要。
4. **极端稀疏或二值化支持有限**：目前主要面向浮点权重重建，对 BNN 支持不足。

---

### 🔮 未来工作方向
1. **优化训练效率**：
   - 设计共享参数的轻量级 INR 架构，减少 per-layer 训练开销。
2. **探索更高容量表示网络**：
   - 如 Polynomial Networks、Wavelet-based INR，进一步提升表达能力。
3. **实现实时推理加速**：
   - 开发专用硬件或缓存机制，提前重建部分权重以降低延迟。
4. **拓展至 LLM 压缩**：
   - 将 Big2Small 应用于大语言模型（LLMs），探索其在超大规模模型中的潜力。
5. **理论深化**：
   - 探索压缩极限（rate-distortion trade-off）、泛化边界等信息论性质。

---

## 总结

> **Big2Small** 不仅是一项新技术，更是一种新范式。它通过将 INR 引入模型压缩领域，实现了从“直接修改权重”到“学习权重的隐式表示”的跃迁。更重要的是，其背后的统一理论为整个模型压缩领域提供了坚实的数学基础，有望引领该领域进入一个更加系统化、可解释、可设计的新时代。

</details>

---

### 15. [Working Paper: Towards a Category-theoretic Comparative Framework for Artificial General Intelligence](https://arxiv.org/abs/2603.28906)

**Authors**: Pablo de los Riscos, Fernando J. Corbacho, Michael A. Arbib  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.28906v1  

#### Abstract
AGI has become the Holly Grail of AI with the promise of level intelligence and the major Tech companies around the world are investing unprecedented amounts of resources in its pursuit. Yet, there does not exist a single formal definition and only some empirical AGI benchmarking frameworks currentl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Working Paper: Towards a Category-theoretic Comparative Framework for Artificial General Intelligence*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在 **Artificial General Intelligence (AGI)** 领域存在一个根本性挑战：尽管已有多种 AGI 架构（如 Reinforcement Learning, Causal RL, Active Inference, AIXI, Schema-based Learning 等），但缺乏一个统一的、形式化的框架来**系统地描述、比较和分析这些架构之间的异同**。

具体而言，现有研究面临以下问题：
- 没有对“AGI 架构”本身的**正式定义**；
- 不同架构之间的关系（如是否等价、可转换、或一方是另一方的特例）无法被精确表达；
- 架构层面的结构性差异（如信息流、模块化程度、反馈机制）与算法实现细节混杂，难以分离；
- 缺乏一种数学语言来刻画架构的**结构性保证**（如收敛性、模因性、因果识别能力）及其在不同架构间的传递性。

### 提出了什么新方法或新思路
本文提出了一种基于 **Category Theory** 的全新理论框架 —— **ArchAgents**，用于构建一个通用的、代数式的 AGI 架构比较体系。

其核心思想包括：

#### （1）将“架构”视为一种代数理论
- 将每个 AGI 架构建模为一个 **free hypergraph category**，由三部分构成：
  - **Syntactic Structure**：定义接口（ports）、基本计算模块（generators）及连接模式（wiring diagrams）；
  - **Knowledge Architecture**：定义知识单元类型（knowledge types）和允许的知识变换（workflows）；
  - **Syntax-Knowledge Interface**：通过一个 **profunctor** 描述语法组件如何与知识资源交互。

#### （2）引入 **Grothendieck fibration** 结构
- 定义范畴 `ArchAgents`：对象是不同的 AGI 架构，态射是保持结构的架构间映射（translations）；
- 定义范畴 `Agents`：对象是具体实现的智能体，形成一个纤维范畴（fibration）`p: Agents → ArchAgents`；
- 这使得可以从一个架构中的智能体“拉回”（reindex）到另一个更抽象或受限的架构中，支持跨架构的行为迁移与比较。

#### （3）分层定义属性（Properties）
- **Structural Properties**：仅依赖于架构的图示结构（如是否存在反馈环、是否可分解）；
- **Informational Properties**：关于信息如何封装、传播和更新（如是否支持模块化知识）；
- **Semantic Properties**：通过 **institution** 和 **proof-carrying certificates** 形式化智能体行为的语义保证（如收敛性证明），并可在架构间传递。

#### （4）提出“架构约束”扩展（Architectural Constraints）
- 在附录中进一步提出将诸如 Bellman 方程、马尔可夫假设等数学条件作为独立的 **constraint schemas** 加入架构定义，避免将其混入语法等式中，提升表达力与不变性。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **抽象层次** | 聚焦具体算法或模型 | 抽象至架构级的代数结构 |
| **可比性** | 各自独立，难直接比较 | 支持形式化翻译与归约（如 RL 是 CRL 的退化） |
| **结构性分析** | 缺乏统一工具 | 可进行图示推理、等价判定、表达力排序 |
| **知识组织建模** | 忽略或隐含处理 | 显式建模知识类型、变换与访问模式 |
| **理论兼容性** | 通常局限于单一范式 | 可集成经典定理（如值函数收敛）作为语义证书 |

---

## 2. 核心实验方法和设置

> ⚠️ 注意：本论文是一篇 **working paper**，属于理论建构阶段，**并未包含传统意义上的“实验”**（即运行代码、训练模型、测试性能）。所谓的“案例研究”（case studies）是**形式化建模与概念分析**，而非实证实验。

### 使用了哪些“数据集”
无实际数据集。文中使用的“实例”均为理论构造：
- Reinforcement Learning (RL)
- Causal Reinforcement Learning (CRL)
- Schema-Based Learning (SBL)
- AIXI

这些不是真实世界的数据，而是作为**形式化建模的对象**，用以展示该框架的表达能力。

### 实验设置和评估指标
所谓“评估”是指对该框架能否准确捕捉不同架构的本质特征进行**概念验证**（conceptual validation）。

#### 分析维度：
1. 是否能形式化定义各架构的 `Syn`, `Know`, 和 `Profunctor`；
2. 是否能识别出关键的结构性差异（如反馈环数量、知识解耦程度）；
3. 是否能建立架构间的态射（如从 SBL 到 CRL 的遗忘映射）；
4. 是否能导出有意义的信息性属性（如“是否支持局部学习”、“是否具备知识复用”）。

#### 评估方式：
- **形式化推导**：使用 category theory 工具进行图示推理；
- **对比表格**：如 Table 1 对比 RL、CRL、SBL 的架构特性；
- **逐步松弛分析**（Stepwise Relaxation）：展示如何从 CRL 出发，通过逐步放松约束得到 SBL，体现架构演化路径。

### 基线方法对比
虽然没有运行程序，但进行了**理论层面的基线对比**：

| 架构 | 特征 |
|------|------|
| **RL** | 单一参数载体 `O`，全局更新，无知识模块化 |
| **CRL** | 引入因果模型 `Ocs`，双反馈环，有限解耦 |
| **SBL** | 多 schema 结构，局部更新，支持组合与创建新知识单元 |

并通过 `ArchAgents` 中的态射关系说明：**RL 和 CRL 可视为 SBL 在特定约束下的退化形式**。

---

## 3. 主要“实验结果”和性能指标

由于是理论工作，此处“结果”指**形式化建模成果与结构发现**。

### 关键建模成果
| 架构 | 成果摘要 |
|------|----------|
| **RL** | 成功建模标准 RL 流程：<br>`Policy`, `EnvInteraction`, `Update`<br>知识层仅含 `O → O` 更新 |
| **CRL** | 扩展为两个知识流：<br>`PolicyUpdate: O ⊗ Ocs ⊗ E → O`<br>`CausalUpdate: Ocs ⊗ E → Ocs`<br>体现因果干预（`Do` 操作） |
| **SBL** | 提出完整认知核（cognitive kernel）：<br>- `CogModActivate` / `CogModExec`<br>- 支持 `SchemaCreate`, `SchemaCombine`, `SchemaRefine`<br>- 明确 body-mind 分离 |
| **AIXI** | 建模 Solomonoff 先验与期望最大化策略<br>引入 `UniversalPrior`, `KernelMixing`, `PosteriorUpd` 等知识操作 |

### 与基线方法的形式化对比结果
| 比较项 | 结果 |
|-------|------|
| **表达力排序** | `SBL > CRL > RL`<br>SBL 支持最多样的知识操作与模块化 |
| **反馈结构** | RL: 单环；CRL: 双耦合环；SBL: 多解耦环 |
| **知识更新粒度** | RL: 全局；CRL: 角色级；SBL: schema 局部 |
| **信息复用能力** | RL: 无；CRL: 有限；SBL: 支持组合与重用 |

### 消融实验（概念性）
文中虽未称“消融”，但通过 **stepwise relaxation** 实现了类似逻辑：

> 从 CRL 出发，依次放松以下约束，最终逼近 SBL：
1. **Factorization of Interfaces**：状态/动作分解为因子
2. **Typed Multi-Model Architecture**：允许多个预测模型共存
3. **Cognitive Modules**：引入异构内部处理单元
4. **Temporal Decoupling and Memory**：加入显式记忆机制
5. **Body-Mind Mediation**：区分感知/执行接口与内部表示

每一步都对应一个架构态射 `f: A → B`，体现了**架构演化路径**。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **AGI 架构可以被统一建模为 hypergraph categories**，从而获得严格的数学基础；
2. ✅ **不同 AGI 范式之间存在形式化的翻译关系**（morphisms），例如 CRL 可视为 RL 的增强版本，而 SBL 是前者的泛化；
3. ✅ **知识管理应作为独立维度建模**，不能仅由语法流程决定；
4. ✅ **架构的表达力可通过其支持的知识操作丰富度衡量**（如是否支持创建、组合、删除 schema）；
5. ✅ **Category Theory 与 AGI 存在强烈共生关系**：AGI 需要 CT 提供抽象工具，而 AGI 的复杂性也将推动 CT 发展。

### 方法的局限性
1. **尚未完全形式化语义约束**：如 Bellman 方程、最优性条件目前仅作为附录设想，尚未整合进主框架；
2. **缺乏动态演化建模**：当前框架静态描述架构，未涉及架构自身如何随时间演进（如 self-modifying systems）；
3. **实现复杂度高**：要将现有深度学习系统映射到此框架，需大量工程与形式化工作；
4. **暂无自动化工具支持**：缺少图形编辑器、类型检查器或推理引擎辅助使用。

### 未来工作方向（原文 Section 7）
| 时间尺度 | 方向 |
|--------|------|
| **Very Short-Term** | 明确 `ArchAgents` 中态射的具体定义；深化属性分类 |
| **Short-Term** | 引入类型本体（ontology of types）；支持 algebraic theories（如 Bellman eqs）作为约束 |
| **Mid-Term** | 建立 `World` 范畴以支持环境建模与实验评估；发展 empirical measurement 接口 |
| **Long-Term** | 提炼 AGI 的必要/充分架构原则；构建协作平台共享架构与测试结果 |

---

## 总结
本文提出了首个基于 **Category Theory** 的 AGI 架构统一比较框架 **ArchAgents**，其核心贡献在于：
- 将“架构”本身形式化为代数结构；
- 区分语法、知识、接口三层；
- 构建 `Agents → ArchAgents` 的纤维结构；
- 支持跨架构推理与属性传递。

虽然尚处理论初期，无传统实验验证，但它为未来 AGI 理论研究提供了强大的**元语言工具**，有望成为连接不同智能范式的基础桥梁。

</details>

---

### 16. [One-for-All: A Lightweight Stabilized and Parameter-Efficient Pre-trained LLM for Time Series Forecasting](https://arxiv.org/abs/2603.29756)

**Authors**: Prasanjit Dey, Soumyabrata Dev, Bianca Schoen-Phelan  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.29756v1  

#### Abstract
We address the challenge of adapting pre-trained Large Language Models (LLMs) for multivariate time-series analysis, where their deployment is often hindered by prohibitive computational and memory demands. Our solution, One-for-All, introduces Gaussian Rank-Stabilized Low-Rank Adapters (rsLoRA) to ...

---

### 17. [Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research](https://arxiv.org/abs/2603.28986)

**Authors**: Martin Legrand, Tao Jiang, Matthieu Feraud, Benjamin Navet, Yousouf Taghzouti, Fabien Gandon, Elise Dumont, Louis-F\'elix Nothias  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.28986v1  

#### Abstract
Current Autonomous Scientific Research (ASR) systems, despite leveraging large language models (LLMs) and agentic architectures, remain constrained by fixed workflows and toolsets that prevent adaptation to evolving tasks and environments. We introduce Mimosa, an evolving multi-agent framework that ...

---

### 18. [REFINE: Real-world Exploration of Interactive Feedback and Student Behaviour](https://arxiv.org/abs/2603.29142)

**Authors**: Fares Fawzi, Seyed Parsa Neshaei, Marta Knezevic, Tanya Nazaretsky, Tanja K\"aser  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.29142v1  

#### Abstract
Formative feedback is central to effective learning, yet providing timely, individualised feedback at scale remains a persistent challenge. While recent work has explored the use of large language models (LLMs) to automate feedback, most existing systems still conceptualise feedback as a static, one...

---

### 19. [Optimizing Donor Outreach for Blood Collection Sessions: A Scalable Decision Support Framework](https://arxiv.org/abs/2603.29643)

**Authors**: Andr\'e Carneiro, Pedro T. Monteiro, Rui Henriques  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.29643v1  

#### Abstract
Blood donation centers face challenges in matching supply with demand while managing donor availability. Although targeted outreach is important, it can cause donor fatigue via over-solicitation. Effective recruitment requires targeting the right donors at the right time, balancing constraints with ...

---

### 20. [Symphony for Medical Coding: A Next-Generation Agentic System for Scalable and Explainable Medical Coding](https://arxiv.org/abs/2603.29709)

**Authors**: Joakim Edin, Andreas Motzfeldt, Simon Flachs, Lars Maal{\o}e  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.29709v1  

#### Abstract
Medical coding translates free-text clinical documentation into standardized codes drawn from classification systems that contain tens of thousands of entries and are updated annually. It is central to billing, clinical research, and quality reporting, yet remains largely manual, slow, and error-pro...

---

### 21. [Dual Perspectives in Emotion Attribution: A Generator-Interpreter Framework for Cross-Cultural Analysis of Emotion in LLMs](https://arxiv.org/abs/2603.29077)

**Authors**: Aizirek Turdubaeva, Uichin Lee  
**Category**: cs.CL  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.29077v1  

#### Abstract
Large language models (LLMs) are increasingly used in cross-cultural systems to understand and adapt to human emotions, which are shaped by cultural norms of expression and interpretation. However, prior work on emotion attribution has focused mainly on interpretation, overlooking the cultural backg...

---

### 22. [SNEAK: Evaluating Strategic Communication and Information Leakage in Large Language Models](https://arxiv.org/abs/2603.29846)

**Authors**: Adar Avsian, Larry Heck  
**Category**: cs.CL  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.29846v1  

#### Abstract
Large language models (LLMs) are increasingly deployed in multi-agent settings where communication must balance informativeness and secrecy. In such settings, an agent may need to signal information to collaborators while preventing an adversary from inferring sensitive details. However, existing LL...

---

### 23. [Parallel Gauss-Jordan Elimination and System Reduction for Efficient Circuit Simulation](https://arxiv.org/abs/2603.28792)

**Authors**: Filip Noveski, Elena Hadzieva  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.28792v1  

#### Abstract
For the purposes of electric circuit simulation, we consider an iterative simulation model based on solving systems of linear equations by Gauss-Jordan elimination (GJE) for individual moments in time. To accelerate the simulation, we propose two independent novel approaches: a parallel GJE algorith...

---

### 24. [Foundations of Polar Linear Algebra](https://arxiv.org/abs/2603.28939)

**Authors**: Giovanni Guasti  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.28939v1  

#### Abstract
This work revisits operator learning from a spectral perspective by introducing Polar Linear Algebra, a structured framework based on polar geometry that combines a linear radial component with a periodic angular component. Starting from this formulation, we define the associated operators and analy...

---

### 25. [\texttt{ReproMIA}: A Comprehensive Analysis of Model Reprogramming for Proactive Membership Inference Attacks](https://arxiv.org/abs/2603.28942)

**Authors**: Chihan Huang, Huaijin Wang, Shuai Wang  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.28942v1  

#### Abstract
The pervasive deployment of deep learning models across critical domains has concurrently intensified privacy concerns due to their inherent propensity for data memorization. While Membership Inference Attacks (MIAs) serve as the gold standard for auditing these privacy vulnerabilities, conventional...

---

### 26. [SciVisAgentBench: A Benchmark for Evaluating Scientific Data Analysis and Visualization Agents](https://arxiv.org/abs/2603.29139)

**Authors**: Kuangshi Ai, Haichao Miao, Kaiyuan Tang, Nathaniel Gorski, Jianxin Sun, Guoxi Liu, Helgi I. Ingolfsson, David Lenz, Hanqi Guo, Hongfeng Yu, Teja Leburu, Michael Molash, Bei Wang, Tom Peterka, Chaoli Wang, Shusen Liu  
**Category**: cs.AI  
**Published**: 2026-04-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.29139v1  

#### Abstract
Recent advances in large language models (LLMs) have enabled agentic systems that translate natural language intent into executable scientific visualization (SciVis) tasks. Despite rapid progress, the community lacks a principled and reproducible benchmark for evaluating these emerging SciVis agents...

---

### 27. [PolarQuant: Optimal Gaussian Weight Quantization via Hadamard Rotation for LLM Compression](https://arxiv.org/abs/2603.29078)

**Authors**: Caio Vicentino  
**Category**: cs.CL  
**Published**: 2026-04-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.29078v1  

#### Abstract
We present PolarQuant, a post-training weight quantization method for large language models (LLMs) that exploits the distributional structure of neural network weights to achieve near-lossless compression. PolarQuant operates in three stages: (1) block-wise normalization to the unit hypersphere, (2)...

---

### 28. [Mathematical Foundations of Modeling ETL Process Chains](https://arxiv.org/abs/2603.29877)

**Authors**: Levin Maier, Lucas Schulze, Robert Lilow, Lukas Hahn, Nikola Krasowski, Arnulf Barth, Sebastian Gaebel, Ferdi G\"uran, Oliver Hanau, Giovanni Wagner, Falk Borgmann, Oleg Arenz, Jan Peters  
**Category**: cs.DC  
**Published**: 2026-04-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.29877v1  

#### Abstract
Extract-Transform-Load (ETL) processes are core components of modern data processing infrastructures. The throughput of processed data records can be adjusted by changing the amount of allocated resources, i.e.~the number of parallel processing threads for each of the three ETL phases, but also depe...

---

### 29. [Realistic Market Impact Modeling for Reinforcement Learning Trading Environments](https://arxiv.org/abs/2603.29086)

**Authors**: Lucas Riera Abbade, Anna Helena Reali Costa  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.29086v1  

#### Abstract
Reinforcement learning (RL) has shown promise for trading, yet most open-source backtesting environments assume negligible or fixed transaction costs, causing agents to learn trading behaviors that fail under realistic execution. We introduce three Gymnasium-compatible trading environments -- MACE (...

---

### 30. [Causality-inspired Federated Learning for Dynamic Spatio-Temporal Graphs](https://arxiv.org/abs/2603.29384)

**Authors**: Yuxuan Liu, Wenchao Xu, Haozhao Wang, Zhiming He, Zhaofeng Shi, Chongyang Xu, Peichao Wang, Boyuan Zhang  
**Category**: cs.LG  
**Published**: 2026-04-01  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.29384v1  

#### Abstract
Federated Graph Learning (FGL) has emerged as a powerful paradigm for decentralized training of graph neural networks while preserving data privacy. However, existing FGL methods are predominantly designed for static graphs and rely on parameter averaging or distribution alignment, which implicitly ...

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
