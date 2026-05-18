# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-18 09:04:17 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [PSD: Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding](https://arxiv.org/abs/2605.15609)

**Authors**: Shengyin Sun, Yiming Li, Renxi Liu, Xinqi Li, Hui-Ling Zhen, Weizhe Lin, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Chen Ma  
**Category**: cs.CL  
**Published**: 2026-05-18  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2605.15609v1  

#### Abstract
Diffusion large language models (dLLMs) generate text by iteratively denoising masked token sequences. Although dLLMs can predict all masked positions in parallel within each step, the large number of denoising iterations still makes inference expensive. This cost can be reduced spatially by unmaski...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PSD: Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Diffusion Large Language Models (dLLMs) 虽然通过并行去噪机制在每一步中可同时预测多个被掩码的 token，从而具备天然的并行生成潜力，但其推理过程仍需大量迭代步骤，导致**高延迟和低硬件利用率**。现有的加速方法存在以下瓶颈：

- **Parallel Decoding** 方法通过每步解码多个 token 来压缩空间维度（spatial dimension），但过度并行会导致低置信度 token 过早提交，引发错误传播，造成质量显著下降。
- **Speculative Decoding** 方法通过推测未来去噪步骤并在一次验证调用中批量处理来压缩时间维度（temporal dimension），但通常仅支持单个 token 解码，未能充分利用 dLLMs 的并行能力。

因此，单一轴向优化面临**速度-质量帕累托前沿（Pareto frontier）的天花板**。

---

### 提出的新方法：Parallel Speculative Decoding (PSD)

PSD 是一种**无需训练、策略无关（policy-agnostic）的框架**，首次将空间并行解码与时间推测验证**统一于一个解码流程中**，实现双轴协同加速。

#### 核心思想
观察到：在 dLLM 去噪过程中，高置信度位置的排序具有**跨步骤稳定性**。基于此，可在不增加模型调用的情况下构建多深度推测草案。

#### 三阶段架构
1. **Spatial Parallel Unmasking**  
   使用任意转移策略（transfer policy）在每步中解码多个 token（如 Confidence Thresholding 或 LocalLeap）。
   
2. **Temporal Speculative Drafting**  
   利用当前步的置信度分数，按降序填充剩余掩码位置，构建一个有向无环图（DAG）形式的多深度推测草案集合，无需额外前向传播。

3. **Batched Verification with Hierarchical Acceptance**  
   将所有草案拼接为一批，在一次前向传播中验证；采用分层接受机制，选择**最深且与验证器预测一致的分支**，防止错误传播。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **通用性** | 可插拔任意 parallel decoding 策略作为 backbone，兼容性强 |
| **效率** | 同时压缩空间与时间维度，实现复合加速，突破单轴上限 |
| **质量保持** | 分层验证机制有效过滤错误推测，避免因激进并行导致的质量崩溃 |
| **零训练成本** | 完全无需微调或训练，即插即用 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **GSM8K**：1,319 道小学数学应用题，测试多步推理能力，使用 chain-of-thought 提示，报告 **Accuracy (%)**
- **HumanEval**：164 个手写 Python 编程任务，通过执行隐藏单元测试评估，报告 **pass@1 (%)**
- **MBPP**：500 个众包 Python 编程任务，更短更多样，报告 **pass@1 (%)**

---

### 实验设置
- **模型**：在三个开源 dLLMs 上进行评估：
  - **Dream-v0-Base-7B**（AR 初始化）
  - **LLaDA-1.5**（从头预训练 + 偏好对齐）
  - **openPangu-7B-Diffusion-Base**（持续预训练 + 块扩散）
- **块大小（Block Size）**：32
- **最大生成长度**：512
- **硬件平台**：单节点 8× NVIDIA V100 GPU
- **实现基础**：PyTorch + HuggingFace Transformers

---

### 评估指标
- **Tokens Per Forward Pass (TPF)**：平均每轮前向传播解码的 token 数量，衡量推理效率
- **Accuracy / pass@1**：任务性能指标
- 所有方法均启用 EOS 提前终止，并只统计 EOS 之前的 token 以公平比较吞吐量

---

### 基线方法对比
共对比 **7 种代表性基线**：

#### 空间并行方法（Spatial Parallel Decoding）
| 方法 | 特点 |
|------|------|
| Greedy Decoding | 每步仅解码最高置信度 token，作为质量基准 |
| Confidence (T=0.9) | 解码所有置信度 > 0.9 的位置 |
| LocalLeap | 基于局部确定性传播解码邻域 token |
| LoPA (k=1,3,5,7) | 探索 k 个候选填充顺序，选择长期并行潜力最高的 |
| ETE (k=1,3,5,7) | 跨块并行探索不确定性 token |
| FDM (k=1,3,5,7) | 同时预测当前与未来 k 步分布 |

#### 时间推测方法（Temporal Speculation）
| 方法 | 特点 |
|------|------|
| Spiffy (d=1,3,5,7) | 自推测解码，支持多步推测验证，是唯一的 temporal-only 基线 |

#### PSD 变体
- **PSD (Confidence, d)**：以 Confidence 为 spatial backbone
- **PSD (LocalLeap, d)**：以 LocalLeap 为 spatial backbone
- 其中 `d ∈ {1,3,5,7}` 表示推测深度

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **最高可达 5.5× TPF**（tokens per forward pass）
- 在大多数配置下，**准确率损失小于 1 个百分点**，接近 greedy decoding 水平
- 在 GSM8K 上，PSD (Confidence, d=7) 达到 **4.0× TPF @ 78.9% 准确率**，而 greedy 仅为 1.0×
- 在 MBPP 上，PSD (Confidence) 实现 **4.9× TPF**，远超 Spiffy 的 ~2.0×

---

### 与基线方法的对比结果
| 对比维度 | 结果 |
|--------|------|
| **vs. Spiffy (temporal-only)** | PSD 在相同质量下提供 **2–2.5× 更高的 TPF**，表明空间并行带来额外增益 |
| **vs. LoPA/FDM (spatial-only)** | 当 parallelism 相当时，PSD 显著优于基线质量：<br>例如在 HumanEval 上，LoPA k=5 达 4.9× TPF 但 pass@1 降至 23.8%，而 PSD (Confidence) 在 4.2× 下保持 35.4%（无损） |
| **vs. Greedy** | PSD 实现 **3–5.5× 加速**，同时保持几乎相同的输出质量 |

---

### 消融实验与关键发现
#### （1）推测深度（speculative depth）的影响
- **d=3 即可捕获大部分收益**，进一步提升至 d=5/7 收益递减
- 增加 d 不引起明显质量下降 → 分层接受机制有效过滤错误草案
- 推荐 **d=3 作为默认值**，兼顾效率与鲁棒性

#### （2）不同 spatial policy 的表现差异
- **PSD (LocalLeap)**：倾向于更高 TPF（更强并行性）
- **PSD (Confidence)**：倾向于更好 accuracy retention（更稳定）
- 用户可根据需求“插拔”不同 backbone

#### （3）任务敏感性分析
- **代码生成（HumanEval/MBPP）比数学推理（GSM8K）更敏感于激进并行**
  - 因代码语法约束强，单个错误 token 可致整个函数失败
- 但 PSD 的 speculative verification 显著缩小了这一差距，成为安全网

#### （4）未来位置可预测性（Figure 5）
- 当前步选中的 high-confidence 位置对未来几步有较强预测力（Precision@K 较高）
- 支持了“confidence ranking 稳定”的假设，为 speculative drafting 提供理论依据

#### （5）阶段互补性分析（Figure 6）
- **早期阶段**：parallel 与 speculative 贡献均衡
- **中期**：context 积累后，parallel 解码高峰出现
- **后期**：masked positions 减少，speculative 更易成功接受深层草案
- 二者在不同阶段互补，共同推动整体效率提升

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **空间与时间加速是互补的**：两者分别压缩不同维度，联合使用可实现**复合加速**，突破单轴帕累托前沿。
2. ✅ **PSD 实现高达 5.5× TPF**，且在多数场景下准确率损失 <1%，显著优于所有基线。
3. ✅ **分层接受机制有效防止错误传播**，使系统能安全地采用激进并行策略而不牺牲质量。
4. ✅ **PSD 是模块化、策略无关的框架**，可灵活集成任何 parallel decoding 方法，具备良好扩展性。
5. ✅ **推测深度 d=3 已接近饱和**，无需更深推测即可获得主要收益，适合实际部署。

---

### 方法的局限性
1. **依赖客观可验证任务**：实验集中在数学与编程等有明确正确答案的任务上，未验证在开放生成任务（如创意写作、摘要）上的表现。
2. **未探索极端领域适配**：虽然默认配置泛化良好，但在高度专业化或分布外领域可能需要超参数调优才能达到最优权衡。
3. **KV Cache 共享虽优化但仍有限**：尽管利用了草案间的共享前缀减少计算，但大规模草案可能导致内存压力。

---

### 未来工作方向
- 将 PSD 扩展至 **多模态 diffusion models**（如文本到图像、视频生成）
- 设计更智能的 **candidate selection 策略**，提升 speculation 成功率
- 探索 **动态调整 speculative depth** 的机制，根据上下文复杂度自适应控制风险
- 在 **边缘设备或低资源场景** 中验证 PSD 的实用性与能耗效益

--- 

> **总结一句话**：  
> PSD 通过将 **spatial parallel unmasking** 与 **temporal speculative verification** 有机结合，提出了一种高效、通用、无需训练的 dLLM 加速框架，在保持生成质量的同时实现了高达 **5.5× 的推理加速**，刷新了 dLLM 推理的帕累托前沿。

</details>

---

### 2. [Response-Conditioned Parallel-to-Sequential Orchestration for Multi-Agent Systems](https://arxiv.org/abs/2605.15573)

**Authors**: Nurbek Tastan, Alex Iacob, Lorenzo Sani, Meghdad Kurmanji, Nicholas D. Lane, Samuel Horvath, Karthik Nandakumar  
**Category**: cs.CL  
**Published**: 2026-05-18  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.15573v1  

#### Abstract
Multi-agent systems can solve complex tasks through collaboration between multiple Large Language Model agents. Existing collaboration frameworks typically operate in either a parallel or a sequential mode. In the parallel mode, agents respond independently to queries followed by aggregation of resp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Response-Conditioned Parallel-to-Sequential Orchestration for Multi-Agent Systems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **multi-agent LLM systems** 主要采用两种协作模式：
- **Parallel 模式**：所有 agent 并行响应，通过投票或聚合得到最终答案。优点是简单可扩展，但计算开销大、冗余高。
- **Sequential 模式**：agent 按照固定拓扑（如链、树）逐步传递信息。能支持错误修正，但需要预设通信结构，设计复杂且难以泛化。

这两种方式在**通信效率、延迟与准确性之间存在权衡困境**。本文提出一个核心问题：
> “是否应该进行 sequential 传播？”——这一决策应基于 agent 的实际输出动态决定，而非预先设定。

---

### 🚀 提出的新方法：NEXA
作者提出了 **NEXA**（Nexus-based Execution Architecture），一种**可训练的响应条件策略**（response-conditioned policy），实现从并行到串行执行的智能切换。

#### 核心机制：
1. **第一阶段：Parallel Draft**
   - 所有 agent 独立生成初始响应 $ R^{(0)}_n = A_n(Q) $
2. **第二阶段：Semantic Embedding**
   - 将每个响应嵌入共享语义空间（使用 `all-MiniLM-L6-v2` 编码器）
   - 计算每个响应的 **contribution score**：$ \phi_n = \text{cos}(r_n, r_{\text{avg}}) $
3. **第三阶段：Graph Prediction**
   - 使用轻量级 **Transformer** 模型预测稀疏有向无环图（sparse DAG）
   - 若图为 **空** → 保持并行，直接聚合输出
   - 若图非空 → 执行一次 **sequential propagation**，按 contribution 排序更新节点
4. **第四阶段：Aggregation**
   - 使用 **judge-free weighted centroid aggregation** 选择最终答案

---

### 🔍 相比现有方法的优势
| 维度 | NEXA 的优势 |
|------|-------------|
| **灵活性** | 不强制选择“纯并行”或“纯串行”，而是由模型根据响应状态自动判断 |
| **高效性** | 显著减少 token 消耗（相比 SelfOrg* 减少 ~35%，相比 GPTSwarm 减少 >50%） |
| **通用性** | 政策不依赖 agent 身份、角色标签或模型家族，仅基于语义内容 |
| **简洁性** | 无需外部 LLM judge、reward model 或复杂的拓扑搜索算法 |
| **理论保障** | 构造上保证 DAG 无环、支持 permutation-equivariance 和 hybrid subsumption |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **AQUA-RAT**：数学推理题（代数文字题）
- **GSM8K**：小学数学应用题
- **HumanEval**：代码生成任务
- **GSM-Hard**, **MMLU**（用于扩展迁移实验）

---

### ⚙️ 实验设置
| 参数 | 设置 |
|------|------|
| 基础 agent 模型 | Qwen2.5-1.5B-Instruct / Qwen2.5-7B-Instruct |
| agent 数量（训练） | N = 10 |
| 测试团队规模 | N ∈ {5, 10, 15, 20}（零样本迁移） |
| Policy 模型架构 | 1-layer, 1-head Transformer encoder |
| 训练目标 | REINFORCE + batch-mean baseline + 边数正则项（sparsity penalty） |
| 优化器 | Adam, lr=0.1, dropout=0.3, batch_size=32 |
| 奖励函数 | $ R_{\text{sp}}(\mathcal{G}) = R_{\text{task}}(\mathcal{G}) - \lambda_{\text{sp}} \cdot |\mathcal{E}| $ |

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 最终答案正确率（mean ± std） |
| **Average Rating** | 方法排名评分（越低越好） |
| **Token Usage** | 总消耗 token 数（prompt + completion） |
| **Mean Edge Count** | 预测通信图的平均边数（衡量通信密度） |
| **Rescue/Harm/Preservation Rate** | 分析通信对答案的影响 |

---

### 🆚 基线方法对比
| 方法 | 类型 |
|------|------|
| Single | 单 agent |
| CoT | Chain-of-Thought |
| SC (Self-Consistency) | 多路径采样 + 投票 |
| SelfOrg* | 自组织通信（单轮 sequential） |
| GPTSwarm | 可优化图结构 |
| AgentPrune | 剪枝不必要的通信 |
| G-Designer | 图神经网络学习拓扑 |

---

## 3. 主要实验结果和性能指标

### 📈 主要性能对比（Table 1）

| Method | AQUA-RAT | HumanEval | GSM8K | **Avg. Acc.** | Avg. Rating | Token Usage |
|--------|----------|-----------|--------|----------------|--------------|--------------|
| Single | 52.62 | 50.41 | 70.07 | 57.70 | 5.67 | 1.15M |
| CoT | 54.46 | 45.93 | 70.47 | 56.95 | 5.33 | 1.30M |
| SC | 56.82 | 12.52 | 71.53 | 46.96 | 5.00 | 11.22M |
| SelfOrg* | 56.46 | 52.03 | 72.60 | 60.36 | 2.67 | 28.41M |
| GPTSwarm | 55.91 | 36.79 | 69.33 | 54.01 | 6.33 | 38.02M |
| NEXA (**Ours**) | **57.74** | **51.42** | **73.53** | **60.90** | **1.33** | **18.36M** |

✅ **关键发现**：
- NEXA 在 **平均准确率** 上达到 **60.90%**，优于所有基线
- **token 消耗仅为 SelfOrg* 的 65%**，GPTSwarm 的 48%
- 在 **accuracy-cost tradeoff 曲线中占据最优区域**

---

### 🔁 泛化能力测试（Generalizability）

#### （1）Agent 数量迁移（Figure 2）
- 训练于 N=10，测试于 N∈{5,15,20}
- 在 N=15 时性能达到峰值，说明策略能有效利用更多候选响应
- 即使团队规模变化仍稳定优于 CoT 和 Single

#### （2）跨任务迁移（Figure 3）
- 在 AQUA-RAT 上训练 → 在 GSM8K 上测试（反之亦然）
- 跨任务差距极小（<0.2 pts），表明学到的是**通用通信规则**，而非记忆特定任务模式

#### （3）模型尺度迁移（Figure 4）
- 在 Qwen2.5-1.5B 上训练 → 在 Qwen2.5-7B 上测试
- 性能几乎持平（GSM8K: 90.48 vs 90.52；AQUA-RAT: 76.98 vs 77.40）
- 表明通信策略**不依赖 backbone 能力水平**

#### （4）模型代际迁移（Figure 5）
- 在 Qwen2.5-1.5B 上训练 → 在 Qwen3.5-2B 上测试
- 仅落后 0.17 pts（77.40 vs 77.73 @ N=5）
- 说明无需每次升级 base model 都重新训练控制器

---

### 🔍 消融实验（Ablation Studies）

#### （1）Policy Backbone 消融（Table 2）
| Backbone | Accuracy (GSM8K) |
|---------|------------------|
| Transformer | 73.53 ± 0.23 |
| GNN | 73.48 ± 0.31 |

➡️ 结果相近 → **核心优势来自 NEXA 框架本身，而非特定网络结构**

#### （2）优化目标消融（Table 3）
| Optimizer | AQUA→AQUA | GSM8K→AQUA |
|----------|------------|-------------|
| PG (Ours) | **57.74** ± 2.31 | **57.56** ± 1.49 |
| GRPO     | 57.56 ± 3.83 | 57.48 ± 2.30 |

➡️ PG 略优，但差异不大 → 策略鲁棒性强

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Hybrid Execution 更优**：  
   “先并行再按需串行”的混合范式显著优于纯并行或纯串行系统。
   
2. **响应条件策略具有强泛化性**：  
   学到的通信政策可在不同 agent 数量、任务、模型规模和代际间迁移，无需微调。

3. **通信高度稀疏且有效**：  
   - 多数情况下只激活少量边（Figure 8：高达 70% 示例使用 ≤50% 可能边）
   - 更强的 backbone 导致更稀疏通信 → 政策学会“强者无需多言”

4. **通信改善质量而不破坏正确性**：  
   - **Rescue rate** 随 agent 数增加而上升（5→20: 19.2% → 23.8%）
   - **Harm rate** 很低（1.6%~2.5%）
   - **Preservation rate >97.5%** → 正确答案基本不受干扰

---

### ⚠️ 局限性（Limitations）
1. 当前评估集中在 **closed-ended reasoning 和 programming 任务**，尚未验证于开放生成、工具调用或多轮规划场景。
2. 依赖 **embedding model** 对 response 差异的捕捉能力。若编码器失效，则 contribution ordering 和 graph prediction 可能失准。
3. 初始 agent 数量仍是效率瓶颈。虽然通信稀疏，但第一轮并行 draft 固定开销较大。

---

### 🔮 未来工作方向
- 结合 **adaptive agent selection**：动态决定初始 draft 数量 + 是否通信
- 扩展至 **multi-round sequential refinement**
- 探索 **real-time interactive setting** 中的应用（如对话系统、机器人协作）
- 引入 **uncertainty-aware triggering**：仅当模型不确定时才启动通信

---

## ✅ 总结
**NEXA 是首个将 parallel 与 sequential multi-agent execution 统一在一个可学习框架下的方法**。它通过分析 agent 的实际输出来决定是否需要进一步通信，并以轻量、judge-free、可泛化的方式实现了更高的 accuracy-cost 效率。其核心思想——“让响应自己决定是否需要沟通”——为 multi-agent system 设计提供了新的范式。

</details>

---

### 3. [DualKV: Shared-Prompt Flash Attention for Efficient RL Training with Large Rollouts and Long Contexts](https://arxiv.org/abs/2605.15422)

**Authors**: Jiading Gai, Shuai Zhang, Xiang Song, Bernie Wang, George Karypis  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.15422v1  

#### Abstract
Modern RL post-training methods such as GRPO and DAPO train on $N$ response sequences of $R$ tokens sampled from a shared prompt of $P$ tokens, but standard FlashAttention replicates all $P$ prompt tokens $N$ times across both forward and backward passes -- duplicating compute and memory on identica...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DualKV: Shared-Prompt Flash Attention for Efficient RL Training with Large Rollouts and Long Contexts**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代强化学习（RL）后训练方法（如 GRPO 和 DAPO）在训练时会为同一个 `prompt` 生成 $ N $ 条响应序列，每条长度为 $ R $，共享一个长度为 $ P $ 的提示。标准的 **FlashAttention-2 (FA2)** 在实现中会对这 $ N $ 条序列分别处理，导致 **相同的 prompt tokens 被重复计算 $ N $ 次**。

这种冗余在以下场景中尤为严重：
- **大 rollout 因子**（$ N \geq 16 $）
- **长上下文**（$ P > 8K $）

这导致：
- 内存占用呈 $ O(N \cdot P \cdot d) $ 增长
- 计算量翻倍，成为策略更新（policy update）阶段的瓶颈

尽管推理阶段已有 **Prefix Caching、Paged Attention** 等共享 prompt KV 的技术，但这些方法：
- 不支持训练反向传播（backward pass）
- 无法处理多个序列对共享 KV 的梯度累积
- 不能直接用于 RL 训练流程

---

### **提出的新方法：DualKV**
DualKV 是首个专为 **RL 训练** 设计的 FlashAttention 变体，通过 **内核级优化** 消除共享 prompt 的重复计算。

#### **核心创新点**
1. **DualKV CUDA Kernel（双区域注意力内核）**
   - 将注意力分为两个阶段：
     - **Context Self-Attention**：仅对共享 prompt 执行一次自注意力
     - **Decoded Attention**：所有响应序列并行地对 `[K_prompt; K_response]` 进行注意力计算
   - 内核在单次 launch 中迭代两个物理上分离的 KV 区域（共享 prompt 与独立 response），避免复制

2. **数据流水线重构（Data Pipeline Redesign）**
   - 将原始的 $ N(P+R) $ token 打包方式改为 $ P + NR $
   - 所有来自同一 prompt 的 $ N $ 条响应被分组到同一批次中
   - 利用 `veRL` 框架跳过 `balance_batch` 和禁用 shuffle，保持 prompt 分组连续性

3. **训练兼容的梯度累积机制**
   - 共享的 $ K_c, V_c $ 会从两个 attention 阶段接收梯度
   - 使用 **fp32 原子累加（atomic accumulation）** 处理多序列并发写入，避免精度损失
   - 最终将累加结果转换回 bf16，保证与 FA2 数值等价

4. **数学等价性保障**
   - DualKV 输出与标准 FA2 完全一致（浮点舍入误差范围内）
   - 无任何近似或精度损失

---

### **相比现有方法的优势**
| 方法 | 支持训练 | KV 去重 | Compute 去重 | 数学精确 | Autograd 兼容 |
|------|----------|---------|---------------|-----------|----------------|
| FA2 | ✅ | ❌ | ❌ | ✅ | ✅ |
| Paged Attention | ❌ | 存储级 | ❌ | ✅ | ❌ |
| Bifurcated Attention | ❌ | ❌ | ❌ | ✅ | ❌ |
| Prefix Grouper | ✅ | ❌（仍传 N 拷贝） | ❌ | ✅ | ✅ |
| **DualKV (Ours)** | ✅ | ✅ | ✅ | ✅ | ✅ |

> ✅ **DualKV 是唯一同时满足四项要求的方法**

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LongReason**：长上下文数学推理数据集，$ P \leq 8192 $, $ R \leq 2048 $
- **GSM8K**：短提示数学题数据集，用于验证短上下文下的有效性
- **合成变体 GSM8K**：控制 $ P \in \{1K, 3K\} $，测试内存扩展能力
- **Llama-3.1-8B**：跨模型家族验证通用性

---

### **实验设置**
- **模型**：
  - Qwen3-8B（主实验）
  - Qwen3-30B-A3B（MoE 模型，多节点）
- **硬件**：
  - 单节点：8×H100-SXM5-80GB（p5.48xlarge）
  - 多节点：16×H100（2 节点）
- **训练配置**：
  - Rollout Factor $ N = 32 $
  - Response Length $ R = 2048 $
  - Prompt Length $ P = 8192 $
  - Micro-batch size per GPU: 4 或 8
  - 使用 FSDP2 + BF16 + Gradient Checkpointing
- **框架集成**：
  - 修改 `veRL` 流水线以支持 prompt 分组
  - 实现 `flash_attn_dualkv_varlen_func` 内核

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Policy Update Latency** | 策略更新前向+反向耗时 |
| **Peak GPU Memory** | 训练峰值显存占用 |
| **End-to-End Step Time** | 完整训练步时间（含 rollout） |
| **MFU (Model FLOPs Utilization)** | 模型 FLOPs 利用率，衡量硬件效率 |
| **Speedup** | 相对于 FA2 的加速比 |
| **Token Reduction Ratio $ p $** | $ p = \frac{N(P+R)}{P + NR} $，理论加速上限 |

---

### **基线方法对比**
- **FA2 (FlashAttention-2)**：标准实现，prompt 复制 $ N $ 次
- **FA3**：Hopper 架构专用内核
- **Prefix Grouper**：框架级优化，但未修改内核，仍传递 N 拷贝 KV
- **Ulysses SP**：序列并行方案，用于缓解 OOM

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **单内核性能（A100, Qwen3-8B）**
| $ N $ | $ P $ | 方法 | F+B 时间 (ms) | 速度提升 | 显存下降 |
|-------|--------|------|----------------|----------|----------|
| 28 | 4K | FA2 | 172.2 | 1.00× | - |
| 28 | 4K | DualKV | 104.2 | **1.65×** | 63% |
| 28 | 16K | FA2 | 1441.0 | 1.00× | - |
| 28 | 16K | DualKV | 371.7 | **3.88×** | 85% |
| 16 | 32K | FA2 | OOM | - | - |
| 16 | 32K | DualKV | 525.8 | **5.48×** | 86% |

> ⚠️ 在 $ P > 32K $ 时，FA2 直接 OOM

---

#### **端到端训练性能（Qwen3-8B, 8×H100）**
| 方法 | mb/GPU | 峰值显存 (GB) | 步骤时间 (s) | Policy Update 加速 | MFU |
|------|--------|----------------|--------------|------------------------|-----|
| FA2 | 4 | 106 | 1017 | 1.00× | 35.8% |
| DualKV | 4 | 81 | 687 | **1.63×** | 58.8% |
| DualKV | 8 | 93 | 622 | **2.09×** | **75.8%** |

- **DAPO 结果更优**（无 KL 参考模型）：
  - Policy Update 加速达 **2.47×**
  - MFU 提升至 **77.4%**

---

#### **多节点 MoE 模型（Qwen3-30B, 16×H100）**
| 方法 | SP | Policy Update (s) | 总步骤时间 (s) | 加速比 |
|------|----|--------------------|------------------|--------|
| FA2 | 4 | 4115 | 5284 | 1.00× |
| DualKV | 1 | **1078** | **1564** | **3.82× / 3.38×** |

- **无需序列并行（SP=1）即可运行**
- 节省通信开销（无 all-to-all）
- 实际训练时间从 **42.6 小时 → 12.6 小时**（节省 ~$6K 成本）

---

#### **与 Prefix Grouper 对比**
| $ P $ | $ mb $ | 方法 | 时间 (ms) | 显存 (GB) | DualKV 优势 |
|--------|--------|------|------------|------------|-------------|
| 8K | 16 | PG | 465 | 50.7 | ❌ OOM |
| 8K | 16 | DualKV | 128 | 9.9 | **4.7× 更快，5.1× 更省内存** |

> ✅ DualKV 消除了所有 per-token 操作的冗余（MLP、Norm、Projection），而 PG 仅优化 attention

---

### **消融实验**
- **fp32 原子累加必要性**：
  - 若使用 bf16 atomicAdd，会导致梯度精度损失
  - 当前设计确保与 FA2 数值等价（`torch.allclose` 验证）
- **数据流水线影响**：
  - 启用 `shuffle=False` 和跳过 `balance_batch` 不改变梯度估计一致性
  - 证明 mini-batch 梯度期望不变（Appendix B）

---

## **4. 关键结论和发现**

### **主要发现**
1. **共享 prompt 的重复计算是当前 RL 训练的主要瓶颈**
   - 在 $ N=32, P=8K $ 下，prompt 冗余占总 token 数的 **~70%**
   - 消除后可带来 **2–3.8× 策略更新加速**

2. **DualKV 实现了全栈去重**
   - 不仅消除 attention 中的 KV 复制
   - 更通过打包 $ P + NR $ 形式，使 **Norm、MLP、Projection 等操作也只处理一次 prompt**

3. **显著提升 MFU 与批大小**
   - MFU 从 **36% → 76%**
   - 支持 **2× 更大的 micro-batch**
   - 在长上下文下，显存增长几乎与 batch size 无关

4. **可消除序列并行依赖**
   - FA2 在 MoE 上需 4-way Ulysses SP 才能运行
   - DualKV 在 SP=1 下即可完成，避免通信开销

5. **通用性强**
   - 适用于 GRPO、DAPO、PPO 等所有基于采样的 RL 方法
   - 在 Qwen 和 Llama 模型上均有效
   - 支持 GQA、variable length、grouped queries

---

### **方法的局限性**
1. **假设单一共享前缀**
   - 当前设计要求一个 micro-batch 内所有序列共享同一 prompt
   - 不支持树状结构或多轮对话中的部分共享前缀

2. **依赖数据流水线重构**
   - 需要训练框架支持 prompt 分组（如修改 `veRL`）
   - 不是完全“即插即用”

3. **加速效果随 $ P/R $ 和 $ N $ 增大而增强**
   - 在短提示（如 GSM8K）下加速有限（约 1.1–1.2×）
   - 最佳收益出现在 $ N \geq 16, P \geq 16K $

---

### **未来工作方向**
1. **扩展至树状共享结构**
   - 支持 Tree-of-Thought、多轮对话等场景
   - 动态识别共享子图并进行 KV 缓存

2. **与 Ulysses SP、Ring Attention 组合**
   - DualKV 解决 intra-batch 冗余，SP 解决 inter-device 长序列分割
   - 联合优化超长上下文（$ P > 100K $）

3. **适配 FA3/Hopper 和 FA4/Blackwell**
   - 利用新一代硬件特性进一步提升吞吐

4. **自动调度策略**
   - 根据 $ N, P, R $ 自动选择是否启用 DualKV
   - 动态调整 micro-batch 分组策略

---

> 🔚 **总结**：DualKV 是首个真正为 **RL 训练** 优化的 FlashAttention 变体，通过 **内核级共享 prompt 去重 + 数据流水线重构**，实现了 **2–3.8× 速度提升、76% MFU、消除序列并行依赖**，将 RL 训练瓶颈从策略更新转移到 rollout 生成，为大规模 RLHF 提供了高效基础设施。

</details>

---

### 4. [Going Beyond the Edge: Distributed Inference of Transformer Models on Ultra-Low-Power Wireless Devices](https://arxiv.org/abs/2605.15694)

**Authors**: Alexander Gr\"afe, Ding Huo, Johannes Berger, Marco Zimmerling, Sebastian Trimpe  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.15694v1  

#### Abstract
Transformer models are rapidly becoming a cornerstone of modern Internet of Things (IoT) applications, yet their computational and memory demands far exceed the capabilities of a single typical ultra-low-power IoT device. We present CATS, a framework for distributed transformer inference on ultra-lo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Going Beyond the Edge: Distributed Inference of Transformer Models on Ultra-Low-Power Wireless Devices*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对**超低功耗无线设备**（如MCU级传感器节点）在执行Transformer模型时面临的三大挑战：
- **C1：计算资源受限**（有限的Flash存储权重、RAM存储激活值、算力不足）
- **C2：通信带宽受限**（低功耗无线链路带宽小、延迟高，Mesh网络多跳加剧瓶颈）
- **C3：通信不可靠**（无线传输易丢包，导致中间激活丢失，影响推理精度）

现有分布式Transformer推理（DTI）方法大多基于高性能边缘设备（如Raspberry Pi）和有线连接，无法适用于真实世界中广泛部署的**超低功耗无线Mesh网络环境**。

---

### 🚀 提出的新方法：CATS框架
作者提出 **CATS**（Collaborative Inference at the Sensor-level），一个专为超低功耗无线设备设计的**通信感知型分布式Transformer推理框架**，其核心创新包括：

#### （1）**SomeGather：一种剪枝化的All-to-All通信原语**
- 对传统的AllGather进行列剪枝（column pruning），仅广播部分activation columns。
- 非广播列在本地处理，无需通信，从而显著降低：
  - 通信开销（bandwidth）
  - 每设备RAM占用（减少接收缓存）
  - 同时间接减少Flash使用（因可配合权重分片）
- 保持模型准确率不变，实现“无损压缩”通信。

#### （2）**基于SomeGather的模型并行划分策略**
- 将Multi-Head Attention按head维度拆分到不同设备。
- 所有跨设备通信统一表达为SomeGather操作，提升通信效率与可扩展性。
- 实现**Flash、RAM、计算负载的同时下降**——这是现有方法无法做到的。

#### （3）**Message Dropout (MD)：模拟丢包的训练机制**
- 在训练阶段随机丢弃来自某些设备的消息，模拟现实中的无线丢包行为。
- 使模型对推理时的实际消息丢失具有鲁棒性。
- 支持多种丢包模式（单设备未收到、全网未收到某设备消息等）。

#### （4）首次实现端到端DTI在超低功耗无线硬件上的部署
- 基于nRF52840 MCU（64MHz Cortex-M4, 1MB Flash, 256KB RAM）构建16节点测试床。
- 使用BLE物理层 + **Mixer协议** 实现高效可靠的all-to-all广播。

---

### 🔍 相比现有方法的优势

| 特性 | CATS | Voltage [Hu & Li, 2024] | Astra [Liu et al., 2025b] | SiracusaDTI [Bochem et al., 2025] |
|------|------|--------------------------|----------------------------|-----------------------------------|
| 减少每设备RAM | ✅（通过SomeGather剪枝） | ✅ | ✅ | ❌（AllReduce增加RAM） |
| 减少每设备Flash | ✅（权重分片） | ❌（全权重存储） | ❌（全权重存储） | ✅ |
| 通信高效（适合Mesh） | ✅（SomeGather + Mixer） | ⚠️（假设特定拓扑） | ⚠️ | ❌（依赖AllReduce） |
| 抗丢包能力 | ✅（MD训练） | ❌ | ❌ | ❌ |
| 可运行模型大小 | 最大达单设备14倍 | ≤ 单设备容量 | ≤ 单设备容量 | ≤ 单设备容量 |

> ✅ CATS是首个同时优化**RAM、Flash、通信、鲁棒性**的DTI方案，真正适配超低功耗无线场景。

---

## 2. 核心实验方法和设置

### 📚 数据集
主要用于仿真评估（accuracy、robustness）：
- **ETT-h2**：电力变压器温度时间序列预测
- **ICD**：工业控制系统数据异常检测
- **London-smart-meters**：家庭用电量预测
- **Traffic**：城市交通流量预测

> 均为典型IoT时间序列任务，采用Time-Series Transformer架构。

---

### ⚙️ 实验设置

#### 硬件平台
- 设备：**nRF52840** 开发板（64 MHz Cortex-M4, 1MB Flash, 256 KB RAM）
- 数量：最多 **16台**
- 网络：**BLE物理层** 构建Wireless Mesh Network，直径≥2跳
- 协议栈：集成 **Mixer** 实现低延迟、高可靠广播

#### 模型配置
- Transformer层数：6
- Feature dimension：128
- Hidden layer in FFN：1 layer
- 使用8-bit量化内核（CMSIS-NN）

#### 评估指标
| 问题 | 指标 |
|------|------|
| Q1（内存扩展性） | 每设备RAM usage、Flash footprint |
| Q2（延迟扩展性） | Attention/Residual Block latency（含计算+通信） |
| Q3（准确性） | MSE（Mean Square Error）预测误差 |
| Q4（鲁棒性） | 不同message loss下test loss变化 |

---

### 🔀 基线方法对比
由于无直接可比端到端系统，采用分问题基线：

| 问题 | 基线方法 |
|------|---------|
| Q1 | Voltage, Astra, SiracusaDTI（假设集成进Mixer框架） |
| Q2 | Centralized execution on single device |
| Q3 | AllGather-based DTI、Normal pruning（普通剪枝匹配通信量） |
| Q4 | 未使用MD训练的相同模型 |

---

## 3. 主要实验结果和性能指标

### 📈 Q1：内存扩展性（C1）
- **Flash使用随设备数线性下降**：16设备时降至单设备的~6.25%
- **RAM使用显著低于AllGather方案**：
  - SomeGather（90% pruning）相比AllGather减少 **67.5% RAM**
- **支持最大模型达1400万参数**，是单设备极限的 **14×**

> 图5显示，在高token长度N下，传统方案受RAM限制严重；而CATS通过pruning大幅扩展可行区域。

---

### ⏱️ Q2：延迟扩展性（C1 & C2）
- **非剪枝情况下**：
  - Attention Block：最高提速 **2.51×**（512 features, 16 devices）
  - Residual Block：从慢0.54× 到快1.38×（512 features）
- **启用90% pruning后**：
  - 通信延迟降低 **高达80%**
  - Attention Block加速达 **4.37×**
  - 整体推理延迟缩放因子达：
    - **10.96×**（attention, 512 features）
    - **6.45×**（residual, 512 features）

> 尽管通信仍占主导（69–91%总耗时），但SomeGather剪枝极大缓解瓶颈。

---

### 🎯 Q3：SomeGather的准确性（C1 & C2）
- **SomeGather vs Normal Pruning**（图7）：
  - Normal pruning：随着通信节省增加，预测误差持续上升
  - **SomeGather**：即使通信节省90%，**MSE几乎不变**
- 表明：**剪枝发生在结构化位置（列级）+ 训练协同调整 → 无损通信压缩**

---

### 🛡️ Q4：Message Dropout抗丢包能力（C3）
- **无MD模型**：10% message loss → 预测误差相对增长可达 **200%**
- **使用MD训练模型**（图8）：
  - 在训练接近的丢包率下表现最优
  - 10%丢包时，相对误差增长仅 **23.6%**（平均）
  - 各数据集上test loss增幅控制在2.5%~23.6%

> MD有效提升了模型对无线信道不确定性的鲁棒性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DTI可在超低功耗无线设备上实现**：CATS首次实现了在MCU级设备上的大规模Transformer协同推理。
2. **SomeGather实现通信与资源双重优化**：
   - 通信量↓90%
   - RAM↓67.5%
   - Flash与计算负载也同步下降
   - **且不牺牲准确率**
3. **MD显著增强鲁棒性**：面对现实无线丢包，模型性能退化被压制在一个可控范围内。
4. **CATS优于所有现有范式**：无论是仅减RAM还是仅减Flash的方法，都无法像CATS一样全面优化。

---

### ⚠️ 局限性
1. **设备数量受限于Attention Head数量**：
   - 当前每个device负责一个head → 最大支持设备数 = head数（通常≤12）
   - 若head数 < device数，则资源利用率不足
2. **静态配置依赖**：
   - 剪枝模式和MD概率需在训练时确定
   - 难以适应动态变化的网络规模或丢包特性
3. **Mixer协议依赖特定硬件同步机制**（如PLL），可能限制通用性

---

### 🔮 未来工作方向
1. **将单个Attention Head进一步分布到多个设备**，突破head数对设备数的上限约束。
2. **开发无需重训练即可适配不同网络配置的预训练Transformer插件机制**，提升部署灵活性。
3. 探索**自适应pruning与dropout机制**，根据实时网络状态动态调整通信策略。
4. 扩展至Decoder-only架构（如LLM）和更多应用场景（如语音、图像分类）。

---

## 总结
> **CATS是首个将分布式Transformer推理推向“超边缘”（beyond the edge）的完整解决方案**。它通过**SomeGather + 分片 + MD**三位一体的设计，在资源极度受限的无线MCU设备上实现了高达14倍于单设备容量的大模型推理，并兼顾了通信效率与鲁棒性，为未来智能IoT系统提供了坚实的技术基础。

</details>

---

### 5. [Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered](https://arxiv.org/abs/2605.15622)

**Authors**: Sijia Liu, Yicheng Lang, Soumyadeep Pal, Changsheng Wang, Yancheng Huang, Chongyu Fan, James Diffenderfer, Bhavya Kailkhura, Yihua Zhang  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.15622v1  

#### Abstract
Zeroth-order (ZO) optimization, learning from finite differences of function evaluations without backpropagation, has recently regained attention in deep learning due to its memory efficiency and applicability to gray- or black-box pipelines. Yet, ZO methods are often dismissed as fundamentally unsc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对**深度学习中零阶优化（Zeroth-Order, ZO）被普遍认为“不可扩展”** 的主流观点提出挑战。当前社区普遍认为 ZO 方法因估计方差高、查询复杂度大而难以用于大规模模型训练，导致其被视为一种低效的“备选方案”。

作者指出，这种悲观看法是**误判而非事实**，根本原因在于研究范式存在“短视”（myopic）——过度聚焦于全参数空间下的梯度估计器设计，忽视了算法、系统与评估层面更广阔的设计空间。

---

### 提出了什么新方法或新思路
论文并未提出单一的新算法，而是从**元认知层面重构对 ZO 优化的理解框架**，提出了六个核心立场（Positions, P1–P6），并据此提炼出一系列颠覆性的设计原则：

#### （1）重新定义可行性边界（P1–P3）
- **P1: 随机方向选择至关重要**  
  强调扰动方向 $ u $ 的结构化设计（如稀疏、预条件）可显著降低方差。
- **P2: 查询-方差权衡必须显式考虑**  
  单纯追求低方差可能带来高昂查询成本，需在二者间取得平衡。
- **P3: 不应忽略方向导数（Directional Derivative, DD）视角**  
  将 ZO 视为方向导数估计任务，提供了一个更强的理论基准（forward gradient），可用于评估 ZO 方法的真实能力。

#### （2）倡导三大未被充分探索的方向（P4–P6）
- **P4: 子空间与谱视角（Subspace & Spectral View）**  
  在低维子空间进行 ZO 优化，利用投影矩阵 $ P \in \mathbb{R}^{d \times r}, r \ll d $ 显著降低估计方差，并可通过谱分析指导子空间选择。
- **P5: ZO 的系统级优势（Systems Advantage）**  
  ZO 的前向执行特性天然支持通信高效分布式训练（仅传输标量差值 + 随机种子）、避免 pipeline bubbles，适合资源受限环境。
- **P6: 任务对齐去混淆（De-obfuscation from Task Alignment）**  
  当前许多 ZO 成功依赖“任务提示”实现任务对齐，这会掩盖真实优化能力；应区分“任务简化收益”与“优化器本身能力”。

#### 衍生行动建议（Call to Actions）
基于上述立场，作者提出五大实践倡议：
- **A1**: 建立严格的评估协议（固定查询预算、引入 forward gradient 基线、对比有无 task alignment）
- **A2**: 超越全空间优化，拥抱子空间与混合 FO-ZO 设计
- **A3**: 探索生成模型（如 Diffusion Models）作为可学习的梯度估计器
- **A4**: 构建 ZO-native 系统栈（如基于 vLLM 的训练引擎）
- **A5**: 拓展应用场景至量子计算、科学模拟等非微分系统

---

### 相比现有方法的优势
| 维度 | 传统 ZO 研究范式 | 本文主张 |
|------|------------------|---------|
| **设计焦点** | Estimator-centric（梯度估计器为中心） | Algorithm-System-Evaluation 协同设计 |
| **搜索空间** | 全参数空间逐元素扰动 | 子空间、结构化扰动、谱优化 |
| **系统考量** | 忽略通信、内存、pipeline 效率 | 主动利用 ZO 的 forward-only 特性构建高效系统 |
| **评估方式** | 报告最终精度，忽略任务对齐影响 | 显式解耦 task complexity 与 optimizer capability |

> ✅ **核心优势**：将 ZO 从“低效替代品”转变为一种具有独特系统优势、适用于边缘设备、联邦学习、混合系统和绿色 AI 的**原生优化范式（native paradigm）**。

---

## 2. 核心实验方法和设置

本论文为一篇 **position paper（立场论文）**，不以提出具体新算法为目标，因此没有传统意义上的端到端实验。其“实验证据”主要体现在以下三方面：

### 使用了哪些数据集
- **下游任务**：SST-2（文本分类）、RTE（自然语言推理）、WiC（词义消歧）
- **模型架构**：Gemma2-2B（20亿参数语言模型）
- 数据来源：GLUE benchmark 中的标准测试集

### 实验设置和评估指标
- **主要指标**：
  - Fine-tuning 准确率（Accuracy）
  - 是否启用 task alignment（通过 prompt 工程对齐预训练目标）
  - 查询次数（Query Budget）隐含控制方差水平
- **对比维度**：
  - 多种 state-of-the-art ZO 方法在有/无 task alignment 下的表现差异
  - 与 forward gradient 方法的潜在性能上限对比（引用已有工作 ZO-Bench）

### 基线方法对比
- **ZO 方法**：
  - MeZO (Malladi et al., 2023)
  - Sparse-MeZO (Liu et al., 2025b)
  - HiZOO (Zhao et al., 2025b)
  - LOZO (Chen et al., 2025)
- **理论基线**：
  - Forward Gradient / Directional Descent（作为性能上界参考）
- **系统基线类比**：
  - DeepSpeed、Megatron-LM、FSDP（反例：不适合 ZO 的 FO-centric 系统）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Figure 2）
在 **Gemma2-2B 上 fine-tuning** 的准确率表现如下（部分示例）：

| 方法 | SST-2 (w/ align) | SST-2 (w/o align) | RTE (w/ align) | RTE (w/o align) |
|------|------------------|-------------------|----------------|------------------|
| MeZO | 91.7             | 86.2              | 53.0           | 47.7             |
| Sparse-MeZO | 92.7       | 86.5              | 54.2           | 47.3             |
| HiZOO | 92.1            | 87.6              | 59.6           | 47.7             |
| LOZO  | 92.2            | 87.6              | 56.3           | 47.3             |

> ⚠️ **观察重点**：所有方法在 **移除 task alignment 后均出现显著性能下降**（平均下降 ~5–6 pts），且相对排名发生变化。

---

### 与基线方法的对比结果
- **vs. Forward Gradient Baseline**（间接比较）  
  引用 ZO-Bench (Zhang et al., 2024c) 结果表明，forward gradient 方法在相同设置下性能明显优于标准 ZO 方法，说明当前 ZO 仍有提升空间。
  
- **vs. FO Methods (SGD/Adam)**  
  文中承认当前 ZO 在绝对性能上仍落后于 FO 方法，但强调这是由于不公平比较（FO 可使用 BP，而 ZO 受限于 query budget 和 task alignment）。一旦考虑内存、通信开销，ZO 在特定场景更具优势。

---

### 消融实验结果（概念性消融）
虽然没有传统消融实验，但论文通过多个角度进行了“思想实验式”的消融分析：
- **消融 task alignment（P6）** → 性能大幅下降 ⇒ 表明当前成功部分归功于任务简化
- **消融 full-space assumption（P4）** → 改用 subspace 可降低方差 ⇒ 子空间是关键突破口
- **消融 estimator-only design（P5）** → 加入系统优化（如 seed reuse、scalar communication）⇒ 可实现近零通信开销

---

## 4. 关键结论和发现

### 论文的主要发现
1. 🔍 **ZO 并非“能力不足”，而是“探索不足”**  
   当前对 ZO 的负面评价源于短视的研究范式，而非本质缺陷。

2. 🧩 **子空间优化是突破方差瓶颈的关键路径**  
   利用低秩结构在 $ r \ll d $ 维子空间中进行 ZO，可在保持梯度保真度的同时大幅降方差。

3. ⚙️ **ZO 是系统友好的优化范式**  
   - 支持轻量通信（scalar + seed）
   - 支持 pipeline parallelism 零气泡调度
   - 可运行于低功耗 GPU 集群，推动绿色 AI

4. 🧪 **当前评估体系严重误导**  
   忽视 task alignment 的影响会导致高估 ZO 算法能力；必须引入 forward gradient 基线和去混淆评估。

5. 🌐 **ZO 应被视为通用学习接口**  
   特别适用于包含 simulator、logic rule、quantum circuit 等不可微组件的 hybrid system。

---

### 方法的局限性
- **尚未形成统一的 ZO-native 系统栈**  
  当前仍依赖移植到 DeepSpeed/FSDP 等 FO 框架，无法发挥全部潜力。
- **缺乏标准化 benchmark**  
  不同工作使用的 query budget、task alignment 方式不一致，难以公平比较。
- **绝对收敛速度仍慢于 FO 方法**  
  在完全公平条件下（相同计算资源），ZO 通常需要更多迭代才能达到相似性能。
- **理论分析滞后于实践需求**  
  对 subspace selection、adaptive query allocation 缺乏严谨理论支撑。

---

### 未来工作方向
1. **构建 ZO-native 训练系统**  
   开发专为 ZO 设计的分布式框架，支持 perturbation parallelism 和 forward-only pipeline。

2. **发展 Learnable ZO Optimizers**  
   使用 Diffusion Models 或其他生成模型直接建模梯度分布，实现“去噪式”梯度估计。

3. **建立标准 ZO Benchmark Suite**  
   包括：固定 query budget、明确 task alignment 设置、提供 forward gradient 上界。

4. **拓展至新兴领域**  
   - 量子机器学习（Quantum ML）：天然兼容测量式访问
   - 科学计算（Physics-informed NNs）：处理非微分物理模块
   - 边缘智能（Edge AI）：低内存、低带宽场景下的自适应训练

5. **融合 FO-ZO Hybrid Paradigm**  
   将 ZO 视为 FO 的补充，在关键层使用 FO，在其余部分使用 ZO，实现最优 trade-off。

---

> 💡 **总结一句话**：  
> 本文呼吁将 ZO 优化从“backpropagation 的拙劣替代品”重新定位为一种**面向未来异构系统、资源约束环境和非微分世界的原生、可持续、系统级优化范式**。

</details>

---

### 6. [Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design](https://arxiv.org/abs/2605.15871)

**Authors**: Alberto Pepe, Chien-Yu Lin, Despoina Magka, Bilge Acun, Yannan Nellie Wu, Anton Protopopov, Carole-Jean Wu, Yoram Bachrach  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.15871v1  

#### Abstract
Toward recursive self-improvement, we investigate LLM agents autonomously designing foundation models beyond standard Transformers. We introduce a dual-framework approach: AIRA-Compose for high-level architecture search, and AIRA-Design for low-level mechanistic implementation. AIRA-Compose uses 11 ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Agentic Discovery of Neural Architectures: AIRA-Compose and AIRA-Design

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的大型语言模型（LLM）仍以 **Transformer** 架构为主，但其存在计算复杂度高（尤其是注意力机制为 $O(n^2)$）、内存占用大等瓶颈。尽管社区已开始探索后Transformer架构（如 **Mamba** 等 State Space Models），但模型设计高度依赖人类专家的经验直觉，在庞大的组合空间中难以系统性地发现最优或非直观的混合架构。

本文提出并验证了一种新的范式：**利用 LLM 代理（Agent）自主进行神经网络架构搜索与设计**，作为迈向 **递归自我改进（Recursive Self-Improvement, RSI）** 的关键一步。

### 提出的新方法与新思路
作者提出了一个双框架体系：

#### ✅ **AIRA-Compose**（高层架构搜索）
- 将传统的 **Neural Architecture Search (NAS)** 转化为多智能体任务。
- 代理在由基本计算单元（primitives）构成的组合空间中搜索，这些单元包括：
  - **Attention (mA)**
  - **MLP (M)**
  - **Mamba (Mb)**
- 采用 **小规模代理训练 + 性能评估 + 外推到大规模** 的流程，借鉴了 Composer 框架的思想，但用 Agent 替代了传统优化器（如贝叶斯优化）。
- 支持两种搜索空间：
  - 两单元池（M, mA）
  - 三单元池（M, mA, Mb）

#### ✅ **AIRA-Design**（低层机制实现）
- 任务难度更高：要求代理从零开始编写完整的 `model.py` 或 `train.py` 文件。
- 包括两个子任务：
  - **Long Range Arena (LRA)**：设计高效的长程依赖建模机制（如 sub-quadratic attention）。
  - **Autoresearch**：在固定时间预算内优化训练脚本，最小化 validation bits-per-byte (BPB)。

这两个框架均基于 **AIRS-BENCH** 任务标准构建，实现了科研闭环自动化。

### 相比现有方法的优势
| 维度 | 传统 NAS | AIRA-Compose | AIRA-Design |
|------|--------|--------------|------------|
| 搜索策略 | 固定算法（如进化、强化学习） | 基于 LLM 的上下文感知假设生成 | 完全开放式的代码生成 |
| 探索能力 | 局部变异，缺乏语义理解 | 可提出新颖结构组合（如“岛状”MLP） | 可实现全新 attention 机制 |
| 效率 | 高度工程化，难扩展 | 模块化、可复用 | 支持端到端训练优化 |
| 创新潜力 | 有限 | 中等 | 高 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### AIRA-Compose
用于小规模代理训练与评估：
- **MAD**：合成 token 操作任务套件（选择性复制、压缩、上下文回忆），使用平均准确率。
- **BabiStories**：儿童故事合成语料，使用交叉熵损失。
- **DCLM**：代码语言建模子集，使用交叉熵损失。

最终大规模预训练使用完整 DCLM 数据集。

#### AIRA-Design
- **Long Range Arena (LRA)**：评估长序列建模能力，包含三个文本任务：
  - **ListOps**（数学表达式解析）
  - **Text Classification**（IMDb 影评分类）
  - **Retrieval**（ACL 摘要相似性判断）
- **Autoresearch**：使用 ClimbMix 语料库进行自回归语言建模，目标是优化训练效率。

---

### 实验设置与评估指标

#### AIRA-Compose 设置
- **代理数量**：最多 11 个 Agent 并行搜索。
- **计算预算**：每个运行最多 24 小时（BabiStories 和 DCLM 为 60 小时）。
- **小规模模型**：16 层，参数量约百万级。
- **外推方式**：Stretching（拉伸）或 Stacking（堆叠）至 350M、1B、3B 参数规模。
- **评估指标**：
  - 验证损失（Validation Loss）
  - 下游任务零样本准确率（Zero-shot Accuracy on 6 tasks）
  - DCLM Core Score（综合多个任务的未加权平均）

#### AIRA-Design 设置
- **代理数量**：最多 20 个 Agent。
- **框架**：AIRA-DoJo，支持 one-shot 与 greedy 搜索策略。
- **评估指标**：
  - **LRA**：测试准确率（Accuracy）
  - **Autoresearch**：验证比特每字节（validation BPB），越低越好。

---

### 基线方法对比
| 类型 | 对比基线 |
|------|---------|
| **基础架构** | Llama 3.2（Transformer）、Nemotron-2/H（Mamba-Transformer 混合） |
| **NAS 方法** | Composer（传统优化器驱动的搜索） |
| **人工 SOTA** | LRA 上的人类最佳模型（Human SOTA） |
| **训练优化** | Autoresearch 公开参考值（0.998 BPB） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ AIRA-Compose 结果（1B 规模，固定 token 预算）
| 模型 | 验证损失 ↓ | 零样本准确率 ↑ | DCLM Core Score ↑ |
|------|-----------|----------------|--------------------|
| Llama 3.2 | 2.815 | 58.4% | 46.9 |
| Composer 最佳 | 2.759 | 59.8% | 47.3 |
| **AIRAformer-D (Stretched)** | **2.734** | **59.7%** | **48.9** |
| **AIRAhybrid-D (Stretched)** | **2.719** | **60.5%** | 48.5 |

> 🔥 **AIRAformer-D** 在准确率上比 Llama 3.2 提升 **2.4%**；  
> 🔥 **AIRAhybrid-D** 提升 **3.8%**。

#### ✅ 缩放效率（IsoFLOP 分析）
- **AIRAformer-C** 比 Llama 3.2 和最佳 Composer 模型分别快 **54%** 和 **71%**。
- **AIRAhybrid-C** 比修改后的 Nemotron-2 快 **23%**，比最佳 Composer 混合模型快 **37%**。

#### ✅ AIRA-Design 结果
- **LRA**：
  - 最佳代理在 **Document Matching** 达到人类 SOTA 的 **97.7%**（差 2.3%）。
  - 在 **Text Classification** 达到 **97.4%**（差 2.6%）。
- **Autoresearch**：
  - 最佳代理（Greedy Opus 4.5 + Literature）达到 **0.968 BPB**，**超越公开参考值（0.998）**。

---

### 与基线方法的对比结果
- 所有 **Agent 发现的架构** 均优于 **Llama 3.2** 和 **Composer 找到的最佳模型**。
- **AIRAhybrid** 系列在混合架构中表现更优，表明 Mamba 与 Attention 的协同效应被有效挖掘。
- 在 **Latency-Validation Loss Pareto 前沿** 上，Agent 发现的模型显著领先于基线。

---

### 消融实验结果
- **不同聚合策略影响**：
  - N1/N2 聚合 + K-means 聚类有助于稳定高频模式。
- **搜索策略对比**：
  - **Greedy scaffold** 明显优于 **one-shot**，说明迭代调试至关重要。
- **文献增强效果**：
  - 在 Autoresearch 中引入 41 篇论文摘要和 14 个代码仓库后，部分代理性能提升明显（如 Opus 4.5+Lit 达到 0.968 BPB）。
- **配置灵活性影响**：
  - “Configurable” LRA 设置允许调参，但较弱代理反而性能下降，说明控制变量能力尚不足。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LLM 代理可以自主发现高性能神经架构**：
   - 发现了 14 种新型架构（7 个 AIRAformer，7 个 AIRAhybrid）。
   - 这些架构在下游任务、缩放效率、延迟等方面全面超越人类设计基线。

2. ✅ **代理不仅能排列积木，还能提出新机制**：
   - 在 LRA 任务中，代理实现了类似 **linear attention**、**hierarchical pooling**、**recurrent memory** 的高效 attention 变体。

3. ✅ **训练脚本优化可自动化**：
   - 在 Autoresearch 上，代理通过调整深度、宽度、学习率调度、优化器组合等方式，实现了比人类更快的收敛速度。

4. ✅ **递归自我改进（RSI）初现雏形**：
   - 代理正在设计“下一代自己所运行的模型”，这是迈向 RSI 的实质性进展。

---

### 方法的局限性
1. **低层次代码生成仍不成熟**：
   - 一次性生成完整 `model.py` 成功率极低（one-shot 无效提交率接近 100%）。
   - 当前成功依赖于 **greedy + iterative refinement** 模式。

2. **缺乏真正的算法创新**：
   - LRA 上的优秀设计多为已有技术（如 Performer、Longformer）的重组，尚未出现革命性新机制。

3. **对框架生态有偏见**：
   - 训练数据偏向 PyTorch，导致在 JAX/Flax 环境下的编码能力受限。

4. **超参数敏感**：
   - “Configurable” 设置反而使部分代理性能下降，说明其调参策略不稳定。

---

### 未来工作方向
1. **升级代理框架**：
   - 引入更强大的 **AIRA2** 或 **AIRA-DoJo++**，支持多 GPU 并行搜索。
   - 开发支持 **增量编辑** 的 scaffold，避免每次重写整个文件。

2. **增强知识整合能力**：
   - 更好地融合外部文献与代码库，提升上下文利用率。

3. **推动完全自主的 RSI 循环**：
   - 将 **aggregate** 与 **scale** 步骤也转化为 Agent 任务，实现全流程自动化。

4. **拓展至更多领域**：
   - 应用于 MoE 架构搜索、硬件感知编译优化、分布式训练策略生成等。

---

> 📌 **总结一句话**：  
> 本文首次系统性展示了 **AI Agent 可以自主发现并实现超越人类设计的神经架构**，不仅在性能上占优，还展现出更高效的缩放特性，标志着向 **递归自我改进** 迈出了坚实一步。

</details>

---

### 7. [Measuring Maximum Activations in Open Large Language Models](https://arxiv.org/abs/2605.15572)

**Authors**: Luxuan Chen, Han Tian, Xinran Chen, Rui Kong, Fang Wang, Jiamin Chen, Yuchen Li, Jiashu Zhao, Shuaiqiang Wang, Haoyi Xiong, Dawei Yin  
**Category**: cs.CL  
**Published**: 2026-05-18  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.15572v1  

#### Abstract
The dynamic range of activations is a first-order constraint for low-bit quantization, activation scaling, and stable LLM inference. Prior work characterized outlier features and massive activations on pre-2024 LLaMA-style models, and the downstream activation-quantization stack inherits that pictur...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Measuring Maximum Activations in Open Large Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文聚焦于**现代开源大语言模型（LLM）中激活值的最大动态范围（maximum activation magnitude）**这一部署关键问题。尽管低比特量化（low-bit quantization）、激活缩放（activation scaling）和稳定推理严重依赖激活的动态范围，但此前研究多基于早期 LLaMA 风格模型，未能系统地重新审视后-LLaMA 时代多样化开源模型中的最大激活行为。

具体而言，论文试图回答以下部署导向的问题：
- 现代开源 LLM 中的激活值最大能达到多大？
- 这种最大值如何随模型家族、架构、训练阶段等变化？
- 是否仍可简单用参数量来预测？

### 提出了什么新方法或新思路
- **统一测量协议（Unified Pipeline）**：提出并实施了一个标准化的测量流程，在 **27 个来自 8 个不同家族的检查点（checkpoints）** 上进行跨模型比较。该协议包括：
  - 使用相同的 5,000 样本多领域语料库；
  - 家族特定的 tokenizer 重分词；
  - 在嵌入层、隐藏状态、注意力输出、MLP/MoE 输出、SwiGLU 门控、最终归一化等组件上使用一致的 PyTorch hooks 收集统计信息。
- **将“大规模激活”从二元现象重构为连续度量**：摒弃了以往基于局部稀疏性和绝对阈值的二元判定（如 [Sun et al., COLM 2024] 的 `|x| > 100` 且 `local ratio ≥ 1000`），转而采用 **全局最大激活值 $ M = \max{|a|} $** 作为核心部署相关指标。
- **首次系统性跨家族、跨架构、跨训练阶段的观测分析**：涵盖 dense、MoE、视觉-语言（vision-language）、中间训练阶段、指令微调（SFT）等多种变体。

### 相比现有方法的优势
- **更贴近实际部署需求**：$ M $ 直接影响 per-tensor 量化尺度选择，是量化误差的关键前因变量。
- **打破“参数量决定一切”的迷思**：揭示了最大激活值受模型家族、架构设计、训练方式等多重因素影响，不能仅由参数规模推断。
- **开放可复现**：作者公开了完整的测量代码（[GitHub](https://github.com/clx1415926/Max_act_llm)）和每检查点的激活统计数据，支持后续研究。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **RedPajama 子集构建的 5,000 样本多领域语料库**，按内容类型划分：
  - 数学/科学（850）
  - 编程代码（850）
  - 英文网页文本（850）
  - 知识类/QA/书籍（850）
  - 中文（400）
  - 低资源语言（300）
  - 其他英文/混合网页（900）
- 控制序列长度多样性：样本被截断为 256–4096 tokens，其中 93% 为长上下文（4096 tokens），平均长度约 3899 tokens。
- 所有模型家族使用其专属 tokenizer 对同一原始文本进行分词，避免 tokenizer 差异带来的偏差。

### 实验设置和评估指标
- **模型集合**：共 27 个检查点，主分析含 24 个，额外加入 3 个 Qwen2.5-Instruct 检查点用于 SFT 影响分析。
  - 包括：Qwen2.5、Qwen2.5-VL、Qwen3、Qwen3.5、Gemma2、Gemma3、Ling-mini、GPT-OSS。
- **测量过程**：
  - 前向传播（forward-only），冻结权重；
  - 使用 PyTorch hooks 流式收集六类组件的激活张量；
  - 统计每层每组件的均值、标准差、RMS、最大值、最小值、绝对值分位数等。
- **核心评估指标**：
  - 主要指标：**全局最大激活值 $ M = \max{|a|} $**（跨所有层和组件）；
  - 辅助指标：层间峰值轨迹、携带最大值的组件（carrier component）、局部稀疏性比率（local ratio）；
  - 验证指标：轻量级 INT-8 量化重建误差（SQNR）。

### 基线方法对比
本文并非提出新的量化算法，而是对现有量化方法的前提假设进行检验：
- 对比了传统“outlier detection”视角（如 [Sun et al., 2024] 的 massive activation 判定）；
- 与 SmoothQuant、AWQ、GPTQ、Outlier Suppression+、QuaRot、FlatQuant 等量化技术形成互补关系——这些方法旨在“消除”或“转移”异常值的影响，而本文则直接“测量”这些异常值本身的大小及其在不同模型间的差异。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **全局最大激活值 $ M $ 跨越近四个数量级**：
  - Qwen3.5 和部分 MoE 模型集中在 $10^2\sim10^3$；
  - Gemma3-27B-it 达到 **~696,320 ($\sim7\times10^5$)**，为最高记录。
- **MoE 架构显著降低峰值**：
  - MoE 检查点的峰值比同规模 dense 模型低 **14.0–23.4×**（例如 Qwen3-30B-A3B vs Qwen3-32B）。
- **残差流（residual stream）主导极端值传播**：
  - 在 24 个主分析检查点中，**22 个的全局最大值出现在 layerwise hidden states**（即残差流）；
  - 仅 GPT-OSS-20B 的最大值来自 MLP 输出。
- **训练阶段影响峰值**：
  - 在 Ling-mini 系列中，随着训练 token 数从 5T 增至 20T，$ M $ 单调上升 **1.34×**（7,648 → 10,240）。
- **指令微调（SFT）压缩后期层峰值**：
  - 在 Qwen2.5 系列中，SFT 主要降低了最后几层的激活峰值（下降 31–48%），但中层高激活区域保持不变。

### 与基线方法的对比结果
- **与 Sun criterion 的不一致性**：
  - 24 个检查点中有 4 个未通过 Sun 的 massive activation 判定（×标记）；
  - 但其中一些（如 Qwen2.5-1.5B）虽局部稀疏性不足，绝对值仍很高；另一些（如 Qwen3.5-0.8B）则整体激活被系统性压制。
  - 表明二元判定无法准确反映量化风险，$ M $ 更具实用性。

### 消融实验结果（Matched Comparisons）
| 对比维度 | 发现 |
|--------|------|
| **MoE vs Dense** | MoE 峰值低 14.0–23.4×，表明稀疏路由具有显著抑制作用 |
| **Vision-Language vs Text-only** | Qwen2.5-VL 峰值比文本版低 1.4–1.6×，模态适配伴随适度压缩 |
| **Base vs Instruct** | SFT 主要压缩晚期层峰值，不改变中层结构 |
| **Training Stage** | 更长训练时间与更高 $ M $ 正相关（Ling-mini 序列） |
| **Family/Generation Evolution** | 不同家族趋势相反：<br>- Qwen: 2.5 → 3 ↑ → 3.5 ↓<br>- Gemma: 2 → 3 ↑↑ |

---

## 4. 关键结论和发现

### 论文的主要发现
1. **最大激活值 $ M $ 是一个独立的模型属性**，不能仅由参数量预测，而是由 **model family、architecture、training stage** 共同决定。
2. **MoE 架构能有效抑制极端激活**，相比 dense 模型可降低峰值达 **23.4×**，对低比特部署友好。
3. **残差流是极端激活的主要载体**（22/24 情况下），因此量化策略应重点关注 hidden states 而非仅 attention 或 MLP 输出。
4. **训练进程本身会增加激活峰值**，即使在固定架构下，更长训练也趋向产生更高的 $ M $。
5. **$ M $ 与 INT-8 量化重建误差（SQNR）显著相关**：更高的 $ M $ 导致更低的 SQNR，验证了其作为部署风险指标的有效性。

### 方法的局限性
- **观察性研究而非因果推断**：仅报告相关性，未解释为何某些家族/架构会产生更高或更低的 $ M $。
- **样本覆盖有限**：语料库以英语为主，缺乏对数学推理链、工具调用轨迹等特殊输入的测试。
- **上下文长度限制**：最大仅使用 4096 tokens，无法评估超长上下文（如 32K+）下的行为变化。
- **匹配对数量少**：MoE vs Dense、VL vs Text 等对比仅基于单一家族内的 $ n=2 $ 对，泛化性受限。
- **未深入机制分析**：未结合 residual stream RMS 曲线或 early-layer step-up blocks 等机制定位峰值成因。

### 未来工作方向
- 开展干预实验（intervention experiments）探究峰值形成的因果机制；
- 扩展至更多家族和架构的 matched-pair 分析；
- 探索 $ M $ 与旋转量化（rotation-based）、KV-cache 优化等先进量化技术的成本关系；
- 构建基于 $ M $ 的标准化 model card 字段，推动其成为开源模型发布的常规指标；
- 研究如何通过训练策略主动控制 $ M $，实现“可量化性优先”的模型设计。

> **核心主张**：  
> $ M = \max{|a|} $ 应被视为一种**可发布、应报告的模型属性**，如同参数量、上下文长度一样，应在任何开源权重发布时附带提供，以指导低比特部署决策。

</details>

---

### 8. [Runtime-Orchestrated Second-Order Optimization for Scalable LLM Training](https://arxiv.org/abs/2605.16184)

**Authors**: Yishun Lu, Junhao Zhang, Zeyu Yang, Wes Armour  
**Category**: cs.DC  
**Published**: 2026-05-18  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.16184v1  

#### Abstract
Second-order methods offer an attractive path toward more sample-efficient LLM training, but their practical use is often blocked by the systems cost of maintaining and updating large matrix-based optimizer states. We introduce \textbf{Asteria}, a runtime system designed to remove this bottleneck by...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Runtime-Orchestrated Second-Order Optimization for Scalable LLM Training

## 1. 论文的主要贡献和创新点

### 解决的问题
第二阶优化方法（如 **Shampoo** 和 **SOAP**）在理论上能显著提升大语言模型（LLM）训练的收敛速度和统计效率，但由于其高昂的系统开销而难以实用化。主要瓶颈包括：
- **内存压力**：需要维护巨大的协方差矩阵（如 $L$ 和 $R$），导致 $O(N^2)$ 内存占用，在单 GPU 或统一内存架构（UMA）下极易触发 OOM。
- **计算阻塞**：周期性的 $O(d^3)$ 矩阵求逆或特征分解操作破坏了 GPU 上的计算-通信重叠，造成严重的周期性延迟尖峰（latency spikes）。
- **同步开销**：分布式场景中依赖全局同步更新预条件器状态，受限于慢速的跨节点通信（如 InfiniBand），无法利用拓扑感知的异步机制。

### 提出的新方法：Asteria
作者提出 **Asteria** —— 一个硬件-软件协同设计的运行时系统，通过将二阶优化逻辑从关键 GPU 路径中解耦来解决上述问题。其三大核心创新如下：

#### （1）Architecture-Adaptive Asymmetric Memory Tiering  
针对二阶优化中不同张量的生命周期差异，采用分层内存管理策略：
- **factor_matrices** 存放于 GPU 显存，用于梯度累积；
- **inverse_factor_matrices** 存放于 CPU 可访问的 UVM / DRAM 中，由 CPU 异步计算并更新；
- 支持可选的 **NVMe 后端** 进行冷数据暂存，结合 `io_uring` 实现高效 I/O 和页面回收（`madvise(MADV_DONTNEED)`）；
- 利用 CUDA 的 `cudaMemPrefetchAsync` 实现按需预取，避免冗余复制。

> ✅ 优势：突破“垂直容量墙”，使 1B 参数模型可在仅 128GB 统一内存 + 单 GB10 GPU 的设备上完成二阶训练。

#### （2）Hook-Orchestrated Shadow-State Pipeline  
引入基于 **PyTorch FSDP hooks** 的轻量级调度机制，实现非侵入式的异步流水线：
- 在 `forward_hook` 和 `full_backward_prehook` 触发时，启动低优先级 CUDA 流进行状态预取；
- 主 GPU 流继续执行前向/反向传播，使用带有有限陈旧度（bounded staleness）的缓存预条件器；
- CPU 工作线程池异步执行 `matrix_inverse_root` 计算，完成后回写至 host-resident 状态区。

> ✅ 优势：完全消除 $O(d^3)$ 操作对主训练路径的影响，平滑 step time 曲线，接近一阶方法的执行稳定性。

#### （3）Bounded-Staleness Selective Coherence  
在多节点训练中，摒弃传统的全量同步模式，转而采用拓扑感知的选择性同步协议：
- 构建 **Topology Cost Graph** 区分 intra-node（NVLink）与 inter-node（InfiniBand）通信代价；
- 维护每个预条件器块的版本号与最后同步步数，仅当陈旧度超过阈值 $S$ 时才参与同步；
- 同步路径直接作用于 CPU 内存中的 `inv_factor_matrices`，避免 GPU ↔ CPU 多次拷贝；
- 层次化聚合：先在节点内平均 → 节点代表间同步 → 广播回本地。

> ✅ 优势：降低通信总量，缓解“全局共识墙”问题，提升强扩展性（strong scaling）表现。

---

## 2. 核心实验方法和设置

### 数据集与模型
- **数据集**：English C4 corpus
- **Tokenizer**：T5 tokenizer
- **序列长度**：1024 tokens
- **模型系列**：OLMo-family
  - **660M**：d_model=1408, 24 层，22 注意力头（GELU）
  - **OLMo-2 1B**：d_model=2048, 24 层，16 头（SwiGLU + RMSNorm）
  - **OLMo-2 7B**：d_model=4096, 32 层，32 头，mlp_hidden_size=22016

### 实验平台
- **Scale-up 设置**：NVIDIA DGX Spark（单 GB10 GPU + 128GB Unified Memory）
- **Scale-out 设置**：多节点 NVIDIA GH200 集群（支持 up to 16 nodes）

### 基线方法对比
| 方法 | 类型 |
|------|------|
| **AdamW** | 一阶优化器（baseline） |
| **SOAP**, **KL-Shampoo** | 原生二阶优化器（native） |
| **Asteria-SOAP**, **Asteria-KL-Shampoo** | 本文提出的运行时增强版本 |

所有方法均基于 **FSDP** 框架，预处理流程一致，学习率通过调优选择最优验证损失配置。

### 评估指标
- **Step Time 分布与分解**：衡量系统稳定性与延迟隐藏能力
- **Training Loss vs Step / Wall Time**：评估收敛行为与时钟时间效率
- **Energy Consumption & Power Profile**：使用板载硬件遥测（hwmon + NVML）测量 SoC、CPU、GPU 能耗
- **Normalized Loss-Reduction Efficiency**：
  $$
  \text{Efficiency}_i = \frac{L_{\text{init}} - L_{\text{final},i}}{E_i / E_{\text{AdamW}}}
  $$
  其中 $L_{\text{init}} \approx 10.377$ 为均匀分布下的初始交叉熵。
- **Strong Scaling Speedup**：固定总 workload 下随节点增加的速度提升

---

## 3. 主要实验结果和性能指标

### （1）在 DGX Spark 上的表现（内存受限场景）

#### Step Time 性能
- **原生 KL-Shampoo / SOAP**：出现明显周期性延迟尖峰（pf=10），峰值达 **96.0s** 与 **80.7s**
- **Asteria 版本**：step time 分布极其平稳，分别降至 **66.5s** 与 **67.9s**，相比原生减少约 **30–40 秒/step**

> 🔹 图 5 显示，Asteria 成功将 preconditioning 时间从关键路径中剥离，实现了近 100% 的延迟隐藏。

#### 能效分析（图 6 & 7）
| 方法 | 总能耗（SoC, % of AdamW） | GPU 能耗 | CPU 能耗 | 归一化损失缩减效率 |
|-------|----------------------------|----------|-----------|------------------------|
| AdamW | 100% | 100% | 100% | 3.0942 |
| Native SOAP | 119.7% | 112.1% | 119.0% | 3.0685 |
| Asteria-SOAP | 114.1% | 114.1% | 128.3% | **3.1299** ↑ |
| Native KL-Shampoo | 117.1% | 115.6% | 125.1% | 3.0562 |
| Asteria-KL-Shampoo | **107.9%** | **107.3%** | 122.5% | **3.3010** ↑↑ |

> ✅ **关键发现**：尽管 CPU 能耗上升（因后台计算），但 GPU 更高效地被利用，总体能耗下降，且每单位能量带来的 loss reduction 显著优于所有基线，**首次实现二阶优化在能效上的超越**。

---

### （2）在 GH200 多节点集群上的表现

#### 收敛效率 vs 墙钟时间（图 8）
- 所有二阶方法在 step-wise 收敛上均优于 AdamW；
- **Asteria 变体在 wall-clock time 上显著快于原生二阶方法**，更早达到相同 loss 水平；
- 表明：**Asteria 不改变优化轨迹，但极大提升了实际训练吞吐**。

#### 陈旧度容忍实验（图 9）
- 当异步陈旧度 $S=1$ 时，runtime gain 有限；
- 当 $S≥3$ 时，训练时间大幅下降并趋于饱和；
- 最终评估 loss 在 $S∈{1,2,3,5,10}$ 下保持稳定，说明 **bounded staleness 不损害最终模型质量**；
- 默认采用 $S=5$ 作为平衡点。

#### 大规模扩展性（图 10 & 11）
- 对 **1B 和 7B 模型**，Asteria-SOAP/KL-Shampoo 在 wall-time 上持续领先 AdamW，并紧贴原生二阶曲线；
- **强扩展性测试（7B 模型）**：
  - 在 16 节点下，Asteria-KL-Shampoo 相比原生版本实现更高 speedup 和更低 step time；
  - 尤其在高并行度下，暴露的二阶开销占比更大，Asteria 的优势更加显著。

---

## 4. 关键结论和发现

### 主要发现
1. **二阶优化的实用性瓶颈是系统层面而非算法本身**：主要障碍在于状态管理方式与现代加速器架构不匹配。
2. **通过运行时解耦可彻底消除 $O(d^3)$ 延迟尖峰**：Asteria 实现了与一阶方法相当的 step time 稳定性和硬件利用率。
3. **无需大规模 GPU 内存聚合即可支持二阶训练**：借助生命周期感知的异构内存分级，可在单节点甚至边缘设备部署二阶优化。
4. **更好的编排带来更高的物理效率**：Asteria 不仅加快训练，还改善了 **energy-loss tradeoff**，使得二阶优化成为更节能的选择。
5. **分布式训练应将预条件器视为 runtime-managed resource**：通过 bounded-staleness + topology-aware synchronization 可有效降低通信负担。

### 方法的局限性
- 当前实现主要适配 **FSDP** 框架，尚未集成到其他并行范式（如 Tensor Parallelism、Pipeline Parallelism）；
- 在极端窄带宽 PCIe 平台上的性能有待验证；
- NVMe I/O 路径依赖 `io_uring`，对操作系统有一定要求；
- 异步机制引入额外复杂性，调试难度略增。

### 未来工作方向
- 扩展至更广泛的并行训练架构（TP/PP/SP）；
- 探索自动调优机制以动态调整 staleness budget $S$ 和 offload 策略；
- 结合量化技术进一步压缩 host-resident 状态体积；
- 将类似思想应用于其他高开销优化器（如 K-FAC、Natural Gradient）；
- 开发统一的“Optimizer Runtime”抽象层，供多种高级优化器复用。

---

> 📌 **总结一句话**：  
> **Asteria 证明了——二阶优化能否实用化，不仅取决于数学设计，更取决于运行时系统如何管理状态放置、异步计算与选择性同步。通过软硬协同的精细化控制，它让二阶优化真正具备了在现实 LLM 训练中落地的能力。**

</details>

---

### 9. [Differentiable Mixture-of-Agents Incentivizes Swarm Intelligence of Large Language Models](https://arxiv.org/abs/2605.15706)

**Authors**: Xingjian Wu, Junkai Lu, Siyu Yan, Xiangfei Qiu, Jilin Hu, Chenjuan Guo, Bin Yang  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.15706v1  

#### Abstract
Recent advances in Large Language Models (LLMs) have catalyzed the development of multi-agent systems (MAS) for complex reasoning tasks. However, existing MAS typically rely on pre-defined or pre-compiled communication topologies, which limits their flexibility and adaptability to dynamic task requi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Differentiable Mixture-of-Agents Incentivizes Swarm Intelligence of Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Multi-Agent Systems (MAS)** 通常依赖于预定义或“预编译”的通信拓扑（如 Chain、Star、Graph），这种静态结构在面对动态任务需求时缺乏灵活性。尤其在复杂推理过程中，预先设计的工作流可能无法适应执行过程中的变化，导致资源浪费或解决方案不足。

此外，当前的“自演化”MAS 虽然能根据查询生成条件图结构，但仍需**先设计再编译**，存在“静态编译困境”，难以实现细粒度的时空弹性调整。

### 提出的新方法：Differentiable Mixture-of-Agents (DMoA)
作者提出 **DMoA** ——一种**可微分、自演化的多智能体框架**，其核心思想是将 Mixture-of-Agents (MoA) 与 Mixture-of-Experts (MoE) 的理念结合，在每一步推理中动态路由并激活最合适的 agent，从而隐式模拟任意通信拓扑。

#### 创新点：
- ✅ **Step-wise 自适应路由机制**  
  不同于传统 MAS 在整个流程中固定 agent 数量与路径，DMoA 在每个推理步骤中基于上下文信息动态选择 agent 子集，实现**时空无界性**（spatio-temporal unboundedness）。
  
- ✅ **可微分的上下文感知路由器 (Differentiable Context-aware Router)**  
  设计了一个由 **Sentence Transformer + GRU-based RNN-Router** 构成的轻量级、端到端可训练的路由机制，能够融合历史决策和当前语义上下文，输出 agent 的稀疏激活权重。

- ✅ **基于预测熵的自监督优化 (Predictive Entropy as Self-supervised Signal)**  
  引入 **Predictive Entropy (PE)** 作为衡量 agent 置信度的指标，并通过 **pair-wise ranking loss** 对齐路由 logits 与 agent 表现顺序，无需外部标注即可进行 test-time training。

---

### 相比现有方法的优势
| 维度 | 传统 MAS / 图方法 | DMoA |
|------|------------------|-------|
| 拓扑灵活性 | 固定或任务级动态（round-wise） | 步骤级动态（step-wise），弹性扩展 |
| 编译方式 | 需“设计-编译”工作流 | 无需显式编译，实时路由 |
| 优化信号 | 强化学习（稀疏奖励） | 自监督熵信号（密集、易获取） |
| 泛化能力 | 需 few-shot 微调 | 支持 zero-shot test-time training |
| 效率 | 全图执行开销大 | 稀疏激活，按需调用 |

---

## 2. 核心实验方法和设置

### 使用的数据集（共9个）
涵盖多种现实世界推理任务：

| 数据集 | 任务类型 | 评估指标 |
|--------|----------|-----------|
| **MMLU** | 广域知识推理 | Accuracy |
| **GSM8K**, **MultiArith**, **SVAMP**, **AQuA** | 数学推理 | Accuracy |
| **HumanEval**, **DS-1000** | 代码生成与调试 | Pass@1 / Accuracy |
| **HotpotQA** | 多跳问答 | Accuracy / F1 |
| **DDXPlus** | 医疗诊断推理 | Accuracy |

> 所有数据集均采用标准划分与官方评估协议。

---

### 实验设置
- **Backbone 模型**：统一使用 `gpt-oss-120b` 作为所有 agent 和 baseline 的底层 LLM。
- **Agent Pool**：包含不同角色（如 Math Solver、Coder、Inspector）、工具（Calculator、Python、Web Search）和提示模板的异构 agent 集合。
- **Summarizer Agent**：负责聚合中间结果并决定是否终止推理。
- **最大推理步数**：设为 20；也可由 summarizer 自主判断终止。
- **路由参数**：
  - Sentence Transformer: `all-MiniLM-L6-v2` (D=384)
  - Router: GRU + Linear Head
  - 温度系数 $ T = 0.1 $
  - 最大路由数量 $ K $

---

### 基线方法对比
分为三类进行比较：

#### （1）单智能体系统（Single-Agent Systems）
- CoT (Chain-of-Thought)
- Self-Consistency (SC)

#### （2）多智能体系统（Multi-Agent Systems）
- 静态拓扑：Chain, Tree, Star, Complete Graph, Random Graph
- 工具增强：AutoGen, MoA, LLM-Debate

#### （3）自演化多智能体系统（Self-Evolving MAS）
- **空间自适应**：GPTSwarm, G-Designer, ARG-Designer, SafeSieve
- **时间自适应**：AFlow, SpecReason, STEER

> 所有可优化方法均在相同 few-shot 设置下训练（40–80 条样本用于适配）。

---

## 3. 主要实验结果和性能指标

### 性能总览（Table 2）
在 **9 个基准测试上的平均准确率达到 89.38%**，显著优于所有基线：

| 方法类别 | 代表方法 | Avg. Accuracy |
|---------|--------|---------------|
| 单智能体 | Vanilla | 74.22 |
| 静态 MAS | Complete Graph | 78.57 |
| 自演化 MAS | SafeSieve | 83.91 |
| **本文方法** | **DMoA (Ours)** | **89.38** ✅ |

> 相较于 vanilla 提升高达 **15.16 个百分点**，且在最难的 DS-1000 和 DDXPlus 上分别提升 **25.94** 和 **26.97**。

---

### 关键性能亮点
- 在 **GSM8K** 上达到 **98.87%** 准确率（+11.72 vs Vanilla）
- 在 **DS-1000** 达到 **64.34%**（+25.94），远超其他方法
- 在 **DDXPlus** 医疗诊断任务上达 **83.37%**，表现最强鲁棒性

---

### Test-Time Training (TTT) 实验（Table 3）
验证了 DMoA 支持 **zero-shot 自适应能力**：

| 方法 | MMLU | GSM8K | HumanEval | DS-1000 | HotpotQA | Avg |
|------|------|--------|------------|----------|-----------|-----|
| DMoA (Few-shot) | 91.35 | 98.87 | 95.62 | 64.34 | 90.38 | 88.11 |
| DMoA (TTT) | 91.80 | 98.65 | 96.04 | 65.44 | 89.50 | 88.29 |
| **DMoA (Few-shot + TTT)** | **93.50** | **99.30** | **97.52** | **65.55** | **91.40** | **91.45** |

✅ 表明 DMoA 可在测试阶段利用前几个 query 的 entropy 信号自我优化，具备**终身学习潜力**。

---

### Ensemble 能力对比（Table 4）
使用多个开源 LLM 构建 agent pool 后，DMoA 的 ensemble 效果甚至超过闭源强模型：

| 方法 | MMLU | GSM8K | HumanEval | DS-1000 | HotpotQA |
|------|------|--------|------------|----------|-----------|
| GPT-4.2 | 83.72 | 88.56 | 79.32 | 42.39 | 84.35 |
| Gemini 3.2 Pro | 86.22 | 86.82 | 85.48 | 44.50 | 84.90 |
| **DMoA (Few-shot+TTT)\*** | **85.52** | **88.74** | **86.72** | **48.97** | **86.50** |

> 注：\* 表示仅使用开源 LLM 构建的 DMoA，仍能在多数任务上媲美甚至超越闭源大模型。

---

### 消融实验（Ablation Study, Table 5）
验证各组件有效性：

| 变体 | MMLU | GSM8K | HumanEval | DS-1000 | HotpotQA |
|------|------|--------|------------|----------|-----------|
| w/ LLM Selector (GPT-4.2) | 85.56 | 92.84 | 87.21 | 54.33 | 84.72 |
| w/ Linear Router | 89.76 | 96.84 | 92.11 | 60.02 | 88.71 |
| w/o Aggregation | 89.58 | 96.67 | 91.94 | 59.86 | 88.83 |
| w/o Adaptive $ k_i $ | 90.47 | 97.58 | 93.26 | 61.72 | 89.41 |
| **Full DMoA** | **91.35** | **98.87** | **95.62** | **64.34** | **90.38** |

✅ 结果表明：
- RNN-Router 显著优于线性或 LLM 控制器；
- 上下文聚合机制对性能至关重要；
- 动态控制 $ k_i $ 是实现高效稀疏激活的关键。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **DMoA 实现了真正意义上的弹性多智能体协作**  
   通过 step-wise 可微分路由，系统可在运行时动态构建通信拓扑，突破了传统“预编译”范式的限制。

2. ✅ **Predictive Entropy 是有效的自监督信号**  
   无需人工标注即可驱动路由器优化，支持 test-time training，使系统具备在线适应新任务的能力。

3. ✅ **稀疏激活 + 按需调度 = 更高效率与更强性能**  
   如 Figure 5 所示，DMoA 在更低 token 消耗下实现了更高准确率，说明其能有效协调计算资源。

4. ✅ **具备卓越的 ensemble capability**  
   能整合多个弱开源 LLM 的能力，形成超越单一闭源模型的群体智能（swarm intelligence）。

5. ✅ **鲁棒性强于静态与部分自演化方法**  
   在对抗性攻击实验中（Figure 3 & 7），DMoA 性能下降最小，因其可通过路由机制规避低置信 agent 的干扰。

---

### 局限性（Limitations）
1. **Agent Pool 规模受限**  
   当候选 agent 数量 $ N $ 过大时，路由难度上升，需更多训练数据或更复杂的路由先验。

2. **Context-Length 敏感性**  
   当前使用 Sentence Transformer 压缩上下文为固定向量，可能丢失长程依赖细节，影响深层推理质量。

3. **训练成本较高**  
   尽管推理稀疏，但在训练/TTT 阶段需运行全部 agent 收集 entropy，带来额外开销。

---

### 未来工作方向
- 探索更高效的 agent 枚举机制，避免穷举所有 LLM×Profile×Tool 组合；
- 使用 backbone LLM 替代 Sentence Transformer 进行上下文编码；
- 将 DMoA 应用于真实场景（教育辅导、科研助手、自动化运维等）；
- 结合 memory 机制实现跨 query 的经验积累，迈向真正的 lifelong learning agent system。

---

> 📌 **一句话总结**：  
> DMoA 通过引入**可微分、上下文感知、熵驱动的 step-wise agent 路由机制**，首次实现了无需预编译的弹性多智能体协作框架，在性能、效率、鲁棒性和泛化能力上全面领先，为构建可持续进化的 LLM swarm 提供了新范式。

</details>

---

### 10. [See Before You Code: Learning Visual Priors for Spatially Aware Educational Animation Generation](https://arxiv.org/abs/2605.15585)

**Authors**: Yuejia Li, Ke He, Junheng Li, Shutong Chen, Jingkang Xia, Zhiyue Su, Junchi Zhang, Mang Ye  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.15585v1  

#### Abstract
Large language models can generate executable code for educational animations, but the resulting renders often exhibit visual defects, including element overlap, misalignment, and broken animation continuity. These defects cannot be reliably detected from the code alone and become apparent only afte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：See Before You Code: Learning Visual Priors for Spatially Aware Educational Animation Generation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLM）虽然能够生成可执行的教育动画代码（如 Manim 脚本），但其渲染结果常出现**视觉缺陷**，例如：
- 元素重叠（element overlap）
- 错位（misalignment）
- 动画连续性断裂（broken animation continuity）

这些问题是**仅从代码无法可靠检测**的，必须在渲染后才能发现。传统方法缺乏对空间布局的显式建模，导致“**LLM couldn’t see before coding**”这一根本瓶颈。

作者将此问题形式化为 **render-feedback-aware constrained code generation**：给定自然语言描述，生成的代码不仅要可执行，其**渲染输出还需满足只能在渲染后评估的结构化质量标准**。

---

### 提出了什么新方法或新思路
提出 **OmniManim** —— 一个面向渲染反馈的教育动画生成框架，核心思想是**分离语义解析与视觉规划**，引入任务专用的 **Vision Agent** 进行显式的视觉先验学习。

#### 主要创新点：
- ✅ **共享场景状态（Shared Scene State）**  
  所有代理通过统一的结构化表示进行协作，提升透明度与可修复性。
  
- ✅ **Vision Agent：显式视觉规划模块**  
  - 预测稀疏关键帧（sparse keyframes）的粗到细（coarse-to-fine）边界框布局
  - 使用 **bounding-box denoising** 技术优化布局
  - 引入 **interpolation-aware objective**，防止中间帧因插值产生冲突（如重叠、遮挡）

- ✅ **结构化渲染反馈循环（Structured Render Feedback Loop）**  
  利用确定性的计算机视觉分析诊断渲染缺陷（如重叠、越界），并指导局部修复而非全量重生成。

- ✅ **构建两个新数据集**  
  - `ManimLayout-1K`：用于训练 Vision Agent 的 1K 教育动画及其布局标注
  - `EduRequire-500`：独立构建的评测基准，覆盖多学科、多任务类型

---

### 相比现有方法的优势
| 对比维度 | 传统单模型/多代理方法 | OmniManim |
|--------|----------------------|----------|
| 视觉控制能力 | 弱，依赖 LLM 隐式推理 | 显式视觉规划 + 渲染反馈 |
| 布局可靠性 | 容易出现重叠、错位 | 显著减少空间冲突 |
| 可解释性与可修复性 | 黑箱生成 | 结构化状态支持局部修正 |
| 动画连贯性 | 插值失败常见 | 插值感知目标函数提前规避风险 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
| 数据集 | 类型 | 规模 | 来源与用途 |
|-------|-----|------|-----------|
| **ManimLayout-1K** | 训练集 | 1,000 个源动画 → 22,579 个关键帧样本 | 从 GitHub 和社区教程中收集并清洗，提取结构化场景图与归一化边界框 |
| **EduRequire-500** | 测试集 | 500 个独立撰写的教学需求 | 由领域专家撰写，涵盖 120 学科标签、16 种任务类型（如概念讲解、数学推导、数据分析等） |

> 注：两者互斥，避免数据泄露；计划开源以促进复现。

---

### 实验设置和评估指标

#### 生成协议
- 每个任务运行 5 次，报告均值
- 最多允许一次修复轮次
- 统一环境：Python 3.12, Manim 0.19.0

#### 评估指标（见 Table 1）
分为四大类：

| 组别 | 指标缩写 | 含义 |
|------|--------|------|
| **Executability** | R@1, R@F | 首次渲染成功率 / 最终成功率 |
| **Instructional Quality** | CA, PC, EN | 内容准确性、教学清晰度、吸引力 |
| **Visual Quality** | OV, LQ, AC, VC | 重叠程度、布局质量、动画连续性、视觉一致性 |
| **Efficiency** | Tok, Time | 总 token 数、端到端耗时 |

> 评估方式结合：
> - **CV-based 自动指标**（基于像素分析）
> - **VLM-as-Judge**（Claude Opus 4.6 多阶段打分）
> - **人工评估**（60 个任务，20 名本科生评分）

---

### 基线方法对比
| 类型 | 基线方法 |
|------|---------|
| **Single-Model Baselines** | GPT-5.4, Kimi K2.5, Gemini 3.1 Pro, MiniMax-M2.7, Qwen3-14B |
| **Multi-Agent Baseline** | Code2Video（代表性的角色分工多代理系统） |

所有方法使用相同提示词和环境，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 方法 | R@1 ↑ | R@F ↑ | OV ↑ | LQ ↑ | AC ↑ | VC ↑ | Tok ↓ | Time ↓ |
|------|-------|-------|------|------|------|------|--------|--------|
| GPT-5.4 | 0.812 | 0.934 | 0.628 | 0.778 | 0.764 | 0.748 | 56K | 108s |
| Code2Video (GPT-5.4) | 0.841 | 0.923 | 0.598 | 0.742 | 0.784 | 0.748 | 69K | 148s |
| **OmniManim (GPT-5.4)** | **0.907** | **0.986** | **0.763** | **0.931** | **0.802** | **0.762** | 73K | **72s** |
| **OmniManim (Gemini)** | **0.884** | **0.972** | **0.778** | **0.942** | **0.818** | **0.781** | 75K | **78s** |

> ✅ OmniManim 在几乎所有视觉质量指标上显著领先，尤其在 **Overlap (OV)** 和 **Layout Quality (LQ)** 上优势明显  
> ⚠️ Token 消耗更高，但得益于部分并行化，**wall-clock time 更短**

---

### 与基线方法的对比结果
- **相比单模型方法**：OmniManim 在 OV 上平均提升 >130%，LQ 提升约 15–20%
- **相比 Code2Video**：尽管后者也是多代理架构，但在空间组织方面仍落后于 OmniManim
  - OV 提升达 +32.1 分（人类评分）
  - LQ 提升 +21.4 分（人类评分）
- **效率优势**：尽管流程更复杂，但由于 Vision Agent 推理与 LLM 生成可并行，总时间反而更低

---

### 消融实验结果（Table 4 & Figure 4）

| 设置 | OV ↑ | LQ ↑ | AC ↑ | VC ↑ |
|------|------|------|------|------|
| Full Vision Agent | 0.763 | 0.931 | 0.802 | 0.762 |
| w/o Linterp（无插值感知目标） | 0.734 (-0.029) | 0.907 (-0.024) | 0.750 (-0.052) | 0.754 (-0.008) |
| w/o Vision Agent | 0.491 (-0.272) | 0.614 (-0.317) | 0.576 (-0.226) | 0.648 (-0.114) |
| Stage 1 Only（仅有粗略先验） | 0.628 (-0.135) | 0.783 (-0.148) | 0.710 (-0.092) | 0.708 (-0.054) |
| Stage 2 Only（仅有精细去噪） | 0.572 (-0.191) | 0.726 (-0.205) | 0.665 (-0.137) | 0.691 (-0.071) |

#### 发现：
- **Vision Agent 是核心驱动力**：移除后所有指标大幅下降
- **粗粒度先验 + 精细去噪互补**：单独使用任一阶段效果有限
- **插值感知目标至关重要**：显著提升动画连续性（AC↓0.052），说明有效缓解了中间帧冲突

> 图 4 定性展示也表明，随着各组件加入，布局逐渐清晰稳定。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **空间布局失败是 LLM 生成教育动画的根本瓶颈**，即使采用多代理架构也无法解决。
2. 🧠 **显式视觉规划优于隐式语言推理**：Vision Agent 通过学习视觉先验，在代码生成前就规避潜在的空间冲突。
3. 🔄 **渲染应纳入生成闭环**：OmniManim 将 rendering 作为反馈信号，实现“see before you code”的理念。
4. 🎯 **interpolation-aware objective 是关键设计**：不仅保证关键帧合法，还确保插值过程安全，极大提升了动画流畅性。
5. 👥 **人工评估验证了教学价值提升**：OmniManim 输出更具条理性和吸引力，尤其在重叠控制和布局协调方面获得高度认可。

---

### 方法的局限性
- 📏 **局限于 Manim 风格布局**：未处理丰富排版、相机运动、长叙事逻辑
- 🖋️ **当前仅建模对象级边界框**：未深入建模字体、图形细节或艺术风格
- ⏳ **短期关键帧规划**：难以应对超长时间跨度动画的时间依赖
- 💬 **仍需人工审核最终内容**：辅助创作工具，非全自动替代方案

---

### 未来工作方向
- 支持更复杂的视觉表达（typography, camera motion）
- 扩展至开放域视频生成（open-domain video generation）
- 引入长期时间建模机制（long-range temporal planning）
- 探索轻量化 Vision Agent 以降低计算成本
- 构建更大规模、更多样化的教育动画数据集

---

> ✅ **总结一句话**：OmniManim 通过引入 **Vision Agent + 显式视觉规划 + 渲染反馈闭环**，实现了“看得见才编码”的教育动画生成范式，在布局质量和动画连贯性上全面超越现有方法。

</details>

---

### 11. [parallelcbf: A composable safety-filter and auditability framework for tensor-parallel reinforcement learning](https://arxiv.org/abs/2605.15509)

**Authors**: Yijun Lu, Zilei Yang, Yuyin Ma  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.15509v1  

#### Abstract
While Isaac Lab provides massive parallel UAV simulation, OmniSafe and safe-control-gym provide constrained-RL benchmarks, and CBFKit provides control-barrier-function synthesis tooling, no existing framework unifies these capabilities for end-to-end safety-constrained training. ParallelCBF is the f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ParallelCBF: A Composable Safety-Filter and Auditability Framework for Tensor-Parallel Reinforcement Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前安全关键型强化学习（safety-critical RL）研究面临**结构性碎片化**问题：
- 不同工具各司其职：Isaac Lab 提供高吞吐仿真，OmniSafe 提供约束RL算法，CBFKit 支持控制屏障函数（CBF）设计，但缺乏统一框架整合这些能力。
- 安全过滤器、行为克隆到RL的warmstart流程、以及可复现性所需的运营规范（如预注册、看门狗机制、故障溯源）均依赖用户自行实现，导致实验难以复现、错误易静默传播。

### **提出的新方法与创新思路**
ParallelCBF 是首个将以下四方面能力统一的开源框架：

#### **(1) SafetyFilter 抽象层（Layer 2）**
- 引入 `SafetyFilter` 抽象接口，解耦 barrier function 计算与环境动力学。
- 提供基于 PyTorch 的 **dual-barrier CBF 实现**：
  - `h_hard(x) = ||r||² - R²`：平方硬屏障，确保物理非碰撞。
  - `h_soft(x,v) = |r| - R - Dₜ(v)`：线性预测屏障，吸收执行器延迟与制动距离。
- 支持向量化批量处理（tensor-parallel），在单障碍场景下闭式求解 QP，无额外正则项（避免动作收缩病态）。

#### **(2) 四层可组合架构（Four-Layer Composable API）**
| 层级 | 功能 |
|------|------|
| `parallelcbf.envs` | 定义 `SafeEnv` 接口，扩展 `gymnasium.Env`，支持安全状态输出 |
| `parallelcbf.safety` | 实现 `SafetyFilter`，提供 CBF 安全过滤 |
| `parallelcbf.algorithms` | 类似 Stable-Baselines3 的 `Algorithm` ABC，支持 `learn`, `predict`, `save/load` |
| `parallelcbf.ops` ✅（最核心创新） | 首创“操作审计层”，将科研实践规范化为 API |

#### **(3) 操作审计层（Operational Auditability Layer）——最大架构创新**
首次将科研运营纪律提升为**版本化、类型检查、测试覆盖的一等公民 API**，而非脚本或约定：
- `PreRegistration`：训练前预注册实验配置（奖励函数、课程安排、阈值等），生成 SHA-256 哈希承诺，防止事后篡改。
- `WatchdogRegistry`：中心化监控指标，触发条件 halt。
- `FailureForensics`：滚动缓冲诊断信息，halt 时自动保存至 JSON 文件。
- `AtomicCheckpoint`：原子化 checkpoint 写入（`.tmp → fsync → rename`）。
- `DatasetAudit`：对数据集进行分布、完整性、BPTT 合法性审计。

> 🔑 **核心主张**：该审计层不是“有用”，而是**可复现实证机器人研究的必要条件**。

### **相比现有方法的优势**
| 能力 | Isaac Lab | OmniSafe | safe-control-gym | CBFKit | **ParallelCBF** |
|------|---------|----------|------------------|--------|----------------|
| 张量并行仿真 | ✅ | ❌ | ❌ | ❌ | ✅（toy 环境） |
| CBF 安全过滤器 | ❌ | ❌ | ⭕（单环境） | ✅（理论） | ✅（batched, dual-barrier） |
| 可组合安全包装器 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 操作审计原语 | ❌ | ❌ | ❌ | ❌ | ✅✅✅（首创） |
| 行为克隆流水线支持 | ❌ | ❌ | ❌ | ❌ | ✅（sharded BC） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **自建行为克隆数据集（V23）**：
  - 总尝试次数：50,000 次 rollout
  - 成功 episode 数：31,415
  - 场景类型：`open`, `single`, `multi`, `dynamic` 四类避障任务
  - 数据集锚定于 SHA-256 哈希：`50e59d2f...`，可通过 `DatasetAudit` 验证完整性。

### **实验设置**
#### **环境**
- 使用内置 toy 环境：`Toy2DAvoidanceVecEnv`（2D 点质量无人机避障）
- 支持 `num_envs ∈ {1,2,4}` 并行 rollout
- 单圆形障碍物，定义 `h_hard` 和 `h_soft`

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **Per-bucket yield** | 各难度场景的成功率 |
| **Aggregate yield** | 整体成功率，用于验证预注册预测 |
| **Test suite runtime** | 全部 39 个测试用例运行时间（<2s） |
| **Validation loss** | 行为克隆训练中的损失下降情况 |
| **Stage 0 success rate** | 在混合场景 rollout 中达到目标的比例（目标 ≥85%） |
| **Hard-barrier violation rate** | 是否发生碰撞（应为 0%） |
| **Out-of-arena rate** | 是否飞出边界（covariate shift 指标） |

#### **基线方法对比**
- **无直接基线框架对比**（因功能不重合），但通过消融方式体现价值：
  - 对比不同 policy backbone（GRU vs Transformer）在同一框架下的表现。
  - 对比有无 `PreRegistration` 和 `Watchdog` 的 pipeline 执行差异。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 数据收集与预注册验证（Section 6.1）**
| 场景类型 | 尝试数 | 接受数 | 观测成功率 | 预测成功率 | 偏差（pp） |
|--------|-------|--------|------------|------------|-----------|
| Open | 9,000 | 8,971 | 99.68% | 100.00% | -0.32 |
| Single | 18,000 | 13,240 | 73.56% | 72.50% | +1.06 |
| Multi | 13,000 | 5,796 | 44.58% | 40.00% | +4.58 |
| Dynamic | 10,000 | 3,408 | 34.08% | 31.00% | +3.08 |
| **Aggregate** | **50,000** | **31,415** | **62.83%** | **60.70%** | **+2.13** |

> ✅ 实际产出高于预估，说明 dry-run 估计可能系统性低估困难场景成功率。

#### **(2) 测试套件性能（Section 6.2）**
- **39 个 pytest 用例**，包含：
  - Hypothesis property tests（验证 `h_hard ≥ 0` 不变性）
  - 向量化广播形状一致性测试（`num_envs ∈ {1,2,4}`）
  - 种子可复现性测试
  - 序列化 round-trip 测试
- **总运行时间：1.67 秒**（GitHub Actions 单线程）
- 覆盖率：整体 ≥90%，其中 `safety/` 达 91%

> 📌 意义：下游用户可在 <2 秒内完成完整安全属性验证，形成“可复现预算”。

#### **(3) 端到端流水线 halt 演示（Section 6.3）**
- 下游 BC 预训练阶段最终 loss = 0.0595
- 但未满足预注册的收敛标准
- `WatchdogRegistry` 触发 halt，阻止失败 checkpoint 进入 critic-warmup 或 PPO 阶段
- `FailureForensics` 成功保存 loss 曲线、梯度范数、激活统计

> ✅ 实现“结构性阻断”，防止计算资源浪费于劣质 checkpoint。

#### **(4) Layer 3 算法替换实验（Section 6.4）**
| Epoch | GRU (65K params) val loss | Transformer (3.17M params) val loss |
|-------|----------------------------|-------------------------------------|
| 1     | —                          | 0.0556                              |
| 2     | —                          | <0.0595 ✅                            |
| 5     | —                          | 0.0263                              |
| 49    | 0.0595                     | **0.00603** ✅（↓一个数量级）         |

> ✅ Transformer 快速超越 GRU 极限，证明框架支持高效算法迭代。

#### **(5) Stage 0 Rollout Evaluation（Section 6.5）**
| 终止类别 | 数量 (%) |
|----------|----------|
| Goal reached | 0 (0.00%) |
| Hard-barrier collision | **0 (0.00%)** ✅ |
| Out-of-arena | 200 (100.00%) |
| Timeout | 0 (0.00%) |

> - ❌ 成功率 0%，未达 ≥85% 标准
> - ✅ **零碰撞**：验证了 `DualBarrierCBF` 在对抗性策略下的有效性（Transformer 主动靠近障碍）
> - 🔧 发现 Layer 1 bug：原 `info["collision"]` 错误标记 arena exit 为 collision，通过 `FailureForensics` 快速定位修复

---

## **4. 关键结论和发现**

### **主要发现**
1. **审计层是可复现性的基础设施**：
   - `PreRegistration + Watchdog + FailureForensics` 形成闭环，使实验 halt 成为“合同行为”而非“主观判断”。
   - 成功阻止劣质 checkpoint 向下游传播，节省大量计算资源。

2. **安全过滤器具有强鲁棒性和独立性**：
   - 即使 policy 完全失效（100% out-of-arena），`DualBarrierCBF` 仍能保证 `h_hard ≥ 0`，实现 **0% 碰撞**。
   - 安全保障不依赖 policy 能力，体现了 **compositional safety** 设计优势。

3. **框架支持安全的算法替换**：
   - 更换 Layer 3 policy（GRU → Transformer）无需修改 Layer 1/2/4。
   - 全部测试通过，mypy 类型检查不变绿，证明接口隔离成功。

4. **dry-run 估计存在偏差**：
   - 小样本 dry-run（~1k episodes）会系统性低估困难场景的真实 yield。
   - 建议将其视为软下界，或扩大 dry-run 规模。

5. **Behavior Cloning 存在 covariate shift 问题**：
   - 低 validation loss ≠ 部署成功。
   - 当前 v0.1 仅支持 offline BC，需 v0.2 引入 DAgger、KL-anchored PPO 等在线微调方法来缓解。

### **局限性**
- **v0.1.0 功能受限**：
  - 仅为 CPU 版本，无 GPU 加速 CBF kernel。
  - 仅支持 toy 2D 环境，尚未集成 Isaac Lab 或真实飞行平台。
  - 缺少实质性 RL 算法（如 KL-anchored PPO），计划在 v0.2.0 实现。
- **审计机制本身也可能出错**：
  - 实验中曾出现 watchdog 因配置解析 bug 未能触发过拟合 halt（已在 v0.1.1 修复）。
  - 说明审计系统自身也需被审计。

### **未来工作方向**
| 版本 | 计划内容 |
|------|--------|
| **v0.2.0** | 
| - | Isaac Lab 集成，支持 3D 张量并行仿真 |
| - | GPU 版本 PyTorch CBF kernel |
| - | 实现 KL-anchored PPO + critic warmup |
| - | Mamba-based policy backbone |
| - | 真实无人机飞行案例研究 |
| **v0.3+** |
| - | Chance-Constrained CBF 扩展 |
| - | 双通道感知架构：分离安全关键状态估计与语义推理 |

---

> 💡 **一句话总结**：  
> **ParallelCBF 不只是一个安全 RL 框架，更是一种“可审计科研工程实践”的新范式——它把“怎么做实验”从个人习惯上升为可编码、可测试、可传承的软件契约。**

</details>

---

### 12. [Solvita: Enhancing Large Language Models for Competitive Programming via Agentic Evolution](https://arxiv.org/abs/2605.15301)

**Authors**: Han Li, Jinyu Tian, Rili Feng, Yuqiao Du, Chong Zheng, Chenyu Wang, Chenchen Liu, Shihao Li, Xinping Lei, Yifan Yao, Weihao Xie, Letian Zhu, Jiaheng Liu  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.15301v1  

#### Abstract
Large language models (LLMs) still struggle with the rigorous reasoning demands of hard competitive programming. While recent multi-agent frameworks attempt to bridge this reliability gap, they remain fundamentally stateless: they rely on static retrieval and discard the valuable problem-solving and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Solvita: Enhancing Large Language Models for Competitive Programming via Agentic Evolution

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在**competitive programming**（竞赛编程）任务中仍面临严峻挑战，尤其是在处理复杂算法推理、边界条件验证和调试方面表现不稳定。尽管已有**multi-agent frameworks**（如 AlphaCodium、MapCoder）尝试通过多步流程提升可靠性，但这些方法本质上是**stateless**（无状态）的：每次解题都从零开始，无法积累和复用过往的解题与调试经验。

Solvita 正是为了解决这一“经验遗忘”问题而提出。

---

### 提出的新方法与创新思路
Solvita 是一个**agentic evolution**（智能体演化）框架，其核心思想是：**不更新 LLM 参数的前提下，实现持续学习与经验积累**。它通过以下四个关键设计实现这一目标：

#### （1）闭环多智能体协作系统（Closed-Loop Multi-Agent Framework）
Solvita 构建了一个由四个专业化智能体组成的闭环系统：
- **Planner**：将自然语言问题转化为形式化数学描述，并预测算法范式（如 `dp`, `graph`, `greedy`）。
- **Solver**：基于策略生成代码，并通过 **patch-based repair**（搜索替换式修复）迭代优化，而非全量重生成。
- **Oracle**：构建**认证的内部测试套件**（certified internal test suite），确保测试的有效性和正确性。
- **Hacker**：进行**针对性对抗攻击**（adversarial hacking），主动寻找能暴露 Bug 的输入。

这四个智能体形成一个反馈闭环，任何失败信号都会传播至整个系统，驱动各智能体协同修正。

#### （2）可训练的知识网络作为宏观记忆（Trainable Knowledge Networks）
每个智能体背后都配有一个**图结构的知识网络**（graph-structured knowledge network），用于存储和动态路由过往经验：
- 网络节点存储问题、元认知分析、可复用技能（skills）等。
- 边权重通过强化学习（REINFORCE）动态调整，依据任务成败信号进行更新。
- 这使得“记忆”不再是静态检索，而是**可学习的路径选择机制**。

#### （3）以智能体反馈作为训练信号（Agentic Feedback as Training Signal）
- 不修改 LLM 权重，而是利用 **Oracle 的认证质量** 和 **Hacker 的漏洞发现** 作为**强化学习奖励信号**。
- 系统能力随使用过程单调提升，实现了“冻结 LLM，进化推理策略”的范式。

#### （4）显著的性能提升
在多个基准上，Solvita 显著超越现有方法，尤其在 hardest 问题上表现突出。

---

### 相比现有方法的优势
| 对比维度 | 现有方法（如 AlphaCodium） | Solvita |
|--------|--------------------------|-------|
| **状态性** | Stateless（每次独立） | Stateful（持续学习） |
| **记忆机制** | 静态 RAG 或无记忆 | 动态图结构知识网络 |
| **修复方式** | 全量重生成 | Patch-based repair（更高效） |
| **验证机制** | 被动测试 | 主动认证 + 对抗攻击 |
| **学习信号** | 无或仅提示工程 | 强化学习驱动的经验演化 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CodeContests (CC)**：165 道题目，来自 DeepMind 的测试集。
- **APPS**：1,000 道题目，覆盖入门、面试、竞赛三个难度层级。
- **AetherCode (AC)**：400 道新颖且经过 Oracle 验证的题目，更具挑战性。
- **Codeforces 实时比赛**：12 场近期的 Div. 2 和 Div. 1+2 比赛，模拟真实人类参赛环境。

---

### 实验设置与评估指标

#### 主要评估指标
- **pass@1**：单次提交即通过所有测试用例的比例，为主要指标。
- **token consumption**：总 token 开销，用于衡量成本效率。
- **residual error analysis**：对剩余错误进行分类（如 Wrong Answer, TLE, MLE 等）。

#### 基线方法对比
- **Single-pass**：标准单次生成。
- **Codex CLI / Claude Code**：商业编码代理。
- **AlphaCodium / MapCoder / AgentCoder**：开源多智能体框架。

所有方法均使用相同的 LLM backbone（如 GPT-5.4、Claude Opus 4.6 等），确保公平比较。

#### 知识网络训练
- 冷启动语料库来自 Codeforces、AtCoder、LeetCode 等平台，经四阶段过滤后保留 **8,017** 道高质量题目。
- 知识网络在训练过程中逐步更新，支持在线学习。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（pass@1 %）

| 方法 \ 数据集 | CodeContests | APPS | AetherCode |
|-------------|------------|------|---------|
| Single-pass | 40.00 | 37.90 | 18.00 |
| Codex CLI | 81.82 | 67.10 | 70.30 |
| Claude Code | 70.91 | 69.00 | 58.79 |
| AlphaCodium | 60.61 | 56.20 | 53.33 |
| MapCoder | 57.58 | 55.10 | 50.91 |
| **Solvita (ours)** | **82.42** | **69.30** | **69.70** |

> ✅ Solvita 在 **14/15** 个 backbone-benchmark 组合中取得最佳成绩，**pass@1 几乎翻倍于单次生成基线**。

---

### 与基线方法的对比结果
- 在 **GPT-5.4 + CodeContests** 上，Solvita 将 pass@1 从 **40.0% 提升至 82.4%**。
- 相比最强开源基线（AlphaCodium），平均提升 **~20–25%**。
- 在 **AetherCode** 这类高难度数据集上优势更为明显。
- 成本方面，token 消耗与开源多智能体框架相当，远低于商业 CLI 工具。

---

### 消融实验结果（Ablation Study）

| 配置 | CodeContests | APPS | AetherCode |
|-----|------------|------|---------|
| Single-pass | 40.00 | 37.90 | 18.00 |
| without training（无知识网络） | 67.70 | 54.50 | 35.00 |
| + Solver network @4.5k | 75.60 | 61.80 | 43.50 |
| + Hacker network @4.5k | 72.00 | 58.00 | 38.50 |
| + Oracle network @4.5k | 74.10 | 60.50 | 42.70 |
| **Full system** | **82.42** | **69.30** | **55.10** |

> 🔍 发现：
> - 多智能体架构本身已带来巨大提升（67.70 vs 40.00）。
> - **Solver 知识网络贡献最大**，其次是 Oracle 和 Hacker。
> - 所有组件具有**累加效应**，非替代关系。

---

### Patch-Based Repair vs. Full Regeneration

| Backbone | 策略 | Solve (%) | 平均迭代数 | Token 节省率 |
|--------|------|----------|-----------|------------|
| GPT-5.4 | Full regeneration | 75.76 | 5.18 | 67.4% |
| GPT-5.4 | **Patch repair** | **82.42** | **3.74** | **91.2%** |

> ✅ Patch 修复不仅准确率更高，且迭代次数更少、token 消耗更低，避免了全量重写导致的回归问题。

---

## 4. 关键结论和发现

### 主要发现
1. **经验积累至关重要**：即使 LLM 参数冻结，通过结构化的知识网络和强化学习信号，也能实现持续的能力进化。
2. **闭环反馈优于开环流程**：Oracle 的认证监督 + Hacker 的对抗攻击形成了强大的自我纠错机制。
3. **patch-based repair 更高效稳定**：相比全量重生成，局部修补能更好保持已有正确性。
4. **多智能体协同具有累加增益**：各模块相互促进，最终效果远超单一组件叠加。

---

### 方法的局限性
1. **冷启动成本高**：需要约 **5,000 道训练题**才能摊平每题成本，初期投入较大。
2. **Hacker 攻击范围有限**：
   - 数学密集型 Bug（如数论不变量、几何精度）难以被当前策略覆盖。
   - 受限于 LLM 的推理视野（reasoning horizon）。
3. **Patch Repair Drift**：
   - 当问题本质错误时，Solver 可能误判为局部 Bug，导致多次无效修补。
   - 当前依赖事后回归检测，缺乏前置判断机制。

---

### 未来工作方向
1. **知识网络暖启动**：利用公开的 editorial、accepted submissions、debugging traces 初始化知识网络，缩短冷启动周期。
2. **跨领域迁移**：
   - 形式化定理证明（Oracle → Proof Checker, Hacker → Countermodel Search）
   - 数学奥赛题（符号验证替代测试用例）
   - 科学计算（可执行模拟器作为验证器）
3. **探索模型微调与角色对齐信用分配的结合**：
   - 如何将 Hacker 产生的对抗信号用于 fine-tuning，同时保持角色分工的清晰性，是一个开放问题。

---

> 📌 **总结**：Solvita 提出了一种全新的“**冻结模型，进化策略**”范式，通过多智能体闭环 + 图结构知识网络 + 强化学习信号，在不修改 LLM 参数的情况下实现了持续学习，显著提升了 LLM 在竞赛编程中的可靠性和准确性，为构建**可进化的 AI 编程助手**提供了重要路径。

</details>

---

### 13. [Fully Open Meditron: An Auditable Pipeline for Clinical LLMs](https://arxiv.org/abs/2605.16215)

**Authors**: Xavier Theimer-Lienhard, Mushtaha El-Amin, Fay Elhassan, Sahaj Vaidya, Victor Cartier-Negadi, David Sasu, Lars Klein, Mary-Anne Hartley  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.16215v1  

#### Abstract
Clinical decision support systems (CDSS) require scrutable, auditable pipelines that enable rigorous, reproducible validation. Yet current LLM-based CDSS remain largely opaque. Most "open" models are open-weight only, releasing parameters while withholding the data provenance, curation procedures, a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fully Open Meditron: An Auditable Pipeline for Clinical LLMs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于大语言模型的临床决策支持系统（LLM-CDSS）普遍存在“黑箱”问题：尽管许多模型声称“开放”，但通常仅发布模型权重（open-weight），而未公开训练数据来源、数据清洗流程、合成数据生成机制以及完整的训练框架。这种不透明性导致模型行为难以审计，限制了其在高风险医疗场景中的可信度与可复现性。

此外，现有医学基准测试（如MCQA）存在饱和现象，且过度依赖多项选择题（Multiple-Choice Questions, MCQ），无法有效衡量真实临床交互中所需的推理能力、情境感知和沟通技巧。

### 提出的新方法与创新
本文提出了 **Fully Open Meditron**，首个端到端完全开放（Fully Open, FO）的临床大语言模型构建管道，涵盖从数据构建、训练到评估的全过程。其核心贡献包括：

#### （1）**全开放医疗适配框架（Fully Open Medical Adaptation Framework）**
- 发布完整的训练栈：包括**训练数据、合成数据生成流程、去污染代码、训练脚本与超参数配置**。
- 所有组件均开源，确保第三方可独立验证、复现并改进整个流程。
- 支持在无专有数据的前提下实现高性能医学专业化。

#### （2）**结构化、医生审核的知识语料库（Clinician-Audited Knowledge Corpus）**
- 整合8个公共医学QA数据集，并通过**三类由医生审核的合成扩展**增强覆盖范围：
  - **考试风格QA**（Synthetic Curated QA）
  - **指南驱动QA**（Guidelines QA）：基于来自16个全球机构的46,469条临床实践指南生成
  - **临床案例片段**（Synthetic MOOVE）：模拟复杂诊断推理
- 引入系统级**去污染机制**（decontamination），防止训练数据与评估基准重叠。
- 使用**黄金标签拒绝采样**（gold-label rejection sampling）降低合成数据中的幻觉风险。

#### （3）**自动化开放式临床评估协议 Auto-MOOVE**
- 提出一种基于 **LLM-as-a-judge** 的评估框架，用于衡量多维度临床推理能力。
- 经过对204名人审评者的数据校准，验证该判断器与人类评价具有统计一致性。
- 评估维度包括：问题理解、逻辑推理、相关性与完整性、安全性（harmlessness）、公平性、情境意识、沟通能力、清晰度及与指南的一致性。

#### （4）**一系列完全开放的医学专家模型家族**
- 应用于五个FO基础模型（Apertus、OLMo、EuroLLM等），首次实现了**完全开放的医学专用LLM族**。
- 其中 **Apertus-70B-MeditronFO** 成为新的FO SOTA（State-of-the-Art）。

### 相比现有方法的优势
| 维度 | 现有方法（如MedGemma） | Fully Open Meditron |
|------|------------------------|--------------------|
| 开放程度 | 仅开放权重（Open-Weights） | 完全开放（Fully Open）：数据、代码、流程全部可审计 |
| 数据透明性 | 不披露训练数据与合成流程 | 明确披露所有数据源与生成prompt |
| 评估方式 | 多项选择题为主（MCQA） | 引入开放式、多维评估 Auto-MOOVE |
| 医疗覆盖广度 | 偏向北美/欧洲常见病 | 加强低资源环境、儿科、急诊等代表性不足领域 |
| 可信度保障 | 缺乏系统去污染与医生审核 | 全流程去污染 + 四位医生小组审核 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### （1）**训练数据（Fully Open Meditron Corpus）**
- **原始QA数据集（Curated QA）**：
  - MedQA, MedMCQA, PubMedQA, MedExpQA, HealthSearchQA, LiveQA, AfriMed-QA v1/v2
  - 总计约217k个样本
- **合成扩展数据**（合计约385k样本）：
  - **Synthetic Curated QA**：基于上述数据生成新考试题型
  - **Guidelines QA**：基于46,469条临床指南生成证据导向问答
  - **Synthetic MOOVE**：基于MOOVE训练集生成开放式临床案例提示

> 合成数据占总token数的~71%，由 **GPT-OSS-120B** 生成，并经过严格过滤。

#### （2）**评估数据集**

| 类型 | 数据集 | 描述 |
|------|-------|------|
| **封闭式QA** | MedQA, MedMCQA, PubMedQA | 标准多项选择题测试 |
| **外部泛化检测** | MedXpertQA | 未见过的专家级临床推理任务 |
| **通用能力烟雾测试** | MMLU-Pro, IFEval, ARC-Challenge | 防止灾难性遗忘 |
| **开放式临床评估** | Auto-MOOVE, HealthBench | 多维度临床推理与沟通质量评估 |

---

### 实验设置

#### 模型选择
- **目标模型**：在以下五种FO base models上进行监督微调（SFT）：
  - Apertus-70B/8B-Instruct
  - OLMo-2-32B-SFT
  - EuroLLM-22B/9B-Instruct
  - Gemma-3-27B-it（作为open-weight对照）
- **对比模型**：
  - MedGemma-27B（最强开源医学模型）
  - Llama-3.1-70B-Meditron
  - GPT-OSS-120B（最大开源模型之一）

#### 训练细节
- 使用 **Axolotl** 框架，采用 FSDP 或 ZeRO-3 进行分布式训练
- 序列长度：4096 tokens
- 学习率、batch size 等按各base模型最佳实践调整（详见Appendix I）

#### 评估指标

| 评估类型 | 指标 |
|---------|------|
| 封闭式QA | Accuracy (%) on MedQA/MedMCQA/PubMedQA/MedXpertQA |
| 综合医学得分 | Unweighted average across MCQA benchmarks |
| 开放式评估 | 
| - Auto-MOOVE | 胜率（Win Rate）、调整胜率（Adjusted Win Rate）、ΔLikert评分 |
| - HealthBench | LLM-as-a-judge 得分（Qwen3-235B-A22B 判断） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）**在标准MCQA上的表现（Table 2）**

| 模型 | Med Avg (%) | 增益（vs base） |
|------|------------|---------------|
| Apertus-70B-Instruct | 47.18 | — |
| → **Apertus-70B-MeditronFO** | **53.77** | **+6.59** |
| Gemma-3-27b-it | 57.55 | — |
| → **Gemma-3-27b-MeditronFO** | **58.63** | **+1.08** |
| MedGemma-27B（SOTA open-weight） | 60.67 | — |

> ✅ **Apertus-70B-MeditronFO 是目前性能最高的 Fully Open 医学模型**，显著超越其基础版本。

#### （2）**在HealthBench上的表现**
| 模型 | HealthBench Score |
|------|------------------|
| Apertus-70B-MeditronFO | 51.86 → **提升至51.86**（原为43.72） |
| Gemma-3-27B-MeditronFO | **58.02** |
| MedGemma-27B | 55.92 |
> ✅ 在独立rubric-based评估中也优于MedGemma（+2.1 pts）

#### （3）**Auto-MOOVE 成对偏好比较（Figure 4）**

| 对比组 | 我方模型胜率（Adjusted Win Rate） |
|--------|-------------------------------|
| Apertus-70B-MeditronFO vs Base | **79.6%** |
| OLMo-2-32B-MeditronFO vs Base | **83.7%** |
| Apertus-8B-MeditronFO vs Base | **87.8%** |
| **Gemma-3-27B-MeditronFO vs MedGemma-27B** | **66.3%** |

> ✅ 所有 *-MeditronFO 模型均被偏好于其对应base；
> ✅ 即使面对更强的MedGemma，Gemma-MeditronFO仍以明显优势获胜。

---

### 消融实验结果（Ablation Studies）

#### （1）**语料成分消融（Table 3）**

| 移除组件 | Med Avg ↓ | Auto-MOOVE ↓ | ΔLikert ↓ |
|----------|-----------|--------------|-----------|
| 完整语料 | 53.77 | 79.6 | 0.40 |
| - Curated QA | 49.74 (**↓4.03**) | 73.4 | 0.27 |
| - Synthetic MOOVE | 53.69 | 75.5 | 0.34 |
| - Guidelines QA | 54.34 | 78.7 | 0.39 |

> 🔍 发现：
> - 移除 **Curated QA** 导致最大性能下降，说明真实考试数据对开放式推理至关重要。
> - 移除 **Synthetic MOOVE** 显著影响 Auto-MOOVE 表现，表明其对开放推理分布建模有效。
> - 移除 **Guidelines QA** 反而略微提升MCQA分数，但不影响开放评估 → 表明指南数据更利于实际临床而非应试。

#### （2）**不同Judge模型下的稳定性（Table 12）**
- 使用8种不同LLM作为judge（Qwen, Gemma, Llama, Nemotron等）
- **Apertus-70B-MeditronFO 始终优于 base**，调整胜率介于73.2%–93.7%
- 所有ΔLikert均为正 → 结果稳健，非因风格匹配偏差所致

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **完全开放的医学LLM可以达到甚至超越部分专有/半开放系统的性能水平**：
   - Apertus-70B-MeditronFO 在多个指标上接近MedGemma，且在某些开放评估中反超。
   - Gemma-3-27B-MeditronFO 在HealthBench和Auto-MOOVE中均优于MedGemma。

2. ✅ **高质量、医生审核的合成数据是弥补真实数据不足的关键**：
   - 合成数据占比高达~64%，但经严格控制后能有效提升模型能力。
   - 特别是在急诊、危重症、低资源背景下的覆盖显著增强（如紧急护理覆盖率从15%→38.7%）。

3. ✅ **传统MCQA不足以反映真实临床能力**：
   - 多项选择题可能鼓励记忆与格式匹配，而忽略沟通、情境理解和安全对齐。
   - Auto-MOOVE揭示了MeditronFO在逻辑推理、上下文感知等方面全面领先。

4. ✅ **端到端可审计性不牺牲性能**：
   - Fully Open 并非性能妥协的代名词；通过严谨的数据工程与流程设计，可在保持透明的同时实现SOTA。

---

### 局限性

1. **Auto-MOOVE judge在安全性维度上不如人类敏感**：
   - 对harmlessness、fairness等关键维度判别力较弱，不能单独用于部署前的安全评估。

2. **去污染为语法层面，非语义层面**：
   - 无法完全排除教师模型通过改写或泛化引入评估内容的风险。

3. **合成数据依赖单一教师模型（GPT-OSS-120B）**：
   - 可能引入风格或推理偏见，虽经消融验证但仍需警惕。

4. **医生审核覆盖有限**：
   - 每个prompt模板仅审查3个样本，可能存在未被发现的系统性错误。

5. **指令跟随能力有所下降**：
   - 部分模型在IFEval等通用任务上退化，建议结合Tulu replay缓解。

---

### 未来工作方向

1. **引入偏好优化（Preference Optimization）** 替代纯SFT，进一步提升对齐性。
2. **开展持续预训练（Continued Pretraining）** 在GUIDELINES语料上，增强先验医学知识。
3. **开发端到端可追溯的教师模型链**，避免依赖闭源或未审计的生成器。
4. **扩展多语言与低资源地区数据覆盖**，推动全球公平医疗AI发展。
5. **建立动态更新机制**，定期纳入最新临床指南与真实世界病例。

---

> 📌 **一句话总结**：  
> **Fully Open Meditron 证明，在不依赖专有数据的前提下，通过医生参与、系统去污染、高质量合成数据与开放式评估，完全可以构建出既高性能又可审计的临床大模型，为可信医疗AI树立了新范式。**

</details>

---

### 14. [ALSO: Adversarial Online Strategy Optimization for Social Agents](https://arxiv.org/abs/2605.15768)

**Authors**: Xiang Li, Liping Yi, Mingze Kong, Min Zhang, Zhongxiang Dai, QingHua Hu  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.15768v1  

#### Abstract
Social simulation provides a compelling testbed for studying social intelligence, where agents interact through multi-turn dialogues under evolving contexts and strategically adapting opponents. Such environments are inherently non-stationary, requiring agents to dynamically adjust their strategies ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ALSO: Adversarial Online Strategy Optimization for Social Agents

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在基于 **Large Language Model (LLM)** 的多智能体社会模拟（social simulation）环境中，传统的 **静态 persona**（角色设定）方法无法适应动态、非平稳（non-stationary）的社会交互环境。现有方法如离线强化学习（offline RL）或外部规划器（external planners）通常假设环境稳定，且训练开销大，难以应对对手策略持续演化带来的挑战。

因此，该论文旨在解决以下核心问题：
- 如何在**非平稳、对抗性**的多轮对话环境中实现**在线策略优化**？
- 如何在**稀疏反馈**下高效探索并泛化策略效果？

---

### 🚀 提出的新方法：ALSO

作者提出了 **ALSO (Adversarial Online Strategy Optimization)**，这是首个面向 LLM 社会智能体的**在线策略优化框架**，其核心创新如下：

#### （1）将策略选择建模为 **Adversarial Multi-Armed Bandit** 问题
- 将“静态 persona + 动态策略指令”组合视为一个“arm”（选项），通过 bandit 算法进行在线选择。
- 不依赖环境稳定性假设，适用于**共演进（co-evolving）** 的对手行为场景。
- 采用类似 **EXP3** 的算法设计，引入随机化选择以增强对策略漂移的鲁棒性。

#### （2）引入轻量级神经代理模型（Neural Surrogate Model）
- 利用历史交互记录预测未尝试策略的潜在奖励，从而**泛化稀疏反馈**。
- 通过语义相似性实现跨策略的知识迁移，提升样本效率。
- 仅更新 surrogate 模型，保持主 LLM 冻结，避免昂贵的微调。

#### （3）闭环在线学习系统
- 构成完整的“观察 → 选策略 → 交互 → 反馈 → 更新”循环，支持持续自适应。

---

### 🔍 相比现有方法的优势

| 维度 | 现有方法（如 OPRO, EvoPrompt, Sotopia-Ω） | ALSO |
|------|----------------------------------------|------|
| 学习范式 | 多为离线优化或固定验证集上的评估 | 完全在线，实时响应动态环境 |
| 非平稳性处理 | 假设 reward 分布稳定 | 显式建模对抗性、非平稳 reward 漂移 |
| 样本效率 | 独立评估每个 prompt，样本利用率低 | 通过 surrogate 泛化 reward，支持跨策略迁移 |
| 训练成本 | 需要额外 LLM 调用或大量预收集数据 | 无需 fine-tuning，仅轻量 surrogate 在线更新 |
| 实时适应能力 | 固定策略池或需重新训练 | 支持动态策略注入与发现 |

---

## 2. 核心实验方法和设置

### 📚 数据集

- **Sotopia**：一个综合性的 LLM 社会智能评测基准，涵盖 90 个双人社会场景，涉及谈判、合作、竞争等七维社交能力。
- **Sotopia-Hard**：Sotopia 的子集，包含 14 个更具挑战性的复杂冲突场景，用于测试极端情况下的表现。

> 所有方法均在同一扩展版 Sotopia 上运行，支持动态策略注入。

---

### ⚙️ 实验设置

| 设置项 | 描述 |
|-------|------|
| 对话长度 | 最多 20 轮对话（turns） |
| 策略空间 | 共 12 种策略指令，覆盖合作、竞争、理性说服、互惠等六大类（见 Table 5） |
| Agent 模型 | DeepSeek-V3.2 或 Qwen-2.5-72B-Instruct |
| Reward 评估器 | 使用 LLM-based evaluator 提供每轮 shaping reward（用于 online 更新） |
| 最终评估 | 使用 GPT-4o 进行 episode-level 统一打分，确保公平比较 |
| 优化模式 | 双边优化（Bilateral）：双方独立运行 ALSO；也对比单边优化 |

---

### 📊 评估指标（来自 Sotopia）

| 指标 | 含义 | 范围 |
|------|------|------|
| **Goal** | 智能体达成私有目标的程度 | [0,10] |
| **Relationship (Rel.)** | 人际关系质量变化 | [-5,5] |
| **Knowledge (Know.)** | 获取新信息的数量 | [0,10] |
| **Overall** | 综合得分（归一化后平均） | [0,1] |

> 所有维度最终被归一化至 [0,1] 并取平均作为 Overall Score。

---

### 🆚 基线方法对比

| 方法 | 类型 | 是否在线 | 是否需要额外 LLM 调用 |
|------|------|----------|------------------------|
| **Vanilla** | 无策略增强 | ❌ | ❌ |
| **OPRO** | 在线 prompt 优化（meta-prompt） | ✅ | ✅ |
| **EvoPrompt** | 基于遗传算法的 prompt 进化 | ✅ | ✅ |
| **INSTINCT** | Neural Bandit + UCB 探索 | ✅ | ✅ |
| **ALSO (Ours)** | Adversarial Bandit + Surrogate | ✅ | ❌ |

> 所有方法共享相同策略池和预算限制，保证可比性。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Sotopia-Hard）

| 方法 | Goal | Rel. | Know. | **Overall** |
|------|------|------|-------|------------|
| Vanilla | 6.52 | 1.32 | 4.37 | 3.02 |
| OPRO | 6.71 | 1.89 | 4.63 | 3.24 |
| EvoPrompt | 6.77 | 1.93 | 5.15 | 3.29 |
| INSTINCT | 6.92 | 2.16 | 5.44 | 3.43 |
| **ALSO (Ours)** | **7.11** | **2.43** | **5.47** | **3.53** |

#### ✅ 性能提升
- **Overall +16.60%** 相比 Vanilla
- **Overall +2.92%** 相比最强基线（INSTINCT）
- **Relationship 维度提升高达 +83.79%**（1.32 → 2.43），表明关系修复能力显著增强

---

### 🔍 消融实验结果（Ablation Study）

| 变体 | Goal | Rel. | Know. | Overall |
|------|------|------|-------|---------|
| **Full ALSO** | 7.93 | 3.07 | 6.46 | **3.91** |
| w/o EXP3 (e-greedy) | 7.50 | 2.71 | 5.32 | 3.61 |
| w/o Score Smoothing | 7.57 | 2.25 | 5.39 | 3.57 |
| w/o Context Embedding | 7.43 | 2.64 | 4.82 | 3.51 |
| **w/o Neural Surrogate** | **6.89** | **2.00** | **4.93** | **3.33** |

> 结论：
- **Neural Surrogate 是最关键组件**，移除后 Overall 下降 0.58（相对下降 14.8%）
- **Score Smoothing** 对 Relationship 影响最大（↓0.82），说明其对追踪动态变化至关重要
- EXP3 式探索优于 e-greedy，验证了对抗环境下随机化的必要性

---

### 🔄 其他重要实验发现

#### （1）跨场景泛化能力（Zero-Shot Transfer）
- 在未见过的 7 个测试场景上：
  - Goal 得分：**7.14 vs. 6.79**（+5.3%）
  - Overall 得分：**3.60 vs. 3.17**（+13.5%）
- 表明 ALSO 学到的是**可迁移的社会交互模式**，而非过拟合特定场景。

#### （2）异构模型配对有效性
- 在 DeepSeek / Qwen / GPT-4o-mini 的混合配对中，ALSO 均带来一致增益（+3% ~ +28%）
- 证明其优化机制具有**通用性和鲁棒性**，不依赖特定 backbone。

#### （3）双边优化优于单边
- 当双方都使用 ALSO 时，Overall 显著更高（p < 0.01）
- 特别是在 **Relationship 和 Knowledge** 维度提升明显，说明协同进化更符合真实社交动态。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **静态 persona 不足以支撑复杂社会互动**，必须引入**动态策略选择机制**。
2. **非平稳性是社会模拟的核心挑战**，传统 bandit 假设失效，需采用 adversarial 设计。
3. **轻量 surrogate 模型能有效泛化稀疏反馈**，极大提升在线学习效率。
4. **ALSO 在最难场景中表现最突出**，尤其在修复破裂关系方面远超基线。
5. **方法具备良好泛化性与兼容性**，可在不同模型、未知场景中稳定生效。

---

### ⚠️ 局限性

1. **策略空间仍为预定义集合**：虽然支持动态扩展，但当前实验基于人工设计的 12 条策略。
2. **依赖高质量 turn-level evaluator**：reward shaping 效果受限于中间评估器的质量。
3. **未探索完全自主的策略生成**：目前策略由 LLM 生成后固定，尚未实现端到端的策略创造。
4. **计算资源集中在 surrogate 训练**：尽管轻量，但在长对话中仍有一定延迟。

---

### 🔮 未来工作方向

1. **结合 LLM 自主生成新策略**：利用 ALSO 的 reward 信号驱动策略创新（prompt generation）。
2. **构建分层策略空间**：将 high-level strategy 与 low-level action primitives 解耦。
3. **应用于真实人类-LLM 交互场景**：如客服、心理咨询、教育辅导等。
4. **引入因果推理机制**：识别策略与 outcome 之间的因果路径，提升解释性。
5. **多智能体协作中的群体策略协调**：扩展至三人及以上复杂社交网络。

---

## ✅ 总结

**ALSO** 是首个专为 **LLM 社会智能体**设计的**对抗式在线策略优化框架**，它通过：
- 将策略选择形式化为 **adversarial bandit**，
- 引入 **neural surrogate** 实现 reward 泛化，
- 构建闭环在线学习系统，

成功解决了社会模拟中**非平稳性、稀疏反馈、高探索成本**三大难题。实验证明，ALSO 在 Sotopia 上实现了 **+16.6% 的整体提升**，并在关系修复等关键维度取得突破性进展，为构建真正具备**社会适应力**的 AI 智能体提供了新范式。

</details>

---

### 15. [Scale: Deep Reinforcement Learning for Container Scheduling in Serverless Edge Computing](https://arxiv.org/abs/2605.15704)

**Authors**: Chen Chen, Zihan Jia, Andrea Sabbioni, Reza Farahani, Lei Jiao  
**Category**: cs.DC  
**Published**: 2026-05-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.15704v1  

#### Abstract
Serverless computing has emerged as a promising computing paradigm for edge computing. However, adopting the event driven model in highly dynamic, heterogeneous, and distributed edge systems poses significant challenges in request placement and resource management. Efficiently allocating requests to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Scale: Deep Reinforcement Learning for Container Scheduling in Serverless Edge Computing》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Serverless Edge Computing (SEC)** 场景下，如何高效地进行 **container scheduling** 是一个关键挑战。由于边缘节点具有 **地理分布广、资源异构性强、工作负载动态变化大** 的特点，传统的调度策略难以同时满足以下目标：
- 最小化 **end-to-end latency**
- 遵守应用定义的 **Service Level Objective (SLO)**
- 减少不必要的 **data movement** 和 **resource over-provisioning**

尤其在事件驱动模型中，冷启动（cold-start）、通信延迟和计算资源竞争等问题加剧了调度复杂性。

---

### 🚀 提出的新方法与创新点

论文提出了名为 **Scale** 的新型容器调度框架，其核心创新包括：

#### （1）SLO-aware 容器调度建模
首次将 **SLO约束、端到端延迟、数据局部性（data locality）** 联合纳入优化目标，构建了一个综合性的整数线性规划（ILP）模型，以最小化总延迟并保证 SLO 合规。

#### （2）基于 Actor-Critic 架构的 DRL 框架
设计了一种 **分层动作空间（hierarchical action space）** 的深度强化学习算法：
- 第一阶段决定请求应分配到哪个 **edge node**
- 第二阶段决定是否复用已有容器或创建新容器（container reuse）
- 动作之间存在逻辑依赖关系，通过内部约束建模确保可行性

#### （3）采用 PPO 算法提升稳定性
使用 **Proximal Policy Optimization (PPO)** 作为策略更新机制，利用 **clipped objective** 抑制训练过程中的剧烈波动，提高在大规模动态环境下的收敛性和鲁棒性。

#### （4）解耦请求调度与容器复用
不同于以往将两者耦合处理的方法，Scale 显式分离这两个决策步骤，在保持灵活性的同时增强了对系统状态的理解能力。

---

### 🔍 相比现有方法的优势

| 维度 | Scale 的优势 |
|------|-------------|
| **性能 vs 决策速度** | 接近最优 ILP 解的质量（仅差 1.11–1.15×），但决策时间减少 **99%** |
| **适应性** | 支持在线推理，适用于高动态、低延迟要求的边缘场景 |
| **SLO 遵从性** | 在合理范围内控制 SLO violation rate（约 4.9%），优于大多数 DRL 方法 |
| **可扩展性** | 基于 DRL 的离线训练 + 在线推断模式，适合大规模部署 |

---

## 2. 核心实验方法和设置

### 📊 数据集与仿真环境

| 类别 | 描述 |
|------|------|
| **Edge Network Topology** | 使用 EUA 数据集中的 **125 个边缘节点** 分布图（墨尔本CBD区域）<br>• 包含不同类型的节点（CPU: 2.4–3.6 GHz, Memory: 10–30 GB） |
| **Workload Traces** | 基于 **Huawei Cloud Dataset**（真实云函数调用日志，持续141天，共200个函数）生成模拟 workload<br>• 使用 **Zipf-β 分布** 模拟边缘请求到达模式 |
| **SLO 设置** | 所有请求的 SLO 设定在 **[200, 400] ms** 范围内，符合实际应用场景 |

---

### 🧪 实验设置与评估指标

#### ✅ 评估平台
- 工具：Stable-Baselines3（PyTorch 实现）
- 硬件：Intel i7-13700H, 32GB RAM
- 代码量：约 2,000 行 Python 代码

#### 🎯 主要评估指标
| 指标 | 说明 |
|------|------|
| **End-to-end Latency** | 包括 cold-start、computation 和 communication 延迟 |
| **P50 / P99 Latency** | 反映典型用户体验与尾部延迟表现 |
| **SLO Violation Rate (%)** | 违反 SLO 的请求数占比 |
| **Decision-making Time per Request** | 单次调度决策耗时（关键用于衡量实时性） |
| **CDF of Latency** | 展示延迟的整体分布情况 |

---

### ⚖️ 基线方法对比

| 基线方法 | 类型 | 说明 |
|--------|-----|------|
| **Midaco-solver** | ILP Solver（全局最优近似） | 使用进化混合算法求解原始 ILP 问题，分别运行 50k、100k、200k 次迭代，代表“准最优”基准 |
| **m-DQN** | Value-based DRL 方法 | 多步 DQN 变体，具备经验回放和目标网络机制，用于验证策略梯度方法的优越性 |

> 注：所有 DRL 方法均采用 **offline training + online inference** 模式。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 方法 | 平均延迟 (ms) | P50 延迟 (ms) | P99 延迟 (ms) | SLO Violation (%) | 决策时间 (s/request) |
|------|----------------|----------------|----------------|--------------------|------------------------|
| **Midaco-200k** | 191.87 | 188.42 | ~284.41 | 1.6 – 4.2 | 12.58 |
| **Scale** | **208.93** | **210.50** | ~302.29 | **4.9** | **~0.02** |
| **m-DQN** | 320.82 | 230.63 | ~341.29 | 4.6 | ~0.02 |

---

### 🔍 对比分析

#### （1）延迟性能接近 ILP 最优解
- Scale 的平均延迟仅为 Midaco-200k 的 **1.09×**，最大延迟为 **1.15×**
- P50 性能差距更小，仅高出约 **1.11×**，表明其在主流请求上调度效果极佳

#### （2）决策效率极大提升
- Scale 的单请求决策时间为 **0.02 秒**，而 Midaco 需要 **2.98–12.58 秒**
- **提速达 99%**，使其完全适用于 **online scheduling** 和 **real-time response** 场景

#### （3）SLO 控制合理
- 尽管 SLO violation rate 略高于 Midaco（4.9% vs 最低 1.6%），但仍远低于 m-DQN 的实际影响（因后者延迟更高）
- 表明 Scale 成功在 **性能、延迟、合规性** 之间取得良好平衡

#### （4）相比 m-DQN 显著领先
- 平均延迟降低 **~31%**
- 尾部延迟（P99）改善明显（302 vs 341 ms）
- 更稳定的表现归功于 PPO 的 **clipped objective** 机制，避免了 DQN 中常见的 overestimation bias 和训练震荡

> ❗未报告消融实验（ablation study），无法量化各模块（如分层动作空间、PPO、state representation）的具体贡献。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **DRL 可有效逼近 ILP 最优解**  
   在复杂的 SEC 场景中，基于 DRL 的方法（尤其是 actor-critic + PPO 架构）能够在极短时间内输出高质量调度决策，性能接近传统优化器的 **1.15×以内**。

2. **分层决策结构提升调度合理性**  
   将 **request placement** 与 **container reuse** 解耦，并引入动作间逻辑约束，显著提高了策略的可行性和可解释性。

3. **PPO 比 value-based DRL 更适合动态环境**  
   相较于 m-DQN，Scale 因采用 PPO 而表现出更强的稳定性与泛化能力，尤其在应对突发流量和资源波动时更具优势。

4. **低决策延迟是边缘调度的关键**  
   Midaco 尽管精度高，但耗时过长（>10秒），无法满足边缘服务的实时性需求；而 Scale 实现了 **“近似最优 + 实时响应”** 的理想组合。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|-------|------|
| **缺乏真实边缘部署验证** | 所有实验基于仿真环境，尚未在真实边缘集群中测试 |
| **未考虑能源消耗** | 当前目标仅聚焦于延迟与 SLO，未涉及 energy efficiency 或碳排放等绿色计算维度 |
| **无消融实验支持设计选择** | 无法验证分层动作空间、PPO、状态编码等组件各自的增益 |
| **假设返回路径延迟可忽略** | 忽略了结果回传的通信开销，可能在某些应用中不成立 |

---

### 🔮 未来工作方向

1. **联合优化延迟与能耗**  
   扩展目标函数，实现 **end-to-end latency 与 energy efficiency 的多目标权衡**。

2. **跨域协同调度**  
   将 Scale 扩展至 **cloud-edge continuum** 架构，支持跨层级资源协调。

3. **支持函数链（function chaining）与 workflow 级调度**  
   当前仅针对单个函数请求，下一步可集成 workflow-aware 调度能力。

4. **在线自适应学习机制**  
   引入 **online fine-tuning** 或 **meta-learning**，使模型能随环境变化持续进化。

---

> **总结一句话**：  
> **Scale 成功实现了在 serverless edge computing 中“接近最优调度质量 + 极低决策延迟”的突破，为构建高性能、低延迟、SLO-aware 的边缘函数服务平台提供了可行的技术路径。**

</details>

---

### 16. [An efficient multi-GPU implementation for the Discontinuous Galerkin ocean model SLIM](https://arxiv.org/abs/2605.16082)

**Authors**: Miguel De Le Court, Vincent Legat, Ange P. Ishimwe, Colin Scherpereel, Emmanuel Hanert, Jonathan Lambrechts  
**Category**: cs.DC  
**Published**: 2026-05-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.16082v1  

#### Abstract
Unstructured-mesh ocean models are increasingly used for coastal applications due to their ability to represent complex geometries and apply local grid refinement where needed. However, their broader use has been hindered by their high computational cost, particularly for models based on the Discont...

---

### 17. [Federated Learning of Spiking Neural Networks under Heterogeneous Temporal Resolutions](https://arxiv.org/abs/2605.15355)

**Authors**: Sanja Karilanova, Subhrakanti Dey, Ay\c{c}a \"Oz\c{c}elikkale  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.15355v1  

#### Abstract
Spiking neural networks (SNNs) are biologically inspired energy-efficient models that use sparse binary spike-based communication between neurons, making them attractive for resource-constrained edge devices. Federated learning enables such devices to train collaboratively without sharing raw data. ...

---

### 18. [LPDS: Evaluating LLM Robustness Through Logic-Preserving Difficulty Scaling](https://arxiv.org/abs/2605.15393)

**Authors**: Philipp Mondorf, Samuel J. Bell, Jesse Dodge, Dieuwke Hupkes  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.15393v1  

#### Abstract
As large language models (LLMs) are increasingly deployed to perform tasks with minimal human oversight, it is crucial that these models operate robustly. In particular, a model that can solve a given problem should not fail simply because certain entities$\unicode{x2013}$such as names, numbers, or ...

---

### 19. [OgBench: A Framework for Evaluating Graph Neural Networks on Omics Data](https://arxiv.org/abs/2605.15511)

**Authors**: Louisa Cornelis, Johan Mathe, Louis Van Langendonck, Guillermo Bern\'ardez, Nina Miolane  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.15511v1  

#### Abstract
Graph Neural Networks (GNNs) have become the dominant framework for inductive graph-level learning. Yet most benchmarks focus on the regime $n \gg p$, where the number of graphs $n$ greatly exceeds the number of nodes per graph $p$. This overlooks biological domains such as omics, which operate in t...

---

### 20. [IO-SVD: Input-Output Whitened SVD for Adaptive-Rank LLM Compression](https://arxiv.org/abs/2605.15626)

**Authors**: Ali Abbasi, Chayne Thrash, Haoran Qin, Hamed Pirsiavash, Soheil Kolouri  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.15626v1  

#### Abstract
Large language models deliver strong performance across language and reasoning tasks, but their storage and compute costs remain major barriers to deployment in resource-constrained and latency-sensitive settings. SVD-based post-training compression offers a hardware-agnostic way to reduce model siz...

---

### 21. [Variational Autoregressive Networks with probability priors](https://arxiv.org/abs/2605.16020)

**Authors**: Piotr Bia{\l}as, Piotr Korcyl, Tomasz Stebel, Dawid Zapolski  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.16020v1  

#### Abstract
Monte Carlo methods are essential across diverse scientific fields, yet their efficiency is frequently hampered by critical slowing down-a sharp increase in autocorrelation times near phase transitions. Although deep learning approaches, such as neural-network-based samplers, have been proposed to a...

---

### 22. [ITGPT: Generative Pretraining on Irregular Timeseries](https://arxiv.org/abs/2605.16069)

**Authors**: Antoine Honor\'e, Ming Xiao  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.16069v1  

#### Abstract
Timeseries regression models often struggle to leverage large volumes of labeled multimodal data, particularly when the data are irregularly sampled or contain missing values. This is common in domains like healthcare and predictive maintenance, where data are collected from unreliable sources, and ...

---

### 23. [SDOF: Taming the Alignment Tax in Multi-Agent Orchestration with State-Constrained Dispatch](https://arxiv.org/abs/2605.15204)

**Authors**: Zhantao Wang  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.15204v1  

#### Abstract
Multi-agent orchestration frameworks such as LangChain, LangGraph, and CrewAI route tasks through graph-based pipelines but do not enforce the stage constraints that govern real business processes. We present SDOF, a framework that treats multi-agent execution as a constrained state machine. SDOF op...

---

### 24. [Nudging Beyond the Comfort Zone: Efficient Strategy-Guided Exploration for RLVR](https://arxiv.org/abs/2605.15726)

**Authors**: Chanuk Lee, Sangwoo Park, Minki Kang, Sung Ju Hwang  
**Category**: cs.AI  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.15726v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has emerged as a scalable paradigm for improving the reasoning capabilities of large language models. However, its effectiveness is fundamentally limited by exploration: the policy can only improve on trajectories it has already sampled. While in...

---

### 25. [Contexting as Recommendation: Evolutionary Collaborative Filtering for Context Engineering](https://arxiv.org/abs/2605.15721)

**Authors**: Jiachen Zhu, Zhuoying Ou, Congmin Zheng, Yuxiang Chen, Zeyu Zheng, Rong Shan, Lingyu Yang, Lionel Z. Wang, Weiwen Liu, Yong Yu, Weinan Zhang, Jianghao Lin  
**Category**: cs.CL  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.15721v1  

#### Abstract
Large Language Models (LLMs) are highly sensitive to their input contexts, motivating the development of automated context engineering. However, existing methods predominantly treat this as a global search problem, seeking a single context strategy that maximizes average performance across a dataset...

---

### 26. [Multi-Level Contextual Token Relation Modeling for Machine-Generated Text Detection](https://arxiv.org/abs/2605.16107)

**Authors**: Chenwang Wu, Yiuming Cheung, Bo Han, Shuhai Zhang, Defu Lian  
**Category**: cs.CL  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.16107v1  

#### Abstract
Machine-generated texts (MGTs) pose risks such as disinformation and phishing, underscoring the need for reliable detection. Metric-based methods, which extract statistically distinguishable features of MGTs, are often more practical than complex model-based methods that are prone to overfitting. Gi...

---

### 27. [SGR: A Stepwise Reasoning Framework for LLMs with External Subgraph Generation](https://arxiv.org/abs/2605.16117)

**Authors**: Xin Zhang, Yang Cao, Baoxing Wu, Kai Song, Siying Li  
**Category**: cs.CL  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.16117v1  

#### Abstract
Large Language Models (LLMs) have demonstrated strong capabilities across diverse NLP applications, such as translation, text generation, and question answering. Nevertheless, they remain limited in complex settings that demand deep reasoning and logical inference. Since these models are trained on ...

---

### 28. [ParamSpMM: Adaptive and Efficient Sparse Matrix-Matrix Multiplication on GPUs for GNNs](https://arxiv.org/abs/2605.15695)

**Authors**: Lixing Zhang, Guanhua Ye, Hongzheng Li, Shigang Li, Yingxia Shao  
**Category**: cs.DC  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.15695v1  

#### Abstract
Fueled by the ability to mine real-world graph data, GNN applications have experienced phenomenal growth. Sparse Matrix-Matrix Multiplication (SpMM) is a critical operator in GNNs. However, existing SpMM designs for GNNs struggle to adapt to diverse input characteristics. In this paper, we first con...

---

### 29. [Controllable Molecular Generative Foundation Models](https://arxiv.org/abs/2605.15354)

**Authors**: Yihan Zhu, Yuhan Liu, Weijiang Li, Tengfei Luo, Meng Jiang  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.15354v1  

#### Abstract
Despite the success of foundation models in language and vision, molecular graph generation still lacks a unified framework for heterogeneous design tasks with reliable controllability. While reinforcement learning (RL) offers a natural post-training mechanism for task-specific optimization, applyin...

---

### 30. [A Retrieval-Enhanced Transformer for Multi-Step Port-of-Call Sequence Prediction in Global Liner Shipping](https://arxiv.org/abs/2605.15937)

**Authors**: Yanzhao Su, Fang He, Yineng Wang  
**Category**: cs.LG  
**Published**: 2026-05-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.15937v1  

#### Abstract
Accurate multi-step port-of-call sequence prediction is vital for tactical resource orchestration and logistical efficiency. However, existing methods struggle with unreliable voyage schedules and the inability of AIS data to provide visibility beyond the immediate next port. To address this, this s...

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
