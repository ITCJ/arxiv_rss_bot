# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-24 06:43:50 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model](https://arxiv.org/abs/2602.19128)

**Authors**: Shiyi Cao, Ziming Mao, Joseph E. Gonzalez, Ion Stoica  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2602.19128v1  

#### Abstract
Optimizing GPU kernels is critical for efficient modern machine learning systems yet remains challenging due to the complex interplay of design factors and rapid hardware evolution. Existing automated approaches typically treat Large Language Models (LLMs) merely as stochastic code generators within...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：K-Search: LLM Kernel Generation via Co-Evolving Intrinsic World Model**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代机器学习系统严重依赖高性能 GPU kernels 来实现高效的训练和推理。然而，手动优化这些 kernels 极其复杂且耗时，原因包括：
- 设计空间巨大（如 tiling、memory layout、synchronization 等）；
- 硬件快速演进（如从 Hopper 到 Blackwell），导致旧优化不再适用；
- 编译和评测开销大，搜索预算有限。

现有的自动化方法（如 OpenEvolve）通常将 **LLM 视为随机代码生成器**，在程序空间中进行基于启发式的进化搜索。这类方法存在以下缺陷：
- 缺乏高层规划能力，无法协调多步结构性变换（例如先重构内存布局再向量化）；
- 中间步骤可能因暂时性能下降或编译错误被过早丢弃；
- 难以探索非单调优化路径（non-monotonic optimization paths）。

---

### **提出了什么新方法或新思路**
本文提出 **K-SEARCH**，其核心是 **Search via Co-Evolving World Model** 范式，主要创新如下：

#### ✅ **引入“世界模型”（World Model）进行分层搜索**
- 将 kernel 生成建模为一个**规划问题**，而非直接在程序空间中盲目搜索。
- 使用 LLM 作为 **intrinsic world model**，维护一个显式的搜索树（search tree），其中：
  - **CLOSED 节点**：已探索并完成局部优化的状态；
  - **OPEN 节点**：待探索的高阶优化意图（如 “fuse head”、“unroll loop”）；
  - 每个节点附带一个由 LLM 估计的优先级分数 $ V \in [0,1] $。

#### ✅ **解耦高层算法规划与低层程序实例化**
- **高阶决策**（Action Selection）：由 world model 基于历史反馈选择最有潜力的优化意图；
- **低阶实现**（Local Refinement）：通过多次采样 LLM 实现该意图，直到连续失败达到阈值（stagnation limit）；
- 这种机制允许系统容忍临时实现缺陷（如语法错误），避免因一次失败就放弃有效策略。

#### ✅ **动态共进化机制（Co-Evolution）**
- World model 不是静态的，而是通过 in-context learning 在每次执行后更新自身认知：
  - **Insert**：添加新的子优化动作；
  - **Update**：调整现有动作的优先级；
  - **Prune**：剪枝无效分支，集中资源到有前景的方向。

---

### **相比现有方法的优势**
| 维度 | 传统进化方法（如 OpenEvolve） | K-SEARCH |
|------|-------------------------------|---------|
| 搜索空间 | 直接在程序文本空间搜索 | 在**高阶优化意图空间**搜索 |
| 规划能力 | 无显式规划，仅靠变异和选择 | 显式利用 LLM 的推理与规划能力 |
| 容错性 | 单次失败即丢弃策略 | 支持多次尝试，容忍实现噪声 |
| 效率 | 大量预算浪费在无效/错误代码上 | 更精准地聚焦于潜在有效的优化路径 |
| 可扩展性 | 难以处理复杂结构性变化 | 支持多步协同优化（如先融合头再拆分序列） |

---

## **2. 核心实验方法和设置**

### **使用的数据集与任务**
实验基于 **FlashInfer-Bench**（Xing et al., 2026）中的多个复杂 kernel，涵盖主流 LLM 推理场景：

| Kernel | 类型 | 特点 |
|--------|------|------|
| **MLA Paged Prefill / Decode** | 多级注意力（Multi-Level Attention） | 使用 paged KV-cache，支持动态批处理 |
| **GQA Paged Decode** | 分组查询注意力（Grouped Query Attention） | 内存受限，需高效复用 KV 数据 |
| **FP8 MoE** | 混合专家模型（Mixture-of-Experts） | 数据依赖路由、负载均衡挑战 |
| **GPUMode TriMul** | 第三方竞赛任务 | AlphaFold3 中的关键模块，计算密集 |

所有任务均使用真实流量捕获的 workload traces 进行评测。

---

### **实验设置和评估指标**

#### 🔹 **评估流程**
- 每轮迭代调用 evaluator 编译、验证功能正确性，并测量延迟；
- 总预算为 **120 次评估**（FlashInfer 任务）或 **300 次**（TriMul）；
- 报告每轮的最佳累计得分（best-so-far score）。

#### 🔹 **目标函数（Objective Score）**
定义为相对于参考 baseline 的加速比（speedup）：
$$
J(x) = s \cdot \frac{p_{\text{ref}}}{p}, \quad s \in \{0,1\}
$$
其中 $ s $ 表示是否通过功能测试，$ p $ 是实际延迟。

最终性能以 **几何平均延迟** 或 **平均 $ J(x) $ 得分** 衡量。

#### 🔹 **硬件环境**
- GPU：NVIDIA H100（FlashInfer）、B200（MoE on Blackwell）、H100（TriMul）
- CUDA 12.8, PyTorch 2.8.0, FlashInfer 0.5.3

---

### **基线方法对比**
比较三种自动化 kernel 生成系统：
1. **OpenEvolve**（Superintelligence, 2025）  
   - 基于 MAP-Elites 和质量多样性搜索的经典进化框架。
2. **ShinkaEvolve**（Lange et al., 2025a）  
   - 引入新颖性感知拒绝机制提升样本效率。
3. **K-SEARCH**（本文方法）  
   - 使用 co-evolving world model 指导搜索。

统一使用 `gemini-3-pro-preview` 作为 LLM backend，初始程序一致，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| Kernel | K-SEARCH 最终得分 | OpenEvolve | ShinkaEvolve | 提升倍数（vs OE） |
|--------|------------------|------------|--------------|------------------|
| **GQA Decode** | 76.0 | 44.2 | 27.7 | **1.72×** |
| **MLA Decode** | 47.1 | 39.9 | 34.7 | **1.18×** |
| **MLA Prefill** | 57.4 | 19.5 | 11.3 | **2.95×** |
| **FP8 MoE (Blackwell)** | 44.1 | 3.09 | 27.9 | **14.3×** |
| **平均提升** | — | — | — | **2.10×** |

> 💡 **特别亮点**：在极具挑战性的 MoE kernel 上实现了 **14.3 倍超越 OpenEvolve** 的性能突破。

---

### **与基线方法的对比结果**

#### 📊 图3(a)：搜索过程曲线
- K-SEARCH 在所有 kernels 上均显著快于基线收敛；
- 平均最终得分达 **56.13**，远超 OpenEvolve（26.68）和 ShinkaEvolve（25.37）；
- 收敛更稳定，方差更低。

#### 📊 图3(b)：逐 workload 性能分布
- 在 152 个 workload 实例中，K-SEARCH 在绝大多数情况下表现最优；
- 仅在小批量（batch_size=1 或 16）的 GQA 任务中略逊于基线（因其采用 split-K 并行策略，对小 batch 开销较大）。

#### 📊 图3(c)：达到特定加速比的工作负载比例
- 在 GQA decode 中：
  - K-SEARCH 在 **87.5%** 的 workload 上实现 ≥1.5× 加速；
  - OpenEvolve 和 ShinkaEvolve 分别仅为 50.0% 和 39.6%。
- 在 MLA prefill 中：
  - K-SEARCH 在 **57.9%** workload 上实现 ≥1.4× 加速；
  - 所有 baseline 均未有任何 workload 达到此水平。

---

### **消融实验结果（隐含分析）**

虽然未设正式消融实验，但文中通过案例分析揭示了关键设计的有效性：

#### ✅ **Local Refinement 的作用**
- 允许多次尝试同一优化意图，过滤掉因语法错误导致的失败；
- 防止因一次编译失败而误判整个策略无效。

#### ✅ **Tree Edit Operations 的有效性**
- **Insert**：能动态提出新假设（如 `low_overhead_split_k`）；
- **Update**：根据证据动态降级无效分支（如 `independent_heads`）；
- **Prune**：及时清理死胡同，节省搜索资源。

#### ✅ **Split-K 并行策略的权衡**
- K-SEARCH 自动选择了适合长序列的 split-K 策略，在大 batch 下优势明显；
- 但也暴露了当前方法缺乏自动适应不同 workload 特征的能力（未来可改进）。

---

## **4. 关键结论和发现**

### **论文的主要发现**
1. ✅ **LLM 不只是代码生成器，更是强大的规划引擎**  
   利用 LLM 的 prior knowledge 构建 world model，可显著提升搜索效率和深度。

2. ✅ **解耦“意图”与“实现”是关键**  
   显式分离 high-level planning 与 low-level instantiation，使系统能够探索复杂的非单调优化路径。

3. ✅ **共进化机制带来持续自我改进能力**  
   通过 in-context learning 动态更新搜索策略，实现类似 model-based RL 的闭环优化。

4. ✅ **在真实复杂 kernel 上实现 SOTA 性能**  
   - 在 MoE kernel 上取得 **14.3× 超越 OpenEvolve** 的成果；
   - 在 GPUMode TriMul 竞赛中达到 **1030 μs**，超越此前所有人类与自动化方案。

---

### **方法的局限性**
1. ❌ **对小 batch 场景适应不佳**  
   当前策略偏向最大化并行利用率，牺牲了极小 batch 的效率。

2. ❌ **依赖高质量 evaluator 和编译环境**  
   若编译失败频繁或 profiler 不准确，会影响 world model 学习。

3. ❌ **尚未完全自动化初始化过程**  
   某些任务仍需提供初始 CUDA 程序（如 MLA decode），否则难以生成可运行代码。

4. ❌ **world model 更新依赖 in-context learning**  
   当前未微调模型，长期记忆受限于上下文长度。

---

### **未来工作方向**
1. ➤ **构建自适应调度器**  
   根据 workload 特征（如 batch size、sequence length）动态选择最优 kernel 结构。

2. ➤ **结合 symbolic reasoning 与 formal verification**  
   提升优化意图的逻辑一致性，减少无效探索。

3. ➤ **端到端 trainable world model**  
   将 world model 参数化并联合训练，替代当前 in-context 更新方式。

4. ➤ **跨任务迁移学习**  
   利用在一个 kernel 上学到的经验指导其他相关 kernel 的优化。

5. ➤ **集成 compiler IR 层面的搜索**  
   在 Triton 或 MLIR 等更高抽象层级应用 co-evolving world model。

---

> 🔗 **开源地址**：[https://github.com/caoshiyi/K-Search](https://github.com/caoshiyi/K-Search)  
> 📄 **论文链接**：[arXiv:2602.19128](https://arxiv.org/abs/2602.19128)

</details>

---

### 2. [Proximity-Based Multi-Turn Optimization: Practical Credit Assignment for LLM Agent Training](https://arxiv.org/abs/2602.19225)

**Authors**: Yangyi Fang, Jiaye Lin, Xiaoliang Fu, Cong Qin, Haolin Shi, Chang Liu, Peilin Zhao  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2602.19225v1  

#### Abstract
Multi-turn LLM agents are becoming pivotal to production systems, spanning customer service automation, e-commerce assistance, and interactive task management, where accurately distinguishing high-value informative signals from stochastic noise is critical for sample-efficient training. In real-worl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Proximity-Based Multi-Turn Optimization: Practical Credit Assignment for LLM Agent Training*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在多轮 LLM Agent 的训练中，**信用分配（credit assignment）** 是一个核心挑战。现有基于分组的策略优化方法（如 GRPO、GiGPO）存在两个关键缺陷：

- **Episode-Level 问题（上下文盲区归一化）**：  
  使用 z-score 归一化计算优势函数时，忽略了任务难度的上下文信息。例如，在成功率高达 75% 的简单任务中失败，可能只是随机噪声，却会被赋予高惩罚；而在成功率仅 25% 的困难任务中成功，则是能力突破，应被显著奖励。但传统方法对两者赋予相似的绝对优势值，导致**误分配学习信号**。

- **Step-Level 问题（硬边界划分）**：  
  现有方法通过精确匹配或相似度阈值将状态划分为离散组（hard boundary partitioning），这会导致：
  - 阈值过严 → 大量单例组（singleton groups），无法进行组内归一化；
  - 阈值过松 → 不同语义的状态被等权重对待，削弱了信用区分能力。

### 提出了什么新方法或新思路
作者提出 **ProxMO（Proximity-based Multi-turn Optimization）**，一种融合全局上下文信息的双层级信用分配框架：

#### ✅ Episode-Level 创新：Success-Rate-Aware Modulation（成功率感知调制）
引入 **Polarized Signal Controller (PSC)**，根据整个 episode 组的经验成功率 $ p $ 动态调整梯度强度：
- 在低成功率任务中，**放大成功信号**（强化罕见突破）；
- 在高成功率任务中，**衰减失败惩罚**（抑制噪声干扰）。

公式形式为：
$$
w(R, p) = 1 + \beta \cdot f(R, p),\quad A_E(T) = w(R(T), p) \cdot A_{\text{z-score}}(T)
$$
其中 $ f(R,p) $ 是基于 sigmoid 函数设计的非线性调制项。

#### ✅ Step-Level 创新：Proximity-Based Soft Aggregation（基于邻近性的软聚合）
摒弃硬分组，采用 **连续语义加权机制** 来构建 step-level 基线：
- 使用 TF-IDF 向量计算状态间的语义相似度；
- 所有状态都参与聚合，权重由温度缩放的 softmax 决定：
  $$
  w_{ij} = \frac{\exp(\text{sim}(s_t^{(i)}, s_k^{(j)}) / \tau)}{\sum \cdots}
  $$
- 基线 $ B_t = \sum w_{ij} R_j^{(k)} $，优势 $ A_s(a_t) = R_t - B_t $

该机制消除了 singleton degeneracy，并实现了细粒度的动作价值估计。

### 相比现有方法的优势
| 特性 | GRPO | GiGPO | ProxMO |
|------|------|-------|--------|
| 是否依赖 critic 网络 | ❌ | ❌ | ❌ |
| 是否考虑任务难度上下文 | ❌ | ❌ | ✅（episode-level） |
| 是否支持细粒度 step-level 信用分配 | ❌ | ⚠️（受限于 exact matching） | ✅（soft aggregation） |
| 对单例状态处理能力 | 无意义归一化 | 优势为 0 | 仍可获得有效信用信号 |
| 插件兼容性 | 标准 GRPO 框架 | 改进型 GRPO | ✅ 可直接集成至 GRPO |

> ✅ **核心优势总结**：ProxMO 在不增加 critic 网络的前提下，实现了更鲁棒、更具上下文感知能力的信用分配，且具备“即插即用”特性，适合工业部署。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ALFWorld**：具身环境，包含 3,827 个任务实例，涵盖六类家庭活动：
  - Pick & Place (Pick)
  - Examine in Light (Look)
  - Clean & Place (Clean)
  - Heat & Place (Heat)
  - Cool & Place (Cool)
  - Pick Two & Place (Pick2)

  任务需通过文本交互完成复杂操作（如“把苹果加热后放进冰箱”），强调长期规划与状态追踪。

- **WebShop**：基于 HTML 的电商购物环境，含 1.1M 商品和 12K 用户指令。
  - 代理需执行搜索、浏览、筛选、购买决策；
  - 观察空间为半结构化 HTML 文本；
  - 模拟真实用户耐心限制（最多 15 步）。

### 实验设置和评估指标
| 设置项 | 配置 |
|--------|------|
| 主干模型 | Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct |
| 超参数（默认） | $\alpha=4.0$, $\beta=0.1$, $\tau=0.1$, $\gamma=0.95$, $w=1.0$, $N=8$ |
| 最大 episode 长度 | ALFWorld: 50 步；WebShop: 15 步 |
| Prompt 长度 | ALFWorld: 2048 tokens；WebShop: 4096 tokens |
| 训练迭代次数 | 150 |
| 随机种子 | 3 次重复实验取均值 ± 标准差 |

#### 评估指标
- **Success Rate (%)**：任务是否最终成功完成；
- **Score (%)**：属性匹配程度（尤其适用于 WebShop）；
- **All**：所有任务的平均得分；
- **Succ.**：仅在成功轨迹上的平均得分。

### 基线方法对比
| 类别 | 方法 |
|------|------|
| 封闭源模型 | GPT-4o, Gemini-2.5-Pro |
| Prompting Agent | ReAct, Reflexion |
| RL Training Methods | GRPO, GiGPO |

> 所有方法使用相同 prompt 模板与训练配置，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### 在 Qwen2.5-1.5B 上的表现（部分关键指标）：

| Method | ALFWorld (All) | WebShop (Score/Succ.) |
|--------|----------------|------------------------|
| GPT-4o | 48.0 | 31.8 / 23.7 |
| Gemini-2.5-Pro | 60.3 | 42.5 / 35.9 |
| GRPO | 70.3 | 73.1 / 52.2 |
| GiGPO | 85.2 | 81.7 / 62.3 |
| **ProxMO (Ours)** | **90.6** | **85.3 / 67.1** |

> 💡 **亮点**：即使是 1.5B 小模型，ProxMO 已超越 GPT-4o 和 Gemini 等超大规模闭源模型。

#### 在 Qwen2.5-7B 上的表现：

| Method | ALFWorld (All) | WebShop (Score/Succ.) |
|--------|----------------|------------------------|
| GRPO | 79.8 | 79.2 / 67.2 |
| GiGPO | 89.5 | 85.5 / 74.8 |
| **ProxMO (Ours)** | **94.5** | **87.2 / 76.5** |

> ✅ 平均提升约 **18–29%**（vs GRPO），尤其在长程任务（如 Look, Cool, Pick2）上增益显著。

### 与基线方法的对比结果
- **vs GRPO**：全面大幅领先，特别是在低成功率任务中表现突出，说明其成功率感知机制有效。
- **vs GiGPO**：尽管 GiGPO 引入了 step-level 匹配，但在高维状态空间下仍受 singleton 困扰，而 ProxMO 通过 soft aggregation 显著缓解此问题。
- **vs GPT-4o/Gemini**：ProxMO 训练的小模型即可匹敌甚至超越顶级闭源模型，验证了其强大的样本效率与泛化能力。

### 消融实验结果（Ablation Study）
在 ALFWorld 上对 Qwen2.5-1.5B 进行消融分析（Figure 4）：

| 变体 | 性能趋势 |
|------|----------|
| **Full ProxMO** | 最优性能，全面领先 |
| **w/o PSC**（移除 episode-level 调制） | 在成功率波动大的任务中明显下降 |
| **w/o PSA**（移除 step-level 软聚合） | 在长程任务（如 Pick2, Look）中性能骤降 |
| **vs GiGPO** | ProxMO 全面优于强基线 |

> 🔍 **发现**：两个模块独立有效，且具有**协同效应（synergy）** —— episode-level 的难度感知增强了 step-level 精确信用的价值。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **信用分配必须考虑上下文**：  
   单纯依赖统计偏差（如 z-score）会因忽略任务难度而导致错误的学习信号分配。

2. **双层级设计至关重要**：  
   - Episode-level 的 success-rate-aware modulation 抑制噪声、增强突破信号；
   - Step-level 的 proximity-based soft aggregation 实现细粒度动作评价，避免 singleton 退化。

3. **连续语义加权优于硬分组**：  
   使用 TF-IDF + 温度加权的方式，使所有状态都能贡献信用信号，提升稳定性与鲁棒性。

4. **高效且可部署**：  
   ProxMO 仅引入轻量级算术运算，**训练开销仅增加 1.09%**（vs GRPO），无需额外神经网络，易于集成到现有 pipeline。

### 方法的局限性
- 当前实验集中在资源高效的中小规模模型（1.5B/7B），尚未在更大基础模型（如 70B+）上验证其扩展性。
- 使用 TF-IDF 表示状态语义，在极端复杂的 multimodal 场景中可能表达能力有限。
- 超参数虽具鲁棒性，但仍需初步调优（如 $\alpha, \beta, \tau$）。

### 未来工作方向
- 将 ProxMO 扩展至 VLM（Vision-Language Model）Agent，结合图像嵌入进行跨模态 proximity 计算。
- 探索动态调整 temperature $\tau$ 或 modulation 参数以实现自适应学习。
- 在更多现实世界场景（如 AutoGPT、GUI 自动化）中测试其通用性与迁移能力。
- 结合 offline RL 或 preference modeling 进一步提升数据利用效率。

---

> 📌 **一句话总结**：  
> **ProxMO 通过 episode-level 成功率感知调制与 step-level 连续语义加权聚合，解决了多轮 LLM Agent 中的上下文盲区与单例退化问题，在保持极低计算开销的同时，显著提升了训练效率与最终性能，具备出色的工业落地潜力。**

</details>

---

### 3. [Leap+Verify: Regime-Adaptive Speculative Weight Prediction for Accelerating Neural Network Training](https://arxiv.org/abs/2602.19580)

**Authors**: Jeremy McEntire  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2602.19580v1  

#### Abstract
We introduce Leap+Verify, a framework that applies speculative execution -- predicting future model weights and validating predictions before acceptance -- to accelerate neural network training. Inspired by speculative decoding in language model inference and by the Automatically Scalable Computatio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Leap+Verify: Regime-Adaptive Speculative Weight Prediction for Accelerating Neural Network Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代神经网络训练是一个**高度顺序化的过程**，每一步梯度更新都依赖于前一步的参数状态，导致训练耗时极长、计算资源消耗巨大。尽管已有研究尝试通过预测未来权重来加速训练（如 weight nowcasting），但这些方法普遍存在两个问题：
- **无条件应用预测**：不验证预测是否合理就直接接受，容易引入错误；
- **忽略训练动态的阶段性差异**：未考虑不同训练阶段（如混沌、过渡、稳定）中权重轨迹的可预测性差异。

Leap+Verify 针对上述问题，提出了一种**基于训练阶段自适应的投机式权重预测框架**，旨在安全且高效地跳过部分梯度更新步骤，从而加速训练。

---

### 提出的新方法与新思路

Leap+Verify 的核心思想是将程序执行中的 **Automatically Scalable Computation (ASC)** 架构和语言模型推理中的 **speculative decoding** 范式迁移到神经网络训练中，构建一个“**预测 → 验证 → 接受**”（predict-then-verify）机制。

其三大创新点如下：

#### （1）Leap+Verify 框架：Verify-then-Accept 机制
- 在每个 checkpoint 处，使用解析方法（analytic extrapolation）预测 $K$ 步后的模型权重 $\theta_{t+K}$；
- 将预测权重加载进模型，在 held-out validation data 上计算损失 $L_{t+K}$；
- 只有当该损失满足预设的接受标准（如严格下降、在一定范围内等）时，才接受跳跃，跳过中间 $K$ 步更新；
- 若拒绝，则继续从当前步正常训练——**预测完全无副作用**。

> ✅ 类比 speculative decoding：小 draft model 生成候选 token，大 target model 并行验证；此处，“draft model” 是 weight predictor，“target model” 是原训练流程。

#### （2）三阶段动态检测机制（Regime-Conditional Prediction）
引入基于 **activation-space cosine similarity** 的实时 regime detector，将训练过程划分为三个动态阶段：
- **Chaotic（混沌）**：表示层变化剧烈，不可预测；
- **Transition（过渡）**：表示趋于稳定，具有一定的规律性；
- **Stable（稳定）**：表示高度一致，适合高精度预测。

> 🔍 判定依据：固定 probe set 上连续 checkpoint 的最终隐藏状态之间的余弦相似度，作为局部 Lyapunov 指数的代理信号。

此机制实现了**条件预测**：仅在 Transition 和 Stable 阶段启用预测，避免在 Chaotic 阶段浪费计算。

#### （3）揭示“动量灾难”与“规模依赖的阶段分布瓶颈”
- 发现 **Adam 动量外推（momentum extrapolation）在所有尺度下均失败**，称为“Universal Momentum Catastrophe”；
- 更重要的是，随着模型变大，训练停留在 **Chaotic 阶段的比例显著上升**，使得即使预测器本身更准确，也因可用窗口太少而难以发挥作用——即真正的瓶颈从 **predictor accuracy** 转向了 **regime availability**。

---

### 相比现有方法的优势

| 方面 | 现有方法（如 WNN, NiNo, XGrad） | Leap+Verify |
|------|-------------------------------|-----------|
| 是否验证预测 | ❌ 无验证，直接应用 | ✅ Verify-then-Accept，确保安全性 |
| 是否感知训练阶段 | ❌ 全程无差别预测 | ✅ 动态检测 regime，只在可预测阶段预测 |
| 外推方式 | 多基于 optimizer state 或 learned model | ✅ 引入 finite-difference predictors（linear/quadratic）避免 norm explosion |
| 可扩展性分析 | 缺乏对 scale 影响的系统研究 | ✅ 揭示“越大越混沌”，指出新瓶颈 |

---

## 2. 核心实验方法和设置

### 使用的数据集与模型
- **数据集**：WikiText-103
- **序列长度**：256
- **模型架构**：
  - **GPT-2 124M**：12 层，768 hidden dim，12 heads
  - **Qwen 2.5-1.5B**：28 层，1536 hidden dim，12 heads
- **优化器**：AdamW ($\beta_1=0.9$, $\beta_2=0.999$, lr=$5\times10^{-5}$)，cosine 学习率调度，warmup 100 步
- **总训练步数**：2000 步，checkpoint 每 50 步保存一次（共 40 个）

### 实验设置
采用三阶段评估协议（three-pass protocol）以保证公平性和可复现性：

1. **Pass 1（Training）**  
   完整训练并记录 activation fingerprint（probe sentences 上的 final hidden states）、validation loss、regime 分类。

2. **Pass 2（K-Sweep）**  
   对非 chaotic checkpoints 测试三种 predictor 在 $K \in \{5,10,25,50,75,100\}$ 下的表现，使用三种 acceptance criteria：
   - **Strict**：$L_{t+K} < L_t$
   - **Adaptive**：$L_{t+K} < L_t + \sigma_L$
   - **Proximity (pct)**：$\left|L_{t+K} - L_t\right| < \epsilon \cdot L_t$（通常取 $\epsilon=0.05$）

3. **Pass 3（Cascades）**  
   在 stable checkpoints 上测试级联预测（cascaded prediction），配置为 $(D,K) \in \{(4,25),(2,50),(10,10)\}$，最多可跳 $D\times K$ 步。

### 基线方法对比
本文并未直接比较传统训练方法（因其目标不是端到端加速而是机制有效性），但隐含对比了以下几类 prior work：
- **Unconditional predictors**：如 WNN [Jang and Han, 2023]、NiNo [Knyazev et al., 2025]、XGrad [Guan et al., 2024]
- **Optimizer-based extrapolation**：如 Adam 动量外推（本文作为 baseline 测试并证明其失败）
- **Program-level speculative execution**：ASC 架构（理论来源）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### （1）Regime 分布随规模剧变（Table 1）
| Model | Chaotic | Transition | Stable |
|-------|--------|----------|--------|
| GPT-2 124M | 4% | 60% | 34% |
| Qwen 2.5-1.5B | 64% | 31% | 2.5% |

> ⚠️ 更大的模型反而更少进入稳定阶段！

#### （2）Strict Acceptance Rate @ K=5（Table 2）
| Predictor / Model | GPT-2 124M (Stable) | Qwen 1.5B (Transition) |
|-------------------|--------------------|------------------------|
| Momentum | ~10% | 0% |
| Linear | **24.3%** | **37.2%** |
| Quadratic | 22.1% | 36.5% |

> ✅ Finite-difference predictors 显著优于 momentum；且在更大模型的 transition 阶段表现更好。

#### （3）Proximity Acceptance (@5%, Table 3)
| K | Linear (124M Stable) | Linear (1.5B Transition) |
|----|---------------------|-------------------------|
| 5 | 98.9% | **100.0%** |
| 10 | 95.3% | 100.0% |
| 25 | 74.6% | 94.6% |
| 50 | 54.1% | 66.2% |

> 📈 表明 short-horizon prediction 几乎总是可接受，尤其在大模型中衰减更慢。

#### （4）Universal Momentum Catastrophe（Table 4）
| K | Predicted Loss / Actual Loss (124M) | Ratio |
|----|----------------------------------|-------|
| 5 | 201 / 1.64 | **122×** |
| 100 | 17691 / 1.64 | **10,764×** |

| K | Predicted Loss / Actual Loss (1.5B) | Ratio |
|----|----------------------------------|-------|
| 5 | 142 / 0.82 | **173×** |
| 100 | 2468 / 0.82 | **3,009×** |

> 💥 动量外推导致预测损失爆炸式增长，无论大小模型均失败。

#### （5）Cross-Seed Consistency（Table 5）
| K | Linear CoV (%) | Quadratic CoV (%) |
|----|---------------|------------------|
| 5 | 0.0% | 0.0% |
| 10 | 0.0% | 0.0% |
| 25 | 5.2% | 7.9% |
| 100 | 63.3% | 149.1% |

> ✅ 结果高度可复现，尤其短 horizon 下几乎零方差。

---

### 与基线方法的对比结果
- **相比 momentum-based prediction**：完全失败，不能用于 speculative training；
- **相比 unconditional nowcasting methods**：Leap+Verify 通过 regime conditioning 和 verify-then-accept 机制，实现了**安全、可控、高效的加速潜力**；
- **相比 ASC/speculative decoding**：验证成本极低（仅需一次 forward pass），具备更强实用性。

---

### 消融实验结果
虽然没有显式的 ablation study 表格，但从设计中可看出以下关键消融结论：

| 组件 | 移除后的影响 |
|------|------------|
| Regime Detector | 导致在 chaotic 阶段无效预测，浪费算力 |
| Verify Step | 若直接接受预测，会引入严重误差甚至崩溃 |
| Finite-difference Predictors | 若仅用 momentum，预测完全失效 |
| Cascade Mechanism | 单步预测可行，但多步级联迅速失败（error accumulation） |

---

## 4. 关键结论和发现

### 主要发现
1. **动量外推普遍失败（Universal Momentum Catastrophe）**  
   使用 Adam 的 $m/\sqrt{v}$ 进行动量外推会导致 **norm explosion**，预测权重远离有效区域，造成验证损失暴涨上百至万倍。这不是数值问题，而是结构性缺陷。

2. **有限差分预测有效（Finite-difference works）**  
   基于实际观察到的 checkpoint delta 进行线性/二次外推（linear/quadratic extrapolation）能实现高达 **37% 的 strict acceptance rate**（Qwen 1.5B, K=5），且 proximity acceptance 在短距离接近 100%。

3. **真正瓶颈是 regime availability，而非 predictor accuracy**  
   虽然更大的模型在 transition/stable 阶段更具可预测性（smooth trajectory），但它们**绝大多数时间处于 chaotic regime**（64% vs. 4%），导致可用于预测的时间窗口极少——这是此前 weight nowcasting 工作未识别的关键瓶颈。

4. **训练阶段边界跨种子高度一致（±50 steps）**  
   表明 regime transition 是由优化 landscape 决定的，而非随机初始化主导，支持 regime detection 的泛化能力。

5. **Leap+Verify 安全、无副作用**  
   所有失败预测均可被 reject，不影响原始训练路径，符合“pure speculation”原则。

---

### 方法的局限性
- **训练步数有限**：仅运行 2000 步，Qwen 1.5B 很少进入 stable 阶段，限制了 long-horizon prediction 的评估；
- **regime thresholds 固定**：Thigh/Tlow 在小模型上校准，可能不适用于更大模型（activation similarity 整体偏高）；
- **cascade 效果不佳**：由于误差累积，深层级联预测快速失败；
- **未实现端到端加速测量**：目前是 offline evaluation，尚未集成到真实训练 loop 中报告 wall-clock speedup。

---

### 未来工作方向
1. **自适应阈值机制**：根据模型规模自动调整 regime detection 的相似度阈值；
2. **Ensemble Collapse**：利用多 seed 的 regime 同步性，动态合并训练进程；
3. **更大规模评估**：在 2.7B、7B 级别模型上测试 predictor hierarchy 和 regime 分布；
4. **在线集成与加速测量**：将 Leap+Verify 集成进训练 pipeline，量化实际 speedup；
5. **结合其他 optimizer**：探索 SGD、Lion 等优化器下的预测行为。

---

> 🔚 总结一句话：**Leap+Verify 提出了一种安全、自适应的投机式训练加速框架，揭示了“动量灾难”和“越大越混沌”的核心现象，指出未来 weight nowcasting 的关键挑战在于 regime availability 而非预测精度本身。**

</details>

---

### 4. [Training-Free Generative Modeling via Kernelized Stochastic Interpolants](https://arxiv.org/abs/2602.20070)

**Authors**: Florentin Coeurdoux, Etienne Lempereur, Nathana\"el Cuvelle-Magar, Thomas Eboli, St\'ephane Mallat, Anastasia Borovykh, Eric Vanden-Eijnden  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2602.20070v1  

#### Abstract
We develop a kernel method for generative modeling within the stochastic interpolant framework, replacing neural network training with linear systems. The drift of the generative SDE is $\hat b_t(x) = \nabla\phi(x)^\top\eta_t$, where $\eta_t\in\R^P$ solves a $P\times P$ system computable from data, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Training-Free Generative Modeling via Kernelized Stochastic Interpolants*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统基于 **diffusion models** 和 **stochastic interpolants** 的生成模型依赖于训练神经网络来估计时间相关的 **drift** 或 **score function**，这一过程计算成本高昂且需要大量调参。本文旨在消除这一训练步骤，提出一种无需训练即可实现高质量生成的方法。

### 🚀 提出的新方法与核心思想
作者提出了 **Kernelized Stochastic Interpolants (KSI)**，将生成建模转化为一个**核方法（kernel method）框架**，其核心是：

- 利用预定义的 **feature map** $\phi: \mathbb{R}^d \to \mathbb{R}^P$ 构造特征梯度 $ \nabla\phi(x) $；
- 将生成 SDE 的漂移项近似为：  
  $$
  b_t(x) = \nabla\phi(x)^T \eta_t
  $$
  其中系数 $\eta_t \in \mathbb{R}^P$ 是通过求解一个从数据中直接计算的 $P \times P$ 线性系统得到；
- 整个“学习”过程被简化为在每个时间步 $t_k$ 上求解一个线性回归问题，无需反向传播或优化迭代。

该方法的关键在于：
- **训练免费（Training-free）**：所有参数在采样前一次性预计算完成；
- **模型可组合（Model combinable）**：支持将多个预训练模型的 velocity fields 直接作为 feature gradients 进行线性融合。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **效率** | 避免了耗时的神经网络训练；仅需解小规模线性系统（$P \ll d$） |
| **灵活性** | 支持多种 feature maps（如 scattering transform、pretrained models），便于跨领域迁移 |
| **模型集成** | 可无缝合并不同架构、训练阶段甚至不同数据域的模型，无需 distillation 或 fine-tuning |
| **理论保障** | 引入最优扩散系数 $D_t^*$ 来最小化路径 KL 散度，提升样本质量 |

此外，相比 **Moment-Guided Diffusion (MGD)**：
- MGD 要求 $D_t \to 0$ 以达到最大熵分布，而本方法利用有限最优 $D_t^*$ 最小化生成误差；
- 本方法更简单（只需一次线性求解），但控制力较弱（不能显式指定匹配哪些统计量）。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 类型 | 维度 | 特点 |
|--------|------|-------|------|
| **S&P 500 日对数收益率** | 金融时间序列 | $d=6064$ | 单一真实轨迹，具有波动聚集、厚尾、杠杆效应等非高斯特性 |
| **3D Turbulence (压力切片)** | 物理场 | $64\times64$ | 各向同性湍流，长程相关与间歇结构 |
| **Dark Matter (对数密度)** | 宇宙学模拟 | $64\times64$ | 显著非高斯，纤维状宇宙网结构 |
| **Magnetic Turbulence (涡度)** | MHD 模拟 | $64\times64$ | 各向异性，含相干涡旋 |
| **Weak Lensing (收敛图)** | 引力透镜模拟 | $64\times64$ | 稀疏尖峰叠加在类高斯背景上 |
| **MNIST** | 图像 | $28\times28$ | 手写数字，用于测试弱模型集成 |
| **CelebA** | 图像 | $128\times128$ | 人脸图像，验证高分辨率扩展能力 |

### ⚙️ 实验设置
- **时间调度**：采用三角函数形式 $\alpha_t = \cos(\pi t / 2), \beta_t = \sin(\pi t / 2)$
- **积分器**：使用专为 $D_t^* \to \infty$（当 $t\to0$）设计的数值积分方案（见 Algorithm 1），无需 clamp 扩散系数
- **feature map 选择**：
  - 时间序列/物理场：**Wavelet Scattering Transform**（Morlet 小波）
  - 图像生成：使用多个 **under-trained U-Net velocity fields** 作为 feature gradients
- **线性系统求解**：
  - 在离散时间网格 $t_k = k/K$ 上预先求解 $\eta_{t_k}$
  - 使用经验估计构造 Gram 矩阵 $K_t$ 和响应向量 $r_t$

### 📊 评估指标
- **视觉定性分析**：生成样本与真实样本对比（如 volatility clustering、filamentary structure）
- **统计一致性检验**：
  - 密度直方图（log-return 分布）
  - 杠杆效应曲线（past return vs future volatility correlation）
- **Oracle Log-Likelihood**（MNIST）：用完全训练好的 U-Net 作为 oracle 评估生成样本似然
- **消融实验**：改变 ensemble size $P$ 观察性能变化

### 🔀 基线方法对比
- 无传统神经网络基线（因 focus 在“无需训练”）
- 对比对象为：
  - 单个弱模型输出（baseline 下限）
  - 不同训练步数的模型集合效果
  - 仅目标域 vs 跨域组合模型（cross-domain composition）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据与结果

#### ✅ 多尺度时间序列生成（S&P 500）
- 成功复现原始序列的：
  - 波动聚集现象（volatility clustering）
  - 厚尾分布（heavy-tailed log-return density）
  - 杠杆效应（negative correlation between past returns and future vol）
- 使用 $P=217$ scattering coefficients，在单一观测轨迹下实现高质量生成（图1）

#### ✅ 物理场生成（64×64）
- 四类物理场均能准确还原视觉结构：
  - 湍流中的间歇脉冲
  - 暗物质的丝状网络
  - 磁流体中的拉长涡旋
  - 弱引力透镜的稀疏峰值
- 使用 $P=6803$ scattering features（图2）

#### ✅ 弱模型集成（MNIST）
- 单个仅训练 50–100 步的 U-Net 输出严重失真；
- 使用 $P=20$ 个 100-step 模型组成的 ensemble 后，生成图像显著改善，出现清晰数字结构（图3 左）；
- Oracle log-likelihood 随 $P$ 单调上升，在 $P \sim 15$ 后趋于饱和（图3 右）；
- 100-step 模型组 consistently 优于 50-step 组。

#### ✅ 高分辨率图像生成（CelebA）
- 单个弱模型（5 epoch）输出模糊、不连贯；
- 使用 $P=25$ 弱模型集成后，生成的人脸图像结构完整、五官合理（图4）；
- 表明方法可扩展至高维自然图像。

#### ✅ 跨域模型组合（Cross-Domain Composition）
- 使用 Fashion-MNIST、EMNIST、Kuzushiji-MNIST 上训练的模型 + MNIST 模型共同构成 $P=40$ ensemble；
- 结果显示：跨域集成生成的数字比仅用 MNIST 模型更 sharp、更连贯（图5）；
- 表明低层次特征（边缘、笔画）可跨任务迁移并被自动加权利用。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **生成可以完全无需训练**：通过 kernelized formulation，将 drift estimation 转化为线性系统求解，实现了真正的 **training-free generation**。
2. **最优扩散系数 $D_t^*$ 至关重要**：
   - $D_t^* = \alpha_t \gamma_t / \beta_t$ 在 $t \to 0$ 时发散（$\infty$），在 $t \to 1$ 时趋于 0；
   - 该设计增强了鲁棒性，允许使用近似 drift 仍获得高质量样本；
   - 数值积分器能自然处理 $D_0 = \infty$，无需 clipping。
3. **feature map 决定表达能力**：
   - Scattering transform 适合多尺度物理过程；
   - Pretrained velocity fields 可直接复用，形成“模型即特征”的新范式。
4. **模型组合优于平均或蒸馏**：
   - 参数平均（model soups）、LoRA merging 等可能退化；
   - 本文方法通过线性系统自动学习最优权重，实现高效集成。

### ⚠️ 局限性
- **feature map 设计敏感**：性能高度依赖所选 $\phi$ 是否能捕捉目标分布的关键结构；
- **有限表达能力**：由于 $\phi$ 是有限维，无法表示任意复杂 drift，存在逼近误差；
- **不适合极端高维稀疏数据**：尚未在文本或其他离散空间验证；
- **缺乏显式控制机制**：不像 MGD 可指定要匹配的矩，控制粒度较低。

### 🔮 未来工作方向
1. **自动化 feature selection**：探索如何自适应地构建或选择最优 feature map；
2. **结合 MGD 与 KSI**：既回归 drift 又约束 moment，兼顾精度与可控性；
3. **应用于科学发现**：如从单次实验观测中生成类似动力学行为（single-trajectory inference）；
4. **在线更新机制**：允许动态增加新模型到 ensemble 中而不重新求解全部系统；
5. **理论深化**：研究 finite-rank kernel 下的泛化误差界与 MMD 收敛性。

---

> **总结一句话**：  
> 本文提出了一种全新的 **training-free 生成范式** —— **Kernelized Stochastic Interpolants**，通过将 drift 学习转化为线性系统求解，并引入最优扩散系数与专用积分器，实现了高质量、无需训练的生成，尤其擅长**多尺度物理建模**与**弱模型集成**，为 diffusion model 的轻量化与模块化提供了新路径。

</details>

---

### 5. [Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning](https://arxiv.org/abs/2602.19917)

**Authors**: Thanh Nguyen, Tung Luu, Tri Ton, Sungwoong Kim, Chang D. Yoo  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.19917v1  

#### Abstract
Offline reinforcement learning (RL) has garnered significant interest due to its safe and easily scalable paradigm. However, training under this paradigm presents its own challenge: the extrapolation error stemming from out-of-distribution (OOD) data. Existing methodologies have endeavored to addres...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本论文针对 **Offline Reinforcement Learning (Offline RL)** 中的核心挑战——**外部分布（Out-of-Distribution, OOD）数据引发的外推误差（extrapolation error）**。由于策略在训练过程中无法与环境交互，当评估未在行为策略（behavior policy）数据中出现的状态-动作对时，Q函数容易高估其价值，导致学习到次优甚至不稳定策略。

现有方法存在以下局限：
- **过于保守**：如CQL等方法对所有OOD动作统一惩罚，抑制了有效探索；
- **不确定性建模效率低**：基于Q-ensemble的方法虽能更好量化不确定性，但计算和内存开销大；
- **依赖精确的行为策略估计**：如BCQ、BEAR等方法受限于对行为策略的建模质量。

---

### 提出了什么新方法或新思路
作者提出了一种全新的 **Uncertainty-Aware Rank-One MIMO Q Network Framework**，其核心创新包括：

#### （1）Rank-One MIMO Q Network 架构
- 设计了一个 **Multi-Input Multi-Output (MIMO)** 结构的Q网络，可同时处理多个输入并输出多个Q值。
- 引入 **“共享权重 + 秩一适配器”**（shared weight + rank-one adapters）机制：
  - 所有ensemble成员共享一个主干网络（common shared network）；
  - 每个成员通过两个低维向量 $v_k$ 和 $s_k$ 构造个性化的秩一矩阵 $(v_k \circ s_k^T)$，与共享权重融合生成专属权重 $W_k = W_0 \circ (v_k s_k^T)$。
- 实现了类似Q-ensemble的不确定性建模能力，但参数量和计算成本接近单个网络。

#### （2）基于Lower Confidence Bound (LCB) 的悲观训练损失
- 在Bellman目标中使用 **最小Q头输出** 来近似LCB，即：
  $$
  \text{Target} = \min_{k=1..K} Q_k(s', a') - \alpha \log \pi(a'|s')
  $$
- 这种设计实现了**选择性惩罚OOD动作**，而非统一压制，提升了策略优化的灵活性。

#### （3）增强稳定性与鲁棒性的辅助机制
- **熵正则化（Entropy Bonus）**：鼓励策略在OOD区域保持高熵，避免过度置信；
- **In-distribution likelihood最大化**：引导策略优先选择数据集中高频出现的动作；
- **Lazy Policy Update**：减少策略更新频率以提升训练稳定性和效率。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **精度** | 显式建模不确定性，实现更精细的OOD识别与惩罚 |
| **效率** | 参数和计算开销远低于传统ensemble方法（接近单网络） |
| **通用性** | 不依赖行为策略估计，适用于各种覆盖程度的数据集（random到expert） |
| **稳定性** | 多重正则化手段缓解Q值发散风险 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验基于 **D4RL benchmark**，涵盖三个MuJoCo连续控制任务：
- **HalfCheetah**
- **Hopper**
- **Walker2d**

每项任务使用五类不同行为策略收集的数据集：
- `random-v2`
- `medium-v2`
- `medium-replay-v2`
- `medium-expert-v2`
- `expert-v2`

这些数据集覆盖从低质量（随机）到高质量（专家）的不同分布特性，全面测试算法鲁棒性。

---

### 实验设置和评估指标
- **训练步数**：3000 epochs，每个epoch 1000 steps，共3M steps；
- **评估方式**：每个算法运行4个随机种子，每次评估取10条轨迹（每条1000步），报告平均return；
- **归一化得分**：
  $$
  \text{normalized score} = 100 \times \frac{\text{score} - \text{score}_{\text{random}}}{\text{score}_{\text{expert}} - \text{score}_{\text{random}}}
  $$
  表示相对于随机策略和专家策略的表现百分比。

---

### 基线方法对比
与多种SOTA Offline RL算法进行比较：
| 方法 | 类型 | 特点 |
|------|------|------|
| **BCQ** | Policy constraint | 动作受限于行为策略附近 |
| **IQL** | Implicit Q-learning | 避免直接查询OOD Q值 |
| **BEAR** | MMD约束 | 强制策略支持集匹配 |
| **UWAC** | Dropout + uncertainty | 不确定性加权更新 |
| **CQL** | Conservative Q-learning | 对OOD动作统一惩罚 |
| **MOPO** | Model-based | 利用动态模型不确定性 |
| **TD3-BC** | Hybrid | TD3 + BC正则项 |
| **EDAC** | Ensemble Q | 多样化梯度增强保守性 |
| **PBRL** | Bootstrapped uncertainty | OOD采样 + ensemble penalization |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见Table 1）
在 **D4RL benchmark 上平均归一化得分达 83.6**，显著优于其他方法：

| 方法 | 平均得分 |
|------|---------|
| BCQ | 49.4 |
| CQL | 67.35 |
| EDAC | 71.2 |
| PBRL | 74.37 |
| **OURS** | **83.6** ✅ |

> **领先第二名PBRL达 +9.23 分**，且在多数子任务上取得SOTA。

---

### 与基线方法的对比结果
- 在 **messy data（如random、medium-replay）** 上表现尤为突出，说明对低质量数据具有更强鲁棒性；
- 在 **expert 数据集** 上也稳定达到或接近最优，得益于in-distribution likelihood项提供的稳定性；
- 超越model-based方法（如MOPO），表明无需复杂动态建模即可获得高性能；
- 明显优于policy-constrained方法（如BCQ、BEAR），验证了灵活策略优化的重要性。

---

### 消融实验结果（Ablation Study）

#### （1）参数 $K$（ensemble size）的影响（Table 3）
| $K$ | Average Return |
|-----|----------------|
| 2   | 0.19           |
| 5   | 92.9           |
| 10  | **112.8** ✅    |
| 15  | 23             |
| 20  | 0.4            |

- $K=10$ 时性能最佳，过高或过低都会下降；
- 验证了 $K$ 可作为控制“悲观程度”的超参数：太小 → 过于乐观；太大 → 过于保守。

#### （2）组件消融分析（Table 4）
| 组件 | Avg Return | Avg Q(s,π(s)) |
|------|------------|---------------|
| 全部（Entropy + Likelihood） | **112.9** ✅ | 373.4 |
| 仅Likelihood | 111.0 | 258.9 |
| 仅Entropy | 107.3 | 267.1 |
| 无额外项 | 109.6 | 453.2 |

- 同时使用熵和似然项效果最好；
- 缺少任一正则项会导致Q值虚高（overestimation），影响稳定性；
- 在expert数据集中，**Likelihood项至关重要**，否则难以收敛。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **不确定性感知是解决OOD问题的关键路径**：通过显式建模Q函数的不确定性，可以实现更精准的悲观估计，而非粗暴压制。
2. **高效的ensemble架构可行且必要**：Rank-One MIMO Q Network 成功将ensemble的优势（多样性、不确定性估计）与单网络的效率结合。
3. **LCB近似是一种高效且有效的训练机制**：使用min-Q head代替完整ensemble backpropagation，大幅加速训练而不牺牲性能。
4. **多维度正则化协同增效**：熵、似然、lazy update共同作用，显著提升训练稳定性与最终性能。

---

### 方法的局限性
- 当前框架仍假设状态空间为完全可观测（fully observed MDP），未考虑部分可观测场景（POMDP）；
- Rank-One结构的有效性依赖于良好初始化（如sign vectors），可能在极端非线性任务中受限；
- 尽管内存高效，但在超大规模网络（如Transformer-based agents）上的扩展性尚未验证。

---

### 未来工作方向
1. **扩展至Model-Based Offline RL**：将MIMO思想应用于动态模型不确定性建模；
2. **引入自适应K机制**：让网络自动调整ensemble大小以适应不同任务难度；
3. **应用于视觉输入任务**：验证该框架在图像观测下的有效性；
4. **结合离线元学习（Meta RL）**：构建跨任务泛化的uncertainty-aware agent；
5. **理论分析LCB近似的偏差边界**：为min-operator提供更强的理论支撑。

---

> ✅ **总体评价**：本文提出的 **Uncertainty-Aware Rank-One MIMO Q Network** 是 Offline RL 领域的一项重要进展，在**性能、效率、稳定性**三者之间取得了卓越平衡，为后续研究提供了高效且实用的新范式。

</details>

---

### 6. [Online decoding of rat self-paced locomotion speed from EEG using recurrent neural networks](https://arxiv.org/abs/2602.18637)

**Authors**: Alejandro de Miguel, Nelson Totah, Uri Maoz  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.18637v1  

#### Abstract
$\textit{Objective.}$ Accurate neural decoding of locomotion holds promise for advancing rehabilitation, prosthetic control, and understanding neural correlates of action. Recent studies have demonstrated decoding of locomotion kinematics across species on motorized treadmills. However, efforts to d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Online decoding of rat self-paced locomotion speed from EEG using recurrent neural networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本研究旨在解决**在自然、自选节奏（self-paced）条件下，如何从非侵入式 EEG 信号中连续、高精度地解码运动速度**这一挑战。以往的研究多集中于：
- 外部驱动的固定速度跑步机任务（enforced treadmill）
- 侵入式记录（如单神经元、ECoG）
- 离散行为分类（如走/停、快/慢），而非连续速度回归

这些问题限制了其在真实世界 BCI 应用中的泛化能力。

### 🚀 提出的新方法与创新思路
- **首次实现基于 skull-surface EEG 的在线、连续解码大鼠自选速度的 BCI 系统**。
- 构建了一个**异步（asynchronous）脑机接口框架**，直接对连续 EEG 流进行滑动窗口处理，输出实时速度估计。
- 采用 **Recurrent Neural Networks（特别是 LSTM）** 进行端到端（end-to-end）回归建模，捕捉 EEG 中的时间动态特征。
- 探索了**跨会话（cross-session）与跨个体（cross-subject）迁移学习**的可能性，并验证了 fine-tuning 在减少校准时间上的潜力。
- 首次系统分析了 EEG 对**过去与未来速度状态的编码能力**（retrospective and prospective decoding），揭示了大脑对动作时序表征的不对称性。

### 🔍 相比现有方法的优势
| 维度 | 传统方法局限 | 本文优势 |
|------|--------------|----------|
| **记录方式** | 多为侵入式（single-unit, ECoG） | 使用**非侵入式 skull-surface EEG**，更具临床转化前景 |
| **行为范式** | 强制节律或试次结构（trial-based） | 采用**完全自选节奏、无提示的连续运动**，更接近自然行为 |
| **解码目标** | 多为分类任务或离散变量 | 实现**连续速度的高精度回归解码**（R²=0.78） |
| **模型架构** | 线性模型为主 | 使用**深度 RNN 模型**，显著提升性能，尤其适合复杂时空模式 |
| **泛化能力探索** | 少有跨会话/个体测试 | 系统评估了**zero-shot 与 transfer learning 性能**，指导实际部署 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **来源**：先前实验收集的 14 只雄性大鼠（head-fixed rats）
- **信号类型**：
  - **EEG**：32通道 skull-surface EEG，覆盖前额叶至视觉皮层（medial prefrontal, somatomotor, motor, visual cortex）
  - **行为信号**：非电机驱动 treadmill 的角位移 → 计算得到连续速度
- **数据规模**：
  - 共 225 个有效会话（来自原始 276 会话，筛选标准见下文）
  - 总计超过 **133 小时**的同步 EEG 与 treadmill 速度数据
  - 下采样后为 100 Hz，共约 **4800 万对样本**

#### ✅ 包含标准（Inclusion Criteria）
1. 所有 32 个 EEG 通道功能正常；
2. 跑步机速度具有足够变异性（IQR > 0.46，排除低活动会话）

---

### ⚙️ 实验设置
- **输入格式**：
  - 滑动窗口长度：**200 ms**（即 20 时间步 × 32 通道）
  - 步长（stride）：**10 ms**，实现近似实时输出
  - 输入标准化：按通道 z-score（使用训练集参数）
- **输出目标**：当前时刻 treadmill speed（连续值，单位 a.u.）
- **训练策略**：
  - 单会话内顺序划分：前 80% 训练，中间 10% 验证，最后 10% 测试
  - 使用 **hold-out validation + early stopping** 防止过拟合
  - 损失函数：Mean Squared Error (MSE)

---

### 📈 评估指标
- 主要指标：
  - **Pearson 相关系数（r）**
  - **决定系数（R²）**
- 统计检验：
  - Shapiro-Wilk 检验正态性
  - Friedman test（多组比较）
  - Wilcoxon signed-rank test + Bonferroni 校正（成对比较）

---

### 🔤 基线方法对比
共比较五种模型：
| 模型 | 类型 | 特点 |
|------|------|------|
| **Linear Regression** | 线性模型 | 基础基准，仅学习线性权重 |
| **Random Forest** | 集成树模型 | 非线性但不显式建模时间依赖 |
| **Feed-Forward Neural Network (FFNN)** | 深度前馈网络 | 展平输入，忽略时间结构 |
| **RNN with LSTM** | 循环神经网络 | 显式建模时间序列动态 |
| **Encoder-Only Transformer** | 自注意力模型 | 捕捉全局上下文，双向注意机制 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据
| 模型 | Median Correlation (r) | Median R² |
|------|------------------------|-----------|
| Linear Regression | 0.64 | 0.39 |
| Random Forest | 0.76 | 0.56 |
| FFNN | 0.87 | 0.76 |
| Transformer | 0.88 | 0.77 |
| **LSTM-RNN（最优）** | **0.88** | **0.78** |

> 💡 **说明**：LSTM 在所有模型中表现最佳，且显著优于其他模型（p < 0.0001）。  
> 在 **82.7% 的会话中，相关性超过此前文献报道的最高值 0.80**；5% 会话达到 R² > 0.90。

---

### 🔁 迁移学习与泛化能力（Transfer Learning）

| 训练策略 | r | R² | 说明 |
|--------|----|-----|------|
| **Single-session (scratch-80%)** | 0.88 | 0.78 | 最佳情况（全量训练） |
| **Cross-session zero-shot** | 0.61 | 0.25 | 同一只动物的不同会话，未微调 |
| **Cross-subject zero-shot** | 0.24 | -0.39 | 不同动物间，性能极差 |
| **Cross-session fine-tune (10%)** | **0.79** | **0.58** | 显著优于 scratch-10% |
| **Cross-subject fine-tune (10%)** | 0.73 | 0.49 | 与 scratch-10% 相当，增益有限 |

> ✅ **结论**：  
> - 动物内部存在**稳定的神经签名（neural signature）可跨会话泛化**  
> - 但**无法跨动物零样本迁移**，表明 signature 具有高度 subject-specific 特征  
> - **fine-tuning 可快速适配新会话**，尤其在同动物场景下效果显著

---

### 🧠 消融实验结果（Ablation Studies）

#### （1）空间贡献分析（Spatial Contribution）
- **视觉皮层（visual cortex）电极单独使用时性能最强**：
  - r = 0.85, R² = 0.72（接近全脑 0.88 / 0.78）
- 最优组合：**motor + visual** 或 **somatomotor + visual**（r ≈ 0.88）
- 排除视觉皮层后性能明显下降（平均 r ≈ 0.84）

> ❗ 发现：尽管传统关注 motor cortex，但 **visual cortex 是最主要的解码信息源**

#### （2）频谱贡献分析（Spectral Contribution）
- **低频段主导解码性能**：
  - **Delta (1–4 Hz)**：r = 0.88, R² = 0.76
  - **Theta (4–8 Hz)**：r = 0.85, R² = 0.72
- 高频段性能显著降低：
  - Beta (13–30 Hz)：r = 0.77, R² = 0.57
  - Gamma (>30 Hz)：r = 0.61, R² = 0.35
- **仅用 delta + theta 几乎可复现全频带性能**

> 🔍 支持证据：PSD 分析显示，随着速度增加，**低频功率增强，尤其在 ~8 Hz 出现峰值**

#### （3）时间前瞻性与回顾性解码
- 测试模型预测 **±1000 ms 内的过去与未来速度**
- 对比基线：
  - **Autocorrelation of speed**
  - **Speed-only RNN（无 EEG 输入）**

| 方向 | 时间偏移 | EEG-based r | Speed-based r | 结论 |
|------|---------|-------------|---------------|------|
| 回顾（Past） | -1000 ms | **0.59** | 0.20 | EEG 更优，衰减慢 |
| 预测（Future） | +1000 ms | **0.34** | 0.15 | EEG 仍更优，但衰减更快 |

> 🎯 **关键发现**：
> - EEG 编码了超出运动本身自相关的信息
> - **对过去速度的重建能力强于对未来速度的预测** → 表明神经信号保留“痕迹”，但前瞻性规划较弱
> - 存在**时间不对称性**：past decoding > future decoding

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **非侵入式 EEG 可以高精度连续解码自选运动速度**：
   - 达到 **r = 0.88, R² = 0.78**，超越多数侵入式研究
2. **LSTM-RNN 是最适合该任务的模型架构**：
   - 显著优于线性与传统机器学习方法
   - 即使 FFNN 和 Transformer 表现良好，RNN 仍略胜一筹
3. **视觉皮层是解码速度的关键区域**：
   - 超越传统 motor cortex，可能与 locomotion-induced brain state modulation 有关
4. **低频振荡（<8 Hz）是主要信息载体**：
   - Delta 与 theta band 提供绝大部分可解码信息
5. **神经活动编码了动作的过去与未来状态**：
   - 支持大脑对行为的“时间嵌入”（temporal embedding）
   - 回顾性信息强于前瞻性信息
6. **个体内神经签名稳定，可迁移；个体间不可直接迁移**：
   - 支持使用 pre-trained model + fine-tuning 加速 BCI 部署

---

### ⚠️ 方法的局限性
1. **动物需头固定（head-fixed）**：
   - 限制了自由移动行为，影响生态效度
2. **EEG 空间分辨率有限**：
   - skull-surface EEG 信噪比较低，难以精确定位深层源
3. **解码输出存在漂移与噪声**：
   - 如静止期仍有非零速度输出，需后处理（如阈值截断）
4. **未实现实时闭环控制**：
   - 当前为 offline 仿真，尚未集成到 real-time BCI 控制 loop 中
5. **任务背景包含 Go/NoGo 视觉辨别**：
   - 虽然强调 self-paced，但仍受任务结构影响，可能引入视觉诱发成分混淆

---

### 🔮 未来工作方向
1. **开发 real-time 在线 BCI 系统**：
   - 集成滤波、延迟补偿、安全锁机制
2. **结合多模态信号融合**：
   - 如加入 EMG、加速度计等辅助信号提升鲁棒性
3. **探索因果建模与预测控制策略**：
   - 利用 future state decoding 实现 exoskeleton 的主动调节
4. **推动跨物种泛化研究**：
   - 探索是否可在人类 EEG 中找到类似 signature
5. **优化迁移学习框架**：
   - 开发通用初始化模型（universal pre-trained decoder）降低个体校准成本
6. **探索神经机制解释**：
   - 为何 visual cortex 对 locomotion 编码如此重要？是否反映 multisensory integration 或 arousal 调控？

---

> 📌 **总体评价**：  
> 本研究是**非侵入式 BCI 在连续运动解码领域的一项重要进展**。它不仅展示了 EEG 的高解码潜力，还揭示了 cortex-wide、low-frequency、subject-specific 的神经编码特性，为下一代自然交互式神经假肢与康复设备提供了理论基础和技术路径。

</details>

---

### 7. [Why ReLU? A Bit-Model Dichotomy for Deep Network Training](https://arxiv.org/abs/2602.19017)

**Authors**: Ilan Doron-Arad, Elchanan Mossel  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.19017v1  

#### Abstract
Theoretical analyses of Empirical Risk Minimization (ERM) are standardly framed within the Real-RAM model of computation. In this setting, training even simple neural networks is known to be $\exists \mathbb{R}$-complete -- a complexity class believed to be harder than NP, that characterizes the dif...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Why ReLU? A Bit-Model Dichotomy for Deep Network Training 论文总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文旨在解决**理论深度学习复杂性分析与实际数字计算现实之间长期存在的脱节问题**。传统理论通常基于 Real-RAM 模型，允许对实数进行无限精度运算，这导致训练神经网络被证明是 $\exists\mathbb{R}$-complete（比 NP 更难）。然而，在实践中，所有计算都在有限精度的硬件上执行。这种理论与实践的鸿沟使得许多理论上的“困难”结果在现实中可能并不成立。

本文的核心问题是：**当我们将注意力限制在具有多项式长度二进制编码的有理数权重时，这些关于训练难度的理论结果是否依然成立？**

### 提出了什么新方法或新思路
论文提出了一个名为 **ERMbit** 的新理论框架，这是一个更符合现实的 **bit-level 模型**下的经验风险最小化（ERM）模型。

- **ERMbit 模型**：在此模型中，网络参数、输入和输出都被约束为有理数，并且其二进制编码长度是多项式有界的。这直接反映了现代计算机使用浮点数（如 IEEE 754）进行有限精度计算的本质。
- **激活函数的复杂性二分法**：论文的核心创新在于揭示了一个由激活函数决定的**尖锐复杂性二分法（sharp dichotomy）**。这个二分法在传统的 Real-RAM 模型下无法被观察到。

### 相比现有方法的优势
相比传统的 Real-RAM 分析，本研究的优势在于：
- **更强的现实相关性**：ERMbit 模型直接建模了数字计算的有限精度本质，使理论分析更贴近实际的深度学习系统。
- **揭示了新的理论洞见**：发现了激活函数的选择（多项式 vs. 分段线性）是决定训练算法复杂性的根本因素，这为理解为何 ReLU 等激活函数在实践中如此成功提供了全新的复杂性理论视角。
- **解释了“梯度爆炸/消失”的本质**：从计算复杂性的角度，将梯度问题归因于非良性（non-benign）的数值溢出/下溢，而非简单的数值大小问题。

## 2. 核心实验方法和设置

需要特别指出的是，**这是一篇以理论证明为核心的理论计算机科学论文，其“实验”主要是数学构造和复杂性理论中的规约（reduction），而非在真实数据集上的机器学习实验**。

### 方法论
论文通过构建复杂的数学规约来证明其理论结论，主要依赖于以下工具：
- **Straight-Line Programs (SLP)**：一种用于表示算术电路的模型，其输出可以是一个非常大的整数，但其计算过程可以用短程序描述。
- **PosSLP 和 BitSLP 问题**：分别指判断 SLP 输出符号和特定比特位的问题，这些问题已被证明具有很高的计算复杂性（#P-hard）。

### 实验设置（理论构造）
- **目标**：将已知的硬问题（如 BitSLP, PosSLP）规约到深度神经网络的 ERMbit 或 Backpropagation 问题。
- **网络架构**：构造了特定的前馈神经网络，其深度与 SLP 的长度成正比。
- **激活函数**：
  - 对于**负向结果**，使用任意次数 ≥2 的非线性多项式激活函数 $ \sigma \in \mathbb{Q}[T] $（如 $x^2$）。
  - 对于**正向结果**，使用标准的分段线性激活函数（如 ReLU）。
- **损失函数**：
  - 为了证明 #P-hardness，使用了一个**非标准的、高效的比特提取损失函数**，可以直接从网络输出中提取某个比特位。
  - 同时也讨论了使用标准的 hinge loss 能得到稍弱的下界。
- **评估指标**：理论复杂性类（如 #P-hard, NP-complete, BPP）。

### 基线方法对比
论文的“基线”是传统的 **Real-RAM 模型下的 ERM 分析**。通过对比两种模型下的复杂性结果，突显了新模型的价值。

## 3. 主要实验结果和性能指标

### 关键理论结果
论文通过严格的数学证明得出了以下核心定理：

#### 对于多项式激活函数（如 $x^2$）
- **Theorem 1.1**: 决定 ERMbit 是 **#P-hard** 的。这意味着它被认为严格难于 NP-complete 问题。
- **Theorem 1.3**: 即使只是确定单个梯度坐标的**符号**（BACKPROP-SIGN），也是**不可能在 BPP 类中**（即不存在有界误差概率多项式时间算法），假设 $ \text{NP} \not\subseteq \text{BPP} $。而确定梯度中某个特定位的值更是 #P-hard。
- **推论**：即使在网络各层的中间值被常数限制的情况下，这些硬度结果依然成立。

#### 对于分段线性激活函数（如 ReLU）
- **Theorem 1.4**: ERMbit 是 **NP-complete** 的。更重要的是，验证一个解和计算精确梯度（一次反向传播）都可以在**多项式时间内完成**。

### 与基线方法的对比结果
| 设置 | Real-valued ERM | ERMbit |
| :--- | :--- | :--- |
| **浅层网络** | $\exists\mathbb{R}$-complete (NP-membership unknown) | **NP-complete** |
| **深层网络 (多项式激活)** | $\exists\mathbb{R}$-complete | **#P-hard (believably outside NP)** |
| **深层网络 (分段线性激活)** | $\exists\mathbb{R}$-complete | **NP-complete** |
| **反向传播符号 (多项式激活)** | 在 P 中 | **不在 BPP 中 (条件性)** |
| **反向传播 (分段线性激活)** | 在 P 中 | **在 P 中** |

### 消融实验
论文没有传统意义上的消融实验，但其理论分析包含了关键变量的影响：
- **深度**：硬度结果依赖于网络的深度，因为深度允许模拟长链的 SLP 计算。
- **激活函数类型**：这是区分复杂性的唯一关键因素。只要是非线性多项式（degree ≥ 2），无论具体形式如何，都导致 #P-hardness。
- **数值范围**：证明了即使中间值被限制在 [-1,1] 内，硬度依然存在，排除了单纯数值大小导致困难的可能性。

## 4. 关键结论和发现

### 主要发现
1.  **有限精度是可学习性的根本决定因素**：论文最核心的结论是，**有限精度约束不是实现细节，而是可学习性的基本决定因素**。在 ERMbit 模型下，训练的复杂性发生了根本性变化。
2.  **激活函数的复杂性二分法**：存在一个由激活函数驱动的尖锐二分法：
    - **多项式激活函数**（如 $x^2$）会引入灾难性的计算复杂性，使得训练和梯度计算在理论上是不可行的（#P-hard）。
    - **分段线性激活函数**（如 ReLU）则保持了良好的复杂性，使得 ERMbit 验证和反向传播可以在多项式时间内完成。
3.  **ReLU 成功的理论解释**：这一发现为现代深度学习广泛采用 ReLU 及其变体（如 Leaky-ReLU）提供了一个深刻的复杂性理论解释：它们避免了由高次多项式激活带来的固有计算困难。
4.  **梯度爆炸/消失的新视角**：论文提出，对于多项式激活，梯度问题不仅仅是数值上的“爆炸”或“消失”，更是一种“**非良性的溢出/下溢**”（non-benign overflow/underflow）。在这种情况下，中间量的二进制模式编码了难以预测的硬算术问题，因此仅仅通过范数裁剪（norm clipping）等常规方法不足以解决问题。

### 方法的局限性
- **最坏情况分析**：论文的结果是基于最坏情况（worst-case）的复杂性理论，证明了存在一些极其困难的实例。这并不意味着所有使用多项式激活的实际训练任务都是不可行的。
- **理想化模型**：尽管 ERMbit 比 Real-RAM 更现实，但它仍然抽象掉了许多工程细节，如具体的浮点舍入模式、NaN、无穷大等。
- **平滑激活函数**：论文没有直接分析 Sigmoid、Tanh 等平滑激活函数。不过，作者在讨论中指出，这些函数在实践中是通过有限精度的查表或多项式逼近实现的，因此可以被视为一种“位有界”（bit-bounded）的激活函数，其行为可能类似于分段线性函数。

### 未来工作方向
- **扩展到其他激活函数**：将此二分法框架应用于分析其他流行的非线性激活函数，如 Swish、GELU 等。
- **量化训练的理论分析**：将 ERMbit 框架进一步发展，为量化感知训练（Quantization-Aware Training, QAT）提供更坚实的理论基础。
- **连接泛化能力**：探索这种计算复杂性二分法是否与模型的泛化能力（generalization）有关联。
- **设计新的激活函数**：基于此理论，指导设计既能保持表达能力又具有良好计算复杂性的新型激活函数。

</details>

---

### 8. [LLMs Can Learn to Reason Via Off-Policy RL](https://arxiv.org/abs/2602.19362)

**Authors**: Daniel Ritter, Owen Oertell, Bradley Guo, Jonathan Chang, Kiant\'e Brantley, Wen Sun  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.19362v1  

#### Abstract
Reinforcement learning (RL) approaches for Large Language Models (LLMs) frequently use on-policy algorithms, such as PPO or GRPO. However, policy lag from distributed training architectures and differences between the training and inference policies break this assumption, making the data off-policy ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLMs Can Learn to Reason Via Off-Policy RL

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大语言模型（LLMs）在推理任务中通过强化学习（RL）进行后训练（post-training）时，面临一个核心挑战：**训练策略（trainer）与推理引擎（inference engine）之间的不一致**。这种不一致性导致生成的数据本质上是 **off-policy** 的，破坏了传统 on-policy RL 方法（如 PPO、GRPO）的基本假设。

尽管已有工作尝试通过重要性采样（importance sampling, IS）或修改推理引擎来“修复”这一问题，但这些方法存在以下缺陷：
- IS 引入额外方差，影响稳定性；
- 修改推理引擎会降低效率且无法完全消除差异；
- 在异步训练中，策略滞后（policy lag）可能高达数百步，加剧 off-policy 程度。

### 🚀 提出的新方法：OAPL
本文提出了一种全新的、**原生支持 off-policy 学习**的算法：

> **Optimal Advantage-based Policy Optimization with Lagged Inference policy (OAPL)**

其核心思想是：
- 将 off-policy 训练建模为一个 **KL 正则化 RL 问题**，其中目标是最小化当前策略 $\pi$ 与推理策略 $T_{\text{vLLM}}$ 之间的 KL 散度，同时最大化奖励。
- 利用该问题的闭式解（closed-form solution），推导出一个基于 **最优优势函数 $A^*$** 的平方回归损失函数：
  $$
  \min_\pi \mathbb{E}_{y \sim T_{\text{vLLM}}} \left[ \left( \frac{1}{\beta} \log \frac{\pi(y|x)}{T_{\text{vLLM}}(y|x)} - (r(x,y) - V^*(x)) \right)^2 \right]
  $$
- 该损失直接使用来自滞后推理策略的数据进行优化，无需任何重要性权重或裁剪操作。

### 🔍 相比现有方法的优势
| 特性 | GRPO + IS | OAPL |
|------|----------|-------|
| 是否需要 IS | 是（引入高方差） | 否 |
| 是否依赖 on-policy 假设 | 是 | 否 |
| 是否需修改推理引擎 | 部分工作需要 | 不需要 |
| 支持异步程度 | 最多 1~几 step | 可达 **400+ gradient steps** |
| 算法复杂度 | 复杂（ratio clipping, token deletion） | 简单（least-squares regression） |
| 样本效率 | 较低 | **提升约 3x** |
| 测试时扩展性（Pass@k） | 易熵坍缩 | 更好 scaling |

> ✅ **OAPL 完全拥抱 off-policy 性质，而非试图掩盖它**，从而实现更稳定、高效、可扩展的 RL 后训练。

---

## 2. 核心实验方法和设置

### 📚 数据集
#### 数学推理任务
- **训练数据**：`Deepscaler`（大规模竞赛级数学题）
- **测试基准**：
  - `AIME 2025`
  - `HMMT 2025`（February & November）
  - `BRUMO 2025`

#### 代码生成任务
- **训练流程**：两阶段离线训练
  1. 第一阶段：从 `DeepCoder` 的训练集生成 8 回复/提示 的离线数据集
  2. 第二阶段：从第一阶段模型抽取 4000 prompts 再次生成数据，继续训练 4 轮
- **评估基准**：`LiveCodeBench v5`（279 个无污染编程题）

### ⚙️ 实验设置
| 设置项 | 描述 |
|--------|------|
| 基础模型（数学） | `Qwen3-4B-Thinking-2507` |
| 基础模型（代码） | `DeepSeek-R1-Distill-Qwen-14B` |
| 最大生成长度 | 数学：16K；代码训练：32K；评估：64K |
| 并行训练 | 异步架构，trainer 与 vLLM 推理引擎分离 |
| OAPL 同步间隔 L | 数学：50 步；代码：≈400 步（每 epoch 一次） |
| GRPO 异步设置 | “off-by-one” 模式（最多滞后 1 步） |

### 📊 评估指标
- **Pass@k**：k 次独立 rollout 中至少有一次正确的概率
  - 使用 Chen et al. (2021) 的无偏估计器
  - 对于数学任务：每个 prompt 采样 10 次
  - 对于代码任务：每个 prompt 采样 20 次
- **训练动态监控**：
  - 序列熵（Sequence Entropy）
  - Pass@1 / Pass@5 / Pass@10 曲线
  - 收敛速度与稳定性

### 🆚 基线方法
- **GRPO + Importance Sampling (IS)**：主流 on-policy 方法，结合 token-level 或 sequence-level IS 来缓解 off-policy 问题
- **DeepCoder**：公开可用的 GRPO 训练代码模型，作为外部强基线

---

## 3. 主要实验结果和性能指标

### 📈 数学推理性能（图1、图2、图4）

| 模型 | Pass@1（平均） | Pass@5 | Pass@10 | 稳定性 |
|------|----------------|--------|---------|--------|
| **OAPL** | **~0.70** | **~0.75** | **~0.80** | ✅ 高 |
| GRPO + IS | ~0.60 | ~0.65 | ~0.70 | ❌ 出现震荡 |

- **OAPL 在所有三个数学基准上全面超越 GRPO**，尤其在 Pass@5 和 Pass@10 上优势明显。
- 图2显示 OAPL 收敛更快、更平稳，而 GRPO 在后期出现波动。

### 🔁 训练稳定性与熵行为（图3）
- **GRPO 出现明显的熵坍缩（entropy collapse）** → 表示策略变得过于确定，损害多样性与泛化能力。
- **OAPL 维持较高且稳定的熵水平** → 表明策略保持探索性，有助于更好的 test-time scaling。
- 即使将 OAPL 的同步间隔拉长至 **L=100**，仍能稳定学习，验证其对极端 off-policy 的鲁棒性。

### 💻 代码生成性能（图5）

| 模型 | Pass@k scaling | 样本量 | 相对效率 |
|------|----------------|--------|-----------|
| **OAPL** | 匹配甚至略优于 DeepCoder | **~200K** | ✅ **3x 更高效** |
| DeepCoder (GRPO) | 略低或相当 | ~650K | 基准 |

- **OAPL 仅用 1/3 的训练样本即达到同等甚至更优性能**。
- 所有模型（包括 base model）均表现出 RL 训练提升了大 k 下的 Pass@k，反驳了“RL 只是分布锐化”的观点。
- OAPL 的 scaling 曲线优于 GRPO，在 k=256 时差距显著。

### 🔍 消融实验（隐含分析）
虽然未明确列出消融表，但文中通过设计对比揭示关键因素：
- **KL 正则对象的选择**：OAPL 使用 $T_{\text{vLLM}}$ 而非固定 reference policy，允许动态更新，避免偏差积累。
- **是否使用 IS**：OAPL 完全避免 IS，证明其非必要。
- **同步频率的影响**：即使 L=400，OAPL 依然有效，说明其对 policy lag 具有极强容忍力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **On-policy 并非必需**：  
   LLM 的 RL 后训练可以且应该拥抱 off-policy 本质，而不是强行“伪装成 on-policy”。

2. **OAPL 更高效、更稳定**：  
   - 无需 IS、clip、token 删除等复杂机制；
   - 支持高度异步训练（>400 steps lag），远超现有方法（通常 <10 steps）；
   - 实现 **3倍以上的样本效率提升**。

3. **改善 test-time scaling**：  
   OAPL 训练出的模型在 **Pass@k（k=1~256）上有更好扩展性**，归因于更高的输出熵和更强的探索能力。

4. **KL 正则 + 价值估计 是关键**：  
   利用 KL-regularized RL 的闭式解构造回归目标，是一种简洁而强大的 off-policy 学习范式。

### ⚠️ 局限性
- 当前 OAPL 假设推理策略 $T_{\text{vLLM}}$ 至少有一定概率覆盖正确答案（否则 $V^*$ 估计失效）。
- 依赖 group-wise rollout 来估计 $V^*(x)$，要求每个问题生成多个样本（G≥5~8）。
- 尚未在人类反馈 RLHF 场景中验证，主要聚焦于程序执行反馈（code/math）。

### 🔮 未来工作方向
- 将 OAPL 扩展到 **multi-turn RLHF** 和 **value function learning**。
- 结合 **offline data**（如人类标注轨迹）进一步提高样本利用率。
- 探索 **更高效的 $V^*$ 估计方法**，减少对大量 rollout 的依赖。
- 研究如何将 OAPL 与其他 alignment 技术（如 safety constraints）结合。

---

> **一句话总结**：  
> 本文颠覆了“RL 必须 on-policy”的固有认知，提出 OAPL —— 一种简单、高效、天然支持 off-policy 的 LLM 推理训练算法，在数学与代码任务上实现了更高性能、更强稳定性与显著提升的样本效率。

</details>

---

### 9. [LAMMI-Pathology: A Tool-Centric Bottom-Up LVLM-Agent Framework for Molecularly Informed Medical Intelligence in Pathology](https://arxiv.org/abs/2602.18773)

**Authors**: Haoyang Su, Shaoting Zhang, Xiaosong Wang  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.18773v1  

#### Abstract
The emergence of tool-calling-based agent systems introduces a more evidence-driven paradigm for pathology image analysis in contrast to the coarse-grained text-image diagnostic approaches. With the recent large-scale experimental adoption of spatial transcriptomics technologies, molecularly validat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LAMMI-Pathology 论文核心总结

## 1. 主要贡献和创新点

### 解决的问题
该论文针对当前病理学图像分析中存在的两个关键问题：
1.  **文本中心化偏差**：现有研究过度依赖文本描述进行诊断，而忽略了病理学以视觉为中心的本质，导致对原始形态学证据的关注不足。
2.  **缺乏证据驱动的推理**：尽管有 CoT 和 RAG 等方法尝试捕捉中间推理步骤，但它们仍依赖于文本中介，未能有效整合临床实践中广泛使用的分子证据（如 IHC、RNA-seq）。

### 提出的新方法与创新
论文提出了 **LAMMI-Pathology**，一个面向病理学领域的工具调用型智能体框架，其核心创新在于：

- **工具中心、自下而上的架构 (Tool-Centric Bottom-Up Architecture)**：
  - 与传统的多智能体系统不同，LAMMI 首先设计并定制领域专用的工具（如基因查询、通路分析等），然后将这些工具按功能聚类形成 **Component Agent**（组件智能体）。
  - 这些组件智能体再由一个基于 **LVLM** 的顶层 **Planner**（规划器）进行分层协调，避免了因管理过长上下文而导致的任务漂移。

- **原子执行节点 (Atomic Execution Nodes, AENs)**：
  - 提出了一种新的轨迹构建机制，AEN 是智能体与外部工具交互的最小可验证单元，定义为三元组 `(Query, Action, Observation)`。
  - AENs 将真实的工具返回结果与 LLM 的推理相结合，生成半模拟的推理轨迹，确保了训练数据的可靠性和可组合性。

- **轨迹感知微调策略 (Trajectory-aware Fine-tuning)**：
  - 设计了一种名为 **Trajectory-aware Adapter (TA)** 的轻量级适配器，通过在 FFN 层动态注入模块，学习区分 `Thought`, `Action`, `Action Input` 等不同轨迹段。
  - 该策略使 Planner 能够更好地对齐多步推理轨迹，从而提升其决策鲁棒性。

### 相比现有方法的优势
- **更高的证据可信度**：通过 AENs 直接整合真实工具输出，实现了真正意义上的证据驱动推理。
- **更强的可扩展性**：自下而上的架构允许本地集成新工具，无需依赖外部 API，降低了系统复杂性。
- **更优的性能与效率**：实验证明，在多个基准上超越了主流方法，同时内存消耗更低。

---

## 2. 核心实验方法和设置

### 数据集
- **ST-Traj**：从空间转录组学文献中提取的 10,684 个成功执行的 AEN 构建而成，最终获得 6,818 条高质量的元轨迹（meta-trajectories），用于训练和评估多步工具调用行为。
- **PathSpatial-DocQA**：一个基于 HEST 和 STimage-1K4M 数据集构建的问答数据集，包含前沿的病理-分子诊断问题。
- **PathMMU**：一个公开的临床来源的病理图像理解 QA 数据集，作为金标准基准。

### 实验设置
- **硬件**：微调使用 4 块 H200 GPU，推理使用 4 块 RTX 4090 GPU。
- **超参数**：最大迭代次数为 8，执行超时为 300 秒，生成长度上限为 2048 tokens。
- **随机种子**：所有实验固定随机种子为 37。

### 评估指标
- **工具冗余率 (Tool Redundancy Rate, TRR)**：衡量重复调用相似输入工具的比例。
- **轨迹成功分数 (Trajectory Success Score, TSS)**：综合评估输出有效性和工具调用成功率。
- **工具一致性 F1 分数 (Tool Consistency F1, TCF1)**：衡量模型调用工具与真实情况的一致性。
- **答案一致性得分 (Answer Consistency Score, ACS)**：基于 LLM 评估的答案语义一致性。
- **幻觉率 (Hallucination Rate, HR)**：检测模型响应中的幻觉现象。
- **F1 分数**：用于闭合式问题的准确率评估。

### 基线方法对比
- **OpenAI-Agents-SDK**：商业级代理 SDK。
- **ReACT**：经典的“思考-行动-观察”范式。
- **MAT-Agent**：一种多模态代理调优方法。
- **MLLM-Tools**：一个多模态大语言模型的工具学习框架。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比

#### 在 PathSpatial-DocQA 数据集上的表现 (Tab. 1)
| Framework | ACS ↑ | HR ↓ | TRR ↓ |
| :--- | :--- | :--- | :--- |
| **LAMMI (Ours)** | **0.809** | 0.338 | **0.000** |
| OpenAI-Agents-SDK | 0.739 | 0.164 | 0.033 |
| ReACT | 0.715 | 0.353 | 0.018 |
| MAT-Agent | 0.462 | 0.662 | 0.049 |
| MLLM-Tools | 0.448 | 0.540 | 0.374 |

> **结论**：LAMMI 在 ACS 上显著优于所有基线，且 TRR 为 0，表明其工具调用高度精准，无冗余。

#### 在 ST-Traj 数据集上的表现 (Tab. 2)
| Framework | TCF1 ↑ | TSS ↑ | ACS ↑ | TRR ↓ |
| :--- | :--- | :--- | :--- | :--- |
| **LAMMI (Ours)** | **0.427** | **0.901** | **0.592** | **0.036** |
| OpenAI-Agents-SDK | 0.370 | 0.861 | 0.492 | 0.025 |
| ReACT | 0.377 | 0.792 | 0.587 | 0.092 |
| MAT-Agent | 0.085 | 0.699 | 0.480 | 0.081 |
| MLLM-Tools | 0.436 | 0.423 | 0.482 | 0.355 |

> **结论**：LAMMI 在 TSS 和 TCF1 上均取得最佳成绩，证明其在复杂多步任务中具有最强的推理一致性和成功率。

#### 在 PathMMU 数据集上的表现 (Tab. 3)
| Framework | ACS ↑ | F1 ↑ |
| :--- | :--- | :--- |
| **LAMMI (Ours)** | **0.582** | **0.503** |
| OpenAI-Agents-SDK | 0.709 | 0.719 |
| ReACT | 0.572 | 0.518 |
| MAT-Agent | 0.469 | 0.395 |
| MLLM-Tools | 0.571 | 0.275 |

> **结论**：虽然 OpenAI-Agents-SDK 综合表现最好，但 LAMMI 在开源框架中表现最优，尤其在 F1 分数上远超其他开源方法。

### 消融实验结果 (Tab. 4)

在 PathSpatial-DocQA 上比较不同微调策略：
- **TA+PE**（本文方法）在不同新工具引入比例 (NITR) 下均保持稳定甚至提升的 TSS，且 TRR 最低。
- **Full+PE** 和 **LoRA+PE** 在 NITR 增加时 TSS 显著下降，表明存在过拟合风险。

> **结论**：TA 微调策略具有强大的泛化能力，能有效适应新工具。

---

## 4. 关键结论和发现

### 主要发现
1. **LAMMI-Pathology 成功桥接了形态学解释与分子验证**，通过工具中心的架构实现了真正证据驱动的病理诊断。
2. **AEN 和轨迹感知微调是关键技术**，它们共同构建了可靠的半模拟推理轨迹，并提升了 Planner 对复杂任务的协调能力。
3. **自下而上的架构显著提高了效率**：相比传统多智能体系统 (MAS)，LAMMI 的平均 GPU 内存占用仅为前者的 34.7%~77.4%，得益于动态的 LLM 链实例化和共享权重设计。

### 方法的局限性
- **依赖高质量工具**：系统的性能受限于底层工具的准确性和覆盖范围。
- **工具调用成本**：频繁的外部工具调用可能增加推理延迟。
- **通用性有待验证**：目前专注于病理学领域，向其他医学影像领域的迁移需要进一步探索。

### 未来工作方向
- **推广到更广泛的医学影像应用**。
- **整合更多样化的分子数据源**（如蛋白质组学、代谢组学）以增强诊断推理。
- **优化工具调用策略**，减少不必要的调用，提高实时性。

</details>

---

### 10. [High Dimensional Procedural Content Generation](https://arxiv.org/abs/2602.18943)

**Authors**: Kaijie Xu, Clark Verbrugge  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.18943v1  

#### Abstract
Procedural content generation (PCG) has made substantial progress in shaping static 2D/3D geometry, while most methods treat gameplay mechanics as auxiliary and optimize only over space. We argue that this limits controllability and expressivity, and formally introduce High-Dimensional PCG (HDPCG): ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# High Dimensional Procedural Content Generation 论文总结

## 1. 主要贡献和创新点

### 解决的问题
传统的 **Procedural Content Generation (PCG)** 方法大多以几何空间为主导（geometry-first），将游戏机制（如时间依赖移动、重力翻转、平行世界切换等）作为辅助因素通过模拟或后处理来实现。这种做法导致：
- 机制结构难以在生成过程中被**直接控制**；
- 生成结果对机制的**可验证性差**；
- 复杂动态与布局之间耦合度高时，生成质量下降。

本文指出，当前缺乏一种**通用、可扩展的表示框架**，能够统一建模空间与非空间的游戏机制维度。

### 提出的新方法：HDPCG
作者提出了 **High-Dimensional Procedural Content Generation (HDPCG)** ——一个将非几何玩法维度提升为联合状态空间一等公民的框架。

#### 核心思想
- 将传统 2D/3D 空间节点扩展为高维状态：  
  $$
  S = X \times D_1 \times D_2 \times \cdots \times D_k
  $$
  其中 $X$ 是空间网格，$D_i$ 是额外的游戏机制维度（如 layer、time、locomotion mode）。
- 构造 **Dimensional-Expanded Graph (DEG)**，在此图上进行路径搜索与验证。
- 支持多种机制原生集成，例如：
  - **Direction-S (Space)**：支持多层地图切换（gravity inversion, parallel-world switching）
  - **Direction-T (Time)**：支持时间展开图（time-expanded graph）建模动态平台、敌人巡逻等

#### 统一四阶段流水线
1. **Abstract Skeleton Generation**：在低维空间规划抽象路径
2. **Controlled Grounding**：将抽象路径实例化为带属性的高维结构
3. **High-Dimensional Validation**：在 DEG 上运行 A*, BFS 或 DP 验证可达性
4. **Multi-Metric Evaluation & Search**：基于多目标指标优化生成参数（如 GA 进化）

### 相比现有方法的优势
| 方面 | HDPCG 优势 |
|------|------------|
| **可控性 (Controllability)** | 可精确控制机制出现频率、间隔（如每 5 步必须有一次 layer switch） |
| **可验证性 (Verifiability)** | 在生成时即保证可玩性（witness path 存在） |
| **表达力 (Expressivity)** | 支持复杂机制组合（layer + time），超越纯几何生成 |
| **模块化设计** | 流水线适用于不同方向（Space/Time），易于扩展 |

---

## 2. 核心实验方法和设置

### 实验方向
论文在两个代表性方向上实例化 HDPCG：
- **Direction-S (Space)**：基于离散 layer 的 4D $(x,y,z,l)$ 地图生成
- **Direction-T (Time)**：基于 time-expanded graph 的动态平台关卡生成

### 数据集与环境
- **无公开真实数据集**，所有实验均为合成生成任务。
- 使用自定义网格规模进行大规模测试：
  - **Small**: $30^3$
  - **Medium**: $50^3$
  - **Large**: $100^3$

### 评估指标
#### Direction-S（空间方向）
| 指标 | 描述 |
|------|------|
| **Switch Density / Spacing Controllability** | 实际开关密度/最小间距 vs 目标值的 MAE 和合规率 |
| **Robustness** | 微扰后重规划成功率（band/global perturbation） |
| **Alternative Route Robustness (ARR)** | 路径附近是否存在替代路线 |
| **Composite Quality Score** | 加权综合得分：路径长度、均匀性、覆盖率等 |

#### Direction-T（时间方向）
| 指标 | 描述 |
|------|------|
| **Ride Ratio Error (MAE)** | 实际乘坐平台帧数占比 vs 目标 |
| **Inter-event Spacing Success** | 平台上下车事件之间的最小时间间隔是否满足要求 |
| **Beat Uniformity** | 动作节奏稳定性（CV of intervals） |
| **Ticks-to-Goal** | 到达终点所需时间步数 |
| **Near-swap Count** | 是否发生与移动障碍物正面碰撞 |
| **Composite Utility $J_{time}$** | 综合得分（含 ride/wait/coverage/runtime） |

### 基线方法对比
#### Direction-S 对比方法
| 方法 | 描述 |
|------|------|
| **NNB (Naive Noise Baseline)** | 最小引导，仅用噪声场驱动路径生成 |
| **NP-A* (Naive Penalty A*)** | 引入排斥势场避免路径交叉 |
| **PF-A* (Potential Field A*)** | 显式设定 switch 锚点，吸引路径穿过指定位置 |

#### Direction-T 对比方法
| 方法 | 描述 |
|------|------|
| **Static Backbone** | 先生成静态路径，再添加动态元素（plan-then-animate） |
| **TEG-A*** | 在 time-expanded graph 上运行简化版 A* 搜索 |
| **TEG-DP** | 使用前向动态规划求解最优时间一致路径 |

---

## 3. 主要实验结果和性能指标

### Direction-S 实验结果

#### ✅ 控制精度（Controllability）
| 方法 | Switch Spacing MAE (越小越好) |
|------|-------------------------------|
| **PF-A*** | **~0.00**（近乎完美） |
| NP-A* | ~0.09 |
| NNB | ~0.28–0.33 |

> **结论**：PF-A* 在 switch spacing 控制上显著优于其他方法，尤其在大尺度下仍保持稳定。

#### ✅ 鲁棒性（Robustness under Perturbation）
| 方法 | 重规划成功率（Large 规模） |
|------|--------------------------|
| **PF-A* (single)** | **0.217** |
| NP-A* / NNB | ≈0 |
| 所有 GA 模式 | ≈0（因追求紧凑布局牺牲冗余） |

> **发现**：PF-A* 因其规则化的 spacing 设计，天然具备更强局部恢复能力。

#### ⏱️ 效率（Runtime）
| 方法 | Large 规模单次运行时间（秒） |
|------|-----------------------------|
| **NNB** | **3.16 ± 1.59** |
| NP-A* | 3.54 |
| **PF-A*** | 4.12 |

> GA 模式耗时大幅上升（>100s），但质量更高。

#### 📊 综合质量（Weighted Score）
| 方法 | Small/Medium | Large |
|------|--------------|-------|
| **NP-A*** | ✔️ 最优 | ❌ 下降 |
| **PF-A*** | 良好 | ✔️ **最优** |

> **趋势**：PF-A* 更适合大规模复杂场景，具有更好的**可扩展性**。

---

### Direction-T 实验结果

#### ✅ 性能排序一致性
整体性能排序为：
$$
\text{TEG-DP} > \text{TEG-A*} > \text{Static Backbone}
$$

#### 🎯 关键指标表现（Large + GA 模式）
| 方法 | Weighted Score | Runtime (s) |
|------|----------------|-----------|
| **TEG-DP** | **84.04 ± 12.99** | 576.72 |
| TEG-A* | 22.42 ± 4.18 | 16.66 |
| Static | 13.20 ± 2.33 | 30.20 |

> TEG-DP 虽慢，但在所有指标上全面领先，尤其是在 ride ratio 和 coverage 上。

#### 🔍 统计显著性检验（Table 5）
- 除 Small/Single 和 Medium/Single 外，**TEG-DP 显著优于 TEG-A***（Cliff’s δ = -1.000, padj < 0.001）
- TEG-A* 显著优于 Static Backbone（所有设置下均显著）

#### 💡 消融洞察
- **Static Backbone** 缺乏时间感知，常生成无法完成的动作序列；
- **TEG-A*** 快速但难以诱导丰富交互（缺少 temporal shaping）；
- **TEG-DP** 通过 cost field 显式鼓励 ride/wait pattern，实现更高质量的时间编排。

---

## 4. 关键结论和发现

### 主要发现
1. **HDPCG 成功统一了几何与机制生成**  
   通过构建 **Dimensional-Expanded Graph (DEG)**，实现了空间与非空间维度的联合推理，使机制成为“头等公民”。

2. **Direction-S 中 PF-A* 表现最佳**  
   - 在 switch spacing 控制上接近完美；
   - 在 large scale 下综合质量最高；
   - 具备一定鲁棒性（得益于规则化结构）。

3. **Direction-T 中 TEG-DP 是最优选择**  
   - 显著优于静态基线和简化 A*；
   - 能有效建模 tight timing challenges（如跳跃窗口）；
   - 支持 designer-controlled pacing（ride ratio, event spacing）。

4. **Unity 案例研究验证可行性**  
   - 成功复现了类似 *VVVVVV*（重力翻转）、*Dishonored 2*（时间切换）、*Titanfall 2*（因果倒置）等机制；
   - 导出的 plan 可直接转化为可玩 3D 关卡，证明端到端流程可行。

### 局限性
1. **理想 TEG-A* 不可扩展**  
   若引入 interaction memory（记录已使用的平台），会导致状态爆炸，实际不可行。

2. **尚未整合 PCGML 或 RL**  
   当前为构造式/搜索式方法，未学习先验或价值函数。

3. **多机制组合未测试**  
   当前只分别验证 Space 和 Time，未尝试 layer + time 联合生成。

4. **鲁棒性未纳入 GA 适应度函数**  
   因多次 replanning 开销过大，目前 GA 倾向于生成紧凑但脆弱的布局。

5. **缺乏用户研究**  
   “fun” 和体验质量仍依赖代理指标，未经过人类玩家验证。

### 未来工作方向
1. **开发多层级规划器**  
   结合 bounded-suboptimal search、event-window unrolling 和强剪枝，使 richer A* 可用于交互式生成。

2. **扩展至其他离散化形式**  
   HDPCG 不依赖具体网格，可推广至 navigation mesh、voxel、graph-based world。

3. **构建 PCGML/RL 训练数据集**  
   利用 HDPCG 生成大量高质量、机制丰富的关卡，训练生成模型。

4. **实现多轴联合生成**  
   将 layer、time、locomotion mode 同时编码进 DEG，支持复杂机制耦合。

5. **加入显式鲁棒性目标**  
   在 GA 中引入 multi-objective optimization，平衡 quality 与 robustness，报告 Pareto front。

6. **开展用户研究**  
   通过 controlled user study 验证 metrics 与 perceived pacing/challenge/fun 的相关性。

---

> **总结一句话**：  
> HDPCG 提供了一个**通用、可验证、可控**的 PCG 新范式，推动程序化关卡生成从“几何优先”迈向“机制优先”，为下一代智能关卡设计奠定基础。

</details>

---

### 11. [Spectral Phase Encoding for Quantum Kernel Methods](https://arxiv.org/abs/2602.19644)

**Authors**: Pablo Herrero G\'omez, Antonio Jimeno Morenilla, David Mu\~noz-Hern\'andez, Higinio Mora Mora  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.19644v1  

#### Abstract
Quantum kernel methods are promising for near-term quantum ma- chine learning, yet their behavior under data corruption remains insuf- ficiently understood. We analyze how quantum feature constructions degrade under controlled additive noise. We introduce Spectral Phase Encoding (SPE), a hybrid cons...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Spectral Phase Encoding for Quantum Kernel Methods**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文聚焦于 **NISQ（Noisy Intermediate-Scale Quantum）时代量子核方法（Quantum Kernel Methods）在数据受到噪声污染时的鲁棒性不足** 问题。尽管量子核方法在理论上具有潜力，但其性能对数据编码方式极为敏感，尤其在存在噪声的情况下容易出现“核浓度”（kernel concentration）现象，导致模型失效。

现有研究多关注峰值准确率，而忽视了在真实场景中不可避免的数据扰动下的稳定性。本文提出应从“鲁棒性优先”的视角重新评估量子核方法的设计。

---

### **提出的新方法：Spectral Phase Encoding (SPE)**

作者提出了 **Spectral Phase Encoding (SPE)** ——一种结合经典频域预处理与量子相位编码的混合构造方法：

- **前端**：采用 **Discrete Fourier Transform (DFT)** 对输入数据进行频域变换，提取低频谱系数。
- **后端**：将谱系数的**相位信息**映射为对角量子门（diagonal quantum gates）的参数，嵌入到量子态中。
- 最终通过 **SWAP Test** 估计量子态之间的重叠度，构建 **QK-DFT** 核矩阵。

> 在统一框架下，该方法被命名为 **QK-DFT**，以强调其“经典前端 + 量子后端”的分离结构。

---

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **结构对齐性** | DFT 能揭示数据中的全局周期性和相关性，与对角量子门天然兼容（因对角门擅长表达相对相位）。 |
| **硬件友好性** | 对角门可在超导平台用“虚拟Z门”实现，几乎无延迟且高保真；电路深度浅，适合 NISQ 设备。 |
| **复数特征兼容性** | 直接利用 DFT 输出的复数相位，避免传统编码（如 amplitude encoding）需丢弃相位或额外近似的问题。 |
| **鲁棒性更强** | 实验证明，在噪声增加时，QK-DFT 的性能退化最慢，优于其他量子变体及部分经典基线。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共使用 **20 个异构的真实世界图像类数据集**，涵盖多种视觉模态：
- 手写数字（Digits, MNIST系列）
- 自然图像（CIFAR-10, STL-10）
- 医学影像（BreastMNIST, OCTMNIST, PneumoniaMNIST）
- 面部表情识别（FER2013）
- 街景数字（SVHN）
- 材质纹理（DTD）

所有数据均统一处理为：
- 灰度图
- 分辨率调整至 `32×32`
- 归一化到 `[0,1]`
- 每类平衡采样 `N=150` 示例用于训练/测试

---

### **实验设置与评估指标**

#### **噪声注入机制**
- 在输入数据上施加 **加性高斯噪声**：  
  $$
  x_{\text{noisy}} = \text{clip}_{[0,1]}(x + \epsilon),\quad \epsilon \sim \mathcal{N}(0, \sigma^2)
  $$
- 测试多个噪声水平 $\sigma \in \{0.00, 0.025, ..., 0.20\}$

#### **评估协议**
- 所有方法共享相同下游分类器：**SVM with precomputed kernel**
- 使用 **stratified train-test split (30% test)**，重复 5 次随机种子取平均
- **关键原则**：仅在干净数据 ($\sigma=0$) 上调参，之后冻结配置，用于所有噪声等级 → 避免“噪声依赖调参”带来的偏差

#### **主要评估指标**
| 指标 | 描述 |
|------|------|
| **Classification Accuracy & Macro-F1** | 主要任务性能 |
| **Kernel Alignment** | 与理想标签核的一致性 |
| **Within-/Between-class Similarity Difference** | 表示学习质量 |
| **Degradation Slope ($\beta_g$)** | 单位噪声引起的准确率下降，为核心鲁棒性度量 |

#### **统计分析方法**
- 采用 **dataset fixed-effects regression model**
- 使用 **wild cluster bootstrap (Rademacher weights, B=4000)** 进行推断，控制跨数据集依赖性
- 输出各方法相对于 QK-DFT 的斜率差及其置信区间

---

### **基线方法对比**

| 方法 | 类型 | 前端 | 后端 |
|------|------|--------|--------|
| **QK-DFT (SPE)** | 量子核 | DFT（频域选择） | Diagonal Gate |
| **QK-PCA** | 量子核 | PCA（方差最大化压缩） | Diagonal Gate |
| **QK-RP** | 量子核 | Random Projection（无结构降维） | Diagonal Gate |
| **SVM_Linear** | 经典核 | 同前端（DFT/PCA/RP） | Linear Kernel |
| **SVM_RBF** | 经典核 | 同前端 | RBF Kernel |

> 所有量子方法共享相同的对角嵌入结构，仅改变前端预处理，从而实现**受控比较**。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1 和 Figure 4）**

| 方法 | 退化斜率 Estimate | 95% CI | vs QK-DFT 差值 | 显著性 |
|------|------------------|--------|---------------|--------|
| **QK-DFT (reference)** | -0.214 | [-0.645, 0.222] | — | — |
| **QK-PCA** | -0.568 | [-1.063, -0.075] | **-0.354** | ✅ (p=1.0) |
| **QK-RP** | -1.367 | [-1.962, -0.760] | **-1.153** | ✅ (p=1.0) |
| **SVM_Linear** | -0.253 | [-0.630, 0.118] | -0.040 | ❌ |
| **SVM_RBF** | -0.362 | [-0.769, 0.041] | **-0.149** | ✅ (p=0.986) |

> 负值表示准确率随噪声上升而下降；数值越小（负得越多），退化越快。

---

### **与基线方法的对比结果**

#### **在量子家族内部**
- **QK-DFT 退化最慢**，显著优于 QK-PCA 和 QK-RP
- QK-RP 表现最差，说明**无结构的随机投影无法抵抗噪声破坏**
- QK-PCA 居中，因其保留最大方差方向，有一定抗噪能力

> ✅ **结论**：**频域结构（DFT）比方差结构（PCA）更有利于提升量子核的鲁棒性**

#### **与经典SVM对比**
- QK-DFT 的退化速率与 **SVM_Linear** 相当（差异不显著）
- 显著优于 **SVM_RBF**，后者在中等噪声下迅速崩溃
- 尽管前端相同（都用了DFT），但**量子对角嵌入比RBF核更稳定**

> ✅ **结论**：鲁棒性不仅来自预处理，还源于**核映射几何本身**——对角相位编码更具稳定性

#### **赢家统计（Figure 3）**
- 在 $\sigma \in [0.05, 0.15]$ 的中等噪声区，**QK-DFT 在 7–8/20 数据集上取得最高准确率**
- 其他方法表现波动大，缺乏持续竞争力

---

### **消融实验与硬件验证（Ablation & Real-device Test）**

#### **硬件微基准测试（IBM Quantum Devices）**
- 在 `ibm_fez` 和 `ibm_marrakesh` 上执行 SWAP Test，估计两个输入对应的量子态重叠
- 比较 **SPE (DFT + Diagonal Gate)** 与 **low-depth angle encoding** 的重叠误差（MAE）

**结果**：
- SPE 在 `n=2,3,4` qubits 下均保持稳定
- 未观察到“灾难性退化”或数值不稳定
- 尽管绝对误差略高于 angle encoding，但趋势可控，具备**实际可执行性**

> 🔧 **意义**：证明 SPE 可在当前 NISQ 硬件上可靠运行，非纯理论构造

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **鲁棒性取决于“结构对齐”的预处理 + 对角嵌入的协同作用**  
   - DFT 揭示的频域结构与对角量子门的物理机制高度匹配，形成正向反馈。
   
2. ✅ **QK-DFT 是目前最稳健的量子核构造之一**  
   - 在噪声递增环境下，其性能衰减最缓慢，且显著优于 QK-PCA 和 QK-RP。
   - 退化行为媲美线性SVM，优于RBF SVM。

3. ✅ **鲁棒性是“表示对齐”与“核几何”的联合产物**  
   - 即使使用相同DFT特征，SVM_RBF仍比 QK-DFT 更易崩溃 → 说明**量子对角嵌入本身提供了额外稳定性**。

4. ✅ **SPE 具备良好的硬件兼容性**  
   - 利用虚拟Z门和浅层电路设计，可在真实设备上稳定执行重叠估计。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **依赖频域结构** | 若原始数据无明显周期性或频谱规律（如文本、表格数据），SPE 效果可能受限 |
| **固定截断策略** | 当前使用固定低频选择，缺乏自适应频率掩码机制 |
| **非通用加速** | 并未声称实现量子优势，而是追求“实用可行 + 鲁棒性强”的中间目标 |
| **经典预处理开销** | 虽然FFT高效，但在大规模数据上仍需考虑整体流水线延迟 |

---

### **未来工作方向**
1. **结构增强扩展**  
   - 结合 covariant quantum kernels，显式建模平移/旋转不变性。

2. **可学习频谱选择**  
   - 引入可训练频率掩码或 wavelet/DCT 替代 DFT，提升灵活性。

3. **缓解核浓度问题**  
   - 探索 fidelity-based kernels 的 error mitigation 技术，延缓高维下的信息坍缩。

4. **可扩展核训练**  
   - 结合 subsampling 或 variational training 优化 kernel alignment。

5. **硬件感知编译优化**  
   - 利用 optimized diagonal decomposition 和 virtual-Z-centric 编译降低系统误差。

6. **连接经典难模拟电路**  
   - 探索与 commuting circuits 的结合，探索潜在的复杂性分离。

7. **面向应用的评估**  
   - 在金融、医疗等具体领域中测试 SPE 的实用性。

8. **误差缓解技术集成**  
   - 在 overlap estimation 中引入专用 error mitigation 方法。

---

> 📌 **总体评价**：  
> 本论文并未追求“超越经典”的量子霸权，而是倡导一种 **“鲁棒性优先”（robustness-first）** 的 NISQ 时代 QML 发展范式。它通过严谨的实证分析表明：**将经典信号处理智慧（如DFT）与量子硬件特性（如对角门）深度融合，是通往实用化量子机器学习的重要路径**。

</details>

---

### 12. [Decoding ML Decision: An Agentic Reasoning Framework for Large-Scale Ranking System](https://arxiv.org/abs/2602.18640)

**Authors**: Longfei Yun, Yihan Wu, Haoran Liu, Xiaoxuan Liu, Ziyun Xu, Yi Wang, Yang Xia, Pengfei Wang, Mingze Gao, Yunxiang Wang, Changfan Chen, Junfeng Pan  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.18640v1  

#### Abstract
Modern large-scale ranking systems operate within a sophisticated landscape of competing objectives, operational constraints, and evolving product requirements. Progress in this domain is increasingly bottlenecked by the engineering context constraint: the arduous process of translating ambiguous pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Decoding ML Decision: An Agentic Reasoning Framework for Large-Scale Ranking System**

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

现代大规模 ranking 系统面临“**工程上下文瓶颈**”（engineering context constraint）——即如何将模糊的产品意图转化为可执行、可验证、且符合生产约束的优化策略。传统方法如 uplift modeling 虽能识别统计上显著的干预策略，但常忽略以下现实因素：

- 特征不稳定性（feature instability）
- 多目标冲突（multi-objective trade-offs）
- 部署可行性（deployment feasibility）
- 手动流程耗时长、难以规模化

这导致许多“理论上最优”的策略在实际中无法部署。

---

### ✅ **提出了什么新方法或新思路**

论文提出 **GEARS**（**Generative Engine for Agentic Ranking Systems**），一个基于 **Agentic Reasoning** 的自主决策框架，将 ranking 优化重构为一个**在可编程实验环境中的自主探索过程**。

其三大核心创新如下：

#### （1）**Agentic Ranking Framework**
- 将 ranking 优化视为一个**交互式、迭代式的自主发现循环**，而非一次性模型推断。
- 代理（Agent）通过与实验系统交互，生成假设、评估可行性，并推荐可部署策略。

#### （2）**Skill-Based Agent Architecture（Specialized Agent Skills）**
- 引入模块化的 **Specialized Agent Skills**，将领域专家知识封装为可复用的推理能力。
- 每个 Skill 包含：
  - 元数据（用于路由）
  - 结构化分析指令
  - 对内部资源（SQL、代码库等）的引用
- 支持 **Vibe Optimization**：操作员可通过自然语言表达高阶意图（如“平衡 metric A 和 B”），由 Agent 自动翻译为算法约束。

#### （3）**Deterministic Lifecycle Governance**
- 在每个可执行动作前插入 **Validation Hook**，强制进行：
  - 特征稳定性审计（feature stability）
  - 统计稳健性检查（cohort consistency）
  - 性能持久性验证（performance persistence）
- 过滤掉依赖短期信号或不稳定特征的“脆弱策略”，确保推荐策略具备长期部署可靠性。

---

### ✅ **相比现有方法的优势**

| 方面 | 传统方法（如 Uplift Modeling） | GEARS |
|------|-------------------------------|-------|
| 决策模式 | 静态模型选择 | 动态、自主探索 |
| 上下文理解 | 忽略工程约束 | 显式建模部署可行性 |
| 可解释性 | 黑箱或有限解释 | 支持推理链追溯 |
| 稳定性保障 | 无机制 | 内置治理钩子（hooks） |
| 人机协作 | 人工主导 | 高层意图驱动，自动化执行 |

> ✅ **优势总结**：GEARS 实现了从“**模型驱动优化**”到“**智能体驱动发现**”的范式转变，兼顾性能、稳定性和可操作性。

---

## 2. 核心实验方法和设置

### 📦 **数据集**

- 构建了一个包含 **20 个内部线上实验** 的 benchmark 数据集。
- 每个实验使用 **GAS**（Large-scale Heterogeneous Treatment Effect framework）生成数百个候选策略。
- 每个候选策略附带多个 metric 的提升值及置信区间。

### 🧪 **实验设置**

- **任务**：给定自然语言指令（如“最大化 metric A 同时不损害 metric B”），从候选集中选出最优策略。
- **输入形式**：表格格式的实验记录（policy candidates + 多 metric 表现）。
- **输出形式**：排序后的推荐策略列表。

### 🎯 **评估指标**

采用信息检索与推荐系统常用指标：

| 指标 | 说明 |
|------|------|
| **Precision@K** | Top-K 推荐中属于真实最优集合的比例 |
| **Recall@K** | 真实最优策略中有多少被召回进 Top-K |
| **NDCG@K** | 考虑排名顺序的质量度量 |
| **Top-1 Accuracy** | 最优预测是否完全匹配真实最优 |
| **Top-1 in GT** | Top-1 是否在真实最优集合中（宽松版） |
| **Ranking Correlation (Spearman’s ρ)** | 整体排序与真实排序的一致性 |

测试 K ∈ {1, 3, 5}

### 🔁 **基线方法对比**

| 基线方法 | 描述 |
|--------|------|
| **Naive Prompting** | 直接提问，无推理引导 |
| **Chain-of-Thought (CoT)** | 引导逐步推理 |
| **Self-Consistency** | 多路径采样 + 投票聚合 |
| **Self-Refine** | 两阶段：先生成 → 再反思修正 |
| **Code-as-Action** | LLM 生成并执行代码来处理数据 |

---

## 3. 主要实验结果和性能指标

### 📊 **整体性能对比（Table 1）**

| Method | nDCG@1 | nDCG@3 | nDCG@5 | Prec@1 | Rec@1 | Top-1 Acc | Top-1 in GT |
|--------|--------|--------|--------|--------|--------|-----------|-------------|
| Naive | 0.57 | 0.70 | 0.74 | 0.57 | 0.36 | 0.44 | 0.57 |
| CoT | 0.68 | 0.80 | 0.83 | 0.68 | 0.43 | 0.57 | 0.68 |
| Self-Consistency | 0.57 | 0.74 | 0.76 | 0.57 | 0.33 | 0.37 | 0.57 |
| Self-Refine | 0.61 | 0.78 | 0.80 | 0.61 | 0.37 | 0.50 | 0.61 |
| **Code-as-Action** | 0.77 | 0.87 | 0.87 | 0.77 | 0.45 | 0.68 | 0.77 |
| **GEARS w/o Skill** | 0.87 | 0.91 | 0.91 | 0.87 | 0.53 | 0.77 | 0.87 |
| **GEARS (Ours)** | **0.94** | **0.96** | **0.96** | **0.94** | **0.56** | **0.86** | **0.94** |

> ✅ **关键发现**：
- GEARS 在所有指标上**全面超越所有基线**，尤其在 **nDCG@1** 和 **Top-1 Accuracy** 上表现突出。
- 即使强基线 **Code-as-Action** 也被显著超越，说明仅靠“可执行代码”不足以解决复杂 ranking 决策问题。

---

### 🔍 **消融实验结果**

| 变体 | 关键变化 | 性能影响 |
|------|--------|---------|
| **GEARS w/o Bash** | 移除确定性预过滤（bash-based filtering） | 性能大幅下降（nDCG@1 从 0.94 → 0.40）<br>→ 表明**预筛选对稳定性至关重要** |
| **GEARS w/o Skill** | 移除 Specialized Agent Skills | 性能中度下降（Top-1 Acc 0.86 → 0.77）<br>→ 表明**Skills 提供额外语义理解增益** |

> ✅ **结论**：**Deterministic Governance** 和 **Skill Modules** 均为核心组件，缺一不可。

---

### 🛡️ **Hooks 对可靠性的提升（Section 4.3）**

- 定义 **User-Cohort Shift Ratio (R_shift)** 作为特征稳定性指标。
- 设立阈值：
  - Binary Cut: R_shift ≤ 15%
  - Quantile Cut: R_shift ≤ 45%
- 实验显示：
  - 不稳定特征（如 Feature 4，R_shift ~50%）被自动过滤。
  - 保留策略在 **一个月回测中保持性能稳定**（见 Figure 4）。

> ✅ **验证了 Deterministic Lifecycle Governance 的有效性**。

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **ranking 优化的本质瓶颈是工程上下文，而非模型能力**  
   → 即使有强大的 HTE 模型（如 GAS），仍需解决“如何选可部署策略”的问题。

2. **Agentic Framework 可桥接高层意图与底层执行**  
   → 通过 **Vibe Optimization**，实现自然语言驱动的策略生成。

3. **模块化 Skills 是知识沉淀的关键**  
   → 将专家经验编码为可调用工具，避免“context rot”和幻觉。

4. **Deterministic Governance 显著提升部署可靠性**  
   → 钩子机制有效防止过拟合瞬时信号，保障长期性能。

5. **GEARS 已在多业务场景落地并取得正向收益**  
   → 见 Table 3：在 9 个不同 surface 上均实现关键 metric 提升（0.01% ~ 0.37%）。

---

### ⚠️ **局限性**

1. **依赖高质量的内部 artifact 和历史数据**  
   → 对缺乏系统化知识管理的组织迁移成本较高。

2. **当前聚焦于 post-experiment 分析，尚未完全闭环控制实验设计**  
   → 未来可扩展至全周期自适应实验（Adaptive Experimentation）。

3. **LLM 的推理一致性仍可能受提示扰动影响**  
   → 尽管有 progressive disclosure 缓解，但极端复杂任务仍存在漂移风险。

---

### 🔮 **未来工作方向**

1. **扩展至实时在线决策**  
   → 将 GEARS 与 Multi-Armed Bandits 或 Bayesian Optimization 结合，实现动态策略调整。

2. **构建通用 Agent Skill Market**  
   → 支持跨团队共享和组合 Skills，形成 ranking 智能生态。

3. **增强因果推理能力**  
   → 引入 counterfactual simulation，支持“what-if”类策略推演。

4. **降低对特定 LLM 的依赖**  
   → 探索轻量化 agent 架构或混合符号系统。

---

## ✅ 总结

**GEARS** 是首个将 **Agentic Reasoning** 系统化应用于大规模 ranking 优化的框架。它不仅提升了策略发现效率，更重要的是解决了“**最优 ≠ 可部署**”这一工业界长期痛点。通过 **Specialized Agent Skills** 和 **Deterministic Lifecycle Governance**，实现了性能、稳定性与可解释性的统一，为下一代 AI 驱动的 ranking 基础设施提供了新范式。

</details>

---

### 13. [Asking the Right Questions: Improving Reasoning with Generated Stepping Stones](https://arxiv.org/abs/2602.19069)

**Authors**: Hengyuan Hu, Tingchen Fu, Minqi Jiang, Alexander H Miller, Yoram Bachrach, Jakob Nicolaus Foerster  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.19069v1  

#### Abstract
Recent years have witnessed tremendous progress in enabling LLMs to solve complex reasoning tasks such as math and coding. As we start to apply LLMs to harder tasks that they may not be able to solve in one shot, it is worth paying attention to their ability to construct intermediate stepping stones...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Asking the Right Questions: Improving Reasoning with Generated Stepping Stones**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
当前的 **Large Language Models (LLMs)** 在处理复杂推理任务（如数学、编程）时，虽然已能通过 **Chain-of-Thought (CoT)** 和 **test-time scaling** 等技术取得进展，但在面对超出其一次性求解能力的难题时，仍缺乏有效的“分解”策略。  
传统方法通常依赖于端到端的推理或验证机制，而忽略了人类解决问题时的关键能力——**提出有洞察力的中间问题（stepping stones）**，例如简化问题、构造类比或设计子任务。

该论文指出：**LLMs 缺乏主动构建“中间台阶”来辅助解决困难问题的能力**，而这正是提升其长程推理潜力的关键。

---

### 🚀 **提出了什么新方法或新思路**
论文提出了 **ARQ (Asking the Right Questions)** 框架，系统性地研究如何让 LLMs 生成有助于解决目标问题的“中间问题”。

#### 核心思想：
- 将复杂问题的求解过程拆分为两步：
  1. **Question Generator**：先生成一个相关的、更简单的“stepping stone”问题（如特例、简化版）。
  2. **Problem Solver**：先解决这个中间问题，再以该问题及其答案作为上下文示例，去解决原始问题。

#### 创新点：
1. **首次将“提问能力”形式化为可评估与训练的任务**，而非仅作为推理链的一部分。
2. 提出了一种 **inference-time scaffold**（推理时框架），可在不修改模型权重的情况下增强现有 LLM 的表现。
3. 进一步将 stepping stone generation 视为一个 **post-training task**，利用合成数据进行 SFT 和 DPO 训练，显著提升 LLM 主动构造有用问题的能力。
4. 探索了 **sequential** 与 **recursive** 多步石生成策略，扩展为“测试时课程学习”（test-time curriculum）。

---

### 🔍 **相比现有方法的优势**
| 方法 | 局限性 | ARQ 的优势 |
|------|--------|-----------|
| **Prompt-based CoT / Self-Ask / Least-to-Most** | 仅在简单模型上有效；对已具备强 CoT 能力的现代 LLM 无效甚至有害 | ARQ 显著提升高难度任务成功率，且效果可迁移 |
| **Tree of Thoughts (ToT)** | 依赖搜索算法，计算开销大 | ARQ 更轻量，聚焦于“问题生成”这一认知环节 |
| **Verifier / Refinement 方法** | 侧重验证与纠错，不主动构建知识 | ARQ 强调“前向引导”，通过构造中间问题建立直觉 |

> ✅ **ARQ 的核心优势在于：它不是直接改进 solver，而是教会模型“如何更好地准备自己去思考”。**

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
- **AIME 2024**：30 道高中数学竞赛题，标准推理基准。
- **AIME 2025**：同上，用于泛化性测试。
- **BeyondAIME (Seed et al., 2025)**：100 道更具挑战性的数学题，格式与 AIME 一致但难度更高。

> 所有问题均需输出单个整数答案，便于自动评估。

---

### ⚙️ **实验设置和评估指标**

#### 模型配置：
- **Solver**：GPT-OSS-120B（简称 GPT-120B），使用不同 reasoning effort（low / high）模拟不同能力层级。
- **Generator**：包括 GPT-120B、Qwen3-8B、Qwen2.5-32B 等 off-the-shelf 或 post-trained 模型。

#### 评估方式：
- **Success Rate**：正确解答的比例（exact match）。
- **Monte Carlo Rollouts**：每个 stepping stone 用 solver 运行多次（默认 20 次），估计其平均增益。
- **Best vs Average**：区分所有生成问题中的最佳表现与平均表现，避免被低质问题拉低。

#### ARQ 流程：
```python
z ~ φ(x)              # 生成 stepping stone
yz ~ π(z)             # 求解 stepping stone
y ~ π(x; z, yz)       # 求解原问题，以 (z, yz) 为上下文
```

---

### 🆚 **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Solver Only** | 直接求解，无任何辅助 |
| **Analogical (Yasunaga et al., 2024)** | 提示模型先生成类比问题并求解 |
| **Least-to-Most (Zhou et al., 2023)** | 分解问题后依次求解 |
| **Plan-and-Solve, Step-Back, Self-Discover** | 其他 prompting 类方法 |
| **Rand** | 随机生成 AIME 风格问题，作为负面对照 |

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据**

#### （1）**ARQ 在 off-the-shelf 模型上的表现（Fig 2）**
| 方法 | AIME 2024 | AIME 2025 | BeyondAIME |
|------|----------|----------|------------|
| Solver Only | 72.8% | 69.8% | 58.6% |
| Analogical | 64.2% | 55.2% | 52.6% |
| Least-to-Most | 66.3% | 53.3% | 49.3% |
| **ARQ (high-effort gen.)** | — | — | **64.2%** ✅ |

> 💡 结论：传统 prompting 方法在强 LLM 上反而**降低性能**；只有当 generator 足够强大时，ARQ 才能在最难的 **BeyondAIME** 上带来增益。

---

#### （2）**最佳 stepping stone 的效果（Fig 3）**
| 方法 | 平均提升（vs Solver Only） |
|------|-------------------------|
| ARQ low (weak gen.) | +5% |
| ARQ high (strong gen.) | **+13%** ✅ |

> 单个最优 stepping stone 可使成功率大幅提升，说明“好问题”的价值极高。

---

#### （3）**transferability 实验（Fig 5）**
- 使用一个 solver 选出的最佳 stepping stones，在另一个不同大小/能力的 solver 上依然有效。
- 例如：GPT-20B medium solver 使用由 GPT-120B high 选出的好问题，性能提升明显，且与自选问题效果相当。

> ✅ 表明好的 stepping stones 是**通用的、非过拟合的启发式工具**。

---

#### （4）**post-training 效果（Fig 6）**
| 模型 | Post-training 前提升 | 后提升 |
|------|--------------------|--------|
| Qwen3-8B | +0.6% | → **+3.1%** ✅ |
| Qwen2.5-32B | 无法生成有效问题 | → **+2.9%** ✅ |

> 经过基于合成数据的 SFT + DPO 训练后，即使是原本不会提问的模型也能学会生成高质量 stepping stones。

---

#### （5）**多 stepping stones 实验（Fig 7）**
| 策略 | 模型 | 最佳提升（3 stones） | 平均提升 |
|------|------|------------------|----------|
| Sequential | GPT-120B high | **+5.2%** | +3.9% |
| Sequential | Post-trained Qwen3-8B | +3.7% | +1.7% |
| Sequential | Post-trained Qwen2.5-32B | +3.8% | +2.7% |
| Recursive | — | ❌ 波动或退化 | 不稳定 |

> ✅ **Sequential 多步石持续增益，recursive 效果不佳**  
> ✅ 即便 post-training 仅针对单步石，也能泛化到多步场景

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **存在高质量的 stepping stone 问题**，它们能显著提升 solver 成功率（最高 +13%）。
2. 好的问题具有**可转移性**：在一个 solver 上有效的 stepping stone，在其他 solver 上也有效。
3. 当前主流 prompting 方法（如 Analogical、Least-to-Most）在先进 reasoning LLM 上**不再有效甚至有害**。
4. 通过 **合成数据 + SFT/DPO**，可以成功 fine-tune LLM 学会“问对问题”。
5. **Sequential 多 stepping stones** 构成有效的 test-time curriculum，性能随数量增加而稳步上升。

---

### ⚠️ **方法的局限性**
1. **依赖高质量 solver 来打分**：当前 scoring 函数 $ S(z,x) = \mathbb{E}[R(x,y)] $ 需要 Monte Carlo rollout，成本较高。
2. **无法保证生成问题的正确性**：若 stepping stone 的答案错误，可能污染上下文。
3. **prompt 设计敏感**：generator 必须理解“什么是好的 stepping stone”，否则易生成无关或误导性问题。
4. **目前仅在数学领域验证**，尚未推广至 coding、规划等其他复杂任务。

---

### 🔮 **未来工作方向**
1. **Scaling to Coding & Planning Tasks**：将 ARQ 应用于编程、agent 决策等更广泛场景。
2. **Online RL for Question Generation**：不再依赖离线合成数据，而是在线优化 generator。
3. **Better Scoring Functions**：探索无需完整 rollout 的高效 reward estimation。
4. **Integrate with Search Algorithms**：将 ARQ 作为 ToT 或 MCTS 中的一个 operator。
5. **Human-in-the-loop Evaluation**：结合人工判断评估 stepping stone 的“启发价值”。

---

## ✅ 总结一句话
> **ARQ 首次系统性地将“提问的艺术”引入 LLM 推理流程，证明了“生成好的中间问题”是一项可学习、可迁移、且极具增益的认知技能，为下一代 reasoning LLM 提供了全新的训练范式。**

</details>

---

### 14. [Whisper: Courtside Edition Enhancing ASR Performance Through LLM-Driven Context Generation](https://arxiv.org/abs/2602.18966)

**Authors**: Yonathan Ron, Shiri Gilboa, Tammuz Dubnov  
**Category**: cs.CL  
**Published**: 2026-02-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.18966v1  

#### Abstract
Domain-specific speech remains a persistent challenge for automatic speech recognition (ASR), even for state-of-the-art systems like OpenAI's Whisper. We introduce Whisper: Courtside Edition, a novel multi-agent large language model (LLM) pipeline that enhances Whisper transcriptions without retrain...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Whisper: Courtside Edition: Enhancing ASR Performance Through LLM-Driven Context Generation》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **领域特定语音识别（Domain-specific ASR）的挑战**：尽管如 Whisper 这样的通用 ASR 模型在通用语境下表现优异，但在高密度专有名词（如球员姓名）、技术术语（如“pick and roll”）和快速口语场景（如 NBA 体育解说）中仍存在显著错误。
- 典型错误包括：
  - **Player name mangling**（如 “Giannis Antetokounmpo” → “yanis anteto kumbo”）
  - **Jargon corruption**（如 “pick and roll” → “picker roll”）
  - **上下文误判**（如 “pass to 80” 未能解析为 “pass to A.D.”）

传统解决方案依赖 **fine-tuning 或 contextual biasing**，但需要大量标注数据、模型修改甚至重新训练，成本高且难以扩展。

---

### 🚀 提出的新方法与创新思路

提出 **Whisper: Courtside Edition** —— 一种无需重训练的多智能体 LLM 驱动的 **prompt-based 增强框架**，通过以下机制提升 ASR 性能：

#### 核心架构：Multi-Agent LLM Pipeline
该系统拦截 Whisper 的第一轮转录输出，并利用多个专业化 LLM Agent 协同生成紧凑的 `initial_prompt`，用于引导 Whisper 的第二次解码过程。

各 Agent 分工如下：
| Agent | 功能 |
|-------|------|
| **Topic Classification Agent** | 识别领域上下文（如 “NBA basketball commentary”） |
| **Named Entity Recognition (NER) Agent** | 抽取并标准化球员/教练姓名，基于官方名单进行模糊匹配（fuzzy matching） |
| **Jargon Extraction Agent** | 识别篮球术语（如 “fast break”, “alley-oop”），结合 RAKE/YAKE 与领域术语表 |
| **Decision Filtering Agents** | 判断是否需插入名字或术语，防止过纠错（over-correction） |
| **Candidate Selector & Sentence Builder** | 综合高价值候选词，构建自然语言句子作为 prompt，优先将关键 token 放在末尾（利用 Whisper 只考虑最后 ≤224 tokens 的特性） |

> 💡 **关键创新**：不是对 Whisper 输出进行事后修正（post-editing），而是利用其内置的 `initial_prompt` 接口，在 **解码阶段注入上下文知识**，实现真正的 **context-aware ASR 增强**。

---

### 🔍 相比现有方法的优势

| 方法 | 是否需 retrain | 是否访问音频 | 是否可扩展 | 主要缺陷 |
|------|----------------|---------------|------------|----------|
| **Fine-tuning / CB-Whisper [3]** | ✅ 是 | N/A | ❌ 成本高 | 需修改模型，部署困难 |
| **LLM Post-Correction** | ❌ 否 | ❌ 否 | ✅ 易部署 | 无法恢复听错内容（无音频信息） |
| **本文方法 (Courtside Edition)** | ❌ 否 | ✅ 是（间接通过 prompt 引导） | ✅ 极高 | 仅依赖 LLM + 知识库 |

✅ **优势总结**：
- **免训练（Training-free）**：适用于任何已部署的 Whisper 实例
- **高效灵活**：只需更新知识库即可适配新领域
- **精准控制**：通过决策过滤器避免引入错误
- **符合 Whisper 解码机制**：自然句式 prompt 更稳定，关键词靠后更有效

---

## 2. 核心实验方法和设置

### 📚 数据集
- **NBA Basketball Commentary Dataset**
  - 包含 **421 段**真实比赛解说音频片段
  - 每段时长 **10–30 秒（中位数 15s）**
  - 覆盖多种广播源、球队组合、比赛情境、口音及背景噪音（观众声、多人说话）
  - 所有片段由 **领域专家人工标注并规范化**（正字法、标点、实体标准化）
  - 拆分策略：按 **比赛场次和转播团队隔离**，防止信息泄露

---

### 📊 实验设置与评估指标

#### 主要指标
- **Word Error Rate (WER)**：
  $$
  \text{WER} = \frac{S + D + I}{N}
  $$
  其中 $S$: 替换，$D$: 删除，$I$: 插入，$N$: 总词数

#### 文本预处理
- 统一大小写、去除标点、数字标准化、空白符规整化，确保评估聚焦于语义准确性

#### 基线方法对比
| Pipeline | 描述 |
|--------|------|
| **Baseline**: `whisper-medium.en` | 原始 Whisper 模型，无任何增强 |
| **P1: Topic-Only** | 仅使用话题标签作为 prompt（如 “NBA basketball commentary”） |
| **P2: LLM Post-Fix** | 使用 GPT-4o 对 Whisper 输出进行后编辑 |
| **P3: Names-Enhanced** | 加入 NER 结果，但无决策过滤 |
| **P4: Full Multi-Agent** | 完整六代理系统，含决策过滤与句子构建 |

所有实验均采用 **paired design**，同一音频运行两次 Whisper（带 prompt 第二遍），比较 WER 差异。

统计检验：**Wilcoxon signed-rank test** + 效应量分析（effect size）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 1）

| Pipeline | Mean WER ± SD | Improvement (%) | Degradation (%) |
|---------|----------------|------------------|------------------|
| Baseline | 0.217 ± 0.232 | — | — |
| P1: Topic-Only | 0.238 ± 0.247 | 20.9% ↓ | 25.9% ↑ |
| P2: LLM Post-Fix | 0.217 ± 0.231 | 19.2% | 19.5% |
| P3: Names-Enhanced | 0.210 ± 0.250 | 36.8% | 20.7% |
| **P4: Full Multi-Agent** | **0.180 ± 0.217** | **40.1%** | **7.1%** |

> ✅ **相对 WER 下降 17.0%**（从 0.217 → 0.180），**p < 0.001**，具有统计显著性和大效应量。

---

### 🔬 消融实验与关键发现

| 观察项 | 发现 |
|-------|------|
| **P1 表现更差** | 仅加宽泛主题反而误导模型，导致 WER 上升（+9.7%） |
| **P2 几乎无效** | LLM 后编辑无法纠正听错的名字或术语，改善与退化比例接近 1:1 |
| **P3 改进明显但风险高** | 名字增强带来一定收益，但因缺乏过滤机制，**退化率高达 20.7%** |
| **P4 显著优于其他** | 在大幅提升准确率的同时，将退化率压至 **仅 7.1%**，体现决策模块的重要性 |

> 📌 **成功案例示例**：
- `"anteto kumbol"` → `"Antetokounmpo"`
- `"picker roll"` → `"pick and roll"`
- `"the pass to 80"` → `"the pass to A.D."`

> ⚠️ **失败模式分析**：
- **Over-correction (3.2%)**：错误地插入未提及的球员名
- **Out-of-scope terminology (1.8%)**：罕见昵称不在术语库中（如 “Chef Curry”）

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **Prompt-based augmentation 是有效的领域自适应路径**  
   无需 fine-tuning，仅通过智能 prompt 构造即可显著提升 ASR 在专业领域的表现。

2. **Multi-agent 架构优于单体 LLM 处理**  
   将任务分解为 topic detection、NER、jargon extraction 和 decision filtering，实现了更精细、可控的上下文注入。

3. **Whisper 的 `initial_prompt` 机制潜力巨大**  
   正确设计 prompt（自然句式、关键词靠后、长度 ≤224 tokens）可最大化其引导能力。

4. **优于 post-processing 方案**  
   因能结合音频信号进行再解码，本方法可“纠正听错”，而纯文本后编辑做不到这一点。

---

### ⚠️ 局限性

1. **依赖外部知识库维护**  
   球员名单、术语表需定期更新，否则会影响 recall。

2. **LLM API 成本与延迟**  
   当前使用 GPT-4o，每段约增加 **\$0.008 成本** 和 **3.3 秒延迟**，虽可接受但不适合超低延迟场景。

3. **Tokenization 不稳定性**  
   某些 prompt 导致输出被截断，因此设置了 fallback 机制（若输出 <80% 原长度则回退）。未来模型有望缓解此问题。

4. **尚未验证跨领域泛化性**  
   当前仅测试于 NBA 解说，医疗、法律等领域的有效性待验证。

---

### 🔮 未来工作方向

| 方向 | 描述 |
|------|------|
| **Cross-Domain Validation** | 在 medical transcription、legal depositions、academic lectures 中验证通用性 |
| **Real-Time Optimization** | 探索 agent 并行化、本地 LLM 部署以降低延迟与成本 |
| **Streaming ASR Integration** | 与流式 ASR 结合，支持实时 captioning |
| **Multimodal Enhancement** | 引入视频信号（如球员识别、比分板）进一步丰富上下文 |
| **Adaptive Learning** | 建立反馈闭环，动态更新知识库与 agent 行为策略 |

---

## ✅ 总结一句话

> **Whisper: Courtside Edition 展示了一种免训练、模块化、基于 multi-agent LLM 的 prompt 增强范式，可在不解动模型的前提下，实现对领域特定 ASR 的高效优化，在 NBA 解说任务上取得了 17.0% 的相对 WER 降低，为 ASR 与 LLM 的协同开辟了新路径。**

</details>

---

### 15. [IAPO: Information-Aware Policy Optimization for Token-Efficient Reasoning](https://arxiv.org/abs/2602.19049)

**Authors**: Yinhan He, Yaochen Zhu, Mingjia Shi, Wendy Zheng, Lin Su, Xiaoqing Wang, Qi Guo, Jundong Li  
**Category**: cs.CL  
**Published**: 2026-02-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.19049v1  

#### Abstract
Large language models increasingly rely on long chains of thought to improve accuracy, yet such gains come with substantial inference-time costs. We revisit token-efficient post-training and argue that existing sequence-level reward-shaping methods offer limited control over how reasoning effort is ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：IAPO: Information-Aware Policy Optimization for Token-Efficient Reasoning

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在进行复杂推理任务时，通常依赖于长链的思维过程（Chain-of-Thought, CoT）来提高准确性。然而，这种策略导致了显著的推理成本增加，尤其是在推理延迟和计算开销方面。现有的基于强化学习（RL）的后训练方法（如GRPO）虽然提升了推理能力，但往往产生冗余、循环或无信息量的推理步骤，造成严重的**推理冗长度（reasoning verbosity）**。

本文指出，当前主流的 token-efficient RL 方法存在一个根本性缺陷：它们是**内容无关的（content-agnostic）**。具体表现为：
- **基于长度的方法**（Length-based）：对所有短输出统一奖励，不区分其中的 token 是否真正有用。
- **基于位置的方法**（Position-based）：对后面的 token 进行惩罚，即使这些 token 可能是关键的结论性步骤。

这导致模型难以区分“必要的推理”和“冗余的探索”，从而无法实现真正的高效推理。

### 提出的新方法和新思路
为解决上述问题，作者提出了 **IAPO (Information-Aware Policy Optimization)**，一种全新的 token-level 优势塑造（advantage shaping）框架。

其核心创新在于引入了**信息论**（information theory）的思想，通过量化每个 token 对最终答案正确性的信息贡献来指导优化。

#### 核心方法
1.  **信息感知优势塑造模块 (Information-Aware Advantage Shaping Module)**：
    - **信息度量（Informativeness Level）**：使用**条件互信息（Conditional Mutual Information, MI）** `I(y; o_t | q, o_<t)` 来衡量第 `t` 个 token `o_t` 在给定查询 `q` 和前序推理 `o_<t` 的条件下，对最终答案 `y` 的不确定性减少程度。
    - **探索调整（Exploration Adjustment）**：为了防止模型过早收敛到过于简短的模式而牺牲准确性，引入了一个基于旧策略概率的探索项 `c_i,t`，以调节探索行为。

2.  **高效条件 MI 估计模块 (Efficient Conditional MI Estimation Module)**：
    - **早期退出估计器（Early-Exit-Based Estimator）**：通过在推理序列中插入一个特殊的提示（如 `</think><answer>`），强制模型提前生成最终答案，从而利用前后两次答案分布的熵差来近似计算条件 MI。
    - **加速技术**：
        - **KV-Cache 预加载**：避免对前序 token 的重复计算，将时间复杂度从 `O(L^3d)` 降低到 `O((K+L^2)d)`。
        - **分块前向传播（Chunk-wise Forwarding）**：批量处理一个 chunk 内的所有 token，进一步摊销计算成本。

### 相比现有方法的优势
- **原理更优**：直接基于 token 的**信息贡献**而非长度或位置进行奖励，实现了对推理质量的精细化控制。
- **效率更高**：提出的估计模块使得在大规模 LLM 上进行 token-level 的信息计算成为可能。
- **效果更好**：在保持甚至提升准确率的同时，显著减少了推理长度，实现了真正的“高性价比”推理。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个具有代表性的数学推理数据集上进行：
- **GSM8K**：小学水平的算术应用题，需要多步数值推理。
- **MATH-500**：竞赛级别的数学问题，难度更高。
- **DAPO-Math-17k**：大规模、多样化的数学数据集，解决方案普遍更长。

### 实验设置和评估指标
- **基础模型**：在 `Qwen2.5` 系列模型（0.5B, 1.5B, 7B 参数规模）上进行后训练。
- **优化算法**：使用 AdamW 优化器，结合 GRPO 框架。
- **评估指标**：
    - **Pass@k**：k 次尝试内答对的样本比例，衡量有效性（effectiveness）。
    - **Length@k**：k 次完成的平均长度，衡量效率（efficiency）。
    - **Ratio@k (Pass@k / Length@k)**：核心指标，衡量**token 效率**，即每个 token 产出正确答案的效能。

### 基线方法对比
与当时最先进的 token-efficient RL 后训练方法进行比较：
- **DAPO**：对过长序列分配零优势。
- **GFPO**：仅对最短的成功序列分配奖励。
- **GTPO**：基于 token 熵进行奖励塑造。
- **S-GRPO**：对超过长度阈值的 token 分配零优势。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **推理长度大幅减少**：在 `Qwen2.5-7B-Instruct` 模型上，相比基线，IAPO 将推理长度最多减少了 **36%**。
- **准确率保持或提升**：在多个设置下，IAPO 不仅没有牺牲准确性，反而在 `MATH-500` 等数据集上取得了更高的 `Pass@k`。
- **token 效率最优**：在所有 `Ratio@k` 指标上，IAPO 在绝大多数情况下达到了**最优或次优**的结果，证明了其在单位 token 成本下的卓越性能。

### 与基线方法的对比结果
从表1（Table 1）的 `Ratio@32` 结果可以看出：
- 在 `Qwen2.5-7B-Instruct` + `GSM8K` 设置下，IAPO 的 `Ratio@32` 达到了 **8.94×10⁻³**，远超第二名（GFPO 的 6.09×10⁻³）。
- 在 `DAPO-Math-17k` 数据集上，IAPO 的 `Ratio@32` 为 **2.14×10⁻²**，同样显著优于其他基线。

### 消融实验结果
通过两个变体进行消融研究：
- **IAPO-NI**：移除基于信息度量的优势项。
- **IAPO-NE**：用下一个 token 的熵减作为替代信息度量。

**结果**：两种变体的 `Ratio@32` 性能均低于完整的 IAPO，证明了：
1.  基于信息度量的优势塑造对提升 token 效率至关重要。
2.  作者提出的条件 MI 估计方法是有效的，且优于简单的替代方案。

---

## 4. 关键结论和发现

### 主要发现
1.  **信息感知是关键**：通过条件 MI 量化 token 的信息贡献，是一种强大且通用的机制，能够有效识别和鼓励生成高信息量的推理步骤。
2.  **理论保证**：论文提供了理论分析，证明 IAPO 能够在保持模型准确性的前提下，单调地减少预期的推理长度。
3.  **实践成功**：IAPO 在多个数据集和模型规模上，一致地实现了**state-of-the-art 的 token 效率**，验证了其有效性。
4.  **可扩展性强**：该方法不仅适用于数学推理，在非数学的常识推理任务（CommonsenseQA）上也表现出色，并且能将实际推理时间减少超过 11%。

### 方法的局限性
- **计算开销**：尽管有 KV-Cache 等优化，但实时估计每个 token 的条件 MI 仍然带来了额外的计算负担，尤其是在训练阶段。
- **估计误差**：早期退出估计器是对真实条件 MI 的近似，可能存在一定的偏差。
- **领域依赖**：该方法的有效性依赖于一个可靠的奖励模型或答案标签来定义“最终答案”。

### 未来工作方向
- **更高效的估计器**：设计更轻量级、更精确的条件 MI 估计方法。
- **扩展到其他任务**：将 IAPO 应用于代码生成、对话系统等更广泛的需要长文本生成的任务。
- **探索其他信息度量**：研究是否可以使用其他信息论或认知科学中的概念来进一步改进 token 的价值评估。
- **在线学习**：探索如何在推理过程中动态应用类似思想，实现自适应的、高效的推理路径规划。

</details>

---

### 16. [BiScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS](https://arxiv.org/abs/2602.18755)

**Authors**: Omar Basit, Yunzhao Liu, Z. Jonny Kong, Y. Charlie Hu  
**Category**: cs.DC  
**Published**: 2026-02-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.18755v1  

#### Abstract
Prefill/decode disaggregation is increasingly adopted in LLM serving to improve the latency-throughput tradeoff and meet strict TTFT and TPOT SLOs. However, LLM inference remains energy-hungry: autoscaling alone is too coarse-grained to track fast workload fluctuations, and applying fine-grained DVF...

---

### 17. [Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines](https://arxiv.org/abs/2602.19265)

**Authors**: Siavash Khodakarami, Vivek Oommen, Nazanin Ahmadi Daryakenari, Maxim Beekenkamp, George Em Karniadakis  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.19265v1  

#### Abstract
Solving partial differential equations (PDEs) by neural networks as well as Kolmogorov-Arnold Networks (KANs), including physics-informed neural networks (PINNs), physics-informed KANs (PIKANs), and neural operators, are known to exhibit spectral bias, whereby low-frequency components of the solutio...

---

### 18. [Fully Convolutional Spatiotemporal Learning for Microstructure Evolution Prediction](https://arxiv.org/abs/2602.19915)

**Authors**: Michael Trimboli, Mohammed Alsubaie, Sirani M. Perera, Ke-Gang Wang, Xianqi Li  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.19915v1  

#### Abstract
Understanding and predicting microstructure evolution is fundamental to materials science, as it governs the resulting properties and performance of materials. Traditional simulation methods, such as phase-field models, offer high-fidelity results but are computationally expensive due to the need to...

---

### 19. [EvalSense: A Framework for Domain-Specific LLM (Meta-)Evaluation](https://arxiv.org/abs/2602.18823)

**Authors**: Adam Dejl, Jonathan Pearson  
**Category**: cs.CL  
**Published**: 2026-02-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.18823v1  

#### Abstract
Robust and comprehensive evaluation of large language models (LLMs) is essential for identifying effective LLM system configurations and mitigating risks associated with deploying LLMs in sensitive domains. However, traditional statistical metrics are poorly suited to open-ended generation tasks, le...

---

### 20. [ucTrace: A Multi-Layer Profiling Tool for UCX-driven Communication](https://arxiv.org/abs/2602.19084)

**Authors**: Emir Gencer (Ko\c{c} University, Turkey), Mohammad Kefah Taha Issa (Ko\c{c} University, Turkey), Ilyas Turimbetov (Ko\c{c} University, Turkey), James D. Trotter (Simula Research Laboratory, Norway), Didem Unat (Ko\c{c} University, Turkey)  
**Category**: cs.DC  
**Published**: 2026-02-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.19084v1  

#### Abstract
UCX is a communication framework that enables low-latency, high-bandwidth communication in HPC systems. With its unified API, UCX facilitates efficient data transfers across multi-node CPU-GPU clusters. UCX is widely used as the transport layer for MPI, particularly in GPU-aware implementations. How...

---

### 21. [Robust Exploration in Directed Controller Synthesis via Reinforcement Learning with Soft Mixture-of-Experts](https://arxiv.org/abs/2602.19244)

**Authors**: Toshihide Ubukata, Zhiyao Wang, Enhong Mu, Jialong Li, Kenji Tei  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.19244v1  

#### Abstract
On-the-fly Directed Controller Synthesis (OTF-DCS) mitigates state-space explosion by incrementally exploring the system and relies critically on an exploration policy to guide search efficiently. Recent reinforcement learning (RL) approaches learn such policies and achieve promising zero-shot gener...

---

### 22. [IR$^3$: Contrastive Inverse Reinforcement Learning for Interpretable Detection and Mitigation of Reward Hacking](https://arxiv.org/abs/2602.19416)

**Authors**: Mohammad Beigi, Ming Jin, Junshan Zhang, Jiaxin Zhang, Qifan Wang, Lifu Huang  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.19416v1  

#### Abstract
Reinforcement Learning from Human Feedback (RLHF) enables powerful LLM alignment but can introduce reward hacking - models exploit spurious correlations in proxy rewards without genuine alignment. Compounding this, the objectives internalized during RLHF remain opaque, making hacking behaviors diffi...

---

### 23. [Ada-RS: Adaptive Rejection Sampling for Selective Thinking](https://arxiv.org/abs/2602.19519)

**Authors**: Yirou Ge, Yixi Li, Alec Chiu, Shivani Shekhar, Zijie Pan, Avinash Thangali, Yun-Shiuan Chuang, Chaitanya Kulkarni, Uma Kona, Linsey Pang, Prakhar Mehrotra  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.19519v1  

#### Abstract
Large language models (LLMs) are increasingly being deployed in cost and latency-sensitive settings. While chain-of-thought improves reasoning, it can waste tokens on simple requests. We study selective thinking for tool-using LLMs and introduce Adaptive Rejection Sampling (Ada-RS), an algorithm-agn...

---

### 24. [A Multimodal Framework for Aligning Human Linguistic Descriptions with Visual Perceptual Data](https://arxiv.org/abs/2602.19562)

**Authors**: Joseph Bingham  
**Category**: cs.AI  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.19562v1  

#### Abstract
Establishing stable mappings between natural language expressions and visual percepts is a foundational problem for both cognitive science and artificial intelligence. Humans routinely ground linguistic reference in noisy, ambiguous perceptual contexts, yet the mechanisms supporting such cross-modal...

---

### 25. [WANSpec: Leveraging Global Compute Capacity for LLM Inference](https://arxiv.org/abs/2602.18931)

**Authors**: Noah Martin, Fahad Dogar  
**Category**: cs.DC  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.18931v1  

#### Abstract
Data centers capable of running large language models (LLMs) are spread across the globe. Some have high end GPUs for running the most advanced models (100B+ parameters), and others are only suitable for smaller models (1B parameters). The most capable GPUs are under high demand thanks to the rapidl...

---

### 26. [A Formal Framework for Predicting Distributed System Performance under Faults](https://arxiv.org/abs/2602.19088)

**Authors**: Ziwei Zhou, Si Liu, Zhou Zhou, Peixin Wang, MIn Zhang  
**Category**: cs.DC  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.19088v1  

#### Abstract
Today's distributed systems operate in complex environments that inevitably involve faults and even adversarial behaviors. Predicting their performance under such environments directly from formal designs remains a longstanding challenge. We present the first formal framework that systematically ena...

---

### 27. [A Risk-Aware UAV-Edge Service Framework for Wildfire Monitoring and Emergency Response](https://arxiv.org/abs/2602.19742)

**Authors**: Yulun Huang, Zhiyu Wang, Rajkumar Buyya  
**Category**: cs.DC  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.19742v1  

#### Abstract
Wildfire monitoring demands timely data collection and processing for early detection and rapid response. UAV-assisted edge computing is a promising approach, but jointly minimizing end-to-end service response time while satisfying energy, revisit time, and capacity constraints remains challenging. ...

---

### 28. [Learning Beyond Optimization: Stress-Gated Dynamical Regime Regulation in Autonomous Systems](https://arxiv.org/abs/2602.18581)

**Authors**: Sheng Ran  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.18581v1  

#### Abstract
Despite their apparent diversity, modern machine learning methods can be reduced to a remarkably simple core principle: learning is achieved by continuously optimizing parameters to minimize or maximize a scalar objective function. This paradigm has been extraordinarily successful for well-defined t...

---

### 29. [Communication-Efficient Personalized Adaptation via Federated-Local Model Merging](https://arxiv.org/abs/2602.18658)

**Authors**: Yinan Zou, Md Kamran Chowdhury Shisher, Christopher G. Brinton, Vishrant Tripathi  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.18658v1  

#### Abstract
Parameter-efficient fine-tuning methods, such as LoRA, offer a practical way to adapt large vision and language models to client tasks. However, this becomes particularly challenging under task-level heterogeneity in federated deployments. In this regime, personalization requires balancing general k...

---

### 30. [CTS-Bench: Benchmarking Graph Coarsening Trade-offs for GNNs in Clock Tree Synthesis](https://arxiv.org/abs/2602.19330)

**Authors**: Barsat Khadka, Kawsher Roxy, Md Rubel Ahmed  
**Category**: cs.LG  
**Published**: 2026-02-24  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.19330v1  

#### Abstract
Graph Neural Networks (GNNs) are increasingly explored for physical design analysis in Electronic Design Automation, particularly for modeling Clock Tree Synthesis behavior such as clock skew and buffering complexity. However, practical deployment remains limited due to the prohibitive memory and ru...

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
