# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-01 07:57:34 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Step-level Optimization for Efficient Computer-use Agents](https://arxiv.org/abs/2604.27151)

**Authors**: Jinbiao Wei, Kangqi Ni, Yilun Zhao, Guo Gan, Arman Cohan  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.27151v1  

#### Abstract
Computer-use agents provide a promising path toward general software automation because they can interact directly with arbitrary graphical user interfaces instead of relying on brittle, application-specific integrations. Despite recent advances in benchmark performance, strong computer-use agents r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Step-level Optimization for Efficient Computer-use Agents**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

当前的 **computer-use agents** 虽然在多步 GUI 任务中表现出色，但通常依赖于在每一步都调用大型多模态模型（如 GPT-5、Claude Sonnet），导致推理成本高昂、延迟高，难以部署到实际生产环境中。这种“均匀计算分配”策略效率低下，因为：

- 多数交互步骤是常规操作，可由小型模型处理；
- 错误集中在少数“高风险时刻”，如：
  - **Progress stalls**：重复动作、无进展循环；
  - **Silent semantic drift**：行为看似合理，但已偏离用户目标。

### ✅ 提出了什么新方法或新思路

提出一种 **event-driven, step-level cascade** 框架，实现细粒度的自适应计算调度：

- 默认使用 **small policy** 执行动作；
- 引入两个轻量级、基于文本的 **learned monitors** 来检测风险事件，仅在必要时触发对 **stronger model** 的调用：
  - **Stuck Monitor**：检测近期行为是否陷入停滞（如重复点击、无效重试）；
  - **Milestone Monitor**：识别语义上有意义的任务检查点，触发稀疏验证。

该框架将“始终启用大模型”的模式转变为 **on-demand compute allocation**，实现了动态、高效的资源利用。

### ✅ 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **效率** | 显著降低大模型调用频率，减少成本与延迟 |
| **模块化** | 可作为插件叠加在已有 agent 上，无需修改底层架构或重新训练大模型 |
| **通用性** | 适用于不同 small/large model 组合，不依赖特定任务启发式规则 |
| **稳定性** | 使用滑动窗口 + 回滞机制避免频繁切换（thrashing） |

相比 query-level cascading 或固定间隔验证，本方法更贴合 GUI 任务的动态性和语义结构。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集

- **OSWorld**  
  - 面向桌面环境的真实操作系统基准测试（Ubuntu VM）
  - 包含浏览器、办公软件、终端、代码编辑器等多样化应用
  - 平均轨迹长度约 25 步，适合长视野任务分析

- **WebArena**（使用 WebArena-Verified 版本）  
  - 面向网页交互的多站点任务基准
  - 涉及登录、搜索、表单填写、购物等多种真实场景
  - 轨迹较短（平均 ~10 步），更具挑战性

> 两者覆盖了 computer-use agents 的两大典型应用场景。

### ⚙️ 实验设置和评估指标

#### 模型配置
| 类型 | 小模型（Small Policy） | 大模型（Large Policy / Verifier） |
|------|------------------------|-------------------------------|
| OSWorld | Qwen3-VL-8B, EvoCUA-8B | Claude Sonnet 4.5, Kimi K2.5 |
| WebArena | gpt-oss-20b, AgentTrek-32B | GPT-5 mini, GPT-5.2 |

#### 评估指标
- **Success Rate (Acc.)**：任务完成率
- **Cost/Task**：单个任务的推理成本（美元）
- **Latency / Req.**：请求响应时间（秒）
- **Large Model Usage**：
  - `A2 Share`：大模型执行步骤占比
  - `Switched`：触发大模型的任务数量

#### 控制变量
- 所有 cascaded 设置均以 small model 为默认策略
- Monitors 基于 ModernBERT 训练，输入仅为文本轨迹（rationale + action），不涉及图像或 DOM 差异

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据（来自 Table 1 & Table 2）

| Setting | Acc. | Cost/Task | Latency | A2 Share |
|--------|------|-----------|---------|----------|
| **EvoCUA-8B (small only)** | 43.3% | $0.022 | 2.6s | 0% |
| **Kimi K2.5 (large only)** | 60.1% | $0.132 | 8.3s | 100% |
| **EvoCUA-8B + Kimi K2.5 (ours)** | **58.2%** | **$0.051** | **4.5s** | **40.5%** |

| Setting | Acc. | Cost/Task | Latency | A2 Share |
|--------|------|-----------|---------|----------|
| **gpt-oss-20b (small only)** | 30.3% | $0.005 | 3.9s | 0% |
| **GPT-5.2 (large only)** | 60.1% | $0.335 | 19.6s | 100% |
| **gpt-oss-20b + GPT-5.2 (ours)** | **57.8%** | **$0.211** | **12.2s** | **66.9%** |

> ✅ 在 OSWorld 上，我们的方法达到接近最强 large model 的准确率（58.2% vs 60.1%），但成本下降 **61.4%**，延迟降低 **45.8%**。

> ✅ 在 WebArena 上，成功率达 57.8%，接近 GPT-5.2 的 60.1%，但节省了 **37% 成本** 和 **37.8% 延迟**。

### 🔁 与基线方法的对比结果

| 对比项 | 结果 |
|-------|------|
| vs. Always-small | 显著提升成功率（+15~27个百分点） |
| vs. Always-large | 接近其性能（差距 <3%），但大幅降低成本（最高节省 **74.6%**） |
| vs. Fixed-interval verification | 更高效且更准确（见 Table 3） |

#### ▶️ 固定间隔 vs 事件驱动（Table 3）

| Policy | OSWorld Acc. | Cost | WebArena Acc. | Cost |
|-------|--------------|------|----------------|------|
| Periodic-k | 55.1% | $0.07 | 52.5% | $0.24 |
| **Stuck + Milestone (ours)** | **58.2%** | **$0.05** | **58.8%** | **$0.21** |

> 事件驱动不仅更便宜，在准确性上也显著优于周期性验证，尤其在短轨迹的 WebArena 中优势明显。

### 🔍 消融实验结果（Figure 3）

| 设置 | OSWorld Acc. | WebArena Acc. |
|------|-------------|---------------|
| No detector | 43.3% | 30.3% |
| + Stuck only | 49.6% | 44.9% |
| + Milestone only | 48.9% | 47.6% |
| **+ Both (full system)** | **58.2%** | **58.8%** |

> ✅ 两种 monitor 具有**互补作用**：
> - **Stuck Monitor** 主要缓解局部失败（如重复点击）；
> - **Milestone Monitor** 抓住语义漂移（silent drift）；
> - 二者结合带来最大增益，说明设计合理。

此外发现：
- 最终性能更多取决于 **large model 的能力上限**；
- small model 主要影响成本，决定何时需要升级。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **GUI 轨迹具有高度异质性**：并非所有步骤都需要强模型介入，多数为常规操作。
2. **失败集中于两类模式**：
   - **Progress stalls**：表现为动作重复、轨迹变长（failed episodes 平均比成功长 2.8×）；
   - **Silent semantic drift**：模型发出 `done` 但实际未完成任务（图 2c）。
3. **事件驱动 cascade 显著改善 cost-quality trade-off**：
   - 成功率接近 always-large policy；
   - 大模型使用减少至 ~40–67%；
   - 成本降低最多达 **74.6%**，延迟降低 **45.8%**。
4. **轻量级文本 monitor 足够有效**：
   - Milestone detector 达到 94.1% 准确率，F1=62.0%
   - Stuck detector 达到 93.9% 准确率，F1=91.5%
   - 表明无需图像即可捕捉关键信号

### ⚠️ 方法的局限性

- **Monitor 训练依赖 LLM 标注**：虽可通过已有轨迹生成标签，但仍需高质量监督信号；
- **Threshold tuning 需权衡**：需人工选择 `θ_s`, `θ_m` 来平衡 cost 与 success；
- **无法完全防止 drift**：若 milestone 判定错误，仍可能错过关键偏差；
- **对 very short tasks 效益有限**：若总步数少，节省空间小。

### 🔮 未来工作方向

1. **自动 threshold adaptation**：根据任务复杂度动态调整触发阈值；
2. **multi-level cascade**：引入多个中间模型形成梯度式升级路径；
3. **end-to-end joint training**：联合优化 small policy 与 monitor；
4. **扩展至其他 modalities**：融合 accessibility tree、DOM 结构增强 monitor 判断；
5. **real-world deployment study**：在企业级系统中测试吞吐量与稳定性表现。

---

## ✅ 总结

本文提出的 **step-level cascade** 是迈向高效、可部署 computer-use agents 的重要一步。它通过引入 **Stuck Monitor** 和 **Milestone Monitor** 实现了精细化的风险感知调度，在保持高性能的同时大幅降低了推理开销。其实验充分、设计实用，具备良好的模块化与泛化能力，为 future agentic systems 的工程落地提供了清晰路径。

</details>

---

### 2. [Length Value Model: Scalable Value Pretraining for Token-Level Length Modeling](https://arxiv.org/abs/2604.27039)

**Authors**: Zhen Zhang, Changyi Yang, Zijie Xia, Zhen Yang, Chengzhi Liu, Zhaotiao Weng, Yepeng Liu, Haobo Chen, Jin Pan, Chenyang Zhao, Yuheng Bu, Alkesh Patel, Zhe Gan, Xin Eric Wang  
**Category**: cs.CL  
**Published**: 2026-05-01  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.27039v1  

#### Abstract
Token serves as the fundamental unit of computation in modern autoregressive models, and generation length directly influences both inference cost and reasoning performance. Despite its importance, existing approaches lack fine-grained length modeling, operating primarily at the coarse-grained seque...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Length Value Model: Scalable Value Pretraining for Token-Level Length Modeling**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代自回归模型（如LLMs和VLMs）以 **token** 为基本生成单位，生成长度直接影响推理成本（计算、KV缓存、延迟）和任务性能（如推理深度）。然而，现有方法对生成长度的建模大多停留在 **sequence-level**（如提示词控制、训练时加惩罚项），缺乏对 **token-level 生成动态** 的细粒度建模。

本文指出，当前缺少一个能够实时预测“剩余生成长度”的 **token-level 长度信号**，从而限制了对生成过程的精细控制与效率优化。

---

### **提出的新方法：Length Value Model (LenVM)**

作者提出了 **Length Value Model (LenVM)** ——一种将生成长度建模为 **value estimation 问题** 的 token-level 框架。

#### **核心思想**
- 将每个生成的 token 视为一次环境交互，赋予其一个固定的负奖励 $ r = -(1-\gamma) $。
- 定义从当前解码状态 $ s_t $ 开始的 **discounted return**：
  $$
  G_t = -\sum_{i=0}^{L-t-1} \gamma^i (1-\gamma) = -(1 - \gamma^{L-t})
  $$
  其中 $ L $ 是总生成长度，$ \gamma \in (0,1) $ 是折扣因子。
- $ G_t \in (-1, 0) $ 是一个有界、单调递增函数，严格对应于剩余生成长度 $ L-t $。
- LenVM 的目标是训练一个值头（value head），预测该 discounted return，作为剩余生成长度的代理信号。

#### **创新点**
1. **首次将长度建模形式化为 value learning 问题**，引入标准强化学习中的 return 和 Bellman 方程框架。
2. **构建了一个 annotation-free、dense、unbiased、scalable 的监督信号**：
   - 无需人工标注；
   - 每个非终止 token 都提供监督信号（dense）；
   - 基于实际采样轨迹，无偏估计；
   - 可随模型规模、prompt 数量、completion 数量自然扩展。
3. **支持多种下游应用**：长度控制、长度预测、性能-效率权衡、生成动态解释。

---

### **相比现有方法的优势**

| 维度 | 现有方法 | LenVM |
|------|--------|-------|
| **建模粒度** | Sequence-level（粗粒度） | Token-level（细粒度） |
| **是否需要微调** | 多数需 fine-tuning 或 retraining | 仅需附加 value head，不修改 base model |
| **控制精度** | 依赖 prompt 指令，难以精确匹配 | 支持硬约束（Equal To）下的高精度控制 |
| **可扩展性** | 依赖标注或复杂训练流程 | 自动构造监督信号，易于大规模预训练 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在多领域混合数据上训练通用 LenVM，避免 domain-specific 过拟合：

| 数据集 | 领域 | 规模 |
|-------|------|-----|
| **OpenCodeReasoning-2** | 编程 | 1.42M prompts |
| **WildChat** | 指令遵循 | 529k chats |
| **DeepMath-103K** | 数学推理 | 103k problems |

每条 prompt 采样多个 completions（最多16个），用于构建密集的 token-level 回归目标。

---

### **实验设置与评估指标**

#### **模型架构**
- 在 Qwen2.5 和 Qwen3 系列 LLM/VLM 上构建 LenVM。
- 添加一个两层 MLP 作为 value head，输出经 sigmoid 映射到 $(-1, 0)$ 区间。

#### **训练目标**
最小化 token-level 的均方误差（MSE）：
$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N \frac{1}{L^{(n)}} \sum_{t=0}^{L^{(n)}-1} (V_\theta(s_t^{(n)}) - G_t^{(n)})^2
$$

#### **评估任务与指标**

| 任务 | 指标 | 说明 |
|------|------|------|
| **长度控制生成** | **Length Score (LS↑)**, **Length Deviation (LD↓)** | 在 LIFEBench 上测试 Equal To / At Most / At Least 三种约束 |
| **性能-效率权衡** | **Pass@1 vs. Avg. Length 曲线** | 对比 LenVM 引导 vs. 硬截断（hard budget） |
| **生成长度预测** | **Mean Relative Error (MRE↓)** | 从 prompt 初始状态预测最终生成长度 |
| **可扩展性分析** | **Validation Loss** | 随模型大小、prompt 数、completion 数变化 |

---

### **基线方法对比**
- **Closed-source frontier models**（GPT-4o, GPT-5, Claude, Gemini）：使用 prompt-based 控制。
- **Hard token budget baseline**：达到预算即截断，视为失败。
- **No LenVM baseline**：原始模型直接生成。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **长度控制生成（LIFEBench）**
| 模型 | Length Score (↑) | Length Deviation (↓) |
|------|------------------|------------------------|
| **Qwen2.5-7B-Instruct (w/o LenVM)** | 30.9 | 71% |
| **+ LenVM (1.5B)** | **64.8** (+33.9) | **44%** (-27pp) |

> **显著优于所有闭源模型**（如 GPT-4o 得分仅 35.5，偏差 74%），表明 LenVM 能实现更精确的 token-level 控制。

#### ✅ **性能-效率权衡（GSM8K @ ~200 tokens）**
| 方法 | Pass@1 |
|------|--------|
| Hard token budget | ~6% |
| **LenVM-guided decoding** | **~63%** |

> 在相同平均长度下，准确率提升超过 **10倍**，说明模型内部存在更短的成功路径，LenVM 成功引导模型选择这些高效路径。

#### ✅ **生成长度预测（Prompt-boundary）**
| 模型大小 | 数学 | 编程 | 指令遵循 |
|---------|------|------|----------|
| 1.5B | 17.0% | 29.0% | 33.0% |
| **32B** | **9.8%** | **14.9%** | **17.1%** |

> MRE 随模型规模持续下降，证明 LenVM 具备良好的 **scalability**。

#### ✅ **可扩展性验证**
- 图3显示，validation loss 在三个维度上一致下降：
  - 模型参数量 ↑
  - 训练 prompt 数 ↑
  - 每个 prompt 的 completion 数 ↑
> 表明 LenVM 是一个理想的 **scalable value pretraining objective**。

---

### **消融实验结果**

| 设计选择 | 发现 |
|--------|------|
| **Target 表示方式** | Discount Return + Sigmoid > Log Length > Normalized Length > Raw Length |
| **Batch 构造策略** | Fully shuffled > grouped batching（后者轻微损害泛化） |
| **Discount factor $\gamma$** | 大 $\gamma$ 更适合早期预测，小 $\gamma$ 更适合末期；实践中取中间值平衡 |
| **数值精度（FP16/BF16/FP32）** | 性能几乎无差异，说明 LenVM **数值稳定** |

---

## **4. 关键结论和发现**

### **主要发现**

1. ✅ **生成长度可以被有效建模为 token-level value signal**  
   LenVM 成功将长度建模纳入标准 value learning 框架，实现了高精度、可解释的 token-level 长度估计。

2. ✅ **LenVM 是一个强大且灵活的 inference-time 控制工具**  
   - 支持硬约束下的精确长度控制（如“必须生成恰好512个token”）；
   - 支持软控制下的性能-效率平滑权衡（通过 exponential tilting）；
   - 可用于调度系统的长度预测（如 early-bird 推理调度）。

3. ✅ **LenVM 提供可解释的生成动态视角**  
   通过分析 TD residual，发现某些 token（如 `ah`, `but`, `wait`）常伴随“转向更长推理”，而另一些（如 `therefore`, `perfect`, ✅ emoji）则标志“即将结束”。

4. ✅ **LenVM 具备良好可扩展性**  
   损失随模型大小、数据量、completion 数量单调下降，适合大规模 value pretraining。

---

### **方法的局限性**

1. **推理延迟增加**：每次解码需额外 forward pass 查询 LenVM，带来额外 latency。
2. **依赖 rollout policy 分布**：训练数据来自固定 generator 的采样，可能无法完全泛化到其他策略。
3. **未进行 RL fine-tuning**：虽然理论上可用于 PPO 中作为 value baseline 或 shaping reward，但本文仅验证 inference-time 应用。
4. **长度反演存在系统性低估**（见 Appendix F）：由于 Jensen 不等式，从预测值反推期望长度会偏低。

---

### **未来工作方向**

1. **将 LenVM 集成到 RL 训练中**：
   - 作为 length-specific value baseline；
   - 作为 potential-based reward shaping 信号，改善 credit assignment。
2. **探索更高效的推理架构**：如 distill LenVM into base model，减少额外计算开销。
3. **跨模型迁移**：训练通用 LenVM 并迁移到不同 family 的 LLMs 上。
4. **结合其他价值信号**：构建 multi-objective value model（如 quality + length + safety）。

---

> **总结一句话**：  
> **LenVM 首次将生成长度建模为 token-level value estimation 问题，提出了一种 annotation-free、dense、scalable 的预训练框架，在长度控制、预测、效率优化等方面展现出强大能力，为未来 LLM 的精细化推理控制提供了新范式。**

</details>

---

### 3. [Early Detection of Water Stress by Plant Electrophysiology: Machine Learning for Irrigation Management](https://arxiv.org/abs/2604.28038)

**Authors**: Eduard Buss, Till Aust, Heiko Hamann  
**Category**: cs.LG  
**Published**: 2026-05-01  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.28038v1  

#### Abstract
Purpose: Fast detection of plant stress is key to plant phenotyping, precision agriculture, and automated crop management. In particular, efficient irrigation management requires early identification of water stress to optimize resource use while maintaining crop performance. Direct physiological se...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Early Detection of Water Stress by Plant Electrophysiology: Machine Learning for Irrigation Management》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本研究旨在解决**精准农业中水资源管理效率低下的问题**，特别是传统灌溉系统无法实时感知植物真实水分状态，导致过度或不足灌溉。作者聚焦于**早期检测植物水分胁迫（water stress）**，以实现基于植物生理反馈的智能灌溉决策。

### 🚀 提出的新方法与创新点
1. **构建了一个基于植物电生理信号（electrophysiological signals）的在线水胁迫检测框架**：
   - 利用自研传感器节点 **PhytoNode** 非侵入式采集番茄植株的电势差（EDP）。
   - 将其应用于温室环境下的实际灌溉管理场景，推动“植物物联网”（Internet of Plants, IoP）落地。

2. **提出了一套完整的机器学习处理流程**，包含：
   - **tsfresh 特征提取**：自动提取约 700 个统计时序特征。
   - **AutoML 流程优化**：使用 NaiveAutoML 自动选择最优分类器及超参数，减少人工调参依赖。
   - **Sequential Backward Selection (SBS)** 进行多变量特征选择，压缩模型复杂度而不损失性能。
   - **Temperature Scaling 概率校准**：提升预测置信度的可靠性，支持更安全的自动化控制。

3. **验证了跨个体泛化能力**：
   - 在训练集中排除部分植株作为独立测试集，证明模型可推广至未见个体。

4. **引入分级灌溉制度**（graded irrigation regimes），更贴近现实农业条件，而非简单“断水 vs 正常”。

### 🔍 相比现有方法的优势
| 方面 | 本文优势 |
|------|--------|
| **传感方式** | 使用低功耗、太阳能驱动、无线通信的 **PhytoNode**，优于仅依赖环境传感器的传统系统；同时验证了商业设备 PhytlSigns 的有效性 |
| **建模策略** | 采用 **AutoML + 特征工程** 路径，在多数时间窗口下表现优于端到端 **Deep Learning** 方法（如 CNN、InceptionTime、Mamba） |
| **实用性增强** | 引入概率校准和 LOWESS 平滑分析，使模型输出更具解释性和决策参考价值 |
| **时间窗口权衡分析** | 明确指出 **30 分钟回看窗口** 是响应速度与分类精度的最佳平衡点 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **对象**：16 株温室种植的番茄（*Solanum lycopersicum*）
- **周期**：连续记录 18 天（2025年6月4日至6月22日）
- **灌溉分组**（第5天起）：
  - **饱和组**（saturated）：持续供水
  - **理想组**（400 mL/day）：正常灌溉
  - **轻度缺水组**（200 mL/day）
  - **重度缺水组**（100 mL/day）
- **数据类型**：
  - **Electrical Differential Potential (EDP)**：采样频率 10 Hz → 下采样为 1 Hz
  - **土壤湿度**：使用电容式传感器监测（0.1 Hz）
- **公开数据**：所有原始与处理后数据已发布于 Zenodo（DOI: [10.5281/zenodo.18873964](https://doi.org/10.5281/zenodo.18873964)）

### ⚙️ 实验设置
- **时间窗口划分**：将信号切分为固定长度的时间段用于分类：
  - `1 min`, `5 min`, `30 min`, `1 h`, `6 h`
- **标签策略**：
  - **Binary Classification**（主任务）：
    - Healthy：前3天数据
    - Stressed：最后几天中除400mL组外的所有处理组
  - **Multiclass Classification**（辅助尝试）：
    - 区分 healthy / overwatered / underwatered
- **训练/验证/测试划分**：
  - 排除两个植株（分别来自过量和100mL组）作为独立测试集
  - 其余按 80%/20% 分为训练集和验证集，并进行 **Group K-Fold** 交叉验证防止数据泄露

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 分类准确率（主要指标） |
| **AUPRC**（Area Under Precision-Recall Curve） | 衡量模型在不同阈值下的排序能力，尤其适用于类别平衡变化场景 |
| **Brier Score** | 预测概率与真实标签之间的均方误差 |
| **Adaptive Calibration Error (ACE)** | 衡量预测置信度与实际频率的一致性 |
| **Reliability Diagrams** | 可视化校准效果 |
| **LOWESS Smoothing** | 分析压力发展趋势与过渡时间点 |

### 🆚 基线方法对比
| 方法 | 类型 | 是否使用 |
|------|------|---------|
| **Histogram Gradient Boosting (HGB)** | Ensemble Tree（由 AutoML 选出） | ✅ 主要方法 |
| **Extra Trees / Random Forest** | Tree-based Ensemble | ❌ 因概率校准兼容性被排除 |
| **CNN** | Deep Learning | ✅ 对比之一 |
| **InceptionTime** | Ensemble CNN for time series | ✅ 对比之一 |
| **Mamba** | Selective State Space Model (SSM) | ✅ 新兴架构尝试 |
| **Optuna** | Hyperparameter Optimization Framework | ✅ 用于 DL 超参搜索 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Binary Classification）

| 时间窗口 | AutoML (HGB) Test Accuracy | 最佳 DL Test Accuracy | 性能差距 |
|----------|----------------------------|------------------------|--------|
| **1 min** | 62.6% | 63.56% (CNN) | ≈ +1% |
| **5 min** | 76.3% | 70.99% (InceptionTime) | -5.3% |
| **30 min** | **83.2%** | 80.21% (Mamba) | -3.0% |
| **1 h** | 84.0% | 84.77% (CNN) | +0.8% |
| **6 h** | **89.6%** | **96.96%** (CNN) | +7.4% |

> ✅ **最高测试准确率达 96.96%**（CNN @ 6h），但综合稳定性仍推荐 HGB。

#### 🏆 综合最佳配置：
- **时间窗口：30 分钟**
- **模型：HGB（经 AutoML 选定）**
- **Test Accuracy：83.2%**
- **Validation Accuracy：91.0%**
- **AUPRC（test）：0.871**

### 🔍 与基线方法对比结果
| 维度 | 结果 |
|------|------|
| **AutoML vs Deep Learning** |  
| - 整体趋势 | HGB 在大多数窗口下更稳定，尤其在 5–30 min 区间显著优于 DL |
| - 训练准确性 | HGB 达到 100%，而 DL 普遍低于 85% |
| - 泛化能力 | HGB 在 test set 上表现更一致；DL 在长窗口（6h）虽高但波动大（std >11%） |
| - 平均测试性能 | CNN 略优（+2%），但 HGB 更鲁棒（std <2%） |

| **Multiclass Classification** |
|------------------------------|
| 所有模型在训练/验证集上表现良好（>90%），但在测试集上崩溃（~48%），说明难以区分具体胁迫类型，仅能有效识别“是否胁迫” |

### 🔁 消融实验结果
| 模块 | 影响 |
|------|------|
| **Feature Selection (SBS)** |
| - 初始特征数：~670 → 精简至 96–200 |
| - 性能影响：最大下降仅 **1.7%（test acc）**，表明大量冗余特征存在 |
| - 示例：30 min 窗口保留 182 特征即可达到 82.5% 准确率（vs 全量 83.2%） |
| **Mutual Information (MI) 预筛选** | 成功将维度从 ~670 降至 200，提高 SBS 效率 |
| **Probability Calibration (Temperature Scaling)** |
| - 温度参数 T > 1 → 模型原生过自信（overconfident）
| - 校准后 ACE 显著降低：
  - 1–30 min 窗口：ACE 从 ~4–6% → **<1%**
  - 提升了预测可信度的实际可用性 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **植物电生理信号可用于早期水胁迫检测**，早于可见症状出现。
2. **30分钟时间窗口是最佳折衷方案**：
   - 平衡了快速响应（较短延迟）与高分类性能（83.2% test accuracy）。
3. **基于 AutoML 的特征工程 pipeline 优于端到端深度学习**：
   - 尽管某些 DL 架构在特定条件下略胜一筹，但 HGB 更加稳健、可解释性强。
4. **模型具备跨个体泛化能力**：
   - 在未参与训练的植株上成功检测到健康→胁迫状态转变，表明响应具有刺激特异性而非个体特异。
5. **状态转换可在灌溉改变后约 4 天内被检测到**（100mL 和 200mL 组），而对照组无此趋势。
6. **概率校准至关重要**：
   - 未经校准的模型严重过自信，可能误导自动控制系统；温度缩放显著改善可靠性。

### ⚠️ 局限性
1. **实验环境受限**：
   - 仅在温室中对单一作物（番茄）进行测试，尚未验证于田间、其他物种或季节变化。
2. **胁迫类型识别能力有限**：
   - 多分类任务失败，目前只能区分“健康 vs 胁迫”，不能判断胁迫原因（如干旱、盐碱、虫害等）。
3. **生理意义尚需验证**：
   - “50% 置信度转折点”是否对应真实的生理临界点，仍需结合植物生物学指标（如气孔导度、叶水势）进一步确认。
4. **样本量较小**：
   - 每组仅 4 株植物，统计效力有限，结果应谨慎外推。

### 🔮 未来工作方向
1. **扩展应用场景**：
   - 应用于更多作物种类、田间环境、多种非生物/生物胁迫（如病害、营养缺乏、高温）。
2. **闭环灌溉控制系统开发**：
   - 将该框架嵌入自动灌溉系统，形成 **biofeedback-driven irrigation control loop**，实现实时调控。
3. **边缘部署与轻量化模型**：
   - 将模型部署至 PhytoNode 微控制器（类似 Aust et al. 2025 中 Mbed Torch Fusion OS），实现本地推理。
4. **融合多模态传感数据**：
   - 结合 sap flow、stem diameter variation、multispectral imaging 等提升检测精度。
5. **动态决策阈值机制**：
   - 根据农业生产目标（节水优先 or 产量最大化）自适应调整分类阈值（via PR 曲线调节）。

---

> 💡 **一句话总结**：  
> 本文提出并验证了一个基于植物电生理信号与 AutoML 的水胁迫早期检测系统，实现了高达 83.2% 的跨个体分类准确率，揭示了 **30 分钟回看窗口 + HGB + 概率校准** 的最优组合，为构建可持续、资源高效的智能灌溉系统提供了关键技术基础。

</details>

---

### 4. [ChipLingo: A Systematic Training Framework for Large Language Models in EDA](https://arxiv.org/abs/2604.27415)

**Authors**: Lei Li, Xingwen Yu, Jianguo Ni, Junxuan Zhu, Jieqiong Zhang, Jian Zhao, Zhi Liu  
**Category**: cs.LG  
**Published**: 2026-05-01  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.27415v1  

#### Abstract
With the rapid advancement of semiconductor technology, Electronic Design Automation (EDA) has become an increasingly knowledge-intensive and document-driven engineering domain. Although large language models (LLMs) have shown strong general capabilities, applying them directly to EDA remains challe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《ChipLingo: A Systematic Training Framework for Large Language Models in EDA》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

电子设计自动化（Electronic Design Automation, EDA）是一个高度**知识密集型**和**文档驱动**的工程领域，其工具链复杂、命令繁多、版本更新快。尽管大型语言模型（LLM）在通用任务中表现出色，但在直接应用于 EDA 场景时面临以下三大挑战：

- **领域专业知识不足**（Insufficient Domain Expertise）：通用 LLM 缺乏对 EDA 工具命令、流程约束等细节的理解。
- **跨工具知识混淆**（Cross-Tool Knowledge Confusion）：不同 EDA 工具间语法差异大，模型容易混淆命令用法。
- **RAG 能力退化**（RAG Capability Degradation）：经过领域训练后，模型更依赖参数化知识，忽视检索到的外部文档，导致 Retrieval-Augmented Generation（RAG）效果下降。

---

### 提出的新方法或新思路

本文提出 **ChipLingo** —— 一个面向 EDA 领域的系统性 LLM 训练框架，包含三个阶段：

1. **Domain-Adaptive Pretraining（领域自适应预训练）**
   - 构建多源融合的 EDA 领域语料库，并引入 **QA-Augmented Pretraining**（问答增强预训练），即在预训练阶段就加入 QA 格式数据，使模型早期学习“知识”与“任务”的关联。

2. **Instruction Alignment Training（指令对齐训练）**
   - 利用高质量 QA 数据进行监督微调（Supervised Fine-Tuning），提升模型理解指令和执行任务的能力。
   - 数据通过强模型（如 GPT-4）蒸馏重写，确保推理链完整、表达规范。

3. **RAG Scenario Fine-Tuning（RAG 场景微调）**
   - 显式构建三种 RAG 场景数据用于训练：
     - 正确检索（利用上下文）
     - 无关检索（忽略噪声）
     - 不完全检索（结合内部知识与片段信息）
   - 以缓解领域训练后的 **RAG 能力退化**问题。

此外，提出 **Partial FT** 参数训练策略：冻结底层若干层参数，仅更新部分高层参数，在保留通用能力的同时实现有效领域适配。

---

### 相比现有方法的优势

| 对比维度 | ChipLingo 的优势 |
|--------|----------------|
| **训练范式** | 系统化三阶段流程，覆盖从知识获取到 RAG 能力建模全过程，优于单一阶段适配。 |
| **数据构造** | 引入 QA-Augmented Pretraining，相比纯文本预训练更能促进任务导向的知识吸收。 |
| **RAG 稳定性** | 显式 RAG 场景训练可恢复并增强模型对外部知识的利用能力，解决“越训越不用检索”的退化现象。 |
| **参数效率与性能平衡** | Partial FT 在领域性能与通用能力保留之间取得更好权衡，优于 LoRA 和 Full FT。 |

---

## 2. 核心实验方法和设置

### 使用的数据集

- **预训练语料来源**：
  - EDA 技术文档（46.7%）
  - 工程师问答记录（18.9%）
  - 教材与技术论文（8.0%）
  - 脚本与使用示例（11.8%）
  - 其他资源（14.6%）
- 总规模超过 **20万页技术文档**，涵盖 **50+ EDA 工具**，覆盖逻辑综合、物理实现、仿真验证、DFT 等多个设计阶段。
- 通过多种方式增强数据：
  - QA 生成
  - 文档重写（Rewriting）
  - Cloze / MCQ 生成

- **指令微调数据集**：
  - 约 40,000 条高质量 QA 对，经强模型蒸馏与清洗，格式为 `(question, reasoning, answer)`。

- **RAG 训练数据**：
  - 使用 Algorithm 1 自动生成三种场景样本：
    - 正确检索（`C_rel`）
    - 无关检索（`C_irr`）
    - 部分检索（`C_partial`）

- **评估基准：EDA-Bench**
  - 自建内部 benchmark，计划公开。
  - 包含数千个来自真实工程场景的短答案问题。
  - 覆盖四类典型 EDA 工具：
    - Logic Synthesis Tools
    - Physical Implementation Tools
    - Simulation Verification Tools
    - Design-for-Test (DFT) Tools

---

### 实验设置和评估指标

#### 主要模型
- **ChipLingo-8B** 和 **ChipLingo-32B**：基于 Qwen3 系列进行训练。
- 所有模型在同一 EDA-Bench 上测试。

#### 评估指标
- **EDA-Bench Accuracy**：使用 LLM-as-a-Judge 方法自动评分（如 GPT-4 或同级模型判断预测答案是否与标准答案一致），辅以人工抽样验证一致性。
- **RAG Gain Rate (Δrag)**：衡量正确检索带来的准确率提升。
- **Noise Impact Degree (Δnoise)**：衡量无关检索对性能的干扰程度。
- **通用能力保留指标**：
  - IFEval（指令遵循）
  - SimpleQA（事实问答）
  - HumanEval（代码生成）

---

### 基线方法对比

| 类别 | 基线模型 |
|------|---------|
| 开源通用模型 | Qwen3-8B, Qwen3-32B, DeepSeek-v3.2 |
| 行业专用模型 | ChipExpert, ChipNeMo（相关工作） |
| 商业闭源模型 | GPT-4, Claude-Sonnet-4.5 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| Model | Parameters | EDA-Bench Accuracy |
|-------|------------|--------------------|
| Qwen3-8B | 8B | 26.85% |
| Qwen3-32B | 32B | 36.30% |
| DeepSeek-v3.2 | 671B (37B active) | 56.28% |
| **ChipLingo-8B** | **8B** | **59.70%** ✅ |
| **ChipLingo-32B** | **32B** | **70.02%** ✅ |
| Claude-Sonnet-4.5 | – | 71.11% |
| GPT-4 | – | 72.35% |

> 💡 **结论**：  
> - ChipLingo-8B 以 8B 参数超越更大规模的 DeepSeek-v3.2（56.28% → 59.70%）。  
> - ChipLingo-32B 接近顶尖商业模型水平（70.02% vs 71–72%）。

---

### 与基线方法的对比结果

- **显著优于同规模基础模型**：
  - ChipLingo-8B 较 Qwen3-8B 提升 **+32.85个百分点**（绝对值）。
- **超越部分超大规模通用模型**：
  - ChipLingo-8B > DeepSeek-v3.2（尽管后者参数更多）。
- **接近商用闭源模型表现**：
  - ChipLingo-32B 仅落后 Claude/GPT-4 约 2%，展现出极强的领域适配潜力。

---

### 消融实验结果

#### （1）QA-Augmented Pretraining 效果

| 预训练策略 | 最终准确率 |
|-----------|----------|
| 仅文档 | 48.8% |
| 加入简单 QA | 53.2% |
| 多种增强混合（Rewriting + QA + Cloze/MCQ） | **59.7%** ✅ |

> ✅ **结论**：QA 数据对领域性能提升最为显著；多策略联合增强效果最佳。

#### （2）参数训练策略比较（基于 Qwen3-8B）

| 方法 | EDA-Bench | IFEval | SimpleQA | HumanEval | General Avg. |
|------|----------|--------|----------|-----------|-------------|
| Base | 26.85% | 87.6 | 36.6 | 87.8 | 70.7 |
| LoRA | 46.8% | 85.2 | 32.4 | 83.6 | 67.1 |
| Full FT | 61.0% | 82.1 | 28.7 | 79.2 | 63.3 |
| **Partial FT** | **59.7%** | **85.8** | **33.8** | **84.5** | **68.0** ✅ |

> ✅ **结论**：
> - Full FT 虽然领域性能最高，但严重损害通用能力。
> - **Partial FT 在领域性能与通用能力保留之间达到最优平衡**。
> - LoRA 在此类知识密集型任务中表现有限，可能因低秩更新无法充分表达复杂知识结构。

#### （3）RAG 能力退化与恢复（关键发现）

| 模型状态 | No Retrieval | +Correct Retrieval (Δrag) | +Irrelevant Retrieval (Δnoise) |
|---------|--------------|----------------------------|-------------------------------|
| Qwen3-8B | 24.5% | 31.8% (**+7.3**) | 23.1% (-1.4) |
| +DAP | 48.2% | 42.7% (**-5.5**) ❌ | 43.0% (-5.2) |
| +DAP+SFT | 52.1% | 48.3% (**-3.8**) ❌ | 47.5% (-4.6) |
| **+DAP+SFT+RAG** | **59.7%** | **64.8% (+5.1)** ✅ | **57.4% (-2.3)** ✅ |

> 🔍 **关键观察**：
> - 经过领域训练（DAP/SFT）后，**正确检索反而降低性能**（负增益），说明模型产生“parametric bias”，过度依赖内部知识。
> - 引入 RAG 场景训练后，**正向增益恢复至 +5.1**，且抗噪能力提升（噪声影响从 -4.6 降至 -2.3）。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **系统性训练框架有效提升 EDA 领域能力**：ChipLingo 三阶段流程显著优于通用模型和简单微调方法。
2. ✅ **QA-Augmented Pretraining 更利于知识-任务对齐**：在预训练阶段引入 QA 数据能加速领域知识掌握。
3. ✅ **Partial FT 是高效折中方案**：在保持通用能力的前提下实现良好领域适配，优于 LoRA 和 Full FT。
4. ✅ **RAG 能力会因领域训练而退化**：这是知识密集型垂直领域的重要现象，不能被忽视。
5. ✅ **显式的 RAG 场景训练可修复退化**：通过模拟多样检索条件，模型重新学会合理利用外部知识。

---

### 方法的局限性

- **工具知识混淆风险**：多工具联合训练虽有协同效应，但也可能导致命令误用（如将 Tool-A 的命令用于 Tool-B 的问题）。
- **数据依赖性强**：性能提升依赖高质量、多样化的领域语料，数据稀疏子领域改进有限。
- **当前仍聚焦单轮问答**：尚未扩展至多步 Agent 协作或全流程自动化。
- **EDA-Bench 尚未公开**：限制第三方复现与横向比较。

---

### 未来工作方向

1. **向 EDA Agent 演进**：从单轮 QA 向多步任务协调、工具调用、反馈闭环等 **harness capabilities** 发展。
2. **扩大任务覆盖范围**：纳入更多 EDA 子任务，如功耗优化、时序违例修复、DRC 修复建议等。
3. **持续扩展训练数据规模**：进一步提升模型泛化能力和长尾问题处理能力。
4. **完善并发布 EDA-Bench**：作为独立研究贡献推动社区发展。
5. **探索动态 RAG 控制机制**：让模型自主判断何时应依赖检索、何时使用内部知识（类似 Self-RAG）。

--- 

> 📌 **总体评价**：  
> ChipLingo 不仅提出了一个高性能的 EDA 领域 LLM 训练框架，更重要的是揭示了**知识密集型领域中 LLM 适配的关键规律**——尤其是 RAG 能力退化这一重要现象及其解决方案，为构建下一代 EDA 智能系统奠定了坚实基础。

</details>

---

### 5. [From Coarse to Fine: Benchmarking and Reward Modeling for Writing-Centric Generation Tasks](https://arxiv.org/abs/2604.27453)

**Authors**: Qingyu Ren, Tianjun Pan, Xingzhou Chen, Xuhong Wang  
**Category**: cs.CL  
**Published**: 2026-05-01  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.27453v1  

#### Abstract
Large language models have achieved remarkable progress in text generation but still struggle with generative writing tasks. In terms of evaluation, existing benchmarks evaluate writing reward models coarsely and fail to measure performance from the perspective of specific requirements. In terms of ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Coarse to Fine: Benchmarking and Reward Modeling for Writing-Centric Generation Tasks*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

当前在**生成式写作任务**（writing-centric generation tasks）中存在两大核心挑战：

- **评估层面**：现有的 **Reward Model Evaluation Benchmarks**（如 RewardBench、JudgeBench）采用粗粒度（coarse-grained）评估方式，仅从知识、推理、安全等宏观维度打分，无法衡量模型对**具体写作要求**（如格式、风格、长度、内容约束）的遵循能力。
  
- **训练层面**：主流训练方法要么依赖昂贵且可能有偏的 **LLM-as-a-judge** 范式，要么基于流畅性（fluency）、连贯性（coherence）等粗略属性训练 Reward Model，缺乏对“是否满足指令中的细粒度要求”的精准建模。

这导致现有方法难以有效提升模型在复杂写作任务中的表现。

---

### 🚀 提出的新方法与创新思路

作者提出了两个核心组件，形成从**评估到训练**的闭环系统：

#### （1）**WEval**：细粒度写作奖励模型评估流水线（Fine-grained Evaluation Pipeline）

- **核心思想**：通过在原始查询中**逐步丢弃（drop out）写作要求**，构建多个候选响应，并依据“保留要求越多 → 质量越高”这一自然逻辑，形成**黄金排序（golden ranking）**。
- 例如，一个包含4个要求的查询会生成4个响应（分别满足4、3、2、1个要求），其理想排序即为按要求数量递减。
- 使用 **Correlation、IL（Instruction-Level）、PL（Prompt-Level）** 等相关性指标衡量 Reward Model 排序与黄金排序的一致性。

> ✅ 创新点：首次实现基于**需求依从性**（requirement adherence）的细粒度、可量化的 Reward Model 评估体系。

#### （2）**WRL**：细粒度强化学习训练框架（Fine-grained Reinforcement Learning Framework）

- **核心机制**：
  - 构造正负样本对：原始完整响应为 `chosen`，通过**选择性丢弃部分要求**生成的响应为 `rejected`。
  - 使用 **Bradley-Terry (BT) Loss** 进行训练，使 Reward Model 对更符合要求的响应赋予更高分数。
  - 将训练好的 Reward Model 用于 **Group Relative Policy Optimization (GRPO)**，指导策略模型（policy model）优化。

> ✅ 创新点：将“需求是否被满足”转化为可学习的奖励信号，实现**细粒度、可解释、低成本**的 Reward Modeling 与 RL 训练。

---

### 🔍 相比现有方法的优势

| 维度 | 现有方法 | 本文方法（WEval + WRL） |
|------|--------|--------------------------|
| **评估粒度** | 宏观、任务级（task-level）判断 | 细粒度、需求级（requirement-level）排序 |
| **评估依据** | 二元偏好或整体质量打分 | 自然形成的 partial order（黄金排序） |
| **训练信号** | LLM-as-a-judge 或粗略属性（如 fluency） | 显式的 requirement adherence 建模 |
| **成本与偏差** | 高计算开销，受 judge LLM 能力影响大 | 无需外部 judge，可控、可复现 |
| **泛化性** | 多局限于特定任务 | 在多种写作领域和 out-of-domain 任务上均有效 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### （1）**评估数据集构建来源**
- **LMSYS-1M**：约100万条真实用户与LLM对话。
- **WildChat**：65.2万条 GPT-3.5/GPT-4 对话记录。
- **PRISM**：8,000 条精心策划的偏好对话语料。

从中筛选出适用于写作任务的 seed instructions，并人工标注原型示例以聚类生成任务子集。

#### （2）**下游评测基准（In-Domain）**
- **WritingBench**：涵盖6大领域、100子领域的综合性写作基准。
- **LongWriter**：评估长文本生成质量（相关性、准确性、连贯性等）。
- **Arena-Write**：基于人类偏好的真实写作任务比较（win rate）。

#### （3）**跨域泛化测试（Out-of-Domain）**
- **DeepResearch Bench-RACE**：评估科研报告生成能力。
- **FINDER_DEFT**：检查研究生成报告是否满足详细格式与内容清单。

---

### 📊 实验设置与评估指标

#### （1）**Reward Model 评估指标（WEval）**

| 指标 | 含义 |
|------|------|
| **Correlation** | Reward Model 排序与黄金排序之间的 Spearman 相关性 |
| **IL (Instruction-Level)** | 所有 item 中相对位置一致的比例 |
| **PL (Prompt-Level)** | 完全匹配黄金排序的 prompt 占比 |

#### （2）**Policy Model 评估指标**
- **WritingBench**：平均得分（标准化至100分制）
- **Arena-Write**：胜率（win rate）vs 强基线
- **LongWriter**：多维评分（Relevance, Accuracy, Coherence, Clarity, Breadth & Depth, Reading Experience）

#### （3）**基线方法对比**

| 类型 | 基线模型 |
|------|---------|
| **闭源模型** | GPT-4o, o1-Preview, Claude-3.5-Sonnet, Gemini-2.5-Pro |
| **开源通用模型** | Qwen3, Llama-3, DeepSeek-R1 |
| **写作增强模型** | LongWriter-llama3.1-8B, LongWriter-glm4-9B, LongWriter-Zero-32B |
| **训练范式对比** | SFT, LLM-as-a-judge RL, WRL with 其他 RM |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）**WRL 显著提升各类模型写作能力（见 Table 2 & 3）**

| 模型 | +WRL 提升（WritingBench） | 是否超越强基线？ |
|------|----------------------------|------------------|
| Qwen2.5-1.5B-Instruct | +5.5 → 达 50.1 | 超越 LongWriter-glm4-9B |
| Qwen2.5-7B-Instruct | +7.4 → 达 64.4 | 超越同规模写作专用模型 |
| Distill-Qwen-14B | +6.0 → 达 66.7 | 当前最优开源之一 |
| Qwen3-8B | +1.3 → 达 76.2 | **超过 GPT-4o 和 o1-Preview** |

> 💡 特别值得注意的是：**Qwen3-8B + WRL** 在 WritingBench 上得分 **76.2**，高于 GPT-4o（75.5）和 o1-Preview（68.6）。

#### （2）**人类偏好胜率显著提高（Arena-Write）**

- Qwen2.5-7B-Instruct-WRL：胜率提升 **4.6%**
- Distill-Qwen-14B-WRL：胜率提升 **8.7%**（最大增益）

表明模型输出更符合人类审美与实用需求。

#### （3）**多维写作质量全面提升（LongWriter）**

| 模型 | 总分提升 | 关键维度改进 |
|------|--------|-------------|
| Qwen2.5-7B-Instruct-WRL | 80.8 → 82.9 | Coherence (+1.9), Clarity (+2.3), Depth (+4.6) |
| Distill-Qwen-14B-WRL | 88.4 → 89.0 | 持续优化空间仍存 |

说明 WRL 不仅提升表面合规性，也增强了深层表达能力。

---

### 🔬 消融实验结果（Ablation Study, Table 5）

| 训练方法 | WritingBench | Arena-Write |
|---------|--------------|-------------|
| Baseline (无训练) | 57.0 | 21.51 |
| + SFT | 59.4 | 16.47 ❌下降 |
| + LLM-as-Judge RL | 62.1 | 24.70 |
| + WRL (Skywork RM) | 61.2 | 23.53 |
| **+ WRL (Our-RM-7B)** | **64.4** ✅ | **26.05** ✅ |

> ✅ 结论：
> - WRL 效果优于 SFT 和 LLM-as-a-judge RL；
> - 使用本文训练的 **Our-RM-7B** 作为 Reward Model 效果最佳；
> - Reward Model 的评估排名（见 Table 4）与下游 RL 表现高度一致 → **WEval 具有强预测性**。

---

### 🔄 泛化性验证（Generalizability）

#### （1）跨教师模型鲁棒性（Table 6 & 7）
- 使用 **GPT-4o** 和 **Gemini-2.5-Pro** 重构评估数据集后，Our-RM-7B 依然排名第一。
- 表明 WEval 的评估结果不依赖于特定 teacher model，具有高鲁棒性。

#### （2）跨任务迁移能力（Figure 4）
- 在 **DeepResearch Bench-RACE** 和 **FINDER_DEFT** 上，WRL 均带来稳定增益：
  - 最高达 **+4.5 分**（RACE）
  - 清单通过率提升最高达 **+14.3%**（DEFT）

证明该方法不仅适用于普通写作，也能迁移到复杂的科研写作场景。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **细粒度评估是可行且必要的**：
   - WEval 成功构建了基于 requirement dropout 的黄金排序，实现了对 Reward Model 的精细化评估。
   - 人类标注者的结果与 WEval 高度一致，验证了其有效性。

2. **细粒度 Reward Modeling 更高效**：
   - 相比 LLM-as-a-judge 或粗粒度属性建模，基于 requirement adherence 的 BT 训练能提供更强的学习信号。
   - Our-RM-7B 在 Correlation（94.6）、IL（97.3）、PL（78.0）三项指标上全面领先（接近人类水平）。

3. **评估与训练形成正向循环**：
   - WEval 的评估结果能准确预测 Reward Model 在实际 RL 中的表现。
   - 形成了“好 RM → 好 policy → 好写作”的可信闭环。

4. **方法具备强泛化能力**：
   - 在不同参数规模、架构、任务类型上均有效。
   - 可迁移到 out-of-domain 的 deep research 写作任务。

---

### ⚠️ 局限性（Limitations）

1. **未在超大规模模型（≥32B）上验证**：目前实验集中在 1.5B~32B 参数范围。
2. **覆盖的需求类型有限**：尽管已涵盖 Style、Format、Length、Content 四类，但仍未能穷尽现实世界中所有复杂写作约束。
3. **依赖高质量 seed instructions**：若初始指令本身模糊或冲突，会影响 dropout 构造的有效性。

---

### 🔮 未来工作方向

1. **扩展需求类型覆盖范围**：引入更多动态、交互式、多轮写作约束。
2. **自动化 requirement extraction**：减少对人工定义 constraint 类型的依赖。
3. **应用于其他生成任务**：如代码生成、数学解题、法律文书撰写等同样强调指令遵循的任务。
4. **探索自监督 requirement dropout 机制**：降低数据构造成本，提升可扩展性。

---

> 🧩 **一句话总结**：  
> 本文提出 **WEval + WRL** 框架，首次实现面向写作任务的**细粒度 Reward Model 评估与训练闭环**，实验证明其不仅能显著提升模型写作能力，且评估结果具有强预测性和泛化性，为下一代指令遵循系统的构建提供了新范式。

</details>

---

### 6. [AutoREC: A software platform for developing reinforcement learning agents for equivalent circuit model generation from electrochemical impedance spectroscopy data](https://arxiv.org/abs/2604.27266)

**Authors**: Ali Jaberi (Clean Energy Innovation Research Center, National Research Council Canada, Mississauga, ON, Canada), Yonatan Kurniawan (Department of Material Science and Engineering, University of Toronto, Toronto, ON, Canada), Robert Black (Clean Energy Innovation Research Center, National Research Council Canada, Mississauga, ON, Canada), Shayan Mousavi M. (Clean Energy Innovation Research Center, National Research Council Canada, Mississauga, ON, Canada), Kabir Verma (Cheriton School of Computer Science, University of Waterloo, Waterloo, ON, Canada), Zoya Sadighi (Clean Energy Innovation Research Center, National Research Council Canada, Mississauga, ON, Canada), Santiago Miret (Lila Sciences, San Francisco, CA, USA), Jason Hattrick-Simpers (Department of Material Science and Engineering, University of Toronto, Toronto, ON, Canada)  
**Category**: cs.LG  
**Published**: 2026-05-01  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.27266v1  

#### Abstract
This paper introduces AutoREC, an open-source Python package for developing reinforcement learning (RL) agents to automatically generate equivalent circuit models (ECMs) from electrochemical impedance spectroscopy (EIS) data. While ECMs are a standard framework for interpreting EIS data, traditional...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《AutoREC: A software platform for developing reinforcement learning agents for equivalent circuit model generation from electrochemical impedance spectroscopy data》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统从 **Electrochemical Impedance Spectroscopy (EIS)** 数据中识别 **Equivalent Circuit Models (ECMs)** 的过程严重依赖领域专家的手动试错，主观性强、效率低，难以扩展到高通量或自主实验系统（如 self-driving laboratories, SDLs）。这成为自动化电化学研究流程中的瓶颈。

### 提出了什么新方法或新思路
本文提出了 **AutoREC** ——一个开源的 Python 软件平台，用于开发基于 **Reinforcement Learning (RL)** 的智能体来自动生成 ECMs。其核心创新在于：

- 将 ECM 构建建模为一个 **Markov Decision Process (MDP)**，将电路拓扑的逐步构建视为一个序列决策问题。
- 使用 **Double Deep Q-Network (DDQN)** 结合 **Prioritized Experience Replay (PER)** 进行训练，并引入专门的 **dead-loop mitigation** 策略来应对大量无效动作导致的学习停滞问题。
- 采用基于 **Gene Expression Programming (GEP)** 的线性染色体编码方式表示电路拓扑，支持灵活且可变复杂度的电路生成。

### 相比现有方法的优势
| 方法类别 | 局限性 | AutoREC 的优势 |
|--------|------|---------------|
| **手动拟合** | 主观、耗时、不可扩展 | 自动化、可重复、适用于大规模数据分析 |
| **分类模型 (如 SVM, CNN)** | 需要标注数据；只能从预定义库中选择电路；无法发现新结构 | 无需标注数据；不限于固定库；能探索并生成新的电路拓扑 |
| **进化算法 (如 GEP)** | 搜索空间大、缺乏指导、效率低、易陷入局部最优 | RL 提供奖励信号引导搜索方向，提高采样效率和收敛速度 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **合成数据集 (Synthetic Dataset)**  
  基于文献 [27] 生成，包含 5 种不同复杂度的参考 ECMs（如 Randles 模型及其嵌套组合），共 1500 条 EIS 谱（每类 300 条），按 9:1 划分为训练/测试集。
  
- **实验数据集 (Experimental Datasets)**  
  包括真实世界的 EIS 数据，涵盖以下系统：
  - 锂离子电池（battery）
  - 腐蚀（corrosion）
  - 氧气析出反应（OER）
  - CO₂ 还原系统（CO2 reduction）

### 实验设置和评估指标
- **环境设计**：
  - **State Space**：由编码后的电路染色体（one-hot 向量）与 EIS 特征向量（实部、虚部、幅值、相位等归一化后拼接）共同构成。
  - **Action Space**：在染色体特定位置替换为某个元件（R, L, P）或操作符（+, /），允许动态改变电路结构。
  - **Reward Function**：多层级奖励机制，综合考虑：
    - 拟合质量（基于 `log-B loss` 和 χ²）
    - 终止条件奖励（达到足够好的拟合）
    - 对无效动作的大惩罚
    - 对模型复杂度（元件数量、并联深度）的惩罚以鼓励简洁性
    - 中间奖励用于策略塑形（reward shaping）

- **训练算法**：
  - 使用 **DDQN + PER**
  - 引入 **dead-loop mitigation**：当连续执行相同无效动作或重复进入同一状态超过阈值时，强制限制后续动作仅从有效集中选取，防止学习卡死。
  - 超参数通过 **Optuna + TPE** 自动优化。

- **评估指标**：
  - **Success Rate**：成功终止并生成高质量拟合 ECM 的 episode 比例。
  - **Average Reward**：最后若干 episodes 的平均累积奖励。
  - **Topology Match Rate**：生成 ECM 是否与参考电路一致（仅用于合成数据）。
  - 在实验数据上采用 **trajectory-based evaluation**：运行完整 episode，从中挑选最佳拟合模型（最低 χ²），而非依赖固定终止阈值。

### 基线方法对比
文中未直接与其他 ML 或进化方法进行端到端性能比较，而是通过消融实验（ablation study）验证自身关键组件的有效性：
- **有无 dead-loop mitigation 的对比**
- 不同超参数配置的影响（见附录）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 数据类型 | Success Rate | 其他关键表现 |
|--------|------------|-------------|
| **合成数据（测试集）** | >99.6% | 成功恢复所有参考电路的基本结构，具备强泛化能力 |
| **实验电池数据** | ~65.5% | 能合理拟合多数样本的半圆特征和扩散尾 |
| **腐蚀/OER 数据** | 定性良好 | 拟合效果较好，尤其 OER 中捕捉到高频微小半圆 |
| **CO₂ 还原数据** | 较弱 | 拟合较差，表明对重叠/压缩特征处理能力有限 |

### 与基线方法的对比结果
- **Dead-loop Mitigation 消融实验**（图6a-b）：
  - 启用该策略后，学习曲线显著加快，最终 reward 更高。
  - 成功率从较低水平提升至 >99.6%，证明其对稳定训练至关重要。

- **电路演化分析**（图7, 图S2-S6）：
  - 所有 episode 均以串联电阻起始。
  - 第一步几乎总是引入并联分支，作为构建 Randles 子电路的基础，体现物理直觉。
  - 后续逐步精细化，如将电阻替换为 CPE，形成容抗行为。

- **复杂度依赖性分析**（图8）：
  - 简单电路（单 Randles 元素）学习快、成功率上升早。
  - 复杂电路（三 Randles 元素）需要更长训练时间，且依赖先掌握简单结构的策略，显示策略的层次性和迁移性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **AutoREC 是一个有效的 RL 平台**：成功将 ECM 自动生成转化为 MDP 问题，实现了高效、自动化的电路构造。
2. **RL 方法具有高度灵活性与泛化能力**：
   - 可生成不在训练集中的新拓扑（如 OER 中含三个 CPE 的结构），优于分类法。
   - 在多种真实电化学系统中展现出良好的适应性。
3. **结构非唯一性是固有问题**：即使拟合良好，也可能存在多个等效电路解释（如嵌套 vs 串行 Randles 单元），这并非模型失败，而是 EIS 本身的物理模糊性所致。
4. **dead-loop mitigation 显著提升训练效率**：解决了无效动作主导 replay buffer 的问题，是实用化训练的关键。

### 方法的局限性
1. **倾向于生成冗余元件**（图6c）：
   - 有时添加对拟合影响极小的额外支路或 CPE，增加模型复杂度却不提升解释力。
   - 表明当前奖励函数对“简约性”的控制仍不足。

2. **训练数据多样性不足限制泛化**：
   - 实验数据中常见的压缩/重叠半圆、过渡态特征在合成数据中缺失。
   - 导致在 CO₂ 还原等复杂场景下表现不佳。

3. **学习不平衡问题**：
   - 简单拓扑学得快，复杂拓扑需更长时间，均匀采样导致资源分配不均。

4. **评估挑战**：
   - 缺乏真实 ECM 标签使得实验数据上的定量评估困难。
   - 固定终止阈值可能误导 success rate（附录 S3 显示其对阈值敏感）。

### 未来工作方向
1. **改进奖励函数设计**：
   - 更系统地惩罚模型复杂度（如结合 BIC/AIC 准则）。
   - 引入物理合理性约束（如参数符号、量级范围）。

2. **优化训练策略**：
   - 采用课程学习（curriculum learning）或优先采样复杂样本，平衡不同难度任务的经验分布。

3. **丰富训练数据**：
   - 构建更具多样性的合成数据集，覆盖过渡态、部分重叠响应、非理想行为等。

4. **增强后处理能力**：
   - 开发自动简化模块，识别并移除低影响元件，提升模型可解释性。

5. **集成到 SDL 工作流**：
   - 将 AutoREC 与实验规划、数据采集模块联动，实现完全闭环的自主电化学表征。

---

> **总体评价**：  
> AutoREC 并非提出单一“终极”RL 模型，而是构建了一个**开放、模块化的开发平台**，推动 RL 在 ECM 自动生成领域的系统性研究。它展示了 RL 在解决此类结构化生成问题上的巨大潜力，同时也揭示了可靠性、简洁性和泛化能力等方面的挑战，为后续研究指明了清晰路径。

</details>

---

### 7. [Auto-FlexSwitch: Efficient Dynamic Model Merging via Learnable Task Vector Compression](https://arxiv.org/abs/2604.28109)

**Authors**: Junqi Gao, Dazhi Zhang, Zhichang Guo, Biqing Qi, Yi Ran, Wangmeng Zuo  
**Category**: cs.LG  
**Published**: 2026-05-01  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.28109v1  

#### Abstract
Model merging has attracted attention as an effective path toward multi-task adaptation by integrating knowledge from multiple task-specific models. Among existing approaches, dynamic merging mitigates performance degradation caused by conflicting parameter updates across tasks by flexibly combining...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Auto-FlexSwitch: Efficient Dynamic Model Merging via Learnable Task Vector Compression

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
动态模型合并（Dynamic Model Merging）通过在推理时灵活组合多个任务特定模型的知识，有效缓解了静态合并中因任务间参数更新冲突导致的性能下降。然而，现有动态合并方法需要为每个任务存储独立的任务向量（task vectors），导致**存储开销巨大**，限制了其在资源受限场景下的应用。

本文旨在解决这一**存储效率瓶颈**，提出一种高效且可学习的任务向量压缩与动态合并框架。

---

### 🚀 提出的新方法与创新思路

作者提出了一个完整的、端到端可学习的轻量化任务向量构建与动态合并框架，主要包括以下四个核心组件：

#### （1）**T-Switch**：基于规则的轻量级任务向量表示
- 将任务向量分解为三个紧凑组件：
  - **Activation Switch**（二值稀疏掩码）：指示激活位置
  - **Polarity Switch**（符号向量）：表示参数极性
  - **Switch Knob**（标量缩放因子）：恢复整体幅度
- 实现高达 **16× 存储压缩**，同时保持高保真度近似。

#### （2）**Auto-Switch**：无需训练的动态合并机制
- 在推理时，基于输入特征与预存的小规模查询集（query set）进行 KNN 检索。
- 自动计算各任务开关的组合权重，实现**免训练、自适应的模型融合**。

#### （3）**FlexSwitch**：可学习的任务向量压缩框架
- 引入 **Learnable Gating Sparsification (LGS)**：通过可学习阈值和温度控制的 Sigmoid 函数实现**可微分稀疏化**，联合优化稀疏率与幅度校准。
- 引入 **Bit-width Adaptive Selection (BAS)**：为不同模块分配最优量化位宽（1/2/4/8 bit），实现细粒度量化。
- 设计 **Sparsity-Aware Storage Strategy (SASS)**：采用分组 COO 格式并自适应选择分组大小，最大化利用稀疏性以降低实际存储占用。

#### （4）**Auto-FlexSwitch**：最终的动态合并方案
- 集成 FlexSwitch 的压缩能力与一种带**可学习低秩度量的 KNN 推理机制**。
- 通过学习一个低秩投影矩阵 $ L \in \mathbb{R}^{r \times e} $ 来增强任务判别性，提升检索准确性。

---

### 🔍 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **存储效率** | 显著优于所有基线，压缩比可达 **40–200×**，远超传统方法（如 Ties-Merging、AdaMerging）。 |
| **性能表现** | 在多种任务上达到甚至超越独立微调（Individual Fine-tuning）和多任务学习（MTL）的平均精度。 |
| **灵活性与通用性** | 支持 Transformer 和 CNN 架构，在图像分类、目标检测、自然语言理解等多领域均有效。 |
| **训练成本** | **无需额外训练路由模块**，仅需少量示例样本即可完成部署，训练时间仅为 MTL 的 **1/10 左右**。 |
| **可扩展性** | FlexSwitch 可作为独立的 LLM 微调权重压缩工具，具备广泛应用潜力。 |

---

## 2. 核心实验方法和设置

### 📚 数据集

#### 图像分类任务（Image Classification）
- **8个视觉数据集**：SUN397, Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD
- **骨干网络**：ViT-B/32, ViT-L/14, ConvNeXt

#### 目标检测任务（Object Detection）
- **RoboFlow 100 (RF100)** 中的 7 个跨域检测任务：Aerial, Videogames, Microscopic, Underwater, Documents, Electromagnetic, Real World
- **骨干网络**：DETR-ResNet50

#### 自然语言理解任务（NLU）
- **GLUE Benchmark** 中的 7 项任务：CoLA, SST-2, MRPC, QQP, MNLI, QNLI, RTE
- **骨干网络**：RoBERTa-base, Mamba-130M

---

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| **评估指标** | - 图像分类：Accuracy (%)<br>- 目标检测：mAP@50 (%)<br>- NLU：Accuracy / MCC<br>- 存储开销：MB |
| **对比维度** | 性能 vs. 存储开销的帕累托前沿（Pareto frontier） |
| **随机性处理** | 所有含随机性的方法报告三次独立运行的平均值 |
| **超参搜索** | 对所有基线方法进行网格搜索以确定最优配置（见 Table II, VI, VIII） |

---

### 🆚 基线方法对比

| 类型 | 方法列表 |
|------|--------|
| **静态合并** | Weight-Averaging, Task-Arithmetic, DARE, Ties-Merging, Fisher Merging, DF-Merge, RegMean |
| **动态合并** | AdaMerging, AdaMerging++, EMR-Merging, Twin-Merging, MoW-Merging |
| **其他** | Pre-trained, Individual Fine-tuned, Traditional MTL |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Tables III–X）

#### 图像分类（ViT-B/32）
| 方法 | AVG Acc (%) | Storage (MB) |
|------|-------------|--------------|
| Individual | 91.01 | — |
| MoW-Merging | 87.28 | 2171.89 |
| Auto-Switch | 90.51 | 185.28 |
| **Auto-FlexSwitch+KL** | **91.00** | **41.51** |

> ✅ **结论**：Auto-FlexSwitch 在仅用 **41.51 MB** 的情况下，达到了接近甚至超过独立微调的性能，存储仅为 MoW-Merging 的 **1.9%**。

#### 图像分类（ViT-L/14）
| 方法 | AVG Acc (%) | Storage (MB) |
|------|-------------|--------------|
| Individual | 94.44 | — |
| Twin-Merging | 93.17 | 1780.62 |
| **Auto-FlexSwitch+KL** | **94.48** | **80.69** |

> ✅ 达成 **SOTA 性能 + 22× 存储压缩**

#### 目标检测（DETR）
| 方法 | AVG mAP@50 (%) | Storage (MB) |
|------|------------------|--------------|
| Individual | 49.99 | — |
| Twin-Merging | 38.05 | 313.98 |
| **Auto-FlexSwitch+MSE** | **42.40** | **46.97** |

> ✅ 提升 **4.35% mAP**，同时减少 **85% 存储**

#### NLU（RoBERTa-base）
| 方法 | AVG Score | Storage (MB) |
|------|----------|--------------|
| Individual | 0.8483 | — |
| Twin-Merging | 0.8281 | 670.08 |
| **Auto-FlexSwitch+KL** | **0.8370** | **51.60** |

> ✅ 超越多数基线，存储仅为 Twin-Merging 的 **7.7%**

---

### 🔬 消融实验结果（Table XI & Fig. 9–11）

| 组件 | 影响 |
|------|------|
| **LGS** | 显著降低存储（~44 MB → ~40 MB），维持性能 |
| **BAS** | 进一步提升性能，尤其在高稀疏率下更鲁棒 |
| **LGS + BAS** | 协同效应明显，允许更高稀疏率收敛，总存储更低 |
| **KNN + Learnable Metric** | 明显提升检索准确率，最终性能再增 0.3–0.7 pts |
| **Exemplar Size** | Auto-FlexSwitch 对样本数量不敏感，**100 示例即达稳定性能** |
| **Neighbor Count** | 设置 $ K=10 $ 即可获得最佳平衡 |

> ✅ 各组件均有明确贡献，且系统具有强鲁棒性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Task Vectors 具有“脉冲式”激活特性**  
   - 仅有少数大幅值参数对任务起关键作用，其余可安全剪枝。
   - 支持高稀疏率（>97%）压缩而不显著损失性能。

2. **Task Vectors 对低比特表示高度鲁棒**  
   - 二值化（sign-only）结合 L2 缩放仍能保持性能。
   - 高稀疏下，量化误差进一步减小。

3. **不同模块对稀疏化与量化敏感性异质性强**  
   - 固定全局策略非最优 → 必须引入**可学习、自适应机制**（LGS + BAS）。

4. **SASS 存储策略极大释放稀疏红利**  
   - 当 sparsity > 50%，SASS 显著优于独立存储（Indep）。
   - 最高可达 **>200× 压缩比**（如 sparsity=0.99 时）。

5. **Auto-FlexSwitch 是当前最高效的动态合并方案**  
   - 在性能、存储、训练成本之间取得最佳平衡。
   - 泛化能力强，适用于 Vision、NLP、Detection 多种任务。

---

### ⚠️ 局限性

1. **依赖高质量微调模型作为输入**  
   - 方法本身不参与微调过程，无法纠正原始模型偏差。

2. **低秩度量的学习可能受初始特征空间限制**  
   - 若预训练模型特征区分度差，检索效果仍受限。

3. **目前未探索在线增量合并机制**  
   - 新任务加入需重新优化 FlexSwitch 参数。

4. **极端压缩下个别任务可能出现性能波动**  
   - 如 RoBERTa 上当 $ \alpha=0.9 $ 时 Auto-Switch 性能骤降。

---

### 🔮 未来工作方向（原文提及）

1. **在微调阶段直接鼓励稀疏性与可量化性**  
   - 减少后处理压缩难度与信息损失。

2. **深入研究主流架构中各组件的可量化程度差异**  
   - 为模块化压缩提供理论指导。

3. **探索具身智能等动态场景中的在线组合决策**  
   - 结合能力维度进行实时任务向量调度。

4. **将 FlexSwitch 推广为通用的 LLM 增量权重存储格式**  
   - 实现“一个基础模型 + 多个极小插件”的部署范式。

---

> 💡 **总结一句话**：  
> **Auto-FlexSwitch 通过“可学习稀疏化 + 自适应量化 + 智能存储 + 高效检索”四重创新，实现了高性能与极致存储效率的统一，是迈向实用化动态模型合并的重要一步。**

</details>

---

### 8. [When Your LLM Reaches End-of-Life: A Framework for Confident Model Migration in Production Systems](https://arxiv.org/abs/2604.27082)

**Authors**: Emma Casey, David Roberts, David Sim, Ian Beaver  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.27082v1  

#### Abstract
We present a framework for migrating production Large Language Model (LLM) based systems when the underlying model reaches end-of-life or requires replacement. The key contribution is a Bayesian statistical approach that calibrates automated evaluation metrics against human judgments, enabling confi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When Your LLM Reaches End-of-Life: A Framework for Confident Model Migration in Production Systems*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对**生产环境中基于 Large Language Model (LLM) 的系统在底层模型达到 End-of-Life（EOL）时如何高效、可靠地进行模型迁移**这一现实挑战。  
该问题在企业级应用中日益普遍，原因包括：
- 第三方托管的 LLM 被弃用（如每约12个月一次）
- 新区域部署需求
- 成本优化、延迟改善、多语言支持等

传统手动评估方式成本高、耗时长，难以应对频繁且大规模的迁移任务。

---

### 🚀 提出的新方法与创新思路

提出了一套**结构化、可复现的企业级 LLM 迁移框架**，其核心创新在于：

#### （1）**Bayesian 统计方法校准自动化评估指标**
- 将有限的人工标注数据用于**校准 automated evaluation metrics（如 RAGAS、LLM-as-a-judge）的真实准确率**
- 构建 confusion matrix，估计每个 metric 的 **True Positive Rate (TPR)** 和 **False Positive Rate (FPR)**
- 利用贝叶斯推断生成“真实正确率”的置信区间（confidence interval），从而实现对模型间差异的**统计显著性判断**

> 🔍 关键洞见：并非所有自动指标都可靠；必须通过人工校准才能用于决策。

#### （2）**分阶段过滤机制（Stage-wise Filtering Framework）**
将模型选择过程分解为多个可解释、可追溯的步骤：
1. 内部合规与成本筛选
2. 输出格式兼容性检查（如 XML/JSON schema compliance）
3. 正确性与 IDK 行为比较（使用校准后的 metric）
4. 风格一致性（style adherence）检测
5. 延迟与响应时间评估
6. 区域覆盖与多模态能力综合选型

此流程大幅降低评估复杂度，并确保最终决策具备质量保障与效率平衡。

#### （3）引入 `new_correctness` 自定义 LLM-based metric
- 不仅依赖 ground truth，还结合完整 context 进行判断
- 更贴近实际业务场景中的“正确”定义（如容忍补充信息、识别遗漏关键项）
- 在 human judgment 上表现出最高一致性（见 Table 1）

---

### ⚖️ 相比现有方法的优势

| 方面 | 传统做法 | 本文方法 |
|------|--------|---------|
| 评估方式 | 依赖标准 benchmark（如 SQuAD, HotpotQA）或盲用自动指标 | **校准自动指标 + 贝叶斯不确定性建模** |
| 数据利用 | 大量人工标注 or 完全无监督 | **小样本人工标注即可支撑全局评估** |
| 可复现性 | 缺乏统一标准，主观性强 | **标准化流程 + 可追踪的淘汰依据** |
| 效率 | 手动评估昂贵，无法规模化 | **适用于多产品、多区域、高频次迁移场景** |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 描述 | 规模 | 用途 |
|-------|------|-----|-----|
| **HotpotQA** | 多跳问答数据集，强调推理与证据整合 | 200 examples | 主要测试集之一 |
| **SQuAD** | 单段落抽取式问答基准 | 200 examples | 初始测试，后因缺乏负例被弃用 |
| **Internal Test Set ("basic")** | 企业自建客户支持风格问题集，涉及真实文档上下文 | 51 examples | 核心业务相关性验证 |

> ❗ 注意：SQuAD 因 true-negative 样本不足导致无法有效校准 metric，最终未纳入主分析。

---

### 🧪 实验设置

#### 候选模型池（Candidate Models）
从以下候选中筛选：
- **Anthropic**: Claude 3 Haiku（baseline）、Claude 3.5 Sonnet、Claude 4.5 Haiku
- **AWS**: Nova Micro, Lite, Lite 2, Pro
- **Google**: Gemma 3 (27B)
- **OpenAI**: GPT-OSS (20B, 120B) → 全部淘汰（XML 格式失败）
- **Alibaba**: Qwen3-32B, Qwen3-235B（含 reasoning mode 开关）

所有模型需满足内部合规审查（bias, privacy, licensing, regional availability）。

---

### 📊 评估指标

#### （1）Correctness Metrics
| Metric | 类型 | 特点 |
|-------|------|-----|
| `ragas_correctness` | 工具包内置 | 易误判补充信息为错误 |
| `llm_correctness` | LLM-as-judge | 对比答案与 ground truth 是否矛盾 |
| `new_correctness` ✅ | 自研 LLM judge | 结合 context 判断事实准确性、完整性、用户影响，最接近 human judgment |

#### （2）其他关键维度
| 指标 | 方法 |
|------|------|
| **IDK Rate** | 统计返回“I don’t know”的比例 |
| **Style Violation** | 子串匹配检测：“according to”, “sources”, “knowledge”等模板化表达 |
| **Response Time** | 中位数延迟（seconds） |
| **Word Count** | 输出长度分布 |
| **Regional Availability** | 是否支持 EMEA/APAC/AMER 等地区 |
| **Cost Tier** | 相对价格等级（Low/Middle/High） |

---

### 🔁 基线方法对比
- **Baseline Model**: Claude 3 Haiku（当前线上模型）
- **对比目标**：是否能在 correctness、IDK 控制、style、latency、cost 等方面持平或优于 baseline
- **无直接 baselines（如 Prompt Tuning 工具链）**，而是以“是否改进 baseline prompt”作为 prompt adaptation 实验的目标

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 2 & 5）

#### ✅ 最终入选模型：**Nova 2 Lite** 与 **Qwen3-32B**

| 模型 | Correctness Δ (vs Haiku) | IDK Rate ↓ | Latency (s) ↓ | Style Clean? | 区域支持 | 成本 |
|------|--------------------------|------------|---------------|-------------|-----------|------|
| **Nova 2 Lite** | +4.85% (basic), +4.75% (hotpot) | 2.13% → ↓ | 0.59s → ↓↓ | 是（0% bad style） | 支持多区 | Middle |
| **Qwen3-32B** | +5.62% / +6.39% | 3.19% → ↓ | 0.63s → ↓↓ | 是（2.1% → reasoning 模式升至 7.5%） | 支持多区 | Low |
| Claude 4.5 Haiku | +12.8% | 4.26% | 1.04s | 是 | 部分受限 | High |
| Claude 3.5 Sonnet | +9.89% | 3.19% | 1.73s → ↑↑ ❌ | 是 | 是 | Middle |

> ✅ Nova 2 Lite 和 Qwen3-32B 在 correctness 上优于 baseline，且 IDK 更低、延迟更低、风格合规。

---

### 📉 淘汰原因汇总

| 模型 | 淘汰原因 |
|------|---------|
| OpenAI GPT-OSS 系列 | 无法稳定输出 XML 格式 |
| Nova Micro/Lite | 正确率提升不显著（CI 包含负值） |
| Gemma 3 27B | 正确率无统计显著优势 |
| Claude 3.5 Sonnet | 延迟过高（>1.7s vs baseline ~1.1s） |
| Qwen3-32B (reasoning mode) | 风格违规率从 2.1% 升至 7.5% |

---

### 🔬 Prompt Adaptation 实验结果（Table 4）

尝试三种 prompt 改进策略：
1. **Manual Capitalization**（大写强调指令）
2. **Amazon Bedrock Prompt Optimizer**
3. **MIPROv2 via DSPy**（基于 new_correctness 或 tokenwise f1 训练）

#### 结果总结：
| 方法 | 效果 |
|------|------|
| Manual (capitals) | 有轻微改进趋势，但未达显著水平 |
| MIPROv2 (dspy) | 在 basic 测试集上降低 error，但显著提高 IDK 率（保守倾向）；在 HotpotQA 上 accuracy 下降 |
| **结论** | **原始 prompt 泛化能力惊人地好**，当前 adaptation 方法未能带来一致收益 |

> 💡 发现：Prompt transferability 较强，无需过度调优即可跨模型工作。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **可靠的模型迁移依赖于 aligned evaluation metrics**  
   > 盲目信任公共 benchmark 或通用 metric（如 RAGAS）会导致错误决策。只有经过 human-calibrated 的 metric（如 `new_correctness`）才具备指导意义。

2. **Bayesian 分析使小样本评估具有统计效力**  
   > 即使仅人工标注几十个样本，也能通过 posterior inference 得到 meaningful confidence intervals，支持科学决策。

3. **结构化迁移框架可复用、可扩展**  
   > 提出的六步法适用于任何企业级 LLM 应用迁移，尤其适合管理多个产品线、区域和模型的服务组合。

4. **Prompt 具备较强跨模型迁移能力**  
   > 当前自动化 prompt tuning 工具（如 MIPROv2）尚未能稳定超越原始 prompt，提示 prompt engineering 的 transferability 可能被低估。

5. **最终选定模型全面优于原系统**  
   > Nova 2 Lite 与 Qwen3-32B 在 correctness、latency、cost、region support 上均表现更优，实现平滑升级。

---

### ⚠️ 局限性（Limitations）

| 限制 | 说明 |
|------|------|
| **语言单一** | 实验仅在英文环境下进行，未验证多语言输出质量 |
| **测试集规模小** | 尤其 internal test set 仅 51 条，可能影响统计精度 |
| **LLM-as-judge 单一模型** | 使用 Claude 4 Sonnet 作为 judge model，可能存在 bias |
| **Style 检查较简单** | 仅基于关键词匹配，缺乏深层语义判断 |
| **SQuAD 被排除** | 因缺少 calibration data 导致信息损失 |

---

### 🔮 未来工作方向

1. **推广至其他 LLM 服务**  
   > 将本框架应用于公司内更多基于 LLM 的产品迁移。

2. **增强 QA 评估精度**  
   > 扩大数据集规模、开发更鲁棒的 automated metrics，提升微小变化检测能力。

3. **构建持续评估系统（Continuous Evaluation System）**  
   > 实时监控模型 drift、评估 vendor 更新、预筛 candidate models。

4. **探索混合 judge model 方法（Mixture-of-Models）**  
   > 减少单个 LLM judge 带来的偏见风险。

5. **深入研究 Prompt Transferability**  
   > 探索为何某些 prompt 能跨模型保持高性能，建立可预测的 transfer theory。

---

## 总结一句话

> 本文提出一个**基于 Bayesian 校准与分阶段过滤的 LLM 迁移框架**，解决了企业在模型 EOL 场景下如何**高效、可信地完成生产系统替换**的关键难题，在真实商业 QA 系统中成功落地并选出性能更优的替代模型，同时揭示了当前自动评估与 prompt tuning 技术的实际边界。

</details>

---

### 9. [METASYMBO: Multi-Agent Language-Guided Metamaterial Discovery via Symbolic Latent Evolution](https://arxiv.org/abs/2604.27300)

**Authors**: Jianpeng Chen, Wangzhi Zhan, Dongqi Fu, Junkai Zhang, Zian Jia, Ling Li, Wei Wang, Dawei Zhou  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.27300v1  

#### Abstract
Metamaterial discovery seeks microstructured materials whose geometry induces targeted mechanical behavior. Existing inverse-design methods can efficiently generate candidates, but they typically require explicit numerical property targets and are less suitable for early-stage exploration, where res...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：METASYMBO: Multi-Agent Language-Guided Metamaterial Discovery via Symbolic Latent Evolution

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

传统 **inverse-design** 方法在超材料（metamaterial）设计中依赖明确的数值属性目标（如精确的弹性模量或泊松比），难以处理早期设计阶段常见的模糊、定性语言描述（如“轻质且抗冲击”）。而虽然 **Large Language Models (LLMs)** 能理解自然语言意图，但缺乏几何感知能力，生成的结构常违反物理约束或仅重复已有文献中的设计。

因此，该论文旨在解决以下两个核心挑战：

- **C1：模态鸿沟（Modality Gap）**：如何桥接语言、几何与物理属性三种不同模态之间的语义不一致？
- **C2：设计空间受限**：如何突破训练数据和文献的设计边界，实现对未知假设的有效探索？

---

### 提出了什么新方法或新思路

作者提出 **METASYMBO**，一个基于多智能体协作的框架，用于语言引导的超材料发现，并引入 **symbolic-driven latent evolution** 技术以增强推理时的可控性和创造性。

#### 核心架构（Three-Agent Framework）

| Agent | 功能 | 技术实现 |
|------|------|--------|
| **Designer** | 语言模态解析 | 利用 LLM 解析自由形式语言提示，从文献中检索语义一致的简单结构作为 **scaffold**（语义锚点） |
| **Generator** | 几何模态生成 | 在解耦的潜在空间（disentangled latent space）中合成候选微结构，支持符号化操作进行演化 |
| **Supervisor** | 属性模态评估 | 结合快速 property predictor 和 LLM evaluator 提供反馈，形成闭环优化 |

#### 关键技术创新：Symbolic-Driven Latent Evolution

在 Generator 中引入可编程的符号逻辑算子（symbolic logic operators），直接在潜在空间中对 scaffold 进行组合、修改和精炼：

- **Union**：将 scaffold 的节点结构并入初始结构，实现结构扩展
- **Mix**：在潜在分布间进行加权插值，实现语义融合
- （附录还包含 Intersection 和 Negation）

这些操作不是直接解码，而是通过梯度下降逐步演化潜在变量，确保输出仍在有效流形内。

---

### 相比现有方法的优势

| 维度 | 传统生成模型（如 CDVAE, DiffCSP） | LLM-only 方法 | METASYMBO |
|------|-------------------------------|--------------|-----------|
| 是否接受语言输入 | ❌ 需要数值条件 | ✅ 支持自然语言 | ✅ 支持自然语言 |
| 结构有效性 | ✅ 高（但受限于训练分布） | ❌ 常生成无效结构 | ✅✅ 更高有效性（周期性接近 98%） |
| 设计新颖性 | ⚠️ 受限于训练数据 | ⚠️ 易复现已有设计 | ✅ 支持跨域探索与组合创新 |
| 控制粒度 | ⚠️ 潜在空间纠缠 | ❌ 无控制 | ✅ 解耦控制（lattice, node, edge, semantic） |
| 探索能力 | ❌ 固定采样 | ❌ 黑箱生成 | ✅ 符号驱动的推理时演化 |

> ✅ **核心优势**：首次统一了语言指导、几何生成与属性感知，在保持高结构有效性的同时实现了更强的语言对齐能力和设计多样性。

---

## 2. 核心实验方法和设置

### 使用的数据集

- **MetaModulus Dataset**：包含 9,871 个周期性晶格结构，每个样本包含：
  - 单元胞拓扑（node coordinates, edge connections）
  - 晶格向量（lattice vectors）
  - 机械性能：Young’s modulus, Shear modulus, Poisson’s ratio
- 数据划分：8,000 用于训练，其余用于测试
- 设计提示（Prompts）：由领域专家提供 100 条语言指令，涵盖“高刚度”、“负泊松比”等概念

---

### 实验设置与评估指标

#### 评估维度与指标

| 维度 | 指标 | 定义说明 |
|------|------|---------|
| **Validity（有效性）** | `Vs%`（Symmetry） | 结构中心对称性比率 |
| | `Vp%`（Periodicity） | 是否满足周期性构造条件 |
| **Diversity（多样性）** | `Cov. R.%`（Coverage Recall） | 生成结构覆盖测试集中多少比例的真实结构 |
| | `Repeat Ratio%` | 重复生成相同结构的比例 |
| **Language-Guidance Effectiveness（语言对齐能力）** | `Prompt Guide Score` | 外部 Supervisor（GPT-4.1 + predictor）打分，衡量生成结构是否符合语言意图（0~1） |

---

### 基线方法对比

#### 生成模型类（Geometry-aware Generators）

| 方法 | 类型 | 特点 |
|------|------|------|
| CDVAE | VAE | 支持晶体结构生成，强调周期性 |
| DiffCSP | Diffusion Model | SE(3)-equivariant，适用于晶格预测 |
| SyMat | VAE | 强调对称性约束 |
| Cond-CDVAE | Conditional VAE | 支持属性条件生成 |

#### 大语言模型类（LLMs）

| 方法 | 类型 | 是否专为推理优化 |
|------|------|----------------|
| GPT-4o-mini | 多模态轻量版 | 否 |
| Llama-4-maverick | 开源通用 LLM | 否 |
| Qwen3-235b | 深度思考模型 | 是（chain-of-thought） |
| Deepseek-Reasoning | RL强化推理模型 | 是 |
| Gemini-2.0-flash-lite | Google高效多模态 | 否 |

> METASYMBO 自身也设置了多个变体进行对比，例如使用不同 LLM（GPT-4o vs Gemini）、不同符号算子（Union vs Mix）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | Vs% ↑ | Vp% ↑ | Cov. R.% ↑ | Repeat Ratio% ↓ | Prompt Guide Score ↑ |
|------|-------|-------|------------|------------------|------------------------|
| CDVAE | 57.03 | 0.40 | 55.85 | N/A | N/A |
| Deepseek-Reasoning | 85.5 | 65.30 | 86.9 | 67.7 | 0.4993 |
| **METASYMBO (GPT-4o-mini, Union)** | **91.31** | **98.35** | **98.7** | **7.43** | **0.5531** |
| **METASYMBO (Gemini-2.0, Union)** | 89.65 | 95.97 | 99.2 | 10.07 | 0.4966 |

> ✅ 所有 METASYMBO 变体均显著优于基线

---

### 与基线方法的对比结果

- **结构有效性提升显著**：
  - 对比最强生成模型 CDVAE，**symmetry 提升达 34%**，**periodicity 提升近 98个百分点**
  - 对比最强 LLM（Deepseek-Reasoning），**periodicity 提升超过 33%**

- **语言对齐能力更强**：
  - METASYMBO 平均 **Prompt Guide Score 达 0.52~0.55**，相比最佳 LLM（0.4993）高出约 **6–7%**
  - 表明其能更准确地将“轻质高强”等抽象语言转化为具体结构

- **设计多样性更高**：
  - 所有 METASYMBO 变体 **Coverage Recall > 90%**，表明广泛探索了设计空间
  - **Repeat Ratio < 11%**，远低于多数 LLM（普遍 >60%），说明避免了模式坍塌

---

### 消融实验结果（Ablation Study）

#### 移除各损失项的影响（Table 2）

| 变体 | Vs% | Vp% | Cov. R.% |
|------|-----|-----|----------|
| Full Model | 91.31 | 98.35 | 98.7 |
| w/o `Ls`（语义对齐） | 50.9 | 57.1 | 93.1 |
| w/o `Lp,e`（Sinkhorn对齐） | 47.8 | 45.7 | 93.6 |
| w/o `Lr`（非重叠正则） | 51.6 | 62.8 | 94.3 |
| w/o `Lprior`（先验正则） | 62.8 | 95.1 | —— |

> 🔍 发现：
- `Lp,e`（基于 Sinkhorn 的节点/边软匹配）影响最大，是维持结构一致性关键
- `Ls` 控制语义对齐，移除后语言得分大幅下降
- `Lprior` 和 `Lr` 有助于防止潜在变量漂移，保障生成稳定性

#### 替换为纯 LLM Agent 的消融

- 将 Generator 和 Supervisor 替换为纯 LLM 后，**重复率飙升至 51.20%**，**有效性显著下降**
- 证明：**必须结合 geometry-aware generator 与 property-aware supervisor**

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **多智能体协同有效弥合模态鸿沟**  
   Designer 提供语义锚点，Generator 实现可控生成，Supervisor 提供快速反馈，三者协作形成高效的 language-to-structure pipeline。

2. ✅ **Symbolic-driven latent evolution 实现可编程语义对齐**  
   在解耦潜在空间中应用 Union/Mix 等符号算子，可在推理时动态调整结构语义，支持超越训练数据的假设探索。

3. ✅ **显著优于现有 LLM 与生成模型**  
   在 **结构有效性、语言对齐、设计多样性** 三大维度全面领先，尤其在周期性（Vp%）上接近完美（>98%）。

4. ✅ **真实案例验证实用价值**  
   成功设计出具有 **负泊松比（auxetic）** 和 **高刚度** 的复杂结构，并通过 **FE simulation** 与 **3D printing** 验证其物理可行性。

---

### 方法的局限性

1. **依赖高质量 scaffold 检索**  
   若 Designer 无法正确识别语义对应的经典结构（如误将立方体当作稳定结构），后续生成可能偏离目标。

2. **符号算子仍较基础**  
   当前仅实现 Union/Mix/Intersection/Negation，尚不支持更复杂的程序化规则（如递归、参数化变形）。

3. **潜在空间表达能力限制**  
   尽管采用 disentangled VAE，但潜在空间仍可能无法完全捕捉所有拓扑变化，尤其对于高度非线性的力学响应。

4. **未考虑制造工艺约束**  
   当前框架未集成打印可行性分析（如悬垂角、最小杆径），需下游人工干预。

---

### 未来工作方向

1. **扩展符号算子库**  
   引入更多可微分的几何操作符（如 extrusion, tapering, fractal recursion），提升结构编辑能力。

2. **引入人类专家回路（Human-in-the-loop）**  
   在关键决策点加入专家审核，提高安全性与可靠性，特别是在医疗或航空航天场景。

3. **端到端训练多智能体策略**  
   当前为模块化协作，未来可尝试使用强化学习联合优化三个 agent 的行为策略。

4. **集成制造感知模块**  
   将 AM（Additive Manufacturing）约束建模进 Generator 或 Supervisor，实现“可打印优先”的设计。

5. **拓展至其他材料系统**  
   如多孔陶瓷、复合材料或多尺度结构，验证框架泛化能力。

---

> 📌 **总体评价**：  
> METASYMBO 是首个将 **language guidance、symbolic reasoning、geometric generation** 有机结合的超材料逆向设计框架。它不仅提升了生成质量与可控性，更重要的是推动了 AI 辅助科学发现从“数值优化工具”向“概念级创意伙伴”的转变。

</details>

---

### 10. [MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction](https://arxiv.org/abs/2604.27393)

**Authors**: Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyue Sun, Yingjing Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, Jiancheng Gui, Luoyuan Zhang, Xian Sun, Fuwei Huang, Moye Chen, Zhuo Lin, Hanyu Liu, Qingxin Gui, Qingzhe Han, Yuyang Wen, Huiping Liu, Rongkang Wang, Yaqi Zhang, Hongliang Wei, Chi Chen, You Li, Kechen Fang, Jie Zhou, Yuxuan Li, Guoyang Zeng, Chaojun Xiao, Yankai Lin, Xu Han, Maosong Sun, Zhiyuan Liu, Yuan Yao  
**Category**: cs.CL  
**Published**: 2026-05-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.27393v1  

#### Abstract
Recent progress in multimodal large language models (MLLMs) has brought AI capabilities from static offline data processing to real-time streaming interaction, yet they still remain far from human-level multimodal interaction. The key bottlenecks are no longer modality coverage or latency alone, but...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前多模态大语言模型（MLLMs）虽然在静态图像、视频、语音等任务上取得进展，但其交互范式仍存在两大瓶颈：
- **串行交互模式**：感知（Perception）与响应（Response）被划分为交替阶段，无法实时融合新输入进行动态调整。
- **被动响应机制**：模型仅对显式用户请求做出反应，缺乏主动理解环境并发起行为的能力。

这导致模型难以实现类人水平的**实时、全双工、情境驱动**的多模态交互。

### 🚀 提出的新方法与新思路
作者提出 **MiniCPM-o 4.5**，一个支持**实时全双工全模态交互**（real-time full-duplex omni-modal interaction）的9B参数开源模型，并引入核心技术框架 **Omni-Flow**。

#### 核心创新点：
1. **Omni-Flow 统一流式框架**
   - 将视觉、音频、文本输入与输出沿共享的时间轴对齐，构建连续的全双工交互流程。
   - 打破传统 turn-based 范式，允许模型在“听”、“看”的同时“说”，实现真正的并发处理。

2. **主动行为建模（Proactive Behavior）**
   - 模型可基于持续感知的环境状态自主发起提醒、评论或描述，无需等待用户指令。
   - 如实时解说体育赛事、主动提示异常事件等。

3. **端到端全模态架构设计**
   - 支持 text、image、video、audio 多模态系统提示（multimodal system prompt），包括参考语音输入，实现高质量语音克隆（voice cloning）。
   - 各模块（encoders → LLM backbone → speech decoders）通过 token-level 隐状态连接，支持联合优化。

4. **时间对齐交错生成策略（TAIL: Time-Aligned Interleaving）**
   - 动态控制每段时间窗口内生成的文本量，使语音播放进度紧贴当前环境状态，避免“语音滞后”问题。

### 🔍 相比现有方法的优势
| 方面 | MiniCPM-o 4.5 | 现有主流模型（如 Qwen3-Omni、Gemini 2.5 Flash） |
|------|----------------|---------------------------------------------|
| 交互范式 | 全双工流式交互（full-duplex streaming） | 半双工/回合制（turn-based） |
| 主动性 | 支持主动行为（proactive） | 完全被动响应 |
| 推理效率 | 可部署于 <12GB RAM 边缘设备 | 通常需高端 GPU 或云端部署 |
| 性能表现 | 在 vision-language 和 omni-modal 上接近甚至超越更大模型 |
| 架构灵活性 | 支持切换为传统模式（兼容 MiniCPM-V 4.5 / MiniCPM-o 2.6） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### （1）**Speech Data**
- **大规模自然语音数据**：来自公开来源的数百万小时无标签语音，用于零样本 TTS、ASR 和多说话人对话训练。
- **口语化对话数据**：由 LLM 生成指令遵循对话后，由专业配音演员录制，强调自然表达、情感变化和语速多样性。

#### （2）**Vision-Language Data**
- 基于 MiniCPM-V 4.5 数据体系扩展，涵盖：
  - 高质量图文对（CapsFusion 合成 + 过滤）
  - 文档与 OCR 数据（OmniDocBench，采用相关性感知掩码）
  - 真实场景查询（含 Chain-of-Thought 重写）
  - 密集视频字幕数据（dense video captioning）

#### （3）**Omni-Modal Full-Duplex Data**
- **Web 音视频数据**：过滤掉单人主导、音画不一致或低信息密度片段，保留真实双工交互场景。
- **人工构造全双工任务数据**：用于训练连续场景描述、主动提醒等功能。

---

### ⚙️ 实验设置与评估指标

#### 评估维度四大类：
| 类别 | 子领域 | 代表数据集 |
|------|--------|-----------|
| **Vision-Language Understanding** | STEM推理、通用理解、文档OCR、多图理解、幻觉检测、视频理解 | OpenCompass, MMBench, MathVista, OCRBench, Mantis-Eval, HallusionBench, MLVU |
| **Speech Understanding & Generation** | ASR、语音翻译、音频理解、语音问答、语音合成质量 | AISHELL, LibriSpeech, CoVoST2, VoiceBench, SeedTTS Test, LongTTS, Expresso |
| **Text Capability** | 指令跟随、知识、数学、代码、多语言 | IFEval, MMLU, CMMLU, GSM8K, HumanEval |
| **Omni-modal & Streaming Interaction** | 全模态理解、实时流式交互 | Daily-Omni, WorldSense, LiveSports-3K-CC |

#### 主要评估指标：
- **准确率（Accuracy）**：如 MMBench、OpenCompass 平均得分
- **错误率（Error Rate）**：CER（Character Error Rate）、WER（Word Error Rate）
- **相似度（Similarity）**：SIM-o（Speaker Similarity）
- **主观评分**：AlpacaEval、Expresso（情绪控制能力）
- **效率指标**：first-token latency、throughput（tokens/s）、memory usage（GB）、RTF（Real-Time Factor）

---

### 🆚 基线方法对比
主要对比模型包括：
- **Gemini 2.5 Flash**（闭源小模型标杆）
- **Qwen3-Omni-30B-A3B**（30B 参数全模态模型）
- **InternVL3.5-8B**（同规模多模态模型）
- **CosyVoice2**（先进语音合成系统）
- **StreamingVLM / LiveCC**（流式视觉模型）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### （1）**Vision-Language 性能**
| 模型 | OpenCompass (instruct) | MMBench EN v1.1 | MathVista | OCRBench |
|------|------------------------|------------------|------------|-----------|
| Gemini 2.5 Flash | 78.5 | 86.6 | 75.3 | 864 |
| Qwen3-VL-8B | — | 84.5 | 77.2 | 896 |
| Qwen3-Omni-30B-A3B | — | 84.9 | 75.9 | 880 |
| **MiniCPM-o 4.5 (9B)** | **77.6** | **87.6** | **80.1** | **876** |

✅ 结论：尽管参数仅为9B，MiniCPM-o 4.5 在多个基准上**超过或接近更大的模型**，尤其在 MMBench 和 OCRBench 表现突出。

#### （2）**Speech Understanding**
| 模型 | CoVoST2 en→zh | VoiceBench AlpacaEval* | Speech TriviaQA |
|------|---------------|--------------------------|------------------|
| Kimi-Audio 9B | 36.6 | 4.46 | 41.9 |
| Qwen3-Omni-30B-A3B | 46.6 | 4.74 | 62.9 |
| **MiniCPM-o 4.5** | **49.9** | **4.81** | **75.5** |

✅ 在跨语言语音翻译、语音问答等方面达到 SOTA。

#### （3）**Speech Generation**
| 模型 | SeedTTS Test-ZH (CER↓) | LongTTS-EN (WER↓) | Expresso*（情感控制） |
|------|-------------------------|--------------------|------------------------|
| CosyVoice2 | 1.45 | 14.80 | 17.9 |
| Qwen3-Omni | 1.41 | 17.33 | N/A |
| **MiniCPM-o 4.5** | **0.86** | **3.37** | **29.8** |

✅ 显著优于基线，在长语音生成稳定性与情感表达方面领先。

#### （4）**Omni-modal Understanding**
| 模型 | Daily-Omni | WorldSense | Video-Holmes | AVUT-Human |
|------|-----------|-------------|---------------|--------------|
| Gemini 2.5 Flash | 79.3 | 52.6 | 51.3 | 65.4 |
| Qwen3-Omni-30B-A3B | 70.7 | 54.0 | 50.4 | 74.2 |
| **MiniCPM-o 4.5** | **80.2** | **55.7** | **64.3** | **78.6** |

✅ 在五项 omni-modal 基准中四项领先。

#### （5）**Full-Duplex Streaming Performance**
| 模型 | LiveSports-3K-CC（Win Rate） |
|------|-------------------------------|
| LiveCC-8B | 41.5 |
| StreamingVLM-8B | 45.6 |
| **MiniCPM-o 4.5** | **54.4** |

✅ 显著提升连续视觉流中的响应准确性，验证 Omni-Flow 有效性。

---

### 🔬 消融实验结果

#### （1）**Omni-Flow 设计选择消融（Table 1）**
| Chunk Size | Boundary | Control | MMLU |
|------------|----------|--------|-------|
| 1.0s | Explicit | LS | 0.65 |
| 1.0s | Explicit | LT | 0.56 |
| 0.2s | Explicit | LS | 0.45 |
| 0.1s | Explicit | LS | 0.32 |

结论：
- 最佳 chunk size 为 **1.0s**，更短会导致决策不稳定。
- **显式边界标记**（explicit boundary）有助于区分输入与输出。
- **Listen-Speak 控制分离**（LS）优于统一预测（LT），解耦控制与内容更稳定。

#### （2）**长度奖励策略比较（Table 9）**
| 方法 | Thinking Mode Avg | Length Reduction |
|------|--------------------|------------------|
| 无长度奖励 | 73.5 | — |
| Kimi K1.5-style | 73.0 | 50.7% |
| **本文方法（smooth length reward）** | **74.3** | **35.3%** |

✅ 提出的平滑长度奖励在减少冗余的同时**保持甚至提升性能**，优于激进压缩策略。

#### （3）**语音生成模式对比（Table 10）**
| 模式 | ZH CER | EN WER | 特点 |
|------|--------|--------|------|
| 非交错 | 1.44 | 2.70 | 文本超前严重 |
| 固定交错 | 0.86 | 2.38 | 发音准确但延迟高 |
| **TAIL（动态交错）** | 1.04 | 3.93 | **时序对齐最优，适合全双工** |

✅ TAIL 在牺牲少量识别精度的前提下实现了最佳的时间同步性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **全双工交互是迈向类人智能的关键一步**  
   Omni-Flow 框架成功将感知与响应从“轮流”变为“并行”，显著提升了模型在动态环境下的适应性和响应及时性。

2. **小模型也能实现强大 omni-modal 能力**  
   仅 9B 参数的 MiniCPM-o 4.5 在多项指标上媲美甚至超越更大模型（如 Qwen3-Omni-30B），证明高效架构设计的重要性。

3. **主动行为可在统一框架下自然涌现**  
   不依赖额外模块，通过时间对齐流式建模即可让模型根据上下文自主发声。

4. **边缘部署成为可能**  
   模型可在 <12GB RAM 的设备上运行，结合 `llama.cpp-omni` 框架实现本地实时交互。

---

### ⚠️ 局限性
1. **长时间流式交互的鲁棒性仍需加强**  
   在复杂、长周期的真实环境中，模型可能出现注意力漂移或记忆衰减。

2. **语音生成偶发不稳定**  
   包括误读、中英文混杂等问题，尤其在快速切换语境时。

3. **网络条件影响体验**  
   Web demo 在弱网环境下可能出现延迟或输出碎片化；推荐本地部署以获得流畅体验。

4. **主动行为较简单**  
   当前主动性局限于提醒、注释等基础功能，尚未具备复杂规划或自我发起任务的能力。

---

### 🔮 未来工作方向
- 提升**长期记忆与情境建模能力**，支持更复杂的辅助任务。
- 引入**具身代理（embodied agent）机制**，实现物理世界中的主动交互。
- 优化**多轮语音流中的韵律一致性**，增强自然感。
- 探索**多模态强化学习**以进一步提升主动性与决策质量。

---

> 💡 **总体评价**：  
> MiniCPM-o 4.5 是首个真正意义上支持**实时全双工全模态交互**的开源 MLLM，标志着多模态模型正从“离线感知”走向“在线共存”。其提出的 **Omni-Flow** 框架有望成为下一代交互式 AI 的标准范式。

</details>

---

### 11. [End-to-End and Phase-Level Performance Optimization for Hyperledger Fabric](https://arxiv.org/abs/2604.27174)

**Authors**: Pavan Sollu, Aniruddha Mukherjee, Divya Pulivarthi, S. R. Eshwar, Gugan Thoppe, Kshitij Pratihast, Tittu Varghese, Hrishikesh Nashikkar, Yogesh Simmhan  
**Category**: cs.DC  
**Published**: 2026-05-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.27174v1  

#### Abstract
Hyperledger Fabric (HLF) is a modular, permissioned blockchain widely adopted in enterprise settings. Enhancing its throughput and latency remains challenging, as optimization decisions made in one phase of the transaction lifecycle can adversely affect other phases. In this work, we present a syste...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：End-to-End and Phase-Level Performance Optimization for Hyperledger Fabric

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

Hyperledger Fabric（HLF）作为企业级许可链的主流平台，其性能优化面临**跨阶段耦合效应**带来的挑战。现有研究大多孤立地优化单个阶段（如 endorsement、ordering 或 commit），忽视了各阶段之间的相互影响。例如：

- 缩小 block size 可降低轻负载下的延迟，但在重负载下会增加 commit 开销；
- 放松 endorsement 节点选择可提升吞吐量，但可能导致更多 MVCC 冲突。

本文系统性地研究了 HLF 交易生命周期中 **endorsement、ordering 和 commit 三个阶段间的交互关系**，揭示了局部最优决策可能损害端到端性能的问题。

---

### **提出了什么新方法或新思路**

#### （1）**两个新型优化机制**

| 方法 | 描述 | 创新性 |
|------|------|--------|
| **Block-level Pipelining（块级流水线）** | 将 commit 阶段划分为两个相位：<br>• Phase 1: VSCC 验证 + 私有数据获取（可并行）<br>• Phase 2: MVCC 检查 + 账本写入（必须串行）<br>在处理 block `b_i` 的 Phase 2 时，并行启动 `b_{i+1}` 的 Phase 1。 | 打破 HLF 原有的“一次只提交一个区块”限制，在不破坏一致性前提下实现跨块流水线，无需依赖复杂的依赖图构建或推测执行。 |
| **Strategic Waiting（策略性等待）** | 当快速节点（leader）领先慢速节点（lagger）超过阈值时，主动暂停自身 commit 进度，同时为落后者提供资源加速（boost），使其能追上进度，维持多领导者并行 endorsement 的能力。 | 首次提出通过协调机制防止 Soft Max-Ht 策略退化为单 leader 模式，从而保持 endorsement 并发性。 |

#### （2）**对三大配置参数的细粒度分析与推荐策略**

| 参数 | 新发现与建议 |
|------|-------------|
| **Private-data Dissemination** | 提出 `(n-1, 1*)` 松散 quorum 策略：广播发送私有数据给所有节点，但只要收到任意一个确认即可继续，兼顾了低 endorsement 延迟与高 commit 效率。 |
| **Block Size Selection** | block size 应随负载动态调整：<br>• 轻负载 → 小 block（减少排队延迟）<br>• 重负载 → 大 block（摊薄每块开销） |
| **Endorsement Peer Selection** | 发现严格策略（如 Max-Ht）导致大量 transaction drop；宽松策略（如 All Peers）虽避免 drop 却引发更多 MVCC invalidation。提出应根据应用容忍度权衡选择。 |

---

### **相比现有方法的优势**

| 对比维度 | 本文优势 |
|---------|----------|
| **系统视角** | 不再孤立分析单一阶段，而是进行 **phase-aware、end-to-end 的联合建模与优化**，更贴近真实部署场景。 |
| **优化机制设计** | 所提方法（如 pipelining）兼容 HLF 原生协议，无需修改共识逻辑或引入复杂状态管理，易于集成。 |
| **实证基础扎实** | 结合生产级 HLF 测试床（GKE 上部署）与校准后的 SimPy 仿真，覆盖多种配置组合与极端情况。 |
| **实用性强** | 给出明确的操作指南（actionable guidance），如“何时启用流水线”、“如何平衡 Phase 1 与 Phase 2 时间”。 |

---

## 2. 核心实验方法和设置

### **使用的数据集与环境**

- **无传统“数据集”**，采用合成 workload 模拟交易流。
- 实验基于 **Hyperledger Fabric v2.4.7** 构建。
- **测试床环境**：
  - Google Kubernetes Engine (GKE) 集群
  - 5 个 Peer 节点（各独占 1 个 VM）
  - 3 个 Raft Orderer（共驻一节点）
  - 5 个 Client Generator（共驻一节点）
  - 每个 VM：24 vCPU, 32GB RAM, 100GB SSD

### **实验设置**

| 参数 | 设置说明 |
|------|----------|
| **Topology** | 单组织（Single Organization）、单 Channel |
| **Workload** | 客户端以固定 TPS 发送交易请求（如 250–800 TPS） |
| **Chaincode** | 简单读写操作，模拟典型智能合约行为 |
| **StateDB** | LevelDB |
| **Simulation Tool** | SimPy 离散事件模拟器，用于扩展核心数、模拟异构延迟等极限场景 |

### **评估指标**

| 指标 | 含义 |
|------|------|
| **End-to-End Latency** | 从客户端发起交易到被写入账本的时间 |
| **Throughput (TPS)** | 成功 commit 的有效交易数 / 秒 |
| **Commit TPS** | 提交阶段处理速度 |
| **Endorsement Drop Rate** | 因背书节点过载而丢弃的交易比例 |
| **MVCC Invalidation Rate** | 因版本冲突失败的交易占比 |
| **Block Creation Time** | orderer 生成 block 所需时间 |
| **Phase 1 / Phase 2 Duration** | 流水线两阶段耗时，用于分析瓶颈 |

### **基线方法对比**

| 基线策略 | 说明 |
|--------|------|
| **Default HLF** | 默认 `(1,1)` 私有数据传播 + Rank List 背书选择 |
| **(n-1, n-1)** | 全广播且需全部确认 |
| **(n-1, 1)** | 全广播但仅需特定 peer 确认（仍等待最慢响应） |
| **Max-Ht / Ranked List / All Peers** | 不同背书节点选择策略 |
| **Serial Commit** | 原始 HLF 提交流程，作为 pipelining 的对照组 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据汇总**

| 方法 / 配置 | 性能提升 | 具体数值 |
|------------|----------|----------|
| **Block-level Pipelining** | 最高提升 **1.9× commit throughput** | 在 `(4-4)` 数据分发下达到 1.5×，`(4-1)` 下达 1.48× |
| **Strategic Waiting** | 提升 **~1.2× throughput** | 从 12.2 TPS → 14.4 TPS（模拟 6000 交易任务） |
| **Relaxed Quorum (n-1,1\*)** | Endorsement 延迟降低至 **182ms**（vs. `(4,1)` 的 360ms）<br>Commit 延迟降至 **~2.5s**（vs. `(1,1)` 的 3.9s） | 实现 endorsement 与 commit 性能双优 |
| **Large Block (1000–1500 tx/block)** under Heavy Load | 吞吐量提升 **~90%**（vs. 500 tx/block）<br>端到端延迟下降 **~35%** | 在 800 TPS 下，TPS 从 657 → 955 |
| **Small Block (500 tx/block)** under Light Load | 端到端延迟仅为大块的 **1/4** | 从 3.5s（2000 tx）降至 0.98s（500 tx） |

### **与基线方法的对比结果**

| 对比项 | 本文方法 | 基线方法 | 差距 |
|-------|----------|----------|------|
| **Endorsement Latency** | `(n-1,1*)`: **182ms** | `(n-1,1)`: **360ms** | ↓ 49% |
| **Commit Latency** | `(n-1,1*)`: **~2.5s** | `(1,1)`: **~3.9s** | ↓ 36% |
| **Endorsement Drop Rate** | Soft Max-Ht: **~15% drop** | Max-Ht: **~30% drop** | ↓ 一半 |
| **MVCC Invalidation Rate** | Ranked List: **较低** | All Peers: **极高（>168k inv. @ p=1）** | 显著优于全开放策略 |

### **消融实验结果（Ablation Study）**

#### （1）**Pipelining 效果依赖 Phase 平衡**

| 核心数（VSCC 并行度） | Phase 1 / Phase 2 时间比 | 吞吐增益（Pipeline / Serial） |
|----------------------|------------------------|-------------------------------|
| 24 cores | ~2.8 | 1.32–1.50× |
| 64 cores | ~1.0 | **1.94×**（峰值） |
| 96 cores | ~0.8 | 1.81×（开始下降） |

> ✅ **结论**：当 Phase 1 ≈ Phase 2 时，流水线效率最高；过度并行化反而造成资源浪费。

#### （2）**Private Data Sharing 影响 Phase 分布**

| 策略 | Pvt Data Fetch 时间 | Phase 1 占主导？ | 是否适合流水线 |
|------|--------------------|------------------|----------------|
| `(1,1)` | >2000ms | 是 | ❌ 效果差（流水线空转） |
| `(4-4)/(4-1)` | ~500ms（本地查找） | 否 | ✅ 效果好 |

> ✅ **结论**：提前广播私有数据是发挥流水线潜力的前提。

---

## 4. 关键结论和发现

### **主要发现**

1. **Endorsement 与 Commit 存在根本性权衡**  
   - 快速 endorsement 往往牺牲 commit 效率，反之亦然。
   - `(n-1, 1*)` 松散 quorum 成功缓解该矛盾。

2. **Block Size 必须负载感知（Load-Aware）**  
   - 轻负载 → 小 block（降低队列延迟）
   - 重负载 → 大 block（摊薄开销，提高吞吐）

3. **Endorsement Policy 需权衡 “Drop” vs “Invalidation”**  
   - 若重视 fail-fast，选 Max-Ht / Soft Max-Ht
   - 若要求零丢弃，选 Ranked List
   - 避免使用 All Peers（MVCC 冲突爆炸）

4. **Pipelining 的收益存在“黄金点”**  
   - 吞吐提升最大发生在 **Phase 1 ≈ Phase 2** 时
   - 过度增加 VSCC 并行度会导致 Phase 2 成为新瓶颈，收益递减

5. **Heterogeneous Peer Speed 导致系统退化**  
   - 快速节点持续领先 → 其他节点失去 endorsement 资格 → 回归单 leader 模式
   - **Strategic Waiting 可有效延缓此过程，维持并发性**

---

### **方法的局限性**

| 局限 | 说明 |
|------|------|
| **实验规模有限** | 当前测试床仅支持单组织，未验证跨组织通信开销的影响 |
| **Strategic Waiting 尚处探索阶段** | 当前为模拟验证，尚未在真实集群中实现资源重定向机制（如 CPU 抢占、带宽调度） |
| **未考虑网络分区或故障恢复** | 所有实验假设稳定网络与节点可用性 |
| **静态配置为主** | 缺少动态自适应控制器来实时调优 block size、quorum 等参数 |

---

### **未来工作方向**

1. **扩展至 Multi-Org 场景**  
   研究跨组织 endorsement、gossip 协议与私有数据同步的性能边界。

2. **开发 Adaptive Controller**  
   设计运行时调控器，根据当前负载、延迟分布自动切换 block size、dissemination 策略等。

3. **联合优化多个机制**  
   探索 pipelining + strategic waiting + relaxed quorum 的协同效应。

4. **真实业务 workload 验证**  
   在 NPCI 的 eRupee 支付系统等实际场景中部署验证。

5. **硬件感知优化**  
   结合 NVMe、RDMA 等新型存储与网络技术，进一步释放 commit 阶段潜力。

--- 

> 📌 **总结一句话**：  
> 本文通过 **phase-level 微基准测试 + end-to-end 联合分析**，揭示了 HLF 中“局部优化反致全局劣化”的现象，并提出 **block-level pipelining** 与 **strategic waiting** 两项新机制，在不改变协议的前提下显著提升了吞吐量与稳定性，为大规模 HLF 部署提供了实用的性能工程指南。

</details>

---

### 12. [Kernelized Advantage Estimation: From Nonparametric Statistics to LLM Reasoning](https://arxiv.org/abs/2604.28005)

**Authors**: Shijin Gong, Kai Ye, Jin Zhu, Xinyu Zhang, Hongyi Zhou, Chengchun Shi  
**Category**: cs.LG  
**Published**: 2026-05-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.28005v1  

#### Abstract
Recent advances in large language models (LLMs) have increasingly relied on reinforcement learning (RL) to improve their reasoning capabilities. Three approaches have been widely adopted: (i) Proximal policy optimization and advantage actor-critic rely on a deep neural network to estimate the value ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Kernelized Advantage Estimation: From Nonparametric Statistics to LLM Reasoning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在大语言模型（LLM）推理任务中，基于强化学习（RL）的方法（如 RLVR）被广泛用于提升模型的推理能力。然而，在**资源受限**（如计算预算有限、无法训练额外网络或采样大量轨迹）的场景下，现有的策略梯度方法面临以下挑战：

- **PPO / A2C 类方法**：依赖一个独立的 value network 来估计价值函数以降低梯度方差，但训练和维护该网络带来显著的计算和内存开销。
- **GRPO 类方法**：虽无需 value network，但需对每个 prompt 采样大量 reasoning 轨迹（如 G=64），才能准确估计价值函数，计算成本高。
- **REINFORCE 类方法**：每 prompt 只采样一条轨迹，计算高效，但梯度估计方差大，样本效率低。

本文聚焦于**每 prompt 只能采样少量轨迹**（甚至仅一条）的实际资源受限场景，旨在实现**低方差、高样本效率且计算轻量**的策略优化。

---

### **提出的新方法：Kernelized Advantage Estimation (KAE)**

作者提出 **KAE**，一种将**非参数统计方法**引入 LLM 推理中的新框架，其核心思想是：

> 利用**核平滑（kernel smoothing）** 对历史训练迭代中同一 prompt 的奖励进行加权平均，从而更准确地估计当前策略下的价值函数 $V_{\theta}(x)$ 和优势函数 $A$。

#### **关键创新点**：
1. **跨训练迭代的信息复用**：
   - 不同于 GRPO 仅利用当前批次内多个 completion 的平均作为 baseline，KAE 利用**过去多个训练步中相同 prompt 的历史奖励**来增强估计。
   - 通过核函数赋予近期奖励更高权重，远期奖励更低权重，形成时间上的平滑估计。

2. **非参数统计的引入**：
   - 将价值函数估计建模为一个**一维非参数回归问题**（以训练步数为自变量，奖励为因变量）。
   - 采用经典的 **Nadaraya-Watson 核估计器**，避免了深度神经网络的复杂建模。

3. **与现有方法的统一视角**：
   - 在算法框架上统一了 A2C、REINFORCE++ 和 GRPO，KAE 可视为一种“时序感知”的动态 baseline 构造方式。

---

### **相比现有方法的优势**
| 方法 | 是否需要 Value Network | 每 prompt 采样数 | 方差控制机制 | 缺陷 |
|------|------------------------|------------------|---------------|-------|
| PPO/A2C | 是 | 1 | 学习型 baseline | 高计算/内存开销 |
| GRPO | 否 | 多（如 64） | 组内平均 baseline | 高采样成本 |
| REINFORCE++ | 否 | 1 | 全局平均 baseline | 偏置大，效率低 |
| **KAE (本文)** | **否** | **可少至 1** | **核平滑历史奖励** | **计算轻量 + 高效 + 低方差** |

> ✅ **KAE 在不增加网络、不增加采样数的前提下，实现了接近“oracle”级别的性能**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **GSM8K**：小学数学应用题，相对简单。
- **MATH**：高中及以上难度数学题，更具挑战性。
- **DAPO**：大规模数学推理训练集（17k 问题），用于多流设置。

### **模型配置**
- **Qwen2.5-1.5B-Instruct**：用于 GSM8K 单流实验。
- **Qwen2.5-Math-1.5B / Qwen2.5-Math-7B**：用于 MATH 和 DAPO 实验。

### **实验设置**
- **训练步数**：200–500 步。
- **Group Size G**：测试了 $G \in \{1, 4, 8\}$，覆盖单流与多流场景。
- **Prompt Sampling Schedule**：
  - 采用**分组重用策略**：将 prompts 分为若干 minibatch，每个 minibatch 连续使用 $J=8$ 或 $10$ 步，以增强历史信息的相关性。
- **核函数选择**：
  - 主要使用 **triangular kernel**，敏感性分析中也测试了 **exponential kernel**。
- **带宽选择**：带宽 $h$ 与有效样本量相关，遵循理论建议。

### **评估指标**
1. **Value Estimation**：
   - **MSE of value estimator**（相对于蒙特卡洛估计的真实值函数）
2. **Gradient Estimation**：
   - **MSE of gradient estimator**（相对于真实梯度）
3. **Policy Optimization**：
   - **Test accuracy** on downstream reasoning benchmarks:
     - AIME24, AIME25, AMC, MATH, Minerva, Olympiad
   - 平均准确率（Avg）

### **基线方法对比**
- **GRPO-type**：GRPO, Dr. GRPO, GPG
- **REINFORCE-type**：REINFORCE, REINFORCE++
- **Oracle**：使用真实价值函数构造优势函数的理想情况（用于理论对标）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### **(1) Value Estimation 性能（Table 1）**
| 方法 | MSE Reduction vs. GRPO | vs. REINFORCE++ |
|------|------------------------|------------------|
| **KAE** | **60%–70%** | **>90%** |

> 📌 示例：在 MATH 数据集 Step 10 上，KAE 的 MSE 仅为 **5.80×10⁻³**，而 GRPO 为 19.4，REINFORCE++ 为 78.8。

#### **(2) Gradient Estimation 性能（Table 2）**
| 方法 | MSE Reduction vs. GRPO | vs. REINFORCE++ |
|------|------------------------|------------------|
| **KAE** | **5.4%–8.5%** | **32.6%–65.4%** |

> 📌 表明更优的价值估计直接转化为更精确的梯度估计。

#### **(3) Policy Optimization 性能（Tables 3 & 4）**

##### **Table 3：MATH 训练，Qwen2.5-Math-7B 模型**
| 方法 | Avg Accuracy |
|------|--------------|
| GRPO | 40.10% |
| Dr. GRPO | 39.91% |
| GPG | 40.33% |
| **KAE** | **41.49%** ✅ |

> 🔺 **平均提升 1.16%**，最高单项提升达 **17.0%**（AIME25）。

##### **Table 4：DAPO 训练，Qwen2.5-Math-7B 模型**
| 方法 | Avg Accuracy |
|------|--------------|
| GRPO | 42.88% |
| Dr. GRPO | 40.92% |
| GPG | 43.49% |
| **KAE** | **44.54%** ✅ |

> 🔺 **平均提升 1.05%**，最高单项提升达 **79.9%**（AIME25）。

#### **(4) 单流设置（G=1）下的稳定性（Figure 4）**
- 在 **GSM8K** 和 **MATH** 上，标准 REINFORCE 出现**准确率下降**（过拟合或训练不稳定）。
- **KAE** 持续稳定上升，最终提升：
  - **GSM8K**: +14.9%
  - **MATH**: +6.6%

> ✅ 显示 KAE 显著提升了训练稳定性。

---

### **消融实验结果（Ablation Study）**

#### **(1) Prompt Sampling Schedule 的作用**
- 引入相同采样策略的 **REINFORCE + Schedule**：
  - 仍出现性能下降，未解决稳定性问题。
- 结论：**采样策略本身不足以带来增益**。

#### **(2) 历史奖励平滑的作用**
- **KAE vs. GRPO + Schedule**（Figure S.1）：
  - KAE 在约 30 步后持续领先。
- 结论：**性能提升主要来自核平滑带来的更优价值估计**，而非采样策略。

#### **(3) 敏感性分析（Figure 3）**
- KAE 在不同 **带宽 $h$** 和 **核函数**（triangular vs. exponential）下表现稳健。
- MSE 始终显著低于 GRPO 和 REINFORCE++。

> ✅ 方法对超参数不敏感，具备良好鲁棒性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **KAE 实现了“oracle-like”性能**：
   - 在资源受限下，其性能接近已知真实价值函数的 oracle 方法。
2. ✅ **非参数统计方法可有效迁移至 LLM 推理**：
   - 核平滑等经典方法在低采样预算下优于深度学习建模。
3. ✅ **历史信息的时间平滑显著提升估计精度**：
   - 跨训练步借用信息比仅依赖当前 batch 更高效。
4. ✅ **改进价值估计 → 改进梯度估计 → 改进策略优化**：
   - 三者形成正向闭环，验证了理论设计的有效性。

---

### **方法的局限性**
- **依赖 prompt 重复出现**：若 prompts 完全随机且无重复，则历史信息不可用。
- **假设策略变化缓慢**：若策略跳跃剧烈，历史奖励可能不再具有参考价值。
- **目前仅适用于 contextual bandit 设置**：尚未扩展到长序列 MDP 场景。

---

### **未来工作方向**
1. **扩展至 MDP 设置**：将核平滑应用于状态-动作对的价值估计。
2. **自适应带宽选择**：设计自动调整 $h$ 的机制，进一步减少调参负担。
3. **结合 shrinkage 与 kernel 方法**：融合跨 prompt 与跨 iteration 的信息借用。
4. **理论扩展至非平稳环境**：研究在策略快速演化下的渐近性质。

---

## **总结**
> **KAE 成功将经典非参数统计思想引入 LLM 推理训练，提出了一种无需额外网络、仅需少量采样的高效策略优化方法。它不仅在理论上证明了“oracle property”，也在实验中全面超越 GRPO 和 REINFORCE 类方法，尤其在小样本和单流场景下展现出卓越的稳定性与性能。这为资源受限下的 LLM post-training 提供了一个极具潜力的新范式。**

</details>

---

### 13. [A Unified Framework of Hyperbolic Graph Representation Learning Methods](https://arxiv.org/abs/2604.28070)

**Authors**: Sof\'ia P\'erez Casulo, Marcelo Fiori, Bernardo Marenco, Federico Larroca  
**Category**: cs.LG  
**Published**: 2026-05-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.28070v1  

#### Abstract
Hyperbolic geometry has emerged as an effective latent space for representing complex networks, owing to its ability to capture hierarchical organization and heterogeneous connectivity patterns using low-dimensional embeddings. As a result, numerous hyperbolic graph representation learning methods h...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《A Unified Framework of Hyperbolic Graph Representation Learning Methods》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Hyperbolic Graph Representation Learning (GRL)** 方法存在以下问题：
- 不同方法实现分散，使用不同编程语言、优化策略和接口；
- 缺乏统一的训练、评估和可视化工具，导致**可复现性差**；
- 难以进行公平、系统性的比较，限制了实际应用和理论理解。

### 提出了什么新方法或新思路
作者提出了 **HypeGRL** —— 一个开源的、统一的 Python 框架，用于集成多种主流的 hyperbolic GRL 方法。其核心设计包括：
- 统一的优化接口（training pipeline）；
- 支持跨模型转换（如 Lorentz ↔ Poincaré ↔ Polar）；
- 内置可视化工具和与 NetworkX 等标准图分析库的无缝对接；
- 支持扩展新方法，具备良好的可拓展性（extensibility）。

集成的方法包括：
- Hydra / Hydra+
- Poincaré Embeddings
- Lorentz Embeddings
- Mercator / D-Mercator
- HyperMap
- Poincaré Maps

### 相比现有方法的优势
- **统一性**：首次将多个异构实现整合到单一框架中，降低使用门槛；
- **可复现性**：提供完整代码和实验脚本（[GitHub](https://github.com/CicadaUY/hypeGRL)），支持公平比较；
- **易用性**：简化了训练、评估和可视化的流程，便于研究者快速验证假设；
- **系统性分析能力**：为后续研究提供了标准化基准平台。

---

## 2. 核心实验方法和设置

### 使用的数据集
#### 合成数据：
- **Balanced Binary Tree**（分支因子2，深度4，共31节点）：用于测试对层次结构的建模能力。

#### 真实世界网络（Real-world Networks）：
| 数据集 | 节点数 | 边数 | 应用领域 |
|-------|--------|------|---------|
| Toggle Switch | 200 | 1,896 | 基因调控网络 |
| Olsson | 382 | 4,214 | 单细胞 RNA-seq 数据 |
| Myeloid Progenitors | 640 | 5,649 | 血细胞发育网络 |
| Polblogs | 1,222 | 16,717 | 政治博客网络（二分类任务） |
| Synthetic Gaussian-wrapped Normal Graph | 1,250 | ~12.5k | 合成超球面生成图 |

此外还报告了各图的 **mean Gromov hyperbolicity (ω_mean)**，用以衡量其内在双曲性强度。

### 实验设置和评估指标

#### 下游任务：
1. **Link Prediction（链接预测）**
   - 设置：随机移除10%边作为正样本，保留非邻接对作为负样本；
   - 评估方式：
     - **F1 Score**（在预测固定数量链接时）
     - **Lift (Top 10%)**：前10%最高分预测中包含的真实正例比例

2. **Node Classification（节点分类）**
   - 设置：基于拓扑学习嵌入，忽略标签；使用 **k-NN 分类器**（k=5 或 10），距离为 hyperbolic metric；
   - 评估指标：**Macro-F1 Score**

3. **Visualization Analysis**
   - 在 polar coordinates $(r, \theta)$ 中可视化嵌入，观察是否能恢复层级结构（radial depth）和相似性聚类（angular separation）

#### 基线方法对比
- **Euclidean Baseline**：Random Dot Product Graph (RDPG)，使用不同维度 $n=2,8,16$
- 所有 hyperbolic 方法均使用二维嵌入（$n=2$），体现低维优势

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables I 和 II）

#### ✅ Link Prediction 结果（Table I）
| Dataset | 最佳方法（F1） | 第二佳方法 | RDPG(n=2) 性能 |
|--------|----------------|------------|---------------|
| Toggle Switch | **D-Mercator**: 57.7% | Poincaré Maps: 57.5% | 13.0% |
| Olsson | **Poincaré Maps**: 29.25% | RDPG(n=16): 30.4% | 8.48% |
| Myeloid Progenitors | **Poincaré Maps**: 48.4% | D-Mercator: 41.9% | 7.37% |

> 🔍 Lift 指标显示：多数真实链接集中在 top 10%，表明 hyperbolic distance 具强判别力。

#### ✅ Node Classification 结果（Table II）
| Dataset | 最佳方法（F1） | 第二佳方法 |
|--------|----------------|------------|
| Synthetic Gaussian | **Hydra+**: 85.03% | Poincaré Maps: 83.92% |
| Polblogs | **Poincaré Embeddings**: 95.10% | Hydra+: 78.90% |

> ⚠️ 注意：D-Mercator 在两个分类任务中表现极差（~17%-50%），说明其对模型误设敏感。

#### ⏱️ 计算效率对比（Fig. 1）
| 方法 | 平均运行时间（秒） |
|------|--------------------|
| Hydra | 0.003 s |
| Hydra+ | 0.02 s |
| D-Mercator | 0.08 s |
| Poincaré Maps | 3.27 s |
| Lorentz Embeddings | **18,367.3 s (~5小时)** ❌ |

> Hydra 是最快的，而 Lorentz Embeddings 因采样策略低效导致计算不可行。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Poincaré Maps 是综合性能最强的方法**：
   - 在 link prediction 和 node classification 中均取得最优或接近最优结果；
   - 尽管计算成本较高（~3.27s），但准确率显著优于其他方法。

2. **Hydra+ 是性能与效率的最佳平衡点**：
   - 准确性高（尤其在合成树和分类任务中）；
   - 运行速度快（<0.02s），适合大规模或实时场景；
   - 特别适用于强调距离保真度的任务。

3. **低维 hyperbolic embeddings 明显优于 Euclidean counterparts**：
   - 即使是 $n=2$ 的 hyperbolic 方法，在许多情况下也优于 $n=16$ 的 RDPG；
   - 尤其在“高度双曲”的图（如 Myeloid Progenitors）上优势更明显。

4. **Representation efficiency 取决于图的几何特性**：
   - 图的 ω_mean 越小（相对于直径），越适合用 hyperbolic geometry 建模；
   - 双曲性弱的图（如 Olsson）中，高维 Euclidean 方法仍具竞争力。

5. **Estimation procedure 比几何模型本身更重要**：
   - 尽管 D-Mercator 和 Poincaré Embeddings 基于相似生成模型，但后者性能远胜前者；
   - 表明优化算法和损失函数设计对最终效果影响巨大。

### 方法的局限性
- **某些方法计算开销大**：如 Lorentz Embeddings 因负采样机制难以扩展；
- **部分方法对模型假设敏感**：如 HyperMap 依赖 degree-age correlation，在规则图中失效；
- **未涵盖所有最新方法**：如 Curvature Adaptive Poincaré Embeddings 等未被集成；
- **缺乏多跳预测或多关系图支持**：目前聚焦简单无向图。

### 未来工作方向
- 扩展 HypeGRL 以支持更多模型（如 adaptive curvature models, temporal graphs）；
- 引入更高效的采样和优化策略（如 mini-batch training, momentum on manifolds）；
- 探索自动选择最优 embedding 方法的元学习机制；
- 将框架应用于更大规模的真实网络（如社交网络、知识图谱）；
- 加强对 embedding 几何性质的解释性分析（如 curvature estimation, cluster separability）。

---

> 📌 **总结一句话**：  
> HypeGRL 构建了一个**可复现、可比较、可扩展**的 hyperbolic GRL 统一框架，并通过系统实验揭示了不同方法在准确性、效率和适用性上的权衡，推动了该领域的标准化发展。

</details>

---

### 14. [Neural Aided Kalman Filtering for UAV State Estimation in Degraded Sensing Environments](https://arxiv.org/abs/2604.28107)

**Authors**: Akhil Gupta, Erhan Guven  
**Category**: cs.LG  
**Published**: 2026-05-01  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.28107v1  

#### Abstract
Accurate state estimation of nonlinear dynamical systems is fundamental to modern aerospace operations across air, sea, and space domains. Online tracking of adversarial unmanned aerial vehicles (UAVs) is especially challenging due to agile nonlinear motion, noisy and sparse sensor measurements, and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Neural Aided Kalman Filtering for UAV State Estimation in Degraded Sensing Environments

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对在**退化感知环境**（degraded sensing environments）下对无人机（UAV）进行精确状态估计的挑战展开研究。这类环境通常具有以下特点：
- 高噪声（high measurement noise）
- 低采样率（sparse observations）
- 控制输入未知且运动高度非线性（agile nonlinear motion with unknown control policies）

传统 **Extended Kalman Filter (EKF)** 和 **Unscented Kalman Filter (UKF)** 在这些条件下表现不佳，因为它们依赖于精确的动力学建模和协方差假设，在高噪声或稀疏数据下容易发散。

---

### 提出的新方法：Bayesian Neural Kalman Filter (BNKF)

作者提出了一种混合框架——**Bayesian Neural Kalman Filter (BNKF)**，其核心思想是：
- 使用一个离线训练的 **Variational Bayesian Neural Network (BNN)** 来替代 Kalman Filter 中的状态转移模型（即预测步骤）。
- 利用 BNN 输出的**均值**作为状态预测，并将其**预测不确定性**（epistemic uncertainty）以分布形式传入后续的 Kalman correction 步骤。
- 最终结合标准的 Kalman 更新步骤完成状态修正。

此外还提出了一个集成变体：**BNKFe**（Ensemble variant），将三维空间中的 $x, y, z$ 分量分别由独立的小型 BNN 预测，再合并为完整状态。

---

### 相比现有方法的优势
| 对比维度 | EKF / UKF | BNKF |
|--------|---------|------|
| 动力学建模 | 依赖解析模型或线性化近似 | 数据驱动学习复杂非线性关系 |
| 不确定性建模 | 固定或启发式设定过程噪声 $Q$ | BNN 通过 KL divergence 学习自适应不确定性 |
| 噪声鲁棒性 | 在高噪声下协方差膨胀严重，易发散 | 显式利用 BNN 的 MC Dropout 推理输出不确定性，提升鲁棒性 |
| 实时性 | 轻量级，适合实时部署 | 推理开销略高但仍满足实时需求（见 Fig. 7） |

> ✅ **创新点总结**：
> 1. 首次将 **BNN 的贝叶斯不确定性**直接用于 Kalman Filter 的协方差传播；
> 2. 构建了一个端到端可训练、适用于在线推理的神经增强滤波器架构；
> 3. 系统评估了该方法在不同噪声水平与采样率下的性能边界。

---

## 2. 核心实验方法和设置

### 数据集
使用公开数据集：**[Synthetic-UAV-Flight-Trajectories](https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories)**  
- 包含超过 5,000 条随机生成的 UAV 轨迹
- 总模拟飞行时间约 20 小时
- 基于 **Gazebo** 物理引擎仿真，具备高保真动力学特性
- 每条轨迹提供 XYZ 坐标系下的 3D position 与 velocity 及对应 timestamp

---

### 传感器测量生成
使用 **Stone Soup** 库模拟雷达观测：
- 观测类型：range, range-rate, bearing, elevation
- 固定传感器位置
- 添加高斯噪声以模拟真实雷达误差

#### 实验变量控制
| 参数 | 设置 |
|-----|------|
| **Noise Level** | 三级：Low / Medium / High（详见 Table I）<br>- 如 High 级别：Range σ=100m, Bearing σ=0.1° |
| **Sampling Rate (SR)** | 三种：1.0（全采样）、0.75、0.50<br>模拟部分丢失或降频采集场景 |

共构建约 15,000 条测试轨迹用于评估。

---

### 输入特征向量定义
$$
z_t = [p_t, \theta_t, \phi_t, \dot{p}_t, \sigma_p, \sigma_\theta, \sigma_\phi, \sigma_{\dot{p}}]
$$
其中前四项为带噪测量值，后四项为其已知标准差。

---

### 评估指标
1. **State Estimate Error (ED)**：欧氏距离  
   $$
   e_t = \sqrt{(x - \hat{x})^2 + (y - \hat{y})^2 + (z - \hat{z})^2}
   $$

2. **Uncertainty Quantification (Det)**：协方差矩阵行列式  
   $$
   V = \det(P)
   $$

3. **Truth Containment (MD)**：Mahalanobis Distance  
   $$
   D = (\mathbf{x} - \hat{\mathbf{x}})^T P^{-1} (\mathbf{x} - \hat{\mathbf{x}})
   $$
   - 理想情况下 $E[D] = 3$（三维状态空间）

---

### 基线方法对比
| 方法 | 描述 |
|-----|------|
| **EKF** | 经典扩展卡尔曼滤波，局部线性化处理非线性观测 |
| **UKF** | 无迹卡尔曼滤波，通过 Sigma Points 近似非线性变换 |
| **BNN (standalone)** | 仅使用 BNN 进行预测，无 Kalman 修正（用于消融分析） |
| **BNKF** | 本文提出的方法：BNN + Kalman correction |
| **BNKFe** | 集成版本，分轴预测后融合 |

所有 BNN 模型结构一致：
- 5 层 Bayesian Linear Layer
- 每层 64 neurons
- 使用 `torchbnn` 实现
- 推理阶段进行 100 次 Monte Carlo 抽样估计均值与方差

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table II）

| Model | High Noise ED ↓ | High Noise MD ↓ | High Noise Det ↓ |
|-------|------------------|------------------|--------------------|
| EKF   | 35.16 m          | 9.35             | 15,740             |
| UKF   | 66.78 m          | 25.17            | 7,262              |
| BNN   | 13.13 m          | 8.94             | 102.26             |
| **BNKF** | **8.63 m**       | **5.68**         | **13.66**          |
| **BNKFe** | **9.24 m**       | **7.11**         | **8.11**           |

> ✅ 所有数值均为五折交叉验证平均值 ± 标准差

---

### 与基线方法的对比结果
- 在**高噪声 + 低采样率**（如 High Noise, SR=0.75）条件下：
  - BNKF 的 ED 比 EKF/UKF 降低 **75%以上**
  - Mahalanobis Distance 更接近理想值 3，说明不确定性更“校准”（well-calibrated）
  - 协方差行列式显著更低，表明预测更集中而不盲目自信

- 在中低噪声环境下，BNKF 依然优于 EKF/UKF，尤其在 **truth containment** 上优势明显。

- UKF 表现在高噪声下反而劣于 EKF，可能因 Sigma Points 放大了噪声影响。

---

### 消融实验结果
- **Standalone BNN vs BNKF**：
  - 单独 BNN 的 ED 和 MD 均高于 BNKF，证明 Kalman correction 步骤有效提升了精度与置信度校准。
  - 例如在 High Noise 下，BNN 的 ED=13.13m → BNKF=8.63m，下降约 34%

- **BNKF vs BNKFe**：
  - BNKFe 在 Determinant 上进一步降低（High Noise 下从 13.66 → 8.11），表示不确定性估计更紧凑
  - 但在 ED 上略有上升（8.63 → 9.24），显示轻微精度牺牲换取更高稳定性
  - 符合集成模型的一般规律：减少方差，略微增加偏差

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **BNKF 在高噪声与稀疏观测下显著优于传统 EKF 和 UKF**，特别是在状态估计精度（ED）、不确定性量化（Det）和真值包含能力（MD）方面全面领先。
2. ✅ **BNN 提供的数据驱动先验能有效捕捉 UAV 的意图驱动运动模式**，弥补了传统 KF 对动态模型依赖的不足。
3. ✅ **将 BNN 的预测不确定性显式引入 Kalman correction 步骤，增强了系统的鲁棒性和置信度校准能力**。
4. ✅ **BNKF 具备良好的实时部署潜力**：单轨迹推理时间低于 EKF 和 UKF（Fig. 7），尽管训练阶段需离线完成。
5. ✅ **Ensemble 设计（BNKFe）可进一步压缩不确定性估计范围**，适用于对置信区间敏感的应用场景。

---

### 方法的局限性
1. ⚠️ **完全基于合成数据训练与验证**：虽然 Gazebo 模拟较真实，但未在真实雷达/UAV 数据上测试，泛化能力有待验证。
2. ⚠️ **假设传感器固定且噪声为高斯分布**：现实中可能存在非高斯噪声（如脉冲干扰）、移动传感器或多源异构传感，当前框架未涵盖。
3. ⚠️ **BNN 训练成本较高**：Variational Inference 和 MC Sampling 导致训练周期长，难以在线更新模型。
4. ⚠️ **未考虑动态环境变化或对抗性干扰建模**：如突然机动、GPS欺骗等极端情况未纳入评估。

---

### 未来工作方向
1. 🔄 将 BNKF 扩展至 **multi-sensor fusion** 场景，支持 IMU、视觉、GNSS 等多模态输入。
2. 🔁 探索 **online adaptation mechanism**，使 BNN 能够根据新观测微调权重（如 continual learning 或 meta-learning）。
3. 🛰️ 在 **真实飞行平台与雷达系统** 上部署并验证 BNKF 的实际性能。
4. 🤖 结合 **Physics-Informed Neural Networks (PINNs)** 引入物理约束，提高外推能力和安全性。
5. 💡 探索轻量化 BNN 架构（如 Binary/Bayesian CNN-LSTM）以适配嵌入式设备部署。

---

> **代码开源地址**：[https://github.com/agupt126/NeuralAugmentedKalmanFiltering-BNKF/](https://github.com/agupt126/NeuralAugmentedKalmanFiltering-BNKF/)  
> 作者强调本研究展示了 **data-driven AI 与经典计算框架融合的巨大潜力**，为复杂不确定环境下的鲁棒状态估计提供了新范式。

</details>

---

### 15. [Compositional Meta-Learning for Mitigating Task Heterogeneity in Physics-Informed Neural Networks](https://arxiv.org/abs/2604.26999)

**Authors**: Beomchul Park, Minsu Koh, Heejo Kong, Seong-Whan Lee  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.26999v1  

#### Abstract
Physics-informed neural networks (PINNs) approximate solutions of partial differential equations (PDEs) by embedding physical laws into the loss function. In parameterized PDE families, variations in coefficients or boundary/initial conditions define distinct tasks. This makes training individual PI...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Compositional Meta-Learning for Mitigating Task Heterogeneity in Physics-Informed Neural Networks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Physics-Informed Neural Networks (PINNs)** 在处理**参数化偏微分方程 (parameterized PDEs)** 时面临的**任务异质性 (task heterogeneity)** 问题。具体而言：
- 不同的 PDE 参数、边界条件 (BCs) 或初始条件 (ICs) 构成不同的“任务”。
- 为每个任务单独训练 PINN 成本高昂，而直接跨任务迁移容易因任务差异大导致**负迁移 (negative transfer)**。
- 现有 meta-learning 方法（如 MAML）依赖单一全局初始化，在任务异质性强、输入特征稀疏（仅坐标输入）、训练任务数量有限的情况下表现不佳。

### 提出的新方法：LAM-PINN
作者提出了 **Learning-Affinity Adaptive Modular Physics-Informed Neural Network (LAM-PINN)**，一种**组合式元学习 (compositional meta-learning)** 框架，其核心创新点如下：

#### （1）基于学习亲和度的任务表示 (Learning-Affinity Task Representation)
- **问题**：仅用 PDE 参数难以捕捉任务间的学习动态差异。
- **解决方案**：提出将 PDE 参数与从**短暂迁移会话 (brief transfer session)** 中提取的损失动态信号结合，构建任务嵌入 (task embedding)。
- **具体做法**：对预训练模型在每个任务上进行短时间（<5% 收敛迭代数）的迁移，记录初始损失 $L_1$、最终损失 $L_2$ 和平均损失 $L_3$，与 PDE 参数拼接后经 log 变换和归一化得到 $f_a$。
- **优势**：该表示能有效反映任务间的“学习亲和度”，即使在仅有坐标输入的情况下也能实现有意义的任务聚类。

#### （2）模块化网络架构与自适应路由 (Modular Architecture with Adaptive Routing)
- **问题**：单一全局初始化无法适应异质任务。
- **解决方案**：将模型分解为：
  - **共享的元网络 (Meta-Network, MN)**：负责精细求解，保持通用性。
  - **簇专用的子网络 (Cluster-Specialized Subnetworks / Input Networks, INs)**：仅专业化于输入邻近层，捕获粗粒度任务特性。
- **自适应路由机制**：引入可学习的路由权重 $\lambda_j$，在迁移阶段动态组合不同子网络的输出，形成任务特定的初始化。
- **优势**：相比固定权重的模块复用，$\lambda$ 的自适应更新使模型能更灵活地选择最相关的知识模块。

#### （3）两阶段训练策略
1. **聚类任务训练 (Clustered Task-wise Training)**：各 IN 在对应任务簇上独立训练。
2. **元训练 (Meta-Training)**：冻结 IN 权重，训练 MN 以泛化到所有簇。

### 相比现有方法的优势
- **避免负迁移**：通过任务聚类和模块化设计，减少不相关任务间的干扰。
- **高效迁移**：在极低计算预算下（仅需传统 PINN 10% 的迭代次数）即可达到高精度。
- **轻量级**：无需复杂的辅助网络（如 Hypernetwork），计算开销小。
- **鲁棒性**：在任务异质性强、训练任务少的场景下仍表现优异。

---

## 2. 核心实验方法和设置

### 数据集与任务生成
- **任务生成方式**：采用 **三因素三水平全因子实验设计 (3-factor, 3-level full-factorial DoE)**，系统性地变化 PDE 参数、ICs 和 BCs，共生成 **27 个训练任务**，模拟真实工程中有限但多样的设计空间。
- **评估的 PDE 家族**：
  1. **Helmholtz 方程**（二维）
  2. **Burgers' 方程**
  3. **线弹性方程 (Linear Elasticity)**（含标准板和带圆孔板两种几何）
  4. **三维 Helmholtz 方程**（用于扩展性测试）

### 实验设置
- **训练任务**：27 个 DoE 生成的任务。
- **测试任务**：10 个未见过的参数配置（在 DoE 范围内插值）。
- **评估模式**：
  - 固定 10 任务基准比较（平均 MSE ± SD）
  - 重复 5 次随机种子取平均
  - 聚类稳定性分析（20 次 k-means 种子）
- **计算资源**：NVIDIA RTX 3090 GPU。

### 评估指标
- **主指标**：**均方误差 (MSE)**，衡量预测解与真解的差距。
- **收敛速度**：MSE 随训练迭代的变化曲线。
- **聚类质量**：轮廓系数 (Silhouette Score) 和调整兰德指数 (Adjusted Rand Index, ARI)。

### 基线方法对比
| 方法 | 类型 | 说明 |
|------|------|------|
| `PINN-Transfer` | 迁移学习 | 预训练模型微调 |
| `MAML` | 优化元学习 | 单一全局初始化 |
| `ConML` | 对比元学习 | 基于任务对比的目标函数 |
| `MAD` | 参数化 PINN | 学习任务隐码引导解码器 |
| `Hyper-LR-PINN` | 超网络 | 轻量级超网络生成低秩权重 |
| `DATS-w` | 难度感知采样 | 结合 MAD 与自适应任务加权 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **平均性能提升**：LAM-PINN 在三个 PDE 基准上，对未见任务实现了 **平均 19.7 倍的 MSE 减少**。
- **计算效率**：仅使用传统 PINN **10% 的每任务训练迭代次数**即达到优越性能。
- **收敛速度**：在迁移初期 MSE 下降最快，表明其初始化更接近最优解。

### 与基线方法的对比结果（Table 1）
| 方法 | Helmholtz (10 Tasks) | Burgers' (10 Tasks) | Linear Elasticity (10 Tasks) |
|------|------------------------|---------------------|-------------------------------|
| `PINN-Transfer` | 4.36E+00 | 5.08E-01 | 3.54E-02 |
| `MAML` | 4.06E+00 | 7.09E-01 | 2.91E-02 |
| `Hyper-LR-PINN` | 1.86E+00 | 1.60E-01 | 4.27E-03 |
| `LAM-PINN (Ours)` | **1.45E-01** | **5.88E-02** | **1.14E-03** |

> ✅ LAM-PINN 在所有任务上均显著优于所有基线，且统计显著 ($p < 0.05$)。

### 消融实验结果（Table 2）
验证了两个核心组件的必要性：
| 配置 | Helmholtz MSE | Burgers' MSE | 关键发现 |
|------|---------------|-------------|----------|
| 仅 PDE 参数聚类 + 自适应路由 | 1.85E+00 | 6.38E-01 | 学习亲和度信号至关重要 |
| PDE 参数 + 随机信号聚类 | 2.80E+00 | 5.95E-01 | 随机信号无帮助 |
| 学习亲和度聚类 + 固定路由 | 1.84E+00 | 4.82E-01 | 自适应路由 $\lambda$ 显著提升性能 |
| **完整 LAM-PINN** | **3.07E-01** | **1.04E-01** | 两者协同作用，缺一不可 |

---

## 4. 关键结论和发现

### 主要发现
1. **任务异质性可通过学习动态建模**：仅靠 PDE 参数不足以区分任务，而从短暂迁移中提取的**学习亲和度信号**是有效的任务相似性代理。
2. **模块化设计优于单一初始化**：将模型分解为**簇专用子网络 + 共享元网络**，能有效缓解负迁移，同时保留迁移效率。
3. **自适应路由机制有效**：可学习的 $\lambda_j$ 能自动聚焦于最相关的子网络，其选择与任务到聚类中心的距离高度一致（Fig. 10c）。
4. **高效且实用**：在资源受限的工程场景下，LAM-PINN 以极低预处理开销（<5% 预算）实现了卓越的泛化能力。

### 方法的局限性
1. **外推性能下降 (OOD Extrapolation)**：当测试任务超出 DoE 边界（如 130% 范围）时，LAM-PINN 性能退化明显，而 `Hyper-LR-PINN` 等基于连续映射的方法更具鲁棒性（Fig. 11）。
2. **依赖聚类质量**：若任务分布复杂或聚类失败，性能可能下降。
3. **模块数量固定**：当前框架需预先设定聚类数 $K$，缺乏自动扩展能力。

### 未来工作方向
1. **增强 OOD 泛化能力**：结合连续参数化方法（如 Hypernetwork）与模块化思想，提升外推鲁棒性。
2. **自动化聚类选择**：引入贝叶斯混合模型等概率方法，实现自动确定 $K$ 和路由初始化。
3. **扩展至复杂几何**：结合坐标变换或局部特征编码，处理更复杂的工业级几何形状。
4. **动态模块扩展**：允许模型在遇到新任务分布时自适应增加新模块。

---

> 🔗 **代码开源**：https://github.com/bc0322/LAM-PINN

</details>

---

### 16. [RHyVE: Competence-Aware Verification and Phase-Aware Deployment for LLM-Generated Reward Hypotheses](https://arxiv.org/abs/2604.28056)

**Authors**: Feiyu Wu, Xu Zheng, Zhuocheng Wang, Yi ming Dai, Hui Li  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.28056v1  

#### Abstract
Large language models (LLMs) make reward design in reinforcement learning substantially more scalable, but generated rewards are not automatically reliable training objectives. Existing work has focused primarily on generating, evolving, or selecting reward candidates, while paying less attention to...

---

### 17. [Perturbation Probing: A Two-Pass-per-Prompt Diagnostic for FFN Behavioral Circuits in Aligned LLMs](https://arxiv.org/abs/2604.27401)

**Authors**: Hongliang Liu, Tung-Ling Li, Yuhao Wu  
**Category**: cs.CL  
**Published**: 2026-05-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.27401v1  

#### Abstract
Perturbation probing generates task-specific causal hypotheses for FFN neurons in large language models using two forward passes per prompt and no backpropagation, followed by a one-time intervention sweep of about 150 passes amortized across all identified neurons. Across eight behavioral circuits,...

---

### 18. [Skills-Coach: A Self-Evolving Skill Optimizer via Training-Free GRPO](https://arxiv.org/abs/2604.27488)

**Authors**: Yu Tian, Jiawei Chen, Lifan Zheng, Mingxiang Tao, Xinyi Zeng, Zhaoxia Yin, Hang Su, Xian Sun  
**Category**: cs.CL  
**Published**: 2026-05-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.27488v1  

#### Abstract
We introduce Skills-Coach, a novel automated framework designed to significantly enhance the self-evolution of skills within Large Language Model (LLM)-based agents. Addressing the current fragmentation of the skill ecosystem, Skills-Coach explores the boundaries of skill capabilities, thereby facil...

---

### 19. [Fidelity, Diversity, and Privacy: A Multi-Dimensional LLM Evaluation for Clinical Data Augmentation](https://arxiv.org/abs/2604.27014)

**Authors**: Guillermo Iglesias, Gema Bello-Orgaz, Mar\'ia Navas-Loro, Cristian Ramirez-Atencia, Merc\`e Salvador Robert, Enrique Baca-Garcia  
**Category**: cs.LG  
**Published**: 2026-05-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.27014v1  

#### Abstract
The scarcity of high-quality annotated medical data, particularly in mental health, poses a significant bottleneck for training robust machine learning models. Privacy regulations restrict data sharing, making synthetic data generation a promising alternative. The use of Large Language Models (LLMs)...

---

### 20. [FMCL: Class-Aware Client Clustering with Foundation Model Representations for Heterogeneous Federated Learning](https://arxiv.org/abs/2604.27510)

**Authors**: Mahad Ali, Laura J. Brattain  
**Category**: cs.LG  
**Published**: 2026-05-01  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.27510v1  

#### Abstract
Federated Learning (FL) enables collaborative model training across distributed clients without sharing raw data, yet its performance deteriorates under statistical heterogeneity. Clustered Federated Learning addresses this challenge by grouping similar clients and training separate models per clust...

---

### 21. [TRUST: A Framework for Decentralized AI Service v.0.1](https://arxiv.org/abs/2604.27132)

**Authors**: Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.27132v1  

#### Abstract
Large Reasoning Models (LRMs) and Multi-Agent Systems (MAS) in high-stakes domains demand reliable verification, yet centralized approaches suffer four limitations: (1) Robustness, with single points of failure vulnerable to attacks and bias; (2) Scalability, as reasoning complexity creates bottlene...

---

### 22. [Exploring Sparse Matrix Multiplication Kernels on the Cerebras CS-3](https://arxiv.org/abs/2604.27985)

**Authors**: Milan Shah, Sheng Di, Michela Becchi  
**Category**: cs.DC  
**Published**: 2026-05-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.27985v1  

#### Abstract
In recent years, novel AI accelerators have emerged as promising alternatives to GPU for AI model training and inference tasks. One such accelerator, the Cerebras CS-3, achieves strong performance on large model training as well as scientific applications like molecular dynamics simulations. While d...

---

### 23. [Akita: A High Usability Simulation Framework for Computer Architecture](https://arxiv.org/abs/2604.28073)

**Authors**: Sabila Al Jannat, Ying Li, Mengyang He, Xuzhong Wang, Huizhi Zhao, Jingxiang Sun, Daoxuan Xu, Enze Xu, Yifan Sun  
**Category**: cs.DC  
**Published**: 2026-05-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.28073v1  

#### Abstract
Computer architecture simulation is essential for evaluating new designs without the need for costly tapeout. The community has developed dozens of valuable simulators that have enabled significant architectural advances. However, using and developing simulators remains a major barrier due to ad-hoc...

---

### 24. [Diagnosing Capability Gaps in Fine-Tuning Data](https://arxiv.org/abs/2604.27547)

**Authors**: Saeid Asgari Taghanaki, Rakshanda Agarwal, Bruce Sun, Rohan Jha, Elias Stengel-Eskin, Sara Malvar, Rui Ying, Yifei Xu, Guilherme Potje, Tusher Chakraborty, Leonardo de Oliveira Nunes, Ranveer Chandra, Emre Kiciman  
**Category**: cs.LG  
**Published**: 2026-05-01  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.27547v1  

#### Abstract
Fine-tuning large language models (LLMs) for domain-specific tasks requires training datasets that comprehensively cover the target capabilities a practitioner needs. Yet identifying which capabilities a dataset fails to support, and doing so before an expensive fine-tuning run, remains a largely un...

---

### 25. [Toward Personalized Digital Twins for Cognitive Decline Assessment: A Multimodal, Uncertainty-Aware Framework](https://arxiv.org/abs/2604.27217)

**Authors**: Bulent Soykan, Gulsah Hancerliogullari Koksalmis, Hsin-Hsiung Huang, Laura J. Brattain  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.27217v1  

#### Abstract
Cognitive decline is highly heterogeneous across individuals, which complicates prognosis, trial design, and treatment planning. We present the Personalized Cognitive Decline Assessment Digital Twin (PCD-DT), a multimodal and uncertainty-aware framework for modeling patient-specific disease trajecto...

---

### 26. [Safe Bilevel Delegation (SBD): A Formal Framework for Runtime Delegation Safety in Multi-Agent Systems](https://arxiv.org/abs/2604.27358)

**Authors**: Yuan Sun  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.27358v1  

#### Abstract
As large language model (LLM) agents are deployed in high-stakes environments, the question of how safely to delegate subtasks to specialized sub-agents becomes critical. Existing work addresses multi-agent architecture selection at design time or provides broad empirical guidelines, but neither pro...

---

### 27. [Generative structure search for efficient and diverse discovery of molecular and crystal structures](https://arxiv.org/abs/2604.27636)

**Authors**: Yifang Qin, Yu Shi, Junfu Tan, Chang Liu, Ming Zhang, Ziheng Lu  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.27636v1  

#### Abstract
Predicting stable and metastable structures is central to molecular and materials discovery, but remains limited by the cost of searching high-dimensional energy landscapes. Deep generative models offer efficient structure sampling, yet their outputs remain shaped by training data and can underexplo...

---

### 28. [The TEA Nets framework combines AI and cognitive network science to model targets, events and actors in text](https://arxiv.org/abs/2604.27673)

**Authors**: Sebastiano Franchini, Alexis Carrillo, Edoardo Sebastiano De Duro, Riccardo Improta, Ali Aghazadeh Ardebili, Massimo Stella  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.27673v1  

#### Abstract
We introduce Target-Event-Agent Networks (TEA Nets) as a computational framework to extract subjects (``Agents"), verbs (``Events"), and objects (``Targets") from texts. Grounded in cognitive network science and artificial intelligence, TEA Nets are implemented as an open-source Python library. We t...

---

### 29. [Iterative Multimodal Retrieval-Augmented Generation for Medical Question Answering](https://arxiv.org/abs/2604.27724)

**Authors**: Xupeng Chen, Binbin Shi, Chenqian Le, Jiaqi Zhang, Kewen Wang, Ran Gong, Jinhan Zhang, Chihang Wang  
**Category**: cs.AI  
**Published**: 2026-05-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.27724v1  

#### Abstract
Medical retrieval-augmented generation (RAG) systems typically operate on text chunks extracted from biomedical literature, discarding the rich visual content (tables, figures, structured layouts) of original document pages. We propose MED-VRAG, an iterative multimodal RAG framework that retrieves a...

---

### 30. [Decoupling the Benefits of Subword Tokenization for Language Model Training via Byte-level Simulation](https://arxiv.org/abs/2604.27263)

**Authors**: Th\'eo Gigant, Bowen Peng, Jeffrey Quesnelle  
**Category**: cs.CL  
**Published**: 2026-05-01  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.27263v1  

#### Abstract
Subword tokenization is an essential part of modern large language models (LLMs), yet its specific contributions to training efficiency and model performance remain poorly understood. In this work, we decouple the effects of subword tokenization by isolating them within a controlled byte-level pretr...

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
