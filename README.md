# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-26 06:53:45 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination](https://arxiv.org/abs/2603.24579)

**Authors**: Zhuo Li, Yupeng Zhang, Pengyu Cheng, Jiajun Song, Mengyu Zhou, Hao Li, Shujie Hu, Yu Qin, Erchao Zhao, Xiaoxi Jiang, Guanjun Jiang  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.24579v1  

#### Abstract
Hallucination remains a critical bottleneck for large language models (LLMs), undermining their reliability in real-world applications, especially in Retrieval-Augmented Generation (RAG) systems. While existing hallucination detection methods employ LLM-as-a-judge to verify LLM outputs against retri...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MARCH: Multi-Agent Reinforced Self-Check for LLM Hallucination 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
**LLM 幻觉（Hallucination）** 是大型语言模型在实际应用中的核心瓶颈，尤其在 **Retrieval-Augmented Generation (RAG)** 场景中，尽管有外部证据支持，LLM 仍可能生成与检索文档矛盾的内容（context-conflicting hallucinations）。  
现有基于 **LLM-as-a-judge** 的幻觉检测方法存在严重的 **确认偏见（confirmation bias）**：验证器（verifier）在看到原始生成结果后，倾向于“合理化”错误而非严格核查事实。

### 提出的新方法
提出 **MARCH (Multi-Agent Reinforced self-Check for Hallucination)** 框架，通过 **多智能体强化学习（MARL）** 和 **信息不对称机制** 实现自检式幻觉抑制。

#### 核心架构：三智能体协同
- **Solver**：生成初始 RAG 回答 $ y $。
- **Proposer**：将回答分解为可验证的原子命题（atomic propositions），形式为 **Question-Answer (QA) 对**。
- **Checker**：**仅基于检索文档 $ D $**，独立回答 Proposer 提出的问题，**完全不接触 Solver 的原始输出 $ y $**。

> ✅ **关键创新**：通过 **信息隔离（information asymmetry）** 打破确认偏见，确保 Checker 进行的是“盲审”（blinded audit），而非对已有文本的复述或美化。

#### 学习机制：零容忍奖励（Zero-Tolerance Reward, ZTR）
- 若任意一个 QA 对的答案不匹配，则整个响应轨迹被判定为失败（reward = 0）。
- 只有所有主张均正确时才给予正向奖励（reward = 1）。
- 强制模型实现 **“全对或全错”** 的严格事实一致性。

### 相比现有方法的优势
| 维度 | 传统方法 | MARCH |
|------|--------|-------|
| 验证方式 | 黑盒式后处理（post-hoc） | 白盒式训练内生化（end-to-end trainable） |
| 偏见控制 | 易受确认偏见影响 | 通过信息隔离消除认知污染 |
| 优化目标 | 整体输出打分（scalar reward） | 主张级细粒度监督（claim-level alignment） |
| 外部依赖 | 依赖人工标注或外部判别器 | 完全自举（self-contained），无需额外标注 |

---

## 2. 核心实验方法和设置

### 数据集
#### 训练数据集（无 ground-truth labels）
- **STEM Setting**: `BioASQ`（生物医学问答）
- **General Setting**: `2WikiMultiHopQA`, `MuSiQue`（通用多跳问答）
- 所有训练仅使用 query + retrieved documents，**不提供标准答案**，模拟真实弱监督场景。

#### 评估数据集
| 类型 | 数据集 | 描述 |
|------|-------|------|
| 幻觉评测 | `RAGTruth`, `FaithBench`, `ContextualJudgeBench`, `FACTS Grounding` | 覆盖 QA、摘要、数据转文本等任务，评估事实一致性 |
| 推理能力 | `HotpotQA`, `2WikiMultiHopQA`, `MuSiQue` | 多跳推理任务，测试下游泛化能力 |

### 实验设置
- **基础模型**：Meta-Llama3.1-8B-Instruct
- **训练框架**：VerL + PPO（Proximal Policy Optimization）
- **采样策略**：Checker 多次采样 + 多数投票以降低随机误差
- **系统提示工程**：
  - Proposer：专精于从文本中提取数值类 QA 对
  - Checker：必须引用原文作为证据，并输出纯数字

### 评估指标
- **Factuality Score (%)**：判断生成内容是否完全由文档支持
- **Consistency Rate (%)**：与参考答案一致的比例
- **Majority Voting**：每个测试样本生成 8 次，由 Qwen3-235B-A22B 作为 judge 进行评判，最终取多数票

### 基线方法对比
- **Closed-source models**: GPT-4o, Gemini 系列, Claude 系列
- **Open models**: Qwen2.5-14B, GLM-4-9B, Llama3.1-70B 等
- **RAG 增强方法**：CoT, IRCoT, Iter-RetGen, RAFT, GenGround
- **微调方法**：SFT, RLHF

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在幻觉评测上的表现（Table 2 & Figure 2）
| 模型 | RAGTruth Avg | FaithBench Avg | FACTS Grounding |
|------|-------------|----------------|----------------|
| Llama3.1-8B | 55.20% | – | 57.09% |
| **MARCH-STEM** | **74.93%** (+19.73) | – | **85.23%** |
| **MARCH-General** | **75.23%** (+20.03) | – | **80.12%** |

> 🎯 **亮点**：一个 **8B 参数模型** 经 MARCH 训练后，在 FACTS Grounding 上超越 GPT-4o（79.20%），接近 Gemini 2.5 Flash（86.60%），达到顶级闭源模型水平。

#### 在 ContextualJudgeBench 上的表现（Table 3）
| 模型 | Average Score |
|------|---------------|
| Llama3.1-8B | 29.7% |
| **MARCH-General** | **51.6%** (+21.9) |
| **MARCH-STEM** | **52.3%** (+22.6) |

> 💡 表明 MARCH 不仅提升幻觉检测，还增强了对回答质量的多维度判断能力（faithfulness, completeness, conciseness）。

#### 在多跳 QA 下游任务上的表现（Table 4）
| 方法 | Backbone | HotpotQA | MuSiQue | 2WikiMQA |
|------|----------|---------|--------|----------|
| IRCoT (GPT-4o) | GPT-4o | 66.4 | 44.2 | 78.0 |
| **MARCH (CoT)** | Llama3.1-8B | **70.6** | **36.2** | **70.6** |
| **MARCH (10-Shot)** | Llama3.1-8B | **73.6** | **40.8** | **69.4** |

> 🔥 **突破性结果**：MARCH 在 **HotpotQA 上达到 73.6%**，**超过 GPT-4o + RAG 的标准流程（64.0%）和 IRCoT（66.4%）**，证明其能显著增强小模型的复杂推理能力。

### 消融实验结果（Table 5 & Table 6）

#### 联合优化的重要性
- **仅更新 Solver** vs **联合更新 Solver + Checker**
  - STEM 上绝对增益达 **+11.6%**
  - 表明 Checker 的审计信号对防止推理漂移至关重要

#### 奖励函数对比（ZTR vs ERR）
| 奖励机制 | 设置 | 平均准确率 |
|--------|------|-----------|
| Error Rate Reward (ERR) | 按错误比例扣分 | 55.46% |
| **Zero-Tolerance Reward (ZTR)** | 全对才给分 | **61.25%** |
> ✅ 证实 **严格一致性约束更有效**

#### 奖励标量设计
- **惩罚式 (-1/0)** > **激励式 (0/1)** （59.06% vs 50.42%）
- 原因：早期训练中正确路径稀疏，“惩罚”提供更强纠错梯度

#### 跨模型家族泛化性
- 在 **Qwen3-8B** 上同样取得显著提升（+11% 以上）
- 表明 MARCH 是 **model-agnostic** 的通用范式

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **信息不对称是打破确认偏见的关键**：通过隔离 Checker 与原始输出，实现了真正独立的事实核查。
2. ✅ **零容忍奖励驱动严格事实对齐**：相比渐进式奖励，硬性约束更能抑制幻觉。
3. ✅ **多智能体共进化可内化验证逻辑**：模型在训练中自发学会“自我质疑”，形成闭环验证能力。
4. ✅ **小模型也能媲美大模型**：8B 模型经 MARCH 训练后，在多个基准上达到甚至超越 GPT-4o 和 Gemini 水平。
5. ✅ **方法具有正交性和可组合性**：可无缝集成 CoT、Few-Shot、RLHF 等技术，带来叠加收益。

### 局限性
1. **数值导向性强**：当前 Proposer 主要关注数字类主张，对非量化陈述（如因果关系、逻辑推论）覆盖有限。
2. **潜在的“少说少错”策略**：模型可能通过减少主张数量来规避惩罚（reward hacking），需引入最小提问数约束缓解。
3. **计算开销较高**：三阶段生成 + 多次采样导致训练成本上升约 3x（见 Appendix B.5）。
4. **依赖高质量 Proposer 设计**：若 Proposer 提取 QA 对不完整或模糊，会影响 Checker 的有效性。

### 未来工作方向
1. 扩展到 **非数值类主张**（如事件顺序、逻辑蕴含）的自动分解。
2. 引入 **动态难度调节机制**，让 Checker 主动挑战高风险主张。
3. 探索 **轻量化部署方案**，如蒸馏 Checker 能力至单模型。
4. 将 MARCH 范式应用于 **Agent Workflow** 中的任务规划与执行验证。
5. 结合 **形式化验证工具** 构建混合可信推理系统。

---

> 🔗 **代码开源地址**：[https://github.com/Qwen-Applications/MARCH](https://github.com/Qwen-Applications/MARCH)  
> 📚 **一句话总结**：MARCH 通过构建一个具备信息隔离的多智能体自检系统，利用强化学习将“事实核查”内化为模型自身能力，为解决 LLM 幻觉提供了可扩展、高效且无需人工标注的新路径。

</details>

---

### 2. [Residual Attention Physics-Informed Neural Networks for Robust Multiphysics Simulation of Steady-State Electrothermal Energy Systems](https://arxiv.org/abs/2603.23578)

**Authors**: Yuqing Zhou, Ze Tao, Fujun Liu  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.23578v1  

#### Abstract
Efficient thermal management and precise field prediction are critical for the design of advanced energy systems, including electrohydrodynamic transport, microfluidic energy harvesters, and electrically driven thermal regulators. However, the steady-state simulation of these electrothermal coupled ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Residual Attention Physics-Informed Neural Networks for Robust Multiphysics Simulation of Steady-State Electrothermal Energy Systems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该研究针对**稳态电热耦合多物理场系统**（steady-state electrothermal coupled multiphysics systems）的高精度模拟难题。这类系统广泛存在于微能源器件、电驱动热调控器和电液动力学传输等应用中，其挑战在于：
- 多物理场强非线性耦合（velocity, pressure, electric potential, temperature）
- 温度依赖的变系数（temperature-dependent coefficients）
- 复杂界面动态（oblique interfaces）
- 经典PINN在优化过程中易出现梯度失衡、局部特征捕捉不足等问题。

传统PINN方法在处理上述复杂场景时往往难以保持结构保真度，尤其在界面区域或系数剧烈变化区域误差显著。

---

### 提出的新方法或新思路
作者提出了一种名为 **Residual Attention Physics-Informed Neural Network (RA-PINN)** 的新型框架，其核心创新包括：

- **统一五场算子建模（Unified Five-Field Operator Formulation）**  
  将速度 $u$、压力 $p$、电势 $\phi$、温度 $T$ 和连续性约束统一为一个向量化的PDE残差算子 $ \mathcal{N}(U) = 0 $，实现对多物理场的联合求解。

- **残差注意力机制（Residual-Attention Mechanism）**  
  在网络主干中引入**残差连接 + 注意力通道调制**结构：
  - 残差路径保留深层特征传递稳定性；
  - 注意力门控放大携带陡峭梯度、局部耦合结构和界面敏感信号的信息通道。

- **自适应残差点采样（Adaptive Residual-Based Collocation Sampling）**  
  动态调整训练点分布，将更多采样点集中在残差较大的区域（如边界层、界面附近），提升难拟合区域的逼近能力。

---

### 相比现有方法的优势
| 方面 | RA-PINN优势 |
|------|-------------|
| **表达能力** | 能有效捕捉局部梯度突变与强耦合结构，优于纯MLP或LSTM类结构 |
| **鲁棒性** | 在变系数、间接约束（如pressure gauge）、斜界面等复杂条件下仍保持高精度 |
| **泛化性** | 通过注意力机制增强模型对物理结构的理解，减少过平滑现象 |

> ✅ **核心优势总结**：RA-PINN 不仅提升了精度，更增强了在工程相关复杂场景下的**稳健性（robustness）**，适用于热管理、微能源系统等实际应用场景。

---

## 2. 核心实验方法和设置

### 使用的数据集
本研究并未使用真实世界采集数据，而是基于**四个精心设计的数值基准案例（benchmark cases）**，均定义在单位正方形域 $\Omega = [0,1] \times [0,1]$ 上，并提供精确解用于验证：

| Case | 物理设定 |
|------|----------|
| **Case 1** | 常系数耦合系统（constant-coefficient coupling） |
| **Case 2** | 压力规范约束（indirect pressure-gauge constraint）$\int_\Omega p\,d\Omega = 0$ |
| **Case 3** | 温度依赖输运系数（temperature-dependent transport）<br>$v(T)=v_0(1+\rho_v T),\ \alpha(T)=\alpha_0(1+\rho_\alpha T)$ |
| **Case 4** | 斜界面一致性（oblique-interface consistency）<br>分片材料参数，强制场连续与通量跳跃 |

所有案例均由配套代码生成参考解（ground truth field maps）供定量比较。

---

### 实验设置和评估指标

#### 模型输入输出
- 输入：空间坐标 $(x, y)$
- 输出：五维场预测向量 $U(x,y;\theta) = [u,v,p,\phi,T]$

#### 损失函数构成
$$
\mathcal{L}_{\text{total}} = \lambda_{\text{res}} \|\mathcal{R}\|^2 + \lambda_b \| \mathcal{R}_b \|^2 + \lambda_{\text{gauge}} |\mathcal{R}_g|^2 + \lambda_r \|\mathcal{R}_r\|^2 + \lambda_{\text{data}} \|U - U_{\text{data}}\|^2 + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}
$$
其中包含PDE残差、边界条件、压力规范项、界面跳变项等。

#### 评估指标
- **MSE**（Mean Squared Error）
- **RMSE**（Root Mean Squared Error）
- **MAE**（Mean Absolute Error）
- **Relative L2 Error**

所有误差按字段分别计算并汇总为“Avg.”行进行整体对比。

---

### 基线方法对比
与以下三种主流PINN架构进行横向比较：
- **Pure-MLP**: 标准全连接前馈网络
- **LSTM-PINN**: 引入序列记忆结构以增强时空依赖建模
- **pLSTM-PINN**: 并行LSTM结构，提升训练效率

> 所有模型采用相同超参初始化策略，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Case | Model | Avg. MSE | Avg. Relative L2 Error |
|------|-------|-----------|------------------------|
| Case 1 | **RA-PINN** | **9.083×10⁻⁷** | **3.235×10⁻³** |
|       | LSTM-PINN     | 2.901×10⁻⁶ | 5.695×10⁻³ |
|       | pLSTM-PINN    | 1.164×10⁻⁵ | 1.205×10⁻² |
|       | Pure-MLP      | 2.249×10⁻⁴ | 5.105×10⁻² |
|  
| Case 2 | **RA-PINN** | **2.053×10⁻⁶** | **7.660×10⁻³** |
|       | LSTM-PINN     | 3.956×10⁻⁶ | 1.038×10⁻² |
|       | pLSTM-PINN    | 9.551×10⁻⁵ | 3.608×10⁻² |
|       | Pure-MLP      | 1.642×10⁻⁴ | 4.868×10⁻² |
|
| Case 3 | **RA-PINN** | **7.119×10⁻⁹** | **5.065×10⁻³** |
|       | LSTM-PINN     | 1.398×10⁻⁸ | 7.155×10⁻³ |
|       | pLSTM-PINN    | 1.719×10⁻⁴ | 8.456×10⁻¹ |
|       | Pure-MLP      | 8.434×10⁻⁷ | 3.031×10⁻² |
|
| Case 4 | **RA-PINN** | **9.845×10⁻⁸** | **1.377×10⁻³** |
|       | LSTM-PINN     | 1.159×10⁻⁷ | 1.449×10⁻³ |
|       | pLSTM-PINN    | 9.296×10⁻⁵ | 3.895×10⁻² |
|       | Pure-MLP      | 5.948×10⁻⁶ | 1.061×10⁻² |

> 🔍 **观察**：RA-PINN 在所有四个案例中均取得最低的平均误差（MSE、Relative L2等），尤其在**Case 3（温度依赖）**中领先幅度最大。

---

### 与基线方法的对比结果

| 对比维度 | 结果总结 |
|--------|---------|
| **精度全面领先** | RA-PINN 在所有字段和所有案例中达到最小误差，尤其在压力 $p$、温度 $T$ 场表现突出 |
| **面对间接约束更强** | 在pressure-gauge（Case 2）任务中，RA-PINN 成功恢复全局压力水平，而其他模型出现明显失真 |
| **应对变系数最稳定** | Case 3 中pLSTM-PINN完全失效（Relative L2 > 80%），而RA-PINN仍保持<0.5%误差 |
| **界面捕捉更清晰** | Case 4 中RA-PINN能准确重构斜界面附近的尖锐过渡，避免振荡或扩散 |

---

### 消融实验分析（隐含于案例对比中）
虽然未明确列出消融实验表格，但从不同案例的设计可视为一种“功能级消融”：

| 模块 | 验证方式 | 发现 |
|------|--------|------|
| **残差注意力机制** | 对比Pure-MLP vs RA-PINN | 显著降低误差（最高达两个数量级） |
| **自适应采样** | 观察高残差点聚集行为（Fig.1） | 更好聚焦于界面与边界层区域 |
| **统一算子框架** | 四个案例共享同一训练流程 | 支持灵活切换物理配置，具备通用性 |

此外，可视化误差图（Fig. 2–5）显示RA-PINN背景噪声最少、结构变形最小。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **RA-PINN 是当前最精确且最稳健的稳态电热耦合求解器之一**，在多种挑战性场景下均优于主流PINN变体。
2. ✅ **残差注意力机制能有效缓解多场耦合中的优化不平衡问题**，使网络同时兼顾光滑大尺度场与局部陡峭结构。
3. ✅ **自适应采样显著提升界面与高梯度区的解析能力**，是保证物理一致性的关键技术。
4. ✅ **统一五场公式化设计提高了框架复用性**，支持快速迁移至不同物理配置。

> 💡 “RA-PINN not only fits better — it understands the physics better.”

---

### 方法的局限性
| 局限 | 描述 |
|------|------|
| ⏱️ **训练耗时长** | 如Table 2所示，RA-PINN训练时间远高于基线：<br>- Case 1: 24.01h vs pLSTM-PINN 1.09h<br>- Case 3: 39.81h（最长） |
| 🧠 **结构复杂度高** | 残差+注意力模块增加实现难度，不利于轻量化部署 |
| 📈 **尚未扩展到瞬态或多尺度问题** | 当前仅验证稳态情形，动态响应能力待验证 |

---

### 未来工作方向
1. **加速训练策略**：探索混合精度训练、分布式优化、warm-start初始化等方式缩短收敛时间。
2. **推广至三维与工业几何**：结合CAD集成与非结构网格映射，应用于真实设备仿真。
3. **嵌入数字孪生系统**：作为在线仿真引擎支持实时监测与控制。
4. **与其他神经算子融合**：尝试与Neural Operator或Transformer结合，提升泛化能力。
5. **不确定性量化**：引入贝叶斯PINN或Dropout机制估计预测置信区间。

---

## 总结

| 维度 | 内容 |
|------|------|
| **核心思想** | 将残差注意力机制引入PINN，构建RA-PINN以增强对局部结构和强耦合场的建模能力 |
| **技术亮点** | 统一五场PDE算子 + 残差注意力模块 + 自适应残差采样 |
| **实验证明** | 在4个典型电热耦合基准上全面超越Pure-MLP、LSTM-PINN、pLSTM-PINN |
| **适用场景** | 热管理系统、微能源器件、电热调节器等需要高保真多物理场仿真的领域 |
| **开源信息** | 代码已公开于GitHub：<br>[https://github.com/Uderwood-TZ/RA-PINN](https://github.com/Uderwood-TZ/RA-PINN-Application-of-RA-PINN-in-Solving-Steady-State-Electrothermal-Coupled-Multiphysics-Problems) |

> ✅ **一句话总结**：  
> RA-PINN 通过**残差注意力机制**与**自适应采样策略**，实现了对稳态电热耦合系统的**高精度、强鲁棒性模拟**，为复杂能量系统的数字化设计提供了可靠工具。

</details>

---

### 3. [Symbolic--KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning](https://arxiv.org/abs/2603.23854)

**Authors**: Salah A Faroughi, Farinaz Mostajeran, Amirhossein Arzani, Shirko Faroughi  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.23854v1  

#### Abstract
Symbolic discovery of governing equations is a long-standing goal in scientific machine learning, yet a fundamental trade-off persists between interpretability and scalable learning. Classical symbolic regression methods yield explicit analytic expressions but rely on combinatorial search, whereas n...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Symbolic-KAN: Kolmogorov-Arnold Networks with Discrete Symbolic Structure for Interpretable Learning**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
在科学机器学习（Scientific Machine Learning, SciML）中，存在一个长期存在的**可解释性与可扩展性之间的权衡**：
- **传统符号回归方法**（如遗传算法、SINDy）能生成显式的解析表达式，具有高度可解释性，但依赖组合搜索，计算成本高，难以扩展到高维问题。
- **深度神经网络**（如MLP、PINN）训练高效、可扩展性强，但其内部表示是“黑箱”的，缺乏可解释性，无法直接提供物理机制洞察。

本文旨在弥合这一鸿沟：如何构建一种既能**从数据中自动发现简洁、可读的符号表达式**，又能**像神经网络一样高效训练并处理复杂高维问题**的模型。

---

### **提出了什么新方法或新思路**
作者提出了一种名为 **Symbolic-KAN**（Symbolic Kolmogorov-Arnold Networks）的新架构，其核心思想是将**离散的符号结构嵌入到可训练的深度网络中**，实现端到端的可解释学习。

#### **关键创新点包括**：
1. **基于Kolmogorov-Arnold定理（KART）的架构设计**  
   利用KART理论，将多元函数分解为一元函数的叠加。Symbolic-KAN在此基础上引入更灵活的学习机制。

2. **动态符号基元选择机制**  
   每个网络单元（unit）通过一个**可学习的标量投影**（learned scalar projection）将输入特征压缩为一维坐标，并在其上应用一组候选的**解析基元**（analytic primitives），如 `{x, x², sin(x), cos(x), tanh(x), exp(x), log(1+|x|)}`。

3. **分层门控机制（Hierarchical Gating）与符号正则化**  
   引入三种可微分的软门控机制，在训练过程中逐步将连续混合转化为离散选择：
   - **Primitive Selection Gate**：使用 **Gumbel-Softmax** 机制，使每个边（edge）最终只激活一个基元。
   - **Edge Selection Mask**：每个单元只保留一个最优的投影边。
   - **Unit Gate**：决定是否保留该隐藏单元，实现结构稀疏化。

4. **无需后处理的符号提取**  
   经过“门控训练 + 离散化”后，网络自然收敛为**紧凑的闭式表达式**（closed-form expression），无需额外的符号拟合步骤。

5. **作为可扩展的基元发现引擎**  
   Symbolic-KAN不仅能用于建模，还能作为**预处理器**，自动识别最相关的解析项，为SINDy等稀疏回归方法构建高质量的候选库。

---

### **相比现有方法的优势**
| 方法 | 可解释性 | 可扩展性 | 是否需后处理 | 能否发现新结构 |
|------|----------|------------|----------------|------------------|
| **Symbolic Regression (e.g., Genetic Programming)** | 高 | 低 | 否 | 是（但效率低） |
| **SINDy** | 高 | 中 | 否 | 否（依赖预设库） |
| **Standard KAN / PINN** | 低 | 高 | 是（困难） | 是 |
| **Symbolic-KAN (本文)** | **高** | **高** | **否** | **是（通过库内组合）** |

> ✅ **优势总结**：兼具**高可解释性**与**高可扩展性**，支持端到端符号发现，避免“表达式膨胀”（expression bloat），且能适应未知非线性参数（如非整数指数）。

---

## 2. 核心实验方法和设置

### **使用的数据集与任务类型**
实验覆盖三大类科学学习场景：

| 任务类别 | 具体问题 | 数据来源 |
|--------|---------|--------|
| **Data-driven Regression** | 多变量函数逼近：<br>- $F(x) = x^2$<br>- $F(x) = \frac{\sin(3x)}{1+x^2} + 0.4\cos(5x)$ | 人工生成采样点 |
| **Inverse Dynamical Systems** | Van der Pol Oscillator 参数识别：<br>$\dot{x} = ay$, $\dot{y} = \mu(1-x^{2.15})y - cx$ | 数值模拟（RK45）生成轨迹 |
| **Physics-Informed PDE Learning** | - Reaction-Diffusion 方程逆问题<br>- Laplace 方程正问题 | 解析解构造源项与边界条件 |

---

### **实验设置与评估指标**

#### **通用设置**
- **网络结构**：多层Symbolic-KAN，典型配置 `[L, Ke, E] = [4,6,3]`（4层，每层6单元，每单元3边）
- **基元库**：$\mathcal{P} = \{0,1,x,x^2,x^3,\sin x,\cos x,\tanh x,e^x,\log(1+|x|)\}$ 或子集
- **训练流程**：
  1. **Stage I**：使用Gumbel-Softmax进行软选择训练，温度 $T(t)$ 逐渐退火。
  2. **Stage II**：硬离散化（one-hot），固定结构，使用L-BFGS精细优化。
- **损失函数**：
  $$
  \mathcal{L} = \lambda_{\text{data}}\mathcal{L}_{\text{data}} + \mathcal{L}_{\text{phys}} + \lambda_{\text{sel}}(t)\mathcal{L}_{\text{sel}} + \lambda_{\text{unit}}\mathcal{L}_{\text{unit}} + \lambda_{\text{bias}}\mathcal{L}_{\text{bias}}
  $$
  其中 $\mathcal{L}_{\text{sel}}$ 包含熵正则与NMS（Non-Maximum Suppression）以促进稀疏和多样性。

#### **评估指标**
- **相对误差（Relative Error）**：
  $$
  \mathcal{E}(F_u) = \frac{\|F_u - F_{\text{true}}\|}{\|F_{\text{true}}\|}
  $$
  对于函数场使用 $L^2$ 范数，对于参数使用绝对误差。
- **参数识别精度**：比较恢复参数与真实值的相对误差。
- **符号结构一致性**：检查所选基元是否匹配真实系统中的解析形式（如sin/cos对应振荡行为）。

---

### **基线方法对比**
| 基线方法 | 描述 |
|--------|------|
| **PINN** | 标准Physics-Informed Neural Network，使用tanh激活 |
| **cPIKAN** | Chebyshev-based Physics-Informed KAN，基于切比雪夫多项式 |
| **SINDy** | Sparse Identification of Nonlinear Dynamics（文中未直接对比，但作为动机提及） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **(1) 数据驱动回归（Table 1）**
| 目标函数 | 相对误差 $\mathcal{E}(F_u)$ | 结果说明 |
|--------|----------------------------|---------|
| $F(x)=x^2$ | $1.04 \times 10^{-5}$ | 成功识别出 $x$ 和 $x^2$ 项，其余被剪枝 |
| $F(x)=\frac{\sin(3x)}{1+x^2} + 0.4\cos(5x)$ | $7.75 \times 10^{-3}$ | 正确选择 `sin`, `cos`, `lorentz (1/(1+x))` 等基元 |

> 🔍 **可视化分析（Fig. 2）** 显示模型不仅准确插值，还在外推区域保持正确趋势，残差极小。

---

#### **(2) Van der Pol 振子参数识别（Table 2）**
| 时间区间 | 相对轨迹误差 $\mathcal{E}(u_e)$ | 参数估计误差 |
|--------|-------------------------------|-------------|
| $t \in [0,20]$ | $6.02 \times 10^{-4}$ | $a$: <0.01%, $\mu$: ~1%, $c$: <0.02% |
| $t \in [0,50]$ | $5.87 \times 10^{-3}$ | 所有参数误差 < 1% |

> 📌 **发现**：即使面对非整数幂次 $x^{2.15}$ 和强非线性阻尼，Symbolic-KAN仍能稳定识别参数，并选择 `sin`, `cos`, `x` 等合理基元。

---

#### **(3) Reaction-Diffusion 方程（Table 3, Fig. 5）**
| 方法 | 域 | 验证误差 $\mathcal{E}(u_e)$ | 反应系数 $K$ 估计 |
|-----|----|---------------------------|------------------|
| **Symbolic-KAN** | $[-2,2]$ | $5.93 \times 10^{-4}$ | $0.6994$ (误差 <0.1%) |
| **cPIKAN** | $[-2,2]$ | $8.25 \times 10^{-4}$ | $0.6998$ |
| **Symbolic-KAN** | $[-4,4]$ | $9.37 \times 10^{-3}$ | $0.6985$ |
| **cPIKAN** | $[-4,4]$ | $2.07 \times 10^{-1}$ | $0.6611$ |
| **PINN** | $[-4,4]$ | $2.15 \times 10^{-1}$ | $0.6809$ |

> ✅ **结论**：当域扩大时，Symbolic-KAN误差下降约 **95%** 于cPIKAN，参数估计精度提升 **5倍以上**。

---

#### **(4) Laplace 方程求解（Table 4, Fig. 6）**
| 方法 | 验证误差 $\mathcal{E}(u_e)$ | 最大绝对误差降低 |
|-----|---------------------------|------------------|
| **Symbolic-KAN** | $1.11 \times 10^{-3}$ | — |
| **cPIKAN** | $8.76 \times 10^{-3}$ | ↓92% |
| **PINN** | $2.71 \times 10^{-3}$ | ↓70% |

> 🎯 **符号发现**：模型自动选择了 `sin`, `cos`, `sinh`, `cosh` 等与真解 $u(x,y)=\sin(\pi x)\sinh(\pi y)$ 一致的基元。

---

### **消融实验（隐含分析）**
虽然未明确列出消融表，但从以下方面体现了设计有效性：
- **门控机制必要性**：若无Gumbel-Softmax与熵正则，无法实现干净的一热选择。
- **双仿射参数作用**：外部投影 $(w,b)$ 控制特征融合几何，内部 $(\gamma,\beta)$ 控制基元局部形变，分离二者提升训练稳定性与可解释性。
- **温度退火策略**：控制探索-利用平衡，防止早熟收敛。

---

## 4. 关键结论和发现

### **主要发现**
1. ✅ **Symbolic-KAN 能可靠地从数据中恢复正确的原始项和控制结构**，无论是在纯数据驱动、动力系统还是PDE场景下。
2. ✅ 在**前向与逆向PDE学习**中，均能产生**高精度解**的同时输出**紧凑的符号表示**，其选择的基元反映真实方程的数学结构。
3. ✅ 相比PINN和cPIKAN，**在更大空间域或更强非线性下仍保持稳定性和高精度**，尤其在参数识别任务中表现卓越。
4. ✅ 可作为**可扩展的基元发现工具**，为SINDy等方法提供数据驱动的候选库，缓解其对先验知识的依赖。

---

### **方法的局限性**
1. ❗ **不能完全解决外推泛化问题**：尽管在某些案例中外推良好，但神经网络固有的外推风险依然存在。
2. ❗ **依赖基元库的设计**：若真实项不在库中（如特殊函数），则无法发现；目前仅支持库内项的组合，而非创造全新函数形式。
3. ❗ **训练过程较复杂**：涉及两阶段训练、温度调度、多种正则项，调参难度高于标准网络。
4. ❗ **尚未验证极端高维问题**：当前实验集中在低至中等维度，大规模应用有待验证。

---

### **未来工作方向**
1. **自动化基元库扩展**：结合符号回归或生成模型，动态生成新的候选函数加入库中。
2. **应用于更复杂的PDE系统**：如Navier-Stokes、Maxwell方程组等多场耦合问题。
3. **集成到闭环控制系统**：利用其可解释性进行安全关键系统的建模与决策。
4. **发展统一框架**：将Symbolic-KAN与SINDy、Bayesian方法结合，形成“发现-验证-修正”的完整科学发现流水线。
5. **硬件加速与稀疏推理优化**：利用其最终稀疏结构实现实时部署。

---

> 💡 **总体评价**：  
> **Symbolic-KAN 是迈向“可解释、可扩展、机制性”科学机器学习的重要一步**。它成功地将神经网络的强大拟合能力与符号模型的透明性结合起来，为从数据中自动发现物理定律提供了实用而强大的新范式。

</details>

---

### 4. [CoordLight: Learning Decentralized Coordination for Network-Wide Traffic Signal Control](https://arxiv.org/abs/2603.24366)

**Authors**: Yifeng Zhang, Harsh Goel, Peizhuo Li, Mehul Damani, Sandeep Chinchali, Guillaume Sartoretti  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.24366v1  

#### Abstract
Adaptive traffic signal control (ATSC) is crucial in alleviating congestion, maximizing throughput and promoting sustainable mobility in ever-expanding cities. Multi-Agent Reinforcement Learning (MARL) has recently shown significant potential in addressing complex traffic dynamics, but the intricaci...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：CoordLight: Learning Decentralized Coordination for Network-Wide Traffic Signal Control**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**自适应交通信号控制（ATSC）**中的两个核心挑战：
- **部分可观测性（Partial Observability）**：在去中心化多智能体系统中，每个路口（agent）只能获取局部交通状态，难以全面理解全局交通动态。
- **协调困难（Coordination Difficulty）**：缺乏有效的机制来建模相邻路口之间的动态依赖关系，导致决策短视、局部最优，影响网络级交通优化。

传统方法如固定时序控制（Fixed-Time）或简单的压力控制（MaxPressure）无法应对复杂动态流量；而现有的 MARL 方法虽有潜力，但在状态表示和邻居协作建模方面仍存在不足。

---

### **提出的新方法与新思路**

#### **(1) Queue Dynamic State Encoding (QDSE)**  
一种**基于车辆排队动力学模型的状态编码方法**，用于增强单个路口对当前及未来拥堵的感知能力。  
QDSE 包含六个车道级特征向量：
- `Q(t)`：停止车辆数（队列长度）
- `Nin(t)`：进入车辆数
- `Nout(t)`：离开车辆数
- `Nr(t)`：移动车辆总数
- `Nfr(t)`：紧随首辆移动车后的车辆数
- `Dfr(t)`：首辆移动车到队尾的距离

> ✅ **优势**：不仅反映当前状态，还能预测下一阶段的入队增量（Δn(t)），使策略更具前瞻性，而非仅反应式响应。

#### **(2) Neighbor-aware Policy Optimization (NAPO)**  
一个**完全去中心化的 MARL 算法**，通过注意力机制识别并加权关键邻居的影响，提升协调效率。

核心设计包括：
- **注意力机制（Attention Mechanism）**：动态计算邻居状态与动作的重要性权重，实现“选择性关注”。
- **增强型 Actor-Critic 架构**：
  - **Actor Network**：采用时空聚合单元（Spatio-Temporal Network, STN），融合空间（邻居状态）和时间（历史状态）信息。
  - **Critic Network**：引入**状态-动作解码器（State-action Decoder）**，显式建模邻居动作的历史依赖，改进优势函数估计。
- **改进的优势函数（Advantage Estimation）**：将邻居影响纳入价值函数，生成更准确的策略梯度更新信号。

> ✅ **优势**：避免盲目协作，聚焦于真正有影响力的邻居，提高训练稳定性与策略质量。

---

### **相比现有方法的优势**
| 维度 | CoordLight | 典型基线方法（如 CoLight、MPLight） |
|------|-----------|-------------------------------|
| **状态表示** | 显式建模排队动态，支持拥堵预测 | 多为静态统计量（如车数、压力） |
| **协调机制** | 动态注意力 + 显式状态-动作依赖建模 | 固定图结构/GAT 或隐式通信 |
| **架构灵活性** | 支持局部观测下的去中心化执行 | 部分依赖中心化训练（CTDE） |
| **可扩展性** | 在196路口大规模网络中表现优异 | 随规模增大性能下降明显 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
三个真实城市交通路网，均来自 **CityFlow** 开源仿真平台：
| 数据集 | 路口数量 | 特点 |
|--------|----------|------|
| **Jinan (China)** | 3×4 = 12 个路口 | 小规模，三种不同流量需求（DJN1–3） |
| **Hangzhou (China)** | 4×4 = 16 个路口 | 中等规模，两种流量需求（DHZ1–2） |
| **New York (USA)** | 7×28 = 196 个路口 | **最大规模测试，极具挑战性** |

所有路口均为四向标准交叉口，每方向三车道。

---

### **实验设置**
- **仿真工具**：CityFlow
- **episode长度**：3600秒
- **相位持续时间**：固定5秒（含2秒黄灯过渡）
- **训练方式**：同质策略（homogeneous policy），即所有路口共享网络参数
- **评估方式**：运行10次独立测试，报告平均值±标准差

---

### **评估指标**
主指标为 **Average Travel Time (sec)** ——越低越好  
辅助分析指标：
- 平均队列长度（Average Queue Length）
- 车辆速度（Speed）
- 队列长度标准差（Std. of Queue）

---

### **基线方法对比**
分为两类：

#### **传统方法**
- **Fixed-Time**：固定周期配时
- **MaxPressure (MP)**：基于上下游车流差的压力控制
- **Advanced-MP**：考虑有效范围内的移动车辆

#### **MARL 方法**
- **CoLight / Advanced-CoLight**：基于 GAT 的图注意力协作
- **MPLight / Advanced-MPLight**：结合压力指标与 FRAP 架构
- **DenseLight**：密集反馈机制
- **SocialLight**：分布式合作学习框架（第二优基准）

> 注：部分结果因无法复现，引用原论文数据（标记为 *）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table II）**

| 方法 \ 数据集 | DJN(1) | DHZ(2) | DNY(2) |
|-------------|--------|--------|--------|
| Fixed-Time | 346.36 | 359.44 | 1660.29 |
| MaxPressure | 273.96 | 348.98 | 1535.77 |
| Advanced-MP | 253.61 | 318.67 | ~1060* |
| CoLight | 276.33 | 297.26 | 1476.18 |
| Advanced-CoLight | 253.95 | 308.62 | 1025.47 |
| DenseLight* | 226.97 | 272.27 | 803.42 |
| SocialLight | 217.92 | 288.55 | 1106.69 |
| **CoordLight (Ours)** | **199.24** | **250.87** | **1039.15** |

> ✅ 所有场景下均取得最佳性能！

---

### **与基线方法的对比结果**
- 在 **Jinan** 上，相比 SocialLight 提升 **6.39% ~ 9.23%**
- 在 **Hangzhou DHZ(2)**（高负载）上，比 DenseLight 提升 **7.87%**
- 在 **New York DNY(2)**（196路口）上，比第二好的 SocialLight 缩短 **6.1%** 行程时间
- **未配对 t-test + Bonferroni 校正**：所有 p-value << 1.57e-4 → 性能提升具有显著统计意义

> 📈 图6显示 CoordLight 在各路口的平均行程时间和方差均为最低，表明其协调一致性最强。

---

### **消融实验结果（Ablation Studies）**

#### **(1) QDSE 状态表示的有效性（Fig. 7a）**
比较五种状态定义：
- VC（Vehicle Count）
- GP / EP（General/Efficient Pressure）
- ATS（Advanced Traffic State）
- DTSE（图像网格化表示）
- **QDSE（本文）**

✅ 结果：
- QDSE 在 **平均旅行时间、队列长度、速度、队列波动性** 上全面优于其他方法
- 即使与更复杂的 DTSE 相比，QDSE 也略胜一筹，说明其在**精度与复杂度之间取得良好平衡**

#### **(2) 模块组件消融（Fig. 7b）**
| 变体 | 描述 | 影响 |
|------|------|------|
| **w/o QDSE** | 替换为 Vehicle Count | 性能大幅下降，验证 QDSE 的重要性 |
| **w/o STN** | 移除时空注意力网络 | 无法捕捉时空依赖，收敛至次优解 |
| **w/o AD** | 移除状态-动作解码器 | 价值函数估计不准，训练不稳定 |
| **CoordLight-Base** | 仅用 FC 层 + IPPO | 远逊于完整模型 |

✅ 发现：**QDSE 和 NAPO 各组件协同作用显著，缺一不可**

#### **(3) 对传感器噪声的鲁棒性（Fig. 8）**
在 QDSE 输入中加入高斯噪声（σ=10m, 20m, 30m）模拟摄像头定位误差：
- 最大旅行时间增加不超过 **2.34%**
- 性能退化平缓，表现出强健鲁棒性

> ✅ 表明 QDSE 具备实际部署潜力

---

## **4. 关键结论和发现**

### **主要发现**
1. **精准的状态表示是高效决策的前提**：QDSE 通过建模排队动态，赋予 agent “预见未来”的能力，显著优于仅使用车数或压力的传统表示。
2. **有针对性的协调优于全连接协作**：NAPO 利用注意力机制识别关键邻居，实现“按需协作”，避免冗余通信与干扰。
3. **去中心化架构仍可实现高性能**：无需中心化训练（CTDE），CoordLight 在196路口的大规模网络中依然稳定且高效。
4. **模块化设计带来广泛适用性**：QDSE 可被集成进其他 MARL 方法（如 IPPO、SocialLight），普遍提升性能。

---

### **方法的局限性**
- 当前假设为**规则网格状路网**，尚未验证在异构路口（非对称、环岛等）上的泛化能力
- 依赖**高质量摄像头输入**（位置、速度），若传感器严重失效可能影响 QDSE 效果
- 动作空间限定为**固定时长相位切换**，未涉及动态相位时长调整
- 未处理突发事件（如事故、紧急车辆优先通行）

---

### **未来工作方向**
1. **拓展至异构网络**：支持不同类型路口、车道配置、非同步信号控制
2. **增强现实适应性**：
   - 引入容错机制应对传感器缺失/延迟
   - 支持在线迁移学习以适应突发交通事件
3. **支持更灵活的动作空间**：
   - 学习动态相位持续时间
   - 决策是否保持当前相位
4. **整合更高层次的城市交通管理目标**：
   - 绿波带协调
   - 应急车辆优先通行
   - 碳排放最小化
5. **探索与宏观交通流模型的联合优化**

---

> 🔚 **总结**：  
> **CoordLight 是一项面向大规模城市交通优化的前沿 MARL 框架**，它通过 **QDSE + NAPO** 的双重创新，在状态建模与多智能体协调两方面实现了突破。实验证明其在多个真实城市数据集上显著超越 SOTA 方法，具备良好的稳定性、可扩展性与实际应用前景。代码已开源：[https://github.com/marmotlab/CoordLight](https://github.com/marmotlab/CoordLight)

</details>

---

### 5. [Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation](https://arxiv.org/abs/2603.23838)

**Authors**: Han Zheng, Yining Ma, Brandon Araki, Jingkai Chen, Cathy Wu  
**Category**: cs.AI  
**Published**: 2026-03-26  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.23838v1  

#### Abstract
Lifelong Multi-Agent Path Finding (MAPF) is critical for modern warehouse automation, which requires multiple robots to continuously navigate conflict-free paths to optimize the overall system throughput. However, the complexity of warehouse environments and the long-term dynamics of lifelong MAPF o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **Lifelong Multi-Agent Path Finding (MAPF)** 在现代仓库自动化中的应用挑战。传统 MAPF 方法多为静态一次性规划，而现实中的仓库系统要求机器人持续执行任务（如分拣、搬运），存在以下长期动态挑战：
- 任务连续到达，需不断重新协调路径；
- 拥堵模式随时间演变，需要前瞻性决策；
- 局部次优决策可能导致级联拥堵甚至死锁；
- 路径质量必须在有限时间内求解。

现有基于搜索的方法（如 CBS、PBS）难以扩展到大规模场景，而纯学习方法又未能稳定超越经典启发式方法。

### 提出的新方法与新思路
作者提出了 **RL-guided Rolling Horizon Prioritized Planning (RL-RH-PP)**，这是首个将 **强化学习 (RL)** 与 **基于搜索的规划器** 结合用于 Lifelong MAPF 的框架。

#### 核心设计思想：
- **Rolling Horizon Prioritized Planning (RH-PP)** 作为基础架构：采用滚动时域机制，在每个时间窗口内使用 Prioritized Planning 进行路径规划，支持持续任务分配。
- **RL 动态生成优先级顺序**：将动态优先级分配建模为一个 **Partially Observable Markov Decision Process (POMDP)**，利用一个基于注意力机制的神经网络自回归地解码高质量的全局优先级顺序。
- **学习引导而非替代搜索**：不端到端生成路径，而是让 RL 学习“如何排序”，再由高效的单智能体路径规划器（如 SIPP）完成具体路径计算，实现“学习指导搜索”（learning-guided planning）。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|--------|
| **效率与可扩展性** | PP 的复杂度是线性的，远优于指数级增长的 CBS/PBS；适合大规模机器人系统。 |
| **长期协调能力** | RL 政策能捕捉时空依赖关系，主动缓解拥堵，避免短视行为导致的级联低效。 |
| **灵活性与泛化性** | 模型可在不同 agent 密度、规划窗口、未知地图布局下实现零样本迁移（zero-shot generalization）。 |
| **性能提升显著** | 平均比随机优先级 RH-PP 提升 **25% 吞吐量**，并在高密度环境下全面超越主流基线。 |

---

## 2. 核心实验方法和设置

### 数据集与仿真环境
构建了两个真实世界启发的仓库模拟环境：
1. **Amazon Fulfillment Center Dense Map**
   - 障碍物密度：15.3%
   - 多条平行通道，窄走廊
2. **Symbotic Warehouse Map**
   - 障碍物密度高达 **56.6%**
   - 包含瓶颈区域、交叉通道等易拥堵结构
   - 首次引入该布局至 Lifelong MAPF 研究中

> 所有实验基于开源 MAPF benchmark 构建，并集成任务调度逻辑。

### 实验设置
- **仿真时长**：T = 800 时间步
- **规划周期**：每 h=5 步进行一次 replanning
- **规划视野**：w ∈ {5, 10, ..., 30}
- **Agent 数量**：N ∈ {80, 100, 120}，训练于 N=120，测试时进行 zero-shot 泛化
- **硬件资源限制**：每步 CPU 时间预算 1 秒

### 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput per Agent (TPA)** | 单个 agent 成功完成的任务数（平均） |
| **Total Throughput** | 所有 agent 完成任务总数（TPA × N） |
| **Solve Time** | 每次规划步骤的平均耗时（CPU wall time + GPU 推理时间） |

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **RH-CBS** | 搜索-based（最优） | 滚动时域下的 Conflict-Based Search，保证局部最优但扩展性差 |
| **RH-PBS** | 搜索-based | Priority-Based Search，支持部分优先级，较高效 |
| **PIBT** | 分布式贪心 | 实时性强，但缺乏全局视角，易陷入局部拥堵 |
| **WPPL** | 混合方法 | 2023 League of Robot Runner 冠军方案，结合 PIBT 初始规划与 LNS 优化 |
| **RH-PP (Random)** | 启发式 | 使用随机采样优先级的 RH-PP，作为本文方法的基础对照 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 场景 | 方法 | TPA | 总吞吐量 | Solve Time (s) |
|------|------|-----|----------|----------------|
| Amazon (N=120) | RL-RH-PP (K=5) | **25.56±0.55** | **3067.2** | 0.96 |
| | WPPL | 23.59±0.26 | 2830.8 | ~1.0 |
| | RH-PBS | 3.37±0.26 | 404.4 | 1.00 |
| Symbotic (N=120) | RL-RH-PP (K=5) | **11.31±2.21** | **1357.2** | 0.99 |
| | RH-PBS | 1.76±1.10 | 211.2 | 1.00 |
| | WPPL | 10.05±1.33 | 1206.0 | ~1.0 |

> ✅ **RL-RH-PP 在两种地图上均取得最高总吞吐量**

### 与基线方法对比结果
- 在 Amazon 地图上，相比 WPPL 提升约 **8.5% 总吞吐量**；
- 在 Symbotic 高障碍密度地图上，**大幅领先所有搜索方法**，尤其在高 agent 密度下表现稳健；
- 相比 RH-PP (random)，平均提升 **25% 吞吐量**，证明 RL 引导的有效性；
- 在推理速度方面，虽略高于轻量级方法（如 PIBT），但仍控制在 1 秒以内，满足实时需求。

### 消融实验结果

#### （1）奖励函数权重影响（ablation on K 和 σ）
- 设置 `K=1000`, `σ=1000` 可获得最佳收敛速度与最终性能；
- 若忽略不可行惩罚（σ=0），模型更难学会避免死锁；
- 结果表明两项惩罚对密集场景至关重要。

#### （2）编码器结构消融
| 编码器变体 | Symbotic 地图最终吞吐量 |
|-----------|------------------------|
| Full Model（时空注意力） | **1539.1** |
| w/o Temporal Attention | 1250.0 |
| w/o Spatial Attention | 1357.2 |
| 替换为 Yan & Wu (2024) 架构 | 未收敛 / 明显下降 |

> ⚠️ 表明 **时序注意力** 对处理长程依赖尤为关键，尤其是在狭窄通道中预测拥堵演化。

#### （3）优先级策略对比
| 方法 | Amazon TPA (N=120) | Symbotic TPA (N=120) |
|------|--------------------|-----------------------|
| DQ-RH-PP（距离查询启发式） | 17.66 | 9.88 |
| RL-RH-PP (K=5) | **25.56** | **11.31** |

> ✅ 学习得到的优先级明显优于手工设计规则。

#### （4）上下文 Bandit 消融（horizon=1）
- 将 RL 替换为仅考虑当前状态的 contextual bandit 模型；
- 虽然初期学习更快，但最终性能显著低于完整 RL-RH-PP；
> 🔍 说明 **多步前瞻能力** 是实现长期高效协调的关键。

---

## 4. 关键结论和发现

### 主要发现
1. **学习可以有效增强经典启发式方法**  
   RL 不必完全取代搜索算法，而是可以通过学习“更好的先验”（如优先级顺序）来显著提升其性能。

2. **RL-RH-PP 能主动管理拥堵**
   - 可视化分析显示，RL 政策会**主动赋予拥堵区内的 agent 更高优先级**；
   - 能够通过“反向移动”等方式为内部 agent 清路，打破潜在死锁；
   - 展现出类似“疏导交通”的高级协调行为。

3. **强大的零样本泛化能力**
   - 在未见过的 agent 数量、规划窗口长度、甚至全新仓库布局（如 In/Out Switch）上仍保持高性能；
   - 得益于基于位置嵌入（position embedding）和注意力机制的设计，具备良好空间理解能力。

4. **适用于高密度、高障碍复杂环境**
   - 在 Symbotic 地图这种极端条件下，传统搜索方法几乎失效，而 RL-RH-PP 依然维持较高吞吐量；
   - 显示出在工业级实际部署中的潜力。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **绝对坐标依赖** | 当前模型依赖固定大小的地图索引，无法直接迁移到尺寸不同的新地图； |
| **Top-K 串行评估开销大** | 当 K 较大时（如 K=200），CPU 上串行评估成为瓶颈； |
| **未联合优化任务分配** | 当前仅解决路径规划，未与任务指派（task assignment）联合建模； |
| **训练成本较高** | 单次训练最长需约 4 天（N=120）； |

### 未来工作方向
1. **实现全地图无关表示**（map-agnostic representation）  
   设计相对坐标或图神经网络编码器，以支持任意规模地图的零样本迁移。

2. **并行化 Top-K 评估流程**  
   利用多线程或多 GPU 加速候选优先级的批量评估，降低延迟。

3. **扩展至联合任务与路径规划**  
   修改 autoregressive decoder 输出 `(agent, task)` 序列，统一优化任务分配与路径协调。

4. **加入鲁棒性建模**  
   考虑传感器噪声、运动延迟、通信中断等不确定性因素，提高现实适用性。

5. **探索其他应用场景**  
   如机场地面调度、自动驾驶车队协同、视频游戏 NPC 控制等具有长期动态特性的领域。

---

> 📌 **总结一句话**：  
> 本论文提出的 **RL-RH-PP** 框架成功实现了 **learning-guided planning** 的范式突破，在 Lifelong MAPF 中通过 RL 学习动态优先级，显著提升了仓库机器人的整体吞吐量与抗拥堵能力，且具备出色的泛化性和实用性，为未来智能仓储系统的路径协调提供了新的技术路径。代码已开源：[https://github.com/MikeZheng777/RL-RH-PP](https://github.com/MikeZheng777/RL-RH-PP)

</details>

---

### 6. [On Gossip Algorithms for Machine Learning with Pairwise Objectives](https://arxiv.org/abs/2603.24128)

**Authors**: Igor Colin (LTCI, S2A, IP Paris), Aur\'elien Bellet (PREMEDICAL), Stephan Cl\'emen\c{c}on (LTCI, IDS, S2A, IP Paris), Joseph Salmon (IROKO, UM)  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.24128v1  

#### Abstract
In the IoT era, information is more and more frequently picked up by connected smart sensors with increasing, though limited, storage, communication and computation abilities. Whether due to privacy constraints or to the structure of the distributed system, the development of statistical learning me...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：On Gossip Algorithms for Machine Learning with Pairwise Objectives

## 1. 论文的主要贡献和创新点

### 解决的问题
本文致力于解决**分布式机器学习中对成对目标函数（pairwise objectives）进行优化**的挑战。传统的 gossip 算法主要针对可分离的目标函数（如数据的平均值），即 $f(\theta; X_1, ..., X_n) = \frac{1}{n}\sum_{i=1}^n f(\theta; X_i)$。然而，许多重要的机器学习任务（如排序、聚类、度量学习）的目标函数是**成对形式**的，例如 U-statistic of degree two：
$$
F(\theta) = \frac{1}{n^2} \sum_{i,j=1}^n f(\theta; X_i, X_j)
$$
这类函数依赖于所有数据点对，而不仅仅是单个数据点，这使得直接应用标准的 gossip 平均算法变得不可能。

### 提出的新方法和新思路
论文在已有工作的基础上，提出了更深入、更全面的理论分析框架，并做出了以下核心贡献：

1.  **完整的非渐近估计分析 (Complete Non-asymptotic Analysis for Estimation)**：
    *   针对用于计算 U-statistic 的 **GoSTA (Gossip-based Stochastic Averaging)** 算法，首次提供了**期望和方差的完整非渐近界**。
    *   之前的分析（Colin et al., 2015）仅关注了估计量的**偏差（bias）**。本文通过定理 1 同时给出了 `expected deviation` 的上界，证明了估计误差以接近 $O(1/\sqrt{t})$ 的速度收敛，填补了理论空白。

2.  **收敛保证与偏差消失证明 (Convergence Guarantees and Vanishing Bias for Optimization)**：
    *   对于基于 **Dual Averaging** 的成对优化算法（Colin et al., 2016），之前的工作虽然给出了收敛率界，但其中包含一个由辅助数据传播引起的**梯度偏差项（gradient bias）**，且无法证明该偏差会消失，因此不能严格保证收敛。
    *   本文的关键创新在于，通过引入**遍历性分析（ergodic argument）**，明确量化了梯度偏差的衰减速率。**核心发现是：该偏差项以与图混合时间（mixing time）相关的指数速率衰减**。这最终证明了偏差项会消失，从而为该优化算法提供了首个严格的非渐近收敛保证。

3.  **新颖的下界分析 (Novel Lower Bound)**：
    *   本文将 Scaman et al. (2018) 的下界论证从可分目标扩展到了成对目标。
    *   推导出的下界揭示了网络拓扑的影响，其形式与已知的可分目标下界类似，但关键区别在于出现了一个新的量 $\Delta$，它代表了节点间**平均的两跳距离（averaged two-hop distance）**，而非简单的直径（diameter）。这深刻地说明了在成对优化中，信息需要通过中间节点进行更复杂的传播。

### 相比现有方法的优势
*   **理论完备性**：相比之前只分析偏差或给出不完整收敛界的文献，本文提供了**第一个完整的、可证明收敛的理论框架**，为 gossip 算法处理成对目标奠定了坚实的理论基础。
*   **揭示关键机制**：明确了**图的谱隙（spectral gap）** 和**混合时间（mixing time）** 是影响成对 gossip 算法效率的核心因素，解释了为什么网络连通性至关重要。
*   **指导实践**：理论结果（如偏差衰减）为实际应用提供了信心，表明即使早期估计有偏，算法最终也能可靠收敛。

---

## 2. 核心实验方法和设置

### 数据集
实验在 **Breast Cancer Wisconsin (Original) Dataset** 上进行。该数据集包含 `n = 699` 个样本，每个样本有 `d = 11` 个特征。

### 实验设置
*   **问题**：最大化 **AUC (Area Under the ROC Curve)**，这是一个典型的成对优化问题。
*   **模型**：采用线性打分函数 $x \mapsto x^\top\theta$。
*   **损失函数**：使用逻辑回归的成对损失（logistic pairwise loss）作为 AUC 的凸代理。
*   **网络拓扑**：为了研究通信约束的影响，实验设计了三种具有不同连通性的网络结构：
    1.  **完全图 (Complete Graph)**：理想情况，任意两点可直接通信，谱隙最大。
    2.  **2D网格 (2D Grid)**：连通性差，直径大，谱隙小。
    3.  **Watts-Strogatz 图**：一种小世界网络，具有介于规则格子和完全连接网络之间的特性，提供中等连通性。
*   **算法**：实现了 **Algorithm 2 (同步)** 和 **Algorithm 7 (异步)** 的 Gossip Dual Averaging for pairwise functions。
*   **评估指标**：
    1.  **目标函数值 (Objective Value / Loss Evolution)**：衡量算法收敛到最优解的速度。
    2.  **共识损失 (Consensus Loss)**：衡量各节点本地参数的差异，反映网络的一致性。
    3.  **偏差项 (Bias Term)**：直接计算并监控理论分析中的梯度偏差 $e(t)$，验证其是否快速衰减。

### 基线方法对比
本实验的重点并非与其他外部算法进行横向比较，而是**验证所提出算法自身的收敛行为和理论预测**。因此，基线可以理解为：
*   不同网络拓扑下的同一算法，用以验证**网络连通性对性能的影响**。
*   理论预测的收敛曲线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **收敛速度**：
    *   如图 1(a) 所示，**完全图 (Complete)** 和 **Watts-Strogatz 图** 的收敛速度远快于 **2D网格 (Grid)**。
    *   这直接验证了理论分析：**更好的网络连通性（更大的谱隙）能显著加速收敛**。

2.  **共识与一致性**：
    *   在连通性好的网络（完全图、Watts-Strogatz）中，各节点的本地目标值迅速趋于一致，共识损失快速下降。
    *   在网格中，共识过程缓慢，导致整体收敛延迟。

3.  **偏差项的衰减**：
    *   如图 1(b) 所示，在所有三种网络拓扑下，**梯度偏差项 $e(t)$ 都迅速收敛到零**。
    *   该偏差项在整个优化过程中始终比目标函数值小几个数量级。
    *   **这是最核心的实验证据**，它强有力地支持了理论分析中“偏差会快速消失”的结论，解释了为何该算法在实践中表现良好。

### 消融实验结果
本文未进行传统意义上的消融实验（如移除某个模块）。但其实验本身可以看作是对**网络拓扑（即通信图结构）** 这一关键变量的“消融”：
*   通过固定算法不变，仅改变网络结构，清晰地展示了**图的连通性是影响算法性能的决定性因素**，完美呼应了理论中关于谱隙和混合时间的论述。

---

## 4. 关键结论和发现

### 主要发现
1.  **理论突破**：成功建立了 gossip 算法用于成对目标的**首个完整且严谨的非渐近收敛理论**，解决了长期存在的偏差分析难题。
2.  **偏差机制**：成对 gossip 中的梯度偏差源于辅助数据在图上的非均匀采样，但该偏差会随着图的混合过程**指数级衰减**，最终不影响全局收敛。
3.  **图论核心**：**图的谱性质（特别是谱隙 $\lambda_{n-1}$）** 是决定算法效率的最关键因素。连通性越好，收敛越快。
4.  **实践有效**：尽管存在理论上的复杂性，所提出的 gossip 算法在真实数据集上表现稳健，能够可靠收敛，且偏差项在早期迭代后即可忽略。

### 方法的局限性
1.  **每节点单数据点假设**：理论分析假设每个节点只持有单个数据点。虽然在 6.1 节讨论了多点情况，但核心理论是基于此简化假设的。
2.  **凸性要求**：优化部分的收敛分析依赖于目标函数的凸性。
3.  **通信开销**：虽然 gossip 协议本身通信高效，但在大规模网络中，频繁的数据交换（即使是单个点）仍可能带来可观的通信成本。

### 未来工作方向
1.  **放宽假设**：将理论推广到每个节点持有多个数据点、非凸目标函数等更一般的情况。
2.  **结合隐私**：探索如何在 gossip 框架下结合 **Differential Privacy**，以满足更强的隐私保护需求（文中 6.2 节提及）。
3.  **提升鲁棒性与公平性**：将鲁棒学习（robustness）和公平性（fairness）等社会性约束融入该框架，实现负责任的分布式学习。
4.  **优化通信**：研究更高效的通信策略，如量化（quantization）、压缩（compression）或稀疏化，进一步降低通信负担。

</details>

---

### 7. [DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving](https://arxiv.org/abs/2603.24587)

**Authors**: Pengxuan Yang, Yupeng Zheng, Deheng Qian, Zebin Xing, Qichao Zhang, Linbo Wang, Yichen Zhang, Shaoyu Guo, Zhongpu Xia, Qiang Chen, Junyu Han, Lingyun Xu, Yifeng Pan, Dongbin Zhao  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.24587v1  

#### Abstract
We introduce DreamerAD, the first latent world model framework that enables efficient reinforcement learning for autonomous driving by compressing diffusion sampling from 100 steps to 1 - achieving 80x speedup while maintaining visual interpretability. Training RL policies on real-world driving data...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **现实世界试错成本高、安全性差**：在真实驾驶环境中进行 Reinforcement Learning (RL) 训练存在高昂的成本和不可接受的安全风险。
- **现有基于扩散模型的 World Model 推理延迟严重**：现有的 pixel-level diffusion world models 虽然支持“想象式训练”（imagination-based training），但其多步采样过程（通常需100步）导致推理延迟高达2秒/帧，无法满足 RL 所需的高频交互需求。
- **像素级目标忽视空间与动态理解**：现有方法过于关注视觉保真度，而忽略了对驾驶安全至关重要的空间结构和动态一致性建模。

### 提出的新方法与创新思路
DreamerAD 是首个将 RL 完全置于 **latent space** 中进行高效训练的自动驾驶框架，提出三大核心技术：

#### （1）**Shortcut Forcing World Model (SF-WM)**  
- 利用递归多分辨率步长压缩机制，在保持预测精度的前提下，将扩散采样从100步压缩至 **1–4步**，实现 **80× 推理加速**。
- 引入 step embedding 和 teacher-student 蒸馏策略，使模型可在任意指定步长下生成高质量预测。

#### （2）**Autoregressive Dense Reward Model (AD-RM)**  
- 在 latent 表示上直接构建自回归奖励模型，输出跨8个驾驶维度的细粒度 step-wise 奖励信号。
- 支持多时间尺度评估（0–4.0s，每0.5s一个节点），实现更精确的信用分配（credit assignment）。

#### （3）**Gaussian Vocabulary Sampling for GRPO**  
- 构建高质量轨迹词表（vocabulary of 256 representative trajectories），并基于当前策略轨迹 $T_{\text{act}}$ 构造高斯分布进行邻域采样。
- 约束探索空间于物理合理的轨迹流形内，避免 world model 的幻觉（hallucination）问题，提升探索效率与稳定性。

### 相比现有方法的优势
| 维度 | DreamerAD | 现有方法（如 Epona、WorldRFT） |
|------|-----------|-------------------------------|
| **推理速度** | 单步采样仅需 **0.03s/帧** | 多步扩散采样达 2s/帧 |
| **训练效率** | 支持高频 RL 交互 | 难以用于在线 RL 优化 |
| **奖励密度** | 密集时序奖励（dense temporal reward） | 仅依赖最终轨迹评分 |
| **探索质量** | 高斯引导的词表采样，动态连续 | 随机高斯采样易产生不连贯轨迹 |

---

## 2. 核心实验方法和设置

### 数据集
- **NavSim Dataset**：基于 nuPlan 构建的大规模自动驾驶仿真数据集。
  - 包含 8 个环视摄像头图像（surround-view RGB）和高质量 LiDAR 点云。
  - 共 **1,192 个训练场景** 和 **136 个测试场景**。
  - 排除静态与匀速场景，保留高挑战性动态交互案例。

### 实验设置
- **基础模型**：采用 Epona [34] 作为 backbone world model，其为基于 flow matching 的 autoregressive diffusion model。
- **训练流程分阶段**：
  1. **Fine-tune World Model** on NavSim（5 epochs）
  2. **Train SF-WM** with shortcut forcing（12 epochs）
  3. **Train AD-RM**（12 epochs）
  4. **RL Training with GRPO**（2 epochs）
- **硬件平台**：32 × NVIDIA H20 GPU；推理速度在单张 H20 上测量。
- **输入格式**：图像 resize 至 512×1024；VisDiT 采样步数设为1，TrajDiT 设为20。

### 评估指标
- **主指标**：
  - **EPDMS**（Extended Predictive Driver Model Score）：NavSim v2 的综合评价指标，涵盖多个维度：
    - 安全类：No Collision (NC), Drivable Area Compliance (DAC), Time-to-Collision (TTC), Traffic Light Compliance (TLC)
    - 性能类：Lane Keeping (LK), Ego Progress (EP), Comfort (Comf./HC/EC)
- **辅助指标**：
  - **PDMS**：NavSim v1 使用的传统版本。
  - **Latency/Frame**：单帧推理耗时，衡量系统实时性。

### 基线方法对比
| 方法 | 类型 | 是否使用 RL | 输入模态 |
|------|------|-------------|----------|
| TransFuser | Behavior Cloning | ❌ | Camera & LiDAR |
| Hydra-MDP | Multimodal Planning | ❌ | C&L |
| UniAD / DriveSupreme | Planning-Oriented | ❌ | C&L |
| iPad / DiffusionDrive | Diffusion-based | ❌ | C-Only |
| WorldRFT [31] | Latent RL | ✅ | C-Only |
| Epona (Base) [34] | Diffusion World Model | ❌ | C-Only |
| **DreamerAD (Ours)** | **Latent RL + SF-WM + AD-RM** | ✅ | **C-Only** |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **NavSim v2 closed-loop benchmark** 上取得：
  - **EPDMS = 87.7**，超越所有现有方法，达到 **state-of-the-art**。
  - 相比 Epona 基线提升 **+2.6 EPDMS points**。

### 与基线方法对比结果（Table 1 & 2）

#### NavSim v2 结果（Table 1）
| 指标 | DreamerAD | Epona (Base) | 提升（Δ） |
|------|-----------|--------------|---------|
| **EPDMS** | **87.7** | 85.1 | **+2.6** |
| NC ↑ | 98.0 | 97.1 | +0.9 |
| DAC ↑ | 97.2 | 95.7 | +1.5 |
| TTC ↑ | 97.4 | 96.3 | +1.1 |
| LK ↑ | 97.5 | 97.0 | +0.5 |
| HC ↑ | 98.3 | 98.0 | +0.3 |
| EC ↑ | 72.4 | 67.8 | +4.6 |
| EP ↓ | 87.8 | 88.6 | -0.8（安全优先权衡） |

> ⚠️ 注：EP 下降表明模型采取更保守策略以换取更高安全性，属合理 trade-off。

#### NavSim v1 结果（Table 2）
- **PDMS = 88.7**，同样优于 Epona（86.2），提升 **+2.5 points**。
- 在 DAC 上提升 **+2.1**，TTC 提升 **+0.5**，显示安全能力全面提升。

### 消融实验结果（Ablation Studies）

#### （1）Shortcut Forcing 效果（Table 3 & 4）
| 采样步数 | Latency/Frame (s) | EPDMS |
|--------|---------------------|-------|
| 16     | 0.40                | 87.7  |
| 4      | 0.10                | 87.8  |
| **1**  | **0.03**            | **87.7** |

✅ 结论：**单步推理即可达到最优性能**，证明 SF-WM 成功消除误差累积问题。

#### （2）Autoregressive Dense Reward Model（AD-RM）有效性
- 移除 AD-RM 后 EPDMS 从 87.7 → 87.0（↓0.7），说明密集奖励对 credit assignment 至关重要。
- 进一步实验（Table 5）表明：
  - 即使用 **20% 训练数据**，也能达到 87.5 EPDMS，接近全量数据表现。
  - 显示 AD-RM 具备强泛化能力，**低数据依赖性强**。

#### （3）Vocabulary Sampling 方法比较（ID 4 vs 5 vs 6）
| 方法 | EPDMS | 动态连续性 | 幻觉得分 |
|------|-------|------------|----------|
| **Ours (Vocab + Gaussian)** | **87.7** | ✅ 平滑轨迹 | 低 |
| WorldRFT [31] | 86.6 | ❌ 动态断裂 | 高 |
| Flow-GRPO [23] | 87.0 | ⚠️ 轻微锯齿 | 中等 |

✅ 结论：**基于词表的高斯采样显著提升轨迹质量和训练稳定性**。

---

## 4. 关键结论和发现

### 主要发现
1. **Latent Space RL 是可行且高效的路径**：DreamerAD 首次证明可在 video generation model 的 latent space 内完成端到端 RL 训练，兼顾效率与可解释性。
2. **Shortcut Forcing 实现极致加速而不牺牲性能**：通过 multi-resolution step compression，将扩散采样压缩至1步，实现 **80× 加速**，同时保持 sharp prediction quality。
3. **Dense Reward + Vocabulary Sampling 提升学习质量**：
   - 自回归奖励模型提供细粒度反馈；
   - 词表约束下的高斯采样确保探索既多样又物理合理。
4. **Imagination-based Training 显著增强安全性**：相比监督微调（SFT），RL 训练后模型学会主动减速避障、合规变道，有效规避碰撞与违规行为（见 Fig. 5）。

### 方法的局限性
- **依赖预训练的 video generation model**：性能受限于 Epona 等 backbone 的表达能力。
- **未引入语言或多模态先验**：相比 AutoVLA、RecogDrive 等 VLA 方法，缺乏高层语义指导。
- **封闭环境假设**：目前仅在 NavSim 模拟器中验证，尚未部署至实车。

### 未来工作方向
- 将 DreamerAD 扩展至 **multi-agent setting**，模拟复杂交通博弈。
- 引入 **language-instructed planning**，结合 VLM 提升意图理解能力。
- 探索 **real-to-sim adaptation**，推动 latent world model 在真实车辆上的部署。
- 开发 **lightweight latent encoder**，进一步降低计算开销，适配车载芯片。

---

> ✅ **总体评价**：DreamerAD 成功打通了“基于 latent world model 的高效 RL 训练”技术链路，为自动驾驶策略优化提供了 **可扩展、低成本、高安全性** 的新范式，具有明确的工业落地潜力。

</details>

---

### 8. [Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping](https://arxiv.org/abs/2603.23998)

**Authors**: Yao Chen, Yilong Chen, Yinqi Yang, Junyuan Shang, Zhenyu Zhang, Zefeng Zhang, Shuaiyi Nie, Shuohuan Wang, Yu Sun, Hua Wu, HaiFeng Wang, Tingwen Liu  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.23998v1  

#### Abstract
Existing approaches to increasing the effective depth of Transformers predominantly rely on parameter reuse, extending computation through recursive execution. Under this paradigm, the network structure remains static along the training timeline, and additional computational depth is uniformly assig...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的通过**参数复用**（parameter reuse）来增加 Transformer 有效深度的方法（如 block-level recurrence）存在以下问题：
- **训练时结构静态**：网络拓扑在训练过程中固定不变。
- **计算冗余严重**：对整个 Transformer block 进行递归循环，导致大量不必要的计算开销（FLOPs 增加约 16–20%）。
- **缺乏细粒度控制**：未区分不同参数组件的功能角色，导致资源分配低效。

这些问题限制了模型在保持高效的同时实现更深层次推理的能力。

---

### **提出了什么新方法或新思路**
作者提出 **Sparse Growing Transformer (SGT)**，一种基于**训练时结构稀疏性**（Training-Time Structural Sparsity）的新范式。其核心思想是：
- **动态生长**：模型深度在训练过程中**逐步自增长**，而非预设固定。
- **渐进式注意力循环**（Progressive Attention Looping）：
  - 仅对**高熵注意力头**（high-entropy attention heads）进行递归循环。
  - 循环从**深层到浅层**（deep-to-shallow）逐层激活，符合模型自然成熟轨迹。

#### **关键机制**
- **Entropy-Guided Attention Looping**：以注意力熵为信号，选择信息密集的关键头进行深度扩展。
- **Progressive Growth Training**：设计增长调度器，在训练中逐步激活新的循环层和头。

---

### **相比现有方法的优势**
| 维度 | SGT | Block Loop（传统方法） |
|------|-----|------------------------|
| **计算效率** | 额外 FLOPs 仅增加 **1–3%** | 增加 **16–20%** |
| **性能提升** | 显著优于基线，尤其在复杂推理任务上 | 提升有限，且性价比低 |
| **结构灵活性** | 动态、稀疏、细粒度控制 | 静态、稠密、粗粒度 |
| **收敛速度** | 更快收敛（单位计算下更低 PPL） | 收敛较慢 |

> ✅ **核心优势**：在极小额外成本下，实现更强的推理能力和更好的长上下文泛化。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **预训练数据**：`C4`（20B tokens 子集）
- **评估任务**（涵盖多领域）：
  - **推理与知识**：ARC-Easy, CommonsenseQA (CSQA), SocialIQA (SIQA), OpenBookQA (OBQA), WinoGrande (WG), HellaSwag (Hella.)
  - **数学推理**：MATH-500, GSM8K, AIME
  - **代码生成**：HumanEval
  - **长上下文检索/问答**：NarrativeQA, InfBenchQA, RULER_NIAH_MK/MV, TriviaQA, HotpotQA
  - **综合评测**：MMLU（STEM, Social Sciences, Humanities, Others）

> 📌 总计 **11 个多样化基准**，覆盖短序列到超长上下文（最长达 ~4000 tokens）。

---

### **实验设置**
- **模型架构**：基于 LLaMA 风格的 OLMo 架构
- **模型规模**：275M, 573M, 1.2B 参数
- **训练配置**：
  - 序列长度：4096
  - 批大小：1024
  - 优化器：AdamW（lr=6e-4）
  - 硬件：8×NVIDIA A100（40GB）
- **评估指标**：
  - 主要：**Perplexity (PPL)** 和 **Reasoning & Knowledge 平均准确率**
  - 消融分析：PPL 变化、FLOPs 开销
  - 泛化能力：长上下文外推性能（Long-context extrapolation）

---

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Vanilla** | 标准 Transformer，无任何递归机制 |
| **Block Loop** | 对选定层的整个 Transformer block 进行递归复用（block-level recurrence） |
| **SGT (Ours)** | 仅对高熵注意力头进行渐进式循环，支持 `Kmax ∈ {1,2,3}` |

> 🔍 所有方法在相同训练条件下公平比较，Block Loop 的循环层也由熵引导选出，确保可比性。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| Model | FLOPs ↑ | Avg Score ↓ |
|-------|--------|-----------|
| Vanilla (573M) | 87.41 | 35.39 |
| Block Loop | 101.76 (**+16.42%**) | 35.82 |
| **SGT (K=3)** | **88.96 (+1.77%)** | **36.41** ✅ |

> 💡 **结论**：SGT 在仅增加 **1.77% FLOPs** 的情况下，平均得分超过 Block Loop **0.59 分**，而后者需付出 **16.42%** 的计算代价。

---

### **在关键推理任务上的表现（573M 规模）**
| Task | Block Loop | SGT (K=3) | 提升 |
|------|------------|----------|------|
| **ARC-E** | 48.07 → 50.00 | **+1.93** |
| **WG** | 50.19 → 53.43 | **+3.24** |
| **CSQA** | 30.30 → 29.89 | ≈持平（但 FLOPs 更低） |

> ⚡️ 特别是在需要**多跳推理**和**语义整合**的任务上，SGT 表现尤为突出。

---

### **训练效率与收敛速度（Figure 4）**
- 在相同 FLOPs 下，SGT 的训练 PPL 显著低于 Vanilla 和 Block Loop。
- 例如，在 65×10¹⁸ FLOPs 时，SGT 比 Vanilla 降低 **0.48 PPL**。
- Block Loop 在同等预算下反而 PPL 更高 → **计算效率低下**。

---

### **消融实验结果**

#### **(1) 不同循环组件的影响（Table 2）**
| 方法 | FLOPs 增加 | PPL 下降 |
|------|-----------|---------|
| Block Loop | +5.48% | -0.413 |
| All Head Loop | +1.83% | -0.470 |
| **High-Entropy Head Loop** | **+0.38%** | **-0.521** ✅ |
| Low-Entropy Head Loop | +0.38% | -0.470 |

> ✅ **高熵头循环效果最好**，且计算开销最小；说明“选对头”比“全量循环”更重要。

#### **(2) 增长方向对比（Table 3）**
| 方法 | Avg Score |
|------|---------|
| Shallow-to-Deep (S2D) | 42.75 |
| **Deep-to-Shallow (D2S, ours)** | **43.49** ✅ |

> ✅ 符合“深层先稳定”的自然演化路径，能带来更稳定的训练和更高性能。

#### **(3) 长上下文外推能力（Table 4）**
| 设置（序列长度） | Vanilla | Block Loop | **SGT (High-Ent Loop)** |
|------------------|--------|------------|------------------------|
| 2048 (2×) | 24.21 | 24.18 | **24.16** |
| 3072 (3×) | 57.38 | 58.80 | **56.62** ✅ |
| 4096 (4×) | 116.94 | 119.48 | **111.66** ✅ |

> ✅ SGT 在长文本外推中持续领先，而 Block Loop 出现退化 → 表明粗粒度循环可能引入噪声。

---

## **4. 关键结论和发现**

### **主要发现**
1. **高熵注意力头是语义集成的关键枢纽**，不是噪声源。
2. **Transformer 层遵循“深-浅成熟轨迹”**：深层头更早分化并稳定，适合优先分配计算资源。
3. **将递归集中在高熵头上可加速收敛**（理论证明见 Appendix E），因这些头具有更强的 token-mixing 能力。
4. **SGT 实现了训练时结构稀疏性**：仅用极少参数子集的增长，即可获得显著性能增益。
5. **内省思考机制可视化**（Figure 6 & 11）显示：
   - 初始循环关注语法元素；
   - 后续循环逐步聚焦于任务相关和答案相关的 token；
   - 实现了从“广撒网”到“精准锚定”的语义聚焦过程。

---

### **方法的局限性**
- **实验规模受限**：最大只验证到 **1.2B 参数**，尚未扩展至更大规模 LLM。
- **超参数探索不足**：未系统搜索最优的 `h`, `L`, `Kmax` 等配置，可能存在进一步优化空间。
- **推理吞吐略降**：虽然优于 Block Loop，但仍比 Vanilla Transformer 稍慢（见 Table 10）。

---

### **未来工作方向**
- 将 SGT 扩展到 **十亿级以上大模型**，验证其可扩展性。
- 探索 **动态调整增长策略**（如基于 loss 或 entropy 变化自动触发）。
- 结合 **input-dependent sparsity**（如 MoE、token routing）形成混合稀疏架构。
- 探索 SGT 在 **fine-tuning** 和 **指令微调** 中的表现。

---

## ✅ 总结一句话
> **SGT 提出了一种“训练时自生长”的稀疏深度分配机制，通过渐进式地对高熵注意力头进行循环，实现了以极小计算代价（+1–3% FLOPs）换取显著推理性能提升的新范式，兼具高效、有效与可解释性。**

</details>

---

### 9. [MetaKube: An Experience-Aware LLM Framework for Kubernetes Failure Diagnosis](https://arxiv.org/abs/2603.23580)

**Authors**: Wei Sun, Ting Wang, Xinran Tian, Wanshun Lan, Xuhan Feng, Haoyue Li, Fangxin Wang  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.23580v1  

#### Abstract
Existing LLM-based Kubernetes diagnostic systems cannot learn from operational experience, operating on static knowledge bases without improving from past resolutions. We present MetaKube, an experience-aware LLM framework through three synergistic innovations: (1) an Episodic Pattern Memory Network...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《MetaKube: An Experience-Aware LLM Framework for Kubernetes Failure Diagnosis》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Models (LLMs)** 的 Kubernetes 故障诊断系统存在三大根本缺陷：
1. **无法从运维经验中学习**：现有系统依赖静态知识库（如 RAG），不支持通过历史故障解决过程进行持续学习。
2. **高质量诊断数据稀缺**：Kubernetes 生态中缺乏结构化、标注良好的故障处理数据，限制了模型训练与检索效果。
3. **企业级数据隐私要求高**：生产环境中敏感日志和配置不能发送至外部 LLM API，而本地部署的大模型（>70B）计算开销过大，小模型（<10B）又能力不足。

---

### 提出的新方法与创新思路
作者提出 **MetaKube** —— 一种具备“经验感知”能力的 LLM 框架，通过三个协同组件实现认知型故障诊断：

#### （1）Episodic Pattern Memory Network (EPMN)
- **功能**：抽象并存储历史故障解决中的模式，形成可复用的经验记忆。
- **机制**：
  - 双粒度记忆架构：保留具体事件（episodic memories）和泛化模式（pattern abstractions）。
  - 置信度校准检索：结合语义相似性、时间新鲜度、历史成功率等多因素动态选择最优匹配。
  - 支持快速模式匹配（intuitive pathway）和引导因果探索（analytical pathway）。

#### （2）Meta-Cognitive Controller
- **功能**：模仿人类专家的认知控制机制，动态决定使用直觉路径还是分析路径。
- **决策依据**：基于 EPMN 返回的记忆置信度（confidence score），若高于阈值 $ t $ 则走轻量直觉路径；否则触发深度图推理。
- **优势**：在速度与准确性之间自适应权衡，提升资源利用效率。

#### （3）KubeLLM：领域专用的小参数 LLM
- **基础模型**：基于开源的 **Qwen3-8B** 构建。
- **增强方式**：采用 **Supervised Fine-Tuning (SFT)** + **Low-Rank Adaptation (LoRA)** 在自建的 **Kubernetes Fault Resolution Dataset (KFRD)** 上微调。
- **目标**：使 8B 小模型达到接近 GPT-4 的诊断能力，同时支持完全本地化部署，保障数据隐私。

---

### 相比现有方法的优势
| 维度 | MetaKube | 传统 RAG / 零样本 LLM |
|------|---------|------------------------|
| 学习能力 | ✅ 持续从操作经验中学习（experience-aware） | ❌ 静态知识，无反馈闭环 |
| 推理灵活性 | ✅ 动态切换直觉 vs 分析路径 | ❌ 固定流程或单一模式 |
| 数据隐私 | ✅ 完全本地部署（on-premise） | ❌ 依赖外部 API 存在泄露风险 |
| 性能-成本平衡 | ✅ 8B 模型 + 领域优化 ≈ GPT-4 表现 | ❌ 大模型昂贵，小模型不准 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### （1）**KubeFault**（主测试集）
- 规模：1,873 个真实世界 Kubernetes 故障场景。
- 来源：由 GPT-5 从 KubeGraph 自动生成，并经电信公司资深运维工程师审核修正。
- 内容涵盖：
  - 自然语言症状描述
  - 集群上下文环境
  - 日志片段（kubectl 输出）
  - 根因标注与因果链解释
  - 可执行修复命令
- 覆盖六大错误类别：
  - Resource, Network, Scheduling, Image, Configuration, System Errors

#### （2）**Kubernetes Fault Resolution Dataset (KFRD)**
- 规模：7,000 个高质量样本（5,000 用于 SFT，2,000 测试）
- 构建流程四阶段：
  1. **问题-解决方案采集**：来自 Stack Overflow、GitHub Issues、技术博客
  2. **问题-尝试-解决方案重构**：显式记录用户失败尝试，增强推理多样性
  3. **LLM 数据增强**：使用 Grok-4 生成合成样本
  4. **Chain-of-Thought 增强**：用 GPT-5 添加详细推理路径

---

### 实验设置与评估指标

#### 评估维度（每项满分 100 分）
| 指标 | 含义 |
|------|------|
| **Effectiveness (Eff.)** | 是否准确识别根因并提供有效解决方案 |
| **Equivalence (Equ.)** | 与参考方案的方法论一致性 |
| **Completeness (Com.)** | 步骤完整性、边缘情况覆盖、命令正确性 |
| **Safety/Accuracy (S/A)** | 是否符合 Kubernetes 最佳实践，避免危险操作 |
| **Average (Avg.)** | 四项平均得分 |

#### 评估方式双轨制：
1. **GPT-5 自动评分**：规模化自动打分
2. **三位专家盲评**：来自不同电信公司的资深运维人员独立评分，确保权威性

#### 基线方法对比
| 类别 | 方法列表 |
|------|----------|
| **零样本 LLM** | GPT-4.1, GPT-4.1-mini, Qwen3-8B |
| **GraphRAG 增强版** | 上述三种模型 + GraphRAG 检索增强 |
| **本工作** | **MetaKube (Ours)**：Qwen3-8B + EPMN + KubeGraph + Meta-Controller + SFT |

所有实验均在本地 GPU 集群完成（A100 ×8 / A6000 ×8），保证公平性和隐私一致性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（GPT-5 评估，100 分制）

| 方法 | Eff. | Equ. | Com. | S/A | **Avg.** |
|------|------|------|------|-----|----------|
| Qwen3-8B (Zero-shot) | 48.7 | 51.2 | 46.1 | 57.4 | **50.9** |
| Qwen3-8B (GraphRAG) | 66.7 | 69.1 | 64.8 | 73.3 | **68.5** |
| GPT-4.1 (GraphRAG) | 89.3 | 92.6 | 91.4 | 94.1 | **91.9** |
| **MetaKube (Ours)** | **91.2** | **90.8** | **87.3** | **92.5** | **90.5** |

> 🔍 **结论**：
- MetaKube 将 Qwen3-8B 的平均表现从 **50.9 → 90.5**，提升 **+39.6 分**（+77.8%）
- 性能逼近最强基线 GPT-4.1 + GraphRAG（仅差 **1.4 分**），远超其他小模型方案
- 在 **Effectiveness 和 Safety/Accuracy** 上甚至略优于 GPT-4.1

---

### 人类专家评估结果（更贴近实际运维需求）

| 方法 | Eff. | Equ. | Com. | S/A | **Avg.** |
|------|------|------|------|-----|----------|
| Qwen3-8B (Zero-shot) | 31.5 | 35.8 | 28.9 | 42.3 | **34.6** |
| GPT-4.1 (GraphRAG) | 73.8 | 77.8 | 71.2 | 79.4 | **75.6** |
| **MetaKube (Ours)** | **75.6** | **74.2** | **69.8** | **81.2** | **75.2** |

> ✅ **亮点**：
- MetaKube 在 **Safety/Accuracy 达到 81.2**，为所有方法最高，说明其建议最安全可靠
- 在有效性（Eff.）上与 GPT-4.1 GraphRAG 并列第一，证明其在真实运维中具有高度实用性

---

### 消融实验结果（Ablation Studies）

#### （1）EPMN 模块消融（Figure 3）
| 指标 | 有 EPMN | 无 EPMN | 提升幅度 |
|------|--------|--------|---------|
| Avg. Score | 90.5 | 78.5 | **+15.3%** |
| Safety/Accuracy | 92.5 | 75.8 | **+16.6%** |

> 💡 **发现**：EPMN 对安全性和整体性能贡献显著，尤其擅长捕捉常见错误模式，防止重复犯错。

#### （2）KubeLLM SFT 微调效果（Figure 4）
- 在 KFRD 测试集上，SFT 后模型性能提升 **+45.5%**
- 8B 模型经领域微调后，可媲美主流 API 模型在 Kubernetes 场景下的表现

#### （3）KubeGraph 消融（Table 2）
| 数据集 | w/ KubeGraph | w/o KubeGraph | 提升 |
|-------|-------------|--------------|------|
| KubeFault (in-domain) | 75.2 | 34.6 | **+117.3%** |
| Telecom Dataset (out-of-domain) | 57.6 | 22.4 | **+157.1%** |

> 🚀 **关键洞察**：KubeGraph 不仅提升领域内表现，还展现出极强的**跨域泛化能力**，表明其捕获的是结构性问题解决模式而非表面特征。

---

## 4. 关键结论和发现

### 主要发现
1. **经验积累对运维 AI 极其重要**：  
   单纯依赖静态知识（如 RAG）无法满足复杂系统的长期演进需求。MetaKube 通过 EPMN 实现“越用越聪明”，验证了**经验驱动学习**的价值。

2. **小模型 + 领域优化 = 大模型级表现**：  
   通过高质量数据集（KFRD）+ 领域微调（SFT+LoRA），Qwen3-8B 能逼近 GPT-4.1 性能，打破“必须用大模型”的迷思。

3. **双路径认知架构更高效智能**：  
   Meta-Cognitive Controller 成功模拟专家思维，在简单问题上快速响应，在复杂问题上深入分析，实现**资源与精度的最优平衡**。

4. **结构化知识图谱是泛化的基石**：  
   KubeGraph 提供了超越文本检索的因果推理能力，尤其在未知故障场景下表现出色，是系统鲁棒性的关键支撑。

---

### 方法的局限性
1. **初始冷启动问题**：EPMN 需要一定数量的历史故障记录才能发挥价值，在新集群初期可能表现受限。
2. **KubeGraph 构建依赖高质量语料**：虽然使用 LLM 自动生成，但仍需人工审核以确保准确性。
3. **实时性挑战**：当前框架尚未集成流式监控信号，仍以“事后诊断”为主，未来可向实时告警联动扩展。

---

### 未来工作方向
1. **引入在线持续学习机制**：让系统在每次成功诊断后自动更新 EPMN 和 KubeGraph，形成闭环进化。
2. **支持多模态输入**：融合 Prometheus 指标、Fluentd 日志、分布式追踪（Jaeger）等多源信号作为诊断输入。
3. **构建开放社区生态**：推动 KFRD、KubeGraph 开源共享，促进整个行业在 Kubernetes AIOps 方向的发展。
4. **探索自动化修复（Auto-Remediation）**：在诊断基础上进一步生成可执行脚本并安全回滚，迈向全自动运维。

---

> ✅ **总结一句话**：  
> **MetaKube 是首个将“经验学习”融入 LLM 的 Kubernetes 诊断框架，用一个 8B 小模型实现了接近 GPT-4 的专业水平，且全程本地运行，兼具高性能、高安全、可持续进化三大优势。**

> 🔗 项目代码已开源：[https://github.com/MetaKube-LLM-for-Kubernetes-Diagnosis/MetaKube](https://github.com/MetaKube-LLM-for-Kubernetes-Diagnosis/MetaKube)

</details>

---

### 10. [Lightweight Fairness for LLM-Based Recommendations via Kernelized Projection and Gated Adapters](https://arxiv.org/abs/2603.23780)

**Authors**: Nan Cui, Wendy Hui Wang, Yue Ning  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.23780v1  

#### Abstract
Large Language Models (LLMs) have introduced new capabilities to recommender systems, enabling dynamic, context-aware, and conversational recommendations. However, LLM-based recommender systems inherit and may amplify social biases embedded in their pre-training data, especially when demographic cue...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Lightweight Fairness for LLM-Based Recommendations via Kernelized Projection and Gated Adapters*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Models (LLMs)** 的推荐系统（RecLLMs）虽然具备强大的语义理解和生成能力，但其预训练数据中嵌入的社会偏见（如性别、年龄、职业等敏感属性）可能在推荐过程中被继承甚至放大。现有公平性方法存在以下问题：
- **需要额外可训练参数**（如 UP5 使用对抗训练模块），增加计算开销；
- **优化不稳定**，对超参数敏感；
- 多属性去偏时容易过度去除有用特征，导致推荐性能下降。

### 🚀 提出的新方法
本文提出一种**轻量级、无需额外训练**的去偏框架，结合两种核心技术：
1. **Kernelized Iterative Null-space Projection (RFF-INLP)**  
   - 将原始 INLP 方法扩展至非线性空间：通过 **Random Fourier Features (RFF)** 显式映射表示到高维核空间，在其中执行闭式（closed-form）正交投影，消除敏感属性的线性和非线性泄露。
   - 引入 **各向同性高斯噪声扰动** 提升投影鲁棒性，防止微调带来的表示漂移。
   - 投影矩阵作为非可训练缓冲区存储，**零梯度、零优化成本**。

2. **Two-level Gated Mixture-of-Experts (MoE) Adapter**
   - 第一级门控（outer gate）：根据上下文动态加权多个属性对应的投影器强度，实现**自适应多属性去偏**；
   - 第二级门控（inner gate）：引入低秩 LoRA 专家网络，选择性恢复因去偏而损失的任务相关信号，形成“**erase-then-repair**”机制；
   - 整体仅引入 $ O(k) $ 额外参数（$k$ 为敏感属性数），保持轻量化。

### 🔍 相比现有方法的优势
| 维度 | 本方法 | 现有方法（如 UP5） |
|------|--------|------------------|
| 可训练参数 | 极少（仅 MoE 中的小型适配器） | 大量（需训练对抗判别器） |
| 优化稳定性 | 高（无对抗训练） | 低（依赖梯度反转，易震荡） |
| 多属性处理能力 | 支持灵活融合与控制 | 多次独立处理易造成信息丢失 |
| 推理延迟 | 几乎无增加 | 显著增加 |

---

## 2. 核心实验方法和设置

### 📚 数据集
使用两个真实世界数据集进行验证：

| 数据集 | 类型 | 敏感属性 | 任务形式 |
|-------|------|----------|---------|
| **MovieLens-1M** | 电影评分数据 | Gender (2类), Age (7类), Occupation (21类) | Sequential & Direct Recommendation |
| **Insurance Dataset**（非洲保险公司客户数据） | 保险产品交互 | Marital Status (8类), Age (5类), Occupation (6类) | Direct Recommendation（缺乏时间戳） |

### ⚙️ 实验设置
- **LLM Backbone**: 冻结的 `Instruct Llama-3.2-1B` 模型；
- **Fine-tuning 方式**: 仅训练 LoRA 适配器（rank=32），主干不更新；
- **去偏组件**:
  - RFF-INLP: $D=4096$ 维随机傅里叶特征，噪声尺度 $\eta=0.05$；
  - MoE Adapter: 属性特定专家 rank=8；
- **公平性阈值**: 当 Counterfactual Leakage Gap (AcL) > $T=0.01$ 时触发投影更新。

### 📊 评估指标
| 指标类型 | 指标名称 | 描述 |
|--------|--------|------|
| **Utility（效用）** | Hit@{1,3,10} | 推荐列表中是否包含真实目标项（Top-K 准确率） |
| **Fairness（公平性）** | AcL (Counterfactual Leakage Gap) ↓ | 序列表示预测敏感属性的能力偏离随机猜测的程度（越接近 0 越公平） |

### 🆚 基线方法
1. **LLaRA** [24]: 基础 RecLLM 框架，无去偏；
2. **P5** [13]: Prompt-based 推荐范式；
3. **UP5** [21]: 当前最先进的公平性增强版本，采用对抗训练。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 2 & 3）

#### ✅ MovieLens - Sequential Recommendation
| Attribute(s) | Method | Hit@1 ↑ | Hit@3 ↑ | Hit@10 ↑ | AcL ↓ |
|-------------|--------|--------|--------|--------|--------|
| G/A/O (All) | **Ours** | **56.08** | **72.28** | **81.67** | **0.17** |
|             | UP5     | 52.30  | 68.50  | 79.10  | 3.21   |

> ➤ **全面领先**：在所有三个指标上均优于 UP5，且 AcL 下降达 **94.7%**。

#### ✅ MovieLens - Direct Recommendation
| Attribute(s) | Method | Hit@1 ↑ | Hit@3 ↑ | Hit@10 ↑ | AcL ↓ |
|-------------|--------|--------|--------|--------|--------|
| G/A/O (All) | **Ours** | **43.20** | **49.42** | **66.06** | **0.78** |
|             | UP5     | 20.18  | 38.79  | 66.78  | 3.21   |

> ➤ **Hit@1 提升超 2 倍**，同时将平均 AcL 从 3.21% 降至 0.78%。

#### ✅ Insurance Dataset - Direct Recommendation
| Attribute(s) | Method | Hit@1 ↑ | Hit@3 ↑ | Hit@10 ↑ | AcL ↓ |
|-------------|--------|--------|--------|--------|--------|
| M/A/O (All) | **Ours** | 57.37 | **91.47** | **99.35** | **0.13** |
|            | UP5     | **81.63** | 91.52 | 97.37 | 0.74 |

> ➤ 尽管在 Hit@1 上略逊于 UP5，但在 **公平性上提升 5.7 倍（0.74 → 0.13）**，且 Hit@3 和 Hit@10 更优，体现更优的 **accuracy-fairness trade-off**。

### 🔬 消融实验分析（文中隐含）
- **RFF vs. Linear INLP**：未显式列出，但从 AcL 接近 0 可推断 RFF 成功捕获并消除了非线性偏见；
- **两级门控设计必要性**：
  - 若无 Level-1 自适应加权 → 多属性间冲突加剧；
  - 若无 Level-2 LoRA 修复 → 推荐准确率显著下降；
- **更新频率**：平均每个 epoch 更新不超过 3 次投影矩阵，说明收敛快、效率高。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Kernelized INLP 可有效去除非线性偏见泄露**  
   利用 RFF 显式升维后执行闭式投影，能够在不触碰模型参数的情况下彻底清除敏感属性信息，实现近乎理想的 counterfactual fairness（AcL ≈ 0）。

2. **Gated MoE Adapter 实现精准“擦除-修复”平衡**  
   通过双层门控机制，在去偏的同时智能恢复任务有用信号，避免传统方法中“去偏即降准”的困境。

3. **轻量高效，适合部署**  
   不引入额外损失函数或复杂优化流程，仅增加少量参数，推理无延迟，适用于大规模 LLM 推荐系统部署。

4. **优于主流对抗式方法（如 UP5）**  
   在多个数据集和任务中，不仅显著提升公平性（AcL ↓），还大幅提升推荐准确性（Hit@k ↑），打破 utility-fairness trade-off。

### ⚠️ 方法的局限性
- **依赖用户交互历史**：序列表示的质量直接影响去偏效果；对于冷启动或交互稀疏用户表现可能受限；
- **仅处理可观测敏感属性**：无法应对隐式或组合型偏见（如交叉歧视）；
- **假设敏感属性标签可用**：实际场景中这些标签往往缺失或需推断。

### 🔮 未来工作方向
- 探索**不依赖用户历史**的去偏方法；
- 扩展至**隐式敏感属性检测与去偏联合建模**；
- 研究**跨语言、跨文化背景下的公平性迁移**；
- 结合 causal reasoning 进一步提升 counterfactual fairness 的理论保障。

---

> 💡 **一句话总结**：  
> 本文提出了一种**无需训练、闭式求解、支持多属性动态调节**的轻量级去偏框架——**Kernelized INLP + Gated MoE Adapter**，在保持甚至提升推荐准确性的前提下，实现了接近完美的 counterfactual fairness，为 LLM-based 推荐系统的公平性工程提供了实用且高效的解决方案。

</details>

---

### 11. [SCoOP: Semantic Consistent Opinion Pooling for Uncertainty Quantification in Multiple Vision-Language Model Systems](https://arxiv.org/abs/2603.23853)

**Authors**: Chung-En Johnny Yu, Brian Jalaian, Nathaniel D. Bastian  
**Category**: cs.AI  
**Published**: 2026-03-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.23853v1  

#### Abstract
Combining multiple Vision-Language Models (VLMs) can enhance multimodal reasoning and robustness, but aggregating heterogeneous models' outputs amplifies uncertainty and increases the risk of hallucinations. We propose SCoOP (Semantic-Consistent Opinion Pooling), a training-free uncertainty quantifi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SCoOP: Semantic Consistent Opinion Pooling for Uncertainty Quantification in Multiple Vision-Language Model Systems

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前的 **Vision-Language Models (VLMs)** 在多模态推理中表现出色，但在安全关键场景（如医疗诊断、多机器人协作）中仍面临严重的 **hallucination**（幻觉）问题——即模型输出与输入图像无关的信息。  
当多个异构 VLMs 被组合成 **multi-VLM system** 时，传统的聚合方式（如多数投票）可能放大个体模型的错误和幻觉，导致系统级可靠性下降。

更重要的是，现有的 **Uncertainty Quantification (UQ)** 方法大多针对单个模型设计，无法有效衡量整个 multi-VLM 系统的集体不确定性，从而难以实现可靠的幻觉检测与高不确定样本的主动拒绝（abstention）。

---

### 🚀 提出了什么新方法或新思路
本文提出 **SCoOP**（**Semantic-Consistent Opinion Pooling**），一种**无需训练**的 UQ 框架，用于在 multi-VLM 系统中进行系统级不确定性量化。

#### 核心思想：
- 将每个 VLM 视为一个提供概率意见的“专家”。
- 通过 **uncertainty-weighted linear opinion pooling** 聚合多个 VLM 的输出：
  - 首先对每个 VLM 进行多次采样，生成响应，并映射到统一选项空间 $ \mathcal{O} $，形成概率分布 $ p_k $。
  - 利用 **Shannon Entropy** 衡量每个模型的不确定性 $ H_k $，并将其转化为置信度权重 $ w_k = \frac{1/H_k}{\sum_{k'} 1/H_{k'}} $。
  - 使用加权平均得到系统级聚合分布：  
    $$
    p_{\text{agg}} = \sum_{k=1}^K w_k p_k
    $$
  - 最终预测 $ \theta^* = \arg\max_j p_{\text{agg}}(\theta_j) $
  - 系统不确定性由 $ H_{\text{agg}} = -\sum_j p_{\text{agg}}(\theta_j)\log p_{\text{agg}}(\theta_j) $ 给出。

---

### 🔍 相比现有方法的优势
| 方面 | SCoOP | 现有方法 |
|------|-------|--------|
| **是否需要训练** | ❌ 不需要（training-free） | 多数需要额外训练或标注数据（如贝叶斯网络、conformal prediction） |
| **是否支持系统级 UQ** | ✅ 显式建模 multi-VLM 系统整体不确定性 | ❌ 多数仅适用于单模型或未评估系统级可靠性 |
| **聚合机制** | ✅ 基于熵的不确定性加权，自动抑制低置信模型影响 | ⚠️ 多数采用等权投票或选择最低不确定性模型（heuristic） |
| **效率** | ✅ 微秒级聚合开销（μs），远低于 VLM 推理时间（秒级） | ⚠️ 训练型方法计算成本高，不适用于实时部署 |

> 💡 **创新亮点总结**：
> - 首个专为 multi-VLM 系统设计的系统级 UQ 框架；
> - 提出“语义一致的意见池化”机制，在保留多样性的同时提升鲁棒性；
> - 实现高效、免训练、可扩展的不确定性感知聚合。

---

## 2. 核心实验方法和设置

### 📚 使用了哪些数据集
在三个主流多模态基准上进行评估：

| 数据集 | 描述 |
|-------|------|
| **ScienceQA** | 科学领域的图文推理任务，涵盖物理、生物、化学等，共 952 个样本 |
| **MMMU** | 大规模跨学科理解与推理基准，测试大学级别知识，子集含 825 个问题 |
| **MMBench** | 综合性真实世界视觉理解评测，关注感知与常识推理，使用 950 个多选题 |

---

### ⚙️ 实验设置和评估指标

#### 模型配置
- 使用 **16 个开源 VLMs**，覆盖五类架构：LLaVA-v1.6, Gemma-3, InternVL3, DeepSeek-VL2, Qwen2.5-VL
- 按参数规模分为四档：Small (2–4B), Medium (12–16B), Large (27–38B), Extra-Large (72–78B)
- 构建所有 $ K \in \{2,3,4,5\} $ 的跨模型组合，共 **52 种 multi-VLM 系统**

#### 设置细节
- 温度 $ T=1.0 $，核采样 $ p=0.9 $，top-K=50
- 每个模型采样 $ N=10 $ 次以估计概率分布
- 所有实验在 NVIDIA B200 GPU 上运行

---

### 📊 评估指标

| 指标 | 全称 | 含义 |
|------|------|------|
| **AUROC** | Area Under the ROC Curve | 幻觉检测能力：越高越好（1.0 为完美区分正确 vs 错误回答） |
| **AURAC** | Area Under the Rejection Accuracy Curve | 主动拒绝能力：随着高不确定性样本被剔除，准确率上升越快越好 |
| **E2E-Latency@p50** | End-to-End Latency (median) | 系统端到端延迟，衡量效率 |

---

### 🔁 基线方法对比
由于缺乏专门针对 multi-VLM 系统的 UQ 方法，作者构建两个启发式 baseline：

| 基线 | 方法描述 |
|------|---------|
| **Naive Selection** | 选择不确定性最低（熵最小）的那个 VLM 的输出作为最终结果 |
| **Majority Voting** | 每个 VLM 投票其最可能的答案，统计得票数决定最终答案；系统不确定性为投票分布的熵 |

两者均基于 **Semantic Entropy [18]** 估计单个模型不确定性。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（ScienceQA 上 3-VLM 系统，Extra-Large 参数）

| 方法 | AUROC (幻觉检测) | AURAC (拒绝增益) |
|------|------------------|------------------|
| **SCoOP (ours)** | **0.866** | **0.907** |
| Majority Voting | 0.757 (+10.9% ↓) | 0.840 (+6.7% ↓) |
| Naive Selection | 0.732 (+13.4% ↓) | 0.818 (+8.9% ↓) |

> ✅ SCoOP 在两项指标上显著领先，分别超越最佳基线约 **10–13%** 和 **7–9%**

---

### 🔁 与基线方法的对比结果

#### ✔️ 幻觉检测（AUROC）
- SCoOP 比两个基线高出 **10–13%**
- 更重要的是，相比未聚合的单模型平均 AUROC（0.789），SCoOP **提升了 7.7%**，而两个基线反而下降了 3–6%
- 表明传统聚合会劣化不确定性信号，而 SCoOP 成功融合互补信息

#### ✔️ 主动拒绝（AURAC）
- SCoOP 达到 **0.907**，优于 Majority Voting（+6.7%）和 Naive Selection（+8.9%）
- 单模型平均 AURAC 为 0.853，SCoOP **提升 5.4%**，说明其不确定性排序更可靠

#### ⏱️ 效率表现（E2E-Latency@p50）
| 方法 | 相对于 SCoOP 的延迟差值（μs） |
|------|-------------------------------|
| Majority Voting | 快 1.87 μs |
| Naive Selection | 快 7.88 μs |

> ❗ 差异仅为微秒级，**完全可忽略**（典型 VLM 推理耗时为秒级）。因此，SCoOP 在几乎无性能代价下实现了大幅 UQ 提升。

---

### 🔍 消融实验与研究问题分析（RQs）

#### RQ1: 模型参数规模的影响？
- 在 Small → Extra-Large 四种尺度下，SCoOP 始终最优
- 相比 Majority Voting 提升 **8.9–10.9% AUROC**
- 相比 Naive Selection 提升 **13.4–15.5% AUROC**
- 表明 SCoOP 可良好扩展至不同规模模型

#### RQ2: 系统大小（#VLMs）的影响？
- 在 2–5 个 VLM 的系统中，SCoOP 性能稳定：
  - 平均 AUROC = **0.824**（vs MV: 0.732, NS: 0.694）
  - 平均 AURAC = **0.876**（vs MV: 0.823, NS: 0.774）
- 性能波动极小（AUROC ≤ ±0.6%，AURAC ≤ ±1.3%），显示强鲁棒性

#### RQ3: 是否牺牲准确性换取 UQ 性能？
- **否！** SCoOP 在提升 UQ 的同时，**保持甚至略微提高任务准确率**
- 如在 Extra-Large 模型上，SCoOP 准确率达 **71.85%**，高于单模型平均（69.75%）和 Majority Voting（71.11%），仅次于 Naive Selection（72.90%）

#### RQ4: 在低质量模型下的鲁棒性？
- 当所有个体 VLM 在 MMBench/MMMU 上准确率 <50% 时：
  - Majority Voting 和 Naive Selection 的 AUROC 接近 0.5（随机水平）
  - SCoOP 仍能达到 **0.608 (MMBench)** 和 **0.667 (MMMU)**
- 表明 SCoOP 能从不可靠模型中提取有效的置信信号

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **系统级 UQ 至关重要**：multi-VLM 系统不能简单套用单模型 UQ 方法，必须考虑模型间共识与分歧。
2. **SCoOP 显著优于启发式聚合**：通过不确定性加权意见池化，SCoOP 实现了更强的幻觉检测与拒绝能力。
3. **高效且实用**：聚合过程仅引入微秒级开销，适合实际部署。
4. **鲁棒性强**：在不同模型规模、系统大小、甚至低性能模型组合下均表现稳定优越。
5. **不牺牲精度**：在增强可靠性的同时，准确率不低于甚至略优于基线。

---

### ⚠️ 局限性
1. **依赖采样次数**：总延迟受 $ K \times N $ 影响（每个模型采样 $ N $ 次），虽聚合快，但整体推理仍较慢。
2. **局限于多选题（MCQ）**：目前仅适用于结构化输出任务，尚未扩展至自由形式 VQA。
3. **假设统一选项空间**：需预先定义候选答案集合，限制了开放域应用。

---

### 🔮 未来工作方向
1. **扩展至 free-form VQA**：将 SCoOP 应用于开放式问答任务，探索语义聚类 + 不确定性融合的新机制。
2. **集成至 agent pipeline**：结合 agentic routing 或 multi-step reasoning，利用系统不确定性指导动态决策路径。
3. **轻量化采样策略**：研究如何减少采样次数 $ N $ 而不影响 UQ 质量，进一步降低端到端延迟。
4. **跨模态校准**：探索视觉与语言模态间的不确定性对齐机制，提升多模态一致性。

---

## ✅ 总结一句话
> **SCoOP 是首个面向 multi-VLM 系统的免训练、高效、系统级 UQ 框架，通过语义一致的不确定性加权意见池化，在幻觉检测与主动拒绝任务上显著超越现有方法，且几乎无额外延迟，极大提升了多模态 AI 系统的可靠性与实用性。**

</details>

---

### 12. [Fast and Faithful: Real-Time Verification for Long-Document Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2603.23508)

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Haichen Zhang, Huamin Chen  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.23508v1  

#### Abstract
Retrieval-augmented generation (RAG) is increasingly deployed in enterprise search and document-centric assistants, where responses must be grounded in long and complex source materials. In practice, verifying that generated answers faithfully reflect retrieved documents is difficult: large language...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast and Faithful: Real-Time Verification for Long-Document Retrieval-Augmented Generation Systems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Retrieval-Augmented Generation (RAG)** 系统中，生成的回答需要基于检索到的长文档进行**忠实性验证**（faithfulness verification），以检测是否存在 **hallucination**（幻觉）。然而，现实中的企业级文档（如法律合同、临床报告）通常长达数万 tokens，远超轻量级验证模型（如 encoder-based 分类器）的上下文窗口（通常为 8K tokens）。这导致现有系统只能对文档的前缀部分进行“截断式验证”（truncated validation），从而可能遗漏深层的关键证据。

同时，虽然大语言模型（LLM-as-judge）可以处理长上下文，但其推理延迟高、成本昂贵，难以满足实时服务需求。因此，存在一个**速度 vs. 上下文完整性**的根本矛盾。

---

### 提出的新方法与创新思路

本文提出了一种**可在生产环境中部署的实时、全文档验证系统**，核心创新如下：

#### (1) **Retrieval-Aware RoPE 扩展方法**
- 将基于 encoder 的模型（如 ModernBERT）的上下文窗口从 8K 扩展至 **32K tokens**。
- 针对直接应用 RoPE scaling 到 encoder 会导致 long-range attention 能力退化的问题，提出 **retrieval-aware masking** 策略：
  - **Long-Range Copy Masking**：强制模型关注远距离重复出现的 token。
  - **Anchor-Reference Masking**：锚定早期位置，要求模型回溯预测后续相同内容。
- 结合 **Elastic Weight Consolidation (EWC)** 正则化 和保守微调策略，防止 fine-tuning 过程中破坏预训练的 long-range 注意力路径。

#### (2) **长上下文幻觉检测器训练**
- 构建了一个支持 32K 上下文的 **token-level hallucination detector**，仅对生成 response 中的每个 token 进行二分类（supported vs. hallucinated）。
- 强调训练数据分布需匹配真实 RAG 输出风格，避免使用高幻觉率的合成数据导致模型过于悲观。

#### (3) **可配置的 Early-Exit 推理框架**
- 在 Transformer 的中间层添加 lightweight adapter，允许模型根据延迟预算提前退出（early exit），实现**显式的精度-延迟权衡**。
- 支持动态调整计算量，在短文档上保持高速，在长文档上仍能实现实时响应。

#### (4) **构建长文档幻觉基准 Long-Context Benchmark**
- 现有 benchmark（如 RAGTruth）文档过短，无法评估 full-document verification。
- 新建测试集基于 **NarrativeQA、QuALITY、GovReport** 等自然长文本，文档长度介于 8K–24K tokens，并注入可控幻觉，用于真实场景下的评估。

---

### 相比现有方法的优势

| 维度 | 现有方法 | 本文方法 |
|------|--------|---------|
| 上下文覆盖 | 截断至 8K，丢失深层证据 | 支持 **32K 全文档访问** |
| 推理速度 | LLM-as-judge 太慢；encoder 快但不完整 | encoder + early exit，兼顾**速度与完整性** |
| 验证粒度 | 多为示例级判断 | **token-level 检测 + span-level 高亮** |
| 实用性 | 学术导向，难部署 | 面向生产，支持**动态配置与吞吐优化** |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### （1）标准幻觉 benchmark（短文档）
- **RAGTruth**：人工标注的 RAG 回答幻觉 span，最长文档约 2.6K tokens，用于验证扩展后是否损害原有能力。

#### （2）新建长文档 benchmark（核心贡献之一）
| 数据集 | 领域 | 文档长度范围 | 数量 |
|-------|-----|-------------|------|
| NarrativeQA | 故事 | 8K–24K tokens | 489 |
| QuALITY | 文章 | 8K–20K tokens | 223 |
| GovReport | 政府文件 | 8K–20K tokens | 88 |
- 总计：**337 测试样本**，平均长度 ~17.5K tokens。
- 构造方式：
  1. 使用 LLM 生成回答；
  2. 通过受控提示注入 50% 幻觉；
  3. 自动标注并人工校验平衡性。

---

### 实验设置与评估指标

#### 评估指标
- **Token F1**：token 级幻觉检测准确率
- **Example F1**：整个 response 是否含幻觉的判断准确率
- **Hallucination Recall**：特别强调——漏检幻觉是高风险事件，因此召回率至关重要

#### 基线方法对比
- **8K 模型（truncated）**：将长文档截断至前 8K tokens 后送入标准 encoder 验证器
- **32K 模型（ours）**：本文提出的 full-context verifier
- 对比不同 exit layer 设置下的 early-exit 版本

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ RQ1: 扩展上下文是否会损害短文档性能？  
> **结论：不会**

| Model | Context | Token F1 | Example F1 |
|-------|--------|----------|------------|
| LettuceDetect-large | 8K | 0.6158 | 79.22% |
| Ours (32K) | 32K | 0.5337 | 77.00% |

- 尽管上下文扩大 4 倍，短文档性能**基本持平**，说明扩展未破坏原有能力。

---

#### ✅ RQ2: 全文档验证是否显著优于截断验证？  
> **结论：大幅提升幻觉召回能力**

| Metric | 32K Model | 8K Model | Improvement |
|--------|-----------|----------|-------------|
| Samples Truncated | 0% | 95% | — |
| Hallucination Recall | **0.55** | **0.06** | **+817%** |
| Hallucination F1 | **0.50** | **0.10** | **+400%** |

- 8K 模型因截断丢失大量证据，几乎无法发现幻觉。
- 本文方法在长文档上**召回率提升超过 8 倍**，证明 full-context verification 至关重要。

> 📌 **补充发现**：文档越长，8K 截断损失越大，性能差距越明显。

---

#### ✅ RQ3: 关键设计选择的影响（消融实验）

##### (a) 损失函数选择
- **Focal Loss / 加权 CrossEntropy**：虽提高 recall，但 precision 显著下降 → 导致过多误报（false alarms），用户体验差。
- **Standard CrossEntropy**：取得最佳 F1，更适合部署环境。

##### (b) 训练数据分布
- 使用高幻觉率的 QA-style 合成数据 → 模型变得“过度悲观”，倾向于将合法回答判为幻觉。
- 使用与目标 RAG 分布匹配的数据（如 DART, E2E）→ 更鲁棒，F1 更高。

##### (c) Long-Context 扩展策略
| 方法 | >1K 距离预测准确率 |
|------|------------------|
| Naive RoPE + MLM | 断崖式下降（<30%） |
| + Retrieval-aware masking + EWC | 维持较高水平（~70%） |

- 表明 long-context 能力瓶颈在于 fine-tuning 动力学，而非位置编码本身。

---

#### ✅ RQ4: Early-Exit 推理效果

| Exit Layer | Example F1 | Compute (% of full) | Speedup |
|----------|------------|--------------------|---------|
| Full model (L22) | 95.5% | 100% | 1.0× |
| Intermediate (L16) | 92.8% | 73% | **1.4×** |
| Mid (L11) | 81.2% | 50% | **2.0×** |
| Early (L6) | 48.2% | 27% | **3.3–3.9×** |

- **L16 出口** 是理想折中点：仅损失 2.7% F1，节省 27% 计算，提速 1.4×。
- 长序列下优势更明显（attention 成本随长度平方增长）。

---

## 4. 关键结论和发现

### 主要发现

1. 🔹 **Full-document verification 对长文档至关重要**  
   - 截断式验证（truncated validation）在现实文档中严重失效，关键证据常位于文档中后部。
   - 8K 模型的 hallucination recall 不足 0.1，实用性极低。

2. 🔹 **Encoder 可以扩展至 32K 上下文而不牺牲性能**  
   - 通过 retrieval-aware masking 和 EWC，成功保留 long-range attention 能力。
   - 扩展后的模型在短文档上表现不降，在长文档上大幅领先。

3. 🔹 **Early-exit 是实现生产级实时性的关键技术**  
   - 在中间层退出即可获得接近完整的检测性能，显著降低延迟。
   - 特别适用于变长输入和异构负载场景。

4. 🔹 **训练数据分布比规模更重要**  
   - 匹配真实 RAG 输出风格的数据更能提升部署稳定性。
   - 单纯增加数据量或使用极端分布会损害泛化能力。

---

### 方法的局限性

1. **当前最大支持 32K tokens**  
   - 某些监管文件或科学论文可能超过 100K tokens，仍需进一步扩展。

2. **依赖高质量标注数据进行 fine-tuning**  
   - 幻觉标注成本高，自动构造可能存在偏差。

3. **Early-exit 层需预先定义**  
   - 缺乏完全动态的 confidence-driven exit 机制。

4. **多语言支持尚未验证**  
   - 当前实验集中在英文领域。

---

### 未来工作方向

1. **扩展至 64K–128K 更长上下文**
2. **开发 confidence-aware dynamic inference**，根据置信度决定是否提前退出
3. **构建多语言长文档 benchmark**，提升跨语言鲁棒性
4. **探索 zero-shot 或 few-shot hallucination detection**，减少对标注数据的依赖
5. **集成到端到端 RAG pipeline 中进行联合优化**

---

> 💡 **实践建议总结（来自 Appendix B）**：
> - 默认使用 **L16 + BS=8** 配置作为实时 API 的推荐设置（延迟 <2ms）
> - 长文档处理采用 **L16 + BS=4** 平衡内存与延迟
> - 最高精度场景使用 **Full model (L22)**，适用于离线质检

该研究为构建**可靠、高效、可扩展的工业级 RAG 系统**提供了重要的技术路径和实践经验。

</details>

---

### 13. [Grounding Arabic LLMs in the Doha Historical Dictionary: Retrieval-Augmented Understanding of Quran and Hadith](https://arxiv.org/abs/2603.23972)

**Authors**: Somaya Eltanbouly, Samer Rashwani  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.23972v1  

#### Abstract
Large language models (LLMs) have achieved remarkable progress in many language tasks, yet they continue to struggle with complex historical and religious Arabic texts such as the Quran and Hadith. To address this limitation, we develop a retrieval-augmented generation (RAG) framework grounded in di...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Grounding Arabic LLMs in the Doha Historical Dictionary: Retrieval-Augmented Understanding of Qur'an and Hadith*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Arabic LLMs** 在处理复杂的**历史与宗教阿拉伯文本**（如《古兰经》和圣训）时表现不佳，主要原因包括：
- 缺乏对**历时性语义演变**（diachronic semantic evolution）的理解；
- 依赖现代网络语料训练，容易产生**时代错位**（anachronistic errors）；
- 面对古典阿拉伯语中的**生僻词、复合表达、变音符号**（diacritics）等语言现象鲁棒性差。

### 🚀 提出的新方法与创新点
本研究提出了一种基于 **Retrieval-Augmented Generation (RAG)** 的新框架，其核心创新在于：
- **首次将 Doha Historical Dictionary of Arabic (DHDA)** —— 一个大规模、结构化的**历时性阿拉伯语词典**——作为外部知识源集成到 RAG 框架中；
- 设计了一个**混合检索管道**（hybrid retrieval pipeline），结合 **BM25（lexical）** 与 **dense embedding models**（如 Nomic Embed v2）进行召回；
- 引入 **cross-encoder re-ranker（BAAI/bge-reranker-v2-m3）** 并对其进行**领域微调**（fine-tuned），以提升相关文档排序精度；
- 提出 **intent-based routing 机制**：在查询预处理阶段识别用户意图（如“词义”、“出处”、“作者”等），据此动态选择提示模板（prompt template）和传递给 LLM 的字段子集，实现精细化生成控制。

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| 知识源 | 通用语料 / 现代词典（如 Riyadh Dictionary） | **DHDA**：覆盖千年历史、含引文证据的**历时性词典** |
| 检索策略 | 单一 sparse 或 dense 检索 | **Hybrid retrieval + fine-tuned re-ranker**，显著提升召回率与排序质量 |
| 生成控制 | 固定 prompt 模板 | **Intent-driven prompt structuring**，按需提供上下文，减少信息冗余与幻觉 |
| 应用场景 | 多集中于 Modern Standard Arabic | 聚焦 **Classical Arabic** 及宗教文本理解 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **主知识库**：**Doha Historical Dictionary of Arabic (DHDA)**  
  - 包含超过 **198,346 条结构化词条**，每条记录包含：
    - 词汇单位（Lexical Unit）
    - 词根（Root）
    - 引文（Shahid）
    - 含义、语境、年代、来源、作者、词源等字段
  - 内容涵盖从早期阿拉伯语到《古兰经》《圣训》的历时发展
- **检索语料库**（Retrieval Corpus）：
  - 将每个词条构造成独立文档，去除变音符号以提高匹配率，但保留原始带 diacritics 形式用于生成阶段
- **问答测试集**（Evaluation QA Dataset）：
  - 自动通过模板生成，共 **2,000 个问题**，聚焦出现在《古兰经》或圣训中的词汇
  - 覆盖 10 类意图（见下表）
  - 子集：**1,000 个“词义类”问题**用于基线比较

#### 表：意图类别分布（Intent Categories）
| Intent | 描述 | Prompt Strategy |
|-------|------|----------------|
| Meaning, Contextual Meaning | 定义与语境含义 | Few-shot |
| Author, Date, Source, Etymology, Inscriptions | 出处、时间、词源等事实型查询 | Zero-shot |
| Part of Speech, Morphology | 语法与形态分析 | Few-shot |
| Other | 多意图或比较类问题 | Zero-shot |

### ⚙️ 实验设置
- **模型使用**：
  - **LLMs**: Fanar, ALLaM（均为 Arabic-centric LLMs）、Gemini 2.5 Pro（作为上限参考）
  - **Embedding Models**: Jina v3, BGE-m3, Arctic-Embed 2.0, Nomic Embed v2
  - **Re-ranker**: BAAI/bge-reranker-v2-m3（微调后）
- **检索配置**：
  - Top-k = 10 文档返回
  - 使用 FAISS-GPU 加速向量搜索
- **生成参数**：
  - Temperature = 0（确保输出确定性）
  - Prompt 构造依据 intent 分类结果
- **评估流程**：
  - 先进行 intent 分类 → 检索 → 重排序 → LLM 生成答案

### 📏 评估指标
| 类别 | 指标 | 说明 |
|-----|------|------|
| **检索性能** | MRR, MAP, Recall@10 (R@10) | 衡量相关文档是否被准确检出并排在前列 |
| **生成正确性** | Gemini-as-a-Judge Score (0–100%) | 使用 **Gemini 2.5 Pro** 对生成答案打分，衡量事实准确性与完整性 |
| **人工验证** | Human Evaluation (n=200) | 验证 Gemini 打分可靠性，计算 Kappa、Pearson 相关性等 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ 检索性能（Retrieval）
| 方法 | R@10 | MRR | MAP |
|------|------|-----|-----|
| BM25（baseline） | 0.586 | 0.677 | 0.522 |
| Nomic Embed v2（best dense） | 0.531 | 0.673 | 0.441 |
| BM25 + Re-ranker | 0.635 | 0.886 | 0.572 |
| **BM25 + Fine-tuned Re-ranker** | **0.647** | **0.936** | **0.611** |
| **Hybrid (BM25+Nomic) + FT Re-ranker** | **0.652** | **0.945** | 0.609 |

> 💡 结论：**微调后的 re-ranker 显著提升所有指标**；混合检索在 MRR 和 R@10 上略优，但 BM25 单独 + 微调 re-ranker 综合表现最佳。

#### ✅ 生成性能（Generation Accuracy）
| 模型 | Baseline | RAG-ZS | RAG-FS |
|------|----------|--------|--------|
| Fanar | 54% | **89%** | 89% |
| ALLaM | 56.85% | **86%** | 78% |
| Gemini | 88% | 96% | 95% |

> ✅ RAG 架构使 Fanar 和 ALLaM 性能提升超 **30个百分点**，准确率均突破 **85%**，大幅缩小与 Gemini 的差距。

#### ✅ 扩展问题集上的平均得分（Zero-shot 设置）
- **Fanar-based RAG**: ~87%
- **ALLaM-based RAG**: ~87%
- 表明系统在多种问题类型上具有**强泛化能力**

#### ✅ 按问题类型的性能分析（Figure 3）
| 问题类型 | Fanar | ALLaM |
|--------|-------|-------|
| Author of Citation | 95.7% | 90.6% |
| Contextual Meaning | 95.0% | 87.5% |
| Source of Citation | 94.3% | 81.9% |
| Historical Date | 85.0% | **90.6%** |
| Part of Speech | **72.2%** | **79.5%** |
| Basic Meaning | 89.9% | 80.0% |

> 🔍 发现：
> - Fanar 在“作者”“语境意义”等问题上表现极佳；
> - ALLaM 在“历史日期”和“词性标注”任务上更稳健；
> - 两模型存在**互补优势**，适合构建 ensemble 系统。

#### ✅ 消融实验与关键发现
- **Intent分类器有效性**：
  - 使用 TF-IDF + Random Forest 实现高精度意图识别，支持下游路由决策
- **Few-shot vs Zero-shot**：
  - Fanar 在 few-shot 下无增益（可能因上下文过长导致注意力分散）
  - ALLaM 在 few-shot 下性能下降至 78%，反映其对**长上下文敏感**
- **Gemini 作为自动评估器的有效性验证**：
  - 与人类评分高度一致：
    - **精确匹配率：83%**
    - 差异 ≤1 类别的比例：>95%
    - **Weighted Cohen’s Kappa = 0.87**（几乎完美一致性）
    - Pearson 相关系数 = 0.87 (p < 0.001)

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **DHDA 是增强 Arabic LLMs 理解古典文本的关键资源**：
   - 提供可验证的历史语义证据，有效缓解 LLM 的“幻觉”与时代错位问题。
2. **RAG + 历时词典显著提升性能**：
   - Fanar 和 ALLaM 在接入 DHDA 后，准确率从约 **55% 提升至 >85%**，接近 Gemini 水平。
3. **混合检索 + 微调 re-ranker 是最优检索方案**：
   - 特别是 **BM25 + fine-tuned cross-encoder** 在各项指标上领先。
4. **Intent-based routing 提升生成效率与准确性**：
   - 动态构造 prompt 减少无关信息干扰，提升指令遵循能力。

### ⚠️ 局限性
1. **语言复杂性仍是瓶颈**：
   - **Diacritics 敏感度不足**：模型常混淆仅变音不同的词形（如 *al-* vs *il-*）
   - **复合短语歧义**：难以区分相似短语的真实指涉对象
2. **检索失败导致错误传播**：
   - 若检索未命中正确文档，模型无法自主纠正
3. **小型 LLM 指令遵循能力弱于 Gemini**：
   - 在“无法回答”情况下未能正确响应（如未输出 `لا يمكن الإجابة`）
4. **问题集为模板生成，缺乏真实用户多样性**

### 🔮 未来工作方向
1. **提升对 diacritics 和复合表达的建模能力**：
   - 开发专门针对古典阿拉伯语的 tokenization 与 embedding 方法
2. **动态调整检索数量与策略**：
   - 根据 query intent 自适应决定 k 值或是否启用 dense retrieval
3. **构建人类撰写的 benchmark 数据集**：
   - 更贴近真实使用场景，评估实际可用性
4. **探索 ensemble 方法**：
   - 利用 Fanar 与 ALLaM 的互补优势，设计投票或融合机制
5. **扩展至其他文化经典文本理解**：
   - 如波斯语文献、伊斯兰哲学著作等

---

> 🔗 **代码与资源公开地址**：[https://github.com/somayaeltanbouly/Doha-Dictionary-RAG](https://github.com/somayaeltanbouly/Doha-Dictionary-RAG)

</details>

---

### 14. [Optimizing Multilingual LLMs via Federated Learning: A Study of Client Language Composition](https://arxiv.org/abs/2603.24242)

**Authors**: Aleix Sant, Jordi Luque, Carlos Escolano  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.24242v1  

#### Abstract
Federated Learning (FL) of Large Language Models (LLMs) in multilingual environments presents significant challenges stemming from heterogeneous language distributions across clients and disparities in language resource availability. To address these challenges, we extended the FederatedScope-LLM fr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimizing Multilingual LLMs via Federated Learning: A Study of Client Language Composition

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于**多语言环境下的 Federated Learning（FL）训练 Large Language Models（LLMs）所面临的挑战**，主要包括：
- 客户端之间语言分布的高度异质性（non-IID），即不同客户端持有不同语言的数据；
- 语言资源不平衡（高资源语言 vs. 低资源语言）导致模型在低资源语言上表现较差；
- 传统 FL 中因数据非独立同分布（non-IID）引发的 **client drift** 问题，在多语言场景下尤为严重。

### 提出的新方法与新思路
1. **扩展 FederatedScope-LLM 框架以支持多语言 FL 实验**  
   - 支持灵活的 prompt 集成、语言感知的样本处理以及多语言 FL 数据管道。
   - 实现了对 LLM 进行多语言联邦指令微调（instruction-tuning）的能力。

2. **提出 Local Dynamic Early Stopping for FL (LDES-FL)**  
   - 一种基于客户端本地验证损失的动态早停机制。
   - 允许每个客户端根据其本地验证性能自主决定是否暂停或恢复训练。
   - 支持“自适应重加入”（adaptive rejoining）：即使已停止训练的客户端，若接收到的新全局模型在其本地验证集上表现更好，可重新启动训练。

3. **系统研究客户端语言组成的影响**  
   - 设计了一系列从完全单语（100% mono）到高度多语（15% mono）的客户端配置，探究 within-client multilinguality 对性能、公平性和效率的影响。

### 相比现有方法的优势
| 方法 | 优势 |
|------|------|
| **LDES-FL** | 相比标准的全局早停机制，能更精细地控制各客户端的收敛节奏，减少不必要的计算开销，提升训练效率；同时允许落后客户端重新参与，缓解训练停滞问题。 |
| **多语言客户端设计** | 通过增加客户端内部的语言多样性，使本地目标更接近全局目标，有效缓解 client drift，尤其提升了低资源语言的表现。 |
| **框架扩展性** | 所构建的实验平台为未来多语言 FL 研究提供了可复用的基础工具链。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ALPACA CLEANED 多语言版本**，覆盖 8 种语言（ISO 639-1 编码）：
  - English (en), Spanish (es), German (de), Catalan (ca), Danish (da), Serbian (sr), Croatian (hr), Basque (eu)
- 每种语言包含 52,002 个样本，经 GPT-4 清洗并大规模英译扩增。
- 每个客户端分配：
  - 48,960 训练样本
  - 1,020 验证样本
- 服务器端保留一个固定的多语言测试集：每语言 501 样本（共 4,008）

### 实验设置
- **模型架构**：`salamandra-2b-instruct`，一个多语言预训练 LLM，涵盖 35 种欧洲语言及代码。
- **参数高效微调（PEFT）**：采用 **LoRA**（Low-Rank Adaptation），rank=16，scaling factor=32。
- **优化器**：OneBitAdam（lr=0.001, grad clip=1.0）
- **本地训练**：每轮通信执行 160 mini-batch 步骤，micro-batch size=2，gradient accumulation=16 → 相当于每轮 10 次 optimizer 更新。
- **通信协议**：FedAvg，所有客户端权重相等（因数据量一致）。
- **早停机制**：主要使用 **LDES-FL**（patience=1），并与标准联邦早停对比。

### 客户端语言组成设计（关键变量）
| 配置 | 单语比例 | 多语比例 | 异质性等级 |
|------|--------|--------|----------|
| 100% mono | 100% | 0% | 最高（最 non-IID） |
| 85% mono | 85% | 15% | ↓ |
| ... | ... | ... | ... |
| 15% mono | 15% | 85% | 最低（趋近 IID） |

> 注：总样本数保持不变，仅调整语言构成比例。

### 评估指标
- **ROUGE-L**：衡量生成文本与参考答案的最长公共子序列匹配度。
- **FBERT（BERTScore）**：基于 BERT 的语义相似度评分。
- **Mean Score**：跨语言平均得分（反映整体质量）
- **Standard Deviation (σ)**：跨语言得分的标准差（反映 multilingual fairness，越小越公平）
- **Total Optimization Steps**：总优化步数（反映训练成本）

### 基线方法对比
| 基线类型 | 描述 |
|--------|------|
| **Base Model** | 未经微调的原始 `salamandra-2b-instruct` 模型 |
| **Local FT (lang)** | 各语言单独集中式微调（无共享） |
| **Local FT (multilingual)** | 所有语言数据合并后集中微调（视为性能上限） |
| **FedAvg (不同 mono/multi 配比)** | 联邦学习下的不同客户端语言组成设置 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3）

| 模型 | Mean ROUGE-L | σ_ROUGE-L | Mean FBERT | σ_FBERT | Optim. Steps |
|------|--------------|-----------|------------|----------|-------------|
| Base Model | 0.167 | 7.70e-2 | 0.867 | 1.57e-2 | — |
| Local FT (multilingual) | **0.237** | 5.90e-2 | **0.884** | **1.12e-2** | 1.83e5 |
| FedAvg (100% mono) | 0.203 | 6.47e-2 | 0.877 | 1.26e-2 | 1.12e4 |
| FedAvg (15% mono) | **0.221** | 6.09e-2 | **0.880** | 1.22e-2 | 7.57e4 |

> ✅ 所有联邦模型均显著优于 Base Model（p < 0.05）

### 与基线方法的对比结果
- **vs. Base Model**：所有 FedAvg 设置均大幅提升性能，特别是在 ROUGE-L 上。
- **vs. Local FT (mono)**：
  - 联邦模型在平均性能上超越所有单语本地微调模型的平均水平；
  - 尤其在中低资源语言上优势明显（见 Table 5）；
  - 但单语本地模型在其专属语言上仍具最强 specialization。
- **vs. Local FT (multilingual)**：
  - 集中式多语言微调仍是性能天花板（Mean ROUGE-L 达 0.237）；
  - 最佳联邦设置（15% mono）达到 0.221，接近上限，且训练成本远低于集中式（仅约 41% 的优化步数）。

### 消融实验结果（LDES-FL vs. Standard Early Stopping）

| 设置 | 方法 | Mean ROUGE-L | Optim. Steps | 节省幅度 |
|------|------|---------------|----------------|---------|
| 100% mono | Standard | 0.202 | 1.44e4 | — |
| 100% mono | **LDES-FL** | ≈0.203 | **1.12e4** | ↓ ~22% |
| 15% mono | Standard | 0.224 | 1.11e5 | — |
| 15% mono | **LDES-FL** | ≈0.221 | **7.57e4** | ↓ ~32% |

> 🔍 结论：LDES-FL 在不牺牲性能的前提下显著降低计算开销，因此被用于后续主实验。

### 不同语言组别的性能增益（Table 5）

| 资源等级 | 语言 | FedAvg (100% mono) → (15% mono) ROUGE-L 提升 | Δ (绝对增益) |
|--------|------|----------------------------------|------------|
| High (H) | en, es, de | 0.251 → 0.263 | +0.012 |
| Medium (M) | ca, da | 0.213 → 0.229 | +0.016 |
| Low (L) | sr, hr, eu | 0.149 → 0.173 | **+0.024** ✅ |

> 📌 发现：**客户端多语化带来的收益随语言资源下降而增大**，显著缩小了高低资源语言间的性能差距。

---

## 4. 关键结论和发现

### 主要发现
1. **客户端语言组成是多语言 FL 的关键设计变量**：
   - 增加客户端内部的多语言性（within-client multilinguality）能显著提升全局模型的质量和公平性。
   - 更多语化的客户端减少了 client drift，使其本地更新更贴近全局目标。

2. **联邦学习更适合构建统一的高质量多语言模型**：
   - 尽管不如集中式多语言微调强大，但 FedAvg 在合理配置下（如 15% mono）可逼近其性能（0.221 vs. 0.237）。
   - 显著优于多个独立的单语本地微调模型的平均表现。

3. **单语本地微调擅长语言特异性优化，但牺牲整体平衡性**：
   - 在特定语言上有最佳表现，但跨语言性能波动大（σ 高），不适合追求通用多语言能力的场景。

4. **LDES-FL 提升训练效率与可持续性**：
   - 动态控制客户端训练状态，避免无效迭代。
   - 特别是在多语客户端中，“重加入”现象频繁发生，说明这些客户端能持续从全局聚合中受益。

5. **低资源语言获益最大**：
   - 当客户端变得更“多语”时，低资源语言的性能提升最为显著（↑16.11%），有助于实现更公平的多语言发展。

### 方法的局限性
- **数据分割未严格对齐**：翻译实例可能出现在不同客户端的不同划分中，可能导致虚假的跨语言迁移效果。
- **假设理想通信条件**：未考虑现实中的带宽限制、设备掉线等问题。
- **固定客户端数量与数据规模**：忽略了真实世界中客户端数据极度不均衡的情况。
- **仅使用 LoRA**：未探索其他 PEFT 方法（如 prefix-tuning 或 adapter）在该设定下的表现差异。

### 未来工作方向
- 构建更严格的平行多语言数据划分，以准确评估真正的 cross-lingual transfer。
- 探索结合语言家族聚类（language family clustering）或个性化 PEFT（如 FedP-EFT）进一步提升性能。
- 将 LDES-FL 应用于更大规模模型或多模态 FL 场景。
- 研究如何在保护隐私的同时引入少量共享代理数据（proxy data）来进一步缓解 non-IID 问题。

--- 

> 💡 **一句话总结**：  
> 本文揭示了**客户端语言多样性是提升多语言 FL 性能的关键杠杆**——越“多语”的客户端，越能训练出更强、更公平的全球模型，尽管代价是更多优化步骤；而提出的 **LDES-FL** 机制则有效提升了训练效率，为绿色、可持续的分布式多语言 AI 提供了新路径。

</details>

---

### 15. [Semantic Centroids and Hierarchical Density-Based Clustering for Cross-Document Software Coreference Resolution](https://arxiv.org/abs/2603.24246)

**Authors**: Julia Matela, Frank Kr\"uger  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.24246v1  

#### Abstract
This paper describes the system submitted to the SOMD 2026 Shared Task for Cross-Document Coreference Resolution (CDCR) of software mentions. Our approach addresses the challenge of identifying and clustering inconsistent software mentions across scientific corpora. We propose a hybrid framework tha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Semantic Centroids and Hierarchical Density-Based Clustering for Cross-Document Software Coreference Resolution*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对**跨文档软件指代消解**（Cross-Document Coreference Resolution, CDCR）任务中的挑战展开研究，具体聚焦于科学文献中软件实体的识别与聚类。这类任务面临以下难点：
- 软件名称存在大量变体（如全称、缩写、版本号等），例如 *SPSS* vs. *Statistical Package for the Social Sciences*；
- 不同文档中提及同一软件时上下文差异大；
- 自动提取的 mention 存在噪声（尤其在 Subtask 2 和 3 中）；
- 数据规模巨大（Subtask 3 达到近 22 万条 mention），对可扩展性要求高。

### 🚀 提出的新方法与创新思路
作者提出了一种**混合式三阶段框架**，结合语义表示、知识库匹配与密度聚类，实现高效且准确的软件 mention 聚类：

1. **基于 Semantic Centroids 的 Knowledge Base 构建**  
   利用训练集中每个 gold cluster 的 mention embeddings 平均生成一个 L2-normalized centroid 向量，构建 KB。这些 centroid 作为已知软件身份的“语义锚点”。

2. **FAISS + 字符串匹配的双通道检索机制**  
   在推理阶段，测试 mention 首先通过：
   - **Exact string match**：基于归一化后的 canonical name 匹配；
   - **High-confidence semantic match**：使用 FAISS 进行快速近似最近邻搜索（Inner Product 即 cosine similarity），设定阈值 0.7；
   - **Medium-confidence string-corroborated match**（仅用于 Subtask 1&2）：相似度介于 0.5~0.7 但有字符串匹配则仍分配至对应 cluster。

3. **HDBSCAN 密度聚类处理未匹配 mention**  
   对无法匹配到 KB 的 mention，采用 HDBSCAN 进行无监督聚类，无需预设簇数量，并能有效识别离群点。

4. **Blocking 策略提升可扩展性（Subtask 3 专用）**  
   为应对超大规模数据（219,950 条 mention），引入两层 blocking：
   - 按 entity type 分块；
   - 若某块超过 20,000 条，则按 canonical name 首字母进一步划分。
   显著降低距离矩阵计算复杂度（从 $O(n^2)$ 变为多个小块内的 $O(k^2)$）。

5. **Surface-form Normalization 与 Abbreviation Resolution**  
   统一大小写、去除非字母数字字符，并利用元数据中的缩写关系建立 short-to-long 名称映射，提升 canonical name 匹配准确性。

### ⚖️ 相比现有方法的优势
| 优势维度 | 具体体现 |
|--------|---------|
| **准确性高** | 在所有三个子任务上均取得接近最优的 CoNLL F1 分数（0.98, 0.98, 0.96） |
| **可扩展性强** | 引入 blocking 策略后可在 CPU 上处理超 20 万 mention，无需 GPU 加速 |
| **鲁棒性强** | 结合语义与字符串信号，在 noisy 自动提取 mention 场景下仍表现稳定 |
| **灵活性好** | HDBSCAN 支持动态发现新 cluster，适应未知软件的出现 |

---

## 2. 核心实验方法和设置

### 📚 数据集
所有数据来自两个公开资源：
- **SoMeSci** (Schindler et al., 2021)
- **SoftwareKG** (Schindler et al., 2022)

分为三个子任务：

| Subtask | 训练集（mentions/clusters） | 测试集（mentions） | 特点 |
|-------|--------------------------|------------------|------|
| **Subtask 1** | 2974 / 733 | 743 | Gold-standard 标注 mention，质量最高 |
| **Subtask 2** | 2860 / 699 | 12,516 | 自动预测 mention，含噪声 |
| **Subtask 3** | 2860 / 699 | 219,950 | 规模极大，强调可扩展性 |

> 注：Subtask 2 和 3 共享相同训练集。

### 🔬 实验设置
- **Embedding 模型**：`all-MiniLM-L6-v2`（Sentence-BERT），输出 384 维向量
- **Pooling 方式**：attention-mask-weighted mean pooling + L2 normalization
- **FAISS 检索**：使用 `IndexFlatIP` 实现精确内积（即 cosine）搜索
- **HDBSCAN 参数**：
  - Subtask 1&2: `min_cluster_size=2`, `min_samples=1`, `cluster_selection_epsilon=0.5`
  - Subtask 3: 更严格的 `epsilon=0.15`，防止过度合并
- **Blocking 策略**（仅 Subtask 3）：
  - 第一层：按 entity type（如 application, plugin）
  - 第二层：若块 > 20k，按 canonical name 首字母分块

### 🎯 评估指标
采用标准 CDCR 评测指标：
- **MUC F1**
- **BCUB F1**
- **CEAFE F1**
- **CoNLL F1**（前三者的平均值）

---

## 3. 主要实验结果和性能指标

### 📊 性能汇总（Table 1）

| Metric       | Subtask 1 | Subtask 2 | Subtask 3 |
|-------------|-----------|-----------|-----------|
| **MUC F1**   | 0.9939    | 0.9916    | 0.9912    |
| **BCUB F1**  | 0.9905    | 0.9858    | 0.9724    |
| **CEAFE F1** | 0.9584    | 0.9521    | 0.9218    |
| **CoNLL F1** | **0.9809** | **0.9765** | **0.9618** |

> 报告中称“achieved CoNLL F1 scores of 0.98, 0.98, and 0.96”，与表格一致。

### 🔍 结果分析
- **Subtask 1 表现最好**：得益于高质量 gold mention，表面形式干净、上下文可靠；
- **Subtask 2 略有下降**：自动提取带来的噪声影响 embedding 质量和匹配精度；
- **Subtask 3 下降较明显**：主要受大规模下语义空间更密集、block 内误合并风险上升的影响，尤其是 BCUB 和 CEAFE 指标下滑较多。

### ⏱️ 可扩展性分析（Table 2 & Figure 2）
- **Embedding 和 FAISS 匹配阶段呈线性增长**，效率极高；
- **HDBSCAN 阶段运行时间非单调**：25% 样本耗时高于 50%，归因于 blocking 策略中个别大块主导执行时间（record linkage 中常见现象）；
- **总耗时控制良好**：处理完整 219,950 条 mention 仅需约 **129 秒**（约 2 分钟），完全满足大规模应用需求。

### ❌ 无明确消融实验
论文未提供系统的 ablation study，但通过不同子任务的表现变化间接验证了各组件的重要性：
- 从 Subtask 1 → 2 的性能下降反映了 mention 提取质量的影响；
- Subtask 3 的参数调整（如更低的 epsilon）说明 blocking 与 clustering 参数对最终效果至关重要。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **混合策略优于纯端到端模型**：将 supervised centroid matching 与 unsupervised HDBSCAN 相结合，既能利用已有知识，又能发现新实体。
2. **语义 + 字符串双信号增强鲁棒性**：尤其在低置信度情况下加入字符串佐证，显著提升 recall 而不牺牲 precision。
3. **Blocking 是大规模 CDCR 的关键**：合理分块可突破 $O(n^2)$ 复杂度瓶颈，使 HDBSCAN 在数十万级数据上依然可行。
4. **任务本质更接近 Entity Disambiguation**：不同于传统 coreference（多 mention 指向同一实体），此处每个 surface form 几乎唯一对应一个软件，具有强不对称性。

### ⚠️ 局限性
1. **依赖训练集构建 KB**：若测试集中出现训练集完全未见的新类型软件，只能靠 HDBSCAN 发现，而其性能受限于 embedding 质量和 blocking 效果；
2. **Abbreviation mapping 依赖元数据**：若 metadata 缺失或错误，会影响 canonicalization 效果；
3. **缺乏细粒度版本区分能力**：当前方法倾向于将 SPSS 28 和 SPSS 归为一类，可能不符合某些应用场景需求；
4. **未进行消融实验**：难以量化各模块的具体贡献。

### 🔮 未来工作方向
1. **Adaptive threshold selection**：根据上下文或 mention 类型动态调整匹配阈值；
2. **Online centroid updating**：在推理过程中不断更新 KB centroid，形成增量学习机制；
3. **Incorporate citation graph 或 document-level context**：利用引用网络或文档主题信息辅助 disambiguation；
4. **Fine-grained version resolution**：区分主版本、次版本，支持更精细的软件追踪。

---

> ✅ **代码开源地址**：https://github.com/matjulia/somd2026  
> 💡 **一句话总结**：该工作提出了一种高效、可扩展的 hybrid CDCR 框架，在保持高精度的同时成功应对了超大规模软件 mention 聚类挑战。

</details>

---

### 16. [LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load](https://arxiv.org/abs/2603.23640)

**Authors**: Pranay Tummalapalli, Sahil Arayakandy, Ritam Pal, Kautuk Kundan  
**Category**: cs.DC  
**Published**: 2026-03-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.23640v1  

#### Abstract
Deploying large language models on-device for always-on personal agents demands sustained inference from hardware tightly constrained in power, thermal envelope, and memory. We benchmark Qwen 2.5 1.5B (4-bit quantised) across four platforms: a Raspberry Pi 5 with Hailo-10H NPU, a Samsung Galaxy S24 ...

---

### 17. [Mixed-signal implementation of feedback-control optimizer for single-layer Spiking Neural Networks](https://arxiv.org/abs/2603.24113)

**Authors**: Jonathan Haag, Christian Metzner, Dmitrii Zendrikov, Giacomo Indiveri, Benjamin Grewe, Chiara De Luca, Matteo Saponati  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.24113v1  

#### Abstract
On-chip learning is key to scalable and adaptive neuromorphic systems, yet existing training methods are either difficult to implement in hardware or overly restrictive. However, recent studies show that feedback-control optimizers can enable expressive, on-chip training of neuromorphic devices. In ...

---

### 18. [Linear-Nonlinear Fusion Neural Operator for Partial Differential Equations](https://arxiv.org/abs/2603.24143)

**Authors**: Heng Wu, Junjie Wang, Benzhuo Lu  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.24143v1  

#### Abstract
Neural operator learning directly constructs the mapping relationship from the equation parameter space to the solution space, enabling efficient direct inference in practical applications without the need for repeated solution of partial differential equations (PDEs) - an advantage that is difficul...

---

### 19. [Evaluating a Multi-Agent Voice-Enabled Smart Speaker for Care Homes: A Safety-Focused Framework](https://arxiv.org/abs/2603.23625)

**Authors**: Zeinab Dehghani, Rameez Raja Kureshi, Koorosh Aslansefat, Faezeh Alsadat Abedi, Dhavalkumar Thakker, Lisa Greaves, Bhupesh Kumar Mishra, Baseer Ahmad, Tanaya Maslekar  
**Category**: cs.AI  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23625v1  

#### Abstract
Artificial intelligence (AI) is increasingly being explored in health and social care to reduce administrative workload and allow staff to spend more time on patient care. This paper evaluates a voice-enabled Care Home Smart Speaker designed to support everyday activities in residential care homes, ...

---

### 20. [DUPLEX: Agentic Dual-System Planning via LLM-Driven Information Extraction](https://arxiv.org/abs/2603.23909)

**Authors**: Keru Hua, Ding Wang, Yaoying Gu, Xiaoguang Ma  
**Category**: cs.AI  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23909v1  

#### Abstract
While Large Language Models (LLMs) provide semantic flexibility for robotic task planning, their susceptibility to hallucination and logical inconsistency limits their reliability in long-horizon domains. To bridge the gap between unstructured environments and rigorous plan synthesis, we propose DUP...

---

### 21. [From AI Assistant to AI Scientist: Autonomous Discovery of LLM-RL Algorithms with LLM Agents](https://arxiv.org/abs/2603.23951)

**Authors**: Sirui Xia, Yikai Zhang, Aili Chen, Siye Wu, Siyu Yuan, Yanghua Xiao  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23951v1  

#### Abstract
Discovering improved policy optimization algorithms for language models remains a costly manual process requiring repeated mechanism-level modification and validation. Unlike simple combinatorial code search, this problem requires searching over algorithmic mechanisms tightly coupled with training d...

---

### 22. [FinToolSyn: A forward synthesis Framework for Financial Tool-Use Dialogue Data with Dynamic Tool Retrieval](https://arxiv.org/abs/2603.24051)

**Authors**: Caishuang Huang, Yang Qiao, Rongyu Zhang, Junjie Ye, Pu Lu, Wenxi Wu, Meng Zhou, Xiku Du, Tao Gui, Qi Zhang, Xuanjing Huang  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.24051v1  

#### Abstract
Tool-use capabilities are vital for Large Language Models (LLMs) in finance, a domain characterized by massive investment targets and data-intensive inquiries. However, existing data synthesis methods typically rely on a reverse synthesis paradigm, generating user queries from pre-sampled tools. Thi...

---

### 23. [Alignment Reduces Expressed but Not Encoded Gender Bias: A Unified Framework and Study](https://arxiv.org/abs/2603.24125)

**Authors**: Nour Bouchouchi, Thiabult Laugel, Xavier Renard, Christophe Marsala, Marie-Jeanne Lesot, Marcin Detyniecki  
**Category**: cs.CL  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.24125v1  

#### Abstract
During training, Large Language Models (LLMs) learn social regularities that can lead to gender bias in downstream applications. Most mitigation efforts focus on reducing bias in generated outputs, typically evaluated on structured benchmarks, which raises two concerns: output-level evaluation does ...

---

### 24. [Kronecker-Structured Nonparametric Spatiotemporal Point Processes](https://arxiv.org/abs/2603.23746)

**Authors**: Zhitong Xu, Qiwei Yuan, Yinghao Chen, Yan Sun, Bin Shen, Shandian Zhe  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23746v1  

#### Abstract
Events in spatiotemporal domains arise in numerous real-world applications, where uncovering event relationships and enabling accurate prediction are central challenges. Classical Poisson and Hawkes processes rely on restrictive parametric assumptions that limit their ability to capture complex inte...

---

### 25. [Wireless communication empowers online scheduling of partially-observable transportation multi-robot systems in a smart factory](https://arxiv.org/abs/2603.23967)

**Authors**: Yaxin Liao, Qimei Cui, Kwang-Cheng Chen, Xiong Li, Jinlian Chen, Xiyu Zhao, Xiaofeng Tao, Ping Zhang  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23967v1  

#### Abstract
Achieving agile and reconfigurable production flows in smart factories depends on online multi-robot task assignment (MRTA), which requires online collision-free and congestion-free route scheduling of transportation multi-robot systems (T-MRS), e.g., collaborative automatic guided vehicles (AGVs). ...

---

### 26. [Lagrangian Relaxation Score-based Generation for Mixed Integer linear Programming](https://arxiv.org/abs/2603.24033)

**Authors**: Ruobing Wang, Xin Li, Yujie Fang, Mingzhong Wang  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.24033v1  

#### Abstract
Predict-and-search (PaS) methods have shown promise for accelerating mixed-integer linear programming (MILP) solving. However, existing approaches typically assume variable independence and rely on deterministic single-point predictions, which limits solution diversityand often necessitates extensiv...

---

### 27. [AVO: Agentic Variation Operators for Autonomous Evolutionary Search](https://arxiv.org/abs/2603.24517)

**Authors**: Terry Chen, Zhifan Ye, Bing Xu, Zihao Ye, Timmy Liu, Ali Hassani, Tianqi Chen, Andrew Kerr, Haicheng Wu, Yang Xu, Yu-Jung Chen, Hanfeng Chen, Aditya Kane, Ronny Krashinsky, Ming-Yu Liu, Vinod Grover, Luis Ceze, Roger Bringmann, John Tran, Wei Liu, Fung Xie, Michael Lightstone, Humphrey Shi  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.24517v1  

#### Abstract
Agentic Variation Operators (AVO) are a new family of evolutionary variation operators that replace the fixed mutation, crossover, and hand-designed heuristics of classical evolutionary search with autonomous coding agents. Rather than confining a language model to candidate generation within a pres...

---

### 28. [AnalogAgent: Self-Improving Analog Circuit Design Automation with LLM Agents](https://arxiv.org/abs/2603.23910)

**Authors**: Zhixuan Bao, Zhuoyi Lin, Jiageng Wang, Jinhai Hu, Yuan Gao, Yaoxin Wu, Xiaoli Li, Xun Xu  
**Category**: cs.AI  
**Published**: 2026-03-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.23910v1  

#### Abstract
Recent advances in large language models (LLMs) suggest strong potential for automating analog circuit design. Yet most LLM-based approaches rely on a single-model loop of generation, diagnosis, and correction, which favors succinct summaries over domain-specific insight and suffers from context att...

---

### 29. [Steering Code LLMs with Activation Directions for Language and Library Control](https://arxiv.org/abs/2603.23629)

**Authors**: Md Mahbubur Rahman, Arjun Guha, Harshitha Menon  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.23629v1  

#### Abstract
Code LLMs often default to particular programming languages and libraries under neutral prompts. We investigate whether these preferences are encoded as approximately linear directions in activation space that can be manipulated at inference time. Using a difference-in-means method, we estimate laye...

---

### 30. [Unveiling Hidden Convexity in Deep Learning: a Sparse Signal Processing Perspective](https://arxiv.org/abs/2603.23831)

**Authors**: Emi Zeger, Mert Pilanci  
**Category**: cs.LG  
**Published**: 2026-03-26  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.23831v1  

#### Abstract
Deep neural networks (DNNs), particularly those using Rectified Linear Unit (ReLU) activation functions, have achieved remarkable success across diverse machine learning tasks, including image recognition, audio processing, and language modeling. Despite this success, the non-convex nature of DNN lo...

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
