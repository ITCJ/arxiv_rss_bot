# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2025-12-31 05:53:44 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [AKG kernel Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis](https://arxiv.org/abs/2512.23424)

**Authors**: Jinye Du, Quan Yuan, Zuyao Zhang, Yanzhi Yi, Jiahui Hu, Wangyi Chen, Yiyang Zhu, Qishui Zheng, Wenxiang Zou, Xiangyu Chang, Zuohe Zheng, Zichun Ye, Chao Liu, Shanni Li, Renwei Zhang, Yiping Deng, Xinwei Hu, Xuefeng Jin, Jie Zhao  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2512.23424v1  

#### Abstract
Modern AI models demand high-performance computation kernels. The growing complexity of LLMs, multimodal architectures, and recommendation systems, combined with techniques like sparsity and quantization, creates significant computational challenges. Moreover, frequent hardware updates and diverse c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AKG Kernel Agent: A Multi-Agent Framework for Cross-Platform Kernel Synthesis

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代AI模型对高性能计算核（computation kernels）的需求日益增长，尤其是大语言模型（LLMs）、多模态架构和推荐系统等复杂模型。然而，手动优化kernel代码面临以下挑战：
- **开发成本高**：需要深厚的算法理解和硬件知识（如内存层次、并行执行模型）。
- **可移植性差**：针对特定硬件（GPU/NPU/CPU）的手写优化难以跨平台复用。
- **自动化瓶颈**：现有的LLM生成方法在正确性和性能之间难以平衡，且缺乏系统化知识集成机制。

因此，如何实现**高效、正确、可移植的自动kernel生成**成为AI系统发展的关键瓶颈。

---

### 提出的新方法与创新点

作者提出了 **AKG Kernel Agent** —— 一个基于多智能体（multi-agent）的自动化kernel生成框架，其核心创新包括：

#### （1）**模块化的多智能体协作架构**
由四个专业化Agent协同完成kernel生成任务：
- **Designer**：分析算子语义和硬件特性，生成与DSL无关的中间表示 **Unified Sketch**，描述并行策略、数据流和内存访问模式。
- **Coder**：将Unified Sketch翻译为目标DSL（如Triton、CUDA-C、TileLang、CPP）的可执行代码。
- **Verifier**：验证生成代码的**正确性**（数值精度）和**性能**（执行时间），提供反馈。
- **Conductor**：作为中央协调者，动态路由错误（syntax → Coder；algorithmic → Designer），实现闭环迭代优化。

> ✅ **优势**：解耦高层优化决策与底层代码实现，提升可解释性和调试效率。

#### （2）**文档驱动的知识集成框架（Document-Driven Integration, DDI）**
通过标准化文档接口（DocSpec）注入知识：
- 支持四种文档类型：基础语法、API文档、专家建议、参考示例。
- 新DSL或硬件只需提供符合规范的文档即可接入，无需修改Agent逻辑。

> ✅ **优势**：极大增强了系统的**扩展性**和**通用性**，支持Triton、CUDA-C、TileLang、AscendC、CPP等多种DSL及GPU/NPU/CPU后端。

#### （3）**分层检索增强生成（Hierarchical Code Retrieval）**
为解决传统RAG在kernel生成中“表面相似但语义不同”的问题，提出三级过滤机制：
1. LLM提取任务特征 → 向量匹配计算逻辑；
2. 硬过滤（DSL/Backend/Operator Type）；
3. 基于形状嵌入的语义匹配（shape compatibility）。

> ✅ **优势**：显著提高检索相关性，减少无效上下文干扰。

#### （4）**基于搜索的迭代优化（Iterative Search-Based Optimization）**
采用岛模型（island model）进行多轮探索：
- 并行生成多个候选kernel；
- 使用LLM对比分析优劣实现，提炼有效优化策略；
- 定期迁移精英个体，保持多样性；
- 利用Unified Sketch作为稳定优化锚点。

> ✅ **优势**：实现系统性的性能爬升，避免局部最优。

---

### 相比现有方法的优势

| 方法类别 | 典型代表 | 局限 | AKG Kernel Agent 的改进 |
|--------|--------|------|-------------------------|
| 单一LLM生成 | GPT-4, Deepseek-R1 | 编译错误多、性能不稳定 | 多Agent分工 + 错误定向修复 |
| 微调专用模型 | KernelLLM, Kevin-32B | 数据稀缺限制泛化能力 | 文档驱动 + 检索增强，缓解数据依赖 |
| 固定流程Agent系统 | Astra, QiMeng-Kernel | 难以适应新平台 | 统一Sketch + DDI，天然支持跨平台 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **KernelBench Level 1** [9]：包含100个固定输入形状的算子，用于标准评测。
- **自研Benchmark（本文贡献之一）**：
  - 包含 **198（动态）/214（静态）个算子**，覆盖8类常见操作：
    - Element-wise（激活函数、算术）
    - Reduction（Softmax, ArgMax）
    - Normalization（LayerNorm, RMSNorm）
    - MatMul（批处理矩阵乘）
    - Fused Ops（SiLU-and-Mul, GELU-and-Mul, FFN块）
  - ✅ 特色：引入**动态输入形状测试**，更贴近真实部署场景。
  - ✅ 修复了KernelBench中存在的reward-hacking漏洞。

---

### 实验设置
- **硬件平台**：
  - GPU：NVIDIA A100（CUDA backend）
  - NPU：Huawei Ascend 910B（Ascend backend）
  - CPU：Intel x86_64
- **目标DSL**：
  - Triton（支持CUDA和Ascend）
  - TileLang
  - CUDA-C
  - CPP
- **前端框架**：PyTorch 2.6
- **LLM后端**：DeepSeek V3.1（non-reasoning mode）

---

### 评估指标
| 指标 | 定义 | 用途 |
|-----|------|------|
| **pass@k** | 在k次独立生成中至少有一次正确的概率 | 衡量**正确率** |
| **Speedup** | $ T_{\text{baseline}} / T_{\text{generated}} $ | 衡量**性能增益** |
| **Fast(p%)** | 达到speedup ≥ p 的算子占比 | 性能达标比例 |
| **Geom. Mean Speedup** | 几何平均speedup（防异常值偏移） | 综合性能评价 |

- **Baseline**：PyTorch Eager模式下的原生实现。
- **配置**：
  - 正确性测试：每算子生成4个样本 → 计算 **pass@4**
  - 性能优化：Evolve模块，P=4（并行生成数），R=3轮迭代，K=2个island

---

## 3. 主要实验结果和性能指标

### （1）正确性结果（pass@4）

#### 在 **KernelBench Level 1** 上的结果（共100算子）：

| DSL-Backend | Overall Pass@4 |
|------------|----------------|
| Triton-CUDA | **100.0%** |
| Triton-Ascend | 75.0% |
| CPP-CPU | 91.0% |
| TileLang | 44.0% |
| CUDA-C | 59.0% |

> 🔹 Triton-CUDA表现最佳，所有类别均达100%正确率。

#### 在 **自研Benchmark（动态形状）** 上的结果：

| Operator Category | Triton-CUDA (%) | Triton-Ascend (%) |
|------------------|------------------|--------------------|
| Element-wise | 100.0 | 98.6 |
| Reduction | 93.3 | 93.3 |
| Normalization | 92.6 | 88.9 |
| MatMul | 72.7 | 72.7 |
| Fused Ops | 71.4 | 57.1 |
| **Overall** | **90.9** | **85.4** |

> ✅ 表明系统在**动态输入下仍具备高鲁棒性**。

---

### （2）性能结果（vs. PyTorch Eager）

#### 在 **KernelBench Level 1** 上的几何平均speedup（Geom. Mean）

| DSL-Backend | Overall Speedup | Fast≥1.0 (%) |
|-----------|------------------|---------------|
| **Triton-Ascend** | **1.46×** | 65.5% |
| Triton-CUDA | 1.06× | 68.0% |
| CPP-CPU | 1.04× | 54.9% |

> 📈 **最高提速达1.46倍**（Triton-Ascend），尤其在Reduce & Norm类算子上表现突出（最高1.66×）。

#### 分类性能亮点：
- **Reduce & Norm**（如LayerNorm）：
  - Triton-Ascend: **1.66×**
  - 原因：PyTorch Eager通常拆分为多个小kernel，而AKG生成的是**融合kernel**，减少中间内存开销。
- **Scan & Loss**：
  - CPP-CPU: 高达 **9.00×**
- **MatMul**：
  - 尽管cuBLAS已高度优化，AKG仍能达到约 **1.1–1.56×**，说明生成质量极高。
- **Convolution**：
  - Triton不原生支持conv，需手动实现滑窗，性能低于baseline（~0.4×），故未作为重点。

---

### （3）消融实验与关键发现（文中隐含分析）

虽然未明确列出消融表，但从设计和结果可推断：
- **Unified Sketch的有效性**：
  - 同一Sketch可在不同backend间复用，实现跨平台迁移。
- **Conductor的adaptive routing价值**：
  - 相比固定流水线，能更快定位错误根源，减少无效迭代。
- **Hierarchical Retrieval的作用**：
  - 显著提升Coder生成质量，尤其是在复杂算子（如fused ops）中。
- **Iterative Optimization的收益**：
  - 多轮演化带来持续性能提升，尤其在初始生成较弱时效果明显。

---

## 4. 关键结论和发现

### 主要结论
1. **多Agent协作是解决kernel生成复杂性的有效范式**：
   - 通过职责分离（design/code/verify/orchestrate），实现了更高正确率和可维护性。
2. **文档即接口（Documentation-as-API）是实现可扩展的关键**：
   - DDI框架使系统能快速适配新DSL和硬件，无需重写核心逻辑。
3. **生成质量足够接近甚至超越人工优化水平**：
   - 在多种DSL-backend组合下达到**100% pass@4**，并在多个类别实现**显著性能提升**（最高1.46×）。
4. **支持动态形状的能力提升了实用性**：
   - 自研benchmark证明系统能在真实变化输入下保持健壮。

---

### 方法的局限性
1. **复杂融合算子仍有挑战**：
   - 如Fused Ops的pass率较低（~70%），需进一步优化。
2. **卷积类算子支持有限**：
   - 当前Triton后端无法充分发挥conv性能，受限于编程模型本身。
3. **部分硬件后端性能尚未完全释放**：
   - 如TileLang和CUDA-C的整体表现不如Triton。
4. **依赖高质量文档输入**：
   - 若文档缺失或不规范，会影响集成效果。

---

### 未来工作方向（作者提出）
1. 引入**强化学习**指导搜索过程，基于性能反馈自动调整优化策略。
2. 扩展Unified Sketch语言，支持更复杂的优化原语（如tensor core利用、异步执行）。
3. 开发**自动化文档生成工具**，从现有代码库中提取DSL/hardware知识。
4. 探索**fine-tuning策略**，利用AKG生成的高质量kernel数据反哺LLM，形成正向循环。

---

## 总结

✅ **AKG Kernel Agent 是一个面向生产级需求的全自动kernel生成系统**，它通过：
- **多Agent协作**实现责任解耦，
- **文档驱动集成**保障可扩展性，
- **分层检索+迭代优化**确保高质量输出，

成功解决了AI时代下kernel开发中的**性能、可移植性与自动化**三大难题。其实验结果表明，在多个平台上均可生成**正确且高性能**的kernel代码，平均提速达 **1.46×**，具备广泛的应用前景。

</details>

---

### 2. [SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search](https://arxiv.org/abs/2512.23167)

**Authors**: Yifan Zhang, Giridhar Ganapavarapu, Srideepika Jayaraman, Bhavna Agrawal, Dhaval Patel, Achille Fokoue  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2512.23167v1  

#### Abstract
Large Language Models (LLMs) often falter at complex planning tasks that require exploration and self-correction, as their linear reasoning process struggles to recover from early mistakes. While search algorithms like Monte Carlo Tree Search (MCTS) can explore alternatives, they are often ineffecti...

---

### 3. [Agent2World: Learning to Generate Symbolic World Models via Adaptive Multi-Agent Feedback](https://arxiv.org/abs/2512.22336)

**Authors**: Mengkang Hu, Bowei Xia, Yuran Wu, Ailing Yu, Yude Zou, Qiguang Chen, Shijian Wang, Jiarui Jin, Kexin Li, Wenxiang Jiao, Yuan Lu, Ping Luo  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.22336v1  

#### Abstract
Symbolic world models (e.g., PDDL domains or executable simulators) are central to model-based planning, but training LLMs to generate such world models is limited by the lack of large-scale verifiable supervision. Current approaches rely primarily on static validation methods that fail to catch beh...

---

### 4. [DICE: Discrete Interpretable Comparative Evaluation with Probabilistic Scoring for Retrieval-Augmented Generation](https://arxiv.org/abs/2512.22629)

**Authors**: Shiyan Liu, Jian Ma, Rui Qu  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2512.22629v1  

#### Abstract
As Retrieval-Augmented Generation (RAG) systems evolve toward more sophisticated architectures, ensuring their trustworthiness through explainable and robust evaluation becomes critical. Existing scalar metrics suffer from limited interpretability, inadequate uncertainty quantification, and computat...

---

### 5. [HalluMat: Detecting Hallucinations in LLM-Generated Materials Science Content Through Multi-Stage Verification](https://arxiv.org/abs/2512.22396)

**Authors**: Bhanu Prakash Vangala, Sajid Mahmud, Pawan Neupane, Joel Selvaraj, Jianlin Cheng  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2512.22396v1  

#### Abstract
Artificial Intelligence (AI), particularly Large Language Models (LLMs), is transforming scientific discovery, enabling rapid knowledge generation and hypothesis formulation. However, a critical challenge is hallucination, where LLMs generate factually incorrect or misleading information, compromisi...

---

### 6. [Replay Failures as Successes: Sample-Efficient Reinforcement Learning for Instruction Following](https://arxiv.org/abs/2512.23457)

**Authors**: Kongcheng Zhang, Qi Yao, Shunyu Liu, Wenjian Zhang, Min Cen, Yang Zhou, Wenkai Fang, Yiru Zhao, Baisheng Lai, Mingli Song  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2512.23457v1  

#### Abstract
Reinforcement Learning (RL) has shown promise for aligning Large Language Models (LLMs) to follow instructions with various constraints. Despite the encouraging results, RL improvement inevitably relies on sampling successful, high-quality responses; however, the initial model often struggles to gen...

---

### 7. [Physics-Informed Neural Networks for Device and Circuit Modeling: A Case Study of NeuroSPICE](https://arxiv.org/abs/2512.23624)

**Authors**: Chien-Ting Tung, Chenming Hu  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2512.23624v1  

#### Abstract
We present NeuroSPICE, a physics-informed neural network (PINN) framework for device and circuit simulation. Unlike conventional SPICE, which relies on time-discretized numerical solvers, NeuroSPICE leverages PINNs to solve circuit differential-algebraic equations (DAEs) by minimizing the residual o...

---

### 8. [SANet: A Semantic-aware Agentic AI Networking Framework for Cross-layer Optimization in 6G](https://arxiv.org/abs/2512.22579)

**Authors**: Yong Xiao, Xubo Li, Haoran Zhou, Yingyu Li, Yayu Gao, Guangming Shi, Ping Zhang, Marwan Krunz  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.22579v1  

#### Abstract
Agentic AI networking (AgentNet) is a novel AI-native networking paradigm in which a large number of specialized AI agents collaborate to perform autonomous decision-making, dynamic environmental adaptation, and complex missions. It has the potential to facilitate real-time network management and op...

---

### 9. [InSPO: Unlocking Intrinsic Self-Reflection for LLM Preference Optimization](https://arxiv.org/abs/2512.23126)

**Authors**: Yu Li, Tian Lan, Zhengling Qi  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2512.23126v1  

#### Abstract
Direct Preference Optimization (DPO) and its variants have become standard for aligning Large Language Models due to their simplicity and offline stability. However, we identify two fundamental limitations. First, the optimal policy depends on arbitrary modeling choices (scalarization function, refe...

---

### 10. [DarkPatterns-LLM: A Multi-Layer Benchmark for Detecting Manipulative and Harmful AI Behavior](https://arxiv.org/abs/2512.22470)

**Authors**: Sadia Asif, Israel Antonio Rosales Laguan, Haris Khan, Shumaila Asif, Muneeb Asif  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.22470v1  

#### Abstract
The proliferation of Large Language Models (LLMs) has intensified concerns about manipulative or deceptive behaviors that can undermine user autonomy, trust, and well-being. Existing safety benchmarks predominantly rely on coarse binary labels and fail to capture the nuanced psychological and social...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*DarkPatterns-LLM: A Multi-Layer Benchmark for Detecting Manipulative and Harmful AI Behavior*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **AI 安全基准**（如 TruthfulQA、SafetyBench、AdvBench）主要依赖**二元分类标签**（安全 vs 不安全），无法捕捉 LLM 输出中**微妙且具有心理操纵性**的行为（即“dark patterns”）。这些行为虽不直接表现为毒性或虚假信息，却通过利用认知偏差、情感脆弱性和权力不对称来削弱用户自主性、信任和福祉。

此外，随着《欧盟人工智能法案》（EU AI Act, 2024）将“操纵性行为”列为高风险，亟需更精细、可解释的安全评估工具。

### 提出的新方法与新思路
作者提出了 **DarkPatterns-LLM**，一个面向 LLM 操纵性行为检测的**多层级基准框架**，其核心是四层分析流水线：

- **Multi-Granular Detection (MGD)**：在细粒度层面识别八种心理操纵机制（如权威偏见、情感胁迫、稀缺性框架等），并定位文本中的操纵片段。
- **Multi-Scale Intent Analysis (MSIAN)**：建模操纵对不同利益相关者（个体、社区、机构、社会）的影响路径与差异。
- **Threat Harmonization Protocol (THP)**：估计操纵效应的长期传播潜力与持续时间，涵盖短期、中期和长期风险。
- **Deep Contextual Risk Alignment (DCRA)**：整合前三层输出，生成可解释的 **Harm Scorecard**，包含总体危害评分、影响分解、证据片段和缓解建议。

同时构建了一个包含 **401 个专家标注样本**的数据集，覆盖七类危害：
- Legal/Power Harm  
- Psychological Harm  
- Emotional Harm  
- Physical Harm  
- Autonomy Harm  
- Economic Harm  
- Societal Harm  

### 相比现有方法的优势
| 维度 | 现有基准（如 SafetyBench） | DarkPatterns-LLM |
|------|--------------------------|------------------|
| 标注方式 | 二元标签（safe/unsafe） | 多维度、细粒度机制标注 |
| 分析深度 | 表层内容判断 | 四层递进式语义-意图-传播-风险建模 |
| 可解释性 | 黑箱判断 | 提供注意力权重、关键片段、影响路径 |
| 风险建模 | 忽略时间动态 | 显式建模长期放大效应（Temporal Risk） |
| 应用价值 | 仅用于评估 | 支持审计、调试与干预设计 |

> ✅ **创新亮点**：首次实现从“是否危险”到“如何危险、对谁危险、何时危险”的系统化、结构化诊断。

---

## 2. 核心实验方法和设置

### 数据集
- **DarkPatterns-LLM 数据集**：共 401 条 instruction-response 对，每条包含：
  - `instruction`：用户输入
  - `rejected`：有害/操纵性响应
  - `accepted`：安全替代响应
  - `discussion`：专家解释为何有害
  - `category`：所属危害类别
- 数据来源：真实模型输出 + 专家构造场景 + 模拟对抗提示
- 分布均衡（各类占比 12.0%–17.2%），平均长度 187 tokens（SD=94）
- 公开地址：[GitHub](https://github.com/sadia-sigma-lab/Benchmark-dataset-for-dark-patterns-in-llms)

### 实验设置
- **测试模型**（6 个前沿 LLM）：
  - GPT-4 (OpenAI)
  - Claude 3.5 Sonnet (Anthropic)
  - Gemini 2.0 Flash (Google)
  - LLaMA-3-70B (Meta)
  - Mistral-Large-2 (Mistral AI)
  - Qwen-2.5-72B (Alibaba)
- **任务**：给定 `instruction` 和 `rejected` 响应，要求模型判断是否安全，并识别危害类型。
- **评估协议**：使用四层框架自动打分，计算以下四个新指标：

#### 评估指标（Metrics）
| 缩写 | 名称 | 含义 |
|------|------|------|
| **MRI** | Manipulation Resistance Index | 考虑心理机制重要性的检测准确率加权得分 |
| **CRS** | Contextual Robustness Score | 在不同上下文中表现的一致性（方差越小越好） |
| **SIAS** | Stakeholder Impact Assessment Score | 对各利益相关者的伤害预测能力 |
| **THDS** | Temporal Harm Dynamics Score | 对长期传播与累积风险的建模能力 |

- **基线方法**：
  - Binary Baseline：基于 SafetyBench 和 TruthfulQA 训练的二分类器（作为传统方法代表）

---

## 3. 主要实验结果和性能指标

### 层级性能对比（Table 2）
| Model | MRI ↑ | CRS ↑ | SIAS ↑ | THDS ↑ |
|-------|--------|--------|---------|---------|
| **Claude 3.5 Sonnet** | **89.7** | **87.3** | **82.6** | **76.4** |
| GPT-4 | 86.4 | 84.1 | 79.8 | 74.2 |
| Gemini 2.0 Flash | 81.3 | 79.6 | 75.4 | 71.8 |
| LLaMA-3-70B | 78.2 | 76.8 | 73.1 | 68.9 |
| Mistral-Large-2 | 74.6 | 73.2 | 70.3 | 66.5 |
| Qwen-2.5-72B | 71.8 | 70.4 | 68.7 | 62.8 |
| Binary Baseline | 65.2 | 62.1 | — | — |

> 🔍 **关键观察**：
> - 所有模型均显著优于二元基线（MRI 提升 >20 pts），验证了多层分析的有效性。
> - 闭源模型（Claude/GPT-4）整体领先，表明其可能接受了更强的安全训练。
> - 开源模型（LLaMA/Qwen）仍有差距，但具备竞争力。

### 按危害类别检测准确率（Table 3 平均值）
| 危害类型 | 平均准确率 |
|----------|------------|
| **Physical Harm** | **84.3%** |
| **Emotional Harm** | **82.6%** |
| Psychological Harm | 80.2% |
| Societal Harm | 77.9% |
| Legal/Power Harm | 76.8% |
| Economic Harm | 75.2% |
| **Autonomy Harm** | **71.4%** ⚠️最低 |

> ❗ **最严重盲区**：所有模型在 **Autonomy Harm** 上表现最差，说明当前 LLM 难以识别那些通过欺骗、紧迫感制造或默认选项诱导等方式**侵蚀用户自主决策权**的行为。

### 时间动态预测能力薄弱
- 所有模型的 **THDS 分数普遍偏低**（62.8–76.4），远低于 MRI。
- 表明当前 LLM 缺乏对“操纵如何随时间扩散、重复暴露后累积影响”的推理能力。
- 示例失败案例（Appendix D）显示模型常忽略长期心理依赖或社会信任崩塌的风险。

### 消融实验（隐含于分析中）
虽然未明确列出消融表，但通过逐层输出分析可得：
- 若仅使用 MGD（局部特征），会遗漏跨情境传播与长期影响；
- 若跳过 MSIAN，则无法区分个体情绪困扰与制度性信任危机；
- THP 的引入显著提升了对“病毒式误导”类内容的风险预判能力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **多层分析显著优于二元判断**：结构化、可解释的四层框架能揭示传统基准忽略的细微操纵模式，提升诊断精度。
2. ✅ **闭源模型安全性更强**：Claude 3.5 和 GPT-4 在各项指标上领先，反映其在安全对齐方面的投入优势。
3. ⚠️ **Autonomy Harm 是系统性盲点**：所有模型都难以识别损害用户代理权（agency）的策略，这在推荐系统、金融咨询等场景中尤为危险。
4. ⚠️ **Temporal Reasoning 能力不足**：模型普遍缺乏对“长期操纵后果”的建模能力，THDS 分数最低。
5. 📌 **Physical & Emotional Harm 更易检测**：因训练数据中对此类显性风险已有较强抑制，模型表现较好。

### 方法局限性
| 局限 | 说明 |
|------|------|
| 数据规模有限 | 仅 401 条样本，不足以支持大规模训练或统计推断 |
| 依赖专家标注 | 场景为人工构造，非真实世界事件，可能存在偏差 |
| 文化与语言局限 | 当前仅英文，基于西方伦理框架，跨文化普适性待验证 |
| 权重主观性 | THP 中的维度权重来自 Delphi 专家共识（Kendall’s W=0.74），仍具主观成分 |
| 侧重检测而非防御 | 当前为诊断工具，尚未集成至训练或运行时防护机制 |

### 未来工作方向
1. **扩展数据集规模与多样性**：纳入更多真实用户交互日志，增加多语言版本。
2. **开发基于诊断信号的干预机制**：将 MRI、THDS 等指标用于 RLHF 或 DPO 训练，强化对 autonomy 和 temporal harm 的敏感性。
3. **构建实时监控系统**：将 DCRA Scorecard 集成至部署管道，实现自动化风险预警。
4. **拓展至多模态与多智能体场景**：研究图像、语音中的 dark patterns，以及多个 AI 协同操纵的可能性。
5. **推动标准化采纳**：倡导将 DarkPatterns-LLM 作为 AI 安全审计的标准组件，尤其适用于高风险领域（医疗、金融、教育）。

---

> 💡 **总结一句话**：  
> *DarkPatterns-LLM* 建立了首个针对 LLM 操纵行为的**多维、可解释、动态化评估标准**，揭示了当前模型在保护用户自主性和预见长期风险方面的根本缺陷，为构建真正可信的 AI 系统提供了关键诊断工具。

</details>

---

### 11. [SAMP-HDRL: Segmented Allocation with Momentum-Adjusted Utility for Multi-agent Portfolio Management via Hierarchical Deep Reinforcement Learning](https://arxiv.org/abs/2512.22895)

**Authors**: Xiaotian Ren, Nuerxiati Abudurexiti, Zhengyong Jiang, Angelos Stefanidis, Hongbin Liu, Jionglong Su  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.22895v1  

#### Abstract
Portfolio optimization in non-stationary markets is challenging due to regime shifts, dynamic correlations, and the limited interpretability of deep reinforcement learning (DRL) policies. We propose a Segmented Allocation with Momentum-Adjusted Utility for Multi-agent Portfolio Management via Hierar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**非平稳市场环境下的投资组合优化挑战**，解决了以下三个关键问题：
- **静态或启发式资产分组**：传统聚类方法无法捕捉金融时间序列的动态特性，导致在市场结构突变时响应滞后。
- **聚类与优化过程脱节**：大多数方法将聚类作为预处理步骤，与后续的DRL优化缺乏端到端反馈，造成目标不一致和信息流断裂。
- **子集更新可扩展性差**：基于规则的动态子集选择机制难以适应奖励驱动的学习，且存在选择偏差。

### 提出的新方法：SAMP-HDRL
提出了一种名为 **SAMP-HDRL (Segmented Allocation with Momentum-Adjusted Utility for Multi-agent Portfolio Management via Hierarchical Deep Reinforcement Learning)** 的新型框架，其核心创新点如下：

#### （1）全局信号与局部决策的联合建模
- 首先通过**动态资产分组**（dynamic asset grouping）将市场划分为高质量和普通资产两组；
- 上层Agent提取跨资产相关性和市场整体动态的**全局表示**；
- 下层Agent在掩码约束下对各自分配的组内进行权重分配，实现**局部优化**；
- 这种“分组→全局建模→局部分配”的设计增强了模型表征能力，提升了适应性和稳定性。

#### （2）基于动态分类的可解释分层决策机制
- 利用**动态资产分类**构建分层决策流程，避免依赖规则或黑箱选择；
- 在统一学习框架中显式建模分组与分配过程，确保层级间信息一致性；
- 缓解了阶段式训练带来的不稳定性，提高了对结构性市场变化的适应能力和**可解释性**。

#### （3）创新的动量调整效用函数用于分段分配
- 将**动量调整**（momentum adjustment）和**反弹检测**（rebound detection）引入资本分配过程；
- 结合历史收益、风险资产和无风险资产，形成一个考虑市场状态的风险敏感效用函数；
- 增强了对持续趋势和突发市场转换的鲁棒性，提供了一种知识驱动的投资组合优化新机制。

#### （4）系统性优势
相比现有方法，SAMP-HDRL实现了：
- **端到端集成**：动态聚类直接嵌入DRL训练循环，而非独立预处理；
- **结构化市场约束**：通过掩码机制强制执行组内分配，提升策略合理性；
- **风险与回报平衡**：效用函数融合风险控制与趋势捕捉，避免过度追逐短期高收益；
- **可解释性强**：SHAP分析揭示了Agent间的互补行为模式。

---

## 2. 核心实验方法和设置

### 数据集
- 使用来自 **Yahoo Finance** 的 **Dow Jones Industrial Average (DJIA)** 成分股数据；
- 构建了三个独立的数据集，每个覆盖四年，前三年用于训练，最后一年用于回测；
- 因为Dow Inc.在2019年拆分，故样本包含29只股票；
- 调整后的收盘价用于构造价格矩阵，以消除分红和拆股影响。

| 回测编号 | 训练期 | 测试期 | 市场特征 |
|---------|--------|--------|--------|
| Backtest 1 | 2016/01/01 – 2019/01/01 | 2019/01/01 – 2020/01/01 | 稳定上涨市场 |
| Backtest 2 | 2017/01/01 – 2020/01/01 | 2020/01/01 – 2021/01/01 | 非平稳、高波动（含疫情冲击） |
| Backtest 3 | 2018/01/01 – 2021/01/01 | 2021/01/01 – 2022/01/01 | 振荡恢复市场 |

### 实验设置
- **聚类频率**：每75个交易日重新执行一次K-means聚类（约季度周期），输入特征为Sortino比率；
- **交易成本**：设定为0.1% per transaction；
- **动作空间**：连续控制，输出为各资产权重向量；
- **算法基础**：采用 **DDPG** 框架，结合Transformer架构作为上层Agent；
- **奖励函数**：基于对数收益，并加入自适应风险厌恶系数。

### 评估指标
- **盈利能力**：
  - **Return**（累计收益率）
- **风险调整后绩效**：
  - **Sharpe Ratio**
  - **Sortino Ratio**
  - **Omega Ratio**

### 基线方法对比
共比较了 **18种基线方法**，分为两类：

#### （1）传统策略（9种）
- CRP, UBAH, MO, UP, EG, PAMR, CWMR, CORN-K, CAPM

#### （2）DRL方法（9种）
- EIIE, FinRL, EST, SARL, II, PPO, PPN, TARN, DeepMPT, LSRE-CAAN, FTRL

其中FTRL是近期表现最强的DRL基线之一。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Backtest 3，最具代表性）
在2021年振荡市场环境下（Backtest 3），SAMP-HDRL取得全面领先：

| 指标 | SAMP-HDRL | 最优基线（LSRE-CAAN） | 提升幅度 |
|------|-----------|------------------------|----------|
| **Return** | 0.3938 | 0.2813 | **+40%** |
| **Sharpe Ratio** | 0.1101 | 0.1025 | **+7.4%** |
| **Sortino Ratio** | 0.1004 | 0.0892 | **+12.6%** |
| **Omega Ratio** | 1.3328 | 1.3121 | **+1.6%** |

> 注：原文强调“相比最强基线，至少提升5%”，此处指综合多个回测场景下的最小改进。

### 与基线方法的整体对比结果
- 在所有三个回测中，SAMP-HDRL均显著优于9个传统和9个DRL基线；
- 特别是在**高波动市场（Backtest 2）** 中优势最为明显：
  - Return 提升 **33%**
  - Sharpe Ratio 提升 **35%**
  - Sortino Ratio 提升 **37%**
  - Omega Ratio 提升 **6%**
- 即使面对强大的FTRL模型，在动荡市场中仍能实现大幅超越。

### 消融实验结果（Ablation Study）
移除关键模块后性能下降显著，验证各组件必要性：

| 移除模块 | Return下降（Backtest 2） | Sharpe下降 | Omega下降 |
|--------|--------------------------|------------|-----------|
| 上层Agent（w/o upper） | ~28% | ~25% | ~5% |
| 下层Agent（w/o lower） | ~25% | ~32% | ~5% |
| 动态聚类（w/o dc） | ~11% | ~13% | ~3% |
| 资本分配（w/o ca） | ~34% | ~32% | ~7% |

> 结论：**资本分配机制**和**上下层协调**对整体性能至关重要，尤其在高波动环境中。

此外，SHAP分析显示：
- 下层Agent 1（普通资产）保持广泛分散投资；
- 下层Agent 2（优质资产）集中关注核心驱动力资产；
- 形成“**多样化 + 集中式**”（diversified + concentrated）的互补决策模式，增强透明度与经济逻辑一致性。

---

## 4. 关键结论和发现

### 主要发现
1. **SAMP-HDRL在非平稳市场中具有卓越鲁棒性**：
   - 在疫情引发的剧烈波动（2020年）和震荡恢复期（2021年）中表现最优；
   - 显著优于纯收益导向型模型（如FTRL），因其内置风险控制机制更适应不确定性。

2. **分层结构与动态聚类的有效整合是成功关键**：
   - 上层Agent捕捉全局信号，下层Agent专注局部优化，二者协同提升决策质量；
   - 动态聚类每季度更新，有效跟踪市场结构演变，避免过时分组带来的风险错配。

3. **动量调整与反弹检测增强趋势应对能力**：
   - 引入动量强度参数和反弹标志位，防止在技术性反弹中误判趋势反转；
   - 实现了对真实反弹机会的选择性加仓，提高收益同时控制下行风险。

4. **可解释性显著提升**：
   - SHAP分析揭示了Agent之间的分工协作机制；
   - 决策过程不再是“黑箱”，而是具备经济直觉的一致性逻辑。

### 方法的局限性
1. **非严格端到端训练**：
   - 上下层Agent采用分阶段训练，未完全实现跨层级梯度传播，可能限制信息流动效率。

2. **未显式建模组间依赖关系**：
   - 当前框架聚焦组内优化，忽略了不同资产群之间的潜在联动效应。

3. **解释方法为事后分析**：
   - SHAP属于post-hoc解释工具，无法实现实时因果推理或在线监控。

4. **数据范围有限**：
   - 仅基于DJIA成分股，未纳入宏观经济、舆情等多模态信号，泛化能力有待验证。

### 未来工作方向
1. 探索更紧密的**端到端优化范式**，实现上下层Agent共同适应；
2. 引入**Graph Neural Networks**或**correlation-aware attention**，显式建模跨集群依赖；
3. 发展**实时可解释框架**，支持因果推断与反事实验证；
4. 扩展至**多模态输入**（宏观指标、文本情绪、跨市场信号）；
5. 提升**可扩展性**，适用于更大资产池和更高频交易场景；
6. 在**真实交易约束**（流动性、滑点、订单簿深度）下进行实证检验。

---

> ✅ 总结：SAMP-HDRL通过将**动态资产分组**、**分层Agent协调**和**效用驱动的资本分配**有机结合，构建了一个兼具高性能、鲁棒性和可解释性的投资组合管理框架，在复杂非平稳市场中展现出显著优势，为DRL在金融领域的应用提供了新的结构化范式。

</details>

---

### 12. [Problems With Large Language Models for Learner Modelling: Why LLMs Alone Fall Short for Responsible Tutoring in K--12 Education](https://arxiv.org/abs/2512.23036)

**Authors**: Danial Hooshyar, Yeongwook Yang, Gustav \v{S}\'i\v{r}, Tommi K\"arkk\"ainen, Raija H\"am\"al\"ainen, Mutlu Cukurova, Roger Azevedo  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.23036v1  

#### Abstract
The rapid rise of large language model (LLM)-based tutors in K--12 education has fostered a misconception that generative models can replace traditional learner modelling for adaptive instruction. This is especially problematic in K--12 settings, which the EU AI Act classifies as high-risk domain re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Problems With Large Language Models for Learner Modelling: Why LLMs Alone Fall Short for Responsible Tutoring in K–12 Education*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对当前在 K–12 教育中日益流行的**大语言模型（LLM）作为智能辅导系统核心引擎**的趋势，提出了一个关键质疑：  
**LLM 是否能够替代传统的 learner modelling（学习者建模）方法，实现负责任的、可靠的自适应教学？**

作者指出，尽管 LLM 在生成自然语言反馈方面表现出色，但由于其缺乏对学习过程的显式建模能力，在**实时、准确、稳定地追踪学生知识状态演变**方面存在根本缺陷。特别是在被欧盟 AI 法案列为“高风险”应用的 K–12 教育场景下，这种不可靠性可能带来严重的教育伦理和实践风险。

### 提出了什么新方法或新思路
论文并未提出一种全新的 learner modelling 架构，而是通过**实证研究揭示了 LLM 在 learner modelling 任务上的结构性不足**，并倡导一种**混合式（hybrid）人机智能框架**作为解决方案。

其核心思想是：
- **LLM 不应独立承担 learner modelling 职能**；
- 应将 LLM 与经过验证的序列化 learner modelling 方法（如 DKT）结合，形成“**以 learner model 为认知核心，LLM 为表达层**”的协同架构；
- 这种 hybrid human-AI intelligence 可兼顾准确性、可解释性和生成灵活性，从而实现真正负责任的 AI 辅导。

### 相比现有方法的优势
- **批判性视角**：不同于多数研究强调 LLM 的潜力，本文系统性地从 learner modelling 的角度揭示其局限，填补了“责任型 AI”在教育落地中的设计空白。
- **实证驱动**：首次在标准知识追踪数据集上，直接比较 fine-tuned LLM 与 DKT 在 next-step prediction 和 temporal coherence 上的表现，提供量化证据。
- **提出可操作路径**：明确建议采用 retrieval-augmented generation（RAG）、神经符号计算（neural-symbolic AI）等混合架构来整合 LLM 与 learner model。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **ASSISTments 2009–2010 non skill-builder dataset**
  - 包含 603,287 条学生-题目交互记录
  - 来源于广泛使用的 K–12 数学在线辅导平台 ASSISTments
  - 字段包括：`user_id`, `problem_id`, `correct`, `hint_count`, `skill_id`, `skill_name`, `order_id` 等
  - 预处理后保留单技能交互，构建时间有序的学生行为序列 `(skill, quiz, correctness)`

### 实验设置和评估指标

#### 模型对比
| 模型 | 类型 | 描述 |
|------|------|------|
| **DKT** | Deep Knowledge Tracing | 基于 GRU 的序列模型，输入为 2K 编码的交互序列，输出各技能掌握概率 |
| **Llama 3 8B (zero-shot)** | LLM | 本地部署，不进行任何训练，仅通过 prompt 推理预测下一题正确性 |
| **Llama 3 8B (fine-tuned)** | LLM | 使用 LoRA 进行参数高效微调，目标同样是预测下一题正确性 |

#### 评估指标
- **全局性能指标**：
  - AUC（ROC 曲线下面积）
  - Accuracy, Precision, Recall, F1-score（阈值 0.5）
- **细粒度时序分析**：
  - **Early/Mid/Late Sequence Errors**：按最优 ROC 阈值划分不同阶段错误率
  - **Stable vs Switching 学生分组**：基于答题序列波动性分类
- **时间一致性（Temporal Coherence）指标**：
  - **Volatility**：同一技能 mastery 概率连续变化的平均绝对值（越低越稳定）
  - **Inconsistency Rate**：mastery 更新方向与实际表现相反的比例（如答对后 mastery 下降）
- **计算效率**：
  - 训练/推理时间
  - 所需硬件资源（GPU 内存等）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| Model | AUC | Accuracy | F1-score (Low Perf) | F1-score (High Perf) |
|-------|-----|----------|----------------------|------------------------|
| **DKT** | **0.83** | **75%** | **68%** | **80%** |
| Llama 3 8B (zero-shot) | 0.69 | 64% | 31% | 76% |
| Llama 3 8B (fine-tuned) | 0.77 | 72% | 65% | 76% |

> ✅ **DKT 显著优于所有 LLM 变体**，即使 fine-tuned LLM 提升了约 8% AUC，仍落后 DKT 约 6%

### 与基线方法的对比结果

#### 时间阶段错误率（Early/Mid/Late Errors）

| Model | Stable – Early | Stable – Late | Switching – Early | Switching – Late |
|-------|---------------|--------------|--------------------|-------------------|
| **DKT** | **0.2975** | **0.1217** | **0.3118** | **0.2742** |
| Zero-shot LLM | 0.3853 | 0.2487 | 0.4054 | 0.3841 |
| Fine-tuned LLM | 0.3563 | 0.2501 | 0.3309 | 0.3430 |

> 🔺 **DKT 在早期阶段错误最低**，这对及时干预至关重要；而 fine-tuned LLM 虽有改进，但在 switching 学生上的 late-stage 错误反而上升。

#### 时间一致性指标

| Model | Volatility | Inconsistency Rate |
|-------|------------|---------------------|
| **DKT** | **0.1075** | **0.4061** |
| Zero-shot LLM | 0.1157 | 0.5012 |
| Fine-tuned LLM | **0.2945** | 0.4525 |

> ⚠️ 尽管 fine-tuning 减少了 inconsistency，但带来了更高的 volatility（剧烈波动），说明其更新不稳定且方向混乱。

### 多技能 mastery 轨迹可视化分析
- **DKT**：产生平滑、渐进的知识掌握曲线，符合真实学习规律。
- **Fine-tuned LLM**：出现频繁跳跃和反向更新（如连续答对后 mastery 反而下降），轨迹不连贯，难以支持可靠的教学决策。

### 计算效率对比
| Model | Training Time | Inference Time | Hardware Requirement |
|-------|---------------|----------------|------------------------|
| **DKT** | ~50 秒 | ~31 秒 | Colab T4 GPU (16GB) |
| Zero-shot LLM | — | ~0.45 小时 | Dual H100 (80GB) |
| Fine-tuned LLM | **~198 小时** | ~0.49 小时 | Dual H100 (80GB) |

> 💡 DKT 在极低资源下即可完成训练，而 fine-tuned LLM 需近 **198 小时高算力训练**，性价比极低。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **LLM 单独无法胜任 learner modelling 任务**：
   - 即使经过领域数据 fine-tuning，LLM 在 next-step prediction 上仍显著落后于 DKT；
   - 其 mastery 更新缺乏 temporal coherence，常出现方向错误或剧烈波动，违背基本教学逻辑。

2. **fine-tuning 无法弥补结构性缺陷**：
   - 微调虽提升性能，但代价高昂且无法达到 DKT 的稳定性；
   - LLM 的本质仍是基于文本模式的概率生成器，而非面向学习动态的建模工具。

3. **LLM 与 learner model 角色应分离**：
   - **learner model（如 DKT）负责精准评估知识状态**；
   - **LLM 负责基于该状态生成个性化反馈、解释或练习题**；
   - 二者协同才能实现既可靠又灵活的智能辅导。

4. **责任型 AI 必须从设计层面嵌入**：
   - “负责任使用”不能依赖事后提示工程或教师审核；
   - 必须在系统架构中内置 learner modelling，确保决策可追溯、可解释、可问责。

### 方法的局限性
- 实验仅基于 **Llama 3 8B** 一种 LLM，未涵盖更大规模模型（如 GPT-4o）或其他架构；
- 微调采用 LoRA，非全参数微调，可能限制 LLM 潜力发挥；
- 数据集为单一数学学科，结论在跨学科或多模态场景中的普适性有待验证；
- 未涉及 affective 或 metacognitive states 的建模，focus 仅限认知层面。

### 未来工作方向
1. 开发更多 **hybrid LLM-KT 框架**，例如：
   - 将 DKT 输出作为 RAG 的检索信号，引导 LLM 生成 grounded feedback；
   - 设计 plug-in instruction 或 sequence adapter 实现模型间通信。
2. 探索 **neural-symbolic AI** 在教育中的应用，将教学规则、概念图谱注入模型。
3. 扩展至多模态 learner modelling（结合眼动、语音、表情等）。
4. 构建面向 educator 的可解释 interface，让教师理解并干预 AI 决策过程。
5. 在真实课堂环境中进行 longitudinal evaluation，检验 hybrid 系统的实际教学效果。

---

> 📌 **一句话总结**：  
> **LLMs 擅长“说话”，但不懂“学习”。真正的责任型 AI 教育系统，必须让 DKT 这样的 learner model 当“大脑”，LLM 当“嘴巴”。**

</details>

---

### 13. [The Gaining Paths to Investment Success: Information-Driven LLM Graph Reasoning for Venture Capital Prediction](https://arxiv.org/abs/2512.23489)

**Authors**: Haoyu Pei, Zhongyang Liu, Xiangyi Xiao, Xiaocong Du, Haipeng Zhang, Kunpeng Zhang, Suting Hong  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.23489v1  

#### Abstract
Most venture capital (VC) investments fail, while a few deliver outsized returns. Accurately predicting startup success requires synthesizing complex relational evidence, including company disclosures, investor track records, and investment network structures, through explicit reasoning to form cohe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：The Gaining Paths to Investment Success: Information-Driven LLM Graph Reasoning for Venture Capital Prediction

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 VC（Venture Capital）预测方法面临以下挑战：
- **缺乏显式推理能力**：传统机器学习模型依赖孤立特征，忽略公司与投资者之间的复杂关系；图神经网络（GNNs）虽能捕捉高阶依赖，但为“黑箱”模型，无法提供可解释的决策依据。
- **LLMs 与图结构的模态不匹配**：大语言模型（LLMs）擅长推理，但其架构针对文本序列优化，难以直接处理图结构数据。
- **路径爆炸与异构证据融合难题**：在投资网络中进行多跳检索时，候选路径数量呈指数增长，且不同来源的信息（如公司披露、投资者履历、图路径）需动态加权融合。

此外，VC 预测属于 **off-graph prediction** 任务——目标变量（初创企业是否成功）不在图内，而图仅作为外部证据源，这与主流的 in-graph QA 任务有本质区别。

### 提出的新方法：MIRAGE-VC
作者提出 **MIRAGE-VC**，一个基于多视角检索增强生成（multi-perspective RAG）的框架，用于 VC 成功预测。其核心创新包括：

#### （1）信息增益驱动的路径选择器（Information-Gain-Driven Path Retriever）
- 将图路径选择建模为**逐步节点扩展问题**，每一步选择能最大化预测准确率提升的邻居节点。
- 使用冻结的 LLM 作为“oracle”计算每个候选扩展带来的 **task-specific information gain**（基于 cross-entropy 减少和置信度变化），训练轻量级 selector 模型来近似该信号。
- 最终将庞大的投资网络压缩为少数几条高价值的 **investment chains**，供 LLM 进行 chain-of-thought 推理。

#### （2）可学习门控的多智能体分析架构（Learnable Gating + Multi-Agent Architecture）
- 设计三个专用 LLM agent 分别分析：
  - **Peer-Company Analyst (PC)**：基于相似公司的历史表现
  - **Investor Profile Analyst (IP)**：基于领投人的背景与过往业绩
  - **Investment Chain Analyst (IC)**：基于图路径中的结构信号
- 引入 **gating network**，根据目标公司的属性（行业、阶段等）动态学习三个 agent 输出的权重，实现自适应融合。
- 最后由 **Manager Agent** 综合所有信息，输出最终预测与自然语言解释。

### 相比现有方法的优势
| 方法类型 | 局限性 | MIRAGE-VC 的优势 |
|--------|------|----------------|
| 传统 ML | 忽略关系结构 | 融合图结构与文本信息 |
| GNNs | 黑箱、不可解释 | 显式路径推理，支持可读 rationale |
| 标准 RAG | 忽视图结构与多跳依赖 | 显式建模图路径并进行价值筛选 |
| 图-LLM 方法（如 GNN-RAG） | 主要面向 in-graph QA | 支持 off-graph 外部目标预测 |
| 单一 agent LLM | 容易冗余或偏倚 | 多 agent 分工 + 动态加权 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **PitchBook Global VC dataset**（2005–2023），包含：
  - 263,729 家初创公司
  - 1,014,157 名个人（创业者/投资人）
  - 投资记录、融资轮次、金额、团队构成、地理位置、关键词标签等
- 构建时间戳异构图 $ G = (V, E) $，其中节点为公司（company）和投资人（investor），边表示投资事件。
- **标签定义**：若公司在种子轮后一年内完成 Series A 融资，则标记为 `Success`（y=1），否则为 `Failure`（y=0）。

### 实验设置
- **训练/验证/测试划分**：随机采样 2,000（path selector） 和 11,000（gating network）家公司，按 70:15:15 划分，保持类别平衡。
- **最终评估集**：选取 2021年10月 至 2023年11月 间首次融资的 2,510 家公司，确保其不在 LLM 预训练语料中（避免数据泄露）。
- **backbone LLM**：使用 **GPT-3.5 Turbo**（知识截止时间为 2021年9月），保证公平性。
- 所有文本编码使用 **Sentence-BERT**。

### 评估指标
- **Precision@K (P@K)**：前 K 名推荐中成功的比例，反映实际投资场景下的实用性。
- **Average Precision@K (AP@K)**：对每月 cohort 计算 P@K 后取平均，衡量时间稳定性。
- **F1 Score, Precision, Recall, AUC-ROC, AUC-PR**：标准分类指标。
- 所有结果为五次独立运行的平均值。

### 基线方法对比
| 类别 | 方法 | 简介 |
|-----|------|------|
| GNN-based | SHGMNN, GST | 基于 meta-path 或自注意力的图神经网络 |
| Embedding-based | BERT Fusion | BERT 编码公司描述 + 结构特征 |
| RAG-based | Standard RAG, GNN-RAG | 检索相关文本或图路径后送入 LLM |
| LLM-driven VC predictor | SSFF | 多智能体 + RAG + 随机森林增强 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）

| 方法 | AP@5 ↑ | AP@10 ↑ | AP@20 ↑ | Precision ↑ | Recall ↑ | F1 ↑ |
|------|--------|---------|---------|-------------|----------|-------|
| SHGMNN | 25.41 | 24.56 | 26.22 | 20.65 | 82.37 | 32.97 |
| GST | 26.71 | 25.71 | 27.14 | 21.75 | 83.54 | 34.51 |
| BERT Fusion | 24.67 | 26.67 | 25.33 | 23.63 | 24.95 | 24.27 |
| Standard RAG | 24.43 | 24.12 | 25.23 | 23.12 | 60.34 | 33.43 |
| SSFF | 28.23 | 30.02 | 28.42 | 23.23 | 69.41 | 34.81 |
| GNN-RAG | 29.42 | 27.53 | 27.04 | 22.81 | 71.10 | 34.54 |
| **Ours (MIRAGE-VC)** | **34.29** | **32.14** | **29.21** | **24.32** | **73.44** | **36.54** |

> ✅ **相比最强基线（SSFF/GNN-RAG）**：
> - **AP@5 提升 +16.6%**
> - **F1 提升 +5.0%**
> - **Precision 提升 +2.9%**

### 消融实验结果（Table 2）

| 移除模块 | Precision | F1 |
|--------|-----------|-----|
| Full Model | 24.32 | 36.54 |
| w/o Graph Retrieval | 23.01 | 34.06 |
| w/o Path Selector (all 3-hop) | 22.72 | 33.29 |
| w/o Path Selector (random) | 23.24 | 34.76 |
| w/o Similar Company | 23.45 | 35.54 |
| w/o Investor Analysis | 23.32 | 35.43 |
| w/o Multi-agent Fusion | 22.97 | 35.13 |
| w/o Gating Network | 24.05 | 35.94 |

> 🔍 发现：
> - 图路径检索是关键组件，移除后 F1 下降 2.5%
> - 随机或全量路径效果差，说明 **information-gain selector 有效过滤噪声**
> - 三种信息流互补，缺一不可
> - 可学习 gating 比固定权重更优（+0.6% F1）

### 其他重要结果
- **AUC-PR 达到 0.354**，比最强基线（SSFF）高 3.8%
- **AUC-ROC 达到 0.591**，优于 GNN-RAG（0.574）
- 正确预测样本的平均路径长度为 **4.44 hops**，错误样本仅为 **3.31 hops**，表明更深的结构上下文有助于推理（Appendix A.5）

---

## 4. 关键结论和发现

### 主要发现
1. **显式图路径推理显著提升 VC 预测性能**：通过信息增益驱动的选择机制，MIRAGE-VC 成功从大规模投资网络中提取出具有高判别力的投资链（investment chains），使 LLM 能够进行 step-by-step reasoning。
2. **异构证据需要动态加权融合**：不同类型的初创企业（如硬件 vs 软件）对 peer company、investor profile 和 graph path 的依赖程度不同，learnable gating 机制能自动适配。
3. **off-graph prediction 需要新的图-LLM 范式**：不同于 in-graph QA，VC 预测要求图作为外部证据源，路径选择应以“边际效用”为导向，而非终点匹配。
4. **可解释性与性能兼顾**：系统不仅输出预测结果，还生成基于多源证据的自然语言 rationale，符合真实 VC 决策流程。

### 方法的局限性
1. **单一批量私有数据集依赖**：完全基于 **PitchBook**，无法公开复现。公共替代品 Crunchbase 数据过时（截至2013），存在预训练污染风险。
2. **局部监督目标（Myopic Supervision）**：路径选择采用贪心策略（逐跳最大化信息增益），可能错过全局最优子图。
3. **计算开销较高**：尽管推理延迟可控（单次约 7.8 秒），但训练过程涉及大量 LLM 查询（~480M tokens）和 GPU 时间（约 10 GPU-hours）。

### 未来工作方向
- 探索 **look-ahead scoring** 或 **sequence-level RL** 来优化路径选择，超越贪心策略。
- 构建 **公开基准数据集**，结合旧版 Crunchbase 与新爬取数据，推动领域可重复研究。
- 将本范式推广至其他 **off-graph prediction 任务**，如推荐系统（user-item affinity from interaction graphs）、信用风险评估（default prediction from transaction networks）等。

--- 

> 📌 **一句话总结**：  
> MIRAGE-VC 提出了一种面向 **off-graph VC 预测** 的新型图-LLM 融合框架，通过 **信息增益驱动的路径选择** 和 **可学习门控的多智能体推理**，实现了更高精度与更强可解释性的投资成功预测，在多个指标上显著超越现有方法。

</details>

---

### 14. [Divergent-Convergent Thinking in Large Language Models for Creative Problem Generation](https://arxiv.org/abs/2512.23601)

**Authors**: Manh Hung Nguyen, Adish Singla  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.23601v1  

#### Abstract
Large language models (LLMs) have significant potential for generating educational questions and problems, enabling educators to create large-scale learning materials. However, LLMs are fundamentally limited by the ``Artificial Hivemind'' effect, where they generate similar responses within the same...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Divergent-Convergent Thinking in Large Language Models for Creative Problem Generation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在生成教育类问题（如编程题、数学题等）方面展现出巨大潜力，但其输出存在“**Artificial Hivemind**”效应——即同一模型内部重复、不同模型之间同质化严重。这导致生成的问题缺乏多样性与创造性，限制了其在促进多样化思维方面的应用价值。

### 提出的新方法：CREATIVEDC
作者提出 **CREATIVEDC**，一种基于两阶段推理的提示方法（two-phase prompting method），灵感来源于人类创造力理论：
- **Wallas 的创造力四阶段理论**（准备、酝酿、启发、验证）
- **Guilford 的发散-收敛思维框架**（divergent-convergent thinking）

该方法将问题生成过程显式分解为两个阶段：
1. **Divergent Thinking Phase（发散思维阶段）**  
   忽略任务约束（如必须使用 `Lists` 编程），仅围绕主题自由探索非常规、奇特、多样化的创意点子。
2. **Convergent Thinking Phase（收敛思维阶段）**  
   从发散阶段产生的想法中选择一个，将其与具体的技术要求（如编程概念）结合，构造出符合规范且具创造性的最终问题。

通过解耦“创意探索”与“约束满足”，CREATIVEDC 鼓励 LLM 在早期广泛探索语义空间，避免过早收敛到常见模式。

### 相比现有方法的优势
| 方法 | 局限性 | CREATIVEDC 的改进 |
|------|--------|------------------|
| 高 Temperature 采样 | 仅增加表面多样性，不提升原创性，甚至降低质量 | 显式引导深层语义探索 |
| Persona Simulation | 引入外部视角，但未改变推理结构 | 可与 persona 结合，进一步增强多样性 |
| Chain-of-Thought (CoT) | 虽有推理链，但仍直接朝目标收敛 | 分离探索与精炼阶段，支持更广的创意搜索 |

> ✅ **核心创新**：首次将 divergent-convergent 思维框架应用于自动化创意问题生成任务，在推理层面重构 LLM 的生成路径。

---

## 2. 核心实验方法和设置

### 数据集与上下文设计
- 使用来自先前研究 [6,22–24] 的情境设定（context），共 4 个主题（Themes）和 5 个编程概念（Concepts）：
  - **Themes**: `"Cooking"`, `"Science Fiction"`, `"Superheroes"`, `"Board Games"`
  - **Concepts**: `"Variables"`, `"Selection Statements"`, `"Loops"`, `"Lists"`, `"Strings"`
- 组合成 **20 种唯一 context**，每个 context 下生成 K=100 个问题。

### 实验设置
- **模型**：
  - 生成模型：`Qwen3-235B-A22B-Instruct-2507`（MoE 架构，开源）
  - 温度设置：`temperature = 1.0`
  - 嵌入模型：`Qwen/Qwen3-Embedding-0.6B`（用于语义相似度计算）
  - 判断模型（LLM-as-a-judge）：`Gemini 2.5 Flash-Lite`（greedy decoding, temp=0）

### 评估指标（三维度评估创造力）
| 维度 | 指标 | 描述 |
|------|------|------|
| **Diversity（多样性）** | LexDiv（词法多样性）<br>SemDiv（语义多样性） | 衡量一组问题之间的差异程度 |
| **Novelty（新颖性）** | LexNov（词法新颖性）<br>SemNov（语义新颖性） | 对比其他方法生成的问题池，衡量“与众不同”的程度 |
| **Utility（实用性）** | 有效性（Validity）<br>相关性（Context Relevance）<br>可理解性（Comprehensibility） | 三项均为二值判断，综合得分为 0 或 1；最终报告有效问题占比 |

> 📌 特别地，**Novelty 的参考语料库 R 是由所有其他方法在同一 context 下生成的问题构成**，形成一个语义密集、极具挑战性的对比基准。

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **BASE** | 来自 [6] 的标准上下文化提示 |
| **CoT** | BASE + “Think step by step” |
| **CREATIVEDC** | BASE + 显式的 divergent-convergent 推理指令 |
| 所有方法均测试是否加入 **Persona Simulation**（来自 Persona Hub 数据集随机采样）的影响 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（表 1 & 图 2）

#### （1）多样性与新颖性显著提升
| 方法 | LexDiv | SemDiv | LexNov | SemNov | Utility (%) |
|------|--------|--------|--------|--------|-------------|
| BASE | 0.74±0.01 | 0.46±0.01 | 0.62±0.01 | 0.20±0.01 | 92.95±0.83 |
| CoT | 0.75±0.01 | 0.46±0.01 | 0.66±0.02 | 0.18±0.01 | 91.35±1.24 |
| **CREATIVEDC** | **0.81±0.00** | **0.54±0.01** | **0.73±0.01** | **0.30±0.01** | **90.85±0.88** |

> ✅ 所有多样性和新颖性指标均显著优于基线（p < 0.01，Wilcoxon Signed-Rank Test）
>
> 🔺 **语义新颖性（SemNov）提升最大**：相比 CoT 提升 **63.5%**

#### （2）Vendi Score：有效独特问题数量
- **Vendi Score** 衡量一组问题中“有效不同的”问题数（范围 1 ~ K）
- 图 3 显示随 K 增加，CREATIVEDC 的增长速度更快：

| K=10 | CREATIVEDC 比 CoT 高 **24.0%**  
| K=100 | CREATIVEDC 比 CoT 高 **72.0%**

> 💡 表明 CREATIVEDC 不仅更多样，而且**扩展性更强**，适合大规模生成场景。

#### （3）加入 Persona 后效果
- 所有方法在加入 persona 后多样性略有提升；
- CREATIVEDC 仍保持领先优势：
  - 语义多样性高 **8.5%**
  - 语义新颖性高 **32.9%**
- Utility 小幅下降（约 1~2%），但在可接受范围内。

#### （4）消融分析（图 4）：不同 context 下的表现
- **主题影响**：
  - `"Cooking"`：Utility 最高（0.97），但多样性最低 → 熟悉主题利于质量，抑制创意
  - `"Science Fiction"` 和 `"Superheroes"`：多样性与新颖性最高 → 更开放的主题激发更多创意
- **编程概念复杂度影响**：
  - 简单概念（如 `"Variables"`）→ 更高 Novelty
  - 复杂概念（如 `"Loops"`）→ 更低 Novelty，可能因约束更强限制了探索空间

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **CREATIVEDC 显著提升了 LLM 生成问题的多样性与新颖性**，同时保持了高水平的实用性（Utility）。
2. ✅ **显式分离“发散探索”与“收敛实现”阶段** 是突破“Artificial Hivemind”效应的关键机制。
3. ✅ CREATIVEDC 支持**规模化生成**，随着样本量增加，其相对于基线的优势持续扩大（Vendi Score 快速上升）。
4. ✅ 更具想象力的主题（如 Superheroes）更能发挥该方法的潜力。

### 方法的局限性
1. 当前仅在单一先进模型（Qwen3-235B）上验证，**泛化性有待跨架构/尺寸模型检验**。
2. 评估依赖自动指标（尤其是 Utility），**缺乏人类对创造力感知的真实反馈**。
3. 应用目前局限于编程问题生成，尚未拓展至其他创意领域（如写作、艺术设计等）。

### 未来工作方向
1. 开展 **human study**，评估生成问题的实际教学价值与创造性感知。
2. 将 CREATIVEDC 扩展到其他创意任务，如：
   - 故事创作（story writing）
   - 诗歌生成（poetry generation）
   - UI/UX 设计提案
3. 探索如何动态调整 divergent/convergent 阶段的深度与迭代次数以优化效率。
4. 结合 fine-tuning 或 RLHF 进一步强化 divergent 探索能力。

---

> 🧠 **一句话总结**：  
> CREATIVEDC 通过模仿人类创造力的认知流程，成功引导 LLM 跳出“集体思维陷阱”，实现了高质量、高多样性、高新颖性的创意问题生成，为 AI 辅助教育内容创作提供了新的范式。

</details>

---

### 15. [Web World Models](https://arxiv.org/abs/2512.23676)

**Authors**: Jichen Feng, Yifan Zhang, Chenggong Zhang, Yifu Lu, Shilong Liu, Mengdi Wang  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2512.23676v1  

#### Abstract
Language agents increasingly require persistent worlds in which they can act, remember, and learn. Existing approaches sit at two extremes: conventional web frameworks provide reliable but fixed contexts backed by databases, while fully generative world models aim for unlimited environments at the e...

---

### 16. [Monadic Context Engineering](https://arxiv.org/abs/2512.22431)

**Authors**: Yifan Zhang, Mengdi Wang  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.22431v1  

#### Abstract
The proliferation of Large Language Models (LLMs) has catalyzed a shift towards autonomous agents capable of complex reasoning and tool use. However, current agent architectures are frequently constructed using imperative, ad hoc patterns. This results in brittle systems plagued by difficulties in s...

---

### 17. [Multi-AI Agent Framework Reveals the "Oxide Gatekeeper" in Aluminum Nanoparticle Oxidation](https://arxiv.org/abs/2512.22529)

**Authors**: Yiming Lu, Tingyu Lu, Di Zhang, Lili Ye, Hao Li  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.22529v1  

#### Abstract
Aluminum nanoparticles (ANPs) are among the most energy-dense solid fuels, yet the atomic mechanisms governing their transition from passivated particles to explosive reactants remain elusive. This stems from a fundamental computational bottleneck: ab initio methods offer quantum accuracy but are re...

---

### 18. [Multimodal Fact-Checking: An Agent-based Approach](https://arxiv.org/abs/2512.22933)

**Authors**: Danni Xu, Shaojing Fan, Xuanang Cheng, Mohan Kankanhalli  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.22933v1  

#### Abstract
The rapid spread of multimodal misinformation poses a growing challenge for automated fact-checking systems. Existing approaches, including large vision language models (LVLMs) and deep multimodal fusion methods, often fall short due to limited reasoning and shallow evidence utilization. A key bottl...

---

### 19. [TCEval: Using Thermal Comfort to Assess Cognitive and Perceptual Abilities of AI](https://arxiv.org/abs/2512.23217)

**Authors**: Jingming Li  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2512.23217v1  

#### Abstract
A critical gap exists in LLM task-specific benchmarks. Thermal comfort, a sophisticated interplay of environmental factors and personal perceptions involving sensory integration and adaptive decision-making, serves as an ideal paradigm for evaluating real-world cognitive capabilities of AI systems. ...

---

### 20. [With Great Capabilities Come Great Responsibilities: Introducing the Agentic Risk & Capability Framework for Governing Agentic AI Systems](https://arxiv.org/abs/2512.22211)

**Authors**: Shaun Khoo, Jessica Foo, Roy Ka-Wei Lee  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.22211v1  

#### Abstract
Agentic AI systems present both significant opportunities and novel risks due to their capacity for autonomous action, encompassing tasks such as code execution, internet interaction, and file modification. This poses considerable challenges for effective organizational governance, particularly in c...

---

### 21. [Lightweight Inference-Time Personalization for Frozen Knowledge Graph Embeddings](https://arxiv.org/abs/2512.22398)

**Authors**: Ozan Oguztuzun, Cerag Oguztuzun  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2512.22398v1  

#### Abstract
Foundation models for knowledge graphs (KGs) achieve strong cohort-level performance in link prediction, yet fail to capture individual user preferences; a key disconnect between general relational reasoning and personalized ranking. We propose GatedBias, a lightweight inference-time personalization...

---

### 22. [Logic Sketch Prompting (LSP): A Deterministic and Interpretable Prompting Method](https://arxiv.org/abs/2512.22258)

**Authors**: Satvik Tripathi  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.22258v1  

#### Abstract
Large language models (LLMs) excel at natural language reasoning but remain unreliable on tasks requiring strict rule adherence, determinism, and auditability. Logic Sketch Prompting (LSP) is a lightweight prompting framework that introduces typed variables, deterministic condition evaluators, and a...

---

### 23. [The Reward Model Selection Crisis in Personalized Alignment](https://arxiv.org/abs/2512.23067)

**Authors**: Fady Rezk, Yuangang Pan, Chuan-Sheng Foo, Xun Xu, Nancy Chen, Henry Gouk, Timothy Hospedales  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.23067v1  

#### Abstract
Personalized alignment from preference data has focused primarily on improving reward model (RM) accuracy, with the implicit assumption that better preference ranking translates to better personalized behavior. However, in deployment, computational constraints necessitate inference-time adaptation v...

---

### 24. [MindWatcher: Toward Smarter Multimodal Tool-Integrated Reasoning](https://arxiv.org/abs/2512.23412)

**Authors**: Jiawei Chen, Xintian Shen, Lihao Zheng, Zhenwei Shao, Hongyuan Zhang, Pengfei Yu, Xudong Rao, Ning Mao, Xiaobo Liu, Lian Wen, Chaoqun Du, Feng Gu, Wei He, Qizhen Li, Shanshan Li, Zide Liu, Jing Luo, Lifu Mu, Xuhao Pan, Chang Ren, Haoyi Sun, Qian Wang, Wei Wang, Hongfu Yang, Jiqing Zhan, Chunpeng Zhou, Zheng Zhou, Hao Ma, Tao Wei, Pan Zhou, Wei Chen  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2512.23412v1  

#### Abstract
Traditional workflow-based agents exhibit limited intelligence when addressing real-world problems requiring tool invocation. Tool-integrated reasoning (TIR) agents capable of autonomous reasoning and tool invocation are rapidly emerging as a powerful approach for complex decision-making tasks invol...

---

### 25. [LLM Agents as VC investors: Predicting Startup Success via RolePlay-Based Collective Simulation](https://arxiv.org/abs/2512.22608)

**Authors**: Zhongyang Liu, Haoyu Pei, Xiangyi Xiao, Xiaocong Du, Yihui Li, Suting Hong, Kunpeng Zhang, Haipeng Zhang  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2512.22608v1  

#### Abstract
Due to the high value and high failure rate of startups, predicting their success has become a critical challenge across interdisciplinary research. Existing approaches typically model success prediction from the perspective of a single decision-maker, overlooking the collective dynamics of investor...

---

### 26. [From Model Choice to Model Belief: Establishing a New Measure for LLM-Based Research](https://arxiv.org/abs/2512.23184)

**Authors**: Hongshen Sun, Juanjuan Zhang  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2512.23184v1  

#### Abstract
Large language models (LLMs) are increasingly used to simulate human behavior, but common practices to use LLM-generated data are inefficient. Treating an LLM's output ("model choice") as a single data point underutilizes the information inherent to the probabilistic nature of LLMs. This paper intro...

---

### 27. [The World Is Bigger! A Computationally-Embedded Perspective on the Big World Hypothesis](https://arxiv.org/abs/2512.23419)

**Authors**: Alex Lewandowski, Adtiya A. Ramesh, Edan Meyer, Dale Schuurmans, Marlos C. Machado  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2512.23419v1  

#### Abstract
Continual learning is often motivated by the idea, known as the big world hypothesis, that "the world is bigger" than the agent. Recent problem formulations capture this idea by explicitly constraining an agent relative to the environment. These constraints lead to solutions in which the agent conti...

---

### 28. [Toward Equitable Recovery: A Fairness-Aware AI Framework for Prioritizing Post-Flood Aid in Bangladesh](https://arxiv.org/abs/2512.22210)

**Authors**: Farjana Yesmin, Romana Akter  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2512.22210v1  

#### Abstract
Post-disaster aid allocation in developing nations often suffers from systematic biases that disadvantage vulnerable regions, perpetuating historical inequities. This paper presents a fairness-aware artificial intelligence framework for prioritizing post-flood aid distribution in Bangladesh, a count...

---

### 29. [Lessons from Neuroscience for AI: How integrating Actions, Compositional Structure and Episodic Memory could enable Safe, Interpretable and Human-Like AI](https://arxiv.org/abs/2512.22568)

**Authors**: Rajesh P. N. Rao, Vishwas Sathish, Linxing Preston Jiang, Matthew Bryan, Prashant Rangarajan  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2512.22568v1  

#### Abstract
The phenomenal advances in large language models (LLMs) and other foundation models over the past few years have been based on optimizing large-scale transformer models on the surprisingly simple objective of minimizing next-token prediction loss, a form of predictive coding that is also the backbon...

---

### 30. [Memento-II: Learning by Stateful Reflective Memory](https://arxiv.org/abs/2512.22716)

**Authors**: Jun Wang  
**Category**: cs.AI  
**Published**: 2025-12-31  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2512.22716v1  

#### Abstract
We propose a theoretical framework for continual and experiential learning in large language model agents that integrates episodic memory with reinforcement learning. The framework identifies reflection as the key mechanism that enables agents to adapt through interaction without back propagation or...

---

## 🔧 Configuration

This bot is configured to look for papers containing the following keywords:
- State Space, SSM, framework, System, Generation, Video, Linear, LLM, RL, RLHF, Inference, Training, Attention, Pipeline, MOE, Sparse, Quantization, Speculative, Efficient, Efficiency, Framework, Parallel, Distributed, Kernel, Decode, Decoding, Prefill, Throughput, Fast, Network, Hardware, Cluster, FP8, FP4, Optimization, Scalable, Communication

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
