# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-18 06:48:40 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation](https://arxiv.org/abs/2603.16219)

**Authors**: Hang Lv, Sheng Liang, Hao Wang, Yongyue Zhang, Hongchao Gu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.16219v1  

#### Abstract
Realizing personalized intelligence faces a core dilemma: sending user history to centralized large language models raises privacy concerns, while on-device small language models lack the reasoning capacity required for high-quality generation. Our pilot study shows that purely local enhancements re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对个性化生成中的一个**核心困境（fundamental dilemma）**：
- **集中式大模型（Centralized LLMs）** 虽然具备强大的推理能力，但需要上传用户的私密历史数据（如对话记录、偏好等），引发严重的**隐私泄露风险**。
- **本地小模型（On-device SLMs）** 虽能保护隐私，但由于容量有限，缺乏复杂推理能力，生成质量低，容易出现逻辑错误或幻觉。

现有方法在“**推理能力**”与“**数据隐私**”之间难以兼顾，导致个性化生成效果受限。

---

### **提出的新方法：SPECSTEER**
作者提出了 **SPECSTEER** —— 一种**非对称协同推理框架（asymmetric collaborative inference framework）**，通过将本地上下文与云端大规模推理相结合，实现高效且个性化的文本生成。

其核心思想是将协作建模为**贝叶斯知识融合（Bayesian knowledge fusion）**，并重新利用 **Speculative Decoding（推测性解码）** 作为分布式对齐协议，构建了一个三阶段流程：

#### **Draft-Verify-Recover Pipeline**
1. **Draft（起草）**  
   - 由本地设备上的 **Specialist（小模型）** 基于私有用户历史生成个性化候选序列。
   - 优势：保留用户意图，无需上传原始数据。

2. **Verify（验证）**  
   - 云端的 **Generalist（大模型）** 使用**比率验证机制（ratio-based verification）** 判断草案是否符合逻辑合理性。
   - 关键创新：不直接比较概率，而是计算 $\frac{P_{LLM}(y)}{P_{SLM^-}(y)}$，其中 $P_{SLM^-}$ 是无上下文的小模型基准分布。
   - 这种方式使验证过程**无需访问私有上下文**，仅依赖公开可得的信息进行逻辑校验。

3. **Recover（恢复）**  
   - 若草案被拒绝，则执行**引导式恢复（steering recovery）**：
     - 将 Specialist 的个性化信号以 **logit injection** 形式注入到 Generalist 的输出中。
     - 公式：$h_{\text{rec}} = h_{LLM} + \beta (h_{SLM} - h_{SLM^-})$
   - 在纠正逻辑错误的同时，仍保持对用户意图的忠实度。

---

### **相比现有方法的优势**
| 维度 | SPECSTEER | 传统方法（如 RAG, LoRA, Token-level Fusion） |
|------|---------|------------------------------------------|
| **隐私保护** | ✅ 完全本地化处理私有上下文 | ❌ 需上传上下文或微调参数 |
| **推理能力** | ✅ 利用云端大模型强推理 | ❌ 受限于小模型能力 |
| **延迟效率** | ✅ 异步验证，通信开销极低（仅传 token ID） | ❌ 同步融合需频繁传输 logits，高延迟 |
| **对齐精度** | ✅ Ratio-based 验证避免误拒个性化内容 | ❌ 粗粒度提示易丢失细节；逐 token 融合易漂移 |

> ✅ **核心突破**：首次将 Speculative Decoding 成功应用于**跨隐私边界的个性化生成任务**，实现了**高质量、低延迟、高隐私**的统一。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LaMP (Personalized Language Model Benchmark)**  
  包含多个个性化生成任务，如新闻标题生成、学术论文标题生成、推特改写等。
- **LongLaMP**  
  更复杂的长文本个性化生成基准，涵盖摘要生成、评论撰写、Reddit 帖子写作等。

> 所有任务均要求模型整合用户历史进行风格/内容适配。

---

### **实验设置**
- **模型组合（Model Pairs）**：
  - Qwen3-0.6B / Qwen3-32B
  - Qwen2.5-1.5B / Qwen2.5-32B
  - Llama-3.21B / Llama-3.1-8B
- **部署架构**：
  - **Edge（本地）**：运行 Specialist（MSLM），拥有用户私有上下文。
  - **Cloud（云端）**：运行 Generalist（MLLM）和通用版 Specialist（MSLM⁻）用于验证。

- **评估指标**：
  - **ROUGE-1 & ROUGE-L**：衡量生成文本与参考答案的重叠程度。
  - **Speedup（加速比）**：每秒生成 token 数量（tokens/s），相对于标准 LLM 推理的速度提升。
  - **Acceptance Rate (α)**：草案被接受的比例，反映协作效率。

- **基线方法对比**：
  - **SLM Direct / LoRA / RAG / RAFT**：各类本地增强方法。
  - **LLM (Zero-shot)**：纯云端大模型，无个性化信息。
  - **Iterative Fusion (CoSteer)**：逐 token 协同推理，高通信成本。
  - **Standard Speculative Decoding (SD)**：无个性化感知的推测解码。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表 2 & 表 3：LongLaMP 上的生成质量对比（ROUGE-1）**
| 方法 | Abstract | Review | Writing |
|------|--------|-------|--------|
| SLM+ (RAG) | 39.89 | 23.18 | 25.50 |
| LLM (32B) | 40.18 | 31.18 | 29.46 |
| **SPECSTEER** | **41.35** | **33.03** | **30.79** |

✅ **全面超越所有基线**，尤其在需要深度推理的 **Review 写作任务上增益显著（+1.85 R1）**。

---

#### **表 4：效率与质量权衡分析（Qwen3-0.6B/32B）**
| 方法 | Speed (tokens/s) | Speedup | Acceptance Rate α |
|------|------------------|---------|--------------------|
| Vanilla LLM | 22.58 | 1.00× | – |
| CoSteer | 9.71 | 0.43× | – |
| LightCoSteer | 16.03 | 0.71× | – |
| Standard SD | 22.13 | 0.98× | ~35% |
| **SPECSTEer (λ=0.5)** | **39.29** | **1.74×** | 56.36% |
| **SPECSteer (λ=0.1)** | **53.29** | **2.36×** | 73.79% |

✅ **最高达 2.36× 推理加速**，远超其他协同方法（多数反而更慢）。

> ⚠️ 注意：随着 λ 减小（放宽验证阈值），接受率上升，速度加快，但 ROUGE 微降，存在合理权衡。

---

### **消融实验结果**

#### **A. 不同 Specialist 质量下的鲁棒性（Appendix A.3）**
| 设置 | Review R1 (SLM+) | Review R1 (**SPECSTEER**) |
|------|------------------|----------------------------|
| 加噪输入（Noise） | 23.54 | **31.78** |
| BM25 检索 | 23.18 | **33.03** |
| BGE 检索 | 25.35 | **33.45** |

➡️ 即使 Specialist 输入质量差，SPECSTEER 仍可通过云端验证修复逻辑错误，**表现出强大鲁棒性**。

---

#### **B. 跨架构部署能力（Appendix A.4）**
- 使用 **Qwen3-0.6B（Specialist） + Llama-3.1-8B（Generalist）**
- 结果显示仍能取得一致增益（如 Review R1 达 32.03，优于两者单独表现）
➡️ 证明框架具有良好的**架构无关性（architecture-agnostic）**，适用于真实异构环境。

---

#### **C. 超参数敏感性分析**
- **β（恢复强度）**：在 [0.5, 2.0] 区间内稳定有效；过大（>2.5）会导致过度偏移，破坏连贯性。
- **λ（验证阈值）**：λ ∈ [0.1, 0.5] 为最优区间，兼顾接受率与生成质量。

---

## **4. 关键结论和发现**

### **主要发现**
1. **本地模型存在“能力鸿沟”（Capacity Deficit）**  
   - 即使采用 RAG、LoRA 等先进技术，小型本地模型也无法匹敌大型通用模型的推理能力。
   - 私有数据带来的信息优势被弱推理能力所抵消。

2. **SPECSTEER 成功弥合了这一鸿沟**  
   - 通过 **Draft-Verify-Recover** 架构，在不牺牲隐私的前提下引入云端强推理。
   - 实现了**个性化意图**与**全局逻辑一致性**的协同优化。

3. **效率与质量兼得**  
   - 相比传统协同方法，SPECSTEER 减少了通信负担，实现了 **2.36× 的端到端加速**。
   - 是首个同时实现**高性能、高隐私、高效率**的边缘-云协同生成方案。

---

### **局限性**
1. **严重依赖 Specialist 的基本可用性**  
   - 如果本地模型完全失效或无法提供可靠个性化信号（如灾难性遗忘），则无法有效恢复。
   - 但论文指出这种情况较少见，且此时 Generalist 至少能保证输出的基本合理性。

2. **对极端分布偏移仍可能误判**  
   - 当用户偏好极度偏离常识时，Ratio-based 验证可能仍将合理内容误拒。
   - 可通过调整 λ 或引入动态阈值缓解。

3. **当前未集成更高级隐私技术**  
   - 如差分隐私、联邦学习等，虽框架支持模块化扩展，但尚未实证。

---

### **未来工作方向**
- 将 SPECSTEER 扩展至多模态个性化代理（如语音助手、视觉推荐）。
- 探索动态自适应的 λ 和 β 控制策略，实现在线调优。
- 结合联邦学习更新 Specialist，形成闭环个性化系统。
- 应用于实时对话系统，研究其在流式生成中的表现。

---

> 📌 **总结一句话**：  
> **SPECSTEER 开创性地将 Speculative Decoding 改造为隐私安全的协同推理协议，实现了“让小模型起草、大模型把关”的高效个性化生成范式，为下一代边缘智能提供了可行路径。**

</details>

---

### 2. [ExpressMind: A Multimodal Pretrained Large Language Model for Expressway Operation](https://arxiv.org/abs/2603.16495)

**Authors**: Zihe Wang, Yihuan Wang, Haiyang Yu. Zhiyong Cui, Xiaojian Liao, Chengcheng Wang, Yonglin Tian, Yongxin Tong  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.16495v1  

#### Abstract
The current expressway operation relies on rule-based and isolated models, which limits the ability to jointly analyze knowledge across different systems. Meanwhile, Large Language Models (LLMs) are increasingly applied in intelligent transportation, advancing traffic models from algorithmic to cogn...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《ExpressMind: A Multimodal Pretrained Large Language Model for Expressway Operation》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前高速公路运营系统依赖**规则驱动**和**孤立模型**，难以实现跨系统的联合知识分析。通用的 **Large Language Models (LLMs)** 虽在智能交通中有所应用，但缺乏对高速公路领域特殊法规、因果逻辑和非标准场景的理解能力。此外，该领域存在以下挑战：
- 多模态数据异构性强（视频、文本、实时流）
- 安全性和准确性要求极高
- 高质量标注的多模态数据稀缺
- 缺乏统一的认知中枢支持复杂推理与决策

### **提出的新方法与创新思路**
本文提出了 **ExpressMind** —— 首个面向高速公路运营的**多模态预训练大语言模型 (MLLM)**，作为智能高速系统的认知核心。其核心创新包括：

#### ✅ **五大核心贡献**
1. **Full-stack Expressway Dataset 构建**
   - 构建了行业首个覆盖文本认知、逻辑推理与视觉感知的全栈数据集，包含：
     - **Express-Insight**: 700万token的专业文本（法规、政策、理论）
     - **Express-QA**: 87万高质量问答对
     - **Express-IncidentCoT**: 1,786条真实事故响应的四阶段 Chain-of-Thought 数据
     - **Express-VQA**: 1,627段监控视频 + 3,200+ VQA 对，涵盖山东、广东等地真实路况

2. **双层预训练范式 (Dual-layer Pre-training)**
   - 第一阶段：无监督训练（Unsupervised Training），学习基础领域知识
   - 第二阶段：全参数监督微调（SFT），引入掩码损失函数聚焦于响应生成，提升任务对齐能力

3. **RL-aligned Chain-of-Thought (RL-CoT) 推理机制**
   - 利用 **Group Relative Policy Optimization (GRPO)** 强化学习算法，对齐模型推理路径与专家处置逻辑
   - 设计三维奖励函数：
     - `R_struct`: 结构完整性（强制“感知-分析-决策-反思”流程）
     - `R_know`: 领域一致性（鼓励使用专业术语，惩罚语言退化）
     - `R_sem`: 语义一致性（与专家案例嵌入相似度匹配）

4. **图增强检索增强生成 (Graph-Augmented RAG)**
   - 基于 **LightRAG** 构建动态知识图谱，支持增量更新
   - 双层级检索机制：
     - **低层检索**：基于实体关键词精确匹配
     - **高层检索**：基于概念语义线索匹配关系边
   - 实现对实时交通状态、突发事件等动态信息的有效索引与融合

5. **视觉优先对齐机制 (Visual-Prior Alignment, VPA)**
   - 在跨模态编码器中引入可学习的交叉注意力重加权机制
   - 动态增强视觉特征权重，建立“视觉优先”的归纳偏置
   - 结合 **MRoPE** 和 **DeepStack** 技术，优化时空位置编码与深层特征融合

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 类型 | 规模 | 内容说明 |
|-------|------|------|---------|
| **Express-Insight** | 文本语料 | >7M tokens | 法规、政策文件、智慧高速教材 |
| **Express-QA** | QA对 | 870,000+ | 自动构造并精炼的问答数据 |
| **Express-IncidentCoT** | CoT推理链 | 1,786条 | 四阶段事故响应逻辑链 |
| **Express-VQA** | 多模态 | 1,627视频 + 3,200+ VQA | 来自山东、广东、天津的真实监控视频 |

### **实验设置**
- **硬件环境**：8 × NVIDIA H20 GPU，CUDA 12.4+, PyTorch 2.4+
- **软件框架**：DeepSpeed, FlashAttention-2, PEFT 支持高效分布式训练
- **总训练耗时**：约 700 GPU 小时
- **模型架构基础**：基于 **Qwen-14B** 进行领域适配

### **评估指标**
#### **文本理解任务（QA）**
- Accuracy, F1-Score, Embedding Similarity, **GPT-Score**（由 GPT-4o 打分，衡量语义深度）

#### **推理对齐任务（Incident Strategy）**
采用 “LLM-as-a-Judge” 框架，从五个维度打分（0–10）：
- **Safety Compliance**（安全合规性）
- **Preventive Insight**（预防洞察力）
- **Logical Consistency**（逻辑一致性）
- **Actionability**（可执行性）
- **Cause Depth**（原因分析深度）

#### **多模态理解任务（Video Understanding）**
- 自动化指标：BLEU-4, ROUGE-L, CIDEr, BERTScore
- 细粒度人工模拟评估维度：
  - Accuracy, Level, Precision, Space, Time, Analysis

### **基线方法对比**
| 类别 | 对比模型 |
|------|--------|
| **通用LLM** | Qwen-32B, Llama-3.3-70B, Baichuan-32B, GLM-4-32B |
| **蒸馏/领域调优模型** | DeepSeek-R1-Distill-Qwen/Llama |
| **多模态模型** | VideoLLaMA3, MiniCPM-V4.5, InternVL3.5, Qwen3-VL |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **预训练阶段表现（Table 1）**
在三大知识类 QA 任务上全面超越所有基线模型：

| 任务 | 最佳基线（Llama-3.3-70B） | **ExpressMind-14B** | 提升幅度 |
|------|--------------------------|------------------------|----------|
| 高速公路法规 QA（MCQ 准确率） | 97.5% | **98.4%** | +0.9pp |
| 智慧高速知识 QA（短答 F1） | 86.9 | **87.8** | +0.9 |
| 智能交通系统知识 QA（GPT-Score） | 84.7 | **86.5** | +1.8 |

> 💡 **GPT-Score 平均达 85.7%**，表明其具备接近专家级的语义理解能力。

#### ✅ **RL-CoT 推理对齐效果（Figure 6）**
在事故响应策略生成任务中，**ExpressMind (Pre-train+RL)** 显著优于所有通用模型：

| 指标 | Qwen-32B | Llama-3.3-70B | **ExpressMind+RL** |
|------|----------|----------------|---------------------|
| Safety Compliance | ~6.5 | ~7.0 | **~8.8** |
| Actionability | ~6.8 | ~7.2 | **~8.5** |
| Cause Depth | ~6.0 | ~6.5 | **~8.0** |

> 🔍 **消融显示 RL 对推理质量起决定性作用**：未经过 RL 对齐的版本性能最弱。

#### ✅ **推理效率（Figure 7）**
- **平均推理延迟仅 13.2ms**，相比 Baichuan-32B 加速 **24.6%**
- 延迟抖动极小，适合毫秒级响应的智能高速场景

#### ✅ **RAG 检索增强消融实验（Figure 9）**
| 设置 | 法规理解 F1 | 技术规范 F1 | 专业词汇出现概率提升 |
|------|------------|--------------|------------------|
| no RAG | 0.73 | 0.61 | — |
| with RAG | 0.78 | 0.78 | +12.3% |
| with LightRAG | **0.88** | **0.87** | **+16.7%** |

> 表明图结构 RAG 显著提升了事实准确性和术语覆盖率。

#### ✅ **多模态理解性能（Table 2 & Figure 10–11）**
在 Express-VQA 上，**ExpressMind-VL** 全面领先：

| 模型 | BLEU-4 ↑ | ROUGE-L ↑ | CIDEr ↑ | BERTScore ↑ |
|------|---------|-----------|----------|-------------|
| Qwen3-VL | 84.85 | 89.30 | 72.96 | 89.18 |
| **ExpressMind-VL** | **85.24** | **89.25** | **73.36** | **89.28** |

在六类典型事件检测中（异常停车、行人闯入、拥堵等）：
- **准确率与召回率均超过 90%**

#### ✅ **VPA 消融实验（Figure 12）**
引入 VPA 后，视觉特征贡献显著增强，在动态场景理解任务中带来明显增益。

---

## **4. 关键结论和发现**

### **主要发现**
1. **ExpressMind 是首个专为高速公路设计的 MLLM**，实现了从“感知-分析-决策”的完整闭环。
2. **RL-CoT 对齐机制有效引导模型遵循专家思维链**，大幅提升策略的安全性与可执行性。
3. **Graph-Augmented RAG 能有效整合动态知识**，解决 LLM 参数静态性带来的知识滞后问题。
4. **VPA 机制强化了视觉主导的多模态融合**，特别适用于以视频为核心的交通监控场景。
5. 在多个任务上，**14B 规模的 ExpressMind 超越了 70B 级别的通用模型**，验证了领域专用化的价值。

### **局限性**
- 当前模型仍依赖较强的数据标注与专家经验构建 CoT 数据
- 实时视频流处理尚未完全部署至边缘设备
- 多车协同、长序列时空推理能力仍有待加强

### **未来工作方向**
1. **增强多模态时空推理能力**，支持更复杂的动态场景建模
2. **深化 Chain-of-Thought 在长文本分析中的对齐能力**
3. **推进模型轻量化**，支持在路侧单元（RSU）等边缘节点部署
4. 探索与车路协同系统（V2X）的深度融合，构建端云一体智能体

---

> 🌐 **代码与数据已开源**：  
> [https://wanderhee.github.io/ExpressMind/](https://wanderhee.github.io/ExpressMind/)  
> 包含模型、数据集、Benchmark 与系统演示。

</details>

---

### 3. [FactorEngine: A Program-level Knowledge-Infused Factor Mining Framework for Quantitative Investment](https://arxiv.org/abs/2603.16365)

**Authors**: Qinhong Lin, Ruitao Feng, Yinglun Feng, Zhenxin Huang, Yukun Chen, Zhongliang Yang, Linna Zhou, Binjie Fei, Jiaqi Liu, Yu Li  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.16365v1  

#### Abstract
We study alpha factor mining, the automated discovery of predictive signals from noisy, non-stationary market data-under a practical requirement that mined factors be directly executable and auditable, and that the discovery process remain computationally tractable at scale. Existing symbolic approa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**量化投资中的 alpha 因子挖掘**（alpha factor mining）任务，旨在从高噪声、非平稳的市场数据中自动发现具有预测能力且可执行、可审计的因子。现有方法存在以下三大挑战：
1. **表达能力受限**：基于符号表达式的方法（如 GP、Alpha158）受限于预定义算子空间，搜索能力有限，难以捕捉复杂市场动态。
2. **因子多样性与稳定性不足**：缺乏有效机制将金融报告等非结构化知识转化为可执行因子，导致因子同质化严重，易衰减。
3. **进化效率低下**：LLM 生成与回测验证之间存在巨大计算速度不匹配，导致整体流程成本高昂。

### **提出的新方法与新思路**
作者提出了 **FactorEngine (FE)** ——一个**程序级、知识注入的因子挖掘框架**，其核心创新包括：

- **程序级因子表示**（Program-level Representation）  
  将因子表示为**图灵完备的 Python 程序**，而非传统符号表达式。这允许使用条件逻辑、循环、控制流等复杂结构，极大提升了模型表达能力和对市场动态的建模灵活性。

- **宏-微协同进化机制**（Macro-Micro Co-evolution）  
  实现三个关键分离：
  1. **逻辑演化 vs 参数优化**：LLM 负责高层次语义推理（宏观逻辑修改），本地计算资源通过 **Bayesian Optimization** 自动调参（微观参数优化）。
  2. **LLM 引导搜索 vs 自动化搜索**：LLM 进行定向启发式探索，Bayesian Search 高效收敛最优参数。
  3. **LLM 使用 vs 本地计算**：Phase 1 用 LLM 快速修复代码语法错误；Phase 2 完全在本地并行执行大规模参数搜索，避免频繁调用 LLM。

- **知识注入启动模块**（Knowledge-Infused Bootstrapping）  
  构建一个闭环多智能体系统，从**金融研究报告**中提取核心思想，经由“理解 → 验证 → 伪代码生成 → 可执行代码合成”流程，自动生成高质量初始因子池，实现领域知识的有效转化。

- **经验知识库与轨迹感知优化**（Chain of Experience, CoE）  
  维护完整的演化路径历史（包括失败尝试），供 LLM 学习“如何从低绩效恢复”，提升探索鲁棒性和成功率。

### **相比现有方法的优势**
| 维度 | 传统方法（GP/NN） | Agent-based 方法（AlphaAgent/RD-Agent） | **FactorEngine (FE)** |
|------|-------------------|----------------------------------------|------------------------|
| 表达能力 | 低（受限于算子集） | 中（仍为 symbolic） | ✅ 高（Turing-complete code） |
| 可解释性 | 高 | 中高 | ✅ 高（代码即因子） |
| 知识利用 | 手工设计 | 报告作为灵感来源 | ✅ 报告→可执行代码 |
| 搜索效率 | 慢 | 慢（依赖 LLM 多轮交互） | ✅ 快（分离逻辑与参数搜索） |
| 因子多样性 | 一般 | 一般 | ✅ 显著更高 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **市场数据**：来自 **Qlib** 的中国 A 股全市场 OHLCV 数据。
- **划分方式**：
  - 训练集：2008-01-01 至 2014-12-31
  - 验证集：2015-01-01 至 2016-12-31
  - 测试集：2017-01-01 至 2024-12-31
- **金融报告数据**：仅使用**2017 年前发布的研报**进行因子提取，防止测试期信息泄露。

### **实验设置**
- **迭代次数**：分别运行 200 和 400 轮因子演化。
- **初始种子**：
  - `FE-alpha`：以人工设计因子（如 Alpha158 子集）初始化。
  - `FE-report`：以从研报中提取的因子初始化。
- **多岛机制**（Multi-Island Evolution）：设置 2 个独立进化岛，每 7 轮迁移 top-3 因子，促进多样性传播。
- **Backbone Model**：统一使用 **Gemini-2.5-Pro** 作为所有 agent 方法的 LLM 后端，确保公平比较。

### **评估指标**

#### **预测性能指标**
- **IC**（Information Coefficient）：预测信号与实际收益的截面相关系数。
- **ICIR**：IC 的均值除以其标准差，衡量稳定性。
- **RIC / RICIR**：Rank IC 及其信息比率。

#### **组合表现指标**
- **AR**（Annualized Return）：年化收益率。
- **IR**（Information Ratio）：超额收益与跟踪误差之比。
- **MDD**（Maximum Drawdown）：最大回撤。
- **SR**（Sharpe Ratio）：夏普比率（无风险利率设为 0）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（400 迭代，CSI300 市场）**

| 方法 | IC | ICIR | RIC | RICIR | AR | SR | MDD |
|------|-----|-------|------|--------|-------|------|-------|
| Alpha158 | 0.0299 | 0.2008 | 0.0331 | 0.2164 | 0.0840 | 0.7440 | 17.49% |
| RD-Agent-2 | 0.0269 | 0.1833 | 0.0300 | 0.1978 | 0.0917 | 0.8113 | 30.68% |
| AlphaAgent-2 | 0.0282 | 0.1978 | 0.0313 | 0.2142 | 0.0673 | 0.6346 | 17.00% |
| **FE-alpha-2** | **0.0315** | **0.2211** | **0.0344** | **0.2360** | **0.0943** | **0.8241** | **15.07%** |
| **FE-report-2** | **0.0474** | **0.3185** | **0.0475** | **0.3146** | **0.1899** | **1.6001** | **12.61%** |

> ✅ **FE-report 在几乎所有指标上显著领先**，尤其在 IC (+58%) 和 AR (+126%) 上远超 Alpha158。

### **与基线方法的对比结果**
- **超越所有 baseline**：无论是在 CSI300 还是 CSI500 市场，FE 在 IC、ICIR、AR、SR 等关键指标上均取得 SOTA 性能。
- **报告驱动优于人工因子**：`FE-report` 始终优于 `FE-alpha`，说明从研报中提取的知识能更有效地引导因子演化。
- **更强的抗衰减能力**：如 Fig. 4 所示，FE 报告因子在 2021 年后停止衰减甚至回升，而其他方法持续下滑。
- **更高的因子多样性**：
  - 使用 MDS 可视化因子相关性，FE 展现出“环形分散”模式，表明因子间独立性强。
  - **Radius of Gyration (RoG)** 达到最高，说明嵌入空间分布最广。
  - **保留率（Keep Ratio）达 57.1%**，远高于 AlphaAgent (21.9%) 和 RD-Agent (3.9%)。

### **消融实验结果**
#### **(1) Bayesian 微搜索的作用**
- 对比有无 Bayesian 参数优化（固定参数 vs Optuna 调参）：
  - 最终性能提升约 **52%**（fitness 0.25 → 0.38）。
  - 收敛速度更快，早期即可识别优质逻辑。
- 结论：**Bayesian 微搜索显著加速进化过程并提高最终质量**。

#### **(2) 多岛机制与 CoE 提示的影响**
- **多岛配置（2-island） > 单岛（1-island）**：显著提升 RIC、AR 和 IR。
- **Chain-of-Experience 提示 > Top-K 提示**：引入完整演化轨迹反馈可进一步提升性能。
- 初始因子数量增加（6 → 10）也带来稳定增益。

#### **(3) Token 效率与可执行性**
| 方法 | 成本 ($) | 时间 (h) | 可执行率 | Debug API 比例 |
|------|----------|---------|------------|----------------|
| RD-Agent | 16.91 | 48.0 | 96% | 68% |
| AlphaAgent | 11.61 | 9.7 | 93% | 51% |
| **FactorEngine** | **12.01** | **0.5** | **99%** | **32%** |

> ✅ FE 在保持相近成本下，**运行时间仅为 AlphaAgent 的 1/20，且可执行率最高、调试开销最低**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **程序级表示是突破表达瓶颈的关键**：将因子视为可进化的完整程序，使模型能够学习复杂的控制流和高阶特征交互，适应快速变化的市场环境。
2. **知识注入大幅提升起点质量和演化效率**：从金融研报中自动提取并转化为可执行因子，不仅提高了初始种群质量，还增强了因子的经济可解释性。
3. **宏-微分离架构极大提升进化效率**：通过解耦 LLM 的语义推理与本地的自动化参数搜索，实现了高效、低成本的大规模因子演化。
4. **经验记忆支持失败学习与鲁棒探索**：记录完整演化轨迹（含失败案例）有助于 LLM 内化“恢复策略”，避免重复踩坑。

### **方法的局限性**
- **依赖高质量研报输入**：若研报本身逻辑错误或过时，可能误导初始因子生成。
- **LLM 推理仍具随机性**：不同运行轨迹可能出现较大差异，需多岛机制缓解。
- **未显式建模交易成本与冲击**：当前回测虽考虑滑点与手续费，但未在优化目标中联合建模。
- **泛化性待验证**：目前实验集中在中国股市，跨市场、跨周期的稳健性有待进一步检验。

### **未来工作方向**
- 扩展至更多模态数据（新闻、社交媒体、财报文本等）。
- 提升模型在分布偏移和交易成本下的鲁棒性。
- 允许 LLM 主动“提问”市场数据（active interrogation），增强探索能力。
- 更好地刻画“多样性”与“泛化性”在程序演化中的本质关系。

---

> 📌 **总结一句话**：  
> **FactorEngine 通过“程序即因子 + 知识注入 + 宏-微协同进化”的范式革新，在保持高可解释性的同时，实现了更高效、更稳定、更具多样性的 alpha 因子挖掘，达到了当前 SOTA 的预测与组合表现。**

</details>

---

### 4. [Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights](https://arxiv.org/abs/2603.16817)

**Authors**: Yi Chen, Daiwei Chen, Sukrut Madhav Chikodikar, Caitlyn Heqi Yin, Ramya Korlakai Vinayak  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.16817v1  

#### Abstract
Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统地研究了基于 **Retrieval-Augmented Generation (RAG)** 的大语言模型（LLM）在应用 **Conformal Factuality**（保角事实性）框架时的可靠性与实用性问题。尽管 RAG 和 Conformal Prediction 都被广泛用于缓解 LLM 的 **hallucination**（幻觉），但两者结合的实际效果缺乏系统分析。

具体而言，论文揭示了以下关键问题：
- **传统 factuality 指标存在缺陷**：如 Empirical Factuality（EF）会将空输出视为“完全正确”，导致高 factuality 但低实用性的误导性结果。
- **Conformal Factuality 的鲁棒性不足**：其统计保证在面对分布偏移（distribution shift）或对抗性干扰（distractors）时容易失效。
- **计算效率与性能权衡不明确**：是否必须使用更大的 LLM 作为 verifier 才能获得更好的效果？

### 提出了什么新方法或新思路
论文提出了 **一套新颖的、关注信息保留度（informativeness）的评估指标体系**，以更真实地反映任务效用：

| 新指标 | 定义与作用 |
|--------|-----------|
| **Non-empty Rate (NR)** | 过滤后输出非空的比例，衡量信息保留程度 |
| **Non-vacuous Empirical Factuality (NvEF)** | 仅在非空输出上计算的 EF，避免空输出的虚假高分 |
| **Sufficient Correctness (SC)** | 输出是否包含足够正确的信息来推断最终答案，衡量任务级效用 |
| **Conditional Sufficient Correctness (CSC)** | 在原始输出已满足 SC 的前提下，过滤后仍保持 SC 的比例，隔离过滤过程的影响 |

此外，论文对 **factuality scoring 函数**进行了全面设计空间探索，比较了：
- **Entailment-based scorers**（如 RoBERTa、DeBERTa）
- **LLM-based model confidence scorers**（不同 prompt 设计、模型规模）

### 相比现有方法的优势
- **评估更全面**：首次引入 **informativeness-aware 指标**，揭示了 factuality 与 usefulness 之间的根本权衡。
- **分析更系统**：覆盖 generation、scoring、calibration、robustness、efficiency 全流程，而非孤立优化某一部分。
- **结论更具实践指导意义**：证明轻量级 verifier 可优于大型 LLM，为部署提供高效方案。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个多样化任务上进行，涵盖开放生成、数学推理与问答：

| 数据集 | 任务类型 | 特点 |
|-------|---------|------|
| **FActScore** | 开放式传记生成 | 601 个名人维基页面，无标准答案，适合评估事实精确性 |
| **FActScore-Rare** | 罕见人物传记 | 聚焦模型参数知识不足的情况，测试 RAG 必要性 |
| **MATH** | 数学推理 | 12K+ 竞赛级数学题，需逻辑推导，参考文本由 `gpt-5-nano` 生成先验知识 |
| **Natural Questions (NQ)** | 开放域问答 | 10K 真实搜索查询，有长/短答案标注 |

### 实验设置
- **Response Generator G**：多种开源 LLM，包括 Qwen3、Llama-3.x、SmolLM2、gpt-oss 等（见 Table 1）。
- **Scoring Function f**：
  - **Entailment-based**：DocNLI 模型、RoBERTa-large-mnli（sentence-level）
  - **LLM-based**：使用不同 prompt 策略（CoT、highlighting、scalar vs boolean、consistency averaging）
- **Calibration Set X**：独立于测试集，用于确定过滤阈值 $ T_\alpha $
- **Pipeline**：`Query + Reference → G → Output → Parser → Claims → Scorer → Scores → Conformal Filter → Merged Output`

### 评估指标
#### 传统指标
- **Empirical Factuality (EF)**：所有保留声明均为事实的比例（空输出视为正确）
- **Power**：平均保留的真实声明比例
- **False Positive Rate (FPR)**：错误声明未被过滤的比例
- **Correctness**：输出与标准答案等价的比例

#### 论文提出的新指标
- **NR, NvEF, SC, CSC**（定义见上文）

### 基线方法对比
- **Scoring Function 对比**：
  - LLM-based model confidence scorer（多变体）
  - Entailment-based scorers（document/sentence-level）
- **Verifier 规模对比**：从 135M 到 117B 参数的模型
- **是否使用 reference**：在 generation 和 scoring 阶段分别控制

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）Conformal Filtering 存在严重有用性（usefulness）瓶颈
- 在高 factuality 水平（如 1−α=0.9）下，**NR 显著下降**，表明大量输出被过滤为空。
- **NvEF 与 EF 差距巨大**：例如在 FActScore 上，EF 可达 0.9，但 NvEF 不足 0.6，说明多数“正确”输出是空的。
- **SC 下降明显**：即使原始输出本可回答问题，过滤后也常丢失关键信息。

> 🔍 **发现**：当前 conformal filtering 更像“安全刹车”，而非“智能编辑”。

#### （2）Conformal Factuality 不具备鲁棒性
- **分布偏移（Distribution Shift）**：
  - 当 calibration 数据来自不同分布（如 GPT-4 生成）时，**EF 显著低于目标水平**，尤其在 MATH 数据集上。
  - 即使某些 scorer 表现稳定，其 **Power 极低**，实用性差（图 9–10）。
- **对抗性干扰（Distractors）**：
  - 注入 25% 幻觉声明后，**EF 急剧下降至 0.6 以下**，远低于目标。
  - 即便通过在 calibration 中加入 distractor 来“预适应”，虽可恢复 EF，但 **NR 下降超过 50%**，代价过高（图 12）。

> ❗ **结论**：Conformal guarantee 依赖 exchangeability 假设，在现实部署中极易被打破。

#### （3）轻量级 verifier 可胜过大型 LLM
- **Entailment-based scorers（如 DeBERTa）**：
  - 在 **FLOPs 上比 LLM-based scorers 低 100× 以上**（表 2）。
  - 在 **Power、SC、NR 上匹配甚至超越** 大型 LLM（如 gpt-oss-20b）。
- **LLM Scorer 缩放规律不成立**：
  - 增大 scorer 模型规模（如从 Qwen3-0.6B 到 32B）**并未带来一致提升**，有时反而下降（图 6）。
  - **Smaller models are competitive**，尤其在成本敏感场景。

> ✅ **亮点**：**DeBERTa-based document entailment** 在效率与性能上达到最佳平衡。

#### （4）Prompt 设计影响有限
- **最有效的策略**：
  - 输出 **numeric score（连续值）** 优于 boolean
  - **consistency averaging（多次采样取均值）** 提升稳定性
- **无效或不稳定策略**：
  - Chain-of-Thought（CoT）和证据高亮（highlighting）**无一致增益**

---

## 4. 关键结论和发现

### 主要发现
1. **Factuality ≠ Usefulness**  
   当前 conformal filtering 框架在追求高 factuality 时，常以牺牲信息量为代价，产生“正确但无用”的空输出。

2. **Conformal Guarantee 不鲁棒**  
   其统计保证高度依赖 calibration 与测试数据的分布一致性，面对实际中的分布偏移或对抗性干扰时极易失效。

3. **轻量级 verifier 更优**  
   **Entailment-based scorers（如 DeBERTa）** 在显著更低的计算成本下，实现了与大型 LLM 相当甚至更优的 filtering 效果。

4. **模型缩放不等于性能提升**  
   增大 verifier 模型规模 **不能保证** 更好的 factuality 或 filtering 效果，小型模型同样有效。

### 方法的局限性
- **依赖高质量 reference**：假设 oracle retriever，忽略了检索阶段的误差。
- **过滤不可逆**：只能删除信息，无法补充或修正，限制了上限。
- **新指标依赖强标注**：SC/CSC 需人工或强模型判断“是否足够回答问题”，自动化难度高。

### 未来工作方向
- **开发鲁棒的 scoring 函数**：能够区分真实声明与 plausible distractors。
- **动态 filtering 策略**：根据 query 类型或上下文调整过滤强度，避免一刀切。
- **端到端训练框架**：联合优化 generation 与 verification，而非两阶段 pipeline。
- **更高效的 verifier 架构**：探索 MoE、蒸馏、适配器等技术构建专用 factuality verifier。

---

> 💡 **总结一句话**：  
> **Conformal Factuality 在 RAG-LLM 中并非“即插即用”的可靠解决方案** —— 它在高 factuality 下常导致 vacuous 输出，且其统计保证在现实中脆弱不堪；而更轻量、更高效的 **entailment-based verifier** 反而提供了更优的性价比选择。未来应转向兼顾 **robustness、usefulness 与 efficiency** 的新型可靠性保障范式。

</details>

---

### 5. [Looking for (Genomic) Needles in a Haystack: Sparsity-Driven Search for Identifying Correlated Genetic Mutations in Cancer](https://arxiv.org/abs/2603.16721)

**Authors**: Ritvik Prabhu, Emil Vatai, Bernard Moussad, Emmanuel Jeannot, Ramu Anandakrishnan, Wu-chun Feng, Mohamed Wahib  
**Category**: cs.DC  
**Published**: 2026-03-18  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.16721v1  

#### Abstract
Cancer typically arises not from a single genetic mutation (i.e., hit) but from multi-hit combinations that accumulate within cells. However, enumerating multi-hit combinations becomes exponentially more expensive computationally as the number of candidate hit gene combinations grow, i.e. on the ord...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Looking for (Genomic) Needles in a Haystack: Sparsity-Driven Search for Identifying Correlated Genetic Mutations in Cancer*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
癌症通常不是由单个基因突变（"hit"）引起，而是由多个基因突变的组合（multi-hit combinations）共同驱动。然而，随着“hit”数量 $ h $ 的增加，候选基因组合的数量呈组合爆炸增长（例如，$ \binom{20,000}{4} \sim 6.4 \times 10^{15} $），使得传统的**Weighted Set Cover (WSC)** 方法在计算上不可行，尤其是在 $ h \geq 4 $ 时。

现有方法面临以下困境：
- **穷举法**：准确但计算成本极高（如4-hit需500年CPU时间）
- **图聚类方法（如 BiGPICC）**：速度快但可能遗漏关键组合，牺牲准确性

### 提出的新方法与创新思路
本文提出了一种名为 **Pruned Depth-First Search (P-DFS)** 的算法框架，其核心思想是利用肿瘤突变数据的高度稀疏性（sparsity）来大幅剪枝搜索空间。

#### 主要创新点：
- **P-DFS 剪枝机制**：
  - 利用突变矩阵中大多数条目为0的特性（平均稀疏度达 **95.61%**）
  - 在深度优先搜索过程中维护一个运行中的位集交集（bitwise AND）
  - 一旦某个部分组合的交集为空（即没有样本同时携带这些突变），立即回溯并剪枝整个子树
- **稀疏性感知预处理**：
  - 将基因按突变频率从低到高排序（即从最稀疏到最密集），使空交集更早出现，提升剪枝效率
- **高性能计算优化**：
  - 使用位运算加速集合操作（intersection, coverage counting）
  - 设计分布式并行架构，采用 **Hierarchical Work Stealing** 和 **Barrier-free Termination Detection** 实现高效负载均衡

### 相比现有方法的优势
| 维度 | 传统 WSC | 图方法（如 BiGPICC） | 本文 P-DFS |
|------|----------|------------------------|-------------|
| 准确性 | 高 | 中等（易漏检） | 高（保留 WSC 评分机制） |
| 可扩展性 | 差（仅支持 ≤3-hit） | 好 | 极好（支持 4-hit 及以上） |
| 效率 | 极低 | 高 | 极高（通过剪枝实现） |

---

## 2. 核心实验方法和设置

### 数据集
- 来源：**The Cancer Genome Atlas (TCGA)**
- 格式：**Mutation Annotation Format (MAF)**
- 处理方式：
  - 过滤沉默突变（silent mutations）
  - 分离肿瘤样本（tumor samples）与正常对照（normal samples）
  - 构建二值化突变矩阵 $ X \in \{0,1\}^{G \times n} $，其中 $ G \approx 20,000 $ 为基因数，$ n $ 为样本数
- 涉及癌种：BLCA、HNSC、BRCA、COAD、LUAD、OV 等共10余种

### 实验设置
- **平台**：日本 **Fugaku 超级计算机**
  - 单节点：Fujitsu A64FX CPU，48核，32GB HBM2内存
  - 并行模型：**MPI**，每核一个 rank
  - 最大规模：**147,456 MPI ranks**（3,072节点）
- **实现细节**：
  - 使用 **bitset 表示法** 存储每个基因的突变模式
  - 所有 rank 复制输入数据（因数据量小，仅几MB）
  - 采用两级通信拓扑（node leader + global communicator）

### 评估指标
| 指标 | 定义 |
|------|------|
| **Pruning Efficiency** | 实际访问的组合数 / 总组合数 |
| **Speedup** | 相对于基线方法的运行时间加速比 |
| **Sensitivity** | 测试集中被覆盖的肿瘤样本比例 |
| **Specificity** | 测试集中未被错误覆盖的正常样本比例 |
| **Solution Size** | 达到全覆盖所需的最小组合数量 |

### 基线方法对比
- **Exhaustive WSC**：完整枚举所有 $ \binom{G}{h} $ 组合，作为黄金标准但不可扩展
- **No-pruning DFS**：无剪枝的深度优先搜索，用于验证剪枝效果
- （隐含对比）**BiGPICC**：图聚类方法，在精度上有妥协

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 数值 |
|------|------|
| **最大加速比** | **~183×**（vs. exhaustive WSC，测于147,456 ranks） |
| **剪枝率（4-hit）** | **90–98%** 的组合被提前剪枝 |
| **实际访问组合占比（4-hit）** | 低至 **0.8%**（如 BLCA 癌症） |
| **端到端运行时间（BLCA, 4-hit）** | 192节点：约133分钟；3,072节点：约20分钟 |
| **估计穷举耗时（3,072节点）** | >52小时（未完成） |

### 与基线方法的对比结果
- **相比 Exhaustive WSC**：
  - 在 **BLCA 4-hit** 上，P-DFS 仅评估 $ 6.42 \times 10^{14} $ 个组合，而理论总数为 $ 6.43 \times 10^{15} $
  - 实现 **近两个数量级的搜索空间压缩**
  - 强扩展性良好：接近理想线性加速（见 Fig. 10）

- **解质量更高**：
  - 如 Table II 所示，在 **BLCA 4-hit** 上，P-DFS 得到的解大小为 **16**，而原始 WSC 为 **19**
  - 平均减少 **80% 的 solution size**，意味着更简洁、更具生物学解释性的致癌组合

### 消融实验结果
- **Work Stealing 的影响**（Fig. 8）：
  - 启用后 worker 运行时间标准差从 **544秒** 降至 **42秒**
  - worker 平均空闲率从 **63.6%** 降至 **22.3%**
  - 显著改善负载均衡，避免长尾效应

- **剪枝有效性随 $ h $ 增加而增强**（Fig. 6）：
  - 2-hit：访问约 68% 组合
  - 3-hit：下降至 ~10%
  - 4-hit：进一步降至 ~1%
  - 表明 **P-DFS 特别适合高阶组合挖掘**

- **强扩展性分析**（Fig. 10）：
  - 4-hit（BLCA）扩展性最好，接近理想加速曲线
  - 2-hit（OV）早期饱和，因其计算量小，通信开销占比上升
  - 验证了“剪枝越多，可并行性越强”的假设

---

## 4. 关键结论和发现

### 主要发现
1. **稀疏性是突破口**：肿瘤突变数据天然高度稀疏（median sparsity: **95.61%**），这一特性可用于设计高效的剪枝策略。
2. **P-DFS 显著降低搜索复杂度**：通过 early termination 和 sparse-first ordering，将原本 NP-hard 的 WSC 枚举转化为可在超算上高效执行的任务。
3. **支持高阶组合发现**：首次实现了对 **4-hit 及以上** 致癌组合的大规模、系统性搜索，突破了以往 $ h \leq 3 $ 的限制。
4. **兼具高精度与高效率**：在保持 WSC 高分类准确性的前提下，实现 **百倍级加速**，且泛化能力优秀（test sensitivity: 0.85–0.98, spec: 0.81–0.99）。

### 方法的局限性
- **依赖稀疏性假设**：若某些癌症类型突变密集（如 hypermutated tumors），剪枝效果可能下降
- **仍属启发式搜索**：虽保留 WSC 评分机制，但剪枝可能导致错过极少数非稀疏路径中的最优解
- **静态编译参数**：`NUMHITS` 和 `BOUND` 需在编译时设定，缺乏动态调整能力
- **数据访问门槛**：依赖 TCGA 控制访问数据，复现实验需审批流程

### 未来工作方向
- 扩展至 **更高阶组合（$ h > 9 $）** 探索极端稀有致癌路径
- 结合 **pathway-level prior knowledge** 进一步引导搜索方向
- 开发 **adaptive pruning thresholds** 以平衡速度与完整性
- 探索 **GPU offloading** 或 **异构加速** 进一步提升单节点吞吐
- 将该框架应用于其他稀疏组合发现任务（如 drug synergy prediction）

---

> ✅ **代码开源地址**：[https://github.com/RitvikPrabhu/P-DFS-Multihit-WSC](https://github.com/RitvikPrabhu/P-DFS-Multihit-WSC)  
> 🔗 支持全部实验复现，包含 preprocessing、distributed execution 与 result verification 脚本。

</details>

---

### 6. [Mask Is What DLLM Needs: A Masked Data Training Paradigm for Diffusion LLMs](https://arxiv.org/abs/2603.15803)

**Authors**: Linrui Ma, Yufei Cui, Kai Han, Yunhe Wang  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.15803v1  

#### Abstract
Discrete diffusion models offer global context awareness and flexible parallel generation. However, uniform random noise schedulers in standard DLLM training overlook the highly non-uniform information density inherent in real-world sequences. This wastes optimization resources on low-density struct...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Mask Is What DLLM Needs: A Masked Data Training Paradigm for Diffusion LLMs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **Discrete Diffusion Language Models (DLLMs)** 在训练中普遍采用**Uniform Random Masking**策略，即对序列中的所有 token 以相同概率进行掩码。这种机制忽略了真实文本中信息密度的高度非均匀性：
- 高信息密度区域（如代码中的控制流语句、数学题中的关键运算步骤）是逻辑推理的核心；
- 低信息密度区域（如语法连接词、标点）更多承担结构粘合作用。

传统方法导致模型在优化时资源分配失衡：过度关注“语法胶水”，而忽视决定任务成败的**逻辑枢纽点**，从而限制了其复杂推理能力。

---

### 🚀 提出的新方法与创新思路

作者提出 **Information Density Driven Smart Noise Scheduler**，核心思想受人类认知学习中的 *Cloze Test* 启发——高效学习应聚焦于重建核心概念而非冗余部分。

#### 主要创新点包括：

1. **信息密度感知的掩码调度机制（Information-Density-Aware Masking）**
   - 引入一个两阶段流程：
     - **Step 1: Info-Dense Region Extraction**  
       利用 LLM 或规则提取高信息密度片段（如代码中的 `if/while` 条件判断、算法关键节点；数学中的中间结果、核心公式）。
     - **Step 2: Complementary Priority Masking**  
       对这些高密度区域赋予更高的掩码优先级（通过 bias weight $ w > 1 $ 控制），实现“有选择地制造挑战”。

2. **互补掩码解耦训练范式（Density-Based Decoupling）**
   - 将每个样本复制为两个互补视图：
     - **Logical Sample**：优先掩码信息密集区 → 强迫模型完成深度逻辑推导。
     - **Syntactic Sample**：保留逻辑骨架，掩码其余结构部分 → 强化语言流畅性和语法一致性。
   - 实现单一训练实例的双重监督目标，提升数据利用率和训练稳定性。

3. **轻量级、低成本的数据增强方式**
   - 只需对少量数据（如10%）进行离线标注即可显著提效，无需修改模型架构或增加参数量。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（Uniform Random Masking） | 本文方法（Density-Driven + Complementary Masking） |
|------|------------------------------------|--------------------------------------------------|
| 掩码策略 | 输入无关、各位置等概率掩码 | 基于信息密度动态调整掩码优先级 |
| 学习目标 | 单一重建任务 | 解耦为逻辑推理 + 结构生成双目标 |
| 数据效率 | 全量标注成本高 | 极低比例标注即可获得显著收益 |
| 性能表现 | 易出现 contextual collapse | 更平滑的 ELBO 优化路径，避免训练崩溃 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **Code Domain**:  
  - `OPC-SFT-Stage2` (~450K 样本)，用于代码生成与理解任务。
- **Math Domain**:  
  - `GSM8K` 和 `MATH500`，涵盖小学到高中水平的数学应用题，测试多步推理能力。

最终微调数据为两者混合，总计约 450K 样本，训练一个 epoch。

---

### ⚙️ 实验设置

- **基础模型**：LLaDA-2.0-mini（基于 diffusion 的 LLM）
- **框架**：dFactory
- **最大序列长度**：2048
- **Batch Size**：全局 16
- **Block Diffusion 设置**：
  - Block size: 32
  - 噪声率 $ \alpha_t \in [0.3, 0.8] $
  - 时间步数 $ T = 32 $
- **测试设置**：
  - 最大生成长度：512
  - 解码步数：32

---

### 🎯 评估指标

- **Pass@1 / Accuracy**：
  - Code 任务：HumanEval、MBPP
  - Math 任务：GSM8K、MATH500
- **平均得分（AVG）**：四项任务的平均准确率

---

### 🔁 基线方法对比

| 方法 | 描述 |
|------|------|
| **Original** | 未经 SFT 微调的原始模型 |
| **Baseline (w=1)** | 使用相同数据但采用 uniform random masking 的标准训练方式 |
| **Ours (w=2)** | 提出的方法，引入信息密度驱动掩码 + 互补采样机制 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1）

| Method | HumanEval | MBPP | GSM8K | MATH500 | **AVG** |
|--------|-----------|------|-------|---------|--------|
| Original | 50.00 | 55.00 | 86.58 | 40.80 | 58.10 |
| Baseline | 57.93 | 56.80 | 69.14 | 37.40 | **55.32** |
| **Ours (w=2)** | **65.24** | 54.00 | **73.92** | **43.60** | **59.19** |

> 💡 **平均提升 +3.87%**，尤其在复杂推理任务上增益明显：
> - HumanEval ↑ +7.31%
> - MATH500 ↑ +6.20%

---

### 🔍 消融实验结果

#### (1) **Bias Weight $ w $ 的影响（Figure 3）**

- 当 $ w = 1 $：退化为 baseline。
- $ w = 2 $ 与 $ w = 0.5 $ 表现接近且最优（AVG ~59.2%），验证了互补掩码下的**分布对称性**。
- 过强先验（$ w=5 $ 或 $ w=0.1 $）导致性能下降至 ~56.05，说明极端偏置会破坏 ELBO 优化。

> ✅ 最佳值在 $ w \approx 2 $ 左右，体现“适度引导”优于强制约束。

---

#### (2) **Soft vs. Hard Priority Masking（Table 2）**

| 类型 | 描述 | AVG（典型值） |
|------|------|----------------|
| Hard Sample ($ w \to \infty $) | 确定性优先覆盖所有关键区域 | ~57.35 |
| **Soft Priority ($ w=2 $)** | 概率性优先掩码（本文方法） | **~59.45** |

> ❗ 发现：确定性硬掩码易引发 **contextual collapse** —— 因关键信息连续成块被遮蔽，造成梯度陡峭、训练不稳定。

---

#### (3) **数据缩放效应（Data Scaling Effect）**

| Code 数据处理比例 | 方法 | AVG |
|--------------------|------|-----|
| 10% | Soft Priority | **59.45** |
| 30% | Soft Priority | 58.22 |
| 100% | Soft Priority | 58.48 |

> 🔍 关键发现：
> - 仅处理 **10% 的数据**就能达到峰值性能；
> - 处理比例越高，反而出现**领域偏移加剧**现象（如 MATH500 下降），表明过度注入 code prior 会影响泛化。

---

#### (4) **是否使用互补掩码（Figure 4）**

| 设置 | 是否互补 | 最优 AVG |
|------|----------|---------|
| 完整方法 | 是 | **59.19** |
| 无互补掩码 | 否 | ~58.5（$ w=0.5 $）|

> ✅ 证明：**Complementary Masking 是不可或缺的组件**，它打破了分布不对称问题，并提升了整体优化稳定性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **信息密度不均是 DLLM 训练效率瓶颈的根本原因之一**。
2. **引入信息密度感知的掩码策略可显著提升模型在 Code/Math 上的零样本推理能力（+4% 平均提升）**。
3. **Complementary Priority Masking 成功将训练目标解耦为“逻辑推理”与“语法连贯”两条路径，协同促进模型全面发展**。
4. **Soft Probabilistic Masking 能有效缓解 block diffusion 中的 contextual collapse 问题，优于 deterministic hard masking**。
5. **极低比例（如10%）的高质量标注即可带来巨大收益，具备极高部署性价比**。

---

### ⚠️ 方法的局限性

1. **依赖外部工具进行 info-dense 区域标注**：
   - 当前使用 GPT-4o 等 black-box LLM 提取关键段落，存在成本和可控性问题。
2. **领域适配性强但通用性待验证**：
   - 规则设计针对 code/math 场景，扩展至其他领域（如对话、摘要）需重新定义“信息密度”标准。
3. **未端到端联合训练**：
   - 提取模块与扩散模型分离，尚未实现自适应、可学习的动态掩码网络。

---

### 🔮 未来工作方向

1. **构建完全自包含的闭环系统**：
   - 探索基于 **Abstract Syntax Tree (AST)** 的规则匹配（适用于 code）；
   - 设计 **end-to-end learnable masking module**，利用模型自身 loss landscape 动态识别 high-density hubs。
2. **探索模型内生的信息密度探测机制**：
   - 利用 confidence score、gradient norm 或 attention entropy 自动定位关键 token。
3. **拓展至多模态 diffusion model**：
   - 将该范式推广至图文、音视频等跨模态场景，研究 multimodal information density 的建模方式。

---

## ✅ 总结一句话

> 本文提出了首个面向 **Diffusion LLMs** 的 **信息密度感知掩码训练范式（Masked Data Training Paradigm）**，通过 **Complementary Priority Masking** 实现逻辑与结构的协同优化，在几乎不增加成本的前提下，显著提升了模型在复杂推理任务上的表现，为下一代高效推理型语言模型提供了新思路。

</details>

---

### 7. [Discovery of interaction and diffusion kernels in particle-to-mean-field multi-agent systems](https://arxiv.org/abs/2603.15927)

**Authors**: Giacomo Albi, Alessandro Alla, Elisa Calzola  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.15927v1  

#### Abstract
We propose a data-driven framework to learn interaction kernels in stochastic multi-agent systems. Our approach aims at identifying the functional form of nonlocal interaction and diffusion terms directly from trajectory data, without any a priori knowledge of the underlying interaction structure. S...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Discovery of interaction and diffusion kernels in particle-to-mean-field multi-agent systems**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
本文研究的是**随机多智能体系统（stochastic multi-agent systems）中交互核（interaction kernel）和扩散核（diffusion kernel）的联合识别问题**。在实际场景中，以下挑战普遍存在：
- **交互结构不可观测**：个体之间的配对交互（pairwise interactions）未被直接记录；
- **轨迹数据有限**：仅有少量甚至单条轨迹可用；
- **系统具有内在随机性**：动力学受噪声驱动，增加了建模难度。

传统方法通常假设完全可观测或依赖大量独立轨迹进行统计推断，难以应对上述现实约束。

---

### **提出了什么新方法或新思路**
作者提出了一种**基于稀疏回归（sparse regression）的数据驱动框架**，用于从部分观测的轨迹数据中同时学习非局部的漂移（drift）和扩散项。其核心创新在于两种互补的策略：

#### （1）**Random-Batch 回归方法（RB Method）**
- 利用 **random-batch sampling** 近似未知的交互矩阵 $ S^n $，通过多次采样构建多个候选估计器；
- 引入加权机制（weighting schemes），根据重构轨迹与真实轨迹的一致性选择最优权重：
  - **Averaging Rule**：对所有 $ K $ 次采样的结果加权平均；
  - **Best-fit Rule**：仅保留误差最小的一次采样结果。
- 该方法在计算上高效，并能补偿潜在交互带来的不确定性。

#### （2）**Mean-Field 重构方法**
- 不再模拟微观交互，而是利用轨迹重建经验粒子密度 $ f(x,t) $；
- 将原始的离散交互问题转化为一个**连续非局部回归问题**，即基于均场近似下的积分形式设计回归矩阵；
- 更好地捕捉宏观统计结构，尤其适用于高密度区域的信息丰富场景。

这两种策略分别从“微观采样”和“宏观逼近”的角度处理缺失交互信息的问题，形成互补。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **适用性更强** | 支持**单条轨迹学习**，无需多组独立初始条件；支持**潜变量交互结构**的学习。 |
| **模型可解释性高** | 输出为解析函数形式的 kernel 表达式（如分段线性函数），而非黑箱模型。 |
| **鲁棒性强** | 在噪声强、数据稀疏、边界信息不足等情况下仍保持稳定性能。 |
| **理论保障** | 提供了关于重构轨迹的先验误差估计（a priori error estimate），证明了方法稳定性。 |

相较于 SINDy 类方法（仅适用于确定性系统）、PINNs（缺乏显式表达、训练成本高）以及早期 kernel learning 方法（需完整交互观测），本方法更具实用性与扩展性。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
所有实验均基于**合成数据（synthetic data）**生成，来源于以下典型多智能体模型的动力学模拟：
- **Bounded Confidence Model**（意见动力学）
- **Attraction-Repulsion Dynamics**（聚集-排斥行为）
- **Cucker-Smale-type Interaction**（长程衰减影响）
- **Nonlocal & Anisotropic Diffusion**（各向异性扩散）

系统规模：$ N = 10^3 \sim 10^5 $ 个 agent  
时间步数：$ M = 201 $ 个 snapshot，时间间隔 $ \Delta t = 0.01 \sim 0.05 $

---

### **实验设置和评估指标**

#### **设置参数**
- 使用 **piecewise linear basis functions** 构造有限维函数空间（如 $ N_b = 8 \sim 21 $）；
- 考虑不同训练窗口长度 $ l $ 和样本数量 $ M_p $，测试方法在短时序下的泛化能力；
- 所有方法仅使用**一条轨迹**进行训练，验证其小样本适应性。

#### **评估指标**
1. **Kernel 重构误差（相对误差）**
   $$
   E_1 := \frac{\|F - \hat{F}\|_{L^1}}{\|F\|_{L^1}}, \quad E_\infty := \frac{\|F - \hat{F}\|_{L^\infty}}{\|F\|_{L^\infty}}
   $$
   分别衡量 $ P(r) $ 和 $ D(r) $ 的逼近精度。

2. **轨迹重构误差**
   - 时间平均 Wasserstein 距离：
     $$
     E_{\text{ave}} := \frac{1}{M} \sum_{n=1}^{M} W_1(\mu_n, \hat{\mu}_n)
     $$
   - 最终时刻误差：
     $$
     E_{\text{fin}} := W_1(\mu_{NT}, \hat{\mu}_{NT})
     $$

3. **二维情形额外指标**
   - 密度图上的 $ L^1 $ 归一化误差：
     $$
     \mathcal{E}_f := \frac{\|\mathbf{f}^N_T - \hat{\mathbf{f}}^N_T\|_{L^1([-1,1]^2)}}{\|\mathbf{f}^N_T\|_{L^1([-1,1]^2)}}
     $$

---

### **基线方法对比**
文中没有引入外部第三方方法作为 baseline，而是将提出的三种实现方式互为对照：
- **Algorithm 1（Random Batch） + Averaging Rule**
- **Algorithm 1（Random Batch） + Best-fit Rule**
- **Algorithm 2（Mean-Field Approach）**

此外，在部分实验中也与理想情况（已知交互矩阵）的结果进行比较，以量化“信息缺失”带来的性能损失。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据汇总**

| 测试 | 方法 | $ E_1(P) $ | $ E_\infty(P) $ | $ E_1(D) $ | $ E_\infty(D) $ | $ E_{\text{ave}} $ | $ E_{\text{fin}} $ |
|------|-------|------------|------------------|------------|------------------|--------------------|---------------------|
| Test 1 (Bounded Conf.) | Mean-Field | 1.01e-1 | 5.28e-1 | — | — | 6.39e-3 | 1.02e-2 |
| Test 2 (Attraction-Repulsion) | Mean-Field | 6.39e-2 | 1.50e-1 | ~3.87e-3 | ~3.71e-3 | — | — |
| Test 3 (Cucker-Smale) | Mean-Field | 4.42e-2 | 4.65e-2 | 1.53e-3 | 3.66e-2 | — | — |
| Test 4 (2D Anisotropic) | Mean-Field | 1.62e-1 | 1.47e-1 | ~4.40e-2 / ~3.47e-2 | — | — | ~1.37e-1 |

> 注：以上数值来自 Tables 2–6，代表典型设置下（如 $ S=3 $）的最佳表现。

---

### **与基线方法的对比结果**
- **Random-Batch vs Mean-Field**：
  - 在大多数测试中，**Mean-Field 方法取得最低误差**，尤其是在非局部扩散和边界区域表现更优；
  - Random-Batch 方法在中等距离范围内有效，但在远距离交互稀少时出现明显偏差（如 Test 3 中 averaging 方法无法恢复大 $ r $ 处的 $ D(r) $）；
  - **Mean-Field 利用全局密度估计缓解了采样偏差问题**。

- **Averaging vs Best-fit Rule**：
  - 两者总体性能相近，但 **best-fit 在某些设置下略胜一筹**（如 Test 2）；
  - Averaging 更稳定，best-fit 对异常值敏感但可能选出更优解。

- **不同训练窗口 $ l $ 的影响**：
  - 增大 $ l $（即减少时间分辨率）会略微增加误差，但整体保持稳健；
  - 表明方法可在低频观测下依然有效。

---

### **消融实验结果**
虽然未明确标注为“ablation study”，但以下设置体现了控制变量分析：
- **是否施加单调性约束**（monotonicity constraints via $ K_p $）：
  - 施加后显著提升 kernel 的物理合理性与收敛速度；
- **是否固定边界值**（如 $ D(\pm1)=0 $）：
  - 可改善边界处的过拟合现象；
- **basis 函数数量变化**：
  - 增加 $ N_b $ 可提高精度，但也可能导致振荡，需结合正则化。

这些设计共同提升了模型的**稳定性与可解释性**。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **即使在交互结构完全未知的情况下，也能准确重构 interaction 和 diffusion kernels**；
2. ✅ **两种策略（Random-Batch 与 Mean-Field）均有效，且精度相当**，说明该框架具有良好的灵活性；
3. ✅ **Mean-Field 方法在处理非局部扩散和边界效应方面更具优势**；
4. ✅ **仅需单条轨迹即可完成学习**，满足现实中数据稀缺的应用需求；
5. ✅ **重构轨迹与真实系统演化高度一致**，验证了所学 kernel 的预测能力；
6. ✅ **提供了理论误差界**，证明了重构系统的稳定性（via Gronwall argument）。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **依赖径向对称假设** | 当前方法假设 $ P(x,y) = P(|x−y|) $，限制了对方向依赖或异质交互的建模能力。 |
| **需要合理选择 basis 函数** | 若网格划分不当或函数类不匹配，可能导致欠拟合或过拟合。 |
| **高维状态空间扩展困难** | 随着维度上升，密度估计和 quadrature 近似变得低效，面临“维度灾难”。 |
| **对极端稀疏交互区域敏感** | 如粒子间距离极大时，数据极少导致 kernel 估计不稳定。 |

---

### **未来工作方向**
1. **拓展至神经网络表示**：使用 NN 参数化 interaction kernel，增强表达能力，适应复杂结构；
2. **应用于真实世界数据**：如社会群体行为、动物迁徙、金融市场等实际 multi-agent 场景；
3. **融合不确定量化**：加入贝叶斯框架以提供置信区间；
4. **处理异构 agent 系统**：允许不同类型个体拥有不同的 interaction law；
5. **在线学习与自适应更新**：动态调整 kernel 以响应环境变化。

---

> **总结一句话**：  
> 本文提出了一套**兼具理论严谨性与实用性的 data-driven 框架**，成功实现了在**交互不可见、数据受限、系统随机**的严苛条件下，对 multi-agent 系统中 interaction 与 diffusion kernels 的联合识别，为复杂系统建模提供了强有力的工具。

</details>

---

### 8. [NextMem: Towards Latent Factual Memory for LLM-based Agents](https://arxiv.org/abs/2603.15634)

**Authors**: Zeyu Zhang, Rui Li, Xiaoyan Zhao, Yang Zhang, Wenjie Wang, Xu Chen, Tat-Seng Chua  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.15634v1  

#### Abstract
Memory is critical for LLM-based agents to preserve past observations for future decision-making, where factual memory serves as its foundational part. However, existing approaches to constructing factual memory face several limitations. Textual methods impose heavy context and indexing burdens, whi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NextMem: Towards Latent Factual Memory for LLM-based Agents

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 LLM-based agents 在管理 **factual memory**（事实记忆）时面临显著挑战：
- **Textual Memory** 方法将记忆以文本形式存储，虽然可读性强，但会带来巨大的 **context length 开销** 和 **indexing 复杂度**，限制了长时记忆的扩展性。
- **Parametric Memory** 方法通过修改模型参数来存储信息，容易引发 **catastrophic forgetting**（灾难性遗忘），且难以精确保存细粒度的事实细节。

因此，如何在保证 **lossless preservation**（无损保留）的前提下，实现高效、可扩展的事实记忆机制，是当前研究的关键瓶颈。

### 提出的新方法与思路
本文提出了 **NextMem** —— 一种基于 **autoregressive autoencoder** 的 **latent factual memory** 框架，其核心思想是：
- 将原始文本编码为紧凑的 **latent representations**（潜在表示），从而大幅压缩存储空间；
- 通过可逆的解码过程，将 latent 表示高保真地重建为原始文本，确保信息无损；
- 设计了两阶段训练策略以稳定优化过程，并引入量化技术进一步降低存储开销。

#### 主要创新点：
1. **Autoregressive Reconstruction Alignment**  
   第一阶段训练中，模型学习从输入文本自回归地重建输出文本，建立“文本→文本”的映射能力，为后续 latent 编码奠定基础。

2. **Progressive Latent Substitution**  
   第二阶段逐步用 latent embeddings 替换输入文本中的前缀块（block-by-block），迫使模型依赖 latent 表示恢复被遮蔽的内容，从而实现“文本→latent→文本”的可逆转换。

3. **Quantization for Storage Efficiency**  
   引入 **4-bit NF4 (NormalFloat) quantization** 对 latent 表示进行压缩，显著减少存储占用，同时保持重建质量几乎不变。

4. **Unified Framework for Storage and Retrieval**  
   latent memory 不仅可用于重建，还可直接作为 **retrieval index** 使用（如计算 cosine similarity），实现了 **memory storage 与 retrieval 的统一**，简化系统架构。

### 相比现有方法的优势
| 维度 | NextMem | Textual Memory | Parametric Memory |
|------|--------|----------------|--------------------|
| 存储效率 | ✅ 高（latent 压缩 + 量化） | ❌ 低（全文本存储） | ⚠️ 中等（需额外参数） |
| 重建精度 | ✅ 极高（接近完美还原） | ✅ 完美 | ❌ 易失真 |
| 可检索性 | ✅ 支持向量检索 | ✅ 支持关键词/语义检索 | ❌ 困难 |
| 扩展性 | ✅ 良好（固定长度 latent） | ❌ 受限于 context window | ✅ 可扩展但成本高 |
| 抗噪鲁棒性 | ✅ 强（实验证明对噪声不敏感） | ✅ 强 | ❌ 弱 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖多个与 agent memory 密切相关的 benchmark，涵盖不同任务类型和上下文长度：

| 数据集 | 描述 |
|-------|------|
| **SQuAD** | 单跳问答，测试阅读理解能力 |
| **HotpotQA** | 多跳推理 QA，要求跨文档整合信息 |
| **RACE** | 英语考试类阅读理解，强调逻辑推理 |
| **LoCoMo** | 多轮对话模拟，评估长期记忆一致性 |
| **LongMemEval** | 用户-助手交互场景下的长期记忆评测 |

### 实验设置与评估指标

#### 三大核心任务设计：
| 任务 | 目标 | 评估方式 |
|------|------|----------|
| **Task 1: Factual Reconstruction**（记忆存储） | 测试 latent memory 是否能准确重建原文 | 使用 F1, ROUGE-1/L, METEOR, BLEU, BertScore |
| **Task 2: Contextual Generation**（记忆利用） | 测试 memory 是否支持下游生成任务 | LLM-as-Judge 判断生成答案准确性（Accuracy） |
| **Task 3: Dense Passage Retrieval**（记忆检索） | 测试 latent 向量是否适合作为检索索引 | Hit@5, Recall@5, MRR@5, MAP@5, DCG@5, NDCG@5 |

#### 模型配置
- **Backbone**: Qwen3-8B
- **Latent Length**: 15 tokens
- **Chunk Size**: 128 tokens
- **Training**: LoRA 微调（rank=16, α=32）
- **Quantization**: 4-bit NF4 + FP8 scale vectors
- **Training Stages**: 先 reconstruction alignment，再 progressive latent substitution（共15步）

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **DeepSeek-OCR** | Vision-based Compression | 将文本转图像后由 LLM 解码 |
| **ICAE** | Context Compression | 使用可学习 memory slot 压缩段落 |
| **DyPRAG** | Parametric Memory | 在测试时生成 LoRA adapter 注入知识 |
| **Textual Memory** | Oracle Baseline | 直接提供原始参考文本 |
| **BGE** | Retrieval-only Encoder | 用于 dense retrieval 对比（不可重建） |

---

## 3. 主要实验结果和性能指标

### Task 1: Factual Reconstruction 结果（关键性能数据）
| 方法 / 数据集 | HotpotQA (F1) | RACE (F1) | SQuAD (F1) | LoCoMo (F1) | LongMemEval (F1) |
|--------------|---------------|-----------|------------|-------------|------------------|
| DyPRAG       | 0.0305        | 0.0696    | 0.0493     | 0.0901      | 0.1338           |
| DeepSeek-OCR | 0.4540        | 0.4068    | 0.3657     | 0.5179      | 0.4685           |
| ICAE         | 0.7890        | 0.6077    | 0.7084     | 0.6986      | 0.7015           |
| **NextMem-Dense** | **0.9820**    | **0.8552**| **0.8920** | **0.9611**  | **0.9436**       |
| **NextMem-Sparse** | **0.9805**    | **0.8554**| **0.8860** | **0.9615**  | **0.9362**       |

✅ **结论**：NextMem 在所有数据集上均取得压倒性优势，F1 分数普遍超过 0.85，部分接近 0.98；即使经过量化（Sparse），性能损失极小。

---

### Task 2: Contextual Generation 结果
| 方法 / 设置 | HotpotQA (DeComp.) | SQuAD (DeComp.) | LoCoMo (DeComp.) | LongMemEval (DeComp.) |
|-----------|--------------------|------------------|-------------------|------------------------|
| ICAE      | 0.8229             | 0.7066           | 0.5215            | 0.5029                 |
| **NextMem-Dense** | **0.8072**         | **0.7572**       | **0.5407**        | **0.5400**             |
| **NextMem-Sparse** | **0.8184**         | **0.7630**       | **0.5263**        | **0.5486**             |
| Oracle (Textual) | 0.9350           | 0.9335           | 0.6986            | 0.6971                 |

📌 **观察**：
- ICAE 在 Compression 模式下表现最好，说明其 latent 更适合直接推理；
- NextMem 在 Decompression 模式下更优，表明其 **重建质量更高**，更适合需要完整信息的任务；
- 存在 trade-off：**reconstruction fidelity vs. direct usability**

---

### Task 3: Dense Passage Retrieval 性能
| 方法 / 数据集 | HotpotQA (MRR@5) | LoCoMo (MRR@5) | LongMemEval (MRR@5) |
|--------------|------------------|----------------|---------------------|
| ICAE         | 0.3187           | 0.0577         | 0.2437              |
| **NextMem-Dense** | **0.5194**       | **0.2418**     | **0.5445**          |
| **NextMem-Sparse** | **0.5107**       | **0.2310**     | **0.5428**          |
| BGE (oracle) | 0.8063           | 0.5061         | 0.6934              |

✅ **结论**：NextMem 显著优于其他可重建模型，在 retrieval 上也展现出强大潜力，接近专用检索模型 BGE 的一半以上性能，而 BGE 无法重建原文。

---

### 消融实验（Ablation Study）结果（RACE 数据集）

| 方法 | F1 | ROUGE-L | METEOR | BertScore |
|------|----|---------|--------|-----------|
| **Dense (Full)** | **0.8552** | **0.8580** | **0.8691** | **0.9735** |
| w/o ST ([SoD] token) | 0.3799 | 0.3804 | 0.4048 | 0.7307 |
| w/o PT (no progressive training) | 0.0159 | 0.0138 | 0.0169 | 0.7686 |
| w/o PS (no progressive substitution) | 0.7389 | 0.7358 | 0.7353 | 0.9502 |
| w/o SQ (no scale in quantization) | 0.0309 | 0.0290 | 0.0442 | 0.7521 |

🔍 **分析**：
- 移除 `[SoD]` 导致性能暴跌 → 该 token 是 latent 解码的关键锚点；
- 移除 progressive training 几乎崩溃 → 证明分阶段训练对稳定性至关重要；
- 移除 progressive substitution 明显下降 → 逐步替换策略有效提升泛化；
- 量化中移除 scale 向量导致严重退化 → scale 是保持精度的核心组件。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Latent factual memory 是可行且高效的范式**  
   NextMem 成功实现了 **高保真、可逆、紧凑** 的事实记忆表示，解决了传统方法在效率与完整性之间的权衡难题。

2. ✅ **latent 表示兼具存储与检索功能**  
   latent vectors 可用于高质量重建，也可直接用于 retrieval，实现 **storage-retrieval unified architecture**，降低系统复杂度。

3. ✅ **progressive training 策略至关重要**  
   一次性训练难以收敛，而 **progressive latent substitution** 能有效引导模型学会 latent-to-text 映射。

4. ✅ **robustness 与 extrapolation 能力强**  
   - 对 Gaussian noise 具有较强鲁棒性（σ≤0.8 时性能稳定）；
   - 能处理超出训练长度的输入（up to 300+ tokens），展现良好外推能力；
   - 量化后性能损失可忽略（NF4 + FP8）。

5. ✅ **semantic structure preserved in latent space**  
   实验显示 latent memory 具备 **spatial-semantic alignment**，即特定位置对应特定语义片段，有利于 fine-grained editing。

---

### 局限性
1. ❌ **direct utilization 能力较弱**  
   在 compression 模式下性能不如 ICAE，说明 latent space 尚未完全优化用于直接推理。

2. ❌ **latent length 固定可能限制灵活性**  
   当前 latent 固定为 15 tokens，对于极短或极长文本可能存在冗余或不足。

3. ❌ **依赖预训练 backbone 和 LoRA 适配器**  
   模型效果受 backbone 影响较大，迁移至小模型的效果尚待验证。

4. ❌ **failure cases in early versions**  
   如 mixture-of-experts、RQ-VAE 等稀疏化尝试失败，说明 latent space 训练本身具有挑战性。

---

### 未来工作方向
1. 🔮 探索 **latent space 的指令跟随能力**（instruction-following in latent space），缩小 compression 与 decompression 模式的差距；
2. 🔁 研究 **双向 editable memory**，允许对 latent memory 进行局部编辑而不影响整体；
3. 🔄 设计 **adaptive latent length** 机制，根据输入动态调整压缩率；
4. 🧠 结合 preference/experience memory，构建 **multi-level memory system**；
5. 💾 探索更高效的 **sparse quantization 或 pruning 方法**，进一步降低部署成本。

---

> ✅ **开源信息**：作者已公开代码与模型 checkpoint：[https://github.com/nuster1128/NextMem](https://github.com/nuster1128/NextMem)

</details>

---

### 9. [Learning to Present: Inverse Specification Rewards for Agentic Slide Generation](https://arxiv.org/abs/2603.16839)

**Authors**: Karthik Ragunath Ananda Kumar, Subrahmanyam Arunachalam  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.16839v1  

#### Abstract
Automated presentation generation remains a challenging task requiring coherent content creation, visual design, and audience-aware communication. This work proposes an OpenEnv-compatible reinforcement learning environment where LLM agents learn to research topics, plan content, and generate profess...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning to Present: Inverse Specification Rewards for Agentic Slide Generation*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
自动化演示文稿生成（Presentation Generation）是一个复杂的多阶段任务，涉及**主题研究、内容规划、视觉设计、叙事连贯性和受众适配**。现有方法通常缺乏系统性的训练框架和可解释的质量评估机制，导致生成结果在逻辑性、忠实度和美学上难以保证。

本文旨在解决以下挑战：
- 复杂的**动作空间**（14个工具，需参数化调用）
- 多维度质量评估难题（结构、内容、美学、一致性等）
- 缺乏对“是否准确传达原始意图”的**整体性评估信号**

---

### 🚀 提出的新方法与创新点

#### （1）**OpenEnv-Compatible RL Environment**
- 构建了一个支持完整演示文稿创作流程的强化学习环境（从 research 到 finalize），符合 OpenEnv 接口标准。
- 定义了五个阶段的工作流：`RESEARCH → PLAN → GENERATE → REFINE → DONE`。
- 提供14个工具，涵盖五大类别：**Research、Content Planning、Design、Structure、Meta**。

#### （2）**多组件奖励系统（Multi-Component Reward System）**
将质量分解为六个正交维度，实现**可解释、可调节的评估**：
| 组件 | 描述 |
|------|------|
| `code_rules` | 结构合规性（如标题、段落数量） |
| `render_quality` | HTML 渲染成功率与有效性 |
| `aesthetic_html` | LLM 对 HTML/CSS 设计质量评分 |
| `aesthetic_visual` | LLM 对渲染后 PNG 视觉效果评分 |
| `content_quality` | 内容相关性、事实依据、唯一性、叙事流畅性 |
| `spec_reconstruction`（**核心创新**） | **逆向规范重建奖励** |

#### （3）**逆向规范奖励（Inverse Specification Reward）**
- **新思路**：提出一种“逆任务”形式的奖励机制 —— 给定生成的幻灯片，让另一个 LLM 尝试仅凭输出反推原始需求（brief）。
- 具体预测字段包括：`topic`, `audience`, `num_slides`, `key_themes`
- 通过比较预测与真实 brief 的匹配程度计算得分：
  $$
  \text{recon} = 0.4·S_{\text{topic}} + 0.25·S_{\text{audience}} + 0.15·S_{\text{count}} + 0.20·S_{\text{themes}}
  $$
- 这是**首次将输入重建作为奖励信号用于演示文稿生成**，能有效捕捉整体连贯性与意图忠实度。

#### （4）**密集步奖励（Dense Step Rewards）**
- 不采用稀疏的终局奖励，而是基于每一步执行后的**质量增量**（ΔQ）给予即时反馈。
- 形式为：  
  $$
  r_{\text{step}} = (Q_{\text{new}} - Q_{\text{old}}) + r_{\text{action}}
  $$
- 符合 **Potential-Based Reward Shaping** 理论，保留最优策略的同时缓解信用分配问题。

#### （5）**工具调用驱动的多格式输出**
- 模型学会通过调用 `generate_slide` 和 `export_to_pptx` 等工具，自动生成 HTML 和 PPTX 文件，无需针对特定格式进行专门训练。

#### （6）**专家轨迹生成 + GRPO 微调**
- 使用 **Claude Opus 4.6** 生成高质量多轮交互轨迹（tool call 序列），作为监督信号。
- 在 Qwen2.5-Coder-7B 上使用 **GRPO（Group Relative Policy Optimization）** 进行微调，仅更新 **0.5% 参数（约40M）**，显著提升效率。

#### （7）开源数据集 SlideRL
- 发布包含 **288 条完整轨迹**（48个brief × 6种模型）的数据集，含每步 tool call、状态、奖励、最终评分。
- 地址：[https://huggingface.co/datasets/KarthikRagunathAnandaKumar/sliderl-multi-turn-rollouts](https://huggingface.co/datasets/KarthikRagunathAnandaKumar/sliderl-multi-turn-rollouts)

---

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **评估方式** | 多维可解释奖励 > 单一指标；引入逆向任务评估整体一致性 |
| **训练效率** | GRPO + LoRA 实现高效微调（仅0.5%参数） |
| **泛化能力** | 学会使用工具链，支持多输出格式（HTML/PPTX） |
| **行为控制** | 密集奖励引导探索，避免无效动作累积 |
| **开放性** | 完整开源环境、代码、数据集，推动社区发展 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **48 个多样化的商业演示简报（business presentation briefs）**
  - 类型覆盖：财务报告、融资路演（Series A/B）、市场分析（EV、云计算、金融科技）、技术评审（网络安全、MLOps）、战略规划（并购、产品路线图）
  - 变量丰富：目标页数（6–10）、受众（董事会、VC、高管、工程师）、置信度（0.3–1.0）、内容类型（结构化 vs 开放式）

---

### ⚙️ 实验设置

#### 模型列表（共6个）
| 模型 | 类型 | 参数规模 |
|------|------|----------|
| **Fine-tuned (Ours)** | LoRA 微调 Qwen2.5-7B | 7B（0.5%可训练） |
| Base Qwen 7B | 原始指令模型 | 7B |
| Claude Opus 4.6 | 商业闭源模型 | 未公开 |
| Claude Sonnet 4.6 | 商业闭源模型 | 未公开 |
| Llama 4 Scout | 开源权重 | 109B（激活17B） |
| GPT OSS 120B | 开源权重 | 120B |

> 所有模型在相同环境下运行，使用相同的 reward pipeline。

#### 评估协议
1. 加载一个 brief
2. 执行最多 35 轮交互（turns）
3. 使用 multi-component reward system 计算综合质量分
4. 输出 `.html` 和 `.pptx` 文件供人工审查

#### 评估指标
- **总体质量得分（Overall Quality Score）**：六项加权平均
- 各子项得分（per-component scores）
- 成功率（Completion Rate）
- 平均生成页数、耗时、步数

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

| 模型 | 总体质量 | 完成率 | 平均时间/简报 |
|------|--------|--------|----------------|
| **Fine-tuned (Ours)** | **0.724** | **95.8%** | 71.6s |
| Base Qwen 7B | 0.544 | 70.8% | 43.8s |
| **Claude Opus 4.6** | **0.794** | 100% | 393.3s |
| Llama 4 Scout | 0.779 | 100% | 155.4s |
| GPT OSS 120B | 0.249 | 31.2% | 66.1s |

> ✅ **我们的微调模型达到 Claude Opus 4.6 的 91.2% 质量水平**

> ✅ **相比基础模型提升 33.1%（0.544 → 0.724）**

> ✅ **完成率从 70.8% 提升至 95.8%**

---

### 🆚 与基线方法对比

| 对比维度 | 结果 |
|--------|------|
| **vs. Base Qwen 7B** | 全面超越，在所有 reward component 上均有提升，尤其 `code_rules`（+36.5%）和 `render_quality`（+35.3%） |
| **vs. Claude Opus 4.6** | 达到其 91.2% 质量，且在 5/48 个 brief 上**反超**（见下表） |
| **vs. Llama 4 Scout** | 达到其 93.0% 质量，结构指标接近，但在 `content_quality` 上仍有差距 |
| **vs. GPT OSS 120B** | 显著优于该大模型（0.724 vs 0.249），揭示**参数数量 ≠ 工具使用能力** |

#### ✅ 在以下5个brief中，我们的模型表现最佳（甚至超过Claude Opus）：

| Brief | 我们的得分 | 第二名（模型） | 差距 |
|------|-----------|---------------|------|
| Cloud Cost Optimization | 0.836 | Sonnet 4.6 (0.788) | +0.048 |
| Content Marketing ROI | 0.826 | Opus 4.6 (0.824) | +0.002 |
| Customer Success Metrics | 0.816 | Opus 4.6 (0.807) | +0.009 |
| B2B Sales Automation | 0.800 | Opus 4.6 (0.770) | +0.030 |
| Edge Computing Analysis | 0.792 | Base Qwen (0.781) | +0.011 |

> 💡 特别值得注意的是：**在12个brief上击败了Claude Opus 4.6（占总数25%）**，而该模型正是用于 aesthetic/content 评分的 LLM-as-Judge，排除了“评委偏见”的可能。

---

### 🔬 消融实验与关键发现

#### （1）GRPO 微调效果显著
- 相比 base model，GRPO 微调带来：
  - +33.1% 总体质量
  - +25pp 完成率
  - 所有 reward components 均提升

#### （2）训练步数与数据规模的影响（Table X）
| 设置 | 步数 | 总体质量 | 完成率 |
|------|-----|----------|--------|
| Curated (3条轨迹) | 100 | 0.623 | 71.2% |
| Curated | 200 | 0.689 | 82.4% |
| Scaled (48条轨迹) | 200 | **0.724** | **95.8%** |
| Scaled | 1000 | 0.0 | 0%（模式崩溃） |

> ✅ 更大规模的数据集有助于早期学习  
> ❌ 但若无 KL 正则化，长期训练会导致**模式崩溃**（policy collapse）

#### （3）参数效率极高
- 仅微调 **0.5% 参数（~40M）**
- 使用 **4-bit量化 + LoRA**，可在单张GPU上完成训练
- 实现与百亿级模型相当的表现

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **指令遵循与工具使用能力比参数量更重要**
   - GPT OSS 120B 因无法正确输出 JSON tool call 而失败（完成率仅31.2%）
   - 表明“大 ≠ 强”，**agentic competence 需要显式训练**

2. **逆向规范奖励是有效的整体质量代理**
   - 能捕捉传统指标忽略的“是否讲清楚了故事”
   - 是首个将其应用于 presentation generation 的工作

3. **多组件奖励架构支持可解释优化**
   - 可定位短板（如 content_quality 或 aesthetic）
   - 支持动态调整权重以适应不同场景

4. **小模型经适当训练可媲美甚至超越大模型**
   - 7B 模型达到 Opus 91.2% 质量，并在部分任务上反超
   - 验证了 **parameter-efficient fine-tuning + RL** 的潜力

5. **GRPO 在非可导、随机 reward 下依然有效**
   - 理论上成立（policy gradient theorem 不要求 reward 可导）
   - 实践中稳定收敛，验证了方法鲁棒性

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **LLM-as-Judge 成本高** | 每步需多次调用 Claude Opus API，增加训练开销 |
| **存在 Reward Hacking 风险** | 如 `review_deck` 工具因恒返回 success 被滥用，引发模式崩溃 |
| **领域特异性强** | 当前 reward 函数针对商业演示优化，迁移到教育或科研需重新校准 |
| **Advantage Estimation 粗糙** | K=2 导致 group-relative advantage 仅为 ±1，信息量有限 |
| **长期训练不稳定** | 无 KL 正则时易发生 policy drift，需 early stopping 或 β > 0 |

---

### 🔮 未来工作方向

1. **扩大 group size K 至 4–8**，获取更精细的 advantage 分布
2. **Reward Model Distillation**：训练轻量本地 reward model 替代 API 调用
3. **引入 KL Regularization (β > 0)**：防止 policy drift，支持长周期训练
4. **防御 Reward Hacking**：
   - 为只读工具添加成本或衰减奖励
   - 引入 repetition penalty
5. **集成人类反馈（Human Feedback）**
6. **扩展至多模态生成**：加入图像合成工具（如 DALL·E、Stable Diffusion）
7. **课程学习（Curriculum Learning）**：从简单到复杂 brief 逐步训练
8. **升级 base model 至 Qwen3**
9. **跨领域迁移逆向奖励范式**：适用于文档摘要、视频脚本等任务

---

## 📎 总结一句话

> 本文提出了一个基于 **GRPO + 多组件奖励 + 逆向规范重建** 的 agentic slide generation 框架，在仅微调 0.5% 参数的情况下，使 7B 级模型达到接近 Claude Opus 4.6 的表现，并首次证明：**在复杂工具调用任务中，行为纪律（instruction adherence）远比模型大小更重要**。

</details>

---

### 10. [Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies](https://arxiv.org/abs/2603.15903)

**Authors**: Nathaniel Imel, Richard Futrell, Michael Franke, Noga Zaslavsky  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.15903v1  

#### Abstract
Natural languages have been argued to evolve under pressure to efficiently compress meanings into words by optimizing the Information Bottleneck (IB) complexity-accuracy tradeoff. However, the underlying social dynamics that could drive the optimization of a language's vocabulary towards efficiency ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文旨在解决语言演化中的两个核心开放问题：

1. **效率的机制来源**：自然语言为何在语义系统中表现出信息论意义上的高效压缩（即在表达准确性和词汇简洁性之间取得最优权衡）？尽管已有研究指出人类语言接近 *Information Bottleneck (IB)* 理论预测的效率边界，但驱动这种效率演化的具体**文化演化机制**尚不清楚。

2. **EGT 与 IB 的关系**：基于 *Evolutionary Game Theory (EGT)* 的信号博弈模型已被广泛用于解释语言如何从零开始通过局部互动涌现，但这些模型是否能产生符合 IB 效率原则的全局最优系统仍未知。

---

### **提出了什么新方法或新思路**

作者提出了一种**统一框架**，将两种理论范式整合：

- **Information Bottleneck (IB)**：提供了一个形式化的目标函数（复杂度-准确性权衡），用以衡量语义系统的效率。
- **Evolutionary Game Theory (EGT)**：通过 *noisy sim-max signaling games* 和 *imprecise conditional imitation dynamic* 建模语言的文化演化过程。

#### 核心创新点包括：

- **首次建立 EGT 动态与 IB 效率之间的桥梁**：证明即使没有显式优化目标，仅通过模仿策略的频率依赖选择（frequency-dependent selection），群体层面的语言也能自发趋近 IB 理论边界。
- **引入“不精确模仿”动态（imprecise imitation）**：该机制允许个体在感知和复制他人行为时存在噪声，更贴近真实社会学习过程，并避免了传统模型对完全理性或固定架构的假设。
- **采用细粒度策略更新机制**：不同于以往基于整套词汇或语法的演化模型，本模型在“状态-信号”配对级别进行策略演化，增强了可解释性和与认知机制的兼容性。

---

### **相比现有方法的优势**

| 方面 | 传统方法局限 | 本文优势 |
|------|--------------|---------|
| **理论基础** | 多数强化学习或多智能体通信研究依赖深度网络架构、超参数调优，缺乏通用性 | 不依赖特定学习算法或神经网络，抽象程度高，更具普适性 |
| **演化机制** | 经典 EGT 模型常假设完美感知、二元奖励，难以刻画模糊性和渐进演化 | 引入感知混淆（state confusion）和连续相似性奖励，更符合现实 |
| **效率解释力** | 以往工作未明确连接 EGT 成功与 IB 信息论最优性 | 明确验证了局部博弈成功可导向全局信息论效率 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **合成语义域（Synthetic Domain）**：  
  使用一个理想化的**数量空间（numerosity space）**作为测试案例：
  - 状态集合 $ X = \{0, 1, ..., 99\} $，共 100 个数值状态。
  - 可用词汇数也为 100，理论上支持一一对应（exact mapping）或粗略划分（coarse-grained categories）。
  - 该设定模拟了如“几个”、“十几”等自然语言数量词的表达方式。

> 注：虽然使用的是人工构造领域，但其结构类比于自然语言中的数量词系统（numeral systems）、颜色命名等典型语义范畴。

---

### **实验设置和评估指标**

#### **动力学模型**
- 采用 **Franke & Correia (2018)** 提出的 *imprecise conditional imitation dynamic*，基于 replicator equation 扩展而来。
- 包含两个混合充分的大群体：**Sender 群体** 和 **Receiver 群体**。
- 每个 agent 拥有确定性策略（pure strategy），集体行为表现为概率分布 $ S(w|x_o) $ 和 $ R(x_o|w) $。

#### **关键参数**
- **$ \gamma $**：控制博弈中“实用精度标准”（pragmatic standard of precision）。值越大表示对接收端还原精度要求越高。
- **$ \sigma $**：感知不确定性参数，决定相近状态被混淆的概率（通过 Gaussian-shaped confusion kernel 实现）。

#### **评估指标**
1. **复杂度 Complexity**：定义为 $ I(M_o; W) $，即编码器输出信号所需的信息量（越低越好）。
2. **准确性 Accuracy**：定义为 $ I(W; X_a) $，即信号携带关于真实世界状态的信息量（越高越好）。
3. **效率损失（Efficiency Loss）**：  
   $ \epsilon = \min_\beta \left( \mathcal{F}_\beta[S] - \mathcal{F}_\beta^* \right) $，衡量当前系统偏离 IB 最优解的程度。
4. **群体适应度（Population Fitness）**：期望相似性 $ \mathbb{E}[\text{sim}(x_a, \hat{x}_a)] $。

#### **可视化工具**
- **信息平面图（Information Plane）**：横轴为复杂度，纵轴为准确性，绘制各演化终点并与 IB 理论边界比较。

---

### **基线方法对比**

| 基线类型 | 描述 |
|--------|------|
| **随机排列控制组（Permuted Controls）** | 将最终收敛系统的词义映射随机打乱，检验效率是否源于结构而非偶然。 |
| **NK99 动态（Nowak & Krakauer, 1999）** | 经典有限群体、对称信号博弈模型，包含采样变异（mutation），但无感知噪声和连续奖励。用于对比不同演化机制的表现。 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

- 在超过 **800 次独立运行**（100 个 $ \gamma $ 值 × 8 随机种子）中，绝大多数系统在 $ 10^5 $ 步内收敛。
- 所有演化出的系统均**非常接近 IB 理论边界**，远优于随机对照组和 NK99 动态产生的系统。
- 最大可达准确度约为 **4.61 bits**（理论最大值），实际最高达到约 **< 4 bits**，表明存在由模仿噪声导致的根本限制。

---

### **与基线方法的对比结果**

| 指标 | 本文模型（FC18 动态） | Permuted 控制组 | NK99 动态 |
|------|------------------------|------------------|-----------|
| **平均效率损失 $ \epsilon $** | 极低（接近 0） | 显著更高 | 更高，且远离 IB 边界 |
| **信息平面上的位置** | 紧贴 IB 边界分布 | 分散于下方区域 | 多数位于左下角（低复杂度、低准确性） |
| **能否实现高准确性** | 能，在高 $ \gamma $ 下逼近上限 | 否 | 否，受限于突变机制和二元奖励 |

> ✅ 结果说明：只有结合**感知混淆 + 渐进奖励 + 不精确模仿**的动力学才能稳定生成高效压缩系统。

---

### **消融实验与关键变量分析**

虽然未严格命名为“消融实验”，但文中系统考察了以下因素的影响：

1. **$ \gamma $（实用精度标准）的作用**：
   - $ \gamma $ 越高 → 系统越倾向于发展出**更复杂、更准确**的词汇系统。
   - $ \gamma $ 与拟合得到的 IB 参数 $ \beta $ 呈现出极强单调关系（Spearman $ \rho = 0.99 $），说明局部博弈压力可调节全局效率权衡。

2. **模仿噪声的影响**：
   - 即使当 $ \gamma \to \infty $（理论上应趋向精确一一映射），系统也无法达到完全双射（bijective mapping）。
   - 表明**不精确模仿本身构成一种正则化机制**，软化类别边界，限制最大精度，但也可能促进泛化。

3. **收敛速度与临界现象**：
   - 在某些 $ \gamma $ 值附近（如 $ \sim 10^{-6}, 10^{-2} $）出现收敛缓慢和效率损失局部升高，暗示可能存在相变行为。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **局部模仿可以驱动全局效率**：  
   即使每个 agent 仅基于频率和收益模仿他人行为，无需全局优化意识，整个群体仍能演化出接近 IB 理论最优的语义系统。

2. ✅ **EGT 成功能导向 IB 效率**：  
   信号博弈中的“成功”（即高相似性回报）与信息论“效率”并非对立，而是可以通过适当的动态机制统一起来。

3. ✅ **实用标准塑造效率格局**：  
   局部语境中的沟通精度期望（$ \gamma $）是决定最终系统落在 IB 曲线上哪个位置的关键参数，为跨语言语义差异提供了潜在解释机制。

4. ⚠️ **噪声限制极限性能**：  
   不精确模仿虽有助于稳定性，但也设定了系统能达到的最大准确性上限，阻止了完全精确系统的演化。

---

### **方法的局限性**

| 局限性 | 说明 |
|-------|------|
| **使用合成领域** | 当前实验基于理想化数量空间，尚未在真实语言数据（如颜色词、亲属称谓）上验证。 |
| **静态环境假设** | 未考虑语义需求分布 $ p(u) $ 的动态变化（如频率漂移、文化变迁）。 |
| **忽略递归与组合性** | 模型聚焦于原子性词汇学习，未涉及语法结构或复合表达能力。 |
| **确定性策略 + 大群体假设** | 忽略小群体中的随机漂变效应，可能低估文化多样性的起源机制。 |

---

### **未来工作方向**

1. **扩展到真实语义领域**：  
   将模型应用于颜色命名、空间指示词、容器分类等具有丰富类型学数据的真实语义系统。

2. **引入非均匀先验 $ p(u) $**：  
   探索高频概念是否更容易形成精细区分，从而进一步贴近 Zipfian 分布等语言统计规律。

3. **结合个体学习偏差**：  
   如 Imel et al. (2025) 发现个体存在偏好 IB-高效系统的归纳偏置，未来可建模这种偏置如何与群体动态交互。

4. **探索数学联系**：  
   进一步研究 replicator dynamics 是否隐式执行某种变分推断或梯度下降，揭示 EGT 与信息论优化之间的深层数学关系。

5. **多层级语言演化建模**：  
   将当前词汇层演化扩展至句法结构、组合规则的共同演化。

---

> 📌 **总结一句话**：  
> 本文证明，**基于不精确模仿的社会学习机制足以在群体层面自发涌现出信息论意义上高度高效的语言词汇系统**，为“为何人类语言既简洁又有效”这一根本问题提供了强有力的机制性解释。

</details>

---

### 11. [Prior-Informed Neural Network Initialization: A Spectral Approach for Function Parameterizing Architectures](https://arxiv.org/abs/2603.16376)

**Authors**: David Orlando Salazar Torres, Diyar Altinses, Andreas Schwung  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.16376v1  

#### Abstract
Neural network architectures designed for function parameterization, such as the Bag-of-Functions (BoF) framework, bridge the gap between the expressivity of deep learning and the interpretability of classical signal processing. However, these models are inherently sensitive to parameter initializat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Prior-Informed Neural Network Initialization: A Spectral Approach for Function Parameterizing Architectures*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统神经网络初始化方法（如 Xavier、Kaiming）是**data-agnostic**（与数据无关）的，假设适用于通用深度架构。然而，在**函数参数化架构**（如 Bag-of-Functions, BoF）中，模型输出直接对应可解释的数学函数（如正弦波、多项式趋势），其参数具有明确物理意义。  
这类模型对初始化极为敏感，随机初始化常导致：
- 收敛缓慢
- 参数漂移严重
- 性能波动大
- 需要大量调参

因此，如何将信号的**内在结构先验**（spectral 和 temporal 特征）融入模型设计与初始化，成为一个关键挑战。

---

### 🚀 提出的新方法与创新思路

本文提出一种**先验引导的神经网络初始化框架**（Prior-Informed Initialization），将信号分析技术与神经网络设计深度融合：

#### （1）基于 FFT 的谱分析指导季节性组件建模
- 使用 **Fast Fourier Transform (FFT)** 分析训练数据，提取主导频率（dominant frequencies）
- 将主导频率的数量 $|K(\tau)|$ 显式用于确定 **BoF 架构中的残差阶段数 $S$**，即模型深度
- 利用频谱系数的均值与方差 $(\mu_{\text{data}}, \sigma_{\text{data}})$ 初始化 **seasonal encoder** 的偏置项，使初始参数分布贴近真实信号

#### （2）基于残差回归的趋势估计与输入维度优化
- 在去除季节性成分后，对残差信号进行 **linear regression** 以估计趋势（slope 和 bias）
- 基于有限样本理论推导出趋势估计的误差界，从而确定 **trend encoder 所需最小输入长度 $N_{\text{in}}$**
- 利用回归系数统计量初始化 trend encoder，同时大幅压缩输入维度（减少 >90%）

#### （3）结构性对齐（Structural Alignment）
通过上述方式实现“**模型容量与数据复杂度匹配**”：
- 避免过深或过浅的架构选择
- 减少冗余参数
- 加速收敛并提升稳定性

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | 本文方法 |
|------|----------|---------|
| **初始化策略** | 随机或固定启发式（如 Kaiming + 固定 $\mu,\sigma$） | 数据驱动、任务感知（data-driven, task-aware） |
| **架构配置** | 手动设定或网格搜索 | 由频谱主导模式数量自动决定 |
| **趋势建模效率** | 使用完整窗口输入 | 仅需少量等距采样点即可稳定估计斜率 |
| **计算效率** | 参数多、FLOPs 高 | 编码器维度显著降低，节省 20–30% 参数与计算量 |
| **可解释性** | 弱 | 强（显式分离 seasonality/trend/event） |

> ✅ **核心优势总结**：无需修改训练流程，仅通过更智能的设计与初始化，即可获得更快收敛、更高精度、更强鲁棒性和更低资源消耗。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 类型 | 描述 |
|-------|------|------|
| **Synthetic Dataset** | 合成数据 | 包含已知频率（3.5, 7.5, 12 Hz）、线性趋势和瞬态事件的信号，共 2000 条，每条 100 时间步，用于验证先验提取准确性 |
| **PJM Hourly** | 真实世界电力需求 | 1998–2001 年美国电网小时级负荷数据，按周切片，体现强周期性 |
| **Thermal Power Plant (TPP)** | 热力发电输出 | 2016–2020 年区域能源系统冬季供暖数据，具丰富多尺度周期与高噪声 |

---

### ⚙️ 实验设置

- **模型架构**：基于 **Bag-of-Functions (BoF)** 框架，采用残差堆叠结构（stacked BoF）
- **对比配置**（共四种）：
  1. **BoF**：标准初始化（baseline）
  2. **H-BoF**：使用文献 [4] 中的启发式初始化，但架构固定
  3. **I-BoF**：引入谱分析决定深度 + 季节性初始化，趋势部分仍为默认
  4. **IT-BoF**：完整提出方法 —— 谱分析决定深度 + 季节性初始化 + 趋势回归初始化 + 输入维度压缩

- **训练细节**：
  - 优化器：Adam ($lr = 1\times10^{-3}$)
  - Batch Size：16
  - Loss：MSE
  - Early Stopping
  - 每组实验运行 10 次独立 trial（不同 seed），报告均值 ± 标准差

- **评估指标**：
  - **Reconstruction MSE**（训练/测试）
  - **Convergence Speed**（loss 下降轨迹）
  - **Parameter Drift**（训练过程中参数总位移）
  - **Computational Efficiency**：参数量（Params）、FLOPs、推理延迟（Latency）、吞吐量（Throughput）

---

### 🆚 基线方法对比

除了内部变体比较外，还在生成式建模任务上与以下 SOTA 模型对比：
- **GAN-based**: R-GAN, CR-GAN, WaveGAN, CRNN-GAN
- **VAE-based**: LSTM-VAE, ITF-VAE, Time-VAE
- 特别设置了 **ITF-VAE-Informed Init** 变体，验证所提初始化在其他架构上的迁移能力

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### （1）合成数据集结果（Table II & Fig. 7）

| Model       | Test MSE (↓)         |
|-----------|---------------------|
| BoF       | 0.7633 ± 0.1091     |
| H-BoF     | 0.1937 ± ?          |
| I-BoF     | 0.0198 ± 0.0034     |
| **IT-BoF** | **0.0220 ± 0.0033** |

> ✔️ 相比 baseline 提升约 **97%**，且方差极小，说明高度稳定。

#### （2）真实世界数据集结果（Table III）

| Dataset | Model   | Test MSE (↓)        |
|--------|--------|--------------------|
| **PJM** | BoF    | 0.0155 ± 0.0008    |
|        | IT-BoF | **0.0074 ± 0.0011** | → ↓52% |
| **TPP** | BoF    | 0.4621 ± 0.0192    |
|        | I-BoF  | **0.1958 ± 0.0027** | → ↓58% |
|        | IT-BoF | 0.2035 ± 0.0077     | → ↓56%（仍具竞争力）|

> ✔️ 在真实场景下依然显著优于所有 baseline。

---

### 📈 收敛行为与参数演化（Fig. 8 & 9）

- **Loss Trajectory**：
  - IT-BoF 和 I-BoF 初始 loss 更低，早期快速下降
  - BoF 收敛慢，波动大
- **Parameter Displacement**：
  - IT-BoF 参数漂移最小，表明优化路径更稳定
  - 表明“良好起点”有效约束了训练动态

---

### 💡 消融实验结果（隐含在配置对比中）

| 组件 | 是否启用 | 效果影响 |
|------|--------|--------|
| 谱分析决定深度 $S$ | 否 → 是 | 显著提升重建质量，避免欠拟合/过拟合 |
| 季节性初始化 | 否 → 是 | 大幅降低初始 loss，加速收敛 |
| 趋势回归 + 输入压缩 | 否 → 是 | 减少 20–30% 参数/FLOPs，保持甚至提升性能 |
| 完整 IT-BoF | 全开 | 综合表现最优，尤其在 PJM 上 |

> ✅ 验证了各模块的有效性，尤其是“结构+初始化”联合设计的价值。

---

### ⚙️ 计算效率（Table IV）

| Dataset | Model   | Params (×10³) | FLOPs (×10³) | Throughput (↑) |
|--------|--------|---------------|-------------|----------------|
| Synthetic | BoF      | ~62           | ~61.5        | 3895           |
|           | **IT-BoF** | **47.38**     | **46.81**    | **4012**       |
| PJM       | BoF      | ~63           | ~62.75       | 5749           |
|           | **IT-BoF** | **44.89**     | **44.53**    | **6224**       |

> ✔️ **参数减少 23–30%，FLOPs 下降类似比例，吞吐量最高提升 8%**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **数据先验可显著改善函数参数化网络的表现**：
   - 仅通过更合理的初始化与架构配置，就能实现接近两个数量级的误差下降。
   
2. **谱分析不仅是预处理工具，更是架构设计蓝图**：
   - 主导频率数可直接映射为模型深度 $S$，实现“数据驱动的拓扑构建”。

3. **趋势估计存在收敛速率差异**：
   - 斜率（slope）估计以 $n^{-3/2}$ 快速收敛，而截距（bias）仅为 $n^{-1/2}$
   - 因此应优先保证斜率估计精度，截距可通过 NN 内部 bias 自动修正 → 支持极小输入窗口

4. **紧凑 ≠ 不准确**：
   - IT-BoF 在大幅压缩 trend encoder 输入维度（如从 168 → 3 或 13）的情况下，仍能达到最佳性能。

5. **初始化即正则化**：
   - 合理的起始点不仅加速训练，还限制参数空间探索范围，增强鲁棒性与泛化能力。

---

### ⚠️ 方法的局限性

1. **依赖信号具备一定周期性**：
   - 若 $p_{\text{spec}}$ 很低（弱季节性），谱分析效果受限，需更多依赖 trend/event 分支。

2. **对非平稳突变建模有限**：
   - 虽然 event branch 可捕捉瞬态，但初始化仍以线性/周期为主，难以覆盖极端非线性动态。

3. **当前框架主要用于离线建模**：
   - 尚未集成到在线学习或自适应更新机制中。

4. **假设时间序列等间隔采样**：
   - 对不规则时间序列的支持需额外插值或修改 FFT 应用方式。

---

### 🔮 未来工作方向

1. **自动化紧凑架构设计**：
   - 进一步结合稀疏表示、剪枝等技术，实现完全自适应的模型大小调节。

2. **扩展至在线/增量学习场景**：
   - 动态更新谱先验与趋势估计，支持流式数据下的持续学习。

3. **融合物理知识与领域先验**：
   - 结合专家知识（如已知设备周期）与数据驱动分析，形成 hybrid prior。

4. **推广至其他函数参数化架构**：
   - 如 N-BEATS、INRs 等，验证该初始化范式的普适性。

---

## ✅ 总结

本论文提出了一种**将经典信号处理思想与现代神经网络设计相结合**的新范式 —— **Prior-Informed Initialization**。它不是提出新的网络结构，而是通过**利用 FFT 和线性回归提取数据内在结构**，来指导 **BoF 架构的深度选择、编码器初始化与输入维度压缩**。

> 🎯 **一句话总结**：  
> “让模型从‘懂信号’的地方开始学”，而不是从零摸索。这种方法在不改变训练过程的前提下，实现了**更快收敛、更高精度、更强鲁棒性与更低计算成本**，为可解释、高效的时间序列建模提供了重要实践路径。

</details>

---

### 12. [Trajectory-Optimized Time Reparameterization for Learning-Compatible Reduced-Order Modeling of Stiff Dynamical Systems](https://arxiv.org/abs/2603.16583)

**Authors**: Joe Standridge, Daniel Livescu, Paul Cizmas  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.16583v1  

#### Abstract
Stiff dynamical systems present a challenge for machine-learning reduced-order models (ML-ROMs), as explicit time integration becomes unstable in stiff regimes while implicit integration within learning loops is computationally expensive and often degrades training efficiency. Time reparameterizatio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Trajectory-Optimized Time Reparameterization for Learning-Compatible Reduced-Order Modeling of Stiff Dynamical Systems**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**stiff dynamical systems**在基于机器学习的降阶模型（ML-ROMs）中的建模难题。这类系统因存在多尺度动态（快速瞬态与慢速模式共存），导致：
- 显式时间积分不稳定；
- 隐式积分计算昂贵且梯度传播困难；
- 传统神经ODE（NODE）框架难以高效训练。

尽管已有研究提出**Time Reparameterization (TR)** 来缓解刚性，但现有TR方法对“可学习性”（learnability）的影响尚不明确，且其设计依赖于求解器行为或几何启发式规则，可能导致时间映射（time map）不平滑、参数敏感或泛化能力差。

---

### **提出的新方法：TOTR**
作者提出了 **Trajectory-Optimized Time Reparameterization (TOTR)**，将时间重参数化本身建模为一个**优化问题**，其核心思想是：

- 在**arc-length坐标系**下定义一个遍历速度剖面 $ v(s) = ds/d\tau $，通过最小化变换后轨迹在拉伸时间 $\tau$ 下的加速度来构造最优的时间映射。
- 优化目标函数为：
  $$
  \mathcal{S}[v] = \int \left[(v v_s)^2 + (\kappa(s) v^2)^2\right] ds
  $$
  其中第一项惩罚切向加速度（速度变化率），第二项惩罚法向加速度（路径曲率影响），从而确保变换后的轨迹在 $\tau$ 中尽可能平滑。

这一方法直接以“被积分的动力学”的平滑性为目标，而非间接依赖求解器步长或极值点等代理信号。

---

### **相比现有方法的优势**
| 特性 | Solver-Directed TR | Extrema-Based TR | **TOTR (本文)** |
|------|---------------------|-------------------|------------------|
| 设计依据 | 自适应求解器步长 | 轨迹极值点位置 | 动力学加速度最小化 |
| 对噪声鲁棒性 | 差（受tolerance影响大） | 较好 | **优秀（更稳定）** |
| 可学习性 | 时间导数剧烈振荡 | 极值缺失时失效（如单调刚性事件） | **平滑、单调、易于拟合** |
| 参数鲁棒性 | 弱（随参数变化不稳定） | 中等 | **强（跨参数一致性高）** |
| 处理非极值刚性事件 | ❌ 失效 | ❌ 无法识别 | ✅ 成功捕捉 |

> ✅ **关键优势总结**：  
> TOTR生成的时间映射不仅有效缓解刚性，而且具有更高的**正则性**（regularity）、更好的**可学习性**（learnability）和更强的**参数鲁棒性**，特别适用于神经ODE框架下的ML-ROM训练。

---

## **2. 核心实验方法和设置**

### **使用的数据集（Benchmark Problems）**
论文在三个典型的stiff dynamical systems上进行验证：

1. **Parameterized Stiff Linear System (SLS)**  
   - 5维线性系统，含参数控制的刚性比。
   - 用于隔离非线性效应，专注于刚性和参数变化的影响。

2. **Van der Pol Oscillator**  
   - 经典非线性刚性系统，在松弛振荡 regime 下表现出快慢交替行为。
   - 测试方法对边界层和慢流形跟踪的能力。

3. **HIRES Chemical Kinetics System**  
   - 8维化学反应动力学模型，包含两个分离的刚性事件（初态塌缩 + 末态耗尽驱动突变）。
   - 尤其挑战在于第二个事件是**单调刚性跃迁**（无极值），考验extrema-based方法的极限。

---

### **实验设置**
- **训练策略**：
  - 使用**Latin Hypercube Sampling (LHS)** 进行超参搜索（网络结构、学习率、预测视界等）。
  - 所有方法共享相同的1000组超参配置，保证公平比较。
  - 使用 `torchdiffeq` 库实现NODE训练，采用Forward Euler积分器加速评估。

- **输入输出归一化**：
  - 状态变量和时间均归一化至 $[-1,1]$ 或 $[0,5]$ 区间，提升训练稳定性。

- **训练流程**：
  - 先预训练1000步（单步前向欧拉）；
  - 再主训练5000 epochs，逐步扩展预测视界。

---

### **评估指标**
| 指标 | 名称 | 定义 | 用途 |
|------|------|------|------|
| **t-MSE** | Stretched-Time Mean Squared Error | 在拉伸时间 $\tau$ 上的状态与时间预测误差 | 衡量**可学习性**（learnability in $\tau$） |
| **MSIE** | Mean Squared Integral Error (物理时间) | 在原始物理时间 $t$ 上积分的误差：<br>$ \text{MSIE} = \frac{1}{N_q T} \sum_i \int (q_{\text{pred}}(t) - q_{\text{true}}(t))^2 dt $ | 衡量**最终物理轨迹精度**（reparameterization-invariant） |

> ⚠️ 注意：低 t-MSE 不一定意味着低 MSIE —— 若时间映射不合理，即使局部拟合好也可能全局重建失败。

---

### **基线方法对比**
- **Solver-Directed TR** [Caldana & Hesthaven, 2025]  
  利用隐式求解器产生的自适应时间网格构造 $\tau$。
- **Extrema-Based TR** [Cortés García et al., 2025]  
  基于轨迹弧长并在极值点附近延展时间。
- **Proposed: TOTR**（本文方法）

所有方法均使用相同神经网络架构联合学习状态演化 $f_{NN}$ 和时间缩放 $\omega_{NN}$。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### ✅ **SLS（刚性线性系统）**
| 方法 | t-MSE (state avg.) | MSIE (state avg.) |
|------|--------------------|-------------------|
| Solver-Directed | ~$10^{-3}$ | ~$10^{-4}$ |
| Extrema-Based | ~$10^{-4}$ | ~$10^{-4}$ |
| **TOTR (Ours)** | **~$10^{-5}$** | **~$10^{-5}$** |

> 🔺 **提升1–2个数量级**，尤其在高刚性区域（如 $u=10^{3.95}$）表现显著优于基线。

#### ✅ **Van der Pol Oscillator**
| 方法 | t-MSE (state) | MSIE (state) |
|------|--------------|-------------|
| Solver-Directed | $10^{-1}$ | $3.08\times10^{-1}$ |
| Extrema-Based | $2.61\times10^{-3}$ | $1.91\times10^{-1}$ |
| **TOTR (Ours)** | $6.96\times10^{-3}$ | **$1.11\times10^{-1}$** |

> 📌 虽然 t-MSE 略高于 extrema-based，但 **MSIE 最低**，说明其重建的物理轨迹最准确。
> ➜ 表明：**t-MSE 并非唯一标准，MSIE 更能反映真实建模质量**。

#### ✅ **HIRES 化学动力学系统**
| 方法 | t-MSE (y7/y8) | MSIE (state) |
|------|--------------|-------------|
| Solver-Directed | ~$10^{-4}$ | $4.60\times10^{-4}$ |
| Extrema-Based | ~$10^{-3}$ | $1.04\times10^{-5}$ |
| **TOTR (Ours)** | **~$10^{-5}$** | **$6.29\times10^{-6}$** |

> 🔺 在单调刚性事件（late-time event）中，extrema-based 因缺乏极值而完全失效；  
> TOTR 成功分配足够 $\tau$ 分辨率，实现最佳拟合。

---

### **消融分析与定性观察**
- **Solver-Directed TR 缺陷**：
  - 时间导数在末端出现“爆炸式增长”（>100倍加速），导致 ML-ROM 难以学习。
  - 对积分容差敏感（见 Fig. 1–2），不同tolerance下产生完全不同 $\tau$ 映射。

- **Extrema-Based TR 缺陷**：
  - 对**无极值的单调刚性跃迁无效**（如 HIRES 第二阶段）；
  - 时间导数高频振荡，增加学习难度。

- **TOTR 优势可视化**：
  - 生成的时间曲线 $t(\tau)$ 更平滑、单调性更好；
  - 加速度分布均匀，避免尖锐转折或突发加速；
  - 在参数变化下保持高度一致的时间映射结构。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **刚性缓解 ≠ 可学习性提升**  
   即使某种TR方法成功“拉平”了刚性，若其引入复杂的时间导数结构（如剧烈振荡、突变），仍会导致训练困难。

2. ✅ **时间映射本身的平滑性至关重要**  
   平滑的 $dt/d\tau$ 曲线能降低梯度幅值、提高优化稳定性，并增强跨参数泛化能力。

3. ✅ **TOTR 显著优于现有TR方法**  
   在所有benchmark中，TOTR 均取得最低的 **MSIE**，表明其在物理时间上的预测最精确。

4. ✅ **arc-length + acceleration minimization 是有效范式**  
   将TR转化为arc-length空间中的变分优化问题，避免了直接离散化带来的病态条件，数值稳定且易于实现。

---

### **局限性**
1. ❗ 当前仅使用固定长度的 $\tau$ 网格和采样密度，未探索分辨率与效率之间的权衡。
2. ❗ Extrema-Based 方法仅测试了cubic spline版本，quintic spline可能改善性能，需进一步比较。
3. ❗ 所有结果基于统一超参搜索，不代表各方法的“最优潜力”，手动调优可能缩小差距。

---

### **未来工作方向**
1. 🔧 开发自适应选择拉伸时间终点 $T_f$ 和采样密度的原则；
2. 🔁 探索将TOTR与其他ROM技术结合（如autoencoder latent space）；
3. 🔄 研究反向重构机制，支持任意物理时间点查询；
4. 🤖 将TOTR嵌入端到端训练流程，联合优化时间映射与动力学网络。

---

## **总结**
> 💡 **TOTR 提供了一个 principled、flexible 且 highly effective 的框架，用于构建面向学习的 time reparameterization。它不再被动地“继承”求解器行为或几何特征，而是主动“设计”一个更适合神经网络学习的时间钟表。**

该研究表明：**在ML-ROM中，一个好的“clock”不仅是数值稳定的，更是平滑的、鲁棒的、可学习的。**  
TOTR 正是在这一理念指导下提出的新型TR范式，为显式、高效、可扩展的多尺度动力系统建模提供了坚实基础。

</details>

---

### 13. [GSI Agent: Domain Knowledge Enhancement for Large Language Models in Green Stormwater Infrastructure](https://arxiv.org/abs/2603.15643)

**Authors**: Shaohuang Wang  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.15643v1  

#### Abstract
Green Stormwater Infrastructure (GSI) systems, such as permeable pavement, rain gardens, and bioretention facilities, require continuous inspection and maintenance to ensure long-term performance. However, domain knowledge about GSI is often scattered across municipal manuals, regulatory documents, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GSI Agent: Domain Knowledge Enhancement for Large Language Models in Green Stormwater Infrastructure

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **领域知识分散**：Green Stormwater Infrastructure (GSI) 领域的知识广泛分布于市政手册、法规文件和检查表单中，缺乏集中化管理。
- **通用 LLM 在专业工程任务中的局限性**：尽管 Large Language Models (LLMs) 具备强大的通用推理能力，但在 GSI 这类高度专业化、规范性强的工程场景中，容易产生“幻觉”（hallucination）或提供不准确的技术建议。
- **非专家用户的使用障碍**：现场维护人员和社区成员难以从复杂文档中快速获取可操作的指导。

### 🚀 提出的新方法与创新思路
作者提出 **GSI Agent** —— 一种面向 GSI 领域的 domain-enhanced LLM 框架，通过三种互补策略系统性增强 LLM 的领域能力：

| 组件 | 核心思想 |
|------|---------|
| **Supervised Fine-Tuning (SFT)** | 在构建的 GSI 指令数据集上对基础 LLM 进行微调，使其掌握 GSI 术语、推理模式和响应格式。采用 LoRA 实现参数高效微调。 |
| **Retrieval-Augmented Generation (RAG)** | 构建基于市政文档的内部 GSI Knowledge Base，在推理时动态检索相关段落作为上下文输入，提升事实准确性并支持知识更新。 |
| **Agent-Based Reasoning Pipeline** | 设计一个智能体工作流，协调检索、上下文整合与结构化输出生成，实现任务导向的灵活推理（如判断、规划、验证等）。 |

此外，论文还发布了首个面向 GSI 检查与维护场景的指令型数据集 —— **GSI Dataset**，用于训练与评估。

### 🔍 相比现有方法的优势
- **系统性融合多策略**：不同于单一使用 SFT 或 RAG 的方法，GSI Agent 将静态知识注入（SFT）、动态知识检索（RAG）与结构化推理控制（Agent）有机结合，兼顾知识深度与灵活性。
- **保持通用能力的同时提升专业性能**：框架在显著提升 GSI 任务表现的同时，未牺牲模型在通用知识上的性能（zero-forgetting）。
- **实用性强**：适用于真实世界中的基础设施运维场景，具备良好的可扩展性和可维护性（例如可通过更新 KB 而无需重新训练来适配新规）。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集名称 | 描述 | 规模 | 特点 |
|----------|------|-------|--------|
| **GSI Dataset** | 自建的领域专用指令数据集，涵盖问答、验证、程序生成、信息提取等任务 | 10,955 条样本 | - 54.2% 包含地理位置（如 Philadelphia）<br>- 73.3% 用于 SFT，26.7% 用于 RAG<br>- 主要任务类型：Question Answering (31.2%)、Verification/Judgment (30.9%)、Generation/Composition (15.2%) |
| **Common Knowledge Dataset** | 外部通用基准数据集，用于评估通用知识保留能力 | 5,000 条样本 | 来源于 MMMU/MMBench，包含 QA、分类、逻辑推理等多样化任务 |

### 📊 实验设置与评估指标

#### 评估维度与指标
| 指标类别 | 指标名称 | 定义与用途 |
|--------|--------|-----------|
| **Lexical Overlap** | BLEU-4, ROUGE-1/2/L | 衡量生成文本与参考答案之间的 n-gram 精确率与召回率，适合短事实性回答 |
| **Label Accuracy** | Micro-F1 | 分类任务中聚合所有类别的 F1 分数 |
| **Semantic Similarity** | Sentence-BERT Cosine Similarity | 基于嵌入空间的语义相似度，弥补词汇重叠不足 |
| **Reasoning Quality** | G-Eval (LLM as Judge) | 使用另一个 LLM 作为裁判打分（1–5），评估正确性与连贯性 |
| **Human Judgment** | Human Expert Score | 小样本人工评分，评价实用性与技术准确性 |

#### 基线方法对比（见 Table 8）
| 基线 | RAG | SFT | Agent | 说明 |
|-----|-----|-----|-------|------|
| **Base LLM** | ❌ | ❌ | ❌ | 直接提示原始模型 |
| **Base LLM + RAG** | ✅ | ❌ | ❌ | 加入检索增强，无参数更新 |
| **Fine-tuned LLM + RAG** | ✅ | ✅ | ✅ | 完整 GSI Agent 框架（LoRA-SFT + RAG + Agent 控制） |

> 主要基础模型为 **Qwen3-VL-2B-Instruct**，并尝试其他开源模型进行可行性比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 10）

| Metric | Common Knowledge Dataset (Base LLM → GSI LLM) | GSI Dataset (Base LLM → GSI LLM) |
|--------|---------------------------------------------|-------------------------------|
| **BLEU-4** | 0.304 → **0.305** （基本不变） | 0.090 → **0.307** （↑241%） |
| **ROUGE-1** | 0.352 → 0.351 | 0.157 → **0.204** |
| **ROUGE-2** | 0.146 → 0.146 | 0.032 → **0.111** |
| **ROUGE-L** | 0.223 → 0.223 | 0.071 → **0.153** |
| **Sentence-BERT** | 0.861 → **0.869** | 0.544 → **0.742** |
| **G-Eval** | 0.82 → **0.84** | 0.57 → **0.79** |

> ✅ **核心发现**：在 GSI 任务上性能大幅提升，而通用知识能力几乎完全保留。

### ⚖️ 与基线方法的对比结果
- **完整 GSI Agent 显著优于所有基线**，尤其在 BLEU-4 和 G-Eval 上远超仅使用 RAG 或仅微调的方法。
- 单独使用 RAG 虽能改善事实性，但受限于检索质量；单独 SFT 可学习领域模式但难适应新政策。
- **组合策略带来协同增益**。

### 🔬 消融实验结果（Table 11）
| 方法 | G-Eval Score |
|------|--------------|
| LLM + RAG | 0.51 |
| LLM + Fine-tuning | 0.63 |
| **LLM + RAG + Fine-tuning** | **0.72** |

> ✅ 结果表明：**SFT 与 RAG 的结合是性能提升的关键驱动因素**，二者缺一不可。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **系统性的 domain knowledge enhancement 是可行且有效的路径**：将 SFT、RAG 与 Agent 控制相结合，可以成功将通用 LLM 适配到专业基础设施领域。
2. **GSI Agent 在领域任务上实现显著性能跃升**：在 GSI Dataset 上 BLEU-4 从 0.090 提高至 0.307，接近通用任务水平。
3. **通用知识能力得以保留**：在 Common Knowledge Dataset 上各项指标稳定，证明该框架不会导致灾难性遗忘（catastrophic forgetting）。
4. **多组件协同优于单一技术路线**：消融实验证明，SFT 与 RAG 的联合使用对复杂工程任务至关重要。

### ⚠️ 方法的局限性
- **依赖高质量文档构建 KB**：若原始市政文档存在错误或缺失，则会影响 RAG 效果。
- **当前 Agent 决策仍较轻量级**：尚未实现复杂的多步工具调用或长期记忆机制。
- **图像理解为可选功能**：虽然支持 vision-language 输入（via Qwen3-VL），但核心仍以文本推理为主。
- **地理泛化能力有限**：目前数据集中 54.2% 来自 Philadelphia，可能影响跨城市迁移效果。

### 🔮 未来工作方向
1. **扩大人类专家评估规模**：当前 human expert evaluation 仅限小样本，需进一步验证实际可用性。
2. **优化 retrieval 策略**：探索更先进的 dense retrieval、multi-hop retrieval 方法以提高上下文相关性。
3. **精细化 error analysis**：针对典型失败案例（如误判维护流程）进行归因分析。
4. **拓展至其他工程领域**：本框架具有通用性，可推广至交通、能源、建筑等其他 regulated infrastructure 应用场景。

---

> 💡 **总体评价**：  
> 本文提出了一个实用、可复现、模块化的 LLM 领域增强范式，不仅推动了 AI 在可持续城市基础设施中的应用，也为如何将大模型落地于高可靠性工程任务提供了重要参考。

</details>

---

### 14. [QV May Be Enough: Toward the Essence of Attention in LLMs](https://arxiv.org/abs/2603.15665)

**Authors**: Zhang Edward  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.15665v1  

#### Abstract
Starting from first principles and a linguistic perspective centered on part-of-speech (POS) and syntactic analysis, this paper explores and derives the underlying essence of the Query-Key-Value (QKV) mechanism within the Transformer architecture. Based on this theoretical foundation, we provide a u...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《QV May Be Enough: Toward the Essence of Attention in LLMs》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文从**第一性原理**和**语言学视角**出发，重新审视了 Transformer 架构中广泛使用的 **QKV（Query-Key-Value）机制**的本质逻辑。尽管 QKV 已成为大模型的标准组件，但其三个组成部分（Q、K、V）在自然语言处理中的**功能角色和内在逻辑尚不明确**。传统数据库类比（检索系统）难以解释其在语言建模中的深层作用。

本文旨在回答以下核心问题：
- Q、K、V 在语言结构中分别对应哪些语言实体？
- 是否可以简化 QKV 结构而不显著损失性能？
- 当前主流变体（如 MQA、GQA、MLA）为何有效？它们之间有何共性？

---

### **提出了什么新方法或新思路**

#### ✅ **提出“QV 范式”作为 QKV 的本质抽象**
基于对 **词性（POS）** 和 **句法结构** 的分析，作者提出：
- **V（Value）** 表示事实语义（shallow-composing），即当前 token 的语义表达；
- **K（Key）** 实际上是 V 经过 **Deep-Matching（深度匹配）** 后的结果，用于建立修饰关系（如形容词修饰名词）；
- **Q（Query）** 是对目标语义方向的“期望”。

由此推导出：**K 可由 V 推导而来**，因此标准 QKV 形式 $ \text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V $ 可以简化为：

> $$
\text{Attention}(Q,V) = \text{softmax}(QV^T/\sqrt{d_k})V
$$

这就是 **QV 范式** —— 即 **用 V 兼任 K 的角色**，从而省去独立的 K 投影矩阵 $ W_k $。

#### ✅ **提出 QV-Ka（Key-after-value & ctx）优化架构**
进一步地，作者提出 **QV-Ka** 模型，认为 K 并非完全独立，而是可以通过 V 和上下文（context）动态生成：

> $$
K = \text{DM}(V, \text{Context})
$$

具体实现方式为：
- 先计算 $ Q = QW_Q $, $ V = VW_V $
- 引入一个轻量级 context 向量 $ G = \text{ctx} \cdot W_{\text{ctx}} $
- 将 $ [G; V] $ 拼接后通过 $ W_K $ 得到 K

这实现了 **K 的延迟生成（after value）**，既保留了匹配能力，又减少了参数冗余。

#### ✅ **统一解释 MQA/GQA/MLA 的本质**
论文指出，当前流行的 KV 缓存压缩技术（如 MQA、GQA、MLA）本质上都是向 **QV 范式靠拢** 的近似形式：
- **MQA**：多个 Q 共享一组 K/V → 近似于 QV 模式
- **GQA**：分组共享 K/V → 更灵活的 QV 变种
- **MLA**：对 K/V 进行隐空间压缩 → 高度优化的软性 QV 模式

这一理论框架为这些架构的成功提供了统一解释。

---

### **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **理论层面** | 提供首个基于语言学原理的可解释性框架，揭示 QKV 内部逻辑，打破“黑箱”认知 |
| **效率层面** | QV 范式减少 $ W_k $ 参数，降低计算与内存开销；QV-Ka 进一步提升参数利用率 |
| **扩展性** | 支持与 AGF、ALiBi 等相对位置编码无缝结合，适用于长序列建模 |
| **指导意义** | 为未来 Attention 架构设计提供清晰优化路径（如 V-Shared、K 动态生成等） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **WMT_17 英德翻译任务（en-de）**  
  用于验证模型在典型 NLP 任务上的表现。

### **实验设置**
| 配置项 | 设置 |
|--------|------|
| 基础架构 | Vanilla Transformer-BIG |
| 层数（LayerNum） | 3（从默认 6 层缩减以加速训练） |
| 模型维度（d_model） | 1024 |
| 注意力头数（h） | 16 |
| 头维度（d_head） | 64 |
| 训练精度 | FP16（半精度） |
| 硬件平台 | 单张 NVIDIA Tesla V100-PCIE-32GB |
| 单次训练时长 | ~15 小时 |

### **评估指标**
- **Validation Accuracy (%)**：主评价指标，衡量翻译质量
- 对比不同模式下的收敛速度与最终准确率

### **基线方法对比**
- **Vanilla QKV**：标准多头注意力机制
- **QV Mode**：提出的简化范式（$ K \leftarrow V $）
- **QV + AGF + PCM-V**：引入 Attention’s Gravitational Field 相对位置编码，消除 PE 干扰
- **QV-Ka**：提出的 Key-after-value 架构，测试 $ d_{\text{ctx}} = d_{\text{head}} $ 和 $ 2d_{\text{head}} $ 两种配置

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 3 & 4）**

| 模式 | 配置 | Valid Accuracy (%) |
|------|------|---------------------|
| QV | Default (Sinusoidal PE) | 70.0756 |
| QKV | Default (Sinusoidal PE) | 70.5911 |
| QV | AGF + PCM-V | **70.5188** |
| QKV | AGF + PCM-V | **70.7305** |
| QV-Ka ($d_{\text{ctx}}=d_{\text{head}}$) | AGF + PCM-V | 70.4998 |
| QV-Ka ($d_{\text{ctx}}=2d_{\text{head}}$) | AGF + PCM-V | **70.6919** |

---

### **与基线方法的对比结果**

- **原始设定下（带 Sinusoidal PE）**：
  - QV 比 QKV 低约 **0.5%** 准确率
  - 主因之一是 **Positional Encoding 干扰**：在 QV 中 V 承担 K 的角色，导致语义与位置耦合更强

- **引入 AGF 相对位置编码后**：
  - QV 性能大幅提升至 **70.5188%**，仅落后 QKV **0.21%**
  - 表明 **超过一半的性能差距来自 PE 干扰**，而非结构缺陷

- **QV-Ka 表现亮眼**：
  - 当 $ d_{\text{ctx}} = 2d_{\text{head}} $ 时，达到 **70.6919%**，已**接近甚至略微超越标准 QKV**
  - 在训练初期还表现出更快收敛趋势，说明其更符合 Attention 的内在逻辑

---

### **消融实验结果**

- **Positional Encoding 影响分析**：
  - 使用 AGF 解耦位置信息后，QV 性能显著提升，证明 **传统 PE 对 QV 不友好**
  - 支持“QV 失效主因是 PE 干扰”的假设

- **DODM（Diffusion of Deep-Matching）现象分析**：
  - QKV 中 DODM 发生在 V 上，有助于强化相关特征
  - QV 中 DODM 发生在 Q 上，造成查询意图扩散，削弱聚焦能力
  - 解释了为何 QKV 通常略优

- **QV-Ka 参数敏感性测试**：
  - $ d_{\text{ctx}} = 1\times d_{\text{head}} $：性能略低于 QV
  - $ d_{\text{ctx}} = 2\times d_{\text{head}} $：性能反超，表明适当增强 context 表达能力至关重要

---

## **4. 关键结论和发现**

### **主要发现**

1. ✅ **QV 范式足以逼近 QKV 性能**  
   在合理的位置编码（如 AGF）支持下，**仅使用 Q 和 V 即可实现与完整 QKV 相当的表现**，挑战了“必须三者俱全”的固有观念。

2. ✅ **MQA/GQA/MLA 本质是 QV 范式的工程近似**  
   这些成功架构并非偶然，而是无意中趋近了更本质的 QV 逻辑，尤其是通过共享 K/V 来模拟 $ K \approx V $ 的关系。

3. ✅ **K 可被重构而非直接投影**  
   提出的 **QV-Ka** 架构证明：**Key 可基于 Value 和上下文动态生成**，不仅可行，且在高维 context 下能达到媲美甚至超越原生 QKV 的效果。

4. ✅ **Attention 的本质是“期望-事实”匹配机制**  
   - Q 表示“期待看到什么样的修饰”
   - V 表示“这里有什么样的内容”
   - K 是 V 经过 Deep-Matching 后的适配版本
   - 因此，**K 是派生量，非基本单元**

---

### **方法的局限性**

- 🔒 **实验规模有限**  
  使用的是小型化 Transformer（3层，d_model=1024），未在大规模 LLM（如百亿级以上）中验证泛化能力。

- ⚠️ **依赖 AGF 等特定位置编码**  
  QV 模式在标准 Sinusoidal PE 下性能下降明显，需配合解耦型位置建模才能发挥优势，限制了即插即用性。

- 🧩 **QV-Ka 增加实现复杂度**  
  虽然参数总量可能更低，但引入 context 向量和拼接操作增加了工程实现负担。

---

### **未来工作方向**

1. **发展纯 QV 架构的大模型验证**
   - 在 MoE 或超大规模 decoder-only 模型中测试 QV 是否仍具竞争力

2. **探索免 K 架构（K-Free Attention）**
   - 若 K 可由 V + context 生成，则未来可完全移除 $ W_k $，实现极致压缩

3. **优化 MLA 中的压缩比率**
   - 当前 K/V 压缩过于激进，建议 **降低 K 的压缩程度以保留更多全局匹配信号**

4. **推动 V-Shared 而非 KV-Shared**
   - 从 KV 共享转向 **仅 V 共享 + 独立 K**，兼顾效率与表达力

5. **构建基于 QV 的新型高效推理架构**
   - 利用 QV-Ka 在 KV Cache 场景下的潜力，开发更适合部署的轻量化 Attention 变体

---

> 💡 **总结一句话**：  
> 本论文从语言学出发，提出 **QV 范式是 Attention 的本质形态**，并通过 QV-Ka 实现了高效且高性能的新架构，为未来 LLM 设计提供了全新的理论基础与优化路径。

</details>

---

### 15. [Prompt Engineering for Scale Development in Generative Psychometrics](https://arxiv.org/abs/2603.15909)

**Authors**: Lara Lee Russell-Lasalandra, Hudson Golino  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.15909v1  

#### Abstract
This Monte Carlo simulation examines how prompt engineering strategies shape the quality of large language model (LLM)--generated personality assessment items within the AI-GENIE framework for generative psychometrics. Item pools targeting the Big Five traits were generated using multiple prompting ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Prompt Engineering for Scale Development in Generative Psychometrics**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本研究聚焦于**生成心理测量学**（Generative Psychometrics）中的一个关键方法论空白：**不同的提示工程策略如何影响大型语言模型**（LLM）**生成的心理测验题目的质量**。尽管LLM已被广泛用于自动生成量表题目（如人格测验），但现有研究大多将“提示”（prompt）视为固定或次要的设计选择，缺乏对不同提示策略系统性的比较。

具体而言，该研究旨在回答以下问题：
- 不同的prompt engineering策略是否以及在多大程度上影响生成题目的冗余性、结构效度和最终保留的题目数量？
- 更先进的模型（如GPT-5.1）是否能从高级提示策略中获益更多？
- 这些效果是否受模型温度（temperature）等超参数调节？

---

### **提出了什么新方法或新思路**

1. **提出并验证了AI-GENIE框架中prompt engineering的关键作用**  
   虽然AI-GENIE（Automatic Item Generation with Network-Integrated Evaluation）此前已被提出（Russell-Lasalandra et al., 2024），但本文首次系统地将其与多种prompt engineering策略结合，并量化其对生成项目池质量的影响。

2. **引入并测试了“自适应提示”**（adaptive prompting）作为最优策略  
   自适应提示通过动态反馈机制，在每次生成后向模型提供已生成项目的列表，并明确指令：“不要重复或改写这些内容”。这实现了**生成-评估-修订**（generate-evaluate-revise）的闭环，有效抑制语义冗余。

3. **揭示了模型能力与提示策略之间的交互效应**  
   首次实证表明，**更先进、更大容量的LLM**（如GPT-5.1）**能显著放大自适应提示的优势**，而较小模型（如GPT-OSS-20B）则收益有限。

---

### **相比现有方法的优势**

| 方面 | 传统方法 / 非自适应提示 | 本文提出的自适应提示 + AI-GENIE |
|------|------------------------|-------------------------------|
| 冗余控制 | 依赖后期算法剔除（如UVA） | 前置预防，减少初始冗余 |
| 结构效度 | 初始NMI较低，需大量修剪 | 初始NMI高，结构更稳定 |
| 题目保留率 | 大幅删减，样本小 | 显著提高最终可用题目数 |
| 可扩展性 | 手动设计提示成本高 | 自动化流程，适合大规模开发 |

> ✅ **核心优势**：**自适应提示不是替代AI-GENIE，而是与其互补——前者提升输入质量，后者进行确定性优化，共同构建高效、可复现的心理测量开发流水线**。

---

## **2. 核心实验方法和设置**

### **使用了哪些数据集**
- **非真实人类作答数据集**，而是采用**蒙特卡洛模拟**（Monte Carlo simulation）生成。
- 目标是为**Big Five人格特质**（OCEAN模型：Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism）生成测验题目。
- 每个trait提供4个属性（如Extraversion → friendly, positive, assertive, energetic），要求每属性生成2句第一人称陈述句。

---

### **实验设置**

#### **变量设计**
| 因素 | 水平 |
|------|------|
| **LLM模型** | GPT-4o, GPT-5.1, GPT-OSS-120B, GPT-OSS-20B |
| **温度**（temperature） | 0.5, 1.0, 1.5（控制创造性） |
| **Prompt类型**（6种） | Basic, Expanded, Few-Shot, Persona, Persona+Few-Shot, Persona+Few-Shot+Adaptive |
| **重复次数** | 每种组合重复100次（共约72,000个独立item pool） |
| **每轮生成目标** | ≥60个题目（通过多次API调用实现） |

#### **AI-GENIE Pipeline 流程**
1. **Generate and Embed**: 使用LLM生成题目 → 用`text-embedding-3-small`编码为向量。
2. **Initial EGA**: 使用Exploratory Graph Analysis估计维度结构，计算初始Normalized Mutual Information (**NMI**)。
3. **UVA迭代去冗余**: 使用Unique Variable Analysis识别并移除高wTO（weighted topological overlap）的冗余项。
4. **bootEGA迭代稳定性筛选**: 移除跨bootstrap样本分配不稳定的题目。
5. **Final EGA**: 在精简后的题库上重新运行EGA，计算最终NMI。

---

### **评估指标**

| 指标 | 含义 |
|------|------|
| **NMI**（Normalized Mutual Information） | 衡量EGA聚类结果与“真实”维度标签的一致性（0~1，越高越好） |
| **UVA Removal Count** | 被判定为冗余而删除的题目数量（越少越好） |
| **Final Item Pool Size** | 经过AI-GENIE过滤后保留的题目数量（越多越好） |
| **Pre-/Post-reduction NMI Gain** | AI-GENIE带来的结构效度提升幅度 |

---

### **基线方法对比**

| Prompt 类型 | 描述 |
|------------|------|
| **Basic**（零样本） | 最简指令：“生成衡量外向性的题目” |
| **Expanded** | 加入质量要求：“高质量、简洁、新颖” |
| **Few-Shot** | 提供John & Srivastava (1999)的真实题目示例（仅模仿格式） |
| **Persona** | 添加角色设定：“你是一位专家心理测量学家” |
| **Persona + Few-Shot** | 上述两者结合 |
| **Adaptive**（本文重点） | 在Persona+Few-Shot基础上，动态传入已生成题目列表，禁止重复 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 模型 | Prompt类型 | 平均最终NMI | UVA移除数 | 最终题目数 |
|------|-----------|-------------|------------|--------------|
| **GPT-5.1** | Basic | ~92% | ~36 | 16–29 |
| | **Adaptive** | **~98%** | **~2.3** | **56–57** |
| **GPT-OSS-120B** | Basic | ~95% | ~31 | 25–32 |
| | **Adaptive** | **~97%** | **~5.1** | **51–52** |
| **GPT-OSS-20B** | Basic | ~92% | ~17 | ~40 |
| | **Adaptive** | **~92.5%** | **~8.0** | ~46–54 |
| **GPT-4o** | Basic (T=1.5) | 94.28% | — | — |
| | **Adaptive (T=1.5)** | **85.84%** ⬇️ | — | **略少** |

> 📊 数据来源：Table 2 和 Figures 2–7

---

### **与基线方法的对比结果**

- **自适应提示显著优于所有非自适应策略**：
  - 在GPT-5.1上，**最终NMI提升达10.8个百分点**（T=0.5时）。
  - **冗余减少超过93.7%**（GPT-5.1）、88.5%（GPT-4o）、83.3%（GPT-OSS-120B）。
  - **保留题目数翻倍以上**（如GPT-5.1从平均<30增至>56）。

- **非自适应策略效果微弱甚至负面**：
  - Few-Shot、Persona等策略仅带来1–4%的NMI提升。
  - 某些条件下反而增加冗余（如Few-Shot在某些模型下UVA移除更多）。

---

### **消融实验结果**

- **自适应组件是唯一带来质变的因素**：
  - 从Basic → Expanded → Persona → Few-Shot → Adaptive，只有加入**adaptive feedback loop**才出现跳跃式改进。
  - 支持“动态反馈 > 静态上下文”的假设。

- **AI-GENIE的增量贡献随输入质量上升而下降**：
  - 当初始item pool质量高（如adaptive prompting下NMI > 91%），AI-GENIE带来的NMI增益缩小至5–6%。
  - 当初始质量低（basic prompt），AI-GENIE可带来7–10%增益。
  - ➡️ 说明**adaptive prompting与AI-GENIE功能互补而非重叠**。

---

## **4. 关键结论和发现**

### **主要发现**

✅ **1. 自适应提示是最强的prompt engineering策略**  
它能显著降低语义冗余、提升初始结构效度、增加最终可用题目数量，尤其适用于新一代大模型。

✅ **2. 效果随模型能力增强而放大**  
GPT-5.1 > GPT-OSS-120B > GPT-OSS-20B，表明**更大的LLM具备更强的in-context learning能力**，能更好地理解和执行复杂的累积约束。

✅ **3. 自适应提示缓解了“创造力 vs. 一致性”的权衡**  
即使在高温（T=1.5）下，也能保持低冗余和高结构效度（除GPT-4o外）。

⚠️ **4. GPT-4o表现出异常敏感性**  
在**T=1.5 + adaptive prompting**条件下，其最终NMI**下降8.4%**，且保留题目数减少。推测原因可能是：
- 高随机性下模型难以遵循复杂约束；
- 安全过滤器或内部机制干扰了反馈循环。

✅ **5. AI-GENIE始终能提升结构效度**  
无论初始prompt质量如何，AI-GENIE都能可靠地改善NMI，但其“边际效益”取决于输入质量。

---

### **方法的局限性**

🔸 **仅限Big Five人格特质**  
该构造定义清晰、训练数据丰富，可能无法推广到新兴或模糊构念（如“数字成瘾”、“职场韧性”）。

🔸 **完全in silico模拟，无人类评审**  
未纳入专家对题目内容效度、歧义性、伦理风险的判断，仍需后续人工审核。

🔸 **模型版本依赖性强**  
LLM持续更新（alignment tuning, safety filters），可能导致相同prompt行为变化，结果不具备长期稳定性。

🔸 **token限制导致多轮生成**  
需多次调用API拼接题目，增加了过程复杂性和潜在偏差。

---

### **未来工作方向**

🚀 **探索模型-提示交互机制**  
深入研究为何GPT-4o在高温下对adaptive提示敏感，其他模型却稳健？是否涉及内部注意力机制或安全层干预？

🚀 **扩展至其他心理构念**  
测试adaptive prompting在情绪障碍、认知能力、价值观等领域的泛化能力。

🚀 **整合人类-in-the-loop评估**  
将AI-GENIE输出交由心理学家评分，建立“自动化生成 + 人工精修”的混合范式。

🚀 **开发自动化prompt optimizer**  
基于强化学习或贝叶斯优化，自动搜索最优prompt配置（如温度、示例数量、反馈频率）。

🚀 **研究多模态生成中的prompt engineering**  
将该框架拓展至图像、语音等情境下的心理测量工具生成。

---

> 💡 **结语**：  
> 本文不仅展示了**adaptive prompting + AI-GENIE**的强大效能，更重要的是确立了一个**科学化、可复制的AI辅助心理测量开发范式**。它提醒我们：**AI不会取代心理测量学家，但懂得使用AI-GENIE和prompt engineering的心理测量学家，将会取代那些不懂的人**。

</details>

---

### 16. [NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics](https://arxiv.org/abs/2603.16148)

**Authors**: Zhengzheng Tang  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.16148v1  

#### Abstract
We ask whether a pure spiking backbone can learn large-scale language modeling from random initialization, without Transformer distillation. We introduce NeuronSpark, a 0.9B-parameter SNN language model trained with next-token prediction and surrogate gradients. The model combines selective state-sp...

---

### 17. [Via Negativa for AI Alignment: Why Negative Constraints Are Structurally Superior to Positive Preferences](https://arxiv.org/abs/2603.16417)

**Authors**: Quan Cheng  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.16417v1  

#### Abstract
Recent empirical results have demonstrated that training large language models (LLMs) with negative-only feedback can match or exceed standard reinforcement learning from human feedback (RLHF). Negative Sample Reinforcement achieves parity with PPO on mathematical reasoning; Distributional Disprefer...

---

### 18. [AdaMem: Adaptive User-Centric Memory for Long-Horizon Dialogue Agents](https://arxiv.org/abs/2603.16496)

**Authors**: Shannan Yan, Jingchen Ni, Leqi Zheng, Jiajun Zhang, Peixi Wu, Dacheng Yin, Jing Lyu, Chun Yuan, Fengyun Rao  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.16496v1  

#### Abstract
Large language model (LLM) agents increasingly rely on external memory to support long-horizon interaction, personalized assistance, and multi-step reasoning. However, existing memory systems still face three core challenges: they often rely too heavily on semantic similarity, which can miss evidenc...

---

### 19. [Omnilingual SONAR: Cross-Lingual and Cross-Modal Sentence Embeddings Bridging Massively Multilingual Text and Speech](https://arxiv.org/abs/2603.16606)

**Authors**: Omnilingual SONAR Team, Jo\~ao Maria Janeiro, Pere-Llu\'is Huguet Cabot, Ioannis Tsiamas, Yen Meng, Vivek Iyer, Guillem Ram\'irez, Loic Barrault, Belen Alastruey, Yu-An Chung, Marta R. Costa-Jussa, David Dale, Kevin Heffernan, Jaehyeong Jo, Artyom Kozhevnikov, Alexandre Mourachko, Christophe Ropers, Holger Schwenk, Paul-Ambroise Duquenne  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.16606v1  

#### Abstract
Cross-lingual sentence encoders typically cover only a few hundred languages and often trade downstream quality for stronger alignment, limiting their adoption. We introduce OmniSONAR, a new family of omnilingual, cross-lingual and cross-modal sentence embedding models that natively embed text, spee...

---

### 20. [pADAM: A Plug-and-Play All-in-One Diffusion Architecture for Multi-Physics Learning](https://arxiv.org/abs/2603.16757)

**Authors**: Amirhossein Mollaali, Bongseok Kim, Christian Moya, Guang Lin  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.16757v1  

#### Abstract
Generalizing across disparate physical laws remains a fundamental challenge for artificial intelligence in science. Existing deep-learning solvers are largely confined to single-equation settings, limiting transfer across physical regimes and inference tasks. Here we introduce pADAM, a unified gener...

---

### 21. [Proactive Rejection and Grounded Execution: A Dual-Stage Intent Analysis Paradigm for Safe and Efficient AIoT Smart Homes](https://arxiv.org/abs/2603.16207)

**Authors**: Xinxin Jin, Zhengwei Ni, Zhengguo Sheng, Victor C. M. Leung  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.16207v1  

#### Abstract
As Large Language Models (LLMs) transition from information providers to embodied agents in the Internet of Things (IoT), they face significant challenges regarding reliability and interaction efficiency. Direct execution of LLM-generated commands often leads to entity hallucinations (e.g., trying t...

---

### 22. [SIA: A Synthesize-Inject-Align Framework for Knowledge-Grounded and Secure E-commerce Search LLMs with Industrial Deployment](https://arxiv.org/abs/2603.16137)

**Authors**: Zhouwei Zhai, Mengxiang Chen, Anmeng Zhang  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.16137v1  

#### Abstract
Large language models offer transformative potential for e-commerce search by enabling intent-aware recommendations. However, their industrial deployment is hindered by two critical challenges: (1) knowledge hallucination due to insufficient encoding of dynamic, fine-grained product knowledge, and (...

---

### 23. [Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR](https://arxiv.org/abs/2603.16184)

**Authors**: Quy-Anh Dang, Chris Ngo  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.16184v1  

#### Abstract
We present Polyglot-Lion, a family of compact multilingual automatic speech recognition (ASR) models tailored for the linguistic landscape of Singapore, covering English, Mandarin, Tamil, and Malay. Our models are obtained by fine-tuning Qwen3-ASR-0.6B and Qwen3-ASR-1.7B exclusively on publicly avai...

---

### 24. [Deriving Hyperparameter Scaling Laws via Modern Optimization Theory](https://arxiv.org/abs/2603.15958)

**Authors**: Egor Shulgin, Dimitri von R\"utte, Tianyue H. Zhang, Niccol\`o Ajroldi, Bernhard Sch\"olkopf, Antonio Orvieto  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.15958v1  

#### Abstract
Hyperparameter transfer has become an important component of modern large-scale training recipes. Existing methods, such as muP, primarily focus on transfer between model sizes, with transfer across batch sizes and training horizons often relying on empirical scaling rules informed by insights from ...

---

### 25. [Determinism in the Undetermined: Deterministic Output in Charge-Conserving Continuous-Time Neuromorphic Systems with Temporal Stochasticity](https://arxiv.org/abs/2603.15987)

**Authors**: Jing Yan, Kang You, Zhezhi He, Yaoyu Zhang  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.15987v1  

#### Abstract
Achieving deterministic computation results in asynchronous neuromorphic systems remains a fundamental challenge due to the inherent temporal stochasticity of continuous-time hardware. To address this, we develop a unified continuous-time framework for spiking neural networks (SNNs) that couples the...

---

### 26. [Physics-integrated neural differentiable modeling for immersed boundary systems](https://arxiv.org/abs/2603.16277)

**Authors**: Chenglin Li, Hang Xu, Jianting Chen, Yanfei Zhang  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.16277v1  

#### Abstract
Accurately, efficiently, and stably computing complex fluid flows and their evolution near solid boundaries over long horizons remains challenging. Conventional numerical solvers require fine grids and small time steps to resolve near-wall dynamics, resulting in high computational costs, while purel...

---

### 27. [Form Follows Function: Recursive Stem Model](https://arxiv.org/abs/2603.15641)

**Authors**: Navid Hakimi  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.15641v1  

#### Abstract
Recursive reasoning models such as Hierarchical Reasoning Model (HRM) and Tiny Recursive Model (TRM) show that small, weight-shared networks can solve compute-heavy and NP puzzles by iteratively refining latent states, but their training typically relies on deep supervision and/or long unrolls that ...

---

### 28. [Theoretical Foundations of Latent Posterior Factors: Formal Guarantees for Multi-Evidence Reasoning](https://arxiv.org/abs/2603.15674)

**Authors**: Aliyu Agboola Alege  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.15674v1  

#### Abstract
We present a complete theoretical characterization of Latent Posterior Factors (LPF), a principled framework for aggregating multiple heterogeneous evidence items in probabilistic prediction tasks. Multi-evidence reasoning arises pervasively in high-stakes domains including healthcare diagnosis, fin...

---

### 29. [Optimizing Hospital Capacity During Pandemics: A Dual-Component Framework for Strategic Patient Relocation](https://arxiv.org/abs/2603.15960)

**Authors**: Sadaf Tabatabaee, Hicham El Baz, Mohammed Khalil Ghali, Nagendra N. Nagarur  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.15960v1  

#### Abstract
The COVID-19 pandemic has placed immense strain on hospital systems worldwide, leading to critical capacity challenges. This research proposes a two-part framework to optimize hospital capacity through patient relocation strategies. The first component involves developing a time series prediction mo...

---

### 30. [MOSAIC: Composable Safety Alignment with Modular Control Tokens](https://arxiv.org/abs/2603.16210)

**Authors**: Jingyu Peng, Hongyu Chen, Jiancheng Dong, Maolin Wang, Wenxi Li, Yuchen Li, Kai Zhang, Xiangyu Zhao  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.16210v1  

#### Abstract
Safety alignment in large language models (LLMs) is commonly implemented as a single static policy embedded in model parameters. However, real-world deployments often require context-dependent safety rules that vary across users, regions, and applications. Existing approaches struggle to provide suc...

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
