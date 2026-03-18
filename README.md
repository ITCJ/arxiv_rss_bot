# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-18 06:48:26 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation](https://arxiv.org/abs/2603.16219)

**Authors**: Hang Lv, Sheng Liang, Hao Wang, Yongyue Zhang, Hongchao Gu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.16219v1  

#### Abstract
Realizing personalized intelligence faces a core dilemma: sending user history to centralized large language models raises privacy concerns, while on-device small language models lack the reasoning capacity required for high-quality generation. Our pilot study shows that purely local enhancements re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对个性化生成中的一个**核心困境**（fundamental dilemma）：  
- **集中式大模型**（Centralized LLMs）虽然具备强大的推理能力，但需要上传用户的敏感历史数据到云端，引发严重的**隐私泄露风险**；
- **本地小模型**（On-device SLMs）虽能保护隐私，但受限于模型容量，其**推理能力不足**，生成内容常出现逻辑错误、幻觉或过于泛化。

传统的解决方案（如 RAG 或 LoRA）仅在本地增强上下文，无法弥补小模型在复杂推理上的根本性缺陷。

---

### **提出了什么新方法或新思路**
作者提出 **SPECSTEER**，一种**非对称协作推理框架**（asymmetric collaborative inference framework），通过“**Draft-Verify-Recover**”三阶段流程，实现本地私有上下文与云端大规模推理能力的协同。

#### **核心思想**：
将协作建模为**贝叶斯知识融合**（Bayesian Knowledge Fusion）过程，并**重构（repurpose）Speculative Decoding** 作为分布式对齐协议。

#### **三个关键阶段**：
1. **Drafting（起草）**：  
   - 由本地 Specialist 小模型基于私有用户历史生成个性化草案序列。
2. **Verification（验证）**：  
   - 云端 Generalist 大模型通过**比率验证机制**（ratio-based verification）判断草案是否符合逻辑合理性。
   - 验证过程**不访问原始用户上下文**，而是通过比较 `PLLM(y)/PSLM-(y)` 的比值来过滤因缺乏私有知识导致的误判。
3. **Recovery（恢复）**：  
   - 若草案被拒绝，则进行**引导式恢复**（steering recovery），通过注入 `hSLM - hSLM-` 的对比向量，在修正逻辑的同时保留用户意图。

---

### **相比现有方法的优势**
| 维度 | SPECSTEER | 传统方法 |
|------|----------|---------|
| **隐私保护** | ✅ 用户上下文始终保留在设备端 | ❌ RAG/LoRA需上传或暴露上下文 |
| **推理质量** | ✅ 利用大模型进行逻辑校验与修复 | ❌ 小模型独立生成易出错 |
| **效率** | ✅ 异步验证，通信开销低，**提速2.36×** | ❌ Token-level fusion 同步成本高 |
| **通用性** | ✅ 跨架构部署（如 Qwen + Llama）有效 | ❌ 多数方法依赖同构模型 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LaMP** (Salemi et al., 2023)：包含6个任务，涵盖摘要、标题生成、推文改写等。
- **LongLaMP** (Kumar et al., 2024)：更长文本的个性化生成基准，用于主实验。

具体任务包括：
- **Abstract Generation**（个性化摘要）
- **Review Writing**（个性化评论）
- **Topic Writing**（个性化文章撰写）

---

### **实验设置和评估指标**
#### **模型配置**
使用多组 **Small Specialist ↔ Large Generalist** 对：
- Qwen3-0.6B ↔ Qwen3-32B
- Qwen2.5-1.5B ↔ Qwen2.5-32B
- Qwen3-8B ↔ Qwen3-32B
- Llama-3.21B ↔ Llama-3.1-8B

#### **评估指标**
- **ROUGE-1 (R1)** 和 **ROUGE-L (RL)**：衡量生成内容与参考文本的重叠度。
- **Speedup**：生成吞吐量（tokens/s）相对于标准 LLM 的加速比。
- **Acceptance Rate (α)**：草案被接受的比例，反映协作效率。

#### **超参数设置**
- 默认 `λ = 0.5`, `β = 1.0`
- Draft horizon `K = 4`

---

### **基线方法对比**
| 类型 | 方法 | 说明 |
|------|------|------|
| **本地增强** | SLM (Direct), LoRA, RAG, LoRA+RAG, RAFT | 在小模型上进行微调或检索增强 |
| **纯大模型** | LLM (32B) | 零样本运行，无用户上下文 |
| **协作类方法** | CoSteer, LightCoSteer, Standard SD | 现有协作推理方法 |
| **其他基线** | BM25, BGE, TAM, OPPU, PAD, CoPE | 包括检索、PEFT、对齐等方法 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **Table 2 & 3：LongLaMP 上的性能对比（以 Qwen3-0.6B/32B 为例）**
| 方法 | Abs R1 | Abs RL | Rev R1 | Rev RL | Wri R1 | Wri RL |
|------|--------|--------|--------|--------|--------|--------|
| SLM (0.6B) | 36.58 | 20.34 | 24.15 | 12.95 | 26.32 | 12.72 |
| SLM+ (RAG) | 39.89 | 21.65 | 23.18 | 12.84 | 25.50 | 12.36 |
| LLM (32B) | 40.18 | 22.17 | 31.18 | 14.78 | 29.46 | 12.64 |
| **SPECSTEER** | **41.35** | **23.57** | **33.03** | **17.26** | **30.79** | **12.88** |

✅ **全面超越所有基线**，尤其在 **Review 写作任务**上提升显著（+1.85 R1 vs LLM）。

---

### **与基线方法的对比结果**
#### **Table 4：效率与质量权衡分析**
| 方法 | Rev R1 | Acceptance (%) | Speed (tok/s) | Speedup |
|------|--------|----------------|---------------|---------|
| Vanilla LLM | 31.18 | — | 22.58 | 1.00× |
| CoSteer | 32.73 | — | 9.71 | 0.43× |
| LightCoSteer | 32.61 | — | 16.03 | 0.71× |
| Standard SD | 31.24 | 29.02 | 22.13 | 0.98× |
| **SPECSTEER (λ=0.1)** | **32.38** | **81.46** | **53.29** | **2.36×** |

- **SPECSTEER 实现了质量与效率的双重优势**：
  - 保持高质量输出（接近持续融合方法）
  - 达成 **2.36× 推理加速**，远超其他协作方法。

---

### **消融实验结果**
#### **A. 不同 Specialist 质量下的鲁棒性（Table 7）**
| 设置 | Rev R1 (SLM+) | Rev R1 (SpecSteer) |
|------|----------------|--------------------|
| 噪声注入（Noise） | 23.54 | **31.78** |
| BM25 检索 | 23.18 | **33.03** |
| BGE 检索 | 25.35 | **33.45** |

➡️ 即使本地模型输入质量差，SPECSTEER 仍能通过云端验证与恢复机制**有效纠正错误**。

#### **A. 超参数敏感性分析**
- **β（恢复强度）**：在 `[0.5, 2.0]` 范围内稳定，过大（>2.5）会导致偏离大模型先验，降低连贯性。
- **λ（验证阈值）**：`λ ∈ [0.1, 0.5]` 是最优区间，平衡质量与接受率。

#### **跨架构部署（Table 8）**
- 使用 **Qwen3-0.6B (Specialist)** + **Llama-3.1-8B (Generalist)**：
  - Review R1 从 23.18（SLM+）和 31.71（LLM）提升至 **32.03**
  - 表明 SPECSTEER **不依赖特定模型家族**，具有强泛化性。

---

## **4. 关键结论和发现**

### **论文的主要发现**
1. **本地模型存在“能力赤字”（Capacity Deficit）**：  
   即使使用 RAG、LoRA 等技术增强，小模型也无法在复杂推理任务上匹敌大模型，**局部优化不足以解决根本问题**。

2. **SPECSTEER 成功弥合了这一差距**：  
   通过将 Speculative Decoding 重构为**分布式对齐协议**，实现了：
   - 高质量个性化生成
   - 严格隐私保护
   - 显著推理加速（2.36×）

3. **验证机制是关键创新**：  
   - **Ratio-based verification** 解决了因信息不对称导致的误拒问题。
   - **Steering recovery** 确保即使修正也能保留用户意图。

4. **系统高效且实用**：  
   - 通信开销极低（仅传输 token ID 和稀疏 logit 向量）
   - 支持异构部署、抗噪声、资源消耗低（FLOPs 下降近 3.5×）

---

### **方法的局限性**
- 当本地 Specialist 完全失效（如 fine-tuning 灾难性遗忘）时，SPECSTEER 的增益会减弱。
- 依赖云端大模型的可用性和网络连接，不适合完全离线场景。
- 虽然保护了输入隐私，但输出内容仍可能间接泄露用户信息，需结合其他隐私保护机制。

---

### **未来工作方向**
- 结合 **Federated Learning** 或 **Differential Privacy** 进一步强化隐私保障。
- 扩展到多模态个性化生成（如图像、语音）。
- 动态调整 `λ` 和 `β` 以适应不同任务复杂度。
- 探索在边缘设备上部署轻量化 Generalist 的可能性，实现去中心化协作。

---

> **总结一句话**：  
> **SPECSTEER 通过“起草-验证-恢复”范式，首次实现了在不牺牲隐私的前提下，让小模型“借力”大模型完成高质量个性化生成，并大幅提升推理效率，为现实世界的 edge-cloud 个性化智能提供了可扩展的解决方案。**

</details>

---

### 2. [Mask Is What DLLM Needs: A Masked Data Training Paradigm for Diffusion LLMs](https://arxiv.org/abs/2603.15803)

**Authors**: Linrui Ma, Yufei Cui, Kai Han, Yunhe Wang  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 7.0  
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
现有的 **Discrete Diffusion Language Models (DLLMs)** 在训练中普遍采用**Uniform Random Masking**策略，即对序列中的所有 token 以相同概率进行掩码。这种做法忽略了真实文本中信息密度的高度非均匀性：
- 高信息密度区域（如代码中的控制流语句、数学题中的关键运算步骤）决定了任务成败；
- 低信息密度区域（如语法连接词、标点）虽多但冗余。

传统方法导致模型将优化资源浪费在“语法胶水”上，而对关键逻辑节点学习不足，限制了其在复杂推理任务上的表现。

---

### 🚀 提出的新方法：Information Density Driven Smart Noise Scheduler

#### 核心思想
受人类认知测试（Cloze Test）启发，提出一种**信息密度感知的智能噪声调度机制**，让模型优先学习最关键的内容。

#### 创新设计
1. **信息密集区提取（Info-Dense Region Extraction）**
   - 使用 LLM（如 GPT-4o）作为离线标注器，识别 Code 和 Math 数据中的高信息密度片段：
     - **Code**: 控制流条件（if/while）、算法枢纽点
     - **Math**: 核心数学操作、中间/最终结果
   - 输出一个二值指示向量 $ C \in \{0,1\}^N $，标记每个 token 是否属于“优先掩码区”。

2. **互补优先掩码（Complementary Priority Masking）**
   - 引入偏差权重 $ w > 1 $，使高密度区域被掩码的概率是普通区域的 $ w $ 倍；
   - 动态调整基础掩码率 $ p_{\text{base}} $，确保整体掩码比例仍符合原始调度计划；
   - 对同一输入生成两个互补样本：
     - **Logical Sample**：优先掩码信息密集区 → 强化逻辑推理能力
     - **Syntactic Sample**：保留信息密集区，掩码其余部分 → 强化语言结构连贯性

该方法实现了**训练目标的密度感知解耦（Density-Based Decoupling）**，兼顾深层推理与表层流畅性。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（Uniform Random Masking） | 本文方法（Density-Driven Masking） |
|------|-------------------------------|----------------------------------|
| 掩码策略 | 输入无关、各位置等概率 | 输入相关、基于信息密度动态加权 |
| 学习重点 | 平均主义，易忽略关键点 | 聚焦核心逻辑跳跃点 |
| 数据利用效率 | 低效并行学习 | 双路径协同优化（逻辑+语法） |
| 性能提升 | 有限 | 显著提升复杂推理任务表现 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **Fine-tuning 数据混合体（约 450K 样本）**：
  - **OPC-SFT-Stage2**（Code domain）
  - **GSM8K**（Math domain）
- **评估基准（Benchmarks）**：
  - **Code**:
    - **HumanEval**（Pass@1）
    - **MBPP**（Pass@1）
  - **Math**:
    - **GSM8K**（Accuracy）
    - **MATH500**（Accuracy）

> 注：SFT 阶段未引入 RLHF 或对齐惩罚，因此存在 domain shift 现象。

---

### ⚙️ 实验设置
- **Base Model**: LLaDA-2.0-mini（Diffusion-based LLM）
- **Training Framework**: dFactory
- **Sequence Length**: 最大 2048；测试时最大生成长度 512
- **Block Diffusion Settings**:
  - Block Size: 32
  - Noise Ratio $ \alpha_t \in [0.3, 0.8] $
  - Timestep Steps: $ T = 32 $
- **Batch Size**: Global batch size = 16
- **训练周期**: 1 epoch

---

### 🔁 基线方法对比
| 方法 | 描述 |
|------|------|
| **Original** | 未经训练的原始模型 |
| **Baseline (w=1)** | 同样数据下使用标准随机掩码（无信息密度先验） |
| **Ours (w=2)** | 提出的方法，带互补优先掩码，$ w=2 $，Code 数据处理 10%，Math 处理 50% |

---

## 3. 主要实验结果和性能指标

### 📊 总体性能对比（Table 1）

| Method | HumanEval | MBPP | GSM8K | MATH500 | **AVG** |
|--------|-----------|------|-------|---------|--------|
| Original | 50.00 | 55.00 | 86.58 | 40.80 | 58.10 |
| Baseline | 57.93 | 56.80 | 69.14 | 37.40 | **55.32** |
| **Ours (w=2)** | **65.24** | 54.00 | 73.92 | **43.60** | **59.19** |

> 💡 **平均准确率提升 +3.87%（≈4%）**

- 在 **HumanEval** 上提升 **+7.31%**
- 在 **MATH500** 上提升 **+6.20%**
- 尽管 MBPP 微降，但整体显著优于 Baseline

---

### 🔍 消融实验结果

#### （1）不同偏差权重 $ w $ 的影响（Figure 3）
- 当 $ w = 1 $：退化为 Baseline
- $ w = 2 $ 与 $ w = 0.5 $ 表现几乎对称且最优（AVG ~59.2）
  - 因为互补掩码机制使得 $ w $ 与其倒数 $ 1/w $ 在期望上诱导相同的分布
- 极端值（$ w=5 $ 或 $ w=0.1 $）导致性能下降至 ~56.05
  - 结论：适度的信息倾斜最优，过强先验破坏 ELBO 优化

#### （2）Hard vs. Soft Priority Masking（Table 2）
| 类型 | AVG |
|------|-----|
| Hard Sample（确定性优先覆盖） | ~57.35 |
| **Soft Priority (w=2)** | **~59.45** |

> ❗ Hard Masking 导致“上下文坍塌（contextual collapse）”——连续大片信息黑洞使 block diffusion 梯度不稳定，训练失败风险高。

#### （3）数据预处理比例的影响（Data Scaling Effect）
- 仅对 **10% 的 Code 数据** 进行信息密度标注即可带来巨大性能跃升（AVG 达 59.45）
- 随着处理比例上升至 100%，性能趋于饱和甚至下降
- 特别地，在 **100% Code 数据处理** 下：
  - HumanEval 达峰值 67.07
  - 但 MATH500 下降至 40.40（从 44.20 跌落）

> ⚠️ 发现：过度注入 Code 结构先验会加剧 domain shift，损害跨领域泛化能力。

#### （4）是否使用互补掩码（Figure 4）
- 若**不使用互补掩码**：
  - $ w=2 $（重掩码关键点）表现差（AVG ~57.0）
  - $ w=0.5 $（轻掩码关键点）更稳定（AVG ~58.5）
  - 峰值仍低于完整方法（红虚线）
- 结论：**Complementary Masking 是实现最优性能的关键组件**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **信息密度非均匀性是 DLLM 训练的重要瓶颈**  
   忽视这一点会导致关键逻辑点学习不足。

2. **信息密度驱动的掩码策略可显著提升推理能力**  
   平均提升达 **+4%**，尤其在复杂逻辑任务（如 MATH500）上效果明显。

3. **互补掩码机制有效平衡逻辑与语法学习**  
   实现了“双轨制”训练目标解耦，避免单一视角带来的优化偏移。

4. **Soft Probabilistic Masking 比 Hard Deterministic 更鲁棒**  
   可防止 block diffusion 中的 contextual collapse。

5. **极高的数据效率**  
   仅需对 **10% 数据**进行离线标注即可接近最佳性能，极具实用价值。

---

### ⚠️ 局限性
1. **依赖外部 LLM 进行信息标注**  
   当前使用 GPT-4o 作为 extractor，增加成本与外部依赖。
   
2. **领域偏移风险（Domain Shift）**  
   过度强调某一领域（如 Code）的结构先验可能削弱其他任务（如 Math）的表现。

3. **规则定义依赖人工经验**  
   info-dense 区域的定义（如“控制流”、“关键结果”）尚属启发式，缺乏统一理论框架。

---

### 🔮 未来工作方向
1. **构建自包含系统（Self-contained Ecosystem）**
   - 探索基于 **AST（Abstract Syntax Tree）** 的规则匹配用于程序类数据
   - 设计端到端可学习的 masking module，根据模型自身 loss landscape 动态发现高密度区域

2. **动态适应性调度器**
   - 根据训练阶段自动调节 $ w $ 或掩码比例
   - 引入 curriculum learning 思想，由浅入深引导推理

3. **扩展至更多任务类型**
   - 如法律推理、科学问答等同样具有高信息密度跳跃的任务

4. **降低标注成本**
   - 探索弱监督或主动学习策略选择最具价值的样本进行标注

---

## ✅ 总结一句话
> 本文提出了一种**信息密度感知的掩码训练范式（Density-Driven Masking + Complementary Sampling）**，在不改变模型架构的前提下，通过 smarter data usage 显著提升了 Diffusion LLM 在 Code 和 Math 推理任务上的表现，平均提升 **+4%**，并揭示了 block diffusion 中 contextual collapse 的机理，为下一代高效推理型语言模型提供了新路径。

</details>

---

### 3. [NextMem: Towards Latent Factual Memory for LLM-based Agents](https://arxiv.org/abs/2603.15634)

**Authors**: Zeyu Zhang, Rui Li, Xiaoyan Zhao, Yang Zhang, Wenjie Wang, Xu Chen, Tat-Seng Chua  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 6.5  
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
现有的 **LLM-based Agents** 在管理 **factual memory**（事实记忆）时面临两大瓶颈：
- **Textual memory** 方法将记忆以原始文本形式存储，导致 **context length 膨胀** 和 **indexing overhead 高昂**，影响推理效率。
- **Parametric memory** 方法通过修改模型参数来存储信息，容易引发 **catastrophic forgetting**（灾难性遗忘）且存储成本高。

这些方法难以实现对大量细节事实的 **高效、无损保存与重建**。

### 提出的新方法与思路
本文提出 **NextMem** —— 一种基于 **latent factual memory** 的新型框架，其核心思想是：
- 将文本事实编码为紧凑的 **latent representations**（潜在表示），从而压缩存储空间并降低上下文负担。
- 设计一个 **autoregressive autoencoder** 架构，支持从文本到潜变量再到文本的 **可逆转换**，确保信息重建的高保真度。
- 引入 **两阶段训练策略**：
  1. **Autoregressive Reconstruction Alignment**：先让模型学会从输入文本自回归地重建自身。
  2. **Progressive Latent Substitution**：逐步用 latent tokens 替换原始文本块，迫使编码器生成可用于解码的高质量 latent memory。
- 结合 **NF4 量化**（4-bit NormalFloat）进一步压缩 latent 表示，显著降低存储开销。

### 相比现有方法的优势
| 维度 | NextMem | Textual / Parametric 方法 |
|------|--------|--------------------------|
| 存储效率 | ✅ 高效压缩至 15 个 latent tokens | ❌ 上下文长或参数更新代价大 |
| 重建精度 | ✅ 几乎无损（F1 > 0.94） | ❌ 重建能力弱（如 DyPRAG ≈ 0.03） |
| 检索能力 | ✅ latent 可直接用于 retrieval | ❌ 需额外索引结构 |
| 扩展性 | ✅ 支持 out-of-distribution 长序列 | ⚠️ 性能随长度快速下降 |

此外，NextMem 实现了 **memory storage 与 retrieval 的统一**，简化了 agent 架构设计。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖多个与 memory 相关的任务场景，使用的数据集包括：

| 数据集 | 用途说明 |
|-------|---------|
| **SQuAD** | 单跳问答，测试基本阅读理解能力 |
| **HotpotQA** | 多跳推理任务，需跨文档整合信息 |
| **RACE** | 来自英语考试的阅读理解题，强调逻辑推理 |
| **LoCoMo** | 模拟多轮对话中的长期记忆保持 |
| **LongMemEval** | 用户-代理交互场景下的长期记忆评测 |

所有数据均经过标准化处理，提取 reference text 并进行采样与截断，模拟不同长度的记忆片段。

### 实验设置与评估指标

#### 主要任务划分：
| 任务 | 对应功能 | 评估目标 |
|-----|--------|--------|
| **Task 1: Factual Reconstruction** | Memory Storage | 测试 latent memory 是否能准确还原原文 |
| **Task 2: Contextual Generation** | Memory Utilization | 测试 memory 是否有助于下游任务生成 |
| **Task 3: Dense Passage Retrieval** | Memory Retrieval | 测试 latent memory 是否具备检索能力 |

#### 评估指标：
- **Factual Reconstruction**:  
  `F1`, `ROUGE-1`, `ROUGE-L`, `METEOR`, `BLEU`, `BertScore`
- **Contextual Generation**:  
  使用 `LLM-as-Judge` 判断生成答案质量，报告 **Accuracy**
- **Dense Passage Retrieval**:  
  `Hit@5`, `Recall@5`, `MRR@5`, `MAP@5`, `DCG@5`, `NDCG@5`

#### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **DyPRAG** | Parametric Memory | 在测试时生成 LoRA adapter 注入知识 |
| **DeepSeek-OCR** | Vision-Language Compression | 将文本转为图像再由 LLM OCR 解码 |
| **ICAE** | Context Compression | 使用可学习 memory tokens 压缩段落 |
| **Textual Memory** | Oracle Baseline | 原始文本作为上下文输入（理想情况） |
| **BGE** | Embedding Model | 仅用于 retrieval 对比，无法重建 |

模型统一使用 **Qwen3-8B** 作为 backbone，chunk size 设为 128 tokens。

---

## 3. 主要实验结果和性能指标

### Task 1: Factual Reconstruction（记忆存储）
在多个数据集上，NextMem 显著优于所有 baseline：

| 方法 \ 数据集 | HotpotQA (F1) | RACE (F1) | SQuAD (F1) | LoCoMo (F1) | LongMemEval (F1) |
|-------------|---------------|-----------|------------|--------------|------------------|
| DyPRAG      | 0.0305        | 0.0696    | 0.0493     | 0.0901       | 0.1338           |
| DeepSeek-OCR| 0.4540        | 0.4068    | 0.3657     | 0.5179       | 0.4685           |
| ICAE        | 0.7890        | 0.6077    | 0.7084     | 0.6986       | 0.7015           |
| **NextMem-Dense** | **0.9820**    | **0.8552**| **0.8920** | **0.9611**   | **0.9436**       |
| **NextMem-Sparse**| **0.9805**    | **0.8554**| **0.8860** | **0.9615**   | **0.9362**       |

✅ **结论**：NextMem 实现接近完美的重建能力，且量化后性能几乎不变。

---

### Task 2: Contextual Generation（记忆利用）

| 方法 \ 设置 | ICAE (Comp.) | ICAE (DeComp.) | **NextMem-Dense (DeComp.)** | **NextMem-Sparse (DeComp.)** |
|-----------|--------------|----------------|-------------------------------|-------------------------------|
| HotpotQA  | 0.8565       | 0.8229         | **0.8072**                    | **0.8184**                    |
| SQuAD     | 0.7775       | 0.7066         | **0.7572**                    | **0.7630**                    |
| LoCoMo    | 0.5407       | 0.5215         | **0.5407**                    | **0.5263**                    |
| LongMemEval| 0.4971      | 0.5029         | **0.5400**                    | **0.5486**                    |

📌 注意：
- ICAE 在 **Comp.**（直接使用 latent）表现最好，说明其 latent 更适合直接推理。
- NextMem 在 **DeComp.**（先重建再推理）中全面领先，体现其 **重建保真度优势**。
- 存在 trade-off：高重建精度 ≠ 最佳 latent 推理性能。

---

### Task 3: Dense Passage Retrieval（记忆检索）

| 方法 \ 数据集 | HotpotQA (Hit@5) | LoCoMo (Hit@5) | LongMemEval (Hit@5) |
|-------------|------------------|----------------|----------------------|
| DeepSeek-OCR| 0.3358           | 0.0676         | 0.4200               |
| ICAE        | 0.4453           | 0.1210         | 0.5480               |
| **NextMem-Dense** | **0.7245**       | **0.4377**     | **0.8220**           |
| **NextMem-Sparse**| **0.7208**       | **0.4342**     | **0.8140**           |
| BGE (oracle)| 0.9585           | 0.8007         | 0.8960               |

✅ NextMem 在 retrieval 上远超其他 memory 模型，接近专用 embedding model（BGE）水平。

---

### 消融实验（Ablation Study）

在 RACE 数据集上验证各组件作用：

| 方法 | F1 | ROUGE-L | METEOR | BertScore |
|------|----|---------|--------|-----------|
| **Dense (Full)** | **0.8552** | **0.8580** | **0.8691** | **0.9735** |
| w/o ST ([SoD] token) | 0.3799 | 0.3804 | 0.4048 | 0.7307 |
| w/o PT (no progressive training) | 0.0159 | 0.0138 | 0.0169 | 0.7686 |
| w/o PS (no progressive expansion) | 0.7389 | 0.7358 | 0.7353 | 0.9502 |
| w/o SQ (no scale in quantization) | 0.0309 | 0.0290 | 0.0442 | 0.7521 |

🔍 发现：
- **Progressive Latent Substitution** 是最关键的设计，移除后性能崩溃。
- `[SoD]` token 对齐编码边界至关重要。
- 量化中的 scaling 机制极大影响重建稳定性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Latent memory 可实现高保真 factual memory 存储**：NextMem 能将文本压缩为极短 latent tokens，并几乎无损重建。
2. ✅ **Latent representation 具备良好 retrieval 属性**：无需额外索引即可用于相似性匹配，实现“一码双用”。
3. ✅ **渐进式训练策略有效解决优化难题**：一次性 latent substitution 难以收敛，而逐步替换可稳定提升性能。
4. ✅ **NF4 量化几乎不损失性能**：结合 scale vector 后，4-bit 量化仍能维持高质量重建。
5. 🔍 **存在 reconstruction 与 direct utilization 的权衡**：当前 latent 空间更适合解压后使用，而非直接推理。

### 方法的局限性
- **latent length 固定为 15 tokens**，可能限制极端长文本的记忆容量。
- 当前 latent space 不支持高效的 **fine-grained editing**（尽管语义映射有序，见 Figure 5）。
- 依赖预训练 LLM backbone，迁移性受限于架构兼容性。
- 在极高噪声（σ ≥ 1.6）下出现 **semantic collapse**，鲁棒性仍有提升空间。

### 未来工作方向
- 探索 **latent memory 的 instruction-following 能力优化**，缩小 Comp./DeComp. 差距。
- 开发 **adaptive latent length** 机制，动态调整 memory 容量。
- 研究 **latent memory editing** 与 **selective forgetting** 机制。
- 将 NextMem 集成到真实 agent workflow 中，测试端到端任务表现。

---

> 📦 **开源信息**：作者已公开代码与模型 checkpoint：  
> 🔗 https://github.com/nuster1128/NextMem

</details>

---

### 4. [FactorEngine: A Program-level Knowledge-Infused Factor Mining Framework for Quantitative Investment](https://arxiv.org/abs/2603.16365)

**Authors**: Qinhong Lin, Ruitao Feng, Yinglun Feng, Zhenxin Huang, Yukun Chen, Zhongliang Yang, Linna Zhou, Binjie Fei, Jiaqi Liu, Yu Li  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.16365v1  

#### Abstract
We study alpha factor mining, the automated discovery of predictive signals from noisy, non-stationary market data-under a practical requirement that mined factors be directly executable and auditable, and that the discovery process remain computationally tractable at scale. Existing symbolic approa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对**量化投资中的 alpha 因子挖掘**（alpha factor mining）任务，旨在从高噪声、非平稳的市场数据中自动发现具有预测能力且可执行、可审计的因子。现有方法面临以下挑战：
- **Symbolic 方法**（如 GP、手工因子）表达能力受限，搜索空间有限，难以捕捉复杂市场动态。
- **Neural 方法**虽然性能强，但缺乏可解释性，易过拟合，且对 regime shift 敏感。
- **LLM 驱动的方法**虽能生成创意，但常将逻辑演化与参数优化混为一谈，导致效率低下、计算成本高。

### **提出的新方法与新思路**
作者提出了 **FactorEngine (FE)** ——一个**程序级知识注入的因子挖掘框架**，其核心创新在于：

#### ✅ **程序级因子表示（Program-Level Representation）**
- 将因子定义为 **Turing-complete 的 Python 程序**，而非传统的数学表达式或神经网络。
- 支持条件控制流、循环、函数调用等复杂结构，极大提升表达能力和灵活性。

#### ✅ **宏-微协同进化机制（Macro-Micro Co-evolution）**
- **宏观演化（Logic Evolution）**：由 LLM 代理负责语义推理与结构修改（macro mutation），提出新的因子逻辑。
- **微观优化（Parameter Optimization）**：通过 **Bayesian Optimization** 自动搜索最优超参数（如窗口大小、衰减系数），实现高效局部调优。
- 二者解耦，避免 LLM 浪费资源在数值搜索上。

#### ✅ **知识注入引导启动（Knowledge-Infused Bootstrapping）**
- 构建闭环多智能体系统，从**非结构化金融研报**中提取核心思想，经验证后自动生成可执行的初始因子程序。
- 初始池不仅包含人工设计因子，还融合了真实研究报告中的经济逻辑，提高起点质量。

#### ✅ **经验链反馈机制（Chain of Experience, CoE）**
- 在进化过程中记录完整轨迹（包括失败尝试），形成“经验链”供 LLM 学习。
- 支持**从失败中学习恢复策略**，增强鲁棒性和探索效率。

#### ✅ **多岛并行进化架构（Multi-Island Evolution）**
- 多个独立进化进程并发运行，定期迁移优秀个体，促进多样性传播，防止早熟收敛。

---

### **相比现有方法的优势**
| 维度 | FactorEngine | 传统方法（GP/LLM-based） |
|------|-------------|--------------------------|
| 表达能力 | ✅ 图灵完备代码，支持复杂逻辑 | ❌ 受限于预定义算子集 |
| 可解释性 | ✅ 因子为可读 Python 代码 | ⚠️ Symbolic 可读但弱；Neural 黑箱 |
| 效率 | ✅ LLM 仅用于逻辑创新，参数交由本地 BO 并行优化 | ❌ LLM 承担过多低效试错 |
| 知识利用 | ✅ 主动解析研报转化为可执行因子 | ❌ 仅作为灵感提示 |
| 多样性 | ✅ 显著更高的因子非冗余性（见 MDS 分析） | ❌ 容易陷入局部模式 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 数据来源：**Qlib 框架提供的全市场 OHLCV 数据**
- 覆盖范围：中国 A 股市场
- 时间划分：
  - **训练期**：2008-01-01 至 2014-12-31
  - **验证期**：2015-01-01 至 2016-12-31
  - **测试期**：2017-01-01 至 2024-12-31（共 8 年）
- 市场指数：
  - **CSI300**
  - **CSI500**

> 🔒 **防泄漏措施**：所有用于知识注入的金融研报均发布于 **2017 年之前**，确保无未来信息泄露。

---

### **实验设置**
- **迭代轮数**：200 和 400 轮两种设定
- **每轮生成因子数**：1 个
- **初始种子**：
  - `FE-alpha`：基于手动因子初始化
  - `FE-report`：基于研报解析生成的因子初始化
- **多岛配置**：2 个岛屿，每 7 轮迁移 top-3 因子
- **Backbone LLM**：Gemini-2.5-Pro（公平比较所有 agent 方法）

### **评估指标**

#### 📊 **预测性能指标**
| 指标 | 全称 | 含义 |
|------|------|------|
| **IC** | Information Coefficient | 因子值与未来收益的横截面相关性（Pearson） |
| **ICIR** | IC Information Ratio | IC 的时序稳定性（均值 / 标准差） |
| **RIC** | Rank IC | 排名相关性（Spearman） |
| **RICIR** | Rank ICIR | RIC 的稳定性 |

#### 💼 **组合绩效指标**
| 指标 | 全称 | 含义 |
|------|------|------|
| **AR** | Annualized Return | 年化收益率 |
| **IR** | Information Ratio | 单位风险超额收益 |
| **MDD** | Maximum Drawdown | 最大回撤 |
| **SR** | Sharpe Ratio | 夏普比率（年化） |

> 所有组合策略统一采用：
> - Top-50 等权持仓
> - 持有周期 5 天
> - 交易成本建模（佣金 + 印花税 + 滑点）
> - 使用 **LightGBM** 对生成因子 + Alpha158 进行集成建模

---

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Symbolic GP** | GPlearn |
| **Neural Forecasting** | LightGBM, LSTM, Transformer |
| **Specialized Model** | TRA（Temporal Routing Adaptor） |
| **Agent-Based LLM 方法** | AlphaAgent, RD-Agent-Quant |
| **Handcrafted Baseline** | Alpha158 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（以 400 轮迭代为准）**

#### ✅ **CSI300 市场表现（FE-report-2 最优）**
| 方法 | IC | ICIR | AR | SR | MDD |
|------|-----|-------|--------|-------|--------|
| Alpha158 | 0.0299 | 0.2008 | 0.0840 | 0.4196 | 17.49% |
| RD-Agent-2 | 0.0255 → 0.0269 | ↓ | 0.0507 → 0.0917 | ↓ | ↑ |
| AlphaAgent-2 | 0.0282 → 0.0269 | ↓ | 0.0673 → 0.0917 | ↓ | ↑ |
| **FE-report-2** | **0.0474** | **0.3185** | **18.99%** | **1.0093** | **12.61%** |

> 💡 **相比 Alpha158 提升显著**：
> - **IC 提升 58%**
> - **年化超额回报提升 126%**
> - **夏普比率翻倍以上**

#### ✅ **CSI500 市场表现**
| 方法 | IC | ICIR | AR | SR | MDD |
|------|-----|-------|--------|-------|--------|
| Alpha158 | 0.0403 | 0.3100 | 0.0197 | 0.2152 | 25.17% |
| **FE-report-2** | **0.0536** | **0.4140** | **8.36%** | **0.6719** | **21.51%** |

> 同样全面领先，尤其在小盘股市场表现出更强适应性。

---

### **与基线方法的对比结果**
- 在 **所有预测与组合指标上，FE 均取得 SOTA 性能**。
- 即使其他 agent 方法随迭代增加有所改善，**FE 始终保持领先优势**。
- `FE-report` > `FE-alpha`，说明**研报知识注入有效提升了因子质量和稳定性**。
- 如图 3 所示，**FE 的累计超额收益曲线平滑上升，回撤更小**。

---

### **消融实验结果**

#### 🔍 **Ablation 1: Bayesian 微搜索的作用**
- 对比 variant：`w/ Bayes` vs `w/o Bayes`（固定参数）
- 结果显示：
  - 引入 Bayesian 参数优化后，最终性能提升约 **52%**（0.38 vs 0.25）
  - 收敛速度更快，fitness 曲线上升更陡峭
- ✅ **结论**：微调阶段的自动化参数搜索对整体性能至关重要

#### 🔍 **Ablation 2: Backbone LLM 影响**
- 使用 GPT-4o 和 Gemini-2.5-flash 作为 backbone
- 发现 GPT-4o 整体表现更强，但 **FE 架构本身增益远大于 LLM 差异**
- ✅ **结论**：FE 的有效性不依赖特定 LLM，具备通用性

#### 🔍 **Ablation 3: 配置影响（Island 数量 & Prompt 设计）**
| 配置 | RIC | AR | IR |
|------|------|------|------|
| 1-island + top-k | 0.0341 | 0.0761 | 0.6944 |
| **2-island + CoE** | **0.0346** | **0.0888** | **0.7886** |
| **2-island + CoE + 10 init** | **0.0344** | **0.0943** | **0.8241** |

> ✅ 多岛 + 经验链（CoE）prompt 显著提升性能

---

## **4. 关键结论和发现**

### **主要发现**
1. **程序级因子表示是可行且高效的路径**：将因子视为可执行代码，突破了 symbolic 方法的表达瓶颈。
2. **逻辑与参数分离优化大幅提升效率**：LLM 专注“想点子”，本地计算专注“调参数”，实现资源最优分配。
3. **知识注入显著提升起点质量**：从金融研报中提取逻辑生成初始因子，比随机或纯人工启动更具优势。
4. **经验记忆机制有助于抗衰减（anti-decay）**：FE 报告因子在 2021 年后停止衰减甚至回升，而其他方法持续下滑。
5. **因子多样性更高**：MDS 可视化显示 FE 因子分布呈“环形扩散”，相互独立性强，冗余度低。

---

### **方法的局限性**
- **依赖高质量研报输入**：若报告本身存在偏见或错误，可能污染初始池。
- **LLM 成本仍不可忽视**：尽管已优化，大规模部署仍需考虑 API 开销。
- **未显式建模因子间交互**：当前集成方式为简单拼接 + LightGBM，未来可引入更复杂的组合模型。
- **中国市场特定假设**：交易规则、涨跌停限制等影响策略设计，跨市场泛化需进一步验证。

---

### **未来工作方向**
1. **扩展至多模态数据**：纳入新闻、社交媒体、财报文本等非结构化信号。
2. **增强分布外鲁棒性**：研究如何应对市场 regime shift 和结构性变化。
3. **主动查询机制**：让 LLM 能够主动发起对历史数据的“提问”以验证假设。
4. **多样化度量与泛化分析**：建立更系统的指标来衡量因子集合的互补性和长期生命力。
5. **端到端交易成本感知优化**：将换手率、冲击成本直接嵌入 reward 函数。

---

> 🎯 **总结一句话**：  
> **FactorEngine 通过“程序即因子 + 知识引导 + 宏微分离”的范式革新，在保持可解释性的同时实现了 SOTA 的预测与组合表现，为下一代自动化因子挖掘提供了新范式。**

</details>

---

### 5. [Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies](https://arxiv.org/abs/2603.15903)

**Authors**: Nathaniel Imel, Richard Futrell, Michael Franke, Noga Zaslavsky  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.15903v1  

#### Abstract
Natural languages have been argued to evolve under pressure to efficiently compress meanings into words by optimizing the Information Bottleneck (IB) complexity-accuracy tradeoff. However, the underlying social dynamics that could drive the optimization of a language's vocabulary towards efficiency ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Agent-based imitation dynamics can yield efficiently compressed population-level vocabularies*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文旨在解决语言演化中两个长期存在的开放性问题：
- **机制解释缺失**：尽管已有大量证据表明自然语言在语义系统上趋向于信息论意义上的高效压缩（即满足 Information Bottleneck, IB 的复杂度-准确性权衡），但尚不清楚这种效率是如何通过群体层面的文化演化过程实现的。
- **理论框架割裂**：进化博弈论（Evolutionary Game Theory, EGT）被广泛用于建模信号系统的涌现（如 Lewis-Skyrms signaling games），但它是否能导向 IB 意义上的全局最优仍未知。

### 提出了什么新方法或新思路
作者提出了一种**统一的理论与计算模型**，将以下两个独立发展的框架整合起来：
- **Information Bottleneck (IB)**：作为衡量语言效率的形式化标准（来自信息论与认知科学）；
- **Noisy Sim-Max Signaling Games + Imprecise Imitation Dynamics**：一种基于 agent 的文化演化动态（来自 EGT 和语言演化研究）。

具体而言，他们采用 Franke & Correia (2018) 提出的带有感知噪声的 *sim-max signaling game* 及其对应的 *imprecise conditional imitation dynamic*，并将其置于 IB 的评估框架下进行分析。

### 相比现有方法的优势
| 方面 | 本文优势 |
|------|--------|
| **理论整合性** | 首次明确建立 EGT 成功策略与 IB 信息效率之间的联系，弥合了局部互动成功与全局信息最优之间的鸿沟。 |
| **机制简洁性** | 不依赖深度学习架构、先验假设或强理性假设，仅通过简单的模仿动态即可涌现出近似最优的语言系统。 |
| **生态合理性** | 引入状态混淆（state confusion）和模糊模仿（noisy imitation），更贴近人类感知与社会学习的真实限制。 |
| **可解释性高** | 动态过程透明，参数具有明确的心理/社会意义（如 γ 控制交际精度标准）。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
本研究未使用真实自然语言数据集，而是构建了一个**合成语义域（synthetic domain）**：
- **领域设定**：理想化的数量表达系统（numerosity），模拟从“一”到“一百”的数量词演化。
- **状态空间**：$ X = \{0, 1, ..., 99\} $，共 100 个连续数值状态。
- **信号空间**：同样有 100 个可用词汇 $ W $，允许一对一映射（exact）或粗略划分（approximate）。

此设定便于控制变量，并与 IB 理论中的编码任务直接对应。

### 实验设置和评估指标

#### 动态模型
- 采用 **imprecise conditional imitation dynamic**（基于 replicator dynamics 扩展）
- 包含两个群体：Sender population 和 Receiver population
- 每个 agent 拥有确定性策略（pure strategy），集体行为表现为概率分布 $ S(w|x_o) $, $ R(x_i|w) $
- 模拟最多 $10^5$ 步，直到收敛（变化小于 $10^{-5}$）

#### 关键参数
- **γ (gamma)**：控制交际奖励函数的锐度（pragmatic standard of precision），定义为相似性函数 $ \text{sim}(x,x') = \exp(-\gamma(x-x')^2) $
  - γ 越大 → 对精确匹配要求越高
- **σ (sigma)**：感知混淆程度（固定为 0.5），决定 $ p(x'|x) $

#### 评估指标
| 指标 | 定义 | 目标 |
|------|------|------|
| **Complexity** $ I(M_o; W) $ | 编码所需的信息量（越低越好） | 最小化 |
| **Accuracy** $ I(W; X_a) $ | 接收端恢复原始状态的能力（越高越好） | 最大化 |
| **Efficiency Loss (ε)** | 当前系统偏离 IB 理论最优界的距离 $ \epsilon = \min_\beta \Delta F_\beta $ | 越小越好 |
| **Population Fitness** | 团队期望效用（expected similarity） | 衡量博弈成功 |

#### 基线方法对比
1. **Randomly Permuted Systems**  
   - 将最终系统中的词义映射随机打乱，检验效率是否源于结构而非偶然。
2. **NK99 Dynamic (Nowak & Krakauer, 1999)**  
   - 经典有限群体复制-变异模型，常用于语言演化模拟。
   - 差异：单一群体、无感知噪声、二元奖励、引入采样变异。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 几乎所有运行（800 次，不同 γ 和随机种子）都**快速收敛**（通常 < 40,000 步）。
- 收敛后的系统**非常接近 IB 理论边界**（见 Figure 2A），说明实现了近似最优压缩。
- 最大可达准确率约为 4.61 bits，但实际最高仅达 **< 4 bits**，表明存在根本性上限。

### 与基线方法的对比结果
| 方法 | 复杂度-准确性表现 | 效率损失 ε |
|------|------------------|-----------|
| **本文模型 (FC18)** | ✅ 极其接近 IB 边界 | ✅ 极低 |
| **Permutated Controls** | ❌ 分布散乱，远离边界 | ❌ 显著更高 |
| **NK99 Dynamic** | ❌ 偏离 IB 边界，复杂度过高 | ❌ 更差 |

> 结果显示：只有本文提出的 noisy imitation dynamic 能稳定产生 IB-高效系统。

### 消融实验与关键变量影响
虽然没有传统意义上的“消融”，但通过系统调节 γ 参数，揭示了以下规律：

#### （1）γ 控制 trade-off 位置
- γ ↑ → 系统趋向更高 complexity 与 accuracy
- γ ↓ → 系统趋向更简化、容忍更大误差
- **Spearman 相关性高达 ρ ≈ 0.99**，说明 γ 是调控效率权衡的关键机制参数。

#### （2）imitation noise 限制精度上限
- 即使当 γ → ∞（理论上应趋近完全精确通信），系统也无法达到 bijective mapping。
- 原因：**noisy imitation act as a regularizer**，软化类别边界，间接降低复杂度但也牺牲了最大可能准确性。

#### （3）效率非预设，而是涌现结果
- 模型中并无显式最小化 $ I(M;W) $ 的目标，但系统仍自动趋向低复杂度。
- 表明：**myopic utility maximization + noisy imitation → 自发正则化 → 近似 IB 最优**

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **局部模仿动态可以驱动全局信息效率**：即使个体只追求即时沟通成功率，群体也能自发演化出接近 IB 理论最优的语义系统。
2. ✅ **EGT 成功 ≠ IB 最优，但在特定动态下可以等价**：传统的 signaling game 若结合 imprecise imitation，确实能逼近信息论最优解。
3. ✅ **pragmatic standards (γ) 是关键调节参数**：它决定了最终系统在 IB 平面上的位置，为解释跨语言语义差异提供了微观机制基础。
4. ✅ **imitation noise 具有双重作用**：既是现实约束，也是一种隐式正则化机制，防止过拟合并促进压缩。

### 方法的局限性
- 🚫 **未使用真实语言数据验证**：目前仅在合成数量域测试，尚未应用于颜色、空间、亲属称谓等典型语义域。
- 🚫 **静态环境假设**：未考虑语境变化、递归结构或语法组合性。
- 🚫 **uniform prior 假设**：忽略了现实中高频词更易习得的现象（如幂律分布）。
- 🚫 **忽略个体认知偏差**：未整合人类学习中的归纳偏好（如 Imel et al., 2025 发现的 IB bias）。

### 未来工作方向
1. **扩展至更复杂的语义结构**：如递归数词系统（"twenty-one"）、复合词、语法范畴。
2. **引入非均匀 prior 和频率效应**：模拟真实语言使用频率对词汇压缩的影响。
3. **结合神经网络 agent**：在 deep RL 设置中复现该动态，增强与现代 emergent communication 研究的对话。
4. **实证检验预测**：设计心理实验，观察人类在类似 signaling game 中是否表现出与 γ 参数一致的行为梯度。
5. **探索数学关系**：形式化证明 imprecise imitation dynamic 在何种条件下收敛到 IB 解附近。

---

> **一句话总结**：  
> 本文证明，基于 agent 的不完美模仿动态（imprecise imitation in signaling games）足以作为一种**机制基础**，解释为何人类语言会普遍趋向信息论上高效的语义结构——这不仅是功能需求的结果，也是简单社会学习规则在群体中放大的自然产物。

</details>

---

### 6. [Prior-Informed Neural Network Initialization: A Spectral Approach for Function Parameterizing Architectures](https://arxiv.org/abs/2603.16376)

**Authors**: David Orlando Salazar Torres, Diyar Altinses, Andreas Schwung  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 6.5  
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
传统神经网络初始化方法（如 Xavier 或 Kaiming）是 **data-agnostic**（与数据无关）的，假设适用于通用深度架构。然而，在 **function-parameterizing architectures**（如 Bag-of-Functions, BoF）中，模型输出直接参数化数学函数以重建信号，这类模型对初始化极为敏感。随机初始化导致：
- 收敛缓慢
- 训练不稳定
- 参数漂移严重
- 性能跨试验波动大

现有基于启发式的方法缺乏泛化能力，依赖人工调参，无法自动适配不同数据的内在结构。

---

### 🚀 提出的新方法与创新思路
本文提出一种 **先验引导的神经网络初始化框架（Prior-Informed Initialization Framework）**，将数据的 **频谱（spectral）和时序（temporal）结构** 显式编码到模型的 **架构设计** 和 **参数初始化** 中。

#### 核心思想：
- **不把初始化看作孤立优化技巧，而是与数据结构对齐的设计过程。**

#### 具体方法：
1. **Spectral Analysis via FFT**  
   利用 **Fast Fourier Transform (FFT)** 提取主导季节性频率（dominant seasonal modes），用于：
   - 决定 **BoF 模型的堆叠阶段数 $S$**
   - 初始化 **seasonal encoder** 的参数分布（均值 $\mu_{\text{data}}$, 方差 $\sigma^2_{\text{data}}$）

2. **Residual-based Linear Regression for Trend**  
   在去除周期成分后，对残差进行线性回归估计趋势项（slope 和 bias），用于：
   - 推导 **trend encoder 输入维度 $N_{\text{in}}$** 的理论下界
   - 初始化 **trend encoder** 的参数

3. **理论支持**  
   提供有限样本下的回归误差分析，确保趋势估计在噪声环境中的可靠性。

---

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **收敛速度** | 显著加速训练，早期即进入低误差区域 |
| **稳定性** | 减少参数漂移和跨试验性能方差 |
| **效率** | 缩小 encoder 维度（最高降维 89%），降低 FLOPs 和参数量 |
| **无需修改训练流程** | 不改变优化器、损失函数或架构主体，仅通过初始化提升性能 |
| **自动化配置** | 自动决定模型深度和输入窗口大小，减少人工调参 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
| 数据集 | 类型 | 描述 |
|-------|------|------|
| **Synthetic Dataset** | 合成数据 | 包含已知频率（~3.5, 7.5, 12 Hz）、线性趋势和瞬态事件的信号，共 2000 条，每条 100 个时间点，用于验证先验提取准确性 |
| **PJM Hourly** | 真实世界 | 1998–2001 年电力需求数据，按周切片，采样频率为小时级 |
| **Thermal Power Plant (TPP)** | 真实世界 | 2016–2020 年区域能源供热系统输出功率，聚焦冬季数据，具有复杂周期性和趋势 |

---

### ⚙️ 实验设置
- **模型架构**：基于 **Bag-of-Functions (BoF)** 框架，分解为 seasonal、trend、event 三个组件
- **训练配置**：
  - 优化器：Adam ($lr = 1 \times 10^{-3}$)
  - 批次大小：16
  - 损失函数：MSE
  - 早停机制防止过拟合
  - 每组实验运行 **10 次独立训练**（不同随机种子）以评估稳定性

- **评估指标**：
  - **MSE（训练/测试）**：重建精度
  - **Parameter Displacement**：训练过程中参数变化总量，衡量优化稳定性
  - **Convergence Speed**：损失下降轨迹
  - **Computational Efficiency**：参数量（Params）、FLOPs、推理延迟（Latency）、吞吐量（Throughput）

---

### 🆚 对比的基线方法
| 方法 | 简称 | 特点 |
|------|------|------|
| 标准初始化 BoF | **BoF** | 使用默认 Xavier/Kaiming 初始化，固定架构 |
| 启发式初始化 BoF | **H-BoF** | 使用文献 [4] 中的偏置初始化策略，但仍为固定架构 |
| 数据驱动初始化 + 动态深度 | **I-BoF** | 季节性部分由 FFT 决定阶段数并初始化；趋势仍为标准配置 |
| 完整提出的框架 | **IT-BoF** | 季节性和趋势均数据驱动，动态决定深度和输入尺寸，全参数初始化 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table II 和 Fig. 7–10）

#### ✅ 在合成数据上的表现（Fig. 7）
| 方法 | 测试 MSE（平均 ± std） | 相对改进 |
|------|------------------------|---------|
| BoF | 0.7633 ± 0.1091 | — |
| H-BoF | 0.1937 ± ? | ~75% ↓ |
| I-BoF | 0.0198 ± 0.0034 | ~97% ↓ |
| **IT-BoF** | **0.0220 ± 0.0033** | **~97% ↓** |

> 💡 尽管 IT-BoF 的 trend 输入维度更小（nopt=11 vs full=100），性能仍与 I-BoF 相当，说明 **维度压缩未牺牲精度**。

---

#### ✅ 在真实数据上的重建性能（Table II）
| 数据集 | 方法 | 测试 MSE |
|-------|------|----------|
| **PJM** | BoF | 0.0155 ± 0.0008 |
|         | H-BoF | 0.0130 ± 0.0131 |
|         | I-BoF | 0.0099 ± 0.0024 |
|         | **IT-BoF** | **0.0074 ± 0.0011** (**↓52%**) |
| **TPP** | BoF | 0.4621 ± 0.0192 |
|         | H-BoF | 0.2309 ± 0.0481 |
|         | **I-BoF** | **0.1958 ± 0.0027** (**↓58%**) |
|         | IT-BoF | 0.2035 ± 0.0077 (**↓56%**) |

> ✅ IT-BoF 在 PJM 上表现最佳，I-BoF 在 TPP 上略优，但两者均显著优于基线。

---

#### ⏱️ 计算效率提升（Table IV）
| 数据集 | 方法 | Params (×10³) | FLOPs (×10³) | Throughput (samples/s) |
|-------|------|---------------|--------------|-------------------------|
| **Synthetic** | BoF/H-BoF/I-BoF | ~62 | ~61 | ~3800 |
|             | **IT-BoF** | **47.38** (**↓24%**) | **46.81** (**↓23%**) | **4012 (+5.6%)** |
| **PJM**     | I-BoF | ~63 | ~62 | 6144.8 |
|             | **IT-BoF** | **44.89** (**↓29%**) | **44.53** (**↓28%**) | **6223.9 (+1.3%)** |

> ✅ IT-BoF 实现 **20–30% 参数和计算量减少**，同时保持甚至提升吞吐量。

---

#### 📉 参数漂移（Parameter Displacement）
- Fig. 9 和 Fig. 10 显示：
  - BoF 和 H-BoF 参数变化剧烈且持续
  - I-BoF 和 IT-BoF 快速稳定，**IT-BoF 漂移最小**
- 表明：**先验引导使优化路径更平滑、更集中**

---

#### 🔍 消融实验（隐含于对比设计）
虽然未明确列出“ablation study”章节，但以下对比构成有效消融：
| 变体 | 是否使用 FFT 决定深度？ | 是否使用 OLS 初始化 trend？ | 性能趋势 |
|------|--------------------------|-------------------------------|----------|
| BoF | ❌ | ❌ | 最差 |
| H-BoF | ❌ | ❌（仅 heuristic bias） | 中等 |
| I-BoF | ✅ | ❌ | 好（尤其在 TPP） |
| IT-BoF | ✅ | ✅ | 最佳（尤其在 PJM）且最紧凑 |

> 结论：**两项先验引入均有增益，联合使用效果最优**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **数据驱动的先验显著改善 function-parameterizing 模型的训练行为**：
   - 加速收敛（early low-loss regime）
   - 提高稳定性（低参数漂移）
   - 降低性能方差（跨试验一致性好）

2. **频谱分析可作为架构设计蓝图**：
   - 主导频率数量 → 决定模型深度 $S$
   - 避免“过深”或“不足”的经验性猜测

3. **趋势估计的有限样本理论指导输入窗口选择**：
   - 推导出 trend encoder 的最小输入长度 $n_{\text{opt}}$
   - 实现高达 **89–90% 输入维度压缩** 而不失精度

4. **无需修改模型或训练流程即可实现全面性能提升**：
   - 仅通过初始化和配置调整，即可超越现有方法

5. **该框架具有泛化能力**：
   - 在合成与多种真实场景（电力、热力）中均有效
   - 可扩展至其他生成模型（如 ITF-VAE）

---

### ⚠️ 局限性
1. **依赖信号具备一定周期性**：若数据无明显频谱峰（low $p_{\text{spec}}$），季节性先验作用减弱。
2. **线性趋势假设限制**：使用 OLS 估计趋势，难以捕捉非线性长期变化。
3. **预处理开销增加**：需额外执行 FFT 和回归分析，虽离线完成，但在大规模流式场景可能成为瓶颈。
4. **未考虑多尺度事件交互**：event component 初始化仍较简单，未充分建模复杂瞬态模式。

---

### 🔮 未来工作方向
1. **自动化紧凑架构搜索**：结合先验提取，构建复杂度自适应的轻量化 BoF 架构。
2. **在线/自适应学习**：让模型在部署中持续更新先验，应对概念漂移（concept drift）。
3. **扩展至多变量时间序列**：将频谱分析推广到多元信号（multivariate spectral analysis）。
4. **融合物理知识约束**：将领域知识（如能量守恒）嵌入先验，进一步增强可解释性。

---

## 总结
本文提出了一种 **将经典信号处理（FFT + 回归）与现代神经网络初始化深度融合** 的新范式。通过从数据中提取 **spectral 和 temporal priors**，实现了 **更高效、更稳定、更紧凑** 的 function-parameterizing 模型训练。其核心价值在于：
> **用少量离线分析换取显著的在线性能提升，且完全兼容现有训练流程。**

该方法为 **interpretable machine learning** 和 **resource-efficient time series modeling** 提供了重要实践路径。

</details>

---

### 7. [Form Follows Function: Recursive Stem Model](https://arxiv.org/abs/2603.15641)

**Authors**: Navid Hakimi  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.15641v1  

#### Abstract
Recursive reasoning models such as Hierarchical Reasoning Model (HRM) and Tiny Recursive Model (TRM) show that small, weight-shared networks can solve compute-heavy and NP puzzles by iteratively refining latent states, but their training typically relies on deep supervision and/or long unrolls that ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Form Follows Function: Recursive Stem Model*

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的递归推理模型（如 **HRM** 和 **TRM**）虽然在解决计算密集型、NP 难题（如 Sudoku、Maze）上表现出色，但其训练通常依赖于 **深度监督（deep supervision）** 或 **长展开（long unrolls）**，这带来了以下问题：
- **训练成本高**：长时间反向传播导致内存占用大、训练时间长；
- **优化不稳定**：梯度消失/爆炸；
- **行为偏差**：模型被鼓励在每一步都输出“看起来正确”的中间状态，导致贪婪策略而非稳定收敛。

### 提出的新方法：Recursive Stem Model (RSM)
RSM 是一种新型的递归推理架构，在保留 TRM 轻量级、权重共享主干的基础上，重构了训练范式，核心思想是：  
> **让网络学习一个稳定的、与深度无关的 transition operator，而不是特定深度的轨迹。**

#### 主要创新点：
1. **终端损失 + 完全历史分离（Terminal Loss with Detached Warm-up）**
   - 仅在最后一步计算 loss，不进行 deep supervision；
   - 所有中间步骤的状态通过 `stop_gradient` 完全断开反向传播路径；
   - 将早期迭代视为“预热”阶段，迫使模型学习对任意初始状态都能有效改进的操作符。

2. **独立增长内外层递归深度（Independent Growth of H and L）**
   - 外循环深度 $H$（slow cycle）控制全局细化次数；
   - 内循环深度 $L$（fast cycle）控制每次外循环中的局部计算强度；
   - 在训练中逐步增加 $H$ 和 $L$，实现浅层训练、深层推理（test-time scaling）。

3. **随机深度机制（Stochastic Depth over Outer Transition）**
   - 在训练时以一定概率决定是否将倒数第二步纳入梯度路径；
   - 缓解因突然增加递归深度带来的分布偏移和训练不稳定。

4. **原生可靠性信号：收敛性诊断（Native Reliability Signal via Convergence）**
   - 利用模型输出是否趋于固定点（fixed point）作为“是否已解决”的判断依据；
   - 结合 domain verifier 可构建鲁棒的停止机制和置信度估计。

### 相比现有方法的优势
| 维度 | RSM | TRM / HRM |
|------|-----|-----------|
| **训练效率** | >20× 更快 | 依赖长展开，训练慢 |
| **错误率** | ≈5× 降低 | 易受贪婪中间行为影响 |
| **测试时扩展性** | 支持任意多步推理（如 20,000 步） | 推理深度受限于训练深度 |
| **稳定性** | 引入 stochastic depth 提升训练鲁棒性 | 深度变化易引发 loss spike |
| **可解释性** | 收敛行为提供天然的“思考完成”信号 | 中间状态难以判断有效性 |

---

## 2. 核心实验方法和设置

### 数据集
- **Sudoku-Extreme**：极端难度数独谜题，输入为部分填充网格，目标为完整正确解。
- **Maze-Hard (30×30)**：复杂迷宫路径规划任务，需从起点找到终点并满足颜色约束等规则。

### 实验设置
- **模型规模**：仅使用 **2.5–5M 参数**，极小模型。
- **训练资源**：单张 A100 GPU，总训练时间约 **1 小时（Sudoku）** 和 **40 分钟（Maze）**。
- **训练预算限制**：所有实验总花费约 \$50（Google Colab），强调低成本验证。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Exact Accuracy (%)** | 输出完全正确的样本比例（精确匹配） |
| **Steps-to-Solve** | 达到稳定且正确解所需的 outer cycle 数 |
| **Test-Time Compute Scaling** | 在远超训练深度（如 $H_{\text{test}} \gg H_{\text{train}}$）下的性能表现 |
| **Convergence Behavior** | 是否达到 fixed point，用于可靠性判断 |

### 基线方法对比
- **TRM (Tiny Recursive Model)**：当前 SOTA，使用 deep supervision 和固定 unroll。
- **HRM (Hierarchical Reasoning Model)**：双频率递归结构，也采用 deep supervision。
- **Chain-of-Thought + LLMs**：虽能提升效果，但依赖大规模参数和外部搜索机制。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 任务 | 方法 | Exact Accuracy | 训练时间 | 测试深度 |
|------|------|----------------|----------|----------|
| **Sudoku-Extreme** | RSM | **97.5%** | ~1 小时 | 最大训练 $H=20$，测试可达 $H=20,000$ |
| **Maze-Hard (30×30)** | RSM (Attention-based) | **~80%** | ~40 分钟 | 同样支持超长推理链 |

> 注：TRM 在相同条件下 Sudoku 上约为 87%，耗时 12 小时。

### 与基线方法的对比结果
- **训练速度提升 >20×**，同时错误率下降约 **5 倍**；
- 在 **test-time compute** 下显著优于 TRM（见 Figure 2），即使训练深度仅为 ~20，也能通过延长推理步数持续提分；
- 在 **Sudoku** 上超过 TRM 近 10 个百分点（97.5% vs 87%）；
- **Maze 任务中表现良好**，作者认为性能上限受限于训练集多样性而非模型能力。

### 消融实验结果（Ablation Studies）
尽管论文未系统开展大规模消融（受限于预算），但仍提供了关键证据：
- **Stochastic Depth 的重要性**（Figure 7）：
  - 移除后训练过程出现剧烈 loss spike；
  - 保留时训练更平稳，尤其在 depth increment 阶段。
- **Learning Rate Warmup for Transitions**（Figure 8）：
  - 在 depth transition 后重置或渐进恢复 LR 可避免优化震荡；
  - 对训练稳定性至关重要。
- **Warm-up Steps 的必要性**：
  - 强制 $H \geq 2$ 即使在早期 curriculum，确保至少有一个 warm-up step；
  - 避免模型“一步到位”，促进 operator 泛化能力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **递归可以作为原生的 test-time compute 机制**  
   RSM 成功实现了“训练短、推理深”的范式迁移，证明小模型也能通过更多推理步解决难例。

2. ✅ **终端损失 + 历史分离能引导更稳健的学习动态**  
   不再奖励“每步都像答案”，而是只关心“最终是否正确”，促使模型发展出真正的迭代改进能力。

3. ✅ **收敛行为本身是一种有价值的可靠性信号**  
   - 若输出仍在变化 → “尚未想清楚”；
   - 若输出稳定 + verifier 通过 → 高可信解；
   - 这为防止 hallucination 提供了 architecture-native 的防护机制。

4. ✅ **RSM 具备类生命系统的计算隐喻**
   - $z_L$（快速计算态） ↔ 高熵探索；
   - $z_H$（慢速提交态） ↔ 低熵稳定；
   - 类似生物体中“干细胞增殖 → 分化输出”的过程（Figure 9），体现 **shared DNA, different function via context**。

### 方法的局限性
1. ❗ **收敛 ≠ 正确性**  
   模型可能陷入错误的 fixed point（spurious attractor），因此必须结合 domain verifier 使用。

2. ❗ **缺乏严格的 contraction 或 equilibrium constraint**  
   不同于 DEQ 使用 implicit differentiation 显式求解平衡点，RSM 依赖经验训练，无法保证全局收敛。

3. ❗ **当前验证集中于 verifier-rich 任务**  
   如 Sudoku、Maze 等有明确规则的任务；对于开放生成任务（如故事写作），如何定义“稳定解”仍是挑战。

4. ❗ **超参调优非系统化**  
   由于资源限制，训练 schedule、growth rate、stochastic probability 等选择偏向“艺术性”而非科学扫参。

### 未来工作方向
1. 🔮 **Develop General Principles for Step-Aligned Supervision**  
   如何设计与推理结构对齐的训练目标？例如 blockwise fine-tuning 匹配 blockwise decoding。

2. 🔮 **Explore Optimal Growth Schedules**  
   能否基于梯度行为、激活分布或信息论指标动态决定何时增长 $H/L$？

3. 🔮 **Integrate Explicit Verification Signals into Training**  
   将 verifier 输出作为 reward 或 constraint 引入训练，进一步提升鲁棒性。

4. 🔮 **Extend to Open-Ended Generation Tasks**  
   结合 preference modeling 或 constraint programming，使 RSM 适用于多解空间任务。

5. 🔮 **Bridge with Implicit Layers and DEQs**  
   探索将 RSM 与 Deep Equilibrium Models 结合，在保持轻量的同时引入更强的收敛保障。

---

> 📌 **总结一句话**：  
> **RSM 重新定义了递归模型的训练契约——不是教它“走多少步到终点”，而是教它“每一步该怎么走”，从而解锁了极致高效的训练与无限延展的推理能力。**

</details>

---

### 8. [QV May Be Enough: Toward the Essence of Attention in LLMs](https://arxiv.org/abs/2603.15665)

**Authors**: Zhang Edward  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.15665v1  

#### Abstract
Starting from first principles and a linguistic perspective centered on part-of-speech (POS) and syntactic analysis, this paper explores and derives the underlying essence of the Query-Key-Value (QKV) mechanism within the Transformer architecture. Based on this theoretical foundation, we provide a u...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**QV May Be Enough: Toward the Essence of Attention in LLMs**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文从**第一性原理**和**语言学视角**（特别是词性标注 POS 和句法分析）出发，重新审视 Transformer 中广泛使用的 **QKV 机制**的本质。尽管 QKV 被普遍采用，但其三个组件（Query、Key、Value）在自然语言处理中的**功能角色和内在逻辑尚不明确**。

传统将 QKV 类比为数据库检索系统（查询-键-值）的解释缺乏对语言结构的深入理解。本文旨在回答：
- Q、K、V 在语言层面分别代表什么？
- 是否可以简化甚至重构当前的 Attention 架构？

### 🚀 提出的新方法与新思路
1. **提出 QV 范式（QV Paradigm）**  
   基于“浅层组合”（Shallow-Composing, SC）与“深层匹配”（Deep-Matching, DM）的理论框架，作者论证了 **Value 可以同时承担 Key 的角色**，从而将标准的 `QKV` 结构简化为 `QV` 模式：
   $$
   \text{Attention}(Q, V) = \text{softmax}\left(\frac{QV^T}{\sqrt{d_k}}\right)V
   $$
   这种设计减少了参数量和计算开销。

2. **引入 QV-Ka（Key-after-value & ctx）优化架构**  
   进一步提出一种新型结构：**Key 不再独立生成，而是基于 Value 和上下文（context）动态推导出来**：
   $$
   K = \text{DM}(V, \text{Context})
   $$
   具体实现中，通过一个轻量级网络利用 $V$ 和额外的 context 向量来合成 $K$，显著降低 Key 的冗余表达。

3. **统一解释主流高效 Attention 架构（MQA/GQA/MLA）**
   本文指出，近年来流行的 KV-Shared 架构如 **MQA、GQA、MLA** 本质上是向 **QV 范式的演进**，而非严格遵循原始 QKV 逻辑。尤其是 MLA，被视为一种高度压缩但灵活的 QV 变体。

### 🔍 相比现有方法的优势
| 方法 | 优势 |
|------|------|
| **QV 范式** | 减少模型复杂度；提升可解释性；保留大部分性能 |
| **QV-Ka** | 在接近 QKV 性能的同时大幅减少参数；训练初期表现更优；符合 Attention 内在逻辑 |
| **理论框架** | 首次从语言学角度系统解释 QKV 功能分工，为后续架构设计提供指导 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **WMT_17 英德翻译任务（en-de）**  
  用于验证不同 Attention 模式的有效性。

### ⚙️ 实验设置
| 配置项 | 设置 |
|--------|------|
| 基础架构 | Vanilla Transformer-BIG |
| 层数（LayerNum） | 3（为加速训练，低于默认 6 层） |
| 精度 | FP16 半精度训练 |
| 硬件平台 | 单张 NVIDIA Tesla V100-PCIE-32GB |
| 训练时长 | ~15 小时每轮 |
| 开源代码 | OpenNMT-py + 自研 AGF 模块（GitHub 可用） |

### 🎯 评估指标
- **Validation Accuracy (%)**：作为主要性能衡量标准
- 对比模式包括：
  - 标准 QKV（带 Sinusoidal PE）
  - QV 模式（原始与 AGF 改进版）
  - 引入 AGF（Attention’s Gravitational Field）相对位置编码
  - 应用 PCM-V 优化技术
  - 新提出的 QV-Ka 模式（不同 context 维度）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（见 Table 3 & 4）

| Mode | Configuration | Valid Accuracy (%) |
|------|---------------|--------------------|
| QV | Default (Sinusoidal PE) | 70.0756 |
| QKV | Default (Sinusoidal PE) | 70.5911 |
| QV | AGF + PCM-V | 70.5188 |
| QKV | AGF + PCM-V | 70.7305 |
| **QV-Ka** ($d_{ctx}=d_{head}$) | AGF + PCM-V | 70.4998 |
| **QV-Ka** ($d_{ctx}=2d_{head}$) | AGF + PCM-V | **70.6919** |

### 🔁 与基线方法对比
- **原始 QV vs QKV**：存在约 **0.5% 的准确率差距**。
- 引入 **AGF 相对位置编码后**，QV 与 QKV 差距缩小至 **仅 0.21%**，说明原差距中近一半来自 Positional Encoding 干扰。
- **QV-Ka (2×d_head)** 表现尤为突出，**几乎完全追平甚至略微超越标准 QKV**，且参数更少、结构更合理。

### 🔍 消融实验发现
1. **Positional Encoding 是性能损失主因之一**  
   在 QV 模式下，由于 V 承担了 K 的角色，也需参与位置计算，导致语义与位置信息耦合过强，影响效果。使用 AGF 解耦后性能大幅提升。

2. **DODM（Deep-Matching Diffusion）效应差异**
   - QKV 中，Deep-Matching 的扩散发生在 **Value 端**，有助于强化相关特征；
   - QV 中，扩散发生在 **Query 端**，造成“意图发散”，削弱注意力集中能力。

3. **QV-Ka 初期收敛更快**  
   图7显示，在训练早期阶段，QV-Ka 的性能上升速度优于标准 QKV，表明其更贴近 Attention 的本质逻辑。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **QV 范式足以逼近 QKV 性能**  
   在合理的相对位置建模（如 AGF）支持下，**省略独立 Key 的 QV 模式能达到与 QKV 几乎相同的性能水平**，挑战了“必须三者分离”的传统认知。

2. **现代高效 Attention 架构本质趋同于 QV**  
   **MQA、GQA、MLA 等 KV-Shared 方法并非真正延续 QKV 逻辑，而是逐步向 QV 范式靠拢**。其中 MLA 是一种软性的、高度优化的 QV 实现。

3. **QV-Ka 是更优的工程实现路径**  
   通过让 Key 由 Value 和 Context 推导而来，既保持了 Key 的灵活性，又避免了完全独立生成带来的冗余，实现了**效率与性能的平衡**。

4. **Attention 的本质是 Deep-Matching + Shallow-Composing**  
   该论文构建了一个可解释的理论框架，揭示 Attention 的核心在于：
   - **Deep-Matching**：建立修饰关系（如形容词→名词）
   - **Shallow-Composing**：组合基础语义特征

---

### ⚠️ 方法的局限性
- **实验规模较小**：仅使用 3 层 Transformer 和中等规模数据集（WMT_17），未在大规模 LLM 上验证泛化能力。
- **未测试长序列外推能力**：虽然 AGF 支持长度外推，但文中未进行 explicit 测试。
- **硬件限制影响结论普适性**：所有实验运行于单卡 V100，难以反映分布式训练下的真实表现。

---

### 🔮 未来工作方向
1. **推广至大模型场景**  
   在百亿/千亿参数级别 LLM 上验证 QV 与 QV-Ka 的有效性与扩展性。

2. **探索 V-Shared 架构替代 KV-Shared**  
   提议从 KV 共享转向 **V-Shared + 独立 K** 架构，即 $(Q_k, V_k)V_o$，兼顾内存效率与表示能力。

3. **优化 MLA 的压缩比率**  
   当前 MLA 对 K/V 压缩过于激进，建议适度降低 K 的压缩率以保留更多全局语义特征。

4. **完全移除 Key 向量的可能性研究**  
   若结合 AGF、T5 或 ALiBi 等相对位置方法，**K 向量可能被彻底弃用**，实现真正的 **GQA/MLA-QvVv** 架构。

---

> 💡 **总结一句话**：  
> 本论文从语言学第一性原理出发，提出 **QV 范式足以捕捉 Attention 本质**，并通过 QV-Ka 实现高效且高性能的 Attention 架构演化，为未来 LLM 设计提供了全新的理论视角与优化路径。

</details>

---

### 9. [Via Negativa for AI Alignment: Why Negative Constraints Are Structurally Superior to Positive Preferences](https://arxiv.org/abs/2603.16417)

**Authors**: Quan Cheng  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.16417v1  

#### Abstract
Recent empirical results have demonstrated that training large language models (LLMs) with negative-only feedback can match or exceed standard reinforcement learning from human feedback (RLHF). Negative Sample Reinforcement achieves parity with PPO on mathematical reasoning; Distributional Disprefer...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Via Negativa for AI Alignment: Why Negative Constraints Are Structurally Superior to Positive Preferences*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文旨在解释一个在大语言模型（LLM）对齐研究中反复出现但缺乏统一理论解释的现象：**仅使用负反馈信号的方法（如惩罚错误而非奖励正确）在多个任务上能够匹配甚至超越传统的基于正负偏好学习的RLHF方法**。同时，传统RLHF方法被观察到会系统性地放大 *sycophancy*（迎合用户倾向），即模型倾向于附和用户的观点而非提供真实正确的答案。

这一现象背后缺乏根本性的理论框架来统一解释“为何负信号如此有效”以及“为何正偏好训练容易失败”。

### 🚀 提出的新方法/新思路
论文提出了一种**结构性不对称理论（Structural Asymmetry Theory）**，将AI对齐问题从认识论层面重新建模：

- **Positive Preferences（正向偏好）** 是连续耦合的、情境依赖的、无限维的，无法通过有限样本完全指定。
- **Negative Constraints（负向约束）** 是离散的、可枚举的、独立可验证的、收敛的，因此更适合用于构建稳定边界。

由此引出核心思想：  
> **对齐应从“学习人类喜欢什么”转向“学习人类拒绝什么”——即采用 via negativa 范式**。

这不是一个新的训练算法，而是一个**统一的理论框架**，用以解释一系列已有经验成果，并为未来方法设计提供指导原则。

### 🔍 相比现有方法的优势
| 维度 | 传统RLHF（Positive Preference-Based） | 本文倡导的 Negative Constraint-Based 方法 |
|------|----------------------------------------|------------------------------------------|
| 信号结构 | 连续、耦合、高维偏好需降维至二元比较 | 离散、独立、可验证的禁止项 |
| 可收敛性 | 不具备单调收敛性，偏好可能冲突 | 随着约束增加，可行空间单调收缩 |
| 抗干扰能力 | 易受表面相关性污染（如sycophancy） | 更鲁棒，因负面规则更易明确界定 |
| 数据效率 | 需大量成对偏好标注 | 可利用非成对、单一样本负反馈 |

---

## 2. 核心实验方法和设置

> ⚠️ 注意：这是一篇**立场论文（position paper）**，作者未进行新的实验，而是基于已有文献中的实证结果进行理论整合与解释。

### 📚 引用的关键方法与数据集
论文综合分析了以下几项已发表工作的实验设置：

| 方法 | 来源 | 数据集 | 主要任务 |
|------|------|--------|---------|
| **NSR (Negative Sample Reinforcement)** | [1] Liu et al., 2025 | MATH, AIME | 数学推理生成 |
| **D2O (Distributional Dispreference Optimization)** | [2] Duan et al., 2024 | 自定义负样本集合 | 对齐训练，仅用被拒响应 |
| **NPO (Negative Preference Optimization)** | [3] Zhang et al., 2024 | Forget datasets | 非学习（unlearning）任务 |
| **KTO (Kahneman-Tversky Optimization)** | [4] Ethayarajh et al., 2024 | 多样化文本交互数据 | 单样本二元标签对齐 |
| **Constitutional AI** | [12] Bai et al., 2022 | 自建宪法原则+自评机制 | 安全性、无害性对齐 |

### 🎯 评估指标（来自引用研究）
- **数学推理性能**：MATH/AIME 准确率（accuracy）
- **安全性**：有害内容生成率、是否违反宪法条款（Yes/No 判断）
- **sycophancy rate**：模型附和用户明显错误陈述的比例
- **响应长度 & 信息密度**：token数、每token包含的独特实质性主张数量
- **unlearning效果**：遗忘目标行为的成功率而不引发灾难性遗忘

### 🔁 基线方法对比
主要对比对象包括：
- **PPO / GRPO**：标准强化学习对齐方法
- **DPO (Direct Preference Optimization)**：当前主流的免强化学习偏好优化
- **标准RLHF**：基于人类偏好的奖励建模 + 强化学习
- **纯预训练模型**：作为起点基准

---

## 3. 主要实验结果和性能指标

以下是论文引用的关键实证结果，支持其理论主张：

### ✅ 性能表现汇总

| 方法 | 关键结果 | 对比优势 |
|------|--------|----------|
| **NSR** | 在MATH/AIME上达到与PPO/GPRO相当的准确率 | 仅惩罚错误路径，无需奖励正确路径即可提升性能 |
| **D2O** | 仅使用dispreferred samples完成有效对齐 | 无需高质量正样本，降低标注成本 |
| **NPO** | 成功实现effective unlearning，避免catastrophic collapse | 表明否定性监督可用于精确删除特定行为 |
| **KTO** | 使用unpaired binary labels，在远少于DPO的数据量下达到同等性能 | 验证loss-averse机制的有效性 |
| **Constitutional AI** | 在harmlessness benchmarks上优于纯RLHF；Claude系列表现出更低sycophancy | 支持“负向原则”更能保障安全性和客观性 |

### 🔍 消融实验（间接引用）
虽然本文无直接消融实验，但它解释了其他工作中的一些关键发现：
- Sharma et al. [5] 发现：人类标注者本身就会偏好*sycophantic responses*，说明正向偏好信号存在内在噪声。
- Shapira et al. [6] 形式化证明：RLHF会放大“认同用户信念”与“获得高评分”之间的协方差，导致sycophancy被强化。
- Yao et al. [13] 发现：仅用2%计算预算的unlearning流程即可达到RLHF级别的安全性 → 表明“排除错误”比“塑造最优”更高效。

---

## 4. 关键结论和发现

### 🌟 主要发现
1. **结构性不对称是根本原因**：
   - 正向偏好本质上是不可穷尽的（inexhaustible）、上下文敏感的连续函数，难以通过有限数据准确建模。
   - 负向约束则是离散、可枚举、可独立验证的，允许系统逐步逼近一个稳定的“安全边界”。

2. **sycophancy 是正偏好训练的结构性缺陷，而非偶然偏差**：
   - 因为真实偏好函数过于复杂，标注过程必然丢失维度，而“agree with user”成为一个低维、强相关的代理特征（proxy feature），被RLHF错误学习并放大。

3. **via negativa 具有收敛保证**：
   - 每增加一条负约束，响应空间只会缩小不会扩大，最终剩余区域内的任意输出都近似可接受。
   - 类比专家成长路径（Dreyfus模型）：不是知道最佳动作，而是避开了所有坏动作。

4. **capability增长可能是negative knowledge积累的结果**：
   - 更强模型（如Claude Opus vs Sonnet）往往输出更短、信息密度更高、更少客套话 —— 这符合“学会了不说什么是错的”的假设。

### ⚠️ 方法的局限性
- **不能完全替代正向学习**：某些属性如 helpfulness、creativity、tone 等本质上是正面且连续的，难以仅靠否定来定义。
- **负约束的完整性挑战**：尽管理论上可枚举，实践中仍可能存在遗漏的重要禁忌（unknown unknowns）。
- **动态社会规范适应难**：静态的负规则难以应对价值随时间演化的场景。

### 🔮 未来工作方向
1. **开发专门收集“rejected responses”或“what’s wrong?”标注的新数据协议**
2. **构建标准化 benchmark 测试 negative knowledge 积累程度**
   - 如测量不同模型在相同query下的 response length、information density、sycophancy rate
3. **分离对齐任务**：
   - 安全性、事实性 → 由 negative constraints 处理
   - 帮助性、风格等 → 保留给 positive preference 学习（但应解耦训练）
4. **探索自动发现 violation 的机制**（如 self-critique + verifier chains）

---

> 💬 **结语金句（原文引用）**  
> _“The chess grandmaster wins by not losing. The aligned model aligns by learning what not to do.”_

</details>

---

### 10. [Looking for (Genomic) Needles in a Haystack: Sparsity-Driven Search for Identifying Correlated Genetic Mutations in Cancer](https://arxiv.org/abs/2603.16721)

**Authors**: Ritvik Prabhu, Emil Vatai, Bernard Moussad, Emmanuel Jeannot, Ramu Anandakrishnan, Wu-chun Feng, Mohamed Wahib  
**Category**: cs.DC  
**Published**: 2026-03-18  
**Score**: 6.0  
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
癌症通常由多个基因突变（即“多击”组合，multi-hit combinations）协同驱动，而非单一突变。然而，随着“击中”数 $ h $ 的增加，候选基因组合的数量呈组合爆炸增长（例如，$ \binom{20,000}{4} \sim 6.4 \times 10^{15} $），使得传统的**Weighted Set Cover (WSC)** 方法在计算上不可行，难以扩展到四击及以上。

现有方法面临两难困境：
- **穷举类方法**（如 WSC）准确但计算昂贵，仅适用于低阶组合（$ h \leq 3 $）；
- **图模型方法**（如 BiGPICC）速度快但牺牲了精度，可能遗漏关键的稀有组合。

### 提出的新方法与创新思路
本文提出了一种名为 **Pruned Depth-First Search (P-DFS)** 的算法框架，其核心思想是利用肿瘤突变数据的**高稀疏性**（sparsity）来大幅剪枝搜索空间。

#### 主要创新点：
- **P-DFS 剪枝机制**：  
  在深度优先搜索过程中，维护一个当前部分组合的位向量交集（bitwise AND）。一旦该交集为空（即没有样本同时携带这些突变），立即回溯并剪除整个子树。这避免了对无效路径的进一步探索。
  
- **稀疏性感知预处理**：  
  将基因按突变频率从稀疏到密集排序，使空交集更早出现，从而提升剪枝效率。

- **高效并行化设计**：
  - 将前两层嵌套循环展平为单个索引 $ \lambda $，生成大量独立任务，支持超大规模并行（可达数十万 MPI ranks）。
  - 采用**分层 MPI 架构**：每个节点设一个 leader 负责本地任务分配，leader 之间通过 work stealing 实现跨节点负载均衡。
  - 使用**无屏障终止检测**（barrier-free termination detection）和**分层集体通信**（hierarchical collectives）优化可扩展性。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **准确性** | 保留了 WSC 的精确评分机制，优于图模型方法（如 BiGPICC） |
| **效率** | 利用稀疏性剪枝，显著减少需评估的组合数量（最高达 98%） |
| **可扩展性** | 支持在超算（如 Fugaku）上运行至 147,456 个 MPI ranks，实现百倍加速 |
| **实用性** | 可处理 $ h=4 $ 甚至更高阶组合，突破传统方法瓶颈 |

---

## 2. 核心实验方法和设置

### 数据集
- 来源：**The Cancer Genome Atlas (TCGA)** 的体细胞突变数据（MAF 格式）
- 预处理：
  - 排除沉默突变（silent mutations）
  - 分离肿瘤样本（tumor samples）与正常对照样本（normal samples）
  - 构建二值突变矩阵 $ X \in \{0,1\}^{G \times n} $，其中 $ G \approx 20,000 $ 为基因数，$ n $ 为样本数
- 使用的癌种：BLCA（膀胱癌）、HNSC（头颈癌）、BRCA（乳腺癌）等共 10+ 种，依据 [Anandakrishnan et al., 2019] 估计所需 hit 数

> ⚠️ 注：所有数据集平均稀疏度高达 **95.61%**（见 Fig. 3），验证了剪枝假设的有效性。

### 实验设置
- 平台：日本 **Fugaku 超级计算机**
  - 每节点：1× Fujitsu A64FX CPU（48 核），32GB HBM2 内存
  - 互连：Tofu-D 六维环面拓扑
  - 运行模式：1 MPI rank / core，共测试 192 至 3,072 节点（即 9,216 至 147,456 ranks）
- 编程模型：纯 MPI（rank-per-core），无 OpenMP 混合并行
- 实现语言：C++17 + MPI + Bitwise Operations

### 评估指标
| 指标 | 定义 |
|------|------|
| **Visited Combinations** | 实际访问的 $ h $-hit 组合数量（衡量剪枝效果） |
| **Wall Time** | 端到端运行时间（衡量性能） |
| **Speedup** | 相对于基线方法或理想强扩展性的加速比 |
| **Sensitivity** | 测试集中被覆盖的肿瘤样本比例 |
| **Specificity** | 测试集中未被错误覆盖的正常样本比例 |
| **Solution Size** | 所需最小组合集合的大小（越小越好） |

### 基线方法对比
- **Baseline 1**: Exhaustive WSC（穷举 WSC），不启用剪枝（`BOUND=OFF`）
- **Baseline 2**: 图模型方法 BiGPICC（作为快速但低精度参考）
- **本方法**: P-DFS + WSC（`BOUND=ON`）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 剪枝效率极高（Fig. 6）
- 对于 **4-hit** 组合，在 BLCA 数据集上仅访问了约 **0.8%** 的理论最大组合数。
- 随着 $ h $ 增加，剪枝效果更明显（因稀疏性导致早期交集更快归零）。

| 癌症类型 | Hits | 总组合数（理论） | 实际访问数 | 剪枝率 |
|---------|------|------------------|------------|--------|
| BLCA    | 4    | $ 6.43 \times 10^{15} $ | $ 6.42 \times 10^{14} $ | ~90% |
| HNSC    | 4    | 同上             | 更少       | >95%?（文中未明示） |

> 🔹 图表显示：**橙色柱越短，剪枝越多**；且 $ h $ 越大，剪枝越多。

#### ✅ 显著的端到端加速（Fig. 7 & 10）
- 在 **3,072 节点（147,456 ranks）** 上：
  - P-DFS 方法完成一次迭代仅需约 **20 分钟**
  - 穷举 WSC 方法预计耗时 **超过 52 小时**
  - **实际加速比达 ~183×**

- 强扩展性接近理想曲线（尤其在 $ h=4 $ 时），表明计算主导（compute-bound）

#### ✅ 更优的解质量（Table II）
- 在多个癌症类型中，P-DFS 得到的 **solution size 更小**，尤其是在 $ h=4 $ 时：
  - 如 BLCA 中，原 WSC 解大小为 19，而 P-DFS 仅为 **16**
  - 平均减小 **~80%**（原文称“reduces the solution size at four hits by 80% average”）

> 💡 原因分析：稀疏基因优先探索 → 更高效的边际增益 → 更快收敛

#### ✅ 出色的泛化能力（Table III & IV）
- **训练集表现**：Sensitivity = 1.0，Specificity > 0.96（除个别外）
- **测试集表现**（held-out 25%）：
  - Sensitivity：0.85–0.98
  - Specificity：0.81–0.99
- 表明所识别的 multi-hit 模式具有生物学意义且可推广

#### ✅ 负载均衡有效（Fig. 8）
- 启用 work stealing 后：
  - worker 运行时间标准差从 **544s → 42s**（下降 13×）
  - 平均空闲率从 **63.6% → 22.3%**
- 显示动态调度显著缓解了任务不均衡问题

---

## 4. 关键结论和发现

### 主要发现
1. **稀疏性是突破口**：肿瘤突变数据的高度稀疏性（median 95.61%）为剪枝提供了天然条件，P-DFS 成功将其转化为计算优势。
2. **P-DFS 显著压缩搜索空间**：通过 early termination 和 sparse-first ordering，将原本指数级增长的搜索空间压缩至可管理范围。
3. **方法兼具精度与效率**：既保持了 WSC 的高判别力，又实现了前所未有的可扩展性，首次实现在超算上系统性地搜索 **4-hit 及以上组合**。
4. **高阶组合更有价值**：随着 $ h $ 增加，剪枝效率反而提高，且得到的组合更具特异性，有助于发现新的致癌通路。

### 方法的局限性
- **依赖稀疏性假设**：若某些癌症类型突变密集（如超突变型肿瘤），剪枝效果可能下降。
- **仍属启发式搜索**：虽然基于 WSC，但 P-DFS 是一种贪心剪枝策略，并不能保证全局最优解。
- **内存复制开销**：每 MPI rank 存储完整数据副本，限制了最大可处理的数据规模（尽管当前数据仅几 MB，尚可接受）。
- **编译期参数绑定**：`NUMHITS` 和 `BOUND` 需在编译时设定，灵活性略低。

### 未来工作方向
1. **扩展至更高阶组合**（$ h=5,6,\dots $）：已有基础，有望揭示更复杂的协同突变网络。
2. **结合功能注释信息**：引入通路、蛋白互作网络等先验知识，进一步指导剪枝或评分函数设计。
3. **支持动态稀疏排序**：在搜索过程中根据剩余样本动态调整基因顺序，可能进一步提升效率。
4. **集成到临床辅助诊断流程**：推动该方法用于个性化精准医疗中的组合靶向治疗推荐。

---

## 附录：代码可用性
- GitHub 仓库：[https://github.com/RitvikPrabhu/P-DFS-Multihit-WSC](https://github.com/RitvikPrabhu/P-DFS-Multihit-WSC)
- 包含完整实现、脚本、日志及复现指南，支持在普通 HPC 集群部署。

</details>

---

### 11. [NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics](https://arxiv.org/abs/2603.16148)

**Authors**: Zhengzheng Tang  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.16148v1  

#### Abstract
We ask whether a pure spiking backbone can learn large-scale language modeling from random initialization, without Transformer distillation. We introduce NeuronSpark, a 0.9B-parameter SNN language model trained with next-token prediction and surrogate gradients. The model combines selective state-sp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前主流的 **Large Language Models (LLMs)** 主要基于 **Transformer** 架构，虽然在自然语言处理任务中表现出色，但其计算成本高、能耗大，且缺乏生物可解释性。而 **Spiking Neural Networks (SNNs)** 被认为是更节能、更具生物合理性的“第三代”神经网络，但在语言建模领域仍处于早期阶段。

现有 SNN 语言模型存在三大瓶颈：
- **依赖蒸馏**（如 SpkBERT）：从预训练的 Transformer 中提取知识，无法验证纯 SNN 是否能从零学习语言；
- **部分非脉冲化**（如 SpkGPT）：仅隐藏层使用脉冲机制，输入/输出等关键模块仍为传统浮点运算；
- **规模受限**：参数量普遍低于 216M，远未达到现代 LLM 规模。

本文旨在回答一个核心问题：  
> **能否构建一个完全基于脉冲神经网络（pure SNN）、从随机初始化开始训练、在标准 next-token 预测任务下实现大规模语言建模的模型？**

---

### **提出了什么新方法或新思路**

作者提出 **NEURONSPARK-0.9B**，一个拥有 **8.74亿参数** 的纯 SNN 语言模型，并引入多项关键技术：

#### ✅ **1. Selective State Space SNN Block（选择性状态空间 SNN 模块）**
- 将 **Parametric Leaky Integrate-and-Fire (PLIF)** 神经元的动力学形式化为一种 **Selective State Space Model (SSM)**，类比 Mamba 模型中的选择性递归机制。
- 输入相关的参数 $ \beta(t), \alpha(t), V_{th}(t) $ 构成动态门控机制，实现对时序信息的选择性记忆与更新。

#### ✅ **2. 泄漏电流激活（Leakage-Current Activation）作为层间通信信号**
- 放弃传统的二值脉冲 $ s[t] \in \{0,1\} $ 传递方式（表达能力弱、梯度传播困难）；
- 使用 **泄漏电流信号** $ \text{leak}[t] = (1-\beta) \cdot V_{\text{post}}[t] $ 作为默认的层间浮点信号，保留时间动态特性并增强表达力。

#### ✅ **3. PonderNet 自适应时间步长**
- 在每个子层内部应用 **PonderNet**，允许不同 token 动态调整所需的 SNN 时间帧数 $ K $；
- 引入几何分布加权聚合与 **ponder cost 正则项**（$ \lambda=0.01 $），防止无限循环。

#### ✅ **4. Triton 融合 PLIF 内核**
- 开发了高效的 **Triton-fused kernels**，将整个 PLIF 前向/反向传播（含 surrogate gradient）融合在一个 CUDA kernel 中执行；
- 包括 per-element 和 row-parameter 两种变体，显著提升训练效率。

#### ✅ **5. SNN 原生稳定技术**
- **残差居中（Residual Centering）**：每层残差连接前减去均值，防止直流漂移；
- **侧抑制归一化（Lateral Inhibition Normalization）**：等价于 RMSNorm，但具有生物学解释；
- **自然梯度补偿（Natural Gradient Compensation）**：两阶段优化调制参数梯度，缓解饱和与跨层不均衡问题。

---

### **相比现有方法的优势**

| 维度 | NEURONSPARK | 先前工作（如 SpkGPT / SpkBERT） |
|------|-------------|-------------------------------|
| 初始化方式 | 从零训练（random init） | 依赖 Transformer 蒸馏 |
| 架构纯度 | 完全 SNN（core stack 全脉冲） | 部分非脉冲组件（embedding/output） |
| 模型规模 | 0.9B 参数 | ≤216M 参数 |
| 生物可解释性 | 高（多尺度神经元、结构驱动计算） | 较低 |
| 可扩展性 | 已验证至 0.9B 规模可行性 | 缺乏端到端可扩展方案 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

| 阶段 | 数据集 | 规模 | 实际使用量 |
|------|--------|------|-----------|
| **预训练（Pretrain）** | Seq-Monkey (Mobvoi, 2023) | ~10B tokens | ~1.4B tokens（约 14%） |
| **监督微调（SFT）** | BelleGroup train_3.5M_CN | ~3.5M 对话样本 | ~42K 样本（约 1.2%） |

> ⚠️ 受限于算力（8×RTX 4090），仅使用小比例子集进行训练。

---

### **实验设置**

- **模型配置**：
  - 参数总量：**874M**（≈0.9B）
  - 层数 $ L = 20 $
  - 隐藏维度 $ D = 896 $
  - 扩展因子 $ N = 8 $
  - 最大 SNN 帧数 $ K = 16 $
  - 上下文长度：512
  - 词表大小：6144（BPE）

- **训练细节**：
  - **预训练**：Adam 优化器，峰值学习率 $ 2\times10^{-4} $，warmup 1000 步，cosine 衰减，bf16 精度；
  - **SFT**：AdamW，学习率 $ 5\times10^{-5} $，weight decay=0.01；
  - 梯度累积 batch size=64，启用 gradient checkpointing；
  - Neuron 参数使用 **10倍基础学习率**。

- **评估指标**：
  - 主要指标：**Training Loss**（预训练 & SFT）
  - 定性分析：生成质量、对话连贯性、推理能力测试（人工评估）
  - 可解释性分析：E[K] 分布、surprisal 相关性、POS 分析、神经元 β 分布等

---

### **基线方法对比**

| 模型 | 参数量 | 是否蒸馏 | 是否全 SNN | 是否支持生成/对话 |
|------|--------|----------|------------|------------------|
| SpkBERT-110M | 110M | 是 | 否 | 否 |
| SpkGPT | 216M | 是 | 否 | 是（部分） |
| **NEURONSPARK-0.9B** | **874M** | **否** | **是** | **是（初步）** |

> 注：由于无公开 Transformer baseline 或标准 benchmark 测试（如 C-Eval），定量对比有限。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 指标 | 数值 |
|------|------|
| **预训练损失（Pretrain Loss）** | **3.6**（从初始 9.0 下降） |
| **SFT 后损失** | **2.1** |
| **训练 Token 总量** | ~1.4B（预训练）+ ~0.4B（SFT） |
| **硬件资源** | 8×RTX 4090，吞吐 ~960 tokens/sec |
| **发布模型** | 已开源至 HuggingFace / ModelScope |

> 📈 图2显示预训练损失稳步下降，表明模型具备有效学习能力。

---

### **与基线方法的对比结果**

尽管没有直接在标准 benchmark 上比较，但从以下方面体现优势：

- **首次实现从零训练的纯 SNN 大模型语言建模**；
- **无需蒸馏即可获得基本语言生成能力**；
- **在极低数据利用率下（14%预训练 + 1.2%SFT）即展现出多轮对话行为**；
- **参数规模达当前 SNN LM 的最先进水平（~4× previous works）**。

---

### **消融实验结果**

通过多个架构变体训练（1K–12K steps），验证各组件必要性：

| 变体 | 训练步数 | 最终 Loss | 说明 |
|------|---------|----------|------|
| **Final V1（完整架构）** | 85K | **3.5** | 成功收敛 |
| MPD-AGL + no Phase 2 | 4.8K | 7.21 | 移除自然梯度补偿后训练停滞 |
| E[K] floor | 1.2K | 7.47 | 加最小步长限制反而恶化 |
| Bounded α | 5.1K | 7.47 | 增益限制导致不稳定 |
| No gradient sync | 0.6K | NaN | 梯度不同步直接崩溃 |

> 🔍 所有消融版本均未能突破 **Loss=7.0**，证明所提稳定技术和架构设计至关重要。

---

## **4. 关键结论和发现**

### **主要发现**

1. ✅ **纯 SNN 架构可以端到端地学习语言建模任务**  
   即使在有限数据和算力条件下，NEURONSPARK-0.9B 也能从随机初始化学会生成语法正确、语义合理的中文句子，并表现出初步的多轮对话能力。

2. ✅ **SNN 动力学天然支持结构感知的自适应计算**  
   - **PonderNet 分配的 E[K]** 明显受 **词性（POS）** 影响：  
     - 标点符号：~5.7  
     - 功能词：~7.4  
     - 名词/动词/形容词：~8.0–8.2  
   - 但与 **预测难度（surprisal）无关**（排除 BOS 后相关系数 r=-0.12），说明计算分配由**句法角色驱动而非统计不确定性**。

3. ✅ **深层结构呈现层级化计算深度**
   - **SNNBlock**（类 Attention）的 E[K] 随层数增加而单调上升（~4 → ~12.7），反映高层需要更长时间整合上下文；
   - **SNNFFN** 则保持稳定（~7–8），符合其局部变换特性。

4. ✅ **神经元自发形成多时间尺度分工**
   - 67.3% 神经元为快响应型（$ \beta < 0.9 $）；
   - 32.7% 为慢记忆型（$ \beta \geq 0.9 $）；
   - 类似生物皮层中 **fast-spiking interneurons** 与 **pyramidal cells** 的共存现象。

5. ✅ **“结构先于语义”的学习路径**
   - 模型掌握了流畅的语言结构（6/6 coherence 测试通过）；
   - 但在算术（0/8）、常识（2/8）、逻辑推理（5/6 正确但多为关键词匹配）上表现极差；
   - 表明当前阶段仅习得“结构性主干”，尚未发展出真正的语义理解与推理能力。

---

### **局限性**

1. **模型规模与上下文仍较小**：0.9B 参数、512 长度，远小于主流 LLM（如 Llama3-8B/70B）；
2. **缺乏量化评测基准**：未报告 C-Eval、CMMLU 等权威中文理解成绩；
3. **无 Transformer 对照实验**：无法判断是否真正具备数据效率优势；
4. **语言单一**：目前仅支持中文；
5. **重复生成与幻觉问题**：存在输出冗余、缺乏深度推理的现象；
6. **可解释性分析为相关性**：尚未建立因果机制。

---

### **未来工作方向**

1. **扩大训练规模**：更多数据、更大模型、更长上下文；
2. **跨语言扩展**：支持英文及其他语言；
3. **部署至类脑芯片**：利用 SNN 的稀疏性在 Loihi、Speck 等 neuromorphic hardware 上验证能效优势；
4. **引入更强监督信号**：结合思维链（Chain-of-Thought）、强化学习等促进语义与推理能力发展；
5. **构建 SNN 特有的预训练范式**：探索更适合脉冲动力学的训练目标；
6. **建立 SNN-LM 专用评测体系**：涵盖 fluency、reasoning、efficiency、biological fidelity 等维度。

---

## **总结**

> **NEURONSPARK 首次证明：一个完全基于脉冲神经网络的架构，可以在没有蒸馏的情况下，从零开始学习大规模语言建模任务。**

它不仅是一个工程上的突破，更揭示了 SNN 在语言处理中可能具有的独特优势——**以生物可解释的方式实现结构敏感的自适应计算**。这为未来开发高效、透明、类脑的语言智能系统提供了重要路径。

🔗 **代码与模型已开源**：  
- GitHub: [https://github.com/Brain2nd/NeuronSpark-V1](https://github.com/Brain2nd/NeuronSpark-V1)  
- HuggingFace: [https://huggingface.co/Brain2nd/NeuronSpark-0.9B](https://huggingface.co/Brain2nd/NeuronSpark-0.9B)  
- ModelScope: [https://www.modelscope.ai/models/Brain2nd/NeuronSpark-0.9B](https://www.modelscope.ai/models/Brain2nd/NeuronSpark-0.9B)

</details>

---

### 12. [Proactive Rejection and Grounded Execution: A Dual-Stage Intent Analysis Paradigm for Safe and Efficient AIoT Smart Homes](https://arxiv.org/abs/2603.16207)

**Authors**: Xinxin Jin, Zhengwei Ni, Zhengguo Sheng, Victor C. M. Leung  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.16207v1  

#### Abstract
As Large Language Models (LLMs) transition from information providers to embodied agents in the Internet of Things (IoT), they face significant challenges regarding reliability and interaction efficiency. Direct execution of LLM-generated commands often leads to entity hallucinations (e.g., trying t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对 **Large Language Models (LLMs)** 在 **AIoT Smart Homes** 场景中作为具身智能体（Embodied Agents）时面临的两大核心挑战：

- **可靠性差距 (Reliability Gap)**：LLM 容易产生 **实体幻觉 (Entity Hallucinations)**，即生成对不存在设备的操作指令（如控制一个家中没有的加湿器），导致执行错误甚至物理安全隐患。
- **交互频率困境 (Interaction Frequency Dilemma)**：现有迭代框架（如 SAGE）在“盲目执行”和“频繁提问”之间摇摆，要么过度打扰用户，要么因缺乏环境感知而错误执行。

### 提出的新方法与思路
作者提出了一种名为 **Dual-Stage Intent-Aware (DS-IA)** 的双阶段意图分析框架，其核心是 **“先分析，后行动” (Analyze-then-Act)** 的主动范式：

- **Stage 1: Global Intent Analysis（全局意图分析）**
  - 作为“语义防火墙 (semantic firewall)”，基于当前家庭环境状态 $ S_t $ 对用户指令进行预判。
  - 将指令分类为三类：
    - `Cvalid`：所有实体均存在且可执行。
    - `Cinvalid`：明确指向不存在的设备，触发 **早期拒绝 (Early Rejection)**。
    - `Cmixed`：混合有效与无效子任务，进入下一阶段处理。
  - 实现 **主动拒绝 (Proactive Rejection)**，避免无效指令进入生成循环。

- **Stage 2: Hierarchical Grounding Verification（分层接地验证）**
  - 引入 **级联验证器 (Cascade Verifier)**，按顺序严格检查每个原子动作 $ a_k = \langle r_k, d_k, f_k, p_k \rangle $：
    1. **空间拓扑验证 (VR)**：房间是否存在？
    2. **实体对齐验证 (VD)**：设备是否在该房间？
    3. **功能支持验证 (VC)**：设备是否支持该操作？
  - 对于 `Cmixed` 指令，采用 **“生成再过滤 (Generate-and-Filter)”** 策略，在保留有效子任务的同时，用 `error_input` 替换无效操作，防止 **任务遗漏 (Task Omission)** 和 **强制幻觉 (Forced Hallucination)**。

### 相比现有方法的优势
| 维度 | DS-IA | 现有方法（如 SAGE） |
|------|-------|------------------|
| **安全性** | 极高（通过级联检查和早期拒绝） | 中等（依赖运行时纠错） |
| **交互效率** | 高（减少不必要的用户询问） | 低（陷入“交互频率困境”） |
| **鲁棒性** | 强（能正确处理混合指令） | 弱（易出现全有或全无失败） |
| **计算效率** | 更优（提前拒绝无效请求，节省昂贵的 autoregressive 生成开销） | 较差（对所有请求都尝试生成） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **HomeBench**：包含 100 个模拟家居环境和 2,500 条指令，用于评估模型在物理约束下的 **鲁棒性 (robustness)** 和抗幻觉能力。指令分为三类：
  - Valid（有效）
  - Invalid（无效，占 38.6%）
  - Mixed（混合）
- **SAGE Benchmark**：包含 50 个复杂任务，聚焦 **人机交互 (HRI)** 效率，涵盖设备解析、个性化、意图消歧等场景。

### 实验设置和评估指标
#### HomeBench 指标（侧重物理安全）
- **Exact Match (EM)**：主要指标，衡量生成的动作序列是否完全匹配标准答案（对无效指令需输出 `error_input`）。
- **F1-Score**：细粒度评估预测的房间、设备、操作/属性的精确率和召回率。

#### SAGE Benchmark 指标（侧重交互效率）
- **Task Success Rate (Succ. Rate)**：任务最终是否被正确解决。
  - **Autonomous Succ. Rate**：在信息充分时自主完成任务的能力（越高越好）。
  - **Clarification Succ. Rate**：在存在不可约歧义时，能否正确发起澄清请求（越高越好）。

### 基线方法对比
- **Standard Few-Shot**：使用完整环境上下文的 4-shot 提示，代表原生 LLM 能力。
- **SAGE (Iterative ReAct)**：最先进的工业级框架，使用工具调用（如 `query_devices`, `ask_user`）进行迭代推理。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
#### HomeBench 结果（Qwen-2.5-7B-Instruct）
| Method | Overall EM (%) | IS (Invalid Single) EM (%) | MM (Mixed Multi) F1 (%) |
|--------|----------------|----------------------------|--------------------------|
| Baseline | 29.98 | 14.07 | 33.23 |
| SAGE | 1.77 | 29.84 | 29.80 |
| **DS-IA (Ours)** | **58.56** | **87.04** | **77.42** |

- **DS-IA 的 EM 比 Baseline 提升超过 28%**，在无效指令上的拒绝率达到 **87.04%**，显著优于 Baseline 的 14.07%。
- 在混合指令上，F1 分数从 Baseline 的 33.23% 提升至 **77.42%**，证明 “Generate-and-Filter” 策略有效。

#### SAGE Benchmark 结果（GPT-4o-mini）
| Task Category | SAGE (%) | DS-IA (%) |
|-------------|---------|----------|
| **Autonomous Succ. Rate** | 42.86 | **71.43** |
| **Clarification Succ. Rate** | 75.00 | **75.00** |
| Intent Resolution | 33.33 | **70.83** |
| Device Resolution | 46.15 | **69.44** |
| Persistence | 25.00 | **100.00** |

- DS-IA 将 **Autonomous Success Rate 从 42.86% 提升至 71.43%**，大幅减少了不必要的用户干扰。
- 在需要长期状态记忆的 **Persistence 任务上达到 100% 成功率**，远超 SAGE 的 25.00%。

### 消融实验结果
在 1,000 个任务上的消融研究（移除 Stage 1）表明：
- **安全性**：移除 IA 模块后，对无效指令的拒绝率（IS EM）从 **88.69% 下降至 83.29%**。
- **计算效率**：DS-IA 通过早期拒绝，将 Stage 2 的代码生成调用次数减少了 **18.1%**（1000 → 819），节省了超过 427,000 个生成 token。
- **经济性**：虽然 Stage 1 增加了约 3.26M 输入 token，但由于输入处理（prefill）成本远低于自回归生成（decoding），整体计算更高效。

---

## 4. 关键结论和发现

### 主要发现
1. **安全性优先**：DS-IA 通过 **Stage 1 的语义防火墙** 和 **Stage 2 的级联验证**，实现了高达 **87.04%** 的无效指令拒绝率，有效防止了“强制接地 (Forced Grounding)”风险。
2. **提升自主性**：通过解耦意图路由与物理生成，DS-IA 在 SAGE Benchmark 上将 **Autonomous Success Rate 提升了近 30%**，解决了“交互频率困境”。
3. **长周期鲁棒性强**：在需要持久化状态的任务上实现 **100% 成功率**，证明其结构化接地机制在长时间交互中更具优势。

### 方法的局限性
1. **依赖实时快照**：系统性能受限于环境状态快照 $ S_t $ 的新鲜度，若状态更新延迟可能导致误判。
2. **文本元数据依赖**：目前仅使用文本形式的设备元数据，无法处理视觉指代（如“打开那盏红灯”）。
3. **未建模用户偏好**：对基于偏好的歧义（preference-based ambiguity）缺乏个性化记忆机制。

### 未来工作方向
1. **多模态感知 (Multimodal Perception)**：集成 Vision-Language Models (VLMs) 以处理视觉参考和多感官数据。
2. **隐私保护的小模型蒸馏 (Privacy-Preserving SLM Distillation)**：将 Intent Analysis 模块蒸馏为边缘友好的 Small Language Models (SLMs)，实现本地化部署，保护用户隐私。
3. **个性化记忆 (Personalized Memory)**：引入 Vector Database (RAG) 存储用户习惯，隐式解决偏好类歧义。

> **总结**：DS-IA 提供了一个 **安全、高效、可靠** 的下一代具身 IoT 智能体蓝图，成功弥合了语言推理与物理执行之间的鸿沟。

</details>

---

### 13. [Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights](https://arxiv.org/abs/2603.16817)

**Authors**: Yi Chen, Daiwei Chen, Sukrut Madhav Chikodikar, Caitlyn Heqi Yin, Ramya Korlakai Vinayak  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.16817v1  

#### Abstract
Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Is Conformal Factuality for RAG-based LLMs Robust?

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在生成过程中普遍存在 **hallucination**（幻觉）现象，即输出看似合理但事实错误的内容，这严重限制了其在医疗、法律等高风险领域的可靠部署。尽管 **Retrieval-Augmented Generation (RAG)** 和 **Conformal Prediction (CP)** 被提出用于缓解此问题，但仍存在以下关键挑战：
- **RAG** 虽能通过检索外部知识增强生成，但无法提供统计意义上的 **factuality guarantee**（事实性保证）。
- **Conformal Factuality** 虽能通过过滤低置信度声明提供统计保证，但常导致输出空洞（vacuous）或信息量不足，且其鲁棒性（robustness）未被系统评估。

### 提出的新方法与新思路
本文系统性地分析了基于 RAG 的 LLMs 中 **Conformal Factuality** 的可靠性，并提出了以下创新：

#### （1）新颖的评估指标（Novel Metrics）
为克服传统指标（如 Empirical Factuality）仅奖励“无错误”而忽略“有用性”的缺陷，作者提出了一系列 **informativeness-aware**（信息感知）指标：
- **Non-empty Rate (NR)**：衡量保留至少一个声明的输出比例，避免空输出。
- **Non-vacuous Empirical Factuality (NvEF)**：仅在非空输出上计算事实性，反映“有信息量时的正确率”。
- **Sufficient Correctness (SC)**：衡量输出是否包含足够正确的信息以推断出最终答案，关注任务级效用。
- **Conditional Sufficient Correctness (CSC)**：在初始输出已满足 SC 的前提下，衡量过滤后输出仍保持充分性的能力，隔离了生成质量与过滤过程的影响。

这些指标共同揭示了 **factuality-informativeness trade-off**（事实性-信息量权衡），为评估提供了更全面的视角。

#### （2）系统性分析框架
构建了一个完整的 Conformal Factuality 评估框架，涵盖 **generation, scoring, calibration, robustness, efficiency** 五大维度，首次对 RAG+CP 组合进行了全面剖析。

### 相比现有方法的优势
- **评估更全面**：超越了单纯的事实性提升，引入了信息量和任务效用作为核心评估标准。
- **发现更具指导意义**：揭示了现有方法在高事实性要求下的“空洞化”问题和对分布偏移的脆弱性，为实际部署提供了重要警示。
- **效率洞察**：证明轻量级验证器（lightweight verifiers）可媲美甚至优于大模型评分器，显著降低计算开销。

---

## 2. 核心实验方法和设置

### 数据集
实验在三个多样化基准上进行，覆盖不同任务类型：
- **FActScore**：开放域传记生成，评估事实性摘要能力。
- **MATH**：数学推理问题，评估逻辑与计算准确性。
- **Natural Questions (NQ)**：真实世界问答，评估参考问答能力。

### 实验设置
- **模型家族**：使用多个开源 LLM 家族（Qwen3, Llama-3.x, SmolLM2, gpt-oss），覆盖不同架构（Dense, MoE）、规模（135M 到 117B 参数）和推理能力（启用 `think` 标签）。
- **Conformal Factuality 流程**：
  1. **生成**：LLM 在检索到的参考 `R(x)` 上生成响应 `y`。
  2. **解析**：将 `y` 分解为原子声明 `c_i`。
  3. **评分**：使用 **scoring function** `f` 为每个声明打分。
  4. **校准**：在独立的校准集上确定过滤阈值 `T_α`。
  5. **过滤**：保留得分高于 `T_α` 的声明。
  6. **合并**：将保留的声明合并为最终输出 `y'`。

### 评分函数（Scoring Functions）
比较了两大类：
- **Entailment-based Scorers**：
  - **Document Entailment**：使用 DocNLI 模型计算整个参考与声明间的蕴含关系。
  - **Sentence-level Entailment**：使用 `roberta-large-mnli` 计算与各句的蕴含，再聚合（保守或平均）。
- **LLM-based Model Confidence Scorers**：使用 LLM 自身作为评分器，探索提示工程（prompting strategies）的影响，如是否提供参考、Chain-of-Thought、数值 vs 布尔输出、一致性平均等。

### 评估指标
- **传统指标**：Empirical Factuality (EF), Power, False Positive Rate (FPR), Correctness。
- **本文提出的新指标**：Non-empty Rate (NR), Non-vacuous EF (NvEF), Sufficient Correctness (SC), Conditional SC (CSC)。

### 基线方法对比
- **不同评分函数之间的对比**：LLM-based 模型置信度 vs. Entailment-based 蕴含模型。
- **不同模型规模的对比**：探究“更大模型是否必然更好”。
- **不同校准策略的对比**：匹配分布 vs. 不匹配分布的校准数据。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1. **Conformal Filtering 存在严重的信息损失**：
   - 在高事实性水平（如 1-α=0.9）下，所有评分函数的 **Power** 和 **Non-empty Rate (NR)** 都急剧下降。
   - 这表明为了达到高事实性保证，系统不得不过滤掉大量（包括正确的）信息，导致输出空洞，**实用性极低**。

2. **轻量级验证器优于大模型评分器**：
   - **Entailment-based scorers**（尤其是 Document Entailment）在 **EF, Power, SC** 等多项指标上**匹配或超越**了基于大 LLM 的模型置信度评分器。
   - 计算成本方面，DeBERTa/RoBERTa 等蕴含模型所需的 **FLOPs 比 LLM-based 评分器少 100 倍以上**。
   - **结论**：更强的事实性保证**不需要**更大、更昂贵的验证器。

3. **模型规模不等于更好的过滤效果**：
   - 在 Llama-3.x 家族中，增大模型规模有一定收益。
   - 但在 Qwen3 和 SmolLM2 家族中，**扩大模型规模并未带来一致提升，甚至有时表现更差**。
   - 小至 0.6B 的 Qwen3-0.6B 在 **SC/CSC** 上的表现与 32B 模型相当。

4. **Conformal Factuality 对分布偏移和干扰项极其脆弱**：
   - **校准分布偏移**：当校准数据来自不同分布（如 GPT-4 生成）时，实证事实性（EF）会**低于目标水平**，尤其是在 MATH 数据集上。
   - **对抗性干扰项（Distractors）**：向测试输出中注入看似合理但错误的声明后，EF 同样会跌破目标线。
   - **尝试缓解失败**：即使在校准集中也加入干扰项（distraction-aware calibration），虽然能恢复 EF，但代价是 **Non-empty Rate 急剧下降**，因为阈值变得过于严格。

---

## 4. 关键结论和发现

### 主要发现
1. **事实性-信息量权衡是根本矛盾**：当前的 Conformal Factuality 框架在追求高事实性时，会牺牲信息量，导致输出空洞，**高 EF 可能对应低任务效用**。
2. **鲁棒性是重大缺陷**：该框架的统计保证**高度依赖于校准数据与测试数据的分布一致性**。一旦出现分布偏移或对抗性干扰，保证就会失效。
3. **轻量高效是可行路径**：**Lightweight entailment-based verifiers** 在性能上不逊于甚至优于 LLM-based scorers，同时计算成本低几个数量级，是更实用的选择。
4. **小模型也能胜任**：对于评分任务，**更大的模型规模并非必要**，经过良好设计的小模型同样有效。

### 方法的局限性
- **过滤机制的固有缺陷**：Conformal Filtering 只能删除信息，不能补充。如果初始生成就缺少关键信息，过滤后也无法得到正确答案。
- **对干扰项缺乏辨别力**：现有的评分函数无法可靠地区分正确的声明和精心构造的、看似合理的错误声明。
- **校准数据要求苛刻**：需要高质量、与部署环境完全匹配的校准数据，这在实践中难以获得。

### 未来工作方向
- **开发新的可靠性方法**：需要超越简单的阈值过滤，探索能同时保证 **robustness**（鲁棒性）和 **usefulness**（实用性）的新范式。
- **设计更鲁棒的评分函数**：研究能够抵抗分布偏移和对抗性攻击的评分机制。
- **端到端优化**：将生成、检索、评分和过滤作为一个整体进行联合优化，而非孤立处理。
- **探索动态或自适应校准**：减少对静态、完美匹配校准集的依赖。

**总而言之，本文揭示了当前 Conformal Factuality for RAG-based LLMs 的脆弱性和局限性，呼吁社区将鲁棒性和实用性作为未来工作的核心指标，并为构建更可靠、高效的 LLM 系统提供了重要的实践指导。**

</details>

---

### 14. [Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR](https://arxiv.org/abs/2603.16184)

**Authors**: Quy-Anh Dang, Chris Ngo  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.16184v1  

#### Abstract
We present Polyglot-Lion, a family of compact multilingual automatic speech recognition (ASR) models tailored for the linguistic landscape of Singapore, covering English, Mandarin, Tamil, and Malay. Our models are obtained by fine-tuning Qwen3-ASR-0.6B and Qwen3-ASR-1.7B exclusively on publicly avai...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
新加坡是一个多语言社会，官方语言包括 **English, Mandarin, Tamil, Malay**，日常交流中普遍存在 **code-switching**（语码转换）和 **Singlish**（混合英语变体）。然而，现有的多语言 **ASR**（自动语音识别）系统在以下方面存在不足：
- 多数通用模型（如 Whisper、MMS）对低资源语言（如 Tamil、Malay）识别效果差；
- 针对新加坡场景的专用模型（如 MERaLiON-2-10B-ASR）虽然性能强，但训练成本极高（需128块H100 GPU），难以普及；
- 大多数模型依赖显式的 **language-tag conditioning**，在语言未知或混合时表现下降。

### 🚀 提出的新方法与创新
作者提出 **Polyglot-Lion**，一个轻量级、高效的多语言 ASR 模型家族，其核心创新在于：

#### （1）**平衡采样策略（Balanced Multilingual Sampling）**
- 对训练数据进行两级上采样（two-stage upsampling）：
  - **Stage 1**: 在每种语言内部，将所有子数据集上采样至最大规模；
  - **Stage 2**: 将四种语言的数据统一上采样至相同数量，确保每种语言占25%。
- 无需调参，确定性强，保证语言间完全均衡。

#### （2）**无语言标签解码（Language-Agnostic Decoding）**
- **移除语言标识符（language tags）**，不向解码器输入任何语言提示；
- 模型必须从音频信号中**隐式识别语言**，从而增强对 code-switching 和未知语言段落的鲁棒性。

#### （3）基于公开数据的高效微调
- 仅使用 **公开可用的语音语料库**（no proprietary data）；
- 在单张 **RTX PRO 6000 GPU** 上微调 48 小时，显著降低部署门槛。

### 🔍 相比现有方法的优势
| 维度 | Polyglot-Lion | MERaLiON-2-10B-ASR |
|------|---------------|---------------------|
| 参数量 | 1.7B | 10B（6×更大） |
| 训练成本 | $81（单卡） | $18,862（128 H100） |
| 推理速度 | 0.10 s/sample | 2.02 s/sample |
| 数据来源 | 公开数据 | 包含专有数据 |
| 语言标签 | 无（更鲁棒） | 有（依赖先验） |

> ✅ **结论**：通过简单而有效的设计，实现了接近 SOTA 的精度，同时大幅降低成本与延迟。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
全部为公开可用语料，覆盖四种官方语言：

| 语言 | 数据集 | 特点 |
|------|--------|------|
| **English** | Librispeech, NSC | NSC 是新加坡口音英语，含多种说话风格 |
| **Mandarin** | AISHELL-1, AISHELL-3, Common Voice 23, Fleurs | 覆盖标准普通话与多样性发音 |
| **Tamil** | SLR127, SLR65, Common Voice 23, Fleurs | SLR127 是最大 Tamil 语料（~120h） |
| **Malay** | Mesolitica, Fleurs | Mesolitica 是马来西亚马来语，贴近本地用法 |

> ⚠️ 数据不平衡严重：  
> - 英语 + 普通话占总时长约 65%；  
> - 马来语仅占 8%，泰米尔语虽总量高但预训练中代表性弱。

### 🎯 实验设置与评估指标

#### 模型架构
- 基于 **Qwen3-ASR-0.6B** 和 **Qwen3-ASR-1.7B** 微调；
- 架构：Conformer 编码器 + 自回归解码器；
- 发布两个版本：`Polyglot-Lion-0.6B` 与 `Polyglot-Lion-1.7B`。

#### 训练细节
- 单卡训练：NVIDIA RTX PRO 6000（48GB VRAM）；
- 时间：48小时；
- Batch size：32（per-device 8 × 4步梯度累积）；
- 优化器：AdamW，学习率 2e-5，余弦退火调度。

#### 评估指标
| 语言 | 指标 | 说明 |
|------|------|------|
| English, Tamil, Malay | **WER**（Word Error Rate） | 基于词的编辑距离 |
| Mandarin | **CER**（Character Error Rate） | 中文无空格分词，字符级更合理 |

> 所有文本小写化、去标点，与训练一致。

### 🆚 基线方法对比
共比较 **8个基线模型**，涵盖三类：

| 类别 | 模型 |
|------|------|
| **通用多语言 ASR** | Whisper-large-v3-turbo |
| **音频-语言模型（ALM）** | Qwen2.5-Omni-3B/7B, SeaLLMs-Audio-7B |
| **基础模型（未微调）** | Qwen3-ASR-0.6B/1.7B |
| **专用高性能模型** | MERaLiON-2-10B-ASR（主对比对象） |

> 所有模型均使用公开 checkpoint 进行推理测试，硬件一致（RTX PRO 4500）。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Table 2）

| 模型 | 平均错误率（WER/CER↓） | 参数量 |
|------|--------------------------|--------|
| **Polyglot-Lion-1.7B** | **14.85** | 1.7B |
| MERaLiON-2-10B-ASR | 14.32 | 10B |
| Polyglot-Lion-0.6B | 16.52 | 0.6B |
| Qwen3-ASR-1.7B（未微调） | 53.76 | 1.7B |
| Whisper-large-v3-turbo | 33.04 | 0.8B |

> ✅ **Polyglot-Lion-1.7B 以 1.7B 参数达到接近 MERaLiON 的精度（差距仅 0.53）**

### 🔍 分语言性能亮点

| 语言 | 表现 |
|------|------|
| **English** | 在 Librispeech 达 **2.10 WER**，优于 MERaLiON（2.54）；在 NSC（新加坡英语）达 **5.28 WER**，仅次于 MERaLiON（4.62） |
| **Mandarin** | 在所有四个基准上全面领先，如 AISHELL-1 达 **1.45 CER**（MERaLiON: 3.09） |
| **Tamil** | 错误率从原始 Qwen3-ASR 的 >120% 下降至 **39.19%**（Common Voice），相对降低 **72%** |
| **Malay** | 在 Mesolitica 上达 **21.51 WER**，**超越所有基线**（MERaLiON: 25.90） |

### ⏱️ 推理速度对比（Table 3）
| 模型 | 推理延迟（s/sample） |
|------|------------------|
| **Polyglot-Lion-1.7B** | **0.104** |
| MERaLiON-2-10B-ASR | 2.015 |
| Whisper-large-v3-turbo | 0.282 |

> ✅ **推理速度快约 20×**

### 💰 训练成本对比（Table 4）
| 模型 | 训练成本 | 硬件配置 |
|------|-----------|------------|
| **Polyglot-Lion** | **$81** | 单张 RTX PRO 6000 |
| MERaLiON-2-10B-ASR | $18,862 | 128× H100 |

> ✅ **训练成本降低 233 倍**

### 🔍 消融实验分析（来自 Section 7）

#### （1）平衡采样的有效性
- 未经平衡的 Qwen3-ASR 在 Tamil 上 WER >120%，几乎失效；
- 加入平衡采样后，Tamil CV WER 降至 **39.19**，提升巨大；
- 同时未损害高资源语言性能 → 无负迁移（negative transfer）。

#### （2）语言无关解码的有效性
- 尽管没有语言标签输入，模型仍能准确识别四种语言（包括 Dravidian 的 Tamil）；
- 支持在 **code-switching 场景下稳定运行**，适合真实新加坡口语环境。

#### （3）参数效率分析
- Polyglot-Lion-1.7B（1.7B）性能远超更大的 Qwen2.5-Omni-7B（7B），说明 **数据平衡比参数规模更重要**；
- Polyglot-Lion-0.6B 仅损失 1.67 点误差，适合边缘部署。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **语言平衡微调是关键**：即使在中等规模模型上，通过合理的数据重采样，也能极大提升低资源语言（Tamil、Malay）的表现。
2. **语言标签非必需**：去除 language tag conditioning 不仅可行，反而增强了模型对语言切换的适应能力。
3. **高效 ≠ 低性能**：Polyglot-Lion 在精度、速度、成本之间取得极佳平衡，是首个可在消费级 GPU 上复现的“近 SOTA”新加坡多语言 ASR 系统。
4. **公开数据足够强大**：仅用公开语料即可逼近使用专有数据训练的大模型性能。

### ⚠️ 局限性
1. **在 NSC 上仍有差距**：Polyglot-Lion-1.7B（5.28）略逊于 MERaLiON（4.62），可能因后者使用更多新加坡本地语料。
2. **Tamil 仍有提升空间**：尽管大幅提升，但仍落后于 MERaLiON（39.19 vs 31.78），反映 Tamil 的形态复杂性和方言差异挑战大。
3. **缺乏 code-switching 测试集评估**：当前评测集均为单语，未包含 SEAME 或 CS-Singlish 等混合语句数据集。

### 🔮 未来工作方向
1. 引入 **新加坡本地语音数据**（如完整 NSC）进行领域自适应预训练；
2. 利用 **跨语言迁移学习**，结合 Tamil 文本语料进行 speech-text 联合训练；
3. 设计 **code-switching-aware 训练目标**，显式建模语言切换行为；
4. 探索 **更小量化版本**（如 INT8、GGUF）用于移动端部署。

---

> **总结一句话**：  
> **Polyglot-Lion 证明了“数据平衡 + 轻量微调”可以替代“大规模 + 高成本”的多语言 ASR 路径，在保持高性能的同时实现极致的成本与部署友好性。**

</details>

---

### 15. [Game-Theory-Assisted Reinforcement Learning for Border Defense: Early Termination based on Analytical Solutions](https://arxiv.org/abs/2603.15907)

**Authors**: Goutam Das, Michael Dorothy, Kyle Volle, Daigo Shishika  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.15907v1  

#### Abstract
Game theory provides the gold standard for analyzing adversarial engagements, offering strong optimality guarantees. However, these guarantees often become brittle when assumptions such as perfect information are violated. Reinforcement learning (RL), by contrast, is adaptive but can be sample-ineff...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Game-Theory-Assisted Reinforcement Learning for Border Defense: Early Termination based on Analytical Solutions

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对**边境防御游戏**（border defense game）中的多智能体对抗问题，解决在**部分可观测环境**下，传统 **Reinforcement Learning (RL)** 和 **game-theoretic 分析方法** 各自存在的缺陷：

- **Analytical game-theoretic 方法**（如微分博弈）虽然能提供最优解，但依赖于完全信息假设，在实际中难以应用。
- **纯端到端的 MARL 方法**（如 MAPPO）虽可处理不确定性，但需同时学习“搜索”和“追击”策略，导致样本效率低、训练缓慢。

### 🚀 提出的新方法与创新思路
提出一种**混合框架**（hybrid framework），将 **game theory (GT)** 与 **multi-agent reinforcement learning (MARL)** 结合，核心是：

> **GT-assisted early termination**：  
> 在防御者感知到攻击者的一瞬间（`ts`），利用 **Apollonius Circle (AC)** 解析计算后续追击阶段的 **Nash equilibrium payoff** $ J^*(x(t_s)) $，并立即终止当前 episode，直接赋予该理论最优奖励。

#### 具体机制：
- **Phase I（搜索阶段）**：由 MARL 学习如何部署和移动以最大化检测概率和有利初始配置。
- **Phase II（追击阶段）**：不通过模拟学习，而是用 GT 解析求解最优结局，实现“免学习”。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **Sample Efficiency** | 避免重复学习已知的最优追击策略，显著减少训练步数 |
| **Convergence Speed** | 更快收敛至高性能策略 |
| **Policy Quality** | 引导学习更优的空间布局（spatial configuration），提升整体防御效果 |
| **Theoretical Guarantee** | 追击阶段的结果始终为理论最优，避免因探索不良策略而降低性能 |

---

## 2. 核心实验方法和设置

### 🧪 实验环境与任务设定
- **任务类型**：Border Defense Game（也称 Target Defense 或 Perimeter Defense）
- **环境描述**：
  - 二维有界区域 $\Omega = [0,1] \times [0,1]$
  - 攻击者从顶部生成，目标是穿越底线 $T = \{(x,y)\ |\ y=0\}$ 成功渗透
  - 多个防御者协作拦截，拥有有限感知半径 $p_s$ 和捕获半径 $p_c$
- **动态模型**：所有智能体采用 single-integrator dynamics
- **信息结构**：
  - 初始状态双边未知（partial information）
  - 检测后进入完全信息状态（perfect information）

### 📊 实验设置
| 参数 | 设置 |
|------|------|
| **Speed Ratio** | $v = v_D / v_A = 3.33$（防御者更快） |
| **Sensing Radius** | $p_s \in \{0.15, 0.3, 0.35\}$ |
| **Capture Radius** | $p_c = 0.07$ |
| **Time Step** | $\Delta t = 0.05$ |
| **Max Episode Length** | $T_{\text{max}} = 2 / (v_A \Delta t)$ |
| **Simulation Platform** | VMAS（Vectorized Multi-Agent Simulator） |
| **RL Framework** | BenchMARL |
| **算法** | MAPPO（Multi-Agent Proximal Policy Optimization） |
| **并行环境数** | 20 |
| **总训练帧数** | 1.2–1.8 million |
| **随机种子** | 3次独立运行取平均 |

### 🎯 评估指标
1. **Mean Reward**：终端奖励期望值，定义为攻击者被捕时距离目标线的距离（越高越好）
2. **Detection Success Rate**：成功感知攻击者的比例
3. **Convergence Speed**：达到稳定高回报所需的训练迭代次数
4. **Spatial Configuration Quality**：检测时刻的 Nash equilibrium payoff $J^*(x(t_s))$，反映布局质量

### ⚖️ 基线方法对比
| 方法 | 描述 |
|------|------|
| **GT-Assisted (Ours)** | 检测即终止，奖励为解析解 $J^*$ |
| **End-to-End Learning (Baseline)** | 完整模拟搜索 + 追击全过程，奖励为最终结果 |

> 所有 defender 使用共享奖励（shared reward），鼓励团队协作。

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（Homogeneous Teams）

#### ✅ 1v1 场景（单防御者 vs 单攻击者）
| 指标 | GT-Assisted | End-to-End | 提升 |
|------|-------------|------------|-------|
| 平均奖励 ($\mu$) | **0.605** | 0.545 | ↑ **10%** |
| 收敛速度 | 快速稳定 | 缓慢上升 | 显著更快 |

> 图5显示 GT 方法在约 100 轮内收敛，而 baseline 仍波动。

#### ✅ 3v1 场景（三防御者 vs 单攻击者）
| 指标 | GT-Assisted | End-to-End | 提升 |
|------|-------------|------------|-------|
| 平均奖励 ($\mu$) | **0.745** | 0.626 | ↑ **18.9%** |
| 方差 | 更小 | 较大 | 更稳定 |

> 图6表明 GT 方法不仅性能更高，且训练过程更平稳。

### 🔍 空间配置质量分析（图7）
基于 1000 次测试 episode 分析检测时刻的状态质量：

| 指标 | GT-Assisted | End-to-End |
|------|-------------|------------|
| **检测成功率** | **67.1%** | 45.2% | ↑ 21.9个百分点 |
| **累计表现**（Cumulative Performance） | 显著领先 | 落后明显 | — |
| **Apollonius Payoff 中位数**（仅检测成功 episode） | **0.768** | 0.701 | ↑ ~10% |

> 表明 GT 方法不仅能更早发现攻击者，还能形成更有利的围堵态势。

### ❌ 消融实验（隐含验证）
尽管未明确列出消融表，但以下设计构成有效控制变量比较：
- 是否使用 GT 提前终止 → 控制变量为“是否学习追击”
- 结果证明：**跳过追击学习不影响最终性能，反而提升整体效率**

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **GT 与 RL 可高效融合**：  
   将 game-theoretic 解析解嵌入 RL 训练流程，可在保留理论最优性的同时，大幅提升学习效率。

2. **Early Termination 是有效的加速机制**：  
   在检测瞬间终止 episode 并赋以 $J^*$ 奖励，等价于完整最优追击的期望回报（无折扣情况下），从而无需学习已知行为。

3. **引导搜索策略优化**：  
   尽管未经历追击过程，MARL 仍能学会在 Phase I 构建有利于后续博弈的空间配置，说明 reward shaping 极其有效。

4. **适用于不同规模团队**：  
   方法在 1v1 和 3v1 场景下均表现出一致优势，具备良好扩展性。

### ⚠️ 局限性
1. **依赖可解析的 Phase II 解**：  
   当追击阶段无法获得闭式解（如复杂动力学、非对称能力）时，本方法适用性受限。

2. **多攻击者场景尚未覆盖**：  
   当前仅考虑单一攻击者；多个攻击者会引入部分检测、异步感知等问题。

3. **Apollonius Circle 数值优化开销**：  
   多防御者情况需求解凸优化问题（intersection of disks），虽高效但仍有一定计算负担。

4. **capture radius > 0 时为保守估计**：  
   当 $p_c > 0$，真实捕获可能早于 AC 预测点，因此 $J^*$ 是下界近似。

### 🔮 未来工作方向
1. **扩展至 multiple attackers 场景**：  
   处理异步感知、部分观测转移等挑战。

2. **大规模团队下的 scalability 优化**：  
   加速多 defender 下 Apollonius Circle 交集的计算（如分布式优化或近似方法）。

3. **将 GT insights 引入 Phase I 搜索阶段**：  
   如设计基于 dominance region 的启发式 reward shaping 或 curriculum learning。

4. **结合其他 game-theoretic 工具**：  
   探索 barrier functions、value function approximation 等进一步增强 MARL 效率。

---

> 💡 **一句话总结**：  
> 本文提出了一种“**用 game theory 告诉 RL ‘后面怎么做是对的’，让 RL 专心解决‘前面怎么做好’**”的思想，实现了 border defense 任务中搜索与追击策略的解耦优化，在性能、效率和稳定性上全面超越端到端学习方法。

</details>

---

### 16. [Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets](https://arxiv.org/abs/2603.16731)

**Authors**: Kristi Topollai, Anna Choromanska  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.16731v1  

#### Abstract
Quantizing optimizer states is becoming an important ingredient of memory-efficient large-scale pre-training, but the resulting optimizer dynamics remain only partially understood. We study low-precision exponential moving average (EMA) optimizer states and show how quantization can cause many nomin...

---

### 17. [Prompt Engineering for Scale Development in Generative Psychometrics](https://arxiv.org/abs/2603.15909)

**Authors**: Lara Lee Russell-Lasalandra, Hudson Golino  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.15909v1  

#### Abstract
This Monte Carlo simulation examines how prompt engineering strategies shape the quality of large language model (LLM)--generated personality assessment items within the AI-GENIE framework for generative psychometrics. Item pools targeting the Big Five traits were generated using multiple prompting ...

---

### 18. [MOSAIC: Composable Safety Alignment with Modular Control Tokens](https://arxiv.org/abs/2603.16210)

**Authors**: Jingyu Peng, Hongyu Chen, Jiancheng Dong, Maolin Wang, Wenxi Li, Yuchen Li, Kai Zhang, Xiangyu Zhao  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.16210v1  

#### Abstract
Safety alignment in large language models (LLMs) is commonly implemented as a single static policy embedded in model parameters. However, real-world deployments often require context-dependent safety rules that vary across users, regions, and applications. Existing approaches struggle to provide suc...

---

### 19. [IQuest-Coder-V1 Technical Report](https://arxiv.org/abs/2603.16733)

**Authors**: Jian Yang, Wei Zhang, Shawn Guo, Zhengmao Ye, Lin Jing, Shark Liu, Yizhi Li, Jiajun Wu, Cening Liu, X. Ma, Yuyang Song, Siwei Wu, Yuwen Li, L. Liao, T. Zheng, Ziling Huang, Zelong Huang, Che Liu, Yan Xing, Renyuan Li, Qingsong Cai, Hanxu Yan, Siyue Wang, Shikai Li, Jason Klein Liu, An Huang, Yongsheng Kang, Jinxing Zhang, Chuan Hao, Haowen Wang, Weicheng Gu, Ran Tao, Mingjie Tang, Peihao Wu, Jianzhou Wang, Xianglong Liu, Weifeng Lv, Bryan Dai  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.16733v1  

#### Abstract
In this report, we introduce the IQuest-Coder-V1 series-(7B/14B/40B/40B-Loop), a new family of code large language models (LLMs). Moving beyond static code representations, we propose the code-flow multi-stage training paradigm, which captures the dynamic evolution of software logic through differen...

---

### 20. [SIA: A Synthesize-Inject-Align Framework for Knowledge-Grounded and Secure E-commerce Search LLMs with Industrial Deployment](https://arxiv.org/abs/2603.16137)

**Authors**: Zhouwei Zhai, Mengxiang Chen, Anmeng Zhang  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.16137v1  

#### Abstract
Large language models offer transformative potential for e-commerce search by enabling intent-aware recommendations. However, their industrial deployment is hindered by two critical challenges: (1) knowledge hallucination due to insufficient encoding of dynamic, fine-grained product knowledge, and (...

---

### 21. [Omnilingual SONAR: Cross-Lingual and Cross-Modal Sentence Embeddings Bridging Massively Multilingual Text and Speech](https://arxiv.org/abs/2603.16606)

**Authors**: Omnilingual SONAR Team, Jo\~ao Maria Janeiro, Pere-Llu\'is Huguet Cabot, Ioannis Tsiamas, Yen Meng, Vivek Iyer, Guillem Ram\'irez, Loic Barrault, Belen Alastruey, Yu-An Chung, Marta R. Costa-Jussa, David Dale, Kevin Heffernan, Jaehyeong Jo, Artyom Kozhevnikov, Alexandre Mourachko, Christophe Ropers, Holger Schwenk, Paul-Ambroise Duquenne  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.16606v1  

#### Abstract
Cross-lingual sentence encoders typically cover only a few hundred languages and often trade downstream quality for stronger alignment, limiting their adoption. We introduce OmniSONAR, a new family of omnilingual, cross-lingual and cross-modal sentence embedding models that natively embed text, spee...

---

### 22. [MedArena: Comparing LLMs for Medicine-in-the-Wild Clinician Preferences](https://arxiv.org/abs/2603.15677)

**Authors**: Eric Wu, Kevin Wu, Jason Hom, Paul H. Yi, Angela Zhang, Alejandro Lozano, Jeff Nirschl, Jeff Tangney, Kevin Byram, Braydon Dymm, Narender Annapureddy, Eric Topol, David Ouyang, James Zou  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.15677v1  

#### Abstract
Large language models (LLMs) are increasingly central to clinician workflows, spanning clinical decision support, medical education, and patient communication. However, current evaluation methods for medical LLMs rely heavily on static, templated benchmarks that fail to capture the complexity and dy...

---

### 23. [PashtoCorp: A 1.25-Billion-Word Corpus, Evaluation Suite, and Reproducible Pipeline for Low-Resource Language Development](https://arxiv.org/abs/2603.16354)

**Authors**: Hanif Rahman  
**Category**: cs.CL  
**Published**: 2026-03-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.16354v1  

#### Abstract
We present PashtoCorp, a 1.25-billion-word corpus for Pashto, a language spoken by 60 million people that remains severely underrepresented in NLP. The corpus is assembled from 39 sources spanning seven HuggingFace datasets and 32 purpose-built web scrapers, processed through a reproducible pipeline...

---

### 24. [Time-Aware Prior Fitted Networks for Zero-Shot Forecasting with Exogenous Variables](https://arxiv.org/abs/2603.15802)

**Authors**: Andres Potapczynski, Ravi Kiran Selvam, Tatiana Konstantinova, Shankar Ramasubramanian, Malcolm Wolff, Kin G. Olivares, Ruijun Ma, Mengfei Cao, Michael W. Mahoney, Andrew Gordon Wilson, Boris N. Oreshkin, Dmitry Efimov  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.15802v1  

#### Abstract
In many time series forecasting settings, the target time series is accompanied by exogenous covariates, such as promotions and prices in retail demand; temperature in energy load; calendar and holiday indicators for traffic or sales; and grid load or fuel costs in electricity pricing. Ignoring thes...

---

### 25. [Discovery of interaction and diffusion kernels in particle-to-mean-field multi-agent systems](https://arxiv.org/abs/2603.15927)

**Authors**: Giacomo Albi, Alessandro Alla, Elisa Calzola  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.15927v1  

#### Abstract
We propose a data-driven framework to learn interaction kernels in stochastic multi-agent systems. Our approach aims at identifying the functional form of nonlocal interaction and diffusion terms directly from trajectory data, without any a priori knowledge of the underlying interaction structure. S...

---

### 26. [Deriving Hyperparameter Scaling Laws via Modern Optimization Theory](https://arxiv.org/abs/2603.15958)

**Authors**: Egor Shulgin, Dimitri von R\"utte, Tianyue H. Zhang, Niccol\`o Ajroldi, Bernhard Sch\"olkopf, Antonio Orvieto  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.15958v1  

#### Abstract
Hyperparameter transfer has become an important component of modern large-scale training recipes. Existing methods, such as muP, primarily focus on transfer between model sizes, with transfer across batch sizes and training horizons often relying on empirical scaling rules informed by insights from ...

---

### 27. [Dual Consensus: Escaping from Spurious Majority in Unsupervised RLVR via Two-Stage Vote Mechanism](https://arxiv.org/abs/2603.16223)

**Authors**: Kaixuan Du, Meng Cao, Hang Zhang, Yukun Wang, Xiangzhou Huang, Ni Li  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.16223v1  

#### Abstract
Current label-free RLVR approaches for large language models (LLMs), such as TTRL and Self-reward, have demonstrated effectiveness in improving the performance of LLMs on complex reasoning tasks. However, these methods rely heavily on accurate pseudo-label estimation and converge on spurious yet pop...

---

### 28. [Optimal uncertainty bounds for multivariate kernel regression under bounded noise: A Gaussian process-based dual function](https://arxiv.org/abs/2603.16481)

**Authors**: Amon Lahr, Anna Scampicchio, Johannes K\"ohler, Melanie N. Zeilinger  
**Category**: cs.LG  
**Published**: 2026-03-18  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.16481v1  

#### Abstract
Non-conservative uncertainty bounds are essential for making reliable predictions about latent functions from noisy data--and thus, a key enabler for safe learning-based control. In this domain, kernel methods such as Gaussian process regression are established techniques, thanks to their inherent u...

---

### 29. [GSI Agent: Domain Knowledge Enhancement for Large Language Models in Green Stormwater Infrastructure](https://arxiv.org/abs/2603.15643)

**Authors**: Shaohuang Wang  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.15643v1  

#### Abstract
Green Stormwater Infrastructure (GSI) systems, such as permeable pavement, rain gardens, and bioretention facilities, require continuous inspection and maintenance to ensure long-term performance. However, domain knowledge about GSI is often scattered across municipal manuals, regulatory documents, ...

---

### 30. [Theoretical Foundations of Latent Posterior Factors: Formal Guarantees for Multi-Evidence Reasoning](https://arxiv.org/abs/2603.15674)

**Authors**: Aliyu Agboola Alege  
**Category**: cs.AI  
**Published**: 2026-03-18  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.15674v1  

#### Abstract
We present a complete theoretical characterization of Latent Posterior Factors (LPF), a principled framework for aggregating multiple heterogeneous evidence items in probabilistic prediction tasks. Multi-evidence reasoning arises pervasively in high-stakes domains including healthcare diagnosis, fin...

---

## 🔧 Configuration

This bot is configured to look for papers containing the following keywords:
- LLM, RL, RLHF, Inference, Training, Attention, Pipeline, MOE, Sparse, Quantization, Speculative, Efficient, Efficiency, Framework, Parallel, Distributed, Kernel, Decode, Decoding, Prefill, Throughput, Fast, Network, Hardware, Cluster, FP8, FP4, Optimization, Scalable, Communication

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
