# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-25 06:48:31 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [PersonalQ: Select, Quantize, and Serve Personalized Diffusion Models for Efficient Inference](https://arxiv.org/abs/2603.22943)

**Authors**: Qirui Wang, Qi Guo, Yiding Sun, Junkai Yang, Dongxu Zhang, Shanmin Pang, Qing Guo  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.22943v1  

#### Abstract
Personalized text-to-image generation lets users fine-tune diffusion models into repositories of concept-specific checkpoints, but serving these repositories efficiently is difficult for two reasons: natural-language requests are often ambiguous and can be misrouted to visually similar checkpoints, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《PersonalQ: Select, Quantize, and Serve Personalized Diffusion Models for Efficient Inference》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
个性化 text-to-image 生成中存在两个核心挑战：
- **Checkpoint 选择困难**：自然语言请求常具有歧义，标准检索方法（如 RAG 或 reranker）难以准确将用户意图映射到正确的 personalized checkpoint，尤其在概念视觉相似或描述重叠时容易误匹配。
- **量化损害个性化质量**：现有的 Post-training Quantization (PTQ) 方法虽然能压缩模型、节省内存，但会破坏对 trigger token 敏感的表示路径，导致生成图像的身份保真度（identity fidelity）和文本对齐能力下降。

### 提出的新方法与创新思路
作者提出 **PersonalQ** ——一个统一框架，通过共享信号 **trigger token** 联结 checkpoint 选择与量化过程，实现高效且保真的个性化模型服务。

#### 主要模块：
- **Check-in**：基于 trigger token 的 intent-aligned checkpoint 选择模块
  - 结合 **intent-aware hybrid retrieval**（稀疏 + 密集检索）
  - 引入 **LLM-based reranking** 对候选 checkpoint 进行重排序
  - 支持 **clarification mechanism**，当多个意图仍可能成立时主动提问以消除歧义
  - 最终输出插入 trigger token 的重写 prompt（如 `bear` → `<bear-v4>`）

- **Trigger-Aware Quantization (TAQ)**：面向 trigger token 的混合精度量化策略
  - 在 cross-attention 中识别并保护由 trigger token 激活的 Key/Value 行及其注意力权重
  - 对非 trigger 路径进行激进低比特量化（如 4-bit），而 trigger 相关部分保持高精度（FP32）
  - 实现“关键路径保护 + 其余路径压缩”的平衡

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **Selection** | 显著优于 Random、Reranker 和 Stylus，在意图对齐上获得最高 LLM-judge 得分与人类偏好率 |
| **Quantization** | 在相同 bit-width 下，TAQ 的 FID 更低、CLIP Score 更高，压缩-质量权衡显著优于 Q-Diffusion、TFMQ-DM、DGQ 等 PTQ 方法 |
| **系统设计** | 首次将 selection 与 quantization 通过 trigger token 统一建模，形成端到端可扩展的服务流水线 |

---

## 2. 核心实验方法和设置

### 数据集
- **Personalized Checkpoint Repository**
  - 构建包含 **1,000 个 personalized Stable Diffusion checkpoints** 的大型仓库
  - 覆盖 **20 个 concept categories**（如 `<dog>`, `<cat>`, `<person>`, `<bear>` 等）
  - 每类有 **50 个时间版本**（v1–v50），模拟真实用户的持续微调行为
  - 使用 DreamBooth（SD1.5）和 LoRA DreamBooth（SDXL-Turbo）训练

- **REPO-PROMPTS Benchmark**
  - 新构建的评估数据集，含 **500 条自然语言查询**
  - 包括三类请求：
    - 单一匹配（350 条）
    - 歧义需澄清（100 条）
    - 无匹配（50 条）
  - 每条标注：`query`, `candidate_pool`, `ground_truth`, `requires_clarification`, `no_match`

### 实验设置与评估指标

#### 评估任务
1. **Checkpoint Selection 性能**
   - **自动指标**：未直接报告 accuracy，依赖 LLM judge
   - **LLM-as-a-Judge**：使用 Gemini/Qwen/GPT-4o 对生成结果打分（Likert 1–5 分），维度包括 subject、style、temporal、context fit，取平均为 **Intent Score**
   - **人工评估**：两两比较（pairwise comparison），统计 **Human Preference Rate**

2. **Quantization 性能**
   - **基准模型**：
     - Stable Diffusion v1.5
     - SDXL-Turbo
   - **量化配置**：
     - W8A8（8-bit weights / 8-bit activations）
     - W8A4（8-bit weights / 4-bit activations）
   - **评估指标**：
     - **FID ↓**（Fréchet Inception Distance）：衡量图像分布距离
     - **CLIP Score ↑**：衡量图文语义一致性
     - **BOPs ↓**（Bit Operations）：计算开销，反映推理效率

#### 基线方法对比
| 类别 | 基线方法 |
|------|--------|
| **Selection** | Random, Reranker (Qwen3-Reranker-4B), Stylus |
| **Quantization** | Q-Diffusion, TFMQ-DM, DGQ |
| **Weight PTQ** | Adaround, BRECQ（用于 block reconstruction） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table I & II）

#### ✅ Quantization 性能（Table I）
| Model | Method | Bit (W/A) | FID (MS-COCO) ↓ | CLIP Score ↑ | BOPs ↓ |
|-------|--------|-----------|------------------|---------------|---------|
| SD1.5 Full Precision | – | 32/32 | 10.96 | 0.315 | 893 |
| SD1.5 TAQ | **Ours** | **8/8** | **11.03** | **0.297** | **56** |
| SD1.5 DGQ | Baseline | 8/8 | 15.24 | 0.291 | 56 |
| SD1.5 TAQ | **Ours** | **8/4** | **38.84** | **0.264** | **28** |
| SD1.5 Q-Diffusion | Baseline | 8/4 | 223.11 | 0.082 | 28 |

> 🔍 **结论**：TAQ 在 W8A8 下几乎接近 full precision 表现；在更激进的 W8A4 下仍大幅领先其他方法（FID 降低 ~50%以上）

#### ✅ Checkpoint Selection 性能（Table II）
| Method | Intent Score (↑) | Human Preference Rate (↑) |
|--------|--------------------|----------------------------|
| Random | 2.14 ± 0.82 | 89.1% |
| Reranker | 3.21 ± 0.76 | 85.7% |
| Stylus | 3.68 ± 0.69 | 82.1% |
| **Check-in (Ours)** | **4.42 ± 0.51** | **100.0%** |

> 🏆 Check-in 在意图对齐上显著胜出，人类评分偏好率达 **89.1%~100%**

#### ✅ 消融实验结果

##### Ablation on Check-in Components（Table III）
| Retrieval | Reasoning | Intent Score | FID | CLIP |
|----------|-----------|--------------|-----|------|
| × | × | 3.21 | 11.74 | 0.289 |
| √ | × | 3.99 | 11.61 | 0.291 |
| × | √ | 4.19 | 11.23 | 0.293 |
| √ | √ | **4.42** | **11.03** | **0.297** |

> 💡 Hybrid retrieval 和 personalized reasoning 均带来稳定增益，联合使用效果最佳

##### Ablation on TAQ（Table IV）
| Quantizer | Separate Trigger | FID (8/8) | CLIP (8/8) | FID (8/4) |
|----------|-------------------|------------|-------------|------------|
| Linear | × | 15.83 | 0.292 | 54.12 |
| Linear | √ | **11.04** | **0.298** | **44.53** |
| Logarithmic | √ | 13.67 | 0.294 | **38.22** |

> ⚠️ 不分离 trigger token 会导致严重性能退化；一旦分离，logarithmic quantizer 在低比特下表现更好

##### Component Synergy（Table V）
| Check-in | TAQ | Intent Score | FID (8/4) |
|----------|-----|----------------|------------|
| × | × | 3.22 | 54.12 |
| √ | × | 4.40 | 51.31 |
| × | √ | 3.23 | 39.53 |
| √ | √ | **4.41** | **38.22** |

> 🔗 二者互补：Check-in 提升 intent alignment，TAQ 保障 fidelity under quantization，联合使用最鲁棒

---

## 4. 关键结论和发现

### 主要发现
1. **Trigger token 是连接 selection 与 quantization 的关键桥梁**
   - 它既是个性化语义的锚点（selection），也是最脆弱的信息通路（quantization）
   - 利用同一信号指导两个阶段的设计，实现了系统级协同优化

2. **Intent-aligned selection 必须结合上下文推理**
   - 单纯基于文本相似性的 retrieval 容易失败
   - 引入 metadata（如 created_at、version）、style tag 和 LLM 推理可显著提升准确性
   - 主动 clarification 可有效解决歧义，且无需暴露 checkpoint ID

3. **Standard PTQ 会严重破坏 personalized concepts**
   - 实验证明 trigger token 的 K/V 行对量化极度敏感（见 Fig. 2）
   - 传统均匀量化策略不可行，必须采用 **pathway-aware** 的混合精度方案

4. **TAQ 实现高效与保真的平衡**
   - 在 W8A8 设置下接近 full precision 质量
   - 在 W8A4 下仍保持可用性，同时减少 **4–8× GPU memory** 和 **16–32× bit operations**

### 局限性
- 当前框架假设 trigger token 已知且已绑定至特定 checkpoint，不涉及 trigger discovery 或 binding 学习
- 所有实验基于 API 调用 MLLM（如 Gemini），本地部署大模型仍面临资源挑战
- 澄清机制依赖对话接口，在单轮交互场景中受限

### 未来工作方向
- 将 Check-in 与 TAQ 扩展至 video generation 或 multi-modal agents
- 探索动态 trigger token 学习与管理机制
- 开发轻量化 MLLM 适配器以支持边缘设备部署
- 构建更大规模、跨用户共享的 personalized model marketplace

---

> ✅ **总结一句话**：  
> **PersonalQ 通过 trigger token 统一了 personalized diffusion model 的 selection 与 quantization，首次实现了高意图对齐、高保真、低资源消耗的端到端服务框架，为大规模个性化生成系统的落地提供了可行路径。**

</details>

---

### 2. [Balancing Safety and Efficiency in Aircraft Health Diagnosis: A Task Decomposition Framework with Heterogeneous Long-Micro Scale Cascading and Knowledge Distillation-based Interpretability](https://arxiv.org/abs/2603.22885)

**Authors**: Xinhang Chen, Zhihuan Wei, Yang Hu, Zhiguo Zeng, Kang Zeng, Suili Yang  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.22885v1  

#### Abstract
Whole-aircraft diagnosis for general aviation faces threefold challenges: data uncertainty, task heterogeneity, and computational inefficiency. Existing end-to-end approaches uniformly model health discrimination and fault characterization, overlooking intrinsic receptive field conflicts between glo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对通用航空飞机健康诊断（Aircraft Health Diagnosis）中的三大核心挑战：
- **数据不确定性**（Data Uncertainty）：真实飞行数据存在传感器噪声、标签模糊等问题；
- **任务异质性**（Task Heterogeneity）：Anomaly Detection（AD）与 Fault Classification（FC）对模型的感受野（receptive field）需求冲突；
- **计算效率瓶颈**（Computational Inefficiency）：端到端（end-to-end）训练在类别极度不平衡的数据上成本高昂。

传统 end-to-end 方法将 AD 和 FC 统一建模，导致全局上下文建模与局部特征提取之间的内在矛盾，且缺乏可解释性。

---

### 提出的新方法与思路
作者提出 **Diagnosis Decomposition Framework (DDF)** 及其具体实现 **Long-Micro Scale Diagnostician (LMSD)**，核心思想是“**任务解耦 + 异构级联**”：

#### （1）**任务分解（Task Decomposition）**
- 显式地将诊断任务分解为两个阶段：
  - **Long Stage (AD)**：使用全局感受野进行异常检测，判断是否偏离正常操作包络；
  - **Micro Stage (FC)**：仅对被判定为异常的样本进行细粒度故障分类，聚焦局部微小特征。
- 架构上采用 **Hard-Threshold Routing** 机制，确保健康样本不进入复杂 FC 模型。

#### （2）**异构长-微尺度级联（Heterogeneous Long-Micro Scale Cascading）**
- **AD 阶段**：采用 **ConvTokMHSA**（Convolutional Tokenizer + Multi-Head Self-Attention），具备全序列感受野，适合捕捉跨时段的操作模式；
- **FC 阶段**：采用 **MMK Net**（Multi-Micro Kernel Network），使用小卷积核（1, 3, 5）提取局部敏感特征，避免引入全局噪声。

#### （3）**基于知识蒸馏的可解释性（Interpretability-by-Design）**
- 引入 **Keyness Extraction Layer (KEL)**，通过知识蒸馏从教师模型中学习输入时间序列上的注意力分布；
- 输出 **Temporal Keyness Vector**，提供物理可追溯的决策依据，区分“操作记忆”与“本质故障特征”。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **任务适应性** | 解决了 end-to-end 模型中 AD 与 FC 的感受野悖论，通过架构分离实现最优子任务适配 |
| **可解释性** | KEL 提供两阶段独立的注意力可视化，支持“为什么”和“在哪”的解释，优于黑箱模型或后验解释方法（如 SHAP/LIME） |
| **计算效率** | 解耦训练策略显著降低训练开销：“大样本轻量模型”用于 AD，“小样本复杂模型”用于 FC，避免无效计算 |
| **安全性** | Hard-threshold 路由机制最小化漏检率（FNR），符合航空领域“宁可误报不可漏检”的安全伦理 |

---

## 2. 核心实验方法和设置

### 数据集
- **NGAFID dataset**：来自塞斯纳 172 机队的真实航空维护数据集。
  - 包含超过 28,935 次飞行（约 31,000 小时）
  - 23 维传感器时间序列（采样频率 1Hz）
  - 关联 36 类非计划性维修事件
  - 具有严重类别不平衡（头类 >2000 次，尾类 <15 次）

分为两个子集：
- **Subset**：19 类，11,446 次飞行（基准验证）
- **Overall**：36 类，28,935 次飞行（完整高复杂场景）

---

### 实验设置
- **预处理**：
  - 缺失值前向填充（Forward Fill）
  - 长度归一化至 2048（Cubic Spline 插值）
  - Z-score 归一化（基于训练集参数）
- **交叉验证**：Stratified 5-Fold Cross-Validation，各 fold 物理隔离
- **随机控制**：固定划分种子，每 fold 独立运行 3 轮取中位数，最终报告五折中位数的中位数（Median-of-Medians）

---

### 评估指标体系
| 类别 | 指标 | 说明 |
|------|------|------|
| **传统分类性能** | ACC, F1, WF1 | 基础判别能力 |
| **安全相关指标** | FNR（False Negative Rate） | 故障误判为健康的比率，越低越好 |
| | **MCWPM**（Multi-Class Weighted Penalty Metric） | 新提出的综合安全指标：<br>$$ \text{MCWPM} = \frac{2TP}{2TP + \alpha_p FN_{\text{health}} + \beta_p FP_{\text{health}}} $$<br>其中 $\alpha_p=2.5$, $\beta_p=1.0$，强调对漏检的惩罚远高于误报 |
| **训练效率** | ET（Epoch Time）、TTT（Total Training Time） | 衡量训练可行性 |
| **推理开销** | IT32（Inference Time for 32 Samples）、MSize（Model Size） | 部署经济性指标 |

---

### 基线方法对比
| 模型 | 类型 | 特点 |
|------|------|------|
| **Bi-LSTM** | RNN | 捕捉双向时序依赖 |
| **InceptionTime** | CNN | 多尺度卷积核提取局部特征 |
| **InceptionTimeAttn** | CNN-Transformer Hybrid | 局部卷积 + 全局自注意力 |
| **ConvTokMHSA / ConvTokSWLA** | Transformer 变体 | 分别使用全局注意力与滑动窗口局部注意力 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Overall 数据集）

| 模型 | MCWPM | F1 | TTT (s) | MSize (MB) |
|------|--------|-----|---------|-----------|
| Bi-LSTM | 0.3372 | 0.0247 | 407.06 | 3.92 |
| InceptionTime | 0.5652 | 0.3570 | 7052.59 | 76.25 |
| MMK Net | 0.5417 | 0.3188 | 3024.93 | 11.43 |
| ConvTokMHSA | 0.5306 | 0.2754 | 942.26 | 32.29 |
| **LMSD (本文)** | **0.6148** | **0.4091** | **2001.63** | **12.97** |

> ✅ **LMSD 在 MCWPM 上比最佳基线提升约 4–8%**

---

### 与基线方法的对比结果
- **在 AD 子任务中**：
  - ConvTokMHSA（全局注意力）表现最优（F1 ≈ 0.76），而 ConvTokSWLA（局部注意力）性能下降明显；
  - 证明 AD 需要全局上下文建模。

- **在 FC 子任务中**：
  - MMK Net 显著优于其他模型（Subset F1 达 0.6228），而 ConvTokMHSA 因引入全局噪声性能崩溃（F1 ≈ 0.2080）；
  - 证明 FC 必须限制感受野以抑制跨阶段噪声。

- **在完整 Diagnosis 任务中**：
  - 所有 end-to-end 模型均面临性能瓶颈，无法兼顾 AD 与 FC；
  - LMSD 通过异构级联突破该瓶颈，在保持较低模型大小的同时实现更高安全指标（MCWPM）。

---

### 消融实验与分析（隐含于设计逻辑）
虽然未明确列出消融表，但从以下对比可视为事实上的消融研究：
- **是否解耦**：LMSD vs. 单一模型 → 显著提升 MCWPM 与训练效率
- **感受野设计**：ConvTokMHSA vs. ConvTokSWLA 在 AD/FC 中表现相反 → 验证“长-微”尺度分离必要性
- **路由机制**：Hard-threshold 路由有效防止健康样本污染 FC 空间，保障安全性

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **任务异质性要求架构解耦**：  
   AD 与 FC 对感受野的需求根本冲突，单一架构无法同时优化二者，必须通过 **architectural separation** 实现最优适配。

2. ✅ **异构级联优于统一建模**：  
   “Long-Micro” 级联策略在任务适应性、安全性、效率三方面全面超越 end-to-end 方法。

3. ✅ **可解释性可以内生于设计**：  
   KEL 成功提取出具有物理意义的时间注意力分布，例如：
   - AD 阶段关注起飞准备阶段（设备应力突变期）；
   - FC 阶段定位特定故障对应的热力学参数同步波动等微观异常。

4. ✅ **计算效率显著提升**：  
   解耦训练使总训练时间减少近 70%（相比 InceptionTimeAttn），且模型更小，更适合部署。

---

### 方法的局限性
1. **数据质量天花板效应**：
   - NGAFID 数据本身存在标签噪声（如预防性维护 vs. 实际故障）、极端长尾分布；
   - 即便模型设计先进，尾类召回率仍受限于样本稀缺（<30 样本类别召回 <40%）。

2. **传感器维度有限**：
   - 当前仅 23 维 1Hz 数据，关键故障信号可能淹没在低方差子空间（PCA 显示第12–18主成分才含判别信息）；
   - 算法难以克服物理信息瓶颈。

3. **硬阈值路由缺乏灵活性**：
   - 当前采用 hard-threshold 决策，未考虑不确定性量化；
   - 可能错放边界样本，未来可引入 soft routing 或贝叶斯决策。

---

### 未来工作方向
1. **方法学层面**：
   - 发展从 **hard-threshold 到 uncertainty-aware soft routing** 的动态融合机制；
   - 引入 **physics-informed embeddings** 支持因果推理。

2. **数据层面**：
   - 提升飞行参数采集密度（更高维、更高频）；
   - 推进“组件-子系统-整机”多层级数据采集与标注；
   - 改进维修语义标注精度。

3. **工程落地**：
   - 适配边缘计算环境，支持在线增量学习；
   - 构建跨机型迁移能力，推动通用航空 PHM 标准化。

---

> 🔚 **总结**：  
> 本文提出的 DDF 框架不仅是一个高性能的航空健康诊断方案，更是一种面向复杂工业系统的 **“可信赖 AI 设计范式”** ——通过任务解耦、异构建模、内生可解释性，实现了 **安全性、效率与可信度的平衡**，为未来智能 PHM 系统提供了可复用的方法论路径。

</details>

---

### 3. [Continuous Optimization for Satisfiability Modulo Theories on Linear Real Arithmetic](https://arxiv.org/abs/2603.22877)

**Authors**: Yunuo Cen, Daniel Ebler, Xuanyao Fong  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.22877v1  

#### Abstract
Efficient solutions for satisfiability modulo theories (SMT) are integral in industrial applications such as hardware verification and design automation. Existing approaches are predominantly based on conflict-driven clause learning, which is structurally difficult to parallelize and therefore scale...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Continuous Optimization for Satisfiability Modulo Theories on Linear Real Arithmetic

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对 **Satisfiability Modulo Theories (SMT)** 中的 **Linear Real Arithmetic (LRA)** 问题，解决传统基于 **Conflict-Driven Clause Learning (CDCL)** 的求解器在处理大规模、高复杂度约束时存在的两大瓶颈：
- **难以并行化**：CDCL(T) 本质上是串行搜索，无法有效利用现代多核硬件（如 GPU）；
- **可扩展性差**：随着变量和约束数量增加，运行时间急剧上升，甚至无法求解。

### 提出了什么新方法或新思路
作者提出了一种全新的连续优化框架 **FOURIERSMT**，其核心思想是将离散的 SMT(LRA) 问题转化为一个**可微分的连续优化问题**，从而支持高效的梯度下降求解。主要技术路径如下：

- **Extended Walsh-Fourier Expansion (xWFE)**  
  将传统的 Walsh-Fourier Expansion 从纯布尔域推广到**混合布尔-实数域**，使得 SMT 公式中的约束可以表示为**分段多元线性多项式**，从而允许使用梯度方法进行局部更新。

- **Extended Binary Decision Diagram (xBDD)**  
  为避免 xWFE 项数随变量指数增长的问题，引入 xBDD 数据结构，将约束编码为概率电路，并证明其 **Circuit-Output Probability (COP)** 与 xWFE 的期望值等价，从而实现高效计算。

- **随机舍入与高斯采样 (Randomized Rounding & Gaussian Sampling)**  
  对布尔变量采用随机舍入，对实数变量采用高斯采样，构建平滑的代理目标函数，使其可微且适合梯度优化。

- **退火策略 (Annealing Strategy)**  
  在优化过程中逐渐减小采样方差（即“温度”），使优化过程先进行全局探索，再逐步聚焦于精确解附近，保证收敛性和最优性。

### 相比现有方法的优势
- **高度并行化**：整个优化流程天然适配 GPU 加速，显著提升大规模问题的求解效率；
- **更强的可扩展性**：能够处理高达 **10,000 变量、700,000 约束**的大规模实例，而传统 CDCL(T) 求解器在此类规模下基本失效；
- **理论保障**：通过定理证明了方法的**完备性（soundness）** 和**收敛性**，确保最终解满足原始 SMT 公式。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在两类问题上进行验证：

1. **随机混合约束基准 (Random Hybrid Constraints)**  
   - 包含 `cardinality`、`not-all-equal (nae)`、`parity (xor)` 等混合布尔-实数约束；
   - 变量规模 $ n \in \{100, 200, ..., 1000\} $，每个规模生成 10 个实例；
   - 引入 $ n $ 个 LRA 原子，以测试实数变量处理能力。

2. **组合优化问题**
   - **调度问题 (Scheduling)**：连续时间任务调度，涉及非重叠、依赖关系、可行性等约束；
   - **布局问题 (Placement)**：3D 芯片模块布局，包含非重叠、布线感知、层间连接等约束；
   - 总共 380 个实例，涵盖不同规模和难度。

### 实验设置和评估指标
- **硬件配置**：
  - 基线求解器运行于 **AMD EPYC 9654 CPU**（最多 64 线程）；
  - FOURIERSMT 运行于 **NVIDIA L40S GPU**；
- **超时限制**：所有实例均设为 **1000 秒**；
- **评估指标**：
  - **PAR-2 Score**：平均运行时间，超时按 2000 秒计，综合衡量速度与鲁棒性；
  - 求解实例数（Number of Solved Instances）；
  - 梯度计算时间（用于分析 GPU 加速效果）。

### 基线方法对比
对比了当前主流的 SMT 求解器：
- **Z3**, **Z3++**, **CVC5**, **YICES2**, **SMTS**, **MATHSAT5**, **SMT-RAT**

同时定义两个虚拟最佳求解器：
- **VBS1**：所有求解器（含 FOURIERSMT）中表现最好的；
- **VBS2**：排除 FOURIERSMT 后的最佳组合，用于量化其带来的增益。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **最大规模测试**：成功求解 **10,000 变量、700,000 约束** 的实例；
- **加速比**：相比现有最先进求解器，**最高达 8 倍加速**；
- **求解能力**：
  - 在随机混合约束上，**FOURIERSMT 成功求解全部 100 个实例**；
  - 所有 CDCL(T) 基线求解器在 $ n \geq 500 $ 时均无法完成求解。

### 与基线方法的对比结果
#### （1）随机混合约束
- 当 $ n = 100 $ 时，FOURIERSMT 稍慢于部分 CDCL 求解器；
- 当 $ n = 500 $ 时，**实现约 100× 速度提升**；
- **VBS1 比 VBS2 快 62.45×，多求解 50 个实例**，表明 FOURIERSMT 是性能跃升的关键。

#### （2）调度问题
- **VBS1 比 VBS2 快 8.18×，多求解 20 个实例**；
- 在最大规模实例上，其他求解器均超时，**FOURIERSMT 全部求解成功**；
- 相比最强基线 SMTS，**快 6.34×**；相比 Z3，**快 16.3×**。

#### （3）布局问题
- **VBS1 比 VBS2 快 4.93×，多求解 10 个实例**；
- 在最大 30 个实例上，其他求解器全部失败；
- 相比 Z3，**快 3.15×**；相比 SMT-RAT，**快 36.0×**。

### 消融实验结果
- **GPU vs CPU 梯度计算**：
  - 小规模实例：CPU 更快（因 GPU 固定开销大）；
  - 大规模实例：**GPU 实现 6.67× 加速**，且运行时间趋于稳定，体现良好并行扩展性；
- **退火策略有效性**：
  - 无退火时收敛缓慢或陷入局部最优；
  - 采用退火后，能快速穿越平坦区域，最终收敛至边界解（即合法布尔赋值）。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **连续优化可用于 SMT 求解**：首次成功将 **CLS (Continuous Local Search)** 框架扩展到 SMT(LRA)，打破了其仅限于 SAT 或混合布尔约束的传统；
2. **GPU 加速潜力巨大**：FOURIERSMT 充分利用 GPU 并行架构，在大规模问题上展现出远超 CPU 的性能优势；
3. **可扩展性突破**：解决了传统 CDCL(T) 求解器在大规模工业场景下的“瓶颈”，为实际应用提供新路径；
4. **理论与实践结合**：不仅提出新算法，还通过严格数学证明（如 Theorem 1, 2）保证 soundness 与 convergence。

### 方法的局限性
- **不完全性 (Incomplete)**：作为局部搜索方法，不能保证在有限时间内找到解（即使存在），因此返回 “UNKNOWN” 是可能的；
- **仅支持 LRA**：目前框架局限于线性实数算术，尚未支持非线性（NRA）、位向量（BITVECTORS）等更复杂的理论；
- **初始化敏感**：性能受初始点影响，需多次运行以提高成功率。

### 未来工作方向
- **扩展至非线性算术 (NRA)**：通过更复杂的平滑技术（如高阶矩匹配）处理非凸可行域；
- **支持更多理论**：集成数组（ARRAYS）、位向量等，构建通用 SMT 求解框架；
- **结合符号推理**：将 CLS 与 CDCL(T) 结合，形成 hybrid 架构，兼顾完备性与效率；
- **自动调参与学习策略**：引入强化学习或元学习，自适应调整退火策略、权重更新机制等超参数。

--- 

> **总结一句话**：  
> FOURIERSMT 首次实现了基于连续优化的 SMT(LRA) 求解器，通过 xWFE + xBDD + 梯度下降 + 退火策略，实现了**高度并行化、强可扩展性与 GPU 加速**，在大规模调度与布局问题上相较 state-of-the-art 求解器取得 **8 倍以上加速**，为工业级 SMT 应用开辟了新路径。

</details>

---

### 4. [EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction](https://arxiv.org/abs/2603.22910)

**Authors**: Yixuan Wang, Shiyu Ji, Yijun Liu, Qingfu Zhu, Wanxiang Che  
**Category**: cs.CL  
**Published**: 2026-03-25  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.22910v1  

#### Abstract
The increasing memory demand of the Key-Value (KV) cache poses a significant bottleneck for Large Language Models (LLMs) in long-context applications. Existing low-rank compression methods often rely on irreversible parameter transformations, sacrificing the flexibility to switch back to full-precis...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
大型语言模型（LLMs）在长上下文推理中面临 **Key-Value (KV) cache 内存占用过大** 的瓶颈。传统低秩压缩方法（如 SVD-based）虽然能减少内存，但通常依赖不可逆的参数变换，导致无法在内存充足时切换回全精度推理模式，牺牲了灵活性。

此外，现有在线压缩方法在高压缩比下性能下降明显，且引入额外延迟。

### **提出了什么新方法或新思路**
本文提出 **EchoKV**，一种灵活、高效的 KV cache 压缩框架，其核心思想是：

- **不进行显式压缩-解压流程**，而是通过一个轻量级网络（lightweight network），利用注意力头之间的 **inter-layer 和 intra-layer 相似性**，从部分保留的 KV cache 中 **重建被丢弃的部分**。
- 允许在运行时 **按需切换**：内存充足时使用 Full KV 推理，内存受限时启用 EchoKV 进行压缩与重建。

具体实现方式：
- 输入特征由两部分组成：
  - **Global Cache Input**：来自组内首层的完整 KV cache，用于捕捉跨层相似性。
  - **Local Cache Input**：当前层部分注意力头的 KV cache，用于保留局部细节。
- 使用简单的线性层作为预测网络，保证推理效率。
- 提出 **EchoKV-Hybrid**：对 Key 和 Value 采用异构压缩策略——Key 使用低秩方法（如 ThinK），Value 使用 Echo 重建，进一步提升性能。

### **相比现有方法的优势**
| 维度 | EchoKV 优势 |
|------|-------------|
| **灵活性** | 支持 Full KV ↔ Compressed KV 的动态切换，适应不同内存场景。 |
| **性能保持** | 在高/低压缩比下均优于主流方法（如 Palu、CommonKV），接近无损。 |
| **训练成本低** | 两阶段微调策略，仅需约 **1 A100 GPU 小时** 即可完成 7B 模型训练。 |
| **兼容性强** | 可与量化、缓存淘汰等技术正交结合，实现更高压缩率。 |
| **吞吐量高** | 在短序列上维持 Full KV 的高吞吐，在长序列避免 OOM。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **LongBench** (Bai et al., 2024)：真实世界多任务长上下文基准，涵盖问答、摘要、代码生成等。
- **RULER** (Hsieh et al., 2024)：合成型长上下文基准，特别设计用于测试“针在 haystack”（Needle In A Haystack, NIAH）等检索能力。

### **实验设置和评估指标**
- **模型**：
  - Llama3.1-8B-Instruct
  - Mistral-7B-Instruct-v0.3
- **压缩比定义**：`Compressed Cache Size / Full Cache Size`
- **测试压缩比**：0.7、0.5、0.3
- **输入长度**：
  - LongBench：使用模型最大上下文长度
  - RULER：固定为 32K
- **硬件平台**：单张 NVIDIA A100-SXM4-80GB GPU，batch size=8
- **评估指标**：
  - 各任务平均得分（如 LongBench Avg）
  - NIAH 任务中的 needle retrieval 准确率
  - 吞吐量（throughput）随输入长度变化趋势

### **基线方法对比**
- **Palu** (Chang et al., 2024)：基于组内 SVD 的低秩压缩。
- **CommonKV** (Wang et al., 2025c)：跨层共享低秩参数。
- **ThinK** (Xu et al., 2024)：查询驱动的 Key 剪枝。
- **MiniCache** (Liu et al., 2024b)：后压缩合并技术。
- **Full KV**：无压缩，作为上限参考。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **LongBench 结果（Llama3.1-8B-Instruct, 压缩比=0.5）**
| Method | LongBench Avg |
|--------|----------------|
| Full KV | 54.75 |
| Palu | 47.47 |
| CommonKV | 46.27 |
| **EchoKV** | **48.53** |
| **EchoKV-Hybrid** | **49.40** |

> ✅ **EchoKV 超越所有基线，接近 Full KV 性能（损失 <6%）**

#### **RULER 结果（压缩比=0.5, NIAH 平均准确率）**
| Method | Llama3.1-8B | Mistral-7B |
|--------|--------------|------------|
| Full KV | 87.63 | 82.74 |
| Palu | 68.14 | 68.56 |
| CommonKV | 69.68 | 36.88 |
| **EchoKV** | **83.52** | **69.13** |
| **EchoKV-Hybrid** | **86.45** | **79.89** |

> ✅ **EchoKV 在合成任务中表现尤为突出，显著优于 SVD 类方法**

#### **极端压缩比（0.3）下的可用性**
- 在 `ratio=0.3` 下，Palu 和 CommonKV 性能严重退化（LongBench Avg < 35），而 **EchoKV 仍保持 45+ 分**，具备基本可用性。

---

### **与基线方法的对比结果**
- **在所有压缩比下，EchoKV 均优于 Palu 和 CommonKV**，尤其在 Mistral 模型上优势更明显（见 Figure 5 可视化）。
- **EchoKV-Hybrid 进一步提升性能**，验证了 Key/Value 异构处理的有效性。
- **MiniCache 和 ThinK 在高压缩比下性能骤降**，说明纯统计剪枝难以应对复杂语义。

---

### **消融实验结果**

#### **(1) 输入特征消融（Table 4）**
| 输入配置 | LongBench Avg |
|---------|----------------|
| Only Local | 45.27 |
| Only Global | 45.02 |
| **Combined (Local + Global)** | **45.74** |

> ✅ **组合输入效果最佳**，说明局部细节与全局结构信息互补。

#### **(2) 损失函数分析（Table 3）**
| Loss Function | Stage | LongBench Avg |
|--------------|-------|----------------|
| KV-MSE | I | 48.99 |
| O-MSE | II | **49.26** |
| QK-KL | II | 49.11 |

> ✅ **O-MSE 损失在性能相当的前提下，训练速度比 QK-KL 快近 3 倍**，且兼容 FlashAttention。

#### **(3) Key vs Value 预测难度分析（Figure 3a）**
- 发现 **Value 更容易通过相似性重建**，而 Key 预测误差更大。
- 支持了 Hybrid 设计：**Key 适合低秩压缩，Value 适合 Echo 重建**。

#### **(4) 训练数据鲁棒性（Figure 3b）**
- 在 LongAlpaca、Alpaca、ShareGPT、C4 上训练的轻量网络，性能差异极小。
> ✅ **方法对训练数据不敏感，具有强泛化性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **KV cache 存在显著的跨层与跨头相似性**，可用于高效重建而非直接压缩。
2. **显式压缩-解压范式非必需**，通过轻量网络直接预测残差组件更灵活高效。
3. **Key 和 Value 应区别对待**：Key 更低秩，适合低秩压缩；Value 更相似，适合邻域重建。
4. **EchoKV 实现了真正的“按需压缩”**：短文本保持高性能，长文本避免 OOM。
5. **训练成本极低**（<1 A100 小时），且无需复杂超参搜索。

### **方法的局限性**
- 当前对局部缓存的选择采用启发式策略（取前 m 个 head），未考虑 head 间重要性差异。
- 对于极度稀疏或非结构化的 KV 分布，重建可能失效。
- Hybrid 方法仍较粗粒度，尚未深入建模 K/V 的分布差异。

### **未来工作方向**
- 设计更精细的 head 选择机制（如基于离线分析选取代表性 head）。
- 深入研究 Key 和 Value 的数值分布特性，开发针对性更强的压缩算法。
- 探索与其他技术（如 quantization、eviction）的联合优化。
- 扩展至多模态 LLM 场景。

---

> 🔚 **总结**：EchoKV 提出了一种新颖、灵活且高效的 KV cache 压缩范式，通过 **相似性驱动的重建机制** 替代传统压缩，实现了性能、灵活性与效率的统一，在长上下文 LLM 推理中展现出巨大应用潜力。

</details>

---

### 5. [PCR: A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving](https://arxiv.org/abs/2603.23049)

**Authors**: Wenfeng Wang, Xiaofeng Hou, Peng Tang, Hengyi Zhou, Jing Wang, Xinkai Wang, Chao Li, Minyi Guo  
**Category**: cs.DC  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.23049v1  

#### Abstract
Retrieval-Augmented Generation (RAG) systems enhance the performance of large language models (LLMs) by incorporating supplementary retrieved documents, enabling more accurate and context-aware responses. However, integrating these external documents often results in very long input sequences, which...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《PCR: A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **Retrieval-Augmented Generation (RAG)** 系统中，通过检索外部文档增强输入上下文，虽然提升了 LLM 回答的准确性和信息量，但也导致输入序列极长，显著增加了 **prefill 阶段** 的计算开销，进而造成 **TTFT (Time to First Token)** 显著上升。这在高吞吐场景下成为严重瓶颈。

尽管已有研究利用 **KV-cache reuse** 来避免重复计算共享前缀的 KV 缓存，但在实际部署中仍面临三大挑战：
- **低缓存命中率**：传统 LRU 替换策略未考虑未来请求，导致频繁淘汰即将重用的块。
- **CPU-GPU 数据传输开销大**：加载和卸载 KV-cache 引入通信延迟。
- **SSD I/O 性能差**：当缓存溢出到 SSD 时，读取速度慢，难以隐藏延迟。

### 提出的新方法与创新点
为解决上述问题，作者提出 **PCR (Prefetch-Enhanced Cache Reuse)** 系统，其核心创新包括以下三项关键技术：

#### （1）**Prefix-Tree Caching + Look-ahead LRU 替换策略**
- 将输入分块并组织成 **prefix tree** 结构，支持高效前缀匹配。
- 提出 **look-ahead LRU** 策略：利用调度队列中的待处理请求信息，提前提升即将被复用的缓存块优先级，减少误淘汰，提高 **cache hit ratio**。

#### （2）**Layer-wise Overlapping**
- 利用 LLM 层状结构特性，在三个独立的 CUDA stream 上并行执行：
  - GPU computation（计算）
  - CPU→GPU KV-cache loading（加载）
  - GPU→CPU KV-cache offloading（卸载）
- 实现层间流水线重叠，有效 **隐藏通信延迟**，仅保留首层加载和末层卸载的开销。

#### （3）**Queue-based Prefetching**
- 在请求仍在等待队列时，就由专用线程从 SSD 异步预取对应的 KV-cache 到 DRAM。
- 当请求真正执行时，可直接从 DRAM 加载，避免昂贵的实时 SSD 读取。
- 配合异步写回机制，实现全链路非阻塞数据移动。

### 相比现有方法的优势
| 方面 | PCR | 现有方法（如 vLLM, LMCache） |
|------|-----|-----------------------------|
| 缓存管理 | 前瞻性替换 + 树结构匹配 | 被动 LRU / Block-level 缓存 |
| 数据传输 | 层级重叠 + 异步预取 | 同步加载/卸载，易阻塞 GPU |
| 存储层级 | 支持 GPU + DRAM + SSD 三级缓存协同 | 多局限于 GPU 或 CPU 内存 |
| 准确性保障 | 完全保留原始 KV-cache，无精度损失 | 部分方法（如 CacheBlend）牺牲准确性换取效率 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **文档库**：Wikipedia dataset（约 859 万英文文档）
- **查询集**：SQuAD dataset
- **嵌入模型**：MiniLM
- **检索方式**：每条查询返回最相关的两个文档
- 构建两个 workload：
  - Workload 1：1,000 输入，平均长度 ~6.8k tokens，KV-cache 复用率 40%
  - Workload 2：2,000 输入，复用率 35%

### 实验设置
- **硬件平台**：
  - 平台1：2× NVIDIA A6000（48GB HBM），256GB DRAM，4TB NVMe SSD
  - 平台2：2× RTX 4090（24GB HBM），128GB DRAM，相同 SSD
- **PCIe 带宽**：双向约 24 GB/s（实测）
- **SSD 读写速度**：读 ~3 GB/s，写 ~500 MB/s
- **模型**：
  - Llama 系列：Llama2-7B, Llama2-13B, Llama3.1-8B, Llama3.2-3B
  - Qwen 系列：Qwen2.5-7B, Qwen2.5-14B
- **输出长度**：固定为 16 tokens（聚焦 prefill 阶段性能）

### 评估指标
- **TTFT (Time to First Token)**：主要指标
- **E2EL (End-to-End Latency)**
- **P50/P95/P99** 尾部延迟
- **Throughput**
- 请求到达服从 **Poisson 过程**，控制不同 arrival rate（0.5 ~ 1.0 req/s）

### 基线方法对比
1. **vLLM**：基于 PagedAttention 的主流 LLM 推理系统，支持 GPU 内部 KV-cache 复用
2. **LMCache**：构建于 vLLM 上的先进 KV-cache 复用系统，支持跨 GPU-CPU-SSD 的缓存与预取
3. **CCache**：本文作者构建的简化版，仅扩展至 CPU 内存
4. **SCCache**：CCache 的进一步扩展，加入 SSD 支持

所有方法均基于 vLLM 实现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- PCR 在多种模型、硬件和负载条件下均取得最优表现。
- 在 **Llama-8B + RTX 4090** 上：
  - 相比 vLLM，TTFT 最高提速达 **2.47×**
  - 绝对延迟降低最高达 **797 ms**
- 在高负载下优势更明显，表明系统具有良好可扩展性。

### 与基线方法的对比结果
| 模型 | 对比对象 | 平均 TTFT 降低幅度 |
|------|----------|--------------------|
| Llama2-7B | SCCache | **36.4%** |
| Llama2-13B | SCCache | **50.9%** |
| Qwen2.5-7B | SCCache | **3.9%** |
| Qwen2.5-14B | SCCache | **14.2%** |

> 注：Llama 系列收益更高，因其 KV-cache 更大，SSD 预取带来的增益更显著。

此外：
- PCR 在 **尾部延迟（P95/P99）** 上也全面领先：
  - 在请求速率为 0.9 req/s 时，PCR 的 E2EL P99 仅为 **86ms**，而 LMCache 和 vLLM 分别为 124ms 和 142ms（降幅超 30%）
- 所有指标随负载增长呈现平滑单调上升趋势，无抖动或饱和现象，体现系统稳定性。

### 消融实验结果
#### （1）Layer-wise Overlapping 贡献
- 单独启用该技术即可带来显著收益，尤其对 KV-cache 较大的模型（如 Llama2-13B）：
  - 在 1.0 req/s 下，TTFT 降低 **37.28%**
- “Only Down”（仅重叠卸载）效果优于“Up+Down”，说明 **offloading 开销是主要瓶颈**

#### （2）Queue-based Prefetching 效果
- 预取窗口大小（window size）影响性能：
  - 对 Llama2-7B，将窗口从 4 扩展到 6，TTFT 进一步下降 **31.06%**（高负载下）
- 高请求率下更多待处理请求可供预取，因此增益更大
- 推荐根据具体模型进行 profiling 以确定最佳 window size

#### （3）综合性能分解（见 Table 1）
| 技术组合 | Llama2-13B @1.0 req/s TTFT | 相比 Base 降低 |
|--------|----------------------------|---------------|
| Base (no optimization) | 487.427 s | — |
| + Overlap | 431.354 s | 11.50% |
| + Prefetch | 335.573 s | **31.15%** |

> 表明 **queue-based prefetching 是最大性能来源**，尤其是在大规模缓存场景下。

---

## 4. 关键结论和发现

### 主要发现
1. **KV-cache reuse 必须结合智能数据调度才能发挥最大效能**：
   - 单纯扩大存储（如加 SSD）可能因 I/O 慢反而劣于重新计算。
   - PCR 通过 **前瞻性替换 + 流水线重叠 + 异步预取** 共同作用，才真正释放多级缓存潜力。

2. **layer-wise overlapping 可有效掩盖 CPU-GPU 通信延迟**：
   - 得益于现代 GPU 高带宽和 LLM 层状结构，单层传输时间远小于计算时间，适合重叠。

3. **SSD 预取时机至关重要**：
   - 利用 **request queue 中的前瞻信息** 进行预取，可在 GPU 忙碌时完成数据准备，实现零等待加载。

4. **PCR 在平均和尾部延迟上均有显著提升**：
   - 不仅优化了平均用户体验，更重要的是改善了 P99 用户体验，这对生产环境意义重大。

### 方法的局限性
- **依赖调度器提供请求队列信息**：需要深度集成到推理框架内部（如 vLLM），通用性受限。
- **chunk size 和 window size 需调优**：不同模型和 workload 下需重新配置以达到最佳性能。
- **对极短输入或低复用率场景增益有限**：主要适用于 RAG 类长输入、高共享特征的场景。

### 未来工作方向
- 自适应调整预取窗口大小和 chunk 粒度，实现全自动优化。
- 扩展至多租户或多任务场景下的缓存隔离与资源分配。
- 探索压缩 + 复用联合设计，在不损精度前提下进一步降低存储成本。
- 将类似思想应用于 MoE 模型或其他具有结构性重复计算的 AI 工作负载。

---

> ✅ **总结一句话**：  
> PCR 通过 **prefix-tree 缓存结构 + layer-wise overlapping + queue-aware prefetching** 三位一体的设计，实现了高效的多级 KV-cache 复用，在不牺牲准确性的前提下，将 RAG 场景下的 TTFT 最高加速 **2.47×**，显著优于现有系统，为低延迟 RAG 推理提供了实用且高性能的解决方案。

</details>

---

### 6. [Cloud-Edge Collaborative Large Models for Robust Photovoltaic Power Forecasting](https://arxiv.org/abs/2603.22343)

**Authors**: Nan Qiao, Sijing Duan, Shuning Wang, Xingyuan Hua, Ju Ren  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.22343v1  

#### Abstract
Photovoltaic (PV) power forecasting in edge-enabled grids requires balancing forecasting accuracy, robustness under weather-driven distribution shifts, and strict latency constraints. Local specialized models are efficient for routine conditions but often degrade under rare ramp events and unseen we...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Cloud-Edge Collaborative Large Models for Robust Photovoltaic Power Forecasting*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对**边缘智能电网中的光伏（PV）功率预测**面临的三大挑战进行研究：

- **准确性与鲁棒性不足**：本地专用模型在常规天气下表现良好，但在罕见的快速功率爬坡事件（ramp events）或未知天气模式下性能急剧下降。
- **延迟敏感性**：云侧大型模型虽具备强大的泛化能力，但频繁调用会引入显著的通信延迟，不满足边缘设备对低延迟决策支持的需求。
- **资源与成本权衡**：完全依赖云端计算导致高带宽消耗和中心化资源瓶颈，而纯边缘方案又无法处理复杂场景。

### **提出了什么新方法或新思路**

作者提出了一种**风险感知的云-边协同框架（risk-aware cloud-edge collaborative framework）**，其核心思想是“按需协作”：

- **三分支架构设计**：
  - **Expert-only 分支**：站点专用的轻量专家模型，用于日常高效推理。
  - **Edge-assisted 分支**：边缘端的小型因果模型，增强本地推理能力。
  - **Cloud-assisted 分支**：云侧大型检索模型，通过 retrieval-prediction 流程提供历史相似案例作为上下文支持。

- **动态路由机制**：
  - 引入一个**轻量级筛选模块（screening module）**，实时评估以下指标：
    - 预测不确定性（predictive uncertainty）
    - 分布偏移风险（out-of-distribution risk）
    - 天气突变强度（weather mutation intensity）
    - 模型间分歧（model disagreement）
  - 基于上述信号生成**路由评分（routing score）**，并通过 **Lyapunov-guided router** 决定是否将任务升级至边缘或云端。

- **自适应融合机制**：
  - 激活的分支输出通过**置信度感知融合（confidence-aware fusion）** 得到最终预测，权重由在线学习算法（entropic FTRL）动态调整。

### **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **系统效率** | 显著减少不必要的云调用，降低通信开销和延迟。 |
| **预测鲁棒性** | 在分布外（OOD）和极端天气条件下保持更高精度。 |
| **理论保障** | 提供了基于 Lyapunov 优化的长期性能保证，证明存在阈值型最优路由策略，并给出 $O(1/V)$ 的最优性差距和 $O(V)$ 的队列积压。 |
| **部署可行性** | 解耦了云侧检索与边缘预测，使资源受限的边缘设备也能利用大模型的知识。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **Shanxi Dataset**：
  - 来自山西电网，包含18个逆变器节点连续19天的数据。
  - 功率采样频率为5分钟，气象日志为10分钟。
  - 总样本数：46,983。

- **Hunan Dataset**：
  - 来自湖南电网，包含5个逆变通道的1分钟级遥测数据。
  - 包含站点级太阳辐射和气象协变量。
  - 总样本数：24,929。

> 数据预处理包括时间对齐、缺失值剔除、边界裁剪，并采用严格的时间顺序划分训练/验证/测试集，避免时间泄露。

### **实验设置和评估指标**

#### **评估维度**

| 类别 | 指标 | 说明 |
|------|------|------|
| **预测准确性** | `nMAE`, `nRMSE` | 归一化平均绝对误差与均方根误差（%FS）。越低越好。 |
| **路由质量** | `AUROC`, `AUPRC` | 路由评分区分“是否应调用云端”的能力。越高越好。仅适用于自适应路由方法。 |
| **鲁棒性** | `REE`（Ramp Event Error） | 快速功率变化期间的预测误差。越低越好。 |
| | `DG`（Degradation Ratio） | OOD 错误 / ID 错误，衡量分布偏移下的退化程度。越接近1越好。 |

#### **基线方法对比**

分为两类：

- **系统级基线（System Baselines）**：
  - **ExO (Expert-only)**：仅使用本地专家模型。
  - **EdO (Edge-only)**：固定使用边缘小模型融合。
  - **CO (Cloud-only)**：始终调用云辅助模型。
  - **ACA (Always-Cloud-Assisted)**：始终启用云分支。
  - **STR (Static-Threshold Routing)**：静态阈值决定是否上云。

- **先进模型基线（State-of-the-art Models）**：
  - **Moirai**, **AIRG**, **STKD-PV**：代表当前先进的时序预测与知识蒸馏方法。

#### **超参数设置**

- Lyapunov trade-off parameter $V = 80$
- 云请求预算 $p_{\text{max}} = 0.5$
- 平均延迟预算 $T_{\text{max}} = 120$ ms
- 检索支持集大小 $K = 8$
- 所有方法在验证集上调参，测试集报告单次运行结果（fixed-seed）

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自Table II）**

| 方法 | Hunan – nMAE ↓ | Hunan – REE ↓ | Hunan – DG ↓ | Hunan – AUROC ↑ | Shanxi – nMAE ↓ | Shanxi – DG ↓ |
|------|----------------|---------------|--------------|------------------|------------------|----------------|
| ExO | 8.54 | 25.41 | 3.50 | — | 7.85 | 1.89 |
| EdO | 6.21 | 18.52 | 2.10 | — | 6.92 | 1.58 |
| CO | 3.15 | 8.56 | 1.15 | — | **4.09** | 1.09 |
| ACA | 4.10 | 11.24 | 1.48 | 0.762 | 4.72 | 1.16 |
| STR | 4.02 | 11.05 | 1.45 | 0.755 | 4.68 | 1.15 |
| Moirai | 3.32 | 9.04 | 1.18 | — | 4.51 | 1.10 |
| Ours (**本文方法**) | **3.08** | **8.45** | **1.12** | **0.924** | 4.17 | **1.08** |

> ✅ 表示本方法为最佳；🟡 表示第二佳。

### **与基线方法的对比结果**

- **综合性能最优**：
  - 在 **Hunan 数据集上全面领先**，所有指标均为第一，尤其在 `REE` 和 `DG` 上大幅优于其他方法，表明其在极端天气和分布偏移下具有更强鲁棒性。
  - 在 **Shanxi 数据集上**，虽然 `nMAE` 略高于 CO，但 `DG = 1.08` 为最低，且 `AUROC = 0.931` 远超 STR（0.871），说明其能更精准识别困难样本并合理调度资源。

- **路由有效性显著提升**：
  - 本文方法的 `AUROC` 和 `AUPRC` 明显高于 STR，证明其基于多维风险感知的动态路由机制优于单一阈值判断。

- **优于纯大模型方法**：
  - 尽管 Moirai 等大模型在准确性上有竞争力，但缺乏对延迟、通信和云负载的控制机制，难以直接部署于边缘环境。本文方法通过选择性调用，在保持高性能的同时满足系统约束。

### **消融实验与敏感性分析（关键发现）**

- **路由分数的影响（Fig. 3）**：
  - 路由分数越高，越倾向于激活边缘或云分支，体现了“难样本才上云”的设计理念。

- **Lyapunov 参数 $V$ 的影响（Fig. 4 & 5）**：
  - 增大 $V$ 有助于提高 `AUROC` 并降低 `DG`，即增强系统对困难样本的识别能力和鲁棒性，但可能导致轻微延迟增加（符合理论预期）。

- **检索集大小 $K$ 的非单调效应（Fig. 6）**：
  - $K=8 \sim 12$ 时性能最佳，过小则上下文不足，过大则引入噪声，验证了“适度检索”的重要性。

- **云预算 $p_{\text{max}}$ 与延迟预算 $T_{\text{max}}$ 的影响（Fig. 7 & 8）**：
  - 放宽预算可提升性能，但收益逐渐饱和，说明系统可通过调参实现灵活权衡。

---

## 4. 关键结论和发现

### **主要发现**

1. **云-边协同是解决边缘光伏预测难题的有效范式**：通过“本地快推 + 按需求助”的方式，兼顾了速度、准确性和鲁棒性。
2. **检索增强的轻量预测可行且高效**：云侧大模型无需完整推理，只需执行检索任务提供历史上下文，即可显著提升边缘小模型的表现。
3. **多维风险感知路由优于单一指标判断**：结合不确定性、OOD、天气突变等信号的路由机制能更可靠地识别需要协助的困难样本。
4. **Lyapunov 控制提供了坚实的理论基础**：所提路由策略具有阈值结构，且能在长期满足延迟、通信和云负载约束的前提下逼近最优性能。

### **方法的局限性**

- **依赖高质量的历史案例库**：若云侧 `Dcld` 中缺乏与当前天气模式匹配的历史记录，检索效果将受限。
- **边缘设备仍需一定算力**：尽管模型轻量化，但仍需运行小型模型和融合逻辑，对极低端设备可能仍有压力。
- **路由决策依赖离线校准**：增益函数 $G_1(r), G_2(r)$ 需要在完整回放集上离线拟合，增加了部署复杂性。

### **未来工作方向**

- 探索**在线持续更新检索库**机制，使其能自适应新增天气模式。
- 研究**跨区域迁移学习**，进一步提升在数据稀缺新站点上的冷启动能力。
- 将该框架扩展至其他能源预测任务，如风电、负荷预测等。
- 结合**causal discovery**技术，进一步增强模型对物理规律的理解与解释性。

--- 

> **总结一句话**：  
> 本文提出了一种**条件自适应的云-边协同大模型框架**，通过**风险感知路由 + 检索增强预测 + Lyapunov 控制**，实现了在严格延迟约束下兼具高精度、强鲁棒性与系统效率的光伏功率预测，为边缘智能在新能源领域的落地提供了新思路。

</details>

---

### 7. [Improving Safety Alignment via Balanced Direct Preference Optimization](https://arxiv.org/abs/2603.22829)

**Authors**: Shiji Zhao, Mengyang Wang, Shukun Xiong, Fangzhou Chen, Qihui Zhu, Shouwei Ruan, Yisong Xiao, Ranjie Duan, Xun Chen, XingXing Wei  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.22829v1  

#### Abstract
With the rapid development and widespread application of Large Language Models (LLMs), their potential safety risks have attracted widespread attention. Reinforcement Learning from Human Feedback (RLHF) has been adopted to enhance the safety performance of LLMs. As a simple and effective alternative...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Improving Safety Alignment via Balanced Direct Preference Optimization*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于 **Large Language Models (LLMs)** 在安全对齐（safety alignment）过程中存在的**严重过拟合（overfitting）问题**。尽管当前主流方法如 **Direct Preference Optimization (DPO)** 能在训练分布内提升安全性，但在面对 **out-of-distribution (OOD)** 场景时泛化能力差，容易被 jailbreak 攻击绕过。

作者指出，这一现象的根本原因在于模型对偏好对中不同响应的理解存在**不平衡性（Imbalanced Preference Comprehension）**——即模型对 preferred response 和 dispreferred response 的理解程度不一致，导致优化过程偏向某一方，从而引发过拟合。

---

### 🚀 提出的新方法与新思路
为解决上述问题，作者提出 **Balanced Direct Preference Optimization (B-DPO)**，其核心思想是：

- 引入 **mutual information** 作为衡量模型对 query 与 response 之间“理解水平”的量化指标。
- 发现：在安全偏好对中，preferred 和 dispreferred responses 的 mutual information 存在显著差异，且这种**不平衡程度越高，模型的安全性能越差**。
- 基于此，设计了一种**自适应加权机制**，动态调整 DPO 损失函数中对 preferred 和 dispreferred responses 的优化强度：
  - 对理解更深（mutual information 更高）的响应降低优化权重，防止过吸引或过排斥；
  - 同时引入一个**缩放因子（scaling factor）** 来稳定整体梯度幅度，避免因重加权导致训练不稳定。

---

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|--------|
| **有效性** | 显著提升了 LLM 在多种安全基准上的表现，尤其在抵抗 jailbreak 攻击方面优于 DPO、SafeDPO 等先进方法。 |
| **通用性** | 不依赖特定数据重构或过滤，适用于任意 DPO 流程，兼容性强。 |
| **稳定性** | 缩放因子保障了优化过程的稳定性，避免传统重加权带来的梯度震荡。 |
| **无性能损失** | 在提升安全性的同时，保持了 competitive 的 general capability（如推理、事实性等），未出现明显的“alignment tax”。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- 主要训练数据来自 **PKU-SafeRLHF** 数据集的一个子集：
  - 包含 **10,000 个 safe queries** 和 **10,000 个 unsafe queries**
  - 每条样本为三元组 `(query x, preferred response yw, dispreferred response yl)`
- 安全对齐任务中，对于 unsafe query，将 safe response 设为 preferred，unsafe response 设为 dispreferred。

---

### ⚙️ 实验设置与评估指标

#### ✅ 模型选择
在三个主流开源 LLM 上进行实验：
- **Qwen-2-7B-Instruct**
- **Mistral-7B-Instruct-v0.3**
- **Vicuna-7B-v1.5**

#### ✅ 评估基准（Benchmarks）
| 类型 | 基准名称 | 用途 |
|------|--------|------|
| **安全性能** | **StrongReject** | 评估对直接有害请求的拒绝能力 |
| | **XsTest** | 测试模型是否表现出过度安全行为或可被诱导生成有害内容 |
| | **GCG**（Gradient-based） | 黑盒迁移攻击下的鲁棒性测试 |
| | **PAIR**（Self-refinement） | 多轮迭代 jailbreak 攻击下的防御能力 |
| **通用能力** | **GSM8K** | 数学推理能力 |
| | **SimpleQA** | 事实准确性 |
| | **AdvGLUE** | 对抗性鲁棒性 |
| | **HHH Alignment** | 帮助性、诚实性和无害性综合评估 |

> 所有安全指标以 **模型产生安全响应的比例（safety score）** 报告。

#### ✅ 基线方法对比
- **DPO** (Rafailov et al., 2023)
- **CPO** (Xu et al., 2024a)
- **SimPO** (Meng et al., 2024)
- **SafeDPO** (Kim et al., 2025) — 当前最先进的安全增强型 DPO 方法

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 Qwen-2-7B-Instruct 为例）

| 方法 | StrongReject | XsTest | GCG | PAIR | Avg. Score |
|------|--------------|--------|-----|------|------------|
| Base | 80.51%       | 82.50% | 97.37% | 58.00% | 68.56% |
| DPO  | 84.03%       | 81.50% | 93.54% | 76.00% | 70.63% |
| SafeDPO | 86.50%    | 80.50% | 90.91% | 68.00% | 69.45% |
| **B-DPO (Ours)** | **92.01%** | **84.50%** | **98.59%** | **80.00%** | **73.29%** |

> ✅ **B-DPO 在所有安全指标上均达到最优**，相比最强基线 SafeDPO 在 StrongReject 上提升 **5.51%**，在 GCG 和 PAIR 攻击下也展现出更强抵抗力。

---

### 🔬 消融实验结果（Ablation Study）

在 Qwen-2-7B-Instruct 上验证各组件作用：

| 方法 | StrongReject | XsTest | AdvGLUE |
|------|--------------|--------|---------|
| DPO | 84.03% | 81.50% | 65.85% |
| DPO + BW（仅加平衡权重） | 85.30% | 82.50% | 65.18% |
| DPO + SF（仅加缩放因子） | 86.58% | 83.00% | 65.31% |
| **B-DPO（完整版）** | **92.01%** | **84.50%** | **65.72%** |

> 💡 结论：
> - 单独使用 BW 或 SF 均有一定提升；
> - **两者结合带来显著增益**，证明了缩放因子对训练稳定性的关键作用；
> - 最终版本实现了最佳安全与通用能力的平衡。

---

### ⏱️ 计算开销分析
- 相比标准 DPO，B-DPO 额外需要约 **0.4 小时** 进行 mutual information 的预计算（基于 reference model 推理）；
- 总耗时从 1.5h → 1.9h（在 NVIDIA RTX A800 上），**增加可控，具备实用价值**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Imbalanced Preference Comprehension 是真实存在的现象**：
   - 在多个 LLM 上观察到 preferred/dispreferred responses 的 mutual information 分布不对称；
   - 这种不平衡会误导优化方向，导致模型记忆具体样本而非学习偏好原则。

2. **不平衡程度与安全性能负相关**：
   - 实验表明，在 mutual information 差异较小（更均衡）的数据上训练的模型，最终安全性和泛化能力更好。

3. **B-DPO 可有效缓解该问题**：
   - 通过基于理解水平的自适应加权，使优化过程更加均衡；
   - 显著提升安全性能，同时维持甚至略微改善通用能力。

---

### ⚠️ 局限性
- 当前研究局限于 **text-only 模态**，尚未扩展至多模态场景（如图文输入）；
- 方法仍基于 DPO 框架，未探索在更复杂的 RLHF（如 PPO）中的应用；
- mutual information 的估计依赖于 reference model，可能引入偏差。

---

### 🔮 未来工作方向
- 将 B-DPO 思路推广至 **RLHF/PPO 框架** 中；
- 探索在 **multimodal LLMs** 中是否存在类似的 comprehension imbalance，并设计相应对策；
- 研究如何在线动态估计并调整 comprehension level，实现端到端自适应优化。

---

## ✅ 总结
本论文揭示了一个被忽视的关键问题——**Imbalanced Preference Comprehension**，并提出了简单而有效的解决方案 **B-DPO**。实验证明，该方法不仅能显著提升 LLM 的安全对齐能力，还能保持良好的通用性能，为构建更鲁棒、可信的 AI 系统提供了新的视角和工具。

</details>

---

### 8. [Model Predictive Control with Differentiable World Models for Offline Reinforcement Learning](https://arxiv.org/abs/2603.22430)

**Authors**: Rohan Deb, Stephen J. Wright, Arindam Banerjee  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.22430v1  

#### Abstract
Offline Reinforcement Learning (RL) aims to learn optimal policies from fixed offline datasets, without further interactions with the environment. Such methods train an offline policy (or value function), and apply it at inference time without further refinement. We introduce an inference time adapt...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Model Predictive Control with Differentiable World Models for Offline Reinforcement Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
离线强化学习（**Offline Reinforcement Learning, Offline RL**）的核心挑战是在不与环境进行额外交互的前提下，从固定的离线数据集中学习最优策略。传统方法通常在训练阶段学习一个固定策略，并在推理时直接部署该策略，**无法利用推理时的状态信息来动态调整策略**，导致在分布外（out-of-distribution）状态下的泛化能力受限。

此外，传统的价值函数估计（如 Q-learning）在离线设置下面临严重的**分布偏移（distribution shift）** 和**高估偏差（overestimation bias）**，尤其是在长视野任务中。

---

### **提出的新方法与新思路**
本文提出了一种基于**模型预测控制（Model Predictive Control, MPC）** 的推理时自适应框架，结合**可微世界模型（Differentiable World Model, DWM）**，实现对策略参数的在线优化。

#### **核心创新点：**
- **Differentiable World Model (DWM) Pipeline**  
  构建了一个端到端可微的世界模型，包含三个组件：
  1. **可微分的扩散采样器（Diffusion Sampler）**：用于模拟状态转移 $ s_{t+1} = f_0(s_t, a_t, \epsilon_t) $
  2. **可微奖励模型（Reward Model）**：预测 $ r(s,a) $
  3. **终端值函数（Terminal Value Function）**：由预训练的 critic $ Q_\omega $ 提供长期回报估计

- **推理时策略优化（Inference-Time Policy Adaptation）**  
  在每个时间步 $ t $，以当前状态 $ s_t $ 为起点，通过 DWM 进行多步想象 rollout（imagined rollouts），构建有限视野的代理目标函数（surrogate objective）：
  $$
  J(\phi) = \mathbb{E}_{\epsilon}\left[\sum_{j=0}^{H-1} \gamma^j r_\theta(s_j, a_j) + \gamma^H Q_\omega(s_H, \pi_\phi(s_H))\right]
  $$
  然后通过反向传播计算梯度 $ \nabla_\phi J(\phi) $，并执行几步梯度上升更新策略参数 $ \phi $，再执行第一个动作。

- **端到端梯度传播**  
  利用扩散模型的可重参数化性质，实现了从最终回报到策略参数的完整梯度链，支持基于想象轨迹的策略优化。

---

### **相比现有方法的优势**
| 方法类型 | 特点 | 本文优势 |
|--------|------|---------|
| **传统 Offline RL（如 TD3+BC, CQL, IQL）** | 固定策略，无推理时调整 | ✅ 动态适应当前状态，提升泛化 |
| **生成式策略（如 Decision Transformer, Diffuser）** | 推理时采样候选轨迹，选择首动作 | ❌ 不更新策略参数<br>✅ 本文通过梯度优化策略本身 |
| **基于世界模型的方法（如 MOPO, MOReL）** | 仅在训练中使用想象数据增强 | ❌ 不用于推理时优化<br>✅ 本文将世界模型用于推理时MPC优化 |

> 🔑 **关键区别**：现有方法要么“训练时用模型”，要么“推理时采样但不优化策略”；而本文是**首次在推理时通过可微世界模型对策略参数进行梯度优化**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在 **D4RL benchmark** 上进行全面评估，涵盖两大类连续控制任务：

- **Gym-MuJoCo 任务（18个数据集）**  
  包括 `halfcheetah`, `hopper`, `walker2d` 的多种设定：
  - random, medium, expert, medium-expert, medium-replay, full-replay

- **AntMaze 任务（6个数据集）**  
  更具挑战性的迷宫导航任务：
  - umaze, medium-play/diverse, large-play/diverse

---

### **实验设置与评估指标**
- **评估指标**：D4RL normalized score（归一化得分），基于10次测试episode的平均累积回报。
- **训练阶段**：
  - 预训练策略和 critic 使用 **ReBRAC**（Behavior Regularized Actor-Critic）
  - 单独训练扩散动力学模型 $ f_0 $ 和奖励模型 $ r_\theta $
- **推理阶段**：
  - Horizon $ H = 5 $
  - Rollout 粒子数 $ M = 10 $
  - 内层优化步数 $ E = 5 $
  - 学习率 $ \eta = 1e^{-3} $
- 所有结果报告均值 ± 标准差（跨多个随机种子）

---

### **基线方法对比**
#### **传统 Offline RL 基线（ensemble-free）**
- TD3+BC
- IQL
- CQL
- SAC-RND
- ReBRAC（作为主干策略）

#### **生成模型与规划基线**
- Decision Transformer (DT)
- Trajectory Transformer (TT)
- MOPO（基于模型的离线优化）
- MOReL（悲观模型）
- MBOP（行为先验约束的模型预测）
- Diffuser（基于扩散的轨迹生成）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **Gym-MuJoCo 平均表现（Table 1）**
| 方法 | 平均归一化得分 |
|------|----------------|
| TD3+BC | 70.3 |
| IQL | 72.9 |
| CQL | 73.6 |
| SAC-RND | 82.6 |
| ReBRAC | 81.53 |
| **MPCwDWM（本文）** | **85.33** ✅ |

> 💡 在 **18个任务中，12个优于所有基线**，且在 **16个任务上超过预训练的 ReBRAC 策略**，说明推理时优化带来了显著增益。

#### **AntMaze 平均表现（Table 3）**
| 方法 | 平均得分 |
|------|----------|
| ReBRAC | 77.62 |
| **MPCwDWM** | **85.07** ✅ |

> 🚀 在最难的 `antmaze-large-play` 和 `large-diverse` 上，得分从 59 → 67 和 51 → 66，提升巨大。

#### **与生成模型方法对比（Table 2）**
| 方法 | 平均得分 |
|------|---------|
| DT | 74.7 |
| TT | 78.9 |
| MOPO | 42.1 |
| MOReL | 72.9 |
| MBOP | 47.8 |
| Diffuser | 77.5 |
| **MPCwDWM** | **94.4** ✅ |

> 🏆 **全面超越所有生成式离线RL方法**，尤其在 medium 和 medium-replay 数据集上优势明显。

---

### **消融实验与分析（文中隐含）**
虽然没有显式的消融表，但从以下方面可推断有效性：
- **扩散模型预测精度验证**（Figure 2 & 3）：随着训练步数增加，状态和奖励预测误差持续下降，表明世界模型质量可靠。
- **未改进情况标注**（如 hopper-random）：部分任务因数据质量差或探索不足，未能提升，说明方法依赖于世界模型的准确性。
- **与 ReBRAC 对比**：在大多数任务上优于其基础策略，证明推理时优化有效。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **推理时优化可行且有效**：即使在离线设置下，也能通过可微世界模型在推理时对策略进行梯度优化，显著提升性能。
2. ✅ **MPC + 可微世界模型 是强大组合**：将 MPC 思想引入 Offline RL，利用短期想象 rollout 来指导策略更新，避免了长视野 Q 函数估计的困难。
3. ✅ **优于现有生成模型方法**：相比 Diffuser 等仅采样不优化的方法，本文通过参数更新获得更强的适应性。
4. ✅ **在复杂任务上提升显著**：尤其在 AntMaze 等长视野、稀疏奖励任务中表现突出。

---

### **方法的局限性**
- ⚠️ **计算开销较大**：每次推理需进行多次 rollout 和反向传播，延迟较高，不适合实时性要求极高的场景。
- ⚠️ **依赖世界模型精度**：若扩散模型未能准确建模真实动力学（如 walker2d-expert 中误差较大），性能提升受限。
- ⚠️ **尚未处理模型错误累积**：想象 rollout 越长，误差可能放大，影响梯度质量。

---

### **未来工作方向**
1. **加速推理过程**：
   - 使用 **multi-step generative models** 或 **flow matching** 减少 rollout 步数。
   - 设计更高效的梯度近似方式（如 decoupled updates）。

2. **改进世界模型训练**：
   - 引入不确定性估计，避免在高方差区域过度信任模型。
   - 结合 **Flow Matching** 或 **Consistent Models** 提升生成效率。

3. **解耦策略更新机制**：
   - 受 **Flow Q-Learning** 启发，设计无需 BPTT（backpropagation through time）的推理时优化方法。

4. **Offline-to-Online 迁移**：
   - 定期更新世界模型以应对分布漂移，增强鲁棒性。

---

## **总结**
本文提出了 **MPCwDWM** —— 一种将 **Model Predictive Control** 与 **Differentiable World Model** 相结合的新型 Offline RL 框架。其核心思想是：**在推理时利用当前状态和可微世界模型进行想象 rollout，并通过梯度优化策略参数**。实验证明，这种方法在 D4RL 多个基准上**一致优于强基线**，特别是在复杂任务中表现出色，为 Offline RL 开辟了“**推理时自适应优化**”的新路径。

</details>

---

### 9. [Spiking Personalized Federated Learning for Brain-Computer Interface-Enabled Immersive Communication](https://arxiv.org/abs/2603.22727)

**Authors**: Chen Shang, Dinh Thai Hoang, Diep N. Nguyen, Jiadong Yu  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.22727v1  

#### Abstract
This work proposes a novel immersive communication framework that leverages brain-computer interface (BCI) to acquire brain signals for inferring user-centric states (e.g., intention and perception-related discomfort), thereby enabling more personalized and robust immersive adaptation under strong i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Spiking Personalized Federated Learning for Brain-Computer Interface-Enabled Immersive Communication*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文针对 **6G 支持下的沉浸式通信系统** 中存在的三大挑战提出解决方案：
1. **个体差异性强（Neurodiversity）**：脑电信号（如 EEG）在不同用户之间以及同一用户不同时刻存在显著异质性（non-IID），导致传统统一模型难以泛化。
2. **隐私敏感性高**：原始脑信号属于高度敏感的生理数据，直接上传至云端进行集中训练会引发严重的隐私泄露风险。
3. **终端设备能效瓶颈**：沉浸式终端（如 AR/VR 头显）受限于电池容量和散热能力，持续运行高功耗的深度学习推理与训练不可持续。

---

### 🚀 提出的新方法与创新思路
作者提出了一个名为 **SNN-enabled Personalized Federated Learning (PFLSNN)** 的新型框架，其核心创新如下：

- **融合 BCI 与 PFL 构建个性化感知闭环**  
  利用 **Brain-Computer Interface (BCI)** 非侵入式采集用户的 EEG 信号，实时推断用户意图（intention）和感知不适（perception-related discomfort），实现更个性化的沉浸式体验自适应调整。

- **引入 Spiking Neural Networks (SNNs) 提升能效**  
  将 SNN 作为模型骨干网络嵌入到 PFL 框架中。利用 SNN 的 **稀疏、事件驱动（event-driven）计算特性**，大幅降低本地训练与推理过程中的 MAC（乘加）操作数量，转而以更低能耗的 AC（累加）操作为主。

- **设计基于 SNN 的个性化联邦学习机制（PFLSNN）**  
  每个用户维护一个个性化模型 $ w_k $，同时共享一个全局参考模型 $ w $，通过 proximal 正则项协调个性化与协同学习之间的平衡。

- **理论证明 SNN 可缓解梯度漂移（Gradient Dissimilarity）**  
  在非独立同分布（non-IID）数据下，用户间梯度差异是影响收敛稳定性的主因。本文从理论上证明：SNN 的 **稀疏放电活动（sparse spiking activity）** 能有效减小梯度范数上界，从而降低用户漂移（user drift），提升训练鲁棒性。

---

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **隐私保护** | 采用 **Federated Learning** 范式，原始 EEG 数据保留在本地，仅交换模型更新，避免数据外泄。 |
| **个性化能力** | 引入 **Personalized FL** 而非标准 FL，允许每个用户保留个性化模型，显著提升对神经多样性（neurodiversity）的适应能力。 |
| **能量效率** | SNN 的稀疏激活使推理能耗相比传统 ANN 下降 **6.46×**，更适合长期在线运行的可穿戴设备。 |
| **性能表现** | 在真实 EEG 数据集上达到最高识别准确率（87.53%），优于所有基线方法。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **公开 EEG 数据集** [5]（PhysioNet）
- 包含 **109 名参与者** 的运动想象（motor imagery）任务记录
- 实验选取其中 3 名用户（Participant 1, 3, 7）作为联邦客户端模拟多用户场景
- 分类任务为 **4 类动作识别**：左手握拳、右手握拳、双手握拳、双足运动

---

### ⚙️ 实验设置
- **网络架构**：统一使用相同的 CNN 结构（保证公平比较）
  - ANN 使用常规 ReLU 激活函数
  - SNN 使用 Leaky Integrate-and-Fire (LIF) 神经元，时间步 $ T=6 $
- **训练参数**：
  - 学习率：0.01
  - Batch size：64
  - Local epoch $ E = 2 $
  - Proximal coefficient $ \mu = 10^{-5} $
- **评估周期**：共训练 50 轮（global rounds）

---

### 🎯 评估指标
1. **识别准确率（Identification Accuracy）**：测试集上的分类精度
2. **推理能耗（Inference Energy Consumption）**：基于突触操作数和平均放电率估算
3. **消融分析**：验证 PFL 和 SNN 各自带来的增益

---

### 🆚 基线方法对比
| 方法 | 简称 | 特点 |
|------|------|------|
| PFL + ANN | PFLANN | 个性化联邦学习 + 传统人工神经网络 |
| FL + SNN | FLSNN | 标准联邦学习 + 脉冲神经网络 |
| FL + ANN | FLANN | 标准联邦学习 + 传统人工神经网络 |
| **PFL + SNN** | **PFLSNN** | **本文提出的方法（最优组合）** |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

| 方法 | 最终识别准确率 | 推理能耗（相对值） |
|------|------------------|--------------------|
| **PFLSNN**（本文） | **87.53%** | **1×**（最低） |
| PFLANN | 81.90% | 6.46× |
| FLSNN | 68.78% | ~1.5× |
| FLANN | 62.12% | ~7× |

> 注：能耗值为相对于 PFLSNN 的倍数；数值越低越好。

---

### 🔁 与基线方法的对比结果
- **准确性方面**：
  - PFLSNN 比 PFLANN 提升 **+5.63%**
  - 比 FLSNN 提升 **+18.75%**
  - 比 FLANN 提升 **+25.41%**
  - 表明 **“个性化” + “SNN” 双重机制带来最大收益**

- **能效方面**：
  - PFLSNN 的推理能耗仅为 PFLANN 的 **1/6.46**
  - 主要得益于 SNN 的稀疏放电（平均 firing rate ≈ 0.12）
  - 每层放电率分别为：0.106, 0.063, 0.058, 0.253

---

### 🔍 消融实验结果
- **切换从 FL 到 PFL**：
  - 在相同 backbone 下，PFL 显著优于 FL
    - PFLSNN vs FLSNN：+18.75%
    - PFLANN vs FLANN：+19.78%
  - 说明 **个性化建模对处理 neurodiversity 至关重要**

- **切换从 ANN 到 SNN**：
  - 在相同学习范式下，SNN 均优于 ANN
    - PFLSNN vs PFLANN：+5.63%
    - FLSNN vs FLANN：+6.66%
  - 说明 **SNN 更适合捕捉 EEG 的时序动态特征，并提升训练稳定性**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **BCI 是实现真正个性化沉浸式通信的关键入口**  
   用户内部状态（如注意力、晕动症）无法仅靠外部传感器感知，必须结合 EEG 等神经信号才能实现闭环自适应。

2. **PFL 显著优于标准 FL 在处理脑信号异质性方面的表现**  
   由于 EEG 数据具有强烈的跨用户和跨通道差异，单一全局模型无法满足个性化需求，PFL 提供了一种有效的折中方案。

3. **SNN 不仅节能，还能增强训练鲁棒性**  
   - **节能**：稀疏事件驱动计算显著减少能耗（实测 **6.46× 节能**）
   - **抗漂移**：理论证明稀疏放电可压缩梯度范数，减轻 non-IID 导致的梯度分歧，提升聚合稳定性

4. **PFLSNN 实现了性能与能效的双重领先**  
   在保持最高识别精度的同时，成为最节能的部署选项，适用于资源受限的 BCI-HMD 设备。

---

### ⚠️ 方法的局限性
1. **SNN 训练复杂度较高**  
   需依赖 BPTT（Back-Propagation Through Time）和 surrogate gradient 方法，训练难度高于 ANN。

2. **当前实验规模较小**  
   仅使用 3 个用户进行联邦模拟，未来需在更大规模、更多样化的用户群体中验证泛化能力。

3. **未考虑动态用户加入/退出机制**  
   实际应用中用户可能随时接入或离线，需进一步研究异步 PFL 机制。

4. **硬件部署尚未验证**  
   当前能耗评估为理论估算，尚未在真实的 neuromorphic chip（如 Loihi, SpiNNaker）上实测。

---

### 🔮 未来工作方向
1. **结合 Neuromorphic Computing 硬件加速 SNN 推理**  
   将 PFLSNN 部署于脉冲神经形态芯片，进一步释放能效潜力。

2. **扩展至多模态生理信号融合**  
   融合 EEG + EOG + EMG + HRV 等多种生理信号，构建更全面的 user state estimator。

3. **探索无监督/自监督 SNN 预训练策略**  
   减少对标注数据的依赖，适应长期连续使用的零校准（zero-calibration）场景。

4. **开发轻量化 SNN 架构搜索（TinySNN-NAS）**  
   自动优化 SNN 结构以适配不同型号的 BCI-HMD 终端。

--- 

> **总结一句话**：  
> 本文提出的 **SNN-enabled PFL** 框架首次将 **Spiking Neural Networks** 与 **Personalized Federated Learning** 结合用于 BCI 驱动的沉浸式通信，在保障隐私的前提下实现了 **高性能、高鲁棒性、超低功耗** 的个性化脑信号解码，为未来 6G 沉浸式系统提供了可行的技术路径。

</details>

---

### 10. [Universal and efficient graph neural networks with dynamic attention for machine learning interatomic potentials](https://arxiv.org/abs/2603.22810)

**Authors**: Shuyu Bi, Zhede Zhao, Qiangchao Sun, Tao Hu, Xionggang Lu, Hongwei Cheng  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.22810v1  

#### Abstract
The core of molecular dynamics simulation fundamentally lies in the interatomic potential. Traditional empirical potentials lack accuracy, while first-principles methods are computationally prohibitive. Machine learning interatomic potentials (MLIPs) promise near-quantum accuracy at linear cost, but...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Universal and efficient graph neural networks with dynamic attention for machine learning interatomic potentials*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统分子动力学（MD）模拟依赖于**经验势函数**（如经典力场），其精度受限且难以描述复杂多体相互作用；而基于第一性原理的**ab initio 分子动力学**（如 DFT）虽具高精度，但计算成本高昂（$O(N^3)$），难以应用于大规模、长时间尺度的系统。尽管机器学习原子间势（MLIPs）在准确性和效率之间提供了折衷，但现有的 **SE(3)-equivariant GNNs** 模型仍面临以下挑战：
- 高阶张量运算导致的**计算复杂度高**；
- 消息传递机制在并行化时存在通信瓶颈；
- 在有限训练数据下易出现过拟合。

### 提出的新方法与创新思路
本文提出了一种新型图神经网络框架——**MLANet**（Machine Learning Advances Neural Network），具备两大核心设计：

#### （1）Geometry-aware Dual-path Dynamic Attention
- 引入一种**双路径动态注意力机制**，将几何信息（方向向量通过 spherical harmonics 编码）与化学特征（原子类型嵌入）解耦处理。
- 注意力权重由查询（query）和键（key）之间的 **tensor product** 构建，并结合温度缩放与 softmax 归一化。
- 同时引入一个基于门控机制（gating）的消息调制模块，增强模型对局部环境细微变化的敏感性。

> ✅ 优势：实现了更精细的几何感知消息传递，提升了模型区分相似构型的能力。

#### （2）Physics-informed Multi-perspective Pooling
- 融合三种池化操作：
  - **Additive pooling**：保留广延性质（如总能量）；
  - **Mean pooling**：捕获强度不变属性；
  - **Max pooling**：突出关键局域环境（如活性位点）。
- 将三者拼接后输入预测头，构建更全面的全局表示。

> ✅ 优势：避免信息丢失，提升泛化能力，无需复杂的后处理层即可实现高精度能量预测。

### 相比现有方法的优势
| 维度 | MLANet 表现 |
|------|-------------|
| **准确性** | 在多个基准任务上达到或接近主流 equivariant 模型（如 NequIP、MACE）水平 |
| **效率** | 显著优于同类 equivariant 模型，训练速度**快一个数量级**，内存占用更低 |
| **稳定性** | 支持长达 300 ps 的稳定分子动力学模拟 |
| **可扩展性** | 可在消费级 GPU（如 NVIDIA 4060 笔记本显卡）运行千原子级别系统 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 | 系统类型 |
|-------|------|--------|
| **QM7** | 小有机分子，最大23个原子，共7,165个样本 | 非周期性分子 |
| **MD17** | 8种小分子的动力学轨迹数据 | 动力学路径 |
| **Mptrj (Li-subset)** | 材料项目数据库中的含锂晶体结构，共 ~22万训练样本 | 周期性无机材料 |
| **SiO₂**, **Ge-Sb-Te**, **Black Phosphorus** | 多晶相、合金、二维材料等 | 固态材料 |
| **Bilayer Graphene** | 双层石墨烯滑移与剥离能面 | 二维范德华材料 |
| **Formate Decomposition** | 甲酸在铜表面分解反应路径 | 表面催化 |
| **Water (bulk)** | 液态水体系，192原子/构型 | 液体系统 |
| **Charged Systems** | 包括 C₁₀H₂⁺/⁻, Ag⁺/⁻, Na₉Cl₉⁰/± 等带电体系 | 带电系统 |
| **QM9 / QM9S** | 大规模有机分子量子化学性质数据集 | 分子光谱与极化率预测 |

### 实验设置与评估指标
- **训练目标**：联合优化能量 $E$ 和力 $\mathbf{F}$ 的 L1 损失：
  $$
  \mathcal{L}_{\text{total}} = \lambda_E \cdot \mathcal{L}_E + \lambda_F \cdot \mathcal{L}_F
  $$
- **评估指标**：
  - 能量 MAE / RMSE（单位：kcal/mol 或 meV/atom）
  - 力 MAE / RMSE（单位：meV/Å 或 eV/Å）
  - 应力预测误差
  - 分子动力学模拟稳定性与时长
  - 训练/推理速度（s/epoch）、显存占用

### 基线方法对比
参与比较的主流模型包括：
- **Invariant GNNs**: SchNet, DimeNet++, PaiNN
- **Equivariant GNNs**: NequIP, MACE, Allegro, CAMP
- **传统力场**: REBO, AIREBO, ReaxFF
- **其他 MLIPs**: GAP, DeePMD, BPNN, ACE

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ QM7 原子化能预测（MAE）
| 方法 | MAE (kcal/mol) |
|------|----------------|
| DTNN | 8.2 |
| SchNet | 74.2 |
| GGST+EK | 3.4 |
| **MLANet** | **3.07 ± 0.09** |

> ✔️ **当前最优结果**

#### ✅ MD17 力预测 MAE（小样本训练）
- 在苯（benzene）系统上取得最低力误差
- 大样本训练（9500+）后进行 MD 模拟，所有测试分子均保持 **300 ps 稳定运行**

#### ✅ Bilayer Graphene 滑移与结合能（RMSE）
| 方法 | Sliding Energy (meV/atom) | Binding Energy (meV/atom) |
|------|----------------------------|------------------------------|
| ReaxFF | 0.7 | 16.0 |
| NequIP | 1.0 | 1.6 |
| hNN | 0.4 | 1.2 |
| **MLANet** | **0.5** | **1.3** |

> ✔️ 接近最先进模型，显著优于大多数传统和 ML 势函数

#### ✅ Formate Decomposition
| 方法 | Force MAE (meV/Å) | Energy MAE (meV/atom) |
|------|--------------------|------------------------|
| NequIP | 47.3 | 0.50 |
| AlphaNet | 42.5 | 0.23 |
| **MLANet** | **44.9** | 2.31 |

> ⚠️ 能量误差较高，但**力预测表现优异**，适合用于反应路径模拟

#### ✅ Water 体系（RMSE）
| 方法 | Energy (meV/atom) | Forces (meV/Å) |
|------|--------------------|-----------------|
| BPNN | 2.3 | 120 |
| MACE | 0.63 | 36 |
| **MLANet** | **0.47** | 60 |

> ✔️ **能量 RMSE 最低**，表明潜力巨大；但力误差略高，可能因小数据集导致轻微过拟合

#### ✅ 带电系统 Force RMSE（eV/Å）
| Dataset | 4G-HDNNP | Maruf’s NequIP | ReaxNet | **MLANet** |
|--------|----------|---------------|---------|-----------|
| C₁₀H₂⁺/⁻ | 0.078 | 0.071 | 0.023 | **0.074** |
| Ag⁺/⁻ | 0.033 | – | 0.005 | **0.024** |
| Na₉Cl₉ | 0.032 | 2.145 | 0.028 | **0.012** |

> ✔️ 在 Na₉Cl₉ 上**超越所有包含长程静电项的模型**，验证了直接嵌入电荷的有效性

### 消融实验结果（见 Figure 2 & 3）
- **不同 $l_{\text{max}}$ 设置的影响**：
  - $l=2$ 在多数情况下优于 $l=3$，尤其在小数据集上；
  - 更高的 $l$ 提升表达能力但也加剧过拟合风险；
  - **建议根据数据规模灵活选择 $l_{\text{max}}$**。
- **计算效率分析**：
  - $l=3$ 模型的内存消耗和每轮训练时间显著高于 $l=1,2$；
  - MLANet 在同等精度下比 NequIP 快 **~10倍**。

---

## 4. 关键结论和发现

### 主要发现
1. **MLANet 实现了精度与效率的良好平衡**：
   - 凭借 dual-path attention 和 multi-perspective pooling，在多种物理化学系统中达到**接近 SOTA 的预测精度**；
   - 同时展现出卓越的**计算效率与可扩展性**，适用于大规模原子模拟。

2. **架构设计有效缓解信息损失与过拟合**：
   - 多视角池化策略增强了系统表征能力；
   - 动态注意力机制提高了对局部几何细节的分辨力。

3. **适用于多样化的应用场景**：
   - 成功建模从有机分子到二维材料、表面催化、液体及带电系统的广泛体系；
   - 特别在**表面反应与长程相互作用建模**方面表现出色。

### 方法的局限性
- **力预测平滑性不足**：作为 direct-force model，其力场光滑性低于基于梯度的能量守恒模型（如 NequIP），限制了其在**过渡态搜索**等需要精确 Hessian 的任务中的应用。
- **对小数据集敏感**：在仅含 1593 构型的 water 数据集中出现轻微过拟合迹象。
- **尚未整合显式的长程静电项**：目前通过直接嵌入电荷处理带电系统，未来仍有改进空间。

### 未来工作方向
1. **融合更高阶物理可观测量**：
   - 利用包含 Hessian 矩阵的大规模训练数据，提升力场对称性与平滑性。
2. **引入 active learning 策略**：
   - 自动生成高质量训练数据，提高数据利用效率。
3. **开发更高效的 equivariant operator**：
   - 进一步降低高阶 irreps 的计算开销。
4. **集成 long-range electrostatic interactions**：
   - 结合 QEq 或 LES 类方法，完善对极化与电荷转移效应的建模。
5. **拓展至超大规模模拟**：
   - 探索在催化、能源材料等关键领域的大尺度、长时间 MD 应用。

---

> 🔚 **总结**：  
> MLANet 是一种兼具**高精度、高效率与强鲁棒性**的新型 equivariant GNN 框架，为实现“**近量子精度 + 线性计算成本**”的终极目标迈出了重要一步，有望成为下一代大规模原子模拟的标准工具之一。

</details>

---

### 11. [Weak-PDE-Net: Discovering Open-Form PDEs via Differentiable Symbolic Networks and Weak Formulation](https://arxiv.org/abs/2603.22951)

**Authors**: Xinxin Li, Xingyu Cui, Jin Qi, Juan Zhang, Da Li, Junping Yin  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.22951v1  

#### Abstract
Discovering governing Partial Differential Equations (PDEs) from sparse and noisy data is a challenging issue in data-driven scientific computing. Conventional sparse regression methods often suffer from two major limitations: (i) the instability of numerical differentiation under sparse and noisy d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Weak-PDE-Net: Discovering Open-Form PDEs via Differentiable Symbolic Networks and Weak Formulation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于稀疏回归的PDE发现方法面临两大挑战：
- **数值微分对噪声敏感**：在稀疏且含噪的数据下，直接估计高阶导数会放大噪声，导致不稳定。
- **候选库限制表达能力**：预定义的函数项库（如多项式、三角函数等）限制了模型发现“开形式”（open-form）PDE的能力，即无法探索未知或复杂组合的数学表达式。

### 提出的新方法与创新思路
作者提出 **Weak-PDE-Net**，一个端到端可微分的框架，用于从稀疏和含噪数据中鲁棒地识别开放形式的PDE。其核心创新包括：

#### ✅ **双模块架构设计**
- **Forward Response Learner**：采用嵌入**可学习高斯核**（learnable Gaussian kernels）的轻量级MLP作为代理模型，自适应捕捉系统动态，缓解标准MLP中的**spectral bias**（频谱偏差），提升对高频特征的拟合能力。
- **Weak-form PDE Generator**：结合**符号网络**（symbolic network）与积分模块，构建弱形式PDE，避免显式数值微分，提高抗噪性。

#### ✅ **可微分神经架构搜索（DNAS）实现Open-form Discovery**
- 引入**Differentiable Neural Architecture Search (DNAS)** 策略，在训练过程中自动搜索最优的符号网络结构。
- 不依赖固定库，而是通过梯度优化动态生成函数项组合，真正实现**open-form PDE discovery**。

#### ✅ **物理一致性增强机制**
- 对于多变量系统，引入：
  - **Galilean Invariance Constraint**：过滤违反伽利略不变性的非物理项（如绝对速度项），保留对流项等合理结构。
  - **Symmetry Equivariance Hypothesis**：针对复值系统（如Nonlinear Schrödinger方程），利用U(1)对称性假设，强制实部与虚部分量之间满足反对称耦合结构，确保物理一致性。

#### ✅ **端到端联合优化**
- 将响应函数学习与PDE结构发现统一在一个可微分框架内，二者相互约束、协同进化，显著提升了在低采样率和高噪声下的鲁棒性。

### 相比现有方法的优势
| 特性 | Weak-PDE-Net | 传统方法（如SINDy, PDE-FIND） | Weak-PDE-LEARN |
|------|-------------|-------------------------------|----------------|
| 数值微分 | ❌ 避免（使用弱形式） | ✅ 显式计算（易受噪声影响） | ✅ 使用RatNN近似，但仍需微分 |
| 候选库灵活性 | ✅ 开放式（DNAS动态构造） | ❌ 固定库 | ❌ 固定库 |
| 可微分性 | ✅ 完全可微，支持端到端训练 | ❌ 分离式流程 | ✅ 可微，但受限于库 |
| 多变量一致性 | ✅ 支持（物理约束引导） | ❌ 各变量独立建模 | ❌ 缺乏全局一致性机制 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖多种物理场景，涵盖不同维度、阶数和非线性程度的PDE：

| 方程类型 | 示例 | 维度 | 特点 |
|--------|------|-------|------|
| 1D Scalar | Burgers, KdV, Kuramoto-Sivashinsky (KS), Chafee-Infante (CI) | d=1, h=1 | 覆盖耗散、色散、混沌、反应扩散现象 |
| Complex-valued | Nonlinear Schrödinger (NLS) | d=1, h=2 | 复场系统，测试对称性假设有效性 |
| 2D Spatial | Wave Equation, Sine-Gordon (SG), Incompressible Navier-Stokes (NS) | d=2 | 测试高维扩展能力和流体动力学建模 |

所有数据均为模拟生成，并加入高斯噪声，空间时间网格不规则采样以模拟真实观测条件。

### 实验设置
- **稀疏性控制**：定义采样比 $ r = N_{\text{data}} / N_{\text{total}} $，范围从 **1.0% 到 50%**。
- **噪声水平**：相对噪声强度 $ \sigma_{\text{NR}} $ 从 **0% 到 100%**。
- **训练策略三阶段**：
  1. **Searching Phase**：DNAS搜索最优符号网络结构。
  2. **Pruning Phase**：L1正则化剪枝冗余连接，简化方程。
  3. **Tuning Phase**：最小二乘回归精调系数，输出最终PDE。

### 评估指标
| 指标 | 公式 | 含义 |
|------|------|------|
| **TPR (True Positivity Ratio)** | $ \frac{\text{TP}}{\text{TP} + \text{FN} + \text{FP}} $ | 正确识别非零项的比例，越高越好（理想为1） |
| **$ E_{\infty}(\xi) $** | $ \max_j \left| \frac{\xi_j - \xi_j^*}{\xi_j^*} \right| $ | 最大相对误差，衡量最差恢复精度 |
| **$ E_2(\xi) $** | $ \frac{\|\xi - \xi^*\|_2}{\|\xi^*\|_2} $ | 归一化RMSE，衡量平均恢复精度 |
| **Reconstruction Error $ \mathcal{L}(U,\hat{U}) $** | MSE on test set | 学习响应函数的准确性 |

### 基线方法对比
- **Weak-PDE-LEARN**：当前最先进的弱形式PDE发现方法，使用Rational Neural Networks（RatNN）进行函数逼近。
- 所有实验在相同数据分布和噪声条件下进行，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（代表性结果汇总）

| 方程 | 采样比 | 噪声 | TPR | $ E_{\infty} $ | $ E_2 $ | 结果说明 |
|------|--------|------|-----|---------------|----------|-----------|
| **Burgers** | 2.5% | 0% | 1.00 | 0.0589 | 0.0557 | 在极低采样下仍能准确恢复 |
| **KdV** | 2.5% | 0% | 1.00 | 0.0719 | 0.0259 | 成功识别三阶导数项 |
| **KS** | 2.5% | 0% | 1.00 | 0.0173 | 0.0149 | 准确捕获四阶耗散项 |
| **NLS** | 10% | 100% | 1.00 | 0.3755 | 0.1887 | 即使100%噪声也能保持结构正确 |
| **Wave (2D)** | 1.0% | 0% | 1.00 | 0.0181 | 0.0135 | 高维系统依然稳健 |
| **NS (vorticity form)** | 5% | 0% | 1.00 | 0.0374 | 0.0141 | 成功识别对流项并去除非物理项 |

> ✅ **所有测试案例中，TPR均达到1.00**，表明Weak-PDE-Net能够稳定恢复正确的PDE结构。

### 与基线方法对比（Table 9）
在Burgers、KdV、KS三个方程上对比Weak-PDE-LEARN：

| 噪声水平 | 方法 | $ E_{\infty} $(KS) | $ E_2 $(KS) | 是否成功恢复结构？ |
|---------|------|--------------------|-------------|------------------|
| 25% | Weak-PDE-LEARN | 0.0100 | 0.0094 | ✅ 是 |
| 100% | Weak-PDE-LEARN | — | — | ❌ 否（TPR < 1） |
| 100% | **Weak-PDE-Net** | **0.1340** | **0.0973** | ✅ 是 |

> 🔺 **Weak-PDE-Net在100%噪声下仍能成功识别KS方程结构，而Weak-PDE-LEARN失败**，显示更强的鲁棒性。

### 消融实验结果（Ablation Study）
比较以下变体：
- **Full Model (FE + NAS)**：完整模型
- **w/o FE**：移除可学习高斯核
- **w/o NAS**：替换为固定符号网络

| 模型 | 平均TPR | 平均$ E_{\infty} $ | 说明 |
|------|--------|------------------|------|
| Full Model | 1.00 | ~10⁻²–10⁻³ | 性能最佳 |
| w/o FE | 0.42 | >1.0 | 重建MSE大幅上升，尤其在Burgers/NS中无法捕捉激波 |
| w/o NAS | 0.33 | >2.0 | 产生过度复杂的方程，含大量虚假项 |

> 🔺 移除任一组件都会导致性能急剧下降，验证了**FE缓解spectral bias** 和 **NAS实现简洁发现** 的关键作用。

---

## 4. 关键结论和发现

### 主要发现
1. **Weak-PDE-Net能够在高度稀疏（低至1%采样）和强噪声（高达100%）条件下，准确恢复各种复杂PDE的结构和参数**。
2. **可学习高斯核有效缓解了MLP的spectral bias**，使其能同时捕捉平滑趋势和尖锐梯度（如激波）。
3. **DNAS策略实现了真正的open-form PDE discovery**，无需预设库即可动态构造数学表达式，并通过剪枝获得简洁解。
4. **物理先验（Galilean Invariance, Symmetry Equivariance）显著提升多变量系统的发现一致性和鲁棒性**，防止出现非物理项。
5. **弱形式结合可微分符号网络是实现鲁棒PDE发现的有效范式**，避免了数值微分带来的误差传播。

### 方法的局限性
1. **无法表示嵌套微分算子**（nested differential operators）  
   如 porous medium equation $ u_t = \nabla \cdot (u \nabla u) $ 中的复合结构难以用当前 $ F(U) $ 形式表达。
   
2. **弱形式积分具有低通滤波效应**  
   虽然抑制噪声，但也可能平滑掉细尺度物理特征（如CI方程中的相界），影响高噪声下的识别精度。

3. **计算成本较高**  
   DNAS和积分模块增加了训练复杂度，尤其在高维空间中。

### 未来工作方向
1. **扩展网络架构以支持嵌套微分操作符**  
   可引入额外的微分层和深层符号网络堆叠，逐步建模更复杂的算子结构。

2. **开发块状训练策略**（block-wise training）  
   提升训练效率，降低内存消耗，适用于更大规模系统。

3. **集成更多物理先验知识**  
   如守恒律、能量耗散、尺度不变性等，进一步压缩搜索空间，提升泛化能力。

4. **应用于真实世界科学数据**  
   如气候、生物、材料等领域中的观测数据，推动AI for Science的实际落地。

---

> 📌 **总结一句话**：  
> **Weak-PDE-Net通过“可学习高斯核 + 可微分符号网络 + 弱形式 + 物理约束”的协同设计，首次实现了在极端噪声和稀疏性下端到端发现开放形式PDE的鲁棒框架，为数据驱动科学发现提供了强大工具。**

</details>

---

### 12. [Detecting Non-Membership in LLM Training Data via Rank Correlations](https://arxiv.org/abs/2603.22707)

**Authors**: Pranav Shetty, Mirazul Haque, Zhiqiang Ma, Xiaomo Liu  
**Category**: cs.CL  
**Published**: 2026-03-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.22707v1  

#### Abstract
As large language models (LLMs) are trained on increasingly vast and opaque text corpora, determining which data contributed to training has become essential for copyright enforcement, compliance auditing, and user trust. While prior work focuses on detecting whether a dataset was used in training (...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Detecting Non-Membership in LLM Training Data via Rank Correlations*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前关于 **Membership Inference Attack (MIA)** 和 **Dataset Inference (DI)** 的研究主要集中在判断某个数据集是否被用于训练 LLM（即“成员检测”）。然而，一个对合规审计、版权争议和用户信任同样重要的**互补问题**——如何**可靠地证明某数据集 *未被* 用于训练**（non-membership detection）——长期被忽视。

现有方法存在两大局限：
1. 需要访问与目标数据分布匹配的已知非成员数据集作为参考，这在实践中（尤其是专有数据）难以获得。
2. 方法本质上是单向的，只能提供“是成员”的证据，而无法提供“非成员”的确凿证据。**未能检测到成员 ≠ 证明是非成员**。

### 🚀 提出的新方法：PRISM
作者提出 **PRISM**（Normalized token log Probability Rank correlation Inference using Spearman for non-Membership detection），一种基于秩相关性的非成员检测框架。

#### 核心思想
- 利用两个模型对同一数据集的 **Min-K%++** 分数进行排序。
- 如果目标模型 $M_T$ **没有** 在该数据集上训练过，则它与一个已知未训练过的参考模型 $M_R$ 会因共享相似的语言先验而对文档难度有相似的排序，导致高 **Spearman 秩相关性**。
- 如果 $M_T$ **已经** 在该数据集上训练过，记忆效应会扭曲其排序，使其与 $M_R$ 的相关性降低。
- 为了量化这一差异，引入一个 **蒸馏参考模型 $M_D$**：它是通过在目标数据集上微调 $M_R$ 并同时从 $M_T$ 进行知识蒸馏得到的，模拟“如果 $M_T$ 被训练会怎样”。

#### 统计检验
定义测试统计量：
$$
\Delta = \rho(M_R, M_T; D) - \rho(M_D, M_T; D)
$$
其中 $\rho$ 是 Spearman 秩相关系数。若 $\Delta > 0$ 且显著，则支持 $M_T$ 未在 $D$ 上训练（非成员）。

采用 **bootstrap 方法** 计算 p-value，进行单侧假设检验：
- $H_0: \Delta \leq 0$ vs. $H_1: \Delta > 0$
- 若 p-value < 0.05，则拒绝 $H_0$，认为有证据支持非成员。

### ⭐ 相比现有方法的优势
| 特性 | PRISM | 传统 MIA/DI |
|------|-------|------------|
| **检测方向** | 明确支持 **非成员检测** | 主要支持成员检测 |
| **参考数据需求** | ❌ 不需要额外的非成员参考数据集 | ✅ 通常需要分布匹配的参考数据 |
| **访问权限** | ✅ 仅需 **grey-box**（logits） | 可能需要更多内部信息或参数 |
| **样本效率** | ✅ 仅需约 **100 文档**即可有效检测 |
| **鲁棒性** | ✅ 基于秩相关，对模型规模和异常值鲁棒 |

---

## 2. 核心实验方法和设置

### 📚 数据集
使用多个公开且明确未被用于训练参考模型的数据子集，确保 ground truth 清晰：
- **The Pile 子集**：Wikipedia, PubMed, ArXiv, CommonCrawl (CC), HackerNews (HN), Ubuntu-IRC, Enron emails, Freelaw
- **Dolma 子集**：Reddit（进一步处理以确保时间戳晚于模型训练截止日期）
- 所有数据均经过去重和长度控制（100–200词，≤512 tokens）

### 🔧 实验设置
- **参考模型 $M_R$**：Pythia 系列（70M–6.9B）、OLMo-1b（用于不同架构测试）
- **目标模型 $M_T$**：
  - **非成员场景**：原始 Pythia-410m（未在测试数据上训练）
  - **成员场景**：Pythia-410m-CPT（在部分测试数据上继续预训练）
- **蒸馏参考模型 $M_D$**：由 $M_R$ 微调 + 从 $M_T$ 蒸馏得到（$\lambda=0.7$, $T=2$）
- **信号分数**：Min-K%++（K=1%）
- **评估方式**：计算 $\Delta$ 的 bootstrap p-value，阈值为 0.05

### 🆚 基线方法对比
- **LLM-DI**：基于多维 MIA 特征训练分类器，需参考数据集。
- **PaCoST**：基于原始与改写文本置信度差异的 paired t-test，原用于 QA 基准污染检测。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Table 2 & 3）

#### 表：PRISM 在非成员检测上的 p-values（部分）
| Target Model       | PubMed | HN   | ArXiv | CC   | Reddit |
|--------------------|--------|------|-------|------|--------|
| Pythia-410m-CPT    | 1.000  | 0.354| 0.113 | 0.302| 0.817  |
| Pythia-410m        | 1.0e-4 | 1.0e-4|1.0e-4|1.0e-4|1.0e-4 |
| Pythia-1b          | 1.0e-4 | 1.0e-4|1.0e-4|1.0e-4|1.0e-4 |

- ✅ **所有已知非成员数据集**（如 Pythia-410m 测试 PubMed 等）均得到 **极低 p-value**（≤1e-4），成功检测为非成员。
- ✅ 对于实际训练过的数据（Pythia-410m-CPT），p-value **远高于 0.05**，**无假阳性**，验证了方法可靠性。
- ✅ 即使使用不同架构的 OLMo-1b 作为参考模型，也能正确识别 Reddit 为非成员。

#### 表：LLM-DI 与 PaCoST 在成员检测上的表现（Table 4）
| Method     | PubMed (410m-CPT) | HN (410m-CPT) | ArXiv (410m-CPT) | CC (410m-CPT) |
|-----------|-------------------|---------------|------------------|---------------|
| LLM-DI    | 0.775             | 0.022         | 0.083            | 0.045         |
| PaCoST    | 0.045             | 0.999         | 0.122            | 0.009         |

- ❌ LLM-DI 仅在 2/5 数据集上成功检测到成员（p<0.05），其余失败。
- ❌ PaCoST 结果不一致，有时两者都检出或都不检出，缺乏区分能力。
- ➡️ 说明现有方法不仅不能用于非成员检测，甚至在成员检测上也不可靠。

### 🔍 消融实验结果

#### （1）参考模型规模影响（Table 5）
- 更小的模型（如 Pythia-70M）作为 $M_R$ 时，相关性更弱，导致检测不稳定。
- **Pythia-1b** 提供最一致的信号，表明参考模型需具备足够容量。

#### （2）数据集大小影响（Figure 3）
- 当文档数 ≥ 150 时，p-value 稳定。
- **低至 100 文档仍可有效检测**，显示高实用性。

#### （3）Min-K%++ 中 K 值的影响（Figure 4）
- 较小的 K（如 1%）能捕捉最强的记忆信号。
- K 增大后，包含更多可预测 token，稀释了高惊讶度 token 的影响，相关性差异减小。

#### （4）直接比较 Min-K%++ 数值的失败尝试（Table 10）
- 使用 paired t-test 直接比较 $M_T$ 与 $M_D$ 的 Min-K%++ 值会导致大量假阳性（尤其对 Pythia-410m-CPT）。
- 说明**绝对分数受模型架构和规模影响大**，而**秩相关更鲁棒**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **非成员检测是可行且必要的**：PRISM 首次系统性解决了 LLM 训练数据中“未使用”的验证问题。
2. **秩相关是可靠信号**：两个未见过某数据集的模型对其文档难度的排序高度一致；一旦其中一个被训练，这种一致性被破坏。
3. **Min-K%++ 是最优代理信号**：相比 Loss、zlib、DC-PPD 等，Min-K%++ 在不同模型间的秩相关差异最大，最适合用于检测。
4. **无需参考数据集**：PRISM 不依赖外部非成员数据，极大提升了实用性和普适性。
5. **高鲁棒性与实用性**：适用于不同架构（OLMo vs. Pythia），仅需少量文档（~100），且无假阳性。

### ⚠️ 局限性
1. **依赖发布日期**：需知道数据集的公开时间以选择合适的 $M_R$（训练截止早于该时间）。
2. **全集假设**：假设整个数据集要么全部包含，要么全部排除，无法处理部分包含的情况。
3. **近似训练模拟**：$M_D$ 是对“若被训练”的近似，长周期完整训练可能影响结果。
4. **适用范围**：主要适用于开放源代码模型广泛出现后的数据。

### 🔮 未来工作方向
- 扩展至部分成员检测（partial membership）。
- 探索无需蒸馏的轻量化版本。
- 应用于真实世界企业级 LLM 合规审计流程。
- 结合 watermarking 技术，构建完整的“成员/非成员”验证生态。

---

> **总结**：PRISM 提出了一种新颖、实用且可靠的框架，首次实现了对 LLM 是否**未使用**特定数据集的**高置信度验证**，填补了现有研究的关键空白，为版权合规、模型审计和用户信任提供了重要工具。

</details>

---

### 13. [Neural Structure Embedding for Symbolic Regression via Continuous Structure Search and Coefficient Optimization](https://arxiv.org/abs/2603.22429)

**Authors**: Fateme Memar, Tao Zhe, Dongjie Wang  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.22429v1  

#### Abstract
Symbolic regression aims to discover human-interpretable equations that explain observational data. However, existing approaches rely heavily on discrete structure search (e.g., genetic programming), which often leads to high computational cost, unstable performance, and limited scalability to large...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Neural Structure Embedding for Symbolic Regression via Continuous Structure Search and Coefficient Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

传统 **Symbolic Regression (SR)** 方法（如 Genetic Programming, GP）存在以下三大瓶颈：

1. **Discrete structure search** 导致搜索空间高度非光滑，微小的结构变化可能引起函数行为的巨大跳跃，造成优化不稳定、效率低。
2. **Structure 和 coefficient 耦合优化** 增加了搜索复杂度，结构改变时系数需重新拟合，影响收敛性和可复现性。
3. 缺乏对 **symbolic structure 的连续表示**，难以实现跨任务的知识迁移和引导式探索。

这些问题导致现有方法在大规模方程空间中扩展性差、鲁棒性弱、计算成本高。

---

### 🚀 提出的新方法：SRCO 框架

作者提出 **SRCO (Structure Embedding and Coefficient Optimization)** —— 一种基于连续结构嵌入的统一框架，将符号回归重构为 **连续结构优化问题**。其核心思想是 **解耦结构发现与系数优化**，分为三个阶段：

#### （1）**Structure Embedding（结构嵌入）**
- 使用 GP 工具（如 `gplearn`）生成大量候选表达式。
- 将所有数值常数抽象为统一占位符 `COF`（Coefficient Placeholder）。
- 将表达式转换为 **postfix notation (逆波兰表示)**，训练一个 **Transformer 模型** 学习结构分布，得到连续的 **structure embedding space** 和概率先验 $ p_\theta(S) $。

> ✅ 优势：学习到结构相似性，支持跨数据集知识复用。

#### （2）**Continuous Structure Search（连续结构搜索）**
- 利用训练好的 Transformer 作为生成模型，在 postfix 空间中进行 **temperature-controlled + top-k sampling**，生成候选结构。
- 通过语法解析器（stack-based parser）、语义规则和复杂度约束（如最大项数）过滤无效结构。
- 引入轻量级 **proxy scorer**（基于采样 log-prob）对候选结构排序。

> ✅ 优势：避免树结构的离散突变，实现平滑、高效的结构探索。

#### （3）**Coefficient Optimization（系数优化）**
- 对每个有效结构中的 `COF` 占位符替换为可学习参数 $ w_1, ..., w_m $。
- 在训练集上使用 **gradient-based optimization**（如梯度下降）最小化 MSE，拟合系数。
- 最终选择满足复杂度约束且测试性能最优的表达式。

> ✅ 优势：固定结构后，系数优化变为标准连续优化问题，稳定高效。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（GP/DSO等） | SRCO |
|------|------------------------|------|
| 结构表示 | 离散（token/tree） | 连续 embedding |
| 搜索方式 | 遗传变异 / token-by-token 生成 | 嵌入空间采样 + 约束生成 |
| 系数优化 | 与结构耦合或局部拟合 | 完全解耦，梯度优化 |
| 可扩展性 | 差（组合爆炸） | 更好（利用先验引导） |
| 复用能力 | 弱（每任务从头开始） | 强（共享结构先验） |

> ✅ **SRCO 首次将 representation learning 作为 SR 的基础组件**，而非辅助模块。

---

## 2. 核心实验方法和设置

### 📚 数据集

使用两个基于 **Feynman 物理方程** 的 benchmark：

1. **Feynman-synthetic**  
   - 合成数据：从真实方程采样输入输出对。
   - 函数集：$\{+, -, \times, \div, \sin, \cos\}$
   - 分为三难度等级：Easy (30 equations), Medium (40), Hard (50)，输入维度分别为 2/3/4。

2. **Feynman-real-world**  
   - 来自真实物理实验的数据集，更具噪声和现实挑战性。
   - 同样按难易分层，使用官方 train/test split。

> 所有方法仅在训练集上学习，报告测试集性能。

---

### ⚙️ 实验设置

- **Structure Corpus 构建**：
  - 对每个方程运行 `gplearn` 生成 1000 个候选表达式。
  - 抽象常数为 `COF`，转为 postfix 表示，用于训练 Transformer prior。

- **Transformer 模型**：
  - 序列建模 postfix tokens，autoregressive 训练最大化 log-likelihood。
  - 输出结构先验 $ p_\theta(S) $，用于采样。

- **Sampling 设置**：
  - 温度控制 + top-k sampling（$ k=50 $）
  - 控制超参数：`max_term`, `max_trig_vars` 等限制表达式复杂度。

- **Coefficient Optimization**：
  - 初始化随机权重，使用 Adam 优化 MSE。
  - 固定结构，独立优化系数。

---

### 📊 评估指标

| 指标 | 公式简述 | 目标 |
|------|---------|------|
| **MSE** | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$ | ↓ 越小越好 |
| **R²** | $1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$ | ↑ 越大越好 |
| **Pearson correlation (ρ)** | 真实值与预测值的相关系数 | ↑ 越大越好 |
| **log(MSE)** | 显示用，便于可视化比较 | ↓ 越小越好 |

> 报告 per-equation 和 tier-averaged 性能。

---

### 🆚 基线方法对比

| 方法 | 类型 | 特点 |
|------|------|------|
| **DSO** | Reinforcement Learning | 基于策略梯度生成表达式，平衡拟合与复杂度 |
| **FFX** | Sparse Regression | 确定性方法，搜索函数库并线性组合 |
| **EFS** | Evolutionary SR | 进化特征合成，迭代构造表达式 |
| **gplearn** | GP Baseline | 标准遗传编程实现 |

> 所有基线使用相同 train/test split 和 operator set，尽可能公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 整体性能对比（Tables 1 & 2）

| Model | R² (Easy) | R² (Medium) | R² (Hard) | log(MSE) ↓ |
|-------|-----------|-------------|-----------|------------|
| **SRCO (synthetic)** | **0.998** | **0.996** | **0.937** | **-30.4 ~ -65.8** |
| DSO | 0.598 | -0.601 | -6.18 | ~45–67 |
| FFX | 0.482 | 0.259 | 0.244 | ~43–68 |
| gplearn | 0.132 | -0.302 | -3.69e13 | ~45–67 |
| EFS | -0.646 | -10.19 | -38.91 | ~45–70 |

> ✅ **SRCO 在所有 tier 上显著优于所有 baseline**，尤其在中高难度下保持高 R²，而其他方法出现负 R²（拟合不如均值）。

---

### 🔍 消融实验：Coefficient Optimization（图2）

- 对比 **gradient-based fitting** vs **stochastic hill-climbing (random search)**。
- 固定结构模板、训练/测试集、优化预算。

| 方法 | R² | ρ | MSE |
|------|----|---|-----|
| Gradient Descent | **0.999** | **0.999** | **~0.001** |
| Random Search | 0.928 | 0.931 | ~0.016 |

> ✅ **梯度优化大幅提升精度**，说明可微系数拟合是关键优势。

---

### 🛡️ 鲁棒性测试：Noise Injection（Table 3）

向测试输入添加高斯噪声（比例 0%–100%），评估模型稳定性。

| Noise (%) | R² | ρ | log(MSE) |
|----------|-----|-----|----------|
| 0        | 0.999 | 0.999 | -24.5 |
| 10       | 0.997 | 0.998 | -23.7 |
| 50       | 0.974 | 0.980 | -20.4 |
| 100      | 0.901 | 0.901 | -12.7 |

> ✅ 性能随噪声单调下降但无崩溃，表明 **SRCO 具备良好泛化与鲁棒性**。

---

### ⏱️ 推理效率（Figure 3）

| 方法 | 平均推理时间 (秒) | 相对速度 |
|------|------------------|----------|
| **SRCO** | **0.00649** | 1× |
| EFS | 0.00651 | ~1× |
| DSO | 0.0171 | 2.6× 慢 |
| FFX | 0.0396 | 6.1× 慢 |
| gplearn | 0.250 | **38.5× 慢** |

> ✅ SRCO 不仅准确，而且 **推理最快**，适合部署。

---

### 🔧 参数敏感性分析（Figures 4–7）

- **max_term**：提升至 18–22 后趋于饱和（R² 从 0.972 → 0.997）
- **max_trig_vars**：超过 3–4 后增益有限

> ✅ 方法对超参不敏感，实用设置明确。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **连续结构嵌入可行且有效**：通过 Transformer 学习 postfix 表达式的 embedding，能够捕捉结构规律并指导高效搜索。
2. **解耦设计显著提升性能**：分离 structure discovery 与 coefficient optimization，使两者各司其职，提高稳定性与准确性。
3. **梯度优化系数至关重要**：相比随机搜索，gradient-based fitting 显著提升最终拟合质量。
4. **SRCO 在 accuracy、robustness、efficiency 上全面领先**：在多个 benchmark 上超越经典与神经 SR 方法。

---

### ⚠️ 局限性

1. **依赖 gplearn 构建初始语料库**：当前结构先验依赖传统 GP 生成的表达式池，若 GP 无法覆盖目标结构，则 embedding 空间受限。
2. **未完全摆脱采样机制**：结构搜索仍基于 sampling，尚未引入更主动的连续优化（如 latent space gradient ascent）。
3. **operator set 固定**：实验限定于基本算子，未验证更复杂函数库下的表现。
4. **benchmark 范围有限**：仅在 Feynman 系列上验证，缺乏更多领域（如化学、金融）的泛化测试。

---

### 🔮 未来工作方向

1. **减少对 GP 的依赖**：探索 self-supervised 或 curriculum learning 方式构建结构语料库。
2. **扩展搜索机制**：尝试在 embedding space 中直接优化结构（e.g., latent optimization）。
3. **多目标优化**：联合优化 accuracy、complexity、interpretability。
4. **更大规模 benchmark**：拓展至更高维、更复杂 operator 的数据集（如 AI Feynman v2+）。
5. **real-world 应用落地**：应用于科学发现、控制系统建模等领域。

---

## ✅ 总结

**SRCO 开创性地将 Symbolic Regression 从离散搜索范式转向连续优化范式**，通过：

- 学习 **continuous structure embedding**
- 实现 **embedding-guided structure search**
- 解耦 **gradient-based coefficient optimization**

在多个维度上实现了 **state-of-the-art 的性能突破**，为符号回归提供了新的研究范式。该工作标志着 **representation learning 正式成为 SR 的核心引擎**，具有重要的理论与应用价值。

</details>

---

### 14. [Double Coupling Architecture and Training Method for Optimization Problems of Differential Algebraic Equations with Parameters](https://arxiv.org/abs/2603.22724)

**Authors**: Wenqiang Yang, Wenyuan Wu, Yong Feng, Changbo Chen  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.22724v1  

#### Abstract
Simulation and modeling are essential in product development, integrated into the design and manufacturing process to enhance efficiency and quality. They are typically represented as complex nonlinear differential algebraic equations. The growing diversity of product requirements demands multi-task...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Double Coupling Architecture and Training Method for Optimization Problems of Differential Algebraic Equations with Parameters*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对**参数化微分代数方程（parametric DAEs）约束下的多任务优化问题**中存在的以下挑战：
- 传统方法（如变分法、直接配点法、分支定界等）在面对高维、强非线性系统时计算成本高昂，难以满足实时性和多目标切换需求；
- 基于PINN的方法虽能避免重复求解DAE，但存在：
  - 软约束导致约束违反（soft penalty导致feasibility差）；
  - 缺乏对参数退化（parametric degradation）的有效处理机制；
  - 每次目标函数变化需重新训练整个网络，效率低下。

### 🚀 提出的新方法与创新思路
作者提出了一种**双耦合物理信息神经网络架构（Dual-PINN）与混合训练策略**，核心创新如下：

#### （1）**双网络解耦架构（Dual-PINN Decoupling Strategy）**
- 将优化过程分解为两个阶段：
  - **Constraint Network (NN<sub>cnstr</sub>)**：离线训练，学习DAE系统的约束流形（solution manifold），即映射 $(t, p) \rightarrow x(t,p)$。
  - **Objective Network (NN<sub>obj</sub>)**：在线训练，仅生成候选参数 $p$，通过冻结的Constraint Network快速评估状态轨迹和目标值。
- 实现“一次训练，多次复用”，显著提升多任务场景下的响应速度。

#### （2）**嵌入式结构分析机制（Embedding-based Structural Analysis）**
- 引入文献[21]中的embedding方法，检测并正则化因Jacobian奇异引起的**参数退化现象**，确保训练数据的可靠性和物理一致性。

#### （3）**松弛变量与全局误差界保障等价性**
- 引入**松弛向量 $\gamma$** 构造带误差界的约束条件：
  $$
  \text{NN}_{\text{cnstr}}(t,p;\theta_c) - \gamma \leq x(t,p) \leq \text{NN}_{\text{cnstr}}(t,p;\theta_c) + \gamma
  $$
- 结合Theorem 4.1推导出紧致的**全局误差上界**，理论上保证了代理优化问题与原问题的**解集等价性**。

#### （4）**遗传算法增强的混合训练框架**
- 在Constraint Network训练中采用**基于遗传算法（GA）的自适应采样策略**：
  - 利用预测误差选择高误差区域的参数样本；
  - 使用交叉与变异生成新样本，动态扩充高质量精确解数据集 $N_B$；
  - 最终结合局部修正（Newton迭代或随机游走）提高精度。

---

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 避免每次目标变更都重新求解DAE或重训练模型；在线优化仅需轻量级Objective Network训练，提速1–2个数量级。 |
| **通用性** | 支持固定约束下任意目标函数的快速切换，适用于产品设计中的多任务优化场景。 |
| **鲁棒性** | GA采样聚焦难解区域，避免均匀采样的低效性；嵌入机制防止退化解影响训练质量。 |
| **理论保障** | 提供误差边界和解等价性证明，增强方法可信度。 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集与案例
实验涵盖三个典型参数化DAE系统：

1. **非线性ODE系统（Example 4.1）**  
   - 来源：[23]，用于验证基本有效性。
   - 方程形式：$\dot{x} = x^4 - 3x^2 - x + 0.4$, 初始条件含参数 $p$。
   - 目标：最小化终端代价 $J(p) = -3x(t_f,p)^3 + (1+p)x(t_f,p)$。

2. **生化反应系统（E. coli钾离子吸收模型）**  
   - 来源：[5]，真实生物调控系统，包含微分与代数方程。
   - 参数：$k_1, k_3$，目标是最小化模型输出与实验测量之间的偏差。

3. **高维线性DAE系统（Example 5.1）**  
   - 来源：[26]，用于测试可扩展性。
   - 形式：$\dot{x} = A x + I \cdot p$, 其中 $A$ 为双对角矩阵，维度从 $n=2$ 到 $n=10$ 变化。

---

### ⚙️ 实验设置
- **硬件平台**：Intel Core i7-14700KF CPU，32GB RAM。
- **网络结构**：
  - Constraint Network：全连接DNN，5–7层，每层128神经元。
  - Objective Network：输入为随机种子 $z \sim U[-1,1]$，输出经tanh变换映射到参数边界内。
- **归一化**：所有变量和参数标准化以提升数值稳定性。
- **误差容忍度**：GA训练中设定最大预测误差阈值 $\alpha = 10^{-3}$。

---

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Training Cost (s)** | Constraint Network离线训练时间 |
| **Prediction Cost (s)** | Objective Network在线推理耗时 |
| **Correction Cost (s)** | 局部优化（如Newton）精修时间 |
| **B&B Cost (s)** | 分支定界法（Branch-and-Bound）总耗时作为基线 |
| **Objective Value $J(p)$** | 最终优化目标值，越小越好 |
| **Prediction Error** | PINN预测解与真解的最大绝对误差 |

---

### 🆚 基线方法对比
- **New OB relaxation [23]** 和 **SBM relaxation [24]**：基于凸松弛的全局优化方法。
- **Method in [25]**：局部优化方法（如Kremling et al.）。
- **ε-global method [5]**：确定性全局优化算法。
- **Branch-and-Bound (B&B)**：经典全局搜索方法，用于性能基准。

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据汇总

#### （1）**非线性ODE示例（Example 4.1）**
| 方法 | 最优目标值 | 迭代次数 |
|------|------------|----------|
| New OB relaxation [23] | -0.0607 | 23 |
| SBM relaxation [24] | -0.0607 | 33 |
| **本文方法** | **-0.0607** | **6** |

> ➤ 仅用6次迭代即达到相同最优值，效率远超传统方法。

#### （2）**E. coli生化系统优化结果（Table 2）**
| 方法 | 参数 $(k_1, k_3)$ | 目标值 $J$ |
|------|------------------|-----------|
| ε-global method [5] | $(4.313\times10^{-3}, 162.93)$ | 0.029096 |
| Method in [25] | $(2.9\times10^{-3}, 90)$ | 0.0712 |
| **Our method (predicted)** | $(4.488\times10^{-3}, 172.31)$ | **0.0296** |
| **Our method (corrected)** | $(4.322\times10^{-3}, 163.25)$ | **0.029089** |

> ➤ 预测解已接近全局最优；经300步随机游走校正后进一步逼近，且**总耗时约1秒**，远低于B&B的52秒。

#### （3）**高维DAE系统性能对比（Table 3）**
| $n$ | Train (s) | Pred (s) | Corr (s) | B&B (s) | 预测误差 |
|-----|----------|---------|--------|--------|--------|
| 2 | 96.1 | 0.6 | 0.003 | 1.4 | ~1e-8 |
| 6 | 5991.9 | 1.5 | 0.005 | 90.6 | ~8e-7 |
| 8 | 18424.8 | 2.7 | 0.013 | 934.6 | ~1.6e-7 |
| 10 | 53472.4 (~15h) | 4.8 | 0.008 | 6083.5 (~1.7h) | ~2.9e-7 |

> ➤ **在线阶段（预测+校正）始终控制在5秒以内**，而B&B随维度指数增长；
> ➤ 即使在$n=10$时，预测误差仍保持在$10^{-7}$量级，精度极高。

---

### 🔬 消融实验与关键观察
- **GA采样 vs 均匀采样**：GA能有效识别误差大的参数区域（如$k_1≈0.01, k_3≈10$），加速收敛。
- **局部校正必要性**：初始预测已很接近最优，但加入Newton迭代或随机游走可进一步提升精度至12位有效数字。
- **误差界有效性**：使用 $\gamma = \max\{|E-3D|, |E+3D|\}$ 提供保守但可靠的置信区间，保障优化稳健性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **双网络架构实现了“约束-目标”解耦**，是实现高效多任务优化的关键突破。
2. **GA驱动的自适应采样显著提升了Constraint Network的泛化能力与训练效率**，尤其在复杂非线性区域表现优异。
3. **嵌入式结构分析有效应对参数退化问题**，增强了DAE系统建模的鲁棒性。
4. **全局误差界提供了理论保障**，使得PINN代理模型可用于可信优化决策。
5. **在线优化速度极快（通常<1秒）**，适合需要实时响应的产品设计与控制系统。

---

### ⚠️ 方法的局限性
1. **离线训练成本较高**：Constraint Network的训练时间随系统维度增加而快速增长（如$n=10$需约15小时），限制其在超大规模系统上的即时部署。
2. **依赖高质量DAE求解器**：需要一个高精度的数值求解器来生成训练用的“ground truth”数据，在某些病态系统中可能不可行。
3. **当前框架主要面向连续参数空间**：对于离散或混合参数优化的支持尚不明确。
4. **Objective Network的设计较简单**：目前仅为从随机种子到参数的映射，未显式建模目标景观结构。

---

### 🔮 未来工作方向
1. **引入增量学习或迁移学习机制**，降低高维系统的离线训练开销。
2. **探索无监督或弱监督方式构建Constraint Network**，减少对精确DAE求解器的依赖。
3. **将框架拓展至PDE-Constrained Optimization或多尺度系统建模**。
4. **集成不确定性量化模块（如Bayesian PINNs）**，提供更完整的置信度评估。
5. **应用于工业级仿真平台**，如电力系统、化工流程、自动驾驶控制等实际工程场景。

---

> **总结一句话**：  
> 本论文提出的**双耦合PINN架构 + GA增强训练 + 误差界保障机制**，为参数化DAE系统的多任务优化提供了**高效、准确、可理论验证**的新范式，推动了物理信息机器学习在科学计算与工程优化中的深度融合。

</details>

---

### 15. [Benchmarking Multi-Agent LLM Architectures for Financial Document Processing: A Comparative Study of Orchestration Patterns, Cost-Accuracy Tradeoffs and Production Scaling Strategies](https://arxiv.org/abs/2603.22651)

**Authors**: Siddhant Kulkarni, Yukta Kulkarni  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.22651v1  

#### Abstract
The adoption of large language models (LLMs) for structured information extraction from financial documents has accelerated rapidly, yet production deployments face fundamental architectural decisions with limited empirical guidance. We present a systematic benchmark comparing four multi-agent orche...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Benchmarking Multi-Agent LLM Architectures for Financial Document Processing*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对**金融文档处理中的结构化信息抽取任务**，系统性地解决了以下关键挑战：
- **多智能体 LLM 架构选择缺乏实证指导**：尽管多智能体系统被广泛用于提升 LLM 在复杂任务上的表现，但在实际生产中如何权衡不同架构的**准确性、成本、延迟和可扩展性**仍缺乏系统性的基准研究。
- **金融监管环境下的高精度与低成本矛盾**：金融机构需从大量 SEC filings（如 10-K、10-Q）中提取关键字段（如财务指标、治理结构、高管薪酬），对准确率要求极高，同时面临高昂的 LLM 调用成本。

### 提出的新方法与新思路
作者设计并比较了四种典型的 **multi-agent orchestration patterns**：
1. **Sequential Pipeline（顺序流水线）**
2. **Parallel Fan-Out with Merge（并行分发-合并）**
3. **Hierarchical Supervisor-Worker（分层监督-工作模式）**
4. **Reflexive Self-Correcting Loop（反射式自修正循环）**

此外，提出了一套**综合评估框架**，将 **cost、latency、token efficiency** 等生产级指标纳入评价体系，并进行了大规模消融实验（ablation studies）探索优化策略。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **评估全面性** | 首次在金融文档处理领域同时衡量 **accuracy、cost、latency、scaling behavior 和 failure modes**，超越传统仅关注 F1 分数的研究。 |
| **生产导向性强** | 引入 **amortized cost per document、token efficiency、throughput-accuracy tradeoff** 等实用指标，直接服务于企业部署决策。 |
| **架构创新组合** | 通过 **semantic caching、model routing、adaptive retry** 的混合配置，在接近顺序架构的成本下恢复了 89% 的反射式架构精度增益。 |

---

## 2. 核心实验方法和设置

### 数据集
- **来源**：SEC EDGAR 全文检索系统
- **规模**：共 **10,000 份** filings，按类型分层采样：
  - **10-K**：4,000 份（年报）
  - **10-Q**：4,000 份（季报）
  - **8-K**：2,000 份（重大事件报告）
- **预处理**：使用 `sec-edgar-downloader` 将 HTML/XBRL 转换为保留表格结构的纯文本。

### 提取字段（共 25 类）
| 域 | 字段示例 |
|----|--------|
| **Financial Metrics (10)** | total revenue, net income, EPS, debt-to-equity ratio |
| **Governance (8)** | board size, CEO duality, audit committee financial expert |
| **Executive Compensation (7)** | CEO total compensation, CEO pay ratio, stock awards |

### Ground Truth 构建
采用三阶段流程确保高质量标注：
1. **自动预标注**：利用 XBRL 标签填充约 60% 的财务字段；
2. **人工标注**：由 12 名具备 CFA/CPA 资质的专业人士完成；
3. **争议仲裁**：资深分析师裁决分歧项，Cohen’s Kappa = 0.91。

### 模型选择（5 种 LLM）
| Model | Provider | Context Window | Cost ($/1M tokens) |
|-------|--------|----------------|--------------------|
| GPT-4o | OpenAI | 128K | 2.50 / 10.00 |
| Claude 3.5 Sonnet | Anthropic | 200K | 3.00 / 15.00 |
| Gemini 1.5 Pro | Google | 1M | 1.25 / 5.00 |
| Llama 3 70B | Meta | 128K | 0.60 / 0.80 |
| Mixtral 8x22B | Mistral | 64K | 0.50 / 0.70 |

> 注：open-weight 模型部署于 4×A100 80GB 集群上，使用 vLLM 推理引擎。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Field-level F1** | 微平均 F1，数值容忍 ±2%，分类要求完全匹配 |
| **Document-level Accuracy** | 所有 25 字段均正确（strict）或最多错 2 个（relaxed）的比例 |
| **End-to-end Latency** | p50 和 p95 延迟（秒） |
| **Cost per Document** | 单文档总调用成本（美元） |
| **Token Efficiency** | 输出有效信息 token 数 / 总消耗 token 数 |

### 基线方法对比
以 **Sequential Pipeline** 作为基础 baseline，与其他三种 multi-agent 架构进行横向比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（以 Claude 3.5 Sonnet 为例）

| Architecture | Field F1 | Doc Acc (strict) | Latency (p50) | Cost ($/doc) |
|--------------|---------|------------------|---------------|---------------|
| **Sequential (A)** | 0.903 | 0.648 | 38.7 s | $0.187 |
| **Parallel (B)** | 0.914 | 0.672 | 21.3 s | $0.221 |
| **Hierarchical (C)** | **0.929** | **0.718** | 46.2 s | $0.261 |
| **Reflexive (D)** | **0.943** | **0.758** | 74.1 s | **$0.430** |

> ✅ Reflexive 架构达到最高准确率（F1=0.943），但成本是 Sequential 的 **2.3×**  
> ✅ Hierarchical 在成本-准确率帕累托前沿上最优：获得 **98.5% 的 Reflexive 准确率，仅花费其 60.7% 成本**

### 不同领域的性能差异（Hierarchical 架构）
| Domain | Mean F1 |
|--------|--------|
| Financial Metrics | 0.931 |
| Governance | 0.881 |
| Executive Compensation | 0.856 |

> 💡 财务指标因 GAAP 标准化格式最易抽取；高管薪酬因 vesting schedule 复杂最难。

### 消融实验结果

#### （1）Semantic Caching（语义缓存）
| 配置 | F1 | Cost ($/doc) | Hit Rate |
|------|-----|-------------|---------|
| 无缓存 | 0.929 | $0.261 | — |
| 字段级缓存 | 0.924 | $0.171 | 38.7% |
| 自适应混合缓存 | 0.926 | $0.182 | 31.4% |

> ⬇️ 缓存带来显著降本（↓34.5%），轻微影响精度。

#### （2）Model Routing（模型路由）
| 策略 | F1 | Cost ($/doc) |
|------|-----|-------------|
| 全部使用 Claude 3.5 Sonnet | 0.929 | $0.261 |
| 两层路由（Claude + Mixtral） | 0.912 | $0.127 |
| 三层路由（Claude + GPT-4o + Mixtral） | 0.918 | $0.143 |

> 🎯 可将简单字段交给廉价模型（如 Mixtral），节省超 50% 成本，保留近 98% 精度。

#### （3）Adaptive Retry（自适应重试）
| 策略 | F1 | Cost ($/doc) |
|------|-----|-------------|
| 无重试 | 0.908 | $0.214 |
| 固定重试一次 | 0.929 | $0.261 |
| 模型升级重试（escalation） | **0.931** | $0.258 |
| 自适应阈值控制 | 0.929 | $0.247 |

> 🔁 “升级到更强模型”策略可进一步提点。

#### （4）联合优化方案：**Hierarchical-Optimized**
结合上述三项技术后的最终推荐配置：

| Configuration | F1 | Cost ($/doc) | Latency |
|---------------|-----|-------------|--------|
| Sequential Baseline | 0.903 | $0.187 | 38.7 s |
| Hierarchical Baseline | 0.929 | $0.261 | 46.2 s |
| Reflexive Baseline | 0.943 | $0.430 | 74.1 s |
| **Hierarchical-Optimized** | **0.924** | **$0.148** | **30.2 s** |

> ✅ **恢复了 89% 的 Reflexive 相对于 Sequential 的精度增益，成本仅为 Sequential 的 1.15×！**

---

## 4. 关键结论和发现

### 主要发现
1. **Hierarchical 架构是最佳性价比选择**  
   在所有架构中占据 **cost-accuracy Pareto frontier** 的主导位置，适合大多数生产场景。

2. **Reflexive 架构虽准但不可扩展**  
   在低吞吐（<25K docs/day）时表现最好，但随着负载增加，其迭代机制导致排队延迟，**在 100K/day 时反成最差**。

3. **Scaling 行为非线性且架构依赖**  
   - Reflexive：F1 下降最快（timeout 截断 correction loop）
   - Sequential：最具 scale resilience（确定性执行）
   - 存在“交叉点”：约 75K docs/day 时 Hierarchical 被 Parallel 超越

4. **失败模式具有架构特异性**
| 架构 | 主要失败模式 |
|------|------------|
| Sequential | Cross-table reference failure (28.4%) |
| Parallel | Context window truncation (14.7%) |
| Hierarchical | Ambiguous disclosure resolution (15.4%) |
| Reflexive | Ambiguous disclosure resolution (**39.3%**) |

> ❗ Reflexive 的“过度思考”问题：面对模糊披露会反复震荡解释，反而降低稳定性。

5. **Token Efficiency 并不随总消耗单调变化**  
   Hierarchical 与 Reflexive 的 token efficiency 同为 **2.78%**，高于 Sequential（2.61%），表明其更高的输出质量补偿了输入开销。

---

### 方法的局限性
1. **语言与地域限制**：仅测试英文 SEC filings，未涵盖 IFRS 或其他司法管辖区文件。
2. **定价动态性**：API 成本基于 2025 年初数据，未来价格变动会影响具体 cost ratio。
3. **字段泛化能力未知**：仅覆盖 25 类固定字段，对新字段的迁移能力未验证。
4. **模型漂移未考虑**：未分析 API 模型随时间的行为退化（model drift）。
5. **Ground Truth 偏向性可能**：专家标注在模糊案例中可能存在主观偏好。

---

### 未来工作方向
1. **Dynamic Architecture Switching**  
   根据文档复杂度动态选择架构（简单文档走 Sequential，复杂走 Reflexive）。

2. **Fine-tuned Specialist Models**  
   使用 distillation 技术训练小型专用模型（如 compensation extractor），替代昂贵通用 LLM。

3. **Streaming Processing Architecture**  
   对长文档（如 10-K）实现边解析边抽取，减少端到端延迟 40–60%。

4. **Cross-Document Reasoning**  
   利用公司历史 filings 辅助当前期报表理解，缓解 temporal confusion 错误。

5. **Formal Verification Layer**  
   添加硬性规则校验（如资产负债恒等式），提供部分字段的**可证明正确性保证**。

--- 

> 📌 **一句话总结**：  
> 本文提供了迄今为止最系统的 **multi-agent LLM 架构 benchmark**，揭示了 **Hierarchical + Hybrid Optimization** 是金融文档处理中最实用的部署路径，在保持高精度的同时实现了极致的成本控制。

</details>

---

### 16. [RelayS2S: A Dual-Path Speculative Generation for Real-Time Dialogue](https://arxiv.org/abs/2603.23346)

**Authors**: Long Mai  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.23346v1  

#### Abstract
Real-time spoken dialogue systems face a fundamental tension between latency and response quality. End-to-end speech-to-speech (S2S) models respond immediately and naturally handle turn-taking, backchanneling, and interruption, but produce semantically weaker outputs. Cascaded pipelines (ASR -> LLM)...

---

### 17. [HGNet: Scalable Foundation Model for Automated Knowledge Graph Generation from Scientific Literature](https://arxiv.org/abs/2603.23136)

**Authors**: Devvrat Joshi, Islem Rekik  
**Category**: cs.CL  
**Published**: 2026-03-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.23136v1  

#### Abstract
Automated knowledge graph (KG) construction is essential for navigating the rapidly expanding body of scientific literature. However, existing approaches struggle to recognize long multi-word entities, often fail to generalize across domains, and typically overlook the hierarchical nature of scienti...

---

### 18. [Communication-Aware Diffusion Load Balancing for Persistently Interacting Objects](https://arxiv.org/abs/2603.23329)

**Authors**: Maya Taylor, Kavitha Chandrasekar, Laxmikant V. Kale  
**Category**: cs.DC  
**Published**: 2026-03-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.23329v1  

#### Abstract
Parallel applications with irregular and time-varying workloads often suffer from load imbalance. Dynamic load balancing techniques address this challenge by redistributing work during execution. We present a new type of distributed diffusion-based load balancing targeted at communication-intensive ...

---

### 19. [SkillRouter: Retrieve-and-Rerank Skill Selection for LLM Agents at Scale](https://arxiv.org/abs/2603.22455)

**Authors**: YanZhao Zheng, ZhenTao Zhang, Chao Ma, YuanQiang Yu, JiHuan Zhu, Baohua Dong, Hangcheng Zhu  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.22455v1  

#### Abstract
As LLM agent ecosystems grow, the number of available skills (tools, plugins) has reached tens of thousands, making it infeasible to inject all skills into an agent's context. This creates a need for skill routing -- retrieving the most relevant skills from a large pool given a user task. The proble...

---

### 20. [GEM: Guided Expectation-Maximization for Behavior-Normalized Candidate Action Selection in Offline RL](https://arxiv.org/abs/2603.23232)

**Authors**: Haoyu Wang, Jingcheng Wang, Shunyu Wu, Xinwei Xiao  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.23232v1  

#### Abstract
Offline reinforcement learning (RL) can fit strong value functions from fixed datasets, yet reliable deployment still hinges on the action selection interface used to query them. When the dataset induces a branched or multimodal action landscape, unimodal policy extraction can blur competing hypothe...

---

### 21. [Similarity-Aware Mixture-of-Experts for Data-Efficient Continual Learning](https://arxiv.org/abs/2603.23436)

**Authors**: Connor Mclaughlin, Nigel Lee, Lili Su  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.23436v1  

#### Abstract
Machine learning models often need to adapt to new data after deployment due to structured or unstructured real-world dynamics. The Continual Learning (CL) framework enables continuous model adaptation, but most existing approaches either assume each task contains sufficiently many data samples or t...

---

### 22. [Intelligence Inertia: Physical Principles and Applications](https://arxiv.org/abs/2603.22347)

**Authors**: Jipeng Han  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.22347v1  

#### Abstract
While Landauer's principle establishes the fundamental thermodynamic floor for information erasure and Fisher Information provides a metric for local curvature in parameter space, these classical frameworks function effectively only as approximations within regimes of sparse rule-constraints. They f...

---

### 23. [Session Risk Memory (SRM): Temporal Authorization for Deterministic Pre-Execution Safety Gates](https://arxiv.org/abs/2603.22350)

**Authors**: Florin Adrian Chitan  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.22350v1  

#### Abstract
Deterministic pre-execution safety gates evaluate whether individual agent actions are compatible with their assigned roles. While effective at per-action authorization, these systems are structurally blind to distributed attacks that decompose harmful intent across multiple individually-compliant s...

---

### 24. [UniDial-EvalKit: A Unified Toolkit for Evaluating Multi-Faceted Conversational Abilities](https://arxiv.org/abs/2603.23160)

**Authors**: Qi Jia, Haodong Zhao, Dun Pei, Xiujie Song, Shibo Wang, Zijian Chen, Zicheng Zhang, Xiangyang Zhu, Guangtao Zhai  
**Category**: cs.CL  
**Published**: 2026-03-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.23160v1  

#### Abstract
Benchmarking AI systems in multi-turn interactive scenarios is essential for understanding their practical capabilities in real-world applications. However, existing evaluation protocols are highly heterogeneous, differing significantly in dataset formats, model interfaces, and evaluation pipelines,...

---

### 25. [A Foundation Model for Instruction-Conditioned In-Context Time Series Tasks](https://arxiv.org/abs/2603.22586)

**Authors**: Anish Saha, Konstantin Shmakov  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.22586v1  

#### Abstract
In-context learning (ICL) allows a model to adapt at inference time by conditioning on examples rather than updating parameters. Existing time-series foundation models use implicit positional context, retrieval, or task-specific objectives, but rarely explicit instruction-conditioned demonstrations....

---

### 26. [Towards The Implicit Bias on Multiclass Separable Data Under Norm Constraints](https://arxiv.org/abs/2603.22824)

**Authors**: Shengping Xie, Zekun Wu, Quan Chen, Kaixu Tang  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.22824v1  

#### Abstract
Implicit bias induced by gradient-based algorithms is essential to the generalization of overparameterized models, yet its mechanisms can be subtle. This work leverages the Normalized Steepest Descent} (NSD) framework to investigate how optimization geometry shapes solutions on multiclass separable ...

---

### 27. [Policy-based Tuning of Autoregressive Image Models with Instance- and Distribution-Level Rewards](https://arxiv.org/abs/2603.23086)

**Authors**: Orhun Bu\u{g}ra Baran, Melih Kandemir, Ramazan Gokberk Cinbis  
**Category**: cs.LG  
**Published**: 2026-03-25  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.23086v1  

#### Abstract
Autoregressive (AR) models are highly effective for image generation, yet their standard maximum-likelihood estimation training lacks direct optimization for sample quality and diversity. While reinforcement learning (RL) has been used to align diffusion models, these methods typically suffer from o...

---

### 28. [The Efficiency Attenuation Phenomenon: A Computational Challenge to the Language of Thought Hypothesis](https://arxiv.org/abs/2603.22312)

**Authors**: Di Zhang  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.22312v1  

#### Abstract
This paper computationally investigates whether thought requires a language-like format, as posited by the Language of Thought (LoT) hypothesis. We introduce the ``AI Private Language'' thought experiment: if two artificial agents develop an efficient, inscrutable communication protocol via multi-ag...

---

### 29. [ABSTRAL: Automatic Design of Multi-Agent Systems Through Iterative Refinement and Topology Optimization](https://arxiv.org/abs/2603.22791)

**Authors**: Weijia Song, Jiashu Yue, Zhe Pang  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.22791v1  

#### Abstract
How should multi-agent systems be designed, and can that design knowledge be captured in a form that is inspectable, revisable, and transferable? We introduce ABSTRAL, a framework that treats MAS architecture as an evolving natural-language document, an artifact refined through contrastive trace ana...

---

### 30. [Ran Score: a LLM-based Evaluation Score for Radiology Report Generation](https://arxiv.org/abs/2603.22935)

**Authors**: Ran Zhang, Yucong Lin, Zhaoli Su, Bowen Liu, Danni Ai, Tianyu Fu, Deqiang Xiao, Jingfan Fan, Yuanyuan Wang, Mingwei Gao, Yuwan Hu, Shuya Gao, Jingtao Li, Jian Yang, Hong Song, Hongliang Sun  
**Category**: cs.AI  
**Published**: 2026-03-25  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.22935v1  

#### Abstract
Chest X-ray report generation and automated evaluation are limited by poor recognition of low-prevalence abnormalities and inadequate handling of clinically important language, including negation and ambiguity. We develop a clinician-guided framework combining human expertise and large language mode...

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
