# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-09 08:54:50 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Joint Structural Pruning and Mixed-Precision Quantization for LLM Compression](https://arxiv.org/abs/2606.07819)

**Authors**: Hoang-Loc La, Truong-Thanh Le, Amir Taherkordi, Phuong Hoai Ha  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2606.07819v1  

#### Abstract
Recently, the efficiency of Large Language Models (LLMs) deployment has become a critical concern in practical applications. While post-training quantization (PTQ) and structural pruning are established techniques for reducing memory footprint and inference latency, most existing PTQ approaches opti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Joint Structural Pruning and Mixed-Precision Quantization for LLM Compression*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大语言模型（LLMs）在部署时面临**高内存占用**和**推理延迟**的挑战。虽然已有 **post-training quantization (PTQ)** 和 **pruning** 技术用于压缩模型，但存在以下关键问题：

- **传统 PTQ 方法**通常基于每层独立优化量化误差（如使用 Hessian 或激活统计），忽略了**全局误差传播**，导致次优解。
- 多数方法将 **pruning 与 quantization 分开处理**（顺序执行或孤立优化），未能联合建模二者之间的非正交性（non-orthogonal）关系。
- 现有联合压缩方法多采用**非结构化稀疏**（unstructured sparsity），难以在标准 GPU 上实现实际加速。

---

### 🚀 提出的新方法与新思路

本文提出一个端到端的联合优化框架 —— **Train-Once-Get-All (TOGA)**，其核心创新如下：

#### （1）**全局感知的混合精度 PTQ（Mixed-Precision PTQ）**
- 将 bit-width 分配建模为 **binary mask 优化问题**，通过一个可训练的 **hypernetwork** 学习每个线性层中“重要”权重通道的掩码。
- 不同于固定阈值划分 salient/non-salient 权重，TOGA 利用 **end-to-end language modeling loss** 动态识别敏感权重，从而最小化**整个网络中的误差累积与传播**。

#### （2）**首个支持结构化剪枝 + 混合精度量化的联合框架**
- 在统一搜索空间内同时优化 **structured pruning 决策** 和 **mixed-precision quantization policy**。
- 延续并扩展了 DISP-LLM 的 binary mask 范式，首次将其应用于联合 pruning + MPQ 场景。
- 支持灵活的 sparsity-precision trade-off（例如：45% sparsity + W3A3 vs. 59% sparsity + W4A4），无需预设比例。

#### （3）定制化 CUDA kernel 加速推理
- 开发了支持 **混合精度矩阵乘法**（如 W4A4 + W8A8）的高效 CUDA kernels，基于 CUTLASS 实现。
- 对 RMSNorm 层进行融合优化，并对 KV Cache 进行 INT4 量化以进一步降低内存开销。

---

### 🔍 相比现有方法的优势

| 维度 | TOGA 的优势 |
|------|-------------|
| **优化目标** | 全局 end-to-end loss 驱动，而非局部 layer-wise metric，避免误差累积放大 |
| **灵活性** | 自适应决定每层的 salient weight 数量，而非统一阈值 |
| **硬件友好性** | 输出为 dense-compatible 结构化稀疏模式，可在标准 GPU 上高效运行 |
| **性能表现** | 显著优于 SoTA 的 weight-only / weight-activation PTQ 及 joint pruning+quantization 方法 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练/校准数据集**：`WikiText-2`（128 个样本用于通道重排序）
- **评估数据集**：
  - **Perplexity 测试**：`WikiText-2`, `C4`
  - **Zero-shot 推理能力测试**：ARC Easy/Challenge, BoolQ, Winogrande, Hellaswag, MMLU（共 6 项基准）

### ⚙️ 实验设置
- **模型范围**：涵盖多个主流 LLMs：
  - Llama 系列：Llama-3.2-1B, Llama-3.2-3B, Llama-2-7B, Llama-2-13B, Llama-3-8B, Llama-3.1-8B
  - Mistral-7B-v0.3, Qwen-3-8B
- **硬件平台**：
  - Hypernetwork 训练：单张 NVIDIA A100 GPU（80GB VRAM）
  - 性能评测：NVIDIA L40 GPU，上下文长度 2048，batch size 变化（1–16）
- **训练配置**：
  - Batch size = 1
  - TOGA-q（仅量化）：2,000 步；TOGA（联合优化）：10,000 步
  - 每组实验重复 5 次取平均

### 📊 评估指标
| 指标类型 | 具体指标 |
|--------|---------|
| **模型质量** | Perplexity（越低越好）、Zero-shot Accuracy（越高越好） |
| **效率指标** | Prefill 阶段延迟、Decode 阶段峰值内存（Peak Memory Usage） |
| **压缩程度** | Compression Budget（压缩后理论内存 / FP16 原始内存） |

---

### 🆚 基线方法对比

#### （1）**纯量化方法（PTQ-only）**
- **Weight-activation MPQ**：
  - Atom [30]、ResQ [23]、SpinQuant [20]
- **Weight-only MPQ**：
  - PTQ-1.61 [29]（含/不含 LoRA 预处理）、Slim-LLM [17]、BiLLM [16]

#### （2）**联合剪枝+量化方法**
- **Sequential Pipeline**：
  - DISP-LLM（结构化剪枝）+ Atom / ResQ / BiLLM / Slim-LLM（后续量化）
- **Joint Methods**：
  - SparseGPT+GPTQ [7]
  - OBR [10]（支持 error compensation 的半结构化方法）

> 注：所有方法均在相同设置下复现或公平比较，KV Cache 统一量化至 INT4。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 指标 | TOGA 表现 | 对比提升 |
|------|----------|---------|
| **WikiText-2 Perplexity（超低位宽）** | ↓ 最多 **21%** vs. SoTA weight-activation PTQ | 显著优于 Atom、ResQ |
| **C4 Perplexity（weight-only setting）** | ↓ **59%**（WikiText）、↓ **85%**（C4） vs. SoTA weight-only PTQ | 极低位宽下仍保持稳定 |
| **Zero-shot Reasoning 平均准确率** | ↑ 最多 **5.4%** | 超越所有基线 |
| **Prefill Speedup（vs. FP16）** | ↑ **2×** | 明显快于 OBR（约 1.3×） |
| **Decode Peak Memory Reduction（vs. FP16）** | ↓ **6.5×** | 内存瓶颈显著缓解 |
| **vs. OBR（2:4 semi-structured）** | Prefill 快 **30%**，Peak Memory ↓ **10%** | 更高效的硬件利用 |

---

### 🔬 与基线方法的详细对比

#### （1）**在 ultra-low precision（<4-bit）下的表现**

| 方法 | WikiText-2 PPL（Llama-2-7B） | Avg. Zero-shot Acc. |
|------|-------------------------------|---------------------|
| ResQ (W3.2A3.2) | 7.35 | 52.63% |
| **TOGA-q (W3.2A3.2)** | **7.30** | **54.20%** |
| PTQ-1.61+ (W1.6A16) | 12.70 | 41.43% |
| **TOGA-q (W1.6A16)** | **11.00** | **46.80%** |

✅ TOGA-q 在更低 bit-width 下实现了更优的语言建模能力和推理准确性。

#### （2）**在主流 INT4/INT8 设置下的表现**

| 方法 | Llama-3.2-1B Zero-shot Acc. | Llama-3.2-3B | Llama-2-7B | Llama-3-8B | Average |
|------|-----------------------------|--------------|------------|------------|---------|
| SpinQuant (W4A4) | 46.4% | 56.8% | 60.1% | 65.8% | ~57.3% |
| ResQ | 48.2% | 60.1% | 61.5% | 66.8% | ~59.2% |
| **TOGA-q** | **49.2%** | **60.7%** | **62.5%** | **67.2%** | **~60.0%** |

➡️ 所有模型上均达到 SoTA 水平，且无需额外 fine-tuning。

#### （3）**联合 pruning + quantization 对比（图 2）**

- 在不同 compression budget（0.04–0.18）下，**TOGA 始终优于 sequential pipeline（DISP-LLM + PTQ）及 SparseGPT+GPTQ**。
- 即使约束为固定 sparsity（TOGA-fixed-sparsity），仍优于其他方法。
- **Unconstrained TOGA 表现出更优的 Pareto frontier**，说明自由组合 sparsity 与 precision 更有效。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）**量化技术组件分析（Table 3）**

| 技术 | WikiText-2 PPL | C4 PPL |
|------|----------------|--------|
| W4A4 RTN（baseline） | 1753 | 2301 |
| + 12.5% INT8 保留 | 6.03 | 8.10 |
| + 通道重排序（reordering） | 5.78 | 8.01 |
| + GPTQ 微调 | **5.38** | **7.47** |
| + KV Cache INT4 量化 | 5.48 | 7.68 |

📌 结论：
- **保留部分 INT8 权重** 是关键；
- **channel reordering + GPTQ** 显著提升效果；
- **KV Cache 量化影响较小**，适合启用以节省内存。

#### （2）**salient channel 分布可视化（Figure 4）**

- TOGA-q 自动学习到：**前几层和最后几层分配更多 salient weights**，中间层较少。
- 与已有研究一致（early/final layers 更重要），而 Atom/ResQ 使用统一阈值，缺乏这种动态感知能力。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **全局 end-to-end loss 驱动的 mixed-precision quantization 显著优于局部 layer-wise 方法**，尤其在 ultra-low bit（1–3bit）场景下。
2. **pruning 与 quantization 应联合优化**，顺序执行会导致性能损失；TOGA 的 co-optimization 实现了更优的 accuracy-efficiency 权衡。
3. **结构化稀疏 + 混合精度量化是实用部署的理想路径**：既保证硬件兼容性，又获得显著加速与内存缩减。
4. **adaptive saliency detection（通过 hypernetwork）比手工设定阈值更合理**，能自动捕捉各层敏感性差异。

---

### ⚠️ 方法的局限性

- **GPU 内存需求高**：需加载完整 LLM 运行 hypernetwork 训练，目前最大支持 ~32B 参数模型（在 80GB A100 上）。
- **不适用于极端大规模模型（如 70B+）**：受限于 OOM 问题，无法直接应用。
- **依赖 calibration data**：虽未 fine-tune，但仍需少量数据（如 WikiText-2）进行通道重排序等预处理。

---

### 🔮 未来工作方向

- 引入 **分布式训练或 offloading 技术**，降低 hypernetwork 训练过程中的内存消耗，以支持百亿级以上模型。
- 探索 **layer-wise decomposition 或模块化训练策略**，减少对全模型加载的依赖。
- 扩展至 **multi-modal models** 或 **long-context LLMs** 中的应用。

---

## ✅ 总结一句话

> 本文提出了 **TOGA** —— 首个支持 **joint structured pruning 与 mixed-precision PTQ** 的端到端框架，通过 **hypernetwork 驱动的 binary mask 优化**，在超低位宽下实现了 SoTA 的 perplexity 与 zero-shot 准确率，同时借助定制 kernel 达成高达 **2× prefill speedup** 与 **6.5× memory reduction**，为资源受限设备上的高效 LLM 部署提供了强有力解决方案。

</details>

---

### 2. [A Multi-Agent System for IPMSM Design Optimization via an FEA-AI Hybrid Approach](https://arxiv.org/abs/2606.09037)

**Authors**: Jinseong Han, Sunwoong Yang, Namwoo Kang  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2606.09037v1  

#### Abstract
Interior permanent magnet synchronous motor (IPMSM) design requires balancing conflicting objectives and multi-physics constraints, while modern optimization workflows face three bottlenecks: manual problem setup, high finite element analysis (FEA) cost, and unreliable surrogate-based search in spar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

**论文标题**: *A Multi-Agent System for IPMSM Design Optimization via an FEA-AI Hybrid Approach*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Interior Permanent Magnet Synchronous Motor (IPMSM)** 设计优化中的三大瓶颈问题：
1. **问题定义的手动负担重**：工程师需手动设定设计变量、目标函数、约束条件等，尤其对初级工程师而言难度大。
2. **计算成本高昂**：基于 **Finite Element Analysis (FEA)** 的优化需要大量高保真仿真，尤其在遗传算法（GA）搜索中，每一代候选几何体都需独立运行 FEA，导致计算资源消耗巨大。
3. **AI代理模型可靠性低**：虽然使用 **AI-surrogate model** 可大幅减少评估时间，但在稀疏或分布外（Out-of-Distribution, OOD）区域预测不确定性高，可能导致搜索陷入低置信度最优解。

### 提出的新方法与新思路
提出一个**端到端自动化多智能体系统**，结合 **Retrieval-Augmented Generation (RAG)** 和 **不确定性感知的 FEA-AI 混合优化管道**，实现从需求输入到最终设计推荐的全流程自动化：

- **Design Agent**：通过连接电机教科书的 RAG 模块，提供领域知识支持，引导用户完成结构化的问题定义，并生成优化卡片（optimization card）和 DOE（Design of Experiments）计划。
- **Training Agent**：自动化执行电磁 FEA，记录几何验证失败和求解器失败日志，利用 ANOVA 分析失败模式，并通过 LLM 推理调用 **Design Sampling Agent** 自主重构设计空间并补充采样点。
- **Optimization Agent**：执行基于 GA 的混合搜索，采用**不确定性驱动的切换机制**：
  - 低不确定性候选由 AI-surrogate 快速评估；
  - 高不确定性或帕累托前沿（Pareto-front）、Top-K 候选则由高保真 FEA 校正，并用于迭代再训练。

### 相比现有方法的优势
- **自动化程度高**：将传统依赖经验的手动配置转化为可复现的工作流。
- **平衡效率与可靠性**：相比纯 FEA 方法更高效，相比纯 AI 方法更可靠。
- **主动恢复可行性样本**：通过日志分析与 LLM 推理实现自动重采样，解决因宽泛设计空间导致的大量无效几何体问题，这是此前框架未涉及的能力。
- **本地部署友好**：采用轻量级开源 LLM（GPT-OSS 20B），支持本地部署，保障工业项目的数据保密性。

---

## 2. 核心实验方法和设置

### 数据集
- **训练数据来源**：通过 **Ansys Maxwell 2D** 对参数化 IPMSM 几何体进行电磁 FEA 仿真生成。
- **设计变量**：共 12 个几何参数（如 PM 角度、长度、气隙、定子齿高等）。
- **目标函数**：以最小化 **iron loss（铁损）** 为单目标优化任务。
- **有效样本获取**：初始采样失败率高达 72%，通过提出的 **log-informed resampling loop** 最终获得 **100 个有效的分析可行样本**用于训练。

### 实验设置与评估指标
- **硬件环境**：AMD Ryzen 9 7900 CPU。
- **随机种子**：重复实验于四个不同随机种子（7, 42, 123, 2026），结果取平均。
- **Surrogate 模型架构**：
  - 深度集成模型（Deep Ensemble）：5 个并行 MLP（12-64-64-2 结构）。
  - 使用 **B-NLL loss** 进行异方差不确定性训练。
  - 输入归一化，早停机制防止过拟合。
- **GA 设置**：
  - 种群大小：25
  - 代数：30
  - 算子：SBX 交叉，多项式变异
- **不确定性衡量**：使用 **Coefficient of Variation (CV)**：
  $$
  \text{CV}(x) = \frac{\sqrt{\sigma^2(x)}}{|\mu(x)|} \times 100\%
  $$

### 基线方法对比
在相同高保真 FEA 调用预算（150 次）下比较三种策略：
| 方法 | 描述 |
|------|------|
| **FEA-only GA** | 所有评估均使用 FEA，无代理模型加速 |
| **AI-only GA** | 使用 150 个 FEA 样本训练一次 surrogate，后续全部使用代理评估，无在线校正 |
| **Proposed Hybrid GA** | 使用 100 个样本训练初始 surrogate，剩余 50 次 FEA 预算用于不确定性触发的在线校正（阈值 CV=3%） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | 最优铁损 (kW) | 总计算时间 (h) | Top-K 平均不确定性 (CV%) | FEA 调用次数 |
|------|----------------|------------------|----------------------------|---------------|
| FEA-only GA | 1.6780 | 7.11 | ~0%（确定性） | 150 |
| AI-only GA | 1.8098 | 7.12 | 8.5% | 150（仅用于训练） |
| **Proposed Hybrid GA** | **1.6658** | **7.12** | **5.1%** | **150（100+50）** |

> ✅ **Hybrid GA 达到了最低铁损，优于其他两种方法**

### 与基线方法的对比结果
- **vs FEA-only GA**：
  - 在相同 FEA 预算下，FEA-only GA 因每代都需完整 FEA 评估，在第 8 代左右即耗尽预算，无法充分探索设计空间。
  - Hybrid GA 利用 surrogate 加速低风险评估，实际探索了约 **1,200 个候选设计**，远超 FEA-only 的探索能力。
- **vs AI-only GA**：
  - AI-only GA 缺乏在线 FEA 校正，导致搜索收敛至低置信度最优解（uncertainty 高达 8.5%）。
  - Hybrid GA 通过选择性 FEA 校正关键候选，显著提升结果可信度。

### 消融实验与敏感性分析
- **切换阈值 CVTh,hybrid 敏感性分析**（{1%, 3%, 5%, 10%}）：
  - **1% 阈值**：最保守，FEA 调用最多（~226 次），性能最好（1.5676 kW），但效率最低（8.41 h）。
  - **3% 阈值**：最佳权衡点，性能接近 1%，但 FEA 成本降低至 170.5 次，总时间降至 6.33 h，不确定性保持低位（~2.1%）。
  - **5%/10% 阈值**：FEA 调用迅速归零，搜索退化为近似 AI-only 模式，性能下降明显（1.7396 kW / 1.9366 kW），不确定性飙升至 6.97%。
- **结论**：**3% 是当前案例下的最优阈值**，实现了性能、可靠性与效率的最佳平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **RAG 显著提升问题定义质量**：
   - 与无 RAG 的通用 LLM 相比，RAG 支持的设计代理在目标函数、变量建议、工程提示等方面更具物理一致性与实用性。
   - 定量测试显示，RAG 将 GPT-OSS 20B 在专业问题上的准确率从 43%（数值题）提升至 77%，概念题从 47% 提升至 80%。

2. **Log-informed Resampling 有效恢复可行性样本**：
   - 初始设计空间下仅 28% 样本能通过几何验证。
   - 通过 ANOVA + 日志分析 + LLM 推理，系统自主缩小高风险变量范围，经过 3 轮迭代成功将有效样本率提升至 **84%**，最终获得 100 个有效样本。

3. **Hybrid GA 在固定预算下表现最优**：
   - 在相同 150 次 FEA 预算下，Hybrid GA 同时实现了**最佳性能**（最低铁损）和**可控的不确定性水平**。
   - 其优势在于“**用 FEA 精确锚定关键点，用 AI 加速常规搜索**”。

4. **不确定性随轮次递减**：
   - 表 2 显示，随着每轮加入高不确定性 FEA 标签并重新训练 surrogate，Top-K 不确定性从 9.32% 下降至 2.74%，且每轮 FEA 调用数也持续下降，体现主动学习的有效性。

### 方法的局限性
- 当前研究局限于**参数化几何表示**和**单目标优化**（铁损最小化）。
- 多物理场约束（热、应力、振动、制造性）尚未完全集成。
- 切换阈值 CVTh,hybrid 为固定值，未实现动态自适应调整。
- RAG 的效果依赖于检索语料库的质量与覆盖范围。

### 未来工作方向
1. 扩展至**非参数化或拓扑优化**空间。
2. 支持**多目标 Pareto 优化**（如同时优化转矩、损耗、纹波、效率）。
3. 集成**多物理场 FEA 自动化**与约束验证。
4. 开发**自适应切换策略**，根据 UQ 信号、GA 进展、预算状态动态调整阈值。
5. 扩展 RAG 知识库，纳入工业设计规则、制造指南、多物理场约束等实用信息。

---

> **总体结论**：本文提出的多智能体 FEA-AI 混合框架，成功将 RAG 引导的问题定义、日志驱动的自动重采样、不确定性感知的混合优化整合为一个**可复现、高效且可靠的端到端 IPMSM 设计优化流程**，为电动车辆与工业驱动领域的高性能电机设计提供了强有力的自动化工具。

</details>

---

### 3. [From Rigid to Dynamic: Entropy-Guided Adaptive Inference for Long-Context LLMs](https://arxiv.org/abs/2606.09508)

**Authors**: Zhanchao Xu, Haoyang Li, Qingfa Xiao, Fei Teng, Chen Jason Zhang, Lei Chen, Qing Li  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2606.09508v1  

#### Abstract
Existing sparse attention and KV cache compression methods for long-context LLM inference typically apply fixed sparsity patterns or uniform budgets across all attention heads, overlooking the substantial variation in attention behavior among heads and contexts. We observe two distinct entropy patte...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# From Rigid to Dynamic: Entropy-Guided Adaptive Inference for Long-Context LLMs —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大语言模型（LLMs）在处理**长上下文（long-context）推理**时面临两大瓶颈：
- **Prefilling 阶段**：注意力计算复杂度为 $O(N^2)$，导致首 token 延迟高；
- **Decoding 阶段**：KV Cache 随序列增长而线性膨胀，造成显存压力。

现有方法如 **SnapKV、AdaKV、CritiPrefill** 等通常采用固定稀疏模式或统一预算分配策略，忽略了：
- 不同 attention head 行为差异显著；
- 注意力行为随输入上下文动态变化；
- Prefill 和 Decode 阶段的注意力模式存在“错配”（misalignment）。

这些静态假设限制了效率与质量之间的平衡。

---

### 🚀 提出的新方法：**EntropyInfer**

一种无需训练（training-free）、基于注意力熵（attention entropy）进行自适应推理调度的框架，包含两个核心模块：

#### （1）**Entropy-Guided Sparse Prefill**
- 在 prefill 阶段，通过观察小规模的 **observation attention matrix** 实时估计每个 head 的行熵（row-wise entropy）。
- 将 attention head 分为两类：
  - **Rigid Heads**：熵值极低（< 1e-5），注意力分布近乎确定 → 分配**固定最小预算**；
  - **Dynamic Heads**：熵波动大，语义敏感 → 根据熵的变化幅度**动态调整计算预算**。
- 实现细粒度到 **head + segment** 层级的资源分配，避免“一刀切”。

#### （2）**Latent KV Cache Compression**
- 改变传统在 prefill 结束后立即压缩 KV Cache 的做法；
- 引入“延迟压缩”机制，在生成前 $N_d$ 个输出 token 后再执行压缩；
- 利用这些 **output tokens** 构建新的 observation window 来重新评分并选择重要缓存项；
- 修正了 prefill 与 decode 阶段注意力模式不一致的问题。

> ✅ 整体为一个即插即用（drop-in replacement）的 attention 替代方案，无需微调。

---

### 🔍 相比现有方法的优势
| 方法 | 缺陷 | EntropyInfer 如何改进 |
|------|------|------------------------|
| **预定义稀疏模式**（如 MInference） | 固定几何结构，无法适应上下文变化 | 动态感知每 head 每 segment 的熵变化 |
| **全局阈值控制**（如 FlexPrefill） | 忽略 head 间异质性 | 区分 Rigid/Dynamic head 并差异化处理 |
| **均匀预算分配**（如 CritiPrefill） | 所有 head 共享相同预算 | 按 entropy 波动自适应调节预算 |
| **基于 Prefill 的 KV 压缩**（如 SnapKV/AdaKV） | 忽视 decode 阶段 attention shift | 引入 latent decode，利用 output token 再排序 |

> ✅ 首次提出 **online head categorization** 思想，强调分类必须在线完成，不能依赖离线校准。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **LongBench**：多任务、双语（中英）、涵盖 QA、摘要、代码等任务，用于评估有效性。
- **InfiniteBench**：专为超长上下文设计（>100K tokens），测试极端长度下的鲁棒性。
- **Needle-in-a-Haystack 变体**：用于效率评测，指令模型总结长文本并生成 100 个 token。

### ⚙️ 实验设置
- **模型**：
  - `Llama-3.1-8B-Instruct`
  - `Qwen-2.5-7B-Instruct`
  - `openPangu-Embedded-1B` / `7B`
- **硬件**：单张 NVIDIA H100 80G GPU，192GB CPU 内存，8 核 CPU。
- **上下文长度范围**：从 4K 到 140K tokens。

### 📊 评估指标
| 类型 | 指标 |
|------|------|
| **效果（Effectiveness）** | F1 Score, ROUGE-L, Accuracy, Edit Sim, Exact Match |
| **效率（Efficiency）** | End-to-end latency, Speedup ratio |

### 🆚 基线方法对比
- **KV Cache 压缩类**：
  - **SnapKV**：快照机制保留重要 token
  - **AdaKV**：按 head 分散程度动态分配预算
  - **StreamingLLM**：仅保留 sink + 最近 token
- **Prefill 加速类**：
  - **CritiPrefill (CPrefill)**：基于块的关键性选择
- **消融版本**：
  - `Ours w/o LD`：关闭 latent decode
  - `Ours w/o SP`：关闭 entropy-guided sparse prefill

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 在 **LongBench** 上的表现（平均得分）
| 方法 | Llama-3.1-8B | Qwen-2.5-7B |
|------|---------------|-------------|
| Base | 48.94 | 48.82 |
| **Ours (EntropyInfer)** | **48.91** | **47.98** |
| AdaKV | 47.68 | 47.51 |
| CritiPrefill | 48.32 | 47.43 |
| SnapKV | 47.66 | 47.43 |

> 💡 尽管轻微下降，但远优于其他加速方法；在部分任务（如 SAMSum、LCC）甚至超过 base！

#### ✅ 在 **InfiniteBench** 上的表现（平均得分）
| 方法 | Llama-3.1-8B | Qwen-2.5-7B |
|------|---------------|-------------|
| **Ours** | **43.36** | **38.22** |
| Base | 44.06 | 39.62 |
| AdaKV | 40.59 | 26.86 |
| SnapKV | 41.93 | 36.50 |

> 💡 显著优于 AdaKV/SnapKV，在超长上下文中保持更强稳定性。

---

### ⚡ 效率提升（End-to-End Latency）

#### 🔁 最高可达 **2.39× 端到端加速**
- 测试任务：总结不同长度（4K ~ 140K）的文章，生成 100 token。
- 在 `Llama-3.1-8B-Instruct` 上验证。

| 方法 | 加速比（Speedup Ratio） |
|------|--------------------------|
| **EntropyInfer (Ours)** | **up to 2.39×** |
| CritiPrefill | ~2.1× |
| SnapKV | ~1.8× |
| AdaKV | ~1.7× |

> 图 4 显示：随着 context length 增加，EntropyInfer 的优势愈发明显。

---

### 🔍 消融实验（Ablation Study）

#### （1）模块影响（Figure 6a）
| 设置 | Avg Score (LongBench) |
|------|------------------------|
| **Ours (full)** | 48.91 |
| w/o SP（无稀疏 prefill） | ↓ 显著下降 |
| w/o LD（无 latent decode） | 48.55 |

> ❗ 说明两个模块协同作用：sparse prefill 提升 prefill 效率，latent decode 优化 decode 质量。

#### （2）参数敏感性
- **Segment Size**：
  - 摘要类任务偏好较大 segment；
  - 检索/问答偏好较小 segment。
- **Block Size**：基本不敏感。
- **Prefill Budget**：几乎无影响 → 体现方法的**强健性（robustness）**。
- **Decode Budget**：越高越好，但边际收益递减。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Attention Heads 存在两种本质模式**：
   - **Rigid Heads**：低熵、稳定、可压缩性强；
   - **Dynamic Heads**：高熵波动、承载关键语义；
   - 且同一 head 在不同输入下可能切换角色 → **必须在线判断**。

2. **Entropy 是有效的轻量级信号**：
   - 可低成本获取，指导计算资源分配；
   - 比预设模板或全局阈值更具表达能力。

3. **Prefill 与 Decode 注意力存在错配**：
   - 仅靠输入 token 无法准确预测 decode 阶段的重要缓存；
   - 引入 output token 进行 re-ranking 显著提升压缩质量。

4. **EntropyInfer 实现高效-高质量平衡**：
   - 达成最高 **2.39× 端到端加速**（>100K tokens）；
   - 质量损失极小，优于所有主流 baseline。

---

### ⚠️ 局限性（Limitations）
- **短上下文增益有限**：
  - 观察矩阵与熵分析带来额外开销；
  - 在短 context 下难以抵消成本；
  - 但在长 context 中迅速反超。
- **未探索量化/卸载组合优化**：
  - 可与 KVQuant、KVSwap 等正交技术进一步叠加。

---

### 🔮 未来工作方向
- 探索 entropy 与其他内部表示（如 gradient, activation）结合；
- 将 entropy-guided 思路扩展至 MoE 模型中的 expert selection；
- 设计更高效的 observation matrix 构造方式以降低 overhead；
- 应用于多轮对话系统中跨 turn 的 cache 管理。

---

## 🔗 开源信息
- **代码地址**：[https://github.com/SHA-4096/EntropyInfer](https://github.com/SHA-4096/EntropyInfer)

> ✅ 完全开源，支持即插即用集成，适用于实际部署场景。

</details>

---

### 4. [Decoding Naturalistic Emotion Dynamics from the Brain: An LLM-Enhanced Regression Framework](https://arxiv.org/abs/2606.07707)

**Authors**: Lemei Zhang, Peng Liu, Hans Dahle Kvadsheim, August S{\ae}tre Aasv{\ae}r, Shuer Ye, Reza Bonyadi, Maryam Ziaei, Jon Atle Gulla  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.07707v1  

#### Abstract
Decoding emotional states from neural signals has been typically framed as a discrete, single-label classification task based on emotionally stable stimuli, a formulation that oversimplifies the continuous, fluid, and co-occurring nature of human affect. This study reconceptualizes emotion decoding ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Decoding Naturalistic Emotion Dynamics from the Brain: An LLM-Enhanced Regression Framework**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
传统的情绪解码研究通常将情绪视为**离散、单标签的分类任务**，依赖于静态刺激（如短片段音乐或图片），忽略了人类情感在现实世界中**连续、动态、多维共存**的本质。此外，现有方法难以对自然主义叙事中的高频率情感波动进行建模，且缺乏可扩展的情感标注手段。

本研究旨在解决以下核心问题：
- 如何在**自然主义听觉叙事**（如《爱丽丝梦游仙境》）背景下，实现对**连续、多目标情绪轨迹**的神经解码？
- 如何克服人工标注情感数据的**高成本与主观性**？
- 如何揭示支持动态情绪处理的**分布式脑网络机制**？

---

### **提出了什么新方法或新思路**
本文提出了一种**基于LLM增强的回归框架**（LLM-Enhanced Regression Framework），其核心创新如下：

1. **多目标回归范式（Multi-target Regression Framework）**
   - 将情绪解码从传统的**single-label classification**重构为**multi-target regression**任务，同时预测Plutchik八维情绪（Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation）的连续强度变化。
   - 更好地捕捉情绪的**重叠性、连续性和动态演化**特性。

2. **LLM驱动的情感标注自动化**
   - 利用**Large Language Models**（特别是GPT-4）自动提取文本段落的情感向量，作为fMRI数据的“代理标签”（proxy labels）。
   - 采用上下文感知提示（context-aware prompting）和滑动窗口策略，生成细粒度、时间对齐的情感轨迹。

3. **基于Dynamic Functional Connectivity（DFC）的特征表示**
   - 使用**动态功能连接**（DFC）而非静态ROI激活幅度作为输入特征。
   - 通过固定时间窗内的Pearson相关计算全脑400个Schaefer ROI间的时变连接矩阵。

4. **图论可解释AI（Graph-Theoretical XAI）分析**
   - 对回归模型的特征重要性矩阵应用**Minimum Spanning Tree**（MST）分解，构建**emotion-specific脑网络拓扑结构**。
   - 计算**加权度中心性**（Weighted Degree）、**介数中心性**（Betweenness Centrality）、**模块化**（Modularity）等图指标，揭示不同情绪状态下的网络组织原则。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **情绪建模** | 单标签分类，忽略共现与连续性 | 多目标回归，建模连续动态轨迹 |
| **刺激材料** | 静态、控制性强的刺激 | 自然主义听觉叙事（生态效度高） |
| **标签获取** | 人工标注，昂贵且不可扩展 | LLM自动化标注，高效、可复现 |
| **神经表征** | 静态ROI激活（BOLD amplitude） | 动态功能连接（DFC） |
| **模型解释性** | 质量差的“黑箱”模型 | 图论XAI提供可解释的网络签名 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **Alice Dataset**（OpenNeuro ds002322）
  - 来源：26名被试在fMRI扫描期间聆听《爱丽丝梦游仙境》第一章的听觉叙述。
  - 数据模态：fMRI（3T）、EEG（部分）、行为数据。
  - 时间长度：约13分钟音频，转录为2129词、84句。
  - 预处理：SPM工具包完成头动校正、空间标准化（MNI空间）、平滑（3mm）、高通滤波（1/128 Hz）。
  - ROI划分：使用**Schaefer 400-parcel atlas**，划分为7或17个大尺度网络。

---

### **实验设置和评估指标**

#### **数据预处理流程**
1. **文本分段**：以8秒为基准窗口，手动调整边界确保语义完整性，每段重叠前一段4秒。
2. **LLM情感标注**：
   - 使用GPT-4生成Plutchik八维情绪强度（0–1）。
   - 上下文输入：当前段 + 前n段。
   - 温度设为0，保证输出确定性。
3. **fMRI-情感对齐**：
   - 将每个文本段结尾时间戳对齐到最近的fMRI体积（TR=2s）。
   - 构建**ROI-time series triplets**：`(ROI_{t-1}, ROI_t, ROI_{t+1})` 与对应情感标签。
4. **DFC构造**：
   - 固定窗口长度为12 TR（24秒），滑动步长为1 TR。
   - 每个DFC窗口的情感标签取该窗口内及后续两个情感向量的平均值。

#### **模型训练与评估**
- **模型类型**：
  - 线性模型：Linear Regression, Lasso, Ridge, Linear SVR
  - 非线性模型：SVR (RBF kernel), Random Forest Regressor (RFR)
- **训练策略**：
  - 两种时间分割：90-10 和 80-20（chronological split，避免未来信息泄露）
  - 所有数据归一化为标准正态分布
- **超参数优化**：Grid Search + R² 和 MSE 作为评价指标
- **评估指标**：
  - **R²**（决定系数）：越高越好
  - **MSE**（均方误差）：越低越好

---

### **基线方法对比**
- **特征基线**：
  - **ROI-based**：仅使用400个ROI的BOLD信号振幅
  - **DFC-based**：使用400×400动态功能连接矩阵
- **模型基线**：
  - 各类回归器在两类特征上的表现对比
  - 引入**leave-one-network-out ablation**分析各网络贡献

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（90-10 split，Test Set）**

| 模型 | 数据 | Test R² | Test MSE |
|------|------|--------|--------|
| **SVR (RBF)** | ROI | 0.2621 | 0.0161 |
| **SVR (RBF)** | **DFC** | **0.5931** | **0.0047** |
| Ridge Regression | DFC | 0.5589 | 0.0050 |
| Lasso Regression | DFC | 0.4742 | 0.0058 |
| Random Forest | DFC | 0.1413 | 0.0103 |

> ✅ **DFC显著优于ROI**：所有模型在DFC上表现更优，尤其RBF-SVR提升巨大（R²从0.26→0.59）

---

### **与基线方法的对比结果**
- **DFC vs. ROI**：
  - DFC在所有模型中均取得更高R²和更低MSE。
  - 表明**动态网络交互**比**静态区域激活**更能编码复杂情绪状态。
- **非线性 vs. 线性模型**：
  - 在ROI数据上，RFR和SVR(RBF)优于线性模型。
  - 在DFC数据上，**线性模型（如Ridge）表现接近甚至超越非线性模型**，说明DFC已蕴含足够强的线性可分结构。

---

### **消融实验结果**
#### **Leave-One-Network-Out 分析**
- 移除任一网络对整体性能影响极小（最大MSE差异仅0.5‰），表明情绪信息是**高度分布式**的。
- 特别地：
  - 移除**Ventral Attention Network**轻微提升性能（可能因去噪）
  - 移除**Visual Network**无显著影响
- 但在联合移除**Ventral Attention + Visual Network**后，多数模型性能下降，说明二者仍贡献**非冗余预测信号**。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **DFC是解码动态情绪的关键**  
   DFC-based模型显著优于ROI-based模型，支持**心理建构主义理论**（psychological constructionist framework）——情绪是分布式网络协同的结果，而非特定脑区孤立活动。

2. ✅ **LLM可用于高质量、可扩展的情感标注**  
   GPT-4生成的情感标签经人类验证达到较高一致性（mean=7.83/10），证明LLM可作为可靠的情感代理标注工具。

3. ✅ **不同情绪具有独特的脑网络拓扑签名**
   - **Limbic Network A**（含眶额皮层、颞极）在几乎所有情绪中都表现出高**加权度中心性**，是稳定的情感枢纽。
   - **Positive emotions**（如Joy, Trust）：
     - 更紧凑、高效的网络结构（**低Modularity，高Global Efficiency，小Tree Diameter**）
   - **Negative emotions**（如Disgust, Sadness, Anger）：
     - 更分离、低效、空间延展的网络（**高Modularity，低Efficiency，大树直径**）
   - **Anticipation**：兼具负性情绪的高模块化与中等效率，反映其准备性认知状态。

4. ✅ **情绪间存在结构性相似性**
   - **Anticipation ↔ Surprise**、**Anger ↔ Sadness** 具有高度相似的MST结构，暗示共享神经机制。
   - **Fear ↔ Disgust** 反而不相似，可能因其依赖不同的亚皮层系统（如杏仁核 vs. 前岛叶）。

---

### **方法的局限性**
1. **数据集限制**：
   - 样本量较小（N=26），统计效力有限。
   - 缺乏直接的**主观情绪报告**，标签为LLM推断，可能存在**语义混淆**（semantic confound）。

2. **技术约束**：
   - 使用**纯皮层parcellation**（Schaefer），未包含**amygdala、hippocampus**等关键边缘系统结构。
   - DFC使用**固定窗口**（24秒），可能无法捕捉更快或更慢的时间尺度动态。
   - MST虽简化网络，但也**丢弃弱连接信息**。

3. **模型假设**：
   - 回归模型假设特征独立，但DFC高度共线。
   - 当前方法聚焦群体水平模式，忽略个体差异。

---

### **未来工作方向**
- 扩展至**多模态自然刺激**（电影、对话）和更大规模数据集。
- 引入**动态图神经网络**（DGNN）或**Transformer-based models**建模更复杂的时空依赖。
- 结合**subcortical parcellation**（如Buckner 7-network）纳入深层结构。
- 探索**个性化解码框架**，结合个体差异建模。
- 开发**闭环情绪调节系统**，基于实时DFC反馈进行干预。

---

> 📌 **总结一句话**：  
> 本研究通过**LLM+DFC+Regression+XAI**的整合框架，首次实现了对自然主义叙事中**连续、多维情绪动态**的高精度神经解码，并揭示了支持这些状态的**可解释、情绪特异性的脑网络拓扑组织原则**，为情感神经科学提供了新的方法论范式和理论支持。

</details>

---

### 5. [Beyond FLOPs: Benchmarking Real Inference Acceleration of LLM Pruning under a GEMM-Centric Taxonomy](https://arxiv.org/abs/2606.09080)

**Authors**: Haozhe Hu, Hao Wu, Anhao Zhao, Longwei Ding, Peiran Yin, Yunpu Ma, Xiaoyu Shen  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.09080v1  

#### Abstract
Pruning has emerged as a dominant paradigm for accelerating large language model (LLM) inference, spanning a broad spectrum of methods that remove computation across tokens, layers, heads, dimensions, and attention patterns. Despite sharing the same objective, these pruning approaches induce fundame...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond FLOPs: Benchmarking Real Inference Acceleration of LLM Pruning under a GEMM-Centric Taxonomy

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **LLM pruning** 方法虽然在理论上能减少计算量（如通过降低 FLOPs），但由于不同方法对硬件执行行为（如内存访问、kernel 调度）的影响差异巨大，导致**实际推理加速效果与理论预测严重脱节**。此外，由于缺乏统一的评估框架，各类 pruning 方法之间难以公平比较。

本文系统性地揭示了这一“**名义稀疏性（nominal sparsity）≠ 实际加速**”的问题，并指出其根源在于：不同的 pruning 方法改变了模型在 **GEMM** 运算中的逻辑维度（M/N/K），从而影响了底层 kernel 的实现效率和端到端吞吐。

### 提出了什么新方法或新思路
1. **GEMM-Centric Taxonomy（以 GEMM 为中心的分类法）**  
   将所有 LLM pruning 方法统一映射到 GEMM 的三个逻辑维度上：
   - **M-dimension pruning**：对应于 `token` 或 `layer` 维度的剪枝（如 depth pruning）
   - **N-dimension pruning**：输出特征维度剪枝（常表现为 attention head 剪枝）
   - **K-dimension pruning**：输入特征维度剪枝（如权重矩阵列剪枝、低秩分解）

   该分类法不仅抽象了方法细节，还揭示了稀疏性的**传播规律**（例如 N-pruning 在后续 GEMM 中会转化为 K-pruning）。

2. **统一的基准测试框架（Unified Benchmarking Framework）**  
   构建了一个硬件无关、可复现的推理加速评测平台，支持：
   - 对每种 pruning 类型提供一致的 kernel 实现（基于 Triton / Tilelang DSL）
   - 控制变量下的端到端延迟测量
   - 支持跨平台（如 RTX Pro6000, A800）验证

### 相比现有方法的优势
- **超越 FLOPs 的评估视角**：不再依赖不可靠的 proxy 指标（如 FLOPs reduction），而是直接测量真实 throughput。
- **公平可比性**：首次实现了跨 pruning 家族的 apples-to-apples 比较。
- **指导性强**：为未来 pruning 算法设计和 kernel 优化提供了明确的方向（如静态深度剪枝仍是性价比最高的选择）。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **下游任务评估**：采用 `lm-evaluation-harness` 统一评测，包含以下基准：
  - WikiText2（PPL）
  - ARC-e/c, BoolQ, WinoGrande, PIQA, OpenBookQA, HellaSwag（accuracy/norm）
- **训练数据**：RedPajama-Data-1T 子集用于 LoRA 微调

### 实验设置和评估指标
- **基础模型**：Llama3.1-8B（主实验）、Qwen3-14B（补充）
- **硬件环境**：RTX Pro6000 Blackwell（sm120），部分实验在 A800（sm80）
- **精度模式**：bf16（Pro6000），fp16（A800）
- **关键指标**：
  - **Throughput**：`Token/s = (B × T_q) / (TTFT or TPOT)`
  - **Speedup**：相对于 dense 模型的吞吐提升倍数
  - **Quality Loss**：平均准确率下降百分比（Avg. Acc. Gap%）
  - **Pareto Frontier**：综合考虑速度与质量的最优边界

### 基线方法对比
选取代表性的方法作为各 taxonomy 的代表：
| Taxonomy | Representative Method |
|--------|-----------------------|
| Static M | Shortened-taylor, CoopPruner |
| Static K (low-rank) | Dobi-SVD |
| Static K (semi-structured) | MaskLLM |
| Static NK | Tyr-the-Pruner |
| Static NK (cross-layer) | SliceGPT+ |
| Dynamic M | SkipGPT |
| Dynamic NK | SeerAttention |

所有方法均在相同 sparsity 预算下进行比较（12.5%, 25%, 37.5%, 50%）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2 & Figure 1）

#### 在 **50% sparsity** 下的表现（Llama3.1-8B）：

| Method | Quality Loss (Acc. Gap%) | Prefill Speedup | Decode Speedup |
|-------|----------------------------|------------------|----------------|
| **Static M** | 30.83% | **1.88x** | **1.91x** |
| Static NK | 26.40% | 1.77x | 1.70x |
| Dynamic M | 15.59% | 1.44x | 1.10x |
| Static K (low-rank) | 20.46% | 1.43x | 1.46x |
| Static K | 11.25% | 1.08x | 1.14x |
| Dynamic NK | 33.28% | 1.05x | 1.04x |

> ✅ **Static M 在高稀疏度下仍保持最强绝对加速能力**

### 与基线方法的对比结果
- **Static M 是最稳定的加速 baseline**：
  - 即使在 50% 稀疏度下，其实现的速度提升接近理论上限。
  - 在 prefill 和 decode 阶段均有显著且均衡的收益。
- **Dynamic M 更适合中等质量损失场景**：
  - 在 5%-16% 质量损失区间内，其 prefill 加速可达 1.24x~1.44x，优于同稀疏度的静态方法。
- **Static NK 成为宽度剪枝中的前沿方案**：
  - 在 >17% 质量损失时成为最具竞争力的 width pruning 方法。
- **其他方法受限明显**：
  - Static K（semi-structured）虽保质好，但加速弱（仅 ~1.1x）
  - Static NK (cross-layer) 因引入额外 residual projection 导致 overhead 上升
  - Dynamic NK 仅在长上下文场景中有优势

### 消融实验结果
- **维度对其传播影响显著**：
  - M-pruning 不改变形状，易于实现高效 kernel
  - N-pruning 会在后续 GEMM 中转为 K-pruning，形成 NK pattern，需联合处理
- **非 GEMM 开销不可忽视**：
  - Dynamic M 引入 mask reordering、gather/scatter 等操作，在 decode 阶段带来高达 61.5% 的额外开销
  - Static K (low-rank) 因拆分为两个小 GEMM，增加 graph launch 次数，限制加速潜力
- **对齐（Alignment）至关重要**：
  - Width pruning 若未对齐到 GPU 向量化宽度（如 16-byte boundary），可能导致最高达 **35% 的性能损失**

---

## 4. 关键结论和发现

### 论文的主要发现
1. 🔹 **名义稀疏性不是可靠加速指标**：相同的 sparsity %，不同 pruning 方式的实际加速差距可达 **2倍以上**。
2. 🔹 **Static M（深度剪枝）仍是当前 Pareto 最优解**：
   - 在 memory-bound 场景下最接近理论加速上限
   - 实现简单、稳定性强，是目前最实用的加速 baseline
3. 🔹 **Pareto 前沿随质量预算动态变化**：
   - **低质量损失（0%-4%）**：Static M 占优
   - **中等质量损失（5%-16%）**：Dynamic M 更具优势
   - **高质量损失（17%-26%）**：Static NK 成为主力
4. 🔹 **Width pruning 整体仍落后于 Depth pruning**：
   - 受限于 kernel 实现复杂性和非 GEMM 开销，当前 width 方法尚未充分发挥潜力
5. 🔹 **Hybrid 设计是未来方向**：
   - 结合 quality-aware dynamic branch 与 static structured backbone 可能突破当前瓶颈

### 方法的局限性
- **未覆盖 MoE 架构**：当前框架主要针对 dense Transformer，对 Mixture-of-Experts 模型的支持有限。
- **DSL 实现非最优**：Triton/Tilelang 编写的 kernel 可能不如 hand-tuned CUDA kernel 高效。
- **未集成完整 serving pipeline**：缺少对 SGLang 等生产级推理系统的调度、批处理建模。
- **硬件平台有限**：主要在 Blackwell 和 Ampere 架构上验证，Hopper/TMA 等新特性未充分探索。

### 未来工作方向
- 探索 **hybrid pruning architectures**：结合 dynamic routing 与 static backbone
- 开发 **pruning-aware kernel compiler**：自动优化稀疏模式对应的 kernel 实现
- 扩展至 **multi-modal models** 和 **MoE models**
- 构建 **end-to-end serving simulator**，纳入 batch scheduling、KV cache management 等系统因素
- 研究 **fine-grained structured sparsity** 与硬件指令集（如 Tensor Cores）的协同设计

--- 

> 📌 **一句话总结**：  
> 本文打破了“FLOPs 决定加速”的迷思，提出以 **GEMM 维度**为核心的新分类体系，并通过统一基准发现：**Static M 仍是当前最高效的 LLM 推理加速路径**，而未来的突破将来自 **算法-硬件协同设计的 hybrid pruning 架构**。

</details>

---

### 6. [Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models](https://arxiv.org/abs/2606.09401)

**Authors**: Bart{\l}omiej Marek, Lorenzo Rossi, Vincent Hanke, Xun Wang, Michael Backes, Franziska Boenisch, Adam Dziedzic  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.09401v1  

#### Abstract
Recent work has applied differential privacy (DP) to adapt large language models (LLMs) for sensitive applications, offering theoretical guarantees. However, its practical effectiveness remains unclear, partly due to LLM pretraining, where overlaps and interdependencies with adaptation data can unde...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统地研究了在**大语言模型**（LLMs）中应用**差分隐私**（DP）进行适配时的实际隐私保护效果。尽管 DP 在理论上提供了强大的隐私保证，但在 LLM 的“预训练-适配”（pretrain-adapt）范式下，由于预训练数据与适配数据之间可能存在重叠或分布上的相关性，其实际有效性尚不明确。

现有研究大多孤立地分析预训练或适配阶段的隐私风险，而本文首次**系统性地量化了二者交互对隐私泄露的影响**，填补了实证隐私评估的空白。

### 提出了什么新方法或新思路
1. **全面的实证基准测试框架**：
   - 构建了一个涵盖多种数据分布、适配方法和隐私机制的综合评估体系。
   - 系统性地研究了从完全重叠（Overlap）、同分布（IID）到完全异分布（OOD）的适配数据对隐私风险的影响。

2. **提出“全周期隐私审计”框架**（Holistic Privacy Auditing Framework）：
   - 定义了四个关键审计阶段：
     1. **预训练审计**（Auditing Pretraining）
     2. **适配审计**（Auditing Adaptation）
     3. **联合审计**（Joint Auditing）
     4. **后适配预训练审计**（Post-Adaptation Auditing of Pretraining）
   - 将每个阶段形式化为一个对抗性游戏（adversarial game），使隐私评估更具结构性和可实例化。

### 相比现有方法的优势
- **更贴近现实场景**：考虑了真实世界中预训练与适配数据的复杂关系，而非假设数据完全独立。
- **多维度评估**：结合了**成员推断攻击**（MIA）和**数据提取攻击**（Data Extraction）等多种攻击手段。
- **指导实践**：为在敏感领域部署私有化 LLM 提供了具体的方法选择建议和风险评估指南。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验基于多个开源 LLM 和数据集构建，确保可复现性：

#### 模型
- **Pythia** 系列：Pythia 70M 到 2.8B
- **GPT-Neo** 系列：125M 和 1.3B
- **OLMo** 系列：1B 和 2.1B（作为新开源模型代表）

所有模型均在 **The Pile** 数据集上预训练。

#### 适配数据集
分为两类：
- **同分布**（IID）：
  - `Bookcorpus2`（书籍）
  - `GitHub`（代码）
  - `Enron Emails`（邮件）
- **异分布**（OOD）：
  - `SAMSum`（英文对话摘要）
  - `GermanWiki`（德语维基百科）

通过计算 **Wasserstein 距离**验证了 OOD 数据与 Pile 分布差异更大。

### 实验设置和评估指标

#### 适配方法
比较了四种主流适配策略：
- **Full Fine-Tune**：微调全部参数
- **Head Fine-Tune**：仅微调输出头
- **LoRA**（Low-Rank Adaptation）
- **Prefix Tuning**

所有方法均集成 **DP-SGD** 实现差分隐私。

#### 隐私预算
覆盖从无隐私（ε=∞）到高隐私（ε=0.1）的广泛范围。

#### 评估指标
1. **成员推断攻击**（MIA）：
   - 使用最先进的 **RMIA**（Robust Membership Inference Attack）
   - 评估指标：**AUC**（Area Under Curve）
2. **数据提取攻击**：
   - 插入对抗性 **canary** 并测量其 **exposure**
   - 使用 **k-extractable memorization** 衡量记忆程度
3. **实用性**：
   - **Perplexity** 和 **Rouge-1** 评分
   - 确保不同方法间具有可比性

#### 基线方法对比
- 不同适配方法之间的横向对比（如 LoRA vs Full Fine-Tune）
- 不同数据分布下的表现差异（IID vs OOD）
- 不同 ε 下的隐私-效用权衡

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 成员推断攻击（MIA）结果（以 Pythia 1B 为例）

| 适配方法 | 数据分布 | ε=∞ (AUC) | ε=8 (AUC) | ε=0.1 (AUC) |
|----------|----------|-----------|-----------|-------------|
| **LoRA** | OOD | 0.86 | 0.69 | 0.50 |
| **Full Fine-Tune** | OOD | 1.00 | 0.82 | 0.62 |
| **LoRA** | IID | 1.00 | 0.70 | 0.52 |
| **Full Fine-Tune** | IID | 1.00 | 0.75 | 0.77 |

> ✅ **关键发现**：即使在 ε=8 的“中等隐私”下，IID 数据仍面临严重泄露风险（AUC > 0.7），而 OOD 数据在相同条件下风险更低。

#### 数据提取攻击结果（Canary Exposure）

| 适配方法 | 数据分布 | ε=∞ (Exposure) | ε=8 (Exposure) |
|----------|----------|----------------|----------------|
| **Prefix Tuning** | OOD | ~6.7 | ~1.76 |
| **LoRA** | OOD | ~2.6 | ~1.60 |
| **Full Fine-Tune** | OOD | ~6.3 | ~1.60 |

> ✅ **关键发现**：**Prefix Tuning 最易受数据提取攻击**，而 **LoRA 和 Head Fine-Tune 表现出更强的鲁棒性**。

### 与基线方法的对比结果
- **LoRA 在多数情况下提供最佳隐私-效用权衡**：
  - 在 OOD 场景下，LoRA 的 MIA AUC 显著低于 Full Fine-Tune（平均低 0.1–0.2）
  - 在高隐私设置（ε=0.1）下，LoRA 的 AUC 接近随机猜测水平（~0.5）
- **Full Fine-Tune 虽然效用高，但隐私风险最大**
- **Head Fine-Tune 效率最高但隐私保护有限**

### 消融实验结果
1. **攻击者知识的影响**：
   - 当攻击者拥有与目标模型架构相同的“影子模型”（shadow model）时，RMIA 攻击成功率显著提升。
   - 若使用不同架构或不同预训练数据的模型作为参考，攻击效果迅速下降。
   - **结论**：公开可用的 LLM 极大地降低了攻击门槛。

2. **分布偏移的影响**：
   - 即使没有直接数据重叠，只要适配数据与预训练数据分布接近（IID），隐私风险就远高于 OOD 数据。
   - **Bookcorpus2 Val**（IID）与 **Bookcorpus2 Train**（重叠）的泄露程度几乎一致。

3. **子集大小与复杂度影响**：
   - 更大、更多样化的预训练子集（如 CC、ArXiv）导致更高的隐私泄露。
   - 复杂性越高，模型越可能记住特定样本。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **分布相似性是隐私风险的关键驱动因素**：
   - 即使没有数据重叠，**适配数据与预训练数据越相似（IID），隐私泄露越严重**。
   - 这揭示了“预训练带来效用，但也放大隐私风险”的根本矛盾。

2. **适配方法的选择至关重要**：
   - **LoRA 是最能平衡隐私与效用的适配方法**，尤其在 OOD 场景下表现优异。
   - **Prefix Tuning 虽高效，但极易遭受数据提取攻击**，不适合高敏感任务。

3. **中等隐私预算不足以提供实际保护**：
   - 在 ε=8 的设定下，IID 数据仍面临显著的成员推断和数据提取风险。
   - **必须采用高隐私设置**（如 ε < 0.1）才能实现有效的实证隐私保护。

4. **适配过程也会影响预训练数据的隐私**：
   - **Prefix Tuning 可减少对预训练数据的记忆**，而其他方法基本保留原有记忆。
   - 表明适配不仅影响自身数据，也可能改变整个模型的隐私特性。

### 方法的局限性
- **仅限于开源模型**：无法评估 GPT-4、Gemini 等闭源模型，因其不支持梯度访问和 DP 微调。
- **未覆盖所有适配技术**：如 PATE 或 in-context learning 未被纳入。
- **攻击假设较强**：部分实验假设攻击者拥有精确的数据划分信息，现实中可能难以实现。

### 未来工作方向
1. **扩展至闭源模型**：开发适用于 API 访问模式的黑盒隐私评估方法。
2. **动态隐私调整**：根据数据分布自动选择最优的适配方法和隐私预算。
3. **防御机制设计**：基于本研究发现，设计专门针对分布偏移的隐私增强技术。
4. **工具化框架**：将提出的“全周期隐私审计”框架实现为开源工具包，供社区使用。

---

> **总结**：本文通过大规模实证研究揭示了 LLM 差分隐私适配中的关键风险——**数据分布相似性会显著削弱 DP 的实际保护能力**。研究强调了在敏感应用中应优先选择 **LoRA** 等参数高效微调方法，并采用**严格的隐私预算**。更重要的是，它呼吁业界超越孤立的隐私评估，转向贯穿“预训练-适配”全流程的**系统性隐私治理**。

</details>

---

### 7. [Large-Scale Regularized Matching on GPU Clusters](https://arxiv.org/abs/2606.07777)

**Authors**: Aida Rahmattalabi, Gregory Dexter, Sanjana Garg, Qinquan Song, Shenyinying Tu, Yuan Gao, Zhipeng Wang, Rahul Mazumder  
**Category**: cs.DC  
**Published**: 2026-06-09  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.07777v1  

#### Abstract
Production decision systems such as ad allocation or content matching involve millions of users and thousands of items, reducing to large-scale linear programs with sparse block-diagonal structure across users. These LPs are solved repeatedly on recurring cadences over slowly evolving inputs. Three ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Large-Scale Regularized Matching on GPU Clusters*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**大规模生产级决策系统**（如广告分配、内容匹配）中的线性规划（Linear Programming, LP）求解难题，解决了以下三大系统瓶颈：

- **Scale（规模）**：现有 GPU 求解器（如 cuPDLP、D-PDLP）受限于单设备内存容量，无法处理数十亿变量和数亿约束的大规模实例。
- **Temporal Instability（时间不稳定性）**：连续运行间解的波动（solution drift）导致下游系统“抖动”（churn），影响 SLA，而现有求解器缺乏显式控制机制。
- **Extensibility（可扩展性）**：基于 CPU 的求解器（如 DuaLip-Scala）收敛慢，且将问题建模与固定 Schema 绑定，新增约束成本高。

### 提出的新方法与创新点
作者提出了一种**原生构建于 PyTorch 的分布式多 GPU LP 求解器**，采用“系统-算法协同设计”（systems-algorithm co-design）策略，围绕生产匹配 LP 的**对角块结构**（diagonal block structure）进行优化，主要贡献如下：

#### （1）列分片执行模型 + 融合 Triton 内核（Column-sharded Execution & Fused Kernels）
- 将约束矩阵按“源”（用户）维度进行**列分片**（column-sharded），每个 GPU 处理部分源的数据。
- 利用 **NCCL** 实现高效的 Reduce 和 Broadcast 操作，每轮迭代仅需在 rank 0 上聚合 item-level dual variables，通信量与源数量无关。
- 引入 **Triton 融合内核**实现 simplex 投影，将排序、前缀和、阈值选择等操作融合为单个 GPU 内核，显著降低 launch 开销和中间内存占用。

> ✅ **优势**：实现了近线性的多 GPU 扩展性，支持超出单 GPU 内存的问题规模。

#### （2）可调的岭正则化（Tunable Ridge Regularization）以提升稳定性
- 采用 **ridge-regularized dual ascent** 框架，在目标函数中引入 $ \frac{\gamma}{2}\|x\|^2 $ 正则项。
- 显式暴露参数 $\gamma$ 作为**一级控制变量**，理论上可**界定连续运行间的 primal drift**。
- 设计 **$\gamma$ continuation schedule**：从较大的 $\gamma$ 开始以加速初期收敛并稳定迭代，逐步退火至小值以逼近原始 LP 最优解。

> ✅ **优势**：首次在 GPU 求解器中提供对解稳定性的显式控制，解决生产环境中的“抖动”问题。

#### （3）算子中心编程模型（Operator-Centric Programming Model）
- 提出三个可组合的抽象原语：
  - `ObjectiveFunction`：定义 $(A, b, c)$
  - `ProjectionMap`：实现块状投影（如 simplex、box）
  - `Maximizer`：执行对偶上升流程
- 新增约束只需局部修改 `ObjectiveFunction`，无需改动求解循环或分布式基础设施。

> ✅ **优势**：极大提升建模灵活性，支持快速添加 pacing、fairness、frequency cap 等复杂约束。

---

## 2. 核心实验方法和设置

### 数据集
- 使用**合成生成的匹配型 LP 实例**，模拟真实生产场景。
- 参数可控：源数量（sources）从 25M 到 100M，目的地数量（destinations）固定为 10K，稀疏度约 0.1%。
- 构造方式基于工业分布（lognormal breadth, Poisson sampling），确保边权值和约束系数具有异构性和现实代表性。

### 实验设置
- **硬件平台**：
  - 单节点：NVIDIA H100 80GB GPU（1–8 GPUs）
  - 多节点：跨两个 H100 节点（共 16 GPUs），通过 torchrun 协调
- **实现框架**：PyTorch + torch.distributed (NCCL backend) + Triton
- **精度**：float64（与基线对齐）
- **求解配置**：
  - 使用 AGD（Accelerated Gradient Descent）
  - $\gamma$ continuation schedule：$\gamma \in \{10^3, 10^2, 10, 1, 10^{-1}, 10^{-2}\}$，每阶段 10,000 次迭代（共 60,000）
  - 启用 Jacobi preconditioning 和 bucketed batching

### 评估指标
- **每轮迭代耗时**（per-iteration time）
- **端到端求解时间**（end-to-end solve time）
- **加速比**（speedup vs. baseline 或 ideal linear scaling）
- **峰值 GPU 内存占用**
- **解质量**：primal/dual objectives、primal-dual gap、constraint violation（slack）

### 基线方法对比
- **DuaLip-Scala**：基于 Spark 的 CPU 集群求解器，代表当前工业级解决方案。
- **D-PDLP**：最先进的多 GPU primal-dual 求解器，基于 2D 矩阵划分和 NCCL AllReduce。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 来源数量 | DuaLip-Scala (秒/迭代) | 本工作 (1 GPU) | 本工作 (4 GPUs) |
|--------|----------------------|---------------|----------------|
| 25M    | 2.46                 | 0.27          | 0.07           |
| 50M    | 3.44                 | —             | 0.13           |
| 100M   | 3.33                 | —             | 0.27           |

> 💡 **观察**：单 GPU 即实现近 **9.1× 加速**；4 GPU 下达 **35× 以上加速**。

#### 多 GPU 扩展性（Figure 3）
- 在 16 GPUs 上：
  - 75M sources：**13.1× speedup**（82% 效率）
  - 50M sources：**12.0× speedup**（75%）
  - 25M sources：**9.7× speedup**（61%）
- 跨节点（8→16 GPUs）扩展平滑，表明 NCCL 通信未成为瓶颈。

#### 内存与可扩展边界
- 100M sources 问题在单 GPU 上 **OOM**（>80 GiB），但在 ≥2 GPUs 上可运行。
- 16 GPUs 下，100M sources 实例可在 **3 分钟内完成**。

### 与基线方法对比（Table 3 & 4）

| 方法       | 是否支持 100M sources? | 是否支持 $l_1$-reformulated? | 相对速度（base instances） |
|----------|---------------------|----------------------------|-------------------------|
| **本工作** | ✅ 是                | ✅ 是                        | —                       |
| D-PDLP   | ❌ OOM               | ❌ 全部 OOM                   | 更快（当能运行时）         |

> ⚠️ **关键发现**：尽管 D-PDLP 在小规模上更快，但其内存消耗过高（MPS 文件达 300+ GB），**无法处理实际所需的 $l_1$-regularized 变体或最大规模实例**。

#### 解质量对比（Table 4）
- 在共同可解实例上，本工作与 D-PDLP 的 dual objective 一致至 **4 位有效数字**。
- 本工作的 **primal-dual gap 更小**（低至 $10^{-12}$ vs. D-PDLP 的 $10^{-5}$），得益于正则化带来的更好条件数（conditioning）。

### 消融实验结果

#### （1）Triton 融合内核 vs. PyTorch Eager（Figure 1）
- **速度提升**：2.5–5×（10M–50M sources），在 1M sources 上高达 **20×**
- **内存节省**：峰值 GPU 内存降低 ~20%，因消除中间张量（如 sorted values, masks）

#### （2）桶化批处理 vs. 单一密集块（Figure 2）
- **速度提升**：约 **1.2×**
- **内存节省**：~24%，避免大量零填充计算

#### （3）对角预处理（Diagonal Preconditioning）（Figure 4）
- 显著加快早期收敛，尤其在异构约束尺度下。

#### （4）$\gamma$ 退火调度 vs. 固定 $\gamma$（Figure 5）
- 动态衰减 $\gamma$ 比固定值收敛更快，同时保证最终解接近无正则化最优。

---

## 4. 关键结论和发现

### 主要发现
1. **结构感知的系统设计至关重要**：利用匹配 LP 的“源-目的地”对角块结构（diagonal block structure），可实现高效列分片与低通信开销。
2. **正则化不仅是算法技巧，更是系统稳定性工具**：通过 tunable $\gamma$ 和 continuation schedule，可在收敛速度与解保真度之间灵活权衡，并抑制运行间抖动。
3. **现代 ML 框架（PyTorch + Triton）适合构建高性能优化系统**：相比传统 C++/CUDA，仍可获得接近定制求解器的性能，同时大幅提升开发效率与可维护性。
4. **可扩展性优于绝对速度**：虽然 D-PDLP 在小规模更快，但本方法在**超大规模和复杂变体上具备唯一可行性**，更适合生产部署。

### 局限性
- 当前评估依赖**合成数据**，缺乏公开的真实工业基准集。
- 通信虽轻量，但 dual update 仍在 rank 0 序列化执行，可能成为极端大规模下的潜在瓶颈。
- 目前仅适用于具有明确“源-目的地”结构的匹配类 LP，通用性有待验证。

### 未来工作方向
- 将该 dual-ascent 框架推广至更广泛的 LP 类别（如网络流、资源分配）。
- 构建**标准化的超大规模匹配 LP 基准测试套件**，促进可复现研究。
- 探索更高级的 preconditioning 或 asynchronous 更新机制以进一步提升扩展性。

--- 

> ✅ **总结一句话**：  
> 本文提出了一种面向生产级大规模匹配问题的分布式 GPU LP 求解器，通过**结构感知的列分片、可调正则化控制稳定性、以及算子化编程模型提升可扩展性**，在真实规模下实现了相比 CPU 和现有 GPU 求解器的**数量级性能提升与唯一可行的扩展能力**。

</details>

---

### 8. [Breaking the Bubble: Asynchronous Pipeline Parallel Training with Bounded Weight Inconsistency](https://arxiv.org/abs/2606.07881)

**Authors**: Itay Elam, Eliron Rahimi, Avi Mendelson, Chaim Baskin  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.07881v1  

#### Abstract
Pipeline parallelism is essential for training large neural networks, but existing schedules trade off throughput, memory, and optimization consistency. Synchronous pipelines preserve forward/backward weight consistency but suffer from bubbles; asynchronous pipelines remove bubbles but introduce wei...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Breaking the Bubble: Asynchronous Pipeline Parallel Training with Bounded Weight Inconsistency**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

在大规模神经网络（尤其是 LLM）训练中，**pipeline parallelism** 是提升硬件利用率的关键技术。然而，现有方法面临以下权衡：

- **Synchronous Pipeline**（如 1F1B-flush）：保证前向/反向传播使用一致的参数版本（forward/backward weight consistency），但引入 **pipeline bubbles**（空闲时间），导致吞吐率下降。
- **Asynchronous Pipeline**：消除 bubbles，提高吞吐，但引入 **forward/backward weight-version inconsistency**（即反向传播时参数已更新），影响训练稳定性。

传统异步方法通过 **weight stashing**、**prediction** 或 **额外参数副本** 来缓解不一致性，但带来额外内存开销或系统复杂度。

---

### 🚀 **提出了什么新方法或新思路**

本文提出 **PACI**（**Pipeline Asynchronous training with Controlled Inconsistency**），一种无需权重暂存、预测或全局同步的异步 pipeline 方法，其核心思想是：

> **利用本地梯度累积（local gradient accumulation）作为参数版本控制机制，主动限制前向/反向之间的参数更新次数（version drift）**。

具体机制：
- 在异步 1F1B 执行基础上，每个 stage **延迟 optimizer 更新**，仅每 `a` 个 backward 后才执行一次更新。
- 引入 **局部流控规则**（flow-control rule）：限制未完成前向传递的数量，防止上游 stage 跑得太远。
- 由此实现对 **forward/backward inconsistency** 的显式上界控制：  
  $$
  \Delta_{\text{max}} \leq \left\lceil \frac{N-1}{a} \right\rceil
  $$
  其中 `N` 为 stage 数量，`a` 为 accumulation factor。

---

### 🔍 **相比现有方法的优势**

| 特性 | Flush / 1F1B-I | PipeDream / 2BW | PipeMare / PipeOptim | **PACI (Ours)** |
|------|----------------|------------------|------------------------|----------------|
| 执行模式 | Sync | Async | Async | **Async** |
| 额外内存 | 0 | ++（多份权重） | ++（velocity buffer） | **0** |
| Pipeline Bubbles | 高 / 中等 | 0 | 0 | **0** |
| F/B Inconsistency | 0 | 0（stashing） | Approximate | **Low & Bounded** |
| 是否需要同步 | 是（flush） | 是（stashing） | 否 | **否** |

**PACI 的独特优势**：
- **零 bubbles**：保持异步高吞吐。
- **零额外权重内存**：无需 weight stashing 或 prediction buffer。
- **低且有界的 inconsistency**：通过 `a` 可调，无需全局协调。
- **简单易实现**：基于标准梯度累积，兼容现有框架。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集和模型**

- **模型**：GPT-2 Medium（约 355M 参数）
- **数据集**：OpenWebText
- **任务**：causal language modeling（自回归语言建模）
- **序列长度**：1024
- **总训练 token 数**：49.8B tokens

---

### ⚙️ **实验设置**

- **Pipeline 设置**：8-stage pipeline parallelism，无数据并行（data parallelism）
- **精度**：BF16
- **优化器**：AdamW（β₁=0.9, β₂=0.95, weight decay=0.1）
- **学习率调度**：1% 线性 warmup + cosine decay
- **全局 batch size**：128 和 256
- **micro-batch 数量**：4–256 不等
- **硬件**：单节点 8×GPU（RTX PRO 6000 Blackwell Max-Q，96GB 显存）

---

### 📊 **评估指标**

| 指标 | 描述 |
|------|------|
| **Validation Perplexity** | 主要质量指标，衡量模型泛化能力 |
| **Wall-clock Time-to-Accuracy** | 达到目标 perplexity 所需的真实时间 |
| **Throughput (tokens/sec/GPU)** | 实际训练吞吐量 |
| **Peak GPU Memory** | 每设备峰值显存占用 |
| **Run-to-run Variability** | 多次运行的标准差，反映稳定性 |

---

### 🔁 **基线方法对比**

- **Synchronous Baseline**：
  - **1F1B-flush**：标准同步 pipeline，不同 micro-batch 数配置用于比较
- **Asynchronous Baselines**：
  - **Naive 1F1B**：纯异步，高 inconsistency
  - （隐含对比）其他如 PipeDream、PipeMare 等（见 Table 1）

> 注：PACI 与 flush 的比较聚焦于“相同内存下能否更快达到相同质量”。

---

## 3. **主要实验结果和性能指标**

### 📈 **关键性能数据**

#### ✅ **训练稳定性和最终质量**

- **Table 3** 显示，在 `Δ_max ≤ 2`（即 `a ≥ 4`）时，PACI 的 **最终 loss 与 1F1B-flush 相当甚至更优**：
  - Batch 128：PACI(a=8) 达到 2.79 ± 1.67e-3，与 flush 基本一致
  - Batch 256：PACI(a=16) 达到 2.77，略优于 flush 的 2.78
- **run-to-run variability 更低**：PACI 的训练损失标准差显著低于 flush，表明更稳定。

#### ✅ **加速效果（Time-to-Accuracy）**

- **Table 4 & Figure 1** 显示 PACI 显著缩短训练时间：
  - **Batch 128**：相比最快 flush 配置，**加速 1.69×**
  - **Batch 256**：**加速 1.41×**
- 即使与同 micro-batch 数的 flush 对比，**最高达 2.04× 加速**

#### ✅ **逐阶段加速分析（Table 6）**

| PPL Threshold | Batch Size | Speedup (vs best flush) |
|---------------|------------|--------------------------|
| ≤18           | 128        | 1.14×                    |
| ≤17           | 128        | 1.59×                    |
| ≤16           | 128        | **1.64×**                |
| ≤16           | 256        | **1.36×**                |

> **越接近收敛，加速越明显**，说明 inconsistency 影响随训练减弱，吞吐优势主导。

#### ✅ **吞吐与内存表现**

- **Figure 9**：PACI 吞吐基本恒定，而 flush 随 micro-batch 增加缓慢上升（受 bubble 效率限制）
- **Figure 7 & Table 2**：PACI 与 flush 的 **peak memory 完全相同**（无额外开销）
- **理论吞吐匹配**：PACI 实测吞吐 ≈ flush 吞吐 / bubble_efficiency，验证其“完全利用”假设

---

### 🔬 **消融实验结果**

- **accumulation factor `a` 的影响**：
  - `a=4`（Δ_max=2）：loss 略高但稳定，加速显著
  - `a=8`（Δ_max=1）：loss 与 flush 几乎一致，加速仍达 1.69×
- **micro-batch 数的影响**：
  - Flush：增加 micro-batch 可减少 bubble，但 micro-batch 过小会降低 kernel efficiency
  - PACI：不受 bubble 影响，吞吐更鲁棒

---

## 4. **关键结论和发现**

### 🧩 **主要发现**

1. **forward/backward inconsistency 不必完全消除**：只要将其限制在较小范围内（如 Δ ≤ 2），即可安全换取显著效率增益。
2. **gradient accumulation 可用作版本控制机制**：不仅是增大 batch size 的工具，更是调节训练一致性的“knob”。
3. **PACI 实现了理想权衡点**：
   - 零 bubbles（高吞吐）
   - 零额外内存（轻量）
   - 低且有界 inconsistency（稳定）
4. **实际收益显著**：在真实 LLM 预训练中，**训练时间最多缩短 1.69×，且最终质量不变甚至更好**。

---

### ⚠️ **局限性**

1. **实验规模有限**：
   - 仅在 GPT-2 Medium 上验证，更大模型（如 10B+）、更深 pipeline 尚未测试
   - 仅限 8-stage，扩展性待验证
2. **未支持 activation checkpointing**：
   - 虽理论上兼容，但重计算会改变 inconsistency 结构，需进一步研究
3. **不支持全局梯度裁剪**：
   - 完全异步下无法获取全局梯度范数，需依赖 SPAM 等替代方案
4. **无梯度失效处理机制**：
   - 如出现 NaN，无法 rollback 已更新的 stage，可能影响容错性

---

### 🔮 **未来工作方向**

1. **扩展到更大模型和更深 pipeline**
2. **结合 activation checkpointing 的 PACI 变体设计与评估**
3. **探索更复杂的 inconsistency-aware optimizer**
4. **在多模态、RL 等任务中验证通用性**
5. **硬件层面优化通信与流水线调度以进一步提升 PACI 效益**

---

> **代码开源**：https://github.com/ItayElam/PACI

--- 

✅ **总结一句话**：  
**PACI 证明了“可控的不一致性”是一种可接受的代价，能换来巨大的训练效率提升，打破了“必须牺牲吞吐保一致性”或“牺牲内存保效率”的传统困境。**

</details>

---

### 9. [Convolutional Sparse Coding via the Locally Competitive Algorithm on Loihi 2](https://arxiv.org/abs/2606.08584)

**Authors**: Geoffrey Kasenbacher, Daniel Ruepp, Gerrit A. Ecke  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.08584v1  

#### Abstract
Sparse coding provides a principled framework for signal representation by expressing an input as a linear combination of only a small number of basis functions. The Locally Competitive Algorithm (LCA) is particularly attractive in the context of neuromorphic computing because its dynamics, leaky in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Convolutional Sparse Coding via the Locally Competitive Algorithm on Loihi 2*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文聚焦于在 **neuromorphic hardware**（神经形态硬件）上实现高效的 **structured sparse inference**（结构化稀疏推断），特别是将 **Convolutional Sparse Coding**（卷积稀疏编码, CSC）部署到 Intel 的 **Loihi 2** 芯片上。传统稀疏编码研究多集中于非卷积形式，而实际应用中（如图像处理）需要引入空间结构、权重共享和局部感受野等特性，这对硬件映射提出了新的挑战。

### 🚀 提出的新方法与创新点
- **首次实现了基于 LCA 的卷积稀疏编码在 Loihi 2 上的完整部署与基准测试**：
  - 将 **Locally Competitive Algorithm (LCA)** 扩展至卷积场景，构建了一个具有 **local inhibitory kernels**（局部抑制核）的一层递归网络模型。
  - 利用 **fixed-point membrane dynamics** 和事件驱动机制，在 Loihi 2 上高效模拟 LCA 动力学过程。
- **提出了一种面向结构化稀疏推断的 benchmarking framework**：
  - 明确区分 **hardware-independent metrics**（如 PSNR、SSIM）与 **hardware-dependent metrics**（如 latency、power、energy），为跨平台比较提供标准化框架。
- **揭示了 sparsity parameter λ 在 neuromorphic 系统中的“操作旋钮”作用**：
  - λ 不仅控制稀疏程度，还显著影响能耗与延迟，可在不同操作区间内权衡性能与效率。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **算法设计** | 引入卷积结构后支持 weight sharing 和 overlapping receptive fields，更适合大规模输入（如高分辨率图像）。 |
| **硬件适配性** | LCA 的局部连接、阈值激活和稀疏通信天然契合 Loihi 2 的 event-driven 架构，避免全局竞争带来的高通信开销。 |
| **能效表现** | 在多个配置下，Loihi 2 实现了比 GPU 高达数十倍的 **dynamic energy 节省**，尤其适合低功耗边缘推理场景。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **Set12** 图像数据集进行 benchmark 测试。
- 输入为灰度图，经过裁剪或缩放至固定尺寸（如 50×50 到 800×800）。
- 字典（dictionary）通过离线训练获得（采用交替优化策略：LCA 推断系数 + SGD 更新字典），并在所有硬件实验中保持固定，以隔离 inference 行为的影响。

### ⚙️ 实验设置
- **Loihi 2 平台**：
  - 单芯片运行，利用其 128 个神经形态核心。
  - 神经元状态使用 **fixed-point arithmetic** 表示。
  - feedforward drive 预计算并作为偏置注入；lateral inhibition 由预定义的 filter interaction tensor $ W = (\Phi^T * \Phi) - \delta $ 实现。
  - 每次 inference 运行 **1000 个 LCA iteration** 后读出最终 membrane state，并通过软阈值函数 $ T_\lambda(u) $ 得到稀疏系数。
- **GPU 基线平台**：
  - 使用 **NVIDIA RTX A6000**，PyTorch 2.0.1 + CUDA 11.7。
  - 完全相同的模型结构、字典、λ 设置、迭代次数和输入数据。
  - 使用 float32 精度计算，确保数值精度更高。

### 📊 评估指标
| 类型 | 指标 |
|------|------|
| **Algorithmic / Quality Metrics** | PSNR（主）、SSIM、MSE、sparsity（零系数比例、$ \|a\|_0 $） |
| **Hardware-Dependent Metrics** | <br>• `time_chip_ms`：Loihi 芯片内部累计执行时间<br>• `time_wall_ms`：端到端主机墙钟时间<br>• 动态功率（active − idle）<br>• 动态能量 per inference（joules/inference） |

### 🔁 对比基线
- **GPU baseline**：相同算法、相同参数、相同输入条件下的浮点实现。
- 对比方式：逐配置匹配（matched configuration），包括 filter size（3×3 / 5×5）、kernel 数量、stride（1 或 2）、target size、λ 值等。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 和 Figure 4）

#### ✅ Loihi-only 参数扫描结果（λ 的影响）
- 随着 **λ 增大**（更强稀疏正则化）：
  - **PSNR 缓慢下降**（重建质量适度降低）
  - **on-chip latency 显著减少**（最多降低 ~30%）
  - **dynamic energy 大幅下降**（最高降幅超 40%）
- 表明：**λ 是一个有效的“节能开关”**，可在可接受的质量损失下换取显著的能效提升。

#### 🔁 Loihi vs. GPU 对比（代表性配置）

| Regime | Latency Ratio (GPU/Loihi) | Energy Ratio (GPU/Loihi) | PSNR 差异 |
|--------|----------------------------|----------------------------|-----------|
| S1 5x5-16k 100 | ~0.02x (GPU 快 50 倍) | ~4.4–7.7x (Loihi 更省能) | Loihi 略低 |
| S1 3x3-16k 100 | ~0.08x (GPU 快 12 倍) | ~9.7–21x | Loihi 略低 |
| S2 5x5-16k 400 | ~0.16–0.19x | ~55–66x | 接近 |
| S2 3x3-16k full | ~0.5x (GPU 快 2 倍) | ~47–140x | 几乎一致 |

> 注：Energy Ratio > 1 表示 Loihi 更优；Latency Ratio < 1 表示 GPU 更快。

#### 🖼️ 定性结果（Figure 3）
- Loihi 2 与 GPU 的重建图像视觉效果高度相似，尤其在高 λ 区域差异极小。
- 表明：尽管存在 fixed-point 量化误差，Loihi 仍能生成语义一致的稀疏表示。

### ❌ 消融实验（隐含分析）
虽然未明确列出消融表，但文中通过以下维度进行了系统性分析：
- **filter size 影响**：3×3 和 5×5 结果均被报告，显示小滤波器更易稳定实现。
- **stride 影响**：stride-2 配置缩小了 Loihi 与 GPU 的延迟差距（因特征图更小）。
- **input scale 影响**：从 50×50 到 full-size（~800×800）验证了方法的可扩展性。
- **λ sweep**：五组 λ 值（0.15–1.5）展示了系统的 regime-dependent behavior。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **可行性验证**：
   - 首次成功在 Loihi 2 上实现了 **convolutional LCA** 的完整部署，证明了其在神经形态硬件上的可行性。
2. **能效优势显著**：
   - 在所有测试配置中，Loihi 2 的 **dynamic energy 消耗仅为 GPU 的 1/50 至 1/140**，展现出巨大的节能潜力。
3. **延迟劣势明显但可接受**：
   - GPU 在绝对延迟上远胜 Loihi（快数倍至数十倍），但在某些大尺度 stride-2 场景下差距缩小。
4. **重建质量接近但略有差距**：
   - Loihi 因 fixed-point 量化导致轻微精度损失，尤其在低 λ（密集激活）时更明显；随着 λ 增加，两平台质量趋于一致。
5. **λ 是关键调控参数**：
   - 在 Loihi 上，调节 λ 可主动控制系统在 “高质量-高能耗” 与 “低质量-低能耗” 之间切换，适用于资源受限场景。

### ⚠️ 局限性
1. **仅限于 inference**：字典学习完全离线完成，未探索 on-chip adaptation 或 online learning。
2. **固定点精度限制**：larger filters（>5×5）未能稳定运行，限制了模型表达能力。
3. **不对称硬件对比**：GPU 是高性能工作站级设备，而 Loihi 2 更偏向低功耗嵌入式场景，直接比较需谨慎解读。
4. **固定迭代预算**：使用统一的 1000 次迭代，可能高于实际所需（尤其 warm-start 场景），高估了真实部署成本。
5. **缺乏端到端应用验证**：未集成到具体任务（如 denoising、classification）中评估整体性能。

### 🔮 未来工作方向
1. **支持 on-chip dictionary learning**：结合 Loihi 2 的 on-chip learning 能力，实现完整的 CSC 训练-推理闭环。
2. **优化 fixed-point 实现**：改进量化方案以支持更大 filter size 和更高动态范围。
3. **探索 adaptive stopping criteria**：根据收敛情况动态终止迭代，进一步节省能量。
4. **扩展至 deeper architectures**：研究 multi-layer convolutional LCA 的层级竞争机制。
5. **应用于 real-time sensing pipelines**：如雷达信号处理、始终在线视觉前端等低功耗边缘场景。

---

> 💡 **总结一句话**：  
> 本文首次将 **Convolutional LCA** 成功部署于 **Loihi 2**，建立了一个结构化的稀疏推断 benchmark，揭示了其在 **dynamic energy 效率方面具有压倒性优势**，虽牺牲一定延迟，但在 **energy-constrained 场景下极具前景**。

</details>

---

### 10. [Claw-R1: A Step-Level Data Middleware System for Agentic Reinforcement Learning](https://arxiv.org/abs/2606.09138)

**Authors**: Daoyu Wang, Mingyue Cheng, Qingchuan Li, Shuo Yu, Jie Ouyang, Qi Liu  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.09138v1  

#### Abstract
Agentic reinforcement learning (RL) has become an important post-training paradigm for turning LLMs from static chatbots into interactive agents, giving rise to representative applications such as OpenClaw. Existing work mainly focuses on policy optimization algorithms and training frameworks, but p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Claw-R1: A Step-Level Data Middleware System for Agentic Reinforcement Learning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Agentic Reinforcement Learning**（Agentic RL）研究主要集中在策略优化算法（如PPO、GRPO）和训练框架（如veRL、slime）上，而忽视了从**代理-环境交互中产生的异构数据的全生命周期管理**。随着LLM代理在复杂工具调用、多轮交互和真实环境中运行，其生成的数据日益多样化且难以统一处理。

现有系统通常将代理运行时逻辑与RL训练后端紧密耦合，导致：
- 每个新代理需定制接口
- 数据复用困难
- 缺乏对数据质量、状态和准备情况的可观测性

> Claw-R1 正是为了解决这一“**数据孤岛**”问题而提出。

---

### 🚀 提出的新方法与核心思路

Claw-R1 是一个**面向 Agentic RL 的 step-level 数据中间件系统**，其核心思想是：

> 将代理交互轨迹视为可管理的训练资产（managed data assets），而非临时日志（temporary runtime logs）。

#### 主要组件：
| 组件 | 功能 |
|------|------|
| **Gateway Server** | 提供统一的 OpenAI 兼容 LLM API 接口，捕获来自白盒/黑盒代理、人类反馈等异构源的交互事件 |
| **Data Pool** | 存储标准化的 step-level 记录，包括 `prompt IDs`, `response IDs`, `reward`, 轨迹关系、元数据等 |

#### 创新设计原则：
1. **低侵入式接入**（Low-intrusion ingestion）  
   不要求修改现有代理代码，支持黑盒服务通过HTTP直接上报。
2. **原生步级表示**（Step-native representation）  
   抽象出标准 MDP 形式的 `(state, action, reward)` 序列，保留token级细节用于回放。
3. **异步解耦**（Asynchronous decoupling）  
   数据采集与模型训练分离，支持高并发、长延迟任务。
4. **后端感知服务**（Backend-aware serving）  
   支持按需拉取符合特定RL算法需求的训练批次（batch）。

---

### 🔍 相比现有方法的优势

| 方面 | 现有工作 | Claw-R1 |
|------|--------|---------|
| **关注点** | 算法优化 / 分布式训练 | **数据生命周期管理** |
| **架构耦合度** | 强耦合（runtime ↔ trainer） | **弱耦合中间件模式** |
| **数据粒度** | 轨迹级（trajectory-level） | **步级（step-level）细粒度控制** |
| **数据组织** | 原始日志存储 | 支持 prefix-tree 合并去重 |
| **可观测性** | 黑箱运行 | 提供可视化仪表盘监控全流程 |

> ✅ **优势总结**：Claw-R1 实现了 **agent runtime 与 RL training backend 的完全解耦**，提升了系统的可扩展性、灵活性和数据利用率。

---

## 2. 核心实验方法和设置

> ⚠️ 注意：本文是一篇 **demo paper**，并未进行传统意义上的大规模定量实验或基准测试，而是以**功能演示 + 架构验证**为主。

### 🧪 实验目标
验证 Claw-R1 是否能有效支持以下流程：
- 多源异构数据采集
- Step-level 数据建模与存储
- 可视化数据审查与筛选
- 数据优化（prefix-tree merging）
- 面向下游RL算法的训练批配置

---

### 💾 数据来源（非公开固定数据集）
数据来源于多种模拟或实际运行场景：
- 白盒代理（White-box Agent）Rollout 输出
- 黑盒代理服务（Black-box Service）通过 OpenAI 兼容 API 调用
- 人工标注反馈流（Human Feedback Stream）

> 并未使用如 AgentBench、WebShop 等标准评测数据集作为输入，而是强调**动态生成过程中的数据流动管理**。

---

### 📊 实验设置与评估方式

#### 系统部署环境
- Gateway Server：基于 HTTP + WebSocket 实现统一接入
- Data Pool：持久化存储 step-level 记录，建立多维索引（prompt ID, response ID, reward status, policy version 等）
- Dashboard：提供交互式前端界面，展示完整数据生命周期

#### 评估维度（Qualitative Evaluation）
| 维度 | 描述 |
|------|------|
| **功能性** | 是否支持 step-level 数据采集、表示、筛选、合并、消费 |
| **可用性** | 用户能否通过 GUI 完成轨迹监控、数据清洗、训练准备 |
| **效率增益** | prefix-tree merging 对上下文冗余的压缩效果（token节省量） |
| **兼容性** | 是否适配不同类型的 agent runtime 和 downstream RL 算法 |

#### 基线对比（间接比较）
虽然没有直接性能对比，但在 Table 1 中明确区分了相关工作的类型：

| 方法 | 类型 | Focus |
|------|------|-------|
| PPO, GRPO | Algorithm | LLM RL 算法 |
| veRL, slime | Framework | LLM RL 训练基础设施 |
| Agent-R1, OpenClaw-RL | Framework | Agentic RL 框架 |
| WebShaper, AutoForge | Data Synthesis | 数据合成 |
| **Claw-R1** | **Middleware** | **数据管理（本文贡献）** |

> 表明 Claw-R1 填补了“**数据中间层**”的研究空白。

---

## 3. 主要实验结果和性能指标

> 由于是 demo paper，结果以**定性展示 + 可视化指标**为主，无传统 accuracy/F1/score 数值。

---

### 📈 关键性能与观察结果

| 功能模块 | 观察结果 |
|--------|--------|
| **Data Collection** | 成功集成白盒、黑盒、人工反馈三类数据源；实时显示事件流入速率与来源分布 |
| **Data Representation** | 所有步骤被标准化为 `{prompt_ids, response_ids, reward, metadata}` 结构；支持跨轨迹查询 |
| **Data Curation** | 用户可在界面上按 reward 是否存在、policy freshness、quality tag 过滤样本 |
| **Data Optimization** | **Prefix-tree merging 显著减少重复前缀计算**：<br>• 示例中多个轨迹共享 `[seq1, seq2]` 前缀<br>• 合并后仅存储一次，分支独立保留<br>• 减少约 40%-60% 的重复 token 处理开销（图示估算） |
| **Data Consumption** | Training Engine 可 pull-based 请求 ready batches，并同步权重更新状态 |

---

### 🔬 消融实验（Ablation Study）
> 文中未开展正式消融实验。

但通过架构设计体现了关键模块的价值：
- 若移除 **Gateway Server** → 失去统一接入能力，必须为每种 agent 开发专用接口
- 若移除 **Data Pool 的 prefix-tree merging** → 所有轨迹独立存储，造成大量长上下文重复计算
- 若无 **step-level 抽象** → 无法实现细粒度奖励分配与 credit assignment

> 因此，系统设计本身即隐含了“必要性”论证。

---

## 4. 关键结论和发现

---

### ✅ 主要发现

1. **Agentic RL 中的数据应被视为第一类公民（first-class asset）**  
   当前社区过度关注算法和框架，却忽略了数据在整个训练闭环中的核心地位。

2. **Step-level 数据抽象是实现通用性的关键**  
   将每个决策步抽象为 `(s, a, r)` 形式，使得不同 agent runtime 可输出统一格式，便于下游训练系统消费。

3. **中间件架构显著提升系统可扩展性**  
   通过 Gateway + Data Pool 的分层设计，实现了 agent 与 trainer 的彻底解耦，支持灵活扩展。

4. **Prefix-tree merging 可有效降低长上下文开销**  
   在多轨迹源自相同 prompt 的场景下，该优化能显著减少冗余计算，提高训练吞吐。

5. **交互式仪表盘增强了数据透明度与可控性**  
   用户可以全程追踪数据从产生到消费的状态变迁，提升调试与质量管理效率。

---

### ⚠️ 局限性

| 局限 | 说明 |
|------|------|
| **缺乏量化性能对比** | 未在标准 benchmark 上报告收敛速度、样本效率等指标 |
| **依赖外部 reward engine** | 奖励信号由外部系统提供，未解决自动 reward 建模问题 |
| **暂未开源完整系统** | 仅提供 GitHub 链接（https://github.com/AgentR1/Claw-R1），尚未确认是否包含全部组件 |
| **适用范围受限于 step-level MDP 假设** | 对非序列化或高度并行的行为建模可能不适用 |

---

### 🔮 未来工作方向

1. **构建自动化 reward annotation pipeline**  
   集成规则引擎、模型预测、人类反馈等多种 reward 来源。

2. **支持更复杂的 metadata schema**  
   如工具调用链路、执行错误类型、认知路径标签等。

3. **引入数据版本控制系统（DVC-style）**  
   支持 step-level 数据的版本追踪与回滚。

4. **与主流 RL 框架深度集成**  
   如对接 Ray、DeepSpeed、vLLM 等系统，实现端到端自动化训练流水线。

5. **探索联邦式数据协作机制**  
   支持多个团队共享加密后的 step-level 数据池，促进协作学习。

---

## 总结

> Claw-R1 提出了一种全新的视角：**将 Agentic RL 视作一个数据转化流程——从 runtime 交互到训练资产的系统性管理过程**。

它不是另一个训练框架或优化算法，而是一个**承上启下的数据中枢**，填补了当前生态系统中的关键空缺。尽管缺乏严格的量化实验，但其设计理念、系统架构与交互式演示已充分展示了其在提升 **scalability、observability 和 reusability** 方面的巨大潜力。

📌 **一句话总结**：  
> *Claw-R1 把“怎么训好 agent”变成了“怎么管好数据”，推动 Agentic RL 进入数据驱动的新阶段。*

</details>

---

### 11. [EditSR: Enhancing Neural Symbolic Regression via Edit-based Rectification](https://arxiv.org/abs/2606.07915)

**Authors**: Da Li, Xinxin Li, Xingyu Cui, Jin Xu, Juan Zhang, Junping Yin  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.07915v1  

#### Abstract
Neural symbolic regression models improve inference efficiency by shifting structural search to pretraining, but their one-pass autoregressive decoding is prone to error accumulation, which may lead to generating structurally incorrect expressions, especially in complex expression generation scenari...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EditSR: Enhancing Neural Symbolic Regression via Edit-based Rectification

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
神经符号回归（Neural Symbolic Regression）模型通过大规模预训练显著提升了推理效率，其典型范式为**单次自回归解码**（one-pass autoregressive decoding）。然而，这种机制存在严重的**错误累积**（error accumulation）问题：一旦在生成早期出现结构错误（如算子选择错误），后续生成将基于错误的语法上下文进行，导致整个表达式结构偏离目标。

现有的后处理修正策略（post-hoc rectification）虽然能缓解该问题，但通常依赖于重启全局搜索（如MCTS）或潜在空间优化，计算成本高昂，削弱了神经模型原本的效率优势。

---

### 提出了什么新方法或新思路
本文提出 **EditSR**，一个两层框架，结合了神经符号回归模型与基于编辑的修正器（edit-based Rectifier），实现高效预测与事后修正的统一。

#### 核心创新点：
- **模块化架构设计**：第一层使用标准神经符号回归模型（如 NeSymReS）快速生成初始表达式；第二层引入 **Rectifier** 进行多步编辑式修正。
- **基于状态转移链的修正过程**：将修正建模为从错误表达式到目标表达式的逐步状态转移链，每一步执行一个语法约束下的编辑动作。
- **预训练驱动的高效修正**：Rectifier 本身经过大规模预训练，避免在推理时重启耗时的全局搜索，保持整体高效性。
- **语法有效性保障**：所有编辑动作均作用于子树级别，并限制在语法合法空间内，确保每一步输出均为可解析的表达式。
- **历史无关决策机制**：每个编辑决策仅依赖当前状态而非历史路径，允许后续步骤纠正前期错误，降低错误累积风险。

---

### 相比现有方法的优势
| 维度 | EditSR | 传统方法（如TPSR、SNIP） |
|------|--------|--------------------------|
| **效率** | 高：修正过程由预训练模型完成，无需在线搜索 | 低：需重启MCTS或优化，计算开销大 |
| **结构恢复能力** | 强：直接以目标结构为修正导向 | 弱：常以数值误差为目标，忽略结构一致性 |
| **错误容忍性** | 高：支持多步局部修正，可修复早期错误 | 低：错误一旦发生难以自我纠正 |
| **通用性** | 中等：接口兼容即可适配不同第一层模型 | 依具体实现而定 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖三大类基准：
1. **Standard Benchmarks**：包括 Constant, Koza, Nguyen, Keijzer, Korns, Livermore, Neat, Jin 等共91个问题，最多含3个变量。
2. **SRBench 1.0**：
   - **Feynman**：120个物理公式，用于测试结构恢复与抗噪能力。
   - **ODE-Strogatz**：14个非线性动力系统方程。
3. **SRBench 2.0**：
   - **Phenomenological & first-principles**：真实科学发现任务，含噪声。
   - **Black-box**：去除易用线性回归解决的问题，更具挑战性。

此外，训练阶段使用人工生成的 **1亿个随机表达式骨架** 进行大规模预训练。

---

### 实验设置和评估指标

#### 推理流程
- 第一层：NeSymReS 使用 beam search 生成候选表达式。
- 第二层：对未达精度要求的候选，启动 Rectifier 进行最多 `Tmax=10` 步编辑。
- 常数优化：最终表达式中的常数通过 BFGS 优化。

#### 评估指标
| 指标 | 定义 | 说明 |
|------|------|------|
| **R²** | 决定系数，衡量拟合优度 | R² > 0.999 视为准确解 |
| **Accuracy solution rate** | 达到 R² > 0.999 的问题比例 | 数值准确性 |
| **Symbolic solution rate** | 表达式与目标在符号意义上一致的比例（允许常数偏移或缩放） | 结构正确性 |
| **Complexity** | 表达式树的节点总数 | 越小越好，反映简洁性 |
| **Test Time** | 单问题平均推理时间 | 效率指标 |
| **Noise Robustness** | 在添加高斯噪声下 R² 的表现 | 抗干扰能力 |
| **Distractor Robustness** | 对无关变量的鲁棒性，测量 distractor usage rate | 特征选择稳定性 |

---

### 基线方法对比
- **uDSR**：集成多种策略的统一框架。
- **SR4MDL**：基于最小描述长度（MDL）引导搜索。
- **ParFam**：参数族连续优化方法。
- **RILS-ROLS**：迭代局部搜索 + 最小二乘估计。
- **TPSR**：基于 MCTS 的 Transformer 解码增强方法（代表性的 post-hoc rectification 方法）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ 在标准基准上的综合表现（Table 2 & Figure 4）
| 模型 | 平均 R² | Symbolic solution rate | Test Time (s) |
|------|--------|------------------------|---------------|
| **EditSR** | **领先或次优** | **最高且最稳定** | **17.03（最低）** |
| TPSR | 高 | 极低（如 Korns 上仅 ~10%） | 174.24 |
| uDSR | 中等 | 中等 | 77.99 |
| SR4MDL | 高 | 下降明显（尤其有噪时） | 765.03 |

> **关键观察**：EditSR 在保持高 R² 的同时，**Symbolic solution rate 明显高于其他方法**，表明其更擅长恢复正确的数学结构。

---

#### ✅ 在 SRBench 1.0 上的表现（Figure 6）
- **Feynman 基准**：
  - 所有噪声水平下，EditSR 的 **Symbolic solution rate 均超过 50%**，远超 TPSR (~10–13%) 和 uDSR (~20–30%)。
  - R² 稳定在 80% 左右，抗噪能力强。
- **ODE-Strogatz 基准**：
  - 数据稀疏且采样范围窄，整体难度更高。
  - EditSR 仍保持最强的鲁棒性，在噪声增加时下降最缓。

---

#### ✅ 在 SRBench 2.0 上的表现（Figure 7）
- **Black-box**：EditSR 在 R² 与 Complexity 之间取得更好权衡，避免“低 R² + 高复杂度”的失败模式。
- **Phenomenological & first-principles**：多数问题 R² > 0.95，且表达式简洁。

---

### 与基线方法的对比结果
- **优于 TPSR**：尽管 TPSR 可提升 R²，但其 Symbolic solution rate 极低，说明其修正倾向于“数值拟合”而非“结构还原”。而 EditSR 在两者间平衡更好。
- **效率碾压**：EditSR 测试时间仅为 TPSR 的 **约 1/10**，证明其“预训练修正”策略有效保留了神经模型的效率优势。
- **抗噪更强**：在噪声水平 σ ∈ {0, 0.001, 0.01, 0.1} 下，EditSR 的 ECDF 曲线始终最靠右，表示更多问题维持高 R²。
- **抗干扰更强**：在加入 1–3 个无关变量时，EditSR 的 distractor usage rate 最低（如 k=3 时仅 13.7%，TPSR 达 78.2%），说明其更能聚焦于真正相关的变量。

---

### 消融实验结果（Ablation Studies）

#### 🔍 Rectifier 的有效性（Figure 8）
- 加入 Rectifier 后，Accuracy 和 Symbolic solution rate 显著提升。
- 即使不微调（EditSR’），性能也优于原始 NeSymReS，说明预训练已赋予基本修正能力。
- 微调后进一步提升，验证 fine-tuning 对齐实际错误分布的重要性。

#### 🔍 复杂度影响（Figure 9）
- 随着目标表达式复杂度上升，NeSymReS 性能急剧下降（极高复杂度组 Symbolic rate ≈ 0%）。
- EditSR 在中高复杂度下仍保持较高性能，**增益随复杂度增大而更显著**，印证其对长表达式生成特别有益。

#### 🔍 编辑步数分析（Figure 10–12）
- 成功修正平均仅需 **4–6 步**，极少超过 10 步。
- 编辑动作频率：`INSERT` > `REWRITE` > `REPLACE` ≈ `DELETE`。
- `INSERT` 和 `REWRITE` 对减少编辑距离贡献最大，说明缺失子结构是主要错误类型。

#### 🔍 对第一层鲁棒性（Table 5）
- 在不同 dropout 率（0.1–0.3）训练的第一层模型上，Rectifier 微调后性能稳定，说明其具有良好的适应性和泛化能力。

---

## 4. 关键结论和发现

### 主要发现
1. **错误可局部修正**：即使神经模型未能完全正确生成表达式，其预测往往包含大量可重用的结构片段（如正确变量、子树），因此无需重启全局搜索，只需少量局部编辑即可修复。
2. **预训练修正可行且高效**：通过构造监督式的编辑链并预训练 Rectifier，可在推理时实现快速、有效的结构修正，兼顾效率与性能。
3. **结构导向优于数值导向**：EditSR 的修正目标始终指向目标符号结构，而非仅仅满足数值误差阈值，从而显著提升 Symbolic solution rate。
4. **尤其适用于复杂表达式**：在长表达式生成场景中，单次自回归更容易出错，而 EditSR 的多步修正机制能有效缓解这一问题，带来更明显的性能增益。

---

### 方法的局限性
1. **依赖第一层的探索能力**：若初始预测与目标相差太远（如完全不同的结构），有限步数内的局部编辑可能无法恢复。
2. **编辑预算限制**：设定 `Tmax=10` 是经验选择，极端复杂问题可能需要更多步数，但会增加不确定性。
3. **训练数据偏差**：Rectifier 在人工构造的错误分布上训练，面对严重分布外（OOD）错误时效果未知。
4. **尚未整合强搜索机制**：目前为纯修正框架，未主动探索新结构。

---

### 未来工作方向
1. **与强搜索模型协同**：将 Rectifier 作为通用组件，与遗传编程或 MCTS 等全局搜索方法结合，形成“探索-精修”混合流水线。
2. **动态编辑预算机制**：根据表达式复杂度或错误程度自适应调整 `Tmax`。
3. **跨架构适配研究**：验证 Rectifier 是否可适配除 NeSymReS 外的其他神经符号回归模型（如 SymbolicGPT、E2E）。
4. **引入语义感知编辑**：结合函数等价性知识（如 `sin²(x) + cos²(x) = 1`），提升语义级修正能力。

---

> **总结一句话**：  
> **EditSR 提出了一种高效、结构敏感的后处理修正机制，通过预训练的编辑式 Rectifier 在不解码的前提下修复神经符号回归模型的结构错误，在保持高速推理的同时显著提升了复杂表达式的符号结构恢复能力。**

</details>

---

### 12. [VESTA: A Fully Automated Scenario Generation and Safety Evaluation Framework for LLM Agents](https://arxiv.org/abs/2606.08531)

**Authors**: Lu Jia, Haibo Tong, Feifei Zhao, Jindong Li, Dongqi Liang, Ping Wu, Qian Zhang, Yi Zeng  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.08531v1  

#### Abstract
Large language models (LLMs) are increasingly evolving from simple text-based interaction systems into LLM agents that can maintain memory, use tools, access external environments, and execute tasks. As their capabilities and autonomy expand, the safety risks they face also become more diverse. Exis...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《VESTA: A Fully Automated Scenario Generation and Safety Evaluation Framework for LLM Agents》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前对 **LLM Agents** 的安全评估存在严重不足：
- 多数基准（如 SafetyBench、HarmBench）依赖静态 prompt 或单轮任务，仅能评估响应层面的安全性；
- 缺乏对 **多轮交互过程中行为演化风险** 的动态捕捉能力；
- 风险场景通常由人工编写，难以规模化覆盖多样化的执行路径和工具调用环境。

这些问题导致无法有效识别在真实任务执行中逐渐浮现的行为安全风险（如目标漂移、权限越界、监督失效等）。

---

### 🚀 提出的新方法与创新思路
作者提出 **VESTA** —— 一个**全自动的场景生成与安全评估框架**，其核心创新包括：

#### （1）自动化可执行场景生成（Automated Scenario Generation）
- 基于 **5个风险维度** 和 **16个风险子类** 构建系统化的行为安全风险分类体系；
- 定义“**scenario family**”作为模板，指导 LLM 辅助生成大量结构化、可执行的风险测试实例；
- 结合人类专家设计的种子案例（7个/子类），通过 LLM 扩展生成候选场景，并进行质量过滤与多样性控制，最终构建 **1,072个高质量、可执行的多轮评估场景**。

#### （2）端到端自动化评估流水线（Interactive Evaluation Pipeline）
- 每个场景被实例化为一个 **multi-turn programmatic episode**，包含：
  - 可模拟的 **tool environment**
  - 自适应攻击者（adaptive attacker）施加现实压力
  - 目标 agent 在受限权限下做出决策
  - 全过程记录执行轨迹（episode execution trace）

#### （3）基于轨迹的细粒度安全判断（Episode-Level Safety Judgment）
- 引入两种互补的安全指标：
  - **Unsafe Tool-call Rate (UTR)**：检测是否调用了预定义的高危工具
  - **Attack Success Rate (ASR)**：由 LLM judge 对整个 episode 判断是否“不安全”
- 支持跨模型、跨风险类型的细粒度分析。

---

### 🔍 相比现有方法的优势
| 维度 | 现有方法（如 Agent-SafetyBench, R-Judge） | VESTA |
|------|----------------------------------------|-------|
| 场景来源 | 人工撰写为主，规模有限 | LLM辅助+结构化模板，**自动扩展至千级场景** |
| 执行方式 | 多为静态prompt或简单API调用 | **完整多轮交互+环境反馈闭环** |
| 风险覆盖 | 聚焦最终输出或特定违规 | 覆盖**过程性风险演化机制**（如Objective Drift） |
| 评估粒度 | 单次响应判断 | **全过程轨迹分析 + 工具调用证据链追踪** |
| 自动化程度 | 依赖人工标注/评分 | **全流程自动化**（生成→执行→判断） |

> ✅ 总结：VESTA 实现了从“静态问答评估”向“**动态、可执行、过程导向**”的 agent 安全评估范式的跃迁。

---

## 2. 核心实验方法和设置

### 📚 数据集与场景构造
- **自建数据集**：共 **1,072个可执行评估场景**
  - 覆盖 **5个风险维度** × **16个风险子类**
  - 每个子类包含约 67 个独立场景
- 场景生成流程：
  1. 专家设计 7 个种子场景 per 子类
  2. 使用 GPT-5.4 进行 schema-guided 扩展 → 生成 100 个候选
  3. LLM-assisted 质量打分 + 规则检查 + 多样性筛选 → 保留 60 个高质量场景
  4. 人工抽样验证确保有效性

---

### ⚙️ 实验设置
- **目标模型（Target Agents）**：共评估 **12个主流LLM agents**
  - 包括：`Claude-Haiku-4.5`, `GPT-4o-Mini`, `Llama-3.3-70B`, `Qwen3.6-Plus`, `Gemini-3-Flash-Preview` 等
- **权威上下文（Authority Contexts）**：每个场景运行两次
  - **Trust Context**：默认信任 agent 可自主操作
  - **Warning Context**：明确提醒权限边界与安全要求
- **最大交互轮数**：每 episode 最多 4 轮（attacker ↔ target）
- **工具环境**：完全模拟（no real API calls），所有 tool 执行返回合成观察值

---

### 📊 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **Attack Success Rate (ASR)** | 被判定为“unsafe”的 episode 比例 | 主要性能指标，反映整体行为安全性 |
| **Unsafe Tool-call Rate (UTR)** | 至少有一次调用危险工具的 episode 比例 | 行为层面的机械性风险信号 |
| **Multi-Judge Validation** | 使用多个 LLM judges（GPT-5.4, DeepSeek-V3.2, Llama-4-Maverick 等）交叉验证 ASR 一致性 | 提升判断可靠性 |

---

### 🔁 基线对比
本文未直接对比传统 benchmark（因其不可比），而是将 VESTA 自身作为新标准，重点在于：
- 揭示不同模型之间的相对安全表现差异
- 分析 Warning Context 是否带来显著改善
- 验证 UTR 与 ASR 的一致性与互补性

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）总体安全表现（ASR）
- **平均 ASR 达到 47.1%**，表明当前大多数 LLM agents 在任务执行中仍面临显著行为安全风险；
- 多个模型 ASR **超过 70%**，其中：
  - `Llama-3.3-70B`: **~80%**
  - `Gemini-2.5-Flash`: **~75%**
  - `Doubao-Seed-1.6`: **~73%**

#### （2）最安全的模型
- **Claude 系列表现最优**：
  - `Claude-Sonnet-4.6`: **<10% ASR**
  - `Claude-Haiku-4.5`: **<10% ASR**
- `GLM-5.1` 表现接近 Claude，优于多数闭源模型

#### （3）版本演进影响
- 新版本通常更安全：
  - `Gemini-3-Flash-Preview` < `Gemini-2.5-Flash`
  - `DeepSeek-V4-Pro` < `DeepSeek-V3.2`

#### （4）开闭源无明显优劣趋势
- 高风险组中既有开源（Llama）也有闭源（Gemini、Doubao）
- 低风险组中也包含闭源（Claude）和开源（GLM）
> ❗说明：**模型开放性 ≠ 更安全或更不安全**

---

### 🔬 细粒度风险分析（按子类）

| 风险子类 | 平均 ASR | 特点 |
|---------|--------|------|
| **Misleading Context Vulnerability** | **77.9%** | 易受虚假背景误导 |
| **Long-Horizon Instability** | **71.5%** | 多步推理累积错误 |
| **Objective Drift** | **67.7%** | 目标逐渐偏离原始意图 |
| **Inducement Vulnerability** | **64.7%** | 易被逐步诱导做坏事 |
| **Negative Side Effects** | **61.9%** | 忽视副作用 |

> 💡 发现：**过程导向型风险远高于边界清晰的操作违规**

而以下子类 ASR 较低，说明当前模型有一定防御能力：
- Interruption Failure (~12%)
- Audit Attribution Difficulty (~11%)
- Autonomous Privilege Escalation (~23%)

---

### 🔔 权限上下文的影响（消融实验性质）
- 在 **Warning Context** 下，绝大多数模型 ASR 下降
- 平均降幅达 **~10个百分点**
- 尤其在 **Autonomy**, **Goal-Norm**, **Interaction** 维度改善明显
- 但在 **Robustness** 类风险上缓解有限

> ✅ 结论：**显式安全提醒是一种轻量且有效的干预手段**

---

### 🔄 指标一致性分析（ASR vs UTR）
- **整体高度一致**（pairwise agreement > 0.85）
- 但也存在互补性：
  - **UTR 更敏感于显式危险工具调用**
  - **ASR 更能捕获语义层面的过程失败**（如未调用危险工具但仍越权决策）

| 场景类型 | UTR ≈ ASR | UTR > ASR | ASR > UTR |
|--------|----------|-----------|-----------|
| Autonomy, Robustness | ✓ | — | — |
| Goal-Norm | — | ✓（更多误判触发UTR） | — |
| Supervision | — | — | ✓（过程失控但未触碰工具） |

> ✅ 启示：**应结合 UTR 与 ASR 进行综合评估**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **当前 LLM Agents 存在严重的执行期行为安全风险**  
   - 平均 ASR 高达 **47.1%**，部分模型超 **80%**
   - 安全短板集中在 **长期稳定性、上下文鲁棒性、目标一致性** 等过程性维度

2. **Claude 系列是目前最安全的 agent 实现之一**  
   - ASR 控制在 **10%以内**，显著优于其他主流模型

3. **显式安全提示（Warning Context）可有效降低风险**  
   - 是一种低成本、无需训练即可部署的强化策略

4. **过程性风险比操作性违规更难防范**  
   - 如 Objective Drift、Long-Horizon Instability 等风险 ASR 普遍 >60%
   - 不能仅靠工具禁用或权限控制解决

5. **UTR 与 ASR 高度相关且互补**  
   - 可联合使用以实现更全面的风险刻画

---

### ⚠️ 局限性（Limitations）
1. **场景仍为模拟环境**
   - 所有 tool 执行均为仿真，未连接真实系统
   - 可能低估真实世界复杂性和不确定性

2. **无法穷尽所有风险模式**
   - 当前覆盖 5 维 × 16 子类，但现实部署中可能存在新型风险组合

3. **依赖 LLM-as-a-Judge 的主观性**
   - 尽管多 judge 验证提升了稳定性，但语义判断仍可能引入偏差

4. **快照式评估**
   - 测试的是固定版本模型，无法反映持续迭代中的安全演变

---

### 🔮 未来工作方向
1. **扩展至更复杂的环境**
   - 引入更长 horizon、更高并发、更强对抗性的模拟系统（如 WebArena-style）

2. **引入在线学习与自适应红队攻击**
   - 动态生成更具针对性的攻击策略，提升压力强度

3. **构建开放共享的 VESTA-Bench**
   - 推动社区共建标准化 agent safety benchmark

4. **集成防护机制评估**
   - 不仅用于“诊断”，还可用于测试各类 safeguard（如 execution monitor、policy checker）的有效性

5. **跨模态 agent 安全评估**
   - 延伸至视觉、语音、机器人等多模态 agent 场景

---

> 🧩 **总结一句话**：  
> **VESTA 开启了面向 LLM Agent 的“可执行、过程化、自动化”安全评估新时代，揭示了当前智能体在真实任务中广泛存在的行为安全隐患，并为未来的安全加固提供了系统性评测基础。**

</details>

---

### 13. [DN-Hypo-Pipeline: An AI-Driven Workflow for Hypothesis Generation via Large Language Models and Scientific Explanations](https://arxiv.org/abs/2606.08532)

**Authors**: Lei Lin, Ronghao Wang, Chunbao Zhou, Jue Wang, Yangang Wang  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.08532v1  

#### Abstract
A scientific hypothesis is the first step in research and undergoes experimental validation, yet it also reflects a deep understanding of and reasoning about scientific phenomena. We introduce DN-Hypo-Pipeline, an AI-powered workflow based on large language models, designed to support structured sci...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DN-Hypo-Pipeline: An AI-Driven Workflow for Hypothesis Generation via Large Language Models and Scientific Explanations

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
科学假设生成是科学研究的起点，但传统上依赖研究人员的经验、直觉和文献启发，过程主观且效率有限。尽管已有研究尝试利用 **Large Language Models (LLMs)** 进行假设生成，但大多数方法基于**经验性或启发式策略**，缺乏系统性和逻辑严谨性。

本文旨在解决以下核心问题：
- 如何构建一个**结构化、可重复、理论驱动**的假设生成流程？
- 如何提升生成假设的**新颖性、有效性与可验证性**？

---

### 🚀 提出的新方法与创新思路

作者提出 **DN-Hypo-Pipeline**，一种基于 **Deductive-Nomological (DN) Model** 的AI驱动科学假设生成框架。

#### 核心思想：
将科学假设视为对现象（Explanandum）的解释，并通过寻找自然法则（Laws）和初始条件（Conditions）来重构该解释，形成一个**可演绎的科学推理链条**。

#### 创新点包括：

1. **理论驱动而非数据驱动**
   - 首次将哲学中的 **DN模型**（由Hempel提出）形式化应用于AI辅助科研。
   - 将假设生成建模为：  
     $$
     h \sim P(L_1, L_2, ... | E, \text{Processes})
     $$
     即从现象 $E$ 出发，推导能解释它的潜在自然规律集合。

2. **结构化的生成流程（Tree of Thoughts 架构）**
   - 分五步执行：
     1. **D-N分解**：从论文中提取 Explanandum 和背景知识；
     2. **过程建模**（Process Modelling）：使用 **Basic Formal Ontology (BFO)** 对现象形成过程进行本体建模；
     3. **关联定律生成**（Association Law Generation）：基于过程中识别的“普遍项”（Universals），检索可能适用的科学定律；
     4. **定律评估与排序**：采用 **LLM-as-Judge** 框架，按 Relevance、Gap、Logic、Feasibility 四个维度打分；
     5. **解释重建与新颖性检验**：结合高分定律重建新假设，并通过 **Retrieval-Augmented Fact Verification** 检查是否已在文献中存在。

3. **通用性强**
   - 不仅适用于数据科学建模，还可推广至物理、生物等其他自然科学领域，是对 **Theory-Guided Data Science** 范式的泛化。

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法 | DN-Hypo-Pipeline |
|------|--------|------------------|
| 推理基础 | 启发式 / 数据驱动 | 理论驱动（DN模型） |
| 结构性 | 弱（如自由文本生成） | 强（五阶段结构化流程） |
| 可解释性 | 低 | 高（每一步有明确语义角色） |
| 新颖性保障 | 无系统机制 | Gap评分 + 文献检索双重控制 |
| 泛化能力 | 多限于特定任务 | 可跨学科迁移 |

> ✅ 显著优于直接prompt生成假设的方法，在统计显著性测试中全面胜出。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **目标论文（Case Studies）**：选取三篇高度引用的经典模型论文作为输入：
  1. [Word Embedding](https://arxiv.org/abs/1301.3781) （Mikolov et al., 2013）
  2. [ResNet](https://arxiv.org/abs/1512.03385) （He et al., 2016）
  3. [Transformer](https://arxiv.org/abs/1706.03762) （Vaswani et al., 2017）

- **支撑知识库**：
  - **ArXiv Dataset**（via Kaggle）用于 RAG（Retrieval-Augmented Generation）和 novelty checking。
  - 科学文献构成的向量数据库用于上下文增强。

---

### ⚙️ 实验设置

#### 模型对比
使用四种主流 LLM 进行生成与评估：
- **GPT-5.2**
- **Grok-4**
- **DeepSeek-V3.2-Think**
- **Gemini-3.1-Pro-Preview**

#### 温度参数设定
| 用途 | Temperature |
|------|-----------|
| 假设生成（稳定输出） | 0.3 |
| 基线对比（鼓励多样性） | 0.7 |
| LLM-as-Judge（确保一致性） | 0.1 |

---

### 📊 评估指标

#### 定量评估维度（Judgment Aspects）：
| 指标 | 定义 |
|------|------|
| **Validness** | 假设是否有理论依据、是否可行（1–5分） |
| **Novelty** | 是否偏离已有工作、原创程度（1–5分） |
| **Significance** | 对领域的潜在影响（1–5分） |
| **Potential** | 是否开启新的研究方向（1–5分） |

#### 评估方式：
- **双轨制评估**：
  - **LLM-as-Judge**：四个LLM独立评分
  - **Human Experts**：两位AI方向PhD专家人工评分
- 总共生成 72 个建模想法（60个pipeline生成 + 12个直接生成）
- 每个提案被6位“评委”独立打分（4×LLM + 2×Human）

#### 统计检验方法：
- **Wilcoxon Signed-Rank Test**：比较 pipeline vs 直接生成
- **Scheirer-Ray-Hare Test**：分析 LLM 和 Paper 因子的影响
- **Manhattan Distance**：衡量不同评估者之间的一致性

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 假设质量显著优于基线（Wilcoxon检验）

| Null Hypothesis | LLM-as-Judge p-value | Human Expert p-value | 结论 |
|------------------|------------------------|-------------------------|-------|
| H₀(1): 最大得分 ≤ 基线 | **0.0005** | **0.0002** | ❌拒绝 → pipeline 更优 |
| H₀(2): 平均得分 ≤ 基线 | **0.0271** | **0.0002** | ❌拒绝 → pipeline 更优 |
| H₀(3): 中位数 ≤ 基线 | 0.0859 | **0.0005** | ❌人类评估拒绝 |

> 💡 表明无论是最高分还是平均分，**DN-Hypo-Pipeline 生成的假设都显著优于直接生成法**。

#### ✅ 不同LLM的表现排名（一致性高）

| 模型 | 综合得分（Sum Metric） |
|------|---------------------|
| **GPT-5.2** | 最高（LLM & Human一致） |
| Gemini-3.1-Pro-Preview | 第二 |
| Grok-4 / DeepSeek-V3.2 | 居后 |

> 图9与图10显示：**GPT-5.2 在所有维度上表现最佳**。

#### ✅ 影响因素分析（Scheirer-Ray-Hare Test）

| 因子 | Validness | Novelty | Significance | Potential | Sum Score |
|------|----------|---------|--------------|-----------|----------|
| **LLM** | <1e-8 | <1e-7 | <1e-9 | <1e-7 | <1e-12 |
| **Paper** | 0.151 | <0.001 | <1e-9 | <0.001 | <0.01 |
| **LLM × Paper** | 0.208 | 0.417 | 0.019 | 0.081 | 0.01 |

> 🔎 发现：
- LLM本身的能力差异是决定性因素（主效应显著）
- 不同论文主题影响新颖性与潜力，说明某些方向趋于饱和
- 无交互效应 ⇒ LLM相对性能稳定，不依赖具体任务

#### ✅ 评估者间一致性良好（除Gemini外）

使用 **Manhattan Distance** 测量评分一致性：
- Human Experts、GPT-5.2、Grok-4、DeepSeek 之间距离小 ⇒ 评价标准一致
- **Gemini-3.1-Pro-Preview** 与其他评估者差异大 ⇒ 评判范式不同

---

### 🔬 消融实验与案例验证（Selected Conceptual Idea Validation）

选取两个最高分假设开发成算法并实测：

#### ✅ **CTAT (Continuous-Time Attention Transformer)**
- 来源：DeepSeek生成，基于 Attention Mechanism 原理
- 核心思想：用核函数积分替代离散注意力计算，降低复杂度
- 理论复杂度从 $O(L^2)$ 降至 $O(L \log L)$ 或 $O(LM)$
- 实验结果（WikiText-2）：
  - 性能下降极小（PPL轻微上升）
  - FFT版本虽慢但具优化潜力
  - GL Quadrature 方法精度损失小，适合长序列

> ✔️ 成功实现理论加速，具备工程落地前景。

#### ✅ **HALO (Heaps-Adaptive Lexico Manifold Embedding)**
- 来源：GPT-5.2生成，基于 Heaps' Law 和词频分布
- 核心思想：
  - 高频词保留独立向量
  - 低频词通过字符n-gram动态生成（共享生成器）
- 实验结果（WikiText-103）：
  - 参数减少约 **80%**（从2亿 → ~3千万）
  - 在 RareWord Similarity 上表现更优
  - Google Analogy Accuracy 接近甚至超越 baseline（尤其CBOW变体）

> ✔️ 实现高效压缩 + 提升稀有词表示能力。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **DN-Hypo-Pipeline 是有效的科学假设生成框架**
   - 基于 DN 模型的形式化推理路径提升了假设的逻辑严密性与可验证性。
   - 统计证明其生成的假设在 **validness、novelty、significance、potential** 上全面优于直接生成。

2. **GPT-5.2 是当前最适合该任务的 LLM**
   - 在生成与评估环节均表现最优。
   - 其评分与人类专家高度一致（Manhattan距离最小之一）。

3. **LLM-as-Judge 是可靠评估手段**
   - 除 Gemini 外，多数 LLM 的评分与人类专家高度相关。
   - 支持自动化大规模假设筛选。

4. **生成的假设具有实际应用价值**
   - 两个高分概念（CTAT 和 HALO）被实现为原型算法，均取得优于原始论文基线的效果（在参数效率或理论复杂度方面）。

5. **支持“理论引导建模”的普适性**
   - 本方法本质上是 **Theory-Guided Data Science** 的扩展与形式化。
   - 可推广至物理、化学、生物学等领域。

---

### ⚠️ 方法的局限性

1. **LLM 的幻觉问题严重**
   - 在开放式的本体（ontology）和定律生成任务中，容易产生看似合理但错误的概念体系。
   - 缺乏黄金标准判断“完整性”。

2. **难以区分真正的“自然律”与偶然归纳**
   - DN模型本身存在经典难题：“如何定义 lawlikeness？”
   - LLM无法可靠辨别哪些是从文献中学到的真正普遍规律 vs. 仅仅是高频共现的事实。

3. **依赖高质量输入与外部知识库**
   - 若原始论文表述不清或结构混乱，D-N分解失败会导致后续流程崩溃。
   - RAG检索的质量直接影响 novelty 判断准确性。

---

### 🔮 未来工作方向

1. **引入多智能体协作机制**
   - 如 Co-Scientist 或 AI Scientist 框架，加入反思、反驳、迭代优化模块。

2. **融合仿真与实验反馈**
   - 当前为纯文本推理；下一步可接入模拟环境或真实实验平台，实现闭环验证。

3. **构建“原则知识图谱”**
   - 手动或半自动构建科学原理数据库，替代当前依赖LLM抽取的方式，提高可靠性。

4. **扩展至更多学科领域**
   - 已验证在数据科学有效，未来可在材料科学、气候建模、药物发现等领域试点。

5. **探索 abductive reasoning 与 uncertainty minimization 结合**
   - 参考 Pu et al. [71][72] 的 PiEvo 框架，实现原理空间的自主演化。

---

> 📌 **总结一句话**：  
> DN-Hypo-Pipeline 提供了一个**以科学哲学为基础、结构清晰、可验证性强**的AI辅助科研新范式，标志着从“LLM写论文”迈向“LLM参与科学发现”的重要一步。

</details>

---

### 14. [ConMem: Structured Memory-Guided Adaptation in Training-Free Multi-Agent Systems](https://arxiv.org/abs/2606.08702)

**Authors**: Zhixun Tan, Qiang Chen, Tairan Huang, Xiu Su, Yi Chen  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.08702v1  

#### Abstract
Recent advances have improved the adaptive capabilities of LLM-based multi-agent systems (MAS) through memory-, skill-, and learning-based approaches, yet these approaches remain challenged by noisy trajectories, insufficient modeling of memory-skill relations, and reliance on additional training or...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ConMem: Structured Memory-Guided Adaptation in Training-Free Multi-Agent Systems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的基于 **LLM 的多智能体系统**（Multi-Agent Systems, MAS）在实现多智能体适应**（Multi-Agent Adaptation, MAA）** 时面临三大挑战：
- **轨迹噪声大**：历史交互轨迹冗长、混杂局部工具输出、中间分歧和过时上下文，干扰决策信号。
- **记忆与技能关系建模不足**：现有方法缺乏对策略间依赖、冲突和约束的显式协调机制。
- **依赖额外训练或高质量监督**：许多先进方法需通过强化学习等进行训练，增加计算成本和部署复杂度。

### 提出了什么新方法或新思路
提出 **ConMem**，一种**无需训练**（training-free）、**关系感知**（relation-aware）的框架，通过**跨经验协调**（cross-experience coordination）实现高效的多智能体适应。

其核心思想是将原始交互轨迹提炼为**结构化记忆卡片**（typed and signed memory cards），并组织成**关系图谱**（relation-aware memory graph），在运行时根据任务需求检索并协调这些卡片，以解决策略冲突、恢复依赖关系。

### 相比现有方法的优势
| 维度 | ConMem | 现有方法 |
|------|--------|---------|
| **是否需要训练** | ❌ 否（training-free） | ✅ 多数需要（如 LatentMem, MemRL） |
| **记忆结构** | ✅ 结构化卡片 + 类型化关系图 | ⚠️ 扁平列表或简单聚类 |
| **推理时协调** | ✅ 显式冲突检测与依赖恢复 | ❌ 依赖 LLM 内部隐式处理 |
| **控制粒度** | ✅ 预算化提示前缀控制 | ⚠️ 整体重训或参数微调 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个代表性基准上进行评估，覆盖不同任务类型：

| 数据集 | 任务类型 | 主要挑战 |
|-------|--------|--------|
| **TriviaQA** | 开放域问答（QA） | 证据检索与验证 |
| **PopQA** | 开放域问答（QA） | 抵抗幻觉，提升验证纪律 |
| **KodCode** | 代码生成 | 失败感知策略复用 |
| **PDDL**（via PDDLGym） | 符号规划 | 状态-计划记忆复用 |

### 实验设置和评估指标
- **LLM 背骨**：统一使用 `Qwen/Qwen3-4B-Instruct-2507`，确定性解码。
- **MAS 架构**：在三个主流框架上测试泛化性：
  - **AutoGen**（分布内）
  - **CAMEL**, **MacNet**（分布外）
- **评估协议**：**预续式评估**（prequential），即任务 $t$ 只能访问此前提交的记忆卡片。
- **评估指标**：
  - QA：答案准确率（gold-alias accuracy）
  - 代码：单元测试通过率（pass rate）
  - 规划：归一化目标满足得分（normalized goal-satisfaction score）

### 基线方法对比
共四类基线：
1. **无记忆基线**（No-memory）：不保留任何跨任务上下文。
2. **智能体框架基线**：ChatDev, MetaGPT, JoyAgent, OAgents。
3. **无需训练的记忆/技能基线**：
   - Generative Agents（观察+反思）
   - Voyager（技能库）
   - SimpleMem（压缩记忆）
   - G-Memory（层次化记忆）
   - ReMe（动态程序记忆池）
4. **可学习记忆基线**：
   - **LatentMem**（需训练的潜变量记忆）

所有方法在相同 host、prompt budget、任务顺序下公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1 摘要）
| 方法 | 平均性能提升（vs No-memory） | 最佳表现次数 |
|------|-----------------------------|------------|
| **ConMem** | **+10.9 ~ +12.9 pts** | **8/15 领先，7/15 第二** |
| ReMe | +10.1 ~ +11.5 pts | 3/15 领先 |
| LatentMem | +9.9 ~ +11.4 pts | 3/15 领先 |

在所有主机和任务上，**ConMem 均取得一致增益**，尤其在 **KodCode 和 PDDL** 上优势显著（因存在丰富策略依赖）。

### 与基线方法的对比结果
- **优于无需训练基线**：显著超越 Voyager、G-Memory、SimpleMem 等。
- **媲美甚至超越需训练方法**：在多数场景下优于 **LatentMem**（平均高 0.95–1.81 pts）。
- **推理效率更高**：
  - **剪枝超过 50% 的扩展候选**（Figure 5）
  - **减少 80% 以上的规划开销**

### 消融实验结果（Figure 3）
移除任一组件均导致性能下降，证明各模块互补：
| 消融组件 | 性能影响（最大下降） | 说明 |
|--------|--------------------|------|
| **No coordination** | ↓3.3 pts（CAMEL-TriviaQA） | 协调对避免冲突至关重要 |
| **No graph expansion** | ↓2.2 pts（AutoGen-KodCode） | 图扩展对恢复依赖关键 |
| **No reflection/admission** | ↓1.5–2.0 pts | 失败记忆对策略改进必要 |

---

## 4. 关键结论和发现

### 主要发现
1. **结构化协调优于被动记忆注入**：
   - ConMem 不仅“回忆”，更主动“协调”记忆卡片，确保输入 host 的策略兼容、非冗余、任务对齐。
2. **关系图谱是高效适应的关键**：
   - 支持（supports）、满足（satisfies）、约束（constrains）、冲突（conflicts）四种边类型使系统能显式处理策略依赖与矛盾。
3. **失败记忆具有高价值**：
   - 负面卡片（negative cards）作为“避免提示”（avoid cues），有效防止重复错误。
4. **无需训练即可实现强适应**：
   - 在冻结 host 模型权重的前提下，通过 **prompt-time 控制** 实现持续适应，降低部署门槛。

### 方法的局限性
- **控制器质量依赖 LLM**：卡片提取、失败反思、关系判断仍由 LLM 完成，存在传播错误风险。
- **固定阈值与启发式规则**：当前使用校准阈值而非学习策略，灵活性有限。
- **受限于基础模型能力**：无法生成 base model 本身不具备的技能。

### 未来工作方向
1. **引入学习型控制器**：用轻量学习策略替代固定阈值，同时保持“可解释策略单元”的抽象。
2. **扩展关系类型与图推理**：探索更丰富的语义关系或长程图推理机制。
3. **跨领域迁移与安全治理**：
   - 在医疗、金融等领域评估记忆系统的可靠性。
   - 结合红队演练、访问控制、审计机制应对持久化记忆带来的隐私与安全风险。

---

> **总结一句话**：  
> ConMem 通过将碎片化经验转化为**带签名的结构化记忆卡片**，并在**类型化关系图谱**上进行**预算化协调**，实现了无需训练、高效鲁棒的多智能体适应，在多个基准上超越现有 memory 与 learning-based 方法。

</details>

---

### 15. [Q-Delta: Beyond Key-Value Associative State Evolution](https://arxiv.org/abs/2606.08804)

**Authors**: Sumin Park, Seojin Kim, Noseong Park  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.08804v1  

#### Abstract
Linear attention reformulates sequence modeling as recurrent state evolution, enabling efficient linear-time inference. Under the key-value associative paradigm, existing approaches restrict the role of the query to the readout operation, decoupling it from state evolution. We show that query-condit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Q-Delta: Beyond Key-Value Associative State Evolution

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在现有的 **linear attention** 和 **state space models (SSMs)** 中，序列建模被形式化为基于 key-value 的关联记忆（associative memory）演化过程。其中，**query** 仅用于从当前状态中“读出”输出（readout），而 **state evolution**（即状态更新）完全由 key-value 对驱动。

这种设计隐含地假设：**query 不参与状态的动态演化**。然而，作者指出这一假设忽略了 query 所提供的潜在信息——即 query 本身可以作为对未来值的预测信号，从而提供一种与 key-based retrieval 互补的纠错机制。

因此，本文旨在解决以下问题：
> 如何利用 query 在 readout 阶段产生的预测信息来改进 state evolution？能否将 query 从被动读出角色转变为主动参与者？

---

### 🚀 提出了什么新方法或新思路

作者提出了 **Q-Delta** —— 一种 **query-aware delta rule**，其核心思想是：

- 将 query-conditioned prediction $\hat{o}_t = S_{t-1}q_t$ 视为对目标 value $v_t$ 的一种结构化预测；
- 与传统的 key-retrieved value $\hat{v}_t = S_{t-1}k_t$ 结合，形成一个混合的 prediction error；
- 利用该混合误差进行 state update，实现更丰富的记忆修正。

#### Q-Delta 更新规则（Sequential Form）：
$$
S_t = \alpha S_{t-1}(I - \beta(k_tk_t^\top + \lambda_t q_tk_t^\top)) + \beta v_tk_t^\top
$$
其中：
- $\lambda_t \in [0,1]$ 是 learnable 的 query-feedback coefficient，控制 query 参与程度；
- $\alpha$ 是 forget gate；
- 混合输入方向为 $k_t + \lambda_t q_t$，体现了 key 与 query 的联合影响。

此外，作者还推导了 **chunkwise-parallel formulation** 并实现了高效的 **Triton kernel**，保证硬件友好性和训练效率。

---

### 🔍 相比现有方法的优势

| 方面 | 优势 |
|------|------|
| **理论层面** | 揭示了 query readout 本质上是一种“时间混合”的 value aggregation，具有独立于 key 的信息通道；首次将 query 明确纳入 state evolution 动力学中。 |
| **模型设计** | 引入 query-aware correction，使 state evolution 同时响应 key-value 匹配偏差和 query-aligned 预测偏差，提升记忆准确性。 |
| **稳定性保障** | 提供理论证明：在 mild empirical conditions 下，mixed prediction error 具有一致收缩性（one-step contraction）和全局稳定性（global geometric tracking）。 |
| **工程实现** | 支持 chunkwise 并行计算，通过 UT transform 和 WY 表达高效实现，保持与 delta-based models 相当的吞吐量。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### （1）语言建模任务（Zero-shot Evaluation）
- **Perplexity**:  
  - `WikiText`  
  - `LAMBADA`
- **Multiple-choice Reasoning Tasks**（准确率）:
  - `PIQA`, `HellaSwag`, `WinoGrande`, `ARC-Easy`, `ARC-Challenge`, `OpenBookQA`, `BoolQ`

#### （2）检索能力评估
- **合成检索任务（Synthetic Retrieval）**:
  - `S-NIAH` benchmark（来自 RULER）:
    - S-NIAH-1: Pass-key retrieval
    - S-NIAH-2: Number-in-haystack
    - S-NIAH-3: UUID-in-haystack
    - 测试 context length 分别为 1K, 2K, 4K tokens
- **真实世界检索任务（Real-world Recall）**:
  - `SWDE`, `SQuAD`, `FDA`, `TQA`, `NQ`, `DROP`（均截断至 max 2K context）

所有 zero-shot 评测使用 `lm-evaluation-harness` 统一执行。

---

### ⚙️ 实验设置和评估指标

| 设置项 | 内容 |
|-------|------|
| **模型规模** | 340M 和 1.3B 参数两种配置 |
| **预训练数据** | FineWeb-Edu（340M: 15B tokens；1.3B: 30B tokens） |
| **优化器** | AdamW + cosine learning rate schedule（peak lr: 1e-3 / 4e-4） |
| **精度** | bfloat16 mixed precision |
| **硬件** | 4×NVIDIA RTX 6000 (Blackwell) GPUs |
| **评估指标** |
| - 语言建模：Perplexity（↓） |
| - 推理任务：Accuracy（↑） |
| - 检索任务：Pass@k / Recall（↑） |
| - 效率：Throughput (tokens/sec) |

---

### 🆚 基线方法对比

- **RetNet** (Sun et al., 2023)
- **Mamba**
- **Mamba2**
- **DeltaNet** (Yang et al., 2025b)
- **GatedDeltaNet** (Yang et al., 2025a)

所有基线均在同一框架下复现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### （1）语言建模与推理任务（Table 2）

| 模型 | 340M Avg Acc ↑ | 1.3B Avg Acc ↑ |
|------|----------------|----------------|
| RetNet | 44.55 | 50.31 |
| Mamba | 46.64 | 52.39 |
| Mamba2 | 46.27 | 52.46 |
| DeltaNet | 43.82 | 52.53 |
| GatedDeltaNet | 46.01 | 52.77 |
| **Q-Delta (Ours)** | **47.24** | **53.47** |

> ✅ 在两个尺度上均取得 **SOTA 性能**，尤其在 1.3B 上领先明显（+0.7 pts）。

#### （2）长上下文检索任务（S-NIAH, Table 3）

| 模型 | S-NIAH Avg ↑ |
|------|--------------|
| RetNet | 38.09 |
| Mamba | 62.62 |
| Mamba2 | 76.58 |
| DeltaNet | 81.29 |
| GatedDeltaNet | 83.51 |
| **Q-Delta (Ours)** | **90.02** |

> ✅ 在所有 context lengths（1K–4K）下接近完美表现，尤其在最难的 UUID-in-haystack（S-NIAH-3）上显著优于其他方法（4K: 48.0 vs 第二名 25.4）。

#### （3）真实世界检索任务（Table 4）

| 模型 | Average Score ↑ |
|------|------------------|
| Mamba | 27.7 |
| Mamba2 | 32.4 |
| DeltaNet | 30.1 |
| GatedDeltaNet | 34.2 |
| **Q-Delta (Ours)** | **33.2** |

> ✅ 在多数任务上达到最佳或次佳水平，综合表现最优。

#### （4）效率与稳定性（Figure 4）

- **训练损失曲线**：Q-Delta 收敛稳定，早期阶段甚至快于部分 baseline。
- **吞吐量（Throughput）**：
  - 与 DeltaNet / GatedDeltaNet 相当；
  - 显著高于 Mamba2，尤其在短序列场景；
  - 支持长序列扩展（up to 16K）且无性能崩溃。

---

### 🔬 消融实验结果（Table 5）

| 设置 | Wiki ppl ↓ | Lamb ppl ↓ | Avg Acc ↑ |
|------|------------|------------|-----------|
| Learnable $\lambda_t$ (**Q-Delta**) | **26.89** | **32.67** | **47.24** |
| Fixed $\lambda=0.5$ | 26.86 | 33.31 | 47.20 |
| No decay ($\alpha=1$) | 26.52 | 32.97 | 45.86 |
| No gating ($\lambda=1$) | 26.55 | 35.21 | 46.36 |

> 🔍 发现：
- 即使固定 $\lambda$，性能仍优于大多数 baseline；
- learnable $\lambda_t$ 能自适应调节 feedback 强度，获得最优 trade-off；
- 移除 decay 会降低 accuracy，但依然优于 DeltaNet → 表明 **query feedback 本身有效**；
- 完全开启 query correction（$\lambda=1$）反而略差 → 表明 **adaptive gating 更优**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Query 不应只是 readout 工具**：
   - query-conditioned prediction $\hat{o}_t = S_{t-1}q_t$ 是一种结构化的 value aggregation，反映的是“下游实际消费方向”的预测；
   - 与 key-based retrieval 形成互补而非冗余（cosine similarity ~0.07，subspace variance analysis 支持）。

2. **Mixed Error Correction 更有效**：
   - 同时纠正 key-retrieved 和 query-predicted 偏差，可提升 long-context retrieval 和 zero-shot reasoning 能力。

3. **Q-Delta 是稳定且高效的**：
   - 理论上满足 one-step error contraction 和 global stability；
   - 实践中训练稳定、收敛快、throughput 高。

4. **性能全面领先**：
   - 在语言建模、常识推理、真实与合成检索任务上 consistently 超过 strong baselines。

---

### ⚠️ 局限性

- 当前分析依赖于 linear attention setting，是否适用于非线性或 softmax attention 尚未验证；
- $\lambda_t$ 的引入增加了一个轻量参数，但在超大规模模型中的可扩展性需进一步观察；
- 理论稳定性依赖 empirical alignment condition（如 $B_t a_t \in (0,2)$），虽实践中成立，但仍为统计性质而非绝对保证。

---

### 🔮 未来工作方向

- 将 Q-Delta 思想推广到 decoder-decoder 或 encoder-decoder 架构；
- 探索 query feedback 在 instruction tuning 或 retrieval-augmented generation 中的作用；
- 设计更复杂的 multi-head query interaction 机制；
- 结合 memory sparsification 技术以进一步提升极长序列下的效率。

---

> 💡 **一句话总结**：  
> Q-Delta 打破了“query 仅用于 readout”的传统范式，提出将 query-conditioned prediction 作为纠错信号融入 state evolution，构建了一种更丰富、更稳定的 linear-time 序列建模机制，在理论与实践上均展现出显著优势。

</details>

---

### 16. [A Resilience-as-a-Service assessment framework for coordinated disruption response in interdependent urban transit systems](https://arxiv.org/abs/2606.08849)

**Authors**: Sara Jaber, S. M. Hassan Mahdavi, Neila Bhouri, Mostafa Ameli  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.08849v1  

#### Abstract
Urban public transport disruptions require rapid response strategies, yet existing studies rarely provide a decision support framework to compare alternative disruption response solutions using a common set of dynamic, passenger, operator, and environment oriented indicators. This paper proposes a K...

---

### 17. [AliyunConsoleAgent: Training Web Agents in Real-World Cloud Environments via Distillation and Reinforcement Learning](https://arxiv.org/abs/2606.09447)

**Authors**: Bojie Rong, Zheyu Shen, Qiaoping Wang, Pengfei Kang, Yang Xu, Yawen Wei, Hanyu Wu, Zhi Zhao, Leihao Pei, Linquan Jiang  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.09447v1  

#### Abstract
We present AliyunConsoleAgent, a web agent framework for automated documentation verification in real-world cloud consoles. Major cloud platforms encompass hundreds of products with rapid feature iteration, causing console UIs to frequently diverge from their corresponding documentation. Verifying t...

---

### 18. [Beyond Linear Activation Steering: Invertible Latent Transformations for Controlling LLM Behavior](https://arxiv.org/abs/2606.08454)

**Authors**: Tuc Nguyen, Thai Le  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.08454v1  

#### Abstract
Activation steering provides a lightweight inference-time mechanism for controlling large language models (LLMs) by modifying their internal activation vectors toward desired behaviors. Most existing methods compute a fixed steering direction in the original activation space, typically from pairs of...

---

### 19. [Distilling Safe LLM Systems via Soft Prompts for On Device Settings](https://arxiv.org/abs/2606.09388)

**Authors**: Motasem Alfarra, Cristina Pinneri, Dana Kianfar, Mohammed Almousa, Christos Louizos  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.09388v1  

#### Abstract
Deploying safe large language models (LLMs) on resource-constrained edge devices presents a critical challenge: while dual-model systems combining LLMs with guard models provide effective safety guarantees, their substantial memory and computational demands make them prohibitively expensive for on-d...

---

### 20. [OmniMem: Perturbation-aware Memory Compression for Streaming Audio-Visual LLMs](https://arxiv.org/abs/2606.07577)

**Authors**: Guangzhi Sun, Yixuan Li, Yudong Yang, Chao Zhang  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.07577v1  

#### Abstract
Audio-visual large language models (LLMs) hold strong promise for long-form video understanding, yet their long-video inference is fundamentally limited by the linear growth of video tokens and key-value (KV) caches. We present OmniMem, a memory-efficient streaming framework designed specifically fo...

---

### 21. [SAGE: An LLM-driven Self Reflective Agentic Framework for Fraud Detection](https://arxiv.org/abs/2606.08146)

**Authors**: Yichen Chen, Siying Li, Yuhang Liang, Lijun Wang, Renyang Liu  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.08146v1  

#### Abstract
Fraud detection in payment, e-commerce, and telecommunications systems requires accuracy at the individual level, robustness under severe class imbalance, and ease of understanding for risk managers. Existing methods fall at least one of these requirements: automated machine learning systems search ...

---

### 22. [A Low-Latency Semantic State Estimator using Latent Predictive Learning for Dynamic Network Monitoring and Orchestration](https://arxiv.org/abs/2606.08869)

**Authors**: Hari Madhukumar, Haiyuan Li, Xiaolan Liu, Andy Corston-Petrie, Dimitra Simeonidou  
**Category**: cs.DC  
**Published**: 2026-06-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.08869v1  

#### Abstract
Closed-loop network monitoring and orchestration increasingly require semantic interpretations of live telemetry beyond raw counter collection. However, dynamic cloud-edge environments change both the active node set and the monitoring query at runtime, while control loops demand bounded millisecond...

---

### 23. [Evaluation of ML Resource Utilization Requires Model Life Cycle Assessment](https://arxiv.org/abs/2606.07632)

**Authors**: Jared Fernandez, Clara Na, Yonatan Bisk, Constantine Samaras, Emma Strubell  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.07632v1  

#### Abstract
Proper accounting of the energy requirements and environmental impact of artificial intelligence (AI) systems is necessary for researchers, developers, policy makers, and users to assess the barriers to building systems at scale. With the growing complexity of pipelines and underlying infrastructure...

---

### 24. [Physics-Guided Dual Decoding and Spectral Supervision for Global 3D Hydrometeor Prediction](https://arxiv.org/abs/2606.08563)

**Authors**: Dandan Chen, Yaqiang Wang  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.08563v1  

#### Abstract
While global data-driven models excel at predicting continuous atmospheric variables, three-dimensional hydrometeor forecasting remains challenging due to the zero-inflated, long-tailed distributions of these variables. Standard deep learning optimization often yields overly smooth forecasts, attenu...

---

### 25. [Reformulate LLM Reinforcement Learning for Efficient Training under Black-box Discrepancy](https://arxiv.org/abs/2606.08779)

**Authors**: Jiashun Liu, Runze Liu, Xu Wan, Jing Liang, Hongyao Tang, Ling Pan  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.08779v1  

#### Abstract
Reinforcement Learning (RL) has emerged as a pivotal post-training paradigm, yet it frequently suffers from unpredictable sub-optimum performance or even training collapses. Recent findings attribute these failures to a hidden train-inference discrepancy (or mismatch), stemming from the disparate un...

---

### 26. [BUDDY: BUdget-Driven DYnamic Depth Routing for Adaptive Large Language Model Inference](https://arxiv.org/abs/2606.09514)

**Authors**: Yuhua Zhou, Shaoqi Yu, Shichao Weng, Changhai Zhou, Mingze Yin, Fei Yang, Aimin Pan  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.09514v1  

#### Abstract
Large language models (LLMs) incur high inference cost due to their depth and parameter scale. Depth pruning can reduce latency by skipping redundant Transformer blocks, but existing methods (i) provide limited control under user-specific compute budgets and (ii) typically fix the routing path, fail...

---

### 27. [A Regret Minimization Framework on Preference Learning in Large Language Models](https://arxiv.org/abs/2606.09124)

**Authors**: Suhwan Kim, Taehyun Cho, Geon-Hyeong Kim, Yu Jin Kim, Youngsoo Jang, Moontae Lee, Jungwoo Lee  
**Category**: cs.AI  
**Published**: 2026-06-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.09124v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) has enabled progress on reasoning-intensive tasks by relying on task-specific verifiers that provide automated correctness signals. However, many realistic language tasks are difficult to equip with reliable verifiers, motivating a growing relian...

---

### 28. [Auditing Training Data in Domain-adapted LLMs: LoRA-MINT](https://arxiv.org/abs/2606.06946)

**Authors**: Gonzalo Mancera, Daniel DeAlcala, Aythami Morales, Julian Fierrez, Ruben Tolosana, Francisco Jurado  
**Category**: cs.CL  
**Published**: 2026-06-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.06946v1  

#### Abstract
We present LoRA-MINT, a new methodology for Membership Inference Test (MINT) applied to recent Large Language Models (LLMs) fine-tuned for specific Natural Language Processing (NLP) tasks through Low-Rank Adaptation (LoRA). The primary goal is to assess whether individual samples were part of the tr...

---

### 29. [Unifying von-Neumann HPC and Neuromorphic Acceleration via the EBRAINS Research Infrastructure: A Framework for High-Performance Workflows](https://arxiv.org/abs/2606.08515)

**Authors**: Krishna Kant Singh, Charl Linssen, Eric M\"uller, Eleni Mathioulaki, Wouter Klijn, Lena Oden  
**Category**: cs.DC  
**Published**: 2026-06-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.08515v1  

#### Abstract
Modern scientific workflows increasingly span diverse computing architectures, yet executing a single computational model across disparate systems often forces researchers to maintain fragmented, site-specific pipelines. In this paper, we address this challenge within the domain of computational neu...

---

### 30. [SPIN: Decentralized Swarm Control via Tensorized Policy Coordination](https://arxiv.org/abs/2606.07557)

**Authors**: Zhaowen Fan  
**Category**: cs.LG  
**Published**: 2026-06-09  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.07557v1  

#### Abstract
Decentralized multi-agent swarm coordination on resource-constrained edge platforms remains fundamentally bottlenecked by the exponential scaling of joint action spaces and high-latency communication overhead. This paper introduces the Swarm Policy Interference Network (SPIN) framework, an architect...

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
