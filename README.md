# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-16 07:16:47 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [A KL Lens on Quantization: Fast, Forward-Only Sensitivity for Mixed-Precision SSM-Transformer Models](https://arxiv.org/abs/2604.13440)

**Authors**: Jason Kong, Nilesh Prasad Pandey, Flavio Ponzina, Tajana Rosing  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 14.5  
**Type**: new  
**ArXiv ID**: 2604.13440v1  

#### Abstract
Deploying Large Language Models (LLMs) on edge devices faces severe computational and memory constraints, limiting real-time processing and on-device intelligence. Hybrid architectures combining Structured State Space Models (SSMs) with transformer-based LLMs offer a balance of efficiency and perfor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**A KL Lens on Quantization: Fast, Forward-Only Sensitivity for Mixed-Precision SSM-Transformer Models**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
在边缘设备上部署大型语言模型（LLMs）面临严重的计算和内存限制。尽管混合架构（如 SSM-Transformer 混合模型）在效率和性能之间取得了良好平衡，但其组件对量化敏感度不一，**统一量化（uniform quantization）会导致某些关键层精度显著下降**，而现有敏感性分析方法（如基于梯度或 SQNR 的方法）在语言建模任务中表现不佳。

此外，许多场景下无法访问域内训练数据（由于隐私或商业限制），因此依赖反向传播或微调的方法不可行。

### ✅ 提出了什么新方法或新思路  
提出了一种**轻量级、无需反向传播（backpropagation-free）、基于前向推理的敏感性分析框架**，用于指导混合精度量化（mixed-precision quantization）：

- **核心思想**：通过仅使用前向传递（forward pass）输出来衡量每一层量化后对模型整体性能的影响。
- **关键创新**：
  - 引入 **KL divergence（Kullback-Leibler 散度）作为量化敏感性的代理指标**，特别是 `KL_student→teacher` 方向。
  - 形式化证明 KL 比广泛使用的 **SQNR（Signal-to-Quantization-Noise Ratio）更贴近语言建模任务的真实性能退化（如 Perplexity 变化）**。
  - 设计了一个完全前向的 pipeline，适用于资源受限且无训练权限的部署环境。

### ✅ 相比现有方法的优势  
| 维度 | 本方法 | 现有方法（如 HAWQ、SQNR-based） |
|------|--------|-------------------------------|
| 是否需要梯度/微调 | ❌ 不需要 | ✅ 通常需要 Hessian 或 retraining |
| 数据需求 | 仅需少量校准样本（inference-only） | 需要访问训练数据或大量统计信息 |
| 适用架构 | 特别适配 SSM-Transformer 混合模型 | 主要针对 CNN 或纯 Transformer |
| 敏感性指标有效性 | **KL_student→teacher 与 PPL 高度相关**（Kendall’s τ ≈ 0.79） | SQNR 与 PPL 关联弱，甚至出现误导排序 |
| 实际收益 | 在保持近 FP16 精度的同时实现高达 7.2× 压缩 | 均匀量化导致严重性能崩溃（如 Mamba-1.4B 上 PPL 从 11.25 升至 60.55） |

---

## 2. 核心实验方法和设置

### ✅ 使用的数据集  
- **WikiText-2**：用于评估语言模型的困惑度（Perplexity, PPL），是标准的小规模语言建模测试集。
- **少量校准样本**：用于执行前向敏感性分析（具体数量未明确说明，但强调“representative dataset”即可）。

### ✅ 实验设置和评估指标  

#### 模型架构
- **Hybrid SSM-Transformer 模型**：
  - **Hymba**（32-layer）
  - **Zamba**
  - **Mamba 系列**：Mamba-130M、Mamba-380M、Mamba-1.4B、Mamba2-130M
- 所有模型均包含 SSM 和 Transformer 子模块（如 attention、MoE、projection layers）。

#### 量化配置
- **混合精度策略**：高敏感层保留为 FP16，其余压缩为 INT4（CPU）或 INT8（GPU）。
- **敏感性排序依据**：每层单独量化后的 `KL_student→teacher` 值。
- **配置编号 p01–p10**：p01 最保守（只量化最不敏感层），p10 最激进（接近 uniform quantization）。

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Perplexity (PPL)** | 主要质量指标，越低越好 |
| **Model Size (MB)** | 模型体积，反映压缩率 |
| **Latency (ms)** | 推理延迟，越低越好 |
| **Throughput (FPS)** | 每秒帧数，越高越好 |
| **Kendall’s τ** | 衡量敏感性指标（KL/SQNR）与真实 PPL 变化的排序一致性 |
| **△PPL** | 单层量化引起的困惑度变化，作为 ground truth ranking |

#### 硬件平台
- **Intel Lunar Lake** 平台（集成 CPU + GPU + NPU），代表下一代 AI 边缘设备。
- 使用 **OpenVINO IR** 转换模型并进行端到端性能剖析。

#### 基线方法对比
| 基线 | 描述 |
|------|------|
| **FP16** | 全精度基准 |
| **Uniform INT8** | 统一 8-bit 量化 |
| **Uniform INT4** | 统一 4-bit 量化（当前工业主流） |
| **SQNR-based MP** | 基于信号噪声比的混合精度方案 |
| **HAWQ-like methods** | 基于 Hessian 的敏感性分析（需梯度） |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

#### 📊 Kendall’s τ 排序相关性（越高越好）
| Metric | Avg. Kendall’s τ |
|--------|------------------|
| **KL_student→teacher** | **0.7911** ✅（最高） |
| SQNR(dB) | 0.7111 |
| KL_teacher→student | -0.1275 |
| △Cross Entropy | -0.0645 |

> 💡 结论：`KL_student→teacher` 显著优于 SQNR（p < 10⁻⁶），能更准确预测哪一层量化会引发最大性能下降。

#### 📈 端到端硬件实测性能（Intel Lunar Lake）

##### Mamba-1.4B（CPU）
| Config | PPL | Size | FPS | Latency |
|--------|-----|------|-----|---------|
| FP16 | 11.22 | 5.2 GB | 2.6 | 384 ms |
| Uniform INT8 | 11.25 | 1.3 GB | 5.1 | 196 ms |
| Uniform INT4 | 60.55 | 723 MB | 5.3 | 190 ms ❌（PPL 崩溃） |
| **KL-MP p05 (ours)** | **11.22** | **1.4 GB** | **5.6** | **178 ms** ✅ |

> 🔥 成果：**模型大小仅为 FP16 的 1/3.7，吞吐提升 2.15×，延迟降低 53.6%，同时保持几乎相同的 PPL！**

##### Mamba2-130M（GPU）
| Config | PPL | Size | FPS | Latency |
|--------|-----|------|-----|---------|
| FP16 | 46.45 | 81 MB | 0.02 | 60,006 ms ❌（极高延迟） |
| KL-MP p02 (ours) | 46.46 | 45 MB | 0.29 | **3,417 ms** ✅ |
| Speedup | – | – | ↑14.5× | ↓**17.6× latency** |

> ⚡ GPU 上主要受益于 INT8 加速而非内存节省，但仍实现巨大延迟优化。

### ✅ 消融实验结果（Ablation Studies）

#### 层级敏感性分布（Hymba 模型）
- **mamba.x_proj** 是最敏感的子模块（平均 △PPL = 0.27），远超其他组件（<0.014）。
- **mamba.dt_proj** 几乎不受影响（△PPL ~ 3.6e-4），可安全用极低位宽表示。
- **Block 31** 是唯一的“热点层”，贡献了超过 70% 的总敏感性预算。

> 🧠 发现：**少数关键层主导整个模型的量化鲁棒性**，支持“top-k 高精度保护”策略的有效性。

#### 敏感性指标对比可视化
- 图 3 显示 `KL_student→teacher` 与 △PPL 呈强正相关，而 SQNR 分布散乱，存在多个反例（即 SQNR 高但 PPL 下降剧烈）。
- 支持理论命题：**SQNR 不是 PPL 的单调函数**（可通过常数偏移构造反例）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KL divergence（尤其是 `KL_student→teacher`）是比 SQNR 更优的语言模型量化敏感性指标**，因其直接作用于概率分布空间，与 Perplexity 有严格的数学联系（Lemma 2 & Corollary 1）。
2. **SSM-Transformer 混合模型具有高度非均匀的量化敏感性**，部分层（如 mamba.x_proj）极其脆弱，必须保留高精度。
3. **仅靠前向推理即可完成有效的敏感性分析**，无需梯度或微调，适合边缘部署和隐私敏感场景。
4. **KL-guided mixed-precision 量化可在 CPU/GPU 上实现接近 FP16 的 PPL，同时达到 Uniform INT4 的压缩水平和吞吐能力**，突破了传统方法的帕累托边界。

### ⚠️ 方法的局限性
- 当前方法假设各层误差独立，未考虑多层联合量化效应（虽然实践中 top-k 已足够有效）。
- KL 计算依赖 softmax 输出，可能受 temperature scaling 影响（文中未讨论归一化细节）。
- GPU 上因 embedding table 占比较大，内存压缩效果有限，主要优势体现在计算加速。

### 🔮 未来工作方向
- 将该框架扩展至更多类型的混合架构（如 RNN-Transformer、MoE-SSM）。
- 探索动态量化策略，在推理过程中自适应调整精度分配。
- 结合硬件感知编译器进一步优化 kernel 调度与内存布局。
- 研究更低比特（如 INT2/INT3）下的 KL 敏感性建模。

---

> ✅ **一句话总结**：本文提出一种基于 KL 散度的前向敏感性分析方法，首次系统揭示了 SSM-Transformer 混合模型的量化脆弱性模式，并实现了在 Intel Lunar Lake 上近无损压缩 7.2×、延迟降低 17.6× 的实际部署突破，为边缘智能提供了高效可靠的量化新范式。  
> 🔗 代码已开源：[https://github.com/jasonkongie/kl-ssm-quant](https://github.com/jasonkongie/kl-ssm-quant)

</details>

---

### 2. [OmniTrace: A Unified Framework for Generation-Time Attribution in Omni-Modal LLMs](https://arxiv.org/abs/2604.13073)

**Authors**: Qianqi Yan, Yichen Guo, Ching-Chen Kuo, Shan Jiang, Hang Yin, Yang Zhao, Xin Eric Wang  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2604.13073v1  

#### Abstract
Modern multimodal large language models (MLLMs) generate fluent responses from interleaved text, image, audio, and video inputs. However, identifying which input sources support each generated statement remains an open challenge. Existing attribution methods are primarily designed for classification...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*OmniTrace: A Unified Framework for Generation-Time Attribution in Omni-Modal LLMs*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 **Omni-Modal LLMs**（如 Qwen2.5-Omni 和 MiniCPM-o）能够处理文本、图像、音频、视频等交错输入并生成流畅的响应。然而，这些模型缺乏透明性——我们无法知道生成的每一句话具体依赖于哪些输入源（例如哪张图片、哪个时间戳的音频）。现有的 **attribution methods** 主要针对分类任务或单模态场景设计，难以直接应用于自回归式的多模态生成过程。

该论文指出三大挑战：
- **Generation-aware**：解释应基于解码过程中的每一步，而非固定目标输出。
- **Omni-modal**：需在统一的 token 时间线上追踪跨模态证据。
- **Semantically interpretable**：原始 token-level 信号噪声大且碎片化，难以形成语义连贯的解释。

### 提出了什么新方法或新思路
作者提出 **OmniTrace**，一个轻量级、模型无关（model-agnostic）的 generation-time attribution 框架。其核心思想是将 attribution 形式化为“**generation-time tracing problem**”，即在解码过程中动态追踪每个生成 token 的来源，并聚合为语义上有意义的 span-level 解释。

**关键机制**：
- **Token-level tracing**：利用 attention weights 或 gradients 等内部信号，在每步解码时计算各输入 token 对当前输出 token 的影响。
- **Span-level aggregation**：将连续 token 的溯源结果聚合成短语或句子级别的解释单元。
- **Confidence-weighted & temporally coherent curation**：通过 POS 加权、置信度加权、运行一致性过滤等方式筛选出最相关的支持源，提升解释的稳定性和可读性。

### 相比现有方法的优势
- ✅ **无需重训练或监督**：完全后处理式（post-hoc），适用于任意 decoder-only omni-modal LLM。
- ✅ **统一框架**：兼容多种底层信号（attention, gradient 等），支持文本、图像、音频、视频等多种模态。
- ✅ **在线解释能力**：可在生成过程中实时提供解释，适合交互式系统。
- ✅ **更强的语义一致性**：相比逐 token 自我归因或 embedding 相似性匹配，span-level 聚合显著提升了解释的稳定性与可理解性。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖视觉、音频、视频三大模态，共 **759 个样本**，涵盖多种任务类型：

| 模态 | 任务类型 | 数据集 | 示例数 | 备注 |
|------|--------|-------|------|------|
| **Visual** | QA | Mantis-eval | 200 | 多图推理 |
| **Visual** | Summarization | MMDialog, CliConSummation | 257 | 图文对话摘要 |
| **Audio** | QA / Summarization | MMAU, MISP | 155 | 会议录音问答与摘要 |
| **Video** | QA | Video-MME | 102 | 视频理解问答 |

### 实验设置和评估指标
- **模型**：Qwen2.5-Omni-7B 和 MiniCPM-o-4.5-9B。
- **硬件**：H200 GPU，确定性解码以保证 attribution 可复现。
- **解码策略**：默认 greedy decoding。

#### 评估指标
- **Visual Tasks**：**Span-level F1**，衡量文本 span 和图像 ID 匹配准确率。
- **Audio/Video Tasks**：**Time-F1**，将时间轴离散为 1 秒 bin，计算二分类 F1。
- **人工验证**：对 26.6% 测试集进行人工标注，LLM 作为 judge 的标签与人类标注的一致性达 **88.17%**，说明自动标注可靠。

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **Self-Attribution** | 使用相同模型回过头来判断自己用了哪些输入（prompt-based） |
| **Embedprocessor** | 使用模型自身 processor 编码生成 span 与源内容，按 cosine similarity 匹配 |
| **EmbedCLIP** | 使用 CLIP 模型提取图文嵌入进行相似性匹配 |
| **Random** | 随机分配源 |

> ⚠️ 注意：embedding-based 方法不适用于 audio/video 的连续时间戳任务（表中标记为 ×），因其依赖离散语义块。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）
在 **Qwen2.5-Omni** 上的表现尤为突出：

| 方法 | Visual Summ. (Text F1) | Visual Summ. (Image F1) | Audio QA (Time-F1) | Video QA (Time-F1) |
|------|------------------------|-------------------------|--------------------|---------------------|
| **OTAttMean** | **75.66** | **76.59** | **49.90** | **40.16** |
| OTRawAtt | 72.51 | 51.82 | 47.64 | 36.53 |
| Self-Attribution | 9.25 | 40.60 | 29.01 | 13.67 |
| Embedprocessor | 17.30 | 14.55 | × | × |
| Random | 10.98 | 8.38 | × | × |

> 在所有任务上，OmniTrace 显著优于所有基线，尤其在 image attribution 上领先超过 **30+ F1 points**。

在 **MiniCPM-o-4.5** 上趋势一致，尽管绝对性能较低，但仍远超 baseline。

### 与基线方法的对比结果
- OmniTrace 所有变体均大幅超越 self-attribution 和 embedding-based 方法。
- 即使是最弱的 OmniTrace 配置也优于最强的 baseline。
- embedding-based 方法表现差，尤其是在 image attribution 上接近随机水平（~5–15 F1），说明仅靠嵌入相似性不足以捕捉生成路径中的因果依赖。

### 消融实验结果（Table 3）
在 Qwen2.5-Omni + OTAttMean 设置下进行消融，发现以下组件至关重要：

| 消融配置 | Image F1 下降幅度 |
|--------|------------------|
| 完整模型（Full Model） | 76.59 |
| w/o POS Weighting | → 20.79 (**↓55.8**) |
| w/o Confidence Weight | → 19.88 (**↓56.7**) |
| w/o Run Coherence | → 19.88 (**↓56.7**) |
| w/o pmin Filtering | → 19.88 (**↓56.7**) |

> 🔴 **关键发现**：移除任一过滤机制都会导致 image attribution 性能崩溃至 ~20 F1，表明原始 token-level tracing 极不稳定，必须通过 confidence-aware 和 coherence-aware 的聚合才能获得可靠解释。

此外：
- **ASR 分割质量直接影响 audio attribution 效果**：高质量 ASR（如 Paraformer）带来更高 Time-F1（83.12 vs raw token 的 22.85）。
- **视频任务中视听融合更优**：同时使用 visual + audio 输入比单独任一模态效果更好（Qwen 达 35.80 vs visual-only 29.61）。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Generation-time tracing 是实现多模态归因的有效范式**：将 attribution 与 autoregressive 解码过程绑定，能更真实地反映模型的信息流动路径。
2. **Span-level aggregation 至关重要**：直接使用 token-level 信号会产生碎片化、不可靠的解释；通过语义 chunk 聚合并结合 confidence 和 coherence 过滤，可大幅提升解释质量。
3. **OmniTrace 具有强鲁棒性和通用性**：在不同模型（Qwen vs MiniCPM）、不同信号（attention vs gradient）、不同模态（text/image/audio/video）下均保持优越性能。
4. **Attribution 与 generation quality 不完全相关**：即使生成答案错误，模型仍可能正确关注到相关证据（见 Fig. 4），说明 attribution 反映的是“推理过程”而非“最终正确性”。

### 方法的局限性
- **依赖高质量的 source segmentation**：如 audio 需要良好的 ASR 输出语义 chunk，否则性能严重下降。
- **早期位置偏差（early-token bias）**：分析显示 attribution 更倾向于选择输入序列前半部分的内容（平均归因位置 $ \bar{p} = 0.44 $）。
- **对视觉 grounding 特别敏感**：视觉信号稀疏且易受干扰，需要复杂的过滤机制才能稳定输出。
- **Gradient-based 方法内存开销大**：在长音频/视频输入上难以应用（表中标记为 +）。

### 未来工作方向
- 探索更高效的 gradient-free attribution 信号，降低部署成本。
- 引入动态 segmentation 机制，减少对外部 ASR/VAD 模块的依赖。
- 将 OmniTrace 应用于模型调试、对抗攻击检测、可信 AI 决策支持等实际场景。
- 开放代码与测试集（计划 MIT License 发布），推动 omni-modal interpretability 社区发展。

---

> 📌 **总结一句话**：  
> **OmniTrace 成功将 generation-time tracing 与 span-level 解释相结合，首次实现了对 omni-modal LLMs 在开放生成任务中的稳定、跨模态、无需训练的 attribution，为构建可解释的多模态智能系统提供了实用基础。**

</details>

---

### 3. [Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](https://arxiv.org/abs/2604.13634)

**Authors**: Xuwen Zhou, Fangxin Liu, Chao Wang, Xiao Zheng, Hao Zheng, Min He, Li Jiang, Haibing Guan  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2604.13634v1  

#### Abstract
Speculative decoding accelerates autoregressive generation by letting draft tokens bypass full verification, but conventional frameworks suffer from frequent false rejections, particularly when draft models produce semantically correct but lexically divergent outputs. In this paper, we present Calib...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **Speculative Decoding (SD)** 虽然通过 draft model 加速了 LLM 推理，但其验证机制依赖严格的 **token-level exact matching**，导致大量语义正确但词法不同（如“x” vs “*”）的 draft token 被错误拒绝（false rejections），浪费计算资源并限制加速潜力。

### ✅ 提出的新方法：Calibrated Speculative Decoding (CSD)
提出一种无需训练的轻量级框架 **CSD**，基于“**Frequency-Guided Candidate Selection, Probability-Guarded Acceptance**”原则，恢复被误判的有效 draft token。

#### 核心模块：
- **Online Correction Memory (OCM)**  
  在线记录历史中高频出现的 draft-target token 对（如 `x` → `*`），作为未来拒绝时的“救援候选”。
- **Semantic Consistency Gating (SCG)**  
  不再要求精确匹配，而是通过目标模型输出的概率比值（logit ratio）判断候选 token 是否在合理置信范围内，实现动态语义一致性校验。

### ✅ 相比现有方法的优势
| 维度 | CSD | 传统 SD / 其他方法 |
|------|-----|------------------|
| **是否需要训练** | ❌ 否（training-free） | Judge Decoding、Fly 等需额外训练或调参 |
| **验证方式** | 动态语义门控（SCG）+ 频率引导 | 严格 token 匹配 或 固定阈值松弛 |
| **系统开销** | 极低（<0.02%） | Tree-based 方法有高 FLOPs 开销；Prompt-based 增加输入长度 |
| **适用性** | 可插拔于任何 rejection sampling 框架 | 多为专用架构 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **数学推理**：GSM8K、MATH500（Hendrycks et al., 2021）
- **代码生成**：HumanEval（Chen, 2021）
- **摘要任务**：CNN/DailyMail（Hermann et al., 2015）
- **指令跟随与长上下文**（附录）：IFEval（Zhou et al., 2023a）、RULER（Hsieh et al., 2024）

### ⚙️ 实验设置
- **模型对**：
  - Llama-3 系列：Llama-3-70B-Instruct（target） + Llama-3.2-1B-Instruct（draft）
  - Qwen-2.5 系列：Qwen-2.5-72B-Instruct + Qwen-2.5-7B-Instruct
- **评估工具**：`lm-evaluation-harness`（Gao et al., 2024）
- **硬件环境**：8×NVIDIA H20 GPU，单 batch 推理
- **解码策略**：主实验使用 greedy decoding（T=0）以消除随机性

### 🎯 评估指标
| 指标 | 含义 |
|------|------|
| **Acc** | 任务准确率（Pass@1 / ROUGE-L） |
| **Tp (Throughput)** | 输出速度（tokens/s） |
| **Spd (Speedup)** | 相对于 Vanilla Decoding 的加速比 |
| **AR (Acceptance Rate)** | draft token 被接受的比例 |

### 🔁 基线方法对比
| 类别 | 方法 |
|------|------|
| **基础 SD** | SpecDecode（Leviathan et al., 2023） |
| **Lossy 放松** | Static Lossy SD（T=0.6） |
| **先进加速** | Swift（Xia et al., 2024）、Lookahead Decoding（Fu et al., 2024） |
| **语义感知并发方法** | Fly（Li et al., 2025）、Reflective Verification（Wang et al., 2025） |

---

## 3. 主要实验结果和性能指标

### 📊 性能总览（Table 1）
| 方法 | 平均 Speedup (Spd) | 最高 Speedup | 准确率变化 |
|------|--------------------|-------------|------------|
| **Vanilla Decoding** | 1.00× | — | 基准 |
| **SpecDecode** | 1.75× (Llama) / 1.66× (Qwen) | 1.90× | 基本持平 |
| **CSD (Ours)** | **2.02× (Llama)** / **1.86× (Qwen)** | **2.33×**（MATH500 & HumanEval） | ✅ 提升 |

> 💡 **关键突破**：CSD 在保持甚至提升 accuracy 的前提下，实现高达 **2.33× 的吞吐加速**。

### 🔍 与语义基线对比（Table 2）
| 方法 | Avg. Speedup | Acceptance Rate (AR) | 优势分析 |
|------|--------------|------------------------|----------|
| **Fly** | 1.75× | 44.8% | 依赖窗口一致性，边界敏感 |
| **Reflective Verification** | 1.76× | 52.6% | 模板复制增加 latency |
| **CSD (Ours)** | **1.89×** | **50.2%** | 零开销过滤，直接利用 logits |

> ✅ CSD 在端到端延迟上优于 prompt-based 和 window-based 方法。

### 🔬 消融实验（Table 3）
| 变体 | HumanEval Acc | Speedup | 分析 |
|------|----------------|---------|------|
| **SpecDecode (Baseline)** | 76.8 | 1.00× | — |
| **SD w/ OCM only** | 70.7 | 1.24× | 仅频率引导 → 引入幻觉 |
| **SD w/ SCG only** | 70.7 | 1.48× | 仅概率门控 → 忽视模式可靠性 |
| **CSD (完整)** | **79.3** | **2.33×** | ✅ 协同作用：安全地解锁有效路径 |

> 🔍 发现：单独使用任一模块都会导致 accuracy 显著下降，证明 **双模块协同是关键**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **False Rejections 是普遍且可预测的**  
   - 20% 的高频 pattern 导致近 70% 的拒绝（图2a），说明可通过统计先验进行修复。
2. **语义有效性高度依赖上下文**  
   - 同一对 token 在不同 context 下的目标概率差异可达多个数量级（图2b），因此不能仅靠频率接受。
3. **CSD 实现高效而安全的加速**  
   - 利用 OCM 提出候选，SCG 进行上下文感知验证，既提高 acceptance rate，又避免引入错误。
4. **不仅加速，还能提效**  
   - 在 HumanEval 和 MATH500 上分别提升 **+2.5 和 +2.0 点 accuracy**，表明 draft model 可帮助跳出 greedy 局部最优。

### ⚠️ 方法的局限性
| 问题 | 描述 |
|------|------|
| **偏离分布精确性** | 不再保证输出分布与 target model 完全一致（非 rejection sampling），存在理论上的统计偏移风险。 |
| **依赖 draft model 质量** | 若 draft model 本身生成幻觉或低质量序列，则无法通过 SCG 验证，导致加速失效。 |
| **高并发扩展未验证** | 当前评估集中在小 batch 场景，OCM 的跨请求同步机制尚未在高吞吐系统中测试。 |
| **缺乏工业引擎集成** | 尚未接入 vLLM 等生产级推理系统，实际部署潜力有待验证。 |

### 🔮 未来工作方向
- 探索 **分布式 OCM 的同步机制**，支持高并发场景。
- 结合 **learned verifier** 与 CSD 的轻量结构，构建混合验证范式。
- 将 CSD 扩展至 **多模态生成**（如图像、音频）中的 speculative decoding。
- 推动与主流推理框架（如 vLLM、TensorRT-LLM）的集成优化。

---

## ✅ 总结
**CSD** 是一个简洁、高效、无需训练的 speculative decoding 增强方案。它通过 **频率引导 + 概率门控** 的双重机制，成功解决了传统 SD 中因词法差异导致的“假拒绝”问题，在多个 LLM 架构和任务上实现了 **最高达 2.33× 的吞吐加速**，同时 **维持甚至提升生成质量**。该方法为 LLM 高效推理提供了一条实用且可推广的新路径。

</details>

---

### 4. [FAST: A Synergistic Framework of Attention and State-space Models for Spatiotemporal Traffic Prediction](https://arxiv.org/abs/2604.13453)

**Authors**: Xinjin Li, Jinghan Cao, Mengyue Wang, Yue Wu, Longxiang Yan, Yeyang Zhou, Ziqi Sha, Yu Ma  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.13453v1  

#### Abstract
Traffic forecasting requires modeling complex temporal dynamics and long-range spatial dependencies over large sensor networks. Existing methods typically face a trade-off between expressiveness and efficiency: Transformer-based models capture global dependencies well but suffer from quadratic compl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FAST: A Synergistic Framework of Attention and State-space Models for Spatiotemporal Traffic Prediction

---

## 1. 论文的主要贡献和创新点

### 解决的问题
交通流量预测需要同时建模**复杂的时序动态**和**长距离空间依赖**，而现有方法在**表达能力**与**计算效率**之间存在权衡：
- **Transformer-based 模型**（如 iTransformer）能捕捉全局依赖，但具有 **O(N²)** 的自注意力复杂度，难以扩展到大规模传感器网络。
- **GNN-based 模型**（如 DCRNN、GWNET）利用图结构建模空间关系，但受限于局部消息传递，难以捕获远距离节点间的功能相关性。
- **Selective State-Space Models**（如 Mamba）具备线性时间复杂度，适合长序列建模，但在处理图结构数据的空间推理方面较弱。

### 提出的新方法与创新思路
作者提出 **FAST**（Framework of Attention and State-space for spatiotemporal Traffic prediction），其核心思想是：  
> **将时间建模与空间建模解耦，并为两者分配最适合的机制** —— 使用 **Attention 进行时间推理**，使用 **Mamba-based SSM 进行空间传播**。

#### 主要创新点包括：

- ✅ **Temporal-Spatial-Temporal (TST) 架构**  
  在每个 ST-Block 中采用 `Temporal → Spatial → Temporal` 的顺序设计：
  - 第一个 Temporal Attention 模块提取各节点的时间特征；
  - Mamba-based Spatial 模块沿传感器维度进行跨节点信息传播；
  - 第二个 Temporal Attention 模块在更新后的空间上下文中精炼时间动态。
  > 实现了时空建模的**交替增强**，而非简单堆叠。

- ✅ **Learnable Multi-source Spatiotemporal Embedding**  
  融合四类输入信息：
  - 历史流量 (`Xdata`)
  - 时间上下文（day-of-week 和 time-of-day）
  - 节点身份与位置联合编码 (`Epnode`)
  > 显式建模异质交通模式和周期性规律。

- ✅ **Hierarchical Skip-connected Prediction Mechanism**  
  将所有层级 TST block 的输出通过可学习投影加权融合，实现多尺度特征聚合，提升多步预测稳定性。

- ✅ **无需显式邻接矩阵的空间建模**  
  利用 Mamba 的 selective state-space propagation 学习动态空间依赖，避免对固定图结构的依赖。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **准确性** | 在多个指标上全面超越 GNN、Attention、Transformer 和 Mamba 类模型 |
| **效率** | 空间建模复杂度为 **线性 O(N)**，显著优于 Transformer 的二次复杂度 |
| **可扩展性** | 适用于大规模传感器网络部署 |
| **通用性** | 不依赖预定义图结构，适应动态拓扑变化 |

---

## 2. 核心实验方法和设置

### 数据集
在三个广泛使用的 **PeMS** 实际交通数据集上进行验证：
- **PeMS04**: 307 个传感器，采样频率 5 分钟
- **PeMS07**: 883 个传感器
- **PeMS08**: 170 个传感器  
> 所有数据来自加州高速公路系统的感应线圈检测器。

### 实验设置
- **任务**：标准多步预测任务
  - 输入历史窗口：12 步（即前 1 小时）
  - 预测未来：12 步（即后 1 小时）
- **划分比例**：训练 : 验证 : 测试 = 8 : 1 : 1
- **硬件平台**：NVIDIA RTX 3080 Ti GPU (12GB)
- **实现框架**：PyTorch 2.0.0

### 评估指标
- **MAE**（Mean Absolute Error）
- **RMSE**（Root Mean Squared Error）
- **MAPE**（Mean Absolute Percentage Error）

### 对比的基线方法（共 10 种）
| 类别 | 方法 |
|------|------|
| **Time Series** | iTransformer, PatchTST |
| **GNN-based** | STGCN, DCRNN, GWNET, STGNCDE |
| **Attention-based** | GMAN, ST-WA |
| **Mamba-based** | MCST-Mamba, TSMamba |

---

## 3. 主要实验结果和性能指标

### 总体性能（Table I）
FAST 在 **所有三个数据集** 上均取得最优或次优表现：

| 数据集 | 指标 | 最佳基线 | FAST 结果 | 提升幅度 |
|--------|-------|-----------|------------|----------|
| PeMS04 | MAE | 19.07 (ST-WA) | **19.00** | ↓0.07 |
|        | RMSE | 30.94 (FAST) | **30.94** | SOTA |
| PeMS07 | MAE | 20.67 (TSMamba) | **20.50** | ↓0.17 |
|        | RMSE | 33.70 (TSMamba) | **33.58** | ↓0.12 |
| PeMS08 | MAE | 15.08 (GWNET) | **14.66** | ↓0.42 (**↓2.8%**) |
|        | RMSE | 24.54 (MCST-Mamba) | **23.59** | ↓0.95 (**↓4.3%**) |

> 🔥 **在 PeMS08 上，FAST 将 RMSE 降低达 4.3%，MAE 降低 2.8%**，显著优于最强 baseline。

此外，在 9 个“数据集-指标”组合中：
- FAST 取得 **6 项第一**
- 取得 **2 项第二**
- MAPE 表现也保持竞争力

---

### 消融实验结果（Ablation Study）

#### （1）Multi-source Spatiotemporal Embedding 的作用（Table II）
移除该模块导致性能大幅下降：

| 数据集 | 指标 | w/o Embedding | FAST | 下降幅度 |
|--------|------|----------------|--------|----------|
| PeMS08 | MAE | 19.98 | **14.66** | ↑↑36.3% |
|        | RMSE | 30.76 | **23.59** | ↑↑30.4% |

> 表明融合历史流量、时间上下文和节点特性的嵌入策略至关重要。

#### （2）TST 架构设计的有效性（Fig. 3）
比较四种变体：
- **w/ Attention**：用 Attention 替代 Spatial Mamba → 计算成本飙升
- **w/ Mamba**：用 Mamba 替代 Temporal Attention → 时间建模能力下降
- **Swapped Mamba-Attention**：交换模块顺序 → 性能下降且效率降低

✅ **结论**：只有原生的 TST 设计（Attention for time, Mamba for space）实现了精度、效率与内存占用的最佳平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **时空建模应分工协作**：  
   时间维度适合使用 **Attention** 捕捉灵活的长期依赖；空间维度可通过 **Mamba-based SSM** 实现高效的信息传播。
   
2. **TST 架构有效促进交互增强**：  
   两阶段 Temporal 模块分别负责“感知原始时间模式”和“基于空间上下文重构时间动态”，形成闭环优化。

3. **统一嵌入 + 层级跳连显著提升鲁棒性**：  
   多源嵌入增强了模型对异构交通行为的理解；skip-connected prediction 改善了深层网络中的梯度流和多尺度融合。

4. **无需显式图结构也能建模复杂空间依赖**：  
   Mamba 的 selective scanning 机制能够自动学习传感器之间的功能性连接，摆脱对先验图的依赖。

---

### 方法的局限性
- 当前模型仍假设传感器位置固定，未考虑移动传感场景（如浮动车 GPS）。
- 虽然不依赖预定义图，但仍隐含地学习静态节点表示，对突发性拓扑变化（如道路封闭）响应有限。
- 实验集中在单变量流量预测，未验证在速度、占有率等多变量联合预测下的泛化能力。

---

### 未来工作方向
1. 扩展至 **multivariate traffic forecasting**（如同时预测流量、速度、密度）
2. 引入 **adaptive spatial structure learning**，动态调整传感器间依赖关系
3. 探索 **multi-task urban prediction**，结合天气、事件等外部信号
4. 应用于更复杂的城市系统，如 **ride-hailing demand prediction** 或 **urban air quality modeling**

---

## 总结
FAST 是一种新颖且高效的 **spatiotemporal forecasting framework**，通过将 **Attention 与时序建模绑定**、**Mamba 与空间传播结合**，构建了一个兼具高表达力与高可扩展性的 TST 架构。实验证明其在主流 PeMS 数据集上显著优于各类先进方法，尤其在 RMSE 和 MAE 上取得突破性进展，为智能交通系统中的实时预测提供了强有力的技术支持。

</details>

---

### 5. [SparseBalance: Load-Balanced Long Context Training with Dynamic Sparse Attention](https://arxiv.org/abs/2604.13847)

**Authors**: Hongtao Xu, Jianchao Tan, Yuxuan Hu, Pengju Lu, Hongyu Wang, Pingwei Sun, Yerui Sun, Yuchen Xie, Xunliang Cai, Mingzhen Li, Weile Jia  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.13847v1  

#### Abstract
While sparse attention mitigates the computational bottleneck of long-context LLM training, its distributed training process exhibits extreme heterogeneity in both \textit{1)} sequence length and \textit{2)} sparsity sensitivity, leading to a severe imbalance problem and sub-optimal model accuracy. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*SparseBalance: Load-Balanced Long Context Training with Dynamic Sparse Attention*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在长上下文大语言模型（LLM）训练中，**稀疏注意力（sparse attention）** 虽然缓解了标准注意力机制带来的二次计算复杂度瓶颈，但在分布式训练过程中仍面临严重的**负载不平衡（load imbalance）** 问题。该问题由两个维度的异构性共同导致：
- **序列长度异构性（sequence length heterogeneity）**：真实数据集中长短序列混合，导致不同微批次（micro-batch）处理时间差异巨大。
- **稀疏敏感性异构性（sparsity sensitivity heterogeneity）**：不同序列和Transformer层对稀疏程度的敏感度不同，固定稀疏度会损害模型精度或浪费算力。

现有方法通常只解决其中一个方面，缺乏系统性的协同优化，导致训练效率低下或模型精度下降。

### 提出了什么新方法或新思路
本文提出 **SparseBalance**，一种**算法-系统协同设计（algorithm-system co-design）框架**，通过动态调整稀疏度来联合优化训练效率与模型精度。其核心创新包括：

1. **Workload-Aware Dynamic Sparsity Tuning (DST)**  
   在运行时进行**双向稀疏度调整**：
   - 对“瓶颈”微批次（straggler）**减少注意力预算（increased sparsity）**，以加速执行；
   - 对非瓶颈微批次**增加注意力预算（reduced sparsity）**，利用流水线中的空闲时间（bubble）提升模型精度，实现“免费精度增益”。

   引入 **anchor-guided thresholding 机制**，基于预测延迟确定调优方向，并通过路由logits限制调优幅度，确保模型质量不显著下降。

2. **Sparsity-Aware Batching (SAB)**  
   一种面向稀疏训练的批处理策略：
   - 利用轻量级稀疏度估计器和**基于延迟的打包策略**，实现粗粒度的负载均衡；
   - 为后续的DST提供更平衡的初始状态，形成“粗调+细调”的协同优化流程。

3. **Profiling-Based Latency Prediction Module**  
   构建一个离线性能分析驱动的延迟预测模型，将**序列长度 + 稀疏度 → 实际执行延迟**映射，为DST和SAB提供精准的性能指导。

### 相比现有方法的优势
- **系统性协同优化**：首次同时考虑序列长度与稀疏敏感性的双重异构性，实现算法与系统的深度耦合。
- **无需牺牲精度换取效率**：通过“压缩慢者、增强快者”的双向调优，在提升效率的同时维持甚至提升模型能力。
- **即插即用兼容性**：复用现有稀疏注意力算法（如MoBA、DSA）的indexer，可无缝集成到主流训练框架中。
- **低开销高收益**：引入的额外计算开销极小（<1.5%），却带来显著端到端加速。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ChatQA2-Long-SFT**：开源长上下文SFT数据集，呈现**双峰分布**（大量<4K和>16K序列）。
- **LongAlign-10k**：另一个真实长文本数据集，具有**长尾分布**（从8K到72K不等）。

### 实验设置
- **硬件环境**：
  - 两组集群：均配备8×H200或H20 GPU（141GB HBM3e），节点内NVLink互联，跨节点InfiniBand。
- **软件栈**：
  - CUDA 12.4, PyTorch 2.5.1, NCCL 2.21.5
  - 基于 **Megatron-LM** 和 **ms-swift** 实现
- **模型**：
  - Qwen2.5-0.5B 和 Qwen2.5-3B
  - 使用 **MoBA** 作为基础稀疏注意力算法（block size=256, top-k=32）
- **并行配置**：
  - 默认采用 **4D Hybrid Parallelism**：DP=4, PP=4, TP=2, SP=2
  - 微批次大小（micro-batch size）=1，全局批次大小=16
- **超参数**：
  - 学习率：1e-6（warm-up 20%）
  - LoRA配置：rank=32, alpha=64, dropout=0.05
  - DST锚点策略：MEAN / MIN / MAX；阈值 $ p \in \{0.1, 0.2\} $

### 评估指标
- **系统效率**：
  - 端到端训练速度（speedup ×）
  - 迭代时间（iteration time）
  - 负载不平衡度（Imbalance = max / mean micro-batch latency）
- **模型精度**：
  - 训练损失曲线收敛性
  - 下游任务表现：
    - **LongBench**：综合评估长上下文理解能力
    - **Needle-in-a-Haystack (NIAH)**：测试精确检索能力
    - **通用零样本推理基准**（ARC, BoolQ, HellaSwag等）

### 基线方法对比
- **Baseline**：原始MoBA + 固定稀疏度 + 长度感知批处理（Length-Based Batching, LBB）
- **消融变体**：
  - +DST：仅启用动态稀疏调优
  - +SAB：仅启用稀疏感知批处理
  - +SAB+DST：完整SparseBalance方案

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型配置 | 数据集 | Speedup (×) |
|---------|--------|-----------|
| Qwen2.5-0.5B | ChatQA2 | 1.30× |
| Qwen2.5-3B | ChatQA2 | 1.33× |
| Qwen2.5-3B LoRA | ChatQA2 | 1.31× |
| Qwen2.5-3B | LongAlign-10k | **1.35×** |

> ✅ **最高实现 1.35× 的端到端训练加速**

### 与基线方法的对比结果
- **效率方面**：
  - 相比Baseline，SparseBalance平均降低迭代时间 **~25–35%**。
  - 在LongAlign-10k上，DST贡献更大（因序列长度较集中，批处理优化空间小）。
  - 在ChatQA2上，SAB作用更明显（因其极端双峰分布利于重组织）。
- **精度方面**：
  - **LongBench 总体得分提升**：MEAN0.1 配置下达到 **39.46%**，优于基线的 **39.28%**（↑+0.18%）。
  - **NIAH 精确检索能力几乎无损**：在64K上下文中，MEAN0.1 达到 **97.53%**，接近基线的 **97.60%**。
  - **通用推理能力持平或略优**：平均zero-shot准确率 **0.6967** vs 基线 **0.6958**。

### 消融实验结果
| 配置 | Imbalance ↓ | Iter. Time (ms) | Speedup × | Overhead (ms) |
|------|------------|------------------|-----------|----------------|
| Baseline | 1.81× | 4202.09 | 1.00× | 0.00 |
| +DST | 1.73× | 3901.57 | 1.08× | 37.14 |
| +SAB | 1.41× | 3470.73 | 1.21× | 1.81 |
| +SAB+DST | **1.34×** | **3107.83** | **1.35×** | 44.23 |

- **SAB单独使用优于LBB**：尽管+LBB也达1.23×，但+SAB在结合DST后表现更优，体现其为DST提供了更好的初始条件。
- **DST是性能跃升的关键**：尤其在高复杂度模型（如3B）中效果更显著。
- **总开销可控**：<1.5% 的迭代时间用于调度与预测。

---

## 4. 关键结论和发现

### 主要发现
1. **稀疏度可以作为运行时调控自由度**：将原本固定的稀疏参数变为可动态调节的资源，用于系统级负载均衡，是一种全新的优化视角。
2. **“双向调优”能实现效率与精度双赢**：压缩慢任务释放关键路径压力，同时增强快任务利用空闲周期提升精度，打破了传统“效率 vs 精度”权衡。
3. **粗粒度批处理 + 细粒度运行时调优 是最优路径**：SAB为DST奠定良好基础，二者协同产生最大收益。
4. **理论FLOPs无法准确反映稀疏注意力实际延迟**：必须依赖实测性能建模（profiling-based）才能做出有效决策。

### 方法的局限性
- **依赖稀疏注意力算法提供indexer**：虽然兼容性强，但仍需底层支持可解释的token重要性评分（如routing logits）。
- **延迟预测模块需离线校准**：当硬件、并行策略或稀疏核变更时，需重新构建查找表。
- **极端稀疏场景可能影响稳定性**：若 $ p $ 设置过大（如 >0.3），可能导致某些敏感序列信息严重丢失。

### 未来工作方向
- 将动态稀疏思想扩展至**推理阶段**，实现训练-推理一致性优化。
- 探索**全自动的 $ p $ 和 anchor 策略选择机制**，适应不同数据与模型动态调整。
- 结合**context parallelism** 或 **sequence parallelism** 进一步优化通信开销。
- 研究如何将该框架应用于**多模态长序列建模**场景。

---

> 📌 **总结一句话**：  
> **SparseBalance 通过算法-系统协同设计，将稀疏注意力的“稀疏度”转化为可调度资源，在不牺牲甚至提升模型精度的前提下，实现了高达 1.35× 的端到端训练加速，为长上下文LLM高效训练提供了新范式。**

</details>

---

### 6. [Automated co-design of high-performance thermodynamic cycles via graph-based hierarchical reinforcement learning](https://arxiv.org/abs/2604.13133)

**Authors**: Wenqing Li, Xu Feng, Peixue Jiang, Yinhai Zhu  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.13133v1  

#### Abstract
Thermodynamic cycles are pivotal in determining the efficacy of energy conversion systems. Traditional design methodologies, which rely on expert knowledge or exhaustive enumeration, are inefficient and lack scalability, thereby constraining the discovery of high-performance cycles. In this study, w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Automated co-design of high-performance thermodynamic cycles via graph-based hierarchical reinforcement learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统热力学循环设计严重依赖专家经验，采用“固定结构 + 参数优化”的范式，导致设计空间受限、效率低下且难以发现新颖高性能结构。现有自动化方法（如HEN、superstructure、graph theory）虽能部分摆脱人工干预，但仍面临**计算效率低、优化能力弱、缺乏自主学习能力**等问题。

### 提出了什么新方法或新思路
本文提出了一种**基于图的分层强化学习（graph-based hierarchical reinforcement learning, HRL）框架**，用于热力学循环的**结构-参数协同设计（co-design）**。其核心创新包括：

- **图编码（Graph-based Encoding）**：将热力学循环抽象为有向图（directed graph），组件为节点（node），连接管道为边（edge），并引入语法约束（grammatical constraints）确保拓扑合法性。
- **物理信息图解码（Physics-informed Graph Decoding）**：结合深度学习代理模型（deep learning thermophysical surrogate）实现图结构到系统状态的稳定求解，避免传统逐点迭代法的数值不稳定性。
- **Manager-Worker 分层架构**：
  - **Manager（高层）**：负责探索结构演化，通过离散动作添加边以构建新拓扑；
  - **Worker（底层）**：对给定结构进行连续参数优化，并反馈性能奖励（如COP或效率），引导搜索方向。
- **端到端自动化流程**：集成图表示、物理代理模型与HRL，形成从编码、解码到协同优化的全自动pipeline。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 设计模式 | 固定结构 + 参数调优 | 结构与参数联合优化（co-design） |
| 探索能力 | 受限于专家假设 | 自主发现新结构 |
| 计算效率 | 枚举或MINLP求解耗时 | 图HRL加速搜索，减少无效尝试 |
| 泛化性 | 特定场景定制 | 可扩展至多种热力系统（heat pump, heat engine等） |
| 智能水平 | 规则驱动 | 数据驱动 + 强化学习自适应 |

---

## 2. 核心实验方法和设置

### 使用的案例系统（数据集）
本研究未使用传统意义上的“数据集”，而是基于两类典型热力系统进行验证：
- **Heat Pump Cycle**：空气源跨临界CO₂热泵系统
- **Heat Engine Cycle**：超临界CO₂ Brayton循环

所有可能的结构在符合物理规则的前提下由算法自动生成。

### 实验设置
- **图规则定义**：设定组件数量上限（如压缩机、膨胀阀各最多一个）、连接有效性检查（connection, pressure, energy, parallelism, heat transfer validity）。
- **状态空间**：邻接矩阵 $ A \in \{0,1\}^{N\times N} $ 表示图结构。
- **动作空间**：Manager每次激活一条边 $ a_{i,j}=1 $。
- **奖励函数**：Worker返回最终性能指标（COP 或 thermal efficiency）作为稀疏奖励。
- **训练过程**：
  - 总训练轮次：5000 episodes
  - 使用 **Proximal Policy Optimization (PPO)** 算法训练Manager
  - Worker采用 **Bayesian Optimization** 进行外层参数优化，内层用MLP surrogate求解状态点

### 评估指标
- **有效循环生成率（valid cycle generation probability）**
- **最优性能指标提升**：
  - Heat Pump：最大COP提升百分比
  - Heat Engine：最高thermal efficiency及相对提升
- **新结构发现数量**

### 基线方法对比
- **Random Search**：随机生成图结构并优化参数
- **Expert-designed / Classical Cycles**：已有文献中的经典配置作为基准（baseline）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 热泵系统（Heat Pump）
- 在5000次episode中，HRL agent生成 **4,675个有效循环**，有效率为 **93.50%**
- 共发现 **22个有效结构**，其中 **18个为全新结构**
- 最佳新结构（Cycle No. 3, 5, 6, 7）在Case 1下达到 **COP = 5.256**，相比基础循环（Cycle 1）**提升4.6%**
- Cycle 5在不同工况下均表现最优，显示强鲁棒性

#### ✅ 热机系统（Heat Engine）
- HRL agent生成 **4,141个有效循环**，有效率达 **82.82%**
- 发现 **26个有效结构**，其中 **21个为新结构**
- 最佳结构（Cycle 13）在Case 1下实现 **thermal efficiency = 0.413**，相比基础循环（Cycle 1）**提升133.3%**

> 🔍 注：部分结构因组件冗余而在优化后退化为简单再生循环，但仍被识别为潜在高价值拓扑。

### 与基线方法的对比结果
| 指标 | HRL Agent | Random Search |
|------|----------|---------------|
| 有效循环生成概率（Heat Pump） | 93.50% | ~0.02% |
| 有效循环生成概率（Heat Engine） | 82.82% | <0.06% |
| 新结构发现数（Heat Pump） | 18 | 无 |
| 新结构发现数（Heat Engine） | 21 | 无 |
| 性能提升（vs. baseline） | +4.6% (HP), +133.3% (HE) | 无法超越经典设计 |

> ❗ Random search几乎无法生成合法结构，凸显HRL在复杂约束下的强大探索能力。

### 消融实验结果（Ablation Study）
文中虽未明确列出消融表格，但通过以下策略验证关键模块作用：
- **Performance Feedback Backpropagation**：将最终性能反向传播至每一步决策，显著改善信用分配（credit assignment）
- **Elite Cycle Memory**：保留高性能轨迹，防止遗忘优质结构
- **Staged Training**：前期鼓励探索（高熵权值），后期聚焦收敛（加强精英记忆）

这些机制共同提升了训练稳定性与全局搜索能力。

---

## 4. 关键结论和发现

### 主要发现
1. **HRL可有效实现热力学循环的自主设计**：无需先验知识即可复现经典结构，并发现大量前所未见的高性能拓扑。
2. **结构敏感性差异明显**：
   - 跨临界CO₂热泵对结构变化高度敏感，微小改动即可带来显著性能增益；
   - 热机系统性能排序更稳定，反映能量转换机制更强健。
3. **内部能量回收机制普遍存在**：多数高性能结构包含IHX（Internal Heat Exchanger），体现算法自发遵循热力学第二定律的设计直觉。
4. **框架具备良好泛化能力**：同一方法适用于热泵与热机系统，表明其可推广至其他复杂热力系统。

### 方法的局限性
1. **图规则仍需人工定义**：当前的连接规则、有效性检查依赖领域知识，尚未完全实现“无监督规则提取”。
2. **单目标优化限制**：仅以COP或效率为目标，未考虑设备成本、系统复杂度、经济性等工程实际因素。
3. **应用范围有限**：目前仅验证于单一循环系统，尚未拓展至联合循环（combined cycle）或多能耦合系统（integrated energy systems）。

### 未来工作方向
1. **通用图规则学习**：探索让agent自动归纳合法连接模式，降低对专家知识的依赖。
2. **多目标优化扩展**：引入Pareto前沿优化，平衡性能、成本与可靠性。
3. **复杂系统延伸**：应用于ORC、吸收式循环、储能集成系统等更复杂的能源系统设计。
4. **实验验证与数字孪生对接**：推动所发现结构进入原型测试阶段，建立“AI设计-仿真-实验”闭环。

---

> 💡 **总体评价**：该研究开创性地将**图神经网络 + 分层强化学习 + 物理代理模型**融合，实现了热力学系统设计从“人主导”向“机器自主创造”的跃迁，是智能能源系统设计的重要里程碑。

</details>

---

### 7. [Enhancing Clustering: An Explainable Approach via Filtered Patterns](https://arxiv.org/abs/2604.12460)

**Authors**: Motaz Ben Hassine (CRIL), Sa\"id Jabbour (CRIL)  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.12460v1  

#### Abstract
Machine learning has become a central research area, with increasing attention devoted to explainable clustering, also known as conceptual clustering, which is a knowledge-driven unsupervised learning paradigm that partitions data into $\theta$ disjoint clusters, where each cluster is described by a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Enhancing Clustering: An Explainable Approach via Filtered Patterns*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **explainable clustering**（可解释聚类）中的一个关键瓶颈：在基于 **k-Relaxed Frequent Patterns (k-RFPs)** 的概念聚类框架中，存在大量 **冗余的候选模式**（redundant patterns）。这些不同的 k-RFPs 可能诱导出相同的 **k-cover**，导致：
- 候选模式集合膨胀
- ILP 模型变量增多
- 求解时间显著增加
- 聚类效率下降

这种冗余不仅浪费计算资源，还可能影响最终聚类的质量和可解释性。

---

### 🆕 提出的新方法与思路
作者提出了 **Optimized Conceptual Clustering Method (OCCM)**，其核心是引入一种 **模式过滤策略**（pattern filtering strategy），以消除冗余的 k-RFPs。

#### 主要创新点包括：
1. **理论分析冗余成因**  
   形式化地定义了当两个不同的 k-RFPs 会产生相同 k-cover 的条件（见 *Proposition 2* 和 *Corollary 1*），为去重提供了理论基础。

2. **提出 Filtered Patterns 概念**  
   定义 **filtered patterns** 集合 $ \mathcal{A}_f $：对于每一个唯一的 k-cover，仅保留一个代表性的 k-RFP。选择标准为 **最大项集**（largest itemset），因其提供更丰富、更具描述性的聚类解释。

3. **设计高效的过滤算法**  
   提出 *Algorithm 1*：先按大小排序所有 k-RFPs，然后逐个计算其 k-cover，并通过哈希映射 `CoverMap` 维护每个 cover 对应的最大模式，自动覆盖较小者。

4. **引入可解释性评估度量**  
   提出两个新指标来评估所选 pattern 的代表性与稳定性：
   - **Shapley Value Variance (SVV)**：衡量 pattern 内部各 item 贡献的不均衡程度。
   - **Average Cluster Stability (ACS)**：衡量移除单个 item 后 cluster 的 Jaccard 相似性变化，反映鲁棒性。

---

### ⚖️ 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **效率提升** | 显著减少 ILP 输入的 pattern 数量，降低求解复杂度，加快运行速度 |
| **质量保持甚至提升** | 在多数数据集上聚类质量（F1-score）不变，在部分数据集（如 Mushroom）反而更高 |
| **增强可解释性** | 优先保留更大的 pattern，提升了 cluster 描述的信息量和表达能力 |
| **理论支撑强** | 不仅是经验性优化，而是建立在对 k-cover 结构的深入理解之上 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验在多个真实世界 transactional 数据集上进行，具体如下（来自 Table 3）：

| Dataset | #Transactions | #Items | Density (%) |
|--------|----------------|---------|-------------|
| Lymph | 148 | 68 | 40 |
| Mushroom | 8,124 | 119 | 18 |
| Primary-Tumor | 336 | 31 | 48 |
| Soybean | 630 | 50 | 32 |
| Tic-tac-toe | 958 | 27 | 33 |
| Vote | 435 | 48 | 33 |

> 所有数据集均为二值化事务型数据，适合概念聚类建模。

---

### ⚙️ 实验设置
- **k 参数固定为 1**：即允许最多缺失 1 个 item 即视为被覆盖（k-cover）
- **聚类数量 $ \theta = 2 $**：所有数据集的真实标签均为两类，任务等价于二分类
- **最小支持度 $ \sigma $**：从 10% 到 40% 多次测试（Phase I），最优值用于后续阶段
- **最大运行时间限制**：1 小时（超时则视为无解）

---

### 📊 评估指标
| 指标 | 用途 |
|------|------|
| **Pattern count reduction rate (Δ%)** | 衡量过滤前后候选 pattern 数量减少比例 |
| **ILP CPU time** | 评估求解效率 |
| **F1-score** | 与 ground-truth 比较，评估聚类质量（优于原作使用的 ICS） |
| **SVV** | 分析 pattern 内 item 贡献分布是否均衡 |
| **ACS** | 分析 cluster 在 item 删除下的稳定性 |
| **Pattern size** | 观察所选 pattern 的规模及其与 ACS 的关系 |

---

### 🔁 基线方法对比
- **Baseline**: CCA-k-RFP-M1 方法（Hassine et al., 2024）——未经过滤的原始 k-RFP + ILP 流程
- **Proposed**: OCCM —— 加入 filtered patterns 步骤后的改进流程

---

## 3. 主要实验结果和性能指标

### 📉 Phase I: 模式数量减少效果（表 4）
- 所有数据集均检测到冗余 pattern，验证了理论预测。
- 平均减少幅度达 **5–10%**，最高达 **26.67%**（Tic-tac-toe, σ=40%）
- 即使在高支持度下仍存在冗余，表明该问题是普遍存在的

> 示例：
> - **Tic-tac-toe @40%**: 从 15 → 11 个 pattern（↓26.67%）
> - **Mushroom @40%**: 1135 → 1008（↓11.19%）
> - **Lymph @10%**: 3.6M → 3.3M（↓7.09%）

✅ 表明过滤有效且必要。

---

### ⏱️ Phase II: ILP 求解时间与聚类质量（表 5）

| Dataset | Baseline Time (s) | OCCM Time (s) | Speed-up | F1-score (Both) |
|--------|--------------------|----------------|-----------|------------------|
| Lymph | 49.66 | **29.74** | ~1.67x | 0.71 |
| Mushroom | 3133.34 | **1144.46** | ~2.74x | ↑0.34 → **0.73** |
| Primary-Tumor | 55.88 | **55.71** | ≈same | ↑0.25 → **0.33** |
| Soybean | 18.29 | **16.99** | ~1.08x | 0.29 |
| Vote | 1206.17 | **234.30** | ~5.15x | 0.51 |

> 注：Tic-tac-toe 超时未得解，两方法均失败。

#### 关键发现：
- **求解时间大幅缩短**：平均提速数倍，Vote 上提速超过 5 倍
- **聚类质量未降反升**：在 Mushroom 和 Primary-Tumor 上 F1-score 显著提高
- **原因分析**：过滤后 ILP 更容易找到高质量解；且保留更大 pattern 提升了描述能力

---

### 🔍 Phase III: 可解释性分析（表 6 & 图 3–4）

#### 发现 1：**Pattern Size 与 ACS 强正相关**
- 更大的 pattern 导致更高的 cluster 稳定性（图 4）
- 如 Soybean 中 pattern 大小从 3→9，ACS 从 0.70→0.97
- 支持“保留最大 itemset”策略的合理性

#### 发现 2：**SVV 与 ACS 多数呈负相关**
- 低 SVV（item 贡献均衡）→ 高 ACS（cluster 稳定）
- 例外：Primary-Tumor 出现正相关，但伴随 pattern size 增大 → 推测 size 是主导因素

#### 发现 3：**大 pattern 更具代表性**
- 大 pattern 不仅稳定，还能更好地捕捉 cluster 特征
- 支持 OCCM 的设计选择：**在相同 cover 下优先选最大 pattern**

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **冗余普遍存在**：不同 k-RFPs 可共享相同 k-cover，造成不必要的搜索开销。
2. **过滤显著提效**：通过保留每 cover 一个最大 pattern，可减少最多 26.67% 的候选集，大幅提升 ILP 求解速度。
3. **聚类质量得以保持甚至提升**：尤其在 Mushroom 等复杂数据集上，F1-score 明显改善。
4. **大 pattern 更优**：实验证明 larger patterns 具有更高的 ACS 和更好的 representativeness，支持 OCCM 的选择策略。
5. **可解释性可量化**：SVV 与 ACS 提供了评估 pattern 解释力的新工具。

---

### ⚠️ 局限性
1. **当前为后处理机制**：过滤步骤独立于 SAT 生成过程，属于 post-processing，仍有优化空间。
2. **SAT 编码复杂性限制在线去重**：尝试将去重约束直接编码进 SAT 导致求解器扩展性差。
3. **仅考虑 k=1 和 θ=2 场景**：泛化到多类或多松弛参数需进一步研究。
4. **目标函数依赖 pattern size**：最大化 size 不一定完全对应最优解释性，缺乏更精细的 interpretability-aware objective。

---

### 🔮 未来工作方向
1. **将 redundancy-awareness 集成进 SAT 生成流程**：开发更紧凑的逻辑约束，在生成阶段避免冗余。
2. **设计新的 ILP 目标函数**：融合 SVV、ACS 等指标，构建 **interpretability-aware optimization**。
3. **扩展至多类别聚类**（θ > 2）和更高 k 值场景。
4. **探索 pattern compression 或 summarization 技术**：进一步精简输出结果，提升人类可读性。

---

## 总结
本文提出的 **OCCM** 框架通过识别并去除诱导相同 k-cover 的冗余 k-RFPs，实现了 **高效、高质量、高可解释性** 的概念聚类。其实验充分验证了：
- 冗余确实存在且有害；
- 过滤策略能显著加速 ILP 求解；
- 保留最大 pattern 不仅合理，而且有助于提升聚类表现；
- 新提出的 SVV 和 ACS 为评估 pattern 质量提供了有力工具。

该工作推动了 **explainable AI** 与 **knowledge-driven clustering** 的融合发展，具有重要的理论价值与应用前景。

</details>

---

### 8. [Hardware-Efficient Neuro-Symbolic Networks with the Exp-Minus-Log Operator](https://arxiv.org/abs/2604.13871)

**Authors**: Eymen Ipek  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.13871v1  

#### Abstract
Deep neural networks (DNNs) deliver state-of-the-art accuracy on regression and classification tasks, yet two structural deficits persistently obstruct their deployment in safety-critical, resource-constrained settings: (i) opacity of the learned function, which precludes formal verification, and (i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Hardware-Efficient Neuro-Symbolic Networks with the Exp-Minus-Log Operator*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对当前深度神经网络（DNNs）在**安全关键、资源受限边缘场景**下的两大结构性缺陷提出解决方案：
1. **模型不可解释性**：传统 DNN 是“黑箱”，难以进行形式化验证（formal verification），阻碍其在医疗、金融、汽车等高风险领域的应用。
2. **硬件部署低效**：DNN 依赖多种异构的激活函数（如 `tanh`, `ReLU`, `GELU`），在 FPGA 或微控制器上需要多个查表（LUT）实现，导致硅面积大、延迟高。

### 提出的新方法与新思路
作者提出一种新型混合架构——**Hybrid DNN-EML 模型**，将传统的 MLP 主干（trunk）与基于 **Exp-Minus-Log (EML)** 算子的符号头（symbolic head）结合：

- **EML 算子定义**：  
  $$
  \text{eml}(x, y) = \exp(x) - \ln(y)
  $$
  该算子由 Odrzywolek (2026) 证明为**连续域上的 Sheffer 算子**，即仅用此单一操作符加常数 1 即可构造所有初等函数（elementary functions），包括加减乘除、指数对数、三角函数等。

- **模型结构设计**：
  - **Stage 1（Trunk）**：标准多层感知机（MLP）提取特征 $ z = \text{MLP}_\theta(x) $
  - **Stage 2（Head）**：一个固定深度 $ D $ 的二叉树结构，每个节点均为 EML 操作，叶子节点为输入变量或常量的仿射组合。最终输出通过实部提取得到。

- **符号化机制（Symbolic Snapping）**：
  利用 Gumbel-Softmax 技巧对叶节点权重进行稀疏化训练，逐步将参数压向单纯形顶点，最终硬截断为 {0,1}，从而获得可读的闭式表达式。

### 相比现有方法的优势
| 方法 | 局限性 | DNN-EML 的优势 |
|------|--------|----------------|
| **MLP / PINN** | 黑箱模型，无解释性；多激活函数增加硬件开销 | 可生成人类可读公式；统一硬件单元 |
| **EQL [9]** | 手工选择算子集，不完备且有奇点（如除零） | 单一完备算子，理论上能表示一切初等函数 |
| **KAN [6,7]** | 边上学习样条，虽可视但仍依赖预设符号库 | 符号表达自动从训练中涌现，非人工指定 |
| **AI-Feynman [8]** | 非端到端可微架构，需后处理搜索 | 全网络可微分，支持梯度优化 |

> ✅ **核心创新**：首次将 **Sheffer 性质的单一连续算子** 引入神经网络设计，在保持表达能力的同时实现了**硬件统一性**与**符号可解释性**的双重提升。

---

## 2. 核心实验方法和设置

尽管本文偏理论分析与架构提案，尚未完成完整实验验证，但文中明确提出了以下实验路径和评估框架：

### 数据集（建议使用）
- **Feynman Symbolic Regression Database** [8]：用于测试模型是否能恢复物理定律的闭式表达。
- **PMSM 和 Battery Surrogate Models** [5]：真实电动汽车动力系统数字孪生场景，符合 ISO 26262 功能安全要求。

### 实验设置
- **模型配置**：
  - Trunk：L 层 MLP，使用 ReLU/GELU 激活
  - Head：EML 树深度 $ D \in \{2,3,4\} $，因经验表明 $ D > 5 $ 极难训练成功
- **训练策略**：
  - 使用 Adam 优化器
  - 应用“硬化调度”（hardening schedule）推动 leaf weights 向 simplex 顶点收敛
  - 对 $\exp$ 和 $\ln$ 进行 clamping 处理以缓解数值不稳定性
- **硬件平台模拟**：
  - 在 Xilinx Zynq UltraScale+ FPGA 上合成 EML cell
  - 对比 tanh-MLP 基线的 LUT 占用与推理延迟

### 评估指标
| 类别 | 指标 |
|------|------|
| **性能** | 推理延迟（latency）、FLOPs、MAC 数量 |
| **效率** | FPGA 资源占用（LUT, DSP）、硅面积估算 |
| **可解释性** | 成功 symbolic snapping 的比例、生成公式的简洁性 |
| **可验证性** | 是否适用于 autoLiRPA、Lean 等形式化工具 |
| **准确性** | RMSE、extrapolation 泛化能力 |

### 基线方法对比
- **MLP**：标准前馈网络
- **PINN** [4]：嵌入 PDE 残差的物理信息网络
- **EQL** [9]：方程学习网络
- **KAN 2.0** [7]：Kolmogorov-Arnold Network
- **AI-Feynman** [8]：符号回归方法

---

## 3. 主要实验结果和性能指标

> ⚠️ 注：本文为研究计划性质，多数结果基于已有文献 [1] 的实证及复杂度推导，而非本工作独立运行的完整实验。

### 关键性能数据（来自理论分析与引用）
| 指标 | 数值/结论 |
|------|----------|
| **单次 EML 计算成本（CPU）** | ~111 FLOPs（$\exp$:50, $\ln$:60, 减法:1） |
| **常见激活函数成本** | $ C_{\text{ReLU}}=1 $, $ C_{\text{tanh}}\sim20 $, $ C_{\text{GELU}}\sim30 $ |
| **EML 树最大有效深度** | $ D \leq 4 $（$ D>5 $ 成功率 <1%） |
| **FPGA 推理延迟趋势** | $ t_{\text{inf,FPGA}} \propto D \cdot T_{\text{eml}} $，其中 $ T_{\text{eml}} \approx T_{\text{MAC}} $（定制单元下） |

### 与基线方法的对比结果（定性总结，见 Table 1）
| Architecture | Interpret. | CPU/GPU inf. | FPGA inf. | Train. | Verif. |
|-------------|------------|--------------|-----------|--------|--------|
| MLP         | ★           | fast         | moderate  | fast   | hard   |
| PINN        | ★           | moderate     | moderate  | slow   | hard   |
| EQL         | ★★★         | moderate     | moderate  | moderate | medium |
| KAN 2.0     | ★★★         | moderate     | moderate  | moderate | medium |
| **DNN-EML (Ours)** | **★★★★**    | **slow**     | **fast**  | **moderate** | **tractable** |

> 🔍 解读：
> - **解释性最强**（四星）：唯一能通过 snapping 得到真正闭式表达的方法
> - **FPGA 推理最快**：得益于统一 EML cell 的流水线设计
> - **CPU/GPU 推理最慢**：因 EML 本身计算昂贵
> - **可验证性最佳**：单一算子简化形式化推理

### 消融实验（隐含分析）
- **Depth ablation**：随着 $ D $ 增加，训练成功率急剧下降（100% @ D=2 → <1% @ D=5），说明必须限制 head 深度
- **Hardware ablation**：在通用硬件上无加速优势；只有在专用 EML cell（FPGA/analog）上才体现延迟降低一个数量级的潜力

---

## 4. 关键结论和发现

### 主要发现
1. **DNN-EML 实现了三重平衡**：
   - ✅ **高表达力**：继承 MLP 特征提取能力 + EML 完备性
   - ✅ **强可解释性**：可通过 symbolic snapping 获得闭式公式
   - ✅ **硬件友好性**：全网络仅需一种 EML cell，极大减少 FPGA LUT 开销

2. **加速效果高度依赖硬件平台**：
   - ❌ 在 CPU/GPU 上：EML 计算太贵，**无法加速推理或训练**
   - ✅ 在定制硬件（FPGA/analog）上：EML cell 可实现 $ O(D) $ 流水线延迟，**推理速度可达传统 MLP 的 10 倍以上**

3. **训练更具挑战性**：
   - 存在 **numerical fragility**：$\exp$ 易溢出，$\ln$ 在负值时产生 NaN
   - 深层树难以收敛，需 clamping 和 careful initialization
   - 因此采用“浅层 EML head + 深层 MLP trunk”是实用折衷

4. **填补领域空白**：
   - 首次将 **Sheffer operator 思想引入神经网络设计**
   - 统一了 neuro-symbolic computing 中的**表示统一性**与**硬件实现一致性**

### 方法的局限性
| 局限 | 描述 |
|------|------|
| **复数域运算** | EML 内部需处理复数（如生成 $ i = \sqrt{-1} $ 需 $ \ln(-1) $），导致内存翻倍、AD 复杂化 |
| **数值不稳定** | $\exp/\ln$ 组合易引发 overflow/underflow，需 clamp，破坏精确梯度流 |
| **深度受限** | $ D > 5 $ 几乎无法训练成功，限制表达复杂函数的能力 |
| **通用硬件性能差** | 在 x86/GPU 上比 ReLU 慢百倍，不适合云端部署 |

### 未来工作方向（作者建议的研究路线图）
1. **实现 PyTorch 模块**：开发支持 complex autograd、clamped exp/ln 和 Gumbel-softmax leaf selection 的 `EMLNode`。
2. **基准测试**：在 Feynman 数据库上对比 EQL 和 KAN 2.0，评估 symbolic recovery rate。
3. **FPGA 实现**：在 Xilinx Zynq 平台上综合 EML cell，测量实际推理延迟。
4. **工业应用验证**：应用于 PMSM 和电池 surrogate 模型，评估 ISO 26262 认证可行性。
5. **理论拓展**：
   - 探索是否存在**单变量 Sheffer 激活函数** $ \phi: \mathbb{R} \to \mathbb{R} $，避免二叉树结构？
   - 能否重新定义 branch cut 使整个 head 保持实数域运算？

---

## 结论
该论文提出了一种极具前瞻性的 **neuro-symbolic 架构范式转变**：利用数学上完备的 **EML Sheffer operator** 构建兼具**可解释性**与**硬件高效性**的混合模型。虽然在通用硬件上不具备优势，但在 **edge AI、安全攸关系统、数字孪生** 等强调**可验证性**与**低延迟**的应用中，DNN-EML 提供了一条通往“准确、透明、可信”AI 的可行路径。

</details>

---

### 9. [BioTrain: Sub-MB, Sub-50mW On-Device Fine-Tuning for Edge-AI on Biosignals](https://arxiv.org/abs/2604.13359)

**Authors**: Run Wang, Victor J. B. Jung, Philip Wiese, Sebastian Frey, Giusy Spacone, Francesco Conti, Alessio Burrello, Luca Benin  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.13359v1  

#### Abstract
Biosignals exhibit substantial cross-subject and cross-session variability, inducing severe domain shifts that degrade post-deployment performance for small, edge-oriented AI models. On-device adaptation is therefore essential to both preserve user privacy and ensure system reliability. However, exi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《BioTrain: Sub-MB, Sub-50mW On-Device Fine-Tuning for Edge-AI on Biosignals》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
- **跨被试（cross-subject）和跨会话（cross-session）变异性** 导致的 **domain shift** 严重影响了边缘AI在生物信号（如EEG、EOG）上的部署性能。
- 现有边缘设备受限于 **内存（memory）、功耗（power）和算力（computation）**，难以支持完整的 **Backpropagation (BP)**，通常只能采用浅层更新（如Linear Probing, LP）或稀疏训练策略，限制了模型适应能力。

### 提出的新方法与创新思路
- **提出 BioTrain 框架**：一个面向资源受限MCU平台的编译器驱动框架，支持在 **亚兆字节（sub-MB）内存和低于50mW功耗** 下实现 **全网络 fine-tuning（full-network fine-tuning）**。
- **关键技术创新**：
  1. **基于 Deeploy 编译器扩展**：将原本用于高效推理的 Deeploy 扩展为支持端到端训练（forward + backward + 参数更新），生成裸机C代码。
  2. **梯度累积（Gradient Accumulation） + Group Normalization (GN)** 替代 Batch Normalization (BN)：
     - 避免 BN 中对 batch 统计量的依赖，消除跨样本同步开销，使小批量甚至单样本处理成为可能。
     - 支持更大的有效 batch size 而不增加峰值内存占用。
  3. **静态内存分配与时间维度分块（Temporal Tiling）**：
     - 利用 OR-Tools 进行约束优化，实现 scratchpad-aware 分块调度。
     - 对长时序生物信号进行 temporal tiling，减少片外访问，提升片上复用率。
  4. **集成 PULPTrainLib 梯度核函数**：引入优化后的 CNN 梯度算子，并适配 Deeploy 的分块机制。

### 相比现有方法的优势
| 方法 | 是否支持 Full BP | 是否支持 BS > 1 | 内存管理自动化 | 实际部署可行性 |
|------|------------------|------------------|----------------|----------------|
| TinyOL / LP 类方法 | ❌ | ❌ | ❌ | ✅（但性能差） |
| Sparse BP 方法（如 TTE） | ⚠️（部分） | ❌ | ❌ | ⚠️ |
| AIfES | ✅ | ❌ | ⚠️（无系统性分块） | 仅限极小模型 |
| **BioTrain** | ✅ | ✅（通过梯度累积） | ✅（全自动编译优化） | ✅（支持真实EEG/EOG模型） |

> BioTrain 是首个在真实可穿戴MCU平台上实现 **完整、高效的 on-chip full BP** 的框架。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 模态 | 被试数 | 会话数 | 通道数 | 分类任务 | 序列长度 |
|-------|------|--------|--------|--------|----------|-----------|
| EEG Dataset [4] | EEG | 5 | 4 | 8 | 二分类（舌动 vs. 休息） | T=1900 (3.8s @500Hz) |
| EOG Dataset [7]（GAPses 平台） | EOG | 5 | 2 | 3（L/R/C） | 11类眼动识别 | T=1000 (2.0s @500Hz) |

### 实验设置
- **硬件平台**：GAP9 MCU（9核RISC-V集群，128kB L1 + 1.5MB L2 SRAM）
- **预训练方式**：Leave-One-Subject-Out (LOSO) 在云端完成，使用 AdamW（lr=1e-3, bs=64, 40 epochs）
- **设备端微调配置**：
  - 优化器：SGD with Momentum (0.9) + Cosine Annealing
  - 学习率：5e-3，weight decay=1e-3
  - 实际 batch size = 1，但通过 **gradient accumulation over 8 steps** 实现 **effective batch size = 8**
  - 微调轮次：30 epochs
  - 重复5次不同随机种子取均值±标准差

### 评估场景
#### Scenario A: Day-1 User Calibration（冷启动校准）
- 使用目标用户第一会话的80%数据进行微调，20%测试
- 模拟首次佩戴时的个性化校准过程

#### Scenario B: Longitudinal Adaptation（纵向自适应）
- 以 S1 微调后的模型为起点
- 在后续每个会话中用前80%数据继续微调，20%测试
- 模拟长期使用中的生理漂移跟踪

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **No-FT** | 不微调，直接零样本推理 |
| **LP (Linear Probing)** | 仅训练最后一层分类头 |
| **Full-FT** | 全网络微调（原始架构含BN） |
| **Edge-FT (BioTrain)** | 全网络微调 + BN → GN 替换 + 编译优化 |

---

## 3. 主要实验结果和性能指标

### 准确率表现（Table III & Figure 4）

| Task | Method | Avg. Acc. (%) |
|------|--------|---------------|
| **EEG (Day-1)** | No-FT | 50.8 ± 8.8 |
| | LP | 74.8 ± 11.4 |
| | Full-FT | 80.0 ± 11.4 |
| | **Edge-FT (Ours)** | **86.4 ± 7.9** ✅ |
| **EEG (Longitudinal)** | No-FT | 49.8 ± 3.8 |
| | LP | 76.2 ± 7.6 |
| | Full-FT | 78.7 ± 7.3 |
| | **Edge-FT (Ours)** | **83.9 ± 13.3** ✅ |
| **EOG (Day-1)** | No-FT | 78.1 ± 18.1 |
| | LP | 83.3 ± 13.0 |
| | Full-FT | 88.7 ± 10.3 |
| | **Edge-FT (Ours)** | **87.7 ± 10.4** ✅ |
| **EOG (Longitudinal)** | No-FT | 84.1 ± 11.2 |
| | LP | 86.5 ± 10.7 |
| | Full-FT | 89.1 ± 8.8 |
| | **Edge-FT (Ours)** | **87.2 ± 7.6** ✅ |

> **关键发现**：
> - Edge-FT 在 EEG 上相比 No-FT 提升高达 **35.6%**，相比 LP 提升约 **7%**
> - 在纵向适应中避免了 LP 和 Full-FT 的“适应滞后”现象（如 EEG S2 性能骤降），表现出更强鲁棒性
> - 尽管 GN 替换了 BN，准确率未下降反而略有提升，说明其更适合边缘训练场景

### 系统级性能指标（Table IV）

| 指标 | EEG (Edge-FT) | EOG (Edge-FT) |
|------|----------------|----------------|
| **Peak L2 Memory Usage** | **0.67 MB** | **0.28 MB** |
| （对比传统 Full-FT） | ↓8.1× (from 5.4MB) | — |
| **Training Throughput** | 17 samples/s | 85 samples/s |
| **Power Consumption** | < **50 mW** | < **50 mW** |
| **Energy per Session**<br>(40 epochs, 200 samples) | 20.16 mJ | 4.48 mJ |
| **Battery Life Estimate**<br>(320mAh, 3.7V) | ~211 sessions (EEG)<br>~951 sessions (EOG) | |
| **Compute Efficiency** | 0.89 GFLOPs/s | 0.22 GFLOPs/s |

> ✅ 所有训练均可完全在片上执行（on-chip BP），无需频繁访问片外存储。

---

## 4. 关键结论和发现

### 主要发现
1. **全网络 fine-tuning 显著优于 LP**：特别是在跨被试和信号漂移场景下，仅更新最后层无法应对复杂的分布偏移。
2. **GN + Gradient Accumulation 可替代 BN**：在小批量边缘训练中不仅可行，且能保持甚至提升性能，同时大幅降低内存压力。
3. **编译器驱动的自动优化至关重要**：手动优化难以扩展，而 BioTrain 通过前端解析、中端分块规划、后端代码生成实现了端到端自动化部署。
4. **BioTrain 实现了实用化的 on-device training**：在真实可穿戴硬件上达成 **sub-MB 内存、sub-50mW 功耗、每秒数十样本吞吐**，支持反复个性化训练。

### 方法的局限性
- 当前仅支持 FP32 浮点训练，尚未集成量化训练（quantized training）以进一步压缩资源消耗。
- 模型结构仍需轻量化设计（如 MI-BMINet、EpiDeNet），不适用于大型Transformer等复杂架构。
- 训练仍需短期监督标签输入（如临床校准阶段），尚未完全实现无监督在线学习。

### 未来工作方向
- 扩展至 **量化训练（Quantized Training）**，支持 INT8 或更低精度训练。
- 探索更多 **生物信号模态**（如ECG、EMG、PPG）的通用适配能力。
- 结合 **Test-Time Adaptation (TTA)** 技术，减少对持续标注数据的依赖。
- 开源发布：项目已开源在 GitHub → [https://github.com/pulp-platform/Deeploy](https://github.com/pulp-platform/Deeploy)

--- 

> **总结一句话**：  
> BioTrain 成功突破了边缘MCU上全网络反向传播的内存与能耗瓶颈，在真实生物信号场景中实现了高效、隐私友好的个性化持续学习，是迈向真正“智能可穿戴”的重要一步。

</details>

---

### 10. [Physics-Informed Neural Networks for Methane Sorption: Cross-Gas Transfer Learning, Ensemble Collapse Under Physics Constraints, and Monte Carlo Dropout Uncertainty Quantification](https://arxiv.org/abs/2604.13992)

**Authors**: Mohammad Nooraiepour, Zezhang Song, Wei Li, Sarah Perez  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.13992v1  

#### Abstract
Accurate methane sorption prediction across heterogeneous coal ranks requires models that combine thermodynamic consistency, efficient knowledge transfer across data-scarce geological systems, and calibrated uncertainty estimates, capabilities that are rarely addressed together in existing framework...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究旨在解决**多源异构地质系统中甲烷吸附容量预测困难**的问题。该问题具有以下挑战：
- **数据稀缺性**：深部煤层采样成本高、数据稀疏；
- **物理一致性缺失**：传统机器学习模型虽精度高，但可能违反热力学原理（如负吸附量、非单调压力-吸附关系）；
- **不确定性量化不可靠**：缺乏对预测置信度的校准估计，难以支持风险敏感型工程决策。

### 提出的新方法与新思路
作者提出了一种**基于物理信息神经网络（Physics-Informed Neural Networks, PINNs）的跨气体迁移学习框架**，其核心创新包括：

#### （1）跨气体迁移学习（Cross-Gas Transfer Learning）
- 将在氢气（H₂）吸附任务上预训练的PINN迁移到甲烷（CH₄）吸附预测中。
- 利用**弹性权重固化（Elastic Weight Consolidation, EWC）** 技术防止灾难性遗忘，保留H₂中学到的通用物理表示。
- 引入**三阶段课程学习（three-phase curriculum learning）**，逐步平衡迁移保护、数据拟合与物理约束。

#### （2）物理约束下的不确定性量化比较
- 系统评估五种贝叶斯不确定性量化（UQ）方法在强物理约束架构中的表现。
- 发现**Monte Carlo Dropout（MC Dropout）** 在效率与校准质量之间达到最优平衡。
- 揭示“**集成坍缩（Ensemble Collapse）**”现象：在共享物理约束下，deep ensembles因解空间受限而丧失功能多样性，导致UQ失效。

#### （3）可解释AI分析验证物理一致性
- 使用SHAP和ALE进行归因分析，确认模型学到的特征重要性与已知煤吸附机制一致（如水分-挥发分交互作用主导）。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **建模能力** | 单变量等温线模型（仅依赖压力），忽略组分异质性 | 多变量PINN，融合12维物理特征（温度、压力、灰分、水分、挥发分及其交互项） |
| **泛化性** | 易外推失败，黑箱模型不可信 | 物理约束确保输出符合热力学规律，提升外推可靠性 |
| **数据效率** | 需大量目标域数据 | 迁移H₂知识显著加速收敛并提高精度 |
| **不确定性量化** | 不可靠或未提供 | MC Dropout提供**良好校准的不确定性**（ECE=0.101, $p_s$=0.708） |

---

## 2. 核心实验方法和设置

### 数据集
- **来源**：来自114个独立煤样实验的993个平衡测量点。
- **覆盖范围**：涵盖从褐煤（lignite）到无烟煤（anthracite）的所有煤阶。
- **输入特征（共12维）**：
  - 原始测量值：压力（MPa）、温度（K）、水分（wt.%）、灰分（wt.%）、挥发分（wt.%）
  - 物理启发衍生特征：
    - 缩减变量：$T_r = T/T_c$, $P_r = P/P_c$
    - 组分特征：固定碳（FC）、有机质分数（OM）
    - 耦合参数：$\beta = 1/(RT)$
    - 交互项：$P \times T$, 水分×挥发分
- **标签**：CH₄吸附容量（m³/t），经 $y = \log(y + 1)$ 变换以稳定方差。

### 实验设置
- **数据划分**：采用**按实验分组的group-aware分割**（GroupShuffleSplit），保证同一煤样的所有数据点不同时出现在训练集和测试集中（避免样本级信息泄露）。
  - 训练集：91个实验（794个样本）
  - 测试集：23个实验（199个样本）
- **超参数调优**：使用GroupKFold交叉验证（5折），防止组内相关性影响评估。

### 评估指标
| 类别 | 指标 |
|------|------|
| **预测准确性** | $R^2$, RMSE, MAE, MaxAE |
| **不确定性校准** | Expected Calibration Error (ECE), Coverage, Sharpness, Error-Uncertainty Spearman correlation ($p_s$) |
| **统计检验** | Bootstrap配对t检验（Bonferroni校正），Cohen’s d效应量 |

### 基线方法对比
| 模型类型 | 具体方法 |
|--------|--------|
| **经典等温线模型** | Langmuir, Freundlich, Sips（仅以压力为输入） |
| **组合基准** | 三种等温线模型平均 |
| **成分感知模型** | 将挥发分和水分作为修正因子引入$q_{\text{max}}$的Langmuir/Sips模型 |
| **随机初始化PINN** | Xavier初始化编码器，无迁移学习 |
| **Deep Ensembles** | 10个独立训练的随机初始化PINN取平均（计算成本×10） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | $R^2$（测试集） | RMSE（m³/t） | 改进幅度 |
|------|------------------|-------------|----------|
| 经典等温线（压力仅） | 0.285 | 7.41 | — |
| 成分感知等温线 | 0.505 | 6.41 | +77% |
| **本文方法（Transfer-learned PINN）** | **0.932** | **2.29** | **+227% vs 经典** |

> ✅ **说明**：相比经典压力-仅模型，本文方法将解释方差提升了227%，误差降低69%。

### 与基线方法的对比结果
- **迁移学习优势显著**：
  - 相比随机初始化PINN，RMSE降低**18.9%**，收敛速度加快**19.4%**。
  - 统计显著（$p < 10^{-32}$, Cohen’s $d > 1.8$），表明迁移效果不仅统计显著且具实际意义。
- **集成方法失效**：
  - Deep Ensemble（10模型）性能与单个随机PINN无差异（$p=0.815$），计算成本却高出10倍。
  - 所有ensemble变体均出现“**集成坍缩**”，即不同初始化/架构最终收敛至几乎相同的解。

### 消融实验结果
| 模型变体 | RMSE | $R^2$ | 分析结论 |
|--------|-------|-------|----------|
| Transfer-learned PINN | 0.139 | 0.962 | 完整方法 |
| Random-random PINN | 0.171 | 0.942 | 迁移带来显著增益 |
| Random-classical PINN | 0.182 | 0.934 | 单纯物理头初始化无效 |
| Deep Ensemble (10模型) | 0.172 | 0.941 | 无收益，计算浪费 |

> 🔍 **关键发现**：
> - **EWC是关键组件**：它允许编码器在保持源任务知识的同时进行微调，相比冻结权重或仅靠物理约束更有效。
> - **协同效应存在**：H₂编码器 + Sips物理头初始化共同作用时才发挥最大效能。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **跨气体迁移学习可行且高效**
   - H₂ → CH₄的知识迁移成功，源于两者共享的**伦敦色散力主导的物理吸附机制**和**Sips型等温线结构**。
   - 尽管分子质量相差8倍（H₂: 2 amu, CH₄: 16 amu），中间表示仍具可迁移性。

2. ❌ **物理约束导致集成坍缩（Ensemble Collapse）**
   - 所有ensemble方法在共享物理损失函数下均趋于收敛到相似解，丧失功能多样性。
   - 原因：四个物理约束（Sips一致性、单调性、边界有效性、van’t Hoff关系）共同压缩了解空间，形成低维流形 $ \mathcal{M} $，抑制了解之间的分歧。

3. ✅ **MC Dropout是最优UQ策略**
   - 在物理约束下，MC Dropout通过在单一优化解附近采样局部后验分布，实现了最佳权衡：
     - 推理开销仅增加1.5×
     - 达到**良好校准**（ECE=0.101, $p_s$=0.708）
   - 而deep ensembles因坍缩反而产生误导性不确定性（甚至出现负相关 $p_s = -0.044$）。

4. 🧠 **模型学习到物理一致的表示**
   - SHAP和ALE分析显示：
     - 最重要的特征是**水分×挥发分交互项**（重要性17.2%），反映低阶煤中水分对孔隙阻塞的放大效应。
     - 温度呈现U型非单调效应（低温增强吸附，高温抑制），解释了为何其线性相关性接近零。
     - 11/12个特征表现出非单调影响，证明非线性建模必要。

### 方法的局限性
- **适用气体有限**：适用于以范德华力为主的非极性气体（如H₂, CH₄）。对于强极性气体（如CO₂, H₂S），表面静电相互作用不同，迁移效果不确定。
- **依赖源任务数据质量**：若H₂训练数据偏差大或覆盖不足，可能传递错误先验。
- **EWC与MC Dropout结合非严格贝叶斯**：当前为近似处理，未来需更严谨的贝叶斯转移学习框架。

### 未来工作方向
1. 扩展至**多组分气体混合物**（competitive sorption）建模；
2. 引入**时间动态建模**，用于非平衡系统；
3. 验证跨材料迁移能力（如页岩、粘土矿物）；
4. 探索**线性化拉普拉斯近似**（linearized Laplace approximation）围绕EWC-MAP估计，实现更严格的转移学习不确定性分解。

---

> 💡 **总体评价**：  
> 本文构建了一个**数据高效、物理可信、不确定性可解释**的地质材料建模框架，不仅在性能上大幅超越传统方法，更重要的是揭示了**物理约束如何重塑深度学习中的不确定性量化范式**，为科学机器学习提供了普适性指导原则。

</details>

---

### 11. [MyoVision: A Mobile Research Tool and NEATBoost-Attention Ensemble Framework for Real Time Chicken Breast Myopathy Detection](https://arxiv.org/abs/2604.13456)

**Authors**: Chaitanya Pallerla, Siavash Mahmoudi, Dongyi Wang  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.13456v1  

#### Abstract
Woody Breast (WB) and Spaghetti Meat (SM) myopathies significantly impact poultry meat quality, yet current detection methods rely either on subjective manual evaluation or costly laboratory-grade imaging systems. We address the problem of low-cost, non-destructive multi-class myopathy classificatio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*MyoVision: A Mobile Research Tool and NEATBoost-Attention Ensemble Framework for Real Time Chicken Breast Myopathy Detection*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该研究针对**鸡肉胸肌病（Chicken Breast Myopathy）**中的 **Woody Breast (WB)** 和 **Spaghetti Meat (SM)** 两种结构性病变，解决其在工业场景中难以实现低成本、非破坏性、多类别自动检测的问题。传统方法依赖人工触诊或昂贵的实验室成像系统（如高光谱成像），存在主观性强、成本高、部署困难等缺陷。

### 提出了什么新方法或新思路
提出了一套完整的端到端解决方案——**MyoVision**，包含以下三大创新：

1. **智能手机透射成像框架（Smartphone-based Transillumination Imaging）**  
   利用消费级智能手机拍摄鸡胸肉在背光照射下的 **14-bit RAW 图像**，捕捉组织内部因密度、纤维硬化和液体再分布导致的宏观光衰减模式，无需专用光学硬件即可获取内部结构信息。

2. **NEATBoost-Attention Ensemble 模型**  
   提出一种基于 **NeuroEvolution of Augmenting Topologies (NEAT)** 的神经进化优化集成模型，融合：
   - **LightGBM**（梯度提升决策树）
   - **Attention-based MLP (AttentionMLP)**（带注意力机制的多层感知机）  
   通过 NEAT 自动搜索最优网络拓扑与超参数，并采用加权概率融合策略整合两类模型优势，适用于小规模异构表格数据。

3. **统一的移动研究平台（Unified Mobile Research Platform）**  
   开发原生 iOS 应用 MyoVision，集成：
   - RAW 图像采集
   - LiDAR 3D 点云获取与网格重建（Screened Poisson Surface Reconstruction）
   - SAM-based 自动分割
   - ChatGPT 辅助分析
   支持标准化、多模态 RGB-D 数据采集与实时辅助决策。

### 相比现有方法的优势
| 维度 | 本工作 | 现有方法 |
|------|--------|----------|
| 成本 | 消费级手机 + 自制光源 | 高光谱相机、NIRS、定制机器视觉系统 |
| 可扩展性 | 移动化、可现场部署 | 固定设备、需控制环境 |
| 多模态能力 | 支持 RAW + 3D depth + segmentation + LLM 分析 | 多为单一模态 |
| 模型优化方式 | NEAT 自动架构搜索，免手动调参 | 手工设计或传统超参优化（如 Grid Search） |
| 分类任务 | 三分类（Normal/WB/SM） | 多为二分类或仅 WB 检测 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **数据来源**：来自商业加工设施的 **336 块鸡胸肉样本**
- **类别分布**：
  - Normal: 85
  - Woody Breast (WB): 154
  - Spaghetti Meat (SM): 97
- **划分方式**：
  - 训练集：251（63N, 135WB, 52SM）
  - 验证集：34（9N, 19WB, 7SM）
  - 测试集：51（13N, 28WB, 10SM）

### 实验设置和评估指标
- **特征提取**：从 RAW 图像中手工提取 **16 个空间与频域描述符**，包括：
  - 梯度统计量（均值、标准差）
  - 局部方差、边缘密度
  - 方向梯度直方图（5 bins）
  - Gabor 滤波响应（0°, 45°, 90°, 135°）
  - 形态学处理得到的致密区域占比（Percentage Dense Area）
- **训练流程**：
  - 使用 **SMOTE** 对训练折进行类别平衡
  - 在开发集上进行 **5 折交叉验证（Stratified CV）**
  - 最终模型在独立测试集上评估
- **评估指标**：
  - Accuracy
  - Weighted Precision
  - Weighted Recall
  - Macro-weighted F1-score

### 基线方法对比
比较了五种代表性 tabular learning 模型：
| 模型 | 类型 |
|------|------|
| LightGBM | 树模型基准 |
| AttentionMLP | 注意力机制神经网络 |
| TabularCNN | 一维卷积捕捉局部特征交互 |
| LightTransformer | 轻量级 Transformer 结构用于表格数据 |
| Random Forest (feature importance) | 特征可解释性分析工具 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（测试集）
| 模型 | Accuracy | Precision | Recall | F1-score |
|------|----------|-----------|--------|----------|
| **NEATBoost-Attention (Ours)** | **82.4%** | **0.85** | **0.82** | **0.83** |
| LightGBM | 79% | 0.80 | 0.78 | 0.78 |
| AttentionMLP | 77% | 0.82 | 0.77 | 0.77 |
| TabularCNN | 75% | 0.85 | 0.75 | 0.77 |
| LightTransformer | 73% | 0.77 | 0.73 | 0.74 |

> ✅ **最佳表现**：NEATBoost-Attention 在所有指标上均领先，F1 提升约 **4–7%**

### 与基线方法的对比结果
- NEATBoost-Attention 显著优于所有单独模型，表明 **LightGBM 与 AttentionMLP 的互补性**被有效利用。
- 尽管 TabularCNN 和 LightTransformer 在某些指标上有亮点（如高 Precision），但整体稳定性不足。
- **Random Forest on raw features** 仅达到 62.7% ±17.2% 交叉验证准确率，说明原始特征空间判别能力有限，需要高级建模增强。

### 消融实验与进一步分析（隐含消融）
- **特征重要性分析**（Fig. 4b）显示：
  - `Percentage Dense Area` 是最重要特征（重要性 0.165）
  - 其次是梯度相关特征（如 Mean Grad. Mag., Grad. Hist. Bin 5）
- **LDA 投影可视化**（Fig. 4a）揭示：
  - Normal 与 WB 有一定线性可分性
  - SM 样本与两者均有重叠 → **SM 是最难区分的类别**
- **混淆矩阵分析**（Fig. 5）：
  - WB 分类最可靠（recall 最高）
  - SM 错误主要发生在与 Normal 和 WB 之间 → 表明其结构特性介于二者之间

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **智能手机透射成像可用于内部肌肉异常检测**  
   2D 光衰减模式能反映 WB（纤维硬化）和 SM（组织断裂）的不同病理结构，证明消费级设备具备潜力替代高价系统。

2. ✅ **NEAT 驱动的集成学习显著提升小数据性能**  
   NEAT 自动演化出适合小规模异构 tabular 数据的模型结构与超参数组合，避免陷入局部最优，提升泛化能力。

3. ✅ **多模型融合优于单一范式**  
   Tree-based（LightGBM）与 Neural Network（AttentionMLP）具有不同归纳偏置，融合后能更好捕捉非线性关系。

4. ✅ **性能媲美更昂贵技术**  
   本方法取得 **82.4% 准确率（F1=0.83）**，优于 Muñoz-Lapeira 等人使用的 VIS-NIR hyperspectral imaging（76.1%），接近 DenseNet121 + 结构光照明显微（83.4%），而硬件成本低数个数量级。

### 方法的局限性
1. ❌ **Spaghetti Meat 分类性能较低**（~70% recall）  
   因其结构特征介于 Normal 与 WB 之间，在当前 2D 图像特征下难以完全分离。

2. ❌ **数据集规模较小且单源**  
   尤其 SM 样本较少，限制模型鲁棒性和泛化能力；缺乏跨工厂数据验证。

3. ❌ **依赖受控照明条件**  
   当前需固定背光源，尚未实现在动态产线上的自适应曝光与稳定成像。

4. ❌ **未充分利用 3D 几何信息**  
   虽已集成 LiDAR 与 mesh reconstruction，但当前分类仍基于 2D 图像特征，3D 结构未参与建模。

### 未来工作方向
1. 🔮 **引入多模态输入**：融合 LiDAR 提取的 **3D geometric descriptors**（如曲率、厚度变化）以增强对 SM 的识别。
2. 🔁 **扩大数据采集范围**：开展多地区、多加工厂联合采样，构建更具代表性的数据集。
3. 🏭 **推进在线部署**：开发传送带集成照明模块与边缘推理系统，支持 real-time inline inspection。
4. 🧩 **拓展应用场景**：将 MyoVision 平台扩展至其他禽肉品质检测任务，如：
   - White Striping grading
   - Bruise detection
   - Skin color evaluation
   - Pale-Soft-Exudative (PSE) meat identification
5. 💬 **深化 AI-assisted workflow**：结合 ChatGPT 进行报告生成、异常建议、操作指导等闭环智能辅助。

--- 

> **总结一句话**：  
> *MyoVision 证明了“手机+透射成像+NEAT优化集成模型”是一条可行且高效的路径，能够在低成本条件下实现接近高端系统的鸡肉肌病三分类精度，同时为农业食品研究提供了一个强大的移动化、标准化、多模态科研平台。*

</details>

---

### 12. [LLM-HYPER: Generative CTR Modeling for Cold-Start Ad Personalization via LLM-Based Hypernetworks](https://arxiv.org/abs/2604.12096)

**Authors**: Luyi Ma, Wanjia Sherry Zhang, Zezhong Fan, Shubham Thakur, Kai Zhao, Kehui Yao, Ayush Agarwal, Rahul Iyer, Jason Cho, Jianpeng Xu, Evren Korpeoglu, Sushant Kumar, Kannan Achan  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.12096v1  

#### Abstract
On online advertising platforms, newly introduced promotional ads face the cold-start problem, as they lack sufficient user feedback for model training. In this work, we propose LLM-HYPER, a novel framework that treats large language models (LLMs) as hypernetworks to directly generate the parameters...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM-HYPER: Generative CTR Modeling for Cold-Start Ad Personalization via LLM-Based Hypernetworks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在在线广告系统中，新上线的广告（cold-start ads）由于缺乏用户交互历史（如点击、转化等），难以训练有效的 **CTR (Click-Through Rate)** 预估模型，导致推荐效果差、冷启动期长。传统方法依赖历史标签进行监督学习，在无标签场景下表现受限。

### 🚀 提出的新方法与创新思路
提出 **LLM-HYPER** —— 一种将 **Large Language Models (LLMs)** 作为 **Hypernetworks** 来直接生成 CTR 模型参数的全新框架，实现无需训练的冷启动个性化广告排序。

#### 核心创新点：
- **LLM as Hypernetwork**：首次将 LLM 视为超网络（hypernetwork），输入广告多模态内容（文本 + 图像）和用户特征定义，直接推理出线性 CTR 模型的权重向量 $ \theta $，跳过传统训练过程。
- **Few-shot Chain-of-Thought (CoT) 推理**：通过 CLIP 编码检索语义相似的历史广告作为少样本示例，构建 CoT prompt，引导 LLM 进行“思考”并输出合理的权重。
- **解耦部署架构**：将耗时的 LLM 推理置于离线阶段完成，线上仅执行轻量级线性计算，满足工业级低延迟要求（p99 < 1ms）。
- **数值稳定性机制**：引入 **weight normalization** 和 **intercept calibration** 技术，确保生成的权重分布稳定且与生产环境兼容。

### 🔍 相比现有方法的优势
| 方法类型 | 局限性 | LLM-HYPER 的优势 |
|--------|------|------------------|
| 传统监督学习（如 LRwarm） | 需要大量历史点击数据，无法用于真正冷启动 | 完全无需标签，适用于零反馈场景 |
| Embedding-based 冷启动（如 EmbT5） | 依赖语义匹配，难以建模复杂偏好 | 利用 LLM 强大的跨模态推理能力捕捉深层意图 |
| 端到端 LLM 推荐器（如 LLM-R / LLM-TR） | 实时调用 LLM 成本高、延迟大、易产生幻觉 | 将 LLM 输出转化为可解释、可控的线性权重，避免实时推理开销 |

此外，该方法具备良好的 **可解释性** 和 **可控性**，符合电商平台对合规性和审计的需求。

---

## 2. 核心实验方法和设置

### 📊 数据集描述
- 使用美国某头部电商平台的真实广告交互数据。
- 包含过去三个月的 **675 个 warm 广告** 和 **100万用户** 的交互记录。
- 构造冷启动模拟场景：
  - **Retired Ad Set**（历史广告）：前 455 个广告及其训练好的模型权重 → 用于检索 few-shot 示例。
  - **Active Ad Set**（待测广告）：后 120 个较新的广告 → 分为训练集（80万用户）和测试集（20万用户）。

> 注：因数据敏感性，具体字段未公开。

### ⚙️ 实验设置与评估指标

#### 评估任务
- **离线评估**：基于测试集预测 CTR，并进行排序性能评估。
- **在线 A/B 测试**：为期 30 天的真实流量实验，验证实际 CTR 表现。

#### 主要评估指标
| 类型 | 指标 |
|------|------|
| 排序性能 | **AUC**, **NDCG@5**, **NDCG@10** |
| 在线效果 | **Relative CTR score**（相对于 warm-start 模型） |
| 效率 | **Latency (ms)**：单次 CTR 预测平均延迟 |
| 可解释性 | **HitRate@5**, **Coverage@5**, **Consistency Rate** |
| 鲁棒性 | **Accuracy of Weight Direction**（反事实扰动下的响应一致性） |

#### 基线方法对比
| 类别 | 基线名称 | 描述 |
|------|---------|------|
| Warm-start | `LRwarm` | 在 active ad 上有监督训练的理想情况（上界） |
| Cold-start Baseline | `LRcold` | 所有冷广告使用历史广告权重的中位数 |
| Embedding-based | `EmbT5` | 基于 Sentence-T5 的用户-广告语义相似度匹配 |
| LLM 推荐器 | `LLM-R`, `LLM-TR` | 端到端使用 LLM 进行排序（zero-shot） |
| LLM-HYPER（本文） | `LH2.5p`, `LH2.5F`, `LH4o`, `LH5.1` | 不同 LLM backbone 的变体，均采用 5-shot CoT |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Model | AUC | NDCG@5 | NDCG@10 | Latency (ms) |
|-------|-----|--------|---------|-------------|
| `LRwarm` (upper bound) | **0.7005** | **0.0639** | **0.0871** | 0.151 |
| `LRcold` | 0.5593 | 0.0328 | 0.0508 | 0.147 |
| `EmbT5` | 0.4105 | 0.0174 | 0.0262 | 0.357 |
| `LLM-R` | 0.0571 | 0.0093 | 0.0221 | 2130 |
| `LLM-TR` | 0.0576 | 0.0089 | 0.0508 | 3780 |
| **LH2.5p (Ours)** | **0.6722** | **0.0456** | **0.0792** | **0.157** |

#### 性能对比亮点：
- **相比最佳冷启动基线 `LRcold`**：
  - AUC ↑ **+20.2%**
  - NDCG@10 ↑ **+55.9%**
- **接近理想 warm-start 模型**：
  - 达到 `LRwarm` 的 **92% CTR 表现**（见在线测试）
- **显著优于其他 LLM 推荐器**：
  - `LLM-R` 和 `LLM-TR` 几乎失效（AUC ≈ 0.057），且延迟高达 **2–3 秒**
- **保持极低延迟**：
  - 所有 LLM-HYPER 变体延迟均在 **~0.16 ms**，与传统线性模型相当

### 🔍 消融实验结果（Table 2 & Figure 3）

#### (1) CoT 与视觉信息的影响（Table 2）
| Model Variant | NDCG@5 | NDCG@10 |
|--------------|--------|---------|
| Zero-shot | 0.0376 | 0.0677 |
| Zero-shot w/o image | 0.0165 | 0.0352 |
| 3-shot | 0.0421 | 0.0798 |
| 5-shot | 0.0456 | 0.0792 |
| 5-shot w/o image | 0.0292 | 0.0506 |

> 结论：
- **加入图像信息提升巨大**（↓30%+ 性能损失若移除）
- **5-shot CoT 效果最优**，但 3-shot 已足够有效
- 即使 zero-shot 也远超传统冷启动方法（↑33.3% NDCG@10）

#### (2) 可解释性分析（Figure 3）
| Metric | Zero-shot | 3-shot | 5-shot |
|--------|-----------|--------|--------|
| **HR@5** | 0.78 | **0.81** | 0.80 |
| **Coverage@5** | 0.317 | 0.330 | **0.351** |
| **Consistency Rate** | >0.95 | >0.95 | >0.95 |

> 结论：
- CoT 显著提升与人类专家标注的一致性
- LLM 的自然语言推理与其数值输出高度一致（>95% 符号一致）

#### (3) 反事实鲁棒性测试（Figure 4）
在三种语义扰动下评估权重变化是否合理：
- **Enhanced**：增强目标特征 → 权重应上升
- **Diminished**：削弱目标特征 → 权重应下降
- **Neutralized**：去特征化 → 权重趋近零

> 结果：
- **LH5.1 在 Diminished 场景下准确率达 0.88**
- 所有模型在 Enhanced 和 Diminished 场景下均表现出强一致性
- 表明 LLM-HYPER 能基于语义理解动态调整权重，而非依赖关键词统计

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 可以作为高效的 hypernetwork**，无需训练即可为冷启动广告生成高质量的 CTR 模型权重。
2. **结合 CLIP 检索 + Few-shot CoT prompt** 能有效引导 LLM 学习历史广告的“权重模式”，实现跨广告的知识迁移。
3. **多模态输入（尤其是图像）至关重要**，去除图像会导致性能大幅下降。
4. **解耦设计实现了高性能与低延迟的统一**：离线生成权重，线上仅运行线性模型，满足工业部署需求。
5. **具备出色的可解释性与鲁棒性**：生成的权重与人类判断一致，且能合理响应语义扰动。

### ⚠️ 方法的局限性
- **依赖高质量的 few-shot 示例**：若历史广告质量差或领域差异大，可能影响生成效果。
- **LLM 生成成本较高**：虽然不在线上调用，但每个新广告需一次完整 LLM 推理（Gemini-2.5-Pro 平均耗时 102.9s/广告）。
- **对 prompt engineering 敏感**：需要精心设计提示词以减少 hallucination 和数值不稳定。
- **仅适用于线性层权重生成**：当前聚焦于 linear scoring layer，扩展至更复杂的 deep model 仍需研究。

### 🔮 未来工作方向
- 探索 **更高效的 LLM 调用策略**，如蒸馏小模型来模仿 LLM-HYPER 的权重生成行为。
- 扩展至 **multi-task 或 deep model 参数生成**，例如生成整个 MLP 层的参数。
- 引入 **feedback loop**，利用上线后的用户反馈迭代优化 prompt 或生成逻辑。
- 研究 **cross-domain cold-start** 场景，提升方法在新品类、新市场中的泛化能力。

---

## ✅ 总结
**LLM-HYPER 是首个将 LLM 用作 hypernetwork 来解决广告冷启动问题的成功实践**。它巧妙地将 LLM 的强大语义推理能力与工业系统的效率、可控性需求相结合，在无需任何用户反馈的情况下，生成高质量、可解释、稳定的 CTR 模型权重。其已在 Walmart 真实电商平台上成功部署，显著缩短冷启动周期，达到接近 warm-start 模型的效果，是 LLM for Systems 的典范之作。

</details>

---

### 13. [Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization](https://arxiv.org/abs/2604.12290)

**Authors**: Yizhe Chi, Deyao Hong, Dapeng Jiang, Tianwei Luo, Kaisen Yang, Boshi Zhang, Zhe Cao, Xiaoyan Fan, Bingxiang He, Han Hao, Weiyang Jin, Dianqiao Lei, Qingle Liu, Houde Qian, Bowen Wang, Situ Wang, Youjie Zheng, Yifan Zhou, Calvin Xiao, Eren Cai, Qinhuai Na  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.12290v1  

#### Abstract
Current LLM agent benchmarks, which predominantly focus on binary pass/fail tasks such as code generation or search-based question answering, often neglect the value of real-world engineering that is often captured through the iterative optimization of feasible designs. To this end, we introduce Fro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前主流的 **LLM agent benchmarks** 主要集中在“从零到一”（0-to-1）的**一次性任务**上，例如代码生成是否通过测试用例、搜索问答是否给出正确答案等。这类任务通常具有**二元奖励**（binary pass/fail），忽略了现实工程中更核心的价值来源：**在已有可行方案基础上进行迭代优化**。

然而，在真实世界工程中（如电池充电算法、结构设计、调度系统等），真正的价值往往来自于对一个**已满足约束的初始方案**进行持续改进，以在有限预算下逼近最优解。这一过程是**开放式的、无理论最优解的、依赖仿真器反馈的连续优化过程**。

现有 benchmark 缺乏对此类“**生成式优化**（generative optimization）”能力的系统性评估。

---

### 提出了什么新方法或新思路

本文提出了 **Frontier-Eng**，一个面向真实世界工程任务的大规模 benchmark，用于评估 AI agents 在 **generative optimization** 场景下的能力。

#### 核心定义：Generative Optimization
- **形式化为一个三元组** $ T = (C, x_0, E) $
  - $ C $: 任务上下文（problem spec, constraints, docs）
  - $ x_0 $: 初始可行解（editable artifact）
  - $ E $: 可执行验证器（evaluator），返回：
    - 可行性标志 $ v(x) \in \{0,1\} $
    - 连续得分 $ s(x) \in \mathbb{R} $（仅当可行时有效）

- **Agent 的目标**：在固定交互预算 $ B $ 内，通过反复提出修改建议、接收验证器反馈、自我修正，最大化找到的最佳可行得分：
  $$
  s^* = \max_{0 \leq t < B, v_t=1} s_t
  $$

- **Agent 架构特点**：基于 LLM 的 proposal 机制 + 搜索策略（如进化搜索、树搜索）构成闭环反馈系统。

---

### 相比现有方法的优势

| 维度 | 传统 Benchmark | Frontier-Eng |
|------|----------------|-------------|
| **任务性质** | 一次性生成，有明确答案 | 迭代优化，无理论最优 |
| **奖励信号** | 二元（pass/fail） | 连续得分 + 硬约束 |
| **评估方式** | 静态测试 | 动态仿真器反馈 |
| **工程真实性** | 低（纯代码/文本） | 高（工业级仿真器） |
| **防作弊机制** | 弱 | 强（read-only verifier, sandboxed execution） |

**优势总结**：
- ✅ 更贴近真实工程研发流程（build-test-refine 循环）
- ✅ 支持跨领域统一评估（computing, robotics, OR, optics, physical design）
- ✅ 引入**可执行验证器**（executable verifier）提供可信、不可篡改的反馈
- ✅ 设计了防止 reward hacking 的三层保障机制：
  1. **隔离性**（Isolation）：验证器与数据只读
  2. **验证器解析评分**（Verifier-parsed scoring）：分数来自仿真日志而非 agent 自报
  3. **评估鲁棒性**（Evaluation robustness）：多场景/随机种子平均，避免过拟合

---

## 2. 核心实验方法和设置

### 使用的数据集

**Frontier-Eng 包含 47 个真实工程任务**，覆盖五大类别：

| 类别 | 任务数 | 示例任务 |
|------|--------|---------|
| **Computing & Quantum Information** | 10 | FlashAttention kernel optimization, SHA3-256 throughput |
| **Operations Research & Decision Science** | 9 | Job-shop scheduling (ABZ/SWV/TA), inventory optimization |
| **Robotics, Control & Energy Systems** | 8 | Quadrotor PID tuning, battery fast-charging profile |
| **Optics & Communication Systems** | 10 | Fiber WDM power allocation, holographic focusing |
| **Physical Sciences & Engineering Design** | 10 | Truss topology optimization (ISCSO), reaction yield Pareto optimization |

任务来源多样：
- 工程竞赛（ISCSO）
- 学术基准（MQT Bench, Summit）
- 经典课程项目（MallocLab）
- 工业仿真工具（MuJoCo, PyBullet, SustainDC）
- 领域专家原创贡献

---

### 实验设置和评估指标

#### 实验设置
- **模型**：评测了 9 个前沿 LLM，包括：
  - `claude-opus-4.6`（Claude 4.6 Opus）
  - `gpt-5.4`, `gpt-oss-120b`
  - `glm-5`, `deepseek-v3.2`, `qwen3-coder-next` 等
- **搜索框架**：统一使用 `openevolve` 框架，预算为 100 次迭代
- **初始化**：所有模型从相同的初始可行解开始
- **环境**：每个任务在声明的运行环境中独立执行（conda/Docker/Python）

#### 评估指标（多层级）

| 指标 | 定义 | 优点 |
|------|------|------|
| **Average Rank**（主指标） | 每个任务内按最佳得分排名，取跨任务平均排名 | 单位无关，公平比较异构任务 |
| **Performance Profile**（分布分析） | $ P_m(\alpha) = \frac{1}{N} \left|\{i: p_{i,m} \leq \alpha\}\right| $，其中 $ p_{i,m} = \frac{s^*_{i}}{s_{i,m}} $ | 展示方法在不同容忍度下的竞争力 |
| **Win Rate over Baseline** | 在多少任务上优于初始解 | 衡量“可靠性提升”能力 |
| **Category-level Breakdown** | 分类别的平均排名 | 揭示模型在特定领域的强弱项 |

---

### 基线方法对比

- **基线不是传统 ML 模型**，而是不同 LLM + 不同搜索策略的组合。
- 对比了三种 search frameworks：
  - `abmcts`: 基于蒙特卡洛树搜索
  - `openevolve`: 结构化进化搜索
  - `shinkaevolve`: 改进版进化搜索，支持更细粒度更新
- 所有对比均在同一任务接口、相同初始解、相同验证器下进行，确保公平。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）模型横向对比（openevolve, 100 iter）

| 模型 | 平均排名（越低越好） | Win Rate over Baseline |
|------|------------------------|--------------------------|
| **claude-opus-4.6** | **3.18** ✅ | 91.5% |
| glm-5 | 4.02 | **97.9%** ✅ |
| gpt-oss-120b | 4.46 | 95.7% |
| deepseek-v3.2 | 4.41 | 93.6% |

👉 **Claude 4.6 Opus 表现最稳健**，在最多任务中排名第一（20/47），且在近最优范围内表现最强。

#### （2）不同搜索框架下模型表现（vs gpt-oss-120b）

| 框架 | Claude Rank | OSS Rank | Claude Wins |
|------|-------------|----------|--------------|
| abmcts | 1.43 | 1.55 | 26 |
| **openevolve** | **1.28** | 1.64 | **30** |
| shinkaevolve | **1.28** | 1.62 | **29** |

👉 **Claude 在所有框架下均优于 gpt-oss-120b**，尤其在 `openevolve` 和 `shinkaevolve` 中优势最大。

---

### 消融实验结果

#### （1）优化动态遵循双幂律（Dual Power-Law Decay）

在 500 次迭代实验中发现：

- **改进频率** 随迭代次数衰减：$ \text{frequency} \propto t^{-1} $
- **改进幅度** 随改进序号衰减：$ \text{magnitude} \propto k^{-1} $

> 🔍 **含义**：早期易获大幅改进，后期改进越来越稀少且微小，边际收益快速趋零。

#### （2）深度 > 宽度（Depth Dominates Width）

在固定总预算 $ B = n \times d \leq 256 $ 下比较：

| 配置（n chains × d depth） | 归一化得分 |
|----------------------------|------------|
| 1 × 256 | **1.00** ✅ |
| 2 × 128 | 0.99 |
| 4 × 64 | 0.99 |
| 8 × 32 | 0.97 |
| 16 × 16 | 0.91 |

👉 **单条长链（deep）显著优于多条短链（wide）**，说明**上下文积累和深层推理至关重要**，重启会丢失关键进展。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Claude 4.6 Opus 是目前最强大的 generative optimization agent**，在多数任务中表现最佳，尤其擅长早期高质量提案。
2. ✅ **openevolve 和 shinkaevolve 框架能更好释放模型潜力**，相比保守的 abmcts 更利于探索。
3. ✅ **优化过程呈现“早快后慢”的幂律规律**，前 50–100 步决定大部分性能增益。
4. ✅ **搜索深度远比宽度重要**：坚持一条长链比并行多条短链更有效，强调**记忆与累积推理**的重要性。
5. ✅ **模型与框架存在交互效应**：
   - Claude 擅长“一步到位”的结构性重写
   - gpt-oss-120b 虽起点低，但在 `shinkaevolve` 下能通过渐进式改进缩小差距

---

### 方法的局限性

- **计算成本高**：每个任务需数百次仿真调用，不适合快速迭代研究。
- **任务静态性**：所有任务固定不变，无法评估 agent 在动态环境中的适应能力。
- **人类标注依赖**：部分任务仍需专家设计初始解与验证逻辑。
- **未涵盖多 agent 协作**：当前为单 agent 优化，未涉及团队协同设计。

---

### 未来工作方向

1. **扩展任务规模与多样性**：加入更多高保真物理仿真（如 CFD、FEM）、实时控制系统。
2. **引入动态与不确定性环境**：允许任务参数随时间变化，测试 agent 的在线适应能力。
3. **构建 agent 协作 benchmark**：模拟工程师团队协作优化复杂系统。
4. **探索更高效的搜索策略**：结合强化学习、贝叶斯优化等提升采样效率。
5. **推动开源生态**：鼓励社区提交新任务，形成可持续演进的工程智能 benchmark 生态。

---

> **结语**：  
> Frontier-Eng 标志着 AI agent benchmark 从“答题机器”向“工程伙伴”的范式转变。它不再问“你能写出正确代码吗？”，而是问“你能在真实约束下，像工程师一样不断逼近极限吗？”——这是迈向 **Engineering AGI** 的关键一步。

</details>

---

### 14. [DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](https://arxiv.org/abs/2604.12812)

**Authors**: Hao Yan, Yuliang Liu, Xingchen Liu, Yuyi Zhang, Minghui Liao, Jihao Wu, Wei Chen, Xiang Bai  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.12812v2  

#### Abstract
Existing Multimodal Large Language Models (MLLMs) suffer from significant performance degradation on the long document understanding task as document length increases. This stems from two fundamental challenges: 1) a low Signal-to-Noise Ratio (SNR), with crucial evidence buried in irrelevant pages; ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Multimodal Large Language Models (MLLMs)** 在处理**长文档理解任务**时面临两个根本挑战：
1. **低信噪比 (Low Signal-to-Noise Ratio, SNR)**：关键证据被大量无关页面淹没，导致模型难以定位有效信息。
2. **监督信号稀缺 (Supervision Scarcity)**：现有数据集通常只提供最终答案，缺乏中间推理步骤（如证据定位、信息整合），导致模型倾向于记忆而非真正推理。

这些问题使得模型在**长文档场景下性能显著下降**，且推理过程不可解释，泛化能力差。

---

### 提出的新方法与创新思路
为解决上述问题，作者提出 **DocSeeker**，其核心是引入一种新的视觉推理范式和训练框架：

#### ✅ 创新点 1：**Analysis-Localization-Reasoning (ALR) 推理范式**
- 受人类认知启发，要求模型遵循结构化推理流程：
  1. **Question Analysis**：分析问题意图；
  2. **Evidence Localization**：显式识别支持答案的关键页面（通过 `evidence_pages` 输出）；
  3. **Reasoning Process**：基于定位到的证据进行逻辑推导；
  4. 最终输出答案及引用页码。
- 引入 **Page-Aware Input Representation**：每个页面前添加文本标识（如 "Page 9"），使模型能将信息锚定到具体页。

> 📌 优势：提升可解释性，强制模型学习跨页信息区分能力，缓解噪声干扰。

#### ✅ 创新点 2：**两阶段训练框架**
1. **Stage I: Supervised Fine-Tuning (SFT)**
   - 使用高效的**知识蒸馏策略**，利用 **Gemini-2.5-Flash** 教师模型生成高质量的 ALR Chain-of-Thought (CoT) 数据。
   - 蒸馏输入仅包含真实证据页 + 少量干扰页，降低成本并提高成功率。
   - 经过自动化（Exact Match）和语义级（GPT-4o）双重验证确保数据质量。

2. **Stage II: Evidence-aware Group Relative Policy Optimization (EviGRPO)**
   - 基于强化学习的优化方法，联合优化三项目标：
     - `R_format`: 输出格式正确性；
     - `R_evidence`: 证据页定位准确率（加权 F1，强调召回）；
     - `R_answer`: 答案正确性（ANLS 指标）。
   - 总奖励函数：  
     $ R = \lambda_1 R_{\text{format}} + \lambda_2 R_{\text{evidence}} + \lambda_3 R_{\text{answer}} $

> 📌 优势：超越模仿学习，直接从结果反馈中优化证据定位与推理能力。

#### ✅ 创新点 3：**Evidence-Guided Resolution Allocation (EGRA)**
- 针对长文档训练内存受限问题，提出差异化分辨率策略：
  - **证据页**：保持高分辨率（1024）；
  - **非证据页**：70% 随机降采样至低分辨率（256），其余保留高分辨率。
- 推理阶段所有页面均用高分辨率处理。

> 📌 优势：减少训练 token 数量，缓解 GPU 内存压力；同时增强信噪比，避免简单截断或丢弃带来的信息损失。

---

### 相比现有方法的优势
| 方面 | DocSeeker | 传统 MLLMs / RAG 方法 |
|------|-----------|------------------------|
| **推理结构** | 显式结构化流程（ALR） | 黑箱式端到端输出 |
| **可解释性** | 支持证据溯源（evidence_pages） | 不透明，无来源标注 |
| **长文档鲁棒性** | 即使文档极长也几乎不退化 | 随长度增加性能急剧下降 |
| **与 RAG 兼容性** | 天然协同，抗检索噪声强 | 对 Top-k 敏感，k 小易漏检，k 大则噪声多 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练数据**：
  - 来源于 **MP-DocVQA** 和 **DUDE**（均为 ≤20 页的多页文档 VQA 数据集）
  - 经过滤除短文档后，共构建 19,386 个样本
  - 通过蒸馏得到 **13,986 个高质量 ALR CoT 样本**用于 SFT

- **评估数据集**：
  | 数据集 | 特点 |
  |-------|------|
  | **MP-DocVQA**, **DUDE** | In-Domain 测试（≤20 页） |
  | **MMLongBench-doc** | OOD，最长达 468 页，含图表 |
  | **LongDocURL** | 平均约 30 页，涵盖网页截图等复杂布局 |
  | **SlideVQA** | 幻灯片类多图问答 |

---

### 实验设置与评估指标
- **主干模型**：基于 **Qwen-2.5-VL-7B-Instruct**
- **训练配置**：
  - SFT 阶段：2 轮，学习率 1e-6
  - GRPO 阶段：6 轮，rollout group size=16，reward 权重 $(0.1, 0.3, 0.6)$
- **推理分辨率**：每页 1024×784，平衡清晰度与上下文容量

- **评估指标**：
  - **ANLS**（Average Normalized Levenshtein Similarity）：用于 MP-DocVQA、DUDE
  - **Accuracy**：用于 MMLongBench-doc、LongDocURL
  - **F1 Score**：用于 SlideVQA

---

### 基线方法对比
比较了三类主流方法：
| 类型 | 代表模型 |
|------|---------|
| **End-to-End MLLMs** | InternVL3, mPLUG-DocOwl2 |
| **Parsing-based** | HiVT5, CREAM, Docpilot, DocVLM |
| **Retrieval-Augmented (RAG)** | Vis-RAG, SV-RAG, VDocRAG, M3DocRAG |
| **Closed-source Commercial** | GPT-4o, GPT-4V, Qwen-VL-Max, Gemini-1.5-Pro |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 方法 | DUDE (ANLS) | MPDocVQA (ANLS) | MMLong. (Acc) | LongDoc. (Acc) | SlideVQA (F1) |
|------|-------------|------------------|----------------|------------------|----------------|
| **Baseline (Qwen-2.5-VL-7B)** | 35.2 | 70.1 | 25.4 | 37.8 | 59.8 |
| **DocSeeker-SFT** | 56.8 | 82.1 | 38.6 | 49.1 | 75.2 |
| **DocSeeker (Ours)** | **57.4** | **86.2** | **40.1** | **51.7** | **77.1** |

> 🔺 **相比 Baseline 提升高达 +30~60%**，尤其在 OOD 长文档上表现突出。

- 在 **MMLongBench-doc** 上，DocSeeker 达到 **40.1% 准确率**，远超其他开源模型（第二名为 32.4%），甚至接近 GPT-4o（42.8%）。
- 在 **LongDocURL** 上达到 **51.7%**，显著优于基线（37.8%）。

---

### 消融实验结果

#### ✅ Ablation on Reasoning Paradigm（Table 3）
| 方法 | MMLong. Acc | F1 |
|------|--------------|-----|
| Raw short-answer data | 27.4 | 27.6 |
| Vanilla CoT | 31.3 | 32.4 |
| **ALR CoT (ours)** | **33.8** | **33.9** |
| w/o Page ID | 30.4 | 31.1 |

> 结论：**ALR 范式 + Page ID 显著提升泛化能力**，证明结构化引导的重要性。

#### ✅ Ablation on Data Size
| 数据量（占全部 ALR CoT） | Acc | F1 |
|----------------------------|-----|-----|
| 20% | 32.7 | 31.7 |
| 40% | 35.8 | 33.8 |
| 80% | 38.2 | 36.5 |
| 100% | **38.6** | **36.9** |

> 结论：**更多 ALR CoT 数据持续带来增益**，说明该范式具有良好的扩展性。

#### ✅ Ablation on Resolution Strategy（Table 4）
| 方法 | Acc | F1 |
|------|-----|-----|
| Fixed-Res (truncated) | 36.6 | 35.8 |
| Fixed-Res (low-res full) | 34.2 | 33.5 |
| **EGRA (ours)** | **38.6** | **36.9** |

> 结论：**EGRA 明显优于固定分辨率策略**，验证其在资源受限下的有效性。

#### ✅ Ablation on Reward Weights（Table 5）
| 方法 | Acc | F1 |
|------|-----|-----|
| SFT only | 38.6 | 36.9 |
| Vanilla GRPO | 38.7 | 36.3 |
| **EviGRPO ($\lambda_2=0.3$)** | **40.1** | **38.4** |

> 结论：**引入证据定位奖励可进一步提升性能**，但权重过高会导致过度关注定位而忽略答案准确性。

---

## 4. 关键结论和发现

### 主要发现
1. **ALR 范式显著提升长文档理解能力**：
   - 强制模型执行“先找证据再推理”的流程，有效对抗长序列中的噪声干扰。
   - 即使训练数据仅为短文档（≤20页），也能**稳健泛化至数百页的 ultra-long 文档**。

2. **结构化监督至关重要**：
   - 仅使用最终答案进行 SFT 导致模型记忆而非推理，OOD 表现差。
   - 高质量 ALR CoT 数据提供了更强的学习信号，推动模型掌握通用推理机制。

3. **EviGRPO 实现精准优化**：
   - 传统的 RL 方法仅优化答案正确性，而 EviGRPO 同时优化证据定位，实现更细粒度控制。

4. **与 RAG 系统天然协同**：
   - 如 Figure 4 所示，当集成 **ColPali retriever** 后，DocSeeker 在不同 Top-k 下均保持高性能，而 baseline 模型随 k 增大迅速崩溃。
   - 表明 DocSeeker 可作为下一代 **Visual RAG 系统的理想 Reader 模块**。

---

### 方法的局限性
1. **推理延迟略有增加**：
   - 因需生成完整 ALR 过程，输出 token 数翻倍（Baseline: 202 → DocSeeker: 401），端到端延迟由 19s 增至 25s（+31.6%）。
   - 但主要瓶颈仍在视觉预填充阶段，影响可控。

2. **依赖高质量 CoT 数据构建**：
   - 虽采用高效蒸馏策略，但仍需强大教师模型（如 Gemini）和严格验证流程，成本较高。

3. **未完全解决极端长度下的 token 限制**：
   - 当前仍受限于 context window，虽 EGRA 缓解了部分压力，但对万页级文档仍具挑战。

---

### 未来工作方向
1. **构建更大规模的 ALR-CoT 数据集**，探索自动生成与迭代优化机制。
2. **将 DocSeeker 与动态 retrieval 系统深度耦合**，实现“检索-精读-再检索”的闭环推理。
3. **探索轻量化版本**，适配移动端或实时应用场景。
4. **拓展至多文档交叉推理、表格深度解析等更复杂任务**。

---

## 总结

📌 **DocSeeker 是首个系统性地将“结构化视觉推理 + 证据定位 + 强化学习优化”结合的长文档理解框架**。它不仅在多个基准上取得 SOTA 性能，更重要的是揭示了一条通往**可解释、强泛化、高鲁棒**的多模态推理系统的可行路径，为未来构建可靠的 **Visual RAG 系统**奠定了坚实基础。

</details>

---

### 15. [RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair](https://arxiv.org/abs/2604.12820)

**Authors**: Jagadeesh Rachapudi, Pranav Singh, Ritali Vatsi, Praful Hambarde, Amit Shukla  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.12820v1  

#### Abstract
Large language models (LLMs) inherently absorb harmful knowledge, misinformation, and personal data during pretraining on large-scale web corpora, with no native mechanism for selective removal. While machine unlearning offers a principled solution, existing approaches are provider-centric, requirin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在预训练过程中会无差别地吸收有害知识、虚假信息和个人隐私数据，且目前缺乏**原生机制**来选择性删除这些已学习的知识。现有的 **Machine Unlearning (MU)** 方法大多由模型服务提供商（MSP）主导，依赖完整的训练流程和保留数据集（retain dataset），普通用户无法自主控制自己的数据是否被遗忘。

这导致两个核心问题：
- 用户隐私权（如 GDPR 和 CCPA 规定的“被遗忘权”）难以真正实现；
- 现有方法不支持在推理阶段（test time）、通过自然语言指令进行交互式遗忘。

### 提出了什么新方法或新思路
本文提出了一种全新的范式——**Interactive Machine Unlearning (IMU)**，并构建了对应的框架 **RePAIR** 来实现该目标。

#### （1）IMU：交互式机器遗忘
允许终端用户在与 LLM 交互时，直接用自然语言发出“请忘记某条信息”的请求，模型即可在推理阶段自主完成参数更新，无需依赖 MSP 或重新训练。

#### （2）RePAIR 框架
一个端到端的三模块系统：
- **Mwatchdog**：从对话历史中检测用户的遗忘意图，并提取需遗忘的 `(pf, rf)` 对；
- **Msurgeon**：生成用于模型修复的代码（repair code）；
- **Mpatient**：原始模型，在接收到修复指令后执行参数修改，变为 `Mhealed`。

#### （3）STAMP：核心算法
提出了两种训练无关（training-free）、单样本（single-sample）遗忘方法：
- **STAMP**（Steering Through Activation Manipulation with PseudoInverse）：
  - 利用伪逆（pseudoinverse）对 MLP 层激活值进行闭式求解更新；
  - 将“遗忘样本”的激活引导至“拒绝子空间”（refusal subspace），使模型学会说 “I don’t know”；
  - 完全无需梯度计算或反向传播。
- **STAMP-LR**（Low-Rank Variant）：
  - 对输入矩阵进行低秩分解，将时间复杂度从 $O(d^3)$ 降至 $O(r^3 + r^2 \cdot d)$；
  - 显著提升效率，适用于边缘设备上的 on-device unlearning。

### 相比现有方法的优势
| 维度 | 现有 MU 方法（如 GA, NPO, FLAT 等） | RePAIR / STAMP |
|------|----------------------------------|----------------|
| 是否需要训练 | 是（需 backpropagation） | 否（training-free） |
| 是否支持单样本遗忘 | 多数为 batch-level，效果差 | 支持，且表现优异 |
| 用户能否参与 | 否（仅限 MSP 操作） | 是（通过自然语言交互） |
| 推理时可执行？ | 否 | 是 |
| 计算开销 | 高（GPU 内存大，耗时长） | 极低（尤其 STAMP-LR） |
| 实际部署可行性 | 差 | 强（适合移动端/本地部署） |

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **WMDP-Bio**（1K 样本）  
   - 用于评估**有害知识抑制**（如医学错误信息）
2. **MMLU**（1K 样本，ground truth 被污染）  
   - 用于评估**虚假信息纠正**
3. **合成个人资料数据集**（2K profile）  
   - 使用 Mistral-7B API 生成，模拟真实世界中的私人信息泄露场景，用于**个人数据擦除**

所有数据集均划分为等量的 `Df`（forget set）和 `Dr`（retain buffer ≤10% of total），另设 `Dref`（200 条“我不知道”类拒绝提示）用于构建 steering vector。

### 实验设置
- **主模型（Mpatient）**：Llama-3-8B
- **监控模型（Mwatchdog）**：Mistral-7B（用于意图识别与配对提取）
- **外科医生模型（Msurgeon）**：Qwen2.5-Coder-7B-Instruct（生成修复代码）
- 所有操作在推理阶段完成，无任何微调或梯度回传

### 评估指标
| 指标 | 含义 | 期望方向 |
|------|------|--------|
| **Accf ↓** | 忘记准确率（越低越好，理想为 0） | ↓ |
| **Accr ↑** | 保留准确率（越高越好，接近 Oracle） | ↑ |
| **F-RL ↓** | 忘记集合上的 ROUGE-L（衡量记忆残留） | ↓ |
| **R-RL ↑** | 保留集合上的 ROUGE-L（保持语义一致性） | ↑ |
| **Perplexity on TinyStories ↓** | 模型整体语言能力（utility） | ↓（越低越好） |
| **RTE (Runtime Efficiency) ↓** | 总运行时间（分钟） | ↓ |

### 基线方法对比
共比较六种 SoTA 方法：
- GA [25]
- NPO [28]
- RMU [17]
- FLAT [24]
- WGA [23]
- ASU [27]

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 方法 | Harmful Knowledge Removal (Accf↓) | Misinformation Removal (Accf↓) | Personal Data Erasure (F-RL↓) | Runtime Efficiency (RTE, min) |
|------|-------------------------------|------------------------------|----------------------------|-----------------------------|
| Base | 75.30                         | 83.70                        | 0.87                       | N/A                         |
| Oracle | N/A                           | N/A                          | N/A                        | N/A                         |
| GA   | 0.00                          | 0.00                         | 0.13                       | 12.25                       |
| NPO  | 0.00                          | 0.10                         | 0.27                       | 11.17                       |
| RMU  | 0.00                          | 0.27                         | 0.16                       | 12.50                       |
| FLAT | 0.01                          | 1.30                         | 0.33                       | 12.13                       |
| WGA  | 2.10                          | 2.47                         | 0.45                       | 11.20                       |
| ASU  | 0.90                          | 0.10                         | 0.07                       | 12.13                       |
| **STAMP**     | **0.00**                      | **0.00**                     | **0.00**                   | **7.13**                    |
| **STAMP-LR**  | **0.00**                      | **0.00**                     | **0.00**                   | **4.25**                    |

> ✅ 所有遗忘任务中，**STAMP 和 STAMP-LR 均达到 Accf = 0.00，F-RL = 0.00**，即完全遗忘目标内容。

### 保留性能与模型效用
| 方法 | Misinformation Accr↑ | Personal Data R-RL↑ | Perplexity ↓ |
|------|--------------------|-------------------|------------|
| Oracle | 85.30              | 0.90              | 5.25       |
| STAMP | 80.13              | 0.79              | 6.02       |
| **STAMP-LR** | **84.47**          | **0.88**          | **7.39**   |

> 🔹 STAMP-LR 在保留准确性上逼近 Oracle，显著优于大多数 baseline（如 GA/NPO 的 Accr < 75）  
> 🔹 模型 utility（perplexity）稳定在 6–8 区间，远好于训练型方法（普遍 >10）

### 运行效率（Speedup）
- STAMP 相比训练型方法平均提速约 **1.7×**
- STAMP-LR 达到最高 **~3× speedup**
- 单样本遗忘场景下优势尤为明显（见 Table 8）

### 消融实验结果

#### （1）单层 vs 全层干预（Table 5）
| 设置 | F-RL↓ | R-RL↑ | RTE(s) |
|------|-----|-----|-------|
| Layer 7 only | 0.00 | 0.85 | **4.36** |
| All layers | 0.00 | 0.88 | 15.40 |

> 💡 发现第 7 层 MLP 激活最具区分性（cosine divergence 最高），单独干预即可获得近似全层效果，速度提升 **~3.8×**

#### （2）低秩参数 `r` 影响（Table 6）
- 当 `r ≥ 64` 时，性能稳定；
- `r < 64` 出现遗忘不彻底现象（capacity 不足）

#### （3）保留集大小敏感性（Table 7）
- 即使 `Dr` 缩减至完整保留集的 10%，性能仍保持稳定
- 表明 RePAIR 对存储资源要求极低，利于边缘部署

#### （4）单样本遗忘能力（Table 8）
| 方法 | Accf↓（|Df|=1） | Accr↑ |
|------|----------------|-------|
| 所有 baseline（GA/NPO/FLAT 等） | **100**（完全未遗忘） | <55 |
| **STAMP** | **0.00** | 70.13 |
| **STAMP-LR** | **0.00** | 73.27 |

> ⚠️ 所有基于训练的方法在单样本情况下彻底失效（梯度信号被 retain set 淹没）  
> ✅ STAMP 类方法是唯一能在单样本下有效工作的方案

---

## 4. 关键结论和发现

### 主要发现
1. **首次实现了真正的用户驱动遗忘（IMU）**  
   用户可通过自然语言指令让 LLM 在推理阶段自主遗忘特定知识，打破对 MSP 的依赖。

2. **STAMP 是首个 training-free + single-sample 的 LLM unlearning 方法**  
   不依赖反向传播，仅通过前向传播和伪逆运算即可完成参数修正，满足实际应用场景需求。

3. **高效且实用性强**  
   - STAMP-LR 实现 ~3× 加速，可在消费级设备上运行；
   - 仅需极少保留数据（10%），降低存储负担；
   - 可精准作用于单一层（如 Layer 7），进一步优化性能。

4. **遗忘质量接近 Oracle 水平**  
   在 Accf、F-RL 上达到近乎完美的 0.00，同时保留能力和语言流畅性高度可控。

### 方法的局限性
1. **仍需小规模 retain buffer**  
   尽管只需 10%，但在严格合规环境下（如 GDPR），任何形式的数据留存都可能构成风险。而 FLAT 等方法可做到 retain-free。

2. **测试时资源消耗仍高于纯推理**  
   虽然比训练轻量，但激活重定向和伪逆计算仍有一定开销，不适合极高吞吐场景。

3. **当前仅适用于文本模态**  
   尚未扩展到多模态基础模型（multimodal foundation models）。

### 未来工作方向
1. 开发 **fully retain-free** 的 unlearning 方法，彻底消除数据留存隐患；
2. 将 RePAIR 扩展至视觉-语言模型（如 LLaVA、Flamingo）；
3. 探索更高效的硬件适配方案（如手机端实时遗忘）；
4. 结合联邦学习或差分隐私，增强整体隐私保护体系。

--- 

> 📌 **总结一句话**：  
> RePAIR 首次实现了**用户可用、设备可跑、即时生效**的 LLM 忘记机制，推动 AI 安全与隐私治理进入“人人可控”的新时代。

</details>

---

### 16. [DeEscalWild: A Real-World Benchmark for Automated De-Escalation Training with SLMs](https://arxiv.org/abs/2604.13075)

**Authors**: Md Hasebul Hasan, Krity Haque Charu, Eshwara Prasad Sridhar, Shuchisnigdha Deb, Mohammad A. Islam  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.13075v1  

#### Abstract
Effective de-escalation is critical for law enforcement safety and community trust, yet traditional training methods lack scalability and realism. While Large Language Models (LLMs) enable dynamic, open-ended simulations, their substantial computational footprint renders them impractical for deploym...

---

### 17. [SAKURAONE: An Open Ethernet-Based AI HPC System and Its Observed Workload Dynamics in a Single-Tenant LLM Development Environment](https://arxiv.org/abs/2604.13600)

**Authors**: Fumikazu Konishi, Yuuki Tsubouchi, Hirofumi Tsuruta  
**Category**: cs.DC  
**Published**: 2026-04-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.13600v1  

#### Abstract
SAKURAONE is a managed high performance computing (HPC) cluster developed and operated by the SAKURA Internet Research Center. It builds on the KOKARYOKU PHY bare metal GPU platform and is optimized for advanced workloads, including large language model (LLM) training. In ISC 2025 TOP500, SAKURAONE ...

---

### 18. [Outperforming Self-Attention Mechanisms in Solar Irradiance Forecasting via Physics-Guided Neural Networks](https://arxiv.org/abs/2604.13455)

**Authors**: Mohammed Ezzaldin Babiker Abdullah, Rufaidah Abdallah Ibrahim Mohammed  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.13455v1  

#### Abstract
Accurate Global Horizontal Irradiance (GHI) forecasting is critical for grid stability, particularly in arid regions characterized by rapid aerosol fluctuations. While recent trends favor computationally expensive Transformer-based architectures, this paper challenges the prevailing "complexity-firs...

---

### 19. [Memory as Metabolism: A Design for Companion Knowledge Systems](https://arxiv.org/abs/2604.12034)

**Authors**: Stefan Miteski  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12034v1  

#### Abstract
Retrieval-Augmented Generation remains the dominant pattern for giving LLMs persistent memory, but a visible cluster of personal wiki-style memory architectures emerged in April 2026 -- design proposals from Karpathy, MemPalace, and LLM Wiki v2 that compile knowledge into an interlinked artifact for...

---

### 20. [RPRA: Predicting an LLM-Judge for Efficient but Performant Inference](https://arxiv.org/abs/2604.12634)

**Authors**: Dylan R. Ashley, Ga\"el Le Lan, Changsheng Zhao, Naina Dhingra, Zhipeng Cai, Ernie Chang, Mingchen Zhuge, Yangyang Shi, Vikas Chandra, J\"urgen Schmidhuber  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12634v1  

#### Abstract
Large language models (LLMs) face a fundamental trade-off between computational efficiency (e.g., number of parameters) and output quality, especially when deployed on computationally limited devices such as phones or laptops. One way to address this challenge is by following the example of humans a...

---

### 21. [BEAM: Bi-level Memory-adaptive Algorithmic Evolution for LLM-Powered Heuristic Design](https://arxiv.org/abs/2604.12898)

**Authors**: Chuyang Xiang, Yichen Wei, Jiale Ma, Handing Wang, Junchi Yan  
**Category**: cs.AI  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12898v1  

#### Abstract
Large Language Model-based Hyper Heuristic (LHH) has recently emerged as an efficient way for automatic heuristic design. However, most existing LHHs just perform well in optimizing a single function within a pre-defined solver. Their single-layer evolution makes them not effective enough to write a...

---

### 22. [EVE: A Domain-Specific LLM Framework for Earth Intelligence](https://arxiv.org/abs/2604.13071)

**Authors**: \`Alex R. Atrio, Antonio Lopez, Jino Rohit, Yassine El Ouahidi, Marcello Politi, Vijayasri Iyer, Umar Jamil, S\'ebastien Brati\`eres, Nicolas Long\'ep\'e  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.13071v1  

#### Abstract
We introduce Earth Virtual Expert (EVE), the first open-source, end-to-end initiative for developing and deploying domain-specialized LLMs for Earth Intelligence. At its core is EVE-Instruct, a domain-adapted 24B model built on Mistral Small 3.2 and optimized for reasoning and question answering. On...

---

### 23. [Hessian-Enhanced Token Attribution (HETA): Interpreting Autoregressive LLMs](https://arxiv.org/abs/2604.13258)

**Authors**: Vishal Pramanik, Maisha Maliha, Nathaniel D. Bastian, Sumit Kumar Jha  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.13258v1  

#### Abstract
Attribution methods seek to explain language model predictions by quantifying the contribution of input tokens to generated outputs. However, most existing techniques are designed for encoder-based architectures and rely on linear approximations that fail to capture the causal and semantic complexit...

---

### 24. [YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference](https://arxiv.org/abs/2604.13556)

**Authors**: You Wu, Ziheng Chen, Yizhen Zhang, Haoyi Wu, Chengting Yu, Yuchi Xu, Wenbo Su, Bo Zheng, Kewei Tu  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.13556v1  

#### Abstract
Cross-layer key-value (KV) compression has been found to be effective in efficient inference of large language models (LLMs). Although they reduce the memory consumption of the KV cache, such methods usually introduce non-negligible performance degradation. In this work, we aim to enhance the perfor...

---

### 25. [MM-Doc-R1: Training Agents for Long Document Visual Question Answering through Multi-turn Reinforcement Learning](https://arxiv.org/abs/2604.13579)

**Authors**: Jiahang Lin, Kai Hu, Binghai Wang, Yuhao Zhou, Zhiheng Xi, Honglin Guo, Shichun Liu, Junzhe Wang, Shihan Dou, Enyu Zhou, Hang Yan, Zhenhua Han, Tao Gui, Qi Zhang, Xuanjing Huang  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.13579v1  

#### Abstract
Conventional Retrieval-Augmented Generation (RAG) systems often struggle with complex multi-hop queries over long documents due to their single-pass retrieval. We introduce MM-Doc-R1, a novel framework that employs an agentic, vision-aware workflow to address long document visual question answering ...

---

### 26. [MedRCube: A Multidimensional Framework for Fine-Grained and In-Depth Evaluation of MLLMs in Medical Imaging](https://arxiv.org/abs/2604.13756)

**Authors**: Zhijie Bao, Fangke Chen, Licheng Bao, Chenhui Zhang, Wei Chen, Jiajie Peng, Zhongyu Wei  
**Category**: cs.CL  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.13756v1  

#### Abstract
The potential of Multimodal Large Language Models (MLLMs) in domain of medical imaging raise the demands of systematic and rigorous evaluation frameworks that are aligned with the real-world medical imaging practice. Existing practices that report single or coarse-grained metrics are lack the granul...

---

### 27. [Multi-Task LLM with LoRA Fine-Tuning for Automated Cancer Staging and Biomarker Extraction](https://arxiv.org/abs/2604.13328)

**Authors**: Jiahao Shao, Anam Nawaz Khan, Christopher Brett, Tom Berg, Xueping Li, Bing Yao  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.13328v1  

#### Abstract
Pathology reports serve as the definitive record for breast cancer staging, yet their unstructured format impedes large-scale data curation. While Large Language Models (LLMs) offer semantic reasoning, their deployment is often limited by high computational costs and hallucination risks. This study ...

---

### 28. [Asymmetric-Loss-Guided Hybrid CNN-BiLSTM-Attention Model for Industrial RUL Prediction with Interpretable Failure Heatmaps](https://arxiv.org/abs/2604.13459)

**Authors**: Mohammed Ezzaldin Babiker Abdullah  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.13459v1  

#### Abstract
Turbofan engine degradation under sustained operational stress necessitates robust prognostic systems capable of accurately estimating the Remaining Useful Life (RUL) of critical components. Existing deep learning approaches frequently fail to simultaneously capture multi-sensor spatial correlations...

---

### 29. [Parameter-efficient Quantum Multi-task Learning](https://arxiv.org/abs/2604.13560)

**Authors**: Hevish Cowlessur, Chandra Thapa, Tansu Alpcan, Seyit Camtepe  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.13560v1  

#### Abstract
Multi-task learning (MTL) improves generalization and data efficiency by jointly learning related tasks through shared representations. In the widely used hard-parameter-sharing setting, a shared backbone is combined with task-specific prediction heads. However, task-specific parameters can grow rap...

---

### 30. [Optimization with SpotOptim](https://arxiv.org/abs/2604.13672)

**Authors**: Thomas Bartz-Beielstein  
**Category**: cs.LG  
**Published**: 2026-04-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.13672v1  

#### Abstract
The `spotoptim` package implements surrogate-model-based optimization of expensive black-box functions in Python. Building on two decades of Sequential Parameter Optimization (SPO) methodology, it provides a Kriging-based optimization loop with Expected Improvement, support for continuous, integer, ...

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
