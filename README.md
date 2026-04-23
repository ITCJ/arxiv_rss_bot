# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-23 07:20:49 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Decoding Text Spans for Efficient and Accurate Named-Entity Recognition](https://arxiv.org/abs/2604.20447)

**Authors**: Andrea Maracani, Savas Ozkan, Junyi Zhu, Sinan Mutlu, Mete Ozay  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2604.20447v1  

#### Abstract
Named Entity Recognition (NER) is a key component in industrial information extraction pipelines, where systems must satisfy strict latency and throughput constraints in addition to strong accuracy. State-of-the-art NER accuracy is often achieved by span-based frameworks, which construct span repres...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Decoding Text Spans for Efficient and Accurate Named-Entity Recognition*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **span-based Named Entity Recognition (NER)** 方法（如 PL-Marker）虽然在准确性上表现优异，但由于其需要枚举大量候选 span 并通过 marker-augmented 输入进行处理，导致推理阶段计算开销巨大，尤其是在长文本和高吞吐场景下难以部署。这限制了其在工业级应用中的可扩展性和效率。

此外，基于 LLM 的生成式 NER 方法（如 InstructUIE、GPT-NER）虽然具备强泛化能力，但自回归解码机制带来极高的延迟和资源消耗，不适合大规模实时服务。

### 提出的新方法与创新思路
本文提出 **SpanDec** 和 **SF-SpanDec**，一种高效且准确的 span-based NER 框架，核心思想是：

- **Decoupled Span Processing (SpanDec)**  
  将 span 表示的处理从主 encoder 中解耦出来，引入一个轻量级的 **decoder** 专门负责 span 分类任务。原始 token 序列由 encoder 正常编码一次，而 span marker 只在 decoder 阶段参与 cross-attention，避免了在 encoder 所有层中重复传播 marker 向量带来的冗余计算。

- **Early Span Filtering (SF-SpanDec)**  
  引入一个轻量级的二分类器（O-classifier），在 decoder 前预测哪些 token 不属于任何实体（即 "O" 类）。基于此过滤掉大量不可能构成有效实体的候选 span，显著减少需处理的 span 数量。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **效率提升** | 推理 throughput 最高提升至 **2.7×**，GFLOPs 最多降低 **8.2×**，远优于 PL-Marker |
| **精度保持甚至超越** | 在多个基准上 F1 超过 token classification +1.8%，与 PL-Marker 相当或略优 |
| **部署友好** | 更适合高并发、低延迟、边缘设备等资源受限场景 |
| **架构兼容性强** | 可无缝集成到现有 span-based 框架中，参数总量可控（用一层 encoder 换 decoder） |

---

## 2. 核心实验方法和设置

### 使用的数据集
共采用四个广泛使用的 NER benchmark 数据集：

| 数据集 | 描述 | 实体类别数 | 训练样本 | 测试样本 |
|--------|------|------------|-----------|-----------|
| **CoNLL++** | CoNLL03 的修正版，英文新闻文本 | 5 | 4.5k | 817 |
| **CrossNER** | 跨领域数据集（如 AI、literature 等） | 40 | 20k | 2.5k |
| **OntoNotes5** | 多类型文本（新闻、广播、网络等） | 19 | 28k | 3.2k |
| **BC5CDR** | PubMed 医学文献，标注化学物与疾病 | 3 | 11k | 5.9k |

> 详细统计见原文 Table 2。

### 实验设置
- **模型架构**：基于三种主流 encoder 进行实验：
  - MiniLM (33M)
  - BERT-Base (110M)
  - RoBERTa-Large (355M)
- **SpanDec 构造方式**：移除原 encoder 最后一层，在相同参数预算下添加一个单层 decoder。
- **最大 span 长度**：设为 8 tokens（遵循 SpanMarker 设置）
- **训练配置**：
  - 使用 PyTorch + HuggingFace Transformers
  - AdamW 优化器，学习率 5e-5，OneCycle scheduler
  - 混合精度训练，batch size 64
- **硬件平台**：NVIDIA A40 GPU（单卡测试 throughput）

### 评估指标
| 指标 | 说明 |
|------|------|
| **F1 Score** | 使用 `seqeval` 工具包计算标准 NER F1 |
| **Throughput** | 每秒处理的样本数（samples/s），反映推理速度 |
| **GFLOPs** | 总浮点运算量，衡量计算复杂度 |
| **Latency** | 单样本推理延迟（图1中展示） |

### 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Token Classification** | 标准 baseline | BIO tagging + linear head，高效但边界识别弱 |
| **PL-Marker** | 当前最优 span-based 方法 | marker 注入 encoder 输入，性能好但计算重 |
| **SplitNER, InstructUIE, GLiNER, GPT-NER 等** | SOTA 对比 | 包括两阶段 QA 方法、LLM prompting 方法等 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 3 & 5）

#### SpanDec vs. PL-Marker vs. Token Classification（MiniLM 示例）
| 方法 | Throughput (×token) | GFLOPs (×token) | Avg F1 |
|------|---------------------|------------------|--------|
| Token Classif. | 1.00× | 1.0× | 84.7 |
| PL-Marker | 0.42× | 8.3× | 85.4 |
| **SpanDec (ours)** | **0.73×** | **1.5×** | **86.2** |

> 结论：SpanDec 在仅增加 50% 计算成本的情况下，将 throughput 提升近一倍于 PL-Marker，并实现更高的 F1。

#### SF-SpanDec 进一步优化（BERT-Base 示例）
| 方法 | Throughput (×token) | GFLOPs (×token) | Avg F1 |
|------|---------------------|------------------|--------|
| Token Classif. | 1.00× | 1.0× | 86.1 |
| PL-Marker | 0.34× | 8.3× | 87.6 |
| SpanDec | 0.61× | 1.5× | 87.8 |
| **SF-SpanDec (ours)** | **0.93×** | **1.01×** | **87.5** |

> 结论：SF-SpanDec 几乎达到 token classification 的效率水平，同时保留了 span-based 方法的高精度优势。

### 与其他 SOTA 方法对比（Table 4）
| 方法 | 参数量 | Throughput (相对) | BC5CDR F1 | CoNLL++ F1 | OntoNotes5 F1 |
|------|--------|--------------------|-----------|------------|---------------|
| InstructUIE (13B) | 13B | <0.05× | 89.0 | 91.5 | 88.6 |
| GPT-NER (175B) | 175B | <0.04× | — | 90.9 | 82.2 |
| **SpanDec (RoBERTa-L)** | **0.35B** | **1.0×** | **90.8** | **95.4** | **91.3** |

> 结论：SpanDec 以不到 1% 的参数量和数量级更高的 throughput，实现了更优或相当的性能。

### 消融实验结果（Table 8, Appendix C）
研究了不同 encoder/decoder 层数组合的影响（MiniLM + CoNLL++）：

| Encoder 层数 | Decoder 层数 | GFLOPs | F1 |
|-------------|--------------|--------|----|
| 11 | 1 | 2.9 | 93.6 |
| 10 | 1 | 2.8 | 93.4 |
| 9 | 1 | 2.6 | 93.2 |
| ... | ... | ... | ↓ |
| 7 | 1 | 2.3 | 92.3 |
| 10 | 2 | 3.9 | 93.4 |
| 7 | 5 | 7.0 | 92.4 |

> 发现：
> - 减少 encoder 层会轻微降低 F1，但显著减少计算；
> - 增加 decoder 层数无法带来明显收益，表明 **单层 decoder 已足够强大**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **解耦式 span 处理是高效的**：将 span marker 的交互推迟到 decoder 阶段，能极大减少冗余计算，无需牺牲精度。
2. ✅ **轻量级 span filtering 极具性价比**：通过简单的 token-level O-classifier 可提前剔除约 85% 的无效 span，几乎不增加开销却大幅提升效率。
3. ✅ **SpanDec 实现了最佳 accuracy-efficiency trade-off**：在保持甚至超过 PL-Marker 精度的同时，推理速度接近 token classification，适用于工业部署。
4. ✅ **单层 decoder 足够胜任 span 分类任务**：复杂的深层 decoder 并无必要，简单结构即可捕获 span 上下文信息。

### 方法的局限性
- **未对超参数进行全面调优**：作者指出当前结果基于标准配置，仍有进一步优化空间。
- **软件实现尚有改进余地**：尽管理论 GFLOPs 与 token classification 接近，实际 throughput 达到 92–93%，暗示底层实现（如 kernel 优化）可进一步提升。
- **依赖预定义的最大 span 长度**（默认 8）：可能影响极长实体的识别能力。

### 未来工作方向
- 探索动态 span length 或 hierarchical filtering 机制；
- 在更多 IE 任务（如 Relation Extraction）中验证该框架的通用性；
- 针对特定硬件（如移动端 NPU）进行定制化部署优化；
- 结合 LLM 进行 weak supervision 或 prompt-driven 初始化，增强 zero-shot 能力。

--- 

> 📌 **一句话总结**：  
> SpanDec 通过 **decoupling span processing** 和 **early span filtering**，在不损失 span-based 方法建模优势的前提下，大幅提升了 NER 系统的推理效率，为高性能工业级 NER 部署提供了新的实用范式。

</details>

---

### 2. [Super Apriel: One Checkpoint, Many Speeds](https://arxiv.org/abs/2604.19877)

**Authors**: SLAM Labs,  :, Oleksiy Ostapenko, Raymond Li, Torsten Scholak, Alireza Mousavi-Hosseini, Aman Tiwari, Denis Kocetkov, Joel Lamy Poirier, Kelechi Ogueji, Nanda H Krishna, Rafael Pardinas, Sathwik Tejaswi Madhusudhan, Shruthan Radhakrishna, Srinivas Sunkara, Valerie Becaert  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.19877v1  

#### Abstract
We release Super Apriel, a 15B-parameter supernet in which every decoder layer provides four trained mixer choices -- Full Attention (FA), Sliding Window Attention (SWA), Kimi Delta Attention (KDA), and Gated DeltaNet (GDN). A placement selects one mixer per layer; placements can be switched between...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Super Apriel: One Checkpoint, Many Speeds》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前大语言模型在长上下文场景下，**Full Attention (FA)** 的推理成本成为瓶颈，尤其是在自回归解码时，KV Cache 随序列长度线性增长，导致内存占用高、吞吐量低。传统混合架构（hybrid architectures）虽然通过部分替换为高效 Token Mixer（如 SWA、GDN）来提升速度，但其 **placement（每层使用哪种 mixer）是固定的**，只能提供单一的速度-质量权衡点。

这带来了以下限制：
- **无法适应多样化的工作负载**（如短提示高并发 vs 长上下文生成）
- **缺乏运行时灵活性**，无法在高峰/低峰时段动态切换策略
- **任务敏感的质量退化不均**，例如长距离检索对 FA 更敏感，而局部推理则不然

### **提出的新方法与创新思路**
Super Apriel 提出了一种 **Token Mixer Supernet** 架构，核心创新如下：

- **单个 Checkpoint 支持多种运行速度**：每个 Decoder 层都内置四种训练好的 Token Mixer 选项：
  - **Full Attention (FA)**
  - **Sliding Window Attention (SWA)**
  - **Kimi Delta Attention (KDA)**
  - **Gated DeltaNet (GDN)**
- **运行时可切换 Placement**：无需重新加载权重，可在请求间动态选择不同的 mixer 组合（即 placement），实现从“教师级质量”到“近 11× 解码速度”的多个预设档位。
- **共享 Checkpoint 支持 Speculative Decoding**：可以直接用高效的混合 placement 作为 draft model，全 FA 作为 target model，无需额外训练 draft 模型。
- **基于 Surrogate 的 Placement 优化框架**：使用 **Cluster Expansion** 构建一个可精确优化的代理模型（surrogate），预测任意 placement 的性能，在极小评估开销下扫描整个 $4^{48}$ 的组合空间，找到帕累托最优的 presets。

### **相比现有方法的优势**
| 方面 | 传统方法 | Super Apriel |
|------|--------|-------------|
| **部署灵活性** | 固定 placement，需多模型部署 | 单 Checkpoint 多速度，运行时切换 |
| **训练效率** | 每个 placement 需独立训练或搜索 | 所有 mixer 并行训练一次完成 |
| **搜索能力** | 启发式搜索（如 Beam Search）易陷入局部最优 | 可控误差下进行全局精确优化（Exact Optimization） |
| **下游适配** | 很难针对特定任务微调最优结构 | 支持任务导向的 SFT 进一步提升特定 preset 性能 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
#### **蒸馏阶段（Distillation）**
- **来源**：Apriel 1.6 的预训练语料 + 监督微调数据
- **构成重点偏向高质量推理轨迹**：
  - 数学与 STEM（5%）
  - 编程（10%）
  - 推理链、结构化解题（29.3%）
  - 图像文本（多模态，49%）
- **总 Token 数**：266B（其中 197B 来自 Apriel 1.5，69B 来自 Apriel 1.6）

#### **监督微调阶段（SFT）**
- **数据集**：指令微调数据（SFT Dataset）
- **构成**：
  - 编程（36.1%）
  - 数学与 STEM（38.7%）
  - 聊天与通用推理（11.3%）
  - 工具调用（12.0%）
  - 安全与内容审核（2.0%）
- **总 Token 数**：最多 137B

---

### **实验设置与评估指标**

#### **模型配置**
- **基础模型**：Apriel 1.6（15B 参数，48 层，Grouped-Query Attention）
- **Mixer 类型**：FA、SWA（w=4096）、KDA、GDN
- **训练框架**：Fast-LLM（ServiceNow Research）
- **硬件**：H100 GPU，最多使用 192 张
- **序列长度**：
  - 蒸馏：16,384
  - SFT：32,768

#### **评估指标**
| 指标类别 | 具体指标 |
|--------|---------|
| **性能（Quality）** | MMLU、GSM8K、MATH500、AIME24/25、GPQA、HLE、LCB、IFEval 等 |
| **速度（Throughput）** | 解码吞吐量（tokens/s），相对全 FA 的 speedup（×） |
| **效率-质量权衡** | 帕累托前沿（Pareto Frontier）：speedup vs. 任务平均准确率 |
| **Speculative Decoding** | 净加速比（Net Speedup），接受率（Acceptance Rate） |

#### **基线方法对比**
- **内部基线**：
  - Apriel-H1（15B，混合 Mamba）
- **外部基线**：
  - Qwen-3.5 27B（MoE + GDN）
  - Nemotron-3-Nano 30B（Mamba + MoE）
  - Falcon-H1R 7B（Mamba）
  - Nemotron-Nano 12B v2
  - OLMo-Hybrid-Think 7B

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| Preset | FA/SWA/KDA/GDN | 平均准确率 | 质量保留率 | @32k 解码 speedup |
|-------|----------------|------------|------------|--------------------|
| all-FA | 48/0/0/0 | **74.2** | 100% | 1.0× |
| RegLklhd-26 | 12/26/6/4 | 71.1 | 96% | **2.9×** |
| RegLklhd-18 | 3/25/4/16 | 69.7 | 94% | **4.8×** |
| RegLklhd-13 | 0/16/13/19 | 60.2 | 81% | **6.9×** |
| RegLklhd-10 | 0/10/5/33 | 57.2 | 77% | **10.7×** |

> ✅ **所有混合 preset 均来自同一 Checkpoint，无需额外训练。**

### **与基线方法的对比结果**
- **质量-速度综合表现最优**：
  - 在 **2.9× speedup** 下，Super Apriel 达到 **71.1** 准确率，显著优于 Apriel-H1（58.4）和 OLMo-Hybrid（56.1）。
  - 在 **6.9× speedup** 下仍保持 **60.2**，远超同类高速模型。
- **最大速度优势**：
  - 最快 preset 达到 **10.7×** 解码速度，适合高吞吐场景。
- **Speculative Decoding 表现优异**：
  - 使用 **all-GDN** 作为 draft，接受率依然很高，净加速可达 **2.75×**（见 Figure 11），验证了 supernet 内部分布一致性好。

### **消融实验结果**
#### **(1) 训练策略对比（0.5B 开发模型）**
- **随机采样（Stochastic）** vs **定向采样（Targeted）**
  - 结果显示：**随机采样最终性能更优**，即使目标 preset 也最终被追上甚至反超。
  - 说明：过早锁定 placement 可能导致“虚假最优”（false optima），因训练信号偏差造成。

#### **(2) 0.5B vs 15B 模型的排名稳定性**
- **0.5B 模型**：placement 排名早期即稳定（Spearman ρ > 0.98）。
- **15B 模型**：尽管总体稳定，但 **帕累托前沿上的 placement 排名波动更大**，尤其在中等成本区域。
- **结论**：小模型上的结论不能直接外推至大模型，必须在目标规模上验证。

#### **(3) SFT 对性能的影响**
- 所有 placement 在 SFT 后均有显著提升：
  - 数学任务提升最大（如 GSM8K +11.6 pts）
  - 即使全 FA 也能从 80.7 → 92.3
- 表明：**SFT 是恢复复杂推理能力的关键步骤**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **单 Checkpoint 多速度是可行且高效的**：通过 supernet 设计，实现了运行时灵活切换，极大简化部署流程。
2. ✅ **Cluster Expansion Surrogate 可实现高效全局搜索**：相比启发式方法，能系统性地探索组合空间，找到真正帕累托最优的 presets。
3. ✅ **Throughput 优势随上下文长度放大**：
   - Super Apriel 的高效 placements 在 **16K → 32K** 上获得 **80–155%** 的相对速度增益。
   - 外部基线仅获 **5–46%** 增益，凸显其架构优势。
4. ❗ **大模型的 placement 排名更具动态性**：前沿 placement 在训练过程中可能漂移，**不应依赖小模型外推**。
5. ✅ **Speculative Decoding 天然兼容**：共享参数确保 draft 和 target 分布接近，无需额外训练 draft 模型。

### **局限性**
| 限制 | 说明 |
|------|------|
| **长距离任务严重退化** | 如 RULER、NIAH 在大量使用 GDN/KDA 时性能骤降（见 Table 12） |
| **代理模型假设短程交互** | Cluster Expansion 截断于短程，若存在长程依赖可能欠拟合 |
| **Log-likelihood 仅为代理指标** | 与 Exact Match 存在潜在差距，尤其在极端 placements 上 |
| **线性成本模型不完美** | 对“少数派 mixer”（singleton）拟合差，需过滤处理 |
| **小模型结论不可靠外推** | 0.5B 上的训练策略结论未必适用于 15B |

### **未来工作方向**
1. **强化学习后训练（RL）**：
   - 使用 Group Relative Policy Optimization（GRPO）进行推理与智能体任务优化。
   - 可引入 KL 正则项以稳定 FA 作为参考。
2. **扩展至其他教师模型与 Mixer 类型**：
   - 验证方法通用性。
   - 尝试 Mamba-2、Lightning Attention 等新型 mixer。
3. **生产级训练策略验证**：
   - 在 15B 规模上对比 **Stochastic vs Targeted** placement 采样，控制变量。
4. **动态路由机制研究**：
   - 实现 per-request 自动选择最优 placement。
   - 挑战：如何定义“最优”？如何训练路由策略？
5. **部署优化**：
   - **Model Thinning**：只保留上线 presets 所需的 mixer 类型。
   - **Memory-aware Preset Selection**：选择 mixer 重叠度高的 presets 以减少显存占用。

---

> 📌 **一句话总结**：  
> **Super Apriel 通过构建一个支持四种 Token Mixer 的 supernet，实现了“一个 Checkpoint，多种运行速度”，结合 surrogate 优化与运行时切换，为 LLM 部署提供了前所未有的灵活性与效率。**

</details>

---

### 3. [Fast Bayesian equipment condition monitoring via simulation based inference: applications to heat exchanger health](https://arxiv.org/abs/2604.20735)

**Authors**: Peter Collett, Alexander Johannes Stasik, Simone Casolo, Signe Riemer-S{\o}rensen  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.20735v1  

#### Abstract
Accurate condition monitoring of industrial equipment requires inferring latent degradation parameters from indirect sensor measurements under uncertainty. While traditional Bayesian methods like Markov Chain Monte Carlo (MCMC) provide rigorous uncertainty quantification, their heavy computational b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast Bayesian equipment condition monitoring via simulation based inference: applications to heat exchanger health

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
工业设备（如**heat exchanger**）的健康状态监测通常依赖于对不可观测的**latent degradation parameters**（如结垢、泄漏）进行推断。传统**Bayesian inference**方法（如**MCMC**）虽能提供严格的不确定性量化，但由于需要反复调用物理仿真模型，计算开销巨大，难以满足实时监控需求。

### ✅ 提出的新方法与创新
本文提出一种基于**Simulation-Based Inference (SBI)** 的AI驱动框架，结合**amortized neural posterior estimation**，实现快速贝叶斯设备状态监测。具体创新包括：

- **无需显式似然函数**：利用前向仿真生成训练数据，通过神经网络学习从观测数据到后验分布的映射，实现**likelihood-free inference**。
- **amortized inference**：在一次离线训练后，后续推理近乎瞬时完成，适用于高频、多资产的实时诊断。
- **应用于复杂故障模式**：针对**heat exchanger**中的**fouling**（结垢）和**leakage**（泄漏）两类典型故障，构建随机退化模型，并实现联合故障识别与参数估计。

### ✅ 相比现有方法的优势
| 维度 | 传统 MCMC | 本文 SBI 方法 |
|------|-----------|----------------|
| 推理速度 | 每次需数千次仿真，耗时数秒至分钟 | 训练后单次推理仅需 **0.029s** |
| 可扩展性 | 难以并行，不适用于多资产 | 支持大规模、实时部署 |
| 不确定性量化 | 准确但代价高 | 保持可靠不确定性估计 |
| 模型兼容性 | 要求可微或解析似然 | 支持“black-box”仿真器 |

> 🔑 **核心优势**：**在保持与MCMC相当诊断精度的同时，将推理速度提升82倍**，为数字孪生和实时PHM系统提供了可行路径。

---

## 2. 核心实验方法和设置

### 📊 数据集
- **合成数据集**：基于一个**stochastic heat exchanger model**生成，包含六类工况（5种故障 + 1种正常）：
  1. **Weak Fouling**（弱结垢）
  2. **Batch Process Shutdown (SD)**（批处理停机导致的突发结垢）
  3. **Boiler Feedwater (FW)**（锅炉给水系统强结垢）
  4. **Mild Leak**（轻微泄漏）
  5. **Severe Leak**（严重泄漏）
  6. **No Failure**（无故障，基线）

- 每种场景生成 **500组带噪声的观测序列**，共 **3,000条时间序列数据**。
- 数据来源：通过**effectiveness-NTU model**模拟温度、流量等传感器读数。

### ⚙️ 实验设置
- **SBI 方法**：
  - 使用 **Sequential Neural Posterior Estimation (SNPE)** 框架。
  - 神经网络架构：**Neural Spline Flow (NSF)**，具备高表达能力。
  - 输入：**25维 summary statistics**（包括温差均值、标准差、趋势等）。
  - 训练：**50,000次前向仿真**用于离线训练。
- **MCMC 基线**：
  - 使用 **NumPyro** 实现，采用 **NUTS sampler**。
  - 每次推理运行4条链，每条含2,000预热步 + 3,000采样步，总计约 **20,000次仿真调用**。

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Failure-mode identification accuracy** | 分类准确率（正确识别故障模式的比例） |
| **Wasserstein distance** | 衡量SBI与MCMC后验分布之间的相似性（越小越好） |
| **CRPS (Continuous Ranked Probability Score)** | 评估概率预测的准确性与锐度（越低越好） |
| **Credible interval coverage** | 后验置信区间是否覆盖真实参数 |
| **Inference time** | 单次推理耗时（核心效率指标） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ 故障识别准确率（Table II）
| 场景 | MCMC 准确率 | SBI 准确率 |
|------|-------------|-----------|
| Weak Fouling | 100% | 100% |
| Batch SD | 100% | 100% |
| Boiler FW | 100% | 100% |
| Mild Leak | 99.8% | 100% |
| Severe Leak | 99.6% | 100% |
| No Failure | 98.2% | 98.6% |

> 💡 **结论**：SBI 在所有场景下均达到与 MCMC 相当甚至更优的分类性能。

#### ✅ 参数估计一致性（Figure 5 & 6）
- **后验中位数高度一致**：SBI 与 MCMC 在关键参数（如 `T`（故障起始时间）、`βf`（结垢强度）、`β`（泄漏速率）、`λ`（事件频率））上的估计几乎重合。
- **Wasserstein距离低**（Fig. 7）：大多数参数的距离集中在低位，表明后验形状高度相似。
- **CRPS相近**（Fig. 8）：说明 SBI 的概率预测质量与 MCMC 相当。

#### ✅ 推理效率对比（Table III & Fig. 10）
| 方法 | 单次推理时间 | 总仿真次数 | 成本回收点 | 加速比 |
|------|---------------|------------|------------|--------|
| MCMC | 2.4 s/call | 900/call | — | 1× |
| SBI | **0.029 s/call** | 5,000（一次性） | ~6次调用后 | **82×** |

> ⚡ **突破性结果**：尽管SBI有较高训练成本，但在**6次推理后即实现成本逆转**，之后每次诊断快82倍。

### 🔍 特殊案例分析：Sparse-event regime（Scenario 2）
- **挑战**：`λ=0.5`（极低事件频率）+ `βf=0.03`（大跳跃），观测窗口内可能仅发生少数几次结垢事件。
- **发现**：
  - MCMC 和 SBI 均倾向于高估 `λ`，因其先验中心为2.0，而真实值为0.5。
  - 这是**结构性不可识别性**（structural identifiability）问题，而非算法缺陷。
  - 尽管参数估计受限，**故障模式仍被正确识别**，且**退化趋势预测合理**。

> ✅ **重要启示**：即使个别参数难以精确恢复，整体故障诊断仍可靠，这对实际PHM系统至关重要。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **SBI 可有效替代 MCMC**：在 heat exchanger 故障诊断任务中，SBI 实现了与 MCMC 相当的诊断精度和不确定性量化能力。
2. **推理速度显著提升**：**加速达82倍**，使贝叶斯推理首次具备在工业级系统中实现实时应用的可能性。
3. **amortization 是关键**：通过离线训练将计算成本摊销，适合需频繁诊断的场景。
4. **robust fault identification**：即使在稀疏事件、弱可观测条件下，也能稳定识别故障模式。
5. **适用于 black-box simulator**：无需访问似然函数或梯度，兼容现有工程仿真工具。

### ⚠️ 局限性
1. **参数可识别性受限**：在稀疏事件或强噪声下，某些参数（如 `λ`）存在结构性模糊，易受先验影响。
2. **依赖合成数据**：当前验证基于理想化模型，尚未在真实工业数据上测试。
3. **summary statistics 设计**：手工设计的特征可能丢失部分时间序列信息；端到端学习更具潜力但更复杂。
4. **训练成本高**：初始需大量仿真（5万次），对复杂系统可能成为瓶颈。

### 🔮 未来工作方向
1. **真实数据验证**：在实际 heat exchanger 或其他工业设备上部署并验证性能。
2. **在线/自适应训练**：引入增量学习机制，应对工况漂移（distributional shift）。
3. **joint latent trajectory inference**：尝试直接推断完整的退化轨迹 `z(t)`，而不仅是参数。
4. **集成至数字孪生平台**：将该框架嵌入工业级 **digital twin** 系统，支持全厂级实时PHM。
5. **与其他AI方法融合**：探索与 **PINNs** 或 **surrogate modeling** 结合，进一步降低仿真负担。

---

## ✅ 总结

本文成功将前沿的 **Simulation-Based Inference (SBI)** 技术引入工业设备健康监测领域，提出了一种**高效、可扩展、不确定性感知**的贝叶斯诊断框架。实验表明，该方法在保持与传统 **MCMC** 相当诊断性能的同时，实现了**82倍的速度提升**，解决了传统贝叶斯方法在实时性上的根本瓶颈。这一成果为在复杂工业系统中实现**实时 probabilistic fault diagnosis** 和 **digital twin** 应用开辟了新路径，具有重要的工程价值和推广前景。

</details>

---

### 4. [Dual-Cluster Memory Agent: Resolving Multi-Paradigm Ambiguity in Optimization Problem Solving](https://arxiv.org/abs/2604.20183)

**Authors**: Xinyu Zhang, Yuchen Wan, Boxuan Zhang, Zesheng Yang, Lingling Zhang, Bifan Wei, Jun Liu  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.20183v1  

#### Abstract
Large Language Models (LLMs) often struggle with structural ambiguity in optimization problems, where a single problem admits multiple related but conflicting modeling paradigms, hindering effective solution generation. To address this, we propose Dual-Cluster Memory Agent (DCM-Agent) to enhance per...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Dual-Cluster Memory Agent: Resolving Multi-Paradigm Ambiguity in Optimization Problem Solving

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（**LLMs**）在解决优化问题时面临**结构歧义（structural ambiguity）**的问题。一个优化问题可能对应多种建模范式（如 **ILP**, **DP**, **CP**），这些范式虽然都相关，但彼此冲突，导致模型产生认知干扰，难以选择合适的建模路径。此外，现有方法（如 fine-tuning 或静态 prompting）缺乏对范式特定错误（如 integrality gap、recurrence failure）的识别与修复能力。

### 提出了什么新方法或新思路
本文提出 **Dual-Cluster Memory Agent (DCM-Agent)**，一种无需训练（training-free）的框架，通过外部化历史经验来增强 LLM 在优化问题求解中的表现。其核心创新包括：

- **Dual-Cluster Memory Construction**  
  将历史解决方案划分为两个独立的集群：
  - **Modeling Cluster**：抽象建模范式（如 ILP、DP）
  - **Coding Cluster**：具体编码实现（如 Gurobipy 模板）
  两者通过加权二分图（bipartite graph）连接，表示建模逻辑与编码策略之间的兼容性。

- **三层次结构化知识提炼**  
  每个集群从实例级节点中提炼出三种通用指导知识：
  - **Approach**：标准解法模板
  - **Checklist**：验证准则（如变量边界、约束一致性）
  - **Pitfall**：常见错误模式（如非线性目标误用 LP 求解器）

- **Memory-Augmented Inference**  
  引入动态的 **generate-verify-repair-backtrack** 推理流程：
  - 利用检索到的知识主动检测并修复错误
  - 当当前路径不可行时，自动回溯至备选路径
  - 实现灵活且鲁棒的多范式导航

### 相比现有方法的优势
| 维度 | DCM-Agent | 现有方法（如 fine-tuning, ReAct） |
|------|-----------|-------------------------------|
| **是否需要训练** | ❌ 否（training-free） | ✅ 是（依赖大量标注数据） |
| **处理范式歧义能力** | ✅ 显式分离建模与编码逻辑 | ❌ 容易混淆不同范式 |
| **错误检测与修复** | ✅ 基于 Checklists 和 Pitfalls 动态纠错 | ❌ 静态提示无法定位深层错误 |
| **可扩展性** | ✅ 支持“知识继承”（大模型构建记忆供小模型使用） | ❌ 性能受限于模型自身规模 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
共使用 **7 个优化基准数据集**，涵盖多种复杂性和应用场景：

| 数据集 | 类型 | 特点 |
|--------|------|------|
| **NL4Opt** | Linear Programming | 自然语言描述的标准 LP 问题 |
| **NLP4LP** | Mixed-Integer Programming | 复杂 MILP 任务 |
| **OptiBench** | High-complexity optimization | 高难度综合测试集 |
| **OptMATH** | Mathematical optimization | 数学建模导向问题 |
| **ComplexLP (from MAMO)** | Complex LP | 结构复杂的线性规划 |
| **IndustryOR** | Real-world industrial problems | 工业界真实案例 |
| **ComplexOR** | Complex OR problems | 多约束组合优化 |

> 💡 记忆库由 **500 个不与上述测试集重叠的历史问题** 构建而成。

### 实验设置和评估指标
- **主干模型**：Qwen3 系列（8B, 30B, 235B）、DeepSeek-V3.2、GPT-5.1
- **评估方式**：端到端准确率（end-to-end solving accuracy）
  - 输入：自然语言问题
  - 输出：可执行代码（Python + Gurobi/PuLP/OR-Tools 等）
  - 成功条件：生成的答案（目标值 + 决策变量）完全匹配真值
- **推理机制**：基于 embedding 的双层检索（instance-level + cluster-level）+ 路径排序（LLMselect）

### 基线方法对比
| 基线方法 | 类型 | 描述 |
|---------|------|------|
| **Baseline (vanilla LLM)** | Prompting-only | 直接生成代码，无辅助机制 |
| **OptiMUS** | Multi-agent | 多智能体协作提升可靠性 |
| **AF-MCTS** | Search-based | 使用蒙特卡洛树搜索逐步构建公式 |
| **OptiTree** | Hierarchical reasoning | 分层分解问题为子问题 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
> 所有结果均为 **平均准确率 (%)**

| 方法 | 平均性能提升 |
|------|-------------|
| Baseline | 27.99% ~ 67.62% （随模型增大） |
| **DCM-Agent** | **49.43% ~ 78.50%** |
| ➜ **相对提升幅度** | **+11% ~ +21%** |

#### 表格摘要（部分代表性结果，单位：%）
| Model | Dataset | Baseline | DCM-Agent | Δ↑ |
|-------|--------|----------|------------|----|
| Qwen3-8B | Avg. | 27.99 | 49.43 | **+21.44** |
| Qwen3-30B | Avg. | 40.52 | 59.61 | **+19.09** |
| Qwen3-235B | Avg. | 57.74 | 71.28 | **+13.54** |
| DeepSeek-V3.2 | Avg. | 57.61 | 71.47 | **+13.86** |
| GPT-5.1 | Avg. | 67.62 | 78.50 | **+10.88** |

> ✅ DCM-Agent 在所有模型尺度上均达到 **SOTA 性能**

### 与基线方法的对比结果
- 在 **Qwen3-235B** 上：
  - 超过 **OptiTree**（原 SOTA）约 **4.6%**
  - 超过 **AF-MCTS** 近 **7%**
- 即使是 **最小的 Qwen3-8B + DCM-Agent**，性能也超过未增强的 **Qwen3-235B Baseline**

### 消融实验结果（Ablation Study）
使用 Qwen3-235B 进行消融分析：

| 设置 | NLP4LP | OptiBench | OptMATH |
|------|--------|-----------|---------|
| **完整 DCM-Agent** | **84.71** | **75.21** | **46.39** |
| 移除 Modeling Cluster | 77.77 | 72.11 | 42.84 |
| 移除 Coding Cluster | 81.08 | 68.60 | 39.84 |

> 🔍 发现：**Modeling Cluster 的作用更为关键**，说明精确的数学建模逻辑比编码细节更能决定求解成败。

### 参数敏感性分析（Parameter Sensitivity）
| 超参数 | 最优值 | 观察现象 |
|--------|--------|----------|
| Retrieval Top-K | K=3 | Bell-shaped 曲线，过高导致上下文冗余 |
| 更新阈值 N（Knowledge Update） | N=5 | 过低易过拟合，过高延迟更新 |
| 规划候选数 M（Path Queue Size） | M=3 | 单调上升，但 M=5 时边际收益下降 |

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **DCM-Agent 显著缓解了多范式歧义问题**  
   通过将建模与编码解耦，并引入结构化知识（Approach/Checklist/Pitfall），有效过滤干扰信号。

2. ✅ **实现了训练免费（training-free）的高性能优化求解**  
   不依赖参数微调，仅利用外部记忆即可大幅提升 LLM 表现。

3. ✅ **揭示了“知识继承（knowledge inheritance）”现象**  
   - 大模型构建的记忆可用于指导小模型
   - 如：GPT-5.1 构建的记忆 + Qwen3-8B 推理 → 性能超越单独使用 GPT-5.1
   - 但存在“容量失配”风险：记忆过于复杂时，小模型难以消化

4. ✅ **推理效率高，优于搜索类方法**  
   - 相比 AF-MCTS（耗时超 200s），DCM-Agent 在 **OptMATH** 上仅需 **73.4s**
   - 实现了精度与效率的最佳平衡

### 方法的局限性
- **初始化延迟较高**：记忆构建阶段需一次性处理大量历史轨迹，存在“沉没成本”
- **依赖高质量历史数据**：若记忆库中存在系统性偏差或错误，会影响泛化能力
- **极端复杂问题仍可能失败**：当所有检索路径均无效时，backtrack 也无法恢复

> 📌 作者强调：该“初始开销”是一次性的，后续推理阶段高度高效，长期使用可摊薄成本。

### 未来工作方向
1. **在线学习机制**：让记忆库能够通过用户交互动态演化，避免周期性重建
2. **跨领域迁移**：探索在调度、物流、金融等垂直领域的适配能力
3. **轻量化部署**：设计更高效的 embedding 与检索机制，支持边缘设备运行
4. **人机协同接口**：允许人类专家介入修正 Pitfall 或补充 Checklists，形成闭环反馈

---

> 🌟 **总体评价**：  
> DCM-Agent 提供了一种新颖且高效的架构，将 LLM 的通用推理能力与结构化领域知识相结合，在不增加训练负担的前提下显著提升了优化问题求解的准确性与鲁棒性，具有良好的可扩展性与实际应用前景。

</details>

---

### 5. [F\textsuperscript{2}LP-AP: Fast \& Flexible Label Propagation with Adaptive Propagation Kernel](https://arxiv.org/abs/2604.20736)

**Authors**: Yutong Shen, Ruizhe Xia, Jingyi Liu, Yinqi Liu  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.20736v1  

#### Abstract
Semi-supervised node classification is a foundational task in graph machine learning, yet state-of-the-art Graph Neural Networks (GNNs) are hindered by significant computational overhead and reliance on strong homophily assumptions. Traditional GNNs require expensive iterative training and multi-lay...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# F²LP-AP: Fast & Flexible Label Propagation with Adaptive Propagation Kernel 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Graph Neural Networks (GNNs)** 在半监督节点分类任务中面临两大瓶颈：
- **计算开销大**：依赖梯度反向传播进行迭代训练，对大规模图效率低下。
- **强同质性假设（Homophily Assumption）**：假设相连节点具有相似标签，导致在异质图（Heterophilous Graphs）上性能严重下降。

同时，现有的无训练方法（如 Label Propagation, LP）虽然高效，但采用**固定传播规则**，缺乏对局部拓扑结构的适应能力，易在密集区域过平滑，在稀疏或异质区域放大噪声。

---

### 🚀 提出的新方法与创新思路
本文提出 **F²LP-AP**（Free-from-training Label Propagation with Adaptive Propagation Kernel），一种**无需训练、计算高效且结构自适应**的节点分类框架。其核心创新包括：

#### （1）**基于 Local Clustering Coefficient (LCC) 的自适应传播核**
- 利用每个节点的 **LCC**（局部聚类系数）作为拓扑感知指标，动态调整传播参数：
  - **传播深度 $K_u$**：高 LCC 节点（稠密社区）使用较浅传播；低 LCC 节点（边界/异质连接）需要更深传播以获取上下文。
  - **跳跃概率 $\alpha_u$**：高同质性区域降低 $\alpha$ 促进特征平滑；高异质风险区提高 $\alpha$ 保留原始特征锚点。
- 映射函数为预定义启发式规则，**无需参数学习**。

> 公式形式：  
> $$
> \alpha_u = f_\alpha(\text{LCC}_u), \quad K_u = \text{round}(g_K(\text{LCC}_u))
> $$

#### （2）**基于 Geometric Median 的鲁棒原型构建**
- 不再使用易受异常值影响的 **arithmetic mean** 构建类别原型。
- 改用 **Geometric Median**（几何中位数），具备高达 50% 的崩溃点（breakdown point），显著增强对噪声和离群点的鲁棒性。
- 使用 Weiszfeld 算法求解，仅需 3–5 次迭代即可收敛。

#### （3）**纯分析式推理流程（Analytical Inference Pipeline）**
整个流程由三个确定性阶段组成：
1. **Robust Prototype Construction**（几何中位数）
2. **Adaptive Feature Propagation**（LCC 驱动的个性化传播）
3. **Analytical Classification**（通过 cosine similarity 匹配原型）

全程**不涉及任何可学习参数或梯度更新**，实现真正的 **training-free** 推理。

---

### 🔍 相比现有方法的优势
| 维度 | 传统 GNNs（如 GCN） | 经典 LP / APPNP | F²LP-AP |
|------|------------------------|------------------|---------|
| 是否需要训练 | 是（耗时） | 否 | 否 ✅ |
| 参数全局统一？ | 是（缺乏灵活性） | 是 | 否 ✅（node-wise 自适应） |
| 对异质图支持 | 差 ❌ | 一般 | 强 ✅ |
| 抗噪能力 | 中等 | 弱 | 强 ✅（几何中位数 + 自适应控制） |
| 推理速度 | 慢（含训练时间） | 快 | 极快 ✅（<1% GCN 时间） |

> ✅ **优势总结**：F²LP-AP 在保持极低计算成本的同时，实现了媲美甚至超越有监督 GNN 的精度，并能灵活适应从强同质到强异质的各种图结构。

---

## 2. 核心实验方法和设置

### 📚 数据集
共选用 **8 个基准图数据集**，按同质性分为两类：
- **强同质图（Homophilous）**：
  - Cora, CiteSeer, PubMed（引用网络）
- **异质图（Heterophilous）**：
  - Texas, Wisconsin, Cornell（WebKB 页面）
  - Chameleon, Squirrel（Wikipedia 页面）

覆盖不同规模（数百至数万节点）、领域和同质比（0.23 ~ 0.85），全面测试泛化能力。

---

### ⚙️ 实验设置与评估指标

| 项目 | 设置说明 |
|------|----------|
| **评估任务** | 半监督节点分类（semi-supervised node classification） |
| **划分方式** | 标准训练/验证/测试划分（如 Cora: 140/500/1000） |
| **评估指标** | - **Accuracy (Acc.)**<br>- **Macro-F1 Score (F1)**<br>- **Execution Time (秒)**（衡量效率） |
| **实现环境** | PyTorch，固定随机种子（seed=0），确保可复现性 |
| **F²LP-AP 参数范围** | - $K \in [2, 15]$<br>- $\alpha \in [0.05, 0.2]$，由 LCC 动态决定 |

---

### 🆚 基线方法对比
涵盖三类代表性方法：

#### （1）**原型学习变体（Ablation Baselines）**
- `PrototypeOnly-Mean`：仅用均值原型 + 无传播
- `PrototypeOnly-GeoMed`：仅用几何中位数原型
- `FixedAPPNP-Proto`：固定参数 APPNP + 原型匹配

#### （2）**经典与 SOTA 模型**
- `GCN*`：有监督 GNN（带训练）
- `LabelProp`：经典标签传播
- `kNN@5`：基于特征最近邻
- `CoHOp`：最新无训练 SOTA 方法

> 所有基线均采用原文推荐超参，在相同硬件下运行，保证公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Dataset (H) | Method | Acc. | F1 | Time (s) |
|------------|--------|------|-----|---------|
| **Cora (0.85)** | GCN* | 0.821 | 0.809 | 1.350 |
|             | F²LP-AP | **0.835** | **0.821** | **0.056** |
| **CiteSeer (0.81)** | GCN* | 0.719 | 0.691 | 1.210 |
|                   | F²LP-AP | **0.708** | **0.685** | **0.092** |
| **PubMed (0.84)** | GCN* | 0.798 | 0.794 | 1.485 |
|                  | F²LP-AP | **0.782** | **0.779** | **0.044** |
| **Texas (0.31)** | GCN* | 0.553 | 0.365 | 1.015 |
|                | F²LP-AP | **0.842** | **0.787** | **0.016** |
| **Wisconsin (0.37)** | GCN* | 0.608 | 0.269 | 1.053 |
|                      | F²LP-AP | **0.825** | **0.589** | **0.024** |
| **Cornell (0.34)** | GCN* | 0.500 | 0.203 | 1.014 |
|                   | F²LP-AP | **0.763** | **0.519** | **0.018** |

> ✅ **亮点总结**：
- 在 **所有异质图上达到 SOTA 性能**（Texas/Wisconsin/Cornell），远超 GCN 和其他无训练方法。
- 在 **Cora 上超越有监督 GCN**，成为整体最佳表现者。
- 推理时间普遍在 **0.01~0.09 秒之间**，约为 GCN 的 **3%~7%**，效率提升一个数量级以上。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）**自适应传播机制的有效性**
- 对比 `FixedAPPNP`（固定 K=5, α=0.1）：
  - Cora 上准确率提升 **26.9%**（0.658 → 0.835）
  - Wisconsin 提升 **13.8%**
- 表明：**动态参数调整对性能至关重要**，尤其在复杂拓扑中。

#### （2）**几何中位数 vs 算术平均**
- 如 Fig. 3 所示，`GeoMedian` 在全部 8 个数据集上均优于或等于 `Mean`。
- 在低同质图（Chameleon, Squirrel）上增益更明显（+12.8%, +14.3%）。
- 结论：**几何中位数是更鲁棒的原型估计策略**。

#### （3）**效率-性能权衡**
- 尽管引入动态参数，F²LP-AP 运行时间仍极短（仅比 FixedAPPNP 多约 0.01s）。
- 实现了“高性能 + 高效率”的理想平衡。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **无训练 ≠ 低性能**：F²LP-AP 证明了无需梯度优化也能达到甚至超越有监督 GNN 的性能。
2. **局部拓扑指导传播更有效**：利用 **LCC** 实现 node-wise 自适应传播，显著提升了模型在异质图上的鲁棒性和表达力。
3. **原型构建方式极大影响性能**：**Geometric Median** 比传统算术平均更具抗噪能力，尤其适用于真实世界中的噪声图数据。
4. **t-SNE 可视化验证表示质量提升**（Fig. 4）：
   - 原始特征存在严重重叠；
   - F²LP-AP 处理后，类内更紧凑，类间分离更清晰，表明其学习到了更具判别性的表示。

---

### ⚠️ 方法的局限性
1. **依赖单一拓扑指标 LCC**：
   - 在极端稀疏或高度噪声的图中，LCC 可能不稳定，影响参数映射精度。
2. **启发式映射函数非数据驱动**：
   - 当前 $f_\alpha$, $g_K$ 是人工设计，未从数据中自动学习最优映射关系。
3. **性能受限于原始特征质量**：
   - 若输入特征本身判别性差，即使传播优化也难以大幅提升性能。
4. **超参数仍需经验调优**：
   - 虽然传播参数自适应，但 $K_{\min}, K_{\max}, \alpha_{\min}, \alpha_{\max}$ 仍需手动设定。

---

### 🔮 未来工作方向
1. **引入多维结构描述符**：
   - 结合多种局部拓扑指标（如 PageRank, Betweenness）替代单一 LCC，提升适应性。
2. **轻量级学习机制融合**：
   - 探索是否可通过极少量参数微调（如 meta-learning）让映射函数数据自适应。
3. **扩展至其他图任务**：
   - 应用于 link prediction、graph classification 或图异常检测等场景。
4. **理论分析自适应机制**：
   - 形式化解释为何 LCC 能有效指导传播行为，建立更坚实的理论基础。

---

## ✅ 总结一句话
> **F²LP-AP 是一种无需训练、基于 LCC 自适应调节传播参数、并结合几何中位数原型的高效图节点分类方法，在兼顾极致推理速度的同时，实现了跨同质/异质图的 SOTA 性能，为 training-free graph learning 提供了新的范式。**

</details>

---

### 6. [Temporally Extended Mixture-of-Experts Models](https://arxiv.org/abs/2604.20156)

**Authors**: Zeyu Shen, Peter Henderson  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.20156v1  

#### Abstract
Mixture-of-Experts models, now popular for scaling capacity at fixed inference speed, switch experts at nearly every token. Once a model outgrows available GPU memory, this churn can render optimizations like offloading and pre-fetching ineffective. We make the case that the options framework in rei...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Temporally Extended Mixture-of-Experts Models**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前的 **Mixture-of-Experts (MoE)** 模型在每个 token 生成时几乎都会切换激活的专家（expert），导致频繁的 **expert switching**。当模型规模超过 GPU 内存容量时，需要将部分专家 **offload** 到主机内存或磁盘，并在需要时动态加载（on-demand loading）。然而，频繁的专家切换会破坏 **prefetching** 和 **caching** 等优化策略的有效性，显著增加推理延迟，限制了大规模 MoE 模型的实际部署效率。

### **提出了什么新方法或新思路**
本文提出 **Temporally Extended MoE (TE-MoE)**，引入 **options framework**（来自强化学习）来建模专家选择过程，使专家集合在多个 token 上持续激活，而非每步都切换。

- **核心思想**：将每个 MoE 层中的 **expert mask** 视为一个 **option**，由一个轻量级的 **controller** 决定何时保持当前专家集合、何时切换到新集合。
- **控制器设计**：基于 **option-critic framework**，并引入 **deliberation cost**（决策成本）作为显式惩罚项，鼓励控制器仅在预期收益大于切换成本时才进行切换。
- **训练方式**：采用 **self-distillation** 策略，以原始 MoE 模型为“教师”，训练带控制器的学生模型，目标是保留原始模型性能的同时最小化切换频率。

### **相比现有方法的优势**
| 方面 | 传统 MoE | 本文 TE-MoE |
|------|--------|------------|
| **专家切换频率** | 极高（接近每 token 一次） | 显著降低（从 >50% 降至 <5%） |
| **内存优化潜力** | 有限（需常驻所有专家） | 高（只需常驻当前活跃专家） |
| **训练方式** | 固定路由机制 | 可学习的动态控制策略 |
| **扩展性** | 新增专家需重新训练 | 支持 **continual learning**，可动态添加专家 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **训练数据**：`Nemotron Post-Training Dataset v2`，包含 10 类任务（chat, code, math, STEM, 多语言等），共 1280 条 prompt。
- **评估基准**：
  - **MATH**：数学推理能力
  - **MMLU**：多任务语言理解
  - **MMMLU**：多语言多任务理解

### **实验设置**
- **基础模型**：`gpt-oss-20b`，24 层 MoE，每层 32 个专家，top-4 路由（k=4）
- **控制器架构**：
  - 每层独立控制器
  - 输入：LLM 隐藏状态 $ h^{(l)} $ 和当前 expert mask
  - 输出：
    - 终止概率 $ \beta $
    - 新选项（expert mask）采样策略（Plackett-Luce 分布）
  - 使用 **LoRA**（rank=16）微调专家和注意力参数
- **奖励函数**：使用 **reverse KL divergence** 作为 per-token reward：
  $$
  r_t = \log p_{\text{teacher}}(a_t | x, a_{<t}) - \log p_{\text{student}}(a_t | x, a_{<t})
  $$
- **训练细节**：
  - Deliberation cost $ \eta \in \{0.02, 0.03, 0.04\} $
  - Batch size: 16 prompts
  - Max sequence length: 512
  - 使用 **teacher mixing**（混合教师输出采样）防止退化

### **评估指标**
- **Switch Rate (%)**：token 级别上 expert mask 发生变化的比例
- **Accuracy (%)**：在 MATH、MMLU、MMMLU 上的准确率
- **Perplexity**：衡量生成质量
- **Repetition Rate**：检测是否陷入重复循环

### **基线方法对比**
| 基线方法 | 描述 |
|--------|------|
| **Frequency-based** | 保留在校准集上最常被激活的 k 个专家 |
| **Reconstruction Loss Minimization** | 选择能最好重构原 MoE 输出的专家子集 |
| **Random Selection** | 每次随机选择 k 个专家 |
| **Wanda (structured pruning)** | 结构化权重剪枝方法 |

> ⚠️ 注意：不与 MoE-infinity 等缓存系统比较，因目标不同——本文关注的是**改变路由行为本身**，而非优化已有路由下的内存管理。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（k=16）**
| Method | MATH Acc (%) | MMLU Acc (%) | MMMLU Acc (%) | Switch Rate (%) |
|--------|---------------|---------------|----------------|------------------|
| Base Model | 71.5 ± 5.9 | 79.5 ± 5.7 | 67.5 ± 6.5 | ~50+ |
| Frequency | 53.5 | 55.5 | 42.0 | — |
| Reconstruction | 51.5 | 35.0 | 48.0 | — |
| Random | 15.0 | 33.5 | 24.0 | — |
| **Ours ($\eta=0.02$)** | **64.0 ± 6.7** | **72.5 ± 6.3** | **59.5 ± 6.9** | **4.2 ± 0.02** |
| **Ours ($\eta=0.04$)** | 55.0 | 63.0 | 49.5 | **1.2** |

> ✅ 在 $ \eta=0.02 $ 下，**switch rate 从 >50% 降至 4.2%**，同时保留了约 **90% 的原始模型准确性**。

### **与基线方法的对比结果**
- 所有静态剪枝方法（frequency, reconstruction, random）均导致严重性能下降（MATH 下降 15–55 pts）
- 本文方法在相同专家预算下（k=16 或 k=8），**显著优于所有基线**，尤其在数学和多语言任务上优势明显
- 即使在更严格的 k=8 设置下，仍能保持合理性能（如 MMLU 达 48.5%）

### **消融实验结果**
- **Deliberation Cost $\eta$ 的影响**：
  - $\eta$ 越大，switch rate 越低（trade-off between efficiency and capability）
  - 可通过调节 $\eta$ 控制切换频率与性能之间的平衡
- **Temporal Continuity 可视化**（Fig. 6 & 7）：
  - 原始模型：expert 激活无时间连续性
  - 本方法：同一 expert mask 持续数十至数百个 token，表现出强 temporal chunking
- **训练稳定性**（Appendix A3）：
  - 未出现 catastrophic repetition
  - 学生模型输出逐渐逼近教师模型（perplexity 下降）

---

## **4. 关键结论和发现**

### **主要发现**
1. **MoE 路由天然适合 temporal abstraction**：将 expert mask 视为 option，能有效建模长期依赖，减少不必要的切换。
2. **轻量训练即可实现高效转换**：无需从头预训练，通过少量 adapter 参数和 self-distillation，即可将标准 MoE 转换为 TE-MoE。
3. **显著降低 switch rate 同时保留性能**：switch rate 可从 >50% 降至 <5%，甚至 <1%，且在 MATH/MMLU/MMMLU 上保留高达 90% 的原始性能。
4. **打开三大应用前景**：
   - **Memory-efficient serving**：仅需加载当前活跃专家，大幅降低 VRAM 需求
   - **Chunk-wise training**：支持按时间块 offload 不活跃专家，降低训练峰值内存
   - **Continual learning**：可动态添加新专家模块，实现能力扩展而不增加推理成本

### **方法的局限性**
| 局限性 | 说明 |
|-------|------|
| **Per-layer 控制器** | 各层独立决策，可能导致跨层不一致；理想情况应联合控制所有层 |
| **Deliberation cost 是超参** | 尚未与真实硬件延迟对齐，未来需根据实际设备 calibrate |
| **评估范围有限** | 未涵盖代码生成、长文本对话等复杂场景 |
| **未完全解耦 temporal extension 与 self-distillation** | 性能提升部分源于参数微调，需进一步 ablation 分离效果 |

### **未来工作方向**
1. **端到端系统实现**：构建支持 TE-MoE 的 inference engine，验证真实环境下的内存与延迟收益。
2. **pre-training 中集成 temporal structure**：在预训练阶段就引入 temporal continuity，打造“天生”高效的 MoE 架构。
3. **跨层联合 options**：定义全局 option，协调所有层同步切换，简化 chunking 与 offloading。
4. **硬件感知 cost modeling**：将 deliberation cost 与具体 GPU/host 延迟绑定，实现自动化的 cost-quality trade-off。
5. **探索更丰富的时间抽象形式**：结合 internal reasoning chain 或 topic dynamics，实现语义驱动的 expert routing。

---

> 📌 **一句话总结**：  
> 本文提出 **Temporally Extended MoE**，利用 **options framework** 和 **deliberation cost**，首次实现了 MoE 路由的**时间扩展性**，在极低切换率下保持高性能，为大规模 MoE 模型的高效部署与持续学习提供了全新范式。

</details>

---

### 7. [Hidden Reliability Risks in Large Language Models: Systematic Identification of Precision-Induced Output Disagreements](https://arxiv.org/abs/2604.19790)

**Authors**: Yifei Wang, Tianlin Li, Xiaohan Zhang, Xiaoyu Zhang, Wei Ma, Mingfei Cheng, Li Pan  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.19790v1  

#### Abstract
Large language models (LLMs) are increasingly deployed under diverse numerical precision configurations, including standard floating-point formats (e.g., bfloat16 and float16) and quantized integer formats (e.g., int16 and int8), to meet efficiency and resource constraints. However, minor inconsiste...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Hidden Reliability Risks in Large Language Models: Systematic Identification of Precision-Induced Output Disagreements

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文揭示并系统研究了一个被广泛忽视的可靠性风险：**数值精度变化（如从 `bfloat16` 切换到 `float16` 或 `int8`）会导致大型语言模型（LLMs）在行为上出现不一致**，尤其是在安全对齐（safety alignment）任务中表现为“精度诱导型越狱”（precision-induced jailbreaks）。即同一个输入，在不同精度下可能一个产生拒绝响应，另一个却输出有害内容。

这一问题在实际部署中极为关键，因为模型通常在高精度环境下训练和评估，但在低精度下推理以节省资源，这种**训练-推理精度不匹配**可能导致安全隐患未被发现。

---

### 提出的新方法：PrecisionDiff

作者提出了 **PrecisionDiff** ——首个用于检测 LLM 中由数值精度引发的行为不一致的自动化差分测试（differential testing）框架。

#### 核心思想：
将同一模型在两种不同数值精度配置下的运行视为两个“实现”，通过生成对精度敏感的测试输入，寻找能引发输出分歧的输入样例。

#### 技术创新：
- **双精度联合优化目标（Dual-Precision Joint Optimization）**：
  同时优化两个目标：
  - 在目标精度（如 `float16`）下诱导有害输出（`yharm`）
  - 在参考精度（如 `bfloat16`）下保持安全拒绝（`ysafe`）
  
  损失函数形式为：
  $$
  \min_{x_{\text{adv}}} \mathcal{L}(f^{[p2]}(x), y_{\text{harm}}) + \lambda \mathcal{L}(f^{[p1]}(x), y_{\text{safe}})
  $$
  这种设计主动引导搜索朝向“精度敏感决策边界”。

- **动量引导的候选搜索（Momentum-guided search）**：
  在离散 token 空间中使用梯度信息进行坐标式更新，并引入动量机制稳定优化过程。

- **根因分析模块（Root-Cause Analysis）**：
  通过前向钩子记录各层激活值，计算 **Mean Absolute Difference (MAD)** 和 **Relative Divergence Lift (RL)**，定位导致分歧放大的关键网络层。

---

### 相比现有方法的优势
| 方面 | PrecisionDiff 的优势 |
|------|------------------------|
| **针对性** | 专门针对“精度差异”这一维度设计，而非通用越狱攻击 |
| **有效性** | 显著高于随机搜索、模糊测试（fuzzing）、遗传算法等基线 |
| **效率** | 虽然每步耗时更高（需双模型前向），但收敛更快，总时间可控 |
| **可解释性** | 提供层级别诊断，指导选择性精度提升（selective precision elevation） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **AdvBench**：从中采样 50 个有害查询作为初始 prompt。
- 分类涵盖：暴力、非法活动、网络犯罪、欺诈、隐私侵犯、仇恨言论、虚假信息、违禁品等。

### 实验设置
- **模型**：5 个开源对齐 LLMs
  - Llama-2-7B-chat-hf
  - Meta-Llama-3-8B
  - Vicuna-7B-v1.5
  - Mistral-7B-Instruct-v0.2
  - Guanaco-7B-HF
- **精度组合**：
  - 浮点型：`bfloat16` vs `float16`
  - 整数量化：`bfloat16` vs `int16`，`int16` vs `int8`
- **控制变量**：
  - 权重相同
  - 输入构造一致
  - 解码方式固定（greedy sampling, `do_sample=False`）
  - 种子一致

### 评估指标
| 指标 | 定义 |
|------|------|
| **Success Rate** | 成功触发精度诱导越狱的比例（共50个query） |
| **Average Iterations** | 所有成功案例首次触发所需的平均迭代次数（越低越好） |
| **Wall-clock Time** | 单次迭代的实际运行时间（秒） |
| **Critical Layers** | 通过 RL 指标识别出的分歧放大层 |

### 基线方法对比
| 基线方法 | 描述 |
|---------|------|
| **Random Search** | 随机替换 token |
| **Fuzzing (AFL++)** | 经典模糊测试工具适配 |
| **Genetic Algorithm** | 遗传算法进行序列进化 |
| **Standard GCG (FP16)** | 单精度下的 Greedy Coordinate Gradient 攻击 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 模型 | BF16 vs FP16（成功率） | INT16 vs INT8（成功率） |
|------|------------------------|--------------------------|
| Guanaco 7B | 78.0% | 100.0% |
| Llama-2 7B | 68.0% | 98.0% |
| Mistral-7B | 96.0% | 100.0% |
| Vicuna-7B | 94.0% | 100.0% |
| **平均** | **84.0%** | **99.5%** |

> ✅ 表明几乎所有主流对齐 LLM 都存在严重的精度诱导行为分歧。

---

### 与基线方法对比（Table 4，Vicuna-7B & Llama-2-7B）

| 方法 | Vicuna-7B（Success / Iter） | Llama-2-7B（Success / Iter） |
|------|-------------------------------|-------------------------------|
| Random Search | 2.0% / 19.0 | 0.0% |
| Fuzzing (AFL++) | 8.0% / 84.0 | 0.0% |
| Genetic Algorithm | 8.0% / 12.3 | 0.0% |
| Standard GCG | 50.0% / 59.0 | 8.0% / 94.3 |
| **PrecisionDiff (Ours)** | **72.0% / 45.0** | **68.0% / 19.4** |

> 🔺 **相对提升**：
> - Vicuna 上比 GCG 提升 **1.4× 成功率**
> - Llama-2 上比 GCG 提升 **8.5× 成功率**

---

### 时间开销分析
- **Standard GCG**: ~5.31 秒/iter（单卡 FP16）
- **PrecisionDiff**: ~9.97 秒/iter（双卡，BF16 + FP16）
- 尽管单步耗时约 **1.88×**，但由于收敛更快（平均少 1.3× 迭代），**总体 wall-clock time 接近甚至更优**，且检测率大幅提高。

---

### 消融与扩展实验结果

#### （1）不同解码策略下的表现（Table 5）
启用 `do_sample=True` 后：
- Llama-3-8B 在多个转换中达到 **100.0% 成功率**
- 表明 **采样随机性会加剧精度差异的影响**

#### （2）预算受限情况（max_step=200，Table 6）
即使将最大迭代数减少 60%，成功率仍维持在 **62.0%~92.0%**，说明方法具有良好的实用性。

#### （3）双向测试不对称性
- `BF16 → FP16`（高→低）成功率普遍高于反向
- 原因：多数模型在 `bfloat16` 下对齐训练，其安全边界更适应 BF16 数值特性

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **精度变化显著影响 LLM 安全行为**  
   同一模型在不同精度下可能出现“一个拒绝、一个越狱”的现象，证明对齐行为并非精度不变（precision-invariant）。

2. ✅ **行为分歧普遍存在且严重**  
   在所有测试模型中均观察到高成功率（最高达 100%），尤其在整数量化（如 `int16`→`int8`）时最为明显。

3. ✅ **某些危害类别更易受精度影响**  
   - 最敏感：Cybercrime（92.9%）、Malware（90.9%）、Fraud（89.9%）
   - 最稳健：Misinformation（71.2%）、Illegal Substances（68.2%）
   - 可能原因：技术类内容与编程相关，对齐边界较窄；社会敏感话题训练覆盖更广。

4. ✅ **分歧放大集中在特定网络组件**
   层级分析显示，分歧主要在以下位置被放大：
   - **输入阶段层（Input-stage layers）**
   - **初始注意力机制中的 WQ/WK 投影矩阵**
   - **输出端模块（LayerNorm 和 LM Head）**

   > 💡 这些是数值敏感操作（如大矩阵乘法、softmax、归一化统计量估计），微小舍入误差会被指数级放大。

5. ✅ **量化感知对齐可能增强鲁棒性**
   Guanaco 使用 QLoRA 微调（4-bit），表现出最低的成功率，暗示**在量化条件下对齐可能带来跨精度稳定性**。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖公开权重模型** | 当前仅验证于开源 LLM，未测试 GPT-4、Claude 等闭源模型 |
| **未覆盖全部混合精度场景** | 如 per-layer mixed precision、dynamic quantization 尚未探索 |
| **自动分类器可能存在误判** | 使用规则/模型判断是否 jailbreak，对边缘案例可能不准 |
| **侧重非对称越狱** | 主要关注“高精度安全 ↔ 低精度越狱”，忽略其他类型的分歧 |

---

### 未来工作方向
1. **开发精度鲁棒的对齐训练方法**  
   如在多种精度下联合训练或正则化，提升 cross-precision consistency。

2. **构建选择性精度提升方案**  
   对关键层（如 WQ/WK、Head）保留高精度，其余层量化，平衡性能与安全性。

3. **扩展至其他任务和模态**  
   如代码生成、多模态模型中的精度敏感性分析。

4. **集成进 CI/CD 流程**  
   将 PrecisionDiff 作为预部署检查工具，确保模型在目标硬件上的行为一致性。

---

> 📌 **总结一句话**：  
> 本文首次系统揭示了 **LLM 的安全对齐行为高度依赖数值精度**，提出 **PrecisionDiff** 框架可高效发现此类隐藏风险，为构建更可靠、鲁棒的 AI 系统提供了新视角与实用工具。

</details>

---

### 8. [Less Languages, Less Tokens: An Efficient Unified Logic Cross-lingual Chain-of-Thought Reasoning Framework](https://arxiv.org/abs/2604.20090)

**Authors**: Chenyuan Zhang, Qiguang Chen, Xie Chen, Zhuotao Tian, Bowen Xing, Meishan Zhang, Libo Qin, Baotian Hu, Min Zhang  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.20090v1  

#### Abstract
Cross-lingual chain-of-thought (XCoT) with self-consistency markedly enhances multilingual reasoning, yet existing methods remain costly due to extensive sampling of full trajectories across languages. Moreover, multilingual LLM representations vary strongly by language, hindering direct feature com...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Less Languages, Less Tokens: An Efficient Unified Logic Cross-lingual Chain-of-Thought Reasoning Framework**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 **Cross-lingual Chain-of-Thought (XCoT)** 推理方法虽然通过多语言自洽性（self-consistency）提升了多语言推理能力，但存在显著的**计算效率瓶颈**：
- **Full-language sampling**：必须为所有候选语言生成完整的推理轨迹，导致大量冗余计算。
- **Full-trace reasoning**：每个语言路径都需完整解码至结束，即使部分路径质量低下或逻辑不一致。

这使得推理成本随语言数量线性增长，尤其在资源受限场景下难以部署。

---

### **提出的新方法：UL-XCoT**
本文提出了首个高效的统一逻辑跨语言推理框架——**Unified Logic XCoT (UL-XCoT)**，从两个维度实现高效推理：

#### ✅ 创新点一：Less Languages —— 统一逻辑空间下的候选语言选择（Candidate Language Selection, CLS）
- 构建一个**语言不变的统一逻辑空间（Unified Logic Space, ULM）**，通过投影操作消除语言表层差异，保留任务相关的推理结构。
- 在该空间中计算不同语言对输入的理解一致性得分（Understanding Similarity Score, USS），仅选择最相关的前 *k* 种语言进行后续推理，大幅减少参与语言数量。

#### ✅ 创新点二：Less Tokens —— 动态 CoT 路径剪枝（Dynamic CoT Pruning, DCP）
- 在解码过程中实时监控各语言路径在统一逻辑空间中的演化轨迹。
- 引入**逻辑质量评分（Logical Quality Score, LQS）**，动态识别并提前终止低质量、发散或冗余的推理路径，避免无效 token 生成。

#### ✅ 最终聚合策略
- 对剩余高质量路径采用投票机制（voting）得出最终答案，在保证准确性的同时极大降低计算开销。

---

### **相比现有方法的优势**
| 方面 | 传统 XCoT 方法 | UL-XCoT |
|------|----------------|---------|
| **语言使用** | 全量语言采样 | 自适应筛选少量高相关语言 |
| **token 开销** | 完整轨迹生成 | 动态剪枝，早期停止 |
| **推理效率** | 高延迟、高成本 | 显著降低 token 数与延迟 |
| **鲁棒性** | 在低资源语言上表现不稳定 | 在低资源语言上更稳定且增益明显 |

> 🔑 **核心优势**：在有限采样预算下实现了**最高效率的跨语言推理**，解决了“为何要为所有语言生成完整路径”的根本问题。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **PolyMath**  
   - 多语言数学推理基准，覆盖 **18 种语言**，包含 4 个难度等级（Low/Medium/High/Top）。
   - 主要用于评估数学推理能力和效率权衡。

2. **MMLU-ProX-Lite**  
   - 多语言多项选择题基准，涵盖 **29 种语言**，涉及广泛的知识与推理类别。
   - 用于验证方法在非数学任务上的泛化能力。

---

### **实验设置与评估指标**

#### **模型与硬件**
- **Backbone 模型**：DeepSeek-R1-Distill-Qwen-7B
- **硬件平台**：NVIDIA RTX A6000 GPU (48GB)
- **最大生成长度**：2048–10240（根据任务难度调整）

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **Accuracy** | 使用 DW-ACC（Difficulty-Weighted Accuracy）作为主指标 |
| **Efficiency** | - 生成 token 总数<br>- 端到端 wall-clock latency（秒） |

#### **控制变量**
- 所有方法使用相同的 backbone 和 prompt 模板（concise-reasoning template）。
- 控制采样预算与 UL-XCoT 的 worst-case 水平对齐，确保公平比较。

---

### **基线方法对比**
| 基线 | 简介 |
|------|------|
| **CoT (Wei et al., 2022)** | 单路径思维链提示 |
| **CLP / CLSP (Qin et al., 2023)** | 跨语言提示及其自洽版本 |
| **SC (Self-Consistency, Wang et al., 2022)** | 多路径采样 + 投票 |
| **AUTOCAP (Zhang et al., 2024)** | 自动语言选择 + 加权聚合 |
| **ST-BoN (Wang et al., 2025c)** | 固定预算下的高效采样方法 |
| **UL-CoT** | UL-XCoT 的单语对照变体（无跨语言交互） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 📊 在 **PolyMath** 上的结果（平均 DW-ACC）
| 方法 | DW-ACC (%) | 平均 Token 数 | 相比 SC 减少 |
|------|------------|--------------|-------------|
| SC | 15.2 | ~9,000 | — |
| AUTOCAP | 18.2 | ~6,500 | ~28% ↓ |
| **UL-XCoT** | **19.0** | **~3,092** | **>65% ↓** |

> ✅ **UL-XCoT 在准确率上达到 SOTA，同时将 token 成本降低超过 65%（vs SC）、超 50%（vs AUTOCAP）**

#### ⏱️ 推理延迟（Latency）
- UL-XCoT 的平均延迟为 **24.6 秒**，显著低于其他方法（如 SC 达 35.9 秒以上）。
- 尤其在高资源语言（如 en/zh）中有效抑制了“过度思考”（overthinking）现象。

---

#### 🌍 在 **MMLU-ProX-Lite** 上的泛化表现
| 方法 | 平均 Accuracy | 平均 Token 数 | 平均 Latency |
|------|---------------|----------------|--------------|
| CLSP | 40.5% | 27,679.3 | 134.2 s |
| **UL-XCoT** | **43.6%** | **10,543.6** | **93.7 s** |

> ✅ 在更广泛的多语言知识推理任务上仍保持精度提升 + 效率飞跃。

---

### **消融实验结果（Ablation Study on PolyMath-Low）**

| 变体 | ACC (%) | Token 数 | Latency (s) |
|------|--------|----------|------------|
| UL-XCoT (完整) | **83.8** | **3,092** | **24.6** |
| w/o CLS | 84.4 | 5,560 | 36.2 |
| w/o DCP | 81.4 | 3,893 | 30.7 |
| w/o ULM | 79.8 | 3,098 | 25.4 |
| w/o 所有模块 | 85.2 | 7,518 | 35.9 |

#### 🔍 分析结论：
- **ULM 是精度的关键贡献者**：移除后 ACC 下降最多（↓4%），说明统一逻辑空间对跨语言比较至关重要。
- **CLS 主要提升效率**：移除后 token 和 latency 显著上升，表明其有效缩小搜索空间。
- **DCP 实现在线剪枝**：虽轻微牺牲精度，但带来巨大效率收益，证明冗余路径可安全裁剪。
- **多语言协作本身有价值**：UL-XCoT 明显优于单语版 UL-CoT，说明跨语言互补性不可替代。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **效率与精度可以兼得**：UL-XCoT 在多个基准上实现了**竞争性甚至更优的准确率**，同时将 token 消耗和延迟削减过半。
2. ✅ **统一逻辑空间是关键使能技术**：通过 ULM 投影，首次实现了跨语言推理状态的直接可比性，支撑了 CLS 与 DCP。
3. ✅ **动态剪枝真正去除低质路径**：LLM-as-a-judge 评测显示被剪枝路径在 step validity 和 completeness 上显著更差。
4. ✅ **对低资源语言更具鲁棒性**：在低资源子集上表现更稳定，而标准 XCoT 方法在此类语言上易失效。
5. ✅ **方法具备良好泛化性**：在 MMLU-ProX-Lite 上同样取得精度与效率双提升。

---

### **局限性**
- **依赖白盒访问 hidden states**：需要获取中间层表示以构建统一逻辑空间，因此目前仅适用于开放权重模型。
- ❌ **不适用于严格黑盒 API**：如 GPT 等闭源模型无法应用此方法，除非提供隐藏层输出接口。
- **超参数敏感性**：pruning ratio *p*、warm-up 步数等需调优以平衡性能与效率。

---

### **未来工作方向**
1. **探索轻量化 ULM 构建方式**：设计无需额外验证集的语言中心估计方法。
2. **扩展至 Tree-of-Thoughts 或 Graph-of-Thoughts**：将统一逻辑机制应用于更复杂的推理拓扑结构。
3. **结合 Reward Modeling 进行路径排序**：利用 cross-lingual reward model 进一步优化路径选择。
4. **推动黑盒适配方案**：研究基于 probing 或 probing-free 的隐式空间对齐技术。

---

> 💡 **一句话总结**：  
> **UL-XCoT 通过“统一逻辑空间 + 自适应语言选择 + 动态路径剪枝”，首次实现了高效、鲁棒、可扩展的跨语言 CoT 推理，在保持甚至提升准确率的同时，将推理成本削减超过 50%，为多语言大模型的实际部署提供了新范式。**

🔗 **代码开源地址**：[https://github.com/chenyuanTKCY/UL-XCoT](https://github.com/chenyuanTKCY/UL-XCoT)

</details>

---

### 9. [FASER: Fine-Grained Phase Management for Speculative Decoding in Dynamic LLM Serving](https://arxiv.org/abs/2604.20503)

**Authors**: Wenyan Chen, Chengzhi Lu, Yanying Lin, Dmitrii Ustiugov  
**Category**: cs.DC  
**Published**: 2026-04-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.20503v1  

#### Abstract
Speculative decoding (SD) is a widely used approach for accelerating decode-heavy LLM inference workloads. While online inference workloads are highly dynamic, existing SD systems are rigid and take a coarse-grained approach to SD management. They typically set the speculative token length for an en...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FASER: Fine-Grained Phase Management for Speculative Decoding in Dynamic LLM Serving

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **Speculative Decoding (SD)** 系统在处理动态的在线 LLM 推理负载时存在严重不足，主要体现在以下三个方面：

- **粗粒度管理**：大多数系统为整个 batch 设置固定的 speculative token 长度，无法适应请求间的多样性。
- **序列化执行**：draft 和 verification 阶段串行执行，导致低负载下 GPU 资源闲置、延迟高；高负载下验证大量被拒 token 浪费计算资源。
- **缺乏细粒度优化**：即使 acceptance rate 较高，仍可能因验证开销过大而无法降低延迟。

这些问题使得现有 SD 系统难以应对真实场景中高达 35× 的流量波动。

---

### 🚀 提出的新方法：FASER
FASER 是一种面向动态 LLM 服务的新型 SD 系统，其核心思想是 **细粒度阶段管理（fine-grained phase management）**，包含三大关键技术：

#### （1）**自适应 drafting（Adaptive Drafter）**
- 动态地为每个请求独立设置 speculative token 长度 $ s_i $，而非统一长度。
- 基于当前 batch size、SM 分配和历史 acceptance 行为，使用 **GP-LCB** 在线搜索最优长度，平衡 speculation 并行性和 verification 开销。

#### （2）**逐 token 早退机制（Token-wise Early Exiter）**
- 在 verification 过程中，每层检查 drafted token 是否值得继续验证。
- 利用 draft 和 target 模型之间的 Top-K logit 差异作为信号，在中间层提前终止对“拒绝后缀”（rejected suffix）的计算。
- 减少目标模型在无效 token 上的冗余计算。

#### （3）**基于 Frontiers 的流水线重叠（Pipeline Overlapper）**
- 将 speculative tokens 组织成可验证的“前沿块”（frontier chunks），允许 verification 在 draft 完成前启动。
- 使用 **Green Contexts** 实现 GPU SM 的空间复用，使 draft 和 verification 可并发执行，减少 pipeline bubbles。
- 实现真正的 producer-consumer 流水线：一边生成下一个 chunk，一边验证当前 chunk。

---

### 🔍 相比现有方法的优势
| 方法 | 自适应长度 | 早退机制 | 阶段并行 | 细粒度控制 |
|------|------------|----------|-----------|--------------|
| SpecInfer | ❌ | ❌ | ❌ | ❌ |
| AdaSpec | ✅ | ❌ | ❌ | ❌ |
| Smurfs | ✅ | ❌ | ✅（粗粒度） | ❌ |
| **FASER（本文）** | ✅ | ✅ | ✅ | ✅ |

> FASER 是首个同时实现 **request-level 自适应 drafting + token-level early exit + hardware-aware pipeline overlap** 的系统。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **ShareGPT**：真实对话数据，平均输入/输出长度为 755 / 200。
- **LongBench**：长上下文任务，平均长度 1738 / 90。
- **HumanEval**：代码生成任务，平均长度 171 / 98。
- 请求到达模式来自 **Azure LLM invocation trace (DynamoLLM)**，具有高度动态性（峰值与谷值 RPS 差异达 35×）。

---

### ⚙️ 实验设置
- **硬件平台**：单台服务器配备两个 NVIDIA H100 GPU（96GB VRAM each），PCIe 4.0 互联。
- **基础框架**：基于 **vLLM v0.15.1** 实现，约 5k 行 Python 代码。
- **模型组合**：
  | Draft Model | Target Model | TP |
  |------------|---------------|-----|
  | Qwen3-0.6B | Qwen3-32B | 1 |
  | Llama3.2-1B | Llama3.3-70B | 2 |

- **评估指标**：
  - **Latency**：端到端响应时间（per request）
  - **Throughput**：每秒输出 token 数（output tokens/sec）
  - **Acceptance Rate**
  - **Early-exit ratio**

---

### 🆚 基线方法对比
- **SpecInfer**：基于树结构的 speculative inference，提升候选多样性。
- **AdaSpec**：支持自适应 speculative 长度，考虑 SLO 约束。
- **Smurfs**：支持多任务混合与跨 batch 的流水线执行。

---

## 3. 主要实验结果和性能指标

### 📈 性能提升汇总
| 指标 | 提升幅度 | 场景说明 |
|------|---------|--------|
| **最大吞吐量提升** | **+53%** | vs. state-of-the-art baseline |
| **最低延迟降低** | **1.92×** | 最高可达 48% 延迟下降 |
| **平均 early-exit 比例** | **5.0% ~ 14.5%** | 成功剪枝低质量 speculative token |

---

### 🔬 具体实验结果

#### （1）端到端性能（End-to-End Performance）
- **延迟（Latency）**：
  - 在 Qwen3 + LongBench 上，FASER 比最强基线降低 **48%** 延迟。
  - 在 Llama3 + LongBench 上，延迟降低 **42%**。
  - 改进主要来源于：**critical path 缩短 + 更少等待 + 更轻的 verification 负担**。

- **吞吐量（Throughput）**：
  - 对 Qwen3 模型对，吞吐提升 **1.53×**；
  - 对 Llama3 模型对，吞吐提升 **1.49×**。
  - 原因：GPU 利用率更高，pipeline 更连续，避免空转。

#### （2）消融实验（Ablation Study）
以 Qwen3 模型对为例，逐步添加组件：

| 系统配置 | 延迟降幅 | 吞吐增益 |
|--------|--------|--------|
| VSD（固定长度） | — | 1.00× |
| + AD（自适应 drafting） | ↓19% | ↑1.35× |
| + EE（early exit） | ↓26% | ↑1.22× |
| + Pipeline Overlap（完整 FASER） | **↓61%** | **↑1.60×** |

> 结果表明三个模块协同效应显著，尤其是 **pipeline overlap 极大释放了并发潜力**。

#### （3）early-exit 效果分析
- **平均 early-exit 比例**：
  - ShareGPT：Qwen3 下达 **14.5%**，Llama3 下 **12.4%**
  - HumanEval：约 **8.8%~9.1%**
- 即使剪枝比例不高，也能有效减少 deep-layer 计算负担，尤其在长尾 token 上效果明显。

#### （4）batch size 与 speculative length 分布
- FASER 能根据负载动态调整 speculative length，通常选择 **5–8** 个 token。
- 在 ShareGPT 中，超过 40% 的 batch 大小 >128，约 30% <8，体现高度动态性，FASER 在此环境下优势更明显。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **更高的 acceptance rate 不一定带来更低延迟**  
   → 因为长 speculative 序列会增加 verification 开销，需权衡 speculation 长度与验证成本。

2. **串行执行是低负载下的主要瓶颈**  
   → draft 阶段阻塞 verification，造成 GPU 空闲；FASER 通过 pipeline overlap 显著缓解该问题。

3. **rejected suffix 的验证浪费巨大**  
   → 在 batch=32, speculative length=6 时，**超过一半 token 被拒绝**，相当于近半 FLOPs 浪费；early exit 可有效规避。

4. **细粒度控制优于全局策略**  
   → request-level 自适应 + token-level 决策 + hardware-aware 调度，共同构成高效 SD 新范式。

---

### ⚠️ 局限性
- **依赖 offline profiler**：需要预先采集不同配置下的性能表，虽误差较小（MAPE < 18%），但在极端变化场景下可能滞后。
- **memory contention 未完全解决**：尽管 Green Contexts 控制 SM 干扰，但共享 HBM 带宽仍可能导致 memory-bound 场景性能下降。
- **目前仅支持 dense 模型**：虽然已扩展至 MoE 模型（见 6.4.2），但专家路由行为复杂，未来需进一步适配。

---

### 🔮 未来工作方向
1. **将 FASER 扩展至更多 SD 变体**  
   → 如 Medusa、EAGLE 等 self-speculative decoding 框架（文中已初步验证，FASER 在 EAGLE 上实现 **2.01× 吞吐提升**）。

2. **支持 disaggregated serving 架构**  
   → 在 prefill 和 decode 分离的 DistServe 场景中应用 FASER 思想。

3. **引入更智能的 early-exit predictor**  
   → 当前使用 Top-K 信号简单有效，未来可训练 lightweight predictor 进一步提高准确率。

4. **探索 temporal multiplexing + spatial multiplexing 联合调度**  
   → 在多租户或多优先级场景中实现 SLO-aware 的资源分配。

---

## 总结
FASER 重新思考了 Speculative Decoding 在 **动态生产环境** 中的设计原则，提出了一套完整的 **细粒度 phase management 框架**。它通过 **adaptive drafting + token-wise early exit + frontier-based pipeline overlap** 三管齐下，解决了传统 SD 系统在负载波动下的性能退化问题。实验证明，FASER 在真实 trace 驱动下实现了最高 **53% 吞吐提升** 和 **1.92× 延迟降低**，且具备良好的通用性和可扩展性，代表了下一代高效 LLM serving 系统的重要方向。

</details>

---

### 10. [Robustness of Spatio-temporal Graph Neural Networks for Fault Location in Partially Observable Distribution Grids](https://arxiv.org/abs/2604.20403)

**Authors**: Burak Karabulut, Carlo Manna, Chris Develder  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.20403v1  

#### Abstract
Fault location in distribution grids is critical for reliability and minimizing outage durations. Yet, it remains challenging due to partial observability, given sparse measurement infrastructure. Recent works show promising results by combining Recurrent Neural Networks (RNNs) and Graph Neural Netw...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

**论文标题**: *Robustness of Spatio-temporal Graph Neural Networks for Fault Location in Partially Observable Distribution Grids*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对**部分可观测配电网络中的故障定位**（fault location）问题展开研究。由于现实中测量设备（如 uPMUs）部署稀疏，导致电网存在**部分可观测性**（partial observability），传统模型（如纯 RNN 或基于完整拓扑的 GNN）在该场景下面临以下挑战：
- 缺乏对电网拓扑结构的有效建模；
- 在稀疏传感器条件下性能下降明显；
- 使用全拓扑图时引入无测量节点（特征为零），造成信息稀释和噪声传播。

### 🚀 提出的新方法与创新思路
本研究提出并系统评估了以下三项关键创新：

#### （1）**新型 GNN 图构建策略：Measured-only Topology**
- 不再将所有物理母线作为图节点，而是**仅使用有 uPMU 测量的节点**来构建 GNN 图。
- 设计了一套**系统化的图构造算法**（Algorithm 1），保留电气连接关系和多相特性，反映真实的部分可观测条件。
- 优势：减少图规模、避免零值节点带来的信息稀释、提升训练效率与鲁棒性。

#### （2）**引入未被探索的 STGNN 架构用于电力系统故障定位**
- 首次将 **GraphSAGE** 和改进版 **GATv2** 引入配电系统 fault location 任务中。
- 构建统一的 **STGNN pipeline**：GRU 提取时间特征 → GNN 进行空间消息传递 → Soft Voting 聚合输出。
- 特别是 GATv2 支持非对称注意力机制，理论上更具表达能力。

#### （3）**全面的基准测试与鲁棒性分析**
- 在标准 IEEE 123-bus feeder 上进行大规模实验；
- 对比多种 GNN 架构（RGCN, RGSAGE, RGATv2）与纯 RNN 基线；
- 引入“绿色配置”（lateral-redirected configuration）模拟更复杂运行状态，验证模型在弱故障信号下的稳定性。

### ⚖️ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **图结构设计** | Measured-only 图显著优于 full-topology 图，尤其在弱信号下防止性能崩溃 |
| **模型架构** | 所有 STGNN 模型均优于纯 GRU，且更稳定（置信区间小） |
| **计算效率** | Measured-only 图使训练时间降低 **6倍** |
| **实用性** | 更贴近实际部署场景（稀疏测量 + 动态拓扑） |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **来源**：通过 **OpenDSS + PyDSS** 仿真生成合成数据。
- **基准系统**：**IEEE 123-bus feeder**，广泛用于 fault diagnosis 研究。
- **测量设备**：共部署 **25 个 uPMUs**，包括单相、双相、三相类型，位置不完全重合于故障点。
- **数据内容**：
  - 三相 RMS voltage magnitude（V1, V2, V3）
  - 时间分辨率为 **1ms**，每段故障事件采样 **20ms（20个 timestep）**
  - 故障前采集 40ms，滑动窗口提取共 **40个样本/故障事件**

### 🔧 实验设置
| 项目 | 设置说明 |
|------|----------|
| **故障类型** | 共 **11类短路故障**：AG/BG/CG, AB/BC/CA, ABG/BCG/CAG, ABC, ABCG |
| **故障数量** | **25个故障位置** × 11种类型 × 100次负载/电阻变化 = 27,500 场景 |
| **总样本数** | 约 **250万滑动窗口样本**（含50%正常情况） |
| **数据划分** | 70% 训练 / 15% 验证 / 15% 测试，保持时间一致性 |
| **输入特征** | RMS电压（Z-score归一化），未使用电流以符合现实部署限制 |

### 🎯 评估指标
- 主要指标：**F1 Score（macro）**
- 辅助分析：**90% 置信区间**（跨不同随机种子训练，衡量稳定性）
- 输出类别：**26类**（25个故障位置 + “无故障”）

### 🆚 基线方法对比
| 模型 | 类型 | 描述 |
|------|------|------|
| **GRU (shared)** | Non-GNN Baseline | 每个 uPMU 独立处理，soft voting 聚合预测 |
| **RGCN** | State-of-the-art GNN | 文献常用 GCN 变体，作为主流 GNN 基线 |
| **RGSAGE-Mean / Max** | Proposed | 基于 GraphSAGE 的 STGNN，采用均值或最大池化聚合 |
| **RGATv2** | Proposed | 改进型图注意力网络，支持非对称注意力 |

此外还比较了两种图拓扑：
- **Full Topology**：包含全部 128 个节点（123主节点+5辅助），未测量节点特征设为0
- **Measured-only**：仅包含 25 个 uPMU 节点，按所提算法连接

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（F1 Score）

| 模型 | 默认配置 F1 | 绿色配置 F1 | CI 宽度（ΔF1） |
|------|-------------|--------------|----------------|
| **GRU** | 86.3 ± 4.2 | 75.5 ± 7.5 | ↑ 3.3 pt |
| **RGCN (full)** | 87.2 ± 0.8 | 71.3 ± 0.1 | ↑ 15.9 pt |
| **RGSAGE-Max (full)** | 87.9 ± 0.9 | 71.4 ± 0.3 | ↑ 16.5 pt |
| **RGATv2 (full)** | 86.2 ± 1.4 | 71.2 ± 0.2 | ↑ 15.0 pt |
| **RGCN (measured-only)** | 94.7 ± 0.3 | 86.7 ± 0.9 | ↑ 8.0 pt |
| **RGSAGE-Max (measured-only)** | 94.9 ± 0.6 | **87.9 ± 0.9** | ↑ 7.0 pt |
| **RGATv2 (measured-only)** | **94.9 ± 0.6** | 86.3 ± 4.2 | ↑ 8.6 pt |

> 注：绿色配置指开关重构后导致故障信号更微弱的情形。

### 🔍 与基线方法的对比结果
- **STGNN vs RNN**：
  - 在默认配置下，STGNN 平均 F1 提升约 **+9个百分点**（从 ~86 到 ~95）
  - 在绿色配置下仍维持 **+11~12个百分点** 的领先优势
  - STGNN 的置信区间极窄（≤ ±1.4%），而 GRU 高达 ±7.5%，表明其**预测更稳定可靠**

- **Measured-only vs Full Topology**：
  - 在默认配置下两者性能接近（均达 ~95 F1）
  - 但在绿色配置下，**full topology 性能骤降至 ~71 F1**，而 measured-only 仍保持在 **86–88 F1**
  - 原因：full topology 中大量零特征节点在消息传递中稀释有效信息，加剧过平滑（oversmoothing）

### ⚙️ 消融实验结果
#### （1）图拓扑的影响（Ablation on Graph Structure）
- 使用 measured-only 图可带来：
  - **高达 +11 F1 的性能增益**（在挑战性场景下）
  - **训练时间减少 6 倍**（见下表）
- 表明：**并非越多节点越好**，合理精简图结构反而提升性能与效率

#### （2）GNN 架构之间的差异
- 所有 GNN 变体在默认配置下表现相近（F1 ∈ [94.7, 94.9]）
- 在绿色配置下：
  - **RGSAGE-Max 表现最佳（87.9 ± 0.9）**
  - RGATv2 表现略逊（86.3 ± 4.2），可能因其注意力机制难以在弱信号下准确加权
- 推论：**max-pooling 更适合捕捉微弱峰值信号**，具有更强鲁棒性

#### （3）训练效率对比（Table 2）
| Model | Measured-only (min) | Full-topology (min) | 加速比 |
|-------|---------------------|----------------------|--------|
| RGATv2 | 65.07 ± 1.18 | 391.34 ± 19.62 | ~6.0× |
| RGSAGE-Max | 62.22 ± 1.14 | 372.38 ± 22.13 | ~6.0× |
| RGCN | 61.56 ± 0.70 | 363.17 ± 5.70 | ~5.9× |

> 结论：**measured-only 图大幅降低训练开销，且不影响甚至提升性能**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **STGNN 显著优于纯 RNN**：
   - 在稀疏 uPMU 条件下，融合时空建模的 STGNN 比纯 GRU 高出最多 **+11 F1**
   - 同时具备更强的**鲁棒性和稳定性**（CI ≤ ±1.4%）

2. **Measured-only 图优于 Full-topology 图**：
   - 尽管后者包含更多拓扑信息，但因引入零特征节点而导致**信息稀释和性能崩溃**
   - 在弱故障信号下，full topology 的 F1 下降超 **15个百分点**，而 measured-only 仅下降约 **7–8个百分点**

3. **图结构设计比模型选择更重要**：
   - 不同 GNN 架构之间性能差距较小；
   - 而图构建方式（measured-only vs full）带来的影响远大于模型本身的选择。

4. **Max-pooling 在挑战性场景中更具优势**：
   - RGSAGE-Max 在绿色配置下表现最优，说明其对**微弱故障信号的敏感性更高**

5. **计算效率大幅提升**：
   - 使用 measured-only 图可实现 **6倍训练加速**，更适合实际工程应用

### ⚠️ 方法的局限性
- 当前实验基于单一标准馈线（IEEE 123-bus），尚未验证在更大规模或不同类型网络上的泛化能力；
- 所有 uPMU 位置固定，未考虑动态新增传感器的情景；
- 未利用边特征（如线路距离、阻抗），可能进一步增强空间建模能力；
- GraphSAGE 未使用子采样（sub-sampling），在更大图上是否仍高效有待验证。

### 🔮 未来工作方向
1. **探索模型的迁移与泛化能力**：
   - 研究 GraphSAGE 的归纳能力（inductive learning）能否适应全新馈线拓扑；
   - 验证预训练模型在新增 uPMU 或拓扑变更后的适应性。

2. **增强 GNN 表达能力**：
   - 引入**边特征**（edge features）如地理距离、线路参数；
   - 探索更深的 GNN 架构或多层注意力机制。

3. **扩展至其他故障诊断任务**：
   - 如 fault type classification、fault resistance estimation；
   - 多任务联合学习框架。

4. **结合 imputation 与 contrastive learning**：
   - 参考 GDIA-GCL 等方法，在极端缺失下进一步提升鲁棒性。

---

> 💡 **一句话总结**：  
> 本文证明，在部分可观测配电系统中，**基于 measured-only 图的 STGNN 框架不仅性能更优、训练更快，而且在复杂工况下更加稳健**，为实际部署提供了高效可靠的解决方案。

</details>

---

### 11. [Efficient Test-Time Inference via Deterministic Exploration of Truncated Decoding Trees](https://arxiv.org/abs/2604.20500)

**Authors**: Xueyan Li, Johannes Zenn, Ekaterina Fadeeva, Guinan Su, Mrinmaya Sachan, Jonas Geiping  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.20500v1  

#### Abstract
Self-consistency boosts inference-time performance by sampling multiple reasoning traces in parallel and voting. However, in constrained domains like math and code, this strategy is compute-inefficient because it samples with replacement, repeatedly revisiting the same high-probability prefixes and ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Efficient Test-Time Inference via Deterministic Exploration of Truncated Decoding Trees*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **test-time scaling**（测试时计算扩展）中，主流方法如 **self-consistency** 通过并行采样多个推理路径（reasoning traces）并进行投票来提升大语言模型（LLM）的性能。然而，在数学、代码等**约束性强的任务**中，该策略存在严重效率问题：
- **重复采样**：由于模型输出分布高度集中，多次采样常生成相同或高度相似的前缀（prefixes）和完成序列（completions），造成大量冗余计算。
- **低覆盖率**：在固定计算预算下，随机采样难以充分探索可能的高概率分支。

### 提出的新方法：**Distinct Leaf Enumeration (DLE)**
DLE 是一种**确定性的解码方法**，将截断采样（truncated sampling）视为对剪枝后的解码树（pruned decoding tree）的遍历，并系统地枚举不同的叶子节点（distinct leaves），而非随机采样。

#### 核心思想
- 将生成过程建模为一棵由 **truncated sampling distribution** 定义的树，其中每个节点代表一个前缀，边代表满足截断规则（如 top-p, min-p, e-sampling）的候选 token。
- DLE 以**确定性方式**遍历这棵树，优先探索具有最高路径概率质量（path probability mass）的未访问分支，从而避免重复生成相同的序列。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **算法层面** | 提升了在固定序列预算下的**coverage**（覆盖的概率质量），更高效地探索高概率分支，避免“浪费”在重复路径上。 |
| **系统层面** | 天然支持**prefix reuse** 和 **KV-cache 共享**，显著减少新生成的 token 数量，降低内存占用和延迟。 |
| **通用性** | 作为 **drop-in replacement**，可无缝集成到多种截断采样器（top-p, min-p, e-sampling）中，不依赖特定采样分布。 |
| **无需额外开销** | 不需要像 Tree-of-Thought 或 p-decoding 那样引入额外的评估或模拟步骤，计算成本更低。 |

---

## 2. 核心实验方法和设置

### 数据集
实验在三个典型任务上进行：
- **GSM8K**：小学数学应用题，评估数学推理能力。
- **HumanEval**：代码生成任务，评估编程能力。
- **MMLU-Pro**：多任务语言理解基准，评估通用推理能力。

### 实验设置
- **模型**：主要使用 `Qwen2.5-0.5B-Instruct`，并在附录中验证了 `Qwen2.5-7B/14B` 和 `Llama3.2-1B/3B-Instruct` 上的结果。
- **截断采样器**：对比了 **top-p & top-k**, **min-p**, **e-sampling** 三种主流截断策略。
- **序列数量（k）**：控制生成的独立序列数（如 k=8, 32），用于与 self-consistency 对比。
- **推理引擎**：使用 **vLLM** 和 **SGLang**，后者显式支持基于前缀的 KV-cache 重用（RadixAttention）。

### 评估指标
| 指标 | 描述 |
|------|------|
| **maj@k / pass@k** | 主要性能指标：`maj@k` 表示多数投票准确率，`pass@k` 表示至少一次通过率。 |
| **Coverage (m_k)** | 定义为所选 k 个序列在截断分布 Q 下的总概率质量，衡量搜索空间的探索程度。 |
| **Cache Hit Rate** | 衡量 prefix reuse 效率，分理论最大值（C_th）和实际命中率（C_act）。 |
| **New Tokens Generated** | 实际新增生成的 token 数量，反映系统效率。 |

### 基线方法对比
- **Self-consistency**（基础及各类变体）
- **Beam search** / **Diverse beam search**
- **DeepConf**（基于置信度过滤）
- **Self-certainty**（基于 KL 散度评分）
- **Path-based strategies**（如 Path-consistency）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1，Qwen2.5-0.5B-Instruct）

| 方法 | GSM8K (maj@8) | HumanEval (pass@8) | MMLU-Pro (maj@8) |
|------|---------------|---------------------|------------------|
| Self-consistency (top-p&k) | 38.74 | 45.73 | 17.35 |
| **DLE (top-p&k)** | **44.43** (+5.7%) | **51.22** (+5.5%) | **18.19** (+0.8%) |
| Self-consistency (min-p) | 39.04 | 46.34 | 17.26 |
| **DLE (min-p)** | **43.97** (+4.9%) | **53.05** (+6.7%) | **17.89** (+0.6%) |
| Self-consistency (e-sampling) | 40.94 | 47.56 | 17.20 |
| **DLE (e-sampling)** | **44.05** (+3.1%) | **52.44** (+4.9%) | **17.97** (+0.8%) |

> ✅ **结论**：DLE 在所有任务和采样器上均显著优于对应的 self-consistency 基线，尤其在 GSM8K 和 HumanEval 上提升明显（+3–9%）。

### 与基线方法的对比结果
- **vs Self-consistency**：在相同序列预算下，DLE 覆盖更多概率质量（见 Figure 5），且生成更少的新 token（Figure 8 左），实现更高准确率。
- **vs Beam search**：Beam search 性能提升有限，甚至随 beam size 增加而下降，表明其不适合此类任务。
- **vs DeepConf / Self-certainty**：这些方法依赖于后处理或评分机制，性能增益较小，且不如 DLE 稳定。

### 消融实验结果（Ablation Studies）

#### (1) 分支策略比较（Table 2, B.1）
- **PROBFIRST**（按路径概率选择分支）和 **DIVFIRST**（尽早分叉）表现最佳。
- **RANDBRANCH**, **DFS**, **GLOBALPROB** 等策略性能较差，说明**优先探索高概率分支**是关键。

#### (2) 早停机制（Early Stopping）效果（Table 2, B.2）
- 引入早停（检测到连续 10 个 token 重复则终止）后，性能略有提升或持平。
- 浪费 token 占比 < 1%，代价极小。
- **Table 3 显示**：被早停的分支中，**超过 85% 最终会得到完全相同的答案**，证明早停有效避免冗余探索。

#### (3) 聚合方式（Table 4, B.4）
- 使用序列概率进行加权聚合（probability-weighted）**并未带来系统性提升**。
- **均匀加权（uniform weighting）最为鲁棒**，说明 DLE 本身已保证了高质量路径的多样性。

#### (4) 缓存命中率（Figure 9）
- DLE 的**理论缓存命中率远高于 self-consistency**。
- SGLang 实现的实际命中率接近理论上限，证明其能有效利用 DLE 的树结构进行 KV-cache 重用。

---

## 4. 关键结论和发现

### 主要发现
1. **Coverage 是有效的性能代理指标**：更高的 coverage 意味着探索了更多高概率的合理延续，直接关联到下游任务性能（Figure 5）。
2. **DLE 显著提升了算法和系统效率**：
   - **算法上**：通过确定性枚举，最大化单位计算预算下的 coverage。
   - **系统上**：天然支持 prefix reuse，在现代推理引擎（如 SGLang）上可大幅降低延迟和内存消耗。
3. **DLE 特别适用于约束性任务**：在数学、代码等正确答案稀疏但高概率路径集中的场景中，DLE 的优势最为明显。
4. **早停机制实用且高效**：能精准识别并终止几乎必然产生重复答案的分支，几乎无额外开销。

### 方法的局限性
- **依赖截断采样器**：DLE 的有效性建立在截断分布能保留正确解的前提下。若正确解位于低概率尾部（low-probability tails），DLE 可能无法覆盖。
- **确定性探索的潜在偏差**：完全确定性策略可能错过某些低概率但高质量的路径，缺乏随机性带来的“探索广度”。
- **树结构爆炸风险**：在截断较宽松（如 p 较大）时，解码树可能迅速膨胀，导致 DLE 探索不完整。

### 未来工作方向
1. **学习更优的分支策略**：结合任务相关的质量信号（如中间步骤正确性）动态调整探索顺序。
2. **混合探索机制**：在高概率区域使用 DLE，在低概率尾部引入随机采样，实现探索与利用的平衡。
3. **训练-推理协同优化**：从树视角出发，设计支持更细粒度信用分配（credit assignment）的训练目标，使模型在分支点更具区分度。
4. **适应性截断**：动态调整截断阈值（如 p 或 ε），根据上下文决定探索范围。

---

> 📌 **总结**：  
> DLE 提供了一种**高效、系统友好、即插即用**的 test-time inference 范式，通过将生成过程显式建模为树并进行确定性枚举，从根本上解决了 self-consistency 中的重复采样问题。其实验结果充分证明了其在提升性能和降低推理成本方面的双重优势，为 LLM 的高效推理提供了新的设计范式。

</details>

---

### 12. [EvoAgent: An Evolvable Agent Framework with Skill Learning and Multi-Agent Delegation](https://arxiv.org/abs/2604.20133)

**Authors**: Aimin Zhang, Jiajing Guo, Fuwei Jia, Chen Lv, Boyu Wang, Fangzheng Li  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.20133v1  

#### Abstract
This paper proposes EvoAgent - an evolvable large language model (LLM) agent framework that integrates structured skill learning with a hierarchical sub-agent delegation mechanism. EvoAgent models skills as multi-file structured capability units equipped with triggering mechanisms and evolutionary m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EvoAgent: An Evolvable Agent Framework with Skill Learning and Multi-Agent Delegation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前 Large Language Model (LLM) Agent 在复杂专业领域（如外贸）面临以下挑战：

- **技能获取依赖人工构建**：主流方法依赖手动编写“Skill”（技能），成本高、扩展性差，且难以保证质量一致性。
- **人机认知不一致**：人类设计的工作流可能不符合 LLM 的推理模式，导致性能下降。
- **缺乏系统性自我进化机制**：现有自优化方法多集中在 prompt 或单个工具层面，无法支持多文件、结构化技能包的迭代生成与验证。
- **多 Agent 架构静态固化**：多数框架采用固定角色和路由策略，缺乏基于经验动态演化的协作逻辑。
- **技能学习与任务调度脱节**：技能积累与全局任务分解之间存在结构性割裂。

### 提出了什么新方法或新思路

本文提出 **EvoAgent** —— 一种可演化（evolvable）的 LLM Agent 框架，融合了**结构化技能学习**与**分层子 Agent 委派机制**，其核心思想源自 **Harness Engineering** 范式，强调通过外部控制层来“驾驭”模型能力而非仅提升模型本身。

#### 四大核心创新点：

1. ✅ **结构化技能表示方法（Structured Skill Representation）**
   - 将技能建模为包含 `workflow instructions`、`executable scripts`、`domain references` 和 `evolutionary metadata` 的多文件单元。
   - 支持懒加载（lazy-loading）引用机制和元数据追踪（如使用次数、成功率），实现持久化存储与按需调用。

2. ✅ **用户驱动的技能自演化机制（User-in-the-loop Skill Self-Evolution）**
   - 不依赖专家标注或真实监督信号，而是通过记录技能的 `usage_count` 和 `execution_success_rate` 驱动闭环优化。
   - 实现无需显式反馈的持续能力增长，缓解人机认知错配问题。

3. ✅ **分层子 Agent 委派架构（Hierarchical Sub-Agent Delegation）**
   - 主 Agent 可按需创建具有独立上下文空间的专用子 Agent，执行特定子任务。
   - 缓解上下文窗口压力，降低认知负荷，提升并行处理能力和系统鲁棒性。

4. ✅ **三层记忆系统 + 对话压缩机制（Three-Layer Memory + Dialogue Compression）**
   - 构建 `SOUL.md` / `USER.md` / `MEMORY.md` 三层记忆体系，支持长期上下文保留与跨会话知识累积。
   - 引入资产索引（asset index）优先保留关键结构信息（如链接、数据表、技能引用），在压缩时维持推理完整性。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

- 数据来源于部署环境中的 **664 条高质量多轮对话样本**，覆盖典型外贸业务场景。
- 最终选取 **20 个样本（共 172 个评估实例）** 进行测试，每个样本含 8–9 轮交互。
- 查询由参数化脚本生成，遍历产品类别、目标市场、买家类型及 **12 种预定义外贸场景**（如市场分析、报价响应、清关、支付风险等），确保分布代表性。

> ⚠️ 注：数据虽为合成生成，但在实际部署环境中采集，分布贴近真实使用情况；未偏向成功触发技能的案例，避免偏差。

### 实验设置和评估指标

#### 评估框架：两层结构

| 层级 | 内容 |
|------|------|
| **基础文本指标** | 字符数统计、长度比、ROUGE-L 相似度 |
| **核心评估方式** | **LLM-as-Judge** 多维度评分 |

#### 五维 LLM-as-Judge 评分标准（Likert 1–5 分）

| 维度 | 描述 |
|------|------|
| `professionalism` | 术语正确性、逻辑严谨性 |
| `accuracy` | 事实/政策/数据准确性 |
| `completeness` | 是否覆盖查询所有关键方面 |
| `practicality` | 回应是否具备可操作性 |
| `language_quality` | 流畅性、格式规范、可读性 |

最终得分取五个维度平均值，并进行意图类型与任务难度分组分析。候选回应顺序随机打乱以消除位置偏见。

### 基线方法对比

| 方法 | 描述 |
|------|------|
| **Baseline** | 单独调用 GPT-5.2，无任何 Agent 框架支持 |
| **EvoAgent + GPT-5.2** | 将 GPT-5.2 接入 EvoAgent 框架后的增强版本 |
| **EvoAgent + GPT-4.1** | 替换底层模型为 GPT-4.1，测试迁移性 |
| **EvoAgent + Qwen3.5-35B-A3B** | 替换为开源本地部署模型，考察低成本可行性 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（GPT-5.2 vs. EvoAgent + GPT-5.2）

| 维度 | GPT-5.2 | EvoAgent + GPT-5.2 | 提升幅度 |
|------|--------|---------------------|----------|
| `professionalism` | 2.703 | **4.762** | +2.059 (**↑76.2%**) |
| `accuracy` | 2.907 | **4.238** | +1.331 (**↑45.8%**) |
| `completeness` | 4.331 | 4.215 | -0.116 （基本稳定） |
| `practicality` | 3.744 | **4.709** | +0.965 (**↑25.8%**) |
| `language_quality` | 4.052 | **4.779** | +0.727 (**↑17.9%**) |
| **Total/Avg** | **3.547** | **4.541** | **↑27.998% (~28%)** |

✅ **总体平均分提升近 28%**，尤其在 **专业性** 和 **准确性** 上表现显著飞跃。

### 与其他模型集成的效果对比（RQ2）

#### 图2：集成前后性能变化
- 所有模型在接入 EvoAgent 后均表现出不同程度的能力放大效应。
- **GPT-5.2 提升最明显**，说明其强推理能力能更好适配 EvoAgent 的结构化流程。

#### 图3：相对 GPT-5.2 的性能占比（集成后）
| 模型 | 相对性能 |
|------|---------|
| GPT-4.1 | ~75% |
| Qwen3.5-35B-A3B | ~74.5% |

🔍 发现：
- GPT-4.1 和 Qwen 在接入 EvoAgent 后出现轻微性能下降（分别下降约13%、15%），表明 **并非所有模型都能受益于该架构**。
- 性能不仅取决于模型自身能力，还受 **prompt 兼容性、指令遵循稳定性、上下文压缩策略** 等因素影响。

> 📌 结论公式化表达：
> ```
> Model_capability = γ₁ × Model_Inference_Capability + γ₂ × Agent_Implementation_Compatibility
> ```

### 消融实验（隐含分析）

虽然文中未明确列出消融实验表格，但从设计中可推断出以下关键组件的作用：

| 组件 | 功能验证 |
|------|--------|
| **三阶段技能匹配**（Keyword → Embedding → LLM） | 平衡效率与准确率，前两阶段快速筛选，第三阶段兜底语义理解 |
| **异步离线进化循环**（Offline Evolution Loop） | 避免实时服务干扰，支持深度会话回顾与技能提取 |
| **技能成熟度评估机制** | 将技能分为 Budding / Growing / Mature / Proficient 四级，指导后续优化与剪枝决策 |

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **EvoAgent 显著提升了 LLM 在专业领域的实用性和可靠性**  
   - 在真实外贸场景下，集成 EvoAgent 后 GPT-5.2 的综合评分提升 **约 28%**，尤其在 `professionalism` 和 `accuracy` 上优势突出。

2. ✅ **Agent 架构与底层模型存在协同效应（synergy）**  
   - 高性能模型（如 GPT-5.2）更能发挥 EvoAgent 的结构化优势；而较弱模型可能因提示风格不匹配反而性能下降。
   - 表明：**Agent 系统性能 ≠ 仅由模型决定，更依赖模型与架构之间的兼容性**。

3. ✅ **Harness Engineering 是构建可靠 Agent 系统的有效范式**  
   - 通过将在线执行（online execution）与离线演化（offline evolution）分离，实现了“确定性执行”与“可持续进化”的工程平衡。

4. ✅ **无需显式标注即可实现技能自演化**  
   - 利用隐式反馈（使用频率、成功率）驱动技能优化，降低了对专家干预的依赖，适合工业落地。

### 方法的局限性

| 限制 | 说明 |
|------|------|
| 🔒 **记忆不可逆丢失** | 压缩机制会导致早期对话细节永久丢失，缺乏自动修剪机制 |
| 👤 **用户画像更新浅层** | 更新依赖启发式阈值，缺乏深层语义推理能力，不支持跨用户知识共享 |
| ⚙️ **仅支持单技能调用** | 当前不支持多技能编排或多任务并行执行 |
| 🛠️ **工程监控缺失** | 缺少成本跟踪、仪表盘监控、企业级认证（JWT/OAuth）等功能 |
| 🔄 **进化延迟** | 离线循环仅在会话结束后触发，无法实现实时适应（real-time adaptation） |

> ❗ 根本瓶颈：**在线执行环与离线演化环为单向耦合**，无法根据运行时退化即时调整策略。

### 未来工作方向

1. **实现双向闭环进化机制**：让离线演化结果能实时注入在线执行过程，支持动态适应。
2. **开发多技能协同引擎**：支持多个 Skill 的组合调用与并行执行。
3. **引入智能记忆管理**：结合重要性评分实现自动记忆保留/淘汰机制。
4. **增强跨用户知识迁移**：在隐私合规前提下探索组织级知识沉淀。
5. **完善企业级运维能力**：增加成本计量、安全审计、权限控制等生产特性。

---

> 💡 **总结一句话**：  
> EvoAgent 提供了一个面向真实商业场景的、可持续进化的 LLM Agent 构建范式，它不是简单地“用更好的模型”，而是通过 **Harness Engineering** 实现“用好模型”，为下一代自主 Agent 系统提供了坚实的工程路径。

</details>

---

### 13. [HiPO: Hierarchical Preference Optimization for Adaptive Reasoning in LLMs](https://arxiv.org/abs/2604.20140)

**Authors**: Darsh Kachroo, Adriana Caraeni, Arjun Prasaath Anbazhagan, Brennan Lagasse, Kevin Zhu  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.20140v1  

#### Abstract
Direct Preference Optimization (DPO) is an effective framework for aligning large language models with human preferences, but it struggles with complex reasoning tasks. DPO optimizes for the likelihood of generating preferred over dispreferred responses in their entirety and lacks the granularity to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HiPO: Hierarchical Preference Optimization for Adaptive Reasoning in LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

传统的 **Direct Preference Optimization (DPO)** 虽然在对齐大语言模型（LLMs）与人类偏好方面高效且稳定，但其将整个响应视为一个整体进行优化，缺乏对复杂推理任务中不同阶段（如问题理解、中间推理、答案生成）的细粒度控制。这导致：

- 无法针对性地优化特定推理环节（例如模糊查询的理解或复杂步骤的逻辑组织）
- 难以适应不同类型的问题需求（如需强规划 vs 强表达）

此外，现有的分层推理方法（如 ReMA）依赖多智能体强化学习（RL），虽然结构化能力强，但训练不稳定、计算开销大。

---

### 🆕 提出的新方法：HiPO（Hierarchical Preference Optimization）

HiPO 是一种扩展 DPO 的新框架，通过**将响应分解为三个语义明确的段落**，实现细粒度的偏好优化：

1. **Rq (Refined Query)**：重述并澄清原始问题，增强上下文理解  
2. **Mt (Meta-thinking)**：逐步推理过程，体现结构化思维  
3. **A (Answer)**：最终答案输出  

在此基础上，HiPO 构建了一个加权的损失函数：

$$
L(\theta) = \sum_{k \in \{Rq, Mt, A, y\}} w_k \cdot L_k(\theta)
$$

其中每个 $ L_k $ 是对应段落的 DPO-style 损失，权重 $ w_k $ 可调节训练重点。

---

### 🔍 相比现有方法的优势

| 维度 | DPO / KTO / RSO | ReMA / Tree of Thoughts | HiPO |
|------|------------------|--------------------------|-------|
| **训练稳定性** | 高（单阶段、无 RL） | 低（依赖 PPO 等 RL 方法） | ✅ 高（继承 DPO） |
| **推理结构化能力** | 低（整体优化） | 高（多代理/搜索机制） | ✅ 高（显式分段） |
| **可调控性** | 无 | 中等 | ✅ 支持按段落加权训练 |
| **效率与部署友好性** | 高 | 低 | ✅ 单次前向传播生成三段 |

> **核心优势**：HiPO 成功融合了 DPO 的训练稳定性与分层推理的结构性，在不引入多阶段推理或复杂架构的前提下实现了“可控”的推理能力提升。

---

## 2. 核心实验方法和设置

### 📚 数据集

- 主要使用来自 **Math Stack Exchange** 的偏好数据集（经 GPT-4.1 处理成 Rq/Mt/A 结构）
- 包含技术性强、需要多步推理的数学问答对
- 利用 GPT-4.1 将原始 response 分解为三个标准化 segment，并保持原有偏好关系

---

### ⚙️ 实验设置

#### 模型基础
- **Qwen2.5-7B-Instruct**
- **Llama-3.1-8B-Instruct**

#### 训练方式
- 实现 HiPO 的两种训练策略：
  1. **Individual Training**：单独优化某一 segment（Rq-only / Mt-only / A-only）
  2. **Stepwise Training**：顺序调整权重（Rq → Mt → Rq+Mt），模拟渐进式能力构建

#### 超参数配置示例（Stepwise）
| 阶段 | wt.Rq | wt.Mt | wt.A | wt.y | lr | epochs |
|------|-------|--------|------|------|-----|--------|
| Rq-bias | 0.60 | 0.15 | 0.15 | 0.10 | 1e-5 | 5 |
| Mt-bias | 0.20 | 0.50 | 0.20 | 0.10 | 8e-6 | 5 |
| Rq+Mt-bias | 0.35 | 0.30 | 0.15 | 0.25 | 5e-6 | 5 |

> 使用 AdamW 优化器，序列长度 512，参考模型冻结

---

### 📊 评估指标

#### 定量指标（Benchmark Accuracy）
- **GSM8K**：小学级应用题
- **Minerva**, **AIME24**, **Gaokao2023**, **MATH500**：涵盖中学到竞赛难度的数学题
- 报告“Final Answer Correctness”

#### 定性评估（LLM-as-a-Judge）
使用 **GPT-4.1** 对生成结果打分（0–10），维度包括：
- **Coherence**：逻辑连贯性、结构组织、符号一致性
- **Accuracy**：事实正确性、领域知识、推理有效性、答案准确
- **Goal Completion**：策略有用性、进展感知、部分成功识别

---

### 🆚 基线方法对比

| 基线 | 描述 |
|------|------|
| **Base** | 未微调的基础指令模型 |
| **Standard DPO** | 在完整 response 上执行标准 DPO |
| **Ablations** | 如仅用某一段落 loss 或合并段落得分 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（+ 表示相对 Base 提升）

#### ✅ Qwen2.5-7B-Instruct（Stepwise Training）

| 方法 | GSM8K | MATH500 | AIME24 | 平均提升 |
|------|--------|---------|--------|----------|
| Base | 76.20% | 60.07% | 4.33% | — |
| DPO | +1.64pp | +0.85pp | -3.67pp | 微弱正向 |
| **HiPO-Rq+Mt-bias** | **+13.89pp** | **+6.43pp** | -1.00pp | **+4.2% avg** |

> 在 GSM8K 上达到 **90.09%** 准确率，远超 DPO 和 Base

#### ✅ Llama-3.1-8B-Instruct（Stepwise Training）

| 方法 | GSM8K | AIME24 | Gaokao |
|------|--------|--------|--------|
| Base | 81.80% | 7.33% | 46.03% |
| DPO | -0.45pp | -1.66pp | -1.30pp |
| **HiPO-Rq-bias** | +1.47pp | +1.67pp | +3.16pp |
| **HiPO-Mt-bias** | +1.75pp | -7.33pp | +1.93pp |

> 最佳为 Rq-bias，平均提升 **+1.83%**

---

### 🔬 消融实验结果（Individual Training）

#### Qwen 模型表现
| 方法 | 平均提升 | 最高增益（GSM8K） |
|------|----------|------------------|
| **Rq-only** | **+4.46%** | **+11.18pp** |
| Mt-only | +0.98% | +3.80pp (Minerva) |
| A-only | **-6.26%** | -8.36pp (MATH500) |

> ❗ A-only 导致显著性能下降，说明单纯优化答案质量反而损害整体推理

#### Llama 模型表现
| 方法 | 平均提升 | 最高增益（AIME24） |
|------|----------|--------------------|
| Mt-only | **+1.57%** | **+11.00pp** |
| Rq-only | -1.96% | +0.50pp (Minerva) |
| A-only | -2.10% | -4.00pp |

> 不同模型对 segment 权重敏感度不同：Qwen 更受益于 Rq 优化，Llama 更依赖 Mt

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **分段优化显著优于整体 DPO**
   - 特别是在 **GSM8K** 等基础推理任务上，HiPO 实现高达 **+13.89%** 的绝对提升
   - 表明对推理流程的结构化解构能有效引导模型学习

2. **不同模型架构偏好不同的训练路径**
   - Qwen 受益于先强化 **Rq（问题重构）**，再加强 **Mt（推理链）**
   - Llama 更适合直接聚焦 **Mt（元思考）**
   > → 显示出 HiPO 的灵活性：可根据模型特性定制训练策略

3. **仅优化最终答案（A-only）有害**
   - 导致中间推理退化，整体性能下降
   - 验证了“只看结果”不利于培养可靠推理能力

4. **HiPO 提升不仅是准确率，更是推理质量**
   - GPT-4.1 评分显示 HiPO 模型在 Coherence、Goal Completion 上全面领先（见附录图2–5）
   - 回答更具条理、错误更鲁棒、策略更合理

---

### ⚠️ 局限性

1. **依赖高质量 segment 分解**
   - 当前依赖 GPT-4.1 进行数据预处理，可能引入偏差
   - 自动分割尚未完全解决

2. **权重设计仍需人工干预**
   - 当前采用固定或手动调度的权重矩阵，未来可探索自动化课程学习（curriculum learning）

3. **集中在数学领域**
   - 是否泛化至其他复杂推理任务（如代码、科学推理）尚待验证

---

### 🔮 未来工作方向

1. **自动 segment detection 与 alignment**
   - 开发无需外部标注即可识别 Rq/Mt/A 的轻量模块

2. **动态权重分配机制**
   - 根据输入问题类型自适应调整各 segment 的训练强度（如模糊问题 → 加大 Rq 权重）

3. **拓展至 Retrieval-Augmented Generation**
   - 将 HiPO 应用于 RAG 场景，分别优化 query reformulation、evidence reasoning、answer synthesis

4. **跨任务迁移研究**
   - 探索在数学中训练出的分层推理能力是否可迁移到逻辑推理、程序合成等领域

---

> 💡 **总结一句话**：  
> **HiPO 通过将 DPO 扩展为层级化的 segment-wise 优化，在保持训练效率的同时实现了对 LLM 推理过程的精细控制，是迈向“可解释、可调试、可进化”推理模型的重要一步。**

</details>

---

### 14. [FSFM: A Biologically-Inspired Framework for Selective Forgetting of Agent Memory](https://arxiv.org/abs/2604.20300)

**Authors**: Yingjie Gu, Bo Xiong, Yijuan Guo, Chao Li, Xiaojing Zhang, Liqiang Wang, Pengcheng Ren, Qi Sun, Jingyao Ma, Shidang Shi  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.20300v1  

#### Abstract
For LLM agents, memory management critically impacts efficiency, quality, and security. While much research focuses on retention, selective forgetting--inspired by human cognitive processes (hippocampal indexing/consolidation theory and Ebbinghaus forgetting curve)--remains underexplored. We argue t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FSFM: A Biologically-Inspired Framework for Selective Forgetting of Agent Memory

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Large Language Model (LLM) Agents** 在长期运行中面临的四大核心挑战：
- **资源约束**：内存持续增长导致存储与计算开销激增；
- **记忆质量下降**：大量无意义、重复内容（如问候语）降低检索效率；
- **信息过时**：用户偏好、事实知识随时间变化，旧记忆失效；
- **安全与隐私风险**：恶意输入、敏感数据（PII）、违规内容被永久保留，违反 GDPR 等法规。

传统“全量记忆”范式已不可持续，而现有研究多聚焦于记忆保留与检索，**选择性遗忘机制尚未系统化建模**。

---

### 提出的新方法与新思路
作者提出 **FSFM (Framework for Selective Forgetting of Memory)** ——一个受生物认知启发的 LLM Agent 记忆选择性遗忘框架，其核心创新在于：

#### （1）神经科学驱动的设计理念
- 借鉴 **Hippocampal Memory Indexing/Consolidation Theory** 构建分层记忆架构（Sensory, Working, Long-Term Memory Layers），模拟人类信息筛选流程。
- 引入 **Ebbinghaus Forgetting Curve** 设计基于时间衰减的记忆保留函数，并扩展为多维动态模型。
- 融合 **Synaptic Pruning** 和 **Memory Reconsolidation** 机制，实现低价值连接的主动修剪与高价值记忆的强化更新。

#### （2）三维统一分析框架
从三个维度系统定义“遗忘”的目标：
- **Efficiency**：通过智能剪枝优化资源使用；
- **Quality**：动态淘汰冗余、低质内容，提升信噪比；
- **Security**：主动清除危险、敏感及合规风险内容。

#### （3）可配置的选择性遗忘策略分类
建立清晰的遗忘机制分类体系：
| 类型 | 触发方式 | 应用场景 |
|------|--------|--------|
| Passive Decay-Based | 时间衰减自动触发 | 通用老化内容清理 |
| Active Deletion-Based | 显式指令触发 | 用户请求删除、“Right to be Forgotten” |
| Safety-Triggered | 安全规则匹配触发 | 危险内容即时隔离 |
| Adaptive Reinforcement-Based | 使用反馈动态调整 | 高频/高相关性内容保活 |

#### （4）模块化系统架构设计
提供完整的工程实现方案，包含四大组件：
- `UltraSafeMemoryManager`：保障内存安全；
- `ImportanceScoringEngine`：多维评分引擎；
- `SelectiveForgettingMechanism`：优先级驱动的遗忘执行器；
- `PerformanceBenchmarkingTool`：纳秒级性能监控工具。

---

### 相比现有方法的优势
| 维度 | 传统方法 | FSFM |
|------|--------|------|
| 忘记逻辑 | 固定周期清理或不分青红皂白地截断 | 多因素加权的重要性评分驱动 |
| 生物合理性 | 缺乏理论支撑 | 深度融合认知神经科学原理 |
| 安全控制 | 被动防御 | 主动识别并优先清除危险内容（-10 分惩罚） |
| 可解释性 | 黑箱操作 | 支持审计日志与策略追溯 |
| 工程落地性 | 多为算法原型 | 提供完整 API 接口与部署建议 |

---

## 2. 核心实验方法和设置

### 数据集
实验基于中国移动“灵犀”营销服务智能助手的真实业务数据，总计 **336万条交互记录**（2025年8月–2026年3月）。采用“垂直+水平”双维度采样策略以确保泛化能力：

| 采样类型 | 来源 | 数量 | 特点 |
|--------|-----|------|------|
| Vertical Sampling | 广东省全量数据 | 443,902 条 | 区域深度行为分析 |
| Horizontal Sampling | 全国31省抽样 | 433,686 条 | 跨区域多样性验证 |

此外，引入 **NVIDIA Aegis-1.0 开源数据集中的 1,000 条危险内容样本**（涵盖仇恨言论、色情、暴力等13类），用于安全性测试。

---

### 实验设置
#### 内存管理配置
| 系统 | 存储容量限制 | 是否启用遗忘机制 |
|------|--------------|------------------|
| FSFM Framework | 70% 容量上限 | 是（按重要性分数裁剪） |
| Baseline System | 无限容量 | 否（全部保留） |

#### 重要性评分公式
$$
\text{Importance Score} = \alpha \cdot CQA + \beta \cdot BVE + \gamma \cdot TRS + \delta \cdot SRC
$$
其中各维度定义如下：
- **CQA (Content Quality Assessment)**：响应完整性（0–3分）
- **BVE (Business Value Evaluation)**：工具业务价值（0–3分）
- **TRS (Temporal Relevance Scoring)**：时效性得分（指数衰减函数）
- **SRC (Security Risk Classification)**：安全扣分项（危险内容 -10，敏感内容 -2）

最终权重设定为：$\alpha=0.4$, $\beta=0.3$, $\gamma=0.2$, $\delta=0.1$

#### 基线方法对比
- **Baseline**: “Remember Everything”策略，不进行任何遗忘；
- **Random Forgetting**: 随机删除超出容量的内容；
- **Old-First Forgetting**: 按时间顺序删除最老的记忆；
- **FSFM (Ours)**: 基于重要性评分的智能遗忘。

---

### 评估指标
| 维度 | 指标 |
|------|------|
| Memory Efficiency | 平均存储占用率、容量利用率 |
| Retrieval Performance | 查询延迟（Latency）、吞吐量（Throughput） |
| Security Control | 危险内容留存率、敏感信息留存率 |
| Content Quality | 高价值内容保留率、通用内容清除率 |
| 统计显著性 | 10轮重复实验，p < 0.001 判定为显著差异 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（Vertical Dataset）

| 维度 | 指标 | FSFM | Baseline | 改进幅度 | p-value |
|------|------|-------|----------|---------|--------|
| Memory Efficiency | 平均存储使用 | 70% cap | 100% | ↓30.0% | <0.001 |
| Retrieval Performance | 平均查询延迟 | 8.56±0.21s | 11.12±0.35s | ↓30.0% | <0.001 |
| | 吞吐量 | 58.5±1.2 q/min | 45.0±1.5 q/min | ↑30.0% | <0.001 |
| Security Control | 危险内容留存率 | 0.0% | 100.0% | ✅100%消除 | <0.001 |
| | 敏感内容留存率 | 54.1% | 100.0% | ↓45.9% | <0.001 |
| Content Quality | 高价值内容保留率 | 70.4% | 100.0% | （合理折损） | <0.001 |
| | 一般内容留存率 | 99.99% → 0% | 100.0% | 几乎完全清除 | 0.35 |

> 注：高价值内容虽有约29.6%被裁剪，但这是在严格容量约束下的帕累托最优平衡。

---

### 与基线方法对比结果（Figure 5）

| 指标 | Random Forgetting | Old-First Forgetting | FSFM |
|------|------------------|---------------------|------|
| Memory Efficiency | 中等 | 较差 | ✅ 最优 |
| Processing Speed | 慢 | 中等 | ✅ 最快（↑1.31x） |
| Content Retention Accuracy | 差（误删高频内容） | 中（误删历史关键信息） | ✅ 最高（精准保留高价值项） |
| Computational Overhead | 高（随机扫描） | 高（排序开销） | ✅ 更低（优先队列优化） |

**结论**：FSFM 在所有维度上全面超越两种启发式遗忘策略。

---

### 消融实验与参数调优
进行了系统的超参数搜索：
- **Decay Rate λ**：对 TRS 设置不同衰减速率，发现 λ=0.05（长期知识）与 λ=0.2（上下文）组合最优；
- **Pruning Batch Size**：每次裁剪 5%/10%/20%，发现 **10%** 在稳定性与效率间达到最佳平衡；
- **Score Weights**：网格搜索确认 α=0.4 时内容质量权重最大，符合业务需求。

10次独立运行的标准差均小于均值的2%，表明结果高度稳定。

---

## 4. 关键结论和发现

### 主要发现
1. **“Forget to Remember More” 成立**：  
   通过战略性遗忘低价值信息，反而能更有效地保护高价值记忆，实现“越忘越多”的悖论式收益。

2. **遗忘是效率、质量与安全的协同优化手段**：  
   不再是被动清理机制，而是主动调控系统状态的核心能力。

3. **生物启发机制具有强工程适用性**：  
   Hippocampal indexing 与 Ebbinghaus 曲线可在向量数据库中有效建模，具备现实部署潜力。

4. **多维重要性评分是关键枢纽**：  
   将内容质量、业务价值、时效性和安全性统一量化，使决策可解释、可配置、可审计。

5. **规模无关性（Scale Independence）**：  
   在单省与全国数据上的表现一致，说明框架具备良好扩展性。

---

### 局限性
- **领域依赖性**：当前验证集中于电信客服场景，跨行业（医疗、金融）有效性待验证；
- **长期累积效应未知**：实验周期较短，多年运行下是否出现偏差积累尚不清楚；
- **主观用户体验缺失**：未收集真实用户的满意度反馈；
- **环境资源受限**：未能在边缘设备上完成端到端部署测试。

---

### 未来工作方向
1. **跨域迁移研究**：在医疗、教育、金融等领域验证通用性；
2. **长期演进监测**：开展年度级部署跟踪，观察遗忘策略的累积影响；
3. **用户感知建模**：引入 human-in-the-loop 机制，结合主观评价优化评分权重；
4. **强化学习优化策略**：将遗忘政策建模为 RL 任务，实现自适应进化；
5. **多模态遗忘支持**：拓展至图像、语音等非文本记忆的联合遗忘机制；
6. **Meta-Learning for Forgetting**：训练元控制器自动调节遗忘策略参数。

---

## 总结
FSFM 是首个将 **认知神经科学原理系统应用于 LLM Agent 记忆管理** 的框架，成功将“选择性遗忘”从概念转化为可实施、可度量、可验证的技术能力。其实验验证充分、架构完整、应用广泛，标志着 AI-Native Memory 系统向更高效、更安全、更人性化方向迈出关键一步。

</details>

---

### 15. [ActuBench: A Multi-Agent LLM Pipeline for Generation and Evaluation of Actuarial Reasoning Tasks](https://arxiv.org/abs/2604.20273)

**Authors**: Jan-Philipp Schmidt  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.20273v1  

#### Abstract
We present ActuBench, a multi-agent LLM pipeline for the automated generation and evaluation of advanced actuarial assessment items aligned with the International Actuarial Association (IAA) Education Syllabus. The pipeline separates four LLM roles by adapter: one agent drafts items, one constructs ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*ActuBench: A Multi-Agent LLM Pipeline for Generation and Evaluation of Actuarial Reasoning Tasks*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在解决**缺乏高质量、可复现且符合国际标准的精算领域语言模型（LLM）基准测试**的问题。现有的 LLM 研究在精算科学方面存在以下不足：
- 缺乏与 **International Actuarial Association (IAA) Education Syllabus** 对齐的英文评估任务；
- 多数研究仅限于狭窄子领域或非英语语境（如中文保险知识）；
- 自动生成的题目质量难以保证，缺乏系统性验证机制。

### 提出的新方法与创新
作者提出了 **ActuBench** —— 一个基于多智能体 LLM 的自动化生成与评估管道，其核心创新包括：

#### （1）四角色分离的 Multi-Agent LLM Pipeline
通过 adapter 将四个独立的 LLM 角色解耦：
- **Agent A（Item Drafter）**：负责从 IAA 学习目标和 Wikipedia 内容中起草题目和正确答案。
- **Agent B（Distractor Constructor）**：专门设计三个具有合理错误逻辑（misconception probe）的干扰项。
- **Agent C（Independent Verifier）**：作为**独立验证者**，不参与内容生成，仅对题干+正确答案、以及完整选项集进行两轮验证，并驱动单次修复循环（one-shot repair loop）。
- **Auxiliary Agent**：处理成本敏感型辅助任务（如 Wikipedia 摘要、主题标签），使用轻量级模型以优化总成本。

> ✅ **核心方法论差异**：强调“验证”与“生成”的角色分离，利用 LLM 在判断错误上优于自身生成的能力（verification-generation asymmetry）。

#### （2）双模态评估协议（Dual Evaluation Protocol）
- **MCQ Mode**：传统四选一选择题模式，测试模型识别最优答案的能力。
- **LLM-as-Judge Mode**：开放作答模式，由 judge LLM 对自由文本回答打分，更贴近真实推理能力。

#### （3）公开可浏览的基准平台
所有 100 道 MCQ 题目、100 道 Judge 题目及全部模型响应均发布于 [https://actubench.de/en/](https://actubench.de/en/)，支持在线查看与检索，无需代码即可审查每道题和每个模型的回答。

#### （4）首次构建 IAA-Syllabus 对齐的英文精算 LLM 基准
填补了该领域的空白，为全球精算教育与实践提供标准化评估工具。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **质量控制** | 引入独立 verifier agent 显著提升题目一致性与事实准确性（>60% 初稿被标记需修复） |
| **可扩展性** | 可按需从新学习目标动态生成新鲜题目，缓解 benchmark contamination 问题 |
| **评估深度** | 双模式评估揭示 MCQ 与 open-ended 表现之间的显著差异，避免误判真实推理能力 |
| **透明性与复现性** | 所有数据公开可视，数据库快照固定，确保结果完全可复现 |

---

## 2. 核心实验方法和设置

### 数据集来源与构造
- **题目来源**：基于 IAA 教育大纲中的学习目标（learning objectives），结合 Wikipedia 提取的知识片段自动生成。
- **最终评估集**：
  - **MCQ Benchmark**：100 道最难的选择题，来自初始 200 道题中模型平均准确率最低的一半（empirically-hardest-100）。
  - **Judge Benchmark**：100 道开放式题目，全部来自“定量计算”类难题，与 MCQ 集互斥。

> 💡 题目难度分为五类硬型 archetype：quantitative calculation, assumption sensitivity, conceptual inversion, edge case/boundary, multi-step logic。

### 实验设置
- **评估模型数量**：共 50 个 LLM，涵盖 8 家提供商：
  - Anthropic, Google, OpenAI, xAI, DeepSeek, Mistral, Cohere, 开源托管端点（Groq, Cerebras）及本地运行（Ollama）。
- **包含 reasoning-mode 变体**：如 `claude-opus-4-6:thinking` vs `claude-opus-4-6`，用于对比分析。

### 评估指标
| 模式 | 输入格式 | 输出解析 | 主要指标 |
|------|----------|---------|-----------|
| **MCQ** | 展示题干 + 四个随机排序选项 | 提取首个 A/B/C/D 字符 | 准确率（Accuracy） |
| **Judge** | 仅展示题干，要求自由作答 | 使用 judge LLM 判断最终答案是否正确 | 正确性 verdict（yes/no），并计算 Judge Accuracy |

> ⚠️ Judge Prompt 设计严格限制其行为：不得重新求解问题，只能比较提取答案与标准答案，允许小数值误差。

### 基线方法对比
- 无直接传统基线（因本工作首创），但进行了多种内部对比：
  - **不同 provider 模型间的横向排名**
  - **standard vs. reasoning-mode 同构模型配对比较**
  - **MCQ vs. Judge 排名相关性分析**

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）MCQ 基准表现
- **最高准确率**：4 个模型达到 **98%**（98/100）：
  - `claude-opus-4-6:thinking`, `gpt-5-mini`, `o3:reasoning`, `o4-mini:reasoning`
- **成本跨度巨大**：相同 98% 准确率下，总运行成本相差 **超过 10 倍**（$0.09 ~ $1.50）

#### （2）零成本/低成本模型表现惊人
- **Cerebras-hosted gpt-oss-120b（开源权重）**：
  - 成本近乎为零（<$0.01）
  - 准确率达 **97%**，仅比榜首低 1 题
- **本地运行 Gemma 4 via Ollama**：
  - 边际成本为零
  - 准确率达 **85%**，接近顶级付费模型

> 🔥 结论：**本地部署的 open-weights 模型已进入 cost-performance Pareto front**。

#### （3）Judge 模式表现
- 最高准确率仅为 **87%**，远低于 MCQ 的 98%
- 表明：当移除四选项提示后，即使是顶尖模型也面临显著挑战

### 与基线方法的对比结果
| 对比维度 | 发现 |
|--------|------|
| **MCQ vs. Judge 排名相关性** | Spearman ρ = 0.68，Kendall τ = 0.50 → 中等一致，非完全重合 |
| **Top-performing models** | 在 MCQ 上接近饱和（98%），但在 Judge 上拉开差距（~85% vs ~70%） |
| **Low-tier models** | 部分小型模型在 Judge 上反而比 MCQ 得分更高（+10–20 pp），因其可通过宽松匹配得分，而 MCQ 要求精确选择 |

### 消融实验与关键统计（Agent-C Repair Analysis）

| 验证阶段 | 初次通过率 | 修复成功率 | 总保留率 |
|--------|------------|------------|----------|
| **Item Level（题干+正确答案）** | ~40% pass | ~75% repair success | ~83% 最终可用 |
| **Distractor Ensemble** | ~52% pass | ~70% repair success | ~85% 最终可用 |

> 📌 **关键发现**：超过 **60% 的初稿题目** 和近 **一半的干扰项集合** 被独立 verifier 标记为失败，证明 self-verification 不足；但 one-shot repair loop 可修复大多数问题。

此外，**reasoning-mode 平均增益仅 +3.5 pp**，代价是约 2.5 倍成本增长，且存在高方差（如 Opus 提升 +13 pp，Gemini Flash 反而下降 2 pp）。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **独立验证至关重要**：
   - 多数 LLM 自行生成的内容存在盲点，独立 verifier（Agent C）能有效捕获 >60% 的潜在错误。
   - “验证比生成更容易” 的不对称性在精算任务中成立。

2. ✅ **本地 open-weights 模型极具竞争力**：
   - 如 Cerebras 托管的 120B 开源模型和本地 Gemma 4，在接近零成本下实现 97%/85% 的 MCQ 准确率，位于 cost-performance 曲线前沿。

3. ✅ **MCQ 会高估真实推理能力**：
   - MCQ 模式压缩了顶部模型的表现差异（天花板效应），而 Judge 模式揭示出约 10–15 pp 的真实差距。
   - **MCQ scaffold inflation** 是严重偏差，不能仅凭 MCQ 排名推断 open-ended reasoning 能力。

4. ✅ **reasoning-mode 并非万能**：
   - 平均收益有限（+3.5 pp），性价比不高；仅在最顶端任务中有明显作用。
   - 成本敏感场景应优先考虑非 reasoning-mode 或开源替代方案。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **样本量有限** | 每个 benchmark 仅含 100 题，中间排名差异可能不具备统计显著性（Wilson CI ≈ ±6 pp） |
| **知识锚定依赖 Wikipedia** | 忽略非英语监管细节（如 Solvency II）、专业文献，导致部分内容降级或缺失 |
| **语言单一** | 当前仅支持英文，无法覆盖德国 DAV 等本地化考试体系 |
| **Judge-model bias 风险** | 尽管采用 ground-truth 匹配，但仍可能存在 position bias、verbosity bias 等影响 |
| **污染风险（Contamination Risk）** | 公开发布的题目未来可能进入训练语料，威胁长期有效性 |

### 未来工作方向
1. **多语言扩展**：开发德语版 ActuBench，适配 DAV 考试生态。
2. **增强知识源**：整合专业白皮书、监管文件、行业报告作为补充事实依据。
3. **鲁棒性检验**：引入多个 judge LLM 进行交叉验证，报告 inter-judge agreement。
4. **动态轮换机制**：定期更新公开题库子集，延缓数据污染进程。
5. **更大规模 benchmark 构建**：生成 400+ 题目以提高统计效力，尤其改善中游模型区分度。

---

> 🏁 **总结一句话**：  
> *ActuBench 不仅创建了首个 IAA 对齐的英文精算 LLM 基准，更重要的是证明了——通过 multi-agent verification + dual-evaluation design，可以在低成本下生成高质量评估内容，并揭示出 MCQ 排行榜背后的“泡沫”，推动 LLM 评估向更真实、更透明的方向发展。*

</details>

---

### 16. [V-tableR1: Process-Supervised Multimodal Table Reasoning with Critic-Guided Policy Optimization](https://arxiv.org/abs/2604.20755)

**Authors**: Yubo Jiang, Yitong An, Xin Yang, Abudukelimu Wuerkaixi, Xuxin Cheng, Fengying Xie, Zhiguo Jiang, Cao Liu, Ke Zeng, Haopeng Zhang  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.20755v1  

#### Abstract
We introduce V-tableR1, a process-supervised reinforcement learning framework that elicits rigorous, verifiable reasoning from multimodal large language models (MLLMs). Current MLLMs trained solely on final outcomes often treat visual reasoning as a black box, relying on superficial pattern matching...

---

### 17. [ESGLens: An LLM-Based RAG Framework for Interactive ESG Report Analysis and Score Prediction](https://arxiv.org/abs/2604.19779)

**Authors**: Tsung-Yu Yang, Meng-Chi Chen  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.19779v1  

#### Abstract
Environmental, Social, and Governance (ESG) reports are central to investment decision-making, yet their length, heterogeneous content, and lack of standardized structure make manual analysis costly and inconsistent. We present ESGLens, a proof-of-concept framework combining retrieval-augmented gene...

---

### 18. [Fast Amortized Fitting of Scientific Signals Across Time and Ensembles via Transferable Neural Fields](https://arxiv.org/abs/2604.19979)

**Authors**: Sophia Zorek, Kushal Vyas, Yuhao Liu, David Lenz, Tom Peterka, Guha Balakrishnan  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.19979v1  

#### Abstract
Neural fields, also known as implicit neural representations (INRs), offer a powerful framework for modeling continuous geometry, but their effectiveness in high-dimensional scientific settings is limited by slow convergence and scaling challenges. In this study, we extend INR models to handle spati...

---

### 19. [Synthetic Flight Data Generation Using Generative Models](https://arxiv.org/abs/2604.20293)

**Authors**: Karim Aly, Alexei Sharpanskykh  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.20293v1  

#### Abstract
The increasing adoption of synthetic data in aviation research offers a promising solution to data scarcity and confidentiality challenges. This study investigates the potential of generative models to produce realistic synthetic flight data and evaluates their quality through a comprehensive four-s...

---

### 20. [Lifecycle-Aware Federated Continual Learning in Mobile Autonomous Systems](https://arxiv.org/abs/2604.20745)

**Authors**: Beining Wu, Jun Huang  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.20745v1  

#### Abstract
Federated continual learning (FCL) allows distributed autonomous fleets to adapt collaboratively to evolving terrain types across extended mission lifecycles. However, current approaches face several key challenges: 1) they use uniform protection strategies that do not account for the varying sensit...

---

### 21. [Cold-Start Forecasting of New Product Life-Cycles via Conditional Diffusion Models](https://arxiv.org/abs/2604.20370)

**Authors**: Ruihan Zhou, Zishi Zhang, Jinhui Han, Yijie Peng, Xiaowei Zhang  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.20370v1  

#### Abstract
Forecasting the life-cycle trajectory of a newly launched product is important for launch planning, resource allocation, and early risk assessment. This task is especially difficult in the pre-launch and early post-launch phases, when product-specific outcome history is limited or unavailable, creat...

---

### 22. [Explicit Dropout: Deterministic Regularization for Transformer Architectures](https://arxiv.org/abs/2604.20505)

**Authors**: Vidhi Agrawal, Illia Oleksiienko, Alexandros Iosifidis  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.20505v1  

#### Abstract
Dropout is a widely used regularization technique in deep learning, but its effects are typically realized through stochastic masking rather than explicit optimization objectives. We propose a deterministic formulation that expresses dropout as an additive regularizer directly incorporated into the ...

---

### 23. [Explainable AML Triage with LLMs: Evidence Retrieval and Counterfactual Checks](https://arxiv.org/abs/2604.19755)

**Authors**: Dorothy Torres, Wei Cheng, Ke Hu  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19755v1  

#### Abstract
Anti-money laundering (AML) transaction monitoring generates large volumes of alerts that must be rapidly triaged by investigators under strict audit and governance constraints. While large language models (LLMs) can summarize heterogeneous evidence and draft rationales, unconstrained generation is ...

---

### 24. [Inference Headroom Ratio: A Diagnostic and Control Framework for Inference Stability Under Constraint](https://arxiv.org/abs/2604.19760)

**Authors**: Robert Reinertsen  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19760v1  

#### Abstract
We present a simulation-based evaluation of the Inference Headroom Ratio (IHR), a dimensionless diagnostic quantity for characterizing inference stability in constrained decision systems. IHR formalizes the relationship between a system's effective inferential capacity C and the combined uncertainty...

---

### 25. [HumorRank: A Tournament-Based Leaderboard for Evaluating Humor Generation in Large Language Models](https://arxiv.org/abs/2604.19786)

**Authors**: Edward Ajayi, Prasenjit Mitra  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.19786v1  

#### Abstract
Evaluating humor in large language models (LLMs) is an open challenge because existing approaches yield isolated, incomparable metrics rather than unified model rankings, making it difficult to track progress across systems. We introduce HumorRank, a tournament-based evaluation framework and leaderb...

---

### 26. [Bootstrapping Post-training Signals for Open-ended Tasks via Rubric-based Self-play on Pre-training Text](https://arxiv.org/abs/2604.20051)

**Authors**: Chengyu Huang, Sheng-Yen Chou, Zhengxin Zhang, Claire Cardie  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.20051v1  

#### Abstract
Self-play has recently emerged as a promising paradigm to train Large Language Models (LLMs). In self-play, the target LLM creates the task input (e.g., ask a question), which it then addresses itself by producing a task output (e.g., give an answer). A reward model evaluates the output, and the rew...

---

### 27. [AFMRL: Attribute-Enhanced Fine-Grained Multi-Modal Representation Learning in E-commerce](https://arxiv.org/abs/2604.20135)

**Authors**: Biao Zhang, Lixin Chen, Bin Zhang, Zongwei Wang, Tong Liu, Bo Zheng  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.20135v1  

#### Abstract
Multimodal representation is crucial for E-commerce tasks such as identical product retrieval. Large representation models (e.g., VLM2Vec) demonstrate strong multimodal understanding capabilities, yet they struggle with fine-grained semantic comprehension, which is essential for distinguishing highl...

---

### 28. [Graph2Counsel: Clinically Grounded Synthetic Counseling Dialogue Generation from Client Psychological Graphs](https://arxiv.org/abs/2604.20382)

**Authors**: Aishik Mandal, Hiba Arnaout, Clarissa W. Ong, Juliet Bockhorst, Kate Sheehan, Rachael Moldow, Tanmoy Chakraborty, Iryna Gurevych  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.20382v1  

#### Abstract
Rising demand for mental health support has increased interest in using Large Language Models (LLMs) for counseling. However, adapting LLMs to this high-risk safety-critical domain is hindered by the scarcity of real-world counseling data due to privacy constraints. Synthetic datasets provide a prom...

---

### 29. [Where Reasoning Breaks: Logic-Aware Path Selection by Controlling Logical Connectives in LLMs Reasoning Chains](https://arxiv.org/abs/2604.20564)

**Authors**: Seunghyun Park, Yuanyuan Lei  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.20564v1  

#### Abstract
While LLMs demonstrate impressive reasoning capabilities, they remain fragile in multi-step logical deduction, where a single transition error can propagate through the entire reasoning chain, leading to unstable performance. In this work, we identify logical connectives as primary points of this st...

---

### 30. [Extending Contract Verification for Parallel Programming Models to Fortran](https://arxiv.org/abs/2604.20410)

**Authors**: Yussur Mustafa Oraji, Christian Bischof  
**Category**: cs.DC  
**Published**: 2026-04-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.20410v1  

#### Abstract
High-performance computing often relies on parallel programming models such as MPI for distributed-memory systems. While powerful, these models are prone to subtle programming errors, leading to development of multiple correctness checking tools. However, these are often limited to C/C++ codes, tied...

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
