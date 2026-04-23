# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-23 07:21:00 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Super Apriel: One Checkpoint, Many Speeds](https://arxiv.org/abs/2604.19877)

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

# Super Apriel: One Checkpoint, Many Speeds 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大语言模型在长上下文生成场景下，**Full Attention (FA)** 的推理成本成为瓶颈，尤其是在自回归解码时，KV Cache 随序列长度线性增长，导致内存和计算效率低下。传统混合架构（hybrid architectures）虽然通过部分替换为高效 Token Mixer（如 SWA、Mamba、GDN 等）提升速度，但其 **placement（每层使用哪种 mixer）是固定的**，只能提供单一的速度-质量权衡点。

这带来了以下限制：
- **无法适应异构负载**：不同任务（短提示高并发 vs 长上下文推理）需要不同的速度-质量配置。
- **缺乏运行时灵活性**：高峰期切换到高效模式、低峰期追求高质量，需部署多个独立模型。
- **任务敏感性差**：某些能力（如长程检索）对 FA 更敏感，而局部生成则不敏感，固定 placement 无法按需调整。

### 提出的新方法与思路
提出 **Super Apriel** —— 一个 **15B 参数的 supernet（超网络）**，其核心创新在于：

- **单个 Checkpoint 支持多种运行速度**：每个 Decoder 层都内置四种训练好的 Token Mixer 选项：
  - **Full Attention (FA)**
  - **Sliding Window Attention (SWA)**
  - **Kimi Delta Attention (KDA)**
  - **Gated DeltaNet (GDN)**
- **动态 Placement 切换**：可在服务时（serving time）为每个请求选择不同的 `placement`（即每层 mixer 的组合），无需重新加载权重。
- **共享参数结构**：所有 mixer 共享 FFN、Embedding 和 LayerNorm 参数，仅 mixer 模块可切换。
- **支持 Speculative Decoding**：利用同一 checkpoint 内的高效 mixer 作为 draft model，全 FA 作为 target model，实现免额外 draft 模型的 speculative decoding。

### 相比现有方法的优势
| 维度 | 传统方法 | Super Apriel |
|------|--------|------------|
| **灵活性** | 固定 placement，单一点 | 单 checkpoint 支持多 speed presets |
| **部署成本** | 多个模型需多次训练、部署、验证 | 一次训练，多个运行模式 |
| **搜索策略** | 设计时决定或后处理搜索 | 在训练后通过 surrogate 模型高效探索整个 $4^{48}$ 空间 |
| **speculative decoding** | 需要单独训练 draft model | 可直接用内部 GDN 等作为 draft |

---

## 2. 核心实验方法和设置

### 数据集
#### 蒸馏阶段（Distillation）
- 总共 **266B tokens**
- 来源于 Apriel 的预训练语料和 SFT 数据
- 特别强调高质量推理轨迹（reasoning traces），包括多步证明、结构化问题求解、有逻辑依赖的代码等
- **关键发现**：若仅用原始预训练数据进行蒸馏，会导致推理能力崩溃；必须使用高质量推理数据才能保留教师模型的能力

#### 监督微调阶段（SFT）
- 使用专门的 SFT 数据集（见 Table 2），涵盖：
  - 数学与 STEM（38.7%）
  - 编程（36.1%）
  - 工具调用、安全、通用对话等

#### 评估基准（Evaluation Suite）
分为 **开发集（dev benchmarks）** 和 **未见测试集（unseen benchmarks）**

| 类型 | 基准名称 | 描述 |
|------|---------|------|
| **开发集** | MMLU, GSM8K, MATH500, AIME24/25, FDA, SWDE, NIAH, RULER | 用于模型开发决策和 placement 优化目标 |
| **未见集** | MMLU-Pro, GPQA, HLE, LCB, T2-Bench, IFEval, AIME(NV) | 完全未参与训练/调优，用于无偏评估泛化能力 |

### 实验设置与评估指标

#### 模型架构
- 基于 **Apriel 1.6**（15B 参数，48 层，Grouped-Query Attention）
- 每层支持 4 种 mixer，总参数约 25B，实际激活 ~15B
- 初始化方式：
  - FA/SWA 权重来自教师模型
  - GDN/KDA 使用 **DIL/KIL 初始化法**，从注意力权重初始化，无需额外蒸馏起点

#### 训练流程
1. **S1: Distillation**（蒸馏）
   - 使用冻结的 Apriel 1.6 教师模型
   - 每步随机选择每层的 mixer（uniform sampling）
   - 损失函数：Activation Matching + Reverse KL + Forward KL
2. **S2: SFT**（监督微调）
   - 仅更新 mixer 权重，冻结其他参数以保持稳定参考
   - 使用 instruction-tuning 数据进行 next-token prediction
   - 采用 **targeted placement sampling**，集中训练推荐的 presets

#### 评估指标
- **质量**：各基准的平均得分（Accuracy / Exact Match / Log-likelihood）
- **速度**：Decode Throughput（tokens/s），相对于 all-FA 的 speedup
- **综合指标**：Speed-Quality Pareto Frontier

#### 基线对比模型
- **内部基线**：
  - Apriel-H1（15B，混合 Mamba）
- **外部基线**：
  - Qwen-3.5 27B（MoE + GDN）
  - Nemotron-3-Nano 30B（Mamba-MoE）
  - Falcon-H1R 7B（Mamba）
  - Nemotron-Nano 12B v2
  - OLMo-Hybrid-Think 7B

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 4 & 13）

| Preset | FA/SWA/KDA/GDN | Avg Acc | Retention | Speedup @32k |
|--------|----------------|---------|-----------|-------------|
| **all-FA** | 48/0/0/0 | 74.2 | 100% | 1.0× |
| **RegLklhd-26** | 12/26/6/4 | 71.1 | 96% | 2.9× |
| **RegLklhd-18** | 3/25/4/16 | 69.7 | 94% | 4.8× |
| **RegLklhd-13** | 0/16/13/19 | 60.2 | 81% | 6.9× |
| **RegLklhd-10** | 0/10/5/33 | 57.2 | 77% | **10.7×** |

> ✅ 所有混合 preset 均从 **同一个 checkpoint** 运行！

### 与基线方法的对比结果
- **RegLklhd-26 (2.9×)**：
  - 质量 **71.1**，远高于 Apriel-H1（58.4）和 OLMo-Hybrid（56.1）
  - 数学平均达 **88.3**，优于多数基线
- **RegLklhd-10 (10.7×)**：
  - 接近 **11 倍解码速度**，仍保持 77% 质量保留率
- **Qwen-3.5 27B**：
  - 最高质量（85.8），但速度仅为 0.55×（慢于 FA）
- **Nemotron-3-Nano 30B**：
  - 4.1× 速度，83.0 分，但模型更大（30B）

👉 **结论**：Super Apriel 在相同规模下提供了更优的 **灵活性-效率-质量平衡**。

### 消融实验结果

#### （1）Placement Ranking Stability（图 5, 6, 7）
- 在 **0.5B 开发模型** 上，placement 排名在训练早期就趋于稳定（Spearman ρ > 0.98）
- 但在 **15B 主模型** 上，前沿（Pareto frontier）上的 placement 排名波动较大，尤其在中等成本区域
- ❗ **重要发现**：小规模实验的结果 **不能直接外推到大规模模型**，必须在目标规模上验证

#### （2）Training Strategy 对比（图 8）
- 在 0.5B 模型上比较三种策略：
  - **Stochastic**（随机采样）
  - **Targeted**（循环训练推荐 presets）
  - **Hybrid**（混合）
- 结果显示：
  - **Stochastic 训练最稳健**，最终所有 placement 表现更好
  - Targeted 虽然初期提升快，但可能陷入“虚假最优”（false optima）
- 因此最终 15B 模型蒸馏阶段采用 **fully stochastic sampling**

#### （3）Context Length 对 Throughput 的影响（图 10）
- Super Apriel 的高效 presets 在更长上下文中优势显著放大：
  - 从 16K → 32K，相对 speedup 提升 **80–155%**
  - 例如 RegLklhd-10：16K 时 4.2× → 32K 时 **10.7×**
- 对比之下，外部混合模型仅提升 **5–46%**
- 原因：SWA/KDA/GDN 的状态大小固定，不受上下文长度影响，而 FA 仍有 KV Cache 增长

#### （4）Speculative Decoding 实验（图 11）
- 使用 all-GDN 作为 draft，all-FA 作为 target
- 接受率（acceptance rate）在整个成本范围内都很高
- **最快 draft（all-GDN）带来最高整体加速比**
- 说明 supernet 内部不同 placement 分布接近，适合 speculative decoding

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **单 checkpoint 可实现多速度运行**：通过动态切换 placement，可在运行时灵活选择速度-质量权衡点。
2. ✅ **吞吐优势随上下文增长而扩大**：在 32K 上下文下，最快 preset 达到 **10.7× 解码速度**，且该优势在长文本中持续增强。
3. ✅ **推荐 presets 覆盖广泛操作点**：从 2.9×（96% 质量）到 10.7×（77% 质量），满足多样需求。
4. ✅ **支持免额外 draft 的 speculative decoding**：利用内部 GDN 作为 draft，即可实现高效 speculative decoding。
5. ⚠️ **小模型上的排名稳定性不可外推至大模型**：15B 模型的 Pareto 前沿存在更高波动性，强调了在目标规模上进行 placement 优化的重要性。

### 方法的局限性
- **长程任务严重退化**：如 RULER、NIAH 等依赖全局上下文的任务，在大量使用 GDN/KDA 的 preset 上表现急剧下降。
- **代理模型假设短程交互**：cluster expansion 截断为短程（range ≤ 3），可能忽略长距离层间依赖。
- **Log-likelihood 作为代理指标**：虽与 exact-match 相关，但仍可能存在偏差。
- **线性成本模型限制**：对“稀有 mixer”（singleton）拟合不佳，排除了部分极端配置。
- **推理引擎依赖性强**：性能受 vLLM 等引擎版本和实现细节影响，存在“移动目标”风险。

### 未来工作方向
1. **强化学习微调（RL）**：计划使用 Group Relative Policy Optimization（GRPO）进一步提升推理和智能体任务表现。
2. **扩展 Mixer Vocabulary**：引入 Mamba-2、Lightning Attention 等新型 mixer，进一步优化前沿。
3. **生产级投放策略研究**：
   - 支持 per-request 动态 placement 路由
   - 开发模型瘦身（model thinning）减少 GPU 显存占用
   - 内存感知的 preset 选择策略
4. **跨教师模型泛化**：验证该方法是否适用于除 Apriel 1.6 以外的其他教师模型。
5. **大规模训练策略验证**：在 15B 规模上系统比较 stochastic vs targeted sampling 的最终效果。

---

## 附录：开源资源（已发布）
- 📦 **模型权重**：`SuperApriel-15B-Base`, `SuperApriel-15B-Instruct`, `SuperApriel-0.5B-Base`
- 💻 **训练代码**：Fast-LLM（含 distillation 与 SFT 流程）
- ⚙️ **推理服务代码**：vLLM 扩展，支持 supernet serving
- 🔍 **Placement 优化工具包**：`place-layers`（即将上线）

</details>

---

### 2. [Decoding Text Spans for Efficient and Accurate Named-Entity Recognition](https://arxiv.org/abs/2604.20447)

**Authors**: Andrea Maracani, Savas Ozkan, Junyi Zhu, Sinan Mutlu, Mete Ozay  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.20447v1  

#### Abstract
Named Entity Recognition (NER) is a key component in industrial information extraction pipelines, where systems must satisfy strict latency and throughput constraints in addition to strong accuracy. State-of-the-art NER accuracy is often achieved by span-based frameworks, which construct span repres...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Decoding Text Spans for Efficient and Accurate Named-Entity Recognition*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **span-based NER** 方法（如 PL-Marker）虽然在准确性上表现优异，但由于其需要枚举大量候选 span 并通过 marker-augmented 输入进行处理，导致推理阶段计算开销巨大，尤其是在长文本和高吞吐场景下难以部署。这限制了其在工业级应用中的实用性。

具体瓶颈包括：
- 枚举 $O(L^2)$ 数量级的候选 span。
- 在每个 encoder 层中重复处理 marker tokens，造成冗余计算。
- 高延迟和高 GFLOPs 不适合大规模或 on-device 部署。

---

### 🚀 提出的新方法：SpanDec 与 SF-SpanDec

#### **(1) Decoupled Span Processing (SpanDec)**
- **核心思想**：将 span 表示的交互计算从主 encoder 中解耦，在最后一层引入一个轻量级的 **decoder** 专门处理 span。
- 具体实现：
  - 文本 tokens 正常经过 encoder 编码一次。
  - span markers（start/end）仅在 decoder 阶段参与 cross-attention，避免在早期 encoder 层中反复传播。
  - 参数总量保持不变：移除原 encoder 的最后一层，替换为一个轻量 decoder 模块。

> ✅ 优势：显著减少冗余计算，提升推理效率，同时保留 span-based 方法对实体边界建模的能力。

#### **(2) Early Span Filtering (SF-SpanDec)**
- **核心机制**：训练一个轻量级二分类器（SF classifier），预测每个 token 是否属于任何 entity（即是否为 "O" 类）。
- 推理时：
  - 利用该分类器提前过滤掉不包含实体的 token。
  - 只保留可能构成有效 span 的候选，大幅缩减需送入 decoder 的 span 数量（实测仅保留约 15%）。

> ✅ 优势：几乎无额外开销，却能极大降低 decoder 负载，进一步优化端到端效率。

---

### 🔍 相比现有方法的优势

| 方法 | 准确性 | 效率 | 可扩展性 | 工业适用性 |
|------|--------|-------|-----------|-------------|
| Token Classification | 中等 | 高 | 高 | ✅ |
| PL-Marker (span-based) | 高 | 低 | 中低 | ❌（高延迟） |
| LLM-based (e.g., InstructUIE) | 中高 | 极低 | 低 | ❌（autoregressive + 大模型） |
| **SpanDec (ours)** | **高** | **高** | **高** | ✅✅✅ |

- **兼顾 accuracy 与 efficiency**：在多个 benchmark 上达到甚至超过 PL-Marker 的 F1 分数，同时大幅提升 throughput 和降低 GFLOPs。
- **更适合生产环境**：适用于 always-on、高并发、资源受限（如 on-device）的应用场景。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

共使用四个广泛采用的 NER benchmark：

| 数据集 | 描述 | 实体类别数 | 训练样本 | 测试样本 |
|--------|------|------------|----------|----------|
| **CoNLL++** | CoNLL03 的修正版，新闻领域 | 5 | 4.5k | 817 |
| **CrossNER** | 跨域数据集（5个领域） | 40 | 20k | 2.5k |
| **OntoNotes5** | 多文体大规模语料 | 19 | 28k | 3.2k |
| **BC5CDR** | 医学文献（化学 & 疾病） | 3 | 11k | 5.9k |

> 所有数据集均用于验证模型在不同 domain 和复杂度下的泛化能力。

---

### ⚙️ 实验设置

#### **模型架构**
- 主干 encoder 使用三种不同规模的预训练模型：
  - **MiniLM** (33M params)
  - **BERT-Base** (110M)
  - **RoBERTa-Large** (355M)
- 对于 SpanDec：移除 encoder 最后一层，加入单层 decoder，总参数量与原始模型相当。

#### **训练细节**
- 框架：PyTorch + HuggingFace Transformers + PyTorch Lightning
- 硬件：8×NVIDIA A40 GPU 训练，单卡测试
- 优化器：AdamW（weight decay=0.01）
- 学习率策略：OneCycle，warm-up ratio=0.03
- Batch size：global 64
- 新增模块学习率 ×10
- 最大 span 长度限制为 8 tokens

---

### 📊 评估指标

| 指标 | 说明 |
|------|------|
| **F1 score** | 使用 `seqeval` 库计算标准 NER F1 |
| **Throughput** | 每秒处理的样本数（samples/s），反映推理速度 |
| **GFLOPs** | 总浮点运算量，衡量计算成本 |
| **Latency** | 单样本推理延迟（图1中展示） |

---

### 🆚 基线方法对比

| 基线方法 | 类型 | 特点 |
|---------|------|------|
| **Token Classification** | Token-level tagging + BIO schema | 高效但边界识别弱 |
| **PL-Marker** | Marker-based span NER | 当前最强 span-based 方法之一 |
| **InstructUIE / GPT-NER / UniNER** | LLM-based generation | 强泛化但极慢 |
| **SplitNER / GLiNER** | Two-stage / Generalist model | 近期高效方案代表 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 3 & 5）

#### ✅ SpanDec vs. PL-Marker vs. Token Classification

| Model | Strategy | Avg F1 | Throughput (×token cls) | GFLOPs (×token cls) |
|-------|----------|--------|--------------------------|---------------------|
| MiniLM | Token Classif. | 84.7 | 1.00× | 1.00× |
| MiniLM | PL-Marker | 85.4 | 0.42× | 8.3× |
| MiniLM | **SpanDec (ours)** | **86.2** | **0.73×** | **1.5×** |
| BERT-B | Token Classif. | 86.1 | 1.00× | 1.00× |
| BERT-B | PL-Marker | 87.6 | 0.34× | 8.3× |
| BERT-B | **SpanDec (ours)** | **87.8** | **0.61×** | **1.5×** |
| RoBERTa-L | Token Classif. | 87.1 | 1.00× | 1.00× |
| RoBERTa-L | PL-Marker | 89.5 | 0.36× | 8.3× |
| RoBERTa-L | **SpanDec (ours)** | **89.4** | **0.64×** | **1.3×** |

> 💡 **结论**：
> - SpanDec 在所有 encoder 上均优于 token classification（平均 +1.8% F1）。
> - 相比 PL-Marker，F1 相当甚至略优，但 **throughput 提升达 2.7×，GFLOPs 降低最多 8.2×**。

---

#### ✅ SF-SpanDec：极致效率版本

| Model | Strategy | Avg F1 | Throughput (×token cls) | GFLOPs (×token cls) |
|-------|----------|--------|--------------------------|---------------------|
| MiniLM | SF-SpanDec | 85.8 | **0.92×** | **1.01×** |
| BERT-B | SF-SpanDec | 87.5 | **0.93×** | **1.01×** |

> 💡 **结论**：
> - 几乎恢复到 token classification 的吞吐水平（92–93%），且 GFLOPs 与之基本持平。
> - 仍保持比 token classification 更高的准确率（+1.1~1.4% F1）。
> - 是 **on-device 或边缘设备部署的理想选择**。

---

### 🔬 消融实验结果（见 Appendix C, Table 8）

研究了 encoder 与 decoder 层数的影响（MiniLM + CoNLL++）：

| 设置 | F1 | GFLOPs | 观察 |
|------|----|--------|------|
| 11 encoder + 1 decoder | 93.6 | 2.9 | 基准配置 |
| ↓ encoder 层数（固定 decoder=1） | ↓ F1 缓慢下降 | ↓ GFLOPs 下降 | 可用于资源极度受限场景 |
| 总层数恒定（encoder↓ + decoder↑） | F1 未提升 | ↑ GFLOPs 明显上升 | **增加 decoder 层数无益**

> ✅ 结论：**一个简单的单层 decoder 就足以完成 span 分类任务**，无需复杂堆叠。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **“解耦”是关键**：将 span interaction 移至 decoder 层可避免 encoder 中的冗余 marker 计算，是提升效率的核心。
2. **轻量 decoder 足够强大**：即使是一个单层 decoder，也能有效融合 span 与上下文信息，实现高性能分类。
3. **early filtering 极具性价比**：通过简单 token-level O-classifier 即可剔除大量无效 span，带来显著效率增益而几乎不影响精度。
4. **SpanDec 实现了最佳 accuracy-efficiency trade-off**：
   - 准确性媲美 PL-Marker
   - 吞吐量接近 token classification
   - 是目前最适合工业部署的 span-based NER 方案之一。

---

### ⚠️ 局限性

1. **超参数未充分调优**：作者指出当前结果基于标准配置，仍有进一步优化空间。
2. **软件实现尚有改进余地**：SF-SpanDec 的实际 throughput 达到了 token classification 的 92–93%，但未完全匹配，暗示底层实现（如 kernel 优化）仍有潜力。
3. **最大 span 长度限制为 8**：可能影响极长实体的识别（尽管数据集中极少出现）。

---

### 🔮 未来工作方向

1. **硬件适配优化**：针对特定芯片（如 mobile SoC）进一步优化 decoder 和 filtering 模块的推理效率。
2. **动态 span length 支持**：探索更灵活的 span 枚举机制以支持任意长度实体。
3. **多任务扩展**：将 SpanDec 框架推广至 Relation Extraction、Event Extraction 等其他 structured prediction 任务。
4. **结合 LLM supervision**：利用 LLM 自动生成 weak labels 来增强 SF classifier 或 span decoder 的训练。

---

## ✅ 总结一句话

> **SpanDec 通过“解耦 span 处理 + 轻量 decoder + 早筛机制”，在不牺牲 accuracy 的前提下，实现了 span-based NER 的高效推理，为工业级高吞吐与 on-device NER 提供了新的最优解。**

</details>

---

### 3. [Fast Bayesian equipment condition monitoring via simulation based inference: applications to heat exchanger health](https://arxiv.org/abs/2604.20735)

**Authors**: Peter Collett, Alexander Johannes Stasik, Simone Casolo, Signe Riemer-S{\o}rensen  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.20735v1  

#### Abstract
Accurate condition monitoring of industrial equipment requires inferring latent degradation parameters from indirect sensor measurements under uncertainty. While traditional Bayesian methods like Markov Chain Monte Carlo (MCMC) provide rigorous uncertainty quantification, their heavy computational b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Fast Bayesian equipment condition monitoring via simulation based inference: applications to heat exchanger health*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
工业设备（如换热器）的健康状态监测依赖于从**间接传感器测量**中推断**潜在退化参数**（latent degradation parameters），例如结垢阻力（fouling resistance）、泄漏率等。传统贝叶斯方法（如 **MCMC**）虽然能提供严格的不确定性量化，但由于需要对物理仿真模型进行数千次迭代评估，计算开销巨大，**无法满足实时监控需求**。

### 🚀 提出的新方法
本文提出一种基于 **Simulation-Based Inference (SBI)** 的AI驱动框架，结合**amortized neural posterior estimation**（特别是 **Sequential Neural Posterior Estimation, SNPE**），实现快速、概率化的故障诊断。

- **核心思想**：在离线阶段通过大量模拟生成 `(参数输入, 仿真输出)` 数据对，训练一个神经密度估计器（Neural Density Estimator），学习从观测数据到**完整后验分布**的映射。
- **关键技术**：采用 **Neural Spline Flow (NSF)** 架构作为密度估计器，能够捕捉复杂、可能多峰的后验分布。

### 🔍 相比现有方法的优势
| 方面 | 传统 MCMC | 本文 SBI 方法 |
|------|-----------|----------------|
| 推理速度 | 每次推理需数千次仿真调用，耗时数秒至分钟 | **训练后推理仅需毫秒级**（0.029s/次） |
| 可扩展性 | 难以部署于多资产、高频实时系统 | **支持近即时、大规模并行诊断** |
| 计算模式 | 每次推理独立运行，成本线性增长 | **训练成本一次性摊销（amortized）**，后续推理极快 |
| 模型兼容性 | 要求可微或显式 likelihood | **适用于“黑箱”仿真器（black-box simulators）**，无需显式 likelihood |

> **创新点总结**：首次将 SBI 成功应用于工业级换热器健康监测场景，实现了**高保真贝叶斯推理与实时性之间的平衡**，为数字孪生中的实时 PHM（Prognostics and Health Management）提供了可行路径。

---

## 2. 核心实验方法和设置

### 📊 数据集
- **合成数据集**：基于一个**确定性 + 随机退化机制**的换热器模型生成。
- **退化建模**：
  - **Tube Fouling**：通过 Compound Poisson Process 模拟突发性结垢事件，控制参数为到达率 `λ` 和跳跃幅度 `βf`。
  - **Internal Leakage**：建模为随时间指数增长的流体损失，参数为泄漏强度 `β`。
- **共生成 6 种场景 × 500 次噪声实现 = 3,000 条记录**
  - 包括：弱结垢、批处理停机式结垢（稀疏大跳）、锅炉给水系统结垢、轻度/重度泄漏、无故障基准。

### ⚙️ 实验设置
- **前向模型**：基于 `effectiveness-NTU` 方法构建的 JAX-JIT 编译换热器仿真器。
- **先验设置**：
  - 故障模式：分类先验 `p(none, fouling, leakage, both) = [0.4, 0.2, 0.2, 0.2]`
  - 连续参数：使用 log-normal 分布（如 `λ ~ LogNormal(log 2.0, 0.5)`）
- **SBI 训练**：
  - 使用 **50,000 组模拟数据**进行离线训练。
  - 输入：25维**summary statistics**（包括温差均值、标准差、趋势等）
  - 模型：Neural Spline Flow (NSF)，MLP conditioner，Adam 优化器。
- **MCMC 基线**：
  - 使用 NumPyro 实现，采用 NUTS + Gibbs 采样。
  - 每任务 4 chains × (150 warm-up + 75 samples) = 900 次仿真调用。

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Failure Mode Accuracy** | 正确识别故障类型的百分比 |
| **Wasserstein Distance (1D)** | 衡量 SBI 与 MCMC 后验分布之间的距离 |
| **CRPS (Continuous Ranked Probability Score)** | 评价概率预测的准确性与锐度（越低越好） |
| **Posterior Median Comparison** | 散点图对比两种方法的点估计一致性 |
| **Credible Interval Coverage** | 置信区间是否覆盖真实值 |
| **Inference Time** | 单次推理耗时（含训练摊销） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ 故障模式识别准确率（Table II）
| 场景 | MCMC 准确率 | SBI 准确率 |
|------|-------------|-----------|
| 弱结垢 | 100% | 100% |
| 批处理停机结垢 | 100% | 100% |
| 锅炉给水结垢 | 100% | 100% |
| 轻度泄漏 | 99.8% | 100% |
| 重度泄漏 | 99.6% | 100% |
| 无故障 | 98.2% | 98.6% |

> ➤ **SBI 在所有场景下均达到与 MCMC 相当甚至略优的分类性能**

#### ⏱ 推理效率（Table III & Fig. 10）
- **MCMC**：单次推理约 **2.4 秒**（Apple M4 Pro）
- **SBI**：单次推理仅 **0.029 秒**（训练后）
- **加速比**：**82× 更快**
- **盈亏平衡点**：仅需约 **6 次推理**即可收回训练成本

> ➤ **SBI 在频繁推理场景中具有压倒性优势**

#### 📈 后验质量对比
- **Wasserstein Distance**（Fig. 7）：大多数参数的距离集中在低位，表明 SBI 后验与 MCMC 高度一致。
- **CRPS**（Fig. 8）：SBI 与 MCMC 的 CRPS 分布几乎重叠，说明其概率预测质量相当。
- **Posterior Medians**（Fig. 5 & 6）：除 `λ` 外，其余参数估计高度一致；`λ` 上 SBI 后验更窄，体现 amortization-induced shrinkage 效应。

#### 🔍 特别分析：稀疏事件场景（Scenario 2 — Batch SD）
- 该场景下 `λ = 0.5`（极低发生率），观测窗口内可能仅发生 0–1 次事件。
- 结果显示：
  - 两种方法都倾向于**高估 `λ`**，因其先验中心为 2.0，而数据信息不足。
  - 但这不影响**故障模式识别**和**起始时间 T 的估计**。
  - 后验预测轨迹仍能合理复现观察到的结垢趋势。

> ➤ 表明即使在结构性不可识别（structural unidentifiability）区域，SBI 仍能支持鲁棒的下游决策。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **SBI 可以在不牺牲诊断精度的前提下，实现比 MCMC 快 82 倍的贝叶斯推理**。
2. **amortized inference 架构使得 SBI 特别适合高频、多资产的实时监控系统**。
3. 尽管某些参数（如 `λ`）在稀疏事件下难以精确识别，但**故障模式和退化趋势仍可被可靠检测**，这对 RUL 预测至关重要。
4. SBI 对“黑箱”仿真器友好，**无需访问 likelihood 或梯度信息**，适用于老旧工业系统集成。

### ⚠ 局限性
1. **依赖合成数据训练**：当前验证基于理想化模型，尚未在真实工业数据上测试。
2. **summary statistics 设计影响性能**：手工设计的 25 维特征可能丢失部分时序信息；若使用端到端嵌入网络可能进一步提升。
3. **稀疏事件下的参数偏倚**：当事件极少时，后验易受先验主导，导致系统性偏差（非方法缺陷，而是问题本身限制）。
4. **训练成本较高**：需一次性生成数万次仿真，对复杂系统可能成为瓶颈。

### 🔮 未来工作方向
1. **真实数据验证**：在实际换热器运行数据上验证框架有效性。
2. **在线/自适应训练**：引入增量学习机制应对工况漂移（distributional shift）。
3. **联合轨迹推断**：尝试直接估计潜变量轨迹 `z(t)` 而非仅参数 `θ`，提高诊断分辨率。
4. **扩展至其他设备**：应用于压缩机、泵、反应釜等多参数工业系统。
5. **集成至数字孪生平台**：作为实时 PHM 模块嵌入工业 AI 平台（如 Cognite Data Fusion）。

---

> **总体结论**：本研究成功展示了 **SBI 作为传统 MCMC 的高效替代方案**，在保持严格不确定性量化的同时，解决了工业贝叶斯推理的**实时性瓶颈**，为实现**可扩展、风险感知的智能运维系统**奠定了基础。

</details>

---

### 4. [F\textsuperscript{2}LP-AP: Fast \& Flexible Label Propagation with Adaptive Propagation Kernel](https://arxiv.org/abs/2604.20736)

**Authors**: Yutong Shen, Ruizhe Xia, Jingyi Liu, Yinqi Liu  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.20736v1  

#### Abstract
Semi-supervised node classification is a foundational task in graph machine learning, yet state-of-the-art Graph Neural Networks (GNNs) are hindered by significant computational overhead and reliance on strong homophily assumptions. Traditional GNNs require expensive iterative training and multi-lay...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：F²LP-AP: Fast & Flexible Label Propagation with Adaptive Propagation Kernel

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前图机器学习中的 **semi-supervised node classification** 面临两大瓶颈：
- **计算效率低**：主流 GNN（如 GCN、GAT）依赖梯度反向传播进行迭代训练，在大规模图上消耗大量 GPU 资源。
- **强同质性假设（homophily assumption）限制泛化能力**：传统 GNN 假设相连节点标签相似（homophilous），但在异质图（heterophilous graphs，如欺诈网络）中性能严重下降。

此外，现有的无训练方法（如 Label Propagation, LP）虽然高效，但采用固定传播规则，无法适应局部拓扑结构变化，导致在稠密区域过平滑、稀疏或异质区域噪声放大。

---

### 🚀 提出的新方法与核心思想
本文提出 **F²LP-AP**（Free-from-training Label Propagation with Adaptive Propagation Kernel），一种**无需训练、高效且自适应**的节点分类框架。其核心创新包括：

#### （1）**基于 Local Clustering Coefficient (LCC) 的自适应传播核**
- 利用每个节点的 **LCC**（局部聚类系数）作为拓扑感知指标，动态调整传播参数：
  - **传播深度 $K_i$**：高 LCC（密集社区）→ 浅层传播；低 LCC（边界/异质连接）→ 更深传播以获取上下文。
  - **teleport probability $\alpha_i$**：高同质性区域降低 $\alpha$ 促进平滑；高异质风险区域提高 $\alpha$ 保留原始特征锚点。
- 映射函数为预定义启发式规则（如 $K = \text{round}(g_K(\text{LCC}))$），**无需参数学习**。

> 🔍 公式示例：  
> $$
> \alpha_i = f_\alpha(\text{LCC}_i),\quad K_i = \text{round}(g_K(\text{LCC}_i))
> $$

#### （2）**基于 Geometric Median 的鲁棒原型构建**
- 不再使用易受异常值影响的 arithmetic mean 构建类别原型。
- 改用 **Geometric Median**（几何中位数）聚合训练样本特征，具有高达 50% 的 break-down point，显著增强对噪声和离群点的鲁棒性。

#### （3）**完全解析式的推理流程（Analytical Inference Pipeline）**
整个流程由三阶段组成，全部为确定性算法，**不涉及任何可学习参数或梯度更新**：
1. **Robust Prototype Construction** → 几何中位数
2. **Adaptive Feature Propagation** → LCC 自适应调节 $K, \alpha$
3. **Analytical Classification** → 通过 cosine similarity 匹配原型

---

### ⚡ 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **Training-free** | 完全避免训练开销，推理即完成预测 |
| **Computational Efficiency** | 推理速度远超 GNN（如在 PubMed 上仅为 GCN 的 ~3% 时间） |
| **Structural Adaptivity** | 动态适配同质与异质子图，克服“homophily bottleneck” |
| **Noise Robustness** | 几何中位数 + 自适应机制有效抑制噪声干扰 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
共选取 **8 个基准图数据集**，按同质比（homophily ratio, H）分为两类：

| 类型 | 数据集 | 描述 |
|------|--------|------|
| **强同质图 (H > 0.8)** | Cora, CiteSeer, PubMed | 引用网络，节点为论文，边为引用关系 |
| **异质图 (H < 0.4)** | Texas, Wisconsin, Cornell, Chameleon, Squirrel | WebKB 和 Wikipedia 页面，链接常跨主题 |

覆盖不同规模（数百至数万节点）、领域和拓扑特性，用于全面验证模型泛化能力。

---

### 🧪 实验设置与评估指标

#### 评估指标
- **Classification Accuracy (Acc.)**
- **Macro-F1 Score (F1)**
- **Execution Time (秒)** —— 衡量计算效率

#### 实现细节
- 所有实验基于 PyTorch，随机种子固定为 0 保证可复现性。
- F²LP-AP 参数范围：
  - 传播步数 $K \in [2, 15]$
  - teleport 概率 $\alpha \in [0.05, 0.2]$
- 所有 baseline 使用原论文最优配置。
- 硬件环境统一，确保公平比较。

---

### 🔁 基线方法对比
涵盖多种技术路线的代表性方法：

| 类别 | 方法 | 说明 |
|------|------|------|
| **Supervised GNN** | GCN* | 主流监督模型，需训练 |
| **Training-free / Non-parametric** | LabelProp, kNN@5 | 经典无训练方法 |
| **State-of-the-Art GNN-free** | CoHOp | 当前先进的无训练基线 |
| **Ablation Baselines** | PrototypeOnly-Mean, PrototypeOnly-GeoMed, FixedAPPNP-Proto | 用于消融分析 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| Dataset (H) | Method | Acc. | F1 | Time (s) |
|------------|--------|------|-----|----------|
| **Cora (0.85)** | GCN* | 0.821 | 0.809 | 1.350 |
| | **F²LP-AP** | **0.835** | **0.821** | **0.056** |
| **CiteSeer (0.81)** | GCN* | 0.719 | 0.691 | 1.210 |
| | **F²LP-AP** | **0.708** | **0.685** | **0.092** |
| **PubMed (0.84)** | GCN* | 0.798 | 0.794 | 1.485 |
| | **F²LP-AP** | **0.782** | **0.779** | **0.044** |
| **Texas (0.31)** | GCN* | 0.553 | 0.365 | 1.015 |
| | **F²LP-AP** | **0.842** | **0.787** | **0.016** |
| **Wisconsin (0.37)** | GCN* | 0.608 | 0.269 | 1.053 |
| | **F²LP-AP** | **0.825** | **0.589** | **0.024** |
| **Cornell (0.34)** | GCN* | 0.500 | 0.203 | 1.014 |
| | **F²LP-AP** | **0.763** | **0.519** | **0.018** |

> ✅ **加粗表示该类别下最佳表现（training-free 或 overall SOTA）**

---

### 🔍 与基线方法对比结果
- 在 **所有异质图（Texas, Wisconsin, Cornell）上达到 SOTA 性能**，大幅超越 GCN 和 CoHOp。
- 在 **同质图（Cora）上超越监督模型 GCN**，成为新的 overall SOTA。
- 在 **CiteSeer 和 PubMed 上为 training-free 方法中最优**。
- **运行时间极短**：平均仅需 **0.01–0.09 秒**，相较 GCN 加速数十倍（如 PubMed 上提速约 33 倍）。
- 显著优于其他无训练方法（如 LabelProp 在 Texas 上仅 13.2% 准确率，而 F²LP-AP 达 84.2%）。

---

### 🔬 消融实验结果（Ablation Study）
#### （1）自适应传播机制有效性
- 对比非自适应版本 **FixedAPPNP-Proto**（固定 $K=5, \alpha=0.1$）：
  - 在 Cora 上准确率提升 **26.9%**（从 0.658 → 0.835）
  - 在 Wisconsin 上提升 **13.8%**
- 结论：**动态参数调整对性能至关重要**，尤其在复杂拓扑中。

#### （2）原型构建方式对比（Figure 3）
- 对比 **arithmetic mean** vs **geometric median**
- 几何中位数在所有数据集上均优于或等于均值法
- 在低同质图（如 Chameleon, Squirrel）上增益更明显（+12.8%, +14.3%）
- 结论：**GeoMedian 提供更强的噪声鲁棒性和稳定性**

#### （3）可视化分析（Figure 4）
- t-SNE 可视化显示：经 F²LP-AP 处理后的特征空间中，**同类节点聚集更紧密，类间分离更清晰**，尤其是在异质图 Texas 上改善显著。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **F²LP-AP 是首个兼具高性能与高效率的 training-free 方法**，在多种图结构上媲美甚至超越监督 GNN。
2. **Local Clustering Coefficient 是有效的拓扑感知信号**，可用于实现 node-wise 自适应传播。
3. **Geometric Median 显著提升 prototype 鲁棒性**，特别适用于存在标签噪声的真实场景。
4. **无需训练也能实现优异分类效果**，挑战了“必须通过优化才能获得好表示”的传统认知。

---

### ⚠️ 方法的局限性
1. **依赖单一拓扑指标 LCC**：在极端稀疏或高度噪声的图中可能失效。
2. **启发式映射函数非数据驱动**：参数映射（如 $f_\alpha$, $g_K$）是人工设计，未从数据中学习，可能存在次优。
3. **性能受限于原始特征质量**：若输入特征本身判别力弱，则难以通过传播修复。
4. **超参仍需调优**：尽管整体稳定，但 $K_{\min}, K_{\max}, \alpha_{\min}, \alpha_{\max}$ 的选择会影响最终性能。

---

### 🔮 未来工作方向
1. 探索多维结构描述符（multi-dimensional structural descriptors）替代单一 LCC，提升适应精度。
2. 引入轻量级学习机制（lightweight learning modules）来自动学习映射函数，兼顾灵活性与免训练优势。
3. 将本框架扩展到其他任务，如 link prediction、graph clustering 或 dynamic graphs。
4. 进一步研究如何解耦结构与特征偏差，提升在极端 heterophily 场景下的表现。

---

> 💬 **总结一句话**：  
> **F²LP-AP 通过“几何中位数 + LCC 自适应传播 + 解析式分类”三部曲，在无需训练的前提下实现了高效、鲁棒、通用的图节点分类，为 training-free graph learning 开辟了新路径。**

</details>

---

### 5. [Dual-Cluster Memory Agent: Resolving Multi-Paradigm Ambiguity in Optimization Problem Solving](https://arxiv.org/abs/2604.20183)

**Authors**: Xinyu Zhang, Yuchen Wan, Boxuan Zhang, Zesheng Yang, Lingling Zhang, Bifan Wei, Jun Liu  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.20183v1  

#### Abstract
Large Language Models (LLMs) often struggle with structural ambiguity in optimization problems, where a single problem admits multiple related but conflicting modeling paradigms, hindering effective solution generation. To address this, we propose Dual-Cluster Memory Agent (DCM-Agent) to enhance per...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文针对**大型语言模型**（LLMs）在求解优化问题时面临的**多范式歧义**（multi-paradigm ambiguity）问题。具体而言，一个优化问题可能同时具备多种建模范式的特征（如整数规划 ILP、动态规划 DP、约束编程 CP），导致 LLM 在推理过程中产生认知干扰，难以选择正确的建模路径，从而生成错误或不可行的解决方案。

### 提出的新方法：Dual-Cluster Memory Agent **(DCM-Agent)**
提出了一种无需训练的框架——**DCM-Agent**，通过外部化历史经验来增强 LLM 的优化建模能力。其核心创新包括：

- **Dual-Cluster Memory Construction**：将历史解决方案分层组织为两个独立的集群：
  - **Modeling Cluster**：抽象的建模范式（如 ILP、DP）
  - **Coding Cluster**：具体的代码实现策略（如 Gurobipy 模板）
  - 构建加权二分图 $G=(V_M, V_C, E)$ 来建模建模逻辑与编码策略之间的兼容性。

- **三层次结构化知识提炼**：从每个集群的历史节点中提取三种通用指导知识：
  - **Approach**：标准求解模板（来自 Type A/B 成功案例）
  - **Checklist**：验证准则（来自成功案例）
  - **Pitfall**：常见错误警告（来自 Type B/C 失败案例）

- **Memory-Augmented Inference**：采用“检索-生成-验证-修复-回溯”（generate-verify-repair-backtrack）的动态推理流程：
  - 利用双层级检索（实例级 + 集群级）获取相关历史路径；
  - 动态调用 `Checklist` 和 `Pitfall` 进行范式特定的错误检测与修复；
  - 若当前路径失败，则自动回溯至备选路径。

### 相比现有方法的优势
- **无需微调**（training-free）：不依赖参数更新，适用于任意规模的 LLM。
- **更强的鲁棒性**：有效缓解多范式干扰，提升建模正确性。
- **可迁移的知识继承**（knowledge inheritance）：大模型构建的记忆可指导小模型达到更优性能。
- **更高的效率**：相比树搜索类方法（如 AF-MCTS），计算开销显著更低。

---

## 2. 核心实验方法和设置

### 数据集
共使用 **7 个优化基准数据集**，涵盖不同复杂度和应用场景：
- **标准基线**：NL4Opt, NLP4LP
- **高复杂度任务**：OptiBench, OptMATH, ComplexLP (from MAMO)
- **工业真实场景**：IndustryOR (IndOR), ComplexOR

记忆库（memory）基于 **500 个非交集的历史问题-解对** 构建，确保与测试集无重叠。

### 实验设置与评估指标
- **主干模型**：Qwen3 系列（8B, 30B, 235B）、DeepSeek-V3.2、GPT-5.1
- **评估指标**：**端到端求解准确率**（end-to-end solving accuracy）
  - 要求生成的代码能正确执行，并且目标函数值和约束输出均匹配真实答案。
- **允许使用的求解器库**：Gurobi, PuLP, OR-Tools, SciPy, NetworkX

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **通用 LLM** | Baseline（零样本提示） |
| **多智能体框架** | OptiMUS |
| **搜索增强方法** | AF-MCTS（蒙特卡洛树搜索）、OptiTree（自适应分解） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）
DCM-Agent 在所有模型尺度上均取得最优性能，平均准确率提升 **11%–21%**：

| 模型 | Baseline (%) | DCM-Agent (%) | 提升幅度 |
|------|--------------|----------------|----------|
| Qwen3-8B | 27.99 | 49.43 | +21.44 |
| Qwen3-30B | 40.52 | 59.61 | +19.09 |
| Qwen3-235B | 57.74 | 71.28 | +13.54 |
| DeepSeek-V3.2 | 57.61 | 71.47 | +13.86 |
| GPT-5.1 | 67.62 | 78.50 | +10.88 |

> ✅ **结论**：DCM-Agent 显著优于所有基线方法，且在小模型上增益更大。

### 时间效率对比（Table 3）
相比搜索密集型方法，DCM-Agent 具有更优的时间-性能权衡：

| 方法 | OptiBench (s) | OptMATH (s) |
|------|---------------|-------------|
| AF-MCTS | 110.8 | 205.7 |
| **DCM-Agent** | **41.3** | **73.4** |

> ⏱️ **优势**：远低于 AF-MCTS，接近 OptiTree，但准确率更高。

### 消融实验结果（Ablation Study, Figure 5）
移除任一模块均导致性能下降，验证双集群协同机制的有效性：

| 设置 | NLP4LP (%) | OptiBench (%) | OptMATH (%) |
|------|------------|----------------|-------------|
| 完整 DCM-Agent | 84.71 | 75.21 | 46.39 |
| 移除 Modeling Cluster | 77.7 | 72.1 | 39.8 |
| 移除 Coding Cluster | 81.0 | 68.6 | 42.8 |

> 🔍 **发现**：建模集群的移除造成更严重的性能损失，说明**精确的数学逻辑抽取是决定性因素**。

---

## 4. 关键结论和发现

### 主要发现
1. **知识继承现象**（Knowledge Inheritance）：
   - 更强模型（如 GPT-5.1）构建的记忆可有效提升弱模型（如 Qwen3-8B）的表现（Table 5）。
   - 表明结构化先验知识可在不同规模 LLM 间迁移，具有良好的**可扩展性**与**实用性**。

2. **内存容量影响显著**（Table 4）：
   - 性能随记忆节点数量增加而稳步上升，证实历史经验广度直接影响泛化能力。

3. **超参敏感性分析**（Table 6）：
   - 最优设置：检索 Top-K=3，知识更新阈值 N=5，候选路径数 M=3。
   - 过大的 K 或 N 反而导致性能下降（输入过长或知识固化）。

### 方法局限性
- **初始化延迟高**：记忆构建阶段需一次性处理大量历史轨迹，存在较高的初始计算开销（“沉没成本”）。
- **依赖高质量记忆源**：若历史数据质量差或覆盖不足，会影响检索效果。
- **未支持在线学习**：目前记忆为静态，无法通过交互持续演化。

### 未来工作方向
- 探索**在线学习机制**，使记忆能够动态演进；
- 扩展至更多领域（如强化学习、控制理论）；
- 研究轻量化记忆压缩技术以降低部署门槛。

---

> 📌 **总体评价**：  
> DCM-Agent 提供了一个高效、可扩展、无需训练的框架，成功解决了 LLM 在优化建模中的多范式歧义难题。其实验设计严谨，结果充分验证了方法的有效性和通用性，为自动化数学建模提供了新的研究范式。

</details>

---

### 6. [FASER: Fine-Grained Phase Management for Speculative Decoding in Dynamic LLM Serving](https://arxiv.org/abs/2604.20503)

**Authors**: Wenyan Chen, Chengzhi Lu, Yanying Lin, Dmitrii Ustiugov  
**Category**: cs.DC  
**Published**: 2026-04-23  
**Score**: 7.5  
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
现有的 **Speculative Decoding (SD)** 系统在处理动态的在线推理负载时存在显著缺陷：
- **粗粒度管理**：大多数系统对整个 batch 使用固定的 speculative token 长度，并串行执行 draft 和 verification 阶段。
- **资源利用率低**：在小 batch 或低负载下，draft 阶段阻塞 verification，导致 GPU 资源闲置；在高负载下，大量计算浪费在最终被拒绝的 speculative suffix 上。
- **无法适应动态变化**：随着请求到达率波动（可高达35×），系统的瓶颈在 draft 和 verification 之间不断切换，而现有方法缺乏细粒度响应能力。

### 🚀 提出的新方法：FASER
FASER 是一种面向动态 LLM 推理场景的新型 SD 系统，其核心是**细粒度相位管理**（fine-grained phase management），包含三大关键技术：

| 创新技术 | 功能说明 |
|--------|--------|
| **Token-wise Early Exit** | 在 verification 过程中逐 token 判断是否可能被接受，若预测为“将被拒绝”，则提前终止该 token 及其后续 speculative suffix 的验证，减少冗余计算。 |
| **Frontier-based Pipeline Overlap** | 将 verification 分解为多个 chunk（称为 frontier），允许 verification 与后续 draft 并发执行，通过 GreenContexts 实现 GPU SM 的空间复用，最小化干扰。 |
| **Online Adaptive Controller** | 基于离线性能表和运行时反馈（如 batch size、acceptance rate）动态调整 speculative length 和 GPU 资源分配（SM 分配比例），实现自适应优化。 |

### 🔍 相比现有方法的优势
| 对比维度 | 现有方法（如 SpecInfer, AdaSpec, Smurfs） | FASER |
|--------|--------------------------------------|-------|
| **Speculative Length 管理** | 批级固定或粗调 | 请求级动态调整 |
| **Verification 效率** | 全序列并行验证，无早期剪枝 | 支持 token 级 early exit，跳过无效后缀 |
| **阶段重叠粒度** | 串行或微批级流水线 | 前沿 chunk 级并发，支持细粒度 overlap |
| **资源调度** | 不显式控制 SM 分配 | 显式空间复用，GreenContexts 控制资源占比 |
| **适应性** | 静态配置为主 | 在线控制器持续优化决策 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **ShareGPT**：真实对话提示数据集，用于模拟通用聊天场景。
- **LongBench**：长上下文理解任务集合，测试长输入下的性能。
- **HumanEval**：代码生成任务，评估复杂逻辑推理能力。

### ⚙️ 实验设置
- **硬件平台**：单/双 NVIDIA H100 GPU（96GB VRAM），PCIe 4.0 互联。
- **模型组合**：
  - Draft Model: `Qwen3-0.6B`, `Llama3.2-1B`
  - Target Model: `Qwen3-32B`, `Llama3.3-70B`（后者使用 TP=2）
- **请求模式**：基于 **Azure LLM invocation trace** 构造动态流量，平均到达率为 26 req/s，具有高峰谷波动特性。
- **实现基础**：基于 **vLLM v0.15.1** 实现，新增约 5k 行 Python 代码。

### 📊 评估指标
| 指标 | 描述 |
|-----|------|
| **Latency** | 每个输出 token 的平均延迟（ms/token） |
| **Throughput** | 单位时间内生成的输出 token 数量（tokens/sec） |
| **Acceptance Rate** | 被接受的 speculative token 占总 speculative token 的比例 |
| **Early Exit Ratio** | 被 early exit 机制剪枝的 token 比例 |
| **MAPE** | 离线性能建模的平均绝对百分比误差，衡量 profiler 准确性 |

### 🆚 基线方法
- **SpecInfer**：基于树结构的 speculative inference，提升多样性但未解决串行瓶颈。
- **AdaSpec**：支持 adaptive speculative length，但仍采用 round-based verify 流程。
- **Smurfs**：引入跨 batch 的 pipeline overlap，但未实现细粒度 spatial multiplexing。

---

## 3. 主要实验结果和性能指标

### 📈 性能提升汇总
| 模型 | 指标 | 提升幅度 | 备注 |
|------|------|---------|------|
| Qwen3 系列 | Throughput | **最高 +53%** | vs. 最佳 baseline |
| Qwen3 系列 | Latency | **降低最多达 1.92×** | 即提速近一倍 |
| Llama3 系列 | Throughput | **最高 +1.49×** | 相对提升明显 |
| Llama3 系列 | Latency | **最高降低 42%** | 特别在 LongBench 上表现优异 |

> 注：所有提升均在高度动态的请求模式下取得，且优势随负载波动加剧而增强。

### 🔬 关键实验发现
#### （1）Early Exit 效果显著
- 平均 early exit 比例：
  - **ShareGPT**: ~14.5% (Qwen3), ~12.4% (Llama3)
  - **HumanEval**: ~8.8% (Qwen3), ~9.1% (Llama3)
- 最高可达 **22.8%** 的 speculative token 被提前剪枝，有效节省 verification 计算。

#### （2）动态 speculative length 更优
- FASER 根据负载自动选择 speculative length（通常在 5–8 之间），避免了固定长度带来的效率损失。
- 图 13 显示，在不同数据集上 speculative length 分布广泛，体现其自适应能力。

#### （3）消融实验（Ablation Study）
使用 Qwen3 模型对各组件进行逐步添加测试（VSD → VSD+AD → VSD+AD+EE → FASER）：

| 组件 | Latency 下降 | Throughput 提升 | 贡献分析 |
|------|-------------|------------------|----------|
| **Adaptive Drafter (AD)** | 最多 19% | 最多 1.35× | 提升 draft 效率，优化 speculation 并行性 |
| **Early Exiter (EE)** | 最多 26% | 最多 1.22× | 直接削减 verification 开销，对 latency 影响更大 |
| **Pipeline Overlapper** | — | 协同增益显著 | 与其他组件结合后，总延迟下降 **61%**，吞吐提升 **1.60×** |

> 结论：三个组件协同作用产生强正向耦合效应。

#### （4）离线 Profiler 准确性高
- **MAPE（平均绝对百分比误差）**：
  - Draft Latency: ≤17.2%
  - Target Latency: ≤7.5%
- 表明性能模型足够准确，可用于指导在线决策。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **粗粒度 SD 管理不适用于动态负载**：传统串行流程和固定 speculative length 严重制约了系统在真实场景中的性能。
2. **细粒度 phase management 至关重要**：
   - **token-wise early exit** 可有效识别并跳过 rejected suffix，大幅减少 verification 冗余。
   - **frontier-based overlap** 实现 draft 与 verification 的 chunk 级并发，提升 GPU 利用率。
3. **自适应控制是核心驱动力**：在线控制器结合历史统计与实时反馈，动态调节 speculative length 与资源分配，使系统能应对负载波动。
4. **FASER 具备良好泛化性**：
   - 成功适配到 **self-speculative decoding**（如 Medusa、EAGLE），分别实现最高 **35%** 和 **50%** 的延迟降低。
   - 在 **MoE 模型**（Qwen2-0.5B / Qwen2-57B-A14B）上仍取得 **16% 延迟下降** 和 **1.38× 吞吐提升**。

### ⚠️ 局限性
- **依赖中间层 logits 信号**：early exit 机制基于 Top-K 概率比较，需访问 draft 与 target 模型之间的交互信息，可能增加运行时开销。
- **SM 分配影响内存带宽竞争**：虽然 GreenContexts 控制 compute 资源，但共享 HBM 和 cache 层仍可能导致 memory contention。
- **当前实现在单机多卡环境**：尚未扩展至分布式或多节点场景，未来需考虑跨设备协调问题。

### 🔮 未来工作方向
1. **支持更多 SD 范式**：进一步集成 MCTS-based speculation 或 multi-drafter 架构。
2. **联合优化 Prefill 与 Decoding**：结合 DistServe 等 prefill-decode disaggregation 技术，实现端到端全流程加速。
3. **更智能的 early exit predictor**：探索轻量神经网络替代 Top-K 规则，提升预测精度。
4. **跨节点 FASER 架构设计**：在 disaggregated serving 场景中部署 FASER，支持更大规模模型推理。

---

> 💡 **一句话总结**：  
> FASER 通过 **fine-grained phase management**——即 **adaptive drafting + token-wise early exit + frontier-based overlap**——实现了对动态 LLM 推理负载的高效响应，在真实 trace 驱动下实现了高达 **53% 吞吐提升** 和 **1.92× 延迟降低**，为下一代高性能 LLM serving 系统提供了新范式。

</details>

---

### 7. [HiPO: Hierarchical Preference Optimization for Adaptive Reasoning in LLMs](https://arxiv.org/abs/2604.20140)

**Authors**: Darsh Kachroo, Adriana Caraeni, Arjun Prasaath Anbazhagan, Brennan Lagasse, Kevin Zhu  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.20140v1  

#### Abstract
Direct Preference Optimization (DPO) is an effective framework for aligning large language models with human preferences, but it struggles with complex reasoning tasks. DPO optimizes for the likelihood of generating preferred over dispreferred responses in their entirety and lacks the granularity to...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HiPO: Hierarchical Preference Optimization for Adaptive Reasoning in LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **标准 DPO 的局限性**：Direct Preference Optimization (DPO) 虽然在对齐大语言模型与人类偏好方面高效稳定，但其将整个响应视为单一单元进行优化，缺乏对复杂推理任务中不同阶段（如问题理解、中间推理、答案生成）的细粒度控制。
- **现有方法的割裂**：当前方法要么擅长稳定的偏好学习（如 DPO 及其变体），要么擅长结构化推理（如 Tree of Thoughts、ReMA），但无法同时兼顾训练稳定性与分层推理能力。

### 🚀 提出的新方法：HiPO（Hierarchical Preference Optimization）
- 将响应分解为三个语义明确的段落：
  1. **Rq (Refined Query)**：重述并澄清原始查询，增强上下文理解。
  2. **Mt (Meta-thinking)**：显式的逐步推理过程（即 Chain-of-Thought 风格）。
  3. **A (Answer)**：最终的答案输出。
- 在 DPO 框架基础上引入**分段加权损失函数**：
  $$
  \mathcal{L}(\theta) = \sum_{k \in \{Rq, Mt, A, y\}} w_k \cdot \mathcal{L}_k(\theta)
  $$
  其中每个 $\mathcal{L}_k$ 是对应段落的 DPO-style 损失，权重 $w_k$ 可调节训练重点。

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|---------|
| **细粒度优化** | 支持针对特定推理环节（如 query 理解或 reasoning 步骤）进行定向强化训练。 |
| **计算效率高** | 单 agent、单 pass 推理架构，保持 DPO 的训练稳定性与低资源消耗特性。 |
| **灵活性强** | 支持“逐级训练”（stepwise training），例如先优化 Rq → 再优化 Mt → 最后联合优化。 |
| **无需多智能体框架** | 不依赖复杂的 multi-agent RL 架构（如 ReMA），避免训练不稳定问题。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 主要使用基于 **Math Stack Exchange** 构建的偏好数据集（来源：HuggingFace `prhegde/preference-data-math-stack-exchange`）。
- 该数据集包含技术性强、需多步推理的数学问答对，适合测试复杂推理能力。

### ⚙️ 实验设置
- **基础模型**：
  - Qwen-2.5-7B-Instruct
  - Llama-3.1-8B-Instruct
- **训练细节**：
  - 使用 AdamW 优化器，初始学习率 $1\times10^{-5}$ 到 $1\times10^{-6}$。
  - 序列长度：512 tokens。
  - 参考模型（reference model）冻结，仅更新策略模型。
- **两种训练范式**：
  1. **Individual Training**：固定某一 segment 权重为 1，其余为 0（如 Rq-only、Mt-only、A-only）。
  2. **Stepwise Training**：按顺序切换权重配置（Rq-bias → Mt-bias → Rq+Mt-bias），实现渐进式训练。

### 📊 评估指标
#### 定量指标（Benchmark Accuracy）
- **GSM8K**：小学级别数学应用题。
- **MATH500**：竞赛级数学难题。
- **AIME24**：美国数学邀请赛题目。
- **Gaokao2023**：中国高考数学真题。
- **Minerva**：综合性数学推理基准。

#### 定性指标（LLM-as-a-Judge）
使用 **GPT-4.1** 对生成结果打分（0–10 分）：
- **Coherence**：逻辑连贯性、结构组织、符号一致性。
- **Accuracy**：事实正确性、领域知识、推理有效性、答案准确性。
- **Goal Completion**：策略有用性、解题进展、部分成功识别、错误鲁棒性。

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **Base** | 未经微调的基础指令模型。 |
| **Standard DPO** | 传统 DPO，直接比较完整 response 的偏好，不分解段落。 |
| **Ablations** | 包括仅优化某一段（如 A-only）、或合并段落等消融设置。 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（以 Final Answer Correctness 为准）

#### ✅ Qwen-2.5-7B-Instruct 结果（Table 4）
| 配置 | GSM8K | MATH500 | AIME24 | 平均提升 |
|-------|--------|----------|--------|------------|
| Base | 76.20% | 60.07% | 4.33% | — |
| DPO | +1.64% | +0.85% | -3.67% | ~+1.0% |
| **HiPO-Rq+Mt-bias** | **+13.89%** | **+6.43%** | -1.00% | **+4.2% avg** |

> 💡 **亮点**：在 GSM8K 上实现 **+13.89%** 的绝对增益，在 MATH500 上达 **+6.43%**。

#### ✅ Llama-3.1-8B-Instruct 结果（Table 3）
| 配置 | GSM8K | Gaokao2023 | AIME24 | 平均提升 |
|-------|--------|-------------|--------|------------|
| Base | 81.80% | 46.03% | 7.33% | — |
| DPO | -0.45% | -1.30% | -1.66% | 下降 |
| **HiPO-Rq-bias** | +1.47% | +3.16% | +1.67% | **+1.83% avg** |

> 💡 **亮点**：Rq-bias 在多个任务上稳定提升，尤其在 Gaokao 和 AIME 表现优于 DPO。

---

### 🔬 消融实验结果（Ablation Studies）

#### Individual Training 表现差异（Appendix B）
| 模型 | 最佳配置 | 效果 |
|------|----------|------|
| **Qwen-2.5-7B** | Rq-only | **平均 +4.46%**，GSM8K 提升 **+11.18%** |
| **Llama-3.1-8B** | Mt-only | 平均 +1.57%，AIME24 提升 **+11.00%** |
| 所有模型 | A-only | **全面负向影响**，平均下降 2–6%，表明单纯优化答案有害于整体推理质量 |

> ❗ **重要发现**：不同架构对 segment 的敏感度不同 ——  
> - Qwen 更受益于 **query refinement（Rq）**
> - Llama 更依赖 **structured reasoning（Mt）**

#### Stepwise Training 进一步提升
- Qwen 通过 **Rq → Mt → Rq+Mt** 渐进训练达到最佳表现（GSM8K 达 90.09%）。
- 表明分阶段、有侧重的训练策略能有效构建复合推理能力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **HiPO 显著优于 Standard DPO**：
   - 在多个数学推理 benchmark 上取得一致且显著的性能提升。
   - 特别是在 **GSM8K** 和 **MATH500** 等需要清晰推理路径的任务中效果突出。

2. **segment-level 优化具有实际意义**：
   - 不同模型架构对不同 reasoning segment 的响应不同，说明可定制化训练策略。
   - Rq 和 Mt 的优化普遍带来正向收益，而 A-only 训练会损害推理完整性。

3. **推理质量全面提升**：
   - GPT-4.1 人工评分显示，HiPO 模型在 **Coherence、Accuracy、Goal Completion** 各维度均得分更高（见 Figure 2–5）。
   - 回应更结构化、逻辑更严密、错误恢复能力更强。

4. **训练策略影响巨大**：
   - “Stepwise” 渐进训练优于单一权重配置，支持“由浅入深”的教学式训练流程。

---

### ⚠️ 局限性
1. **依赖外部模型进行数据标注**：
   - 当前 segment 分解由 GPT-4.1 自动生成，可能引入偏差或噪声。
2. **尚未验证泛化到非数学领域**：
   - 实验集中于数学推理，是否适用于代码、科学推导或其他复杂任务仍待研究。
3. **权重选择依赖经验调参**：
   - 如何自动确定最优 segment 权重仍是开放问题。

---

### 🔮 未来工作方向
1. **自动化 segment 权重调整机制**：
   - 引入 meta-learning 或 curriculum learning 动态分配训练重点。
2. **扩展至 Retrieval-Augmented Generation（RAG）系统**：
   - 利用 Rq/Mt/A 分解优化检索相关性、证据整合与回答生成。
3. **跨领域迁移实验**：
   - 将 HiPO 应用于编程、法律推理、科学研究等需要多层次思考的任务。
4. **减少对外部标注的依赖**：
   - 探索 self-decomposition 或 weakly-supervised 方法来自动生成 reasoning segments。

---

## ✅ 总结
HiPO 成功地将 **DPO 的训练稳定性** 与 **结构化推理的层次性** 相结合，提出了一种简单却高效的分层偏好优化框架。实验证明其不仅能提升准确率，还能显著改善模型的推理透明度与可控性，是迈向“可解释、可干预、可成长”的高级推理模型的重要一步。

</details>

---

### 8. [Less Languages, Less Tokens: An Efficient Unified Logic Cross-lingual Chain-of-Thought Reasoning Framework](https://arxiv.org/abs/2604.20090)

**Authors**: Chenyuan Zhang, Qiguang Chen, Xie Chen, Zhuotao Tian, Bowen Xing, Meishan Zhang, Libo Qin, Baotian Hu, Min Zhang  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.20090v1  

#### Abstract
Cross-lingual chain-of-thought (XCoT) with self-consistency markedly enhances multilingual reasoning, yet existing methods remain costly due to extensive sampling of full trajectories across languages. Moreover, multilingual LLM representations vary strongly by language, hindering direct feature com...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Less Languages, Less Tokens: An Efficient Unified Logic Cross-lingual Chain-of-Thought Reasoning Framework**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **Cross-lingual Chain-of-Thought (XCoT)** 推理框架虽然能通过多语言自洽性（self-consistency）提升多语言推理能力，但存在显著的**计算效率瓶颈**：
- **Full-language sampling**：对所有候选语言生成完整的推理轨迹，造成大量冗余计算。
- **Full-trace reasoning**：必须生成完整 CoT 路径，无法在低质量路径上提前终止，导致 token 浪费。

这些问题使得推理成本随语言数和路径长度线性增长，尤其在资源受限场景下难以部署。

---

### 🚀 提出的新方法：UL-XCoT
作者提出 **UL-XCoT**（Unified Logic XCoT），首个高效的统一逻辑跨语言推理框架，从两个维度实现“少语言、少 token”：

#### （1）**Less Languages**：基于统一逻辑空间的语言选择（Candidate Language Selection, CLS）
- 构建一个**语言不变的统一逻辑空间**（Unified Logic Space），通过投影操作消除语言表层差异。
- 在该空间中计算不同语言对输入的理解一致性得分（Understanding Similarity Score, USS），仅保留最相关的前 *k* 种语言进行后续推理。

#### （2）**Less Tokens**：动态推理路径剪枝（Dynamic CoT Pruning, DCP）
- 在解码过程中监控各语言路径在统一逻辑空间中的演化轨迹。
- 引入**逻辑质量评分**（Logical Quality Score, LQS），识别并在线剪枝逻辑不一致或偏离的低质量路径。
- 实现早期停止，避免无效生成。

#### （3）聚合机制
- 对剩余高质量路径采用投票机制（voting）得出最终答案。

---

### 🔍 相比现有方法的优势
| 维度 | 传统 XCoT 方法 | UL-XCoT |
|------|----------------|---------|
| 语言使用 | 全量语言枚举 | 查询自适应的小候选集（less languages） |
| 推理过程 | 完整路径生成 | 动态剪枝低质量路径（less tokens） |
| 表示可比性 | 语言特异性表示不可直接比较 | 统一逻辑空间支持跨语言轨迹比较 |
| 效率 | 成本高，扩展性差 | 显著降低 token 和延迟开销 |

> ✅ **核心优势**：在保持甚至提升准确率的前提下，大幅降低推理成本，特别适用于低资源语言和实际部署场景。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **PolyMath**：包含 18 种平行语言的数学推理基准，涵盖四种难度等级（Low/Medium/High/Top），用于主实验。
- **MMLU-ProX-Lite**：覆盖 29 种语言的多选题知识推理基准，测试泛化能力。

---

### ⚙️ 实验设置
- **模型**：DeepSeek-R1-Distill-Qwen-7B
- **硬件**：NVIDIA RTX A6000 GPUs (48GB)
- **最大生成长度**：根据任务难度设定为 2048–10240
- **Prompt 模板**：使用 concise-reasoning template 控制推理步数和格式，确保公平比较。

---

### 📊 评估指标
| 类型 | 指标 |
|------|------|
| **有效性** |  
| - 准确率 | DW-ACC（Difficulty-Weighted Accuracy） |
| - 分项 ACC | 各语言及各难度级别下的准确率 |
| **效率** |  
| - Token 开销 | 生成 token 总数 |
| - 延迟 | 端到端 wall-clock latency（秒） |

> 所有基线控制在 UL-XCoT 最坏情况采样预算下进行对比。

---

### 🆚 基线方法
| 方法 | 类型 | 特点 |
|------|------|------|
| **CoT** | 单路径提示 | 基础链式思维 |
| **CLP / CLSP** | 跨语言提示 | 利用多语言信号增强鲁棒性 |
| **SC**（Self-Consistency） | 多路径采样 | 多轨迹投票 |
| **AUTOCAP** | 自适应 XCoT | 自动选择语言并加权聚合 |
| **ST-BoN** | 高效采样 | 固定预算下单语高效推理 |
| **UL-CoT** | 消融对照 | 仅保留 DCP 模块的单语版本 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 在 **PolyMath** 上的结果（18 languages）
| 方法 | 平均 DW-ACC | 平均 Token 数 | 相比 SC ↓ | 相比 AUTOCAP ↓ |
|------|-------------|---------------|-----------|----------------|
| **UL-XCoT** | **19.0** | **3,092** | >65% | >50% |
| SC | 15.2 | ~9,000+ | — | — |
| AUTOCAP | 18.2 | ~6,500 | — | — |

- UL-XCoT 在 **平均准确率上排名第一**，且在几乎所有语言上表现最优。
- **token 消耗减少超过 50%（vs AUTOCAP）、超 65%（vs SC）**。
- 延迟也显著更低（见 Figure 4），尤其在高资源语言如英语、中文上避免了“过度思考”（overthinking）现象。

#### ✅ 在 **MMLU-ProX-Lite** 上的泛化结果（29 languages）
| 方法 | 平均 ACC | 平均 Token | 平均 Latency |
|------|--------|------------|--------------|
| **CLSP** | 40.5 | 27,679.3 | 134.2s |
| **UL-XCoT** | **43.6** | **10,543.6** | **93.7s** |

- 准确率提升 +3.1%，同时 token 下降约 **62%**，延迟下降约 **30%**。
- 在 29 种语言中，UL-XCoT 在 **19 种语言上优于 CLSP**，2 种持平。

---

### 🔬 消融实验（Ablation Study on PolyMath-Low）

| 变体 | ACC | Token (#) | Latency (s) |
|------|-----|-----------|------------|
| UL-XCoT w/o CLS | 84.4 | 5,560 | 36.2 |
| UL-XCoT w/o DCP | 81.4 | 3,893 | 30.7 |
| UL-XCoT w/o ULM | 79.8 | 3,098 | 25.4 |
| **UL-XCoT (full)** | **83.8** | **3,092** | **24.6** |

#### 发现：
- **ULM 是精度核心**：移除后 ACC 明显下降，说明统一逻辑空间对跨语言比较至关重要。
- **CLS 提升效率**：无 CLS 时 token 和延迟激增，表明语言预筛选有效减少搜索空间。
- **DCP 节省计算**：无 DCP 导致更多 token 消耗，但性能提升有限，证明其剪枝有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **效率与性能可兼得**：
   - UL-XCoT 在显著降低 token 和延迟的同时，实现了与最强基线相当甚至更优的准确率。
   - “少语言 + 少 token”策略并未牺牲推理质量。

2. **统一逻辑空间的有效性**：
   - 通过投影去除语言变异子空间，成功构建了可比的跨语言表示。
   - PCA 可视化显示 ULM 显著提升了跨语言嵌入的一致性（Figure 6）。
   - 层间隐藏状态演化更稳定（Figure 7）。

3. **CLS 具有自适应性和公平性**：
   - 不同语言被选中的频率分布均匀（3.7%~7.7%），无明显偏向高资源语言。
   - 支持查询驱动的语言选择，而非固定列表。

4. **DCP 真正剪枝低质量路径**：
   - LLM-as-a-judge 评测显示，被剪枝路径在 step validity 和 completeness 上得分显著更低（Figure 10）。
   - 剪枝比例 $ p \in [0.55, 0.7] $ 可获得最佳 trade-off（Figure 9）。

5. **对低资源语言更具鲁棒性**：
   - 在低资源语言子集上，UL-XCoT 表现更稳定，而标准 XCoT 方法易失效。
   - 表明其在弱信号环境下仍能维持高质量推理。

---

### ⚠️ 局限性
- **依赖白盒访问**：需要获取中间层 hidden states 进行逻辑空间投影和轨迹监控。
- **不适用于黑盒 API**：当前方法难以直接应用于 GPT-4、Claude 等封闭模型。
- **超参数敏感性**：如 warm-up 步长、窗口大小、剪枝比例等需调优。

---

### 🔮 未来工作方向
- 探索**黑盒替代方案**：例如通过输出行为模拟逻辑一致性信号。
- 扩展至其他多模态或多任务场景（如视觉问答、代码生成）。
- 结合强化学习进一步优化语言选择与剪枝策略。
- 构建轻量化统一逻辑模块，便于部署于边缘设备。

---

## ✅ 总结一句话
> **UL-XCoT 通过构建统一逻辑空间，实现了“更少语言、更少 token”的高效跨语言推理，在保持高准确率的同时将解码成本降低超过 50%，并在低资源语言上展现出更强鲁棒性，是迈向实用化多语言大模型推理的重要一步。**

🔗 代码已开源：[https://github.com/chenyuanTKCY/UL-XCoT](https://github.com/chenyuanTKCY/UL-XCoT)

</details>

---

### 9. [Temporally Extended Mixture-of-Experts Models](https://arxiv.org/abs/2604.20156)

**Authors**: Zeyu Shen, Peter Henderson  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.20156v1  

#### Abstract
Mixture-of-Experts models, now popular for scaling capacity at fixed inference speed, switch experts at nearly every token. Once a model outgrows available GPU memory, this churn can render optimizations like offloading and pre-fetching ineffective. We make the case that the options framework in rei...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Temporally Extended Mixture-of-Experts Models

## 1. 论文的主要贡献和创新点

### 解决的问题
现代 **Mixture-of-Experts (MoE)** 模型在推理时频繁切换激活的专家（expert），几乎每个 token 都可能触发一次切换。当模型总参数量超出 GPU 显存容量时，需要将未使用的专家权重 **offload** 到主机内存或磁盘，并在需要时重新加载。这种高频率的切换会导致：
- **加载延迟** 削减推理吞吐
- **预取（prefetching）和缓存策略失效**，因为难以预测下一个 token 所需的专家

当前 MoE 架构大多假设所有专家都能驻留 GPU，忽略了切换成本。

### 提出的新方法
本文提出 **Temporally Extended Mixture-of-Experts (TE-MoE)**，其核心思想是：
- 将 MoE 中的 **expert mask（允许被路由的专家子集）** 视为强化学习中的 **option**。
- 在每个 MoE 层引入一个轻量级 **controller**，该 controller 学习决定：
  - **何时保持当前的 expert mask**（即不切换）
  - **何时切换并选择新的 expert mask**

该方法基于 **Option-Critic 框架** 并引入 **deliberation cost**（决策成本）作为显式的优化目标项。只有当预期收益超过切换成本时，controller 才会执行切换。

### 相比现有方法的优势
| 方面 | 现有方法 | 本文方法 (TE-MoE) |
|------|--------|------------------|
| **专家切换模式** | 几乎每步都切换，无时间连续性 | 显著降低切换频率，expert mask 持续多个 token |
| **内存效率** | 需要为所有专家预留加载能力 | 只需在 GPU 上保留少量活跃专家，大幅降低 VRAM 占用 |
| **训练方式** | 多为静态剪枝或启发式缓存 | 动态、可学习的切换策略，能自适应不同输入 |
| **扩展性** | 固定专家池 | 支持 **continual learning**，可动态添加新专家 |

---

## 2. 核心实验方法和设置

### 数据集
- **训练数据集**：`Nemotron Post-Training Dataset v2`，包含 10 个类别（chat, code, math, STEM, 多语言等）的 prompt。
- **评估数据集**：
  - `MATH`：数学推理任务
  - `MMLU`：大规模多任务语言理解
  - `MMMLU`：多语言 MMLU

### 实验设置
- **基础模型**：`gpt-oss-20b`，包含 24 层 MoE，每层 32 个专家，top-4 路由。
- **控制器设计**：
  - 每层独立一个 controller，包含：
    - **Termination Head**：决定是否终止当前 option（即是否切换）
    - **Selection Head**：决定切换到哪个新的 expert mask
    - **Value & Option-Value Heads**：用于策略梯度更新
  - 使用 **Low-Rank Adaptation (LoRA)** 微调专家和注意力模块。
- **奖励函数**：采用 **self-distillation** 策略，使用原始模型作为 teacher，计算学生模型输出与 teacher 输出之间的 **reverse KL 散度** 作为每 token 奖励。
- **deliberation cost**：设为超参数 `η ∈ {0.02, 0.03, 0.04}`，控制切换的代价。

### 评估指标
- **Switch Rate (%)**：expert mask 发生变化的 token 比例。
- **Accuracy (%)**：在 MATH、MMLU、MMMLU 上的准确率。
- **Perplexity** 和 **Repetition Rate**：用于监控训练稳定性。

### 基线方法对比
- **Frequency-based selection**：保留校准集上最常被激活的 k 个专家。
- **Reconstruction loss minimization**：选择能最好重构原 MoE 层输出的 k 个专家子集。
- **Random selection**：随机选择 k 个专家。
- **Wanda (structured pruning)**：结构化权重剪枝方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（k=16）

| 方法 | MATH Acc (%) | MMLU Acc (%) | MMMLU Acc (%) | Switch Rate (%) |
|------|--------------|--------------|---------------|-----------------|
| Base Model | 71.5 | 79.5 | 67.5 | >50 |
| Ours (η=0.02) | **64.0** | **72.5** | **59.5** | **4.1** |
| Ours (η=0.03) | 58.5 | 67.5 | 56.5 | 1.3 |
| Ours (η=0.04) | 55.0 | 63.0 | 49.5 | 1.2 |

> ✅ **结论**：在 `k=16` 下，TE-MoE 将 switch rate 从 >50% 降至 **<5%**，同时保留了高达 **90% 的 base model 准确率**。

### 关键性能数据（k=8）

| 方法 | MATH Acc (%) | MMLU Acc (%) | MMMLU Acc (%) | Switch Rate (%) |
|------|--------------|--------------|---------------|-----------------|
| Base Model | 71.5 | 79.5 | 67.5 | >50 |
| Ours (η=0.02) | **27.5** | **48.5** | **39.0** | **9.2** |
| Ours (η=0.03) | 23.0 | 41.0 | 31.5 | 8.0 |
| Ours (η=0.04) | 15.5 | 38.0 | 22.5 | 5.4 |

> ✅ **结论**：即使在更严格的 `k=8` 条件下，switch rate 仍可控制在 10% 以下，且性能显著优于所有静态剪枝基线。

### 与基线方法对比
- 在所有任务和设置下，**TE-MoE 均显著优于所有 pruning 基线**。
- 例如，在 `k=16` 下，TE-MoE (η=0.02) 的 MATH 准确率为 64.0%，而最佳基线（frequency）仅为 53.5%。
- 静态剪枝方法在 `k=8` 时性能急剧下降（如 MATH 降至 11.5%），而 TE-MoE 仍能维持一定推理能力。

### 消融实验与分析
- **deliberation cost 控制 trade-off**：随着 `η` 增大，switch rate 单调下降，准确率也随之下降，验证了该超参数的有效性。
- **Temporal Continuity 可视化**：图 6 和图 7 显示，使用 controller 后，expert mask 在多个 token 上保持稳定，展现出明显的“时间块”结构。
- **训练稳定性**：重复率（repetition rate）和教师模型困惑度（teacher perplexity）均保持稳定，表明模型未陷入退化。

---

## 4. 关键结论和发现

### 主要发现
1. **MoE 的频繁切换是次优的**：现有 MoE 模型几乎每步都切换专家，导致无法有效利用 offloading 和 prefetching 等内存优化技术。
2. **Temporal Extension 是可行的**：通过引入基于 **Option-Critic** 的 controller，可以将 MoE 转换为具有时间连续性的系统，**switch rate 可从 >50% 降至 <5%**。
3. **性能-效率可权衡**：通过调节 **deliberation cost**，可以在 **accuracy** 和 **switch rate** 之间进行平滑 trade-off。
4. **无需大规模重训**：仅通过轻量级 adapter 和 self-distillation，即可将已有预训练 MoE 模型转换为 TE-MoE。

### 方法的局限性
- **Per-layer 独立控制**：各层的 controller 独立决策，可能导致不同层在不同时间点切换，不利于统一的内存管理。理想情况是跨层同步切换。
- **Deliberation Cost 是超参数**：当前 `η` 是人工设定的，尚未与真实硬件延迟（如 PCIe 传输时间）直接关联。
- **评估范围有限**：仅在 MATH、MMLU、MMMLU 上评估，未覆盖代码生成、长文本生成等任务。
- **未完全解耦影响因素**：性能提升来自 **dynamic routing** 还是 **self-distillation** 的权重微调，尚需进一步 ablation study。

### 未来工作方向
- **将 Temporal Extension 内建于预训练**：在 pre-training 阶段就引入时间抽象，使模型天生具备低切换特性。
- **实现跨层联合 option**：定义覆盖所有层的全局 option，实现同步切换，简化内存调度。
- **构建端到端高效服务系统**：结合 MoE-infinity 等 offloading 技术，实现真正的 memory-efficient serving。
- **探索与神经可塑性和持续学习的联系**：利用固定大小的活跃专家池，动态添加新专家以适应新任务。

---

> 📌 **总结**：本文提出了 **Temporally Extended MoE** 的新范式，通过引入 **option-based controller** 和 **deliberation cost**，成功将专家切换率大幅降低，为 **memory-efficient serving**、**chunk-wise training** 和 **continual learning** 提供了新的可能性，是一条将 **RL 理论** 应用于 **大模型架构设计** 的典范工作。

</details>

---

### 10. [Robustness of Spatio-temporal Graph Neural Networks for Fault Location in Partially Observable Distribution Grids](https://arxiv.org/abs/2604.20403)

**Authors**: Burak Karabulut, Carlo Manna, Chris Develder  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.20403v1  

#### Abstract
Fault location in distribution grids is critical for reliability and minimizing outage durations. Yet, it remains challenging due to partial observability, given sparse measurement infrastructure. Recent works show promising results by combining Recurrent Neural Networks (RNNs) and Graph Neural Netw...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Robustness of Spatio-temporal Graph Neural Networks for Fault Location in Partially Observable Distribution Grids**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在配电网络中，由于测量设备（如 uPMUs）部署稀疏，导致系统存在**部分可观测性（partial observability）**，这给故障定位带来了巨大挑战。传统方法（如基于阻抗或电压骤降的模型）依赖高密度观测和静态假设，在现代复杂、动态的电网中表现不佳。现有的 GNN 方法通常直接将完整电网拓扑作为图结构输入，未考虑实际测量缺失带来的噪声和计算开销。

本文针对以下三个研究空白展开：
- 当前 GNN 架构在配电网故障定位中的系统性比较不足；
- 缺乏对图结构构建策略的探索（是否必须使用全拓扑？）；
- 多数研究基于理想化、密集传感器配置，缺乏对真实稀疏部署场景的验证。

---

### **提出的新方法与新思路**

#### ✅ **(1) 提出“measured-only”图构造算法**
- 不再采用传统的“full-topology”方式（即所有节点都纳入图中，无测量节点特征置零），而是仅以**有 uPMU 测量的节点**作为图节点，构建更符合实际部分可观测条件的 GNN 图。
- 该图通过保留物理连接关系和相位特性，显式建模稀疏观测下的电气连通性。

#### ✅ **(2) 引入并评估新型 STGNN 架构**
首次将 **GraphSAGE** 和改进版 **GATv2** 应用于配电网故障定位任务，构建了两种新的 Spatio-temporal GNN 模型：
- **RGSAGE**（Recurrent + GraphSAGE）
- **RGATv2**（Recurrent + GATv2）

这些模型相比传统 GCN 更具灵活性，能更好地适应拓扑变化。

#### ✅ **(3) 全面基准测试与鲁棒性分析**
在 IEEE 123-bus 馈线系统上进行了系统的定量比较，涵盖多种 GNN 架构与 RNN 基线，并引入“switch reconfiguration”模拟更具挑战性的运行工况（绿色配置），评估模型在弱故障信号下的稳定性。

---

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **图结构设计** | “measured-only” 减少噪声传播（避免零填充节点干扰消息传递），提升鲁棒性 |
| **计算效率** | 训练时间减少 **6倍**（因图规模显著缩小） |
| **性能表现** | 所有 STGNN 模型均优于纯 RNN 基线，F1 最高提升达 **+11个百分点** |
| **稳定性** | STGNN 模型具有更窄的置信区间（±1.4% vs RNN 的 ±7.5%），表明更强的训练一致性与抗扰能力 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **仿真平台**：OpenDSS + PyDSS 接口生成合成数据
- **基准系统**：IEEE 123-bus 馈线系统（广泛用于故障诊断研究）
- **测量设备**：共部署 **25个 uPMUs**，包括单相、双相和三相类型，反映现实中异构、稀疏的传感器布局
- **故障类型**：共 **11类短路故障**（AG/BG/CG, AB/BC/CA, ABG/BCG/CAG, ABC, ABCG）
- **故障参数多样性**：
  - 故障位置：25个不同地点
  - 负荷变化：负载乘子 $ L \sim U(0.5, 1.3) $
  - 故障电阻：0.1Ω（近金属性）、1Ω、10Ω
- **每种故障场景运行100次模拟**，共生成约 **250万样本窗口**

### **输入特征与预处理**
- 输入为 **三相 RMS 电压幅值**（$V_1, V_2, V_3$）
- 时间序列长度：**S=20**（对应20ms，采样率1ms）
- 特征归一化：Z-score normalization
- 数据划分：70%训练 / 15%验证 / 15%测试，保持时间一致性

### **评估指标**
- 主要指标：**F1 Score**（宏平均）
- 统计稳健性：报告多次随机种子训练下的 **90% 置信区间**
- 分类任务：**26分类问题**（25个故障位置 + 1个“无故障”标签）

### **基线方法对比**
| 类别 | 模型 |
|------|------|
| **非GNN基线** | Shared GRU（独立处理各节点，软投票聚合） |
| **STGNN基线** | RGCN（Recurrent + GCN） |
| **本文提出模型** | RGSAGE-Mean, RGSAGE-Max, RGATv2 |
| **图结构对比** | Measured-only vs Full-topology（含128节点，未测节点特征为0） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（F1 Score）**

| 模型 | 默认配置 F1 | 绿色配置（Switch Reconfig） F1 |
|------|------------|-------------------------------|
| **GRU（RNN Only）** | 86.3 ± 4.2 | **75.5 ± 7.5** |
| **RGCN** | 94.7 ± 0.3 | 87.2 ± 0.8 |
| **RGSAGE-Mean** | 94.7 ± 0.1 | 87.9 ± 0.9 |
| **RGSAGE-Max** | 94.9 ± 0.6 | **87.9 ± 0.9** |
| **RGATv2** | **94.9 ± 0.6** | 86.7 ± 0.9 |

> 注：“绿色配置”指通过开关操作改变潮流路径，导致故障信号更微弱。

---

### **与基线方法的对比结果**
- ✅ **所有 STGNN 模型显著优于 GRU 基线**：
  - 在默认配置下，F1 提升 **~8–9个百分点**
  - 在挑战性更高的绿色配置下，仍维持 **~12个百分点领先**
- ✅ **STGNN 内部性能接近**：
  - 各 GNN 变体之间差异较小（F1 差距 <1%），说明在当前任务下，简单 GNN 即可有效捕捉空间依赖
  - 但在绿色配置中，**RGSAGE-Max 表现最佳**，表明 max-pooling 对提取微弱峰值信号更有效

---

### **消融实验结果**

#### 🔹 **图结构影响（Measured-only vs Full-topology）**
| 模型 | 默认配置 F1 | 绿色配置 F1 |
|------|------------|-------------|
| **Measured-only** | ~94.7 | **86–88** |
| **Full-topology** | ~94.5 | **~71.3** ❌ |

- 在默认配置中两者性能相当；
- 但在绿色配置中，**full-topology 性能急剧下降至 ~71.3**，而 measured-only 保持稳定（>86）
- 原因：full-topology 中大量无测量节点（zero-feature）在 message passing 过程中引入噪声，削弱了关键信号

#### 🔹 **训练效率对比**
| 模型 | Measured-only 训练时间（min） | Full-topology 训练时间（min） | 加速比 |
|------|-----------------------------|------------------------------|--------|
| RGCN | 61.56 ± 0.70 | 363.17 ± 5.70 | **~5.9×** |
| RGATv2 | 65.07 ± 1.18 | 391.34 ± 19.62 | **~6.0×** |

- 所有模型在 measured-only 图上训练速度提升约 **6倍**
- 尽管 RGATv2 因注意力机制最耗时，但仍远快于 full-topology 下的任何模型

---

## **4. 关键结论和发现**

### **主要发现**
1. **STGNN 显著优于纯 RNN 方法**  
   - 利用图结构建模电网拓扑，使模型能够整合分布式测量信息，实现全局推理
   - 在稀疏 uPMU 条件下仍能达到 >94% F1，且稳定性强（CI ≤ ±1.4%）

2. **GNN 架构选择对最终性能影响有限**  
   - 在本任务中，RGCN、RGSAGE、RGATv2 表现相近
   - 但在更复杂的动态环境中（如绿色配置），**RGSAGE-Max 表现出更强鲁棒性**，可能因其 max-pooling 能更好保留异常信号

3. **“measured-only” 图结构是更优实践方案**  
   - 不仅不牺牲性能，反而在挑战性条件下**大幅优于 full-topology**
   - 同时带来 **6倍训练加速** 和更低内存占用
   - 避免了 zero-feature 节点引发的 **oversmoothing 与噪声传播问题**

4. **软投票（soft voting）对 RNN 重要，对 GNN 影响小**  
   - GRU 基线依赖节点预测聚合来提升性能（+~0.10 F1）
   - GNN 本身已通过 message passing 实现信息融合，故聚合增益仅 ~0.01

---

### **方法的局限性**
- 当前实验集中在单一标准馈线（IEEE 123-bus），尚未验证跨馈线泛化能力
- 所有模型均为直推式（transductive），难以直接应用于新增节点或拓扑变更场景
- 未利用边特征（如线路距离、阻抗），可能限制对长距离故障传播的建模
- 仅使用电压数据，未结合电流或其他同步量

---

### **未来工作方向**
1. **探索归纳式 GNN 模型的迁移能力**  
   - 如 GraphSAGE 具备归纳学习能力，有望用于从未见过的新馈线或拓扑重构场景
2. **增强模型对拓扑变化的适应性**  
   - 研究 GATv2 是否能在复杂开关切换模式下自适应调整注意力权重
3. **扩展至其他故障诊断任务**  
   - 如 fault type classification、fault severity estimation
4. **引入更多物理先验**  
   - 加入边特征（line impedance, distance）
   - 结合 power flow equations 设计 physics-informed GNN
5. **轻量化部署研究**  
   - 利用 measured-only 图的小规模优势，推动边缘端实时故障定位应用

--- 

> 📌 **一句话总结**：  
> 本文证明，在部分可观测配电系统中，采用 **measured-only 图结构 + STGNN 架构** 是一种**更高效、更鲁棒、更贴近现实工程需求**的故障定位解决方案，尤其在面对弱故障信号时展现出压倒性优势。

</details>

---

### 11. [Efficient Test-Time Inference via Deterministic Exploration of Truncated Decoding Trees](https://arxiv.org/abs/2604.20500)

**Authors**: Xueyan Li, Johannes Zenn, Ekaterina Fadeeva, Guinan Su, Mrinmaya Sachan, Jonas Geiping  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 6.5  
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
在 **test-time inference** 中，主流的 **self-consistency** 方法通过并行采样多个推理路径（reasoning traces）并进行投票来提升性能。然而，在数学、代码生成等**约束性强的任务**中，该方法存在严重的计算效率问题：
- **重复采样**：多次生成相同的高概率前缀（prefix）和完整路径（duplicate completions）。
- **冗余计算**：无法有效复用共享前缀，导致大量 token 被重复生成。

### 提出的新方法：**Distinct Leaf Enumeration (DLE)**
DLE 是一种**确定性的解码策略**，将截断采样（truncated sampling）视为对一个剪枝后的解码树（pruned decoding tree）的遍历，并系统地枚举不同的叶子节点（distinct leaves），而非随机采样。

#### 核心思想：
- 将生成过程建模为一棵由 **truncated sampling distribution** 定义的树。
- 在每个分支点（branching point），确定性地探索未访问过的替代 token，避免重复。
- 利用 **prefix reuse** 和 **KV cache 共享** 减少冗余计算。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **算法层面** | 避免重复采样，提高搜索空间覆盖率（coverage），在相同序列预算下探索更多高质量路径。 |
| **系统层面** | 支持 prefix caching（如 vLLM、SGLang），显著减少新生成 token 数量，降低延迟和内存开销。 |
| **通用性** | 可作为 drop-in replacement 应用于多种截断采样器（top-p, min-p, e-sampling）。 |

---

## 2. 核心实验方法和设置

### 数据集
- **GSM8K**：小学数学应用题，评估数学推理能力。
- **HumanEval**：代码生成任务，评估编程能力。
- **MMLU-Pro**：多领域语言理解基准，更具挑战性和鲁棒性。

### 实验设置
- **模型**：Qwen2.5-0.5B/7B/14B-Instruct 和 Llama3.2-1B/3B-Instruct。
- **采样器变体**：top-p & top-k, min-p, e-sampling。
- **对比序列数 $k$**：通常设为 8 或 32。
- **硬件平台**：B200 / H100 GPU，使用 vLLM 和 SGLang 推理引擎。

### 评估指标
| 指标 | 含义 |
|------|------|
| `maj@k` | Majority voting 下前 $k$ 条路径的答案准确率（用于 GSM8K, MMLU-Pro）。 |
| `pass@k` | 前 $k$ 条路径中至少有一条通过测试用例的比例（用于 HumanEval）。 |
| **Coverage ($m_k$)** | 枚举集合在截断分布下的总概率质量，衡量搜索空间覆盖程度。 |
| **Cache Hit Rate** | 实际缓存命中率 vs 理论最大可重用比例，反映系统效率。 |
| **New Tokens Generated** | 新生成的 token 数量，衡量计算成本。 |

### 基线方法对比
- **Self-consistency**（with various samplers）
- **Beam search**
- **Diverse beam search**
- **DeepConf**（基于置信度过滤）
- **Self-certainty**（基于 KL 散度评分）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 Qwen2.5-0.5B-Instruct 为例）

| 方法 | GSM8K (maj@8) | HumanEval (pass@8) | MMLU-Pro (maj@8) |
|------|----------------|--------------------|------------------|
| Self-consistency (e-sampling) | 40.94 | 47.56 | 17.20 |
| **DLE (e-sampling)** | **44.05** (+3.11) | **52.44** (+4.88) | **17.97** (+0.77) |
| Self-consistency (min-p) | 39.04 | 46.34 | 17.26 |
| **DLE (min-p)** | **43.97** (+4.93) | **53.05** (+6.71) | **17.89** (+0.63) |
| Self-consistency (top-p+k) | 38.74 | 45.73 | 17.35 |
| **DLE (top-p+k)** | **44.43** (+5.69) | **51.22** (+5.49) | **18.19** (+0.84) |

> ✅ **结论**：DLE 在所有任务和采样器上均显著优于对应的 self-consistency 基线。

### 与基线方法的对比结果
- **vs Self-consistency**：在相同 $k$ 下，DLE 达到更高 accuracy，且所需新生成 token 更少（见图8）。
- **vs Beam search**：beam search 表现不稳定，甚至随 beam size 增加而下降；DLE 更稳定高效。
- **vs DeepConf / Self-certainty**：这些方法依赖于后处理或评分机制，提升有限；DLE 从源头优化采样过程。

### 消融实验结果（Ablation Studies）

#### （1）不同分支策略比较（Table 2）
| 分支策略 | GSM8K (maj@8) | HumanEval (pass@8) |
|---------|---------------|---------------------|
| **PROBFIRST**（默认） | 44.05 | 52.44 |
| **DIVFIRST**（早分叉） | 44.35 | 51.22 |
| RANDBRANCH | 44.43 | 53.05 |
| DFS | 34.12 | 31.71 |
| GLOBALPROB | 35.25 | 40.24 |

> 🔍 **发现**：PROBFIRST 和 DIVFIRST 表现相近，均优于其他策略；DFS 和 GLOBALPROB 明显较差。

#### （2）Early Stopping 消融
- 移除 early stopping 导致轻微性能下降（约 0.5–1%），但节省了约 85% 的“无效”分支继续生成。
- 触发 early stopping 的分支中，**超过 85% 最终会得到完全相同的答案**，说明其有效性。

#### （3）Answer Aggregation 方式（Table 4）
| 加权方式 | Qwen2.5-0.5B (maj@8) | Qwen2.5-7B (maj@8) |
|--------|-----------------------|----------------------|
| **Equal weighting** | **0.4405** | **0.8886** |
| Probability-weighted | 0.4177 | 0.8916 |

> ⚠️ **重要发现**：尽管可以获得每条路径的概率 $Q(x)$，但**均匀加权效果更稳健**，尤其在小模型上。

---

## 4. 关键结论和发现

### 主要发现
1. **Coverage 是有效的性能代理指标**：
   - DLE 在相同 $k$ 下实现了更高的 **coverage**，直接对应更好的下游性能。
   - 图5显示：accuracy 随 coverage 单调上升，DLE 曲线始终位于 self-consistency 上方。

2. **DLE 显著提升推理效率**：
   - 在固定 token 预算下，DLE 能完成更多完整序列（见图8右）。
   - 在达到相同 accuracy 时，DLE 所需新 token 数量远低于 self-consistency（见图8左）。

3. **系统级优化潜力巨大**：
   - DLE 天然支持 prefix reuse，**理论 cache hit rate 高达 80%+**。
   - 实际运行中（SGLang），实际命中率接近理论上限，验证了工程可行性（图9左）。

4. **适用于约束性强的任务场景**：
   - 在数学、代码等正确答案稀疏的任务中，避免重复探索尤为重要。
   - DLE 特别适合部署在 **memory-constrained** 或低批大小（batch size=1）环境中。

### 方法的局限性
- **依赖截断采样器的有效性**：若 truncation rule 过严，可能导致搜索空间过小。
- **确定性可能限制探索广度**：完全放弃随机性，在某些开放域任务中可能不如混合策略灵活。
- **对模型 overconfidence 敏感**：现代 LLM 输出分布过于尖锐，影响分支多样性。

### 未来工作方向
1. **学习更优的分支策略**：结合 sequence probability 与 task-specific quality signal。
2. **混合确定性与随机探索**：例如在高概率区域用 DLE，低概率尾部用 sampling。
3. **训练与解码协同设计**：利用 branch points 提供细粒度 credit assignment，改进 RLHF。
4. **扩展至更多解码范式**：如 tree-of-thought、p-decoding 等。

---

> 📌 **总结一句话**：  
> **DLE 通过将 test-time 推理重构为对截断解码树的确定性遍历，在算法和系统两个层面提升了搜索效率，实现了“更少计算，更好性能”的目标，尤其适用于数学与代码等高重复性推理任务。**

</details>

---

### 12. [Explicit Dropout: Deterministic Regularization for Transformer Architectures](https://arxiv.org/abs/2604.20505)

**Authors**: Vidhi Agrawal, Illia Oleksiienko, Alexandros Iosifidis  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.20505v1  

#### Abstract
Dropout is a widely used regularization technique in deep learning, but its effects are typically realized through stochastic masking rather than explicit optimization objectives. We propose a deterministic formulation that expresses dropout as an additive regularizer directly incorporated into the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Explicit Dropout: Deterministic Regularization for Transformer Architectures**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
传统的 **Dropout** 是一种广泛使用的正则化技术，但其作用机制是**隐式的、随机的**（stochastic masking），依赖于在训练过程中对神经元进行随机失活。这种实现方式存在以下问题：
- 正则化效果难以精确控制；
- 缺乏明确的优化目标表达；
- 难以解释其对模型泛化能力的具体影响。

本文从**显式正则化**（explicit regularization）的角度重新审视 Dropout，提出将其转化为一个**确定性的、可直接加入损失函数的正则项**。

---

### 🆕 提出的新方法
作者提出了 **Explicit Dropout** —— 一种将 Dropout 效应形式化为**加性正则化项**（additive regularizer）的方法，直接嵌入到训练损失中。

#### 核心思想：
- 将传统 Dropout 的期望效应推导为一个**确定性正则项** $ R(\theta) $，从而构建新的训练目标：
  $$
  \mathcal{J}_{\text{final}} = \mathcal{J}_{\text{task}} + \lambda_q J_q + \lambda_k J_k + \lambda_v J_v + \lambda_{\text{ff}} J_{\text{ff}}
  $$
- 推导了适用于 **Transformer 架构** 中多个组件的显式正则化项：
  - Attention 中的 Query (Q)
  - Key (K)
  - Value (V)
  - Feed-Forward Network (FFN)

每个部分都可以通过独立的正则化系数 $\lambda$ 控制强度，实现细粒度调节。

---

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **可控性** | 可通过超参数 $\lambda$ 显式控制各模块的正则化强度，无需依赖随机掩码 |
| **确定性** | 移除了训练中的随机扰动，提升训练稳定性与可复现性 |
| **可解释性** | 正则化项具有明确数学表达，便于分析其对注意力机制的影响 |
| **灵活性** | 支持分别对 Q/K/V/FFN 应用不同强度的正则化，适配不同任务需求 |

> 💡 特别地，该方法避免了 Arora et al. [1] 所提显式 Dropout 在 Attention 上表现差的问题，因其考虑了 Attention 内部的矩阵乘法与 Softmax 归一化结构，而非简单特征级 dropout。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集
实验覆盖多模态任务，验证方法通用性：

| 数据集 | 任务 | 描述 |
|-------|------|------|
| **CIFAR-10 / CIFAR-100** | 图像分类 | 标准图像基准，用于评估 ViT 性能 |
| **THUMOS14** | 时序动作检测 | 视频中定位人类动作，使用 TSN 提取的 Kinetics 和 ActivityNet 特征 |
| **GTZAN** | 音频分类 | 音乐流派识别，使用 VGGish 提取 Mel spectrogram 特征 |

---

### ⚙️ 实验设置
- **模型架构**：
  - CIFAR：7-layer Vision Transformer (ViT)
  - THUMOS14 & GTZAN：2-layer lightweight Transformer encoder
- **输入处理**：
  - 图像 → Patch Embedding
  - 视频/音频 → 预训练特征提取后作为 Token 序列输入
- **评估协议**：
  - 训练/验证集按 70:30 划分
  - 报告 5 次独立运行的平均值 ± 标准差
  - 调整学习率和 $\lambda$ 寻找最佳验证性能

---

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **隐式 Dropout（主流）** | DropAttention [23], DropKey [20], 标准 FFN Dropout |
| **显式正则化（Prior Work）** | Arora et al. [1] 的显式 Dropout 公式（应用于 Q/K/V） |
| **消融配置** | 不同位置应用 Explicit Dropout（如仅 Q、仅 V、AV 等） |

---

## 3. **主要实验结果和性能指标**

### 📊 关键性能数据汇总

#### ✅ **CIFAR-10（Test Accuracy %）**
| 方法 | 准确率 |
|------|--------|
| DropAttention (Implicit) | 85.24 ± 0.75 |
| DropKey (Implicit) | 85.45 ± 0.41 |
| **Explicit (V)** | **86.38 ± 0.44** ✅ |
| Explicit (AV) | 86.11 ± 0.41 |
| Arora et al. [1] (V) | 59.22 ± 1.80 ❌ |

> ➤ **Value 分支上的显式 Dropout 表现最优**，显著优于所有隐式方法。

---

#### ✅ **CIFAR-100（Test Accuracy %）**
| 方法 | 准确率 |
|------|--------|
| DropKey (Implicit) | 59.11 ± 1.23 |
| **Explicit (AV)** | **56.62 ± 2.03** |
| None / Explicit | 56.81 ± 2.84 |

> ➤ 尽管绝对性能略低于最强隐式方法，但 **Explicit Dropout 仍保持竞争力且更稳定**。

---

#### ✅ **THUMOS14（mAP %）**
| 特征来源 | 方法 | mAP |
|---------|------|-----|
| Kinetics | DropKey | 64.58 ± 0.33 |
| | **Explicit (V)** | **64.68 ± 0.36** ✅ |
| ActivityNet | DropKey | 55.87 ± 0.61 |
| | **Explicit (V)** | **56.51 ± 0.37** ✅ |

> ➤ 在复杂时序建模任务中，**Explicit Dropout 实现 SOTA 性能**。

---

#### ✅ **GTZAN（Test Accuracy %）**
| 方法 | 准确率 |
|------|--------|
| Implicit Baseline | ~85.00 |
| **Explicit (K)** | **85.78 ± 0.65** ✅ |
| Explicit (Q) | 85.68 ± 0.65 |

> ➤ 在低复杂度任务中，**Key 分支正则化最有效**，表明任务依赖性。

---

### 🔍 消融实验结果（Ablation Study on CIFAR-10）

| 组件 | 最佳准确率 | 备注 |
|------|-----------|------|
| **Value (V)** | **86.38%** | 对学习率和 $\lambda$ 最鲁棒 |
| **Query (Q)** | 83.92% | 高学习率下性能下降明显 |
| **Key (K)** | 84.06% | 同样对高学习率敏感 |
| **AV (Q+V)** | 86.11% | 接近 V-only，但略低 |

> ➤ **Value 投影是最适合施加显式 Dropout 的位置**，因其不直接影响 Attention 权重计算，扰动更温和。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **Dropout 可被形式化为显式正则化项**，无需随机掩码即可实现等效甚至更强的正则化效果。
2. **显式 Dropout 在多种任务上匹配或超越隐式方法**，尤其在 Value 和 FFN 模块中表现突出。
3. **正则化位置至关重要**：
   - **Value 和 FFN**：适合强正则化，提升泛化；
   - **Query 和 Key**：扰动会破坏 Attention 分布，导致不稳定。
4. **相比 Arora et al. [1]**，本文提出的正则化项更好地建模了 Attention 结构（如 Softmax 和矩阵乘积），因此在 Q/K/V 上表现更好。

---

### ⚠️ 局限性
1. **未统一处理多输入同时 Dropout**：目前无法推导当 Q、K、V 同时应用 Dropout 时的联合正则化表达式。
2. **理论推导基于简化假设**：例如独立 Bernoulli 掩码、线性投影近似等，在深层非线性网络中可能存在偏差。
3. **计算开销略增**：需维护额外的协方差矩阵项（如 $X^TX$），尽管可通过滑动估计缓解。

---

### 🔮 未来工作方向
1. **扩展至其他架构**：如 Convolutional Layers（文中指出可通过 im2col 转换自然推广）。
2. **动态调整 $\lambda$**：结合自适应正则化策略（如 Rademacher-based bound 优化）。
3. **与其他显式正则化结合**：如与 $l_2$、Jacobian regularization 联合使用。
4. **探索更多结构化正则形式**：引入注意力模式先验（如稀疏性、局部性）增强正则项设计。

---

> 📌 **总体评价**：  
> 本论文成功将经典的 **Dropout** 从“黑箱式随机技巧”转变为“白盒式可解释正则化工具”，为 Transformer 的正则化提供了更具控制性和理论基础的新范式。代码已开源：[GitHub - vidhi0206/Explicit-dropout](https://github.com/vidhi0206/Explicit-dropout)。

</details>

---

### 13. [The Tool-Overuse Illusion: Why Does LLM Prefer External Tools over Internal Knowledge?](https://arxiv.org/abs/2604.19749)

**Authors**: Yirong Zeng, Shen You, Yufei Liu, Qunyao Du, Xiao Ding, Yutai Hou, Yuxian Wang, Wu Ning, Haonan Song, Dandan Tu, Bibo Cai, Ting Liu  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19749v1  

#### Abstract
Equipping LLMs with external tools effectively addresses internal reasoning limitations. However, it introduces a critical yet under-explored phenomenon: tool overuse, the unnecessary tool-use during reasoning. In this paper, we first reveal this phenomenon is pervasive across diverse LLMs. We then ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《The Tool-Overuse Illusion: Why Does LLM Prefer External Tools over Internal Knowledge?》论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本论文首次系统性地研究了**大语言模型在工具集成推理（Tool-Integrated Reasoning, TIR）中的“工具过度使用”（Tool Overuse）现象**。该现象指模型在无需调用外部工具的情况下仍频繁或不必要地调用工具，例如：
- **冗余使用（Redundant Usage）**：对简单任务（如 `1+2*3`）也调用代码解释器；
- **无关使用（Irrelevant Usage）**：对与工具无关的任务（如“谁是 Elon Musk？”）强行调用工具。

这种行为不仅增加计算开销和延迟，还会因引入噪声上下文而**降低准确率**。

---

### 🧠 提出的新方法与新思路

论文从两个根本机制揭示并缓解工具过度使用：

#### （1）**知识认知错觉（Knowledge Epistemic Illusion）**
- **发现**：模型无法准确感知自身内部知识边界（internal knowledge boundary），误以为已达到能力极限，从而盲目依赖外部工具。
- **解决方法**：提出 **知识感知的直接偏好优化（Knowledge-aware Direct Preference Optimization, K-DPO）**
  - 构建偏好对：对比“最少工具调用” vs “过多工具调用”的推理路径；
  - 引导模型信任其内部知识，在真正需要时才调用工具。

#### （2）**结果导向奖励陷阱（Outcome-Only Reward Trap）**
- **发现**：当前主流训练范式 RLVR（Reinforcement Learning with Verifiable Rewards）仅以最终答案正确性为奖励信号，忽视推理效率，导致模型为确保正确而不计成本地调用工具。
- **解决方法**：设计**平衡型奖励机制（Balanced Outcome-Efficiency Reward）**
  - 在原有结果奖励基础上，**每多一次工具调用施加惩罚**（如 -0.05/次）；
  - 鼓励模型在保证正确性的前提下最小化工具使用。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| **目标函数** | 仅优化最终准确性（outcome-only） | 同时优化准确性与工具效率 |
| **训练信号** | 忽视过程冗余 | 显式惩罚无效工具调用 |
| **模型认知** | 缺乏对自身知识边界的意识 | 通过 DPO 对齐感知边界与真实能力 |
| **泛化性** | 多数工作聚焦于提升工具使用能力 | 首次系统分析“过度使用”这一反向问题 |

> 💡 创新点总结：  
> 本文不是简单改进工具选择策略，而是**从认知机制和训练激励机制两个层面揭示“为何会过度假设工具必要”这一深层问题**，提供了可解释、可干预的理论框架。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 描述 |
|-------|------|
| **GSM8K** | 包含约1.3k测试样本的小学数学应用题，用于评估基础多步推理能力。 |
| **AIME24 / AIME25** | 美国数学邀请赛（Olympiad-level）题目，共60道高难度数学题，代表前沿挑战。 |
| **混合测试集** | 将上述三个数据集合并用于边界分析（avg@1024）。 |

---

### ⚙️ 实验设置与评估指标

#### 评估流程
1. **Base Reasoning**：禁用工具，纯文本推理；
2. **Task Categorization**：按 `aug@8 ≥ 0.5` 划分“简单”与“复杂”样本；
3. **Autonomous Tool Use**：允许模型自主决定是否调用工具（Python解释器）；

#### 主要评估指标
| 指标 | 定义 |
|------|------|
| **avg@k** | k次独立采样中至少有一次正确的概率，衡量知识可用性（k=8 或 1024） |
| **Tool Frequency / Tool Call Turns** | 每个问题平均调用工具次数 |
| **Accuracy w/ Tool** | 工具增强下的推理准确率 |
| **Irrelevance Detection Score** | 在无相关工具场景下拒绝调用的能力（越高越好） |

#### 基线模型分类
| 类别 | 示例模型 |
|------|--------|
| **Frontier Models (API)** | GPT-5.2, Gemini-3, Claude-4.5, DeepSeek-V3.2 |
| **RLVR-Trained Models** | ReTool-7B/32B, SimpleTIR-7B, ZeroTIR-7B |
| **OSS Foundation Models** | Qwen3-8B, Llama3.1-7B, Qwen2.5系列 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）工具过度使用的普遍性（Table 1）
- 所有模型在**简单任务上平均调用工具 0.93 次/问题**；
- 工具使用导致**准确率下降 3.29% ~ 14.48%**（尤其在 Llama 系列更严重）；
- **RLVR 训练模型工具调用频率比开源基线高出 65%以上**，说明训练方式加剧了问题。

#### （2）无关工具调用检测（Figure 2）
- 前沿模型平均仅能正确识别 **80.2% 的无关请求**（即仍有近 20% 错误调用）；
- 开源模型表现更差，错误率达 **37.5%**。

---

### 🔁 消融实验结果

#### （1）K-DPO 方法效果（Figure 4 & Table 2）
| 模型 | 工具调用减少 | 准确率提升 |
|------|-------------|-----------|
| **Qwen3-8B + K-DPO** | ↓22.7% (2.29 → 1.77) | ↑1% |
| **ReTool-32B + K-DPO** | ↓82.8% (3.09 → 0.53) | ↑3% |

> ✅ 表明：减少不必要的工具调用反而提升了性能，验证了“工具非越多越好”。

#### （2）平衡奖励机制效果（Figure 6 & Table 2）
| 模型规模 | 工具调用减少 | 准确率变化 |
|----------|--------------|------------|
| **7B 模型** | ↓66.7% (5.1 → 1.7) | -1.1%（基本持平） |
| **32B 模型** | ↓60.7% (2.8 → 1.1) | +0.9% |

> ✅ 表明：即使大幅削减工具调用，也能保持甚至提升准确率。

#### （3）消融组大小与温度的影响（Table 5）
- 更大的 rollout group size（G=32）和更高 temperature（T=1.2）有助于探索最优策略；
- 大组采样使模型更好估计“最少工具即可成功的上限”，从而学习更高效的行为。

---

## 4. 关键结论和发现

### 🔑 主要发现

1. **工具过度使用是一个广泛存在且有害的现象**：
   - 不仅浪费资源，还可能损害简单任务的推理性能；
   - 即使顶级模型也无法避免。

2. **两大根本驱动机制被实证揭示**：
   - **知识认知错觉**：模型低估自身知识，误判边界；
   - **结果导向奖励陷阱**：RLVR 训练鼓励“只要答对就行”，无视效率。

3. **两种缓解策略均有效且互补**：
   - K-DPO 改善模型对自身能力的认知；
   - 平衡奖励改变训练激励结构；
   - 二者均可显著减少工具调用而不牺牲精度。

4. **理论支持**：
   - 推导出工具调用的理性条件（Marginal Reliability Gain > Marginal Cost）；
   - 当效率系数 λ → 0 时，任何微小增益都会触发工具调用 → 过度使用成为“理性选择”。

---

### ⚠️ 局限性

| 局限 | 说明 |
|------|------|
| **仅限 code-as-tool 场景** | 实验集中在 Python 解释器执行计算任务，未涵盖搜索、数据库等其他工具类型。 |
| **avg@1024 是代理指标** | 虽然能反映知识上限，但不能精确定位“何时应调用工具”的动态阈值。 |
| **未实现完全自动化决策** | 当前方法仍需人工构建偏好数据或设定惩罚系数，尚未形成自适应机制。 |

---

### 🔮 未来工作方向

1. **动态边界感知机制**：开发可根据任务不确定性实时判断是否调用工具的模块；
2. **多模态工具整合分析**：将研究扩展至搜索引擎、API、视觉工具等多种外部资源；
3. **自监督式效率优化**：无需人工标注偏好，自动识别冗余动作并进行剪枝；
4. **构建标准评测基准**：推出专门针对“工具必要性判断”（tool necessity benchmark）的新标准，如 WTU-Eval（已在文中引用）。

---

## ✅ 总结一句话

> 本文首次揭示了 LLM 在工具使用中存在“**宁可错用，不敢不用**”的认知偏差，并从**内在认知错觉**与**外在奖励误导**双重角度提供了解释与解决方案，推动构建更可靠、高效的 Tool-Augmented LLMs。

</details>

---

### 14. [V-tableR1: Process-Supervised Multimodal Table Reasoning with Critic-Guided Policy Optimization](https://arxiv.org/abs/2604.20755)

**Authors**: Yubo Jiang, Yitong An, Xin Yang, Abudukelimu Wuerkaixi, Xuxin Cheng, Fengying Xie, Zhiguo Jiang, Cao Liu, Ke Zeng, Haopeng Zhang  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.20755v1  

#### Abstract
We introduce V-tableR1, a process-supervised reinforcement learning framework that elicits rigorous, verifiable reasoning from multimodal large language models (MLLMs). Current MLLMs trained solely on final outcomes often treat visual reasoning as a black box, relying on superficial pattern matching...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：V-tableR1: Process-Supervised Multimodal Table Reasoning with Critic-Guided Policy Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Vision-Language Models (VLMs)** 在处理表格推理任务时存在系统性失败，尤其是在需要**精确单元格定位、多跳逻辑链或多步数值计算**的任务中。尽管这些模型具备良好的视觉感知能力（能正确读取单个单元格），但其推理过程常依赖于表面模式匹配（shortcut learning）而非严谨的逻辑推导。

根本原因在于主流训练范式（如 **Supervised Fine-Tuning, SFT**）仅监督最终答案，对中间推理路径缺乏约束，导致模型即使通过幻觉（hallucination）或猜测得到正确答案，也能获得正向反馈。

此外，将 **Reinforcement Learning with Verifiable Rewards (RLVR)** 扩展到多模态领域面临挑战，尤其是自然图像中的主观性和像素空间的模糊性使得“过程奖励”难以构建。

### 🚀 提出的新方法与创新思路
为解决上述问题，作者提出 **V-tableR1** ——一个基于**过程监督**（process supervision）的多模态强化学习框架，专用于表格推理任务。其核心思想是：  
> 将黑盒式的端到端推理转变为可验证、分步骤的 **Visual Chain-of-Thought (V-CoT)** 推理流程，并通过一个独立的 critic VLM 提供细粒度的过程反馈。

#### 主要创新点包括：

- **Visual Chain-of-Thought Dataset**  
  构建了一个大规模带注释的数据集，其中每条样本都包含详细的**步骤级推理轨迹**，并明确标注了逻辑操作对应的视觉锚点（如 `<cell: Row 2, Col 3>`），使抽象推理可被验证。

- **Critic-Guided Process Evaluation**  
  设计了一个专用的 **critic VLM** 来评估生成的 V-CoT 轨迹在每一步的逻辑与视觉一致性。该 critic 不仅判断最终答案是否正确，还动态打分推理过程的质量（`T_proc ∈ [0,1]`），并通过门控机制调节最终奖励。

- **Process-Guided Direct Alignment Policy Optimization (PGPO)**  
  提出一种新型 RL 算法 PGPO，融合了以下关键技术：
  - **Decoupled Policy Constraints**：借鉴 DAPO 的去耦合裁剪策略，提升长序列推理的稳定性；
  - **Length-aware Dynamic Sampling**：参考 LSPO，按推理长度筛选高信息量样本进行优化，提高效率；
  - **Fine-grained Process Feedback Integration**：将 critic 输出的过程得分嵌入奖励函数，直接引导策略学习正确的推理路径。

### 🔍 相比现有方法的优势
| 维度 | 传统方法（SFT / Outcome-Reward RL） | V-tableR1 |
|------|-------------------------------|----------|
| 推理透明性 | 黑箱推理，无法追踪错误来源 | 显式输出 V-CoT 和视觉锚点 |
| 幻觉控制 | 无机制惩罚中间步骤幻觉 | 明确识别并惩罚视觉幻觉与捷径猜测 |
| 奖励信号 | 稀疏、仅关注结果 | 密集、覆盖全过程 |
| 泛化能力 | 容易过拟合训练分布 | 更强的 out-of-distribution 表现 |
| 参数效率 | 需要大模型弥补推理缺陷 | 小模型即可实现 SOTA 性能 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
研究综合评估了 **7 个标准表格推理数据集**，涵盖两类任务：

#### ✅ Table Fact Verification (TFV)
- **TabFact**：基于维基百科表格的事实验证，判断语句是否被表支持。
- **InfoTabs**：半结构化 infobox 上的逻辑推理，需整合隐含知识。

#### ✅ Table Question Answering (TQA)
- **FinQA**：金融报告中的复杂数值计算（加减乘除等）。
- **HiTab**：具有层次标题的统计报表，强调结构理解。
- **TAT-QA**：混合文本与表格的信息提取与计算。
- **TabMWP**：基于表格的数学应用题，要求多步推理。
- **WikiTableQuestions (WTQ)**：高度组合性的查询，涉及聚合、最值、多跳查找。

> 数据集详情见原文 Table 1，包含训练/测试规模、平均分辨率与文件大小。

### 🧪 实验设置与评估指标
- **主干模型**：基于 Qwen3-VL 系列（2B, 4B, 8B, 32B 参数）构建 policy VLM。
- **训练流程**：
  1. 先进行 SFT 初始化，教会模型生成带视觉锚点的 V-CoT；
  2. 再使用 PGPO 进行 RL 微调，引入 critic 提供的过程奖励。
- **评估指标**：
  - **Accuracy**：主要指标，衡量最终答案正确率。
  - **LLM-as-a-Judge**：使用 GPT-5 对生成的 CoT 在 **Accuracy、Logic、Visual Grounding** 三个维度评分（1–5 分）。
- **对比方式**：
  - 与闭源模型（GPT-4o, Gemini-1.5-Flash）及开源模型（InternVL, LLaVA, Qwen-VL, QVQ）进行全面比较。
  - 开展消融实验分析各组件贡献。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 2）

| 模型 | TFV Avg. | TQA Avg. | 最佳表现 |
|------|---------|---------|--------|
| GPT-4.1 | 89.88 | 48.19 | — |
| Gemini-1.5-Flash | 89.80 | 66.55 | — |
| **V-tableR1 (4B)** | **88.45** | **55.44** | **开源模型第一** |
| Qwen3-VL-32B | 80.34 | 54.92 | — |
| QVQ-72B | 76.16 | 54.94 | — |

#### 在具体任务上的亮点表现：
- **FinQA**：V-tableR1 (4B) 达到 **28.98%**，显著优于 InternVL-2.5-14B (24.27%) 和 QVQ-72B (23.12%)。
- **TabMWP**：达到 **83.38%**，接近 QVQ-72B (82.20%)，远超其他同规模模型。
- **HiTab**：从 SFT 基线 36.58% 提升至 **47.24%**（+10.66% 绝对增益）。
- **TabFact**：从 83.88% → **87.95%**，逼近 GPT-4.1 水平。

> ⚡ 特别值得注意的是：**V-tableR1-4B 模型参数仅为 Qwen2.5-VL-72B 的 ~1/18，但在多个任务上全面超越后者**，证明了过程监督带来的巨大参数效率优势。

### 🔁 与基线方法的对比结果
- **vs. SFT 模型**：所有 RL 微调后的 V-tableR1 均大幅超越其 SFT 前身，表明 PGPO 成功提升了推理质量。
- **vs. 闭源大模型**：在 TFV 任务上接近甚至超过 GPT-4.1 和 Gemini；在 TQA 上虽仍落后于最强闭源系统，但已处于领先开源梯队。
- **vs. 专用表格模型（如 Table-LLaVA）**：后者在 FinQA 上仅得 5.29%~7.71%，说明单纯 SFT 无法解决深层推理问题。

### 🔍 消融实验结果（Table 3 & Figure 3）
以 Qwen3-VL-2B 为基础开展消融研究：

| 方法 | TFV-Avg | TQA-Avg |
|------|--------|--------|
| GRPO | 81.34 | 46.43 |
| DAPO | 82.48 | 49.04 |
| PGPO (w/o Process Supervision) | 83.12 | 49.34 |
| **PGPO (Ours)** | **86.12** | **51.04** |

#### 发现：
- 加入 **decoupled clipping**（DAPO）带来约 +2.6% TQA 提升，增强训练稳定性。
- 引入 **length-aware sampling** 后进一步小幅提升。
- **最关键的增益来自 critic-guided process supervision**：单独贡献约 +4.6% TQA 和 +4.8% TFV 提升，证实过程反馈是突破性能瓶颈的核心。

#### 收敛性分析（Figure 3）：
- GRPO 出现严重 reward collapse（第40步崩溃），显示典型的 reward hacking。
- DAPO 更平稳但仍早停。
- **PGPO 不仅恢复探索低谷，还能持续上升至最高稳定奖励水平（>0.7）**，体现更强的鲁棒性和全局搜索能力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **过程监督是破解多模态推理瓶颈的关键**  
   单纯扩大模型规模或增加数据无法根治幻觉与捷径学习；只有显式监督每一步推理，才能建立真正可靠的视觉-语言联合推理能力。

2. **表格是理想的 RLVR 多模态试验场**  
   表格的刚性网格结构和确定性逻辑使其成为首个适合部署严格过程奖励的视觉任务领域，为未来多模态 RL 提供范式参考。

3. **小模型 + 强推理 > 大模型 + 弱逻辑**  
   V-tableR1-4B 在多项任务上击败高达 18× 参数的模型，证明**推理架构设计比单纯堆参数更重要**。

4. **critic VLM 可有效区分三种行为路径**  
   - Path 1（Rigorous Inference）：正确推理 → 高奖励
   - Path 2（Visual Hallucination）：错位定位 → 被检测
   - Path 3（Shortcut Guessing）：跳过定位猜答案 → 被保守惩罚

---

### ⚠️ 方法的局限性
- **依赖表格结构先验**：目前仅适用于规则表格，难以直接迁移到自由排版文档或图表。
- **critic 训练成本较高**：需专门训练一个 32B 级别的 critic VLM，增加了整体资源消耗。
- **合成负例的真实性限制**：错误轨迹由启发式扰动生成，可能未完全覆盖真实世界中的复杂错误模式。

---

### 🔮 未来工作方向
1. **扩展至非表格类结构化视觉任务**：如 PDF 解析、图表理解、表单填写等。
2. **轻量化 critic 设计**：探索更高效的 process verifier 架构，降低部署门槛。
3. **自迭代过程奖励机制**：让 policy 与 critic 共同进化，形成闭环优化。
4. **跨任务迁移学习**：将在表格上学到的严谨推理能力泛化到其他需要精确视觉定位的任务中。

---

## 总结
> **V-tableR1 通过“过程监督 + critic 引导 + 新型 RL 算法 PGPO”，首次实现了对多模态推理全过程的精细化控制，在不依赖超大规模的前提下达到了开源领域的 SOTA 表现。它不仅是一项技术突破，更为构建可信、可解释的视觉语言智能提供了新的方法论路径。**

</details>

---

### 15. [Where Reasoning Breaks: Logic-Aware Path Selection by Controlling Logical Connectives in LLMs Reasoning Chains](https://arxiv.org/abs/2604.20564)

**Authors**: Seunghyun Park, Yuanyuan Lei  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.20564v1  

#### Abstract
While LLMs demonstrate impressive reasoning capabilities, they remain fragile in multi-step logical deduction, where a single transition error can propagate through the entire reasoning chain, leading to unstable performance. In this work, we identify logical connectives as primary points of this st...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Where Reasoning Breaks: Logic-Aware Path Selection by Controlling Logical Connectives in LLMs Reasoning Chains*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在多步逻辑推理中表现出色，但其推理链（reasoning chain）极为脆弱。**单个过渡步骤的错误**（如逻辑连接词选择不当）会通过整个推理链传播，导致最终答案错误。这种结构性脆弱性严重影响了模型在数学、程序合成和复杂决策等严谨领域的可靠性。

本文指出，**逻辑连接词（logical connectives）** 是推理链中最关键的“断裂点”（breaking points）。这些词（如 *therefore*, *however*, *but*, *because*）作为显式的语言枢纽，决定了推理的方向。一旦选错，整个推理路径可能被误导。

### 提出了什么新方法或新思路
作者提出一个**多层次干预框架**，专门针对推理过程中的**逻辑连接词位置**进行精准控制，以引导模型走向正确的逻辑路径。该框架包含三个互补层次的方法：

1. **Gradient-based Logical Steering（基于梯度的逻辑引导）**
   - 在激活层（activation level）对模型内部表示进行干预。
   - 利用目标逻辑连接词的梯度信息构建一个 `steering vector`，在推理时注入到隐藏状态中，将模型的表示空间向“有效逻辑方向”偏移。
   - **优点**：无需修改模型权重，训练免费（training-free），轻量高效。

2. **Localized Branching（局部分支搜索）**
   - 在推理层（inference level）进行干预。
   - 当检测到高不确定性的逻辑连接词位置时，触发前瞻搜索（look-ahead），生成多个候选连接词的延续路径，并通过熵（entropy）和置信度（confidence）评分选择最优分支。
   - **优点**：仅在关键节点进行计算开销，避免了全局搜索的低效性。

3. **Targeted Transition Preference Optimization (TTPO)**
   - 在训练层（training level）进行干预。
   - 一种“外科手术式”的强化学习目标，**仅优化逻辑连接词位置上的单个 token 偏好**。
   - 使用 DPO（Direct Preference Optimization）的思想，但只关注逻辑转折点的正负样本对。
   - **优点**：计算成本极低，避免了全序列微调可能导致的灾难性遗忘。

### 相比现有方法的优势
- **精准性**：不同于传统的全局优化方法（如 Beam Search、Self-Consistency），本方法聚焦于**逻辑连接词这一小部分高杠杆（high-leverage）位置**，实现了“四两拨千斤”的效果。
- **效率高**：干预计算集中在少数关键节点，**维持了接近贪婪解码（greedy decoding）的效率**，远优于需要大量额外计算的全局搜索方法。
- **模块化与互补性**：三种方法分别作用于不同阶段（激活、推理、训练），可独立使用也可组合，提供了灵活的优化路径。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在五个逻辑推理基准上进行，涵盖多种推理形式：
- **ZebraLogic**：多类分类，强调多步演绎。
- **BIG-Bench Hard (deductive subset)**：严格逻辑推理子集。
- **RuleBERT**：基于规则的推理。
- **LogiQA 2.0**：自然语言理解中的逻辑推理。
- **ProntoQA**：形式/隐式逻辑转换。

此外，使用 **OpenThoughts** 数据集用于提取 `steering vector`（不参与推理评估）。

### 实验设置和评估指标
- **模型**：在两个系列的指令微调模型上测试：
  - Gemma-3-4b-it 和 Gemma-3-12b-it
  - Phi-4-mini-instruct 和 Phi-4-reasoning-plus
- **评估指标**：主要使用任务准确率（accuracy）。
- **效率指标**：报告 Token Cost（× Greedy）和 Wall-Clock Time（× Greedy），衡量计算开销。

### 基线方法对比
- **Greedy Decoding**：基础基线，单次前向传递。
- **Beam Search**：经典的全局搜索方法（beam=5）。
- **Self-Consistency (n=5)**：采样多条路径并投票，代表强基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 4 & Table 5）
| 方法 | ZebraLogic (Gemma-3-4b-it) | BBH (Ded.) | RuleBERT | LogiQA 2.0 | ProntoQA | 平均 |
|------|----------------------------|------------|----------|------------|----------|------|
| Greedy | 38.8 | 75.3 | 60.3 | 55.2 | 90.0 | 63.9 |
| Beam Search | 33.4 | 77.0 | 67.7 | 54.0 | 90.6 | 64.5 |
| **Branching** | **42.0** | 74.7 | **63.4** | **56.8** | 90.2 | **65.4** |
| **TTPO** | 40.4 | **75.9** | 60.8 | 56.0 | **90.4** | **64.7** |

- **Branching** 在 ZebraLogic 上提升显著（+3.2），且在 LogiQA 2.0 上也优于 Greedy 和 Beam。
- **TTPO** 在 BBH (Ded.) 上表现最佳，且在 ProntoQA 上达到 90.4。
- 对于更大的 **Gemma-3-12b-it**，TTPO 在 ZebraLogic 上达到 **55.6**，优于 Greedy (53.2) 和 Beam (51.8)。

### 与基线方法的对比结果
- **Beam Search 不稳定**：在某些数据集（如 ZebraLogic）上甚至**低于 Greedy**，说明全局搜索不一定能改善逻辑推理。
- **本方法更优**：提出的三种方法在多数情况下优于或匹配 Greedy 和 Beam，且**效率远高于后者**。
- **效率对比（Table 12）**：
  - Self-Consistency (n=5) 的 Token Cost 为 2.90×~4.37× Greedy。
  - 本文的 **Branching** 仅为 1.18×~1.02×，**Steering** 和 **TTPO** 接近 1.0×。
  - 在 **Gemma-3-4b-it** 上，本文最佳方法平均准确率 **65.7**，**高于 Self-Consistency 的 65.2**，但计算成本仅为 **≤1.45×**，远低于后者的 2.90×。

### 消融实验结果
- **方法互补性**（Table 5）：
  - **TTPO + Steering** 和 **TTPO + Branching** 组合进一步提升了性能，表明三种方法提供的是**互补的控制信号**，而非冗余。
  - 例如，在 Phi-4-mini-instruct 上，TTPO + Branching 在 RuleBERT 上达到 52.1，优于单独使用任一方法。
- **TTPO 分布锐化**（Figure 6）：
  - TTPO 训练后的模型在逻辑连接词位置表现出**更低的熵（entropy）和更高的置信度（confidence）**，说明其成功“锐化”了概率分布，减少了不确定性。

---

## 4. 关键结论和发现

### 主要发现
1. **逻辑连接词是推理链的“薄弱环节”**：
   - 实证分析显示，逻辑连接词位置具有**最高的 token-level entropy**。
   - 单个连接词的随机替换导致正确推理链崩溃的概率（41.1%）远高于其他高熵 token（13.0%），证明其**因果杠杆作用（causal leverage）极高**。
   - 成功的推理修复往往涉及**跨逻辑关系类别**的转变（如从 Causal 转向 Instantiation），而非同义词替换。

2. **局部干预优于全局搜索**：
   - 通过仅在逻辑连接词这一小部分关键节点进行干预，即可实现与或超越全局搜索方法的性能，同时保持极高的推理效率。

3. **多层次干预的有效性**：
   - 无论是激活层的 Steering、推理层的 Branching，还是训练层的 TTPO，都能独立地提升推理性能，验证了“控制逻辑连接词”这一核心思想的有效性。

### 方法的局限性
- **依赖显式语言标记**：方法依赖于预定义的逻辑连接词集合 $ \mathcal{S}_\text{l} $，无法处理**没有显式连接词的隐式推理步骤**。
- **分词器敏感性**：多 token 连接词（如 "as a result"）的检测可能受分词器（tokenizer）影响，存在对齐误差风险。
- **连接词密度限制效果**：在连接词稀疏的任务（如 BBH, LogiQA 2.0）上，干预机会少，性能增益有限。

### 未来工作方向
- 扩展方法以识别和干预**隐式逻辑关系**。
- 探索更鲁棒的连接词检测机制，减少对分词器的依赖。
- 将该框架应用于更广泛的推理任务，如数学证明、代码生成等。
- 结合外部符号系统（symbolic systems）进一步增强逻辑一致性。

--- 

> **总结**：本文揭示了 LLMs 推理失败的关键在于**逻辑连接词的选择**，并提出了一个高效、精准的多层次干预框架。实验证明，**聚焦于这些“逻辑枢纽”进行控制，可以在几乎不增加计算成本的情况下，显著提升多步推理的准确性和稳定性**，为改进 LLMs 的逻辑能力提供了一条新的范式。

</details>

---

### 16. [Fast Amortized Fitting of Scientific Signals Across Time and Ensembles via Transferable Neural Fields](https://arxiv.org/abs/2604.19979)

**Authors**: Sophia Zorek, Kushal Vyas, Yuhao Liu, David Lenz, Tom Peterka, Guha Balakrishnan  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.19979v1  

#### Abstract
Neural fields, also known as implicit neural representations (INRs), offer a powerful framework for modeling continuous geometry, but their effectiveness in high-dimensional scientific settings is limited by slow convergence and scaling challenges. In this study, we extend INR models to handle spati...

---

### 17. [Lever: Inference-Time Policy Reuse under Support Constraints](https://arxiv.org/abs/2604.20174)

**Authors**: Ihor Vitenki, Noha Ibrahim, Sihem Amer-Yahia  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.20174v1  

#### Abstract
Reinforcement learning (RL) policies are typically trained for fixed objectives, making reuse difficult when task requirements change. We study inference-time policy reuse: given a library of pre-trained policies and a new composite objective, can a high-quality policy be constructed entirely offlin...

---

### 18. [Distributional Value Estimation Without Target Networks for Robust Quality-Diversity](https://arxiv.org/abs/2604.20381)

**Authors**: Behrad Koohy, Jamie Bayne  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.20381v1  

#### Abstract
Quality-Diversity (QD) algorithms excel at discovering diverse repertoires of skills, but are hindered by poor sample efficiency and often require tens of millions of environment steps to solve complex locomotion tasks. Recent advances in Reinforcement Learning (RL) have shown that high Update-to-Da...

---

### 19. [A Hierarchical MARL-Based Approach for Coordinated Retail P2P Trading and Wholesale Market Participation of DERs](https://arxiv.org/abs/2604.20586)

**Authors**: Patrick Wilk, Ethan Cantor, Yikui Liu, Jie Li  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.20586v1  

#### Abstract
The ongoing shift towards decentralization of the electric energy sector, driven by the growing electrification across end-use sectors, and widespread adoption of distributed energy resources (DERs), necessitates their active participation in the electricity markets to support grid operations. Furth...

---

### 20. [Hidden Reliability Risks in Large Language Models: Systematic Identification of Precision-Induced Output Disagreements](https://arxiv.org/abs/2604.19790)

**Authors**: Yifei Wang, Tianlin Li, Xiaohan Zhang, Xiaoyu Zhang, Wei Ma, Mingfei Cheng, Li Pan  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.19790v1  

#### Abstract
Large language models (LLMs) are increasingly deployed under diverse numerical precision configurations, including standard floating-point formats (e.g., bfloat16 and float16) and quantized integer formats (e.g., int16 and int8), to meet efficiency and resource constraints. However, minor inconsiste...

---

### 21. [ActuBench: A Multi-Agent LLM Pipeline for Generation and Evaluation of Actuarial Reasoning Tasks](https://arxiv.org/abs/2604.20273)

**Authors**: Jan-Philipp Schmidt  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.20273v1  

#### Abstract
We present ActuBench, a multi-agent LLM pipeline for the automated generation and evaluation of advanced actuarial assessment items aligned with the International Actuarial Association (IAA) Education Syllabus. The pipeline separates four LLM roles by adapter: one agent drafts items, one constructs ...

---

### 22. [ESGLens: An LLM-Based RAG Framework for Interactive ESG Report Analysis and Score Prediction](https://arxiv.org/abs/2604.19779)

**Authors**: Tsung-Yu Yang, Meng-Chi Chen  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.19779v1  

#### Abstract
Environmental, Social, and Governance (ESG) reports are central to investment decision-making, yet their length, heterogeneous content, and lack of standardized structure make manual analysis costly and inconsistent. We present ESGLens, a proof-of-concept framework combining retrieval-augmented gene...

---

### 23. [Parallel-SFT: Improving Zero-Shot Cross-Programming-Language Transfer for Code RL](https://arxiv.org/abs/2604.20835)

**Authors**: Zhaofeng Wu, Shiqi Wang, Boya Peng, Anuj Goyal, Melanie Kambadur, Sebastian Ruder, Yoon Kim, Chloe Bi  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.20835v1  

#### Abstract
Modern language models demonstrate impressive coding capabilities in common programming languages (PLs), such as C++ and Python, but their performance in lower-resource PLs is often limited by training data availability. In principle, however, most programming skills are universal across PLs, so the...

---

### 24. [Lifecycle-Aware Federated Continual Learning in Mobile Autonomous Systems](https://arxiv.org/abs/2604.20745)

**Authors**: Beining Wu, Jun Huang  
**Category**: cs.LG  
**Published**: 2026-04-23  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.20745v1  

#### Abstract
Federated continual learning (FCL) allows distributed autonomous fleets to adapt collaboratively to evolving terrain types across extended mission lifecycles. However, current approaches face several key challenges: 1) they use uniform protection strategies that do not account for the varying sensit...

---

### 25. [The AI Telco Engineer: Toward Autonomous Discovery of Wireless Communications Algorithms](https://arxiv.org/abs/2604.19803)

**Authors**: Fay\c{c}al A\"it Aoudia, Jakob Hoydis, Sebastian Cammerer, Lorenzo Maggi, Gian Marti, Alexander Keller  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.19803v1  

#### Abstract
Agentic AI is rapidly transforming the way research is conducted, from prototyping ideas to reproducing results found in the literature. In this paper, we explore the ability of agentic AI to autonomously design wireless communication algorithms. To that end, we implement a dedicated framework that ...

---

### 26. [Emergence Transformer: Dynamical Temporal Attention Matters](https://arxiv.org/abs/2604.19816)

**Authors**: Zihan Zhou, Bo-Wei Qin, Kai Du, Wei Lin  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.19816v1  

#### Abstract
The Transformer, a breakthrough architecture in artificial intelligence, owes its success to the attention mechanism, which utilizes long-range interactions in sequential data, enabling the emergent coherence between large language models (LLMs) and data distributions. However, temporal attention, t...

---

### 27. [EvoAgent: An Evolvable Agent Framework with Skill Learning and Multi-Agent Delegation](https://arxiv.org/abs/2604.20133)

**Authors**: Aimin Zhang, Jiajing Guo, Fuwei Jia, Chen Lv, Boyu Wang, Fangzheng Li  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.20133v1  

#### Abstract
This paper proposes EvoAgent - an evolvable large language model (LLM) agent framework that integrates structured skill learning with a hierarchical sub-agent delegation mechanism. EvoAgent models skills as multi-file structured capability units equipped with triggering mechanisms and evolutionary m...

---

### 28. [FSFM: A Biologically-Inspired Framework for Selective Forgetting of Agent Memory](https://arxiv.org/abs/2604.20300)

**Authors**: Yingjie Gu, Bo Xiong, Yijuan Guo, Chao Li, Xiaojing Zhang, Liqiang Wang, Pengcheng Ren, Qi Sun, Jingyao Ma, Shidang Shi  
**Category**: cs.AI  
**Published**: 2026-04-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.20300v1  

#### Abstract
For LLM agents, memory management critically impacts efficiency, quality, and security. While much research focuses on retention, selective forgetting--inspired by human cognitive processes (hippocampal indexing/consolidation theory and Ebbinghaus forgetting curve)--remains underexplored. We argue t...

---

### 29. [Avoiding Overthinking and Underthinking: Curriculum-Aware Budget Scheduling for LLMs](https://arxiv.org/abs/2604.19780)

**Authors**: Amirul Rahman, Aisha Karim, Kenji Nakamura, Yi-Fan Ng  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.19780v1  

#### Abstract
Scaling test-time compute via extended reasoning has become a key paradigm for improving the capabilities of large language models (LLMs). However, existing approaches optimize reasoning under fixed or uniformly sampled token budgets, ignoring the fundamental mismatch between problem difficulty and ...

---

### 30. [Bootstrapping Post-training Signals for Open-ended Tasks via Rubric-based Self-play on Pre-training Text](https://arxiv.org/abs/2604.20051)

**Authors**: Chengyu Huang, Sheng-Yen Chou, Zhengxin Zhang, Claire Cardie  
**Category**: cs.CL  
**Published**: 2026-04-23  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.20051v1  

#### Abstract
Self-play has recently emerged as a promising paradigm to train Large Language Models (LLMs). In self-play, the target LLM creates the task input (e.g., ask a question), which it then addresses itself by producing a task output (e.g., give an answer). A reward model evaluates the output, and the rew...

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
