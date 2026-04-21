# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-21 07:19:12 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing](https://arxiv.org/abs/2604.18170)

**Authors**: Ziyang Liu  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2604.18170v1  

#### Abstract
LLMs edit text and code by autoregressively regenerating the full output, even when most tokens appear verbatim in the input. We study Copy-as-Decode, a decoding-layer mechanism that recasts edit generation as structured decoding over a two-primitive grammar:  references an input line range, ... emi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）在执行文本或代码编辑任务时，通常采用**自回归生成**（autoregressive decoding）方式从头生成整个输出。然而，大多数编辑操作实际上只是对输入文档进行局部修改，大部分内容是直接保留的。这种“全量重生成”模式存在两个显著问题：

- **计算资源浪费**：重复生成已存在于输入中的 token，导致延迟随输出长度增长，而非编辑规模。
- **语义漂移风险**：模型可能在本应保持不变的内容上产生错误。

### 提出的新方法：Copy-as-Decode
作者提出了一种名为 **Copy-as-Decode** 的解码层机制，将编辑生成重构为一种**语法约束的结构化解码过程**，其核心思想如下：

#### （1）引入两原语语法（Two-Primitive Grammar）
定义一个程序化语法，控制模型输出格式：
```xml
<program>
  (<copy lines="i-j"/> | <gen>new_content</gen>)*
</program>
```
- `<copy lines="i-j"/>`：引用输入文档中第 `i` 到 `j` 行的内容（verbatim copy）。
- `<gen>...</gen>`：生成新的自由文本内容。

该语法由一个**词元级有限状态机**（token-level FSM）强制执行，确保输出始终符合语法，解析率为 100%。

#### （2）并行预填充拼接（Parallel-Prefill Splice）
当模型决定执行 `<copy lines="i-j"/>` 操作时，系统不再逐个 token 自回归生成，而是：
- 直接从输入文档提取对应行的 token；
- 通过一次 **parallel-prefill forward pass** 将这些 token 批量插入 KV Cache。

这相当于用 **1 次前向传播** 替代 `N` 次自回归解码步骤，大幅降低延迟。

#### （3）类比于推测性解码（Speculative Decoding），但确定性更强
- 类似于推测性解码使用小模型作为 draft，Copy-as-Decode 将**输入文档本身作为 draft**。
- 不同之处在于：接受是**确定性的**（acceptance=1 by construction），无需概率验证，避免了拒绝风险。

---

### 相比现有方法的优势

| 方法 | 输出格式 | 是否保证复制一致性 | 是否加速复制部分 | 备注 |
|------|----------|---------------------|--------------------|------|
| 全量自回归（Full Regeneration） | 原始文本 | ❌ | ❌ | 高延迟，易漂移 |
| Search/Replace（如 Aider） | `SEARCH: ..., REPLACE: ...` | ❌（锚点歧义） | ❌ | 可能匹配多个位置，导致错误替换 |
| Unified Diff | `@@ hunk @@ ...` | ✅（行号定位） | ❌ | 仍需自回归生成所有内容 |
| **Copy-as-Decode（本文）** | `<copy lines="i-j"/>` | ✅（精确引用） | ✅（并行拼接） | **唯一同时实现低 token 开销和完美 round-trip 的方法** |

---

## 2. 核心实验方法和设置

### 数据集
实验基于三个编辑任务数据集，共 **482 个样本**：

| 数据集 | 类型 | 描述 |
|--------|------|------|
| **ProbeEdit** | 文本编辑 | 154 个短文本编辑任务（会议记录、技术文档等），黄金输出明确 |
| **HumanEvalPack-Fix (Python)** | 代码修复 | 164 个 Python 函数 bug 修复任务 |
| **HumanEvalPack-Fix (JavaScript)** | 代码修复 | 164 个 JS 函数 bug 修复任务 |

### 实验设置
- **模型**：Qwen2.5-{1.5B, 7B}-Instruct（BF16）
- **硬件**：单张 A100 80GB GPU
- **运行时**：HuggingFace Transformers（eager 模式），无 vLLM/SGLang 集成
- **上下文长度**：1024 tokens
- **评估协议**：
  - **Oracle 程序构造**：使用 `difflib.SequenceMatcher` 对齐输入与黄金输出，自动生成最优 `<copy>` 和 `<gen>` 序列。
  - **Resolver**：确定性地展开程序为最终输出，用于评估是否能完美重建黄金输出。

### 评估指标
1. **Kernel Speedup**：单次 copy 拼接的延迟 vs. N 次自回归解码。
2. **Copy Ceiling**：黄金输出中可被 `<copy>` 覆盖的 token 比例（上限分析）。
3. **bndexact**：结合 span 分布与 kernel 加速比的闭式 wall-clock 上限。
4. **Exact Match (EM)**：输出与黄金完全一致的比例。
5. **Pipeline Losslessness**：Oracle 程序能否通过 Resolver 完美还原黄金输出。

### 基线方法对比
- **Full Regeneration**：标准自回归生成。
- **Search/Replace**：Aider 风格的搜索替换块。
- **Unified Diff**：标准 diff hunk 格式。
- **Prompt-level CRP**：仅提示中使用语法，无 FSM 强制。

---

## 3. 主要实验结果和性能指标

### （1）Kernel Speedup（拼接效率）
在 Qwen2.5-7B 上，不同长度的 copy 操作相比自回归解码的速度提升：

| Copy 长度 (N) | 速度提升 (Speedup) |
|---------------|------------------|
| 8 tokens      | 6.8×             |
| 64 tokens     | 43.4×            |
| 512 tokens    | **90.5×**         |

> 在 1.5B 模型上最高可达 **303×**，表明小模型更受益于并行前向传播。

### （2）Copy Ceiling（复制覆盖率）
黄金输出中可通过 `<copy>` 引用的 token 比例：

| 数据集 | Copy Coverage (f_line) |
|--------|------------------------|
| ProbeEdit | **97.8%** |
| HEvalFix-Py | 74.1% |
| HEvalFix-JS | 78.8% |
| **Pooled** | **93.8%** |

> 表明绝大多数编辑内容是直接保留的。

### （3）端到端理论加速上限：bndexact
结合 copy 覆盖率与 kernel 加速，得出理想情况下的最大 wall-clock 加速比：

| 数据集 | bndexact（理论最大加速） |
|--------|-------------------------|
| ProbeEdit | **29.0×** |
| HEvalFix-Py | 3.4× |
| HEvalFix-JS | 4.2× |
| **Pooled** | **13.0×** |

> 这是任何 copy-aware 编辑器所能达到的性能上限。

### （4）与基线方法对比（Table 3）
在 **format-theoretic** 层面比较输出 token 数和 round-trip EM：

| 方法 | 平均输出 token | Round-Trip EM | 优点/缺点 |
|------|----------------|--------------|-----------|
| Full Regeneration | 最高 | 1.00 | 高开销 |
| Search/Replace | 较低 | **0.81–0.94** | 存在锚点歧义 |
| Unified Diff | 中等偏高 | 1.00 | 头部开销大 |
| **Copy-as-Decode** | **最低或相当** | **1.00** | ✅ 唯一兼具低开销与高保真 |

> Copy-as-Decode 是唯一在**输出 token 数最少**的同时还能**100% 正确应用编辑**的方法。

### （5）消融实验与敏感性分析

#### FSM 的必要性
- 移除 FSM 后，即使使用相同语法提示，模型也容易偏离语法（如漏闭合标签），导致解析失败。
- FSM 保证了 100% 语法正确性。

#### 拼接的必要性
- 若只保留 FSM 但不启用拼接，则延迟接近全量生成，仅节省少量输出 token。
- 拼接是实现低延迟的关键。

#### Span Selection 精度要求极高（Perturbation Study）
对 oracle 的 line indices 添加 ±1 的噪声后，EM 急剧下降：

| 噪声半径 e | Pooled EM |
|-----------|-----------|
| 0         | 100.0%    |
| 1         | **15.48%** |
| 2         | 9.42%     |

> 表明 span selection 必须近乎完美才能发挥机制潜力。

### （6）监督微调试点（SFT Pilot）
在 Qwen2.5-Coder-1.5B 上进行小规模 SFT，目标是让模型学会生成 oracle program：

| 训练配置 | Held-out EM | 95% CI |
|---------|-------------|--------|
| 未训练（任何策略） | 0/33 (0%) | [0.0, 10.4] |
| SFT + FSM（131 示例） | 12/99 (12.1%) | [7.0, 20.1] |
| SFT + FSM（385 示例） | 17/99 (17.2%) | [11.0, 26.0] |

> 表明 span selection 是**可学习的**，但当前性能远未达部署水平。

---

## 4. 关键结论和发现

### 主要发现
1. **Copy-as-Decode 是一种高效的编辑范式**：通过语法约束 + 并行拼接，理论上可实现高达 **29× 的端到端加速**。
2. **绝大多数编辑内容是可复制的**：在真实数据集中，**74–98% 的黄金 token 可通过行级 copy 获得**。
3. **现有格式存在根本缺陷**：Search/Replace 易受锚点歧义影响；Unified Diff 开销大；Copy-as-Decode 是目前唯一兼顾**低 token 成本**与**高保真度**的方案。
4. **机制本身是 sound 的**：Oracle 程序可在 Resolver 中 100% 精确还原黄金输出（pipeline losslessness），说明任何失败都源于 span selection，而非机制设计。
5. **Span selection 极其敏感**：±1 行误差即可使 EM 从 100% 降至 15%，说明下游模型必须具备极高的定位精度。

### 方法的局限性
1. **尚未集成到生产级推理框架**：未与 vLLM/SGLang 等批处理系统集成，当前结果为理论上限或 format-only 测试。
2. **缺乏高性能的 span selector**：当前 SFT 仅达到 12–17% EM，远低于实用门槛。
3. **仅支持单文件编辑**：不支持多文件或 agent-style 工作流（如 SWE-bench）。
4. **粒度限制**：当前为行级 copy，无法处理行内细粒度修改（如表达式替换）。token-level 扩展有潜力（覆盖率达 91–99%），但选择难度更高。

### 未来工作方向
1. **训练更强的 span selector**：使用更大模型（如 7B）、更多数据、QLoRA 微调，并针对性优化 span-end 准确率。
2. **实现 token-level copy primitive**：支持 `<copy tokens="a-b"/>`，进一步提升覆盖范围。
3. **集成到 vLLM/SGLang**：实现真正的 batched serving 下的 KV 拼接。
4. **扩展至多文件和 AST 级编辑**：支持跨文件引用和语法树节点级操作。
5. **与推测性解码组合**：在 `<gen>` 区域使用 speculative decoding，进一步加速自由生成部分。

---

> **总结一句话**：  
> **Copy-as-Decode 提供了一个“复制即解码”的新范式，通过语法约束与并行 KV 拼接，理论上可将 LLM 编辑延迟降低一个数量级，但其实用性高度依赖于一个高精度的 span selection 模型。**

</details>

---

### 2. [SinkRouter: Sink-Aware Routing for Efficient Long-Context Decoding in Large Language and Multimodal Models](https://arxiv.org/abs/2604.16883)

**Authors**: Junnan Liu, Xinyan Liu, Peifeng Gao, Zhaobo Qi, Beichen Zhang, Weigang Zhang, Antoni Bert Chen  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2604.16883v1  

#### Abstract
In long-context decoding for LLMs and LMMs, attention becomes increasingly memory-bound because each decoding step must load a large amount of KV-cache data from GPU memory. Existing acceleration strategies often trade efficiency for accuracy by relying on heuristic pruning that may discard useful i...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*SinkRouter: Sink-Aware Routing for Efficient Long-Context Decoding in Large Language and Multimodal Models*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **Long-context decoding** 过程中，随着上下文长度增长，模型每一步解码都需要加载庞大的 **KV-cache** 数据，导致计算过程严重受制于 **GPU 内存带宽**（memory-bound），造成显著延迟。现有加速方法（如 token 剪枝、KV-cache 压缩）通常以牺牲准确性为代价，且缺乏对 **attention sink 现象** 的深入机制理解。

### 🚀 提出的新方法与创新思路
提出 **SinkRouter** —— 一种无需训练、即插即用（plug-and-play）的 **head-level selective routing 框架**，其核心思想是：
- 将 **attention sink**（如 BOS token 吸收大量注意力）视为一种 **可学习的、稳定的、低影响的固定点（e-fixed point）**，而非需要保留的“锚点”。
- 利用 sink token 的几何特性（如 key 方向集中、value 范数接近零）作为运行时信号，在解码前预测某些 attention head 是否会进入“近似无操作”状态。
- 若判断某 **KV group** 将进入 sink-dominant 状态，则跳过历史 KV-cache 加载，并用零值替代输出，从而减少内存访问。

### 🔍 相比现有方法的优势
| 维度 | SinkRouter | 传统方法（如 H2O、StreamingLLM） |
|------|-----------|-------------------------------|
| **机制理解** | 基于对 sink 的机制建模（low-impact fixed point） | 多依赖启发式规则（如保留高 attention token） |
| **操作粒度** | **Head-level / KV-group-level**，契合 GQA 架构 | Token-level 或 cache-level，难以对齐硬件执行单元 |
| **是否需训练** | ❌ 完全无需训练 | 多数无需训练，但部分需微调 |
| **兼容性** | ✅ 可与 KV-cache 压缩等方法结合 | 通常独立使用 |
| **硬件效率** | ✅ 配套硬件感知 Triton kernel，实现 block-level 分支与 Split-K 并行 | 多数未优化底层 kernel |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **文本模型基准**：
  - **LoNGBENCH**：多语言、多任务长上下文理解
  - **INFINITEBENCH**：极端长度检索与精度敏感任务
- **多模态模型基准**：
  - **MMVP**, **CVBENCH**, **MILEBENCH**：测试视觉-语言模型在长上下文下的感知与推理能力

### ⚙️ 实验设置与评估指标
- **模型**：
  - 文本模型：`Llama-3.1-8B`, `Llama-3.1-70B`, `Yi-9B-200K`
  - 多模态模型：`LLaVA-1.5-7B`, `LLaVA-1.5-13B`
- **硬件平台**：NVIDIA RTX PRO 6000 GPU，bf16 精度
- **评估指标**：
  - **准确性**：各 benchmark 的标准评分（如 accuracy, F1, AUPRC）
  - **效率**：端到端 per-token 解码延迟、speedup 倍数
- **预算对齐**：所有方法控制在约 **40% KV-cache 保留率** 下进行公平比较

### 🆚 基线方法对比
| 类型 | 基线方法 |
|------|--------|
| **Token/KV 剪枝** | H2O, Scissorhands, SNAPKV |
| **Sink 利用/保留** | StreamingLLM |
| **自适应缓存策略** | FastGen |
| **多模态剪枝** | LOOK-M, FASTV |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ 准确性保持优异
- 在 **LoNGBENCH** 上，SinkRouter 相比 Full Attention 的平均性能下降极小：
  - `Llama-3.1-8B`: -0.69%
  - `Yi-9B-200K`: -1.35%
  - `Llama-3.1-70B`: -0.08%
- 显著优于其他基线（如 FastGen 平均下降达 -4.47%，StreamingLLM 达 -17.76%）

#### ✅ 极端上下文下仍稳定
- 在 **INFINITEBENCH** 上，SinkRouter 在 `kv_retrieval` 和 `math_find` 等精度敏感任务上表现几乎与 Full Attention 一致（+0.04% avg），而 FastGen 下降达 -3.95%

#### ✅ 多模态场景同样有效
- 在 **MMVP/CVBench/MileBench** 上，SinkRouter 与 Full Attention 差距极小（平均 <0.5%），远优于 LOOK-M（如 MileBench 上 -5.77%）

#### ⚡ 显著加速效果
- 在 **512K 上下文长度** 下，实现最高 **2.03× 端到端 speedup**
- 随着上下文增长，加速比持续提升（见 Figure 7）：
  - 64K → 1.24×
  - 128K → 1.40×
  - 256K → 1.67×
  - 512K → 2.03×
- 加速主要来源于 **attention 路径延迟大幅降低**，尤其在长上下文中占主导地位

#### 🔍 消融实验支持设计选择（附录）
- **阈值校准有效性**：采用长度自适应阈值（length-dependent threshold）比固定阈值更稳定，skip ratio 控制更精准（Appendix A）
- **proxy 可靠性**：`cos(q, K_BOS)` 作为路由代理具有高 AUPRC（0.773），在 KV-group 级别具备良好预测能力（Appendix C）
- **layer-wise 分析**：BOS-dominant heads 确实产生更小的 residual write 且方向不一致，验证其“弱贡献”假设（Appendix B）

---

## 4. 关键结论和发现

### 🎯 主要结论
1. **Attention sink 是一种可利用的机制性现象**，而非需要保留的“锚点”。它本质上是一个 **低影响、易到达、可控的 e-fixed point**，可用于识别冗余计算。
2. **SinkRouter 成功将 sink 从“被动保留对象”转变为“主动跳过信号”**，实现了无需训练的高效 head-level 路由。
3. **系统级优化至关重要**：配套的 Triton kernel 支持 block-level 分支与 Split-K 并行，确保算法稀疏性转化为实际硬件加速。
4. **方法通用性强**：在多种 LLM 和 LMM 架构上均有效，适用于不同规模和模态的模型。

### ⚠️ 局限性
- 当前方法主要依赖 BOS token 的 sink 特性，在非 BOS 初始化或 sink 不明显的模型中可能效果受限。
- 路由决策基于单一 proxy（cosine similarity），可能存在误判，尤其是在语义复杂或边界 case 中。
- 前两层因早期处理特殊性被排除路由，说明机制在浅层行为有所不同。

### 🔮 未来工作方向
- 探索更多类型的 sink（如句首、段落标记）或动态生成的 sink 向量。
- 结合其他稀疏化技术（如 MLP 跳过）构建全模型级 adaptive inference 框架。
- 扩展至 prefill 阶段，进一步压缩长上下文预填充开销。
- 在边缘设备或低带宽场景部署，验证其实际系统收益。

---

> **一句话总结**：  
> SinkRouter 通过机制性理解 attention sink 现象，提出了一种无需训练、硬件友好的 head-level 路由方法，在保持几乎无损准确性的前提下，实现了高达 **2.03× 的长上下文解码加速**，为高效 LLM/LMM 推理提供了新范式。

</details>

---

### 3. [Cloud-native and Distributed Systems for Efficient and Scalable Large Language Models -- A Research Agenda](https://arxiv.org/abs/2604.17227)

**Authors**: Minxian Xu, Jingfeng Wu, Shengye Song, Satish Narayana Srirama, Bahman Javad, Rajiv Ranjan, Devki Nandan Jha, Sa Wang, Wenhong Tian, Huanle Xu, Li Li, Zizhao Mo, Shuo Ren, Thomas Kunz, Petar Kochovski, Vlado Stankovski, Kejiang Ye, Chengzhong Xu, Rajkumar Buyya  
**Category**: cs.DC  
**Published**: 2026-04-21  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.17227v1  

#### Abstract
The rapid rise of Large Language Models (LLMs) has revolutionized various artificial intelligence (AI) applications, from natural language processing to code generation. However, the computational demands of these models, particularly in training and inference, present significant challenges. Tradit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Cloud-native and Distributed Systems for Efficient and Scalable Large Language Models – A Research Agenda*

该论文并非一篇以实验验证为核心的传统研究型论文，而是一篇**愿景性（Vision Paper）的研究议程（Research Agenda）**。它不提出单一的新算法或系统，也不报告具体的实验结果，而是旨在**系统性地梳理当前挑战、整合新兴趋势，并为未来研究指明方向**。

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
论文聚焦于**大规模语言模型（Large Language Models, LLMs）在训练和推理过程中面临的巨大计算、内存、存储和网络资源需求**，以及这些需求对传统云平台和分布式系统的严峻挑战。具体问题包括：
- **资源管理效率低下**：传统基于CPU利用率的`autoscaling`策略无法有效应对LLM特有的突发性（bursty）、长尾延迟和多租户竞争。
- **系统架构不匹配**：传统的微服务和容器化抽象（如Kubernetes）未针对LLM的高带宽通信、KV-cache依赖和相位异构性进行优化。
- **能效与可持续性问题**：LLM的训练和推理消耗大量能源，缺乏碳感知（carbon-aware）的调度机制。
- **部署范式碎片化**：缺乏统一标准，导致跨平台互操作性差、可复现性低。

### 提出了什么新方法或新思路
作为一篇研究议程，其“新方法”体现在**提出了一套完整的、面向未来的系统级研究框架和方向**，而非一个具体的技术方案。核心思路是：**必须将LLM视为一种新型工作负载（workload），并据此重新设计云原生（cloud-native）和分布式系统**。

主要创新点包括：
- **提出了“LLM-aware”系统设计理念**：强调系统软件（如调度器、编排器、运行时）需要理解LLM内部特性（如prefill/decode阶段差异、KV-cache压力、token生成模式）。
- **构建了全面的研究分类体系**：从计算、系统软件、运维管理、隐私安全、数据管理和标准化六个维度系统性地分析挑战。
- **前瞻性地探讨了新兴技术融合**：深入讨论了`serverless inference`、`federated learning`、`quantum computing`、`neuromorphic computing`等如何重塑LLM基础设施。
- **明确了未来十大研究方向**：如`LLM@home`、`personalized LLM`、`incentive models`等，为社区提供清晰路线图。

### 相比现有方法的优势
本研究的优势不在于性能提升，而在于**视角的广度、深度和前瞻性**：
- **综合性强**：首次系统性地将LLM与云原生/分布式系统结合，填补了领域空白。
- **问题导向明确**：直面LLM规模化落地中的真实痛点，而非仅关注模型精度。
- **推动标准化**：呼吁建立统一的接口、基准测试（benchmarking）和部署规范，促进生态健康发展。
- **倡导跨学科协作**：强调AI、系统、硬件、安全、政策等多方合作的重要性。

---

## 2. 核心实验方法和设置

由于本文是**愿景性论文（Vision Paper）**，**并未包含任何原始实验**。

- **无数据集使用**：文中未引用或使用任何用于训练或评估的具体数据集。
- **无实验设置**：没有描述实验环境、硬件配置或代码实现。
- **无基线方法对比**：没有通过实验对比所提方法与现有基线（如Naive Serving、Monolithic Scaling）的性能差异。

文中提到的“state of the art”部分，是对**已有研究成果**（如vLLM, DistServe, Medusa等）的综述与分析，用以支撑其提出的挑战和未来方向，而非作者自己进行的实验。

---

## 3. 主要实验结果和性能指标

**本文没有报告任何实验结果或性能指标**。

文中提及的性能改进均来自对**其他研究工作的引用和总结**，例如：
- 引用`vLLM`实现了高效的`PagedAttention`，显著提升了吞吐量。
- 引用`DistServe`通过分离prefill和decode阶段优化了TTFT（Time to First Token）和TPOT（Time Per Output Token）。
- 引用`Medusa`通过CUDA图预热减少了冷启动延迟。

这些结果被用来论证现有技术的潜力和局限，从而引出更深层次的研究需求。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM是颠覆性工作负载**：其规模、动态性和资源敏感性远超传统AI应用，要求系统设计的根本性变革。
2. **云原生+分布式是必由之路**：只有结合`microservices`、`containerization`、`Kubernetes`的弹性与`distributed systems`的并行、容错能力，才能支撑LLM的高效扩展。
3. **细粒度、智能的资源管理是核心**：未来的调度器需具备预测性（predictive）、多目标（latency, cost, energy）、细粒度（token-level）和LLM-aware的特性。
4. **新兴技术将深度融合**：`serverless`、`federated learning`、`quantum`等不仅是补充，更是重构LLM生态的关键力量。
5. **标准化与协作至关重要**：缺乏统一标准是阻碍LLM产业化的瓶颈，亟需学术界、工业界和政策制定者共同推进。

### 方法的局限性
- **非实证性**：作为观点论文，其结论基于逻辑推演和文献综述，缺乏实验数据直接支持。
- **覆盖面广但深度有限**：每个研究方向都点到为止，难以深入探讨具体技术细节。
- **前瞻性带来的不确定性**：部分未来方向（如量子加速LLM）仍处于早期探索阶段，其实用性有待验证。

### 未来工作方向
论文在第6节明确提出了一系列未来研究方向，包括：
- **硬件加速**：支持稀疏矩阵乘法（SpMM）的专用硬件。
- **量子计算**：探索其在LLM优化子任务中的应用。
- **LLM赋能系统软件**：利用LLM自身来优化云资源管理（如`LLM for Automatic Cloud Management`）。
- **数学基础**：深化软硬件协同设计（co-design）的理论基础。
- **Prompt工程自动化**：发展`prompt compression`和`promptware engineering`。
- **去中心化智能**：构建`Decentralized Intelligence`网络。
- **激励模型**：设计公平的数据与算力共享激励机制。
- **家庭LLM**（`LLM@home`）：推动模型在边缘设备的本地化部署。
- **个性化LLM**：实现高效、安全的用户级模型定制。

---

## 总结

该论文是一份极具价值的**战略蓝图**，而非战术手册。它成功地：
- **定义了问题域**：清晰阐述了LLM规模化面临的核心系统挑战。
- **绘制了路线图**：提出了多层次、多维度的未来研究议程。
- **凝聚了共识**：呼吁跨领域协作，推动LLM基础设施向更高效、可持续、普惠的方向发展。

尽管缺乏实验验证，但其深刻的洞察力和前瞻性的视野，使其成为指导未来几年LLM系统研究的重要文献。

</details>

---

### 4. [CoLLM: A Unified Framework for Co-execution of LLMs Federated Fine-tuning and Inference](https://arxiv.org/abs/2604.16400)

**Authors**: Shaoyuan Huang, Xiaokai Wang, Na Yan, Xiaofei Wang, Wenyu Wang, Yansha Deng  
**Category**: cs.DC  
**Published**: 2026-04-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.16400v1  

#### Abstract
As Large Language Models (LLMs) are increasingly adopted in edge intelligence to power domain-specific applications and personalized services, the quality and efficiency of the LLM post-training phase-including fine-tuning and inference, have become critical due to constrained resources. Although re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**CoLLM: A Unified Framework for Co-execution of LLMs Federated Fine-tuning and Inference**

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

在边缘智能（edge intelligence）场景中，大型语言模型（LLMs）的部署通常需要同时支持 **Federated Parameter-Efficient Fine-tuning (FL PEFT)** 和 **低延迟推理（inference）**。然而，当前系统将这两个任务视为独立的工作负载，导致以下问题：

- **资源争用严重**：fine-tuning 占用大量 GPU 资源，影响推理吞吐量；
- **模型更新延迟**：fine-tuning 的早期高质量改进无法及时用于推理服务；
- **冗余部署开销大**：采用时间复用（temporal multiplexing）或空间复用（spatial multiplexing）需重复加载模型副本，造成内存浪费和调度开销。

### ✅ **提出了什么新方法或新思路**

本文提出 **CoLLM** —— 一种统一的 **LLM 联邦微调与推理协同执行框架**，其核心思想是：

> 在共享的模型副本上实现 fine-tuning 与 inference 的共存与参数实时共享。

为此，CoLLM 引入两个关键技术：

#### （1）**Intra-replica Model Sharing（副本内模型共享机制）**
- 采用 **unmerged inference** 范式，保持 LoRA adapter 与 backbone 分离，避免合并后无法继续训练的问题；
- 设计 **shadow adapter 策略**，维护两组 adapter 参数（`A_act` 和 `A_shd`），通过原子交换实现无锁同步：
  - 推理只读取激活状态的 adapter（`A_act`）；
  - 微调仅写入影子 adapter（`A_shd`）；
  - 完成一步训练后，原子交换两者角色，确保一致性且不阻塞推理。

#### （2）**Two-Timescale Inter-replica Coordination（跨副本两级协调算法，TTCA）**
- **粗粒度（FL round-level）调度器**：基于指数移动平均（EMA）预测请求趋势，优化每轮 fine-tuning 批大小 $ b_R $ 和推理速率 $ v_{i,R} $，平衡长期质量增益与短期效率；
- **细粒度（slot-level）分发器**：动态调整请求分配，并引入 **reactive correction** 机制应对过载/欠载情况：
  - 若 SLO 违规率过高 → 减少 fine-tuning 负载；
  - 若请求到达率偏低 → 增加 fine-tuning 强度以提升模型质量。

### ✅ **相比现有方法的优势**

| 维度 | 传统方法（如 FedLS、dLoRA） | CoLLM |
|------|-------------------------------|--------|
| 部署方式 | 分离部署（separate replicas） | 共享副本（single replica） |
| 模型更新延迟 | 需 redeployment 后才生效 | 实时集成（single-step latency） |
| 内存开销 | 双倍 backbone 加载（>30GB for LLaMA-13B） | 仅一份 backbone + 双 adapter（<1% 开销） |
| 调度灵活性 | 固定策略或静态配置 | 自适应两级协调（TTCA） |
| 性能指标 | 仅关注 throughput 或 accuracy | 提出并优化 **goodput**（质量感知吞吐） |

---

## 2. 核心实验方法和设置

### ✅ **使用的数据集**

| 类型 | 数据集名称 | 用途 | 样本数 |
|------|-----------|------|--------|
| Code Generation | ManimCode, CodeAlpaca, CodeInstruct | fine-tuning & inference | 4.4k ~ 122k |
| Conversation | alpaca, GPTeacher, OpenInstruct | fine-tuning & inference | 50k ~ 499k |

> 每个 replica 分配不同子集模拟 **non-IID 数据异构性**；30% 样本作为推理请求重放。

### ✅ **真实流量轨迹**
- 使用来自 **Azure** 的真实 LLM 服务 trace：
  - `Azure-Code`（代码生成）
  - `Azure-Conv`（对话）
- 请求持续 4 小时，按任务类型路由至对应 replica。

### ✅ **模型与硬件环境**

- **模型**：
  - `LLaMA3.1-8B`
  - `Qwen3-4B`
- **硬件平台**：
  - 2 台服务器，各含 4×NVIDIA A30 GPUs（24GB），通过 RDMA 互联；
  - 构建 8 个逻辑 replica（每个绑定一个 GPU）；
  - 1 个作为 server（聚合参数），其余为 client。

### ✅ **评估指标**

| 指标 | 定义 |
|------|------|
| **Throughput (req/s)** | 满足 SLO（0.8s 延迟）的每秒成功响应请求数 |
| **Goodput (Q-req/s)** | $ \text{throughput} \times Q(R) $，其中 $ Q(R) = 1 / \text{CE Loss} $，衡量“质量感知吞吐” |
| **SLO Violation Rate** | 超出延迟限制的请求占比 |
| **Memory Footprint** | GPU 显存占用情况 |

### ✅ **基线方法对比**

| 基线 | 描述 |
|------|------|
| **dLoRA** | 支持动态批处理和迁移的 LoRA 推理系统（inference-only） |
| **Shepherd** | SLO 感知的 DNN 推理调度系统 |
| **Vanilla PEFT** | HuggingFace 默认 PEFT 设置（inference-only） |
| **FedLS** | 支持联合训练与推理的联邦学习系统（原为 CNN 设计，适配用于 LLM） |

---

## 3. 主要实验结果和性能指标

### ✅ **关键性能数据**

#### 🔹 **推理吞吐量（Throughput）**
- 在 `LLaMA-8B` 上，CoLLM 达到最高吞吐：
  - 对话任务：比 FedLS 高 **1.5×**
  - 代码生成任务：接近 inference-only 系统（dLoRA/Shepherd）
- 在 `Qwen3-4B` 上，所有系统表现接近，但 CoLLM **仍维持并发 fine-tuning**，而其他系统未进行训练。

> 表明 CoLLM 成功抑制了 fine-tuning 对推理的干扰。

#### 🔹 **质量感知吞吐（Goodput）**
- `LLaMA-8B + Code Generation`：
  - 比 dLoRA 高 **2.2×**
  - 比 Shepherd 高 **2.0×**
  - 比 PEFT 高 **3.0×**
  - 比 FedLS 高 **1.4×**
- `Qwen3-4B` 上也有 **1.6×~1.9×** 提升。

> 说明 CoLLM 不仅高效，还能更快地将 fine-tuning 收益转化为实际服务质量。

#### 🔹 **可扩展性测试（Load Scaling）**
- 使用 `Azure-Conv` 流量从 0.5× 到 3× 增强压力：
  - 在高负载（3×）下，FedLS 出现性能崩溃（throughput/goodput 下降）；
  - CoLLM 保持近线性增长，且 goodput 显著领先。

> 展示了 TTCA 在动态负载下的鲁棒性和自适应能力。

### ✅ **消融实验结果**

#### 🔹 **模型共享机制的影响（Microscopic Analysis）**
- **CDF of Inference Quality**：
  - “Model Sharing” 方案中 >96% 请求质量得分 ≥1；
  - “Separated Deployment” 仅 40% 达到该水平；
  - “Inference-only” <10% 达标。
- 结论：**实时集成 fine-tuning 更新显著提升响应质量连续性**。

#### 🔹 **TTCA 的有效性验证（CoLLM vs CoLLM@fixed）**
- `CoLLM@fixed`：关闭 TTCA，使用固定 batch size 和轮询调度；
- 实验显示：
  - 最优固定配置（batch=8）下，goodput 仅为 CoLLM 的 **~40%**；
  - CoLLM 动态调节策略带来 **超过 2× 的 goodput 提升**。

> 证明了两级协调机制对复杂动态环境的关键作用。

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **分离式部署代价高昂**：无论是 temporal 还是 spatial multiplexing，都会引入显著的内存冗余和调度延迟。
2. **协同执行可行且必要**：通过 unmerged inference + shadow adapter 可安全实现 fine-tuning 与 inference 共享参数空间。
3. **early fine-tuning gains 至关重要**：早期迭代带来的质量跃升若不能即时利用，会造成巨大机会损失。
4. **goodput 是更合理的评估指标**：单纯追求 throughput 忽视质量演进，无法反映真实用户体验。
5. **两级协调优于单一层级控制**：结合长期规划与短期反馈，才能兼顾稳定性与敏捷性。

### ⚠️ **方法的局限性**

- 当前实现基于 LoRA 类 PEFT 方法，对全参数微调（full fine-tuning）支持有限；
- shadow adapter 增加约 **O(2r(m+n))** 显存开销（虽小于 1%，但在极端资源受限设备可能成为瓶颈）；
- 假设通信开销可忽略，在广域网或弱网络条件下需进一步优化聚合频率；
- 实验集中在文本任务，对多模态 LLM 的适用性有待验证。

### 🔮 **未来工作方向**

- 扩展至更多 PEFT 方法（如 QLoRA、IA³）及量化感知协同执行；
- 引入 **priority-aware dispatching**，支持差异化 SLO 的混合任务；
- 探索 **client-side proactive coordination**，减少中心化调度依赖；
- 结合 RLHF 或 user feedback 构建闭环优化 pipeline，实现持续进化；
- 在真实边缘设备（如手机、IoT）上部署原型系统，验证端到端可行性。

---

> 📌 **一句话总结**：  
> CoLLM 首次实现了 **LLM 联邦微调与推理在共享副本上的安全、高效协同执行**，通过 **模型共享机制 + 两级协调算法**，在不牺牲推理性能的前提下，将 fine-tuning 的质量收益近乎实时地带入服务过程，最终实现高达 **3× 的 goodput 提升**，为边缘侧 LLM 的持续演进提供了全新范式。

</details>

---

### 5. [Towards Intelligent Legal Document Analysis: CNN-Driven Classification of Case Law Texts](https://arxiv.org/abs/2604.17674)

**Authors**: Moinul Hossain, Sourav Rabi Das, Zikrul Shariar Ayon, Sadia Afrin Promi, Ahnaf Atef Choudhury, Shakila Rahman, Jia Uddin  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.17674v1  

#### Abstract
Legal practitioners and judicial institutions face an ever-growing volume of case-law documents characterised by formalised language, lengthy sentence structures, and highly specialised terminology, making manual triage both time-consuming and error-prone. This work presents a lightweight yet high-a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题：** *Towards Intelligent Legal Document Analysis: CNN-Driven Classification of Case Law Texts*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
法律从业者面临日益增长的判例文书（case law）数量，这些文本具有正式语言、复杂句法结构和大量专业术语（如拉丁语引用），导致人工分类耗时且易出错。现有主流方法依赖于大规模预训练模型（如BERT），存在计算资源消耗大、推理延迟高、部署成本高等问题。

本文旨在解决以下挑战：
- 如何在保持高准确率的同时降低模型复杂度；
- 如何有效处理法律文本中的形态变异（morphological variability）、罕见词和 out-of-vocabulary（OOV）术语；
- 如何构建轻量级、高效且可解释的法律文本分类系统。

---

### 🚀 提出的新方法与创新点
提出了一种**轻量级但高性能的CNN驱动框架**，用于判例引证处理（citation-treatment）分类任务，其核心由三部分组成：

1. **Lemmatisation-based Preprocessing（基于词形还原的预处理）**  
   - 使用WordNet词形还原器将不同屈折形式统一为标准词元（如“citing”, “cited” → “cite”），减少词汇稀疏性和噪声。
   
2. **Subword-aware FastText Embeddings（子词感知嵌入）**  
   - 利用FastText的字符n-gram机制生成词向量，能有效表示罕见法律术语、拉丁短语和未登录词（OOV），无需手动构建词典。

3. **Multi-kernel 1D-CNN 分类器**  
   - 采用多个卷积核（kernel sizes = {2, 3, 5}）并行提取多尺度文本特征：
     - kernel=2：捕捉双词引证线索（bigram cues）
     - kernel=3：识别中等长度法律短语
     - kernel=5：建模长句结构和从句模式
   - 全局最大池化 + 拼接 + Dropout + Softmax完成最终分类。

该架构被称为：**CNN + FastText + Lemmatisation**

---

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **准确性** | 超越fine-tuned BERT，在accuracy和F1上均取得SOTA表现 |
| **效率** | 参数仅5.1M（BERT为110M），推理延迟低至0.31ms/文档（比BERT快13倍以上） |
| **资源需求** | 显存占用小（1.3GB GPU Memory），适合边缘部署和实时分析 |
| **可复用性** | 预处理流程通用性强，适用于其他法律文本分类任务 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Legal Citation Text Classification Dataset** [Bansal, 2022]
- 来源：Kaggle公开数据集
- 规模：约25,000条标注判例文书
- 字段：
  - `Case ID`：唯一标识符
  - `Case Outcome`：引证处理类别标签（共5类：Positive, Neutral, Negative, Distinguished, Overruled）
  - `Case Title` 和 `Case Text`：案件标题与全文意见书
- 数据划分：**75%训练集 / 25%测试集**，分层抽样保证类别平衡

---

### ⚙️ 实验设置
- **开发环境**：
  - Python 3.10, PyTorch 2.1
  - 硬件：NVIDIA RTX 3060 (12GB VRAM), Intel i7-12700F, 16GB RAM
- **超参数配置**：
  - 优化器：Adam (`lr=1e-3`)
  - Batch Size：32
  - Epochs：最多50轮，早停策略（3轮无提升即停止）
  - Dropout Rate：0.4
  - FastText维度：500维，上下文窗口=3，最小频次=2
- **评估指标**：
  - Accuracy, Precision, Recall, Macro-F1
  - AUC-ROC（Receiver Operating Characteristic Area Under Curve）
  - 推理延迟（Inference Latency）
  - 模型大小（Params）、GPU内存占用

---

### 🆚 基线方法对比
共比较四种代表性基线模型：
| 模型 | 特点 |
|------|------|
| **KNN (TF-IDF)** | 传统方法，可解释性强但缺乏语义理解能力 |
| **CNN (Random Embedding)** | 卷积网络基础版，使用随机初始化词向量 |
| **LSTM + FastText** | 序列建模能力强，利用子词信息，训练慢 |
| **BERT (Base, Fine-tuned)** | 当前主流Transformer模型，性能强但资源开销大 |
| **Legal-BERT / BERT-CNN (Reported)** | 引用已有研究结果作为参考 |

所有模型在相同预处理条件下进行公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 性能汇总（见Table 2）

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | AUC (%) |
|-------|--------------|----------------|------------|--------|---------|
| KNN (TF-IDF) | 89.42 | 88.95 | 89.42 | 88.60 | 90.10 |
| CNN (Random) | 93.51 | 93.20 | 93.51 | 92.88 | 94.21 |
| LSTM + FastText | 95.68 | 95.40 | 95.68 | 95.10 | 96.02 |
| BERT (Fine-tuned) | 97.12 | 97.05 | 97.12 | 96.88 | 97.45 |
| **Proposed (Ours)** | **97.26** | **97.28** | **97.26** | **96.82** | **97.83** |

✅ **关键发现**：
- 所有指标全面超越基线模型，尤其是**AUC达到97.83%**，显示极强的类别区分能力。
- 在accuracy上略胜BERT（97.26% vs 97.12%），F1也更高（96.82% vs 96.88%接近持平）。
- ROC曲线最贴近左上角，表明灵敏度与特异性最优。

---

### 🔍 消融实验结果（Ablation Study）

探究不同卷积核组合对性能的影响（Table 3）：

| Kernel Sizes | Accuracy (%) |
|--------------|-------------|
| [3]          | 95.8        |
| [3, 4]       | 96.4        |
| **[2, 3, 5]** (**Proposed**) | **97.26**   |

📌 结论：
- 多尺度卷积核显著提升性能；
- kernel=2 捕捉短引证模式（如“cited”、“refers to”）至关重要；
- 三种尺度结合提供非冗余特征，实现最佳权衡。

此外，消融验证了各组件贡献：
- 移除Lemmatisation → 准确率下降约1.2%
- 使用原始文本而非FastText训练embedding → 下降约1.5%
- 表明**Lemmatisation + FastText + Multi-kernel CNN**三者协同增效。

---

### 📈 其他重要实验观察
- **训练稳定性良好**：训练/验证准确率收敛一致，gap小于1%，无明显过拟合（Figure 4）。
- **鲁棒性测试**：在token embedding中加入高斯噪声（σ=0.05）后，准确率仍保持在95.9%，仅下降1.3%，说明模型抗干扰能力强。
- **混淆矩阵分析**（Figure 3）：
  - 错误主要集中于语义相近类别之间：
    - Positive ↔ Neutral
    - Distinguished ↔ Overruled
  - 这反映了法律语言本身的模糊性，即使是人类标注员也可能难以区分。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **轻量CNN架构可媲美甚至超越重型Transformer模型**  
   尽管BERT类模型在自然语言任务中占主导地位，但在特定领域（如法律文本）中，经过精心设计的CNN结构通过**多尺度局部特征提取 + 子词嵌入 + 形态规范化**，能够实现同等甚至更优性能。

2. **Lemmatisation + FastText 是处理法律术语的关键**  
   法律文本中频繁出现变体动词（如“distinguish”, “distinguishing”, “distinguished”）和拉丁术语（如“obiter dicta”），传统word-level embedding难以覆盖。而本方案通过词干归一化与子词分解，显著提升了OOV处理能力和语义一致性。

3. **推理效率极具优势**  
   - 参数量仅为BERT的 **~4.6%**
   - 推理速度是BERT的 **13.7倍**（0.31ms vs 4.25ms）
   - 单epoch训练时间仅26.8秒（BERT需215.6秒）
   - 极适合部署于司法机构内部系统或移动端应用。

---

### ⚠️ 局限性
1. **语言限制**：当前模型基于英文法律语料训练，尚未验证在多语言或多司法管辖区（multijurisdictional）场景下的泛化能力。
2. **上下文建模有限**：CNN主要捕获局部模式，缺乏Transformer那样的全局注意力机制，在需要长距离依赖的任务中可能受限。
3. **可解释性不足**：虽然比BERT更轻便，但仍属于黑箱模型，缺乏对判决依据的显式解释支持。

---

### 🔮 未来工作方向
1. **融合轻量级Transformer模块**：引入如Linformer或MobileBERT等压缩注意力机制，增强长程依赖建模能力。
2. **跨域迁移学习**：扩展至不同国家/地区的法律体系（如中国、欧盟法规）以提升通用性。
3. **集成Explainable AI技术**：结合注意力可视化、LIME或SHAP方法，提高预测结果对律师和法官的可信度。
4. **结合Retrieval-Augmented Generation（RAG）**：引入先例检索机制，提升判断依据的透明性和合理性。
5. **数据增强策略探索**：针对低资源法律领域，使用生成式方法扩充训练样本。

---

## ✅ 总结
本文提出了一种**高效、准确、轻量化的法律文本分类框架 CNN + FastText + Lemmatisation**，在引证处理任务上实现了**97.26% accuracy** 和 **97.83% AUC** 的SOTA性能，同时具备远优于BERT的推理效率。实验证明，**合理的结构设计与领域适配的预处理策略**，可以在不依赖大规模Transformer的情况下，实现智能法律文档分析的实用化落地，为下一代AI司法辅助系统提供了可行路径。

</details>

---

### 6. [HieraSparse: Hierarchical Semi-Structured Sparse KV Attention](https://arxiv.org/abs/2604.16864)

**Authors**: Haoxuan Wang, Chen Wang  
**Category**: cs.DC  
**Published**: 2026-04-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.16864v1  

#### Abstract
The deployment of long-context Large Language Models (LLMs) poses significant challenges due to the intense computational cost of self-attention and the substantial memory overhead of the Key-Value Cache (KV Cache). In this paper, we introduce HieraSparse, a hierarchical KV Cache compression framewo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HieraSparse: Hierarchical Semi-Structured Sparse KV Attention 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在处理长上下文时面临两大瓶颈：
- **计算瓶颈**：自注意力机制具有 $O(n^2)$ 的时间复杂度，在长序列下显著增加 `Time-to-First-Token`（TTFT） 和 `Time-per-Output-Token`（TPOT）。
- **内存瓶颈**：Key-Value Cache（KV Cache）随序列长度线性增长，导致显存占用巨大，甚至超出GPU容量。

现有方法如 **unstructured sparsity**（例如 MUSTAFAR）虽然能实现细粒度剪枝，但由于采用“load-as-sparse, compute-as-dense”模式，无法将稀疏性转化为实际加速，尤其在计算密集的 prefill 阶段效果有限。

---

### 提出的新方法与创新思路
作者提出 **HieraSparse** —— 一种支持硬件加速的分层半结构化稀疏 KV Cache 压缩框架，其核心创新包括：

#### ✅ 分层稀疏设计（Hierarchical Block-Based Management）
- 支持混合存储：**dense blocks** 与 **N:M semi-structured sparse blocks** 共存。
- 可灵活控制块级（block-level）和元素级（element-level）的稀疏模式，保留关键区域（如 attention sinks、local windows）为 dense，其余进行压缩。

#### ✅ 利用 Sparse Tensor Core 加速
- 首次将 **GPU 的 N:M sparse tensor cores** 应用于 KV Cache attention，真正实现“稀疏即高效”。
- 设计了基于 **Trans-Both** 架构的 attention kernel：通过转置两个 GEMM 操作，使 K 和 V 成为稀疏操作数，从而适配 sparse tensor core 要求。

#### ✅ 支持 Prefill 和 Decode 两阶段加速
- 现有方法多只优化 decode 阶段；HieraSparse 是首个同时在 **prefill 和 decode** 阶段实现 semi-structured sparse acceleration 的系统。
- 实现了在线 near-zero-overhead 的压缩 kernel，避免格式转换带来的延迟开销。

#### ✅ 高效内核实现
- 引入多项优化技术：
  - **异步流水线（Asynchronous Pipelining）**：隐藏 HBM 到 shared memory 的加载延迟。
  - **寄存器级重布局（In-fragment Re-layout）**：利用 `movmatrix` 指令完成 PT 矩阵转置，无需共享内存或 warp shuffle。
  - **片上内存复用与专用 kernel**：节省 shared memory 占用，提升 occupancy。

---

### 相比现有方法的优势
| 维度 | HieraSparse | MUSTAFAR（代表 SOTA unstructured 方法） |
|------|-----------|-----------------------------|
| 稀疏类型 | N:M semi-structured | Unstructured |
| 是否可被硬件加速 | ✅ 是（sparse tensor core） | ❌ 否（compute-as-dense） |
| 支持 prefill 加速 | ✅ 是 | ❌ 否 |
| 实际速度提升 | 高达 4.57× attention speedup | 接近无加速甚至变慢 |
| 内存压缩率 | 更高（理论最高 ~1.78×） | 较低（约 1.5×） |
| 压缩开销 | <0.5% prefill latency | 最高达 11.7% |

---

## 2. 核心实验方法和设置

### 数据集
- **LongBench**：一个广泛使用的多任务长文本理解基准，涵盖单文档问答、多文档问答、摘要、少样本学习、合成任务和代码补全等共16项任务。
- 用于评估生成质量（generation quality），报告平均得分。

### 实验设置
- **模型**：
  - Llama-3.1-8B-Instruct
  - Mistral-7B-Instruct-v0.2
  - Qwen3-8B
- **硬件平台**：
  - NVIDIA L40S GPU（48GB DRAM）
  - CUDA 12.8, PyTorch 2.10.0
- **上下文长度**：从 32K 到 160K 不等
- **批大小（batch size）**：8（prefill），1（decode）
- **稀疏配置**：
  - 固定前 64 个 “sink” tokens 和最后 256 个 “local window” tokens 为 dense。
  - 其余部分按 block 进行 N:M（2:4）剪枝。

---

### 评估指标
| 类别 | 指标 |
|------|------|
| **效率** | Attention kernel latency, Speedup (vs. dense), Memory compression ratio, End-to-end TTFT & TPOT |
| **质量** | LongBench 平均得分 |
| **系统开销** | Compression overhead (%) |

---

### 基线方法对比
- **Dense Baseline**：标准 dense KV Cache + FlashAttention
- **MUSTAFAR**：当前最先进的 fine-grained unstructured KV Cache pruning 方法，仅支持 decode 阶段

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔹 Attention Kernel 性能（32K context, batch=8）
| 方法 | Prefill Speedup | Decode Speedup | 压缩开销 |
|------|---------------|----------------|----------|
| HieraSparse (Sk=1.0, Sv=1.0) | **1.85×** | **1.71×** | <0.5% |
| HieraSparse (Sk=0.0, Sv=1.0) | 1.36× | 1.28× | <0.5% |
| MUSTAFAR (K0.5, V0.5) | N/A | ~0.37×（相对更慢） | 11.7% |

> 💡 注：MUSTAFAR 在 decode 上反而比 dense 更慢，因其需频繁解压 bitmap。

#### 🔹 与 MUSTAFAR 在相同稀疏度下的对比
- 在相同 KV Cache 稀疏水平下：
  - HieraSparse 实现 **1.2× 更高的 KV 压缩比**
  - 实现 **4.57× 的 attention 速度提升**

#### 🔹 内存压缩效率
- 理论最大压缩比可达 **1.78×**（当 Sk=Sv=1.0）
- 实测压缩率与理论值高度一致，优于 MUSTAFAR 最多 **1.2×**

#### 🔹 端到端性能（Llama-3.1-8B-Instruct, 128K context）
| 方法 | TTFT (Speedup) | TPOT (Speedup) | Peak Memory |
|------|----------------|----------------|-------------|
| Dense | 39.8s | 98.4ms | 34.62 GiB |
| HieraSparse (Sk=0.0, Sv=1.0) | 32.5s (**1.22×**) | 80.7ms (**1.22×**) | 31.12 GiB |
| HieraSparse (Sk=1.0, Sv=1.0) | 28.8s (**1.41×**) | 64.0ms (**1.54×**) | 27.62 GiB |

---

### 消融实验结果（Ablation Study）

#### 不同稀疏策略对生成质量的影响（Table IV & Fig. 6）
| 设置 | LongBench Score | Prefill Speedup | Decode Speedup |
|------|------------------|------------------|----------------|
| Prefill: Sk=0.0, Sv=1.0<br>Decode: Sk=0.0, Sv=1.0 | 47.32 (Llama) | 1.34× | 1.28× |
| Prefill: Sk=0.0, Sv=1.0<br>Decode: Sk=1.0, Sv=1.0 | 45.59 (Llama) | 1.34× | **1.71×** |

> 发现：**key cache 对剪枝更敏感**，因此在 prefill 阶段保持 key dense 可有效维持生成质量。

#### 内核优化贡献（Fig. 4）
各优化对 prefill kernel 的加速贡献：
- **基础 sparse kernel**：~1.17×
- + **Async Pipeline**：→ 1.30×
- + **In-fragment Re-layout**：→ 1.34×
- + **Shared Memory Merge & Specialization**：→ **1.85×**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **半结构化稀疏 + sparse tensor core 是实现真实加速的关键路径**，远胜于 unstructured sparsity。
2. ✅ **HieraSparse 成功将稀疏性转化为效率**：在 prefill 和 decode 两阶段均实现显著加速（最高达 1.85× 和 1.71×）。
3. ✅ **分层设计提供了灵活的质量-效率权衡机制**：可通过调节 block sparsity（Sk/Sv）平衡性能与精度。
4. ✅ **key cache 比 value cache 更重要**：剪枝 key 会导致更大性能下降，建议在 prefill 中优先保留 key dense。
5. ✅ **压缩开销极低**：<0.5%，几乎不影响 TTFT。

---

### 方法的局限性
1. **依赖特定硬件**：必须使用支持 N:M sparse tensor core 的 GPU（如 NVIDIA Ampere/Hopper 或 AMD MI300X）。
2. **magnitude-based pruning 精度有限**：当前采用简单幅值剪枝，未探索更高级的离线学习型剪枝策略。
3. **不兼容所有 attention 变体**：某些特殊 attention 结构可能难以适配 Trans-Both 设计。
4. **对 Mistral 等模型更敏感**：不同架构对结构性剪枝的鲁棒性存在差异，需调参。

---

### 未来工作方向
1. **结合更智能的剪枝算法**：探索基于训练或 prompt-aware 的 fine-grained pruning，进一步释放 prefill 加速潜力。
2. **支持 unstructured to structured 映射**：借鉴 TASDER、VENOM 等工作，动态将非结构稀疏映射到结构稀疏硬件上。
3. **与量化联合优化**：结合 KV Cache 的 low-bit quantization（如 KIVI）以进一步降低内存和带宽需求。
4. **扩展至 prefix caching 场景**：在 RAG、memory-augmented agent 中实现高效的缓存复用与压缩。

---

> 📚 **开源地址**：https://github.com/psl-ntu/HieraSparse

</details>

---

### 7. [Sampling for Quality: Training-Free Reward-Guided LLM Decoding via Sequential Monte Carlo](https://arxiv.org/abs/2604.16453)

**Authors**: Jelena Markovic-Voronov, Wenhui Zhu, Bo Long, Zhipeng Wang, Suyash Gupta, Kayhan Behdin, Bee-Chung Chen, Deepak Agarwal  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.16453v1  

#### Abstract
We introduce a principled probabilistic framework for reward-guided decoding in large language models, addressing the limitations of standard decoding methods that optimize token-level likelihood rather than sequence-level quality. Our method defines a reward-augmented target distribution over compl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Sampling for Quality: Training-Free Reward-Guided LLM Decoding via Sequential Monte Carlo**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 LLM 解码策略（如 beam search、nucleus sampling）主要优化 **token-level likelihood**，而非整个生成序列的**整体质量**。这在需要全局一致性的任务中（如代码生成、数学推理）存在明显缺陷。

此外，尽管已有工作引入 reward signals（如 verifier、process reward model）来评估输出质量，但这些信号通常以启发式方式使用（如 best-of-N、re-ranking），并未真正融入采样过程，导致无法定义一个统一的概率目标分布。

---

### ✅ 提出的新方法与核心思想

本文提出一种**训练免费（training-free）**、基于 **Sequential Monte Carlo (SMC)** 的 reward-guided 解码框架，其核心是：

#### 🌟 方法名称：
**Reward-augmented probabilistic decoding via SMC**

#### 🧠 核心思想：
- 定义一个新的**奖励增强的目标分布**（reward-augmented target distribution）：
  
  $$
  \pi(x_{1:T}|q) \propto \prod_t m_t(x_t|q,x_{<t}) \cdot \prod_t p_t(x_{1:t}, q)
  $$

  其中：
  - $m_t$ 是从 base LLM 导出的 transition factor；
  - $p_t$ 是 prefix-dependent reward potential（例如来自 verifier 或 PRM 的打分）；

- 该方法不修改模型权重，仅在 inference 阶段通过 SMC 对这个新分布进行采样，实现“无训练提升”。

#### 🔍 创新设计：
1. **两种中间目标（intermediate targets）**：
   - **Prefix-only target**：计算高效，但不能准确匹配最终分布的边际。
   - **Lookahead target**：引入 future-value correction（即 lookahead term $L_t$），使得中间分布**精确等于完整路径分布的边际**，从而显著减少方差。

2. **Block-wise Resample-Move SMC with MH Rejuvenation**：
   - 支持按块生成（block-wise generation），提高效率；
   - 在 resampling 后对低质量重复粒子执行 **Metropolis-Hastings (MH) rejuvenation**，恢复多样性并引导向高质量轨迹演化。

3. **Decoupling SMC 和 MH 的目标函数**：
   - SMC 阶段使用 **Target I（tempered base） + prefix-only weights** → 快速探索；
   - MH rejuvenation 使用 **Target II（powered base） + lookahead weights** → 精准修正；
   - 实现“快慢结合”的协同优化。

---

### ✅ 相比现有方法的优势

| 特性 | 本方法 | Best-of-N / Re-ranking | Power Sampling (e.g., MCMC/SMC) | RL Fine-tuning |
|------|--------|--------------------------|-------------------------------|---------------|
| 是否需训练 | ❌ 否（training-free） | ❌ 否 | ❌ 否 | ✅ 是 |
| 是否整合 reward 到采样分布 | ✅ 是（principled） | ❌ 否（post-hoc） | ⚠️ 有限（无 prefix reward） | ✅ 是 |
| 是否建模 sequence-level 质量 | ✅ 是 | ⚠️ 间接 | ✅ 是（power prior） | ✅ 是 |
| 是否支持 lookahead | ✅ 是（online 估计） | ❌ 否 | ❌ 否（prefix-only） | ❌ 否 |
| 可扩展性与效率 | ✅ 高（block-wise + MH） | ❌ 随 N 增长迅速饱和 | ✅ 中等 | ❌ 成本高 |

> 💡 总结：这是首个将 **reward potentials**、**exact lookahead marginals** 和 **SMC with MH rejuvenation** 统一在一个 principled 概率框架下的 inference-time 方法。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **HumanEval**：评估代码生成功能正确性（pass@1）；
- **MATH500**：数学问题求解，评估最终答案 exact match；
- **GPQA Diamond Split**：科学领域多选题，测试复杂推理能力。

---

### 🧪 实验设置

| 设置项 | 描述 |
|-------|------|
| **Base Models** | Qwen2.5-7B, Qwen2.5-Math-7B, DeepSeek-Math-7B |
| **最大长度** | 3072 tokens，遇到 EOS 提前终止 |
| **实现平台** | 基于 vLLM 构建统一 inference-time 框架 |
| **超参数默认值** | $\alpha=4.0$, $N=16$ particles, $J=2$ lookahead rollouts, $S=2$ MH steps |
| **Block Size $B$** | HumanEval: 64；MATH/GPQA: 512（适应不同推理粒度） |
| **Rollout Temp $T_{\text{rol}}$** | HumanEval: 0.1；MATH/GPQA: 0.3 |

---

### 🔁 奖励信号设计
- **HumanEval**：
  - 主要 reward：unit test 执行结果（timeout 5s）
  - 辅助 reward：syntax reward（权重 0.3）
- **MATH500 & GPQA**：
  - 使用 Process Reward Models（PRM）逐 step 打分：
    - Act-X（MATH）
    - ThinkPRM-1.5B（GPQA）

---

### 🆚 基线方法对比
| 类型 | 方法列表 |
|------|---------|
| 基础解码 | Base（temp=1）、Low-temp sampling |
| 采样增强 | Best-of-N |
| MCMC 方法 | MCMC Power Sampling (Karan & Du, 2025) |
| SMC 方法 | Scalable Power Sampling (Ji et al., 2026), Power-SMC (Azizi et al., 2026) |
| Reward-guided SMC | SMC (reward) (Lew et al., 2023) — 仅 prefix，无 lookahead/MH |
| 强化学习 | GRPO（available 时作为上限参考） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（Pass@1）

| Model | Task | Base | Best-of-N | SMC (reward) | **Ours** | GRPO |
|-------|------|------|-----------|--------------|----------|------|
| Qwen2.5-7B | HumanEval | 0.329 | 0.609 | 0.787 | **0.878** | 0.561 |
| Qwen2.5-7B | MATH500 | 0.498 | 0.650 | 0.710 | **0.790** | 0.740 |
| Qwen2.5-7B | GPQA | 0.278 | 0.282 | 0.323 | **0.384** | 0.354 |
| Qwen2.5-Math-7B | HumanEval | 0.329 | 0.512 | 0.750 | **0.854** | 0.537 |
| DeepSeek-Math-7B | HumanEval | 0.415 | 0.433 | 0.628 | **0.781** | 0.524 |

> ✅ **最高提升达 +54.9%（绝对增益）**，远超所有 baseline。

---

### 🆚 与最强基线对比
- 在 HumanEval 上，相比 Scalable Power Sampling 最高领先 **+12.2%**；
- 相比 best-of-N 提升 **+9.1% ~ +15.3%**；
- 即使面对专门用于数学任务微调的 **GRPO（RL 方法）**，仍全面超越：
  - HumanEval 上高出最多 **+31.7%**；
  - MATH500 上也显著领先（如 Qwen2.5-7B: 0.790 vs 0.740）。

---

### 🔍 消融实验结果（Ablation Study）

#### 🎯 Lookahead 机制的影响
比较 **ours (with lookahead + MH)** vs **SMC (reward) (prefix-only, no MH)**：

| Model | HumanEval Δ↑ | MATH500 Δ↑ |
|-------|---------------|------------|
| Qwen2.5-7B | +9.1% | +8.0% |
| Qwen2.5-Math-7B | +10.4% | +2.6% |
| DeepSeek-Math-7B | **+15.3%** | **+8.8%** |

> 💡 发现：lookahead 尤其在长程依赖强的任务中作用巨大，可避免陷入局部最优。

---

#### 📈 Token Budget vs Performance（图1）
- **Best-of-N 与普通 SMC (reward)** 在 $N > 16$ 后趋于饱和（约 0.73 pass@1）；
- **本方法持续提升**，在 ~144K tokens/problem 时达到 **0.94 pass@1**；
- 表明性能增益并非来自更多采样，而是更高效的 compute allocation。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Sequence-level reward modeling 至关重要**：
   - 将 reward potentials 显式嵌入目标分布，比 post-hoc ranking 更有效。

2. **Lookahead 是突破瓶颈的关键**：
   - Prefix-only 方法容易陷入 myopic sampling；
   - Lookahead 通过估计未来 reward landscape，实现全局导向。

3. **Training-free 方法也能超越 RL 微调**：
   - 在多个任务上超过 GRPO，说明高质量推理可通过 inference-time 方法激发。

4. **SMC + MH rejuvenation 提供优雅平衡**：
   - SMC 快速收敛到高 reward 区域；
   - MH 修复退化、提升探索，形成正反馈循环。

---

### ⚠️ 局限性

1. **计算开销较高**：
   - 需要多次 rollout 来估计 lookahead term $L_t$；
   - 不适合极低延迟场景。

2. **依赖外部 reward model 质量**：
   - 若 PRM 或 verifier 不可靠，可能误导搜索方向。

3. **当前为 deterministic 任务设计**：
   - 对开放生成任务（如对话）适配性有待验证。

---

### 🔮 未来工作方向

1. **Adaptive Compute Allocation**：
   - 动态分配 lookahead rollout 数量，依据粒子不确定性或决策难度。

2. **Early Stopping for High-confidence Particles**：
   - 对确定性强的路径减少计算，聚焦资源于关键分歧点。

3. **用于 RL Data Generation**：
   - 利用本方法生成高质量 reasoning traces，用于后续 RLHF 或 DPO 训练。

4. **扩展至 agent workflows**：
   - 结合工具调用、反思机制，在 multi-step agent 中实现 reward-guided planning。

---

> 📌 **一句话总结**：  
> 本文提出了一个 principled、training-free 的 reward-guided 解码框架，通过 SMC + lookahead + MH rejuvenation，在无需任何模型更新的情况下，实现了在 code、math、scientific reasoning 等任务上的 state-of-the-art 性能，证明了 inference-time 方法的巨大潜力。

</details>

---

### 8. [Multi-Label Phase Diagram Prediction in Complex Alloys via Physics-Informed Graph Attention Networks](https://arxiv.org/abs/2604.16468)

**Authors**: Eunjeong Park, Amrita Basak  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.16468v1  

#### Abstract
Accurate phase equilibria are foundational to alloy design because they encode the underlying thermodynamics governing stability, transformations, and processing windows. However, while the CALculation of Phase Diagrams (CALPHAD) provides a rigorous thermodynamic framework, exploring multicomponent ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Multi-Label Phase Diagram Prediction in Complex Alloys via Physics-Informed Graph Attention Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
- **多组分合金相图预测的高计算成本**：传统CALPHAD方法虽精确，但在高维成分-温度空间中进行密集采样非常耗时，难以支持快速合金筛选。
- **现有机器学习模型缺乏物理一致性**：纯数据驱动的ML模型常输出热力学上不可能的相组合（如违反Gibbs相律），尤其在相边界附近产生不合理的多相共存。

### 🚀 提出的新方法与创新思路
- **提出一种基于GATv2的Physics-Informed Graph Attention Network (PI-GAT)**：
  - 将每个成分-温度状态表示为一个四节点的“元素图”（element graph），节点特征包含原子分数和Magpie描述符。
  - 使用**Graph Attention Network (GATv2)** 学习元素间的交互权重，捕捉跨元素稳定性关系。
  - 首次将**multi-label phase-set prediction**应用于完整相集合预测，而非仅预测相数或液相线温度等简化任务。

- **引入双重物理约束机制**：
  1. **Physics-Informed Loss**：训练阶段加入三项轻量级热力学惩罚项：
     - **Gibbs Phase Rule (GPR)**：限制共存相数量不超过独立组元数。
     - **Local Smoothness**：鼓励邻近成分间预测平滑变化。
     - **Pure Phase Feasibility**：确保纯元素角点只激活单一相。
  2. **Physics-Informed Decoding**：推理阶段采用确定性投影策略，按顺序执行：
     - 角点可行性修正 → 局部概率平滑 → Gibbs基数裁剪。
     - 实现**硬性约束满足**，保证输出严格符合物理规律。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **准确性** | 在dense grid测试中达到~96% exact-set accuracy，优于传统ML方法。 |
| **泛化能力** | 成功外推至未见的ternary和quaternary系统（>91%准确率）。 |
| **物理一致性** | 显著减少非法相组合（如binary中出现3相），提升工程可信度。 |
| **可扩展性** | 图结构天然适配任意组元系统，框架具有合金无关性（alloy-agnostic）。 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **来源**：使用`pycalphad`结合NIST Solder Thermodynamic Database生成约 **25,000个平衡态样本**。
- **体系**：聚焦于**Ag-Bi-Cu-Sn**四元焊料系统及其子系统。
- **覆盖范围**：
  - **6个binary subsystems**（如Ag-Bi, Cu-Sn）
  - **3个in-domain ternary subsystems**（Ag-Bi-Cu, Ag-Cu-Sn, Bi-Cu-Sn）用于训练/验证
  - **1个out-of-domain ternary**（Ag-Bi-Sn）用于外推测试
  - **1个quaternary section at 700°C** 用于高维外推评估
- **标签构建**：从CALPHAD输出的相分数中提取**9个相关相的存在性**（binary label），阈值为 $ \epsilon = 10^{-6} $

### ⚙️ 实验设置
- **输入表示**：
  - 节点特征：原子分数 + 8维Magpie属性（如电负性、原子半径等）
  - 全连接有向图（4 nodes）
- **模型架构**：
  - 3层GATv2（每层4 heads）
  - Global Mean Pooling + 温度拼接 → MLP输出9个phase logits
- **优化配置**：
  - 使用**Optuna**进行贝叶斯超参搜索（learning rate, dropout, hidden dim等）
  - 最优参数：hidden dim=160, lr=1e-3, dropout=0.05, batch size=32
  - 损失函数：Class-balanced Focal Loss
  - 优化器：AdamW + Cosine Annealing LR Schedule

### 📈 评估指标
| 指标 | 定义 |
|------|------|
| **Macro-F1 Score** | 所有相类别的F1平均值，对稀有相更敏感 |
| **Exact-set Accuracy** | 完全匹配真实相集合的比例（即所有相都正确） |
| **Subset Accuracy** | 按binary/ternary子系统分别统计的exact-match率 |
| **Mismatch Count** | 错误预测的样本总数 |

### 🆚 基线方法对比
- **Baseline GNN**：无物理约束的标准GAT模型
- **GNN + Physics-Informed Loss**：分别添加GPR / Smoothness / Pure-phase损失项
- **GNN + Physics-Informed Decoding**：推理阶段应用确定性约束投影
- 所有模型均使用seed ensembling（10 runs平均）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 模型 | Macro-F1 | Exact-set Accuracy (in-domain) | Out-of-Domain (Ag-Bi-Sn) | Quaternary (700°C) |
|------|----------|-------------------------------|---------------------------|--------------------|
| Baseline GNN | 0.9513 ± 0.0141 | ~93.98% | — | — |
| + Physics-Informed Loss | 0.9577 ± 0.0049 | ~94.60% | — | — |
| + Physics-Informed Decoding | **0.9623 ± 0.0035** | **~96%** | **99.32%** | **91.78%** |

> 注：exact-set accuracy指整个相集合完全正确的比例。

### 🔁 与基线方法对比结果
- **相比baseline GNN**：
  - Macro-F1提升约 **1.1个百分点**
  - Exact-set accuracy提升超过 **2个百分点**
  - 方差显著降低（std从±0.014降至±0.0035），表明预测更稳定可靠。
- **Physics-Informed Decoding > Physics-Informed Loss**：
  - 尽管两者都嵌入相同物理知识，但**解码阶段的硬约束效果更好**
  - 因其避免了训练中的梯度冲突，且能**确定性地消除非法输出**

### 🔍 消融实验结果（Ablation Study）
#### （1）不同物理损失的影响（λ_q 扫描）
| Loss Type | 最佳λ | Max Macro-F1 |
|---------|-------|-------------|
| Gibbs Phase Rule | 0.15 | **0.9382** |
| Local Smoothness | 0.10 | 0.9344 |
| Pure Phase Feasibility | 0.05 | 0.9335 |

👉 结论：**GPR损失最有效**，说明控制相多重性是关键。

#### （2）约束顺序的重要性（Decoding阶段）
- 正确顺序：**Pure → Smooth → GPR**
  - 先清除角点非法相
  - 再平滑局部噪声
  - 最后施加全局基数上限
- 若顺序颠倒会导致修复失败或过度裁剪。

#### （3）dense-grid stress test
- 在1 at.% 和 5°C分辨率下生成细密网格进行插值测试：
  - **错误高度集中在相边界附近**（eutectic, peritectic区域）
  - 单相/双相内部区域几乎完全一致
  - 表明模型具备良好的**内插能力**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **PI-GAT能高效学习复杂合金中的相稳定性模式**：
   - 图注意力机制成功捕获了元素间非线性的相互作用。
   - 多标签分类框架更适合实际冶金需求（需知道具体哪些相共存）。

2. **物理信息的引入方式至关重要**：
   - **Training-time soft penalty** 可改善学习动态，但无法保证处处可行。
   - **Inference-time hard projection** 能彻底消除违反Gibbs规则的情况，实现**零违规输出**。

3. **模型具备强外推能力**：
   - 对未参与训练的**Ag-Bi-Sn ternary**达到 **99.32% exact-set accuracy**
   - 对**quaternary system at 700°C**仍保持 **91.78% 准确率**
   - 表明该方法可用于指导新材料探索。

4. **误差分布揭示模型行为**：
   - 错误集中于sharp phase boundaries和invariant reactions
   - 支持将其作为“不确定性指示器”，辅助实验设计优先关注边界区域。

### ⚠️ 方法的局限性
- **依赖高质量CALPHAD数据库**：若底层热力学模型不准，则ML代理也会继承偏差。
- **当前仅预测相存在与否，未建模相分数**：缺少定量信息（如lever rule应用受限）。
- **Magpie描述符可能不足以刻画所有化学趋势**：对于强电子效应或磁性系统可能不足。
- **四元以上系统的可视化与解释难度增加**。

### 🔮 未来工作方向
1. **引入相分数预测**：结合Gibbs能量估计与lever rule约束，实现可微分的相含量预测。
2. **融合更多物理先验**：如扩散势、活度系数、界面能等，增强机理解释性。
3. **扩展至开放数据库生态**：整合Materials Project、OQMD等大规模第一性原理数据。
4. **发展主动学习策略**：结合不确定性估计，自动选择最有价值的成分点进行CALPHAD计算，形成闭环优化流程。
5. **探索foundation model路径**：利用LLM预训练+微调范式处理跨体系相图预测任务。

---

> 💡 **总体评价**：本文提出了一种兼具**高精度、强泛化性和物理保真度**的相图代理模型，通过**graph learning + physics-informed decoding**的协同设计，在保持计算效率的同时大幅提升了预测可靠性，为多组分合金快速筛选提供了有力工具。

</details>

---

### 9. [FedLLM: A Privacy-Preserving Federated Large Language Model for Explainable Traffic Flow Prediction](https://arxiv.org/abs/2604.16612)

**Authors**: Seerat Kaur, Sukhjit Singh Sehra, Dariush Ebrahimi  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.16612v1  

#### Abstract
Traffic prediction plays a central role in intelligent transportation systems (ITS) by supporting real-time decision-making, congestion management, and long-term planning. However, many existing approaches face practical limitations. Most spatio-temporal models are trained on centralized data, rely ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FedLLM: A Privacy-Preserving Federated Large Language Model for Explainable Traffic Flow Prediction

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该研究旨在解决智能交通系统（ITS）中**交通流预测**面临的三大挑战：
- **数据隐私与分布式特性**：真实世界中的交通数据由多个独立机构（如不同区域的交通管理部门）持有，受隐私和治理限制，难以集中化处理。
- **模型可解释性不足**：传统时空深度学习模型（如STGCN、DCRNN）仅输出数值预测，缺乏对预测结果的推理过程，不利于实际决策支持。
- **非独立同分布（non-IID）数据下的泛化能力差**：各区域交通模式差异大，导致集中训练模型在跨区场景下表现不佳。

### 提出的新方法与思路
作者提出 **FedLLM**，一个结合 **Federated Learning (FL)** 与 **Domain-Adapted Large Language Model (LLM)** 的新型框架，用于可解释的短期交通流预测（15–60分钟）。其四大核心创新点如下：

1. **Composite Selection Score (CSS)**  
   - 一种基于多准则的数据驱动高速公路选择方法，综合考虑流量均值、时间变异性、传感器可靠性与空间覆盖度，确保所选训练走廊具有结构多样性，提升模型泛化能力。

2. **领域自适应LLM（Domain-Adapted LLM）**  
   - 将原始数值型交通数据转换为**结构化自然语言提示（structured prompts）**，输入至Qwen2.5-1.5B-Instruct模型，并通过QLoRA进行高效微调。
   - 模型不仅输出预测值，还生成**step-by-step自然语言解释**，实现可解释预测。

3. **集成FL与LLM的联邦框架（FedLLM）**  
   - 首次将LLM应用于联邦设置下的高速公路交通预测任务。
   - 各客户端本地训练，仅交换轻量级的LoRA适配器参数（约70.5MB/轮），而非完整模型权重（2.9GB），显著降低通信开销并保护数据隐私。

4. **结构化Prompt设计支持跨区域迁移**  
   - Prompt编码静态属性（坐标、车道数）、统计摘要（日/周流量模式）、动态上下文（最近12个观测值）及空间邻域信息，使模型具备情境感知与跨区域泛化能力。

### 相比现有方法的优势
| 维度 | 传统方法（如STGCN, DCRNN） | 联邦图模型（如FedASTGCN） | FedLLM |
|------|----------------------------|-----------------------------|--------|
| 数据隐私 | ❌ 中心化训练 | ✅ 联邦架构 | ✅ 联邦 + 参数隔离 |
| 可解释性 | ❌ 数值输出 | ❌ 数值输出 | ✅ 自然语言解释 |
| 通信效率 | ❌ 不适用 | ❌ 全权重传输 | ✅ 仅LoRA参数交换 |
| 泛化能力 | ⚠️ 依赖集中数据 | ⚠️ 有限 | ✅ 强跨区零样本迁移 |

---

## 2. 核心实验方法和设置

### 数据集
- **主数据集**：[**LargeST**](https://arxiv.org/abs/2307.09288) — 来源于加州PeMS系统的大型基准数据集。
  - 时间跨度：2017–2021年（5年）
  - 节点数量：8,600个环形检测器
  - 时间粒度：15分钟聚合
  - 地理范围：涵盖洛杉矶大都会区（District 12）作为训练测试区，旧金山湾区（District 4）用于零样本迁移评估。

### 实验设置
- **训练/测试划分**：
  - 训练集：2019年1月–6月
  - 测试集：2019年7月–12月
- **联邦客户端配置**：
  - 客户端数量：4个异构高速公路走廊（SR261-S, SR57-N, SR133-N, SR133-S）
  - 每轮本地训练步数：200步
  - 通信轮次：2轮
  - 聚合算法：FedAvg（按样本加权）
- **模型初始化**：
  - 所有客户端从预训练的**domain-adapted LLM checkpoint**启动，加速收敛。

### 评估指标
采用四项标准回归指标衡量预测性能：
- **RMSE**（Root Mean Squared Error）
- **MAE**（Mean Absolute Error）
- **MAPE**（Mean Absolute Percentage Error）
- **R²**（Coefficient of Determination）

同时评估**zero-shot cross-region generalization**能力，在未参与训练的GBA区域进行测试。

### 基线方法对比
#### 中央化基线（Centralized Baselines）
- GRU, FC-LSTM（纯时序）
- STGCN, DCRNN, AGCRN, ASTGNN（图神经网络）
- Centralized Qwen（无微调的原始LLM）

#### 联邦基线（Federated Baselines）
- Fed-GDAN（图扩散注意力网络）
- FedASTGCN（拓扑感知联邦框架）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（总体平均）

| 模型 | RMSE ↓ | MAE ↓ | R² ↑ | MAPE ↓ |
|------|--------|-------|------|--------|
| **FedLLM** | **23.31** | **15.07** | **0.985** | **21.84%** |
| Domain-Adapted LLM | 35.66 | 24.38 | 0.960 | 16.02% |
| FedASTGCN | 42.91 | 26.57 | 0.947 | 37.83% |
| STGCN | 43.43 | 32.07 | 0.910 | 21.06% |
| FC-LSTM | 44.92 | 29.47 | 0.930 | 24.31% |
| Centralized Qwen | 44.63 | 32.02 | 0.775 | 28.26% |

> ✅ **FedLLM在所有指标上全面超越所有基线模型**

### 与基线方法的对比结果
- **相比最佳中央化模型（STGCN）**：
  - RMSE降低 **46.3%**
  - R²提升至 **0.985**（接近完美拟合）
- **相比联邦基线FedASTGCN**：
  - RMSE降低 **45.8%**
  - MAE降低 **43.1%**
  - 且FedLLM提供唯一具备**自然语言解释**的能力
- **长时域稳定性**：
  - FedLLM的R²从15分钟（0.989）到60分钟（0.983）仅下降0.006，远优于其他模型（如Qwen下降0.315），表明其长期预测更稳定。

### 消融实验结果（Ablation Study）
在不同训练/测试规模下验证模型鲁棒性（见Table VII）：

| 设置（Train/Test） | RMSE | MAE | R² |
|------------------|------|-----|----|
| 1000 / 500 | 23.31 | 15.07 | 0.985 |
| 2000 / 1000 | 24.40 | 15.46 | 0.936 |
| 5000 / 3000 | 24.20 | 15.51 | 0.935 |

> 🔍 **即使使用极小样本（1000条训练样本）也能取得最优性能**，说明模型具有高度**数据效率**和**可扩展性**。

### 零样本跨区域迁移性能（GBA District 4）
| 指标 | 1000样本 | 2000样本 | 5000样本 |
|------|---------|---------|---------|
| **R²** | 0.916 | **0.927** | 0.926 |
| RMSE | 43.07 | 41.77 | 41.93 |

> 🌍 在完全未见过的地理区域仍保持高R²（>0.91），证明其强大的**跨区域泛化能力**。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM + FL 是可行且高效的组合**：
   - 首次成功将LLM引入联邦交通预测任务，证明其在隐私保护、可解释性和高性能之间实现了良好平衡。
2. **结构化Prompt是关键**：
   - 将交通上下文编码为自然语言prompt，使LLM能进行情境推理，显著提升预测准确率与可解释性。
3. **联邦训练反而提升性能**：
   - FedLLM性能优于其中心化版本（Domain-Adapted LLM），表明在异构客户端上联合训练有助于捕捉更广泛的交通模式，增强泛化能力。
4. **低通信成本与高实用性**：
   - 仅交换LoRA参数（<1%总参数量），适合带宽受限的实际部署环境。

### 方法的局限性
1. **聚合策略简单**：
   - 当前使用标准FedAvg，未考虑交通特性的动态加权（如拥堵程度、数据新鲜度）。
2. **模型容量有限**：
   - 使用1.5B参数的Qwen模型，更大模型（如LLaMA-3, DeepSeek-V3）可能进一步提升性能。
3. **应用场景受限**：
   - 当前仅验证于高速公路场景，尚未拓展至城市道路、信号灯交叉口等复杂网络。
4. **未探索个性化FL**：
   - 所有客户端共享同一全局模型，缺乏针对特定路段的个性化适配机制。

### 未来工作方向
1. **引入Traffic-Aware Aggregation**：
   - 设计基于交通特征（如流量波动、事件密度）的动态客户端权重机制。
2. **扩大训练规模与模型容量**：
   - 接入更多客户端、增加通信轮次，并尝试更大规模LLM。
3. **拓展至多样化交通场景**：
   - 应用于METR-LA、PEMS-BAY、城市路网等数据集，验证通用性。
4. **发展Hierarchical FL架构**：
   - 构建“城市级-区域级-国家级”多层联邦体系，支持大规模协同建模。
5. **开发交通专用预训练策略**：
   - 在大规模交通语料上进行LLM预训练，减少对下游微调数据的依赖。

---

> ✅ **总结**：FedLLM开创性地融合了**Federated Learning**与**Large Language Models**，构建了一个**隐私保护、可解释、高性能**的交通流预测框架。其实验结果表明，该方法不仅在预测精度上大幅超越现有模型，还能生成自然语言解释，并具备出色的跨区域泛化能力，为未来智能交通系统的可信AI部署提供了新范式。

</details>

---

### 10. [Chronax: A Jax Library for Univariate Statistical Forecasting and Conformal Inference](https://arxiv.org/abs/2604.16719)

**Authors**: Xan Carey, Yash Deshmukh, Aileen Huang, Sunit Jadhav, Omkar Tekawade, Lorraine Yang, Anvesha Tiwary, Gerardo Riano, Amy Greenwald, Denizalp Goktas  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.16719v1  

#### Abstract
Time-series forecasting is central to many scientific and industrial domains, such as energy systems, climate modeling, finance, and retail. While forecasting methods have evolved from classical statistical models to automated, and neural approaches, the surrounding software ecosystem remains anchor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Chronax: A Jax Library for Univariate Statistical Forecasting and Conformal Inference 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的时间序列预测库（如 StatsForecast）虽然在传统 CPU 上通过 Numba 加速提升了性能，但仍存在以下三大瓶颈：
- **并行能力有限**：难以高效处理大规模异构时间序列集合（multi-series forecasting），依赖 Python 循环或手动批处理。
- **执行效率受限**：基于解释器驱动的控制流和 CPU 数值计算，无法充分利用现代硬件（GPU/TPU）的加速潜力。
- **缺乏可微分性**：面向对象的设计阻碍了与现代机器学习生态（尤其是 JAX 生态）的无缝集成，限制了端到端优化的可能性。

### 提出的新方法与新思路
作者提出了 **Chronax** —— 一个原生支持 JAX 的单变量统计预测库，其核心设计围绕 **函数式纯度（functional purity）** 和 **可组合变换（composable transformations）** 构建：
- 将预处理、模型训练、多步预测等流程全部表示为 **pure JAX functions**。
- 利用 `jit`（即时编译）、`vmap`（自动向量化）、`lax.scan`（循环融合）和 `grad`（自动微分）实现端到端优化。
- 支持 **model-agnostic conformal inference**，将置信区间构建也纳入函数式框架中。

### 相比现有方法的优势
| 维度 | Chronax | 传统库（如 StatsForecast） |
|------|--------|--------------------------|
| **执行模式** | 函数式、无状态 | 面向对象、有状态 |
| **硬件支持** | 支持 CPU/GPU/TPU，XLA 编译优化 | 主要依赖 CPU，Numba 加速 |
| **并行化** | 通过 `vmap` 自动向量化，天然支持大规模多序列并行 | 手动批处理或多进程 |
| **可微分性** | 全流程可微，支持梯度传播至输入 | 不支持或部分支持 |
| **集成性** | 与 JAX 科学计算和 ML 生态无缝对接 | 与其他框架集成复杂 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个真实世界时间序列数据集上进行，覆盖不同领域和规模：

| 数据集 | 领域 | 观测数 | 描述 |
|-------|-----|--------|------|
| **Airline Passengers** | Travel | 144 | 1949–1960 年每月航空乘客数量 |
| **Daily Female Births** | Health | 365 | 1959 年加州每日女性出生人数 |
| **Room Temperatures** | Physics | 7056 | 物联网设备记录的每小时室温数据 |

> 注：数据集大小递增，用于测试可扩展性。

### 实验设置
- **对比框架**：Chronax vs. **StatsForecast**（当前主流高性能统计预测库）
- **统一接口**：使用 `fit_predict` 接口封装两个库，确保公平比较。
- **环境隔离**：采用多环境架构，避免 JAX 与 Numba 依赖冲突。
- **硬件同步**：对 JAX 模型调用 `.block_until_ready()` 确保测量实际 GPU 计算时间而非异步调度延迟。
- **数据传输前置**：数组格式转换（NumPy → JAX DeviceArray）在计时循环外完成。

### 评估指标
#### 性能指标（Latency）
- **Cold Start Time (`T_cold`)**：首次运行总耗时，包含 JIT 编译开销。
- **Warm Start Time (`T_warm`)**：稳定状态下平均推理时间（5 次迭代均值）。

#### 预测准确性指标
- **MAPE**（Mean Absolute Percentage Error）
- **MAE**（Mean Absolute Error）
- **RMSE**（Root Mean Squared Error）
- **MASE**（Mean Absolute Scaled Error）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（Chronax / StatsForecast Ratio）

| 指标 | Airline Passengers | Daily Female Births | Room Temperature |
|------|--------------------|---------------------|------------------|
| **Cold Start (Mean)** | 44.54× | 19.32× | 5.82× |
| **Warm Start (Mean)** | 0.76× | 0.62× | 0.16× |
| **MAPE (Mean)** | 1.11× | 1.01× | 0.95× |
| **MAE (Mean)** | 1.10× | 1.01× | 0.94× |
| **RMSE (Mean)** | 1.09× | 1.02× | 0.94× |
| **MASE (Mean)** | 1.10× | 1.01× | 0.94× |

> ✅ **说明**：比率 < 1.0× 表示 Chronax 更优；> 1.0× 表示 StatsForecast 更优。

### 与基线方法的对比结果
- **冷启动性能**：
  - Chronax 明显更慢（高 6–45 倍），因其需进行 XLA 编译。
  - 但随着数据量增大，相对差距缩小（从 44× 降至 5.8×）。
  
- **热启动性能**：
  - Chronax 在 warm inference 上显著更快：
    - 最大提速达 **6.25 倍**（Room Temperature 上平均仅需 16% 时间）。
    - 中位数级别提速更为惊人（median warm ratio 达 0.05–0.07×），表明多数模型在编译后极快。

- **预测精度**：
  - 在小数据集（Airline）上，Chronax 精度略低（MAPE 高约 10%）。
  - 在大数据集（Room Temperature）上，Chronax **全面优于** StatsForecast（所有误差指标下降约 5–6%）。
  - 表明其在大规模场景下不仅更快，而且更准。

### 消融实验分析（隐含于设计）
尽管未明确列出消融实验，但从设计原则可推断：
- **`vmap` 并行化**：实现了 conformal validation 的并行计算，相比串行 K 折交叉验证大幅提升速度。
- **JIT 编译代价换取长期收益**：前期编译成本高，但后续重复调用效率极高，适合频繁 retraining 场景。
- **函数式抽象提升可组合性**：支持 conformal intervals 与任意模型即插即用，无需重写逻辑。

---

## 4. 关键结论和发现

### 主要发现
1. **JAX 范式适用于统计预测系统重构**：
   - 尽管冷启动较慢，但 warm performance 的巨大优势证明了编译驱动执行的有效性。
2. **函数式设计带来多重好处**：
   - 天然支持大规模并行（`vmap`）、端到端可微分、透明数据流。
3. **性能与精度随数据规模正相关**：
   - 数据越大，Chronax 相对于传统库的优势越明显（both speed and accuracy）。
4. **conformal inference 可被优雅集成**：
   - 利用 `vmap` 实现 walk-forward validation 的并行化，显著加速不确定性量化过程。

### 方法的局限性
- **冷启动延迟高**：不适合一次性、低频调用场景。
- **当前仅支持 univariate 模型**：尚未扩展至多元时间序列或多模态协变量。
- **模型种类仍以经典统计为主**：缺少深度学习类模型（如 DeepAR、Informer）。
- **依赖 JAX 生态**：对不熟悉 JAX 的用户有一定学习门槛。

### 未来工作方向
1. **扩展模型族**：
   - 引入 multivariate 和 deep learning-based forecasters，保持相同 JAX 接口。
2. **构建混合管道**：
   - 支持统计模型与神经网络的联合建模（hybrid pipelines）。
3. **增强 conformal inference 功能**：
   - 支持 hierarchical forecasting 下的一致性置信区间。
   - 提升对分布偏移（distribution shift）的鲁棒性。
4. **深化与 JAX 生态整合**：
   - 支持 differentiable hyperparameter tuning。
   - 与科学模拟器（scientific simulators）联合优化。
   - 实现组件级联合优化（如预处理滤波器、潜变量表示、不确定性包装器）。

---

> 🔗 **代码开源地址**：[https://github.com/Smlcrm/Chronax](https://github.com/Smlcrm/Chronax)

</details>

---

### 11. [Cross-Family Speculative Decoding for Polish Language Models on Apple~Silicon: An Empirical Evaluation of Bielik~11B with UAG-Extended MLX-LM](https://arxiv.org/abs/2604.16368)

**Authors**: Krzysztof Fonal  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.16368v1  

#### Abstract
Speculative decoding accelerates LLM inference by using a small draft model to propose k candidate tokens for a target model to verify. While effective for same-tokenizer pairs on high-bandwidth GPUs, its applicability to cross-family pairs with mismatched tokenizers and consumer-grade unified memor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究解决了在**Apple Silicon**统一内存架构上进行**跨家族（cross-family）** 大语言模型（LLM）**推测解码（speculative decoding）** 的三大挑战：
1. **跨分词器（cross-tokenizer）兼容性问题**：主流推测解码要求 draft 和 target 模型共享 tokenizer，但波兰语模型 Bielik 家族内部因采用不同基础架构（Mistral vs. Qwen2.5）而使用不兼容的 tokenizer（Mistral tokenizer vs. APT4），导致无法直接应用标准推测解码。
2. **缺乏 Apple Silicon 支持**：尽管 Universal Assisted Generation (UAG) 技术支持跨 tokenizer 推测解码，但其尚未被集成到 Apple 的 MLX-LM 框架中，限制了 Apple 用户利用该技术的能力。
3. **硬件假设失效风险**：推测解码的理论加速依赖于“验证成本可摊销”的假设，但在 Apple Silicon 的低带宽统一内存架构下，这一假设是否成立尚无实证。

### 提出的新方法与思路
作者提出并实现了 **UAG-Extended MLX-LM**，将 Universal Assisted Generation (UAG) 集成到 MLX-LM 框架中，支持跨 tokenizer 的推测解码。具体创新包括：
- **实现两种 token 转换策略**：
  - **Naive Token Translation**：简单地将 draft tokens 解码为字符串，再用 target tokenizer 重新编码。
  - **Context-Aware Token Translation**：在重编码前，拼接 `p` 个已接受的上下文 tokens 作为前缀，以提供足够的边界上下文，显著提升对齐准确性。
- **开源实现**：代码已公开，填补了 MLX 生态在跨 tokenizer 推测解码方面的空白。

### 相比现有方法的优势
- **扩展了 draft 模型选择范围**：不再局限于同一模型家族，允许使用如 Qwen2.5、Llama 等通用小模型作为波兰语大模型的 draft 模型。
- **首次支持 Apple Silicon 上的跨 tokenizer 推测解码**：为本地化、隐私优先的部署场景提供了新的优化工具。
- **提出了硬件感知的速度公式**：推导出适用于 Apple Silicon 统一内存架构的参数化速度提升公式，更准确地预测实际性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在三个波兰语数据集上进行，以覆盖不同的生成场景：
1. **Polish Wikipedia**：来自波兰维基百科的条目，包含大量结构化、重复性的文本（如列表、信息框）。
2. **pl_alpaca**：Alpaca 指令跟随数据集的波兰语翻译，代表多样化的指令跟随任务。
3. **Synthetic short questions**：人工生成的简短波兰语问题，用于测试对话式、短输出场景。

### 实验设置
- **目标模型（Target Model）**：`Bielik 11B-Instruct` (基于 Mistral 架构，8-bit 量化)。
- **draft 模型（Draft Models）**：
  - `Bielik 1.5B` (Qwen2.5 基础，使用 APT4 波兰语 tokenizer)
  - `Qwen2.5-1.5B`
  - `Llama 3.2-1B`
- **硬件平台**：Apple M2 Pro (32GB 统一内存，~200 GB/s 带宽)。
- **draft 长度（k）**：主要评估 `k ∈ {2, 4}`，并在 `pl_alpaca` 数据集上进行了 `k=6` 的确认性实验。
- **token 转换策略**：对每个 draft 模型评估三种条件：无转换（no translation）、朴素转换（naive translation）、上下文感知转换（context-aware, p=5）。

### 评估指标
- **Token Acceptance Rate (α)**：目标模型接受的 draft tokens 的比例。
- **Tokens Per Second (TPS)**：端到端生成吞吐量。
- **Speedup**：推测解码 TPS 相对于自回归（autoregressive）基线的加速比。

### 基线方法对比
- **Baseline**：仅使用 `Bielik 11B` 进行自回归解码，无任何 draft 模型。
- **对比条件**：将上述三种 token 转换策略与三个 draft 模型的所有组合，与 Baseline 进行 TPS 和 Speedup 对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **Baseline TPS**：约 **14.6–15.1 TPS**。
- **最高 Acceptance Rate**：在 `k=2` 时，`Qwen2.5-1.5B` 在 Wikipedia 数据集上达到 **44.6%** 的接受率（上下文感知转换）。
- **最高 Speedup**：在 `k=2` 且 Acceptance Rate 较高时（60-70% 区间），`Llama-3.2-1B` 和 `Qwen2.5-1.5B` 可达到 **1.43–1.59×** 的加速。

### 与基线方法的对比结果
1. **Context-Aware Translation 显著优于其他方法**：
   - 在所有 9 种 draft 模型-数据集组合中，上下文感知转换的接受率均最高。
   - 朴素转换表现最差，甚至低于“无转换”条件，因为它引入了有害的边界错位。
2. **通用模型优于专用模型**：
   - 尽管 `Bielik 1.5B` 是专为波兰语设计的模型，但其接受率（31.1%）反而低于通用的 `Qwen2.5-1.5B`（44.6%）和 `Llama 3.2-1B`（42.0%）。这归因于其 APT4 tokenizer 与目标模型的 Mistral tokenizer 存在严重的分割策略不匹配。
3. **内容依赖性强**：
   - 在结构化、可预测的 **Wikipedia** 数据上，推测解码效果最好，部分配置可达 **1.06×** 加速。
   - 在多样化、不可预测的 **pl_alpaca** 和 **synthetic** 数据上，难以超越自回归基线。
4. **k 值增加导致性能下降**：
   - 尽管 `k=4` 时接受率更高（如 Qwen 达到 53.8%），但由于 draft 模型需要执行更多次前向传播，开销剧增，导致所有配置的平均速度都**远低于基线**（Speedup ≈ 0.59–0.70×）。

### 消融实验结果
- **Token 转换策略消融**：明确证明了 `Context-Aware > No Translation > Naive Translation` 的性能排序。
- **draft 长度（k）消融**：`k=2` 是唯一可能获得正向加速的配置；`k=4` 及以上均因开销过大而失败。
- **Break-Even 分析**：
  - `k=2` 时，break-even 接受率约为 **38–53%**，在实践中可以达到。
  - `k=4` 时，break-even 接受率飙升至 **77–92%+**，远超实际能达到的水平，解释了为何 `k=4` 会失败。

---

## 4. 关键结论和发现

### 主要发现
1. **上下文感知转换是必需的**：在跨 tokenizer 场景下，`context-aware token translation` 是实现有效推测解码的前提，而非简单的优化。
2. **通用 draft 模型可能优于专用模型**：对于 Bielik 11B，通用的 `Qwen2.5-1.5B` 和 `Llama 3.2-1B` 作为 draft 模型的表现优于同家族的 `Bielik 1.5B`，凸显了 tokenizer 兼容性的重要性超过领域特异性。
3. **推测解码的效果高度依赖于内容类型**：在**结构化、重复性高的文本**（如维基百科、代码）上效果显著，而在**开放式的指令跟随任务**上则难以超越基线。
4. **Apple Silicon 上的验证成本无法完全摊销**：由于统一内存带宽较低（200 GB/s），draft 模型的 `k` 次前向传播开销巨大，使得 `k>2` 的配置得不偿失。这与高带宽 GPU（如 A100）上的情况截然不同。
5. **提出了硬件感知的速度公式**：推导出 `Speedup(α,k) = (1+αk) / ((k+1)r + 1 + βk²)`，其中 `r` 是模型大小比，`β` 是表征额外开销的经验常数，能准确描述在 M2 Pro 上的性能表现。

### 方法的局限性
- **k 值受限**：在当前硬件下，`k=2` 是上限，更大的 draft 长度会因开销过大而适得其反。
- **Token 转换的工程复杂性**：上下文感知转换存在“零新 token”等边界吸收问题，缓存回滚（cache rewinding）等解决方案又带来额外计算开销。
- **Token-Level Intersection (TLI) 不适用**：对于 Bielik-Mistral 这类 tokenizer 差异大的组合，TLI 的词汇交集过小，会严重损害 draft 模型的流畅性。

### 未来工作方向
- **改进边界处理机制**：开发更智能的边界检测算法，动态调整前缀窗口，避免 token 合并问题。
- **实现 Token-Level Intersection (TLI) for MLX-LM**：探索其在词汇交集较大的模型对上的可行性。
- **局部缓存回滚**：只回滚受影响的 token 边界，而非整个上下文窗口，以降低开销。
- **在更高带宽硬件上验证**：在 M3 Ultra 或 M4 Max 等设备上测试，验证随着带宽提升，推测解码的收益是否会改善。
- **评估新架构模型**：如即将发布的基于 Nemotron 架构的 Bielik 模型，测试其对跨家族推测解码的影响。

</details>

---

### 12. [AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation](https://arxiv.org/abs/2604.16625)

**Authors**: Weihua Du, Jingming Zhuo, Yixin Dong, Andre Wang He, Weiwei Sun, Zeyu Zheng, Manupa Karunaratne, Ivan Fox, Tim Dettmers, Tianqi Chen, Yiming Yang, Sean Welleck  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.16625v1  

#### Abstract
Recent large language model (LLM) agents have shown promise in using execution feedback for test-time adaptation. However, robust self-improvement remains far from solved: most approaches still treat each problem instance independently, without accumulating reusable knowledge. This limitation is par...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AdaExplore: Failure-Driven Adaptation and Diversity-Preserving Search for Efficient Kernel Generation

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Models (LLMs)** 的代码生成在**低资源、领域特定语言**（如 Triton）中面临两大挑战：
- **可行性瓶颈（Feasibility Bottleneck）**：由于训练数据稀少，LLM 生成的内核代码常因语法错误、内存访问违规等问题导致编译或运行失败。
- **局部最优陷阱（Locality Bottleneck）**：传统迭代优化方法（如 iterative refinement）容易陷入局部最优，难以进行结构性重构以实现显著性能提升。

这些问题使得 LLM 在 GPU kernel runtime optimization 上表现不佳，尤其是在复杂任务中。

---

### 提出的新方法：AdaExplore
作者提出 **AdaExplore**，一个两阶段的 LLM 代理框架，结合 **failure-driven adaptation** 和 **diversity-preserving search**，无需额外微调即可实现高效内核生成。

#### 两个核心机制：

1. **Adapt: 失败驱动的适应（Failure-Driven Adaptation）**
   - 在合成任务上运行 LLM 代理，收集执行失败反馈。
   - 将重复出现的失败模式提炼为可复用的 **cross-task skill memory**（跨任务技能记忆），例如“不能在 Triton kernel 中调用 `tl.float32` 作为函数”。
   - 这些规则作为系统提示注入后续生成过程，显著提高生成正确性。

2. **Explore: 多样性保持搜索（Diversity-Preserving Search）**
   - 构建一棵候选内核的 **search tree**，而非单一链式优化路径。
   - 支持两种操作：
     - **Small Step**：局部修补（local refinement），用于精细调优。
     - **Large Step**：结构再生（structural regeneration），跳出局部最优。
   - 使用 **UCT-style node selection** 和 **representative kernel pool** 来平衡探索与利用。

---

### 相比现有方法的优势
| 方法 | 局限性 | AdaExplore 的改进 |
|------|--------|------------------|
| 单次生成（Single-pass） | 正确率低，性能差 | 通过 adaptation 显著提升正确率 |
| Parallel Sampling | 缺乏长期记忆，多样性有限 | 引入 skill memory 提高有效性 |
| Iterative Refinement | 易陷局部最优，缺乏结构性变化 | 通过 tree search 和 large step 实现全局探索 |
| OpenEvolve 等进化方法 | 依赖 population，上下文受限 | 更高效的树结构 + 双重记忆机制 |

> ✅ **核心优势**：AdaExplore 实现了 **test-time scaling**，即随着计算预算增加，性能持续提升，且无需模型微调。

---

## 2. 核心实验方法和设置

### 数据集
- **KernelBench** (Ouyang et al., 2025)：主要测试平台，按难度分为三级：
  - **Level-1**：单算子（用于训练任务合成）
  - **Level-2**：简单融合内核（如 fused add + RMSNorm）
  - **Level-3**：模型级工作负载（如 ResNet, LSTM 组件）
- **FlashInfer-Bench** (Xing et al., 2026)：真实 LLM 推理流水线中的内核任务，含专家编写的 CUDA 基线（如 FlashInfer 实现）。

---

### 实验设置
- **基础模型**：默认使用 **GPT-5-mini**，也测试了 GPT-5、Claude-4.6-Opus、Qwen3-Coder-Next。
- **测试时预算（Test-time budget）**：最多 200 步（step），每步包含生成、执行、反馈循环。
- **硬件环境**：NVIDIA A6000/B200/GPU，固定频率下测量运行时间。

---

### 评估指标
| 指标 | 定义 |
|------|------|
| **Acc.** | 在预算内至少生成一个功能正确的内核的比例 |
| **Speedup** | 相对于 PyTorch eager 实现的最佳加速比（上限 10×） |
| **Fast@1.2 / Fast@2** | 生成速度超过基线 1.2× 或 2× 的比例 |

> ⚠️ 注意：Speedup 被截断至 10×，避免极端值主导平均值。

---

### 基线方法对比
| 类型 | 方法 |
|------|------|
| 单次生成 | GPT-5-mini, GPT-5, Claude-4.6-Opus, AutoTriton |
| 多轮优化 | Parallel Sampling (PS), Iterative Refinement (IR) |
| 进化方法 | OpenEvolve, DR. Kernel (RL-based) |
| 消融变体 | IR w. SM（带 skill memory）、PS w. SM 等 |

所有多轮方法均以 GPT-5-mini 为基础模型。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（KernelBench）

| 方法 | Level-2 Speedup | Level-3 Speedup | Acc. (L2/L3) |
|------|------------------|------------------|-------------|
| GPT-5-mini (single-pass) | 0.34× | 0.21× | 22%/22% |
| IR w. SM (50 steps) | 2.59× | 1.31× | 100%/100% |
| OpenEvolve w. SM | 1.91× | 1.47× | 100%/100% |
| **AdaExplore (50 steps)** | **2.65×** | **1.55×** | **100%/100%** |
| **AdaExplore (100 steps)** | **3.12×** | **1.72×** | **100%/100%** |
| **AdaExplore (200 steps)** | **3.41×** | **1.78×** | **100%/100%** |

> 🔺 **结论**：AdaExplore 在 **Level-2 达到 3.12× 加速**，**Level-3 达到 1.72× 加速**，远超其他方法，并随步数持续提升。

---

### 与其他方法对比
- 在 Level-2 上，AdaExplore 比最强非自身基线（IR w. SM）高出 **0.53×**。
- 在 Level-3 上，其优势更明显，因 IR 方法受限于局部编辑，而 AdaExplore 可通过 **large step** 实现结构跃迁。
- **test-time scaling 曲线显示**：AdaExplore 性能无饱和迹象，而其他方法在 ~50 步后趋于平缓（见 Figure 3）。

---

### 消融实验结果（Ablation Study）

| 变体 | Speedup (L2) | Fast@1.2 | 说明 |
|------|--------------|----------|------|
| Full AdaExplore | **2.65×** | **71%** | 完整方法 |
| w/o MCTS (chain-only) | 2.48× | 64% | 树结构对多样性至关重要 |
| w/o Large Step | 2.62× | 71% | small step 已足够部分优化 |
| w/o Small Step | 2.35× | 60% | 缺少局部调优影响稳定性 |
| w/o Skill Memory | 2.32× | 56% | 正确性下降，搜索效率降低 |
| w/o Representative Kernel Pool | 2.30× | 63% | 长期进度记忆重要 |

> ✅ **关键发现**：**skill memory** 对正确性提升最大；**tree search** 和 **large step** 共同支持跳出局部最优。

---

## 4. 关键结论和发现

### 主要发现
1. **失败是可学习的**：低层 kernel 生成的失败高度集中在少数几类语法和语义约束上，可通过 **frequency-based filtering** 提炼出稳定、可迁移的规则。
2. **test-time adaptation 有效**：无需微调，仅靠执行反馈积累的知识即可大幅提升 LLM 在低资源语言上的表现。
3. **structured search 至关重要**：相比链式优化，**tree-based search** 能保留多个有潜力的方向，避免过早收敛。
4. **组合机制效果最佳**：**Adapt + Explore** 协同作用——前者确保进入可行域，后者实现高效探索。

---

### 方法的局限性
1. **对 heavily optimized kernels 效果有限**：
   - 如 GEMM 内核，cuBLAS 已经高度优化，AdaExplore 仅达到其 42% 性能。
2. **依赖高质量执行反馈**：
   - 需要稳定的编译、运行和 profiling 环境，远程评估服务增加了复杂性。
3. **无法自动引入新硬件特性**：
   - 如 Blackwell 架构的 QMMA 指令、FP8 量化等，仍需人工引导或外部知识注入。

---

### 未来工作方向
1. **集成专家知识或参考实现**：将 FlashInfer、cuBLAS 等高性能实现纳入提示或检索增强。
2. **支持更多 DSL 和硬件架构**：扩展至 CUDA、HIP、Metal 等语言，适配 AMD、Apple Silicon。
3. **自动化 hyperparameter tuning**：动态调整 `plarge`、`Cexplore` 等参数以适应不同任务难度。
4. **构建开放协作的 skill memory 社区**：共享跨项目、跨团队的 failure patterns，形成“集体智能”。

---

> 📦 **开源信息**：  
> 项目已公开：[https://github.com/StigLidu/AdaExplore](https://github.com/StigLidu/AdaExplore)

</details>

---

### 13. [OPSDL: On-Policy Self-Distillation for Long-Context Language Models](https://arxiv.org/abs/2604.17535)

**Authors**: Xinsen Zhang, Zhenkai Ding, Tianjun Pan, Run Yang, Chun Kang, Xue Xiong, Jingnan Gu  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.17535v1  

#### Abstract
Extending the effective context length of large language models (LLMs) remains a central challenge for real-world applications. While recent post-training methods have made progress in long-context scaling, they either rely on high-quality supervision data or sparse sequence-level rewards, leading t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：OPSDL: On-Policy Self-Distillation for Long-Context Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 Large Language Models (LLMs) 在扩展 **effective context length** 方面面临显著挑战。尽管模型可以接受更长输入（如通过改进 positional encoding），但在实际推理中，其有效利用长上下文的能力远低于理论最大窗口。这一“**最大上下文长度 vs. 有效上下文能力**”之间的差距限制了 LLM 在长文档理解、代码库级分析和多跳推理等任务中的表现。

现有方法如 Supervised Fine-Tuning (SFT) 或基于 DPO 的偏好优化（如 LongPO）存在以下问题：
- 依赖高质量人工标注数据或外部 reward model；
- 使用稀疏的 sequence-level 奖励信号，导致训练不稳定且样本效率低；
- 引入额外组件（如 verifier 模型），增加系统复杂性。

---

### 🚀 提出的新方法：OPSDL
作者提出 **OPSDL (On-Policy Self-Distillation for Long-context)**，一种无需外部监督的自蒸馏框架，用于提升 LLM 的长上下文建模能力。

#### 核心思想：
- 利用模型自身在 **short-context** 下更强、更准确的行为作为 **self-teacher**；
- 对比同一模型在 **full long-context (student)** 和 **extracted short-context (teacher)** 下生成的 token 分布；
- 通过 **point-wise reverse KL divergence** 构造 token-level 的优势函数（advantage），指导 long-context 生成过程。

#### 具体流程：
1. 给定一个长文档 $ C_L $，从中提取保留核心信息的短片段 $ C_s $；
2. 基于 $ C_s $ 自动生成 query $ Q $，确保该问题在 $ C_L $ 和 $ C_s $ 中均可回答；
3. 模型以 $ C_L, Q $ 为条件生成响应（on-policy rollout）；
4. 同一模型以 $ C_s, Q $ 为条件提供每个 token 的 teacher 分布；
5. 计算每 token 的 **reverse KL advantage**：
   $$
   A_t = \log \frac{P_{\text{teacher}}(y_t | C_s, Q, y_{<t})}{P_{\text{student}}(y_t | C_L, Q, y_{<t})}
   $$
6. 使用此优势进行 policy gradient 更新，使 long-context 输出向 short-context 行为对齐。

---

### 🔍 相比现有方法的优势
| 特性 | OPSDL | SFT / DPO 类方法（如 LongPO） |
|------|-------|-------------------------------|
| 是否需要人工标注 | ❌ 否 | ✅ 是或依赖合成偏好对 |
| 是否依赖 reward model | ❌ 否 | ✅ 是（部分方法） |
| 训练信号粒度 | ✅ **token-level**, dense | ❌ sequence-level, sparse |
| 样本效率 | ✅ 高（on-policy + 密集信号） | ❌ 较低 |
| 系统复杂性 | ✅ 极简（无辅助模块） | ❌ 复杂（需 verifier/reward model） |
| 是否破坏 short-context 能力 | ❌ 否（实验证明保持） | ⚠️ 可能下降（如 Long-SFT） |

> ✅ **核心创新**：首次将 **on-policy self-distillation** 应用于 long-context 学习，并利用模型内在的 short-context 优势作为动态 self-teacher，实现稳定高效的自我进化。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练数据构造**：从原始长文本语料中自动构建三元组 $(C_L, C_s, Q)$，无需人工标注。
  - $ C_L $：完整长文档
  - $ C_s \subset C_L $：关键信息子段（长度适配标准 context window）
  - $ Q $：通过 Self-Instruct 机制基于 $ C_s $ 自动生成的问题

- **评估基准**：
  - **RULER** (Hsieh et al., 2024)：合成型 long-context 测试套件，量化模型在不同 context length（4K–128K）下的检索与跟踪能力。
  - **LongBench V2** (Bai et al., 2025)：真实场景下的多任务长文档理解 benchmark，涵盖多种 domain 和难度级别（Easy/Medium/Hard）。

---

### ⚙️ 实验设置
- **Backbone Models**：
  - Qwen2.5-7B-Instruct
  - Qwen2.5-14B-Instruct
  - Qwen2.5-32B-Instruct
- **Baseline 方法对比**：
  - **Long-SFT**：直接在 long-context 数据上做 SFT
  - **LongPO**：基于 short-to-long preference pair 的 DPO 优化
  - **Qwen2.5-*-Instruct-1M**：官方发布的百万 token 上下文优化版本（multi-stage SFT + extrapolation 技术）
- **评估指标**：
  - RULER：各 context length 下的准确率及平均得分
  - LongBench V2：总体平均分（Overall）、按难度划分得分
  - 通用能力保留测试：MMLU、ARC-C、Hellaswag、Winogrande、MT-Bench

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1）

| Model | Method | RULER Avg. ↑ | LongBench V2 Overall ↑ | Total Avg. ↑ |
|-------|--------|---------------|--------------------------|----------------|
| 7B | Base | 77.16 | 26.2 | 51.68 |
|     | +Long-SFT | 81.88 | 26.1 | 53.99 |
|     | +LongPO | 83.34 | 27.5 | 55.42 |
|     | **+OPSDL (Ours)** | **86.32** | **32.6** | **56.61** |
|     | Qwen2.5-7B-Instruct-1M | 90.26 | 30.6 | 60.43 |
| 14B | Base | 83.61 | 31.6 | 57.61 |
|     | +Long-SFT | 88.77 | 32.5 | 60.64 |
|     | +OPSDL (Ours) | **90.90** | **33.7** | **62.30** |
|     | Qwen2.5-14B-Instruct-1M | 94.18 | 36.4 | 65.29 |
| 32B | Base | 86.41 | — | 59.65 |
|     | +Long-SFT | 91.90 | 32.9 | 63.05 |
|     | +OPSDL (Ours) | **93.36** | **36.5** | **64.93** |

> ✅ **关键观察**：
- 在所有规模下，**OPSDL 显著优于 Long-SFT 和 LongPO**；
- 性能增益随 context length 增加而放大，在 128K 上提升达 **+48.7 pts (7B)**；
- **LongPO 在 14B/32B 上无法收敛**，而 OPSDL 训练稳定；
- OPSDL 接近甚至逼近专门训练的 **-1M 版本**，差距缩小至 ~3–4 pts。

---

### 🔬 消融实验与关键发现（隐含在主实验中）
虽然未设独立消融表，但从设计可推断以下有效性来源：
- **on-policy 生成**：避免 off-policy 数据带来的分布偏移；
- **token-level 优势信号**：仅当 long-context 行为偏离 short-anchor 时才触发梯度更新，提高学习针对性；
- **self-teacher 动态演化**：teacher 与 student 共同更新，保证监督信号始终与当前策略一致；
- **no external supervision**：完全摆脱 reward model 或人工标注依赖。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **模型自身的 short-context 能力是强大的 self-teacher**：
   - 即使没有外部知识注入，仅靠提取关键上下文即可构建高质量监督信号；
   - 这种“降噪+对齐”机制有效缓解 long-context 中的信息干扰与 hallucination。

2. **dense token-level 信号优于 sparse sequence-level 偏好**：
   - Reverse KL 提供连续、细粒度的优化方向；
   - 相比 DPO 的 binary preference，更适合 long-horizon 决策问题。

3. **训练高效且可扩展**：
   - 不依赖 reward modeling 或复杂的 pipeline；
   - 在 7B–32B 模型上均稳定训练并持续提点。

4. **不牺牲 short-context 性能**：
   - 如 Table 2 所示，OPSDL 在 MMLU、ARC-C 等 short-context benchmark 上仅有约 **1.3% 平均下降**；
   - 而 Long-SFT 下降达 3–4%，表明传统方法易造成能力退化。

---

### ⚠️ 局限性
- 当前方法依赖于能够从 $ C_L $ 中正确提取 $ C_s $，若 extraction 不准会影响 teacher 质量；
- 尚未探索非 contiguous 的 evidence 提取方式（如 multi-hop 支持句抽取）；
- 所有实验基于 Qwen 系列模型，虽具泛化性但仍需更多架构验证。

---

### 🔮 未来工作方向
- 结合 retrieval-augmented 方法自动识别 relevant chunks；
- 将 OPSDL 扩展到多模态 long-context 场景（如长视频理解）；
- 探索与其他 post-training 方法（如 RLHF）结合的可能性；
- 研究如何进一步压缩 $ C_s $ 以增强 teacher 的抗噪能力。

---

## ✅ 总结
**OPSDL** 是一种简洁、高效、可扩展的 long-context post-training 新范式。它通过 **on-policy self-distillation** 机制，让模型用自己的 short-context “最佳实践” 来指导 long-context 行为，实现了：
- 更高的样本效率
- 更稳定的训练过程
- 更强的长上下文性能
- 更好的通用能力保留

> 💡 **一句话总结**：  
> *“最好的老师，是你自己在清晰环境下的样子。”*  
> OPSDL 正是通过这种自我参照的方式，解锁了 LLM 在超长上下文中的稳健推理潜力。

</details>

---

### 14. [GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling](https://arxiv.org/abs/2604.18556)

**Authors**: Alireza Dadgarnia, Soroush Tabesh, Mahdi Nikdan, Michael Helcig, Eldar Kurtic, Dan Alistarh  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.18556v1  

#### Abstract
Weight quantization has become a standard tool for efficient LLM deployment, especially for local inference, where models are now routinely served at 2-3 bits per parameter. The state of the art is currently split into two sets of methods: simple scalar quantization techniques, such as GPTQ or AWQ, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前大语言模型（LLMs）的权重量化（weight quantization）领域存在一个显著的“代际差距”：
- **第一代方法**（如 GPTQ、AWQ）采用简单的 **scalar quantization**，实现简单且兼容现有推理内核（如 llama.cpp），但在低于 3–4 bits per parameter（bpp）时精度急剧下降。
- **第二代方法**（如 QTIP、GPTVQ、AQLM）采用更复杂的 **vector quantization (VQ)** 或 **trellis quantization**，在 2–3 bpp 下精度更高，但实现复杂、难以扩展，且需要专用解码内核，限制了实际部署。

本文提出：**这个精度差距是否本质上是表示能力的差距，还是优化不足导致的？**

### **提出的新方法：GSQ (Gumbel-Softmax Quantization)**
- **核心思想**：将传统的离散量化问题转化为一个**可微分的离散分配问题**，通过 **Gumbel-Softmax Relaxation** 对每个权重坐标的网格分配进行联合优化。
- **关键技术**：
  - **Gumbel-Softmax Sampling**：为每个权重坐标引入可学习的 logits，通过 Gumbel-Softmax 生成软量化权重，使重建损失对 scale 和离散分配都可微。
  - **温度退火（Temperature Annealing）**：训练过程中逐渐降低温度，使软分配收敛到硬的离散网格点。
  - **局部偏移参数化（Local-Shift Parameterization）**：对于高比特（如 3-bit），不直接对整个 $2^b$ 大小的网格进行松弛，而是只学习相对于初始 GPTQ 解的**小范围偏移**（如 ±2），将每坐标参数量从 $2^b$ 降至常数（如 5），大幅降低内存开销。

### **相比现有方法的优势**
| 维度 | GSQ | 传统 Scalar (GPTQ/AWQ) | Vector/Trellis (QTIP/GPTVQ) |
|------|-----|------------------------|----------------------------|
| **精度** | ✅ 接近 VQ 方法，在 2–3 bpp 下大幅超越传统 scalar | ❌ 在低比特下精度差 | ✅ 高 |
| **实现复杂度** | ✅ 与 scalar 方法相当，无需 codebook | ✅ 极简 | ❌ 复杂，需专用 kernel |
| **推理兼容性** | ✅ 完全兼容现有 scalar 推理栈（如 llama.cpp, vLLM） | ✅ 兼容 | ❌ 不兼容，需定制解码 |
| **可扩展性** | ✅ 可扩展至万亿参数 MoE 模型 | ✅ 可扩展 | ❌ 在 MoE 上难应用 |
| **优化方式** | ✅ 联合优化 scale 和离散分配 | ❌ 分离优化或贪心 | ✅ 联合优化 |

> **核心结论**：低比特下的精度差距**主要是优化差距而非表示能力差距**。通过更好的离散优化，标准 scalar 格式也能接近 VQ 的精度。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **校准数据（Calibration Data）**：
  - **Llama 模型**：FineWeb-Edu（4096 条，长度 4096）
  - **Kimi-K2.5 模型**：OpenThoughts（4096 条，长度 4096）
- **评估数据集**：
  - **通用零样本基准**：ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande
  - **数学与编码**：AIME25, GPQA:Diamond, MATH500, LiveCodeBench-v6
  - **长上下文**：OpenAI-MRCR（支持至 256k 上下文）
  - **数学推理**：GSM8K（flexible）

### **实验设置**
- **模型**：
  - **Dense Models**：Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct
  - **MoE Model**：Kimi-K2.5（trillion-scale）
- **量化配置**：
  - **比特宽度**：1.58-bit（ternary）、2-bit、3-bit
  - **分组大小（Group Size）**：128
  - **量化格式**：symmetric scalar quantization，无 zero-point
  - **非均匀量化**：混合 2/3-bit，实现 2.37 / 2.62 bpp
- **训练细节**：
  - **优化目标**：模块输出的均方误差（MSE）
  - **温度调度**：从 2 线性退火至 0.05
  - **缩放因子 K**：从 100 到 500
  - **优化器**：Lion（避免 AdamW 在梯度消失时停滞）
  - **梯度累积**：用于降低 Gumbel-Softmax 采样方差

### **基线方法对比**
- **Scalar Baselines**：
  - GPTQ
  - QuIP
  - EfficientQAT（允许 asymmetric quantization，更具优势）
- **Vector Quantization Baselines**：
  - QTIP（state-of-the-art VQ）
  - PV-Tuning（基于 AQLM）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Llama-3.1-70B-Instruct）**

#### **2-bit 结果**
| 方法 | Avg. Accuracy | ARC-C | HellaSwag | WinoGrande |
|------|---------------|-------|-----------|------------|
| FP16 | 78.99 | 63.48 | 84.58 | 79.01 |
| **GSQ (ours)** | **77.25** | **61.69** | **82.95** | **77.51** |
| EfficientQAT | 77.40 | 61.86 | 82.79 | 76.01 |
| QTIP | 77.25 | 61.69 | 82.95 | 77.51 |
| GPTQ | 57.38 | 38.23 | 60.11 | 57.14 |

- **GSQ 超越所有 scalar 方法**，平均精度提升 **4.14 pts**（vs EfficientQAT）。
- **仅落后于 QTIP 1.68 pts**，而 QTIP 是更复杂的 VQ 方法。
- **GSQ 使用 symmetric quantization，而基线允许 asymmetric**，说明其优势来自**优化而非表示灵活性**。

#### **3-bit 结果**
| 方法 | Avg. Accuracy |
|------|---------------|
| FP16 | 78.99 |
| **GSQ (ours)** | **77.99** |
| EfficientQAT | 77.40 |
| QTIP | 78.17 |

- GSQ 在 3-bit 下已**基本追平 QTIP**，在 70B 模型上差距仅 **0.18 pts**。

#### **非均匀量化（2.37 / 2.62 bpp）**
- **2.62 bpp**：平均精度 **77.50**，接近 3-bit 表现，压缩率更高。
- **2.37 bpp**：平均精度 **77.40**，优于 2-bit GPTQ（57.38），实现**精度-压缩率的良好权衡**。

#### **Ternary (1.58-bit) 结果**
- GSQ 在 **1.58-bit** 下表现优于 **2-bit GPTQ/QuIP**，甚至接近 **2-bit EfficientQAT**。
- 证明其在极低比特下仍有效。

### **消融实验结果**
- **端到端 scale 微调**（A.2）：
  - 在 block-wise 优化后，仅微调 scale 参数，可在 2-bit 模型上带来 **~1.12 pts** 的额外增益。
  - 说明全局 scale 调整能进一步提升性能。
- **块级 vs 端到端优化**（A.3）：
  - 即使在 2:4 structured sparsity 任务中，GSQ 的 block-wise 优化也优于 MaskLLM 的 end-to-end 训练。
  - 证明其优化框架的有效性。

### **推理速度**
| 方法 | Avg. bit/param | Speedup (vs BF16) |
|------|----------------|-------------------|
| Uniform 3-bit | 3.00 | 4.80× |
| Non-uniform 2.62 | 2.62 | 4.99× |
| Non-uniform 2.37 | 2.37 | 5.46× |
| Uniform 2-bit | 2.00 | 6.20× |

- GSQ 支持高效 scalar GEMM，实现高达 **6.2×** 的吞吐提升。

### **Kimi-K2.5 (Trillion-Scale MoE) 结果**
- **首次成功对万亿参数 MoE 模型进行 2-bit 量化**。
- 在数学和代码任务上表现优异：
  - **MATH500**: 96.68 → **97.32**
  - **LiveCodeBench v6**: 61.37 → **69.37**
- **长上下文性能**：在 ≤32k 上下文下保持良好，>64k 后略有下降。
- **局限性**：在 GPQA Diamond（科学问答）上下降明显（89.29 → 76.57），归因于**校准数据偏向数学/代码**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **精度差距本质是优化差距**：通过 Gumbel-Softmax 实现的联合优化，standard scalar quantization 可以在 2–3 bpp 下逼近甚至媲美复杂的 VQ 方法。
2. **GSQ 实现了精度、效率与兼容性的统一**：
   - 精度接近 VQ
   - 实现简单，兼容现有 scalar 推理栈
   - 可扩展至万亿参数 MoE 模型
3. **非均匀量化自然支持**：GSQ 可轻松实现混合比特分配，优化精度-压缩率权衡。
4. **校准数据至关重要**：在 Kimi-K2.5 上的表现表明，校准数据的分布直接影响量化后的领域泛化能力。

### **方法的局限性**
- **计算开销较高**：由于引入辅助 logits，训练内存占用是原模型的 2–5 倍。
- **依赖校准数据质量**：性能高度依赖校准数据的代表性。
- **未覆盖激活/缓存量化**：目前仅针对 weight-only PTQ。

### **未来工作方向**
- 扩展至 **activation quantization** 和 **KV-cache quantization**。
- 探索更高效的 relaxation 方式，支持 **1-bit 甚至 sub-1-bit** 量化。
- 引入 **task-aware objectives** 进行端到端优化。
- 研究 **动态量化** 或 **自适应校准** 策略。

> **最终结论**：GSQ 表明，**硬件友好的 scalar quantization 仍有巨大潜力**，关键在于认真对待离散优化问题。这一思路为高效 LLM 部署提供了新的可能性。

</details>

---

### 15. [SCATR: Simple Calibrated Test-Time Ranking](https://arxiv.org/abs/2604.16535)

**Authors**: Divya Shyamal, Marta Kne\v{z}evi\'c, Lan Tran, Chanakya Ekbote, Vijay Lingam, Paul Pu Liang  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.16535v1  

#### Abstract
Test-time scaling (TTS) improves large language models (LLMs) by allocating additional compute at inference time. In practice, TTS is often achieved through parallel scaling: generating multiple candidate responses and selecting the best via a Best-of-N (BoN) strategy. Its effectiveness therefore hi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SCATR: Simple Calibrated Test-Time Ranking 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Test-time scaling (TTS)** 中的 **Best-of-N (BoN)** 响应选择问题展开研究。在 TTS 框架下，模型会生成多个候选响应，然后通过一个评分函数选择最优输出。然而，现有方法存在以下问题：

- **轻量级置信度启发式方法**（如基于 token log-probabilities 的 confidence metrics）虽然高效，但信号校准差（poorly calibrated），常表现接近随机选择，尤其在开放域任务（如代码生成、数学推理）中效果有限。
- **强学习型评分器**（如 Process Reward Models, PRMs）虽有效，但训练和部署成本高昂，需要大量标注数据和计算资源。

### 提出的新方法
作者提出 **SCATR**（Simple Calibrated Test-time Ranking），一种简单高效的 BoN 排名方法，其核心思想是：

- 利用语言模型**倒数第二层**（penultimate layer）的**隐藏表示**（hidden representations）作为质量信号，而非仅依赖最终的 token 概率。
- 在一个小规模的**校准集**（calibration set）上训练一个轻量级的 MLP 评分模型（scoring model），将隐藏状态映射为正确性估计。
- 该方法实现了**模型特定**（model-specific）和**领域特定**（domain-specific）的适应性，同时保持极低的开销。

### 相比现有方法的优势
- **高效率**：相比 PRM 和 LoRA 微调，SCATR 的可训练参数减少高达 **8000×**，训练和推理延迟分别降低 **150×** 和 **1000×**。
- **高性能**：在编码和数学推理任务上，SCATR 比现有的置信度基线平均提升高达 **9%**，甚至在某些设置下超越强大的 PRM 基线，准确率提升达 **7.8%**（数学）和 **4.2%**（编码）。
- **数据高效**：仅需几百个校准问题即可训练出有效的评分器，展现出极强的数据效率。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了**代码生成**和**数学推理**两大领域：

- **代码生成**：
  - `HumanEval` (Chen et al., 2021)
  - `KodCode` (Xu et al., 2025)，使用其中的 1K 子集
- **数学推理**：
  - `AIME24`, `AIME25`
  - `MATH-500` (Lightman et al., 2023)

### 实验设置和评估指标
- **模型**：在五个不同规模和家族的 LLM 上进行测试：
  - `Qwen3-1.7B`, `OLMo-2-7B`, `Qwen2.5-14B-Instruct`, `GPT-OSS-20B`, `Qwen3-30B-A3B`
- **TTS 设置**：对每个提示（prompt）并行生成 `N=16` 个候选响应（rollouts）。
- **校准集**：使用约 100–1000 个带二元标签（正确/错误）的 prompt-response 对来训练 SCATR 的 MLP 评分器。
- **评估指标**：
  - **准确性**（Accuracy）：BoN 选择后返回的最高分响应的正确率。
  - **效率指标**：可训练参数数量、训练时间（TFLOPs, min）、推理延迟（ms）。
  - **消融研究**：分析不同层、校准集大小、评分器架构的影响。

### 基线方法对比
- **随机选择**（Random）
- **多数投票**（Majority Voting）——适用于有唯一答案的任务
- **置信度启发式**（Confidence-based Heuristics）：
  - `C_avg` (Average Trace Confidence)
  - `C_tail` (Tail Confidence)
  - `C_bottom-10%` (Bottom 10% Group Confidence)
- **学习型验证器**（Learned Verifiers）：
  - `ReasonFlux-PRM-1.5B/7B`：强大的过程奖励模型（Process Reward Model）
  - `LoRA`：在基础模型上微调 LoRA 适配器以预测响应正确性

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **相对于置信度基线**：SCATR 在数学和编码任务上平均提升 **6–9%**，最大提升达 **9.1%**（见 Figure 3）。
- **相对于 PRM**：
  - 在 `Qwen-1.7B` + `AIME` 上，SCATR 准确率 **42.4%** vs. ReasonFlux-1.5B 的 **37.8%**，**+4.6%**。
  - 在 `Qwen-1.7B` + `MATH-500` 上，SCATR **88.3%** vs. ReasonFlux-1.5B **87.3%**，**+1.0%**。
  - 同时，SCATR 的推理速度比 PRM 快 **1000×**（见 Table 10）。
- **相对于 LoRA**：
  - 在 `GPT-OSS-20B` + `KodCode` 上，SCATR **86.8%** vs. LoRA **85.2%**，**+1.6%**。
  - SCATR 可训练参数仅为 LoRA 的 **~1/8000**，推理速度快 **1000×**（见 Table 8 & 9）。

### 与基线方法的对比结果
| 方法 | 准确率 | 参数量 | 推理延迟 | 相对于 SCATR |
|------|--------|--------|----------|-------------|
| **置信度启发式** | ~80–85% | 极低 | 极快 | **显著更低** (+ up to 9%) |
| **ReasonFlux-PRM-1.5B** | 高 | 1.5B | 高 (~100ms) | **相当或略低**，但慢 1000× |
| **LoRA** | 高 | ~2–7M | 中等 (~50ms) | **相当或略低**，但慢 1000× |
| **SCATR (ours)** | **最高或相当** | **~2–7M** | **极快 (~0.2ms)** | **基准** |

### 消融实验结果
- **层的选择**（Layer Choice）：
  - 使用倒数第二层（L-1）表现稳健且方差小，并非绝对最优但是一个可靠的默认选择（见 Figure 5 左）。
- **校准集大小**（Calibration Set Size）：
  - 性能随校准数据增加而提升，但在 `KodCode`（1000题）上已趋于饱和，表明方法具有良好的数据效率（见 Figure 5 右）。
- **评分器设计**（Scoring Function Design）：
  - 更复杂的架构（如 Transformer-based ensemble）并未带来一致增益，简单的单层 MLP 在倒数第二层表示上已是强基线（见 Table 3）。

---

## 4. 关键结论和发现

### 主要发现
1. **固定置信度度量信号有限**：基于 token 概率的置信度启发式在 BoN 选择中表现不佳，常接近随机水平，凸显了其在自由形式任务中的局限性。
2. **隐藏状态蕴含丰富质量信号**：模型内部的隐藏表示（尤其是倒数第二层）包含了远超 logits 的上下文信息，可用于更可靠地判断响应质量。
3. **轻量级校准非常有效**：仅需少量校准数据训练一个小型 MLP，即可实现对模型输出的精准打分，达到甚至超越复杂 PRM 的效果。
4. **卓越的准确率-效率权衡**：SCATR 在性能上匹敌或超越强基线的同时，实现了数量级的效率提升，使其成为可扩展 TTS 的理想选择。

### 方法的局限性
- **依赖中间表示**：必须访问模型的中间隐藏层，因此仅适用于 **open-weight models**，无法用于黑盒 API 模型。
- **单表示限制**：目前仅使用最后一个非填充 token 的单一嵌入，未探索更丰富的信号（如 token-level states, attention patterns）。
- **多轮对话支持不足**：当前方法针对单轮推理任务，如何扩展到多轮交互场景仍是开放问题。
- **小模型上的性能饱和**：在较小或能力较弱的模型上，增加候选数量可能导致性能停滞或下降（见 Figure 6）。

### 未来工作方向
- 将 SCATR 扩展到 **multi-turn** 或 **interactive** 推理场景。
- 探索更丰富的内部信号，如注意力模式、层间动态变化等。
- 研究 **hybrid design**：结合轻量级 SCATR 与重型 PRM，在简单问题上使用 SCATR 快速处理，复杂问题上触发 PRM 进行深度验证。
- 开发无需访问隐藏层的替代方案，以支持闭源模型。

</details>

---

### 16. [Efficient Federated RLHF via Zeroth-Order Policy Optimization](https://arxiv.org/abs/2604.17747)

**Authors**: Deyi Wang, Qining Zhang, Lei Ying  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.17747v1  

#### Abstract
This paper considers reinforcement learning from human feedback in a federated learning setting with resource-constrained agents, such as edge devices. We propose an efficient federated RLHF algorithm, named Partitioned, Sign-based Stochastic Zeroth-order Policy Optimization (Par-S$^2$ZPO). The algo...

---

### 17. [HopRank: Self-Supervised LLM Preference-Tuning on Graphs for Few-Shot Node Classification](https://arxiv.org/abs/2604.17271)

**Authors**: Ziqing Wang, Kaize Ding  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.17271v1  

#### Abstract
Node classification on text-attributed graphs (TAGs) is a fundamental task with broad applications in citation analysis, social networks, and recommendation systems. Current GNN-based approaches suffer from shallow text encoding and heavy dependence on labeled data, limiting their effectiveness in l...

---

### 18. [River-LLM: Large Language Model Seamless Exit Based on KV Share](https://arxiv.org/abs/2604.18396)

**Authors**: Yingtao Shen, An Zou  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.18396v1  

#### Abstract
Large Language Models (LLMs) have demonstrated exceptional performance across diverse domains but are increasingly constrained by high inference latency. Early Exit has emerged as a promising solution to accelerate inference by dynamically bypassing redundant layers. However, in decoder-only archite...

---

### 19. [CCCL: In-GPU Compression-Coupled Collective Communication](https://arxiv.org/abs/2604.17172)

**Authors**: Chon Lam Lao, Zhiying Xu, Zhuang Wang, Ziming Mao, Delong Meng, Jia Zhen, Jun Wu, Ion Stoica, Yida Wang, Yang Zhou  
**Category**: cs.DC  
**Published**: 2026-04-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.17172v1  

#### Abstract
Collective communication incurs significant overhead in LLM workloads. Although overlapping communication with computation in application-level is a common strategy, it often requires substantial code modifications and is impractical for many workloads (e.g., tensor and expert parallelism). We prese...

---

### 20. [UniCon: Unified Framework for Efficient Contrastive Alignment via Kernels](https://arxiv.org/abs/2604.16678)

**Authors**: Hangke Sui, Yuqing Wang, Minh N Do  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.16678v1  

#### Abstract
Contrastive objectives power state-of-the-art multimodal models, but their training remains slow, relying on long stochastic optimization. We propose a Unified Framework for Efficient Contrastive Alignment via Kernels (UniCon), which spans linear and nonlinear encoders as well as one-to-one and many...

---

### 21. [Open-TQ-Metal: Fused Compressed-Domain Attention for Long-Context LLM Inference on Apple Silicon](https://arxiv.org/abs/2604.16957)

**Authors**: Sai Vegasena  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.16957v1  

#### Abstract
We present Open-TQ-Metal, the first implementation of fused compressed-domain attention on Apple Silicon, enabling 128K-context inference for Llama 3.1 70B on a single 64GB consumer Mac -- a configuration impossible with all existing inference frameworks. Open-TQ-Metal quantizes the KV cache to int4...

---

### 22. [Fully Analog Resonant Recurrent Neural Network via Metacircuit](https://arxiv.org/abs/2604.17277)

**Authors**: Zixin Zhou, Tianxi Jiang, Menglong Yang, Zhihua Feng, Qingbo He, Shiwu Zhang  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.17277v1  

#### Abstract
Physical neural networks offer a transformative route to edge intelligence, providing superior inference speed and energy efficiency compared to conventional digital architectures. However, realizing scalable, end-to-end, fully analog recurrent neural networks for temporal information processing rem...

---

### 23. [Towards a Data-Parameter Correspondence for LLMs: A Preliminary Discussion](https://arxiv.org/abs/2604.17384)

**Authors**: Ou Wu  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.17384v1  

#### Abstract
Large language model optimization has historically bifurcated into isolated data-centric and model-centric paradigms: the former manipulates involved samples through selection, augmentation, or poisoning, while the latter tunes model weights via masking, quantization, or low-rank adaptation. This pa...

---

### 24. [FlashFPS: Efficient Farthest Point Sampling for Large-Scale Point Clouds via Pruning and Caching](https://arxiv.org/abs/2604.17720)

**Authors**: Yuzhe Fu (Helen), Hancheng Ye (Helen), Cong Guo (Helen), Junyao Zhang (Helen), Qinsi Wang (Helen), Yueqian Lin (Helen), Changchun Zhou (Helen),  Hai (Helen),  Li, Yiran Chen  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.17720v1  

#### Abstract
Point-based Neural Networks (PNNs) have become a key approach for point cloud processing. However, a core operation in these models, Farthest Point Sampling (FPS), often introduces significant inference latency, especially for large-scale processing. Despite existing CUDA- and hardware-level optimiz...

---

### 25. [Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF](https://arxiv.org/abs/2604.17769)

**Authors**: Yuan Fang, Yiming Luo, Aimin Zhou, Fei Tan  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.17769v1  

#### Abstract
Ensuring the safety of large language models (LLMs) requires robust red teaming, yet the systematic synthesis of high-quality toxic data remains under-explored. We propose Reverse Constitutional AI (R-CAI), a framework for automated and controllable adversarial data generation that moves beyond isol...

---

### 26. [KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving](https://arxiv.org/abs/2604.16682)

**Authors**: Yichao Yuan, Mosharaf Chowdhury, Nishil Talati  
**Category**: cs.DC  
**Published**: 2026-04-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.16682v1  

#### Abstract
Power has become a central bottleneck for AI inference. This problem is becoming more urgent as agentic AI emerges as a major workload class, yet prior power-management techniques focus almost entirely on single-turn LLM serving. Our analysis shows that agentic serving behaves fundamentally differen...

---

### 27. [AutoOR: Scalably Post-training LLMs to Autoformalize Operations Research Problems](https://arxiv.org/abs/2604.16804)

**Authors**: Sumeet Ramesh Motwani, Chuan Du, Aleksander Petrov, Christopher Davis, Philip Torr, Antonio Papania-Davis, Weishi Yan  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.16804v1  

#### Abstract
Optimization problems are central to decision-making in manufacturing, logistics, scheduling, and other industrial settings. Translating complicated descriptions of these problems into solver-ready formulations requires specialized operations research (OR) expertise, making it hard to scale. We pres...

---

### 28. [Federated Rule Ensemble Method in Medical Data](https://arxiv.org/abs/2604.17956)

**Authors**: Ke Wan, Kensuke Tanioka, Toshio Shimokawa  
**Category**: cs.LG  
**Published**: 2026-04-21  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.17956v1  

#### Abstract
Machine learning has become integral to medical research and is increasingly applied in clinical settings to support diagnosis and decision-making; however, its effectiveness depends on access to large, diverse datasets, which are limited within single institutions. Although integrating data across ...

---

### 29. [LACE: Lattice Attention for Cross-thread Exploration](https://arxiv.org/abs/2604.15529)

**Authors**: Yang Li, Zirui Zhang, Yang Liu, Chengzhi Mao  
**Category**: cs.AI  
**Published**: 2026-04-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.15529v1  

#### Abstract
Current large language models reason in isolation. Although it is common to sample multiple reasoning paths in parallel, these trajectories do not interact, and often fail in the same redundant ways. We introduce LACE, a framework that transforms reasoning from a collection of independent trials int...

---

### 30. [ONTO: A Token-Efficient Columnar Notation for LLM Input Optimization](https://arxiv.org/abs/2604.17512)

**Authors**: Harshavardhanan Deekeswar  
**Category**: cs.CL  
**Published**: 2026-04-21  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.17512v1  

#### Abstract
Serialization formats designed for document interchange impose structural overhead that becomes prohibitive when large language models consume operational data at scale. A modest dataset of 1,000 IoT sensor readings serialized as JSON requires approximately 80,000 tokens - the majority spent on repe...

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
