# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-13 06:17:21 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization](https://arxiv.org/abs/2603.11873)

**Authors**: Qiyang Li, Rui Kong, Yuchen Li, Hengyi Cai, Shuaiqiang Wang, Linghe Kong, Guihai Chen, Dawei Yin  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 15.5  
**Type**: new  
**ArXiv ID**: 2603.11873v1  

#### Abstract
The integration of dynamic, sparse structures like Mixture-of-Experts (MoE) with parameter-efficient adapters (e.g., LoRA) is a powerful technique for enhancing Large Language Models (LLMs). However, this architectural enhancement comes at a steep cost: despite minimal increases in computational loa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
动态适配器（dynamic adapters），如基于 **Mixture-of-Experts (MoE)** 和 **LoRA** 的架构，在提升大语言模型（LLMs）能力方面表现出色，但其推理延迟极高。尽管计算量仅增加约1–5%，解码速度却可能下降 **2.5倍以上**。

作者通过细粒度性能分析发现，瓶颈并非来自计算本身，而是由于传统动态路由机制导致的**大量碎片化、串行的 CUDA kernel 调用**，造成严重的系统开销。

> 🔍 **核心洞察**：推理延迟的主要来源是 *kernel launch overhead*，而非 FLOPs 增加。

---

### 🚀 提出的新方法与创新思路

提出 **AdaFuse** —— 一种算法与硬件协同设计（system-algorithm co-design）的高效动态适配器框架，包含两大核心技术：

#### （1）Token-Level Pre-Gating（令牌级预门控）
- 改变传统的逐层（layer-wise）或逐块（block-wise）动态路由方式。
- 在第一层使用一个全局 **Top-2 Router** 对整个 token 的所有 adapter 层进行一次性路由决策。
- 实现“决定一次，处处适用”（decide-once, apply-everywhere）策略，将原本动态的执行路径静态化，为后续优化提供基础。

#### （2）Fused Kernel Optimization（融合内核优化）
- 设计专用 CUDA kernel：**SGMM (Segmented Gather Matrix Multiplication)**。
- 将多个被激活的 LoRA adapter 参数在单次 kernel 调用中合并到 backbone 模型中。
- 显著减少 kernel launch 次数，避免频繁内存访问和上下文切换。

> 💡 这种“先预判 + 再融合”的设计，从根本上改变了动态 adapter 的执行范式。

---

### ⚖️ 相比现有方法的优势

| 维度 | 传统动态 adapter（如 MoRAL, PESC） | AdaFuse |
|------|-------------------------------|--------|
| 路由粒度 | Layer-wise / Block-wise | **Token-wise** |
| 路由频率 | 每层重复计算 | **仅首层一次决策** |
| Adapter 合并 | 无法提前合并 | **可全局预合并** |
| Kernel 调用 | 多次、碎片化 | **单次、融合调用（SGMM）** |
| 推理效率 | 极低（+250%~950% 延迟） | **接近原生模型（仅 +29%）** |

> ✅ 在保持甚至超越 baseline 准确率的同时，实现高达 **2.4–2.7× 的解码加速**。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### （1）通用能力评估（General Capability）
- **训练数据**：
  - SlimORCA（多任务指令数据）
  - Magicoder（代码生成）
  - MetaMathQA（数学问答）
- **测试基准（via LM-Eval-Harness）**：
  - ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, MT-Bench

#### （2）领域特定任务（Domain-Specific Customization）
- ScienceQA, CommonsenseQA, OpenbookQA

#### （3）运行时效率测试
- **ShareGPT 数据集**：模拟真实用户查询
- 服务 50 个 query，每个生成 200 个 token，测量平均解码延迟

---

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| 主干模型 | Llama2-7B, Mistral-7B |
| Adapter 类型 | LoRA（rank=64/128） |
| 专家数量 | 通常 N=4~16 |
| 路由机制 | Top-2 selection |
| 硬件平台 | NVIDIA GPU（具体型号未详述，但强调 CUDA 优化） |

#### 评估指标：
- **准确性**：各 benchmark 上的 accuracy
- **推理延迟**：ms/token（解码阶段）
- **显存占用**：Peak GPU memory (GiB)
- **参数增长**：Parameter size (% increase)
- **计算复杂度**：FLOPS 变化

---

### 🆚 基线方法对比

| 方法 | 类型 | 特点 |
|------|------|------|
| **LoRA** | Static adapter | 单一适配器，无动态选择 |
| **MoRAL** | Layer-wise dynamic | 每层独立路由，典型代表 |
| **MOLA** | Layer-wise MoE | 类似 MoRAL，侧重多任务 |
| **PESC** | Block-wise dynamic | 按模块分组路由 |

> 所有 baseline 均引入显著延迟，而 AdaFuse 旨在打破“高性能 ↔ 高延迟”权衡。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### （1）准确率表现（Competitive Accuracy）

| 方法 | General Avg (%) | Domain-Specific Avg (Llama2-7B) | Domain-Specific Avg (Mistral-7B) |
|------|------------------|-------------------------------|----------------------------------|
| PESC (best baseline) | 60.45 | 81.47 | 87.06 |
| **AdaFuse (ours)** | **60.12** | **83.60** | **87.24** |

> ✅ AdaFuse 在多数任务上达到或超过 SOTA 动态 adapter 的精度，尤其在 **CommonsenseQA** 和 **MMLU** 表现出更强推理能力。

---

#### （2）推理延迟对比（Dramatic Speedup）

| 方法 | 解码延迟 (ms/token) | 相对原始模型增幅 |
|------|--------------------|------------------|
| Llama2-7B (base) | 2.4 | — |
| MOLA | 25.3 | **+954%** |
| PESC | 8.5 | +254% |
| MoRAL | 8.6 | +258% |
| **AdaFuse (ours)** | **3.1** | **+29%** |

> 🔥 **AdaFuse 比最快的 baseline（PESC）快 2.7 倍，比 MoRAL 快 2.8 倍！**

---

#### （3）消融实验（Ablation Study）

| 方法 | 解码延迟 (ms/token) | 增幅 |
|------|--------------------|-----|
| Llama2-7B | 2.4 | — |
| MoRAL | 8.5 | +254% |
| MoRAL + Simple Merge | 4.5 | +88% |
| AdaFuse + Simple Merge | 4.2 | +75% |
| **AdaFuse + SGMM (full)** | **3.1** | **+29%** |

> 🔍 发现：
> - 即使采用 token-wise 路由，若不使用 **SGMM kernel**，仍存在明显延迟。
> - **SGMM 是实现极致优化的关键组件**，贡献了近一半的速度提升。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **动态 adapter 的高延迟主因不是计算，而是 CUDA kernel 开销**  
   → 系统层面的设计缺陷限制了算法潜力。

2. **Token-level pre-gating 可有效静态化动态路由路径**  
   → 允许在 token 处理前完成所有 adapter 的选择与合并。

3. **Fused kernel（SGMM）能极大降低 kernel launch 次数**  
   → 实现真正的“一次合并，全程使用”，逼近原生推理效率。

4. **AdaFuse 成功弥合了“模型表达力”与“推理效率”之间的鸿沟**  
   → 实现 **accuracy 不降、speed 提升 >2.4×** 的双赢。

---

### ⚠️ 方法的局限性

1. **依赖语义一致性假设**  
   → 假设同一 token 在不同层应激活相同专家。虽然实验证明该假设合理，但在极端跨层语义变化场景下可能失效。

2. **prefilling 阶段未优化**  
   → 当前优化集中于 decoding 阶段，prefill 性能与传统方法相当。

3. **定制 kernel 移植性受限**  
   → SGMM 是高度定制化的 CUDA kernel，依赖特定 GPU 架构，部署门槛较高。

4. **专家共享模式固定**  
   → 所有层共用相同的专家集合，可能不如完全分层专家灵活。

---

### 🔮 未来工作方向

1. **扩展至 vision-language 或 multimodal models**  
   → 探索 AdaFuse 在 CLIP、Flamingo 等架构中的应用。

2. **支持更复杂的路由策略**  
   → 如 conditional depth-wise routing 或 feedback-driven gating。

3. **开发通用化推理后端支持**  
   → 将 SGMM 集成进主流推理引擎（如 vLLM、TensorRT-LLM）。

4. **探索自动化的 rank / expert 数量配置机制**  
   → 结合 NAS 或强化学习实现自适应资源配置。

---

## ✅ 总结

> **AdaFuse 是首个从系统-算法协同角度解决动态 adapter 推理瓶颈的工作**。它通过 **token-level pre-gating + fused SGMM kernel** 的组合拳，实现了：
>
> - ✅ **精度媲美 SOTA 动态 adapter**
> - ✅ **解码速度快 2.4–2.7×**
> - ✅ **延迟仅比原模型高 29%**
>
> 为构建**高性能且高效率**的可扩展 LLM 提供了新的设计范式，有望成为未来动态 PEFT 技术的标准基础设施之一。

</details>

---

### 2. [Where Matters More Than What: Decoding-aligned KV Cache Compression via Position-aware Pseudo Queries](https://arxiv.org/abs/2603.11564)

**Authors**: Zhenxu Tian, Yi Su, Juntao Li, Min Zhang  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2603.11564v1  

#### Abstract
The Key-Value (KV) cache is crucial for efficient Large Language Models (LLMs) inference, but excessively long contexts drastically increase KV cache memory footprint. Existing KV cache compression methods typically rely on input-side attention patterns within a prompt observation window to estimate...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Where Matters More Than What: Decoding-aligned KV Cache Compression via Position-aware Pseudo Queries

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Large Language Models (LLMs)** 的推理过程中，**Key-Value (KV) cache** 是提升自回归解码效率的关键机制。然而，随着输入上下文长度的增长，KV cache 的内存占用急剧增加，严重制约了模型在长文本任务中的部署效率。

现有的 **KV cache 压缩方法**（如 token eviction）通常依赖于预填充阶段（prefill stage）中基于输入序列的注意力模式来评估 token 重要性。这类方法存在一个根本缺陷：它们的“观察窗口”（observation window）是**以输入为中心的**，无法准确反映**解码阶段动态生成过程**中真正重要的 token。

因此，这些方法容易误删对后续生成至关重要的 token，尤其在复杂或噪声较多的长上下文中表现不佳。

---

### 提出了什么新方法或新思路
本文提出了一种全新的 KV cache 压缩框架：**Decoding-aligned KV cache compression via position-aware pseudo queries (DapQ)**。

其核心思想是：
> **位置信息比语义内容更重要**（Where matters more than what），即在构建用于模拟解码查询的伪查询（pseudo queries）时，**正确的 positional encoding** 比真实的语义内容更能决定注意力分布。

#### DapQ 方法流程如下：
1. **构造伪上下文**：在原始输入序列后附加一组人工构造的 `Tpseudo` token（语义内容可任意选择）。
2. **赋予正确位置 ID**：将这些 `Tpseudo` 的位置 ID 设置为模型即将生成的前 N 个 token 的位置（例如 `[Lp, Lp+1, ..., Lp+N-1]`）。
3. **获取伪查询 Qpseudo**：在 prefill 阶段处理扩展后的序列，从 `Tpseudo` 对应的位置提取出带有正确 RoPE 编码的 **pseudo queries**。
4. **重要性评估与压缩**：利用 `Qpseudo` 与原始 prompt 中所有 keys 计算注意力得分，聚合后保留得分最高的 Top-K tokens，其余被剔除。
5. **开始解码**：丢弃整个 `Tpseudo` 段，从位置 `Lp` 开始进行正常的自回归解码。

该方法通过 **position-aware pseudo queries** 构建了一个与实际解码过程对齐的、动态的观察窗口，从而更精准地识别关键 token。

---

### 相比现有方法的优势
- **更高的对齐性**：相比 SnapKV、StreamingLLM 等仅关注输入末尾的静态窗口，DapQ 显式模拟了解码初期的查询行为，显著提升了重要性评估的准确性。
- **轻量高效**：无需像 LAQ++ 那样进行两阶段生成，避免了额外的内存峰值和计算开销。
- **鲁棒性强**：对 `Qpseudo` 的语义内容不敏感，允许灵活设计而不会影响性能。
- **通用性好**：在多种 LLMs 和任务上均表现出色，尤其在极端压缩条件下仍能保持接近无损性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖多个主流长上下文基准测试，涵盖不同任务类型：

| 数据集 | 任务类型 |
|--------|----------|
| **LongBench** | 多语言、多任务长上下文理解（单文档问答、多文档问答、摘要等） |
| **LongBenchV2** | 更复杂的现实场景长上下文推理 |
| **Ruler** | 测试真实上下文长度能力（特别是 Needle-in-a-Haystack 类型任务） |
| **HELMET** | 综合评估长上下文模型性能 |
| **Needle-in-a-Haystack (NIAH)** | 合成任务，检测模型能否从海量无关文本中定位关键信息 |

---

### 实验设置和评估指标

#### 模型
- LLaMA-3-8B-Instruct
- LLaMA-3.1-8B-Instruct
- Qwen2.5-7B-Instruct
- Qwen3-8B (Reasoning OFF)

#### 压缩预算（KV Cache Size）
- 多种严格限制条件：256, 128, 64, 甚至低至 3% 的总缓存比例

#### 评估指标
- **任务准确率（Accuracy / F1 / Rouge-L 等）**
- **Recall@K**：衡量保留的 token 是否覆盖了由真实响应查询选出的关键 token
- **Throughput (tokens/s)**：吞吐量
- **Time-to-First-Token (TTFT)**：首 token 延迟
- **内存使用（Memory Usage）**

---

### 基线方法对比
| 方法 | 简要说明 |
|------|--------|
| **FullKV** | 不压缩，保留全部 KV |
| **SnapKV** | 使用最近 token 构建观察窗口，结合池化注意力评分 |
| **PyramidKV** | 动态分配各层 cache 预算，基于跨层注意力分布 |
| **H2O** | 识别“重击者”（Heavy Hitter）token 并平衡近期与高频 token 保留 |
| **StreamingLLM (SLM)** | 利用 attention sink 机制维持上下文连贯性 |
| **LaCache** | 采用阶梯形 pattern 在浅层保留早期 token，深层保留后期 token |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 在 **Needle-in-a-Haystack (NIAH)** 上的表现（LLaMA-3-8B-Instruct, KV Size=256）
| 方法 | 准确率 (Acc) |
|------|-------------|
| FullKV | 100.00% |
| H2O | 66.81% |
| PyramidKV | 93.94% |
| SnapKV | 90.97% |
| **DapQ** | **99.46%** |

> 🔥 **DapQ 达到了近乎无损的性能（99.5%）**，远超其他压缩方法，在仅保留极小部分 cache 的情况下仍能精准找回“针”。

---

#### ✅ 在 **LongBench** 上平均得分（KV Size=64）
| 方法 | Average Score |
|------|----------------|
| FullKV | 40.96 |
| SnapKV | 40.78 |
| **DapQ** | **41.81** |

> 即使在最严苛的压缩下（64 tokens），DapQ 依然优于所有基线。

---

#### ✅ 在 **LongBenchV2 Hard 类别** 上（LLaMA-3-8B-Instruct, KV=64）
| 方法 | Accuracy |
|------|----------|
| SnapKV | 22.51% |
| **DapQ** | **29.26%** |

> **绝对提升 +6.75%**，显示其在高难度推理任务上的强大优势。

---

#### ✅ 在 **Ruler S-NIAH-3 任务** 上（KV=512）
| 方法 | Accuracy |
|------|----------|
| SnapKV | 1.4% |
| H2O | 2.4% |
| **DapQ** | **59.6%** |

> 跨越式提升，证明其对长距离依赖建模的有效性。

---

#### ✅ 效率指标（LLaMA-3.1-8B-Instruct, Batch Size=50）
| 方法 | Throughput (tokens/s) | TTFT (s) |
|------|------------------------|----------|
| FullKV | OOM | – |
| SnapKV | 40.23 | 1.1278 |
| **DapQ** | **40.12** | **1.1298** |

> 性能与 SnapKV 几乎持平，**额外 prefill 开销可忽略不计**，具备实用价值。

---

### 消融实验结果

#### 📌 **Qpseudo 语义内容的影响**
- 使用不同语义构造 `Qpseudo`（前缀+后缀拼接 / 随机 token / 固定无意义句）：
  - 性能波动极小（CV ≈ 1%）
  - **结论**：语义内容非关键因素，验证了“位置主导”的假设。

#### 📌 **Qpseudo 长度（观察窗口大小 N）的影响**
- 存在一个**非单调最优值**：
  - 小窗口（N < 32）：信息不足，召回率低
  - 中等窗口（N ≈ 32–64）：性能达到峰值
  - 过大窗口（N > 64）：引入过多推测性 future queries，导致注意力信号稀释，性能下降
- **结论**：并非越大越好，需权衡代表性与噪声干扰。

#### 📌 **插入位置的影响**
- 若将 `Qpseudo` 插入原始上下文中间而非末尾：
  - 受限于 causal attention mask，无法访问完整上下文
  - 导致全局注意力模式重建失败，性能显著下降
- **结论**：必须放置在 prompt 之后，并赋予正确 future position IDs。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **位置信息主导 query 表示**：在 query 向量中，**positional encoding 的作用远大于 semantic content**。即使使用完全无关的内容，只要位置正确，就能高度近似真实解码查询。
2. **高质量 pseudo queries 可合成**：无需生成真实输出 token，即可通过 future-position-enriched 伪 token 构造出有效的 pseudo queries。
3. **解码对齐的观察窗口至关重要**：传统的 input-centric 观察窗口与实际 generation context 错配；DapQ 通过 position-aware design 实现了更好的对齐。
4. **极致压缩下的稳定性**：在仅保留 3% KV cache 的极端条件下，DapQ 仍能在 NIAH 任务上实现近无损恢复（99.5%），展现出卓越的压缩潜力。

---

### 方法的局限性
1. **语义仍有潜在作用**：虽然位置占主导，但实验表明当语义也匹配时，相似度更高（Table 1）。当前方法未充分利用语义优化空间。
2. **层间差异未建模**：不同网络层可能对位置/语义的敏感度不同，统一的 `Qpseudo` 设计可能不是每层最优。
3. **固定长度窗口**：目前使用固定长度的 pseudo query 窗口，未能根据生成动态调整。

---

### 未来工作方向
1. **智能构造 Qpseudo 内容**：探索如何低成本地选择或生成更具语义一致性的 `Tpseudo`，进一步提升近似质量。
2. **分层自适应 approximation**：设计 layer-wise 或 adaptive 的 pseudo query 机制，根据不同层的注意力特性定制策略。
3. **动态窗口调度**：根据上下文复杂度或生成状态动态调整 `Qpseudo` 的数量和位置。
4. **与其他压缩技术融合**：结合 quantization、sharing、low-rank decomposition 等方法，打造端到端高效的推理系统。

---

> 💡 **一句话总结**：  
> DapQ 揭示了“**Where > What**”这一关键洞察，提出通过 **position-aware pseudo queries** 构建解码对齐的观察窗口，在几乎零额外开销的前提下实现了当前最先进的 KV cache 压缩效果，尤其在极端内存受限场景下展现出巨大潜力。

</details>

---

### 3. [Speculative Decoding Scaling Laws (SDSL): Throughput Optimization Made Simple](https://arxiv.org/abs/2603.11053)

**Authors**: Amirhossein Bozorgkhoo, Igor Molybog  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2603.11053v1  

#### Abstract
Speculative decoding is a technique that uses multiple language models to accelerate infer- ence. Previous works have used an experi- mental approach to optimize the throughput of the inference pipeline, which involves LLM training and can be costly. This study of spec- ulative decoding proposes a t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Speculative Decoding Scaling Laws (SDSL): Throughput Optimization Made Simple*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
Speculative Decoding 是一种通过使用小型 **draft model** 预测候选 token，并由大型 **target model** 并行验证来加速 LLM 推理的技术。然而，其性能高度依赖于 draft model 的选择——不合适的 draft model 可能引入延迟瓶颈，反而降低吞吐量。

当前主流方法依赖**经验搜索**（empirical search）和跨架构基准测试来选择 draft model，这需要大量计算资源和训练成本，效率低下且不可复用。

本论文旨在解决这一问题：**能否在训练前就预测出最优的 draft model 大小？**

---

### 🚀 提出了什么新方法或新思路
作者提出了 **Speculative Decoding Scaling Laws (SDSL)** ——一个**解析框架**，将预训练阶段的 Scaling Laws 与推理阶段的 throughput 效率连接起来，从而实现对下游 speculative decoding 系统的吞吐量优化。

#### 主要创新点包括：
- **建立了一个分析性关系**：  
  提出一个简单公式 $ \alpha = ax + by + c $，其中：
  - $ \alpha $：预期 token 接受率（expected token acceptance）
  - $ x $：draft model 的 **perplexity**
  - $ y $：target model 的 **perplexity**  
  该关系表明 $ \alpha $ 主要由 draft model 的质量决定。

- **推导出最优 draft model 大小的经验法则**：  
  在假设 draft 和 target 模型均从零开始预训练的前提下，得出：
  $$
  N_{\text{opt}} = \mu M + M_0
  $$
  其中 $ N_{\text{opt}} $ 是最优 draft model 参数量，$ M $ 是 target model 参数量。研究发现：
  - 最优 draft model 应约为 target model 的 **1/200**（即两个数量级更小）。
  - 这一比例在不同模型家族中具有鲁棒性。

- **构建可复用的 SDSL 框架**：  
  利用已有的 pre-training scaling laws（如 Hoffmann et al., 2022），结合本文提出的 $ \alpha $ 模型，直接预测 throughput，无需额外实验即可为任意目标模型推荐 draft 架构。

---

### 🔍 相比现有方法的优势
| 方面 | 现有方法 | 本文 SDSL 方法 |
|------|--------|---------------|
| **设计方式** | 经验试错、大规模 benchmarking | 分析建模、提前预测 |
| **资源消耗** | 高昂（需多次训练/部署） | 极低（仅需已有 scaling law 数据） |
| **通用性** | 特定配置下有效 | 跨模型族、跨数据规模适用 |
| **可解释性** | 黑箱调参 | 明确揭示 $ N^* \propto M $ 的规律 |

> ✅ **优势总结**：SDSL 将 speculative decoding 的系统设计从“试出来”变为“算出来”，显著降低了部署复杂性和研发成本。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **主评估数据集**：**HellaSwag**  
  包含常识推理任务，用于评估模型生成能力及 perplexity。
- 所有模型在此数据集上进行 perplexity 测量和 token acceptance rate ($ \alpha $) 估计。

---

### ⚙️ 实验设置
- **模型家族覆盖广泛**：
  - Target Models: `OPT-13B`, `OPT-30B`, `OPT-66B`, `LLaMA3-70B`, `LLaMA3.1-70B`, `Qwen1.5-14B~110B`, `Qwen2.5-14B~72B`, `Seed-OSS-36B`
  - Draft Models: `OPT-125M~2.7B`, `Qwen1.5-0.5B~4B`, `Qwen2.5-0.5B~3B`

- **实现工具**：使用 **Microsoft DeepSpeed** 库实现 speculative decoding。

- **tokenizer 处理**：采用统一策略，先用 target model 生成文本，再用 draft model 的 tokenizer 重新分词，确保兼容性。

- **lookahead length $ y $**：设为可变参数，在分析中取理论最优值以最大化 throughput。

---

### 📊 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **Token Acceptance Rate (TAR)** | 被 target model 接受的 draft token 比例 | 估算 $ \alpha $ 的基础 |
| **Expected Acceptance $ \alpha $** | 前缀上的平均接受概率 | 衡量 draft 与 target 对齐程度 |
| **Throughput (tokens/FLOP)** | 单位 FLOP 下生成的 token 数 | 主要优化目标（硬件无关） |
| **Wall-clock Latency (token/sec)** | 实际运行时间下的吞吐量 | 验证理论结果的实际有效性 |
| - Time-to-First-Token (TTFT) | 生成第一个 token 的延迟 | 系统响应速度 |
| - Total Generation Time (TTOT) | 生成固定长度序列总耗时 | 整体效率 |
| - Time-per-Output-Token (TPOT) | 平均每 token 耗时 | 吞吐稳定性 |

---

### 🔀 基线方法对比
本文未直接对比其他算法类 baseline（如 Self-Speculative Decoding 或 DistillSpec），而是聚焦于：
- **与纯经验方法对比**：展示无需 exhaustive search 即可准确预测 $ N^* $
- **消融分析**：验证 scaling law 中各变量的影响权重
- **latency-based validation**：用真实端到端延迟验证 throughput 预测的有效性

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）$ \alpha $ 与 perplexity 的关系
- 回归拟合得到：
  $$
  \alpha = -0.0067x + 0.0130y + 0.6421 \quad (R^2 = 0.60)
  $$
- 发现：
  - $ \alpha $ 与 **draft model perplexity $ x $** 强负相关（越低越好）
  - 与 target model perplexity $ y $ 关系较弱
  - **draft model 质量是主导因素**

#### （2）最优 draft model 大小 $ N^* $
- 数值优化后得到：
  $$
  N^* \approx 2.71 \times 10^{-3} \cdot M + 8.71 \times 10^7
  $$
- 当 $ M $ 较大时，渐近趋于：
  $$
  N^*/M \to 0.00271 \approx 1/370
  $$
  但在实际范围内平均接近 **1/200**。

| Target Model | Size (B) | Predicted $ N^* $ (M) | Ratio $ N^*/M $ |
|--------------|----------|-------------------------|------------------|
| OPT-13B      | 13       | ~36                    | ~0.28% (~1/360) |
| Qwen1.5-110B | 110      | ~371                   | ~0.34% (~1/290) |
| LLaMA3-70B   | 70       | ~283                   | ~0.40% (~1/250) |

> ✅ 总体趋势：**draft model 应比 target 小约两个数量级（~200x）**

---

### 🔁 与基线方法的对比结果
- **无需训练即可预测 $ N^* $**：
  - 传统方法需尝试多个 draft 架构 → 成本高
  - SDSL 只需输入 $ M $ 即可输出建议 $ N $

- **latency 验证结果支持 throughput 预测**：
  - 在 `OPT-13B` 上测试多种 draft models
  - 结果显示：当 $ N \approx N^* $ 时，TTFT、TTOT、TPOT 均达到最低
  - 即使跨 family（如 Qwen → OPT），趋势依然成立

> 图表显示：随着 $ |N - N^*| / M $ 增大，latency 单调上升，证明预测准确性。

---

### 🔍 消融实验结果
- **dataset size 影响较小**：
  - 改变 draft training data $ D $ 或 target data $ D' $ 对 $ N^* $ 影响微弱
  - $ \log D $ 的系数仅为 $ -6.07 \times 10^{-5} $，几乎可忽略
  - 表明：**只要训练数据量级相当（如 ~trillion tokens），dataset size 不是关键因素**

- **normalization collapse**：
  - 将 $ N^*/M $ 归一化后，不同 $ D $ 下曲线高度重合
  - 说明 $ M $ 是主导变量，其余为次级修正项

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **draft model 的大小应与 target model 成线性关系**：
   $$
   N_{\text{opt}} \propto M
   $$
   且比例常数约为 $ 0.5\% \sim 0.3\% $，即 **~200–300 倍更小**。

2. **最优 draft model 设计主要取决于 target model 大小 $ M $**，而受训练数据量影响极小。

3. **token 接受率 $ \alpha $ 主要由 draft model 的 perplexity 决定**，target model 的影响较弱。

4. **throughput 最大化等价于 latency 最小化**：
   - 在 `OPT-13B` 上实测验证，预测的 $ N^* $ 确实对应最小延迟

5. **该规律在多个模型族（OPT/Qwen/LLaMA）中具有一致性**，表明其泛化能力强。

---

### ⚠️ 方法的局限性
- **假设条件较强**：
  - draft 与 target 模型需在相似数据分布和训练流程下训练
  - 若存在 domain specialization、post-training alignment 或架构差异过大（如 MoE vs Dense），可能失效

- **未涵盖非自回归或 encoder-decoder 架构**：
  - 当前分析限于 autoregressive text-only models
  - 不适用于 T5、Unifying Frameworks 或 multi-modal systems

- **依赖高质量的 pre-training scaling laws**：
  - 若原始 scaling law 不准确（如参数外推区域），会影响最终预测精度

---

### 🔮 未来工作方向
1. **扩展至多模态与 encoder-decoder 架构**  
   探索 Vision-Language 或 Speech-to-Text 场景下的 speculative decoding scaling laws。

2. **纳入架构参数（depth/width ratio）**  
   如 Yan et al. (2025) 所示，depth-to-width 比也影响 latency，未来可联合建模。

3. **动态调整 $ y $ 和 $ N $ 的 runtime control policy**  
   结合 SDSL 与 online adaptation，实现自适应 speculative decoding。

4. **探索 distillation-based draft models 的 scaling behavior**  
   当前 draft 多为独立训练，若使用 knowledge distillation，是否仍遵循相同规律？

---

## ✅ 总结一句话
> 本文提出 **SDSL** 框架，首次将 speculative decoding 的设计从“经验试错”转变为“理论预测”，揭示了 **最优 draft model 应约为 target model 的 1/200** 这一简洁而强大的规律，极大简化了高效 LLM 推理系统的工程实践。

</details>

---

### 4. [AutoScout: Structured Optimization for Automating ML System Configuration](https://arxiv.org/abs/2603.11603)

**Authors**: Jimmy Shong, Yuhan Ding, Yihan Jiang, Liheng Jing, Haonan Chen, Gaokai Zhang, Aditya Akella, Fan Lai  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2603.11603v1  

#### Abstract
Machine learning (ML) systems expose a rapidly expanding configuration space spanning model-parallelism strategies, communication optimizations, and low-level runtime parameters. End-to-end system efficiency is highly sensitive to these choices, yet identifying high-performance configurations is cha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AutoScout: Structured Optimization for Automating ML System Configuration**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代 **ML 系统**（如 vLLM、Megatron）暴露了日益庞大的 **configuration space**，涵盖 **model-parallelism 策略**、**通信优化** 和 **低层运行时参数**。这些配置对系统端到端效率高度敏感，但存在以下挑战：
- **异构特征类型**：稀疏（categorical，如并行策略）与密集（continuous，如通信桶大小）参数共存。
- **层级依赖关系**：下游参数仅在特定上游决策下有效（如 `tp-comm` 依赖于 `tp > 1`）。
- **高 profiling 成本**：每次评估需真实 GPU 执行，耗时且昂贵。
- **现有方法局限**：要么只优化子集（如仅 3D 并行），要么依赖难以泛化的启发式规则。

### **提出了什么新方法或新思路**
提出 **AutoScout**，一个通用的 ML 系统配置优化器，用于训练、微调和推理任务。其核心创新包括：

- **混合优化框架（Hybrid Optimization Framework）**：
  - 将配置空间建模为 **mixed discrete-continuous 优化问题**，具有强层级依赖。
  - 分离 **sparse structural decisions**（如并行策略）与 **dense execution parameters**（如 `ddp_bucket`）进行联合优化。

- **Hierarchical Hybrid Search Optimizer**：
  - **Sparse Optimizer**：使用 **Monte Carlo Tree Search (MCTS)** 探索稀疏结构决策树。
  - **Dense Optimizer**：采用 **coordinate-wise SGD** 对连续参数进行梯度引导优化。

- **自适应协调机制（Adaptive Orchestrator）**：
  - 引入 **tournament-based feature prioritization**，在线学习最优特征顺序，提升 MCTS 效率。
  - 使用 **hybrid bandit mechanism**（基于 UCB1）动态分配搜索预算给 sparse 或 dense 优化器。

- **成本感知评估（Cost-Aware Evaluation）**：
  - 构建 **multi-fidelity simulator ensemble**（多个保真度模拟器），早期使用低成本预测。
  - 设计 **adaptive fidelity switch**：当模拟误差超过阈值时自动切换至真实 profiling，避免误导。

### **相比现有方法的优势**
- **更广的优化范围**：超越传统仅优化 3D 并行的方法，覆盖执行级参数，挖掘更大性能潜力。
- **更强的泛化能力**：不依赖固定启发式，适应不同模型、硬件、部署目标。
- **更高的搜索效率**：通过结构化探索 + 模拟器引导，显著减少搜索步数和 wall-clock 时间。
- **更好的鲁棒性**：能处理噪声模拟器输出，并稳定收敛。

---

## **2. 核心实验方法和设置**

### **使用的数据集与模型**
- **训练任务**：
  - `LLAMA-3.2-3B`, `LLAMA-3.1-NEMOTRON-NANO-VL-8B-V1`, `QWEN3-30B-A3B`
  - 数据集：`LMSYS-Chat-1M`
- **推理任务**：
  - `META-LLAMA-3-8B-INSTRUCT` on `LMSYS-Chat-1M`

### **实验设置**
- **硬件环境**：
  - 集群包含 8×A100 和 4×A40 GPU。
- **配置空间规模**：
  - 最多达 ~30,000 个可行配置，涵盖：
    - 并行度（`pp`, `tp`, `dp`, `ep`, `cp`）
    - 微批次大小（`mbs`）
    - 激活重计算（`ar`）
    - 通信参数（`ddp_bucket`, `tp-comm`）
    - 序列并行（`sp`）等
- **模拟器设计**：
  - 构建多个线性回归模拟器作为 ensemble（见附录 B）：
    - `3D-Parallelism`, `5D-Parallelism`, `DDP-Aware`, `Communication-Aware`
  - 使用加权平均（基于 R² 分数）融合预测。

### **评估指标**
- **最终性能**：端到端延迟
  - 训练：秒每迭代（s/iter）
  - 推理：毫秒每 token（ms/token）
- **搜索效率**：
  - 搜索步数（Search Steps）
  - 实际耗时（Real Time / Wall-Clock Time）
- **统计可靠性**：所有结果取 20 次独立运行均值。

### **基线方法对比**
| 方法 | 类型 | 说明 |
|------|------|------|
| **vLLM** | 基线框架 | 使用专家调优默认配置 |
| **Megatron-LM** | 基线框架 | 使用推荐的并行与执行设置 |
| **UDO** | MCTS-based | 通用系统配置优化器 |
| **CherryPick** | BO-based | 基于贝叶斯优化的云作业配置框架 |
| **Metis** | Auto-3D-Parallel | 自动发现最佳 3D 模型并行方案 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 在多种模型和任务上，AutoScout 找到的配置实现：
  - **1.3–3.0× 的训练速度提升**（vs. Megatron / expert-tuned 设置）
  - 推理任务上最高达 **1.02×** 性能增益（vs. vLLM 默认）

### **与基线方法的对比结果**
| 指标 | 结果 |
|------|------|
| **端到端性能** | 始终优于所有 baseline，尤其在复杂模型（如 QWEN-MoE）上表现突出 |
| **搜索效率（步数）** | 较 CherryPick / UDO 减少 **28.6%–93.07%** 的搜索步数 |
| **搜索时间加速** | 较现有 configurator 快 **13.7–16.5×**（最高达 22.9×） |
| **收敛稳定性** | 不会早停或陷入局部最优，表现出优秀的 **anytime performance** |

> 示例：在 QWEN-MoE 上，AutoScout 仅用约 13% 的搜索步数即超越 CherryPick 和 UDO 的最终结果。

### **消融实验结果（Ablation Studies）**
#### **组件有效性分析（图8）**
- **移除 Dense Optimizer**：性能下降 **1.46×** → 表明 dense 参数调优至关重要。
- **移除 Orchestrator**：搜索效率与质量双降 → 验证 adaptive 协调机制的关键作用。
- **禁用 Simulator Ensemble**：收敛变慢 **1.19×** → 显示低成本模拟器的有效“热启动”价值。

#### **Tournament 规模影响（图9）**
- **K=5~10**：平衡探索效率与开销，表现最佳。
- **K=40**：初期进展慢（探索稀释），但最终仍可收敛 → 说明机制具备可扩展性。

#### **配置空间扩展性（图10）**
- 即使从 3D → 5D → Full 配置空间逐步扩大，AutoScout 收敛时间仅轻微增加。
- 更丰富的特征反而提供更强信号，有助于更快剪枝。

#### **模拟器鲁棒性测试（图11）**
- 注入 **0% / 40% / 80% 噪声** 后，AutoScout 仍能稳定收敛至近优解。
- 自适应 fidelity switch 成功防止噪声误导，体现强健性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **ML 系统配置是 structured mixed optimization 问题**，必须显式建模稀疏-密集耦合与层级依赖。
2. **仅优化并行策略远远不够**：即使固定并行方案，其他执行参数仍可能导致高达 **42×** 的性能差异。
3. **联合优化 sparse 结构与 dense 参数** 可显著释放性能潜力，而 AutoScout 实现了高效协同。
4. **模拟器 + 自适应切换机制** 能大幅降低 profiling 开销，同时避免因模拟偏差导致失败。
5. **无需人工调度**：bandit-based orchestrator 可自动平衡 exploration 与 exploitation。

### **方法的局限性**
- 当前依赖轻量级线性模拟器，在极端非线性场景中可能预测不准。
- MCTS 初始化需要一定 warm-up（tournament phase），对极短预算任务略有负担。
- 尚未验证在超大规模集群（如千卡以上）上的可扩展性。

### **未来工作方向**
- 引入更强大的神经模拟器（neural surrogates）以提升预测精度。
- 扩展支持更多异构硬件（TPU、NPUs）与新型并行范式（如 zero-redundancy）。
- 探索跨任务迁移学习，复用历史优化经验以进一步加速冷启动。
- 集成进生产级 ML 平台（如 Kubernetes-based serving systems）实现实时自适应配置。

---

> ✅ **总结一句话**：  
> **AutoScout 通过结构化混合优化 + 自适应协调 + 多保真评估，在复杂 ML 系统配置空间中实现了高效、鲁棒、高性能的自动化调优，显著超越现有方法。**

</details>

---

### 5. [Efficient Generative Modeling with Unitary Matrix Product States Using Riemannian Optimization](https://arxiv.org/abs/2603.12026)

**Authors**: Haotong Duan, Zhongming Chen, Ngai Wong  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.12026v1  

#### Abstract
Tensor networks, which are originally developed for characterizing complex quantum many-body systems, have recently emerged as a powerful framework for capturing high-dimensional probability distributions with strong physical interpretability. This paper systematically studies matrix product states ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient Generative Modeling with Unitary Matrix Product States Using Riemannian Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的基于 **Matrix Product State (MPS)** 的生成模型在训练过程中存在以下问题：
- 参数更新时存在**全局缩放自由度（global scaling degrees of freedom）**，导致优化路径不唯一，梯度更新容易振荡或收敛缓慢。
- 使用标准欧氏梯度下降（Euclidean gradient descent）进行无约束优化时，由于归一化系数的相对变化对局部概率分布无影响，造成“平坦方向”（flat directions），降低训练效率和稳定性。

### 🚀 提出的新方法与创新思路
本文提出了一种新的生成建模框架：
- **Unitary MPS (UMPS)**：引入单位球面约束（unit-sphere constraint），强制 MPS 波函数满足 $\|\psi\|_F = 1$，从而消除冗余的全局缩放自由度。
- **Riemannian Optimization + Space-Decoupling Algorithm**：
  - 将优化问题转化为定义在流形上的约束优化问题（manifold-constrained optimization）。
  - 利用 **Riemannian 梯度下降** 在张量流形交集上进行参数更新，保持几何结构不变。
  - 引入 **space-decoupling 方法**，将低秩约束与单位范数约束解耦到两个独立空间中，实现高效并行优化。

### ⚖️ 相比现有方法的优势
| 方面 | 传统 MPS 方法 | 本文 UMPS-SD 方法 |
|------|----------------|--------------------|
| **训练稳定性** | 易出现边界振荡、更新不稳定 | 更新轨迹更直接，显著减少振荡 |
| **收敛速度** | 缓慢，需大量迭代才能收敛 | 快速适应数据结构，在少数循环内大幅下降 NLL |
| **计算效率** | 投影操作带来额外开销 | 避免显式投影，利用 retraction 自动维持约束 |
| **表达能力** | 保留 MPS 的高表达力 | 同样具备强表达能力，且更具可解释性 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
1. **Bars-and-Stripes (BAS) Dataset**
   - 图像尺寸：16×16 二值图像
   - 内容：所有可能的水平条纹和垂直条纹图案
   - 总样本数：131,070
   - 训练集大小 $|T|$：最多取 500 个样本用于小规模测试

2. **EMNIST Dataset**
   - 包括手写数字和字母（EMNIST-Letters 子集）
   - 图像尺寸：28×28 → 展平为 784 维向量
   - 用于评估模型在真实复杂数据上的泛化能力和重建质量

### ⚙️ 实验设置
- **预处理**：使用 MATLAB `reshape` 函数按列优先顺序展平图像为一维向量。
- **最大 bond dimension ($r_{\text{max}}$)**：控制模型容量，设为 200–500 不等。
- **学习率 (learning rate)**：初始设定为 0.007 或调整至 $1\times10^{-3}$ 等。
- **最大训练轮数 (lmax)**：通常设为 4–25 轮 sweep（从左到右再返回）。
- **初始化**：采用 right-canonical 形式初始化，确保初始归一化。

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Negative Log-Likelihood (NLL)** | 主要优化目标，衡量模型拟合经验分布的能力 |
| **Training Time / Computation Time** | 衡量算法效率，记录达到特定 NLL 所需时间 |
| **Sample Quality** | 可视化生成图像的质量，判断是否符合数据先验（如条纹连续性） |
| **Image Reconstruction** | 对部分遮蔽图像补全另一半，评估推断能力 |

### 🔁 基线方法对比
- **Baseline**: Han et al. [13] 提出的标准 MPS 生成模型（交替梯度下降 + 投影）
- **本文方法**: **UMPS-SD**（Unitary MPS + Riemannian Optimization + Space-Decoupling）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 在 **BAS 数据集** 上的结果（图 4–5）
- **快速收敛**：NLL 在前几轮迅速下降，loop=4 时已能生成清晰有效的条纹图像（见图 4b）。
- **稳定 bond dimension**：平均 bond dimension $r_{\text{mean}}$ 最终趋于稳定，不超过 $r_{\text{max}}$（表 II），说明低秩结构被有效保持。
- **数据规模 vs. 模型容量**（图 5a）：
  - 当 $|T| > r_{\text{max}}$ 时，NLL 明显上升 → 表明模型容量不足
  - 更大的 $r_{\text{max}}$ 可提升建模能力
- **单次更新耗时**（图 5b）：即使在 $r_{\text{max}}=400$ 下，每次更新仍可在约 350 秒内完成，具备实用性。

#### ✅ 在 **EMNIST 数据集** 上的结果（图 6–8，表 III–IV）
| 指标 | 结果 |
|------|------|
| **NLL 收敛速度** | UMPS-SD 在 **3 个 loop 内** 将 NLL 从 167.7 降至 **13.01**；而标准 MPS 需要 **25 个 loop** 才能达到 12.88 |
| **训练时间效率** | 达到相近精度时，UMPS-SD 比原方法快 **高达 27 倍**（表 III） |
| **不同学习率表现**（表 IV） | 多组学习率下均表现出更快下降趋势，验证鲁棒性 |

#### 🖼️ 生成与重建质量（图 7–9）
- **生成图像**（图 7）：
  - MPS 模型生成图像噪声多、结构模糊
  - UMPS-SD 生成图像边缘清晰、笔画连贯，保真度更高
- **图像补全任务**（图 8）：
  - 输入右半边像素（位置 393–784），预测左半边
  - UMPS-SD 成功恢复“4”、“5”的完整笔画，连接自然
  - MPS 出现断裂、扭曲甚至误识别（如“1”变成其他形状）
- **bond dimension 影响**（图 9）：
  - $r_{\text{max}} < |T|$ 时重建质量差
  - $r_{\text{max}} \geq 150$ 后改善趋于饱和，表明存在最优容量区间

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Unitary MPS 有效去除冗余自由度**，使优化集中在真正影响概率分布的方向上，极大提升了训练稳定性。
2. **Riemannian Optimization + Space-Decoupling 架构显著加速收敛**，相比传统交替梯度法效率提升达 **27×**。
3. **无需 MCMC 采样**：得益于 MPS 的规范形式，可通过逐位条件采样高效生成新样本（见图 2）。
4. **支持高效的推理与补全任务**：在图像修复中展现出强大先验建模能力，优于传统 MPS。

### ⚠️ 方法的局限性
1. **目前仅适用于二值化图像**：无法直接处理 RGB 彩色图像或多通道输入。
2. **受限于一维链式结构（chain structure）**：MPS 的纠缠结构较简单，难以捕捉二维图像中的长程依赖关系。
3. **高 bond dimension 带来内存压力**：虽然算法高效，但 $r_{\text{max}}$ 过大会增加存储和计算负担。

### 🔮 未来工作方向
1. **扩展至更高维张量网络**：
   - 探索 **Projected Entangled Pair States (PEPS)** 等二维结构以增强图像建模能力。
   - 结合 **Variational Monte Carlo (VMC)** 方法缓解 PEPS 优化复杂性。
2. **自适应学习率机制**：
   - 设计 Riemannian 版本的 **Adam / Adagrad**，在切空间维护梯度矩估计。
3. **随机优化与方差缩减**：
   - 引入 **stochastic mini-batch training** 并结合 **variance-reduction techniques** 提升大规模数据下的可扩展性。
4. **标准化 gauge fixing 策略**：
   - 分析 gauge 自由度对变分优化的影响，设计统一的固定策略以抑制虚假能量极小值。

---

> **源码公开**：作者已将代码发布于 GitHub  
> 🔗 https://github.com/haotong-Duan/UnitaryMPS-SpaceDecoupling

--- 

📌 **总结一句话**：  
本文通过引入 **Unitary MPS** 和 **Riemannian 流形优化**，构建了一个高效、稳定、快速收敛的生成模型框架，在多个基准数据集上实现了比传统 MPS 方法数量级级别的性能提升，为张量网络在机器学习中的应用提供了重要推进。

</details>

---

### 6. [Inverse Neural Operator for ODE Parameter Optimization](https://arxiv.org/abs/2603.11854)

**Authors**: Zhi-Song Liu, Wenqing Peng, Helmi Toropainen, Ammar Kheder, Andreas Rupp, Holger Froning, Xiaojie Lin, Michael Boy  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.11854v1  

#### Abstract
We propose the Inverse Neural Operator (INO), a two-stage framework for recovering hidden ODE parameters from sparse, partial observations. In Stage 1, a Conditional Fourier Neural Operator (C-FNO) with cross-attention learns a differentiable surrogate that reconstructs full ODE trajectories from ar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Inverse Neural Operator for ODE Parameter Optimization》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**从稀疏、部分观测中恢复常微分方程（ODE）隐藏参数**这一**病态逆问题**提出解决方案。在现实场景中，如大气化学建模或基因调控网络分析，通常面临以下挑战：
- **Observational Sparsity**：仅能获取少量离散时间点的观测；
- **Partial Observability**：系统状态不完全可观测；
- **Stiff Dynamics**：系统具有多尺度、刚性动态，导致传统梯度优化不稳定。

传统的基于梯度的方法（如 adjoint 方法）在这些条件下容易出现**梯度消失/爆炸、收敛慢、陷入局部最优**等问题。

---

### 提出的新方法与新思路
作者提出了 **Inverse Neural Operator (INO)** ——一种两阶段框架，将参数反演视为**摊销生成任务**（amortized generative task），而非逐实例优化问题。

#### 主要创新点包括：

1. **Conditional Fourier Neural Operator (C-FNO) + Cross-Attention**
   - 在第一阶段训练一个条件化的神经算子，输入为稀疏观测 $ u(t_{\text{rand}}) $ 和初始条件 $ u(t_0) $，以及假设的 ODE 参数 $ k $，输出为完整的 ODE 轨迹。
   - 引入 **affine feature modulation**（受 FiLM 启发）实现对 FNO 的参数条件控制；
   - 加入 **Cross-Attention** 模块作为频谱正则化器，抑制由截断 FFT 导致的高频伪影（Gibbs 现象），提升轨迹物理一致性。

2. **Amortized Drifting Model (ADM)**
   - 第二阶段冻结 C-FNO 作为前向代理（surrogate），训练一个无需雅可比（Jacobian-free）的漂移模型来学习参数空间中的速度场。
   - ADM 构造监督信号的方式是：基于多个样本的预测残差（residual）构建**核加权漂移场**（kernel-weighted drifting field）：
     $$
     v_{\text{drift}}(k_i) = \sum_j K_{ij} \|R_j\|^2 (k_j - k_i)
     $$
     其中 $ K_{ij} = \exp(-\|R_i - R_j\|^2 / \sigma^2) $ 衡量残差相似性。
   - 此过程完全避免了通过神经算子进行反向传播，从而规避了 stiff 系统中的 Jacobian 不稳定性。

3. **理论连接**
   - 将 ADM 的更新机制与 **mean-field interacting particle systems** 和 McKean-Vlasov 方程联系起来，证明其在适当条件下能收敛至真实参数分布。
   - 类比于 Stein Variational Gradient Descent (SVGD)，但 ADM 更偏向“共识优化”而非后验采样。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **稳定性** | 避免 Jacobian 计算，在 stiff 系统中更稳定 |
| **效率** | 推理仅需 ~0.23 秒，相比迭代 GD 快 **487×** |
| **精度** | 参数恢复 MAE 显著优于所有 baseline |
| **泛化能力** | 支持任意稀疏观测模式（训练时随机采样时间点） |

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **POLLU**  
   - 来自大气化学建模的经典 stiff ODE 系统（Verwer, 1994）
   - 包含 **20 种化学物种** 和 **25 个未知反应速率系数**（跨越多个数量级）
   - 动态高度非线性和刚性

2. **GRN (Gene Regulatory Network)**  
   - 合成的基因调控网络 ODE 模型
   - 形式：$ \frac{dx}{dt} = c + Kg(x) - \gamma x $
   - 固定基础转录率和衰减速率，估计 **40 个激活的调控系数**（稀疏先验）
   - 反应矩阵 $ K \in \mathbb{R}^{20\times20} $ 中主对角线及其邻近元素被激活

---

### 实验设置
- **数据生成**：
  - 使用 Latin Hypercube Sampling (LHS) 生成 50,000 训练样本，1,000 测试样本；
  - 每条轨迹模拟 100 个时间步（$ N=100 $）；
  - 输入仅提供 **M=3 个随机稀疏观测点**（远少于总时间步）；
  - ODE 参数归一化到 [0,1]，轨迹标准化。

- **模型训练**：
  - Stage 1 (C-FNO)：1000 epochs，lr=1e-3，batch size=32；
  - Stage 2 (ADM)：1000 epochs，lr=1e-4；
  - 所有实验在单张 NVIDIA V100 GPU 上完成。

- **评估指标**：
  - **Parameter Recovery**：MSE、MAE、Std（标准差）于恢复参数；
  - **Trajectory Fitting**：使用恢复参数代入 C-FNO 得到的轨迹与真值之间的 MSE、MAE；
  - **推理时间**：每样本 wall-clock 时间。

---

### 基线方法对比
分为三类共 8 种方法：

| 类别 | 方法 | 是否需要梯度 |
|-------|--------|-------------|
| **Gradient-based** | Gradient Descent, SGLD, MCMC | ✅ 是 |
| **Gradient-free** | CMAES | ❌ 否（黑箱查询） |
| **Inverse Operator** | iFNO, NIO, SPIN-ODE, Flow Matching | ❌ 否（摊销推理） |

> 所有方法共享相同的预训练 C-FNO 作为前向代理，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | POLLU 参数 MSE ↓ | GRN 参数 MSE ↓ | 推理时间 (s) |
|------|------------------|----------------|--------------|
| Gradient Descent | 0.0218 | 0.0092 | ~112 |
| CMAES | 0.0272 | 0.0189 | >100 |
| SPIN-ODE | 0.0794 | 0.0364 | ~0.2 |
| Flow Matching | 0.0300 | 0.0152 | 0.21 |
| **INO (Ours)** | **0.0117** | **0.0084** | **0.23** |

> ✅ **INO 在两个数据集上均取得最低参数误差（MSE）**  
> ⚡ **推理时间仅为 ~0.23s，相较 GD 实现 487× 加速**

---

### 与基线方法的对比结果
- **相比 Gradient Descent**：
  - 参数 MSE 下降 **46%**（0.0218 → 0.0117 on POLLU）；
  - 无需任何反向传播，稳定性更高；
  - 速度提升近三个数量级。

- **相比其他摊销方法（如 iFNO, NIO, SPIN-ODE）**：
  - 这些方法直接回归参数，目标是最小化轨迹误差，导致“轨迹拟合好但参数不准”；
  - INO 显式优化参数准确性，实现了更好的逆映射保真度。

- **相比 Flow Matching**：
  - Flow Matching 仍依赖 $ \nabla_k \mathcal{L} $ 构造监督信号，存在 Jacobian 不稳定风险；
  - ADM 完全无 Jacobian，且性能高出 **2.6×**（0.0300 → 0.0117）。

---

### 消融实验结果（Ablation Study）

#### （1）C-FNO 架构消融（Table 2）
| 配置 | POLLU MSE ↓ |
|------|------------|
| Baseline FNO | 0.1561 |
| + C-FNO (affine mod) | 0.0799 |
| + Cross-Attention | 0.0715 |
| + Random Observation Sampling | 0.0694 |
| **Full Model (C-FNO + Attn + Rand)** | **0.0559** |

> ✅ Cross-Attention 显著抑制高频振荡，提升轨迹平滑性（见 Fig. 6a）  
> ✅ 随机采样观测增强泛化能力

#### （2）ADM 消融（Table 3）
| 方法 | POLLU MSE ↓ | 时间 (s) |
|------|------------|---------|
| MLP Regression | 0.1023 | 0.05 |
| GD-SGD (100 iter) | 0.0218 | 112 |
| FM-Grad（带梯度监督） | 0.0300 | 0.21 |
| **ADM (Ours)** | **0.0117** | 0.23 |

> ✅ 即使架构相同，**Jacobian-free 的监督方式本身带来显著增益**（0.0300 → 0.0117）  
> ✅ 摊销推理可在极短时间内达到高精度

---

## 4. 关键结论和发现

### 主要发现
1. **摊销式逆建模可行且高效**：INO 成功将 ODE 参数反演转化为快速前向推理任务，适用于实时或大规模应用场景。
2. **Cross-Attention 是有效的频谱正则化手段**：有效缓解 FNO 在稀疏数据下的 Gibbs 现象，提高轨迹物理合理性。
3. **Jacobian-free 学习显著提升稳定性**：ADM 通过残差空间的核耦合驱动参数演化，绕开敏感梯度计算，在 stiff 系统中表现卓越。
4. **轨迹拟合 ≠ 参数准确**：许多 inverse operator 方法虽能重建轨迹，但参数估计偏差大；INO 显式优化参数空间，实现双重保真。

---

### 方法的局限性
- 当前验证集中在两个 benchmark 上（POLLU, GRN），尚未扩展到真实世界复杂实验数据；
- 假设观测时间点可对齐，未处理不规则采样（irregular sampling）；
- 对噪声建模有限，异方差噪声（heteroscedastic noise）和部分物种不可观测情形有待研究；
- ADM 依赖 mini-batch 内样本交互，可能影响极端低资源场景下的性能。

---

### 未来工作方向
1. **拓展至 PDE 场景**：结合 spatiotemporal neural operators 处理偏微分方程逆问题；
2. **集成不确定性量化**：引入贝叶斯视角，提供参数估计的置信区间；
3. **应用于真实科学实验**：如质谱数据分析、气候观测反演等实际测量受限的任务；
4. **轻量化部署**：进一步压缩 ADM 模型以支持边缘设备运行。

---

> ✅ **总结一句话**：  
> **INO 通过“条件神经算子 + Jacobian-free 漂移模型”的设计，在保持高参数恢复精度的同时，实现了超过 487 倍的速度提升，为 stiff ODE 系统的高效逆建模提供了新范式。**

</details>

---

### 7. [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201)

**Authors**: Yushi Bai, Qian Dong, Ting Jiang, Xin Lv, Zhengxiao Du, Aohan Zeng, Jie Tang, Juanzi Li  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.12201v1  

#### Abstract
Long-context agentic workflows have emerged as a defining use case for large language models, making attention efficiency critical for both inference speed and serving cost. Sparse attention addresses this challenge effectively, and DeepSeek Sparse Attention (DSA) is a representative production-grad...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
在长上下文（long-context）场景中，大型语言模型（LLMs）的推理效率受到 **self-attention** 机制的 $O(L^2)$ 复杂度限制。虽然 **DeepSeek Sparse Attention (DSA)** 通过引入轻量级的 **lightning indexer** 将核心 attention 从 $O(L^2)$ 降低到 $O(Lk)$，但该 indexer 本身仍需在每一层独立运行，其计算复杂度为 $O(NL^2)$（$N$ 为层数），成为长序列推理中的主要瓶颈。

此外，实验证明不同层之间的 **top-k token selection** 高度相似，表明大量 indexer 计算是冗余的。

### **提出了什么新方法或新思路**
提出 **IndexCache**，一种通过跨层索引复用（cross-layer index reuse）来加速稀疏注意力的方法：

- 将 Transformer 层划分为两类：
  - **F (Full) 层**：保留 lightning indexer，独立计算 top-k 索引。
  - **S (Shared) 层**：不运行 indexer，直接复用最近前一个 F 层的 top-k 索引。
- 引入两种互补策略优化配置：
  - **Training-free IndexCache**：无需重新训练，基于校准集上的语言建模损失（LM loss）使用贪心搜索选择最优 F 层位置。
  - **Training-aware IndexCache**：在训练阶段引入 **multi-layer distillation loss**，使每个保留的 indexer 学习服务于多个后续 S 层，从而支持更简单的共享模式（如均匀间隔）也能保持高性能。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **效率提升** | 最多可移除 75% 的 indexer 计算，显著减少 prefill 和 decode 时间。 |
| **无额外内存开销** | 仅增加一个条件分支，缓存的索引张量可被覆盖，无需额外 GPU 显存。 |
| **通用性强** | 可应用于任何动态 token selection 的稀疏 attention 架构（如 MoBA、NSA）。 |
| **灵活性高** | 支持训练即插即用（training-free）与端到端优化（training-aware）两种模式。 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
#### **长上下文任务（Long-Context Benchmarks）**
- **MRCR v2**：多轮指代消解（Multi-round Coreference Resolution）
- **GraphWalks**：图结构推理任务
- **LongBench v2**：现实长文本理解与推理
- **RULER**：评估真实上下文长度能力
- **AA-LCR**：人工分析机构发布的长上下文推理评测

#### **通用与推理任务（General & Reasoning Benchmarks）**
- **AIME 2025**：数学竞赛题推理
- **GPQA-Diamond**：研究生级别问答
- **LiveCodeBench v6**：代码生成评测
- **IFBench v2**：指令遵循能力测试

### **实验设置和评估指标**
- **模型**：
  - 主要实验：30B 参数的 DSA 模型（基于 GLM-4.7-Flash）
  - 扩展实验：744B 参数的 GLM-5 模型（40B active）
- **上下文长度**：10K, 60K, 120K, 200K tokens
- **评估指标**：
  - **Prefill time (s)**：首 token 延迟
  - **Decode throughput (tok/s)**：每请求解码速度 & 全 KV cache 利用下的总吞吐
  - **Benchmark Score (%)**：各任务平均得分
- **实现平台**：SGLang + dp_attention（dp_size=8），运行于 NVIDIA H100 节点

### **基线方法对比**
- **Baseline**：标准 DSA（每层都运行 indexer）
- **Uniform Interleaving**：每隔若干层保留一个 indexer（如 FSSS...）
- **IndexCache (1/2)**：保留一半 indexer
- **IndexCache (1/4)**：保留四分之一 indexer
- 对比方式包括：
  - 性能下降程度（benchmark 分数）
  - 推理加速比（speedup）

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

| 指标 | 上下文长度 | DSA 基线 | IndexCache (1/4) | 加速比 |
|------|-----------|----------|------------------|--------|
| **Prefill Time (s)** | 200K | 19.5 | 10.7 | **1.82×** |
| **Decode Throughput (per request, tok/s)** | 200K | 58.0 | 86.0 | **1.48×** |
| **Decode Throughput (full KV, tok/s)** | 200K | 197 | 297 | **1.51×** |

> 在 10K 上下文中也实现了 1.27× prefill 加速，说明即使短序列也有收益。

### **与基线方法的对比结果**

#### ✅ **Training-free IndexCache（贪心搜索模式）**
- 在保留 **1/4 indexer** 的情况下：
  - 长上下文任务平均分数从原始 DSA 的 **50.2 → 49.9**
  - 显著优于均匀交错（Uniform Interleaving）的 **43.0**
- 表明“**哪些层保留 indexer”比“保留多少”更重要**

#### ✅ **Training-aware IndexCache（带多层蒸馏损失）**
- 即使采用简单 **uniform interleaving (1/2 或 1/4)**：
  - 长上下文平均得分与完整 DSA 几乎持平（51.6 vs. 51.0）
  - 一般推理任务也保持一致（~74.5）
- 移除 multi-layer distillation 后性能明显下降（Long Avg 降至 49.8），证明该损失函数有效。

#### ✅ **在 GLM-5 上的扩展实验**
| 方法 | Long Avg | 相对原模型 | End-to-End Speedup |
|------|---------|------------|--------------------|
| Original DSA | 78.4 | — | 1.0× |
| IndexCache (1/2) + Searched | 78.7 | ↑ | **~1.2×** |
| IndexCache (1/4) + Searched | 78.0 | ≈ | — |

> 表明 IndexCache 在千亿参数规模上依然有效且稳定。

### **消融实验结果**

| 实验设置 | Long Avg | G&R Avg | 说明 |
|--------|--------|--------|------|
| Uniform Interleaving (1/2) | 47.4 | 74.3 | 明显退化 |
| Greedy Searched (1/2) | 50.3 | 74.4 | 恢复至接近原模型 |
| Uniform Interleaving (1/4) | 43.0 | 73.8 | 严重退化 |
| Greedy Searched (1/4) | 49.9 | 74.9 | 几乎无损 |
| Remove Cross-layer Loss | 49.8 | 74.5 | 证明 multi-layer distillation 必要 |

> 结论：**局部相似性（如 attention 输出余弦相似度）不能作为共享模式选择的有效代理指标**；必须使用全局指标（如 LM loss）进行优化。

---

## 4. **关键结论和发现**

### **主要发现**
1. 🔹 **indexer 的输出具有高度跨层稳定性**：
   - 相邻层间 top-k 重叠率达 70–100%
   - 存在明显的功能块结构（block-wise clustering）

2. 🔹 **大多数 indexer 是冗余的**：
   - 最多可移除 75% 的 indexer 而不造成显著质量损失

3. 🔹 **“保留哪几层”比“保留多少”更重要**：
   - 贪心搜索能识别出关键敏感层（early/transitional layers）
   - 均匀交错会破坏这些关键层导致性能骤降

4. 🔹 **训练感知设计可消除模式敏感性**：
   - 引入 multi-layer distillation 后，即使是 uniform pattern 也能达到 full-indexer 性能
   - 说明模型可通过训练适应 index sharing

5. 🔹 **IndexCache 是轻量高效的系统级优化**：
   - 仅需一次条件判断，无额外显存开销
   - 可无缝集成进现有 DSA 推理流程

### **方法的局限性**
- 当 indexer 保留比例极低（如 1/8）时，即便使用贪心搜索，性能仍显著下降（Long Avg 从 50.2 → 46.1）
- 当前方法依赖于已有 indexer 输出，无法进一步压缩 indexer 自身结构
- 在某些极端推理路径中可能存在未被覆盖的关键 token 丢失风险

### **未来工作方向**
- 将 IndexCache 应用于其他稀疏 attention 架构（如 MoBA、NSA）
- 探索动态自适应的 sharing pattern（根据输入内容调整 F/S 层分布）
- 结合 KV Cache 压缩技术（如 OmniKV、SwiftKV）实现联合优化
- 在更大规模生产模型（如 GLM-5）上部署并验证训练感知版本的实际收益

---

> 💡 **一句话总结**：  
> **IndexCache 揭示了稀疏 attention 中 indexer 的跨层冗余性，并通过智能索引复用，在几乎不损失性能的前提下，将 DSA 的推理速度提升高达 1.82×，为下一代长上下文 LLM 推理系统提供了高效实用的新范式。**

</details>

---

### 8. [Learning Transferable Sensor Models via Language-Informed Pretraining](https://arxiv.org/abs/2603.11950)

**Authors**: Yuliang Chen, Arvind Pillai, Yu Yvonne Wu, Tess Z. Griffin, Lisa Marsch, Michael V. Heinz, Nicholas C. Jacobson, Andrew Campbell  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.11950v1  

#### Abstract
Modern sensing systems generate large volumes of unlabeled multivariate time-series data. This abundance of unlabeled data makes self-supervised learning (SSL) a natural approach for learning transferable representations. However, most existing approaches are optimized for reconstruction or forecast...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning Transferable Sensor Models via Language-Informed Pretraining

## 1. 论文的主要贡献和创新点

### 解决的问题
现代传感系统（sensing systems）产生大量无标签的多变量时间序列（multivariate time-series）数据。虽然自监督学习（SSL）被广泛用于学习可迁移表示，但大多数现有方法专注于**重构（reconstruction）或预测（forecasting）目标**，导致学到的表征缺乏对下游分类和推理任务至关重要的**语义结构（semantic structure）**。

此外，尽管近期出现了一些传感器-语言对齐（sensor-language alignment）方法以提升语义泛化能力，但这些方法通常受限于**固定的传感器配置**（如预定义通道数、信号长度或时间分辨率），限制了其在跨域场景中的适用性。

### 提出的新方法：SLIP
为解决上述问题，本文提出了 **SLIP (Sensor Language-Informed Pretraining)**，一个开源的框架，旨在学习能够泛化到多样化传感器设置的语言对齐表示。

#### 核心创新点：
- **统一的对比对齐与条件描述生成**：  
  SLIP 结合了**对比对齐（contrastive alignment）** 和**传感器条件下的文本描述生成（sensor-conditioned captioning）**，同时支持判别性理解（discriminative understanding）和生成式推理（generative reasoning）。

- **灵活的 Patch Embedder (FlexMLP)**：  
  引入了一种轻量级、参数共享的 **FlexMLP** 机制，使模型能够在推理时动态适应不同的时间分辨率和可变长度输入，而无需重新训练。

- **重用解码器优先语言模型**：  
  将一个预训练的仅解码器（decoder-only）语言模型（如 Gemma）通过交叉注意力（cross-attention）改造为编码器-解码器架构，实现了高效的训练，并支持开放词汇的生成。

### 相比现有方法的优势
| 特性 | SLIP | 其他方法（如 Chronos, SensorLM） |
|------|------|-------------------------------|
| 时间分辨率自适应 | ✅ 支持 | ❌ 固定分辨率 |
| 可变长度输入 | ✅ 支持 | ❌ 需固定长度 |
| 开放词汇生成 | ✅ 支持 | ❌ 有限或不支持 |
| 跨模态检索 | ✅ 支持 | ⚠️ 部分支持 |
| 问答能力 | ✅ 支持 | ❌ 不支持 |

---

## 2. 核心实验方法和设置

### 数据集
#### 预训练数据集
- 构建了一个包含 **60万组传感器-文本对** 的大规模数据集，涵盖超过 **10亿个时间点**。
- 数据来源广泛，覆盖健康、环境、物联网（IoT）、能源、交通等多个领域。
- 文本描述由 LLM 自动生成，包含统计、结构和语义三个层次的信息。

#### 下游评估数据集（共11个）
| 类别 | 数据集 |
|------|-------|
| **活动识别** | WISDM, UCI-HAR |
| **临床诊断** | Stroke, Diabetes, Hypertension, Sleep Stage, Heart Condition (PTB-XL) |
| **压力预测** | WESAD, StudentLife |
| **城市感知** | Obstacles, BeijingAQI |

此外还使用了：
- **问答数据集**：HAR-CoT, Sleep-CoT, ECG-QA-CoT, TSQA（多项选择题）
- **描述生成数据集**：M4

### 实验设置与评估指标

#### 主要任务与评估方式：
| 任务 | 评估方式 | 指标 |
|------|----------|------|
| **传感器分类** | Linear probing（冻结特征+训练线性分类器） | Top-1 Accuracy |
| **零样本分类** | Sensor-Text Retrieval（基于余弦相似度最近邻匹配） | Top-1 Accuracy (R@1) |
| **传感器问答（QA）** | 监督微调（SFT）后生成答案 | Accuracy |
| **传感器描述生成（Captioning）** | 微调后生成自然语言描述 | BLEU@4, METEOR, ROUGE-L, BERTScore, SBERTSimilarity |

#### 基线方法对比
- **传统方法**：Statistical ML
- **自监督学习**：SimMTM, TF-C
- **通用时间序列基础模型**：Chronos, Chronos2, Sundial
- **结合语言监督的方法**：Normwear, ChatTS, OpenTSLM
- **上界（监督训练）**：PatchTST

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | SLIP 表现 | 对比基线 |
|------|---------|--------|
| **平均线性探针准确率（Linear Probing）** | **77.14%** | Normwear: 72.82%（相对提升 **+5.93%**） |
| **零样本分类平均准确率** | **39.42%** | Normwear: 30.42% |
| **传感器问答平均准确率（SFT）** | **64.83%** | OpenTSLM-Flamingo: ~50% |
| **描述生成质量（BERTScore）** | **0.887** | OpenTSLM: 0.8858 |

> ✅ SLIP 在所有任务中均达到 SOTA 性能。

### 与基线方法的对比结果
- 在 **11个下游任务** 上，SLIPBase 的线性探针性能全面超越所有基线，甚至接近全监督训练的 PatchTST（76.2%）。
- 在 **零样本设置下**，SLIP 使用的推理 token 数仅为 LLM 基线的 **1% 左右**（平均 300 vs. 37k），效率极高。
- 在 **问答任务** 中，SLIPsFT 仅需 **4轮微调** 即大幅超越 OpenTSLM 的多阶段复杂训练流程。

### 消融实验结果（Ablation Studies）

| 消融设置 | 分类 | 零样本 | 问答 |
|--------|------|--------|-----|
| **完整 SLIP** | 77.14 | 39.36 | 64.83 |
| **仅对比损失（Contrastive-only）** | ↓2.37 | ↓2.46 | ↓9.98 |
| **仅描述损失（Caption-only）** | ↓2.84 | ↓14.24 | ↓7.57 |
| **随机配对（Random paired）** | ↓14.99 | ↓16.98 | ↓29.75 |
| **无 FlexMLP（固定 patch size=16）** | ↓2.98 | ↓4.42 | ↓3.45 |
| **冻结文本编码器** | ↓3.58 | ↓3.68 | ↓7.74 |
| **小模型（40M 参数）** | ↓2.30 | ↓12.65 | ↓10.84 |

> 🔍 结论：
> - **双目标联合训练** 至关重要，缺一不可。
> - **FlexMLP** 显著提升灵活性与性能。
> - **文本编码器微调** 是实现良好对齐的关键。
> - 更大的传感器编码器有助于跨模态任务。

---

## 4. 关键结论和发现

### 主要发现
1. **语义对齐优于纯重建目标**：  
   仅优化重构或预测目标的模型（如 Chronos2）虽能准确预测信号，但其表征无法支持下游分类（见 Figure 1 示例），存在“预测-分类鸿沟”（forecasting-classification gap）。

2. **双目标协同作用显著**：  
   对比学习确保全局语义对齐，而描述生成提供细粒度监督，二者结合显著提升表征质量。

3. **FlexMLP 实现真正的泛化能力**：  
   支持不同采样频率和序列长度的能力使得 SLIP 可直接应用于多样化的现实世界传感器部署。

4. **高效且强大的零样本迁移能力**：  
   SLIP 在极低计算开销下实现了优异的零样本性能，尤其在行为模式丰富的任务（如压力检测）中表现突出。

### 方法的局限性
1. **语言模型主干固定**：  
   本文未探索其他 LLM 主干的影响，可能限制进一步优化空间。

2. **长上下文成本高**：  
   由于依赖自由形式文本描述而非短标签，预训练时的上下文长度和计算成本较高。

3. **未分析幻觉与忠实性**：  
   缺乏对生成内容是否忠实反映原始信号的归因分析（attribution analysis），存在潜在幻觉风险。

4. **特定领域表现受限**：  
   在高度专业化任务（如临床诊断）上，专用模型（如 Normwear）因更匹配的数据分布仍具优势。

### 未来工作方向
- 探索更高效的解码策略（如 selective decoding）以降低长文本生成成本。
- 引入归因机制以增强输出的可信度与可解释性。
- 扩展至更多模态（如音频、视频）构建多模态传感器语言模型。
- 研究动态调整语言模型主干的可能性，实现端到端优化。

---

> 📌 **总结**：  
> SLIP 成功地将 **CoCa** 的思想扩展到传感器领域，提出了一种**统一、灵活、语义丰富**的传感器语言预训练范式。它不仅在多个任务上取得 SOTA 结果，更重要的是为构建下一代通用传感器基础模型提供了新的设计原则和开源资源（代码与数据已公开）。

</details>

---

### 9. [A Robust and Efficient Multi-Agent Reinforcement Learning Framework for Traffic Signal Control](https://arxiv.org/abs/2603.12096)

**Authors**: Sheng-You Huang, Hsiao-Chuan Chang, Yen-Chi Chen, Ting-Han Wei, I-Hau Yeh, Sheng-Yao Kuan, Chien-Yao Wang, Hsuan-Han Lee, I-Chen Wu  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.12096v1  

#### Abstract
Reinforcement Learning (RL) in Traffic Signal Control (TSC) faces significant hurdles in real-world deployment due to limited generalization to dynamic traffic flow variations. Existing approaches often overfit static patterns and use action spaces incompatible with driver expectations. This paper p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Robust and Efficient Multi-Agent Reinforcement Learning Framework for Traffic Signal Control

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**深度强化学习（DRL）在交通信号控制（TSC）中实际部署面临的三大挑战**：
- **环境泛化能力差**：现有 RL 模型容易过拟合静态交通模式，在动态变化的真实交通流下表现不佳。
- **动作空间设计不安全或不稳定**：传统动作空间难以平衡响应速度与控制稳定性，可能违反驾驶员预期的相位顺序。
- **系统可扩展性不足**：集中式观测虽有效但不可扩展；局部观测则缺乏协调能力。

### 提出的新方法与创新
作者提出了一种**鲁棒且高效的 Multi-Agent Reinforcement Learning (MARL) 框架**，包含三个核心技术贡献：

#### ✅ 创新点 1：Turning Ratio Randomization（转向比例随机化）
- 在每个训练 episode 开始时，对各转向流量的比例施加乘性噪声并重新归一化。
- **目的**：增强 agent 对非平稳交通分布的鲁棒性，防止其记忆固定调度策略（open-loop behavior），促使其基于状态进行决策。
- **优势**：相比仅改变车流量（volume），此方法避免引入奖励漂移（reward instability），更专注于学习绿灯分配逻辑。

#### ✅ 创新点 2：Exponential Phase Duration Adjustment（指数级相位持续时间调整）
- 设计一种“粗到细”（coarse-to-fine）的动作空间：  
  动作集为 $\Delta t \in \{0, \pm\lambda^0, \pm\lambda^1, \pm\lambda^2, \pm\lambda^3\}$（如 $\lambda=2$ 时为 $\{0,\pm1,\pm2,\pm4,\pm8\}$ 秒）。
- 在周期结束前微调下一相位的绿灯时长。
- **优势**：
  - 大步长应对突发拥堵（快速反应）；
  - 小步长维持稳态精度（减少震荡）；
  - 符合 cyclic phase 安全要求，保持黄灯/全红灯不变。

#### ✅ 创新点 3：Neighbor-Based Observation with CTDE 架构
- 采用 Centralized Training with Decentralized Execution (CTDE) 范式，结合 Multi-Agent PPO (MAPPO) 算法。
- 观测范围限定为 **本地 + 直接上下游邻居交叉口的信息**（neighbor-level observation）。
- **优势**：
  - 训练阶段利用全局信息优化 critic，提升信用分配准确性；
  - 执行阶段仅依赖局部通信，具备良好可扩展性；
  - 实现接近 global observation 的协调效果，远优于 pure local 方案。

---

## 2. 核心实验方法和设置

### 使用的数据集与仿真环境
- **仿真平台**：PTV Vissim —— 工业界标准微观交通模拟器，采用 Wiedemann car-following model，能真实反映人类驾驶行为。
- **路网结构**：台湾桃园市中正东路的五个连续信号交叉口数字孪生模型（短间距、高交互性）。
- **真实交通数据来源**：基于实际检测器采集的24小时车流数据，提取高峰与平峰两种场景。

### 实验设置
- **训练与测试分离**：
  - **训练仅使用高峰时段数据**（~4800 vehs/hr），以高压条件驱动策略优化；
  - **测试涵盖高峰和平峰**（~1800 vehs/hr），检验泛化能力。
- **车辆类型**：仅考虑四轮机动车。

### 评估指标
遵循最新 TSC 综述 [23] 推荐的标准指标：
| 指标 | 含义 |
|------|------|
| **ATT** (Average Travel Time) ↓ | 平均行程时间（秒/车） |
| **AWT** (Average Waiting Time) ↓ | 平均等待时间（秒/车） |
| **AD** (Average Delay) ↓ | 平均延误（秒/车） |
| **VC** (Vehicle Count) ↑ | 单位时间内通过车辆数（辆/小时） |

### 基线方法对比
| 类别 | 方法 |
|------|------|
| 传统方法 | FixTime（绿波优化定时方案）、MaxPressure（经典启发式算法） |
| 标准 RL 方法 | 不同观测范围下的 $M_{\text{static}}^{local}, M_{\text{static}}^{neighbor}, M_{\text{static}}^{global}$（静态转向比训练） |
| 本文方法 | $M_{\text{randomized}}^{local}, M_{\text{randomized}}^{neighbor}, M_{\text{randomized}}^{global}$（引入转向随机化） |

此外还进行了消融实验，比较：
- **是否使用 CTDE**：MAPPO vs IPPO
- **不同动作空间设计**：Exponential vs Linear Adjustment

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| 方法 | 场景 | ATT (s) | AWT (s) | AD (s) | VC (vehs/h) |
|------|------|--------|--------|-------|------------|
| FixTime | 高峰 | 383.92 | 352.87 | 319.04 | 4015.87 |
| MaxPressure | 高峰 | 265.79 | 285.93 | 196.54 | 4223.80 |
| $M_{\text{randomized}}^{global}$ | 高峰 | **230.58** | **231.01** | **160.34** | **4416.53** |
| $M_{\text{randomized}}^{neighbor}$ | 高峰 | 230.58 | 231.01 | 160.34 | 4416.53 |
| $M_{\text{randomized}}^{global}$ | 平峰 | **119.32** | **36.12** | **48.33** | **1802.80** |
| $M_{\text{randomized}}^{neighbor}$ | 平峰 | 124.37 | 44.09 | 53.44 | 1808.47 |

> 注：“↓”表示越小越好，“↑”表示越大越好。

### 与基线方法的关键对比结果
- 在**高峰场景**中，所提方法（$M_{\text{randomized}}^{global/neighbor}$）将 ATT 从 MaxPressure 的 265.79s 进一步降低至 **230.58s**（下降约 13.2%），同时提高通行量（VC）。
- 在更具挑战性的**平峰场景**（未见于训练）中：
  - 标准 RL 方法（$M_{\text{static}}$）性能严重退化，甚至不如 MaxPressure；
  - 所提方法仍保持优异表现，$M_{\text{randomized}}^{neighbor}$ 的 ATT 达到 **124.37s**，优于 MaxPressure 的 126.57s；
  - 表明 **Turning Ratio Randomization 显著提升了泛化能力和鲁棒性**。

### 消融实验结果

#### ✅ CTDE vs Non-CTDE（Table 3）
| 方法 | ATT (高峰) | ATT (平峰) |
|------|-----------|-----------|
| IPPO (non-CTDE) | 298.43s | 134.20s |
| MAPPO (CTDE, ours) | **230.58s** | **124.37s** |

👉 结论：CTDE 架构显著提升性能，尤其在非平稳环境中稳定信用分配，促进多智能体协作。

#### ✅ 动作空间设计对比（Table 4）
| 动作空间 | 类型 | ATT (高峰) | ATT (平峰) |
|--------|-----|-----------|-----------|
| Linear {0,±2,±4,±6,±8} | 小尺度 | 263.11s | 158.10s ❌ |
| Linear {0,±5,±10,±15,±20} | 大尺度 | 283.56s | 144.96s ❌ |
| Exponential {0,±1,±2,±4,±8} ($\lambda=2$) | ours | **230.58s** | **124.37s** ✅ |

👉 结论：指数型动作空间在所有场景下均最优，实现了“大扰动快速响应 + 小扰动精细调节”的双重优势。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Turning Ratio Randomization 是提升泛化性的关键正则化手段**：  
   能有效防止 agent 过拟合静态模式，在未见过的低负载场景中依然表现稳健。
   
2. ✅ **Exponential Phase Duration Adjustment 实现了稳定性与响应性的统一**：  
   相比线性调整，指数动作空间更适合处理动态交通需求波动。

3. ✅ **Neighbor-Level Observation + CTDE 可实现近似全局协调的高性能控制**：  
   在无需全局信息输入的前提下，通过 centralized critic 学习协同策略，解决了 scalability 与 optimality 的矛盾。

4. ✅ 所提框架在 Vissim 高保真仿真中验证有效，平均等待时间（AWT）降低 **超过 10%**，具备向现实世界迁移的潜力。

### 方法的局限性
- 当前实验局限于一条 arterial road 上的线性交叉口群，尚未扩展至复杂 grid network。
- 转向比例扰动假设独立于车流量，未来可探索联合扰动机制。
- 未考虑行人、非机动车等多模态交通参与者。

### 未来工作方向
- 将框架推广至大规模网格状路网（grid networks）；
- 引入 multi-modal traffic data（如公交、骑行、行人）进行综合调控；
- 探索在线自适应机制，实现持续学习与策略更新；
- 推进 real-world deployment，开展实地试点测试。

--- 

> 📌 总结一句话：  
> 本论文通过 **Turning Ratio Randomization + Exponential Action Space + Neighbor-based CTDE** 三重机制，构建了一个**鲁棒、高效、可扩展**的 MARL-TSC 框架，在高保真 Vissim 仿真中展现出卓越的泛化能力与控制稳定性，为 DRL 在真实交通系统中的落地提供了可行路径。

</details>

---

### 10. [LLM-Assisted Causal Structure Disambiguation and Factor Extraction for Legal Judgment Prediction](https://arxiv.org/abs/2603.11446)

**Authors**: Yuzhi Liang, Lixiang Ma, Xinrong Zhu  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.11446v1  

#### Abstract
Mainstream methods for Legal Judgment Prediction (LJP) based on Pre-trained Language Models (PLMs) heavily rely on the statistical correlation between case facts and judgment results. This paradigm lacks explicit modeling of legal constituent elements and underlying causal logic, making models prone...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM-Assisted Causal Structure Disambiguation and Factor Extraction for Legal Judgment Prediction

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前主流的 **Legal Judgment Prediction (LJP)** 方法基于 **Pre-trained Language Models (PLMs)**，严重依赖案例事实与判决结果之间的**统计相关性**，缺乏对法律构成要素（legal constituent elements）及其内在**因果逻辑**的显式建模。这导致模型容易学习到虚假相关性（spurious correlations），在面对文本扰动时鲁棒性差。

此外，现有的因果推理方法在真实法律文本中面临两大瓶颈：
1. **法律因素提取不准确**：传统关键词提取方法（如YAKE）或通用信息抽取工具难以区分实质性法律要素与高频叙事成分（如人名、地名），引入大量噪声。
2. **因果结构不确定性高**：由于法律特征稀疏且存在 **Markov equivalence**，传统的因果发现算法（如GFCI）只能输出部分祖先图（PAG），其中许多边的方向是模糊的。

---

### 提出的新方法与新思路

本文提出一个融合 **Large Language Model (LLM) prior knowledge** 与**统计因果发现**的增强型因果推理框架，实现从非结构化案情描述到因果感知判决预测的闭环流程。其核心创新包括：

#### （1）**Coarse-to-Fine 混合法律因素提取机制**
- **粗筛阶段**：使用 **YAKE** 算法进行无监督关键词提取，并通过均匀采样保证长尾关键证据的覆盖。
- **精炼阶段**：结合 **Retrieval-Augmented Generation (RAG)** 和 **Chain-of-Thought (CoT)** 推理提示，引导LLM将候选词映射为标准法律要素，并过滤掉非要素噪声（如 PERSON、GPE、DATE）。
- 最终构建语义纯净、符合法学逻辑的因果变量空间。

#### （2）**LLM辅助的因果结构消歧机制**
- 针对GFCI输出的PAG中方向不确定的边（如 `u o-o v`），利用LLM作为“软法律知识库”进行概率评估。
- 设计结构化提示模板，输入节点语义、相关法律条文和案例上下文，让LLM判断三种可能关系的概率分布：`u → v`, `v → u`, 或 `u ↔ v`。
- 结合司法逻辑约束（如“事实决定判决”，禁止 Y→X）和时间先后约束（cause precedes effect），对图结构进行剪枝。

#### （3）**基于因果图的注意力引导预测模型**
- 利用多图采样与 **Propensity Score Matching (PSM)** 估计每条边的因果强度（ATE）。
- 将多个因果图的好坏拟合度（BIC得分）作为权重，加权聚合得到整体因果强度。
- 在预训练语言模型（ALBERT）中，以该因果强度为先验，约束自注意力机制中的注意力权重，使模型更关注具有强因果效应的词。

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **准确性** | 显著优于主流PLM和因果增强模型，在多个任务上达到SOTA |
| **鲁棒性** | 对文本扰动（如标点变化）更具稳定性，避免虚假相关 |
| **可解释性** | 显式建模法律要素间的因果路径，提升决策透明度 |
| **低资源适应性** | 在few-shot场景下表现优异，数据效率更高 |
| **结构可靠性** | 有效缓解Markov等价带来的方向不确定性问题 |

---

## 2. 核心实验方法和设置

### 使用的数据集

共使用 **5个基准数据集**，涵盖中英文、不同任务类型：

| 数据集 | 类型 | 语言 | 规模 | 特点 |
|--------|------|-------|------|------|
| **LEVEN** | 法律事件检测 | 中文 | 8,116 条 | 包含108类事件，最大中文法律事件数据集 |
| **QA** | 法律咨询分类 | 中文 | 203,459 条 | 47类法律问题分类 |
| **CAIL2018** | 刑事罪名预测 | 中文 | 30,183 条 | 聚焦易混淆罪名区分（如诈骗 vs 敲诈勒索） |
| **LEDGAR** | 合同条款分类 | 英文 | 80,000 条 | 来自SEC文件，每标签对应合同子句主题 |
| **Overruling** | 先例推翻检测 | 英文 | 2,400 条 | 检测是否明确推翻先前判例 |

---

### 实验设置与评估指标

- **任务形式**：多类别文本分类（罪名预测、事件识别等）
- **主指标**：**Accuracy**（准确率）
- **训练比例控制**：测试了从 **1% 到 100%** 不同训练数据比例下的性能，验证低资源能力
- **因果图生成**：采样 Q=100 个因果图，使用 BIC 加权集成
- **LLM 使用方式**：用于法律因素筛选与因果方向评分，**不参与最终预测**

---

### 基线方法对比

共比较了 **17种基线模型**，分为四类：

#### （1）传统与深度学习模型
- BiLSTM, BiLSTM+CRF
- NPC（基于压缩距离的零参数方法）

#### （2）主流PLM模型
- BERT, BERT+CRF
- XLM-RoBERTa, Legal-RoBERTa, InLegalBERT

#### （3）大模型嵌入与原型学习
- LLMEmbed（轻量LLM多层嵌入）
- ProtoLens（原型学习）

#### （4）因果增强模型
- AC-NLG（基于反事实解码的去偏生成）
- CASAM（因果感知自注意力机制）

还包含多个 **zero-shot LLM** 对比（Qwen, Llama2, Gemma）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Accuracy %）

| 模型 | LEVEN | QA | CAIL | LEDGAR | Overruling | 平均 |
|------|-------|-----|------|---------|------------|-------|
| BERT | 72.37 | 78.64 | 88.45 | 87.01 | 95.69 | 84.43 |
| Legal-RoBERTa | 35.46 | 76.87 | 62.93 | 87.29 | 96.57 | 79.82 |
| CASAM | 71.08 | 81.37 | 86.19 | 87.21 | 92.75 | 83.72 |
| AC-NLG | 74.11 | 83.85 | 88.48 | 82.28 | 89.56 | 83.66 |
| **Ours** | **74.26** | **85.72** | **89.31** | **88.31** | **97.05** | **86.93** |

> ✅ 在所有数据集中均取得**最优或次优**成绩，尤其在 **QA 和 CAIL** 上领先显著。

---

### 与基线方法的对比结果

- 在 **50%训练数据**条件下：
  - 平均准确率达 **85.72%**，超过 BERT+CRF（83.02%）约 **2.7个百分点**。
  - 超过 XLM-RoBERTa（78.31%）达 **7.4个百分点**。
- 在 **1%极低资源场景**下：
  - 平均准确率为 **61.17%**，远超第二名 LLMEmbed（55.07%）。
  - 在 LEVEN 上达到 **28.89%**，比 BERT（1.79%）高出近 **16倍**。
- 在 **few-shot setting（5%数据）** 下，性能提升达 **12.54个百分点**，显示极强的数据利用率。

---

### 消融实验结果

在 CAIL 数据集上进行了三组消融实验（平均Accuracy）：

| 模型变体 | 描述 | 准确率 | 相对下降 |
|--------|------|--------|----------|
| Full Model | 完整框架 | **89.31%** | — |
| w/o LLM Factor Refinement | 移除LLM精炼，仅用YAKE | 87.31% | ↓2.0 ppt |
| w/o Knowledge Augmentation | 移除法律知识注入（prompt中无条文） | 87.89% | ↓1.42 ppt |
| w/o Causal Constraint | 移除注意力因果约束模块 | 86.19% | ↓3.12 ppt |

> 🔍 表明三大组件均对性能有显著贡献，尤其是**因果注意力约束机制**影响最大。

---

### 控制变量实验：LLM选边 vs 随机加边

设计严格对照实验验证LLM消歧的有效性：
- **实验组**：使用LLM选择不确定边的方向
- **控制组**：随机添加相同数量的边（保持拓扑密度一致）

👉 结果显示：**LLM选边在所有训练比例和任务中全面胜出**，即使在1%数据下仍保持稳定优势。

> 📌 证明性能增益来自LLM注入的**高质量语义逻辑**，而非简单的图复杂度增加。

---

## 4. 关键结论和发现

### 主要发现

1. **LLM可有效充当“软法律知识库”**  
   在缺乏干预数据的情况下，LLM能基于法律常识和条文理解，为模糊因果方向提供可靠先验判断。

2. **混合提取机制显著提升法律因素质量**  
   “统计初筛 + LLM语义精炼”的两阶段策略，有效去除命名实体噪声，提高要素纯度。

3. **因果结构不确定性可通过概率采样建模**  
   多图采样 + BIC加权集成的方式，既保留了不确定性，又实现了稳健推理。

4. **因果先验能显著增强注意力机制**  
   将因果强度作为注意力引导信号，使模型聚焦于真正影响判决的关键事实。

5. **方法在低资源和高相似罪名任务中优势明显**  
   尤其适用于 **“诈骗 vs 敲诈勒索”、“贪污 vs 挪用公款”** 等细微差异场景。

---

### 方法的局限性

1. **LLM依赖性强**：若LLM本身存在法律幻觉或领域偏差，会影响因果图质量。
2. **计算开销较大**：涉及多次LLM调用、图采样与PSM估计，推理成本高于普通PLM。
3. **无法处理隐变量完全未知的情况**：虽使用GFCI处理confounder，但仍受限于可观测变量集合。
4. **提示工程敏感**：LLM输出受prompt设计影响较大，需精心构造推理链。

---

### 未来工作方向

1. **探索更高效的LLM代理机制**：如蒸馏小型专家模型替代LLM进行实时推理。
2. **引入动态干预模拟**：结合反事实推理进一步验证因果路径合理性。
3. **扩展至多跳推理任务**：应用于法律问答、判例类推等需要深层推理的任务。
4. **跨法系迁移研究**：验证框架在大陆法系与英美法系之间的泛化能力。
5. **构建可视化因果解释界面**：辅助法官理解AI推荐背后的逻辑链条。

---

> ✅ **总结一句话**：本文首次系统性地将 **LLM的法律常识** 与 **统计因果发现** 相结合，解决了LJP任务中“要素噪声”与“结构模糊”两大难题，实现了更准确、更鲁棒、更具解释性的法律判决预测。

</details>

---

### 11. [Try, Check and Retry: A Divide-and-Conquer Framework for Boosting Long-context Tool-Calling Performance of LLMs](https://arxiv.org/abs/2603.11495)

**Authors**: Kunfeng Chen, Qihuang Zhong, Juhua Liu, Bo Du, Dacheng Tao  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.11495v1  

#### Abstract
Tool-calling empowers Large Language Models (LLMs) to interact with external environments. However, current methods often struggle to handle massive and noisy candidate tools in long-context tool-calling tasks, limiting their real-world application. To this end, we propose Tool-DC, a Divide-and-Conq...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Try, Check and Retry: A Divide-and-Conquer Framework for Boosting Long-context Tool-Calling Performance of LLMs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Large Language Models (LLMs)** 在执行 **tool-calling**（工具调用）任务时面临两大挑战：
- **长上下文问题**：当候选工具数量增多时，输入上下文变长，导致模型推理困难，性能显著下降。
- **混淆工具问题**：存在语义相似但参数定义不同的工具，容易引发参数填充错误（argument-filling errors）。

这些问题在真实场景中尤为突出，限制了 LLMs 在复杂环境中的应用。

---

### 🚀 提出的新方法：Tool-DC 框架
作者提出 **Tool-DC**，一个基于 **Divide-and-Conquer**（分而治之）思想的框架，通过 “**Try-Check-Retry**” 范式提升 LLMs 在长上下文下的 tool-calling 性能。

#### 核心思想：“Try-Check-Retry”
1. **Try（尝试）**  
   将所有候选工具划分为多个子组（subgroups），并行进行局部推理（local inference），降低单次推理的上下文长度与噪声干扰。
   
2. **Check（验证）**  
   利用 **schema constraints** 对生成的工具调用进行严格校验，过滤掉无效调用（如函数名错误、参数缺失、类型不匹配等）。
   
3. **Retry（重试）**  
   基于验证后的有效工具集合，重新进行全局决策，利用 LLM 的自省能力（self-reflection）优化最终输出。

---

### 🔍 两种实现方式
| 方法 | 类型 | 特点 |
|------|------|------|
| **Tool-DC (TF)** | Training-Free（无需训练） | 即插即用，灵活部署，适用于任何 black-box LLM |
| **Tool-DC (TB)** | Training-Based（需微调） | 将 Try-Check-Retry 推理轨迹构造成 CoT 数据，通过 SFT 内化到模型参数中，提升推理效率 |

---

### ⭐ 相比现有方法的优势
| 方面 | Tool-DC 的优势 |
|------|----------------|
| **鲁棒性更强** | 显著缓解因工具数量增加带来的性能衰减，尤其对小模型更有效 |
| **无需依赖外部检索器质量** | 不完全依赖 retriever（如 BM25），即使漏检也能通过分组机制找回 |
| **可泛化性强** | 支持多种 LLM 架构（Qwen, Llama, Gemma 等）和闭源模型（GPT-4o-mini, DeepSeek-V3.2） |
| **推理可控性高** | 通过 schema 验证保证输出合法性，减少 hallucination |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 描述 |
|-------|------|
| **BFCL (Berkeley Function-Calling Leaderboard)** (Patil et al., 2025) | 包含 Non-Live（合成）和 Live（人工编写）两类任务，涵盖 single/multiple/parallel function calling 场景 |
| **ACEBench** (Chen et al., 2025a) | 多领域工具调用基准，覆盖技术、金融、健康等 8 大类，测试集包含 828 个样本 |

> 💡 **扩展设置（Extended Setting）**：为模拟真实世界噪声，将候选工具从标准的 <10 个扩展至 **20 个**，加入无关工具以测试鲁棒性。

---

### 🎯 实验设置与评估指标
| 设置项 | 内容 |
|--------|------|
| **评估模式** | Standard Setting（原生工具列表）、Extended Setting（扩展至 20 工具） |
| **评估指标** | **AST Exact-Match Accuracy**（抽象语法树精确匹配准确率） |
| **模型范围** | Qwen2.5 系列（1.5B/3B/7B）、Qwen3-4B、Llama-3.1/3.2、Gemma-3-it、GPT-4o-mini、DeepSeek-V3.2 |
| **Retriever** | 使用 **BM25** 进行初始 Top-K 检索（K=min(5, N)） |
| **Group 数量 K** | 默认设为 5，支持敏感性分析 |

---

### 🆚 基线方法对比
#### ✅ Training-Free 基线：
- `GT_Funs`：仅使用正确工具作为上下文（理想上限）
- `All_Funs`：直接提供全部候选工具
- `Top-K`：仅保留检索出的 Top-K 工具
- `HiTEC-ICL` (Cui et al., 2025)：基于手动设计的 error checklist 引导推理
- `ToolGT (Prompting)` (Dang et al., 2025)：课程式提示引导逐步调用

#### ✅ Training-Based 基线：
- 基础模型 + Vanilla SFT（无推理链）
- 多个专有模型（OpenAI o3, Claude-Haiku-4.5, Gemini-3-Pro 等）
- 开源专用模型（xLAM, ToolACE, Hammer2.1 等）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 和 Figure 4）

#### 在 **Qwen2.5 系列模型** 上的表现（Extended Setting）：
| 模型 | 方法 | BFCL Overall ↑ | ACEBench Overall ↑ | 相对 All_Funs 提升 |
|------|------|------------------|--------------------|---------------------|
| Qwen2.5-1.5B | Tool-DC(TF) | 63.07% | 46.08% | **+25.10% avg** |
| Qwen2.5-3B | Tool-DC(TF) | 64.77% | 48.17% | +11.45% avg |
| Qwen2.5-7B | Tool-DC(TF) | 77.20% | 58.83% | +14.11% avg |

> 🔺 Tool-DC(TF) 在小模型上增益最大，说明其有效缓解了小模型处理长上下文的能力瓶颈。

---

#### 在其他主流 LLM 上的一致性提升（Figure 4）：
| 模型 | Tool-DC(TF) vs All_Funs 提升 |
|------|-------------------------------|
| Llama-3.2-1B | +37.1% |
| Gemma-3-it-1B | +20.4% |
| GPT-4o-mini | **+5.3%** |
| DeepSeek-V3.2 | +9.7% |

> ✅ 表明 Tool-DC(TF) 具备良好的跨模型通用性。

---

#### Training-Based 结果（Table 2）：
| 模型 | 方法 | BFCL Overall |
|------|------|--------------|
| Qwen2.5-7B-Instruct | Vanilla SFT | 79.24% |
| Qwen2.5-7B-Instruct | **Tool-DC(TB)** | **83.16%**（↑+6.21%） |
| | **vs OpenAI o3 (77.58%)** | ✅ **超越** |
| | **vs Claude-Haiku-4.5 (82.59%)** | ✅ **略胜** |

> 🔥 **Qwen2.5-7B + Tool-DC(TB) 达到了甚至优于部分闭源大模型的效果！**

---

### 🔍 消融实验结果（Table 3）
在 Qwen2.5-3B 上进行消融研究（Extended Setting）：

| 方法 | Non-Live | Live | Overall |
|------|---------|------|---------|
| Full Tool-DC(TF) | 71.79 | 57.74 | **64.77** |
| w/o Try | 44.19 | 29.39 | 36.79 ↓ |
| w/o Check | 52.06 | 52.41 | 52.24 ↓ |
| w/o Retry | 37.33 | 3.18 | **5.26** ↓↓ |

> ❗ **Remove Retry → 性能崩溃！**  
说明 Retry 阶段对于整合正向信号、完成最终决策至关重要。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **长上下文是 tool-calling 的主要瓶颈**  
   所有模型在工具数量上升后均出现明显性能下降，尤其是小模型。

2. **简单的上下文扩展无法解决问题**  
   即使使用支持百万 token 的 **Qwen2.5-7B-Instruct-1M** 或 **InternLM2.5-7B-Chat-1M**，在 Extended Setting 下仍表现糟糕（All_Funs 仅得 16.29%），证明不能靠“加长上下文窗口”解决根本问题。

3. **Tool-DC 显著提升鲁棒性和准确性**  
   - Tool-DC(TF) 平均带来 **+25.10%** 的相对增益；
   - Tool-DC(TB) 可使开源模型达到或超过商用闭源模型水平。

4. **分组策略具有强健性**  
   敏感性分析显示（Figure 6, 8），Tool-DC 对 group 数量 $ K $ 不敏感，在 $ K \in [4,6] $ 时性能最优且稳定。

5. **推理开销可控**  
   尽管 Tool-DC(TF) 需多次前向传播，但性能增益远超延迟代价（见 Figure 7）。

---

### ⚠️ 局限性
1. **依赖高质量 seed dataset**  
   当前 Tool-DC(TB) 的 CoT 数据基于 `xlam-function-calling-60k` 构建，缺乏多样性与大规模噪声场景覆盖。

2. **仅验证单步调用**  
   实验集中在 single-step tool-calling，未测试 multi-step 或 nested 场景（如最新 BFCL 中的复杂代理任务）。

3. **推理延迟较高（针对 TF）**  
   Tool-DC(TF) 需要多轮调用，不适合低延迟场景，更适合精度优先的应用。

---

### 🔮 未来工作方向
1. 构建更大规模、更具噪声的真实 world-like 训练集；
2. 探索结合 **Reinforcement Learning**（如 GRPO）进一步优化 tool-calling 策略；
3. 扩展至 **multi-step / agent-level** 工具调度任务；
4. 设计更高效的分组与聚合机制，降低推理成本。

---

## ✅ 总结
**Tool-DC** 是一种简单而强大的 divide-and-conquer 框架，通过 **Try-Check-Retry** 范式显著提升了 LLMs 在 **long-context、noisy candidate tools** 场景下的 tool-calling 能力。无论是即插即用的 **Tool-DC(TF)** 还是高效推理的 **Tool-DC(TB)**，都在多个模型和基准上实现了 SOTA 表现，甚至让开源中小模型媲美或超越顶级闭源模型，为构建可靠 AI Agent 提供了重要技术路径。

</details>

---

### 12. [FlexRec: Adapting LLM-based Recommenders for Flexible Needs via Reinforcement Learning](https://arxiv.org/abs/2603.11901)

**Authors**: Yijun Pan, Weikang Qiu, Qiyao Ma, Mingxuan Ju, Tong Zhao, Neil Shah, Rex Ying  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.11901v1  

#### Abstract
Modern recommender systems must adapt to dynamic, need-specific objectives for diverse recommendation scenarios, yet most traditional recommenders are optimized for a single static target and struggle to reconfigure behavior on demand. Recent advances in reinforcement-learning-based post-training ha...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FlexRec: Adapting LLM-based Recommenders for Flexible Needs via Reinforcement Learning**

---

## **1. 主要贡献和创新点**

### **解决的问题**
现代推荐系统通常被优化为单一静态目标（如点击率或购买转化），难以动态适应多样化的用户需求和业务目标。例如，用户意图可能在“最大化兴趣”、“探索小众内容”和“推广热门趋势”之间切换，而传统推荐模型无法灵活响应这些变化。

此外，现有的基于 **Reinforcement Learning from Verifiable Rewards (RLVR)** 的 LLM 推荐方法面临两大挑战：
1. **粗粒度信用分配**：序列级奖励（sequence-level rewards）无法区分排名中每个项目放置的贡献，导致训练信号稀疏且不精确。
2. **稀疏且噪声大的反馈**：真实场景中用户交互数据稀少，依赖学习型 critic 补全奖励时易引入高方差估计，影响策略更新稳定性。

---

### **提出的新方法：FlexRec**
FlexRec 是一个用于后训练（post-training）LLM 推荐器的强化学习框架，通过以下两个核心机制提升性能与鲁棒性：

#### **(1) Swap-based Item-level Reward（基于交换的项目级奖励）**
- 利用**反事实交换操作**（counterfactual swap）评估每个项目在当前位置的边际贡献。
- 对于第 $k$ 个位置的项目 $a_k$，计算将其与后续任意位置 $j > k$ 的项目交换后对整体排序质量（如 NDCG）的影响。
- 定义 item-level 改进量为：
  $$
  \Delta_k(y;x) = \mathbb{E}_{j \sim \text{Unif}([k+1:K])}[R_n(y^{(k\leftrightarrow j)};x) - R_n(y;x)]
  $$
- 最终得到**因果感知、位置敏感、可比较的项目级奖励**，实现细粒度信用分配。

> ✅ 优势：相比 Rec-R1 的序列级 GRPO 和 ConvRec-R1 的 rank-level 奖励，该设计避免了前缀依赖偏差，支持跨 rollout 的公平归一化。

#### **(2) Uncertainty-aware GRPO（不确定性感知的 GRPO 更新）**
- 引入一个神经网络 **critic** 同时预测用户-项目交互得分及其**预测方差**（uncertainty）。
- 在计算优势函数时，利用方差加权，降低高不确定性样本的更新权重：
  $$
  c_i = \frac{1}{u_i + \epsilon}, \quad A_{\text{final}} = c_i \cdot A_i
  $$
- 实现“方向保留、强度调节”的稳健更新。

> ✅ 优势：有效缓解稀疏反馈下 critic 错误传播问题，防止过自信错误奖励主导训练过程。

---

### **相比现有方法的优势**
| 方法 | 缺陷 | FlexRec 的改进 |
|------|------|----------------|
| **Rec-R1** | 序列级奖励 → 粗粒度信用分配 | 细粒度 item-level 奖励 → 更高效学习 |
| **ConvRec-R1** | Rank-level 奖励不可比（受前缀影响） | 因果交换奖励消除偏差，具备可比性 |
| **传统 CF + KNN** | 无不确定性建模，噪声大 | 显式建模 reward uncertainty，稳定训练 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 类型 | 特点 |
|--------|------|------|
| **KuaiRec** | 短视频推荐 | 全观测交互 + 丰富元数据；子采样至 10% 模拟稀疏反馈 |
| **MovieLens-1M (ML-1M)** | 电影评分推荐 | 长期偏好建模基准 |
| **ESCI** | 商品搜索推荐 | 查询驱动排序任务，非用户历史依赖 |

> 所有数据集均构建为闭集重排序（closed-set reranking）任务，候选集大小 $C=30$

---

### **实验设置与评估指标**

#### **评估指标**
- **NDCG@K** ($K=5,10,30$): 衡量整体排序质量
- **Recall@5**: 衡量 top-5 是否包含正例
- **MRR@5**: 衡量首个相关项的位置

#### **训练细节**
- 使用 **Qwen2.5-3B-Instruct** 作为 backbone LLM
- RL 后训练采用 **Verl** 库，在 4×A100 上进行 full-parameter 微调
- 包含 KL 正则（0.01）和熵正则（0.005）以控制策略漂移

#### **需要建模（Need Construction）**
作者构造了四种显式需求指令用于训练与测试：
1. **Maximizing Interest**：按观看比例或评分排序
2. **Explore New Topics (Niche Discovery)**：鼓励未接触过的主题/类型
3. **Trend Promotion**：结合近期流行度（过去24小时观看数）
4. **Product Search**：基于查询匹配度（exact/substitute/complement）

---

### **基线方法对比**
| 类别 | 方法 | 描述 |
|------|------|------|
| **传统重排序器** | BERT4Rec, STAR | 基于 Transformer 或 LLM embedding 的序列推荐模型 |
| **零样本 LLM** | GPT-4o, Qwen2.5-3B | 不微调直接提示生成排序 |
| **后训练 LLM 推荐器** | TALLRec, Rec-R1, Rank-GRPO | 分别代表 SFT、序列级 GRPO、rank-level GRPO 方法 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1 & Table 3）**

#### **单需求性能（Maximizing Interest）**
| 方法 | KuaiRec N@5 | KuaiRec R@5 ↑ | ML-1M N@5 | ML-1M R@5 |
|------|-------------|---------------|-----------|----------|
| BERT4Rec | 0.415 | 0.182 | 0.502 | 0.128 |
| GPT-4o (zero-shot) | 0.376 | 0.154 | 0.508 | 0.139 |
| Rec-R1 | 0.391 | 0.174 | 0.554 | 0.170 |
| **FlexRec (Ours)** | **0.597** | **0.335** | **0.615** | **0.235** |
| △ vs Qwen baseline | **+59.2%** | **+109.4%** | **+23.7%** | **+79.4%** |

> 💡 FlexRec 在 KuaiRec 上 Recall@5 提升超过 **109%**，NDCG@5 提升近 **60%**

#### **多需求泛化能力（Zero-shot Transfer）**
| 训练目标 | 测试目标 | FlexRec R@5 (KuaiRec) | 最佳 baseline R@5 | 提升幅度 |
|--------|---------|-----------------------|--------------------|----------|
| Max-Interest | Explore New Topics | **0.165** | 0.147 (Rec-R1) | **+17.9%** |
| Max-Interest | Trend Promotion | **0.165** | 0.152 (TALLRec) | **+24.1%** |

> ✅ 即使仅在一个需求上训练，FlexRec 能有效迁移到其他需求，体现其通用性。

#### **产品搜索任务表现（ESCI）**
| 方法 | N@5 | R@5 |
|------|-----|-----|
| GPT-4o | 0.502 | 0.647 |
| Rec-R1 | 0.504 | 0.652 |
| **FlexRec** | **0.528** | **0.678** |
| △ vs Qwen | +17.6% | +15.9% |

> 🎯 FlexRec 不仅适用于用户行为序列推荐，也适用于纯查询驱动的搜索任务。

---

### **消融实验结果**

#### **Ablation on Reward Formulation（Table 7）**
| 奖励形式 | N@5 (GRPO) | N@5 (PPO) |
|--------|------------|-----------|
| Independent contribution | 0.461 | 0.390 |
| Non-causal swap | 0.607 | 0.383 |
| **Causal swap (ours)** | **0.607** | **0.621** |

> 🔍 因果约束（仅与剩余池交换）至关重要，违反会导致性能下降。

#### **Ablation on Reward Signals（Table 8）**
| 方法 | N@5 | R@5 |
|------|-----|-----|
| User-KNN CF | 0.410 | 0.194 |
| Item-KNN CF | 0.417 | 0.208 |
| Raw critic (no uncertainty) | 0.566 | 0.280 |
| **FlexRec (uncertainty-aware)** | **0.595** | **0.319** |

> ⚠️ 显式建模 uncertainty 可带来额外 **+3–5%** 性能增益，并显著提升训练稳定性（见 Figure 4）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **细粒度奖励显著提升学习效率**  
   FlexRec 的 swap-based item-level reward 提供密集、位置感知的监督信号，克服了传统 RL 中 credit assignment 过于粗糙的问题，加速收敛并提高最终性能。

2. **不确定性建模是稀疏反馈下的关键**  
   在用户交互高度稀疏的场景中，直接使用 critic 预测值会因误差累积导致训练崩溃。FlexRec 通过 variance-aware weighting 实现鲁棒优化。

3. **单一 LLM 可成为通用推荐引擎**  
   一个联合训练于多个需求的 FlexRec 模型可通过指令切换行为模式（见 Figure 2 和 G.1 示例），实现“一个模型，多种用途”，具备作为 **universal recommender** 的潜力。

4. **推理过程更具可解释性**  
   模型能根据需求生成合理推理路径，如识别“用户未接触的主题”或“最近24小时热门内容”，增强了透明性和可信度。

---

### **局限性**
- 当前工作聚焦于 **closed-set reranking** 场景，假设候选集已由检索模块提供，未涉及端到端 retrieval。
- 奖励标签来源于历史信号（如 watch ratio、ratings），可能存在偏差，未考虑开放世界中新物品动态加入的情况。
- 计算开销：item-level reward 引入 $O(K^2)$ 开销，虽在 $K=30$ 下可接受，但在更大候选集上需优化。

---

### **未来工作方向**
1. 将 FlexRec 扩展至 **retrieval-augmented recommendation** 架构，整合召回阶段。
2. 探索更高效的 swap 采样策略（如 top-j sampling）以降低 item-level reward 计算成本。
3. 引入 human feedback 或 preference modeling 替代代理 reward，进一步提升对齐质量。
4. 研究如何将 FlexRec 应用于多模态推荐（如图文、视频）场景。

---

> ✅ **总结一句话**：  
> **FlexRec 通过因果感知的 item-level 奖励 + 不确定性感知的 GRPO 更新，在多样化推荐需求下实现了更强、更稳定、更通用的 LLM 推荐器训练范式，大幅超越传统与 LLM-based 基线。**

</details>

---

### 13. [DocSage: An Information Structuring Agent for Multi-Doc Multi-Entity Question Answering](https://arxiv.org/abs/2603.11798)

**Authors**: Teng Lin, Yizhang Zhu, Zhengxuan Zhang, Yuyu Luo, Nan Tang  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.11798v1  

#### Abstract
Multi-document Multi-entity Question Answering inherently demands models to track implicit logic between multiple entities across scattered documents. However, existing Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) frameworks suffer from critical limitations: standard RAG's v...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《DocSage: An Information Structuring Agent for Multi-Doc Multi-Entity Question Answering》核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文聚焦于**Multi-document Multi-entity Question Answering (MDMEQA)** 这一复杂任务，即在多个非结构化文档中跨实体进行隐含逻辑推理以生成答案。现有方法面临以下核心挑战：

- **标准 RAG** 基于向量相似度的检索过于粗粒度，容易遗漏关键事实；
- **Graph-based RAG** 难以高效整合碎片化的复杂关系网络，且图构建成本高；
- 所有方法普遍缺乏 **schema awareness**，导致无法系统组织跨文档实体关系，证据链断裂、推理不准确。

### **提出了什么新方法或新思路**

作者提出 **DocSage** ——一个端到端的 **agentic framework**，通过动态结构化信息来增强多文档多实体问答能力。其核心思想是模仿人类认知过程：将原始文本转化为结构化知识以简化复杂推理。

DocSage 包含三个协同工作的模块：

1. **Interactive Schema Discovery Module (ASK 算法)**  
   动态推断查询相关的最小可连接 **relational schema**，识别必要实体、属性与关系。采用交互式提问机制主动澄清歧义、补充缺失信息，提升 schema 准确性。

2. **Logic-Aware Structured Extraction Module (CLEAR 机制)**  
   将非结构化文本转换为语义一致的 **relational tables**，并引入双层纠错机制：
   - 单点置信度评估（基于 LoRA 微调 + Conformal Prediction）
   - 跨记录逻辑一致性检查（如函数依赖、时间约束、外键完整性）

3. **Schema-Guided Relational Reasoning Module**  
   在结构化数据库上执行基于 schema 的 **multi-hop relational reasoning**，利用 SQL 引擎完成精确的 join、filter 和 aggregation 操作，避免 LLM 注意力扩散（attention diffusion）。

### **相比现有方法的优势**

- ✅ **精准事实定位**：通过 SQL 支持的索引实现关键事实的 pinpoint 定位；
- ✅ **天然支持跨文档实体连接**：关系表形式天然支持多文档间的 entity join；
- ✅ **缓解 LLM 注意力稀释**：结构化表示减少对长上下文建模的依赖；
- ✅ **可验证性与可解释性**：每条答案均可追溯至原始文档中的具体证据片段（provenance tracing）；
- ✅ **错误控制保障**：CLEAR 机制提供统计与符号层面的双重错误保证。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **MEBench** (Lin et al., 2025a)  
  专为多实体问答设计的基准，包含 4,780 个精心构造的问题，分为三类：
  - Comparison（比较）
  - Statistics（统计分析）
  - Relationship（关系推理）

- **Loong** (Wang et al., 2024)  
  测试模型在超长文档下的推理能力，包含四类任务：
  - Spotlight Locating（精确定位）
  - Comparison（比较）
  - Clustering（聚类）
  - Chain of Reasoning（多跳推理）  
  文档长度从 10K 到 250K tokens 不等，用于评估信息分散场景下的鲁棒性。

### **实验设置和评估指标**

| 组件 | 设置 |
|------|------|
| **LLMs** | 主要使用 GPT-4o 和 Qwen3；信息抽取使用 Mistral-7B |
| **Reasoning Model** | GPT-4o |
| **评估指标** | - MEBench：**Accuracy**（整体及各子任务）<br>- Loong：**Avg Score (0–100)** + **Perfect Rate (Exact Match, EM)** |

### **基线方法对比**

- **GPT-4o**：强生成式基线
- **GPT-4o + RAG**：标准检索增强框架
- **GraphRAG** (Edge et al., 2024)：基于知识图谱的 RAG
- **StructRAG** (Li et al., 2024)：结构感知型 RAG，动态选择结构化表示

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **MEBench 结果（Accuracy）**

| 方法 | Comparison | Statistics | Relationship | **Overall** |
|------|------------|------------|--------------|-------------|
| GPT-4o | 0.262 | 0.353 | 0.407 | 0.338 |
| GPT-4o + RAG | 0.696 | 0.579 | 0.593 | 0.620 |
| GraphRAG | 0.618 | 0.558 | 0.593 | 0.586 |
| StructRAG | 0.678 | 0.588 | 0.573 | 0.612 |
| **DocSage (Ours)** | **0.934** | **0.908** | **0.812** | **0.892** |

> 📌 **提升幅度**：相比最强基线 GPT-4o + RAG，**绝对准确率提升 27.2%**。

#### **Loong 结果（Avg Score / Perfect Rate）**

| 方法 | Overall Score / EM |
|------|---------------------|
| GPT-4o | 54.17 / 0.26 |
| GPT-4o + RAG | 43.05 / 0.18 |
| GraphRAG | 33.44 / 0.07 |
| StructRAG | 60.56 / 0.23 |
| **DocSage (Ours)** | **68.29 / 0.53** |

> 📌 **Perfect Rate 超过第二名两倍以上**，表明答案更完整、更可靠。

### **与基线方法的对比结果**

- DocSage 在所有任务类型、文档长度和数据集中均显著优于所有基线；
- 特别是在 **高实体密度**（Set3 >100 entities）和 **长文档**（200K–250K tokens）场景下表现最为稳健；
- 其他模型随文档增长性能急剧下降，而 DocSage 下降最缓，证明其良好的**可扩展性**。

### **消融实验结果（Ablation Study）**

#### **在 MEBench 上的结果（Overall Accuracy）**

| 变体 | Accuracy | 下降 |
|------|----------|-------|
| DocSage (Full) | 0.892 | — |
| w/o Structured Extraction | 0.691 | ▼20.1pp |
| w/o Schema-Guided Reasoning | 0.773 | ▼11.9pp |
| w/o Schema Discovery | 0.781 | ▼11.1pp |
| w/o CLEAR (no logic check) | 0.849 | ▼4.3pp |
| w/ Passive Schema Discovery | 0.863 | ▼2.9pp |

> 🔍 **结论**：Structured Extraction 是最关键模块，其次是 Schema-Guided Reasoning。

#### **在 Loong 上的关键发现**

- **Schema Discovery 对长文档至关重要**：移除后在 Chain of Reasoning 任务中下降达 -13.22 分；
- **Structured Extraction 提升比较类任务精度**：因需精确属性聚合；
- **CLEAR 错误纠正机制有效维持数据完整性**，尤其在多步推理中；
- **ASK 交互式发现带来增量收益**，特别是在模糊引用解析中。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **结构化信息组织是解决 MDMEQA 的有效范式**：将非结构化文本动态转化为 relational schema + tables 显著提升了跨文档推理能力。
2. ✅ **agentic workflow 设计具有优势**：通过“发现 → 提取 → 推理”的闭环流程，实现了对信息碎片化、隐含关系和高精度要求的有效应对。
3. ✅ **SQL 引擎赋能复杂推理**：将 multi-hop join 和 aggregation 交给数据库引擎处理，避免了 LLM 的注意力稀释和幻觉问题。
4. ✅ **误差控制机制至关重要**：CLEAR 提供的统计+符号双重校验显著提高了数据质量和最终答案可靠性。

### **方法的局限性**

- ⚠️ **计算开销较高**：由于采用迭代式的 agentic 流程（尤其是 ASK 和 CLEAR），推理延迟高于传统 RAG；
- ⚠️ **依赖基础模型能力**：schema 发现与提取效果受限于所用 LLM 的理解能力和泛化水平；
- ⚠️ **假设文档具有一定语义一致性**：面对极端噪声、矛盾或高度专业化领域文本时性能可能下降。

### **未来工作方向**

- 优化 agent 决策效率，降低推理延迟；
- 探索轻量化 schema 发现与 extraction 模块，适配更多场景；
- 扩展至 **multi-modal** 数据（如表格、图像 caption）的联合结构化；
- 构建开放平台支持 real-time 文档流的动态知识库更新。

---

> 🔗 **代码与数据已开源**：https://anonymous.4open.science/r/DocSage-07A7

</details>

---

### 14. [HPC Containers for EBRAINS: Towards Portable Cross-Domain Software Environment](https://arxiv.org/abs/2603.12044)

**Authors**: Krishna Kant Singh, Eric M\"uller, Eleni Mathioulaki, Wouter Klijn, Lena Oden  
**Category**: cs.DC  
**Published**: 2026-03-13  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.12044v1  

#### Abstract
Deploying complex, distributed scientific workflows across diverse HPC sites is often hindered by site-specific dependencies and complex build environments. This paper investigates the design and performance of portable HPC container images capable of encapsulating MPI- and CUDA-enabled software sta...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题：** *HPC Containers for EBRAINS: Towards Portable Cross-Domain Software Environment*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在跨多个异构 **High-Performance Computing (HPC)** 平台部署复杂的科学计算工作流时，常面临以下挑战：
- 软件环境依赖冲突（如 MPI、CUDA 版本不一致）
- 编译工具链和系统库的平台特异性导致构建困难
- 难以实现可重复、可移植的运行环境
- 容器化后可能牺牲底层硬件性能（尤其是 GPU 和高速网络通信）

这些问题严重阻碍了科研成果的**可复现性**和**跨机构协作效率**。

### ✅ 提出的新方法与创新思路
本文提出并验证了一种基于 **Apptainer** 的**便携式 HPC 容器化策略**，其核心是采用 **PMIx-based hybrid containerization** 架构：
- 将完整的 **MPI stack**（包括 Open MPI 5、UCX、PRRTE、hwloc 等）打包进容器镜像中
- 利用 **Slurm 的 `--mpi=pmix` 接口**，使容器内的 MPI 进程通过 PMIx 协议与主机资源管理器通信
- 动态挂载主机的硬件驱动（如 `/dev/infiniband`, `/dev/nvidia*`），实现对 InfiniBand 和 GPU 的直接访问
- 使用统一的 **Spack-based EBRAINS Software Distribution (ESD)** 构建流程生成容器镜像

该方法实现了“一次构建，到处运行”（build once, run anywhere）的目标，同时保持接近裸金属（bare-metal）的性能表现。

### ✅ 相比现有方法的优势
| 方面 | 传统做法 | 本文方法 |
|------|----------|---------|
| 可移植性 | 需为每个站点重新编译 | 单一镜像跨平台运行 |
| 性能隔离 | 通常牺牲通信性能 | 几乎无通信开销 |
| 易用性 | 依赖专家手动配置 | 自动化 CI 流水线支持 |
| 可复现性 | 环境差异大 | 完全版本锁定的软件栈 |
| 兼容性 | 依赖特定 MPI 实现 | 支持标准 PMIx 接口 |

> 💡 **关键优势**：无需牺牲性能即可实现高性能计算应用的**端到端可移植性与可复现性**。

---

## 2. 核心实验方法和设置

### ✅ 实验平台
在两个生产级 HPC 集群上进行测试：
- **Karolina**（IT4Innovations, Czech Republic）
  - CPU: AMD EPYC 7H12 (128 cores/node)
  - GPU: 8× NVIDIA A100 per node, NVLink12
  - 网络: InfiniBand HDR
- **JURECA-DC**（Forschungszentrum Jülich, Germany）
  - CPU: AMD EPYC 7742 (128 cores/node)
  - GPU: 4× NVIDIA A100 per node, NVLink4
  - 网络: InfiniBand HDR100

> 两者的操作系统、GCC、CUDA、OpenMPI 等版本均不同，用于检验跨平台兼容性。

### ✅ 容器构建方式
- 使用 **Apptainer (Singularity)** 构建容器镜像
- 构建两种镜像：
  - CPU-only 镜像（Rocky Linux 10.1）
  - GPU-accelerated 镜像（CUDA 12.2）
- 所有依赖由 **Spack** 管理，确保可重现构建过程
- 容器内自带完整 Open MPI 5 + UCX + PMIx 栈

### ✅ 评估指标与基准测试
#### 微基准测试（Microbenchmarks）：
| 工具 | 测试内容 |
|------|--------|
| **OSU_init** | MPI 初始化时间（startup latency） |
| **OSU_latency** | 点对点通信延迟（intra-/inter-node） |
| **NCCL-tests (all_reduce_perf)** | GPU 集合通信带宽（bus bandwidth） |

#### 应用级基准测试（Neuroscience Workloads）：
| 模拟器 | 测试场景 |
|-------|--------|
| **Arbor** | Ring network benchmark（强扩展性 & 弱扩展性，CPU/GPU） |
| **NEURON** | Ringtest benchmark（仅 CPU） |

#### 对比模式：
- **Native Execution**：原生模块加载环境（host-native）
- **Apptainer Container**：相同任务在容器中执行
- 所有测试均使用相同的 Slurm 提交脚本，仅替换执行命令为 `apptainer exec`

---

## 3. 主要实验结果和性能指标

### ✅ 微基准测试结果

#### 🔹 OSU_init（MPI 初始化时间）
| 平台 | 结果 |
|------|-----|
| **Karolina** | 容器初始化慢于原生，且随节点数增加差距扩大（256 节点时高出约 30–40%） |
| **JURECA-DC** | 容器初始化**快 50%**！表明容器环境可能绕过了某些冗余探测步骤 |

> 📌 **结论**：初始化开销具有平台依赖性，并非固定值；容器不一定更慢。

#### 🔹 OSU_latency（通信延迟）
- **小消息（≤1 KiB）**：
  - 容器开销 < 0.2 μs（intra-node），< 0.05 μs（inter-node）
- **中等/大消息（>128 KiB）**：
  - 容器与原生性能几乎一致，差异在测量噪声范围内

> ✅ **总体结论**：**Apptainer 在通信延迟方面引入的开销可忽略不计**

#### 🔹 NCCL-tests（GPU 集合通信）
| 场景 | 性能偏差 |
|------|--------|
| **单节点（NVLink 内部通信）** | 最大偏差仅 **1.29%**（JURECA） |
| **双节点（RDMA over InfiniBand）** | 带宽一致性极高，偏差 ≤ **0.09%**（Karolina）、≤ **0.01%**（JURECA） |

> ✅ **结论**：**GPUDirect RDMA 在容器中正常工作，无性能损失**

---

### ✅ 应用级性能结果

#### 🔹 Arbor（CPU）
- **强扩展性（Strong Scaling）**
  - Karolina：容器效率 62.6%，原生 67.5%
  - JURECA：容器效率达 **98.0%**，优于原生（95.9%）
- **弱扩展性（Weak Scaling）**
  - 容器与原生性能曲线高度重合，最大偏差 < 6%

> ✅ **结论**：CPU 模拟下容器无显著性能退化

#### 🔹 NEURON（CPU）
- 强/弱扩展性测试中，容器与原生运行时间**几乎完全重叠**
- 差异在运行间波动范围内，**无统计显著性**

> ✅ **结论**：容器对纯 CPU 类神经模拟无影响

#### 🔹 Arbor（GPU）
| 指标 | 观察结果 |
|------|--------|
| **相对开销** | 容器比原生慢 **12% ~ 19%** |
| **开销性质** | 固定相对开销（constant relative overhead） |
| **弱扩展性** | 绝对延迟恒定（~+13 秒），不随规模增长而恶化 |
| **强扩展性** | 开销占比随节点增多下降（从 18.2% → 12.7%） |

> ⚠️ **注意**：尽管存在开销，但**扩展行为未受损**，仍保持良好并行效率（最高达 79.6% @64 nodes）

> ❓ **潜在原因推测**：可能是由于容器内外 **CUDA 用户态库版本不匹配**（容器用 12.2，主机用 12.4/13.0）所致，需进一步分析。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **便携式 HPC 容器可行且高效**
   - 基于 **PMIx + Apptainer** 的混合架构可在不同 HPC 平台上运行同一镜像
   - 无需重新编译或调优，实现真正的“write once, run anywhere”

2. **通信性能几乎无损**
   - MPI 和 NCCL 的点对点及集合通信性能与裸金属基本一致
   - 支持 GPUDirect RDMA 和 NVLink，证明容器不影响底层硬件直通

3. **CPU 应用性能持平**
   - Arbor 和 NEURON 在容器中的运行效率与原生相当
   - 弱扩展性和强扩展性均保持理想趋势

4. **GPU 存在轻微但稳定的开销**
   - 容器化导致约 **12–19% 的额外耗时**
   - 该开销为**固定比例型**，不会随问题规模放大，不影响扩展性

5. **容器可作为系统诊断工具**
   - 通过对比容器与原生性能，发现了 JURECA-DC 上隐藏的 MPI 启动延迟异常
   - 表明容器可作为标准化的“黄金参考”用于 HPC 系统健康检查

---

### ⚠️ 方法的局限性
1. **GPU 性能开销尚未完全解释**
   - 当前未深入分析 CUDA 用户空间库（如 `libcuda.so`）版本差异的影响
   - 缺乏内核级 profiling 数据定位瓶颈

2. **评估范围有限**
   - 仅测试了 ESD 中部分工具（Arbor、NEURON）
   - 未涵盖多组件耦合的工作流（multi-step pipelines）

3. **自动化程度不足**
   - 当前仍需人工查看日志判断是否使用最优传输路径（如 RoCE vs TCP）
   - 无法自动检测和修复次优配置

---

### 🔮 未来工作方向
1. **集成至 ESD 的 CI/CD 流水线**
   - 实现全自动化的容器构建、测试与发布
   - 让研究人员可直接拉取“即插即用”的性能验证镜像

2. **开发自动化日志分析机制**
   - 解析 UCX、Open MPI、NCCL 的 debug 日志
   - 自动识别并告警非最优通信路径（如 fallback to TCP）

3. **探索透明绑定主机优化库**
   - 利用 **EESSI** 或类似方案动态挂载站点预优化的驱动和库
   - 提升容器性能的同时保留便携性

4. **支持组合式工作流测试**
   - 在 CI 中加入跨工具流水线的功能与性能验证
   - 确保复杂科学流程在容器中仍能高效运行

5. **推进“软件流式交付”（software streaming）**
   - 借鉴 CVMFS/EESSI 思路，按需加载容器内容
   - 减少启动时间和存储压力

---

## ✅ 总结
本论文成功验证了 **Apptainer-based HPC 容器**在真实科研场景下的**可行性、便携性与高性能**。它不仅解决了跨平台部署难题，还为 EBRAINS 等大型研究基础设施提供了**统一、可复现、易分发的软件分发模型**。虽然 GPU 上存在轻微开销，但整体性能损失可控，且不影响扩展能力。未来将推动该方法融入自动化构建体系，最终实现“一键部署、处处高效”的愿景。

</details>

---

### 15. [NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication](https://arxiv.org/abs/2603.11438)

**Authors**: Yusheng Zheng  
**Category**: cs.DC  
**Published**: 2026-03-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.11438v1  

#### Abstract
NCCL is the de facto standard for collective GPU communication in large-scale distributed training, relying heavily on plugins to customize runtime behavior. However, these plugins execute as unverified native code within NCCL's address space, risking job crashes, silent state corruption, and downti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NCCLbpf: Verified, Composable Policy Execution for GPU Collective Communication

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
NCCL 是大规模分布式训练中 GPU 集体通信（collective communication）的事实标准，其通过 **plugin 插件机制**（如 tuner、profiler、net plugin）实现运行时策略定制。然而，这些插件以 **未经验证的原生代码（native code）** 形式在 NCCL 地址空间中执行，存在严重安全隐患：
- 插件中的空指针解引用、无限循环、竞态条件等可导致训练任务崩溃或静默失败；
- 插件间缺乏结构化状态共享机制，无法实现闭环自适应策略；
- 更新插件需重启训练作业，造成生产环境中的服务中断。

这些问题在大型集群中代价高昂（如 Llama 3 预训练期间因通信问题频繁中断）。

---

### 🚀 提出的新方法与创新思路
作者提出 **NCCLbpf** —— 一种将 **userspace eBPF 运行时**嵌入 NCCL 插件接口的新型框架，无需修改 NCCL 源码即可实现安全、可组合、热更新的策略执行。

#### 主要创新点：
1. **Load-time 静态验证机制**
   - 所有 eBPF 插件在加载时由 PREVAIL-based 验证器进行内存安全、终止性、栈安全等检查，防止不安全行为（如空指针访问、越界读写）在运行时发生。
   - 验证失败则拒绝加载，保障系统始终处于“已验证”状态。

2. **基于 Typed Maps 的跨插件状态共享**
   - 引入 eBPF 的 `maps` 机制作为结构化、类型安全的状态存储，支持 tuner 与 profiler 插件之间共享延迟、通道数等信息。
   - 实现了此前无法做到的 **闭环反馈控制**（closed-loop adaptation），例如根据历史性能动态调整通信参数。

3. **原子级策略热更新（Atomic Hot-Reload）**
   - 支持在运行时无缝替换策略，通过 compare-and-swap 原子切换函数指针。
   - 更新过程无调用丢失，且旧策略继续服务未完成请求，确保高可用。

4. **零侵入式集成**
   - 完全兼容 NCCL 现有的 plugin ABI（如 `ncclTunerPlugin_v3/v5`, `ncclProfilerPlugin_v1/v6`），仅需替换 `.so` 文件即可启用，无需重新编译 NCCL 或部署内核模块。

---

### 🔍 相比现有方法的优势

| 对比维度 | 传统 NCCL Plugin | AutoCCL / 自定义 Native Plugin | NCCLbpf |
|--------|------------------|-------------------------------|--------|
| 安全性 | ❌ 无验证，易崩溃 | ❌ 同样为 native code，风险相同 | ✅ Load-time 验证，杜绝内存错误 |
| 可组合性 | ❌ 插件孤立，无法共享状态 | ❌ 依赖 ad-hoc 共享内存，易出错 | ✅ 结构化 maps 支持跨插件协作 |
| 动态更新 | ❌ 必须重启作业 | ❌ 同样需要重启 | ✅ 原子热更新，零中断 |
| 性能开销 | ⚠️ 原生性能 | ⚠️ 原生性能 | ✅ <0.03% 额外延迟 |
| 易用性 | ✅ 成熟 API | ✅ 灵活但危险 | ⚠️ 需掌握 eBPF 编程（未来可通过 DSL 改善） |

> 💡 **核心优势总结**：NCCLbpf 在几乎不牺牲性能的前提下，将 NCCL 插件从“不可靠的黑盒”转变为“可验证、可组合、可持续演进”的安全扩展平台。

---

## 2. 核心实验方法和设置

### 🧪 实验平台配置
- **GPU 节点**：8× NVIDIA B300 SXM6（Blackwell 架构），每卡 275GB 显存
- **互联方式**：NVLink 5（带宽达 1.8TB/s/GPU）
- **软件栈**：
  - CUDA 13.0
  - NCCL 2.29.7
  - bpftime（轻量级 userspace eBPF 运行时）
- **CPU 微基准测试**：AMD EPYC 9575F（240 核），用于测量单次调用延迟

---

### 📊 评估指标
| 指标类别 | 具体指标 |
|--------|--------|
| **性能开销** | 单次策略决策延迟（ns）、端到端 collective latency 影响 |
| **安全性** | 不安全程序是否被验证器拦截 |
| **热更新能力** | 切换耗时、是否有调用丢失 |
| **功能有效性** | 是否能提升 AllReduce/AllGather 吞吐量 |
| **稳定性** | 多次运行的方差（CV）、是否存在异常波动 |

---

### 🔁 基线方法对比
- **Native Baseline**：相同逻辑的原生 C++ 插件（`-O2` 编译），用于衡量 eBPF 层额外开销
- **No-plugin Baseline**：关闭所有插件的原始 NCCL 行为
- **NCCL Default Policy**：NCCL 内置默认算法选择（如 NVLS for AllReduce）
- **Bad Policy（对照组）**：人为设计的低效策略（如强制使用 1 channel）以验证策略影响力

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）eBPF 策略执行延迟（CPU 微基准）
| 策略类型 | P50 延迟 (ns) | 相对原生开销 (ΔP50) |
|--------|-------------|------------------|
| Native baseline | 20 ns | — |
| noop（空策略） | 100 ns | +80 ns |
| size_aware_v2 | 100 ns | +80 ns |
| lookup_only | 130 ns | +110 ns |
| lookup_update | 140 ns | +120 ns |
| slo_enforcer（最复杂） | 150 ns | +130 ns |

> ✅ **结论**：最复杂的 eBPF 策略也仅引入 **80–130 ns** 开销，占典型 collective latency（~394 μs for 128MiB AllReduce）的 **<0.03%**。

#### （2）端到端 GPU 通信开销（NVLink）
- 小消息（8B–256KiB）：增加约 **1.3 μs 固定开销**（主要来自插件框架初始化，非 eBPF JIT）
- 大消息（≥4MiB）：开销低于测量噪声（<0.1%），可忽略

#### （3）热更新性能
- **总 reload 时间**：~9.4 ms（含验证 + JIT）
- **关键路径切换时间**（compare-and-swap）：**1.07 μs**
- **连续 400,000 次调用中**：**零调用丢失**
- 若新策略验证失败：旧策略持续运行，系统保持可用

#### （4）安全性测试结果
- 测试 14 个 eBPF 程序（7 安全 + 7 不安全）
- 所有不安全程序均被验证器拒绝，包括：
  - Null pointer dereference
  - Out-of-bounds access
  - Unbounded loop
  - Stack overflow
  - Division by zero
  - Illegal helper call
  - Input field write
- 错误提示明确，便于调试

> 示例：
> ```text
> VERIFIER REJECT: RO is a pointer to map_value_or_null; must check != NULL before dereference at insn 7
> ```

#### （5）实际性能优化效果（Case Study）

##### ➤ Message-size-aware eBPF Policy 提升 AllReduce 吞吐
| 消息大小 | NCCL 默认 (NVLS) | eBPF Policy (Ring/LL128 or Simple) | 提升幅度 |
|--------|------------------|------------------------------------|---------|
| 4MiB   | 133.5 GB/s       | 148.1 GB/s                         | +10.9%  |
| 8MiB   | 196.3 GB/s       | 249.7 GB/s                         | **+27.2%** |
| 16MiB  | 278.8 GB/s       | 337.4 GB/s                         | +21.0%  |
| 32MiB  | 349.3 GB/s       | 402.4 GB/s                         | +15.2%  |
| 128MiB | 596.9 GB/s       | 628.9 GB/s                         | +5.4%   |
| 256MiB+ | 更优             | 略差                               | 自动回退 |

> ✅ **结论**：该策略在 4–128 MiB 区间平均提升 **5.5–26.5%** 吞吐，且在其他区间自动回退至默认策略，无负面影响。

##### ➤ 稳定性表现（128MiB AllGather ×20 runs）
| 配置 | 平均吞吐 (GB/s) | 标准差 | 变异系数 (CV) |
|-----|----------------|-------|--------------|
| NCCL Default | 565.6 ± 0.9 | 0.9 | 0.15% |
| NCCLbpf Policy | 565.5 ± 0.6 | 0.6 | **0.10%**（↓32%） |

> ✅ 政策版本更稳定，无异常离群值（default 有一次 562.6 GB/s 的 outlier）

---

### 🔍 消融实验与功能验证

#### （1）跨插件可组合性验证（Profiler → Tuner Feedback Loop）
- 设计一个 **adaptive channels policy**：
  - 初始保守设置 `nChannels=2`
  - Profiler 收集每次通信的延迟并写入 shared eBPF map
  - Tuner 读取延迟，若高于阈值则逐步增加通道数
- 实验流程：
  1. 正常阶段：通道数从 2 → 12（约 100k 次调用）
  2. 注入延迟干扰（模拟拥塞）：通道数回落至 2
  3. 恢复正常：通道数再次上升至 12
- ✅ 成功验证了 **闭环自适应能力**，这是原生 NCCL 插件架构无法实现的功能。

#### （2）Net Plugin 扩展能力
- 实现了一个包裹 Socket transport 的 eBPF net plugin
- 在 `isend/irecv` 中插入计数逻辑，统计连接数与传输字节数
- 开销：<2%，证明 eBPF 可高效介入数据平面（data-path）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **NCCL 插件的安全隐患是真实且严重的**，已有多个生产级 bug 报告（use-after-free、死锁、segfault）。
2. **eBPF 模型非常适合移植到高性能库扩展场景**，其“验证后执行”范式可在极低开销下提供强安全保障。
3. **结构化状态共享（maps）解锁了新的控制能力**，首次实现了 tuner 与 profiler 的协同闭环调控。
4. **热更新完全可行且实用**，切换延迟微秒级，无调用丢失，适合生产环境快速迭代。
5. **即使简单策略也能带来显著性能收益**（最高 +27%），说明当前 NCCL 默认策略仍有优化空间。

---

### ⚠️ 方法的局限性
| 局限性 | 说明 |
|------|------|
| **策略语义无法验证** | eBPF 验证器只保证内存安全和终止性，不能阻止“逻辑错误”策略（如故意降速） |
| **编程门槛较高** | 需熟悉 eBPF 编程模型和受限 C 语法，对 ML 工程师不够友好 |
| **当前覆盖范围有限** | 仅支持 tuner、profiler、net 插件；env plugin 尚未接入 |
| **尚未验证多节点场景** | 当前实验集中在单节点 NVLink，InfiniBand 多节点待测 |
| **不支持自定义 collective 算法** | NCCLbpf 仅调度内置算法，不能实现全新通信原语 |

---

### 🔮 未来工作方向
1. **开发高层策略 DSL**：将常见调优模式抽象为领域专用语言，自动编译为 eBPF 字节码，降低使用门槛。
2. **扩展至 RCCL 和其他通信库**：AMD 的 RCCL 也有类似插件机制，有望实现跨厂商通用框架。
3. **支持更多插件类型**：如 env plugin、custom transport backend。
4. **多节点与 InfiniBand 验证**：在真实数据中心环境中评估可扩展性和网络影响。
5. **结合机器学习进行自动调优**：利用 eBPF 收集的细粒度 telemetry 数据训练轻量级在线控制器。

---

## 🔚 总结
**NCCLbpf 是一次成功的“操作系统级安全思想向高性能计算库迁移”的实践**。它借助 eBPF 的 verified extensibility 模型，在几乎零性能损失的情况下，解决了 NCCL 插件长期存在的安全性、可维护性和可组合性难题。其实验结果充分证明了该方法的有效性与实用性，为未来构建更智能、更可靠的分布式训练基础设施提供了重要基础。

> 🔗 项目开源地址：[https://github.com/eunomia-bpf/nccl-eBPF](https://github.com/eunomia-bpf/nccl-eBPF)

</details>

---

### 16. [AGMARL-DKS: An Adaptive Graph-Enhanced Multi-Agent Reinforcement Learning for Dynamic Kubernetes Scheduling](https://arxiv.org/abs/2603.12031)

**Authors**: Hamed Hamzeh  
**Category**: cs.DC  
**Published**: 2026-03-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.12031v1  

#### Abstract
State-of-the-art cloud-native applications require intelligent schedulers that can effectively balance system stability, resource utilisation, and associated costs. While Kubernetes provides feasibility-based placement by default, recent research efforts have explored the use of reinforcement learni...

---

### 17. [Beyond Barren Plateaus: A Scalable Quantum Convolutional Architecture for High-Fidelity Image Classification](https://arxiv.org/abs/2603.11131)

**Authors**: Radhakrishnan Delhibabu  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.11131v1  

#### Abstract
While Quantum Convolutional Neural Networks (QCNNs) offer a theoretical paradigm for quantum machine learning, their practical implementation is severely bottlenecked by barren plateaus -- the exponential vanishing of gradients -- and poor empirical accuracy compared to classical counterparts. In th...

---

### 18. [Relaxed Efficient Acquisition of Context and Temporal Features](https://arxiv.org/abs/2603.11370)

**Authors**: Yunni Qu (The University of North Carolina at Chapel Hill), Dzung Dinh (The University of North Carolina at Chapel Hill), Grant King (University of Michigan), Whitney Ringwald (University of Minnisota Twin Cities), Bing Cai Kok (The University of North Carolina at Chapel Hill), Kathleen Gates (The University of North Carolina at Chapel Hill), Aiden Wright (University of Michigan), Junier Oliva (The University of North Carolina at Chapel Hill)  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.11370v1  

#### Abstract
In many biomedical applications, measurements are not freely available at inference time: each laboratory test, imaging modality, or assessment incurs financial cost, time burden, or patient risk. Longitudinal active feature acquisition (LAFA) seeks to optimize predictive performance under such cons...

---

### 19. [Deep Learning Network-Temporal Models For Traffic Prediction](https://arxiv.org/abs/2603.11475)

**Authors**: Yufeng Xin, Ethan Fan  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.11475v1  

#### Abstract
Time series analysis is critical for emerging net- work intelligent control and management functions. However, existing statistical-based and shallow machine learning models have shown limited prediction capabilities on multivariate time series. The intricate topological interdependency and complex ...

---

### 20. [Reversible Lifelong Model Editing via Semantic Routing-Based LoRA](https://arxiv.org/abs/2603.11239)

**Authors**: Haihua Luo, Xuming Ran, Tommi K\"arkk\"ainen, Zhonghua Chen, Jiangrong Shen, Qi Xu, Fengyu Cong  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.11239v1  

#### Abstract
The dynamic evolution of real-world necessitates model editing within Large Language Models. While existing methods explore modular isolation or parameter-efficient strategies, they still suffer from semantic drift or knowledge forgetting due to continual updating. To address these challenges, we pr...

---

### 21. [COMPASS: The explainable agentic framework for Sovereignty, Sustainability, Compliance, and Ethics](https://arxiv.org/abs/2603.11277)

**Authors**: Jean-S\'ebastien,  Dessureault,  Alain-Thierry, Iliho Manzi,  Soukaina, Alaoui Ismaili,  Khadim,  Lo,  Mireille,  Lalancette,  \'Eric,  B\'elanger  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.11277v1  

#### Abstract
The rapid proliferation of large language model (LLM)-based agentic systems raises critical concerns regarding digital sovereignty, environmental sustainability, regulatory compliance, and ethical alignment. Whilst existing frameworks address individual dimensions in isolation, no unified architectu...

---

### 22. [CreativeBench: Benchmarking and Enhancing Machine Creativity via Self-Evolving Challenges](https://arxiv.org/abs/2603.11863)

**Authors**: Zi-Han Wang, Lam Nguyen, Zhengyang Zhao, Mengyue Yang, Chengwei Qin, Yujiu Yang, Linyi Yang  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.11863v1  

#### Abstract
The saturation of high-quality pre-training data has shifted research focus toward evolutionary systems capable of continuously generating novel artifacts, leading to the success of AlphaEvolve. However, the progress of such systems is hindered by the lack of rigorous, quantitative evaluation. To ta...

---

### 23. [CLASP: Defending Hybrid Large Language Models Against Hidden State Poisoning Attacks](https://arxiv.org/abs/2603.12206)

**Authors**: Alexandre Le Mercier, Thomas Demeester, Chris Develder  
**Category**: cs.CL  
**Published**: 2026-03-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.12206v1  

#### Abstract
State space models (SSMs) like Mamba have gained significant traction as efficient alternatives to Transformers, achieving linear complexity while maintaining competitive performance. However, Hidden State Poisoning Attacks (HiSPAs), a recently discovered vulnerability that corrupts SSM memory throu...

---

### 24. [Deep Learning-Based Metamodeling of Nonlinear Stochastic Dynamic Systems under Parametric and Predictive Uncertainty](https://arxiv.org/abs/2603.12012)

**Authors**: Haimiti Atila, Seymour M. J. Spence  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.12012v1  

#### Abstract
Modeling high-dimensional, nonlinear dynamic structural systems under natural hazards presents formidable computational challenges, especially when simultaneously accounting for uncertainties in external loads and structural parameters. Studies have successfully incorporated uncertainties related to...

---

### 25. [The Unlearning Mirage: A Dynamic Framework for Evaluating LLM Unlearning](https://arxiv.org/abs/2603.11266)

**Authors**: Raj Sanjay Shah, Jing Huang, Keerthiram Murugesan, Nathalie Baracaldo, Diyi Yang  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.11266v1  

#### Abstract
Unlearning in Large Language Models (LLMs) aims to enhance safety, mitigate biases, and comply with legal mandates, such as the right to be forgotten. However, existing unlearning methods are brittle: minor query modifications, such as multi-hop reasoning and entity aliasing, can recover supposedly ...

---

### 26. [Systematic Scaling Analysis of Jailbreak Attacks in Large Language Models](https://arxiv.org/abs/2603.11149)

**Authors**: Xiangwen Wang, Ananth Balashankar, Varun Chandrasekaran  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.11149v1  

#### Abstract
Large language models remain vulnerable to jailbreak attacks, yet we still lack a systematic understanding of how jailbreak success scales with attacker effort across methods, model families, and harm types. We initiate a scaling-law framework for jailbreaks by treating each attack as a compute-boun...

---

### 27. [Language Generation with Replay: A Learning-Theoretic View of Model Collapse](https://arxiv.org/abs/2603.11784)

**Authors**: Giorgio Racca, Michal Valko, Amartya Sanyal  
**Category**: cs.LG  
**Published**: 2026-03-13  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.11784v1  

#### Abstract
As scaling laws push the training of frontier large language models (LLMs) toward ever-growing data requirements, training pipelines are approaching a regime where much of the publicly available online text may be consumed. At the same time, widespread LLM usage increases the volume of machine-gener...

---

### 28. [Measuring AI Agents' Progress on Multi-Step Cyber Attack Scenarios](https://arxiv.org/abs/2603.11214)

**Authors**: Linus Folkerts, Will Payne, Simon Inman, Philippos Giavridis, Joe Skinner, Sam Deverett, James Aung, Ekin Zorer, Michael Schmatz, Mahmoud Ghanem, John Wilkinson, Alan Steer, Vy Hong, Jessica Wang  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.11214v1  

#### Abstract
We evaluate the autonomous cyber-attack capabilities of frontier AI models on two purpose-built cyber ranges-a 32-step corporate network attack and a 7-step industrial control system attack-that require chaining heterogeneous capabilities across extended action sequences. By comparing seven models r...

---

### 29. [LLM-Augmented Digital Twin for Policy Evaluation in Short-Video Platforms](https://arxiv.org/abs/2603.11333)

**Authors**: Haoting Zhang (Max), Yunduan Lin (Max), Jinghai He (Max), Denglin Jiang (Max),  Zuo-Jun (Max),  Shen, Zeyu Zheng  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.11333v1  

#### Abstract
Short-video platforms are closed-loop, human-in-the-loop ecosystems where platform policy, creator incentives, and user behavior co-evolve. This feedback structure makes counterfactual policy evaluation difficult in production, especially for long-horizon and distributional outcomes. The challenge i...

---

### 30. [Verified Multi-Agent Orchestration: A Plan-Execute-Verify-Replan Framework for Complex Query Resolution](https://arxiv.org/abs/2603.11445)

**Authors**: Xing Zhang, Yanwei Cui, Guanghui Wang, Qucy Wei Qiu, Ziyuan Li, Fangwei Han, Yajing Huang, Hengzhi Qiu, Bin Zhu, Peiyang He  
**Category**: cs.AI  
**Published**: 2026-03-13  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.11445v1  

#### Abstract
We present Verified Multi-Agent Orchestration (VMAO), a framework that coordinates specialized LLM-based agents through a verification-driven iterative loop. Given a complex query, our system decomposes it into a directed acyclic graph (DAG) of sub-questions, executes them through domain-specific ag...

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
