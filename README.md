# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-02 09:57:06 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [DREAM-S: Speculative Decoding with Searchable Drafting and Target-Aware Refinement for Multimodal Generation](https://arxiv.org/abs/2606.00535)

**Authors**: Zining Liu, Yunhai Hu, Tianhua Xia, Bo Bao, Eric Sather, Vithursan Thangarasa, Sai Qian Zhang  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2606.00535v1  

#### Abstract
Speculative decoding (SD) has proven to be an effective technique for accelerating autoregressive generation in large language models (LLMs) however, its application to vision-language models (VLMs) remains relatively unexplored. We propose~\textit{DREAM-S}, a novel SD framework designed specificall...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DREAM-S: Speculative Decoding with Searchable Drafting and Target-Aware Refinement for Multimodal Generation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **Speculative Decoding (SD)** 技术在 **Large Language Models (LLMs)** 中已被证明能有效加速自回归生成，但在 **Vision-Language Models (VLMs)** 上的应用仍处于探索阶段。VLMs 面临更高的计算开销，尤其是视觉输入带来的额外 FLOPs 和 KV Cache 负担。传统 SD 方法难以直接迁移至多模态场景。

DREAM-S 旨在解决以下挑战：
- 如何设计高效的 **draft model**，使其既能被快速执行，又能获得高 **acceptance ratio**；
- 如何在硬件层面优化 draft model 的架构以实现最优速度提升；
- 如何利用 target model 的中间特征来提升 draft model 的预测准确性。

---

### 🚀 提出的新方法与创新思路

DREAM-S 是一种专为 **VLMs 多模态生成** 设计的新型 **Speculative Decoding 框架**，其核心创新包括：

#### （1）基于 Neural Architecture Search (NAS) 的可搜索草稿机制（Searchable Drafting）
- 引入 **target-aware supernet training** 框架，在训练过程中联合优化多种 draft model 架构配置。
- 自动搜索并确定三个关键维度的最佳组合：
  - **Attention Head Pruning**：动态剪枝注意力头以降低计算量；
  - **Visual Token Compression**：根据 target model 注意力重要性选择保留的图像 token；
  - **Adaptive Target Feature Injection**：从 target model 不同层中自适应提取特征用于 cross-attention 监督。

> 🔍 *通过 NAS 在硬件平台感知下自动识别最优 draft 配置，无需人工调参。*

#### （2）目标感知的中间特征蒸馏（Target-Aware Refinement）
- 提出 **Adaptive Intermediate Feature Distillation (AIFD)** 方法，基于 attention entropy 和跨层变化 ΔAE 动态选择最稳定的中间层作为监督信号。
- 使用 **cross-attention 机制** 将 target model 的中间表示注入 draft model，增强知识迁移效率。

> 💡 *解决了“静态固定层监督”导致不稳定或次优的问题，显著提升 draft token 的 accept length。*

#### （3）两阶段渐进式训练（Two-Phase Progressive Training, TPPT）
- **Phase 1**: Warm-up training，对完整 supernet 进行初始化训练；
- **Phase 2**: Multi-resolution training，采用 OFA 风格 progressive shrinking，训练各种子网络配置。

---

### ⚖️ 相比现有方法的优势

| 方面 | DREAM-S 的优势 |
|------|----------------|
| **通用性** | 支持多种 VLM 架构（LLaVA、Pixtral、SmolVLM），且可适配不同硬件（A100/H100/RTX8000） |
| **自动化程度高** | 无需手动设计 draft model 结构，NAS 自动完成最优配置搜索 |
| **性能更强** | 显著优于 EAGLE、Medusa、Hydra 等主流 SD 方法，最高达 **3.85× speedup** |
| **鲁棒性强** | 即使在 zero-shot 设置下也保持领先，说明增益来自架构而非过拟合 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练数据**：
  - 主要使用 `LLaVA-mix665k` 数据集中的 55,000 个样本；
  - 补充各评测基准中与测试集无交集的 1,000 个样本进行领域适配。
- **评估数据集**（共六个）：
  - **MMT-Bench**：综合性多任务 VLM 测评基准
  - **SEED-Bench-2**
  - **ScienceQA**
  - **OCRBench**：侧重 OCR 能力
  - **ChartQA**：图表理解与推理
  - **MathVista**：数学视觉推理

---

### ⚙️ 实验设置

| 项目 | 设置详情 |
|------|----------|
| **目标模型（Target Model）** | 冻结参数，包括：<br>• LLaVA-v1.6-Vicuna-7B / 13B<br>• Pixtral-12B<br>• SmolVLM-2B |
| **Draft Model 架构** | 包含 3 个 decoder 层的小型 Transformer |
| **训练方式** | Two-Phase Progressive Training (TPPT)，总迭代次数 68,000 |
| **优化器** | AdamW (β₁=0.9, β₂=0.95)，学习率 3e-5，梯度裁剪 0.5 |
| **硬件平台** | 单张 NVIDIA A100 80GB GPU（推理）；四张 A100 用于 supernet 训练 |
| **NAS 搜索策略** | Exhaustive search 离线进行一次，针对每种 model-hardware 组合找出最优 draft 子网 |

---

### 📊 评估指标

| 指标 | 定义 |
|------|------|
| **Speedup Ratio (S)** | $ S = \frac{t_{AR}}{t_{\text{method}}} $，即标准 autoregressive 解码耗时与当前方法耗时之比 |
| **Average Accepted Token Length (T)** | 每轮验证中被 target model 接受的连续 draft token 数量均值，反映 draft 准确性 |

---

### 🆚 基线方法对比

共比较了 **8 种 state-of-the-art SD 方法**，全部适配于 VLM 场景：
- **SPD** (Gagrani et al., 2024)
- **Kangaroo** (Liu et al., 2024a)
- **Medusa** (Cai et al., 2024)
- **Hydra** (Ankner et al., 2024)
- **EAGLE / EAGLE-2 / EAGLE-3** (Li et al., 2024d,c,2025)
- **DREAM** (Hu et al., 2025b)

此外还与并发工作 **ViSpec** (Kang et al., 2025) 进行公平对比（相同训练数据）。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 2）

| Model | 方法 | 平均 Speedup (S) | 最高 Speedup |
|-------|------|------------------|--------------|
| LLaVA-7B | DREAM-S | **2.35×** | 2.67× (MMT-Bench) |
| LLaVA-13B | DREAM-S | **3.16×** | **3.85×** |
| Pixtral-12B | DREAM-S | **2.69×** | 3.01× |
| SmolVLM-2B | DREAM-S | **2.32×** | 3.12× |

> ✅ **最高达到 3.85× 加速**，远超现有方法。

---

### 🔁 与基线方法的对比结果

#### ✅ 对比 EAGLE-3：
| Model | DREAM-S vs EAGLE-3 (Speedup ↑) |
|-------|-------------------------------|
| LLaVA-7B | +10% (2.35× vs 2.13×) |
| LLaVA-13B | +9% (3.16× vs 2.89×) |
| Pixtral-12B | +10% (2.69× vs 2.45×) |
| SmolVLM-2B | +5% (2.32× vs 2.20×) |

#### ✅ 对比 DREAM：
| Model | DREAM-S vs DREAM (Speedup ↑) |
|-------|------------------------------|
| LLaVA-7B | +5% (2.35× vs 2.23×) |
| LLaVA-13B | +3% (3.16× vs 3.06×) |
| Pixtral-12B | +2% (2.69× vs 2.60×) |

> 💬 尽管部分情况下 acceptance length 略低，但由于更激进的 **head pruning 和 token compression**，整体速度更快。

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）NAS 搜索维度的影响（Figure 3b）
关闭某一搜索维度后性能下降：
- 移除 **Head Pruning**：speedup ↓ 至 2.62×
- 移除 **Visual Token Pruning**：↓ 至 2.51×
- 固定 **Feature Injection Layer**（非自适应）：↓ 至 2.48×  
→ 证明所有三个维度都至关重要。

#### （2）AIFD 蒸馏策略的有效性（Figure 3c）
相比静态层监督（S-25%, S-50%, S-75%），**adaptive selection based on entropy + Δentropy** 显著提升性能，验证了动态选择的重要性。

#### （3）λ 权重影响分析（Figure 3d）
- 当 `λ_feat = λ_distill = 0.2`, `λ_KL = 1.0` 时效果最佳；
- 过高的特征监督权重（如 0.4）会损害泛化能力。

#### （4）硬件适应性测试（Table 4）
在不同 GPU 上均优于 EAGLE-3：
| GPU | DREAM-S Speedup | EAGLE-3 Speedup |
|-----|------------------|------------------|
| H100 | **2.99×** | 2.67× |
| A100 | **2.58×** | 2.26× |
| RTX8000 | **2.23×** | 1.88× |

> 👉 展现出强大的 **hardware-aware 优化能力**。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **NAS 可有效应用于 VLM 的 Speculative Decoding 设计**，自动发现最优 draft 架构、压缩比例与交互策略。
2. **Adaptive Intermediate Feature Distillation (AIFD)** 利用 attention entropy 动态选择稳定且富含语义的中间层，显著提升 draft accuracy。
3. **DREAM-S 实现高达 3.85× 的端到端加速**，全面超越现有 SD 方法，在多个 VLM 和 benchmark 上保持领先。
4. **更大的 target model 更受益于 DREAM-S**，表明其在复杂模型上的潜力更大。
5. **zero-shot 和 fixed-configuration 实验表明性能增益源于架构优势而非数据过拟合**。

---

### ⚠️ 方法的局限性

1. **系统级贡献为主**：未提出新的 NAS 算法或理论解码原理，而是构建了一个面向多模态的实用化搜索空间。
2. **搜索空间有限**：目前依赖 exhaustive search，当架构更复杂或设备更多样时可能需引入 predictor-based search（如 OFA）。
3. **不适用于 MoE 类架构**：对于专家路由机制复杂的模型（如 MoE-based VLMs），当前的 head pruning 和 token compression 可能不再适用，需要重新设计搜索维度。

---

### 🔮 未来工作方向

1. 扩展至 **MoE 架构 VLMs**，设计兼容 expert routing 的 draft 搜索策略；
2. 引入 **predictor-based NAS 搜索器**，应对更大规模的架构空间；
3. 探索 **动态 draft window size 调整机制**，进一步平衡 speculation aggressiveness 与效率；
4. 应用于其他多模态任务，如视频问答、具身智能等。

---

📌 **代码已开源**：https://github.com/SAI-Lab-NYU/DREAM-S

</details>

---

### 2. [Scaling LLM Inference Beyond Amdahl`s Limits via Eliminating Non-Scalable Overheads](https://arxiv.org/abs/2606.01927)

**Authors**: Alan Zhao, Cyril Y. He, Wei Xu  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2606.01927v1  

#### Abstract
Deployers of online LLM services usually seek to maximize cluster-wide performance given a fixed number of GPUs. Tensor parallelism (TP) is necessary to fit modern models but scales sub-linearly as the TP degree t grows, due to cross-GPU communication and non-scalable runtime work, as predicted by A...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Scaling LLM Inference Beyond Amdahl's Limits via Eliminating Non-Scalable Overheads*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大型语言模型（LLM）在线推理服务受限于 **Amdahl's Law**：尽管采用 Tensor Parallelism (TP) 可以将大模型分布到多个 GPU 上，但由于跨 GPU 通信开销以及非并行化任务（如调度、采样、I/O 处理等），系统吞吐量随 TP 度增加呈现亚线性甚至负向扩展。同时，KV Cache 内存压力导致频繁的 GPU-CPU 数据交换和请求抢占。

核心矛盾在于：
- **高 TP 度** → 更好的内存效率（缓解 KV Cache 压力）
- **低 TP 度** → 更少通信开销（避免性能退化）

因此存在一个经验最优 TP 度 $ t_e $，而传统系统难以提升该上限。

### 提出的新方法：Albireo
Albireo 是一种新型 LLM 推理引擎，通过消除非可扩展（non-scalable）开销来突破 Amdahl’s Law 的限制，其三大核心优化如下：

#### ✅ **Optimization 1: Optimistic Asynchronous Scheduling（乐观异步调度）**
- 引入“迭代依赖的序列管理”机制，预测每个序列在下一迭代中的资源需求（长度、KV 块数）。
- 假设所有序列都会继续生成（仅最后一次失败），提前进行内存预分配和调度决策。
- 实现了调度任务 $ T_1 $ 与前向计算 $ T_3 $ 的完全重叠，将 CPU 调度延迟从 ~4ms 降至 **<80μs**。

#### ✅ **Optimization 2: Early-Feedback Backfill（早期反馈回填）**
- 打破输入处理 $ T_2 $ 和输出处理 $ T_5 $ 对采样完成的依赖。
- 在采样过程中一旦生成 token ID，立即通过快速路径回填至输入处理器，用于构建下一轮模型输入。
- 使得 $ T_2 $ 和 $ T_5 $ 可与 $ T_3 $ 并行执行，显著减少 CPU 阻塞时间。

#### ✅ **Optimization 3: Sequence-Parallel Sampling（序列并行采样）**
- 将原本集中在 driver GPU 的采样任务按 batch 维度拆分到所有 TP worker 上。
- 利用 `all-to-all` 通信交换 logits，并行执行采样，最后聚合结果。
- 消除单点瓶颈，实现采样任务的真正并行化，支持任意 TP 度下的高效采样。

> 💡 **关键思想**：不是加速可扩展部分（如 forward），而是**缩小不可扩展部分的比例 (1-P)**，从而突破 Amdahl’s Law 的理论天花板。

### 相比现有方法的优势
| 方面 | vLLM / SGLang | Albireo |
|------|----------------|---------|
| 调度 | 同步阻塞 | 异步重叠（乐观预测） |
| 输入/输出处理 | 串行等待采样完成 | 早期回填，提前启动 |
| 采样 | 单 GPU 执行 | 多 GPU 并行 |
| 非可扩展开销占比 | 高（约 41%） | 极低（降低 89%） |
| 实际可达到的 $ t_e $ | 较小（如 Qwen-32B: $ t_e=2 $） | 更大（同模型 $ t_e=4 $） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Databricks-dolly-15k**：用于生成真实用户 prompt，模拟实际推理负载。
- Prompt 长度分布广泛，适用于测试长上下文场景下的 KV Cache 行为。

### 实验设置
#### 测试平台（Testbeds）
| 名称 | GPU 配置 | 互联方式 |
|------|--------|--------|
| H100M | 8×H100 (80GB) | NVLink |
| A100M | 8×A100 (80GB) | NVLink |
| A100N | 8×A100 (80GB) | PCIe |

#### 模型范围（FP16 精度）
共测试 8 个主流 LLM，覆盖不同规模：
- **Tiny**: Llama-2-7B, Qwen-2.5-7B
- **Small**: Llama-2-13B, Qwen-2.5-14B
- **Moderate**: Qwen-2.5-32B, QwQ-32B
- **Large**: Llama-3.1-70B, Qwen-2.5-72B

#### 配置参数
- 默认 per-GPU batch size: **32**
- TP degree $ t $ 设置依据公式估算的经验最优值：
  $$
  t_e \approx 4M / C
  $$
  其中 $ M $ 为模型权重大小，$ C $ 为单卡显存容量。
- Pipeline Parallelism (PP) 禁用，聚焦单节点内推理优化。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Throughput** | 每秒生成 token 数（token/s） |
| **Latency (TPOT)** | 每个输出 token 的平均延迟（Time-per-Output-Token） |
| **GPU Utilization** | GPU 利用率（SM active cycles） |
| **Power Usage & Energy** | 功耗（W）与总能耗（J） |
| **Tail Latency** | 99%/99.9% 百分位 TPOT |

### 基线方法对比
- **vLLM (v0.11.2)**：当前最先进开源推理引擎，使用 PagedAttention。
- **SGLang (v0.5.5)**：支持结构化生成与零开销调度的新兴系统。
- 所有框架均启用默认性能优化选项。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（H100M 平台）

| 模型类别 | Throughput 提升 | Latency 下降 | GPU Util ↑ | Energy ↓ |
|--------|------------------|-------------|------------|----------|
| Tiny (t=1) | ~1.3× | ~22% | ~10% | ~30% |
| Small (t=2) | ~1.5× | ~30% | ~20% | ~40% |
| Moderate (t=4) | ~1.7× | ~40% | ~28% | ~50% |
| Large (t=8) | **~1.9×** | **~48%** | **~40%** | **~54%** |

> 🔥 在生产环境中部署 Llama-3.1-70B 时，Albireo 实现了高达 **2× 的吞吐提升**。

### 与基线方法的对比结果
#### 📈 吞吐量（Throughput）
- 在相同 GPU 数量下，Albireo 支持更高的有效 TP 度：
  - Qwen-2.5-32B: $ t_e $ 从 2 → **4**
  - Llama-3.1-70B: $ t_e $ 从 4 → **8**
- 实现 **superlinear scaling**：当 $ t \leq t_e $ 时，$ T(t) > 2 \times T(t/2) $
- 图 10 显示，在 moderate/large 模型上，Albireo 在 $ t=4 $ 或 $ t=8 $ 时仍保持增长趋势，而 vLLM 已开始下降。

#### ⏱️ 延迟（Latency）
- **TPOT 平均降低 22%-48%**，尤其对大模型更明显。
- **尾部延迟大幅改善**：
  - QwQ-32B 的 99.9% TPOT 从 298ms（vLLM）降至 **76ms**（Albireo）
- TTFT（首 token 时间）基本不变（因 prefill 不涉及采样优化）

#### 💡 能效表现
- 平均功耗降低 **15%**
- 推理时间缩短 **47%**
- 总体能耗减少 **54%**

### 消融实验结果（Ablation Study）

#### 各组件贡献分析（图 15）
| 模型类型 | 异步调度贡献 | 并行采样贡献 |
|--------|--------------|--------------|
| Tiny/Small | 主导（接近全部增益） | 几乎无贡献（t=1 无法并行） |
| Moderate/Large | 显著贡献 | 显著贡献（两者协同作用） |

#### 单任务耗时对比（Qwen-2.5-32B, t=4）

| Task | vLLM | Albireo | 缩减比例 |
|------|------|---------|----------|
| $ T_1 $ (Scheduling) | ~4ms | **~5μs** | >99.8% |
| $ T_2 $ (Input Proc.) | ~4ms | **~40μs** | >99% |
| $ T_4 $ (Sampling) | ~6ms | **~1.5ms** | 75% |
| $ T_5 $ (Output Proc.) | ~0.5ms | **~25μs** | >95% |

> ❗ 总计消除 **89% 的非可扩展开销**，是性能飞跃的根本原因。

---

## 4. 关键结论和发现

### 主要发现
1. **Amdahl’s Law 的瓶颈不在 forward，而在非并行任务**  
   当前 LLM 推理系统的性能天花板由调度、I/O、采样等 CPU-bound 任务决定，而非 GPU 计算能力。

2. **可通过任务解耦与并行化突破传统缩放极限**  
   Albireo 证明：只要将非可扩展部分尽可能重叠或并行化，就能显著提高有效 TP 度 $ t_e $，实现超线性扩展。

3. **更大的 $ t_e $ 带来更高聚合吞吐量**  
   公式 $ t_e \approx 4M/C $ 可作为经验指导；Albireo 能让这一上限翻倍，极大提升集群利用率。

4. **生产环境收益更加显著**  
   在真实负载波动场景下，随着并发请求数上升，Albireo 的优势持续扩大，最终达到 **2× 吞吐**。

### 局限性
- **目前仅支持单节点部署**：未解决 multi-node 场景下的 TP all-reduce 和 PP stage imbalance 问题。
- **不兼容某些特殊采样策略**：若采样需全局状态同步（如 beam search），可能影响并行性。
- **额外通信开销依赖硬件带宽**：虽然实验证明 `scatter` 可被隐藏，但在低带宽设备上可能存在风险。

### 未来工作方向
1. **扩展至 Hybrid TP-PP 多节点架构**  
   结合 pipeline bubbles 优化与 stage 负载均衡，进一步释放超大规模模型潜力。

2. **探索更激进的任务流水线设计**  
   如多步前瞻调度（multi-iteration prediction）、动态负载迁移等。

3. **适配更多后训练技术（如 Speculative Decoding）**  
   将 sequence-parallel 思路应用于 draft model 与 verify model 的协同推理。

4. **面向边缘设备轻量化版本**  
   在资源受限设备上应用类似思想，优化内存与能效。

---

> ✅ **总结一句话**：  
> Albireo 通过 **异步调度 + 早期回填 + 序列并行采样**，将 LLM 推理中长期被忽视的“非计算开销”压缩到极致，成功突破 Amdahl’s Law 的束缚，在多种模型和平台上实现了 **最高达 2× 吞吐、48% 延迟下降、54% 节能** 的卓越性能，为下一代高性能推理系统提供了全新范式。

</details>

---

### 3. [ViBE: Co-Optimizing Workload Skew and Hardware Variability for MoE Serving](https://arxiv.org/abs/2606.00735)

**Authors**: Seokjin Go, Marko Scrbak, Ephrem Wu, Srilatha Manne, Divya Mahajan  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.00735v1  

#### Abstract
In distributed Mixture-of-Experts (MoE) inference, input-dependent token routing interacts with GPU performance variability to create persistent stragglers under synchronized execution, where the slowest GPU determines layer latency. This performance variability is inherent to modern accelerators: m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ViBE: Co-Optimizing Workload Skew and Hardware Variability for MoE Serving 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在分布式 **Mixture-of-Experts (MoE)** 推理系统中，传统负载均衡方法（如 token balancing）假设所有 GPU 性能一致，忽略了现代加速器之间固有的 **hardware variability**（硬件性能差异）。这种差异来源于制造工艺、功耗限制和温度变化，导致即使 token 分配均匀，不同 GPU 的执行时间仍存在显著差异。

由于 MoE 层采用 **expert parallelism (EP)**，每层需要所有 GPU 同步完成计算，因此最慢的 GPU（straggler）决定了整层延迟。这使得仅靠平衡 token 数量无法有效降低尾部延迟（tail latency），从而影响 **SLO attainment** 和系统利用率。

---

### 提出了什么新方法或新思路
本文提出 **ViBE (Variability-Informed Binning of Experts)** ——一种硬件感知的专家放置框架，其核心思想是：

> **将 workload skew 与 hardware variability 联合优化，而非分别处理。**

具体创新点如下：

- ✅ **执行时间平衡代替 token 平衡**  
  不再追求每个 GPU 处理相同数量的 token，而是通过建模每个 GPU 的实际性能 $ f(n) $，将高负载专家分配给高性能 GPU，低负载专家分配给低性能 GPU，以对齐各 GPU 的完成时间。

- ✅ **设备级性能建模（Per-GPU Performance Modeling）**  
  对每个 GPU 进行微基准测试，构建 token 数量到 kernel latency 的映射函数 $ f(n) $，捕捉个体性能差异。

- ✅ **动态漂移感知重校准（Drift-aware Recalibration）**  
  在线监控请求分布的变化（如 batch size、输入长度、prefill/decode 阶段切换），当 **cosine distance** 超过阈值时触发轻量级增量更新，避免频繁全量重排。

- ✅ **增量式专家重排（Incremental Placement Update）**  
  使用贪心交换策略最小化专家迁移开销，通常只需 5–30 次跨 GPU 交换即可恢复平衡，远低于完整重分配所需的 >200 次。

---

### 相比现有方法的优势

| 方法 | 是否考虑硬件差异 | 是否联合优化 | 动态适应能力 | 主要目标 |
|------|------------------|---------------|----------------|-----------|
| vLLM (contiguous) | ❌ | ❌ | ❌ | 内存连续性 |
| EPLB [12] | ❌ | ❌ | ⚠️ 固定周期重校准 | Token 平衡 |
| **ViBE (Ours)** | ✅ | ✅ | ✅ 漂移触发 + 增量更新 | **Execution-time 平衡** |

> 🔑 关键优势：ViBE 将硬件变异性从“效率杀手”转变为“优化杠杆”，实现更稳定的端到端性能。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Sonnet**：固定输入长度（1024 tokens）、固定输出长度（128 tokens），用于控制变量分析。
- **ShareGPT**：真实用户对话数据，平均输入 219.2 tokens，输出 200.8 tokens，具有高度可变性和路由波动。

---

### 实验设置
- **硬件平台**：
  - 单节点 8× AMD Instinct™ MI325X GPUs（TDP 1000W）
  - 部分实验扩展至 MI300X 系统及模拟大规模集群
- **软件栈**：
  - vLLM v0.14.2 + PyTorch v2.9.0 + ROCm v7.0
  - AITER 作为 attention/MoE kernel backend
- **模型配置**：
  - **DeepSeek-V3**：256 experts, FP8, 8-way EP
  - **Qwen-3-235B**：128 experts, FP8, 8-way EP
- **并行策略**：Hybrid TP (degree 8) + EP (degree 8)

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **TTFT (Time to First Token)** | 用户感知的响应延迟，关注 P50/P90/P99 |
| **TPOT (Time Per Output Token)** | 生成阶段吞吐量指标 |
| **SLO Attainment** | 请求满足延迟 SLA 的比例（如 TTFT ≤ 350ms） |
| **Goodput** | 单位时间内成功完成且符合 SLO 的请求数量 |
| **Kernel Latency Gap** | 最快与最慢 GPU 在 MoE kernel 上的执行时间差 |
| **Clock Frequency Spread** | GPU 实际运行频率的标准差，反映利用率一致性 |

---

### 基线方法对比
| 策略 | 描述 |
|------|------|
| **vLLM** | 默认连续专家划分（contiguous partitioning） |
| **EPLB [12]** | 基于路由频率的 token 平衡方法，忽略硬件差异 |
| **ViBE** | 本文方法，基于性能模型进行 execution-time 平衡 |

所有策略均离线生成 placement 并保持静态，以隔离 placement 策略本身的影响。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 提升幅度 | 场景说明 |
|------|----------|---------|
| **SLO Attainment** | ↑ **14%**（平均） | 可持续请求速率提升，DeepSeek-V3 达 12%，Qwen-3 达 15% |
| **P90 TTFT** | ↓ **up to 45%** | Sonnet 数据集上表现最佳 |
| **P99 TTFT** | ↓ **up to 30%** | 显著改善尾部用户体验 |
| **MoE Kernel Latency Gap** | ↓ 63.9% (vs vLLM), ↓19.6% (vs EPLB) | 更小的 straggler 影响 |
| **Barrier Idle Time** | ↓ **41%**（相比 EPLB） | 减少同步等待时间 |
| **Total Sync MoE Latency** | ↓ **35%** | 综合执行效率提升 |

---

### 与基线方法的对比结果

#### （1）SLO Attainment（图8）
- 在 Sonnet 上顺序稳定：`vLLM < EPLB < ViBE`
- 在 ShareGPT 上，EPLB 因未考虑硬件差异，在高负载下可能将热点专家分配给慢 GPU，导致性能倒退；而 ViBE 持续领先。

#### （2）端到端延迟（图9）
- 所有结果归一化为 vLLM 在最低 QPS 下的表现。
- 当系统进入 compute-bound 区域（高 QPS），ViBE 优势明显：
  - DeepSeek-V3：P90 TTFT 最多降低 45%
  - Qwen-3：因模型较轻，更容易饱和，ViBE 相对 EPLB 的 SLO frontier 扩展达 **15%**

#### （3）内核级性能（图10）
- **Latency Gap Boxplot**：ViBE 显著压缩了最快与最慢 GPU 的执行时间差距。
- **Frequency Distribution**：ViBE 下各 GPU 的 clock frequency 更集中，表明负载更均衡，无明显空转或拥塞。

---

### 消融实验结果

#### （1）动态重校准有效性（图11）
- **Cross-workload 测试**（SG→SN / SN→SG）：
  - 静态 placement 在 workload drift 后性能下降明显。
  - 自适应版本（Adaptive ViBE/EPLB）通过在线重校准恢复大部分性能损失。
  - 例如在 SG→SN 中，ViBE 的 90% SLO 支持 QPS 从 1.68 提升至 **1.80 QPS/GPU**。

#### （2）重校准开销（图12）
- 每次 rearrangement 引发短暂 TTFT 抖动（秒级），但很快恢复。
- 增量更新机制使迁移量减少一个数量级以上（~200 → ~20 次 swap）。

#### （3）敏感性分析（图13–14）
- 在人为引入电压-频率偏移（skewed system）后，ViBE 能自动识别慢 GPU 并减少其负载，进一步拉开与 EPLB 的差距。
- 即使在低 variability 系统（MI300X）中，ViBE 依然提供稳定增益。

#### （4）扩展性预测（图15）
- 模拟 80× MI300X 节点的大规模部署：
  - ViBE 在 16–32 GPU 规模时达到最优效果。
  - 超过 64 GPU 后，每 GPU 专家数过少，placement 灵活性下降，算法收敛。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Token 平衡 ≠ 时间平衡**  
   即使 token 完全均匀分布，硬件 variability 仍会导致高达 **7% 的 kernel 执行时间差异**（图6），成为尾延迟瓶颈。

2. ✅ **硬件 variability 是 first-order 问题**  
   在 MoE 推理中，尤其是 prefill 阶段接近 TDP 时，DVFS 导致的频率分化显著放大性能差异，必须显式建模。

3. ✅ **联合优化优于孤立处理**  
   ViBE 利用 workload skew 来抵消 hardware asymmetry，实现了“变劣势为优势”的设计哲学。

4. ✅ **动态漂移需智能响应**  
   固定周期重校准既浪费资源又反应迟钝；ViBE 的 drift-triggered + incremental 更新机制实现了高效自适应。

5. ✅ **收益随规模增加而增强**  
   随着 EP group size 增大，性能离散度累积效应更强，ViBE 的优化空间更大。

---

### 方法的局限性
- 🚫 **依赖准确的 per-GPU 性能建模**：若设备老化或冷却条件剧烈变化，需重新校准模型。
- 🚫 **增量更新仍引发短时抖动**：虽然已最小化，但在严格 SLA 场景下仍可能造成影响。
- 🚫 **当前聚焦单节点内 variability**：跨节点、跨机架的异构性尚未完全覆盖。
- 🚫 **未探索非均匀专家分配**：目前每个 GPU 分配相同数量专家，未来可研究弹性分配。

---

### 未来工作方向
- 🔮 **支持 rack-scale 系统的 co-design**：结合 TP grouping 与 EP placement，进行全局 variability-aware 并行策略设计。
- 🔮 **预测性重校准**：利用流量预测提前调整 placement，避免突发负载冲击。
- 🔮 **fault-tolerant placement**：将慢速 GPU 视为潜在故障点，主动规避关键路径。
- 🔮 **集成进编译器栈**：将 ViBE 思想嵌入 Triton 或其他 DSL 编译器，实现自动化部署。

---

> 💡 **总结一句话**：  
> ViBE 首次揭示了 **MoE serving 中 workload skew 与 hardware variability 的耦合效应**，并通过 **execution-time-aware placement + drift-resilient recalibration** 实现了更高效、更鲁棒的推理服务，在多个真实场景中显著提升了 SLO 满足率与用户体验。

</details>

---

### 4. [TwinQuant: Learnable Subspace Decomposition for 4-Bit LLM Quantization](https://arxiv.org/abs/2606.01556)

**Authors**: Haodong Wang, Junjie Liu, Zicong Hong, Qianli Liu, Jian Lin, Song Guo, Xu Chen  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.01556v1  

#### Abstract
4-bit quantization reduces the memory footprint and latency of large language model inference, but its aggressive precision reduction can severely degrade accuracy. Prior methods address this by decomposing each weight matrix into two components (e.g., via singular value decomposition) and quantizin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TwinQuant: Learnable Subspace Decomposition for 4-Bit LLM Quantization — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
4-bit 量化（W4A4）虽然能显著降低大语言模型（LLM）的内存占用和推理延迟，但会因权重和激活值的**重尾分布**（heavy-tailed）和**各向异性**（anisotropic）特性导致严重的精度下降。传统方法如 SVDQuant 虽通过低秩分解吸收异常值，但在主流 LLM 中存在以下问题：
- **奇异值衰减慢**：LLM 权重矩阵的奇异值谱衰减缓慢，小秩无法有效捕获所有异常值；
- **高开销困境**：增大秩可提升效果，但带来显著的额外存储和计算开销；
- **低秩组件难量化**：直接对 SVD 得到的低秩部分进行 4-bit 量化仍会产生严重误差。

### 🚀 提出的新方法：TwinQuant
TwinQuant 是一种面向 4-bit LLM 量化的新型框架，其核心是**可学习的子空间分解**（Learnable Subspace Decomposition），主要创新如下：

#### （1）**联合优化的双分支量化架构**
将每个权重矩阵 $ W $ 分解为：
$$
W \approx UV + R
$$
其中：
- $ UV $：低秩分量（low-rank component）
- $ R $：残差分量（residual component）

与固定 SVD 不同，TwinQuant **联合学习两个分量的表示空间**，使其更适应 4-bit 量化。

#### （2）**可学习的子空间变换**
引入两类可训练变换来重塑数值分布：
- **全局正交变换 $ Q \in \text{Stiefel} $**：作用于输入激活和残差，平滑动态范围；
- **层特定可逆变换 $ G \in \text{GL}(n,\mathbb{R}) $**：调整低秩因子 $ U, V $，减少方向偏斜。

这些变换在离线校准阶段学习，并可折叠进权重中，**不增加推理开销**。

#### （3）**融合双组分核函数（Fused Dual-Component Kernel）**
设计专用 CUDA 内核，实现：
- 在芯片上流水执行两阶段低秩 GEMM；
- 中间结果在线重量化并打包；
- 与残差路径合并后单次写回全局内存；
- 避免中间内存流量，极大提升效率。

---

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **精度保持** | 显著优于 RTN、SmoothQuant、QuaRot、SVDQuant 等，在 W4A4 下接近 FP16 表现 |
| **效率提升** | 融合内核减少 kernel launch 和 global memory 访问，端到端加速达 **1.8×** |
| **通用性强** | 在 LLaMA3 和 Qwen3 多个规模模型上均表现稳定 |
| **无需微调** | 属于 Post-Training Quantization（PTQ），无需反向传播或大规模训练 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **校准数据集**：从 `WikiText2` 随机采样 128 个 prompt，每条 2048 tokens，用于学习变换参数。
- **评估任务**：
  - **语言建模**：WikiText2 上的 Perplexity（PPL）
  - **零样本推理**（zero-shot accuracy）：
    - ARC-Challenge / ARC-Easy
    - HellaSwag
    - LAMBADA
    - PIQA
    - WinoGrande

使用 `lm-eval` 工具包统一评测。

---

### ⚙️ 实验设置
| 项目 | 设置 |
|------|------|
| **模型** | LLaMA3（3B, 8B）、Qwen3（4B, 8B, 14B, 32B） |
| **量化配置** | W4A4 / W4A8 / W4A16（weight-activation bitwidth） |
| **分组大小** | Group size = 128（group-wise quantization） |
| **低秩秩数** | 默认 $ r = 128 $ |
| **硬件平台** | <br>• Environment 1: RTX 4090 (24GB) + Xeon Gold 6430<br>• Environment 2: L20 (48GB) + Xeon Platinum 8457C |
| **推理序列长度** | 输入 1024 tokens，输出 256 tokens |

---

### 🆚 基线方法对比
共比较 8 种主流 PTQ 方法：
| 方法 | 类型 |
|------|------|
| **RTN** | Round-to-Nearest，无校准 |
| **GPTQ** | Hessian-aware 权重量化 |
| **AWQ** | 激活感知权重重缩放 |
| **SmoothQuant** | 通道级缩放平衡激活与权重 |
| **QuaRot** | 固定 Hadamard 变换去相关 |
| **SpinQuant** | 学习正交旋转优化量化 |
| **FlatQuant** | 学习仿射变换压平分布 |
| **SVDQuant** | SVD 分解 + 低秩补偿（FP16 低秩） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & A.3-A.5）

#### （1）**零样本平均准确率（↑越高越好）**
| 模型 | 方法 | W4A4 准确率 |
|------|------|------------|
| LLaMA3-8B | FP16 | 73.5% |
| LLaMA3-8B | SpinQuant | 69.4% |
| LLaMA3-8B | SVDQuant | 68.2% |
| LLaMA3-8B | **TwinQuant** | **70.9%** |
| Qwen3-32B | FP16 | 75.2% |
| Qwen3-32B | SpinQuant | 72.0% |
| Qwen3-32B | SVDQuant | 72.2% |
| Qwen3-32B | **TwinQuant** | **73.3%** |

✅ **TwinQuant 平均比 SVDQuant 高 0.9–2.7 个百分点，接近 FP16 性能（仅差 0.6–2.6 pts）**

#### （2）**WikiText2 Perplexity（↓越低越好）**
| 模型 | 方法 | W4A4 PPL |
|------|------|---------|
| LLaMA3-8B | FP16 | 8.6 |
| LLaMA3-8B | RTN | 92.9 |
| LLaMA3-8B | SmoothQuant | 19.4 |
| LLaMA3-8B | SpinQuant | 9.5 |
| LLaMA3-8B | SVDQuant | 12.5 |
| LLaMA3-8B | **TwinQuant** | **9.4** |
| Qwen3-32B | FP16 | 7.6 |
| Qwen3-32B | SpinQuant | 12.1 |
| Qwen3-32B | **TwinQuant** | **10.4** |

✅ **TwinQuant 在多个模型上实现了最低或接近最低的 PPL，远优于其他 W4A4 方法**

---

### 🔁 与基线方法对比结果
| 对比项 | 结果 |
|-------|------|
| vs. **RTN / SmoothQuant** | 后者在 W4A4 下崩溃（PPL > 1000），而 TwinQuant 保持稳定 |
| vs. **SpinQuant / FlatQuant** | 在多数模型上取得更高准确率和更低 PPL |
| vs. **SVDQuant**（rank=128） | 平均零样本准确率高 1.1–2.7 pts，PPL 降低最多达 3.1 |
| vs. **AWQ（W4A16）** | 在更严格的 W4A4 下仍达到相近甚至更好性能 |

---

### 🔍 消融实验结果（Table 3）

| 方法 | LLaMA3-8B Acc / PPL | Qwen3-8B Acc / PPL |
|------|---------------------|--------------------|
| FP16 | 73.5 / 8.6 | 71.6 / 9.7 |
| Naive 4-bit | 41.7 / 91.6 | 41.5 / 4188 |
| +Low-Rank (SVD) | 61.1 / 19.6 | 62.8 / 20.3 |
| +Hadamard | 64.9 / 12.4 | 66.0 / 16.3 |
| **TwinQuant** | **70.9 / 9.4** | **70.2 / 13.2** |

📌 **结论**：
- 单纯加低秩可大幅恢复性能；
- 固定 Hadamard 进一步改善；
- **可学习的 $ Q $ 和 $ G $ 变换带来最大增益**，验证了 learnable decomposition 的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LLM 权重不具备快速奇异值衰减**，传统 SVD-based 方法在小秩下效果有限；
2. **直接量化低秩分量会引入严重误差**，必须对其分布进行优化；
3. **可学习的子空间分解能显著降低 4-bit 量化误差**，通过联合优化 $ Q $ 和 $ G $ 实现“量化友好”的表示；
4. **融合内核设计至关重要**：避免中间内存访问使低秩路径高效可用；
5. **TwinQuant 在 W4A4 下实现近 FP16 精度**，同时获得高达 **1.8× 的端到端加速**。

---

### ⚠️ 方法的局限性
1. **当前主要针对 Dense LLM**：未验证在 MoE 架构或多模态模型上的适用性；
2. **依赖校准数据**：需要少量数据进行离线优化，可能影响隐私敏感场景部署；
3. **GPU 特定优化**：融合内核基于 Tensor Core 设计，迁移到其他硬件需重新适配；
4. **超长上下文行为未知**：在 ultra-long-context 场景下的表现尚未充分测试；
5. **训练开销不可忽略**：尽管属于 PTQ，但优化时间随模型增大显著上升（最大达 389 分钟）。

---

### 🔮 未来工作方向
1. 扩展至 **MoE 和多模态模型**；
2. 开发更轻量、数据高效的优化策略（如 zero-data 或合成数据校准）；
3. 推广融合内核至更多 **AI 加速器架构**（如 TPU、NPU）；
4. 探索 **动态秩选择机制**，根据不同层自适应分配资源；
5. 结合 **量化感知训练**（QAT）进一步压缩极限。

---

## ✅ 总结一句话
> **TwinQuant 通过可学习的子空间分解与系统级融合内核，在 W4A4 下实现了接近 FP16 的精度和最高 1.8× 的端到端加速，为高效 LLM 推理提供了新的 PTQ 范式。**

</details>

---

### 5. [Parallelizing Large-Scale Tensor Network Contraction on Multiple GPUs](https://arxiv.org/abs/2606.01852)

**Authors**: Feng Pan, Hanfeng Gu, Paul Springer, Xipeng Li  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.01852v1  

#### Abstract
Exact tensor network contraction underpins quantum circuit simulation, quantum error correction, combinatorial optimization, and many-body dynamics. The dominant parallelization strategy, slicing, scales exponentially and incurs redundant computation. We present a multi-GPU framework that instead di...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Parallelizing Large-Scale Tensor Network Contraction on Multiple GPUs*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统大规模 **Tensor Network (TN)** 收缩在多 GPU 上的并行化主要依赖 **slicing** 技术，即固定某些索引以降低内存峰值。然而，slicing 存在两个根本缺陷：
- **指数级冗余计算**：每增加一个 sliced 索引，子任务数量翻倍，导致大量重复计算。
- **无法突破单设备内存墙**：当中间张量过大时，仅靠 slicing 无法有效利用多 GPU 的聚合内存。

本文旨在解决这一系统级瓶颈：如何在保留精确收缩的前提下，高效地将大规模 TN 收缩分布到多个 GPU 上执行。

---

### 提出的新方法与核心思想

作者提出了一种全新的 **通信感知分布式收缩框架**，其核心是将中间张量显式地跨 GPU 分布，并通过精细调度最小化通信开销。该框架包含两大关键技术：

#### （1）**GEMM-Oriented Mode Reordering**
- 对给定的收缩路径进行离线重排序，使得每个 pairwise contraction 都能映射为高效的 **GEMM**（矩阵乘法）操作。
- 利用“剩余生命周期”（remaining lifetime）对模式（mode）排序：长寿命模式前置，短寿命（即将被收缩）模式后置。
- 结果：所有输入张量均处于 `[retained | reduced]` 形式，无需运行时转置（transpose），直接调用 cuTENSOR 的高性能 kernel。

#### （2）**Communication-Aware Mode Distribution Planning**
- 引入动态规划（Dynamic Programming, DP）策略，决定：
  - 哪些张量需要分布（distribution）
  - 哪些 mode 应被分区（partitioned）
  - 何时进行 **redistribution** 或 **gather**
- 成本模型综合考虑：
  - 局部计算时间（基于 `Bdev`, `Fdev`）
  - 通信时间（基于 `Bnet`, 消息延迟 `λ`, 块粒度 `nbik`, `sblk`）
- DP 自动选择在“张量体积谷值”处进行 redistribution，避免在大张量时发生高代价通信。

最终，整个流程由 **cuTENSORMp** 库实现：它接收分布计划，自动协调本地计算与跨进程通信（如 NCCL collectives），并支持流水线化的 **compute-communication overlap**。

---

### 相比现有方法的优势

| 方面 | Slicing（主流方法） | 本文方法（Distribution + Communication-Aware Scheduling） |
|------|---------------------|----------------------------------------------------------|
| 内存扩展性 | 有限，依赖冗余计算 | 可充分利用多 GPU 聚合内存（如 DGX H100 的 640GB） |
| 计算效率 | 指数级冗余 FLOPs | 显著减少总 FLOP 数（复杂度下降可达百万倍） |
| 通信控制 | 无通信，完全独立 | 显式建模通信成本，主动优化通信时机与粒度 |
| 扩展能力 | 多节点下仍受限于 slicing 开销 | 在 1024 GPU 上仍保持远超线性加速 |

> ✅ **本质区别**：  
> - **Slicing 是“空间换时间”**（更多任务换取更低内存）  
> - **Distribution 是“通信换计算”**（引入通信以大幅削减 FLOPs）

---

## 2. 核心实验方法和设置

### 使用的数据集与工作负载（Workloads）

共测试四类典型 TN 应用：

| 工作负载 | 描述 |
|--------|------|
| **Circuit (n60m24)** | Zuchongzhi 量子电路模拟（60 量子比特，深度 24） |
| **Hexagonal 8×8** | 六边形晶格上的量子多体动力学 |
| **Rectangular 49×20** | 矩形晶格多体动力学 |
| **Triangular 49×24** | 三角晶格多体动力学 |

此外还分析了 QEC（距离 7 表面码解码）和组合优化（King’s graph 独立集枚举）作为动机示例。

---

### 实验平台

- **硬件**：NVIDIA DGX H100 节点
  - 每节点 8× H100 GPU（80GB HBM3）
  - 单节点内：NVLink（900 GB/s per GPU）
  - 多节点间：InfiniBand（400 Gb/s per GPU）
- **软件栈**：
  - 路径查找器：cotengra 类启发式算法（固定约 1 小时预算）
  - 分布执行引擎：**cuTENSORMp**
  - 通信后端：NCCL（自动选择 NVLink/NVSwitch/InfiniBand RDMA）

---

### 评估指标

| 指标 | 定义 | 说明 |
|------|------|------|
| `Projected Full Speedup` $ S_p $ | $ \frac{t_1 \cdot 2^{b_1}}{t_p \cdot 2^{b_p}} $ | 包含 slicing 并行性的端到端加速比 |
| `Extra Speedup` $ E_p $ | $ \frac{S_p}{P} $ | **核心指标**：超出理想 slicing 的额外加速（衡量分布有效性） |
| `Complexity Reduction` $ R_p $ | $ \frac{C_{t,1}}{C_{t,P}} $ | 仅考虑 FLOP 减少的理想理论上限（不含通信） |
| `TFLOP/s per GPU` | 实测每 GPU 性能 | 衡量实际利用率 |

> 📌 注意：由于路径优化本身是 NP-hard，不同 GPU 数下的最优路径质量不一致，因此结果呈现非单调趋势。

---

### 基线方法对比

- **Baseline**：理想化的 **embarrassingly parallel slicing**
  - 假设每个 slice 可完美分配至各 GPU，无任何通信
  - 加速比应等于 GPU 数量 $ P $
- **本文方法 vs Baseline**：额外加速 $ E_p > 1 $ 即表示优于纯 slicing

---

## 3. 主要实验结果和性能指标

### 单节点性能（8× H100，NVLink）

| Workload | Projected Speedup | Extra Speedup $ E_p $ | Complexity Reduction | TFLOP/s per GPU |
|---------|--------------------|--------------------------|------------------------|------------------|
| Circuit (n60m24) | 148× | **18.5×** | 18.5× | 28.1 |
| Hexagonal 8×8 | 1,383× | **172.9×** | 197.8× | 31.6 |
| Rectangular 49×20 | 75× | **9.4×** | 9.3× | 32.9 |
| Triangular 49×24 | 56× | **7.0×** | 7.4× | 31.8 |

✅ **关键观察**：
- 所有任务在单节点上均实现 **7–173× 的额外加速**
- 实际加速接近理论 FLOP 减少（达到 **87–101%**），表明 **NVLink 高带宽使通信几乎可忽略**

---

### 多节点扩展至 1024 GPU（InfiniBand）

| Workload | Per-slice Runtime (s) | Sliced Bonds $ b_p $ | Projected Speedup | Extra Speedup $ E_p $ | Complexity Reduction |
|---------|------------------------|------------------------|--------------------|--------------------------|------------------------|
| Circuit (n60m24) | 20.19 | 20 | 42.8K× | **41.8×** | 418× |
| Hexagonal 8×8 | 113.27 | 6 | 69.5M× | **67,869×** | 1.49M× |
| Rectangular 49×20 | 34.70 | 14 | 221.2K× | **216.0×** | 3,154× |
| Triangular 49×24 | 12.19 | 14 | 135.7K× | **132.6×** | 986× |

✅ **惊人结果**：
- 在六边形多体动力学上，**额外加速达 67,869×**！
- slicing 索引从 37 降至仅 6，极大减少了子任务数（$ 2^{37} \to 2^6 $）
- 尽管 InfiniBand 带宽远低于 NVLink，但方法依然有效

---

### 消融实验与关键发现（隐含分析）

虽然未明确列出消融表，但从设计中可推断以下关键验证：

- **DP 规划 vs 启发式延迟 redistribution**：
  - 若推迟 redistribution 至必须时刻（如某 mode 即将被收缩），可能发生在张量极大时（如 256GB），且 stride 不利，导致千万级小块传输，陷入 **latency-bound**。
  - DP 主动在“体积谷值”（如 16–32GB）提前重分布，通信总量仅占总移动量 **4.6%**（见 Fig. 5）。

- **前缀分布 + 生命周期排序的协同效应**：
  - 分布最长寿命 mode → 这些 mode 位于最外层 → 每个 GPU 分得连续内存块 → 最大化通信粒度，避免碎片化。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **Distribution 比 Slicing 更具扩展潜力**：
   - Slicing 导致指数冗余，而 Distribution 通过共享内存显著降低 FLOPs。
   - 在前沿 TN 问题中，**通信换计算** 是更优范式。

2. ✅ **通信是瓶颈而非计算**：
   - 在 NVLink 上，通信开销极低，额外加速几乎等于理论 FLOP 减少；
   - 在 InfiniBand 上，通信成为主导成本，但仍能获得数十至数万倍额外加速。

3. ✅ **自动化调度至关重要**：
   - 手工设计分布策略难以应对复杂 use-chain；
   - 基于 DP 的通信感知规划能自动发现最优 redistribution 时机。

4. ✅ **通用性强**：
   - 方法适用于量子电路、QEC、组合优化、多体动力学等多种 TN 应用；
   - 仅依赖收缩树结构、mode 生命周期和形状信息，**不依赖领域知识**。

---

### 方法的局限性

| 局限 | 说明 |
|------|------|
| **路径优化质量影响整体性能** | 当前框架假设路径已固定；若路径本身不佳，分布也无法弥补。路径搜索与分布调度尚未联合优化。 |
| **高度依赖高速互连** | 在低带宽网络（如普通 Ethernet）上，通信开销可能压倒收益。 |
| **目前仅支持规则分布（prefix-based）** | 无法处理非连续或不规则分块（如 jagged partitions）。 |
| **未涵盖混合精度或量化** | 如 [Fu et al. SC24] 中使用的低精度加速未集成进来。 |

---

### 未来工作方向

1. **联合路径搜索与分布规划**：
   - 当前路径优化未考虑分布可行性；
   - 未来可构建统一成本模型，同时优化 contraction path 和 distribution plan。

2. **扩展至 MNNVL（Multi-Node NVLink）架构**：
   - 如 NVIDIA GB200 NVL72 提供跨节点 NVLink 级带宽（1.8 TB/s per GPU）；
   - 预计将进一步释放分布潜力。

3. **支持异构内存系统（HBM + DRAM + SSD）**：
   - 结合 offloading 技术，处理超大规模 TN。

4. **集成 slicing 与 distribution 的混合策略**：
   - 对部分索引 slicing，其余采用分布，实现更灵活的权衡。

---

> 🔚 **总结一句话**：  
> 本文打破了传统 slicing 的思维定式，首次实现了 **通信感知的大规模 TN 分布式收缩**，在 1024 GPU 上取得了高达 **6.7万倍** 超越理想 slicing 的额外加速，为模拟前沿量子系统提供了新的系统级解决方案。

</details>

---

### 6. [ART: Attention Run-time Termination for Efficient Large Language Model Decoding](https://arxiv.org/abs/2606.00024)

**Authors**: Chen Qiu, Guozhong Li, Panos Kalnis  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.00024v1  

#### Abstract
Long-context decoding in Large Language Models (LLMs) is severely constrained by the memory bandwidth required to fetch the extensive Key-Value (KV) cache. Most existing KV management methods rely on key-only pruning before decoding, despite the evidence that attention outputs depend jointly on keys...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ART: Attention Run-time Termination for Efficient Large Language Model Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Large Language Models (LLMs)** 的长上下文解码过程中，**Key-Value (KV) cache** 的内存带宽需求随序列长度线性增长，成为推理延迟和内存消耗的主要瓶颈。现有的 **KV pruning** 方法大多基于 **key-only** 的重要性估计（如 attention score），忽略了 **value (V)** 向量对最终输出的实际影响。这导致一些低 attention score 但高 value 贡献的 token 被错误丢弃，损害模型准确性。

此外，已有尝试引入 value 信息的方法通常依赖额外预测器、离线分析或预计算，带来显著推理开销，难以实用化。

### 提出了什么新方法或新思路
本文提出 **Attention Run-time Termination (ART)**，一种轻量级、运行时动态终止机制，通过监控 **attention 输出的累积过程** 来判断是否可以提前终止 KV block 的访问。

- **核心思想**：利用现代 **FlashAttention-style kernels** 在块状执行（block-wise execution）中自然暴露的中间 attention 输出状态，实时监测其变化。
- **动态终止**：当后续 KV block 对 attention 输出的增量贡献变得可忽略时，立即终止计算和内存访问，跳过剩余 KV blocks。
- **output-aware**：不同于静态的 key-based pruning，ART 是输出感知的，隐式地捕捉了 **keys 和 values 的联合影响**。

### 相比现有方法的优势
- **正交性（Orthogonal）**：ART 不替代现有 KV cache 管理方法，而是作为一层轻量级控制机制，可无缝集成到任何 KV pruning 策略中（如 StreamingLLM, SnapKV, PyramidKV）。
- **零预估开销**：无需额外预测器或离线分析，直接利用 kernel 执行中的中间状态，计算开销极低。
- **高效率**：在大 batch size 下实现高达 **20% 的生成吞吐量提升**。
- **高保真度**：在 LongBench 上保持与基线相当的准确性，平均得分仅下降约 0.8%。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **LongBench (Bai et al., 2024)**：作为主要评测基准，包含 21 个数据集，覆盖 6 大任务类别：
  - Single-document QA
  - Multi-document QA
  - Summarization
  - Few-shot Learning
  - Synthetic Tasks
  - Code Completion
- 数据集支持中英双语，上下文长度从数千词到数万字符不等，适合评估长上下文能力。

### 实验设置和评估指标
- **模型**：主实验使用 **Mistral-7B-Instruct-v0.3**；扩展实验使用 **Llama-3.1-70B-Instruct**。
- **硬件**：单张 **NVIDIA A100-SXM4 80GB GPU**。
- **评估阶段**：聚焦于 **decoding 阶段**，此时 KV cache 管理是主要瓶颈。
- **评估指标**：
  - **Decode TPOT (Time Per Output Token, ms/token)**：衡量解码延迟。
  - **Generation Throughput (tokens/s)**：衡量系统吞吐量，尤其关注不同 batch size 下的表现。
  - **LongBench Score**：综合评估生成质量与长上下文理解能力。
  - **FlashAttention Kernel Time**：隔离 ART 对核心 attention kernel 的影响。

### 基线方法对比
- **Baseline**：Full KV caching（保留全部 KV cache）
- **StreamingLLM (Xiao et al., 2024)**：基于 attention sink 的流式缓存策略。
- **SnapKV (Li et al., 2024)**：基于观察窗口的重要性评分压缩。
- **PyramidKV (Cai et al., 2025)**：基于金字塔信息漏斗的动态压缩。
- 所有方法均测试了 **80% (0.8)** 和 **20% (0.2)** 的 KV cache 保留率，并与 ART 结合进行对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **生成吞吐量提升**：在大 batch size 下，ART 相比 state-of-the-art 基线实现了 **20% 更高的 generation throughput**。
- **延迟降低**：在多种配置下，ART 显著降低了 **TPOT**，尤其在高压缩率（如 20%）下效果更明显。
- **准确性保持**：
  - 在 Full KV 下，ART 的 LongBench 平均得分达到基线的 **99.2%**。
  - 与其他方法结合后，准确率波动极小，多数情况下得分变化小于 ±1。

### 与基线方法的对比结果
| 方法组合 | TPOT 降低 | Throughput 提升 | 准确率变化 |
|---------|----------|----------------|------------|
| Baseline + ART | ↓ 显著 | ↑ ~15-20% | -0.8% |
| StreamingLLM + ART | ↓ 显著 | ↑ ~18% | -0.09% |
| SnapKV + ART | ↓ 显著 | ↑ ~20% | -0.80% |
| PyramidKV + ART | ↓ 显著 | ↑ ~19% | -0.57% |

> 图表显示，在所有 batch size 和保留率下，**ART + 基线** 均优于单独基线，且优势随 batch size 增大而增强。

### 消融实验结果
消融研究验证了 ART 三个核心组件的必要性：

| 方法变体 | QA Score ↓ | Sum Score ↓ | Overall Score ↓ | Kernel Time ↓ | 说明 |
|--------|-----------|-----------|---------------|----------------|------|
| ART (完整) | — | — | 45.91 | 0.633ms | 基准 |
| w/o `dscale` | -20.8 | -5.22 | **-26.57** | -78.7% | 过早终止，严重损精度 |
| w/o `ddirection` | -0.26 | -0.15 | **-2.04** | -0.8% | 方向未控，鲁棒性下降 |
| w/o `patience` | -2.0 | -0.44 | **-4.37** | -20.1% | 抗抖动能力弱，误判多 |

- **`dscale`** 是正确性的关键保障。
- **`ddirection`** 提升收敛检测的鲁棒性。
- **`patience`** 机制有效过滤瞬时波动，防止误终止。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **attention 输出具有早期稳定特性**：在大多数情况下，attention 输出在处理完部分 KV blocks 后即趋于稳定，后续计算为冗余。
2. **output-aware 动态终止优于静态 pruning**：基于实际输出变化的运行时决策，能更准确地识别冗余计算，兼顾效率与精度。
3. **ART 是通用且轻量的加速模块**：其设计与现有 KV cache 管理方法正交，可即插即用，带来一致的性能增益。
4. **大 batch size 下收益更大**：由于 ART 的开销恒定而节省随 batch 线性增长，因此在高并发场景下优势更显著。

### 方法的局限性
- **依赖 FlashAttention 架构**：目前实现基于 FlashAttention 的 block-wise 执行模式，对其他 attention 实现方式适配需调整。
- **对极短上下文增益有限**：在短序列中，KV cache 本身较小，提前终止机会少。
- **超参数敏感性**：虽然整体鲁棒，但 `dscale` 等阈值需合理设置以平衡速度与精度（见 sensitivity analysis）。
- **在超大规模模型中相对增益减弱**：如 Llama-3.1-70B 实验所示，因 FFN 等层成为新瓶颈，attention 优化的相对占比下降。

### 未来工作方向
- **自适应稳定性阈值**：探索 per-layer 或 dynamic threshold，根据上下文内容自动调节终止条件。
- **与硬件反馈机制结合**：结合 GPU 内存带宽监控等硬件信号，实现更智能的运行时调度。
- **扩展至训练场景**：探索 ART 在 long-sequence training 中的应用潜力。
- **支持更多 attention 变体**：将 ART 推广至稀疏 attention、linear attention 等其他高效 attention 形式。

--- 

> **总结**：ART 提出了一种新颖的 **output-aware, run-time termination** 思路，通过监控 attention 输出的收敛性来动态跳过冗余 KV 访问，在几乎不损失精度的前提下显著提升了 LLM 长上下文解码效率，是一项实用性强、易于集成的系统级优化。

</details>

---

### 7. [HeLoCo: Efficient asynchronous low-communication training under data and device heterogeneity](https://arxiv.org/abs/2606.00271)

**Authors**: Abdullah Al Asif, Patrick Diem, Juan Pablo Mu\~noz, Felix Wolf, Ali Jannesari, Arya Mazaheri  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.00271v1  

#### Abstract
Distributed Low-Communication (DiLoCo) training reduces communication overhead by allowing workers to perform multiple local optimization steps before sending pseudo-gradients to a global outer update. Its asynchronous variant further improves hardware utilization by removing synchronization barrier...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# HeLoCo: Efficient asynchronous low-communication training under data and device heterogeneity  
**论文核心总结**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
在 **asynchronous low-communication (DiLoCo)** 分布式训练中，存在两个关键挑战：
- **Gradient staleness**：异步更新导致部分 worker 返回基于过时模型状态的 **pseudo-gradient**，这些 stale 更新可能与当前全局优化方向不一致。
- **Data and device heterogeneity**：当 worker 训练速度不同（device heterogeneity）且本地数据为 **non-IID** 时，staleness 会加剧方向冲突，严重影响收敛性和模型质量。

现有方法（如 asynchronous Nesterov、MLA）对整个 pseudo-gradient 进行统一校正，忽略了不同参数块（tensor blocks）内部可能存在部分对齐、部分冲突的情况，导致校正过于粗粒度。

---

### **提出了什么新方法或新思路**
提出 **HeLoCo (Heterogeneity-aware Low-Communication training)**，一种**方向感知的细粒度校正方法**，其核心思想是：
- 利用 **outer momentum** 作为当前全局优化轨迹的参考方向。
- 在 **tensor block 级别**（如权重矩阵、偏置向量）进行校正，而非对整个 pseudo-gradient 统一处理。
- 引入 **look-ahead 初始化**：在 worker 开始本地训练前，使用 momentum 预测的未来模型状态进行初始化，减少初始位置偏差。

具体校正策略分为三类：
1. **高对齐块（$c_b \geq c_{ok}$）**：保留不变。
2. **反向冲突块（$c_b < 0$）**：仅衰减与 momentum 方向相反的分量。
3. **弱对齐块（$0 \leq c_b < c_{ok}$）**：平滑地向 momentum 方向旋转，保持模长不变。

---

### **相比现有方法的优势**
- **更精细的控制**：避免“一刀切”式校正，保留有用信息，只修正冲突部分。
- **更强的鲁棒性**：在系统异构（worker 速度差异大）和数据异构（non-IID）并存场景下表现显著优于 baseline。
- **无需增加通信开销**：仅需访问已有 momentum buffer，额外计算成本为 $O(d)$，可忽略不计。

---

## 2. 核心实验方法和设置

### **使用的数据集**
- **mC4** 多语言数据集，用于训练 **TinyGPT** 模型（15M 参数）。
- 每种语言代表一个独立的数据域（domain），模拟 **non-IID** 场景。

### **实验设置**
- **模型架构**：decoder-only Transformer（TinyGPT 风格）。
- **训练模式**：asynchronous DiLoCo，每个 worker 执行 H 步本地 SGD 后返回 pseudo-gradient。
- **异构模拟**：
  - **设备异构**：设定不同 worker 的每步耗时（1s, 2s, 6s, 15s），造成更新频率和 staleness 差异。
  - **数据异构**：将不同语言数据分配给固定 worker，形成 non-IID 分布。
- **评估指标**：
  - **Validation loss**（主指标）
  - 固定 **token budget** 下的 loss
  - 固定 **wall-clock time** 下的 loss

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **async-Nesterov** | 异步 DiLoCo + Nesterov 动量 |
| **async-MLA** | 异步 DiLoCo + Momentum Look-Ahead 校正 |
| **sync-Nesterov** | 同步 DiLoCo，无 staleness |
| **DyLU** | 动态调整本地步数以平衡参与度 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**
| 场景 | HeLoCo 提升幅度 |
|------|------------------|
| **固定 token 预算** | 相比 async-MLA 最高提升 **7.5%**（loss 下降） |
| **固定 wall-clock 时间** | 相比 async-MLA 最高提升 **3.3%** |
| **严重异构 + non-IID** | 相比 sync-Nesterov 最高提升 **22.1%**（时间效率优势） |

> 注：提升指 validation loss 更低，性能更好。

---

### **与基线方法的对比结果**
- 在 **heterogeneous + non-IID** 设置下（图2c）：
  - HeLoCo 最终 loss：**7.91**
  - MLA：7.96
  - Nesterov：8.29
  → 显示 HeLoCo 显著缓解了 stale 更新带来的负面影响。

- 在极端异构配置 `(1,15,15,15,15)` 中（表1）：
  - HeLoCo 在多数情况下仍优于 async-MLA 和 async-Nesterov。
  - 尽管 sync-Nesterov 在 token 效率上占优，但其 wall-clock 时间极长（受最慢 worker 拖累），而 HeLoCo 在时间效率上全面领先。

- **按语言分析**（图3）：
  - 快速 worker（English, staleness=0.72）：HeLoCo 改进微弱（7.91 vs 7.93）
  - 慢速 worker（German, staleness=14.38）：改进达 **0.13** loss 单位
  → 说明 HeLoCo 对高 staleness worker 的校正效果更明显。

---

### **消融实验结果**
- **DyLU 对比**：虽然 DyLU 能降低 staleness variance，但在相同设置下，**HeLoCo without DyLU 仍优于 HeLoCo + DyLU**，表明保持充分本地训练 + 精细校正 比 减少本地训练 更有效。
- **同步 vs 异步**：即使在 low-staleness 场景，HeLoCo 仍略优于 MLA，说明 per-block correction 在 non-IID 数据下本身就具有价值。

---

## 4. 关键结论和发现

### **主要发现**
1. **Staleness 不是均匀问题**：同一个 stale pseudo-gradient 内部，不同 tensor blocks 与当前优化方向的对齐程度差异很大。
2. **tensor-level correction 更有效**：相比全局校正，基于 outer momentum 的 block-wise 校正能更精准地保留有益梯度、修正有害分量。
3. **HeLoCo 在异构环境中优势显著**：尤其在 **device heterogeneity + non-IID data** 并存时，性能远超现有异步方法，甚至超越同步训练的时间效率。
4. **look-ahead 初始化 + momentum-guided correction 协同增益**：两者结合有效缩小了本地训练起点与全局模型之间的“轨迹差距”。

---

### **方法的局限性**
- 在 **极端 staleness**（如 staleness > 15）场景下，部分 pseudo-gradient 可能已完全不可靠，此时校正不如直接丢弃。
- 当前方法未动态调整校正强度，对极度 stale 的更新缺乏过滤机制。
- 实验规模较小（5 workers, 15M 模型），尚未验证在更大规模 LLM 训练中的可扩展性。

---

### **未来工作方向**
1. **自适应 stale gradient 过滤机制**：结合 staleness 程度和 alignment 统计量，动态决定是否校正、衰减或丢弃更新。
2. **扩展到更大模型和更多 worker**：在千卡级别集群上测试 HeLoCo 的可扩展性和稳定性。
3. **与其他 low-communication 技术结合**：如与 **DiLoCoX**（压缩）、**NoLoCo**（去中心化）等框架集成，构建更高效的分布式训练系统。
4. **理论分析**：提供 HeLoCo 在非凸、non-IID 设定下的收敛性证明。

---

> **总结一句话**：  
> HeLoCo 通过引入 momentum-guided 的 tensor-level 校正机制，在不增加通信成本的前提下，显著提升了异步低通信训练在 **data and device heterogeneity** 下的稳定性和效率，为构建全球分布式、松耦合的大模型训练系统提供了新思路。

</details>

---

### 8. [MindZero: Learning Online Mental Reasoning With Zero Annotations](https://arxiv.org/abs/2606.00240)

**Authors**: Shunchi Zhang, Jin Lu, Chuanyang Jin, Yichao Zhou, Zhining Zhang, Tianmin Shu  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.00240v1  

#### Abstract
Effective real-world assistance requires AI agents with robust Theory of Mind (ToM): inferring human mental states from their behavior. Despite recent advances, several key challenges remain, including (1) online inference with robust uncertainty updates over multiple hypotheses; (2) efficient reaso...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：MindZero: Learning Online Mental Reasoning With Zero Annotations**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文旨在解决**现实世界中AI助手进行在线心智推理（online mental reasoning）所面临的三大挑战**：
1. **缺乏真实心智状态标注数据**：在真实场景（如家庭、网页辅助）中，获取人类信念、目标等心智状态的标注成本极高且不可靠。
2. **需要实时、高效的推理能力**：传统基于模型的推理方法（如Bayesian inverse planning, BIP）虽准确但计算开销大，难以支持实时协作。
3. **需稳健地处理不确定性**：用户意图随时间动态变化，系统必须能持续更新多个假设并管理其置信度。

### **提出了什么新方法或新思路**
作者提出 **MindZero** —— 一种**无需心智状态标注的自监督强化学习框架（Self-Supervised Reinforcement Learning, SSRL）**，用于训练多模态大语言模型（MLLMs）执行高效且鲁棒的在线心智推理。

- **核心思想**：将心智推理建模为一个**解释一致性优化问题**。模型被奖励生成能够最大化观测行为似然性的心理状态假设（即“解释”用户行为），而非直接预测动作。
- **训练机制**：
  - 模型输出一组心智假设 $ m $ 及其概率分布 $ q $。
  - 奖励由两部分构成：
    - 行动似然（Action Likelihood）：由规划器或LLM估计给定心智假设下观察到的行为的概率。
    - 心智先验（Mental Prior）：由LLM判断该假设是否符合常识（如“把苹果放进洗碗机”概率低）。
  - 总奖励为加权对数似然减去熵正则项，鼓励探索多样且合理的假设。

- **推理阶段**：训练后的MindZero可在**单次前向传播（single-pass inference）** 中完成推理，实现快速响应。

### **相比现有方法的优势**
| 方法类型 | 代表 | 缺陷 | MindZero优势 |
|--------|------|------|-------------|
| Prompting-based | Chain-of-Thought | 易犯系统性错误，无法处理复杂行为 | 内化了结构化推理模式 |
| Model-based | BIP + LLM (e.g., AutoToM) | 推理时搜索空间大，速度慢、成本高 | 单次推理，速度快，适合实时应用 |
| Learning-based | 监督训练ToM模型 | 需要大量标注数据，扩展性差 | **完全无标注训练**，可扩展性强 |

> ✅ **核心优势**：**兼具模型驱动方法的鲁棒性和小规模LLM的效率**，实现了高质量、低延迟、无需标注的心智推理。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **GridWorld Domain**
   - **Construction Environment** (Jha et al., 2024)：二维网格世界，人类代理需搬运特定颜色物体配对。
   - 包含视觉输入（图像）和动作轨迹。
2. **Household Domain**
   - **MMToM-QA** (Jin et al., 2024)：多模态心智推理问答基准，涉及人物在家庭环境中寻找物品的信念与目标。
   - **Online Watch-And-Help (O-WAH)** (Puig et al., 2023)：具身环境下的主动协助任务，在VirtualHome模拟器中运行。

### **实验设置和评估指标**
#### **任务类型**
- **Question Answering (QA)**：直接回答关于用户心智状态的问题。
- **Proactive Assistance**：作为助手，实时推断用户目标并主动提供帮助（如提前拿取餐具）。

#### **评估指标**
| 任务 | 主要指标 | 定义 |
|-----|---------|------|
| QA | Accuracy ↑ | 正确回答心智相关问题的比例 |
| Proactive Assistance | Speedup ↑ | $ \text{speedup} = \frac{T_{\text{human}} - T_{\text{collab}}}{T_{\text{human}}} $，表示任务完成时间缩短比例 |
| 推理效率 | TFLOPs ↓ / Inference Cost | 浮点运算量，衡量计算开销 |

#### **基线方法对比**
| 类别 | 基线方法 |
|------|--------|
| **Base Models** | Qwen3-VL-4B, Qwen3-VL-8B, Llama-3.1-8B, Llama-3.2-3B, Qwen3-4B |
| **Large Models (Zero-Shot)** | GPT-5.2, Gemini-3-Flash, Qwen3-235B-A22B |
| **Test-Time Scaling Methods** | ThoughtTracing (Kim et al., 2025), AutoToM (Zhang et al., 2025) |
| **随机对照** | Random Goal |

> ⚠️ 注意：ThoughtTracing 和 AutoToM 因推理慢未参与 Proactive Assistance 对比。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) Question Answering 结果**
| 数据集 | 方法 | Accuracy | TFLOPs |
|-------|------|----------|--------|
| **GridWorld QA** | Qwen3-VL-4B (Base) | 37.7% | 3.6 |
| | **MindZero w/ Qwen3-VL-4B** | **95.0%** | 3.6 |
| | AutoToM w/ Qwen3-VL-4B | 49.3% | 344.4 |
| | **提升倍数** | **+2.5× 准确率，成本仅为1%** | |
| **Household QA** | Qwen3-4B (Base) | 42.8% | 10.9 |
| | **MindZero w/ Qwen3-4B** | **72.7%** | 13.1 |
| | ThoughtTracing w/ Qwen3-4B | 54.5% | 291.2 |
| | AutoToM w/ Qwen3-4B | 54.7% | 177.5 |
| | **提升倍数** | **+1.7× 准确率，成本仅约1/20** | |

> ✅ MindZero 在两种QA任务上均显著超越所有基线，尤其在效率上碾压test-time scaling方法。

#### **(2) Proactive Assistance 结果**
| 环境 | 方法 | Speedup | TFLOPs |
|------|------|---------|--------|
| **GridWorld** | Qwen3-VL-4B (Base) | 1.4% | 151.7 |
| | **MindZero w/ Qwen3-VL-4B** | **23.0%** | 161.4 |
| | GPT-5.2 / Gemini-3-Flash | 0.0% | Proprietary |
| **Household** | Qwen3-4B (Base) | 2.3% | 213.1 |
| | **MindZero w/ Qwen3-4B** | **19.1%** | 201.2 |
| | Gemini-3-Flash | 17.7% | Proprietary |

> ✅ MindZero 在两个主动协助任务中均取得最佳性能，**大幅优于基础模型和大型闭源模型**。

#### **(3) 消融实验（Ablation Study on Qwen3-4B）**
| 消融条件 | Speedup | 相比完整模型下降 |
|--------|--------|----------------|
| **完整 MindZero** | 19.1% | — |
| w/o Prior Modeling | 17.0% | -2.1% |
| w/o Multiple Hypotheses | 10.3% | -8.8% |
| w/o Entropy Bonus | 5.2% | -13.9% |

> 🔍 发现：
> - **熵正则项（Entropy Bonus）最关键**：防止过早收敛，维持假设多样性。
> - **多假设维护（Multiple Hypotheses）至关重要**：允许系统延迟决策直到证据充分。
> - **显式先验建模有效抑制不合理假设**，避免“奖励黑客”。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **心智推理可以作为一种自监督技能来学习**：MindZero 证明了仅通过行为数据即可训练出强大的ToM能力，无需任何心智状态标注。
2. ✅ **小规模MLLM经适当训练可超越大型模型**：MindZero 使用 Qwen3-4B 等小型开源模型，在准确性和效率上全面超越 GPT-5.2、Gemini-3-Flash 等超大规模模型。
3. ✅ **单次推理可内化复杂的模型驱动推理过程**：通过SSRL训练，模型将原本需多次迭代的BIP推理“蒸馏”进一次前向传播中。
4. ✅ **不确定性建模是主动协助成功的关键**：MindZero 能随时间逐步提高目标预测准确性（见Figure 5），从而做出更及时有效的协助决策。

### **方法的局限性**
1. ❌ **不支持递归心智推理（recursive reasoning）**：当前框架未建模“我认为你认为我……”这类嵌套推理。
2. ❌ **长序列输入带来token压力**：随着任务进行，历史轨迹增长，输入长度线性增加，影响可扩展性。
3. ❌ **依赖外部奖励模型的质量**：行动似然和心智先验依赖LLM或规划器估算，若其不准会影响训练信号。

### **未来工作方向**
1. 🔄 扩展至**多智能体递归心智推理**训练。
2. 💡 设计更高效的模型结构以应对**长输入序列挑战**。
3. 🔐 探索**公平性、问责制与隐私保护**的心智推理模型，推动伦理AI发展。

---

> **Impact Statement**：  
> MindZero 为构建**可规模化、可部署、真正理解人类意图的现实世界AI助手**提供了新范式。它弥合了**可解释性、鲁棒性与推理效率**之间的长期鸿沟，有望广泛应用于智能家居、数字服务、人机协同等领域。同时，作者强调需警惕滥用风险（如操纵、监控），倡导透明、知情同意与负责任部署。

</details>

---

### 9. [Threshold-Based Exclusive Batching for LLM Inference](https://arxiv.org/abs/2606.00516)

**Authors**: Weifang Zhang, Yuzhou Nie, Bowen Pang, Guangrui Ma, Shining Wu  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.00516v1  

#### Abstract
Mixed batching (MB)--interleaving prefill and decode in a single batch--has become the standard scheduling strategy for large language model (LLM) inference due to its efficiency in maximizing compute and memory utilization. However, through controlled experiments, we find that prefill-decode interf...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Threshold-Based Exclusive Batching for LLM Inference

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **Large Language Model (LLM)** 推理中的调度策略进行了深入研究，挑战了当前主流的 **Mixed Batching (MB)** 调度范式的普遍优越性假设。作者发现，在特定硬件条件下，**Mixed Batching** 并非总是最优选择。

具体而言，论文揭示了以下核心问题：
- **Prefill-Decode Interference**：在混合批处理中，prefill 阶段（计算密集型）和 decode 阶段（内存带宽密集型）在同一 GPU 上并行执行会因资源竞争而产生干扰。
- **硬件依赖性**：这种干扰的程度高度依赖于 GPU 的 **memory bandwidth**。在带宽受限的 GPU 上，这种干扰会显著增加每步的边际成本，从而降低整体吞吐量。

### 提出的新方法和新思路
为解决上述问题，论文提出了一套基于阈值的优化框架，其核心创新点如下：

1.  **理论分析与闭式条件 (Closed-Form Condition)**：
    - 推导了一个**闭式表达式**，用于判断 **Exclusive Batching (EB)** 和 **Mixed Batching (MB)** 在何种条件下性能更优。该条件由 **边际成本差距 (marginal-cost gap)** 和 **摊销固定成本优势 (amortized fixed-cost advantage)** 共同决定。

2.  **自适应独占批处理调度器 (Adaptive EB Scheduler, EB(k\*))**：
    - 提出了一个名为 `EB(k*)` 的自适应调度器，它能在线动态计算最优的相位切换阈值 `k*`。
    - 该调度器结合了渐近最优的**相位切换阈值 (phase-switching thresholds)** 和具有概率性 OOM 保证的**内存安全批大小 (memory-safe batch sizing)**。

3.  **混合调度器 EB+**：
    - 设计了一个混合调度器 **EB+**，它能够在线应用上述的“交叉条件”（crossover condition），在 `EB` 和 `MB` 之间进行动态切换。
    - 这种设计无需人工干预，即可在不同流量模式下自动选择最优策略。

### 相比现有方法的优势
- **更高的吞吐量**：在带宽受限的 GPU 上，`EB(k*)` 可实现高达 **41.9%** 的吞吐量提升。
- **更强的鲁棒性**：`EB+` 能够在非平稳流量（如请求分布或并发数发生突变）下，始终达到最高或接近最高的吞吐量。
- **无需额外硬件**：`EB+` 的性能可媲美甚至超越需要专用 P/D GPU 池的 **PD-disaggregation** 方案，但无需额外的硬件投入。
- **自动化与通用性**：整个框架是自适应的，避免了繁琐的手动调参，适用于不同的硬件和工作负载组合。

---

## 2. 核心实验方法和设置

### 数据集与工作负载
实验采用了多种合成和真实世界的工作负载来验证方法的有效性：
- **合成工作负载**：通过控制输入/输出长度比例，构造了三种典型场景：
  - Decode-heavy (128 input / 1024 output)
  - Balanced (512 / 512)
  - Prefill-heavy (1024 / 128)
- **真实世界工作负载**：
  - **ShareGPT**: 对话数据集。
  - **LongBench**: 长上下文理解基准。
  - **NuminaMath**: 思维链推理任务。
  - **WildChat**: 多轮对话日志。

### 实验设置
- **硬件平台**：在四种不同 memory bandwidth 的 GPU 上进行评估，以覆盖广泛的硬件环境：
  - **高带宽**：`H200` (4.8 TB/s), `B300` (8.0 TB/s)
  - **带宽受限**：`RTX PRO 6000` (1.792 TB/s), `L40S` (0.864 TB/s)
- **模型**：主要使用 `Qwen3-8B` 和 `Qwen3-30B-A3B` (MoE)，辅以 `Gemma-3-1B-IT` 进行补充实验。
- **评估指标**：
  - **吞吐量 (Throughput)**：以 `tokens/s` 和 `requests/s (RPS)` 衡量。
  - **延迟 (Latency)**：首令牌延迟 **TTFT (Time to First Token)**，每令牌延迟 **TPOT (Time Per Output Token)**，以及初始延迟 **ITL (Initial Token Latency)**。
  - **Goodput**：在满足 SLO（如 TTFT < 10s, TPOT < 100ms）前提下的有效吞吐量。

### 基线方法对比
- **v0**：vLLM v0 的独占批处理调度器，等价于 `EB(k=1)`。
- **v1**：vLLM v1 的混合批处理调度器，作为当前主流的 **MB** 基线。
- **EB(k\*)**：本文提出的自适应独占批处理调度器。
- **EB+**：本文提出的混合调度器。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **在带宽受限 GPU (RTX PRO 6000) 上**：
    - `EB(k*)` 在所有工作负载上均优于 `v1` (MB)。
    - 对于 `Qwen3-8B`，平均 RPS 提升 **+7.9%**；对于 `Qwen3-30B-A3B`，平均提升 **+1.4%**。
    - 在 `L40S` 上，`EB(k*)` 的 RPS 达到 **14.70**，比 `v1` 的 **10.36** 高出 **41.9%**。

2.  **在高带宽 GPU (H200) 上**：
    - `v1` (MB) 的优势更为明显，因为充足的带宽缓解了 prefill-decode 干扰。
    - `EB(k*)` 的性能与 `v1` 接近，但在某些配置下略有落后。

3.  **混合调度器 EB+ 的表现**：
    - 在非平稳流量下，`EB+` 始终能获得最佳或次佳的吞吐量。
    - 与 `v1` 相比，`EB+` 在 `RTX PRO 6000` 上的吞吐量最高提升了 **36.4%**。
    - 在中等负载下，`EB+` 能将 `TPOT` 降低至 `v1` 的 **1.8倍** 以下。

4.  **消融实验结果**：
    - **相位切换阈值验证**：实验证明，本文推导的渐近最优阈值 `θ*` 在实际大批次场景下性能接近最优，验证了理论的正确性。
    - **IFR 修正有效性**：对于具有 Increasing Failure Rate (IFR) 特性的现实工作负载，应用 IFR 修正后的阈值 `Δθ` 能带来正向增益，证明了其必要性。
    - **参数敏感性**：`EB(k*)` 调度器对参数估计误差不敏感，即使估计有偏差，也能保持较高的性能，表现出良好的鲁棒性。

---

## 4. 关键结论和发现

### 主要发现
1.  **MB 并非普适最优**：**Mixed Batching (MB)** 的优势并非绝对，其性能与 **GPU memory bandwidth**、**模型规模** 和 **工作负载组成** 密切相关。
2.  **带宽是关键因素**：在带宽受限的硬件上，**Exclusive Batching (EB)** 通过避免 prefill-decode 内存带宽竞争，可以显著提升吞吐量。
3.  **存在明确的性能交叉点**：当 decode token 占总 batch 的比例超过某个阈值时，MB 的边际成本会超过纯 decode 成本。这个阈值在 `RTX PRO 6000` 上仅为 **20%**，而在 `H200` 上则高达 **80%**。
4.  **自适应是王道**：静态地选择 `EB` 或 `MB` 都不是最优解。**EB+** 通过在线动态决策，能够在不同场景下自动选择最优策略，实现了全局效率最大化。

### 方法的局限性
- **理论假设**：部分理论推导（如流体近似 fluid approximation）基于大批次和饱和状态的假设，可能在小批次或低负载场景下精度下降。
- **实现复杂性**：虽然 `EB+` 自动化程度高，但其内部逻辑（如在线参数估计、阈值计算）比简单的 `MB` 更为复杂。
- **对极端工作负载的泛化**：虽然在多种工作负载上表现良好，但对于一些极其特殊或未见过的请求模式，其自适应能力仍需进一步验证。

### 未来工作方向
- **更精细的 SLO 感知调度**：将 SLO（如严格的 TTFT 或 TPOT 要求）直接融入调度决策过程，实现更智能的权衡。
- **与并行技术结合**：探索将该调度框架与 **sequence parallelism** 或 **pipeline parallelism** 等分布式训练/推理技术相结合，以应对更大规模的模型。
- **扩展到其他架构**：验证该方法在非 Transformer 架构或其他类型的生成式模型上的适用性。

</details>

---

### 10. [EVA-Net: Subject-Independent EEG Motor Decoding with Video-Derived Motor Priors](https://arxiv.org/abs/2606.01884)

**Authors**: Ziyuan Li, Yueyu Sun, Yimeng Zhang  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.01884v1  

#### Abstract
Practical non-invasive Brain-Computer Interface (BCI) systems require EEG decoders with strong cross-subject generalization and minimal calibration. However, inter-subject variability and signal non-stationarity often entangle motor semantics with subject-specific noise, limiting subject-independent...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《EVA-Net: Subject-Independent EEG Motor Decoding with Video-Derived Motor Priors》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前非侵入式 **Brain-Computer Interface (BCI)** 系统在实际应用中面临两大挑战：
- **Inter-subject variability**（被试间差异）：不同个体的 EEG 信号存在显著差异。
- **Signal non-stationarity**（信号非平稳性）：同一被试在不同时段的脑电活动也会变化。

这些因素导致大多数深度学习模型在训练集上表现良好，但在未见过的新被试（unseen subjects）上泛化能力差，严重依赖大量校准数据，限制了 **calibration-light BCI** 的部署。

此外，已有研究尝试引入 **text** 作为语义锚点（如 EEG-CLIP），但文本描述是静态、离散且缺乏对运动过程动态细节的刻画，难以有效引导 EEG 表征学习。

---

### 🚀 提出的新方法与创新思路
本文提出 **EVA-Net**，一种两阶段的 **subject-independent EEG motor decoding** 框架，其核心创新在于：

#### （1）使用 **action video** 作为动态语义先验（motor priors）
- 利用动作视频提供丰富的时空动态信息（spatiotemporal dynamics），更贴近真实的 motor execution 过程。
- 在训练阶段将 EEG 信号与对应的动作视频进行跨模态对齐，使网络学习到更具泛化性的运动语义表示。

#### （2）两阶段训练范式设计
- **Stage 1: Cross-Modal Alignment**
  - 使用双编码器结构（EEG Conformer + VideoMAE）提取模态特征。
  - 通过 **cross-modal contrastive learning** 和 **supervised contrastive loss** 将 EEG 与视频嵌入映射至共享语义空间，减少被试特异性噪声。
  
- **Stage 2: Prior-Guided Classification**
  - 引入 **video-derived prototypes**（基于视频聚类得到的类别原型）构建分类头。
  - 结合 **knowledge distillation** 将视频中的高级语义知识迁移到纯 EEG 分类器中。
  - 最终模型仅需 EEG 输入即可推理，**无实时多模态开销**，适用于实际应用场景。

#### （3）保留 EEG-only inference
- 视频仅用于离线训练，不参与在线推断，兼顾高性能与实用性。

---

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | EVA-Net |
|------|--------|---------|
| 语义监督方式 | 手工标注文本（静态、稀疏） | 动作视频（连续、细粒度动态） |
| 泛化能力 | 易过拟合于被试特定模式 | 学习更通用的运动语义 |
| 推理效率 | 单模态高效 | 同样为单模态推理（仅 EEG） |
| 跨被试性能 | 通常下降明显 | 显著提升 LOSO 表现 |

> ✅ **核心优势**：**以视频为“教师”训练 EEG “学生”，实现强 cross-subject generalization，同时保持轻量级部署。**

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 描述 |
|-------|------|
| **EEGMMI** | 64通道、160Hz采样率，来自109名被试，执行左右手/双手/双脚运动任务，共14轮实验。 |
| **BCIC-IV-2a** | 基于提示的 motor imagery 数据集，9名被试，四类任务（左手、右手、脚、舌头），每会话288次试验。 |

> ⚠️ 为建立跨模态对应关系，作者额外构建了一个辅助视频数据集：
> - 包含5个动作类别：左/右/双手、脚、舌。
> - 由10人录制，在5种不同场景下拍摄，增强视觉多样性。
> - 视频切分为 4秒片段，每类1000个样本。

---

### 🧪 实验设置与评估指标

#### 评估协议（Evaluation Protocols）
| 协议 | 描述 |
|------|------|
| **Cross-Session Subject-Dependent** | Session 1 训练，Session 2 测试，5折交叉验证用于早停。 |
| **Subject-Independent (LOSO)** | Leave-One-Subject-Out：每次留一个被试测试，其余训练。最能反映跨被试泛化能力。 |
| **Pooled-Subject K-Fold CV** | 所有被试数据混合后进行 K 折交叉验证，缓解单被试样本不足问题。 |

#### 主要评估指标
- **Accuracy**（准确率）
- **Macro-F1**（宏平均 F1 分数）
- **Kappa Score**（Kappa 系数，衡量一致性）

---

### 🆚 基线方法对比
比较了五种主流深度学习 EEG 解码模型：
1. **EEGNet**：轻量级 CNN 架构
2. **ShallowConvNet**：浅层卷积网络
3. **EEGCCT**：紧凑型 Convolutional Transformer
4. **EEG Conformer**：CNN-Transformer 混合架构
5. **MSVTNet**：多尺度 Vision Transformer 用于 MI-EEG

> 所有基线均在同一实验条件下复现。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Tables 1 & 2）

#### 在 **EEGMMI** 上的表现（LOSO 设置）
| 方法 | Accuracy | Macro-F1 | Kappa |
|------|----------|----------|--------|
| EEG Conformer (best baseline) | 62.56% | 61.40% | 0.50 |
| **EVA-Net (Ours)** | **72.30%** (+8.66%) | **70.80%** (+8.13%) | **0.70** (+0.13) |

> ✅ **LOSO 准确率提升达 8.66%**，表明在跨被试场景下具有显著优势。

#### 在 **BCIC-IV-2a** 上的表现（subject-independent）
| 方法 | Accuracy |
|------|----------|
| EEGCCT (best baseline) | 67.90% |
| **EVA-Net (Ours)** | **71.25%** (+3.35%) |

> ✅ 在独立被试设置下仍取得最高精度，且在 subject-dependent 场景中也具备竞争力（75.80%），说明并非牺牲个体性能换取泛化。

#### Pooled K-Fold 结果（EEGMMI）
| 方法 | Accuracy |
|------|----------|
| EEG Conformer | 72.69% |
| **EVA-Net** | **76.10%** |

> ✅ 即便在数据充足的情况下，EVA-Net 依然领先，进一步验证其表征学习的有效性。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）模态对比：Video vs. Text
- 替换视频为文本（Transformer 编码的文本描述）重新训练。
- 使用 **cross-subject FID heatmap** 分析原型分布一致性。
- **结果**：文本原型在不同被试间的方差更大 → 更难解耦 subject-specific 噪声。
- **结论**：**video 是比 text 更稳定、有效的语义锚点**。

#### （2）组件消融分析（见 Fig. 4）
移除以下任一组件均导致性能下降：
| 组件 | 影响 |
|------|------|
| **w/o Proto-Head** | 性能下降最严重 → prototype 分类头至关重要 |
| **w/o Cproto**（原型正则） | 下降明显 → 有助于维持共享空间对齐 |
| **w/o LKD**（知识蒸馏） | 下降可观 → 跨模态知识迁移有效 |
| **w/o class-balanced sampling** | 性能波动增大 → 对抗类别不平衡重要 |

> ✅ 正相关性发现：**跨模态 cosine similarity 越高，分类准确率越高**，证明对齐质量直接影响最终性能。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Action videos 可作为更优的语义先验**，相比文本能更好地捕捉 motor process 的动态特性，显著提升 EEG 表征的泛化能力。
2. **EVA-Net 在 subject-independent 设置下表现卓越**，尤其在 LOSO 协议中大幅提升准确率（+8.66% @ EEGMMI），验证其强大的跨被试迁移能力。
3. **两阶段框架成功实现了“视频引导、EEG 推理”**：既利用了多模态信息，又避免了运行时复杂度增加。
4. **消融实验证明各模块协同作用**，特别是 prototype-based classification 与 knowledge distillation 对性能提升至关重要。

---

### ⚠️ 局限性
1. **依赖高质量同步视频数据**：需要人工采集并标注动作视频，成本较高。
2. **目前仅支持类别级对齐**（category-level pairing），尚未实现逐帧时间对齐（temporal alignment）。
3. **视频多样性有限**：尽管尝试覆盖多种场景，但仍可能引入新的偏差。
4. **未在更多任务类型上验证**（如情绪识别、语言理解等），适用范围有待扩展。

---

### 🔮 未来工作方向
1. **探索自动生成视频或使用大规模预训练视频模型**（如 InternVideo）替代人工采集数据。
2. **引入时间对齐机制**（如 DTW 或 attention alignment）实现更精细的 EEG-Video 动态匹配。
3. **拓展至其他 BCI 范式**：如 SSVEP、ERP、speech imagery 等。
4. **结合生理建模**：融合生物力学模型进一步约束运动先验。
5. **端到端可微分训练策略**：打破两阶段分离训练的限制，进一步优化整体性能。

---

## ✅ 总结一句话
> **EVA-Net 首次系统性地将 action video 作为动态语义先验引入 EEG 解码，通过跨模态对齐与知识迁移，显著提升了 subject-independent BCI 的泛化能力，为低校准、高鲁棒的脑机接口提供了新范式。**

</details>

---

### 11. [Resonant Context Anchoring: Decoupling Attention Routing and Signal Gain at Inference Time](https://arxiv.org/abs/2606.01923)

**Authors**: Mingkuan Zhao, Yide Gao, Wentao Hu, Suquan Chen, Tianchen Huang, Zhenhua An, Zetao Chang, Xiayu Sun, Yuheng Min  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.01923v1  

#### Abstract
Large Language Models (LLMs) frequently exhibit "contextual disregard" when faced with input evidence that conflicts with their internal parametric memory, leading to persistent factual hallucinations. Existing mitigation strategies primarily rely on suppressing specific neuron activations or employ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Resonant Context Anchoring: Decoupling Attention Routing and Signal Gain at Inference Time

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在面对与内部参数化记忆（parametric memory）相冲突的输入证据时，常表现出“**contextual disregard**”（上下文忽视），即优先依赖其训练中固化的事先知识而非当前提供的事实，导致**factual hallucinations**（事实性幻觉）。这种现象严重削弱了模型在高可靠性、知识密集型任务中的可信度。

### 🚀 提出的新方法：Resonant Context Anchoring (RCA)
作者提出了一种轻量级、推理时干预的方法——**Resonant Context Anchoring (RCA)**，其核心思想是：
- 将 **self-attention 机制中的路由决策（routing logic）与信号增益（signal gain）进行正交解耦**。
- 利用原始的 pre-softmax attention scores 作为语义对齐的瞬时度量，构建一个非线性整流（non-linear rectification）机制，动态放大与查询高度相关的 context token 的 Value 向量范数（norm），从而增强上下文信号的能量。

该方法不改变 attention probability 分布，仅调节信息强度，在保留语法正确性的同时提升事实一致性。

### 🔍 相比现有方法的优势
| 方法类别 | 代表技术 | 缺陷 | RCA 的优势 |
|--------|--------|------|-----------|
| **Supervised Fine-Tuning (SFT)** | 微调对齐数据 | 高成本、过拟合风险、泛化差 | ✅ 无需训练（training-free）、零额外训练开销 |
| **Contrastive Decoding (CD)** | CAD, DoLa 等 | 多次前向传播 → 推理延迟翻倍 | ✅ 单次前向、计算开销可忽略（element-wise ops） |
| **Activation Engineering** | 抑制特定 attention head 或 neuron | 破坏语法结构、增加 perplexity | ✅ 不抑制任何通路，仅增强信号，保持 fluency |

> ✅ **核心优势总结**：  
> RCA 是一种 **plug-and-play、zero-cost、high-efficiency** 的推理策略，在不牺牲生成流畅性的前提下显著提升事实准确性。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验覆盖三类典型场景：

| 类别 | 数据集 | 描述 |
|------|-------|------|
| **上下文忠实性** | XSum | 单文档摘要任务，评估生成内容与源文本的事实一致性 |
| **强知识冲突** | NQ-Swap | 替换维基百科中的实体（如国家/首都），迫使模型选择外部错误上下文 vs 内部正确知识 |
| | MemoTrap | 测试模型是否能抵抗刻板记忆完成反直觉谚语 |
| **通用能力保留** | TruthfulQA, TriviaQA, PopQA | 闭卷问答，验证无相关上下文时模型的世界知识不受影响 |

### 📊 评估指标
| 指标 | 用途 |
|------|------|
| **FactKB**, **AlignScore** | 衡量生成文本与原文的事实一致性（越高越好） |
| **ROUGE-L** | 衡量内容覆盖率，检验信息丰富度 |
| **Exact Match (EM)** | 在 NQ-Swap 中衡量是否遵循修改后的上下文 |
| **Micro/Macro Accuracy** | 在 MemoTrap 中评估准确率 |
| **MC1 / MC2** | TruthfulQA 中的选择题得分，检验真实性 |

### ⚙️ 实验设置
- **模型**：Llama-3-8B-Instruct 和 Llama-3-70B-Instruct
- **基线方法**：Standard greedy decoding（标准贪婪解码）
- **实现方式**：作为参数自由模块集成到 Hugging Face Transformers 库中
- **硬件**：NVIDIA A100 GPU 集群
- **超参数**：共振敏感系数 `γ` 进行网格搜索（0.02 ~ 0.12）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

#### ✅ 上下文忠实性（XSum）
| 模型 | FactKB ↑ | AlignScore ↑ | ROUGE-L ↑ |
|------|---------|-------------|----------|
| Llama-3-8B (Baseline) | 47.61 | 58.20 | 19.90 |
| Llama-3-8B (RCA)     | **50.14** (+2.53) | **59.45** (+1.25) | **20.03** (+0.13) |
| Llama-3-70B (Baseline) | 61.32 | 65.10 | 22.41 |
| Llama-3-70B (RCA)     | **61.72** (+0.40) | **65.88** (+0.78) | 22.21 (-0.20) |

> ✔️ 所有指标均提升，且 **ROUGE-L 未下降**，说明信息完整性得以保持。

#### ✅ 知识冲突解决（NQ-Swap & MemoTrap）
| 模型 | NQ-Swap EM ↑ | MemoTrap Micro Acc ↑ | Macro Acc ↑ |
|------|--------------|------------------------|-------------|
| Llama-3-8B | 60.62 → **64.54** (+3.92) | — | — |
| Llama-3-70B | 76.11 → **77.46** (+1.35) | 66.52 → **77.35** (+10.83) | 68.47 → **75.58** (+7.11) |

> ✔️ 在大模型上效果更显著，尤其在 MemoTrap 上实现巨大跃升，表明 RCA 能有效打破 parametric attractor。

### 🔬 消融实验与参数敏感性分析（Table 2）

| γ 设置 | Llama-3-8B (NQ-Swap EM) | 观察 |
|-------|--------------------------|------|
| γ = 0 (Baseline) | 60.62 | — |
| γ = 0.04 | **64.54** | 最优值 |
| γ = 0.05 | 64.10 | 接近最优 |
| γ = 0.08 | 62.80 | 开始下降 |
| γ = 0.10 | 58.40 | 劣于基线 |
| γ = 0.12 | 52.10 | 显著退化 |

> 🔍 发现存在明显的“sweet spot”区间（0.04–0.05），超出后因过度放大噪声导致输出质量下降。

### 🛡️ 安全性评估（Table 3）—— 验证“无害性”

| 任务 | 指标 | Baseline | RCA | 差异 |
|------|------|---------|-----|------|
| TruthfulQA | MC1 | 38.92 | 38.92 | 0.00 |
| | MC2 | 55.64 | 55.61 | -0.03 |
| TriviaQA | EM | 56.58 | 56.52 | -0.06 |
| PopQA | Acc | 26.64 | 26.59 | -0.05 |

> ✅ 几乎无差异，证明 RCA 在无关或空上下文场景下自动“休眠”，不影响通用能力。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **幻觉主因是信号衰减而非路由失败**：  
   LLM 忽视上下文的根本原因不是无法识别相关信息（routing error），而是 context signal 在深层网络传播过程中能量不足（gain deficit），被 parametric noise 淹没 → **SNR 过低**。

2. **RCA 成功提升了残差流中的 SNR**：  
   通过动态放大高对齐度 Value 向量的范数，RCA 显著增强了上下文信号的相对权重，使生成轨迹转向真实证据流形。

3. **实现了 Pareto 改进**：  
   在多个任务上同时提升 **faithfulness** 与 **fluency**，且不损害通用语言理解能力，是一种理想的“always-on”解码策略。

4. **轻量高效，部署友好**：  
   仅引入 element-wise 运算，无额外前向传递，推理延迟几乎不变，适合实时应用。

### ⚠️ 方法的局限性
1. **需调参 `γ`**：虽然跨模型稳定（8B 用 0.04，70B 用 0.05），但仍需小规模开发集校准。
2. **理论基于线性子空间假设**：实际 Transformer 残差流高度非线性，当前建模为简化近似。
3. **尚未在 MoE 或其他架构广泛验证**：尽管理论上 architecture-agnostic，目前实验集中于 Llama-3 系列。

### 🔮 未来工作方向
- 自动化 `γ` 的动态调整机制（例如基于 query-context 相关性自适应缩放）
- 扩展至多模态 context anchoring（如图像、表格）
- 结合 retrieval-augmented generation (RAG) 构建统一可信生成框架
- 探索 RCA 在思维链（CoT）推理中对中间步骤事实性的锚定作用

---

> 💡 **一句话总结**：  
> RCA 提供了一个全新视角——**从信号动力学角度干预 LLM 推理过程**，通过解耦 attention 的“路由”与“增益”，以极低成本实现对事实性的精准控制，为构建高可靠生成系统提供了实用而优雅的解决方案。

</details>

---

### 12. [Dialectics of Alignment: Harnessing Unsafe Knowledge for Dynamic Safety Routing](https://arxiv.org/abs/2606.00686)

**Authors**: Maryam Hashemzadeh, Jerry Huang, Minseon Kim, Marc-Alexandre C\^ot\'e, Sarath Chandar  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.00686v1  

#### Abstract
The prevailing paradigm in large language model (LLM) alignment operates via erasure, filtering unsafe data or training models to strictly refuse harmful prompts. While effective at reducing immediate toxicity, this approach fundamentally constricts the model's epistemological scope, resulting in ov...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Dialectics of Alignment: Harnessing Unsafe Knowledge for Dynamic Safety Routing

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **LLM alignment**（对齐）范式依赖“**安全即擦除**”（safety-by-erasure），即通过过滤不安全数据或训练模型严格拒绝有害提示来提升安全性。这种方法虽然能减少直接毒性，但也导致模型过度保守，对许多敏感但无害的查询（如科学研究、心理求助等）也进行**泛化拒绝**（blanket refusal），输出如“I can't help with that”的无信息响应，严重损害了模型的**有用性**（informativeness）和用户体验。

### 提出的新方法与新思路
本文提出了一种**辩证式对齐**（dialectical approach to alignment）的新范式，其核心洞见是：  
> **真正的安全性并非来自对不安全知识的抹除，而是对其的受控整合**（controlled integration）。

为此，作者提出了 **SafeMoE** 框架，一个基于 **Mixture-of-Experts (MoE)** 的轻量级架构，具体创新如下：
- **隔离不安全知识**：将从大量不安全语料中学习到的领域特定知识，封装进独立的 **Low-Rank Adapter (LoRA) experts** 中。这些专家被称为“不安全专家”（unsafe experts），它们专门掌握高风险领域的深度知识。
- **动态路由机制**：引入一个轻量级的 **gating network**（路由器），该网络仅在少量精心策划的**安全且有信息量的响应**（safe-informative responses）上进行训练。在推理时，路由器动态地选择并组合多个不安全专家，引导生成过程，从而在利用其专业知识的同时，确保最终输出符合安全约束。
- **范式转变**：实现了从“**避免生成有害内容**”到“**利用有害知识生成安全且有信息量的响应**”的范式转变。

### 相比现有方法的优势
- **更高的安全性与信息量**：相比传统方法，SafeMoE 在显著提高安全性的同时，大幅提升了响应的信息量，用建设性的对话替代了无意义的拒绝。
- **极高的数据效率**：路由器的训练仅需**极少的安全样本**（文中仅用不到800条，甚至可降至100条），即可实现卓越性能，解决了高质量安全数据标注成本高昂的问题。
- **强大的零样本泛化能力**：路由机制展现出对未见过的领域和更广泛安全任务的强大零样本（zero-shot）泛化能力，无需针对每个新领域进行监督训练。
- **知识的有效利用**：将通常被视为“污染源”的不安全数据，转变为宝贵的领域知识来源，实现了资源的再利用。

## 2. 核心实验方法和设置

### 数据集
- **不安全数据**（`D_us`）：使用 **PKU-SafeRLHF** 数据集，包含18个不同的有害类别（如药物武器、经济犯罪、网络犯罪、心理伤害等）。这些数据用于训练各个 **LoRA experts**。
- **安全且有信息量的数据**（`D_s`）：为了训练路由器，作者使用 **GPT-4** 对上述有害提示生成了安全且有信息量的响应。对于 `SafeMoE-4` 等变体，仅在4个类别上各收集200条，总计 **800条** 安全样本。这些数据质量高，旨在提供替代方案而非简单拒绝。

### 实验设置和评估指标
- **基础模型**（Base Model）：主要使用 **Mistral-7B**，并验证了在 **Qwen-3B** 上的有效性。
- **模型变体**：构建了不同规模的 SafeMoE 模型，如 `SafeMoE-4`（4个专家）、`SafeMoE-L`（10个专家）、`SafeMoE-XL`（18个专家）。
- **评估框架**：采用 **LLM-as-a-Judge** 范式，使用 **GPT-4o** 作为裁判模型进行自动化评估。
- **评估指标**：
  1. **Safety Percentage (安全率)**：响应被判定为安全的比例（0-100%）。
  2. **Informativeness Score (信息量得分)**：对安全的响应，由裁判模型打分（1-10分），综合考量帮助性、相关性、准确性、深度等因素。

### 基线方法对比
- **基础模型**：`Mistral-7B`, `Qwen-3B`。
- **标准对齐模型**：`Mistral-SFT`（全量微调）、`Zephyr`、`RealSafe-R1` 等。
- **其他安全方法**：`SafeLoRA`, `SN-Tune`, `Oyster-I 14B`。
- **自建对照组**：
  - **Refusal-Only**：用固定拒绝消息训练的专家，测试拒绝行为的影响。
  - **SFT-800**：仅用800条安全数据对基础模型进行全量微调（SFT）。
  - **Knowledge-Only**：仅用医学、金融等领域的知识专家，不含任何不安全数据。

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **Table 1** 和 **Table 3** 的综合评估中，`SafeMoE-XL` 取得了压倒性优势：
- **平均安全率**：达到 **90.8%**，相比基础模型 `Mistral-7B` 的 **17.4%** 有巨大飞跃。
- **平均信息量得分**：达到 **8.1**，而基础模型仅为 **5.6**。
- **相对提升**：相比基线，安全率有超过 **20%的相对提升**（绝对增益超15%），同时信息量也显著更高。

### 与基线方法的对比结果
- **全面超越**：在所有基准测试（AdvBench, HarmBench, BeaverTails, HarmfulQA, PKU-Safe）上，`SafeMoE-XL` 均显著优于所有基线模型。
  - 例如，在 **AdvBench** 上，`SafeMoE-XL` 达到 **97.2%** 的安全率，远超第二名 `Oyster-I 14B` 的80.5%。
- **小模型上的成功**：即使在较小的 `Qwen-3B` 模型上，`SafeMoE-Qwen` 的安全率也达到了 **62.4%**，信息量得分为 **7.6**，分别比其基线高出近 **5倍** 和 **1分**，证明了方法的普适性。

### 消融实验结果
- **不安全专家至关重要**：`Knowledge-Only` 模型表现极差（信息量得分约1.8），证明了**不安全专家所蕴含的领域知识是生成有信息量响应的关键**。
- **安全数据效率极高**：如 **Figure 3** 所示，当每类安全样本数增加到 **100条** 时，`SafeMoE` 的信息量得分已能匹配需要数千条数据训练的 `RealSafe-R1` 基线，证明了其惊人的数据效率。
- **动态路由是核心**：消融实验证明，静态的路由方式（如第一层路由）效果远不如动态的逐token路由，后者能有效引导生成轨迹远离有害区域。
- **专家数量越多越好**：从 `SafeMoE-4` 到 `SafeMoE-XL`，随着专家数量的增加，模型的安全性和信息量均持续提升。

## 4. 关键结论和发现

### 主要发现
1. **不安全数据是宝藏而非垃圾**：富含领域知识的不安全数据不应被丢弃，而是可以被隔离和利用的宝贵资产。
2. **受控整合优于简单擦除**：通过 **SafeMoE** 这样的架构，可以实现对不安全知识的“受控整合”，从而在保证安全的前提下，最大化模型的有用性。
3. **动态路由是关键机制**：轻量级的路由器能够以极低的数据成本，学会如何“指挥”庞大的不安全专家库，合成出既安全又详尽的响应。
4. **范式转变**：本研究挑战了主流的“安全即拒绝”范式，提出了一种新的对齐哲学——**真正的安全是通过智慧地驾驭危险知识来实现的**。

### 方法的局限性
- **计算开销**：尽管使用了LoRA，但维护一个专家库和执行逐token路由仍会带来比单个稠密模型更高的内存和延迟开销。
- **负载均衡**：在大规模部署时，可能需要高级的负载均衡策略来防止某些专家过载或闲置。
- **潜在的路由瓶颈**：路由机制本身可能成为性能瓶颈，尤其是在处理复杂查询时。

### 未来工作方向
- **探索更强的领域分离**：研究如何在专家内部实现更强的领域隔离，以缓解对抗性编码带来的安全隐患。
- **集成到现有MoE模型**：探索如何将SafeMoE的思想融入现有的大型MoE LLM（如Mixtral）中，而无需额外添加LoRA专家。
- **优化复杂推理**：当前方法在MMLU和GSM8K等复杂推理任务上存在权衡，未来工作可探索如何在保持安全的同时，进一步优化这些能力。

</details>

---

### 13. [Local MixVR: Breaking the Communication-Sample Dependence in Distributed Learning](https://arxiv.org/abs/2606.01128)

**Authors**: Tehila Dahan, Bassel Hamoud, Roie Reshef, Martin Jaggi, Kfir Y. Levy  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.01128v1  

#### Abstract
Communication overhead is a crucial bottleneck in scalable distributed learning. While existing methods aim to efficiently utilize data points, such as Local SGD, Minibatch SGD, and their accelerated variants, they still exhibit communication-round complexity that scales with the total number of sam...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Local MixVR: Breaking the Communication-Sample Dependence in Distributed Learning*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在分布式学习中，**通信开销**是可扩展训练的关键瓶颈。尽管已有方法如 **Local SGD** 和 **Minibatch SGD** 通过本地更新或增大批量来减少通信轮数，但它们的**通信复杂度仍依赖于总样本数 $N$**。这意味着当数据量增大时，所需的最小通信轮数也随之增加，限制了大规模场景下的效率。

此外，现有最优方法（如 **Minibatch Accelerated SGD, ASGD**）虽然加速收敛，但在常见训练设置下仍未被超越，尤其是在 $M \leq O(N^{1/4})$ 的情形中。

### 提出的新方法与思路
本文提出 **Local MixVR** —— 一种全新的分布式优化框架，结合三种方差缩减技术以打破通信轮数对样本总量 $N$ 的依赖：

1. **Local Double-Momentum (u²-SGD)**  
   每个 worker 在本地使用双动量机制：
   - **Anytime Averaging**：平滑模型轨迹，防止快速漂移；
   - **STORM Estimator**：修正标准动量中的陈旧梯度偏差，实现更强的方差缩减。

2. **Budget Mixing（预算混合）**  
   将每轮 $K$ 个样本划分为两部分：
   - $(1-\alpha)K$ 用于本地优化（推进优化路径）；
   - $\alpha K$ 用于同步前的 **minibatch averaging**（降低噪声注入），从而平衡“前进”与“稳定”。

3. **Drift Correction Mechanism（漂移校正机制）**  
   在同步步骤中引入梯度差异项：
   $$
   \text{Correction} = \nabla f(x_t; B) - \nabla f(x_t^{(i)}; B)
   $$
   利用相同 minibatch 在全局参数 $x_t$ 和本地参数 $x_t^{(i)}$ 上的梯度差，纠正因本地更新积累的偏置。

### 相比现有方法的优势
- ✅ **首次实现通信轮数独立于总样本数 $N$**：仅依赖 worker 数量 $M$，即 $R_{\min} = \Omega(M)$。
- ✅ **在 $M \leq O(N^{1/4})$ 下优于 Minibatch ASGD**：打破了长期存在的性能上限。
- ✅ **理论保证更优的通信-计算权衡**：见 Table 1，Local MixVR 的收敛速率和最小通信轮数均为当前最优。

| Method | Rate [$\mathcal{O}(\cdot)$] | $R_{\min}$ [$\Omega(\cdot)$] |
|--------|-------------------------------|-------------------------------|
| Minibatch SGD | $1/\sqrt{MKR}$ | $N^{1/2}$ |
| Minibatch ASGD | $1/(KR)^{1/2}$ | $N^{1/4}$ |
| Local SGD | $1/K^{1/3}R^{2/3}$ | $M N^{1/2}$ |
| **Local MixVR (Ours)** | $1/(KR)^{1/2}$ | **$M$** |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MNIST**：手写数字分类任务，60K 训练样本，10K 测试样本。
- **CIFAR-10**：图像分类任务，50K 训练样本，10K 测试样本。

### 实验设置与评估指标
- **Workers 数量**：
  - MNIST：4 workers
  - CIFAR-10：8 workers
- **本地批量大小（local batch size）**：
  - MNIST：4
  - CIFAR-10：16
- **训练周期**：
  - MNIST：2 epochs
  - CIFAR-10：30 epochs
- **评估指标**：**测试准确率（Test Accuracy）** 随通信轮数 $R$ 的变化曲线。
- **调参范围**：
  - 学习率：{0.01, 0.05, 0.1}
  - 混合参数 $\alpha$：{0.05, 0.1, 0.25, 0.5, 0.75}
  - 动量系数：$\beta = 0.9$（对应文中 $\gamma = 0.95$ for u²-SGD）

### 基线方法对比
- **Local SGD**
- **Local Momentum**
- **Minibatch SGD**
- **Minibatch ASGD**

所有方法在相同的通信预算（即相同 $R$）下进行比较，确保公平性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Figure 1）
- 在 **MNIST（4 workers）** 和 **CIFAR-10（8 workers）** 上，**Local MixVR 显著优于所有基线方法**，尤其在低通信轮数（small $R$）区域表现突出。
- 即使在极少数通信轮（如 $R < 200$）下，Local MixVR 仍能保持高准确率，而其他方法严重下降。

### 与基线方法的对比结果
- **相比 Minibatch ASGD**：
  - 在相同 $R$ 下达到更高准确率；
  - 达到目标精度所需通信轮数更少（例如，在 CIFAR-10 上减少约 30–50%）。
- **相比 Local SGD / Local Momentum**：
  - 更强的抗漂移能力，允许更多本地步而不损失稳定性；
  - 收敛更快且最终性能更高。

### 消融实验（未明确展示，但可通过设计推断）
虽然论文未提供显式的消融图，但从方法设计可得出以下结论：
- 若移除 **drift correction**，同步时梯度偏置将累积，导致性能下降；
- 若不采用 **budget mixing**（即全为本地步），worker drift 加剧；
- 若仅用单动量而非 **double-momentum**，方差控制不足，影响收敛速度。

> 注：作者在附录中提供了完整的理论分析支持各组件必要性。

---

## 4. 关键结论和发现

### 主要发现
1. **通信轮数可以完全脱离样本总数 $N$ 的约束**：这是首个实现 $R_{\min} = \Omega(M)$ 的分布式算法。
2. **Local MixVR 在实际相关设置下（$M \leq O(N^{1/4})$）超越 Minibatch ASGD**：
   - 如 ImageNet-1K 中 $M=8$, $N \sim 1.28\times10^6$，有 $R_{\text{ASGD}} / R_{\text{MixVR}} \sim 4.2$；
   - FineWeb 数据集上可达 **30.7 倍通信节省潜力**。
3. **多机制协同作用至关重要**：local double-momentum + budget mixing + drift correction 构成一个鲁棒、高效的统一框架。

### 方法的局限性
- 当前分析限于 **凸函数设定**，非凸情况下的理论保证尚未建立。
- 超参数 $\alpha$ 需要调优，自动化选择策略有待研究。
- 实际带宽与延迟建模较弱，未考虑异构网络环境。

### 未来工作方向
1. 推导在 **期望平滑函数（expectation over smooth functions）假设下的下界**，验证 Local MixVR 是否已达最优。
2. 开发 **accelerated variants** of Local MixVR，进一步提升收敛速度。
3. 扩展至 **non-IID 和联邦学习场景**，增强对客户端异质性的适应性。
4. 探索 **自适应 $\alpha$ 控制机制**，动态调整本地优化与同步稳定之间的权衡。

--- 

> 🔚 **总结一句话**：  
> **Local MixVR 是首个打破通信轮数与样本总量依赖关系的分布式学习算法，在理论和实践中均实现了对 Minibatch ASGD 的超越，为海量数据下的高效训练提供了新范式。**

</details>

---

### 14. [A combination of noise and bilateral filters achieve supralinear and scalable adversarial robustness in CNNs](https://arxiv.org/abs/2606.02267)

**Authors**: Nicolas Stalder, Benjamin F. Grewe, Matteo Saponati, Pau Vilimelis Aceituno  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.02267v1  

#### Abstract
The vulnerability of deep neural networks to adversarial examples poses a significant challenge for real-world deployment. Existing techniques to enhance deep network robustness rely on adversarial training, an approach that is powerful but computationally intensive and typically tailored to specifi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A combination of noise and bilateral filters achieve supralinear and scalable adversarial robustness in CNNs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
深度神经网络（CNNs）在面对**Adversarial Attacks (AAs)** 时表现出严重的脆弱性，即微小且人类难以察觉的像素扰动即可导致模型误分类。虽然 **Adversarial Training (AT)** 是目前最有效的防御手段之一，但它存在以下显著缺陷：
- **计算成本极高**：需要约 9 倍于标准训练的 FLOPs；
- **泛化能力差**：其鲁棒性依赖于训练中见过的攻击类型，对未见攻击泛化性弱；
- **扩展性差**：随着模型规模增大，训练开销急剧上升。

此外，现有的简单防御方法（如添加噪声或图像滤波）虽计算开销低，但单独使用时提升有限。

### 提出的新方法或新思路
本文提出了一种**理论驱动的简单预处理器（preprocessor）**，将两种看似独立的方法——**Gaussian Noise 添加** 和 **Bilateral Filtering**——进行组合，并从理论上证明二者通过**互补机制**增强对抗鲁棒性：

- **Gaussian Noise** 有效抵御空间上**稀疏、细丝状（filamentary）**的攻击（因其随机性可“打乱”梯度方向）；
- **Bilateral Filter** 有效抑制靠近决策边界边缘的**大球形（spherical）**扰动（因其能将扰动推回图像流形）；
- 二者的结合能够覆盖更广泛的攻击形态，从而实现**超线性（supralinear）**的鲁棒性增益。

该预处理器在训练和推理阶段均应用，形式为：  
`Input → Add Gaussian Noise → Apply Multiple Bilateral Filters`

### 相比现有方法的优势
- ✅ **高效率**：引入的计算开销极小（negligible computational overhead），尤其相比 AT；
- ✅ **强鲁棒性**：在多种攻击下实现接近甚至超越 SOTA 的 robust accuracy；
- ✅ **可扩展性强**：在不同模型大小和训练预算下均保持高效，达到相同性能所需 FLOPs 显著更低；
- ✅ **易于集成**：作为独立预处理模块，可无缝嵌入现有训练流程；
- ✅ **理论支撑明确**：首次从几何角度分析噪声与滤波的互补机制，提供设计原则。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要数据集：**CIFAR-10**（用于主实验与消融研究）
- 验证泛化性：**Imagenet10**（ImageNet 的 10 类子集，验证跨数据集有效性）

### 实验设置和评估指标
#### 模型架构
- **Standard CNNs**：EfficientNet-B0（用于消融实验）
- **SOTA 对比模型**：Wide Residual Networks (**WRN**) 系列（如 WRN-28-4, WRN-28-10, WRN-82-12），使用 Swish 激活函数

#### 训练协议
- 使用 **TRADES** 框架进行 Adversarial Training；
- 结合 **Elucidating Diffusion Model (EDM)** 生成的合成数据增强训练；
- 预处理器在训练和推理时均启用；
- 噪声顺序默认为：**先加噪声，再滤波**（理论建议以提升 clean accuracy）

#### 评估指标
- **Clean Accuracy**：原始测试集准确率
- **Robust Accuracy**：在多种对抗攻击下的准确率，包括：
  - **FGSM**, **APGD-L∞**, **APGD-L2**
  - **EoTPGD**（Expectation over Transformation，针对随机化防御）
  - **C&W Attack**
  - **AutoAttack**（多攻击自动集成基准）
  - 自定义攻击 **TABPDA**（True Average BPDA，专门绕过预处理器梯度掩码）
- **FLOPs**：训练和推理阶段的浮点运算量，用于衡量效率

### 基线方法对比
- **Standard CNN**：无任何防御
- **+Bil. Filter**：仅双边滤波
- **+Gaussian Noise**：仅加噪
- **+JPEG Compression**：经典图像压缩防御
- **Fast Adversarial Training (Fast AT)**：轻量级 AT 方法
- **SOTA 模型**：来自 RobustBench 排行榜的先进模型（如 WRN-94-16）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（基于 WRN-82-12 模型）

| 方法 | Clean Acc (%) | AutoAttack Robust Acc (%) | 训练 FLOPs (相对比例) | 参数量 |
|------|---------------|----------------------------|------------------------|--------|
| SOTA (WRN-94-16) | 93.68 | 73.71 | 100% | 100% |
| WRN-82-12 (无预处理器) | 93.04 | 71.41 | ~60% | ~50% |
| **WRN-82-12 + Preprocessor** | **90.12** | **74.32** | **~35%** | **~50%** |

> ✅ **结果亮点**：
> - 在 **AutoAttack** 上排名第二，总体排名第三；
> - **鲁棒准确率提升 +0.6%** 超越此前 SOTA；
> - 仅使用 **~35% 的训练 FLOPs**、**~50% 的参数量**、**~33% 的训练轮数** 和 **~15% 的数据量**。

### 与基线方法的对比结果
- 在所有测试攻击中，**Noise + Bilateral Filter 组合始终优于任一单一方法**；
- 相比 **Fast AT**：
  - 在 C&W 攻击上表现更好（47.2% vs 20.9%）；
  - 尽管在某些攻击（如 FGSM）上略逊，但整体更均衡；
- 相比 **JPEG Compression**：
  - 在对抗 AAs 和 Gaussian noise 上均更优。

### 消融实验结果（CIFAR-10, EfficientNet-B0）

| 方法 | Clean Acc | FGSM | L∞ | EoT | L2 | C&W |
|------|-----------|------|-----|-----|-----|-----|
| Standard CNN | 74.5% | 3.5% | 0.2% | 0.2% | 1.3% | 0.6% |
| +Bil. | 69.0% | 10.0% | 1.0% | 1.2% | 11.9% | 0.5% |
| +Noise | 68.5% | 22.8% | 25.5% | 12.0% | 49.6% | 43.0% |
| **+Noise + Bil.** | **67.9%** | **33.9%** | **36.5%** | **18.9%** | **58.5%** | **47.2%** |
| Linear Gain (sum) | — | 25.8% | 26.1% | 12.8% | 58.9% | 42.3% |
| **Actual Gain** | — | **30.4%** | **36.2%** | **18.7%** | **57.2%** | **46.6%** |

> 🔍 **关键观察**：
> - 在 **4/6 种攻击** 下，组合增益 **超过线性叠加**，证实 **supralinear effect**；
> - 尽管 clean accuracy 下降，但下降幅度小于两者之和，说明存在协同效应；
> - 最佳配置：**多次迭代双边滤波（如 10–50 次）+ 方差为 0.03² 的 Gaussian Noise**

### 可扩展性分析（Scaling Efficiency）
- 图 2 显示，在 **相同训练 FLOPs** 下，本文方法显著优于 prior works；
- 达到相同 robust accuracy 所需训练 FLOPs **仅为竞品的 15%–50%**；
- 推理阶段也更高效：匹配性能模型需 **6–9× 更多 inference FLOPs**（图 7）；

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **噪声与滤波机制互补**：理论与实验证明，Gaussian Noise 与 Bilateral Filter 抵御不同类型的对抗攻击，联合使用可实现 **supralinear robustness gain**。
2. ✅ **预处理器高效且通用**：极低计算代价下大幅提升鲁棒性，适用于各种 CNN 架构与数据集。
3. ✅ **与 Adversarial Training 协同增效**：即使在 SOTA AT 框架下，加入该预处理器仍能进一步提升鲁棒性，同时大幅降低训练资源需求。
4. ✅ **具备真实防御价值**：在 AutoAttack 等综合基准上达到顶尖水平，且对自定义攻击 **TABPDA** 仍有增益，表明其提升非仅源于梯度掩码（gradient masking），而是真正增强了内在鲁棒性。

### 方法的局限性
- ❗ **diminishing returns**：随着模型规模和训练资源增加，相对增益逐渐减小（可能因模型已接近饱和）；
- ❗ **clean accuracy 微降**：预处理会轻微影响干净样本性能，尽管降幅可控；
- ❗ **顺序敏感性**：噪声与滤波的顺序会影响效果，文中推荐“先噪声后滤波”以平衡鲁棒性与 clean accuracy；
- ❗ **在 Imagenet10 上未见超线性增益**：可能因单个组件已接近性能上限，导致组合优势不明显（见 Table 3）。

### 未来工作方向
- 🔍 探索其他类型的 **noise**（如 Poisson、Uniform）、**filter**（如 Median、JPEG）及其组合；
- 🧠 引入 **生物启发机制**：借鉴人脑视觉系统中的噪声与边缘保持特性优化预处理器；
- ⚔️ 设计更强的 **定制攻击** 来进一步检验和改进预处理器；
- 🔁 将该思想推广至 **Vision Transformers (ViTs)** 等非 CNN 架构；
- 📈 研究 **scaling laws** 下噪声、滤波与数据集之间的交互规律；
- 🛠️ 将 **BPDA/EoT-style attacks** 整合进训练过程，提升训练效率与鲁棒性。

---

> 💡 **总结一句话**：  
> 本文提出一个**简单、高效、理论清晰**的预处理策略——**Gaussian Noise + Bilateral Filtering**，通过揭示二者在对抗鲁棒性上的**互补几何机制**，实现了**超线性且可扩展的防御增益**，在显著降低计算成本的同时达到甚至超越当前 SOTA 水平，为构建实用化鲁棒视觉系统提供了新范式。

</details>

---

### 15. [Brain-Atlas-Guided Generative Counterfactual Attention for Explainable Cognitive Decline Diagnosis Using Multimodal Connectomes](https://arxiv.org/abs/2606.01237)

**Authors**: Xiongri Shen, Jiaqi Wang, Zhenxi Song, Yi Zhong, Leilei Zhao, Xin He, Baiying Lei, Zhiguo Zhang  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.01237v1  

#### Abstract
Mild cognitive impairment (MCI) and subjective cognitive decline (SCD) are closely associated with the early Alzheimer's disease continuum, where accurate and explainable diagnosis is important for early risk assessment and intervention. Existing connectome-based deep learning models can improve cla...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Brain-Atlas-Guided Generative Counterfactual Attention for Explainable Cognitive Decline Diagnosis Using Multimodal Connectomes

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**认知衰退早期阶段**（如主观认知衰退 SCD 和轻度认知障碍 MCI）的诊断难题，旨在解决现有基于 **connectome** 的深度学习模型存在的“黑箱”问题。尽管这些模型在分类性能上有所提升，但缺乏对决策过程的可解释性，无法明确指出哪些功能连接（FC）或结构连接（SC）的变化驱动了诊断结果。

### 提出的新方法与创新思路
作者提出了一种名为 **Generative Counterfactual Attention-guided Network (GCAN)** 的新框架，其核心创新点如下：

- **生成式反事实注意力机制（Generative Counterfactual Attention）**  
  将诊断任务建模为一个**源标签到目标标签的反事实生成问题**。通过从源状态（如 HC）生成目标状态（如 MCI）的 connectome，并将两者差异作为 **counterfactual attention map**，从而识别出与疾病进展最相关的连接变化。

- **图谱感知双向 Transformer（Atlas-aware Bidirectional Transformer, AABT）**  
  设计了一种受脑图谱引导的编码-解码架构，在网络层面进行 token 化处理，保留 connectome 的拓扑结构，确保生成的 connectome 符合大脑功能网络组织规律。

- **多模态 FC-SC 联合反事实推理**  
  首次将反事实推理扩展至 **functional connectivity (FC)** 与 **structural connectivity (SC)** 的联合建模，分别生成 FC 和 SC 的反事实注意力图，揭示功能重组与结构退化的互补模式。

- **分离式预训练策略防止数据泄露**  
  使用独立的 FC 和 SC 预训练分类器提供先验信息用于反事实生成，但不参与最终诊断分类，避免测试集信息泄露。

### 相比现有方法的优势
| 维度 | GCAN 优势 |
|------|----------|
| **可解释性** | 提供**有方向性的解释**（增强/抑制连接），而非仅相关性；支持跨状态转换分析（如 HC→SCD→MCI） |
| **模型设计** | 显式建模状态间最小必要变化，更贴近临床认知衰退的渐进过程 |
| **多模态融合** | 分别提取 FC 和 SC 的反事实注意力，避免简单拼接导致的信息混淆 |
| **结构保持** | AABT 引入脑图谱约束，保证生成 connectome 的生物学合理性 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 类型 | 用途 | 样本量（HC/SCD/MCI） |
|--------|------|------|---------------------|
| **GUTCM** | 医院采集 | 单模态 FC 实验 | 77 / 75 / 99 |
| **ADNI** | 公开数据库 | 单模态 & 多模态实验 | 67 / 22 / 95 |
| **SLIM** | 公开数据集 | FC 预训练辅助 | 仅 HC（915） |
| **BJE** | 公开数据集 | SC 预训练辅助 | 仅 HC（180） |

> 注：多模态实验使用 ADNI 中具有完整 FC 和 SC 数据的子集（54 HC, 84 SCD, 60 MCI）。

### 实验设置与评估指标
- **任务类型**：三类二元分类任务  
  - HC vs. SCD  
  - HC vs. MCI  
  - SCD vs. MCI  

- **评估指标**：
  - 主要指标：Accuracy (ACC), Recall, Precision, F1-score
  - 可靠性分析：五折交叉验证 + 95% 置信区间
  - 合成质量评估：PSNR, SSIM, Correlation, MAE, MSE（归一化后）
  - 注意力重叠度：Top-k Jaccard Index（FC vs. SC）

- **基线方法对比**：
  - 多个近期发表的 SOTA 模型，包括：
    - Ramzan et al. (2020)
    - Wen et al. (2020)
    - Tang et al. (2024)
    - He et al. (2024)
    - Feng et al. (2025)
    - Huang et al. (2026)
  - 不同 backbone 对比：CNN, ResNet, Transformer, ViT, GCN 等

- **消融实验设计**：
  - 是否引入 counterfactual attention
  - 是否使用 AABT 结构
  - 是否采用多模态联合建模

---

## 3. 主要实验结果和性能指标

### 关键性能数据（单模态 FC）

| 任务 | 方法 | ACC (医院) | F1 (医院) | ACC (ADNI) | F1 (ADNI) |
|------|------|------------|-----------|------------|-----------|
| HC vs. SCD | **Proposed\*** | **0.933** | **0.929** | **0.728** | **0.704** |
| HC vs. MCI | **Proposed\*** | **0.747** | **0.821** | **0.697** | **0.756** |
| SCD vs. MCI | **Proposed\*** | **0.949** | **0.966** | **0.731** | **0.792** |

> \* 表示使用了 counterfactual attention。所有任务中，GCAN 均取得最优或接近最优表现。

#### 与基线方法对比
- 在 **HC vs. SCD** 上，GCAN 的 F1 提升显著（医院数据从 ~0.53 到 0.929），表明其对细微功能变化高度敏感。
- 在 **SCD vs. MCI** 上，GCAN 成功捕捉到从主观感受到客观损伤的过渡特征，ACC 达 0.949，远超多数基线。

### 多模态 FC-SC 实验结果
- **最佳模型**：`ResNet+Transformer*` 在多个任务中表现领先。
- **HC vs. SCD** 平均 ACC 达 **76.00±23.32%**，F1 为 **72.62±27.04%**
- 性能波动较大（标准差高），反映小样本下多模态建模的挑战。

### 消融实验结果
#### （1）反事实注意力有效性（Ablation on CA）
| 任务 | 模型 | CA | ACC (医院) | F1 (医院) |
|------|------|----|------------|-----------|
| HC vs. SCD | ResNet | × | 0.800 | 0.533 |
| HC vs. SCD | ResNet | ✓ | **0.933** | **0.929** |
| HC vs. MCI | ResNet | × | 0.632 | 0.722 |
| HC vs. MCI | ResNet | ✓ | **0.747** | **0.821** |

> 引入 counterfactual attention 后，几乎所有指标均有显著提升，证明其有效增强了判别能力。

#### （2）AABT 的作用
- 移除 AABT 后，虽然部分 SC 指标（如 SSIM）略有上升，但整体 FC 重建质量下降。
- **可视化显示**：无 AABT 时注意力图更平滑、缺乏网络边界感；加入 AABT 后，注意力集中在 DMN、FPN、CON 等已知认知相关网络内，更具神经生物学意义。

#### （3）合成质量评估
- **平均 PSNR**: 23.80 dB（FC）, 21.23 dB（SC）
- **平均 Correlation**: 0.6745（FC）, 0.6347（SC）
- 表明生成的 connectome 在全局结构和相对连接强度上保持合理保真度。

#### （4）FC 与 SC 注意力重叠分析
| 过渡方向 | Top-20% 正向重叠 | Top-20% 负向重叠 |
|----------|------------------|------------------|
| HC → MCI | 10.75% | 10.83% |
| SCD → MCI | 9.14% | 12.84% |
| SCD → HC | 11.41% | 11.97% |

> 表明 FC 与 SC 的反事实注意力存在**部分共享但非完全一致**的病变通路，支持功能与结构改变的协同但异步特性。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **反事实注意力能有效定位疾病相关连接变化**  
   GCAN 生成的注意力图集中于 **Default Mode Network (DMN)**、**Fronto-Parietal Network (FPN)**、**Cingulo-Opercular Network (CON)** 等与记忆、执行控制相关的高级认知网络，符合阿尔茨海默病的病理机制。

2. ✅ **多模态建模揭示功能与结构的互补动态**  
   - **FC** 注意力更广泛、动态性强，反映功能重组；
   - **SC** 注意力更稀疏、稳定，体现结构性退化；
   - 二者在关键网络中有适度重叠，提示 **structure-function decoupling** 是认知衰退的重要标志。

3. ✅ **AABT 提升了解释的结构合理性**  
   图谱引导的 token 编码使注意力分布更符合大脑网络分区，提升了结果的生物学可信度。

4. ✅ **优于 CAM 类方法的解释能力**  
   与 Grad-CAM 和 Score-CAM 相比，GCAN 提供了**带符号的方向性解释**（连接增强 vs 抑制），且模式更集中、更具过渡特异性。

### 局限性
1. 🛑 **依赖群体平均先验**：反事实生成使用类别均值 connectome，未能建模个体化疾病轨迹。
2. 🛑 **多模态样本量有限**：ADNI 子集较小，影响统计稳健性和泛化能力。
3. 🛑 **解释≠因果**：反事实注意力是模型推断结果，仍需纵向数据验证其是否真实对应疾病进展。
4. 🛑 **计算复杂度较高**：双模块生成器-判别器结构训练成本大。

### 未来工作方向
- 构建**纵向多中心队列**，验证反事实路径是否匹配真实认知衰退轨迹。
- 引入**个体化反事实先验**（如基于年龄、基因等）。
- 探索**端到端的反事实连续空间建模**，以刻画 AD continuum 的渐变过程。
- 加强**忠实性评估**（faithfulness evaluation）：如 deletion/insertion test、randomization check、临床变量相关性分析。
- 扩展至其他神经退行性疾病（如帕金森病、额颞叶痴呆）的应用。

---

> **总结**：本文提出的 GCAN 框架不仅在诊断性能上达到先进水平，更重要的是提供了一种**可解释、可追溯、具方向性**的认知衰退 connectome 分析范式，推动了 XAI 在临床神经影像中的透明化应用。

</details>

---

### 16. [ExpWeaver: LLM Agents Learn from Experience via Latent RAG](https://arxiv.org/abs/2606.01041)

**Authors**: Tao Feng, Tianyang Luo, Jingjun Xu, Zhigang Hua, Yan Xie, Shuang Yang, Ge Liu, Jiaxuan You  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.01041v1  

#### Abstract
Experience learning has achieved promising results in enhancing LLM agent planning and reasoning by integrating past interactions as reusable knowledge. However, existing methods remain confined to explicit text space, retrieving experiences via semantic similarity and concatenating them into the co...

---

### 17. [Thinking Economically: A Hierarchical Framework for Adaptive-Complexity Reasoning in LLMs](https://arxiv.org/abs/2606.01168)

**Authors**: Yubo Gao, Haotian Wu, Hong Chen, Junquan Huang, Yibo Yan, Jungang Li, Zihao Dongfang, Sicheng Tao, Puay Siew Tan, Jie Zhang, Xuming Hu  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.01168v1  

#### Abstract
Chain-of-Thought (CoT) has significantly enhanced LLM reasoning, yet often incurs substantial computational overhead due to "overthinking": generating excessively long rationales without commensurate accuracy gains. Existing efficiency methods typically apply uniform compression, which overlooks a c...

---

### 18. [LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](https://arxiv.org/abs/2606.00228)

**Authors**: Yao Lai, Xuyuan Xiong, Zeyue Xue, Guojin Chen, Jing Wang, Xihui Liu, Rui Zhang, Robert Mullins, Bei Yu, Ping Luo  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.00228v1  

#### Abstract
In semiconductor manufacturing, lithography projects circuit layouts onto silicon wafers through an optical mask. As circuit features shrink below the wavelength of light, optical diffraction causes the printed patterns to deviate from their intended layouts. Inverse Lithography Technology (ILT) add...

---

### 19. [ProjQ: Project-and-Quantize for Adapter-Aware LLM Compression](https://arxiv.org/abs/2606.00494)

**Authors**: Wneya Yu, Chao Zhang, Li Wang, Samson Lasaulce, Merouane Debbah  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.00494v1  

#### Abstract
Post-Training Quantization (PTQ) and Low-Rank Adaptation (LoRA) constitute the standard pipeline for efficient Large Language Model (LLM) deployment. However, applying them sequentially poses a problem: PTQ often leaves behind random noise that is spread out (across the model's weights) in a way LoR...

---

### 20. [LASER: Loss-Aware Singular-value Decomposition and Rank Allocation for Efficient Low-Precision Vision-Language Models](https://arxiv.org/abs/2606.00573)

**Authors**: Haiyu Wang, Yutong Wang, Leshu Li, Yihui Ren, Sai Qian Zhang  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.00573v1  

#### Abstract
Vision-language models (VLMs) deliver strong multimodal reasoning capabilities, but their large computational cost and high parameter counts make deployment challenging on resource-constrained devices. Low-rank decomposition has emerged as a promising compression technique, yet existing methods ofte...

---

### 21. [Efficient Test-time Inference for Generative Planning Models](https://arxiv.org/abs/2606.00618)

**Authors**: Robert Gieselmann, Mihai Samson, Federico Pecora, Jeremy L. Wyatt  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.00618v1  

#### Abstract
Generative models have emerged as a powerful paradigm for AI planning, yet their performance remains constrained by the training data distribution. One approach is to improve generated solutions during inference by scaling test-time compute. A more efficient alternative is to optimize the inference ...

---

### 22. [Latent Reward Steering: An Adaptive Inference-Time Framework that Implicitly Promotes Cognitive Behaviors in Reasoning LLMs](https://arxiv.org/abs/2606.00726)

**Authors**: Jiakang Li, Guanyu Zhu, Can Jin, Chenxi Huang, Dexu Yu, Ronghao Chen, Yang Zhou, Hongwu Peng, Xuanqi Lan, Dimitris N. Metaxas, Youhua Li  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.00726v1  

#### Abstract
Strong reasoning depends not only on model knowledge but also on how effectively cognitive behaviors are deployed during generation. Existing methods often rely on explicit behavior-level control, making them insufficiently adaptive when failures and required corrections vary across reasoning states...

---

### 23. [Recognize Your Orchestrator: An Entropy Dynamics Perspective for LLM Multi-Agent Systems](https://arxiv.org/abs/2606.01351)

**Authors**: Junze Zhu, Weihao Chen, Xuanwang Zhang, Zhen Wu, Xinyu Dai  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.01351v1  

#### Abstract
The transition from single-turn models to Multi-Agent Systems (MAS) promises enhanced problem-solving capabilities, yet the centralized orchestration topology remains a critical point of fragility. To analyze this, we propose a Mean-Field Entropy Dynamics framework, modeling the orchestration proces...

---

### 24. [MobEvolve: An Agentic Self-Evolving Heuristic System for Interpretable Human Mobility Generation](https://arxiv.org/abs/2606.01640)

**Authors**: Junlin He, Yihong Tang, Tong Nie, Ao Qu, Yuebing Liang, Hamzeh Alizadeh, Bang Liu, Wei Ma, Lijun Sun  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.01640v1  

#### Abstract
Human mobility generation aims to synthesize realistic trip chains for target populations based on individual features. Existing paradigms, including deep generative models, LLM-based methods, and traditional heuristics, struggle to satisfy the complex demands of this task while simultaneously maint...

---

### 25. [Cost-Aware Diffusion Draft Trees for Speculative Decoding](https://arxiv.org/abs/2606.01813)

**Authors**: Shuai Zhang, Huachuan Qiu, Hongliang He, Yong Dai  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.01813v1  

#### Abstract
Speculative decoding accelerates inference by having a lightweight drafter propose tokens verified in parallel by the target language model. Block diffusion drafters such as DFlash generate an entire draft block in one pass, yielding per-position marginals; DDTree uses these to build a candidate tre...

---

### 26. [DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding](https://arxiv.org/abs/2606.02091)

**Authors**: Jiebin Zhang, Zhenghan Yu, Song Liu, Eugene J. Yu, Zheng Li, Dawei Zhu, Jiangshan Duo, Weimin Xiong, Yifan Song, Guanghua Yu, Jianchen Zhu, Sujian Li  
**Category**: cs.CL  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.02091v1  

#### Abstract
Block diffusion speculative decoding accelerates LLM inference by predicting all tokens within a block simultaneously for the target model to verify in parallel. Predicting an entire block at once requires a sufficiently capable draft model and effective utilization of the target model's internal kn...

---

### 27. [The Cartan-Topos Protocol: A Unified Geometric and Categorical Framework for Resilient Multi-Agent Coordination](https://arxiv.org/abs/2606.00714)

**Authors**: Manuel Hern\'andez, Eduardo S\'anchez-Soto  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.00714v1  

#### Abstract
Multi-agent coordination faces a fundamental divide between continuous Euclidean consensus, which fails under non-integrable constraints, and discrete symbolic logic, which collapses under open-world assumptions. This report presents a unified geometric and categorical framework bridging these parad...

---

### 28. [Observation, Not Prediction: Conversation-Level Disaggregated Scheduling for Agentic Serving](https://arxiv.org/abs/2606.01839)

**Authors**: Jianru Ding, Ryien Hosseini, Pouya Mahdi Gholami, Mingyuan Xiang, Henry Hoffmann  
**Category**: cs.DC  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.01839v1  

#### Abstract
LLM-based agents resolve a user task through many turns of dependent inference and tool calls, producing a workload whose total cost is unknown when the task arrives. Existing multi-turn systems keep the turn as the scheduling unit and decide, turn by turn, whether to disaggregate prefill from decod...

---

### 29. [Learning-based Directed Graph Abstraction of Combinatorial Spaces for Order-Preserving Search in Mixed-Combinatorial Nonlinear Optimization](https://arxiv.org/abs/2606.01425)

**Authors**: Gishnu Madhu, Feng Liu, Souma Chowdhury  
**Category**: cs.LG  
**Published**: 2026-06-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.01425v1  

#### Abstract
Mixed-combinatorial nonlinear programming (MCNLP) problems arise in many engineering design and planning applications, e.g., due to categorical, component, and geometric design choices, as well as joint task and motion planning. Traditional representations of combinatorial spaces, such as integer or...

---

### 30. [Emergent Collaborative Deliberation in Multi-Model AI Systems: A BFT-Derived Protocol for Epistemic Synthesis](https://arxiv.org/abs/2606.00005)

**Authors**: VD Doske  
**Category**: cs.AI  
**Published**: 2026-06-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.00005v1  

#### Abstract
We present the Consilium Protocol, a Byzantine Fault Tolerance-derived architecture for structured multi-model AI deliberation that treats inter-model disagreement as epistemic signal rather than error. The protocol assigns engineered cognitive personas to language models -- separating what a model ...

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
