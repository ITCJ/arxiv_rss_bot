# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-02-06 06:36:49 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference](https://arxiv.org/abs/2602.05145)

**Authors**: Jiyoung Park, Hankyu Jang, Changseok Song, Wookeun Jung  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2602.05145v1  

#### Abstract
Speculative decoding can substantially accelerate LLM inference, but realizing its benefits in practice is challenging due to evolving workloads and system-level constraints. We present TIDE (Temporal Incremental Draft Engine), a serving-engine-native framework that integrates online draft adaptatio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
- **Speculative Decoding 在动态负载下的性能退化**：尽管 speculative decoding 能显著加速 LLM 推理，但其效果高度依赖于 draft model 和 target model 的对齐程度。在真实生产环境中，输入 workloads 随时间不断变化（如用户行为、prompt 模板更新），导致 draft-target alignment 下降，acceptance rate 锐减，从而削弱甚至抵消加速收益。
- **在线训练开销大且难以集成到高性能推理系统中**：现有在线适应方法通常需要重新运行 target model 来生成训练信号（如 logits 或 hidden states），带来额外计算开销，并占用宝贵的推理资源。

### 提出了什么新方法或新思路
提出 **TIDE (Temporal Incremental Draft Engine)** ——一种**服务引擎原生的自适应 speculative decoding 框架**，具备以下核心设计：

1. ✅ **零开销训练信号提取（Zero-overhead Training Signal Generation）**
   - 复用 target model 在验证阶段已计算出的中间 hidden states 作为训练信号，无需重新加载或重算 target model。
   - 实现方式：在推理过程中异步提取并传输 hidden states 至共享存储，与主推理流水线重叠执行，几乎无性能损失。

2. ✅ **自适应运行时控制（Adaptive Runtime Control）**
   - 动态判断何时启用 speculative decoding 和 draft model 训练。
   - 基于 batch size 和 acceptance length 实时估算 speedup，仅当有益时才开启 speculation。
   - 通过监控 acceptance rate 的短期与长期移动平均值，检测分布偏移后触发训练，避免无效训练。

3. ✅ **异构 GPU 利用（Heterogeneous GPU Utilization）**
   - 将 inference serving 与 draft model training 解耦，分别部署在不同类型的 GPU 上。
   - 示例：H100 执行高吞吐推理，MI250 承担 draft model 训练任务，提升整体集群利用率。

4. ✅ **增量式在线 draft model 更新机制**
   - draft model 异步训练并在验证性能提升后热更新至推理引擎，实现持续自我优化。

### 相比现有方法的优势
| 维度 | 现有方法（如 SpecForge） | TIDE |
|------|------------------------|------|
| **训练信号获取** | 需离线 prefill 或在线重算 target model 输出 | 复用推理过程中的 hidden states，零额外开销 |
| **训练效率** | 存储开销大（offline）或训练慢（online） | 训练时间减少 1.67×，存储需求降低 >20× |
| **系统集成性** | 多为独立训练流程，难嵌入生产系统 | 原生集成于推理引擎，支持实时自适应 |
| **资源利用** | 通常使用同构 GPU | 支持异构部署，提升硬件利用率 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ShareGPT**：开放对话数据集（多语言子集用于模拟分布偏移）
- **Science**：科学文本（来自 CAMEL-AI.org 的 biology/chemistry/physics 数据）
- **EvolCodeAlpaca**：代码生成任务
- **NuminaMath**：数学推理任务
- **Alpaca 多语言变体**（Korean, Arabic, Chinese, French）：用于测试跨语言分布偏移场景

### 实验设置
- **Target Models**：
  - `gpt-oss-120b`, `Qwen3-235B-A22B`, `Llama-4-Scout-17B-16E`, `Llama-3.3-70B-Instruct`
- **Draft Model 架构**：
  - 基于 EAGLE-3，单 decoder layer + LM head，预测 next token 基于 target model 的 low/middle/high 层 hidden states
- **候选 token 数量（γ）**：固定为 3（经消融实验证明最优）
- **硬件配置**：
  - 推理节点：NVIDIA H100（8 GPUs）
  - 训练节点：AMD Instinct MI250（4 GPUs）
- **系统实现基础**：
  - 推理引擎基于 **SGLang**
  - 训练引擎基于 **SpecForge**

### 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput** | 单位时间内处理的 token 数量（越高越好） |
| **Acceptance Length (E[l])** | 平均每轮 speculation 成功接受的 draft token 数量 |
| **Speedup** | 相对于 vanilla autoregressive decoding 的加速比 |
| **Training Time** | 完成一轮 draft model 微调所需时间 |
| **Storage Overhead** | 存储训练信号所需的磁盘空间 |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **SpecForge Offline** | 先 prefill 获取 hidden states 并持久化，再训练；存储开销大 |
| **SpecForge Online** | 每次训练都重新运行 target model 生成 hidden states；计算开销高 |
| **TIDE-default** | 始终启用 speculation，不进行动态控制 |
| **TIDE-adaptive** | 启用 adaptive 控制逻辑（speculation 开关 + selective training） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔹 吞吐量提升
- 在多个真实 workload 上，TIDE 实现 **最高达 1.15× 的端到端 throughput 提升**（相比静态 speculative decoding）。
- 提升幅度因 dataset 而异：
  - **Science**: +1.15×
  - **NuminaMath**: +1.12×
  - **EvolCodeAlpaca**: +1.10×
  - **ShareGPT**: 提升有限（仅 ~1.02×），因其 high entropy 和 discourse 变化频繁，speculative decoding 本身增益较小。

#### 🔹 训练效率对比（以 gpt-oss-120b + ShareGPT 为例）
| 方法 | Prefill 时间 | Train 时间 | 总耗时 | Speedup |
|------|-------------|-----------|--------|---------|
| SpecForge Offline | 6.16 hr | 9.16 hr | 15.32 hr | 1.00× |
| SpecForge Online | 18.48 hr | 9.16 hr | 27.64 hr | 0.55× |
| **TIDE** | — | **9.16 hr** | **9.16 hr** | **1.67×** |

> ✅ TIDE 消除了 prefill 开销，训练速度是 offline 的 **1.67×**，是 online 的 **3.02×**

#### 🔹 存储开销对比
| Target Model | SpecForge Offline | TIDE |
|--------------|--------------------|-------|
| gpt-oss-120b | 4.66 TB | 0.19 TB (**↓96%**) |
| Qwen3-235B-A22B | 19.89 TB | 0.82 TB |
| Llama-3.3-70B-Instruct | 46.40 TB | 1.92 TB |

> ✅ TIDE 仅需缓存当前训练批次的 hidden states，存储需求降低两个数量级。

#### 🔹 异构 GPU 利用效果
- **推理吞吐差距大**：H100 推理吞吐是 MI250 的 **6.76×**
- **训练吞吐差距小**：H100 训练吞吐仅为 MI250 的 **2.44×**
- 因此将训练任务交给低性能 GPU 更划算。

最终系统级吞吐提升：
- 使用 H100（推理）+ MI250（训练）组合，TIDE 实现 **1.08–1.22× 的相对 throughput 提升**（vs all-inference baseline）。

#### 🔹 自适应控制有效性（Figure 9）
- 在连续语言切换（Korean → Arabic → Chinese → French）的压力测试中：
  - **TIDE-default**：遇到分布偏移时 throughput 明显下降
  - **TIDE-adaptive**：自动关闭 speculation，避免负优化，更快恢复性能

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Speculative decoding 的收益高度依赖 workload 特性**：
   - 结构化输出（如 Science、Math）更易学习，acceptance length 更高。
   - 开放式对话（如 ShareGPT）难以有效 speculation。

2. ✅ **零开销训练信号复用是可行且高效的**：
   - 利用推理过程中的副产品（hidden states）可完全替代昂贵的 re-computation。

3. ✅ **自适应控制至关重要**：
   - 不加选择地始终 speculation 或 training 会导致资源浪费甚至性能倒退。
   - 基于 acceptance length 和 batch size 的动态决策能显著提升鲁棒性。

4. ✅ **异构 GPU 分配策略具有实际价值**：
   - 高端 GPU 擅长推理，低端 GPU 在训练轻量 draft model 上性价比更高。
   - 解耦训练与推理使异构集群利用率最大化。

### 方法的局限性
- ❗ **依赖特定 draft model 架构（如 EAGLE-3）**：必须基于 target model 的 hidden states 进行预测，通用性受限。
- ❗ **仅适用于支持 hidden state 输出的推理框架**：需深度集成至 SGLang/vLLM 等系统。
- ❗ **对 extremely high-entropy workloads 效果有限**：如自由创作类任务，speculative decoding 本身增益小。
- ❗ **冷启动问题**：初始 draft model 性能差时可能无法触发有效训练信号收集。

### 未来工作方向
- 🔄 **探索更通用的 draft model 架构**，降低对 target model hidden states 的依赖。
- 🧠 **引入强化学习或 bandit 算法**，进一步优化 speculation 和 training 的触发策略。
- ⚙️ **支持更多异构设备类型**（如 Intel GPUs、国产加速器）。
- 📈 **扩展至多 draft model 协同机制**，应对更复杂的混合 workload 场景。

--- 

> 💡 **总结一句话**：  
> **TIDE 将 speculative decoding 从“静态加速技巧”转变为“动态自优化系统”，通过零开销信号复用、自适应控制和异构资源调度，在真实非平稳 workload 中实现了可持续的推理加速。**

</details>

---

### 2. [Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs](https://arxiv.org/abs/2602.05191)

**Authors**: Wentao Ni, Kangqi Zhang, Zhongming Yu, Oren Nelson, Mingu Lee, Hong Cai, Fatih Porikli, Jongryool Kim, Zhijian Liu, Jishen Zhao  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2602.05191v1  

#### Abstract
As long-context inference becomes central to large language models (LLMs), attention over growing key-value caches emerges as a dominant decoding bottleneck, motivating sparse attention for scalable inference. Fixed-budget top-k sparse attention cannot adapt to heterogeneous attention distributions ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在**长上下文大语言模型（LLMs）推理**中，随着上下文长度增长（可达数万甚至数十万 tokens），**注意力机制的计算开销急剧上升**，成为解码阶段的主要瓶颈。传统的 **fixed-budget top-k sparse attention** 方法存在以下缺陷：

- **无法适应注意力分布的异质性**：不同层（layer）、头（head）和解码步（decode step）的注意力分布差异显著，固定预算（如固定选择 256 个 token）会导致某些情况下保留不足（精度损失），另一些情况过度保留（效率低下）。
- **缺乏对注意力质量的显式控制**：top-k 方法不保证保留的注意力质量（即累计注意力质量 mass），而 **top-p 方法通过设定概率阈值 p 来保留前 p% 的注意力质量**，理论上能提供更强的精度保障。

然而，现有 top-p 方法（如 *Twilight*）也存在问题：
- **token-level 估计成本高**：需要对所有 token 进行近似注意力打分，计算和排序开销随上下文线性增长。
- **估计不可靠**：由于使用固定 token 预算进行估计，导致实际保留的注意力质量波动大，常低于目标 p 值（如 p=0.95）。

### 🚀 提出的新方法：Double-P

作者提出 **Double-P**，一种**分层的 top-p 稀疏注意力框架**，通过两个阶段的 top-p 选择实现高效且准确的稀疏化：

#### 创新点 1：**两阶段分层 top-p 设计**

1. **第一阶段：Cluster-Level Top-P Estimation**
   - 在预填充阶段将 KV Cache 按 key 向量聚类为多个 cluster。
   - 使用 **size-weighted centroid**（带大小权重的聚类中心）快速估算每个 cluster 的注意力得分。
   - 在 cluster 粒度上执行 top-p，筛选出可能贡献重要注意力的 cluster 集合。

2. **第二阶段：Adaptive Token-Level Top-P Refinement**
   - 对选中的 cluster，动态决定是否进行精确的 token-level 注意力计算。
   - 引入第二个 top-p 参数 $p_2$，仅对高影响 cluster 执行精确计算，其余用 centroid 近似。
   - 实现“按需精算”，避免不必要的 token 级计算。

#### 创新点 2：高效的 GPU Kernel 实现
- 自定义 **Top-P kernel**，支持对已排序张量进行前缀和 + 早停。
- 融合 **token 和 cluster 的 gather 操作**，提升内存局部性。
- 使用 **FlashAttention 变体**统一处理精确和近似部分，减少 kernel launch 开销。

### 🔍 相比现有方法的优势

| 方法 | 优势 |
|------|------|
| **vs. Top-k (Quest, RetroInfer)** | 支持自适应预算，提供 top-p 精度保证，避免固定预算的不平衡问题 |
| **vs. Token-level Top-p (Twilight)** | 显著降低估计开销（从 token 级降到 cluster 级），避免线性增长的 SpGEMV 和排序 |
| **vs. Cluster-based (RetroInfer)** | 引入概率驱动的 top-p 控制，而非固定 cluster 数量，精度更高 |

---

## 2. 核心实验方法和设置

### 📊 数据集

- **RULER** (Hsieh et al., 2024)  
  包含 13 个任务，上下文长度从 4K 到 128K，用于综合评估长上下文能力。
- **LongBench** (Bai et al., 2024)  
  包含 21 个真实场景任务，涵盖问答、摘要、推理等，平均输入长度 5K–15K。

### ⚙️ 实验设置

- **模型**：LLaMA-3.1-8B 和 Qwen-3-8B，上下文长度测试至 128K。
- **硬件**：单 NVIDIA H100 PCIe GPU (80GB)，CUDA 12.8，PyTorch 2.8。
- **保留策略**：所有方法均保留 **sink tokens (4)** 和 **sliding window tokens (64)**，稀疏注意力仅作用于中间 tokens。

### 📈 评估指标

| 指标 | 描述 |
|------|------|
| **Accuracy** | 下游任务平均得分（如 RULER Avg., LongBench Avg.） |
| **End-to-End Decoding Latency** | 每输出 token 的平均延迟（ms） |
| **Attention Latency Breakdown** | 注意力各阶段耗时（SpGEMV、Top-P 选择、稀疏注意力） |
| **Speedup** | 相对于基线或全注意力的速度提升倍数 |

### 🆚 基线方法对比

| 基线 | 类型 | 说明 |
|------|------|------|
| **Quest** | Page-level top-k | 基于 key bounds 选择 top-k pages，取 25% tokens |
| **RetroInfer** | Cluster-based top-k | 固定数量 cluster 检索，结合 centroid 近似 |
| **Quest + Twilight** | Token-level top-p | 在 Quest 基础上应用 Twilight 的 top-p 选择 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ 准确率表现（LLaMA-3.1-8B）

| 方法 | RULER (16K) | RULER (32K) | RULER (64K) | LongBench |
|------|------------|------------|------------|-----------|
| Full Attention | 93.25 | 90.00 | 85.36 | 39.33 |
| Quest | 89.24 | 85.95 | 83.61 | 38.76 |
| RetroInfer | 91.56 | 88.12 | 83.77 | 39.03 |
| Quest + Twilight | 86.73 | 86.50 | 81.87 | 38.97 |
| **Double-P (Ours)** | **92.87** | **89.91** | **84.55** | **39.06** |

- **相比最强基线 RetroInfer**，Double-P 在 RULER 上分别提升 **+1.31**, **+1.79**, **+0.78** 绝对分。
- **平均提升 +1.26 分**，**接近全注意力精度**，实现“near-zero accuracy drop”。

#### ⚡ 效率表现

| 指标 | 结果 |
|------|------|
| **Attention-level Speedup** | 最高达 **1.74× vs Quest-Twilight**, **1.78× vs RetroInfer** |
| **End-to-End Decoding Speedup** | 最高 **1.26× vs RetroInfer**, **1.11× vs Quest** |
| **相比 Full Attention** | 最高 **2.23× 加速** |
| **Top-p Estimation Overhead** | Double-P 将估计开销降至可忽略水平（见 Figure 9） |

#### 🔍 消融实验（Ablation Study）

- 图 10 展示了不同 $(p_1, p_2)$ 配置下的精度-延迟权衡：
  - $p_1$: cluster-level top-p 阈值
  - $p_2$: token-level refinement 阈值
- 实践中选择 $(p_1, p_2) = (0.95, 0.7)$ 在 LLaMA-3.1-8B 上取得最佳平衡。
- 更高的 $p_1$ 和 $p_2$ 提升精度但增加延迟，验证了设计的可控性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Fixed-budget top-k 方法无法可靠满足 top-p 要求**  
   实验表明，即使使用平均预算（如 k=256），仍有超过 20% 的 attention head 无法达到 p=0.95 的质量要求。

2. **Token-level top-p 估计成本过高**  
   SpGEMV 和排序操作占总延迟的 **60% 以上**，严重限制端到端加速潜力。

3. **Double-P 实现了精度与效率的帕累托前沿**  
   如 Figure 1 所示，Double-P 在精度-延迟图中形成 **Pareto frontier**，优于所有基线。

4. **分层设计是关键**  
   先用 cluster-level 低成本估计缩小搜索空间，再用 adaptive token-level refinement 精细控制，实现了“**粗中有细，细中求省**”。

### ⚠️ 方法的局限性

- **依赖聚类质量**：k-means 聚类假设语义相似 token 可被聚在一起，极端稀疏或噪声数据可能影响效果。
- **额外预处理开销**：聚类过程在 prefill 阶段引入一定计算成本（但可接受）。
- **超参数调优需求**：$p_1$ 和 $p_2$ 需根据模型和任务调整，自动化调参可作为未来方向。

### 🔮 未来工作方向

- **动态调整 $p_1$, $p_2$**：基于当前 query 或上下文复杂度自动调节阈值。
- **与其他优化技术结合**：如 KV Cache 量化（KVQuant）、GQA/PagedAttention 等。
- **扩展到训练阶段**：探索 Double-P 在训练中的可行性，进一步降低长序列训练成本。
- **硬件协同设计**：针对 Double-P 的访问模式设计专用加速器或 kernel。

---

> **总结**：Double-P 通过**分层 top-p 稀疏注意力框架**，成功解决了长上下文 LLM 推理中**精度与效率难以兼顾**的问题。它不仅提供了**更强的 top-p 精度保证**，还通过**cluster-level 估计 + adaptive refinement** 显著降低了计算开销，在多个基准上实现了**接近全注意力的精度**和**最高 2.23× 的端到端加速**，为未来长上下文模型的高效部署提供了坚实基础。

</details>

---

### 3. [OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale](https://arxiv.org/abs/2602.05711)

**Authors**: Jingze Shi, Zhangyang Peng, Yizhang Zhu, Yifan Wu, Guang Liu, Yuyu Luo  
**Category**: cs.CL  
**Published**: 2026-02-06  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2602.05711v1  

#### Abstract
Mixture-of-Experts (MoE) architectures are evolving towards finer granularity to improve parameter efficiency. However, existing MoE designs face an inherent trade-off between the granularity of expert specialization and hardware execution efficiency. We propose OmniMoE, a system-algorithm co-design...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **Mixture-of-Experts (MoE)** 架构在**专家粒度**上面临根本性权衡：
- **粗粒度 MoE**（如 DeepSeekMoE）：硬件效率高（dense matmul），但激活不精确，存在冗余计算。
- **细粒度 MoE**（如 PEER）：参数利用率高，但因内存访问分散导致严重的 **memory-bound** 问题，推理延迟极高。

OmniMoE 的目标是：**在保持细粒度专家高参数效率的同时，实现粗粒度架构级别的硬件执行效率**。

---

### 🚀 提出的新方法与创新思路

OmniMoE 是一个 **system-algorithm co-designed** 框架，通过三项核心技术协同解决上述挑战：

#### （1）Atomic Experts + Dynamic Expert Assembly (DEA)
- 将专家粒度推向逻辑极限——**Atomic Expert**：每个专家仅由一对向量 $(w_{\text{in}}, w_{\text{out}})$ 参数化，构成最小可路由单元。
- 引入 **Dynamic Expert Assembly (DEA)**：对每个 token 动态检索并组合多个 Atomic Experts，形成 token-specific 的高效非线性变换。
- 所有专家参数集中存储为全局矩阵 $W, V \in \mathbb{R}^{N \times d}$，支持高效 gather 操作。

> ✅ 优势：极大提升模型表达能力与长尾知识检索精度，同时避免静态 embedding 类设计（如 PKM）缺乏非线性变换的问题。

#### （2）Cartesian Product Router
- 面对百万级专家带来的路由开销（$O(Nd)$ 投影不可行），提出将一维专家索引空间分解为二维网格（$N_r \times N_c$, $N = N_r N_c$）。
- 路由器分别预测行分布 $p_r(i|x)$ 和列分布 $p_c(j|x)$，联合得分 $p(i,j|x) \approx p_r(i|x) \cdot p_c(j|x)$。
- 实现方式：两个低维投影 $W_r, W_c$ 替代单一 $W_g$，将路由复杂度从 $O(Nd)$ 降至 $O(\sqrt{N}d)$。

> ✅ 优势：使大规模细粒度路由变得可行且高效；支持高达百万级专家池。

#### （3）Expert-Centric Scheduling
- 改变传统 **token-centric** 执行顺序（每个 token 独立拉取参数），转为 **expert-centric** 执行范式。
- 步骤：
  1. 收集所有 token 的路由请求；
  2. 按激活专家分组；
  3. 在每组内按 token ID 排序；
  4. 使用 **Grouped GEMM** 批量处理同一专家下的多个 token。
- 效果：将随机内存访问（scatter/gather）转化为连续、可重用的密集矩阵运算。

> ✅ 优势：彻底缓解 memory bandwidth 瓶颈，GPU 利用率显著提升。

---

### 🔍 相比现有方法的优势
| 维度 | 粗粒度 MoE（如 DeepSeekMoE） | 细粒度 MoE（如 PEER） | **OmniMoE（本工作）** |
|------|-------------------------------|------------------------|--------------------------|
| 参数效率 | 低（大块激活，冗余多） | 高（精准控制） | ⭐ 极高（atomic level） |
| 表达能力 | 高（完整 FFN 结构） | 低（常为线性聚合） | ⭐ 高（动态组装非线性块） |
| 路由效率 | 高（小专家数） | 低（全投影代价大） | ⭐ 高（factorized routing） |
| 内存访问 | 连续（dense） | 分散（random I/O） | ⭐ 连续（grouped coalescing） |
| 推理速度 | 快 | 慢（memory-bound） | ⭐⭐ 极快（compute-bound） |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **预训练语料**：SmolLMCorpus（400亿 token）
  - 包含 Web、Textbook、Code、Math 四类高质量文本。
- **下游评估基准**（7项零样本任务）：
  - **MMLU**（多任务知识）
  - **TriviaQA**（事实回忆）
  - **ARC**（科学推理）
  - **PIQA**（物理常识）
  - **HellaSwag**（常识推断）
  - **OBQA**（开放书本问答）
  - **Winogrande**（共指消解）

> 使用 Hugging Face LightEval 工具包统一评测。

---

### ⚙️ 实验设置
- **模型规模**：主比较使用 **6.4B 总参数 / 1.7B 激活参数** 的 MoE 模型。
- **骨干网络一致**：所有方法共享相同的 Transformer 结构（depth, width, GQA 等），仅替换 FFN 模块。
- **公平对比原则**：
  - 所有模型从头预训练（scratch training），排除 checkpoint 差异影响。
  - 控制相同激活参数预算、训练 FLOPs、数据集。
- **评估指标**：
  - **下游性能**：zero-shot accuracy 平均值
  - **系统效率**：inference latency（ms）、peak memory（GB）
  - **扩展性分析**：scaling laws（perplexity vs. FLOPs / Act Params）

---

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Dense** | 全激活 MLP | 基准上限（无 MoE） |
| **Gshard** | Coarse-grained MoE | Top-K 路由，标准实现 |
| **DeepSeekMoE** | Coarse-grained MoE | 含 shared expert，当前主流 |
| **PKM** | Fine-grained MoE | Product Key Memory 设计 |
| **PEER** | Fine-grained MoE | 百万级轻量专家，state-of-the-art 细粒度方案 |
| **OmniMoE (Ours)** | Hybrid Fine-grained | Atomic Experts + Cartesian Router + Expert-Centric Scheduling |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）下游任务表现（Zero-Shot Accuracy）
| Model | MMLU | TriviaQA | ARC | PIQA | HellaSwag | OBQA | Winogrande | **Avg** |
|-------|------|----------|-----|------|-----------|--------|------------|--------|
| Dense | 35.4 | 9.4 | 53.4 | 72.9 | 56.1 | 37.0 | 57.3 | 45.9 |
| DeepSeekMoE | 37.1 | 17.4 | 60.7 | 77.2 | 61.2 | 38.9 | 59.1 | 50.2 |
| PEER | 37.4 | 16.9 | 57.4 | 75.9 | 56.3 | 39.1 | 59.4 | 48.9 |
| **OmniMoE (Ours)** | **37.5** | **18.5** | **61.0** | **78.7** | **60.9** | **40.3** | **59.7** | **50.9** |

> ✅ OmniMoE 在 **7项平均准确率上达到 50.9%**，超越最强基线 DeepSeekMoE（+0.7）和细粒度 PEER（+2.0）。

#### （2）推理效率（Latency & Memory）
- 输入长度：**4,096 tokens**
- 激活参数相近（~28M）

| Method | Latency (ms) | Speedup vs PEER |
|--------|---------------|------------------|
| PEER | 73.0 | 1× |
| DeepSeekMoE | 102.0 | — |
| **OmniMoE (Ours)** | **6.7** | **10.9× faster** |

> ⚡ OmniMoE 实现 **10.9倍于 PEER 的推理加速**，且内存占用与粗粒度 MoE 相当。

#### （3）Scaling Laws 表现
- 在不同规模下（80M → 1.7B 激活参数），OmniMoE 始终以更低 FLOPs 达到更优 validation perplexity。
- 表明其兼具 **更高的计算效率（compute efficiency）和参数效率（parameter efficiency）**。

---

### 🔍 消融实验结果（Ablation Study）

| 方法变体 | Latency↑ | Memory↑ | PPL↑ | Knowledge↓ | Reasoning↓ | Expert Usage↓ | Unevenness↑ |
|---------|--------|--------|------|-------------|--------------|----------------|--------------|
| Full Model | 1.0x | 1.0x | 1.0× | 1.0x | 1.0x | 100% | 0.24 |
| w/o Shared MLP | 0.86x | 0.98x | 1.2x | 0.91x | 0.79x | 100% | 0.27 |
| w/o Cartesian Router | 30.6x | 337.5x | 1.4x | 0.66x | 0.79x | 4% | 0.77 |
| w/o Expert-Centric Sched | 24.8x | 417.7x | 1.0x | 1.0x | 1.0x | 100% | 0.24 |

> 💡 发现：
- **Cartesian Router** 对降低路由开销至关重要，否则内存暴涨 300+ 倍。
- **Expert-Centric Scheduling** 是性能飞跃的关键，消除 memory bottleneck。
- **Shared MLP** 虽轻微增加成本，但显著提升泛化与推理能力，不可或缺。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **细粒度 MoE 可以既快又准**：通过算法-系统协同设计，OmniMoE 成功打破“细粒度=慢”的固有认知。
2. **Atomic Expert + DEA** 提供了极致灵活的参数组合机制，实现 token-level 精细化激活。
3. **Cartesian Product Router** 有效解决了百万级专家下的路由爆炸问题，复杂度降至 $O(\sqrt{N})$。
4. **Expert-Centric Scheduling** 是性能突破的核心，将 scattered I/O 转换为 Grouped GEMM，释放 Tensor Core 潜能。
5. **混合架构优越性**：shared dense MLP 处理通用语义，routed atomic experts 专注长尾知识，二者互补。

---

### ⚠️ 局限性
- 当前实现依赖 Triton 自定义 kernel，在通用性上可能受限于特定硬件平台（如 NVIDIA GPU）。
- 虽然通信开销饱和（见 Appendix C），但在超大规模分布式训练中仍需进一步验证稳定性。
- 对 extremely sparse 场景（极少数 token 激活某专家）的 Grouped GEMM 利用率可能下降。

---

### 🔮 未来工作方向
- 探索更灵活的 **multi-level routing hierarchy**，结合 coarse + atomic 专家。
- 将 OmniMoE 思路推广至 **vision, multimodal, and agent-based models**。
- 开发自动编译器支持 **automatic scheduling optimization**，降低部署门槛。
- 研究如何动态调整 Atomic Expert 数量与结构，实现 lifelong learning。

---

> 🔗 **代码已开源**：[https://github.com/flash-algo/omni-moe](https://github.com/flash-algo/omni-moe)  
> 📄 Preprint 发布时间：February 6, 2026

</details>

---

### 4. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)

**Authors**: Jian Chen, Yesheng Liang, Zhijian Liu  
**Category**: cs.CL  
**Published**: 2026-02-06  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2602.06036v1  

#### Abstract
Autoregressive large language models (LLMs) deliver strong performance but require inherently sequential decoding, leading to high inference latency and poor GPU utilization. Speculative decoding mitigates this bottleneck by using a fast draft model whose outputs are verified in parallel by the targ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DFlash: Block Diffusion for Flash Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

大型语言模型（LLMs）在推理时采用**自回归生成**（autoregressive generation），逐个 token 生成输出，导致严重的**序列化瓶颈**，表现为：
- 推理延迟高
- GPU 利用率低
- 长文本生成效率差

尽管已有**投机解码**（speculative decoding）技术通过轻量级 draft model 加速推理，但主流方法（如 EAGLE-3）仍依赖**自回归 drafting**，无法突破序列生成的限制。

同时，虽然**扩散语言模型**（dLLMs）支持并行生成，但其独立生成质量通常低于自回归模型，且多步去噪过程拖慢速度。

---

### 🚀 提出的新方法与核心创新

**DFlash** 是一种基于**块扩散模型**（block diffusion model）的新型投机解码框架，其核心思想是：

> 将扩散模型作为**高效并行 draft model**，利用目标 LLM 的隐藏特征进行条件引导，实现高质量、低延迟的块级 token 预测。

#### 主要创新点：

1. **并行 drafting 架构**
   - 使用轻量级 block diffusion model 在单次前向传播中并行预测多个 token（block-wise generation）
   - 显著降低 drafting latency，打破 autoregressive drafting 的串行瓶颈

2. **基于目标模型上下文的强条件建模**
   - 从目标 LLM 中提取多层 hidden features，融合为 **target context feature**
   - 通过 **KV 注入机制**（KV injection）将该特征注入到 draft model 的每一层 Key 和 Value 投影中
   - 使 draft model 能够“继承”目标模型对未来 token 的隐含预测能力

3. **训练策略优化**
   - **随机锚点采样**：训练时随机选择响应中的 token 作为 block 起始点，提升数据多样性
   - **位置加权损失函数**：对 block 内靠前的 token 分配更高权重（指数衰减），因为早期错误会阻断整个 block 的接受
   - **共享嵌入层与 LM Head**：与目标模型共享 token embedding 和输出头，减少参数量并增强对齐

4. **轻量化设计**
   - draft model 仅需 5 层 Transformer（Qwen3-Coder 为 8 层），参数极少
   - 支持高效长上下文训练与部署

---

### 🔍 相比现有方法的优势

| 方法 | 类型 | Drafting 方式 | 是否并行 | Acceptance Length | Latency | Memory |
|------|------|----------------|-----------|--------------------|---------|--------|
| EAGLE-3 | Autoregressive | Tree-based | ❌ 串行 | 中等 (~3–4) | 较高 | 低 |
| DiffuSpec / SpecDiff-2 | Diffusion | Full dLLM (7B) | ✅ 并行 | 高 | 高（大模型） | 高 |
| PARD | AR mimic diffusion | Parallel AR | ✅ | 低 | 低 | 低 |
| **DFlash** | **Block Diffusion** | **Parallel block** | ✅✅ | **极高 (~6–8)** | **极低** | **低** |

> ✅ DFlash 实现了**高 acceptance length** 与**低 drafting latency** 的帕累托最优。

---

## 2. 核心实验方法和设置

### 📚 数据集

- **训练数据**：
  - 混合约 80 万样本
  - 来源：NVIDIA Nemotron Post-Training Dataset V2、CodeAlpaca
  - 使用目标模型生成响应以保证对齐（target-aligned responses）

- **评估任务分类**：
  - **Math**：GSM8K、MATH-500、AIME25
  - **Code**：HumanEval、MBPP、LiveCodeBench (LCB)
  - **Chat**：MT-Bench、Alpaca

---

### ⚙️ 实验设置

- **模型**：
  - Qwen3 系列：Qwen3-4B、Qwen3-8B、Qwen3-Coder-30B-A3B-Instruct
  - LLaMA-3.1-8B-Instruct

- **硬件平台**：
  - 主要使用 NVIDIA H200 和 B200 GPU
  - SGLang 框架 + FlashAttention-4（FA4）后端用于真实服务场景测试

- **评估指标**：
  - **平均接受长度**（Average Acceptance Length, $\bar{T}$）：每轮验证成功接受的 token 数
  - **端到端加速比**（End-to-end Speedup）：相对于标准 autoregressive decoding 的吞吐提升
  - **Throughput (tokens/sec)**：在并发请求下的实际吞吐量

- **基线方法对比**：
  - **Baseline**：标准 autoregressive decoding
  - **EAGLE-3**：当前最先进的 speculative decoding 方法（tree-based, autoregressive drafting）
  - （未比较其他 dLLM-based 方法因缺乏开源实现）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & Table 3）

#### 在 Qwen3-8B 上的表现（greedy decoding, temperature=0）：

| 方法 | 平均加速比 | 最高加速比 | 平均接受长度 $\bar{T}$ |
|------|------------|-------------|--------------------------|
| EAGLE-3 (16) | ~1.8–2.2× | — | ~3.0–3.7 |
| **DFlash (16)** | **~4.9×** | **6.1×** | **~6.5–7.9** |

> ✅ DFlash 实现 **超过 6× 的 lossless 加速**，是 EAGLE-3 的 **2.5 倍以上**

#### 在 SGLang 框架下的真实服务表现（Qwen3-8B, concurrency=16）：

| 方法 | Throughput (tok/s) | Speedup | $\bar{T}$ |
|------|---------------------|---------|-----------|
| Baseline | 868 | 1.0× | — |
| DFlash | **4858** | **5.1×** | 8.0 |

> ✅ 即使在高并发下仍保持显著加速，验证了实用性和可扩展性

#### 在 LLaMA-3.1-8B 上的结果（SGLang, FA4 backend）：

| 方法 | Task | Speedup (@concurrency=1) | $\bar{T}$ |
|------|------|----------------------------|-----------|
| EAGLE-3 (60) | HumanEval | 2.0× | 4.65 |
| **DFlash (10)** | HumanEval | **2.8×** | **4.91** |

> ✅ 在不同架构上均优于 EAGLE-3，泛化能力强

---

### 🔍 消融实验结果（Ablation Studies）

#### ▶️ 不同 draft model 深度的影响（Table 5）

| 层数 | Math500 Speedup | $\bar{T}$ |
|------|------------------|-----------|
| 3-L | 4.69× | 5.64 |
| **5-L** | **4.71×** | **5.99** |
| 8-L | 4.64× | 6.33 |

> ✅ **5 层模型取得最佳平衡**：更深虽能提高 $\bar{T}$，但 drafting latency 上升反而降低整体 speedup

#### ▶️ 目标模型隐藏层数量的影响（Table 6）

| 提取 hidden features 层数 | $\bar{T}$ |
|--------------------------|----------|
| 3 | ~4.5 |
| **5** | **~5.6–5.8** |

> ✅ 更多层特征提供更丰富的上下文，显著提升 acceptance length

#### ▶️ 训练与推理 block size 匹配性（Table 7）

| Train → Test | Math500 $\bar{T}$ |
|--------------|---------------------|
| 16 → 16 | 6.33 |
| 16 → 8 | 5.09 |
| 8 → 16 | 5.02 |
| 8 → 8 | 5.21 |

> ✅ **大 block 训练模型可良好泛化至小 block 推理**，支持动态调度；反之不行

#### ▶️ 损失函数加权 vs 均匀加权（Figure 5）

- 使用指数衰减的位置加权损失 → **收敛更快、acceptance length 更高**

#### ▶️ 是否使用目标模型上下文（Table 8）

- 无 context feature 的 diffusion drafter：
  - 仅达到 ~2.8–3.7× speedup
  - $\bar{T} \approx 3.3–4.6$
> ❗证明：**目标模型的 hidden features 是实现高质量 drafting 的关键**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **扩散模型不必追求端到端生成质量**，可在 speculative decoding 中作为高性能 draft model 发挥独特优势。
2. **目标模型的 hidden features 含有丰富的未来 token 信息**，可用于指导 draft model 进行高质量并行预测。
3. **KV injection + block diffusion** 架构实现了 drafting latency 与 acceptance length 的双重优化。
4. DFlash 在多种模型、任务、框架下均实现 **>6× lossless 加速**，远超 EAGLE-3 等 SOTA 方法。
5. 其设计允许灵活调整 block size、depth 等参数，在不同部署场景中保持高效。

---

### ⚠️ 方法的局限性

1. **依赖目标模型的 hidden states 提取**，需修改或 hook 目标模型内部结构，可能增加集成复杂度。
2. 当前实现主要针对 decoder-only 模型，对 encoder-decoder 架构适配尚不明确。
3. 虽然 draft model 很小，但训练阶段需要缓存大量 target hidden features，存储开销较大（尤其离线训练）。
4. 未开放与其他 dLLM-based speculative 方法（如 DiffuSpec）的直接对比（因无开源代码）。

---

### 🔮 未来工作方向

1. **自适应 block size 调度**：根据负载动态调整 block 大小以最大化吞吐
2. **zero-shot transferability**：探索一个通用 draft model 是否可跨多个目标模型使用
3. **蒸馏或压缩 target context feature**：降低训练和部署时的内存占用
4. **支持更多 generation pattern**：如 streaming、function calling 等复杂场景
5. **探索 diffusion drafter 的架构搜索**：寻找更优的轻量结构

---

## 总结

> **DFlash 成功将 diffusion LLM 的并行性与 speculative decoding 的可靠性结合，提出了一种“轻量扩散 draft + 强大自回归验证”的新范式，不仅大幅提升了推理速度（最高 >6×），还揭示了 diffusion 模型在 LLM 加速中的全新角色——不再是替代者，而是高效的协同者。**

这一工作有望推动 diffusion LLM 的实用化进程，并为下一代高效 LLM inference 框架提供重要参考。

</details>

---

### 5. [Euphonium: Steering Video Flow Matching via Process Reward Gradient Guided Stochastic Dynamics](https://arxiv.org/abs/2602.04928)

**Authors**: Ruizhe Zhong, Jiesong Lian, Xiaoyue Mi, Zixiang Zhou, Yuan Zhou, Qinglin Lu, Junchi Yan  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2602.04928v1  

#### Abstract
While online Reinforcement Learning has emerged as a crucial technique for aligning flow matching models with human preferences, current approaches are hindered by inefficient exploration during training rollouts. Relying on undirected stochasticity and sparse outcome rewards, these methods struggle...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Euphonium: Steering Video Flow Matching via Process Reward Gradient Guided Stochastic Dynamics》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于 **Reinforcement Learning (RL)** 的视频生成后训练方法（如 Flow-GRPO、DanceGRPO）在对齐人类偏好方面存在**探索效率低下**的问题。这些方法依赖于无导向的随机扰动（undirected stochasticity）进行策略探索，并仅在完整视频生成后获得稀疏的结果奖励（outcome rewards），导致：
- 高质量样本难以被发现；
- 训练过程数据利用率低；
- 收敛速度慢。

### 提出了什么新方法或新思路
作者提出 **Euphonium**，一种通过**过程奖励梯度引导的随机动力学**来主动引导生成过程的新框架。其核心思想包括：

#### ✅ **Guided Exploration via Process Reward Gradient**
将采样过程建模为一个理论上有据可依的 **Stochastic Differential Equation (SDE)**，显式地将 **Process Reward Model (PRM)** 的梯度注入到 flow drift 中：
$$
dX_t = \left[u_\theta(X_t,t) - \epsilon_t \nabla U_t(X_t)\right]dt + \sqrt{2\epsilon_t}dW_t
$$
其中 $U_t(x)$ 是结合了 flow prior 和 PRM 的增强势能函数。这实现了**每一步的密集引导**，使模型在潜空间中主动向高奖励区域移动。

#### ✅ **Dual-Reward Optimization**
引入双奖励机制：
- **Latent-space Process Reward**：来自 PRM，在中间时间步提供细粒度反馈，提升信用分配效率；
- **Pixel-space Outcome Reward**：来自 ORM（Outcome Reward Model），确保最终视觉质量和提示一致性。

#### ✅ **Reward-Gradient-Free Inference**
设计了一个**策略蒸馏目标（Policy Distillation）**，将训练阶段的奖励梯度信号内化到 flow network 权重中，从而在推理时无需加载外部 PRM，保持与基础生成器相同的部署方式。

### 相比现有方法的优势
| 维度 | Euphonium | Flow-GRPO / DanceGRPO |
|------|----------|------------------------|
| 探索方式 | **有向引导探索**（reward gradient 显式引导） | 无导向随机探索 |
| 奖励密度 | **密集过程奖励 + 结果奖励** | 仅稀疏结果奖励 |
| 推理依赖 | ❌ 不需要 PRM（蒸馏后） | ✅ 通常不依赖，但无法利用过程信号 |
| 收敛速度 | ⬆️ **快 1.66×** | 基准水平 |
| 对齐效果 | ⬆️ 更优（VBench2 总分最高） | 较弱 |

此外，该框架在理论上统一了现有方法（如 Flow-GRPO、DanceGRPO 可视为 reward-free 特例），提供了更广义的视角。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Reward Model 训练数据**：
  - 包含 200,000 个由 20,000 个唯一 prompt 生成的视频样本；
  - 采用成对标注（pairwise preference annotations），区分正负样本（基于视觉质量与运动连贯性）。
- **GRPO 训练数据**：
  - 使用 10,000 个 prompt（来自 DanceGRPO 和内部人像类数据源）；
  - 严格隔离于 reward model 的训练集，避免过拟合。

### 实验设置和评估指标
- **主干模型**：HunyuanVideo-14B（开源大模型）
- **采样步数（训练）**：16 步 Euler-Maruyama 离散化
- **评估分辨率与帧数**：640×640, 81 frames（高分辨率长序列）
- **评估指标**：**VBench2**（权威视频生成评测套件），包含以下子项：
  - Total Score
  - Creativity
  - Commonsense
  - Controllability
  - Human Fidelity
  - Physics

### 基线方法对比
| 方法 | 类型 |
|------|------|
| Base Model (HunyuanVideo) | 未经过 RL 微调的基础模型 |
| Flow-GRPO (Liu et al., 2025b) | 引入 SDE 进行随机探索的 RL 方法 |
| DanceGRPO (Xue et al., 2025) | 使用共享噪声策略改进信用分配的 RL 方法 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（VBench2 总分）
| 方法 | Total Score |
|------|-------------|
| Base Model | 51.09 |
| Flow-GRPO | 51.52 |
| DanceGRPO | 51.85 |
| **Euphonium (Ours)** | **54.24** ✅ |

> **提升幅度**：相比最强基线 DanceGRPO 提升 **+2.39 分**，绝对领先。

### 各维度表现（部分突出项）
| 维度 | Euphonium | 最佳基线 | 提升 |
|------|-----------|----------|------|
| **Commonsense** | **67.17** | 62.87 (Base) | +4.3 |
| **Controllability** | **26.88** | 25.08 (DanceGRPO) | +1.8 |
| **Human Fidelity** | **88.91** | 88.10 (DanceGRPO) | +0.81 |
| **Physics** | **46.84** | 45.15 (Base) | +1.69 |

> 在 **4/5 子维度**上取得第一，仅在 Creativity 上略低于 Flow-GRPO（41.42 vs 42.42），但仍具竞争力。

### 与基线方法的对比结果
- **收敛速度**：达到相同性能水平所需训练步数减少 **1.66×**（见 Figure 1）；
- **采样效率更高**：得益于过程奖励的密集指导，更快找到高质量轨迹；
- **视觉质量更优**：Figure 2 显示 Euphonium 生成的视频在动作连贯性、细节还原和 prompt adherence 上明显优于基线。

### 消融实验结果（Ablation Study）

#### 🔹 移除主动引导（w/o Active Steering）
| 设置 | VBench2 Total |
|------|---------------|
| 完整 Euphonium | 54.24 |
| w/o Reward Gradient Guidance | 53.61 |
| **下降 Δ** | **-0.63** |

> 表明 reward gradient 引导对性能至关重要。

#### 🔹 移除双奖励组件
| 设置 | VBench2 Total |
|------|---------------|
| w/o PRM Advantage（无过程奖励优势） | 53.95 |
| w/o ORM Advantage（无结果奖励优势） | 53.59 |

> 说明两者均重要，尤其是 ORM 对最终视觉保真度的关键作用。

#### 🔹 Reward-Gradient Guidance 超参数分析
| 指导强度 λ | 总分 |
|------------|------|
| 0.01（太弱） | 53.61 |
| **0.1（适中）** | **54.24** ✅ |
| 1.0（太强） | 52.86 |

> 过强引导会破坏 flow dynamics，导致生成失真。

| 指导时间窗口 | 总分 |
|--------------|------|
| 无指导 | 53.61 |
| 全程指导 (0≤t≤1) | 53.64 |
| **后半段指导 (0.5≤t≤1)** | **54.24** ✅ |
| 后四分之一 (0.75≤t≤1) | 54.14 |

> 后半段指导最优——避开早期结构形成干扰，保留足够优化窗口。

---

## 4. 关键结论和发现

### 主要发现
1. **有向探索显著优于无导向探索**：通过将 PRM 梯度注入 SDE drift，实现 step-level 密集引导，大幅提升探索效率。
2. **双奖励机制协同增效**：
   - Latent PRM 提供高效信用分配；
   - Pixel ORM 锚定最终感知质量。
3. **策略蒸馏是实用部署的关键**：
   - “Inference RGG” 因需同时加载 PRM 导致 OOM（单卡 H20 上失败）；
   - **Distilled 模型无需外部 LRM，推理轻量且性能最佳（54.24）**。
4. **理论统一性**：Euphonium 的 SDE 形式在 reward=0 时退化为 Flow-GRPO/DanceGRPO，证明其为通用框架。

### 方法的局限性
1. **Latent PRM 的泛化能力有限**：
   - 当前 PRM 依赖特定 VAE 的 latent space，难以跨架构迁移；
   - 对不同生成器需重新训练 PRM。
2. **Latent Space Reward 的可靠性假设**：
   - 虽然 PRM 在各噪声级别下准确率 >70%，但仍可能误判复杂语义状态。
3. **计算开销仍存在边际增加**：
   - 尽管 overhead 很小（延迟 +2.4%，显存 +8.5%），但在极致成本敏感场景仍需权衡。

### 未来工作方向
1. **开发通用 Latent Reward Model**：
   - 利用 **Representation Autoencoder (RAE)** 或固定视觉编码器（如 DINOv2）构建跨模型共享 latent space；
   - 实现“即插即用”的 backbone-agnostic PRM。
2. **动态调整指导强度**：
   - 根据生成阶段自适应调节 λ 或激活窗口，进一步优化引导节奏。
3. **扩展至其他生成任务**：
   - 应用于 text-to-audio、3D generation 等需要长期一致性控制的任务。
4. **探索更高效的梯度估计方式**：
   - 如 low-rank approximation 或 implicit differentiation，降低 PRM 训练成本。

---

> 📌 **总结一句话**：  
> **Euphonium 通过引入“过程奖励梯度引导 + 双奖励优化 + 策略蒸馏”的闭环设计，在不增加推理负担的前提下，实现了更高效、更精准的人类偏好对齐，推动了视频生成 RL 微调技术的发展。**

</details>

---

### 6. [AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent](https://arxiv.org/abs/2602.03955)

**Authors**: Yinyi Luo, Yiqiao Jin, Weichen Yu, Mengqi Zhang, Srijan Kumar, Xiaoxiao Li, Weijie Xu, Xin Chen, Jindong Wang  
**Category**: cs.AI  
**Published**: 2026-02-06  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.03955v1  

#### Abstract
While large language model (LLM) multi-agent systems achieve superior reasoning performance through iterative debate, practical deployment is limited by their high computational cost and error propagation. This paper proposes AgentArk, a novel framework to distill multi-agent dynamics into the weigh...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
多智能体系统（Multi-Agent Systems, MAS）通过多个大语言模型（LLM）之间的辩论、批判和共识机制，在复杂推理任务中表现出色。然而，MAS 存在两大瓶颈：
- **高计算成本**：推理时需多次调用多个模型，导致延迟高、资源消耗大。
- **错误传播风险**：个体幻觉或偏见可能在交互中被放大，影响整体鲁棒性。

因此，如何将 MAS 的协同推理能力“内化”到单个模型中，使其具备多智能体的思维模式，同时保持高效推理，成为一个关键挑战。

### **提出的新方法与思路**
本文提出了 **AgentArk**，一个将多智能体推理动态蒸馏到单一 LLM 中的通用框架。其核心思想是：
> 将多智能体在测试时的显式交互过程，转化为单个模型内部隐含的推理能力。

为此，AgentArk 设计了三个层次递进的蒸馏策略：

| 方法 | 描述 |
|------|------|
| **Reasoning-Enhanced SFT (RSFT)** | 在监督微调中引入完整的多智能体推理轨迹作为监督信号，使学生模型学习生成高质量的 CoT 链条。 |
| **Data Augmentation (DA)** | 从多智能体辩论中提取多样化的正确推理路径进行数据增强，提升模型对不同解题策略的泛化能力。 |
| **Process-Aware Distillation (PAD)** | 利用 **Process Reward Model (PRM)** 对每一步推理打分，并结合 **Group Relative Policy Optimization (GRPO)** 进行强化学习，让学生模型学会自我纠错与反思。 |

### **相比现有方法的优势**
- **去耦合设计**：不依赖特定的 MAS 架构或角色设定，适用于任意基于交互的多智能体系统。
- **过程级监督**：超越仅模仿最终答案的传统蒸馏，捕捉“冲突-修正”的辩证推理动态。
- **可扩展性强**：支持跨模型族（cross-family）、跨规模（teacher→student）甚至跨模态（multimodal）的知识迁移。
- **效率优势**：训练开销前置，推理阶段仅为单次前向传播，显著降低部署成本。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 类型 | 说明 |
|--------|------|------|
| **GSM8K** | 数学推理 | 多步算术应用题 |
| **MATH** | 数学推理 | 更难的数学竞赛题 |
| **MetaMathQA (MMQA)** | 增强数学 | 包含多样化解法的数学问答 |
| **MedMCQA** | 医疗领域 | 医学考试选择题，强调专业知识 |
| **HotpotQA / QASPER / QMSum** | 开放域推理 | 用于评估零样本泛化能力（未参与训练） |

### **实验设置**
- **教师模型**：`Qwen3-32B`, `Gemma3-27B-it`, `Qwen3-8B`
- **学生模型**：`Qwen3-8B`, `Qwen3-1.7B`, `Qwen3-0.6B`, `Llama3-8B`, `Gemma-7B`
- **蒸馏方式**：从大模型向小模型、同族/异族之间进行知识转移
- **多智能体配置**：5~20 个 agent 参与辩论，最多 3 轮迭代
- **评估指标**：
  - 主要指标：**Accuracy**
  - 推理质量分析：Perplexity、Step Decomposition、Intermediate Verification、Error Localization、Coherence（由 InternLM-2.5-20b 自动评分）
  - 泛化性：在 OOD 数据集上的表现
  - 鲁棒性：在 TruthfulQA 上的 BLEU/ROUGE/BERTScore

### **基线方法对比**
| 基线 | 说明 |
|------|------|
| **Single Agent** | 原始学生模型，无任何蒸馏 |
| **Vanilla Multi-Agent Debate** | 多智能体直接协作推理（高成本） |
| **Standard SFT** | 仅用标准输入输出对微调 |
| **RSFT / DA / PAD** | 三种蒸馏策略单独及组合使用 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- AgentArk 平均将单个 agent 的性能提升 **4.8%**，接近原始多智能体系统的水平，但推理成本仅为后者的极小部分。
- 在 **GSM8K** 上，`Qwen3-0.6B` 经 PAD 蒸馏后准确率从 41.93 提升至 **44.61**（↑2.68），而多智能体平均为 ~45。
- 在 **MedMCQA** 上，`Qwen3-8B` 从 59.65 提升至 **63.12**（↑3.47），显示对专业领域的有效迁移。

#### **不同方法比较（以 Qwen3-32B → Qwen3-8B 为例）**
| 方法 | GSM8K ↑ | MedMCQA ↑ |
|------|---------|-----------|
| Single Agent | 88.17 | 59.65 |
| RSFT | 89.05 | 60.04 |
| DA | 89.57 | 59.86 |
| **PAD** | **89.02** | **63.12** ✅ |

> 🔍 **PAD 在 MedMCQA 上提升最大**，表明其对复杂逻辑和错误检测更有效。

#### **跨家族蒸馏效果更强**
- 当 teacher 和 student 属于不同模型族（如 `Qwen → Llama` 或 `Gemma → Qwen`）时，增益更大。
- 表明异构架构更能受益于外部推理模式注入。

#### **消融实验结果**
| 发现 | 内容 |
|------|------|
| ✅ **PRM 容量更重要** | 使用更大的 PRM（如 8B）即使训练小模型（0.6B）也能带来显著提升；反之弱 PRM 限制上限。 |
| ⚠️ **学生容量是瓶颈** | 小模型（如 0.6B）无法吸收过多教师多样性，超过 5 个 agent 后性能不再上升甚至下降。 |
| 📈 **PAD 最稳定** | 随着训练数据增加，RSFT 和 DA 出现波动甚至退化，而 PAD 表现稳健，说明**质量优于数量**。 |
| 🔗 **方法兼容性好** | RSFT+DA、PAD+DA 等组合能进一步小幅提升性能（见 Table 7）。 |

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **单模型可以内化多智能体推理能力**  
   通过合理的蒸馏策略，单个 LLM 可以学会类似“内心辩论”的自我反思机制，实现接近多智能体的推理质量。

2. ✅ **过程监督（PAD）优于结果监督**  
   引入 PRM 对中间步骤进行奖励建模，比单纯模仿最终答案或轨迹更有效地传递推理行为。

3. ✅ **推理质量 > 数据数量**  
   单纯堆叠更多推理轨迹不会持续提效，反而可能导致过拟合；高质量、高信号的过程反馈才是关键。

4. ✅ **增强鲁棒性与泛化性**  
   蒸馏后的模型在 TruthfulQA 和 OOD 任务上表现更好，说明其学到的是通用推理能力而非表面模式匹配。

5. ✅ **可扩展至多模态 LLM（MLLM）**  
   初步实验显示，AgentArk 可成功蒸馏至 `Qwen2.5-VL-3B`，尽管增益较小，但仍验证了跨模态潜力。

### **局限性**
- 实验集中在数学和医疗等结构化推理任务，尚未覆盖工具调用、长期记忆等复杂场景。
- 当前框架对 PRM 和 GRPO 的依赖较高，训练成本较大（约 20 小时 on 8×H100）。
- 对超小型模型（如 <1B）提升有限，存在容量天花板。
- 未充分探索除“辩论”外的其他 MAS 范式（如协作、分工）。

### **未来工作方向**
- 探索自适应蒸馏策略：根据任务难度动态选择是否启用 PAD。
- 构建模块化 PRM：针对不同推理环节（分解、验证、纠错）设计专用奖励模型。
- 扩展至真实世界代理任务：如工具使用、环境交互、安全决策支持。
- 研究轻量化版本：降低 PAD 的训练门槛，便于边缘设备部署。
- 探索反向蒸馏：让小模型指导大模型，形成闭环优化。

---

> 💡 **一句话总结**：  
> **AgentArk 成功地将“群体智慧”压缩进“个体大脑”，实现了高效、鲁棒且可泛化的单模型高级推理能力，为未来低成本、高性能的 AI Agent 部署提供了新范式。**

🔗 代码地址：[https://github.com/AIFrontierLab/AgentArk](https://github.com/AIFrontierLab/AgentArk)

</details>

---

### 7. [A$^2$-LLM: An End-to-end Conversational Audio Avatar Large Language Model](https://arxiv.org/abs/2602.04913)

**Authors**: Xiaolin Hu, Hang Yuan, Xinzhu Sang, Binbin Yan, Zhou Yu, Cong Huang, Kai Chen  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.04913v1  

#### Abstract
Developing expressive and responsive conversational digital humans is a cornerstone of next-generation human-computer interaction. While large language models (LLMs) have significantly enhanced dialogue capabilities, most current systems still rely on cascaded architectures that connect independent ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**A²-LLM: An End-to-end Conversational Audio Avatar Large Language Model**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的对话式数字人系统普遍采用**级联架构**（cascaded pipeline），例如：  
`ASR → LLM → TTS → Animation`。这种架构存在以下关键缺陷：
- **高延迟**（high latency）：模块间串行处理导致响应慢。
- **误差累积**（accumulated errors）：每个模块独立训练，错误逐层传播。
- **语义-情感鸿沟**（Semantic-Emotion Gap）：面部动画仅依赖音频信号，缺乏对上下文语义的理解，导致表情僵硬、不自然（如“哈哈”时嘴唇动但脸上无笑意）。

此外，现有方法大多只生成语音或2D视频，难以满足VR/XR等沉浸式场景所需的**几何一致性3D面部动画**。

---

### 🚀 提出的新方法与创新思路

#### （1）**A²-LLM：端到端多模态大模型框架**
- 首次将语言理解、语音生成与**3D facial motion generation**统一在一个LLM中进行联合建模。
- 不再依赖中间文本表示，而是直接从输入音频生成同步的输出音频和3D面部动作参数（FLAME parameters）。
- 引入 **Motion Connector** 模块，通过 cross-attention 将 LLM 的 audio-aligned hidden states 映射为 facial motion tokens。

#### （2）**Residual Motion Tokenization**
- 使用 **RVQ-VAE** 对连续的 FLAME 参数序列进行分层离散化编码，形成 hierarchical motion tokens。
- 使 facial dynamics 可以像语言一样被 autoregressively 生成，实现与音频和文本的统一 token 流处理。

#### （3）**FLAME-QA 数据集**
- 构建首个面向指令微调的高质量多模态问答数据集，格式为 `(Question, Response)` 三元组：
  - `Q_audio`, `Q_text`
  - `R_audio`, `R_text`, `R_visual`（即 FLAME 参数）
- 所有样本均经过语义清洗，并由 LLM 自动生成上下文相关的问题，确保 facial 表情受语义驱动而非仅跟随声学特征。

#### （4）**三阶段课程学习策略（Curriculum Training Strategy）**
1. **Stage 1**: 冻结 LLM，预训练 Motion Connector；
2. **Stage 2**: LoRA Reset —— 重置 LoRA 权重，避免灾难性遗忘；
3. **Stage 3**: 在高动态情感子集上进行情感指令微调，提升表达力。

---

### 🔍 相比现有方法的优势

| 维度 | 传统级联系统 | A²-LLM |
|------|-------------|--------|
| 架构 | Cascaded (模块解耦) | End-to-end (统一建模) |
| 延迟 | >3s（流式）| ~500ms TTFA |
| 情感表达 | 依赖显式标签或后处理 | 由语义深度驱动，无需额外条件 |
| 同步性 | 易出现口型与表情脱节 | 音频-面部动作高度协同 |
| 几何一致性 | 多数为2D像素合成 | 原生支持3D FLAME模型，适用于VR/XR |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### **FLAME-QA**（本文提出）
- 规模：约 **100k** 高质量多模态 QA 样本。
- 来源：基于 **VoxCeleb** 原始视频，使用 SMIRK 提取 FLAME 参数。
- 构造流程：
  1. Whisper 进行 ASR 获得转录文本；
  2. GPT-5.1 清洗文本并生成对应问题；
  3. IndexTTS2 合成问题音频；
  4. 最终得到 `(Q_audio, Q_text, R_audio, R_text, R_visual)` 完整三元组。
- 特色子集：约 **1k 高动态情感样本**，由 InfiniteTalk 生成，包含丰富情绪（笑、惊讶、轻蔑等）。

---

### ⚙️ 实验设置

#### 模型架构
- **Backbone**: Step-Audio-2-mini（基于 Qwen2.5-7B 和 Qwen2-Audio 编码器）
- **Motion Tokenizer**: RVQ-VAE（压缩率 G=5，Nq=6 层量化器）
- **Motion Connector**: 6-layer Transformer decoder，接收降采样后的 LLM hidden states 作为 Query，历史 motion embeddings 作为 KV
- **训练方式**：LoRA 微调（rank=64），配合 Motion Connector 联合优化

#### 推理模式
- 自回归生成 interleaved 文本与音频 tokens；
- Audio-Anchored Motion Generation：在每段音频生成过程中实时预测 facial motion tokens。

---

### 📊 评估指标

#### （1）实时性能
- **TTFT**（Time To First Token）：首token延迟
- **TTFA**（Time To First Action）：首次面部动作延迟
- **RTF**（Real-Time Factor）：生成时间 / 内容时长，越低越好

#### （2）语言能力
使用 **OpenVoiceBench**：
- AlpacaEval（指令遵循）
- TriviaQA、WebQuestions（知识问答）
- Reasoning QA（逻辑推理）

#### （3）面部动画质量
##### 空间维度
- **MOD**（Mouth Opening Distance）：口型垂直开合 MAE（mm），越小越好
- **UFD**（Upper Face Dynamics）：上脸动态强度（参考自由指标），越高越好

##### 时间维度
- **Temporal Correlation**：整体节奏同步性（PCC）
- **Velocity Correlation**：运动方向一致性
- **Lip Width Correlation**：横向拉伸同步性（微笑等）
- **Liveliness Ratio**：动作活力比（接近1.0最佳）
- **Peak Align**：最大开口时间差（ms），越小越好

#### （4）主观评价
- 用户偏好研究（N=60）：两两对比，打分表达力

---

### 🆚 基线方法对比

| 类型 | 方法 |
|------|------|
| 级联系统 | ASR → LLM → TTS → Animation pipeline |
| 音频驱动动画 | ARTalk, CodeTalker, FaceFormer |
| 高保真扩散模型 | DiffPoseTalk（pseudo-oracle）|

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **TTFA** | **535.53 ms**（优化后） |
| **RTF** | **0.703x**（快于实时） |
| **语言性能（AlpacaEval）** | **74.20**（SOTA among audio-native models） |
| **MOD（口型精度）** | **5.08 ± 0.88 mm**（优于多数基线） |
| **UFD（上脸表现力）** | **11.13 ± 1.48**（远超基线） |
| **Temporal Correlation** | **0.464**（+112% vs ARTalk） |
| **Liveliness Ratio** | **1.087**（接近真实动态幅度） |
| **Peak Align** | **114.3 ms**（极佳音画同步） |

---

### 🔁 与基线方法对比结果

#### （1）语言能力（Table 2）
- A²-LLM 在所有 audio-native 模型中表现最优：
  - AlpacaEval: **74.20**（vs 第二名 Qwen2.5-Omni: 72.76）
  - TriviaQA: **79.90**
- 性能接近纯文本模型（Qwen3-8B: 78.19），证明引入 motion token 未损害语言能力。

#### （2）面部动画空间质量（Table 3）

| Model | MOD ↓ | UFD ↑ |
|-------|-------|-------|
| ARTalk | 4.60 | 9.40 |
| CodeTalker | 5.29 | 2.38 |
| FaceFormer | 5.75 | 3.14 |
| **A²-LLM (Ours)** | **5.08** | **11.13** |

👉 **结论**：A²-LLM 在保持良好 lip-sync 的同时，显著提升了上脸情感表达能力。

#### （3）时间动态分析（Table 4）

| Metric | ARTalk | A²-LLM |
|--------|--------|--------|
| Temporal Correlation | 0.218 | **0.464** |
| Velocity Correlation | -0.309 | **0.111**（正相关！） |
| Lip Width Correlation | 0.477 | **0.604** |
| Liveliness Ratio | 0.804 | **1.087** |
| Peak Align (ms) | 116.6 | **114.3** |

👉 **结论**：A²-LLM 动作更自然、节奏一致、能量充沛，无“过平滑”现象。

#### （4）用户偏好研究（Table 5）

| 对比对象 | 赢率（Win %） | 平局 | 输 |
|----------|----------------|------|----|
| vs DiffPoseTalk | **71.7%** | 10.0% | 18.3% |
| vs ARTalk | **75.0%** | 5.0% | 20.0% |

👉 即便面对高保真 diffusion 模型，人类仍认为 A²-LLM 更具表现力。

---

### 🔍 消融实验结果（Table 6）

| 指标 | Adapter-Only（冻结LLM） | Joint Training（本文方法） |
|------|--------------------------|----------------------------|
| Temporal Correlation | 0.028 | **0.464** |
| Lip Width Correlation | 0.057 | **0.604** |
| Peak Align (ms) | 515.05 | **114.30** |

👉 **关键发现**：必须对 LLM 进行微调才能实现精确的相位对齐；否则会出现严重滞后（>500ms），几乎无法同步。

---

## 4. 关键结论和发现

### ✅ 主要结论

1. **端到端建模可有效弥合 Semantic-Emotion Gap**  
   A²-LLM 利用 LLM 的深层语义理解，驱动上下脸协调的表情，而不仅是机械 lip-sync。

2. **motion tokenization 是可行路径**  
   将 facial dynamics 离散化为 tokens 并与 text/audio 统一建模，是实现多模态联合生成的有效范式。

3. **高质量 instruction-following 多模态数据至关重要**  
   FLAME-QA 的 QA 结构迫使模型将 facial 表情与对话意图绑定，而非简单模仿声学信号。

4. **实时性与表现力可以兼得**  
   在仅 **500ms 左右延迟**下，实现了优于现有非实时模型的情感表达能力。

---

### ⚠️ 局限性

1. **语言限制**：目前仅支持英语，尚未扩展至多语言场景。
2. **身体动作缺失**：仅建模面部，未涉及手势、头部姿态或全身动作。
3. **身份固定**：使用固定 identity shape，个性化定制能力有限。
4. **数据依赖性强**：FLAME-QA 依赖外部 TTS 和 LLM 生成问题，可能存在偏差。

---

### 🔮 未来工作方向

1. **多语言支持**：构建跨语言版本的 FLAME-QA-X。
2. **全身体动画扩展**：将 end-to-end 范式推广至 full-body gesture generation。
3. **个性化可控性**：支持用户自定义 avatar identity 与 personality。
4. **交互式反馈闭环**：结合 gaze、点头等非语言行为，增强双向互动体验。

---

> 💡 **一句话总结**：  
> A²-LLM 成功将 **LLM 的语义理解能力**注入 **3D 数字人面部动画**，实现了**低延迟、高表达力、语义一致**的端到端对话式 avatar 生成，为下一代沉浸式 HCI 提供了坚实基础。

</details>

---

### 8. [Stochastic hierarchical data-driven optimization: application to plasma-surface kinetics](https://arxiv.org/abs/2602.04975)

**Authors**: Jos\'e Afonso, Vasco Guerra, Pedro Viegas  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.04975v1  

#### Abstract
This work introduces a stochastic hierarchical optimization framework inspired by Sloppy Model theory for the efficient calibration of physical models. Central to this method is the use of a reduced Hessian approximation, which identifies and targets the stiff parameter subspace using minimal simula...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Stochastic Hierarchical Data-Driven Optimization: Application to Plasma-Surface Kinetics*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对**复杂物理系统建模中的参数校准难题**，尤其是在以下挑战下：
- **高维且病态（ill-conditioned）的优化景观**：模型参数空间维度高，但仅有少数“刚性”（stiff）参数组合主导系统行为，其余“松散”（sloppy）参数对输出影响微弱。
- **计算成本高昂**：动力学模拟（simulator）耗时严重，无法承受大量采样。
- **缺乏梯度信息**：模拟器通常不提供解析或数值梯度，限制了基于梯度的优化方法的应用。
- **数据稀疏性**：实验测量有限，导致反问题（inverse problem）高度不适定。

这些问题在等离子体-表面相互作用（plasma-surface interactions）建模中尤为突出，例如表面反应参数（如吸附系数、能垒）难以通过第一性原理或实验精确获得。

---

### 提出了什么新方法或新思路
作者提出了一种**受“Sloppy Model”理论启发的随机分层优化框架**（Stochastic Hierarchical Optimization Framework），其核心思想是：
- 将优化过程分解为两个子空间：**刚性子空间**（stiff subspace）和**松散子空间**（sloppy subspace）。
- 利用**简化Hessian近似**（reduced Hessian approximation）来识别主导系统行为的低维流形（low-dimensional latent manifold）。
- 采用**分步优化策略**：
  1. **刚性优化**：在由最大曲率方向张成的子空间中快速收敛到低能量谷底；
  2. **松散再对齐与优化**：在正交的松散子空间中进行坐标系旋转并进一步搜索。

该方法的关键技术是**随机低秩Hessian代理**（stochastic reduced Hessian proxy），它通过在一个随机低维子空间 $\Omega \subset \mathbb{R}^n$ 上投影 Gauss-Newton Hessian 来隐式估计主曲率方向，仅需 $k+1$ 次模拟调用（$k \ll n$），而非完整的 $n+1$ 次。

此外，作者构建了一个**基于最大似然估计的 principled 概率目标函数**，以严格处理实验噪声和模型不确定性。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **样本效率**（Sample Efficiency） | 显著优于传统全局优化算法（如DE、CMA-ES）和局部方法（如Powell），在相同模拟次数下更快达到更低损失。 |
| **可扩展性** | 随机Hessian策略将计算开销从 $O(n)$ 降低至 $O(k)$，适用于高维参数空间（即使 $n$ 很大）。 |
| **无需代理模型** | 不依赖高斯过程（GP）等统计代理模型，避免因代理失配导致陷入虚假极小值的风险。 |
| **几何感知能力** | 主动利用损失景观的各向异性结构，在狭窄、细长的山谷中高效导航。 |
| **鲁棒推断** | 结合Hessian分析提供参数不确定性量化，区分“可识别”与“不可识别”参数。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **实验数据来源**：整合自文献 [18–20] 及未发表测量（来自Laboratoire de Physique des Plasmas）。
- **数据规模**：共 **225 组稳态条件**下的实验数据。
- **变量范围**：
  - 气体：O₂/CO₂混合气，总流量 7.4 sccm
  - 压力：0.2 – 10 Torr
  - 放电电流：10 – 40 mA
  - 壁温：-20°C 至 50°C
- **观测量**（Observable）：原子氧的有效复合概率 $y_o$

> 数据按 **80%训练集（N=180） / 20%测试集（N=45）** 分割用于交叉验证。

---

### 实验设置和评估指标

#### 优化目标
最小化基于最大似然推导的目标函数：
$$
\mathcal{L}(\theta) = \frac{1}{2} \sum_{i} \left( \frac{r_i(\theta)}{\sigma_i} \right)^2
$$
其中 $r_i = E_i - M_i(\theta)$ 是残差，$\sigma_i$ 为实验误差。

#### 参数设置
- 优化参数数量：**29个不确定度最高的参数**
  - 能垒（Ea）
  - 斜因子（steric factors, $k_0$）
  - 物理吸附物种脱附频率参数（A, B, E）

#### 基线方法对比
| 方法 | 类型 | 说明 |
|------|------|------|
| **Differential Evolution (DE)** | 全局探索 | 种群型启发式算法，用于评估全局搜索能力 |
| **CMA-ES** | 自适应演化策略 | 能适应病态地形，学习协方差矩阵 |
| **Trust Region Reflective (TRF)** | 局部优化（带边界约束） | Levenberg-Marquardt 变体，有限差分估计Jacobian |
| **Powell’s Method** | 无梯度局部优化 | 迭代线搜索，无需梯度 |
| **Gaussian Process (GP)** | 代理模型优化 | 使用贝叶斯优化进行比较 |

所有算法均从同一初始猜测出发（$\mathcal{L}(\theta^{(0)}) \sim 700$ vs 默认值 $\sim 0.1$），确保公平比较。

#### 评估指标
- **训练损失**（$ \mathcal{L}_{\text{train}} $）随模拟调用次数的变化 → 衡量**样本效率**
- **测试损失**（$ \mathcal{L}_{\text{test}} $）→ 评估泛化能力，防止过拟合
- **收敛速度与最终精度**
- **Hessian特征谱分析** → 验证模型的“sloppy”性质

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **Hierarchical 方法（Exact & Stochastic）** 在前几十次迭代中表现出**最快下降速度**。
- **Stochastic Reduced Hessian ($k=18$)** 在极少模拟调用下即可逼近 Exact 方法性能。
- 最终测试损失与其他优秀方法（如TRF）相当，表明**良好泛化性**。
- 五次独立交叉验证中，测试集 $R^2 = 0.736$，$\mathcal{L}_{\text{test}}$ 范围为 **0.054 – 0.087**，显示结果稳定。

---

### 与基线方法的对比结果
| 方法 | 样本效率 | 最终精度 | 备注 |
|------|----------|----------|------|
| **Hierarchical (Exact)** | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | 快速收敛，但每次迭代成本较高 |
| **Hierarchical (Stochastic, k=18)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐☆ | **最优平衡点**，极高效地捕捉主曲率 |
| **TRF** | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐☆ | 强劲对手，最终可达相似精度，但前期较慢 |
| **CMA-ES** | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ | 对病态地形有一定适应，但仍慢于分层法 |
| **DE** | ⭐☆☆☆☆ | ⭐⭐☆☆☆ | 因各向同性搜索难以穿越细长山谷 |
| **Powell** | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ | 易陷局部极小，效率低 |
| **GP** | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ | 代理模型构建本身代价高，且存在失配风险 |

> 图2(a) 显示：Hierarchical 方法在约 **50次模拟调用内** 达到其他方法需数百次才能达到的损失水平。

---

### 消融实验结果（Ablation Study）
虽然文中未明确标注“ablation”，但以下实验证实了关键设计的有效性：
- **不同 $k$ 值的随机子空间比较**（图2b）：
  - 即使 $k=3$ 或 $k=5$（远小于 $n=29$），也能实现快速初期下降。
  - 证明：只需少量方向即可捕获主导几何结构，支持方法的**可扩展性假设**。
- **Hessian特征谱分析**（图2d）：
  - 特征值呈指数衰减，证实模型具有典型的 **sloppy structure**。
  - 刚性模式（前几个大特征值）与松散模式之间存在明显能隙（spectral gap），为分层优化提供了理论基础。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **物理模型普遍存在 sloppy structure**：尽管参数众多，但系统行为由少数刚性组合决定，其余参数高度不确定。
2. **几何引导的优化显著提升效率**：通过显式识别并优先优化刚性子空间，可在极少数模拟调用下逼近最优解。
3. **随机低秩Hessian是高效的几何探测器**：作为“线性自编码器”，它能以 $O(k)$ 成本有效提取本地曲率主轴，适合昂贵模拟器。
4. **参数不确定性可通过Hessian定量刻画**：
   - 刚性参数（如脱附频率A/B/E、CO化学吸附斜因子$k_{0.32}, k_{0.37}, k_{0.39}$、亚稳态能量$E_c, E_{\min}$）被紧密约束。
   - 松散参数（如部分亚稳态反应速率）则具有宽泛置信区间，这是模型结构性缺陷而非数据不足所致。

> 如图4所示，这些“uncolored bars”对应 sloppy parameters，其不确定性本质上不可消除。

---

### 方法的局限性
- **局部方法本质**：不能保证找到全局最优，但在 sloppy models 中，近优解集合 $S_\epsilon$ 通常是连通的大区域，因此局部收敛已足够。
- **依赖残差较小的前提**：Gauss-Newton Hessian 的有效性建立在残差接近零的基础上，若初始猜测太差可能失效。
- **需要手动设定阈值**：如 stiff subspace variance threshold $\gamma=0.9$ 和 reduced sloppy threshold $\tau=10^{-4}$，虽有经验依据，但仍属超参调节。
- **实现复杂度较高**：相比标准优化器，需自行实现分层逻辑与Hessian估计。

---

### 未来工作方向
1. **深入物理解释**：利用数据驱动结果探究Pyrex表面上原子氧复合的微观机制。
2. **外推能力研究**：当前验证集中于插值任务，未来将测试模型在训练域之外的操作条件下是否仍具预测力。
3. **框架通用化推广**：将此方法应用于其他复杂反应网络，如生化系统、燃烧化学、催化过程等。
4. **结合主动学习**：动态选择最具信息量的实验条件进行模拟，进一步减少总查询数。

---

> ✅ **总结一句话**：  
> 本文提出的 **stochastic hierarchical optimization** 框架通过融合 **Sloppy Model 理论** 与 **reduced Hessian 技术**，实现了在**极高计算成本与数据稀缺双重限制下**对复杂物理模型的高效、稳健参数校准，为 plasma-surface kinetics 等领域的建模提供了强有力的工具，并具备广泛的可迁移潜力。

</details>

---

### 9. [Learning, Solving and Optimizing PDEs with TensorGalerkin: an efficient high-performance Galerkin assembly algorithm](https://arxiv.org/abs/2602.05052)

**Authors**: Shizheng Wen, Mingyuan Chi, Tianwei Yu, Ben Moseley, Mike Yan Michelis, Pu Ren, Hao Sun, Siddhartha Mishra  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2602.05052v1  

#### Abstract
We present a unified algorithmic framework for the numerical solution, constrained optimization, and physics-informed learning of PDEs with a variational structure. Our framework is based on a Galerkin discretization of the underlying variational forms, and its high efficiency stems from a novel hig...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Learning, Solving and Optimizing PDEs with TensorGalerkin

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于 **Galerkin** 或 **有限元法 (FEM)** 的 PDE 数值求解在现代自动微分（AD）框架（如 PyTorch）中面临严重的效率瓶颈，尤其是在 GPU 上运行时。主要问题包括：
- **Python 循环开销大**：传统 FEM 组装过程依赖对网格元素（element）的逐个循环，在 Python 层面执行时引入显著解释器开销。
- **计算图碎片化**：局部基函数索引的循环导致自动微分系统生成高度碎片化的计算图，严重拖慢反向传播速度。
- **AD 开销高**：物理信息神经网络（如 PINNs）依赖 `torch.autograd` 计算空间导数，带来大量计算图嵌套和内存消耗。

这些问题限制了其在 **多查询任务**（如 PDE-constrained optimization、operator learning）中的应用。

---

### 提出的新方法：TENSORGALERKIN
作者提出 **TENSORGALERKIN** —— 一种高效的、高性能的 Galerkin 组装算法，其核心思想是将传统的“循环-散列加”（scatter-add）组装过程重构为 **纯张量化的 Map-Reduce 范式**。

#### 创新架构
- **Stage I: Batch-Map（全张量化物理计算）**
  - 将所有单元的局部刚度矩阵 $ K_{\text{local}} $ 和载荷向量 $ F_{\text{local}} $ 的计算统一为一个 **密集张量收缩（dense tensor contraction）** 操作。
  - 使用 `torch.einsum` 实现，融合了 quadrature 点、基函数索引和元素维度，避免任何显式循环。
  - 输出为形状为 `(E, k, k)` 的局部张量，其中 E 是元素数量，k 是每个元素的自由度。

- **Stage II: Sparse-Reduce（拓扑感知的稀疏规约）**
  - 使用预计算的 **Routing Matrices**（$ S_{\text{mat}}, S_{\text{vec}} $）将局部张量聚合为全局稀疏矩阵 $ K $ 和向量 $ F $。
  - 全局组装通过一次 **稀疏矩阵乘法 (SpMM)** 完成：
    $$
    F = S_{\text{vec}} \cdot \text{vec}(F_{\text{local}}), \quad K = \text{CSR}(L, S_{\text{mat}} \cdot \text{vec}(K_{\text{local}}))
    $$
  - 完全消除原子操作（atomic operations），实现确定性、高效且可微的组装。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **效率** | 显著减少 Python 开销和 AD 图碎片，提升 GPU 利用率，实现 **1–2 个数量级的速度提升**。 |
| **可微性** | 整个流程端到端可微（end-to-end differentiable），天然支持梯度反向传播，无需手动推导伴随方程。 |
| **灵活性** | 支持动态网格（dynamic mesh），无 JIT 重编译延迟（相比 JAX-FEM）。 |
| **精度** | 使用解析形状函数梯度（analytical shape gradients），避免 AD 引入的数值误差。 |

---

## 2. 核心实验方法和设置

### 下游应用场景与任务
TENSORGALERKIN 被部署于三个下游任务，形成完整工具链：
1. **TENSORMESH**：高效数值 PDE 求解器
2. **TENSORPILS**：物理信息驱动的算子学习（Physics-Informed Learning System）
3. **TENSOROPT**：PDE 约束优化与逆设计

---

### 数据集与问题设置
| 任务 | PDE 类型 | 几何域 | 网格类型 | 输入分布 |
|------|--------|--------|---------|----------|
| **Numerical Solver** | 3D Poisson, 3D Linear Elasticity | 单位立方体、空心立方体 | 四面体网格 (tetrahedral) | 单一源项 |
| **Neural PDE Solver** | 2D Poisson (checkerboard forcing) | 单位正方形 | 非结构三角网格 | 多尺度不连续源项 $ f_K(x,y) $ |
| **Operator Learning** | Wave Equation (hyperbolic), Allen-Cahn (parabolic) | 圆形域、L 形域 | 非结构三角网格 | 随机初值（multi-frequency sine expansion） |
| **Inverse Design** | 2D Linear Elasticity (SIMP) | 悬臂梁矩形域 | 结构化四边形单元 (QUAD4) | 密度场优化 |

---

### 评估指标
| 任务 | 主要指标 |
|------|--------|
| 数值求解器 | 运行时间 (runtime)、相对残差 (RelRes)、相对误差 (RelErr) |
| 神经 PDE 求解器 | 相对 L2 误差 (%)、训练吞吐量 (it/s) |
| 算子学习 | 相对 L2 误差（ID / OOD 测试集） |
| 逆设计 | 总耗时、收敛步数、合规性（compliance）下降率 |

---

### 基线方法对比
| 任务 | 对比基线 |
|------|--------|
| 数值求解 | FEniCS (CPU), scikit-fem (SKFEM, CPU), JAX-FEM (CPU/GPU), PINN |
| 神经求解器 | PINN, VPINN, Deep Ritz |
| 算子学习 | Data-Driven GNN, PI-DeepONet |
| 逆设计 | JAX-FEM + LU solver |

---

## 3. 主要实验结果和性能指标

### ✅ 数值求解器性能（TENSORMESH）
- **3D Poisson 方程**：
  - GPU 版本比 FEniCS 快 **10 倍以上**。
  - 在百万级 DoF 下仍保持稳定加速。
- **3D 弹性力学**：
  - 比 CPU 版 TENSORMESH 快近 **100 倍**。
- **残差分析**：
  - TENSORMESH 达到最小线性系统残差，精度优于或等于其他 FEM 工具。
- **批处理生成**：
  - 批量生成 7k DoF Poisson 解时，CUDA 版本在 batch size=100 时几乎零额外开销，远超 CPU 基线。

> 🔺 图表支持：Fig 2, Fig B.1, Fig B.4

---

### ✅ 神经 PDE 求解器性能（TENSORPILS）
| Method | Rel. L2 Error (K=8) | Speed (Adam it/s) |
|--------|---------------------|------------------|
| PINN | 34.77% | 20.1 |
| VPINN | 154.10% | 54.9 |
| Deep Ritz | 10.60% | 58.7 |
| **TENSORPILS (Ours)** | **10.05%** | **117.8** |

- **误差降低 50%+** 于最近基线（Deep Ritz），同时速度快 **2 倍以上**。
- **前向损失计算扩展性**：
  - PINN 损失随 DoF 增长呈指数上升（AD 图开销）。
  - TENSORPILS 几乎恒定开销，接近有限差分（FDM）水平。

> 🔺 表格支持：Table 1, Fig 3

---

### ✅ 物理信息算子学习（TENSORPILS）
| Model | Wave (ID) | Wave (OOD) | AC (ID) | AC (OOD) |
|-------|-----------|------------|--------|---------|
| Data-Driven | 0.089±0.013 | 0.230±0.017 | 0.135±0.042 | 0.152±0.080 |
| PI-DeepONet | 0.626±0.033 | 0.863±0.018 | 0.743±0.163 | 8.536±6.306 |
| **TENSORPILS** | **0.085±0.010** | **0.090±0.006** | **0.110±0.014** | **0.083±0.013** |

- **OOD 泛化能力极强**：TENSORPILS 在外推任务中误差仅轻微上升，而数据驱动模型误差翻倍，PI-DeepONet 完全失效。
- **无需标签数据**：TENSORPILS 为 data-free 方法，却优于使用 16 个样本训练的数据驱动模型。

> 🔺 表格支持：Table 2, Fig B.13–B.15

---

### ✅ PDE 约束逆设计（TENSOROPT）
| Stage | JAX-FEM | TENSOROPT (Ours) | Speedup |
|-------|--------|------------------|---------|
| Setup Time | 2.62 s | 0.58 s | **4.5×** |
| Optimization Loop | 28.51 s | 7.77 s | **3.7×** |
| **Total Time** | **31.13 s** | **8.35 s** | **3.7×** |

- 最终设计拓扑一致，合规性差异 < 0.33%，验证准确性。
- 加速源于：无循环组装 + 高效可微求解器 + 避免 JIT 编译。

> 🔺 表格支持：Table 3, Fig B.17–B.18

---

### ✅ 消融实验（Ablation Study）
- **数据效率分析**（Fig B.16）：
  - TENSORPILS 在仅 **1 个训练样本** 下即可达到 ~10% 误差。
  - 数据驱动方法需要更多样本才能收敛，且泛化差。
- **证明 Galerkin 损失本身具有强归纳偏置**，适合小数据场景。

---

## 4. 关键结论和发现

### 主要发现
1. **Galerkin 组装是性能瓶颈的关键**：传统 scatter-add 模式在现代 AD 框架中不可持续。
2. **TENSORGALERKIN 实现统一高效框架**：
   - 一套引擎支持 **求解、学习、优化** 三大任务。
   - 通过 **Map-Reduce + SpMM** 实现极致并行与可微性。
3. **解析梯度优于自动微分**：
   - 使用 shape function gradients 替代 `autograd.grad()` 极大提升效率与精度。
4. **物理信息先验至关重要**：
   - 在低数据和 OOD 场景下，TENSORPILS 显著优于纯数据驱动方法。
5. **端到端可微性简化优化流程**：
   - 无需手动实现伴随变量法，梯度自动传播。

---

### 局限性
- **假设 PDE 具有变分结构**（variational structure）：仅适用于能写成双线性形式 $ a_p(u,v) = l_p(v) $ 的 PDE。
- 不直接支持非协调方法（如 DG）、非线性复杂耦合系统。
- 当前实现集中在 2D/3D 标量/矢量椭圆、抛物、双曲方程，更复杂系统需进一步扩展。

---

### 未来工作方向
1. 扩展至 **非协调有限元方法**（Discontinuous Galerkin, Petrov-Galerkin）。
2. 支持更复杂的 **时间步进策略**（如自适应时间步、隐式 RK）。
3. 探索 **三维复杂几何下的大规模系统** 应用。
4. 应用于真实世界场景：如流体控制、材料设计、气候建模等 PDE-constrained control 问题。

---

> 📌 **项目主页**：[https://camlab-ethz.github.io/TensorGalerkin](https://camlab-ethz.github.io/TensorGalerkin)

</details>

---

### 10. [E-Globe: Scalable $\epsilon$-Global Verification of Neural Networks via Tight Upper Bounds and Pattern-Aware Branching](https://arxiv.org/abs/2602.05068)

**Authors**: Wenting Li, Saif R. Kazi, Russell Bent, Duo Zhou, Huan Zhang  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2602.05068v1  

#### Abstract
Neural networks achieve strong empirical performance, but robustness concerns still hinder deployment in safety-critical applications. Formal verification provides robustness guarantees, but current methods face a scalability-completeness trade-off. We propose a hybrid verifier in a branch-and-bound...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# E-Globe: Scalable $\epsilon$-Global Verification of Neural Networks via Tight Upper Bounds and Pattern-Aware Branching  
**论文核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
神经网络在安全关键领域（如电力系统、自动驾驶）的应用受限于其**鲁棒性验证的可扩展性与完备性之间的权衡**。现有方法面临以下挑战：
- **MIP-based 完全验证器**：虽然能获得全局最优解 $f^*$，但计算复杂度随网络规模指数增长，难以扩展到大模型。
- **松弛-based 不完全验证器**（如 CROWN）：速度快，但仅提供下界 $l \leq f^*$，无法量化优化间隙（optimality gap），导致大量“未知”状态。
- **对抗攻击方法**（如 PGD）：启发式搜索上界 $u \geq f^*$，但易陷入局部最优，上界松散且不可靠。

本文旨在通过**高效地同时紧缩上下界**，实现接近全局最优的 $\epsilon$-global 验证，在保证精度的同时大幅提升效率。

---

### 🚀 提出的新方法与创新点

E-Globe 是一个基于 **Branch-and-Bound (BaB)** 框架的混合验证器，核心创新如下：

#### (i) **NLP-CC 上界求解器（Tight Upper Bounding）**
- 将每个 ReLU 激活函数用 **Complementarity Constraints (CC)** 精确建模，构建一个非线性规划问题（NLP-CC）。
- 该 reformulation 是**精确等价**的：任何可行解都对应原始网络的一个有效激活模式，输出值即为合法上界 $u = f(x)$。
- 利用 KKT 条件解释 ReLU 切换行为，确保输入-输出图不变（invariant feasible region）。
- 在满足严格互补性（strict complementarity）时，上界是紧致的（tight）。

#### (ii) **Warm-started NLP with Low-Rank KKT Updates**
- 在 BaB 过程中复用父节点的 NLP 解作为 warm-start。
- 分支仅改变少量神经元相位，因此只需对 KKT 系统进行**低秩修正**（rank ≤ 4），显著加速后续 NLP 求解。
- 实践中带来 **2–5× 的速度提升**。

#### (iii) **Pattern-Aligned Strong Branching**
- 利用 NLP-CC 返回的当前最优激活模式 $a_{\text{NLP}}$ 作为“导航信号”。
- 改进传统的 Filtered Smart Branching (FSB)，引入正则项使其优先选择与 $a_{\text{NLP}}$ 对齐的分裂方向：
  $$
  s_a(C_i) = s(C_i) + \lambda \cdot m(a(C_i), a_{\text{NLP}})
  $$
  其中 $m$ 表示不稳定神经元相位匹配的比例。
- 显著减少无效分支，更快提升下界。

#### (iv) **$\epsilon$-Global Verification Framework**
- 同时维护上下界 $[l, u]$，当 $u - l \leq \epsilon$ 时停止，返回 $\epsilon$-optimal certificate。
- 若 $u < 0$：立即返回 **Unsafe** 并给出反例；若 $l > 0$：返回 **Safe**。
- 实现早期终止，避免穷举所有子问题。

---

### 🔍 相比现有方法的优势

| 维度 | E-Globe | MIP | PGD | 松弛方法（如 CROWN） |
|------|--------|-----|-----|------------------|
| 上界质量 | ✅ 紧致、可靠 | ✅ 最优 | ❌ 松散、可能失败 | ❌ 无上界 |
| 下界质量 | ✅ 可认证（via B-CROWN） | ✅ 最优 | ❌ 无 | ✅ 可认证 |
| 效率 | ⭐⭐⭐ 高（尤其大规模） | ⭐ 极慢（指数级） | ⭐⭐ 快但不完整 | ⭐⭐⭐ 快但gap未知 |
| 可扩展性 | ✅ 良好（多项式趋势） | ❌ 差 | ✅ 良好 | ✅ 良好 |
| 输出完整性 | ✅ $\epsilon$-gap 或明确结论 | ✅ 完备 | ❌ 不完备 | ❌ “未知”多 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **MNIST**：输入维度 784，用于小扰动（$\delta \leq 0.01$）和大扰动（$\delta \leq 0.1$）测试。
- **CIFAR-10**：输入维度 3072，更具挑战性，测试更大扰动场景（$\delta = 0.01, 0.03$）。

### 🏗️ 网络架构
- **MNIST**：全连接网络 `NoSoftmaxNet`，两层隐藏层（50 units），无 softmax 层。
- **CIFAR-10**：全连接 ReLU MLP，两层（256 units）。

### ⚙️ 实验设置
- 扰动集：$\ell_\infty$ 球 $C = \{x : \|x - x_0\|_\infty \leq \delta\}$。
- 验证目标：最小分类 margin $f(x) = z_k - z_a$（$k$: 正确类，$a$: 攻击类）。
- 使用 **B-CROWN** 作为下界传播方法。
- NLP-CC 使用 **IPOPT** 求解，MIP 使用 **Gurobi**。
- 实验平台：
  - Mac M4（CPU/GPU）用于上界实验；
  - AMD 32核服务器 + 64 GPUs 用于完整 BaB 实验。

### 📈 评估指标
| 指标 | 定义 |
|------|------|
| $\Delta_\delta = |u - f^*|$ 或 $|l - f^*|$ | 绝对误差（以 MIP 或高保真求解器为 ground truth） |
| $\Delta_\delta^{\text{rel}} = \Delta_\delta / |f^*|$ | 相对误差 |
| 上界成功率 $\phi(\%)$ | 成功找到有效上界的案例占比 |
| Runtime | 单个样本平均运行时间（秒） |
| Speedup | 相对于 MIP 的加速比 |

### 🆚 基线方法对比
- **下界方法**：CROWN-IBP, CROWN, $\alpha$-CROWN
- **上界方法**：PGD
- **完全验证器**：MIP
- **组合验证器**：$\alpha$-B-CROWN（用于完整 BaB 对比）

---

## 3. 主要实验结果和性能指标

### 📉 上界紧致性（Tables 1–4）
| 方法 | MNIST ($\delta=0.1$) $\Delta_{0.1}$ | MNIST ($\delta=0.01$) $\Delta_{0.01}$ | CIFAR-10 ($\delta=0.03$) $\Delta_{0.03}$ |
|------|-------------------------------|----------------------------------|------------------------------------|
| CROWN-IBP | 68.01 | 2.57 | — |
| CROWN | 42.73 | 0.4719 | — |
| $\alpha$-CROWN | 35.53 | 0.4673 | — |
| **E-Globe$_u$** | **0.43** | **0.0004** | **0.003** |
| PGD | — | — | 0.204 |

> ✅ E-Globe$_u$ 的上界远优于所有基线，即使在大扰动下仍保持极小误差（< 0.05），而 PGD 上界松散且成功率低（仅 42% @ CIFAR-10）。

### ⏱️ 效率对比（Figure 6）
- 当 binary variables 数量 > 120 时，**MIP 运行时间呈指数增长**，多数超时（>2500s）。
- **E-Globe$_u$** 运行时间几乎不受变量数影响，维持在 0–30 秒内。
- 在 binary vars > 180 场景下，**E-Globe 比 MIP 快 2–3 个数量级**。

### 🔥 Warm-start 加速效果（Figure 7）
- 使用 low-rank KKT warm-start 后，每轮 NLP 求解时间下降 **2–5×**。
- 特别是在前几轮分支中，warm-start 显著缩短收敛时间。

### 🌱 Pattern-Aligned Branching 消融实验（Figure 9）
- 引入 pattern alignment（$\lambda > 0$）后，**下界上升速度明显加快**。
- $\lambda = 0.1$ 时表现最佳，在约 500 轮后稳定领先标准 FSB 方法。
- 说明 NLP 提供的激活模式是高质量引导信号。

### 📈 完整 E-Globe 性能（Figure 8）
- 在难例（case 42）上，MIP 耗时 >2000s 才收敛。
- E-Globe 在不同 $\epsilon$ 下均实现大幅加速：
  - $\epsilon = 0.1$：约 **20× speedup**
  - $\epsilon = 0.5$：可达 **>100× speedup**
- gap 随 branch round 快速缩小，主要得益于下界快速上升（B-CROWN + pattern-aware branching）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **NLP-CC 是一种高效且准确的上界生成机制**：它保留了 ReLU 的精确结构，产生的每个可行解都是有效的 counterexample，并能提供紧致上界。
2. **上下界协同可极大提升验证效率**：通过 tight upper bound 实现快速 reject unsafe 情况，结合 pattern-guided branching 快速 tighten lower bound，避免 exhaustive search。
3. **warm-start 和低秩更新显著降低 NLP 开销**：使得在 BaB 中频繁调用 NLP 成为可行。
4. **pattern-aligned branching 是关键设计**：利用 NLP 提供的局部最优模式指导分支，显著提高分支效率。
5. **E-Globe 在实践中接近 complete verifier 的覆盖率**，但在运行时间上实现数量级提升。

---

### ⚠️ 方法的局限性
1. **理论上属于 incomplete verifier**：尽管实践中绝大多数情况都能解决，但仍存在无法在时限内达到 $\epsilon$-gap 的极端案例。
2. **依赖高质量初始 bounds**：需先用 a-CROWN 获取中间层 bounds，否则 NLP-CC 可能难以收敛。
3. **GPU batching 未完全并行化**：目前 B-CROWN 支持 GPU batch，但 NLP-CC 求解仍是串行，仍有优化空间。
4. **对非常深或复杂结构（如 Transformer）支持有限**：当前实验集中在 FC 和简单 CNN 结构。

---

### 🔮 未来工作方向
1. **进一步融合 local solver 与 convex relaxation**：探索更深层次的协同机制，例如将 NLP 解用于构造更强的 convex surrogate。
2. **扩展至其他激活函数和网络结构**：如 SiLU、GeLU、ResNet、Vision Transformer 等。
3. **开发分布式/并行化版本**：利用多 GPU/CPU 并行处理多个 subdomains。
4. **应用于 real-time verification 场景**：如自动驾驶中的在线鲁棒性监控。
5. **理论分析 NLP-CC 的 landscape properties**：为何其局部解常接近全局最优？是否存在隐式的泛化结构？

--- 

> 💡 **总结一句话**：  
> **E-Globe 通过 NLP-CC 精确上界 + pattern-aware branching + warm-start 机制，在保持高精度的同时实现了比 MIP 验证器快 2–3 个数量级的速度，推动神经网络验证迈向实用化 $\epsilon$-global 最优时代。**

</details>

---

### 11. [Agent-Omit: Training Efficient LLM Agents for Adaptive Thought and Observation Omission via Agentic Reinforcement Learning](https://arxiv.org/abs/2602.04284)

**Authors**: Yansong Ning, Jun Fang, Naiqiang Tan, Hao Liu  
**Category**: cs.AI  
**Published**: 2026-02-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.04284v1  

#### Abstract
Managing agent thought and observation during multi-turn agent-environment interactions is an emerging strategy to improve agent efficiency. However, existing studies treat the entire interaction trajectories equally, overlooking the thought necessity and observation utility varies across turns. To ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Agent-Omit: Training Efficient LLM Agents for Adaptive Thought and Observation Omission via Agentic Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **LLM Agent** 在多轮与环境交互过程中，普遍存在生成冗余 **Thought**（推理过程）和累积过多历史 **Observation**（环境反馈）的问题。这导致上下文长度迅速增长，显著降低推理效率（token cost 高），限制了其在实际场景中的应用。

现有研究通常对整个交互轨迹进行统一压缩或剪枝（如固定长度截断、LLM summarization），忽略了 **不同交互轮次中 Thought 和 Observation 的必要性是动态变化的**。例如：
- 初始规划阶段的 Thought 至关重要；
- 中间执行阶段可能无需复杂推理；
- 早期 Observations 在最终答案生成时往往已无用。

因此，如何实现**自适应地、选择性地省略冗余内容**成为提升效率的关键。

### 提出的新方法与思路
本文提出 **Agent-Omit**，一个统一的训练框架，使 LLM Agent 能够通过 **Agentic Reinforcement Learning** 学习到自适应省略（adaptive omission）的能力。其核心创新包括：

1. **统一的分析框架**  
   首次从“轮次级别”（turn-level）定量分析 Thought 与 Observation 对 **Effectiveness**（任务准确率）和 **Efficiency**（token 开销）的影响，验证了“非均匀必要性”的假设。

2. **两阶段训练范式**
   - **Agent Omission Behavior Synthesis（冷启动微调）**  
     构建单轮与多轮省略场景的合成数据，用于 SFT（Supervised Fine-Tuning），教会模型如何执行 `<think></think>`（空思考）和 `<omit_tool_response_N_...>`（省略指定轮次观察）等格式化行为。
   - **Omit-Aware Agentic RL（省略感知强化学习）**
     引入双采样机制（Dual Sampling）和定制化的省略奖励（Omission Reward），让策略能在保留原始上下文的同时学习省略决策，避免因“上下文改变”而导致无法学习。

3. **理论保障**
   证明所学省略策略的性能偏差由 KL 散度上界控制，为方法稳定性提供理论支持。

### 相比现有方法的优势
| 类别 | 典型方法 | 局限 | Agent-Omit 优势 |
|------|--------|-------|----------------|
| **Thought Management (TM)** | DEPO, ToolLight, Thinking-Retention | 固定压缩策略，缺乏灵活性；易丢失关键信息 | 自适应判断是否需要推理 |
| **Observation Management (OM)** | Observation-Mask, DeepMiner | 启发式规则（如滑窗），不能泛化 | 动态识别可省略的历史观测 |
| **TOM（联合管理）** | MEM-Agent, ReSum | 依赖外部 LLM summarizer，引入额外开销且与主推理脱节 | 内生式压缩，端到端优化 |

> ✅ **核心优势**：不是简单删减，而是让 Agent “学会何时可以安全跳过”，实现了更灵活、高效、可扩展的上下文管理机制。

---

## 2. 核心实验方法和设置

### 使用的数据集
在五个多样化基准上进行评估，覆盖多种任务类型：

| 数据集 | 任务类型 | 最大回合数 | 测试样本数 |
|-------|---------|------------|-----------|
| **DeepSearch** | 知识密集型搜索问答 | 8 | 400 |
| **WebShop** | 电商网站导航与购买 | 12 | 200 |
| **TextCraft** | 文本版 Minecraft 长程规划 | 20 | 100 |
| **BabyAI** | 网格世界指令跟随 | 10 | 90 |
| **SciWorld** | 科学实验模拟与推理 | 10 | 200 |

这些任务均来自 **AgentGym-RL** 统一评测平台，确保公平比较。

### 实验设置与评估指标
- **Backbone 模型**：Qwen3-4B / Qwen3-8B
- **训练流程**：
  1. **SFT 冷启动**：使用约 2–4K 合成省略数据，全参数微调。
  2. **Agentic RL 微调**：基于 GRPO 算法，结合 dual sampling 与 omission reward。
- **评估指标**：
  - **Pass@1**：任务成功率（主要衡量 effectiveness）
  - **Avg Tok. ↓**：平均每轮输出 token 数量（衡量 efficiency）
  - **Effectiveness-Efficiency Trade-off**：综合考量准确率与成本

### 基线方法对比
分为两类：

#### （1）前沿 LLM Agents（Frontier Models）
- DeepSeek-R1-0528, DeepSeek-V3.2
- OpenAI o3 / o4-mini
- Qwen3-235B-A22B, Qwen3-Next-80B-A3B, Qwen3-32B

> 目标：验证 Agent-Omit 是否能以小模型媲美甚至超越大模型性能。

#### （2）高效 Agent 构建方法（Efficient Agent Methods）
| 类别 | 方法 |
|------|------|
| **TM** | Thinking-Retention, DEPO, Tool-Light |
| **OM** | Observation-Mask, DeepMiner |
| **TOM** | MEM-Agent, ReSum |
| **Ours** | Agent-Omit-8B-RL |

> 目标：验证在相同 backbone 下，Agent-Omit 的效率增益是否最优。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 Qwen3-8B 为基础）

#### ✅ 与前沿 LLM Agents 对比（Table 2）
| Model | DeepSearch (Pass@1) | DeepSearch (Tok↓) | WebShop (Pass@1) | WebShop (Tok↓) |
|-------|------------------------|--------------------|------------------|---------------|
| DeepSeek-R1-0528 | 25.25 | 6,412 | 19.37 | 11,308 |
| Qwen3-32B | 19.00 | 6,640 | 11.31 | 11,872 |
| **Agent-Omit-8B-RL** | **26.56** | **4,356** | **23.57** | **8,764** |

> 📌 在多个任务上达到 SOTA 准确率，同时 token 消耗显著低于大多数 reasoning-mode 模型。

#### ✅ 与高效 Agent 方法对比（Table 3）
| Method | DeepSearch (Pass@1↑) | DeepSearch (Tok↓) | WebShop (Pass@1↑) | WebShop (Tok↓) |
|--------|------------------------|--------------------|------------------|---------------|
| Base (Qwen3-8B) | 17.75 | 8,281 | 6.93 | 16,741 |
| ReSum | 22.28 | 5,724 | 17.80 | 9,251 |
| **Agent-Omit-8B-RL** | **24.56** | **4,356** | **23.57** | **8,764** |

> ✅ **唯一同时实现最高准确率与最低 token 成本的方法**，展现出最佳的 **effectiveness-efficiency trade-off**。

### 消融实验结果（Ablation Study）

在 WebShop 上对 Agent-Omit-8B 进行消融（Figure 5）：

| 变体 | Pass@1 | Avg Tok |
|------|--------|--------|
| Full Agent-Omit (SFT + RL) | **23.57** | **8,764** |
| w/o STO (无单轮省略数据) | 21.2 | ~8,900 |
| w/o MTO (无多轮省略数据) | 20.8 | ~9,100 |
| w/o PT (无 Partial Trajectory) | 20.1 | ~9,300 |
| w/o OR (无 Omission Reward) | 20.5 | ~10,200 |

> 🔍 发现：
- **SFT 阶段**：单轮省略数据（STO）最为关键，奠定基础能力。
- **RL 阶段**：Partial Trajectory 采样比 Full 更重要；Omission Reward 是驱动 token 下降的核心动力。
- **双阶段协同增益明显**，缺一不可。

---

## 4. 关键结论和发现

### 主要发现
1. **Thought 与 Observation 的必要性随轮次动态变化**  
   并非所有轮次都需要详细推理或完整历史上下文。中间轮次是省略的主要窗口期。

2. **自适应省略可行且有效**  
   通过适当的训练机制，LLM Agent 可学会在不影响性能的前提下主动省略冗余内容。

3. **Agent-Omit 显著提升小模型竞争力**  
   Agent-Omit-8B-RL 在多个任务上超越更大规模的 frontier models，尤其在 token 效率方面优势巨大。

4. **省略行为分布符合预期**  
   分析显示 Agent 平均每条轨迹省略 **3–4 轮**，且集中在 **第 3–10 轮**（中间执行阶段），与人类直觉一致（Figure 6）。

### 方法的局限性
- **依赖高质量合成数据构建冷启动集**：若初始 omission rollouts 不准确，可能导致错误模式固化。
- **当前仅适用于文本型交互环境**：对于视觉或多模态 Agent 尚未验证。
- **Omission Reward 设计敏感**：需 careful tuning 权重 $ \rho $，否则可能陷入 reward hacking。

### 未来工作方向
1. **将 omission data synthesis 扩展至预训练阶段**，探索大规模 omission-aware pretraining。
2. **应用于更大规模的 LLMs**（如 Qwen3-72B 或 DeepSeek-V3），进一步释放潜力。
3. **扩展至多模态 Agent**，研究视觉/语音 context 的自适应省略机制。
4. **构建通用的 omission policy adapter**，实现跨任务迁移。

---

> 💡 **一句话总结**：  
> **Agent-Omit 提出了一种“会偷懒”的智能体训练方式——它不盲目推理也不死记历史，而是在恰当的时候选择“跳过”，从而在保持高准确率的同时大幅降低成本，为高效 LLM Agent 设计提供了新范式。**

GitHub 代码与数据已开源：[https://github.com/usail-hkust/Agent-Omit](https://github.com/usail-hkust/Agent-Omit)

</details>

---

### 12. [FedMosaic: Federated Retrieval-Augmented Generation via Parametric Adapters](https://arxiv.org/abs/2602.05235)

**Authors**: Zhilin Liang, Yuxiang Wang, Zimu Zhou, Hainan Zhang, Boyi Liu, Yongxin Tong  
**Category**: cs.CL  
**Published**: 2026-02-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.05235v1  

#### Abstract
Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding generation in external knowledge to improve factuality and reduce hallucinations. Yet most deployments assume a centralized corpus, which is infeasible in privacy aware domains where knowledge remains siloed. Thi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FedMosaic: Federated Retrieval-Augmented Generation via Parametric Adapters

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **Retrieval-Augmented Generation (RAG)** 依赖于将检索到的原始文本插入 LLM 的上下文中进行生成，这在隐私敏感领域（如医疗、金融）面临严重挑战，因为这些领域的数据通常分散在多个机构（即“数据孤岛”）中，受法规（如 HIPAA、GDPR）限制，**无法共享原始文档**。

现有的联邦 RAG（FedRAG）方法大多基于 in-context RAG，仍需传输原始文本，违反了**本地性约束**（locality constraint）。而直接应用 parametric RAG 到联邦场景又面临两大挑战：
- **存储与通信开销高**：为每个文档训练独立的 LoRA adapter 导致数量爆炸。
- **破坏性聚合**（destructive aggregation）：简单平均来自不同孤岛的 adapters 会引入噪声和参数冲突，降低准确性。

### 提出的新方法与创新思路
本文提出 **FedMosaic** —— 首个满足 locality constraint 的联邦 RAG 框架，基于 **parametric adapters** 构建，通过以下两个核心技术解决上述问题：

#### ✅ 创新点一：多文档参数化适配器（Multi-Document Parametric Adapters）
- 将语义相似的文档聚类，并为每个聚类训练一个共享的 LoRA adapter。
- 引入**文档级二值掩码**（document-specific binary masks），使每个文档仅激活 adapter 中特定子集的参数，从而保留细粒度知识并缓解**组内干扰**（intra-silo interference）。
- 采用位打包（bit-packing）进一步压缩掩码存储。

> 📌 *优势*：显著减少 adapter 数量，降低存储和通信成本，同时保持 per-document specificity。

#### ✅ 创新点二：选择性适配器聚合（Selective Adapter Aggregation）
- 在推理阶段，各 silo 上传检索文档的相关性分数及其掩码（不传 adapter 或原文）。
- 服务器基于相关性和**掩码重叠度**（overlap）选择最相关且参数冲突最小的一组文档。
- 请求对应的 adapters 后，在掩码控制下加权聚合，生成最终答案。

> 📌 *优势*：避免无关或冲突 adapters 的负面影响，提升准确率，实现“相关性感知 + 冲突感知”的聚合。

### 相比现有方法的优势
| 维度 | FedMosaic | 现有方法（如 in-context FedRAG, PRAG） |
|------|-----------|----------------------------|
| **隐私保护** | ✔️ 不传输任何原始文档 | ❌ in-context 方法必须传原文 |
| **通信效率** | ↓ 降低 91.4% | 高昂（尤其 per-document adapter） |
| **存储开销** | ↓ 降低 78.8%–86.3% | 随文档数线性增长 |
| **准确性** | ↑ 平均提升 10.9% F1 | 易受噪声和冲突影响 |
| **灵活性** | 支持动态组合知识 | 联邦微调需重新训练 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在四个主流问答数据集上进行，涵盖多种推理类型：
- **HotpotQA**（HQA）：多跳推理（multi-hop），含 Bridge 和 Compare 类型
- **2WikiMultihopQA**（2WQA）：复杂推理任务，分 Bridge / Compare / Inf / Compose 四类
- **PopQA**（PQA）：常识问答
- **ComplexWebQuestions**（CWQ）：复杂 Web 查询

此外还进行了隐私攻击评估使用的数据集：
- **Enron Emails**
- **WikiText**

### 实验设置
- **模型架构**：
  - 主干 LLM：`LLaMA3.2-1B-Instruct` 和 `LLaMA3-8B-Instruct`
  - 适配器技术：LoRA（Low-Rank Adaptation）
- **联邦设置**：
  - 数据按主题使用 Dirichlet 分配（α=0.1）划分为多个 silo
  - 每个 silo 拥有本地文档库 $ \mathcal{D}_m $
- **离线阶段**：
  - 文档聚类 → 训练 cluster-level adapter → 学习 document-specific mask
- **在线阶段**：
  - 查询广播 → 本地检索与重排序 → 上报 relevance score + mask → 服务端选择 → 获取 adapter → 掩码聚合 → 生成答案

### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (F1 Score)** | 回答正确性的主要衡量标准 |
| **Privacy Protection Rate** | 对抗 target/prefix 数据提取攻击的能力 |
| **Communication Efficiency** | 每次查询从 silo 发送到 server 的参数量 |
| **Storage Overhead** | silo 侧额外存储的 adapter 和 mask 大小 |

### 基线方法对比
共四类 baseline 进行公平比较：
1. **Local RAG**  
   - Standard RAG, CoTRAG, ReAct, Dargin
2. **In-context FedRAG**  
   - FRAG, MKPQA, RAGRoute
3. **Federated Fine-Tuning (FedFT)**  
   - FedIT, FLoRA
4. **Parametric RAG**  
   - PRAG

> ⚠️ 注意：隐私保护 prompt 工程方法（如 DP-Prompt, Sage）因性能严重下降未作为主 baseline。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 方法 | Avg. F1 提升 |
|------|-------------|
| **FedMosaic (Ours)** | **0.3841**（最高） |
| 最强 Baseline (FLoRA) | ~0.3497 |
| ➜ **平均高出 10.9% F1** | ✅ |

具体表现亮点：
- 在 **2WQA-Bridge** 上达到 **0.4453**，比第二名高约 13%
- 在 **2WQA-Compose** 上达 **0.0940**，远超 PRAG 的 0.0462
- 在 **CWQ** 上达 **0.3841**，优于所有基线

### 与基线方法的对比结果
| 对比维度 | 结果 |
|--------|------|
| **vs In-context FedRAG** | 准确性更高 + 完全满足 locality constraint |
| **vs Federated Fine-Tuning** | 无需频繁 retraining，支持按需激活知识 |
| **vs Parametric RAG (PRAG)** | 更低开销 + 更高鲁棒性（避免破坏性聚合） |

> 🔍 特别指出：PRAG 在更大模型（LLaMA3-8B）上性能退化明显，而 FedMosaic 表现稳定。

### 消融实验结果（Ablation Study）

#### （1）聚类对开销的影响（Fig. 4）
- 当每 cluster 包含最多 10 个文档时：
  - **存储成本降至无聚类版本的 11.23%**
  - **单次查询通信成本降至 4.86%**
- 掩码本身仅占 ~1% 存储空间，可忽略

#### （2）掩码的有效性（Fig. 5）
- 加入 document-specific mask 后：
  - next-token loss 下降更快
  - 模型准确率持续领先“无 mask”变体
  - 验证了 LoRA 参数具有稀疏可分离性假设

#### （3）选择性聚合的效果（Table 5）
- 随着选择的 top-k 增大，“无选择”版本性能下降（因噪声增加）
- FedMosaic 在 k=5 后趋于稳定，**Inf 类任务提升达 20.7%**

#### （4）top-k 检索的影响（Fig. 6）
- FedMosaic 在不同 top-k 设置下表现更**稳定且一致领先**
- 在 HQA-Compare 上平均优于最强 baseline **10.17%**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Parametric RAG 是实现隐私安全 FedRAG 的可行路径**，但需专门设计以应对联邦环境下的效率与精度挑战。
2. ✅ **文档聚类 + 掩码机制** 能有效平衡存储效率与知识特异性。
3. ✅ **选择性聚合策略** 显著优于盲目平均，是提升准确率的关键。
4. ✅ FedMosaic 在多个数据集和模型规模下均表现出色，具备良好**可扩展性**。
5. ✅ 实验证明其对数据提取攻击具有更强抵抗力，**隐私性优于 in-context 方法**。

### 方法的局限性
- **聚类质量依赖嵌入表示**：若初始文档向量不能很好反映语义，则可能导致错误分组。
- **掩码学习引入额外训练开销**：虽然只训练 mask，但仍需额外计算资源。
- **NP-hard 的选择问题**：全局最优选择不可行，当前使用贪心算法近似求解。
- **假设同构模型**：要求所有 silo 使用相同 base LLM 和 re-ranker，可能限制实际部署灵活性。

### 未来工作方向
- 设计更高效的掩码学习机制（如联合优化 adapter 与 mask）
- 探索异构联邦 RAG 场景下的自适应对齐方法
- 扩展至 streaming document 更新场景，支持增量式 adapter 更新
- 结合 compressed adapter 技术进一步压缩通信负载
- 探索在真实医疗/金融系统中的落地应用与合规审计支持

---

> 💡 **总结一句话**：  
> **FedMosaic 是首个真正满足 locality constraint 的高效、准确、隐私安全的联邦 RAG 框架，通过 multi-document adapters + selective aggregation 实现了知识集成的“马赛克式拼接”，为分布式知识系统的构建提供了新范式。**

</details>

---

### 13. [RRAttention: Dynamic Block Sparse Attention via Per-Head Round-Robin Shifts for Long-Context Inference](https://arxiv.org/abs/2602.05853)

**Authors**: Siran Liu, Guoxia Wang, Sa Wang, Jinle Zeng, HaoYang Xie, Siyu Lou, JiaBin Yang, DianHai Yu, Haifeng Wang, Chao Yang  
**Category**: cs.CL  
**Published**: 2026-02-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.05853v1  

#### Abstract
The quadratic complexity of attention mechanisms poses a critical bottleneck for large language models processing long contexts. While dynamic sparse attention methods offer input-adaptive efficiency, they face fundamental trade-offs: requiring preprocessing, lacking global evaluation, violating que...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# RRAttention: Dynamic Block Sparse Attention via Per-Head Round-Robin Shifts for Long-Context Inference —— 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLMs）在处理长上下文时面临 **attention 机制的二次复杂度 $O(L^2)$** 问题，导致推理成本高昂，难以部署于超长序列场景（如 128K tokens）。虽然已有动态稀疏 attention 方法试图缓解该问题，但普遍存在以下权衡（trade-offs）：

- 需要离线预训练或模式搜索（preprocessing），限制部署灵活性；
- 缺乏全局评估能力，无法捕捉长距离依赖；
- 违反 query independence，导致注意力分布被污染；
- 不同 attention head 间策略不一致，增加实现复杂性；
- Softmax 粒度粗，影响精度。

### **提出了什么新方法或新思路**

本文提出 **RRAttention**，一种全新的动态块稀疏 attention 方法，其核心是 **Per-Head Round-Robin（头轮询）采样策略**。

#### 主要设计思想：
- 在每个 stride 内对不同 attention head 轮流选择不同的 query 位置进行重要性估计。
- 公式化为：  
  $$
  P(i, h) = iS + (S - 1 - (h \mod S))
  $$
  其中 $S$ 是 stride 大小，$h$ 是 head index。
- 所有 head 合作完成一个 stride 内所有位置的覆盖，避免信息遗漏。

#### 三阶段流程：
1. **Query Sampling with Head Round-Robin Strategy**  
   每个 head 在其对应位置采样 query，保持 query independence。
2. **Stride-level Importance Estimation**  
   对 key 进行 stride 级聚合，计算跨 stride 的重要性得分，将复杂度从 $O(L^2)$ 降至 $O(L^2/S^2)$。
3. **Block-level Selection via Top-T Thresholding**  
   将 stride 得分聚合到 block 级别，并保留累计重要性超过阈值 $T$ 的 blocks；同时保护最后一个 query block 以维持生成质量。

### **相比现有方法的优势**

| 维度 | RRAttention | 其他方法（如 XAttention、FlexPrefill） |
|------|-------------|----------------------------------------|
| **无需预处理（Preprocessing-free）** | ✅ | ❌（部分需离线训练/模式分配） |
| **支持全局评估（Global Evaluation）** | ✅ | ❌（如 FlexPrefill 只用最后 query） |
| **保持 query 独立性（Query Independence）** | ✅ | ❌（如 XAttention 跨 query 聚合） |
| **模式无关（Pattern-agnostic）** | ✅ | ❌（如 MInference/FlexPrefill 区分垂直/斜线模式） |
| **高效 softmax 粒度（Stride-level）** | ✅ | ⚠️（token-level 更慢） |

> ✅ RRAttention 是目前唯一同时满足这五个理想属性的方法。

---

## 2. 核心实验方法和设置

### **使用的数据集**

| 类型 | 数据集 | 描述 |
|------|-------|------|
| **自然语言理解** | [HELMET](https://arxiv.org/abs/2502.11089) | 包含 7 大类任务：<br>- 合成回忆（Recall）<br>- 检索增强生成（RAG）<br>- 多样本上下文学习（ICL）<br>- 引用生成（Cite）<br>- 文档重排序（Rerank）<br>- 长文档问答（LongQA）<br>- 摘要（Summarization） |
| **多模态视频理解** | [Video-MME](https://arxiv.org/abs/2405.21075) | 包含 900 个视频、2700 条多选题，涵盖感知、推理、信息整合等 12 种任务，测试模型对长时间视频的理解能力。 |

### **实验设置和评估指标**

- **模型**：
  - `Meta-LLaMA-3.1-8B-Instruct`（支持 128K）
  - `Qwen2.5-7B-Instruct`（基于 YARN 扩展至 128K）
  - `Qwen2-VL-7B-Instruct`（用于 Video-MME）
  - 补充实验还用了 `Yi-9B-200K` 和 `Qwen3-30B-A3B`

- **上下文长度**：8K → 128K tokens

- **评估指标**：
  - 平均准确率（Avg. Score）
  - 稀疏度（Sparsity）：跳过的 attention block 比例
  - 推理速度（FPS / Time）
  - 模式搜索开销（Pattern Search Overhead）

- **稀疏配置**：
  - 保守设置：$T=0.95$, $\gamma=0.99$
  - 激进设置：$T=0.90$, $\gamma=0.95$

- **硬件平台**：NVIDIA H100 GPUs

### **基线方法对比**

| 方法 | 特点 |
|------|------|
| **FlashAttention** | 密集 attention 基线，衡量原始性能上限 |
| **FlexPrefill** | 使用最后一个 query 发现 vertical/slash 模式，依赖 JS 散度判断可靠性 |
| **XAttention** | 使用 anti-diagonal 采样 + stride 聚合，速度快但违反 query independence |
| **RRAttention (Ours)** | 本文提出方法，head-round-robin + stride-level aggregation |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 📊 在 HELMET 上的整体表现（128K context）

| 方法 | 模型 | Avg. Score | Sparsity | 相对于 Full Attention 的恢复率 |
|------|------|------------|----------|-------------------------------|
| FullAttention | Llama | 49.74 | 0% | 100% |
| FlexPrefill ($\gamma=0.99$) | Llama | 49.87 | 50.54% | ~100.3% |
| XAttention ($T=0.95$) | Llama | 49.45 | 66.22% | 99.4% |
| **RRAttention ($T=0.95$)** | **Llama** | **50.37** | **66.02%** | **>100% (达 101.3%)** |

> 🔥 **RRAttention 在 Llama 上恢复了超过 100% 的 Full Attention 性能，同时仅计算约一半的 attention blocks！**

#### 🚀 推理效率提升

- 在 128K context 下，**RRAttention 实现 2.4× 端到端加速**。
- **模式搜索时间减少 18.2%** 相比 XAttention（见 Figure 3b），得益于更高效的 head-round-robin 采样与 stride-level aggregation。

#### 🎯 多模态任务（Video-MME）表现

| 设置 | 方法 | Avg. Score (Long Videos) | Sparsity |
|------|------|-------------------------|----------|
| 1fps | FullAttention | 55.20 | 0% |
| 1fps | XAttention ($T=0.95$) | 56.10 | 37.50% |
| 1fps | **RRAttention ($T=0.95$)** | **56.20** | **34.70%** |

> ✅ RRAttention 在视频理解中也取得最佳性能，尤其在中长视频上优势明显，说明其全局评估能力对时空建模至关重要。

---

### **与基线方法的对比结果**

| 对比维度 | RRAttention vs Baselines |
|---------|---------------------------|
| **准确性** | 在所有 context 长度下均优于 FlexPrefill 和 XAttention，平均高出 0.5–1.5 分 |
| **稀疏性-精度权衡** | 在相同稀疏度下精度更高，或在相同精度下实现更高稀疏度 |
| **泛化性** | 在 Llama、Qwen、Yi、Qwen3 四大架构上均表现最优，验证通用性 |
| **细粒度任务表现** | 在 Recall、LongQA、Rerank 等需要全局理解的任务上显著领先 |

> 💡 例如在 Qwen-128K 上，RRAttention 达到 38.51 分，而 FlexPrefill 仅为 35.52 分（相差 +3.0 分），且稀疏度更高（60.97% vs 48.20%）。

---

### **消融实验结果**

#### ✅ 最后 query block 保护机制（Last Q Block Protection）

- 应用于 XAttention 后，性能从 55.74 提升至 55.92（+0.18），但 RRAttention 本身已达 56.24。
- 结论：**pattern discovery 比 protection 更关键**。

#### ✅ 不同 RR 策略比较（Head-RR vs Layer-RR vs Hybrid-RR）

| 方法 | Avg. Score |
|------|------------|
| w/o RR | 55.65 |
| **Head-RR** | **55.80** ✅ |
| Layer-RR | 55.54 |
| Hybrid-RR | 55.61 |

> ✅ **Head-level RR 效果最好**，因其确保每个 stride 内的位置都能被充分采样。

#### ✅ Stride 大小影响（S=4,8,16,32）

- 当 $S \leq 16$ 时性能稳定；
- $S=32$ 时因聚合过粗导致性能下降。
- 推荐使用 $S=8$ 或 $S=16$，兼顾效率与精度。

#### ✅ Block Selection 准确性分析（Appendix D）

| 方法 | Average Precision ↑ | Recall | F1 Score ↑ |
|------|---------------------|--------|------------|
| XAttention | 12.48% | 93.35% | 26.81 |
| **RRAttention** | **13.05%** (+0.57%) | 93.05% | **27.58** (+0.77) |

> ✅ RRAttention 具有更高的 **precision** 和 **F1**，表明其 block 选择更精准，误报更少。

---

## 4. 关键结论和发现

### **主要发现**

1. **RRAttention 是首个同时满足五大理想属性的动态稀疏 attention 方法**：
   - Preprocessing-free
   - Global Evaluation
   - Query Independence
   - Pattern-agnostic
   - Stride-level Softmax

2. **通过 head-round-robin 采样实现了“完全位置覆盖”与“query 独立性”的统一**，解决了传统方法的信息丢失与干扰问题。

3. **在多种任务和模型上均恢复 >99% 甚至超过 100% 的 Full Attention 性能**，证明其不仅能逼近原性能，还能起到正则化作用，过滤噪声。

4. **在 128K context 下实现 2.4× 加速，且模式搜索开销更低**，适合实际部署。

5. **在多模态视频理解中表现优异**，说明其对复杂时空依赖建模能力强。

---

### **方法的局限性**

- **极端 stride 配置下的边界问题**：当 stride 大小 $S$ 超过 attention head 数量时，无法保证每个位置都被采样，可能导致重要信息遗漏。
- **当前仅应用于 prefill 阶段**，未扩展至 decoding 阶段，仍有优化空间。
- **仍有一定运行时开销用于 pattern discovery**，虽已很低，但不如静态方法零成本。

> ⚠️ 但作者指出：这些极限情况在实践中很少出现，推荐的 $S=8$ 或 $16$ 完全可避免此问题。

---

### **未来工作方向**

1. **工程优化**：
   - 升级至 FlashAttention-3，利用 warp specialization 和更好内存调度进一步提速。

2. **训练感知稀疏（Training-aware Sparse Attention）**：
   - 在训练阶段引入稀疏监督，让模型学会预测稀疏模式，消除推理时 pattern search 开销。

3. **扩展至 decoding 阶段**：
   - 将 RR 思想用于 KV Cache 压缩，降低每 token 的延迟和显存占用。

4. **结合其他加速技术**：
   - 与 PagedAttention、KV Cache Quantization 等正交技术联合使用，构建全栈长上下文推理引擎。

---

> ✅ **总结一句话**：  
> **RRAttention 通过巧妙的 head-round-robin 采样策略，在不牺牲任何理论性质的前提下，实现了当前最先进的动态稀疏 attention 性能与效率平衡，是迈向实用化超长上下文推理的重要一步。**

</details>

---

### 14. [DSB: Dynamic Sliding Block Scheduling for Diffusion LLMs](https://arxiv.org/abs/2602.05992)

**Authors**: Lizhuo Luo, Shenggui Li, Yonggang Wen, Tianwei Zhang  
**Category**: cs.CL  
**Published**: 2026-02-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.05992v1  

#### Abstract
Diffusion large language models (dLLMs) have emerged as a promising alternative for text generation, distinguished by their native support for parallel decoding. In practice, block inference is crucial for avoiding order misalignment in global bidirectional decoding and improving output quality. How...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DSB: Dynamic Sliding Block Scheduling for Diffusion LLMs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前在 **diffusion large language models (dLLMs)** 中广泛使用的 **naive block-diffusion** 推理策略存在以下关键缺陷：

- **固定块调度（fixed, predefined block schedule）** 忽略了语义难度和上下文动态变化。
- 强制在低置信度位置提前解码（premature commitment），导致错误传播。
- 高置信度但位于块边界外的位置被延迟解码，降低并行效率。

这种“一刀切”的块划分方式造成了 **generation quality** 和 **inference efficiency** 之间的次优权衡。

---

### 🚀 提出的新方法与新思路

作者提出 **Dynamic Sliding Block (DSB)** ——一种无需训练的动态块调度机制，其核心思想是：

- 维护一个**滑动且大小可变的活动块（active block）**，而非固定的静态块。
- 在每一步迭代中：
  - 根据当前已解码状态动态调整块的起始位置（左边界）。
  - 动态扩展右边界以保持至少 $ S_{\text{init}} $ 个未解码 token，上限为 $ S_{\text{max}} $。
- 实现更灵活的 semi-autoregressive 解码：既保留因果性，又提升局部并行性。

此外，针对 DSB 引入的 **KV-cache 不稳定性问题**，提出了专用缓存机制：

> **DSB Cache**：引入一个位于活动块前的 **prefix window**，该窗口与活动块一起在每步刷新 KV 状态，并周期性执行全局刷新，从而稳定缓存、避免频繁失效。

---

### 🔍 相比现有方法的优势

| 方面 | Naive Block | DSB |
|------|------------|-----|
| 调度灵活性 | ❌ 固定块大小与顺序 | ✅ 动态滑动 + 自适应尺寸 |
| 语义感知能力 | ❌ 完全忽略置信度 | ✅ 延迟低置信解码，优先高置信输出 |
| 并行效率 | ⚠️ 边界处浪费并行机会 | ✅ 更早释放易解 token |
| KV 缓存兼容性 | ✅ 支持 Dual/Prefx Cache | ✅ 专为滑动设计，避免状态震荡 |
| 是否需要训练 | ✅ 是（如 WeDLM） | ✅ 否（training-free） |

> ✅ **DSB 是首个完全 training-free 的动态滑动块调度方案**，显著优于固定块策略，同时避免了复杂训练开销。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

涵盖多种任务类型，共 **5 个基准测试集**：

| 数据集 | 类型 | 示例任务 |
|--------|------|----------|
| **GSM8K** (5-shot) | 数学推理 | 解数学文字题 |
| **MATH** (4-shot) | 复杂数学 | 高中竞赛级题目 |
| **HumanEval** (0-shot) | 代码生成 | Python 函数补全 |
| **MBPP** (3-shot) | 编程任务 | 小规模编程问题 |
| **BBH** (3-shot) | 综合推理 | Big-Bench Hard 子集 |

---

### ⚙️ 实验设置与评估指标

#### 模型
- **LLaDA-8B-Instruct**, **LLaDA-1.5**
- **Dream-v0-Base-7B**, **Dream-v0-Instruct-7B**

#### 硬件
- 单张 **NVIDIA H200 140G GPU**

#### 参数配置
- 生成长度：256
- 初始块大小 $ S_{\text{init}} $：32
- 最大块大小 $ S_{\text{max}} $：32（DSB const.）或无限制（DSB greedy）
- 最小 prefix window 长度 $ l_{\text{pmin}} $：24（LLaDA），4（Dream）
- 并行解码置信阈值：0.9

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Accuracy (%)** | 衡量生成质量 |
| **TPS (Tokens Per Second)** | 衡量推理吞吐量，反映效率 |

---

### 🆚 基线方法对比

从三个维度进行比较：

| 维度 | 基线方法 |
|------|---------|
| **Decoding Strategy** | - Vanilla Top-1 Sampling<br>- Confidence-aware Parallel Decoding (Fast-dLLM) |
| **Block Scheduling** | - Naive Block Scheduling（固定块） |
| **KV Caching** | - Dual Cache（缓存非活动块） |

> 所有对比均在同一框架下实现，确保公平性。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

#### 在 **LLaDA-8B-Instruct + GSM8K** 上的表现：

| 方法 | Accuracy ↑ | TPS ↑ |
|------|-----------|-------|
| Vanilla (no cache) | 77.79 | 14.94 |
| Naive Block + Dual Cache | 77.40 | 92.26 |
| **DSB (const.) + DSB Cache** | **80.14** | **98.10** |
| **DSB (greedy) + DSB Cache** | **80.29** | **99.61** |

✅ **准确率提升约 2.9%，吞吐提升 >7%**

#### 在 **Dream-v0-Instruct-7B + GSM8K** 上：

| 方法 | Accuracy | TPS |
|------|---------|-----|
| Naive Block + Dual Cache | 67.32 | 72.18 |
| **DSB (greedy) + DSB Cache** | **73.08** | **75.27** |

✅ **准确率大幅提升近 6%，同时维持更高吞吐**

---

### 🔁 与基线方法的整体对比结论

- 在几乎所有模型和 benchmark 上，**DSB + DSB Cache** 均实现了：
  - **更高的 Accuracy**
  - **更高的 TPS**
- 特别是在结合 KV Cache 后，优势更加明显。
- 即使在 Dream 系列上因 AR 初始化表现波动，仍能在多个场景下取得增益。

> 💡 **DSB 实现了 generation quality 与 inference speed 的双重提升，突破传统 quality-speed trade-off。**

---

### 🔍 消融实验结果（Ablation Studies）

#### （1）DSB Cache 中 prefix window 的作用（Table 2）

| 方法 | GSM8K Acc / TPS |
|------|------------------|
| DSB (const.) + Dual Cache（无 prefix window） | 76.42 / 78.93 |
| **DSB (const.) + DSB Cache（含 prefix window）** | **80.14 / 98.10** |

➡️ 移除 prefix window 导致：
- **Accuracy ↓ 3.7 pts**
- **TPS ↓ 19.17**

> ✅ 证明 prefix window 对稳定 KV-cache 至关重要。

#### （2）不同 $ S_{\text{init}} $ 的影响（Figure 4）

- DSB 对初始块长度鲁棒性强。
- 当 $ S_{\text{init}} = 64 $ 时，naive block 性能下降明显，而 DSB 仍保持稳定。

#### （3）不同生成长度 $ L $ 的表现（Figure 5）

- 随着 $ L $ 增加，DSB 依然保持对 vanilla sampler 的质量和速度优势。
- 显示其在长序列生成中的潜力。

#### （4）$ S_{\text{max}} $ 与 $ l_{\text{pmin}} $ 敏感性分析（Figures 6 & 7）

- $ S_{\text{max}} $ 过大会削弱因果约束，略微牺牲 accuracy 换取 TPS。
- $ l_{\text{pmin}} $ 存在最优值（如 24 for LLaDA），过大反而降低效率。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **固定块调度是瓶颈**：naive block 忽视语义难度，造成质量与效率双损。
2. **DSB 显著改善 semi-autoregressive 推理**：
   - 动态滑动块能自适应上下文演化。
   - 延迟不确定 token，提前释放高置信 token。
3. **DSB Cache 解决滑动带来的 KV 不稳定问题**：
   - prefix window + 周期刷新机制有效维持缓存一致性。
4. **training-free 设计更具实用性**：
   - 无需额外训练，即插即用，适用于各类 dLLM 架构。

> 🎯 **DSB 将 dLLM 推理推向新的 quality-speed frontier。**

---

### ⚠️ 方法的局限性

1. **依赖置信度估计**：性能受限于模型自身 confidence calibration 能力。
2. **极端长文本尚未验证**：目前实验集中在 ~256 长度，超长文本效果待探索。
3. **suffix window 尝试失败**（Appendix A）：
   - 尝试添加后缀窗口未能带来一致收益，说明前向上下文更重要。
4. **对某些架构增益有限**：如 Dream 系列因 AR 初始化导致部分场景增益不显著。

---

### 🔮 未来工作方向

1. **将 DSB 思想融入预训练或后训练阶段**：
   - 训练时模拟动态块掩码，进一步对齐训练与推理。
2. **结合 early stopping 或 adaptive termination**：
   - 动态决定何时停止 denoising，进一步提速。
3. **探索更智能的 block size 控制策略**：
   - 基于语义单元（句子、短语）自动划分块大小。
4. **扩展到多模态 diffusion 模型**：
   - 如图像-文本联合生成中应用动态块调度。

---

> 🔗 **开源地址**：[https://github.com/lizhuo-luo/DSB](https://github.com/lizhuo-luo/DSB)  
> 📄 **论文版本**：Preprint, February 6, 2026

</details>

---

### 15. [TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](https://arxiv.org/abs/2602.04929)

**Authors**: Junhan Kim, Yeo Jeong Park, Seungwoo Son, Chungman Lee, Ho-young Kim, Joonyoung Kim, Yongkweon Jeon  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.04929v1  

#### Abstract
The rapid growth of large language models (LLMs) has heightened the importance of post-training quantization (PTQ) for reducing memory and computation costs. Among PTQ methods, GPTQ has gained significant attention for its efficiency, enabling billion-scale LLMs to be quantized within a few GPU hour...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLMs）在部署时面临高内存占用和计算成本的挑战，**Post-Training Quantization (PTQ)** 是一种有效的解决方案。然而，现有主流方法存在以下瓶颈：

- **GPTQ** 虽然高效，但假设层间独立，忽略了注意力模块中的跨层依赖关系，在低比特（如 INT2）量化下精度严重下降。
- **BoA** 通过引入注意力感知的 Hessian 近似，建模了跨层依赖，显著提升了精度，但其必须对输出通道（out-channels）进行**逐个串行量化**，导致效率远低于 GPTQ。

因此，本文旨在解决 **“如何在保持甚至提升 BoA 高精度的同时，大幅加速其量化过程”** 的核心矛盾。

---

### **提出了什么新方法或新思路**

作者提出 **TURBOBOA**，一种无需反向传播的 PTQ 算法，通过三项关键技术实现效率与精度的双重突破：

#### (i) **多通道联合量化（Joint Quantization of Multiple Out-Channels）**
- **核心思想**：不再逐个量化 out-channels，而是**同时量化 N 个通道**，将串行操作转换为并行处理，从根本上减少迭代次数。
- **关键技术**：提出一个**闭式误差补偿规则**（closed-form error compensation rule），在联合量化后，显式地将这些通道间的依赖关系纳入误差补偿中，确保精度不因并行化而损失。
- **效果**：相比 BoA 的完全串行，该策略实现了超过 **3倍的速度提升**。

#### (ii) **前序量化层误差补偿（Error Compensation for Pre-Quantized Layers）**
- **问题**：BoA 忽略了来自先前已量化层的误差传播，这些误差会扰动当前层的输入分布，导致误差累积。
- **解决方案**：在误差补偿目标函数中，显式加入由输入偏差 $ \Delta X $ 引起的额外失真项 $ GW\Delta X $，使量化模型能更忠实地复现全精度（FP）模型的行为。
- **技术区别**：不同于 GPTAQ 假设 $ H_{out} = I $（忽略通道间相关性），TURBOBOA 在一般且可能稠密的 $ H_{out} $ 下推导出更新规则，保留了注意力感知的通道依赖。

#### (iii) **自适应网格选择与坐标下降精炼（Adaptive Grid Selection with CD-based Refinement）**
- **问题**：BoA 使用固定的量化网格，但在迭代过程中权重被持续更新，导致初始网格与实际权重分布错位，尤其在低比特下影响显著。
- **解决方案**：
  1. **动态网格计算**：在每次量化前，基于**最新更新的权重**重新计算量化网格，保证对齐。
  2. **网格精炼**：在所有权重整数量化完成后，冻结整数权重 $ W_{int} $，仅通过**坐标下降（Coordinate Descent, CD）** 优化缩放因子 $ s $，以进一步最小化注意力重建误差。

---

### **相比现有方法的优势**

| 维度 | GPTQ | BoA | **TURBOBOA** |
| :--- | :--- | :--- | :--- |
| **效率** | ⭐⭐⭐⭐⭐ (高，并行量化) | ⭐⭐ (低，串行量化) | ⭐⭐⭐⭐ (高，联合量化) |
| **精度** | ⭐⭐ (低，忽略跨层依赖) | ⭐⭐⭐⭐ (高，注意力感知) | ⭐⭐⭐⭐⭐ (**更高**) |
| **误差传播处理** | ❌ | ❌ | ✅ (显式补偿) |
| **网格对齐** | ❌ | ❌ | ✅ (自适应+精炼) |

**总结优势**：TURBOBOA 成功打破了 BoA 中“精度高则速度慢”的固有 trade-off，实现了**速度与精度的双重超越**。

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **校准数据集（Calibration Data）**：用于量化过程的微调和参数学习。
  - `WikiText-2` (Wiki2)：随机采样 128 条长度为 2048 的序列。
- **测试数据集（Test Sets）**：用于评估量化后的模型性能。
  - `WikiText-2` (Wiki2)
  - `C4`

### **实验设置和评估指标**

- **模型**：Llama 系列模型，包括 `Llama3.2-1B`, `Llama3.2-3B`, `Llama3-8B`, `Llama3.1-70B`, `Llama2-7B`, `Llama2-13B`。
- **硬件**：NVIDIA H100 GPUs (80GB)，70B 模型使用双卡。
- **量化配置**：
  - **权重量化**：INT2 和 INT3。
  - **权激活量化**：W2A4KV4 / W2A4KV16（权重2bit，激活4bit，KV Cache 4bit 或 16bit）。
- **评估指标**：
  - **Perplexity (PPL)**：在 Wiki2 和 C4 测试集上，越低越好。
  - **Zero-shot Accuracy**：在 8 个常识推理任务上的平均准确率，越高越好。
  - **量化时间**：衡量算法效率。

### **基线方法对比**

- **基础量化器**：`RTN` (Round-to-Nearest), `GPTQ`。
- **先进方法**：`BoA` (直接基线)。
- **变换类方法**（Transformation-based）：
  - `QuaRot`, `SpinQuant`, `OSTQuant` (用于抑制异常值，常与 GPTQ/BoA 结合)。
- **其他**：`GPTAQ` (用于验证通道依赖的重要性)。

---

## 3. 主要实验结果和性能指标

### **关键性能数据与对比结果**

#### **(1) 速度对比 (Table 2)**

- 在 `Llama3.1-70B` 模型上，当 $ N=16 $ 时，TURBOBOA 将 BoA 的量化时间从 **16.99 小时**缩短至 **5.636 小时**，实现了 **超过 3 倍的加速**。
- 即使在较小模型上，加速也十分显著（如 1B 模型从 13.32 分钟降至 4.363 分钟）。

#### **(2) 权重-仅量化 (Weight-only Quantization) (Table 4)**

- **INT2 量化**：在 `Llama3.2-1B` 上，结合 `QuaRot`，TURBOBOA 将 Wiki2 PPL 从 BoA 的 **40.86** 显著降低到 **33.33**。
- **零样本准确率**：在 `Llama2-13B` 上，TURBOBOA 达到了 **69.07%**，非常接近全精度基线（69.83%），且比 BoA 高出至少 2 个百分点。

#### **(3) 权重-激活量化 (Weight-Activation Quantization) (Table 5)**

- 在 `W2A4KV16` 设置下，结合 `OSTQuant`，TURBOBOA 在 `Llama3.2-3B` 上将 C4 PPL 从 BoA 的 **74.04** 降低到 **63.75**。
- **零样本准确率增益巨大**：在 `Llama2-13B` 的 `W2A4KV4` 设置下，TURBOBOA 达到 **55.86%**，比 BoA 高出 **3 个百分点以上**，比 GPTQ 高出 **15 个百分点以上**。

#### **(4) 消融实验结果 (Ablation Studies)**

- **联合量化 (F1)**：验证了 $ N=16 $ 是效率与精度的最佳平衡点，更大的 $ N $ 加速收益递减。
- **误差补偿 (F2)**：在 `Llama3.2-1B` 上，单独加入 F2 可将 Wiki2 PPL 从 41.85 降至 37.15，证明了处理误差传播的有效性。
- **自适应网格 (F3)**：单独加入 F3 可将 PPL 降至 39.45，证明了动态网格对齐的重要性。
- **综合效果**：F2 和 F3 的组合带来了最佳性能（PPL 降至 33.33），表明二者互补。

---

## 4. 关键结论和发现

### **主要发现**

1. **联合量化是可行的**：即使同时量化多个 out-channels，只要配合精心设计的闭式误差补偿规则，也能有效捕捉通道间依赖，避免精度大幅下降。
2. **误差传播不容忽视**：来自前序层的量化误差会显著影响深层网络的性能，显式补偿是提升大模型量化鲁棒性的关键。
3. **动态对齐至关重要**：固定的量化网格在迭代 PTQ 中是次优的，根据更新后的权重动态调整网格并进行精炼，能持续提升最终精度。
4. **TURBOBOA 实现 SOTA**：在结合 `QuaRot`、`SpinQuant` 或 `OSTQuant` 等变换方法后，TURBOBOA 在 **weight-only** 和 **weight-activation** 量化两个领域均达到了最先进的（state-of-the-art）性能。

### **方法的局限性**

- **理论分析不足**：虽然实验表明联合量化 $ N $ 很大时性能依然稳定，但缺乏关于 $ N $ 与精度损失之间严格的理论误差界分析。
- **超参数敏感性**：稳定系数 $ \alpha $ 需要针对不同模型进行调优，自动化程度有待提高。

### **未来工作方向**

- 为联合量化参数 $ N $ 建立形式化的理论误差分析框架。
- 探索更高效的网格精炼算法，或将其与量化过程更紧密地耦合。
- 将 TURBOBOA 的思想扩展到其他类型的神经网络架构或更复杂的量化方案（如混合精度）。

</details>

---

### 16. [TADS: Task-Aware Data Selection for Multi-Task Multimodal Pre-Training](https://arxiv.org/abs/2602.05251)

**Authors**: Guanjie Cheng, Boyi Li, Lingyu Sun, Mengying Zhu, Yangyang Wu, Xinkui Zhao, Shuiguang Deng  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2602.05251v1  

#### Abstract
Large-scale multimodal pre-trained models like CLIP rely heavily on high-quality training data, yet raw web-crawled datasets are often noisy, misaligned, and redundant, leading to inefficient training and suboptimal generalization. Existing data selection methods are either heuristic-based, sufferin...

---

### 17. [ReThinker: Scientific Reasoning by Rethinking with Guided Reflection and Confidence Control](https://arxiv.org/abs/2602.04496)

**Authors**: Zhentao Tang, Yuqi Cui, Shixiong Kai, Wenqian Zhao, Ke Ye, Xing Li, Anxin Tian, Zehua Pei, Hui-Ling Zhen, Shoubo Hu, Xiaoguang Li, Yunhe Wang, Mingxuan Yuan  
**Category**: cs.AI  
**Published**: 2026-02-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.04496v1  

#### Abstract
Expert-level scientific reasoning remains challenging for large language models, particularly on benchmarks such as Humanity's Last Exam (HLE), where rigid tool pipelines, brittle multi-agent coordination, and inefficient test-time scaling often limit performance. We introduce ReThinker, a confidenc...

---

### 18. [SpectraKAN: Conditioning Spectral Operators](https://arxiv.org/abs/2602.05187)

**Authors**: Chun-Wun Cheng, Carola-Bibiane Sch\"onlieb, Angelica I. Aviles-Rivero  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.05187v1  

#### Abstract
Spectral neural operators, particularly Fourier Neural Operators (FNO), are a powerful framework for learning solution operators of partial differential equations (PDEs) due to their efficient global mixing in the frequency domain. However, existing spectral operators rely on static Fourier kernels ...

---

### 19. [CORP: Closed-Form One-shot Representation-Preserving Structured Pruning for Vision Transformers](https://arxiv.org/abs/2602.05243)

**Authors**: Boxiang Zhang, Baijian Yang  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.05243v1  

#### Abstract
Vision Transformers achieve strong accuracy but incur high compute and memory cost. Structured pruning can reduce inference cost, but most methods rely on retraining or multi-stage optimization. These requirements limit post-training deployment. We propose \textbf{CORP}, a closed-form one-shot struc...

---

### 20. [A Unified Framework for Rethinking Policy Divergence Measures in GRPO](https://arxiv.org/abs/2602.05494)

**Authors**: Qingyuan Wu, Yuhui Wang, Simon Sinong Zhan, Yanning Dai, Shilong Deng, Sarra Habchi, Qi Zhu, Matthias Gall\'e, Chao Huang  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.05494v1  

#### Abstract
Reinforcement Learning with Verified Reward (RLVR) has emerged as a critical paradigm for advancing the reasoning capabilities of Large Language Models (LLMs). Most existing RLVR methods, such as GRPO and its variants, ensure stable updates by constraining policy divergence through clipping likeliho...

---

### 21. [Exact Recovery in the Data Block Model](https://arxiv.org/abs/2602.05852)

**Authors**: Amir R. Asadi, Akbar Davoodi, Ramin Javadi, Farzad Parvaresh  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2602.05852v1  

#### Abstract
Community detection in networks is a fundamental problem in machine learning and statistical inference, with applications in social networks, biological systems, and communication networks. The stochastic block model (SBM) serves as a canonical framework for studying community structure, and exact r...

---

### 22. [Empirical-MCTS: Continuous Agent Evolution via Dual-Experience Monte Carlo Tree Search](https://arxiv.org/abs/2602.04248)

**Authors**: Hao Lu, Haoyuan Huang, Yulin Zhou, Chen Li, Ningxin Zhu  
**Category**: cs.AI  
**Published**: 2026-02-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.04248v1  

#### Abstract
Inference-time scaling strategies, particularly Monte Carlo Tree Search (MCTS), have significantly enhanced the reasoning capabilities of Large Language Models (LLMs). However, current approaches remain predominantly stateless, discarding successful reasoning patterns after each problem instance and...

---

### 23. [KV-CoRE: Benchmarking Data-Dependent Low-Rank Compressibility of KV-Caches in LLMs](https://arxiv.org/abs/2602.05929)

**Authors**: Jian Chen, Zhuoran Wang, Jiayu Qin, Ming Li, Meng Wang, Changyou Chen, Yin Chen, Qizhen Weng, Yirui Liu  
**Category**: cs.CL  
**Published**: 2026-02-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.05929v1  

#### Abstract
Large language models rely on kv-caches to avoid redundant computation during autoregressive decoding, but as context length grows, reading and writing the cache can quickly saturate GPU memory bandwidth. Recent work has explored KV-cache compression, yet most approaches neglect the data-dependent n...

---

### 24. [SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel](https://arxiv.org/abs/2602.04915)

**Authors**: Jose Miguel Luna, Taha Bouhsine, Krzysztof Choromanski  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.04915v1  

#### Abstract
We propose a new class of linear-time attention mechanisms based on a relaxed and computationally efficient formulation of the recently introduced E-Product, often referred to as the Yat-kernel (Bouhsine, 2025). The resulting interactions are geometry-aware and inspired by inverse-square interaction...

---

### 25. [Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](https://arxiv.org/abs/2602.04990)

**Authors**: Ioannis Anagnostides, Itai Zilberstein, Zachary W. Sollie, Arman Kilic, Tuomas Sandholm  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.04990v1  

#### Abstract
The allocation of scarce donor organs constitutes one of the most consequential algorithmic challenges in healthcare. While the field is rapidly transitioning from rigid, rule-based systems to machine learning and data-driven optimization, we argue that current approaches often overlook a fundamenta...

---

### 26. [Variational Speculative Decoding: Rethinking Draft Training from Token Likelihood to Sequence Acceptance](https://arxiv.org/abs/2602.05774)

**Authors**: Xiandong Zou, Jianshu Li, Jing Huang, Pan Zhou  
**Category**: cs.LG  
**Published**: 2026-02-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2602.05774v1  

#### Abstract
Speculative decoding accelerates inference for (M)LLMs, yet a training-decoding discrepancy persists: while existing methods optimize single greedy trajectories, decoding involves verifying and ranking multiple sampled draft paths. We propose Variational Speculative Decoding (VSD), formulating draft...

---

### 27. [Scaling In-Context Online Learning Capability of LLMs via Cross-Episode Meta-RL](https://arxiv.org/abs/2602.04089)

**Authors**: Xiaofeng Lin, Sirou Zhu, Yilei Chen, Mingyu Chen, Hejian Sang, Ioannis Paschalidis, Zhipeng Wang, Aldo Pacchiano, Xuezhou Zhang  
**Category**: cs.AI  
**Published**: 2026-02-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.04089v1  

#### Abstract
Large language models (LLMs) achieve strong performance when all task-relevant information is available upfront, as in static prediction and instruction-following problems. However, many real-world decision-making tasks are inherently online: crucial information must be acquired through interaction,...

---

### 28. [WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2602.04634)

**Authors**: Zelai Xu, Zhexuan Xu, Ruize Zhang, Chunyang Zhu, Shi Yu, Weilin Liu, Quanlu Zhang, Wenbo Ding, Chao Yu, Yu Wang  
**Category**: cs.AI  
**Published**: 2026-02-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.04634v1  

#### Abstract
Recent advancements in Large Language Models (LLMs) have largely focused on depth scaling, where a single agent solves long-horizon problems with multi-turn reasoning and tool use. However, as tasks grow broader, the key bottleneck shifts from individual competence to organizational capability. In t...

---

### 29. [BioACE: An Automated Framework for Biomedical Answer and Citation Evaluations](https://arxiv.org/abs/2602.04982)

**Authors**: Deepak Gupta, Davis Bartels, Dina Demner-Fuhsman  
**Category**: cs.CL  
**Published**: 2026-02-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.04982v1  

#### Abstract
With the increasing use of large language models (LLMs) for generating answers to biomedical questions, it is crucial to evaluate the quality of the generated answers and the references provided to support the facts in the generated answers. Evaluation of text generated by LLMs remains a challenge f...

---

### 30. [Late-to-Early Training: LET LLMs Learn Earlier, So Faster and Better](https://arxiv.org/abs/2602.05393)

**Authors**: Ji Zhao, Yufei Gu, Shitong Shao, Xun Zhou, Liang Xiang, Zeke Xie  
**Category**: cs.CL  
**Published**: 2026-02-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2602.05393v1  

#### Abstract
As Large Language Models (LLMs) achieve remarkable empirical success through scaling model and data size, pretraining has become increasingly critical yet computationally prohibitive, hindering rapid development. Despite the availability of numerous pretrained LLMs developed at significant computati...

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
