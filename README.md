# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-07 07:01:47 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Communication-free Sampling and 4D Hybrid Parallelism for Scalable Mini-batch GNN Training](https://arxiv.org/abs/2604.02651)

**Authors**: Cunyang Wei, Siddharth Singh, Aishwarya Sarkar, Daniel Nichols, Tisha Patel, Aditya K. Ranjan, Sayan Ghosh, Ali Jannesari, Nathan R. Tallent, Abhinav Bhatele  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2604.02651v1  

#### Abstract
Graph neural networks (GNNs) are widely used for learning on graph datasets derived from various real-world scenarios. Learning from extremely large graphs requires distributed training, and mini-batching with sampling is a popular approach for parallelizing GNN training. Existing distributed mini-b...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Communication-free Sampling and 4D Hybrid Parallelism for Scalable Mini-batch GNN Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的分布式 **mini-batch GNN training** 框架面临两大瓶颈：
- **采样开销高**：主流采样方法（如 GraphSAGE、GraphSAINT）依赖 CPU 或跨设备通信获取邻居节点和特征，导致严重的通信延迟。
- **扩展性差**：传统仅使用 **data parallelism (DP)** 的框架在增加 GPU 数量时，并不能有效降低端到端训练时间，甚至因通信开销上升而变慢。

### 提出的新方法与创新思路
作者提出 **ScaleGNN** —— 一个开源的、支持大规模扩展的 **4D 并行框架**，结合以下核心技术：

#### ✅ **Communication-free 分布式采样算法**
- 基于 **uniform vertex sampling** 的策略：每个 GPU 独立地从全局顶点集中均匀采样子集 $ S $，无需任何进程间通信。
- 构造诱导子图 $ G_S = (S, E_S) $，并通过 **unbiased edge rescaling**（对非自环边权重除以概率 $ p = (B-1)/(N-1) $）纠正采样偏差，保证聚合操作是全图聚合的无偏估计。

#### ✅ **4D Hybrid Parallelism 架构**
将总 GPU 组织为四维虚拟网格 $ G_d \times G_x \times G_y \times G_z $：
- **Data Parallelism ($ G_d $)**：复制模型副本，各组处理不同 mini-batch，通过 all-reduce 同步梯度。
- **3D Parallel Matrix Multiplication (3D PMM)**：在每组内将稀疏/稠密矩阵乘法分布到 $ G_x \times G_y \times G_z $ 的三维网格上，显著减少单个 GPU 内存压力和通信量。

#### ✅ 多项系统级优化
- **Overlap sampling with training**：用独立 CUDA stream 预取下一个 mini-batch，完全隐藏采样延迟。
- **Low-precision collective communication**：仅对 3D PMM 中的 all-reduce 使用 BF16，保留数值敏感部分（如 RMSNorm）为 FP32。
- **Kernel fusion**：融合 RMSNorm、ReLU、Dropout 等逐元素操作成单个 kernel，减少内存往返。
- **Communication-computation overlap**：利用 NCCL 异步特性重叠正反向传播中的 all-reduce 操作。

### 相比现有方法的优势
| 方面 | ScaleGNN | 传统方法（如 DistDGL, SALIENT++） |
|------|--------|-------------------------------|
| **采样方式** | GPU 上无通信采样 | CPU 采样 + 跨设备 fetch 远程邻居 |
| **并行维度** | 4D（DP + 3D PMM） | 通常只有 DP 或简单模型并行 |
| **通信开销** | 极低（仅梯度同步） | 高频远程特征访问 |
| **可扩展性** | 支持数千 GPU | 扩展性受限 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共五种图数据集，涵盖不同规模与领域：
| 数据集 | 类型 | 节点数 | 边数 | 任务 |
|-------|-----|--------|-------|------|
| `ogbn-products` | 商品分类 | ~2.4M | ~61M | 单标签分类 |
| `Reddit` | 社区分类 | ~233K | ~58M | 单标签分类 |
| `Isolate-3-8M` | 蛋白质相似网络 | ~3.8M | - | 合成特征用于扩展测试 |
| `Products-14M` | Amazon 商品图 | ~14M | ~115M | 合成特征 |
| `ogbn-papers100M` | 引用网络 | ~111M | ~1.6B | 多标签分类 |

> 注：后两者无原始特征，生成随机输入特征和合成类别以测试扩展能力。

### 实验设置
- **硬件平台**：
  - **Perlmutter**：NVIDIA A100 GPUs（最多 2048）
  - **Frontier**：AMD MI250X GCDs（最多 2048）
  - **Tuolumne**：AMD MI300A APUs（最多 1024）
- **模型架构**：基于 GCN，加入 RMSNorm、Residual Connection、Dropout。
- **批大小**：固定 mini-batch size $ B $，随 $ G_d $ 增加而增大有效 batch size。

### 评估指标
- **主指标**：**端到端训练时间达到目标准确率**
  - Reddit：95%
  - ogbn-products：79%
- **辅助指标**：
  - 每 epoch 时间（scaling 实验）
  - 测试精度（accuracy）
  - 消融实验中各项优化带来的加速比

### 基线方法对比
| 基线 | 类型 | 特点 |
|------|------|------|
| **BNS-GCN** | Full-graph | 全图训练 + boundary sampling 减少通信 |
| **DistDGL** | Mini-batch | 图分区 + 分布式 KV 存储 + neighbor sampling |
| **MassiveGNN** | Mini-batch | DistDGL 延伸，优化 feature fetching |
| **SALIENT++** | Mini-batch | CPU 加速采样 + feature caching |

> 所有 baseline 均调优至最佳配置。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔹 端到端训练速度提升
| 平台 | 数据集 | ScaleGNN 时间 | 最佳 Baseline | 加速比 |
|------|--------|---------------|----------------|---------|
| Perlmutter | ogbn-products | **3.80s @64 GPUs** | SALIENT++ (13.25s) | **3.5×** |
| Perlmutter | Reddit | **0.98s @16 GPUs** | SALIENT++ (3.13s) | **3.2×** |
| Frontier | ogbn-products | **10.2s @32 GCDs** | MassiveGNN (1651.73s) | **162×** |
| Frontier | Reddit | **0.98s @16 GCDs** | DistDGL (596.09s) | **608×** |

> ⚠️ 注意：部分 baseline 在小规模下也未能收敛至目标精度（见 Figure 6）。

#### 🔹 强扩展性（Strong Scaling）
ScaleGNN 在三大超算平台上均展现出优异的强扩展能力：

| 平台 | 数据集 | GPU/GCD 数量范围 | 速度提升倍数 |
|------|--------|--------------------|--------------|
| Perlmutter | ogbn-papers100M | 64 → 2048 | **21.7×** |
| Frontier | Products-14M | 32 → 1024 | **22.4×** |
| Tuolumne | Products-14M | 32 → 1024 | **17.2×** |

> 即使在 2048 GPU/GCD 上仍保持良好效率。

#### 🔹 评估阶段性能优势
由于 ScaleGNN 使用 **3D PMM 分布全图与模型**，支持高效的分布式 full-graph evaluation：

| 数据集 | ScaleGNN | BNS-GCN | SALIENT++ | DistDGL/MassiveGNN |
|--------|----------|----------|------------|---------------------|
| ogbn-products (8 GPUs) | **0.19s** | 6.89s (**36×**) | 10.12s (**54×**) | 20.82s (**111×**) |
| Reddit (4 GPUs) | **0.05s** | 1.79s (**36×**) | 1.13s (**23×**) | 12.50s (**250×**) |

> 传统方法需重复采样或回退到 CPU 推理，严重拖慢整体流程。

### 消融实验结果（Ablation Study）
在 `ogbn-products` 上进行逐步优化验证（8 和 32 GPU）：

| 优化项 | 对 epoch time 的贡献（相对 baseline） |
|--------|------------------------------------|
| **Overlap sampling** | ↓24% |
| **BF16 communication** | ↓17% (8 GPU), ↓16% (32 GPU) |
| **Kernel fusion** | ↓6% (8 GPU), ↓4% (32 GPU) |
| **Comm-comp overlap** | ↓3% (8 GPU), ↓2% (32 GPU) |
| **累计加速** | **1.75× (8 GPU), 1.66× (32 GPU)** |

> 表明所有四项优化均有实际收益，且可叠加。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Communication-free sampling 是可行且高效的**  
   uniform vertex sampling + unbiased rescaling 可实现无通信采样，同时达到甚至超过 GraphSAGE 和 GraphSAINT 的精度（Table I）：
   - ogbn-products: **81.3%** vs GraphSAINT (80.2%), GraphSAGE (79.6%)

2. ✅ **4D hybrid parallelism 显著提升可扩展性**  
   结合 DP 与 3D PMM，使得 ScaleGNN 能够高效扩展至 **2048 GPUs / GCDs**，突破传统 mini-batch 框架的瓶颈。

3. ✅ **采样不再是性能瓶颈**  
   通过流水线预取（overlap），采样时间被彻底移出关键路径，在大规模下占比极低（Figure 8）。

4. ✅ **端到端训练效率远超现有系统**  
   在多个平台和数据集上实现 **数十至上百倍的速度提升**，尤其在 Frontier 上表现惊人。

### 方法的局限性
- **依赖高质量随机种子同步**：所有 GPU 必须共享相同随机状态以构造一致的采样集。
- **适用于静态图**：当前设计假设图结构不变；动态图需额外机制维护一致性。
- **对非常深的 GNN 层可能受限于 layer rotation 周期**（周期为3），但可通过调整分片策略缓解。

### 未来工作方向
- 将 4D 并行扩展至其他 GNN 架构（如 GAT、Transformer-based GNNs）。
- 支持异构计算环境下的自动并行策略选择。
- 探索更复杂的采样策略（如重要性采样）在 communication-free 框架下的实现。
- 集成自动混合精度（AMP）与 fault tolerance 机制，增强生产可用性。

---

> **一句话总结**：  
> ScaleGNN 通过 **communication-free uniform sampling + 4D hybrid parallelism（DP + 3D PMM）**，实现了前所未有的 mini-batch GNN 训练可扩展性和端到端性能，是迈向超大规模图学习的重要一步。

</details>

---

### 2. [DARE: Diffusion Large Language Models Alignment and Reinforcement Executor](https://arxiv.org/abs/2604.04215)

**Authors**: Jingyi Yang, Yuxian Jiang, Xuhao Hu, Shuang Cheng, Biqing Qi, Jing Shao  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.04215v1  

#### Abstract
Diffusion large language models (dLLMs) are emerging as a compelling alternative to dominant autoregressive models, replacing strictly sequential token generation with iterative denoising and parallel generation dynamics. However, their open-source ecosystem remains fragmented across model families ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**DARE: Diffusion Large Language Models Alignment and Reinforcement Executor**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **diffusion large language models (dLLMs)** 的研究面临严重的**系统碎片化**问题：
- 不同 dLLM 模型（如 LLaDA、Dream、SDAR、LLaDA2.x）通常依赖独立的代码库进行 post-training（后训练），缺乏统一框架。
- 强化学习（RL）目标、rollout 实现、reward 接口、评估脚本等均分散在各论文专属仓库中，导致：
  - **复现困难**，工程成本高；
  - **算法比较不公平**，因执行环境不一致；
  - **无法共享基础设施**，阻碍社区协作。

此外，传统 LLM 的 RL 框架（如基于自回归生成、token-level log-probabilities）**不能直接用于 dLLMs**，因为 dLLMs 使用迭代去噪（iterative denoising）、并行生成、非顺序采样等机制。

---

### 🚀 提出的新方法：DARE 框架

作者提出 **DARE (dLLMs Alignment and Reinforcement Executor)** ——一个**开源、统一的 dLLM 后训练与评估框架**，其核心创新包括：

#### 主要贡献：
1. **统一执行栈（Unified Execution Stack）**
   - 支持多种 dLLM 架构：**masked diffusion LM (MDLMs)** 和 **block diffusion LM (BDLMs)**。
   - 集成主流模型家族：LLaDA、Dream、SDAR、LLaDA-MoE、LLaDA2.0/2.1。
   - 统一支持多种 post-training 范式：
     - Supervised Fine-Tuning (SFT)
     - Parameter-Efficient Fine-Tuning (PEFT)
     - Preference Optimization (如 DPO, VRPO)
     - 多种 dLLM-specific RL 算法（如 VRPO, D1, Coupled-GRPO, MDPO, CJ-GRPO, SPG, BGPO, EBPO）

2. **模块化设计与可插拔架构**
   - 抽象为三大组件：**Worker（角色）**, **DataFlow（数据流）**, **Workflow（优化流程）**。
   - 算法差异通过“hook”机制实现，而非重写整个 pipeline，提升可扩展性和公平对比能力。

3. **系统级优化（System-Level Acceleration）**
   - **训练侧优化**：
     - 对 MDLMs 使用 `flash_attn_varlen_func` 减少 padding 开销；
     - 对 BDLMs 使用 FlexAttention 支持块状注意力约束。
   - **推理/rollout 侧优化**：
     - MDLMs：采用 **Fast-dLLM + KV Cache** 加速采样；
     - BDLMs：集成 LMDeploy 和 SGLang 实现高效 rollout。
   - **解耦训练与 rollout 后端**：不同阶段使用最优 backend，实现端到端约 **4×（MDLM）至 14×（BDLM）加速**。

4. **内置 dLLM-aware 评估平台**
   - 基于 OpenCompass 构建，支持多 benchmark 自动评测。
   - 支持模型特定的 inference backend（如 Fast-dLLM for MDLMs, LMDeploy for BDLMs），确保评估一致性。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方式 | DARE |
|------|--------|------|
| **可复现性** | Paper-specific code，难以复现 | 统一框架，一键运行 |
| **公平比较** | 执行环境混杂，结果不可比 | 相同 workflow 下对比算法 |
| **工程负担** | 每次需重构 pipeline | 即插即用式算法接入 |
| **系统效率** | 缺乏针对性优化 | 训练/rollout 双路径加速 |
| **生态整合** | 孤立发展 | 连接 verl + OpenCompass 生态 |

> 💡 DARE 并非提出新算法，而是构建了一个**可重用的研究基底（research substrate）**，推动 dLLM 领域从“论文+私有代码”向“标准化平台+开放协作”演进。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
DARE 在多个任务上进行了 post-training 和评估，涵盖以下领域：

| 任务类别 | 数据集 |
|--------|-------|
| **通用问答 / 推理** | MMLU, MMLU-Pro, Hellaswag, ARC-C |
| **数学推理** | GSM8K, MATH, AIME24, AIME25, OlympiadBench |
| **代码生成** | HumanEval, MBPP |
| **规划任务** | Countdown, Sudoku |

---

### ⚙️ 实验设置
- **模型骨干**：
  - LLaDA-8B-Instruct（MDLM）
  - Dream-7B-Instruct（MDLM）
  - SDAR-8B/30B-Chat（BDLM）
  - LLaDA2.0/2.1-mini（BDLM）

- **训练配置**：
  - Rollout group size: 8
  - Block length: 32
  - KL 正则化默认关闭
  - Monte Carlo 采样数：16（用于 ELBO-based 方法）
  - 最大响应长度：512（数学）、256（规划）
  - 去噪步数（diffusion steps）：256（数学）、128（规划）
  - 训练轮数：1 epoch

- **评估指标**：
  - 准确率（Accuracy）为主，如 MMLU、GSM8K、HumanEval pass@1 等。
  - 训练曲线监控 reward 收敛性与稳定性。

- **基线方法对比**
  包括但不限于：
  - Baseline（SFT）
  - d1 (Zhao et al., 2025b)
  - Coupled-GRPO (Gong et al., 2025)
  - VRPO (Zhu et al., 2025a)
  - CJ-GRPO (Yang et al., 2025)
  - SPG (Wang et al., 2025a)
  - BGPO (Lin et al., 2025)
  - EBPO (Bie et al., 2026)

> 所有方法均在 DARE 框架内实现，保证 rollout、reward、evaluation 环境一致。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（来自 Table 2–5）

#### 表格概览：
| 模型 | MMLU | GSM8K | MATH | HumanEval |
|------|------|-------|------|-----------|
| LLaDA-8B-Instruct | 65.24 | 76.5 → **85.6** (CJ-GRPO) | 34.6 → **41.0** (Coupled-GRPO) | 46.9 → **52.4** (VRPO) |
| Dream-7B-Instruct | 66.83 | 77.2 → **85.7** (SPG) | 39.6 → **50.7** (CJ-GRPO) | 57.9 → **61.6** (Coupled-GRPO) |
| SDAR-8B-Chat | 77.23 | 91.36 | 78.40 | 79.88 |
| LLaDA2.1-mini | 69.91 | 86.13 | 84.56 | 81.10 |

> 注：箭头表示经 DARE 中 RL 微调后的最佳提升。

---

### 🔬 详细算法对比结果

#### 数学任务（Table 3）
- **LLaDA-8B** 上：
  - CJ-GRPO 在 GSM8K 上表现最好（85.6），Coupled-GRPO 在 MATH 上领先（41.0）。
- **Dream-7B** 上：
  - CJ-GRPO 全面领先（GSM8K: 85.7, MATH: 50.7），显著优于 baseline。

#### 代码任务（Table 4）
- **LLaDA-8B**：VRPO 在 HumanEval（52.4）和 MBPP（42.8）上最强。
- **Dream-7B**：Coupled-GRPO 表现最佳（HumanEval: 61.6, MBPP: 60.3）。

#### 规划任务（Table 5）
- **Countdown**：Coupled-GRPO 显著胜出（77.9 vs baseline 16.8）。
- **Sudoku**：BGPO 表现最佳（42.6），远超其他方法。

> ❗重要发现：**没有单一算法在所有任务和模型上都最优**，性能高度依赖 backbone 和任务类型。

---

### ⚖️ 消融与系统性能分析

#### 系统加速效果（Figure 2）
- **训练延迟（SFT）**：
  - 使用 `flash_attn_varlen_func` 相比 eager mode 实现 **2.0× 加速**（22.1s → 10.8s/iter）。
- **rollout 延迟**：
  - Fast-dLLM + flash_attn_with_kvcache 将 rollout 延迟从 161.6s 降至 73.5s，**提速 2.2×**。
  - 整体 RL pipeline 实现 **4× 加速（MDLM）** 和 **超过 14× 加速（BDLM）**。

#### 算法稳定性分析（Figure 3）
- **稳定算法**：d1、Coupled-GRPO、CJ-GRPO reward 曲线平滑，收敛可靠。
- **不稳定算法**：
  - SPG、BGPO（ELBO-based）在样本不足时出现 reward collapse（如 LLaDA 上 Countdown 任务）。
  - Dream 上 SPG 在数学和代码任务中 reward 下降明显。

> 结论：**ELBO-based 方法对 MC 采样数敏感，需更多计算资源维持稳定；而一步去噪类方法（如 d1）更鲁棒。**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **No Silver Bullet Algorithm**  
   没有一种 RL 算法在所有任务和模型上均占优。例如：
   - CJ-GRPO 在 Dream 上数学强，但在 LLaDA 上代码弱；
   - VRPO 擅长代码，但规划任务一般；
   - BGPO 在 Sudoku 上突出，但在 Countdown 上崩溃。

2. **Backbone Matters**  
   同一算法在不同模型上的表现差异巨大。例如 SPG 在 Dream 上代码性能差，说明算法需与模型结构协同设计。

3. **Stability ≠ Performance**  
   高性能算法（如 BGPO）可能不稳定；简单方法（如 d1）虽非最高分，但更稳健，适合作为 baseline。

4. **System Optimization 是关键瓶颈**  
   训练/rollout 后端选择对整体效率影响巨大，**decoupling rollout and training backend** 是有效策略。

5. **DARE 提升可复现性与公平性**  
   将原本分散在多个仓库中的算法统一到同一 executor 中，使得跨 paper 比较成为可能。

---

### ⚠️ 局限性
- 当前主要支持文本 dLLMs，尚未扩展至 vision-language 或 multimodal diffusion models。
- 某些 closed-source 方法（如 VRPO、EBPO）仅部分集成，可能存在实现偏差。
- 缺乏大规模超参搜索或 long-horizon reasoning 任务验证（如 Agent-based tasks）。
- 对 extremely long context（>32K）的支持仍待加强。

---

### 🔮 未来工作方向
1. **模型扩展**：
   - 支持 diffusion vision-language models（如 Gemini Diffusion）。
   - 集成 variable-length generation 方法（如 p-EOS Yang et al., 2026）。

2. **算法演进**：
   - 吸纳新型 RL 目标函数、control variates、stability-enhancing techniques。
   - 探索 hybrid training schemes（如 combining SFT + RL + CoT distillation）。

3. **系统增强**：
   - 更细粒度的 efficiency ablation（memory, latency, energy）。
   - 部署友好的 evaluation backends（支持量化、蒸馏、边缘设备）。

4. **社区共建**：
   - 开源生态建设，鼓励第三方贡献新算法、新模型适配器。
   - 构建 dLLM Leaderboard，推动标准化 benchmarking。

---

## 总结

> **DARE 不是一个新算法，而是一个“让好算法被看见”的基础设施。**

它解决了 dLLM 领域日益严重的**碎片化、不可比、难复现**问题，通过统一框架实现了：
- ✅ 多模型支持（MDLM & BDLM）
- ✅ 多算法集成（SFT, PEFT, PO, RL）
- ✅ 高效系统优化（训练/rollout 解耦加速）
- ✅ 可信评估闭环（dLLM-aware evaluation）

实验表明，在 DARE 框架下，不同算法的真实性能差异得以暴露，且系统优化带来显著效率增益。该工作标志着 dLLM 研究正从“模型为中心”迈向“生态系统为中心”，为未来大规模协作奠定了基础。

</details>

---

### 3. [Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers](https://arxiv.org/abs/2604.02344)

**Authors**: J\k{e}drzej Maczan  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.02344v1  

#### Abstract
WebGPU's security-focused design imposes per-operation validation that compounds across the many small dispatches in neural network inference, yet the true cost of this overhead is poorly characterized. We present a systematic characterization of WebGPU dispatch overhead for LLM inference at batch s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文系统性地研究了 **WebGPU 在 LLM 推理中的调度开销（dispatch overhead）**，这是当前 WebGPU 用于高性能机器学习部署时面临的关键瓶颈。由于 WebGPU 的安全设计引入了每操作验证、命令缓冲区提交等机制，这些开销在 LLM 中大量小操作的场景下被显著放大，但其真实成本此前缺乏准确量化。

### 提出了什么新方法或新思路
- **Sequential-Dispatch Methodology（顺序调度测量法）**  
  提出了一种新的微基准测试方法，通过连续执行多个 dispatch 并仅在末尾同步，从而分离出真实的 per-dispatch 开销，避免传统单次操作测量中因频繁 GPU-CPU 同步而高估开销的问题。
  
- **构建了 torch-webgpu：一个基于 PrivateUse 的 PyTorch 外部后端**  
  开发了一个完整的从 `torch.compile()` FX 图到 WGSL 着色器的编译流程，并集成 Dawn 实现 WebGPU 加速，为 WebGPU 上的端到端 LLM 推理提供了可复现的研究平台。

- **提出“per-dispatch cost”与“per-operation overhead”的区分**  
  明确将总延迟分解为：
  - **Per-dispatch cost**：纯 WebGPU API 层面的开销（如 encoder 创建、bind group 设置、submit）
  - **Per-operation overhead**：包含 Python 解释器、框架逻辑在内的完整操作开销（约 95 μs）

### 相比现有方法的优势
- 首次跨 **四家 GPU 厂商（NVIDIA, AMD, Apple, Intel）、三种 backend（Vulkan, Metal, D3D12）和三大浏览器（Chrome, Safari, Firefox）** 对 WebGPU 调度开销进行统一量化。
- 揭示了 naive 单操作 benchmark 会高估开销达 **~20×**，纠正了社区对 WebGPU 性能的认知偏差。
- 通过 kernel fusion 实验提供因果证据，确认 **per-operation overhead 是 batch=1 下的主要瓶颈**，而非 kernel 效率本身。

---

## 2. 核心实验方法和设置

### 使用了哪些模型
- 主要测试模型为：
  - **Qwen2.5-0.5B-Instruct**（494M 参数，24 层）
  - **Qwen2.5-1.5B-Instruct**（1.54B 参数，28 层）
- 所有推理任务均为 **autoregressive generation**，输入提示 `"The capital of France is"`，生成 50 个 token。

### 实验设置
- **硬件平台**：
  - 主平台：NVIDIA RTX 5090 + AMD Ryzen 7 9800X3D + Ubuntu 24.04
  - 其他平台：Windows 笔记本（RTX PRO 2000）、macOS（Apple M2）
- **软件栈**：
  - 自研后端：`torch-webgpu`（基于 Dawn + WGSL）
  - 对比基线：CUDA（PyTorch）、MPS（PyTorch）、ONNX Runtime (WebGPU)、WebLLM（浏览器）
- **实现方式**：
  - 使用 `torch.compile()` 获取 FX graph，分析计算节点数量（共 1,911 节点，其中 ~876 可能成为 dispatch）
  - 实现多种 kernel fusion（RMSNorm, MLP gate/up/silu, K+V projection）

### 评估指标
| 指标 | 定义 |
|------|------|
| **Tokens/sec (tok/s)** | 总生成 token 数 / 总耗时，反映 decode 吞吐量 |
| **Time to First Token (TTFT)** | 从开始到输出第一个 token 的时间，反映 prefill + 第一步 decode 延迟 |
| **Coefficient of Variation (CV)** | 标准差 / 均值，衡量运行稳定性 |
| **Per-dispatch cost** | 通过 sequential dispatch 测得的单次 dispatch 开销（μs） |
| **Per-operation overhead** | 由融合前后 TTFT 差异推导得出的每操作总开销 |

### 基线方法对比
| 基线 | 类型 | 精度 | 平台 |
|------|------|------|------|
| CUDA (eager/compiled) | 原生 GPU | fp16/fp32 | Linux/Windows |
| MPS | Apple 原生加速 | fp16/fp32 | macOS |
| CPU (PyTorch eager) | CPU 推理 | fp32 | 多平台 |
| ONNX Runtime (WebGPU) | WebGPU 原生运行时 | fp32 | RTX 5090 |
| WebLLM | 浏览器内推理引擎 | q4f16 | Chrome/Safari/Firefox |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Qwen2.5-0.5B）

#### 端到端吞吐量（tok/s）
| Backend | Dtype | Tok/s | vs CUDA(fp16) |
|--------|-------|--------|----------------|
| CUDA (RTX 5090) | fp16 | 185.5 | 1.00× |
| MPS (M2) | fp16 | 47.8 | 0.26× |
| **torch-webgpu (fused)** | **fp32** | **21.0** | **0.11×** |
| ONNX Runtime (WebGPU) | fp32 | 13.1 | 0.07× |
| CPU (Ryzen) | fp32 | 13.7 | 0.07× |

> 注：`torch-webgpu` 在 float32 下达到 21.0 tok/s，约为 CUDA fp16 的 11%，但若与 dtype-matched 的 CUDA fp32 对比，则差距缩小至约 **8.8× → 1.4×**

#### 跨平台性能比较（dtype-matched float32）
| Platform | Accelerator | Tok/s | vs WebGPU |
|---------|-------------|--------|----------|
| Linux | RTX 5090 (CUDA) | 185.5 | 8.8× |
| Windows | RTX PRO 2000 (CUDA) | 30.1 | **1.4×** |
| macOS | M2 (MPS) | 12.9 | 0.61× |

> 尽管 RTX PRO 2000 的算力仅为 RTX 5090 的 ~1/6，却实现了 WebGPU 的 **1.4× 吞吐**，说明性能瓶颈不在计算能力，而在 **dispatch/framework 开销**。

### 微基准：真实 per-dispatch 开销
| Implementation | Backend | Per-dispatch cost (μs) |
|----------------|---------|------------------------|
| Dawn (RTX 5090) | Vulkan | **23.8** |
| wgpu (RTX 5090) | Vulkan | 35.8 |
| Chrome (RTX 5090) | Vulkan | 32.8 |
| Safari (M2) | Metal | **31.7** |
| wgpu (M2) | Metal | 71.1 |
| Firefox (all) | D3D12/Metal | ~1040 ✅（疑似 rate-limiting） |

> - **Vulkan 后端普遍在 24–36 μs**
> - **Metal 后端差异大**：Safari 优化极好（31.7 μs），但 wgpu-native 达 71.1 μs（相差 **2.2×**）
> - **Firefox 存在严重限制**，不适用于 ML 推理

### 消融实验结果

#### Kernel Fusion 影响（控制变量实验）
| 配置 | Dispatches saved | Tok/s | TTFT (ms) | 提升 |
|------|------------------|--------|-----------|------|
| 无融合（baseline） | 0 | 13.5 | 71.4 | — |
| + RMSNorm fusion (6→1) | 240 | 19.4 | 46.6 | +44% |
| + MLP gate/up/silu fusion | +48 | 20.5 | 43.3 | +6% |
| + K+V projection fusion | +24 | 20.6 | 41.6 | +0.5% |
| **总计** | **312** | **21.0** | **↓41%** | **↑53%** |

> - **RMSNorm 和 MLP 融合带来显著收益**
> - **K+V 融合无统计显著性**（p=0.42），且中间变量仅占 ~1.8MB，证明提升来自减少 dispatch 而非内存节省
> - 支持核心论断：**减少 dispatch 数量是关键优化路径**

#### 不同 backend 的 fusion 效果对比
| Backend | RMSNorm Fusion Speedup |
|--------|------------------------|
| Vulkan (native) | 1.4–1.7× ✅ |
| Metal (wgpu) | 0.95× ❌ |
| Metal (Safari) | 0.91× ❌ |
| Vulkan (Chrome) | 1.06× ⚠️ |

> 表明 **backend 选择决定优化策略有效性**：Vulkan 可受益于 fusion，Metal 则不然。

#### WebGPU 内核效率（kernel efficiency）
| Operation | Dimensions | TFLOP/s | % Peak (RTX 5090) |
|----------|------------|--------|-------------------|
| MLP up proj | 896×896×4864 | 1.22 | **1.2%** |
| MLP down proj | 896×4864×896 | 2.06 | **2.0%** |
| Toy matmul | 256×256×256 | 0.03 | <0.1% |

> - 当前未优化 WGSL 实现仅利用 **1–2% 的 FP32 峰值性能**
> - 第三方报告可达 ~17%，说明仍有巨大优化空间

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Naive benchmark 高估 WebGPU 调度开销达 ~20×**  
   顺序调度测量法揭示真实 per-dispatch cost 为：
   - **Vulkan: 24–36 μs**
   - **Metal: 32–71 μs**
   - 而非传统单次测量得到的数百微秒。

2. ✅ **Per-operation overhead 主导 batch=1 推理延迟**  
   - 总开销约 **~95 μs/operation**，其中：
     - WebGPU API 层：~24–36 μs
     - Python/Framework 层：~59–71 μs
   - 即使 kernel 效率达到 17%，该 overhead 仍将是主要瓶颈。

3. ✅ **Backend 选择比厂商更重要**  
   - Vulkan 普遍优于 Metal
   - 同一 backend 下不同实现差异巨大（如 Safari Metal vs wgpu-native Metal：**2.2× 差距**）

4. ✅ **Kernel fusion 在 Vulkan 上有效，在 Metal 上无效**  
   - Vulkan 上 fusion 可提升 **53% 吞吐**
   - Metal 上 fusion 几乎无益甚至负向（可能因编译器已优化）

5. ✅ **Firefox 存在疑似 rate-limiting 行为**  
   - 所有平台上 per-dispatch 成本均高达 ~1040 μs，不适合 ML 推理

6. ✅ **dtype 不匹配严重扭曲性能对比**  
   - torch-webgpu 使用 fp32，而 CUDA/MPS 默认用 fp16
   - 若统一为 fp32，WebGPU 与移动级 CUDA GPU（RTX PRO 2000）差距仅为 **1.4×**，远小于宣称的 8.8×

### 方法的局限性
| 限制项 | 说明 |
|--------|------|
| **仅支持 batch=1** | 所有实验均为自回归生成，未探索 batch inference 的开销摊销潜力 |
| **仅限 float32** | 未能启用 WGSL float16，无法公平对比主流 fp16 推理方案 |
| **torch-webgpu 仅验证于单一平台** | 主要结果基于 RTX 5090 + Dawn，缺乏跨平台验证 |
| **kernel 效率低** | 报告的 1–2% 是未优化结果，不代表 WebGPU 极限 |
| **overhead 分解具不确定性** | 框架开销部分为推导值，误差估计约 ±30% |

### 未来工作方向
1. **支持 batched inference**  
   探索更大 batch size 下 dispatch overhead 是否被有效摊销（附录 F 预测 crossover batch 在 7–119 之间）。

2. **实现 WGSL float16 支持**  
   引入半精度计算以缩小与 CUDA/MPS 的直接性能差距。

3. **开发 WebGPU Graph Capture 机制**  
   类似 CUDA Graphs，捕获并重放整个推理图，彻底消除 per-dispatch 开销。

4. **推动 WebGPU 规范改进**  
   提议添加 compute graph capture、persistent kernels、cooperative groups 等特性，权衡安全性与性能。

5. **更深入的浏览器内核分析**  
   理解 Safari 为何在 Metal 上表现优异，以及 Firefox 的 rate-limiting 机制。

6. **集成自动调优工具（如 Triton）**  
   提升 WGSL kernel 的计算效率，逼近 17%+ 的理论上限。

---

> 🔗 **开源资源**：所有代码、基准脚本、原始数据均已公开  
> GitHub: [https://github.com/jmaczan/torch-webgpu](https://github.com/jmaczan/torch-webgpu)

</details>

---

### 4. [CAWN: Continuous Acoustic Wave Networks for Autoregressive Language Modeling](https://arxiv.org/abs/2604.04250)

**Authors**: Dejan \v{C}ugalj, Aleksandar Jevremovic  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.04250v1  

#### Abstract
Modern Large Language Models (LLMs) rely on Transformer self-attention, which scales quadratically with sequence length. Recent linear-time alternatives, like State Space Models (SSMs), often suffer from signal degradation over extended contexts. We introduce the Continuous Acoustic Wave Network (CA...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CAWN: Continuous Acoustic Wave Networks for Autoregressive Language Modeling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大语言模型（LLMs）普遍依赖 **Transformer** 架构中的 **self-attention** 机制，其计算和内存复杂度为 $O(L^2)$，严重限制了可处理的上下文长度（即“$O(L^2)$ 内存墙”）。虽然已有如 **State Space Models (SSMs)** 和 **RetNet** 等线性时间 $O(L)$ 方法，但它们在超长序列上常面临信号衰减或数值不稳定问题。

本文旨在突破这一瓶颈，提出一种全新的、完全连续的序列混合架构，以实现高效且鲁棒的超长上下文建模。

---

### 🚀 提出的新方法与核心创新

作者提出了 **Continuous Acoustic Wave Network (CAWN)**，将离散的 token 序列映射为连续的复数域声波信号，通过**波的干涉与共振**来建模语言结构。其核心创新如下：

#### **1. Complex-Domain Wave Embedding**
- 将隐藏状态投影为多头复数域 **phasor**（相量），同时跟踪幅度和相位。
- 用连续的波形叠加替代离散的 token 对注意力计算。

#### **2. Causal Phase Accumulation (因果相位累积)**
- 引入一个严格线性时间 $O(L)$ 的序列混合机制。
- 利用 **true-complex phase rotation**（基于欧拉公式）动态累积历史相位，天然编码相对位置信息。

#### **3. Dual-Gated Selective Phase Resonance**
为防止长程信号退化，设计了双重门控机制：
- **Frequency-Dependent Retention Gate**：低频保留全局记忆（高保留率 ~0.9999），高频用于局部快速更新（低保留率 ~0.5）。
- **Hard-Threshold Gating with STE**：使用 **Straight-Through Estimator (STE)** 将微弱信号强制置零，防止数值下溢和噪声积累。

#### **4. Temporal Syntax Cache**
- 在全局波投影前，使用 **depth-wise 1D-Convolution (kernel=3)** 缓冲短期语法依赖，解耦局部与全局建模。

#### **5. Depth-wise Harmonic Convolution**
- 在谐波通道间进行深度卷积，允许相邻频率交互滤波，提升参数效率。

#### **6. Block Attention Residuals**
- 受 Moonshot AI 启发，引入深度维度上的 attention residuals，定期“切断”残差流并归档早期块状态，避免深层网络中的信号稀释。
- 支持从早期块中检索未被稀释的 phasor 状态。

#### **7. Custom Triton Kernels for Hardware Efficiency**
- 使用自定义 **Triton kernel** 实现相位累加器，在 SRAM 中完成全部复数运算，绕过高带宽内存（HBM）瓶颈。
- 支持 **float32 真复数运算**，确保数值稳定性。

---

### 🔍 相比现有方法的优势

| 特性 | CAWN | Transformer | SSMs (如 Mamba) | FNet/RWKV |
|------|------|------------|------------------|-----------|
| 时间复杂度 | $O(L)$ | $O(L^2)$ | $O(L)$ | $O(L)$ |
| 空间复杂度（推理） | $O(1)$ state-passing | $O(L)$ KV Cache | $O(1)$ state | $O(1)$ state |
| 超长上下文支持 | ✅ 至 2M tokens | ❌ 受限 | ⚠️ 易信号衰减 | ⚠️ 数值漂移 |
| 参数效率 | 高（谐波共享） | 低 | 中等 | 中等 |
| 可解释性 | 高（物理类比：波干涉） | 低 | 中等 | 中等 |

> ✅ **核心优势**：CAWN 在保持 $O(L)$ 训练复杂度的同时，实现了真正的 $O(1)$ 推理状态传递，并通过物理启发的波动力学机制，显著提升了超长上下文下的信号保真度。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **训练数据**：1000亿 token 的英文语料，混合比例为：
  - 50% FineWeb-Edu PDFs
  - 30% DCLM
  - 20% 标准 FineWeb-Edu
- **验证数据**：`Salesforce/wikitext-103`，用于评估泛化能力。
- **Tokenization**：采用 Meta-Llama-2 的 **BPE tokenizer**，词表大小 $V=32,000$。
- **训练方式**：无限流式训练（infinite streaming pipeline），使用 `IterableDataset`，避免过拟合。

---

### ⚙️ 实验设置

#### **模型规模**
- 模型名称：**CAWN-150M**
- 参数量：约 1.5 亿
- 结构：
  - Embedding 维度 $D = 896$
  - 层数 $N = 16$（每 4 层一组，共 4 个 block）
  - Acoustic Heads $H = 4$，每头 $K = 64$ 谐波 → 总谐波数 256
  - FFN 扩展因子 4×（3584 units）

#### **训练配置**
- 优化器：AdamW（bfloat16 混合精度）
- 学习率：Cosine Annealing，峰值 $8.0 \times 10^{-4}$，5% 线性预热
- 批大小：micro-batch size=7，gradient accumulation=36 → effective batch=252
- 序列长度：$L = 1024$（训练窗口），但**相位状态跨批次缓存**，实现“无限上下文”预训练
- Dropout：0.1
- 初始化：Normal($\mu=0, \sigma=0.02$)

#### **特殊训练策略**
- **Learned Contextual Denoising**：
  - 在上下文中随机注入大量无意义“垃圾 token”，后接目标查询。
  - 强制输入门 $\beta$ 关闭以保护相位状态，显式训练模型在噪声中提取语义。

---

### 📊 评估指标与任务

| 任务 | 指标 | 目的 |
|------|------|------|
| **VRAM 占用测试** | Peak VRAM (GB) | 验证 $O(L)$ 内存扩展性 |
| **生成吞吐量** | tokens/sec | 测试 $O(1)$ 推理效率 |
| **语言建模** | Validation Perplexity (WikiText-103) | 衡量基础语言能力 |
| **零样本推理** | PIQA Accuracy, ARC-Easy Accuracy | 测试逻辑与常识推理 |
| **极端上下文检索** | Targeted Semantic Retrieval @ 1M–2M tokens | 验证超长距离记忆能力 |

---

### 🔁 基线方法对比
- **Transformer Baseline**：标准 $O(L^2)$ attention 模型（作为理论天花板）
- **Pythia-160M**：同规模开源模型，用于 token-level 学习效率对比
- **Llama / Pythia optimized baselines**：使用 FlashAttention 的高效实现，用于吞吐量对比
- **SmolLM-135M**：高度饱和的 Llama 架构模型（训练 600B tokens），作为性能上限参考

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### **1. 内存效率（Table 1 & Figure 2）**
| 序列长度 | Transformer VRAM | CAWN VRAM |
|--------|------------------|----------|
| 1,024 | 2.82 GB | 2.26 GB |
| 2,048 | 3.89 GB | 2.64 GB |
| 4,096 | OOM（8GB GPU） | 3.40 GB |
| 8,192 | — | 4.91 GB |
| 37,831 | — | 7.34 GB |
| **2,000,000** | — | **8.72 GB** ✅ |

> 💡 **结论**：CAWN 成功绕过 $O(L^2)$ KV Cache 瓶颈，VRAM 随序列增长呈线性上升，并在超过 32k 后通过 **chunked prefill + O(1) state caching** 严格稳定在 **8.72 GB**。

---

#### **2. 自回归生成效率（Table 2）**

| Context Length | Llama Baseline | Pythia Baseline | **CAWN-150M** |
|--------------|----------------|------------------|-------------|
| 1,024        | 75.86 tok/s     | 96.81 tok/s       | **51.75 tok/s** |
| 2,048        | 74.18 tok/s     | 96.09 tok/s       | **52.06 tok/s** |
| ...          | ...             | ...               | ...         |
| 16,384       | 75.79 tok/s     | 95.04 tok/s       | **52.38 tok/s** |

> ✅ **关键发现**：尽管当前吞吐略低于成熟 FlashAttention 实现，但 CAWN 的生成速度**完全不受上下文长度影响**，呈现完美的 **flat $O(1)$ 曲线**，证明其状态压缩机制有效。

---

#### **3. 语言建模性能（Figure 3）**

- 在 WikiText-103 上持续单调下降困惑度：
  - Step 244k: 157.18
  - Step 500k: ~95.00
  - Step 752k (~5.4B tokens): **~75.00**
- **对比 Pythia-160M（仅训练 2.1B tokens）**：
  - CAWN 在约 300k 步（~2.1B tokens）时已超越其困惑度（127.85）
- **对比 SmolLM-135M（训练 600B tokens）**：
  - 最终困惑度 18.18，仍优于 CAWN 当前阶段，但差距合理（数据量差百倍）

> ✅ **结论**：CAWN 具备与标准 Transformer 相当甚至更优的**单位 token 学习效率**。

---

#### **4. 零样本推理能力（Table 3）**

| Model | Params | Train Tokens | PIQA Acc | ARC-Easy Acc |
|-------|--------|---------------|----------|--------------|
| Pythia-160M | 160M | 2.1B | 55.50% | 30.64% |
| **CAWN-150M** | **150M** | **~5B** | **60.23%** | **45.45%** |
| SmolLM-135M | 135M | 600B | 68.55% | 61.74% |

> ✅ **结论**：尽管参数更小、训练数据更少，CAWN 已展现出显著更强的**常识与科学推理能力**，证明复数域波干涉能有效捕捉抽象语义。

---

#### **5. 极端上下文检索（Table 4）**

| 距离（tokens） | Red | Blue | Green | VRAM |
|----------------|-----|------|-------|------|
| 650            | √   |      |       | 2.42 GB |
| 19,000         |     |      |       | 3.78 GB |
| 37,800         |     |      |       | 7.34 GB |
| 100,000        | √√√ |      |       | 8.72 GB |
| 1,000,000      |     | √    | √     | 8.72 GB |
| **2,000,000**  | √   | √    | **FAIL** | **8.72 GB** |

> ✅ **成就**：成功在 **2 million tokens** 距离下精确检索目标信息，VRAM 不增。
> ⚠️ **边界现象**：在 2M 处出现选择性失败（“Green” 丢失），推测因特定谐波相位进入**破坏性干涉**或 bfloat16 精度误差累积。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **语言可以被建模为波的共振与干涉**  
   CAWN 实验证明，无需显式的 $O(L^2)$ attention 矩阵，仅通过复数域 phasor 的连续叠加即可实现高效的语法与语义建模。

2. **真正实现了 $O(1)$ 推理状态传递**  
   通过固定大小的 phase state 和 chunked prefill，CAWN 在长达 2M tokens 的上下文中保持恒定 VRAM 与生成延迟。

3. **具备卓越的上下文去噪与长期记忆能力**  
   得益于 Hard-Threshold Gating 和 Frequency-Dependent Retention，模型能在海量噪声中精准定位目标信息。

4. **学习效率优于同级 Transformer**  
   在更少训练步数和参数下，CAWN 在 PIQA 和 ARC-Easy 上表现更优，表明其架构具有更高的表示效率。

---

### ⚠️ 局限性

1. **硬件依赖性强**  
   性能高度依赖自定义 Triton kernel 和 float32 复数运算，通用性受限。

2. **极端长度下的相位干扰风险**  
   在 2M tokens 出现选择性失败，提示存在**谐波周期性冲突**或**精度极限**问题。

3. **当前吞吐较低**  
   由于缺乏硬件级优化（如 FlashAttention），生成速度目前落后于主流架构。

4. **尚未扩展至十亿级以上参数**  
   当前仅验证了 150M 规模，更大模型的表现待验证。

---

### 🔮 未来工作方向

1. **向 1B–7B 参数规模扩展**  
   验证 CAWN 在大规模下的可扩展性和性能潜力。

2. **迁移至万亿 token 级训练数据**  
   使用更大语料进一步逼近 SmolLM 等饱和模型的性能上限。

3. **深入研究相位边界失效机制**  
   进行谱分析，绘制 amplitude tracker 轨迹，定位破坏性干涉的具体频率与时间点。

4. **探索更高精度格式（如 float64）或动态缩放机制**  
   解决长程数值漂移问题。

5. **开发专用推理引擎**  
   实现端到端 $O(1)$ 生成部署，充分发挥其无限上下文优势。

---

## ✅ 总结

**CAWN** 是一项颠覆性的尝试，它挑战了“必须用 attention 建模语言”的范式，提出了一种**基于复数域声波共振**的全新序列建模范式。其实验证明：

- 它能以 $O(L)$ 训练成本和 $O(1)$ 推理开销，处理长达 **2 million tokens** 的上下文；
- 在语言建模和推理任务上表现出色，学习效率优于同级 Transformer；
- 其物理直觉强、可解释性高，为下一代无限上下文 LLM 提供了全新路径。

> 🌟 **一句话总结**：  
> **CAWN 用“声波干涉”代替“注意力矩阵”，让语言模型在 200 万 token 中依然“记得你最初说了什么”。**

</details>

---

### 5. [Structured Causal Video Reasoning via Multi-Objective Alignment](https://arxiv.org/abs/2604.04415)

**Authors**: Zinuo Li, Yongxin Guo, Jun Liu, Jiawei Zhan, Xi Jiang, Chengjie Wang, Mohammed Bennamoun, Farid Boussaid, Feng Zheng, Qiuhong Ke  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.04415v1  

#### Abstract
Human understanding of video dynamics is typically grounded in a structured mental representation of entities, actions, and temporal relations, rather than relying solely on immediate deductive reasoning. In contrast, existing Video-LLMs largely depend on unstructured video reasoning, where critical...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Structured Causal Video Reasoning via Multi-Objective Alignment**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前的 **Video-LLMs** 在视频理解中普遍采用非结构化的 **Chain-of-Thought (CoT)** 推理方式，导致以下问题：
- **冗长且低效**：推理过程包含大量无关视觉线索，淹没关键证据。
- **因果建模弱**：难以捕捉事件间的时序因果关系，推理易漂移（reasoning drift）。
- **可解释性差**：中间推理步骤缺乏明确约束，难以验证。

这与人类通过构建**结构化心智表征**（实体、动作、时间关系）进行动态理解的认知机制相悖。

---

### **提出的新方法与思路**
作者提出 **“Structure-First” 范式**，核心是引入 **Structured Event Facts** 作为推理前的显式先验：

#### **(1) Structured Event Facts**
在正式推理前，模型首先从视频中提取高密度的结构化事实，包括：
- 时间区间 `[time]`
- 人物 `[person]`
- 动作 `[human_action]`
- 场景 `[scene]`
- 物体 `[object]`
- 镜头 `[camera]`
- 因果事件描述 `[casual_event_caption]`

这些结构化事实为后续推理提供**紧凑、可验证、因果锚定**的基础。

#### **(2) 四阶段训练流程**
为有效训练模型生成并利用结构化事实，设计了渐进式训练流程：
1. **Stage 1: Facts Training**  
   训练模型准确输出结构化事实。
2. **Stage 1.5: Format Warm-Start**  
   引入 `<thinking>` 和 `<answering>` 标签格式，预热结构化输出。
3. **Stage 2: Thinking Warm-Start**  
   基于事实进行结构化因果推理训练。
4. **Stage 3: RL-based Post-training**  
   使用强化学习对齐多目标优化。

#### **(3) Pareto-Frontier guided Advantage Balancing (P-FAB)**
在 RL 阶段，传统 **GRPO** 难以平衡多个冲突目标（如事实完整性 vs. 推理长度）。为此提出 **P-FAB** 算法：
- 将多目标奖励向量视为独立信号。
- 借鉴 **Multiple Gradient Descent Algorithm (MGDA)**，求解最小范数组合，逼近 **Pareto-Frontier**。
- 动态平衡不同目标，避免稀有但关键信号被掩盖。

#### **(4) CausalFact-60K 数据集**
构建了一个包含 60K 条样本的数据集，专用于训练结构化因果推理，涵盖高质量视频标注与因果思维链。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **推理结构** | 非结构化 CoT，冗长易漂移 | 结构化事实引导，简洁、聚焦 |
| **因果建模** | 弱，依赖帧检索 | 显式因果验证（Antecedent → Action → Consequence） |
| **可解释性** | 黑箱推理 | 中间事实可验证 |
| **优化策略** | 单一标量奖励，易失衡 | 多目标动态平衡（P-FAB） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

#### **视频时序定位（Temporal Grounding）**
- **Charades-TimeLens**：高精度室内活动标注，强调细粒度时间边界。
- **ActivityNet-TimeLens**：重新标注的 ActivityNet-Captions，修正松散边界。
- **ActivityNet-Captions**：大规模开放域视频，用于密集描述与定位。

#### **通用视频理解（General Understanding）**
- **VideoMME**：长视频理解基准，覆盖电影、体育等。
- **MLVU**：长视频综合任务，含主题推理（TR）、第一人称理解（Ego）。
- **ETBench**：细粒度时间敏感任务，如事件匹配、因果问答。
- **NExT-GQA**：基于视频片段的因果与时间问答。

---

### **实验设置与评估指标**

| 设置项 | 描述 |
|-------|------|
| **基础模型** | Qwen3-VL-4B-Instruct |
| **训练阶段** | 四阶段：Facts → Format Warm-up → Thinking → RL (P-FAB) |
| **RL 奖励** | 四维奖励：<br>- `Format`（结构合规）<br>- `Linear IoU`（定位精度）<br>- `Accuracy`（答案正确率）<br>- `Length`（长度效率） |
| **评估指标** | - `R1@IoU`（0.3/0.5/0.7）<br>- `Accuracy` / `F1` / `Recall`<br>- `TR Acc`, `Ego Acc` |

---

### **基线方法对比**
- **闭源模型**：GPT-4o, GPT-5, Gemini-2.5-Pro
- **开源模型**：
  - Qwen3-VL-4B-Instruct
  - Qwen3-VL-4B-Thinking
  - Time-R1-7B, TRACE-7B, VideoChat-R1-7B
- **消融变体**：
  - w/o Facts
  - w/o Thinking
  - w/o RL
  - GRPO 替代 P-FAB

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2 & 3）**

| 模型 | ActivityNet R1@0.5 | Charades R1@0.7 | VideoMME Acc | NExT-GQA Acc |
|------|---------------------|------------------|---------------|---------------|
| Qwen3-VL-4B-Instruct | 35.8 | 18.4 | 63.9 | 72.1 |
| Qwen3-VL-4B-Thinking | 31.7 | 17.8 | 63.1 | 66.6 |
| **Factum-4B (本文)** | **48.4** | **21.6** | **64.7** | **73.6** |

> ✅ **显著提升**：在 ActivityNet 上 R1@0.5 提升 **12.6%**，R1@0.7 提升 **3.2%**，显示更强的时间边界对齐能力。

---

### **与基线方法的对比结果**
- **超越 7B 模型**：尽管仅 4B 参数，Factum-4B 在多个任务上优于 Time-R1-7B 和 VideoChat-R1-7B。
- **媲美闭源系统**：
  - 在 **ETBench** 的 TVG 和 TEM 任务上，**超过 GPT-4o**。
  - 在 **MLVU** 上达到与 GPT-4o 相当水平。
- **推理更可靠**：相比 Thinking 模型，**未出现性能下降**，说明结构化推理避免了“推理税”。

---

### **消融实验结果（Table 1）**

| 变体 | ActivityNet R1@0.5 | VideoMME Acc | 说明 |
|------|---------------------|---------------|------|
| w/o Facts | 41.6 | 60.8 | 性能下降明显，说明结构化事实至关重要 |
| w/o Thinking | 40.4 | 58.5 | 缺乏因果推理桥接，事实无法有效利用 |
| w/o RL | 41.6 | 59.1 | RL 后训练带来显著增益（+6.8%） |
| GRPO (G=8) | 45.2 | 63.2 | P-FAB 更优 |
| **P-FAB (G=8)** | **45.7** | **63.5** | 显示多目标平衡优势 |

> 🔍 **关键发现**：P-FAB 在更大的生成组（G=8）下优势更明显，说明其在复杂多目标场景中更具鲁棒性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **结构化先验优于非结构化推理**  
   引入 **Structured Event Facts** 显著提升了推理的准确性、简洁性和可解释性。

2. **因果验证机制增强时序理解**  
   通过 `Global Search → Causal Verification → Final Alignment` 流程，模型能更好地捕捉事件间的前后依赖。

3. **P-FAB 有效解决多目标冲突**  
   相比标准 GRPO，P-FAB 能动态平衡格式、精度、长度等目标，尤其在稀有但关键信号上表现更好。

4. **小模型也能实现强性能**  
   **Factum-4B** 仅 4B 参数，在多项任务上超越更大模型，证明**结构化推理 + 多目标优化**的价值。

---

### **局限性**
- **数据规模限制**：当前训练数据仍有限，需进一步扩展以覆盖更多场景。
- **依赖高质量标注**：Structured Event Facts 的生成依赖精确的时间标注，对低质量数据泛化能力未知。
- **计算成本较高**：四阶段训练流程复杂，尤其是 RL 阶段需要多次采样。

---

### **未来工作方向**
- 扩展 **CausalFact** 数据集至更多领域（如医疗、教育视频）。
- 探索 **自监督/弱监督** 方式生成结构化事实，降低标注依赖。
- 将 P-FAB 应用于其他多模态任务（如图文推理、机器人决策）。
- 构建端到端的结构化推理框架，减少训练阶段割裂。

---

> **总结**：本文提出了一个认知启发的 **结构化因果视频推理框架**，通过 **Structured Event Facts + P-FAB 多目标优化**，实现了更可靠、可解释、高性能的视频理解，为未来 **evidence-grounded 视频推理系统** 提供了新方向。

</details>

---

### 6. [Communication-Efficient Collaborative LLM Inference over LEO Satellite Networks](https://arxiv.org/abs/2604.04654)

**Authors**: Songge Zhang (Sherman), Wen Wu (Sherman), Liang Li (Sherman), Ye Wang (Sherman),  Xuemin (Sherman),  Shen  
**Category**: cs.DC  
**Published**: 2026-04-07  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.04654v1  

#### Abstract
Low Earth orbit (LEO) satellites play an essential role in intelligent Earth observation by leveraging artificial intelligence models. However, limited onboard memory and excessive inference delay prevent the practical deployment of large language models (LLMs) on a single satellite. In this paper, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Communication-Efficient Collaborative LLM Inference over LEO Satellite Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **LEO 卫星网络**中部署 **Large Language Models (LLMs)** 面临两大挑战：
- **Onboard memory constraints**：单颗卫星内存有限，无法容纳完整的 LLM。
- **High inference delay**：模型推理过程计算密集，且跨卫星通信开销大。

传统方法如模型压缩（pruning、quantization）虽能减小模型体积，但会牺牲 **inference accuracy**。而现有的 **model splitting** 多为两段式分割，难以适应多卫星协作场景下的资源异构性和通信瓶颈。

---

### 🚀 提出的新方法与创新思路

本文提出了一种 **通信高效的协同 LLM 推理方案**，核心创新如下：

#### （1）**多阶段模型分割 + 协同推理架构**
- 将完整 LLM 分割为多个子模型（sub-models），分别部署在不同 LEO 卫星上。
- 通过 **inter-satellite links (ISLs)** 传递中间激活值（intermediate activations），实现端到端协同推理。
- 支持 **K-stage 多段分割**，适用于异构卫星网络。

#### （2）**Pipeline 并行机制**
- 引入 **pipeline parallelism**，使各卫星在传输当前 batch 激活的同时，开始处理下一个 batch 的计算任务。
- 显著减少空闲等待时间，提升资源利用率和吞吐量。

#### （3）**自适应激活压缩方案（Adaptive Activation Compression）**
- 设计基于 **Gumbel-mask 的可学习稀疏化模块**，动态选择重要激活特征。
- 结合 **量化（quantization）** 和 **熵编码（entropy coding）** 进一步压缩传输数据。
- 压缩策略是可训练的，能保留对任务敏感的信息，避免 Top-k 等固定策略导致的关键信息丢失。

#### （4）**联合优化框架 + 图搜索算法**
- 构建一个 **混合整数非线性规划（MINLP）问题**，联合优化：
  - 模型分层策略（layer assignment）
  - 各阶段激活压缩比（compression ratios）
- 目标是最小化总推理延迟，同时满足：
  - 卫星内存限制（onboard memory）
  - 推理精度下限（inference accuracy）
- 将该问题转化为 **有向无环图（DAG）上的最短路径搜索问题**。
- 提出一种 **改进的 A\*-based 搜索算法**，结合内外层迭代优化（outer layer 搜索分割策略，inner layer 优化压缩比）。

---

### 🔍 相比现有方法的优势

| 方面 | 本文方法优势 |
|------|---------------|
| **部署可行性** | 支持超大规模 LLM 在资源受限卫星上的部署 |
| **通信效率** | 自适应压缩显著降低 ISL 通信开销 |
| **推理速度** | Pipeline 并行 + 最优分层策略大幅缩短端到端延迟 |
| **精度保持** | Learnable 压缩机制优于固定阈值或 Top-k 方法，精度损失 <1% |
| **系统灵活性** | 支持异构卫星环境下的动态资源调度 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **EuroSAT**：27,000 张遥感图像，10 类地物分类，分辨率 64×64。
- **RESISC45**：31,500 张遥感图像，45 类场景分类，分辨率 256×256。

> 用于模拟地球观测中的视觉理解任务。

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| **卫星数量** | 5 颗 LEO 卫星（Walker Delta 星座） |
| **轨道高度** | 500 km，倾角 53° |
| **计算设备** | 4× NVIDIA Jetson AGX Orin（模拟不同功耗模式：15W/30W/50W） |
| **地面服务器** | NVIDIA RTX 4070 Ti GPU |
| **通信链路** | <br>- **ISL（星间链路）**：FSO 光学链路，速率 0.5 Gbps<br>- **S2G（星地链路）**：Ka 波段，速率最高 6 Gbps |
| **批大小（batch size）** | 64 |
| **测试模型** | Vision Transformers：<br>- ViT-B (0.086B)<br>- ViT-L (0.307B)<br>- ViT-H (0.632B)<br>- ViT-G (1.8B) |

---

### 📊 评估指标

| 指标 | 定义 |
|------|------|
| **Inference Latency** | 从任务启动到结果传回地面站的总时间 |
| **Communication Overhead** | 整个推理过程中传输的数据总量（含 ISL 和 S2G） |
| **Inference Accuracy** | 分类任务准确率，衡量压缩与分割对性能的影响 |
| **Optimization Gain** | 所提优化算法相比启发式策略的加速比 |

---

### 🆚 基线方法对比

| 基线方法 | 描述 |
|--------|------|
| **Ground-only** | 卫星仅采集原始图像并下传，由地面服务器执行全模型推理 |
| **Single-satellite** | 图像发送至某一颗具备算力的卫星，本地完成全部推理 |
| **Heuristic Splitting** | 按照计算能力比例分配层数，不进行联合优化 |
| **Uniform Partition** | 层均匀分配给各卫星 |
| **Top-k Sparsification** | 固定选取前 k 个最大激活值进行传输，作为压缩对照 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 性能维度 | 本文方法表现 |
|---------|-------------|
| **推理延迟降低** | 最高 **减少 42%**（vs. Ground-only） |
| **通信开销降低** | 最高 **减少 71.7%**（vs. Ground-only） |
| **精度损失控制** | 均小于 **1%**，部分场景甚至优于基线 |
| **加速比** | 达到 **1.58× ~ 2.86×** 不等，取决于模型规模和链路条件 |

---

### 🔁 与基线方法对比结果

#### （1）**推理延迟对比（Fig. 3–6）**
- 在所有分辨率（240p–1080p）和 S2G 速率（0.2–0.8 Gbps）下，所提方法均取得最低延迟。
- 在 1080p 下，延迟比 Ground-only 低 **58%**，比 Single-satellite 低 **46%**。
- 当 S2G 速率为 0.8 Gbps 时，仍比 Ground-only 快 **42%**，说明其优势不仅依赖于地面链路瓶颈。

#### （2）**通信开销对比（Fig. 7）**
- 所提方法通信开销比 Ground-only 低 **71.7%**。
- 虽略高于 Single-satellite（因其无需 ISL 通信），但后者推理延迟极高，尤其在大模型下不可行。

#### （3）**不同模型规模下的表现（Fig. 5）**
- 对 **ViT-G（1.8B）** 等大型模型效果最显著，因单星无法承载。
- 对小型模型（如 ViT-B），Single-satellite 更高效，表明本方法更适合 **billion-scale LLMs**。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）**压缩组件逐级分析（Fig. 8）**
- **Sparsification（Gumbel-mask）**：平均压缩比 **3.96×**
- **Quantization（8-bit）**：进一步提升至 **11.56×**
- **Lossless Coding（熵编码）**：最终达 **25.82×**
- 总体压缩超过 **25×**，验证了多级压缩的有效性。

#### （2）**不同压缩方法精度对比（Table IV & V）**

| 方法 | ViT-Large + EuroSAT | ViT-Large + RESISC45 |
|------|---------------------|------------------------|
| Baseline（无压缩） | 98.36% | 96.67% |
| **GumbelMask（本文）** | **98.31%**（↓0.05%） | **95.97%**（↓0.7%） |
| Top-k | 98.30%（↓0.06%） | 95.54%（↓1.13%） |

> → GumbelMask 在保持更高精度的同时实现更强压缩，尤其在复杂数据集（RESISC45）上优势明显。

#### （3）**分层位置鲁棒性测试（Fig. 10）**
- 在 200 种不同的模型切分点下测试验证精度。
- **超过 194 次精度下降 <1%**，证明所提压缩机制对分层位置不敏感，具有强鲁棒性。

#### （4）**优化算法有效性（Fig. 12）**
- 相比 Heuristic 和 Uniform 策略，所提 A\*-based 算法将总延迟降低 **103%**。
- 表明联合优化在异构环境中至关重要。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **协同推理 + Pipeline 并行可有效缓解 LEO 卫星资源瓶颈**，支持 billion-scale LLM 部署。
2. **自适应激活压缩（GumbelMask + Quantization + Entropy Coding）** 可在几乎不影响精度的前提下（<1% loss），实现高达 **25× 的通信压缩**。
3. **联合优化模型分层与压缩比** 是提升系统性能的关键，简单启发式策略远不如所提图搜索算法。
4. 所提方案在 **高分辨率输入、低带宽 S2G 场景下优势更显著**，适合未来智能遥感应用。

---

### ⚠️ 方法的局限性

1. **训练开销增加**：Gumbel-mask 模块需离线训练，增加了前期准备成本（但不影响在线推理）。
2. **依赖稳定 ISL**：假设星间链路稳定，未考虑动态拓扑变化或链路中断情况。
3. **未覆盖多轨道协同**：目前仅限于单轨道平面内的卫星协作，尚未扩展至 MEO/GEO 或跨轨道网络。
4. **硬件仿真为主**：实验基于 Jetson Orin 模拟，尚未在真实太空环境中验证。

---

### 🔮 未来工作方向

1. **扩展至 multi-orbit satellite constellations**，研究跨轨道协同推理机制。
2. **引入容错机制**，应对卫星失效或链路中断等异常情况。
3. **支持 split fine-tuning**，实现在轨模型更新与个性化适配。
4. **探索更轻量化的可学习压缩模块**，降低训练与部署门槛。

---

> 💡 **总体评价**：  
> 本文首次系统性地提出了面向 LEO 卫星网络的大模型协同推理框架，融合了 **model splitting、pipeline parallelism、adaptive activation compression 和联合优化算法**，为未来 **spaceborne AI** 和 **6G space-air-ground integrated networks** 提供了重要的技术路径。

</details>

---

### 7. [AdaHOP: Fast and Accurate Low-Precision Training via Outlier-Pattern-Aware Rotation](https://arxiv.org/abs/2604.02525)

**Authors**: Seonggon Kim, Alireza Khodamoradi, Kristof Denolf, Eunhyeok Park  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.02525v1  

#### Abstract
Low-precision training (LPT) commonly employs Hadamard transforms to suppress outliers and mitigate quantization error in large language models (LLMs). However, prior methods apply a fixed transform uniformly, despite substantial variation in outlier structures across tensors. Through the first syst...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AdaHOP: Fast and Accurate Low-Precision Training via Outlier-Pattern-Aware Rotation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
低精度训练（**Low-Precision Training, LPT**）在提升大语言模型（**LLMs**）训练效率方面具有巨大潜力，但面临一个核心挑战：**outliers**（异常值）。这些稀疏但极端的数值会显著放大量化误差，导致训练不稳定和模型质量下降。

现有方法（如Hadamard变换）通常采用**统一策略**（uniform transform），即对所有层和计算路径应用相同的变换方式（如固定使用IHT或OHT），忽略了不同张量中outlier结构的多样性。这种“一刀切”的策略在某些情况下不仅无效，甚至可能加剧量化误差。

### 提出了什么新方法或新思路
本文提出 **AdaHOP**（**Adaptive Hadamard transform with Outlier-Pattern-aware strategy**），其核心创新在于：

- **系统性分析了LLM中的outlier模式**：首次通过大规模实证研究，将权重（weight）、激活（activation）和梯度（gradient）中的outlier结构归纳为三种稳定且可预测的模式：
  - **Row-wise (R)**：异常值集中在少数行。
  - **Column-wise (C)**：异常值集中在少数列。
  - **None (N)**：无明显集中趋势。
  
- **提出了基于模式对（pattern pair）的自适应策略**：AdaHOP认识到Hadamard变换的有效性取决于其平滑方向是否与operand的outlier方向正交。因此，它为每一对矩阵乘法（如`Gy @ X`）的输入张量（A, B）动态选择最优策略：
  - 若模式对适合内维平滑（如CN、NN），则使用 **Inner Hadamard Transform (IHT)**。
  - 若IHT无效（如RN、RC、CC），则结合 **Selective Outlier Extraction (OE)**，将主导outliers路由到高精度路径（BF16），其余部分用IHT处理。

- **硬件感知实现**：设计了融合的Triton内核，在AMD CDNA4架构上高效执行模式检测、IHT、OE和混合精度GEMM，最小化开销。

### 相比现有方法的优势
- **更优的准确性**：相比固定变换方法（如MXFP4+Hadamard、HALO），AdaHOP能更精准地抑制量化误差，达到接近BF16的训练质量。
- **更高的效率**：相比需要额外全尺寸变换的OHT（如HALO），AdaHOP的OE+IHT设计计算和内存开销更低，实现了更高的吞吐量。
- **更强的鲁棒性**：通过一次性的校准阶段即可捕获稳定的outlier模式，无需运行时检测，策略轻量且可靠。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要训练数据集：**C4 (Colossal Clean Crawled Corpus)**。
- 下游零样本评估任务：
  - **PIQA**（物理常识推理）
  - **HellaSwag**（情境补全）
  - **ARC-Easy (ARC-E)**（科学问答）
  - **LAMBADA**（长距离依赖词预测）

### 实验设置和评估指标
- **模型规模**：在四种不同规模的LLM上进行验证：
  - Llama3.2-1B
  - Llama3.2-3B
  - Instella-3B
  - Llama3.1-8B
- **训练配置**：
  - 序列长度：4096 tokens
  - 批大小：128
  - 优化器：AdamW
  - 学习率：4e-4
  - 训练步数按Chinchilla定律缩放（1B模型：40B tokens；8B模型：160B tokens）
- **评估指标**：
  - **训练损失**（Training Loss）及其与BF16的差距
  - **下游任务零样本准确率**（Zero-shot Accuracy）
  - **内存消耗**（Memory Consumption）
  - **训练吞吐量**（Throughput, tok/s）
  - **内核级延迟与加速比**（Kernel Latency & Speedup）

### 基线方法对比
- **BF16**：全精度训练，作为性能上限。
- **Naive MXFP4**：直接量化，无任何outlier抑制。
- **MXFP4+Hadamard**：在所有层统一应用IHT。
- **Tseng et al.**：使用随机Hadamard进行无偏梯度估计。
- **HALO**：在梯度路径上应用OHT以应对row-wise梯度和column-wise激活。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **训练质量**：
  - AdaHOP在所有模型上均实现了**最低的训练损失差距**（< 0.01 vs BF16），显著优于其他MXFP4方法（见图1）。
  - 在**下游任务平均准确率**上，AdaHOP-Lv2在多个模型上达到或超过BF16水平：
    - **Instella-3B**：AdaHOP-Lv2达 **56.05%**，超越BF16（55.75%）和所有基线。
    - **Llama3.1-8B**：AdaHOP-Lv2达 **61.43%**，接近BF16（61.68%），远超HALO（60.37%）。

- **效率指标**：
  - **内存压缩**：AdaHOP-Lv1实现**3.6×内存压缩**（从76.00GB降至20.94GB）。
  - **内核加速**：在典型矩阵乘法上，AdaHOP实现**1.59–1.80×的GEMM内核加速**（vs BF16，见表5）。
  - **端到端吞吐**：AdaHOP-Lv1吞吐达 **13,247 tok/s**，略高于BF16（12,946 tok/s），显著优于HALO（10,482 tok/s）。

### 与基线方法的对比结果
| 方法 | 内存 (GB) | 吞吐 (tok/s) | 质量评级 |
|------|----------|-------------|---------|
| BF16 | 76.00 | 12,946 | ○ |
| MXFP4+Hadamard | 20.60 | 14,312 | △ |
| HALO | 20.60 | 10,482 | △ |
| **AdaHOP-Lv1** | **20.94** | **13,247** | **○** |
| **AdaHOP-Lv2** | **28.04** | **13,134** | **○** |

> 注：质量评级 ○ 表示与BF16差距<1%，△ 表示差距<3%，× 表示严重退化。

- **关键对比**：
  - **vs HALO**：AdaHOP在保持更高吞吐的同时，实现了更好的训练质量和下游性能。HALO因OHT引入的额外FWHT操作导致严重性能瓶颈。
  - **vs Tseng et al.**：AdaHOP通过模式感知策略，在相同精度下获得了更低的量化误差和更高的准确率。

### 消融实验结果
- **模式稳定性**：实验证明outlier模式在训练早期即稳定（见图4），支持一次性校准的有效性。
- **OE的重要性**：对于RN、RC等IHT无效的模式对，OE是降低误差的关键。移除OE会导致性能显著下降。
- **Lv1 vs Lv2**：
  - **AdaHOP-Lv1**：对CC模式使用OE-Right + IHT，平衡效率与精度。
  - **AdaHOP-Lv2**：对CC模式（多出现在Key/Value投影层）完全使用BF16，进一步提升敏感层的保真度。
  - 结果显示，从Lv1升级到Lv2可带来**最高达0.8个百分点的准确率提升**（如Instella-3B），证明了对attention-critical层保留高精度的价值。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Outlier模式是结构性和稳定的**：LLM中的outlier并非随机噪声，而是遵循**Row-wise、Column-wise、None**三种可预测的模式，且在训练过程中高度稳定。
2. **统一变换策略是次优的**：Hadamard变换的效果强烈依赖于其方向与outlier结构的对齐程度。盲目应用固定变换（如IHT或OHT）无法在所有计算路径上有效。
3. **自适应策略是更优解**：通过校准识别模式对，并据此选择IHT或OE+IHT，可以在极低开销下实现接近全精度的训练质量。
4. **硬件协同设计至关重要**：利用现代加速器（如AMD CDNA4）的混合精度并行能力，可以高效实现OE等复杂策略，避免传统方法（如OHT）的性能陷阱。

### 方法的局限性
- **依赖固定Hadamard矩阵**：当前使用预定义的Walsh-Hadamard矩阵，未探索学习型旋转矩阵（learned rotation）的潜力。
- **模式分析范围有限**：目前仅在Llama-family和Instella模型上验证，尚未扩展到MoE（如Mixtral）或不同归一化机制的模型。
- **固定提取数量**：OE中提取的outlier行/列数（k=64）是全局固定的，未根据各层的outlier严重程度进行自适应调整。

### 未来工作方向
1. **结合学习型旋转**：将AdaHOP与SpinQuant等学习旋转的方法结合，实现数据驱动的最优变换。
2. **扩展模型架构**：在更多类型的LLM（如Mixtral、Gemma）上验证outlier模式的普适性。
3. **支持更多量化格式**：将AdaHOP框架推广到其他低精度格式（如FP8、INT4）。
4. **自适应k值选择**：研究基于每层outlier强度的动态k值选择策略，进一步优化精度-效率权衡。

> **总结**：AdaHOP通过揭示并利用LLM中outlier的结构性和稳定性，提出了一种高效、准确的自适应低精度训练框架，为实现真正实用的大模型低精度训练提供了新范式。

</details>

---

### 8. [Fast NF4 Dequantization Kernels for Large Language Model Inference](https://arxiv.org/abs/2604.02556)

**Authors**: Xiangbo Qi, Chaoyi Jiang, Murali Annavaram  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.02556v1  

#### Abstract
Large language models (LLMs) have grown beyond the memory capacity of single GPU devices, necessitating quantization techniques for practical deployment. While NF4 (4-bit NormalFloat) quantization enables 4$\times$ memory reduction, inference on current NVIDIA GPUs (e.g., Ampere A100) requires expen...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast NF4 Dequantization Kernels for Large Language Model Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大型语言模型（LLMs）参数量已远超单个GPU内存容量，因此广泛采用 **NF4（4-bit NormalFloat）量化** 技术以实现4倍内存压缩。然而，NVIDIA Ampere架构（如A100）不支持原生4-bit计算，推理时必须将NF4权重**反量化为FP16格式**，这一过程涉及大量全局内存（global memory）访问和复杂的索引逻辑，导致显著的性能瓶颈。

论文通过系统性分析发现，在Qwen3-32B等模型中，**dequantization 占据端到端延迟的21–40%**，成为制约推理效率的关键环节。

### 🚀 提出的新方法与创新思路
作者提出一种**轻量级共享内存优化策略**，从GPU内存层次结构出发，针对性地解决NF4反量化中的两大瓶颈：

1. **共享内存加载策略（Shared Memory LUT Caching）**  
   将仅需64字节的16元素NF4查找表（LUT）由单个线程一次性加载至**shared memory**，供整个thread block复用，避免每个线程重复从高延迟的global memory读取。

2. **简化索引计算（Simplified Index Computation）**  
   替代原有基于4层条件分支树的复杂索引方式，改用位操作（bit masking and shifting）直接提取4-bit index，消除warp divergence并大幅减少指令数。

> 🔑 核心洞察：利用shared memory的低延迟（~19 cycles）相比global memory（~290 cycles）高达12–15×的速度优势，并结合数据重用特性，实现高效反量化。

### ⚖️ 相比现有方法的优势
| 维度 | 本工作 | 现有方案（如BitsAndBytes） |
|------|--------|-----------------------------|
| 内存访问模式 | 利用shared memory广播机制，减少LUT流量64× | 每线程独立访问global memory |
| 指令开销 | 2条无分支指令完成indexing | 多达7条带条件分支的指令 |
| 兼容性 | 完全兼容HuggingFace + BitsAndBytes生态 | 同样兼容，但未优化底层kernel |
| 部署成本 | 零模型转换、零离线预处理，即插即用 | 可能需要kernel融合或特殊编译 |

✅ **优势总结**：
- 轻量级设计（仅增加64 bytes shared memory/thread block）
- 显著加速且无需修改模型或训练流程
- 可无缝集成进现有生产系统

---

## 2. 核心实验方法和设置

### 📚 数据集
- **GSM8K**：用于生成输入prompt，测试不同长度序列下的推理表现
- 所有实验均在token级别进行准确性验证，确保输出bit-exact一致

### ⚙️ 实验设置
| 项目 | 配置 |
|------|------|
| 硬件平台 | 单块 **NVIDIA A100-80GB GPU**<br>CPU: AMD EPYC 7513 (32核), RAM: 64GB |
| 软件环境 | CUDA 12.6, PyTorch 2.1, BitsAndBytes 0.47.0 |
| GPU频率控制 | 使用`nvidia-smi`锁定最大频率（1410 MHz），排除动态调频影响 |
| 测量工具 | PyTorch Profiler + CUPTI接口，微秒级精度，包含kernel launch overhead和memory transfer |
| 批次大小（Batch Size） | 2, 4, 8, 16, 32, 64 |
| 模型范围 | **Gemma 27B**, **Qwen3 32B**, **Llama3.3 70B** |

### 📊 评估指标
- **端到端延迟（End-to-end Latency）**：从输入到生成完整响应的时间
- **吞吐量（Throughput）**：tokens per second
- **Kernel级延迟**：NF4 dequantization kernel执行时间
- **Speedup ratio**：相对于baseline的加速比

### 🔁 基线方法对比
- **Baseline**: 开源实现 [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) 中的标准NF4反量化kernel
- **Optimized**: 本文提出的shared memory + direct indexing优化版本
- 对比保持API完全兼容，仅替换底层kernel函数 `kDequantizeBlockwise`

---

## 3. 主要实验结果和性能指标

### 📈 Kernel级性能提升（Table II）
在所有模型和batch size下，**反量化kernel实现2.0–2.2×加速**：

| Batch Size | Gemma 27B | Qwen3 32B | Llama3.3 70B |
|------------|-----------|-----------|--------------|
| 2          | 2.10×     | 2.20×     | 2.04×        |
| 4          | 2.10×     | 2.19×     | 2.04×        |
| 8          | 2.11×     | 2.19×     | 2.04×        |
| 16         | 2.10×     | 2.19×     | 2.03×        |
| 32         | 2.11×     | 2.19×     | 2.05×        |
| 64         | 2.08×     | 2.15×     | 2.03×        |
| **Average** | **2.10×** | **2.19×** | **2.04×**    |

> 💡 加速一致性表明该优化针对的是**通用内存瓶颈**，而非特定模型结构。

### 🚀 端到端性能提升（Figure 4）
| 模型 | 平均加速比 | 最高加速比 |
|------|------------|------------|
| **Llama3.3 70B** | 1.52× | **1.54×** (@ batch 2) |
| **Qwen3 32B**    | 1.18× | 1.29× (@ batch 32) |
| **Gemma 27B**    | 1.10× | 1.32× (@ batch 64) |

> 📌 更大模型收益更高 → 因其更深层数导致更多dequantization调用，优化效果更明显。

### 📦 吞吐量提升
- **Llama3.3 70B @ batch 2**: 从 ~250 → ~385 tokens/s (**+54%**)
- **Qwen3 32B @ batch 32**: 283 → 368 tokens/s (**+30%**)
- **Gemma 27B @ batch 64**: 506 → 633 tokens/s (**+25%**)

### 🔍 消融分析（隐含于文中）
虽然未明确列出消融实验表格，但从设计可分解为两个正交优化组件：

| 组件 | 性能增益来源 |
|------|-------------|
| **Shared Memory LUT Cache** | 减少global memory访问，利用12–15× latency优势 |
| **Direct Indexing (Bit Manipulation)** | 指令数从7降至2，消除branch divergence，提升warp利用率 |

二者协同作用，共同促成整体加速。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Dequantization是当前NF4推理的主要瓶颈**，尤其在大模型中占比可达30–40%，远高于其计算复杂度应有的开销。
2. **内存访问模式决定性能上限**：当前实现受限于频繁的global memory lookup，而非算力不足。
3. **轻量级优化也能带来巨大收益**：仅使用64 bytes shared memory + 极简逻辑重构，即可实现2倍以上kernel加速。
4. **优化效果随模型规模增大而增强**：更大模型因更多layer和weight matrix，对dequantization优化更敏感。

### ⚠️ 局限性
- 当前优化聚焦于**weight-only dequantization**，未考虑activation量化或KV Cache量化场景。
- 依赖GPU shared memory资源，极端情况下可能与其他kernel竞争有限容量（但在Ampere/A100上影响极小）。
- 仅适用于静态LUT结构的量化方案（如NF4），对动态编码方案适配性待验证。

### 🔮 未来工作方向
1. **扩展至其他量化格式**：如FP4、INT4等，探索统一的高速反量化框架。
2. **结合kernel fusion**：将dequantization与MatMul进一步融合，减少中间数据搬移。
3. **支持多GPU分布式场景**：研究跨设备共享LUT缓存的可能性。
4. **适配新一代GPU架构**（如Hopper、Blackwell）：利用更新的memory hierarchy特性进一步压榨性能。

---

## ✅ 总结
本文提出了一种简单而高效的NF4反量化优化方案，通过**共享内存缓存LUT + 位运算简化索引**，实现了**2.0–2.2× kernel加速** 和 **最高1.54×端到端推理加速**，且完全兼容HuggingFace生态系统。该工作证明了：即使在高度优化的现代ML系统中，**对底层内存访问模式的深入理解仍能带来显著性能突破**，为LLM高效部署提供了实用、即插即用的解决方案。

</details>

---

### 9. [STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation](https://arxiv.org/abs/2604.02756)

**Authors**: Zijin Liu, Xu Geng, Wenshuai Xu, Xiang Zhao, Yan Xia, You Song  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.02756v1  

#### Abstract
Accurate crowd simulation is crucial for public safety management, emergency evacuation planning, and intelligent transportation systems. However, existing methods, which typically model crowds as a collection of independent individual trajectories, are limited in their ability to capture macroscopi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有 crowd simulation 方法存在以下局限：
- **微观建模主导**：主流方法（如 SFM、data-driven 模型）通常将人群视为独立个体轨迹的集合，忽略了宏观物理规律（如质量守恒），导致长期模拟中误差累积、稳定性差。
- **缺乏物理一致性**：纯 data-driven 方法（如 STGCNN、MID）虽能捕捉复杂行为模式，但常产生违反物理规律的行为（如非自然拥堵或碰撞）。
- **推理效率低**：基于 diffusion 的方法（如 SPDiff）需要多次前向传播，计算开销大，难以用于大规模实时仿真。

### 🚀 提出的新方法与创新思路
作者提出 **STDDN**（Spatio-Temporal Decoupled Differential Equation Network），一种融合宏观物理约束与深度学习的新型框架，其核心思想是：
> 将 crowd 视为连续介质，利用流体力学中的 **continuity equation** 对密度演化进行建模，并通过 Neural ODE 实现宏观物理规律对微观轨迹预测的端到端正则化。

#### 主要创新点包括：
1. **统一的宏-微耦合建模框架**
   - 引入 **continuity equation** 作为可微分的物理先验，将轨迹预测转化为密度场传输过程。
   - 使用 **Neural ODE** 建模宏观密度演化，由微观轨迹预测驱动，实现物理一致性与表达能力的结合。

2. **物理可解释的动态图网络设计**
   - 构建 **Density-Velocity Coupled Graph Learning (DVCG)** 模块，以当前速度为入边、未来速度为出边，显式建模跨时间步的密度通量（density flux），增强模型可解释性。

3. **两个可微结构提升物理一致性**
   - **Differentiable Density Mapping (DDM)**：基于 RBF 的软分配策略，避免传统硬划分带来的梯度不连续问题。
   - **Continuous Grid Detection (CGD)**：使用 Jensen-Shannon 散度量化跨网格移动程度，生成可微的 cross-grid mask，确保质量守恒且支持反向传播。

4. **高效轻量的节点嵌入机制**
   - 提出低秩参数化的 **Node Embedding (NE)**，将存储复杂度从 $O(N^2)$ 降至 $O(N \cdot d)$，显著降低内存占用。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **物理一致性** | 显式引入 continuity equation，保证密度演化符合质量守恒定律 |
| **长期稳定性** | 宏观约束有效抑制误差累积，适合长时序仿真 |
| **推理效率** | 单次前向即可完成预测，相比 diffusion 模型大幅减少延迟 |
| **泛化能力** | 在多种场景（高密度、高速变化）下均表现优异 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
在四个真实世界轨迹数据集上进行全面评估：
| 数据集 | 场景特点 | 时间长度 | 密度特征 |
|--------|----------|-----------|------------|
| **GC** | 高密度广场人流 | 300秒 | >200人/分钟 |
| **UCY**（含 ZARA1, ZARA2, UCY） | 街道行人流 | 216秒 | 中高密度 |
| **ETH**（含 ETH, HOTEL） | 校园与酒店门口 | 原始分割 | 多样化运动模式 |

> 所有数据经过坐标变换与时间插值处理，确保时空对齐。

### 📊 实验设置与评估指标

#### 评估指标
| 类别 | 指标 | 说明 |
|------|------|------|
| **准确性** | MAE（Mean Absolute Error） | 轨迹点平均偏移距离 |
|          | OT（Optimal Transport Distance） | 度量轨迹形状相似性，更关注整体分布匹配 |
| **效率** | #Pars（参数量） | 模型大小 |
|         | Latency（单帧推理延迟，ms） | 推理速度的关键指标 |

#### 基线方法分类对比
| 类型 | 代表方法 |
|------|----------|
| **Physics-based** | SFM, CA |
| **Data-driven** | STGCNN, PECNet, MID |
| **Physics-guided** | PCS, NSP, SPDiff |

> 特别强调与当前最优的 physics-guided 方法 SPDiff 进行比较。

---

## 3. 主要实验结果和性能指标

### 📈 性能对比（取自 Table 1 和 Table 2）

| Dataset | Method | MAE ↓ | OT ↓ | #Pars | Latency ↓ (ms) |
|--------|--------|-------|------|--------|----------------|
| **GC** | SPDiff | 0.9116 | 1.3925 | 0.14M | 206.99 |
|        | **Ours** | **0.8875** (-2.6%) | **1.3582** (-2.46%) | 0.17M | **86.85** (**↓58%**) |
| **UCY** | SPDiff | 1.8760 | 4.0564 | 0.22M | 471.05 |
|         | **Ours** | **1.7747** (-5.39%) | **3.6503** (-10.01%) | 0.07M | **44.66** (**↓90%**) |
| **ETH** | SPDiff | 0.5527 | 0.8706 | 0.18M | 81.41 |
|         | **Ours** | **0.5185** (-6.0%) | **0.6918** (-19.81%) | 0.20M | **30.57** (**↓62%**) |
| **HOTEL** | SPDiff | 0.3380 | 0.1646 | 0.16M | 68.57 |
|           | **Ours** | **0.2952** (-12.66%) | **0.1445** (-12.21%) | 0.05M | **17.50** (**↓75%**) |

> ✅ **关键发现**：
> - 在所有数据集上，STDDN 在 **MAE 和 OT** 上全面超越 SPDiff 等 SOTA 方法。
> - 推理延迟降低 **50%-90%**，尤其在 UCY 上实现近 **10倍加速**。
> - 参数量更低（最低仅 0.05M），具备工程部署潜力。

### 🔍 消融实验结果（Ablation Study，Table 3）

| 变体 | 描述 | GC MAE ↑ | UCY MAE ↑ |
|------|------|---------|----------|
| **Ours** | 完整模型 | 0.8875 | 1.7747 |
| w/o ODE | 移除 Neural ODE 约束 | 1.3784 (+55%) | 2.4867 (+40%) |
| w/o Cross-net | 移除 CGD 模块 | 0.9784 (+10%) | 1.8926 (+6.6%) |
| w/o NN loss | 仅用 ODE loss 训练 | 1.2387 (+40%) | 1.9327 (+8.9%) |
| w/o NE | 移除节点嵌入 | 0.8921 (+0.5%) | 1.7917 (+0.9%) |
| Discrete NN | 替换为离散更新 | 0.8875 (=) | 1.7747 (=) |

> 💡 **结论**：
> - **Neural ODE 是关键**：移除后误差显著上升，验证了宏观物理约束对抑制误差累积的有效性。
> - **CGD 至关重要**：准确建模跨网格流动对保持质量守恒至关重要。
> - **双损失协同作用**：单独依赖物理或数据都会导致性能下降，二者需联合优化。
> - **Euler 求解器最优**：尽管 Dopri5/RK4 数值精度更高，但在本任务中反而性能下降，因其引入中间状态破坏了离散观测一致性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **宏观物理规律可有效指导微观建模**  
   利用 **continuity equation** 作为结构性约束，能够显著提升 crowd simulation 的物理一致性和长期稳定性。

2. **宏-微耦合优于纯微观建模**  
   将人群视为连续介质并建模其密度演化，比单纯追踪个体轨迹更能捕捉集体动力学特性。

3. **可微分设计是端到端训练的关键**  
   DDM 与 CGD 模块解决了离散化带来的梯度不连续问题，使物理约束能真正参与梯度优化。

4. **推理效率大幅提升**  
   相比 diffusion-based 方法需多步去噪，STDDN 采用单步 autoregressive 预测，实现 **数量级级别的加速**。

### ⚠️ 局限性
- **空间离散化依赖网格粒度**：过细的网格会增加计算负担，过粗则丢失细节；需手动调参平衡。
- **假设人群为连续场**：在极稀疏场景下可能不如个体模型灵活。
- **未考虑异质性因素**：如行人年龄、意图多样性等未显式建模。

### 🔮 未来工作方向
- 扩展至 **三维空间与多层建筑** 中的人群模拟。
- 结合 **更多流体方程**（如 Navier-Stokes）建模加速度与压力场。
- 探索 **adaptive grid refinement** 技术以动态调整分辨率。
- 应用于 **城市规划、应急疏散系统** 等实际工程场景。

---

> 🔗 **代码开源地址**：[https://github.com/liuzjin/STDDN](https://github.com/liuzjin/STDDN)

</details>

---

### 10. [Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms](https://arxiv.org/abs/2604.02927)

**Authors**: Andreas Boltres, Niklas Freymuth, Benjamin Schichtholz, Michael K\"onig, Gerhard Neumann  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.02927v1  

#### Abstract
Routing algorithms are crucial for efficient computer network operations, and in many settings they must be able to react to traffic bursts within milliseconds. Live telemetry data can provide informative signals to routing algorithms, and recent work has trained neural networks to exploit such sign...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms

## 1. 论文的主要贡献和创新点

### 解决的问题
传统路由算法（如 OSPF、EIGRP）在面对突发流量时反应迟缓，难以实现毫秒级的动态调整。尽管已有研究尝试使用 **Machine Learning (ML)** 和 **Reinforcement Learning (RL)** 进行流量感知路由优化，但这些方法存在以下关键缺陷：
- **忽略通信延迟**：许多神经路由算法假设可以获取无延迟的全局网络状态（"birds-eye view"），这在真实网络中不可行。
- **纯局部观测限制**：部分分布式方法仅依赖本地遥测数据，缺乏对全网状态的有效感知。
- **推理延迟未建模**：现有框架通常忽略模型推理本身所需的时间，导致在真实部署中性能下降。

这些问题使得现有神经路由算法在**近实时（near-real-time）** 场景下的可部署性存疑。

### 提出的新方法与新思路
本文提出了一套完整的解决方案，核心包括：

#### （1）**延迟感知仿真框架（Delay-Aware Simulation Framework）**
- 将遥测感知路由建模为一个**延迟感知的闭环控制问题**。
- 显式地模拟了**通信延迟**（状态和动作传播）和**推理延迟**（模型计算时间）。
- 支持多种部署模式，包括集中式（Central）、分布式（Local）等，以评估不同架构的实际效果。

#### （2）**新型神经路由算法 LOGGIA**
- **LOGGIA**（LOg-space link weight prediction on Graphs with Guided update epochs and Implicit-Alpha entropy adaptation）
- **图神经网络架构**：直接在原始拓扑图上运行 **Message Passing Networks (MPNs)**，预测链路权重。
- **Log-Space 权重预测**：在对数空间中预测链路权重，提升数值稳定性和训练效率。
- **两阶段训练协议**：
  - **预训练阶段**：使用 **Imitation Learning (IL)** 模仿静态路由策略（如 EIGRP），实现“热启动”。
  - **强化学习阶段**：采用改进的 **Proximal Policy Optimization (PPO)**，结合最大熵探索（类似 SAC）和早期停止机制，提高训练稳定性。

### 相比现有方法的优势
- **更贴近现实**：首次在训练和评估中显式建模通信与推理延迟，结果更具实际意义。
- **更高的性能**：在延迟感知环境下，LOGGIA 是唯一能持续超越最短路径（Shortest Path, SP）基线的神经算法。
- **更好的泛化能力**：即使只在一个小型网络（如 `mini5`）上训练，也能泛化到包含上百节点的未知网络拓扑。
- **高效的部署**：推荐使用**完全分布式部署**（Local-Multi），即每个路由器独立观测和决策，性能最佳。

---

## 2. 核心实验方法和设置

### 数据集与网络拓扑
实验在多种合成和真实网络拓扑上进行：
- **`mini5`**：5 节点的小型合成网络。
- **`B4`**：Google 的数据中心广域网，12 节点，17 条链路。
- **`GEANT`**：欧洲科研教育网络，27 节点，38 条链路。
- **`nx` 系列**：可变规模的合成拓扑族，包括 `nx-XS` (6–10 节点)、`nx-S` (11–25 节点)、`nx-M` (26–50)、`nx-L` (51–100)，用于测试泛化能力。

### 实验设置
- **仿真平台**：基于 **ns-3** 的包级网络模拟器，通过 **ns3-ai** 使用共享内存实现高速交互。
- **时间粒度**：每轮控制周期为 **5ms**，模拟 2 秒的连续网络运行。
- **流量模型**：混合 **80% TCP 流** 和 **20% UDP 流**，流量强度经过调优，确保静态路由下必然出现丢包。
- **部署模式**：评估了五种组合，重点关注 **Local-Multi**（完全分布式）模式。

### 评估指标
- **主要指标**：**Goodput（有效吞吐量）**，即成功送达的数据量（单位：MB），作为优化目标。
- **辅助指标**：平均延迟（Delay）、队列负载（Queue Load）、TCP 丢弃量（TCP Discard）等。

### 基线方法对比
- **静态最短路径（SP）基线**：
  - `SPRIP`：基于跳数最少。
  - `SPEIGRP`：基于带宽和延迟的复合度量。
  - `SPOSFP`：基于带宽的度量。
- **神经路由基线**：
  - `MAGNNETO`：集中式图神经网络路由。
  - `FieldLines` 和 `M-Slim`：来自 Boltres et al. (2024) 的神经路由方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 `B4` 拓扑上，**LOGGIA** 的平均 Goodput 达到 **333.37 MB**，显著高于最佳 SP 基线 `SPRIP` 的 329.47 MB。
- 在更大的 `GEANT` 拓扑上，**LOGGIA** 达到 **460.71 MB**，远超 `SPEIGRP` 的 441.86 MB。

### 与基线方法的对比结果
- **所有其他神经基线在延迟感知环境下均失败**：`MAGNNETO`、`FieldLines`、`M-Slim` 在考虑通信和推理延迟后，性能均低于甚至不如静态最短路径算法。
- **LOGGIA 是唯一胜出者**：在所有测试拓扑和部署模式下，只有 LOGGIA 能在延迟感知环境中稳定超越 SP 基线。
- **泛化能力强**：在 `nx` 系列拓扑上，仅用 `mini5` 训练的 LOGGIA 表现优于用 `B4` 训练的版本，且扩展性与 SP 基线相当。

### 消融实验结果
#### （1）**架构设计消融（C.1）**
- **Log-Space 预测**：对性能提升至关重要。
- **直接在原始图上操作**（而非 Line Digraph）：优于转换图表示。
- **更深的 GNN**（L=4）：比浅层网络（L=2）表现更好。

#### （2）**训练机制消融（C.2）**
- **早期停止** 和 **最大熵探索** 显著提升了 LOGGIA 的训练稳定性和最终性能。
- 对于 `M-Slim`，这些改进效果不明显。

#### （3）**预训练消融（C.4）**
- **Imitation Learning (IL) 作为预训练**：能显著提升后续 PPO 训练的性能和稳定性。
- **Behavioral Cloning (BC)**：作为独立训练器优于 IL，但作为预训练阶段则劣于 IL。

#### （4）**部署模式影响（图6, 图17）**
- **Local-Multi 模式最优**：完全分布式部署在所有延迟感知模式中性能最好。
- **推理延迟（λac）越大，性能越差**：验证了快速推理的重要性。
- **训练时是否考虑延迟**：对多智能体训练（MAPPO）有正面影响，但对单智能体（PPO）影响不大。

---

## 4. 关键结论和发现

### 主要发现
1. **延迟是神经路由成败的关键**：一旦引入真实的通信和推理延迟，几乎所有现有神经路由算法都会失效。**必须在训练中显式建模这些延迟**。
2. **LOGGIA 是有效的**：其结合 **IL 预训练**、**Log-Space GNN** 和 **改进 PPO** 的设计，在延迟感知环境下仍能稳定超越传统路由协议。
3. **完全分布式部署最优**：**Local-Multi** 模式（每个路由器独立观测和决策）性能最佳，优于任何中心化或半中心化方案。
4. **硬件速度直接影响性能**：更快的 CPU 可降低推理延迟，从而提升最终的 Goodput。
5. **简单训练即可泛化**：无需复杂训练集，仅在一个小网络上训练，就能泛化到大型未知网络。

### 局限性
- **单路径路由**：目前仅支持基于单一成本度量的最短路径，无法像 BGP 那样支持多路径或复杂策略。
- **理想化通信模型**：假设控制信道带宽无限，忽略了控制消息本身的拥塞问题。
- **未建模表项更新延迟**：转发表安装过程中的延迟未被完全考虑。
- **TCP 乱序问题**：神经路由可能导致路径频繁切换，引发 TCP 包乱序和重传。

### 未来工作方向
- 扩展至**多路径路由**和**更复杂的路由策略**。
- 引入**控制信道带宽限制**和**消息压缩**技术。
- 探索**实时 MDP** 或**网络化 MDP** 的理论框架，为延迟感知路由提供更强的理论支撑。
- 设计专门针对 **TCP 流量** 的路由策略，减少路径切换带来的负面影响。

</details>

---

### 11. [MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents](https://arxiv.org/abs/2604.04853)

**Authors**: Shu Wang, Edwin Yu, Oscar Love, Tom Zhang, Tom Wong, Steve Scargall, Charles Fan  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.04853v1  

#### Abstract
Large Language Model (LLM) agents require persistent memory to maintain personalization, factual continuity, and long-horizon reasoning, yet standard context-window and retrieval-augmented generation (RAG) pipelines degrade over multi-session interactions. We present MemMachine, an open-source memor...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **LLM** 的 **AI Agent** 在实现个性化、长期记忆和跨会话任务执行时面临以下挑战：
- **静态参数限制**：LLM 的权重固定，无法从交互中动态学习新知识。
- **上下文窗口有限**：长对话历史难以完整保留，导致“丢失在中间”（Lost in the Middle）现象。
- **传统 RAG 的局限性**：标准 **Retrieval-Augmented Generation (RAG)** 主要面向静态文档检索，不支持动态、双向的 agent-user 交互。
- **现有记忆系统成本高且易出错**：如 **Mem0** 和 **Zep** 依赖 LLM 进行每条消息的事实提取，导致高昂的 token 成本和累积性错误。

### 提出的新方法与创新
**MemMachine** 是一个开源的、**ground-truth-preserving** 架构的记忆系统，其核心创新包括：

#### ✅ 1. **Ground-Truth-Preserving 架构**
- 存储原始对话片段（raw conversational episodes），以句子为单位进行索引。
- 最小化对 LLM 的依赖，仅在摘要生成和 profile 提取等高层抽象阶段调用 LLM，避免频繁的事实抽取引入误差。

#### ✅ 2. **Contextualized Retrieval（上下文化检索）**
- 引入“核事件”（nucleus episode）概念，检索匹配句后，自动扩展其前后邻近的对话轮次形成“episode cluster”。
- 解决了对话中语义相关证据分散于多轮、嵌入相似度低的问题，显著提升召回率。

#### ✅ 3. **三层记忆架构**
- **Short-Term Memory (STM)**：维护近期对话上下文，支持快速访问。
- **Long-Term Episodic Memory (LTM)**：持久化存储所有历史对话，支持向量检索。
- **Profile Memory (Semantic Memory)**：从对话中提取用户偏好、事实和行为模式，用于个性化响应。

#### ✅ 4. **Retrieval Agent（检索代理）**
- 一种 LLM 驱动的检索流水线，能根据查询类型路由到不同策略：
  - **Direct Search**：单跳查询。
  - **SplitQuery**：并行分解多实体查询。
  - **ChainOfQuery**：迭代链式推理处理多跳依赖。
- 支持 **late binding problem**（后期绑定问题），即中间实体未知时仍可逐步推理。

#### ✅ 5. **成本效率优势**
- 相比 **Mem0**，输入 token 减少约 **80%**，大幅降低运行成本。
- 通过减少不必要的 LLM 调用，提升了系统的可扩展性和实用性。

---

## 2. 核心实验方法和设置

### 数据集
论文在多个权威基准上进行了评估：

| 数据集 | 描述 |
|-------|------|
| **LoCoMo** | 多会话对话记忆基准，涵盖单跳、多跳、时间推理和开放域问答（共 1,540 个问题）。 |
| **LongMemEvals (ICLR 2025)** | 评估五项核心能力：信息提取、跨会话推理、时间推理、知识更新和拒绝回答。 |
| **HotpotQA hard** | 多跳问答数据集，强调推理链和证据整合。 |
| **WikiMultiHop** | 含噪声的多跳问答测试，模拟真实场景下的记忆干扰。 |
| **EpBench** | 基于合成叙事的 episodic memory 评测，规模达 100K–1M tokens。 |

### 实验设置
- **环境配置**：
  - CPU: 8vCPU, RAM: 16GiB
  - 数据库：PostgreSQL（pgvector）、Neo4j（图数据库）
  - 嵌入模型：`text-embedding-3-small`
  - 重排序器：AWS Cohere `rerank-v3`
- **LLM 设置**：
  - 评估模型（eval-LLM）：gpt-4.1-mini, gpt-4o-mini, gpt-5, gpt-5-mini
  - 判断模型（judge-LLM）：gpt-4o-mini 或 gpt-5-mini

### 评估指标
| 指标 | 说明 |
|------|------|
| **LLM Judge Score (llm_score)** | 主要指标，由 judge-LLM 判断生成答案是否与 ground truth 语义一致（0/1 分）。 |
| **BLEU / F1 Score** | 补充指标，衡量 n-gram 重叠和词级精确率/召回率。 |
| **Recall** | 黄金支持事实的检索成功率。 |
| **Token Cost** | 输入/输出 token 数量，反映系统效率。 |

### 基线方法对比
- **Mem0**：主流生产级记忆系统，依赖 LLM 提取事实。
- **Zep**：基于时间知识图谱的记忆架构。
- **Memobase / LangMem**：其他开源记忆框架。
- **OpenAI 全局记忆**：ChatGPT 内置记忆功能。
- **Mastra Observational Memory**：压缩日志式记忆，始终保留在上下文中。
- **MemOS**：全栈内存操作系统，支持 parametric 和 activation memory。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| Benchmark | Metric | Result |
|----------|--------|--------|
| **LoCoMo** | Overall Score (gpt-4.1-mini) | **0.9169** |
| **LongMemEvals** | Ablation Best (gpt-5-mini) | **93.0%** |
| **HotpotQA hard** | Retrieval Agent Accuracy | **93.2%** |
| **WikiMultiHop** | Retrieval Agent Accuracy | **92.6%** |
| **vs. Mem0** | Input Token Reduction | **~80% less** |

### 与基线方法对比（LoCoMo 总体得分）

| System | Overall Score |
|--------|----------------|
| **MemMachine (gpt-4.1-mini)** | **0.9169** |
| Memobase | 0.7578 |
| Zep | 0.7514 |
| Mem0 | 0.6688 |
| LangMem | 0.5810 |
| OpenAI | 0.5290 |

> 🔍 **结论**：MemMachine 显著领先，比第二名 **Memobase** 高出 **+9.7 个百分点**。

### 消融实验结果（LongMemEvals, gpt-5-mini）

| 优化维度 | 贡献（Δ Score） |
|---------|----------------|
| **Retrieval Depth Tuning (k=20→30)** | **+4.2%** ⬆️（最大增益） |
| **Context Formatting** | **+2.0%** |
| **Search Prompt Design** | **+1.8%** |
| **Query Bias Correction**（偏向用户消息） | **+1.4%** |
| **Sentence Chunking**（句子级分块） | **+0.8%** |
| **Answer Model: GPT-5 → GPT-5-mini** | **+2.6%** 🚀（反直觉发现） |

> 💡 **关键发现**：
> - **检索阶段优化 > 摄取阶段优化**：如何检索比如何存储更重要。
> - **GPT-5-mini 反超 GPT-5**：更小模型配合简洁 prompt 效果更好，说明存在 **model-prompt co-optimization**。

### Retrieval Agent 性能（HotpotQA hard）

| Strategy | Accuracy | Recall |
|--------|---------|--------|
| **Overall (Agent)** | **93.2%** | **95.5%** |
| ChainOfQuery（多跳） | 92.27% | **95.31%** |
| SplitQuery（并行） | 94.07% | 92.83% |
| Direct Search | 93.53% | 89.31% |

> ✅ 在随机噪声注入下，Retrieval Agent 仍保持鲁棒性，验证其适用于真实复杂场景。

---

## 4. 关键结论和发现

### 主要结论
1. **Ground-truth preservation 是关键**  
   保存原始对话记录而非依赖 LLM 抽取，能有效防止信息失真和误差累积，是构建可信 agent 的基础。

2. **检索优于压缩（Retrieval > Compaction）**  
   尽管压缩（如 Mastra 的 observational memory）可节省 token 并启用 prompt caching，但牺牲了对原始 episode 的按需检索能力。对于需要审计、合规或多跳推理的应用，**检索机制不可替代**。

3. **检索阶段优化主导性能提升**  
   检索深度、上下文格式、搜索提示设计等远比摄取阶段的句子分块重要。这表明：**“怎么查”比“怎么存”更重要**。

4. **小模型 + 好 prompt > 大模型 + 差 prompt**  
   **GPT-5-mini** 在优化 prompt 下表现优于 **GPT-5**，揭示了 **prompt engineering 与 model selection 必须协同优化**，不能简单复用旧 prompt。

5. **Retrieval Agent 实现智能路由**  
   通过 LLM 路由器将查询分类，并采用专用策略处理多跳、扇出等复杂查询，在保持成本可控的同时显著提升准确率。

6. **MemMachine 高效且可扩展**  
   - 输入 token 比 Mem0 少 **80%**。
   - 支持 multi-tenancy 和 session 隔离，适合企业部署。
   - 开源、模块化设计，兼容多种 LLM 和数据库后端。

### 局限性与威胁
- **评估敏感性**：结果依赖于特定 LLM 版本、prompt 模板和 provider 更新。
- **跨系统比较混合来源**：部分基线数据来自公开报告，可能存在预处理差异。
- **未覆盖全部场景**：缺乏多语言、多模态、实时性极强的任务测试。
- **消融实验独立分析**：未探索维度间的交互效应（如 chunking 与 k 的联合影响）。
- **LongMemEval 子集先行测试**：早期配置使用子集，可能影响最终趋势判断。

### 未来工作方向
| 方向 | 说明 |
|------|------|
| **Procedural Memory** | 存储工具使用模式、工作流策略等“怎么做”的知识。 |
| **Enhanced Temporal Reasoning** | 专门的时间索引与查询扩展技术。 |
| **Adaptive Retrieval Depth** | 动态调整 `k` 值，基于查询复杂度和模型能力。 |
| **Memory Consolidation & Forgetting** | 模拟人类记忆机制，优先保留高频访问内容。 |
| **Multi-modal Memory** | 支持图像、音频等非文本数据的记忆管理。 |
| **Function-calling Code Mode** | 使用代码执行代替重复 LLM 调用，进一步降本增效（参考 Anthropic MCP）。 |
| **Budget-aware Routing** | 在 token 成本和延迟约束下自动选择最优策略。 |

---

> 📌 **总结一句话**：  
> **MemMachine 证明了“保留原始事实 + 智能检索策略”的组合，能够在低成本、高准确性、强可解释性的前提下，为个性化 AI Agent 提供强大而可靠的长期记忆能力。**

</details>

---

### 12. [GENSERVE: Efficient Co-Serving of Heterogeneous Diffusion Model Workloads](https://arxiv.org/abs/2604.04335)

**Authors**: Fanjiang Ye, Zhangke Li, Xinrui Zhong, Ethan Ma, Russell Chen, Kaijian Wang, Jingwei Zuo, Desen Sun, Ye Cao, Triston Cao, Myungjin Lee, Arvind Krishnamurthy, Yuke Wang  
**Category**: cs.DC  
**Published**: 2026-04-07  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.04335v1  

#### Abstract
Diffusion models have emerged as the prevailing approach for text-to-image (T2I) and text-to-video (T2V) generation, yet production platforms must increasingly serve both modalities on shared GPU clusters while meeting stringent latency SLOs. Co-serving such heterogeneous workloads is challenging: T...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**GENSERVE: Efficient Co-Serving of Heterogeneous Diffusion Model Workloads**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前生产环境中，**text-to-image (T2I)** 和 **text-to-video (T2V)** 扩散模型（diffusion models）常被部署在共享的 GPU 集群上，但二者在计算需求、并行特性、延迟目标等方面存在巨大差异：
- T2V 请求的每步运行时间远长于 T2I（最高可达 20×）
- T2I 对延迟敏感，而 T2V 任务耗时更长
- 现有系统采用静态资源配置和统一调度策略（如 FIFO、SJF），导致严重的 **SLO violation**（服务等级目标不达标）

传统调度策略无法平衡异构请求之间的资源竞争：
- **FIFO** 导致视频请求阻塞图像请求（HOL blocking）
- **SJF/SRTF** 优先处理短任务，导致视频请求饥饿

### 提出的新方法与核心思路
**GENSERVE** 是一个专为混合 T2I/T2V 工作负载设计的高效共服务系统，其核心思想是：  
> 利用扩散模型执行过程中的 **可预测性（predictability）** 和 **步骤级可抢占性（step-level preemptibility）**，实现细粒度、异构感知的资源管理。

#### 主要创新机制：
1. **智能视频预占（Intelligent Video Preemption）**
   - 在 denoising 步骤边界处暂停正在运行的视频任务，释放 GPU 给紧急图像请求
   - 基于 **deadline slack** 选择最安全的预占对象（即仍有足够松弛时间完成的任务）
   - 中断状态以紧凑的 latent tensor 形式保留在 GPU 内存中，恢复开销极低（毫秒级）

2. **弹性资源分配（Elastic Resource Allocation）**
   - **动态批处理（Dynamic Batching）**：对同分辨率图像进行 SLO-aware 批处理，提升利用率而不牺牲延迟
   - **弹性 Sequence Parallelism (SP)**：根据视频分辨率和集群负载动态调整 SP degree，优化 T2V 的通信/计算权衡

3. **SLO-aware 调度器（Stepwise SLO-aware Scheduler）**
   - 将资源分配建模为轻量级的 **动态规划（DP）问题**
   - 在每个调度轮次联合决策：是否预占、如何批处理、SP 度数切换
   - 目标函数最大化所有并发请求的 **SLO 达成率（SAR）**

### 相比现有方法的优势
| 特性 | 现有系统（如 RASP, SRTF） | GENSERVE |
|------|--------------------------|----------|
| 资源配置 | 静态 SP 分配 | 动态 SP + 动态批处理 |
| 调度粒度 | 请求级或粗粒度 | **步骤级（step-level）** |
| 异构支持 | 单一任务类型优化 | 显式支持 T2I/T2V 混合 |
| 决策方式 | 局部启发式 | 全局联合优化（DP） |
| SLO 平衡能力 | 差（偏向短任务） | 强（兼顾长短任务） |

---

## 2. 核心实验方法和设置

### 数据集与工作负载
- **T2I 模型**：Stable Diffusion 3.5 Medium（2.5B 参数）
- **T2V 模型**：Wan2.2-T2V-5B（5B 参数），生成 81 帧视频
- **提示词来源**：
  - 图像任务：DiffusionDB
  - 视频任务：VBench
- **合成工作负载** 包含 100 个请求，控制以下维度变化：
  - **任务比例**：轻图像（20% video）、均衡（50%）、重视频（80%）
  - **到达模式**：泊松分布（Poisson） vs. 突发流量（Bursty）
  - **分辨率分布**：均匀 vs. 偏斜（Dirichlet α=1.0）

### 实验平台
- **硬件**：8×NVIDIA RTX PRO 6000 Blackwell GPU（96GB GDDR7）
- **软件栈**：CUDA 12.9, PyTorch 2.8, NCCL
- **实现**：约 10K LOC Python，基于 Hugging Face Diffusers 和 xFuser 构建

### 评估指标
- **SLO Attainment Rate (SAR)**：按时完成请求的比例（总体 / 按模态拆分）
- **Per-request Latency CDF**：衡量尾延迟表现
- **调度开销**：DP solver 决策时间、预占恢复延迟
- **内存占用**：paused VideoState 的 VRAM 开销

### 基线方法对比
| 编号 | 方法 | 描述 |
|------|------|------|
| B1 | FCFS | 先到先得，无抢占 |
| B2 | SJF | 最短作业优先 |
| B3 | SRTF | 最短剩余时间优先 + 抢占 |
| B4 | RASP | 分辨率感知静态 SP 分配（{256p→1, 480p→2, 720p→4}） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（默认配置：均衡混合、arrival rate=24 req/min、SLO scale σ=1.0）

| 方法 | Overall SAR | Image SAR | Video SAR |
|------|-------------|-----------|-----------|
| FCFS | 59%         | 58%       | 60%       |
| SJF  | 61%         | 60%       | 62%       |
| SRTF | 71%         | 96%       | 46%       |
| RASP | 56%         | 64%       | 48%       |
| **GENSERVE** | **78%**     | **100%**  | **56%**   |

> ✅ **GENSERVE 在总体 SLO 达成率上领先最强基线（SRTF）达 7 个百分点**

### 不同场景下的对比优势

#### E1: SLO 宽松程度（σ = 0.8 → 1.3）
- 当 SLO 放宽至 σ=1.3 时，GENSERVE 达到 **90% 总体 SAR**（SRTF 仅 83%）
- 图像 SAR 在 σ≥1.0 时达到 **100%**
- 视频 SAR 显著优于 SRTF（56% vs. 44%），说明其避免了过度抢占

#### E2: 工作负载组成（从轻视频到重视频）
- 在 **重视频负载（80% video）** 下：
  - GENSERVE 实现 **41% 总体 SAR**
  - 比 SRTF（26%）高 **15 pp**，比 SJF（33%）高 **8 pp**
  - 视频 SAR 达 **31%**，是 SRTF（14%）的 **2.3 倍**

#### E3: 请求到达速率（12 → 36 req/min）
- 在中等负载（18 req/min）下，GENSERVE 达到 **87% SAR**，比 SRTF（74%）高出 **13 pp**
- 即使在高压下（36 req/min），仍保持 **64% SAR**，显著优于其他方法

#### E4: 端到端延迟
- **图像请求**：
  - GENSERVE 的 P90 延迟为 **5.8 秒**，相比 FCFS（18.0 秒）降低 **3.1×**
- **视频请求**：
  - 中位延迟降低 **41%**（52s vs. 89s）
  - 尾延迟略高（P99: 229s vs. 166s），但这是有意权衡——保护大多数请求满足 SLO

### 消融实验结果（Ablation Study）

| 配置 | Overall SAR | Image SAR | Video SAR | 关键发现 |
|------|-------------|-----------|-----------|---------|
| Baseline (FCFS) | 20% | 20% | 20% | 视频垄断 GPU，图像严重排队 |
| + Preemption | 39% | 68% | 10% | 图像改善明显，但视频受损严重（无协调） |
| + DP Solver | 58% | 94% | 22% | 协调后大幅减少无效抢占，双模态提升 |
| + SP Switching | **63%** | **94%** | **32%** | 弹性 SP 进一步释放潜力 |

> 🔍 发现：**DP Solver 是关键协调器**，单独带来 +19 pp 提升；**SP Switching 提供加速替代方案**，减少中断次数

### 其他关键实证分析
- **调度开销极低**：DP solver 平均耗时 < 0.5ms，最大不超过 1.9ms，占单步时间 < 0.25%
- **预占恢复开销小**：720p@SP=8 下 resume 时间为 0.868ms，占比 < 0.12%
- **内存开销可忽略**：最大 paused state 仅 **27.2MB**（720p, 81帧），远低于 GPU 容量（96GB）

---

## 4. 关键结论和发现

### 主要结论
1. **扩散模型的可预测性和步骤级可抢占性是高效共服务的关键前提**  
   > 使得系统能精确估计剩余成本，并在不影响质量的前提下进行细粒度调度。

2. **单一策略无法应对异构负载，必须采用联合优化框架**  
   > GENSERVE 通过将 **preemption、batching、SP switching** 统一纳入 DP 调度器，实现了全局最优资源分配。

3. **复制式共服务（replicated co-serving）优于专用分区（dedicated partitioning）**  
   > 动态复用所有 GPU 可避免资源碎片化，在不同负载比例下均取得更高利用率。

4. **SLO-aware 调度需权衡而非极端偏好**  
   > GENSERVE 不盲目抢占视频，而是评估净收益，从而在保障图像响应的同时维持合理的视频成功率。

### 方法的局限性
- **依赖离线性能剖面（profiling）**：需要预先测量不同配置下的 `T_step`，可能难以适应频繁更新的模型版本
- **假设模型权重可共驻内存**：若未来更大规模 DiT 模型无法同时加载，则需引入 weight migration 开销
- **未考虑跨节点扩展**：当前实验限于单服务器内多 GPU 场景，尚未验证跨节点通信的影响

### 未来工作方向
- 支持更多模态（如 T2A、T23D）的统一共服务体系
- 结合 caching 技术进一步降低 per-step 成本（如 MixFusion、FlexCache）
- 探索在线自适应 profiling，减少对手工配置的依赖
- 扩展至 disaggregated infrastructure（解耦架构），支持更灵活的资源编排

---

> 📌 **一句话总结**：  
> **GENSERVE 通过揭示并利用扩散模型“可预测 + 可抢占”的结构性特征，构建了一个轻量、高效的 SLO-aware 调度框架，在混合 T2I/T2V 共服务场景中实现了高达 44% 的 SLO 达成率提升，显著超越现有系统。**

</details>

---

### 13. [Communication-Efficient Distributed Learning with Differential Privacy](https://arxiv.org/abs/2604.02558)

**Authors**: Xiaoxing Ren, Yuwen Ma, Nicola Bastianello, Karl H. Johansson, Thomas Parisini, Andreas A. Malikopoulos  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.02558v1  

#### Abstract
We address nonconvex learning problems over undirected networks. In particular, we focus on the challenge of designing an algorithm that is both communication-efficient and that guarantees the privacy of the agents' data. The first goal is achieved through a local training approach, which reduces co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Communication-Efficient Distributed Learning with Differential Privacy**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题  
本文针对**非凸学习问题**在**无向网络**中的分布式优化场景，聚焦两个核心挑战：
- **通信效率低**：传统分布式算法需要频繁通信，导致带宽消耗大。
- **隐私泄露风险**：共享模型参数可能被用于反推个体的私有训练数据（如通过梯度泄露攻击）。

目标是设计一种**同时具备高通信效率和强隐私保护能力**的分布式学习算法。

---

### ✅ 提出的新方法或新思路  
提出了一种名为 **LT-ADMM-DP (Local Training ADMM with Differential Privacy)** 的新算法，其核心创新包括：

1. **Local Training + ADMM 架构**  
   - 借鉴 LT-ADMM 框架，在每次全局迭代前执行多个本地训练轮次（T epochs），显著减少通信频率。
   - 使用随机梯度下降（SGD）进行本地更新，提升计算效率。

2. **差分隐私机制集成于本地训练阶段**  
   - 在本地梯度上引入双重保护：
     - **Gradient Clipping**：限制单个样本对梯度的影响，控制敏感度（sensitivity）。
     - **Additive Gaussian Noise**：在梯度中加入 $\mathcal{N}(0, \sigma^2 I)$ 噪声，实现 $(\epsilon, \delta)$-Differential Privacy (DP)。
   - 利用 **Rényi Differential Privacy (RDP)** 进行更紧致的隐私预算分析。

3. **理论保障全面**
   - 提供了算法收敛性证明：LT-ADMM-DP 收敛到非凸问题的一个驻点附近，误差有界。
   - 给出了严格的 DP 隐私保证，确保每个 agent 的数据无法从共享模型中被推断。

---

### ✅ 相比现有方法的优势  

| 方面 | LT-ADMM-DP 的优势 |
|------|------------------|
| **通信效率** | 显著优于 PORTER 和 PriSMA，仅需每 $T$ 轮通信一次，而其他方法每轮都通信。 |
| **隐私保护** | 同等隐私预算下提供更强的实际隐私保障，且支持灵活调节噪声与剪裁超参数。 |
| **性能表现** | 在相同 $\epsilon$ 下达到更高的分类准确率和更快的收敛速度。 |
| **理论完整性** | 同时提供收敛性和隐私性的严格数学证明，兼顾效率与安全。 |

---

## 2. **核心实验方法和设置**

### ✅ 数据集  
实验未直接使用公开真实数据集，而是构建了一个**合成分类任务**，局部损失函数为：

$$
f_i(x) = \frac{1}{m} \sum_{h=1}^{m} \left[ \log(1 + \exp(-b_{i,h} a_{i,h}^\top x)) + \frac{\lambda}{2} \|x\|^2 \right]
$$

其中：
- $a_{i,h} \in \mathbb{R}^n$: 特征向量
- $b_{i,h} \in \{-1, 1\}$: 标签
- $\lambda = 0.1$: 正则化系数

---

### ✅ 实验设置  
- **网络拓扑**：环形网络（ring network）
- **节点数量**：$N = 10$
- **模型维度**：$n = 5$
- **每节点样本数**：$m = 1000$
- **Mini-batch size**：$|B| = 8$
- **总迭代次数**：$K = 4000$
- **本地训练步数**：$T = 4$

#### ⚙️ 超参数配置
| 方法 | 参数设置 |
|------|---------|
| **LT-ADMM-DP** | $\gamma = \beta = 0.1$, $p = 0.1$, $C = 1$, $T = 4$, $\sigma_e = 0.5$ |
| **PORTER [21]** | Step size = 0.1, $C = 1$, $\sigma = 0.103$ |
| **PriSMA [22]** | $\gamma_y = 0.025$, $\eta = 0.025$, $C_1=C_2=1$, $\sigma_0=0.1794$, $\sigma_1=0.0155$ |

所有方法调整噪声水平以达到相同的 **隐私预算 $\epsilon = 19.6$**, $\delta = 10^{-4}$。

---

### ✅ 评估指标  
- **Optimization Error**: $|\nabla F(x_k)|$（梯度范数）
- **Classification Accuracy**：测试准确率
- **Total Time Cost**：考虑通信与计算开销的时间复杂度模型
  - 局部梯度计算时间：$t_g = 0.1$
  - 单次通信耗时：$t_c = 1$

---

### ✅ 基线方法对比  
- **PORTER [21]**：基于梯度裁剪和压缩的去中心化非凸优化方法
- **PriSMA [22]**：面向异构数据的分布式差分隐私学习算法

---

## 3. **主要实验结果和性能指标**

### ✅ 关键性能数据（见 Fig. 1）

| 指标 | LT-ADMM-DP | PORTER | PriSMA |
|------|------------|--------|--------|
| 最终分类准确率 | **≈89.5%** | ≈86.5% | ≈87.0% |
| 收敛速度（达 85% 所需时间） | **最快** | 较慢 | 最慢 |
| 最终 $|\nabla F(x_k)|$ | **最小** | 较大 | 最大 |

> 注：横轴按各算法的“时间复杂度”缩放（见 Table I），公平比较实际运行效率。

---

### ✅ 时间成本对比（Table I）

| 方法 | 每 $T$ 轮时间成本 |
|------|------------------|
| PORTER | $T(t_g + 2t_c)$ |
| PriSMA | $T(2t_g + t_c)$ |
| **LT-ADMM-DP** | $T t_g + t_c$ ✅（最低） |

👉 **LT-ADMM-DP 通信次数仅为其他方法的 1/4，大幅降低通信开销。**

---

### ✅ 性能对比结论  
- 在相同隐私预算 $\epsilon = 19.6$ 下：
  - LT-ADMM-DP 实现了**最快的收敛速度**和**最高的最终精度**。
  - 其通信效率远高于基线方法，尤其适合带宽受限环境。
- 图像显示 LT-ADMM-DP 曲线始终领先，验证了其综合优势。

---

### ✅ 消融实验（隐含分析）  
虽然文中未明确列出消融实验表格，但在理论部分进行了关键变量影响分析：
- **噪声方差 $\sigma^2$**：增大可提高隐私性，但会增加稳态误差。
- **本地训练步数 $T$**：增加可减少通信，但可能导致偏离全局最优。
- **剪裁阈值 $C$**：影响梯度敏感度，进而影响所需噪声大小。

👉 表明存在 **privacy-accuracy-efficiency 三者之间的权衡（trade-off）**，可通过调参优化。

---

## 4. **关键结论和发现**

### ✅ 主要发现  
1. **LT-ADMM-DP 成功实现了通信高效与差分隐私的统一**：
   - 通过 local training 显著减少通信频次；
   - 通过 clipped noisy gradients 提供严格 DP 保证。

2. **理论与实践一致性强**：
   - 收敛性分析表明算法能逼近非凸问题的驻点；
   - 隐私分析基于 RDP 得到紧致的 $(\epsilon, \delta)$-DP 上界。

3. **在相同隐私预算下性能超越 SOTA 方法**：
   - 相比 PORTER 和 PriSMA，LT-ADMM-DP 在准确率和收敛速度上均占优。

---

### ⚠️ 方法的局限性  
1. **依赖固定剪裁阈值 $C$**：未采用自适应 clipping 策略，可能影响隐私效用平衡。
2. **假设图连通且无向**：不适用于动态或有向网络拓扑。
3. **理论分析基于简化假设**：如梯度方差有界、L-smoothness 等，在极端非凸或病态问题中可能失效。
4. **未处理高度数据异构性（Non-IID）**：尽管 Assumption 3 引入了梯度差异项，但未专门优化应对极端分布偏移。

---

### 🔮 未来工作方向（作者指出）  
1. **研究自适应剪裁策略（adaptive clipping）**：根据训练进程动态调整 $C$，进一步提升隐私-效用权衡。
2. **增强对数据异质性（data heterogeneity）的鲁棒性**：设计更适合 Non-IID 场景的更新规则。
3. **扩展至异步或动态网络**：适应更复杂的现实通信环境。
4. **结合压缩技术**：进一步降低通信负载，实现 joint communication-and-computation efficiency。

---

## ✅ 总结评价

| 维度 | 评分（满分5★） | 说明 |
|------|---------------|------|
| 创新性 | ★★★★★ | 将 Local Training、ADMM 与 DP 完美融合，架构新颖 |
| 理论深度 | ★★★★☆ | 收敛性 + RDP 分析完整，附录详尽 |
| 实验充分性 | ★★★★☆ | 对比主流 baselines，指标合理，可视化清晰 |
| 实用价值 | ★★★★★ | 特别适用于边缘设备、IoT、医疗等隐私敏感场景 |
| 可复现性 | ★★★★☆ | 超参数详细，伪代码清晰，易于实现 |

> **总体评价**：一篇兼具理论严谨性与工程实用性的优秀工作，为**高效且安全的分布式学习**提供了可靠解决方案。

</details>

---

### 14. [FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving](https://arxiv.org/abs/2604.02715)

**Authors**: Qingxiu Liu, Cyril Y. He, Hanser Jiang, Zion Wang, Alan Zhao, Patrick P. C. Lee  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.02715v1  

#### Abstract
Mixture-of-Experts (MoE) models have become a dominant paradigm for scaling large language models, but their rapidly growing parameter sizes introduce a fundamental inefficiency during inference: most expert weights remain idle in GPU memory while competing with performance-critical runtime state su...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代 **Mixture-of-Experts (MoE)** 大语言模型虽然通过稀疏激活提升了参数规模而不显著增加计算量，但在推理过程中存在严重的**内存效率问题**：
- 所有专家权重（expert weights）在推理期间始终驻留在 GPU 内存中，即使大部分专家处于闲置状态。
- 这些闲置权重与对吞吐量至关重要的运行时状态（如 **KV Cache**）竞争有限的 GPU 显存资源。
- 由于 **KV Cache 容量直接决定服务吞吐量（throughput）**，这种资源错配导致显存利用率低下、性能受限。

### **提出的新方法与创新思路**
论文提出了 **FluxMoE**，一种全新的 MoE 推理系统，其核心思想是将专家参数从持久性的 GPU 驻留中解耦，引入 **Expert Paging（专家分页）抽象**：

> **模型 = 计算图 + 流式参数**  
> （`model = compute graph + streamed parameters`）

具体机制包括：
1. **PagedTensor**：提供张量虚拟化抽象，为每个专家张量分配稳定的虚拟地址，动态绑定物理内存块，无需修改底层计算内核（如 PyTorch/Triton）。
2. **带宽均衡的存储层级（Bandwidth-Balanced Storage Hierarchy）**：
   - 将专家参数分布在压缩后的 GPU 显存 和 主机 DRAM 中。
   - 利用 **选择性 Huffman 编码** 对专家权重中的指数位（exponent bits）进行无损压缩（节省约 20% 显存），并在 GPU 上实时解压。
   - 通过比例分配策略使各存储后端加载时间对齐，最大化聚合带宽。
3. **预算感知的驻留规划器（Budget-Aware Residency Planner）**：
   - 动态调整保留在 GPU 中的专家比例 α。
   - 根据 `compute-to-load ratio (p)` 实现闭环控制：当 `p < 0.9` 时增加驻留以缓解 I/O 瓶颈；当 `p > 1` 时减少驻留以释放显存给 KV Cache。

### **相比现有方法的优势**
| 方面 | 传统方法（如 vLLM） | FluxMoE |
|------|------------------------|---------|
| **专家管理** | 全部常驻 GPU | 按需流式加载，仅保留必要部分 |
| **显存利用** | 被大量闲置专家占用 | 显存优先分配给 KV Cache 和激活缓冲区 |
| **扩展性** | 受限于 GPU 显存容量 | 支持总参数远超 GPU 显存的模型部署 |
| **兼容性** | 不支持大 MoE 模型 | 无缝集成 vLLM，无需修改 kernel |
| **精度保障** | — | 使用无损压缩，不牺牲模型准确性 |

---

## **2. 核心实验方法和设置**

### **使用的模型与数据集**
- **模型**：
  - `Mixtral-8×7B-Instruct`（32 层，47B 参数）
  - `Qwen3-Next-80B-A3B-Instruct`（48 层，80B 参数）
- **数据集**：
  - **ShareGPT Dataset**：真实用户对话数据，用于构建多轮对话请求负载。
- **目标场景**：高吞吐、长上下文、大批量的推理服务场景。

### **实验设置**
- **硬件平台**：
  - 4× NVIDIA L40 GPUs（每卡 48GB GDDR6）
  - Intel Xeon Platinum 8358 CPU，2TB Host DRAM
- **并行方式**：
  - 使用 Tensor Parallelism（TP=4 或 TP=2）模拟多 GPU 推理环境。
- **评估维度**：
  - 批处理大小（batch size）：32–256
  - 上下文长度（context length）：1,024–4,096 tokens

### **评估指标**
- **主要指标**：
  - **Aggregate Throughput (tokens/sec)**：系统每秒生成的 token 总数，反映整体服务能力。
- **次要指标**：
  - 内存使用情况（KV Cache vs Expert Weights）
  - 吞吐稳定性
  - 管理开销（overhead）

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **vLLM** | 行业标准框架，所有专家必须驻留 GPU，KV Cache 不足时向主机交换 |
| **vLLM-O** | 改进版 vLLM，部分专家卸载到主机 DRAM，但无压缩且调度粗粒度 |
| **FluxMoE-H** | 消融版本，采用整层压缩或卸载，缺乏细粒度带宽平衡优化 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **Exp#1：性能边界 regime（足够显存）**
- 在 `Qwen3-Next-80B-A3B-Instruct` 上测试（TP=4）：
  - 当 batch size = 256，context length = 4096：
    - **FluxMoE 达到最高 3.0× 吞吐提升** 相比 vLLM。
    - 相比 vLLM-O 提升达 **3.7×**。
  - 小批量时因解压开销略低于 vLLM（约 63.9%），但仍优于 vLLM-O。
  - 随着 batch size 增加，vLLM 因频繁 KV Cache 交换导致吞吐下降 32.2%，而 FluxMoE 维持稳定增长。

#### ✅ **Exp#2：容量边界 regime（显存严重受限）**
- 在 `Mixtral-8×7B-Instruct` 上部署（TP=2，显存不足）：
  - vLLM 直接 OOM，无法运行。
  - FluxMoE 成功运行，并实现：
    - 比 vLLM-O 高出 **28.5%–22.9%** 的吞吐（batch=256）。
  - 原因：FluxMoE 利用压缩 + 带宽均衡避免 PCIe 成为瓶颈。

#### ✅ **Exp#3：动态驻留适应性测试**
- 动态调节 α（专家驻留率）可有效应对 KV Cache 增长带来的压力。
- 在连续推理中，Planner 自动触发 7 次调整，共释放约 **5.3 GB GPU 显存**。
- **吞吐未低于固定 α=1.0 的基准**，证明 I/O 开销被完全隐藏。

#### ✅ **Exp#4：PagedTensor 开销分析**
- 在所有专家均驻留 GPU 的理想条件下测试管理开销：
  - 最大性能损失仅为 **3.0%**（在 batch=64, ctx=4096 时）。
  - 表明 PagedTensor 引入的虚拟化机制几乎无额外计算负担。

---

## **4. 关键结论和发现**

### **主要发现**
1. **专家权重不应长期驻留 GPU**：它们是“瞬态资源”，应按需加载、即用即弃。
2. **KV Cache 是吞吐瓶颈的关键**：最大化其可用显存空间是提升 throughput 的最有效手段。
3. **Expert Paging 可实现完美计算-I/O 重叠**：通过双层滑动窗口与异步流控，加载延迟可完全隐藏在前一层计算中。
4. **无损压缩 + 分层存储 + 动态调控 = 高效 MoE 推理新范式**：FluxMoE 在不影响准确性的前提下突破显存限制。

### **方法的局限性**
- **依赖 PCIe 带宽**：若主机传输带宽过低，可能成为新的瓶颈（尽管已通过压缩缓解）。
- **当前原型未启用全三态驻留模型**：目前仅支持压缩 GPU 驻留 与 主机卸载两种状态，未来可进一步细分。
- **冷启动阶段略有延迟**：首次加载仍需预热，但后续迭代可保持高效流水线。

### **未来工作方向**
- 支持更复杂的 **multi-tier storage hierarchy**（如 NVMe、RDMA 网络存储）。
- 结合 **dynamic KV Cache resizing** 技术，在运行时自动扩展缓存容量。
- 探索 **routing-aware prefetching**，提前预测下一跳专家以进一步降低延迟。
- 将 Expert Paging 思想推广至其他稀疏模型架构（如 Sparse Attention、Block-Sparse Networks）。

---

> 💡 **一句话总结**：  
> **FluxMoE 通过将 MoE 专家视为“可分页资源”而非“常驻数据”，实现了高达 3.0× 的推理吞吐提升，为大规模 MoE 模型的高效部署提供了全新路径。**

</details>

---

### 15. [Combee: Scaling Prompt Learning for Self-Improving Language Model Agents](https://arxiv.org/abs/2604.04247)

**Authors**: Hanchen Li, Runyuan He, Qizheng Zhang, Changxiu Ji, Qiuyang Mang, Xiaokun Chen, Lakshya A Agrawal, Wei-Liang Liao, Eric Yang, Alvin Cheung, James Zou, Kunle Olukotun, Ion Stoica, Joseph E. Gonzalez  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.04247v1  

#### Abstract
Recent advances in prompt learning allow large language model agents to acquire task-relevant knowledge from inference-time context without parameter changes. For example, existing methods (like ACE or GEPA) can learn system prompts to improve accuracy based on previous agent runs. However, these me...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Combee: Scaling Prompt Learning for Self-Improving Language Model Agents

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前的 **Prompt Learning** 方法（如 ACE、GEPA）虽然能够在推理时通过 `generate-reflect-update` 范式从交互轨迹中学习并改进系统提示（system prompt），但这些方法主要设计用于**单智能体或低并行度场景**。

当尝试通过增加 **batch size** 来提升并行度以加速学习时，会出现严重的 **Context Overload（上下文过载）** 问题：  
- 聚合器 LLM 需要处理大量并行生成的反思（reflections），导致其只能保留泛化性强但价值较低的通用模式，而丢失了对性能提升至关重要的**细粒度、高价值知识**。
- 这使得大规模并行训练反而导致准确率显著下降，甚至接近无上下文学习的基线水平。

因此，**如何在保持学习质量的前提下高效地扩展 prompt learning 到高并行度环境**，是本文要解决的核心挑战。

---

### ✅ 提出了什么新方法或新思路

作者提出 **Combee** ——一个支持可扩展并行 prompt learning 的分布式框架，采用 **Map-Shuffle-Reduce** 范式：

#### （1）**Parallel Scan Aggregation（并行扫描聚合）**
- 将所有 agent 的反思划分为多个子组，在每个子组内先进行局部聚合，再将各子组的更新结果进一步聚合为全局更新。
- 类似于并行计算中的前缀和（prefix sum）算法，避免单次输入过长导致的 context overload。
- 默认分组数为 $\sqrt{n}$，实现层级负载均衡。

#### （2）**Augmented Shuffling（增强型打乱机制）**
- 对每个 reflection 进行 $p$ 次复制（默认 $p=2$），然后随机打乱后分发给不同 worker。
- 增加重要信息被选中参与聚合的概率，提高鲁棒性，灵感来自 **Self-Consistency** 思想。

#### （3）**Dynamic Batch Size Controller（动态批大小控制器）**
- 自动探测不同 batch size 下的延迟，并拟合幂律曲线模型 $T_{\text{epoch}} = A \cdot \text{bs}^{-\alpha}$。
- 动态选择使每单位 batch 带来边际延迟减少低于阈值的最大 batch size，平衡速度与稳定性。

---

### ✅ 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **效率** | 支持高并行度运行，训练时间大幅缩短（最高达 **17× speedup**）。 |
| **质量** | 在高 batch size 下仍能保留更多高质量、具体化的知识条目，避免性能退化。 |
| **通用性** | 框架无关（framework-agnostic），可集成到 ACE、GEPA 等主流 prompt learning 方法中。 |
| **成本** | 与基线方法相比，总成本基本持平（equivalent cost）。 |

---

## 2. 核心实验方法和设置

### ✅ 使用了哪些数据集

| 数据集 | 类型 | 任务描述 |
|--------|------|----------|
| **AppWorld** | Agent Benchmark | 多步 API 交互任务，评估智能体完成复杂目标的能力（Task Goal Completion, TGC；Scenario Goal Completion, SGC） |
| **Terminal-Bench 2.0** | Agent Benchmark | 命令行软件工程任务，测试代码生成与调试能力，评估 Accuracy@1 |
| **Formula** | Domain-Specific | 数值推理任务，基于结构化财务文件进行公式计算 |
| **FiNER** | Domain-Specific | 金融实体识别（Financial Numeric Entity Recognition），在 XBRL 文档中标注数值实体 |

---

### ✅ 实验设置和评估指标

#### 模型配置
- 主干 LLM：**DeepSeek-V3.1**（128K 上下文窗口）
- 部分实验也验证了 **GPT-OSS 120B** 上的有效性

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Accuracy / Accuracy@1** | 分类或生成任务的正确率 |
| **TGC / SGC** | AppWorld 中的任务/场景完成率 |
| **Playbook Size (tokens)** | 学习得到的上下文 artifact 的长度，反映知识保留量 |
| **Training Time / Cost** | 训练耗时与 API 调用成本（美元） |

#### 并行设置
- Batch size 范围：1 ~ 125
- Combee 在 batch 40 或 30 下运行，远高于传统方法可行范围

---

### ✅ 基线方法对比

| 基线方法 | 描述 |
|---------|------|
| **ACE** | 将经验提炼为 playbook 的 prompt learning 方法 |
| **GEPA** | 基于进化搜索优化 system prompt 的方法 |
| **Naive Parallel Prompt Learning** | 直接将多个 reflections 拼接送入 aggregator，代表简单并行化策略 |
| **Top-K Retrieval** | 使用嵌入聚类后取每类代表 reflection 输入 |
| **Summarization** | 先对 reflections 做摘要再输入 aggregator |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

| 方法 | 数据集 | Batch Size | 准确率 | 训练时间 | Speedup |
|------|--------|------------|--------|-----------|---------|
| Naive ACE (bs=1) | AppWorld | 1 | 58.1% | 86 min | 1× |
| Naive ACE (bs=40) | AppWorld | 40 | 55.7% | 5 min | 17.2× |
| **Combee + ACE** | AppWorld | 40 | **65.8%** | **7 min** | **~12×** |
| Naive ACE (bs=1) | Terminal-Bench | 1 | 37.9% | 42.4 min | 1× |
| Naive ACE (bs=30) | Terminal-Bench | 30 | 31.0% | 2.1 min | ~20× |
| **Combee + ACE** | Terminal-Bench | 30 | **35.6%** | **2.4 min** | **~17.7×** |
| **Combee + GEPA** | Formula | - | **87.0%** | <50% baseline time | >2× |
| **Combee + ACE** | FiNER | - | **76.0%** | >2.4× faster than quality-matching baseline | >2.4× |

> 注：Combee 在 **AppWorld** 上实现了 **12× 加速**的同时，准确率**反超顺序训练基线**；在 **Terminal-Bench** 上实现 **17× 加速**且性能接近最优。

---

### ✅ 与基线方法的对比结果

| 对比项 | 结果 |
|-------|------|
| **vs. Naive Scaling** | 在高 batch size 下，naive 方法性能严重下降（如 Formula 从 87.0% → 72.5%），而 Combee 保持稳定甚至提升。 |
| **vs. Top-K / Summarization** | 两种缓解 context overload 的 prompt-level 方法效果有限，生成质量明显低于 Combee。 |
| **vs. Sequential (bs=1)** | Combee 在仅多花少量时间的情况下达到更高性能，实现了“更快且更好”。 |

---

### ✅ 消融实验结果

#### （1）**Dynamic Batch Size Controller 消融（Figure 6）**
- 固定 batch size 会导致要么太慢（小 batch）、要么不稳定（大 batch）。
- 动态控制器能自动找到性价比最高的 batch size，节省延迟而不牺牲精度。

#### （2）**Augmented Shuffling 消融（Figure 7）**
- 移除 shuffling 后，学习质量波动更大，尤其在非理想 subgroup size 下表现更差。
- 证明了重复投喂 reflection 可有效防止信息丢失。

#### （3）**Subgroup Size 影响**
- 当 subgroup size 接近 $\sqrt{\text{batch size}}$ 时性能最佳，验证了 parallel scan 设计合理性。

---

## 4. 关键结论和发现

### ✅ 论文的主要发现

1. **Context Overload 是限制 prompt learning 扩展性的根本瓶颈**：即使未超出上下文长度限制，简单堆叠 reflections 也会导致 aggregator 丢失关键细节。
2. **Combee 成功打破“速度-质量”权衡**：通过 parallel scan 和 augmented shuffling，可在高并行度下依然保留丰富的细粒度知识。
3. **Playbook 大小直接反映学习质量**：Combee 学得的 playbook 显著更长（如 AppWorld 中 6,887 vs. 526 tokens），说明其真正聚合了更多信息。
4. **框架具有良好的泛化能力**：在 agent benchmark 与 domain-specific task 上均有效，且适用于 ACE、GEPA 不同 backend。

---

### ✅ 方法的局限性（B节 Limitations）

1. 目前仅验证了与 ACE 和 GEPA 的集成，尚未测试其他结构差异较大的 memory abstraction（如程序库、检索增强技能库）。
2. 动态 batch size 控制依赖幂律延迟模型，可能不适用于延迟特性不同的部署环境。
3. 假设同步执行，未来可探索异步变体（类似异步 SGD）以进一步提升吞吐。

---

### ✅ 未来工作方向

1. 扩展至更多类型的 context artifact（如 program library、retrieval-augmented skill store）。
2. 引入异步或部分同步执行机制，适应异构环境。
3. 探索更复杂的 aggregation topology（如树形、图状结构）以进一步优化通信开销。
4. 将 Combee 应用于真实世界的大规模 agentic 系统（如自动编程、科研助手等）。

---

## 🔚 总结

**Combee** 是首个系统性解决 **prompt learning 可扩展性**问题的框架。它通过 **Parallel Scan + Augmented Shuffling + Dynamic Control** 三重机制，在不牺牲学习质量的前提下，实现了高达 **17× 的训练加速**，同时保持与顺序学习相当甚至更优的性能。该工作标志着 prompt learning 正式进入“大规模并行学习”的新时代，为构建持续自我进化的语言模型智能体提供了坚实基础。

</details>

---

### 16. [TriAttention: Efficient Long Reasoning with Trigonometric KV Compression](https://arxiv.org/abs/2604.04921)

**Authors**: Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.04921v1  

#### Abstract
Extended reasoning in large language models (LLMs) creates severe KV cache memory bottlenecks. Leading KV cache compression methods estimate KV importance using attention scores from recent post-RoPE queries. However, queries rotate with position during RoPE, making representative queries very few, ...

---

### 17. [A Numerical Method for Coupling Parameterized Physics-Informed Neural Networks and FDM for Advanced Thermal-Hydraulic System Simulation](https://arxiv.org/abs/2604.02663)

**Authors**: Jeesuk Shin, Donggyun Seo, Sihyeong Yu, Joongoo Jeon  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.02663v1  

#### Abstract
Severe accident analysis using system-level codes such as MELCOR is indispensable for nuclear safety assessment, yet the computational cost of repeated simulations poses a significant bottleneck for parametric studies and uncertainty quantification. Existing surrogate models accelerate these analyse...

---

### 18. [Beyond Semantic Manipulation: Token-Space Attacks on Reward Models](https://arxiv.org/abs/2604.02686)

**Authors**: Yuheng Zhang, Mingyue Huo, Minghao Zhu, Mengxue Zhang, Nan Jiang  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.02686v1  

#### Abstract
Reward models (RMs) are widely used as optimization targets in reinforcement learning from human feedback (RLHF), yet they remain vulnerable to reward hacking. Existing attacks mainly operate within the semantic space, constructing human-readable adversarial outputs that exploit RM biases. In this w...

---

### 19. [PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training](https://arxiv.org/abs/2604.03675)

**Authors**: Erhan Zhang, Yiqun Chen, Zechun Niu, Wei Yang, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.03675v1  

#### Abstract
In agentic search, large language models (LLMs) are trained to perform multi-turn retrieval and reasoning for complex tasks such as multi-hop question answering (QA). However, current search-based Reinforcement Learning (RL) methods suffer from two core limitations: expensive long-horizon rollouts a...

---

### 20. [RL-Driven Sustainable Land-Use Allocation for the Lake Malawi Basin](https://arxiv.org/abs/2604.03768)

**Authors**: Ying Yao  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.03768v1  

#### Abstract
Unsustainable land-use practices in ecologically sensitive regions threaten biodiversity, water resources, and the livelihoods of millions. This paper presents a deep reinforcement learning (RL) framework for optimizing land-use allocation in the Lake Malawi Basin to maximize total ecosystem service...

---

### 21. [LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning](https://arxiv.org/abs/2604.02338)

**Authors**: Md Kowsher, Haris Mansoor, Nusrat Jahan Prottasha, Ozlem Garibay, Victor Zhu, Zhengping Ji, Chen Chen  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.02338v1  

#### Abstract
MoE-PEFT methods combine Mixture of Experts with parameter-efficient fine-tuning for multi-task adaptation, but require separate adapters per expert causing trainable parameters to scale linearly with expert count and limiting applicability to adapter-based architectures. We propose LiME (Lightweigh...

---

### 22. [IC3-Evolve: Proof-/Witness-Gated Offline LLM-Driven Heuristic Evolution for IC3 Hardware Model Checking](https://arxiv.org/abs/2604.03232)

**Authors**: Mingkai Miao, Guangyu Hu, Ziyi Yang, Hongce Zhang  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.03232v1  

#### Abstract
IC3, also known as property-directed reachability (PDR), is a commonly-used algorithm for hardware safety model checking. It checks if a state transition system complies with a given safety property. IC3 either returns UNSAFE (indicating property violation) with a counterexample trace, or SAFE with ...

---

### 23. [TableVision: A Large-Scale Benchmark for Spatially Grounded Reasoning over Complex Hierarchical Tables](https://arxiv.org/abs/2604.03660)

**Authors**: Xiaoyu Chen, Lu Dai, Hanqing Wang, Zhuoyu Li, Wenbin Dai, Yanzong Zheng, Zhenggang Xia, Junyong Lin, Hui Xiong  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.03660v1  

#### Abstract
Structured tables are essential for conveying high-density information in professional domains such as finance, healthcare, and scientific research. Despite the progress in Multimodal Large Language Models (MLLMs), reasoning performance remains limited for complex tables with hierarchical layouts. I...

---

### 24. [Solar-VLM: Multimodal Vision-Language Models for Augmented Solar Power Forecasting](https://arxiv.org/abs/2604.04145)

**Authors**: Hang Fan, Haoran Pei, Runze Liang, Weican Liu, Long Cheng, Wei Wei  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.04145v1  

#### Abstract
Photovoltaic (PV) power forecasting plays a critical role in power system dispatch and market participation. Because PV generation is highly sensitive to weather conditions and cloud motion, accurate forecasting requires effective modeling of complex spatiotemporal dependencies across multiple infor...

---

### 25. [RESCORE: LLM-Driven Simulation Recovery in Control Systems Research Papers](https://arxiv.org/abs/2604.04324)

**Authors**: Vineet Bhat, Shiqing Wei, Ali Umut Kaypak, Prashanth Krishnamurthy, Ramesh Karri, Farshad Khorrami  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.04324v1  

#### Abstract
Reconstructing numerical simulations from control systems research papers is often hindered by underspecified parameters and ambiguous implementation details. We define the task of Paper to Simulation Recoverability, the ability of an automated system to generate executable code that faithfully repr...

---

### 26. [ShieldNet: Network-Level Guardrails against Emerging Supply-Chain Injections in Agentic Systems](https://arxiv.org/abs/2604.04426)

**Authors**: Zhuowen Yuan, Zhaorun Chen, Zhen Xiang, Nathaniel D. Bastian, Seyyed Hadi Hashemi, Chaowei Xiao, Wenbo Guo, Bo Li  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.04426v1  

#### Abstract
Existing research on LLM agent security mainly focuses on prompt injection and unsafe input/output behaviors. However, as agents increasingly rely on third-party tools and MCP servers, a new class of supply-chain threats has emerged, where malicious behaviors are embedded in seemingly benign tools, ...

---

### 27. [Memory Intelligence Agent](https://arxiv.org/abs/2604.04503)

**Authors**: Jingyang Qiao, Weicheng Meng, Yu Cheng, Zhihang Lin, Zhizhong Zhang, Xin Tan, Jingyu Gong, Kun Shao, Yuan Xie  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.04503v1  

#### Abstract
Deep research agents (DRAs) integrate LLM reasoning with external tools. Memory systems enable DRAs to leverage historical experiences, which are essential for efficient reasoning and autonomous evolution. Existing methods rely on retrieving similar trajectories from memory to aid reasoning, while s...

---

### 28. [Receding-Horizon Control via Drifting Models](https://arxiv.org/abs/2604.04528)

**Authors**: Daniele Foffano, Alessio Russo, Alexandre Proutiere  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.04528v1  

#### Abstract
We study the problem of trajectory optimization in settings where the system dynamics are unknown and it is not possible to simulate trajectories through a surrogate model. When an offline dataset of trajectories is available, an agent could directly learn a trajectory generator by distribution matc...

---

### 29. [AI Trust OS -- A Continuous Governance Framework for Autonomous AI Observability and Zero-Trust Compliance in Enterprise Environments](https://arxiv.org/abs/2604.04749)

**Authors**: Eranga Bandara, Asanga Gunaratna, Ross Gore, Abdul Rahman, Ravi Mukkamala, Sachin Shetty, Sachini Rajapakse, Isurunima Kularathna, Peter Foytik, Safdar H. Bouk, Xueping Liang, Amin Hass, Ng Wee Keong, Kasun De Zoysa  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.04749v1  

#### Abstract
The accelerating adoption of large language models, retrieval-augmented generation pipelines, and multi-agent AI workflows has created a structural governance crisis. Organizations cannot govern what they cannot see, and existing compliance methodologies built for deterministic web applications prov...

---

### 30. [MERIT: Multilingual Expert-Reward Informed Tuning for Chinese-Centric Low-Resource Machine Translation](https://arxiv.org/abs/2604.04839)

**Authors**: Zhixiang Lu, Chong Zhang, Chenyu Xue, Angelos Stefanidis, Chong Li, Jionglong Su, Zhengyong Jiang  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.04839v1  

#### Abstract
Neural machine translation (NMT) from Chinese to low-resource Southeast Asian languages remains severely constrained by the extreme scarcity of clean parallel corpora and the pervasive noise in existing mined data. This chronic shortage not only impedes effective model training but also sustains a l...

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
