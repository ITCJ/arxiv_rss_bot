# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-13 08:23:45 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [AB-Sparse: Sparse Attention with Adaptive Block Size for Accurate and Efficient Long-Context Inference](https://arxiv.org/abs/2605.12110)

**Authors**: Di Liu, Ruitian Wang, Chen Chen, Mingliang Gong, Yongjie Yuan, Han Zhao, Yu Feng, Quan Chen, Minyi Guo  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2605.12110v1  

#### Abstract
As large language models scale to longer contexts, loading the growing KV cache during attention computation becomes a critical bottleneck. Previous work has shown that attention computation is dominated by a small subset of tokens. This motivates block sparse attention methods that partition the KV...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：AB-Sparse: Sparse Attention with Adaptive Block Size for Accurate and Efficient Long-Context Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在大语言模型（LLMs）处理长上下文时，**KV Cache 的加载成为推理过程中的内存瓶颈**，尤其是在解码阶段。为缓解这一问题，已有研究提出 **block sparse attention** 方法，将 KV Cache 划分为固定大小的块，并仅加载重要性最高的 Top-K 块进行注意力计算。

然而，现有方法普遍采用**统一的 block size** 应用于所有 attention head，忽略了不同 attention head 对 block 粒度敏感性的显著差异。这种“一刀切”的策略导致：
- 对**敏感 head**：过大的 block size 会丢失关键 token，降低准确率；
- 对**不敏感 head**：过小的 block size 增加不必要的 centroid 数量，提升内存开销。

因此，**如何在不牺牲吞吐量的前提下提升 block sparse attention 的精度**，是本文要解决的核心问题。

---

### **提出了什么新方法或新思路**
作者提出 **AB-Sparse**，一个**无需训练、算法-系统协同设计**的框架，通过以下三个核心组件实现更精准高效的稀疏注意力：

1. **Adaptive Block Size Allocation（自适应块大小分配）**
   - 观察到不同 attention head 对 block granularity 的敏感性具有输入无关性和稳定性。
   - 在部署前通过轻量级校准（calibration）确定每个 head 的最优 block size，敏感 head 分配更细粒度（小 block），不敏感 head 使用粗粒度（大 block）。

2. **Lossless Centroid Quantization（无损质心量化）**
   - 发现 centroid 仅用于排序选择 Top-K 块，对精度不敏感。
   - 采用 **INT4 asymmetric per-channel quantization** 对 centroid 进行压缩，在几乎无精度损失下大幅减少内存占用。

3. **Custom GPU Kernels（定制化 GPU 内核）**
   - 设计三个专用 CUDA kernel 支持变长 block size 高效执行：
     - **Fused query-centroid estimation**：使用 prefix-sum indexing 实现无填充批处理；
     - **Batched Top-K selection**：支持每 head 不同的 centroid 数量；
     - **Heterogeneous paged attention**：利用逻辑块到物理页的 stride 映射，兼容标准 paged KV cache 管理。

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **准确性** | 自适应 block size 更好保留重要 token，显著提升 recall 和任务准确率 |
| **效率** | INT4 量化减少 memory traffic；定制 kernel 避免 padding 和 gather 开销 |
| **实用性** | 无需重新训练，可作为 drop-in 替换集成到 Quest / ArkVale 等系统中 |
| **兼容性** | 保持与 paged KV cache 的兼容性，适合实际部署 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **校准集（Calibration Set）**：`Wikipedia` 数据集（50 个样本）
- **主评估基准**：
  - **RULER**：合成基准，测试长上下文能力（检索、多跳推理、聚合、问答等 13 项任务）
  - **LongBench**：真实世界长文本理解任务（单/多文档 QA、摘要、少样本学习、代码补全等）
- **生成任务评估**：`AIME24`, `AMC23`, `MATH500` —— 长输出数学推理任务

---

### **实验设置和评估指标**

| 设置项 | 描述 |
|-------|------|
| **模型** | Llama-3.1-8B, Qwen3-8B, Qwen3-32B（均支持最长 128K 上下文） |
| **硬件平台** | NVIDIA A100-80GB 和 H800-80GB GPU |
| **KV 缓存预算** | 固定为 4%，平均 block size = 32（与 baselines 对齐） |
| **评估上下文长度** | 从 16K 到 256K 不等 |
| **主要指标** | 
| - **Accuracy (%)**：RULER 和 LongBench 上的任务得分 |
| - **Pass@4**：长生成任务的正确率 |
| - **Decoding Latency (ms)**：注意力计算延迟 |
| - **Throughput (tokens/s)**：解码吞吐量 |
| - **Attention Recall**：选中的 block 中包含高注意力分数 token 的比例 |

---

### **基线方法对比**
- **Full Attention**：完整注意力机制（无稀疏化，作为上限参考）
- **Quest**：基于 min-max pooling 的 block importance 估计
- **ArkVale**：使用 bounding-volume centroid 表示 block
- **AB-Sparse-Quest / AB-Sparse-ArkVale**：本文方法分别构建在这两个 baseline 之上

> 所有稀疏方法均控制相同 KV token 预算（4%）和平均 block size（32）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **准确率提升显著**
| 方法 | RULER 平均准确率提升 | LongBench 平均准确率提升 |
|------|------------------------|----------------------------|
| AB-Sparse-Quest vs Quest | **+3.51% ~ +5.43%** | **+2.47% ~ +2.62%** |
| AB-Sparse-ArkVale vs ArkVale | **+3.80% ~ +3.16%** | **+1.92% ~ +1.52%** |

> 在 Qwen3-32B 上，AB-Sparse 将 RULER 准确率从 76.24% 提升至 **81.67%**，接近 Full Attention 的 84.51%

#### ✅ **长生成任务表现优异**
| 基准 | Quest (pass@4) | AB-Sparse-Quest (pass@4) | 提升 |
|------|----------------|--------------------------|------|
| AIME24 | 20.0% | **23.3%** | +3.3% |
| AMC23 | 47.5% | **60.0%** | +12.5% |
| MATH500 | 74.0% | **76.0%** | +2.0% |
| **Average** | **47.2%** | **53.1%** | **+5.9%** |

> 表明 AB-Sparse 不仅适用于长输入任务，也有效支持**长输出生成场景**

#### ✅ **效率无损失，甚至更优**
- **延迟方面**：AB-Sparse 与 Quest 相当，在长上下文（如 256K）下因 INT4 量化减少 memory traffic 而**更快**
- **吞吐量方面**：
  - Batch size=1：吞吐量相近
  - Batch size=4：达到 Quest 的 **1.59× 吞吐量**
  - 原因：prefix-sum indexing 消除 padding，kernel 更高效

#### ✅ **消融实验结果**

##### （1）Centroid Quantization 效果（图13）
| 量化方式 | 准确率影响 |
|---------|-----------|
| BF16（未量化） | 基线 |
| INT8 | 几乎无损 |
| **INT4 asymmetric per-channel** | **与 BF16 几乎一致，最佳性价比选择** |
| INT2 | 显著下降 |

✅ 结论：**INT4 量化可在无损前提下压缩 centroid 存储空间达 4×**

##### （2）Custom Kernel 效果（图14）
| 操作 | 加速比（vs Naive 实现） |
|------|------------------------|
| Estimation | **5.6×** |
| Top-K Selection | **9.4×** |
| Attention Computation | **3.1×** |

✅ 定制 kernel 极大提升了各阶段执行效率，尤其在 Top-K 阶段避免串行处理

---

## **4. 关键结论和发现**

### **主要发现**
1. **Attention heads 对 block granularity 具有高度异质性（heterogeneous sensitivity）**
   - 有些 head 对 block size 极其敏感（需小 block）
   - 有些 head 几乎不受影响（可用大 block）
   - 强制统一 block size 导致次优 trade-off

2. **Per-head block size sensitivity 是稳定的、输入无关的**
   - 可通过少量校准样本离线确定最优配置，无需在线调整

3. **Centroid 可安全压缩**
   - 仅用于 ranking，INT4 asymmetric per-channel quantization 可实现无损压缩

4. **AB-Sparse 是通用插件式增强**
   - 可无缝集成到 Quest、ArkVale 等系统中，带来一致增益

---

### **方法的局限性**
- **依赖校准集代表性**：若实际输入分布与 calibration set 差异过大，可能影响 block size 分配质量（但实验显示泛化良好）
- **当前仅支持静态分配**：运行时无法动态调整 block size
- **对极低比特量化（<INT4）仍敏感**：INT2 会导致 recall 明显下降

---

### **未来工作方向**
- 探索 **dynamic block size adaptation**，根据输入内容实时调整
- 将 adaptive allocation 思路扩展到其他稀疏结构（如 token selection）
- 结合 **model pruning 或 MoE** 进一步优化端到端推理效率
- 推广至更多架构（如 Mamba、RetNet）和模态（多模态 LLM）

---

> **总结一句话**：  
> **AB-Sparse 通过“按头定制 block size + 无损 centroid 压缩 + 高效 kernel 实现”，在零吞吐代价下实现了高达 5.43% 的准确率提升，为高效长上下文推理提供了实用且普适的新范式。**

</details>

---

### 2. [Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models](https://arxiv.org/abs/2605.11854)

**Authors**: Kecheng Chen, Ziru Liu, Xijia Tao, Hui Liu, Yibing Liu, Xinyu Fu, Shi Wu, Suiyun Zhang, Dandan Tu, Lingpeng Kong, Rui Liu, Haoliang Li  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 13.0  
**Type**: new  
**ArXiv ID**: 2605.11854v1  

#### Abstract
Diffusion Language Models (DLMs) have recently emerged as a promising alternative to autoregressive language models, offering stronger global awareness and highly parallel generation. However, post-training DLMs with standard Negative Evidence Lower Bound (NELBO)-based supervised fine-tuning remains...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Diffusion Language Models (DLMs) 在推理时采用**confidence-guided、multi-step 的易到难（easy-to-hard）去噪轨迹**，而训练阶段通常使用标准的 **Negative Evidence Lower Bound (NELBO)** 目标函数，通过**随机均匀掩码**进行单步重建。这种“训练-推理不一致”（training-inference discrepancy）导致模型难以充分吸收推理过程中的结构性知识。

尽管已有工作利用 self-distilled trajectories 进行后训练（post-training），但这些方法主要聚焦于**采样步数压缩和推理加速**（如 dInfer、Seed Diffusion），并未真正提升模型能力，甚至在完整扩散解码下性能下降。

### 提出了什么新方法或新思路
本文提出 **Trajectory-Aligned optimization via Boltzmann Modeling (TABOM)**，一种基于 self-distilled 轨迹的新型后训练框架，其核心思想是：

- 将推理过程中 token 的 unmasking 顺序建模为一个 **Boltzmann 分布**，该分布以预测熵（predictive entropy）为能量项，体现“易先难后”的归纳偏置（inductive bias）。
- 设计了一个**可计算的成对排序目标（pairwise ranking objective）** 来近似优化 KL 散度，使模型学习到与实际推理轨迹一致的确定性排序。

### 相比现有方法的优势
| 方法 | 主要目标 | 是否增强模型能力 | 是否缓解灾难性遗忘 |
|------|--------|------------------|--------------------|
| SFT-GT | 提升 in-domain 性能 | ✅ | ❌（严重遗忘 OOD 能力） |
| SFT-SD / dInfer / T3D | 利用轨迹加速推理 | ⚠️有限提升 | ✅（保留原有能力） |
| **TABOM (Ours)** | **对齐训练-推理动态，实现知识获取** | ✅✅（显著提升 in-domain & OOD） | ✅✅（完全缓解遗忘） |

> ✅ TABOM 首次将 self-distilled 轨迹用于**真正的知识获取**而非仅加速，实现了性能与泛化的双赢。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **数学推理（Mathematical Reasoning）**:  
  - `MixChain-Z-PRM12K`（12K 查询）
- **代码生成（Code Generation）**:  
  - `Ling-Coder-SFT`（18K 查询）

所有 self-distilled 数据由 base model 自主生成，确保分布一致性。

### 实验设置和评估指标
#### 模型
- **Dream-7B-Instruct**
- **LLaDA-8B-Instruct**

#### 训练配置
- 使用 **LoRA**（rank=16, α=16）进行参数高效微调
- 学习率：2e-5，cosine decay，warm-up 50 步
- batch size: 32（8 GPUs × 4）
- 训练轮数：5 epochs
- TABOM 参数：窗口大小 $W=32$，margin $\gamma \in \{0.1,0.2,0.3\}$，ranking loss weight $\lambda \in \{1,2\}$

#### 评估任务与指标
| 类别 | 任务 | 指标 |
|------|------|------|
| 数学推理 | GSM8K, MATH500 | 准确率（Accuracy） |
| 代码生成 | HumanEval, MBPP | Pass@1 |
| 指令遵循 | IFEval | 指令准确率 |

同时报告 **in-domain** 和 **out-of-distribution (OOD)** 表现。

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **No-SFT** | 原始 DLM，无任何微调 |
| **SFT-GT** | 使用离线真实标签（ground-truth）进行标准 SFT |
| **SFT-SD** | 使用 self-distilled 轨迹进行 SFT |
| **dInfer** | 学习压缩过渡路径以加速推理 |
| **T3D** | 使用直接判别优化（DDO）进行轨迹自蒸馏 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Dream-7B-Instruct）

#### 表 2：代码生成领域微调结果（平均 in-domain +5.15%，OOD +0.60%）
| Method | HumanEval↑ | MBPP↑ | Avg.↑ | GSM8K↑ | MATH500↑ | IFEval↑ | OOD Avg.↑ |
|--------|------------|-------|--------|--------|----------|---------|-----------|
| No-SFT | 52.66 | 58.00 | 55.33 | 81.41 | 39.80 | 56.56 | 59.26 |
| SFT-GT | 61.55 (+8.89) | 58.00 | **59.78** | 52.33 (-29.08) | 32.40 | 46.21 | 43.65 |
| SFT-SD | 53.66 (+1.00) | 59.20 | 56.43 | 81.81 (+0.40) | 41.60 | 57.10 | 60.17 |
| **TABOM** | **60.36 (+7.70)** | **60.60 (+2.60)** | **60.48** | 81.73 (+0.32) | **42.40 (+2.60)** | 55.45 | **59.86** |

> ✅ TABOM 在保持 OOD 能力的同时，接近 SFT-GT 的 in-domain 表现，并全面超越 SFT-SD。

#### 表 3：数学推理领域微调结果（平均 in-domain +2.10%，OOD +2.24%）
| Method | GSM8K↑ | MATH500↑ | Avg.↑ | HumanEval↑ | MBPP↑ | IFEval↑ | OOD Avg.↑ |
|--------|--------|-----------|--------|------------|-------|---------|-----------|
| No-SFT | 81.41 | 39.80 | 60.61 | 52.66 | 58.00 | 56.56 | 55.74 |
| SFT-GT | 80.12 | 37.40 | 58.76 | 46.34 | 58.00 | 53.23 | 52.52 |
| SFT-SD | 81.95 | 39.80 | 60.88 | 57.92 | 58.60 | 56.01 | 57.51 |
| **TABOM** | **84.31 (+2.90)** | **41.10 (+1.30)** | **62.71** | **58.54 (+5.88)** | **59.20 (+1.20)** | **56.19** | **57.98** |

> ✅ TABOM 实现了 **in-domain 和 OOD 双向提升**，首次打破“增益 vs 忘记”的权衡困境。

### 消融实验结果（Table 5）
在 Dream 上对数学推理任务进行组件分析：

| 设置 | GSM8K | MATH500 | HumanEval | MBPP |
|------|--------|----------|------------|-------|
| SFT-SD (Base) | 81.95 | 39.80 | 57.92 | 58.60 |
| + Traj Masking only | 82.18 | 41.20 | 56.45 | 58.70 |
| + Pairwise Ranking (Global) | 83.10 | 40.20 | 57.50 | 58.20 |
| **+ Pairwise Ranking (Local, W=32)** | **84.31** | **41.10** | **58.54** | **59.20** |

> 🔍 发现：
> - 仅使用 trajectory-aware masking 提升有限；
> - 加入 pairwise ranking 显著提升性能；
> - **局部窗口（local window）优于全局比较**，避免跨阶段噪声干扰。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Self-distilled 轨迹本身具有更低的优化壁垒**（lower optimization barrier），但直接用 NELBO 微调只能带来边际收益。
2. **训练-推理不一致是根本瓶颈**：标准 SFT 引入 uniform inductive bias，破坏了推理所需的 easy-to-hard 结构。
3. **TABOM 成功对齐了模型的熵景观（entropy landscape）与推理轨迹**，使得模型在每一步都能更好地区分“易”与“难”token。
4. 提出的 **Trajectory Discrimination Score (TDS)** 定量验证了这一对齐效果：
   - Dream 上 MBPP 的 TDS 从 SFT-SD 的 0.138 提升至 TABOM 的 **0.929**
   - 表明 TABOM 真正重塑了不确定性分布，而非简单复用样本。

### 方法的局限性
- 当前 TABOM 依赖于高质量的 self-distilled 轨迹，若 base model 初始能力弱，则生成轨迹质量受限。
- 局部窗口设计虽有效，但仍需手动设定 $W$，尚未实现自适应调度。
- 当前未探索更复杂的能量函数形式（如引入上下文依赖）。

### 未来工作方向
- 将 TABOM 扩展至多模态 diffusion models。
- 探索动态窗口机制或 attention-based ranking 策略。
- 结合 RL 与 TABOM，进一步引导高阶推理结构的学习。
- 理论上深入研究 Boltzmann 分布与最优解码路径之间的关系。

---

> 💡 **一句话总结**：  
> TABOM 首次将 self-distilled 轨迹从“推理加速工具”转变为“知识获取媒介”，通过 **Boltzmann Modeling + Pairwise Ranking** 实现了 DLMs 在性能、泛化与稳定性上的全面突破，为下一代 diffusion-based LLMs 的训练范式提供了新方向。

</details>

---

### 3. [Generalization Bounds of Emergent Communications for Agentic AI Networking](https://arxiv.org/abs/2605.08613)

**Authors**: Yong Xiao, Jingxuan Chai, Guangming Shi, Ping Zhang  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2605.08613v1  

#### Abstract
The evolution of 6G networking toward agentic AI networking (AgentNet) systems requires a shift from traditional data pipelines to task-aware, agentic AI-native communication solutions. Emergent communication, a novel communication paradigm in which autonomous agents learn their own signaling protoc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Generalization Bounds of Emergent Communications for Agentic AI Networking

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统通信网络依赖**预定义、任务无关的协议**（如固定帧结构和信令流程），难以适应 Agentic AI Networking（AgentNet）中异构智能体之间动态、多模态、任务驱动的协作需求。现有 **emergent communication**（EC）框架虽然能自动生成通信协议，但普遍存在以下缺陷：
- 忽视物理层约束（如带宽、计算复杂度）
- 缺乏信息论基础，泛化性和鲁棒性差
- 决策模块与通信模块分离训练，导致冗余和不一致性

### 🚀 提出的新方法与创新思路
本文提出了一种**基于分布式信息瓶颈**（Distributed Information Bottleneck, DIB）的新型 emergent communication 框架，其核心创新包括：

1. **联合优化损失函数**（Joint Loss Function）  
   将 agent 的决策函数（decision-making function）与通信信号学习统一到一个目标中，避免模块割裂。

2. **多智能体多任务 DIB 理论建模**  
   引入两个关键互信息项进行权衡：
   - $ I(Y_k; C_{-k,k}) $：最大化通信信号对任务目标 $Y_k$ 的相关性（task relevance）
   - $ I(S_k; C_{k,-k}) $：最小化消息表示的复杂度（即 MDL，Minimum Description Length），控制通信开销和过拟合

3. **理论泛化界分析**（Generalization Bounds）  
   首次为 decentralized emergent communication 协议推导出在未见环境状态下的**去中心化推理泛化误差上界**，基于 Rényi divergence 和 sub-Gaussian 假设。

4. **信息-计算权衡的可量化性**  
   提供数学工具来衡量“保留多少任务相关信息” vs “压缩多少冗余信息”，实现资源受限下的最优表示学习。

### 🔍 相比现有方法的优势
| 维度 | 本文方法 | 现有主流方法（如 EC-SOTA） |
|------|--------|--------------------------|
| 架构设计 | 联合训练决策与通信模型 | 模块化独立训练（如 Autoencoder + Policy） |
| 理论支撑 | 基于 DIB 的信息论基础，具备泛化保证 | 多为启发式、实验驱动，缺乏理论保障 |
| 泛化能力 | 显著更低的 generalization error | 容易过拟合训练数据分布 |
| 通信效率 | 自动学习紧凑语义信号，降低带宽占用 | 可能传输高维潜变量，通信开销大 |
| 抗噪鲁棒性 | MDL 正则化抑制环境噪声干扰 | 易受观测噪声影响 |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **应用层数据**：采用公开的真实智能手机流量数据集 [15]，涵盖五类典型移动应用：
  - Live Streaming（直播）
  - Video Conferencing（视频会议）
  - Mobile Gaming（手游）
  - Web Browsing
  - Social Media
- 数据包含真实时间序列的行为模式、带宽波动和延迟敏感特征，贴近实际 6G 场景。

### ⚙️ 实验设置
- **原型系统搭建**：基于开源 RAN 和软件化 5G Core 构建硬件测试平台，包含：
  - UE（User Equipment）
  - gNodeB（gNB）
  - 5G Core（5GC）
- **双层 Agent 设计**：
  - **Application-layer Agent**：观察应用层流量，预测需求并发送信号
  - **Physical-layer Agent**：接收信号后动态调整无线资源配置（如调制编码策略、资源块分配）
- **训练方式**：深度学习驱动的 emergent communication，通过端到端训练学习通信协议
- **部署模式**：完全去中心化 inference，在 unseen environmental states 上测试泛化性能

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | 应用层 agent 成功完成任务的比例（如维持高清流畅播放） |
| **Generalization Error** | 训练损失与推理损失之间的差距 $\left|\mathcal{L}_{\text{pop}} - \mathcal{L}_{\text{emp}}\right|$ |
| **Convergence Speed** | 达到稳定低误差所需的迭代次数 |
| **Error Floor** | 收敛后的最小泛化误差水平 |

### 🆚 基线方法对比
- **EC-SOTA** [16]：当前最先进的基准方法，使用独立的 autoencoder 学习通信表示，再用于策略训练
  - 特点：模块分离、先学表示后做决策
  - 属于典型的两阶段 pipeline 架构

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Fig. 2 与 Fig. 3）

#### （1）准确率表现（Fig. 2）
- 在相同迭代轮数下（达 10k 次），本文方法的应用层 agent 准确率稳定在 **~85%–90%**
- EC-SOTA 最终仅达到约 **~75%–80%**
- 表明本方法能更有效地捕捉任务关键语义

#### （2）泛化误差比较（Fig. 3）
| 方法 | 泛化误差峰值 | 收敛速度 | 最终 error floor |
|------|-------------|---------|------------------|
| EC-SOTA | >15% | 缓慢（>8k iter） | ~8–10% |
| **本文方法** | <8% | 快速（<4k iter） | **~2–3%** |

- **优势显著**：本文方法不仅收敛更快，且最终泛化误差下降 **60%+**
- 不同应用场景（尤其是 Live Streaming 和 Mobile Gaming 等高动态场景）均表现出更强稳定性

#### （3）消融实验隐含结论（文中未明确列出表格，但从分析可得）
- 若移除 DIB 正则项（特别是 MDL 项），会出现“informational collapse”现象 —— 消息变得冗长且充满噪声
- 若不联合优化决策与通信，则存在“representation drift”风险 —— 通信含义随训练漂移，破坏语义一致性
- 使用 variational bounds 替代原始互信息项是可行且高效的近似方案

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **联合优化优于模块化设计**  
   将 decision-making 与 emergent communication 融合在一个 loss 中，显著提升性能与泛化能力。

2. **DIB 是有效的理论指导工具**  
   多智能体多任务场景下，DIB 成功实现了 task relevance 与 computational complexity 的平衡。

3. **MDL 正则化增强鲁棒性**  
   最小描述长度原则有效防止过拟合环境噪声，提升协议在未知状态下的稳定性。

4. **emergent protocol 可具备理论保证**  
   首次给出 decentralized emergent communication 的 generalization bounds，填补理论空白。

5. **真实硬件验证可行性**  
   在基于 5G RAN 的原型系统上成功部署，证明该框架具备向 6G AgentNet 落地的潜力。

### ⚠️ 方法的局限性
- 当前分析假设 loss 为 σ-sub-Gaussian，可能在极端非平稳环境中失效
- 多 agent 间 reward signal $Y_k$ 需预先定义或离线提供，限制了完全开放任务场景的应用
- 扩展至大规模 agent 网络时，Rényi divergence 的估计成本较高
- 当前实验集中在双 agent 场景，尚未验证超大规模组网性能

### 🔮 未来工作方向
1. **扩展至 hierarchical AgentNet 架构**，支持更多层级（MAC, Transport, Application）协同
2. **引入 causal representation learning**，进一步解耦任务相关因子与环境干扰
3. **在线 adaptation 机制**：允许在 inference 阶段持续更新 prior distribution $p(w)$
4. **结合 LLM-based agent**，探索生成式 AI 作为 emergent communicator 的潜力
5. **标准化 emergent protocol 接口**，推动其从 research concept 向 industry standard 演进

---

> **总结一句话**：  
> 本文首次将 **multi-agent multi-task DIB 理论** 引入 **Agentic AI Networking** 中的 emergent communication，提出了一个兼具**理论严谨性**与**工程实用性**的联合学习框架，并通过真实硬件原型验证了其在泛化性、收敛速度和鲁棒性上的全面领先。

</details>

---

### 4. [Agent-X: Full Pipeline Acceleration of On-device AI Agents](https://arxiv.org/abs/2605.10380)

**Authors**: Jinha Chung, Byeongjun Shin, Jiin Kim, Minsoo Rhu  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.10380v1  

#### Abstract
LLM-based agents deliver state-of-the-art performance across tasks but incur high end-to-end latency on edge devices. We introduce Agent-X, a software-only, accuracy-preserving framework that accelerates both the prefill and decode stages of on-device agent workloads. Agent-X's two key components re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Agent-X: Full Pipeline Acceleration of On-device AI Agents》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **On-device AI Agents**（基于大语言模型的本地智能体）在边缘设备上运行时面临显著的端到端延迟问题，尤其是在资源受限的环境下。
- 传统研究主要关注云侧LLM的**decode阶段优化**，而本文指出，在on-device agent场景中，**prefill 和 decode 阶段均构成性能瓶颈**，需要全流水线加速。

### **提出了什么新方法或新思路**
提出 **Agent-X** —— 一种纯软件、不损失准确性的端到端加速框架，包含两个核心组件：

#### **PromptWeaver**
- **目标**：加速 **prefill 阶段**
- **方法**：
  - 动态重构输入 prompt 结构，将原本动态插入的 `tool descriptions` 和 `guidelines` 改为静态预置，从而启用高效的 **prefix caching**。
  - 利用 **tool co-activation locality**（工具共激活局部性）进行聚类，并通过 **cluster combination selection** 算法选择高频组合，预先计算并缓存其 KV cache 到 SSD。
- **效果**：大幅减少运行时需重新计算的“uncacheable” token 数量。

#### **ExSpec**
- **目标**：加速 **decode 阶段**
- **方法**：
  - 引入一种 **LLM-free speculative decoding** 方案，使用基于 few-shot 示例构建的 **n-gram lookup table (LUT)** 作为轻量级 draft model。
  - 设计 **selective decoding** 机制：当 LUT 中无匹配上下文时，直接回退至 autoregressive 生成，避免无效验证带来的“multi-token tax”开销。
- **优势**：无需训练 draft LLM，内存占用极小（仅几KB），且规避了多token验证的性能惩罚。

### **相比现有方法的优势**
| 维度 | Agent-X 的优势 |
|------|----------------|
| **适用性** | 专为 on-device agent 定制，解决其特有的 prefill + decode 双重瓶颈 |
| **效率** | 不依赖额外硬件，纯软件方案可无缝集成现有系统（如 TinyAgent） |
| **准确性** | 保持任务精度不变（iso-accuracy） |
| **资源消耗** | ExSpec 使用 LUT 而非小型 LLM，节省数百MB内存；PromptWeaver 缓存存储于 SSD，不影响运行时内存 |
| **通用性** | 所提设计原则适用于其他具有类似 prompt 结构和输出模式的 agent 系统 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **TinyAgent-dataset** [68]：包含 1,022 个用于 fine-tuning 的测试样本，涵盖最多 16 种不同工具的任务请求。
- 数据用于：
  - PromptWeaver 的 tool clustering 与 combination selection
  - 端到端性能评估

### **实验设置**
- **模型**：
  - 后端 LLM：**TinyAgent-7B**（基于 WizardLM-2-7B 微调）
- **平台**：
  - 硬件：Apple Mac mini (M4 Pro)，64GB RAM，512GB SSD
  - 软件栈：MLX-LM [10] + MLX-engine [43]，macOS Sequoia 15.5
- **实现**：
  - Agent-X 基于 **TinyAgent [19]** 构建，集成 ToolRAG 流程
  - PromptWeaver 的 KV cache 存储于 SSD，按需加载
  - ExSpec 的 LUT 在每次查询时动态构建

### **评估指标**
| 指标 | 描述 |
|------|------|
| **End-to-end latency** | 单个 agent 任务从开始到完成的总时间 |
| **Prefill / Decode latency** | 分阶段延迟测量 |
| **Speedup** | 相对于 baseline 的加速比 |
| **Planner accuracy** | 输出计划 DAG 与 ground truth 匹配的准确率 |
| **KV cache reuse rate** | 缓存命中的 token 比例 |
| **Draft token accuracy** | speculative decoding 中被接受的 draft token 比例 |

### **基线方法对比**
- **Baseline**：原始 TinyAgent 实现
- **Static caching**：仅对完全静态部分启用 prefix caching
- **SpecDec**：使用小型 LLM（如 Llama-3.2-1B-Instruct）作为 draft model 的 speculative decoding
- **Ablation variants**：分别测试 PromptWeaver 和 ExSpec 的独立效果

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 模块 | 性能提升 |
|------|---------|
| **PromptWeaver** | 
| - Planner prefill 加速 | **1.57×** |
| - Arbiter prefill 加速 | **4.35×** |
| - 平均 uncacheable tokens 减少 | **70%**（从 1,711 → 519） |
| **ExSpec** |
| - Decode stage 加速 | **1.73×** |
| - draft token accuracy（selective） | Planner: **0.25**, Arbiter: **0.26** |
| - 回退 autoregressive 次数 | Planner: ~17次/查询，Arbiter: ~37次/查询 |
| **Agent-X（完整 pipeline）** |
| - 端到端平均加速 | **1.61×** |

> ✅ **无精度损失**：Planner accuracy 达 **0.841**（baseline: 0.836），甚至略有提升。

### **与基线方法的对比结果**
- **vs. Static caching**：
  - Static caching 仅带来 **1.01×** Planner 加速，因早期动态 token 阻碍缓存复用。
  - PromptWeaver 通过结构重组实现 **1.57×** 加速，显著优于传统 prefix caching。
  
- **vs. SpecDec（LLM-based speculative decoding）**：
  - SpecDec 在实际系统中出现 **性能倒退**（slowdown），主因是：
    - 多token验证引入 **multi-token tax**（验证延迟不成比例上升）
    - draft model 自身推理耗时高
  - ExSpec 避免上述问题，实现 **1.73× decode 加速**

### **消融实验结果**
| 配置 | End-to-end Speedup |
|------|--------------------|
| Baseline | 1.00× |
| + PromptWeaver (PW) | **1.16×** |
| + ExSpec (ES) | **1.43×** |
| + PW + ES (Agent-X) | **1.61×** |

- 表明两个模块具有**正交增益**，联合使用效果最佳。

#### **PromptWeaver 参数敏感性分析**
- **附加 tool-use examples 数量 K**：
  - K=0：accuracy 下降
  - K=1：accuracy 最高（0.841），为最优选择
  - K>1：accuracy 下降，cacheable token 比例下降
- **KV cache budget（cluster 数量）**：
  - 15 clusters 时达到 **74.4% 工具使用示例覆盖率**，存储开销仅 **6.26 GB**

#### **ExSpec 参数分析**
- **n-gram 大小 n**：
  - n=2（bigram）：draft accuracy 降至 0.10
  - n=3（trigram）：最佳平衡（accuracy=0.25）
  - n=4（quadgram）：accuracy 提升但生成 draft 更保守，最终 decode 更慢（-5.1%）
- **LUT 构建范围**：
  - 使用全部输入 vs. 仅 few-shot + query
  - 后者（ExSpec-few-shot）进一步提速 **3%（Planner）、1%（Arbiter）**

---

## **4. 关键结论和发现**

### **主要发现**
1. **On-device agent 的性能瓶颈不同于云端 LLM**：
   - 不再是 decode 单一主导，而是 **prefill 与 decode 共同构成瓶颈**。
   - 原因：长输入 prompt + 边缘设备算力/带宽受限。

2. **Agent-specific 特性可被高效利用**：
   - **Tool co-activation locality** 和 **few-shot template grounding** 是可预测性的来源。
   - PromptWeaver 和 ExSpec 正是基于这些语义规律设计。

3. **轻量级机制优于复杂模型复制**：
   - 使用 **n-gram LUT 替代 draft LLM** 可实现更高效率。
   - speculative decoding 的成功关键在于 **低开销 + 高命中率**，而非模型能力。

4. **纯软件方案即可实现显著加速**：
   - Agent-X 无需修改硬件或部署专用加速器，即可实现 **1.61× 端到端加速**，具备强实用性。

### **方法的局限性**
- **依赖 SSD 存储**：KV cache 预计算增加 SSD 占用（~6GB），可能影响低端设备。
- **Prompt 长度增加**：PromptWeaver 导致输入 token 数翻倍（1,739 → 3,790），轻微增加 decode 内存压力（+2.2% T/POT）。
- **领域迁移成本**：若工具集发生重大变更，需重新运行 clustering 与 combination selection 流程。
- **当前仅支持文本型 agent**：未覆盖视觉或多模态 agent 场景。

### **未来工作方向**
- **扩展至多模态 agent**：结合视觉 prompt 的 caching 与 speculative generation。
- **自适应 cache 管理**：根据设备负载动态调整 cache budget 与 eviction 策略。
- **跨用户个性化缓存**：支持用户行为驱动的 prompt pattern 学习与缓存定制。
- **探索更紧凑的 draft model 形式**：如 FSM-based generator 或 rule-based fallback。
- **集成量化与稀疏化技术**：与 DecDEC [55]、EdgeMoE [79] 等协同优化，进一步降低资源消耗。

---

> 🔚 **总结**：  
> Agent-X 是首个系统性分析并优化 on-device AI agent 全流程延迟的工作。它揭示了边缘 agent 的独特瓶颈，并提出 **PromptWeaver + ExSpec** 这一对称式轻量加速架构，在不牺牲精度的前提下实现 **1.61× 端到端加速**，为构建快速、私密、可用的本地 AI agent 提供了实用路径。

</details>

---

### 5. [BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion](https://arxiv.org/abs/2605.11577)

**Authors**: Shaobin Zhuang, Yuang Ai, Jiaming Han, Xiaohui Li, Huaibo Huang, Xiangyu Yue, Xuefeng Hu, Kun Xu, Yali Wang, Hao Chen  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.11577v1  

#### Abstract
Autoregressive language models generate text one token at a time, yet natural language is inherently structured in multi-token units, including phrases, n-grams, and collocations that carry meaning jointly. This one-token bottleneck limits both the expressiveness of the model during pre-training and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# BitLM: Unlocking Multi-Token Language Generation with Bitwise Continuous Diffusion  
—— 核心结论与实验结果总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **autoregressive language model**（AR LLM）采用“逐 token 生成”范式，即通过 `softmax` 在大词汇表上预测下一个 token ID。这种模式存在两个根本瓶颈：
- **表达能力受限**：语言本质上是多 token 单位（如短语、n-gram）共同承载语义，但模型被迫进行原子化、独立的 token 决策。
- **推理效率低下**：生成过程高度串行，限制了吞吐量。

尽管已有工作尝试缓解该问题（如 speculative decoding、multi-token prediction），但大多仍依赖于 `vocabulary softmax` 接口，未能改变输出空间的本质几何结构。

---

### 🚀 提出的新方法与核心思想
论文提出 **BitLM**，一种全新的语言建模框架，其核心创新在于：
- **将 token 表示为固定长度的二进制码（binary code）**
  - 每个 token ID 映射为一个 B-bit 的 ±1 向量（例如 B=18），位于超立方体顶点。
- **用 bitwise diffusion 替代 vocabulary softmax**
  - 不再直接分类 token，而是对未来的多个 token 的 binary codes 进行联合去噪。
- **引入 block-causal 因果结构**
  - 在 block 内允许全连接注意力，在 block 间保持 left-to-right 因果依赖，实现块内并行、块间有序。

> 🔁 生成不再是“选择下一个 token”，而是“逐步结晶出一段离散符号”的迭代过程。

---

### ⚖️ 相比现有方法的优势
| 维度 | 传统方法（如 Speculative Decoding） | BitLM |
|------|-------------------------------|-------|
| 并行性来源 | 外部解码策略（post-hoc） | 模型原生接口（native） |
| 输出空间 | Vocabulary-level categorical space | Bitwise continuous space |
| 因果结构 | 保留 AR 分布（目标一致） | 改变参数化方式，不追求模仿 AR 分布 |
| 联合建模能力 | 多 token 预测仍为独立分类 | 块内 token 和 bit 可联合建模 |
| 扩展性 | 加速有限，需额外小模型 | 更高效的训练与推理潜力 |

> 💡 BitLM 将“并行生成”从**加速技巧**转变为**生成范式的自然结果**。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **预训练数据**：FineWeb 子集（350B tokens）
  - 来自公开网页文本，广泛用于现代 LLM 预训练。
- **微调任务**：XSum（新闻摘要数据集）
  - 包含约 20 万条 BBC 新闻及其单句摘要，评估生成质量。

---

### ⚙️ 实验设置
- **模型架构基础**：
  - **Backbone**：基于 Qwen-3 架构（Transformer）
  - **Diffusion Head**：借鉴 BitDance 设计，轻量级 denoiser
- **关键参数**：
  - Block size $ m = 4 $
  - Binary code length $ B = 18 $（覆盖常见 tokenizer 空间）
  - Denoising steps $ K = 15 $
  - Classifier-Free Guidance (CFG) = 9.0
- **训练细节**：
  - 使用 AdamW 优化器（lr=1e-4, β1=0.9, β2=0.95）
  - 序列打包至 16384 tokens / sample，提升训练效率
  - 损失函数：L2 loss on predicted clean embeddings
- **推理流程**：
  - 使用 ODE solver 模拟 diffusion 轨迹
  - 每个 block 经过 K 步迭代 denoising 后取 `sign()` 投影回 binary space
  - 解码为 token ID 并更新 KV Cache

---

### 🧪 评估指标与基线对比
#### 主要指标
- **ROUGE-1 / ROUGE-2 / ROUGE-L**：衡量生成摘要与参考之间的 n-gram 重叠程度（标准化后报告）

#### 对比基线
| 类型 | 方法 |
|------|------|
| 强基线 | Lead-3, Pointer-Generator (PTGEN) |
| 模型变体 | BitLM w/ LM Head（softmax 版本）、BitLM w/ Diffusion Head |
| 训练阶段 | Pretrained (PT) vs Fine-tuned (FT) |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（XSum 测试集）

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|--------|---------|---------|---------|
| ILead-3 | 16.30 | 1.60 | 11.95 |
| PTGEN (See et al., 2017) | 29.70 | 9.21 | 23.24 |
| BitLM 8B w/ LM Head (FT) | 23.20 | 4.45 | 18.04 |
| **BitLM 8B w/ Diff. Head (FT)** | **26.05** | **6.44** | **20.12** |

> ✅ 微调后的 BitLM 在所有 ROUGE 指标上显著优于自身 softmax 版本，且超过未加 coverage mechanism 的 PTGEN。

---

### 🔍 消融实验分析（见 Figure 4）
- **Denoising Steps 影响**：
  - 性能随步数增加先升后稳，$ K=15 $ 达到最优
- **Classifier-Free Guidance 影响**：
  - CFG=9.0 时性能最佳，过高会导致过度锐化与重复
- **结论**：适当的 diffusion 控制策略可有效提升生成质量

---

### 📈 可扩展性验证（Figure 3）
- 成功预训练了 **0.6B, 1.7B, 4B, 8B** 四种规模的 BitLM
- 随着模型增大，pretraining loss 持续下降，表明：
  - **BitLM 具有良好的可扩展性**
  - 无需特殊设计即可适配主流架构

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **One-token-at-a-time 不是必须的**  
   语言生成可以脱离 `vocabulary softmax` 接口，转而以 **bitwise diffusion** 实现更灵活的结构化输出。

2. **并行生成可以是模型原生属性**  
   通过 block-causal + diffusion head 的设计，**multi-token 联合生成成为模型内在机制**，而非外部加速手段。

3. **输出空间几何影响生成行为**  
   从 simplex 上的分类变为 hypercube 上的连续去噪，改变了生成的动态路径，支持迭代 refinement。

4. **可行性已初步验证**  
   BitLM 可在大规模数据上成功预训练，并通过微调完成下游任务（如 XSum），说明 binary-space 语言建模是可行路径。

---

### ⚠️ 局限性
- 当前在 XSum 上的表现仍低于最强 pointer-generator 模型（如 PTGEN+COV）
- 缺乏精确 copy 或 alignment 机制，难以处理高保真词汇复制任务
- Binary code 为固定映射，未探索 learned binary representations
- 推理延迟尚未量化，并行优势需进一步 benchmark 验证

---

### 🔮 未来工作方向
1. **Learned Binary Codebooks**  
   探索可学习的 binary 编码方案，可能提升表示效率与语义一致性。

2. **Adaptive Block Sizes**  
   动态调整 block size，平衡流畅性与并行度。

3. **Hybrid Architectures**  
   结合 softmax 与 diffusion head，兼顾局部精度与全局并行。

4. **Efficiency Benchmarking**  
   定量测量推理速度与吞吐量提升，验证实际部署价值。

5. **多模态扩展**  
   利用统一 binary interface 支持 text-image-audio 联合建模（参考 UniWeTok）。

---

## 📌 总结一句话
> **BitLM 提出了一种范式转变：将语言生成从“逐 token 分类”重构为“基于 binary space 的块级扩散结晶”，在保留因果推理能力的同时，原生支持多 token 并行生成，揭示了输出空间几何设计作为新型建模维度的巨大潜力。**

</details>

---

### 6. [Efficient LLM-based Advertising via Model Compression and Parallel Verification](https://arxiv.org/abs/2605.11582)

**Authors**: Wenxin Dong, Chang Gao, Guanghui Yu, Xuewu Jiao, Mingqing Hu, Qiang Fu, Peng Xu, Penghui Wei, Hui Xu, Yue Xing, Shuanglong Li, Lin Liu  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.11582v1  

#### Abstract
Large language models (LLMs) have shown remarkable potential in advertising scenarios such as ad creative generation and targeted advertising. However, deploying LLMs in real-time advertising systems poses significant challenges due to their high inference latency and computational cost. In this pap...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Efficient LLM-based Advertising via Model Compression and Parallel Verification》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**大语言模型（LLM）在在线广告场景中推理延迟高、计算成本大**的问题，提出了一套高效的生成式广告投放框架。传统方法虽然在推荐精度上表现优异，但在实时性要求高的商业场景（如广告创意生成和定向投放）中面临严重的**可扩展性-效率困境（scalability-efficiency dilemma）**。

### 提出的新方法与新思路
作者提出了一个名为 **Efficient Generative Targeting** 的综合优化框架，结合了**模型压缩**与**前缀树并行验证（Prefix Tree Parallel Verification, PTPV）** 两大核心技术：

#### 主要创新点：
- **Index-Compressed 2bit-CSR 数据结构**  
  改进传统的 Compressed Sparse Row (CSR)，通过二进制编码压缩索引，将索引和权重大小减少至原始 CSR 的 30%，显著降低内存带宽开销。

- **自适应分组量化（Adaptive Group-Wise Quantization）**  
  根据不同线性层对量化误差的敏感度，动态调整分组粒度：敏感层采用细粒度（更多组）、非敏感层采用粗粒度（更少组），实现从 FP16 到 INT4 的高效压缩，在保持精度的同时提升推理速度。

- **层自适应半结构化稀疏化（Layer-Adaptive Semi-Structured Sparsity）**  
  对 Transformer 各层应用不同的 N:M 稀疏策略（如关键层保留 2:4 密度，次要层使用 1:4），兼顾模型质量与加速效果。

- **定制化 SparseGemv 加速内核**  
  自主开发支持 INT4 权重仅量化（weight-only）和混合稀疏矩阵乘法的 CUDA 内核，填补了 NVIDIA cuSparse/cuSparseLT 在 GEMV 场景下的空白。

- **前缀树约束的并行验证机制（Prefix Tree-based Parallel Verification）**  
  首次将**层次聚类算法**用于构建语义结构化的前缀树（Trie），并在解码过程中动态判断启动并行验证的最佳时机，结合 **tree-based speculative decoding** 和 **beam search** 实现一步解码剩余序列，大幅缩短解码步数。

- **完整的工作流整合**  
  据作者所知，这是首个将 **prefix tree-constrained decoding 与 beam search 完整结合** 并应用于广告生成任务的研究。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 推理速度提升超过 78%，实际部署中达到 **1.8× 以上加速** |
| **精度保留** | 在 Recall、BLEU、Meteor 等指标上损失极小，优于纯剪枝或量化方法 |
| **硬件适配性** | 定制内核专为 GEMV 工作负载优化，适用于主流 GPU（A10/A30） |
| **工业实用性** | 已在百度广告平台上线，处理真实大规模流量 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **定向广告（Targeted Advertising）**：公司内部收集的商业流量私有数据集（未公开）
- **广告创意生成（Ad Creative Generation）**：使用公开学术数据集 **CSL**（Chinese Scientific Literature Dataset），包含约 39.6 万篇中文论文元数据

### 实验设置
- **基础模型**：ERNIE 1.5B（百度自研 LLM）
- **框架实现**：基于 PaddlePaddle 构建
- **硬件环境**：
  - 广告创意生成：NVIDIA A10 GPU，beam size = 1
  - 定向广告：NVIDIA A30 GPU，beam size = 20（因需筛选更多候选广告）
- **任务类型**：
  - 广告创意生成：关键词摘要、文案重写
  - 定向广告：基于用户查询实时生成相关广告

### 评估指标
| 任务 | 主要指标 |
|------|----------|
| 定向广告 | **Latency（延迟）**, **Recall（召回率）** |
| 广告创意生成 | **BLEU**, **Meteor**, **AvgLen（平均长度）**, **Per-token latency（每 token 延迟）** |

### 基线方法对比
- **Baseline (FP16)**：全精度浮点模型（无压缩）
- **Quantization**：仅进行 INT4 分组量化
- **Sparse + Quant**：稀疏化 + 量化组合
- **Sparse(2:4)** / **Sparse(1:4)**：不同程度的 N:M 剪枝
- **Sparse(Mix) + Quant**：混合稀疏比例 + 量化（最终方案）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自图2、图3及表1）

#### 定向广告场景（Fig. 2）
| 方法 | Recall | Per-token Latency (ms) | 相对加速比 |
|------|--------|-------------------------|------------|
| Baseline (FP16) | 60.0% | 6.6 | ×1.00 |
| Quantization | 59.8% | 4.8 | ×1.37 |
| Sparse + Quant | 59.5% | 4.0 | ×1.65 |
| **Sparse + Quant + PTPV（完整方案）** | **60.0%** | **3.7** | **×1.78** |

> ✅ 在几乎不牺牲 Recall 的前提下，实现 **1.78× 的端到端加速**

#### 广告创意生成（Fig. 3）
- 完整方案在 BLEU 和 Meteor 上仅有轻微下降（<3%），但每 token 延迟显著降低
- **Hybrid 方法（Sparse + Quant）实现了最佳效率-质量权衡**

#### 消融实验（Table 1）
| 技术 | Latency (ms) | Meteor | BLEU | Speedup |
|------|---------------|--------|-------|---------|
| Baseline (FP16) | 6.6 | 0.6345 | 0.4247 | ×1.00 |
| Quantization | 4.8 | 0.6283 | 0.4178 | ×1.37 |
| Sparsification (2:4) | 5.3 | 0.6260 | 0.4161 | ×1.25 |
| Sparsification (1:4) | 4.6 | 0.5549 | 0.3476 | ×1.43 |
| Sparse(2:4)+Quant | 4.0 | 0.6195 | 0.4103 | ×1.65 |
| Sparse(1:4)+Quant | 3.5 | 0.5446 | 0.3369 | ×1.89 |
| **Sparse(Mix)+Quant** | **3.7** | **0.6127** | **0.4038** | **×1.78** |

> 🔍 发现：
> - 量化带来显著加速且精度损失小
> - 轻度稀疏（2:4）效率增益有限；重度稀疏（1:4）虽快但质量下降明显
> - **混合稀疏 + 量化 + PTPV 是最优配置**

---

## 4. 关键结论和发现

### 主要发现
1. **模型压缩与解码优化必须协同设计**：单独使用量化或稀疏化难以满足工业级实时需求，只有系统级整合才能突破瓶颈。
2. **稀疏程度需按层自适应调节**：Transformer 不同层的重要性差异显著，统一剪枝会损害关键信息。
3. **前缀树结构能有效引导并行解码**：通过语义聚类构建 Trie，可在保证输出合法性的前提下极大提升 beam search 效率。
4. **动态触发并行验证时机至关重要**：过早或过晚启动都会导致资源浪费或精度下降，应基于实时计算开销与收益权衡决定。

### 方法的局限性
- 当前优化策略高度依赖**特定业务模式**（如广告 ID 结构、高频关键词分布），通用性受限
- 所提方法主要面向**商业广告场景**，迁移到其他领域（如新闻推荐、对话系统）可能需要重新调优
- 前缀树构建依赖高质量文本标识符（如“Taobao”、“Baidu”），对噪声数据鲁棒性有待验证

### 未来工作方向
- 引入 **adaptive algorithms** 和 **reinforcement learning** 动态调整压缩策略与并行触发阈值
- 提升框架的**跨域适应能力**，探索更通用的稀疏化与量化范式
- 进一步优化 **memory access pattern** 和 **kernel fusion**，挖掘更低层级的硬件潜力
- 探索与 MoE（Mixture of Experts）架构的结合，实现更大规模模型的高效服务

--- 

> 📌 **总结一句话**：本文提出了一套面向工业级 LLM 广告系统的端到端优化方案，通过**自适应量化 + 层级稀疏 + 定制内核 + 前缀树并行验证**，在几乎不损精度的前提下实现 **1.8× 以上的推理加速**，并已成功部署于百度广告平台，具有极强的工程落地价值。

</details>

---

### 7. [Training-Inference Consistent Segmented Execution for Long-Context LLMs](https://arxiv.org/abs/2605.11744)

**Authors**: Xianpeng Shang, Jiang Li, Zehua Duo, Qianyi Cai, Xiangdong Su  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.11744v1  

#### Abstract
Transformer-based large language models face severe scalability challenges in long-context generation due to the computational and memory costs of full-context attention. Under practical computation and memory constraints, many inference-efficient long-context methods improve efficiency by adopting ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Training-Inference Consistent Segmented Execution for Long-Context LLMs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Transformer** 的 **Large Language Models (LLMs)** 在处理长上下文（long-context）生成任务时面临严重的可扩展性挑战，其根源在于 **full-context attention** 的计算和内存开销呈二次增长。为缓解这一问题，许多推理阶段采用 **bounded-context** 或 **segment-level execution**（如滑动窗口、chunked attention）等高效策略。

然而，这些方法通常在训练时仍使用 **full-context attention**，而在推理时切换为受限执行模式，导致 **training-inference mismatch**（训练-推理不一致）。这种不一致性体现在：
- 执行语义不同（execution semantics）
- 跨段状态演化（cross-segment state evolution）不一致

这可能导致模型依赖训练中存在但推理中不可用的信息，从而损害长上下文下的稳定性与泛化能力。

---

### 提出的新方法
作者提出了一种 **training-inference consistent segmented execution framework**，即**训练与推理一致的分段执行框架**，其核心思想是将分段执行作为共享的建模假设，而非仅在推理时优化。

#### 核心机制：
- 将输入序列划分为非重叠的 segments。
- 定义两个跨段输入接口：
  1. **Carried KV state (C)**：一个固定大小的 KV 尾部，是**唯一可微分的跨段状态**，用于局部连续性建模。
  2. **Retrieved KV prefix (R)**：从历史 KV 池中检索得到的前缀，以 **forward-only** 方式提供长距离上下文，**不参与梯度传播**。

#### 学习机制设计：
- 使用 **Truncated Backpropagation Through Time (TBPTT)**，限制梯度最多回传 $ K $ 个 segment（实验中 $ K=1 $ 最优），确保梯度路径与推理时的状态链完全对齐。
- 证明：在此设定下，TBPTT 计算的是 **inference-consistent objective 的精确梯度**，而非近似值。

#### 架构实现：
- **Head- and layer-sparse long-range retrieval**：仅在部分层（如 `{6,8,11,18}`）和部分头（prior-based selection）启用 long-range retrieval，其余保持 local computation。
- 引入 **RoPE re-indexing** 保证位置编码在拼接前缀时正确。

---

### 相比现有方法的优势
| 维度 | 传统方法（如 MInference, StreamingLLM） | 本文方法 |
|------|----------------------------------------|---------|
| **训练-推理一致性** | ❌ 不一致（训练 full-context，推理受限） | ✅ 严格一致 |
| **梯度传播** | 可能依赖推理中不存在的历史信息 | 仅通过 carried KV 回传，与推理一致 |
| **长程依赖建模** | 多数通过稀疏注意力或压缩 | 显式分离：短程（可微）+ 长程（前向检索） |
| **理论保障** | 无 | TBPTT 在约束条件下计算的是目标函数的**精确梯度** |

---

## 2. 核心实验方法和设置

### 数据集
- **PG19**：用于评估语言建模困惑度（PPL）随上下文长度变化的趋势。
- **LongBench**（Bai et al., 2024）：多任务长上下文理解基准，涵盖问答、摘要、代码、合成任务等。
  - 包括子集 **LongBench-E** 和完整版 **Standard LongBench**。
- **RULER**（Hsieh et al., 2024）：系统性测试长度外推能力的任务，包括：
  - **CWE**（Common Words Extraction）
  - **FWE**（Frequent Words Extraction）

### 实验设置
- **模型主干**：
  - 主要使用 **LLaMA2-7B-32K** 和 **LLaMA2-7B-80K**
  - 补充实验使用 **LLaMA3.1-8B-Instruct**
- **分段参数**：
  - Segment length $ S = 4096 $
  - Carried KV length $ M = 512 $
  - Retrieved prefix length $ R = 512 $
  - TBPTT depth $ K = 1 $（默认）
- **训练方式**：
  - 对齐方法（ours, CCA）进行 fine-tuning
  - 其他 baseline 使用预训练权重直接评测

### 评估指标
| 指标 | 描述 |
|------|------|
| **Perplexity (PPL)** | 语言建模稳定性 |
| **Average Score on LongBench** | 下游任务综合表现 |
| **Prefill Time (TTFT)** | Prompt 处理延迟（秒） |
| **Peak GPU Memory** | Prefill 阶段峰值显存占用（GB） |
| **Recall Accuracy on RULER** | 长度外推能力（4K → 64K） |

### 基线方法对比
| 方法 | 类型 | 是否训练-推理一致 |
|------|------|------------------|
| **Full Attention** | 全注意力参考 | ❌（资源不可扩展） |
| **MInference** | 推理时稀疏注意力 | ❌ |
| **StreamingLLM** | 滑动窗口 + sink tokens | ❌ |
| **DuoAttention** | 分离 streaming 与 retrieval heads | ❌ |
| **CCA (Core Context Aware)** | 压缩上下文训练 | ✅（有限对齐） |
| **Ours** | 分段执行 + 显式一致性 | ✅（严格对齐） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ **LongBench-E 平均得分（越高越好）**
| 方法 | LLaMA2-32K | LLaMA2-80K |
|------|------------|------------|
| Vanilla Self-Attention | 23.13 | 23.38 |
| StreamingLLM | 21.90 | 21.56 |
| CCA | 21.12 | 21.98 |
| DuoAttention | 23.00 | 22.94 |
| **Ours** | **23.24** | **24.17** ✅ |

> 在 80K 上达到 **最高平均分 24.17**，尤其在 **Summarization** 和 **Multi-document QA** 上显著领先。

#### ✅ **RULER 长度外推能力（Avg* on 4K–32K）**
| 方法 | CWE (↑) | FWE (↑) |
|------|--------|--------|
| Full Attention | 32.94 | 41.33 |
| StreamingLLM | 27.78 | 41.37 |
| DuoAttention | 32.33 | 43.42 |
| **Ours** | **46.39** ✅ | **43.88** ✅ |

> 在 64K 上多数方法崩溃（得分为“-”），而 ours 仍保留 **2.00 (CWE)** 和 **34.17 (FWE)** 的非零准确率，显示更强鲁棒性。

#### ✅ **Prefill 效率（LLaMA2-7B-32K @ 64K context）**
| 方法 | Peak Memory (↓) | Prefill Time (↓) |
|------|------------------|------------------|
| FlashAttention | ~70 GB | ~13.5 s |
| DuoAttention | ~60 GB | ~10.5 s |
| MInference | ~60 GB | ~9 s |
| **Ours** | **~19 GB** ✅ | **~7 s** ✅ |

> 较 full attention 实现 **约 6× 内存降低**（128K 时），并保持较低延迟，**latency-memory trade-off 最优**（见 Figure 6）。

---

### 消融实验结果

#### 🔍 **训练-推理一致性影响（Table 3 & G.3）**
| 方法 | LongBench-E Avg | PPL @ 64K |
|------|------------------|----------|
| **Aligned (TBPTT=1)** | 24.17 | 7.07 |
| **Misaligned**（训练 full，推理 segment） | 11.91 ↓ | 111.38 ↑ |

> 移除一致性导致性能**腰斩以上**，验证了训练-推理对齐的必要性。

#### 🔍 **TBPTT 深度 $ K $ 影响**
| $ K $ | Avg Score |
|-------|-----------|
| 1 | 24.17 ✅ |
| 2 | 24.07 ↓ |

> 更深的梯度回传并未提升性能，反而轻微下降，说明 **$ K=1 $ 已足够且最优**，符合“短程状态传递 + 长程前向检索”的设计理念。

#### 🔍 **局部状态容量（Local KV Size）**
| Size | PPL Avg | LongBench Avg |
|------|--------|--------------|
| 0 | 7.16 | 23.27 |
| 512 | 7.10 | 24.17 ✅ |
| 1024 | 7.07 | 24.19 |

> 引入局部 KV 显著提升性能，但进一步增大收益递减，表明适度容量即可。

#### 🔍 **长程模块层数与头分组**
- 增加 long-range layers 数量（0→4）显著提升下游任务表现（Avg 从 22.63→24.17），但对 PPL 几乎无影响，说明其作用在于**跨段推理而非语言建模本身**。
- **Prior-based head grouping**（基于先验选择 retrieval heads）优于 contiguous 或 interleaved 分组，说明应将 long-range 能力分配给具有 retrieval 特性的 heads。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **训练-推理一致性至关重要**：即使模型结构支持长上下文，若训练与推理执行语义不一致，会导致严重性能退化。
2. ✅ **TBPTT 可计算精确梯度**：在严格限制跨段递归的情况下，截断反向传播不再是近似，而是 inference-consistent objective 的**精确梯度估计**。
3. ✅ **$ K=1 $ 足够有效**：仅允许梯度回传一个 segment 即可获得最佳性能，说明模型更依赖局部连续性而非长期信用分配。
4. ✅ **显式分离短程与长程通道更优**：通过 carried KV（可微）和 retrieved KV（前向）分离建模，既保证效率又维持性能。
5. ✅ **显著提升可扩展性**：在 128K 上实现 **~6× 峰值内存降低**，使超长上下文部署成为可能。

---

### 方法的局限性
- **依赖 fine-tuning**：需要对预训练模型进行对齐训练，无法直接应用于 vanilla checkpoints。
- **长程检索开销随历史增长**：虽然 attention-visible context 有界，但 retrieval pool 的存储随序列增长线性增加（尽管稀疏化缓解）。
- **head/layer 配置需调优**：long-range head selection 和 layer placement 影响性能，目前依赖先验知识或搜索。

---

### 未来工作方向
- 将该框架扩展至 **encoder-decoder** 或 **multimodal models**（如处理百万 token 的 Gemini 场景）。
- 探索 **adaptive segment length** 或 **dynamic retrieval scope**。
- 结合 **quantization** 或 **offloading** 进一步降低部署成本。
- 研究如何自动识别适合 long-range 的 attention heads，减少人工先验依赖。

---

> 💡 **一句话总结**：  
> 本文提出了首个**严格训练-推理一致的分段执行框架**，通过分离可微局部状态与前向长程检索，在保持高性能的同时大幅降低长上下文 LLM 的内存与延迟，为超长上下文模型的高效、稳定部署提供了新范式。

</details>

---

### 8. [Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference](https://arxiv.org/abs/2605.11581)

**Authors**: Wenxin Dong, Mingqing Hu, Guanghui Yu, Qiang Fu, Peng Xu, Hui Xu, Yue Xing, Xuewu Jiao, Shuanglong Li, Lin Liu  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.11581v1  

#### Abstract
When large language models (LLMs) serve real-time inference in commercial online advertising systems, end-to-end latency must be strictly bounded to the millisecond range. Yet every token generated during the decode phase triggers thousands of kernel launches, and kernel launch overhead alone can ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference》核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在商业在线广告系统等**低延迟、高实时性**场景中，大语言模型（LLM）推理的端到端延迟必须控制在毫秒级。然而，传统推理框架在解码阶段（Decode）会触发数千次 Kernel Launch，导致 **Kernel Launch Overhead 占据端到端时间高达 14.6%**。此外，频繁的 HBM 内存访问也带来严重瓶颈。

MegaKernel 技术通过将多个算子融合进一个持久化 Kernel（Persistent Kernel），理论上可消除 Launch 开销和 HBM 往返延迟。但现有方案存在两大缺陷：

- **手写优化（如 Stanford MegaKernel）**：高度依赖特定硬件（如 Hopper/Blackwell 架构），缺乏对 NVIDIA Ada 架构（如 L20 GPU）的支持，且难以泛化到不同模型（如 Qwen）；
- **自动编译方案（如 Mirage MPK）**：引入运行时动态分支判断（如基于共享内存页状态的 `if-else`），破坏指令流水线，影响超低延迟场景下的性能。

此外，NVIDIA Ada 架构资源受限：
- 缺少 TMA 硬件支持，需软件模拟异步数据传输；
- 共享内存仅 128KB（H100 为 227KB），严重压缩流水线阶段和分块大小（tile size）的优化空间。

---

### **提出了什么新方法或新思路**

作者提出 **Ada-MK**，一种面向资源受限 GPU（NVIDIA Ada）的自适应 MegaKernel 优化框架，其三大核心创新如下：

#### **(1) 自适应共享内存管理（Adaptive Shared Memory Management）**
- 构建**三维共享内存约束模型**：综合考虑硬件规格（SMEM 容量）、模型架构（权重/激活大小）、动态负载（batch size）；
- 引入 **K-dimension 细粒度分裂**：将计算维度 K 分割，每轮仅加载所需权重子块，**峰值共享内存需求降低 50%**；
- 实现**跨算子页面复用**：
  - **Activation-Weight Page Reuse**：激活加载至寄存器后，释放其共享内存用于权重存储，提升流水线深度；
  - **Activation-Output Page Reuse**：激活内存释放后用于存储 MMA 输出，提高内存利用率。

#### **(2) 基于 MLIR 的细粒度 DAG 离线搜索（Fine-grained DAG-based Automatic Search）**
- 利用 **MLIR Lowering** 将高层算子分解为 PTX 级细粒度依赖图（DAG）；
- 基于别名分析（Alias Analysis）构建精确的 RAW（Read-After-Write）依赖关系；
- 在离线阶段进行**资源感知的自动搜索**，确定最优执行路径，并将其**固化（solidify）为静态代码**，**完全消除运行时分支决策开销**；
- 支持更细粒度的并行机会挖掘，优于 Ansor 等传统 Auto-Tuning 框架。

#### **(3) 异构混合推理引擎（Heterogeneous Hybrid Inference Engine）**
- 将 MegaKernel 作为插件嵌入 **TensorRT-LLM**，实现“**Prefill 用原生算子，Decode 用 MegaKernel**”的混合执行模式：
  - **Prefill 阶段**：长序列、大批量，计算密集型，TensorRT-LLM 更高效；
  - **Decode 阶段**：单 token 生成，IO 密集型，MegaKernel 可充分发挥优势；
- 复用 TensorRT-LLM 已有功能（如 Prefix-tree Decoding），避免业务重构成本。

---

### **相比现有方法的优势**

| 方面 | Ada-MK | Stanford MegaKernel | Mirage MPK |
|------|--------|---------------------|------------|
| **硬件兼容性** | ✅ 支持 Ada 架构（无 TMA） | ❌ 仅支持 Hopper/Blackwell | ✅ 支持通用架构 |
| **运行时开销** | ✅ 无运行时分支（离线固化路径） | ✅ 无运行时分支 | ❌ 存在 `if-else` 分支 |
| **模型泛化性** | ✅ 支持 Qwen 等多种模型 | ❌ 仅支持 Llama-1B 等少数模型 | ✅ 支持通用模型 |
| **部署成本** | ✅ 插件式集成，零业务改造 | ❌ 需重写整个推理流程 | ❌ 需适配新运行时 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **固定短序列任务**：`input=64`, `output=12`，模拟低延迟、短文本生成场景；
- **真实任务数据集**：
  - **CSL 数据集**：中等长度上下文输入（~200–1000 tokens），反映实际检索/推荐场景；
  - **Human-eval 数据集**：代码生成任务，测试复杂逻辑下的推理稳定性。

### **实验设置**

- **硬件平台**：单台服务器，配备 **NVIDIA L20 GPU**（Ada 架构，48GB GDDR6，SMEM=128KB）；
- **模型**：
  - Qwen3-1.7B（GPTQ-W4A16）
  - Qwen2.5-1.5B（GPTQ-W4A16）
- **批处理大小（Batch Size）**：1, 2, 4, 8, 16；
- **评估模式**：离线批量推理（offline batch mode），确保并发可控，排除调度干扰；
- **评估指标**：**生成吞吐量（Throughput, tokens/s）**，越高越好。

### **基线方法对比**

| 框架 | 版本 | 说明 |
|------|------|------|
| **vLLM** | v0.19.0 | 高吞吐推理框架，PagedAttention 优化 KV Cache |
| **SGLang** | v0.5.10 | 结构化生成高性能服务框架 |
| **TensorRT-LLM (vanilla)** | v1.1.0rc5 | NVIDIA 官方推理框架（基线） |
| **Ada-MK** | —— | 本文方法（TensorRT-LLM + MegaKernel 插件） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### **(1) 固定短序列（in64/out12）**

| 模型 | Batch Size | Ada-MK vs. TensorRT-LLM | Ada-MK vs. vLLM |
|------|------------|-------------------------|-----------------|
| Qwen3-1.7B | 1 | **↑23.6%** | ↑50.2% |
| Qwen2.5-1.5B | 1 | ↑15.6% | ↑64.5% |

> 🔹 Ada-MK 在所有 batch size 下均优于所有基线；
> 🔹 小 batch（BS=1/2）增益最显著，适合低延迟在线服务。

#### **(2) CSL 数据集（中等长度上下文）**

- BS=1~8：Ada-MK 吞吐最高；
- BS=16：**vLLM 超过 Ada-MK 3.5%**，表明在高并发长上下文下，系统级调度（如 KV Cache 管理）开始占据主导。

#### **(3) Human-eval 数据集（代码生成）**

- 即使在 BS=16，Ada-MK 仍保持最高吞吐；
- 相比 vanilla TensorRT-LLM 提升 **19.5%**；
- 相比 vLLM 和 SGLang 分别领先 **1.6%** 和 **3.5%**，显示其在复杂任务中的强扩展性。

---

### **消融实验结果（隐含在文中分析）**

虽然未设独立消融章节，但通过以下分析可推断各模块贡献：

| 优化项 | 性能增益来源 |
|--------|-------------|
| **K-dimension Splitting** | 共享内存峰值 ↓50%，允许更深流水线（从 2→4 stage） |
| **DAG-based Search + Path Solidification** | 消除运行时分支，提升指令发射效率；相比原始 MegaKernel Decode 阶段性能 ↑30% |
| **Warp Allocation Refinement** | Consumer Warps 从 16→8，Storer 与 Consumer 延迟对齐，减少流水线气泡 |
| **异构引擎设计** | Prefill 保留高吞吐能力，Decode 实现低延迟，整体端到端延迟 ↓10%–50% |

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **Ada-MK 是首个在商业在线广告系统中成功工业落地的 MegaKernel 方案**；
2. ✅ 在 **小 batch、短序列、低延迟场景**下，Ada-MK 显著优于 vLLM、SGLang 和 vanilla TensorRT-LLM，最大吞吐提升达 **23.6%**；
3. ✅ 通过 **离线 DAG 搜索 + 执行路径固化**，实现了“**编译时决策，运行时零开销**”，完美适配超低延迟要求；
4. ✅ **自适应共享内存管理**有效缓解 Ada 架构资源紧张问题，重建高效流水线；
5. ✅ **异构混合引擎**兼顾 Prefill 高吞吐与 Decode 低延迟，是 MegaKernel 工业化部署的可行路径。

---

### **方法的局限性**

1. **优势集中在 Decode 阶段**：Prefill 阶段因计算密集且 Launch 次数少，MegaKernel 改进有限；
2. **在高 batch + 长序列场景下优势缩小**：vLLM/SGLang 的系统级优化（如 KV Cache 调度）可能反超；
3. **依赖 MLIR 编译基础设施**：对编译器栈有一定门槛，部署复杂度高于纯 Kernel 优化；
4. **目前仅支持 GPTQ-W4A16 量化**，尚未覆盖更多量化格式（如 AWQ、FP8）。

---

### **未来工作方向**

1. **扩展至更大规模模型**：探索 Ada-MK 在 10B+ 模型上的适用性；
2. **迁移至下一代 Blackwell 架构**：结合 TMA 硬件支持，进一步释放潜力；
3. **支持更多量化方案**：集成 AWQ、FP8 等，提升通用性；
4. **动态负载适配**：探索在运行时根据请求特征自适应切换 MegaKernel 模式。

---

> **总结一句话**：  
> Ada-MK 通过 **“离线搜索 + 编译时固化”** 的范式革新，解决了 MegaKernel 在资源受限 GPU 上的部署难题，在保持工业级可用性的前提下，首次实现了其在商业低延迟场景中的大规模落地。

</details>

---

### 9. [GriNNder: Breaking the Memory Capacity Wall in Full-Graph GNN Training with Storage Offloading](https://arxiv.org/abs/2605.11517)

**Authors**: Jaeyong Song, Seongyeon Park, Hongsun Jang, Jaewon Jung, Hunseong Lim, Junguk Hong, Jinho Lee  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.11517v1  

#### Abstract
Full-graph training of graph neural networks (GNNs) is widely used as it enables direct validation of algorithmic improvements by preserving complete neighborhood information. However, it typically requires multiple GPUs or servers, incurring substantial hardware and inter-device communication costs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：GriNNder: Breaking the Memory Capacity Wall in Full-Graph GNN Training with Storage Offloading**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **Full-Graph GNN Training 的内存瓶颈**：全图训练（full-graph GNN training）虽然能保留完整的邻域信息，提升模型准确性和算法验证的可靠性，但需要在内存中存储所有节点在每一层的激活值（activations）和梯度（gradients），导致显存（GPU memory）和主机内存（host memory）迅速耗尽。
- **现有方法的局限性**：
  - **分布式训练**：依赖多GPU或多服务器，成本高且通信开销大。
  - **单机方法**：如 micro-batch 或 host memory offloading（如 HongTu），仍受限于 GPU 和 host memory 容量，且图划分（partitioning）本身可能超出内存限制。

### **提出了什么新方法或新思路**
提出 **GriNNder**，是首个利用 **存储设备（storage）作为额外内存层级** 来支持 full-graph GNN 训练的框架，其核心是 **Structured Storage Offloading (SSO)** 框架，包含三个协同机制：

1. **Partition-wise Graph Caching**  
   - 观察到跨分区依赖遵循幂律分布（power-law distribution），设计基于分区的缓存策略，在 host memory 中高效缓存中间数据，减少低效的随机存储访问。

2. **Grad-engine Activation Regathering**  
   - 引入“重聚集”（regathering）机制替代传统的 activation snapshot 存储，避免冗余的 I/O 和内存占用，显著降低存储流量。

3. **Switching-aware Partitioning**  
   - 设计轻量级、内存高效的图划分算法，避免传统划分器（如 METIS）在划分过程中消耗数百GB内存的问题。

### **相比现有方法的优势**
- **突破内存墙**：首次实现仅用单个 GPU 即可进行大规模 full-graph GNN 训练。
- **高性能**：训练速度最高可达 SOTA 基线的 **9.78× 加速**，吞吐量媲美分布式系统。
- **低成本**：无需昂贵的多GPU集群，适用于资源受限的研究环境。
- **兼容性强**：不修改训练算法本身，用户只需继承 `GriNNderGNN` 类即可迁移现有 PyG 代码。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 节点数 | 边数 | 类型 |
|--------|--------|------|------|
| **Products** | 2.4M | 61.9M | 商品共购网络 |
| **IGBM** | 10M | 120.1M | 引用网络 |
| **Papers** | 111M | 1.6B | 引用网络 |
| **Kronecker 合成图** | 4.2M–33.6M | — | 可扩展性测试 |

### **实验设置**
- **硬件配置**：
  - 主实验平台：单 GPU 工作站（RTX A5000, 24GB GPU, 128GB DDR5, PCIe 5.0 NVMe SSD）
  - 对比分布式系统：4台服务器 × 4×RTX A6000（共16 GPU），通过 Infiniband 连接
- **模型**：
  - GCN（3/5层）、GAT、GraphSAGE
  - 隐藏维度：256（默认）
- **评估指标**：
  - 每 epoch 训练时间（min）
  - 吞吐量（throughput）
  - 内存/显存使用量
  - 缓存命中率（cache hit rate）
  - SSD 写入总量（write volume）

### **基线方法对比**
| 基线方法 | 类型 | 特点 |
|----------|------|------|
| **Betty (Yang et al., 2023)** | Micro-batch | 批次级图划分，易因邻居爆炸（neighbor explosion）OOM |
| **Ginex (Park et al., 2022)** | 存储增强 mini-batch | 利用 SSD 缓存初始特征，但仍为 mini-batch |
| **HongTu (Wang et al., 2023a)** | Host memory offloading | 将激活值卸载至 host memory，但存在快照冗余 |
| **CAGNET (Tripathy et al., 2020)** | 分布式 full-graph | 多GPU训练，通信开销大 |
| **Sancus (Peng et al., 2022)** | 分布式 + 通信优化 | 使用过时激活减少通信，非精确 full-graph |
| **ROC (Jia et al., 2020a)** | Naive storage extension | 直接将 full-graph 扩展至存储，性能极差 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### **表1：不同方法在各数据集上的每 epoch 时间（分钟）**

| 方法 | PRODUCTS | IGBM | PAPERS |
|------|----------|------|--------|
| BETTY | 0.61 | 28.71 | GPU OOM |
| GINEX | 9.00 | GPU OOM | 17.72 |
| HONGTU | 0.17 | 6.46 | Swap OOM |
| **GRD (ours)** | **0.12** | **0.93** | **9.07** |
| CAGNET | 0.21 | 1.41 | 10.01* |
| SANCUS | 0.19 | 0.77* | GPU OOM |

> 注：`*` 表示使用 host memory checkpointing；Sancus 在 Papers 上无法运行。

- **GriNNder 在 IGBM 上比 HongTu 快 6.97×（3层） / 9.78×（5层）**
- **在 Papers 上成功训练而其他方法均 OOM**
- **在单 GPU 上性能优于 16-GPU 分布式系统（CAGNET）**

#### **合成图可扩展性测试（Table 2）**
| 图大小 | HongTu (5层) | GRD (5层) | Speedup |
|--------|-------------|-----------|---------|
| 4.2M | 0.83 | 0.57 | 1.46× |
| 8.4M | 1.99 | 1.14 | 1.75× |
| 16.8M | 19.15 | 3.71 | **5.16×** |
| 33.6M | 96.99 | 7.76 | **12.50×** |

> 显示 GriNNder 在更大图上优势更明显。

### **消融实验结果（Ablation Study）**
#### **表3：不同组件对性能的影响（IGBM, 5层）**

| 方法 | 隐藏维度=384 | 隐藏维度=512 | 隐藏维度=1024 |
|------|----------------|----------------|------------------|
| HONGTU | 25.07 | 31.81 | 93.42 |
| GRD-G (仅 regathering) | 10.26 | 12.50 | 42.14 |
| GRD-GC (完整方案) | **2.54** | **3.37** | **13.65** |

- **Grad-engine regathering 贡献约 2–3× 加速**
- **Partition-wise caching 在高维下贡献额外 3–4× 加速**
- 总体加速达 **12.34×**

#### **缓存命中率（Table 11）**
| 数据集 | Cache Hit Rate |
|--------|----------------|
| Products | 28.57% |
| IGBM | 53.70% |
| Papers | **83.63%** |

> 更大的图带来更高的重用率，缓存效率更高。

---

## **4. 关键结论和发现**

### **主要发现**
1. **存储可以成为有效的内存扩展手段**：现代 NVMe SSD（带宽 >10 GB/s，容量 TB 级）足以支撑 full-graph GNN 训练中的中间状态存储。
2. **传统 offloading 方法不可直接套用**：LLM 或 mini-batch GNN 的 offloading 技术无法解决 full-graph GNN 的图结构依赖和数据冗余问题。
3. **GriNNder 的 SSO 框架有效协调三级内存体系**（GPU-host-storage），通过 **caching**, **regathering**, **bypass** 实现高效训练。
4. **单 GPU 可媲美甚至超越分布式系统**：在通信受限环境下，GriNNder 的单机性能优于多机分布式训练。

### **方法的局限性**
- **依赖 NVMe SSD**：若使用低带宽 HDD 或 SATA SSD，性能会下降（见 Figure 13b，仍优于基线）。
- **最坏情况下的缓存失效**：当图依赖均匀分布在多个分区时，partition-wise 缓存可能产生额外开销（作者留作未来工作）。
- **未处理动态图**：当前针对静态图设计，动态图需扩展数据加载和划分模块。

### **未来工作方向**
- **结合稀疏性压缩**（sparsity-based compression）进一步减少写入量。
- **支持动态图更新**：利用 switching-aware partitioning 的流式特性适应增量变化。
- **集成 staleness 或 gradient compression**：与误差补偿类方法正交结合，进一步提升性能。
- **探索异构 GNN 支持**：已初步验证（Appendix R），可进一步优化。

---

> ✅ **总结一句话**：  
> **GriNNder 是首个打破 full-graph GNN 训练内存墙的存储卸载框架，通过 structured storage offloading 实现单 GPU 上的大规模训练，性能超越分布式系统，为资源受限场景提供了高效、低成本的解决方案。**

</details>

---

### 10. [GraphFlash: Enabling Fast and Elastic Graph Processing on Serverless Infrastructure](https://arxiv.org/abs/2605.11631)

**Authors**: Chen Zhao, Parsa Poorsistani, Mohammad Goudarzi, Tawfiq Islam, Adel N. Toosi  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.11631v1  

#### Abstract
Graph processing systems are essential for analyzing large-scale data with complex relationships, yet most existing frameworks rely on statically provisioned clusters, resulting in poor elasticity and inefficient resource utilization under dynamic workloads. Serverless computing offers automatic sca...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《GraphFlash: Enabling Fast and Elastic Graph Processing on Serverless Infrastructure》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统图处理系统（如 Giraph、GraphScope）依赖静态配置的集群，资源利用率低、弹性差，难以应对动态、突发或短时的图分析任务。而现有的 **serverless 图处理框架**（如 Graphless、FaaSGraph）存在以下问题：
- **性能瓶颈**：依赖外部存储进行状态管理，通信开销高；
- **违背 serverless 原则**：FaaSGraph 使用共享内存和代理机制，牺牲了无状态性和隔离性；
- **冷启动频繁、消息粒度过细**，导致延迟高、成本高。

### 提出的新方法与创新思路
作者提出 **GraphFlash** —— 一个完全基于 **serverless infrastructure** 构建的高性能、高弹性的图处理框架，其核心设计包括：

#### ✅ 子图中心化编程模型（subgraph-centric model）
- 以子图为计算单元，减少跨分区通信频率；
- 支持 GRAPE 引擎的 PEval/IncVal 接口，兼容主流算法开发。

#### ✅ 双执行模式自适应调度
- **Rotating Mode（旋转模式）**：适用于资源受限环境，多个子图由少量函数轮转处理；
- **Pinned Mode（绑定模式）**：当资源充足时，每个子图固定分配给一个函数，避免重复加载和冷启动。

#### ✅ 针对 serverless 瓶颈的系统级优化
| 优化技术 | 作用 |
|--------|------|
| **Partition-aware Key Aggregation** | 将顶点级消息聚合为子图级传输，将每 worker 的 key 访问从 O(v) 降为 O(p)，显著降低 I/O 开销 |
| **Intra-function Partition Co-location** | 单个函数内并发处理多个子图，共享边界顶点数据，减少冗余存储与网络传输 |
| **Superstep-aware Activation** | 在早期 superstep 不检查顶点活跃性，避免不必要的通信；后期开启激活机制提升效率 |
| **二进制序列化 + zstd 压缩 + Varint 编码** | 减少序列化开销与内存占用 |
| **Message Batching + Cilium CNI** | 优化容器网络接口，缓解 packet 处理瓶颈 |

### 相比现有方法的优势
| 维度 | GraphFlash | Graphless | FaaSGraph |
|------|------------|-----------|----------|
| Serverless 兼容性 | ✅ 完全符合 | ✅ 符合 | ❌ 违反（需共享内存） |
| 性能 | ⭐ 高 | ⚠️ 低 | ⭐ 高 |
| 成本效率 | ⭐ 高 | ⚠️ 低 | N/A（非纯 serverless） |
| 资源弹性 | ✅ 动态伸缩 | ✅ | ❌ 固定资源需求 |

> GraphFlash 是首个在保持 serverless 特性的同时实现接近传统分布式系统性能的图处理框架。

---

## 2. 核心实验方法和设置

### 数据集
使用多种真实与合成图数据，覆盖不同规模与结构特征：

| 数据集 | 类型 | 顶点数 | 边数 | 特点 |
|-------|------|--------|------|------|
| `dota-league (DL)` | 真实 | 61.1K | 50.9M | 小图，高密度 |
| `com-friendster (CF)` | 真实 | 65.6M | 1.81B | 大社交图 |
| `graph500-23~28 (G3–G8)` | 合成 | 4.6M ~ 121.2M | 129M ~ 4.23B | 可扩展性测试 |
| `datagen-9_2_zf (ZF)` | 合成 | 434.9M | 1.04B | 顶点密集型 |

### 图算法
- **BFS**（广度优先搜索）
- **PageRank**
- **CDLP**（标签传播社区检测）
- **WCC**（弱连通分量）

### 实验平台
- **本地部署**：基于 Knative 的 serverless 平台，4 节点集群（AMD EPYC 9474F, 128GB RAM, 25Gbps）
- **云部署**：AWS Lambda（x86 架构）
- **MaaS 层**：
  - Knative：Dragonfly（元数据）+ MinIO（数据）
  - AWS：S3 替代 MinIO

### 评估指标
- 执行时间（Execution Time）
- 资源消耗：Core·seconds、GB·seconds
- 成本（AWS 上按实际计费模型计算）
- 并发函数数量变化趋势
- 消融实验中的性能增益（Speedup）

### 基线方法对比
| 框架 | 类型 | 是否 Serverless |
|------|------|----------------|
| **GraphFlash**（pinned / rotating） | 本文方法 | ✅ |
| **Graphless** [13] | Serverless | ✅ |
| **FaaSGraph** [14] | 类 Serverless | ❌（依赖 co-location） |
| **GraphScope** [8] | 分布式系统 | ❌ |
| **Giraph** [5] | 传统 BSP 框架 | ❌ |

> 所有对比均在同一硬件环境下运行，确保公平性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Fig. 7 和 Table II）

#### 🔹 在小中等图上（DL, G3, G5, G7）的表现
- **GraphFlash（pinned mode）** 在所有算法上均优于 Graphless 和 Giraph；
- 对比 GraphScope，在小图上更快（因免去编译开销），在大图上性能相当；
- 最高达到 **127× 速度提升**（CDLP on DL vs Graphless）；
- 在 G7 上仍能与 GraphScope 持平，说明可扩展性强。

#### 🔹 在 AWS Lambda 上的大图表现（Table II）
| Dataset (Partitions) | BFS (s) | PageRank (s) | CDLP (s) | WCC (s) |
|----------------------|---------|--------------|----------|---------|
| CF (64)              | 102.95  | 154.14       | 281.94   | 126.02  |
| ZF (256)             | 226.46  | 328.18       | 321.48   | 241.83  |
| G8 (256)             | 167.07  | 311.61       | 321.48   | 199.87  |

> 表明 GraphFlash 可稳定处理超大规模图（如 4.23B 条边），且性能随并行度增加保持良好。

### 与基线方法的对比结果

#### 📊 执行时间对比（Fig. 7）
- GraphFlash 比 Graphless 快 **12× ~ 127×**；
- 比 Giraph 快数倍；
- 在大图上媲美 GraphScope 和 FaaSGraph。

#### 💰 成本效率对比（Fig. 8, Fig. 11）
- 在 DL 数据集上，GraphFlash 相比 Graphless：
  - **最多节省 98.8% 资源消耗**（PageRank）；
  - **成本降低高达 99.97%**（AWS Lambda 计费）；
  - 并发函数峰值更低，利用率更高。

#### ⏱️ AWS Lambda 上的速度优势（Fig. 10）
- 即使单个 Lambda 函数仅分配 0.4 vCPU，GraphFlash 仍实现：
  - **BFS 加速 9×**
  - **PageRank 加速超过 48×**

### 消融实验结果（Ablation Study）

#### ✂️ Partition-aware Key Aggregation（表 III）
| Dataset | Algorithm | Speedup |
|--------|-----------|--------|
| G5     | WCC       | **5.42×** |
| G5     | CDLP      | 4.06× |
| G3     | WCC       | 3.81× |

> 显示该优化对大图和通信密集型算法效果最明显。

#### ✂️ Intra-function Partition Co-location（表 IV & V）
- **执行时间加速 1.07× ~ 1.34×**
- **内存使用减少超过 50%**（G6 上从 52.3GB → 23.2GB）

> 证明多子图共置有效缓解内存压力，并提升吞吐。

#### ✂️ Superstep-aware Activation（表 VI）
- **WCC 最多提速 25%**（G6 上从 21.8s → 17.6s）
- 对 BFS/CDLP 也有约 5–10% 提升

> 表明智能激活策略可避免早期无效计算。

---

## 4. 关键结论和发现

### 主要发现
1. **Serverless 图处理可以既快又省**  
   GraphFlash 首次证明：通过合理的系统设计与优化，serverless 架构不仅能支持图处理，还能在性能上媲美传统分布式系统。

2. **双执行模式实现灵活弹性**  
   - **Rotating Mode** 支持极低资源配置下的高效运行；
   - **Pinned Mode** 在资源充足时释放极致性能；
   - 用户可根据预算和 SLA 自主选择。

3. **系统级优化至关重要**  
   - 细粒度消息聚合、内存复用、批处理等技巧是突破 serverless 性能瓶颈的关键；
   - 单纯移植传统模型无法获得理想收益。

4. **成本优势显著**  
   在相同功能下，GraphFlash 资源消耗仅为 Graphless 的 **1%~3%**，真正实现“按需付费”。

### 方法的局限性
- 当前仍依赖 **MaaS（外部存储）** 进行通信，存在 I/O 延迟；
- **冷启动问题虽被缓解但未根除**，尤其在旋转模式下；
- 目前未支持动态图更新或流式图处理；
- 对超大规模图（如 >100B 边）的极限性能尚未验证。

### 未来工作方向
- 设计支持 **function-to-function direct communication** 的 serverless 平台，绕过 MaaS 中转；
- 探索 **异构资源调度**（如 GPU + CPU 混合）；
- 支持 **dynamic graph processing** 与时序图分析；
- 进一步优化冷启动与预热机制（warm pool）；
- 构建端到端的 serverless 图分析流水线（含 ETL、查询、可视化）。

---

> ✅ **总结一句话**：  
> **GraphFlash 成功弥合了 serverless 的弹性优势与图处理的性能需求之间的鸿沟，是迈向“普惠化、自动化图分析”的重要一步。**

</details>

---

### 11. [CATS: Cascaded Adaptive Tree Speculation for Memory-Limited LLM Inference Acceleration](https://arxiv.org/abs/2605.11186)

**Authors**: Yuning Han, Yangchenchen Jin, Dylan Zhao, Jingwei Sun  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.11186v1  

#### Abstract
Auto-regressive decoding in Large Language Models (LLMs) is inherently memory-bound: every generation step requires loading the model weights and intermediate results from memory (e.g., High-Bandwidth Memory (HBM) for GPU servers), making throughput bottlenecked by memory bandwidth rather than compu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CATS: Cascaded Adaptive Tree Speculation for Memory-Limited LLM Inference Acceleration

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有 **speculative decoding** 方法（如 EAGLE、Medusa、Kangaroo）在服务器端表现优异，但其设计依赖于模型权重常驻 **High-Bandwidth Memory (HBM)** 的假设。然而，在 **memory-limited edge 设备**（如 Jetson AGX Orin）上，DRAM 容量不足以容纳完整的 LLM 模型（如 7B 参数模型需约 14GB），更不用说额外的 draft model。因此，模型权重必须从 **flash 存储逐块加载到 DRAM**，导致 **flash→DRAM 数据传输成为新的瓶颈**。

传统 speculative decoding 在这种场景下不仅无法加速，反而因引入额外参数和内存访问而 **性能下降**。

---

### 🚀 提出的新方法：CATS（Cascaded Adaptive Tree Speculation）
CATS 是一种专为 **memory-limited 设备** 设计的 **self-speculative decoding 框架**，通过三级级联验证机制最大化 token 接受率并减少总推理时间，同时 **不增加设备上的峰值内存占用**（仍等于目标模型本身）。

#### 核心思想：
将推理过程分为三个阶段，基于可用 DRAM 预算自适应划分层边界：

1. **Drafting Stage（草案生成）**
   - 使用浅层子网络（Layers 1 到 $L_{DM}$）+ 轻量级 adapter 进行多轮 auto-regressive 草案 token 生成。
   - 保持 draft model 浅以控制每步延迟和内存开销。

2. **Shallow Verification Stage（浅层验证）**
   - 将中间层（$L_{DM+1}$ 到 $L_{SV}$）也加载进 DRAM，与 draft model 一起 **一次性从 flash 加载**。
   - 并行验证 y 个草案 token，并对不匹配位置生成 **correction candidates**。
   - 构建包含主链与修正分支的 **verification tree**。

3. **Target Verification Stage（目标验证）**
   - 剩余深层（$L_{SV+1}$ 到 $L_{final}$）按需从 flash 分块加载。
   - 对 verification tree 执行 **batched forward pass**，选择最长接受前缀输出。

> 💡 **关键洞察**：利用一次 flash→DRAM 传输，完成 draft + shallow verify 两阶段计算，避免重复传输开销；并通过树形结构并行处理多个候选路径。

---

### 🔍 相比现有方法的优势

| 维度 | 传统 speculative decoding | CATS |
|------|----------------------------|------|
| 内存占用 | 需同时存储 target + draft model → 不适用于边缘设备 | 峰值内存 = target model 单独运行时，无需额外空间 |
| 数据移动 | 忽略 flash→DRAM 瓶颈，优化 HBM→SRAM | 显式建模 flash→DRAM 为瓶颈，最小化传输次数 |
| draft 能力 | 多数方法需额外 adapter 或小模型 | 自蒸馏 adapter + Reduced KL Loss 提升小模型 drafting 能力 |
| 验证效率 | 单一验证路径或固定结构 | 动态构建 verification tree，支持并行修正尝试 |
| 通用性 | 多数依赖特定架构 | 模型无关（model-agnostic），适用于任意 Transformer-based LLM |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与任务
在以下五个代表性基准上进行评估：
- **Spec-bench**：综合性能测试套件
- **MT-bench**：多轮对话质量评估
- **GSM8K**：数学推理能力
- **Alpaca**：指令跟随任务
- **HumanEval**：代码生成能力

---

### ⚙️ 实验设置
- **模型**：
  - Vicuna-7B / 13B
  - LLaMA2-7B / 13B
- **硬件平台**：
  - **Server**：NVIDIA B200 GPU（用于测量 mean accepted length）
  - **Edge Device**：NVIDIA Jetson AGX Orin（用于测量 end-to-end wall-clock speedup）
- **训练细节**：
  - Adapter 使用 ShareGPT 数据集微调
  - 学习率：Vicuna 系列 1e-6，LLaMA2 系列 1e-5
  - 训练耗时：<13 小时（2×B200）

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Mean Accepted Length (T)** | 每次 target model 验证平均接受的 token 数量（越高越好） |
| **End-to-End Speedup (S↑)** | 实际端到端 wall-clock 时间加速比（真实设备测得） |
| **Tokens/s** | 吞吐量 |
| **MT-Bench Score** | 由 GPT-4o 评判生成质量，验证是否牺牲质量换速度 |

---

### 🆚 基线方法对比
- **Chain-based baselines**：
  - REST（检索式草案）
  - Lookahead decoding
  - Kangaroo
  - Medusa
  - EAGLE-self（作者复现）
- **Tree-based baselines**：
  - EAGLE-self (tree)
  - CATS w/ EAGLE（CATS + EAGLE 的高概率候选分支增强）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & 2）

| 模型 | 方法 | Mean Acc. Len. | End-to-End Speedup |
|------|------|----------------|--------------------|
| Vicuna-7B | CATS (chain) | **3.06** | **3.18×** |
| Vicuna-7B | Kangaroo | 2.49 | 1.92× |
| Vicuna-7B | Medusa | 2.48 | 2.49× |
| Vicuna-7B | CATS w/ EAGLE (tree) | **3.71** | **3.71×** |
| LLaMA2-7B | CATS | **4.65** | **4.65×** |
| LLaMA2-7B | Kangaroo | 4.15 | 3.58× |
| LLaMA2-7B | CATS w/ EAGLE | **5.08** | **5.08×** ✅ |
| LLaMA2-13B | CATS w/ EAGLE | **4.81** | **4.75×** |

> ✅ **最高达到 5.08× 的端到端加速**，显著优于所有 baseline。

---

### 📊 与基线方法的对比结果
- CATS 在所有模型和任务上均 **全面超越 SOTA 方法**：
  - 相比 Kangaroo，平均提升 **1.45× 更高的 speedup**
  - 相比 Medusa 和 EAGLE，在 edge 上优势更加明显（因其额外参数加重 flash 传输负担）
- 在 **relaxed decoding** 设置下（允许一定偏差），CATS 依然保持高质量输出，且加速更强。

---

### 🔬 消融实验结果（Ablation Study）

#### (1) Memory Budget 分析（Table 3）
在不同 DRAM 预算（2GB / 6GB / 8GB）下，CATS 均表现出色：
- 即使在仅 2GB 可用内存下，仍可实现 **2.82× 加速**
- 通过调节 $L_{SV}$ 自适应利用内存资源，体现框架灵活性

#### (2) BPT（Bytes Per Token）分析（Table 4）
| 方法 | BPT (GB/tok) | Speedup |
|------|--------------|---------|
| Baseline | 12.95 | 1.00× |
| Kangaroo ($L_{DM}=3$) | 8.37 | 1.93× |
| Kangaroo ($L_{DM}=15$) | 11.10 | 1.53× ❌ |
| **CATS ($L_{DM}=3, L_{SV}=15$)** | **5.84** | **3.16×** ✅ |

> 深化 draft model 反而增加 BPT 和计算成本，降低实际加速；
> CATS 通过 shallow verification 提升接受率而不增加 drafting 开销。

#### (3) Drafting Steps ($y$) 分析（Table 5）
| $y$ | CATS Mean Acc. | BPT | Speedup |
|-----|----------------|-----|---------|
| 3 | 2.71 | 5.68 | 2.82× |
| 5 | 3.05 | 5.84 | 3.16× ✅ |
| 7 | 3.23 | 6.26 | 3.34× |
| 10 | 3.33 | 7.16 | 3.43× |

> $y=5$ 是最佳平衡点：进一步增加 drafting 步数带来的收益递减，但 BPT 持续上升。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Flash→DRAM 是边缘设备上的真正瓶颈**，现有 speculative decoding 方法未考虑此约束，导致在 edge 上性能退化。
2. **CATS 成功将 speculative decoding 适配至 memory-limited 场景**，提出三阶段级联验证机制，在不增加内存的前提下大幅提升 token 接受率。
3. **Shallow verification 比 deep drafting 更有效**：与其加深 draft model 增加重复传输开销，不如用中间层做一次并行验证。
4. **Reduced KL Loss 提升小模型能力**：聚焦 top-K 高概率 token 进行蒸馏，避免低概率 token 浪费模型容量。
5. **CATS 是通用框架**，可与其他方法（如 EAGLE）结合，进一步提升性能。

---

### ⚠️ 局限性
1. 当前评估集中在 **7B–13B 规模模型**，更大模型（如 70B）的表现尚待验证。
2. 需要 **预先训练 adapter**，带来一定的训练开销（尽管只需一次）。
3. 依赖 **Transformer 层对齐结构**，难以直接应用于非 Transformer 架构（如 State Space Models）。

---

### 🔮 未来工作方向
- 扩展至 **multi-modal models** 和 **larger-scale models**
- 探索 **zero-shot adapter adaptation**，减少训练依赖
- 结合 **KV cache offloading** 技术，进一步缓解内存压力
- 支持 **动态调整 $L_{DM}, L_{SV}$** 以应对不同输入长度和负载变化

---

> 🔗 **开源地址**：https://github.com/ElizaFuLan/CATS.git  
> 📄 **原文链接**：https://arxiv.org/abs/2605.11186

</details>

---

### 12. [Arcane: An Assertion Reduction Framework through Semantic Clustering and MCTS-Guided Rule Exploring](https://arxiv.org/abs/2605.10107)

**Authors**: Hongqin Lyu, Yonghao Wang, Zhiteng Chao, Tiancheng Wang, Huawei Li  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.10107v1  

#### Abstract
Assertion-based Verification (ABV) is essential for ensuring that hardware designs conform to their intended specifications. However, existing automated assertion-generation approaches, such as LLM-based frameworks, often generate large numbers of redundant assertions, which significantly degrade si...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Arcane: An Assertion Reduction Framework through Semantic Clustering and MCTS-Guided Rule Exploring

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题  
当前基于 **Assertion-based Verification (ABV)** 的硬件验证中，无论是传统方法还是基于 **LLM** 的自动化生成技术，都会产生大量**冗余断言（redundant assertions）**。这些冗余不仅增加了仿真开销（simulation overhead），还降低了验证效率，尤其是在大规模设计中尤为显著。

已有工作未能在**保持错误检测能力的前提下有效去除冗余断言**，而直接删除或简化可能破坏原始语义，导致覆盖率下降。

---

### 🚀 提出的新方法与创新思路  

本文提出 **Arcane** —— 一种高效的断言约简框架，其核心创新在于：

#### （1）**两阶段聚类机制（Coarse-to-Fine Clustering）**
- **第一层：BERT-guided 粗粒度语义分类**
  - 将形式化断言转换为自然语言描述，利用预训练 BERT 模型提取语义向量，通过余弦相似性进行初步分组。
- **第二层：Lasso-based 细粒度行为一致性分析**
  - 构建每个断言对应的 **Büchi Automaton**，采样 lasso 路径并分析接受行为，使用 **Jaccard Index** 衡量功能一致性，避免文本相似但逻辑冲突的误聚类（如 `A→B` vs `A→¬B`）。
  - 对于命题逻辑断言，则枚举所有真值赋值以提高效率。

> 💡 创新点：结合 **语义嵌入（semantic embedding）** 和 **自动机行为分析（acceptance behavior）** 实现高精度聚类，确保后续约简不跨语义类别干扰。

#### （2）**MCTS-Guided 规则探索机制**
- 将断言约简建模为一个 **确定性 MDP（Markov Decision Process）**：
  - **State**：当前断言集合的状态
  - **Action**：应用五种预定义的语义保持规则之一
  - **Transition Function**：执行规则后更新断言集
  - **Reward**：基于断言数量减少量（Δ|S|）和原子谓词减少量（Δ|AP|）
- 使用 **Monte Carlo Tree Search (MCTS)** 在庞大的规则组合空间中高效搜索最优约简路径，采用 UCT 策略平衡探索与利用。

> 💡 创新点：首次将 MCTS 引入断言约简任务，实现对复杂规则顺序依赖关系的有效优化，避免穷举搜索。

#### （3）**严格语义保持的约简规则体系**
定义了五个可组合使用的逻辑等价/蕴含保留规则：
1. 单断言前后件简化（Single-assertion pre/post reduction）
2. 同前提后件合取合并（Common-antecedent POST Conjunction）
3. 同结论前件析取合并（Common-consequent PRE Disjunction）
4. 断言间等价判定（Pairwise Equivalence）
5. 断言间蕴含判定（Pairwise Implication）

所有规则均保证 **逻辑等价或语义包含**，并通过 **CNF 蕴含检查** 或 **SPOT 工具中的 Büchi Automata 比较** 验证。

---

### 🔍 相比现有方法的优势  

| 方面 | Arcane 的优势 |
|------|----------------|
| **冗余处理能力** | 显著优于仅依赖语法或静态模式匹配的方法（如 HARM、GoldMine），能识别深层语义重复 |
| **语义保全性** | 不牺牲 formal coverage 和 mutation detection 能力，真正实现“安全约简” |
| **效率提升** | 减少高达 76.2% 的断言数，带来最高 **6.1x 的仿真加速** |
| **通用性与自动化** | 支持多种来源断言（LLM / trace-mining），无需人工干预 |

---

## 2. 核心实验方法和设置

### 📦 数据集  
使用公开基准 **AssertionBench [20]**，包含：
- **112 个硬件设计模块**
- 每个模块配有 RTL 代码、波形轨迹及对应断言
- 断言来自两类生成方式：
  - **HARM**：基于波形挖掘的传统方法
  - **LLM-based generator**：基于大模型生成的现代方法  
→ 构成异构、真实场景下的测试集

---

### ⚙️ 实验设置  

| 项目 | 设置说明 |
|------|----------|
| **形式验证工具** | Cadence JasperGold (v21.12.002) 进行 FPV 验证 |
| **仿真工具** | Synopsys VCS (v2016.06) 执行仿真 |
| **运行平台** | Intel Xeon Gold 6148 @ 2.40GHz, 629GB RAM |
| **超参数配置** |
| - Lasso 样本数 | 500 |
| - BERT/Lasso 权重 | α=0.4, β=0.6 |
| - 相似度阈值 | 0.85 |
| - 并行线程数 | 64 |

---

### 🎯 评估指标（Evaluation Metrics）

| 指标 | 缩写 | 含义 |
|------|------|------|
| 断言数量 | N | 原始与约简后的断言总数 |
| Proof Core | PC | 形式验证所需的最小逻辑单元，衡量 formal coverage |
| 错误检测率 | ER (Mutation Testing) | 变异测试中捕获缺陷的比例，反映实际验证质量 |
| 处理时间 | PT | Arcane 自身的运行耗时（一次性成本） |
| 仿真时间 | RT | VCS 仿真运行时间（反复执行，影响更大） |
| 聚类质量 | DBI (Davies-Bouldin Index) | 数值越低表示聚类效果越好 |

> ✅ 关键原则：约简后的断言必须保持 **PC 不变** 且 **ER 不下降**

---

### 🔁 基线方法对比  
本文未直接比较其他端到端断言约简系统（因无同类工作），而是从组件层面进行消融分析，并隐含对比以下策略：
- **纯 BERT 聚类** vs **BERT + Lasso 聚类**
- **随机规则应用** vs **MCTS-guided 规则序列探索**
- **全局聚类** vs **先 BERT 分区再局部 lasso 分析**

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table III）

| 设计模块 | 断言减少比例 | PC 保持？ | ER 保持？ | 仿真加速比 |
|--------|--------------|-----------|-----------|-------------|
| ca_prng | **76.2%** | ✅ 是 | ✅ 是 | **6.1x** |
| control_unit | 69.5% | ✅ 是 | ✅ 是 | 3.46x |
| eth_cop | 74.5% | ✅ 是 | ✅ 是 | 3.42x |
| eth_receivecontrol | 70.3% | ✅ 是 | ✅ 是 | 2.83x |
| MAC_rx_ctrl | 68.2% | ✅ 是 | ✅ 是 | 2.67x |
| MAC_tx_Ctrl | 71.9% | ✅ 是 | ✅ 是 | 3.68x |

> ✅ **全部案例中，PC 和 ER 完全不变**，证明语义完整性完全保留。

---

### 📈 整体分布表现（Figure 5）
- 在 112 个设计上：
  - HARM 生成断言：平均约简 **78%**
  - LLM 生成断言：平均约简 **71%**
- 所有设计至少减少 **68%**，表明框架具有高度稳定性。

---

### 🔍 消融实验结果（Ablation Study）

#### （1）聚类策略对比（Table IV）  
| 设计模块 | 方法 | 运行时间 | DBI（聚类质量） |
|--------|------|---------|----------------|
| MAC_tx_Ctrl | Lasso-only | 43321.48s (~12h) | 0.3419 |
| | **BERT + Lasso** | **1274.51s (~21min)** | **0.3626** |

> ⚠️ DBI 略微上升（<0.023），但运行时间降低 **34x**！  
✅ 说明 **BERT 预分类极大加速了 lasso 分析**，代价极小。

#### （2）聚类可视化（Figure 6）
- **仅用 BERT**：聚类边界模糊、存在明显重叠
- **BERT + Lasso**：簇更紧凑，边界清晰，内聚性强
→ 验证了双层聚类的有效性

---

### 🧪 MCTS 搜索有效性
- 初始化优先尝试 Rule 1（变量简化），加快早期收敛
- 设置早停条件：连续 3 次迭代 reward 不增即终止
- 最终选择累计 reward 最高的路径输出结果
→ 在合理时间内找到高质量约简方案，避免暴力搜索

---

## 4. 关键结论和发现

### ✅ 主要发现  

1. **断言冗余普遍存在**，即使 LLM 生成也仍有 20%-30% 冗余，亟需系统性约简手段。
2. **单纯语义相似性不足以支撑准确聚类**，必须结合行为级分析（如 Büchi acceptance）才能避免误合并。
3. **规则顺序显著影响最终约简效果**，MCTS 能有效导航复杂的组合空间。
4. **Arcane 可实现高达 76.2% 的断言压缩率**，同时 **完全保持 formal coverage 与 mutation detection 能力**。
5. **仿真时间最高提速 6.1x**，大幅降低验证周期成本。
6. **BERT 预分类 + 局部 lasso 分析** 是兼顾效率与精度的关键设计。

---

### ⚠️ 方法的局限性  

1. **依赖 LTL 可表达性**：目前只处理可转化为 LTL 的断言，对于高度非线性或复杂时序结构的支持有限。
2. **MCTS 参数敏感性**：探索权重 `c` 和迭代次数会影响结果稳定性，需调参。
3. **初始聚类误差传播风险**：若 BERT 初步分类严重错误，可能导致后续 lasso 分析失效。
4. **未支持增量式约简**：面对动态新增断言，需重新运行全流程。

---

### 🔮 未来工作方向  

1. **扩展至更多 assertion language 特性**：支持 SVA 中的 `local variables`, `sequence concatenation` 等高级特性。
2. **引入强化学习优化 MCTS 策略网络**：用 learned policy 替代随机 rollout，进一步提升搜索效率。
3. **支持在线/增量式断言管理**：适应 CI/CD 流水线中的持续验证需求。
4. **集成进 LLM 断言生成 pipeline**：作为 post-processing 模块，构建“生成-约简-部署”一体化流程。
5. **跨模块断言去重**：研究不同 IP 模块间的公共性质，实现系统级断言共享。

---

## 总结  

> **Arcane 是首个将语义聚类与 MCTS 搜索相结合的断言约简框架，在不影响验证质量的前提下实现了高效、安全的冗余消除。其实验结果充分证明了该方法在真实性、有效性与可扩展性方面的巨大潜力，为下一代智能验证基础设施提供了重要基础。**

</details>

---

### 13. [ReCoVer: Resilient LLM Pre-Training System via Fault-Tolerant Collective and Versatile Workload](https://arxiv.org/abs/2605.11215)

**Authors**: Ziyue Liu, Zhengyang Wang, Ruijie Zhang, Avinash Maurya, Hui Zhou, Paul Hovland, Sheng Di, Franck Cappello, Bogdan Nicolae, Zheng Zhang  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.11215v1  

#### Abstract
Pre-training large language models on massive GPU clusters has made hardware faults routine rather than rare, driving the need for resilient training systems. Yet existing frameworks either focus on specific parallelism schemes or risk drifting away from a failure-free training trajectory. We propos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**RECoVER: Resilient LLM Pre-Training System via Fault-Tolerant Collective and Versatile Workload**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在超大规模 GPU 集群（如 100k+ GPU）上进行大语言模型（LLM）预训练时，硬件故障变得极为频繁（Mean-Time-Between-Failures, MTBF 可低至几分钟）。传统的 **checkpoint-and-restart** 方法因重启开销巨大（需重新初始化通信后端、加载模型状态等），导致有效吞吐极低，超过 50% 的 GPU 小时被浪费。

此外，现有前向恢复（forward recovery）方法存在以下不足：
- **局限于通信层**：仅修复通信原语，无法保证整个训练栈的容错；
- **缺乏通用性**：依赖特定并行策略（如 pipeline-centric 设计），难以适配 HSDP 等现代并行方案；
- **破坏计算等价性**：失败后减少 microbatch 数量，改变梯度噪声尺度，导致训练轨迹漂移。

---

### ✅ 提出的新方法与核心思想

作者提出 **RECoVER**，一个面向 LLM 预训练的弹性容错系统，其核心是通过三层解耦协议实现“**每轮迭代保持全局 batch 大小不变**”这一单一不变量（invariant），从而确保训练轨迹与无故障运行**随机等价**（stochastically equivalent）。

#### 三大创新层设计：

| 层级 | 功能 | 技术要点 |
|------|------|---------|
| **底层：Fault-Tolerant Collectives** | 隔离故障传播 | 基于 **ULFM** 构建 `ULFM_ALLREDUCE` 和 `ULFM_CONSENSUS`，支持在 rank 失效后就地修复通信组，避免作业中断。 |
| **中层：In-Step Fine-Grained Recovery** | 保留迭代内进度 | 在失败发生时，对已部分归约的梯度桶（gradient bucket）进行快照回滚与重归约，防止梯度污染，无需回滚或重放。 |
| **顶层：Versatile-Workload Policy** | 动态负载再分配 | 故障后动态调整幸存副本（survivor）的工作负载（microbatch 数量），通过引入 **spare replica** 和 **boundary minor** 角色，确保总 microbatch 数恒为 `B = W_init × G_init`。 |

> 💡 **关键洞察**：只要至少有一个副本存活，每个 iteration 聚合相同数量的 microbatch 梯度，则优化路径在统计意义上与无故障运行一致。

---

### ✅ 相比现有方法的优势

| 维度 | RECoVER | 现有方法（如 FTAR、Drop-and-go） |
|------|--------|-------------------------------|
| **计算等价性** | ✅ 严格保持 | ❌ 全局 batch 缩减，导致 loss spike 和轨迹偏移 |
| **通用性** | ✅ 支持 3D Parallelism 和 HSDP | ❌ 多数仅支持特定并行模式 |
| **资源效率** | ✅ 无预分配空闲副本，不浪费 GPU 小时 | ❌ Hot-spare 方案冗余高，成本昂贵 |
| **恢复机制** | ✅ 前向恢复，不停机 | ❌ Checkpoint-Restart 需重启，停机时间长 |

---

## 2. 核心实验方法和设置

### ✅ 数据集
- 使用 **C4 dataset** 进行 LLM 预训练任务。
- 模型：LLaMA-style 模型（7B 参数用于 3D 并行实验，1B 参数用于 HSDP 实验）。

### ✅ 实验设置

| 项目 | 设置详情 |
|------|----------|
| **硬件平台** | 最多使用 512 张 NVIDIA A100 40GB GPU（128 节点） |
| **并行策略** | 支持两种主流并行方式：<br>• **3D Parallelism**（TP×PP×DP）<br>• **Hybrid Sharded Data Parallel (HSDP)** |
| **初始配置** | • 初始副本数 $W_{\text{init}} = 64$<br>• 初始梯度累积步 $G_{\text{init}} = 128$<br>• 全局 batch 大小 $B = 8192$ microbatches |
| **故障注入** | 注入 **256 次 GPU 故障**，分布在整个训练过程中，间隔 5 个 iteration；所有故障均发生在 **gradient synchronization 阶段**（最严苛场景） |
| **对比基线** | • **Checkpoint-and-Restart**：标准检查点恢复<br>• **AdaptiveWorldPolicy**：作为消融实验中的弱 baseline（允许 global batch 缩减） |

### ✅ 评估指标

| 指标 | 定义 | 说明 |
|------|------|------|
| **Effective Throughput** | $\frac{\text{Processed Tokens}}{\text{Runtime} \times \text{Alive GPUs}}$ | 衡量单位活跃 GPU 的实际处理能力，消除规模变化影响 |
| **Cumulative Processed Tokens** | 总共处理的 token 数量 | 衡量整体训练进度 |
| **Training Loss Curve** | 与无故障基准的损失曲线对比 | 验证是否保持训练轨迹一致性 |
| **Wall-clock Breakdown** | 单次故障恢复各阶段耗时分析 | 对比恢复效率 |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据

| 指标 | 结果 |
|------|------|
| **训练轨迹一致性** | RECoVER 的 loss 曲线与无故障运行 **完全重合**，无任何 spikes 或震荡（见 Figure 7a） |
| **有效吞吐提升** | 在连续故障下，相比 checkpoint-restart 基线，**有效吞吐提高 2.23×** |
| **GPU 小时利用率** | 在 234 GPU-hours 内，RECoVER 多处理 **+102M tokens（+74.9%）** |
| **长期优势扩大** | 随着训练延长，RECoVER 的优势持续扩大，因为幸存者承担更多计算，摊薄通信开销 |

### ✅ 与基线方法对比

| 场景 | RECoVER vs Checkpoint-Restart |
|------|------------------------------|
| **单次故障恢复时间** | RECoVER 成本基本恒定（不丢弃工作）；基线随 checkpoint interval 增加而线性上升 |
| **多次故障累积影响** | 基线吞吐稳定低位；RECoVER 吞吐随故障增加反而上升（due to amortization） |
| **最优 checkpoint 频率** | 即使在 checkpoint 最优频率（N=2）下，RECoVER 仍胜出 |
| **生产环境可扩展性** | 在 100k GPU 规模，重启时间预计达 10 分钟，远高于 MTBF（~5 分钟），传统方法将几乎无法推进；RECoVER 不受此限 |

### ✅ 消融实验结果

- **禁用 Versatile-Workload（即 AdaptiveWorldPolicy）**：
  - 全局 batch 随副本减少而缩减；
  - 出现明显 loss spike 和训练漂移；
  - 验证了顶层策略对于保持训练等价性的必要性。

- **不同并行架构表现**：
  - **RECoVER-3D**：在 3D 并行下表现优异，吞吐最终反超无故障运行（因计算密集化）；
  - **RECoVER-HSDP**：同样保持轨迹一致，在 1338 GPU-hours 下多处理 **+47.4% tokens**，验证了框架的通用性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **前向恢复可行且高效**：通过精细控制通信、梯度恢复和负载调度，可在不中断训练的前提下实现高弹性。
2. **保持全局 batch 不变是关键**：这是维持训练轨迹等价性的充分条件，优于简单“继续训练”策略。
3. **故障可转化为计算密度提升**：随着副本减少，幸存者执行更多 microbatch，进入更计算密集状态，反而提升 per-GPU 利用率。
4. **设计具有高度通用性**：同一框架无缝集成 3D Parallelism 与 HSDP，适用于主流 LLM 训练栈。

### ✅ 方法的局限性

- **副本内部不可细粒度恢复**：若副本内任一设备失败，则整个 replica 被废弃（当前设计以 replica 为原子单位）；
- **未回收失效副本资源**：目前直接丢弃故障 replica 的剩余设备，未来可考虑复用；
- **暂不支持动态新副本加入**：无法在后期重新扩容集群以补偿性能下降。

### ✅ 未来工作方向

1. **回收故障副本中的健康设备**：进一步提升资源利用率；
2. **支持动态副本 rejoin 机制**：允许新节点加入，控制训练节奏；
3. **扩展至其他模型类型**：如 diffusion models 或 recommendation systems；
4. **结合高级 checkpointing 技术**：与 lazy/asynchronous checkpointing 结合，构建混合容错体系。

---

> 🔚 **总结一句话**：  
> **RECoVER 通过“修复-恢复-调整-继续”四步协议，在不牺牲训练质量的前提下，实现了超大规模 LLM 预训练系统的高弹性与高资源利用率，是迈向百万 GPU 训练时代的关键一步。**

</details>

---

### 14. [SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization](https://arxiv.org/abs/2605.12245)

**Authors**: Chengzhu Bao, Xianglong Yan, Zhiteng Li, Guangshuo Qin, Guanghua Yu, Yulun Zhang  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.12245v1  

#### Abstract
NVFP4 has recently emerged as an efficient 4-bit microscaling format for large language models (LLMs), offering superior numerical fidelity with native hardware support. However, existing methods often yield suboptimal performance due to inflexible scale selection and the coupled treatment of quanti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **NVFP4**（一种4位微缩放浮点格式）量化方法在处理大语言模型（LLMs）时存在两个关键瓶颈：
1. **Scale Selection 不灵活**：现有方法通常采用固定的启发式规则（如基于最大值的缩放）或有限的离散搜索来确定全局和块级 scale，难以适应 LLM 权重复杂且不规则的分布。
2. **Quantization 和 Dequantization Scale 耦合**：硬件限制要求块级 scale 必须以低精度（FP8/E4M3）存储，并同时用于量化和反量化过程，导致 scale 本身的量化误差会传播到重建中。

### 提出的新方法与新思路
作者提出 **SOAR**（Scale Optimization for Accurate Reconstruction），一个全新的后训练量化（PTQ）框架，包含两大核心技术：

#### ✅ **Closed-form Joint Scale Optimization (CJSO)**  
- **联合优化全局和块级 scale**：通过最小化重建误差，推导出关于全局 scale $ \alpha $ 和块级 scale $ \Delta_i $ 的闭式解析解（closed-form updates）。
- 在每次迭代中交替更新 scale 和 FP4 量化分配，实现高效、精确的联合优化。

#### ✅ **Decoupled Scale Search (DSS)**  
- **解耦量化与反量化 scale**：
  - 引入高精度的 **量化 scale $ \Delta_q $**（仅用于训练时决定 FP4 映射）
  - 保留硬件兼容的 **反量化 scale $ \Delta_a $**（最终存储为 E4M3）
- 执行局部联合搜索，选择使重建误差最小的 $ (\Delta_q, \Delta_a) $ 组合，显著降低因 scale 量化带来的精度损失。

### 相比现有方法的优势
| 特性 | SOAR | 其他方法（如 4over6, RaZeR） |
|------|------|-------------------------------|
| Scale 优化方式 | 解析式联合优化 + 局部精细搜索 | 固定规则 / 简单二元选择 |
| Scale 耦合性 | 解耦（$ \Delta_q \neq \Delta_a $） | 耦合（同一 scale 双重用途） |
| 精度恢复能力 | 更优的重建保真度 | 受限于 scale 表示精度 |
| 硬件开销 | 零额外内存开销（$ \Delta_q $ 不存储） | 同样无开销，但利用效率低 |
| 泛化性 | 可扩展至 MXFP4 等其他微缩放格式 | 多为特定设计 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **校准数据集（Calibration Set）**（用于部分对比方法如 GPTQ）：
  - `WikiText-2`（128 个样本，序列长度 2048）
- **评估任务**（zero-shot 推理基准）：
  - **常识推理**：`WinoGrande`, `PIQA`, `HellaSwag`
  - **科学常识**：`ARC-Easy`, `ARC-Challenge`
  - **综合知识与数学推理**：`MMLU`, `GSM8K`

### 实验设置和评估指标
- **量化配置**：W4A4（权重和激活均为 4-bit NVFP4）
- **模型范围**：
  - `LLaMA-3.1-8B-Instruct`
  - `LLaMA-3.2-1B/3B-Instruct`
  - `Qwen3-4B/8B`
- **评估指标**：
  - **主指标**：多个 zero-shot 任务上的平均准确率（Avg. Accuracy）
  - **辅助指标**：`MMLU` 和 `GSM8K` 上的单项得分
  - **语言建模能力**：`WikiText2` 和 `C4` 上的困惑度（Perplexity ↓）
- **实现细节**：
  - 使用 PyTorch 和 HuggingFace Transformers
  - SOAR 迭代次数设为 15（早停机制：相对 MSE 改善 < 1e-3）
  - 所有实验在 NVIDIA A800 GPU 上进行

### 基线方法对比
| 方法 | 类型 | 是否使用校准数据 |
|------|------|------------------|
| FP16 | 全精度基准 | — |
| NVFP4 (Baseline) | 原始 NVFP4 缩放 | 否 |
| 4over6 (Cook et al., 2025) | 自适应块缩放策略 | 否 |
| RaZeR (Chen et al., 2025) | 冗余零 remapping 提升数值覆盖 | 否 |
| GPTQ (Frantar et al., 2023) | 基于 Hessian 的重建优化 | 是 |
| Ours + GPTQ | SOAR 与 GPTQ 结合 | 是 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### 🔹 Zero-shot 平均准确率（Avg. ↑）
| 模型 | 方法 | Avg. Accuracy |
|------|------|---------------|
| Qwen3-8B | NVFP4 | 68.75 |
| Qwen3-8B | 4over6 | 69.15 |
| Qwen3-8B | RaZeR | **70.12** |
| Qwen3-8B | **SOAR** | **70.68** ✅ |
| LLaMA-3.2-3B | RaZeR | 65.19 |
| LLaMA-3.2-3B | **SOAR** | **66.00** ✅ |

> 💡 **SOAR 在 Qwen3-8B 上超越当前 SOTA (RaZeR) 达 +0.56%，是目前最高的 NVFP4 性能。**

#### 🔹 数学与知识推理任务表现（Table 2）
| 模型 | 方法 | MMLU | GSM8K | Avg. |
|------|------|-------|--------|------|
| Qwen3-8B | NVFP4 | 70.70 | 79.30 | 75.00 |
| Qwen3-8B | RaZeR | 71.31 | 82.41 | 76.86 |
| Qwen3-8B | **SOAR** | **71.47** | **82.56** | **77.02** ✅ |
| LLaMA-3.2-3B | RaZeR | 59.29 | 72.93 | 66.11 |
| LLaMA-3.2-3B | **SOAR** | 59.23 | **73.62** | **66.43** ✅ |

> 📈 **SOAR 在 MMLU 和 GSM8K 上持续领先，尤其在 GSM8K 上提升明显（+0.69 vs RaZeR）。**

#### 🔹 与 GPTQ 结合的效果（Table 3）
| 模型 | 方法 | Avg. Accuracy |
|------|------|----------------|
| LLaMA-3.1-8B | GPTQ | 72.95% |
| LLaMA-3.1-8B | **SOAR + GPTQ** | **73.18%** ✅ |
| LLaMA-3.2-3B | GPTQ | 65.42% |
| LLaMA-3.2-3B | **SOAR + GPTQ** | **65.82%** ✅ |

> ⚙️ **即使作为 calibration-free 方法，SOAR 仍能与 GPTQ 协同增益，进一步提升性能。**

### 消融实验结果（Table 4）

| 方法 | Wiki2 PPL ↓ | C4 PPL ↓ | Avg. Acc ↑ |
|------|-------------|----------|------------|
| NVFP4 | 11.98 | 15.53 | 65.02 |
| +CJSO | 12.04 | 15.53 | 65.64 |
| +DSS | 11.96 | 15.45 | 65.50 |
| **SOAR (CJSO+DSS)** | **11.88** | **15.44** | **66.00** ✅ |

> 🔍 **CJSO 和 DSS 各自带来独立增益，二者结合达到最佳效果，验证了模块互补性。**

### DSS 对 MXFP4 的泛化能力（Table 5）
| 模型 | 方法 | Avg. Acc |
|------|------|---------|
| LLaMA-3.2-3B | MXFP4 | 62.46 |
| LLaMA-3.2-3B | **DSS-enhanced** | **63.32** ✅ (+0.86) |
| Qwen3-4B | MXFP4 | 61.80 |
| Qwen3-4B | **DSS-enhanced** | **62.27** ✅ (+0.47) |

> 🔄 **DSS 可推广至 MXFP4，证明其解决的是微缩放格式中的通用 scale 误差问题。**

---

## 4. 关键结论和发现

### 主要发现
1. **Scale Selection 是 NVFP4 性能瓶颈的关键因素**：传统固定规则无法充分拟合 LLM 权重分布，而 SOAR 的 CJSO 提供了更优的解析式联合优化路径。
2. **Scale 耦合引入不必要的误差**：将高精度量化决策与低精度反量化绑定是次优的；DSS 通过解耦实现了“用高精度思考，用低精度执行”的理想范式。
3. **无需增加硬件开销即可提效**：SOAR 完全兼容 NVFP4 存储格式（仍只存 E4M3 scale），却显著提升了重建质量，适合部署。
4. **具有良好的可组合性和泛化性**：
   - 可无缝集成进 GPTQ 等 calibration-based 框架；
   - DSS 可迁移到 MXFP4 等其他微缩放格式。

### 方法的局限性
- **计算成本略高**：相比一次性缩放策略，SOAR 需要多轮迭代优化（约 15–37 分钟，见 Table 8），不适合极快量化场景。
- **未考虑激活敏感性**：当前为 calibration-free 设计，未利用输入数据动态调整 scale（尽管作者指出这是未来方向）。
- **依赖 FP4 表示特性**：对 E2M1 的 FP4 格式做了适配，若目标硬件使用不同 FP4 encoding，需重新调参。

### 未来工作方向
1. **引入 Activation-aware Objective**：利用少量 calibration data 构建 layer-wise 输出失真最小化目标，进一步提升 scale 的上下文适应性。
2. **探索自动化搜索空间设计**：当前 DSS 使用人工定义的搜索网格（如 β ∈ [0.5,1.5]），未来可用强化学习或贝叶斯优化自动探索最优策略。
3. **扩展至训练时量化（QAT）**：将 CJSO/DSS 思想融入量化感知训练，实现端到端优化。
4. **支持更多硬件原生格式**：将 SOAR 推广至 TPUs 或其他 AI 加速器支持的微缩放格式。

---

> ✅ **总结一句话**：  
> **SOAR 通过“闭式联合优化”和“解耦 scale 搜索”，在不增加任何硬件开销的前提下，显著提升了 NVFP4 量化的重建精度，在多个 LLM 上实现了新的 state-of-the-art 性能。**  
> 代码已开源：[https://github.com/steven-bao1/SOAR](https://github.com/steven-bao1/SOAR)

</details>

---

### 15. [C2L-Net: A Data-Driven Model for State-of-Charge Estimation of Lithium-Ion Batteries During Discharge](https://arxiv.org/abs/2605.08653)

**Authors**: Khoa Tran, T. Nguyen-Thoi, Vin Nguyen-Thai, Duong Tran Anh, Hung-Cuong Trinh, Tri Le  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.08653v1  

#### Abstract
Accurate state-of-charge (SOC) estimation is critical for the safe and efficient operation of lithium-ion batteries in battery management systems (BMS). Although data-driven approaches can effectively capture nonlinear battery dynamics, many existing methods rely on long historical input sequences, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：C2L-Net: A Data-Driven Model for State-of-Charge Estimation of Lithium-Ion Batteries During Discharge

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的基于 data-driven 的 SOC（State-of-Charge）估计方法存在以下关键问题：
- **依赖长历史输入序列**：许多模型需要数百秒甚至数千秒的历史数据，导致高计算开销，难以部署在实时、资源受限的 BMS（Battery Management System）中。
- **零填充引入位置偏差（padding-induced positional bias）**：在驾驶周期开始阶段，由于历史数据不足，需用零值填充，导致模型可能学习到“高SOC样本对应长零填充”的虚假模式，而非真实的电池动态特性。
- **缺乏对最新测量值的快速响应机制**：传统模型通常均匀处理历史数据，未能显式区分长期上下文信息与最具时效性的最新测量。

### 提出的新方法与创新思路
作者提出 **C2L-Net**（Context-to-Latest Network），一种新颖的 **context-to-latest 数据驱动框架**，其核心思想是将“上下文编码”与“最新测量更新”解耦。

#### 主要创新点包括：
- ✅ **短窗口 SOC 估计范式（Short-window SOC Estimation Paradigm）**  
  仅使用 **20 秒（L=200 步）的短历史窗口**作为输入，避免了长序列依赖和初始阶段的零填充问题，提升了在线估计的真实性和鲁棒性。

- ✅ **Chunk-based 特征提取机制**  
  将输入序列划分为多个 chunk，并结合：
  - **Theta Attention Pooling**：为每个 chunk 内的时间步分配注意力权重，保留局部重要变化；
  - **Fourier-based Seasonality Basis**：通过傅里叶基生成代表局部趋势与周期性的 token，实现紧凑表示并降低序列长度。

- ✅ **因果上下文编码器（Causal Context Encoder）**  
  结合 GRU 与 **Causal Cosine Attention**：
  - 使用 **cosine similarity** 替代传统的 dot-product attention，增强对特征方向的敏感性，减少幅值影响；
  - 引入 **causal mask** 确保无未来信息泄露，符合实际推理场景。

- ✅ **受递归滤波启发的最新测量解码器（Latest-Measurement Decoder）**  
  类似于 Kalman Filter 的状态更新机制，使用 **GRUCell** 将当前最新的测量值 $ x_t $ 融入已编码的上下文状态 $ g_t $，实现动态修正，显著提升对突变工况的响应能力。

### 相比现有方法的优势
| 优势维度 | 具体表现 |
|--------|--------|
| **准确性** | 在多种温度条件下达到 SOTA 或具有竞争力的精度（MAE 最低至 0.4118%） |
| **效率** | 推理速度比 TCN-Short 快 **60×**，比 TTSNet 快 **40×** |
| **参数量小** | 仅 161,347 参数，模型大小仅 0.62 MB，适合嵌入式部署 |
| **泛化性强** | 在未见过的驾驶循环（如 PDMHC）上表现稳定 |
| **现实适用性高** | 不依赖长历史、无 padding 偏差，适用于真实 BMS 场景 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Public Lithium-ion Battery Drive-Cycle Dataset** [20]
- **电池类型**：LG INR 21700 M50LT（NMC 化学体系）
- **额定容量**：4.93 Ah，**额定电压**：3.69 V
- **测试条件**：涵盖 **5 种固定环境温度**（5°C, 15°C, 25°C, 35°C, 45°C）
- **驱动循环**：共 12 种不同负载模式，包括城市、高速、重载等典型场景

#### 数据划分策略：
| 集合 | 包含的 Drive Cycles |
|------|---------------------|
| **训练集** | BCDC, LA92, CSHVC, HWFET, IM, US06, PDTCB, OCTBC |
| **验证集** | HHDDT, FTP-72 |
| **测试集** | FTP-75, PDMHC（完全未见于训练） |

> ✅ 强调跨驾驶循环的泛化能力评估

### 实验设置
- **硬件平台**：NVIDIA RTX 3060 GPU, AMD Ryzen 7 CPU, 32GB RAM
- **实现框架**：PyTorch
- **优化器**：AdamW
- **学习率**：5e-4
- **Batch Size**：128
- **训练轮数**：100 epochs
- **输入归一化**：Min-Max Normalization（基于训练集拟合）

### 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **MAE** | Mean Absolute Error (%) | 平均误差水平 |
| **RMSE** | Root Mean Squared Error (%) | 对大误差更敏感 |
| **MAX** | Maximum Absolute Error (%) | 极端情况下的最大偏差 |
| **Latency & Throughput** | 推理延迟（ms）、吞吐量（inferences/s） | 衡量计算效率 |

### 基线方法对比
- **基础架构**：TCN, LSTM, GRU, Transformer Encoder
- **先进方法**：
  - **TTSNet** [19]：基于 Temporal Transformer 的多分支结构
  - **TCN-Short** [20]：采用短感受野（~0.8s）以缓解 padding 问题

> 所有基线均在同一实验环境下复现，确保公平比较

---

## 3. 主要实验结果和性能指标

### 关键性能数据（平均 MAE/RMSE across FTP-75 & PDMHC）

| 温度 | 方法 | MAE (%) | RMSE (%) | MAX (%) |
|------|------|---------|----------|---------|
| **45°C** | **C2L-Net (Ours)** | **0.4118** | **0.5320** | **2.4923** |
|          | LSTM | 0.4191 | 0.5585 | 3.8197 |
| **35°C** | LSTM | **0.6666** | **0.8892** | — |
|          | C2L-Net | 0.7529 | 1.0363 | 6.3149 |
| **25°C** | **C2L-Net** | **0.6708** | **0.9576** | 6.3822 |
| **15°C** | **C2L-Net** | **0.7529** | **1.0363** | 6.3149 |
| **5°C**  | **C2L-Net** | **1.0386** | **1.3299** | **7.3633** |

✅ **C2L-Net 在 4/5 个温度条件下取得最优的平均 MAE 和 RMSE**

### 与基线方法的整体对比
- 在 **低温（5°C）和常温（25°C）下显著优于所有基线**，尤其在 PDMHC 上表现最佳；
- 即使在 35°C 下略逊于 LSTM，但仍优于其他模型（如 TCN-Short、TTSNet）；
- **TCN 和 TCN-Short 出现极大 MAX error**（如 25°C 下高达 54%），表明其稳定性差；
- **Transformer 和 TTSNet 虽然准确但计算成本极高**，不适合实时应用。

### 消融实验结果（Ablation Study at 5°C）

#### （1）模块有效性分析（Table 2）
| 配置 | Avg MAE (%) | Avg RMSE (%) |
|------|-------------|--------------|
| **完整 C2L-Net（GRU-Cosine-GRUCell）** | **1.0386** | **1.3299** |
| 替换为 TCN 特征提取 | 1.5856 | 2.0634 |
| Encoder-only（Multi-head Self-Attention） | 3.3816 | 4.9397 |
| Encoder-only（Cosine Similarity） | 5.7616 | 8.6385 |
| GRU-GRUCell（无 Cosine Attention） | 1.0792 | 1.4302 |

➡️ 结论：
- **Chunk-based 特征提取有效降低误差约 35%**
- **加入 Latest-Measurement Decoder 可大幅改善性能**
- **Causal Cosine Attention 比标准 attention 更鲁棒**

#### （2）超参数敏感性分析（Fig. 3）
- **输入长度 L=200（20s）效果最好**，过短（L=50）损失上下文，过长无明显增益；
- **隐藏维度 d=128 最优**，d=512 导致过拟合；
- **Dropout p=0.2 提供最佳正则化效果**；
- **Seasonality Harmonics K=10 效果最佳**，过多或过少均下降。

---

## 4. 关键结论和发现

### 主要发现
1. 🔍 **短窗口 + 显式最新测量更新可实现高效且准确的 SOC 估计**  
   C2L-Net 成功证明：无需长达数千秒的历史数据，仅用 **20 秒短窗 + 最新测量融合机制** 即可在复杂动态负载下保持高精度。

2. 🧠 **Chunk-based 编码 + 因果注意力能有效建模局部时序动态**  
   Theta Attention Pool 与 Seasonality Basis 的组合能够在压缩序列的同时保留关键局部特征。

3. ⚙️ **受 Kalman Filter 启发的设计显著提升响应速度**  
   Latest-Measurement Decoder 类似于 learnable state update rule，使模型能快速适应电流突变、负载切换等瞬态事件。

4. ⏱️ **超高推理效率使其极具工程落地潜力**  
   达到 **3368.4 inferences/s** 的吞吐量，远超现有方法，满足车载 BMS 的实时性要求。

### 方法的局限性
- 当前实验基于 **fixed-temperature conditions**，未直接验证在剧烈温度变化下的表现；
- 虽然温度作为输入特征被纳入，但未显式建模热-电耦合物理过程；
- Chunk 划分方式为固定长度，可能无法自适应不同放电速率下的动态时间尺度。

### 未来工作方向
1. **扩展至 dynamic-temperature driving profiles**，进一步贴近真实用车场景；
2. **集成 Physics-Informed Neural Network (PINN)** 框架，将电化学先验知识融入网络，提升外推能力和解释性；
3. **探索 adaptive chunking 或 hierarchical modeling**，以应对多时间尺度的电池行为；
4. **在真实车辆平台上进行实车验证与部署测试**。

---

> ✅ **总体评价**：  
> C2L-Net 是一项面向实际应用的创新性研究，不仅在精度上达到领先水平，更重要的是在 **效率、简洁性、现实适用性** 上实现了突破，为下一代轻量化、高性能 BMS 提供了可行的技术路径。

</details>

---

### 16. [SOMA: Efficient Multi-turn LLM Serving via Small Language Model](https://arxiv.org/abs/2605.11317)

**Authors**: Xueqi Cheng, Qiong Wu, Zhengyi Zhou, Xugui Zhou, Tyler Derr, Yushun Dong  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.11317v1  

#### Abstract
Large Language Models (LLMs) are increasingly deployed in multi-turn dialogue settings where preserving conversational context across turns is essential. A standard serving practice concatenates the full dialogue history at every turn, which reliably maintains coherence but incurs substantial cost i...

---

### 17. [MLCommons Chakra: Advancing Performance Benchmarking and Co-design using Standardized Execution Traces](https://arxiv.org/abs/2605.11333)

**Authors**: Srinivas Sridharan, Andy Balogh, Bradford M. Beckmann, Brian Coutinho, Louis Feng, Sheng Fu, Sanshan Gao, Mehryar Garakani, Taekyung Heo, David Kanter, Josh Ladd, Ziwei Li, Winston Liu, Changhai Man, Dan Mihailescu, Spandan More, Joongun Park, Ashwin Ramachandran, Vinay Ramakrishnaiah, Saeed Rashidi, Vijay Janapa Reddi, Puneet Sharma, Phio Tian, William Won, Hanjiang Wu, Huan Xu, Jinsun Yoo, Tushar Krishna  
**Category**: cs.DC  
**Published**: 2026-05-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.11333v1  

#### Abstract
The fast pace of artificial intelligence~(AI) innovation demands an agile methodology for observation, reproduction and optimization of distributed machine learning~(ML) workload behavior in production AI systems and enables efficient software-hardware~(SW-HW) co-design for future systems. We presen...

---

### 18. [On the Importance of Multistability for Horizon Generalization in Reinforcement Learning](https://arxiv.org/abs/2605.12206)

**Authors**: Asad Bakija, Florent De Geeter, Julien Brandoit, Pierre Sacr\'e, Guillaume Drion  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.12206v1  

#### Abstract
In reinforcement learning (RL), agents acting in partially observable Markov decision processes (POMDPs) must rely on memory, typically encoded in a recurrent neural network (RNN), to integrate information from past observations. Long-horizon POMDPs, in which the relevant observation and the optimal...

---

### 19. [Neural-Schwarz Tiling for Geometry-Universal PDE Solving at Scale](https://arxiv.org/abs/2605.12343)

**Authors**: Paolo Secchi, Daniel S. Balint, Marco Maurizi  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.12343v1  

#### Abstract
Most learned PDE solvers follow a global-surrogate paradigm: a neural operator is trained to map full problem descriptions to full solution fields for a prescribed distribution of geometries, boundary conditions, and coefficients. This has enabled fast inference within fixed problem families, but li...

---

### 20. [Learning, Fast and Slow: Towards LLMs That Adapt Continually](https://arxiv.org/abs/2605.12484)

**Authors**: Rishabh Tiwari, Kusha Sareen, Lakshya A Agrawal, Joseph E. Gonzalez, Matei Zaharia, Kurt Keutzer, Inderjit S Dhillon, Rishabh Agarwal, Devvrit Khatri  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.12484v1  

#### Abstract
Large language models (LLMs) are trained for downstream tasks by updating their parameters (e.g., via RL). However, updating parameters forces them to absorb task-specific information, which can result in catastrophic forgetting and loss of plasticity. In contrast, in-context learning with fixed LLM...

---

### 21. [PRISM: Pareto-Efficient Retrieval over Intent-Aware Structured Memory for Long-Horizon Agents](https://arxiv.org/abs/2605.12260)

**Authors**: Jingyi Peng, Zhongwei Wan, Weiting Liu, Qiuzhuang Sun  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.12260v1  

#### Abstract
Long-horizon language agents accumulate conversation history far faster than any fixed context window can hold, making memory management critical to both answer accuracy and serving cost. Existing approaches either expand the context window without addressing what is retrieved, perform heavy ingesti...

---

### 22. [LEAP: Unlocking dLLM Parallelism via Lookahead Early-Convergence Token Detection](https://arxiv.org/abs/2605.10980)

**Authors**: Haohui Zhang, Zhiye Wang, Xiaoying Gan, Xinbing Wang, Bo Jiang  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.10980v1  

#### Abstract
Diffusion Language Models (dLLMs) have garnered significant attention for their potential in highly parallel processing. The parallel capabilities of existing dLLMs stem from the assumption of conditional independence at high confidence levels, which ensures negligible discrepancy between the margin...

---

### 23. [U-STS-LLM A Unified Spatio-Temporal Steered Large Language Model for Traffic Prediction and Imputation](https://arxiv.org/abs/2605.11735)

**Authors**: Yichen Zhang, Jun Li  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.11735v1  

#### Abstract
The efficient operation of modern cellular networks hinges on the accurate analysis of spatio-temporal traffic data. Mastering these patterns is essential for core network functions, chiefly forecasting future load to pre-empt congestion and imputing missing values caused by sensor failures or trans...

---

### 24. [Auto-Rubric as Reward: From Implicit Preferences to Explicit Multimodal Generative Criteria](https://arxiv.org/abs/2605.08354)

**Authors**: Juanxi Tian, Fengyuan Liu, Jiaming Han, Yilei Jiang, Yongliang Wu, Yesheng Liu, Haodong Li, Furong Xu, Wanhua Li  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.08354v1  

#### Abstract
Aligning multimodal generative models with human preferences demands reward signals that respect the compositional, multi-dimensional structure of human judgment. Prevailing RLHF approaches reduce this structure to scalar or pairwise labels, collapsing nuanced preferences into opaque parametric prox...

---

### 25. [SkillLens: Adaptive Multi-Granularity Skill Reuse for Cost-Efficient LLM Agents](https://arxiv.org/abs/2605.08386)

**Authors**: Yongliang Miao, Ziyang Yu, Liang Zhao, Bowen Zhu, Hasibul Haque  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.08386v1  

#### Abstract
Skill libraries have become a practical way for LLM agents to reuse procedural experience across tasks. However, existing systems typically treat skills as flat, single-resolution prompt blocks. This creates a tension between relevance and cost: injecting coarse skills can introduce irrelevant or mi...

---

### 26. [AHD Agent: Agentic Reinforcement Learning for Automatic Heuristic Design](https://arxiv.org/abs/2605.08756)

**Authors**: Haoze Lv, Ning Lu, Ziang Zhou, Shengcai Liu  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.08756v1  

#### Abstract
Automatic heuristic design (AHD) has emerged as a promising paradigm for solving NP-hard combinatorial optimization problems (COPs). Recent works show that large language models (LLMs), when integrated into well-designed frameworks (i.e., LLM-AHD), can autonomously discover high-performing heuristic...

---

### 27. [On Predicting the Post-training Potential of Pre-trained LLMs](https://arxiv.org/abs/2605.11978)

**Authors**: Xiaoyuan Li, Yubo Ma, Kexin Yang, Moxin Li, Keqin Bao, Wenie Wang, Fuli Feng, Dayiheng Liu  
**Category**: cs.CL  
**Published**: 2026-05-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.11978v1  

#### Abstract
The performance of Large Language Models (LLMs) on downstream tasks is fundamentally constrained by the capabilities acquired during pre-training. However, traditional benchmarks like MMLU often fail to reflect a base model's plasticity in complex open-ended scenarios, leading to inefficient model s...

---

### 28. [AESOP: Adversarial Execution-path Selection to Overload Deep Learning Pipelines](https://arxiv.org/abs/2605.10987)

**Authors**: Tingxi Li, Mingfang Ji, Ravishka Shemal Rathnasuriya, Simin Chen, Yitao Hu, Wei Yang  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.10987v1  

#### Abstract
Modern machine learning deployments increasingly compose specialized models into dynamic inference pipelines, where upstream components produce intermediate predictions that determine the workload and inputs of downstream components. The cost of processing an input is therefore not determined by any...

---

### 29. [gym-invmgmt: An Open Benchmarking Framework for Inventory Management Methods](https://arxiv.org/abs/2605.11355)

**Authors**: Reza Barati, Qinmin Vivian Hu  
**Category**: cs.LG  
**Published**: 2026-05-13  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.11355v1  

#### Abstract
Inventory-policy comparisons are often difficult to interpret because performance depends on the evaluation contract as much as on the policy itself. Differences in topology, demand regime, information access, feasibility constraints, shortage treatment, and Key Performance Indicator (KPI) definitio...

---

### 30. [Latency Analysis and Optimization of Alpamayo 1 via Efficient Trajectory Generation](https://arxiv.org/abs/2605.08975)

**Authors**: Yunseong Jeon, Namcheol Lee, Yoonsu Lee, Jangwoon Park, Sol Ahn, Jong-Chan Kim, Seongsoo Hong  
**Category**: cs.AI  
**Published**: 2026-05-13  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.08975v1  

#### Abstract
Reasoning-based end-to-end (E2E) autonomous driving has recently emerged as a promising approach to improving the interpretability of driving decisions as it can generate human-readable reasoning together with predicted trajectories. Such approaches commonly generate multiple trajectories to capture...

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
