# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-28 08:00:07 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers](https://arxiv.org/abs/2604.23798)

**Authors**: Chih-Chung Hsu, Xin-Di Ma, Wo-Ting Liao, Chia-Ming Lee  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2604.23798v1  

#### Abstract
Existing attention accelerators often trade exact softmax semantics, depend on fused Tensor Core kernels, or incur sequential depth that limits FP32 throughput on long sequences. We present \textbf{ELSA}, an algorithmic reformulation of online softmax attention that (i)~preserves exact softmax seman...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers — 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Vision Transformers (ViT) 中的 Multi-Head Self-Attention (MHSA) 存在 **O(n²) 内存开销**，尤其在高分辨率图像处理中（如 4K 图像）会导致数 TB 的 FP32 内存需求，远超当前硬件能力。此外，现有加速方案存在以下缺陷：
- **近似方法**（如 Performer、Linformer）：修改 attention 操作，需重新训练模型，不适用于已训练好的 foundation models（如 CLIP、LLaMA）。
- **硬件融合内核**（如 FlashAttention-2/3）：依赖 Tensor Core（HMMA/GMMA），在 FP32 下无高效路径，且无法部署于边缘设备（如 Jetson TX2）。
- **内存优化实现**（如 ME-SDPA）：虽支持 FP32，但为串行计算，深度为 O(n)，导致长序列下延迟极高。

### 🚀 提出的新方法：ELSA
提出 **Exact Linear-Scan Attention (ELSA)**，一种算法层面的在线 softmax 重构方法，其核心思想是：
- 将 online softmax 的递推过程建模为一个 **关联幺半群 (associative monoid)** `(m, S, W)` 上的前缀扫描（prefix scan）。
- 利用并行前缀扫描算法（如 Hillis-Steele + Blelloch）将计算深度从 O(n) 降低至 **O(log n)**，同时保持精确的 softmax 语义。

### 🔍 相比现有方法的优势
| 特性 | ELSA | FlashAttention | ME-SDPA | 近似方法 |
|------|------|----------------|---------|----------|
| **Exact softmax** | ✅ | ✅ (FP16) ❌ (FP32) | ✅ | ❌ |
| **FP32 高精度支持** | ✅ | ❌（无竞争性路径） | ✅ | ❌ |
| **无需重训练** | ✅ | ✅ | ✅ | ❌ |
| **Tensor Core 无关** | ✅ | ❌ | ✅ | ✅ |
| **并行深度** | **O(log n)** | O(n/T) | O(n/T) | O(log n) |
| **硬件通用性** | ✅（A100 到 Jetson TX2） | ❌（仅限 Ampere/Hopper） | ✅ | ✅ |

> ✅ **ELSA 是目前唯一在 FP32 下兼具精确性、高吞吐、低深度、硬件无关性的 attention 内核。**

---

## 2. 核心实验方法和设置

### 📚 数据集与任务
- **合成序列基准测试**：n = 64–16,384 tokens，用于评估 attention 内核本身性能。
- **视觉任务**：
  - ImageNet-1K 分类（ViT-B/16, Swin-T）
  - CLIP 零样本推理（ViT-L/14）
  - 超光谱分类（Pavia, Salinas, WHU，使用 HSIMAE）
- **语言任务**：
  - BERT 情感分析（SST-2, IMDB）
  - LLaMA-13B 推理（host-device offloading）
- **3D 视觉**：VGGT 和 FastVGGT 用于多视角三维重建。

### ⚙️ 实验设置
- **硬件平台**：
  - 主平台：NVIDIA A100 (40GB, CUDA 12.6, PyTorch 2.6)
  - 边缘设备：Jetson TX2
- **精度模式**：FP32、FP16、TF32-Turbo
- **实现方式**：基于 Triton 和 CUDA C++ 实现，**无 Tensor Core 依赖**。
- **批大小**：多数实验使用 batch=8（ImageNet），部分使用 batch=1（LLaMA offloading）。

### 📊 评估指标
- **吞吐量 (Throughput)**：images/s 或 M tok/s
- **延迟 (Latency)**：ms
- **峰值显存 (Peak VRAM)**：GB
- **速度提升 (Speedup)**：vs. ME-SDPA / Math kernel
- **数值误差**：相对 L2 误差、argmax 不一致率等

### 🆚 基线方法对比
- **Exact 方法**：
  - `Math-SDPA`（PyTorch 默认）
  - `ME-SDPA`（xFormers 中的内存高效实现）
  - `FlashAttention-2/3`（FP16 下对比）
- **排除方法**：所有近似 attention（如 Performer、Linformer）因需重训练，不在公平比较范围内。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（FP32）

#### 在 A100 上（1K–16K tokens）：
- **相比 ME-SDPA**：
  - **速度提升 1.3–3.5×**
  - **BERT 情感任务上达 1.97–2.27× 加速**
- **显存效率**：
  - 在 16K tokens 下，ELSA 显存为 **0.19 GB**，显著低于 FA2 (0.29 GB) 和 FA3 (0.36 GB)。
- **长序列稳定性**：
  - 标准 softmax 在 16K tokens 下 OOM，而 ELSA 仍稳定运行。

#### 在 Jetson TX2 上（边缘设备）：
- **速度提升 1.5–1.6×** vs. Math kernel
- **显存几乎不变**，确认其适用于资源受限场景。

#### 在 LLaMA-13B 主机-设备卸载场景（≥32K tokens）：
- **吞吐提升 17.8–20.2%** vs. ME-SDPA
- 因更低显存占用，计算可更早开始，有效隐藏 PCIe 传输延迟。

#### 在 ImageNet-1K 全模型推理（FP16, batch=8）：
- **ViT 系列**：ELSA 吞吐达 **1064–1309 img/s**，比 FA2 快 **29–65%**
- **Swin 系列**：比 FA2 快 **2–14%**
- **显存更低**：在多个配置下显存低于 Math baseline。

#### 在 CLIP 图像编码器（FP32）：
- **ViT-L/14 @ 336px**：ELSA-Turbo 实现 **1.062× 速度提升**

#### 在 HSIMAE 超光谱分类：
- **吞吐提升 37–62%** vs. ME-SDPA
- 显存增加可忽略（<0.01 GB）

### 🔬 消融实验结果
- **块大小 (Block Size)**：B=128 在延迟与占用之间取得最佳平衡。
- **累加器精度**：
  - 使用 FP16 累加器在 >2K tokens 时误差剧增；
  - **FP32 累加器全程稳定**，验证其必要性。
- **扫描层级设计**：
  - 两层扫描（Hillis-Steele + Blelloch）在共享内存带宽与全局通信间取得最优权衡。
- **变体性能**（Table 15）：
  - `ELSA-Strict (FP32)`：1.36 M tok/s，显存 0.336 GB
  - `ELSA-Turbo (TF32)`：1.43 M tok/s，显存降至 0.288 GB
  - `ELSA-Lean`：显存匹配 ME-SDPA，吞吐 0.13 M tok/s，适合内存敏感场景。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **ELSA 成功将 online softmax 重构为可并行的前缀扫描问题**，首次在 **FP32 精度下实现 O(log n) 并行深度** 的精确 attention。
2. **无需任何硬件特定指令**（如 Tensor Core），即可在 A100 和 Jetson TX2 上统一部署，真正实现 **hardware-agnostic**。
3. **作为 drop-in replacement**，可直接替换预训练模型中的 attention 层，**无需重训练或权重修改**，对 CLIP、ViT、LLaMA、VGGT 等均适用。
4. **在 FP16 下接近 FlashAttention 性能，在 FP32 下全面超越现有方案**，尤其在长序列和高精度场景优势显著。
5. **理论误差界为 O(u log n)**，实验证明其数值稳定性极佳，argmax 完全一致，输出漂移低于浮点精度极限。

### ⚠️ 局限性
1. **算术复杂度仍为 O(n²)**：ELSA 优化的是内存和并行深度，而非 FLOPs，因此在计算密集型场景提升有限。
2. **短序列优势不明显**：当 n < 1K 时，扫描合并开销可能抵消并行收益。
3. **窗口注意力（window attention）增益较小**：在 Swin 等局部窗口模型中，因序列短，ELSA 提升有限（但仍优于 Math kernel）。
4. **多 GPU 扩展性待研究**：跨设备的 monoid reduction 引入同步瓶颈，需专门调度策略。

### 🔮 未来工作方向
- 结合稀疏性或低秩结构以进一步降低 FLOPs。
- 扩展至分布式多 GPU 训练场景，优化 sequence parallelism。
- 支持更多硬件架构（Hopper、Ada Lovelace、非 NVIDIA 加速器）。
- 探索在生成式推理（decode phase）中的应用。

---

> 💡 **总结**：  
> **ELSA 通过将 softmax attention 重构为 associative monoid 上的 prefix scan，实现了首个在 FP32 下兼具精确性、O(log n) 深度、O(n) 内存、硬件无关性的 attention 内核。它不仅是算法上的突破，更是工程上的实用解决方案，为高精度、长序列、边缘部署的 ViT 和 LLM 推理提供了强大支持。**  
> 代码开源地址：[https://github.com/ming0531/ELSA](https://github.com/ming0531/ELSA)

</details>

---

### 2. [JigsawRL: Assembling RL Pipelines for Efficient LLM Post-Training](https://arxiv.org/abs/2604.23838)

**Authors**: Zhengding Hu, Hehua Ouyang, Chang Chen, Zaifeng Pan, Yue Guan, Zhongkai Yu, Zhen Wang, Steven Swanson, Yufei Ding  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2604.23838v1  

#### Abstract
We present JigsawRL, a cost-efficient framework that explores Pipeline Multiplexing as a new dimension of RL parallelism. JigsawRL decomposes each pipeline into a Sub-Stage Graph that exposes the intra-stage and inter-worker imbalance hidden by stage-level systems. On this abstraction, JigsawRL reso...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# JigsawRL: Assembling RL Pipelines for Efficient LLM Post-Training 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Reinforcement Learning (RL)** 框架在进行大语言模型（LLM）后训练时存在严重的**资源利用率低下**和**成本效率差**的问题。主要原因包括：
- **Rollout 阶段的长尾效应**：少数样本解码时间极长，拖慢整个批次。
- **阶段间同步开销**：Rollout 和 Training 必须交替执行，导致 GPU 大量空闲。
- **阶段内与跨 Worker 不平衡**：传统框架仅以“Stage”为单位调度，无法感知更细粒度的负载不均。

这些问题导致即使增加 GPU 资源，**吞吐量提升有限而金钱成本迅速上升**（见图1），MFU（Model FLOPs Utilization）普遍低于 10%。

---

### 提出的新方法与创新思路
JigsawRL 提出了一个全新的 RL 并行维度：**Pipeline Multiplexing（流水线多路复用）**，其核心是将多个并发的 RL 流水线进行高效协同调度，从而提高整体资源利用率。

#### 主要创新点如下：

| 创新机制 | 描述 |
|--------|------|
| **Sub-Stage Graph 抽象** | 将粗粒度的 Rollout/Training 阶段进一步分解为具有不同计算与内存特征的 **sub-stage**（如 Prefill、Decoding、Tool Calling），暴露 stage-level 系统无法看到的内部不平衡。 |
| **Sub-stage Multiplexing（子阶段多路复用）** | 动态地将来自不同 pipeline 的互补型 sub-stage（如 compute-bound 的 Training 与 memory-bound 的 Decoding）并行执行在同一 GPU 上，并通过动态分配 SM 和内存资源减少干扰。 |
| **Sub-stage Merging（子阶段合并）** | 识别出“长尾 rollout”样本，在 DP workers 之间迁移这些低效任务，集中到少数 worker 执行，释放其他 worker 的资源用于高利用率任务。 |
| **Look-ahead Heuristic 调度器** | 构建基于依赖图的调度策略，预测未来几步的关键路径，选择能最小化整体完成时间的 multiplexing 和 merging 操作，避免局部最优陷阱。 |

---

### 相比现有方法的优势
| 对比维度 | JigsawRL | 传统方法（如 Verl, StreamRL） |
|--------|---------|-----------------------------|
| **并行粒度** | Sub-stage 级别 | Stage 或 Pipeline 级别 |
| **资源利用方式** | 动态共享 + 补偿性调度 | 静态划分或完全隔离 |
| **处理长尾能力** | 显式迁移与聚合 | 依赖丢弃或容忍延迟 |
| **适用场景** | 支持异构 pipeline 共存 | 多为同构 pipeline 设计 |
| **成本效率** | 显著提升（最高达 1.85× 吞吐） | 成本随规模增长快速上升 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Single-Turn Reasoning**:  
  - `GSM8K`：小学数学应用题  
  - `MATH`：高中难度数学题
- **Multi-Turn Interaction**:  
  - `AIME`：复杂推理与自我修正任务
- **Tool-Usage / RAG**:  
  - `HotpotQA`：多跳问答，结合外部检索  
  - 外部数据库：`MS MARCO` + FAISS IVF4096 索引

### 模型配置
- **Base Models**: Qwen3-0.6B, Qwen3-4B, Qwen3-32B
- **Instruct-Tuned Models**: Qwen3-4B-Instruct, Llama-3.1-8B-Instruct
- **Distilled Model**: DeepSeek-R1-Distill-Qwen2.5-14B

### 实验平台
- **单节点**：8 × H100 GPU（NVLink 连接）
- **多节点扩展测试**：最多 64 × A100 GPU（每节点 4 GPU，HBM 80GB）

### 评估指标
| 指标 | 定义 |
|-----|------|
| **Throughput (token/s)** | 所有并发 pipeline 每秒处理的 token 总数（主指标） |
| **MFU (%)** | 实际算力利用率占峰值 FLOPs 的比例 |
| **Latency Increase** | 单个 pipeline 步骤延迟相对于独占运行的增长倍数 |
| **Cost Efficiency** | 吞吐量与云服务费用（AWS EC2 A100 定价）的比值 |

### 基线方法对比
| 类型 | 基线系统 | 特点 |
|------|----------|------|
| **Synchronous RL** | `Verl`, `RollMux` | 严格同步 rollout-train 循环；RollMux 引入时间多路复用 |
| **Asynchronous RL** | `StreamRL`, `AReaL` | 允许 off-policy 执行，缓解同步阻塞 |
| **Rollout 多路复用** | `JigsawRL-MuxServe` | 仅对 rollout 阶段使用 MuxServe 进行空间复用作为消融对照 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 场景 | 吞吐提升（vs. Verl） | 最高 MFU 提升 |
|------|--------------------|--------------|
| 单节点同构 pipeline（8 H100） | **平均 1.56×，最高 1.95×** | MFU 从 ~2.3% → **~3.9%**（↑1.7×） |
| 多节点扩展（64 A100） | **最高 1.85×**（DeepSeek-R1-Distill-14B） | 随规模扩大优势增强 |
| 异步 RL 设置下 | **vs. StreamRL: 1.25× avg, up to 1.54×**<br>vs. AReaL: 1.21× avg, up to 1.41× | 在异构负载中仍有效 |
| 异构 pipeline 共存 | Qwen3-0.6B + Llama-3.1-8B | 吞吐提升显著，且单 pipeline 延迟增加 ≤1.34× |

> 📊 图12、图13 展示了在多种 pipeline 和模型上的稳定吞吐领先。

---

### 与基线方法的对比结果
| 对比项 | 结果 |
|-------|------|
| **vs. Verl（同步）** | 吞吐平均提升 **1.56×~1.85×**，尤其在 rollout 占比较高（>70%）的任务中增益更大 |
| **vs. RollMux（时分复用）** | 平均高出 **1.27×**，因 RollMux 无法解决训练阶段串行化问题 |
| **vs. StreamRL/AReaL（异步）** | 在保持更低数据陈旧性风险的前提下，实现更高吞吐，说明 **multiplexing 可独立于 async 优化生效** |
| **vs. JigsawRL-MuxServe** | 证明仅复用 rollout 不够，**fine-grained sub-stage multiplexing 是关键** |

---

### 消融实验结果
#### （1）Dynamic Sub-stage Multiplexing 消融（图19）
- **Verl（串行）**：Rollout 期间 MFU 极低（<5%）
- **MuxServe（仅 rollout 复用）**：部分改善，但 Training 仍串行，限制吞吐
- **JigsawRL（全子阶段复用）**：MFU 提升至 **3.89%**，较 Verl ↑1.7×，验证了 compute/memory 互补调度的有效性

#### （2）Inter-DP Workload Migration 消融（图20）
- **无迁移**：长尾 rollout 分布在所有 DP worker 上，持续干扰其他 pipeline
- **启用迁移**：将长尾样本迁移到特定 worker 后，其余 worker 可自由参与 Training multiplexing，**整体 step latency 下降明显**

#### （3）Look-ahead Scheduling vs. Greedy（图11）
- **Greedy 调度**：短期最优但破坏后续 co-location 机会
- **Look-ahead 调度**：通过前瞻关键路径，保留未来高价值复用机会，最终完成时间更短

---

## 4. 关键结论和发现

### 主要发现
1. **Stage-level 调度存在根本性盲区**：Rollout 内部的 Prefill/Decoding 差异、Tool Calling 引入的 CPU 等待等，都需 sub-stage 级抽象才能捕捉。
2. **Sub-stage 具有强互补性**：Training 是 compute-bound，Decoding 是 memory-bound，二者可安全并发执行。
3. **Pipeline Multiplexing 是新的高效并行范式**：相比单纯扩大 TP/DP，它提供了更高性价比的扩展路径。
4. **长尾问题可通过迁移而非丢弃解决**：JigsawRL 通过迁移而非丢弃 long-tail 样本，在不影响收敛性的前提下提升了利用率。
5. **Temporal Consistency 存在于 rollout 行为中**：相邻 step 的 batch size 曲线高度相似，使得基于历史 profile 的预测调度可行。

---

### 方法的局限性
- 当前主要针对 **dense 模型**（非 MoE），未考虑专家稀疏激活带来的新不平衡。
- 依赖 **CUDA Graph** 和 **NVIDIA Green Context** 等底层支持，硬件依赖较强。
- **延迟有一定增加**：虽然吞吐大幅提升，但单 pipeline 的 step latency 平均增加约 **1.48×**，不适合对延迟极度敏感的应用。
- **初始化开销**：需要离线构建 slowdown lookup 表，虽轻量但仍需预热。

---

### 未来工作方向
1. **扩展至 MoE 架构**：利用专家级不平衡进行更细粒度的 multiplexing。
2. **支持 LoRA-style 参数高效微调**：多个 adapter 共享 base model，天然适合 multiplexing。
3. **自动化资源预算搜索**：当前 SM/Mem 划分是手动离散化，未来可引入强化学习自动探索最优配置。
4. **Serverless 与弹性调度集成**：结合云原生环境实现按需启停与抢占式资源利用（如 RLBoost [56]）。

---

> ✅ **总结一句话**：  
> JigsawRL 通过提出 **Sub-Stage Graph** 抽象和 **Pipeline Multiplexing** 新范式，首次将 RL pipeline 的调度精细到 sub-stage 级别，实现了高达 **1.85× 的吞吐提升**，为 LLM 后训练提供了一条高性价比的规模化路径。

</details>

---

### 3. [SDSL-Solver: Scalable Distributed Sparse Linear Solvers for Large-Scale Interior Point Methods](https://arxiv.org/abs/2604.23979)

**Authors**: Shaofeng Yang, Yunting Wang, Yingying Cheng, Fan Zhang, Xin He, Guangming Tan  
**Category**: cs.DC  
**Published**: 2026-04-28  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2604.23979v1  

#### Abstract
The solution of sparse linear systems constitutes the dominant computational bottleneck in interior point methods (IPMs), frequently consuming over 70\% of the total solution time. As optimization problems scale to millions of variables, direct solvers encounter prohibitive fill-in, excessive memory...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**SDSL-Solver: Scalable Distributed Sparse Linear Solvers for Large-Scale Interior Point Methods**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大规模优化问题中，**Interior Point Methods (IPMs)** 是求解线性规划、二次规划等凸优化问题的核心算法。然而，其计算瓶颈在于每轮迭代中需要求解一个大型稀疏线性系统 $Ax = b$，该步骤通常占总时间的 **70%以上**。

随着问题规模扩展至百万级变量，传统方法面临以下挑战：
- **Direct solvers**（如 PARDISO）因矩阵填充（fill-in）导致内存爆炸且并行扩展性差；
- **Iterative solvers**（如 PETSc + Block Jacobi）虽可分布式运行，但在病态（ill-conditioned）系统上收敛缓慢甚至失败；
- 缺乏对 IPMs 迭代过程中系数矩阵缓慢变化特性的有效利用。

---

### 🚀 提出的新方法与创新思路

作者提出 **SDSL-Solver** —— 一种面向 IPMs 的可扩展分布式稀疏线性求解器框架，具备四大核心技术：

#### （1）双模式分布式架构自适应选择
- **Block Jacobi (BJ)**：适用于对角占优、良态系统，具有高并行度和低通信开销。
- **Bordered Block Diagonal (BBD)**：针对病态系统设计，通过 Schur complement 技术保留子域间耦合关系，提升预条件质量。
- **自适应切换机制**：初始使用 BJ；若求解失败则自动切换为 BBD，并保持后续迭代一致。

> ✅ 创新点：结合两种方法优势，在效率与鲁棒性之间实现动态平衡。

#### （2）基于数值分布的稀疏过滤（Numerics-based Sparse Filtering）
- 分析矩阵非对角元绝对值相对于对角元的比例；
- 若满足 $|a_{ij}| < \tau \cdot |a_{ii}|$ 且 $|a_{ij}| < \tau \cdot |a_{jj}|$，则丢弃该元素；
- 构建更稀疏的预条件矩阵 $P$，显著降低 ILU/Cholesky 因子化成本。

> ✅ 创新点：利用 IPMs 中矩阵条目幅值差异大的特性，智能剪枝以加速预条件构造。

#### （3）对角修正技术（Diagonal Correction）
- 在预条件矩阵 $P$ 上添加小常数 $\delta$：  
  $$
  p_{ii} := p_{ii} + \delta \cdot \text{sign}(p_{ii})
  $$
- 强化对角占优性，改善病态系统的谱性质，提高 Krylov 收敛速度。

> ✅ 创新点：仅修改预条件器而不改变原方程，保证最终解精度不受影响。

#### （4）预条件器重用策略（Preconditioner Reuse）
- 利用 IPMs 各次迭代间矩阵变化缓慢的特点；
- 复用符号分解（symbolic factorization），仅更新数值因子；
- 更激进情况下可跨多步复用完整预条件器。

> ✅ 创新点：大幅摊销昂贵的预条件构建代价，尤其适合中期迭代阶段。

---

### 🔍 相比现有方法的优势

| 方面 | SDSL-Solver | PETSc (Block Jacobi) | PARDISO (Direct) |
|------|-------------|------------------------|------------------|
| 并行扩展性 | 高（支持多节点 MPI+OpenMP） | 高 | 低（共享内存为主） |
| 内存效率 | 高（稀疏过滤 + 迭代法） | 高 | 差（fill-in 严重） |
| 数值鲁棒性 | 强（支持 diagonal correction） | 弱（忽略全局耦合） | 强但易崩溃于极端病态 |
| 对 IPMs 特性适配 | 完全适配（reuse + 自适应） | 黑箱处理 | 黑箱处理 |

---

## 2. 核心实验方法和设置

### 📊 数据集与测试问题分类

共使用两类基准问题，总计 **22 个实例**，涵盖公开与华为私有数据集：

| 类别 | 代表数据集 | 特征 |
|------|-----------|------|
| **Block Jacobi Benchmarks**（良态） | `PageRank_1m`, `PageRank_5m`, `com-youtube`, `cit-patents`, `L2CTA3D`, `thk_48/63` | 对角占优，适合 BJ 方法 |
| **BBD Benchmarks**（病态） | `NetworkPlan_*`, `ScucRelax_*`, `BuildingEnergy`, `L1_sixm250obs` | 条件数极高，需强预条件 |

> 所有问题维度从数万到 **超过五百万** 不等。

---

### ⚙️ 实验设置

#### 硬件平台
- **X86 集群**：4 节点 × 52 核（Intel CPU），730GB 内存/节点
- **Kunpeng 集群**：4 节点 × 128 核（鲲鹏-920），521GB 内存/节点

#### 求解器框架集成
- 所有实验嵌入华为 **OptVerse IPM solver** 框架；
- 替换默认的 PARDISO 模块为 SDSL-Solver，其余流程不变（KKT 构造、步长搜索等）。

#### 基线对比方法
| 基线 | 描述 |
|------|------|
| **PETSc** | 分布式迭代求解器，采用 Row-wise 分区 + Block Jacobi + ILU(0) 预条件 |
| **PARDISO (MKL)** | 单节点直接求解器，默认使用 LDLT 或 LU 分解，多线程并行 |

#### 评估指标
- **单次求解时间**（Single-solve wall-clock time）
- **端到端 IPMs 总耗时**（End-to-end solve time）
- **Krylov 迭代次数**
- **加速比**（Speedup over baselines）
- **收敛稳定性**（是否出现 numerical breakdown）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### （1）多节点加速比（4节点配置下平均表现）

| 方法 | 相比 PETSc | 相比 PARDISO |
|------|------------|--------------|
| **SDSL-Solver (Block Jacobi)** | **6.23×** | **97.54×** |
| **SDSL-Solver (BBD)** | **7.77×** | **5.85×** |

> 💡 注：PARDISO 在部分大问题上因 fill-in 超出 32-bit 整数索引限制而完全无法运行。

#### （2）最大加速比示例
- 在 `com-youtube` 上，相比 PETSc 达到 **53.27×** 加速；
- 在 `L2CTA3D` 第10步，相比 PARDISO 实现 **319.35×** 加速。

#### （3）端到端性能（X86 平台）

| 问题 | SDSL-Solver 时间（秒） | 相当于 PARDISO 的加速比 |
|------|-------------------------|--------------------------|
| `L2CTA3D` | 176.0 | ~9.09× |
| `thk_48` | 3,780.1 | ~6.83× |
| `thk_63` | 1,589.0 | ~1.32× |

> ✅ 所有问题均成功收敛，而 PARDISO 在多个案例中报 “Numerical Issues Detected”。

---

### 🔬 消融实验结果

#### （1）稀疏过滤效果（Table 2）
- 在 `L2CTA3D` 上，应用 $\tau=10^{-3}$ 过滤后，预条件矩阵非零元减少至原始的 **16%~26%**；
- 总 IPMs 时间从 916s 降至 377s，获得 **2.43×** 加速。

#### （2）对角修正作用（Table 3）
- PARDISO 在全部 **15 个病态问题** 上均未能收敛；
- SDSL-Solver 应用 $\delta=10^{-12}$ 后，**全部成功收敛至最优解**。

#### （3）预条件器重用收益（Table 4）

| 问题 | 是否启用重用 | 时间（秒） | 加速比 |
|------|---------------|------------|--------|
| `NetworkPlan_6` | 否 | 441.32 | — |
| | 是 | 125.3 | **3.53×** |
| `L1_sixm250obs` | 否 | 27.94 | — |
| | 是 | 20.70 | **1.35×** |

> 🌟 特别地，在 `L1_sixm250obs` 中，旧预条件器引入轻微扰动反而起到隐式正则化作用，将迭代步数从 53 减少到 28。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **IPMs 中的稀疏线性求解是决定整体可扩展性的关键瓶颈**，必须结合算法与系统层面协同优化。
2. **Krylov 子空间方法 + 自适应预条件** 可在保证精度的同时实现远超 direct solvers 的性能和可扩展性。
3. **稀疏过滤 + 对角修正 + 预条件重用** 三者协同，能显著降低每次迭代的计算开销。
4. **Block Jacobi 与 BBD 的自适应切换机制** 成功兼顾了高效性与鲁棒性，特别适合工业级复杂问题。
5. **SDSL-Solver 在真实 IPM 框架中稳定运行**，并在多个千万级规模问题上实现百倍级加速。

---

### ⚠️ 局限性

1. **BBD 方法中 Schur complement 的接口矩阵可能变稠密**，成为根节点计算瓶颈；
2. 当前稀疏过滤阈值 $\tau$ 和对角修正参数 $\delta$ 依赖人工设定，缺乏自适应调参机制；
3. 在 **Kunpeng 平台** 上，由于浮点舍入误差累积，某些问题（如 `thk_48`, `thk_63`）未能收敛，暴露了迭代法在异构硬件上的数值稳定性挑战；
4. 尚未支持 GPU/NPU 加速，SpMV 和三角求解仍有进一步优化空间。

---

### 🔮 未来工作方向

1. **GPU/NPU 加速**：将 SpMV 和预条件应用内核移植至异构设备，充分发挥硬件吞吐能力；
2. **自适应参数选择**：开发基于运行时诊断的自动 $\tau$, $\delta$ 调整策略，减少人工干预；
3. **扩展至更广优化类型**：支持 Second-Order Cone Programming (SOCP) 和 Semidefinite Programming (SDP)；
4. **fault tolerance 与 checkpointing**：增强大规模分布式下的容错能力；
5. **与自动微分系统集成**：打造端到端 AI-native optimization pipeline。

---

> ✅ **总体评价**：  
> SDSL-Solver 是首个将 **数值感知预条件设计**、**分布式架构选择** 与 **IPMs 动态特性利用** 深度融合的稀疏求解器框架，在性能、鲁棒性和可扩展性方面全面超越主流工具，为工业级大规模优化提供了坚实基础。

</details>

---

### 4. [ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection](https://arxiv.org/abs/2604.23585)

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.23585v1  

#### Abstract
Financial institutions must track over 60,000 regulatory events annually, overwhelming manual compliance teams; the industry has paid over USD 300 billion in fines and settlements since the 2008 financial crisis. We present ComplianceNLP, an end-to-end system that automatically monitors regulatory c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
金融合规领域面临三大挑战：
- **监管文本量巨大**：全球金融机构每年需处理超过6万条监管更新，远超人工团队处理能力。
- **跨框架合规差距检测困难**：机构需同时遵守多个监管框架（如SEC、MiFID II、Basel III），且条款间存在大量交叉引用（cross-reference），导致义务提取和政策对齐复杂。
- **大模型幻觉风险高**：LLM在生成合规建议时易产生事实错误，影响可信度。

### **提出了什么新方法或新思路**
提出 **CoMPLIANCENLP**，一个端到端的自动化合规监控系统，集成以下三大创新模块：

1. **KG-Augmented RAG Pipeline**
   - 构建包含 **12,847个条款节点** 和 **34,219条边** 的 **Regulatory Knowledge Graph (RKG)**，显式建模条款间的 `AMENDS`, `SUPERSEDES`, `CROSSREFERENCES` 等关系。
   - 在检索阶段引入 **KG重排序机制**：结合向量相似度与图距离，提升相关条款召回率。

2. **Multi-task Obligation Extraction**
   - 基于 **LEGAL-BERT** 编码器，联合训练三个任务：
     - **NER**：识别23类金融实体（如 `CAPITAL_REQUIREMENT`, `JURISDICTION`）。
     - **Deontic Classification**：判断义务类型（OBLIGATION, PROHIBITION等）。
     - **Cross-reference Resolution**：解析条款间的引用链。
   - 共享编码器实现多任务协同学习。

3. **Compliance Gap Analysis with Severity-Aware Scoring**
   - 将提取出的义务与内部政策进行对齐，计算 `alignment score`。
   - 使用 **LLaMA-3** 生成器分类为 COMPLIANT / PARTIAL GAP / FULL GAP，并输出补救建议。
   - 引入 **MiniCheck** 进行生成结果的事实核查，确保输出可追溯至原始条款。

此外，在生产优化方面：
- 采用 **知识蒸馏**（70B → 8B）和 **Medusa投机解码**，实现 **2.8× 推理加速**。
- 发现监管文本熵值低（H=2.31 bits），使得 **Medusa草案token接受率达91.3%**，显著高于通用文本（82.7%）。

### **相比现有方法的优势**
| 维度 | 现有方法（如GPT-4o+RAG） | CoMPLIANCENLP |
|------|--------------------------|---------------|
| **准确性** | 易受幻觉影响，依赖纯embedding检索 | KG增强检索+多任务提取，F1提升+3.5 |
| **结构化理解** | 难以处理深层交叉引用 | 显式建模KG，支持多跳推理 |
| **生产效率** | 推理延迟高 | 蒸馏+Medusa实现sub-second p50延迟 |
| **部署可行性** | 缺乏完整端到端流程 | 支持并行运行、信任校准、持续维护 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **REGOBLIGATION**
   - 规模：1,847条标注句子（SEC: 712, MiFID II: 614, Basel III: 521）
   - 任务：NER（23类）、Deontic分类（4类）、Cross-reference resolution
   - 来源：真实监管文件，由三位合规专家标注，IAA K=0.84

2. **GAP-BENCH**
   - 规模：423个义务-政策对（Compliant: 210, Partial Gap: 128, Full Gap: 85）
   - 任务：合规差距检测
   - 来源：某金融机构匿名审查记录，K=0.81

3. **外部基准测试**
   - ObliQA (5,574 questions)
   - COLING 2025 Challenge (312 questions)

### **实验设置和评估指标**
- **主评估阈值**：gap detection threshold $\theta = 0.6$
- **部署优化阈值**：$\theta = 0.45$（优先提高召回率）
- **关键指标**：
  - **F1 Score**（NER, Deontic, Gap Detection）
  - **EM / F1**（Regulatory QA）
  - **Grounding Accuracy**（MiniCheck验证结果与人工标注的一致性）
  - **p50/p99 Latency**（推理延迟）
  - **End-to-End F1**：考虑误差传播的真实场景性能

### **基线方法对比**
| 类别 | 基线模型 |
|------|--------|
| **无检索** | GPT-4(5-shot), GPT-4o(5-shot), LEGAL-BERT, FinBERT |
| **带检索** | GPT-4+RAG, GPT-4o+RAG, LLaMA-3-8B+RAG, LLaMA-3-70B*, RIRAG |
| **消融对照** | w/o KG reranking, w/o multi-task, w/o MiniCheck |

所有实验报告3次随机种子平均值，显著性检验采用paired bootstrap (n=10,000)。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | CoMPLIANCENLP | 最佳基线 (GPT-4o+RAG) | 提升 |
|------|----------------|------------------------|------|
| **Gap Detection F1** ($\theta=0.6$) | **87.7** | 84.2 | **+3.5** |
| **NER F1** | **91.3** | 88.6 | **+2.7** |
| **Deontic F1** | **92.7** | 90.5 | **+2.2** |
| **Regulatory QA F1** | **71.9** | 66.8 | **+5.1** |
| **Grounding Accuracy** | **94.2%** | ~85.1% | **+9.1pp** |
| **p50 Inference Speed** | **659 ms** | 1,847 ms | **2.8× faster** |

> 注：LLaMA-3-70B*为教师模型（未蒸馏前），CoMPLIANCENLP仍优于其F1表现。

### **与基线方法的对比结果**
- 相比 **GPT-4o+RAG**（相同检索模块，无KG/多任务/MiniCheck）：
  - 所有任务均显著提升（p<0.05）
  - 特别是在 **NER** 和 **Gap Detection** 上优势明显，说明领域特定设计至关重要。
- 相比 **LLaMA-3-8B+RAG**（同学生模型，无领域组件）：
  - NER +3.4 F1，Gap +4.2 F1 → 表明性能增益来自 **KG重排序、多任务训练、MiniCheck** 而非仅模型容量。

### **消融实验结果**
| 消融配置 | Gap F1 | ΔF1 | 主要影响 |
|---------|-------|-----|--------|
| 完整系统 | 87.7 | — | — |
| **w/o KG reranking** | 83.1 | **-4.6** | 最大降幅，证明KG结构知识关键 |
| w/o multi-task | 84.9 | -2.8 | 多任务协同提升NER与XRef |
| w/o MiniCheck | 87.2 | -0.5 | 对F1影响小，但**接地准确率从94.2%降至86.7%** |

> **加法分析**进一步验证：
- KG重排序贡献最大（+2.2~2.5 F1）
- 多任务提取次之（+0.7~0.8 F1）
- MiniCheck主要提升接地质量而非F1

### **端到端误差传播分析**
在150份文档上模拟全流程：
- **End-to-End F1**: **83.4**（vs. 理想87.7）
- 主要误差来源：
  - NER边界错误（68%）
  - Cross-reference失败（18%）
  - Deontic误判（4%）

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **结构化知识优于纯嵌入检索**
   - KG重排序带来最大边际收益（+4.6 F1），尤其在处理多跳交叉引用时不可或缺。
   - 例如，Basel III中“d424 → d295 → CRR Art. 412”链条若无KG将完全丢失上下文。

2. ✅ **公式化语言利于高效推理**
   - 监管文本词汇受限、熵值低（H=2.31），使 **Medusa投机解码接受率达91.3%**。
   - 启示：该技术可能同样适用于医疗记录、专利文书等低熵领域。

3. ✅ **分析师更关注召回率而非F1**
   - 单次漏检会严重削弱信任，因此部署采用 **recall-optimized threshold ($\theta=0.45$)**。
   - 并行运行期间达到 **96.0%估计召回率** 和 **90.7%精度**。

4. ⚠️ **GRC集成难度超过模型开发**
   - 将系统接入已有GRC平台耗时约3个月，相当于整个模型研发周期。
   - 主要障碍：旧有分类体系（847类）与业务功能不匹配，需构建双向映射层。

5. 🤝 **组织采纳需分阶段建立信任**
   - 初期分析师复核率达78%，后期降至23%。
   - 最有效策略是发布每周“系统vs人工”对比报告，积累可信证据。

### **局限性**
1. 当前覆盖 **3个框架**（约占年度更新量48%），尚未扩展至全部60K+法规。
2. 主要评估集 **GAP-BENCH仅来自单一机构**（423样本），Full Gap F1置信区间较宽（±4.2）。
3. 仅支持 **英文文本**，未涵盖其他语言。
4. 用户研究规模有限（12位分析师，96次更新），且为非盲态设计。
5. **p99延迟为1,082ms**，略高于sub-second目标，尾部延迟主要由深KG遍历引发。
6. 生产召回率为估算值，存在结构性不确定性（两方可能共享盲点）。

### **未来工作方向**
1. 扩展至更多监管框架（目标：全覆盖60K+年更新）。
2. 完成跨机构 **Gap-Bench扩展版**（预计~1,200样本，Q3 2026发布）。
3. 推进部署阶段：
   - Phase 3（Q3 2026）：影子部署，逐步赋予低风险决策自主权。
   - Phase 4（Q1 2027）：全生产部署，仅关键发现需人工复核。
4. 加强对新型监管结构的主动监测（如嵌套条件义务、时间触发规则）。
5. 开发更细粒度的 **clause-level grounding model**，避免条款混淆（如Art. 25(1) vs 25(2)）。

> 🔗 **代码与数据**：https://github.com/bettyguo/ComplianceNLP

</details>

---

### 5. [MetaGAI: A Large-Scale and High-Quality Benchmark for Generative AI Model and Data Card Generation](https://arxiv.org/abs/2604.23539)

**Authors**: Haoxuan Zhang, Ruochi Li, Yang Zhang, Zhenni Liang, Junhua Ding, Ting Xiao, Haihua Chen  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.23539v1  

#### Abstract
The rapid proliferation of Generative AI necessitates rigorous documentation standards for transparency and governance. However, manual creation of Model and Data Cards is not scalable, while automated approaches lack large-scale, high-fidelity benchmarks for systematic evaluation. We introduce Meta...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- **生成式AI（GenAI）模型与数据卡文档缺失**：随着GenAI模型数量激增（如Hugging Face上已有超200万模型），大量开源模型缺乏标准化的 **Model Card** 和 **Data Card** 文档，导致透明度不足、可复现性差、合规审计困难。
- **自动化生成质量不可靠**：现有自动文档生成方法存在严重幻觉（hallucination）、不完整性和主观偏差，且缺乏大规模、高质量的基准来系统评估其性能。

### **提出了什么新方法或新思路**
- **MetaGAI**：一个大规模、高保真度的基准数据集，用于评估自动化 **Model & Data Card** 生成任务。
  - 包含 **2,541个经过验证的三元组**（论文 + GitHub + Hugging Face）。
  - 采用 **多智能体框架**（Multi-Agent Framework）构建：
    - **Retriever Agent**：从多源文本中检索与Schema字段相关的证据片段。
    - **Generator Agent**：使用多个不同架构的LLM（OLMo-3-7B, Llama-3.1-8B, Qwen2.5-7B）并行生成草稿，提升多样性。
    - **Editor Agent**：由更大模型（如GPT-OSS-20B）整合多个草稿，交叉验证原始证据，合并非冗余信息，输出最终高保真卡片。
  - 引入 **四维人工在环验证**（human-in-the-loop validation）确保质量：
    1. 检索策略有效性（D1）
    2. 生成器分歧分析（D2）
    3. 编辑器效果评估（D3）
    4. 编辑器架构选择（D4）

### **相比现有方法的优势**
| 维度 | 传统方法（如CARDGEN） | MetaGAI |
|------|------------------------|--------|
| 数据来源 | 单一来源（仅论文） | 多源三角验证（论文 + GitHub + HF） |
| 地面真值构建 | 聚合已有文档 | 通过多智能体+人工验证主动构造 |
| 生成方式 | 单模型生成 | 多模型集成 + 编辑器精炼 |
| 评估标准 | 自动化指标为主 | 结合LLM-as-a-Judge + 人类专家评分 |
| 规模与质量 | 小规模、噪声大 | 大规模（2,541条）、高保真 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MetaGAI Benchmark**：自建数据集，源自对 **arXiv、GitHub、Hugging Face** 的系统性三角匹配。
  - 初始候选：15,727个
  - 经过规范过滤后：4,068个
  - 最终通过语义一致性验证保留：**2,541个高质量三元组**
- 数据分布：
  - 时间跨度：2019–2025年，逐年增长
  - 领域集中于 **Computer Vision (cs.CV)** 和 **Computational Linguistics (cs.CL)**

### **实验设置和评估指标**

#### **任务定义**
- 输入：仅学术论文文本 $ P $
- 输出：结构化的 Model/Data Card $ C $
- 目标函数：$ C = f_\theta(P) $，最小化与多源参考卡 $ C_{gt}(P,G,H) $ 的差异

#### **评估指标**
| 类型 | 指标 | 描述 |
|------|------|------|
| **自动化指标** | Completeness | 字段召回率：$\frac{|K(C) \cap K(C_{gt})|}{|K(C_{gt})|}$ |
| | ROUGE-L | 词法重叠度 |
| | BERTScore (F1) | 语义相似性 |
| **人工/LLM评估** | LLM-as-a-Judge | 使用 GPT-OSS-120B、Llama-3.3-70B、Qwen3-235B 对生成内容进行五维打分（1–5 Likert量表）：<br>• Faithfulness（忠实性）<br>• Relevance（相关性）<br>• Accuracy（准确性）<br>• Consistency（一致性）<br>• Usefulness（实用性） |
| **成本效率** | Cost Index | 归一化推理成本（基于每百万输入token + 20万输出token的定价） |

#### **基线方法对比**
- **Open-Weight Models**：
  - Dense：Mistral-Small-24B, Gemma-3-27B, Qwen3-32B
  - MoE：GPT-OSS-20B, Nemotron-Nano-30B-A3B, **Qwen3-30B-A3B-Instruct**
- **Closed-Source Models**：
  - GPT-5-Mini / GPT-5-Nano
  - Gemini-2.5-Flash / Flash-Lite

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见Table 3）**

| Model | Model Card Qual. Avg | Data Card Qual. Avg | Completeness (Model) | Cost Index |
|-------|------------------------|----------------------|------------------------|-----------|
| **Qwen3-30B-A3B-Instruct (MoE)** | **4.55** | **4.30** | **0.786** | **0.15** |
| GPT-5-Mini | 4.28 | 4.07 | 0.556 | 0.65 |
| Gemini-2.5-Flash | 4.00 | 3.83 | 0.443 | 0.80 |
| Gemma-3-27B (Dense) | 3.94 | 3.72 | 0.734 | 1.15 |
| Mistral-Small-24B (Dense) | 3.64 | 3.28 | 0.386 | 0.51 |

> ✅ **Qwen3-30B-A3B-Instruct 在质量和成本上均达到 Pareto 最优**

### **与基线方法的对比结果**
- **MoE 架构显著优于 Dense 模型**：
  - 同属Qwen系列，MoE版（Qwen3-30B-A3B）比Dense版（Qwen3-32B）高出近 **1分**（4.55 vs 3.60）。
  - 成本更低（0.15 vs 1.15），说明稀疏激活更高效。
- **开放权重模型超越闭源API模型**：
  - Qwen3-30B-A3B 质量高于 GPT-5-Mini（4.55 vs 4.28），成本仅为 **1/4.3**。
  - Gemini 系列性价比最低（质量低、价格高）。
- **传统指标失效**：
  - ROUGE-L 与真实质量呈**负相关**：Mistral最高ROUGE-L但质量最低；Qwen3-30B-A3B因抽象能力强，ROUGE-L较低但质量最高。
  - BERTScore 压缩所有模型得分至狭窄区间（0.10–0.20），无法区分优劣。

### **消融实验结果**
#### **(1) 编辑器有效性（D3）**
- 编辑后的卡片质量显著提升：
  - **Bench_GPT-OSS** 平均得分为 **4.41**，远高于原始生成器草稿（Raw Baseline: 3.71）
  - 提升幅度达 **15–20%**，证明编辑器能有效减少幻觉、增强完整性。

#### **(2) 多源上下文 ablation（G.2）**
- 即使提供完整的多源上下文（Paper + GitHub + HF），模型表现也无明显改善（Δ < 0.02）。
- 表明当前LLM的瓶颈在于**长上下文推理能力**，而非信息缺失。

#### **(3) 信息密度影响（G.4）**
- 地面真值中的信息密度（Completeness）解释了 **31% 的 BERTScore 方差**（R²=0.31）。
- 抽象字段（如Ethical Considerations）因原文描述稀疏，生成难度更高。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **稀疏MoE架构在成本-质量权衡中占据绝对优势**：
   - 特别是当参数规模 ≥30B 时，MoE展现出更强的信息合成能力。
2. ❌ **传统文本匹配指标（ROUGE-L, BERTScore）不能反映生成质量**：
   - 它们惩罚抽象归纳，偏好逐字复制，与实际需求背道而驰。
3. 🔁 **存在“忠实性-完整性”根本权衡（Faithfulness-Completeness Trade-off）**：
   - 高质量模型（如Qwen3-30B-A3B）虽几乎无幻觉（Faithfulness≈5），但仍遗漏约21%字段（Completeness=0.786）。
4. 📉 **Data Card 生成普遍难于 Model Card**：
   - 所有模型在Data Card上的得分平均低0.2–0.4分，因其要求更细粒度的生命周期文档（隐私、安全、维护等），而论文对此描述极少。

### **方法的局限性**
- **仅限文本模态**：未处理图表、表格等非文本信息。
- **孤立任务建模**：未考虑论文、模型、数据集之间的复杂依赖关系。
- **潜在风险传播**：若生成卡片被广泛复用，错误可能放大，需加强验证机制。

### **未来工作方向**
- 支持 **多模态理解**（图像、表格解析）
- 构建 **图结构表示** 以建模GenAI生态系统的依赖网络
- 开发更鲁棒的 **验证与纠错机制**
- 探索 **轻量化部署方案** 以支持边缘场景下的实时文档生成

---

> 🔗 **代码与数据已开源**：  
> https://github.com/haoxuan-unt2024/MetaGAI-Benchmark

</details>

---

### 6. [STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator](https://arxiv.org/abs/2604.24544)

**Authors**: Alessio Sordo, Lingxiao Du, Meeka-Hanna Lenisa, Evgeny Bogdanov, Maxim Romanovsky  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.24544v1  

#### Abstract
The increasing reliance on Large Language Models (LLMs) across diverse sectors highlights the need for robust domain-specific and language-specific evaluation datasets; however, the collection of such datasets is challenging due to privacy concerns, regulatory restrictions, and the time cost for man...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前对 **Large Language Models (LLMs)** 的评估严重依赖人工构建的、领域特定或语言特定的基准数据集。然而，这类数据集的创建面临以下挑战：
- **隐私与合规限制**：在金融、医疗等受监管领域难以使用真实数据；
- **高成本与低效率**：手动标注耗时且昂贵；
- **多语言支持不足**：大多数基准以英语为主，非英语数据常通过机器翻译获得，导致“translationese”现象和文化不适应；
- **可扩展性差**：现有自动化方法多基于已有数据增强或匿名化，无法真正实现从零生成。

因此，亟需一种**完全自动、可定制、高质量、多语言支持**的合成数据生成与评估框架。

---

### **提出了什么新方法或新思路**
本文提出 **STELLAR-E** —— 一个端到端（end-to-end）的、由 LLM 驱动的合成数据生成与评估系统，用于为 LLM 应用程序生成高质量的指令-答案对（Instruction-Answer, I&A），并直接进行模型评估。

其核心架构分为两个阶段：
1. **合成数据引擎（Synthetic Data Engine）**  
   改进自 TGRT Self-Instruct 框架，实现可控、可定制的 I&A 数据生成。
2. **评估流水线（Evaluation Pipeline）**  
   结合统计指标与 **LLM-as-a-Judge** 方法，评估生成数据的有效性和挑战性。

---

### **相比现有方法的优势**
| 维度 | STELLAR-E 的优势 |
|------|------------------|
| **数据来源独立性** | 不依赖任何预存数据集，避免数据污染与隐私泄露风险 |
| **高度可定制性** | 可灵活控制生成数据的语言、语义、格式、数量及难度 |
| **全流程自动化** | 覆盖主题生成 → 指令生成 → 回答生成 → 质量优化 → 多维度评估 |
| **质量保障机制** | 引入反馈循环（Feedback Loop）、**DVE**（多样性增强）和 **DFE**（难度增强）模块 |
| **多语言原生支持** | 直接生成目标语言数据，而非翻译英文数据，保留语言与文化特异性 |
| **可扩展性强** | 支持大规模生成（数千至上万条 I&A 对），适用于 CI/CD 中的 LLMOps 流程 |

> ✅ **创新亮点**：首次将 **G-Eval** 改造为精细化评分工具（1–10 分制、温度设为 0、分项打分），并在多个阶段引入该机制进行闭环优化。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **真实基准（Ground Truth）**：
  - `Mintaka_en_real`：原始英文版 Mintaka 数据集
  - `Mintaka_it_real`：专业人工翻译的意大利语版本（作为高质量非英语基准）
- **对照组**：
  - `mintaka_it_translated`：将英文 Mintaka 通过 API 自动翻译成意大利语，模拟传统翻译路径
- **合成数据集**：
  - `mintaka_en_synthetic` 和 `mintaka_it_synthetic`：由 STELLAR-E 在相同 Question Types 下生成的英/意双语数据

所有数据集最终均随机采样 **1,500 条 I&A 对** 进行公平比较。

---

### **实验设置**
- **生成参数**：
  - 使用 8 种 Question Types（QTs）
  - 每轮迭代生成 50 条指令，共 50 轮
  - G-Eval 阈值设定为 **T = 8**（满分 10），仅保留达标样本
  - DVE 相似度阈值固定为 **0.3**
- **模型配置**：
  - **生成模型**：Gemini-1.5-pro-002
  - **过滤与评估模型**：Gemini-2.0-flash-001（高效判断质量）
  - **最终 Judge 模型**：Gemini-2.5-Pro（用于 meta-evaluation）

---

### **评估指标**
| 指标 | 说明 |
|------|------|
| **ROUGE-L** | 衡量生成答案与参考答案之间的最长公共子序列，反映词汇重叠程度 |
| **BERTScore F1** | 基于上下文嵌入的语义相似度，尤其适合跨语言评估 |
| **Answer Relevance** | 判断回答是否紧扣问题，避免冗余或离题 |
| **Custom G-Eval** | 改进版 LLM-as-a-Judge，综合评估 Accuracy、Relevance、Completeness，输出 0–10 分 |

---

### **基线方法对比**
- **Real Dataset**：人类标注的真实数据（黄金标准）
- **Translated Dataset**：机器翻译得到的非英语数据
- **Synthetic Dataset (w/o DVE/DFE)**：未启用多样性与难度增强的合成数据
- **Synthetic Dataset (w/ DVE)**：启用多样性增强
- **Synthetic Dataset (w/ DVE & DFE)**：完整启用两项增强机制

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（强模型 vs 弱模型）**

#### **强模型（Gemini 2.5 Flash）结果（见 Table 3）**
| 数据集 | G-Eval 平均得分 | 相较 Real 的 Δ |
|--------|------------------|---------------|
| Real English | 8.17 | — |
| Synthetic EN (DVE+DFE) | 8.74 | **+5.7%** |
| Real Italian | 8.00 | — |
| Synthetic IT (DVE+DFE) | 8.59 | **+5.8%** |

✅ 合成数据在 G-Eval 上略高于真实数据，差距约 **+5.7%**，表明其具备相当的评估能力。

#### **弱模型（Llama 2 Chat 13B）结果（见 Table 4）**
| 数据集 | G-Eval 平均得分 | 相较 Real 的 Δ |
|--------|------------------|---------------|
| Real English | 5.69 | — |
| Synthetic EN (DVE+DFE) | 6.78 | **+10.9%** |
| Real Italian | 4.28 | — |
| Synthetic IT (DVE+DFE) | 4.35 | **+0.7%** |

⚠️ 小模型在合成数据上表现提升更明显，尤其在英语中高出 **+10.9%**，暗示合成数据可能对小模型“过于友好”。

---

### **与基线方法的对比结果**
| 观察点 | 发现 |
|-------|------|
| **翻译数据 vs 合成数据** | 机器翻译的 Italian 数据平均 G-Eval 得分 **低于真实数据（+2.3%）**，说明翻译引入噪声；而合成数据更接近真实水平 |
| **DVE/DFE 效果显著** | 启用 DVE+DFE 后，G-Eval 差距从 +12.6% 缩小至 +5.7%，证明难度与多样性增强有效提升了挑战性 |
| **Rouge-L 显著下降** | DVE+DFE 数据上的 Rouge-L 分数最低，说明模型生成的答案句式更多样，不再简单复制模板 |
| **BERTScore 接近真实数据** | 在 DVE+DFE 设置下，BERTScore F1 与真实数据基本持平甚至略优，显示语义一致性良好 |

---

### **消融实验结果**
| 配置 | G-Eval 增幅（vs Real） | 说明 |
|------|------------------------|------|
| 无 DVE/DFE | +12.6% | 基础合成数据明显“太容易” |
| +DVE | ~+9.7% | 多样性提升有助于降低过拟合 |
| +DVE+DFE | **+5.7%** | 难度增强是缩小差距的关键因素 |
| 单独 DFE | 未报告 | 但 DFE 通过 paraphrase 提升复杂性，使指令更具对抗性（adversarial） |

> 🔍 **分解分析**：DFE 显著降低了 Relevance 得分（-7.7%），使其更贴近真实数据分布，说明它成功增加了“偏离主题”的风险，从而更好测试模型稳定性。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **STELLAR-E 能生成与真实基准性能相当的合成评估数据集**，平均 G-Eval 差距仅为 **+5.7%**，验证了其作为替代方案的可行性。
2. ✅ **DVE 与 DFE 模块显著提升数据质量与挑战性**，尤其是在抑制简单化、模式化输出方面效果突出。
3. ✅ **直接生成非英语数据优于机器翻译路径**，能更好地保持语言自然性与任务难度。
4. ⚠️ **小模型在合成数据上表现被高估**（如 Llama 13B 英文 Δ=+10.9%），提示合成数据可能存在结构性线索或简化倾向，未能完全复现真实数据的认知负荷。
5. 🌍 **文化特异性仍是挑战**：尽管避免了翻译问题，但仍需人工评估来确认合成内容的文化适配性。

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **潜在系统性偏差** | 所有环节依赖 LLM，可能导致隐性偏见累积，自动化指标难以捕捉 |
| **对小模型“放水”现象** | 合成数据可能无意中包含更容易触发正确响应的关键词或结构 |
| **仅验证单一基准（Mintaka）** | 泛化能力有待在更多领域（如金融、法律）和语言上验证 |
| **缺乏人类主观评估** | 当前结果全靠自动指标，缺少 native speaker 对文化相关性的打分 |
| **未解决训练-评估数据污染风险** | 若将此类合成数据用于训练，再用其评估，可能出现过拟合 |

---

### **未来工作方向**
1. **扩大 meta-evaluation 范围**：在更多模型家族（如 Llama、Qwen、DeepSeek）上测试，减少 self-enhancement bias。
2. **引入人类评估**：组织 native speakers 对意大利语等非英语 I&A 对进行文化适宜性与真实性评分。
3. **拓展至 RAG 场景**：生成 synthetic source documents，并构建基于文档的 I&A 对，用于评估 Retrieval-Augmented Generation 系统。
4. **探索生成器/裁判模型集成（Ensemble）**：使用多个 judge LLM 投票，提高评估可靠性。
5. **应用于 LLMOps 监控**：将该系统嵌入持续集成流程，实现自动化、高频次的质量回归检测。

---

## **总结**
> **STELLAR-E 是首个实现完全自动化、可定制、端到端 LLM 应用评估的合成数据框架**。它突破了传统依赖人工标注或翻译数据的瓶颈，在保证数据隐私的同时，提供了接近真实基准的评估效力。虽然目前对小型模型存在一定程度的“宽松”，但通过 DVE 和 DFE 的设计已大幅缓解这一问题。未来若结合人类反馈与多模态扩展，有望成为下一代 LLM 质量保障的核心基础设施。

</details>

---

### 7. [Unfolding an Atomistic World: Atomistic Simulation of Reactor Pressure Vessel Steel Across Year-and-Meter Scales](https://arxiv.org/abs/2604.24091)

**Authors**: Haozhi Han, Ruge Zhang, Haoquan Chen, Yifeng Chen, Haipeng Jia, Liang Yuan, Yunquan Zhang, Ting Cao, Yunxin Liu, Ya-Qin Zhang, Kun Li  
**Category**: cs.DC  
**Published**: 2026-04-28  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.24091v1  

#### Abstract
Lifetime prediction of reactor pressure vessel (RPV) steel requires bridging atomistic degradation mechanisms with service-scale spatial and temporal regimes, from Angstroms and picoseconds to meters and decades. Existing engineering-scale models provide long-range reach but rely on fitted degradati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Unfolding an Atomistic World: Atomistic Simulation of Reactor Pressure Vessel Steel Across Year-and-Meter Scales*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

该论文致力于解决**核反应堆压力容器（RPV）钢寿命预测中的跨尺度难题**。具体而言，RPV钢在长期服役过程中会因辐照、热老化和机械载荷而发生不可逆的脆化和强度退化，其根本机制源于原子尺度（angstrom, picosecond）的缺陷演化，但实际工程关注的是米级空间和数十年时间尺度上的宏观性能变化。

传统方法存在两大瓶颈：
- **工程尺度模型**（如rate theory, cluster dynamics）虽能覆盖年-米尺度，但依赖经验拟合的降解规律，牺牲了原子尺度的物理保真度（atomistic fidelity）。
- **原子尺度模拟方法**（如AKMC）虽能精确追踪原子事件，但受限于计算效率，无法跨越到实际服役的时间和空间尺度（例如，模拟一年演化需约30年墙钟时间）。

因此，核心挑战是：**如何在不牺牲原子保真度的前提下，实现从原子尺度到工程尺度的直接模拟？**

---

### 提出了什么新方法或新思路

作者提出了 **AtomWorld** —— 一个面向RPV钢寿命预测的**原子世界建模框架**（atomistic world-modeling framework），通过算法、高性能计算（HPC）和应用三个层面的协同设计，首次实现了跨“年-米”尺度的原子级模拟。

#### 核心创新点：

1. **算法层：将经典AKMC重构为“原子世界模型”**
   - 将系统演化建模为基于**强化学习**（Reinforcement Learning, RL）的Markov决策过程（MDP），其中原子构型为状态，原子跃迁为动作。
   - 引入**全局动力学认知**（Global Kinetic Cognition）：通过一个集中式critic网络学习长时程动力学结构，并指导局部策略决策，避免陷入短周期无效循环（super-basin trapping）。
   - 设计**物理时间对齐机制**（Physical Time Alignment）：基于泊松方程（Poisson equation）重建符合AKMC语义的物理时间增量，确保时间推进的物理一致性。

2. **HPC层：与现代超算架构协同优化**
   - **计算密集化重构**（Compute-Centric Reformulation）：将不规则的速率枚举转换为矩阵运算（GEMV/GEMM），适配现代AI加速器。
   - **异步子晶格并行**（Asynchronous Sublattice Parallelism）：消除全局同步开销，提升负载均衡。
   - **移位通信策略**（Shift Communication Strategy）：将邻域通信分解为维度串行传递，显著降低通信消息数量。

3. **应用层：引入介观体素并行框架**（Mesoscopic Voxel-Parallel Framework）
   - 将全尺寸RPV分解为220万个独立演化的**体素**（voxel），每个体素代表一个具有局部温场、辐照条件的统计代表性微结构单元。
   - 采用**动态任务调度**（Dynamic Voxel Scheduling）处理不同体素间巨大的计算异质性。

---

### 相比现有方法的优势

| 维度 | 传统工程模型 | 传统AKMC | **AtomWorld** |
|------|--------------|----------|----------------|
| **空间尺度** | 米级（✔） | 微米级（✘） | **米级**（✔） |
| **时间尺度** | 年级（✔） | 秒级（✘） | **年级**（✔） |
| **原子保真度** | 低（✘） | 高（✔） | **高**（✔） |
| **可扩展性** | 中等 | 受限于同步与通信 | **极强**（92–97%弱扩展效率） |
| **计算成本** | 低 | 极高（~30年/年模拟） | **1.71天/年模拟** |

> ✅ **AtomWorld首次实现了三者统一：原子保真度 + 工程空间尺度 + 服役时间尺度。**

---

## 2. 核心实验方法和设置

### 使用的数据集

- **训练数据**：来自小规模（200³）AKMC模拟生成的原子跃迁轨迹，涵盖多种材料成分（Cu, Ni, Mn等）、温度（230–400°C）、点缺陷浓度（1–1000 appm）、中子通量（10⁹–10¹¹ n/cm²s）和辐照剂量（10⁻⁴–1 dpa）。
- **输入特征**：原子类型、相对坐标、局部缺陷类型、邻接关系、候选跃迁掩码。
- **测试对象**：中国第三代CAP1400反应堆的ASME SA508 Grade 3 Class 1 RPV钢。

---

### 实验设置和评估指标

#### HPC平台（5台顶级超算）：
| 类型 | 超算名称 | 架构 |
|------|---------|------|
| CPU-based | Lineshine, Tianhe-3, New Sunway | ARM/国产异构处理器 |
| GPU-based | ORISE, Tecorigin | 国产DCU/GPGPU加速器 |

#### 评估指标：
1. **正确性验证**：与参考AKMC轨迹对比**进阶因子**（advancement factor）和能量弛豫路径。
2. **可扩展性**：强扩展（strong scaling）与弱扩展（weak scaling）效率。
3. **峰值性能**：FLOP/s（含FP64利用率）。
4. **时间到解**（Time-to-solution）：模拟**一个服务年**所需墙钟时间。
5. **空间覆盖能力**：总模拟体积（cm³）、原子总数。

#### 基线方法对比：
- **OpenKMC**, **MISA-AKMC**：当前最先进的大规模AKMC系统。
- **State-of-the-art工程模型**：如cluster dynamics等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | AtomWorld 结果 | 对比（State-of-the-art） | 提升倍数 |
|------|----------------|--------------------------|----------|
| **最大空间覆盖** | 9.91 cm³（0.23m厚 × 12.64m高） | ~10⁻⁵ cm³（仅微米级） | **>10⁷×** |
| **最大原子数** | 14.85 quintillion（1.485×10¹⁹） atoms | ~10¹⁵（quadrillion） | **~1000×** |
| **时间到解**（1服务年） | **1.71 天** | ~30 年 | **~6,400×** |
| **60年寿命预测耗时** | ~103 天 | 不可行 | ✅ 首次可达 |
| **峰值性能** | **1.27 EFLOP/s**（Lineshine） | — | 占用48% FP64峰值 |
| **并行效率** | 弱扩展：88–97%，强扩展：85–95% | 典型<80% | 显著提升 |

---

### 与基线方法的对比结果

- 在相同物理条件下，AtomWorld的**微结构演化轨迹**（如Cu沉淀、空位团簇增长）与参考AKMC高度一致（见Fig. 4），验证了其物理正确性。
- 相比OpenKMC，在L=6400的晶格上，AtomWorld实现**452.3×加速**（Fig. 3）。
- 在Lineshine上达到**20.6×强扩展加速比**（96%效率），远超传统AKMC的同步瓶颈限制。

---

### 消融实验（隐含分析）

虽然未明确列出消融表，但文中通过模块化设计展示了各组件必要性：
- 若无**全局critic**，模型易陷入局部循环，无法实现长时程有效演化。
- 若无**shift communication**，通信开销将成为主要瓶颈。
- 若无**voxel-parallel**，无法实现零通信的大规模并行。

---

## 4. 关键结论和发现

### 主要发现

1. **首次实现跨“年-米”尺度的原子级模拟**：AtomWorld打破了原子模拟只能用于“微观解释”的局限，使其成为**可预测的工程工具**。
2. **原子世界建模是可行的新范式**：通过将物理演化重构为RL驱动的世界模型，可在保持原子保真度的同时大幅提升时间推进效率。
3. **算法-HPC-应用协同设计至关重要**：单一层面优化不足以突破尺度壁垒，必须三者联动。
4. **体素化不失为工程近似下的最优解**：在保证局部动力学准确性的前提下，通过统计集合恢复宏观行为，是连接多尺度的有效桥梁。

---

### 方法的局限性

1. **仍依赖体素独立假设**：忽略体素间的长程应力耦合或裂纹传播，适用于早期脆化预测，但难以模拟最终断裂。
2. **训练依赖ab initio势函数**：模型泛化能力受限于训练数据覆盖的化学空间和环境条件。
3. **初始缺陷态需外部设定**：未完全实现从“第一性原理”到“全生命周期”的端到端模拟。
4. **硬件依赖性强**：极致性能需特定架构支持（如SME、HIP等），通用性受限。

---

### 未来工作方向

1. **引入跨体素交互机制**：加入应力场反馈或裂纹萌生-扩展模型，迈向全RPV完整性评估。
2. **扩展至其他核材料**：如燃料包壳、堆内构件等。
3. **结合在线监测数据进行数字孪生**：实现真实反应堆的实时寿命预测。
4. **探索更高效的RL架构**：如transformer-based world model，进一步提升长时程推理能力。
5. **开源框架与社区生态建设**：推动AtomWorld成为核材料模拟的标准工具链。

---

> 🔚 **总结**：  
> *AtomWorld* 不仅是一项技术突破，更标志着**计算材料科学范式的转变**——从“模拟局部机制”转向“推演完整生命周期”。它为核能系统的安全延寿提供了前所未有的科学依据，也为其他极端环境材料的跨尺度建模开辟了新路径。

</details>

---

### 8. [Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis](https://arxiv.org/abs/2604.23072)

**Authors**: Junyan Cheng, Kyle Richardson, Peter Chin  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.23072v1  

#### Abstract
Large language model (LLM) agents are increasingly tasked with complex real-world analysis (e.g., in financial forecasting, scientific discovery), yet their reasoning suffers from stochastic instability and lacks a verifiable, compositional structure. To address this, we introduce Analytica, a novel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis**  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### **解决的问题**
当前基于 **Large Language Model (LLM)** 的智能体在执行复杂现实分析任务（如金融预测、科学发现）时面临两大核心挑战：
- **随机不稳定性 (Stochastic Instability)**：LLM 的自由文本推理过程存在高方差，导致结果不可复现。
- **缺乏可验证的组合结构 (Lack of Verifiable Compositional Structure)**：传统链式思维（Chain-of-Thought, CoT）等方法生成的是非结构化文本，难以追溯、验证和进行“假设分析”（what-if analysis）。

### **提出的新方法与新思路**
论文提出了 **Analytica**，一种基于 **Soft Propositional Reasoning (SPR)** 的新型 LLM 智能体架构。

#### **核心思想：Soft Propositional Reasoning (SPR)**
将复杂的分析任务重构为对一系列“软真值命题”（soft truth values）的估计过程。每个命题的真值是一个介于 0 和 1 之间的概率值，代表其成立的置信度。整个分析过程被形式化为一个树状结构，通过分解、验证和合成三个阶段来最小化估计误差。

#### **Analytica 架构的三大组件**
1. **Analyzer (分析器)**：将根命题递归分解为子命题树，直至得到可验证的叶节点。
2. **Grounder (接地器)**：并行地使用工具增强的 LLM 对叶节点进行验证，并为其分配软真值。论文引入了一种创新的 **Jupyter Notebook Grounder**，能够执行代码、调用 API 进行数据驱动分析。
3. **Synthesizer (合成器)**：自底向上递归地将子命题的软真值聚合回父命题。采用 **线性合成规则 (Linear Synthesis Rule)**，即 $p_{\text{true}} = \beta_0 + \sum \beta_i \cdot p_{\text{true}_i}$，以有效平均掉随机噪声。

### **相比现有方法的优势**
- **更高的准确性与更低的方差**：通过分解降低偏差，通过线性合成减少方差。
- **更强的鲁棒性与可扩展性**：支持大规模并行处理，计算时间随分析深度呈近线性增长。
- **支持交互式“假设分析”**：允许用户修改任意节点的真值，系统能快速重新合成（resynthesis），实现高效的场景模拟。
- **成本效益高**：特别是使用 Jupyter Notebook Grounder 时，在接近顶尖性能的同时大幅降低成本和耗时。

---

## 2. 核心实验方法和设置

### **使用的数据集**
实验在 **736 个真实世界的经济与金融预测挑战** 上进行，这些任务天然符合真/假命题预测的形式：
- **金融市场的“多头 vs. 空头”预测**：例如，“今年持有 $NVDA 股票是最佳策略吗？”
- **Polymarkets 等预测市场中的未来事件预测**：例如，“谁将赢得 2024 年美国总统大选？”

所有事件均经过筛选，确保其解决日期在模型知识截止日期（2024年6月1日）之后，以保证预测的真实性。

### **实验设置和评估指标**
- **基础模型**：统一使用 `o3-2025-04-16` 模型，温度设为 0.1 以减少随机性。
- **搜索工具**：使用 Exa.ai 提供的网络搜索。
- **评估指标**：
  - **Accuracy (准确率)**：是否为最优选项分配了最高的 $p_{\text{true}}$。
  - **Soft Score / Hard Score**：衡量决策的实际回报。
  - **Brier Score (BS)**：衡量预测分布的均方误差。
  - **Variance (方差)**：衡量预测的稳定性。
  - **API Cost 和 Wall-clock Time**：衡量效率。

### **基线方法对比**
- **独立基线 (Standalone Baselines)**：
  - `Basic Search`：仅依赖网络搜索。
  - `Deep Research`：OpenAI 的深度研究代理。
  - `Jupyter Notebook`：本文提出的高级 Grounder。
- **推理框架基线 (Reasoning Frameworks)**：
  - `Tree-of-Thoughts (ToT)`
  - `Graph-of-Thoughts (GoT)`
  - `Forest-of-Thoughts (FoT)`
- **Analytica 变体**：与不同 Grounder 结合，并测试三种合成规则：`Vanilla`, `Simple Logic`, `Linear`。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**
- **最高准确率**：`Analytica` + `Deep Research Grounder` + `Linear Synthesis` 达到 **71.06%** 的准确率。
- **最低方差**：该配置下的预测方差仅为 **6.02%**，表现出极高的稳定性。
- **平均提升**：相比多种基线，`Analytica` 平均提升了 **15.84%** 的准确率。

### **与基线方法的对比结果**
| 方法 | 准确率 (Accu.) | 方差 (Var.) | 成本 (Cost) | 时间 (Time) |
| :--- | :--- | :--- | :--- | :--- |
| **Deep Research** | 63.04% | 9.28% | $4.02 | 7.60m |
| **+ Analytica-L** | **71.06%** | **6.02%** | $14.10 | 30.01m |
| **Jupyter NB** | 61.96% | 12.28% | $0.07 | 2.61m |
| **+ Analytica-L** | **70.11%** | **7.28%** | $1.36 | 14.15m |

**结论**：
- `Analytica` 显著超越了所有基线方法，即使是最强大的 `Deep Research`。
- `Jupyter Notebook Grounder` 表现出惊人的**成本效益**：在成本降低 **90.35%**、时间减少 **52.85%** 的情况下，达到了接近 `Deep Research` 的性能（仅低 1.34%）。

### **消融实验结果**
- **合成规则的影响**：
  - `Linear` 规则表现最佳（71.06%），显著优于 `Vanilla` (69.16%) 和 `Simple Logic` (66.30%)。
  - `Simple Logic` 规则对噪声极为敏感，鲁棒性差。
- **Grounder 的影响**：
  - 更强大的 Grounder（如 `Deep Research` 或 `Jupyter Notebook`）能显著提升最终性能，是决定准确率的关键因素。
- **开放权重模型的适应性**：
  - `Analytica` 在 `OpenAI-OSS-20B` 等小型开源模型上也能带来超过 **15%** 的准确率提升，证明了其通用性和可及性。

---

## 4. 关键结论和发现

### **主要发现**
1. **SPR 是有效的**：将复杂分析转化为软真值命题的估计，为 LLM 推理提供了更稳健、可验证的框架。
2. **分解与合成相辅相成**：`Analyzer` 的分解降低了单个判断的难度（减小偏差），`Synthesizer` 的线性聚合平滑了随机噪声（减小方差）。
3. **线性合成规则至关重要**：其恒定的敏感性（constant sensitivity）和自然的平滑效应使其在噪声环境中远优于逻辑规则。
4. **高度可扩展**：递归调用和并行处理使得 `Analytica` 能够处理指数级增长的分析复杂度，而计算时间仅呈近线性增长。
5. **成本效益显著**：`Jupyter Notebook Grounder` 证明了自动化数据分析师的可行性，为高性能分析提供了低成本路径。

### **方法的局限性**
1. **独立性假设**：框架在子命题相互独立时效果最佳，但现实中命题间可能存在相关性。
2. **合成器可靠性**：合成器估算的系数 $\beta_i$ 若不准确，可能引入新的错误。
3. **同质化 Grounder**：目前对所有叶节点使用相同的 Grounder，未能根据命题特性动态选择最合适的专家。

### **未来工作方向**
- **动态 Grounder 路由 (Dynamic Grounder Routing)**：借鉴 `model routing` 技术，为不同类型的命题自动分配最匹配的 Grounder。
- **建模命题间相关性**：开发能够显式处理子命题间协方差的合成机制。
- **提升合成器的鲁棒性**：探索更先进的技术（如 PGMs）来生成更可靠的合成系数。
- **应用领域拓展**：将 `Analytica` 应用于更多高风险领域，如政策制定、科学研究和机器人决策。

</details>

---

### 9. [Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](https://arxiv.org/abs/2604.23623)

**Authors**: Zichuan Fu, Xian Wu, Guojing Li, Yejing Wang, Yijun Chen, Zihao Zhao, Yixuan Luo, Hanyu Yan, Yefeng Zheng, Xiangyu Zhao  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.23623v1  

#### Abstract
Recent advancements in large language models (LLMs) have catalyzed the rise of reasoning-intensive inference paradigms, where models perform explicit step-by-step reasoning before generating final answers. While such approaches improve answer quality and interpretability, they incur substantial comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
当前的 **Large Language Models (LLMs)** 在执行复杂推理任务时（如数学解题、代码生成）通常采用“thinking paradigm”——即先进行长链的内部推理（reasoning chain），再输出最终答案。这种模式虽然提升了准确性和可解释性，但带来了显著的 **计算开销**（inference latency 和 token 成本高），尤其在实时应用或预算受限场景下难以部署。

此外，现有优化方法（如 Reinforcement Fine-Tuning, RFT）存在以下局限：
- 需要对 LLM 进行训练，可能损害其通用能力；
- 不适用于仅提供 API 接口的闭源模型。

因此，本文提出一个无需修改 LLM 的高效协作框架。

---

### 🚀 提出的新方法与核心思路
作者提出了 **Tandem**，一种基于 **mentor-intern 架构** 的 LLM-SLM 协作推理框架：

#### 核心思想：
- **LLM 作为导师（Mentor）**：不负责完整推理，而是生成轻量级的 **Thinking Insights**（思考洞察），包括四个模块化组件：
  1. **Goal**：明确目标与约束
  2. **Planning**：高层策略规划
  3. **Retrieval**：相关知识召回
  4. **Action**：关键逻辑步骤
- **SLM 作为实习生（Intern）**：接收这些结构化指导后，完成详细的推理过程并生成最终答案。

#### 创新机制：**Cost-aware Termination**
引入一个基于 **SLM 输出不确定性**（perplexity 和 entropy）的分类器（classifier），动态判断当前 LLM 提供的指导是否已足够让 SLM 正确作答。若足够，则提前终止 LLM 的进一步推理，实现自适应控制。

> 🔍 这种“分阶段+早停”的设计实现了 **按需分配计算资源**，简单问题少花成本，难题才调用深层推理。

---

### ⚖️ 相比现有方法的优势
| 维度 | Tandem 的优势 |
|------|----------------|
| **效率** | 显著降低计算成本（约减少 40% TFLOPs） |
| **兼容性** | 无需训练 LLM，支持 API 调用的黑盒模型（如 GPT-4o-mini） |
| **泛化性** | Sufficiency Classifier 可跨领域迁移（从 MATH 到 HumanEval） |
| **性能** | 不仅更省，而且更准：超越单独 LLM 的准确率 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **MATH** (Hendrycks et al., 2021)  
  - 包含 12.5K 数学竞赛题，涵盖 7 个子领域（代数、几何等），难度分为 5 级。
  - 测试集共 5K 样本。
- **GSM8K** (Cobbe et al., 2021)  
  - 小学级别数学应用题，强调多步推理，共 8.5K 样本。
- **HumanEval** (Chen et al., 2021)  
  - 用于验证跨域泛化能力的代码生成基准，包含 164 个编程问题。

---

### 🧪 实验设置
#### 模型组合
- **LLM 候选**：DeepSeek-R1-Distill-Qwen-32B（简称 32B）、Qwen3-32B、GPT-4o-mini（API）
- **SLM 候选**：DeepSeek-7B、Qwen3-8B
- 所有模型均在 **deterministic 模式** 下运行（temperature=0）

#### 思考努力层级（Effort Levels）
将 LLM 的推理分为三个阶段，对应不同长度预算：
- **Low**: 100 tokens
- **Medium**: 500 tokens
- **High**: 1,000 tokens

#### Sufficiency Classifier 设计
- 输入特征：来自 SLM 处理输入时的 token-level **perplexity** 和 **entropy** 序列统计量（均值、标准差、趋势等）
- 模型结构：两层 MLP（64→32 units），ReLU + Dropout
- 训练方式：在 MATH 训练集上训练，标签为“该指导下 SLM 是否能正确作答”

---

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy** | 正确解答的比例（标准评测协议） |
| **Inference Length** | 总生成 token 数（LLM + SLM） |
| **Computational Cost** | 近似为 `TFLOPs = 2 × (|θ_L| × L_L + |θ_S| × (L_L + L_S))` |

---

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **Single LLM (32B)** | 单独使用大模型进行完整推理 |
| **Single SLM (7B)** | 单独使用小模型 |
| **Fixed-budget Collaboration** | 固定使用某一级别的 LLM 指导（low/medium/high） |
| **Budget Forcing** | 截断 LLM 的推理长度以节省成本 |
| **LLM Cascade** | 一次性决定走 SLM 或 Full LLM 路径 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & Table 6）

| 方法 | MATH 准确率 (%) | 计算成本 (TFLOPs) | 相比 32B 的成本降幅 |
|------|------------------|--------------------|---------------------|
| SLM-only (7B) | 77.14 | 38.25 | — |
| LLM-only (32B) | 80.90 | 168.35 | — |
| **Tandem (7B+32B)** | **83.46** | **99.72** | **~40.8% ↓** |
| Budget Forcing | 82.18 | 108.74 | ~35.4% ↓ |
| LLM Cascade | 82.60 | 95.33 | ~43.4% ↓ |

> ✅ **Tandem 在提升准确率的同时，实现了接近最优的成本压缩效果。**

---

### 🔍 与其他方法对比亮点
- **相比 Budget Forcing**：
  - 准确率更高（+1.28%）
  - 成本更低（-8.3%）
  - 原因：Tandem 传递的是**结构化洞察**而非冗长推理链
- **相比 LLM Cascade**：
  - 准确率高出 +0.86%
  - 原因：Tandem 是**逐阶段判断**，而 Cascade 是一次性路由决策，灵活性不足

---

### 🧩 消融实验与关键发现

#### （1）跨家族模型协作（Cross-Family Generalization）
| 组合 | MATH Acc (%) | Cost (TFLOPs) |
|------|---------------|----------------|
| DeepSeek-7B + Qwen3-32B | 79.96 | 58.06 |
| DeepSeek-7B + DeepSeek-32B | 83.34 | 97.95 |

✅ 表明结构化 Thinking Insights 具备良好的**跨模型族可理解性**。

#### （2）模型大小匹配的影响（Model Size Gap）
- 当 SLM 过小时（如 1.5B），即使有 32B 指导也难以有效利用 → 改进有限
- 最佳协作发生在 **中等能力差距** 之间（如 7B + 32B）
- 结论：需要合理的 **capability alignment**

#### （3）非思考模式下的有效性（Non-thinking Mode）
即使 LLM 不启用 thinking mode，Tandem 仍能通过常规输出提取有效信息进行指导：
- 平均准确率达 **82.56%**
- 成本比 32B 单独运行低 **36.7%**

#### （4）API 可访问模型的应用
使用 **GPT-4o-mini** 和 **gpt-oss-120b** 作为远程 LLM：
- 成功集成且性能提升明显
- 成本反而下降 → 因 SLM 更快解决问题，减少了昂贵 LLM 的调用时间

#### （5）跨领域迁移能力
- Sufficiency Classifier 在 **未重新训练** 的情况下应用于 **HumanEval**（代码生成）：
  - 准确率达 **85.37%**，优于最强固定预算基线（83.54%）
- 表明 **PPL/entropy 特征具有领域无关性**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **轻量级结构化指导 > 完整推理链**
   - LLM 提供的 **Goal, Planning, Retrieval, Action** 四类 Thinking Insights 足以引导 SLM 完成高质量推理。
2. **动态早停机制显著提效**
   - 基于 SLM 自身置信度的 sufficiency classifier 可有效识别“何时停止 LLM 指导”，实现资源按需分配。
3. **协同效应存在**
   - LLM + SLM 的组合表现 **超过任一模型单独运行**，体现互补优势。
4. **高度实用与可部署**
   - 支持 API 模型、无需微调、classifier 可跨域复用，适合工业落地。

---

### ⚠️ 局限性（Limitations）
1. **领域泛化待验证**
   - 当前实验集中于数学与代码，尚未验证在常识推理、开放问答等任务上的表现。
2. **依赖标注数据训练 classifier**
   - 虽然可跨域迁移，但仍需至少在一个领域上有带标签数据进行训练。
3. **固定双模型架构**
   - 当前仅为一对一协作，未探索多模型协同、角色动态切换等更复杂的协作模式。

---

### 🔮 未来工作方向
- 探索 **zero-shot sufficiency detection**，减少监督需求
- 引入 **multi-agent collaboration**，允许多个 LLM/SLM 动态分工
- 扩展至 **vision-language、语音交互** 等多模态场景
- 研究 **feedback loop** 机制，使 SLM 可反向请求补充指导

---

## 🔚 总结
**Tandem** 是一项极具工程价值的研究，它巧妙地将认知科学中的“模块化思维”理念融入 LLM 推理系统，构建了一个高效、灵活、可扩展的 **LLM-SLM 协同推理范式**。其实验充分、分析深入，在保持甚至提升性能的前提下，大幅降低了推理成本，并展现出强大的泛化能力和实用性，为大规模语言模型的实际部署提供了新的解决方案。

</details>

---

### 10. [Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling](https://arxiv.org/abs/2604.24715)

**Authors**: Parsa Ashrafi Fashi, Utkarsh Saxena, Mehdi Rezagholizadeh, Aref Jafari, Akash Haridas, Mingyu Yang, Vansh Bhatia, Guihong Li, Vikram Appia, Emad Barsoum  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.24715v1  

#### Abstract
Hybrid sequence models that combine efficient Transformer components with linear sequence modeling blocks are a promising alternative to pure Transformers, but most are still pretrained from scratch and therefore fail to reuse existing Transformer checkpoints. We study upcycling as a practical path ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **Transformer-based LLMs** 在处理长上下文时面临 **二次方计算复杂度** 和 **KV-cache内存爆炸** 的瓶颈。虽然已有研究提出将纯Transformer模型替换为结合 **SSM（如Mamba）或线性注意力模块** 的混合架构（Hybrid Models），以提升效率，但这些方法大多需要从头预训练（pretrain from scratch），无法有效复用已有的大规模预训练模型的知识。

此外，现有的 **模型升级回收（upcycling）** 方法（如MambaInLlama、Zebra-Llama）虽能将预训练的Transformer转换为混合架构，但其优化目标主要集中在保持短上下文性能（short-context quality），而忽略了对现代LLM至关重要的 **长上下文能力（long-context capability）**。

### 提出的新方法：HyLo
本文提出了 **HyLo (HYbrid LOng-context)**，一种全新的、面向长上下文感知的模型升级回收方案，旨在将预训练的Transformer LLM高效地转换为兼具高性能和高效率的混合模型，同时**兼顾短上下文质量和显著增强长上下文能力**。

#### 核心创新点
1.  **长上下文感知的升级回收策略 (Long-context-aware upcycling)**：
    *   首次将“长上下文能力”作为upcycling的核心优化目标，而非仅关注短上下文性能。
    *   提出了一种结合 **Multi-Head Latent Attention (MLA)** 和线性块（**Mamba2** 或 **Gated DeltaNet (GDN)**）的混合架构设计，其中MLA提供强大的注意力能力，而Mamba2/GDN则无KV-cache开销，实现高效的长序列建模。

2.  **渐进式长上下文训练 (Extended long-context training regime)**：
    *   采用分阶段（staged）的训练方式，将训练上下文长度从传统的2K-8K扩展到**高达64K tokens**。
    *   这种直接在长序列上进行微调的策略，被证明是提升模型长上下文泛化能力的关键。

3.  **教师引导的长上下文蒸馏 (Teacher-guided long-context distillation)**：
    *   引入基于 **chunk-wise KL散度** 的监督信号，在长上下文训练中稳定优化过程。
    *   通过一个强大的预训练Transformer教师模型（如8B Llama）来指导学生模型（HyLo）的学习，确保知识的有效迁移，尤其是在长距离依赖的捕捉上。

4.  **高吞吐量推理服务集成 (High throughput inference serving)**：
    *   将HyLo模型成功集成到 **vLLM** 推理框架中。
    *   实现了高达 **2M tokens** 的prefill和解码能力，相比基线Llama-3.2-3B（64K上限）实现了**30倍以上的上下文扩展**，并减少了**超过90%的KV-cache内存占用**。

### 相比现有方法的优势
- **效率极高**：无需从头预训练，大幅降低计算成本。
- **能力更强**：在保持优秀短上下文性能的同时，长上下文能力远超现有upcycling基线。
- **部署友好**：显著降低KV-cache内存，支持超长上下文的实际部署。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **短上下文评估**：使用 `lm-eval-harness` 中的标准基准，包括：
    - **Common Sense Reasoning**: ARC-Challenge, ARC-Easy, HellaSwag, OpenBookQA, PIQA, RACE, WinoGrande.
    - **数学推理**: GSM8K.
- **长上下文评估**：使用 **RULER** 基准测试套件，专门用于评估模型在不同长度（8K, 16K, 32K, 64K）下的长上下文理解与推理能力。
- **训练数据**：在多个配置下使用了总计 **10B tokens** 的数据进行微调。

### 实验设置和评估指标
- **主干模型 (Backbone Models)**：基于三种预训练模型进行upcycling：
    - Llama-3.2-1B
    - Llama-3.2-3B
    - Qwen3-1.7B
- **评估指标**：
    - **短上下文**：各任务的准确率（Accuracy）和平均得分。
    - **长上下文**：RULER基准在不同上下文长度（8K, 16K, 32K, 64K）下的平均准确率。
    - **数学能力**：GSM8K上的准确率。
    - **推理效率**：TTFT (Time to First Token) 和 TPOT (Time Per Output Token)。

### 基线方法对比
- **Upcycling Baselines**:
    - MambaInLlama
    - Llamba
    - Zebra-Llama
    - M1
    - HypeNet
- **其他对比模型**:
    - Jet-Nemotron-2B (从头预训练，使用400B tokens)

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **长上下文性能显著领先**：
    - 在 **RULER** 基准上，所有HyLo变体均**显著优于**所有基线模型。
    - 例如，在 **HyLo-Qwen-1.7B** 上，尽管只用了 **10B tokens** 进行微调，但在 **GSM8K** 和 **RULER-64K** 上的表现**显著超越**了使用 **400B tokens** 从头训练的 **JetNemotron-2B**。
    - 在 **Llama-3.2-3B** 基础上构建的 **HyLo-Llama-6MLA22M2**，在64K上下文长度下，RULER平均准确率远高于Zebra-Llama等基线。

2.  **短上下文性能保持竞争力**：
    - 尽管进行了长上下文训练，HyLo模型在短上下文任务（如ARC, HellaSwag等）上的性能与基线模型相当，甚至略有优势，证明了其知识保留能力。

3.  **KV-Cache内存大幅减少**：
    - HyLo模型通过减少MLA层的数量和利用无缓存的Mamba2/GDN层，实现了**超过90%的KV-cache内存节省**。
    - 这使得在8张AMD MI300X GPU上，能够支持**高达2M tokens**的上下文处理，而同等规模的纯Transformer模型（如Llama-3.2-3B）在64K上下文时就会因内存不足（OOM）而失败。

### 消融实验结果
1.  **训练上下文长度的影响**：
    - 直接在 **64K** 上训练的模型，其长上下文性能**远优于**在8K上训练后通过 **YaRN** 位置插值扩展的模型，证明了长上下文训练的必要性。

2.  **知识蒸馏 (KD) 的影响**：
    - 使用更大的教师模型（如8B）进行蒸馏，能带来更显著的性能提升，尤其是在长上下文任务上（RULER-64K提升达22%）。
    - 蒸馏对数学推理能力（GSM8K）也有巨大帮助（提升可达6.3%）。

3.  **增强型中间层蒸馏 (Enhanced-ILD)**：
    - 本文提出的在token-mixer输出上也进行对齐的Enhanced-ILD损失，相比原始ILD，能持续提升短上下文和数学推理性能。

4.  **架构设计选择**：
    - 实验表明，一些在预训练中有效的技术（如NoPE, Gated Attention），在upcycling场景下效果不佳，说明upcycling有其独特的优化挑战。

---

## 4. 关键结论和发现

### 主要发现
1.  **长上下文能力可以被“升级回收”**：通过精心设计的训练策略（长上下文训练+教师蒸馏），完全可以将预训练Transformer的知识迁移到混合架构中，并**主动增强其长上下文能力**，而不仅仅是被动保留。
2.  **直接长上下文训练至关重要**：简单的零样本扩展（如YaRN）效果有限，**在长序列上进行实际的微调是获得强大长上下文泛化能力的关键**。
3.  **教师蒸馏是稳定器和加速器**：强大的教师模型不仅能稳定长上下文的优化过程，还能显著提升学生的最终性能，尤其是在复杂的推理任务上。
4.  **HyLo是高效且实用的**：该方法避免了昂贵的预训练，同时产出的模型在效率（内存、延迟）和能力（长短上下文）上都极具竞争力，非常适合实际部署。

### 方法的局限性
- **仍存在性能差距**：尽管在长上下文上表现优异，但在某些极端情况下，可能仍未完全达到从头开始为长上下文优化的纯混合模型的极限。
- **依赖高质量教师**：方法的效果高度依赖于教师模型的质量和规模。
- **架构搜索空间**：MLA与线性块的最佳组合比例和层间布局仍需进一步探索。

### 未来工作方向
- **缩小长上下文性能差距**：进一步优化训练方法，以完全闭合与理想长上下文模型之间的性能差距。
- **提高蒸馏效率**：探索更轻量级或更高效的蒸馏机制。
- **扩展应用场景**：将此框架应用于更多下游任务，特别是那些对鲁棒长上下文推理至关重要的领域，如法律文档分析、长代码生成等。

</details>

---

### 11. [GreenDyGNN: Runtime-Adaptive Energy-Efficient Communication for Distributed GNN Training](https://arxiv.org/abs/2604.23139)

**Authors**: Arefin Niam, Tevfik Kosar, M. S. Q. Zulkar Nine  
**Category**: cs.DC  
**Published**: 2026-04-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.23139v1  

#### Abstract
Distributed GNN training is dominated by remote feature fetching, which can be very costly. Multi-hop neighborhood sampling crosses partition boundaries and triggers fine-grained RPCs whose fixed initiation cost and GPU-stall latency waste energy. Prior systems try to reduce this overhead with presa...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**GreenDyGNN: Runtime-Adaptive Energy-Efficient Communication for Distributed GNN Training**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
分布式图神经网络（Distributed GNN）训练中的主要瓶颈是跨分区的**远程特征获取（remote feature fetching）**。由于多跳邻居采样常跨越图分区边界，导致频繁的小规模 RPC 调用，带来高昂的固定启动开销（initiation overhead）和 GPU 等待能耗（GPU stall energy）。  
现有系统如 RapidGNN 采用静态缓存策略（epoch-level static caching），无法应对运行时网络拥塞的变化，导致在动态环境下能效下降。

> **核心问题**：静态缓存策略无法适应时间变化的网络拥塞，造成能量浪费高达 45%。

---

### 🚀 提出的新方法与创新思路

GreenDyGNN 提出了一种**运行时自适应的缓存控制机制**，将缓存窗口管理建模为一个**序列决策问题（sequential decision problem）**，并引入强化学习进行动态优化。

#### 主要创新点：
1. **细粒度窗口化缓存重建（Fine-grained window-based cache rebuilds）**  
   不再局限于每轮 epoch 重建一次缓存，而是在每个训练窗口（window）边界动态决定是否重建，并调整窗口大小 $W$ 和各分区的缓存分配。

2. **基于强化学习的自适应控制器（Double-DQN Agent）**  
   设计了一个轻量级的 Double-DQN 智能体，在校准后的模拟器中通过 **domain-randomized congestion** 进行训练，实现从仿真到真实集群的迁移（sim-to-real transfer）。

3. **异步双缓冲预取流水线（Asynchronous double-buffered pipeline）**  
   缓存重建完全脱离主训练流程，避免对 GPU 执行造成阻塞，使得自适应决策“零开销”。

4. **端到端能效优化目标**  
   明确以**总系统能量消耗（GPU + CPU）** 为核心优化目标，而非仅关注吞吐或延迟。

---

### 🔍 相比现有方法的优势

| 方面 | 现有方法（如 RapidGNN） | GreenDyGNN |
|------|--------------------------|-----------|
| 缓存策略 | 静态、每 epoch 重建一次 | 动态、窗口级在线调整 |
| 网络感知 | 无，依赖离线配置 | 实时感知拥塞状态 $\theta_o$ |
| 决策方式 | 固定规则或启发式 | 强化学习驱动的联合决策（$W$, 分配权重） |
| 适应能力 | 无法响应动态拥塞 | 可跟踪非平稳网络变化 |
| 能效表现 | 在稳定网络下有效 | 在拥塞下显著优于所有静态策略 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **OGBN-Products**（2.4M 节点，61.9M 边）
- **Reddit**（233K 节点，114M 边）
- **OGBN-Papers100M**（111M 节点，1.6B 边）

所有图使用 METIS 分区，划分为 4 个部分部署在 4 个节点上。

---

### ⚙️ 实验平台与设置
- **硬件环境**：4 节点 Chameleon Cloud 集群，每节点含 Intel Xeon CPU + 2×NVIDIA P100 GPU，25 Gbps 以太网。
- **模型架构**：2-layer GraphSAGE，fan-out={10,25}，30 epochs。
- **批大小**：$B \in \{1000, 2000, 3000\}$，默认使用 2000。
- **通信层**：基于 DGL 的 DistTensor RPC。

---

### 🧪 拥塞注入策略
- 使用 Linux `tc netem` 在 DGL RPC 端口（30050）注入延迟。
- 时间变化模式：周期性切换干净/拥塞阶段（每 7 个 epoch 循环）。
- 拥塞强度：单链路或多链路增加 15–25ms 单向延迟。
- 控制变量：梯度同步流量不受影响（走独立端口）。

---

### 📊 评估指标
| 指标类别 | 具体内容 |
|--------|--------|
| **能效指标** | 总能量（GPU + CPU，单位 kJ）、CPU/GPU 分项能耗 |
| **性能指标** | 每 epoch 时间（ET）、累计 wall-clock time |
| **缓存行为** | 缓存命中率（hit rate）、重建频率、fetch latency |
| **模型质量** | 准确率（accuracy），验证收敛一致性 |

---

### 🆚 基线方法对比
| 基线 | 描述 |
|-----|------|
| **Default DGL** | Vanilla 分布式训练，按需拉取特征（no cache） |
| **BGL** | 多层缓存 + I/O 优化，支持预取但无自适应 |
| **RapidGNN** | 当前最优静态缓存方案，epoch-level rebuild，缓存容量 100,000 节点 |
| **GreenDyGNN (ours)** | 本文提出的方法，支持 runtime-adaptive rebuild + allocation |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（B=2000，拥塞条件下）

| 方法 | OGBN-Products (kJ) | Reddit (kJ) | OGBN-Papers100M (kJ) |
|------|--------------------|------------|-----------------------|
| Default DGL | 284.3 | 326.8 | 452.2 |
| BGL | 218.8 | 278.0 | 342.7 |
| RapidGNN | 213.0 | 239.0 | 380.4 |
| **GreenDyGNN (Ours)** | **203.9** | **189.4** | **307.2** |

> ✅ **最高节能达 43%**（vs. Default DGL on Reddit）

---

### 🔁 与最强静态基线（RapidGNN）对比
- **节能提升**：**4–24%** 的总能量降低
- **最大收益场景**：在 **OGBN-Papers100M** 上减少 **73.2 kJ** 累积能耗（至第30轮）
- **CPU 能耗显著下降**：因减少了昂贵的远程 fetch 操作（RPC initiation cost 下降）
- **GPU 能耗接近持平**：两者均通过缓存降低了 idle power 浪费

---

### 📉 拥塞带来的额外能耗（相对自身干净环境的增长）

| 方法 | Products | Reddit | Papers100M |
|------|---------|--------|-----------|
| Default DGL | 30% | 45% | 50% |
| RapidGNN | 18% | 31% | 45% |
| **GreenDyGNN** | **15%** | **5%** | **19%** |

> 💡 GreenDyGNN 成功吸收了多达 **26个百分点** 的拥塞开销（如 Reddit 和 Papers100M），远超静态方法。

---

### 📊 收敛速度与准确率
- **准确率一致**：所有方法间差异在 1–3 个百分点内（属于运行波动范围），说明 GreenDyGNN 不影响模型收敛。
- **更快达到目标精度**：得益于更短的 epoch 时间，GreenDyGNN 在 wall-clock time 上率先收敛（见 Fig. 10）。
- **平均 epoch 时间缩短最多达 43%**（如 Reddit, B=2000）

---

### 🔬 消融实验（Ablation Study, B=2000）

| 变体 | Products | Reddit | Papers100M |
|------|--------|--------|------------|
| w/o RL ($W=16$ fixed) | 218.9 | 204.2 | 336.1 |
| w/o Cost Weights (uniform alloc.) | 210.0 | 195.8 | 318.0 |
| **Full GreenDyGNN** | **203.9** | **189.4** | **307.2** |

#### 结论：
- **RL 控制器贡献最大**：关闭后能耗上升 6.9%–8.6%，表明动态调整 $W$ 是关键。
- **Per-owner cost weighting 提供补充增益**：额外节省约 2.9%–3.4%，尤其在不对称拥塞下效果明显。
- 二者协同作用，共同实现最优能效。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **运行时网络变化是影响分布式 GNN 能效的一等公民问题**  
   即使轻微拥塞（如 4ms 延迟）也会使静态缓存策略偏离最优操作点，导致能耗激增。

2. **缓存窗口大小应作为运行时控制变量，而非超参数**  
   将 $W$ 视为可调控制变量，允许系统根据当前网络状态动态平衡“重建开销”与“miss 开销”。

3. **强化学习可在低频决策场景下高效工作**  
   尽管每几十步才做一次决策，但其长期累积效应显著，且通过 sim-to-real + domain randomization 实现良好泛化。

4. **GreenDyGNN 在拥塞下大幅领先，在干净环境下不退化**  
   - 拥塞下：节能 **4–24%** vs. RapidGNN，最高 **43%** vs. Default DGL
   - 干净下：性能与最佳静态策略相差 <2%，证明无过度调节代价

5. **异步双缓冲设计实现了真正的“零开销”自适应**  
   决策推理时间远小于单次 RPC 延迟，不影响训练流水线。

---

### ⚠️ 局限性
- **目前假设同构网络与计算资源**：未考虑混合带宽链路（如 RDMA + Ethernet）或异构 GPU。
- **动作空间有限**：仅支持离散窗口大小和简单缓存偏置策略，尚未探索连续控制或多维动作。
- **依赖模拟器校准**：虽然 sim-to-real 表现良好，但仍需一次性实测数据来构建成本模型。
- **未整合 GPU frequency scaling**：仍有进一步节能潜力未挖掘。

---

### 🔮 未来工作方向
1. 扩展至 **异构硬件环境**（mixed interconnects, RDMA support）
2. 将 **GPU frequency / power limit** 加入动作空间，联合优化通信与计算能耗
3. 引入 **在线微调机制**，利用生产 telemetry 动态更新模拟器参数
4. 探索 **更复杂的拥塞预测模型**（如结合历史趋势与时序建模）
5. 应用于更大规模工业图训练系统（如推荐系统、知识图谱）

---

> 📌 **一句话总结**：  
> **GreenDyGNN 首次将运行时网络感知与强化学习相结合，实现了对分布式 GNN 训练中通信能耗的动态最优控制，在真实拥塞场景下相较最强静态缓存策略节能达 24%，同时保持训练效率与模型准确性。**

</details>

---

### 12. [A Differentiable Framework for Global Circulation Model Precipitation Bias Correction](https://arxiv.org/abs/2604.23045)

**Authors**: Kamlesh Sawadekar, Seth McGinnis, Peijun Li, Chaopeng Shen  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.23045v1  

#### Abstract
Systematic biases in Global Circulation Model (GCM) outputs limit their direct applicability in regional planning, necessitating bias correction. Correcting precipitation is particularly challenging due to its non-Gaussian distribution, intermittent nature, and non-linear extremes. However, traditio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Differentiable Framework for Global Circulation Model Precipitation Bias Correction

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Global Circulation Models (GCMs) 在区域气候影响评估中存在**系统性偏差**（systematic biases），尤其是在降水（precipitation）模拟方面。由于降水具有**非高斯分布、间歇性、非线性极端事件**等特性，传统统计方法难以有效校正其偏差。此外，现有机器学习方法多为“黑箱”，缺乏可解释性和跨模型/跨区域的泛化能力。

### 提出的新方法
本文提出了一种**可微分的降水偏差校正框架**——**δCLIMBA**（或 dCLIMBA），其核心思想是：
- 构建一个**参数化的单调变换函数**，通过优化 GCM 输出与参考再分析数据（如 Livneh）之间的分位数匹配来学习该变换。
- 利用**可微分编程**（differentiable programming）将物理启发的结构嵌入模型，实现端到端优化，并保持对物理规律的遵从。

### 相比现有方法的优势
| 特性 | δCLIMBA | 传统方法（如 QM, QDM） | LOCA2 |
|------|--------|------------------------|-------|
| **可解释性** | ✅ 高（参数化变换可分析） | ❌ 低（黑箱或静态映射） | ❌ 低（基于类比） |
| **空间一致性** | ✅ 显式建模（spatial attention） | ❌ 点对点独立处理 | ✅ 强（类比机制） |
| **趋势保留** | ⚠️ 部分保留（无显式机制但表现良好） | ❌ 差（尤其 QM/QDM 衰减趋势） | ❌ 差（衰减明显） |
| **跨区域泛化** | ✅ 可推广至未见区域（利用地形等静态属性） | ❌ 不可泛化（需本地标定） | ❌ 不可泛化（定制产品） |
| **极端值校正** | ✅ 强（加权分位数损失强调尾部） | ⚠️ 易过度放大尾部 | ✅ 较好 |

---

## 2. 核心实验方法和设置

### 数据集
- **GCM 数据**：来自 **CMIP6** 的6个模型（ACCESS-CM2, GFDL-ESM4, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-LR, MRI-ESM2-0）
  - 时间范围：历史期（1979–2000训练，1965–1978验证），测试期（2001–2014），未来情景（SSP5-8.5, 2015–2099）
  - 区域：**CONUS**（美国本土）
- **参考数据**：
  - **Livneh 再分析数据集**（作为观测基准）
  - **LOCA2** 下尺度产品（用于对比，上采样至 GCM 分辨率）
- **辅助输入特征**：
  - 地形：**elevation**, **slope**, **aspect**（来自 SRTMGL1）
  - **land cover**（来自 NALCMS）

### 实验设置
- **训练周期**：100 epochs，使用4块 A100 GPU 并行训练约7小时
- **推理速度**：仅需2分钟即可生成整个 CONUS 区域的偏差校正结果
- **时空编码器设计**：
  - **NN1（时间编码器）**：采用 CNN-1d（2层，每层64神经元）
  - **NN2（空间编码器）**：基于 **Transformer** 的自注意力机制，考虑地理距离、方位角等相对位置信息
- **输出变换**：使用 `softplus` 基函数构造**单调递增映射**，确保物理合理性

### 评估指标
#### （1）边际偏差（Marginal Bias）
- 使用 **ETCCDI 指数** 进行评估：
  - **强度类**：Rx1day, Rx5day
  - **频率类**：R10mm, R20mm
  - **持续性类**：CDD, CWD
  - **极端总量类**：R95pTOT, R99pTOT
- 评价方式：计算各格点的**平均百分比偏差**（mean percentage bias）

#### （2）空间结构
- 使用 **Fractal Dimension (FD)** 衡量降水场的空间聚集性和多尺度结构
- 方法：box-counting 法应用于不同分位数阈值下的二值化降水场
- 指标：**MAE**（相对于 Livneh 的 FD 曲线）

#### （3）趋势保留
- 定义 **Trend Bias (TB)**：
  $$
  TB = 100 \cdot \frac{T_{\text{debiased}} - T_{\text{raw}}}{T_{\text{raw}}}
  $$
  其中 $T$ 是未来与历史时期的统计量变化（如均值、95th分位数）
- 评估是否扭曲原始 GCM 的气候变化信号

#### （4）空间泛化能力
- 设计“数据稀缺”场景：在 **Upper Mississippi** 区域训练，在相邻的 **Ohio** 区域测试（完全未参与训练）
- 验证模型能否利用地形等先验信息进行外推

### 基线方法对比
| 方法 | 类型 | 是否趋势保留 | 是否空间一致 |
|------|------|---------------|----------------|
| **Quantile Mapping (QM)** | 统计分位映射 | ❌ | ❌ |
| **ISIMIP3BASD** | 参数化趋势保留 | ✅ | ⚠️ |
| **ECDFM** | 非参数分位差 | ✅ | ❌ |
| **QDM** | 分位增量映射 | ✅ | ❌ |
| **LOCA2 (up-scaled)** | 类比下尺度 | ❌ | ✅ |

所有方法除 LOCA2 外均使用 `ibicus` 框架统一实现。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 指标 | δCLIMBA 表现 | 对比说明 |
|------|--------------|----------|
| **Fractal Dimension MAE** | **0.021** | 优于原始 GCM (0.040)，接近 LOCA2 (0.017) |
| **R99pTOT 偏差** | 中位数接近零，分布窄 | 显著优于 QM（严重高估） |
| **趋势保留（GFDL-ESM4）** | TB 接近零（mean 和 95th） | 优于 QDM 和 LOCA2，略逊于 ECDFM/ISIMIP |
| **空间泛化（Ohio 区域）** | R99pTOT/Rx5day 正偏差显著降低 | 证明具备一定外推能力 |

### 与基线方法的对比结果

#### （1）分位数分布校正（Figure 3）
- δCLIMBA 成功将所有6个 GCM 的日降水 CDF 曲线对齐至 Livneh，且**不引发非物理的极端放大**。
- QM、ECDFM、QDM 常出现**尾部过拟合**（尤其在 Phoenix、Orlando）。
- LOCA2 表现较好，但在某些城市（如 Yosemite）仍存在残余偏差。

#### （2）ETCCDI 指数偏差（Figure 4 & 5）
- δCLIMBA 在多数指数上表现出**更小的偏差离散度和更接近零的中位数**。
- 特别是在 **R99pTOT** 上，δCLIMBA 显著优于 QM（后者在全国范围内高估严重）。
- 在东南部湿润地区（如 Florida），δCLIMBA 更好地抑制了 GCM 的湿偏差。

#### （3）空间结构（Figure 6）
- δCLIMBA 的 FD MAE = **0.021**，仅次于 LOCA2 (0.017)，远优于纯分位映射方法（通常破坏空间结构）。
- 表明其**空间注意力机制有效捕捉了邻域间的物理依赖关系**。

#### （4）趋势保留（Figure 7）
- 尽管没有显式趋势保留机制，δCLIMBA 在 GFDL-ESM4 上对 mean 和 95th percentile 的趋势保留优于 QDM 和 LOCA2。
- 但在 wet days 数量上的趋势保留不如 ISIMIP 和 ECDFM。
- **注意**：此行为不具备跨 GCM 一致性（在其他模型上未复现），表明为**涌现性质而非结构性优势**。

#### （5）空间泛化（Figure 8）
- 在未见的 Ohio 区域，δCLIMBA 仍能有效减少 R99pTOT 和 Rx5day 的正偏差。
- 但 SDII 和 R20mm 等强度敏感指数仍有残余高估，且部分格点偏差反而增大（IQR 更宽）。
- 表明当前框架在**跨区域迁移时稳定性有限**，但仍优于无法外推的传统方法。

---

## 4. 关键结论和发现

### 主要发现
1. **δCLIMBA 能有效校正 GCM 降水的系统性偏差**，特别是在极端事件（tail behavior）和空间结构方面表现优异。
2. 通过引入**可微分的单调变换 + 空间注意力机制**，实现了**物理合理、可解释、高效**的偏差校正。
3. 模型展现出一定的**时间与空间泛化能力**，可在无本地观测的区域进行外推，这是传统方法无法做到的。
4. 在趋势保留方面虽无显式机制，但在特定 GCM 上表现优于主流方法，提示可通过进一步设计提升。

### 局限性
1. **趋势保留不具备普适性**：目前仅为特定 GCM 的涌现现象，尚未成为稳定特性。
2. **强度指标（SDII, R20）校正效果有限**：可能存在对湿日强度分布的过度拉伸。
3. **空间注意力范围受限**：当前 patch-based 注意力无法建模长程依赖（如大尺度天气系统）。
4. **未解决多变量一致性问题**：仅针对降水单变量校正，未考虑与其他变量（温度、风速等）的协变关系。

### 未来工作方向
1. **构建通用偏差校正基础模型**（foundation model）：实现 one-shot 或 few-shot bias correction，适应任意 GCM。
2. **增强空间建模能力**：引入图神经网络（GNN）或全局注意力机制以捕捉长程空间依赖。
3. **集成显式趋势保留机制**：结合 QDM 或 ISIMIP 思路，在损失函数中加入趋势约束项。
4. **探索偏差成因解释性**：利用可微分特性反演神经网络参数，分析不同地理/气候条件下偏差来源。
5. **扩展至多变量联合校正**：发展 multivariate differentiable bias adjustment 框架。

--- 

> ✅ **总结一句话**：  
> δCLIMBA 提供了一个**模块化、高效、可解释且具有一定泛化能力**的 GCM 降水偏差校正新范式，推动了 bias correction 从“经验统计”向“物理-数据融合”的范式转变。

</details>

---

### 13. [PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks](https://arxiv.org/abs/2604.23148)

**Authors**: Tianlong Yu, Yang Yang, Ziyi Zhou, Jiaying Xu, Siwei Li, Tong Guan, Kailong Wang, Ting Bi  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.23148v1  

#### Abstract
The emerging threat of AR-LLM-based Social Engineering (AR-LLM-SE) attacks (e.g. SEAR) poses a significant risk to real-world social interactions. In such an attack, a malicious actor uses Augmented Reality (AR) glasses to capture a target visual and vocal data. A Large Language Model (LLM) then ana...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 **AR-LLM** 的社会工程攻击框架（如 SEAR）在实际应用中面临两大瓶颈：

1. **Cold-start personalization（冷启动个性化）**：  
   依赖 **Retrieval-Augmented Generation (RAG)** 进行个人资料构建，导致首次交互时存在显著延迟（约 43.3 秒），破坏对话流畅性。

2. **Static attack strategies（静态攻击策略）**：  
   使用预设、固定阶段的提示模板（如“先破冰，再建立信任”），缺乏对目标实时反应的动态适应能力，难以应对非线性、不可预测的人类互动。

---

### **提出的新方法与创新思路**

作者提出 **PhySE** —— 一个融合心理学理论的实时 AR-LLM 社会工程攻击框架，包含两个核心组件：

#### ✅ **(1) VLM-Based Social-Context Training（基于视觉语言模型的社会上下文训练）**
- 采用 **Parameter-Efficient Fine-Tuning (PEFT)** 和 **LoRA** 对 **LLaVA-v1.5-7B** 进行微调。
- 将社交相关信息（如身份线索、兴趣、组织归属等）内化到 **Vision-Language Model (VLM)** 中，实现无需检索即可快速生成用户画像。
- 引入 **cross-modal contrastive alignment**（跨模态对比对齐）优化图像与文本描述的一致性。

> **优势**：消除冷启动延迟，提升 profile 一致性，支持低延迟实时交互。

#### ✅ **(2) Adaptive Psychological Agent（自适应心理代理）**
- 构建一个基于心理学理论的路由机制，动态选择三类策略：
  - **Warmth/Rapport（亲和力建立）**：通过共情、镜像、自我披露等方式建立连接。
  - **Credibility/Commitment（可信度强化）**：利用互惠、社会证明、权威线索增强可信感。
  - **Motivation/Action（动机引导）**：在信任足够后发起小请求，逐步升级至高风险行为。
- 引入 **latent trust state** 模型，基于目标响应信号（如回应积极性、犹豫程度）动态调整策略路径。

> **优势**：突破固定脚本限制，实现基于心理状态演化的动态战术控制，提升说服力与自然度。

---

### **相比现有方法的优势**
| 维度 | SEAR / Baseline | PhySE |
|------|------------------|--------|
| 冷启动延迟 | 高（~43.3s） | 显著降低（10.5s） |
| 策略灵活性 | 固定阶段、静态提示 | 动态路由、心理驱动 |
| 对话连贯性 | 易断裂、不一致 | 更稳定、上下文一致 |
| 用户体验 | 较好（4.73） | 最优（4.83） |
| 攻击有效性 | 中等 | 显著提升（尤其在 SMS/Call 场景） |

---

## **2. 核心实验方法和设置**

### **数据集**
- **PhySE Dataset**：由作者构建并公开发布，包含：
  - **360 条标注对话**，来自 **60 名参与者** 在真实场景下的互动（咖啡馆、社交活动等）。
  - 多模态数据流：AR 眼镜采集的视频、音频、环境元数据。
  - 公开社交痕迹：用于个性化的文本、图片、短视频。
  - 后续调查问卷：评估信任、自然度、接受意愿等主观指标。
  - 路由轨迹记录：每轮决策的心理状态与策略选择。

> 🔗 数据集与代码已开源：https://github.com/2192537130/PhySE

---

### **实验设置**
- **硬件平台**：
  - AR 设备：RayNeo X2（Android，6GB RAM）
  - 服务器：NVIDIA RTX 4090 + Intel Platinum 8352 CPU
- **模型配置**：
  - Base Model: LLaVA-v1.5-7B
  - VLM 微调：CLIP ViT-L/14 + LoRA (r=128, α=256)
  - Agent 控制：ReAct-style reasoning loop

---

### **评估指标**
| 类别 | 指标 |
|------|------|
| **用户体验质量** | Social Experience Score（5 分制 Likert 量表）<br>涵盖：自然度（naturalness）、真诚性（sincerity）、节奏（pacing）、相关性（relevance）等 11 个维度 |
| **攻击有效性** | 成功率指标：<br>- Photo Link（点击共享链接）<br>- Social App（添加社交好友）<br>- SMS（打开短信）<br>- Phone Call（接听电话） |
| **系统性能** | 延迟指标：<br>- Profile Generation Latency（最小、最大、P90、平均）<br>- Agent Response Time |
| **消融分析** | 移除 VLM 训练 / 心理代理后的性能下降情况 |

---

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Basic Conversation** | 无技术辅助的纯人工对话 |
| **Naive AR + LLM** | 使用 AR 捕获 + 多模态 LLM，但无个性化或策略设计 |
| **SEAR** | 当前主流 AR-LLM-SE 框架，依赖 RAG 与固定阶段提示 |
| **PhySE (Full)** | 完整框架（VLM + 心理代理） |
| **PhySE w/o VLM** | 移除社会上下文 VLM 训练 |
| **PhySE w/o Psychological Agent** | 移除心理路由模块 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### 📊 **社交体验得分（Social Experience Score）**
| 方法 | 平均分 (E[Score]) | 标准差 (σ) | 5分占比 |
|------|------------------|-----------|--------|
| Basic Conversation | 3.03 | 1.30 | 25.0% |
| Naive AR + LLM | 4.13 | 0.72 | 33.3% |
| SEAR | 4.73 | 0.51 | 76.7% |
| **PhySE (Ours)** | **4.83** | **0.37** | **83.3%** |

> ✅ PhySE 取得最高平均分且方差最低，表明其不仅表现更强，而且更稳定可靠。

---

#### ⏱️ **延迟对比（Profile Generation Latency）**
| 方法 | 组件 | 平均延迟 | P90 延迟 |
|------|------|----------|---------|
| SEAR | Multimodal LLM | 43.3 s | 52.7 s |
| **PhySE** | Trained VLM | **10.5 s** | **19.7 s** |

> ✅ 延迟减少 **75.7%**，有效缓解冷启动瓶颈。

---

#### 🎯 **社会工程攻击成功率**
![Figure 4: Social-Engineering Effectiveness](data:image/png;base64,...)

| 渠道 | SEAR | PhySE |
|------|------|-------|
| Photo Link | ~58% | ~64% |
| Social App | ~52% | ~60% |
| **SMS** | ~40% | **~54%** |
| **Phone Call** | ~38% | **~52%** |

> ✅ 在需要更高信任的渠道（SMS/Call）上提升最明显，说明 PhySE 更擅长建立深层信任关系。

---

#### 🔍 **消融实验结果（Ablation Study）**
| 配置 | 社交体验得分 | 下降幅度 |
|------|--------------|----------|
| Full PhySE | 4.83 ± 0.37 | — |
| w/o Trained VLM | 2.93 ± 1.14 | ↓39.3% |
| w/o Psychological Agent | 3.00 ± 1.13 | ↓37.9% |

> ✅ 两个模块均至关重要，单独缺失会导致性能大幅退化，验证了双组件协同设计的有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **理论驱动的心理适应显著提升 AR-LLM 社会工程攻击效果**：  
   将 **Stereotype Content Model (SCM)** 和 **Trust & Influence Theory** 融入 agent 决策过程，使攻击更具现实合理性与情感共鸣。

2. **VLM 内化社会知识可有效解决冷启动问题**：  
   通过 **social-context training** 替代传统 RAG，实现毫秒级 profile 生成，保障对话自然流动。

3. **动态策略路由优于固定脚本**：  
   自适应 agent 能根据目标反馈实时切换策略（如遇怀疑则退回 rapport 阶段），避免过早暴露意图。

4. **PhySE 在多维度全面超越现有方法**：  
   不仅在攻击成功率上领先，在用户体验、稳定性、延迟等方面也取得最优平衡。

---

### **局限性**
1. **依赖高质量公共社交数据**：若目标无足够线上痕迹，profile 准确性可能下降。
2. **伦理风险极高**：该框架本身为恶意用途设计，需严格管控以防滥用。
3. **现实部署仍受限于 AR 硬件算力**：目前依赖外接服务器，尚未完全端侧化。
4. **语言表达偶有人工痕迹**：部分建议语句略显机械，影响长期沉浸感。

---

### **未来工作方向**
1. **防御机制研究**：开发针对 PhySE 类攻击的检测工具，如异常对话模式识别、心理操纵信号预警。
2. **轻量化模型部署**：推动 VLM 与 agent 在 AR 设备本地运行，提升实用性。
3. **跨文化适应性扩展**：不同文化背景下社会工程策略差异大，需进行本地化调优。
4. **多智能体协作攻击模拟**：探索多个 PhySE agent 协同执行复杂社会工程任务的可能性。
5. **政策与法规建议**：呼吁制定关于 AR 感知 + LLM 联合使用的隐私保护标准。

---

> 💡 **总结一句话**：  
> **PhySE 首次将心理学理论系统性地融入 AR-LLM 社会工程攻击流程，实现了从“自动化脚本”到“类人心理博弈”的跃迁，在效率、真实性与攻击成功率上全面突破现有极限。**

</details>

---

### 14. [Bridging Reasoning and Action: Hybrid LLM-RL Framework for Efficient Cross-Domain Task-Oriented Dialogue](https://arxiv.org/abs/2604.23345)

**Authors**: Yangyang Zhao, Linfan Dai, Li Cai, Bowen Xing, Libo Qin  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.23345v1  

#### Abstract
Cross-domain task-oriented dialogue requires reasoning over implicit and explicit feasibility constraints while planning long-horizon, multi-turn actions. Large language models (LLMs) can infer such constraints but are unreliable over long horizons, while Reinforcement learning (RL) optimizes long-h...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Bridging Reasoning and Action: Hybrid LLM-RL Framework for Efficient Cross-Domain Task-Oriented Dialogue 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
跨领域任务导向型对话（Cross-domain Task-Oriented Dialogue, TOD）系统面临一个核心挑战：**如何在长周期、多轮次的交互中，同时处理显式和隐式的可行性约束（feasibility constraints）**。

- 显式约束（如“目的地是纽约”）可直接从用户话语中提取；
- 隐式约束（如“酒店入住时间必须晚于航班到达时间”）需要常识推理或时序逻辑推断。

现有方法存在明显瓶颈：
- **纯 LLM 方法**：虽具备强大的推理能力，但输出易出现幻觉（hallucination）和跨轮次不一致，难以稳定支持长期决策。
- **纯 RL 方法**：擅长优化长周期策略，但依赖准确完整的状态表示，无法从原始对话中恢复缺失的隐式约束。
- **简单结合 LLM + RL**：未经验证的 LLM 输出会污染状态表示，误导策略学习，导致性能下降。

### 提出了什么新方法或新思路
作者提出 **Verified LLM-Knowledge empowered RL (VLK-RL)**，一种将 LLM 推理与 RL 决策解耦的混合框架，其核心思想是：

> **将 LLM 的约束推理能力用于构建更丰富、可靠的状态表示，而非直接生成动作。**

该框架包含三个模块化阶段：

1. **Dual-role LLM Cross-examination（双角色交叉质询）**  
   - 使用两个 LLM 分别扮演 **Respondent（提出候选约束）** 和 **Judge（通过提问验证约束）**。
   - 通过多轮问答机制检测逻辑矛盾、模糊回应等信号，过滤不可靠的推理结果，提升知识可靠性。

2. **Text-to-Slot Mapper（文本到槽位映射器）**  
   - 将经过验证的自然语言约束转换为结构化的 `slot-value` 对。
   - 利用 **Sentence-BERT** 进行语义相似度匹配，实现值归一化（如 “NYC downtown” → “Manhattan”），确保与数据库本体对齐。

3. **RL-based Policy Optimization（基于 RL 的策略优化）**  
   - 在增强后的结构化状态 $ s' = s_t \cup v_t $ 上训练标准 RL 策略（默认使用 PPO）。
   - 不改变原有策略架构、动作空间或奖励函数，仅通过状态增强提升鲁棒性。

### 相比现有方法的优势
- ✅ **可靠性更高**：通过双角色交叉质询显著减少 LLM 幻觉和不一致性。
- ✅ **可执行性强**：结构化槽位表示可直接对接数据库和下游策略模型。
- ✅ **模块化设计**：各组件独立可替换，兼容多种 LLM 和 RL 后端。
- ✅ **无需微调**：所有 LLM 均以 off-the-shelf 方式使用，无任务特定训练。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **MultiWOZ 2.1**  
  包含超过 10k 条人类对话语料，涵盖 7 个领域（酒店、火车、餐厅等），具有丰富的跨域依赖关系。
- **Frames**  
  Wizard-of-Oz 数据集，子任务之间有强可行性要求（如行程顺序、时间衔接），更适合测试复杂约束建模能力。

统一使用 **ConvLab-2** 工具包进行模拟环境搭建、数据库访问和评估。

### 实验设置和评估指标
#### 超参数配置
- 最大对话长度 $ L = 30 $
- 批大小 $ = 100 $
- 交叉质询轮数 $ R = 5 $
- 归一化阈值 $ \tau = 0.7 $
- RL 算法：PPO（默认）
- LLM 主干模型：Qwen2-7B-Instruct, Qwen1.5-14B-Chat (GPTQ-Int4), GPT-4o-mini

#### 评估指标
| 指标 | 说明 |
|------|------|
| **Precision / Recall / F1** | 对话行为（dialogue act）级别的预测准确性 |
| **Complete Rate** | 成功完成所有子任务的比例 |
| **Success Rate** | 完成且未违反任何约束的任务比例 |
| **Avg. Turns (Succ / All)** | 成功对话/全部对话的平均轮次（越低越好） |

此外还进行了 **人工评测（Human Evaluation）**，由 30 名标注员打分：
- **SR (Success Rate)**：任务是否成功完成
- **HR (Human Rating)**：流畅性、自然性和冗余度（1–5 分制）

### 基线方法对比
| 类型 | 基线模型 |
|------|---------|
| **RL-based** | PPO, ACGOS |
| **LLM-based** | GALAXY, GDP-Zero, TransferTOD |
| **DST-based** | CAPID |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

#### MultiWOZ 2.1 结果（最高值加粗）
| Model | Success/Tot | Complete/Tot | Avg. Turn (All) |
|-------|-------------|----------------|------------------|
| PPO | 0.3815 | 0.4912 | 20.94 |
| GDP-Zero | 0.6024 | 0.7025 | 22.22 |
| CAPID | 0.5875 | 0.6820 | 20.00 |
| **VLK-RL (Qwen-14B)** | **0.7214** | **0.8006** | **17.35** |

#### Frames 结果
| Model | Success/Tot | Complete/Tot | Avg. Turn (All) |
|-------|-------------|----------------|------------------|
| PPO | 0.4235 | 0.6031 | 18.56 |
| GDP-Zero | 0.5890 | 0.7215 | 17.84 |
| CAPID | 0.5782 | 0.6701 | 20.00 |
| **VLK-RL (Qwen-14B)** | **0.7239** | **0.8063** | **15.91** |

> 🔍 **观察**：VLK-RL 在两个数据集上均取得最优表现，尤其在约束更强的 **Frames** 上优势更为显著。

### 与基线方法的对比结果
- 相比最强 RL 基线（ACGOS）：
  - Success 提升约 **+27%**（MultiWOZ）、**+30%**（Frames）
  - 平均轮次减少 **~3 轮以上**
- 相比最强 LLM 基线（GDP-Zero）：
  - 更高成功率的同时，对话更短、更高效
- 相比 DST 基线（CAPID）：
  - 显著降低因忽略隐式约束导致的失败

### 消融实验结果（Ablation Study）
使用 VLK-RL(Qwen-14B) 进行消融分析（见 Figure 4）：

| 变体 | Success/Tot ↓ | Complete/Tot ↓ | Avg. Turn (All) ↑ |
|------|---------------|----------------|--------------------|
| Full VLK-RL | **0.7214** | **0.8006** | **17.35** |
| w/o Cross-Examination | 0.4900 | 0.5400 | 21.30 |
| w/o T2S Mapper | 0.5124 | 0.5732 | 20.87 |
| w/o RL (LLM-only) | 0.5308 | 0.5901 | 21.02 |
| w/o LLM (RL-only) | 0.3815 | 0.4912 | 20.94 |

> 📌 **结论**：
- 移除任一组件都会造成显著性能下降；
- **双角色验证** 和 **槽位映射** 是提升成功率的关键；
- 单独使用 LLM 或 RL 均不足以应对复杂跨域任务。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **状态完整性是跨域 TOD 的关键瓶颈**  
   是否能捕捉显式与隐式可行性约束，决定了策略能否做出全局有效的长期规划。

2. **LLM 与 RL 应解耦而非紧耦合**  
   将 LLM 用于“知识提取 + 状态增强”，而 RL 专注于“策略优化”，比端到端联合建模更稳定、更高效。

3. **双角色交叉质询有效抑制幻觉**  
   实验表明，该机制可将幻觉率从 23.5% 降至 6.5%，不一致性从 22.0% 降至 7.5%（见 Appendix F）。

4. **结构化表示优于自由文本注入**  
   即使使用经过验证的知识，若不进行槽位归一化，仍会导致执行失败或歧义（见 w/o T2S Mapper 实验）。

5. **框架具有良好的扩展性与兼容性**
   - 支持不同 LLM（Qwen / GPT-4o-mini）和 RL 后端（PPO / DQN / PG）；
   - 同模型双角色优于跨模型组合（见 Appendix D）。

### 方法的局限性
1. **计算开销较大**  
   双 LLM 交互 + 多轮质询带来额外延迟，不适合资源受限场景。

2. **依赖预训练 LLM 的质量**  
   若 LLM 缺乏相关常识或推理能力弱，会影响初始约束提取效果。

3. **本体覆盖有限**  
   无法表达超出预定义 `slot-value` 结构的复杂约束（如时间区间、偏好排序等）。

4. **验证非完备**  
   交叉质询不能保证发现所有错误，尤其在模糊或多义上下文中可能漏检。

### 未来工作方向
- 将验证与映射模块蒸馏为轻量级模型，降低部署成本。
- 引入外部知识检索（Retrieval-Augmented）辅助罕见约束判断。
- 扩展约束表示形式，支持更复杂的逻辑结构（如一阶谓词、时间逻辑）。
- 探索动态本体适配机制，提升对新领域的泛化能力。

--- 

> ✅ **总体评价**：VLK-RL 提出了一种**模块化、可解释、高鲁棒性**的 LLM-RL 协同范式，在跨域 TOD 任务中实现了显著性能突破，为构建可信的智能对话系统提供了重要参考路径。

</details>

---

### 15. [OS-SPEAR: A Toolkit for the Safety, Performance,Efficiency, and Robustness Analysis of OS Agents](https://arxiv.org/abs/2604.24348)

**Authors**: Zheng Wu, Yi Hua, Zhaoyuan Huang, Chenhao Xue, Yijie Lu, Pengzhou Cheng, Zongru Wu, Lingzhong Dong, Gongshen Liu, Xinghao Jiang, Zhuosheng Zhang  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.24348v1  

#### Abstract
The evolution of Multimodal Large Language Models (MLLMs) has shifted the focus from text generation to active behavioral execution, particularly via OS agents navigating complex GUIs. However, the transition of these agents into trustworthy daily partners is hindered by a lack of rigorous evaluatio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：OS-SPEAR: A Toolkit for the Safety, Performance, Efficiency, and Robustness Analysis of OS Agents

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前对 **OS Agent**（操作系统代理）的评估存在严重不足，主要集中在任务完成率（task completion rate），而忽视了实际部署中至关重要的四个维度：
- **Safety（安全）**：面对环境干扰、恶意诱导时的行为可靠性；
- **Performance（性能）**：在不同难度任务下的真实执行能力；
- **Efficiency（效率）**：推理延迟与 token 消耗等用户成本相关指标；
- **Robustness（鲁棒性）**：在视觉与文本输入受到扰动时的稳定性。

现有基准（如 WebArena、AndroidWorld）大多仅关注单一维度，且存在以下问题：
- 安全场景狭窄（如只测试弹窗）；
- 轨迹标注噪声大，包含低价值或无法完成的任务；
- 效率评估仅用“步数”衡量，忽略时间和 token 成本；
- 鲁棒性评估局限于单模态（仅视觉或仅文本）。

---

### 🚀 提出的新方法与创新
作者提出 **OS-SPEAR** —— 一个系统性的多维评估工具包，涵盖四大子集：

| 子集 | 功能 |
|------|------|
| **S-subset (Safety)** | 包含环境干扰（pop-ups）、现实异常（网络中断）、对抗误导（prompt injection）三类风险，全面测试 agent 在危险情境下的行为安全性。 |
| **P-subset (Performance)** | 通过轨迹价值估计（trajectory value estimation）和分层采样（stratified sampling）筛选高质量、高区分度的轨迹，避免低质数据带来的偏差。 |
| **E-subset (Efficiency)** | 引入双重视角：**时间延迟** 和 **token 消耗**（按主流 API 计费方式加权：input + 3 × output）。 |
| **R-subset (Robustness)** | 设计 **10 种跨模态扰动**，覆盖视觉（mask、zoom-in、Gaussian noise）和文本（state conflict、bad memory、irrelevant knowledge）两大模态。 |

此外，提供一个基于 **multi-agent system** 的自动化分析工具，可生成人类可读的诊断报告，提升评估结果的可解释性。

---

### 🔍 相比现有方法的优势
| 维度 | OS-SPEAR 的优势 |
|------|------------------|
| **全面性** | 首个同时覆盖 Safety, Performance, Efficiency, Robustness 四大维度的统一框架。 |
| **真实性** | 扰动设计贴近真实世界（real-world anomalies），而非人工构造的理想化攻击。 |
| **标准化** | 提供统一评分机制与综合排名体系，便于横向比较不同模型。 |
| **实用性** | 考虑用户关心的成本因素（token、time），更具工程指导意义。 |
| **可扩展性** | 子集模块化设计，支持后续新增任务或扰动类型。 |

---

## 2. 核心实验方法和设置

### 📚 数据集来源
OS-SPEAR 并非从零构建，而是基于多个现有基准进行清洗、重构与增强：
- **AITW**, **AndroidControl**, **GUI-Odyssey**：作为初始轨迹池用于构建 P-subset 和 R-subset。
- **Screenspot**, **MemGUI-Bench**, **EnvDistraction** 等：为 S-subset 提供安全场景模板。
- 最终整合成四个专用子集，总计包含数千条高质量轨迹与扰动样本。

---

### ⚙️ 实验设置
- **被测对象**：共评测 **22 个主流 OS Agent**，包括：
  - 通用型 MLLM：Qwen2.5-VL 系列、Qwen3-VL 系列、GLM-4.5V
  - 专用型 Agent：UI-TARS 系列、GUI-Owl 系列、OS-Atlas-Pro、AgentCPM-GUI 等
- **运行环境**：模拟真实 GUI 环境（手机/电脑界面截图 + 指令输入）
- **执行流程**：每个 agent 接收指令 → 观察截图 → 输出 action（CLICK / TYPE / SCROLL 等）→ 执行并更新状态 → 循环直至完成或超时。

---

### 📊 评估指标
#### 各子集独立指标：
| 子集 | 主要指标 |
|------|--------|
| **S-subset** | Gold（正确动作比例）、Dist.（被误导动作比例）、Inv.（无效动作比例）；最终以 `Gold - Dist.` 得分排序 |
| **P-subset** | Type Accuracy（动作类型准确率）、SR（step-wise success rate）、TSR（trajectory success rate） |
| **E-subset** | 平均每步 inference time、input/output token 数；总成本 = input + 3×output |
| **R-subset** | 在 10 种扰动下相对于 normal 条件的性能下降幅度（ΔSR），越小越鲁棒 |

#### 综合排名规则：
- 每个子集内独立排名；
- 总体排名 = 四个子集排名的平均值。

---

### 🆚 基线方法对比
所有 22 个模型均为当前主流 OS Agent，构成互为基线的横向对比组。特别关注：
- 通用 vs 专用模型（如 Qwen3-VL vs UI-TARS）
- 同一系列不同规模（如 Qwen3-VL-2B vs Qwen3-VL-32B）
- 参数量大小的影响（small vs large models）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table V）

| 模型 | S排名 | P排名 | E排名 | R排名 | **总排名** |
|------|-----|-----|-----|-----|----------|
| **UI-Venus-1.5-8B** | 1 | 5 | 7 | 10 | **1** |
| UI-TARS-72B-SFT | 4 | 4 | 12 | 7 | 2 |
| UI-TARS-7B-SFT | 6 | 13 | 3 | 6 | 3 |
| GUI-Owl-32B | 9 | 2 | 18 | 1 | 4 |
| AgentCPM-GUI-8B | 15 | 1 | 2 | 13 | 5 |

> ✅ **UI-Venus-1.5-8B** 凭借均衡表现获得第一，尤其在安全性和效率上表现出色。

---

### 🔬 与基线方法的关键对比发现

#### （1）专用 Agent 显著优于通用模型
- 所有进入总榜前 8 的模型均为 **专为 OS 任务设计的 agent**；
- 通用模型（如 Qwen、GLM）虽理解能力强，但在安全与鲁棒性上更易受干扰。

#### （2）效率 ≠ 性能
- **OS-Atlas-Pro-7B** 在 E-subset 中排名第一（极高效），但因安全薄弱（S排名第20）拖累总体表现；
- 表明追求极致效率可能牺牲安全性。

#### （3）参数规模并非万能
- 更大的模型通常更安全（如 ≥32B 模型普遍 S 排名靠前）；
- 但在同系列中，**Qwen3-VL-4B 总体排名高于 Qwen3-VL-32B**，说明单纯扩参不等于更强综合能力。

#### （4）GUI-Owl-32B 是最稳健的 agent
- 在 R-subset 中排名第一，对多种扰动具有强抵抗力；
- 但因其生成大量 thought 导致 token 消耗过高，E-subset 排名垫底（第18），影响总成绩。

---

### 🔍 消融实验与关键观察（隐含于分析中）
虽然未设传统消融实验，但通过控制变量得出以下结论：
- **轨迹质量显著影响性能评估准确性**：使用原始数据会导致 TSR 虚高，经 P-subset 清洗后更能反映真实水平。
- **扰动强度验证鲁棒性梯度**：随着 Gaussian noise 强度增加，多数模型性能逐步下降，证明 R-subset 具备敏感性。
- **跨模态扰动揭示脆弱点**：例如，“state conflict”提示任务已完成，仍导致部分 agent 错误终止，暴露逻辑缺陷。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **效率常以牺牲安全与鲁棒性为代价**  
   → 快速响应的模型往往跳过充分推理，容易被误导。

2. **专用 Agent > 通用 MLLM**  
   → 领域适配（domain adaptation）有效提升了任务执行的安全性、效率与一致性。

3. **大模型更安全，但不一定更优**  
   → 大模型能更好识别潜在风险，但若无架构优化，会带来高昂推理成本。

4. **视觉完整性至关重要**  
   → 即使目标区域可见，mask 或 zoom-in 也会大幅降低性能，表明 agent 过度依赖全局上下文。

5. **文本扰动中，“无关记忆”与“状态冲突”最具破坏力**  
   → 表明 agent 对新增语义信号过于敏感，缺乏信息过滤机制。

6. **时间与 token 成本无强相关性**  
   → 有些模型推理快但输出冗长，token 消耗反而更高。

---

### ⚠️ 方法的局限性
- **静态评估为主**：目前测试基于预定义轨迹，缺乏动态交互反馈（如用户中途干预）；
- **扰动种类有限**：尽管已有10种扰动，但仍难以穷尽真实世界的复杂干扰；
- **平台偏向性**：主要面向移动端 GUI，对桌面端或多设备协同支持较弱；
- **未考虑长期记忆影响**：所有任务相对独立，未测试跨任务记忆累积效应。

---

### 🔮 未来工作方向
1. **引入动态对抗环境**：允许 adversary 实时生成扰动，形成红蓝对抗测试机制；
2. **扩展至多轮复杂任务流**：评估 agent 在长时间、多目标任务中的持续表现；
3. **加入 human-in-the-loop 评估**：结合人工判断提升评估信度；
4. **开发轻量化版本**：推动 OS-SPEAR 成为开源社区标准 benchmark；
5. **探索自动修复建议机制**：不仅诊断问题，还能推荐改进策略（如 prompt 修改、微调数据建议）。

---

## 总结

> **OS-SPEAR 是首个将 Safety、Performance、Efficiency、Robustness 四维一体融合的 OS Agent 评估框架**。它不仅填补了现有 benchmark 的结构性空白，还通过严谨的数据筛选与多模态扰动设计，揭示了当前 agent 在真实应用场景中的核心短板。其发布的 **22 模型排行榜** 为研究者提供了宝贵的参考基准，标志着 OS Agent 评估正式迈向系统化、标准化时代。

🔗 开源地址：[https://github.com/Wuzheng02/OS-SPEAR](https://github.com/Wuzheng02/OS-SPEAR)

</details>

---

### 16. [CoFi-PGMA: Counterfactual Policy Gradients under Filtered Feedback for Multi-Agent LLMs](https://arxiv.org/abs/2604.22785)

**Authors**: Stela Tong, Elai Ben-Gal  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.22785v1  

#### Abstract
Large language model (LLM) deployments increasingly rely on multi-agent architectures in which multiple models either compete through routing mechanisms or collaborate to produce a final answer. In both settings, the learning signal received by each agent is filtered by the system mechanism. Routing...

---

### 17. [Fed-DLoRA: Efficient Wireless Federated Learning with Dynamic Low-Rank Adaptation](https://arxiv.org/abs/2604.24103)

**Authors**: Huaicheng Li, Junhui Zhao, Haoyu Quan, Xiaoming Wang  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.24103v1  

#### Abstract
Federated learning (FL) offers a promising distributed learning paradigm for internet of vehicles (IoV) applications. However, it faces challenges from communication overhead and dynamic environments. Model compression techniques reduce computing and communication burden yet create trade-offs betwee...

---

### 18. [RouteNLP: Closed-Loop LLM Routing with Conformal Cascading and Distillation Co-Optimization](https://arxiv.org/abs/2604.23577)

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.23577v1  

#### Abstract
Serving diverse NLP workloads with large language models is costly: at one enterprise partner, inference costs exceeded $200K/month despite over 70% of queries being routine tasks well within the capability of smaller models. We present RouteNLP, a closed-loop framework that routes queries across a ...

---

### 19. [LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models](https://arxiv.org/abs/2604.23809)

**Authors**: Tianchun Li, Haochen Liu, Vishwa Pardeshi, Xingchen Wang, Tianci Liu, Huijun Zhao, Wei Fan, Jing Gao  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.23809v1  

#### Abstract
Small language models (SLMs) are promising for real-world deployment due to their efficiency and low operational cost. However, their limited capacity struggles with high-stakes legal reasoning tasks that require coherent statute interpretation and logically consistent deduction. Furthermore, traini...

---

### 20. [Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines](https://arxiv.org/abs/2604.23178)

**Authors**: Sadman Kabir Soumik  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23178v1  

#### Abstract
LLM-as-a-Judge has become the dominant paradigm for evaluating language model outputs, yet LLM judges exhibit systematic biases that compromise evaluation reliability. We present a comprehensive empirical study comparing nine debiasing strategies across five judge models from four provider families ...

---

### 21. [Agentic Adversarial Rewriting Exposes Architectural Vulnerabilities in Black-Box NLP Pipelines](https://arxiv.org/abs/2604.23483)

**Authors**: Mazal Bethany, Kim-Kwang Raymond Choo, Nishant Vishwamitra, Peyman Najafirad  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23483v1  

#### Abstract
Multi-component natural language processing (NLP) pipelines are increasingly deployed for high-stakes decisions, yet no existing adversarial method can test their robustness under realistic conditions: binary-only feedback, no gradient access, and strict query budgets. We formalize this strict black...

---

### 22. [LLM-Guided Agentic Floor Plan Parsing for Accessible Indoor Navigation of Blind and Low-Vision People](https://arxiv.org/abs/2604.23970)

**Authors**: Aydin Ayanzadeh, Tim Oates  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23970v1  

#### Abstract
Indoor navigation remains a critical accessibility challenge for the blind and low-vision (BLV) individuals, as existing solutions rely on costly per-building infrastructure. We present an agentic framework that converts a single floor plan image into a structured, retrievable knowledge base to gene...

---

### 23. [Beyond the Attention Stability Boundary: Agentic Self-Synthesizing Reasoning Protocols](https://arxiv.org/abs/2604.24512)

**Authors**: Dahlia Shehata, Ming Li  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.24512v1  

#### Abstract
As LLM agents transition to autonomous digital coworkers, maintaining deterministic goal-directedness in non-linear multi-turn conversations emerged as an architectural bottleneck. We identify and formalize a systemic failure mode termed the Attention Latch in decoder-only autoregressive Transformer...

---

### 24. [XGRAG: A Graph-Native Framework for Explaining KG-based Retrieval-Augmented Generation](https://arxiv.org/abs/2604.24623)

**Authors**: Zhuoling Li, Ha Linh Hong Tran Nguyen, Valeria Bladinieres, Maxim Romanovsky  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.24623v1  

#### Abstract
Graph-based Retrieval-Augmented Generation (GraphRAG) extends traditional RAG by using knowledge graphs (KGs) to give large language models (LLMs) a structured, semantically coherent context, yielding more grounded answers. However, GraphRAG reasoning process remains a black-box, limiting our abilit...

---

### 25. [TSAssistant: A Human-in-the-Loop Agentic Framework for Automated Target Safety Assessment](https://arxiv.org/abs/2604.23938)

**Authors**: Xiaochen Zheng, Zhiwen Jiang, Melanie Guerard, Klas Hatje, Tatyana Doktorova  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23938v1  

#### Abstract
Target Safety Assessment (TSA) requires systematic integration of heterogeneous evidence, including genetic, transcriptomic, target homology, pharmacological, and clinical data, to evaluate potential safety liabilities of therapeutic targets. This process is inherently iterative and expert-driven, p...

---

### 26. [DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](https://arxiv.org/abs/2604.24320)

**Authors**: Junshuo Zhang, Chengrui Huang, Feng Guo, Zihan Li, Ke Shi, Menghua Jiang, Jiguo Yu, Shuo Shang, Shen Gao  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.24320v1  

#### Abstract
Large language model (LLM) agents that follow the sequential "reason-then-act" paradigm have achieved superior performance in many complex tasks.However, these methods suffer from limited exploration and incomplete environmental understanding, as they interact with only a single environment per step...

---

### 27. [Learning Without Adversarial Training: A Physics-Informed Neural Network for Secure Power System State Estimation under False Data Injection Attacks](https://arxiv.org/abs/2604.22784)

**Authors**: Solon Falas, Markos Asprou, Charalambos Konstantinou, Maria K. Michael  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.22784v1  

#### Abstract
State estimation is a cornerstone of power system control-center operations, and its robust operation is increasingly a cyber-physical security concern as modern grids become more digitalized and communication-intensive. Neural network-based approaches have gained attention as alternatives to conven...

---

### 28. [Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces](https://arxiv.org/abs/2604.22948)

**Authors**: Jie Hu, Lingyun Chen, Geeho Kim, Jinyoung Choi, Bohyung Han, Do Young Eun  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.22948v1  

#### Abstract
History-dependent sampling can reduce long-run Monte Carlo variance by discouraging redundant revisits, but existing schemes typically encode history through empirical measure on finite state spaces, which is infeasible in high-dimensional discrete configuration spaces or ill-posed in continuous dom...

---

### 29. [Efficient VQ-QAT and Mixed Vector/Linear quantized Neural Networks](https://arxiv.org/abs/2604.23172)

**Authors**: Terry Gou, Puneet Gupta  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23172v1  

#### Abstract
In this work, we developed and tested 3 techniques for vector quantization (VQ) based model weight compression. To mitigate codebook collapse and enable end-to-end training, we adopted cosine similarity-based assignment. Building on ideas from attention-based formulations in Differentiable K-Means (...

---

### 30. [Hamiltonian Graph Inference Networks: Joint structure discovery and dynamics prediction for lattice Hamiltonian systems from trajectory data](https://arxiv.org/abs/2604.23606)

**Authors**: Ru Geng, Panayotis Kevrekidis, Yixian Gao, Hong-Kun Zhang, Jian Zu  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23606v1  

#### Abstract
Lattice Hamiltonian systems underpin models across condensed matter, nonlinear optics, and biophysics, yet learning their dynamics from data is obstructed by two unknowns: the interaction topology and whether node dynamics are homogeneous. Existing graph-based approaches either assume the graph is g...

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
