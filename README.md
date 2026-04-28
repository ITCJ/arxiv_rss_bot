# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-28 07:59:52 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [JigsawRL: Assembling RL Pipelines for Efficient LLM Post-Training](https://arxiv.org/abs/2604.23838)

**Authors**: Zhengding Hu, Hehua Ouyang, Chang Chen, Zaifeng Pan, Yue Guan, Zhongkai Yu, Zhen Wang, Steven Swanson, Yufei Ding  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.23838v1  

#### Abstract
We present JigsawRL, a cost-efficient framework that explores Pipeline Multiplexing as a new dimension of RL parallelism. JigsawRL decomposes each pipeline into a Sub-Stage Graph that exposes the intra-stage and inter-worker imbalance hidden by stage-level systems. On this abstraction, JigsawRL reso...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：JigsawRL: Assembling RL Pipelines for Efficient LLM Post-Training**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 LLM 强化学习（RL）后训练框架在**成本效率**方面存在严重不足。尽管已有系统优化了时间效率，但由于以下原因导致资源利用率低下：
- **Rollout 阶段的长尾效应**：少数样本解码时间极长，拖慢整个 batch，造成 GPU 利用率下降。
- **阶段间与阶段内的负载不均衡**：Rollout 和 Training 阶段之间同步阻塞，且单个阶段内部也存在计算密度波动（如 Prefill 与 Decoding）。
- **多工作负载并行时的资源争用与碎片化**：现有系统无法有效协调多个并发 RL pipeline 的执行。

这些问题导致即使增加 GPU 资源，吞吐量提升有限，而金钱成本迅速上升（见图1），**MFU（Model FLOPs Utilization）普遍低于10%**。

---

### **提出的新方法与思路**
作者提出了 **JigsawRL**，一种基于 **Pipeline Multiplexing** 的新型 RL 并行范式，其核心是将传统粗粒度的“阶段级”调度细化为“子阶段级”调度。

#### **关键创新点：**
1. **Sub-Stage Graph 抽象**
   - 将每个 RL pipeline 分解为细粒度的 **Sub-Stage Graph**，暴露 stage-level 系统无法看到的 **intra-stage 和 inter-worker 不平衡**。
   - 子阶段按 token 处理量划分（如 `[0,128)`, `[128,1024)`, `≥1024`），具有不同的计算与内存特征。

2. **Sub-Stage Multiplexing（子阶段复用）**
   - 利用不同子阶段对资源需求的互补性（如 compute-bound Training vs. memory-bound long-tail Rollout），通过动态资源分配实现跨 pipeline 的并发执行。
   - 使用 **NVIDIA Green Context** 进行动态 SM 分配，避免计算资源争用。

3. **Sub-Stage Merging（子阶段合并）**
   - 识别出多个 DP worker 上低利用率的长尾 rollout 子阶段，将其迁移到少数 worker 上聚合执行，释放其他 worker 资源用于高价值任务。
   - 利用 KV Cache 重计算机制完成迁移。

4. **Look-ahead Graph Scheduling**
   - 将调度建模为图问题，采用前瞻启发式算法选择最优的子阶段组合与资源分配策略，最小化关键路径延迟。

---

### **相比现有方法的优势**
| 维度 | JigsawRL | 传统方法（Verl, StreamRL 等） |
|------|----------|-------------------------------|
| **抽象粒度** | Sub-stage level | Stage level |
| **资源利用** | 动态分配 SM/Mem，支持互补复用 | 固定或粗粒度划分 |
| **长尾处理** | 跨 DP worker 迁移与聚合 | 仅限于单 pipeline 内部缓解 |
| **调度智能** | 基于图模型的 lookahead 调度 | 贪心或静态调度 |
| **适用场景** | 支持异构 pipeline 混合部署 | 通常假设同构 pipeline |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Single-turn 推理任务**：
  - `GSM8K`：数学应用题
  - `MATH`：复杂数学问题
- **Multi-turn 自我迭代任务**：
  - `AIME`：更具挑战性的推理数据集
- **Tool-using 外部工具调用任务**：
  - `HotpotQA`：多跳问答，结合 RAG
  - 外部数据库：`MS MARCO` + FAISS IVF4096 索引

---

### **模型**
- **Base Models**：`Qwen3-0.6B`, `Qwen3-4B`, `Qwen3-32B`
- **Instruct-tuned Models**：`Qwen3-4B-Instruct`, `Llama-3.1-8B-Instruct`
- **Distilled Model**：`DeepSeek-R1-Distill-Qwen2.5-14B`

---

### **实验设置**
- **硬件平台**：
  - 单节点：8 × H100 GPUs（NVLink）
  - 多节点：最多 64 × A100 GPUs（每节点 4 GPU，NVLink + InfiniBand）
- **RL 算法**：GRPO（Group Relative Policy Optimization）
- **全局 batch size**：64，group size = 4，最大响应长度 8192 tokens
- **Backend 实现**：
  - Rollout：SGLang / vLLM
  - Training：FSDP / Megatron

---

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Throughput (token/s)** | 所有并发 pipeline 每秒处理的总 token 数（主指标） |
| **Latency Increase** | 单个 pipeline 步骤延迟相对于独占运行的增长倍数 |
| **MFU (%)** | GPU 实际算力利用率（FLOPs utilization） |
| **Cost Efficiency** | 吞吐量与金钱成本之间的权衡关系 |

---

### **基线方法对比**
| 类型 | 基线方法 | 特点 |
|------|--------|------|
| **Synchronous RL** | `Verl`, `RollMux` | 严格同步交替 rollout 与 training |
| **Asynchronous RL** | `StreamRL`, `AReaL` | 允许一定 staleness 的异步执行 |
| **Rollout Multiplexing** | `JigsawRL-MuxServe` | 仅在 rollout 阶段使用 MuxServe 复用 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **吞吐量提升显著**
- 在 **4–64 块 H100/A100 GPU** 上测试：
  - 相比 **Verl（同步）**：最高达 **1.85× 吞吐提升**，平均 **1.56×**
  - 相比 **StreamRL / AReaL（异步）**：最高达 **1.54× 吞吐提升**
- 图13 显示，在 64 A100 GPU 上仍保持良好扩展性，**JigsawRL 达到 Verl 的 1.75× 吞吐**

#### ✅ **资源利用率大幅提升**
- 平均 **MFU 提升至 3.89%**（Verl 仅为 2.29%，JigsawRL-MuxServe 为 2.57%）——见图19
- 通过子阶段复用，有效填充了原本空闲的时间窗口

#### ✅ **延迟增长可控**
- 尽管引入了 multiplexing，但 per-pipeline latency 增加适度：
  - 平均仅 **1.48× 延迟增长**，最低可达 **1.14×**
  - 远优于串行运行两个 pipeline 的 **2× 延耗**

#### ✅ **支持异构 pipeline 混合部署**
- 可同时运行不同模型大小（如 Qwen3-0.6B 与 Llama-3.1-8B）或不同类型（同步 + 异步）pipeline
- 在异构设置下，吞吐提升更明显（小模型可填补大模型长尾间隙）

---

### **消融实验结果**

#### 🔹 **Dynamic Sub-Stage Multiplexing 的有效性**
- 对比三种策略（图19）：
  - `Verl`（串行）：MFU 最低（2.29%）
  - `JigsawRL-MuxServe`（仅 rollout 复用）：略有改善（2.57%），但 training 仍串行
  - `JigsawRL`（全子阶段动态复用）：MFU 提升至 **3.89%**，**较 Verl 提升 1.7×**

#### 🔹 **Inter-DP Workload Migration 的影响**
- 图20 显示，启用跨 DP worker 的长尾样本迁移后：
  - Pipeline A 的长尾 rollout 被集中到一个 worker
  - 显著减少对 Pipeline B 的干扰
  - 总体训练步骤延迟降低，**实现双赢**

#### 🔹 **与其它并行方式的兼容性**
- 图18 表明，JigsawRL 在相同 TP/DP 设置下，进一步带来 **1.16× ~ 1.48× 吞吐增益**
- 说明 **Pipeline Multiplexing 是一种正交的新维度并行方式**，可与其他并行技术叠加使用

---

## **4. 关键结论和发现**

### **主要发现**
1. **Rollout 阶段的长尾效应是制约 RL 成本效率的根本瓶颈**，必须从“子阶段”视角重新建模。
2. **不同子阶段具有互补的资源需求模式**（compute-bound vs. memory-bound），为跨 pipeline 复用提供了机会。
3. **Pipeline Multiplexing 是一种新的 RL 并行维度**，能显著提升资源利用率和吞吐量。
4. **JigsawRL 的设计在同步与异步 RL 中均有效**，且支持异构混合部署，具备强通用性。

---

### **方法的局限性**
1. **目前聚焦中小规模 dense 模型**（0.6B–32B），尚未验证在 MoE 架构上的效果。
2. **依赖近期历史行为进行预测建图**，若 workload 变化剧烈可能影响调度准确性。
3. **KV Cache 重计算带来一定开销**，虽被摊销但仍需权衡。
4. **未完全解决异步 RL 中的数据陈旧性（data staleness）问题**，仅作为补充优化。

---

### **未来工作方向**
1. **扩展至 MoE 架构**：利用专家级别的不平衡性，实现更细粒度的 multiplexing。
2. **集成 LoRA-style 参数高效微调**：共享 base model 下，adapter 的 memory footprint 差异天然适合 multiplexing。
3. **支持 Serverless 与 Preemptible 资源调度**：适应云原生环境下的弹性部署。
4. **自动化的 parallelism search 与 multiplexing 联合优化**：与 ReaL 等框架结合，实现端到端最优配置。

---

> 📌 **一句话总结**：  
> JigsawRL 通过 **Sub-Stage Graph** 抽象和 **look-ahead graph scheduling**，首次将 RL pipeline 的复用粒度推进到子阶段级别，在 4–64 GPU 规模下实现了高达 **1.85× 的吞吐提升**，为 LLM 后训练提供了一条全新的 **cost-efficient** 路径。

</details>

---

### 2. [ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers](https://arxiv.org/abs/2604.23798)

**Authors**: Chih-Chung Hsu, Xin-Di Ma, Wo-Ting Liao, Chia-Ming Lee  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 10.0  
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
Vision Transformers (ViT) 中的 Multi-Head Self-Attention (MHSA) 存在 **O(n²) 内存开销**，尤其在高分辨率图像（如 4K 图像）处理时，需要超过 4TB 的 FP32 显存来存储注意力矩阵，远超当前硬件能力。此外，在医疗影像、高光谱分析等对精度要求极高的领域，**FP32 精度不可妥协**。

现有加速方案存在以下缺陷：
- **近似注意力方法**（如 Performer、Linformer）：修改了 attention 操作本身，需重新训练模型，不适用于已训练好的 Foundation Models（如 CLIP、LLaMA）。
- **硬件融合内核**（如 FlashAttention-2/3）：依赖 Tensor Core（HMMA/GMMA），在 FP32 下无高效路径，且无法部署于边缘设备（如 Jetson TX2）。
- **内存优化实现**（如 ME-SDPA）：虽支持 FP32，但为串行计算，深度为 O(n)，导致长序列下延迟极高。

### 🚀 提出的新方法：ELSA
提出 **Exact Linear-Scan Attention (ELSA)**，一种算法级重构的在线 softmax 注意力机制，其核心思想是将在线 softmax 更新转化为一个**结合了状态三元组 (m, S, W) 的结合性幺半群 (associative monoid)** 上的前缀扫描（prefix scan）操作。

#### 创新点：
1. **保持精确 softmax 语义**  
   在实数运算中保留完全精确的 softmax 输出，并在 FP32 下提供可证明的相对误差界：**O(u log n)**，其中 u 是机器精度单位。
   
2. **并行化瓶颈突破**  
   将原本 O(n) 深度的串行计算转换为 **O(log n) 并行深度** 和 **O(n) 额外内存**，显著降低同步开销。

3. **硬件无关性 (Hardware-agnostic)**  
   不依赖 Tensor Core，使用 Triton 和 CUDA C++ 实现，可在从数据中心 GPU（A100）到嵌入式设备（Jetson TX2）上运行，是目前唯一能在全精度 FP32 下实现 O(log n) 深度的通用 exact-attention 内核。

4. **无需重训练 (Retraining-free)**  
   可作为预训练大模型（如 ViT、CLIP、LLaMA、VGGT）的即插即用替代品，无需任何权重调整或微调。

---

## 2. 核心实验方法和设置

### 📚 数据集与任务
- **合成序列基准测试**：长度从 64 到 16,384 tokens，用于评估注意力内核本身的延迟与内存。
- **视觉任务**：
  - ImageNet-1K 分类（ViT-B/16, Swin-T）
  - CLIP 零样本推理（ViT-L/14）
  - 超高光谱图像分类（Pavia, Salinas, WHU）使用 HSI-MAE
- **语言任务**：
  - BERT 情感分析（SST-2, IMDB）
  - LLaMA-13B 推理（host-device offloading 场景）
- **3D 视觉重建**：VGGT 和 FastVGGT 模型

### ⚙️ 实验设置
- **平台**：
  - 主要：NVIDIA A100 (40GB, CUDA 12.6, PyTorch 2.6)
  - 边缘设备：Jetson TX2
- **精度模式**：FP32（主）、FP16、TF32-Turbo
- **批大小**：多数实验使用 batch=8（ImageNet），部分为 batch=1（LLaMA offloading）
- **评估指标**：
  - 吞吐量（Throughput, img/s 或 M tok/s）
  - 延迟（Latency, ms）
  - 峰值显存（Peak VRAM, GB）
  - 速度提升（Speedup ×）
  - 数值误差（Drift metrics）

### 🔁 基线方法对比
仅比较 **exact、drop-in 替代** 的方法，排除需重训练的近似方法：
| 方法 | 是否 exact | 支持 FP32 | 是否硬件无关 | 是否需重训练 |
|------|-----------|------------|----------------|----------------|
| **PyTorch Math-SDPA** | ✅ | ✅ | ✅ | ✅ |
| **ME-SDPA (xFormers)** | ✅ | ✅ | ✅ | ✅ |
| **FlashAttention-2/3 (FA2/FA3)** | ❌（FP32 fallback） | ❌（无高效路径） | ❌（依赖 Tensor Core） | ✅ |
| **ELSA (Ours)** | ✅ | ✅ | ✅ | ✅ |

> 注：FA2/FA3 在 FP16 下表现优异，但在 FP32 下退化为未优化的 SIMD 路径，因此在 FP32 对比中不具竞争力。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 场景 | 指标 | ELSA 表现 | 对比基线 | 提升幅度 |
|------|------|----------|---------|---------|
| A100, FP32, 1K–16K tokens | 延迟 | 1.3–3.5× 快于 ME-SDPA | ME-SDPA | ↑ 30–250% |
| A100, BERT (SST-2/IMDB) | 速度 | 1.97× (SST-2), 2.27× (IMDB) | ME-SDPA | ↑ 97–127% |
| Jetson TX2, 64–900 tokens | 延迟 | ~1.5–1.6× 快于 Math kernel | Math-SDPA | ↓ 35–38% |
| LLaMA-13B offloading, ≥32K tokens | 吞吐量 | 17.8–20.2% 增益 | ME-SDPA | ↑ 近五分之一 |
| FP16, 16K tokens | 显存峰值 | 0.19 GB | FA2: 0.29 GB, FA3: 0.36 GB | ↓ 34–47% |
| ViT 全模型训练（ImageNet） | 吞吐量 | 最高达 1309 img/s (ViT-T) | FA2: 791 img/s | ↑ 65% |

### 🔍 详细对比结果

#### ✅ FP32 性能优势（长序列）
- 在 1K–16K tokens 下，ELSA 相比 ME-SDPA 实现 **1.3–3.5× 的延迟降低**。
- 在 BERT 情感任务中，达到 **1.97×～2.27× 速度提升**，且随序列增长增益更明显。
- 在 Jetson TX2 上，即使无 Tensor Core，仍实现 **约 1.5–1.6× 加速**，验证其硬件无关性。

#### ✅ FP16 性能接近硬件融合内核
- 在 FP16 下，ELSA 吞吐量接近 FA2/FA3，尤其在长序列（>16K）下差距进一步缩小。
- **峰值显存最低**：在 16K tokens 下仅为 **0.19 GB**，显著低于 FA2 (0.29 GB) 和 FA3 (0.36 GB)。

#### ✅ 下游任务全面领先
- **ImageNet-1K 全模型吞吐量**：在所有 ViT 架构上均优于 FA2，最高提速 **65%**（ViT-T）；在 Swin 架构上也优于 baseline。
- **CLIP 图像编码器**：在 ViT-L/14 @336px 下，ELSA-Turbo 实现 **1.062× 速度提升**。
- **LLaMA-13B offloading**：在 ≥32K tokens 时，因更低内存占用，计算可更早开始，**减少等待时间，提升端到端吞吐量 17.8–20.2%**。

#### 🔬 消融实验（Ablation Studies）
- **块大小 B=128** 为最优选择，平衡 occupancy 与共享内存使用。
- **累加器精度**：若使用 FP16 存储 (S, W)，在 >2K tokens 时数值误差急剧上升；FP32 累加器稳定。
- **变长序列测试**：ELSA 的零拷贝设计减少了 layout 转换开销，尽管 kernel 时间稍长，但端到端延迟与 FA2 相当。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **ELSA 是首个在 FP32 下实现 O(log n) 并行深度的 exact-attention 内核**，打破了传统在线 softmax 的串行瓶颈。
2. 通过将 softmax 状态建模为 **(m, S, W) 上的结合性幺半群**，使得 attention 可以用标准前缀扫描算法（Hillis-Steele + Blelloch）高效并行化。
3. **无需牺牲精度或重训练**，即可为 ViT、LLM 等模型带来显著加速，尤其在长序列、高精度场景下优势巨大。
4. **硬件无关设计使其可跨平台部署**，从 A100 到 Jetson TX2 均可运行，填补了边缘设备上的高性能 FP32 attention 空白。

### ⚠️ 局限性
1. **算术复杂度仍为 O(n²)**：ELSA 优化的是内存和 I/O，而非 FLOPs，因此在计算密集型短序列场景下可能不如高度优化的 FA2/FA3。
2. **短序列与窗口注意力增益有限**：在 Swin Transformer 等使用小窗口（n ≤ 196）的架构中，O(log n) 深度优势减弱，ELSA 仍优于 Math kernel，但不及 FA2。
3. **多 GPU 扩展性待研究**：前缀扫描引入跨块依赖，可能影响 sequence parallelism，需更复杂的流水线调度。
4. **当前验证集中在 Ampere 架构**：对 Hopper、Ada Lovelace 及非 NVIDIA 加速器的支持尚未充分验证。

### 🔮 未来工作方向
- 结合稀疏性或低秩结构以降低 FLOPs。
- 扩展至分布式多 GPU 训练场景，探索高效的并行化策略。
- 进一步拓宽硬件支持，覆盖更多 GPU 架构及非 NVIDIA 平台。
- 探索在生成式推理（decode phase）中的应用，尤其是长上下文生成。

---

> **代码开源地址**：[https://github.com/ming0531/ELSA](https://github.com/ming0531/ELSA)  
> **一句话总结**：ELSA 通过将 softmax attention 重构为可并行的前缀扫描操作，在不牺牲精度、无需重训练的前提下，实现了硬件无关、内存轻量、速度快的 exact attention，是高精度长序列 ViT 与 LLM 推理的理想选择。

</details>

---

### 3. [Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling](https://arxiv.org/abs/2604.24715)

**Authors**: Parsa Ashrafi Fashi, Utkarsh Saxena, Mehdi Rezagholizadeh, Aref Jafari, Akash Haridas, Mingyu Yang, Vansh Bhatia, Guihong Li, Vikram Appia, Emad Barsoum  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.24715v1  

#### Abstract
Hybrid sequence models that combine efficient Transformer components with linear sequence modeling blocks are a promising alternative to pure Transformers, but most are still pretrained from scratch and therefore fail to reuse existing Transformer checkpoints. We study upcycling as a practical path ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **Transformer-based LLMs** 虽然在多项任务上表现优异，但其训练成本极高，且推理时的 **KV-cache 内存开销** 随序列长度呈二次方增长，严重限制了长上下文（long-context）能力的应用。虽然已有研究提出将 Transformer 模型“升级”为混合架构（如 MambaInLlama、Zebra-Llama），以提升效率，但这些方法大多只关注**短上下文性能**，忽视了对长上下文能力的保留与增强。

本文旨在解决这一关键问题：**如何在不从头预训练的前提下，将已有的预训练 Transformer 模型高效地“升级”（upcycling）为兼具高性能和强长上下文能力的混合模型**。

### 提出的新方法：HyLo
作者提出了名为 **HyLo (HYbrid LOng-context)** 的新型升级方案，其核心是一个结合了架构设计、分阶段训练和教师引导蒸馏的完整流程。

#### 主要创新点：
1. **Long-context-aware 模型升级策略**  
   将“保持长上下文能力”作为升级的核心目标之一，而不仅仅是维持短上下文性能。这是对现有 upcycling 工作的重要补充。

2. **混合架构设计**  
   结合 **Multi-Head Latent Attention (MLA)** 和线性块（**Mamba2** 或 **Gated DeltaNet (GDN)**）构建混合模型。MLA 通过低秩压缩显著减少 KV-cache，而 Mamba2/GDN 则完全无 KV-cache，从而实现内存效率与建模能力的平衡。

3. **扩展的长上下文训练机制**  
   采用**分阶段训练**（staged training），将训练上下文长度从传统的 2K 扩展到 **64K tokens**，系统性地分析训练长度对长上下文泛化的影响。

4. **教师引导的长上下文蒸馏**  
   引入基于 **chunk-wise KL 散度监督** 的知识蒸馏，在长上下文场景下稳定优化过程，并显著提升长上下文性能。

5. **高吞吐推理服务集成**  
   将 HyLo 模型集成到 **vLLM** 推理引擎中，支持高达 **2M tokens** 的预填充（prefill）和解码（decoding），远超 Llama-3.2-3B 的 64K 上限。

### 相比现有方法的优势
- **无需从头预训练**：复用现有预训练模型参数，大幅降低计算成本。
- **更强的长上下文能力**：相比仅关注短上下文的 upcycling 方法（如 Zebra-Llama），HyLo 在 RULER 等长上下文基准上表现更优。
- **更高的内存效率**：KV-cache 内存减少 **>90%**，支持超长上下文推理。
- **通用性强**：在 Llama 和 Qwen 两种骨干模型上均验证有效，兼容 Mamba2 和 GDN 两种线性模块。

---

## 2. 核心实验方法和设置

### 数据集
- **短上下文评测**：使用 `lm-eval-harness` 中的多个常识推理任务：
  - ARC-Challenge (ARC), ARC-Easy (ARE)
  - HellaSwag (HS), OpenBookQA (OB)
  - PIQA, RACE (RA), Winograd (WG)
- **长上下文评测**：使用 **RULER** 基准中的全部 13 个任务，评估在 8K、16K、32K、64K 等不同上下文长度下的表现。
- **数学推理**：使用 **GSM8K** 数据集。

### 实验设置
- **骨干模型**：Llama-3.2-1B, Llama-3.2-3B, Qwen3-1.7B
- **训练方式**：
  - 分两个阶段：
    1. **Intermediate Layer Distillation (ILD)**：在 2K 上下文进行轻量级蒸馏。
    2. **长上下文 SFT**：在 8K 或 64K 上下文进行端到端微调，使用 KL 蒸馏损失。
- **硬件**：8× AMD MI300X GPU，使用 FSDP 进行分布式训练。
- **推理测试**：在 vLLM 上测试从 8K 到 2M 的 **TTFT (Time to First Token)** 和 **TPOT (Time Per Output Token)**。

### 基线方法对比
- **Upcycling 方法**：
  - MambaInLlama
  - Llamba
  - Zebra-Llama
  - M1
  - HypeNet
- **其他模型**：
  - Jet-Nemotron-2B（从头训练，400B tokens）
  - 原始 Llama/Qwen 模型

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **上下文扩展能力**：HyLo 可将可用上下文长度扩展至 **32×**（如从 64K 到 2M tokens）。
- **KV-cache 减少**：超过 **90%** 的内存节省。
- **推理规模**：支持在 8× MI300X 上进行 **2M-token** 的 prefill 和 decoding。

### 与基线方法的对比结果
| 模型 | RULER 平均 (64K) | GSM8K |
|------|------------------|--------|
| **HyLo-Qwen-1.7B** | **54.6%** | **76.0%** |
| Jet-Nemotron-2B | 37.5% | 73.3% |
| Zebra-Llama-3B | 60.7% | 47.8% |
| **HyLo-Llama-3B** | **71.0%** | **72.4%** |

> **关键发现**：尽管 Jet-Nemotron-2B 经过 400B tokens 的训练，HyLo-Qwen-1.7B 仅用 10B tokens 微调，却在 GSM8K 和 RULER 上全面超越。

### 消融实验结果
1. **训练上下文长度的影响**：
   - 在 64K 上训练的模型，其长上下文性能显著优于在 8K 训练后通过 YaRN 扩展的模型（图3）。
   - 证明了**直接长上下文训练**的有效性。

2. **知识蒸馏（KD）的作用**：
   - 使用 8B 教师模型进行蒸馏，可使 RULER-64K 性能提升 **22%**。
   - 蒸馏对长上下文性能的增益远大于对短上下文的增益。

3. **Enhanced-ILD 损失的有效性**（表6）：
   - 引入 token-mixer 输出的对齐损失后，GSM8K 性能提升 **+6.3%**（从 37.2 → 43.5）。
   - 表明中间层表示对齐对数学推理能力有显著帮助。

4. **架构设计选择**：
   - 移除位置编码（NoPE）或引入注意力门控（Gated Attention）在 upcycling 场景下**未带来收益**，说明这些技术依赖于预训练阶段的配合。

---

## 4. 关键结论和发现

### 主要发现
1. **长上下文感知的升级是可行且高效的**：HyLo 成功将预训练 Transformer 模型转化为兼具强短上下文和卓越长上下文能力的混合模型。
2. **直接长上下文训练优于后处理扩展**：在 64K 上训练比在 8K 训练后使用 YaRN 扩展效果更好。
3. **教师引导蒸馏对长上下文至关重要**：KL 蒸馏显著提升了长上下文泛化能力，尤其是在大教师模型下效果更明显。
4. **HyLo 具备部署优势**：通过 vLLM 集成，实现了高达 2M tokens 的高效推理，解决了实际应用中的内存瓶颈。

### 方法的局限性
- 当前升级仍需一定量的长上下文数据（约 10B tokens），尚未完全消除数据依赖。
- 对教师模型的大小和质量有一定依赖，小教师模型增益有限。
- 架构设计（如 MLA 层数）需要手动配置，缺乏自动化搜索机制。

### 未来工作方向
- 进一步缩小长上下文长度上的性能差距。
- 提升蒸馏效率，减少对大规模教师模型的依赖。
- 将该框架扩展到更多下游任务，尤其是需要鲁棒长上下文推理的场景（如代码生成、法律文档理解等）。

</details>

---

### 4. [SDSL-Solver: Scalable Distributed Sparse Linear Solvers for Large-Scale Interior Point Methods](https://arxiv.org/abs/2604.23979)

**Authors**: Shaofeng Yang, Yunting Wang, Yingying Cheng, Fan Zhang, Xin He, Guangming Tan  
**Category**: cs.DC  
**Published**: 2026-04-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.23979v1  

#### Abstract
The solution of sparse linear systems constitutes the dominant computational bottleneck in interior point methods (IPMs), frequently consuming over 70\% of the total solution time. As optimization problems scale to millions of variables, direct solvers encounter prohibitive fill-in, excessive memory...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*SDSL-Solver: Scalable Distributed Sparse Linear Solvers for Large-Scale Interior Point Methods*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在大规模优化问题中，**Interior Point Methods (IPMs)** 是求解线性规划、二次规划和凸优化问题的核心算法。然而，其计算瓶颈在于每轮迭代中需要求解一个大型稀疏线性系统 $Ax = b$，该步骤通常占据总时间的 **70%以上**。

随着问题规模扩展至百万级变量，传统方法面临以下挑战：
- **Direct solvers**（如 PARDISO）因矩阵填充（fill-in）导致内存爆炸且并行扩展性差；
- **分布式迭代方法**（如 PETSc + Block Jacobi）虽具可扩展性，但在病态（ill-conditioned）系统上收敛缓慢甚至失败；
- 缺乏对 IPM 迭代间系数矩阵变化缓慢特性的有效利用。

---

### 提出的新方法与创新思路

作者提出 **SDSL-Solver** —— 一种面向 IPMs 的可扩展分布式稀疏线性求解器框架，包含四大核心技术：

#### （1）双模式自适应分布式架构
设计两种互补的分布式并行策略，并根据运行时诊断自动切换：
- **Block Jacobi (BJ)**：适用于对角占优、良态系统，具有高并行度和低通信开销。
- **Bordered Block Diagonal (BBD)**：针对病态系统，通过 Schur complement 技术保留子域间的全局耦合信息，显著提升预条件质量。

> ✅ 创新点：首次将 BBD 分解用于 IPMs 中的分布式预条件构造，兼顾精度与可扩展性。

#### （2）基于数值分布的稀疏过滤算法（Numerics-based Sparse Filtering）
观察到 IPM 系数矩阵中大量非对角元素绝对值远小于对应行列对角元，提出一种阈值过滤机制：
- 若 $|a_{ij}| < \tau \cdot |a_{ii}|$ 且 $|a_{ij}| < \tau \cdot |a_{jj}|$，则丢弃该非对角项；
- 使用完全 Cholesky 因子化处理滤波后矩阵以减少精度损失。

> ✅ 创新点：结合代数结构与数值特性构建高效预条件器，在保证收敛的同时大幅降低因子化成本。

#### （3）对角修正技术（Diagonal Correction）
为应对 IPM 后期极端病态问题（condition number 可增长多个数量级），引入小常数 $\delta > 0$ 对预条件器对角元进行增强：
$$
p_{ii} := p_{ii} + \delta \cdot \text{sign}(p_{ii})
$$
仅作用于预条件器，不影响原始方程解的准确性。

> ✅ 创新点：不修改原矩阵的前提下改善谱性质，显著提高迭代法鲁棒性。

#### （4）预条件器重用策略（Preconditioner Reuse）
利用连续 IPM 迭代中矩阵变化缓慢的特点，复用符号分解（symbolic factorization）甚至数值因子，摊销昂贵的预条件器构建开销。

> ✅ 创新点：实现跨迭代优化，尤其在中期迭代中效果显著。

---

### 相比现有方法的优势

| 维度 | SDSL-Solver | PETSc (Block Jacobi) | PARDISO (Direct) |
|------|-------------|------------------------|------------------|
| 并行扩展性 | 高（支持多节点 MPI+OpenMP） | 高 | 低（共享内存为主） |
| 内存效率 | 高（稀疏迭代 + filtering） | 高 | 极低（fill-in 严重） |
| 数值鲁棒性 | 强（支持 diagonal correction） | 弱（易发散） | 强但受限于内存 |
| 收敛速度 | 快（高质量 preconditioning） | 慢（忽略块间耦合） | 精确但慢 |
| 自适应能力 | 支持 BJ/BBD 动态切换 | 固定 | 不适用 |

---

## 2. 核心实验方法和设置

### 数据集
共测试两类基准问题，涵盖从十万到五百万维的大规模实例：

#### （1）Block Jacobi 类（良态、对角占优）
- 来源：PageRank 基准、Hans Mittelmann 基准
- 示例：`PageRank_1m`, `PageRank_5m`, `com-youtube`, `cit-patents`, `L2CTA3D`, `thk_48`, `thk_63`

#### （2）BBD 类（病态、需强预条件）
- 来源：华为网络规划数据、SCUC 安全校核问题、MIPLIB 2017、Hans Mittelmann
- 示例：`NetworkPlan_*`, `ScucRelax_*`, `BuildingEnergy`, `L1_sixm250obs`

> 所有矩阵维度 $n$ 范围：约 40K ~ 7.6M；nnz 最高达 42M+

---

### 实验平台

| 平台 | 节点数 | CPU | 每节点核心数 | 内存 |
|------|-------|-----|--------------|------|
| X86 Cluster | 4 | 2.6GHz, 52 cores/node | 4×13=52 | 730GB |
| Kunpeng Cluster | 4 | Kunpeng-920, 2.6GHz | 4×32=128 | 521GB |

---

### 评估指标

- **单次求解时间**（single-solve wall-clock time）
- **端到端 IPM 总耗时**（end-to-end solution time）
- **迭代次数**（Krylov 迭代数）
- **加速比**（speedup vs. baseline）
- **收敛稳定性**（是否出现 numerical issues）

---

### 基线方法对比

| 基线 | 描述 |
|------|------|
| **PETSc** | 分布式迭代框架，采用 Row-wise 分区 + Block Jacobi + ILU(0)，作为多节点对比基线 |
| **PARDISO (MKL)** | 共享内存直接求解器，默认使用 LDLT 或 LU 分解，作为单节点性能与鲁棒性基线 |

> 所有实验均集成于华为 **OptVerse** IPM 求解器中，替换其内部线性求解模块，保持其余流程一致。

---

## 3. 主要实验结果和性能指标

### 单节点性能（Single-node, X86）

#### （1）稀疏过滤（Sparse Filtering）效果（L2CTA3D）
- 应用 $\tau=10^{-3}\sim10^{-8}$ 动态过滤，预条件器非零元减少至原始的 **4%-26%**
- 相比 PARDISO LDLT（无预处理）：**总时间从 916s 降至 377s，提速 2.43×**

#### （2）对角修正（Diagonal Correction）鲁棒性
- 在全部 15 个病态问题上，PARDISO 均报 “Numerical Issues Detected” 导致失败；
- SDSL-Solver（$\delta=10^{-12}$）全部成功收敛至最优解。

#### （3）预条件器重用（Preconditioner Reuse）
| 问题 | 无重用时间 | 有重用时间 | 加速比 |
|------|------------|------------|--------|
| `NetworkPlan_6` | 441.32s | 125.3s | **3.53×** |
| `L1_sixm250obs` | 27.94s | 20.70s | **1.35×** |

> 注：重用还带来隐式正则化效应，使后者迭代步数从 53 减少到 28。

---

### 多节点性能（4-node, X86）

#### （1）Block Jacobi 模式 vs. PETSc 和 PARDISO
| 指标 | vs. PETSc | vs. PARDISO |
|------|----------|-------------|
| **平均加速比** | **6.23×** | **97.54×** |
| 最大加速比 | 53.27× (`com-youtube`) | >300× |
| 特殊情况 | PARDISO 在多个大问题上因 fill-in 超出 32-bit 整数索引而崩溃 |

> ✅ 表明 SDSL-Solver 在大规模良态问题上兼具高性能与强扩展性。

#### （2）BBD 模式 vs. PETSc 和 PARDISO
| 指标 | vs. PETSc | vs. PARDISO |
|------|----------|-------------|
| **平均加速比** | **7.77×** | **5.85×** |
| 最大加速比 | 14.39× (`net2`) | 12.23× (`net10`) |

> 尽管 BBD 中使用完整 LU 分解更耗时，但因其极低的 IGCR 迭代次数（常为 2–3 步），总体仍优于基线。

---

### 端到端 IPM 性能（End-to-End）

| 问题 | X86 时间 (s) | Kunpeng 时间 (s) |
|------|---------------|------------------|
| `pg1m` | 316.8 | 523.5 |
| `citp` | 1,566.3 | 2,210.4 |
| `comy` | 2,351.0 | 5,216.1 |
| `pg5m` | 4,088.0 | 9,989.0 |
| `L2C` | 176.0 | 382.6 |

> 在 X86 上相比单节点 PARDISO 实现 **1.32× ~ 9.09×** 的端到端加速。

> ⚠️ 注意：Kunpeng 上部分问题未能收敛，归因于浮点舍入误差积累加剧了病态系统的不稳定性。

---

### 消融实验（Ablation Study）
虽然未明确列出消融表格，但从各节分析可见：
- **稀疏过滤** → 显著降低预条件器构建时间（最高达数十倍）
- **对角修正** → 是解决病态问题不可替代的关键组件
- **预条件器重用** → 在中后期迭代中提供额外 1.35×~3.53× 加速
- **BBD 架构** → 将 PETSc 的数百次 BiCGSTAB 迭代压缩至 2–3 次 IGCR 迭代

---

## 4. 关键结论和发现

### 主要发现

1. **IPM 中的线性求解瓶颈可通过分布式迭代方法有效突破**，尤其是结合 Krylov 方法与高质量预条件器的设计。
2. **数值感知的预条件器优化（filtering + diagonal correction）比纯结构方法更有效**，能同时提升效率与鲁棒性。
3. **BBD 分解是一种高效的全局耦合并行策略**，特别适合处理病态、弱对角占优系统。
4. **预条件器重用不仅是性能优化手段，还可起到隐式正则化作用**，有助于稳定晚期迭代。
5. **混合编程模型（MPI+OpenMP）在真实集群中表现优越**，能灵活适配不同资源配置。

---

### 方法的局限性

1. **依赖图划分工具（如 METIS/ParMETIS）的质量**，影响 BBD 接口大小与负载均衡；
2. **Schur complement 求解可能成为瓶颈**，当接口过大时根节点负担重；
3. **对超大规模 GPU/NPU 架构支持不足**，当前主要面向 CPU 集群；
4. **Kunpeng 平台上的数值稳定性问题暴露了跨平台移植挑战**，尤其在长序列迭代中累积误差的影响；
5. **参数调优仍需人工干预**（如 filtering threshold $\tau$, correction $\delta$）。

---

### 未来工作方向（原文提及）

1. **GPU/NPU 加速**：对 SpMV 和预条件器应用内核进行异构加速；
2. **自适应阈值选择**：实现 $\tau$ 和 $\delta$ 的动态调整，减少手动调参；
3. **扩展至二阶锥规划（SOCP）与半定规划（SDP）**，覆盖更广的凸优化场景。

---

## 总结

✅ **SDSL-Solver 成功实现了 IPMs 中稀疏线性求解的“可扩展性 + 鲁棒性 + 高效性”三重目标**，是目前少数能在千万级变量问题上稳定运行且显著超越主流求解器的分布式迭代方案。

🔹 其核心思想——**“数值感知预条件 + 自适应并行分解 + 跨迭代优化”**——为下一代工业级优化求解器提供了重要参考路径。

</details>

---

### 5. [Unfolding an Atomistic World: Atomistic Simulation of Reactor Pressure Vessel Steel Across Year-and-Meter Scales](https://arxiv.org/abs/2604.24091)

**Authors**: Haozhi Han, Ruge Zhang, Haoquan Chen, Yifeng Chen, Haipeng Jia, Liang Yuan, Yunquan Zhang, Ting Cao, Yunxin Liu, Ya-Qin Zhang, Kun Li  
**Category**: cs.DC  
**Published**: 2026-04-28  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.24091v1  

#### Abstract
Lifetime prediction of reactor pressure vessel (RPV) steel requires bridging atomistic degradation mechanisms with service-scale spatial and temporal regimes, from Angstroms and picoseconds to meters and decades. Existing engineering-scale models provide long-range reach but rely on fitted degradati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Unfolding an Atomistic World: Atomistic Simulation of Reactor Pressure Vessel Steel Across Year-and-Meter Scales**

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**
核反应堆压力容器（Reactor Pressure Vessel, RPV）钢在长期辐照、高温和高压环境下会发生不可逆的脆化（embrittlement），从而影响核电站的安全运行寿命。准确预测其服役寿命是核能系统中的核心挑战。

该问题的根本难点在于**极端多尺度性**：
- **空间尺度**：从原子级缺陷（Ångstrom）到米级压力容器结构（meter）。
- **时间尺度**：从皮秒（picosecond）的原子跃迁事件到数十年（decades）的服役周期。

传统方法无法同时兼顾：
- **工程尺度模型**（如 rate theory, cluster dynamics）虽能覆盖宏观时空，但依赖经验拟合，牺牲了原子级机理的真实性。
- **原子级模拟方法**（如 AKMC）虽能精确捕捉微观机制，但受限于计算效率，难以扩展到年-米尺度。

### **提出了什么新方法或新思路**
本文提出 **AtomWorld** —— 一种面向 RPV 钢寿命预测的**原子世界建模框架**（atomistic world-modeling framework），通过算法、高性能计算（HPC）与应用三者协同设计，首次实现了跨越“年-米”尺度的原子级模拟。

#### 核心创新分为三个层面：

| 层面 | 创新内容 |
|------|--------|
| **Algorithmic Innovation** | 将经典 AKMC 重构为基于 **Reinforcement Learning** 的原子世界模型：<br>- 使用 **Actor-Critic 架构**（PPO）学习具有长期动力学感知的状态转移。<br>- 引入 **Consequence-aware 决策机制**，避免陷入局部可逆循环（super-basin trapping）。<br>- 采用 **Poisson-based 物理时间对齐**，恢复正确的 AKMC 时间语义。 |
| **HPC Innovation** | 面向现代超算架构进行协同优化：<br>- **Compute-Centric Reformulation**：将不规则的事件选择转化为矩阵运算（GEMM/GEMV），适配 GPU/CPU 加速器。<br>- **Asynchronous Sublattice Parallelism**：消除全局同步瓶颈，提升并行效率。<br>- **Shift Communication Strategy**：将全邻域通信降为维度级流水通信，显著降低通信开销。 |
| **Application Innovation** | 提出 **Mesoscopic Voxel-Parallel Framework**：<br>- 将整块 RPV 分解为百万级物理一致的 **mesoscopic voxel**（代表体元），每个 voxel 独立执行原子模拟。<br>- 基于温度梯度进行 **Temperature-Guided Discretization**，确保各 voxel 近似等温。<br>- 使用 **Dynamic Voxel Scheduling** 动态调度异构负载，提高资源利用率。 |

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **时空可达性** | 首次实现 **year-and-meter scale** 的原子级模拟，突破传统 AKMC 的“30年墙”。 |
| **保真度** | 保留了原子级动力学因果链，无需依赖经验公式或粗粒化假设。 |
| **可扩展性** | 支持千万亿至十万亿亿原子系统，在多种超算平台（CPU/GPU）上实现高效扩展。 |
| **实用性** | 可用于实际 RPV 结构件（如 CAP1400）的全尺寸寿命预测。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**
- **训练数据**：来自小规模 AKMC 模拟生成的原子跃迁轨迹，涵盖广泛条件：
  - 温度范围：230–400°C
  - 合金成分：Cu (0.02–0.26 at.%), Ni (0.38–1.54 at.%), Mn, Si, P 等
  - 缺陷浓度：1–1000 appm
  - 中子通量：10⁹–10¹¹ n/cm²/s
  - 辐照剂量：10⁻⁴–1 dpa
- 输入特征包括：原子类型、相对坐标、局部缺陷状态、邻接关系、候选跃迁掩码。

### **实验设置**
- **目标材料**：中国第三代核电站 CAP1400 的 RPV 钢，ASME SA508 Grade 3 Class 1。
- **几何建模**：
  - 完整壁厚：0.23 m
  - 轴向高度：12.64 m
  - 总体积：9.91 cm³
  - 使用 **2,200,000 个 mesoscopic voxel** 表示
- **单个 voxel 规模**：最大达 15,000³ 原子（约 6.75 万亿原子/voxel）
- **总原子数**：高达 **14.85 quintillion atoms**（1.485×10¹⁹）

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Accuracy** | 与参考 AKMC 和实验数据对比微结构演化路径（如 advancement factor） |
| **Scalability** | 强/弱扩展效率（strong/weak scaling efficiency） |
| **Peak Performance** | 实测浮点峰值性能（EFLOP/s） |
| **Time-to-Solution** | 模拟一个服务年所需的墙钟时间（wall-clock time） |
| **Correctness** | 是否保持原始 AKMC 的动力学行为和热力学演化一致性 |

### **基线方法对比**
- **State-of-the-art AKMC 方法**：
  - OpenKMC
  - MISA-AKMC
  - TensorKMC
- 主要差距体现在：
  - 最大仅支持 quadrillion-atom（10¹⁵）级别
  - 模拟一年需约 **30 年墙钟时间**
  - 空间覆盖不足 10⁵ μm³，远小于工程需求

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 指标 | AtomWorld 结果 | 对比（State-of-the-art） | 提升倍数 |
|------|----------------|--------------------------|---------|
| **最大系统规模** | 14.85 quintillion atoms (1.485×10¹⁹) | ~10¹⁵ atoms | **~1,000× 更大** |
| **空间覆盖** | 9.91 cm³（0.23 m × 12.64 m） | < 0.001 cm³ | **>10⁷× 更大** |
| **时间推进速度** | 1.71 天模拟 1 服务年 | ~30 年模拟 1 服务年 | **~6,400× 加速** |
| **全生命周期预测** | 60 年寿命可在 **103 天内完成** | 不可行 | 首次实现 |
| **峰值性能** | **1.27 EFLOP/s**（Lineshine 上） | < 0.1 EFLOP/s | 当前最高纪录 |
| **FP64 利用率** | 达到 Lineshine 峰值性能的 **48%** | 典型 HPC 应用 < 20% | 显著领先 |
| **强扩展效率** | 85–97%（跨 5 大超算） | 通常 < 80% | 接近理想 |
| **弱扩展效率** | 88–97% | 通常 < 85% | 极高可扩展性 |

### **与基线方法的对比结果**
- 在相同物理条件下，AtomWorld 与参考 AKMC 的 **advancement factor 曲线高度一致**（见 Fig. 4），验证了其动力学保真性。
- 相比 OpenKMC，在 L=6400 的晶格上，AtomWorld 实现 **452.3× 的加速比**（Fig. 3）。
- 在 **5 大领导级超算**（Lineshine, Tianhe-3, New Sunway, ORISE, Tecorigin）上均实现高效运行，证明其硬件普适性。

### **消融实验结果（Ablation Study）**
虽然文中未明确列出“消融实验”章节，但从设计分析中可推断以下关键组件的作用：
- **Local Atomic Policies + Global Critic**：若无 critic 的 long-horizon guidance，策略会陷入短程循环，导致 super-basin trapping。
- **Poisson Time Alignment**：若直接使用 softmax 时间采样，将破坏 AKMC 的时间统计特性。
- **Shift Communication**：相比 all-neighbor exchange，通信消息数减少 **~6×**，大幅缓解通信瓶颈。
- **Voxel-Parallelism**：使问题具备天然的 embarrassingly parallel 结构，支撑百万级任务并发。

---

## 4. 关键结论和发现

### **主要发现**
1. ✅ **首次实现原子级模拟跨越“年-米”尺度**：AtomWorld 成功将原子级动力学推进至工程可用的时间与空间尺度，打破了长期以来的计算壁垒。
2. ✅ **AI 与 HPC 协同设计的巨大潜力**：通过将 RL 引入物理演化过程，并与超算架构深度耦合，实现了科学模拟范式的转变。
3. ✅ **voxel-parallel 是连接微观与宏观的有效桥梁**：无需全原子重建即可获得统计意义上真实的宏观退化行为。
4. ✅ **AtomWorld 在多个超算平台上表现出卓越的扩展性和性能**，峰值达到 **1.27 EFLOP/s**，接近硬件极限。

### **方法的局限性**
- **依赖高质量 ab initio 能量景观**：当前模型仍需预先构建或在线计算迁移势垒，尚未完全端到端学习能量函数。
- **初始缺陷分布依赖工程假设**：如 sink density、初始 vacancy concentration 等仍需外部输入。
- **目前聚焦于静态热-辐照场**：未考虑随时间演化的应力场或冷却剂化学变化。
- **训练成本较高**：尽管推理零样本可扩展，但训练仍需大量 AKMC 轨迹。

### **未来工作方向**
- 扩展至其他核材料系统（如燃料包壳、堆内构件）。
- 耦合机械载荷与裂纹扩展模型，实现从脆化到断裂的全流程预测。
- 引入 **end-to-end energy learning**，结合 **MLIP**（Machine Learned Interatomic Potentials）进一步减少对第一性原理计算的依赖。
- 探索 **real-time adaptive voxel refinement**，在关键区域动态加密模拟分辨率。
- 推动该框架成为下一代数字孪生核电站的核心引擎之一。

---

> **总结一句话**：  
> **AtomWorld 通过“算法-算力-应用”三位一体的协同创新，首次实现了 RPV 钢在原子精度下的全尺度服役寿命模拟，标志着原子级科学模拟进入“预测性工程”新时代。**

</details>

---

### 6. [GreenDyGNN: Runtime-Adaptive Energy-Efficient Communication for Distributed GNN Training](https://arxiv.org/abs/2604.23139)

**Authors**: Arefin Niam, Tevfik Kosar, M. S. Q. Zulkar Nine  
**Category**: cs.DC  
**Published**: 2026-04-28  
**Score**: 8.0  
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
分布式 GNN 训练中的主要瓶颈是跨分区的 **remote feature fetching**，即在多跳邻居采样过程中频繁触发细粒度的远程过程调用（RPC）。这些操作带来两个关键问题：
- **高固定启动开销（initiation overhead）**：每个 RPC 都有固定的 CPU 和网络协议处理成本，对小批量请求尤其不经济。
- **GPU stall 能耗浪费**：当 GPU 等待远程特征时，仍以接近空闲功耗运行，造成大量能量浪费。

现有系统（如 RapidGNN）采用静态缓存策略（epoch-level caching），无法应对运行时网络拥塞变化，导致能效下降。

> 🔍 **核心洞察**：静态缓存策略在动态网络环境下表现脆弱，即使轻微的链路延迟波动也会使最优缓存窗口偏移，增加高达 45% 的能耗。

---

### 🚀 提出的新方法与创新思路

GreenDyGNN 提出将缓存管理建模为一个 **runtime-adaptive 控制问题**，并引入以下关键技术：

#### （1）**细粒度窗口化缓存重建（Fine-grained Window-based Cache Rebuilding）**
- 不再仅在一个 epoch 开始时重建一次缓存，而是在训练过程中按“窗口”（每 $ W $ 步）动态重建。
- 每个窗口内预取未来 $ W $ 个 mini-batch 所需的热点远程节点特征，摊销 RPC 启动成本。

#### （2）**基于强化学习的自适应控制（Double-DQN Agent）**
- 将缓存窗口大小 $ W $ 和各分区所有者（owner）的缓存分配联合决策建模为 **Markov Decision Process (MDP)**。
- 使用 **Double-DQN** 算法训练轻量级 RL agent，在模拟器中通过 **domain randomization** 学习应对各种拥塞模式。
- 决策输入包括：各链路延迟估计、缓存命中率、系统负载等实时状态。

#### （3）**异步双缓冲流水线（Asynchronous Double-Buffered Pipeline）**
- 实现采样、缓存构建、特征解析与 GPU 训练完全解耦。
- 缓存重建在后台进行，不影响主训练流程，实现 **零感知适应开销（effectively overhead-free adaptation）**。

#### （4）**Sim-to-Real Transfer 设计**
- 在真实集群上采集通信与能耗参数，构建校准后的轻量级模拟器（calibrated simulator）。
- RL 训练在模拟器中完成（约 20 分钟），避免昂贵的在线训练。
- 域随机化（domain randomization）提升策略泛化能力，部署后性能损失小于 4%。

---

### ⚖️ 相比现有方法的优势

| 方面 | 现有方法（如 RapidGNN） | GreenDyGNN |
|------|--------------------------|-----------|
| 缓存策略 | 静态、epoch 级别 | 动态、window 级别 |
| 对网络变化响应 | 无 | 实时感知与调整 |
| 控制机制 | 固定规则或离线优化 | 强化学习驱动的联合决策 |
| 能效表现 | 在稳定网络下良好 | 在动态拥塞下显著更优 |
| 适应开销 | 低（但不灵活） | 几乎为零（得益于异步管道） |

> ✅ GreenDyGNN 成功实现了“**clean condition 下媲美最优静态策略，congested condition 下大幅超越**”的目标。

---

## 2. 核心实验方法和设置

### 📊 数据集
- **OGBN-Products**（2.4M 节点，61.9M 边）
- **Reddit**（233K 节点，114M 边）
- **OGBN-Papers100M**（111M 节点，1.6B 边）

均使用 METIS 进行 4 分区划分，部署于 4 个节点。

---

### 💻 实验平台
- **硬件环境**：Chameleon Cloud 上的 4 节点集群
  - 每节点：Intel Xeon CPU，2× NVIDIA P100 GPU，25 Gbps Ethernet
- **软件栈**：基于 DGL + PyTorch DDP 构建
- **模型**：2-layer GraphSAGE，fan-out = {10, 25}
- **训练配置**：30 epochs，batch size ∈ {1000, 2000, 3000}

---

### 🌪️ 拥塞注入设置
- 使用 `tc netem` 在 DGL RPC 端口（30050）注入延迟
- 模拟 **时间可变拥塞（time-varying congestion）**：
  - 清洁阶段（epochs 0–2）
  - 拥塞阶段（epochs 3–9）：单/双链路延迟 15–25ms，周期性切换
  - 最终清洁 epoch
- Gradient sync 流量不受影响（走独立端口）

---

### 📈 评估指标
| 指标类别 | 具体指标 |
|--------|---------|
| **能效** | 总能耗（GPU + CPU，全节点总和，单位 kJ） |
| **性能** | 平均 epoch 时间（ET, s）、累计 wall-clock time |
| **缓存行为** | 缓存命中率、重建频率、远程 fetch 数量 |
| **收敛性** | Accuracy vs. Wall Time 曲线 |
| **公平比较** | 各方法自身 clean baseline 的能耗增长百分比 |

---

### 🆚 基线方法对比
| 方法 | 描述 |
|------|------|
| **Default DGL** | Vanilla 分布式 GNN，按需远程 fetch |
| **BGL** | 多层缓存 + I/O 优化，支持 prefetch-during-sampling |
| **RapidGNN** | Epoch-level 静态热点缓存（当前最优静态策略） |
| **GreenDyGNN (ours)** | 本文提出的方法，含 RL 控制器与动态缓存重建 |

---

## 3. 主要实验结果和性能指标

### 📉 关键性能数据（$ B=2000 $，拥塞条件下）

| Dataset | GreenDyGNN 总能耗 (kJ) | vs. Default DGL ↓ | vs. RapidGNN ↓ |
|--------|------------------------|------------------|---------------|
| OGBN-Products | 203.9 | **28%↓** | **4.8%↓** |
| Reddit | 189.4 | **42%↓** | **21.2%↓** |
| OGBN-Papers100M | 307.2 | **32%↓** | **19.0%↓** |

> ✅ 在全部 9 种配置中（3 数据集 × 3 batch sizes），GreenDyGNN 在 **8/9 场景下取得最低总能耗**，且在所有场景中拥有最快 epoch 时间。

---

### 🔍 与基线方法对比亮点

#### （1）**显著降低拥塞带来的额外能耗**
- 图 5 显示，在拥塞下各方法相对于自身 clean 条件的能耗增幅：
  - Default DGL / BGL：+30% ~ +50%
  - RapidGNN：+18% ~ +45%
  - **GreenDyGNN：仅 +5% ~ +19%**
- 在 Reddit 和 Papers100M 上，GreenDyGNN **吸收了多达 26 个百分点的拥塞开销**，这是静态缓存无法做到的。

#### （2）**CPU 能耗大幅下降是主因**
- GPU 能耗差异较小（两者都减少 stall），但 **CPU 能耗差距明显**：
  - 例如 Reddit ($B=2000$):  
    - RapidGNN CPU: 230.0 kJ  
    - GreenDyGNN CPU: **180.6 kJ**（↓21%）
- 原因：GreenDyGNN 更智能地减少了昂贵的远程 fetch 次数和成本。

#### （3）**加速模型收敛**
- 图 10 显示，GreenDyGNN 在相同 wall-clock time 内达到更高 accuracy。
- 收敛速度优势源于更低的 epoch 时间和更少的通信阻塞。

---

### 🔬 消融实验结果（Ablation Study）

| 变体 | 描述 | OGBN-Papers100M 能耗 (kJ) | 相对增耗 |
|------|------|----------------------------|---------|
| **GreenDyGNN (full)** | 完整版本（RL + cost-weighted allocation） | **307.2** | — |
| w/o Cost Weights | 仅启用 RL 调整 $ W $，但缓存均匀分配 | 318.0 | +3.4% |
| w/o RL (static $ W=16 $) | 固定窗口，无 RL 控制 | 336.1 | **+8.6%** |

> ✅ 结论：
- **RL 控制器起主导作用**（节省 ~7–8%），它能根据拥塞动态缩短 $ W $，提高缓存新鲜度。
- **Per-owner cost weighting 提供补充收益**（额外节省 ~3%），将更多缓存资源导向高延迟链路。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Runtime 网络变化是影响分布式 GNN 能效的一等公民问题**  
   即使是几毫秒的链路延迟波动，也会显著改变最优缓存策略，静态方法难以适应。

2. **缓存窗口大小应作为 runtime 控制变量而非超参**  
   GreenDyGNN 验证了将 $ W $ 和 cache allocation 联合优化的有效性。

3. **Sim-to-Real + Domain Randomization 是可行路径**  
   在轻量模拟器中训练的 RL 策略可在真实集群上高效部署，无需在线训练。

4. **异步双缓冲架构使 adaptation 几乎无代价**  
   保证了 RL 决策不会成为性能瓶颈。

5. **GreenDyGNN 在 clean 和 congested 条件下均表现优异**
   - 拥塞下：节能 **最高达 43% vs. DGL，领先 RapidGNN 4–24%**
   - 无拥塞下：与最佳静态策略差距 <2%，无额外惩罚

---

### ⚠️ 局限性

1. **依赖 METIS 分区结果**  
   未解决图划分本身的问题，而是假设已存在固定分区。

2. **动作空间有限**  
   当前 RL 动作仅为离散的 $ W \in \{1,2,...,128\} $ 和简单偏向策略，尚未探索连续或更复杂调度。

3. **未整合硬件级节能手段**  
   如 GPU frequency scaling 或 DVFS，仍有进一步协同优化空间。

4. **目前仅验证于同构环境**  
   对混合 GPU、RDMA 等异构场景的支持有待扩展。

---

### 🔮 未来工作方向

1. **扩展至异构硬件环境**  
   支持混合 GPU 类型、RDMA 加速通信下的联合控制。

2. **加入 GPU 功耗调节作为动作维度**  
   实现 compute-side 与 communication-side 的端到端能效协同优化。

3. **在线微调模拟器参数**  
   利用生产环境 telemetry 数据持续更新 cost model，提升长期鲁棒性。

4. **探索更复杂的拥塞预测机制**  
   结合历史趋势与拓扑信息，提前预判拥塞模式。

---

> 📌 **一句话总结**：  
> **GreenDyGNN 首次将 runtime 自适应思想引入分布式 GNN 缓存控制，利用 RL + sim-to-real + 异步流水线，在动态网络中实现了高达 43% 的节能，并全面超越所有静态策略，同时保持 clean 条件下的最优性能。**

</details>

---

### 7. [Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](https://arxiv.org/abs/2604.23623)

**Authors**: Zichuan Fu, Xian Wu, Guojing Li, Yejing Wang, Yijun Chen, Zihao Zhao, Yixuan Luo, Hanyu Yan, Yefeng Zheng, Xiangyu Zhao  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23623v1  

#### Abstract
Recent advancements in large language models (LLMs) have catalyzed the rise of reasoning-intensive inference paradigms, where models perform explicit step-by-step reasoning before generating final answers. While such approaches improve answer quality and interpretability, they incur substantial comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Large Language Models (LLMs)** 在执行复杂任务（如数学推理、代码生成）时，通常采用“思维链”（Chain-of-Thought, CoT）或“显式推理”（thinking paradigm）模式，虽然提升了准确性和可解释性，但带来了巨大的 **计算开销**。这类模型生成的推理链往往长达数千 tokens，显著增加推理延迟和部署成本。

此外，已有优化方法（如强化学习微调 RFT）存在以下限制：
- 需要对 LLM 进行持续训练，可能损害其通用能力；
- 不适用于仅提供 API 接口的闭源模型。

因此，如何在保留高质量推理的同时大幅降低计算成本，成为一个关键挑战。

### 提出的新方法与思路
本文提出 **Tandem**，一种新颖的 **LLM-SLM 协作框架**，将大型语言模型（LLM）作为“导师”（mentor），小型语言模型（SLM）作为“实习生”（intern），实现高效协同推理。

#### 核心机制：
- **分阶段推理洞察提取**：LLM 仅生成轻量级的四类 **Thinking Insights**：
  1. **Goal**：明确问题目标；
  2. **Planning**：制定高层策略；
  3. **Retrieval**：召回相关知识；
  4. **Action**：执行关键逻辑步骤。
- **SLM 完成最终推理**：SLM 利用这些结构化指导完成详细推理并输出答案。
- **代价感知的连续判断机制（Cost-aware Continual Judgment）**：
  - SLM 基于自身输出的 **Perplexity** 和 **Entropy** 动态评估当前指导是否足够；
  - 一个 MLP 分类器预测“充分性得分”，决定是否提前终止 LLM 的进一步思考，从而节省资源。

该设计受人类认知架构 ACT-R 启发，实现了模块化分工。

### 相比现有方法的优势
| 维度 | Tandem | 传统方法 |
|------|--------|----------|
| **无需训练 LLM** | ✅ 支持 API 调用的黑盒 LLM | ❌ 多数需微调 |
| **计算效率高** | ✅ 平均减少 ~40% 成本 | ⚠️ 固定长度推理仍昂贵 |
| **动态适应难度** | ✅ 自动为简单问题早停 | ❌ 固定预算易浪费或不足 |
| **跨领域迁移性强** | ✅ Sufficiency Classifier 可零样本迁移到 Code Generation | ❌ 多数方法领域特定 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MATH** (Hendrycks et al., 2021)：5,000 测试样本，涵盖 7 个数学子领域（代数、几何等），5 个难度等级，强调多步推理。
- **GSM8K** (Cobbe et al., 2021)：1,000 小学数学题，侧重算术推理。
- **HumanEval** (Chen et al., 2021)：164 编程题目，用于跨域泛化测试。

### 实验设置
- **模型组合**：
  - LLMs: DeepSeek-R1-Distill-Qwen-32B, Qwen3-32B, GPT-4o-mini, gpt-oss-120b（API）
  - SLMs: DeepSeek-7B, Qwen3-8B
- **推理模式**：
  - LLM 在 thinking mode 下运行（temperature=0, top_p=1.0）
  - 分三个努力层级（effort level）：低（100 tokens）、中（500）、高（1000）
- **协作流程**：
  - LLM 逐阶段生成 Thinking Insights；
  - SLM 实时判断是否已获足够信息以独立完成任务；
  - 若满足，则停止 LLM 输出，由 SLM 完成回答。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Accuracy** | 正确解答的比例 |
| **Inference Length** | 总生成 token 数（LLM + SLM） |
| **Computational Cost** | 近似 TFLOPs，公式：<br>$ \text{Cost} = \frac{1}{10^{12}} \cdot (|\theta_L| \cdot L_L + |\theta_S| \cdot (L_L + L_S)) $ |

### 基线方法对比
- **Single Model**：仅使用 LLM 或 SLM 独立推理。
- **Fixed-Length Collaboration**：固定 LLM 输出长度（low/medium/high）后交由 SLM 完成。
- **Budget Forcing**：截断 LLM 的完整推理至固定 token 数。
- **LLM Cascade**：基于二分类器一次性决定走 SLM 还是 LLM 路径。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 MATH 数据集）

| 方法 | Accuracy (%) | Computational Cost (TFLOPs) | 相比 LLM 成本降幅 |
|------|-------------|----------------------------|------------------|
| SLM (7B) only | 77.14 | 38.25 | — |
| LLM (32B) only | 80.90 | 168.35 | — |
| **Tandem (7B+32B)** | **83.46** | **99.72** | **~40.8%** |

> ✅ Tandem 不仅比单独 LLM **高出 2.56 个百分点准确率**，且仅消耗约 **59% 的计算成本**。

### 与其他高效方法对比（Table 6）

| 方法 | Accuracy (%) | Cost (TFLOPs) |
|------|-------------|---------------|
| LLM Cascade | 82.60 | 95.33 |
| Budget Forcing | 82.18 | 108.74 |
| **Tandem** | **83.46** | **99.72** |

> 🔺 Tandem 在保持竞争力成本的同时达到最高精度，优于一次性路由决策的方法。

### 跨家族协作效果（Table 2）
- **DeepSeek-7B + Qwen3-32B**：
  - MATH 准确率达 **79.96%**，远超任一单模型；
  - 成本仅为 Qwen3-32B 单独使用的 **三分之一以下**。
- 表明 Tandem 的结构化指导具有良好的 **跨模型家族兼容性**。

### 消融实验与关键发现

#### （1）指导长度的影响（Figure 5）
- 即使极短指导（如 200 tokens），也能显著提升 SLM 表现；
- 性能随指导长度先升后波动，验证了 **自适应终止机制的必要性**。

#### （2）模型规模匹配的重要性（Table 3）
- 最佳协作发生在 **能力差距适中** 的 LLM-SLM 对之间；
- 若 SLM 过小（如 1.5B），难以理解高级抽象指导，增益有限；
- 若差距太小，则互补性弱，收益边际递减。

#### （3）API 可访问 LLM 的有效性（Table 4）
- 使用 GPT-4o-mini 或 gpt-oss-120b 通过 API 提供指导：
  - 在多个科目上超越本地 SLM 和远程 LLM 单独表现；
  - 成本更低（尤其 GPT-oss-120b + DeepSeek-7B 组合）；
- 验证 Tandem **完全兼容闭源 API 模型**。

#### （4）跨领域泛化能力（Table 5）
- 在 **HumanEval**（代码生成）上应用在 MATH 上训练的 Sufficiency Classifier（无重训）：
  - 达到 **85.37% 准确率**，超过最强固定预算基线（83.54%）；
- 表明 **Perplexity/Entropy 特征具有领域无关性**，支持零样本迁移。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **轻量级高质量指导 > 完整冗长推理**  
   结构化的 Thinking Insights 可有效替代完整的 CoT，实现“少而精”的引导。

2. ✅ **动态控制优于静态预算**  
   Tandem 的 cost-aware continual judgment 能根据问题难度灵活分配资源，避免过度推理或指导不足。

3. ✅ **SLM 具备自我胜任力感知（Self-Competence Awareness）**  
   基于输出分布的统计特征（如 entropy）即可可靠判断是否需要更多指导，无需访问隐藏层。

4. ✅ **协作增益依赖合理的能力配比**  
   “导师-实习生”关系要求实习生具备基本的理解与执行能力，否则无法承接高级指导。

5. ✅ **方法具有强泛化性**  
   - 支持不同模型家族（DeepSeek/Qwen/GPT）；
   - 支持 API 黑盒调用；
   - 支持跨任务迁移（Math → Code）。

### 局限性
1. **领域泛化尚未全面验证**  
   当前实验集中于数学与编程，对于常识推理、开放问答等任务的有效性有待探索。

2. **仍需标注数据训练 Sufficiency Classifier**  
   尽管可在域间迁移，但初始训练仍需至少一个领域的标注样本（正确/错误判断）。

3. **协作模式较简单**  
   当前为固定的一对一“导师-实习生”结构，未探索更复杂的多模型协作（如 debate、multi-agent pipeline）。

### 未来工作方向
- 探索 **无监督或弱监督方式构建 Sufficiency Classifier**，减少标注依赖。
- 扩展至 **多模型动态编排系统**，实现角色自适应切换。
- 将 Tandem 架构应用于 **real-time interactive agents** 中，支持交互式纠错与反思。
- 结合 **speculative decoding** 技术进一步加速 SLM 推理过程。

---

> 📌 **一句话总结**：  
> Tandem 通过让 LLM 提供紧凑的结构化思考洞察，并由 SLM 动态判断何时足以独立完成任务，实现了高质量与高效率兼得的新型协作范式，在数学与代码任务上验证了其优越性，且具备良好泛化潜力。

</details>

---

### 8. [ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection](https://arxiv.org/abs/2604.23585)

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 7.5  
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
- **跨框架合规差距检测困难**：法规之间存在大量交叉引用（cross-reference），且不同机构内部政策表述不一，导致自动化系统难以准确识别合规缺口（compliance gap）。
- **大模型幻觉风险高**：LLM在生成合规建议时容易产生事实性错误（hallucination），影响决策可靠性。

### **提出了什么新方法或新思路**
本文提出 **CoMPLIANCENLP**，一个端到端的自动化合规监控系统，其核心创新包括：

- **KG-augmented RAG（知识图谱增强的检索增强生成）**  
  构建了一个包含 **12,847 条条款** 的 **Regulatory Knowledge Graph (RKG)**，通过图结构关系对检索结果进行重排序（KG re-ranking），显著提升跨参考条款的召回率。

- **多任务义务抽取（Multi-task Obligation Extraction）**  
  在共享的 **LEGAL-BERT 编码器** 上联合训练三个任务：
  - NER（命名实体识别）
  - Deontic Classification（道义模态分类：OBLIGATION/PROHIBITION等）
  - Cross-reference Resolution（交叉引用解析）
  实现从原始监管文本中自动提取结构化义务。

- **生产级优化流水线（Production Optimization Pipeline）**
  - **知识蒸馏（Knowledge Distillation）**：将 LLaMA-3-70B 蒸馏为 8B 模型，实现 2.2× 推理加速。
  - **Medusa Speculative Decoding**：结合领域特定的 Medusa 头部，利用监管文本低熵特性（H=2.31 bits），实现 **91.3% 的草稿token接受率**，最终达成 **2.8× 整体推理加速**。

### **相比现有方法的优势**
| 维度 | CoMPLIANCENLP | 现有方法（如 GPT-4o+RAG、商业GRC平台） |
|------|----------------|----------------------------------------|
| **准确性** | 87.7 F1（gap detection） | GPT-4o+RAG: 84.2 F1 |
| **接地准确性（grounding accuracy）** | 94.2%（r=0.83 vs human） | GPT-4o+RAG: ~85.1% |
| **延迟** | p50 = 659ms（sub-second） | 商业平台依赖规则引擎，无法实时响应 |
| **覆盖范围** | 支持 SEC、MiFID II、Basel III 三大框架 | 多数系统仅支持单一框架 |
| **部署成熟度** | 已完成4个月并行运行验证 | 多为研究原型 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
| 数据集 | 描述 |
|-------|------|
| **REGOBLIGATION** | 1,847 条标注的监管句子，涵盖 SEC (712)、MiFID II (614)、Basel III (521)，用于义务抽取任务（NER、deontic、cross-ref）。计划公开发布。 |
| **GAP-BENCH** | 423 个“义务-政策”配对样本，来自一家金融机构的真实合规审查记录，用于 gap detection 评估。含三类标签：COMPLIANT / PARTIAL GAP / FULL GAP。 |
| **ObliQA** 和 **COLING 2025 Challenge** | 外部基准数据集，用于 QA 任务对比。 |

### **实验设置和评估指标**
- **主要任务**：
  - **Gap Detection F1**：核心指标，衡量系统识别合规缺口的能力。
  - **NER F1 / Deontic F1 / XRef F1**：子任务性能。
  - **QA EM/F1**：在 ObliQA 上的问答表现。
  - **Grounding Accuracy**：使用 MiniCheck 验证生成内容是否忠实于源文档（与人类标注对比）。
- **阈值设定**：
  - 评估阈值：`θ = 0.6`
  - 部署阈值（recall-optimized）：`θ = 0.45`

### **基线方法对比**
| 基线 | 类型 |
|------|------|
| `GPT-4(5-shot)` / `GPT-4o(5-shot)` | 零样本提示 |
| `LEGAL-BERT` / `FinBERT` | 法律领域预训练模型 |
| `GPT-4+RAG` / `GPT-4o+RAG` | 使用相同混合检索策略但无 KG/multi-task/MiniCheck |
| `LLaMA-3-8B+RAG` | 蒸馏学生模型，无领域组件 |
| `LLaMA-3-70B*` | 教师模型（蒸馏前） |
| `RIRAG` (Bayer et al., 2025) | 监管领域 QA 系统 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | CoMPLIANCENLP | 最佳基线（GPT-4o+RAG） | 提升 |
|------|---------------|------------------------|------|
| **Gap Detection F1** (`θ=0.6`) | **87.7** | 84.2 | **+3.5** |
| **NER F1** | **91.3** | 88.6 | **+2.7** |
| **Deontic F1** | **92.7** | 90.5 | **+2.2** |
| **QA F1** (ObliQA) | **71.9** | 66.8 | **+5.1** |
| **Grounding Accuracy** | **94.2%** | ~85.1% | **+9.1pp** |
| **End-to-End F1**（误差传播下） | **83.4** | — | — |

> 注：所有提升均在 p<0.05 水平上显著（paired bootstrap test）

### **与基线方法的对比结果**
- 相比 `GPT-4o+RAG`，在相同检索条件下，CoMPLIANCENLP 凭借 **KG re-ranking、multi-task extraction、MiniCheck** 三大模块实现全面超越。
- 相比 `LLaMA-3-8B+RAG`（同架构但无领域组件），仍取得 **+4.2 gap F1** 提升，证明领域适配的有效性。
- 在 **QA 任务** 上大幅领先 RIRAG（71.9 vs 54.2 F1），显示更强的监管理解能力。

### **消融实验结果**
| 消融配置 | Gap F1 | Δ |
|----------|--------|----|
| 完整系统 | 87.7 | — |
| **w/o KG re-ranking** | 83.1 | **-4.6** |
| **w/o multi-task** | 84.9 | -2.8 |
| **w/o MiniCheck** | 87.2 | -0.5（但 grounding ↓7.5pp） |

> **结论**：KG re-ranking 是最大贡献者，说明**结构性知识对跨参考任务至关重要**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **KG re-ranking 是最关键的设计**  
   结构化知识图谱能有效解决嵌套交叉引用问题，在消融中造成最大性能下降（-4.6 F1）。

2. ✅ **监管文本低熵特性利于高效推理**  
   监管语言词汇受限（H=2.31 bits vs 一般文本 3.87），使得 **Medusa speculative decoding 的 token 接受率达到 91.3%**，远高于通用场景（82.7%），为其他低熵领域（如医疗、专利）提供借鉴。

3. ✅ **分析师更关注召回率而非 F1**  
   用户研究表明，**漏检（false negative）严重损害信任**，因此部署采用 recall-optimized 阈值（θ=0.45），以牺牲部分精度换取更高召回。

4. ⚠️ **GRC 集成难度超过模型开发**  
   将系统接入已有 GRC 平台耗时约3个月，与模型研发周期相当。主要挑战是**合规分类体系映射**（regulatory source → business function）。

5. 🤝 **组织采纳需要阶段性信任建设**  
   初始阶段分析师复核率达78%，第4个月降至23%。最有效的干预是发布“系统 vs 手动”周报，建立透明信任。

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **覆盖范围有限** | 当前仅支持三大框架（约占年更新量48%），扩展需新增解析器。 |
| **数据集规模小** | GAP-BENCH 仅423例，Full Gap 类别置信区间宽（±4.2）。 |
| **语言限制** | 仅支持英文文本。 |
| **部署阶段限制** | 当前为并行运行（parallel-run），所有输出仍需人工审核，未体现完全自主价值。 |
| **尾延迟较高** | p99 推理延迟达 1,082ms，略高于 sub-second 目标。 |
| **KG 更新延迟** | 存在最长18小时的盲区，在紧急监管事件中可能滞后。 |

### **未来工作方向**
- 扩展至更多监管框架（目标覆盖全部6万+年度更新）。
- 完成跨机构 **Gap-Bench 扩展版**（~1,200 示例）并公开。
- 推进部署进入 **Phase 3（影子模式）和 Phase 4（全生产）**，逐步减少人工审查。
- 开发流式 KG 更新机制，缩短盲区时间。
- 探索更细粒度的 grounding 模型（如 clause-level）。
- 加强对隐性义务（implicit obligations）和多跳交叉引用的建模。

---

> **代码与复现**：所有材料已开源 → [GitHub: bettyguo/ComplianceNLP](https://github.com/bettyguo/ComplianceNLP)

</details>

---

### 9. [Fed-DLoRA: Efficient Wireless Federated Learning with Dynamic Low-Rank Adaptation](https://arxiv.org/abs/2604.24103)

**Authors**: Huaicheng Li, Junhui Zhao, Haoyu Quan, Xiaoming Wang  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.24103v1  

#### Abstract
Federated learning (FL) offers a promising distributed learning paradigm for internet of vehicles (IoV) applications. However, it faces challenges from communication overhead and dynamic environments. Model compression techniques reduce computing and communication burden yet create trade-offs betwee...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fed-DLoRA: Efficient Wireless Federated Learning with Dynamic Low-Rank Adaptation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对 **Internet of Vehicles (IoV)** 场景下的 **Federated Learning (FL)** 面临的关键挑战：
- **通信开销大**：频繁的模型参数上传下载导致高延迟。
- **计算资源受限**：车载设备（ICV）算力有限，难以支持大规模模型训练。
- **动态环境复杂**：车辆高速移动，导致网络拓扑快速变化，参与设备不稳定。

现有基于 **Low-Rank Adaptation (LoRA)** 的 FL 方法多假设静态网络，无法适应 IoV 中的动态调度需求。

---

### 🚀 提出的新方法与创新思路

#### （1）提出 **Fed-DLoRA** 算法
- 将 **LoRA 技术集成到 FL 框架中**，仅对低秩矩阵 $ B $ 和 $ A $ 进行训练和传输，冻结主干模型权重 $ W_0 $。
- 显著减少本地训练参数数量和上行链路通信负载，提升 ICV 参与率。

#### （2）理论收敛性分析
- 结合 **Stochastic Gradient Descent (SGD)** 和 **Singular Value Decomposition (SVD)** 对 Fed-DLoRA 进行收敛分析。
- 推导出梯度差距（central vs. local）与 LoRA rank、ICV 调度策略之间的定量关系：
  $$
  \mathbb{E}\left[\|\nabla L(x(t))\|^2\right] \leq (\text{constant}) + M^2(K - Lr)
  $$
  其中 $ r $ 为 LoRA 秩，$ K $ 为总奇异值数，表明 **更高的 LoRA rank 和更多 ICV 参与有助于加速收敛**。

#### （3）联合优化框架 + ARBVS 算法
- 构建了一个多变量联合优化问题，目标是最大化系统性能（精度 + 收敛速度），同时满足时延与带宽约束。
- 提出 **Adaptive Rank, Bandwidth and Vehicle Selection (ARBVS)** 算法：
  - 采用 **枚举 + 贪心策略** 实现低复杂度求解。
  - 每轮动态选择最优的 LoRA rank $ r $、分配带宽 $ b_n $、筛选可参与的 ICV 集合 $ S $。

---

### 🔍 相比现有方法的优势
| 维度 | Fed-DLoRA | 传统方法（如 FedAvg, FedPT, FedRA） |
|------|-----------|-------------------------------|
| 参数量 | 极小（仅传 LoRA 矩阵） | 全模型参数上传 |
| 通信效率 | 大幅提升（降低 uplink cost） | 高带宽消耗 |
| 动态适应性 | 支持每轮动态调整 rank、bandwidth、vehicle | 固定配置或随机选择 |
| 系统性能 | 更快收敛、更高准确率 | 收敛慢，易受非 IID 数据影响 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **CIFAR-10** 和 **CIFAR-100**
  - 来自 `torchvision` 官方库，未做预处理。
  - 用于图像分类任务，模拟车载视觉感知场景。

### ⚙️ 实验设置
| 项目 | 设置说明 |
|------|----------|
| **模型架构** | 缩减版 ResNet18（channel width=32），约 275 万参数 |
| **LoRA 应用层** | 主要应用于卷积层，也包含 FC 层以保持统一框架 |
| **训练方式** | 所有模型从零开始随机初始化，无预训练权重 |
| **数据分布** | 分为 IID 与 non-IID 两种设定：<br>- IID：数据均匀随机分配<br>- non-IID：每个 ICV 只分配 3 类（CIFAR-10）或 30 类（CIFAR-100）样本 |
| **ICV 数量** | 每轮覆盖区域内有 20 辆 ICV 可选 |
| **硬件平台** | NVIDIA RTX 4090 GPU, Intel 14900KF CPU, Ubuntu 20.04, PyTorch 2.3.0 |
| **超参数** | SGD 优化器，batch size=32，epoch=4，learning rate=0.01 |

### 📡 通信与计算参数
| 参数 | 值 |
|------|----|
| 基站覆盖半径 $ R $ | 500 m |
| 车速范围 $ v $ | 12–22 m/s |
| 总带宽 $ B $ | 10 MHz |
| 发射功率 $ P $ | 28 dBm |
| 噪声谱密度 $ N_0 $ | -174 dBm/Hz |
| 信道模型 | Path loss: $ 128.1 + 37.6 \log_{10}(X) $ (km) |
| CPU 频率 $ f_n $ | [1.9, 3] GHz |
| 每样本计算周期 $ w_n $ | [0.8, 1.2]×10⁷ cycles |

### 🎯 评估指标
1. **Test Accuracy (%)**：随通信轮次的变化曲线。
2. **Convergence Speed**：达到目标精度（如 50%）所需时间。
3. **Communication Efficiency**：累计上行通信成本（MB）vs. 准确率。
4. **Latency**：单轮训练+上传总延迟是否满足驻留时间约束。
5. **Ablation Study**：不同 rank 和 ICV 参与比例的影响。

### 🆚 基线方法对比
- **FedAvg**：标准联邦平均算法。
- **FedPT**：部分参数冻结（冻结 ResNet18 第2、3残差块）。
- **FedRA**：基于 LoRA 的随机掩码分配聚合机制。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Fig. 3 & 4）
| 方法 | CIFAR-10 (IID, 第20轮) | CIFAR-100 (IID, 最终) | Non-IID 性能 |
|------|------------------------|------------------------|-------------|
| **Fed-DLoRA (Ours)** | **66.50%** | ~38.5% | 显著优于所有基线 |
| FedAvg | 51.01% | ~30.2% | 在 non-IID 下表现最差 |
| FedPT | 56.48% | ~33.1% | 中等提升 |
| FedRA | 57.87% | ~34.8% | 初始阶段较慢，后期追赶 |

> ✅ Fed-DLoRA 在 **第20轮即超越其他方法最终性能**，收敛极快。

---

### ⏱️ 收敛速度对比（Fig. 5）
在 **CIFAR-10 IID** 下达到 **50% 测试准确率**的时间节省：
| 对比对象 | 时间减少幅度 |
|---------|---------------|
| vs. FedAvg | ↓ **39.39%** |
| vs. FedPT  | ↓ **30.60%** |
| vs. FedRA  | ↓ **21.43%** |

> 💡 即使增加 ICV 数量，Fed-DLoRA 仍保持领先优势，体现其调度鲁棒性。

---

### 📉 通信效率对比（Fig. 6）
在达到 **50% 准确率**时的上行通信成本节省：
| 方法 | 通信成本降低 |
|------|--------------|
| vs. FedAvg | ↓ **77.49%** |
| vs. FedPT  | ↓ **51.55%** |
| vs. FedRA  | ↓ **33.90%** |

> 🌟 Fed-DLoRA 和 FedRA 因使用 LoRA 显示出更密集的数据点分布，单位通信成本获得更高增益。

---

### 🔬 消融实验结果（Fig. 7）
比较 ARBVS 与其他调度策略（随机选择 20%/40%/60% ICV，固定 rank=4 或 16）：
- **相同 rank 下**：参与 ICV 越多 → 最终准确率越高（带宽充足则聚合更充分）。
- **相同 ICV 数下**：rank 越高 → 准确率越高（更强的局部学习能力）。
- **ARBVS 自适应选择最优组合** → 实现最高准确率，验证了动态联合优化的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **LoRA 显著降低 FL 开销**：通过仅更新和传输低秩矩阵，大幅减少通信与计算负担。
2. **动态调度至关重要**：在 IoV 动态环境中，每轮自适应选择 rank、bandwidth、vehicle 是实现高性能的关键。
3. **理论指导设计**：基于 SVD 的梯度差距分析揭示了 LoRA rank 与收敛性的内在联系，支撑了优化建模。
4. **ARBVS 高效实用**：枚举+贪心策略可在多项式时间内找到近优解，适用于实时车载场景。
5. **non-IID 场景下优势更明显**：更多 ICV 参与带来更丰富的类别覆盖，缓解数据偏差问题。

---

### ⚠️ 方法的局限性
1. **依赖 FDMA 架构**：当前模型基于 Frequency Division Multiple Access 设计，扩展至其他多址方式（如 NOMA）需重新建模。
2. **集中式调度决策**：由基站统一决策，可能成为瓶颈；未来可探索分布式版本。
3. **LoRA rank 上限受限于原始模型大小**：过高的 rank 会导致压缩失效，需谨慎设置上限 $ R $。
4. **未考虑 V2V 协作**：当前仅使用 V2I 通信，忽略邻近车辆间的协作潜力。

---

### 🔮 未来工作方向
1. **扩展至 Large Language Models (LLMs)**：将 Fed-DLoRA 应用于车载 LLM 的个性化微调。
2. **支持多基站协同**：研究跨 BS 的联邦学习与 LoRA 参数迁移。
3. **引入强化学习进行调度**：替代枚举策略，实现端到端自适应控制。
4. **结合 Over-the-Air Computation (AirComp)**：进一步提升无线资源利用效率。
5. **隐私-效率权衡研究**：在 LoRA 基础上加入 Differential Privacy 或 Secure Aggregation。

---

> 📌 **总结一句话**：  
> **Fed-DLoRA 通过“LoRA + 动态联合优化”双轮驱动，在保证模型性能的同时，显著提升了 IoV 场景下 FL 的通信效率、收敛速度与系统适应性，为高效智能车联网学习提供了新范式。**

</details>

---

### 10. [Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis](https://arxiv.org/abs/2604.23072)

**Authors**: Junyan Cheng, Kyle Richardson, Peter Chin  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.23072v1  

#### Abstract
Large language model (LLM) agents are increasingly tasked with complex real-world analysis (e.g., in financial forecasting, scientific discovery), yet their reasoning suffers from stochastic instability and lacks a verifiable, compositional structure. To address this, we introduce Analytica, a novel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 **Large Language Model (LLM)** 的智能体在进行复杂现实世界分析（如金融预测、科学发现）时，存在两大核心缺陷：
1. **推理不稳定**（Stochastic Instability）：LLM 的生成过程具有随机性，导致相同输入可能产生不同输出，影响决策可靠性。
2. **缺乏可验证的结构化推理**：传统 **Chain-of-Thought (CoT)** 等方法依赖自由文本推理，难以形式化建模误差来源，也无法支持交互式“假设分析”（what-if analysis）。

### **提出的新方法与新思路**
论文提出了 **Analytica**，一种基于 **Soft Propositional Reasoning (SPR)** 的新型 LLM 智能体架构。

#### **核心思想：Soft Propositional Reasoning (SPR)**
- 将复杂分析任务重构为对多个“结果命题”（outcome propositions）的**软真值**（soft truth value）估计问题。
- “软真值”是一个介于 0 和 1 之间的概率值，表示该命题为真的置信度。
- 通过将复杂命题分解为子命题树，并系统性地估计和聚合这些软真值，实现**可验证、可组合**的推理。

#### **Analytica 架构**
采用“分而治之”（divide-and-conquer）的并行框架，包含三个核心组件：
1. **Analyzer (分析器)**：将根命题递归分解为子命题树，直至得到可验证的叶节点。
2. **Grounder (接地器)**：使用工具增强的 LLM 对叶节点进行验证和评分，赋予其软真值。
   - 特别引入了 **Jupyter Notebook Grounder**，模拟人类分析师，通过调用 API、编写代码、生成图表进行数据驱动分析。
3. **Synthesizer (合成器)**：自底向上递归聚合子命题的软真值，计算父命题的最终软真值。
   - 采用**鲁棒的线性合成规则**（robust linear synthesis rule），有效平均掉随机噪声。

### **相比现有方法的优势**
- **降低偏差**（Reduce Bias）：通过深度分解，将复杂问题简化为易于处理的原子命题，由强大的 Grounder 进行精确验证。
- **降低方差**（Reduce Variance）：线性合成规则像加权平均一样平滑噪声，避免非线性逻辑操作（如 AND/OR）带来的“尖锐转折”（tipping points）。
- **高可扩展性与效率**：支持大规模并行执行，计算时间随分析深度呈近线性增长（near-linear time complexity）。
- **支持交互式分析**：通过 **Resynthesis** 机制，用户可手动修改任一节点的真值，系统仅需快速重算受影响分支，即可进行“假设分析”。

---

## **2. 核心实验方法和设置**

### **数据集**
在 **736 个真实世界的经济与金融预测挑战**上进行评估，任务自然表现为真假命题预测，主要包括两类：
1. **金融市场任务**（Financial Market Tasks）：对资产（股票、指数、商品等）做出“长期持有 vs. 做空”的一年期预测。
2. **预测市场任务**（Predictive Market Tasks）：直接使用 Polymarket 等平台的选项，例如“谁将赢得 2024 年美国总统大选？”。

所有事件均经过筛选，确保其解决日期在模型知识截止日期（2024年6月1日）之后，以保证预测的真实性。

### **实验设置和评估指标**
- **基础模型**：主要使用 `o3-2025-04-16` 模型。
- **温度**（Temperature）：设为 0.1，以减少随机性。
- **搜索工具**：使用 Exa.ai 提供的网络搜索。

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **Accuracy (Accu.)** | 预测正确的比例，衡量 top-1 正确性。 |
| **Hard Score** | 最高 `p_true` 选项的实际回报价值。 |
| **Soft Score** | 所有选项的 `p_true` 加权平均回报。 |
| **Brier Score (BS)** | 预测分布的均方误差（MSE），越低越好。 |
| **Variance (Var.)** | 多次运行下 Hard Score 的方差，衡量稳定性。 |
| **API Cost & Wall-clock Time** | 衡量成本和响应时间。 |

### **基线方法对比**
- **独立基线**（Standalone Agents）：
  - `Basic Search`：仅依赖网络搜索。
  - `Deep Research`：OpenAI 的深度研究智能体。
  - `Jupyter Notebook`：本文提出的高级 Grounder。
- **推理框架基线**：
  - `Tree-of-Thoughts (ToT)`
  - `Graph-of-Thoughts (GoT)`
  - `Forest-of-Thoughts (FoT)`
- **Analytica 变体**：
  - 不同 **Synthesis Rule**：`Vanilla`, `Simple Logic`, `Linear`。
  - 不同 **Grounder**：`Basic Search`, `Deep Research`, `Jupyter Notebook`。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **最佳性能**：`Analytica-L`（使用线性合成规则）配合 `Deep Research` Grounder 达到 **71.06%** 的准确率，是所有方法中最高的。
- **最低方差**：`Analytica-L` 的预测方差仅为 **6.02%**，远低于其他方法，表现出极高的稳定性。
- **成本效益**：`Analytica-L` 配合 `Jupyter Notebook` Grounder 在 **70.11%** 准确率下，成本降低了 **90.35%**，时间减少了 **52.85%**，展现出极强的成本效益。

### **与基线方法的对比结果**
- 相比多种基线方法，`Analytica` 平均提升了 **15.84%** 的准确率。
- 在所有任务类别中（股票、指数、基金、加密货币等），`Analytica` 均优于其对应的独立 Grounder。
- `Linear` 合成规则显著优于 `Vanilla` 和 `Simple Logic` 规则，验证了其理论优势。

### **消融实验结果**
- **合成规则对比**（Table 2）：
  - `Linear` 规则准确率最高（71.06%），方差最低（6.02%）。
  - `Simple Logic` 规则准确率最低（65.62%），且对噪声极为敏感。
- **Grounder 消融**（Table 3）：
  - `Jupyter Notebook` Grounder 成本极低，但配合 `Analytica` 后性能接近昂贵的 `Deep Research`。
- **可扩展性测试**（Table 1）：
  - 分析深度增加 54 倍（节点数从 19.9 到 1075.3），计算时间仅增加 12 倍，证明了其**近线性时间复杂度**和卓越的可扩展性。
- **开放权重模型测试**（Table 6）：
  - `Analytica` 在 `OpenAI-OSS-20B` 等小模型上也能带来超过 **15%** 的提升，表明其对模型规模不敏感，适用于边缘设备。

---

## **4. 关键结论和发现**

### **主要发现**
1. **SPR 是有效的框架**：将复杂分析转化为软真值估计问题，能够形式化地建模和最小化推理误差（偏差和方差）。
2. **线性合成规则是关键**：相比非线性逻辑规则，线性规则具有**恒定的敏感性**、**平滑噪声**和**优雅降级**三大特性，是实现稳定聚合的理论最优选择。
3. **并行分解是可扩展性的基石**：递归调用和局部性设计使得 Analytica 能够高效处理指数级增长的分析复杂度。
4. **Jupyter Grounder 具有高性价比**：尽管功能强大，但其成本远低于 `Deep Research`，是实用部署的理想选择。

### **方法的局限性**
1. **子命题独立性假设**：框架性能依赖于子命题间的独立性，但在现实中，因素之间可能存在相关性。
2. **合成器系数的可靠性**：合成器估算的权重（β）若出现错误，会影响最终结果。
3. **统一的 Grounder 策略**：目前对所有叶节点使用相同的 Grounder，未来可探索根据命题类型动态路由到最合适的 Grounder。

### **未来工作方向**
- 探索更复杂的 **Probabilistic Graphical Models (PGMs)** 来建模子命题间的依赖关系。
- 开发更鲁棒的机制来学习和验证合成器的系数。
- 实现 **Adaptive Grounder Routing**，根据不同命题的性质自动选择最合适的分析工具（如代码、搜索、数学求解器）。
- 将 Analytica 应用于更多领域，如机器人规划、政策制定和医疗诊断，构建可靠的自主决策系统。

</details>

---

### 11. [Bridging Reasoning and Action: Hybrid LLM-RL Framework for Efficient Cross-Domain Task-Oriented Dialogue](https://arxiv.org/abs/2604.23345)

**Authors**: Yangyang Zhao, Linfan Dai, Li Cai, Bowen Xing, Libo Qin  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 7.0  
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
跨领域任务导向对话（Cross-domain Task-Oriented Dialogue, TOD）系统面临一个核心挑战：**如何在长周期、多轮次交互中同时处理显式和隐式的可行性约束（feasibility constraints）**。

- 显式约束如“目的地是纽约”、“单人入住”等可直接从用户话语中提取；
- 隐式约束如“酒店入住时间必须晚于航班到达时间”则需要常识推理或时序逻辑推断。

现有方法存在明显瓶颈：
- **LLM-based 方法** 虽能进行复杂推理，但输出易产生幻觉（hallucination）、跨轮不一致，且自由文本难以与结构化数据库对齐；
- **RL-based 方法** 擅长长期决策优化，但依赖准确完整的状态表示，一旦关键约束缺失，策略学习将严重退化；
- 简单地将 LLM 推理结果直接输入 RL 策略会导致状态污染，进而误导策略训练。

---

### 提出了什么新方法或新思路
作者提出 **Verified LLM-Knowledge empowered RL (VLK-RL)**，一种混合 LLM-RL 框架，通过模块化设计桥接推理与行动。

#### 核心思想：解耦推理与控制（Decoupling Reasoning from Control）
VLK-RL 将 LLM 的知识推理能力用于增强对话状态构建，而非直接生成动作。其流程分为三阶段：

1. **Dual-role LLM Cross-examination（推理验证）**
   - 使用两个角色分工的 LLM：**Respondent** 提出候选约束，**Judge** 通过多轮质询进行交叉审查（cross-examination），过滤幻觉和矛盾。
   - 受事实核查机制启发，模拟“问答辩论”，提升推理可靠性。

2. **Text-to-Slot Mapper（结构化接地）**
   - 将经验证的自然语言约束映射为本体对齐的 slot-value 对。
   - 利用 schema-guided extraction 和基于 Sentence-BERT 的语义相似度归一化，确保值与数据库条目兼容。

3. **RL-based Policy Optimization（策略优化）**
   - 在增强后的结构化状态 $ s' = s_t \cup V_t $ 上运行标准 RL 算法（如 PPO），实现稳健的长周期决策。

该框架实现了：
- ✅ **知识可靠性**（通过双角色验证）
- ✅ **模态对齐性**（通过文本到槽位映射）
- ✅ **无需外部监督**

---

### 相比现有方法的优势
| 维度 | VLK-RL | 传统 LLM-only / RL-only |
|------|--------|--------------------------|
| **鲁棒性** | 高（验证+结构化） | 低（幻觉累积、状态不完整） |
| **泛化性** | 强（显式+隐式约束建模） | 弱（仅依赖表面信息） |
| **可集成性** | 模块化，兼容多种 RL backbone | 架构紧耦合，难扩展 |
| **执行安全性** | 输出可执行、符合 ontology | 自由文本可能导致执行失败 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **MultiWOZ 2.1**：包含超过 10k 条人类对话语料，涵盖 7 个领域（hotel, train, taxi 等），具有丰富的跨域依赖关系。
- **Frames**：Wizard-of-Oz 数据集，子任务之间有强可行性要求（如行程顺序、时间衔接），更适合测试隐式约束建模能力。

所有实验均基于 **ConvLab-2** 工具包进行仿真环境搭建、数据库访问与评估。

---

### 实验设置和评估指标

#### 超参数设置
- 训练轮数：300
- 最大对话长度：30
- Batch size：100
- Cross-examination 回合数：5
- Semantic similarity threshold：0.7
- RL 算法：默认使用 **PPO**

#### LLM Backbone
- 开箱即用（off-the-shelf），无微调
- 包括：
  - `Qwen2-7B-Instruct`
  - `Qwen1.5-14B-Chat`（GPTQ-Int4）
  - `GPT-4o-mini`

#### 评估指标（来自 ConvLab-2）
| 指标 | 含义 |
|------|------|
| **Avg. Precision / Recall / F1** | 对话行为（dialogue act）预测准确性 |
| **Complete Rate** | 所有目标领域完成的比例 |
| **Success Rate** | 成功完成全部任务的比例（考虑约束满足） |
| **Avg. Turn (Succ)** | 成功对话的平均轮次（越低越好） |
| **Avg. Turn (All)** | 所有对话的平均轮次（反映效率） |
| **Human Rating (HR)** | 人工评分（1–5 Likert scale），衡量流畅性、自然性和冗余度 |

---

### 基线方法对比
| 类型 | 基线模型 |
|------|---------|
| **RL-based** | PPO, ACGOS |
| **LLM-based** | GALAXY, GDP-Zero, TransferTOD |
| **DST-based** | CAPID |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

#### MultiWOZ 2.1 结果
| Model | Success/Tot | Complete/Tot | Avg. Turn (All) |
|-------|-------------|---------------|------------------|
| PPO | 0.3815 | 0.4912 | 20.94 |
| CAPID | 0.5875 | 0.6820 | 20.00 |
| VLK-RL (Qwen-14B) | **0.7214** | **0.8006** | **17.35** |

#### Frames 结果
| Model | Success/Tot | Complete/Tot | Avg. Turn (All) |
|-------|-------------|---------------|------------------|
| PPO | 0.4235 | 0.6031 | 18.56 |
| CAPID | 0.5782 | 0.6701 | 20.00 |
| VLK-RL (Qwen-14B) | **0.7239** | **0.8063** | **15.91** |

> ✅ **结论**：VLK-RL 在两个数据集上均显著优于所有基线，在 Success Rate 上最高提升达 **~34个百分点**（vs PPO），且对话更短、更高效。

---

### 与基线方法的对比结果
- **相比 RL-only 方法（PPO/ACGOS）**：
  - 显著提高 Success Rate 和 Completion Rate；
  - 更少冗余提问，减少无效探索。
- **相比 LLM-only 方法（GALAXY/GDP-Zero）**：
  - 尽管 dialogue act F1 相近，但端到端成功率更高；
  - 表明未经验证的推理会积累错误，影响最终任务达成。
- **相比 DST 方法（CAPID）**：
  - CAPID 改进了显式信息追踪，但仍无法有效捕捉隐式约束；
  - VLK-RL 通过 LLM + 验证机制弥补此短板。

> 🔍 特别是在 **Frames** 数据集上优势更大，因其子任务耦合紧密，隐式约束更为关键。

---

### 消融实验结果（Ablation Study）

使用 `VLK-RL(Qwen-14B)` 进行消融分析（Fig. 4）：

| 变体 | Success/Tot ↓ | Complete/Tot ↓ | Avg. Turn (All) ↑ |
|------|----------------|------------------|--------------------|
| Full VLK-RL | **0.7214** | **0.8006** | **17.35** |
| w/o Cross-Examination | 0.4900 | 0.5400 | 21.30 |
| w/o T2S Mapper | 0.5124 | 0.5732 | 20.87 |
| w/o RL (LLM-only) | 0.5308 | 0.5901 | 21.02 |
| w/o LLM (RL-only) | 0.3815 | 0.4912 | 20.94 |

#### 关键发现：
- 移除任一组件都导致性能大幅下降；
- **Cross-examination 最关键**：去除后 Success 下降超 20%，说明未验证的 LLM 输出极具破坏性；
- **T2S Mapper 至关重要**：即使约束已验证，若未结构化归一化，仍无法被 RL 有效利用；
- **RL 优化不可替代**：纯 LLM 动作选择缺乏长期规划能力。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **可行性约束完整性是跨域 TOD 的核心瓶颈**，而 VLK-RL 通过“验证+结构化”路径成功打通 LLM 推理与 RL 决策之间的鸿沟。
2. ✅ **双角色交叉审查机制显著降低幻觉率**（App. F 显示 hallucination 从 23.5% 降至 6.5%），并提升推理一致性。
3. ✅ **Ontology-aligned text-to-slot mapping 是实现可靠执行的关键环节**，避免因格式错乱或值不匹配导致执行失败。
4. ✅ **模块化设计带来高兼容性**：VLK-RL 可无缝接入不同 RL backbone（如 DQN、PG），且性能增益稳定（App. C）。
5. ✅ **在低资源环境下依然表现优越**：即使从零训练，VLK-RL 成功率仍远高于其他方法（App. H），证明其数据高效性。

---

### 方法的局限性
1. **延迟与计算开销较高**：依赖 LLM 进行每轮推理与验证，增加响应延迟，不利于实时部署。
2. **验证非完全可靠**：双角色机制虽有效，但仍可能遗漏细微或罕见的常识依赖，尤其在模糊用户意图下。
3. **本体覆盖限制**：若约束无法用预定义的 slot-value 形式表达（如复杂逻辑规则），则会被部分丢失。
4. **LLM 性能依赖性强**：整体效果受限于所选 LLM 的推理能力和稳定性。

---

### 未来工作方向
1. **轻量化推理模块**：将验证与推理模块蒸馏至小型模型，提升推理速度与部署可行性。
2. **引入检索增强**：结合外部知识库支持稀有或专业领域的约束推理。
3. **扩展约束表示形式**：超越 slot-value，支持逻辑公式、时序图谱等形式以表达更复杂的依赖关系。
4. **动态本体适配**：开发可自适应扩展 ontology 的机制，提升系统灵活性。

---

> 📌 **总体评价**：  
> VLK-RL 提供了一种**原则性强、模块清晰、可扩展性好**的混合架构范式，为 LLM 与 RL 在复杂任务场景下的协同提供了可靠接口，推动了可信、可控、可执行的智能对话系统发展。

</details>

---

### 12. [STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator](https://arxiv.org/abs/2604.24544)

**Authors**: Alessio Sordo, Lingxiao Du, Meeka-Hanna Lenisa, Evgeny Bogdanov, Maxim Romanovsky  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.24544v1  

#### Abstract
The increasing reliance on Large Language Models (LLMs) across diverse sectors highlights the need for robust domain-specific and language-specific evaluation datasets; however, the collection of such datasets is challenging due to privacy concerns, regulatory restrictions, and the time cost for man...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前对 **Large Language Models (LLMs)** 的评估严重依赖人工标注的基准数据集（如Mintaka），但这些数据集存在以下瓶颈：

- **隐私与合规限制**：在金融、医疗等受监管领域难以获取真实数据；
- **创建成本高**：人工标注耗时、昂贵且难以规模化；
- **多语言支持不足**：主流基准以英语为主，非英语语言缺乏高质量资源；
- **现有自动化方法局限**：多数依赖已有数据增强或翻译，无法生成完全合成、可控的多语言指令-答案（I&A）对。

### 🚀 提出的新方法与新思路

本文提出 **STELLAR-E** ——一个**全自动化、端到端的合成数据生成与评估系统**，用于为LLM应用构建高质量、可定制的评估基准。

其核心架构分为两个阶段：
1. **合成数据引擎**：基于改进的 **TGRT Self-Instruct 框架**，通过多轮反馈循环生成可控、多样、难度可调的 I&A 对；
2. **评估流水线**：结合统计指标与 **LLM-as-a-Judge**（如Gemini 2.5 Pro）进行自动质量评估。

### 🔍 相比现有方法的优势

| 特性 | STELLAR-E | 其他方法（如YourBench, OmniEval, BENCHAGENTS） |
|------|-----------|-----------------------------------------------|
| 是否依赖真实数据 | ❌ 完全无需输入文档或真实语料 | ✅ 多数依赖输入文本或专家构造的数据 |
| 可扩展性 | ✅ 支持任意数量的 I&A 对生成 | ⚠️ 通常受限于输入规模或过滤机制导致样本减少 |
| 多语言支持 | ✅ 内建多语言生成能力，避免翻译伪影（translationese） | ⚠️ 非英语常靠机器翻译，文化适配差 |
| 质量控制机制 | ✅ 多阶段反馈循环 + G-Eval 过滤 + DVE/DFE优化 | ⚠️ 多为单次生成+简单过滤，无迭代优化 |
| 难度与多样性调控 | ✅ 支持 **Difficulty Enhancement (DFE)** 和 **Diversity Enhancement (DVE)** | ⚠️ 缺乏显式难度提升机制 |

> 💡 创新亮点：首次实现“从零开始”生成**语义可控、语言可调、难度可增强**的多语言评估数据，并形成闭环评估流程。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **真实基准数据集**：
  - **Mintaka**：一个多语言、复杂的端到端问答数据集，作为黄金标准（ground truth）。
  - 包含英文原版（`mintaka_en_real`）和专业翻译意大利语版本（`mintaka_it_real`）。

- **对照组数据集**：
  - **机器翻译版**：将英文Mintaka用API自动翻译成意大利语（`mintaka_it_translated`），用于衡量翻译质量的影响。

- **合成数据集**：
  - `mintaka_en_synthetic` / `mintaka_it_synthetic`：由STELLAR-E生成的英意双语合成数据，基于8个Question Types（QTs）设计。

> 所有数据集最终均随机采样 **1,500个 I&A 对** 用于公平比较。

### ⚙️ 实验设置

- **生成参数**（见Table 2）：
  - QT数量：8
  - 每轮采样主题数：5（共20个候选）
  - 迭代次数：50
  - 每轮生成指令数：50
  - DVE相似度阈值：0.3
  - G-Eval评分阈值 $ T = 8 $（满分10分）

- **模型选择**：
  - **生成模型**：`Gemini-1.5-pro-002`
  - **评估/过滤模型**：`Gemini-2.0-flash-001`（生成阶段）、`Gemini-2.5-Pro`（最终meta-evaluation）
  - **嵌入模型**：`bge-m3`（用于DVE阶段的语义去重）

- **被评测模型**（Evaluatee Models）：
  - **强模型**：`Gemini 2.5 Flash`（MMLU-Pro得分83.6%）
  - **弱模型**：`Llama 2 Chat 13B`（MMLU-Pro得分25.3%）

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **G-Eval** | 自定义版LLM-as-a-Judge框架，采用Chain-of-Thought推理，评估Accuracy、Relevance、Completeness三项，平均得分（0–10） |
| **ROUGE-L** | 衡量生成答案与参考答案之间的最长公共子序列，反映词汇重叠 |
| **BERTScore F1** | 基于上下文嵌入的语义相似性度量，尤其适用于跨语言场景 |
| **Answer Relevance (ARel)** | 评估回答是否紧扣问题，惩罚离题或冗余内容 |

> 所有指标均在**合成 vs 真实数据集上对比运行**，计算性能差距（Δ）。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 3 & 4）

#### 在强模型上的表现（Gemini 2.5 Flash）

| 数据集 | G-Eval Δ | ROUGE-L ↓ | BERTScore F1 ↑ | ARel ↑ |
|--------|----------|------------|------------------|---------|
| Real English | — | 0.397 | 0.547 | 0.523 |
| Synthetic EN (DVE+DFE) | **+5.7%** | -31.1% | -7.3% | +13.0% |
| Translated IT | +2.3% | +2.0% | +3.1% | -1.2% |
| Synthetic IT (DVE+DFE) | **+5.8%** | -13.9% | +13.9% | +16.3% |

> ✅ 合成数据在G-Eval上仅比真实数据高约 **+5.7%**，表明其挑战性接近真实基准。

#### 在弱模型上的表现（Llama 2 Chat 13B）

| 数据集 | G-Eval Δ | ROUGE-L ↓ | BERTScore F1 ↑ | ARel ↑ |
|--------|----------|------------|------------------|---------|
| Real English | — | 0.169 | 0.389 | 0.233 |
| Synthetic EN (DVE+DFE) | **+10.9%** | -3.5% | +15.4% | +23.7% |
| Real Italian | — | 0.160 | 0.401 | 0.242 |
| Synthetic IT (DVE+DFE) | **+0.7%** | +0.3% | +19.2% | +16.2% |

> ⚠️ 弱模型在合成数据上表现显著更好（G-Eval高出10.9%），说明小模型可能“利用”了合成数据中的模式。

### 🔁 消融实验结果（Ablation Study）

| 配置 | G-Eval Δ（EN） | 效果分析 |
|------|----------------|----------|
| 原始合成（无DVE/DFE） | +12.6% | 明显高于真实数据，题目太简单 |
| +DVE（多样性增强） | +9.7% | 降低重复性，略有改善 |
| +DVE+DFE（难度增强） | **+5.7%** | 最接近真实数据，验证DFE有效性 |

> ✅ **DFE模块显著提升了任务难度**，使合成数据更贴近真实挑战水平。

此外发现：
- **DVE+DFE 合成数据的 ROUGE-L 分数最低** → 回答句法差异更大，说明模型需更多推理而非照搬模板；
- **BERTScore 和 Answer Relevance 更高** → 回答语义更完整、相关性强；
- **意大利语合成数据几乎与真实数据持平（Δ=+0.7%）** → 显示多语言生成的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **合成数据可达近似真实基准的质量水平**：
   - 在强模型上，STELLAR-E生成的合成数据与真实Mintaka的G-Eval差距仅为 **+5.7%**，证明其具备作为替代基准的能力。

2. **DFE与DVE机制有效提升数据质量**：
   - 难度增强（DFE）成功提高了指令的对抗性，减少了“过于简单”的问题；
   - 多样性增强（DVE）通过embedding距离过滤，显著降低了冗余。

3. **小模型更容易“过拟合”合成数据**：
   - 弱模型（Llama 2 13B）在合成数据上得分膨胀明显（Δ=+10.9%），提示合成数据可能存在潜在模式偏差，易被小模型捕捉。

4. **机器翻译不如原生多语言生成**：
   - 自动翻译的意大利语数据虽略易（+2.3% G-Eval），但缺乏文化适应性，而STELLAR-E能直接生成符合目标语言习惯的内容。

5. **端到端自动化可行**：
   - 整个流程无需人工干预即可完成从主题生成到评估打分的全过程，适合集成进 **LLMOps CI/CD 流水线**。

### ⚠️ 局限性

1. **仍依赖LLM作为生成器与评判者**：
   - 存在**self-enhancement bias**风险（即生成模型偏好自身风格）；
   - 当前未使用多模型集成judge，可能引入系统性偏见。

2. **仅在一个基准（Mintaka）上验证**：
   - 泛化能力有待在更多领域（如金融、法律）和语言中测试。

3. **缺乏人类评估**：
   - 尽管自动指标良好，但仍需native speaker参与评估文化适配性和自然性。

4. **复杂任务支持有限**：
   - 当前主要针对单轮I&A对，尚未扩展至multi-turn对话或多跳推理任务。

### 🔮 未来工作方向

1. **扩大meta-evaluation范围**：
   - 在多个基准（如FinancialQA、LegalBench）上验证通用性；
   - 引入更多家族的LLM（如Llama、Mixtral、Claude）进行交叉评估。

2. **引入人类评估环节**：
   - 对少量合成样本进行人工打分，识别潜在的文化不敏感或语言不自然问题。

3. **扩展至RAG系统评估**：
   - 生成合成source documents + grounded I&A pairs，用于评估Retrieval-Augmented Generation系统。

4. **探索训练用途**：
   - 将该pipeline生成的数据用于RLHF或SFT，但需警惕**dataset contamination**问题。

5. **模块复用性开发**：
   - 各组件（如Topic Generator、DFE Module）可独立使用于其他合成数据项目。

---

> 🧩 **总结一句话**：  
> STELLAR-E实现了**无需真实数据、全自动、可定制、多语言**的LLM评估数据生成，是迈向高效、安全、可扩展LLMOps的重要一步，尤其适用于高监管行业的持续质量监控。

</details>

---

### 13. [DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents](https://arxiv.org/abs/2604.24320)

**Authors**: Junshuo Zhang, Chengrui Huang, Feng Guo, Zihan Li, Ke Shi, Menghua Jiang, Jiguo Yu, Shuo Shang, Shen Gao  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.24320v1  

#### Abstract
Large language model (LLM) agents that follow the sequential "reason-then-act" paradigm have achieved superior performance in many complex tasks.However, these methods suffer from limited exploration and incomplete environmental understanding, as they interact with only a single environment per step...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 LLM-based agents 多采用 **ReAct** 范式（Reason-then-Act），即在每个时间步仅与一个环境交互一次。这种**串行探索**方式存在以下局限：
- **探索不充分**：代理只能从单一路径获取环境反馈，导致对环境的理解片面、有偏。
- **行为冗余**：多次采样尝试（multi-sampling）仍倾向于收敛到相似动作，缺乏多样性。
- **效率低下**：并行采样多个轨迹会显著增加 token 消耗和推理延迟。

### **提出的新方法与新思路**
本文提出了 **Diverse Parallel Exploration Policy Optimization (DPEPO)**，一种全新的强化学习框架，其核心思想是：
- **并行探索范式（Parallel Exploration）**：允许 agent 同时与多个平行环境（parallel environments）交互，实现跨轨迹经验共享。
- **多样化探索奖励机制**：设计了两个新颖的 step-level 奖励来主动抑制冗余行为，促进多样探索：
  - **Diverse Action Reward (DAR)**：惩罚相同动作在深度（同一环境内）和宽度（不同环境间）上的重复。
  - **Diverse State Transition Reward (DTR)**：惩罚相同状态转移（state-action transition）的重复。
- **分层奖励结构**：结合 trajectory-level 成功奖励与上述 step-level 奖励，通过 group-relative advantage 提供多粒度优化信号。

### **相比现有方法的优势**
- **更全面的环境认知**：通过并行探索构建更完整的环境模型，提升决策质量。
- **更高的样本效率与任务成功率**：在 ALFWorld 和 ScienceWorld 上达到 SOTA 性能。
- **可比甚至更优的推理效率**：尽管 token 预算略高，但由于轨迹更短、决策更高效，实际运行时间更少。
- **无需复杂奖励模型或 Critic 网络**：基于规则的奖励设计，训练简单高效。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **ALFWorld**：基于文本的交互式环境，模拟家庭日常任务（如“把铅笔放到桌子上”），需多步推理和常识理解。
- **ScienceWorld**：科学推理基准，涵盖 10 个小学科学主题（如电路、生命周期），任务包含多个子目标。

### **实验设置与评估指标**
- **模型基础**：基于 Qwen2.5-Instruct 和 Qwen3 系列模型进行微调。
- **并行环境数 K=4**：每个推理步骤最多与 4 个平行环境交互。
- **最大步数 T=25**。
- **训练流程**：
  1. **SFT 冷启动阶段**：使用人工标注 + DeepSeek-V3.2 合成的并行探索轨迹进行监督微调。
  2. **RL 优化阶段**：采用 group-based policy optimization（类似 GRPO），无 Critic 模型。
- **评估指标**：
  - **Success Rate (%)**：任务完成率（主指标）。
  - **Average Steps / Token Usage / Inference Time**：效率指标。

### **基线方法对比**
- **闭源大模型**：GPT-4o、DeepSeek-V3、DeepSeek-R1。
- **ReAct 范式下的 RL 方法**：
  - **GRPO**：基于组的策略优化。
  - **GiGPO**：引入锚定状态分组的 step-level 奖励。
  - **RLVMR**：使用元推理奖励改善探索。
  - **SPEAR**：结合自模仿与内在奖励。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 方法 | ALFWorld (All) | ScienceWorld (Avg.) |
|------|----------------|---------------------|
| GPT-4o | 48.0% | 45.2% |
| DeepSeek-R1 | 75.0% | 27.6% |
| GRPO (7B) | 77.5% | 35.3% |
| RLVMR (7B) | 91.6% | 47.5% |
| **DPEPO (7B)** | **98.6%** | **61.4%** |

> ✅ 在 ALFWorld 上超越最强基线 RLVMR 近 **7 个百分点**，在 ScienceWorld 上提升超 **13 个百分点**。

### **与基线方法的对比结果**
- **显著优于所有 ReAct 范式方法**：无论是在已见任务（In-Domain）还是未见任务（OOD/L1/L2）上，DPEPO 均取得 SOTA。
- **泛化能力强**：在 ALFWorld OOD 和 ScienceWorld L2（最难）上表现依然领先。
- **效率优势明显**：
  - 尽管 token 消耗更高（2283.4 vs GiGPO 1115.1），但平均步数更少（12.3 vs 15.2），**总推理时间更低（44.7s vs 70.8s）**。
  - 并行动作并行执行，不增加实际延迟。

### **消融实验结果**
| 方法 | ALFWorld (All) | ScienceWorld (Avg.) |
|------|----------------|---------------------|
| ColdStart | 93.6% | 58.9% |
| DPEPO w/o DAR | 97.1% | 59.0% |
| DPEPO w/o DTR | 96.4% | 59.2% |
| DPEPO w/o DAR & DTR | 96.4% | 60.6% |
| **DPEPO (完整)** | **98.6%** | **61.4%** |

> 🔍 **发现**：
> - 移除任一奖励（DAR 或 DTR）均导致性能下降，说明两者协同作用。
> - 完整 DPEPO 显著优于其他变体，验证了**多样化奖励对性能提升的关键作用**。
> - 可视化显示 DPEPO 的动作与状态转移重复率最低，探索多样性最高。

---

## **4. 关键结论和发现**

### **主要发现**
1. **并行探索显著提升环境理解能力**：同时与多个环境交互使 agent 能快速建立全局认知，避免局部最优陷阱。
2. **多样性奖励有效抑制冗余行为**：DAR 和 DTR 成功引导 agent 执行更丰富、更有效的探索策略。
3. **DPEPO 实现性能与效率双赢**：不仅成功率 SOTA，且因决策更精准而缩短了任务完成路径，提升了实际推理效率。
4. **方法具有良好的扩展性**：随着模型规模增大（Qwen2.5 → Qwen3），性能持续提升，表现出良好 scaling law。

### **方法的局限性**
- **依赖可复制的环境副本**：在真实物理世界或难以并行化的场景中（如机器人控制），构建多个独立但初始一致的环境较困难。
- **上下文长度限制**：虽然设计了压缩提示策略，但长序列历史仍可能逼近模型 context window 极限。
- **潜在的过探索风险**：若无适当约束，可能因过度并行而导致资源浪费（但文中指出 context limit 和多样性奖励本身构成自然约束）。

### **未来工作方向**
- **拓展至现实场景**：研究如何让 agent 并行处理**不同类型的任务**而非同任务多环境，以适应真实应用。
- **动态调整并行数量 K**：根据任务难度自动调节探索广度，进一步优化效率。
- **结合记忆机制**：将跨环境获得的知识显式整合进长期记忆，增强知识迁移能力。
- **应用于更多 agent 场景**：如代码生成（并行测试）、GUI 自动化（多浏览器实例）、工具调用（并行 API 请求）等。

---

> 📌 **总结一句话**：  
> **DPEPO 通过引入“并行探索 + 多样性奖励”的新范式，突破了传统 ReAct 的串行瓶颈，在提升 LLM agent 探索广度与决策质量的同时，实现了 SOTA 性能与高效推理的统一。**

</details>

---

### 14. [CoFi-PGMA: Counterfactual Policy Gradients under Filtered Feedback for Multi-Agent LLMs](https://arxiv.org/abs/2604.22785)

**Authors**: Stela Tong, Elai Ben-Gal  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.22785v1  

#### Abstract
Large language model (LLM) deployments increasingly rely on multi-agent architectures in which multiple models either compete through routing mechanisms or collaborate to produce a final answer. In both settings, the learning signal received by each agent is filtered by the system mechanism. Routing...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# CoFi-PGMA: Counterfactual Policy Gradients under Filtered Feedback for Multi-Agent LLMs  
**论文核心总结**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Reinforcement Learning from Human Feedback (RLHF)** 主要针对单一策略模型设计，但在实际部署中，越来越多的 LLM 系统采用 **multi-agent 架构**，例如：
- **Routing（路由）机制**：多个专家模型生成候选答案，由控制器选择一个输出，仅该响应获得反馈（selection-gated feedback）。
- **Collaborative（协作）机制**：多个角色（如 planner-solver-critic）协同生成最终答案，共享同一个最终奖励。

在这两种机制下，每个 agent 接收到的反馈是“被过滤”的（filtered feedback），导致标准 RLHF 存在以下问题：
- 在 routing 中，未被选中的 agent 完全得不到学习信号 → **统计偏差（statistical bias）**
- 在 collaboration 中，所有 agent 共享奖励，无法区分个体贡献 → **信用分配模糊（credit assignment problem）**

因此，直接对每个 agent 应用 naive RLHF 是 **misspecified** 的。

---

### 提出的新方法：CoFi-PGMA
作者提出 **CoFi-PGMA**（Counterfactual Policy Gradients under Filtered Feedback for Multi-Agent LLMs），一种统一框架，用于在 filtered feedback 下进行多智能体训练。

#### 核心思想
引入 **counterfactual marginal contribution（反事实边际贡献）** 作为正确的 per-agent 学习信号：

> △i,t = E[G − G⁻ⁱ | z, aᵢ]  
> 即：agent i 的贡献 = 当前系统回报 − 若 agent i 不参与时的反事实回报

这一定义在两种机制下分别对应已有方法的推广：
- **Routing 场景**：等价于使用 **doubly robust estimator** 对未选中候选者进行 off-policy 估计
- **Collaboration 场景**：等价于 **leave-one-out (LOO) difference reward**

#### 技术实现
- 使用 **GRPO**（Generalized Reward Policy Optimization）作为底层优化算法
- 引入 multiturn-aware reward（如 MR 指标）以捕捉长期影响
- 设计了适用于两种机制的具体 estimator 实现方式（见附录 G）

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **理论统一性** | 统一处理 routing 和 collaboration 两类 filtered feedback 机制 |
| **去偏性** | 在 routing 中纠正 selection-gated bias；在 collaboration 中减少 credit attribution variance |
| **实用性** | 可集成到现有 RLHF 流程中，兼容 GRPO、LoRA 等主流技术 |
| **信号质量提升** | 提高信噪比，避免 free-rider 问题和错误归因 |

---

## 2. 核心实验方法和设置

### 数据集
- **GSM8K**：数学推理数据集，包含小学水平的应用题，评估 exact-match 准确率。

### 实验设置
- **任务形式**：单轮决策（single-turn），每 prompt 多 agent 并行生成完整解答
- **Agent 架构**：
  - 基础模型：`meta-llama/Llama-3.2-3B-Instruct`
  - 使用 **LoRA adapters** 实现 K=3 个 specialized agents：
    - Direct-computation agent
    - Equation-first agent
    - Final-answer-only agent
- **Router**：
  - 基于轻量级 reward predictor + softmax 路由
  - 探索参数 annealing：T 从 1.0 → 0.7，ε 从 0.05 → 0.03
- **训练配置**：
  - Batch size: 1，Group size G=4（每 prompt 采样 4 组候选）
  - AdamW 优化器，lr=1e-5，clip norm=1.0
  - GRPO clipping=0.2，KL coeff=0.02
  - 总共 150 updates，512 训练样本，128 验证样本

---

### 评估指标
| 指标 | 含义 |
|------|------|
| **Exact-match Accuracy** | 主要性能指标 |
| **Routing Entropy** | 衡量路由多样性，防止 collapse |
| **Brier Score** | Reward predictor 的校准程度 |
| **Specialization Score** | 路由是否倾向于选择“专业匹配”的 agent |
| **Oracle Accuracy & Regret** | Oracle 选择最佳候选的准确率；regret = oracle acc − 实际路由 acc |

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Frozen single-agent** | 无训练的单 agent 基线（LoRA 未微调） |
| **Winner-take-all GRPO** | 传统 routing 训练方式：只有被选中的 agent 得到 reward |
| **DR-GRPO (Ours)** | 使用 doubly robust estimator 估计每个 agent 的 marginal contribution |
| **Oracle routing** | 非可训练诊断：回放中选择 reward 最高的候选（上界） |

---

## 3. 主要实验结果和性能指标

### 性能对比（Table 2）

| Method | Accuracy | Reward | Entropy | Brier | Specialization |
|--------|----------|--------|---------|-------|----------------|
| Frozen single-agent | 0.352 | 0.352 | — | — | — |
| Winner-take-all GRPO | 0.370 | 0.370 | 0.68 | 0.50 | 0.46 |
| **DR-GRPO (Ours)** | **0.410** | **0.410** | **0.79** | **0.43** | **0.58** |

> ✅ **绝对提升 5.8% vs 冻结基线，4.0% vs Winner-take-all**

---

### Oracle 路由分析（Table 3）

| Method | Router Acc | Oracle Acc | Regret |
|--------|------------|------------|--------|
| Winner-take-all GRPO | 0.37 | 0.45 | 0.08 |
| **DR-GRPO** | **0.41** | **0.47** | **0.06** |

> ✅ DR-GRPO 不仅提升了实际路由准确率，也提高了候选质量（oracle acc ↑），且 **regret 更低**

---

### 关键发现
- **更高的 entropy**（0.79 vs 0.68）表明 DR-GRPO 避免了过早收敛到单一 agent
- **更低的 Brier score**（0.43 vs 0.50）说明 reward predictor 更加校准
- **specialization score 提升**（0.58 vs 0.46）表明 agent 更趋向于发挥其专长

> 这些辅助指标证明：**counterfactual correction 改善了整个系统的 learning dynamics**

---

## 4. 关键结论和发现

### 主要结论
1. **Naive RLHF 在 multi-agent 系统中是 misspecified 的**，因为 filtered feedback 导致学习信号偏差或高方差。
2. **Counterfactual marginal contribution 是更合理的 per-agent 学习目标**，能够统一处理 routing 与 collaboration 场景。
3. **CoFi-PGMA 显著优于 winner-take-all baseline**，在 GSM8K 上实现 0.41 准确率，验证了反事实估计的有效性。
4. 该方法不仅提升最终性能，还改善了 **router calibration、agent specialization 和 exploration diversity**。

---

### 局限性
- 实验规模较小：仅使用 Llama-3.2-3B 模型、小样本子集、单 seed
- 尚未在 **collaborative generation** 场景中进行实证验证（仅理论推导）
- Reward predictor 和 router 本身也是学习中的组件，存在联合训练稳定性挑战

---

### 未来工作方向
1. 扩展到更大模型（如 70B+）、更多训练数据和多 seed 实验
2. 在真实 collaborative pipeline（如 planner-solver-editor）中验证 LOO estimator 效果
3. 结合更复杂的 multiturn-aware reward（如 MR 指标）进行 long-horizon credit assignment
4. 探索更高效的 counterfactual rollout 方法（降低计算开销）
5. 将框架扩展至 tool-using、agent memory 等复杂 multi-agent LLM 架构

---

> 📌 **总体评价**：  
> CoFi-PGMA 提供了一个 **principled、unified、practical** 的视角来解决 multi-agent LLM 中的 filtered feedback 问题。它将 off-policy evaluation、counterfactual reasoning 与 modern RLHF 工具（GRPO）相结合，为未来构建可学习的 mixture-of-experts 和 agent pipelines 提供了重要基础。

</details>

---

### 15. [PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks](https://arxiv.org/abs/2604.23148)

**Authors**: Tianlong Yu, Yang Yang, Ziyi Zhou, Jiaying Xu, Siwei Li, Tong Guan, Kailong Wang, Ting Bi  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.23148v1  

#### Abstract
The emerging threat of AR-LLM-based Social Engineering (AR-LLM-SE) attacks (e.g. SEAR) poses a significant risk to real-world social interactions. In such an attack, a malicious actor uses Augmented Reality (AR) glasses to capture a target visual and vocal data. A Large Language Model (LLM) then ana...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

当前基于 **AR-LLM** 的 Social Engineering（社会工程学攻击）系统（如 SEAR）在实际应用中面临两大瓶颈：

1. **Cold-start Personalization（冷启动个性化）**  
   - 现有系统依赖 **Retrieval-Augmented Generation (RAG)** 构建目标画像，导致首次交互时存在显著延迟（约 43.3 秒），破坏对话流畅性。
   
2. **Static Attack Strategies（静态攻击策略）**  
   - 攻击策略采用固定阶段模板（如“先破冰、再建立信任”），缺乏对目标实时反应的动态适应能力，难以应对非线性、高摩擦的现实对话。

---

### 🚀 提出的新方法：PhySE 框架

PhySE 是一个支持心理策略自适应的实时 AR-LLM 社会工程框架，包含两个核心创新模块：

#### （1）**VLM-Based Social-Context Training**  
- 利用 **Parameter-Efficient Fine-Tuning (PEFT)** 和 **LoRA** 对 **Vision-Language Model (VLM)** 进行社交上下文预训练。
- 将社交线索（如身份特征、兴趣、背景）内化为模型参数，实现无需检索即可快速生成目标画像。
- 优势：消除 RAG 引入的冷启动延迟，提升首次交互响应速度与一致性。

#### （2）**Adaptive Psychological Agent**  
- 设计一个基于心理学理论的动态路由代理，依据实时交互信号选择策略类别。
- 基于 **Stereotype Content Model (SCM)** 和 **Trust & Influence Model**，将策略分为三类工具箱：
  - **Warmth/Rapport（亲和/关系建立）**：通过共情、镜像、自我披露等增强好感。
  - **Credibility/Commitment（可信度/承诺）**：利用互惠、社会证明、权威线索建立可信形象。
  - **Motivation/Action（动机/行动引导）**：在信任足够时提出小请求，逐步升级至敏感操作。
- 引入 **Latent Trust State** 模型，量化目标信任水平，并据此决定是否升级或降级策略。

---

### 🔍 相比现有方法的优势

| 维度 | SEAR / Baseline | PhySE |
|------|------------------|--------|
| **响应延迟** | 高（~43.3s profile generation） | 显著降低（~10.5s） |
| **策略灵活性** | 固定脚本，无法动态调整 | 动态路由，基于信任状态切换策略 |
| **对话自然性** | 存在明显停顿，易被察觉 | 更流畅、真实感更强 |
| **攻击有效性** | 中等 | 显著提升多通道合规率 |

---

## 2. 核心实验方法和设置

### 📚 数据集

- **PhySE Dataset**：由作者构建并公开发布，包含：
  - **360 条标注对话**，来自 **60 名参与者** 在真实场景下的互动（如咖啡馆、社交活动）。
  - 多模态数据：AR眼镜采集的视觉、语音流、环境元数据。
  - 公开社交痕迹：用于个性化的文本、图像、短视频资料。
  - 后续调查问卷：评估信任、自然度、接受意愿等主观指标。
  - 路由记录：每轮决策的心理策略路径。

> ⚠️ 所有研究均通过 **IRB 审批**，数据匿名化处理，符合伦理规范。

---

### 🧪 实验设置与评估指标

#### 实验设计
- 参与者交替扮演攻击者与目标角色，减少角色偏差。
- 设置 **7 种对比条件**：
  1. Basic Conversation（无辅助）
  2. Naive AR + Multimodal LLM
  3. SEAR（基线）
  4. PhySE（完整框架）
  5. PhySE w/o VLM 优化
  6. PhySE w/o Psychological Agent

#### 评估维度

| 类型 | 指标 |
|------|------|
| **用户体验** | Social Experience Score（5分制）、自然度、真诚感、节奏感等11个维度 |
| **攻击有效性** | 用户后续行为合规率：<br>• Photo Link（点击链接）<br>• Social App（加社交好友）<br>• SMS（打开短信）<br>• Phone Call（接听电话） |
| **系统性能** | Profile Generation Latency（最小值、最大值、P90、平均值） |
| **消融分析** | 移除各组件后的性能变化 |

#### 基线方法对比
- **Basic Conversation**：纯人工交流
- **Naive AR + LLM**：AR感知 + 直接调用 LLM 生成建议
- **SEAR**：当前最先进的 AR-LLM-SE 框架，依赖 RAG 和固定策略

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### （1）**Social Experience Score（社会体验评分）**

| 方法 | 平均得分 | 标准差 |
|------|---------|--------|
| Basic Conversation | 3.03 | 1.30 |
| Naive AR + LLM | 4.13 | 0.72 |
| SEAR | 4.73 | 0.51 |
| **PhySE** | **4.83** | **0.37** |

- **83.3%** 的用户给 PhySE 打出最高分（5分），**无一人评3分以下**。
- 表明 PhySE 提供了最稳定、高质量的交互体验。

#### （2）**攻击有效性（Social Engineering Effectiveness）**

| 渠道 | SEAR 成功率 | PhySE 成功率 | 提升幅度 |
|------|-------------|--------------|----------|
| Photo Link | ~60% | ~75% | ↑15% |
| Social App | ~55% | ~70% | ↑15% |
| SMS | ~45% | ~65% | ↑20% |
| Phone Call | ~40% | ~60% | ↑20% |

> ✅ PhySE 在需要更高信任的渠道（SMS、Call）上提升更显著，说明其心理策略更有效建立深层信任。

#### （3）**延迟性能对比（Profile Generation Latency）**

| 方法 | 组件 | 平均延迟 | P90 延迟 |
|------|------|----------|-----------|
| SEAR | Multimodal LLM | **43.3s** | 52.7s |
| PhySE | Trained VLM | **10.5s** | 19.7s |

- **延迟下降超过 75%**，解决了冷启动瓶颈。
- 心理代理部分虽略有增加（5.8s vs 2.8s），但波动极小（4.8–6.3s），更具可预测性。

#### （4）**消融实验（Ablation Study）**

| 配置 | 平均体验分 | 标准差 |
|------|------------|--------|
| PhySE（完整） | **4.83** | 0.37 |
| w/o Trained VLM | 2.93 | 1.14 |
| w/o Psychological Agent | 3.00 | 1.13 |

- 移除任一组件均导致性能暴跌约 **38%**，且稳定性大幅下降。
- 证实两个模块缺一不可，协同作用显著。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **理论驱动的心理策略能显著提升 AR-LLM-SE 攻击效果**  
   - 动态切换 Warmth、Credibility、Action 策略，比固定脚本更贴近人类说服机制。

2. **VLM 内化社交知识可有效解决冷启动问题**  
   - 无需在线检索即可生成连贯画像，大幅提升首回合响应效率。

3. **低延迟 + 高自然性 = 更强欺骗性**  
   - PhySE 不仅更快，而且对话更自然、真诚感更强，使目标更愿意继续互动甚至分享信息。

4. **真实世界中的攻击可行性已被验证**  
   - 在真实社交场景中收集的数据表明，此类攻击已具备现实威胁潜力。

---

### ⚠️ 局限性

1. **仍依赖公共社交数据进行个性化**  
   - 若目标数字足迹稀少，VLM 推断准确性可能下降。

2. **心理状态建模仍为简化版本**  
   - 当前的 Latent Trust State 是标量估计，未完全捕捉复杂情绪动态。

3. **硬件限制影响部署广度**  
   - 当前依赖高性能服务器运行 LLM，尚未完全边缘化。

4. **伦理争议大，需严格监管**  
   - 技术本身可用于恶意目的，必须配套防御机制。

---

### 🔮 未来工作方向

1. **开发实时检测与干预机制**  
   - 如基于行为异常识别潜在 AR-LLM-SE 攻击。

2. **增强隐私保护技术**  
   - 在 AR 设备层实现视觉/语音模糊化、身份脱敏。

3. **扩展多文化心理模型适配**  
   - 当前策略基于通用心理学理论，未来可针对不同文化背景定制。

4. **推动政策与法规制定**  
   - 呼吁对消费级 AR 设备的感知能力进行法律约束。

5. **开源防御基准测试平台**  
   - 利用发布的 **PhySE Dataset** 构建攻防对抗评测体系。

---

> 🔗 **代码与数据集已开源**：[https://github.com/2192537130/PhySE](https://github.com/2192537130/PhySE)

> 📢 本文不仅是对新型攻击范式的揭示，更为未来的 **AR 安全、人机交互伦理、AI 防御研究** 提供了重要基础。

</details>

---

### 16. [MetaGAI: A Large-Scale and High-Quality Benchmark for Generative AI Model and Data Card Generation](https://arxiv.org/abs/2604.23539)

**Authors**: Haoxuan Zhang, Ruochi Li, Yang Zhang, Zhenni Liang, Junhua Ding, Ting Xiao, Haihua Chen  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.23539v1  

#### Abstract
The rapid proliferation of Generative AI necessitates rigorous documentation standards for transparency and governance. However, manual creation of Model and Data Cards is not scalable, while automated approaches lack large-scale, high-fidelity benchmarks for systematic evaluation. We introduce Meta...

---

### 17. [RouteNLP: Closed-Loop LLM Routing with Conformal Cascading and Distillation Co-Optimization](https://arxiv.org/abs/2604.23577)

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.23577v1  

#### Abstract
Serving diverse NLP workloads with large language models is costly: at one enterprise partner, inference costs exceeded $200K/month despite over 70% of queries being routine tasks well within the capability of smaller models. We present RouteNLP, a closed-loop framework that routes queries across a ...

---

### 18. [LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models](https://arxiv.org/abs/2604.23809)

**Authors**: Tianchun Li, Haochen Liu, Vishwa Pardeshi, Xingchen Wang, Tianci Liu, Huijun Zhao, Wei Fan, Jing Gao  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.23809v1  

#### Abstract
Small language models (SLMs) are promising for real-world deployment due to their efficiency and low operational cost. However, their limited capacity struggles with high-stakes legal reasoning tasks that require coherent statute interpretation and logically consistent deduction. Furthermore, traini...

---

### 19. [Learning Without Adversarial Training: A Physics-Informed Neural Network for Secure Power System State Estimation under False Data Injection Attacks](https://arxiv.org/abs/2604.22784)

**Authors**: Solon Falas, Markos Asprou, Charalambos Konstantinou, Maria K. Michael  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.22784v1  

#### Abstract
State estimation is a cornerstone of power system control-center operations, and its robust operation is increasingly a cyber-physical security concern as modern grids become more digitalized and communication-intensive. Neural network-based approaches have gained attention as alternatives to conven...

---

### 20. [On-Device Vision Training, Deployment, and Inference on a Thumb-Sized Microcontroller](https://arxiv.org/abs/2604.23012)

**Authors**: Jeremy Ellis  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.23012v1  

#### Abstract
This paper presents a complete, end-to-end on-device vision machine learning pipeline, comprising data acquisition, two-layer CNN training with Adam optimization, and real-time inference, executing entirely on a microcontroller-class device costing $15-40 USD. Unlike cloud-based workflows that requi...

---

### 21. [Efficient VQ-QAT and Mixed Vector/Linear quantized Neural Networks](https://arxiv.org/abs/2604.23172)

**Authors**: Terry Gou, Puneet Gupta  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.23172v1  

#### Abstract
In this work, we developed and tested 3 techniques for vector quantization (VQ) based model weight compression. To mitigate codebook collapse and enable end-to-end training, we adopted cosine similarity-based assignment. Building on ideas from attention-based formulations in Differentiable K-Means (...

---

### 22. [Agentic Adversarial Rewriting Exposes Architectural Vulnerabilities in Black-Box NLP Pipelines](https://arxiv.org/abs/2604.23483)

**Authors**: Mazal Bethany, Kim-Kwang Raymond Choo, Nishant Vishwamitra, Peyman Najafirad  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.23483v1  

#### Abstract
Multi-component natural language processing (NLP) pipelines are increasingly deployed for high-stakes decisions, yet no existing adversarial method can test their robustness under realistic conditions: binary-only feedback, no gradient access, and strict query budgets. We formalize this strict black...

---

### 23. [LLM-Guided Agentic Floor Plan Parsing for Accessible Indoor Navigation of Blind and Low-Vision People](https://arxiv.org/abs/2604.23970)

**Authors**: Aydin Ayanzadeh, Tim Oates  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.23970v1  

#### Abstract
Indoor navigation remains a critical accessibility challenge for the blind and low-vision (BLV) individuals, as existing solutions rely on costly per-building infrastructure. We present an agentic framework that converts a single floor plan image into a structured, retrievable knowledge base to gene...

---

### 24. [Stabilizing Efficient Reasoning with Step-Level Advantage Selection](https://arxiv.org/abs/2604.24003)

**Authors**: Han Wang, Xiaodong Yu, Jialian Wu, Jiang Liu, Ximeng Sun, Mohit Bansal, Zicheng Liu  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.24003v1  

#### Abstract
Large language models (LLMs) achieve strong reasoning performance by allocating substantial computation at inference time, often generating long and verbose reasoning traces. While recent work on efficient reasoning reduces this overhead through length-based rewards or pruning, many approaches are p...

---

### 25. [OS-SPEAR: A Toolkit for the Safety, Performance,Efficiency, and Robustness Analysis of OS Agents](https://arxiv.org/abs/2604.24348)

**Authors**: Zheng Wu, Yi Hua, Zhaoyuan Huang, Chenhao Xue, Yijie Lu, Pengzhou Cheng, Zongru Wu, Lingzhong Dong, Gongshen Liu, Xinghao Jiang, Zhuosheng Zhang  
**Category**: cs.CL  
**Published**: 2026-04-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.24348v1  

#### Abstract
The evolution of Multimodal Large Language Models (MLLMs) has shifted the focus from text generation to active behavioral execution, particularly via OS agents navigating complex GUIs. However, the transition of these agents into trustworthy daily partners is hindered by a lack of rigorous evaluatio...

---

### 26. [A Differentiable Framework for Global Circulation Model Precipitation Bias Correction](https://arxiv.org/abs/2604.23045)

**Authors**: Kamlesh Sawadekar, Seth McGinnis, Peijun Li, Chaopeng Shen  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.23045v1  

#### Abstract
Systematic biases in Global Circulation Model (GCM) outputs limit their direct applicability in regional planning, necessitating bias correction. Correcting precipitation is particularly challenging due to its non-Gaussian distribution, intermittent nature, and non-linear extremes. However, traditio...

---

### 27. [A Layer Separation Optimization Framework for Cross-Entropy Training in Deep Learning](https://arxiv.org/abs/2604.23225)

**Authors**: Yaru Liu, Michael K. Ng, Yiqi Gu  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.23225v1  

#### Abstract
This paper investigates the deep learning optimization problem with softmax cross-entropy loss. We propose a layer separation strategy to alleviate the strong nonconvexity encountered during training deep networks. For cross-entropy models with fully connected and convolutional neural networks, we i...

---

### 28. [IMPA-Net: Meteorology-Aware Multi-Scale Attention and Dynamic Loss for Extreme Convective Radar Nowcasting](https://arxiv.org/abs/2604.24224)

**Authors**: Haofei Cui, Guangxin He, Juanzhen Sun, Jingjia Luo, Haonan Chen, Xiaoran Zhuang, Mingxuan Chen, Xian Xiao  
**Category**: cs.LG  
**Published**: 2026-04-28  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.24224v1  

#### Abstract
Short-range prediction of convective precipitation from weather radar observations is essential for severe weather warnings. However, deep learning models trained with pixel-wise error metrics tend to produce overly smooth forecasts that suppress intense echoes critical for hazard detection. This is...

---

### 29. [Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines](https://arxiv.org/abs/2604.23178)

**Authors**: Sadman Kabir Soumik  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.23178v1  

#### Abstract
LLM-as-a-Judge has become the dominant paradigm for evaluating language model outputs, yet LLM judges exhibit systematic biases that compromise evaluation reliability. We present a comprehensive empirical study comparing nine debiasing strategies across five judge models from four provider families ...

---

### 30. [Discovering Agentic Safety Specifications from 1-Bit Danger Signals](https://arxiv.org/abs/2604.23210)

**Authors**: V\'ictor Gallego  
**Category**: cs.AI  
**Published**: 2026-04-28  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.23210v1  

#### Abstract
Can large language model agents discover hidden safety objectives through experience alone? We introduce EPO-Safe (Experiential Prompt Optimization for Safe Agents), a framework where an LLM iteratively generates action plans, receives sparse binary danger warnings, and evolves a natural language be...

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
