# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-04 08:13:07 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Space Network of Experts: Architecture and Expert Placement](https://arxiv.org/abs/2605.00515)

**Authors**: Zhanwei Wang, Huiling Yang, Min Sheng, Khaled B. Letaief, Kaibin Huang  
**Category**: cs.DC  
**Published**: 2026-05-04  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2605.00515v1  

#### Abstract
Leveraging continuous solar energy harvesting at high efficiency, space data centers are envisioned as a promising platform for executing energy-intensive large language models (LLMs). Recognizing this advantage, space and AI conglomerates (e.g., SpaceX, Google) are actively investing in this vision...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Space Network of Experts: Architecture and Expert Placement*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**在低地球轨道（LEO）卫星网络上分布式部署大规模混合专家模型（MoE）时面临的高延迟挑战**。由于单个卫星的计算、存储资源有限，且星间激光链路（ISL）受轨道运动和空间环境影响而具有动态拓扑和高传播延迟，传统的模型放置策略无法有效支持低延迟的自回归推理（如大语言模型 token 生成）。

### 提出的新方法与新思路
作者提出了 **Space-XNet** 框架，其核心是**两层专家放置策略**，旨在最小化端到端（E2E）token 生成延迟：

1. **Layer Placement（层级放置）**  
   利用 LEO 卫星星座沿轨道方向形成的**环状通信模式**，将整个卫星网络按轨道方向划分为多个子网（subnet），每个子网承载一个 MoE 层。这种“环形流水线”设计使得最后一层的输出可以直接回传至第一层，契合自回归推理的数据流特性。

2. **Intra-layer Expert Placement（层内专家放置）**  
   在每一层内部，优化专家（expert）到具体卫星的映射关系。提出了一种**基于激活概率与路径延迟匹配的最优放置原则**：
   > **频繁被激活的专家应分配给预期路径延迟较低的卫星。**

   为此，论文定义了“期望路径延迟”（expected path latency）作为优化目标，并通过理论证明得出闭式解：将专家按**激活概率降序排列**，卫星按**期望路径延迟升序排列**，然后一一对应映射即可实现最优。

### 相比现有方法的优势
- **首次系统解决 MoE 模型在动态空间网络中的放置问题**，填补了地面数据中心与边缘网络之外的研究空白。
- **架构与算法协同设计**：不仅提出放置策略，还设计了适配空间特性的 Space-XNet 架构（如 gateway 卫星居中部署、token 路由协议等）。
- **理论可证最优性**：在合理的建模假设下，推导出具有直观解释且易于实现的排序匹配规则。
- **适应性强**：方法不依赖特定星座类型（如 Walker、Rosette），可通过预计算期望路径延迟推广。

---

## 2. 核心实验方法和设置

### 数据集
使用 **8 个标准英文推理与问答数据集**进行评估，均来自 `lm-evaluation-harness` 框架：
- OpenBookQA
- PIQA
- ARC-E
- ARC-C
- WinoGrande
- BoolQ
- SciQ
- HellaSwag

这些数据集用于模拟真实 LLM 推理负载，测试不同任务下的平均 token 生成延迟。

### 实验设置
- **卫星网络**：
  - 极地轨道 LEO 星座，共 33 个轨道面，每面 32 颗卫星 → 总计 **1056 颗卫星**
  - 轨道高度：550 km，倾角 87°
  - 时间划分：200 个时间槽（time slots），模拟拓扑动态变化
  - ISL 模型：考虑指向/捕获/跟踪（PAT）能力限制（角速度阈值 0.12 rad/s）和空间天气导致的链路中断（生存概率 95%）
- **MoE 模型配置**：
  - 模型：**LLaMA-MoE-3.5B**（约 3.5B 激活参数）
  - 结构：32 个 MoE 层，每层 8 个专家，Top-2 激活
  - 单次前向计算量：36.3 TFLOPs（序列长度 4096）
- **评估指标**：
  - 主要指标：**E2E token generation latency**（端到端 token 生成延迟，单位：秒）
  - 辅助分析：各层推理延迟、消融实验对比

### 基线方法对比
| 基线方法 | 描述 |
|--------|------|
| **RandPlace** | 所有专家和 gateway 随机分配到卫星 |
| **RandIntra** | 分层子网划分（ring-based layer placement），但层内专家随机分配 |
| **RandIntra-CG** | 同上，且 gateway 放置在子网中心（central gateway），仅专家仍随机 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | 平均 E2E Token Generation Latency (s) |
|------|-------------------------------------|
| RandPlace | ~5.29 |
| RandIntra | ~4.15 |
| RandIntra-CG | ~3.35 |
| **Space-XNet (Ours)** | **~1.05** |

> ✅ **Space-XNet 实现至少 3 倍以上的延迟降低**，相比最强基线 RandIntra-CG 也有超过 **3 倍提升**。

### 与基线方法的对比结果
- **RandPlace → RandIntra**：延迟下降约 21%，说明**分层子网划分本身就能显著改善通信局部性**。
- **RandIntra → RandIntra-CG**：延迟再降约 19%，表明**gateway 居中部署能进一步减少路由跳数**。
- **RandIntra-CG → Space-XNet**：延迟骤降超 68%，验证了**激活感知的专家放置策略的巨大增益**。

### 消融实验结果
- 图 6(a) 显示，Space-XNet 不仅整体延迟最低，且**各层延迟方差更小**，说明其瓶颈更均衡。
- 图 6(b) 综合八项任务的结果一致显示 Space-XNet 全面领先。
- 参数敏感性分析（图 7）表明：
  - 随着轨道高度增加，所有方法延迟上升，但 Space-XNet 增幅最小；
  - 星座规模越大，Space-XNet 可选候选卫星越多，性能越优；而随机方法反而可能因搜索空间变大而恶化；
  - 在不同链路存活率和 ISL 跟踪能力下，Space-XNet 始终保持稳定优势。

---

## 4. 关键结论和发现

### 主要发现
1. **空间 AI 必须进行“模型-网络”协同设计**：不能直接套用地面数据中心或边缘网络的放置策略，必须考虑空间特有的高延迟、动态拓扑和资源约束。
2. **环形拓扑可用于构建高效流水线**：利用 LEO 星座天然的环状结构进行 MoE 层划分，可极大提升自回归推理效率。
3. **专家激活频率与路径延迟之间存在强耦合关系**：高频专家应优先部署在低延迟路径节点上，这是降低 E2E 延迟的关键。
4. **理论最优策略具备实用价值**：尽管模型复杂，但最终导出的“排序后对齐”策略简单高效，适合实际部署。

### 方法的局限性
- 当前假设为“一星一专家”，未考虑多专家共存场景下的计算竞争。
- 依赖于对专家激活分布的事先估计（可通过训练阶段统计获得）。
- 期望路径延迟需离线计算或周期更新，在极端快速变化环境下可能存在滞后。
- 未联合优化路由策略，当前采用最短路径路由。

### 未来工作方向
- 扩展至 **multi-expert satellite** 场景，研究计算与通信之间的权衡（propagation-compute tradeoff）。
- 设计 **link-state-aware token routing** 策略，增强对突发链路中断的鲁棒性。
- 探索 **跨层联合优化**（placement + scheduling + routing）框架。
- 将 Space-XNet 架构扩展至 **Walker 星座、GEO-LEO 混合网络**等多样化拓扑。
- 研究如何在轨动态调整专家分布以适应任务迁移或负载漂移。

--- 

> 📌 **总结一句话**：  
> *Space-XNet 通过“环形分层 + 激活感知专家映射”的协同设计，在千星规模 LEO 网络上实现了 MoE 模型推理延迟的三倍以上压缩，为构建高效的空间 AI 基础设施提供了可行路径。*

</details>

---

### 2. [AGoQ: Activation and Gradient Quantization for Memory-Efficient Distributed Training of LLMs](https://arxiv.org/abs/2605.00539)

**Authors**: Wenxiang Lin, Juntao Huang, Luhan Zhang, Laili Li, Xiang Bao, Mengyang Zhang, Bing Wang, Shaohuai Shi  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.00539v1  

#### Abstract
Quantization is a key method for reducing the GPU memory requirement of training large language models (LLMs). Yet, current approaches are ineffective for 4-bit activations and 8-bit gradients, which would easily cause slow convergence or accuracy loss. To address this, we introduce AGoQ, incorporat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AGoQ: Activation and Gradient Quantization for Memory-Efficient Distributed Training of LLMs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大语言模型（LLMs）训练面临严重的 **GPU memory bottleneck**，尤其是在激活值（activations）和梯度（gradients）存储方面。尽管已有量化技术（如 8-bit 激活或梯度压缩），但直接应用 **4-bit 激活** 或 **8-bit 梯度存储** 容易导致训练收敛缓慢或精度下降。

现有方法存在以下不足：
- **Jetfire** 和 **COAT** 支持 8-bit 激活，但无法扩展到更低比特（如 4-bit）；
- 多数梯度压缩仅用于通信阶段，梯度仍以 FP32 存储，内存节省有限；
- 缺乏对不同层类型和 pipeline 阶段的差异化处理，统一量化策略会引入过大误差。

---

### 提出的新方法：AGoQ
作者提出 **AGoQ**（Activation and Gradient Quantization），一种面向分布式 LLM 训练的高效量化系统，包含两大核心技术：

#### （1）Layer-Aware Activation Quantization (LAAQ)
- **核心思想**：根据不同层类型（Attention, FFN, RMSNorm 等）及其在 Pipeline Parallelism 中的位置，动态分配激活值的量化比特宽度。
- **关键技术细节**：
  - 对于 **RMSNorm、SiLU & Multiply、FFN** 等模块，采用 **4-bit block-wise FP4 quantization**；
  - 对于 **Attention 模块中的 Q/K/V 投影输出**，不进行量化（因其梯度误差放大严重）；
  - 引入 **Dynamic Bit-width Compensation (DBCA-PP)**：利用 pipeline 各 stage 内存占用不均的特点，在内存较空闲的设备上使用更高 bit-width 来补偿精度损失，从而实现“近似 4-bit”平均存储。

#### （2）Precision-Preserved Gradient Quantization (QuanGrad)
- **目标**：将梯度全程以 **8-bit (FP8)** 存储并完成 All-Reduce 通信，同时避免溢出和精度损失。
- **关键技术设计**：
  - **本地累积时高精度加法**：每次 mini-batch 的局部梯度先 dequantize 到 FP16 进行累加，再 quantize 回 FP8 更新主梯度；
  - **All-Reduce 分解为 All-to-All + All-Gather**：
    - 先通过 All-to-All 发送 FP8 数据；
    - 各设备 dequantize 到 FP32 执行本地 reduce；
    - 再 quantize 回 FP8 并执行 All-Gather；
    - 避免了 Reduce-Scatter 中 FP8 加法可能引发的溢出问题。

---

### 相比现有方法的优势
| 方法 | 激活位宽 | 梯度存储 | 梯度通信 | 是否支持 4-bit 激活 | 是否端到端 8-bit 梯度 |
|------|----------|-----------|------------|------------------------|----------------------------|
| Megatron-LM | BF16 | FP32 | FP32 All-Reduce | ❌ | ❌ |
| COAT | ~8-bit | FP8 | FP8 All-Reduce | ❌ | ✅ |
| AGoQ (**本文**) | **~4-bit** | **FP8** | **FP8 All-Reduce (精度保持)** | ✅ | ✅ |

- **更极致的内存压缩**：激活内存减少约 3 倍，梯度内存减少 75%；
- **更高的训练吞吐**：无需牺牲收敛性即可提速达 1.34×；
- **兼容性强**：可与 8-bit Adam（如 bnb.optim）等优化器量化方案结合。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **预训练语料**：
  - OpenWebText（用于 LLaMA2-7B 和 LLaMA3-8B 的 pretraining）
- **下游任务评估**：
  - ARC-Challenge/Easy
  - HellaSwag
  - PIQA
  - SciQ
  - Winogrande

> 注：主要用于 zero-shot accuracy 评估，验证模型泛化能力。

---

### 实验设置
| 参数 | 设置 |
|------|------|
| **硬件平台** | 64-GPU 集群（NVIDIA A6000，200Gbps InfiniBand）；部分实验使用 Huawei Ascend 910 NPU |
| **软件环境** | Ubuntu-20.04, CUDA-12.1, PyTorch-2.1.2, NCCL-2.18.5 |
| **模型规模** | LLaMA2-7B, LLaMA3-8B, LLaMA2-13B, CodeLLaMA-34B, OLMo-1B |
| **序列长度** | 最高达 80K tokens |
| **并行策略** | TP=8, PP=1~8, DP 自适应配置 |
| **批大小** | Global Batch Size: 16~64, Micro-batch Size: 1 |

---

### 评估指标
- **内存消耗**（Memory Consumption）：峰值 GPU/NPU 显存占用（MB/GB）
- **训练速度**（Throughput）：每秒处理的样本数（samples/sec）或单 step 时间（ms）
- **收敛性**（Convergence）：pretraining loss 曲线对比
- **下游准确率**（Zero-shot Accuracy）：多个 NLP 推理任务上的表现
- **通信延迟**：All-Reduce 等集体操作耗时分析

---

### 基线方法对比
- **Megatron-LM**（BF16 baseline）
- **Megatron-LM + ZeRO-1/2/3**
- **DeepSpeed**（含 ZeRO 系列）
- **COAT**（FP8 激活与优化器状态压缩）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **最大内存降低** | ↓ **52%** vs. Megatron-LM（GPU 上从 46.1GB → 22.3GB） |
| **最高训练加速比** | ↑ **1.34×** vs. Megatron-LM / ZeRO-1（LLaMA2-13B @ 80K seq len） |
| **平均加速比** | 1.23× vs. Megatron-LM, 1.19× vs. ZeRO-1（跨 372 次实验） |
| **激活内存压缩** | 从 28U → **7.75U**（↓ 72%） |
| **梯度内存压缩** | 相比 COAT ↓ **75%** |
| **端到端速度提升 vs. COAT** | **1.1×**（且 COAT 在 32K 出现 OOM） |

---

### 与基线方法的对比结果

#### （1）内存对比（Table 4）
| 方法 | GPU Memory (LLaMA2-13B) |
|------|--------------------------|
| Megatron-LM | 46.1 GB |
| ZeRO-1 ("O") | 37.7 GB |
| AGoQ ("A+O+G") | **22.3 GB** |

> AGoQ 实现 **53% 峰值内存下降**。

#### （2）训练时间对比（Table 2）
在 LLaMA2-13B 上，随着序列长度增加，AGoQ 优势愈发明显：

| Seq Len | Megatron-LM (ms) | AGoQ (ms) | Speedup |
|---------|------------------|-----------|---------|
| 32K     | 37,635           | 36,568    | 1.03×   |
| 64K     | 104,444          | 82,519    | 1.27×   |
| **80K** | **149,667**      | **111,422** | **1.34×** |

#### （3）vs. COAT（Table 3）
| Seq Len | Method | Time (ms) | Memory (MB) |
|---------|--------|-----------|-------------|
| 24K     | COAT   | 6291      | 94,100      |
|         | AGoQ   | 6161      | **66,852**  |
| 32K     | COAT   | OOM       | —           |
|         | AGoQ   | 8076      | 86,012      |

> AGoQ 不仅避免 OOM，还实现 **1.1× 更快的端到端训练速度**。

---

### 消融实验结果

#### （1）模块贡献分析（Fig. 9）
逐步加入量化模块显著降低迭代时间：
- 仅启用 **8-bit Optimizer ("O")**：轻微提速
- 加入 **Activation Quantization ("A+O")**：进一步提速
- 完整 **A+O+G (AGoQ)**：最快，证明梯度量化贡献最大

#### （2）DBCA-PP 消融（Table 8）
启用 Dynamic Bit-width Compensation 后，多数任务 accuracy 提升：
- `arc_c`: ↑ 0.68%
- `piqa`: ↑ 0.27%
- `sciq`: ↑ 1.6%

> 表明动态 bit-width 分配有助于维持模型精度。

#### （3）Kernel Fusion 加速效果（Table 7）
融合 quantization/dequantization 与 GEMM 可带来 **1.03–1.11×** 局部加速，平均 **1.07×**。

#### （4）长序列内存节省（Table 11）
在极端长序列下，AGoQ 内存优势更加突出：
- 32K 序列：↓ **66%**（48,606 MB → 16,594 MB）
- 64K 序列：↓ **59%**

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **4-bit 激活量化是可行的**，但必须基于层类型和 pipeline 位置进行精细化控制（layer-aware）；
2. ✅ **Attention 模块对量化极其敏感**，其输入不应量化，否则梯度误差急剧上升（理论分析与实测一致）；
3. ✅ **8-bit 梯度可以全程存储和通信**，只要合理设计累积路径（dequantize-add-quantize）和 All-Reduce 分解机制；
4. ✅ **pipeline memory imbalance 是可利用资源**，DBCA-PP 能有效提升低比特下的稳定性；
5. ✅ **AGoQ 与现有优化器量化完全兼容**，可叠加使用实现更大收益。

---

### 方法的局限性
- **依赖硬件支持 FP8**：目前仅 NVIDIA Hopper 架构和 Ascend NPU 支持原生 FP8 运算；
- **需定制 CUDA kernel**：实现 kernel fusion 和通信优化需要底层开发投入；
- **理论误差分析假设理想条件**：实际中可能存在非线性累积效应未被建模；
- **尚未测试超大规模集群 (>64 GPUs)**：扩展性有待进一步验证。

---

### 未来工作方向
- 将 AGoQ 扩展至 **fully sharded training**（如 FSDP）场景；
- 探索 **sub-4-bit 激活量化**（如 3-bit 或混合精度）的可能性；
- 结合 **sparsity** 与量化，构建更高效的训练栈；
- 开发自动化的 **bit-width allocation policy**，适配不同模型架构；
- 支持更多 **non-transformer 架构**（如 Mamba、RWKV）。

--- 

> 📌 **总结一句话**：  
> **AGoQ 通过 layer-aware 激活量化与 precision-preserving 梯度量化，在几乎无损收敛性的前提下，实现了高达 52% 的内存节省和 1.34× 的训练加速，为大规模 LLM 训练提供了实用的量化解决方案。**

</details>

---

### 3. [Tempus: A Temporally Scalable Resource-Invariant GEMM Streaming Framework for Versal AI Edge](https://arxiv.org/abs/2605.00536)

**Authors**: M. Grailoo, J. N\'u\~nez-Y\'a\~nez  
**Category**: cs.DC  
**Published**: 2026-05-04  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.00536v1  

#### Abstract
Scaling laws for Large Language Models (LLMs) establish that model quality improves with computational scale, yet edge deployment imposes strict constraints on compute, memory, and power. Since General Matrix Multiplication (GEMM) accounts for up to 90\% of inference time, efficient GEMM acceleratio...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TEMPUS: A Temporally Scalable Resource-Invariant GEMM Streaming Framework for Versal AI Edge**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代大型语言模型（LLMs）在边缘设备上的部署面临严峻挑战。尽管其性能随计算规模提升而增强，但边缘平台受限于**计算资源、内存容量和功耗预算**。其中，**General Matrix Multiplication (GEMM)** 占据了高达90%的推理时间，是性能瓶颈所在。

现有的 State-of-the-Art (SOTA) 加速框架（如 ARIES、CHARM 2.0）依赖于**空间扩展（spatial scaling）**，即通过将任务分布到数百个 AIE 核心上来最大化吞吐量。然而，这种方法在资源受限的边缘 SoC（如 AMD Versal AI Edge VE2302）上存在以下问题：
- **物理实现失败**：布线拥塞导致 Place-and-Route 失败；
- **资源饱和**：大量占用 URAM、DSP 和 PLIO 资源，限制了非 GEMM 内核（如 Softmax、LayerNorm）的集成；
- **能效低下**：高功耗不符合边缘设备可持续运行需求。

---

### **提出的新方法与创新思路**
论文提出了 **TEMPUS** —— 一种**资源不变的时间可扩展 GEMM 流水线框架**，专为 Versal AI Edge 平台设计。其核心思想是：  
> **以“时间扩展”替代“空间扩展”，固定硬件资源使用，实现高效、可持续的 GEMM 加速。**

#### **主要创新点如下：**

1. ✅ **Resource-Invariant Frugality Framework（资源不变节俭架构）**
   - 使用固定的 **16 个 AIE-ML 核心** 构成计算块，不随矩阵大小增加而扩展；
   - 利用**迭代图执行（iterative AIE-ML graph execution）** 和**算法级数据分块与复制（tiling & replication）** 实现对大规模 GEMM 的支持；
   - 在 Programmable Logic (PL) 中仅使用轻量级 FIFO 和固定缓冲区，**完全避免使用 URAM 和 DSP**（利用率 0.00%），保留 PL 资源用于其他关键算子。

2. ✅ **Platform-Aware Utility (PAU) 指标**
   - 提出一个综合评估指标 PAU，归一化性能相对于平台物理潜力（core count, power, I/O, peak throughput）的表现；
   - 公式定义为：
     $$
     \text{PAU} = \frac{\text{TOPS}}{\text{Cores} \times \text{Power(W)} \times \text{PLIO} \times \text{Theoretical Peak}}
     $$
   - 更公平地比较不同规模设备上的架构效率。

3. ✅ **高效率的 Compute-Transfer 重叠机制**
   - 利用高速 **Cascade Stream 接口**（512-bit 宽）进行低延迟部分和约简（partial sum reduction），达到 **Initiation Interval (II) = 1**；
   - 设计**无死锁 DATAFLOW 协议**，结合 **packet switching 与 broadcast circuit switching**，最大化 PLIO 复用并隐藏通信延迟。

4. ✅ **分析建模指导参数优化**
   - 建立理论模型推导关键调度参数：
     - `GRAPH_ITER_CNT`：控制时间扩展次数；
     - `Kernel Size (DIM)`：决定微内核尺寸；
     - `Replication Factor`：优化数据复用。

---

### **相比现有方法的优势**
| 维度 | 现有 SOTA 方法（如 ARIES） | TEMPUS |
|------|----------------------------|--------|
| 扩展方式 | 空间扩展（数百核） | 时间扩展（固定16核） |
| 资源利用 | 高 URAM/DSP/BRAM 占用（>76%） | **0.00% URAM/DSP**，仅轻量 FIFO |
| 功耗 | 高（~76W） | **10.677W 总芯片功耗** |
| 可持续性 | 不适用于边缘设备 | 支持异构协同（保留 PL 资源） |
| 形状适应性 | 对窄形/宽形 GEMM 效率骤降 | **形状无关（shape-agnostic）高效** |
| 实际部署可行性 | 存在 PnR 失败风险 | 成功部署于 VE2302 边缘芯片 |

---

## **2. 核心实验方法和设置**

### **实验平台**
- **硬件平台**：AMD Versal AI Edge **VE2302 ACAP**（XCVE2302-1LSESFVA784-E）
- **AI Engine 类型**：AIE-ML v1
- **核心数量**：共 34 个 AIE-ML 核心 → 实际使用 **16 个**
- **时钟频率**：PL 工作在 312.5 MHz
- **工具链**：AMD Vitis™ 2024.1

### **测试负载**
- 主要测试 **INT16/INT32 精度下的 GEMM 运算**
- 矩阵规模从 $32^3$ 到 $1024^3$，覆盖 **32,768× 的运算量增长**
- 包括典型 LLM 结构件中的矩形 GEMM：
  - 解码投影层（narrow shapes）：如 `8×1024×1024`
  - 注意力头（fragmented shapes）：如 `128×768×64`
  - FFN 层（wide shapes）：如 `768×3072×768`

### **评估指标**
| 指标类别 | 具体指标 |
|---------|----------|
| **性能** | Throughput (GOPS/TOPS), Latency (ms), II=1 是否达成 |
| **资源** | LUT, BRAM, URAM, DSP, CLB Registers 利用率 |
| **功耗** | AIE Power, Memory Power, Total On-Chip Power |
| **效率** | Platform-Aware Utility (PAU), T/C (TOPS/core), T/P (TOPS/W) |
| **可扩展性** | 工作负载扩展下的延迟增长趋势 |

### **基线方法对比**
与多个 SOTA 框架进行横向对比：
- **ARIES**：基于 MLIR 的编译流程，大规模空间扩展（352 cores）
- **CHARM 2.0**：异构分区加速器，288 cores
- **AUTOMM**：资源感知 DSE，288 cores
- **AUTOSA**：多面体编译器生成脉动阵列

> ⚠️ 注：这些基线大多运行在高端设备（VCK190/VE2802）上，拥有更多核心和更高带宽，因此采用 **PAU 指标** 来实现跨平台公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（1024³ INT16 GEMM）**

| 指标 | 数值 | 说明 |
|------|------|------|
| **AIE Cores Used** | 16 (47%) | 固定小规模核心块 |
| **Achieved Throughput** | **607 GOPS** | 实测有效吞吐 |
| **Core Computation Latency** | 3.537 ms | 实际计算耗时 |
| **Total On-Chip Power** | **10.677 W** | 极低功耗 |
| **Energy Efficiency (Total)** | 56.87 GOPS/W | 系统级能效 |
| **URAM Utilization** | **0.00%** | 完全未使用 |
| **DSP Utilization** | **0.00%** | 完全未使用 |

> 💡 表明：即使只用了约一半的 AIE 核心，也能在极低资源消耗下实现高性能。

---

### **与基线方法的对比结果（Table VI）**

| 框架 | Cores | TOPS | Power (W) | URAM% | PAU Factor |
|------|-------|-------|-----------|--------|-------------|
| **TEMPUS (Temporal)** | 16 | 0.607 | 10.677 | 0.00% | **211.2×** |
| ARIES (Spatial) | 352 | 15.86 | 76.30 | 76.03% | 1.0× |
| CHARM 2.0 | 288 | 10.03 | 64.80 | 82.94% | 1.2× |

#### **核心优势体现为：**
- **211.2× 更高的 Platform-Aware Utility (PAU)**：证明其在资源受限场景下具有压倒性架构优势；
- **22.0× Core Frugality**：单位性能所需核心数少两个数量级；
- **7.1× Power Frugality**：功耗效率显著提升；
- **6.3× I/O Frugality**：PLIO 资源复用率极高；
- **0.00% URAM/DSP 占用**：为 Softmax、LayerNorm 等留出充足 PL 资源。

---

### **消融实验与扩展性分析**

#### **(1) Tile Dimension (DIM) 缩放影响（Table III）**
- 固定 $512^3$ 工作负载，调整微内核大小 DIM；
- 当 DIM 从 4 增加到 128，**吞吐量提升达 10.5×**；
- 表明：更大的本地内存允许更大 tile，减少迭代次数，提高效率；
- **瓶颈在于 AIE-ML 本地内存上限（INT16 下最大 DIM=128）**。

#### **(2) 工作负载扩展分析（Table IV）**
- 从 $32^3$ 到 $1024^3$，操作数增长 **32,768×**；
- 实际延迟仅增长 **6.8×**（从 ~0.4ms 到 3.537ms）；
- 显示出优秀的**开销摊销能力（amortization of fixed overheads）**；
- 在 $512^3$ 且 DIM=128 时接近理想扩展；但在 $1024^3$ 时因 DIM 下降至 64 导致非线性延迟上升。

#### **(3) 形状无关性验证（Table VIII）**
- 在多种典型 LLM 矩形 GEMM 上测试（narrow/wide/fragmented）；
- 与等效立方体延迟对比，差异极小；
- 例如：
  - `8×1024×1024` vs `192³`：延迟分别为 1.527ms vs 1.637ms；
  - `128×768×3072` vs `768³`：1.258ms vs 1.637ms；
- 证明 TEMPUS **不受矩阵形状影响，具备高度通用性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **时间扩展优于空间扩展**：在边缘设备上，固定资源+时间迭代的方式比盲目堆核更可持续、更高效；
2. ✅ **资源不变性至关重要**：保持 URAM/DSP 利用率为 0.00%，确保系统可扩展性和异构协同能力；
3. ✅ **PAU 是衡量边缘 AI 架构优劣的有效指标**：传统 TOPS 指标误导性强，必须结合资源成本评估；
4. ✅ **高带宽 Cascade + DATAFLOW 流水线可实现 II=1**：保证流水线满载，充分发挥硬件潜力；
5. ✅ **TEMPUS 是 shape-agnostic 的**：适用于各种 LLM 子模块中的矩形 GEMM，无“利用率崩溃”问题。

---

### **方法的局限性**
1. ❗ **受限于 AIE-ML 本地内存容量**：最大 tile 尺寸（DIM）受制于每个核心的内存大小，无法进一步扩大以提升效率；
2. ❗ **当前软件栈限制精度支持**：虽硬件支持 INT4/INT8/BFLOAT16，但受限于 Vitis 工具链和 Xilinx DSP Library，目前仅报告 INT16/INT32 结果；
3. ❗ **绝对峰值低于大规模空间方案**：虽然 PAU 更高，但总吞吐（0.607 TOPS）仍低于 ARIES（15.86 TOPS），不适合数据中心级部署；
4. ❗ **依赖定制 HLS kernel 和图编程**：开发门槛较高，自动化程度有待提升。

---

### **未来工作方向**
1. 🔮 **探索更大本地内存配置或压缩技术**：突破 DIM 上限，进一步降低迭代次数；
2. 🔮 **扩展至更多数据类型（INT8/BF16）**：适配主流 LLM 推理格式，提升实用性；
3. 🔮 **构建全自动编译流程**：将 TEMPUS 思想集成进 MLIR 或高层次综合工具，降低使用门槛；
4. 🔮 **应用于完整 LLM 推理流水线**：联合优化 GEMM + Softmax + LayerNorm + Attention，实现端到端加速；
5. 🔮 **推广至其他边缘 AI SoC 架构**：验证该“时间优先、资源节俭”范式是否具有普适性。

---

> 📦 **开源地址**：https://github.com/mgrailoo/Versal_AI_ML_Engines_GEMM  
> 代码已公开，便于复现与社区贡献。

</details>

---

### 4. [Conformalized Quantum DeepONet Ensembles for Scalable Operator Learning with Distribution-Free Uncertainty](https://arxiv.org/abs/2605.00330)

**Authors**: Purav Matlia, Christian Moya, Guang Lin  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2605.00330v1  

#### Abstract
Operator learning enables fast surrogate modeling of high-dimensional dynamical systems, but existing approaches face two fundamental limitations: quadratic inference complexity and unreliable uncertainty quantification in safety-critical settings. We propose Conformalized Quantum DeepONet Ensembles...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Conformalized Quantum DeepONet Ensembles for Scalable Operator Learning with Distribution-Free Uncertainty

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**神经算子学习**（operator learning）在实际应用中的两个根本性挑战：
- **可扩展性瓶颈**：传统 DeepONet 等模型依赖稠密矩阵乘法，推理复杂度为 $O(n^2)$，难以处理细粒度离散化场景；
- **不确定性量化不可靠**：现有方法缺乏严格的分布无关（distribution-free）覆盖保证，尤其在安全关键系统中风险较高。

此外，在量子机器学习背景下，**集成模型**（ensembles）虽能提升 epistemic uncertainty 估计能力，但其并行执行会线性增加量子比特资源需求，形成新的硬件瓶颈。

---

### 提出的新方法与新思路
作者提出 **Conformalized Quantum DeepONet Ensembles** 框架，融合三大核心技术：

#### （1）**可扩展的量子算子推理**
- 利用 **Quantum Orthogonal Neural Networks (QOrthoNNs)** 替代经典全连接层。
- 将前向传播复杂度从 $O(n^2)$ 降低至 $O(n)$，支持高效细粒度求解。
- 基于 Reconfigurable Beam Splitter (RBS) 门构建参数化量子电路（PQC），实现正交变换。

#### （2）**资源高效的混合与叠加集成架构**
为解决集成模型带来的量子资源开销问题，提出两种策略：
- **Hybrid Classical-Quantum Architecture**  
  将低频调用的子网络（如 branch net）替换为经典网络，保留高频 trunk net 的量子加速优势，显著减少噪声影响。
- **Superposed Parameterized Quantum Circuits (SPQCs)**  
  使用地址量子比特控制旋转操作，将多个集成成员编码进单一量子电路中，实现一次性状态准备、演化与测量，仅需 $n+1+\lceil\log L\rceil$ 个量子比特（而非 $L \cdot n$）。

#### （3）**分布无关的不确定性量化**
- 结合 **adaptive conformal prediction** 与集成输出，提供有限样本下的严格覆盖保证。
- 定义自适应非一致性分数（nonconformity score）：
  $$
  r_{ij} = \frac{|s_{ij} - \mu(u)(y_i)|}{\sigma(u)(y_i) + \epsilon}
  $$
- 利用校准集计算经验分位数 $q$，构造预测集：
  $$
  C(u,y) = \{ v : |v - \mu(u)(y)| \leq q \cdot \sigma(u)(y) \}
  $$
  满足 $\mathbb{P}(s \in C(u,y)) \geq 1-\alpha$。

---

### 相比现有方法的优势
| 维度 | 本工作 | 现有方法 |
|------|--------|----------|
| 推理效率 | $O(n)$ | $O(n^2)$（经典 DeepONet） |
| 不确定性保证 | 分布无关、数学严格 | 多为启发式或基于假设（如高斯先验） |
| 集成资源消耗 | 对数级空间增长（SPQC） | 线性增长（naive parallelism） |
| 适用场景 | 可扩展至大规模动态系统仿真 | 受限于计算成本与置信度可靠性 |

---

## 2. 核心实验方法和设置

### 数据集
实验涵盖两类任务：

#### （1）合成 PDE 任务
- **Antiderivative Operator**：输入函数 $v(x)$，输出其积分 $u(x)$，定义域 $x \in [0,1]$，初始条件 $u(0)=0$。
  - 输入采样自高斯随机场（GRF），协方差核为平方指数核。
- **Advection Equation**：一维平流方程 $\partial_t u + \partial_x u = 0$，周期边界条件。
  - 初始条件由 Exp-Sine-Squared 核生成 GRF。

#### （2）真实电力系统数据
- **Offline V-to-V**：电压到电压轨迹映射（瞬态到瞬态），用于事后分析。
- **Offline V-to-P**：电压到有功功率轨迹映射（跨物理量预测）。
- **Online V-to-V**：滑动窗口下的实时电压预测，测试非交换性数据下的鲁棒性。

---

### 实验设置与评估指标

#### 模型配置
- 使用 **Quantum DeepONet** 架构，branch 和 trunk 子网均采用 QOrthoNN。
- 层数、宽度见 Table 2 和 Table 4。
- 集成大小 $L=4$ 或 $8$。
- 激活函数：SiLU。
- 优化器：Adam，学习率衰减策略（Lambda decay）。

#### 量子模拟设置
- **理想模拟**：直接模拟等效的经典正交网络，避免状态向量指数开销。
- **噪声模型**：
  - 简化模型：单/双量子比特 depolarizing noise，强度 $\lambda \in \{0.0002, ..., 0.0008\}$。
  - 真实设备模型：基于 IBM Brisbane、Torino、Marrakesh 的 Qiskit Aer 噪声快照（含 readout error, T1/T2, gate error）。
- 错误缓解：利用 unary subspace 特性进行后选择（post-selection），丢弃非单位汉明权重的测量结果。

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Relative L2 Error (%)** | 预测与真值之间的相对误差 |
| **Coverage (%)** | 测试点落入 conformal 区间的比例（目标 90%） |
| **Average Width** | 预测区间平均宽度 |
| **Peak Uncertainty (Max Width)** | 最大预测区间宽度 |

#### 基线对比
- 无显式基线模型列出，但隐含对比对象包括：
  - 经典 DeepONet（理论复杂度更高）
  - 单一量子 DeepONet（无集成，uncertainty 不可靠）
  - 未使用 conformal 的集成方法（无分布无关保证）

---

## 3. 主要实验结果和性能指标

### 合成任务表现（Table 1）

| 实验 | Rel. L2 Err.(%) | Cov.(%) | Avg. Width | Peak Uncertainty |
|------|------------------|---------|------------|------------------|
| Antiderivative (L=4) | 0.46 | 88.40 | 0.004 | 0.044 |
| Antiderivative (L=8) | 0.46 | 92.13 | 0.005 | 0.080 |
| Advection (L=4) | 2.38 | 89.36 | 0.062 | 0.751 |
| Advection (L=8) | 2.28 | 88.99 | 0.053 | 0.621 |

✅ 所有任务均达到接近或超过目标覆盖率（90%），表明 conformal 框架有效。

---

### 真实电力系统任务（Table 3）

| 实验 | Rel. L2 Err.(%) | Cov.(%) | Avg. Width | Peak Uncertainty |
|------|------------------|---------|------------|------------------|
| Online V-to-V | 5.49 | 90.11 | 0.148 | 1.328 |
| Offline V-to-V | 12.49 | 89.66 | 0.345 | 2.691 |
| Offline V-to-P | 4.08 | 89.74 | 0.001 | 0.063 |

✅ 在真实世界电力动态建模中仍保持良好 coverage 与合理区间宽度。

---

### 关键消融实验结果

#### （1）SPQC 资源效率验证（Figure 7 & Table 5）
- SPQC 将 $L=4$ 集成所需量子比特从 20 减少到 **7**。
- 电路深度从串行执行的 $4\times95=380$ 控制在 **272**（得益于流水线优化）。
- CZ 门数量上升（因控制门增多），但其他门未线性增长。
- ✅ **SPQC 在精度、coverage、interval width 上完全匹配标准集成**，证明其有效性。

#### （2）Hybrid 架构噪声敏感性分析（Figure 6）
- 当 branch 网络更复杂时（antiderivative 任务），**classical-branch hybrid 表现最优**（最低 L2 error 与最窄区间）。
- classical-trunk hybrid 改善有限，说明应优先替换主导噪声源的子网络。
- ✅ 验证了 hybrid 设计原则：识别并替换主要噪声瓶颈模块。

#### （3）现实噪声下 coverage 鲁棒性（Figure 3）
- 在 IBM Brisbane/Torino/Marrakesh 噪声模型下，随着 shot 数增加，coverage 均稳定维持在 **≥90%**。
- 表明只要噪声特性在校准与测试阶段保持平稳，exchangeability 假设成立，conformal 保证依然有效。

#### （4）浅层电路中的“噪声诱导正则化”现象（Figure 4–5）
- 在某些浅层电路中，**更高的 depolarizing noise（$\lambda$）反而提升了 coverage 并缩小了区间宽度**。
- 可能源于噪声对过自信预测的抑制作用（类似 dropout 效果）。
- ⚠️ 此为经验观察，尚未理论解释。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **首次实现了兼具可扩展性与严格不确定性保证的量子算子学习框架**。
2. ✅ **SPQC 成功压缩集成模型至单电路，实现资源对数增长，突破硬件瓶颈**。
3. ✅ **Conformal prediction 在真实量子噪声下仍能维持目标覆盖率（≥90%）**，验证其工程可行性。
4. ✅ **Hybrid 架构可根据子网络频率差异灵活设计，平衡性能与噪声**。
5. 🔍 观察到浅层电路中“噪声有益于 uncertainty calibration”的反直觉现象，提示未来可探索主动噪声调控机制。

---

### 方法的局限性
1. ❗ **Exchangeability 假设限制在线部署**：时间序列滑动窗口破坏独立同分布假设，此时 coverage 仅为经验结果，无理论保证。
2. ❗ **当前验证局限于小规模电路**：受限于经典模拟开销，无法全面评估 >20 qubits 场景。
3. ❗ **SPQC 深度随 $L$ 线性增长**：虽然空间节省显著，但深度代价可能受当前 NISQ 设备相干时间限制。
4. ❗ **噪声漂移威胁 conformal 有效性**：若硬件噪声在校准后发生变化，则 exchangeability 被破坏。

---

### 未来工作方向
1. 🔄 发展适用于**非交换数据流**的 conformal 方法，如 **sequential conformal prediction** 或 **weighted calibration**。
2. 🔬 深入研究“噪声诱导正则化”机制，探索是否可通过可控噪声提升 uncertainty quality。
3. 💡 比较 hybrid 与 pure SPQC 在不同硬件成熟度下的性价比权衡。
4. 🚀 推动该框架向更大规模真实电力系统迁移，并在真实量子硬件上部署验证。

---

> **代码开源地址**：[https://github.com/purav-0000/conformalized-quantum-deeponet](https://github.com/purav-0000/conformalized-quantum-deeponet)

</details>

---

### 5. [Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding](https://arxiv.org/abs/2605.00342)

**Authors**: Lehan Pan, Ziyang Tao, Ruoyu Pang, Xiao Wang, Jianjun Zhao, Yanyong Zhang  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.00342v1  

#### Abstract
Tree-based speculative decoding accelerates autoregressive generation by verifying multiple draft candidates in parallel, but this advantage weakens for sparse Mixture-of-Experts (MoE) models. As the draft tree grows, different branches activate different experts, expanding the union of activated ex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在 **Mixture-of-Experts (MoE)** 模型中，传统的 **tree-based speculative decoding**（如 EAGLE-3）虽然能并行验证多个 draft token 以加速生成，但存在一个关键瓶颈：

- 不同的 draft 分支会激活不同的 **experts**，导致目标模型在一次验证过程中需要加载 **所有分支激活专家的并集（union of activated experts）**。
- 随着 draft tree 变大，该并集迅速扩张，显著增加 **target-side verification cost** 和内存带宽压力，反而抵消了加速收益。

因此，标准 speculative decoding 在 MoE 上的效率受限于 **验证开销的增长快于接受率提升**。

---

### 🆕 提出了什么新方法或新思路

作者提出 **EVICT**（Expert-aware Verification via Identifying Cost-effective Tree prefixes），一种 **无需训练、无超参数、且 lossless** 的自适应验证策略。

#### 核心思想：
> “Make every verified token count” —— 并非盲目扩大 draft tree 来最大化接受 token 数量，而是选择 **性价比最高（benefit-to-cost 最优）的验证前缀子树**。

#### 创新机制：
1. **Utility-Guided Tree Truncation**  
   在每一步解码时，动态决定应验证多少个 draft 节点：
   - 使用 **drafter 模型输出的概率** 估算每个节点被最终接受的可能性（即 `Score(v)`）。
   - 结合 **离线预采样的 verification cost profile C(k)**（k 个节点验证所需延迟）。
   - 定义效用函数（speculation utility）：
     $$
     U(k) = \frac{\text{CAR} \cdot \mathbb{E}[A(T_k)]}{C(k)}
     $$
     其中 $\mathbb{E}[A(T_k)]$ 是预期接受长度，CAR 是 autoregressive 解码每 token 延迟。
   - 选择使 $U(k)$ 最大的 $k^*$，仅验证前 $k^*$ 个高价值节点。

2. **系统级兼容设计（SGLang Integration）**
   - 支持 **CUDA graph capture**：预先为不同验证长度捕获多个 target graph，运行时直接调用。
   - 将在线决策逻辑（如 score 计算、argmax）**融合进 draft graph** 中，避免引入 CPU 控制开销。

---

### 🔍 相比现有方法的优势

| 方面 | EVICT vs. 现有方法 |
|------|------------------|
| **性能增益** | 平均比 SOTA 方法 **EAGLE-3 快 1.21×**，最高达 **2.35×** 超过 vanilla autoregressive decoding |
| **成本控制** | 显著减少不必要的 expert 激活，平均降低 **32.5% active experts** 和 **26.6% verification latency** |
| **通用性** | 无需训练、不修改模型结构、无超参数调节，适用于任意 MoE + speculative decoding 架构 |
| **系统友好** | 完全兼容高性能推理框架（如 SGLang），通过 graph fusion 避免 runtime overhead |

---

## 2. 核心实验方法和设置

### 📚 数据集

使用六个代表性 benchmark 进行综合评估：

| 数据集 | 任务类型 |
|--------|---------|
| **Alpaca** | 指令遵循 |
| **GSM8K** | 数学推理 |
| **HumanEval** | 代码生成 |
| **QA** (Natural Questions) | 开放问答 |
| **MT-Bench** | 多轮对话质量 |
| **CNN/DM** | 文本摘要 |

---

### ⚙️ 实验设置

- **模型架构**：基于三种 MoE backbone：
  - `Qwen3-30B-A3B`（30B 总参，3B 激活）
  - `Ling-flash-2.0`（103B 总参，6.1B 激活）
  - `Qwen3-235B-A22B`（235B 总参，22B 激活）

- **实现平台**：全部基于 **SGLang** 推理引擎，在 NVIDIA A100 GPU 上测试。

- **评估指标**：
  - **Decoding Speed (token/s)**：主性能指标
  - **Mean Accepted Tokens (MAT)**：衡量 draft 接受效率
  - **Verification Latency / Active Experts**：分析 MoE 特定开销

- **Baseline 对比方法**：
  - **Vanilla**：标准 autoregressive decoding
  - **Lookahead**：无辅助 draft model 的多分支 speculative 方法
  - **EAGLE-3**：当前最优 tree-based speculative decoding 方法
  - **+DDD**（Dynamic Depth Decoding）：基于 confidence 动态剪枝 depth，但仍固定验证预算

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1, 5, 6）

| 方法 | Qwen3-30B-A3B 平均速度 | 相对 Vanilla 加速比 |
|------|------------------------|--------------------|
| Vanilla | 144.46 token/s | 1.00× |
| EAGLE-3 | 176.34 token/s | 1.22× |
| **EVICT** | **202.82 token/s** | **1.40×** |

> 在其他两个模型上表现更优：
- **Ling-flash-2.0**: EVICT 达到 **1.59×** 加速（vs EAGLE-3 的 1.28×）
- **Qwen3-235B-A22B**: EVICT 达到 **1.80×** 加速（vs EAGLE-3 的 1.44×）

✅ **平均比 EAGLE-3 快 1.21×**

---

### 🔁 与基线方法对比结果

| 维度 | EVICT 表现 |
|------|----------|
| **vs EAGLE-3** | 严格超越所有 benchmark 和 temperature 设置下的性能 |
| **vs DDD** | 即便都采用动态策略，EVICT 因显式建模 **MoE verification cost** 而胜出 |
| **vs Lookahead** | 后者在 MoE 场景下甚至慢于 vanilla，说明高质量 draft + 成本感知缺一不可 |

📌 **特别优势场景**：
- 在 **GSM8K** 和 **HumanEval** 等结构化输出任务中，EVICT 提升尤为明显（相对 EAGLE-3 达 **1.23–1.27×**），因其能有效保留高置信路径。

---

### 🔍 消融实验结果

#### （1）框架开销影响（Figure 6）

| 实现方式 | EVICT vs EAGLE-3 性能 |
|--------|---------------------|
| PyTorch Eager | EVICT 反而略差（CPU 控制开销大） |
| SGLang Eager | 基本持平 |
| **SGLang + CUDA Graph** | **EVICT 显著领先** |

👉 结论：**必须结合 graph fusion 才能释放算法潜力**。

#### （2）是否考虑 cost-aware selection（Figure 7）

比较以下策略：
- EAGLE-3 ($p=1.0$)
- Score-coverage 截断（$p=0.7, 0.4$）
- DDD
- **EVICT（cost-aware）**

结果表明：
- 单纯依赖 drafter confidence 或覆盖率会导致次优截断；
- **只有联合建模 verification cost 的 EVICT 能稳定达到最高速度**；
- EVICT 无需手动设定阈值（hyperparameter-free），自动找到最优 $k^*$。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **MoE speculative decoding 的瓶颈是 verification cost，而非 draft 覆盖率**  
   单纯追求更多 accepted tokens 会导致专家激活爆炸，适得其反。

2. **“Every verified token must count”**  
   应优先验证那些 **预期收益远高于计算代价** 的 token，而不是贪多求全。

3. **fine-grained drafter signal + offline cost profiling = 高效自适应决策**  
   利用 draft 概率估计收益，结合实测延迟 profile 决策，可在不损失精度的前提下大幅提升效率。

4. **算法-系统协同设计至关重要**  
   若不能无缝集成到现代推理框架（如 SGLang），额外控制流将引入严重 overhead。EVICT 通过 **graph capture & fusion** 实现零额外 runtime 开销。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖稳定的 cost profile** | 假设 $C(k)$ 在不同 context 下近似恒定；极端负载变化可能影响准确性 |
| **对 drafter calibration 敏感** | 若 draft model 输出概率不准（poorly calibrated），$\mathbb{E}[A(T_k)]$ 估计偏差会影响决策 |
| **未优化 drafting 阶段本身** | 仍沿用 EAGLE-3 的 drafting 策略，未来可联合优化 drafting 与 verification |

---

### 🔮 未来工作方向

1. **动态 cost modeling**  
   引入 context-aware 的 cost estimator 替代静态 profile，适应更复杂场景。

2. **端到端 joint optimization of drafting & verification**  
   设计统一目标函数同时优化 draft tree 构造与验证策略。

3. **扩展至其他稀疏架构**  
   如 block-wise sparsity、channel-wise MoE 等，探索通用 cost-aware speculative framework。

4. **支持 high-concurrency serving**  
   当前聚焦 low-latency 单请求场景，未来可研究 batch-level 自适应调度。

---

> 💡 **一句话总结**：  
> **EVICT 通过“效用驱动”的自适应验证机制，在 MoE 模型上实现了高效、无损、免调参的 speculative decoding 新范式，揭示了“少而精”的验证优于“多而泛”的传统策略。**

</details>

---

### 6. [Eliminating Hidden Serialization in Multi-Node Megakernel Communication](https://arxiv.org/abs/2605.00686)

**Authors**: Byungsoo Oh, Rachee Singh  
**Category**: cs.DC  
**Published**: 2026-05-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.00686v1  

#### Abstract
Recent megakernel designs for Mixture-of-Experts (MoE) inference fuse expert computation with fine-grained, GPU-initiated communication into a single persistent GPU kernel, and outperform collective-based MoE on a single node by overlapping data transfer with compute at tile granularity. This benefi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Eliminating Hidden Serialization in Multi-Node Megakernel Communication*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文揭示并解决了**多节点 megakernel 架构在 MoE（Mixture-of-Experts）推理中的性能退化问题**。尽管单节点 megakernel 设计通过将专家计算与 GPU 发起的细粒度通信融合在一个持久内核中，实现了计算与通信的重叠，但在扩展到多节点时，其性能反而显著下降（最高可达 10× 慢于单节点趋势）。

根本原因被定位为：**基于代理（proxy-based）RDMA 传输路径中存在的“隐藏序列化”（hidden serialization）**。具体而言，每个 `put-with-signal` 操作都需要在 NIC 提交路径上插入一个 `fence` 来保证数据写入先于信号通知，而这个 `fence` 会清空 NIC 流水线，导致代理线程阻塞。随着并发传输数量增加（如 96 路并发），这种序列化开销急剧增长，破坏了原本期望的细粒度重叠。

### 提出了什么新方法或新思路
作者提出了 **Perseus**，一种消除多节点 megakernel 中隐藏序列化的系统，包含两个互补的技术：

1. **Decoupled Signaling（解耦信号机制）**  
   将数据传输（PUT）与信号通知（SIGNAL）分离。所有 CTA 并发地提交非阻塞的 PUT 操作，仅由每组（按目标 GPU 分组）的一个 leader CTA 在最后统一执行一次 `fence` 并批量发送所有 SIGNAL。这将 `fence` 的次数从 **每专家一次** 减少到 **每远程 GPU 一次**，在 Qwen3 配置下实现 **8× 的 fence 数量减少**。

2. **NIC-side Ordering（NIC 端排序机制）**  
   将排序责任从 CPU 代理转移到 NIC 硬件。代理不再执行阻塞式的 `fence` 等待完成，而是为 SIGNAL 请求附加一个硬件 `fence flag`（如 `FI_FENCE` 或 `IBV_SEND_FENCE`）。NIC 硬件自动确保该 SIGNAL 在所有前置操作完成后才被处理，从而 **代理永不阻塞**，完全消除了软件侧的序列化瓶颈。

### 相比现有方法的优势
- **无需改变应用接口**：Perseus 完全兼容现有的 `put-with-signal` 编程模型，用户无需修改 megakernel 应用代码。
- **显著提升性能**：在 proxy-based 传输上实现高达 **10.3× 的端到端加速**。
- **超越 GPU-direct 性能**：在 IBRC 上，Perseus 的性能达到甚至超过原生的 IBGDA GPU-direct 传输（最高 **1.2×**），证明了**序列化机制而非传输类型本身才是性能瓶颈**。
- **通用性强**：优化可移植至多种 ML 框架，例如在 Triton-distributed 的 ALLTOALL 基准测试中实现 **79× 加速**。

---

## 2. 核心实验方法和设置

### 使用的模型（非传统数据集）
论文评估了三种具有不同计算-通信比的 MoE 模型，以覆盖广泛的工作负载：

| Model | Type | Experts (E) | Top-k (k) |
|-------|------|-----------|--------|
| **Qwen3-30B** | 通信密集型 | 128 | 8 |
| **GPT-OSS-120B** | 平衡型 | 128 | 4 |
| **DeepSeek-V3** | 计算密集型 | 256 | 8 |

### 实验设置
- **硬件平台**：
  - **Perlmutter**：最多 16 节点（64× A100 GPU），使用 **Libfabric + Slingshot-11**（proxy-based）。
  - **商业 GPU 云**：最多 4 节点（32× H100 GPU），使用 **IBRC（proxy-based）** 和 **IBGDA（GPU-direct）**。
- **序列长度（S）**：从 256 到 64K tokens，用于评估从小消息（overhead-dominated）到大消息（transfer-dominated）的性能。
- **评估模式**：采用 **弱扩展（weak scaling）**，保持每 GPU 工作负载恒定，观察随节点数增加的端到端延迟变化。

### 评估指标
- **端到端前向传播延迟（end-to-end forward latency）**
- **TensorCore 利用率（SM utilization）**
- **信号效率（signaling efficiency）**：`put+signal` 吞吐量相对于纯 `put-only` 的比率。
- **Fence 开销分解**：使用 α-β 模型分析固定开销（α）和每字节传输成本（β）。

### 基线方法对比
- **Vanilla FlashMoE**：原始的 megakernel 实现，使用标准的 `put-with-signal` 和 CPU 代理。
- **NCCL collective ALLTOALL**：传统的 CPU 驱动集体通信方式。
- **IBGDA GPU-direct**：作为高性能上限的对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **Libfabric** 上，Perseus 对 Qwen3-30B 实现最高 **10.3× 的端到端加速**。
- 在 **IBRC** 上，Perseus 实现最高 **2.47× 加速**，并在 S=64K 时达到峰值。
- Perseus 在 IBRC 上的性能 **匹配或超过原生 IBGDA GPU-direct**，最高快 **1.2×**。
- 在 **Triton-distributed 的 ALLTOALL 微基准测试**中，Perseus 实现 **79× 加速**，将同步开销降低 99%。

### 与基线方法的对比结果
- **弱扩展性恢复**：Vanilla 在 8 节点上 Qwen3 的延迟恶化达 10×，而 Perseus 实现近似线性扩展。
- **吞吐恢复**：在 96 路并发、4KB 消息下，`put+signal` 吞吐从 vanilla 的 **2%**（相对 `put-only`）恢复到 Perseus 的 **74%**。
- **SM 利用率提升**：Qwen3 的 TensorCore 利用率从 vanilla 的 **31%** 提升至 Perseus 的 **95%**。

### 消融实验结果
- **Decoupled Signaling 单独作用**：在 2 节点上带来 1.2–1.6× 加速（主要来自 fence 数量减少）。
- **NIC-side Ordering 单独作用**：在 8 节点上带来 1.3–2.6× 加速（主要来自消除代理阻塞）。
- **两者结合（Perseus）**：在 8 节点上实现 **1.5–3.5× 加速**，验证了二者互补性——前者降低序列化频率，后者消除每次序列化的代价。

---

## 4. 关键结论和发现

### 主要发现
1. **多节点 megakernel 性能退化源于“隐藏序列化”**，而非 RDMA 本身的带宽或延迟。
2. **proxy-based 传输并非天生劣于 GPU-direct**；真正的瓶颈是 `fence` 引入的软件侧序列化。
3. **通过 Decoupled Signaling 和 NIC-side Ordering 可彻底消除该瓶颈**，使 proxy-based 传输性能媲美甚至超越 GPU-direct。
4. **megakernel 的潜力在多节点上依然巨大**，只要解决好传输层的序列化问题。

### 方法的局限性
- 当前实现基于 NVSHMEM，虽具通用性，但仍需适配特定通信库。
- 对极端不均衡的专家路由（skewed routing）有一定影响，但实验显示仍能保持显著加速（2.0× 以上）。
- 在 GPU-direct 传输上增益较小（约 1.25×），因原本无 proxy 阻塞问题。

### 未来工作方向
- 将 Perseus 集成到主流 serving 引擎（如 vLLM、SGLang）和训练框架（如 Megatron）中。
- 进一步探索在更大规模（>64 节点）和异构网络环境下的表现。
- 研究如何将类似思想应用于其他需要细粒度 GPU-initiated communication 的场景（如 AllReduce、P2P 同步等）。

> **一句话总结**：Perseus 揭示了多节点 megakernel 的性能瓶颈在于传输层的“隐藏序列化”，并通过解耦信号与 NIC 硬件排序，实现了高达 10.3× 的加速，并证明了 proxy-based 传输在正确设计下可匹敌 GPU-direct 性能。

</details>

---

### 7. [Model-Based Reinforcement Learning with Double Oracle Efficiency in Policy Optimization and Offline Estimation](https://arxiv.org/abs/2605.00393)

**Authors**: Haichen Hu, Jian Qian, David Simchi-Levi  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.00393v1  

#### Abstract
Reinforcement learning (RL) in large environments often suffers from severe computational bottlenecks, as conventional regret minimization algorithms require repeated, costly calls to planning and statistical estimation oracles. While recent advances have explored offline oracle-efficient algorithms...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Model-Based Reinforcement Learning with Double Oracle Efficiency in Policy Optimization and Offline Estimation

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文针对**大规模或连续状态-动作空间下的强化学习（Reinforcement Learning, RL）中存在的计算瓶颈**问题展开研究。传统基于 regret 最小化的算法在每次交互后都需要调用一次统计估计（statistical estimation）和策略优化（policy optimization）的 oracle，导致计算成本高昂，尤其在状态空间 $S$ 和动作空间 $A$ 很大时变得不可行。

此外，现有“离线 oracle 高效”（offline oracle-efficient）算法虽然减少了对在线估计器的依赖，但其**策略优化 oracle 的调用次数仍与 $|S|$ 或 $|A|$ 成正比**，无法扩展到无限状态/动作空间。因此，如何设计一种同时在**统计估计**和**策略优化**两个方面都高效的算法，是一个未解决的关键挑战。

### 提出了什么新方法或新思路
作者提出了一种名为 **DOERL (Doubly Oracle-Efficient Reinforcement Learning)** 的新算法框架，其核心思想是：

- **双 oracle 高效性**：通过引入 **log-barrier 正则化** 和 **log-determinant 正则化** 技术，结合 **trusted occupancy measure** 的概念，实现了在整个学习过程中仅需极少数调用 offline 统计估计 oracle 和 policy optimization oracle。
- **epoch-based 学习机制**：将整个交互过程划分为多个 epoch，在每个 epoch 内重复执行相同的策略，从而显著减少策略切换频率和 oracle 调用次数。
- **适用于复杂环境**：该方法不仅适用于传统的 **tabular MDP**，还被成功推广至具有**无限状态空间**和**任意动作空间**的 **linear MDP** 模型。

### 相比现有方法的优势
| 方面 | 本工作 (DOERL) | 现有方法 (如 Qian et al., 2024) |
|------|----------------|-------------------------------|
| **Regret Bound (Tabular)** | $O(\sqrt{T})$ — 达到最优 | $O(\sqrt{T})$ |
| **Estimation Oracle Calls** | $O(H \log \log T)$ 或 $O(H \log T)$ | $O(H \log \log T)$ |
| **Planning Oracle Calls** | $O(H \log \log T)$ 或 $O(H \log T)$ | $O(HSA \log \log T)$ |
| **对 $|S|, |A|$ 的依赖** | **完全独立** | 显式依赖于 $|S||A|$ |
| **适用模型范围** | 支持 **linear MDP**（无限状态/动作） | 仅限于 **tabular MDP** |

> ✅ **核心优势**：首次实现了 **doubly oracle-efficient**（双重 oracle 高效），即在统计估计和策略优化两方面的 oracle 调用次数均与状态/动作空间大小无关，并且可扩展到无限维空间。

---

## 2. 核心实验方法和设置

> ⚠️ **注意**：该论文为理论导向的研究，**并未包含传统意义上的数值实验或真实数据集测试**。所有“结果”均为**理论分析得出的数学保证**，而非在具体数据集上的运行表现。

### 使用了哪些数据集
- **无实际数据集**。研究基于标准的理论设定进行分析。
- 在 **tabular MDP** 场景下，假设状态空间 $S$ 和动作空间 $A$ 是有限集合。
- 在 **linear MDP** 场景下，允许状态空间为可数无限，动作空间为任意集合。

### 实验设置和评估指标
- **交互协议**：采用标准的 episodic RL 设置，总共有 $T$ 个 episode，每个 episode 长度为 $H$。
- **目标**：最小化 **cumulative regret**：
  $$
  \text{Reg}(T) = \sum_{t=1}^T \left( V^\star - V^{\pi_t} \right)
  $$
- **oracle 调用次数**：作为衡量计算效率的核心指标，记录在整个 $T$ 轮交互中调用了多少次 offline estimation oracle 和 policy optimization oracle。

### 基线方法对比
论文通过理论比较的方式，与以下代表性工作进行了对比（见 Table 1）：

| 基线方法 | Regret | Estimation Oracle | Planning Oracle | 模型类型 |
|--------|--------|-------------------|------------------|----------|
| Foster et al. (2021) | $O(\sqrt{T})$ | $O(T)$ online | $O(T)$ | General |
| Levy et al. (2024) | $O(\sqrt{T})$ | $O(T)$ offline | $O(T)$ | Tabular MDP |
| Qian et al. (2024) | $O(\sqrt{T})$ | $O(H \log \log T)$ | $O(HSA \log \log T)$ | Tabular MDP |
| **This Work (DOERL)** | $O(\sqrt{T})$ | $O(H \log \log T)$ | $O(H \log \log T)$ | **Tabular / Linear MDP** |

---

## 3. 主要实验结果和性能指标

由于是纯理论工作，以下“结果”指**理论证明所得的性能上界**。

### 关键性能数据

#### 对于 **Tabular MDP**：
- **Regret 上界**：$O(\sqrt{T})$ — 达到信息论最优。
- **Oracle 调用次数**：
  - 当 $T$ 已知时：仅需 $O(H \log \log T)$ 次调用。
  - 当 $T$ 未知时：需 $O(H \log T)$ 次调用。
- **关键特性**：上述 oracle 调用次数 **与 $|S|$ 和 $|A|$ 完全无关**。

#### 对于 **Linear MDP**（无限状态空间）：
- **Regret 上界**：$O(T^{4/5})$ — 虽非最优，但为**首个有意义的 sublinear regret** 结果。
- **Oracle 调用次数**：同样保持 $O(H \log \log T)$ 或 $O(H \log T)$。
- **意义重大**：这是**第一个能在无限状态/动作空间下实现 offline oracle 高效的 regret minimization 算法**。

### 与基线方法的对比结果
- 相比 Qian et al. (2024)，在 tabular MDP 下，**planning oracle 调用从 $O(HSA \log \log T)$ 降至 $O(H \log \log T)$**，实现了指数级的计算效率提升。
- 首次将 offline oracle 高效算法推广到 **linear MDP**，突破了此前仅限于 tabular 模型的限制。

### 消融实验结果（如有）
- **无传统消融实验**。
- 但在理论分析中，作者通过不同参数设置（如 $T_m = 2(T/H)^{1-2^{-m}}$ vs $T_m = 2^m$）展示了 **epoch schedule** 对 oracle 调用频率的影响，间接验证了设计的有效性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **双 oracle 高效是可行的**：通过精心设计的 log-barrier/log-determinant 正则化与 trusted occupancy measure，可以构建出在 **estimation** 和 **planning** 两方面都只需极少 oracle 调用的 RL 算法。
2. ✅ **规划复杂度可与状态空间解耦**：策略优化 oracle 的调用次数**无需依赖于 $|S|$ 或 $|A|$**，打破了传统方法的维度诅咒。
3. ✅ **可扩展至复杂模型**：所提框架能自然地推广到 **linear MDP**，解决了无限状态空间下的 offline oracle 高效学习问题，这是领域内的一个重要突破。

### 方法的局限性
1. **Regret Bound 在 Linear MDP 下非最优**：当前 regret 为 $O(T^{4/5})$，而理想目标应为 $O(\sqrt{T})$。
2. **依赖 strong oracle 假设**：虽然调用次数少，但仍假设存在能精确求解复杂优化问题的 **offline regression oracle** 和 **policy optimization oracle**，这些 oracle 在现实中可能难以实现（尤其是对于深度神经网络等复杂函数类）。
3. **理论性质强，缺乏实证验证**：尚未在实际任务（如 Mujoco、Atari）上验证其有效性与实用性。

### 未来工作方向
1. **改进 Linear MDP 下的 regret bound**：设计新算法以达到 $O(\sqrt{T})$ regret。
2. **探索更弱的 oracle 假设**：研究在近似 oracle 或更易实现的 oracle 下是否仍能保持高效性。
3. **实证研究**：将 DOERL 框架应用于实际 RL 任务，评估其在真实环境中的计算效率与性能表现。
4. **推广至更广模型类**：尝试将此框架扩展到低秩 MDP、kernel MDP 等其他结构化模型。
5. **理解 oracle 高效性的根本极限**：刻画 offline oracle 与 online oracle 在能力上的本质差距，建立统一的理论框架。

---

> 📌 **总结**：本文提出了 **DOERL**，一个在理论层面实现**双重 oracle 高效**的强化学习算法。它在 **tabular MDP** 中达到了最优 regret 且 oracle 调用与状态/动作空间无关；并首次将其扩展到 **linear MDP**，为无限状态空间下的高效 RL 打开了新的大门。尽管目前仍是理论成果，但它为解决大规模 RL 的计算瓶颈提供了全新的视角和强有力的工具。

</details>

---

### 8. [Learning How and What to Memorize: Cognition-Inspired Two-Stage Optimization for Evolving Memory](https://arxiv.org/abs/2605.00702)

**Authors**: Derong Xu, Shuochen Liu, Pengfei Luo, Pengyue Jia, Yingyi Zhang, Yi Wen, Yimin Deng, Wenlin Zhang, Enhong Chen, Xiangyu Zhao, Tong Xu  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.00702v1  

#### Abstract
Large language model (LLM) agents require long-term user memory for consistent personalization, but limited context windows hinder tracking evolving preferences over long interactions. Existing memory systems mainly rely on static, hand-crafted update rules; although reinforcement learning (RL)-base...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning How and What to Memorize: Cognition-Inspired Two-Stage Optimization for Evolving Memory*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）代理在长期个性化交互中需要维护**动态演化的用户记忆**，以跟踪用户的偏好变化。然而，受限于有限的 **context window**，LLM 难以保留完整的对话历史。现有的记忆系统存在以下问题：
- **静态规则**：依赖手工设计的记忆更新模板，缺乏适应性。
- **稀疏奖励**：基于强化学习（RL）的方法通常仅依赖最终任务的成功作为稀疏奖励，导致训练不稳定、探索困难。

### 提出的新方法与思路
本文提出 **MemCoE**（Memory Cognition-inspired Optimization），一种受认知科学启发的**两阶段优化框架**，解耦“如何组织记忆”与“存储什么内容”。

#### 创新点：
- **认知机制类比**：借鉴人类大脑中 **prefrontal cortex**（负责模式选择与配置）与 **hippocampus**（负责具体细节编码）的功能分工，将记忆演化过程分为两个阶段：
  1. **Memory Guideline Induction (MGI)**：诱导一个全局的、自然语言形式的**记忆指南**（guideline），定义记忆应如何被组织（即“how to memorize”）。
  2. **Guideline-Aligned Memory Policy Optimization (GMPO)**：利用该指南构建**结构化的过程奖励**（process reward），通过多轮 RL 学习一个遵循指南的记忆更新策略（即“what to memorize”）。

- **文本梯度（Textual Gradients）**：MGI 阶段将对比反馈（contrastive feedback）解释为“文本梯度”，用于迭代优化记忆指南。
- **批聚合（Batch Aggregation）**：跨批次聚合文本梯度，提升指南的泛化性和稳定性。

### 相比现有方法的优势
- **更强的监督信号**：通过引入**过程级奖励**（guideline-aligned rewards），解决了传统 RL 方法因稀疏奖励导致的训练不稳定性。
- **更高效的搜索空间**：先由 MGI 固定记忆操作的“模式”，再由 GMPO 在此约束下学习具体内容，显著缩小了策略搜索空间。
- **良好的可迁移性**：诱导出的记忆指南可在不同 LLM 间迁移，实现“一次优化，多处部署”。
- **更高的鲁棒性与效率**：在长历史、高噪声场景下表现更优，且内存演化过程更高效。

---

## 2. 核心实验方法和设置

### 使用的数据集
在三个个性化记忆基准上进行评估：
- **PersonaMem**：模拟用户长期对话历史，测试对偏好演化的追踪能力，包含 32K 和 128K token 规模。
- **PrefEval**：强调显式（explicit）与隐式（implicit）偏好的推理，插入干扰对话以测试长程依赖。
- **PersonaBench**：异构用户语料库（对话、AI 交互、电商记录），测试在噪声环境下进行个性化问答的能力。

### 实验设置与评估指标
- **主干模型**：主要使用 `Qwen2.5-7B-Instruct`，部分对比实验涉及 `gpt-4o-mini`, `gemini-2.5-flash`, `GPT-5`。
- **检索模块**：采用 `all-MiniLM-L6-v2` 进行向量检索，Top-10。
- **评估指标**：
  - PersonaMem & PrefEval：**Accuracy**
  - PersonaBench：**Macro F1**

### 基线方法对比
| 类型 | 方法 |
|------|------|
| 上下文基线 | `Long Context`（直接输入原始历史） |
| 检索增强 | `RAG`, `Mem0`, `A-Mem`, `LightMem` |
| 强化学习记忆代理 | `MemAgent`, `MEM-α` |
| 后训练基线 | `SFT`, `PPO`, `GRPO` |

所有方法在相同 backbone、数据划分和超参数下公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）
| Method | PersonaMem (32K) | PersonaMem (128K) | PrefEval (Exp.) | PrefEval (Imp.) | PersonaBench (Overall) | **Overall** |
|--------|------------------|------------------|---------------|---------------|-----------------------|-----------|
| Long Context | 34.36 | 25.05 | 31.70 | 30.80 | 29.00 | 26.90 |
| RAG | 48.67 | 38.90 | 47.80 | 32.40 | 28.16 | 36.68 |
| MemAgent | 53.58 | 43.59 | 72.30 | 63.60 | 19.36 | 45.00 |
| **MemCoE (Ours)** | **57.06** | **47.24** | **81.30** | **69.90** | **29.89** | **52.02** |

> ✅ **MemCoE 在所有设置下均取得最佳性能**，平均准确率领先第二名约 **7.02%**。

### 与基线方法的对比结果
- 在长上下文（128K）和高噪声（PersonaBench noise=0.7）场景下，传统方法性能急剧下降，而 **MemCoE 表现最稳健**。
- 在显式和隐式偏好推理上均大幅超越 RL 基线（如 MemAgent），表明其能更好捕捉复杂偏好模式。
- 图 3 显示 MemCoE 在**性能-效率权衡**上处于前沿，优于多数方法。

### 消融实验结果（见 Table 2）
| 变体 | PersonaMem (32K) | PrefEval (Exp.) | 结论 |
|------|------------------|----------------|------|
| **Full Model** | **57.06** | **81.30** | — |
| w/o CF (无对比反馈) | 56.44 | 78.30 | 对比反馈提升可靠性 |
| w/o GR (无指南奖励) | 56.24 | 79.50 | 指南奖励有效引导更新 |
| w/o MGI (无第一阶段) | 54.81 | 73.20 | MGI 至关重要，尤其对偏好保持 |
| w/o GMPO (无第二阶段) | 53.37 | 77.40 | GMPO 对长程追踪至关重要 |
| w/o ALL (无两阶段) | 48.47 | 71.70 | 两阶段协同是成功关键 |

> 🔍 消融实验证明：**MGI 与 GMPO 缺一不可**，两者共同作用才能实现最优性能。

---

## 4. 关键结论和发现

### 主要发现
1. **解耦“组织”与“内容”是有效的**：受认知科学启发的两阶段设计（MGI + GMPO）能显著提升记忆系统的稳定性与准确性。
2. **过程奖励优于稀疏奖励**：通过指南生成的**结构化过程奖励**（guideline-aligned rewards）为 RL 提供了更强的学习信号。
3. **诱导的指南具有强泛化性**：
   - 可在不同 LLM 间迁移（Table 3），优化一次即可部署到多种模型。
   - 能抵抗噪声干扰，在长历史中保持偏好信息（图 7 显示 10 轮后仍保持 74% 偏好留存）。
4. **优于后训练方法**：相比直接对问答任务进行 SFT/PPO，MemCoE 通过显式建模记忆演化机制，在长期个性化任务上表现更优（Table 11）。

### 方法的局限性
- **依赖 LLM 评分器**：GMPO 阶段依赖 LLM 对 memory update 是否符合指南进行打分，其可靠性直接影响性能。
- **超参数敏感**：每轮 token 预算（token budget）和演化轮数需仔细调参；过小易累积错误，过大则单步处理复杂。
- **单目标优化**：当前框架未显式平衡多个目标（如稳定性 vs. 可塑性、信息量 vs. 简洁性），扩展至多目标较难。

### 未来工作方向
- 设计更鲁棒的**多目标记忆演化控制器**。
- 探索**无需外部评分器**的自监督指南学习机制。
- 将 MemCoE 扩展至**多模态记忆**（如视觉、听觉）管理。
- 研究**用户可控的记忆编辑接口**，支持隐私保护下的记忆查看、修正与删除。

--- 

> 📌 **总结**：MemCoE 通过认知启发的两阶段设计，首次实现了对“如何记忆”与“记忆什么”的联合学习，在多个个性化记忆基准上实现了 SOTA 性能，兼具**高效性、鲁棒性与可迁移性**，为构建长期个性化的 LLM 代理提供了新范式。

</details>

---

### 9. [BWLA: Breaking the Barrier of W1AX Post-Training Quantization for LLMs](https://arxiv.org/abs/2605.00422)

**Authors**: Zhixiong Zhao, Zukang Xu, Dawei Yang  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.00422v1  

#### Abstract
Large language models (LLMs) have driven major progress in NLP, yet their substantial memory and compute demands still hinder practical deployment. Binarization can compress weights to 1 bit, fundamentally lowering compute and bandwidth cost. However, existing methods cannot address activation heavy...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《BWLA: Breaking the Barrier of W1AX Post-Training Quantization for LLMs》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

当前大型语言模型（LLMs）在部署时面临巨大的内存和计算开销。虽然量化（quantization）是主流压缩技术，但现有的 **post-training quantization (PTQ)** 方法在实现 **W1AX**（即权重1-bit + 激活低比特）时存在严重瓶颈：

- **Weight-codebook mismatch**：LLM 权重通常呈单峰（unimodal）分布，而二值化（binarization）的理想目标是双峰（bimodal）分布（±1），导致量化误差大。
- **Heavy-tailed activations**：激活值具有长尾分布和极端离群值（outliers），难以进行低比特量化。

因此，现有方法要么只对权重进行二值化（保留高精度激活），无法实现端到端加速；要么联合量化时性能急剧下降。

---

### 提出了什么新方法或新思路

本文提出了 **BWLA**（Binarized Weights and Low-bit Activations），首个无需再训练（retraining-free）即可实现高精度 **W1AX** 的 PTQ 框架。

其核心由两个模块组成：

#### （1）Orthogonal-Kronecker Transformation (OKT)

- **目标**：将单峰权重分布重塑为对称双峰形式，并同时抑制激活的长尾。
- **方法**：
  - 利用正交变换 $ R \in O(m) $，满足 $ R^{-1} = R^T $，保证前向等价性。
  - 采用 **Kronecker 分解** $ R = R_1 \otimes R_2 $，将 $ m \times m $ 的稠密矩阵分解为两个小矩阵，显著降低计算复杂度（从 $ O(m^2) $ 降至 $ O(m^{3/2}) $）。
  - 通过 EM 最小化策略优化，鼓励权重聚类到 ±c 附近。

#### （2）Proximal SVD Projection (PSP)

- **目标**：进一步消除 OKT 后残留的结构化误差，强化双峰结构。
- **方法**：
  - 引入一个低秩残差矩阵 $ M = AB $ 进行微调。
  - 使用 proximal SVD 投影，在保持低参数开销的同时吸收剩余异常值，提升量化鲁棒性。

---

### 相比现有方法的优势

| 维度 | 现有方法（如 BiLLM, ARB-LLM, DBellQuant） | BWLA |
|------|----------------------------------------|-------|
| 是否需要 retraining/QAT | 否（但效果有限） | 否 ✅ |
| 是否支持 W1AX（1-bit 权重 + 低比特激活） | ❌ 多数仅 weight-only 或需高精度激活 | ✅ 支持 W1A6/W1A8 |
| 性能保持能力 | 在激活量化后性能骤降 | 高精度保持，甚至接近 FP16 |
| 推理加速潜力 | 有限（因仍需 dequantize 高精度激活） | 显著端到端加速（减少带宽与计算） |
| 计算开销 | 较低 | 极轻量级（OKT+PSP 仅增加 ~0.1 bit 开销） |

> ✅ **首次实现了高质量、无需训练的 W1AX PTQ**，打破了长期存在的“激活必须高精度”假设。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

- **校准数据集（Calibration）**：
  - `WikiText2`：用于 PTQ 参数校准（128 个文本片段）
- **评估数据集**：
  - **通用能力**：
    - `WikiText2`, `C4`：评估 perplexity
  - **零样本任务（Zero-shot QA）**：
    - `ARC-Challenge/Easy`, `HellaSwag`, `PIQA`, `WinoGrande`
  - **复杂推理任务**：
    - `MMLU`（多领域知识）
    - `GSM8K`（数学推理）
    - `HumanEval`（代码生成）

---

### 实验设置和评估指标

| 设置项 | 描述 |
|--------|------|
| **模型家族** | LLaMA2（7B, 13B, 70B）、LLaMA3-8B、Qwen3（8B, 14B, 32B）及其指令微调版本（如 Qwen3-32B-Instruct） |
| **量化配置** | 
| - 权重 | 对称 per-channel 量化 |
| - 激活 | 不对称 per-token 量化 |
| - 激活比特 | 测试 A16（全精度）、A6、A4 |
| **硬件平台** | NVIDIA A6000 / RTX A6000 GPU |
| **评估指标** |
| - Perplexity（越低越好） |
| - Zero-shot Accuracy 平均值（越高越好） |
| - 推理吞吐量（Tokens/sec） |
| - 内存占用（GB） |
| - 量化耗时（Quantization Time） |

---

### 基线方法对比

| 方法 | 类型 | 是否支持 W1AX | 特点 |
|------|------|----------------|------|
| `FP16` | 全精度基准 | ❌ | 性能上限 |
| `RTN`（Round-to-Nearest） | 基础量化 | ❌ | 无优化，性能差 |
| `GPTQ` | Hessian引导误差补偿 | ❌ | 主流低比特 PTQ，不适用于 binarization |
| `OSTQuant` | 正交旋转 + 缩放 | ❌ | 支持低比特但非二值化 |
| `BiLLM` | 局部 Hessian 加权处理显著权重 | ⚠️ Weight-only | 当前 SOTA 二值化方法之一 |
| `ARB-LLM` | 交替细化列组位图 | ⚠️ Weight-only | 性能较强但依赖复杂搜索 |
| `DBellQuant` | 双峰变换 + Learnable Scaling | ⚠️ 尝试联合量化 | 当前最先进尝试，但仍不稳定 |

> BWLA 是唯一在所有设置下稳定支持 **W1A6** 且性能远超上述方法的方案。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 在 Qwen3-32B 上的表现（A6 激活）

| 指标 | 数值 |
|------|------|
| **WikiText2 Perplexity** | **11.92**（vs. FP16 的 7.61，SOTA 如 BiLLM 达 38） |
| **5项 Zero-shot 任务平均准确率提升** | **>70%**（相比原始方法） |
| **推理速度提升** | **3.26×**（vs. FP16） |
| **内存节省** | 从 120+ GB → **3.94 GB**（>80%） |

> 💡 即使在如此激进的压缩下，仍保持极高的语义保真度。

---

### 与基线方法的对比结果（见 Table 1 & Table 6）

| 方法 | Qwen3-32B @ W1A6 | Perplexity ↓ | Zero-shot Avg ↑ |
|------|------------------|-------------|------------------|
| `FP16` | - | 7.61 | 72.42 |
| `BiLLM` | W1A16 | 35.05 | 35.05 |
| `BiLLM` | W1A6 | 35.05 | 35.05 |
| `ARB-LLM` | W1A6 | 32.11 | 32.11 |
| `DBellQuant` | W1A6 | — | —（部分失败） |
| `BWLA` | **W1A6** | **14.33** | **67.17** ✅ |

> ✅ BWLA 在激活量化至 6-bit 后，**perplexity 下降超过 50%**，**zero-shot 准确率提升超过 90%**，远胜其他方法。

---

### 消融实验结果（Ablation Study）

#### （1）组件有效性（Table 4）

| 模型 | OKT | PSP | Perplexity (Qwen3-32B) | Zero-shot (A6) |
|------|-----|-----|------------------------|----------------|
| Baseline | × | × | 4e4 | 35.05 |
| OKT only | √ | × | 15.65 | 51.21 |
| PSP only | × | √ | 19.77 | 47.19 |
| **OKT+PSP** | √ | √ | **14.33** | **67.17** ✅ |

> ✅ 两者协同作用显著，OKT 起主导作用，PSP 提供额外增益。

#### （2）Overhead-Performance Trade-off（Figure 4）

- **OKT 中 Kronecker 维度比 $ n_1/n_2 $**：
  - 当 $ n_1 \approx n_2 \approx \sqrt{m} $ 时达到最优平衡，可将引入的额外 bit 宽度从 ~1bit 降至 **0.01bit**。
- **PSP 截断秩比例（rank ratio）**：
  - 最佳值为 **0.005**，此时性能损失仅 ~1%，节省约 0.3bit 开销。

> 表明 BWLA 设计高度高效，可在极低开销下获得最大收益。

---

## 4. 关键结论和发现

### 论文的主要发现

1. ✅ **Unimodal 权重 + Heavy-tailed 激活 是 W1AX 的根本障碍**，传统方法无法同时解决。
2. ✅ **正交变换（OKT）可以统一解决权重双峰化与激活平滑化**，利用 $ R^{-1}=R^T $ 保证前向不变性。
3. ✅ **Kronecker 结构使大规模正交变换变得可行**，解决了计算与存储瓶颈。
4. ✅ **PSP 提供轻量级残差修正机制**，进一步吸收难以消除的异常值。
5. ✅ **BWLA 实现了首个高质量、无需训练的 W1AX PTQ**，在多个 LLM 上验证有效。

---

### 方法的局限性

1. **极端低比特激活稳定性不足**：
   - 在 **W1A4** 场景下模型稳定性明显下降，说明当前激活平滑机制不足以应对更严苛条件。
2. **变换形式受限于线性正交操作**：
   - OKT 使用的是线性旋转，可能无法充分捕捉现代 LLM 权重空间中的非线性几何特性。
3. **格式限制**：
   - 当前优化针对整数量化格式（如 INT4/INT6），尚未扩展至混合精度或新兴浮点格式（如 MXFP4）。

---

### 未来工作方向

1. 探索 **轻量级非线性变换** 替代纯正交旋转，以更好建模复杂权重分布。
2. 扩展至 **混合精度量化** 和 **低精度浮点格式**（如 FP4, MXFP4），提升动态范围适应性。
3. 将 BWLA 应用于 **Mixture-of-Experts (MoE)** 架构，探索稀疏性与二值化的协同效应。
4. 探索 **硬件友好的部署方案**，结合专用 ASIC/FPGA 实现极致能效比。

---

> 🔚 **总结**：  
> BWLA 成功突破了 W1AX PTQ 的长期技术壁垒，提出了一套高效、无需训练的联合量化框架，为 LLM 在边缘设备上的超高效部署开辟了新路径。其实验全面、设计精巧，代表了当前 LLM 二值化领域的最前沿进展。

</details>

---

### 10. [Decouple before Integration: Test-time Synthesis of SFT and RLVR Task Vectors](https://arxiv.org/abs/2605.00610)

**Authors**: Chaohao Yuan, Chenghao Xiao, Yu Rong, Hong Cheng, Long-Kai Huang  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.00610v1  

#### Abstract
SFT and RLVR represent two fundamental yet distinct paradigms for LLM post-training, each excelling in distinct dimensions. SFT expands knowledge breadth while RLVR enhances reasoning depth. Yet integrating these complementary strengths remains a formidable challenge. Sequential training can cause c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Decouple before Integration: Test-time Synthesis of SFT and RLVR Task Vectors**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

该论文聚焦于 **LLM 后训练（post-training）中 SFT（Supervised Fine-Tuning）与 RLVR（Reinforcement Learning with Verifiable Rewards）难以有效融合的问题**。

- **SFT** 擅长扩展模型的**知识广度**（如事实、任务特定知识），而 **RLVR** 擅长增强**推理深度**（reasoning depth）。
- 然而，现有的集成方式（如顺序训练或联合优化）面临严重挑战：
  - **灾难性遗忘**（catastrophic forgetting）：RLVR 阶段可能覆盖 SFT 学到的知识。
  - **梯度冲突**（gradient conflicts）：SFT 和 RLVR 的优化目标存在方向性冲突。
  - **更新信号不兼容**：两者在参数空间中的更新模式差异巨大。

因此，论文提出一个根本性问题：**这些失败是训练策略不当所致，还是源于两种范式在结构上的深层不兼容？**

---

### ✅ 提出了什么新方法或新思路

作者提出了 **Decoupled Test-time Synthesis (DoTS)** ——一种**解耦式测试时合成框架**，其核心思想是：

> **训练阶段完全独立，仅在推理时通过 task vector arithmetic 合成能力**，无需任何参数更新。

#### DoTS 的三大关键技术组件：

1. **Selective Sparsification（选择性稀疏化）**
   - 对 SFT 和 RLVR 的 task vectors 进行基于幅度的剪枝（保留 top-k% 参数）。
   - 采用 **norm-preserving rescaling** 保持原始向量强度，缓解因稀疏化导致的信号衰减。
   - 显著降低 sign interference（符号干扰）从 44.91% → 最低至 7.1%。

2. **Difficulty-Aware Data Selection（难度感知的数据选择）**
   - 构建一个小规模（默认 64 条）、无标签的适配查询集（adaptation queries）。
   - 查询按“难度”分层采样：基于 SFT 和 RLVR 模型输出的一致性（consistency）打分。
     - 太简单或太难的题被过滤，保留中等难度样本以提供可靠优化信号。

3. **Bayesian Coefficient Optimization（贝叶斯系数优化）**
   - 在稀疏化后的 task vectors 上搜索最优组合系数 $ \lambda_{SFT}, \lambda_{RLVR} $。
   - 优化目标为 **Pareto frontier 上的一致性（consistency）最大化与困惑度（perplexity）最小化**。
   - 使用 **Tree-structured Parzen Estimator (TPE)** 进行黑箱贝叶斯搜索，仅需约 20 GPU 小时。

---

### ✅ 相比现有方法的优势

| 维度 | DoTS | 传统训练集成方法（如 SFT+RL, LUFFY） |
|------|------|-------------------------------|
| **计算成本** | ~20 GPU 小时（约 3% 成本） | >600 GPU 小时 |
| **数据需求** | 仅需 64 无标签 query | 数万条 on/off-policy 数据 |
| **灵活性** | 可复用已有 checkpoint，无需重新训练 | 必须端到端训练 |
| **稳定性** | 避免训练过程中的梯度冲突和过拟合 | 容易出现训练不稳定、遗忘等问题 |
| **泛化性** | 学得的系数可迁移到 OOD 任务（如 ARC-C, GPQA） |

---

## 2. 核心实验方法和设置

### ✅ 使用了哪些数据集

#### 主要评估基准（数学推理）：
- **AIME 2024 / 2025**
- **AMC**
- **MATH500**
- **Minerva**
- **OlympiadBench**

#### 跨领域泛化评估（Out-of-Domain QA）：
- **ARC-C**
- **GPQA**
- **MMLU-Pro**

所有测试均未对这些 OOD 数据进行微调或重调系数。

---

### ✅ 实验设置和评估指标

| 设置项 | 描述 |
|-------|------|
| **主干模型** | Qwen2.5-Math-7B（主要）、Qwen2.5-Math-1.5B、LLaMA3.1-8B |
| **源 Checkpoint** | 使用公开的 SFT 和 RLVR 模型（如 LUFFY、ExGRPO、ReLIFT）提取 task vectors |
| **评估方式** | - AIME/AMC：`average@32`<br>- MATH500/Minerva/Olympiad：`pass@1` |
| **生成参数** | 温度 0.6，最大长度 8192（Qwen），2048（LLaMA） |
| **适配集大小** | 64 个无标签 query |
| **搜索次数** | 100 次贝叶斯试验 |

---

### ✅ 基线方法对比

| 类别 | 基线方法 |
|------|---------|
| **基础模型** | Qwen2.5-base, Qwen2.5-Instruct |
| **纯 RLVR 方法** | SimpleRL-Zero, OpenReasoner-Zero, PRIME-Zero, Oat-Zero, ExGRPO |
| **训练集成方法** | SFT, On-Policy RL, RL w/ SFT Loss, SFT+RL, LUFFY, ReLIFT |
| **模型合并方法** | TIES-Merging, DARE, TIES-Merging*（替换 sparsification 模块） |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（Qwen2.5-Math-7B）

| Model | Average Score |
|-------|---------------|
| SFT | 44.1 |
| On-Policy RL | 45.5 |
| LUFFY | 49.2 |
| **DoTS (SFT + On-Policy RL)** | **49.3** ✅ |
| **DoTS (ExGRPO + ReLIFT)** | **50.6** ✅ |

> 🔺 DoTS 在多个数学推理 benchmark 上 **匹配甚至超越最强训练集成方法**，且平均得分达 **50.6**，超过 SOTA 的 LUFFY（49.2）**1.4 分**。

---

### ✅ 与基线方法的对比结果

| 对比维度 | 结果 |
|--------|------|
| **vs. 标准模型合并（TIES/DARE）** | DARE 表现极差（14.5），TIES-Merging 仅 37.8；说明直接合并因 magnitude/sign 不匹配而失效 |
| **vs. TIES-Merging*** | 改进版达 48.4，但仍低于 DoTS（49.3），表明**自适应系数搜索至关重要** |
| **vs. SFT+RL 顺序训练** | DoTS 超越该常见流程，且无需额外训练 |
| **跨 backbone 泛化** | 在 Qwen-1.5B 和 LLaMA-8B 上仍有效，尤其当 SFT++ 引入更难题时提升显著（LLaMA 上从 13.2 → 22.6） |

---

### ✅ 消融实验结果（Ablation Study）

| 方法变体 | Average Score | 相比完整 DoTS 下降 |
|--------|----------------|------------------|
| **完整 DoTS** | **49.3** | — |
| w/o Data Selection（随机选 query） | 48.0 | ↓1.3 |
| w/o Sparsification | 49.1 | ↓0.2 |
| 固定系数 (1.0, 1.0) | 47.8 | ↓1.5 |
| 固定系数 (0.5, 0.5) | 46.3 | ↓3.0 |

> 🔍 发现：
> - **难度感知数据选择** 最关键，错误采样会显著拉低性能。
> - **稀疏化虽影响较小，但在固定系数下作用明显**（见 Table 10：+0.9 分）。
> - **贝叶斯搜索优于固定系数**，证明 adaptive tuning 必要。

---

## 4. 关键结论和发现

### ✅ 论文的主要发现

1. **SFT 与 RLVR 存在结构性不兼容**（structural incompatibility）：
   - **~30× magnitude disparity**：SFT task vector 的 L2 范数远大于 RLVR。
   - **~45% sign interference**：近一半参数更新方向相反。
   - **异构模块分布**：SFT 更新集中在 LayerNorm，RLVR 更分散（Attention, LM Head 等）。

2. **尽管存在冲突，二者也具有互补性**：
   - 更新集中在不同模块，意味着它们修改的是模型的不同部分。
   - 若能控制干扰，**后处理合成是可行且高效的路径**。

3. **DoTS 是高效且通用的能力融合范式**：
   - 无需训练，仅靠 task vector arithmetic 即可实现 SOTA 性能。
   - 所学系数具备**跨任务迁移能力**，在 ARC-C、GPQA、MMLU-Pro 上均优于各单一模型。

---

### ✅ 方法的局限性

- **依赖高质量、互补的 source checkpoints**：
  - 若 SFT 或 RLVR 模型本身能力弱或重叠度高（如 LLaMA 上标准 SFT 数据受限），DoTS 提升有限。
- **当前为全局系数**：
  - 使用统一的 $ \lambda_{SFT}, \lambda_{RLVR} $，未考虑 layer-wise 或 module-wise 差异化加权。
- **对非常弱的 backbone 效果受限**：
  - 如 LLaMA3.1-8B 上若不使用更强的 SFT++，增益较小。

---

### ✅ 未来工作方向

1. **更精细的 composition 策略**：
   - 探索 layer-wise、module-wise 或 token-level 的 task vector 合成。
2. **扩展到其他任务类型**：
   - 如开放生成（open-ended generation）、工具使用（tool use）、多模态等。
3. **理论分析 task vector 几何性质**：
   - 深入理解 SFT 与 RLVR 在参数空间中的正交性与互补性边界。
4. **自动化适配集构建**：
   - 减少人工设计 difficulty scoring 规则，实现完全自适应 query 选择。

---

> 📌 **一句话总结**：  
> DoTS 通过揭示 SFT 与 RLVR 在 task vector 层面的结构性差异，提出了一种“先解耦训练、后合成推理”的新范式，在几乎零训练成本下实现了媲美甚至超越复杂训练集成方法的效果，并展现出良好的泛化性和实用性。

</details>

---

### 11. [Agent Capsules: Quality-Gated Granularity Control for Multi-Agent LLM Pipelines](https://arxiv.org/abs/2605.00410)

**Authors**: Aninda Ray  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.00410v1  

#### Abstract
A multi-agent pipeline with N agents typically issues N LLM calls per run. Merging agents into fewer calls (compound execution) promises token savings, but naively merged calls silently degrade quality through tool loss and prompt compression. We present Agent Capsules, an adaptive execution runtime...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Agent Capsules: Quality-Gated Granularity Control for Multi-Agent LLM Pipelines**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
多智能体 LLM 流水线（multi-agent LLM pipelines）在执行时通常为每个 agent 发起一次独立的 LLM 调用，导致高昂的 **input token 开销** 和 **per-call 固定延迟**。虽然将多个 agent 合并为单次调用（compound execution）可显著节省成本，但这种合并会引发两个严重问题：
- **工具丢失（tool loss）**：合并后 agent 无法正常调用其专属工具。
- **提示压缩（prompt compression）**：模型倾向于对多个 agent 的指令生成浅层、简略的响应，而非深度推理。

现有框架（如 LangGraph、CrewAI）缺乏自动判断何时安全合并 agent 的机制，需手动工程权衡效率与质量。

---

### **提出了什么新方法或新思路**
作者提出 **Agent Capsules (AC)** ——一个自适应运行时系统，将多智能体流水线执行视为一个受质量约束的优化问题。其核心机制包括：

#### **(1) Composition Score（组合评分）**
基于运行时行为信号计算的“行为指纹”，预测是否可以安全合并 agent，无需 per-model 配置。信号包括：
- 协调开销比例（coordination overhead ratio）
- agent 数量（n）
- 工具调用密度（tool-call density）
- 依赖链深度（dependency depth）

该分数是**行为性的**（behavioral），而非能力排名（capability ranking）。

#### **(2) Quality Gate（质量门控）**
使用更强的 LLM judge 动态评估 compound 输出质量，并与 fine-grained 基线对比。若滚动平均质量低于设定阈值 `quality_floor`，则回退到细粒度执行。

#### **(3) Escalation Ladder（升级阶梯）**
当标准合并失败时，逐步升级执行策略，而非直接放弃合并：
- **Standard Compound** → **Two-Phase Compound** → **Sequential Compound**
  - **Two-Phase**：先并行收集工具输出（Phase A），再统一推理（Phase B），保留工具访问。
  - **Sequential**：逐个 agent 调用，累积上下文，恢复 per-agent 深度。

> ✅ **关键洞察**：增加合并前的推理上下文（如更丰富的 Phase A）反而加剧压缩；真正解法是减少合并程度（即 un-merge），而非重写提示。

#### **(4) Fine-Grained Mode 本身已是优化层**
即使未进入 compound 模式，AC 也通过以下机制提升效率：
- **Topology-aware context injection**：仅注入必要依赖上下文。
- **Cache-aligned prompts**：利用 Anthropic 缓存机制。
- **Auto output guidance**：动态添加“简洁输出”提示。
- **Per-group policy resolution**：按组独立决策。

---

### **相比现有方法的优势**
| 维度 | Agent Capsules | 现有方法（LangGraph / DSPy） |
|------|----------------|-------------------------------|
| 执行模式选择 | 自动、自适应、带质量保障 | 手动配置或静态编译 |
| 效率来源 | 运行时观察 + 结构优化 | 人工工程 / 编译时提示优化 |
| 泛化性 | 无需 per-pipeline 工程 | 需针对特定 pipeline 调优 |
| 安全性 | 质量门控防止劣化输出 | 无内置质量监控 |

---

## 2. **核心实验方法和设置**

### **使用的数据集与流水线拓扑**
论文在 **4 种不同拓扑的 pipeline** 上进行验证，涵盖金融与软件工程领域：

| Pipeline ID | 名称 | Agent 数 | 拓扑结构 | 描述 |
|------------|------|----------|---------|------|
| P-1 | due_diligence | 5 | Sequential | 尽职调查流程 |
| P-2 | code_review | 6 | Fan-out + Converge | 并行代码审查 |
| P-3 | long_chain_research | 8 | Long Sequential Chain | 深度研究链 |
| P-4 | multi_source_brief | 14 | Fan-out + Converge | 多源情报简报 |

所有对比均使用相同冻结的 pipeline 定义，确保公平。

---

### **实验设置与评估指标**

#### **模型与提供商**
- **主测模型**：`Sonnet`, `Haiku`（Anthropic）
- **验证模型**：`GPT-4o`, `GPT-4o-mini`, `gemini-2.5-flash-lite`

#### **评估协议**
- **Judge 模型**：
  - Anthropic 流水线 → `claude-opus-4-6`
  - OpenAI/Google 流水线 → `gpt-4o`
- **质量维度**：事实完整性、推理深度、连贯性（0–1 分）
- **噪声控制**：报告 judge 的最小可检测差异（MDD: Opus=0.030, GPT-4o=0.065）

#### **基线方法对比**
| 类型 | 基线 | 说明 |
|------|------|------|
| **手工调优基线** | Hand-tuned LangGraph | 14-agent 情报流水线，人工编写合并逻辑 |
| **编译时优化基线** | DSPy (uncompiled & MIPROv2) | 使用 BootstrapFewShot 和 MIPROv2 编译提示 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### **(1) 与 LangGraph 对比（14-agent 情报流水线，Haiku）**
| 指标 | Agent Capsules | LangGraph | 改进 |
|------|---------------|-----------|------|
| Fine 模式输入 token | 5.1K | 10.5K | ↓ **51%** |
| Compound 模式输入 token | 3.2K | 5.6K | ↓ **42%** |
| Fine 模式质量 | 0.827 | 0.807 | ↑ **+0.020** |
| Compound 模式质量 | 0.815 | 0.798 | ↑ **+0.017** |

> 🔥 **亮点**：即使在未合并阶段，AC 已因上下文精简和缓存对齐等机制超越手工实现。

#### **(2) 与 DSPy 对比（5-agent 尽职调查，Sonnet）**
| 方法 | 总 token | 质量 | 相比 AC（seq compound） |
|------|--------|------|-----------------------|
| AC (sequential compound) | **40,711** | **0.761** | — |
| DSPy (uncompiled) | 50,501 | 0.749 | ↑ 19% 更多 token |
| DSPy (MIPROv2) | 128,733 | 0.709 | ↑ **68% 更多 token**, ↓ 质量 |

✅ AC 在更少 token 下达到更高或持平质量，且无需训练数据。

---

### **消融实验结果**

#### **(1) Escalation Ladder 验证（code_review, Sonnet, aggressive）**
| 配置 | 质量 | Tokens | Compound 触发率 |
|------|------|--------|------------------|
| escalation=False | 0.313 | 189,632 | 1/7 |
| escalation=True | **0.724** | **170,734** | **5/7** |
| Δ | **+0.411** | ↓10% | ↑4/7 |

> ✅ Escalation 不仅修复质量，还稳定控制器决策，使 compound 成为可持续选项。

#### **(2) Composition Score 的有效性**
| Model | Score | 是否触发 compound |
|-------|-------|------------------|
| Haiku | 0.264–0.299 | ✅ 是（高工具调用率） |
| Sonnet | 0.245 | ✅ 是 |
| GPT-4o | 0.181 | ❌ 否 |
| Gemini-flash | 0.208 | ❌ 否 |

👉 分数准确区分了适合合并的行为模式（高 tool-call 密度），而非由模型能力决定。

#### **(3) Controlled Negative Result：增强 Phase A 反而更差**
尝试在 Two-Phase 中加入“推理预处理”阶段，结果：
- **Analysis 质量** ↑ +0.084
- **Research 质量** ↓ -0.067 ~ -0.175

> ❗ 注入更多上下文加剧了压缩，证实“减少合并”才是正解。

---

## 4. **关键结论和发现**

### **主要发现**
1. **Batching ≠ Better Efficiency**  
   盲目合并 agent 会导致质量崩溃。真正的效率来自**有质量保障的自适应合并**。

2. **Quality Gate 实现 Oracle-equivalent Routing**  
   控制器在所有 `(model, group, mode)` 组合中，其路由决策与“已知最优”的 oracle 完全一致，**无需 per-model 配置**。

3. **Escalation Ladder 是结构性答案**  
   当合并失败时，应通过升级执行策略（two-phase → sequential）来恢复质量，而不是简单回退。

4. **Fine-grained Mode 也是优化起点**  
   即使不合并，AC 的上下文剪枝、缓存对齐、动态输出引导等机制也能显著优于手工实现。

5. **Synthesis-Only Pattern 是安全默认项**  
   无工具的聚合类任务（如写作、总结）在所有模型上均可安全合并。

---

### **局限性**
| 局限 | 说明 |
|------|------|
| **Pipeline 覆盖有限** | 仅测试 ≤14 agent 的流水线，未验证 >20 agent 或创意写作等新领域 |
| **GPU 级指标缺失** | 仅提供 API 层面的 token 与延迟，缺乏 MFU、KV cache 压力等底层指标 |
| **Judge 依赖性强** | 质量评估依赖外部 judge，跨 provider 比较不可靠 |
| **显式推理模型未覆盖** | 如 o1、DeepSeek-R1 等输出 CoT token 的模型可能影响 overhead_ratio 判断 |
| **Auto Guidance 校准范围有限** | `1,500 tok/agent` 阈值基于信息检索类任务，其他场景需重新校准 |

---

### **未来工作方向**
1. **Knowledge-Base Compound Execution**  
   将 Two-Phase 的“扁平上下文注入”升级为共享知识库，支持 selective retrieval，提升可扩展性。

2. **Cross-Pipeline Competitive Benchmarks**  
   当前对比分别使用不同 pipeline（LangGraph vs 14-agent, DSPy vs 5-agent）。未来应在同一 pipeline 上交叉对比，验证优势普适性。

3. **支持显式推理模型（explicit-reasoning models）**  
   适配 o1、R1 等模型的 token 成本结构，避免将有效推理误判为协调开销。

4. **自动化 calibrate 工具**  
   提供 `Pipeline.calibrate()` 接口，基于样本任务自动推荐 `compose_at`, `quality_floor` 等参数。

---

## ✅ **总结**
**Agent Capsules** 提出了一种全新的视角：将多智能体执行视为一个**受质量约束的运行时优化问题**。它通过 **Composition Score + Quality Gate + Escalation Ladder** 三重机制，在无需人工干预的情况下，实现了比手工调优和编译时优化更高效、更安全的执行策略。

> 🏆 **核心价值**：不是“能否合并”，而是“如何安全地合并”。  
> 💡 **范式转变**：从“静态设计”走向“动态适应”，让 compound execution 成为默认安全选项。

</details>

---

### 12. [SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters](https://arxiv.org/abs/2605.00528)

**Authors**: Dongxin Guo, Jikun Wu, Siu Ming Yiu  
**Category**: cs.DC  
**Published**: 2026-05-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.00528v1  

#### Abstract
AI agents execute tens to hundreds of chained LLM calls per task, yet GPU schedulers treat each call as independent, discarding gigabytes of intermediate state between steps and inflating end-to-end latency by 3-8x. We argue that this request-level abstraction is fundamentally mismatched to compound...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **GPU 调度器**（如 vLLM、SGLang）将每个 LLM 推理请求视为独立任务进行调度，而忽略了 **AI Agent 工作负载的本质特性**：  
- **多步链式调用**：一个 Agent 任务通常包含数十到上百次 LLM 调用，形成 Thought-Action-Observation 循环；
- **中间状态依赖**：每一步推理都依赖前一步的输出和外部工具结果，且需保留巨大的 **KV Cache**（可达数 GB）；
- **bursty 请求模式**：请求成簇出现，具有强会话相关性。

现有系统在每次工具调用间隙丢弃 KV Cache，导致后续步骤必须重新生成，造成高达 6× 的端到端延迟膨胀，并浪费大量 GPU 内存。

---

### 提出的新方法与核心思想
SAGA 提出 **“程序级调度”**（program-level scheduling），即把整个 Agent 的执行流程（而非单个推理请求）作为一级调度单位。其三大核心技术机制为：

#### ✅ **(1) Agent Execution Graphs (AEGs)**
- 将 Agent 的多步推理逻辑建模为有向图 $ G = (V, E, P, \phi) $，其中节点表示 LLM 步骤，边表示执行依赖与转移概率。
- 利用 AEG 预测 KV Cache 复用机会，在工具调用期间智能保留缓存。

#### ✅ **(2) Session-Affinity Batching with Work Stealing**
- 引入 **会话亲和性路由**，确保同一 Agent 的所有步骤尽可能由同一个 worker 执行，最大化 cache hit。
- 结合 **随机化 work stealing** 机制维持全局负载均衡，避免因亲和性导致的负载倾斜。

#### ✅ **(3) Agent Fair Share (AFS) 公平性调度**
- 定义基于 **任务完成紧迫度** 的公平性指标 AFS：
  $$
  \text{AFS}_i = \sum_{t \in T_i} \frac{\text{work\_remain}(t)}{\text{deadline}(t) - t_{\text{now}}}
  $$
- 使用 **Lyapunov drift 分析** 证明 AFS 在合理假设下可提供有界的任务完成时间偏差保证。

---

### 相比现有方法的优势
| 维度 | 现有方法（如 vLLM+APC） | SAGA |
|------|------------------------|-------|
| 调度粒度 | Request-level | **Workflow-atomic** |
| 缓存管理 | LRU 或 prefix caching，不跨 tool-call | **Tool-call-aware TTL + WA-LRU** |
| 批处理策略 | 最大吞吐导向，忽略会话关联 | **Session-affinity + work stealing** |
| 公平性 | 按请求或租户分配资源 | **按任务完成时间定义公平性（AFS）** |
| 理论支撑 | 缺乏在线最优性分析 | **实证 competitive ratio 达 1.31× Belady 最优** |

> 🔑 **核心科学贡献**：首次量化表明，一旦可观测 workflow DAG，在线 KV 缓存管理可逼近离线最优（Belady）策略至 **1.31× 以内**。

---

## 2. 核心实验方法和设置

### 数据集
- **SWE-bench**：500 个真实 GitHub 编程任务，平均 37 步，最长达 150 步。
- **WebArena**：812 个浏览器自动化任务，模拟人类操作网页。
- **BurstGPT-derived synthetic workload**：合成多租户场景，含 3 类租户（heavy/medium/light），用于测试公平性和 SLO。

### 实验设置
- **硬件平台**：64 块 NVIDIA A100-80GB GPU（8 节点 × 8 卡），NVLink + 200Gbps InfiniBand。
- **模型**：Llama-3-70B-Instruct，上下文长度 32K，KV Cache 约 10.7GB/会话。
- **实现基础**：基于 vLLM v0.6.0 扩展，新增约 8.5K 行 Python 和 1.2K 行 C++/CUDA 代码。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Task Completion Time (TCT)** | 从提交到最终结果返回的时间（秒） |
| **GPU Memory Utilization** | GPU 显存中有效 KV Cache 占比 |
| **Throughput** | 每分钟完成的任务数 |
| **SLO Attainment** | 在预期时间 1.5× 内完成的任务比例 |
| **Competitive Ratio** | 实际 cache 再生成本 vs. Belady 离线最优的成本比值 |

### 基线方法
| 基线 | 特点 |
|------|------|
| **vLLM v0.6.0** | 原始版本，无 prefix caching |
| **vLLM+APC (v0.15.1)** | 启用 Automatic Prefix Caching 和 Affinity Routing，当前 SOTA |
| **SGLang v0.5.8** | 支持 structured program 执行 |
| **Llumnix v1.2** | 支持 KV Cache 迁移 |
| **TRT-LLM + Scaffolding** | 支持多步推理编排 |
| **vLLM+KVFlow** | 流程感知缓存管理（仅本地） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 3）
| System | SWE-bench TCT (s) | WebArena TCT (s) | GPU Mem (%) |
|--------|--------------------|------------------|-------------|
| vLLM v0.6.0 | 612.3 | 178.4 | 42.1 |
| vLLM+APC | 352.1 | 127.3 | 58.7 |
| **SAGA** | **203.4** | **82.1** | **71.3** |

#### ✅ 性能提升总结：
- **端到端延迟降低**：
  - 相比 vLLM+APC：**几何平均加速 1.64×**（SWE-bench 1.73×，WebArena 1.55×，p<0.001）
  - 相比原始 vLLM：最高达 **3.01× 加速**
- **内存利用率提升**：
  - 提升 **1.22×**，从 58.7% → 71.3%
- **SLO 达成率**：
  - 多租户干扰下达到 **99.2% SLO attainment**（vLLM 仅为 67.3%）
- **理论竞争力验证**：
  - WA-LRU 的实证 competitive ratio 为 **1.31×**，接近 Belady 最优（见 Table 2）

---

### 消融实验结果（Table 4）
| 配置 | TCT (s) | 相对完整版增幅 |
|------|--------|---------------|
| Full SAGA | 203.4 | — |
| w/o Workflow-aware eviction | 312.8 | +54% |
| w/o Tool-call TTL | 289.1 | +42% |
| w/o Session affinity | 398.2 | +96% ⚠️最大影响 |
| w/o Work stealing | 267.3 | +31% |
| w/o AFS fairness | 218.7 | +8% |

> 💡 **关键发现**：`Session affinity` 是最核心组件，贡献近半性能增益；`AFS` 对单任务影响小但在多租户中至关重要。

---

### 其他重要实验观察
- **KV Cache 再生开销显著下降**（Figure 1a）：
  - vLLM：38% 时间用于再生 cache
  - SAGA：降至 **8%**
- **参数鲁棒性强**（Table 9）：
  - 所有参数扰动范围内 TCT 变化 <8%，系统稳定可靠
- **工具延迟方差敏感性可控**（Table 10）：
  - 当工具延迟变异系数 CV ≤ 2.0 时，性能退化 <24%

---

## 4. 关键结论和发现

### 主要结论
1. **Request-level 抽象不再适用于 Compound AI Workloads**：
   - 忽略 workflow 结构会导致严重的 cache 再生开销和资源浪费。
2. **Workflow-aware 调度是高效 Agent Serving 的关键**：
   - 显式利用 AEG 可使在线缓存管理逼近离线最优（1.31× competitive ratio）。
3. **SAGA 实现了延迟与公平性的统一优化**：
   - 在保持高内存利用率的同时，大幅提升任务完成速度和 SLO 达成率。
4. **Trade-off 明确**：
   - SAGA 以约 **30% 吞吐量损失** 换取显著延迟改善，适合交互式部署（如 Copilot、Q Developer），不适合批处理场景。

---

### 方法的局限性（Section 1.5）
| 局限 | 说明 |
|------|------|
| **Workflow Observability 依赖** | 若无框架提示（LangChain/AutoGen 日志），需依赖 pattern inference，TCT 下降 12–18% |
| **动态多 Agent 框架支持弱** | 如 AutoGen 中 agent 自主辩论生成结构，难以静态建模 |
| **极端长尾工具调用仍可能触发误驱逐** | 黑天鹅事件（>5×P99）超出 TTL 预测范围 |
| **未覆盖 MoE 架构与跨数据中心部署** | 当前仅支持 dense 模型，且为单数据中心设计 |
| **高负载过载行为未经实证** | >95% 显存占用下的表现未充分测试 |

---

### 未来工作方向（Section 11）
1. **学习型 TTL 策略**：结合 bandit learning 实现 regret-bounded 的自适应 TTL。
2. **更紧的竞争比边界**：探索信息论下限，是否可达常数 competitive ratio？
3. **复杂性分析**：形式化证明 workflow-aware scheduling 是否 NP-hard。
4. **Geo-distributed AFS**：扩展至跨数据中心场景，考虑网络迁移代价。
5. **Multi-agent Coordination Optimization**：联合优化多个交互 Agent 的执行图。
6. **与 Speculative Execution 结合**：融合 SpecActions/Sherlock 等预测执行技术，进一步压缩延迟。

---

> 📌 **总体评价**：SAGA 是首个将 **workflow structure 显式引入分布式 LLM 调度** 的系统，通过“程序即调度单元”的范式转变，实现了对 compound AI workloads 的本质优化，为下一代 AI Agent 平台提供了重要的架构参考。

</details>

---

### 13. [Consistent Diffusion Language Models](https://arxiv.org/abs/2605.00161)

**Authors**: Hasan Amin, Yuan Gao, Yaser Souri, Subhojit Som, Ming Yin, Rajiv Khanna, Xia Song  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.00161v1  

#### Abstract
Diffusion language models (DLMs) are an attractive alternative to autoregressive models because they promise sublinear-time, parallel generation, yet practical gains remain elusive as high-quality samples still demand hundreds of refinement steps. In continuous domains, consistency training along th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Consistent Diffusion Language Models 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决 **Diffusion Language Models (DLMs)** 在实际应用中的一个核心瓶颈：**生成高质量文本需要数百次去噪步骤，导致效率低下**。尽管 DLMs 理论上支持并行生成，有望实现亚线性时间生成，但其缓慢的采样速度严重削弱了这一优势。

在连续域（如图像）中，**Consistency Models (CMs)** 通过利用 **Probability-Flow ODE (PF-ODE)** 实现了高效的少步甚至单步生成。然而，在离散的文本空间中，不存在类似的确定性轨迹（PF-ODE），因此无法直接将 CMs 的成功经验迁移到 DLMs 上。

### 提出的新方法和新思路
为了解决上述问题，作者提出了 **Multi-Path Discrete Consistency (MPDC)** 原则，并基于此构建了 **Consistent Diffusion Language Model (CDLM)**。

- **核心思想转变**：作者认为，在离散空间中，不应执着于寻找不存在的“确定性路径”，而应拥抱其固有的“随机性”。他们指出，离散扩散过程提供了一个丰富的、解析可得的**随机路径族**——即连接任意两个噪声水平 `s < t` 的**精确后验桥 (exact posterior bridge)**。
- **MPDC 原则**：该原则要求训练一个去噪器（denoiser）在期望意义上对这些不同的随机路径保持一致（path-invariant）。这意味着，从一个高度噪声的状态 `xt` 直接预测原始数据 `x0` 的结果，应该与先通过真实后验桥“跳跃”到一个中间状态 `xs`，再从 `xs` 预测 `x0` 的结果在期望上是等价的。
- **CDLM 框架**：CDLM 是一个**单阶段、无教师 (teacher-free)** 的训练框架。它通过最小化一个一致性损失（consistency loss）来训练一个时间条件预测器 `fθ(xt, t)`，该损失强制模型在不同长度的随机路径上的预测达成一致。

### 相比现有方法的优势
- **高效且高质量**：CDLM 能够在极少数步骤内（few-step regime）生成高质量文本，显著优于现有的基础 DLMs。
- **超越多阶段蒸馏模型**：令人惊讶的是，作为单阶段训练的模型，CDLM 的性能经常能匹配甚至超过那些复杂的、依赖预训练教师模型进行多轮蒸馏（如 SDTT, DUO-DCD）的基线模型。
- **统一视角**：MPDC 提供了一个统一的理论框架，证明了标准的掩码扩散（masked diffusion）、连续一致性模型、渐进式蒸馏（progressive distillation）和离散蒸馏方法（如 SDTT, DUO-DCD）都可以被视为 CDLM 框架在特定条件下的极限、近似或特例。
- **多样性保持**：相比一些蒸馏方法（如 DUO-DCD）因使用确定性投影而导致生成多样性下降（低熵），CDLM 通过使用真实的随机后验桥，能够更好地保持生成样本的多样性。

## 2. 核心实验方法和设置

### 数据集
- 主要预训练数据集为 **OpenWebText**。
- 条件生成任务使用了四个数据集：**OpenWebText**, **Lambada**, **Wikitext-103**, 和 **PTB**。

### 实验设置和评估指标
- **模型规模**：主要比较了 110M 参数的模型。
- **采样精度**：报告了 FP64 和 FP32 两种采样精度下的结果。
- **评估指标**：
  - **无条件生成 (Unconditional Generation)**：使用 **Perplexity (PPL)** 和 **Entropy** 作为核心指标。PPL 衡量生成质量，Entropy 衡量生成多样性。
  - **条件生成 (Conditional Generation)**：使用 **Perplexity (PPL)**、**BLEU** 分数和 **MAUVE** 分数。其中，BLEU 用于衡量生成文本与参考文本的相似度，以评估模型保留输入条件的能力。
- **采样步骤**：在广泛的采样预算下进行评估，从 4 步到 1024 步不等，重点关注少步（few-step）场景。

### 基线方法对比
实验将 CDLM 与两大类基线进行了对比：
- **基础模型 (Base Models, 单阶段训练)**：
  - **MDLM**: 基于掩码先验的标准扩散语言模型。
  - **DUO**: 基于均匀先验的扩散语言模型。
- **蒸馏模型 (Distilled Models, 多阶段训练)**：
  - **MDLM + SDTT**: 在 MDLM 基础上进行自蒸馏。
  - **DUO + DCD**: 在 DUO 基础上进行离散一致性蒸馏。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **无条件生成**：
  - 如图2和表1所示，**MCDLM-PPLOptimized** 在所有采样步数下均显著优于基础模型 **MDLM** 和 **DUO**。
  - 更重要的是，**MCDLM-PPLOptimized** 在大多数采样步数下（尤其是中高步数）的表现优于或匹敌多阶段蒸馏模型 **SDTT** 和 **DUO-DCD**。
  - **速度提升**：MCDLM-PPLOptimized 相比 MDLM 实现了 64x-128x 的加速。相比自回归（AR）基线，它在 32-64 步内即可达到相似性能，实现了 16x-32x 的 NFE（Number of Function Evaluations）减少。
- **条件生成**：
  - 如表3所示，**MCDLM-PPLOptimized** 在 PPL 和 BLEU 分数上均优于 **SDTT**，表明其在生成流畅且忠实于输入条件的文本方面具有更强的能力。
  - **DUO-DCD (greedy)** 因其确定性投影机制，生成的文本与输入条件差异巨大，BLEU 分数极低（被灰显），说明其在条件生成任务上表现不佳。

### 消融实验结果
论文通过消融实验验证了其设计选择的有效性：
- **步长调度器 (Step size scheduler)**：使用线性递减的调度器更有利于优化少步生成性能。
- **最大步长正则化 (Max-step regularizer)**：引入 `Kms` 项至关重要。没有它，模型会迅速发生模式崩溃（mode collapse），熵值急剧下降（如 `Kms=0` 时，8步熵仅为 3.2）。增加 `Kms` 可以平衡生成质量和多样性。
- **距离度量 (Distance metric)**：
  - **Forward KL**：不稳定，容易出现“均匀漂移”（uniform drift），导致困惑度灾难性恶化。
  - **Backward KL**：倾向于模式寻求（mode-seeking），虽然 PPL 较好，但会严重惩罚方差，导致熵值过低（模式崩溃）。
  - **JSD (Jensen-Shannon Divergence)**：提供了最佳的稳定性和质量-多样性权衡，是最终采用的选择。

## 4. 关键结论和发现

### 主要发现
1.  **随机一致性是离散加速的关键**：在离散空间中，用“期望意义上的随机路径一致性”替代“确定性路径一致性”是一个根本性的、成功的范式转变。
2.  **CDLM 是一个强大的单阶段框架**：CDLM 仅通过单阶段训练，无需任何教师模型，就能在生成质量和效率上达到甚至超越复杂的多阶段蒸馏方法，重新定义了快速离散生成的前沿。
3.  **统一的理论视角**：MPDC 原则为理解各种离散生成模型（从标准扩散到各类蒸馏方法）提供了一个统一的理论透镜，揭示了它们之间的内在联系。

### 方法的局限性
- **非单步生成**：CDLM 明确放弃了在完全噪声状态下进行单步生成的目标。最优的单步预测器是从纯噪声中预测无条件边际分布，这本身就是一个高度多模态的任务。因此，CDLM 的设计目标是在少数几步（如 4-8 步）内完成全局结构的解析，而非一步到位。
- **对训练稳定性有要求**：由于其自参照（self-referential）的损失函数，CDLM 的训练可能面临模式崩溃或均匀漂移的风险，需要精心设计的训练策略（如 EMA、JSD 散度、max-step 正则化）来稳定。

### 未来工作方向
- **通用性探索**：CDLM 框架不仅适用于语言，还可以推广到其他具有可处理后验的离散结构领域，如生物序列、图结构或程序合成。
- **作为更好的基础模型**：CDLM 可以作为一个更强大的基础模型，用于下一代离散生成方法的预训练或后训练，有望带来下游应用的全面提升。
- **算法优化**：CDLM 作为一个灵活的框架，其设计空间广阔，未来可以探索自适应调度、更优的发散度量等，进一步提升算法和端到端的效率。

</details>

---

### 14. [NLPOpt-Net: A Learning Method for Nonlinear Optimization with Feasibility Guarantees](https://arxiv.org/abs/2605.00260)

**Authors**: Bimol Nath Roy, Rahul Golder, MM Faruque Hasan  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.00260v1  

#### Abstract
Nonlinear Parametric Optimization Network (NLPOpt-Net) is an unsupervised learning architecture to solve constrained nonlinear programs (NLP). Given the structure of an NLP, it learns the parametric solution maps with guaranteed constraint satisfaction. The architecture consists of a backbone neural...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：NLPOpt-Net: A Learning Method for Nonlinear Optimization with Feasibility Guarantees**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
该论文针对**参数化非线性优化问题**（parametric NLP）的快速求解需求，解决以下核心挑战：
- 传统优化器（如 IPOPT、OSQP）虽能精确求解单个实例，但在需要**反复求解大量相似问题**（如模型预测控制、实时优化）时计算开销大。
- 现有基于神经网络（NN）的“学习优化”方法（如 PINNs、soft-constrained NN）虽然推理速度快，但**无法保证约束可行性**，导致预测解可能违反物理或系统约束，不可行。
- 现有硬约束方法（如 DC3、OptNet）在处理非线性约束时存在精度不足或效率低的问题。

### **提出了什么新方法或新思路**
提出 **NLPOpt-Net** —— 一种结合深度学习与优化理论的**无监督学习框架**，用于求解带约束的非线性规划（NLP），其核心设计包括：

- **双阶段架构**：
  - **Backbone NN**：前馈神经网络，输出初始预测 $\hat{y}$ 和对偶变量。
  - **k-layered Projection Layer**：一个可微分的多层投影模块，将 NN 预测**投影回原始约束流形** $S$ 上，确保最终输出严格满足所有约束。

- **创新的投影机制**：
  - 不同于传统的**欧氏距离最小化**投影，NLPOpt-Net 在每层投影中求解一个**局部二次近似的目标函数**（而非距离），同时满足线性化的约束。
  - 该近似目标为 $ \tilde{f}(y) = f(\hat{y}) + \nabla f(\hat{y})^T(y-\hat{y}) + \frac{1}{2}(y-\hat{y})^T H_d (y-\hat{y}) $，其中 $H_d = p \cdot \text{diag}(\nabla^2 f)$ 仅取 Hessian 对角元素。
  - 此设计赋予投影**下降性质**（descent property）：在凸问题下，投影后的解不会比原始 NN 预测更差，甚至可能更优。

- **高效求解与反向传播**：
  - 投影子问题为结构化 QP，采用**无矩阵求逆的 Chambolle-Pock (CP) 算法**进行前向求解。
  - 反向传播利用**隐函数定理**（implicit function theorem）计算梯度，避免展开整个迭代过程，实现高效训练。

- **部署优化**：
  - 训练后可将投影层与 NN 解耦，使用 **AOT 编译（C语言）** 加速推理，显著提升单次推断速度。

### **相比现有方法的优势**
| 方面 | NLPOpt-Net | 传统 NN / PINNs | DC3 | OptNet |
|------|------------|----------------|-----|--------|
| **可行性保证** | ✅ 严格满足所有约束（机器精度） | ❌ 软约束，常有违反 | ⚠️ 多数可行，但有小概率违反不等式 | ✅ 仅支持 QP |
| **最优性** | ✅ 近零 optimality gap | ❌ 常低于真最优值 | ❌ 平均 gap >150% | ✅ 但仅限 QP |
| **适用范围** | ✅ 支持 QP, QCQP, NLP, 非凸 NLP | ✅ 通用 | ✅ 支持非线性 | ❌ 仅限凸优化 |
| **推理速度** | ✅ 单次推断 ~2ms（native） | ✅ 极快 | ✅ 快 | ✅ 快 |
| **主动集预测** | ✅ 准确预测 active set 和对偶变量 | ❌ 无 | ⚠️ 不准确 | ✅ 但有限 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
论文未使用公开数据集，而是**人工生成多种参数化优化问题**，每类包含 2000 个实例（80% 训练，20% 测试），维度统一为：
- 决策变量：100
- 参数：50
- 等式约束：50
- 不等式约束：50

具体问题类型包括：
- **Convex QP**：二次目标 + 仿射约束
- **Convex QCQP**：二次目标 + 二次约束
- **Convex NLP**：二次目标 + 指数型非线性不等式约束
- **Nonconvex NLP**：二次目标 + $\sin(y)$ 项（来自 DC3 论文）

### **实验设置和评估指标**

#### **评估指标**
- **Optimality Gap (AOG)**：
  $$
  \text{AOG} = \frac{\text{Model Objective} - \text{Best Bound}}{\text{Average Objective}} \times 100\%
  $$
- **Feasibility**：
  - 最大/平均等式残差：$\|h(x,y)\|$
  - 最大/平均不等式违反：$\max(0, g(x,y))$
- **Speed**：
  - 单实例推理时间（end-to-end）
  - 批量推理时间（400 instances/batch）

#### **基线方法对比**
- **Optimizer**：CVXPY + OSQP (QP), SCS (QCQP/NLP), SLSQP (非凸)
- **NN**：标准 ANN + 软约束损失
- **Eq. NN**：预测部分变量，通过等式约束补全
- **DC3**：当前 SOTA 硬约束学习方法

#### **硬件环境**
- 训练：48核 CPU + NVIDIA A100 GPU + 360GB RAM
- 推理测试：Intel i7-13700 + 32GB RAM (Linux)

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 2）**

| 方法 | 问题类型 | Avg. Obj. | Max Eq. Viol. | Max Ineq. Viol. | Time (s/batch) | AOG (%) |
|------|----------|-----------|---------------|------------------|----------------|---------|
| **OSQP** | Convex QP | -8.730 | 0.000 | 0.000 | 0.0039 | 0.000 |
| **DC3** | Convex QP | -3.213 | 0.000 | 0.607 | 0.012 | >150% |
| **NLPOpt-Net** | Convex QP | **-8.730** | **0.000** | **0.000** | 0.0357 | **0.081** |
| **SCS** | Convex QCQP | 3.227 | 0.000 | 0.000 | 0.0514 | 0.000 |
| **NLPOpt-Net** | Convex QCQP | **3.227** | **0.000** | **0.000** | 0.4260 | **0.000** |
| **SLSQP** | Nonconvex NLP | -3.833 | 0.000 | 0.000 | 0.0638 | 0.000 |
| **NLPOpt-Net** | Nonconvex NLP | **-3.832** | **0.000** | **0.000** | 0.0780 | **0.026** |

### **与基线方法的对比结果**
- **可行性**：NLPOpt-Net 在所有问题上实现了**零约束违反**（机器精度），而 DC3 存在明显的不等式违反。
- **最优性**：NLPOpt-Net 的目标值与优化器几乎完全一致（AOG ≈ 0%），远优于 DC3（>150% gap）和普通 NN。
- **推理速度**：
  - 使用 `'native'`（C编译）模式，单实例推理时间 **2.06ms**，与 OSQP 同数量级。
  - 比 JAX 版本加速约 **50×**（95.14ms → 2.06ms）。

### **消融实验结果（Appendix B）**
- **样本量影响**（Table 6）：当训练样本减少至 20%，NLPOpt-Net 仍保持高精度，而 DC3 性能急剧下降。
- **约束数量影响**（Tables 4–5）：推理时间随约束增加而上升，但可行性始终保证。
- **主动集预测**（Table 7）：active set agreement 达到 **>99.7%**，表明能准确识别活跃约束，支持 multiparametric programming 应用。

---

## **4. 关键结论和发现**

### **主要发现**
1. **可行性与最优性可兼得**：NLPOpt-Net 成功实现了**严格可行性**与**近最优性**的统一，解决了软约束方法的根本缺陷。
2. **投影即优化**：提出的基于目标近似的投影具有**下降性质**，不仅能修复可行性，还能提升解的质量。
3. **泛化能力强**：不仅在凸问题上表现优异，在非凸 NLP 上也优于现有学习方法。
4. **实用性强**：通过 AOT 编译，推理速度达到实际部署水平，接近传统优化器。

### **方法的局限性**
- **训练时间较长**：由于包含迭代投影层，训练耗时高于普通 NN。
- **依赖问题结构**：投影效率受约束矩阵密度影响；稠密问题计算成本较高。
- **理论保证限于凸问题**：下降性质和收敛性分析基于凸性假设，非凸情况为经验成功。

### **未来工作方向**
- 扩展至**混合整数非线性规划**（MINLP）。
- 结合**稀疏矩阵技术**以处理超大规模问题。
- 探索**在线自适应学习**，应对参数分布漂移。
- 将框架应用于**实时 MPC、能源系统调度、材料设计**等实际场景。

---

> **代码与工具包**：作者已开源 `NLPOpt-Net`，支持 GPU 加速，可通过 `pip install nlpoptnet` 安装，项目地址：[https://github.com/souls-tamu/nlpoptnet](https://github.com/souls-tamu/nlpoptnet)

</details>

---

### 15. [ViLegalNLI: Natural Language Inference for Vietnamese Legal Texts](https://arxiv.org/abs/2605.00116)

**Authors**: Nhung Thi-Hong Duong, Mai Ngoc Ho, Tin Van Huynh, Kiet Van Nguyen  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.00116v1  

#### Abstract
In this article, we introduce ViLegalNLI, the first large-scale Vietnamese Natural Language Inference (NLI) dataset specifically constructed for the legal domain. The dataset consists of 42,012 premise-hypothesis pairs derived from official statutory documents and annotated with binary inference lab...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ViLegalNLI: Natural Language Inference for Vietnamese Legal Texts

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **越南语法律领域缺乏高质量的 Natural Language Inference (NLI)** 数据集。尽管越南有大量公开的法定文件，但尚未被系统地用于构建法律推理基准。
- 现有的 NLI 数据集主要集中于英语，且多为通用或医疗等非法律领域，无法满足越南语法律文本理解的需求。

### 🚀 提出的新方法与创新
- **提出 ViLegalNLI**：首个大规模、专为越南语法律文本设计的 NLI 数据集，包含 **42,012 对前提-假设对 (premise-hypothesis pairs)**，均标注为 **Entailment** 或 **Non-entailment**。
- **半自动数据生成框架**：
  - 利用大语言模型（如 Gemini-2.5 Flash）进行受控的假设生成；
  - 结合多模型交叉验证（GPT-4o, DeepSeek-R1, LLaMA-4 Scout）提升标签可靠性；
  - 引入 **artifact mitigation 策略**（如 PMI 分析 + 受控改写），减少模型依赖表面词汇线索的风险。
- **系统化质量控制流程**：
  - 使用 Fleiss’ Kappa ≥ 0.85 作为提示工程优化标准；
  - 采用共识过滤机制（至少两个模型同意才保留样本）；
  - 控制数据难度并确保跨子领域的公平分布。

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **语言覆盖** | 首个专注于越南语法律文本的 NLI 数据集，填补了低资源语言在法律 AI 中的关键空白 |
| **领域适配性** | 数据源自真实有效的法定条文，涵盖条件逻辑、交叉引用、专业术语等典型法律结构 |
| **构建效率与可扩展性** | 半自动化流程显著降低人工标注成本，同时保证高一致性 |
| **抗干扰能力** | 显式识别并缓解了 lexical artifacts（如“khi”倾向 Entailment，“du”倾向 Non-entailment），提高测试真实性 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **ViLegalNLI**（本文提出）：
  - 来源：从 [LuatVietNam.vn](https://luatvietnam.vn/) 抓取的 **168 份现行有效法定文件**；
  - 覆盖 **27 个法律子领域**（如行政组织、自然资源环境、知识产权等）；
  - 总量：42,012 条样本，按 8:1:1 划分为 train/dev/test；
  - 特征：平均句长 ~43 tokens，具有较高语义复杂性和结构多样性。

### ⚙️ 实验设置
- **任务形式**：二分类 NLI（Entailment vs. Non-entailment）
- **训练配置**：
  - 优化器：Adam，学习率 1e-5；
  - Batch Size：16（训练），32（评估）；
  - Epochs：5；
  - 使用 FP16 混合精度训练以提升效率；
  - Gradient Accumulation factor=2。
- **评估方式**：
  - 在 dev set 上选择最佳 checkpoint；
  - 最终报告 test set 上的结果。

### 📊 评估指标
| 指标 | 公式/说明 |
|------|---------|
| **Accuracy** | $(TP + TN) / (TP + TN + FP + FN)$ —— 整体准确率 |
| **Macro-F1 Score** | 对 Entailment 和 Non-entailment 类分别计算 F1 后取平均 —— 更适用于类别不平衡场景 |

### 🧪 基线方法对比
共测试四类模型：

| 类别 | 代表模型 |
|------|--------|
| **Multilingual PLMs** | mBERT, XLM-R (base/large), InfoXLM (base/large) |
| **Vietnamese Monolingual Models** | PhoBERT (base/large), viBERT, CafeBERT |
| **Improved Transformer Architectures** | DeBERTa V3 (base/large) |
| **Large Language Models (LLMs)** | 
| - Zero-shot Prompting | Gemma-3, Qwen2.5 |
| - Few-shot Prompting | Gemma-3, Qwen2.5 |
| - Fine-tuning | Gemma-2 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Test Set）

| Model | Accuracy (%) | F1-score (%) |
|-------|--------------|-------------|
| **Qwen2.5 (few-shot)** | **90.72** | **90.64** |
| **Gemma-3 (few-shot)** | 88.92 | 88.86 |
| **CafeBERT** | 87.49 | 87.36 |
| **InfoXLM (large)** | 87.98 | 87.85 |
| **XLM-R (large)** | 86.37 | 86.19 |
| **PhoBERT (large)** | 84.98 | 84.78 |
| **Gemma-2 (fine-tuned)** | 83.74 | 81.80 |
| **Gemma-3 (zero-shot)** | 80.76 | 80.72 |
| **Qwen2.5 (zero-shot)** | 79.62 | 77.83 |

> ✅ **结论**：**Few-shot LLMs 表现最优**，远超传统 fine-tuned encoder 模型。

### 🔁 与基线方法对比
- **Few-shot LLM > Fine-tuned PLM > Zero-shot LLM**
  - 尽管 Gemma-2 经过微调，其表现仍低于未微调但使用 few-shot 提示的 Gemma-3 和 Qwen2.5；
  - 表明 **上下文学习 (in-context learning)** 在法律推理中极具潜力。
- **Vietnamese-specific 模型优于通用 multilingual 模型**
  - CafeBERT > InfoXLM > XLM-R，在越南语文本上更具优势；
  - 体现领域适应预训练的重要性。

### 🔍 消融实验与分析（Ablation Insights）
虽然没有传统意义上的模块消融实验，但通过以下维度进行了深入分析：

#### （1）**假设长度影响**
- 中等长度（21–60 tokens）效果最好；
- 过短 → 信息不足；过长 → 噪声增加；
- **LLMs 对极端长度更鲁棒**，尤其 few-shot 设置下。

#### （2）**词汇重叠分析（Lexical Overlap）**
- 使用 Jaccard、LCS、New Word Rate 度量；
- 发现：
  - 准确率峰值出现在中等相似度区间（11%-30% Jaccard）；
  - 高重叠反而导致部分 PLM 性能下降（易误判为 Entailment）；
  - **LLMs 更少依赖表面匹配**，表现出更强的深层语义推理能力。

#### （3）**标签类别差异**
- 所有模型在 **Non-entailment 上表现更好**；
- **Entailment 更难**，需精确捕捉隐含逻辑关系；
- Few-shot LLMs 在两类间表现最均衡。

#### （4）**跨域泛化能力（Cross-Domain Evaluation）**
| Model | In-Domain Acc (%) | Cross-Domain Acc (%) |
|-------|-------------------|------------------------|
| XLM-R (large) | 86.37 | 87.55 |
| CafeBERT | 87.49 | 87.98 |

- **跨域性能略有提升或持平**，表明模型具备一定泛化能力；
- 但仍受限于深层法律推理（如例外处理、多条款联动）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Few-shot LLMs 是当前最优方案**  
   在无需微调的情况下，通过提示即可实现接近人类水平的法律推理能力，尤其 Qwen2.5 表现突出。

2. **越南语专用模型优于多语言模型**  
   PhoBERT、CafeBERT 等本地化预训练模型在越南法律文本上明显优于 mBERT/XLM-R，凸显语言特性建模的重要性。

3. **Entailment 比 Non-entailment 更具挑战性**  
   因其要求严格语义蕴含，涉及隐含后果、一般到具体的规则应用等复杂推理。

4. **模型仍依赖表面线索，深层推理能力有限**  
   - 错误分析显示，模型常因“词汇相似”而错误判断为 Entailment（Type 2: N→E）；
   - 在需要多步推理或结合多个条款时失败率高（Entailment Rules 6–9）。

5. **数据中的 annotation artifacts 可被有效缓解**  
   通过 PMI 分析识别 trigger words（如 “khi”→Entailment，“du”→Non-entailment），并进行受控改写，提升了数据质量和评估可信度。

### ⚠️ 局限性
- **数据来源单一**：全部来自成文法（statutory law），未包含判例、合同或司法解释；
- **推理深度有限**：目前仅支持句子级推理，缺乏段落或文档级复杂论证建模；
- **人工验证比例较低**：大部分依赖模型共识，可能存在系统性偏差残留；
- **动态法律更新未考虑**：法律会随时间修订，当前数据为静态快照。

### 🔮 未来工作方向
1. **扩展至更多法律文书类型**：纳入法院判决书、合同协议、行政处罚决定书等；
2. **引入更复杂的推理类型**：
   - 多跳推理（multi-hop reasoning）
   - 例外处理（exception handling）
   - 冲突解决（conflict resolution）
3. **发展段落/文档级 NLI 任务**；
4. **探索更好的 prompt calibration 方法**，减轻 LLM 的保守预测偏见（如默认输出 Non-entailment）；
5. **开放数据共享**：作者承诺将在论文接受后公开数据集链接，推动社区研究。

---

> 💡 **总体评价**：  
> ViLegalNLI 是越南法律 NLP 领域的一项奠基性工作。它不仅提供了首个高质量法律 NLI 基准，还展示了如何利用 LLM 构建可靠、可扩展的专业领域数据集。其实验全面、分析深入，为低资源语言的法律人工智能发展树立了新标杆。

</details>

---

### 16. [Confidence Estimation in Automatic Short Answer Grading with LLMs](https://arxiv.org/abs/2605.00200)

**Authors**: Longwei Cong, Sonja Hahn, Sebastian Gombert, Leon Camus, Hendrik Drachsler, Ulf Kroehne  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.00200v1  

#### Abstract
Automatic Short Answer Grading (ASAG) with generative large language models (LLMs) has recently demonstrated strong performance without task-specific fine-tuning, while also enabling the generation of synthetic feedback for educational assessment. Despite these advances, LLM-based grading remains im...

---

### 17. [A11y-Compressor: A Framework for Enhancing the Efficiency of GUI Agent Observations through Visual Context Reconstruction and Redundancy Reduction](https://arxiv.org/abs/2605.00551)

**Authors**: Michito Takeshita, Takuro Kawada, Takumi Ohashi, Shunsuke Kitada, Hitoshi Iyatomi  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.00551v1  

#### Abstract
AI agents that interact with graphical user interfaces (GUIs) require effective observation representations for reliable grounding. The accessibility tree is a commonly used text-based format that encodes UI element attributes, but it suffers from redundancy and lacks structural information such as ...

---

### 18. [Scale-Aware Adversarial Analysis: A Diagnostic for Generative AI in Multiscale Complex Systems](https://arxiv.org/abs/2605.00510)

**Authors**: Mengke Zhao, Guang-Xing Li, Duo Xu, Keping Qiu  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.00510v1  

#### Abstract
Complex physical systems, from supersonic turbulence to the macroscopic structure of the universe, are governed by continuous multiscale dynamics. While modern machine learning architectures excel at mapping the high-dimensional observables of these systems, it remains unclear whether they internali...

---

### 19. [Retrieval-Augmented Reasoning for Chartered Accountancy](https://arxiv.org/abs/2605.00257)

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Ali Imam Abidi  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.00257v1  

#### Abstract
The inception of Large Language Models (LLMs) has catalyzed AI adoption in the finance sector, yet their reliability in complex, jurisdiction-specific tasks like Indian Chartered Accountancy (CA) remains limited. The models display difficulty in executing numerical tasks which require multiple steps...

---

### 20. [MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents](https://arxiv.org/abs/2605.00356)

**Authors**: Tianyu Hu, Weikai Lin, Weizhi Zhang, Jing Ma, Song Wang  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.00356v1  

#### Abstract
Long-term conversational agents must decide which turns to store in external memory, yet recent systems rely on autoregressive LLM generation at every turn to make that decision. We present MemRouter, a write-side memory router that decouples memory admission from the downstream answer backbone and ...

---

### 21. [LLM-Emu: Native Runtime Emulation of LLM Inference via Profile-Driven Sampling](https://arxiv.org/abs/2605.00616)

**Authors**: Wei Da, Evangelia Kalyvianaki  
**Category**: cs.DC  
**Published**: 2026-05-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.00616v1  

#### Abstract
Realistic evaluation of LLM serving systems requires online workloads, dynamic arrivals, queueing, and the serving engine's local scheduling for execution batching, but running such experiments on GPUs is expensive. Existing simulators reduce this cost, but often operate offline or in time-warped mo...

---

### 22. [Towards Robust and Scalable Density-based Clustering via Graph Propagation](https://arxiv.org/abs/2605.00390)

**Authors**: Yingtao Zheng, Hugo Phibbs, Ninh Pham  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2605.00390v1  

#### Abstract
We present \textit{CluProp}, a novel framework that reimagines varied-density clustering in high-dimensional spaces as a label propagation process over neighborhood graphs. Our approach formally bridges the gap between density-based clustering and graph connectivity, leveraging efficient propagation...

---

### 23. [Structure-Aware Chunking for Tabular Data in Retrieval-Augmented Generation](https://arxiv.org/abs/2605.00318)

**Authors**: Pooja Guttal, Varun Magotra, Vasudeva Mahavishnu, Natasha Chanto, Sidharth Sivaprasad, Manas Gaur  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.00318v1  

#### Abstract
Tabular documents such as CSV and Excel files are widely used in enterprise data pipelines, yet existing chunking strategies for retrieval-augmented generation (RAG) are primarily designed for unstructured text and do not account for tabular structure. We propose a structure-aware tabular chunking (...

---

### 24. [Data Deletion Can Help in Adaptive RL](https://arxiv.org/abs/2605.00298)

**Authors**: Param Budhraja, Aditya Gangrade, Alex Olshevsky, Venkatesh Saligrama  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.00298v1  

#### Abstract
Deploying reinforcement learning policies in the real world requires adapting to time-varying environments. We study this problem in the contextual Markov Decision Process (cMDP) framework, where a family of environments is indexed by a low-dimensional context unknown at test time. The standard appr...

---

### 25. [Observable Performance Does Not Fully Reflect System Organization: A Multi-Level Analysis of Gait Dynamics Under Occlusal Constraint](https://arxiv.org/abs/2605.00778)

**Authors**: Jacques Raynal, Pierre Slangen, Jacques Margerit  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2605.00778v1  

#### Abstract
In biomechanical systems, observable performance is often used as a proxy for underlying system organization. However, this assumption implicitly presumes a correspondence between output metrics and internal system states that may not hold in adaptive systems. In this study, the vertical dimension o...

---

### 26. [How Frontier LLMs Adapt to Neurodivergence Context: A Measurement Framework for Surface vs. Structural Change in System-Prompted Responses](https://arxiv.org/abs/2605.00113)

**Authors**: Ishan Gupta, Pavlo Buryi  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.00113v1  

#### Abstract
We examine if frontier chat-based large language models (LLMs) adjust their outputs based on neurodivergence (ND) context in system prompts and describe the nature of these adjustments. Specifically, we propose NDBench, a 576-output benchmark involving two frontier models, three system prompt types ...

---

### 27. [Are You the A-hole? A Fair, Multi-Perspective Ethical Reasoning Framework](https://arxiv.org/abs/2605.00270)

**Authors**: Sheza Munir, Ahanaf Rodoshi, Sumin Lee, Feiran Chang, Xujie Si, Syed Ishtiaque Ahmed  
**Category**: cs.CL  
**Published**: 2026-05-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.00270v1  

#### Abstract
Standard methods for aggregating natural language judgments, such as majority voting, often fail to produce logically consistent results when applied to high-conflict domains, treating differing opinions as noise. We propose a neuro-symbolic aggregation framework that formalizes conflict resolution ...

---

### 28. [AirFM-DDA: Air-Interface Foundation Model in the Delay-Doppler-Angle Domain for AI-Native 6G](https://arxiv.org/abs/2605.00020)

**Authors**: Kejia Bian, Meixia Tao, Jianhua Mo, Zhiyong Chen, Leyan Chen  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.00020v1  

#### Abstract
The success of large foundation models is catalyzing a new paradigm for AI-native 6G network design: wireless foundation models for physical layer design. However, existing models often operate on channel state information (CSI) in the space-time-frequency (STF) domain, where distinct multipath comp...

---

### 29. [Smart Ensemble Learning Framework for Predicting Groundwater Heavy Metal Pollution](https://arxiv.org/abs/2605.00056)

**Authors**: T. Ansah-Narh, G. Y. Afrifa, J. B. Tandoh, K. Asare, M. Addi, K. E. Yorke, D. M. A. Akpoley, K. Aidoo, S. K. Fosuhene  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.00056v1  

#### Abstract
Groundwater in the Densu Basin is increasingly threatened by heavy metal contamination, but conventional methods fail to capture the statistical complexity and spatial heterogeneity of pollution indicators. A key challenge is modelling the Heavy Metal Pollution Index (HPI), which is typically skewed...

---

### 30. [Comparative Analysis of Polygon-Based and Global Machine Learning Models for Bus Occupancy Prediction](https://arxiv.org/abs/2605.00083)

**Authors**: Daniel Azenkot, Michael Fire, Eran Ben Elia  
**Category**: cs.LG  
**Published**: 2026-05-04  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2605.00083v1  

#### Abstract
Accurate forecasting of bus ridership (passengers numbers) is crucial for efficient management and optimization of public transport systems. Traditional forecasting models often fail to capture the unique and localized dynamics of different urban areas by treating the entire city as a single, homoge...

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
