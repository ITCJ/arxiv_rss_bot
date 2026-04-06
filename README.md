# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-06 07:16:23 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Analyzing Reverse Address Translation Overheads in Multi-GPU Scale-Up Pods](https://arxiv.org/abs/2604.02473)

**Authors**: Amel Fatima, Tuan Ta, Bradford M. Beckmann  
**Category**: cs.DC  
**Published**: 2026-04-06  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.02473v1  

#### Abstract
Distributed ML workloads rely heavily on collective communication across multi-GPU, multi-node systems. Emerging scale-up fabrics, such as NVLink and UALink, enable direct memory access across nodes but introduce a critical destination-side translation step: translating Network Physical Addresses (N...

---

### 2. [MSAO: Adaptive Modality Sparsity-Aware Offloading with Edge-Cloud Collaboration for Efficient Multimodal LLM Inference](https://arxiv.org/abs/2604.02945)

**Authors**: Zheming Yang, Qi Guo, Jun Wan, Jiarui Ruan, Yunqing Hu, Chang Zhao, Xiangyang Li  
**Category**: cs.DC  
**Published**: 2026-04-06  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.02945v1  

#### Abstract
Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities but impose substantial computational and latency burdens, posing critical challenges for deployment on resource-constrained edge devices. In this paper, we propose MSAO, an adaptive modality sparsity-aware of...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《MSAO: Adaptive Modality Sparsity-Aware Offloading with Edge-Cloud Collaboration for Efficient Multimodal LLM Inference》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
Multimodal Large Language Models (MLLMs) 虽然具备强大的跨模态推理能力，但在资源受限的边缘设备上部署时面临以下挑战：
- **高计算开销**：多模态输入（图像、视频、音频、文本）联合处理导致推理负担剧增。
- **通信延迟大**：完全依赖云端推理会因大量数据传输带来显著延迟和带宽压力。
- **静态卸载策略低效**：现有 edge-cloud 协同框架通常采用统一的卸载策略，忽略不同模态之间的异构性和冗余性（如视觉中的空间/时间冗余），造成资源浪费。

### 🚀 提出的新方法与创新思路
作者提出 **MSAO**（Modality Sparsity-Aware Offloading），一种自适应、感知模态稀疏性的边缘-云协同卸载框架，其核心创新包括：

#### （1）轻量级异构模态感知模块（Lightweight Heterogeneous Modality-Aware via Fine-Grained Sparsity）
- 引入 **Modality Activation Sparsity (MAS)** 度量指标，通过细粒度分析量化每个模态对当前任务的重要性。
- 在空间、时间和模态三个维度进行联合分析：
  - **Spatial Sparsity**：识别图像中不重要的背景区域并剪枝。
  - **Temporal Sparsity**：利用 LSH 哈希检测视频帧间的相似性，跳过冗余帧。
  - **Modal Sparsity**：基于 prompt 与各模态特征的相关性，判断模态的信息增益。
- 该模块仅需早期 encoder 层输出，计算开销极小（<2% 总延迟）。

#### （2）自适应推测性边云协同卸载机制（Adaptive Speculative Edge-Cloud Collaborative Offloading）
- 动态决定哪些模态/部分应本地处理、哪些应卸载至云端。
- 利用 **confidence-guided speculative execution** 隐藏通信延迟：
  - 边缘端运行小型 draft model 快速生成候选 token。
  - 当模型置信度高时，并行发送缓存供云端验证；否则立即异步卸载中间状态。
- 决策依据 MAS 分数 + 实时系统状态（带宽、内存等）。

### 🔍 相比现有方法的优势
| 维度 | 传统方法缺陷 | MSAO 改进 |
|------|--------------|----------|
| **卸载策略** | 统一处理所有模态，未考虑稀疏性 | 自适应感知 MAS，按需压缩或舍弃冗余模态 |
| **效率优化** | 被动等待通信完成 | 推测执行实现计算与通信重叠 |
| **资源利用** | 浪费带宽传输无用视觉信息 | 边缘预过滤，减少上行流量 |
| **精度保持** | 简单压缩易损失关键信息 | MAS 指导保留高重要性内容，维持准确率 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **VQAv2**：大规模视觉问答数据集，含 25 万张图像和 110 万个问题。从中随机采样 5,000 张用于测试。
- **MMBench**：综合性多模态评测基准，覆盖 20 种能力维度（对象识别、属性推理、场景理解等），使用完整测试集。

### ⚙️ 实验设置
- **硬件平台**：
  - **云端服务器**：NVIDIA A100 (40GB)
  - **边缘设备**：NVIDIA RTX 3090 (24GB)
- **模型配置**：
  - **边缘 draft model**：Qwen2-VL-2B（20亿参数）
  - **云端 full model**：Qwen2.5-VL-7B（70亿参数），共享 tokenizer 和架构设计以支持 speculative decoding。
- **网络模拟**：
  - 带宽设置为三种典型水平：200 Mbps（低）、300 Mbps（中）、400 Mbps（高）
  - RTT 固定为 20ms

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy** | VQAv2 使用标准 VQA 准确率；MMBench 报告 20 个维度平均准确率 |
| **Throughput** | 每秒生成 token 数（tokens/s） |
| **End-to-End Latency** | 输入提交到最终响应生成的总耗时（ms） |
| **Computing Overhead** | 推理过程消耗的总 FLOPs |
| **Memory Overhead** | 推理峰值 GPU 显存占用（GB） |

### 🔁 基线方法对比
1. **Cloud-only**：全部输入传至云端，由 Qwen2.5-VL-7B 完整推理。
2. **Edge-only**：仅在边缘运行 Qwen2-VL-2B 小模型。
3. **PerLLM [39]**：基于层划分的边云协同推理框架（layer-wise offloading）。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 指标 | 提升幅度 | 说明 |
|------|---------|------|
| **End-to-End Latency** | ↓ **30%** | 平均降低约 30%，最高达 50% 以上 |
| **Resource Overhead** | ↓ **30%–65%** | 包括计算与内存开销大幅下降 |
| **Throughput** | ↑ **1.5× – 2.3×** | 显著提升系统吞吐能力 |
| **Accuracy** | ≈ Cloud-only | 与云端全模型差距 <0.4%，远优于其他基线 |

### 🆚 与基线方法对比结果

#### （1）准确性（Accuracy）
| 方法 | VQAv2 @200Mbps | MMBench @400Mbps |
|------|----------------|------------------|
| Cloud-only | 76.3% | 76.5% |
| Edge-only | 61.4% | 61.2% |
| PerLLM | 71.3% | 69.9% |
| **MSAO** | **76.1%** | **76.3%** |

✅ **结论**：MSAO 在几乎不牺牲准确性的前提下，实现了接近云端最优的性能。

#### （2）吞吐量（Throughput）
- 在 VQAv2 上（400 Mbps）：
  - MSAO：**128 tokens/s**
  - Cloud-only：35 tokens/s → **提升 2.66×**
  - Edge-only：~60 tokens/s → **提升 >2×**
  - PerLLM：~75 tokens/s → **提升 ~1.7×**

#### （3）端到端延迟（Latency）
- 相比 Cloud-only：**降低 >50%**
- 相比 Edge-only：**降低 45%–55%**
- 相比 PerLLM：**降低 >30%**

#### （4）资源开销
- **计算开销（FLOPs）**：
  - 相比 Cloud-only：↓30%–65%
  - 相比 PerLLM：↓35%–50%
- **显存开销（GPU Memory）**：
  - 在 200 Mbps 下，边缘显存从 25.0 GB（Cloud-only）降至 **9.0 GB**（↓64%）
  - MSAO 显存消耗随带宽变化极小，表现出强鲁棒性。

### 🔍 消融实验结果（Ablation Study）

比较两个变体：
- **w/o Modality-Aware**：禁用 MAS 感知，采用统一卸载策略
- **w/o Collaborative Scheduling**：移除推测性调度，使用静态任务分配

| 变体 | Accuracy ↓ | Latency ↑ | Compute/Memory ↑ |
|------|-----------|----------|------------------|
| w/o Modality-Aware | -6.8% (VQAv2), -7.6% (MMBench) | — | — |
| w/o Collaborative | — | +48.3% (VQAv2), +45.2% (MMBench) | 显著上升 |

✅ **结论**：两个核心组件缺一不可，共同支撑高效且精准的推理。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **模态稀疏性是可被有效建模的关键信号**：
   - 视觉输入普遍存在空间/时间冗余，而文本/音频更密集。
   - MAS 指标能以极低成本识别冗余信息，指导智能卸载。

2. **自适应卸载 + 推测执行可显著提升效率**：
   - 通过 confidence-guided speculative decoding，实现“计算掩盖通信”，大幅提升吞吐并降低延迟。

3. **MSAO 实现了性能与效率的帕累托最优**：
   - 在几乎不损失 accuracy 的前提下，全面超越各类 baseline，在 latency、throughput、resource overhead 多方面取得显著优势。

### ⚠️ 方法的局限性
- **依赖 dual-model 架构**：需要边缘 draft model 与云端 full model 兼容，限制了通用性。
- **MAS 模块仍需一定计算资源**：尽管已很轻量，但在极端低功耗边缘设备上可能仍有负担。
- **未考虑动态环境下的在线学习能力**：当前 MAS 参数为离线设定，缺乏对长期分布漂移的适应机制。

### 🔮 未来工作方向
1. **引入在线自适应机制**：根据实时反馈动态调整 MAS 权重与 confidence threshold。
2. **扩展至更大规模 edge-cloud 系统**：在真实分布式环境中验证可扩展性。
3. **探索 zero-shot 模态重要性估计**：减少对特定任务标注数据的依赖。
4. **结合 model compression 与 sparsity-aware offloading**：进一步压缩传输内容。

---

> **总结一句话**：  
> MSAO 通过 **感知模态稀疏性（MAS）+ 自适应推测性卸载**，首次将“**按需传输、智能分工、计算通信重叠**”理念系统化应用于 MLLM 边云协同推理，在保持高精度的同时实现了 **30% 降延迟、65% 降资源、2.3× 吞吐提升**，为高效多模态 AI 部署提供了新范式。

</details>

---

### 3. [AdaHOP: Fast and Accurate Low-Precision Training via Outlier-Pattern-Aware Rotation](https://arxiv.org/abs/2604.02525)

**Authors**: Seonggon Kim, Alireza Khodamoradi, Kristof Denolf, Eunhyeok Park  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 8.0  
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
低精度训练（**Low-Precision Training, LPT**）在大语言模型（**LLMs**）中面临一个核心挑战：**outliers**（离群值）——即少数极端值会显著放大量化误差，导致模型收敛困难和性能下降。传统方法通常采用固定的 **Hadamard 变换** 来抑制 outliers，但这些方法忽略了不同张量（weights, activations, gradients）之间 outlier 结构的差异性。

本文指出，**统一应用固定变换是次优甚至有害的**，因为其有效性高度依赖于变换方向与 outlier 结构之间的对齐关系。

---

### 提出了什么新方法或新思路
作者提出了 **AdaHOP**（Adaptive Hadamard transform with Outlier-Pattern-aware strategy），一种基于 outlier 模式感知的自适应低精度训练框架。其核心思想包括：

- **系统性分析 outlier 模式**：首次对 LLM 中 weights、activations 和 gradients 的 outlier 分布进行了系统研究，识别出三种结构化模式：
  - **Row-wise (R)**：outliers 集中在某些行
  - **Column-wise (C)**：outliers 集中在某些列
  - **None (N)**：无明显集中趋势
- **模式对决定最优策略**：每个矩阵乘法涉及两个操作数（如 `Gy` 和 `X`），其组合形成 **pattern pair**（如 RC, CN, NN 等）。不同 pair 对应不同的最优处理策略。
- **自适应策略选择**：
  - 对可被内维平滑（inner-dimension smoothing）缓解的 pair（如 CN, NN）直接使用 **IHT**（Inner Hadamard Transform）
  - 对 IHT 无效的 pair（如 RN, RC, CC）则结合 **Selective Outlier Extraction (OE)**，将主导 outliers 路由到高精度路径（BF16），其余部分用低精度计算。

---

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **准确性** | 显著降低量化误差，在 MXFP4 精度下达到接近 BF16 的训练质量 |
| **效率** | 通过硬件感知的 **Triton 内核融合** 实现高效执行，避免额外开销 |
| **通用性** | 策略基于实际观测的稳定模式，适用于多种 LLM 架构 |
| **灵活性** | 提供两个版本：AdaHOP-Lv1（轻量混合精度）、AdaHOP-Lv2（关键层全精度），可在精度与效率间灵活权衡 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- 主要训练数据集：**C4**（Colossal Clean Crawled Corpus）
- 下游零样本评估任务：
  - **PIQA**
  - **HellaSwag**
  - **ARC-Easy**
  - **LAMBADA**

---

### 实验设置和评估指标

#### 模型规模
- **Llama3.2-1B**, **Llama3.2-3B**, **Instella-3B**, **Llama3.1-8B**

#### 训练配置
- 序列长度：4096 tokens
- 全局 batch size：128
- 优化器：AdamW（学习率 4e-4，epsilon 1e-8）
- Warmup 步数：200
- 总训练 token 数按 Chinchilla 定律缩放（如 1B 模型训练 40B tokens）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Training Loss** | 与 BF16 的差距 |
| **Zero-shot Accuracy (%)** | 四个下游任务平均准确率 |
| **Memory Consumption** | 线性层内存占用（GB） |
| **Throughput (tok/s)** | 每秒处理 token 数 |
| **Kernel Latency** | GEMM 核心延迟（ms） |

---

### 基线方法对比
| 方法 | 简介 |
|------|------|
| **BF16** | 全精度训练，作为上界 |
| **MXFP4** | 朴素低精度量化，无任何 outlier 抑制 |
| **MXFP4+Hadamard** | 统一应用 IHT |
| **Tseng et al.** | 使用随机 Hadamard 进行无偏梯度估计 |
| **HALO** | 固定使用 OHT 处理梯度路径 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📈 零样本准确率（Table 3）
| 方法 | Llama3.2-1B (Avg) | Instella-3B (Avg) | Llama3.1-8B (Avg) |
|------|-------------------|-------------------|--------------------|
| BF16 | 52.76 | 55.75 | 61.68 |
| MXFP4+Hadamard | 51.70 | 54.56 | — |
| HALO | 52.43 | 54.86 | 60.37 |
| **AdaHOP-Lv2** | **52.73** | **56.05** | **61.43** |

> ✅ **AdaHOP-Lv2 在所有模型上均取得最佳或接近最佳的零样本准确率**，尤其在 Instella-3B 上超越 BF16（56.05 > 55.75）

---

#### 💾 内存与吞吐（Table 4 & 5）

| 方法 | Memory (GB) | Throughput (tok/s) | Speedup vs BF16 |
|------|-------------|---------------------|------------------|
| BF16 | 76.00 | 12,945.51 | 1.0× |
| AdaHOP-Lv1 | 20.94 | 13,246.88 | **1.59–1.80×** (kernel) |
| AdaHOP-Lv2 | 28.04 | 13,134.02 | **1.59–1.80×** (kernel) |

- **内存压缩比**：
  - AdaHOP-Lv1 达到 **3.6×** 压缩（76 → 20.94 GB）
  - AdaHOP-Lv2 达到 **2.7×** 压缩（76 → 28.04 GB）
- **内核级加速**：高达 **1.8× GEMM 加速**

---

#### 🔍 消融实验与关键发现
- **模式稳定性验证**（Figure 4 & 6）：
  - Outlier 模式在训练过程中高度稳定，支持一次性校准（calibration）
  - 不同 block 深度存在梯度模式变化（早期和晚期出现 Row-wise）
- **策略有效性验证**（Figure 3）：
  - IHT 仅对 CR pair 有效，对 RC、RR、CC 等无效甚至有害
  - OE + IHT 在 RN、RC、CC 等难处理 pair 上显著优于单一变换
- **版本对比**：
  - AdaHOP-Lv2 在注意力敏感层（Key/Value projection）保留 BF16，带来约 **0.8% 准确率提升**（Instella-3B）

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Outliers 并非随机分布**，而是呈现三种结构化、稳定的模式（Row-wise, Column-wise, None）。
2. **统一的 Hadamard 变换策略是次优的**，其效果严重依赖于变换方向与 outlier 模式的对齐程度。
3. **最优策略应随 pattern pair 动态调整**：IHT 适用于部分 pair，而其他需引入 selective OE。
4. **模式具有时间稳定性**，可通过短时 BF16 校准阶段可靠检测，无需运行时判断。
5. **硬件感知实现至关重要**：通过 Triton 内核融合，AdaHOP 将自适应逻辑的开销最小化，实现高效执行。

---

### 方法的局限性
1. **固定 Walsh-Hadamard 矩阵**：未探索 learnable rotation matrices（如 SpinQuant）可能带来的进一步增益。
2. **当前分析集中在 Llama-family 和 Instella 模型**：是否泛化到 MoE 架构（如 Mixtral）、不同归一化方式等尚待验证。
3. **k=64 固定提取数量**：未根据每层 outlier 严重程度动态调整，可能存在次优。
4. **依赖特定硬件特性**（CDNA4 的 MFMA tile 尺寸）：移植到其他架构可能需要重新调优。

---

### 未来工作方向
1. **结合 Learnable Rotations**：将 AdaHOP 与数据驱动的旋转矩阵（如 SpinQuant）结合，实现更精细的能量重分布。
2. **扩展至更多模型架构**：验证 outlier 模式在 Mixtral、Gemma、Phi 等模型中的普适性。
3. **推广至其他数值格式**：将 pattern-aware 设计应用于 INT4、FP8 等其他低精度格式。
4. **自适应 outlier 提取机制**：设计 per-layer 或 per-step 的动态 k 选择策略，以更好平衡精度与效率。
5. **端到端训练加速优化**：不仅优化 GEMM，还考虑 Attention、Normalization 等模块的协同低精度设计。

---

> ✅ **总结一句话**：  
> AdaHOP 通过揭示并利用 LLM 中 outlier 的结构化与稳定性特征，提出了一种“因材施教”的自适应低精度训练范式，在保持 BF16 级别训练质量的同时，实现了高达 3.6× 内存压缩和 1.8× 内核加速，为高效 LPT 提供了新的设计原则。

</details>

---

### 4. [Fast NF4 Dequantization Kernels for Large Language Model Inference](https://arxiv.org/abs/2604.02556)

**Authors**: Xiangbo Qi, Chaoyi Jiang, Murali Annavaram  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 8.0  
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
当前大型语言模型（LLMs）参数量已远超单个GPU的显存容量，因此广泛采用 **NF4 (4-bit NormalFloat)** 量化技术以实现4倍内存压缩。然而，NVIDIA当前架构（如Ampere A100）不支持原生4-bit计算，推理时必须将NF4权重**反量化为FP16**格式进行矩阵乘法运算。

这一过程导致严重的性能瓶颈：
- 反量化操作频繁访问全局内存（global memory），延迟高达290个时钟周期；
- 原有实现中每个线程重复加载相同的查找表（LUT），造成大量冗余内存流量；
- 使用多级条件分支树解码NF4索引，引发warp divergence，降低SIMT效率。

这些因素使得**反量化开销占端到端延迟的21%-40%**，成为制约LLM推理吞吐的关键瓶颈。

---

### 🚀 提出的新方法与创新思路

作者提出一种**轻量级共享内存优化方案**，通过以下两个核心技术改进NF4反量化内核：

#### （1）**Shared Memory LUT缓存**
- 将仅需64字节的16元素NF4查找表从constant memory加载至**shared memory**；
- 由每个thread block中的thread 0一次性加载，其余线程同步后复用；
- 利用shared memory的广播能力（broadcast read），避免重复global memory访问；
- 内存延迟从290 cycle降至约19 cycle，获得**12–15×的访问速度提升**。

#### （2）**简化索引计算逻辑**
- 替换原有4层条件判断树（conditional tree decoding）为直接位操作（bit masking & shifting）；
- 每个权重的索引计算从**7条指令减少至2条**，且无分支；
- 完全消除warp内部控制流分歧（warp divergence），提高执行效率。

该方法设计为**即插即用模块**，兼容HuggingFace Transformers与BitsAndBytes生态系统，无需离线预处理或模型转换。

---

### 🔍 相比现有方法的优势

| 维度 | 本文方法 | 现有方法（如BitsAndBytes） |
|------|----------|-----------------------------|
| 内存访问模式 | 共享内存 + 广播读取 | 每线程独立访问global memory |
| 索引计算方式 | Bit操作（无分支） | 多级条件判断树（有warp divergence） |
| 工程复杂度 | 极低（仅修改kernel） | 高（需kernel fusion、定制编译器等） |
| 生态兼容性 | 完全兼容HF/BitsAndBytes | 往往破坏现有流程 |
| 显存占用 | +64 bytes / block（可忽略） | 无额外开销 |

> ✅ **优势总结**：在极小工程代价下，实现显著性能加速，同时保持生态兼容性和数值一致性（bit-exact output）。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **GSM8K**：用于生成输入prompt，测试数学推理任务下的推理性能；
- 所有实验均基于真实用户提示序列，确保负载具有代表性。

---

### ⚙️ 实验设置

| 项目 | 配置 |
|------|------|
| 硬件平台 | 单块 **NVIDIA A100-80GB GPU**<br>CPU: AMD EPYC 7513 (32核), RAM: 64GB |
| 软件环境 | CUDA 12.6, PyTorch 2.1, BitsAndBytes 0.47.0 |
| 锁频设置 | 使用`nvidia-smi`锁定GPU频率为最大值（1410 MHz），排除动态调频干扰 |
| 测量工具 | PyTorch Profiler + CUPTI接口，微秒级精度，包含kernel launch overhead和memory transfer |
| 评估模型 | **Gemma 27B**, **Qwen3 32B**, **Llama3.3 70B** |
| 批次大小（batch size） | 2, 4, 8, 16, 32, 64 |
| 输出验证 | token-by-token比对，确认优化前后输出完全一致（bit-exact） |

---

### 🧪 基线方法对比
- **Baseline**: 开源的 **BitsAndBytes** 实现中的NF4反量化kernel；
- **Optimized**: 本文提出的共享内存+位操作优化版本；
- 对比维度：
  - 端到端推理延迟（end-to-end latency）
  - 吞吐量（tokens/sec）
  - 反量化kernel执行时间
  - 指令数与warp效率

---

## 3. 主要实验结果和性能指标

### 📈 Kernel-Level 性能提升（表 II）

| 模型 | 平均加速比（kernel speedup） |
|------|----------------------------|
| Gemma 27B | 2.10× |
| Qwen3 32B | 2.19× |
| Llama3.3 70B | 2.04× |

> ✅ 在所有模型和batch size上均实现 **2.0–2.2× 的反量化kernel加速**，表明优化针对的是底层硬件瓶颈而非特定模型结构。

#### 关键原因分析：
- **Shared memory访问延迟仅为global memory的~6.5%**（19 vs 290 cycles）；
- **每block节省64次重复LUT读取**（64 threads × 1 load each → 1 load total）；
- **指令数下降71%**，从7条分支指令简化为2条位操作指令。

---

### 🚀 End-to-End 性能提升（图4）

| 模型 | 最高端到端加速比 | 典型场景说明 |
|------|------------------|------------|
| **Llama3.3 70B** | **1.54×**（batch=2） | 大模型受益最明显，因反量化占比更高 |
| **Qwen3 32B** | 1.29×（batch=32） | 中等规模模型也有显著收益 |
| **Gemma 27B** | 1.32×（batch=64） | 小模型在大batch下增益上升 |

> 💡 加速效果随模型增大而增强，因为更大模型拥有更多层数和参数，反量化总耗时占比更高。

---

### 📊 吞吐量提升（tokens per second）

| 模型 | 示例提升 |
|------|--------|
| Llama3.3 70B | +54% 吞吐（batch=2） |
| Qwen3 32B | 从283 → 368 tokens/s（+30%） |
| Gemma 27B | 从506 → 633 tokens/s（+25%，batch=64） |

> 这些提升可直接转化为生产系统中更高的服务容量和更低的单位推理成本。

---

### ❌ 消融实验（隐含于文中分析）

虽然未明确列出消融表格，但文中通过多个角度验证各组件有效性：

| 组件 | 效果证据 |
|------|--------|
| **Shared Memory LUT** | 分析显示global memory访问减少64×/block；latency优势达12–15× |
| **Bit Manipulation Indexing** | 指令数从7→2，消除branch divergence，warp效率接近理想状态 |
| **Single-thread Load策略** | 实测小数据量下协调开销大于并行收益，串行加载更高效 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **反量化是当前NF4推理的主要瓶颈**  
   - 占端到端延迟21%-40%，尤其在中小batch下更为突出；
   - 并非计算密集型，而是典型的**memory-bound**操作。

2. **GPU内存层次利用不足是根本原因**  
   - 当前实现忽视shared memory的数据重用潜力；
   - 重复global memory访问浪费带宽和时钟周期。

3. **轻量级优化也能带来巨大收益**  
   - 仅使用**64 bytes shared memory / block**；
   - 修改极少代码即可集成进现有框架；
   - 实现**2.0–2.2× kernel加速** 和 **最高1.54×端到端提速**。

4. **大模型受益更显著**  
   - 模型越大，反量化总耗时越长，优化带来的相对收益越高。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **依赖Ampere及以上架构特性** | shared memory广播、constant cache等特性在旧GPU上可能表现不同 |
| **对compute-bound场景增益有限** | 若模型已高度优化（如使用TensorRT-LLM），其他部分成为瓶颈，则本优化边际效益下降 |
| **仅适用于静态LUT结构** | NF4的16-level固定分布才适合此缓存策略，动态量化方法难以应用 |

---

### 🔮 未来工作方向

1. **扩展至其他量化格式**  
   - 如FP4、INT4或其他自定义4-bit格式，探索通用化LUT缓存架构。

2. **结合Kernel Fusion进一步优化**  
   - 将dequantize + matmul融合为单一kernel，减少中间数据驻留。

3. **适配新一代GPU架构（Hopper, Blackwell）**  
   - 利用新的memory hierarchy特性（如L2 cache partitioning）进一步压榨性能。

4. **支持分布式推理场景下的跨设备LUT共享**  
   - 在multi-GPU setup中统一管理常量表，减少冗余传输。

---

## ✅ 总结

| 项目 | 内容 |
|------|------|
| **核心思想** | 利用shared memory缓存NF4 LUT + 位操作替代分支解码 |
| **关键技术** | 单线程加载 + `_syncthreads()` + bit shift/mask |
| **性能成果** | 2.0–2.2× kernel加速，最高1.54×端到端提速 |
| **工程价值** | 轻量、兼容、易部署，可在现有HF生态中即插即用 |
| **理论意义** | 揭示了“轻量级内存优化”在LLM推理中的巨大潜力 |

> 🔧 **一句话总结**：本文通过一个简单却深刻的GPU内存层级优化，在几乎零迁移成本的前提下，显著缓解了NF4反量化瓶颈，为大规模语言模型在现有硬件上的高效推理提供了实用解决方案。

</details>

---

### 5. [Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms](https://arxiv.org/abs/2604.02927)

**Authors**: Andreas Boltres, Niklas Freymuth, Benjamin Schichtholz, Michael K\"onig, Gerhard Neumann  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.02927v1  

#### Abstract
Routing algorithms are crucial for efficient computer network operations, and in many settings they must be able to react to traffic bursts within milliseconds. Live telemetry data can provide informative signals to routing algorithms, and recent work has trained neural networks to exploit such sign...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**神经路由算法在现实网络部署中的可行性问题**展开研究。现有基于深度学习的路由算法存在以下关键缺陷：
- **忽略通信延迟**：许多方法假设可以获取无延迟的全局网络状态（如“鸟瞰”视图），这在真实网络中无法实现。
- **纯局部观测限制**：另一些分布式方法仅依赖本地遥测数据，缺乏对全局网络态势的有效感知。
- **训练与部署脱节**：训练时忽略推理延迟（inference delay）和状态传播延迟，导致模型在真实近实时控制场景下性能下降。

这些问题使得现有神经路由算法难以在需要毫秒级响应的真实网络环境中部署。

### 提出的新方法与思路
作者提出了一个系统性的解决方案，其核心创新包括：

#### （1）**延迟感知的仿真框架（Delay-Aware Simulation Framework）**
- 将遥测感知路由建模为一个**延迟感知的闭环控制问题**。
- 显式模拟了**通信延迟**（状态从各节点汇聚到观察者所需时间）和**推理延迟**（神经网络计算动作的时间）。
- 支持多种部署模式（Central-Single, Local-Multi等），以评估不同架构在真实延迟下的表现。

#### （2）**新型神经路由算法 LOGGIA**
- **全称**：LOg-space link weight prediction on Graphs with Guided update epochs and Implicit-Alpha entropy adaptation.
- **架构设计**：
  - 使用 **Message Passing Networks (MPNs)** 处理带有属性的拓扑-遥测图。
  - 在**对数空间（log-space）预测链路权重**，提升数值稳定性。
  - 采用**两阶段策略**：GNN 输出链路权重 → Dijkstra 算法计算最短路径。
- **训练机制**：
  - 结合 **Imitation Learning (IL) 预训练**（模仿 EIGRP 等静态协议）进行“热启动”。
  - 使用改进的 **Proximal Policy Optimization (PPO)**，引入最大熵探索和早停机制以提高训练稳定性。

### 相比现有方法的优势
| 方面 | 现有方法缺陷 | LOGGIA 的优势 |
|------|-------------|--------------|
| **延迟处理** | 忽略或理想化延迟 | 显式建模通信与推理延迟，更贴近现实 |
| **可扩展性** | 多数方法难以扩展到大网络 | 可在单一小拓扑上训练，泛化至100节点网络 |
| **部署模式** | 中心化或非协调分布式 | 支持 Fully Local Distributed (Local-Multi)，性能最优 |
| **训练效率** | RL 训练不稳定 | IL 预训练显著提升收敛速度与最终性能 |

---

## 2. 核心实验方法和设置

### 数据集与网络拓扑
实验在多种合成与真实网络拓扑上进行：
- **mini5**：5节点小型合成网络。
- **B4**：Google 跨数据中心网络（12节点，17条链路）。
- **GEANT**：欧洲科研教育网络（27节点，38条链路）。
- **nx-family**：一系列可变规模的合成拓扑（nx-XS: 6–10节点, nx-S: 11–25节点, ..., nx-L: 51–100节点）。

流量模型包含 **80% TCP 流量** 和 **20% UDP 流量**，符合现代数据中心特征。

### 实验设置
- **控制粒度**：每 **5ms** 执行一次路由决策（near-real time）。
- **仿真平台**：基于 `ns-3` 实现包级模拟，通过 `ns3-ai` 使用共享内存进行高速交互。
- **训练配置**：
  - 每个算法训练8个随机种子。
  - 使用 **IL 预训练10轮 + PPO/MAPPO 训练10轮**。
  - 总训练步数：128,000 steps。

### 评估指标
- **主指标**：**Goodput（有效吞吐量）**，即成功送达的数据量（单位：MB）。
- 辅助指标：
  - 平均延迟（Delay）
  - 队列负载（Queue Load）
  - TCP 丢包数（TCP Discard）

### 基线方法对比
| 类型 | 基线名称 | 描述 |
|------|--------|------|
| **静态路由** | SPRIP | 最少跳数路由（类似 RIP） |
| | SPEIGRP | 基于带宽/延迟的链路成本（类似 EIGRP） |
| | SPOSFP | 基于带宽的链路成本（类似 OSPF） |
| **神经路由** | MAGNNETO | 中心化多智能体 GNN 路由 |
| | FieldLines | 分布式但仅用本地遥测 |
| | M-Slim | 中心化控制器，忽略延迟 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Figure 5 和 Tables 3–4）

| 方法 | B4 拓扑 Goodput (MB) | GEANT 拓扑 Goodput (MB) |
|------|---------------------|------------------------|
| **SPEIGRP (最佳 SP 基线)** | 329.47 | 441.86 |
| **MAGNNETO** | 311.57 | 420.94 |
| **FieldLines** | 296.18 | 450.33 |
| **M-Slim** | 310.99 | 438.19 |
| **LOGGIA (ours)** | **333.37** | **460.71** |

> ✅ LOGGIA 在两个真实拓扑上均显著优于所有基线，是**唯一超越静态最短路径协议的神经路由算法**。

### 与基线方法的对比结果
- 在所有测试拓扑（mini5, B4, nx-XS, GEANT）中，**除 LOGGIA 外的所有神经路由算法在考虑延迟后性能均低于静态路由基线**。
- LOGGIA 的优势在更大、更复杂的拓扑中更为明显。
- 在延迟敏感设置下，中心化方法（如 MAGNNETO）因通信瓶颈性能急剧下降。

### 消融实验结果（Ablation Studies）

#### （1）架构组件消融（Figure 11）
LOGGIA 的三个关键设计均带来正向增益：
- 使用 **log-space 权重预测** → 提升稳定性。
- 直接操作原始图（而非 line digraph）→ 更高效。
- 更深的 GNN（L=4）和更大隐藏维度（d=32）→ 更强表达能力。

#### （2）训练机制消融（Figure 14）
- **IL 预训练至关重要**：单独使用 IL 效果差，但作为 PPO 前置阶段可显著提升最终性能并降低方差。
- **BC（行为克隆）不如交互式 IL**：非交互式预训练效果较差。

#### （3）路径级探索消融（Figure 13）
- 尝试使用 Yen’s algorithm 进行路径采样探索 → 性能**大幅下降**。
- 表明当前的**边级（edge-level）探索已足够有效**。

#### （4）多智能体训练消融（Figure 15）
- 不同 MAPPO 变体（IPPO, HVPPO）性能相近。
- **训练时使用中央观察者（Central observer）反而效果更好**，尽管部署时使用 Local-Multi。
- 表明**训练与部署架构可以解耦**。

---

## 4. 关键结论和发现

### 主要发现
1. **延迟是决定性因素**：
   - 忽略通信和推理延迟会导致对神经路由算法性能的严重高估。
   - 在延迟感知设置下，几乎所有现有神经路由算法都**无法超越传统静态路由**。

2. **LOGGIA 是首个真正有效的延迟感知神经路由算法**：
   - 在 Local-Multi 部署模式下，**始终优于最短路径基线**。
   - 得益于 IL 预训练、log-space 参数化和稳定 RL 优化。

3. **Fully Local Distributed 部署最优**：
   - **Local-Multi**（每个路由器独立观测与决策）是唯一能在延迟下持续胜出的模式。
   - 中心化架构因额外通信开销成为性能瓶颈。

4. **良好的泛化能力**：
   - 即使只在 **5节点 mini5 网络上训练**，LOGGIA 也能泛化到 **100节点 nx-L 网络**，且性能接近在大网络上直接训练的结果。

5. **硬件影响显著**：
   - 更快的 CPU 可减少推理时间，在延迟敏感场景下带来可测量的性能提升（Table 5）。

### 方法的局限性
- **单路径路由**：目前仅支持基于单一成本的最短路径，不支持 ECMP 或显式多路径。
- **未建模复杂协议逻辑**：如 BGP 的策略路由、TCP 拥塞控制的深层交互。
- **假设无限带宽信道**：状态/动作传输假设使用专用高带宽通道，未考虑带宽受限情况。
- **前向表更新延迟未建模**：未模拟实际交换机中路由表更新的延迟。

### 未来工作方向
- 扩展至**多路径路由**（Multipath Routing）。
- 支持更复杂的决策逻辑，如 BGP 风格的策略路由。
- 引入**压缩机制**以降低状态同步的通信开销。
- 探索**在线自适应压缩**或**稀疏更新**机制。
- 将形式化框架与 **Networked MDP**、**Real-Time MDP** 理论结合，建立更坚实的理论基础。

</details>

---

### 6. [Communication-Efficient Distributed Learning with Differential Privacy](https://arxiv.org/abs/2604.02558)

**Authors**: Xiaoxing Ren, Yuwen Ma, Nicola Bastianello, Karl H. Johansson, Thomas Parisini, Andreas A. Malikopoulos  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.02558v1  

#### Abstract
We address nonconvex learning problems over undirected networks. In particular, we focus on the challenge of designing an algorithm that is both communication-efficient and that guarantees the privacy of the agents' data. The first goal is achieved through a local training approach, which reduces co...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Communication-Efficient Distributed Learning with Differential Privacy*

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**非凸分布式学习**（nonconvex distributed learning）中的两个核心挑战：
- **通信效率低**（communication inefficiency）：频繁通信导致带宽消耗大、延迟高；
- **隐私泄露风险**（privacy leakage）：共享模型参数可能被用于重构原始训练数据。

现有方法通常在效率与隐私之间存在权衡，而本文旨在设计一种**同时具备高通信效率和强差分隐私保障**（differential privacy, DP）的分布式算法。

---

### 提出的新方法：LT-ADMM-DP
作者提出了一种名为 **LT-ADMM-DP**（Local Training ADMM with Differential Privacy）的新型分布式优化算法，其核心思想结合了以下技术：

- **Local Training**：每个 agent 在多个本地迭代（T 步）后才进行一次通信，显著降低通信频率；
- **Stochastic Gradients**：使用 mini-batch 随机梯度提升计算效率；
- **Gradient Clipping + Additive Noise**：在本地训练过程中对梯度进行裁剪并添加高斯噪声，以实现 **Rényi Differential Privacy (RDP)**；
- **基于 ADMM 的框架**：利用 Alternating Direction Method of Multipliers（ADMM）结构支持去中心化网络拓扑。

该方法实现了：
- 通信效率提升（减少通信轮次）
- 局部数据隐私保护（满足 $(\epsilon,\delta)$-DP）

---

### 相比现有方法的优势
| 维度 | LT-ADMM-DP | 现有方法（如 PORTER [21], PriSMA [22]） |
|------|------------|----------------------------------------|
| **通信效率** | 更高（仅需 $T \cdot t_g + t_c$ 时间成本） | 较低（每步都需通信或压缩） |
| **隐私机制** | 同时使用 clipping 和 noise，理论可证 RDP → DP 转换 | 多数仅考虑 centralized setting 或 weaker privacy |
| **收敛性分析** | 提供非凸问题下收敛至 stationary point 的误差界 |
| **性能表现** | 在相同隐私预算下，收敛更快、准确率更高 |

> ✅ 创新点总结：首次将 Local Training 与 ADMM 框架结合，并引入严格 DP 机制，在保证隐私的同时大幅提升通信效率。

---

## 2. 核心实验方法和设置

### 数据集与任务
- **任务类型**：分类任务（classification）
- **局部损失函数形式**：
  $$
  f_i(x) = \frac{1}{m_i} \sum_{h=1}^{m_i} \left[\log(1+\exp(-b_{i,h} a_{i,h}^\top x)) + \frac{\lambda}{2}\|x\|^2\right]
  $$
  即带有 $\ell_2$ 正则化的逻辑回归（nonconvex regularized loss）。
- **模拟数据生成**：特征向量 $a_{i,h} \in \mathbb{R}^n$ 和标签 $b_{i,h} \in \{-1,1\}$ 随机生成。

---

### 实验设置
| 参数 | 设置值 |
|------|-------|
| 网络拓扑 | Ring network（环形网络），$N=10$ 个 agents |
| 模型维度 | $n=5$ |
| 每个 agent 的样本数 | $m_i = 1000$ |
| Mini-batch size | $|B|=8$ |
| 总迭代次数 | $K=4000$ |
| 本地训练步数 | $T=4$ |

#### 隐私设置
- 设定 $\delta_i = 10^{-4}$，目标达到统一的隐私预算 $\epsilon \approx 19.6$
- 对比方法通过调节噪声标准差 $\sigma$ 匹配相同 $\epsilon$

#### 基线方法
1. **PORTER [21]**：基于梯度裁剪和通信压缩的去中心化 DP 方法
2. **PriSMA [22]**：适用于异构数据的分布式 DP 学习算法

#### 评估指标
- **Optimization Error**: $\|\nabla F(x_k)\|$（全局梯度范数）
- **Classification Accuracy**：测试准确率
- **时间复杂度建模**：
  - 局部梯度计算耗时：$t_g = 0.1$
  - 一轮通信耗时：$t_c = 1$
- 所有结果按实际时间缩放横轴（x-axis scaled by time complexity）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Fig. 1）

| 指标 | LT-ADMM-DP | PORTER | PriSMA |
|------|-----------|--------|--------|
| 收敛速度（error下降） | 最快，在 ~15000 时间单位内接近稳定 | 较慢 | 最慢 |
| 最终分类准确率 | ≈ **0.88–0.90** | ≈ 0.75 | ≈ 0.80 |
| 最终 $\|\nabla F(x_k)\|$ | 下降至约 $10^{-3}$ 量级 | 停留在 $10^{-2}$ 左右 | 类似 PORTER |

> 📊 图表说明：
> - Fig. 1(a) 显示 LT-ADMM-DP 在更短时间内达到更低的梯度误差；
> - Fig. 1(b) 显示其分类准确率明显优于其他两种方法。

---

### 时间成本对比（Table I）

| 方法 | 每 $T$ 次迭代的时间开销 |
|------|--------------------------|
| PORTER | $T(t_g + 2t_c)$ |
| PriSMA | $T(2t_g + t_c)$ |
| **LT-ADMM-DP** | $T t_g + t_c$ ✅（最小） |

👉 表明 LT-ADMM-DP 在通信上具有显著优势，尤其当 $t_c \gg t_g$ 时效果更明显。

---

### 消融实验（隐含分析）
虽然未明确列出消融实验表格，但从理论分析中可推断关键组件的影响：

| 组件 | 影响 |
|------|------|
| **Local Training ($T>1$)** | 提升通信效率，但增加稳态误差风险；需平衡 $T$ 与收敛性 |
| **Gradient Clipping ($C$)** | 控制 sensitivity，直接影响 DP bound 中的 $\Delta_2.f$ |
| **Noise Variance ($\sigma^2$)** | 增大噪声提升隐私，但损害模型最优性（trade-off） |
| **Step size $\gamma, \beta$** | 受网络连通性（algebraic connectivity $\lambda_2$）影响，连接越弱，步长需越小 |

> Remark 1 指出：更大的 $T$ 或 $\gamma$ 加速收敛，但也增大 steady-state error —— 揭示了效率与精度之间的权衡。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **LT-ADMM-DP 成功实现了通信效率与差分隐私的双重目标**：
   - 通过 local training 减少通信频率；
   - 通过 clipped noisy gradients 实现严格的 $(\epsilon,\delta)$-DP 保证。
   
2. ✅ **理论收敛性成立**：
   - 在非凸设定下，算法收敛到 stationary point 的一个有界邻域；
   - 误差上界依赖于：梯度方差 $\sigma_g^2$、噪声方差 $\sigma_e^2$、数据异质性 $\sigma_f$、网络拓扑 $\lambda_2$。

3. ✅ **隐私保障可量化**：
   - 利用 RDP 分析工具，导出了紧致的 privacy budget 上界（Theorem 2）；
   - 支持跨多轮迭代的 compositional privacy accounting。

4. ✅ **实验证明优越性**：
   - 在相同隐私预算下，LT-ADMM-DP 收敛更快、最终准确率更高；
   - 时间成本最低，适合资源受限场景。

---

### 方法的局限性
- **假设较强**：
  - 要求网络为 connected undirected graph；
  - 假设 loss functions 是 L-smooth（限制某些病态模型）；
  - 假设随机梯度无偏且方差有界（现实数据可能存在 heavy-tailed noise）。
- **固定 clipping threshold**：未采用 adaptive clipping（如 [19]），可能导致次优隐私-效用权衡。
- **同质化数据假设较弱**：尽管考虑了 gradient variation bound（Assumption 3），但在极端异构数据下性能可能下降。

---

### 未来工作方向（作者指出）
1. **Adaptive Clipping Strategies**：动态调整 clipping threshold 以优化隐私预算分配；
2. **Handling Data Heterogeneity**：进一步研究在高度 non-IID 数据下的鲁棒性和收敛性；
3. **扩展至 Federated Learning Setting**：应用于 client-server 架构中的联邦学习系统；
4. **异步版本设计**：支持 asynchronous updates 以适应真实网络延迟。

---

## 总结

| 方面 | 内容 |
|------|------|
| **核心贡献** | 提出 LT-ADMM-DP，融合 local training 与 DP 机制，解决通信效率与隐私保护的矛盾 |
| **关键技术** | Stochastic gradients + Gradient clipping + Gaussian noise + ADMM 框架 |
| **理论成果** | 证明收敛至 stationary point 的误差界，并提供 RDP → DP 的严格转换 |
| **实验结果** | 在相同 $\epsilon$ 下，相比 PORTER 和 PriSMA，收敛更快、准确率更高、时间成本更低 |
| **意义** | 推动了 privacy-preserving distributed learning 向高效、实用方向发展 |

> 🔚 **一句话总结**：  
> LT-ADMM-DP 是首个在去中心化非凸学习中同时实现**高通信效率**与**严格差分隐私保障**的算法，实验证明其在性能和隐私之间取得了优异平衡。

</details>

---

### 7. [Chart-RL: Policy Optimization Reinforcement Learning for Enhanced Visual Reasoning in Chart Question Answering with Vision Language Models](https://arxiv.org/abs/2604.03157)

**Authors**: Yunfei Bai, Amit Dhanda, Shekhar Jain  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.03157v1  

#### Abstract
The recent advancements in Vision Language Models (VLMs) have demonstrated progress toward true intelligence requiring robust reasoning capabilities. Beyond pattern recognition, linguistic reasoning must integrate with visual comprehension, particularly for Chart Question Answering (CQA) tasks invol...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Chart-RL: Policy Optimization Reinforcement Learning for Enhanced Visual Reasoning in Chart Question Answering with Vision Language Models》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Vision Language Models (VLMs)** 在处理 **Chart Question Answering (CQA)** 任务时面临以下挑战：
- **数值提取不精确**：难以准确读取图表中的数字，尤其是重叠或堆叠元素。
- **隐含视觉关系理解困难**：如趋势、比例等非显式信息。
- **注意力机制不足**：无法有效捕捉图表的空间结构和层次关系。
- **多步推理能力弱**：在需要跨区域关联、计算和逻辑推导的任务中表现不佳。

### 🚀 提出的新方法：Chart-RL
作者提出 **Chart-RL**，一种基于 **Reinforcement Learning (RL)** 的策略优化框架，用于增强 VLMs 的图表理解和视觉推理能力。

#### 核心创新点：
- **端到端的反馈驱动学习**：通过 RL 构建反馈循环，直接优化模型的视觉感知与逻辑推理过程。
- **集成多种先进策略优化技术**：
  - **GRPO (Group-based Reinforcement Learning from Policy Optimization)**
  - **DAPO (Direct Advantage Policy Optimization)**
  - **GSPO (Group Sequence Policy Optimization)**
- **结合 PEFT/LoRA 实现高效训练**：
  - 采用 **Low-Rank Adaptation (LoRA)** 进行参数高效微调（Parameter-Efficient Fine-Tuning），仅需单张 GPU 即可完成训练，显著降低资源消耗。
- **无需监督微调（SFT）预阶段**：直接对基础模型进行 RL 微调，简化流程并避免依赖高质量标注数据。

### 🔍 相比现有方法的优势
| 维度 | Chart-RL 的优势 |
|------|----------------|
| **性能** | 显著提升 CQA 准确率，超越更大规模的基础模型 |
| **效率** | 推理延迟从 >30 秒降至 <10 秒，训练可在单卡上完成 |
| **通用性** | 可适配多种 VLM 架构，具备良好泛化能力 |
| **可部署性** | 参数量小、硬件要求低，适合生产环境部署 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **主数据集**：**ChartQAPro**  
  - 包含 1,948 个真实世界图表样本，来源多样（如 Our World in Data, Statista 等）
  - 覆盖多种图表类型：bar chart, line chart, pie chart, scatter plot
  - 问题类型丰富：
    - Factoid (55%)：事实查询
    - Conversational (16%)：上下文对话
    - Fact-checking (13%)：验证判断
    - Multiple-choice (11%)：选择题
    - Hypothetical (5%)：假设推理
    - Unanswerable：不可回答问题（测试集中排除）

### ⚙️ 实验设置
- **模型架构**：
  - 主要基于 **Qwen3-VL-4B-Instruct** 和 **Qwen3-VL-8B-Instruct**
  - 使用 LoRA 进行 PEFT，rank=256, alpha=1024，仅更新 query/value 投影层
- **训练配置**：
  - 单 GPU（24GB 内存），bf16 精度
  - 学习率：1e-5，batch size: 2/device
  - 最大 prompt 长度：8192 tokens，completion: 1024 tokens
  - 每轮生成 2 个候选响应用于组内比较
- **图像预处理**：
  - 图像统一缩放到宽度为 300 像素，保持长宽比
  - 减少内存占用 70%，不影响精度

### 🎯 评估指标
- **Answer Accuracy (Acc)**：由 GPT-4 作为 judge 对比预测答案与真实答案的一致性
- **Inference Latency (s)**：单次推理耗时（秒）
- **Latency-Accuracy Trade-off**：综合评估效率与性能的帕累托前沿

### 🆚 基线方法对比
| 类别 | 模型名称 |
|------|---------|
| **闭源 SOTA MLLM** | Claude Sonnet 3.7, Claude Sonnet 4.5 |
| **开源 VLM** | Qwen2-VL, Qwen3-VL, Janus-Pro, InternVL, LLaVA |
| **本工作微调模型** | Qwen3-VL-4B-Instruct + GRPO / DAPO / GSPO |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 和正文）

| 模型 | Acc (0–1) | Latency (s) |
|------|-----------|------------|
| **Claude Sonnet 3.7** (SOTA) | **0.769** | – |
| Qwen3-VL-8B-Instruct (Base) | 0.580 | 31.59 |
| Qwen3-VL-4B-Instruct (Base) | 0.396 | 10.04 |
| → + GRPO | **0.627** | 9.84 |
| → + DAPO | **0.634** | 9.48 |
| → + GSPO | 0.622 | 9.69 |

> ✅ **关键发现**：
> - 尽管参数仅为一半（4B vs 8B），**RL 微调后的 Qwen3-VL-4B 模型全面超越其更大的基础版本**。
> - **DAPO 效果最佳**，达到 **0.634 准确率**，较原始 4B 模型提升 **+23.8%**，较 8B 基础模型也高出 **+54 个百分点**。
> - 推理延迟从 31.59s 降至约 **9.5s**，**减少 71%**，实现高精度与低延迟的双重优势。

### 🔬 消融实验与分析
- **训练动态监控**（Figure 2）显示：
  - 所有 RL 方法（GRPO/DAPO/GSPO）均呈现奖励（Reward）持续上升、损失下降、熵稳定减少的趋势，表明策略成功收敛。
  - 完成长度（Mean Length）在约 150 步后趋于稳定，说明快速学会生成简洁有效的推理链。
- **Chain-of-Thought (CoT) 分析** 表明：
  - RL 微调显著提升了中间推理步骤的质量，增强了多步逻辑连贯性和数学计算准确性。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **Chart-RL 显著提升 VLM 的图表推理能力**：
   - 在复杂 CQA 任务中，特别是涉及多步计算、趋势外推、统计相关性判断等问题上，RL 微调模型表现出更强的理解与推理能力。
2. **参数效率极高**：
   - 使用仅 **0.5% 可训练参数**（via LoRA），即可实现超越更大模型的性能，打破“越大越好”的范式。
3. **计算资源友好**：
   - 全程可在单张消费级 GPU 上完成训练与推理，极大降低了部署门槛。
4. **优于传统监督微调路径**：
   - 无需高质量人工标注的 CoT 数据，直接通过 RL + LLM Judge 自动构建反馈信号，更具扩展性。

### ⚠️ 局限性
- **依赖 LLM Judge 作为奖励函数**：
  - 虽然 GPT-4 判断能力强，但仍可能引入噪声，尤其在模糊语义或数值近似场景下。
  - 存在“reward hacking”风险，即模型可能学会欺骗奖励模型而非真正理解图表。
- **泛化能力仍受限于训练分布**：
  - 当前实验集中在 ChartQAPro 数据集，对极端新颖图表结构的迁移能力有待验证。
- **未探索更复杂的多图联合推理**：
  - 多图表对比、跨图推理等高级任务尚未充分覆盖。

### 🔮 未来工作方向
- **多阶段奖励精炼（Multi-stage Reward Refinement）**：
  - 先用 LLM Judge 进行初步训练，再引入人类反馈或集成多个 reward model 提升稳定性。
- **引入环境交互机制**：
  - 结合工具调用（如 Python 执行器）进行数值验证，形成闭环反馈。
- **扩展至其他视觉推理任务**：
  - 如科学图表解释、医学图像问答、地理空间数据分析等。
- **探索 MoE 架构下的 RL 应用**：
  - 利用 GSPO 在 Mixture-of-Experts 模型中的优势，进一步提升长文本输出稳定性。

---

## 总结一句话
> **Chart-RL 通过将强化学习与参数高效微调相结合，在极低资源消耗下实现了对 VLM 图表推理能力的显著增强，推动了智能系统向“真正理解可视化信息”迈出关键一步。**

</details>

---

### 8. [InfoSeeker: A Scalable Hierarchical Parallel Agent Framework for Web Information Seeking](https://arxiv.org/abs/2604.02971)

**Authors**: Ka Yiu Lee, Yuxuan Huang, Zhiyuan He, Huichi Zhou, Weilin Luo, Kun Shao, Meng Fang, Jun Wang  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02971v1  

#### Abstract
Recent agentic search systems have made substantial progress by emphasising deep, multi-step reasoning. However, this focus often overlooks the challenges of wide-scale information synthesis, where agents must aggregate large volumes of heterogeneous evidence across many sources. As a result, most e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# InfoSeeker: A Scalable Hierarchical Parallel Agent Framework for Web Information Seeking 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Large Language Model (LLM) agent** 系统在处理大规模信息合成任务时面临三大挑战：
- **Context saturation**：传统框架上下文窗口有限，难以容纳从数十个网页聚合的信息。
- **Cascading error propagation**：顺序执行模式下，早期错误会随推理链累积放大。
- **High end-to-end latency**：串行架构导致任务延迟严重，影响实用性。

这些问题在需要“宽域”信息整合的任务中尤为突出，例如填充完整表格、跨源验证实体属性等。

### 提出的新方法
提出 **InfoSeeker** —— 一个基于 **near-decomposability** 原理的分层并行 Agent 框架，包含三个层级：
- **Host Agent**：战略层，负责高层规划与全局状态维护。
- **Manager Agents**：领域管理层（如 Search Manager、Browser Manager），分解任务、协调执行、进行质量验证与结果聚合。
- **Worker Agents**：工具执行层，通过 **Model Context Protocol (MCP)** 并行调用工具（如搜索、浏览）。

该设计实现了：
- **严格上下文隔离**：各层仅传递摘要信息，防止上下文膨胀。
- **MapReduce 式执行范式**：Manager 负责“Map”（分解）、Worker 并行执行、“Reduce”（聚合）。
- **动态并行化与容错机制**：支持跨 Manager 和 Worker 层的大规模并行，并具备自动重试与浏览器接管能力。

### 相比现有方法的优势
| 维度 | InfoSeeker | 传统方法（如 ReAct、Gemini DeepResearch） |
|------|------------|----------------------------------------|
| 架构 | 分层、模块化、近可分解 | 单体或浅层多代理 |
| 上下文管理 | 隔离上下文，仅传播摘要 | 共享长上下文，易饱和 |
| 执行方式 | 大规模并行 | 主要串行或有限并行 |
| 错误控制 | 支持反思（reflection）、验证与重试 | 错误易传播，缺乏修正机制 |
| 效率 | 3–5× 速度提升 | 高延迟 |

---

## 2. 核心实验方法和设置

### 数据集
- **WideSearch** [Wong et al., 2025]  
  - 英文/中文混合，要求从大量异构网页中提取结构化信息（如填表）。
  - 强调**完整性约束**（exhaustive entity discovery）和**属性验证**。
  - 包含数百个需多跳检索与跨页推理的任务。

- **BrowseComp-zh** [Zhou et al., 2025]  
  - 中文网络环境下的复杂浏览任务基准。
  - 包含 289 个专家设计的多跳问题，覆盖 11 个领域。
  - 强调对非英文 DOM 结构的理解与交互导航能力。

### 实验设置
- **模型配置**：
  - Host & Manager 使用 `gpt-5.1` 进行高保真规划。
  - Worker 使用轻量级 `gpt-5-mini` 实现高效执行。
- **工具集成**：
  - 搜索：Firecrawl via MCP
  - 浏览：Playwright 容器化实例（支持中文字体渲染）
  - 文件系统与代码执行：沙箱环境
- **评估协议**：遵循各基准官方设定，确保公平比较。

### 评估指标
| 基准 | 指标说明 |
|------|----------|
| **WideSearch** | - **Success Rate**：完全正确匹配（精确匹配）<br>- **Row F1**：行级别召回与准确率（实体完整性）<br>- **Item F1**：细粒度属性正确性<br>- Avg@4 / Max@4：多次运行平均与最佳表现 |
| **BrowseComp-zh** | - **Accuracy**：最终答案正确的比例 |

### 基线方法对比
涵盖三类主流系统：
1. **单智能体模型**（Single Agent）：Claude Sonnet 4、Gemini 2.5 Pro、OpenAI o3-high 等。
2. **端到端商业系统**（End-to-End Systems）：Gemini Deep Research、OpenAI Deep Research。
3. **多智能体框架**（Multi-Agent Framework）：GPT-Researcher 类似架构。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### WideSearch-en 结果（Table 1）
| 模型/系统 | Success Rate (Avg@4) | Item F1 (Avg@4) |
|---------|-----------------------|----------------|
| OpenAI o3-high (Multi-Agent) | 5.10% | 57.30% |
| **InfoSeeker (Ours)** | **8.38%** (+64%) | **70.27%** |

> 在 Max@4 设置下，InfoSeeker 成功率达 **9.50%**，Item F1 达 **75.11%**。

#### BrowseComp-zh 结果（Table 2）
| 模型/系统 | Accuracy |
|----------|----------|
| OpenAI DeepResearch | 42.9% |
| BrowseMaster (SOTA 开源) | 46.5% |
| **InfoSeeker (Ours)** | **52.9%** |

> 超越最强商业系统 **10 个百分点**，展现卓越的中文网页理解能力。

### 与基线方法的对比结果
- 在 WideSearch 上：
  - 相比最强基线（OpenAI o3-high），**Success Rate 提升 64%**。
  - Row F1 提升 **30%**（38.5 → 50.13），表明更强的结构一致性。
- 在 BrowseComp-zh 上：
  - 准确率领先第二名 **6.4%**，证明其在真实中文生态中的泛化优势。
- 推理效率方面（Figure 3）：
  - 相比 Gemini Deep Research 和 OpenAI Deep Research，实现 **2.6–4.6× 的推理加速**。
  - 得益于 Worker 层的并行执行与上下文隔离。

### 消融实验结果
#### Ablation on Worker Pool Size（Figure 4）
- 当 Worker 数量从 1 增加至 17 时：
  - 端到端延迟从 **911 秒降至 162 秒**，获得 **~5.7× 加速**。
- 表明弱耦合子任务可通过并行显著提升吞吐量。

#### 单智能体对照实验（Table 5）
| 系统 | Success Rate | Item F1 |
|------|-------------|---------|
| GPT-5.1 单智能体 | 6.00% | 35.74% |
| InfoSeeker（相同工具+模型） | **12.50%** | **75.21%** |

> 即使使用相同的 backbone 模型和工具访问权限，InfoSeeker 仍取得 **翻倍以上的性能提升**，说明性能增益主要来自**架构设计本身**。

---

## 4. 关键结论和发现

### 主要发现
1. **深度推理 ≠ 宽度合成能力**  
   当前 LLM agent 社区过度关注“深推理”（deep reasoning），但现实场景更常面临“宽合成”（wide synthesis）挑战。单纯扩大上下文或模型规模无法根本解决 context saturation 与 error propagation。

2. **分层并行架构是应对宽域任务的关键**  
   InfoSeeker 通过 **Host-Managers-Workers** 三层结构，实现了：
   - 功能解耦（functional decoupling）
   - 上下文隔离（context isolation）
   - 并行扩展（parallel scalability）

3. **MapReduce 范式适用于 agentic workflow**  
   将任务分解为可并行子任务（Map）、由 Workers 执行、再聚合为摘要（Reduce），有效分离了“推理深度”与“执行宽度”。

4. **协作式 Manager 设计增强鲁棒性**  
   如 Search Manager 遇到反爬或 CAPTCHA 时，可主动推荐切换至 Browser Manager（见 Figure 6），体现系统的自适应能力。

### 方法的局限性
1. **依赖外部 API 与工具可用性**  
   性能受限于 MCP 工具的稳定性、速率限制与并发能力。

2. **提示工程依赖性强**  
   当前 Manager 与 Host 的行为高度依赖手工设计的 prompt，通用性受 backbone model 影响。

3. **极端数据量仍可能超限**  
   如 Figure 8 所示，在处理“所有 AMD Zen CPU”这类超大规模 SKU 列表时，仍因 token 超出限制（300k）而被迫返回样本表。

4. **实体链接能力有待加强**  
   如 Figure 7 显示，面对模糊术语（如“变异型”），系统可能误解为类别而非具体疾病名，导致返回 plausible but incorrect 答案。

### 未来工作方向
1. **自动化任务分解与协调策略学习**  
   探索使用 **multi-agent reinforcement learning** 学习最优分解与调度策略，减少对 in-context learning 的依赖。

2. **训练小型专用模型以降低成本**  
   当前依赖 GPT-5 级别模型，未来计划训练更小、更高效的专用模型用于 Manager 与 Worker 角色。

3. **增强实体约束与类型感知能力**  
   引入更强的 schema-aware reasoning 机制，避免 answer-type mismatch。

4. **提升长周期任务的记忆与状态管理能力**  
   当前 Host 仅依赖压缩上下文，未来可探索外挂记忆模块以支持更复杂的长期规划。

---

> ✅ **代码已开源**：[https://github.com/agent-on-the-fly/InfoSeeker](https://github.com/agent-on-the-fly/InfoSeeker)

</details>

---

### 9. [Revealing the Learning Dynamics of Long-Context Continual Pre-training](https://arxiv.org/abs/2604.02650)

**Authors**: Yupu Liang, Shuang Chen, Guanwei Zhang, Shaolei Wang, Suncong Zheng  
**Category**: cs.CL  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02650v1  

#### Abstract
Existing studies on Long-Context Continual Pre-training (LCCP) mainly focus on small-scale models and limited data regimes (tens of billions of tokens). We argue that directly migrating these small-scale settings to industrial-grade models risks insufficient adaptation and premature training termina...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Revealing the Learning Dynamics of Long-Context Continual Pre-training》论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前关于 **Long-Context Continual Pre-training (LCCP)** 的研究大多基于小规模模型（如7B参数）和有限数据（数十亿tokens），其结论难以推广到工业级大模型。此外，主流评估方式（如 **Needle-in-a-Haystack, NIAH**）存在“**欺骗性饱和**”（deceptive saturation）现象——即指标提前饱和，无法真实反映模型内在能力的持续提升。

该论文系统性地揭示了在工业级 LLM 上进行 LCCP 的学习动态，填补了学术研究与工业实践之间的鸿沟。

---

### 🚀 提出的新方法与新思路

1. **三层次分析框架（Hierarchical Framework）**  
   首次从三个层面系统分析 LCCP 的学习过程：
   - **行为层（Behavioral Level）**：通过轻量级 Supervised Fine-Tuning (SFT) 探针评估下游任务表现。
   - **概率层（Probabilistic Level）**：提出 **连续型 NIAH（Continuous NIAH）**，将传统二值准确率替换为基于 **Perplexity (PPL)** 的细粒度生成置信度评估。
   - **机制层（Mechanistic Level）**：识别并追踪“**检索头（retrieval heads）**”的注意力演化，作为低资源训练监控工具。

2. **揭示“欺骗性饱和”现象**  
   发现传统 NIAH 准确率在训练早期就达到 100%，但实际模型仍在持续优化；而 PPL 能更真实地反映长期上下文建模的渐进改进。

3. **提出高效训练监测机制**  
   利用 retrieval head 的数量和平均得分作为内部指标，可替代耗时的 SFT 和下游评测，实现快速、低成本的训练进度监控。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法 | 本文方法 |
|------|----------|-----------|
| **评估粒度** | 粗粒度（是否答对） | 细粒度（生成概率变化） |
| **适用对象** | 小模型、小数据 | 工业级 MoE 模型（80B 参数） |
| **评估成本** | 需要 SFT + 下游 benchmark | 可直接用于 base model，无需微调 |
| **可靠性** | 易出现“假饱和” | 更强相关性，反映真实收敛状态 |

---

## 2. 核心实验方法和设置

### 🧪 使用的数据集

- **训练数据（LCCP阶段）**：
  - 总量：**200B tokens**
  - 构成：25% 短文本（≤32K）+ 75% 长文本（>32K）
  - 来源分布：
    - Common Crawl: 36.3%
    - Books: 28.6%
    - arXiv: 24.0%
    - Code: 10.8%
    - Wikipedia: 0.3%

- **评估数据集**：
  - **RULER**（选取 QA 子集）
  - **MRCR**
  - **LongBio**（paraphrase, pronoun, standard 子集）
  - 自定义 **NIAH 测试样本**（用于 PPL 分析）
  - 内部中英文语料库及代码仓库（用于常规 PPL 测评）

---

### ⚙️ 实验设置

| 项目 | 设置 |
|------|------|
| 模型 | **Hunyuan-A13B**（Sparse MoE，总参数 80B，激活 13B/token） |
| 初始上下文长度 | 32K → 扩展至 64K |
| RoPE base frequency | 从 500K 提升至 2M |
| 学习率 | 恒定 $1.2 \times 10^{-5}$ |
| Batch size | 全局 16M tokens |
| 训练轨迹 | 跨越 200B tokens 的多个 checkpoint 追踪 |

---

### 📊 评估指标

| 层级 | 指标 |
|------|------|
| 行为层 | SFT 后在 RULER / MRCR / LongBio 上的 Pass@3 得分 |
| 概率层 | NIAH 任务中的 **answer token PPL**（连续型 NIAH） |
| 机制层 | **retrieval head 数量** 和 **平均 retrieval score** |
| 相关性验证 | Pearson correlation 与 downstream SFT performance 的关系 |

---

### 🔁 基线方法对比

- **传统 NIAH 准确率** vs. **本文提出的 NIAH PPL**
- 不同训练阶段的 SFT 性能趋势 vs. retrieval head 演化趋势
- 小规模研究中观察到的“几十亿token即饱和” vs. 本研究中“需超150B tokens才稳定”

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）行为层：SFT 探针结果（图1）
- RULER 得分从初始 68.68 提升至 100B token 时的 **79.44**
- MRCR 和 LongBio 在 50B token 后边际收益下降，但仍持续提升
- **完全稳定需超过 150B tokens**

#### （2）概率层：NIAH PPL vs Accuracy（图3）
- **原始 NIAH 准确率**：在 20B token 达到 100%，之后无变化（**欺骗性饱和**）
- **NIAH PPL**：持续下降，直到 **150B token 才趋于平稳**
- **Pearson 相关性（vs SFT performance）**：
  | 指标 | 平均相关性 |
  |------|------------|
  | NIAH Score | 0.7486 |
  | **NIAH PPL** | **-0.8210**（负相关，PPL越低越好） |

> ➤ **NIAH PPL 与下游性能的相关性显著更高**

#### （3）机制层：retrieval head 演化（图4–5）
- retrieval head 数量随训练 token 增加而上升
- 平均 retrieval score 持续提高
- **与下游性能高度正相关**：
  | 指标 | 平均 Pearson 相关性 |
  |------|---------------------|
  | # of retrieval heads | 0.7428 |
  | Avg. retrieval score | **0.7878** |

> ➤ retrieval head 指标可作为高效的训练进展代理指标

#### （4）PPL 缩放规律（图6）
- PPL 随 log(N) 呈近似线性下降：
  $$
  \text{PPL} = A \cdot \log(N) + B
  $$
- 符合神经网络 scaling law，表明增益逐渐衰减

---

### 🔍 消融实验与进一步发现

#### （1）缓解 “Lost in the Middle” 现象（图7）
- 初始模型在中间位置（depth 30%-80%）PPL 明显偏高
- 经过充分 LCCP（150B+ tokens）后，中部信息检索能力显著增强，PPL 下降明显

#### （2）抗干扰能力提升（图8–9）
- 干扰上下文增长时，未训练模型 PPL 快速上升
- 经 LCCP 训练后，模型对长干扰具有更强鲁棒性，PPL 保持低位

#### （3）retrieval head 的稳定性（表5）
- Top 30 retrieval heads 在不同训练阶段重叠率 >93%
- Spearman 秩相关系数 >0.88
> ➤ retrieval 功能主体在预训练阶段已确定，LCCP 主要是**放大而非重构**

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **工业级 LLM 的 LCCP 需要海量数据**  
   - 小规模研究中“几十亿tokens即饱和”的结论不适用于工业模型
   - **Hunyuan-A13B 至少需要 150B+ tokens 才能达到真正稳定**

2. **传统 NIAH 存在“欺骗性饱和”**  
   - 准确率早熟饱和，不能反映真实学习进程
   - **PPL 是更可靠的内在收敛指标**

3. **retrieval heads 是有效的训练监控信号**  
   - 其演化与下游性能强相关
   - 可用于**免SFT、低开销的实时训练诊断**

4. **LCCP 显著提升长上下文鲁棒性和定位能力**  
   - 缓解“lost in the middle”
   - 增强对抗长距离干扰的能力

5. **功能专业化发生在预训练阶段**  
   - retrieval heads 的身份基本固定于初期
   - LCCP 更像是“调音”而非“重建”

---

### ⚠️ 方法的局限性

1. **仅针对单一模型架构（Hunyuan-A13B）**
   - 结论在其他 MoE 或 Dense 架构上的泛化性待验证

2. **计算资源门槛极高**
   - 200B token LCCP 成本约 $4 \times 10^{23}$ FLOPs，难以复现

3. **缺乏完整 Post-training pipeline 验证**
   - 未涵盖 RLHF 或复杂推理任务的影响

4. **retrieval head 定义依赖特定任务格式**
   - 当前 retrieval score 基于 NIAH 格式设计，通用性有待扩展

---

### 🔮 未来工作方向（来自 Limitation 节）

1. **Ultra-Long Context Scaling**  
   - 扩展至 256K+ 上下文窗口，验证饱和阈值是否迁移

2. **Comprehensive Alignment Pipeline**  
   - 加入完整的 SFT 与 RLHF，研究 LCCP 对指令遵循和复杂推理的影响

3. **Systematic Ablation Studies**  
   - 控制变量分析数据配比、RoPE 频率等超参影响

4. **Mechanistic Intervention**  
   - 主动干预 retrieval heads 是否可加速训练或提升事实性

5. **Cross-Model Generalization**  
   - 在 DeepSeek、Qwen 等开源模型上验证本框架普适性

---

> 💡 **总结一句话**：  
> 本论文首次系统揭示了工业级 LLM 在长上下文持续预训练中的真实学习动态，提出了以 **PPL 和 retrieval head** 为核心的多层级监控体系，打破了“小模型经验可外推”的迷思，并为高效、可预测的 LCCP 提供了理论与工程双重指导。

</details>

---

### 10. [Digital Twin-Assisted In-Network and Edge Collaboration for Joint User Association, Task Offloading, and Resource Allocation in the Metaverse](https://arxiv.org/abs/2604.02938)

**Authors**: Ibrahim Aliyu, Seungmin Oh, Sangwon Oh, Jinsul Kim  
**Category**: cs.DC  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02938v1  

#### Abstract
Advancements in extended reality (XR) are driving the development of the metaverse, which demands efficient real-time transformation of 2D scenes into 3D objects, a computation-intensive process that necessitates task offloading because of complex perception, visual, and audio processing. This chall...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Digital Twin-Assisted In-Network and Edge Collaboration for Joint User Association, Task Offloading, and Resource Allocation in the Metaverse

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**元宇宙（Metaverse）中扩展现实（XR）应用**面临的以下挑战提出解决方案：
- **计算密集型任务处理需求**：XR 应用需要将 2D 场景实时转换为 3D 对象，涉及复杂的感知、视觉和音频处理，本地设备难以承担。
- **上行链路（UL）与下行链路（DL）数据不对称**：UL 传输原始 2D 数据，而 DL 需要返回渲染后的高体积 3D 内容，导致资源分配不均和延迟增加。
- **传统 MEC 架构的局限性**：多接入边缘计算（MEC）存在资源争用、信令开销大、可扩展性差等问题。
- **动态环境下的分布式决策难题**：用户任务到达具有随机性，无线信道易受干扰，且需联合优化用户关联、任务卸载模式、部分卸载比例及 DL 功率分配。

### 提出的新方法与思路
作者提出了一种**基于数字孪生（Digital Twin, DT）的协同框架——DT-assisted INC-E**，其核心创新包括：

#### （1）**DT-assisted INC-E 架构**
- 将 **In-Network Computing (INC)** 节点引入 MEC 系统，实现任务在中间网络节点的预处理（如特征提取），减轻 MEC 服务器负担。
- 利用 **Digital Twin 技术**构建物理系统的虚拟镜像，用于实时同步状态、预测性能并辅助决策，提升系统智能化水平。
- 支持 **URLLC（Ultra-Reliable Low-Latency Communication）**，确保低时延高可靠通信。

#### （2）**Stackelberg Markov Game 建模**
- 将运营商（operator）作为领导者（leader），XR 用户设备（XUDs）作为跟随者（followers），建模为分层博弈。
- 运营商决定 **Offloading Mode (OFMO)** 和 **Downlink Power Allocation (POAL)**；
- XUDs 自主选择信道和部分卸载比例以最大化自身效用。

#### （3）**异步多智能体强化学习算法（Nash-AMRL）**
- 设计 **Asynchronous Hybrid Multi-Agent Reinforcement Learning (AHMRL)** 框架：
  - **UL Agent** 输出每个用户的协作偏好分数，并通过 **0-1 Knapsack Solver** 转换为可行的二进制 OFMO 决策，满足 INC 容量约束。
  - **DL Agent** 输出连续的 DL 功率分配策略。
  - 引入 **Hybrid Critic** 架构（含 UL、DL、Global 三个价值头），实现细粒度奖励分解与跨阶段知识共享。

#### （4）**去中心化用户关联机制**
- XUDs 在运营商指导下进行干扰感知的信道选择，形成一个 **Exact Potential Game (EPG)**，最终收敛到 **Nash Equilibrium (NE)**，实现自组织用户-INC 节点关联。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| 卸载架构 | 单纯 MEC 或静态协作 | INC + MEC 协同，支持渐进式推理 |
| 决策机制 | 集中式或静态规则 | 分布式、自适应、基于博弈与 MARL |
| UL/DL 优化 | 通常独立处理 | 联合优化，考虑不对称流量特性 |
| 智能化程度 | 缺乏实时反馈与预测能力 | 借助 DT 实现状态镜像与预测 |
| 可扩展性 | 易受用户增长影响 | 去中心化设计支持大规模部署 |

---

## 2. 核心实验方法和设置

### 实验设置
- **仿真场景**：6 个 XUDs 分布在 200m×200m 区域内，配备 4 个 INC 节点和 1 个 MEC 服务器。
- **通信模型**：
  - 使用 **URLLC 模型**，考虑有限块长度（finite blocklength）下的传输速率。
  - 上行采用 **Match Filtering + Successive Interference Cancellation (MF-SIC)** 抑制干扰。
- **任务模型**：
  - 任务大小 $I_m \in [1,5]$ MB，计算负载 $C_m \in [1,5]$ Gcycles，延迟容忍 $T_{\text{max}} \in [5,15]$ ms。
  - 3D 渲染后数据放大倍数 $q \sim \mathcal{U}[1,10]$。
- **DT 模型**：引入处理速率估计偏差（discrepancy = 0.3），模拟真实与虚拟系统之间的差异。

### 评估指标（KPIs）
| 类别 | 指标 |
|------|------|
| **用户侧** | Utility, Uplink Rate, End-to-End Latency ($T^{e2e}$), Energy Consumption |
| **运营商侧** | UL Reward, DL Reward, Global Reward |
| **系统级** | Performance Gain (PG), Cost Gain (基于加权延迟-能耗成本) |
| **公平性与鲁棒性** | CDF of Cost Gain, Convergence Behavior |

### 基线方法对比
| 基线名称 | 描述 |
|--------|------|
| **GM-RN (Game-Random)** | OFMO 和 POAL 随机分配，作为下界 |
| **Equal Policy** | 固定 50% 协作卸载 + 均匀功率分配 |
| **Proportional Policy** | 卸载概率和功率按资源可用性和信道增益成比例分配 |
| **AAHC [19]** | 异步混合强化学习基准，但无 Knapsack 模块和 Hybrid Critic |
| **MASC / AC** | 消融变体：Multi-Actor Shared-Critic 与独立 Actor-Critic |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **系统效用（Utility）** | AHMRL 达到最高值，相比 MASC 提升约 4.5%，相比 AC 提升约 7% |
| **上行速率（Uplink Rate）** | AHMRL 平均提升 3.22% @ K=1，最大可达 0.99 AUC |
| **端到端延迟（$T^{e2e}$）** | AHMRL 在多数训练阶段表现最优，相比 MASC 最多降低 3.73% |
| **能量效率** | AHMRL 在 K=2 时节能 +0.66%，但在高密度场景略高于 MASC |
| **全局奖励（Global Reward）** | AHMRL 在小规模 INC 配置下（K=2）优于 MASC 和 AC 超过 33% |

### 与基线方法的对比结果
- **相比启发式策略（GM-RN, EQ-PLC, PROP-PLC）**：
  - 所有 MARL 方法显著优于启发式策略。
  - **超过 90% 的用户在 AHMRL 下获得 ≥1.5 的成本增益**，而启发式方法仅不到 50%。
- **相比学习型基线（MASC, AC）**：
  - **AHMRL 在通信密集型任务中表现最佳**，尤其在 UL Rate 和 Global Utility 上领先。
  - **MASC 在计算密集型、高用户密度场景更具竞争力**，因其共享 critic 更稳定。
  - **AC 表现波动较大**，缺乏全局协调导致收敛不稳定。

### 消融实验结果
| 变体 | 性能表现 |
|------|----------|
| **AHMRL（完整版）** | 综合性能最优，平衡延迟、能效与效用 |
| **MASC（单共享 critic）** | 学习更慢但后期收敛更稳，在高负载下反超 AHMRL |
| **AC（独立 critic）** | UL Rate 较高（达 0.96 AUC），但奖励波动大，整体效用最低 |
| **移除 Knapsack 模块** | OFMO 不满足容量约束，系统不稳定 |
| **固定 OFMO 或 POAL** | 效用下降 >15%，验证联合优化必要性 |

> ✅ **Table II 显示**：在不同任务负载和异构性下，AHMRL 在平均 AUC 指标上全面领先（UL AUC: 0.74, Util AUC: 0.72）。

---

## 4. 关键结论和发现

### 主要发现
1. **DT-assisted INC-E 架构有效缓解了 MEC 的瓶颈**，通过在网络层预处理任务，显著降低了 MEC 负载和端到端延迟。
2. **Nash-AMRL 框架能够实现高效的联合优化**：
   - 运营商通过异步 MARL 学习 OFMO 与 POAL；
   - XUDs 通过潜在博弈实现去中心化、抗干扰的用户关联。
3. **Hybrid Critic 设计提升了学习稳定性与性能**：
   - 多价值头结构有助于分离 UL/DL 信号，避免梯度冲突。
   - AHMRL 在大多数场景下优于 MASC 和 AC。
4. **系统具备良好的可扩展性与鲁棒性**：
   - 随着用户数量增加，AHMRL 的延迟增长最平缓。
   - 在不同任务分布下均能保持高性能，适用于动态元宇宙环境。

### 方法的局限性
1. **依赖 DT 模型精度**：若 DT 与物理系统偏差过大，可能导致决策失准。
2. **Knapsack 求解复杂度为 $O(M \cdot C_{\text{max}})$**：当用户数或 INC 容量极大时可能影响实时性。
3. **未考虑移动性建模**：当前假设用户在时隙内静止，未来需扩展至移动场景。
4. **离线训练开销较高**：虽然在线推理轻量，但训练过程仍需大量样本。

### 未来工作方向
1. **引入联邦学习或迁移学习**，减少对全局状态的依赖，增强隐私保护。
2. **结合移动性预测**，支持高速移动用户的服务连续性保障。
3. **探索更多 INC 层次结构**，如多跳 in-network processing。
4. **硬件原型验证**：在真实测试平台上部署 DT-incorporated INC-E 系统。
5. **纳入碳排放等绿色指标**，构建可持续的 Metaverse 资源管理框架。

--- 

> 📌 **总结一句话**：  
> 本文提出了一种基于 **Digital Twin** 和 **In-Network Computing** 的新型元宇宙资源协同框架，通过 **Stackelberg Game + Nash-AMRL** 实现了用户自主卸载与运营商智能调度的高效协同，在延迟、能效、公平性等方面全面超越现有方法，为下一代沉浸式服务提供了可扩展的技术路径。

</details>

---

### 11. [FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving](https://arxiv.org/abs/2604.02715)

**Authors**: Qingxiu Liu, Cyril Y. He, Hanser Jiang, Zion Wang, Alan Zhao, Patrick P. C. Lee  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02715v1  

#### Abstract
Mixture-of-Experts (MoE) models have become a dominant paradigm for scaling large language models, but their rapidly growing parameter sizes introduce a fundamental inefficiency during inference: most expert weights remain idle in GPU memory while competing with performance-critical runtime state su...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代 **Mixture-of-Experts (MoE)** 大语言模型虽然能高效扩展参数规模，但在推理过程中存在严重的**内存效率低下**问题：
- 所有专家权重（expert weights）在推理期间始终驻留在 GPU 内存中，即使大部分专家处于闲置状态。
- 这些闲置权重与对性能至关重要的运行时状态（如 **KV Cache**）竞争有限的 GPU 显存资源。
- 由于 **KV Cache 容量直接决定服务吞吐量（throughput）**，这种资源争用导致显存利用率低、推理性能下降。

### **提出的新方法与思路**
论文提出了 **FluxMoE**，一种全新的 MoE 推理系统，其核心思想是 **“专家分页”（expert paging）**，即：
- 将专家参数视为**流式、临时资源**，而非持久驻留的静态数据。
- 在执行某一层时，才将该层所需的专家权重从外部存储加载到 GPU 显存；执行完毕后立即释放。
- 由此实现公式化的执行模型：  
  `model = compute graph + streamed parameters`

### **三大关键技术机制**
1. **PagedTensor**  
   - 类似于操作系统虚拟内存管理，为每个专家张量提供稳定的虚拟地址（tensor handle），动态绑定物理内存块（tensor buffer）。
   - 支持计算内核无需修改即可访问按需加载的权重。

2. **带宽均衡的专家存储层次（Bandwidth-Balanced Storage Hierarchy）**
   - 构建多级存储：压缩后的 GPU 显存 + 主机 DRAM。
   - 采用比例分配策略，使各存储后端的加载时间均衡，最大化整体加载带宽。

3. **预算感知的驻留规划器（Budget-Aware Residency Planner）**
   - 动态调整保留在 GPU 中的专家比例 α。
   - 根据当前 **compute-to-load ratio (p)** 和 KV Cache 占用情况，闭环控制 α，优先保障 KV Cache 容量。

### **相比现有方法的优势**
| 维度 | 传统方法（如 vLLM） | FluxMoE |
|------|---------------------|---------|
| 权重管理 | 全部专家常驻 GPU | 按需流式加载，仅保留必要部分 |
| 显存利用 | 被大量闲置权重占用 | 更多空间用于 KV Cache 和激活缓冲区 |
| 可扩展性 | 受限于 GPU 显存容量 | 支持远超 GPU 容量的模型部署 |
| 性能表现 | 高负载下因显存不足导致吞吐骤降 | 显著提升高批大小、长上下文场景下的吞吐 |

---

## **2. 核心实验方法和设置**

### **使用的模型与数据集**
- **模型**：
  - `Mixtral-8×7B-Instruct`（47B 参数）
  - `Qwen3-Next-80B-A3B-Instruct`（80B 参数）
- **数据集**：`ShareGPT` —— 包含真实对话记录的数据集，用于构建推理请求。

### **实验设置**
- **硬件平台**：
  - 服务器节点：Intel Xeon Platinum 8358，2TB 主机 DRAM，4×NVIDIA L40 GPU（每卡 48GB GDDR6）
  - 使用 Tensor Parallelism（TP=4 或 TP=2）进行分布式推理。
- **工作负载设计**：
  - 批大小（batch size）：32 ~ 256
  - 上下文长度（context length）：1,024 ~ 4,096 tokens
  - 聚焦于**高吞吐、资源受限**的服务场景。

### **评估指标**
- **主要指标**：**Aggregate Throughput (tokens/sec)**  
  衡量单位时间内生成的总 token 数，反映系统整体服务能力。
- **次要分析指标**：
  - KV Cache 利用率
  - 显存占用分布
  - Compute-to-load ratio $ p $

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **vLLM** | 工业界标准框架，要求所有专家常驻 GPU，KV Cache 不足时向主机交换 |
| **vLLM-O** | 改进版 vLLM，支持部分专家卸载至主机 DRAM，但无压缩机制 |
| **FluxMoE-H** | 消融版本，仅使用粗粒度压缩+卸载，缺乏带宽均衡调度 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **Exp#1：性能边界环境（足够显存）**
- 在 `Qwen3-Next-80B-A3B` 上测试（TP=4）
- **最大吞吐增益达 3.0× vs vLLM**
- 当 batch size=256, context=4096 时：
  - vLLM 吞吐下降 32.2%（因频繁 KV Cache 交换）
  - FluxMoE 维持稳定增长，达到 **~900 tokens/sec**
- 上下文扩展至 4096 时，vLLM 性能下降 45.3%，而 FluxMoE 仅下降 20.5%

#### ✅ **Exp#2：容量边界环境（显存严重受限）**
- 在 `Mixtral-8×7B` 上测试（TP=2，显存不足）
- vLLM 因 OOM 无法运行
- FluxMoE 成功部署并实现：
  - 比 vLLM-O 提升 **>10× 吞吐**
  - 在 batch=256, ctx=4096 下，vLLM-O 仅 3.7 tokens/sec，FluxMoE 达到约 **40 tokens/sec**
- FluxMoE 比 FluxMoE-H 提升 **22.9% ~ 28.5%**，验证了带宽均衡设计的有效性

#### ✅ **Exp#3：动态驻留适应性测试**
- 动态调节专家驻留比例 α
- 在推理过程中逐步释放专家（共释放 ~5.3GB 显存）
- 结果显示：
  - 吞吐未低于固定 α=1.0 的水平
  - 证明 I/O 开销被完全隐藏在计算流水线中
  - 实现“零代价”的显存回收

#### ✅ **Exp#4：PagedTensor 开销分析**
- 在所有专家均驻留 GPU 的理想条件下测试
- 最大管理开销仅为 **3.0%**（出现在 batch=64, ctx=4096）
- 表明 PagedTensor 的虚拟化机制引入的额外开销可忽略不计

---

## **4. 关键结论和发现**

### **主要发现**
1. **专家权重长期驻留 GPU 是重大资源浪费**，尤其是在解码阶段。
2. **KV Cache 容量是决定 MoE 推理吞吐的关键瓶颈**，应优先保障其显存配额。
3. **通过 expert paging 实现“逻辑身份”与“物理驻留”的解耦**，可在不影响精度的前提下大幅提升资源利用率。
4. **带宽均衡的多级存储 + 动态驻留控制** 可实现计算与 I/O 的完美重叠，避免流水线停顿。
5. FluxMoE 在内存密集型场景下实现了高达 **3.0× 的吞吐提升**，且兼容现有推理框架（基于 vLLM 实现）。

### **方法的局限性**
- **依赖 PCIe 带宽**：当主机传输带宽成为瓶颈时，仍可能影响性能。
- **压缩仅作用于 exponent 位**：虽有效，但仍有进一步优化空间（如结构化稀疏、量化等）。
- **当前原型假设专家调度已知**：未集成路由预测模块，适用于预知路由路径的场景。

### **未来工作方向**
- 扩展支持 **统一三态驻留模型**（uncompressed GPU / compressed GPU / CPU offloaded）
- 集成 **轻量级路由预测器**，实现更智能的预取策略
- 探索 **跨节点分布式 expert paging**，支持超大规模 MoE 模型的云原生部署
- 结合 **动态 KV Cache 扩容机制**，将节省的显存实时用于提升并发能力

---

> 💡 **一句话总结**：  
> **FluxMoE 通过“专家分页”机制，将 MoE 模型的专家参数从 GPU 显存中解放出来，显著提升了 KV Cache 容量和推理吞吐，在资源受限环境下实现最高 3.0× 的性能飞跃，同时保持模型精度不变。**

</details>

---

### 12. [STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation](https://arxiv.org/abs/2604.02756)

**Authors**: Zijin Liu, Xu Geng, Wenshuai Xu, Xiang Zhao, Yan Xia, You Song  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.02756v1  

#### Abstract
Accurate crowd simulation is crucial for public safety management, emergency evacuation planning, and intelligent transportation systems. However, existing methods, which typically model crowds as a collection of independent individual trajectories, are limited in their ability to capture macroscopi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的 crowd simulation 方法存在以下局限性：
- **微观建模主导**：主流方法将人群视为独立个体轨迹的集合，忽略了宏观物理规律（如质量守恒），导致长期模拟中误差累积、稳定性差。
- **物理一致性不足**：纯 data-driven 深度学习方法（如 GNN、diffusion models）虽能捕捉复杂模式，但常产生违反物理规律的行为（如不合理的拥堵或碰撞）。
- **推理效率低**：基于 diffusion 或自回归生成的方法计算开销大，难以应用于大规模实时仿真。

### **提出的新方法与思路**
作者提出了 **Spatio-Temporal Decoupled Differential Equation Network (STDDN)**，其核心思想是：
- 将人群视为连续介质，引入流体力学中的 **continuity equation** 作为强物理约束，指导微观轨迹预测。
- 构建一个 **Neural ODE** 框架来建模宏观密度场的时间演化，并通过可微分模块连接微观个体运动与宏观密度变化。
- 实现 **macro-micro 耦合建模**：宏观物理规律对微观轨迹预测进行端到端的正则化，提升稳定性和物理一致性。

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **物理一致性** | 显式嵌入 continuity equation，确保质量守恒，减少非物理行为（如异常聚集）。 |
| **长期稳定性** | 宏观约束有效抑制误差在时间上的累积，显著提升长时预测性能。 |
| **推理效率** | 单步前向传播即可完成密度演化建模，避免 diffusion 模型多步去噪过程，大幅降低延迟。 |
| **可解释性增强** | 动态图结构显式建模密度通量（flux），提高模型决策过程的物理可解释性。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在四个真实世界行人轨迹数据集上进行了评估：
- **GC**：高密度场景，选取300秒密集片段。
- **UCY**：包含 ZARA1、ZARA2 和 UCY 子场景，共216秒。
- **ETH** 与 **HOTEL**：标准测试集，采用原始训练/测试划分。
> 所有数据均经过坐标转换和立方插值处理至统一时间步长（0.08s）以提高精度。

### **实验设置与评估指标**

#### **评估指标**
| 类别 | 指标 | 描述 |
|------|------|------|
| **轨迹准确性** | MAE, OT, FDE, DTW, MMD | 衡量位置误差、分布相似性、终点偏差等 |
| **物理合理性** | #Colli, DEA | 碰撞次数、局部密度估计准确性 |
| **推理效率** | Latency (ms), FPS, GFLOPs, #Pars | 推理延迟、帧率、浮点运算量、参数量 |
| **训练成本** | Training time, GPU Memory | 训练耗时与显存占用（见 Appendix A.8） |

#### **基线方法对比**
分为三类进行比较：
- **Physics-based**: SFM, CA
- **Data-driven**: STGCNN, PECNet, MID
- **Physics-guided DL**: PCS, NSP, SPDiff

其中，**SPDiff** 是当前最先进的 physics-informed diffusion model，作为主要对比基准。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（平均5次运行）**

#### **表1：GC 和 UCY 数据集上的总体性能**
| Model | GC MAE ↓ | GC OT ↓ | GC Latency ↓ | UCY MAE ↓ | UCY OT ↓ | UCY Latency ↓ |
|-------|----------|---------|--------------|------------|-----------|----------------|
| SPDiff | 0.9116 | 1.3925 | 206.99 ms | 1.8760 | 4.0564 | 471.05 ms |
| **Ours (STDDN)** | **0.8875** | **1.3582** | **86.85 ms** | **1.7747** | **3.6503** | **44.66 ms** |

> ✅ **GC 上提升 2.6% MAE，延迟下降 58%；UCY 上提升 5.4% MAE，延迟下降 90%**

#### **表2：ETH 和 HOTEL 数据集上的性能**
| Model | ETH MAE ↓ | ETH OT ↓ | ETH Latency ↓ | HOTEL MAE ↓ | HOTEL OT ↓ | HOTEL Latency ↓ |
|-------|-----------|----------|----------------|---------------|-------------|------------------|
| SPDiff | 0.5527 | 0.8706 | 81.41 ms | 0.3380 | 0.1646 | 68.57 ms |
| **Ours (STDDN)** | **0.5185** | **0.6918** | **30.57 ms** | **0.2952** | **0.1445** | **17.50 ms** |

> ✅ **ETH 上 MAE 提升 6.0%，OT 下降 19.8%；HOTEL 上 MAE 提升 12.7%，延迟下降 75%**

### **与其他方法的对比结果**
- 在所有数据集上，STDDN 均取得 **最佳 MAE 和 OT 性能**，且远超第二名 SPDiff。
- 推理速度方面，**Latency 平均降低 50–90%**，**FPS 提升数倍**（如 HOTEL 上达 57 FPS vs SPDiff 的 14.6 FPS）。
- 参数量更少（仅 0.07M–0.20M），模型轻量化优势明显。

### **消融实验结果（Ablation Study）**
在 GC 和 UCY 上验证各组件作用：

| 变体 | GC MAE ↑ | GC OT ↑ | UCY MAE ↑ | UCY OT ↑ | 结论 |
|------|----------|---------|------------|-----------|--------|
| w/o ODE | 1.3784 | 2.4956 | 2.4867 | 6.0586 | 移除 Neural ODE 导致严重误差累积 |
| w/o Cross-net | 0.9784 | 1.4732 | 1.8926 | 4.9532 | 缺少 cross-grid detection 损害质量守恒 |
| w/o NN loss | 1.2387 | 2.3466 | 1.9327 | 4.2514 | 仅依赖物理损失无法拟合复杂动态 |
| Discrete NN | 0.8875 | 1.3582 | 1.7747 | 3.6503 | 性能与完整模型相当，说明离散建模已足够 |
| Dopri5/RK4 | >0.8875 | >1.3582 | >1.7747 | >3.6503 | 高阶求解器反而性能下降，Euler 更适合任务特性 |

> 🔍 **关键发现**：
> - **Neural ODE + continuity constraint 是抑制误差累积的关键**。
> - **CGD 模块对保持物理一致性至关重要**。
> - **Euler solver 最优**：因其与离散时间观测对齐，兼顾效率与精度。

---

## **4. 关键结论和发现**

### **主要发现**
1. **宏观物理约束显著提升长期仿真稳定性**  
   引入 continuity equation 作为结构性先验，能有效引导微观轨迹预测，缓解传统自回归模型中的误差传播问题。

2. **宏观-微观耦合优于纯微观建模**  
   STDDN 成功实现了从“个体行为模拟”到“群体流动建模”的范式转变，提升了全局一致性和物理合理性。

3. **高效推理架构设计可行且必要**  
   相比 diffusion-based 方法需多次迭代去噪，STDDN 利用 single-step ODE solver 实现快速前向推理，在保证精度的同时大幅提升效率。

4. **可微分模块设计保障训练稳定性**  
   - **Differentiable Density Mapping (DDM)**：基于 RBF 的软分配避免硬量化带来的梯度不连续。
   - **Continuous Grid Detection (CGD)**：利用 JS 散度建模跨格流动，实现端到端可导。

### **方法的局限性**
1. **边界效应未完全建模**  
   当前行人进出场景被视为状态初始化/终止，未显式建模为源项（source/sink）。理想情况下应扩展 continuity equation 为：
   $$
   \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = S
   $$
   其中 $S$ 表示人流进出。

2. **网格离散化带来表达限制**  
   固定粒度的 grid partition 可能在复杂空间结构中丢失细节，未来可探索 continuous spatial representation（如 implicit functions）或多尺度图结构。

3. **强物理约束可能压制潜在运动模式**  
   过于严格的物理正则化可能限制模型学习某些非典型但合理的社会行为，未来可研究 soft-constraint 或 adaptive weighting 机制。

### **未来工作方向**
- 引入 **source/sink terms** 处理开放系统的人流进出预测。
- 探索 **multi-scale graph 或 continuous spatial modeling** 改善空间表达能力。
- 设计 **adaptive physics-data fusion 权重机制**，平衡归纳偏置与数据驱动灵活性。
- 应用于更大规模场景（如城市级疏散模拟），结合 **sparsification、model compression、parallel acceleration** 提升工程实用性。

---

> 📌 **总结一句话**：  
> **STDDN 通过将 continuity equation 融入 Neural ODE 框架，首次实现了宏观物理规律对微观轨迹预测的端到端引导，在提升 long-term crowd simulation 准确性与稳定性的同时，大幅降低了推理延迟，为物理一致、高效可扩展的智能仿真提供了新范式。**

</details>

---

### 13. [Aligning Progress and Feasibility: A Neuro-Symbolic Dual Memory Framework for Long-Horizon LLM Agents](https://arxiv.org/abs/2604.02734)

**Authors**: Bin Wen, Ruoxuan Zhang, Yang Chen, Hongxia Xie, Lan-Zhe Guo  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.02734v1  

#### Abstract
Large language models (LLMs) have demonstrated strong potential in long-horizon decision-making tasks, such as embodied manipulation and web interaction. However, agents frequently struggle with endless trial-and-error loops or deviate from the main objective in complex environments. We attribute th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Aligning Progress and Feasibility: A Neuro-Symbolic Dual Memory Framework for Long-Horizon LLM Agents

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

该论文针对 **Long-Horizon LLM Agents** 在复杂任务中常见的两类失败模式：

- **Progress Drift（进度漂移）**：代理在全局上偏离任务目标，无法有效推进任务阶段。
- **Feasibility Violation（可行性违反）**：代理执行局部不可行的动作（如尝试将物体放入未打开的柜子），导致无效动作和试错循环。

作者指出，现有方法通常用单一机制同时处理这两个问题，但由于二者本质不同——**Progress 需要模糊语义泛化，Feasibility 需要严格逻辑约束**——这种统一范式存在根本性缺陷。

---

### 🚀 提出了什么新方法或新思路

提出 **Neuro-Symbolic Dual Memory Framework（神经符号双记忆框架）**，明确解耦两种能力：

| 模块 | 类型 | 功能 |
|------|------|------|
| **Progress Memory** | Neural（基于神经网络） | 从成功轨迹中提取**语义蓝图（Procedural Blueprint）**，提供**阶段感知的语义引导**，防止 Progress Drift |
| **Feasibility Memory** | Symbolic（基于符号逻辑） | 从失败交互中归纳出可执行的 **Python 验证函数（Verification Functions）**，进行**严格的动作可行性检查**，防止 Feasibility Violation |

该框架在推理过程中同步调用两个记忆模块，形成“**神经引导 + 符号验证**”的协同决策闭环。

---

### 🔍 相比现有方法的优势

- **功能分离更合理**：Progress 对应语义规划（适合神经模型），Feasibility 对应逻辑验证（适合符号系统），避免神经模型在硬约束上的幻觉。
- **可解释性强**：Feasibility Memory 输出错误原因和修正建议，支持 LLM 迭代优化。
- **模块互补**：消融实验证明两个模块分别解决不同失败模式，组合后效果显著提升。
- **通用性强**：框架设计不依赖特定环境，通过环境适配器即可迁移至不同任务。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

在三个具有长程依赖和复杂约束的基准上进行评估：

| 数据集 | 任务类型 | 特点 |
|-------|--------|------|
| **ALFWorld** | 文本版具身操作（Embodied Manipulation） | 模拟家庭环境中导航、拾取、加热等操作，需满足物理状态约束 |
| **WebShop** | 网页购物决策（Web Interaction） | 在模拟电商网站中搜索、筛选、购买商品，需遵循页面状态和UI规则 |
| **TextCraft** | 文本版合成制造（Compositional Synthesis） | 类似 Minecraft 的物品合成任务，需递归构建中间材料 |

所有测试任务均来自与训练集**完全不重叠的 unseen split**，确保公平性。

---

### 📊 实验设置和评估指标

#### 主要指标：

- **Success Rate (SR)**：任务完成率
- **Score**（仅 WebShop）：综合得分（考虑价格、匹配度）
- **Invalid Action Rate (IAR)**：因违反环境规则被拒绝的动作占比
- **Average Trajectory Length (ATL)**：平均动作步数，反映效率

#### 实验配置：

- **Backbone Model**：统一使用 `gpt-4o-2024-11-20`，temperature=0
- **离线训练集**：各任务使用 50 个独立于测试集的训练任务收集经验
- **推理预算**：ALFWorld（50步）、WebShop（15步）、TextCraft（40步）

---

### ⚔️ 基线方法对比

对比了当前主流的 Long-Horizon Agent 方法：

| 基线方法 | 核心机制 |
|--------|--------|
| **ReAct** | 推理与行动交替 |
| **Reflexion** | 失败后语言反思 |
| **ADaPT** | 层次化任务分解 |
| **StateAct** | 显式状态追踪 |
| **ExpeL** | 经验规则抽取 |
| **WALL-E 2.0** | 神经符号世界模型对齐 |
| **AWM** | 工作流模式抽象为文本记忆 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Table 1）

| Method | ALFWorld SR (%) | WebShop SR (%) | WebShop Score | TextCraft SR (%) |
|--------|------------------|----------------|---------------|------------------|
| ReAct | 76.87 | 32 | 0.5010 | 62 |
| Reflexion | 82.66 | 35 | 0.5204 | 69 |
| ExpeL | 85.07 | 29 | 0.4582 | 88 |
| AWM | 88.81 | 32 | 0.5160 | 66 |
| **Ours** | **94.78** | **51** | **0.7132** | **94** |

> ✅ **全面领先**：在所有三项任务上均取得最优表现，尤其在 WebShop 上成功率提升 **16个百分点**，评分提升超 **20%**。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）双记忆模块的互补性（Table 2）

| 方法 | SR (%) | IAR (%) | ATL |
|------|--------|---------|-----|
| Ours (Full) | **94.78** | **11.81** | **14.60** |
| w/o Progress Memory | 90.30 | 12.98 | 20.30 |
| w/o Feasibility Memory | 85.82 | **26.33** | 20.49 |

> - 移除 **Feasibility Memory** → IAR 翻倍，说明其有效抑制无效动作  
> - 移除 **Progress Memory** → ATL 显著增加，说明其保障高效阶段推进  
> - 两者缺一不可，**互补而非冗余**

#### （2）Progress Memory 设计分析（Table 3）

| 配置 | SR (%) | ATL |
|------|--------|-----|
| No Memory | 90 | 21.22 |
| Standard RAG（检索完整轨迹） | 84 | 21.64 |
| Blueprint + Task Retrieval | 92 | 16.42 |
| **Blueprint + Anchor Retrieval** | **94** | **16.18** |

> - 单纯检索成功轨迹反而降低性能（干扰上下文）
> - **结构化蓝图 + 锚点级检索** 是关键，实现精准阶段对齐

#### （3）Feasibility Memory 机制比较（Table 4）

| 配置 | SR (%) | IAR (%) | ATL |
|------|--------|---------|-----|
| No Rules | 88 | 21.63 | 20.90 |
| Prompt Rules | 84 | 12.11 | 18.16 |
| **Verifier Rules（可执行代码）** | **94** | 11.00 | **16.18** |
| Prompt + Verifier | 92 | **4.66** | 16.32 |

> - 虽然 Prompt + Verifier 的 IAR 最低，但 SR 和 ATL 不及单独 Verifier
> - 表明**过于保守的语言提示会阻碍任务进展**
> - **可执行的符号验证器** 在“减少错误”和“保持进展”之间达到最佳平衡

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Long-Horizon Agent 失败源于双重对齐危机**：
   - 全局 **Progress Alignment** 与局部 **Feasibility Alignment** 是两类本质不同的挑战，不应由同一机制处理。

2. **神经符号双记忆是更优架构选择**：
   - **Progress Memory（神经）** 擅长语义泛化，指导任务阶段演进；
   - **Feasibility Memory（符号）** 擅长逻辑验证，拦截非法动作；
   - 二者协同实现了“**稳定执行 + 高效推进**”。

3. **结构化经验优于原始轨迹记忆**：
   - 将成功轨迹提炼为 **Procedural Blueprint** 并按 **Anchor** 检索，显著提升引导质量。

4. **可执行验证器优于语言提示约束**：
   - 使用 Python 函数进行符号验证，比在 prompt 中加入规则更可靠、更灵活。

---

### ⚠️ 方法的局限性

- **依赖离线轨迹数据**：需要一定数量的成功/失败交互来构建双记忆，**在稀疏奖励或难以获取失败信号的环境中可能受限**。
- **符号规则归纳依赖 Inductor Agent**：规则生成过程仍依赖 LLM 的归纳能力，可能存在噪声或遗漏。
- **环境适配成本**：虽然框架通用，但每个环境需定制 **scene graph 构建逻辑** 和 **valid signal 定义**。

---

### 🔮 未来工作方向

- 在 **reward 更稀疏、failure 更隐蔽** 的真实场景中探索记忆构建策略。
- 引入 **在线学习机制**，动态更新双记忆以适应新任务分布。
- 探索 **自动化 rule induction**，减少对 LLM 归纳的依赖。
- 扩展到 **多模态 Agent**（如视觉输入），结合 VLM 与双记忆框架。

---

> 💡 **一句话总结**：  
> 本文提出 **Neuro-Symbolic Dual Memory Framework**，通过解耦 **Progress Memory（神经）** 与 **Feasibility Memory（符号）**，分别应对长程任务中的**语义漂移**与**动作不可行**问题，在 ALFWorld、WebShop 和 TextCraft 上全面超越现有方法，验证了“**不同失败模式应由不同机制解决**”的设计哲学。

</details>

---

### 14. [ESL-Bench: An Event-Driven Synthetic Longitudinal Benchmark for Health Agents](https://arxiv.org/abs/2604.02834)

**Authors**: Chao Li, Cailiang Liu, Ang Gao, Kexin Deng, Shu Zhang, Langping Xu, Xiaotong Shi, Xionghao Ding, Jian Pei, Xun Jiang  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.02834v1  

#### Abstract
Longitudinal health agents must reason across multi-source trajectories that combine continuous device streams, sparse clinical exams, and episodic life events - yet evaluating them is hard: real-world data cannot be released at scale, and temporally grounded attribution questions seldom admit defin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ESL-Bench: An Event-Driven Synthetic Longitudinal Benchmark for Health Agents 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前在评估**纵向健康代理（longitudinal health agents）**时面临三大挑战：

- **隐私与可扩展性问题**：真实世界医疗数据因隐私限制难以大规模公开共享。
- **缺乏可验证的真值（ground truth）**：现实 EHR 数据中事件边界模糊、混杂因素多，“是什么导致了某项指标变化？”这类归因问题往往没有明确答案。
- **现有基准不具诊断性**：现有数据集（如 MIMIC-IV、EHRSHOT）多为单模态、短周期数据，无法支持跨源、跨时间跨度的复杂推理任务。

这些问题使得对健康代理的**结构化时序推理能力**（temporal reasoning）难以进行公平、可复现、细粒度的评估。

---

### 🚀 提出的新方法与新思路

作者提出 **ESL-Bench**，一个**事件驱动的合成纵向健康基准**（event-driven synthetic longitudinal benchmark），其核心思想是：

> 将每个用户的健康轨迹建模为：  
> **基础状态 + 离散事件的叠加效应**

#### 创新机制：
- **事件作为一等公民（first-class objects）**：每一个生活或临床事件（如“开始慢跑”、“高钠饮食”）都带有显式的**影响参数**（affected indicators, magnitude, onset/fade-out 时间）。
- **透明的动力学模型**：每个事件通过 **sigmoid-onset + exponential-decay kernel** 影响生理指标，多个事件以**软饱和叠加**方式组合，确保生理合理性。
- **混合生成管道（hybrid pipeline）**：
  - **LLM 负责稀疏语义决策**：生成 profile、事件描述、exam narrative。
  - **算法模拟负责密集动态**：基于确定性方程生成每日 device stream，保证可追溯性和一致性。

---

### 🔍 相比现有方法的优势

| 维度 | ESL-Bench 优势 |
|------|----------------|
| **Ground Truth 可计算性** | 所有“归因”类问题的答案均可从事件-指标关系图中程序化推导，实现**机制可恢复性（mechanism recovery）** |
| **可控性与可审计性** | 显式定义事件动力学，避免黑箱生成；提供 audit report 追踪合规性 |
| **多维度、分难度评估** | 设计 5 个维度 × 3 个难度层级的 10,000 个查询，精准定位模型短板 |
| **支持多跳推理测试** | 特别设计 Comparison 和 Explanation 类问题，要求跨事件、跨来源连接证据 |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

- **ESL-Bench 自研合成数据集**：
  - 包含 **100 名合成用户**，每人拥有 **1–5 年** 的完整轨迹。
  - 每人数据包 `B` 包括：
    - Profile（慢性病、用药史）
    - Trajectory Plan（叙事弧线，分阶段健康主题）
    - Daily Device Stream（静息心率、步数、SpO₂ 等）
    - Sparse Exam Visits（实验室检查）
    - Event Log（带参数的生活/健康事件）
    - Audit Report（质量监控）

---

### 🧪 实验设置与评估指标

#### 评估任务设计：**五维三级查询体系**

| 维度 | 描述 | 示例 |
|------|------|------|
| **Lookup** | 查找特定时间点的数值或属性 | “2024-03-15 的静息心率是多少？” |
| **Trend** | 分析趋势、波动、转折点 | “哪个月步数均值最高？” |
| **Comparison** | 跨事件/时间段比较 | “运动前后静息心率变化比？” |
| **Anomaly** | 异常检测与持续性分析 | “空腹血糖是否曾异常？” |
| **Explanation** | 因果归因与主导事件识别 | “为何空腹血糖下降？列出贡献最大的事件” |

> 每个用户配 100 个查询（共 10,000），按 **20% Easy, 30% Medium, 50% Hard** 分布。

#### 评分协议（Two-stage Scoring Protocol）

1. **Stage 1: Programmatic Checks**
   - 数值容忍匹配：`|v_pred - v_gt| ≤ max(ε_abs, ε_rel * |v_gt|)`，其中 `ε_abs=0.01`, `ε_rel=0.01`
   - 集合精确匹配（exact set match）
   - 失败则得分为 0

2. **Stage 2: LLM Judge（GPT-4.1）打分**
   - 对通过 Stage 1 的响应按 rubric 打 0–2 分，维度相关：
     - Lookup/Anomaly：正确性 + 格式
     - Trend/Comparison：统计逻辑 + 推理链
     - Explanation：证据排序 + 非因果语言控制
   - 最终得分 = `binary_gate × normalized_rubric_score`

---

### 🆚 基线方法对比

| 范式 | 方法 | 描述 |
|------|------|------|
| **LLM w/ Tools** | GPT-4.2, GPT-4.4, Gemini 3 Flash, Sonnet 4.6, MiniMax M2.5, GLM-5 | 使用原生工具调用能力，访问统一工具接口（lookup, query, read, search） |
| **DB Agent** | Theta General / Expert / Smart Expert | 基于 GPT-4.4，在 DuckDB 上执行结构化 API 调用（filter, aggregate, join） |
| **Memory RAG** | HippoRAG (k=10/20/50) | 基于 Hippocampus 启发的记忆架构，支持长期记忆整合与检索 |

> 所有方法输入均为相同结构化 artifact，输出需符合统一 JSON schema 以便自动评分。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 10 & 11）

| 方法 | 总体准确率 | Lookup | Trend | Comp. | Anom. | Expl. |
|------|-----------|--------|-------|--------|--------|--------|
| **Gemini 3 Flash** | **62.9%** | 73.5% | **94.8%** | 74.4% | 64.3% | 25.3% |
| **Theta General** | 57.9% | 73.8% | 70.0% | 67.0% | 66.1% | 26.4% |
| **HippoRAG (k=50)** | 37.7% | 67.7% | 38.0% | **18.1%** | 35.7% | 17.9% |
| **GPT-4.2** | 45.4% | 52.7% | 71.0% | 41.0% | 56.4% | 15.5% |

#### 按难度分层表现（Hard 查询尤为关键）：

| 方法 | Easy | Medium | Hard |
|------|------|--------|------|
| **Gemini 3 Flash** | 75.0% | 70.6% | **55.0%** |
| **Theta Expert** | 73.8% | 55.2% | 51.9% |
| **HippoRAG (k=50)** | 68.8% | 34.3% | 30.4% |

---

### 🔬 与基线方法的对比结果

- **DB Agents 显著优于 Memory RAG**：
  - DB Agents 准确率：**48–58%**
  - Memory RAG 准确率：**30–38%**
  - 差距集中在 **Comparison（差达 50pp）和 Explanation（差约 10pp）**

- **LLM w/ Tools 表现分化大**：
  - Gemini 3 Flash 表现最佳（62.9%），尤其在 **Trend（94.8%）**
  - GPT-4.2 表现最弱（45.4%），显示模型本身能力差异显著

- **Memory RAG 在 Comparison 上崩溃**：
  - 最高达仅 **18.1%**（k=50），说明其难以完成**多跳 join** 和跨源对齐

- **增加检索预算（k）效果有限**：
  - HippoRAG 从 k=10 → k=50，仅在 Comparison 上提升明显，其余维度无改善

---

### ❌ 消融实验与错误分析（Error Analysis）

虽未正式命名“消融”，但通过案例揭示关键瓶颈：

#### Case 1: Cross-source indicator confusion（HippoRAG）
- 错误将 **diastolic BP** 的 exam chunk 误用于 **systolic BP** 的 Comparison 查询
- 根本原因：chunk 边界割裂相关字段，**语义相似性无法区分指标细节**

#### Case 2: Temporal window misalignment（GPT-4.2）
- 将事件结束日当作参考点计算“前14天”，导致窗口重叠
- 属于系统性 **temporal anchoring error**，常见于 LLM w/ tools

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **结构化时序推理是瓶颈**：
   - 当前主流方法（尤其是 Memory RAG）在 **Comparison 和 Explanation** 上表现极差，暴露其**多跳推理与证据归因能力不足**。

2. **DB Agents 更适合复杂查询**：
   - 结构化 API 支持 **filter-aggregate-join** 操作，在需要精确时间对齐和跨源聚合的任务中具有压倒性优势。

3. **LLM 本身能力决定上限**：
   - 即使使用相同工具接口，不同 LLM（如 Gemini vs GPT-4.2）表现差异巨大，表明**推理与工具调度能力至关重要**。

4. **难度梯度有效分离模型能力**：
   - Easy 查询多数模型可胜任；
   - Hard 查询（尤其是 Explanation）普遍低于 30%，成为“能力试金石”。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **非生理仿真器** | 事件-指标关系简化，**不应解释为真实临床效应大小** |
| **依赖 LLM 生成质量** | Profile、event narrative 受 LLM 偏见影响，可能引入开发者假设偏差 |
| **理想化事件边界** | 合成事件边界清晰，而真实临床记录更模糊、碎片化 |
| **解释维度非因果推断** | 测的是“机制可恢复性”，而非真实世界中的“因果发现” |
| **生成成本高** | 单用户平均消耗 ~614 次 Gemini API 调用，约 \$74（按当前定价） |

---

### 🔮 未来工作方向

1. **建模设备非佩戴与采样不规则性**：作为一等过程纳入生成流程。
2. **增强跨指标约束**：超越共享噪声因子，引入更复杂的生理耦合模型。
3. **引入部分信用评分（partial credit）**：
   - 如 set-F1、tolerance sweep，缓解严格 exact-match 带来的惩罚。
4. **多语言支持**：扩展 multilingual query 能力。
5. **真实数据校准（real-data calibration）**：
   - 使用真实队列统计数据锚定事件频率、指标分布，缩小外部有效性差距。
6. **新增模态**：
   - 加入 imaging reports、free-text notes，进一步拓宽 benchmark 范围。

---

## 总结

> **ESL-Bench 不是一个追求临床保真的模拟器，而是一个追求“可诊断性”的推理压力测试平台。**

它通过**事件驱动 + 显式动力学 + 程序化真值**的设计，首次实现了对健康代理**结构化时序推理能力**的可控、可复现、细粒度评估。实验表明，当前 Memory RAG 架构在复杂纵向任务中严重受限，而 DB-native agents 和强推理 LLM 更具潜力。该工作为下一代健康代理的发展提供了清晰的能力坐标系。

</details>

---

### 15. [Multi-Turn Reinforcement Learning for Tool-Calling Agents with Iterative Reward Calibration](https://arxiv.org/abs/2604.02869)

**Authors**: Wachiravit Modecrua, Krittanon Kaewtawee, Krittin Pachtrachai, Touchapon Kraisingkorn  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.02869v1  

#### Abstract
Training tool-calling agents with reinforcement learning on multi-turn tasks remains challenging due to sparse outcome rewards and difficult credit assignment across conversation turns. We present the first application of MT-GRPO (Multi-Turn Group Relative Policy Optimization) combined with GTPO (Ge...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Multi-Turn Reinforcement Learning for Tool-Calling Agents with Iterative Reward Calibration*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文针对在多轮对话任务中训练 **tool-calling agents** 所面临的三大挑战：
- **稀疏奖励（Sparse rewards）**：通常只有最终任务成功与否的二元信号（0/1），难以进行有效的信用分配（credit assignment）。
- **跨回合信用分配困难**：工具调用分散在多个对话回合中，如何将最终奖励合理地归因到每个动作是关键难题。
- **密集每回合奖励（dense per-turn rewards）设计不当导致性能下降**：即使直觉上合理的奖励设计，也可能因“优势方向错位”（advantage misalignment）而严重损害模型表现（最多下降14个百分点）。

---

### 🚀 提出的新方法与创新思路

#### （1）**MT-GRPO + GTPO Hybrid Advantage**
首次将 **MT-GRPO**（Multi-Turn Group Relative Policy Optimization）与 **GTPO**（Generalized Token-level Policy Optimization）结合，提出一种混合优势函数（hybrid advantage）公式：

$$
\text{hybrid } A_{i,k} = \text{GN}\left(\sum_{l=k}^{K-1} \gamma^{l-k} r_l + (1-\gamma^0)\right) + \lambda \cdot A_o
$$

其中：
- $\text{GN}$ 表示 group normalization，
- $\gamma=0.9$ 是折扣因子，
- $\lambda=0..3$ 用于衰减 outcome advantage $A_o$。

👉 **优势**：通过折扣机制自然减弱远期 outcome 优势对早期回合的影响，同时保留弱但正确方向的全局信号，彻底消除“优势错配”问题。

#### （2）**Iterative Reward Calibration (IRC)** —— 核心创新
提出系统化奖励校准流程，避免凭直觉设计奖励带来的风险。IRC 包括以下步骤：
1. 收集 rollout 数据；
2. 使用 **point-biserial correlation** 分析各奖励层级（tier）与任务成功的实证相关性；
3. 动态调整奖励值，确保其与 discriminative power 成正比；
4. 验证 group-normalized 后的优势方向是否符合预期；
5. 迭代优化直至无方向冲突且相关性达标。

👉 **关键发现驱动的奖励修正**：
- `read-only` 工具调用（如查询）应设为 **r=0**（原设为0.3），因其判别力极低（+0.1pp）；
- `state-changing` 非黄金操作应从轻微奖励改为 **惩罚（r=-0.1）**；
- 引入 `_deep_equal` 函数处理 JSON 参数差异（排序、类型转换等），减少 **23.5% 的误匹配（false positives）**。

#### （3）**Deep Argument Comparison**
开发语义等价判断函数 `_deep_equal`，标准化参数比较逻辑，显著降低奖励噪声。

---

### 🔍 相比现有方法的优势
| 方法 | 局限性 | 本文改进 |
|------|--------|---------|
| 原始 MT-GRPO | 密集奖励下出现 advantage misalignment | 引入 GTPO 折扣 + $\lambda$ 衰减解决 |
| 稀疏奖励训练 | 缺乏细粒度指导 | IRC 提供可解释、高判别性的密集奖励 |
| 直觉式奖励设计 | 易造成性能退化（↓14pp） | 数据驱动校准，提升稳定性与性能 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Tau-Bench (v1)**：用于训练。
  - 领域：航空公司客服任务（flight search, booking, cancellation, policy compliance）。
  - 特点：包含用户指令、golden action 序列、数据库状态哈希验证。
- **Tau2-Bench (v2)**：独立更新版本，用于测试。
  - 50个新任务 × 4次模拟 = 200次评估。
  - ✅ 保证非重叠训练/测试集，衡量泛化能力而非记忆。

### ⚙️ 实验设置
- **模型**：
  - `Qwen3.5-4B`（4B dense）
  - `Qwen3-30B-A3B MoE`（30.5B total, 3B active）
- **用户模拟器（User Simulator）**：
  - 训练时使用 **DeepSeek-V3**
  - 测试时使用 **GPT-4.1**（greedy decoding, temp=0）
- **框架与硬件**：
  - 使用 `verl` 框架 + Megatron-Core
  - 8×NVIDIA H20 GPU（96GB each）

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Pass%** | 数据库状态匹配目标的比例（主指标） |
| **Pass'** | 单任务4次全部成功 |
| **Average Reward** | 平均每条轨迹得分 |
| **Action Accuracy** | 工具调用名称与参数完全匹配率 |
| **Turns / Duration** | 对话轮数与时长，衡量效率 |

### 🆚 基线方法对比
| 基线 | 类型 |
|------|------|
| Sparse Reward Only (MT-GRPO V3) | 仅 outcome 奖励 |
| Naive Dense Reward (V5) | 直觉设定奖励（read=0.3, state=0.1） |
| No Training (Base) | SFT 或基础 checkpoint |
| Frontier Models | GPT-4.1, GPT-4o, Claude Sonnet 4.5 等闭源模型 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 5）

| Model | Size | Base Pass% | Ours (MT-GRPO + IRC) | Δ |
|-------|------|------------|------------------------|----|
| Qwen3.5-4B | 4B | 63.8% | **66.7%** | **+2.9pp** |
| Qwen3-30B-A3B MoE | 30.5B | 58.0% | **69.5%** | **+11.5pp** |

> 💡 **亮点**：
> - **4B 模型超越 GPT-4.1 (49.4%) 和 GPT-4o (42.8%)**，尽管体积小约 **50倍**。
> - **30.5B MoE 接近 Claude Sonnet 4.5 (70.0%)**，差距仅 0.5pp。

---

### 🔬 消融实验结果（Table 6 & 7）

#### Qwen3.5-4B 消融路径（8个版本迭代）

| Version | Reward Design | Pass% | 发现 |
|--------|----------------|--------|------|
| Base | None | 63.8% | 强基线 |
| MT-GRPO (V3) | Sparse only | 64.6% | +0.8pp，但后期下降 |
| V5 | Naive dense (read=0.3) | 57.3% | **↓6.5pp！证明错误奖励有害** |
| V6 | IRC (read=0.0, state=-0.1) | 59.1→66.7% | 校准后恢复并超越 |
| V8 | IRC + deep_equal + prompt | 68.0% | 最终最佳配置 |

> ✅ 结论：IRC 是逆转性能退化的关键；`deep_equal` 和 prompt engineering 进一步提效。

#### MoE 模型结果（Table 7）
- Naive GRPO 和 Dense Rewards 均导致性能低于基线（58.0%）；
- MT-GRPO + Sparse → 68.0% (+10pp)；
- 加上 IRC → **69.5% (+11.5pp)**，达到当前最优。

---

### 🧪 定性分析（Task 9: Flight Cancellation with Flattery）

| 指标 | Base Model | Trained Model |
|------|------------|---------------|
| Turns | 56 | **28** (-50%) |
| Duration | 1,633s (~27min) | **568s (~9.5min)** (-65%) |
| Action Accuracy | 0/2 (0%) | **2/2 (100%)** |
| Tool Calls | 8+ | **4** |
| Behavior | 冗余推理、重复总结、忽略策略 | 直接调用、精准参数、抗操纵 |

> 🎯 成功原因：
> 1. **Action Grounding**：正确选择工具参数；
> 2. **Efficiency**：去除冗余对话；
> 3. **Manipulation Resistance**：无视用户奉承，坚持政策合规。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **密集奖励需谨慎设计**：  
   > “Intuitive rewards can catastrophically degrade performance.”  
   即使微小的 reward discriminativeness 与 advantage direction 不一致，也会导致训练失败。

2. **稀疏奖励“意外有效”**：  
   因其产生大量 “dead turns”（零梯度），反而实现了 **gradient focusing**，将学习集中在关键决策点上。

3. **IRC 可修复甚至超越稀疏奖励效果**：  
   经过判别性分析后的密集奖励不仅能避免退化，还能带来额外增益（+2.9 ~ +11.5pp）。

4. **GTPO Hybrid 消除 advantage mismatch**：  
   在 5,952 条 rollouts 中实现 **zero advantage misalignment**，相比标准 MT-GRPO 的 2 处错配更稳定。

5. **小模型也能超越大模型**：  
   4B 模型经 RL 训练后超过 GPT-4.1 和 GPT-4o，说明高效训练策略的重要性远超参数规模。

---

### ⚠️ 局限性
- **领域限制**：目前仅在 Tau-Bench 的 airline domain 上验证，虽有 cross-domain 尝试（retail 77.4%，telecom 32.0%），但迁移能力有限。
- **Simulator Distribution Shift**：训练用 DeepSeek-V3，评测用 GPT-4.1，可能影响结果一致性。
- **超参数依赖**：$\gamma=0.9$, $\lambda=0.3$ 在单一领域调优，通用性待验证。
- **人工 IRC 循环**：当前需手动分析 rollout 数据，尚未自动化。

---

### 🔮 未来工作方向
- **Automated IRC via EDG**（Empirical Discriminative Gating）：
  开发在线算法，周期性基于近期 rollout 自动计算 reward tier 权重，实现动态适应策略演化。
- 扩展至更多 domains（如医疗、金融）；
- 探索 multi-reward decomposition 与 GDPO 结合；
- 构建真实 human-in-the-loop 用户反馈 pipeline。

---

> 📢 **一句话总结**：  
> 本文揭示了 dense reward 设计中的“优势错配”陷阱，提出了 **IRC + MT-GRPO/GTPO hybrid** 方法，在 Tau-Bench 上实现了小模型超越大模型的突破，并发布了首个公开的 RL 训练 recipe 与分析工具。

</details>

---

### 16. [DSBD: Dual-Aligned Structural Basis Distillation for Graph Domain Adaptation](https://arxiv.org/abs/2604.03154)

**Authors**: Yingxu Wang, Kunyu Zhang, Jiaxin Huang, Mengzhu Wang, Mingyan Xiao, Siyang Gao, Nan Yin  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.03154v1  

#### Abstract
Graph domain adaptation (GDA) aims to transfer knowledge from a labeled source graph to an unlabeled target graph under distribution shifts. However, existing methods are largely feature-centric and overlook structural discrepancies, which become particularly detrimental under significant topology s...

---

### 17. [Reinforcement Learning-based Knowledge Distillation with LLM-as-a-Judge](https://arxiv.org/abs/2604.02621)

**Authors**: Yiyang Shen, Lifu Tu, Weiran Wang  
**Category**: cs.CL  
**Published**: 2026-04-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.02621v1  

#### Abstract
Reinforcement Learning (RL) has been shown to substantially improve the reasoning capability of small and large language models (LLMs), but existing approaches typically rely on verifiable rewards, hence ground truth labels. We propose an RL framework that uses rewards from an LLM that acts as a jud...

---

### 18. [TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing](https://arxiv.org/abs/2604.03143)

**Authors**: Zhuohang Bian, Feiyang Wu, Chengrui Zhang, Hangcheng Dong, Yun Liang, Youwei Zhuo  
**Category**: cs.DC  
**Published**: 2026-04-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.03143v1  

#### Abstract
Multi-agent LLM applications organize execution in synchronized rounds where a central scheduler gathers outputs from all agents and redistributes the combined context. This All-Gather communication pattern creates massive KV Cache redundancy, because every agent's prompt contains the same shared ou...

---

### 19. [Homophily-aware Supervised Contrastive Counterfactual Augmented Fair Graph Neural Network](https://arxiv.org/abs/2604.02342)

**Authors**: Mahdi Tavassoli Kejani, Fadi Dornaika, Charlotte Laclau, Jean-Michel Loubes  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.02342v1  

#### Abstract
In recent years, Graph Neural Networks (GNNs) have achieved remarkable success in tasks such as node classification, link prediction, and graph representation learning. However, they remain susceptible to biases that can arise not only from node attributes but also from the graph structure itself. A...

---

### 20. [Adaptive Semantic Communication for Wireless Image Transmission Leveraging Mixture-of-Experts Mechanism](https://arxiv.org/abs/2604.02691)

**Authors**: Haowen Wan, Qianqian Yang  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.02691v1  

#### Abstract
Deep learning based semantic communication has achieved significant progress in wireless image transmission, but most existing schemes rely on fixed models and thus lack robustness to diverse image contents and dynamic channel conditions. To improve adaptability, recent studies have developed adapti...

---

### 21. [Structure-Aware Commitment Reduction for Network-Constrained Unit Commitment with Solver-Preserving Guarantees](https://arxiv.org/abs/2604.02788)

**Authors**: Guangwen Wang, Jiaqi Wu, Yang Weng, Baosen Zhang  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.02788v1  

#### Abstract
The growing number of individual generating units, hybrid resources, and security constraints has significantly increased the computational burden of network-constrained unit commitment (UC), where most solution time is spent exploring branch-and-bound trees over unit-hour binary variables. To reduc...

---

### 22. [Toward an Operational GNN-Based Multimesh Surrogate for Fast Flood Forecasting](https://arxiv.org/abs/2604.02876)

**Authors**: Valentin Mercier (Toulouse INP, IRIT, EPE UT), Serge Gratton (IRIT, EPE UT, Toulouse INP), Lapeyre Corentin (NVIDIA), Gwena\"el Chevallet  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.02876v1  

#### Abstract
Operational flood forecasting still relies on high-fidelity two-dimensional hydraulic solvers, but their runtime can be prohibitive for rapid decision support on large urban floodplains. In parallel, AI-based surrogate models have shown strong potential in several areas of computational physics for ...

---

### 23. [CharTool: Tool-Integrated Visual Reasoning for Chart Understanding](https://arxiv.org/abs/2604.02794)

**Authors**: Situo Zhang, Yifan Zhang, Zichen Zhu, Da Ma, Lei Pan, Danyang Zhang, Zihan Zhao, Lu Chen, Kai Yu  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.02794v1  

#### Abstract
Charts are ubiquitous in scientific and financial literature for presenting structured data. However, chart reasoning remains challenging for multimodal large language models (MLLMs) due to the lack of high-quality training data, as well as the need for fine-grained visual grounding and precise nume...

---

### 24. [Agentic-MME: What Agentic Capability Really Brings to Multimodal Intelligence?](https://arxiv.org/abs/2604.03016)

**Authors**: Qianshan Wei, Yishan Yang, Siyi Wang, Jinglin Chen, Binyu Wang, Jiaming Wang, Shuang Chen, Zechen Li, Yang Shi, Yuqi Tang, Weining Wang, Yi Yu, Chaoyou Fu, Qi Li, Yi-Fan Zhang  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.03016v1  

#### Abstract
Multimodal Large Language Models (MLLMs) are evolving from passive observers into active agents, solving problems through Visual Expansion (invoking visual tools) and Knowledge Expansion (open-web search). However, existing evaluations fall short: they lack flexible tool integration, test visual and...

---

### 25. [An Empirical Study of Many-Shot In-Context Learning for Machine Translation of Low-Resource Languages](https://arxiv.org/abs/2604.02596)

**Authors**: Yinhan Lu, Gaganpreet Jhajj, Chen Zhang, Anietie Andy, David Ifeoluwa Adelani  
**Category**: cs.CL  
**Published**: 2026-04-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.02596v1  

#### Abstract
In-context learning (ICL) allows large language models (LLMs) to adapt to new tasks from a few examples, making it promising for languages underrepresented in pre-training. Recent work on many-shot ICL suggests that modern LLMs can further benefit from larger ICL examples enabled by their long conte...

---

### 26. [StoryScope: Investigating idiosyncrasies in AI fiction](https://arxiv.org/abs/2604.03136)

**Authors**: Jenna Russell, Rishanth Rajendhran, Mohit Iyyer, John Wieting  
**Category**: cs.CL  
**Published**: 2026-04-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.03136v1  

#### Abstract
As AI-generated fiction becomes increasingly prevalent, questions of authorship and originality are becoming central to how written work is evaluated. While most existing work in this space focuses on identifying surface-level signatures of AI writing, we ask instead whether AI-generated stories can...

---

### 27. [Accelerating Nonlinear Time-History Analysis with Complex Constitutive Laws via Heterogeneous Memory Management: From 3D Seismic Simulation to Neural Network Training](https://arxiv.org/abs/2604.02755)

**Authors**: Tsuyoshi Ichimura, Kohei Fujita, Hideaki Ito, Muneo Hori, Lalith Maddegedara  
**Category**: cs.DC  
**Published**: 2026-04-06  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.02755v1  

#### Abstract
Nonlinear time-history evolution problems employing high-fidelity physical models are essential in numerous scientific domains. However, these problems face a critical dual bottleneck: the immense computational cost of time-stepping and the massive memory requirements for maintaining a vast array of...

---

### 28. [Interpretable Deep Reinforcement Learning for Element-level Bridge Life-cycle Optimization](https://arxiv.org/abs/2604.02528)

**Authors**: Seyyed Amirhossein Moayyedi, David Y. Yang  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.02528v1  

#### Abstract
The new Specifications for the National Bridge Inventory (SNBI), in effect from 2022, emphasize the use of element-level condition states (CS) for risk-based bridge management. Instead of a general component rating, element-level condition data use an array of relative CS quantities (i.e., CS propor...

---

### 29. [EMS: Multi-Agent Voting via Efficient Majority-then-Stopping](https://arxiv.org/abs/2604.02863)

**Authors**: Yiqing Liu, Hantao Yao, Wu Liu, Yongdong Zhang  
**Category**: cs.AI  
**Published**: 2026-04-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.02863v1  

#### Abstract
Majority voting is the standard for aggregating multi-agent responses into a final decision. However, traditional methods typically require all agents to complete their reasoning before aggregation begins, leading to significant computational overhead, as many responses become redundant once a major...

---

### 30. [SocioEval: A Template-Based Framework for Evaluating Socioeconomic Status Bias in Foundation Models](https://arxiv.org/abs/2604.02660)

**Authors**: Divyanshu Kumar, Ishita Gupta, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi  
**Category**: cs.CL  
**Published**: 2026-04-06  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.02660v1  

#### Abstract
As Large Language Models (LLMs) increasingly power decision-making systems across critical domains, understanding and mitigating their biases becomes essential for responsible AI deployment. Although bias assessment frameworks have proliferated for attributes such as race and gender, socioeconomic s...

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
