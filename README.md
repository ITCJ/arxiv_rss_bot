# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-14 07:16:51 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding](https://arxiv.org/abs/2604.10152)

**Authors**: Jehyeon Bang, Eunyeong Cho, Ranggi Hwang, Jinha Chung, Minsoo Rhu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2604.10152v1  

#### Abstract
The Mixture-of-Experts (MoE) architecture has emerged as a promising approach to mitigate the rising computational costs of large language models (LLMs) by selectively activating parameters. However, its high memory requirements and sub-optimal parameter efficiency pose significant challenges for ef...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Mixture-of-Experts (MoE) 架构虽然通过稀疏激活降低了计算开销，但其巨大的内存需求导致在实际部署中面临严重挑战。主流的 **CPU-offloading** 方法将非活跃专家参数卸载到 CPU DRAM 中以节省 GPU 显存，但由于 PCIe 带宽有限，频繁地从 CPU 向 GPU 迁移专家参数会引入极高的通信延迟，尤其是在 **大 batch 推理场景下**，该瓶颈尤为突出。

现有方法如 **MoE-Overlap**（重叠迁移与计算）和 **MoE-Caching**（缓存热点专家）仅能在小 batch 或特定条件下缓解问题，无法从根本上解决通信带宽受限的问题。

### 提出了什么新方法或新思路
本文提出 **SpecMoE**，一种基于 **Self-Assisted Speculative Decoding** 的高效 MoE 推理系统，其核心思想是：

- **算法层面**：提出 **self-assisted speculative decoding** 算法，利用目标 MoE 模型自身的一部分（非专家层 + 少量“热点”专家）作为 **draft model**，无需额外训练或微调。
- **系统层面**：在 speculation 阶段完全避免 CPU-to-GPU 通信；所有专家迁移被合并到 verification 阶段进行批处理，显著减少冗余传输。

### 相比现有方法的优势
- **无需额外模型训练**：相比需要独立 draft model 的 speculative decoding 方法（如 EAGLE、Medusa），SpecMoE 完全免训练，节省开发成本。
- **显著降低通信开销**：通过 speculative 批处理机制，大幅减少 PCIe 数据传输量。
- **适用于大 batch 场景**：特别优化了 batched inference 性能，在大 batch 下优势更明显。
- **通用性强**：不依赖特定架构修改，可应用于任意 MoE 模型。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **NLLB-MoE**：在 **WMT-14 英法翻译任务** 上评估。
- **Mixtral-8x7B 和 Llama-4-Scout**：在 **CNN-DailyMail 文本摘要任务** 上评估。

### 实验设置
- **硬件平台**：
  - 单块 **NVIDIA H100 GPU**（96GB HBM3）
  - 双路 **Intel Xeon Platinum 8558 CPU**（共 1TB DDR5）
  - 通过 **PCIe 5.0** 互联（单向带宽 64 GB/s）
- **软件环境**：
  - PyTorch 2.5.0 + CUDA 12.6
  - Hugging Face Transformers 4.51.0 + Accelerate 1.6.0
  - 自定义实现支持动态 draft model 切换

### 评估指标
- **End-to-end inference latency**（端到端推理延迟）
- **Throughput (tokens/sec)**（吞吐量）
- **CPU-to-GPU data transfer size**（CPU-GPU 数据传输量）
- **Tokens generated per step $T(\gamma)$**（每步生成的 token 数，衡量 speculative 效率）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **MoE-OnDemand** | 每个 token 步骤按需从 CPU 加载所需专家 |
| **MoE-Overlap** | 尝试重叠专家迁移与计算（理想化 oracle 版本） |
| **MoE-Caching** | 缓存访问频率最高的前 10% 专家在 GPU 上 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **batch size = 256** 下，SpecMoE 相比 MoE-OnDemand 实现最高 **4.30× 的吞吐提升**。
- CPU-to-GPU 数据传输量减少高达 **76.73%**（vs. MoE-OnDemand/MoE-Overlap），相比 MoE-Caching 也减少了 **71.89%**。
- 即使在低专家热度（low expert hotness）模型上，仍取得显著收益：
  - **Mixtral-8x7B**：最高 **2.17× 吞吐提升**
  - **Llama-4-Scout**：最高 **1.42× 吞吐提升**

### 与基线方法的对比结果
| 指标 | SpecMoE vs. MoE-OnDemand | vs. MoE-Overlap | vs. MoE-Caching |
|------|--------------------------|------------------|------------------|
| 最大吞吐提升 | **4.30×** | >4.30× | >3.70× |
| 数据传输减少 | **76.73%** | 76.73% | **71.89%** |
| 大 batch 可扩展性 | 显著优于 | 明显优于 | 明显优于 |
| GPU 内存占用（用于缓存） | 仅缓存 **4 个专家/block** | 不适用 | 缓存 **13 个专家/block**（10%） |

> 注：SpecMoE 虽然在 batch=1 时略逊于 MoE-Caching，但在实际主流的大 batch 场景下全面超越。

### 消融实验结果
#### （1）专家选择策略对比（Table II）
| 策略 | Tokens/step ($T(\gamma)$) | 相对加速比 |
|------|----------------------------|-------------|
| Random | 6.812 | 1.000 |
| Hot-Global（静态全局热点） | 7.220 | 1.084 |
| **Hot-Temporal（动态时序热点）** | **7.265** | **1.143** |

✅ 结论：**动态感知时序局部性的专家替换策略效果最佳**，比静态策略多提升约 6%。

#### （2）draft 专家数量影响（Table III）
| Draft Experts (N) | Tokens/step | Throughput (tokens/sec) |
|--------------------|--------------|---------------------------|
| 2 | 6.929 | 116.40 |
| **4** | **7.265** | **122.97** ✅ |
| 8 | 7.620 | 121.31 |
| 16 | 7.819 | 117.16 |

✅ 结论：存在最优值（N=4），过多专家反而因计算延迟增加而降低整体吞吐。

#### （3）affinity-based 路由机制有效性（Figure 13）
- 使用预计算的 **affinity table**（基于 L2 距离）替代未命中专家后，平均每步生成 token 数比随机选择高 **7.23%**。
- 表明该机制能有效逼近原模型行为，维持高质量 draft 输出。

---

## 4. 关键结论和发现

### 主要发现
1. **Speculative decoding 可有效应用于 MoE 推理**，且无需额外 draft model —— SpecMoE 是首个实现这一点的工作。
2. **self-assisted design** 充分利用了 MoE 模型内部的专家激活热区特性，构建轻量级、高质量的 draft model。
3. **大 batch 推理下通信成为主导瓶颈**，而 SpecMoE 通过合并迁移请求、消除冗余传输，从根本上缓解了这一问题。
4. 即使在专家分布较均匀的模型（如 Mixtral-8x7B）上，SpecMoE 依然表现稳健，说明其设计具有良好的泛化能力。
5. 在 **SSD-offloading** 场景下（Figure 16），SpecMoE 仍能实现平均 **2.25× 吞吐提升**，远超其他基线（<1.3×），证明其在高延迟存储场景下的潜力。

### 方法的局限性
- 当 batch size 极小时（如=1），可能不如精心缓存的 MoE-Caching。
- 依赖一定程度的专家激活局部性（temporal locality），极端随机路由模式可能削弱性能。
- 当前实现假设专家大小一致，对异构 MoE 模型的支持需进一步验证。

### 未来工作方向
- 扩展至 **异构 MoE 架构** 和 **动态路由策略**。
- 探索在 **多 GPU + 多节点分布式环境** 下的应用。
- 结合 **KV Cache offloading** 进一步优化内存效率。
- 研究如何自适应调整 $\gamma$（draft token 数量）和 $N$（draft 专家数）以实现动态最优化。

---

> ✅ **总体评价**：  
> SpecMoE 是一个简洁而高效的系统级创新，巧妙结合了 **speculative decoding** 与 **MoE 内部结构特性**，在不增加训练负担的前提下，显著提升了 CPU-offloaded MoE 推理系统的性能与可扩展性，尤其适合现代大规模、批处理式的 LLM serving 场景。

</details>

---

### 2. [Tessera: Unlocking Heterogeneous GPUs through Kernel-Granularity Disaggregation](https://arxiv.org/abs/2604.10180)

**Authors**: Tiancheng Hu, Jin Qin, Zheng Wang, Junhao Hu, Yuzheng Wang, Lei Chen, Yizhou Shan, Mingxing Zhang, Ting Cao, Chunwei Xia, Huimin Cui, Tao Xie, Chenxi Wang  
**Category**: cs.DC  
**Published**: 2026-04-14  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.10180v1  

#### Abstract
Disaggregation maps parts of an AI workload to different types of GPUs, offering a path to utilize modern heterogeneous GPU clusters. However, existing solutions operate at a coarse granularity and are tightly coupled to specific model architectures, leaving much room for performance improvement. Th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Tessera: Unlocking Heterogeneous GPUs through Kernel-Granularity Disaggregation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现代数据中心的 GPU 集群日益**异构化**（heterogeneous），包含不同型号、代际和性能特征的 GPU（如 A100、H100、L40s、RTX Pro 6000 等）。然而，现有的 AI 推理调度方法在利用这种硬件多样性方面存在严重不足：

- **粒度过粗**：现有 disaggregation 方法（如 Prefill-Decode 或 Attention-FFN）以**执行阶段**（phase）或**计算块**（block）为单位进行任务划分，无法捕捉更细粒度的性能差异。
- **模型耦合性强**：这些方法依赖特定模型架构（如 Transformer 的 prefill/decode 分离），难以推广到非标准架构（如 Mamba、Diffusion Models）。

这导致大量性能和成本效率潜力未被挖掘。

---

### **提出的新方法与核心思路**
论文提出了 **Tessera**，首个基于 **kernel-granularity disaggregation**（内核级拆分）的异构 GPU 推理系统。

#### **核心洞察（Key Insight）**
- **应用异构性存在于 kernel 层面**：即使在同一 phase 或 block 内，不同 kernel 对计算、内存带宽的需求差异巨大（例如 `cublasGemv` 是 memory-bound，而 `FlashAttention` 是 compute-bound）。
- **kernel 是最合适的对齐粒度**：将计算任务按 kernel 拆分并映射到最适合其特性的 GPU 上，能最大化性能与成本效率。

#### **Tessera 的三大设计组件**
| 组件 | 功能 |
|------|------|
| **Kernel Analyzer** | 在 PTX 层面对 kernel 进行静态分析，提取精确的内存访问边界和 **Read-After-Write (RAW)** 数据依赖关系，确保跨 GPU 执行的正确性。 |
| **Policy Planner** | 基于分析结果，构建混合整数线性规划（MILP）模型，生成最优调度策略：<br>- 吞吐量优先（throughput-oriented）<br>- 延迟优先（latency-oriented） |
| **GPU Worker + Online Monitor** | 运行时执行调度策略，采用 pipelined request processing 和 priority-aware stream scheduling 来重叠通信与计算；在线监控器动态切换策略以适应负载变化。 |

---

### **相比现有方法的优势**
| 维度 | Tessera | 现有方法（PD/AF Disaggregation） |
|------|--------|-------------------------------|
| **粒度** | Kernel-level | Phase/block-level |
| **通用性** | Model-agnostic，适用于 LLM、MLLM、SSM、Diffusion 等多种架构 | 仅适用于特定架构（如 Transformer） |
| **性能潜力** | 充分挖掘 kernel 级异构性 | 忽略 kernel 内部差异，造成“错配” |
| **灵活性** | 支持动态策略切换（throughput ↔ latency） | 多为静态策略 |

---

## **2. 核心实验方法和设置**

### **使用的模型与工作负载**
覆盖四类主流 AI 模型，验证泛化能力：
1. **LLMs**: Llama-3 8B, GPT-oss 20B
2. **SSMs**: Mamba-Codestral 7B（替代 attention 的序列建模）
3. **MLLMs**: Qwen2.5-VL 7B（图文多模态）
4. **Diffusion Models**: Stable Diffusion 3.5（图像生成）

输入数据来源：
- Splitwise conversation dataset（LLM）
- COCO captioning（MLLM）
- PartiPrompts（Diffusion）

---

### **实验设置**
#### **硬件平台**
- **单节点双卡配置**：
  - A100 + L40s
  - H100 + RTX Pro 6000
  - B200 + H100
- **集群扩展测试**：
  - 2×A100 + 1×L40s
  - 8×B200 + 8×H100（用于 Qwen-3 235B）

所有 GPU 间通过 200/400 Gbps RDMA NIC 互联。

#### **评估指标**
| 指标 | 定义 |
|------|------|
| **Throughput** | tokens/s（LLM/SSM/MLLM）、images/min（Diffusion） |
| **Latency** | 平均端到端请求延迟（normalized per token） |
| **Cost Efficiency (Perf/$)** | 吞吐量 / 总 GPU 租赁成本（归一化） |
| **SLO Compliance** | 在给定延迟约束下可服务的最大请求率 |

---

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **Homogeneous (左/右 GPU)** | 单一类型 GPU 推理，无 disaggregation |
| **Prefill-Decode Disaggregation (PD Dis.)** | 将 prefill 放高算力 GPU，decode 放高带宽 GPU（如 DistServe） |
| **Attention-FFN Disaggregation (AF Dis.)** | 将 attention 和 FFN 分别放不同 GPU（如 MegaScale-Infer） |

> ⚠️ 注意：PD 不适用于 Diffusion（无 prefill/decode 划分），AF 不适用于 Mamba 和 Diffusion（无 attention/FFN 结构）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **离线吞吐量提升（Offline Throughput）**
- **平均提升 1.5× ~ 2.3×** 超过 PD Disaggregation
- **平均提升 1.35× ~ 1.7×** 超过 AF Disaggregation
- 在 GPT-oss 20B 上达到最高 **2.3×** 吞吐增益

> 图 6 显示，在所有五种 workload 和三种 GPU 组合中，Tessera 均显著领先。

#### ✅ **成本效率（Cost Efficiency, Perf/$）**
| GPU Pair | Tessera 提升（vs PD） | Tessera 提升（vs AF） |
|---------|---------------------|---------------------|
| A100+L40s | **1.5×** | **1.4×** |
| H100+RTX Pro 6000 | **1.6×** | **1.5×** |
| B200+H100 | **1.01×** | **1.01×** |

> 表 III 显示，Tessera 成本效率全面优于基线，甚至在某些组合下超过单个高端 GPU（如 H100 alone）。

> 🔥 **惊人发现**：一个异构 GPU 对（如 H100 + RTX Pro 6000）在 Tessera 下的**吞吐量超过两个 H100**，且成本更低！

#### ✅ **在线延迟表现（Online Latency）**
- 在低负载下，Tessera 使用 latency-oriented policy，实现 **1.3× 更低延迟**（vs PD）和 **1.2×**（vs AF）
- 在 SLO = 50ms/token 下，Tessera 可承载 **1.3× 更高的请求速率**而不违反约束

#### ✅ **集群规模扩展性**
- 在 2×A100 + 1×L40s 和 8×B200 + 8×H100 场景下：
  - 吞吐量仍比 PD 提高 **1.5×**
  - 比 AF 提高 **1.4×**
- 证明 Tessera 可自然组合（compose）with **Tensor Parallelism (TP)**，无需修改通信拓扑。

---

### **消融实验与敏感性分析**

#### 🔍 **Pipelined Request Processing 效果**
- 无流水线 → 吞吐仅为理论最优的 ~50%
- 加入流水线 → 提升 1.47×
- 加入 priority-aware scheduling → 达到 **96.6% 最优吞吐**
- GPU 利用率从 <60% 提升至 >95%

#### 🔍 **Online Monitor 参数敏感性**
- 窗口大小 $ W = 300ms $，阈值 $ \beta = 1.5 $ 时取得最佳平衡
- 设置过于激进（$ W=30ms $）会导致频繁策略切换，增加 32% 延迟
- 设置过于保守（$ \beta=3.0 $）会延迟响应，导致峰值延迟上升 55%

#### 🔍 **网络带宽鲁棒性**
- 即使将 RDMA 带宽从 200Gbps 降至 25Gbps：
  - 离线吞吐仅下降 <6%（得益于流水线隐藏通信开销）
  - 在线延迟仍低于纯 A100 基线
- 极端情况下自动退化为单 GPU 执行，无性能断崖

#### 🔍 **MILP 求解时间**
- 对含 1000 kernels 的 DDG，求解耗时约 0.43s
- 对超大规模模型（如 DeepSeek-V3 671B，~1500 kernels）可在 1s 内完成
- 支持离线预计算，不影响运行时性能

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Kernel-level heterogeneity 是真实且可观测的**：
   - 多达 67% 的 kernel 在 L40s 上快于 A100
   - 平均有 36% 的总执行时间由“更适合便宜 GPU”的 kernel 构成

2. ✅ **Coarse-grained disaggregation 存在根本性局限**：
   - Phase/block 级拆分强制将异构 kernel 放在同一 GPU，造成资源浪费

3. ✅ **Fine-grained kernel disaggregation 是可行且高效的**：
   - 通过 PTX 分析可准确提取依赖
   - 通过 pipelining 可有效掩盖通信开销
   - 通过 MILP 可实现全局优化

4. ✅ **异构 ≠ 劣势，而是优化机会**：
   - 正确调度下，**廉价 GPU + 高端 GPU 的组合可以超越两个高端 GPU**
   - 成本效率提升高达 **1.6×**

---

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **不支持间接内存访问** | 若 kernel 通过 global variable 访问 buffer，Analyzer 无法获取地址（但实测中此类 kernel 极少） |
| **依赖 CUDA Graph** | 当前实现假设执行路径静态，需框架支持 CUDA Graph；否则需预处理录制 trace |
| **MILP 可扩展性** | 虽然当前规模可接受，但在超大规模（>10 GPUs）时可能需要近似算法 |
| **跨厂商异构性未支持** | 当前聚焦 NVIDIA 生态，未处理 AMD/NPU 等跨平台挑战（因软件栈碎片化） |

---

### **未来工作方向**
1. **支持动态 shape 和 speculative execution**：增强对运行时动态行为的适应能力
2. **引入学习-based scheduler**：替代 MILP，加速调度决策
3. **扩展至 CPU-GPU 或 NPU-GPU 协同 disaggregation**
4. **支持更多编程模型**（Beyond PyTorch/vLLM）
5. **探索 disaggregation 与 quantization、sparsity 的联合优化**

---

## **总结**
Tessera 是首个将 disaggregation 推向 **kernel granularity** 的系统，揭示了现有 coarse-grained 方法的巨大性能缺口。它通过 **PTX-level dependency analysis + MILP scheduling + pipelined execution**，实现了对异构 GPU 的高效利用，在吞吐、延迟、成本效率上全面超越现有方案，并具备良好的通用性和扩展性。该工作表明：**GPU 异构性不应被视为负担，而应作为系统优化的核心驱动力**。

</details>

---

### 3. [Leveraging Mathematical Reasoning of LLMs for Efficient GPU Thread Mapping](https://arxiv.org/abs/2604.10387)

**Authors**: Jose Maureira, Crist\'obal A. Navarro, Hector Ferrada, Luis Veas-Castillo  
**Category**: cs.DC  
**Published**: 2026-04-14  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.10387v1  

#### Abstract
Mapping parallel threads onto non-box-shaped domains is a known challenge in GPU computing that, if done efficiently, can prevent severe performance penalties from allocating unnecessary computational resources. Currently, achieving this optimal efficiency requires significant analytical human time ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Leveraging Mathematical Reasoning of LLMs for Efficient GPU Thread Mapping*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在 **GPU computing** 中，将并行线程高效映射到非规则几何域（如三角形、四面体、分形等）是一个长期挑战。传统的 **Bounding Box (BB)** 映射方式会分配大量无效线程块，造成严重的计算资源浪费（idle threads），导致执行时间延长和能耗增加。

手动为每个特定几何形状推导精确的 `O(1)` 或 `O(log N)` 映射函数需要大量数学分析工作，效率低下且难以扩展。

---

### 🚀 提出的新方法与创新思路
本文提出一种全新的自动化框架，利用 **Large Language Models (LLMs)** 的符号推理能力（symbolic reasoning），通过 **in-context learning** 自动推导适用于复杂几何域的 GPU 线程映射函数。

#### 核心创新点：
- **首次将 LLMs 应用于 GPU thread mapping 的数学公式自动推导**，将其视为“算法归纳”（algorithm induction）任务而非传统数值拟合。
- 利用 **open-weights LLMs** 在本地完成整个推理过程，实现完全自主可控（sovereign）的科研流程。
- 构建了一个四阶段自动化流水线：
  1. **Context Sampling**：从目标几何域采样前 N 个坐标点；
  2. **Symbolic Inference**：LLM 基于 few-shot 示例进行模式识别与逻辑反向工程；
  3. **Algorithmic Synthesis**：生成可直接部署的解析代码（analytical code）；
  4. **Integration & Deployment**：集成至 CUDA/HIP 内核中优化线程调度。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | 本论文优势 |
|------|--------|------------|
| **Manual Derivation** | 耗时耗力，依赖专家知识 | 完全自动化，无需人工干预 |
| **Traditional Symbolic Regression (SR)** | 本质是连续数值拟合，无法保证离散整数精度 | LLMs 可进行精确的离散逻辑推理，输出绝对正确的索引函数 |
| **Naive Bounding Box (BB)** | 大量线程浪费，能效极低 | 推导出的映射函数消除所有 block waste |

> ✅ **关键洞察**：虽然 LLM 推理阶段有较高的能量开销（尤其对 reasoning-focused 模型如 DeepSeek-R1），但这是一次性投资（one-time upfront cost），而生成的映射函数可在后续无数次 GPU 执行中带来巨大节能收益。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集 / 几何域
共测试六种具有递增复杂度的 **computational domains**，涵盖密集结构与分形结构：

| Domain | 类型 | Ground Truth Complexity |
|--------|------|--------------------------|
| 2D Triangular | Dense | `O(1)`（基于三角数逆运算） |
| 3D Pyramid | Dense | `O(1)`（立方根相关表达式） |
| 2D Sierpinski Gasket | Fractal | `O(log₃N)` |
| 2D Sierpinski Carpet | Fractal | `O(log₈N)` |
| 3D Sierpinski Pyramid | Fractal | `O(log₄N)` |
| **3D Menger Sponge** | Fractal | `O(log₂₀N)`（最具挑战性） |

所有 ground truth 映射函数均来自已有文献（见 Table I）。

---

### ⚙️ 实验设置

#### 模型选择（全部为 open-weights LLMs）：
- 包括：`DeepSeek-R1`, `Gemma-3`, `Llama 3.3/4`, `Mistral-Nemo`, `Nemotron`, `Qwen3`, `GPT-OSS`
- 参数范围：12B ~ 235B
- 运行环境：GGUF 格式，本地运行于 **Patagón supercomputer** 上的 DGX 节点（4×NVIDIA A100 40GB）

#### In-Context Learning 设置（模拟真实稀疏数据场景）：
- **Stage 20**：提供前 20 个坐标作为上下文
- **Stage 50**：提供前 50 个
- **Stage 100**：提供前 100 个

#### 验证方式：
- 对生成的 Python 函数，在 **N = 1,000,000** 规模下验证其输出是否与 ground truth 完全一致。

---

### 🎯 评估指标

| 指标 | 描述 |
|------|------|
| **Ordered Accuracy** | 输出序列与 ground truth 完全按序匹配的比例（严格标准） |
| **Any-order Accuracy** | 是否覆盖所有正确坐标（允许顺序不同，“Silver Standard”） |
| **Big-O Efficiency** | 生成代码的时间复杂度是否最优（静态 + 动态分析） |
| **Execution Time (ms)** | 在真实 CUDA kernel 中的运行时间 |
| **Energy Consumption (J)** | GPU 实际能耗（含 idle power） |
| **Points/Joule** | 推理阶段的能量效率（衡量 LLM 自身能耗代价） |

---

### 🔁 基线方法对比
| 基线 | 描述 |
|-----|------|
| **Bounding Box (BB)** | 最朴素策略，使用包围盒分配线程，大量浪费 |
| **Paper (Human-derived)** | 来自文献的手工推导最优解，作为黄金标准 |
| **Traditional SR Methods** | 如 Neural Symbolic Regression、SymFormer 等（文中指出其失败原因） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ 成功案例（Top-performing LLMs 表现优异）
| Domain | Best Model(s) | Ordered Acc. | Complexity |
|-------|---------------|-------------|-----------|
| 2D Triangular | `OSS:120b`, `R1:70b`, `Qw3:32b` | 100% @ all stages | `O(1)` |
| 2D Sierpinski Gasket | `OSS:120b` | 100% @ Stage 50/100 | `O(log N)` |
| 2D Sierpinski Carpet | `OSS:120b` (Stage 100), `Qw3:235b` (Stage 20/50) | 100% | `O(log N)` |
| 3D Triangular | `OSS:120b`, `Qw3:32b` | 100% | `O(1)` |
| 3D Sierpinski Pyramid | `OSS:120b` | 100% @ Stage 20/100 | `O(log N)` |

> 💡 特别注意：`R1:70b` 在 3D Triangular 上虽仅得 0.11% Ordered 准确率（Stage 20），但 Any-order 达 82.7%，说明它已“理解”几何结构，只是排序未对齐。

---

#### ❌ 当前极限：“Menger Limit”
| Domain | 结果 | 分析 |
|-------|------|------|
| **3D Menger Sponge** | 所有模型 Any-order < 1%, Ordered ≈ 0% | 即使最大模型也无法捕捉其复杂的三维递归空洞移除机制 |

> ➤ 此构成当前 open-weight LLM 的“**reasoning ceiling**”，称为 **Menger Limit**，成为未来模型发展的基准测试。

---

#### ⚡ 性能与能效提升（实际 GPU 执行阶段）

##### ▶ 密集几何（3D Pyramid）
| 方法 | Time (ms) | Energy (J) | Wasted Blocks |
|------|-----------|------------|----------------|
| Bounding Box | 2530.65 | 282.67 | ~83% |
| Optimal (Paper) | 3.84 | 0.92 | 0 |
| LLM-inferred (e.g., OSS:120b) | 3.84–29.31 | 0.92–5.99 | 0 |

> ✅ 消除浪费后，速度提升约 **650×**，能耗降低 **~300×**

##### ▶ 分形几何（3D Sierpinski）
| 方法 | Time (ms) | Energy (J) | Wasted Blocks |
|------|-----------|------------|----------------|
| Bounding Box (projected) | ~15,949 | ~1,591 | >99.9% |
| Optimal / LLM-inferred | **3.30** | **0.55** | 0 |

> ✅ **有效加速达 4833×，节能高达 2890×**  
> 💬 “一次推理成本即被完全摊销”

---

#### 🔋 消融实验：LLM 推理阶段的能耗代价

| 因素 | 影响 |
|------|------|
| **模型大小** | 更大参数模型（如 `qwen3:235b`）因内存带宽压力导致更低的 **Points/Joule** 效率 |
| **Reasoning Mechanism** | `DeepSeek-R1` 使用 Chain-of-Thought，推理时间长，单位能耗处理点更少（高 energy penalty） |
| **上下文长度** | 增加 in-context 示例（20→100）通常提高效率（减少 hallucination 和编译错误） |

> ⚠️ 尽管某些模型推理效率较低，但只要最终生成正确映射函数，其长期收益远超初期投入。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **LLMs 可胜任精确的 symbolic reasoning 任务**  
   现代 open-weights LLMs 能成功推导出复杂 2D/3D 密集域和分形域的精确 `O(1)` 或 `O(log N)` 映射函数，**显著优于传统 symbolic regression 方法**（后者无法满足离散精确性要求）。

2. **存在明显的“能量权衡”（Energy Trade-off）**  
   - **前期**：LLM 推理（尤其 reasoning-heavy 模型）能耗较高；
   - **后期**：生成的解析内核在 GPU 上执行时几乎零浪费，带来数量级的 **time 和 energy 节省**（最高达 4833× 加速，2890× 节能）。

3. **发现了当前 open-weight 生态系统的“Menger Limit”**  
   所有当前开源模型在 **3D Menger Sponge** 上均表现极差（Any-order < 1%），揭示了其在高度递归三维结构上的推理瓶颈，为下一代模型提供了明确 benchmark。

4. **并非越大越好：推理机制比参数规模更重要**  
   - `DeepSeek-R1:70b`（70B）在部分任务上优于更大模型；
   - `Qwen3:235b` 虽然参数最多，但在多个任务上未胜出；
   - 表明 **Chain-of-Thought 等 reasoning 机制** 是突破复杂问题的关键。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **仅适用于确定性数学模式** | 不适用于无规律的 unstructured mesh 或任意点云数据 |
| **依赖高质量 few-shot prompt 设计** | 当前 success 依赖精心构造的 prompt（见 Appendix A） |
| **初始推理能耗高** | 尤其对 CoT 类模型，需高性能 GPU 支持 |
| **尚未突破 3D 递归分形极限** | Menger Sponge 仍是不可逾越的障碍 |

---

### 🔮 未来工作方向

1. **轻量化 fine-tuned 小模型**  
   针对 discrete spatial reasoning 任务微调小型 LLMs（<10B），以降低推理能耗。

2. **拓展至异构 HPC 拓扑**  
   将框架推广到 adaptive refinement grids、unstructured meshes 等更广泛场景。

3. **监控 open-weight 模型进展**  
   持续评估新一代 MoE、RL-enhanced 模型能否突破 **Menger Limit**。

4. **构建自动化编译流水线**  
   实现从几何描述 → LLM 推导 → CUDA kernel 自动生成 → 性能验证的全流程闭环系统。

---

> 🔗 **代码与数据公开**：  
> GitHub 仓库：[https://github.com/aspiadevs/llm-gpu-thread-mapping](https://github.com/aspiadevs/llm-gpu-thread-mapping)  
> 确保研究完全可复现（fully reproducible）。

</details>

---

### 4. [Frugal Knowledge Graph Construction with Local LLMs: A Zero-Shot Pipeline, Self-Consistency and Wisdom of Artificial Crowds](https://arxiv.org/abs/2604.11104)

**Authors**: Pierre Jourlin (LIA)  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.11104v1  

#### Abstract
This paper presents an empirical study of a multi-model zero-shot pipeline for knowledge graph construction and exploitation, executed entirely through local inference on consumer-grade hardware. We propose a reproducible evaluation framework integrating two external benchmarks (DocRED, HotpotQA), W...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Frugal Knowledge Graph Construction with Local LLMs: A Zero-Shot Pipeline, Self-Consistency and Wisdom of Artificial Crowds*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文致力于解决**在消费级硬件上以零训练、本地推理的方式构建和利用高质量知识图谱（Knowledge Graph, KG）** 的挑战。传统方法依赖大规模监督训练和昂贵的云计算资源，而本文探索如何通过量化后的本地 LLM（Local LLMs）实现“节俭 AI”（frugal AI），即低成本、低能耗、高可复现的知识图谱构建与问答系统。

### 提出的新方法与创新思路
1. **SYNSYNTH 多模型零样本 pipeline**  
   设计了一个端到端自动化 pipeline，将四项任务（关系抽取、文本转查询、多跳推理、对话式 RAG）分配给不同的量化 LLM，并通过 Ollama 实现本地 JSON Schema 约束解码，确保输出格式正确。

2. **基于提示工程（prompt engineering）和同义词匹配的关系抽取优化**  
   引入结构化 prompt（列出全部 96 种有效 relation）、禁止 `no_relation` 输出、按语义类别提供规则指导，并结合手动构建的 **relation synonym dictionary** 进行软匹配（soft matching），显著提升 F1 分数。

3. **自一致性（self-consistency）与跨模型多样性机制**  
   探索了两种多样性策略：
   - **Stochastic diversity**：同一模型多次采样（T=0.7）后进行多数投票。
   - **Architectural diversity**：不同架构模型之间的组合或级联。

4. **信心路由级联机制（confidence-routing cascade）**  
   创新性地提出一种动态级联机制：当主模型（Phi-4）内部响应一致性较低时，自动将问题重定向至更强的第二模型（GPT-OSS）。该机制利用“共识悖论”现象，避免盲目信任高共识答案。

5. **对“人工群体智慧”（wisdom of artificial crowds）的实证研究**  
   发现 LLM 多样本之间高度一致的答案反而更可能是集体幻觉（collective hallucination），中等共识区间的答案最具信息价值，呼应人类群体智能中的社会影响效应。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **成本与可持续性** | 全流程运行于单张 RTX 3090，耗时约 5 小时，碳足迹仅 ~0.09kg CO₂eq，远低于云端 API 或多 GPU 训练方案 |
| **无需训练** | 完全零样本（zero-shot），无任何 fine-tuning，适合快速部署 |
| **可复现性** | 开源代码与原始结果，支持本地复现 |
| **抗幻觉能力** | RAGAS faithfulness 达 0.96，表明回答高度依赖图谱上下文 |
| **性能竞争力** | 在零样本本地设置下，接近甚至超越部分监督模型的表现 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 用途 | 样本量 | 特点 |
|--------|------|--------|------|
| **DocRED** | 关系抽取（Relation Extraction） | 500 文档 | 包含跨句实体间关系标注，共 96 类 |
| **HotpotQA** | 多跳推理（Multi-hop Reasoning） | 500 问题 | 需要至少两步推理，标准指标为 EM 和 F1 |
| **Synthetic WebQuestionsSP-style** | 文本转 Cypher 查询（Text-to-Query） | 200 问题 | 自生成数据，包含参考 Cypher 查询语句 |
| **Synthetic + RAGAS** | 反事实评估与反幻觉测试 | 50 问题 | 自动生成并用于 RAGAS 框架评估 faithfulness、relevance 等 |

> ⚠️ 注意：Text-to-Query 和 RAG 数据为合成数据，存在潜在循环偏差风险。

### 实验设置
- **硬件环境**：
  - CPU: Intel Core i9-12900HK
  - 内存: 32GB DDR5
  - GPU: NVIDIA RTX 3090 (24GB VRAM)
- **软件框架**：
  - Ollama v0.20.0，支持原生 JSON Schema 输出约束
  - 模型均采用 Q4_K_M 量化格式（GGUF）
- **超参数**：
  - 温度（temperature）= 0.3（基础），0.7（self-consistency）
  - top_p = 0.9
  - 上下文长度（num_ctx）= 8192
  - 随机种子固定为 42 保证可复现性

### 评估指标
| 任务 | 主要指标 |
|------|----------|
| 关系抽取 | Precision, Recall, F1（软匹配） |
| 文本转查询 | Accuracy, Valid Cypher Rate |
| 多跳推理 | Exact Match (EM), Token-F1 |
| RAG 效果 | RAGAS: Faithfulness, Relevance, Context Precision |
| 所有指标均报告 95% Bootstrap 置信区间 |

### 基线方法对比
| 方法 | 范式 | F1 / EM | 来源 |
|------|------|---------|------|
| DREEAM | Supervised (fine-tuned) | 80.2% | [9] |
| ATLOP | Supervised | 77.8% | [8] |
| GPT-3 few-shot | Cloud API | ~72% | [10] |
| GPT-3 zero-shot | Cloud API | ~30% | [10] |
| ChatGPT zero-shot | Cloud API | ~25% | [11] |
| SYNSYNTH (Gemma-4 Q4) | Zero-shot (local) | **70.2%** | 本文 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总
| 任务 | 指标 | 结果（95% CI） |
|------|------|----------------|
| **关系抽取**（DocRED） | F1 | **0.702 ± 0.04** |
| **文本转查询**（Cypher） | Accuracy | **0.795 ± 0.06** |
| | Valid Cypher Rate | 1.00 |
| **多跳推理**（HotpotQA） | EM | **0.458 ± 0.04** |
| | Token-F1 | 0.520 |
| **RAGAS 评估**（n=50） | Faithfulness | **0.957 ± 0.04** |
| | Context Precision | 0.948 |

### 与基线方法对比
- **关系抽取 F1 达 0.702**，远超 GPT-3 零样本（~0.30）和 ChatGPT（~0.25），逼近监督模型 DREEAM（0.802），差距仅 10 个百分点。
- **多跳推理 EM=0.458**，虽低于最佳监督系统（~0.70），但在完全零样本且本地执行条件下表现优异。
- **Text-to-Query 准确率达 0.80**，说明 LLM 能准确理解自然语言并生成符合 Neo4j 图数据库语法的 Cypher 查询。

### 消融实验结果
#### （1）Pipeline 版本演进（V1 → V5b）
| 版本 | 关键改进 | EM（多跳） | F1（RE） |
|------|----------|------------|-----------|
| V1 | 基线（自由解析 JSON） | 0.462 | 0.263 |
| V2 | + Constrained Decoding (JSON Schema) | 0.462 | 0.260 |
| V3 | + Prompt Engineering + Synonyms | 0.458 | **0.702** ✅ |
| V5a | + Self-consistency (k=3) | **0.482** | 0.702 |
| V5b | + Confidence-routing Cascade | **0.552** ✅ | 0.702 |

> 🔍 发现：**Constrained decoding 对 F1 几乎无提升**，关键增益来自 prompt 设计与 synonym 匹配。

#### （2）Prompt 工程迁移实验（Table 20）
将 V3 的 prompt 优化应用于其他模型（Mistral-Small、Phi-4、GPT-OSS），**仅在 Gemma-4 上带来巨大收益（F1 从 0.039 → 0.702）**，其他模型基本无变化甚至轻微下降。说明 **prompt 与模型架构强耦合，不可直接迁移**。

#### （3）Self-consistency 与 Cascade 效果
| 方法 | EM（500 HotpotQA） | 提升 |
|------|--------------------|------|
| Phi-4 Zero-shot | 0.462 | — |
| Self-consistency (k=3) | 0.482 | +2.0 pts |
| Random Cascade (45%) | 0.524 | +6.2 pts |
| **Confidence-routing Cascade (k=5)** | **0.552** | **+9.0 pts** ✅ |

> ✅ Cascade 在仅使用两个模型的情况下，优于八模型投票（EM=0.446），性价比极高。

#### （4）QLoRA 微调实验
- 在 Qwen2.5-7B 上进行 QLoRA 微调，若训练数据未对齐推理格式（V1），会出现灾难性遗忘（EM 从 0.44 降至 0.19）；
- 若训练格式与推理一致（V4），则能稳定在 EM=0.406，达到 Phi-4 14B 零样本性能的 88%，验证了**训练范式对齐的重要性**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Prompt engineering 是决定性因素**  
   在关系抽取任务中，Gemma-4 的 F1 从原始 0.039 提升至 0.702，**全部增益来自 prompt 重构与 synonym 匹配**，而非模型本身或 constrained decoding。

2. 🔄 **自一致性（self-consistency）效果有限但揭示深层机制**  
   单一模型多次采样可小幅提升 EM（+2 pts），更重要的是揭示了“**共识悖论**”：高一致性（≥80%）往往对应错误答案（集体幻觉），而中等一致性（40%-80%）才是最有潜力区域。

3. 🧠 **“人工群体智慧”现象再现于 LLM**  
   不同模型的回答具有互补性。三模型 oracle 可恢复 46.4% 的难例，远高于任一单一模型。这表明 **architectural diversity 比 stochastic diversity 更具潜力**。

4. 🚀 **信心路由级联机制取得最优性能**  
   利用 agreement 作为路由信号，在低共识时切换模型，实现了 EM=0.552 的最佳结果，**优于零样本、自一致性、随机级联和八模型投票**，是当前最高效的集成策略。

5. 💡 **知识缺口主导失败原因**  
   分析 124 个所有模型都无法解答的问题，发现：
   - 51.6% 是由于**知识缺失**（obsure facts）
   - 25.8% 是**数值推理困难**（dates, quantities）
   - 14.5% 是**格式不匹配**（alias, granularity）
   表明瓶颈更多在于**预训练知识覆盖不足**，而非推理能力。

### 方法的局限性
1. **评估数据部分为合成生成**，尤其是 Text-to-Query 和 RAGAS 数据集，存在**循环偏差风险**，可能高估实际性能。
2. **仅限英文任务**，未测试法语或其他语言场景。
3. **提示工程不可迁移**：V3 的 prompt 优化仅对 Gemma-4 有效，限制通用性。
4. **部分模型无法兼容 JSON 输出**：如 Qwen-3.5 和 DeepSeek-R1 因内部思维链封装导致 JSON 解析失败（100% 错误率）。
5. **内在任务天花板存在**：68.5% 的难例即使使用更大模型也无法解决，受限于上下文外的知识缺失。

### 未来工作方向
1. **扩展至垂直领域**：医疗、法律等专业领域的 KG 构建。
2. **非英语语言支持**：开展多语言（如法语、德语）评估。
3. **增强检索机制**：引入外部检索来补充缺失上下文，突破知识瓶颈。
4. **改进信心校准**：开发外部 calibrator（如 Platt scaling）或图谱验证模块，提升路由可靠性。
5. **探索 QLoRA 在关系抽取中的应用**。
6. **研究 prompt 在不同架构间的可迁移性**，建立通用优化框架。

---

> 📌 **总体评价**：  
> 本文是一次极具启发性的“节俭 AI”实践，证明了**无需训练、仅靠精心设计的 prompt 与本地 LLM 即可在消费硬件上构建实用级知识图谱系统**。其提出的 **confidence-routing cascade** 和对 **consensus paradox** 的观察，为未来 LLM 集成与鲁棒性研究提供了重要洞见。

</details>

---

### 5. [FlexVector: A SpMM Vector Processor with Flexible VRF for GCNs on Varying-Sparsity Graphs](https://arxiv.org/abs/2604.10113)

**Authors**: Bohan Li, Shengmin Li, Xinyu Shi, Enyi Yao, Francky Catthoor, Simei Yang  
**Category**: cs.DC  
**Published**: 2026-04-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.10113v1  

#### Abstract
Graph Convolutional Networks (GCNs) are widely adopted for tasks involving relational or graph-structured data and can be formulated as two-stage sparse-dense matrix multiplication (SpMM) during inference. However, existing accelerators often struggle with the irregular workloads induced by power-la...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FlexVector: A SpMM Vector Processor with Flexible VRF for GCNs on Varying-Sparsity Graphs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Graph Convolutional Networks (GCNs) 在推理阶段的核心是稀疏-稠密矩阵乘法（SpMM），但由于现实图数据普遍遵循 **power-law 分布**（少数“超级节点”连接度极高，多数节点连接稀疏），导致计算负载高度不规则，引发以下挑战：
- 内存访问不规则，缓存效率低；
- 工作负载不平衡，硬件资源利用率低下；
- 固定大小的 on-chip 缓冲区难以适配不同稀疏模式。

现有加速器如 GROW[9] 虽采用统一引擎和 row-wise dataflow，但依赖大容量 cache（数百 KB）来预取高连通节点，成本高昂且对小缓冲配置不敏感。

---

### 提出的新方法与创新思路
本文提出 **FlexVector**，一种基于向量处理器架构、以 **灵活 VRF（Flexible Vector Register File）为核心** 的统一 GCN 推理加速器，通过软硬件协同设计应对变稀疏性图结构。

#### 主要创新点如下：

| 创新维度 | 具体内容 |
|--------|--------|
| **硬件设计** | 引入 **软件管理的灵活 VRF**，分为 **固定区域（fixed region）** 和 **动态区域（dynamic region）**，支持运行时调整边界，适应不同稀疏模式下的数据重用特性。 |
| **数据流优化** | 设计 **分层 dataflow**：<br>- **Buffer-VRF 层面**：采用 **row-wise product dataflow**，实现全行 VRF 访问，提升 lane 利用率；<br>- **DRAM-Buffer 层面**：采用 **inner-product dataflow**，最大化输出驻留，减少部分和存储开销。 |
| **指令集架构（ISA）** | 提出 **粗粒度 ISA（coarse-grained ISA）**，操作粒度为“一个稀疏行 × 稠密子矩阵”，解耦数据移动（MV_Dyn）与计算（CMP），简化调度并降低指令数量。 |
| **图预处理策略** | 提出 **混合图预处理（hybrid preprocessing）**：<br>- **Inter-tile edge-cut**：使用 METIS 进行跨 tile 边切割，保持局部性；<br>- **Intra-tile vertex-cut**：在 tile 内部切分高连通顶点，平衡各行非零元分布，缓解 VRF 容量压力。 |

---

### 相比现有方法的优势
| 对比项 | GROW[9]（代表工作） | FlexVector（本文） |
|------|---------------------|------------------|
| 架构类型 | 统一 SpMM 引擎 | 向量处理器架构 |
| 不规则处理机制 | Cache-centric（DRAM → Cache） | **VRF-centric**（Buffer → Registers） |
| 存储层次 | 大容量 cache（~512KB） | 小型多缓冲 + **Flexible VRF**（仅 2KB Dense Buffer） |
| 预处理目标 | Cache-oriented edge-cut | **VRF-oriented edge-cut + vertex-cut** |
| 数据流 | Row-wise（Cache 层） | **Hierarchical**: Inner-product（DRAM-Buffer）+ Row-wise（Buffer-VRF） |
| 控制粒度 | 细粒度（nonzero × dense row） | **粗粒度**（sparse row × dense submatrix） |
| 资源效率 | 高性能需大 cache 支持 | 在同等 buffer 容量下显著更优 |

> ✅ **核心优势**：将重复的不规则内存访问从 DRAM-cache 层转移到 buffer-VRF 接口，利用灵活 VRF 和多缓冲隐藏延迟，在极小缓冲条件下仍能高效运行。

---

## 2. 核心实验方法和设置

### 使用的数据集
在五个真实世界的 GCN 数据集上进行评估，覆盖多种规模与稀疏模式：

| Dataset | Nodes | Edges | Feature Dim |
|--------|-------|-------|-------------|
| Cora | 2,708 | 5,429 | 1,433 |
| CiteSeer | 3,327 | 4,732 | 3,703 |
| Pubmed | 19,717 | 44,338 | 500 |
| Reddit | 232,965 | 11,606,919 | 602 |
| Yelp | 716,847 | 13,954,819 | 300 |

这些数据集具有典型的 power-law 度分布特征。

---

### 实验设置与评估指标

#### 实现方式
- RTL 使用 SystemVerilog 实现；
- 使用 Synopsys Design Compiler（28nm 工艺，1GHz）综合获取面积与功耗；
- 使用 CACTI 7.0 建模 SRAM/VRF；
- 外部 DRAM 采用 HBM 1.0（带宽 128 GB/s，能耗 7 pJ/bit）；
- 自研 Python 指令驱动模拟器用于 PPA 分析。

#### 默认配置（FlexVector）
| 参数 | 配置 |
|------|------|
| Dense Buffer | 2KB |
| Sparse Buffer | 256B |
| Multi-buffer factor (m) | 6 |
| VRF Width | 128 bit |
| VRF Depth | 6×2（double-VRF 模式） |
| Tile Size | 16×16 或 64×64（可变） |

---

### 基线方法对比
构建两个 GROW-like 基线系统进行公平比较：

1. **GROW-like**  
   - 缓冲配置与 FlexVector 相同（2KB Dense Buffer, m=6）
   - 保留 GROW 的核心机制：cache-centric、run-ahead 执行、fine-grained ISA

2. **GROW-like+**  
   - 使用原始 GROW 的大缓存配置（512KB Dense Buffer, 12KB Sparse Buffer, m=2273）

> 所有对比均控制 buffer 容量一致，确保公平性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（vs. GROW-like）

| 指标 | 结果 |
|------|------|
| **平均加速比（Speedup）** | **3.78×** |
| **平均能效提升（Energy Reduction）** | **40.5% 更低能耗** |
| **芯片面积开销** | 仅增加 **4.7%**（因控制器稍复杂） |

> ⚡️ 在相同 buffer 容量（2KB）下，FlexVector 显著优于 cache-centric 设计。

---

### 与 GROW-like+ 的对比
尽管 GROW-like+ 使用超大缓存（512KB），达到 **1.54× 更高速度**，但代价巨大：
- **面积增加超过 50×**
- **能耗高出 7.2×**

> ❗ 表明 FlexVector 在资源受限场景下具备更强的实用性和性价比。

---

### 消融实验结果（Ablation Study）

逐步添加优化模块，观察性能演进（五数据集几何平均）：

| 阶段 | Speedup | Energy | Area | 说明 |
|------|--------|--------|------|------|
| GROW-like (baseline) | 1.00× | 100% | — | 基线 |
| FlexVector (m=1) | 1.21× | ~same | ↓4.9% | 移动不规则访问至 VRF 接口 |
| + Multi-buffering (m=6) | **3.34×** | ↓36% | ↑4.9% | 隐藏 DRAM 延迟 |
| + Double-VRF | 3.51× | ↓2.1% | no ↑ | 重叠数据加载与计算 |
| + Vertex-cut | 3.52× | ↓2.0% | ↓3.1% (VRF) | 平衡稀疏行密度 |
| + Flexible k (fixed+dyncamic VRF) | **3.78×** | ↓3.1% | no ↑ | 动态调整固定区大小 |

> 🔍 最终 **+Flexible k** 贡献最大增益（额外 7.4% 加速），证明灵活 VRF 极大提升了 VRF 效率。

---

### 其他关键实验发现

#### （1）Algorithm 2（top-k selection）有效性
- 在 CiteSeer 上测试，自适应选择的 `k`（固定区大小）可在所有 VRF 深度配置下达到 **最优静态配置的 98% 性能以内**；
- 无需离线调参即可实现近似最优。

#### （2）DRAM 访问大幅减少
- FlexVector 的 DRAM 访问次数比 GROW-like 减少 **3.0×–8.6×**；
- 即使在 m=1（最小缓冲）下也优于 m=8 的 GROW-like。

#### （3）能量构成分析
- 小 buffer 下：能量主要来自 DRAM 访问 → FlexVector 显著节省；
- 大 buffer 下（m=2273）：能量由 SRAM 主导 → GROW-like 因接近零 miss 而略优，但总体能耗仍远高于 FlexVector。

#### （4）VRF 规模影响
- **VRF Length = 512 bit** 是 PPA 权衡最佳点；
- 更宽 VRF 提升并行度，但收益递减（受 DRAM 带宽限制）；
- **D=8×2 或 D=16×2 的深度** 可提供 6.77–7.26× 加速，同时面积增幅可控（<2×）。

---

## 4. 关键结论和发现

### 主要发现
1. **VRF-centric 架构优于 Cache-centric**：将不规则访问瓶颈从 DRAM-cache 层下沉到 buffer-VRF 接口，结合灵活 VRF 和多缓冲，可在极小缓冲下实现高性能。
2. **分层 dataflow 是关键**：row-wise（VRF 层）+ inner-product（DRAM 层）组合兼顾计算效率与内存复用。
3. **粗粒度 ISA 提升效率**：虽牺牲细粒度灵活性，但显著降低指令数与调度复杂度。
4. **vertex-cut 预处理有效缓解 power-law 影响**：通过拆分超级节点，均衡稀疏行密度，降低 VRF 深度需求。
5. **自适应 fixed-dynamic VRF 分区算法（Algorithm 2）接近最优**：无需人工调参即可适配不同 tile 的稀疏模式。

---

### 方法的局限性
- **依赖图预处理**：需要离线执行 METIS 和 vertex-cut，增加部署复杂性；
- **当前仅支持整数运算（8/32-bit）**：未涵盖浮点或混合精度训练场景；
- **单引擎设计**：未探索多 FlexVector 引擎间的扩展性与通信开销；
- **tile size 敏感性**：性能随 tile 配置变化较大，需 careful tuning。

---

### 未来工作方向
1. **多引擎集成**：构建可扩展的多 FlexVector 阵列，支持更大图的分布式处理；
2. **支持训练任务**：扩展至 GCN 训练中的反向传播与梯度更新；
3. **自动化编译流程**：开发端到端编译器，自动完成图划分、tile 调度与 VRF 配置；
4. **支持动态图**：增强对流式更新图的支持能力；
5. **探索新型 dataflow 组合**：如 outer-product 或 hybrid 方案用于特定子图结构。

---

> ✅ **总结一句话**：  
> **FlexVector 通过“VRF-centric + hierarchical dataflow + coarse-grained ISA + hybrid preprocessing”的软硬件协同设计，在极小缓冲条件下实现了高达 3.78× 的加速和 40.5% 的节能，为面向变稀疏图的高效 GCN 推理提供了新范式。**

</details>

---

### 6. [Energy-Efficient Federated Edge Learning For Small-Scale Datasets in Large IoT Networks](https://arxiv.org/abs/2604.10662)

**Authors**: Haihui Xie, Wenkun Wen, Shuwu Chen, Zhaogang Shu, Minghua Xia  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.10662v1  

#### Abstract
Large-scale Internet of Things (IoT) networks enable intelligent services such as smart cities and autonomous driving, but often face resource constraints. Collecting heterogeneous sensory data, especially in small-scale datasets, is challenging, and independent edge nodes can lead to inefficient re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Energy-Efficient Federated Edge Learning For Small-Scale Datasets in Large IoT Networks

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在大规模 **IoT 网络**中，边缘设备通常面临以下挑战：

- **资源受限**（如通信带宽、计算能力、能量预算）
- **数据异构且规模小**（small-scale datasets），难以支持高效训练
- 传统 **Federated Learning (FL)** 方法过度关注学习性能，忽视了通信资源约束
- 现有 **task-oriented** 或 **task-aware** 资源分配方案存在冗余计算、低利用率或适应性差的问题

因此，如何在小样本条件下实现**高能效、高收敛速度的联邦边缘学习**成为关键难题。

---

### **提出了什么新方法或新思路**

本文提出了一种**面向小规模数据集的能量高效联邦边缘学习框架**，其核心创新包括：

#### ✅ **1. 预期学习损失（Expected Learning Loss）建模**
- 推导了一个理论模型来量化**训练样本数量与学习性能之间的关系**。
- 引入参数 $ \mathcal{F}_i(\cdot) \sim a |\mathcal{X}(t)|^{-b} $ 来拟合预期损失随数据量增长而下降的趋势（遵循“收益递减”规律）。
- 该模型为**以学习为导向的通信资源优化**提供了理论基础。

#### ✅ **2. 随机在线学习算法（Stochastic Online Learning Algorithm）**
- 设计了一种自适应机制，在训练过程中动态调整收集的数据量。
- 利用 **Majorization-Minimization (MM) 框架**推导出收敛边界，确保模型更新稳定。
- 支持从流式数据中持续学习，提升对环境变化的适应性。

#### ✅ **3. 分布式优化算法（FoM-Based Distributed Algorithm）**
- 针对大规模 IoT 网络设计了去中心化的 First-Order Method (FoM) 算法。
- 通过消除冗余变量并固定优化方向，显著提高**收敛概率与可扩展性**。
- 相比传统的 ADMM 方法，具有更低的通信开销和更高的资源利用率。

#### ✅ **4. 协同云-边-端架构**
- 构建了 cloud-edge-end 协同系统，由云端协调全局参数聚合与功率分配，边缘节点负责本地数据采集与模型训练。

---

### **相比现有方法的优势**

| 对比维度 | 本文方法 | 现有方法（如 SRM、QoT-Max） |
|--------|---------|-----------------------------|
| **目标导向** | 学习性能驱动（learning-oriented） | 吞吐率驱动（throughput-oriented）或静态任务感知 |
| **数据利用** | 动态采集 + 在线学习 | 依赖预存的大规模离线数据集 |
| **可扩展性** | 支持大规模分布式部署（FoM算法） | 多为集中式或小规模优化 |
| **资源效率** | 显著降低能耗，提升单位能耗下的学习增益 | 忽视学习质量，易造成资源浪费 |

> 📌 **核心优势总结**：首次将 **expected learning loss** 作为桥梁，统一了 **resource allocation** 与 **learning performance** 的联合优化，在小样本场景下实现了更优的能效-精度权衡。

---

## 2. 核心实验方法和设置

### **使用的数据集**

实验基于三个典型的自动驾驶感知任务，数据由开源仿真平台生成：

| 任务 | 数据类型 | 数据来源 |
|------|----------|----------|
| **Weather Classification** | RGB 图像 | CarlaFLCAV 平台模拟 |
| **Traffic Sign Recognition** | RGB 图像 | CarlaFLCAV 平台模拟 |
| **Object Detection** | LiDAR 点云数据 | CarlaFLCAV + IR-SIM 联合生成 |

此外，在 **collision avoidance 导航案例研究** 中使用了真实风格的 LiDAR 数据包进行验证。

---

### **实验设置**

- **网络拓扑**：
  - 总设备数 $|\mathcal{K}| = 20$
  - 边缘节点数 $I = 10$
  - 每个边缘节点服务若干 IoT 设备
- **无线参数**：
  - 带宽 $B = 4$ MHz
  - 发射功率上限 $P = 50$ mW
  - 噪声功率 $\sigma^2 = -77$ dBm
  - 传输时间 $T = 200$ s
  - 信道模型：瑞利衰落 + 路径损耗（$h_k \sim \mathcal{CN}(0, \rho_k I)$）

- **控制周期**：每轮控制间隔内执行多个 SGD 步骤，之后进行一次资源再分配。

---

### **评估指标**

| 指标 | 定义 | 用途 |
|------|------|------|
| **Learning Loss** | 模型预测误差（如 MSE、交叉熵） | 衡量学习性能 |
| **MSE (Mean Squared Error)** | $||p(t)-p(t-1)||^2 + ||\lambda(t)-\lambda(t-1)||^2 + ...$ | 评估算法收敛速度 |
| **CPU Time / Computational Complexity** | 算法运行时间 | 评估计算开销 |
| **Collision Rate ($r_c$)** | 每障碍物平均碰撞次数 | 自主导航安全性 |
| **Goal Rate ($r_g$)** | 成功到达终点的比例 | 自主导航有效性 |
| **Learning Rate ($r_t$)** | 训练时间占比 | 反映系统响应效率 |

---

### **基线方法对比**

| 基线方法 | 简介 |
|---------|------|
| **SRM [9]** | 最大化总速率（Sum-Rate Maximization），仅考虑信道状态，忽略学习目标 |
| **QoT-Max [14]** | 最大化训练质量（Quality-of-Training），带宽受限下的信息保留策略 |
| **ADMM-based LCPA [12]** | 基于 ADMM 的局部通信功率分配，适用于中小规模网络 |

> ⚠️ 所有对比均在同一仿真环境下进行，保证公平性。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔹 **图3：学习损失 vs. 训练样本数**
- 所有任务中，**实验损失与预期损失高度一致**，验证了所提 expected loss 模型的有效性。
- 小样本时（如 < 500）过拟合严重；当样本数达到约 2000 后趋于饱和，符合“收益递减”规律。

#### 🔹 **图5：学习损失 vs. 传输时间**
- 在相同传输时间内，**FoM-based 方法的学习损失最低**。
- 相比 SRM 下降超过 **30%**，相比 QoT-Max 也有 **~8–12%** 提升。
- 表明本文方法能更有效地利用有限通信资源获取高质量梯度更新。

#### 🔹 **图6：MSE 收敛曲线**
- 引入 **momentum 加速项**后，FoM 算法收敛速度明显加快。
- 收敛速率从 $O(1/t)$ 提升至接近 $O(1/t^2)$。
- 当边缘节点数 $I=10$ 时，加速版本仍保持快速收敛，体现良好可扩展性。

#### 🔹 **图7：平均 CPU 时间 vs. 边缘节点数**
- **FoM-based 分布式算法** 的 CPU 时间几乎不随节点增加而上升（维持在 0.1~0.18 秒）。
- **MM-based 集中式算法** 时间随节点数急剧上升，不适合大规模部署。

#### 🔹 **表 II：综合性能比较**

| 方法 | 复杂度 | 分布式能力 | 在线学习 | 学习性能 |
|------|--------|------------|-----------|------------|
| SRM [9] | $O((K-1)^{3.5})$ | ❌ | ❌ | 低 |
| QoT-Max [14] | $O(K/\sqrt{N})$ | ❌ | ❌ | 高 |
| ADMM-LCPA [12] | $O((K^2+K)/I)$ | ✅ | ❌ | 高 |
| MM-based (本文) | $O((I+2K+2)^{3.5})$ | ❌ | ❌ | 高 |
| **FoM-based (本文)** | **$O(K/I)$** | ✅ | ✅ | **高** |

> ✅ **FoM 方法是唯一同时具备低复杂度、强分布性、支持在线学习且高性能的方法**。

---

### **消融实验结果**

虽然文中未明确列出“ablation study”章节，但从多组对比可得出以下隐含消融结论：

| 组件 | 是否启用 | 结果影响 |
|------|----------|----------|
| **Expected Loss 模型** | 否 | 若直接采用 SRM/QoT-Max，无法建立数据量与学习性能的显式联系，导致资源错配 |
| **MM 收敛边界保障** | 否 | 缺乏稳定性约束可能导致震荡甚至发散，尤其在小样本初期 |
| **Momentum 加速** | 否 | 收敛速度下降约 40%，尤其在高干扰或多节点场景下表现更差 |
| **分布式 FoM 架构** | 否 | 无法扩展到大规模网络，集中式求解器成为瓶颈 |

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **学习性能与通信资源配置之间存在本质耦合关系**，不能孤立优化。
2. ✅ **预期学习损失（expected learning loss）是一个有效的代理指标**，可用于指导资源调度。
3. ✅ **小样本条件下，“渐进式数据采集 + 在线学习”优于一次性加载固定数据集**。
4. ✅ **FoM-based 分布式算法在大规模 IoT 场景下展现出卓越的可扩展性和鲁棒性**。
5. ✅ 在自主导航任务中，本文方法显著降低了碰撞率（↓35%）、提高了目标达成率（↑20%），优于 SRM 和 QoT-Max。

---

### **方法的局限性**

| 局限性 | 说明 |
|--------|------|
| **假设数据分布平稳** | 当前分析基于 stationary distribution，若出现剧烈概念漂移（concept drift），需额外引入滑动窗口或遗忘因子 |
| **未建模本地计算能耗** | 仅优化通信能耗，未联合考虑本地 SGD 更新的 CPU/GPU 开销 |
| **依赖初始样本门槛** | 根据 Corollary 1，每个节点需满足最小初始样本数 $A_i \geq A_{\min}$，否则可能无法收敛 |
| **参数敏感性** | 拟合参数 $(a,b)$ 和噪声常数 $(\epsilon_1,\epsilon_2)$ 影响性能，需通过 warm-up 阶段估计 |

---

### **未来工作方向**

1. **扩展至多任务学习（Multi-task Learning）**：支持同一设备上并行处理多种感知任务。
2. **融合语义通信（Semantic Communication）**：进一步压缩非关键信息，提升传输效率。
3. **引入强化学习进行动态资源调控**：应对非平稳环境下的信道与数据分布变化。
4. **跨模态联邦学习**：整合图像、点云、语音等多模态传感器数据。
5. **硬件原型验证**：在真实嵌入式边缘设备（如 Jetson Nano、Raspberry Pi）上部署验证。

---

> 🧩 **总体评价**：  
> 本论文系统地构建了一个**理论严谨、工程可行、性能优越**的联邦边缘学习框架，特别适合**资源受限、数据稀疏的大规模 IoT 应用场景**（如智慧城市、工业物联网、无人系统）。其提出的 **expected learning loss + FoM 分布式优化范式**，有望成为下一代 **edge intelligence** 系统设计的重要参考。

</details>

---

### 7. [OOWM: Structuring Embodied Reasoning and Planning via Object-Oriented Programmatic World Modeling](https://arxiv.org/abs/2604.09580)

**Authors**: Hongyu Chen, Liang Lin, Guangrun Wang  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.09580v1  

#### Abstract
Standard Chain-of-Thought (CoT) prompting empowers Large Language Models (LLMs) with reasoning capabilities, yet its reliance on linear natural language is inherently insufficient for effective world modeling in embodied tasks. While text offers flexibility, it fails to explicitly represent the stat...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**OOWM: Structuring Embodied Reasoning and Planning via Object-Oriented Programmatic World Modeling**

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

传统基于 **Chain-of-Thought (CoT)** 的 LLM 推理在 **embodied AI**（具身智能）任务中存在严重缺陷：
- 文本推理是**线性的、非结构化的**，难以建模物理环境中的多维状态空间；
- 缺乏对**对象属性、层次关系、动作因果依赖**的显式表示；
- 导致生成的计划存在：
  - **状态模糊性**（无法区分静态属性与动态状态）
  - **逻辑不一致**（长程规划中步骤冲突）
  - **不可执行性**（需额外翻译为机器人控制策略）

这些问题限制了 LLM 在真实世界机器人任务（如房间清理、烹饪等）中的可靠部署。

---

### ✅ **提出了什么新方法或新思路**

提出 **Object-Oriented World Modeling (OOWM)** 框架，将世界建模从“文本流”转变为“软件系统设计”：

#### 🌟 核心思想
将世界模型 $ W $ 定义为一个符号元组：
$$
W = (S, T)
$$
- **$ S $：State Abstraction (Gstate)**  
  使用 **UML Class Diagram** 显式建模环境中的对象层次、属性和关系（如 `Bed`, `Desk`, `Clothing` 及其 `isDirty`, `location` 等状态）。
- **$ T $：Transition Logic / Control Policy (Gcontrol)**  
  使用 **UML Activity Diagram** 表示可执行的控制流程（顺序、分支、循环），直接对应机器人动作序列。

> 💡 **类比**：就像程序员用 UML 设计软件系统一样，OOWM 要求 AI “设计”一个关于物理世界的程序化模型。

#### 🛠️ 技术实现
- 输出格式为 **PlantUML 代码**（可渲染为可视化图表），确保语法正确性和结构一致性。
- 引入三阶段训练流程：
  1. **Supervised Fine-Tuning (SFT)**：学习生成合法的 UML 结构；
  2. **Reinforcement Learning Fine-Tuning (RLFT)**：通过 GRPO 优化计划语义正确性；
  3. **Outcome-based GRPO**：仅凭最终计划奖励反向优化隐式的 state abstraction。

---

### ✅ **相比现有方法的优势**

| 方面 | 传统方法（Text-CoT / GoT） | OOWM |
|------|----------------------------|------|
| **表示能力** | 线性自然语言，表达力弱 | 支持继承、聚合、封装的对象模型 |
| **结构严谨性** | 易出现逻辑跳跃、遗漏前提条件 | UML 强制结构完整性 |
| **可执行性** | 需后处理转换为动作脚本 | Activity Diagram 天然可执行 |
| **可解释性** | 黑箱推理过程 | 图形化、模块化，易于调试 |
| **泛化潜力** | 依赖 prompt 工程 | 基于通用建模范式，支持跨任务迁移 |

> 🔥 **关键突破**：首次将 **软件工程形式化语言（UML）** 引入具身推理，实现了“可编程的世界模型”。

---

## 2. 核心实验方法和设置

### 📚 **使用的数据集**

提出并构建了新基准数据集：**MRoom-30k**
- 包含 **30,792 张杂乱室内场景图像**，来源于 Google/Bing/Baidu/Rednote 和 Messy Rooms Dataset；
- 场景涵盖卧室、客厅等多种家庭环境，具有丰富的视觉混乱度；
- 提供两种标注格式：
  - **UML 格式**（PlantUML 序列化代码）：用于训练 OOWM；
  - **Unstructured Text 格式**：用于与传统 CoT 对比。

#### 数据集划分：
- **Reasoning-Enhanced Subset (1k)**：完整标注 `Gstate` + `Gcontrol`，用于 SFT 和 RLFT；
- **Base Planning Set (29k)**：仅标注 `Gcontrol`，用于大规模 GRPO 微调。

---

### ⚙️ **实验设置**

#### 模型架构
- 主干模型：**InternVL 2.5**（多模态大模型，结合 InternViT-300M 视觉编码器 + InternLM 2.5 语言解码器）；
- 输入：图像 + 清洁指令（如“请制定详细的清洁计划”）；
- 输出：PlantUML 代码（嵌套在 `<think>` 和 `<answer>` 标签内）。

#### 训练流程（Three-Stage Pipeline）
| 阶段 | 方法 | 目标 |
|------|------|------|
| Stage 1 | SFT | 学习生成语法正确的 PlantUML |
| Stage 2 | RLFT + GRPO | 在 1k 数据上优化 `Gcontrol` 语义对齐 |
| Stage 3 | Outcome-based GRPO | 在 29k 数据上进行大规模强化学习，隐式优化 `Gstate` |

---

### 📊 **评估指标**

采用 **Structure-Aware Semantic Evaluation**，避免传统 ROUGE 指标对逻辑无效性的忽略：

#### 分区评估（将 Activity Diagram 拆分为三个功能块）：
1. **Messy Areas Identification**（识别混乱区域）
2. **Priority Order**（优先级排序）
3. **Specific Steps**（具体操作步骤）

#### 评价方式：
- **Semantic Fidelity (Similarity)**：使用 `all-MiniLM-L12-v2` 编码节点，计算匹配节点间的平均余弦相似度；
- **Execution Statistics**（分类指标）：
  - Precision, Recall, F1
  - **Recall 被视为 Task Execution Success Rate**（完成必要动作的比例）

> ✅ 所有文本输出均由 GPT-4o 自动转为 UML 后再评分，保证公平比较。

---

### 🆚 **基线方法对比**

| 基线方法 | 类型 |
|--------|------|
| Text-CoT (VLM-R1 style) | Unstructured Text → Text |
| Tree of Thoughts (ToT) | 复杂文本推理树 |
| Graph of Thoughts (GoT) | 图结构推理 |
| Hybrid Strategy (Text → UML) | 文本推理 + 结构化输出 |
| OOWM 2-Stage | 全结构化（SFT only） |
| OOWM 3-Stage | 全结构化 + GRPO 优化（本文完整方法） |

---

## 3. 主要实验结果和性能指标

### 📈 **主实验结果（MRoom-30k 测试集）**

| Method | Similarity | Precision | **Recall (Success Rate)** | F1 |
|-------|------------|-----------|-----------------------------|-----|
| ToT [31] | 0.4209 | 0.4854 | 0.4639 | 0.4695 |
| GoT [2] | 0.5383 | 0.5263 | 0.5579 | 0.5371 |
| Unstructured Baseline | 0.5498 | 0.5489 | 0.6280 | 0.5811 |
| Hybrid Strategy (Text→OOWM) | 0.5617 | 0.5304 | 0.6536 | 0.5803 |
| OOWM 2-Stage | 0.5562 | 0.5384 | 0.6438 | 0.5812 |
| **OOWM 3-Stage (Ours)** | **0.5694** | **0.5326** | **0.6744** | **0.5904** |

> ✅ **OOWM 3-Stage 在所有关键指标上均取得最优表现**，尤其在 **Recall（任务成功率）提升显著**。

---

### 🔍 **消融实验结果**

#### （1）SFT 初始化的必要性（Ablation Study）
- 若跳过 SFT 直接进行 GRPO，模型完全无法收敛；
- 原因：PlantUML 语法空间稀疏，随机探索几乎不可能触发有效 reward；
- ✅ **SFT 是结构引导的关键前提**，提供“语法先验”。

#### （2）GRPO vs Extended SFT
- SFT 在约第 5 个 epoch 后达到性能饱和；
- 切换至 GRPO 后，所有指标持续上升；
- ✅ **Outcome-based reward 能打破模仿学习瓶颈**，推动模型超越专家示范。

#### （3）Latent Reward Propagation 效果
- 即使没有显式监督 `Gstate`，GRPO 仍能通过 `Gcontrol` 的 reward 反向优化内部状态建模；
- 实验证明该机制有效提升了 plan 的语义一致性。

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **结构化胜过自由文本**  
   尽管文本 CoT 可达高精度（保守输出），但**缺乏持久状态记忆导致召回率低**；而 OOWM 显式维护 `Gstate`，显著提高任务完成率。

2. **UML 是理想的具身建模语言**  
   Class Diagram 和 Activity Diagram 天然契合感知→规划的任务结构，且具备标准化语义，优于 ad-hoc 图结构。

3. **outcome-based RL 可优化隐式推理结构**  
   即使没有中间步骤标注，也能通过最终 plan 的质量反向塑造高质量的 state abstraction。

4. **OOWM 支持跨任务泛化**  
   在未见过的 **Cooking** 和 **Painting** 任务上测试，OOWM 3-Stage 依然领先：
   - Cooking：F1 达 0.3824（远超 baseline 的 0.2694）
   - Painting：虽挑战更大，但仍保持最高 Similarity 和 Precision

---

### ⚠️ **局限性**

1. **依赖强大的视觉基础模型**  
   当前方法基于 InternVL，若 backbone 无法准确识别物体，则 `Gstate` 构建失败。

2. **UML 有一定学习门槛**  
   虽然 PlantUML 简洁，但仍比纯文本更难生成，需要精细的 SFT 初始化。

3. **现实执行仍有差距**  
   当前评估基于模拟 clean plan，尚未连接真实机器人执行闭环。

4. **对未知对象鲁棒性不足**  
   如 Painting 任务中遇到罕见工具时，OOWM 因坚持结构完整性反而表现下降。

---

### 🔮 **未来工作方向**

1. **扩展到更多 UML 图类型**  
   如 Sequence Diagram 建模多智能体协作，State Machine Diagram 建模长期行为状态。

2. **与神经符号系统集成**  
   将 OOWM 作为高层控制器，驱动底层神经网络执行器。

3. **引入用户反馈机制**  
   支持人类修改 UML 图以纠正错误 plan，形成可交互的 world model 编辑器。

4. **构建真实世界 OOWM 数据集**  
   使用机器人采集真实清洁过程视频与日志，建立带执行轨迹的 structured world model annotations。

---

## ✅ 总结

**OOWM 开辟了一条全新的具身智能发展路径**：  
> 不再让 AI “说” 出计划，而是让它“设计”一个可运行的程序化世界模型。

它通过引入 **UML + OOP 范式 + outcome-driven RL**，实现了：
- 更强的结构化推理能力
- 更高的任务执行成功率
- 更好的可解释性与可控性

这标志着 **从“语言推理”迈向“系统建模”** 的重要一步，为下一代可信赖的家用机器人奠定了理论基础。

</details>

---

### 8. [DERM-3R: A Resource-Efficient Multimodal Agents Framework for Dermatologic Diagnosis and Treatment in Real-World Clinical Settings](https://arxiv.org/abs/2604.09596)

**Authors**: Ziwen Chen, Zhendong Wang, Chongjing Wang, Yurui Dong, Luozhijie Jin, Jihao Gu, Kui Chen, Jiaxi Yang, Bingjie Lu, Zhou Zhang, Jirui Dai, Changyong Luo, Xiameng Gai, Haibing Lan, Zhi Liu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.09596v1  

#### Abstract
Dermatologic diseases impose a large and growing global burden, affecting billions and substantially reducing quality of life. While modern therapies can rapidly control acute symptoms, long-term outcomes are often limited by single-target paradigms, recurrent courses, and insufficient attention to ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DERM-3R: A Resource-Efficient Multimodal Agents Framework for Dermatologic Diagnosis and Treatment in Real-World Clinical Settings

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

传统中医（TCM）在皮肤病诊疗中具有整体辨证论治的优势，但其临床实践面临以下挑战：
- **知识体系非标准化**：诊断依赖医生经验，缺乏统一标准。
- **多模态记录不完整**：病历常缺失关键图像或文本信息，图像-文本对应关系断裂。
- **专家推理难以规模化**：高水平TCM医生培养周期长，难以复制。

同时，现有的大型多模态大模型（Multimodal LLMs）如 GPT-5、Gemini 等虽然参数量巨大，但在医学领域存在：
- **高能耗与部署成本高昂**
- **幻觉严重**（hallucination）
- **对小样本、资源受限场景适应性差**

该研究旨在解决：**如何在有限数据和计算资源下，构建一个高效、可靠、符合临床逻辑的TCM皮肤病智能诊断框架？**

---

### 提出了什么新方法或新思路

作者提出 **DERM-3R** —— 一种**轻量级、多智能体协作的多模态框架**，用于真实临床环境下的皮肤病诊断与治疗决策。

#### 核心思想：任务分解 + 多智能体协同

将复杂的皮肤病临床决策过程解构为三个阶段，并由三个专门的 Agent 分别处理：

1. **DERM-Rec**（Recognition Agent）  
   - 输入：单张皮肤病变图像  
   - 输出：细粒度的皮损形态描述（如颜色、分布、鳞屑等）  
   - 功能：实现**精细的视觉语义理解**

2. **DERM-Rep**（Representation Agent）  
   - 输入：多视角皮损图像集合  
   - 输出：整合后的患者整体皮损描述 + 对应的 TCM 病机分析  
   - 功能：从局部到全局的信息聚合与病理建模

3. **DERM-Reason**（Reasoning Agent）  
   - 输入：DERM-Rep 的输出 + 患者病史、症状、舌脉等  
   - 输出：完整的 TCM 辨证分型、治则、方剂推荐  
   - 功能：端到端的多模态临床推理

> 所有 Agent 均基于 **Qwen2.5-VL-7B** 构建，并集成 **Tianyi**（专精于TCM的语言模型），通过 **LoRA** 进行参数高效微调。

---

### 相比现有方法的优势

| 维度 | 传统通用大模型（如 GPT-5.1, Gemini-3-Flash） | DERM-3R |
|------|---------------------------------------------|---------|
| 参数规模 | 百亿级以上 | 仅7B，轻量化 |
| 数据需求 | 需要万亿token训练 | 仅使用103例真实TCM银屑病病例 |
| 推理方式 | 单一模型端到端 | 多Agent协同、任务分解 |
| 医学逻辑一致性 | 弱，易出现幻觉 | 强，符合TCM“辨证-立法-处方”链条 |
| 资源效率 | 高功耗、难部署 | 适合临床实际部署 |
| 性能表现 | 在特定任务上不稳定 | 在多项任务上超越甚至优于百亿参数模型 |

> ✅ **核心优势**：**用更少的数据和更小的模型，实现了更强的专业化推理能力**，验证了“结构化建模 > 暴力扩展”的可行性。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集

所有数据来源于**北京鼓楼中医医院的真实TCM银屑病临床病例**，共103个病例，每个病例包含多次随访记录。

根据任务划分，构建了三个专用数据集：

| Agent | 数据集名称 | 内容 | 规模 |
|-------|-----------|------|-----|
| DERM-Rec | `DRec` | 单图-皮损描述对 | 518 对 |
| DERM-Rep | `DRep` | 多图-综合描述 + TCM病机分析 | 148 例 |
| DERM-Reason | `DReason` | 完整病历输入 + 专家标注的辨证与处方 | 134 例 |

> 所有标注均由多位TCM皮肤科专家独立完成并交叉验证，确保质量和一致性。

---

### 实验设置和评估指标

采用**混合评估框架**，结合自动评估与人类专家评估，全面衡量模型性能。

#### 自动评估（Automatic Evaluation）

1. **基础指标（Basic Metrics）**
   - **BLEU-4** 和 **ROUGE-L**：衡量生成文本与金标准之间的词汇重叠度。

2. **LLM-as-a-Judge**（增强版）
   - 使用多个 Judge 模型（GPT-5.2, Gemini-3-Flash, DeepSeek-V3.2）进行打分。
   - 引入 **RAG（Retrieval-Augmented Generation）机制**，注入权威TCM知识库，提升评判的专业性和可靠性。

#### 人工评估（Human Evaluation）

- **多中心交叉验证**：来自全国 **9家医院** 的 **15位资深皮肤科医生** 参与盲评。
- 评分维度（每项满分10分，总分60分）：
  - 皮损描述（Dermatologic Lesion Description）
  - 病因病机分析（Analysis of Etiology and Pathogenesis）
  - 辨证分型（Syndrome Differentiation）
  - 治则选择（Treatment Principle）
  - 处方药物（Prescriptions and Medications）
  - 可读性（Readability）

---

### 基线方法对比

选取四类主流多模态大模型作为基线：

| 类型 | 模型 |
|------|------|
| 百亿参数通用模型 | **GPT-5.1-instant**, **Gemini-3-Flash** |
| 同规模开源模型 | **Qwen2.5-VL-7B**, **Qwen3-VL-8B** |

> 特别说明：未引入更大参数的Qwen3系列变体，是出于**临床可部署性**的实际考虑。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 表1：DERM-Rep 在皮损描述与病机分析上的 BLEU & ROUGE 结果

| 模型 | 描述 BLEU-4 | 病机 BLEU-4 | 总 BLEU-4 | 描述 ROUGE-L | 病机 ROUGE-L | 总 ROUGE-L |
|------|-------------|-------------|------------|----------------|----------------|---------------|
| GPT-5.1-instant | 0.0714 | 0.0105 | 0.0410 | 0.2331 | 0.1758 | 0.2044 |
| Gemini-3-Flash | 0.0584 | 0.0102 | 0.0343 | 0.2324 | 0.1423 | 0.1874 |
| Qwen2.5-VL-7B | 0.0543 | 0.0658 | 0.0600 | 0.3325 | 0.2682 | 0.3004 |
| Qwen3-VL-8B | 0.0354 | 0.0243 | 0.0298 | 0.2662 | 0.1767 | 0.2214 |
| **DERM-Rep** | **0.2298** | **0.1246** | **0.1772** | **0.4786** | **0.3763** | **0.4275** |

> ✅ **结论**：DERM-Rep 在所有指标上显著领先，平均得分约为基线模型的 **2–5倍**。

---

#### 表2：DERM-Reason 在五大临床子任务上的 BLEU & ROUGE 结果

| 子任务 | GPT-5.1 | Gemini | Qwen2.5 | Qwen3 | **DERM-Reason** |
|--------|--------|--------|--------|--------|------------------|
| 病机分析（BLEU-4） | 0.1293 | 0.0379 | 0.1299 | 0.1615 | **0.1930** |
| 辨证分型（BLEU-4） | 0.0884 | 0.0589 | 0.1070 | 0.0868 | **0.1254** |
| 治则选择（BLEU-4） | 0.1747 | 0.0729 | 0.2016 | 0.2075 | **0.4877** |
| 方剂选择（BLEU-4） | 0.0487 | 0.0574 | 0.0518 | 0.0363 | **0.3997** |
| 处方生成（BLEU-4） | 0.3908 | 0.4340 | 0.3325 | 0.4381 | 0.2379 |
| **平均 BLEU-4** | 0.1664 | 0.1322 | 0.1646 | 0.1861 | **0.2887** |
| **平均 ROUGE-L** | 0.3735 | 0.3048 | 0.3921 | 0.2809 | **0.5802** |

> ✅ **关键发现**：
> - 在**治则选择**和**方剂匹配**任务上，DERM-Reason 明显优于所有基线（BLEU-4 提升超100%）。
> - 处方生成任务因开放性强，n-gram指标偏低，但人工评价仍具优势。

---

#### LLM-as-a-Judge 评估结果

| 模型 | DeepSeek-V3.2 评分 | Gemini-3-Flash 评分 | GPT-5.2 评分 | **平均总分** |
|------|--------------------|------------------------|--------------|-------------|
| GPT-5.1-instant | 17.1111 | 33.1433 | 20.6856 | 23.6462 |
| Qwen2.5-VL-7B | 13.5000 | 16.5000 | 16.6600 | 18.8866 |
| **DERM-Reason** | **29.8500** | **39.2442** | **34.2158** | **34.4366** |

> ✅ **结论**：DERM-Reason 在三位不同 Judge 下均取得最高分，且**方差更低**（~5 vs GPT-5.1的~7），表明其输出更稳定、鲁棒。

---

#### 人工评估结果（Fig. 5）

| 模型 | 总分（满分60） | 方差 |
|------|----------------|------|
| Qwen3-VL-8B | 39.42 | 1.91 |
| Qwen2.5-VL-7B | 35.54 | 2.06 |
| GPT-5.1-instant | 41.23 | 2.03 |
| Gemini-3-Flash | 41.17 | 2.05 |
| **DERM-3R** | **44.16** | **1.49** |

> ✅ **结论**：DERM-3R 不仅得分最高，且**方差最小**，说明其在不同医生眼中表现最一致、最可信。

---

### 消融实验（隐含在设计中）

尽管未设显式消融实验，但从以下对比可看出 DERM-3R 的有效性来源：

- **相比同规模 Qwen 模型**：性能大幅提升 → 证明 **任务分解 + LoRA 微调策略有效**
- **相比百亿参数通用模型**：在部分任务反超 → 证明 **领域感知（domain-aware）建模 > 参数暴力堆砌**
- **人工评估中稳定性更高** → 证明 **多Agent结构增强了推理一致性**

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **结构化多Agent框架优于单一通用模型**  
   将复杂临床决策拆解为识别、表征、推理三步，使每个模块专注特定任务，显著提升整体性能。

2. ✅ **轻量化模型 + 小样本训练可媲美甚至超越百亿参数模型**  
   仅用103例真实病例，在资源受限条件下，DERM-3R 在多个任务上达到SOTA水平。

3. ✅ **领域知识融合至关重要**  
   集成 **Tianyi**（TCM专用语言模型）和 **RAG增强评判系统**，极大提升了医学逻辑一致性与专业性。

4. ✅ **多模态联合推理是TCM AI落地的关键路径**  
   必须同时处理图像（皮损）与文本（症状、舌脉），才能实现准确辨证。

5. ✅ **性能与稳定性兼备**  
   DERM-3R 不仅得分高，且在不同医生和Judge模型下波动最小，具备临床应用潜力。

---

### 方法的局限性

1. **数据来源单一**  
   所有训练数据来自**单一中心、单一学派**，可能限制泛化能力。

2. **疾病种类有限**  
   当前仅验证于**银屑病**，尚未扩展至湿疹、痤疮等其他常见皮肤病。

3. **缓解期皮损识别较弱**  
   因训练集中缓解期样本较少（仅24.8%），导致对该阶段描述准确性下降。

4. **无法完全替代医生**  
   仍需医生审核最终建议，目前定位为“辅助决策工具”。

---

### 未来工作方向

1. **拓展疾病谱系**  
   收集更多类型的皮肤病数据，验证 DERM-3R 的跨病种适用性。

2. **引入更多模态**  
   加入**皮肤镜（dermoscopy）**、**组织病理图像**等，提供更丰富的纹理与结构信息。

3. **多流派知识融合**  
   探索如何安全地融合不同TCM学派的知识，避免内部逻辑冲突。

4. **构建闭环学习系统**  
   将医生反馈纳入模型迭代，形成“人机互教”机制，持续优化性能。

5. **推进真实世界临床试验**  
   在真实门诊环境中部署测试，评估其对诊疗效率与质量的影响。

---

> 📌 **一句话总结**：  
> **DERM-3R 成功证明了“小而精”的结构化多Agent框架，能够在极低资源消耗下，实现媲美甚至超越百亿参数通用模型的中医皮肤病智能诊疗能力，为医学AI的实用化开辟了一条高效、可靠的新路径。**

</details>

---

### 9. [Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale](https://arxiv.org/abs/2604.11554)

**Authors**: Liujie Zhang, Benzhe Ning, Rui Yang, Xiaoyan Yu, Jiaxing Li, Lumeng Wu, Jia Liu, Minghao Li, Weihang Chen, Weiqi Hu, Lei Zhang  
**Category**: cs.CL  
**Published**: 2026-04-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.11554v1  

#### Abstract
Reinforcement learning (RL) post-training has proven effective at unlocking reasoning, self-reflection, and tool-use capabilities in large language models. As models extend to omni-modal inputs and agentic multi-turn workflows, RL training systems face three interdependent challenges: heterogeneous ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

随着大模型向 **omni-modal**（支持图像、文本、音频、视频等多模态输入）和 **agentic**（多轮推理、工具调用、搜索增强生成）方向发展，传统的 **Reinforcement Learning (RL)** 后训练系统面临三大挑战：

1. **异构模态流水线（Heterogeneous modality pipelines）**  
   多模态数据在表示、大小、预处理延迟、内存占用等方面差异巨大，传统以文本为中心的框架难以统一高效处理。

2. **大规模下的运行鲁棒性（Operational robustness at scale）**  
   多模态任务存在严重的长尾延迟（如30秒视频处理）、易发生 OOM（Out-of-Memory），且在数百至数千 GPU 上运行时需容忍硬件故障、NCCL 超时和服务崩溃。

3. **执行解耦与策略灵活性（Execution decoupling and policy flexibility）**  
   同步训练中，训练器常因等待最慢的 rollout 而空闲；完全异步虽提升吞吐，但引入 **policy staleness**（策略陈旧性）影响收敛。如何在 **throughput** 和 **policy freshness** 之间灵活权衡是关键。

---

### 提出了什么新方法或新思路

作者提出 **Relax** ——一个开源的、面向大规模 omni-modal 和 agentic 场景的 RL 训练引擎，通过三层协同设计解决上述问题：

#### 1) **Role-Isolated Service Architecture (S3.2)**  
每个 RL 角色（如 Actor、Critic、Rollout、Reward Model）作为独立的、故障隔离的服务运行于 **Ray Serve** 上，具备以下优势：
- **故障隔离**：单个服务崩溃不影响全局训练。
- **弹性伸缩**：可独立扩展 rollout 或 reward 服务。
- **生命周期管理**：支持 per-role 的检查点恢复与升级。

引入 **Distributed Checkpoint Service (DCS)**，将权重同步从训练流程中剥离，实现低延迟、跨集群的模型分发（支持 NCCL 和 TCP 两种后端）。

#### 2) **Staleness-Unified Asynchronous Training (S3.3)**  
基于 **TransferQueue (TQ)** 构建异步数据总线，实现：
- **统一训练模式控制**：通过单一参数 `max_staleness` 控制训练模式：
  - `max_staleness=0` → on-policy
  - `max_staleness=1` → near-on-policy
  - `max_staleness>1` → fully asynchronous off-policy
- **流式微批次调度（Streaming micro-batch）**：rollout 完成一个 micro-batch 即写入 TQ，训练器持续消费，避免长尾阻塞。
- **字段级解耦（Field-level decoupling）**：不同模态字段（text/image/audio/video）可独立写入与读取，下游无需等待所有模态完成。

#### 3) **Omni-Modal Agentic RL 支持 (S4)**  
- **原生多模态流水线**：统一处理图像、文本、音频、视频，支持动态加载与模态感知并行策略。
- **模态感知并行策略**：
  - **ViT Tensor Parallelism**：视觉编码器在 TP 组内复制，AllReduce 合并特征。
  - **Encoder-Aware Pipeline Parallelism**：将 ViT、Whisper 等编码器置于 PP0 阶段，减少跨阶段通信。
- **可扩展的 agentic 接口**：支持多轮 rollout、自定义 reward 服务、工具调用与沙箱集成。

---

### 相比现有方法的优势

| 方面 | Relax | 现有框架（如 veRL、OpenRLHF） |
|------|-------|-----------------------------|
| 架构 | 服务化、角色隔离 | 多为单体或混合架构 |
| 异步支持 | 统一 `max_staleness` 参数控制 | 需不同代码路径或配置模板 |
| 多模态支持 | 原生 omni-native 设计 | 文本优先，多模态为“打补丁” |
| 故障恢复 | 分级恢复（in-place / global restart） | 通常整机重启 |
| MoE 稳定性 | R3 支持仅 1.9% 开销 | veRL 中 R3 开销达 32% |
| 扩展性 | 支持跨集群弹性伸缩（未来开源） | 有限 |

---

## 2. 核心实验方法和设置

### 使用的数据集

| 数据集 | 类型 | 任务描述 |
|--------|------|----------|
| **Echo Ink (AVQA-R1-6K)** | 图像+音频+文本 | 多模态问答（Qwen3-Omni-30B） |
| **NextQA** | 视频（0–30s 子集） | 视频理解与推理 |
| **DAPO-MATH-17k** | 纯文本 | 数学推理任务（Qwen3-4B / Qwen3-30B-A3B） |
| **Deepeyes** | 图像+工具调用 | 多轮视觉问答（crop/zoom 工具） |
| **LLaVA-Video-178K** | 视频 | 视频理解训练子集 |

---

### 实验设置和评估指标

#### 模型规模
- **Qwen3-4B**：密集模型，用于与 veRL 对比 end-to-end 性能。
- **Qwen3-Omni-30B**：omni-modal 模型，验证多模态收敛。
- **Qwen3-30B-A3B**：MoE 模型（30B 总参，3B 激活），用于 R3 消融实验。

#### 硬件配置
- **16×H800 GPUs**：用于 Qwen3-4B 和 MoE 实验。
- **16×H20 GPUs**：用于 Qwen3-Omni-30B 多模态训练。

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Step Time (s)** | 每训练步耗时 |
| **Steps/Hour** | 每小时训练步数 |
| **Speedup vs Colocate** | 相对于 colocate 模式的加速比 |
| **Reward Convergence** | 奖励值随训练步数的变化 |
| **Routing Mismatch** | MoE 模型中 rollout 与 training 的专家选择不一致率 |
| **Trainer Idle Ratio** | 训练器空闲时间占比 |

---

### 基线方法对比

- **veRL**：当前主流 RL 框架，支持 HybridFlow 编程模型和 3D 并行。
- **Colocate Mode**：Relax 内部的同步模式，训练与推理共享 GPU。
- **AsyncFlow / AReaL**：其他异步 RL 框架，但未整合 omni-modal 支持。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

| 实验配置 | 方法 | Step Time | Steps/Hour | Speedup vs Colocate |
|---------|------|-----------|------------|---------------------|
| Qwen3-4B / DAPO-MATH | Relax (Async Off-Policy) | 128.6s | 28.0 | **1.76×** |
| Qwen3-4B / DAPO-MATH | Relax (Async On-Policy) | 201.0s | 17.9 | 1.12× |
| Qwen3-4B / DAPO-MATH | Relax (Colocate) | 225.9s | 15.9 | 1.00× |
| Qwen3-Omni-30B / Echo Ink | Relax (Fully Async) | 133.6s | 26.9 | **2.00×** |
| Qwen3-Omni-30B / Echo Ink | Relax (Colocate) | 267.4s | 13.5 | 1.00× |

> ✅ **Relax 在更大模型上异步优势更明显**：从 1.76× 提升至 2.00×。

---

### 与基线方法的对比结果

#### vs veRL（Qwen3-4B / DAPO-MATH）
| 指标 | Relax | veRL | 优势 |
|------|-------|------|------|
| Step Time | 125.6s | 150.5s | **1.20× 更快** |
| Steps/Hour | 28.7 | 23.9 | +20% |
| Rollout on Critical Path | 0s（完全并行） | 38.2s | ✅ 避免阻塞 |
| Ref LogP 计算开销 | 0s（资源分离） | 27.3s | ✅ 不占训练时间 |

> 📌 Relax 利用 **streaming micro-batch + 资源分离**，几乎消除训练等待。

---

### 消融实验结果

#### R3（Rollout Routing Replay）在 MoE 模型上的开销对比
| 配置 | Routing Mismatch 下降 | Step Time Overhead |
|------|------------------------|--------------------|
| veRL + R3 | ~38% | **+32%** |
| Relax + R3 | ~38% | **+1.9%** |

> ✅ Relax 通过优化序列化路径（NCCL zero-copy broadcast）将 R3 开销降至近乎零。

#### 多模态收敛稳定性
- **Qwen3-Omni-30B 在 Echo Ink（图像+音频）上**：450 步内奖励从 0.72 → 0.93。
- **NextQA（视频）上连续训练 2000 步**：奖励单调上升至 0.93，无崩溃或退化，标准差稳定在 0.04–0.06。

#### agentic RL 收敛性（Deepeyes）
- Relax 与 veRL 收敛曲线几乎重合，最终平均奖励分别为 **2.0000** 与 **1.9783**，验证多轮轨迹处理正确性。

---

## 4. 关键结论和发现

### 主要发现

1. **服务化架构 + 异步数据总线 是应对 omni-modal RL 挑战的有效范式**：
   - 角色隔离提升鲁棒性。
   - TransferQueue 实现字段级异步，天然适配多模态与 agentic 流水线。

2. **`max_staleness` 统一控制训练模式**：
   - 无需修改代码即可切换 on-policy / off-policy，极大提升实验灵活性。

3. **异步训练显著提升吞吐，且不损害收敛性**：
   - 在 Qwen3-4B 上达 **1.76×** 加速，在 Qwen3-Omni-30B 上达 **2.00×**。
   - 所有模式最终收敛到相同 reward 水平。

4. **R3 在 Relax 中近乎零开销**：
   - 仅 **+1.9%** 步时开销，而 veRL 中高达 **+32%**，使 MoE 模型训练更稳定可行。

5. **支持稳定多模态 RL 收敛**：
   - 在图像、文本、音频、视频上均实现超过 2000 步的稳定训练。

---

### 方法的局限性

1. **暂不支持生成式多模态 RL**：
   - 当前仅支持多模态输入 + 文本输出，尚未支持图像/音频生成（如 text-to-image reward optimization）。

2. **部署复杂度较高**：
   - 需要部署 Ray Serve、DCS、TransferQueue 等多个组件，对小型团队门槛较高。

3. **超大规模模型验证仍在进行**：
   - 当前实验最大为 30B–35B 模型，397B+ 超大模型的验证尚未完成。

4. **弹性伸缩功能暂未开源**：
   - 虽已实现，但 Autoscaler 和跨集群扩缩容功能尚未发布。

---

### 未来工作方向

1. **支持生成式多模态 RL**：
   - 扩展 reward 接口与解码策略，支持图像/音频生成任务。

2. **开放弹性伸缩能力**：
   - 发布基于 REST API 的动态扩缩容模块，支持 AutoScaler。

3. **丰富 agentic 场景支持**：
   - 集成 SWE-Agent、CodeAgent、Search-Augmented Reasoning 等复杂任务。

4. **进一步降低 FP 精度差异**：
   - 探索 FP8、INT8 等更低精度下的数值一致性优化。

5. **社区生态建设**：
   - 提供更多 recipe 示例，降低使用门槛。

---

> 🔗 **项目地址**：[https://github.com/rednote-ai/Relax](https://github.com/rednote-ai/Relax)  
> 📅 **发布日期**：April 14, 2026

</details>

---

### 10. [Exact Certification of Neural Networks and Partition Aggregation Ensembles against Label Poisoning](https://arxiv.org/abs/2604.11416)

**Authors**: Ajinkya Mohgaonkar, Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Stephan G\"unnemann  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.11416v1  

#### Abstract
Label-flipping attacks, which corrupt training labels to induce misclassifications at inference, remain a major threat to supervised learning models. This drives the need for robustness certificates that provide formal guarantees about a model's robustness under adversarially corrupted labels. Exist...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Exact Certification of Neural Networks and Partition Aggregation Ensembles against Label Poisoning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**标签翻转攻击**（label-flipping attacks）下的模型鲁棒性认证问题展开研究。这类攻击通过篡改训练数据中的标签来诱导模型在推理时产生错误预测，对监督学习构成严重威胁。

现有的认证框架（如 SS-DPA）多采用**黑盒**（black-box）方式处理集成中的基分类器，即仅依赖其输出而不利用内部结构信息，导致认证界限过于保守，无法反映模型的真实鲁棒性。

### 提出的新方法与新思路
作者提出了两个核心方法：

- **EnsembleCert**：首个面向**分块聚合集成**（partition aggregation ensembles）的**白盒认证框架**。它利用基分类器的内部信息（white-box knowledge），将每个分块上的白盒证书聚合为整个集成的紧致鲁棒性保证。
  
- **ScaLabelCert**：一种高效计算**无限宽神经网络**（infinitely wide neural networks）在标签翻转攻击下**精确证书**（exact certificate）的方法。该方法基于**神经正切核**（Neural Tangent Kernel, NTK）理论，将宽神经网络等价于核方法（如 kernel SVM 或 kernel regression），从而实现多项式时间内的精确认证。

### 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **认证精度** | 白盒信息显著提升了认证界限的紧致性，远优于传统黑盒方法（如 SS-DPA）。 |
| **计算效率** | ScaLabelCert 将原本 NP-hard 的认证问题简化为多项式时间可解，首次实现了对宽神经网络的**可扩展精确认证**。 |
| **适用性广** | EnsembleCert 是通用框架，可适配多种基分类器（如无限宽 NN、有限宽 NN、平滑线性分类器等）。 |
| **颠覆认知** | 实验表明，**少量分块 + 白盒信息**即可超越**大量分块 + 黑盒信息**的效果，挑战了“深度分块是强鲁棒性的必要条件”这一主流观点。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CIFAR-10**
- **MNIST**
- **Binary MNIST 1-vs-7**（二分类任务）

### 实验设置
- **特征提取**：使用预训练的自监督模型提取特征：
  - MNIST → RotNet
  - CIFAR-10 → SimCLR
- **基分类器设计**：
  - **无限宽神经网络**：通过 NTK 等价转换为 kernel SVM 或 kernel ridge regression。
  - **有限宽网络**：全连接线性分类器，结合梯度界方法（gradient-based parameter bounding）进行认证。
  - **平滑线性分类器**：用于对比 randomized smoothing 方法。
- **分块策略**：训练数据被划分为 $N_p$ 个互不相交的子集，每个基分类器在其中一个子集上训练。
- **集成决策**：多数投票机制（majority vote）决定最终预测。

### 评估指标
- **Certified Accuracy**：在最多 $r$ 个标签被翻转的情况下仍能正确且鲁棒预测的测试样本比例。
- **Median Certified Robustness (MCR)**：50% 正确分类样本所能承受的最大标签翻转数量（中位数级别）。

### 基线方法对比
- **SS-DPA**：当前最先进的黑盒分块聚合防御方法，作为主要对比基线。
- **Gradient-based parameter bounding**（Sosnin et al., 2025）：用于对比白盒认证方法的有效性。
- **Randomized Smoothing**（Rosenfeld et al., 2020）：另一类集成防御方法，也以黑盒形式集成到 EnsembleCert 中进行比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 在 CIFAR-10 上的表现（使用 kernel SVM 作为基模型）
- EnsembleCert（仅用 **10 个分块**）相比 SS-DPA（使用 **1000 个分块**）：
  - **MCR 提升高达 26.5%**（图1b）。
  - 所需分块数减少 **100倍**。
- 即使只使用极少数分块（如 10–50），EnsembleCert 的认证准确率也全面超越 SS-DPA。

#### 在 MNIST 和 CIFAR-10 上的泛化表现
- 对于 **kernel regression** 模型，在高正则化参数 $\lambda$ 下，增加分块反而会**降低**认证鲁棒性（robustness decay），说明过度分块可能损害真实鲁棒潜力。
- 当使用足够小的 $C$ 参数时，**kernel SVM** 的 MCR 几乎**不随分块数变化**，直到收敛至黑盒界限，表明少量分块已足够。

#### 与其他白盒方法对比
- **ScaLabelCert vs Gradient-based bounding**（Sosnin et al., 2025）：
  - 在 CIFAR-10 上，ScaLabelCert 显著优于梯度界方法，认证准确率更高。
  - 梯度界方法因松弛过松，很快变得 vacuous（无意义），而 ScaLabelCert 提供的是**精确或近精确**的界限。

### 消融实验结果
- **白盒 vs 黑盒**：注入白盒知识后，认证鲁棒性在低到中等分块数下均有**显著提升**。
- **分块数量的影响**：
  - 白盒证书的优越性在小分块数下最为明显。
  - 随着分块数增加，白盒与黑盒证书逐渐**收敛**，因为极端情况下单个标签翻转确实可能改变一个基分类器的预测（worst-case assumption 成立）。
- **独立模型 vs 分块集成**：
  - 实验发现，在某些设置下，**不分块的单一模型**（$N_p=1$）的认证准确率甚至高于其分块集成版本。
  - 这引发了对“分块聚合是否真正增强鲁棒性”的深刻反思。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **白盒信息极大提升认证能力**：利用基分类器的内部结构信息可以显著收紧认证界限，揭示集成的真实鲁棒潜力。
2. ✅ **ScaLabelCert 实现首次可扩展的精确认证**：基于 NTK 的等价性，首次实现了对宽神经网络在标签翻转攻击下的**多项式时间精确认证**。
3. ❌ **“越多分块越好”并非普适真理**：实验表明，**少量分块 + 白盒信息**即可达到甚至超过**大量分块 + 黑盒信息**的效果，挑战了现有范式。
4. ⚠️ **分块可能削弱鲁棒性**：对于本身具有高鲁棒性的基分类器（如高正则化 kernel regression），过度分块反而可能导致认证鲁棒性下降。
5. 🔁 **单一模型有时优于集成**：在某些场景下，不分块的完整训练模型比其分块集成更具认证鲁棒性，提示我们重新思考集成的意义。

### 方法的局限性
- **主要适用于无限宽网络**：ScaLabelCert 的理论基础依赖于 NTK 极限，对**有限宽度神经网络**仅为渐近精确。
- **计算复杂度仍受分块数影响**：虽然整体为多项式时间，但 MCKP 求解步骤的复杂度为 $O(K \cdot N_p)$，在极大集成中仍有开销。
- **尚未支持更复杂的攻击模型**：目前聚焦于标签翻转攻击，对特征扰动等 clean-label 攻击的支持仍在探索中（见 Appendix C）。

### 未来工作方向
- 开发适用于**有限宽度神经网络**的高效白盒认证方法。
- 探索如何将 EnsembleCert 扩展至其他威胁模型，如 **clean-label attacks** 和 **backdoor attacks**。
- 研究如何设计最优的分块策略，在计算成本与认证鲁棒性之间取得平衡。
- 将白盒认证思想推广至其他集成范式，如 bootstrap aggregation 和 randomized smoothing。

---

> **总结一句话**：  
> 本文通过引入 **EnsembleCert** 和 **ScaLabelCert**，首次实现了对神经网络及其集成的**白盒级精确鲁棒性认证**，不仅大幅提升了认证强度，更从根本上挑战了“必须重度分块才能获得强鲁棒性”的既有认知，为可信机器学习提供了新的理论工具和实践路径。

</details>

---

### 11. [Learning How Much to Think: Difficulty-Aware Dynamic MoEs for Graph Node Classification](https://arxiv.org/abs/2604.11473)

**Authors**: Jiajun Zhou, Yadong Li, Xuanze Chen, Chen Ma, Chuang Zhao, Shanqing Yu, Qi Xuan  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.11473v1  

#### Abstract
Mixture-of-Experts (MoE) architectures offer a scalable path for Graph Neural Networks (GNNs) in node classification tasks but typically rely on static and rigid routing strategies that enforce a uniform expert budget or coarse-grained expert toggles on all nodes. This limitation overlooks the varyi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Learning How Much to Think: Difficulty-Aware Dynamic MoEs for Graph Node Classification*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现有的 **Graph MoE** 架构在图节点分类任务中通常采用**静态、刚性的路由策略**（如固定的 `top-k` 选择），导致所有节点被分配相同的专家预算（expert budget）。这种设计忽略了图中节点判别难度的异质性，引发两个核心问题：

- **对“简单”节点过度计算**：低熵（homophilous）区域的节点本可由少量专家处理，却被迫激活多个专家，造成冗余计算和噪声引入。
- **对“困难”节点欠拟合**：高熵（heterophilous 或边界）节点需要更强的非线性和推理能力，但固定的小 `k` 限制了模型表达力。

### **提出了什么新方法或新思路**

本文提出 **D²MoE (Difficulty-Aware Dynamic Mixture-of-Experts)**，一种全新的动态 MoE 框架，其核心思想是：

> **将“思考多少”（how much to think）的问题转化为基于节点难度的连续、细粒度专家资源分配问题。**

具体创新包括：

- **以预测熵（predictive entropy）作为实时难度代理**：利用前一轮训练的预测分布计算每个节点的不确定性，量化其判别难度。
- **提出 difficulty-driven top-p 路由机制**：
  - 不再使用固定的 `top-k`，而是为每个节点动态生成一个累积概率阈值 `p`。
  - 通过 Sigmoid 映射将预测熵映射到 `(0,1)` 区间，控制 `p` 的大小。
  - 采用 **top-p 路由**：选择最小数量的专家，使其累计路由权重超过 `p`，实现按需激活。
- **实现连续、细粒度的专家预算缩放**：简单节点（低熵）仅激活少数专家（甚至 `k=1`），困难节点（高熵）则自动调用更多专家进行协同推理。

### **相比现有方法的优势**

| 优势维度 | 描述 |
|--------|------|
| **灵活性** | 打破了静态 `top-k` 的刚性约束，支持节点级、连续的计算资源分配。 |
| **效率** | 显著降低内存消耗和训练时间，尤其在大规模图上。 |
| **性能** | 在异配图（heterophilous graphs）上表现尤为突出，解决欠拟合问题。 |
| **理论支撑** | 通过偏差-方差权衡分析，证明最优专家数 `k*` 与预测熵呈正相关，为方法提供理论依据（Theorem 1）。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

共在 **13 个基准图数据集** 上进行实验，涵盖同配（homophilous）与异配（heterophilous）场景：

- **Homophilous**:  
  `Computers`, `Photo`, `Coauthor CS`, `Coauthor Physics`, `Wiki-CS`, `Facebook`, `Ogbn-arxiv`
- **Heterophilous**:  
  `Actor`, `Chameleon-filtered`, `Squirrel-filtered`, `Roman-empire`, `Tolokers`, `Penn94`

### **实验设置和评估指标**

- **任务**：半监督节点分类（semi-supervised node classification）
- **划分比例**：标准 48% / 32% / 20%（训练/验证/测试），`Ogbn-arxiv` 使用官方划分
- **评估指标**：平均测试准确率（test accuracy %）± 标准差（10 次随机种子运行）
- **硬件环境**：NVIDIA A100 GPU (40GB)

### **基线方法对比**

共比较 **19 个基线方法**，分为四类：

| 类别 | 代表方法 |
|------|--------|
| **Vanilla GNNs** | `MLP`, `GCN`, `GraphSAGE`, `GAT` |
| **Heterophilic GNNs** | `H2GCN`, `GPRGNN`, `FAGCN`, `ACMGCN`, `FSGNN` |
| **Graph Transformers (GTs)** | `Vanilla GT`, `ANS-GT`, `NAGphormer`, `SGFormer`, `Exphormer`, `Difformer` |
| **Graph MoEs** | `GMoE`, `DAMoE`, `NodeMoE`, `Mowst`, `Moscat` |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

- **在全部 13 个数据集上达到 SOTA 性能**，显著优于所有基线。
- 在**异配图**上提升最为明显，最高**准确率提升达 7.92%**（如 `Squirrel-fix` 上从 42.24% → 44.11%）。
- 在 `Penn94` 和 `Ogbn-arxiv` 等大规模图上：
  - **内存消耗减少高达 73.07%**
  - **训练时间缩短 46.53%**

### **与基线方法的对比结果**

| 对比项 | 结果 |
|-------|------|
| **vs. 静态 MoEs (GMoE, DAMoE)** | D²MoE 在所有数据集上均超越，尤其在高熵节点上优势显著（见 Fig. 3）。 |
| **vs. 条件激活 MoE (Mowst)** | Mowst 采用“强-弱”二元切换，仍属粗粒度；D²MoE 实现连续调节，性能更优。 |
| **vs. Graph Transformers** | GTs 多因 OOM（Out-of-Memory）失败（如 `Vanilla GT` 在多个大图上无法运行），而 D²MoE 保持高效稳定。 |

### **消融实验结果（Ablation Study）**

| 变体 | 描述 | 性能影响 |
|------|------|----------|
| **Static top-k** | 固定专家数 `k` | 平均下降 1.46%，验证静态策略的局限性 |
| **Fixed top-p** | 全局统一 `p` 阈值 | 下降 5.00%，说明需节点级自适应 |
| **Random top-p** | 随机生成 `p`，不依赖熵 | 下降 1.25%，证明预测熵是有效难度代理 |
| **w/o LRE** | 移除路由熵正则化 | 下降 1.13%，导致路由分布平坦，激活过多专家 |
| **w/o LLB** | 移除负载均衡损失 | 下降 2.28%，导致专家崩溃（某些专家被过度使用） |

> ✅ 结论：**动态路由 + 难度感知 + 正则化** 三者缺一不可。

---

## 4. 关键结论和发现

### **主要发现**

1. **预测熵是有效的节点难度代理**：实验表明，分类准确率随预测熵单调递减，验证了熵与判别难度的高度相关性。
2. **动态资源分配优于静态模式**：D²MoE 成功实现了“易题简答，难题深思”的智能推理范式。
3. **专家出现功能分化**：
   - “**通才专家**”（generalists）主导低熵节点处理；
   - “**专才专家**”（specialists）在高熵区域被激活，解决复杂歧义。
4. **资源分配具有空间合理性**：
   - t-SNE 可视化显示，专家激活集中在类别边界和混杂区域（Fig. 6），体现“边界聚焦”策略。

### **方法的局限性**

- **历史依赖性**：当前版本依赖前一轮的预测作为难度信号，在冷启动阶段需特殊处理（如初始化全激活）。
- **极端标签噪声下的熵校准问题**：在高噪声场景下，预测熵可能失真，影响路由决策。
- **存储开销**：维护历史预测会带来额外内存负担，不利于超大规模图（如 billion-scale）部署。

### **未来工作方向**

1. 设计**抗噪的不确定性度量**，提升在噪声图上的鲁棒性。
2. 开发**无需历史记录的轻量级估计器**，支持超大规模图的在线推理。
3. 将该自适应范式扩展至**时序图（temporal graphs）** 和流式学习场景。

---

> **总结**：D²MoE 通过引入 **difficulty-aware top-p routing**，成功将静态 MoE 升级为动态、自适应的推理系统，不仅在性能上实现 SOTA，更在效率与灵活性之间取得卓越平衡，为大规模图学习提供了新的 scalable 范式。

</details>

---

### 12. [Introspective Diffusion Language Models](https://arxiv.org/abs/2604.11035)

**Authors**: Yifan Yu, Yuqing Jian, Junxiong Wang, Zhongzhu Zhou, Donglin Zhuang, Xinyu Fang, Sri Yanamandra, Xiaoxia Wu, Qingyang Wu, Shuaiwen Leon Song, Tri Dao, Ben Athiwaratkun, James Zou, Fan Lai, Chenfeng Xu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.11035v1  

#### Abstract
Diffusion language models promise parallel generation, yet still lag behind autoregressive (AR) models in quality. We stem this gap to a failure of introspective consistency: AR models agree with their own generations, while DLMs often do not. We define the introspective acceptance rate, which measu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Introspective Diffusion Language Models**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前的 **Diffusion Language Models (DLMs)** 虽然理论上支持并行生成，但在实际中仍显著落后于 **Autoregressive (AR) 模型** 的生成质量。作者指出，这一差距的根本原因在于 **缺乏“内省一致性”（introspective consistency）**：  
- AR 模型在生成 token 后，其后续推理过程会自然地验证这些 token 是否符合自身预测分布（即 `p = q`）。  
- 而大多数 DLMs 在训练时未强制这种一致性，导致模型生成的 token 与其自我验证结果不一致（`p ≠ q`），从而影响连贯性和推理能力。

此外，现有 DLM 推理系统存在以下问题：
- **计算效率低**：多步去噪 + KV commit 导致高 FLOPs 开销；
- **与 AR serving 栈不兼容**：无法利用成熟的 AR 推理优化（如 paged KV cache、continuous batching）。

---

### **提出的新方法与思路**

作者提出了 **Introspective Diffusion Language Model (I-DLM)**，其核心思想是：
> “从 AR 出发，保留其内在优势，而非从扩散出发逐步逼近 AR。”

#### **三大核心技术组件：**

1. **Introspective-Consistency Training（内省一致性训练）**
   - 使用 **严格因果注意力（strict causal attention）** 和 **logit shift**，确保模型在训练时同时学习生成（decode）和验证（introspect）路径。
   - 采用 **全掩码训练目标（all-masked objective）**：输入序列全部替换为 `[MASK]`，拼接原始序列作为参考，使每个位置都参与监督信号，提升训练效率。
   - 引入 **auto-balanced loss**：动态平衡 masked 区域（生成任务）和 clean 区域（验证任务）的损失权重，防止一方主导梯度更新。

2. **Introspective Strided Decoding (ISD)**
   - 一种新型单次前向传播解码算法，在每一步中：
     - 生成新的 stride tokens；
     - 同时对上一步生成的 tokens 进行验证（基于因果锚定分布 `p`）；
     - 利用 `p/q` 接受准则决定是否接受 token 或重采样。
   - 支持自适应 stride：简单 token 并行生成，困难 token 回退到类似 AR 的逐个生成。
   - 可扩展为 **Lossless ISD (R-ISD)**：通过 **gated LoRA residual adaptation**，仅在 `[MASK]` 位置激活 LoRA，保证 introspection 使用纯 base model 权重，实现输出与 base AR 模型 **bit-for-bit lossless**。

3. **AR-Compatible Serving Stack**
   - 构建在 SGLang 等现代 AR serving 系统之上，直接复用已有优化：
     - **CUDA graph capture**
     - **paged KV cache**
     - **continuous batching**
   - 设计 **stationary-batch scheduler** 减少 CPU-GPU 同步开销；
   - 实现 **kernel fusion**（如将验证步骤融合进一个 Triton kernel）以降低延迟。

---

### **相比现有方法的优势**

| 维度 | I-DLM | 典型 DLM（如 SDAR、LLaDA） |
|------|-------|-----------------------------|
| **生成质量** | ✅ 匹配同规模 AR 模型 | ❌ 显著低于 AR 模型 |
| **内省一致性** | ✅ 高（α ≈ 0.98） | ❌ 低（α < 0.7） |
| **计算效率** | ✅ Compute efficiency >1 | ❌ OH 高，TPF 增益被抵消 |
| **系统兼容性** | ✅ 可集成进 AR serving 栈 | ❌ 需定制 pipeline |
| **吞吐量** | ✅ 高并发下 3×+ 提升 | ❌ 扩展性差 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共覆盖 **15 个基准测试**，分为四类：

| 类别 | 数据集 |
|------|--------|
| **知识与推理** | ARC-C, MMLU, MMLU-Pro, GPQA-D, GPQA |
| **数学推理** | GSM8K, MATH-500, MathBench, AIME-24, AIME-25 |
| **代码生成** | HumanEval, MBPP, LiveCodeBench-v6 (LCB-v6) |
| **指令遵循** | IFEval |

所有任务均启用 `thinking` 模式（允许模型进行链式思考）。

---

### **实验设置与评估指标**

#### **模型配置**
- **I-DLM-8B / I-DLM-32B**：由 Qwen3-8B / Qwen3-32B 转换而来。
- 训练数据量：仅 **4.5B tokens**（远少于 SDAR 的 54B）。
- 使用 DeepSpeed ZeRO-2，bf16 混合精度。

#### **评估指标**
| 指标类型 | 指标名称 | 说明 |
|--------|--------|------|
| **质量** | Accuracy / pass@1 | 各任务标准准确率 |
| **效率** | Tokens Per Second (TPS) | 请求级吞吐（latency）和服务器级吞吐（throughput） |
| **一致性** | Introspective Acceptance Rate (α) | 衡量模型是否“认同”自己生成的内容 |

#### **硬件环境**
- GPU：NVIDIA H100 80GB SXM（部分使用 B200）
- 推理框架：SGLang
- CUDA Graph 启用，FlashInfer 加速 attention

---

### **基线方法对比**

| 类型 | 基线模型 |
|------|---------|
| **DLMs** | LLaDA-2.1-mini (16B), SDAR (8B), NBDiff (7B), WeDLM (8B), TiDAR (8B), Fast-dLLM (7B), Mercury Coder Small, Gemini Diffusion |
| **Speculative Decoding** | EAGLE-3（基于辅助 draft model 的 AR 加速） |
| **AR 基线** | Qwen3-8B / Qwen3-32B（原生 AR 模型） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 模型 | AIME-24 | AIME-25 | LCB-v6 | MATH-500 | HumanEval |
|------|--------|--------|--------|----------|-----------|
| **I-DLM-8B** | **69.6** | 60.8 | **45.7** | **96.8** | **93.3** |
| LLaDA-2.1-mini (16B) | 43.3 | 63.3 | 45.4 | 85.0 | 86.0 |
| SDAR (8B) | 10.0 | 10.0 | 16.6 | 78.6 | 78.7 |
| **Qwen3-8B (AR)** | 73.1 | 65.4 | 50.3 | 95.8 | 95.1 |

> ✅ **I-DLM-8B 在多个任务上接近甚至超越同规模 AR 模型，且大幅领先所有其他 DLM。**

---

### **与基线方法的对比结果**

#### **质量方面**
- I-DLM-8B 在 **AIME-24 上比 LLaDA-2.1-mini 高出 26.3 分**，比 SDAR 高出近 **60 分**。
- 在 **LiveCodeBench-v6 上达到 45.7**，超过 SDAR（16.6）两倍以上。
- I-DLM-32B 更是 **全面超越百亿美元级模型 LLaDA-2.1-flash (100B)**。

#### **效率方面（图 5）**
- 在并发数 C=32 时：
  - I-DLM 吞吐达 **~340 tok/s**；
  - 比 LLaDA-2.1-mini 快 **3.1×**；
  - 比 SDAR 快 **4.0×**；
- 即使与 **EAGLE-3（speculative decoding）** 对比，I-DLM 仍快 **~1.4×**，且无需额外 draft model。

#### **内省一致性（α）**
| 模型 | α (Introspective Acceptance Rate) |
|------|----------------------------------|
| AR (Qwen3-8B) | 1.0（理论最优） |
| **I-DLM-8B** | **0.984** |
| LLaDA 2.1 (16B) | 0.933 |
| SDAR (8B) | 0.699 |

> ✅ I-DLM 实现了接近 AR 模型的一致性，而其他 DLM 差距明显。

---

### **消融实验结果**

#### **训练设计消融（图 6a）**
移除任一组件都会导致性能下降：
- 移除 causal attention + logit shift → HumanEval 下降至 **60.3**（原为 92.7）；
- 表明 **introspective consistency 是长程推理的关键**。

#### **系统优化消融（图 6b）**
在 C=32 时各优化贡献如下：
- **CUDA graph capture**：+42% ~ +76%
- **stationary-batch decode loop**：+11% ~ +21%
- **argmax proposals**：+11% ~ +15%
- **paged-only attention**：+10% ~ +14%

> ✅ 系统级优化带来 **2.1–2.5× 的端到端吞吐提升**。

#### **不同 stride 大小的影响（表 3）**
| N | TPF | MATH-500 | MBPP |
|---|-----|----------|------|
| 2 | 1.80 | 96.8 | 93.4 |
| 3 | 2.48 | 95.8 | 92.8 |
| 4 | 2.96 | 96.8 | 92.2 |
| 8 | 4.01 | 94.6 | 88.3 |

> ✅ 随着 stride 增大，TPF 几乎线性增长，质量保持稳定。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Introspective Consistency 是 DLM 落后于 AR 的根本原因**：
   - AR 模型天然具备生成与验证一致的能力；
   - I-DLM 通过 causal masking + logit shift 显式恢复该性质。

2. **高质量并行生成是可以实现的**：
   - I-DLM 是首个在 **质量上匹配同规模 AR 模型** 的 DLM；
   - 同时实现 **更高的吞吐量** 和 **更低的推理成本**。

3. **系统与算法必须协同设计**：
   - ISD 算法结构兼容 AR serving 架构；
   - 可直接继承 SGLang 的优化，避免重复造轮子。

4. **数据效率极高**：
   - 仅需 **4.5B tokens** 即可完成转换训练；
   - 相比 SDAR（54B tokens）节省 **12 倍数据**。

---

### **方法的局限性**
- 当前 ISD 的最大 stride 受限于训练时的最大跨度（需继续训练才能扩展）；
- Lossless R-ISD 引入 LoRA，略微增加内存占用；
- 目前主要适配 Qwen 系列模型，泛化性有待进一步验证。

---

### **未来工作方向**
- 将 I-DLM 思路推广至更多架构（如 Llama、Phi）；
- 探索更高效的 stride 扩展机制（无需重新训练）；
- 结合 multi-token prediction 与 introspective decoding；
- 应用于图像、音频等多模态 diffusion 模型中的“一致性”研究。

---

> 🔚 **总结一句话**：  
> **I-DLM 成功弥合了 DLM 与 AR 模型之间的质量和效率鸿沟，首次实现了“既快又准”的并行文本生成，并为下一代高效 LLM 推理提供了可部署的新范式。**

</details>

---

### 13. [End-to-end Automated Deep Neural Network Optimization for PPG-based Blood Pressure Estimation on Wearables](https://arxiv.org/abs/2604.10117)

**Authors**: Francesco Carlucci, Giovanni Pollo, Xiaying Wang, Massimo Poncino, Enrico Macii, Luca Benini, Sara Vinco, Alessio Burrello, Daniele Jahier Pagliari  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.10117v1  

#### Abstract
Photoplethysmography (PPG)-based blood pressure (BP) estimation is a challenging task, particularly on resource-constrained wearable devices. However, fully on-board processing is desirable to ensure user data confidentiality. Recent deep neural networks (DNNs) have achieved high BP estimation accur...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：End-to-end Automated Deep Neural Network Optimization for PPG-based Blood Pressure Estimation on Wearables

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **PPG-based Blood Pressure (BP) estimation** 是一项极具挑战性的任务，尤其是在资源受限的可穿戴设备上。
- 现有的深度神经网络（DNN）模型虽然精度高，但通常具有**高参数量、高计算复杂度和高能耗**，难以在低功耗 SoC 上实现完全**片上处理（on-board processing）**。
- 当前商业方案多依赖云端处理，带来**延迟、隐私泄露和实时性差**等问题。

### 🚀 提出的新方法与创新
本文提出了一种**端到端自动化 DNN 优化流水线**，结合以下三个关键技术：
1. **硬件感知的神经架构搜索（Hardware-aware Neural Architecture Search, NAS）**  
   - 使用梯度驱动的 SuperNet 方法自动选择最优层组合（如标准卷积、depthwise separable convolution、identity mapping），并优化网络深度。
2. **结构化剪枝（Structured Pruning, PIT）**  
   - 通过可训练掩码移除冗余的输出通道、滤波器尺寸或调整膨胀率，进一步压缩模型。
3. **混合精度搜索（Mixed-Precision Search, MPS）**  
   - 对权重和激活进行整数量化（int2, int4, int8），生成可在无 FPU 的嵌入式芯片上高效运行的模型。

此外，还引入了**患者特异性微调（Patient-specific Fine-tuning）** 流程，在部署后进一步提升个体用户的预测准确性。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **自动化程度** | 完全自动化流程，无需人工干预设计决策 |
| **硬件适配性** | 显式考虑内存、算力等硬件约束，确保模型可部署于 ultra-low-power SoC |
| **精度-效率权衡** | 在显著降低模型大小的同时保持甚至提升预测精度 |
| **隐私保护** | 支持全设备端推理，避免原始生理信号外传 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
研究基于四个公开的 PPG-BP 数据集，并采用统一预处理协议（来自 [27]）以保证公平比较：

| 数据集 | 样本数 | 特点 |
|-------|--------|------|
| **Sensors** | ~1,195 ICU 患者 | 来自 MIMIC-III，含 PPG 和 ABP 波形 |
| **UCI** | 最大 | 来自 MIMIC-I Waveform DB，数据丰富但无受试者 ID |
| **BCG** | 40 名患者 | 高采样率（125Hz），每名患者约 75 个样本，适合 fine-tuning 实验 |
| **PPGBP** | 219 名患者 | 小型数据集，仅提供离散 SBP/DBP 值，不含完整 ABP 波形 |

> 所有数据均按**受试者划分（subject-wise split）** 进行 5 折交叉验证，防止数据泄露。

### 🧪 实验设置与评估指标

#### 评估指标
- **Mean Absolute Error (MAE)**：主评价指标
  - $ \text{MAE}_{\text{SBP}} = \mathbb{E}[|\text{SBP}_{\text{true}} - \text{SBP}_{\text{pred}}|] $
  - $ \text{MAE}_{\text{DBP}} = \mathbb{E}[|\text{DBP}_{\text{true}} - \text{DBP}_{\text{pred}}|] $
- **模型大小（参数量 / 内存占用）**
- **推理延迟（Latency）**
- **能量消耗（Energy Consumption）**

#### 部署平台
- **目标硬件**：GreenWaves GAP8 SoC（RISC-V 多核 ultra-low-power 芯片）
  - 主频：100 MHz
  - 片上内存：512 KB L2 Scratchpad
  - 支持 SIMD 加速
- **部署工具链**：DORY + PULP-NN / PULP-NN-Mixed 库
- **量化支持**：int8、mixed-precision（int2/int4/int8）

#### 基线方法对比
- **经典机器学习模型**：Random Forest (RF)、Support Vector Regression (SVR)
- **DNN 基线模型**：ResNet、U-Net（来自 [27] 的 SOTA 模型）
- **其他先进模型**：Conv-Transformer [44]、V-Net 等

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **最大误差降低** | 达到比 SOTA DNN **低 7.99% MAE** |
| **参数压缩比** | 最高达 **83× 参数减少**，且精度损失可忽略 |
| **模型大小** | 所有优化后模型 < **55 KB**，远低于 GAP8 的 512 KB 限制 |
| **平均推理延迟** | **142.14 ms**（最小 52.87 ms，最大 241.33 ms） |
| **平均能量消耗** | **7.25 mJ/inference**（最低 2.70 mJ） |
| **fine-tuning 提升** | 患者个性化微调使 MAE 下降最高达 **64.27%（DBP）** 和 **61.1%（SBP）** |

> ⚠️ 注：所有模型均可在 GAP8 上独立运行，无需外部存储或云连接。

### 🔁 与基线方法的对比结果（摘自 Table 4）

| 数据集 | 指标 | 本文最优模型 | SOTA 基线 | 表现 |
|-------|------|--------------|-----------|--------|
| **BCG** | DBP MAE | **7.26 mmHg** | 7.34 (SVR) | ✅ 更优 |
|        | SBP MAE | **11.07 mmHg** | 11.45 (SVR) | ✅ 更优 |
|        | 参数量 | **64.8k** | 491k (V-Net) | ✅ 减少 7.5× |
| **Sensors** | DBP MAE | **7.50 mmHg** | 7.50 (SVR) | ✅ 相当 |
|           | SBP MAE | **15.51 mmHg** | 15.60 (SVR) | ✅ 略优 |
|           | 参数量 | **41.2k** | 416k (SVR) | ✅ 减少 10× |
| **UCI** | DBP MAE | **7.69 mmHg** | 8.07 (SVR) | ✅ 显著更优 |
|         | SBP MAE | **16.32 mmHg** | 16.85 (RF) | ✅ 显著更优 |
|         | 参数量 | **8.43k** | >4M (SVR) | ✅ 减少超 500× |

> 💡 在 UCI 这类大数据集中，DNN 优势明显；而在小数据集 PPGBP 中，传统 ML（如 SVR）仍占优。

### 🔍 消融实验结果（Ablation Study）

| 优化阶段 | 效果说明 |
|--------|----------|
| **NAS** | 在多个数据集上生成帕累托前沿更优的架构，例如在 BCG 上将 ResNet 压缩 3.8× 并提升精度 |
| **Pruning (PIT)** | 进一步压缩模型规模，如在 BCG 上实现 **7.5× 参数减少 + 7.99% MAE 降低** |
| **Quantization (MPS/QAT)** | 实现整数部署，内存需求下降至原浮点模型的 **1/4～1/10**，能效大幅提升 |
| **Fine-tuning** | 微调后 DBP MAE 从 7.51 → **2.68 mmHg**（降幅 64.27%），达到接近临床级精度 |

> 📌 **最佳部署模型满足 AAMI 协议核心标准**（ME ≤ 5 mmHg, STD ≤ 8 mmHg），其中 DBP ME = **1.39 mmHg**, STD = **2.36 mmHg**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **自动化 DNN 压缩可行且有效**：通过 NAS + Pruning + MPS 的级联优化，可以在不牺牲精度的前提下大幅压缩模型。
2. **优化后的模型适合边缘部署**：所有模型均能在 GAP8 上实时运行（<250ms/inference），支持连续监测。
3. **个性化微调极大提升精度**：即使只用 20% 数据微调，也能获得超过 50% 的误差下降，具备临床应用潜力。
4. **DNN 在大规模数据下优于传统 ML**：尤其在 UCI 数据集上，本文方法全面超越 RF/SVR，证明了深度学习在复杂场景中的泛化能力。

### ⚠️ 局限性
- **数据集偏差**：BCG 数据仅来自 40 名患者，代表性有限；PPGBP 缺乏完整 ABP 波形，无法用于 signal-to-signal 模型训练。
- **fine-tuning 依赖高质量标签**：需要用户提供一次准确的血压测量作为校准输入，可能影响用户体验。
- **未测试运动伪影鲁棒性**：当前实验基于静态采集数据，尚未验证在真实动态环境下的稳定性。

### 🔮 未来工作方向
1. **自适应在线学习（Adaptive On-device Learning）**：实现在设备端持续更新模型，应对长期生理变化。
2. **多模态融合（Multi-sensor Fusion）**：结合 IMU、ECG、体温等信号，提升抗干扰能力和估计鲁棒性。
3. **跨设备迁移能力研究**：探索模型在不同品牌/型号传感器间的泛化性能。
4. **符合医疗认证标准的完整验证**：开展符合 AAMI/ISO 81060-2 的大规模临床试验。

---

## 总结

✅ 本文成功构建了一个**全自动、硬件感知的 DNN 优化框架**，实现了高性能、低功耗、小体积的 PPG-BP 估算模型，推动了**真正意义上的隐私安全、全天候可穿戴血压监测系统**的发展。  
🎯 其方法不仅适用于 BP 估计，也为其他生物信号处理任务提供了可复用的边缘 AI 设计范式。

</details>

---

### 14. [Tracing the Roots: A Multi-Agent Framework for Uncovering Data Lineage in Post-Training LLMs](https://arxiv.org/abs/2604.10480)

**Authors**: Yu Li, Xiaoran Shang, Qizhi Pei, Yun Zhu, Xin Gao, Honglin Lin, Zhanping Zhong, Zhuoshi Pan, Zheng Liu, Xiaoyang Wang, Conghui He, Dahua Lin, Feng Zhao, Lijun Wu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.10480v1  

#### Abstract
Post-training data plays a pivotal role in shaping the capabilities of Large Language Models (LLMs), yet datasets are often treated as isolated artifacts, overlooking the systemic connections that underlie their evolution. To disentangle these complex relationships, we introduce the concept of \text...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Tracing the Roots: A Multi-Agent Framework for Uncovering Data Lineage in Post-Training LLMs*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前大语言模型（LLM）的 **post-training data** 构建缺乏系统性的溯源机制，导致两个核心风险：
- **结构性冗余（Structural Redundancy）**：多个看似独立的数据集因隐式继承相同上游来源而语义趋同，削弱了数据规模增长的实际价值。
- **基准污染传播（Benchmark Contamination Propagation）**：测试集样本通过上游数据集被无意继承，造成评估结果“虚假提升”，损害模型可信度。

现有方法多依赖于 **sample-level 扫描**（如 N-gram 匹配、语义嵌入），效率低且难以追踪跨数据集的污染路径。

### 提出的新方法与新思路
本文提出 **data lineage** 概念，并构建了一个 **multi-agent 协作框架** 来自动重构 post-training 数据集的演化图谱（evolutionary graph）：
- 将数据集视为节点（Node），继承关系视为有向边（Edge），形成一个 **directed graph G=(V, E)**。
- 引入 **leaf node**（无上游来源）和 **internal node**（有可追溯来源）的概念，界定自动化探索边界。
- 设计四阶段递归流水线：
  1. **Candidate Validation**：验证可用性并统一发布日期。
  2. **Multi-source Information Retrieval**：从 HuggingFace README、GitHub、arXiv 等获取非结构化文档。
  3. **Semantic Source Inference**：利用 LLM agents 抽取 `<Source, Relationship, Confidence, Evidence>` 四元组。
  4. **Aggregation & Recursive Expansion**：去重、标准化 ID、验证时间顺序，并递归扩展上游。

### 相比现有方法的优势
| 维度 | 传统方法 | 本文方法 |
|------|--------|---------|
| **分析粒度** | Sample-level | **Topological-level**（基于 lineage 结构） |
| **效率** | 需扫描百万级样本，成本高 | 仅需分析少量 provenance 文档 |
| **鲁棒性** | 易受 paraphrasing、semantic drift 影响 | 依赖依赖关系而非文本相似性，更稳健 |
| **可解释性** | 黑箱匹配 | 可追溯污染/冗余的具体路径 |
| **扩展性** | 难以规模化 | 支持实时追踪新出现的数据集 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **种子数据集（Seed Datasets）**：共 83 个高影响力数据集，覆盖四个领域：
  - **General**（如 `alpaca`, `FLAN`, `OpenHermes-2.5`）
  - **Math**（如 `gsm8k`, `hendrycks_math`, `MetaMathQA`）
  - **Code**（如 `apps`, `code_contests`）
  - **Science**（如 `SciBench`, `camel-ai/biology`）
- 最终构建的 **lineage graph** 包含 **430 个唯一节点** 和 **971 条继承边**。

### 实验设置
- **框架实现**：基于 LangChain 构建 workflow。
- **LLM Agents**：
  - **Sourcing / Tracing Agent**：GPT-5.1（高精度）
  - **Extracting Agent**：Gemini-2.5-Flash（高速处理）
  - **Aggregation Agent**：Gemini-2.5-Pro（强推理+检索能力）
- **时间范围**：限定为 2020 年后发布的数据集（对应 GPT-3 时代）。
- **人工校验机制**：对低置信度抽取结果进行专家复核，防止 LLM hallucination。

### 评估指标
1. **拓扑统计指标**：
   - Average Depth
   - In-Degree / Out-Degree
   - Leaf Node Ratio
2. **冗余率（Redundancy Rate）**：
   - 基于 `(instruction, input, output)` 三元组的精确匹配计算重复比例。
3. **污染率（Contamination Rate）**：
   - 对比训练数据与多个 benchmark（如 Omni-MATH, TheoremQA, LiveCodeBench）之间的样本重叠。
4. **多样性指标**：
   - **Vendi Score**：衡量语义簇的有效数量（越高越好）
   - **Centroid Distance**：衡量指令在嵌入空间中的分散程度（越高越好）

### 基线方法对比
- 多个主流 post-training 数据集作为 baseline：
  - `OpenHermes-2.5`
  - `tulu-3-sft-mixture`
  - `MegaScience`
  - `OpenThoughts3`
  - `herculesv1`

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）拓扑结构分析结果（Table 1）
| Domain | Nodes | Avg. Depth | Out-Deg. | Leaf % |
|-------|-------|------------|----------|--------|
| Math | 99 | **2.92** | 1.54 | 38.38% |
| Code | 98 | 2.12 | 1.36 | 43.88% |
| General | 285 | 1.05 | 1.29 | **68.42%** |
| Science | 44 | 2.82 | 1.25 | 47.73% |

> 发现：数学领域呈现 **vertical refinement（深度精炼）** 范式；通用领域则是 **horizontal accumulation（横向聚合）**。

#### （2）冗余分析（Table 3）
- `open-instruct-v1` 冗余率达 **46.48%**（因其重复包含其超集 `self_instruct`）。
- `Fast-Math-R1-SFT` 存在 **5.30%** 的冗余，源于同时引入 `OpenR1-Math-220k` 及其父集。

#### （3）污染传播分析（Figure 6 & Table 10）
- `DeepScaleR-Preview-Dataset` 对 Omni-MATH 的污染率高达 **79.48%**。
- `Caco-1.3M` 虽未直接包含 Omni-MATH，但通过污染的中间体间接继承 **37.95%** 的样本。
- 共发现 **19 个数据集** 在多个 benchmark 上存在不同程度的污染。

#### （4）多样性对比（Table 4）
| Dataset | Size | Vendi Score | Centroid Dist. |
|--------|------|-------------|----------------|
| Ours (Provenance-based) | **570K** | **452.44** | **0.6385** |
| OpenHermes-2.5 | 615K | 437.76 | 0.6271 |
| MegaScience | 1.2M | 373.78 | 0.6150 |
| OpenThoughts3 | 1.2M | 133.26 | 0.4970 |

> 结论：**即使规模更小，provenance-based sampling 方法仍显著优于更大规模的混合数据集**，证明“质量 > 数量”。

---

## 4. 关键结论和发现

### 主要发现
1. **领域演化范式差异显著**：
   - 数学：依赖少数核心锚点（如 `hendrycks_math`, `gsm8k`）进行多代递归精炼。
   - 通用：广泛聚合，趋于饱和（2025 年新 leaf node 锐减）。
   - 科学：极度稀缺（仅 44 节点），严重依赖其他领域资源。

2. **代码是连接通用与数学的关键桥梁**：
   - 同时吸收来自 general（语言表达）和 math（逻辑推理）的信息，扮演“操作化”角色。

3. **结构性问题普遍存在**：
   - 高达 **17/83** 的种子数据集存在明显冗余。
   - benchmark 污染具有强传染性，可通过 lineage 图精准溯源。

4. **lineage-aware 数据构建更高效**：
   - 从 root nodes 出发采样即可获得更高语义多样性，无需复杂过滤。

### 方法的局限性
1. **LLM Hallucination 风险**：尽管有验证机制，低置信抽取仍需人工干预。
2. **文档透明度依赖**：若作者未披露上游来源，则无法恢复真实 lineage。
3. **relation 类型推断有限**：部分文档描述模糊，relation 默认标记为 “Direct Inclusion”。

### 未来工作方向
- 构建开放的 **Data Lineage Registry**，推动社区共享 provenance 信息。
- 开发 **automated decontamination pipelines**，结合 lineage 与 content 扫描。
- 探索 **dynamic lineage tracking**，支持模型训练过程中的实时监控。
- 将 lineage 分析应用于 **multimodal data** 和 **model editing** 场景。

--- 

> **总结一句话**：  
> 本文首次系统性地将 **data lineage** 引入 LLM post-training 生态，提出一种高效的 **multi-agent 自动化溯源框架**，揭示了数据演化中的深层结构性问题，并展示了如何利用 lineage 指导更高质量、更多样化的数据构建，为下一代数据工程提供了范式转变。

</details>

---

### 15. [Beyond Compliance: A Resistance-Informed Motivation Reasoning Framework for Challenging Psychological Client Simulation](https://arxiv.org/abs/2604.10507)

**Authors**: Danni Liu, Bo Liu, Yuxin Hu, Hantao Zhao, Yan Liu, Ding Ding, Jiahui Jin, Jiuxin Cao  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.10507v1  

#### Abstract
Psychological client simulators have emerged as a scalable solution for training and evaluating counselor trainees and psychological LLMs. Yet existing simulators exhibit unrealistic over-compliance, leaving counselors underprepared for the challenging behaviors common in real-world practice. To bri...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前基于LLM的心理学**client simulator**普遍存在“过度顺从”（over-compliance）问题：即模拟的来访者过于开放、合作、情绪稳定，无法真实再现现实中常见的**client resistance**（来访者阻抗）行为。这种偏差导致：
- 心理咨询师培训场景失真；
- 心理学LLM的评估缺乏挑战性；
- 限制了对复杂治疗互动的理解与优化。

### 提出了什么新方法或新思路
本文提出 **ResistClient**，一个基于**Client Resistance Theory**的挑战性来访者模拟框架，并引入 **Resistance-Informed Motivation Reasoning (RIMR)** 两阶段训练范式：

1. **Resistance-Informed Supervised Fine-Tuning (SFT)**  
   构建大规模**Resistance-Informed Psychological Conversations (RPC)** 数据集，通过监督微调纠正预训练LLM的顺从偏见。

2. **Motivation Reasoning Reinforcement Learning (MRRL)**  
   引入显式的**动机推理过程**（motivation reasoning），在生成回应前进行三步结构化推理：
   - **Profile Reflection**：整合5P profile中的稳定心理特征；
   - **Situation Awareness**：分析当前对话情境与咨询师干预；
   - **Reaction Decision**：决定反应类型（如Defensive Resistance）与行为特征。

   并采用**process-supervised reinforcement learning**（基于GRPO算法）联合优化推理真实性与回应一致性。

### 相比现有方法的优势
| 维度 | 现有方法 | ResistClient |
|------|---------|-------------|
| **行为真实性** | 表层模仿，缺乏心理深度 | 基于理论的系统性建模 |
| **内部机制** | 黑箱响应生成 | 显式动机推理链 |
| **挑战多样性** | 单一或随机挑战（如情绪注入） | 多类型、情境适配的阻抗行为 |
| **文化适配性** | 通用设计 | 基于中国语境构建，体现间接性阻抗偏好 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **RPC Dataset**：本文构建的新数据集，包含：
  - **1,849** 场完整心理咨询会话；
  - 覆盖14个常见主题（如抑郁、人际关系、职业压力等）；
  - 每场会话附带经验证的**5P client profile**（Presenting Problems, Predisposing/Precipitating/Perpetuating/Protective Factors）；
  - **1,761** 场会话包含标注的阻抗行为，细分为5类阻抗 + 2类合作反应。

> 注：原始数据来自真实对话库 **ProPsyC**，通过**Resistance-Informed Conversation Rewriting**框架重写以增强阻抗表现。

### 实验设置和评估指标

#### 自动化评估指标
| 指标 | 含义 |
|------|------|
| **Precision / Recall / F1** | 阻抗生成的准确率、召回率与综合得分 |
| **RTF (Resistance Trigger Frequency)** | 阻抗触发频率 |
| **CCR (Client Cooperation Rate)** | 客户合作率（越低表示挑战性越强） |
| **Turns** | 对话轮次（越长表示交互难度越高） |
| **Coh. (Coherence)** | 基于嵌入向量的语义连贯性 |

#### 人工评估指标（0–3分制）
| 指标 | 含义 |
|------|------|
| **Fid. (Fidelity)** | 阻抗类型的保真度 |
| **Rat. (Rationality)** | 阻抗出现的情境合理性 |
| **Qua. (Quality)** | 动机推理的质量 |
| **Real. (Realism)** | 整体行为的真实性 |
| **Cons. (Consistency)** | 与profile的一致性 |

### 基线方法对比
#### 阻抗模拟能力对比（RQ1）
- **大型闭源模型**：GPT-5.1, DeepSeek-V3.2, Kimi-K2-thinking, GLM-4.6
- **开源小模型**：Qwen3-8B, DeepSeek-R1-8B
- **消融变体**：Qwen3-8B-SFT（仅SFT）

#### 挑战行为质量对比（RQ3）
- **Patient-V**：基础profile-conditioned模拟
- **AnnaAgent**：通过情绪扰动器注入随机情绪标签
- **Yang et al. (2025b)**：通过低接收度控制诱导挑战

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）

| 模型 | Precision (%) | Recall (%) | F1 (%) | RTF (%) | Fid. | Rat. | Qua. |
|------|---------------|------------|--------|---------|------|------|------|
| GPT-5.1 | 59.31 | 62.88 | 61.04 | 35.94 | 1.42 | 1.35 | 2.52 |
| DeepSeek-V3.2 | 52.87 | 57.56 | 55.12 | 36.91 | 1.29 | 1.21 | 2.40 |
| Qwen3-8B-SFT | 63.54 | 73.90 | **68.33** | 39.43 | 1.46 | 1.41 | 2.39 |
| **ResistClient** | **70.38** | **78.95** | **74.42** | **38.03** | **1.63** | **1.58** | **2.61** |

> ✅ **ResistClient 在所有指标上显著优于所有基线模型**

### 挑战行为质量对比（Table 2）

| 模型 | CCR (%) | Turns | Coh. | Real. | Cons. |
|------|--------|-------|------|-------|-------|
| Patient-V | 87.94 | 11.24 | 0.51 | 1.87 | 1.32 |
| AnnaAgent | 78.62 | 12.65 | 0.62 | 1.95 | 1.60 |
| Yang et al. | 62.33 | 16.67 | 0.68 | 2.01 | 1.83 |
| **ResistClient** | **60.84** | **17.88** | **0.73** | **2.39** | **1.75** |

> ✅ **最低合作率 + 最长对话轮数 + 最高连贯性与真实性**

### 消融实验结果（Ablation Study）
- **Prompt-only Qwen3-8B**：严重偏向合作性反应，混淆矩阵显示极低阻抗生成能力。
- **Qwen3-8B-SFT**：显著提升阻抗生成能力（F1达68.33），表明**RPC数据集本身具有强大增益**。
- **完整ResistClient（+MRRL）**：进一步提升至F1=74.42，尤其减少不同类型阻抗间的混淆，证明**动机推理机制对行为一致性和分类准确性至关重要**。

---

## 4. 关键结论和发现

### 主要发现
1. **阻抗行为必须基于心理理论建模**，而非简单的情绪扰动或信息隐藏。
2. **动机推理机制**（motivation reasoning）能显著提升行为的心理连贯性与情境适应性。
3. **ResistClient 成功实现了挑战强度与行为真实性的平衡**，既避免了“无脑对抗”，也克服了“过度顺从”。
4. 当前主流**心理学LLM在面对真实阻抗时普遍表现不佳**：易引发阻抗（RTF高达39–52%）、咨询漂移（CDD高）、进展有限（CPD低），凸显了高质量挑战性训练环境的必要性。

### 方法的局限性
1. **文化局限性**：数据集基于中国心理咨询语境构建，阻抗类型分布（如Compliant Resistance最常见）可能不适用于其他文化背景。
2. **评估规模有限**：依赖少数专家评估，可能存在视角偏差。
3. **单边模拟**：仅关注来访者侧，未建模咨询师如何有效应对阻抗。
4. **伦理约束**：无法使用真实患者数据，所有对话为合成生成。

### 未来工作方向
1. 扩展至**跨文化心理咨询场景**，构建多语言、多文化版本的RPC数据集。
2. 将ResistClient部署为**可扩展的虚拟培训平台**，服务于更广泛的心理咨询师群体。
3. 开发具备**阻抗管理能力的咨询师代理**（counselor agent），实现双向动态适应。
4. 探索**个性化阻抗演化建模**，模拟阻抗随疗程推进的变化轨迹。

> 🔍 总结：**ResistClient 不仅是一个更真实的client simulator，更是推动心理学LLM走向临床可靠性的关键基础设施**。

</details>

---

### 16. [ZoomR: Memory Efficient Reasoning through Multi-Granularity Key Value Retrieval](https://arxiv.org/abs/2604.10898)

**Authors**: David H. Yang, Yuxuan Zhu, Mohammad Mohammadi Amiri, Keerthiram Murugesan, Tejaswini Pedapati, Subhajit Chaudhury, Pin-Yu Chen  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.10898v1  

#### Abstract
Large language models (LLMs) have shown great performance on complex reasoning tasks but often require generating long intermediate thoughts before reaching a final answer. During generation, LLMs rely on a key-value (KV) cache for autoregressive decoding. However, the memory footprint of the KV cac...

---

### 17. [MADQRL: Distributed Quantum Reinforcement Learning Framework for Multi-Agent Environments](https://arxiv.org/abs/2604.11131)

**Authors**: Abhishek Sawaika, Samuel Yen-Chi Chen, Udaya Parampalli, Rajkumar Buyya  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.11131v1  

#### Abstract
Reinforcement learning (RL) is one of the most practical ways to learn from real-life use-cases. Motivated from the cognitive methods used by humans makes it a widely acceptable strategy in the field of artificial intelligence. Most of the environments used for RL are often high-dimensional, and tra...

---

### 18. [UniToolCall: Unifying Tool-Use Representation, Data, and Evaluation for LLM Agents](https://arxiv.org/abs/2604.11557)

**Authors**: Yijuan Liang, Xinghao Chen, Yifan Ge, Ziyi Wu, Hao Wu, Changyu Zeng, Wei Xing, Xiaoyu Shen  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.11557v1  

#### Abstract
Tool-use capability is a fundamental component of LLM agents, enabling them to interact with external systems through structured function calls. However, existing research exhibits inconsistent interaction representations, largely overlooks the structural distribution of tool-use trajectories, and r...

---

### 19. [Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](https://arxiv.org/abs/2604.10072)

**Authors**: Chao Xue, Yao Wang, Mengqiao Liu, Di Liang, Xingsheng Han, Peiyang Liu, Xianjie Wu, Chenyao Lu, Lei Jiang, Yu Lu, Haibo Shi, Shuang Liang, Minlong Peng, Flora D. Salim  
**Category**: cs.CL  
**Published**: 2026-04-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.10072v1  

#### Abstract
Recent advancements in the Generative Reward Model (GRM) have demonstrated its potential to enhance the reasoning abilities of LLMs through Chain-of-Thought (CoT) prompting. Despite these gains, existing implementations of GRM suffer from two critical limitations. First, CoT prompting is applied ind...

---

### 20. [HTAA: Enhancing LLM Planning via Hybrid Toolset Agentization & Adaptation](https://arxiv.org/abs/2604.10917)

**Authors**: Chengrui Huang, Junshuo Zhang, Zhiyuan Ma, Xikun Wang, Ximeng Wang, Menghua Jiang, Gang Zeng, Zhaobing Han, Shen Gao, Shuo Shang  
**Category**: cs.CL  
**Published**: 2026-04-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.10917v1  

#### Abstract
Enabling large language models to scale and reliably use hundreds of tools is critical for real-world applications, yet challenging due to the inefficiency and error accumulation inherent in flat tool-calling architectures. To address this, we propose Hybrid Toolset Agentization & Adaptation (HTAA),...

---

### 21. [Efficient Matrix Implementation for Rotary Position Embedding](https://arxiv.org/abs/2604.09742)

**Authors**: Chen Minqi, Zhongqi Yue, Shihao Zhang, Yun Xu, Peng Wu, kaixiang Xu, Zeyi Huang, Hanwang Zhang  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.09742v1  

#### Abstract
Rotary Position Embedding (RoPE) has become a core component of modern Transformer architectures across language, vision, and 3D domains. However, existing implementations rely on vector-level split and merge operations that introduce non-negligible computational overhead, often overlooked in attent...

---

### 22. [NeuroFlow: Toward Unified Visual Encoding and Decoding from Neural Activity](https://arxiv.org/abs/2604.09817)

**Authors**: Weijian Mai, Mu Nan, Yu Zhu, Jiahang Cao, Rui Zhang, Yuqin Dai, Chunfeng Song, Andrew F. Luo, Jiamin Wu  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.09817v1  

#### Abstract
Visual encoding and decoding models act as gateways to understanding the neural mechanisms underlying human visual perception. Typically, visual encoding models that predict brain activity from stimuli and decoding models that reproduce stimuli from brain activity are treated as distinct tasks, requ...

---

### 23. [Robust Adversarial Policy Optimization Under Dynamics Uncertainty](https://arxiv.org/abs/2604.10974)

**Authors**: Mintae Kim, Koushil Sreenath  
**Category**: cs.LG  
**Published**: 2026-04-14  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.10974v1  

#### Abstract
Reinforcement learning (RL) policies often fail under dynamics that differ from training, a gap not fully addressed by domain randomization or existing adversarial RL methods. Distributionally robust RL provides a formal remedy but still relies on surrogate adversaries to approximate intractable pri...

---

### 24. [Hubble: An LLM-Driven Agentic Framework for Safe and Automated Alpha Factor Discovery](https://arxiv.org/abs/2604.09601)

**Authors**: Runze Shi, Shengyu Yan, Yuecheng Cai, Chengxi Lv  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.09601v1  

#### Abstract
Discovering predictive alpha factors in quantitative finance remains a formidable challenge due to the vast combinatorial search space and inherently low signal-to-noise ratios in financial data. Existing automated methods, particularly genetic programming, often produce complex, uninterpretable for...

---

### 25. [Competing with AI Scientists: Agent-Driven Approach to Astrophysics Research](https://arxiv.org/abs/2604.09621)

**Authors**: Thomas Borrett, Licong Xu, Andy Nilipour, Boris Bolliet, Sebastien Pierre, Erwan Allys, Celia Lecat, Biwei Dai, Po-Wen Chang, Wahid Bhimji  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.09621v1  

#### Abstract
We present an agent-driven approach to the construction of parameter inference pipelines for scientific data analysis. Our method leverages a multi-agent system, Cmbagent (the analysis system of the AI scientist Denario), in which specialized agents collaborate to generate research ideas, write and ...

---

### 26. [New Hybrid Fine-Tuning Paradigm for LLMs: Algorithm Design and Convergence Analysis Framework](https://arxiv.org/abs/2604.09940)

**Authors**: Shaocong Ma, Peiran Yu, Heng Huang  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.09940v1  

#### Abstract
Fine-tuning Large Language Models (LLMs) typically involves either full fine-tuning, which updates all model parameters, or Parameter-Efficient Fine-Tuning (PEFT), which adjusts a small subset of parameters. However, both approaches have inherent limitations: full fine-tuning is computationally expe...

---

### 27. [Zero-shot World Models Are Developmentally Efficient Learners](https://arxiv.org/abs/2604.10333)

**Authors**: Khai Loong Aw, Klemen Kotar, Wanhee Lee, Seungwoo Kim, Khaled Jedoui, Rahul Venkatesh, Lilian Naing Chen, Michael C. Frank, Daniel L. K. Yamins  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.10333v1  

#### Abstract
Young children demonstrate early abilities to understand their physical world, estimating depth, motion, object coherence, interactions, and many other aspects of physical scene understanding. Children are both data-efficient and flexible cognitive systems, creating competence despite extremely limi...

---

### 28. [TrajOnco: a multi-agent framework for temporal reasoning over longitudinal EHR for multi-cancer early detection](https://arxiv.org/abs/2604.10386)

**Authors**: Sihang Zeng, Young Won Kim, Wilson Lau, Ehsan Alipour, Ruth Etzioni, Meliha Yetisgen, Anand Oka  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.10386v1  

#### Abstract
Accurate estimation of cancer risk from longitudinal electronic health records (EHRs) could support earlier detection and improved care, but modeling such complex patient trajectories remains challenging. We present TrajOnco, a training-free, multi-agent large language model (LLM) framework designed...

---

### 29. [FedRio: Personalized Federated Social Bot Detection via Cooperative Reinforced Contrastive Adversarial Distillation](https://arxiv.org/abs/2604.10678)

**Authors**: Yingguang Yang, Hao Liu, Xin Zhang, Yunhui Liu, Yutong Xia, Qi Wu, Hao Peng, Taoran Liang, Bin Chong, Tieke He, Philip S. Yu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.10678v1  

#### Abstract
Social bot detection is critical to the stability and security of online social platforms. However, current state-of-the-art bot detection models are largely developed in isolation, overlooking the benefits of leveraging shared detection patterns across platforms to improve performance and promptly ...

---

### 30. [From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](https://arxiv.org/abs/2604.11137)

**Authors**: Chen Zhan, Xiaoyu Tan, Gengchen Ma, Yu-Jie Xiong, Xiaoyan Jiang, Xihe Qiu  
**Category**: cs.AI  
**Published**: 2026-04-14  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.11137v1  

#### Abstract
The integration of Large Language Models (LLMs) into clinical decision support is critically obstructed by their opaque and often unreliable reasoning. In the high-stakes domain of healthcare, correct answers alone are insufficient; clinical practice demands full transparency to ensure patient safet...

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
