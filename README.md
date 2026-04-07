# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-07 07:03:28 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Communication-free Sampling and 4D Hybrid Parallelism for Scalable Mini-batch GNN Training](https://arxiv.org/abs/2604.02651)

**Authors**: Cunyang Wei, Siddharth Singh, Aishwarya Sarkar, Daniel Nichols, Tisha Patel, Aditya K. Ranjan, Sayan Ghosh, Ali Jannesari, Nathan R. Tallent, Abhinav Bhatele  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 12.0  
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
- **采样开销高**：主流方法如 GraphSAGE 和 GraphSAINT 需要跨设备通信进行邻居采样或归一化系数计算，导致严重的通信延迟。
- **扩展性差**：依赖传统的 **data parallelism**，难以高效利用大规模 GPU 集群，训练时间无法随 GPU 数量增加而有效减少。

### 提出的新方法与创新思路
作者提出了 **ScaleGNN** —— 一个开源的、支持可扩展 mini-batch GNN 训练的 **4D 并行框架**，其核心创新包括：

#### ✅ **Communication-free 分布式采样算法**
- 基于 **uniform vertex sampling**，每个 GPU 可独立地从本地图分片中采样顶点并构建子图（subgraph），无需任何进程间通信。
- 引入 **unbiased edge rescaling** 技术（通过全局常数 $ p = (B-1)/(N-1) $ 调整边权重），保证 mini-batch 聚合是全图聚合的无偏估计。

#### ✅ **4D Hybrid Parallelism 架构**
将总 GPU 组织成四维虚拟网格 $ G_d \times G_x \times G_y \times G_z $：
- **Data Parallelism ($G_d$)**：不同组处理不同的 mini-batch，梯度通过 all-reduce 同步。
- **3D Parallel Matrix Multiplication (3D PMM)**：在每组内部使用三维张量并行来分布 SpMM 和 GEMM 运算，显著降低通信开销。

#### ✅ 多项系统级优化
- **采样与训练流水线重叠**：用独立 CUDA stream 预取下一个 mini-batch，消除采样对关键路径的影响。
- **低精度集体通信**：仅对 3D PMM 中的 all-reduce 使用 BF16，保留数值敏感操作为 FP32。
- **Kernel fusion**：融合 RMSNorm、ReLU、Dropout 等 element-wise 操作以减少内存访问。
- **通信-计算重叠**：在反向传播中并行执行正交维度上的 all-reduce。

### 相比现有方法的优势
| 方面 | ScaleGNN | 传统方法（如 DistDGL、MassiveGNN） |
|------|---------|-------------------------------|
| 采样方式 | 无通信、GPU端统一顶点采样 | CPU端采样 + 跨节点特征获取 |
| 并行策略 | 4D hybrid（数据 + 3D 张量） | 单纯 data parallelism |
| 图数据布局 | 每个 DP 组内完整保存图（via 3D PMM） | 图被分割，需远程访问邻居 |
| 扩展能力 | 支持数千 GPU 强扩展 | 扩展后 epoch 数上升，end-to-end 时间反而变长 |

---

## 2. 核心实验方法和设置

### 使用的数据集
共五种图数据集，涵盖不同规模与领域：
| 数据集 | 类型 | 节点数 | 边数 | 任务 |
|-------|------|--------|------|-----|
| **ogbn-products** | 商品分类 | 2.4M | 60M | 节点分类 |
| **Reddit** | 社区分类 | 233K | 58M | 节点分类 |
| **ogbn-papers100M** | 引用网络 | 111M | 1.6B | 节点分类 |
| **Products-14M** | 商品网络 | 14M | 115M | 合成标签分类 |
| **Isolate-3-8M** | 蛋白质相似性 | 3.8M | - | 合成标签分类 |

> 注：后两者用于测试扩展性，输入特征随机生成，类别按度分布分配。

### 实验平台
- **Perlmutter**：NVIDIA A100 GPU，最多使用 2048 GPUs
- **Frontier**：AMD MI250X GCDs，最多使用 2048 GCDs
- **Tuolumne**：AMD MI300A APUs，最多使用 1024 GPUs

均为 HPE Cray EX 架构，Slingshot-11 网络互联。

### 评估指标
- **End-to-end training time to target accuracy**：
  - Reddit：目标准确率 95%
  - ogbn-products：目标准确率 79%
- **Epoch time**：用于强扩展性分析
- **Evaluation round time**：衡量推理效率

### 基线方法对比
| 基线 | 类型 | 特点 |
|------|------|------|
| **BNS-GCN** | Full-graph | 使用边界节点采样减少通信 |
| **DistDGL** | Mini-batch | 基于 DGL 的分布式实现，需远程特征获取 |
| **MassiveGNN** | Mini-batch | 在 DistDGL 上优化特征预取 |
| **SALIENT++** | Mini-batch | 加速 CPU 侧采样 + 缓存机制 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 🔥 End-to-end 训练加速比（vs. SOTA）
| 平台 | 数据集 | 最大 GPU/GCD 数 | Speedup |
|------|--------|------------------|--------|
| Perlmutter | ogbn-products | 64 GPUs | **3.5×** vs. SALIENT++ |
| Frontier | ogbn-products | 32 GCDs | **162×** vs. MassiveGNN, **228×** vs. DistDGL |

> DistDGL 在 Frontier 上甚至未能达到 79% 准确率。

#### 📈 强扩展性表现
| 平台 | 数据集 | GPU/GCD 规模 | Epoch Time Speedup |
|------|--------|---------------|--------------------|
| Perlmutter | ogbn-papers100M | 64 → 2048 GPUs | **21.7×** |
| Frontier | Products-14M | 32 → 1024 GCDs | **22.4×** |
| Tuolumne | Products-14M | 32 → 1024 GPUs | **17.2×** |

> 显示出良好的线性扩展趋势。

#### ⏱️ 单轮评估耗时对比（8 GPUs / 4 GPUs）
| 系统 | Reddit (4 GPUs) | ogbn-products (8 GPUs) |
|------|------------------|------------------------|
| ScaleGNN | **0.05s** | **0.19s** |
| BNS-GCN | 1.79s | 6.89s |
| SALIENT++ | 1.13s | 10.12s |
| DistDGL/MassiveGNN | 12.50s | 20.82s |

> ScaleGNN 在评估阶段快 **36–250×**

### 与基线方法的对比结果
- 在 **Reddit** 上，ScaleGNN 用 16 GPUs 仅需 **0.98s** 达到目标准确率，而 SALIENT++ 需要 3.13s，BNS-GCN 更高达 11.7s。
- 在 **ogbn-products** 上，64 GPUs 下 ScaleGNN 仅需 **3.80s**，而 SALIENT++ 需 13.25s，BNS-GCN 需 40.46s。
- 多数基线在增加 GPU 后并未缩短 end-to-end 时间，原因在于：
  - 每 epoch 时间下降有限；
  - 所需收敛 epoch 数反而增加。

### 消融实验结果（ablation study）
在 `ogbn-products` 上基于 8 和 32 GPUs 进行逐步优化验证：

| 优化步骤 | Epoch Time 下降幅度 |
|--------|---------------------|
| Baseline | 1.0× |
| + Overlap sampling with training | ↓24% |
| + Low-precision communication (BF16 all-reduce) | ↓额外 17% (DP1), 16% (DP4) |
| + Kernel fusion (RMSNorm+ReLU+Dropout) | ↓额外 6% (DP1), 4% (DP4) |
| + Communication-computation overlap | ↓额外 3% (DP1), 2% (DP4) |
| **累计加速比** | **1.75× (8 GPUs)**, **1.66× (32 GPUs)** |

> 表明各项优化均有效且可叠加。

---

## 4. 关键结论和发现

### 主要发现
1. **Communication-free sampling 是可行且高效的**：
   - uniform vertex sampling + unbiased rescaling 可保持模型准确性，甚至优于 GraphSAGE 和 GraphSAINT。
   - 实现完全去中心化的采样流程，极大提升扩展性。

2. **4D hybrid parallelism 显著提升可扩展性**：
   - 结合 data parallelism 和 3D PMM，使训练能有效扩展至 **2048 GPUs / GCDs**。
   - 3D PMM 将通信量从 $ O(n) $ 降至 $ O(n^{2/3}) $，远优于传统 1D/2D 方法。

3. **end-to-end 性能由多因素共同决定**：
   - 不只是训练速度，**evaluation 效率**也至关重要。ScaleGNN 利用分布式前向传播实现快速全图推理。

4. **现有框架存在“伪扩展”现象**：
   - 增加 GPU 数量虽提高吞吐，但因收敛变慢，最终 **end-to-end 时间不降反升**。

### 方法的局限性
- **依赖高质量图划分**：虽然采样无通信，但初始图分区质量影响负载均衡。
- **仅适用于 GCN-like 模型**：目前主要针对具有 SpMM+GEMM 结构的 GNN 层设计，对更复杂架构（如 attention-based）适配尚待研究。
- **内存占用较高**：由于每个 DP 组持有完整的图副本（分块存储），对超大图仍可能受限于显存容量。

### 未来工作方向
- 扩展至其他 GNN 架构（如 GAT、Transformer-based GNNs）。
- 探索异构集群下的自适应资源调度策略。
- 结合模型压缩技术进一步降低通信与存储成本。
- 支持动态图或流式图场景下的持续训练。

--- 

> ✅ **一句话总结**：  
> ScaleGNN 通过提出 **communication-free uniform vertex sampling** 与 **4D hybrid parallelism（data + 3D PMM）**，首次实现了在数千 GPU 上高效、可扩展的 mini-batch GNN 训练，在多个真实图数据集上取得最高达 **228× 的 end-to-end 加速比**，同时保持甚至超越 SOTA 模型精度。

</details>

---

### 2. [Communication-Efficient Collaborative LLM Inference over LEO Satellite Networks](https://arxiv.org/abs/2604.04654)

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

# **论文总结：Communication-Efficient Collaborative LLM Inference over LEO Satellite Networks**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在低地球轨道（LEO）卫星网络中部署大型语言模型（LLM）面临两大挑战：
- **资源受限**：单颗卫星的内存和计算能力有限，难以承载完整的LLM。
- **通信开销高**：跨卫星协作推理需频繁传输中间激活（intermediate activations），导致显著的通信延迟和带宽消耗。

现有方法如模型压缩（pruning、quantization）虽能减小模型体积，但会牺牲推理精度；而简单的模型分割（model splitting）则易造成负载不均和流水线阻塞。

---

### **提出的新方法与创新思路**
本文提出了一种**通信高效的协同LLM推理框架**，核心创新如下：

1. **多卫星协同推理架构（Collaborative Inference over LEO Network）**  
   将一个完整的LLM按层拆分为多个子模型（sub-models），分别部署于不同的计算卫星上，通过**星间链路（ISL）** 传递中间激活，实现分布式推理。

2. **流水线并行机制（Pipeline Parallelism）**  
   引入流水线执行策略，使得各卫星在计算当前批次的同时，可并发地接收前一颗卫星的下一批次输入，从而**重叠计算与通信时间**，减少整体延迟。

3. **自适应激活压缩方案（Adaptive Activation Compression）**  
   设计了一个端到端可训练的压缩模块，结合以下技术：
   - **Gumbel-Mask稀疏化**：学习选择最具信息量的激活特征，动态生成二值掩码。
   - **量化（Quantization）**：对非零元素进行低位宽编码。
   - **熵编码（Entropy Coding）**：进一步压缩稀疏后的表示。
   该方案在保持高精度的前提下大幅降低传输数据量。

4. **联合优化问题建模与求解算法**
   构造了一个混合整数非线性规划（MINLP）问题，联合优化：
   - 模型分层策略（layer assignment）
   - 各阶段激活压缩比（compression ratios）
   
   并将其转化为**有向无环图（DAG）上的最短路径搜索问题**，设计了一种改进的**A\*-based搜索算法**进行高效求解。

---

### **相比现有方法的优势**
| 维度 | 本方案优势 |
|------|-----------|
| **推理延迟** | 利用流水线并行显著缩短端到端延迟 |
| **通信效率** | 自适应压缩有效减少星间传输开销 |
| **精度保持** | 学习型压缩机制优于固定规则（如Top-k），误差累积更小 |
| **资源适配性** | 支持异构卫星环境下的动态负载均衡 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **EuroSAT**：基于Sentinel-2遥感影像的场景分类数据集，共10类，27,000张64×64 RGB图像。
- **RESISC45**：45类遥感场景数据集，31,500张256×256 RGB图像。

用于模拟真实地球观测任务中的视觉理解需求。

---

### **模型配置**
采用四种不同规模的Vision Transformer作为LLM代表：
| 模型 | 参数量 | 内存占用 |
|------|--------|---------|
| ViT-B | 0.086 Billion | ~2 GB |
| ViT-L | 0.307 Billion | ~4 GB |
| ViT-H | 0.632 Billion | ~7 GB |
| ViT-G | 1.8 Billion | ~12 GB |

其中ViT-G超出单颗卫星内存容量（8GB），必须依赖协同推理。

---

### **实验设置**
- **卫星网络拓扑**：Walker Delta星座（12颗卫星，高度500km，倾角53°），选取5颗参与计算。
- **硬件仿真平台**：
  - 地面站：NVIDIA RTX 4070Ti GPU
  - 卫星节点：4台NVIDIA Jetson AGX Orin（模拟不同功耗模式：15W/30W/50W）
- **通信参数**：
  - 星间链路（ISL）速率：0.5 Gbps（FSO光链路）
  - 星地链路（S2G）速率：6 Gbps（Ka波段）

---

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Inference Latency** | 从任务启动到地面接收到最终结果的时间 |
| **Communication Overhead** | 整个推理过程中传输的总数据量（含ISL与S2G） |
| **Inference Accuracy** | 分类准确率，衡量压缩与分割带来的精度损失 |
| **Optimization Gain** | 所提优化算法相对于启发式策略的性能提升 |

---

### **基线方法对比**
1. **Ground-only**：原始图像经中继卫星传回地面，在高性能服务器上完成全模型推理。
2. **Single-satellite**：将完整模型部署于单颗卫星，本地推理后仅上传结果。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | 提出方法 vs. 最优基线 | 提升幅度 |
|------|------------------------|----------|
| **推理延迟（End-to-End Delay）** | vs. Ground-only | ↓ 最高 **42%** |
| **通信开销（Communication Overhead）** | vs. Ground-only | ↓ 最高 **71.7%** |
| **精度损失（Accuracy Drop）** | 相对于无压缩基准 | < **1%** |

> 在1080p分辨率输入下，延迟降低达 **58%**；在低S2G速率（200 Mbps）场景下仍实现超 **100% 的加速比**。

---

### **与基线方法的对比结果**
- **vs. Ground-only**：
  - 显著减少下行链路负担（无需传输原始高清图像）
  - 推理延迟更低，尤其在带宽受限时优势明显
- **vs. Single-satellite**：
  - 能处理更大模型（如ViT-G），突破单星内存限制
  - 虽引入一定协作开销，但总体延迟更低（得益于负载均衡）

> 如图12所示，所提算法相较启发式分配（heuristic）和均匀划分（uniform），**总延迟分别降低103%和2.03倍**。

---

### **消融实验结果**
#### （1）压缩组件逐级效果（Fig. 8）
以ViT-G为例，每阶段压缩比逐步提升：
- **稀疏化（Sparsification）**：平均压缩约 **4×**
- **量化（Quantization）**：增至 **11.56×**
- **熵编码（Lossless Encoding）**：最终达 **25.82×**

表明三级压缩具有良好的叠加效应。

#### （2）不同压缩策略对比（Table IV & V）
| 压缩方式 | 准确率表现 |
|--------|------------|
| **GumbelMask（本文）** | 几乎无损（<1%下降），鲁棒性强 |
| **Top-k** | 小模型上严重退化（如ViT-Tiny在RESISC45上↓近5%） |

说明**可学习的动态选择机制**优于静态阈值法。

#### （3）模型分割位置敏感性分析（Fig. 10）
- 在200种不同的分层策略下，**超过97%的测试点精度偏差在±1%以内**
- 表明所提压缩方案对**分层位置不敏感**，具备强泛化能力

---

## **4. 关键结论和发现**

### **主要发现**
1. **协同推理 + 流水线并行是解决LEO卫星LLM部署的有效路径**，可在资源受限条件下实现高效推理。
2. **自适应激活压缩显著降低通信成本**，且通过学习机制有效保留关键语义信息，避免传统压缩带来的精度塌陷。
3. **联合优化模型分割与压缩策略至关重要**，启发式方法无法应对异构环境下的性能瓶颈。
4. 所提**DAG + A\*** 搜索算法能在合理时间内找到接近最优的部署策略，适用于实际系统调度。

---

### **方法的局限性**
1. **依赖离线训练压缩模块**：Gumbel-Mask需预先训练，可能增加部署复杂度。
2. **未考虑动态拓扑变化**：LEO卫星高速移动可能导致ISL中断，当前模型假设链路稳定。
3. **仅限同轨协作**：尚未扩展至跨轨道或多层网络（multi-orbit）场景。

---

### **未来工作方向**
1. **拓展至多轨道卫星网络**：研究跨极轨、倾斜轨道间的协同推理机制。
2. **支持动态重配置**：根据实时链路状态和任务负载在线调整分层与压缩策略。
3. **融合联邦学习或Split Learning范式**：实现模型训练与推理的一体化星上协同。
4. **探索更轻量化的压缩架构**：降低边缘设备上的训练与推理开销。

--- 

> ✅ **总结一句话**：  
> 本文首次实现了面向LEO卫星网络的**通信高效、精度保持、资源适配**的协同LLM推理框架，为未来6G空天地一体化智能计算提供了关键技术支撑。

</details>

---

### 3. [Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers](https://arxiv.org/abs/2604.02344)

**Authors**: J\k{e}drzej Maczan  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.02344v1  

#### Abstract
WebGPU's security-focused design imposes per-operation validation that compounds across the many small dispatches in neural network inference, yet the true cost of this overhead is poorly characterized. We present a systematic characterization of WebGPU dispatch overhead for LLM inference at batch s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
本论文系统性地研究了 **WebGPU 在 LLM 推理中的调度开销（dispatch overhead）**，解决了以下关键问题：
- WebGPU 因其安全设计引入了每操作验证机制，导致大量小规模 dispatch 操作累积出显著延迟。
- 当前对 WebGPU 调度开销的真实成本缺乏准确刻画，传统单次操作基准测试严重高估实际开销。
- 不同 GPU 厂商、后端（backend）和浏览器之间的性能差异未被充分量化。

### 🧩 提出了什么新方法或新思路
1. **Sequential-Dispatch Methodology（顺序调度测量法）**
   - 提出一种新的微基准测试方法：通过连续执行多个 dispatch 并仅在末尾同步，隔离出真正的 per-dispatch 开销，避免将 GPU-CPU 同步时间错误归入 dispatch 成本。
   - 揭示传统“单操作+立即同步”方法会高估开销达 **~20×**。

2. **区分两种开销层级**
   - 明确区分：
     - **Per-dispatch cost**：纯 WebGPU API 层面的开销（如 encoder 创建、bind group 设置、submit），直接测量。
     - **Per-operation overhead**：包含 Python / 框架层（如 PyTorch 动态图解释、tensor 元数据处理）在内的总开销，约 95 μs。
   - 这种划分对于优化具有指导意义。

3. **构建自定义工具链 torch-webgpu**
   - 开发了一个基于 `PrivateUse` 的 PyTorch 外部编译后端 `torch-webgpu`，支持 `torch.compile()` FX 图到 WGSL 的自动转换。
   - 构建 FX-to-WebGPU 编译器，实现端到端 LLM 推理流程。

### 🔍 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| **准确性** | 首次提供跨厂商、跨平台、跨浏览器的精确 WebGPU dispatch 开销数据 |
| **方法论** | Sequential-dispatch 方法更真实反映运行时行为，纠正了长期误解 |
| **可复现性** | 所有代码、基准脚本、原始数据开源（GitHub: [jmaczan/torch-webgpu](https://github.com/jmaczan/torch-webgpu)） |
| **实用性** | 实验覆盖主流硬件（NVIDIA/AMD/Apple/Intel）、三大浏览器（Chrome/Safari/Firefox）和多种 backend（Vulkan/Metal/D3D12） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与模型
- **模型**：Qwen2.5-0.5B 和 Qwen2.5-1.5B（均为 Instruct 版本）
- **任务**：自回归文本生成（autoregressive generation）
- **输入格式**：5-token 提示词 `"The capital of France is"`，生成 50 个 token
- **FX Graph 分析**：识别出每次前向传播中约 876 个计算节点可能成为 WebGPU dispatch

### ⚙️ 实验设置
| 组件 | 配置 |
|------|------|
| **主测试平台** | NVIDIA RTX 5090 + AMD Ryzen 7 9800X3D + Ubuntu 24.04 + PyTorch 2.9.1+cu128 |
| **其他平台** | Windows 11（RTX PRO 2000）、macOS（Apple M2）用于跨平台验证 |
| **WebGPU 实现** | Dawn（C++）、wgpu-native（Rust）、Chrome 144、Safari 26.2、Firefox 147 |
| **后端支持** | Vulkan（Linux）、Metal（macOS）、D3D12（Windows） |
| **精度模式** | 主要使用 float32；部分基线使用 float16（CUDA/MPS） |

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Tokens/sec (tok/s)** | 总生成 token 数 / 总耗时（主要反映 decode 阶段吞吐） |
| **Time to First Token (TTFT)** | 从开始到输出第一个 token 的时间（包含 prefill + 第一个 decode step） |
| **Coefficient of Variation (CV)** | 标准差 / 均值，衡量运行稳定性 |
| **Per-dispatch cost** | 单次 dispatch 的 CPU 侧开销（μs） |
| **Per-operation overhead** | 包含框架层的完整操作开销（μs） |

### 🔄 基线方法对比
| 基线 | 描述 |
|------|------|
| **CUDA (NVIDIA)** | 使用 `torch.compile()` 或 eager mode 的原生 CUDA 加速，作为高性能上限参考 |
| **MPS (Apple M2)** | Apple Metal Performance Shaders，macOS 上的本地加速方案 |
| **CPU (PyTorch Eager)** | 纯 CPU 推理，作为最低性能下限 |
| **ONNX Runtime (WebGPU)** | 微软 ONNX Runtime 的 WebGPU Execution Provider，用于比较不同 WebGPU 实现质量 |
| **WebLLM (Browser)** | 基于 TVM 的浏览器内 LLM 引擎，使用 q4f16 量化模型进行推理 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Qwen2.5-0.5B）

#### 🔹 WebGPU 调度开销（核心发现）
| 实现 | Per-dispatch cost (μs) | Backend |
|------|------------------------|---------|
| **Dawn (RTX 5090)** | 23.8 μs | Vulkan |
| **wgpu-native (RTX 5090)** | 35.8 μs | Vulkan |
| **Chrome (Linux/Vulkan)** | 32.8 μs | Vulkan |
| **Safari (M2/Metal)** | 31.7 μs | Metal |
| **wgpu-native (M2/Metal)** | 71.1 μs | Metal |
| **Firefox (any)** | ~1040 μs | Rate-limited |

> 💡 **结论**：真实 per-dispatch 开销远低于传统估计（过去认为是数百 μs），且 backend 选择（Vulkan vs Metal）是主导因素。

#### 🔹 端到端推理性能（RTX 5090, float32）
| 后端 | Tok/s | TTFT (ms) | vs CUDA (fp16) |
|------|-------|-----------|---------------|
| **torch-webgpu (fused)** | **21.0** | 41.6 | 0.11× |
| **ONNX Runtime (WebGPU)** | 13.1 | 73.5 | 0.07× |
| **CUDA (compiled, fp16)** | 185.5 | 5.4 | 1.00× |
| **CPU (eager)** | 13.7 | 72.8 | 0.07× |

- **融合优化效果显著**：kernel fusion 将 dispatch 数从 876 减少到 564，吞吐提升 **53%**。
- **无融合版本接近 CPU 表现**：说明调度开销足以抵消 GPU 计算优势。

#### 🔹 dtype 匹配下的公平比较
| 设备 | CUDA (float32) | WebGPU (float32) | 性能比 |
|------|----------------|------------------|--------|
| **RTX PRO 2000** | 30.1 tok/s | — | — |
| **RTX 5090 (torch-webgpu)** | — | 21.0 tok/s | **1.4× 更快** |

> ⚠️ 注意：尽管 RTX PRO 2000 的理论算力仅为 RTX 5090 的 ~1/6，但在 dtype 匹配下仍比 WebGPU 快 1.4×，表明 **调度与框架开销是主要瓶颈**。

---

### 🔬 消融实验结果

#### ✅ Kernel Fusion 效果分析（Table 5）
| 优化策略 | 节省 dispatch 数 | 吞吐提升 | 显著性 |
|--------|------------------|----------|--------|
| Fused RMSNorm (6→1) | 240 | +44% | p < 0.001 |
| Fused MLP gate+up+silu (3→1) | 48 | +6% | p < 0.001 |
| Fused K+V projection (2→1) | 24 | +0.5% | 不显著 |
| **总计** | **312** | **+53%** | — |

> 📌 Fusion 的收益主要来自减少 per-operation overhead，而非节省中间内存。

#### ❌ 其他优化尝试（无效）
| 优化 | 效果 | 原因 |
|------|------|------|
| **Command batching** | 无改善 | 自回归生成强制每 token 同步，无法批量提交 |
| **Buffer pooling / Bind group caching** | 无改善 | 开销集中在 `queue.submit()`，不在资源创建 |
| **Device-side argmax** | 无统计显著性 | Metal 上映射开销固定高，Vulkan 上方差大 |

#### 🔁 Backend 差异性实验
| 优化 | Vulkan (native) | Metal (native) | Chrome | Safari | Firefox |
|------|------------------|--------------|--------|--------|--------|
| **RMSNorm Fusion** | √ 1.4–1.7× | × 0.95× | ~1.06× | × 0.91× | — |
| **Tiled MLP (7→3 dispatch)** | 1.17× | 2.0× | — | — | — |

> 📌 **Vulkan 受益于 fusion，Metal 则不然**，说明优化策略需针对 backend 定制。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **真实 per-dispatch 开销远低于预期**  
   - Vulkan 上为 **24–36 μs**，Metal 上为 **32–71 μs**，比传统单操作测试低 **~20×**。
   - 主要瓶颈在于 `queue.submit()`（占 40%），其次是 encoder 创建与 finish。

2. **Per-operation overhead 是端到端瓶颈**  
   - 总开销约为 **95–99 μs**，其中：
     - WebGPU API 层：~24–36 μs
     - Python / 框架层：~59–71 μs
   - 在 batch=1 场景下，该开销远超 kernel 执行时间。

3. **Kernel Fusion 是最有效的优化手段**  
   - 减少 dispatch 数量可带来高达 **53% 的吞吐提升**。
   - Fusion 是否有效高度依赖 backend：**Vulkan 收益明显，Metal 几乎无益**。

4. **Backend 和 Implementation 影响巨大**
   - 同一 backend 下不同实现间差异可达 **2.2×**（如 Safari Metal vs wgpu-native Metal）。
   - Firefox 存在疑似 rate-limiting 行为，per-dispatch 成本高达 ~1040 μs，不适合 ML 推理。

5. **当前 WebGPU 推理效率极低**
   - torch-webgpu 在 RTX 5090 上仅达到 CUDA（fp16）性能的 **11–12%**。
   - WGSL kernel 效率仅达 FP32 peak 的 **1–2%**（第三方可达 ~17%，说明仍有空间）。

6. **Batch=1 场景下为“调度绑定”而非“计算绑定”**
   - 所有操作均处于 overhead-bound 状态，即使最大 matmul 也需要 batch ≥ 7 才能进入 compute-bound。
   - 类似 roofline 分析中的“低算术强度”区域。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **单一平台验证 overhead 拆解** | torch-webgpu 的详细开销分析仅在 RTX 5090/Dawn 上完成，跨平台泛化性待验证 |
| **仅支持 float32** | 未实现 WGSL float16 支持，与主流 CUDA/MPS 的 fp16 对比不公平 |
| **batch size=1 限制** | 所有实验均为单请求推理，不适用于批处理场景 |
| **未使用 vendor-specific 优化** | WGSL shader 未采用 bank conflict-free shared memory、vectorized FMA 等高级技巧 |
| **缺乏 GPU 内部 profiling** | WebGPU timestamp queries 粒度粗，无法获取 kernel 级细粒度数据 |

---

### 🔮 未来工作方向
1. **支持 batch > 1 推理**  
   - 验证调度开销是否可在更大 batch 下被摊销。

2. **实现 WGSL float16 支持**  
   - 消除 dtype 差异，实现更公平的性能对比。

3. **探索 WebGPU Spec 层级优化**
   - 如 compute graph capture/replay（类 CUDA Graphs）、persistent kernels、cooperative groups 等。

4. **开发更高效的 WGSL kernel 库**
   - 借鉴 Triton、webgpu-blas 等项目经验，提升 matmul 等基础算子效率至 10%+ peak。

5. **推动浏览器厂商优化 WebGPU 实现**
   - 特别是 Firefox 的 rate-limiting 机制应允许 ML 工作负载绕过。

6. **集成 into 主流 ML 框架**
   - 将 `torch-webgpu` 或类似后端纳入 PyTorch 官方生态，提升可用性。

---

> 📢 **总体评价**：本文是对 WebGPU 在 LLM 推理中实用性的首次系统性实证研究，揭示了其核心瓶颈并非算力不足，而是 **API 设计与软件栈协同造成的结构性开销**。虽然当前性能有限，但其跨平台、安全性、隐私保护等特性使其在浏览器端、边缘设备上有独特价值。未来若能在 spec 和实现层面进一步优化，有望成为 Web 上轻量级 AI 推理的重要基础设施。

</details>

---

### 4. [DARE: Diffusion Large Language Models Alignment and Reinforcement Executor](https://arxiv.org/abs/2604.04215)

**Authors**: Jingyi Yang, Yuxian Jiang, Xuhao Hu, Shuang Cheng, Biqing Qi, Jing Shao  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.04215v1  

#### Abstract
Diffusion large language models (dLLMs) are emerging as a compelling alternative to dominant autoregressive models, replacing strictly sequential token generation with iterative denoising and parallel generation dynamics. However, their open-source ecosystem remains fragmented across model families ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**DARE: Diffusion Large Language Models Alignment and Reinforcement Executor**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

当前 **diffusion large language models (dLLMs)** 领域面临严重的系统级碎片化问题：
- 不同研究团队发布的 dLLM 后训练（post-training）方法（如 SFT、RLHF、Preference Optimization）通常以独立代码库形式发布。
- 这些实现往往绑定特定模型架构（如 LLaDA、Dream、SDAR），并采用不一致的 rollout 实现、reward 接口和评估脚本。
- 导致以下三大问题：
  1. **研究迭代缓慢**：复用或比较算法需大量工程重写。
  2. **公平比较困难**：算法差异与执行差异混杂。
  3. **复现成本高**：缺乏统一框架支持。

此外，传统 LLM 的 RL 框架（如 Verl）无法直接用于 dLLMs，因为 dLLMs 的生成过程是基于 **denoising 轨迹**而非自回归 token 预测，需要专门的前向/反向过程建模和 likelihood surrogate 设计。

---

### 🚀 提出了什么新方法或新思路

提出 **DARE**（**dLLMs Alignment and Reinforcement Executor**），一个专为 dLLMs 设计的开源后训练与评估统一框架。

#### 核心设计思想：
- **统一执行栈（Unified Execution Stack）**：整合多种 dLLM 架构、训练算法、rollout 引擎和评估流程。
- **模块化解耦**：将通用流程（worker、dataflow、workflow）与模型/算法特异性逻辑分离。
- **系统级优化优先**：针对训练、rollout 和评估分别进行加速优化。

#### 支持的关键功能：
| 功能类别 | 支持内容 |
|--------|--------|
| **模型家族** | MDLMs（如 LLaDA, Dream, LLaDA-MoE）、BDLMs（如 SDAR, LLaDA2.0/2.1） |
| **训练范式** | SFT, PEFT (LoRA), Preference Optimization (DPO/VRPO), 多种 dLLM-specific RL 算法 |
| **RL 算法** | VRPO, D1, Coupled-GRPO, MDPO, CJ-GRPO, SPG, BGPO, EBPO 等 |
| **评估平台** | 集成 OpenCompass，支持多 benchmark 自动评测 |

---

### 🔍 相比现有方法的优势

| 维度 | 优势说明 |
|------|----------|
| **可复现性** | 将碎片化的 paper-specific 实现整合进统一框架，显著降低复现门槛 |
| **可比性** | 所有算法在相同 rollout、reward、evaluation 协议下运行，实现“苹果对苹果”比较 |
| **灵活性** | 新算法可通过插件方式接入，无需重构整个 pipeline |
| **效率提升** | 系统级优化带来高达 **4×（MDLM）至 14×（BDLM）** 的端到端 RL 流水线加速 |
| **生态整合** | 基于成熟的 `verl`（训练）和 `OpenCompass`（评估）构建，具备良好扩展性 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 任务类型 | 数据集 |
|--------|-------|
| **通用问答 / 推理** | MMLU, MMLU-Pro, Hellaswag, ARC-C |
| **数学推理** | GSM8K, MATH, AIME24/25, OlympiadBench |
| **代码生成** | HumanEval, MBPP |
| **规划任务** | Countdown, Sudoku（来自 d1 论文） |

---

### ⚙️ 实验设置和评估指标

#### 训练设置：
- **backbone 模型**：LLaDA-8B-Instruct, Dream-7B-Instruct, SDAR-8B-Chat, LLaDA2.x-mini
- **训练任务**：数学（GSM8K + MATH）、代码（DeepCoder 子集）、规划（Countdown + Sudoku）
- **超参数统一**：
  - Rollout group size = 8
  - Block length = 32
  - KL 正则默认关闭
  - Monte Carlo 采样数 = 16（用于 ELBO-based 方法）
  - Max response length: 数学/规划=512，代码=256
  - Diffusion steps: 数学=256，规划=128
  - 训练周期：1 epoch

#### 评估指标：
- **准确率（Accuracy）**：各 benchmark 上的标准 metric（如 pass@1）
- **训练曲线稳定性**：观察 reward 收敛行为与是否出现 collapse
- **延迟（Latency）**：每轮训练迭代时间、每次 rollout 步骤耗时

#### 对比的基线方法：
- **Baseline**：未经过 RL 微调的初始模型
- **SOTA dLLM-RL 方法**：
  - ELBO-based: VRPO, SPG, BGPO, EBPO
  - One-step denoising: D1, Coupled-GRPO
  - Trajectory-aware: MDPO, CJ-GRPO

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（摘录自 Tables 2–5）

#### 表 2：基础模型在多个 benchmark 上的表现（部分）

| Model | MMLU | GSM8K | MATH | HumanEval |
|-------|------|--------|------|-----------|
| LLaDA-8B-Instruct | 65.24 | 79.68 | 41.08 | 46.34 |
| Dream-7B-Instruct | 66.83 | 83.24 | 48.02 | 78.05 |
| SDAR-30B-A3B | 79.16 | 92.49 | 68.56 | 84.15 |
| LLaDA2.1-mini | 69.91 | 86.13 | 84.56 | 81.10 |

> ✅ 显示 DARE 成功复现主流 dLLM 在标准 benchmark 上的表现。

---

#### 表 3：数学任务上的算法对比（GSM8K / MATH）

**(a) LLaDA-8B-Instruct**
| Algorithm | GSM8K | MATH |
|---------|--------|------|
| Baseline | 76.5 | 34.6 |
| **CJ-GRPO** | **85.6** | 39.2 |
| Coupled-GRPO | 85.3 | **41.0** |

**(b) Dream-7B-Instruct**
| Algorithm | GSM8K | MATH |
|---------|--------|------|
| Baseline | 77.2 | 39.6 |
| **CJ-GRPO** | **85.7** | **50.7** |
| d1 | 82.5 | 49.7 |

> ✅ CJ-GRPO 和 Coupled-GRPO 在数学任务上表现最优；效果依赖 backbone。

---

#### 表 4：代码任务上的算法对比（HumanEval / MBPP）

**(a) LLaDA-8B-Instruct**
| Algorithm | HumanEval | MBPP |
|---------|------------|------|
| Baseline | 46.9 | 37.9 |
| **VRPO** | **52.4** | 42.8 |
| SPG | 48.8 | **41.9** |

**(b) Dream-7B-Instruct**
| Algorithm | HumanEval | MBPP |
|---------|------------|------|
| Baseline | 57.9 | 56.2 |
| **Coupled-GRPO** | **61.6** | **60.3** |
| d1 | 60.7 | 56.5 |

> ✅ VRPO 和 Coupled-GRPO 在代码任务中领先；不同 backbone 下最优算法不同。

---

#### 表 5：规划任务（Countdown / Sudoku）

| Algorithm | Countdown | Sudoku |
|---------|------------|--------|
| Baseline | 16.8 | 26.2 |
| Coupled-GRPO | **77.9** | 21.3 |
| BGPO | 10.0 | **42.6** |

> ✅ 不同任务偏好不同算法：Coupled-GRPO 擅长 Countdown，BGPO 更适合 Sudoku。

---

### ⏱️ 加速效果（Figure 2）

| 场景 | 性能增益 |
|------|--------|
| **MDLM SFT 训练延迟** | 从 22.1s → 10.8s（约 **2.0× speedup**） |
| **MDLM Rollout 延迟** | 从 161.6s → 73.4s（约 **2.2× speedup**） |
| **端到端 MDLM RL Pipeline** | **~4× 加速** |
| **BDLM RL Pipeline** | **>14× 加速**（得益于 LMDeploy/SGLang + fused kernels） |

> ✅ 系统优化贡献巨大，尤其 rollout 阶段。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **没有单一最优算法**：
   - 不同任务（数学、代码、规划）下表现最好的 RL 算法不同。
   - 同一算法在不同 backbone（LLaDA vs Dream）上表现差异显著。
   - 例如：SPG 在 Dream 上表现差，在 LLaDA 上尚可。

2. **算法稳定性存在明显差异**：
   - **CJ-GRPO、Coupled-GRPO、d1** 展现出更稳定的训练 reward 曲线。
   - **ELBO-based 方法（如 SPG、BGPO）更容易出现 late-stage reward collapse**，尤其是在 MC sample 数不足时。
   - 图 3 显示这些 collapse 并非偶然，而是与估计方差相关。

3. **系统优化至关重要**：
   - 解耦 rollout 与 actor update 的 attention backend 可带来数量级加速。
   - Fast-dLLM + KV Cache 显著提升 MDLM rollout 效率。
   - FlexAttention 和 fused kernels 对 BDLM 训练至关重要。

4. **统一框架的价值凸显**：
   - 在 DARE 中才能清晰看到：“无绝对赢家”的本质是任务与模型敏感性，而非实现偏差。
   - 揭示了当前 ELBO-based 方法的脆弱性，提示社区应发展更鲁棒的 surrogate objective。

---

### ⚠️ 方法的局限性

1. **尚未覆盖所有 dLLM 变体**：
   - 当前主要支持 MDLM 和 BDLM，尚未集成 vision-language 或 omni-modal diffusion LM。
   
2. **依赖外部组件稳定性**：
   - 依赖 `verl`, `OpenCompass`, `Fast-dLLM`, `LMDeploy` 等外部库，其更新可能影响兼容性。

3. **部署支持有限**：
   - 虽然强调系统优化，但目前仍聚焦于研究场景下的训练与评估，生产级部署能力有待加强。

4. **蒙特卡洛估计开销大**：
   - ELBO-based 方法需要较多 MC samples 才能稳定，增加计算负担。

---

### 🔮 未来工作方向（作者建议）

1. **模型扩展**：
   - 支持 diffusion vision-language models 和 multimodal dLLMs。

2. **算法演进**：
   - 吸纳新的 policy gradient estimator、control variates、stability-enhancing 技术。
   - 探索 variable-length denoising 的统一接口（如 p-EOS Yang et al., 2026）。

3. **系统深化**：
   - 增加更多 efficiency ablation 工具。
   - 开发面向部署的 evaluation backend（如低延迟 serving 模拟）。

4. **开放协作**：
   - 鼓励社区贡献新算法插件、新模型适配器和新 benchmark。

---

## 总结

> **DARE 并不是一个新算法，而是一个推动 dLLM 领域走向标准化、可复现、可比较的研究基础设施**。

它通过统一执行栈解决了当前 dLLM 后训练生态的碎片化问题，并结合系统级优化实现了显著的性能提升。其实验揭示了一个重要事实：**当前 dLLM-RL 方法的效果高度依赖于任务、模型架构和实现细节**，这正是需要 DARE 这类统一框架的根本原因。

📌 **一句话总结**：  
**DARE 是 dLLM 时代的 `HuggingFace + Accelerate + TRL` 的融合体，为该领域的可持续发展提供了坚实基础**。

</details>

---

### 5. [Structured Causal Video Reasoning via Multi-Objective Alignment](https://arxiv.org/abs/2604.04415)

**Authors**: Zinuo Li, Yongxin Guo, Jun Liu, Jiawei Zhan, Xi Jiang, Chengjie Wang, Mohammed Bennamoun, Farid Boussaid, Feng Zheng, Qiuhong Ke  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.04415v1  

#### Abstract
Human understanding of video dynamics is typically grounded in a structured mental representation of entities, actions, and temporal relations, rather than relying solely on immediate deductive reasoning. In contrast, existing Video-LLMs largely depend on unstructured video reasoning, where critical...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Structured Causal Video Reasoning via Multi-Objective Alignment**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 **Video-LLMs**（Video Large Language Models）在视频理解中普遍采用非结构化的 **Chain-of-Thought (CoT)** 推理方式，导致以下问题：
- 关键视觉证据被淹没在冗长、无序的文本描述中；
- 时间因果关系建模薄弱，推理过程易漂移（reasoning drift）；
- 缺乏可解释性和中间证据验证机制。

这与人类通过构建**结构化心智表征**（entities, actions, temporal relations）进行动态理解的认知机制存在显著差距。

---

### **提出的新方法与思路**
本文提出一种“**Structure-First**”范式，核心是引入 **Structured Event Facts**（结构化事件事实）作为推理前的显式约束：

#### **(1) Structured Event Facts**
- 在推理前从视频中提取紧凑、高密度的结构化摘要，包含：
  - 时间区间 `[time]`
  - 人物 `[person]`
  - 动作 `[human_action]`
  - 场景 `[scene]`
  - 物体 `[object]`
  - 镜头 `[camera]`
  - 因果事件描述 `[casual_event_caption]`
- 这些事实构成一个**可验证的中间表示**，为后续推理提供锚点。

#### **(2) 四阶段训练流程**
为有效训练模型输出并利用结构化事实，设计了渐进式训练 pipeline：
1. **Facts Training (Stage 1)**：训练模型生成高质量的事实描述；
2. **Format Warm-Start (Stage 1.5)**：强制模型以 `<thinking>` 格式输出，预热结构格式；
3. **Thinking Warm-Start (Stage 2)**：训练基于事实的因果推理能力；
4. **RL-based Post-training (Stage 3)**：使用强化学习对齐多目标优化。

#### **(3) Pareto-Frontier guided Advantage Balancing (P-FAB)**
在 RL 阶段，传统方法如 **GRPO** 难以平衡多个冲突目标（如事实完整性 vs. 推理长度）。为此提出 **P-FAB** 算法：
- 将多目标奖励向量视为独立信号；
- 借鉴 **Multiple Gradient Descent Algorithm (MGDA)**，求解最小范数组合方向；
- 动态调整优势权重，逼近 **Pareto-Frontier**，实现公平且稳定的优化。

---

### **相比现有方法的优势**
| 维度 | 传统方法 | 本方法 |
|------|--------|--------|
| **推理结构** | 非结构化 CoT，冗长易漂移 | 结构化先验 + 显式因果验证 |
| **可解释性** | 黑箱推理，难以追溯 | 中间事实可读、可验证 |
| **时间建模** | 孤立帧检索为主 | 强调事件边界与前后因逻辑 |
| **优化策略** | 单标量奖励聚合，忽略权衡 | 多目标动态平衡，保留梯度差异 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**

#### **训练数据集**
- 构建了 **CausalFact-60K** 数据集，包含约 60K 条样本，源自多个高质量 VTG（Video Temporal Grounding）数据集：
  - **ActivityNet-Captions**（主导）
  - **QVHighlights**, **COIN**, **Charades-STA**, **Molmo2**, **YouCookII**
- 视频平均时长 109.4 秒，覆盖教程、体育、日常生活等丰富主题。

#### **评估数据集**

| 类型 | 数据集 | 描述 |
|------|-------|------|
| **Temporal Grounding** | Charades-TimeLens | 高精度重标注数据集，强调细粒度定位 |
| | ActivityNet-TimeLens | 对 ActivityNet-Captions 的严格边界重标注 |
| | ActivityNet-Captions | 大规模开放域视频定位基准 |
| **General Understanding** | VideoMME | 长视频多模态理解，涵盖电影、纪录片等 |
| | MLVU | 包含话题推理（TR）、第一人称视角理解（Ego）等任务 |
| | ETBench | 8项细粒度时间敏感任务，如 TVG、TEM、TAL |
| | NExT-GQA | 要求答案接地于具体视频片段的因果/时序问答 |

---

### **实验设置与评估指标**

#### **模型架构**
- 基础模型：`Qwen3-VL-4B-Instruct`
- 最终模型：**Factum-4B**（基于上述四阶段训练得到）

#### **训练细节**
- **Stage 1–2**：LoRA 微调，不同学习率配置防止灾难性遗忘；
- **Stage 3 (RL)**：全参数更新，使用 P-FAB 算法；
- 视频采样帧率：1fps（低于多数基线），max frames=128（Stage 3）；
- 使用 `Gemini-2.5-Pro` 和 `Qwen3-VL` 相互生成与评判，保证数据质量。

#### **评估指标**
| 任务类型 | 主要指标 |
|--------|---------|
| Temporal Grounding | R@1 (IoU=0.3, 0.5, 0.7) |
| QA / Reasoning | Accuracy (%) |
| Causal Reasoning | TEMRec, TALF1, EPMF1 等 ETBench 子任务指标 |
| 整体性能 | NExT-GQA Acc, VideoMME Acc |

---

### **基线方法对比**
- **Closed-source baselines**：
  - GPT-4o, GPT-5, Gemini-2.5-Pro
- **Open-source baselines**：
  - Qwen3-VL-4B-Instruct / Thinking
  - VideoChat-R1-7B, TRACE-7B, Time-R1-7B
  - VTimeLLM-7B, TinyLLaVA-Video-R1-3B

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **Temporal Grounding 性能（Table 2）**

| Model | ActivityNet-Captions R@1@0.5 | R@1@0.7 | Charades-TimeLens R@1@0.7 |
|-------|-------------------------------|--------|-----------------------------|
| Qwen3-VL-4B-Instruct | 35.8% | 21.6% | 18.4% |
| Qwen3-VL-4B-Thinking | 31.7% | 19.0% | 17.8% |
| **Factum-4B (ours)** | **48.4%** | **28.1%** | **21.6%** |

> 💡 在 ActivityNet 上提升 **12.6pp @0.5 IoU**，远超同规模模型。

#### ✅ **General Video Understanding 性能（Table 3）**

| Model | VideoMME Acc | MLVU TR Acc | NExT-GQA Acc |
|-------|--------------|------------|-------------|
| Qwen3-VL-4B-Instruct | 63.9% | 80.4% | 72.1% |
| Qwen3-VL-4B-Thinking | 63.1% | 79.5% | 66.6% |
| **Factum-4B (ours)** | **64.7%** | **80.6%** | **73.6%** |

> 🏆 在 **ETBench** 的 TVG 和 TEM 任务上分别达到 **66.1%** 和 **26.8%**，甚至超过 GPT-4o。

---

### **与基线方法的对比结果**
- **优于所有开源 4B~7B 模型**，尤其在细粒度时间定位和因果推理任务上表现突出；
- 在低帧率（1fps）下仍超越多数使用更高帧率（2fps, 2048 frames）的闭源系统；
- **首次证明小模型可通过结构化推理超越大模型**。

---

### **消融实验结果（Table 1）**

| 变体 | ActivityNet R@1@0.5 | VideoMME Acc | MLVU TR Acc |
|------|--------------------|--------------|------------|
| Full Model (**Factum-4B**) | **48.4%** | **64.7%** | **80.6%** |
| w/o Facts | 41.6% | 60.8% | 76.8% |
| w/o Thinking | 40.4% | 58.5% | 75.6% |
| w/o RL | 41.6% | 59.1% | 76.8% |
| GRPO (G=8) | 45.7% | 63.5% | 79.8% |
| **P-FAB (G=8)** | **45.7% → 48.4%** | **63.5 → 64.7%** | **79.8 → 80.6%** |

> 🔍 发现：
- **Facts 和 Thinking 缺一不可**，移除任一模块均导致严重性能下降；
- **P-FAB 明显优于标准 GRPO**，尤其在 group size 增大时优势更明显（G=8 时差距达 2.5%）；
- **RL 后训练带来 8.3% 的绝对增益**，验证其有效性。

---

## **4. 关键结论和发现**

### **主要发现**
1. **结构化先验显著提升推理可靠性**：
   - Structured Event Facts 提供了清晰、可验证的中间证据，避免了传统 CoT 的“幻觉”和漂移问题。
   
2. **小模型也能实现强推理**：
   - **Factum-4B**（仅 4B 参数）在多个任务上超越 7B 甚至闭源大模型，表明**结构设计比模型规模更重要**。

3. **多目标冲突需显式建模**：
   - 在 RL 中，单纯加权平均奖励会掩盖关键权衡；
   - **P-FAB 能动态识别稀有但重要的目标信号**（如高 IoU + 正确格式），实现更优 Pareto 平衡。

4. **非结构化思考反而有害**：
   - 实验显示 `Qwen3-VL-4B-Thinking` 多数情况下**不如**指令微调版本，说明盲目增加推理步骤可能削弱性能。

---

### **局限性**
- **训练数据受限**：当前 CausalFact-60K 仍有限，尚未完全扩展到所有视频理解场景；
- **依赖人工定义 schema**：结构化字段（如 `[camera]`, `[casual_event_caption]`）需要领域知识设计；
- **计算成本较高**：四阶段训练流程复杂，尤其是 RL 阶段需大量采样。

---

### **未来工作方向**
- 扩展 CausalFact 数据集至更大规模、更多样化的视频来源；
- 探索自动发现最优结构 schema 的方法（如 latent structure discovery）；
- 将 P-FAB 应用于其他多模态任务（如图文生成、机器人决策）；
- 结合 memory 或 external tools 实现长期视频记忆推理。

---

> ✅ **总结一句话**：  
> 本文通过引入 **Structured Event Facts + P-FAB 多目标优化**，实现了**更可靠、可解释、高性能的小模型视频因果推理**，为下一代 Video-LLMs 提供了新范式。

</details>

---

### 6. [AdaHOP: Fast and Accurate Low-Precision Training via Outlier-Pattern-Aware Rotation](https://arxiv.org/abs/2604.02525)

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
低精度训练（**Low-Precision Training, LPT**）在大语言模型（**LLMs**）中面临一个核心挑战：**outliers（异常值）**。这些稀疏但极端的数值会显著放大量化误差，导致训练不稳定、收敛困难甚至性能下降。

现有方法（如 **Hadamard Transform**）通常采用**固定变换策略**（例如统一应用 Inner Hadamard Transform, IHT），假设所有张量的 outlier 结构相同。然而，该论文指出这一假设是错误的——不同层、不同计算路径（前向、反向传播）中的权重（Weights）、激活（Activations）和梯度（Gradients）具有**高度异构且稳定的 outlier 模式**，统一处理会导致次优甚至有害的量化效果。

### 提出了什么新方法或新思路
论文提出 **AdaHOP**（**Adaptive Hadamard transform with Outlier-Pattern-aware strategy**），其核心思想是：

- **细粒度分析**：首次系统性地识别出 LLM 张量中存在三种结构化 outlier 模式：
  - **Row-wise (R)**：异常值集中在少数行。
  - **Column-wise (C)**：异常值集中在少数列。
  - **None (N)**：无明显集中模式。
- **模式感知策略**：针对每一对矩阵乘法操作数（如 `Gy` 和 `X` 在 `Gw = Gy @ X` 中）的 outlier 模式组合（共9种可能），动态选择最优的量化策略。
- **自适应混合精度设计**：
  - 对于 IHT 有效的模式对（如 **CR**），直接应用 **IHT**。
  - 对于 IHT 无效的模式对（如 **RC**, **RN**, **CC**），引入 **Selective Outlier Extraction (OE)**，将主导异常值提取到高精度（BF16）路径计算，其余部分用 IHT 处理。

### 相比现有方法的优势
- **更高的准确性**：通过精准匹配变换方向与 outlier 结构，显著降低量化误差，实现与 BF16 全精度训练相当的质量。
- **更高的效率**：相比需要额外全局变换的 OHT 方法（如 HALO），AdaHOP 的 OE 设计开销更低，实现了更高的 kernel 加速比。
- **硬件友好**：结合 **Triton** 编写的融合内核，在 AMD CDNA4 架构上高效执行，最小化数据移动和计算开销。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **C4 (Colossal Clean Crawled Corpus)**：作为主要训练数据集。

### 实验设置和评估指标
- **模型**：在四种不同规模的 LLM 上进行验证：
  - Llama3.2-1B
  - Llama3.2-3B
  - Instella-3B
  - Llama3.1-8B
- **训练配置**：
  - 序列长度：4096
  - 批大小：128
  - 优化器：AdamW
  - 学习率：4e-4
  - 训练步数按 Chinchilla 定律缩放（1B 模型训练 40B tokens，8B 模型训练 160B tokens）。
- **评估指标**：
  - **训练损失**（Training Loss）及其与 BF16 的差距。
  - **下游任务零样本准确率**（Zero-shot Accuracy）：PIQA、HellaSwag、ARC-Easy、LAMBADA。
  - **内存消耗**（Memory Consumption）
  - **训练吞吐量**（Throughput, tok/s）
  - **Kernel 延迟与加速比**

### 基线方法对比
- **BF16**：全精度训练，作为质量上限。
- **Naive MXFP4**：无任何 outlier 抑制的纯低精度训练。
- **MXFP4 + Hadamard**：统一应用 IHT。
- **Tseng et al.**：基于随机 Hadamard 的 MXFP4 训练方法。
- **HALO**：应用 OHT 到梯度路径的先进方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **训练质量**：
  - AdaHOP 在所有模型上均达到或接近 BF16 的训练损失水平（图1），损失差距始终小于 0.01。
  - 在 **Instella-3B** 上，AdaHOP-Lv2 的平均零样本准确率达到 **56.05%**，**超过 BF16 基线（55.75%）**，并在 PIQA 和 ARC-Easy 上实现超越。
- **效率指标**（以 Llama3.1-8B 为例，表4 & 表5）：
  - **内存压缩**：AdaHOP-Lv1 将线性层内存从 BF16 的 **76.00 GB** 压缩至 **20.94 GB**，实现 **3.6× 内存压缩**；Lv2 版本为 28.04 GB（2.7×）。
  - **Kernel 加速**：在典型矩阵尺寸下，AdaHOP 实现 **1.59× 至 1.80×** 的 kernel 级 GEMM 加速比（vs. BF16）。
  - **端到端吞吐**：AdaHOP-Lv1 吞吐为 **13,247 tok/s**，略高于 BF16（12,946 tok/s）。

### 与基线方法的对比结果
- **质量方面**：
  - AdaHOP 在所有 MXFP4 方法中取得**最高或第二高的下游任务准确率**（表3）。
  - 显著优于 Naive MXFP4 和 MXFP4+Hadamard，也优于 Tseng et al. 和 HALO。
  - **AdaHOP-Lv2** 通过在注意力敏感层（CC 模式）保留 BF16 精度，进一步提升了关键任务（如 ARC-Easy）的表现。
- **效率方面**：
  - **HALO** 虽然有效，但因 OHT 引入大量额外 FWHT 计算和数据移动，其吞吐量（10,482 tok/s）**远低于其他所有方法**，甚至低于 BF16。
  - AdaHOP 在保持 BF16 级质量的同时，**吞吐量显著优于 HALO 和 Tseng et al.**。

### 消融实验结果
- **模式稳定性**（图4 & 图6）：实验证明，outlier 模式在训练早期即稳定形成，支持通过短时 BF16 校准（calibration）一次性确定模式，无需运行时检测。
- **OE 的有效性**：理论分析（Theorem 2）和实验表明，OE 能有效消除由严重 outlier 导致的量化误差放大因子 `√γ(A)`。
- **k=64 的选择**：该值由 AMD CDNA4 的 **mfma_scale_f32_32x32x64_f8f6f4** 指令决定，能高效覆盖 BF16 异常子矩阵，平衡精度与效率。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Outlier 模式是结构化且稳定的**：LLM 中的 outlier 并非随机，而是遵循 **Row-wise, Column-wise, None** 三种模式，并在训练过程中保持稳定。
2. **“一刀切”的变换策略是次优的**：Hadamard 变换的有效性**强烈依赖于其平滑方向是否正交于 outlier 方向**。统一应用 IHT 或 OHT 无法在所有计算路径上都有效。
3. **自适应策略至关重要**：根据每对操作数的具体 outlier 模式对（Pattern Pair）选择最优策略（IHT 或 OE+IHT），是实现高质量低精度训练的关键。
4. **硬件协同设计提升效率**：通过 Triton 实现的融合内核，AdaHOP 能在现代加速器（如 AMD CDNA4）上高效执行混合精度计算，实现速度与内存的双重优势。

### 方法的局限性
1. **固定变换矩阵**：目前使用固定的 Walsh-Hadamard 矩阵，未探索学习型旋转矩阵（learned rotation matrices）的潜力。
2. **模型架构泛化性待验证**：分析主要基于 Llama-family 和 Instella 模型，需扩展到其他架构（如 Mixtral, Gemma）以验证普适性。
3. **数值格式限制**：当前框架建立在 MXFP4 上，可扩展至其他低精度格式（如 FP8, INT4）。
4. **超参数固定**：提取的异常行/列数 `k=64` 是全局固定的，未根据每层的 outlier 严重程度进行自适应调整。

### 未来工作方向
1. **结合学习型旋转**：将 AdaHOP 与 **SpinQuant** 等学习型旋转方法结合，可能获得更低的量化误差。
2. **扩展到更多模型架构**：在 MoE、不同归一化方案的模型上验证 outlier 模式的普遍性。
3. **推广到其他量化格式**：将 pattern-aware 思路应用于 FP8、INT4 等格式的训练。
4. **自适应 OE 策略**：研究 per-layer 的 `k` 值选择策略，以进一步优化精度-效率权衡。

</details>

---

### 7. [Fast NF4 Dequantization Kernels for Large Language Model Inference](https://arxiv.org/abs/2604.02556)

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
当前主流的 **NF4 (4-bit NormalFloat)** 量化技术虽然能将大模型内存占用减少 4×，但在 NVIDIA 当前架构（如 Ampere A100）上缺乏原生 4-bit 支持，因此每次矩阵乘法前都需要将权重从 NF4 **反量化**（dequantize）为 FP16。这一过程涉及大量对全局内存（global memory）中查找表（LUT）的访问，造成严重的性能瓶颈。

作者通过分析发现，在 Qwen3-32B 模型中，**dequantization 占据了端到端延迟的 21–40%**，成为推理效率的关键制约因素。

---

### 🚀 提出的新方法与创新思路

本文提出了一种**轻量级共享内存优化策略**，核心思想是利用 GPU 的内存层次结构来缓解内存访问瓶颈：

1. **共享内存缓存 LUT**  
   将仅需 64 字节的 16 元素 NF4 查找表（LUT）由单个线程加载至 **shared memory**，供整个 thread block 复用，避免每个线程重复从高延迟的 global memory 加载。

2. **简化索引计算逻辑**  
   替换原有基于 4 层条件分支树的复杂索引方式，采用位操作（bit masking & shifting）实现直接寻址，消除 warp divergence 并大幅降低指令数。

3. **保持生态系统兼容性**  
   完全兼容 HuggingFace Transformers 和 BitsAndBytes 框架，无需离线预处理、模型转换或修改训练流程，可即插即用。

---

### 🔍 相比现有方法的优势

| 维度 | 本工作 | 现有方法（如 BitsAndBytes 基线） |
|------|--------|-------------------------------|
| 内存访问模式 | 利用 shared memory，单次加载 LUT，广播复用 | 每个线程独立访问 global memory 中 LUT |
| 索引机制 | 无分支位操作（2 条指令） | 多层条件判断树（最多 7 条指令），存在 warp divergence |
| 工程复杂度 | 极低，仅修改 kernel 实现 | 无需改动框架，不依赖 kernel fusion 或特殊编译器 |
| 生态兼容性 | 零侵入，支持现有量化模型直接部署 | 同样兼容，但性能较差 |
| 性能增益 | 显著提升 kernel 和 end-to-end 推理速度 | 存在严重内存瓶颈 |

> 💡 **优势总结**：以极小的工程代价（仅使用 64 bytes shared memory / block），实现了显著的性能加速，且具备良好的通用性和可部署性。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **GSM8K**：用于生成输入 prompt，测试不同长度和 batch size 下的推理表现。
- 所有实验均基于真实任务场景下的文本输入进行 token-by-token 推理。

---

### ⚙️ 实验设置

| 项目 | 配置 |
|------|------|
| 硬件平台 | 单张 **NVIDIA A100-80GB GPU**，AMD EPYC 7513 CPU（32核），64GB RAM |
| 软件环境 | CUDA 12.6, PyTorch 2.1, BitsAndBytes 0.47.0 |
| GPU 设置 | 锁定最大频率（1410 MHz），消除动态调频影响 |
| 测试模型 | **Gemma 27B**, **Qwen3 32B**, **Llama3.3 70B** |
| Batch Sizes | 2, 4, 8, 16, 32, 64 |
| 评估粒度 | kernel-level latency + end-to-end inference latency & throughput |
| 测量工具 | PyTorch Profiler + CUPTI，微秒级精度，包含 kernel launch 和 memory transfer 开销 |
| 可复现性 | 固定随机种子，三次运行取平均值 |

---

### 🔁 基线方法对比
- **Baseline**：开源的 **BitsAndBytes** 实现中的 `kDequantizeBlockwise` kernel。
- **Optimized**：本文提出的共享内存 + 直接索引版本。
- 对比维度：
  - Dequantization kernel 延迟
  - 端到端推理延迟（end-to-end latency）
  - 吞吐量（tokens/sec）
  - 指令数量与 warp divergence 情况

---

## 3. 主要实验结果和性能指标

### 📈 Kernel-Level 性能提升（表 II）

| Batch Size | Gemma 27B | Qwen3 32B | Llama3.3 70B | 平均 |
|------------|-----------|-----------|--------------|-------|
| 2          | 2.10×     | 2.20×     | 2.04×        | ~2.11× |
| 4          | 2.10×     | 2.19×     | 2.04×        |       |
| 8          | 2.11×     | 2.19×     | 2.04×        |       |
| 16         | 2.10×     | 2.19×     | 2.03×        |       |
| 32         | 2.11×     | 2.19×     | 2.05×        |       |
| 64         | 2.08×     | 2.15×     | 2.03×        |       |
| **Average**| **2.10×** | **2.19×** | **2.04×**    | **2.0–2.2×** |

✅ **结论**：在所有模型和 batch size 下，**dequantization kernel 实现了稳定 2.0–2.2× 的加速**，说明优化针对的是底层内存瓶颈，而非特定模型结构。

---

### 🚀 端到端性能提升（图 4）

| 模型 | 平均加速比 | 最高加速比 |
|------|------------|------------|
| **Llama3.3 70B** | **1.52×** | **1.54×** @ batch 2 |
| **Qwen3 32B**    | **1.18×** | **1.29×** @ batch 32 |
| **Gemma 27B**    | **1.10×** | **1.32×** @ batch 64 |

- 更大的模型受益更明显（因 dequantization 占比更高）。
- 小 batch size 下加速更显著（matrix computation 并行度低，memory bottleneck 更突出）。

---

### 📊 吞吐量提升（tokens/sec）

| 模型 | 最高吞吐提升 |
|------|----------------|
| Llama3.3 70B | ↑1.54× （batch 2） |
| Qwen3 32B    | ↑1.30× （283 → 368 tokens/s @ batch 32） |
| Gemma 27B    | ↑1.25× （506 → 633 tokens/s @ batch 64） |

> 吞吐提升直接转化为生产系统更高的服务容量和更低的单位成本。

---

### 🔍 消融分析与关键观察

- **Shared Memory 利用**：shared memory 访问延迟仅 **19 cycles** vs global memory **290 cycles**，获得 **12–15× 的访存延迟优势**。
- **指令数减少**：索引计算从 **7 条带分支指令 → 2 条无分支指令**，降幅达 **71%**。
- **消除 warp divergence**：所有线程执行相同指令流，提升 SIMT 效率。
- **LUT 流量降低 64×**：每 block 原需 64 次 global memory load，现仅需 1 次 shared memory load。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Dequantization 是当前 NF4 推理的主要瓶颈**，即使其计算简单，但由于频繁的 global memory 访问导致严重性能损失。
2. **GPU 内存层级优化具有巨大潜力**：合理利用 shared memory 可以显著缓解访存压力，带来数倍加速。
3. **轻量级优化也能产生重大影响**：仅使用 **64 bytes / block** 的 shared memory 和少量代码重构，即可实现 **2.0–2.2× kernel speedup** 和最高 **1.54× 端到端加速**。
4. **模型越大，收益越高**：随着参数规模增加，dequantization 在总时间中的占比上升，优化效果更加显著。
5. **无需牺牲生态兼容性**：该优化完全兼容 HuggingFace + BitsAndBytes，无需任何模型重训练或转换。

---

### ⚠️ 方法的局限性

- **依赖固定大小 LUT**：仅适用于像 NF4 这样具有静态 16-level 分布的量化方案；对于动态量化可能需要扩展设计。
- **对 compute-bound 场景增益有限**：若模型已高度优化至计算密集型（如大规模 batch 下的 matmul 占主导），则 dequantization 优化边际效应下降。
- **未探索其他硬件平台**：目前仅在 Ampere A100 上验证，Hopper 或 Blackwell 架构可能有不同的访存特性。

---

### 🔮 未来工作方向

1. **扩展至其他量化格式**：将类似思想应用于 GPTQ、AWQ 等非均匀量化方案的 dequantization kernel。
2. **结合 kernel fusion**：进一步融合 dequantization 与 matmul 操作，减少中间数据搬运。
3. **适配新一代 GPU 架构**：研究在 Hopper H100 或未来 Blackwell 上如何最大化 shared memory 和 tensor memory 的协同效益。
4. **自动调优集成**：将此类优化封装进自动 kernel generator（如 Triton）中，实现自适应部署。

---

## ✅ 总结

本文提出了一种**简洁而高效**的 NF4 反量化优化方法，通过将 LUT 加载至 shared memory 并简化索引逻辑，在几乎不增加工程复杂度的前提下，实现了 **2.0–2.2× 的 kernel 加速** 和高达 **1.54× 的端到端推理提速**。该方法不仅性能卓越，而且完全兼容现有生态，为在现有 GPU 基础设施上高效部署超大规模语言模型提供了实用解决方案。

</details>

---

### 8. [Towards Near-Real-Time Telemetry-Aware Routing with Neural Routing Algorithms](https://arxiv.org/abs/2604.02927)

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

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统路由算法（如 OSPF、EIGRP）在面对突发流量或网络拓扑变化时反应缓慢，难以实现**毫秒级响应**。虽然已有研究尝试使用 **Machine Learning (ML)** 和 **Reinforcement Learning (RL)** 来优化路由决策，但存在以下关键缺陷：
- **忽略通信延迟**：多数神经路由算法假设能获取无延迟的全局网络状态（“birds-eye” view），这在真实网络中不可行。
- **纯局部观测**：部分分布式方法仅依赖本地遥测数据，缺乏协调性，导致次优决策。
- **训练与部署脱节**：训练环境未模拟实际的**状态传播延迟**和**模型推理延迟**，导致算法在真实场景中表现不佳。

因此，如何设计一个**延迟感知**（delay-aware）且能有效利用遥测数据的神经路由算法，是迈向现实部署的关键挑战。

### 提出的新方法：LOGGIA
本文提出了一套完整的解决方案，核心包括：

#### （1）延迟感知的仿真框架
- 将遥测感知路由建模为一个**延迟感知的闭环控制问题**。
- 在 **ns-3** 包级仿真器基础上，结合 **ns3-ai** 的共享内存机制，构建了一个高保真训练与评估框架。
- 显式建模了：
  - **通信延迟**：状态信息通过最短路径聚合，引入传播延迟。
  - **推理延迟**：模型前向计算所需时间，可通过硬件加速参数 $\lambda_{ac}$ 调整。
  - 支持多种部署模式（Central-Single, Local-Multi 等），以评估不同架构的影响。

#### （2）新型神经路由算法 LOGGIA
- **全称**：LOg-space link weight prediction on Graphs with Guided update epochs and Implicit-Alpha entropy adaptation.
- **架构设计**：
  - 使用 **Message Passing Networks (MPNs)** 处理带属性的拓扑-遥测图。
  - **直接在原始图上操作**，而非转换为 line digraph，简化流程并提升效率。
  - **对数空间链路权重预测**：输出 $\mu$ 和 $\sigma$，最终权重为 $\exp(\mu)$，有助于稳定训练。
- **两阶段路由**：
  1. GNN 输出链路权重。
  2. 使用 **Dijkstra 算法**计算最短路径，生成转发表。
- **训练策略**：
  - **预训练**：采用 **Imitation Learning (IL)**，模仿 EIGRP 等静态协议进行 warm-start。
  - **主训练**：基于 **Proximal Policy Optimization (PPO)**，引入：
    - 最大熵探索（类似 SAC）。
    - 自适应温度系数。
    - 早停机制（early stopping）防止过拟合。

### 相比现有方法的优势
| 特性 | 传统方法 (OSPF/EIGRP) | 现有神经方法 (M-Slim, MAGNNETO) | **LOGGIA (本文)** |
|------|------------------------|-------------------------------|------------------|
| 响应速度 | 秒级 | 忽略延迟，理论快 | **显式建模延迟，实际快** |
| 遥测利用 | 有限（静态度量） | 是（但假设无延迟） | **是，且支持延迟感知聚合** |
| 可扩展性 | 高 | 中等（集中式瓶颈） | **高，支持完全分布式部署** |
| 实际可部署性 | 高 | 低（理想化假设） | **高（贴近真实约束）** |

---

## 2. 核心实验方法和设置

### 数据集与网络拓扑
- 使用多种合成与真实网络拓扑进行训练和测试：
  - `mini5`：5节点小型网络。
  - `B4`：Google 数据中心广域网（12节点，17条链路）。
  - `GEANT`：欧洲科研教育网络（27节点，38条链路）。
  - `nx-XS`, `nx-S`, `nx-M`, `nx-L`：一系列可变规模的合成拓扑（6–100 节点）。
- 所有拓扑均配置合理的链路速率（50–200 Mbps）和延迟（1–10 ms）。

### 流量模型
- 模拟 **2秒** 的连续网络运行，划分为 **400个时间步**（每步 $T=5\,\text{ms}$）。
- 流量组成：
  - **80% TCP 流**：模拟数据中心典型应用。
  - **20% UDP 流**：恒定比特率。
- 流到达时间和大小基于真实测量数据生成，并调整强度以确保丢包不可避免，考验路由鲁棒性。

### 评估指标
- **主要指标**：**Goodput（交付数据量，单位 MB）** —— 每回合成功送达目的地的数据总量。
- 辅助指标：
  - 平均包延迟（Delay）
  - 缓冲区负载（Queue Load）
  - TCP 丢弃量（TCP Discard）

### 基线方法对比
- **静态最短路径 (SP) 基线**：
  - `SPRIP`：最小跳数路由。
  - `SPEIGRP`：EIGRP 默认度量（带宽+延迟）。
  - `SPOSF`：OSPF 带宽相关度量。
- **神经路由基线**：
  - `MAGNNETO`：基于 GNN 的多智能体 TE 方法。
  - `M-Slim` 和 `FieldLines`：来自 Boltres et al. (2024)，代表当前先进水平。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Figure 5）

| 方法 | mini5 (MB) | B4 (MB) | nx-XS (MB) | GEANT (MB) |
|------|-----------|--------|----------|----------|
| **最佳 SP 基线** | ~330 | ~325 | ~450 | ~442 |
| **MAGNNETO** | <330 | ~312 | ~420 | ~421 |
| **FieldLines** | <330 | ~296 | ~450 | ~450 |
| **M-Slim** | <330 | ~311 | ~438 | ~438 |
| **LOGGIA (本文)** | **~340** | **~333** | **~460** | **~461** |

✅ **结论**：**只有 LOGGIA 在所有拓扑上持续超越最佳 SP 基线**；其他神经方法在考虑延迟后表现退化甚至不如静态路由。

### 与基线方法的对比结果
- **在延迟感知设置下（Local-Multi 部署）**：
  - LOGGIA 比最佳 SP 基线平均多传输 **5–10%** 的数据。
  - 其他神经方法（M-Slim, MAGNNETO）**无法稳定超越 SP 基线**，尤其在 B4 和 GEANT 上显著落后。
- **跨拓扑泛化能力**（Figure 7）：
  - 即使只在 `mini5`（5节点）上训练，LOGGIA 也能泛化到高达 **100节点** 的 `nx-L` 拓扑，并保持性能优势。
  - 表明其具有良好的**零样本迁移能力**。

### 消融实验结果（Ablation Studies）

#### （1）架构组件消融（Figure 11）
- 移除任一设计都会导致性能下降：
  - 不使用对数空间预测 → 训练不稳定。
  - 使用 line digraph → 效率更低。
  - 减少 MPN 层数或维度 → 表达能力不足。
- ✅ 证明 LOGGIA 的每个设计选择都至关重要。

#### （2）训练策略消融（Figure 14）
- **单独使用 IL 或 BC（Behavioral Cloning）无法达到竞争性能**。
- **IL 预训练 + PPO 微调** 显著提升最终性能和训练稳定性。
- BC 作为独立训练器优于 IL，但作为预训练阶段**劣于 IL**。

#### （3）多智能体训练配置（Figure 15）
- 多种 MAPPO 变体（IPPO, HVPPO）性能相近。
- **中央观察者 + 分布式奖励** 在训练阶段效果最好。
- 但**部署必须使用 Local-Multi** 才能获得最佳性能。

#### （4）延迟影响分析（Figures 6 & 17）
- 引入通信和推理延迟后，所有方法性能下降。
- 性能排序：`Birdseye-Single > Central-Single ≈ Central-Multi < Local-Multi`
- ❗ **唯一能在延迟感知设置下超越基线的是 Local-Multi 部署的 LOGGIA**。
- 推理延迟越大（$\lambda_{ac} \uparrow$），性能越差，凸显**快速推理的重要性**。

---

## 4. 关键结论和发现

### 主要发现
1. **延迟建模至关重要**：
   - 忽视通信和推理延迟会导致对神经路由算法的**过度乐观评估**。
   - 在更真实的延迟感知环境中，大多数现有神经方法**无法超越静态最短路径路由**。

2. **部署架构决定成败**：
   - **完全分布式的 Local-Multi 架构**（每个路由器独立观测与决策）在评估中表现最佳。
   - 中央化架构因额外通信开销而受限，尤其在大规模网络中。

3. **LOGGIA 的有效性与可扩展性**：
   - 是**唯一能在延迟感知设置下持续优于 SP 基线的神经方法**。
   - 具备出色的泛化能力，可在小拓扑上训练，迁移到大网络。

4. **硬件速度直接影响性能**（Appendix D.4）：
   - 更快的 CPU 显著降低推理时间，从而在 $\lambda_{ac}=1$ 设置下提升吞吐量。
   - 表明**专用硬件加速**（如 FPGA/GPU）将是未来方向。

### 方法的局限性
- **单路径路由**：目前仅支持基于单一成本的最短路径，不支持 ECMP 或多路径负载均衡。
- **转发表更新频率限制**：当前为 5ms 粒度，仍远慢于线速转发（纳秒级）。
- **通信开销未压缩**：状态同步消息可能在网络中造成负担，需引入压缩或稀疏更新机制。
- **MDP 建模简化**：假设拓扑不变、确定性行为，未考虑链路故障或动态拓扑变化。

### 未来工作方向
- 扩展至**多路径路由**（Multipath Routing）和更复杂的策略（如 BGP 风格策略）。
- 结合 **P4** 或 **In-Band Network Telemetry (INT)** 实现更高效的遥测采集。
- 探索**轻量化模型设计**与**边缘推理优化**，进一步降低延迟。
- 研究**在线自适应机制**，应对拓扑变更和长期流量漂移。
- 将本框架的思想推广至其他**实时网络控制系统**，如电力网、交通调度等。

---

> **总结一句话**：  
> 本文提出了首个在**真实延迟约束下仍能超越传统路由协议**的神经路由算法 **LOGGIA**，并通过严谨的实验揭示了“**延迟建模”和“完全分布式部署**”是神经路由走向实用化的两大关键支柱。

</details>

---

### 9. [CAWN: Continuous Acoustic Wave Networks for Autoregressive Language Modeling](https://arxiv.org/abs/2604.04250)

**Authors**: Dejan \v{C}ugalj, Aleksandar Jevremovic  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.04250v1  

#### Abstract
Modern Large Language Models (LLMs) rely on Transformer self-attention, which scales quadratically with sequence length. Recent linear-time alternatives, like State Space Models (SSMs), often suffer from signal degradation over extended contexts. We introduce the Continuous Acoustic Wave Network (CA...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CAWN: Continuous Acoustic Wave Networks for Autoregressive Language Modeling

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大型语言模型（LLMs）依赖于 **Transformer** 架构中的 **self-attention** 机制，其计算和内存复杂度为 $O(L^2)$，随序列长度 $L$ 二次增长，严重限制了上下文窗口的扩展。此外，现有的线性时间替代方案如 **State Space Models (SSMs)** 虽然实现了 $O(L)$ 复杂度，但在超长上下文中常出现信号衰减（signal degradation）问题。

CAWN 提出了一种全新的范式，旨在解决：
- 自注意力的 **$O(L^2)$ 内存墙**
- 长距离依赖建模中的 **信号退化**
- 深层网络中残差流的 **表示稀释（dilution）**

---

### 提出的新方法与核心创新
CAWN（**Continuous Acoustic Wave Network**）是一种完全连续的序列混合架构，将离散 token 映射为复数域（Complex Domain）中的声波相位信号，通过波的干涉实现语言建模。

#### 主要创新点包括：

| 创新模块 | 技术描述 |
|--------|---------|
| **Complex-Domain Wave Embedding** | 将隐藏状态动态投影为多头复数域 phasor（相量），跟踪幅度与相位，而非显式计算 token 对之间的关系 |
| **Causal Phase Accumulation** | 因果性的 $O(L)$ 相位累加机制，利用欧拉公式进行真复数旋转，天然编码相对位置信息 |
| **Dual-Gated Selective Phase Resonance** | 双门控机制：<br>• **Frequency-Dependent Retention Gate**：低频保留全局记忆，高频用于局部缓存<br>• **Hard-Threshold Gating with STE**：使用直通估计器（Straight-Through Estimator）将弱信号强制归零，防止数值下溢 |
| **Temporal Syntax Cache** | 在全局波投影前引入 depth-wise 1D-Conv（核大小=3），捕获短期语法依赖 |
| **Depth-wise Harmonic Convolution** | 在谐波通道间应用深度卷积，促进相邻频率交互，提升参数效率 |
| **Block Attention Residuals** | 引入“残差切断”技术，在块边界归档早期 block 状态并重置残差流，深层可通过 depth-wise attention 动态检索原始未稀释的 phasor 状态 |

---

### 相比现有方法的优势

| 维度 | CAWN | SSMs (如 Mamba) | Standard Transformer |
|------|------|------------------|-----------------------|
| 时间复杂度 | $O(L)$ | $O(L)$ | $O(L^2)$ |
| 空间复杂度（推理） | $O(1)$ state-passing | $O(1)$ state | $O(L)$ KV Cache |
| 上下文扩展能力 | 支持百万级 token | 受限于状态压缩质量 | 受限于 $O(L^2)$ 内存 |
| 信号保真度 | 高（物理相位编码） | 中等（递归压缩易失真） | 高但成本高昂 |
| 硬件效率 | 高（SRAM 内 Triton kernel 实现） | 较高 | 依赖 FlashAttention 优化 |

> ✅ **核心优势总结**：CAWN 实现了严格 $O(L)$ 训练和 $O(1)$ 推理状态传递，突破了传统 $O(L^2)$ 内存瓶颈，并在超长上下文中保持信号完整性。

---

## 2. 核心实验方法和设置

### 数据集
- **训练数据**：1000亿 token 的英文语料，构成如下：
  - 50% FineWeb-Edu PDFs
  - 30% DCLM
  - 20% FineWeb-Edu 其他部分
- **验证数据**：`Salesforce/wikitext-103`，用于评估泛化能力和困惑度（Perplexity）
- **Tokenization**：采用 Meta-Llama-2 的 BPE 分词器，词汇表大小 $V=32,000$

### 实验设置
- **模型规模**：CAWN-150M（1.5亿参数）
  - 隐藏维度 $D=896$
  - 层数 $N=16$（每4层一组，共4个 macro-block）
  - Acoustic Heads $H=4$，每个含 $K=64$ 谐波 → 总谐波数 256
- **训练方式**：
  - 使用 `IterableDataset` 实现无限流式训练（infinite streaming）
  - 序列长度 $L=1024$，梯度累积步数=36，微批次大小=7 → 有效 batch size=252
  - 优化器：AdamW（bfloat16 混合精度），学习率 cosine annealing（峰值 $8.0\times10^{-4}$）
  - 连续 phase state ($\Phi_t$) 在 micro-batch 间持久缓存，支持无限长度共振

### 特殊训练策略
- **Learned Contextual Denoising**：随机注入大量无意义“垃圾”token 并随后提出目标查询，迫使输入门 $\beta$ 学会主动过滤噪声，增强抗干扰能力。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Peak VRAM Usage** | 测量不同序列长度下的显存占用，验证是否突破 $O(L^2)$ 内存墙 |
| **Autoregressive Throughput (tok/s)** | 解码速度，测试 $O(1)$ 缓存的实际效率 |
| **Validation Perplexity** | 在 wikitext-103 上的困惑度，衡量语言建模能力 |
| **Zero-Shot Accuracy** | 在 PIQA 和 ARC-Easy 上的准确率，评估逻辑推理能力 |
| **Targeted Semantic Retrieval** | 埋藏关键词于长上下文中，测试极端距离的信息检索能力 |

### 基线方法对比
- **Standard Transformer Baseline**：数学上 $O(L^2)$ 的注意力机制
- **Pythia-160M**：开源的 $O(L^2)$ 模型，作为同规模基准
- **Llama / GPT-2 / SmolLM**：作为成熟模型的性能上限参考

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| 最大处理序列长度 | **2,000,000 tokens** |
| 峰值 VRAM 占用 | **稳定在 8.72 GB**（即使到 2M tokens） |
| 自回归生成速度 | **~52 tok/s**（恒定，不随 $L$ 增加下降） |
| 验证困惑度（5.4B tokens 后） | **~75.00**（wikitext-103） |
| PIQA 准确率（zero-shot） | **60.23%** |
| ARC-Easy 准确率（zero-shot） | **45.45%** |

---

### 与基线方法的对比结果

#### （1）内存效率对比（Table 1 & Figure 2）
| 序列长度 | Standard Transformer | CAWN |
|---------|------------------------|------|
| 4,096 tokens | OOM（8GB GPU） | 3.40 GB |
| 8,192 tokens | — | 4.91 GB |
| 2M tokens | — | **8.72 GB（plateau）** |

> 🔥 **结论**：CAWN 成功绕过 KV Cache 膨胀问题，VRAM 使用量在超过 32k token 后进入平台期，实现真正的 $O(1)$ 缓存。

#### （2）自回归吞吐量（Table 2）
所有模型在 $L \leq 16,384$ 时均能维持高速，但 CAWN 展现出**完全平坦的速度曲线**，证明其不受上下文长度影响。

#### （3）语言建模能力（Figure 3）
- CAWN 在约 **300k 微步（~2.1B tokens）** 时即追平 Pythia-160M 的性能。
- 到 **752k 步（~5.4B tokens）** 时，困惑度降至 **~75**，远低于基线，显示更强的学习效率。

#### （4）零样本推理能力（Table 3）
| 模型 | 参数量 | 训练量 | PIQA | ARC-Easy |
|------|--------|--------|------|----------|
| Pythia-160M | 160M | 2.1B | 55.50% | 30.64% |
| **CAWN (ours)** | **150M** | **5.0B** | **60.23%** | **45.45%** |
| SmolLM（上限） | 135M | 600B | 68.55% | 61.74% |

> ✅ 尽管参数更少、训练量仅为十分之一，CAWN 已显著超越早期阶段的标准 Transformer，证明其**连续波干涉机制可有效捕捉语义与逻辑结构**。

#### （5）极端上下文检索能力（Table 4）
| 距离 | Tokens | 检索成功率 | VRAM |
|------|--------|------------|------|
| 标准上下文 | 650 | √ | 2.42 GB |
| 中等扩展 | 37,800 | √ | 7.34 GB |
| 极端长距 | 1,000,000 | √ | 8.72 GB |
| 相位边界 | 2,000,000 | ❌（仅 Green 失败） | 8.72 GB |

> 🧪 **发现**：在 200万 token 距离处出现选择性失败，推测是由于特定谐波频段发生**破坏性干涉（destructive interference）** 或 bfloat16 数值误差累积所致。

---

### 消融实验（隐含分析）
虽然未明确列出消融表，但从设计中可推断以下关键组件的作用：
- **Hard-Threshold Gating + STE**：防止弱信号泄漏，保障长期稳定性
- **Temporal Syntax Cache**：解耦短程语法与长程语义，避免相位污染
- **Block Attention Residuals**：防止深层残差稀释，保留早期纯净 phasor
- **Float32 Phase Accumulation**：确保百万级累加不因 float16 截断而失效

---

## 4. 关键结论和发现

### 主要结论
1. ✅ **语言可以被建模为声波共振现象**：离散 token 间的语义关系可通过复数域中的构造性和破坏性波干涉来表达。
2. ✅ **无需 self-attention 也能实现高效语言建模**：CAWN 完全摒弃了 $O(L^2)$ 注意力，改用 $O(L)$ 相位累加 + $O(1)$ 状态传递。
3. ✅ **支持真正意义上的无限上下文**：借助 chunked prefill 和固定大小 phase state，模型可在恒定显存下处理高达 **2 million tokens** 的上下文。
4. ✅ **具备原生抗噪与上下文去噪能力**：通过训练策略，模型学会主动关闭噪声通道，实现“显式学习的关联召回”。

---

### 方法的局限性
| 问题 | 描述 |
|------|------|
| **硬件依赖性强** | 依赖定制 Triton kernel 和 float32 精度支持，难以直接部署于普通框架 |
| **极端距离信号丢失** | 在 2M token 处出现选择性失败，可能源于谐波周期性抵消 |
| **当前规模较小** | 当前最大仅 150M 参数，尚未验证在 7B+ 规模下的表现 |
| **缺乏理论解释工具** | 复数相位状态难以可视化和解释，调试困难 |

---

### 未来工作方向
1. **扩大模型规模**：推进至 **1B–7B 参数级别**，并在 **万亿 token 级别数据集** 上预训练
2. **谱分析研究**：对失败案例进行频谱追踪，绘制 amplitude tracker 曲线，定位相位抵消机制
3. **混合精度优化**：探索如何在保持稳定性的前提下降低 float32 使用范围
4. **无限上下文 API 设计**：构建基于 chunked state-passing 的推理引擎，支持工业级长文本处理
5. **跨模态扩展**：尝试将 acoustic wave 思路应用于语音、音乐或多模态序列建模

---

## 总结
> **CAWN 开创性地将自然语言视为一种“声学共振系统”，用复数域中的连续波干涉取代离散的矩阵注意力，在理论上和实践中同时实现了 $O(L)$ 训练与 $O(1)$ 推理的突破。它不仅打破了传统 Transformer 的内存墙，还展示了在百万级上下文中依然稳健的信息检索能力，为下一代无限上下文基础模型提供了全新路径。**

</details>

---

### 10. [Communication-Efficient Distributed Learning with Differential Privacy](https://arxiv.org/abs/2604.02558)

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

# **论文总结：Communication-Efficient Distributed Learning with Differential Privacy**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
本文针对**非凸分布式学习**中的两个核心挑战：
- **通信效率低**：传统分布式优化算法需要频繁通信，导致高带宽消耗；
- **隐私泄露风险**：共享的模型参数可能被用于推断参与方的私有训练数据（如通过梯度反演攻击）。

现有方法通常在效率与隐私之间存在权衡，难以同时满足高性能和强隐私保护。

---

### **提出的新方法与创新点**
作者提出了名为 **LT-ADMM-DP (Local Training ADMM with Differential Privacy)** 的新算法，其核心创新包括：

- **本地训练（Local Training）机制**：
  - 代理（agent）在多个本地 epoch 中进行独立更新，仅周期性地与其他邻居通信，显著减少通信频率，提升 **communication-efficiency**。
  
- **差分隐私（DP）保障机制**：
  - 在本地梯度计算中引入双重保护：
    1. **梯度裁剪（gradient clipping）**：限制单个样本对梯度的影响；
    2. **高斯噪声注入（additive Gaussian noise）**：扰动梯度以实现 **Rényi Differential Privacy (RDP)**。
  - 结合子采样放大（subsampling amplification），进一步增强隐私保护。

- **理论保证**：
  - 首次为结合 **local training + ADMM + DP** 的框架提供了统一的收敛性和隐私性分析。
  - 证明了算法收敛到目标函数的一个**稳定点附近**，且误差受噪声方差、梯度异质性等因素控制。
  - 形式化证明了该算法满足 $(\epsilon, \delta)$-DP。

---

### **相比现有方法的优势**
| 维度 | LT-ADMM-DP | 现有方法（如 PORTER、PriSMA） |
|------|------------|-------------------------------|
| **通信效率** | 更少通信轮次（T local steps per global step） | 多数每步都需通信 |
| **隐私机制** | 显式结合 clipping + noise + RDP 分析 | 多基于简单噪声添加 |
| **收敛性分析** | 支持非凸目标下的有界收敛 | 部分假设强凸或均匀数据分布 |
| **实用性** | 更适合资源受限边缘设备网络 | 对通信要求更高 |

> ✅ **优势总结**：在相同隐私预算下，实现了更快的收敛速度和更高的分类准确率。

---

## **2. 核心实验方法和设置**

### **数据集与任务**
- **任务类型**：二分类任务（binary classification）
- **局部损失函数形式**：
  $$
  f_i(x) = \frac{1}{m} \sum_{h=1}^{m} \left[\log(1+\exp(-b_{i,h} a_{i,h}^T x)) + \frac{\lambda}{2}\|x\|^2\right]
  $$
  即带有 $\ell_2$ 正则化的逻辑回归（nonconvex due to regularization? 注：实际是凸的，但论文称“nonconvex learning”，可能是泛指深度学习场景）。
- **合成数据生成**：
  - $N=10$ 个 agent 构成环形网络（ring network）
  - 每个 agent 拥有 $m=1000$ 条样本，特征维度 $n=5$
  - mini-batch size: $|B|=8$

---

### **实验设置**
- **算法对比**：
  - **LT-ADMM-DP**（本文方法）
  - **PORTER** [21]：基于梯度裁剪和压缩的去中心化方法
  - **PriSMA** [22]：支持差分隐私的分布式优化器
- **超参数配置**：
  - LT-ADMM-DP: $\gamma = \beta = 0.1$, $p=0.1$, $C=1$, $T=4$, $K=4000$
  - 所有方法调整噪声标准差以达到相同的 **总隐私预算 $\epsilon \approx 19.6$**, $\delta = 10^{-4}$
- **时间成本建模**：
  - 局部梯度计算耗时：$t_g = 0.1$
  - 一轮通信耗时：$t_c = 1$
  - 总时间按表 I 计算：

| Algorithm | Time per T iterations |
|---------|------------------------|
| PORTER  | $T(t_g + 2t_c)$        |
| PriSMA  | $T(2t_g + t_c)$        |
| LT-ADMM-DP | $T t_g + t_c$         |

> ⏱️ 可见 LT-ADMM-DP 时间开销最小。

---

### **评估指标**
1. **最优误差（Optimality Error）**：$\|\nabla F(x_k)\|$，衡量接近驻点的程度
2. **分类准确率（Classification Accuracy）**
3. **总运行时间（Wall-clock time）**：考虑通信与计算的实际耗时
4. **隐私预算 $\epsilon$**：在固定 $\delta=10^{-4}$ 下比较

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 指标 | LT-ADMM-DP | PORTER | PriSMA |
|------|------------|--------|--------|
| 最终分类准确率 | **~89.5%** | ~86.5% | ~87.0% |
| 达到相同准确率所需时间 | **最快**（约节省 30–40%） | 较慢 | 最慢 |
| 总时间（K=4000） | **≈ 4100 单位** | ≈ 8400 | ≈ 9200 |
| 收敛速度（误差下降斜率） | 最陡峭 | 平缓 | 中等 |

> 📊 图1(a)(b) 显示，在相同隐私预算下，LT-ADMM-DP 不仅收敛更快，而且最终精度更高。

---

### **与基线方法的对比结果**
- **优于 PORTER 和 PriSMA**：
  - 在相同 $\epsilon$ 下，LT-ADMM-DP 实现了 **更高的分类准确率** 和 **更低的时间开销**。
  - 尤其在早期阶段表现出明显加速，得益于本地训练减少通信阻塞。
- **通信效率显著提升**：
  - 每 $T=4$ 轮本地更新才通信一次，而其他方法多为每轮通信。

---

### **消融实验（隐含分析）**
虽然未明确列出消融实验表格，但从理论分析可推导出以下影响因素的作用：

| 因素 | 影响 |
|------|------|
| **增加 $T$（本地训练步数）** | 加快收敛，但增大稳态误差（trade-off） |
| **减小噪声 $\sigma$** | 提升模型性能，但降低隐私保护强度 |
| **更强连通图（higher algebraic connectivity $\lambda_2$）** | 允许更大步长 $\gamma$，加快收敛 |
| **梯度裁剪阈值 $C$** | 控制敏感度，直接影响 $\Delta_2 f$ 和隐私成本 |

> 🔍 这些关系在 Theorem 1 和 Remark 1 中得到理论支持。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **LT-ADMM-DP 成功平衡了效率与隐私**：
   - 利用 local training 显著减少通信；
   - 通过 clipped noisy gradients 实现严格 $(\epsilon,\delta)$-DP。
2. ✅ **在相同隐私预算下性能超越 SOTA 方法**：
   - 收敛更快、精度更高、总耗时更短。
3. ✅ **理论与实践一致**：
   - 收敛性分析表明误差受噪声、数据异质性、网络拓扑共同影响；
   - 实验验证了这些趋势。

---

### **方法的局限性**
- ❗ **非凸问题仅保证收敛至邻近稳定点**，而非全局最优；
- ❗ 假设数据异质性有界（Assumption 3），在极端非IID场景下性能可能下降；
- ❗ 当前裁剪阈值 $C$ 是固定的，未自适应调整，可能导致信息损失或隐私浪费；
- ❗ 实验基于合成数据，尚未在真实大规模数据集（如 CIFAR、MNIST）上验证。

---

### **未来工作方向**
1. **自适应裁剪策略（adaptive clipping）**：
   - 动态调整 $C$ 以平衡隐私与效用。
2. **处理高度异构数据（highly heterogeneous data）**：
   - 引入个性化模型或正则化项缓解 drift。
3. **扩展至联邦学习架构**：
   - 支持部分参与、异步更新等更现实设定。
4. **探索其他隐私机制组合**：
   - 如结合 **compression + DP** 进一步降低通信与隐私开销。

---

> ✅ **总体评价**：本文提出了一种实用性强、理论扎实的分布式学习框架，在隐私保护与系统效率之间取得了良好平衡，为边缘智能和隐私敏感应用提供了有力工具。

</details>

---

### 11. [PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training](https://arxiv.org/abs/2604.03675)

**Authors**: Erhan Zhang, Yiqun Chen, Zechun Niu, Wei Yang, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.03675v1  

#### Abstract
In agentic search, large language models (LLMs) are trained to perform multi-turn retrieval and reasoning for complex tasks such as multi-hop question answering (QA). However, current search-based Reinforcement Learning (RL) methods suffer from two core limitations: expensive long-horizon rollouts a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Reinforcement Learning (RL)** 的 **agentic search** 方法在训练过程中面临两个核心挑战：
- **Rollout 利用率低**：一次完整的多轮搜索轨迹（multi-turn search trajectory）生成成本高昂，但通常仅作为单个训练样本使用，中间状态未被有效利用。
- **Reward Sparsity**：监督信号仅来自最终答案的正确性（outcome reward），缺乏对中间步骤的有效反馈，导致长期信用分配（credit assignment）困难。

### 提出的新方法：PRAISE
作者提出 **PRAISE**（Prefix-based Rollout reuse for Agentic search with Intermediate Step rEwards），其核心思想是：
- 利用搜索轨迹的**前缀结构**（prefix structure），从完整轨迹中提取多个中间状态 $ s_t $（即前 $ t $ 轮的信息状态）。
- 在每个前缀状态 $ s_t $ 上，让模型生成一个**中间答案** $ y_t $，并计算其得分。

该方法通过两种方式提升训练效率与效果：
1. **Rollout Reuse**：将一条完整轨迹转化为 $ T+1 $ 条前缀样本，显著提高数据利用率。
2. **Intermediate Step Rewards**：通过比较相邻前缀答案得分的变化（$ v_t - v_{t-1} $），构建**过程奖励**（process reward），实现细粒度的 step-level 监督。

### 相比现有方法的优势
| 维度 | PRAISE 的优势 |
|------|----------------|
| **无需额外标注** | 不依赖 gold sub-questions、人工标注证据等外部监督信号。 |
| **无需独立 reward model** | 使用**共享模型**（shared model）同时进行搜索决策和前缀评估，避免 evaluator-policy 不一致问题。 |
| **联合优化** | 搜索策略与前缀评估能力共同演进，提升一致性与训练稳定性。 |
| **通用性强** | 适用于任意可验证任务（verifiable rewards），不局限于数学或代码推理。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **训练集**：
  - `HotpotQA` (Yang et al., 2018)
  - `2WikiMultihopQA` (Ho et al., 2020)
- **测试集**（跨数据集评估）：
  - `NQ` (Natural Questions)
  - `HotpotQA`
  - `2WikiMultihopQA`
  - `Bamboogle` (Press et al., 2022)
  - `MuSiQue` (Trivedi et al., 2022)

### 实验设置与评估指标
- **Base Model**：`Qwen2.5-7B`
- **Retriever**：`E5` 模型
- **检索语料库**：Wikipedia
- **训练框架**：基于 **PPO** 的 RLVR（Reinforcement Learning with Verifiable Rewards）
- **评估指标**：
  - **F1** 和 **EM**（Exact Match）
- **关键超参数**：
  - 过程奖励权重 $ \alpha = 0.5 $

### 基线方法对比
分为三类：
1. **非智能体方法**：
   - Direct LLM
   - Naive RAG (Lewis et al., 2020)
2. **Agentic Search 方法**：
   - Search-o1 (Li et al., 2025)
   - Search-R1 (Jin et al., 2025)
   - R1-Searcher (Song et al., 2025)
3. **过程监督方法**：
   - StepSearch (Zheng et al., 2025)
   - ReasonRAG (Zhang et al., 2025)
   - TIPS (Xie et al., 2026)

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 1）
| Method | Avg. F1 | Avg. EM |
|--------|---------|--------|
| R1-Searcher (Best Baseline) | 45.94 | 34.21 |
| **PRAISE (Ours)** | **50.02** | **37.46** |
| **Improvement** | **+4.08** | **+3.25** |

> ✅ 在所有五个 benchmark 上，PRAISE 均取得 **SOTA 性能**，且显著优于最强 baseline。

#### 典型任务提升示例：
- **HotpotQA (F1)**: +5.07
- **2Wiki (F1)**: +5.67
- **Bamboogle (EM)**: +4.00

表明 PRAISE 特别擅长处理需要**持续证据积累**的多跳推理任务。

---

### 消融实验结果（Table 2）

| 变体 | HotpotQA F1 ↓ | 2Wiki F1 ↓ | MuSiQue F1 ↓ |
|------|---------------|------------|--------------|
| Full PRAISE | 60.62 | 58.14 | 30.73 |
| w/o joint optimization (Policy as Evaluator) | -3.46 | -0.33 | -3.27 |
| w/o process reward ($\alpha=0$) | -1.82 | +0.50 | -3.11 |
| w/o prefix evaluator | -2.20 | -0.14 | -1.94 |

#### 关键发现：
- **Joint optimization 至关重要**：冻结 evaluator 或不联合训练会导致性能下降，说明 evaluator 必须随 policy 一起演化以保持一致性。
- **Process reward 显著增益**：移除过程奖励会明显降低性能，尤其在复杂任务上（如 MuSiQue）。
- **Prefix evaluator 自身可学习**：图 2 显示，在 joint training 下，evaluator 对各阶段前缀的评分能力持续提升。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **最大化 rollout 价值** 是提升 agentic search 训练效率的关键路径。
2. ✅ **前缀重用 + 过程奖励** 的组合能有效缓解 reward sparsity 并增强 credit assignment。
3. ✅ **共享模型 + 联合优化** 架构优于分离式设计，提升了系统简洁性与训练稳定性。
4. ✅ PRAISE 在多种 multi-hop QA 任务上均表现出强鲁棒性和泛化能力。

### 方法的局限性
- 当前方法依赖于**可验证的答案评分函数**（verifiable scoring function），难以直接应用于开放式生成任务（如创意写作）。
- 前缀答案生成引入额外计算开销（仅限训练阶段），可能影响大规模训练吞吐量。
- 所有实验基于固定最大搜索步数，动态终止机制尚未探索。

### 未来工作方向
- 将 PRAISE 扩展至其他 long-horizon 决策任务，如代码生成、规划与控制。
- 探索更高效的 prefix sampling 策略（如仅采样高价值前缀）。
- 结合 curriculum learning 动态调整 process reward 权重 $ \alpha $。
- 研究如何将 prefix evaluator 蒸馏回主策略模型以减少部署复杂度。

---

> 📌 **一句话总结**：  
> PRAISE 通过**前缀重用**与**相邻前缀收益差构建过程奖励**，实现了高效、细粒度的 agentic search 训练，在无需额外标注或 reward model 的前提下，显著超越现有方法。

</details>

---

### 12. [FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving](https://arxiv.org/abs/2604.02715)

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
现代 **Mixture-of-Experts (MoE)** 大语言模型虽然通过稀疏激活提升了参数容量，但在推理过程中存在严重的**内存效率问题**：
- 所有专家权重（expert weights）在推理期间始终驻留在 GPU 内存中，即使大部分处于闲置状态。
- 这些闲置权重与对性能至关重要的运行时状态（如 **KV Cache**）竞争有限的 GPU 显存资源。
- 由于 **KV Cache 容量直接决定服务吞吐量（throughput）**，这种内存分配方式导致显存利用率低下、推理性能受限。

### **提出的新方法与思路**
作者提出了 **FluxMoE**，一种全新的 MoE 推理系统，其核心思想是：  
> **将专家参数从持久化的 GPU 驻留状态中解耦，转而视为可按需流式加载的临时资源。**

这一理念被形式化为 **Expert Paging（专家分页）** 抽象，实现了一个新的执行模型：
```
model = compute graph + streamed parameters
```

### **三大核心技术机制**
1. **PagedTensor**  
   - 提供张量虚拟化抽象，将逻辑张量地址与物理 GPU 内存解耦。
   - 类似于操作系统虚拟内存管理，支持动态映射和回收物理内存块。
   - 不修改底层计算内核（如 PyTorch/Triton），兼容性强。

2. **带宽均衡的存储层次（Bandwidth-Balanced Storage Hierarchy）**
   - 构建多级存储后端：压缩的 GPU 显存 + 主机 DRAM。
   - 采用比例分配策略，使各后端的加载时间对齐，最大化整体 I/O 吞吐。
   - 对专家权重中的 **指数位（exponent bits）进行选择性 Huffman 编码压缩**，节省约 20% 显存占用。

3. **预算感知的驻留规划器（Budget-Aware Residency Planner）**
   - 动态调节保留在 GPU 中的专家比例 α。
   - 基于 `compute-to-load ratio`（p = T_comp / T_load）闭环控制：
     - 若 p > 1：计算主导，减少 α，释放显存给 KV Cache。
     - 若 p < 0.9：I/O 主导，增加 α，提升加载速度。
   - 优先保障 KV Cache 内存需求，避免因 swapping 导致性能下降。

### **相比现有方法的优势**
| 维度 | 传统方法（如 vLLM） | FluxMoE |
|------|---------------------|--------|
| 参数驻留 | 全部专家常驻 GPU | 按需流式加载，仅保留必要部分 |
| 显存利用 | 被大量闲置权重占据 | 更多用于 KV Cache 和激活缓冲区 |
| 可扩展性 | 受限于单卡显存容量 | 支持总参数远超 GPU 显存的模型部署 |
| 性能表现 | 在大 batch/context 下易出现瓶颈 | 显著提升吞吐量，尤其在内存密集场景 |

---

## **2. 核心实验方法和设置**

### **使用的模型与数据集**
- **模型**：
  - `Mixtral-8×7B-Instruct`（47B 参数）
  - `Qwen3-Next-80B-A3B-Instruct`（80B 参数）
- **数据集**：
  - **ShareGPT**：真实用户对话数据集，用于构建推理请求负载。

### **实验设置**
- **硬件平台**：
  - 4× NVIDIA L40 GPUs（每卡 48GB GDDR6）
  - Intel Xeon Platinum 8358 CPU，2TB 主机 DRAM
- **并行策略**：
  - 使用 Tensor Parallelism（TP=4 或 TP=2）以适应大模型。
- **测试场景**：
  - **性能边界场景（Performance-bound）**：显存充足，vLLM 可正常运行。
  - **容量边界场景（Capacity-bound）**：显存紧张，vLLM 会 OOM。
- **输入配置**：
  - Batch Size：32–256
  - Context Length：1,024–4,096 tokens

### **评估指标**
- **主要指标**：**Aggregate Throughput (tokens/sec)**  
  衡量系统整体生成能力，适用于高并发服务场景。
- **次要分析指标**：
  - KV Cache 占用趋势
  - 实际加载延迟 vs 计算延迟
  - 显存使用分布

### **基线方法对比**
| 基线 | 描述 |
|------|------|
| **vLLM** | 工业界标准框架，所有专家必须驻留 GPU，KV Cache 不足时向主机 swap |
| **vLLM-O** | 改进版 vLLM，支持专家 offloading 到主机 DRAM，但无压缩机制 |
| **FluxMoE-H** | 消融版本，仅使用粗粒度层级别压缩与卸载，缺乏细粒度带宽优化 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
#### ✅ **Exp#1：性能边界场景（Qwen3-Next-80B-A3B，TP=4）**
- 在 **batch=256, ctx=4096** 条件下：
  - **FluxMoE 达到最高 3.0× 吞吐提升** 相比原生 vLLM。
  - 相比 vLLM-O 提升达 **3.7×**。
- 小 batch（32）下略有开销（~64% 性能），源于解压成本；但随 batch 增大优势迅速显现。

#### ✅ **Exp#2：容量边界场景（Mixtral-8×7B，TP=2）**
- vLLM 因 OOM 无法运行。
- FluxMoE 成功部署并实现高效推理：
  - 在 batch=256, ctx=4096 时达到 **~4.7 tokens/s**。
  - 相比 vLLM-O 提升 **28.5%–22.9%**。
  - 显著优于 FluxMoE-H，证明了带宽均衡调度的有效性。

#### ✅ **Exp#3：动态驻留自适应稳定性**
- 动态调整 α 可有效应对 KV Cache 增长带来的内存压力。
- 在连续推理中，**吞吐未低于固定 α=1.0 的基准水平**，说明 I/O 开销被完全隐藏。
- 最终释放约 **5.3 GB 显存**，可用于多租户共置或其他任务。

#### ✅ **Exp#4：PagedTensor 开销分析**
- 在所有专家均驻留 GPU 的理想条件下测试管理开销。
- **最大额外开销仅为 3.0%**（出现在 batch=64, ctx=4096）。
- 表明 PagedTensor 的虚拟化机制引入的运行时代价极低。

---

## **4. 关键结论和发现**

### **主要发现**
1. **MoE 推理的瓶颈本质是内存资源错配**：
   - 当前系统将“静态权重”与“动态状态”同等对待，导致 KV Cache 资源受限。
   - **解耦专家驻留是突破性能上限的关键路径**。

2. **Expert Paging 是可行且高效的范式转变**：
   - 通过流式加载 + 异步流水线，可以将 I/O 完全掩盖在计算之下。
   - 即使模型总大小远超 GPU 显存，仍可维持高性能推理。

3. **带宽均衡设计至关重要**：
   - 单纯 offload 会导致 PCIe 成为瓶颈。
   - FluxMoE 的多级存储协同调度显著缓解 I/O 瓶颈。

4. **动态调控优于静态配置**：
   - 固定压缩/卸载策略无法适应变化的工作负载。
   - 预算感知控制器实现了自动平衡，在不同场景下保持最优性能。

### **局限性**
- **依赖稳定的 I/O 带宽**：若 PCIe 或压缩吞吐不稳定，可能影响流水线效率。
- **当前原型未支持训练阶段**：仅聚焦于推理优化。
- **未探索更复杂的预测性预取机制**：目前基于确定性的滑动窗口加载。

### **未来工作方向**
- 扩展至 **训练场景**，支持大规模 MoE 模型的分布式训练内存优化。
- 结合 **active expert prediction** 实现更智能的预取策略。
- 探索与 **disaggregated serving 架构**（如 Mooncake、DistServe）深度集成。
- 支持 **动态 KV Cache 扩容**，进一步释放 FluxMoE 节省的显存潜力。

---

> **总结一句话**：  
> **FluxMoE 通过“专家分页”机制，将 MoE 模型的专家参数从 GPU 显存中解放出来，实现了高达 3.0× 的推理吞吐提升，为下一代高密度 MoE 服务提供了高效、可扩展的新范式。**

</details>

---

### 13. [STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation](https://arxiv.org/abs/2604.02756)

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

# STDDN: A Physics-Guided Deep Learning Framework for Crowd Simulation 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 crowd simulation 方法存在以下两大核心挑战：
- **微观建模的误差累积**：主流方法将人群视为独立个体轨迹的集合，缺乏对宏观物理规律（如质量守恒）的建模，导致长期预测中误差不断积累，仿真稳定性差。
- **深度学习方法效率低下**：基于 diffusion model 或自回归生成的方法推理速度慢、计算开销大，难以应用于大规模实时场景。

### 提出的新方法与创新思路
作者提出 **STDDN (Spatio-Temporal Decoupled Differential Equation Network)**，一种融合宏观物理约束与微观轨迹预测的新型框架。其核心思想是：
- 将人群运动类比为流体流动，引入流体力学中的 **Continuity Equation** 作为强物理先验，指导轨迹预测过程。
- 构建一个由 **Neural ODE** 驱动的宏观密度演化模块，通过微分方程建模群体密度随时间的变化，并将其与微观轨迹预测网络进行端到端耦合。

#### 主要创新点包括：
- ✅ **统一的宏-微耦合建模范式**  
  首次将 Continuity Equation 以可微分形式嵌入到深度学习框架中，实现宏观物理规律对微观行为预测的显式正则化，显著提升仿真的物理一致性和全局稳定性。

- ✅ **物理可解释的动态图网络设计**  
  设计 **Density-Velocity Coupled Graph Learning (DVCG)** 模块，利用当前时刻的速度作为入边、下一时刻的预测速度作为出边构建动态图，显式建模跨网格的密度通量（flux），增强模型可解释性。

- ✅ **两个关键的可微分结构设计**
  - **Differentiable Density Mapping (DDM)**：采用基于 RBF 的软分配策略，避免传统硬划分带来的梯度不连续问题。
  - **Continuous Cross-Grid Detection (CGD)**：使用 Jensen-Shannon 散度量化行人跨越网格的程度，并通过 sigmoid 映射生成连续的跨网格掩码，确保质量守恒且支持反向传播。

- ✅ **高效推理机制**  
  在训练阶段使用 Neural ODE 进行物理正则化；在推理阶段仅需调用训练好的 `f₀` 网络进行单步前向预测，无需 ODE 求解器，大幅降低延迟。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **准确性** | 显著优于 SOTA 方法，在多个真实数据集上 MAE 和 OT 指标领先 |
| **稳定性** | 宏观物理约束有效抑制长期预测中的误差累积 |
| **效率** | 推理速度比 diffusion-based 方法快数倍至数十倍 |
| **物理一致性** | 更好地保持密度守恒，减少碰撞与不合理聚集 |

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个公开的真实世界轨迹数据集上进行了全面评估：
- **GC**：高密度城市街道场景，选取 300 秒密集片段
- **UCY**：包含 ZARA1、ZARA2 和 UCY 三个子场景，平均密度较高
- **ETH** 与 **HOTEL**：校园与酒店环境，行人速度快、交互复杂

所有数据均经过坐标转换与立方插值处理，统一时间步长为 Δt = 0.08s。

### 实验设置与评估指标

#### 评估指标
| 指标 | 含义 |
|------|------|
| **MAE** | 平均绝对误差，衡量位置预测精度 |
| **OT (Optimal Transport Distance)** | 最优传输距离，评估轨迹分布相似性 |
| **#Pars** | 模型参数量，反映模型大小 |
| **Latency (ms)** | 单帧推理延迟，评估运行效率 |
| **FPS** | 每秒可模拟帧数，体现吞吐能力 |
| **FDE**, **#Colli**, **DEA** | 分别评估终点误差、碰撞次数、密度估计准确性等物理合理性 |

#### 基线方法对比
分为三类进行比较：
- **Physics-based**: SFM (Social Force Model), CA (Cellular Automaton)
- **Data-driven**: STGCNN, PECNet, MID
- **Physics-guided**: PCS, NSP, SPDiff

其中 SPDiff 是当时最先进的 physics-informed diffusion model，作为主要对比对象。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & Table 2）

| Dataset | 方法 | MAE ↓ | OT ↓ | Latency (ms) ↓ |
|--------|------|-------|------|----------------|
| **GC** | SPDiff | 0.9116 | 1.3925 | 206.99 |
|        | **Ours (STDDN)** | **0.8875** | **1.3582** | **86.85** |
| **UCY** | SPDiff | 1.8760 | 4.0564 | 471.05 |
|         | **Ours (STDDN)** | **1.7747** | **3.6503** | **44.66** |
| **ETH** | SPDiff | 0.5527 | 0.8706 | 81.41 |
|         | **Ours (STDDN)** | **0.5185** | **0.6918** | **30.57** |
| **HOTEL** | SPDiff | 0.3380 | 0.1646 | 68.57 |
|           | **Ours (STDDN)** | **0.2952** | **0.1445** | **17.50** |

> ✅ 所有数据集上，STDDN 在 **准确率（MAE/OT）** 和 **推理效率（Latency/FPS）** 上均全面超越 SOTA 方法。

#### 性能提升总结：
- **推理速度提升**：相比 SPDiff，延迟降低 **50%~90%**，FPS 提升显著（见 Table 6 & 7）
- **参数量更少**：模型更加轻量化（如 UCY 上仅 0.07M 参数 vs SPDiff 的 0.22M）
- **物理合理性更强**：FDE 更低、#Colli 更少、DEA 更优（见 Appendix 表格）

### 消融实验结果（Ablation Study, Table 3）

消融实验验证了各组件的重要性：

| 变体 | GC MAE | UCY MAE | 结论 |
|------|--------|---------|------|
| **Full Model (Ours)** | 0.8875 | 1.7747 | — |
| w/o ODE | 1.3784 | 2.4867 | 移除 Neural ODE 导致严重性能下降，证明物理约束对抑制误差累积至关重要 |
| w/o Cross-net (CGD) | 0.9784 | 1.8926 | 跨网格检测失效后密度守恒被破坏，性能明显退化 |
| w/o NN loss | 1.2387 | 1.9327 | 仅依赖物理损失无法捕捉复杂人类行为，需结合 data-driven 学习 |
| w/o NE | 0.8921 | 1.7917 | 节点嵌入有助于压缩参数并保留表达力 |
| Discrete NN | 0.8875 | 1.7747 | 性能接近完整模型，说明模型本质是在离散时空格点上传导人流 |

> 🔍 发现：虽然使用 Neural ODE，但 Euler 求解器表现最优，表明任务更适合离散时间建模而非连续动力系统假设。

---

## 4. 关键结论和发现

### 主要发现
1. **宏观物理规律可有效指导微观轨迹预测**  
   引入 Continuity Equation 作为结构性约束，能显著提升 crowd simulation 的长期稳定性和物理一致性。

2. **宏-微协同建模优于纯微观或纯物理方法**  
   STDDN 成功融合了 data-driven 方法的强大表征能力和 physics-based 方法的理论可靠性，在准确性和合理性之间取得更好平衡。

3. **高效的端到端训练 + 快速推理架构可行**  
   利用 Neural ODE 进行训练时的物理正则化，而在推理时剥离 ODE 求解器，实现了“训练时严谨、推理时高效”的设计范式。

4. **可微分工程设计对物理一致性至关重要**  
   DDM 与 CGD 模块解决了离散化带来的梯度断裂问题，使物理约束能够真正参与优化过程。

### 方法的局限性（Limitations）
- **边界效应未完全建模**：当前框架假设封闭系统，未显式建模人员进出（source/sink）。未来可通过扩展 Continuity Equation 加入源项 $ \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = S $ 来解决。
- **网格离散化带来计算负担**：尽管 NE 模块缓解了问题，但在超大区域仍可能受限于内存。
- **强物理约束可能压制隐含行为模式**：过于严格的守恒可能限制模型拟合某些非典型但合理的局部动态。

### 未来工作方向
1. 探索 **soft-constraint 或 adaptive weighting 机制**，动态平衡 data-driven 与 physics-driven 的影响。
2. 引入 **multi-scale graph structures** 或 **implicit neural representations** 替代固定网格，提升空间建模灵活性。
3. 扩展模型以支持 **open-world simulation**，预测入口/出口区域及人流源汇。
4. 应用于更大规模场景，如 **emergency evacuation planning** 和 **intelligent transportation systems**，推动工程落地。

---

> 📌 **代码开源地址**：[https://github.com/liuzjin/STDDN](https://github.com/liuzjin/STDDN)

</details>

---

### 14. [Combee: Scaling Prompt Learning for Self-Improving Language Model Agents](https://arxiv.org/abs/2604.04247)

**Authors**: Hanchen Li, Runyuan He, Qizheng Zhang, Changxiu Ji, Qiuyang Mang, Xiaokun Chen, Lakshya A Agrawal, Wei-Liang Liao, Eric Yang, Alvin Cheung, James Zou, Kunle Olukotun, Ion Stoica, Joseph E. Gonzalez  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.04247v1  

#### Abstract
Recent advances in prompt learning allow large language model agents to acquire task-relevant knowledge from inference-time context without parameter changes. For example, existing methods (like ACE or GEPA) can learn system prompts to improve accuracy based on previous agent runs. However, these me...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Combee: Scaling Prompt Learning for Self-Improving Language Model Agents

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有 **prompt learning** 方法（如 ACE、GEPA）主要在单智能体或低并行度场景下运行，难以高效地从大量 **agentic traces** 中学习。当简单增加并行度（batch size）时，负责整合反思（reflection）的 **aggregator LLM** 会因输入过长而出现“**context overload**”现象——即只能保留泛化性强但价值较低的通用模式，丢失对下游任务至关重要的具体、高价值信息，导致最终准确率显著下降。

例如，在 Formula 数据集上，将 batch size 从 1 增加到 100，准确率从 87.0% 下降到 72.5%。

### 提出了什么新方法或新思路
作者提出 **Combee**，一个用于可扩展 prompt learning 的分布式框架，其核心是 **Map-Shuffle-Reduce** 范式：
- **Map**：多个 agent 并行执行任务并生成反思。
- **Shuffle**：引入 **augmented shuffling** 机制，复制并打乱反思条目，确保每个反思有更多机会被聚合。
- **Reduce**：采用 **parallel scan aggregation** 算法，通过分层扫描方式聚合局部更新，避免单次处理过长上下文。

此外，Combee 还引入了 **dynamic batch size controller**，自动平衡训练延迟与学习质量。

### 相比现有方法的优势
- **高效并行**：支持高并行度下的 prompt learning，实现高达 **17× 的加速**。
- **保持甚至提升质量**：在显著减少训练时间的同时，达到与顺序学习相当甚至更高的准确率。
- **无质量损失**：解决了 naive scaling 导致的 context overload 问题。
- **框架无关**：可集成到现有的 generate-reflect-update 框架（如 ACE、GEPA），无需大幅修改。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **Agentic Benchmarks**：
  - **AppWorld**：评估多步 API 交互任务，使用 **Task Goal Completion (TGC)** 和 **Scenario Goal Completion (SGC)** 作为指标。
  - **Terminal-Bench 2.0**：包含 89 个命令行任务，测试软件工程能力，评估 **Accuracy@1**。
- **Domain-Specific Benchmarks**：
  - **Formula**：数值推理任务，基于结构化文件。
  - **FiNER**：金融实体识别（Financial Entity Recognition），细粒度 XBRL 文档标注。

### 实验设置和评估指标
- **基础模型**：主要使用 **DeepSeek-V3.1**（128K 上下文窗口）。
- **评估指标**：
  - AppWorld：TGC、SGC、Avg。
  - Terminal-Bench：Accuracy@1。
  - Formula/FiNER：Accuracy。
- **训练成本**：以美元（$）为单位报告训练成本。
- **动态批大小控制器**：基于幂律延迟曲线选择边际延迟下降低于阈值的最大 batch size。

### 基线方法对比
- **Sequential Baseline**：batch size = 1。
- **Naive Parallel Scaling**：直接增加 batch size（如 5, 10, 20, 40, 100）。
- **Prompt-Level Mitigations**：
  - **Top-K Retrieval**：聚类后每组取一条反思。
  - **Summarization**：先总结所有反思再输入。
- **Combee**：本文方法，基于 ACE 或 GEPA 构建。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 方法 | Batch Size | AppWorld (Avg) | Terminal-Bench (Acc@1) | Formula (Acc) | FiNER (Acc) |
|------|------------|----------------|------------------------|---------------|-------------|
| ReAct (no context) | – | 53.3 | 32.2% | – | – |
| ReAct + ACE (seq) | 1 | 58.1 | 37.9% | 87.0% | 76.0% |
| ReAct + ACE (naive, bs=40) | 40 | 55.7 | – | – | – |
| **ReAct + Combee (bs=40)** | 40 | **65.8** | – | – | – |
| Terminus-2 + ACE (seq) | 1 | – | 37.9% | – | – |
| Terminus-2 + ACE (naive, bs=30) | 30 | – | 31.0% | – | – |
| **Terminus-2 + Combee (bs=30)** | 30 | – | **35.6%** | – | – |
| GEPA (seq) | 1 | – | – | ~87.0% | ~76.0% |
| **Combee + GEPA** | dynamic | – | – | **≈87.0%** | **≈76.0%** |

### 与基线方法的对比结果
- **速度提升**：Combee 在 AppWorld 上实现 **12× 加速**，在 Terminal-Bench 上实现 **17× 加速**，相比顺序学习。
- **质量保持**：在 batch size=40 时，Combee 的 playbook 大小为 **6,887 tokens**，远高于 naive 方法的 **526 tokens**，表明其保留了更多信息。
- **优于其他缓解策略**：**Top-K Retrieval** 和 **Summarization** 效果远差于 Combee，无法解决 context overload。

### 消融实验结果
- **Dynamic Batch Size Controller**（图6）：
  - 固定 batch size 会导致延迟增加或质量下降。
  - 动态控制器能自动找到最优 batch size，兼顾速度与质量。
- **Augmented Shuffling**（图7）：
  - 移除该机制后，学习鲁棒性显著下降，尤其在不同 subgroup size 下波动剧烈。
  - 验证了其对防止信息丢失的重要性。
- **Subgroup Size**：
  - 当 subgroup size ≈ √bs 时效果最佳，验证了 parallel scan 设计的有效性。
- **跨模型迁移**（图8）：
  - Combee 在 **GPT-OSS 120B** 上同样有效，证明其设计具有模型无关性。

---

## 4. 关键结论和发现

### 论文的主要发现
- **Context Overload 是真实且严重的问题**：即使在 128K 上下文窗口内，naive 增加并行度也会导致 aggregator LLM 丢失高价值信息。
- **Combee 有效打破“速度-质量”权衡**：通过 **parallel scan aggregation** 和 **augmented shuffling**，实现了高并行度下的高质量学习。
- **动态批大小控制器提升效率**：自动化调参，适应不同训练阶段的需求。
- **方法具有通用性**：兼容 ACE、GEPA 等主流 prompt learning 框架，并可在不同任务和模型上迁移。

### 方法的局限性
- 目前仅验证了在 **ACE** 和 **GEPA** 上的应用，尚未测试更复杂的 context artifact 结构（如程序库、检索增强技能库）。
- **动态批大小控制器** 依赖幂律延迟模型和固定阈值，可能不适用于延迟特征差异较大的任务。
- 假设同步并行执行，未探索异步或部分同步变体。

### 未来工作方向
- 集成更多类型的 prompt learning 方法，验证其通用性。
- 改进动态控制器以适应更广泛的延迟分布。
- 探索 **异步 Combee**，借鉴异步 SGD 思想，进一步提升异构环境下的吞吐量。
- 将 Combee 与并行任务求解系统结合，构建端到端高效的 agentic pipeline。

---

> **总结**：Combee 是首个系统性解决 prompt learning 可扩展性问题的框架，通过 **Map-Shuffle-Reduce** 范式和动态控制机制，实现了**高速、高质量、低成本**的并行上下文学习，为大规模语言模型智能体的自我进化提供了坚实基础。

</details>

---

### 15. [MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents](https://arxiv.org/abs/2604.04853)

**Authors**: Shu Wang, Edwin Yu, Oscar Love, Tom Zhang, Tom Wong, Steve Scargall, Charles Fan  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 6.0  
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
当前基于 **Large Language Model (LLM)** 的 AI Agent 在实现个性化、长期记忆和跨会话任务执行时面临以下挑战：
- **静态参数限制**：LLM 的权重在训练后固定，无法从交互中动态学习新知识。
- **上下文窗口有限**：受限于 context window，长历史对话难以完整保留。
- **传统 RAG 的局限性**：Retrieval-Augmented Generation (RAG) 主要针对静态文档检索，不适用于动态、双向演化的用户交互场景。
- **现有记忆系统成本高且易出错**：如 Mem0、Zep 等依赖频繁 LLM 调用进行提取、聚合和更新，导致 token 开销大、事实漂移（factual drift）和误差累积。

### 提出的新方法与创新思路
MemMachine 是一个开源的、**ground-truth-preserving** 架构的记忆系统，其核心设计思想是“**存储原始数据，延迟智能处理**”，主要创新点如下：

#### ✅ 1. **Ground-truth-preserving 架构**
- 存储原始对话片段（raw conversational episodes），以句子为单位索引（sentence-level indexing），避免早期 LLM 提取带来的信息丢失和错误传播。
- 最小化对 LLM 的依赖，仅在必要时（如摘要生成、profile 抽取）调用 LLM。

#### ✅ 2. **Contextualized Retrieval（上下文化检索）**
- 引入“核 episode + 邻近上下文”的检索机制：找到语义匹配的 nucleus episode 后，自动扩展前后若干轮对话形成 episode cluster。
- 解决了对话中语义分散、嵌入相似度低的问题，显著提升多跳推理和上下文依赖查询的召回率。

#### ✅ 3. **双层记忆架构**
- **Episodic Memory**：短期（STM）与长期（LTM）结合，STM 维持最近上下文并生成压缩摘要；LTM 持久化所有历史。
- **Profile Memory（语义记忆）**：从对话中提取用户偏好、行为模式等结构化信息，支持个性化响应。

#### ✅ 4. **Retrieval Agent（检索代理）**
- 一种可选启用的 LLM 编排检索管道，能根据查询类型路由到不同策略：
  - `ChainOfQuery`：用于多跳依赖链（iterative evidence accumulation）
  - `SplitQuery`：用于并行多实体查找（fan-out）
  - `Direct Search`：单跳直接检索
- 支持 **multi-query reranking**，确保中间证据也能被有效排序。

#### ✅ 5. 成本效率优势
- 相比 Mem0，输入 token 减少约 **80%**，大幅降低运行成本。
- 通过减少不必要的 LLM 提取操作，提升了系统的准确性和稳定性。

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据集 | 描述 |
|-------|------|
| **LoCoMo** | 多会话对话记忆基准，包含 single-hop、multi-hop、temporal 和 open-domain 四类问题（共 1,540 题）。 |
| **LongMemEvals (ICLR 2025)** | 评估五项核心能力：信息提取、跨会话推理、时间推理、知识更新、拒绝回答。每个样本嵌入约 115K tokens 的聊天历史。 |
| **HotpotQA hard** | 多跳问答数据集，强调跨文档推理能力。 |
| **WikiMultiHop** | 包含噪声注入的多跳推理测试集，模拟真实环境中记忆混杂的情况。 |
| **EpBench** | 基于合成叙事语料的 episodic memory 评测，规模达 100K–1M tokens。 |

### 实验设置与评估指标
- **环境配置**：
  - CPU: 8vCPU, RAM: 16GiB, OS: Ubuntu 24.04
  - 数据库存储：PostgreSQL (pgvector) + Neo4j (图数据库)
  - Embedding Model: OpenAI `text-embedding-3-small`
  - Reranker: AWS Cohere `rerank-v3`
- **评估指标**：
  - **LLM Judge Score (llm_score)**：由 judge-LLM 判断答案是否与 ground truth 语义一致（0/1），为主要指标。
  - BLEU、F1：辅助指标。
  - **Recall**：衡量是否成功检索到支持事实。
  - **Token Cost**：记录每问输入/输出 token 数量，评估效率。

### 基线方法对比
- **Mem0**：主流生产级记忆系统，依赖 LLM 提取事实。
- **Zep**：基于时间知识图谱的记忆架构。
- **Memobase / LangMem**：其他开源记忆框架。
- **OpenAI 原生记忆**：ChatGPT 内置记忆功能作为 baseline。
- 所有比较均使用公开报告结果或重新运行以保证公平性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 全文）

| Benchmark | Metric | Result |
|---------|--------|--------|
| **LoCoMo** | Overall Score (gpt-4.1-mini) | **0.9169** |
| **LongMemEvals** | Ablation Best Accuracy | **93.0%** |
| **HotpotQA hard** | Retrieval Agent Accuracy | **93.2%** |
| **WikiMultiHop** | Retrieval Agent Accuracy (noise injected) | **92.6%** |
| **vs. Mem0** | Input Token Reduction | **~80% less** |

### 与基线方法的对比结果

#### 🔹 LoCoMo 对比（Table 11）
| System | Overall Score |
|--------|---------------|
| **MemMachine (gpt-4.1-mini)** | **0.9169** |
| Memobase | 0.7578 |
| Zep | 0.7514 |
| Mem0 | 0.6688 |
| LangMem | 0.5810 |
| OpenAI | 0.5290 |

👉 MemMachine 比第二名高出 **+9.7 pts**，尤其在 single-hop 和 multi-hop 上表现优异。

#### 🔹 LongMemEvals 消融实验（Table 13）
| 优化维度 | 贡献（ΔScore） |
|----------|----------------|
| **Retrieval depth tuning (k=20→30)** | **+4.2%** ⬆️ |
| **Context formatting** | **+2.0%** |
| **Search prompt design** | **+1.8%** |
| **Query bias correction ("user:" prefix)** | **+1.4%** |
| **Sentence chunking** | +0.8% |
| **GPT-5 → GPT-5-mini (answer model)** | **+2.6%** |

📌 发现：**检索阶段优化远超摄入阶段优化**，说明“如何检索”比“如何存储”更重要。

#### 🔹 模型选择反直觉发现
- **GPT-5-mini > GPT-5**：当配合优化后的 prompt（Edwin3）时，小型模型反而更优（+2.6%），且成本更低。
- 原因推测：GPT-5 内建的 chain-of-thought 可能干扰简洁指令；而 GPT-5-mini 更擅长遵循直接指令。

#### 🔹 Retrieval Agent 性能拆解（Table 4）
| Strategy | Accuracy | Recall |
|--------|----------|--------|
| **ChainOfQuery** | 92.27% | **95.31%** |
| **SplitQuery** | 94.07% | 92.83% |
| **MemMachine (direct)** | 93.53% | 89.31% |
| **Overall** | **93.20%** | **92.31%** |

✅ 显示 agent mode 在复杂查询上带来显著增益。

---

## 4. 关键结论和发现

### 主要发现
1. **Ground-truth preservation + adaptive retrieval = robust memory behavior**
   - 保持原始 episode 完整性，并在其上叠加智能检索策略，比早期压缩更能保障准确性。
   
2. **Retrieval-stage optimization dominates accuracy**
   - 检索深度、上下文格式、搜索提示词设计等影响远大于 ingestion 阶段的 sentence chunking。
   - 意味着应优先投资于“如何读取记忆”，而非“如何写入”。

3. **模型与 prompt 必须 co-optimize**
   - 不同 generation 的 LLM 对 prompt 敏感度不同，不能复用旧 prompt。
   - 小模型（GPT-5-mini）配合适当 prompt 可超越大模型，实现更高性价比。

4. **Retrieval Agent 可组合、可扩展**
   - 工具树结构允许灵活添加新策略（如 temporal reasoning agent）。
   - multi-query reranking 机制解决了多步推理中的证据留存问题。

5. **成本效益显著**
   - 输入 token 减少 ~80%，适合大规模部署。
   - 支持本地部署（local LLMs）、混合架构，满足隐私需求。

### 局限性（Limitations）
- **Temporal reasoning 仍有提升空间**：尽管已有改进，但在纯时间推理任务上略逊于专精系统。
- **未实现 procedural memory**：目前不支持动作策略、工具使用模式的学习与复用。
- **Benchmark sensitivity**：结果受 eval-LLM、prompt template 和 provider 更新影响。
- **缺乏多模态支持**：当前仅处理文本型对话记忆。
- **ablation 未考虑交互效应**：各优化维度独立测试，未探索组合影响（如 chunking × k-depth）。

### 未来工作方向（Future Work）
- ✅ 实现 **procedural memory**：存储和复用 workflow、tool-use patterns。
- ✅ 增强 **temporal reasoning**：开发专用时间索引与查询扩展技术。
- ✅ 支持 **adaptive retrieval depth**：根据 query complexity 动态调整 `k`。
- ✅ 探索 **memory consolidation & forgetting**：模仿人类记忆机制，自动归档冷数据。
- ✅ 扩展至 **multi-modal memory**：支持图像、音频等非文本内容。
- ✅ 支持更多向量数据库后端：如 ChromaDB、Milvus。
- ✅ 引入 **reinforcement learning** 优化 retrieval policy。
- ✅ 开发 **function-calling code mode**：让 agent 输出可执行代码而非调用预定义工具列表，进一步节省 token。

---

> 💡 **总结一句话**：  
> **MemMachine 证明了“保真存储 + 智能检索”的架构优于“早期压缩 + 重度 LLM 提取”的范式，在准确性、效率和可维护性之间取得了卓越平衡，为下一代个性化 AI Agent 提供了坚实的记忆基础设施。**

</details>

---

### 16. [TriAttention: Efficient Long Reasoning with Trigonometric KV Compression](https://arxiv.org/abs/2604.04921)

**Authors**: Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.04921v1  

#### Abstract
Extended reasoning in large language models (LLMs) creates severe KV cache memory bottlenecks. Leading KV cache compression methods estimate KV importance using attention scores from recent post-RoPE queries. However, queries rotate with position during RoPE, making representative queries very few, ...

---

### 17. [Beyond Semantic Manipulation: Token-Space Attacks on Reward Models](https://arxiv.org/abs/2604.02686)

**Authors**: Yuheng Zhang, Mingyue Huo, Minghao Zhu, Mengxue Zhang, Nan Jiang  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.02686v1  

#### Abstract
Reward models (RMs) are widely used as optimization targets in reinforcement learning from human feedback (RLHF), yet they remain vulnerable to reward hacking. Existing attacks mainly operate within the semantic space, constructing human-readable adversarial outputs that exploit RM biases. In this w...

---

### 18. [DSBD: Dual-Aligned Structural Basis Distillation for Graph Domain Adaptation](https://arxiv.org/abs/2604.03154)

**Authors**: Yingxu Wang, Kunyu Zhang, Jiaxin Huang, Mengzhu Wang, Mingyan Xiao, Siyang Gao, Nan Yin  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.03154v1  

#### Abstract
Graph domain adaptation (GDA) aims to transfer knowledge from a labeled source graph to an unlabeled target graph under distribution shifts. However, existing methods are largely feature-centric and overlook structural discrepancies, which become particularly detrimental under significant topology s...

---

### 19. [PRISM: LLM-Guided Semantic Clustering for High-Precision Topics](https://arxiv.org/abs/2604.03180)

**Authors**: Connor Douglas, Utkucan Balci, Joseph Aylett-Bullock  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.03180v1  

#### Abstract
In this paper, we propose Precision-Informed Semantic Modeling (PRISM), a structured topic modeling framework combining the benefits of rich representations captured by LLMs with the low cost and interpretability of latent semantic clustering methods. PRISM fine-tunes a sentence encoding model using...

---

### 20. [TABQAWORLD: Optimizing Multimodal Reasoning for Multi-Turn Table Question Answering](https://arxiv.org/abs/2604.03393)

**Authors**: Tung Sum Thomas Kwok, Xinyu Wang, Xiaofeng Lin, Peng Lu, Chunhe Wang, Changlun Li, Hanwei Wu, Nan Tang, Elisa Kreiss, Guang Cheng  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.03393v1  

#### Abstract
Multimodal reasoning has emerged as a powerful framework for enhancing reasoning capabilities of reasoning models. While multi-turn table reasoning methods have improved reasoning accuracy through tool use and reward modeling, they rely on fixed text serialization for table state readouts. This intr...

---

### 21. [RL-Driven Sustainable Land-Use Allocation for the Lake Malawi Basin](https://arxiv.org/abs/2604.03768)

**Authors**: Ying Yao  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.03768v1  

#### Abstract
Unsustainable land-use practices in ecologically sensitive regions threaten biodiversity, water resources, and the livelihoods of millions. This paper presents a deep reinforcement learning (RL) framework for optimizing land-use allocation in the Lake Malawi Basin to maximize total ecosystem service...

---

### 22. [PanLUNA: An Efficient and Robust Query-Unified Multimodal Model for Edge Biosignal Intelligence](https://arxiv.org/abs/2604.04297)

**Authors**: Marija Zelic, Anna Tegon, Yawei Li, Thorir Mar Ingolfsson, Luca Benini  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.04297v1  

#### Abstract
Physiological foundation models (FMs) have shown promise for biosignal representation learning, yet most remain confined to a single modality such as EEG, ECG, or PPG, largely because paired multimodal datasets are scarce. In this paper, we present PanLUNA, a compact 5.4M-parameter pan-modal FM that...

---

### 23. [What Makes a Sale? Rethinking End-to-End Seller--Buyer Retail Dynamics with LLM Agents](https://arxiv.org/abs/2604.04468)

**Authors**: Jeonghwan Choi, Jibin Hwang, Gyeonghun Sun, Minjeong Ban, Taewon Yun, Hyeonjae Cheon, Hwanjun Song  
**Category**: cs.AI  
**Published**: 2026-04-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.04468v1  

#### Abstract
Evaluating retail strategies before deployment is difficult, as outcomes are determined across multiple stages, from seller-side persuasion through buyer-seller interaction to purchase decisions. However, existing retail simulators capture only partial aspects of this process and do not model cross-...

---

### 24. [RUQuant: Towards Refining Uniform Quantization for Large Language Models](https://arxiv.org/abs/2604.04013)

**Authors**: Han Liu, Haotian Gao, Changya Li, Feng Zhang, Xiaotong Zhang, Wei Wang, Hong Yu  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.04013v1  

#### Abstract
The increasing size and complexity of large language models (LLMs) have raised significant challenges in deployment efficiency, particularly under resource constraints. Post-training quantization (PTQ) has emerged as a practical solution by compressing models without requiring retraining. While exis...

---

### 25. [Synthetic Sandbox for Training Machine Learning Engineering Agents](https://arxiv.org/abs/2604.04872)

**Authors**: Yuhang Zhou, Lizhu Zhang, Yifan Wu, Jiayi Liu, Xiangjun Fan, Zhuokai Zhao, Hong Yan  
**Category**: cs.CL  
**Published**: 2026-04-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.04872v1  

#### Abstract
As large language model agents advance beyond software engineering (SWE) tasks toward machine learning engineering (MLE), verifying agent behavior becomes orders of magnitude more expensive: while SWE tasks can be verified via fast-executing unit tests, MLE verification requires running full ML pipe...

---

### 26. [GENSERVE: Efficient Co-Serving of Heterogeneous Diffusion Model Workloads](https://arxiv.org/abs/2604.04335)

**Authors**: Fanjiang Ye, Zhangke Li, Xinrui Zhong, Ethan Ma, Russell Chen, Kaijian Wang, Jingwei Zuo, Desen Sun, Ye Cao, Triston Cao, Myungjin Lee, Arvind Krishnamurthy, Yuke Wang  
**Category**: cs.DC  
**Published**: 2026-04-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.04335v1  

#### Abstract
Diffusion models have emerged as the prevailing approach for text-to-image (T2I) and text-to-video (T2V) generation, yet production platforms must increasingly serve both modalities on shared GPU clusters while meeting stringent latency SLOs. Co-serving such heterogeneous workloads is challenging: T...

---

### 27. [LiME: Lightweight Mixture of Experts for Efficient Multimodal Multi-task Learning](https://arxiv.org/abs/2604.02338)

**Authors**: Md Kowsher, Haris Mansoor, Nusrat Jahan Prottasha, Ozlem Garibay, Victor Zhu, Zhengping Ji, Chen Chen  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.02338v1  

#### Abstract
MoE-PEFT methods combine Mixture of Experts with parameter-efficient fine-tuning for multi-task adaptation, but require separate adapters per expert causing trainable parameters to scale linearly with expert count and limiting applicability to adapter-based architectures. We propose LiME (Lightweigh...

---

### 28. [Homophily-aware Supervised Contrastive Counterfactual Augmented Fair Graph Neural Network](https://arxiv.org/abs/2604.02342)

**Authors**: Mahdi Tavassoli Kejani, Fadi Dornaika, Charlotte Laclau, Jean-Michel Loubes  
**Category**: cs.LG  
**Published**: 2026-04-07  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.02342v1  

#### Abstract
In recent years, Graph Neural Networks (GNNs) have achieved remarkable success in tasks such as node classification, link prediction, and graph representation learning. However, they remain susceptible to biases that can arise not only from node attributes but also from the graph structure itself. A...

---

### 29. [Adaptive Semantic Communication for Wireless Image Transmission Leveraging Mixture-of-Experts Mechanism](https://arxiv.org/abs/2604.02691)

**Authors**: Haowen Wan, Qianqian Yang  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.02691v1  

#### Abstract
Deep learning based semantic communication has achieved significant progress in wireless image transmission, but most existing schemes rely on fixed models and thus lack robustness to diverse image contents and dynamic channel conditions. To improve adaptability, recent studies have developed adapti...

---

### 30. [Structure-Aware Commitment Reduction for Network-Constrained Unit Commitment with Solver-Preserving Guarantees](https://arxiv.org/abs/2604.02788)

**Authors**: Guangwen Wang, Jiaqi Wu, Yang Weng, Baosen Zhang  
**Category**: cs.LG  
**Published**: 2026-04-06  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.02788v1  

#### Abstract
The growing number of individual generating units, hybrid resources, and security constraints has significantly increased the computational burden of network-constrained unit commitment (UC), where most solution time is spent exploring branch-and-bound trees over unit-hour binary variables. To reduc...

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
