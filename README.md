# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-02 06:58:28 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [From Skew to Symmetry: Node-Interconnect Multi-Path Balancing with Execution-time Planning for Modern GPU Clusters](https://arxiv.org/abs/2604.00317)

**Authors**: Jinghan Yao, Kaushik Kandadi, Bharath Ramesh, Hari Subramoni, Dhabaleswar K. Panda  
**Category**: cs.DC  
**Published**: 2026-04-02  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2604.00317v1  

#### Abstract
Modern GPU-based high-performance computing clusters offer unprecedented communication bandwidth through heterogeneous intra-node interconnects and inter-node networks. However, despite this high aggregate bandwidth, many real-world communication patterns fail to fully utilize the available hardware...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Skew to Symmetry: Node-Interconnect Multi-Path Balancing with Execution-time Planning for Modern GPU Clusters

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代 GPU 集群虽然具备极高的聚合通信带宽（如 NVLink、NVSwitch、多轨 InfiniBand/RoCE），但在实际应用中，许多重要通信模式（如 Mixture-of-Experts 中的 All-to-Allv）表现出严重的**流量倾斜（traffic skew）**。这种倾斜导致部分链路过载而其他链路闲置，引发拥塞、尾延迟激增和可扩展性瓶颈。

现有通信库（如 NCCL、MPI/UCX）依赖静态路径选择策略（如最短路径路由或多轨哈希分片），无法在运行时动态响应负载变化，因此难以充分利用硬件提供的丰富连接能力。

---

### 提出了什么新方法或新思路
作者提出 **NIMBLE**（Node-Interconnect Multi-path BaLancing with Execution-time orchestration），一种**端点驱动的运行时通信编排系统**，其核心思想是：

- **动态重平衡**：在运行时持续监控链路利用率，并基于实时负载将通信流量智能地重新分布到所有可用的 intra-node（如 NVLink）和 inter-node（如多轨 InfiniBand）路径上。
- **容量归一化的最小拥塞优化**：采用基于乘法权重更新（multiplicative-weights scheme）的快速近似算法，求解一个以“最小化最大链路拥塞”为目标的多商品流（MCF）问题。
- **GPU 内核级 RDMA 流水线**：利用 CUDA-aware GPU kernel 实现高效的数据转发，支持通过中间 GPU 和 rail-matched NIC 进行多跳传输，避免主机干预，实现非阻塞流水线。

---

### 相比现有方法的优势
| 维度 | 现有方法（NCCL/MPI/UCX） | NIMBLE |
|------|--------------------------|--------|
| 路由策略 | 静态拓扑感知（初始化时确定） | 动态、运行时感知链路负载 |
| 多路径利用 | 多为传输层条带化（transport-level striping） | 应用层细粒度拆分与调度 |
| 拥塞适应性 | 无 | 显式建模瓶颈链路并规避 |
| 透明性 | 高（无需修改应用） | 同样高，兼容现有 API（如 NCCL Send/Recv, AlltoAll） |
| 性能提升 | 在均衡流量下表现良好 | 在**倾斜流量下显著优于基线**，同时不牺牲均衡场景性能 |

---

## 2. 核心实验方法和设置

### 实验平台（Hardware）
- **节点配置**：每节点配备 4 个 NVIDIA H100 SXM5 GPU，通过全连接 NVLink 互联。
- **内存**：每个 GPU 拥有 94 GB HBM2e。
- **网络**：每节点配 4 个 NDR400 InfiniBand HCA（400 Gb/s），支持 GPUDirect RDMA，且每个 NIC 与一个 GPU 硬件绑定（rail-matched）。
- **CPU**：双 Intel Xeon Platinum 8470。

### 软件环境（Software）
- 对比对象：
  - **NCCL v2.26**
  - **OpenMPI v5.0.7 + UCX v1.18.0**（启用 CUDA-aware）
- 所有测试均确保 GPU 与 NIC 的硬件亲和性一致。

### 应用与工作负载（Workloads）
- **点对点通信（Send/Recv）**：用于验证 intra-node 和 inter-node 多路径加速效果。
- **Skewed All-to-Allv**：模拟 MoE 架构中的 token dispatch 与 combine 阶段，人为引入“热点”接收者（hotspot ratio 控制倾斜程度）。
- **端到端 MoE 块**：包含 dispatch → compute → combine 完整流程，评估真实 LLM 推理场景下的性能增益。
- **1D Stencil**：用于测量编排算法本身的开销。

### 评估指标
- **吞吐量（Throughput / Bandwidth）**
- **端到端通信延迟（Latency）**
- **尾延迟（p99 Latency）**
- **算法运行时间（Orchestration Overhead）**
- **Speedup**（相对于 NCCL 或 OpenMPI）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ Intra-node 多路径聚合带宽（图6a）
- **直接 NVLink**：峰值约 **120 GB/s**
- **单个中间 GPU 转发（2-hop）**：达 **213.1 GB/s**（提升 ~1.78×）
- **两个中间 GPU 转发（3-path）**：达 **278.2 GB/s**（提升 ~2.3×）

> 注：多路径收益随消息大小增加而显现，**>64 MB 时趋于饱和**；小消息（≤1 MB）禁用多路径以防开销过大。

#### ✅ Inter-node 多轨聚合带宽（图6b）
- 单 NIC：饱和于 ~45.1 GB/s
- 四 NIC 并行（rail-matched）：达到 **170.0 GB/s**（理论 4× 加速，实测 ~3.8×）

> 表明 NIMBLE 可有效聚合多轨带宽，且转发开销极低。

#### ✅ Skewed All-to-Allv 加速（图7）
- 在 hotspot ratio ≥ 0.7 时，NIMBLE 相比 NCCL 最高提速 **5.2×**
- 在轻度倾斜或小消息场景下，性能与 NCCL 相当，甚至略低于 OpenMPI（因后者使用 DMA 引擎更易饱和小包）

#### ✅ 端到端 MoE 性能（图8）
- 在全局 token 数为 16K、hotspot ratio=0.9 场景下，**端到端速度提升最高达 1.35×**
- 提升主要来自 dispatch 与 combine 阶段的通信压缩，compute 时间不变
- 建议启用条件：`global_tokens > 16K && hotspot_ratio ≥ 0.7`

#### ✅ 编排算法开销（表 I）
- 算法执行时间仅 **0.03–0.05 ms**，远小于通信耗时（如 256MB 通信需 2–6 ms），可忽略不计

---

### 与基线方法对比结果
| 场景 | NIMBLE vs NCCL | NIMBLE vs OpenMPI |
|------|----------------|-------------------|
| 均衡流量 | ≈持平 | ≈持平 |
| 小消息倾斜 | ≈持平或略差 | 可能稍差（kernel-driven vs DMA-driven） |
| 大消息严重倾斜 | **最高 5.2× 更快** | **最高 3.4× 更快**（点对点） |
| MoE 端到端 | **最高 1.35× 加速** | —— |

---

### 消融实验与设计分析
- **多跳深度限制**：实验表明，超过 1 个 intra-node hop 后收益递减，甚至负向（同步开销、L2/HBM 争用）。故默认限制为最多 1 个中间 GPU。
- **size-aware penalty**：对小消息施加路由惩罚，防止不必要的多路径拆分。
- **hysteresis-based load tracking**：避免路径震荡（oscillation）。
- **per-destination reassembly queue**：保证消息顺序性和确定性。

---

## 4. 关键结论和发现

### 主要发现
1. **通信不平衡是现代 HPC/AI 工作负载的关键瓶颈**，尤其在 MoE、推荐系统、稀疏计算等场景中普遍存在。
2. **静态通信库无法应对动态倾斜**，即使拥有高带宽硬件也无法发挥潜力。
3. **NIMBLE 通过运行时链路感知的多路径编排，显著缓解拥塞热点**，在倾斜负载下实现数倍加速。
4. **GPU kernel-based RDMA pipelining 是实现低开销多跳转发的关键技术**，支持透明、高效的中间节点转发。
5. **该方案完全兼容现有编程模型**（如 PyTorch、NCCL API），无需修改应用程序逻辑。

---

### 方法的局限性
1. **在 NVSwitch 架构中受限**：如 NVIDIA DGX 系统中，GPU 不直接相连，而是通过中央 NVSwitch 互联，导致 intra-node 多路径转发不可行（缺少冗余直连链路）。
2. **仅适用于单作业内部优化**：NIMBLE 不解决跨作业（multi-tenant）间的资源竞争，需依赖底层网络的拥塞控制机制（如 DCQCN/HPCC）保障公平性。
3. **小消息增益有限**：由于多路径拆分和同步开销，对 ≤1 MB 的消息关闭多路径以保持效率。

---

### 未来工作方向
1. **进一步降低编排开销**：探索基于 InfiniBand GPUDirect Async（IBGDA）的实现，减少 CPU 参与，提升异步调度能力。
2. **集成至更广泛的通信库生态**：除 NCCL 外，支持 RCCL、OneCCL 等。
3. **结合专家放置（expert placement）进行联合优化**：与 Lazarus、Pro-Prophet 等系统协同，在调度层与通信层共同实现负载均衡。
4. **扩展至更复杂的拓扑结构**：支持 hierarchical multi-rail、disaggregated 架构下的路径编排。

---

> **总体评价**：  
> NIMBLE 提出了一种**实用、高效、透明**的运行时通信优化框架，成功将原本“被浪费”的链路带宽转化为实际性能提升，代表了下一代高性能 GPU 集群通信系统的重要演进方向。

</details>

---

### 2. [Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention](https://arxiv.org/abs/2604.00754)

**Authors**: Zehao Jin, Yanan Sui  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.00754v1  

#### Abstract
The whole-brain connectome of a fruit fly comprises over 130K neurons connected with a probability of merely 0.02%, yet achieves an average shortest path of only 4.4 hops. Despite being highly structured at the circuit level, the network's long-range connections are broadly distributed across brain ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **Sliding Window Attention (SWA)** 虽然在计算复杂度上为 $O(nw)$，具有良好的效率，但由于其**局部性限制**，每层只能看到固定窗口内的上下文，导致多层堆叠后感受野增长缓慢（线性增长 $lw$），难以覆盖长序列中的全局依赖关系。这在语言建模等需要长程推理的任务中成为瓶颈。

### 提出的新方法与新思路
受果蝇全脑连接组（**fruit fly connectome**）启发，作者提出 **Stochastic Attention (SA)**，一种无需额外可学习参数的注意力增强机制：

- 在每一层 **应用随机排列（random permutation）** 将输入 token 序列打乱；
- 在打乱后的空间中执行标准的 SWA；
- 注意力输出后再通过逆排列恢复原始顺序。

该设计将原本固定的局部窗口转化为**跨序列的随机局部邻域**，从而在不增加计算成本的前提下引入“**随机长程捷径（stochastic shortcuts）**”，实现高效的全局信息流动。

进一步地，提出了 **gated SA + SWA 组合架构**：
- 并行运行 SA 和 SWA 分支；
- 使用两个独立的 **sigmoid gating 机制** 动态融合两路输出；
- 模拟 connectome 中“局部聚类 + 随机远距离连接”的小世界网络特性（small-world property）。

### 相比现有方法的优势
| 特性 | SA | SWA | Full Attention | MoBA |
|------|----|-------|----------------|--------|
| 时间复杂度 | $O(nw)$ | $O(nw)$ | $O(n^2)$ | $O(nw)$ |
| 可学习参数增量 | 无 | 无 | — | 有 |
| 感受野增长速度 | 指数级 $O(\log_w n)$ | 线性 $O(lw)$ | 全局（1层） | 依赖块选择策略 |
| 实现难度 | 极低（仅需 index shuffle） | 标准模块 | 高 | 中等 |

> ✅ **优势总结**：
> - **参数免费**：不修改 attention 内部结构，仅添加 $O(n)$ 的索引重排操作；
> - **即插即用**：可作为任何基于 SWA 的模型的 drop-in 替换；
> - **理论保障**：证明了 $O(\log_w n)$ 层即可实现全序列覆盖，显著优于 SWA 的 $O(n/w)$；
> - **互补性强**：SA 提供全局探索能力，SWA 提供局部连贯性，二者结合达到最优平衡。

---

## 2. 核心实验方法和设置

### 数据集
1. **预训练任务**：
   - 使用 **SlimPajama** 子集（6B tokens，共约15B训练token）
   - 模型规模：~360M 参数 decoder-only Transformer

2. **训练后零样本评估（zero-shot evaluation）**：
   - WikiText（困惑度）
   - LAMBADA（完形填空准确率与困惑度）
   - PIQA、HellaSwag、WinoGrande、ARC-Easy（常识推理任务，报告准确率）

3. **训练自由推理测试（training-free inference）**：
   - 在已发布的 **Qwen3-8B** 和 **Qwen3-30B-A3B** 上进行 attention 替换实验
   - 评估基准（7项）：
     - MMLU（多任务理解）
     - BoolQ（逻辑判断）
     - HellaSwag（情境预测）
     - LAMBADA（长程依赖预测）
     - ARC-Easy / ARC-Challenge
     - HumanEval（代码生成）

### 实验设置
- **序列长度**：2048（训练）、最长至32K（效率分析）
- **窗口大小 $w$**：默认 256；消融实验中测试 $w \in \{16,32,\dots,512\}$
- **硬件平台**：单张 A100 80GB GPU
- **实现方式**：基于 **FlexAttention** 实现动态 mask，支持 permutation 的高效 gather/scatter 操作
- **位置编码处理**：RoPE 使用原始位置索引，确保与 pre-trained 模型兼容

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Full Attention** | 完整因果注意力（上限参考） |
| **SWA** | 标准滑动窗口注意力 |
| **SA** | 本文提出的随机排列 + SWA |
| **SA + SWA (gated)** | 双路径门控融合结构 |
| **MoBA (Mixture of Block Attention)** | 当前先进稀疏注意力方法，用于公平比较 |

---

## 3. 主要实验结果和性能指标

### （1）预训练语言模型结果（Table 1）

| Model | Wiki.ppl ↓ | LAMBADA ppl ↓ | LAMBADA acc ↑ | Avg Acc ↑ |
|-------|------------|---------------|----------------|-----------|
| Full Attention | 51.34 | 185.3 | 469.2 | 34.9 |
| SWA ($w=256$) | 57.05 | 156.1 | 370.6 | 35.1 |
| SA ($w=256$) | 75.83 | 260.1 | 785.9 | 34.3 |
| **SA+SWA (ours)** | **51.98** | **131.7** | **371.6** | **35.9** |

> 🔍 **关键观察**：
> - 单独使用 SA 导致 **ppl 显著上升**（75.83），说明破坏了局部连续性，不适合单独使用；
> - **SA+SWA 融合模型** 同时获得：
>   - 接近 Full Attention 的 **语言建模质量**（Wiki.ppl ≈ 52）
>   - 最佳的 **下游任务平均准确率（35.9）**
>   - **LAMBADA 困惑度最低（131.7）**，表明其擅长捕捉长距离语义依赖

### （2）训练自由推理结果（Qwen3 系列）

#### 总体趋势（Figure 4）
- 在相同 $O(nw)$ 计算预算下，**Stochastic Attention 恢复 full attention 性能的速度最快**；
- 在 **Qwen3-8B** 上，当 $w_{\text{eff}}=128$ 时，SA 达到 **70.9% avg acc**（接近 baseline 71.5%），而 SWA 仅为 62.2%；
- 在 **Qwen3-30B-A3B** 上，$w=64$ 时 SA 达到 **73.2%**，远超 SWA（47.0%）和 MoBA（66.3%）

#### 关键任务表现（Tables 4–5）
| 方法 | MMLU (8B) | BoolQ (8B) | LAMBADA (8B) | Avg (8B) |
|------|----------|-----------|--------------|---------|
| SWA ($w=64$) | 46.0 | 63.8 | 50.2 | 50.7 |
| MoBA ($k=2$) | 56.4 | 75.7 | 44.7 | 62.9 |
| **Stochastic (ours)** | **63.0** | **77.1** | **62.9** | **66.4** |

> ✅ **优势领域**：
> - **MMLU、BoolQ、LAMBADA** 等需跨段落整合信息的任务提升最明显；
> - 即使在极小窗口（如 $w=32$）下，SA 仍保持较高性能（MMLU: 44.4 vs SWA: 29.0），体现其强大的全局混合能力。

### （3）效率分析（Table 2）
| Sequence Length | SA Latency (ms) | Full Attn Latency (ms) | Speedup |
|------------------|------------------|--------------------------|---------|
| 2,048           | 5.4              | 8.0                      | 1.5×    |
| 8,192           | 15.2             | 99.7                     | 6.6×    |
| 32,768          | 52.8             | 1,477                    | **28.0×** |

> ⚡️ 随着序列增长，SA 相对于 Full Attention 的加速比呈指数上升，符合 $O(nw)$ vs $O(n^2)$ 的理论预期。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **connectome 启发有效**：果蝇大脑中“局部密集连接 + 随机远程投射”形成的小世界网络结构，可以成功迁移到 attention 架构设计中，实现高效且表达能力强的信息路由。
2. ✅ **随机重排是强大归纳偏置**：通过对 token 序列进行层间独立的随机排列，可在 $O(nw)$ 成本下实现 **指数级感受野扩展**，仅需 $O(\log_w n)$ 层即可覆盖整个序列。
3. ✅ **SA 与 SWA 是互补机制**：
   - SWA 提供 **局部一致性**（low perplexity）
   - SA 提供 **全局可达性**（high reasoning accuracy）
   - 两者通过轻量门控融合后，性能超越单一机制及 MoBA
4. ✅ **即插即用有效性验证**：在未重新训练的情况下，将 SA 替换进 Qwen3 系列大模型，在多个 benchmark 上**匹配甚至超过 full attention 表现**，证明其泛化能力和实用性。

### 方法的局限性
- ❗ **对位置编码敏感**：必须保留原始位置信息（如 RoPE 不随 permutation 改变），否则会破坏模型的时间感知能力；
- ❗ **极端短窗口下仍有差距**：当 $w < 32$ 时，即使 SA 也无法完全恢复 full attention 性能；
- ❗ **理论假设简化**：分析中采用 circular window 和 uniform permutation，实际中可能略有偏差；
- ❗ **未探索更复杂的 permutation 策略**：当前使用完全随机排列，未来可尝试结构化或学习型 permutation。

### 未来工作方向
- 🔄 探索 **structured stochastic routing**：例如基于 content-aware 或 layer-adaptive 的 permutation 策略；
- 🧠 结合 **neuroscience 更深层洞见**：如 rich-club 组织、motif over-representation 等 connectome 特征是否可用于指导 attention 设计；
- 💡 扩展至 **vision、multimodal、graph Transformers** 等其他模态；
- 🔍 研究 **SA 在 KV Cache 压缩、长文本摘要** 中的应用潜力；
- 🤝 将 SA 整合进主流框架（如 HuggingFace、vLLM），推动其工业部署。

---

> 📌 **一句话总结**：  
> *Stochastic Attention* 从生物神经网络中汲取灵感，通过简单的随机重排操作，实现了在 $O(nw)$ 时间内构建全局感受野的能力，是一种高效、通用、可插拔的 attention 增强范式，为构建兼具效率与表达力的下一代 LLM 提供了新路径。

</details>

---

### 3. [Universal YOCO for Efficient Depth Scaling](https://arxiv.org/abs/2604.01220)

**Authors**: Yutao Sun, Li Dong, Tianzhu Ye, Shaohan Huang, Jianyong Wang, Furu Wei  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.01220v1  

#### Abstract
The rise of test-time scaling has remarkably boosted the reasoning and agentic proficiency of Large Language Models (LLMs). Yet, standard Transformers struggle to scale inference-time compute efficiently, as conventional looping strategies suffer from high computational overhead and a KV cache that ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Universal YOCO for Efficient Depth Scaling**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前的 **Transformer** 架构在支持 **test-time scaling**（推理时计算扩展）方面存在显著瓶颈：
- 传统循环机制（如 Universal Transformer）虽然能增加计算深度，但会带来高昂的计算开销。
- 随着深度增加，**Key-Value (KV) Cache** 内存占用呈线性增长，严重影响长上下文建模效率和推理吞吐量。

此外，单纯依赖后训练策略进行推理增强效率较低，未能充分利用预训练阶段的知识表达能力。

---

### **提出了什么新方法或新思路**
本文提出 **Universal YOCO (YOCO-U)**，一种结合 **YOCO 架构** 与 **递归计算** 的新型高效深度扩展框架。

#### **核心设计思想：**
- 基于 **YOCO decoder-decoder 架构**，将模型分为两个模块：
  - **Self-Decoder**：处理输入序列，使用 **Efficient Self-Attention (ESA)**（如滑动窗口注意力），生成紧凑的全局 KV Cache。
  - **Cross-Decoder**：复用该共享 KV Cache 进行自回归预测。
- 引入 **Universal Self-Decoder**：对 Self-Decoder 模块进行多轮迭代（T 次），通过参数共享实现递归计算，提升表征能力而不增加参数数量。

#### **关键创新点：**
1. **Partial Recursion（部分递归）**  
   仅在浅层、高效的 Self-Decoder 中应用递归，避免在整个网络中重复执行高成本的全局注意力。
   
2. **Constant Global KV Cache**  
   全局 KV Cache 只需构建一次并保持不变，不受迭代次数 $T$ 影响，极大降低内存开销。

3. **Synergy of Efficient Attention + Recursion**  
   将高效注意力机制与递归计算结合，在有限 FLOPs 下显著增强模型表达力。

---

### **相比现有方法的优势**
| 特性 | Standard Transformer | Universal Transformer | YOCO | YOCO-U |
|------|------------------------|-------------------------|------|--------|
| KV Cache 复杂度 | $O(LND)$ | $O(LTND)$ | $O((N+WL)D)$ | $O((N + WTL)D)$ ✅ |
| Prefilling 时间 | $O(LN^2D)$ ❌ | 更差 | $O(\tilde{L}ND)$ ✅ | $O(\tilde{L}TND)$ ✅ |
| 是否支持递归 | 否 | 是（全模型）❌ | 否 | 是（局部）✅ |
| 参数利用率 | 低 | 高 | 高 | 更高 ✅ |
| 推理效率 | 一般 | 差 ❌ | 高 ✅ | 高 ✅ |

> ✅ 表示优势项；❌ 表示劣势项

YOCO-U 在不牺牲推理效率的前提下实现了有效的深度扩展，是首个将 **高效注意力架构** 与 **递归计算** 成功融合用于可扩展 LLM 的工作。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **语言建模训练数据**：大规模通用文本语料（未具体命名，但训练 token 达 300B）
- **下游任务评估基准**：
  - **常识推理**：ARC-C, ARC-E, PIQA, OBQA
  - **上下文理解**：LAMBADA
  - **数学推理**：GSM8K, MATH, SVAMP, ASDiv, MAWPS, CARP, TABMWP, Gaokao, OlympiadBench, CollegeMath, AMC23（共11个）
  - **代码与书籍长文本建模**：Book & Code 数据集（用于长序列 perplexity 测试）
  - **信息检索能力测试**：Needle In A Haystack (NIAH)

---

### **实验设置和评估指标**

#### **模型配置**
- 主要模型大小：**1.3B 激活参数（总参数达 10B，采用 MoE）**
- 层数：20 层 → 分为 10 层 Self-Decoder + 10 层 Cross-Decoder
- Self-Decoder 使用 **Sliding Window Attention (SWA)**，窗口大小 512
- Cross-Decoder 使用标准 Multi-Head Attention，并启用 **NoPE** 位置编码以增强全局检索能力
- 默认递归次数 $T=3$

#### **训练细节**
- 训练步数：75k 步（约 300B tokens）
- 批大小：4M tokens
- 优化器：AdamW ($\beta_1=0.9, \beta_2=0.95$)，学习率 $1\times10^{-3}$
- 硬件平台：AMD MI300X GPU

#### **评估指标**
- **语言建模损失（Validation Loss）**
- **Perplexity (ppl)**：衡量语言建模质量
- **Accuracy (%)**：各类任务准确率
- **Normalized Accuracy (acc_n)**：针对多选题的长度归一化精度
- **Prefill / Decode Throughput (tokens/s)**：推理吞吐量
- **KV Cache Memory (MB)**：缓存内存占用
- **Scaling Behavior**：FLOPs vs Loss, Tokens vs Loss 曲线

---

### **基线方法对比**
| 基线模型 | 类型 | 说明 |
|--------|-----|------|
| **Transformer** | 标准解码器 | RoPE 编码，无 MoE |
| **YOCO** | 非递归 YOCO | 作为 YOCO-U 的非递归对照 |
| **Universal Transformer (UT)** | 全模型递归 | 整体模块循环 2 次 |
| **RINS** | 早期层递归 Transformer | 前 10 层非递归，后 10 层循环 3 次 |
| **ParScale** | 并行扩展方法 | 利用多个 KV 前缀并行处理 |

所有递归变体均控制总 FLOPs 约为标准 Transformer 的 2 倍，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **语言建模性能（Table 2 & Figure 2）**
- 在相同 FLOPs 下，YOCO-U 相比 YOCO：
  - **验证损失降低 $\Delta L = 0.033$**
  - **平均任务得分从 41.78 提升至 46.23（+4.45 pts）**
- 在相同训练 token 数下，YOCO-U 仅需 **80B tokens** 即达到 YOCO 使用 **210B tokens** 的性能 → **节省约 62% 训练数据**

#### **数学推理能力（Figure 3）**
- 经过 Thinking SFT 微调后，在 11 个数学基准上的表现：
  - YOCO-U **全面优于 YOCO 基线**
  - **平均准确率提升高达 24.4%**

#### **通用与长上下文任务（Table 3）**
| 模型 | Avg Acc ↑ | KV Cache ↓ |
|------|----------|-----------|
| Transformer | 47.1 | 10240 MB (@256K) |
| YOCO | 47.0 | 522 MB |
| RINS | 48.3 | 20480 MB |
| **YOCO-U** | **48.3** | **542 MB** |

- YOCO-U 与 RINS 性能达到同一水平（+1.2 pts 超越非递归模型），但 **KV Cache 仅为 RINS 的 ~2.6%**
- 显示其在保持高性能的同时具备极佳的内存效率

#### **长上下文建模能力（Figure 4）**
- 在 Book 和 Code 数据上测试不同前缀长度下的最后 512 token 的 perplexity
- YOCO-U 表现与 RINS 相当，明显优于非递归模型（Transformer/YOCO）
- 证明其有效利用长距离依赖关系

#### **信息检索能力（Table 4）**
| 模型 | S-NIAH-1 | S-NIAH-2 |
|------|---------|---------|
| Transformer | 0.87 | 0.82 |
| YOCO | 1.00 | 0.86 |
| RINS | 0.99 | 0.91 |
| **YOCO-U** | **1.00** | **0.95** |

- YOCO-U 在 Needle-in-a-Haystack 测试中表现最佳，尤其在双针情况下仍保持高召回率，表明其强大的信息定位与访问能力

---

### **消融实验结果（Table 5）**

| 消融变体 | Avg Acc |
|--------|--------|
| YOCO (non-recursive) | 46.95 |
| YOCO-U (proposed) | **48.25** |
| Upper Loop (Cross-Decoder 递归) | 47.34 |
| Upper Loop w/o Shared KV | 46.41 |
| Deeper model (instead of wide) | 48.59 |

#### **关键发现：**
- 将递归应用于 **更深层 Cross-Decoder** 效果不如作用于浅层 Self-Decoder，说明“**think deeper in shallow blocks**”更有效。
- 即使改变模型宽深比例，“Deeper”版本中 YOCO-U 依然受益，说明其增益来自架构本身而非布局偏置。
- “Upper Loop w/o Shared KV” 性能下降，凸显 **Shared KV Cache 设计的重要性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **递归计算 + 高效注意力 = 高效深度扩展的新范式**  
   YOCO-U 成功验证了将递归限制在浅层高效模块中的可行性，既能提升表征能力，又不会引发 KV Cache 膨胀。

2. ✅ **全局 KV Cache 只需构建一次即可复用**  
   这是 YOCO 系列的核心优势，YOCO-U 完美继承并在递归场景下进一步放大这一优势。

3. ✅ **显著提升 token 与参数利用率**  
   - 达到同等性能所需训练 token 减少 **62%**
   - 在相同激活参数下，性能接近甚至超越更大模型

4. ✅ **推理效率高度可控**  
   - Prefilling 时间保持线性复杂度
   - Decoding 吞吐量仅比非递归 YOCO 下降 5%
   - KV Cache 增长可忽略（主要来自局部窗口）

5. 🔍 **表示分析显示收敛趋势**  
   Angular distance 分析表明，随着迭代次数增加，表示逐渐趋于稳定（接近固定点），暗示存在最优迭代次数。

---

### **方法的局限性**
- 当前递归次数 $T$ 是固定的，缺乏动态调整机制（如根据输入难度自适应循环次数）。
- 实验集中在 MoE 架构，是否适用于纯 Dense 模型有待更多验证。
- 对极端超长序列（>1M）的支持尚未测试，尽管理论上有潜力。

---

### **未来工作方向**
1. **Dynamic Recursion**  
   结合 Early Exit 或 Halting Unit，实现 **adaptive loop iteration**，按需分配计算资源。

2. **Hybrid Scaling Strategies**  
   探索 YOCO-U 与其他 scaling 方法（如 ParScale、MoE、宽度扩展）的组合，形成统一的 **compute-efficient scaling law**。

3. **Extension to Multimodal & Agent Settings**  
   将 YOCO-U 应用于视觉-语言或多智能体系统，探索其在复杂决策链中的潜力。

4. **Hardware-aware Optimization**  
   针对 YOCO-U 的内存访问模式优化 Kernel 实现（如 FlashAttention 变种），进一步释放吞吐潜力。

---

> 📌 **一句话总结**：  
> **YOCO-U 开创性地将递归计算引入高效注意力架构，在几乎不增加 KV Cache 开销的前提下实现了卓越的能力-效率权衡，为下一代可扩展、低成本的大语言模型提供了极具前景的技术路径。**

</details>

---

### 4. [TENT: A Declarative Slice Spraying Engine for Performant and Resilient Data Movement in Disaggregated LLM Serving](https://arxiv.org/abs/2604.00368)

**Authors**: Feng Ren, Ruoyu Qin, Teng Ma, Shangming Cai, Zheng Liu, Chao Lei, Dejiang Zhu, Ke Yang, Zheming Li, Jialei Cui, Weixiao Huang, Yikai Zhao, Yineng Zhang, Hao Wu, Xiang Gao, Yuhao Fu, Jinlei Jiang, Yongwei Wu, Mingxing Zhang  
**Category**: cs.DC  
**Published**: 2026-04-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.00368v1  

#### Abstract
Modern GPU clusters are built upon a complex hierarchy of heterogeneous interconnects, ranging from multi-rail RDMA to proprietary fabrics such as Multi-Node NVLink and Ascend UB. Orchestrating these diverse links effectively remains a critical challenge in disaggregated LLM serving. Operating Moonc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TENT: A Declarative Slice Spraying Engine for Performant and Resilient Data Movement in Disaggregated LLM Serving

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

现代大规模 GPU 集群采用异构互联架构（如 NVLink、RDMA、MNNVL、Ascend UB 等），但在**解耦式大语言模型（Disaggregated LLM）服务**场景下，现有的数据传输引擎（如 Mooncake TE、NIXL、UCCL）存在以下三大瓶颈：

1. **静态绑定（Static Binding）导致通信孤岛**  
   应用在初始化时必须显式绑定特定传输后端（如 NVLink 或 RDMA），不同部署无法互通，形成隔离域。

2. **无状态条带化（State-Blind Striping）浪费多轨带宽**  
   多轨（multi-rail）系统虽提升峰值带宽，但现有引擎采用轮询或哈希分片策略，忽略链路拥塞信号，导致“最慢轨道决定整体延迟”（Head-of-Line Blocking）。

3. **脆弱的执行机制依赖人工干预**  
   当 NIC 故障或链路退化时，现有引擎将错误暴露给上层应用，缺乏自动恢复能力，导致性能长期下降甚至中断。

---

### 🚀 提出的新方法：TENT

TENT 是一个**声明式切片喷洒引擎**（Declarative Slice Spraying Engine），其核心思想是**将传输意图与物理执行解耦**，实现动态、自适应、高弹性的数据移动。

#### 创新点：

1. **从静态绑定到动态编排（Dynamic Orchestration）**  
   - 引入统一的 **Segment 抽象**，屏蔽底层存储介质差异（GPU HBM、Host DRAM、SSD）。
   - 运行时根据全局拓扑图选择最优路径，支持跨设备多跳路由（staged routing），无需应用介入。

2. **从无状态条带化到遥测驱动的切片喷洒（Telemetry-Driven Slice Spraying）**  
   - 将大象流（elephant flows）拆分为细粒度切片（默认 64KB）。
   - 基于实时遥测（队列长度、完成时间）动态调度每个切片至**预计完成时间最短**的链路。
   - 支持负载感知溢出（load-aware spillover），避免慢轨阻塞。

3. **从脆弱执行到弹性自愈（Resilient Self-Healing）**  
   - 在数据平面内嵌入双层容错机制：
     - **链路层**：检测退化轨道并临时排除，后台探测恢复后重新纳入。
     - **传输层**：当某 backend 完全失效（如 NVLink 驱动崩溃），自动切换至备用协议（如 RDMA/TCP）。
   - 所有重试与路径切换对应用透明，实现 **<50ms 自动恢复**。

---

### 🔍 相比现有方法的优势

| 维度 | 现有方法（Mooncake TE / NIXL / UCCL） | TENT |
|------|----------------------------------------|------|
| 路径选择 | 编译期/启动期静态绑定 | 请求级动态决策 |
| 分片策略 | 固定轮询或哈希，无视链路状态 | 遥测驱动，按预测完成时间调度 |
| 容错机制 | 控制面报错，需上层重建连接 | 数据面自动重试与路径切换 |
| 可移植性 | 依赖特定 vendor stack | 插件式 backend（<800 LOC），跨 RDMA/NVLink/MNNVL/Ascend 等 |
| 性能表现 | 易受慢轨拖累，尾延迟高 | 充分利用聚合带宽，尾延迟显著降低 |

---

## 2. 核心实验方法和设置

### 🧪 实验平台

- **主测试环境**：H800 HGX 节点 × 8 GPU，每节点配备：
  - 8 × 200 Gbps RoCE NICs（multi-rail RDMA）
  - GPU 间通过 NVLink 互联
  - 双路 Intel Xeon CPU，NUMA 架构
- **其他验证环境**：MNNVL 集群、Ascend UB/HIXL 集群、NVMe-oF 存储系统

### 📊 评估指标

| 指标 | 含义 |
|------|------|
| **Throughput** | 输入 token 吞吐量（tok/s） |
| **TTFT** | Time-To-First-Token，首 token 延迟 |
| **P90/P99 TTFT** | 尾部延迟 |
| **TPOT** | Time-Per-Output-Token |
| **Checkpoint Apply Time** | 模型参数更新耗时 |
| **Recovery Time** | 故障后吞吐恢复时间 |

### 🆚 基线方法对比

| 基线 | 说明 |
|------|------|
| **Mooncake TE** | 工业界生产级 P2P 引擎，本文前身 |
| **NIXL** | NVIDIA 推出的跨 NIC 传输框架，基于 UCX |
| **UCCL-P2P** | UCCL 的点对点模块，支持多轨 RDMA |
| **Round-Robin / Hash-based Striping** | 传统条带化策略作为微基准对照 |

---

## 3. 主要实验结果和性能指标

### 📈 端到端性能提升（End-to-End Workloads）

#### （1）SGLang HiCache + KVCache 复用场景

| 指标 | Baseline（无 HiCache） | Mooncake TE | **TENT** | 提升幅度 |
|------|------------------------|-------------|----------|----------|
| 输入吞吐（tok/s） | 20,757 | 58,006 | **78,759** | ↑ **1.36× vs Mooncake TE** |
| P90 TTFT（秒） | 4.02 | 0.90 | **0.67** | ↓ **26.4%** |
| 第10轮平均 TTFT | 4.09 | 0.97 | **0.66** | ↓ 32% |

> 💡 **关键原因**：TENT 能智能识别 intra-node 流量并优先使用 NVLink，而 Mooncake TE 统一走 RDMA。

#### （2）Moonshot Checkpoint Engine（RL 参数更新）

| 模型 | Mooncake TE（秒） | **TENT（秒）** | 加速比 |
|------|--------------------|----------------|--------|
| Qwen3-235B-A22B | 12.87 | **10.34** | ↑ **19.7%** |
| GLM-4.5-Air | 7.17 | **5.30** | ↑ **26.1%** |

> ✅ 在 256× H20 半生产集群上，TENT 可在数十秒内完成万亿参数模型的权重刷新。

---

### ⚙️ 微基准测试（Microbenchmarks）

#### （1）Host-to-Host 多轨 RDMA 读写

| 指标 | TENT vs Mooncake TE | 结果 |
|------|---------------------|------|
| 写吞吐（≥1MB） | ↑ **33.7%** | 达理论极限更高比例 |
| P99 延迟 | ↓ 至 **27.6%** | 显著缓解慢轨影响 |

> 📌 UCCL-P2P 因单 NIC 绑定受限于 per-NIC 带宽；NIXL 仅启用部分 NIC。

#### （2）GPU-to-GPU 跨节点写（KVCache 典型负载）

| 指标 | TENT vs Mooncake TE |
|------|---------------------|
| 吞吐 | ↑ **2.1×** |
| P99 延迟 | ↓ 至 **46.7%** |

> ✅ TENT 动态启用 tier-2 NICs 分担负载，而 Mooncake TE 死守 tier-1 NIC 导致饱和。

#### （3）并发扩展性（Submission Threads）

- **GPU-to-GPU 读带宽**（4MB block）：
  - TENT 在 16 线程即达峰值 **144 GB/s**（>77% 理论上限）
  - Mooncake TE 需更多线程仍难饱和，易因“慢轨主导”而性能波动

#### （4）批大小扩展性（Batch Size）

- 固定单线程，batch size 从 1 → 128：
  - TENT 吞吐接近硬件极限（800 Gbps）
  - 比 Mooncake TE 高 **1.16–2.72×**
  - P90 延迟平均 ↓ **27.06%**

---

### 🔍 消融实验与敏感性分析

#### （1）调度参数敏感性（P1: NUMA Penalty）

- 在 GPU-to-GPU 场景下调整 `P1`（tier-2 访问惩罚系数）：
  - 最优值约为 **P1 = 3**
  - 即使误设为 1 或 10，性能下降有限（<15%）
  - 归功于 **EWMA 反馈机制**持续修正预测误差

#### （2）遥测反馈有效性

- 若关闭队列深度反馈，退化为轮询：
  - P99 延迟上升至 **3.8×**
  - 吞吐下降约 30%
- 证明遥测驱动调度是性能优势的关键来源

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **声明式接口 + 动态编排 > 指令式绑定**  
   将“**我要传什么**”与“**怎么传**”分离，使系统能随运行时状态自适应优化。

2. **细粒度切片喷洒可消除 HoL Blocking**  
   遥测驱动的 per-slice 调度有效规避慢轨，释放多轨系统的真正潜力。

3. **数据面自愈可行且高效**  
   在 <50ms 内完成故障切片重调度，无需应用层干预，极大提升系统鲁棒性。

4. **抽象开销极低，具备强可移植性**  
   TENT 在 NVLink、MNNVL、Ascend UB 上均接近原生性能（见 Table 4）：
   - NVLink: 172.0 GB/s（理论 204.5）
   - MNNVL: 781.8 GB/s（理论 956.2）
   - io_uring: 6.0 GB/s（匹配原生）

---

### ⚠️ 局限性

1. **依赖底层 transport 的稳定性与可观测性**  
   若 backend 不上报准确 completion 延迟或 queue depth，调度模型精度会下降。

2. **小包场景收益有限**  
   对极小传输（<64KB），切片调度带来的调度开销可能抵消收益。

3. **中心化调度器潜在瓶颈**  
   当前 orchestrator 为轻量控制面，但在超大规模集群中仍需考虑分布式扩展。

---

### 🔮 未来工作方向

1. **支持更复杂的拓扑感知策略**  
   如结合 switch hop count、拥塞窗口等网络层信息进一步优化路径选择。

2. **引入 ML-based 预测模型替代线性估计**  
   利用历史行为学习更精准的服务时间预测函数。

3. **与 MoE、EP 等计算范式深度协同**  
   将 slice spraying 与 expert placement 联合优化，实现 compute-data co-scheduling。

4. **开放生态集成**  
   已开源至 GitHub（https://github.com/kvcache-ai/Mooncake），计划支持更多 backend（如 CXL.mem、OmniPath）。

---

## 总结

> **TENT 通过“声明式 + 切片喷洒 + 自愈”三位一体设计，在异构 LLM 集群中实现了高性能、高弹性、高可移植的数据移动引擎。它不仅是传输优化工具，更是构建下一代解耦式 AI 基础设施的核心数据平面组件。**

</details>

---

### 5. [Execution-Verified Reinforcement Learning for Optimization Modeling](https://arxiv.org/abs/2604.00442)

**Authors**: Runda Guan, Xiangqing Shen, Jiajun Zhang, Yifan Zhang, Jian Cheng, Rui Xia  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.00442v1  

#### Abstract
Automating optimization modeling with LLMs is a promising path toward scalable decision intelligence, but existing approaches either rely on agentic pipelines built on closed-source LLMs with high inference latency, or fine-tune smaller LLMs using costly process supervision that often overfits to a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Execution-Verified Reinforcement Learning for Optimization Modeling

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的优化建模自动化方法面临两大挑战：
- **高成本的过程监督**（Process Supervision）：依赖人工标注变量定义、约束推导等中间步骤，标注成本高昂且难以扩展。
- **跨求解器泛化能力差**：基于特定求解器（如 Gurobi）训练的模型在迁移到其他求解器（如 OR-Tools）时性能急剧下降，因为模型过度拟合了特定 API 的语法模式。

### 提出了什么新方法或新思路
提出 **Execution-Verified Optimization Modeling (EVOM)**，一种基于执行验证的强化学习框架，其核心思想是：
- 将数学规划求解器（如 Gurobi）视为一个**确定性的交互式验证器**（deterministic, interactive verifier）。
- 采用“生成-执行-反馈-更新”（generate-execute-feedback-update）的闭环学习范式。
- 仅依赖**结果级监督**（outcome-only supervision），即问题-答案对 $(q, a)$，无需任何中间过程标注。

### 相比现有方法的优势
| 维度 | 传统方法（SFT） | EVOM |
|------|------------------|-------|
| 监督信号 | 需要精细的过程监督（如参考代码） | 仅需最终答案 + 执行反馈 |
| 跨求解器迁移 | 需重构数据集并重新训练 | 支持零样本迁移（zero-shot transfer） |
| 适配新求解器 | 成本高，需大量标注 | 低成本，只需切换执行环境继续训练 |
| 泛化性 | 易过拟合特定求解器语法 | 学习通用数学建模逻辑 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **NL4OPT**：NeurIPS 2022 竞赛数据集，包含 289 个线性规划问题。
- **MAMO**：评估 LLM 数学建模能力的基准，分为 EasyLP 和 ComplexLP 子集。
- **IndustryOR**：首个工业级 OR 问题基准，涵盖 13 个行业的真实场景。
- **OptiBench**：端到端优化问题求解基准，包含线性和非线性问题。

所有数据均转换为仅保留自然语言描述 $q$ 和真实输出 $a$（最优目标值或状态标签），丢弃原始过程级注释。

### 实验设置和评估指标
- **基础模型**：Qwen2.5-7B
- **求解器后端**：Gurobi、OR-Tools、COPT
- **训练算法**：GRPO（Group Relative Policy Optimization）和 DAPO（Decoupled Clipping and Dynamic Sampling）
- **沙箱执行**：每个程序在 10 秒超时、2GB 内存限制下运行，捕获执行状态、目标值和日志。
- **评估指标**：**Accuracy**，即预测结果在相对误差容忍度 $\epsilon_{\text{eval}}$ 内的比例：
  $$
  \frac{|u - a|}{\max(|a|, \delta)} \leq \epsilon_{\text{eval}}
  $$
  主要使用 $\epsilon_{\text{eval}} = 0.05$，严格测试使用 $10^{-4}$。

### 基线方法对比
- **Prompting-based Models**：
  - DeepSeek-R1、OpenAI o1、GPT-4o（标准、CoT、CoE）、OptiMUS
- **Training-based Models**：
  - ORLM (SFT)：基于完整 OR-Instruct 数据集进行监督微调的代表方法
- **本方法**：
  - EVOM (GRPO/DAPO)

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Gurobi 后端，$\epsilon_{\text{eval}}=0.05$）

| Model | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndustryOR | Avg |
|-------|-----------|--------|--------|--------|------------|-----|
| ORLM (SFT) | 60.96 | 84.89 | 88.34 | 35.71 | 27.00 | 59.38 |
| **EVOM (GRPO)** | **62.95** | **84.08** | **88.19** | **34.28** | **31.00** | **60.10** |

✅ **结论**：EVOM 在平均性能上**匹配甚至略优于**需要过程监督的 SFT 方法。

### 零样本求解器迁移（Zero-Shot Transfer to OR-Tools）

| Model | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndusOR |
|-------|-----------|--------|--------|--------|---------|
| ORLM (SFT) | 3.49 | 4.89 | 0.00 | 1.42 | 6.00 |
| **EVOM (GRPO)** | **54.31** | **77.55** | **84.81** | **22.27** | **24.00** |

✅ **结论**：EVOM 展现出强大的零样本迁移能力，而 SFT 模型几乎完全失效。

### 低成本适配新求解器（Low-Cost Adaptation）

| Solver | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndustryOR |
|--------|-----------|--------|--------|--------|------------|
| Gurobi Δ | +21.76 | +28.98 | +22.24 | +5.85 | +10.00 |
| OR-Tools Δ | +11.63 | +17.96 | +28.99 | -2.37 | +11.00 |

✅ **结论**：通过简单的环境切换和继续训练，EVOM 可快速适应新求解器，显著提升性能。

### 消融实验结果
- **显式推理块 `<think>` 的作用**：
  - 移除 `<think>` 导致性能大幅下降，尤其在复杂任务（如 IndustryOR）上。
  - 表明 `<think>` 不是事后解释，而是必要的内部推理空间。
- **优化器选择（GRPO vs DAPO）**：
  - 两者性能几乎一致，说明效果主要来自执行反馈而非优化器设计。
- **小规模模型表现**：
  - 即使是 Qwen2.5-3B 模型，在 GRPO 训练后也能接近 7B 基础模型的表现，显示该方法对小模型有放大效应。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **结果级监督足以驱动复杂优化建模**：无需昂贵的过程监督，仅靠执行反馈即可让模型学会从自然语言到可执行代码的映射。
2. ✅ **求解器即验证器**（Solver-as-Verifier）：将求解器作为外部验证模块，构建闭环学习系统，是实现可扩展决策智能的有效路径。
3. ✅ **跨求解器泛化成为可能**：通过将求解器视为验证环境的一部分，实现了真正的“即插即用”式迁移。
4. ✅ **显式推理有助于学习**：`<think>` 块被证明是必要组件，它充当了策略梯度优化中的“内部工作区”。

### 方法的局限性
- **冷启动问题**：对于预训练中极少出现的求解器（如 COPT），初始生成成功率极低，导致稀疏奖励问题。
- **深层语义错误仍难解决**：执行反馈能有效纠正语法和变量类型错误，但对复杂的**约束遗漏或逻辑错误**改善有限（见错误分析）。
- **非数值问题支持不足**：当前框架主要针对可量化的目标函数，对定性或组合搜索类问题支持较弱。

### 未来工作方向
- 探索更细粒度的分层奖励机制（Hierarchical Reward），以引导关键约束的正确建模。
- 结合少量过程提示（Process Hints）与执行反馈，进一步提升复杂问题的建模能力。
- 扩展至多阶段、动态优化问题，支持在线学习与自适应调整。
- 研究如何利用 EVOM 框架进行逆向工程，从已有解决方案反推问题结构。

> **总体评价**：EVOM 提供了一条轻量、高效、可扩展的优化建模自动化路径，推动了决策智能系统的实用化进程。

</details>

---

### 6. [Adaptive Parallel Monte Carlo Tree Search for Efficient Test-time Compute Scaling](https://arxiv.org/abs/2604.00510)

**Authors**: Hongbeen Kim, Juhyun Lee, Sanghyeon Lee, Kwanghoon Choi, Jaehyuk Huh  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.00510v1  

#### Abstract
Monte Carlo Tree Search (MCTS) is an effective test-time compute scaling (TTCS) method for improving the reasoning performance of large language models, but its highly variable execution time leads to severe long-tail latency in practice. Existing optimizations such as positive early exit, reduce la...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Adaptive Parallel Monte Carlo Tree Search for Efficient Test-time Compute Scaling*

---

## 1. 论文的主要贡献和创新点

### ✅ **解决了什么问题**

Monte Carlo Tree Search (**MCTS**) 是一种有效的 **Test-time Compute Scaling (TTCS)** 方法，能够显著提升大语言模型（**LLM**）在复杂推理任务中的表现。然而，其**高度可变的执行时间**导致在实际部署中出现严重的 **long-tail latency**（长尾延迟），尤其是在高并发请求下，少数计算密集型请求会严重拖慢系统响应。

现有优化方法如 **positive early exit** 能在找到高质量解时提前终止搜索，但在搜索长期无进展的情况下无效，无法解决“无望”搜索带来的资源浪费。

---

### ✅ **提出了什么新方法或新思路**

本文从**系统级资源管理**视角出发，提出两个核心机制：

#### （1）**Negative Early Exit（负向早退）**
- **思想**：识别并剪枝那些**不可能产生高质量解**的 MCTS 搜索路径。
- **判断标准**：若当前搜索树的所有叶节点（leaf nodes）均为“**futile**”（即其累积奖励已无法达到接受阈值），则判定整个搜索无望，立即终止。
- **理论依据**：在基于最小值或累积乘积的轨迹评分机制下，最终得分受叶节点限制，因此低分叶节点预示着无望成功。

#### （2）**Adaptive Boosting（自适应增强）**
- **思想**：将因 early exit 节省下来的计算资源，动态重新分配给更有潜力的并发搜索任务。
- **调度策略**：基于任务的等待时间和接近完成程度（progress ratio > 90%）动态调整并行度（Degree of Parallelism, DOP），优先加速即将完成的任务，从而减少尾部延迟。

此外，作者还实现了：
- **WU-PUCT**：扩展 WU-UCT 并行机制至 PUCT 框架，支持使用“飞行中”统计量进行节点选择，提高并行效率。
- **Selective Futility Check**：利用早期步骤得分与最终得分的相关性，跳过低潜力分支的冗余检查，进一步提升效率。

---

### ✅ **相比现有方法的优势**

| 对比维度 | 现有方法（如 Positive Early Exit） | 本文方法（PE + NE + Boosting） |
|--------|-------------------------------|------------------------------|
| **尾延迟控制** | 仅对“好情况”有效，无法处理无进展搜索 | 主动识别并终止“无望”搜索，直接降低尾延迟 |
| **资源利用率** | 资源静态分配，易造成浪费 | 动态回收与再分配，提升整体吞吐 |
| **系统视角** | 加速单个搜索过程 | 将 MCTS 视为需主动管理的系统资源池 |
| **并行控制** | 固定或最大并行度 | 自适应调节 DOP，避免资源争用 |

---

## 2. 核心实验方法和设置

### 📚 **使用的数据集**

- **Math500**：从 MATH 数据集中提取的 500 道数学题，用于评估逐步求解准确性。
- **AMC23**：2023 年美国数学竞赛题目，用于测试泛化能力。

> 注：由于 AMC23 样本少（仅 40 题），性能测试时通过添加不同前缀进行负载增强，但准确率报告仍基于原始数据集。

---

### ⚙️ **实验设置**

- **硬件平台**：单节点，配备 4 块 NVIDIA H100-SXM 80GB GPU（NVLink 连接）
  - 2 块用于生成模型推理
  - 2 块用于奖励模型（**PRM**）推理
- **模型配置**：
  - 生成模型：`Llama-3.1-8B-Instruct`, `Qwen2.5-14B-Instruct`
  - 奖励模型：`Qwen2.5-Math-PRM-7B`
- **集成框架**：基于 **vLLM** 实现系统调度与内存管理。

---

### 🎯 **评估指标**

| 指标 | 描述 |
|------|------|
| **p50 / p99 End-to-End Latency** | 请求端到端延迟的中位数与第99百分位数，衡量平均与尾部性能 |
| **Throughput (req/sec)** | 单位时间内处理的请求数量 |
| **Accuracy (%)** | 正确解答的比例，用于验证方法不牺牲推理质量 |
| **Generated Tokens** | 平均生成 token 数，反映计算开销 |

---

### 🔁 **基线方法对比**

| 方法 | 描述 |
|------|------|
| **Beam Search** | 并行搜索基线，beam size=8，启用 positive early exit |
| **Vanilla MCTS** | 序列化 MCTS，无任何优化 |
| **PE (Positive Early Exit)** | 达到置信阈值后提前退出 |
| **PE + NE** | PE + 负向早退 |
| **PE + NE + Boosting** | 完整系统，含资源再分配 |

---

## 3. 主要实验结果和性能指标

### 📈 **关键性能数据**

#### （1）**尾延迟（p99 Latency）大幅下降**

- 在 `(Math500, Llama)` 场景下：
  - 相比 **Vanilla MCTS**，p99 延迟降低 **2.83×**
  - 相比 **PE-only**，p99 延迟降低 **1.46×**
- 在高负载场景下，**PE+NE+Boosting** 持续优于所有基线。

> 图 7 显示，在请求速率上升时，本文方法能有效抑制尾延迟增长。

#### （2）**吞吐量（Throughput）显著提升**

- 最高实现 **2.44× 吞吐提升**（vs Vanilla MCTS）
- 典型增益：
  - `(Math500, Qwen)`：+1.76×
  - `(Math500, Llama)`：+1.37×
  - `(AMC23, Llama)`：+1.15×

> 图 8 表明，early exit 释放的资源被高效重用于其他请求，形成正向循环。

#### （3）**消融实验结果**

| 方法 | p99 Latency ↓ | Throughput ↑ | 说明 |
|------|----------------|--------------|------|
| Vanilla | 1.00× | 1.00× | 基线 |
| + PE | ~0.72× | ~1.30× | 快速收敛请求受益 |
| + PE + NE | ~0.67× | ~1.36× | 进一步削减无望搜索 |
| + PE + NE + Boosting | **~0.35×** | **~2.44×** | 资源再分配带来质变 |

> 结果表明：**Boosting 是实现最大收益的关键组件**。

#### （4）**准确率保持稳定**

| 方法 | Math500 (Llama) | AMC23 (Llama) |
|------|------------------|---------------|
| Vanilla MCTS | 75.3% | 57.5% |
| PE | 76.2% | 47.5% |
| Ours (Full) | **74.8%** | **55.0%** |

> 准确率波动在合理范围内，未出现显著下降。AMC23 上的小样本导致方差较大，但趋势可控。

---

## 4. 关键结论和发现

### ✅ **主要发现**

1. **MCTS 的部署瓶颈不在精度，而在资源管理**  
   - 尾延迟本质上是系统级资源争用问题，需从调度角度解决。

2. **Negative Early Exit 可有效识别“无望”搜索**  
   - 利用轨迹得分上界性质，可在数学上证明某些搜索注定失败，提前终止安全且高效。

3. **资源再分配（Boosting）是放大收益的关键**  
   - 单纯剪枝只能节省资源，而**动态重分配**才能转化为更低延迟和更高吞吐。

4. **系统设计应兼顾微观剪枝与宏观调度**  
   - 本文提出的“细粒度剪枝 + 预测性调度”架构，为 LLM 推理服务提供了新范式。

---

### ⚠️ **方法的局限性**

1. **依赖高质量 PRM（Process Reward Model）**
   - 若 PRM 评分不准，negative early exit 可能误判可行路径为“futile”，导致漏解。

2. **并行化存在开销**
   - 在简单任务主导的负载中（如 `(amc23, Qwen)`），过度并行反而增加调度开销，轻微降低吞吐。

3. **对搜索结构敏感**
   - 方法假设 MCTS 使用特定聚合方式（如 cum-product），可能不适用于所有变体。

---

### 🔮 **未来工作方向**

1. **更智能的 early exit 判定机制**
   - 引入轻量预测模型，学习何时该终止搜索，而非依赖固定阈值。

2. **跨请求的知识迁移**
   - 利用历史搜索模式指导当前任务的资源分配，进一步提升调度效率。

3. **支持更多 TTCS 方法**
   - 将 adaptive boosting 思想推广至 Best-of-N、Tree-of-Thought 等其他推理框架。

4. **异构硬件适配**
   - 在 CPU-GPU 混合或分布式环境下优化 rollout 分发与同步。

---

## ✅ 总结

本文提出了一套面向 **MCTS-based TTCS** 的系统级优化方案，通过引入 **Negative Early Exit** 和 **Adaptive Boosting**，实现了：

- **高达 2.83× 的 p99 延迟降低**
- **最高 2.44× 的吞吐提升**
- **推理准确率基本不变**

> 💡 **核心洞见**：将 LLM 推理视为一个需要动态资源调配的“操作系统”，而非单纯的算法加速问题。这一系统思维为未来高效、可扩展的推理服务奠定了基础。

</details>

---

### 7. [Agent Q-Mix: Selecting the Right Action for LLM Multi-Agent Systems through Reinforcement Learning](https://arxiv.org/abs/2604.00344)

**Authors**: Eric Hanchen Jiang, Levina Li, Rui Sun, Xiao Liang, Yubei Li, Yuchen Wu, Haozheng Luo, Hengli Li, Zhi Zhang, Zhaolu Kang, Kai-Wei Chang, Ying Nian Wu  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.00344v1  

#### Abstract
Large Language Models (LLMs) have shown remarkable performance in completing various tasks. However, solving complex problems often requires the coordination of multiple agents, raising a fundamental question: how to effectively select and interconnect these agents. In this paper, we propose \textbf...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Agent Q-Mix: Selecting the Right Action for LLM Multi-Agent Systems through Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 LLM-based Multi-Agent Systems（MAS）在解决复杂任务时，通常依赖**静态通信拓扑**（如链式、星型、全连接图）或**中心化生成的动态拓扑**。这些方法存在以下问题：
- **缺乏灵活性**：固定拓扑无法根据任务难度自适应调整通信强度。
- **中心化瓶颈**：现有自适应方法由一个全局控制器决定整个通信图，导致部署时缺乏去中心化执行能力。
- **鲁棒性差**：当某个代理失效或提供错误输出时，系统难以动态隔离故障节点。

本文提出的问题是：**如何实现一种去中心化的、可学习的通信拓扑选择机制，使每个 agent 能独立决策其通信行为，并联合形成高效的协作结构？**

---

### 提出了什么新方法或新思路
作者提出了 **Agent Q-Mix**，一个基于 **Cooperative Multi-Agent Reinforcement Learning (MARL)** 的框架，将通信拓扑学习建模为一个 **Networked Multi-Agent Markov Decision Process (Networked MMDP)**。

#### 核心思想：
- 将每个 agent 的通信动作定义为一组**离散且可解释的动作空间**（6种），例如：
  - `solo_process`：独立工作
  - `broadcast_all`：广播给所有其他 agent
  - `selective_query`：向特定 agent 发起查询
  - `aggregate_refine`：聚合所有输入进行精炼
  - `execute_verify`：将结果传给下一 agent 验证
  - `debate_check`：与某一 agent 进行双向辩论
- 所有 agent 在每轮中**独立选择自己的通信动作**，共同诱导出当前轮次的通信图 $ G_t $。
- 使用 **QMIX value factorization** 实现 **Centralized Training with Decentralized Execution (CTDE)**，确保训练时利用全局信息，推理时各 agent 可独立决策。

#### 架构设计：
- **Topology-aware GNN Encoder**：编码当前通信图结构，捕捉拓扑上下文。
- **GRU Memory**：维护跨轮次的历史状态，支持多轮推理。
- **Per-agent Q-heads + QMIX Mixer**：每个 agent 输出局部 Q 值，通过单调混合网络组合成联合价值函数，保证 IGM（Individual-Global-Max）性质成立。

---

### 相比现有方法的优势
| 维度 | Agent Q-Mix | 现有方法（如 G-Designer, GTD, AutoGen） |
|------|-------------|----------------------------------------|
| **通信控制方式** | 去中心化、agent 自主决策 | 中心化生成整张图 |
| **适应性** | 动态按任务难度调整通信密度 | 固定或一次性生成 |
| **可扩展性** | 支持在线调整、容错能力强 | 对 agent 失效敏感 |
| **效率** | 显著降低 token 开销 | 容易产生冗余通信 |
| **理论保障** | 满足 IGM 性质，支持一致的去中心化执行 | 缺乏分解依据 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖三大类共 **7个基准任务**，外加一个挑战性综合测试集 **Humanity's Last Exam (HLE)**：

| 类别 | 数据集 | 描述 |
|------|-------|------|
| **Coding** | LiveCodeBench v6, HumanEval | 编程生成与修复任务 |
| **Reasoning** | MMLU-Pro | 多领域多选题推理（涵盖科学、经济、医学等） |
| **Mathematics** | AIME25, AIME26, HMMT, Beyond-AIME | 高中数学竞赛题，难度递增 |
| **Comprehensive** | Humanity's Last Exam (HLE) | 新提出的跨学科高难 MCQ 测试集，前250题用于评估 |

---

### 实验设置和评估指标

#### 模型配置
- **Backbone LLMs**：
  - GPT-OSS:120B（开源大模型）
  - Gemini-3.1-Flash-Lite（轻量级商业模型）
- **Agent Team Structure**：
  - 每个任务使用 3 个 domain-specialist agents + 1 FinalRefer 决策节点
  - 角色包括：Math Solver, Programming Expert, Inspector, AnalyzeAgent 等（见 Appendix J）

#### 训练细节
- **训练样本数**：仅用 **15个示例/领域** 进行训练，验证小样本泛化能力
- **通信轮数**：
  - 数学任务：T=3
  - 编码/推理任务：T=2
- **奖励函数**：
  $$
  R = w_{\text{acc}} \cdot \text{accuracy} - w_{\text{tok}} \cdot \min\left(\frac{\text{tokens\_used}}{\text{max\_tokens}}, 1\right)
  $$
  其中 $\text{max\_tokens}=10,000$，平衡准确率与 token 成本。

#### 评估指标
- **Accuracy (%)**：主要性能指标
- **Total Token Usage**：衡量通信效率
- **Robustness to Adversarial Agents**：替换一个 agent 为恶意 agent 后的性能下降幅度
- **Ablation Studies**：分析 agent 数量、训练数据量、通信轮数、reward 权重的影响

---

### 基线方法对比
分为四类 baseline：

| 类型 | 方法 |
|------|------|
| **Single-agent** | Base (direct prompting) |
| **Static MAS** | LLM-Debate |
| **Adaptive Topology** | GPTSwarm, AgentDropout, G-Designer, MaAS, TopoDIM, GTD |
| **Commercial Frameworks** | LangGraph, AutoGen, Microsoft Agent Framework, Lobster |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（GPT-OSS:120B 上平均准确率）

| 方法 | Average Accuracy (%) |
|------|------------------------|
| **Agent Q-Mix (Ours)** | **72.73** ✅ |
| GTD (Best prior adaptive) | 60.37 |
| AutoGen (Best commercial) | 68.60 |
| LangGraph | 56.42 |
| LLM-Debate | 55.46 |

> ➤ **提升 +12.36 pts vs. 最佳自适应方法，+4.13 pts vs. 最佳商业框架**

---

### 各任务表现亮点

| 任务 | Agent Q-Mix | 最佳基线 | 提升 |
|------|-----------|---------|-----|
| **HMMT (Math)** | 53.33% | 43.33% (GTD) | **+10.0 pts** |
| **Beyond-AIME** | 42.00% | 37.00% (AutoGen) | **+5.0 pts** |
| **MMLU-Pro (Reasoning)** | 92.86% | 88.57% (TopoDIM) | +4.29 pts |
| **LiveCodeBench** | 100.00% | 100.00% | 并列最优 |
| **HumanEval** | 97.56% | 96.95% (AutoGen) | +0.61 pts |

---

### 在 Gemini-3.1-Flash-Lite 上的表现
尽管 backbone 更弱，Agent Q-Mix 仍取得最强数学性能：
- AIME25: **40.00%**
- AIME26: **60.00%**
- HMMT: **50.00%**
- Beyond-AIME: **34.00%**
- 平均准确率达 **66.90%**，优于所有 baseline

---

### Token 效率对比（GPT-OSS:120B）

| 方法 | MMLU-Pro Tokens | Beyond-AIME Tokens |
|------|------------------|--------------------|
| **Agent Q-Mix** | **112K** ✅ | **708K** ✅ |
| 其他 multi-agent 方法 | 471K ~ 2.71M | 1.00M ~ 2.68M |
| 单 agent baseline (Lobster) | 97K | — |

> ➤ **通信开销仅为其他 multi-agent 方法的 1/4 到 1/24，同时保持更高准确率**

---

### 鲁棒性实验结果（MMLU-Pro 上注入 adversarial agent）

| 方法 | 正常准确率 | 攻击后准确率 | 下降 Δ |
|------|------------|--------------|--------|
| **Agent Q-Mix** | 92.86% | 90.00% | **-2.86** ✅ |
| LLM-Debate | 92.86% | 84.29% | -8.57 |
| AutoGen | 88.57% | 78.57% | -10.00 |
| GPTSwarm | 87.14% | 77.14% | -10.00 |

> ➤ 表明 Agent Q-Mix 能通过学习减少对不可靠 agent 的依赖，具备更强 fault tolerance

---

### 消融实验结果（Ablation Studies）

#### (a) Agent 数量影响（Gemini）
- 从 1→4 agents，Beyond-AIME 准确率从 18% → 38%
- 超过 4 人后收益趋缓，说明存在“边际协作效应”

#### (b) 训练数据量影响
- 仅需 **15 个训练样例/领域** 即可达接近上限性能（95.73%）
- 表明 QMIX policy 具备强泛化能力和 sample efficiency

#### (c) Reward 权重影响
- $ w_{\text{acc}} = 1.5 $ 时达到最佳 trade-off
- 更高的 accuracy 权重会略微增加 token 消耗，但收益递减

#### (d) 通信轮数影响
- 数学任务需要 T=3 达到峰值（96.95%）
- 编码/推理任务 T=2 已足够，符合任务复杂度直觉

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **去中心化的通信动作选择是可行且有效的**：通过将通信建模为 MARL 中的局部动作，Agent Q-Mix 成功实现了 agent-level 自主决策。
2. ✅ **QMIX 是适配该场景的理想框架**：其 monotonic mixing 结构天然满足 IGM 性质，支持 CTDE，使得训练与部署分离成为可能。
3. ✅ **Learned topology 显著提升性能与效率**：
   - 在最难的数学任务上带来 **+10 pts 提升**
   - 在简单任务上自动抑制通信，节省 **高达 96% 的 token 开销**
4. ✅ **具备强鲁棒性**：面对 adversarial agent，性能下降最小（仅 -2.86 pts），表明策略能动态隔离异常节点。
5. ✅ **跨 backbone 泛化能力强**：在同一套 policy 下，在 GPT-OSS 和 Gemini 上均表现优异。

---

### 方法的局限性
1. **动作空间固定为6类**：虽然已覆盖常见模式，但在更复杂协作场景中可能不够灵活（如引入工具调用、外部数据库访问等）。
2. **依赖预设 agent 角色**：agent 的 specialization 依赖人工 prompt 设计，尚未实现完全 end-to-end 的角色演化。
3. **通信图构建为 deterministic mapping**：从动作向量到邻接矩阵的映射 $ \phi(u_t) $ 是确定性的，未考虑不确定性传播。
4. **扩展至大规模 agent 团队仍有挑战**：目前最多测试到 10 个 agent，scalability 需进一步验证。

---

### 未来工作方向
1. **Dynamic Action Space**：允许 agent 在运行时创建新的通信模式或子群组（subgroup formation）。
2. **End-to-End Role Learning**：结合 emergent communication 与 role discovery，让 agent 自主演化身份。
3. **Hierarchical Topology Control**：引入 meta-controller 学习何时切换通信范式（如从 debate 切换到 broadcast）。
4. **Integration with Planning & Memory**：将 topology learning 与 long-term planning、memory retrieval 耦合，构建更完整的 agentic workflow。
5. **Real-world Deployment**：应用于软件开发 pipeline、科研辅助系统等真实场景，验证实用价值。

---

> 🔗 **代码开源地址**：[https://github.com/ericjiang18/Agent-Q-Mix](https://github.com/ericjiang18/Agent-Q-Mix)

</details>

---

### 8. [Reclaiming Idle CPU Cycles on Kubernetes: Sparse-Domain Multiplexing for Concurrent MPI-CFD Simulations](https://arxiv.org/abs/2604.00377)

**Authors**: Tianfang Xie  
**Category**: cs.DC  
**Published**: 2026-04-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.00377v1  

#### Abstract
When MPI-parallel simulations run on shared Kubernetes clusters, conventional CPU scheduling leaves the vast majority of provisioned cycles idle at synchronization barriers. This paper presents a multiplexing framework that reclaims this idle capacity by co-locating multiple simulations on the same ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Reclaiming Idle CPU Cycles on Kubernetes: Sparse-Domain Multiplexing for Concurrent MPI-CFD Simulations*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在共享的 **Kubernetes** 集群上运行 **MPI-parallel CFD**（计算流体力学）模拟时，传统的 CPU 调度策略会导致大量已分配的 CPU 周期在 **MPI 同步屏障**（如 `MPI_Allreduce`）处空闲。这种资源浪费直接影响云成本效率，尤其对于需要执行大量独立仿真的工程团队而言，**集群吞吐量**（simulations per dollar）远比单次仿真耗时更重要。

### 提出的新方法与思路
本文提出了一种名为 **Sparse-Domain Multiplexing** 的框架，通过以下机制回收闲置 CPU 容量：

- **PMPI-based duty-cycle profiling**：利用 PMPI 接口对每个 MPI rank 进行轻量级性能剖析，量化其在计算与通信/同步之间的 CPU 利用率（即 **duty cycle**），识别出“稀疏域”（sparse-domain）rank（如远场区域）存在高达 95% 的空闲时间。
- **Proportional CPU allocation**：基于 duty cycle 动态调整各 rank 的 CPU requests（例如从 1000mcpu 降至 67mcpu），采用 **requests-only（Burstable QoS）** 模式避免 CFS 带宽限制导致的严重性能退化。
- **Co-location of multiple simulations**：将多个 CFD 仿真共置于同一物理节点上，利用稀疏 rank 的空闲周期并发执行其他仿真的密集 rank，实现 CPU 多路复用。
- **Dynamic controller with In-Place Pod Vertical Scaling (KEP-1287)**：设计一个自动化控制器，完成从 profiling、动态调整 pod CPU 请求（无需重启）、部署新仿真到公平性监控的全流程。

### 相比现有方法的优势
| 维度 | 本文方法 | 现有方法 |
|------|--------|--------|
| **CPU 利用率** | 显著提升，回收 88% 的闲置容量 | 固定等量分配，利用率低 |
| **调度粒度** | 子任务级（per-rank）差异化调度 | 整体或均匀分配 |
| **弹性能力** | 支持运行时垂直伸缩（CPU） | 多数仅支持水平扩展或需重启 |
| **适用场景** | 特别适合高吞吐、参数扫描类 CFD 工作流 | 更关注单任务延迟优化 |
| **首次应用** | **首个将 KEP-1287 应用于运行中 MPI 工作负载的实践** | ARC-V 等仅用于内存 |

---

## 2. 核心实验方法和设置

### 数据集与仿真案例
- **CFD 案例**：NACA 0012 翼型在马赫数 0.72 下的可压缩稳态流动，使用 OpenFOAM 的 `rhoSimpleFoam` 求解器和 k-ω SST 湍流模型。
- **网格规模**：498,834 六面体单元（hexahedral cells）。
- **域分解方式**：
  - **Concentric decomposition**：手动按距翼型中心的距离划分为三个区域（近壁区 dense、中间 medium、远场 sparse），赋予不同权重（15, 5, 1），以显式制造负载不平衡。
  - **Scotch 分区**：标准图划分法作为对照，验证方法通用性。

### 实验设置
- **平台**：AWS EC2 上的 12 节点 **c5.2xlarge** 集群（每节点 8 vCPU，共 96 worker vCPU），使用 **k3s v1.35.0**。
- **容器镜像**：自定义 Docker 镜像，含 OpenFOAM 10、Open MPI 4.1 和 SSH。
- **MPI 隔离**：禁用共享内存传输层（`--mca btl tcp,self`），确保跨作业无干扰。
- **Pod 规格**：每个仿真由 16 个 worker pod + 1 launcher pod 构成，使用 **Burstable QoS**（仅设 CPU requests，无 limits）。

### 评估指标
| 指标 | 定义 |
|------|------|
| **Wall-clock time** | 单个仿真的实际运行时间（秒） |
| **Throughput gain** | $ \frac{N}{T_N} / \frac{1}{T_1} $，即单位时间内完成的仿真数量倍数 |
| **Per-case degradation** | 并发下平均单仿真的延长时间百分比 |
| **Scheduling efficiency** | $ \text{Throughput} / N $，衡量每增加一个并发任务的实际收益 |
| **Fairness (A/B ratio)** | 不同仿真间完成时间的比例，越接近 1 越公平 |
| **Cost per simulation** | 总云成本除以完成的仿真数 |

### 基线方法对比
| 配置 | 描述 |
|------|------|
| **C-1E** | 单仿真，所有 rank 均匀分配 1000mcpu（Equal allocation） |
| **C-1P** | 单仿真，按 duty cycle 比例分配 CPU（Proportional allocation） |
| **C-2E** | 双仿真，均使用等量分配 |
| **C-2P** | 双仿真，均使用比例分配（本文主推配置） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 配置 | 单仿真时间 (s) | 总耗时 (s) | 吞吐增益 | 成本节省 |
|------|----------------|------------|----------|----------|
| C-1P (baseline) | 1249 | 1249 | 1.00× | — |
| C-2P | 1331 / 1410 | 1410 | **1.77×** | **44%** |
| C-3P | ~1446 | 1446 | **2.59×** | 62% |
| C-4P | ~1604 | 1604 | **3.11×** | 68% |
| C-5P | ~1670 | 1670 | **3.74×** | 73% |

> 注：C-2P 中两仿真的 wall-clock 时间分别为 1331s 和 1410s，makespan 为 1410s。

### 与基线方法对比
- **吞吐量提升显著**：双仿真仅带来约 10–13% 的额外延迟，即可实现 **1.77× 吞吐增益**。
- **比例分配更高效**：虽然 C-2E（等量）略快于 C-2P（1286s vs 1410s），但其消耗的 CPU request 是后者的 2.7 倍（32vCPU vs 12vCPU），限制了更高密度共置（如 N=5 时可能超出 ResourceQuota）。
- **公平性良好**：所有配置下的 A/B 时间比均 < 1.10，表明调度未造成严重偏斜。

### 消融实验与建模分析
- **duty cycle 测量结果**：
  - 稀疏 rank（weight 1）平均 duty cycle：**5.0%**
  - 中等 rank（weight 5）：**11.5%**
  - 密集 rank（weight 15）：**19.4%**
  - 所有 rank 平均空闲率达 **80–95%**，其中 MPI 同步占主导。
- **理论回收容量**：基于公式计算，总可回收 CPU 容量达 **88%**。
- **Throughput Prediction Model**：
  - 提出线性竞争模型：$ T_N = T_1 (1 + \beta(p_N - p_1)) $
  - 单参数 $ \beta = 0.524 $ 可拟合 N=1…5 所有数据，预测误差 **±4% 内**。
  - 仅用 N=2 数据训练模型也能保守估计更高 N 的性能（略微高估延迟），适用于容量规划。

### 动态控制器表现
- **全自动流程**：从单仿真开始，自动完成 profiling → resize → 部署新仿真 → 监控公平性。
- **性能结果**：4 个并发仿真，总耗时 **1537s**，吞吐达 **3.25×**，优于静态 C-4P（3.11×）。
- **操作细节**：
  - 64 次 **in-place pod resize**（零重启）
  - 4 轮 profiling
  - 总自动化耗时 53 分钟
- **意义**：**首次实现对运行中 MPI 工作负载的 in-place CPU vertical scaling**。

---

## 4. 关键结论和发现

### 主要发现
1. **MPI-CFD 在 Kubernetes 上存在巨大 CPU 浪费**：即使是最密集的 near-wall rank，也有超过 80% 的时间处于 MPI 同步等待状态。
2. **Sparse-Domain Multiplexing 可有效回收空闲周期**：通过比例分配 + 共置，可在几乎不增加硬件成本的前提下，将集群吞吐量提升至 **3.74×（N=5）**。
3. **Pareto 最优在 N=3**：N=3 时调度效率达 **86%**，是吞吐与延迟之间的最佳权衡点（“knee of the curve”）。
4. **方法具有普适性**：在标准 **Scotch 分区** 下同样观察到 87–92% 的空闲率，说明负载不均衡并非特定分区所致。
5. **动态控制可行且高效**：结合 KEP-1287 实现零重启的全自动多路复用，达到 **3.25× 吞吐**，验证了生产级自动化的潜力。

### 方法的局限性
- **依赖低 duty cycle**：当 duty cycle 接近 50% 或更高时（如复杂化学反应模拟），空闲窗口减少，multiplexing 增益下降。
- **内存带宽潜在瓶颈**：虽当前实验未达饱和，但在高密度共置下（N≥6），memory bandwidth 可能成为新的竞争源。
- **尚未处理故障场景**：动态控制器目前为 PoC 级别，未涵盖 kubelet 压力、resize 失败、部分部署失败等情况。
- **单一 NUMA 架构**：实验未涉及 NUMA-aware 调度，在更大规模或多 NUMA 场景下需进一步优化。

### 未来工作方向
1. **异构工作负载配对**（Heterogeneous workload pairing）：
   - 利用 Kubernetes pod affinity 将“稀疏”与“密集”仿真有意搭配，最大化 CPU 利用率。
2. **燃烧化学加速**（Combustion chemistry acceleration）：
   - 结合神经网络代理模型处理非刚性单元，降低局部计算强度，进一步提升 packing 效率。
3. **增强控制器鲁棒性**：
   - 引入容错机制、重试逻辑、资源压力响应，向生产环境迈进。
4. **集成高级调度器**：
   - 与 Volcano、Kueue 等集群级调度器协同，实现跨作业队列管理与 intra-job rank-level 多路复用的联合优化。

---

> ✅ **代码与数据公开**：所有实验脚本、Kubernetes manifest、PMPI 库、动态控制器及分析代码均已开源：  
> 🔗 [https://github.com/Xieldor/K8s-CFD-Multiplexing](https://github.com/Xieldor/K8s-CFD-Multiplexing)

</details>

---

### 9. [SAGE: Subsurface AI-driven Geostatistical Extraction with proxy posterior](https://arxiv.org/abs/2604.00307)

**Authors**: Huseyin Tuna Erdinc, Ipsita Bhar, Rafael Orozco, Thales Souza, Felix J. Herrmann  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.00307v1  

#### Abstract
Recent advances in generative networks have enabled new approaches to subsurface velocity model synthesis, offering a compelling alternative to traditional methods such as Full Waveform Inversion. However, these approaches predominantly rely on the availability of large-scale datasets of high-qualit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SAGE: Subsurface AI-driven Geostatistical Extraction with proxy posterior》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统地震反演方法（如 **Full Waveform Inversion, FWI**）依赖高质量、高分辨率的地下速度模型训练数据，但在实际应用中，这类完整且地质合理的速度模型（velocity models）极为稀缺。此外，现有基于生成网络的方法通常需要大量全观测的速度样本作为先验，难以在稀疏观测条件下（如仅有少量井数据 well logs 和迁移地震图像）进行有效建模。

SAGE 针对这一**数据稀缺与不完全观测**的现实挑战，提出了一种无需完整速度模型即可学习地质统计特性的新框架。

---

### 🚀 提出的新方法与核心思想
SAGE（Subsurface AI-driven Geostatistical Extraction）是一种基于**代理后验（proxy posterior）学习**的生成框架，其核心创新包括：

1. **从不完全观测中学习代理后验分布**  
   利用仅有的稀疏 **well-log 测量**（列向掩码观测）和对应的 **migrated seismic images**（如 RTM 图像），训练一个条件生成模型来近似 $ p(x|y) $，即速度场 $ x $ 在给定迁移图像 $ y $ 下的后验分布。

2. **推理阶段仅需迁移图像即可生成全分辨率速度场**  
   训练完成后，SAGE 可以仅以 migrated image 为输入，生成符合地质规律的高分辨率速度实现（realizations），而井信息已被隐式编码到学习到的分布中。

3. **结合 Simulation-Based Inference (SBI) 与 Score-Based Diffusion Models**  
   使用基于分数的扩散网络（score-based diffusion network）作为生成器，并引入**双重掩码机制**（original mask + subsampling mask）防止模型退化为简单的插值器。

4. **支持下游任务的数据增强能力**  
   SAGE 生成的合成速度样本可用于训练其他任务专用网络（如 WISE），缓解真实数据不足问题。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法 / 现有生成模型 | SAGE |
|------|------------------------|------|
| 数据需求 | 需要大量完整速度模型（full $ x $） | 仅需部分观测（masked $ x_{obs} $）+ migrated image $ y $ |
| 实际适用性 | 在现场数据中受限于缺乏真值模型 | 更贴近工业实践，适用于真实稀疏井场景 |
| 推理效率 | 多数为逐次优化或采样 | amortized inference，快速生成 |
| 下游支持 | 通常独立使用 | 可作为 prior generator 支持 WISE 等 inversion 框架 |

> ✅ **优势总结**：SAGE 实现了“用不完整的监督信号学习地质一致性先验”，是迈向**自监督、数据高效型地球物理建模**的重要一步。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

#### （1）合成数据集（Synthetic Dataset）
- 来源：从 **3D Compass 模型**切片得到的 2D 地球声速模型
- 规模：1000 个 velocity realizations
- 分辨率：256 × 512 网格，空间步长 12.5 米（覆盖 3.2km × 6.4km）
- 正演模拟：
  - 16 个震源，256 个接收器
  - Ricker wavelet（主频 20Hz），记录时长 3.2 秒
  - 添加 10dB 彩色高斯噪声
- 成像方式：Reverse-Time Migration (RTM)，背景模型为高斯平滑版本
- 工具：使用 **JUDI** 进行波场模拟与成像

> ⚠️ 注意：训练时不使用完整速度模型，而是将每个模型保留 **5/256 列作为 well-log 观测**（约 99% 被遮蔽）

#### （2）真实数据集（Field Data）
- 来源：英国国家数据仓库（**UK National Data Repository**）
- 预处理流程：
  - Checkshot 时间-深度转换
  - Well tie 对齐
  - 提取准二维剖面（quasi-2D line）
  - 动学校正一致的下采样
- 井数量：共 40 口井（非常稀疏）
- 策略：采用 **预训练 + 微调（fine-tuning）** 策略，将在 Compass 上训练的模型迁移到真实数据

---

### 🧪 实验设置与评估指标

| 设置项 | 内容 |
|-------|------|
| 网络架构 | U-Net 作为 denoising network |
| 输入 | noisy partial velocity + RTM image + mask + noise level |
| 训练目标 | 基于 denoising score matching 的条件扩散损失 |
| 掩码策略 | 引入二级 subsampling mask 防止 trivial solution |
| 训练时间 | 约 20 GPU 小时 |
| 推理方式 | 通过反向 SDE 采样生成多个 realization |

#### 评估指标
- **SSIM**（Structural Similarity Index Measure）：衡量重建结构相似性
- **Posterior Mean vs Ground Truth**：比较均值估计质量
- **Posterior Standard Deviation**：不确定性量化能力
- **Trace Comparison**：沿井位置的速度曲线对比
- **视觉地质合理性**：是否反映复杂构造特征

---

### 🔁 基线方法对比
论文未直接与其他端到端生成模型对比（因多数要求完整 $ x $），但通过以下方式体现优越性：

1. **与 ground truth 比较**：验证 SAGE 在 synthetic 数据上的重建精度
2. **用于训练 WISE 的替代效果**：比较用 SAGE 生成数据 vs 真实数据训练 WISE 的性能差异
3. **消融设计本身构成对比**：如无 subsampling mask 是否导致崩溃

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

| 实验场景 | 性能表现 |
|---------|----------|
| **SAGE 在 synthetic 数据上的 posterior mean SSIM** | **0.82 – 0.84**（图2、图3） |
| **与 ground truth 的 trace 匹配程度** | 在保留井位置高度一致，细节略有平滑 |
| **不确定性地图（posterior std）** | 高方差区域对应地质复杂区（如断层带），具有解释性 |
| **用于训练 WISE 的 SAGE 样本效果** | WISE 后验均值 SSIM 达 **0.84**（相比用真实数据训练的 0.88）<br>→ 性能下降仅 **~4.5%**，表明 SAGE 样本具备高质量先验信息 |
| **真实数据推理结果（图4）** | 生成速度场呈现合理地质结构，与 migrated image 特征对齐<br>并与独立井数据吻合良好 |

---

### 🔍 消融实验分析（隐含于方法设计）

虽然未明确列出 ablation table，但从方法描述可推知关键组件作用：

| 组件 | 作用 | 若缺失后果 |
|------|------|------------|
| **Subsampling Mask ($\tilde{A}$)** | 防止模型退化为 $ D_\theta(\cdot) = A^+x_{obs} $ | 模型只恢复已知列，忽略 RTM 条件信息 |
| **双模态输入（$x_{obs}, y$）联合训练** | 实现 inpainting + conditioning 联合建模 | 无法泛化至仅靠 $y$ 推理 |
| **Score-based Diffusion Framework** | 支持多模态采样与不确定性建模 | 限制为单一确定性输出 |

> 💡 实验表明：移除 subsampling mask 会导致训练失败或过拟合局部观测。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **可以从不完全观测中学习有效的地质代理后验**  
   SAGE 成功证明：即使没有完整的速度模型，也能利用 paired RTM images 与 sparse well logs 学习出统计一致、地质合理的速度分布。

2. **推理过程摆脱对井数据的依赖**  
   一旦训练完成，SAGE 可仅凭 migrated image 输出全分辨率速度场，极大提升部署灵活性。

3. **生成样本可用于训练下游 inversion 网络（如 WISE）**  
   SAGE 充当“数据引擎”，解决了 WISE 类方法在真实场景中因缺乏训练数据而无法使用的瓶颈。

4. **不确定性估计具有地质意义**  
   posterior standard deviation 能识别结构复杂区域，提供可信度指引。

5. **在真实数据上初步验证可行**  
   即使仅有 40 口井，通过 fine-tuning 仍能生成合理结果，展示了强泛化能力。

---

### ⚠️ 局限性

| 局限 | 说明 |
|------|------|
| **依赖高质量 migrated image** | 若 RTM 图像存在严重 artifacts 或照明不足，会影响生成质量 |
| **当前为 2D 框架** | 实际应用多为 3D，扩展至三维尚需验证 |
| **平滑效应** | 对细小地层结构有一定模糊倾向，源于信息缺失下的贝叶斯平均行为 |
| **未建模岩性或沉积环境先验** | 完全依赖数据驱动，缺乏显式地质规则嵌入 |

---

### 🔮 未来工作方向

1. **扩展至 3D setting**  
   将 SAGE 推广至三维体积处理，更贴合实际勘探需求。

2. **融合更多观测模态**  
   如加入 gravity/magnetic data、CIGs 或 impedance profiles，进一步约束生成空间。

3. **集成物理约束（Physics-Informed Learning）**  
   在扩散过程中引入波动方程残差等物理损失，提高生成模型的物理一致性。

4. **构建大规模 field-scale 数据集**  
   推动建立公开 benchmark（类似 GeoFWI [15]），促进社区发展。

5. **闭环反馈用于 FWI 初始化**  
   将 SAGE 生成的 ensemble 作为 FWI 的初始模型集合，提升收敛稳定性。

---

> 🔗 **代码开源地址**：[https://github.com/slimgroup/SAGE](https://github.com/slimgroup/SAGE)  
> 📄 **相关工作联动**：SAGE 与 WISE 构成“先验学习 → 不确定性反演”完整链条，代表新一代 AI-driven seismic inversion 范式。

</details>

---

### 10. [LLM REgression with a Latent Iterative State Head](https://arxiv.org/abs/2604.01206)

**Authors**: Yiheng Su, Matthew Lease  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.01206v1  

#### Abstract
We present RELISH (REgression with a Latent Iterative State Head), a novel, lightweight architecture designed for text regression with large language models. Rather than decoding numeric targets as text or aggregating multiple generated outputs, RELISH predicts scalar values directly from frozen LLM...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LLM REgression with a Latent Iterative State Head (RELISH)

## 1. 论文的主要贡献和创新点

### 解决的问题
在大型语言模型（LLM）主导的自然语言处理时代，大多数任务被统一为“文本到文本”（text-to-text）范式。然而，**回归任务**（regression）——如语义相似度评分、机器翻译质量估计等——其目标是预测连续的标量值，而非生成文本。传统的基于 LLM 的回归方法存在以下问题：

- **Autoregressive Decoding**（自回归解码）：将数值作为文本生成，依赖于 token 级别的交叉熵损失，无法直接优化数值误差（如 MSE），对数值邻近性不敏感。
- **Regression-aware Inference**（回归感知推理）：通过贝叶斯最优决策规则（如后验均值）来改善预测，但需要多次采样或枚举候选值，计算成本高。
- **Predictive Head**（预测头）：直接从 LLM 的隐藏状态预测数值，效率高，但通常使用简单的池化操作（如 mean-pooling 或 [CLS] token），可能丢失细粒度的回归相关信息。

### 提出的新方法
本文提出了 **RELISH**（**RE**gression with a **L**atent **I**terative **S**tate **H**ead），一种专为 LLM 回归设计的轻量级、表达能力强的预测头架构。

- **核心思想**：不采用静态池化，而是通过一个**迭代细化机制**（iterative refinement mechanism）来动态地从 token-level 表示中提取信息。
- **具体实现**：
  1. 维持一个可学习的**潜在状态**（latent state）`r[0]`。
  2. 在 `L` 个步骤中，通过 **cross-attention** 机制，让该潜在状态反复与 LLM 输出的所有 token 表示进行交互，从而逐步提炼出最相关的特征。
  3. 最终，将经过 `L` 次迭代后的潜在状态 `r[L]` 输入一个线性回归器，得到最终的标量预测。

### 相比现有方法的优势
- **高性能**：在多个数据集和 LLM 上，性能显著优于所有三大类基线方法。
- **高效率**：推理过程为单次前向传播（single-pass），避免了多轮采样的高开销。
- **参数高效**（Parameter-efficient）：仅需约 3.4-3.7M 可训练参数（占 LLM 总参数的 0.01%-0.04%），远低于 LoRA 等微调方法（0.26%-0.42%）。
- **架构优势**：相比静态池化，迭代的 cross-attention 能更鲁棒地捕捉和整合分布式的、局部的回归相关信号。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖了两个经典的文本回归任务，共 **5 个数据集**：
- **语义文本相似度**（Semantic Textual Similarity, STS）：
  - **STS-Benchmark (STS-B)**：英文句子对，评分范围 [0, 5]。
  - **SICKR-STS**：来自 SICK 数据集的语义相关性子集，评分范围 [1, 5]。
- **机器翻译质量估计**（Machine Translation Quality Estimation, WMT）：
  - **WMT_EN_ZH**：英译中，评分范围 [0, 100]。
  - **WMT_RU_EN**：俄译英，评分范围 [0, 100]。
  - **WMT_SI_EN**：僧伽罗语译英，评分范围 [0, 100]。

### 实验设置和评估指标
- **LLM Backbones**：使用了 **4 个不同的 LLM** 进行评估：
  - Llama 3.1 8B Instruct
  - Qwen3 8B
  - Qwen3 32B
  - Gemma 3 27B Instruct
- **训练设置**：评估了两种模式：
  - **Frozen**：仅训练 RELISH 模块，LLM 主干冻结。
  - **Fine-tuned (LoRA/RAFT)**：使用 LoRA 或 RAFT 对 LLM 主干进行微调。
- **评估指标**：
  - **Pearson Correlation (r)**：衡量线性相关性，越高越好。
  - **Spearman Correlation (ρ)**：衡量排序一致性，越高越好。
  - **Range-Normalized RMSE (NRMSE)**：将原始 RMSE 归一化到数据集的得分范围内，越低越好，便于跨数据集比较。

### 基线方法对比
与三大类 LLM 回归方法的代表基线进行了全面对比：
- **Autoregressive Decoding**：
  - Zero-shot Prompting
  - Many-shot Prompting (128 个示例)
- **Regression-aware Inference**：
  - RAIL (Regression-aware inference with LLMs)
  - RAFT (Regression-aware fine-tuning)
- **Predictive Head**：
  - Linear Regression (Lin.)
  - Multi-Layer Perceptron (MLP)，其隐藏层大小与 RELISH 的参数量匹配以保证公平性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在 **Table 3** 中，对 5 个数据集、4 个 LLM 和 3 次随机种子的结果进行了宏平均（macro-averaged），结果如下：

| 方法 | Pearson (r) ↑ | Spearman (ρ) ↑ | NRMSE ↓ |
| :--- | :--- | :--- | :--- |
| **RELISH (Ours)** | **76.3 ± 0.2** | **74.0 ± 0.1** | **13.3 ± 0.3** |
| RAFT (2nd best) | 71.5 ± 0.4 | 69.0 ± 0.2 | 16.6 ± 0.3 |
| LoRA + MLP | 64.8 ± 0.1 | 61.7 ± 0.1 | 16.7 ± 0.1 |

- RELISH 相对于第二好的方法 RAFT，在 Pearson、Spearman 和 NRMSE 上分别提升了 **6.7%**、**7.2%** 和 **19.9%**。

### 与基线方法的对比结果
- **全面领先**：RELISH 在几乎所有数据集和 LLM 组合上都取得了最佳性能（见 Table 14-18）。
- **超越复杂方法**：即使与需要大量计算资源的 RAFT 相比，RELISH 依然表现更优。
- **优于简单方法**：显著优于零样本提示（zero-shot prompting）和简单的线性/MLP 预测头。

### 消融实验结果
#### (1) 损失函数的影响 (S E.1)
- 对比了使用 **Huber Loss** 和 **MSE Loss** 训练的 RELISH。
- 结果显示，两种损失函数下的 RELISH 性能都非常好，且非常接近。
- **结论**：RELISH 的性能增益主要来自于其**架构设计**，而非特定的损失函数选择。

#### (2) 迭代深度的影响 (S E.2)
- 系统地改变了迭代次数 `L`（从 1 到 5）。
- `L=1`（即单步 attention-pooling）已是一个强基线，但性能不如完整的 RELISH。
- 当 `L≥2` 时，性能有显著提升，并在 `L=3` 左右达到稳定，验证了**迭代细化**的有效性。
- 不同 LLM 的最优 `L` 可能不同，例如 Gemma 3 27B 在 `L=5` 时仍在提升。

---

## 4. 关键结论和发现

### 主要发现
1. **架构设计是关键**：RELISH 的成功证明了，为回归任务设计一个专门的、能够动态聚合 token 信息的预测头，比简单地复用生成式 LLM 的解码能力或使用静态池化要有效得多。
2. **性能与效率兼得**：RELISH 同时实现了**最先进的性能**和**极高的参数/计算效率**，打破了“高性能必须高开销”的固有印象。
3. **对任务敏感**：RELISH 在机器翻译质量估计（WMT）任务上的提升尤为显著，因为这些任务更依赖于捕捉输入序列中的局部错误和细微差异，而这正是其迭代注意力机制的强项。

### 方法的局限性
- **依赖预训练表示**：RELISH 在冻结的 LLM 主干上运行，其性能受限于主干模型本身编码的知识质量。对于需要全新领域知识的任务，可能仍需微调主干。
- **序列长度扩展性**：其基于 cross-attention 的机制，计算复杂度随输入序列长度线性增长，对于超长文本（如整篇论文）可能成为瓶颈。
- **任务范围有限**：目前评估集中在成对文本输入的回归任务（如相似度、质量估计）。对于单句情感分析或价格预测等任务，其泛化能力有待验证。

### 未来工作方向
- 将 RELISH 扩展到更多类型的回归任务，如情感分析（预测连续评分）、产品价格预测等。
- 探索如何将其应用于**不确定性量化**（Uncertainty Quantification），例如用分位数回归头替代线性回归头。
- 研究其在 **Reward Modeling** 和 **LLM-as-a-judge** 等需要标量打分的应用场景中的潜力。

</details>

---

### 11. [Speeding Up Mixed-Integer Programming Solvers with Sparse Learning for Branching](https://arxiv.org/abs/2604.00094)

**Authors**: Selin Bayramo\u{g}lu, George L Nemhauser, Nikolaos V Sahinidis  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.00094v1  

#### Abstract
Machine learning is increasingly used to improve decisions within branch-and-bound algorithms for mixed-integer programming. Many existing approaches rely on deep learning, which often requires very large training datasets and substantial computational resources for both training and deployment, typ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Speeding Up Mixed-Integer Programming Solvers with Sparse Learning for Branching*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文聚焦于**混合整数规划（MIP）求解器中的分支策略（branching）效率问题**。传统高效的分支规则如 **strong branching (SB)** 虽能生成较小的搜索树，但计算代价极高。现有基于深度学习的方法（如 GNN）虽可近似 SB 行为，但依赖大规模训练数据、GPU 加速和复杂部署，难以在资源受限场景下实用。

### 提出的新方法与新思路
作者提出了一种**基于稀疏学习（sparse learning）的轻量级模型**，用于预测 strong branching 分数，从而指导分支决策。其核心思想是：
- 构建**稀疏的线性或二次回归模型**（sparse linear/quadratic models），通过 **lasso 正则化** 自动选择最具解释性的特征组合。
- 使用**动态更新的变量级特征**（而非静态图结构），并引入**二次项交互特征**以增强表达能力。
- 模型完全在 **CPU 上训练和部署**，不依赖 GPU 或深度学习框架。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **参数量** | 模型参数不足 state-of-the-art GNN 的 **4%**（见 Table 2）。 |
| **计算效率** | 在 CPU 上运行时，推理速度显著快于默认 SCIP 和 GNN-CPU/GPU 方法。 |
| **数据效率** | 在“小样本”设置下仅需约 25,000 个候选观测即可达到良好性能，远低于 GNN 所需的大规模数据收集成本。 |
| **部署简易性** | 模型简单、可解释性强，易于集成到现代 MIP 求解器（如 SCIP）中，无需专用硬件。 |

---

## 2. 核心实验方法和设置

### 数据集
在四个经典的 NP-hard 问题上进行实验：
- **Set Covering (SC)**
- **Combinatorial Auctions (CA)**
- **Capacitated Facility Location (FL)**
- **Maximum Independent Set (IS)**

这些问题是学习型分支研究的标准基准（源自 Gasse et al., 2019），每个问题生成：
- 10,000 个训练实例
- 各 2,000 个验证与测试实例
- 额外构建不同规模（small/medium/large）的评估实例集用于性能比较

### 实验设置与评估指标
#### 设置
- 使用 **SCIP 8.0.0 + SoPlex 6.0.0** 作为底层求解器
- 所有运行限制为 **单线程、1 小时时限**
- 关闭非根节点割平面生成，禁用预处理重启
- GNN 推理分别在 **CPU (GNN-C)** 和 **NVIDIA Tesla V100 GPU (GNN-G)** 上执行

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Solved** | 在时限内成功求解的问题数量 |
| **Time** | 1-shifted geometric mean CPU 时间（秒） |
| **Nodes** | 1-shifted geometric mean 分支定界节点数 |
| **Accuracy** | 预测最高分候选变量与 SB 一致的比例（node-level top-1 匹配率） |

### 基线方法对比
| 方法 | 类型 | 说明 |
|------|------|------|
| **SCIP** | 默认策略 | Hybrid branching（可靠性伪成本分支），现代求解器默认配置 |
| **VFS** | 黄金标准 | Vanilla Full Strong Branching，作为学习目标 |
| **GNN-C / GNN-G** | SOTA ML 方法 | 图神经网络模型，在 CPU/GPU 上运行，代表当前最优 ML 分支方法 |
| **QL** | 本文方法（大样本） | 基于大量数据训练的稀疏二次模型 |
| **QS** | 本文方法（小样本） | 基于少量数据、按问题规模定制训练的小样本模型 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 4）

| 方法 | 平均 Time ↓ | 平均 Nodes ↓ | Solved ↑ |
|------|-------------|--------------|----------|
| **SCIP** | 基准 | 基准 | 基准 |
| **GNN-C** | 多数情况下慢于 SCIP | 节点更少 | 相当或略差 |
| **GNN-G** | 快于 GNN-C，部分优于 SCIP | 最少节点之一 | 中等 |
| **QL / QS** | **全面优于 SCIP 和 GNN-C**<br>在大型 CA 上提速 **25%+** | 节点数接近 GNN，远少于 SCIP | **持平或更好** |

> ✅ 特别地，在 **large CA 和 large IS** 问题上，**QS 模型解决了更多问题且运行更快**，表明小样本+规模匹配训练的有效性。

### 与基线方法的对比结果
- **相比 SCIP**：所有稀疏模型（QL/QS）在平均运行时间上**持续超越默认求解器**，即使在小样本训练下也表现稳健。
- **相比 GNN**：
  - GNN 在 **accuracy 和 node count** 上略优（见 Table 3），因其更强的拟合能力；
  - 但在实际运行时间上，**QL/QS 显著快于 GNN-C，甚至快于 GNN-G**，说明低延迟推理带来的收益超过了精度损失。
- **内存开销**：
  - 稀疏模型仅比 SCIP 多消耗 **~10% 内存**
  - GNN 运行则需要 **5–15 倍于 SCIP 的内存**

### 消融实验结果

#### （1）模型大小影响（Table 5）
- 限制模型非零系数数量（dfmax=25~1000）会导致性能下降，尤其在大问题上。
- 即使将 QL 模型压缩至 ~1000 参数，仍保持较强竞争力，验证了**稀疏性与性能之间的良好平衡**。

#### （2）线性 vs 二次特征（Table 6）
| 问题 | QL（含二次项） | LL（仅线性） | 差距 |
|------|----------------|-------------|------|
| SC-large | 69 solved, 1196s | 45 solved, 1915s | ❌ 显著劣化 |
| IS-medium | 100 solved, 8.8s | 100 solved, 9.3s | ⚠️ 可接受但变慢 |
| IS-large | 69 solved, 1196s | 45 solved, 1915s | ❌ 严重退化 |

> 🔍 结论：**二次特征扩展对中大型问题至关重要**，尤其是在 SC 和 IS 上，线性模型不足以捕捉复杂的分支行为。

---

## 4. 关键结论和发现

### 主要发现
1. **简单有效的替代方案存在**：  
   即使是非常简单的稀疏二次模型，也能在预测 strong branching 行为方面取得与复杂 GNN 相当的效果，并在端到端求解时间上实现反超。

2. **效率优于精度**：  
   尽管 GNN 在 accuracy 上更高、生成的搜索树更小，但其高昂的推理成本使其整体运行时间落后于轻量级稀疏模型。这揭示了一个重要权衡：**快速而足够好的分支决策 > 缓慢而精确的决策**。

3. **小样本训练可行且有效**：  
   仅需约 25,000 个候选观测即可训练出高性能模型，且支持按问题规模分别训练（size-matched training），极大提升了在现实应用中的实用性。

4. **可解释性增强信任与调试能力**：  
   模型中保留的关键特征（如 `ceiling distance × floor distance`, `solution value²` 等）具有明确语义，有助于理解 strong branching 的内在机制。

### 方法的局限性
- 当前方法仍专注于模仿 **strong branching**，未探索更广义的分支策略空间（如 general disjunctions）。
- 虽然在四大标准问题上表现优异，但在高度异构或工业级 MIP 实例上的泛化能力有待进一步验证。
- 依赖手工设计的特征工程，尚未完全实现端到端的学习范式。

### 未来工作方向
- 将稀疏建模思想推广至其他 B&B 组件（如 node selection, cutting plane selection）。
- 探索结合强化学习（RL）的目标驱动训练方式，摆脱对 SB 专家策略的依赖。
- 开发自动特征构造与选择机制，减少人工干预。
- 在真实工业应用场景中部署并评估模型的实际效益。

---

> 📌 **总结一句话**：  
> 本论文证明了——**通过稀疏学习构建的小型、高效、可解释的分支模型，可以在不牺牲性能的前提下，显著加速 MIP 求解过程，尤其适合无 GPU 支持、数据有限的真实世界部署环境**。

</details>

---

### 12. [TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models](https://arxiv.org/abs/2604.00666)

**Authors**: Lingjie Chen, Ruizhong Qiu, Yuyu Fan, Yanjun Zhao, Hanghang Tong  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.00666v1  

#### Abstract
Diffusion language models (DLMs) offer a promising path toward low-latency generation through parallel decoding, but their practical efficiency depends heavily on the decoding trajectory. In practice, this advantage often fails to fully materialize because standard training does not provide explicit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
**Train-Inference Mismatch in DLMs**  
扩散语言模型（Diffusion Language Models, DLMs）理论上支持并行解码，从而实现低延迟生成。然而，在实践中，标准的训练方式（如均匀随机掩码策略）并未对 token 的揭示顺序（decoding trajectory）提供显式监督，导致训练时与推理时存在严重不匹配（train-inference mismatch）。这使得模型在推理阶段倾向于退化为类似自回归（AR）的行为，无法充分利用并行能力。

此外，已有的一些轨迹优化方法（如基于 DLM 自身采样的蒸馏方法）虽然有效，但依赖昂贵的 diffusion sampling 过程，计算开销巨大，难以实用。

---

### 🚀 提出的新方法：TRIMS
作者提出 **Trajectory-Ranked Instruction Masked Supervision (TRIMS)**，一种轻量级、高效的监督微调框架，用于增强 DLM 的解码轨迹学习。

#### 核心思想：
- 利用一个**轻量级的自回归教师模型（AR teacher）** 在单次前向传播中估计每个 token 的预测难度（difficulty），通过 $-\log p(y_i | x, y_{<i})$ 得到 NLL 分数。
- 将这些难度分数离散化为 $K$ 个量化桶（quantile-based buckets），形成“难→易”的优先级排序。
- 设计一种**轨迹感知的掩码策略（trajectory-aware masking）**：在训练中模拟“先解决难词、再补全简单词”的硬到易（hard-to-easy）解码过程。

#### 具体流程（见图2）：
1. **Trajectory Signals from AR Teachers**：用 AR 模型打分，无需 rollout 或复杂采样。
2. **Difficulty Bucketing**：将 token 按难度划分为有序桶，保证分布均衡。
3. **Trajectory-aware Masking**：动态选择 bucket 阈值 $k$，决定哪些 token 视为上下文（低桶号）、哪些保持 masked（高桶号），并设置不同掩码概率（$p_{\text{future}}=0.95$, $p_{\text{context}}=0.05$）。

---

### 🔍 相比现有方法的优势
| 方面 | TRIMS | 蒸馏类方法（如 dParallel, d3LLM） |
|------|-------|-------------------------------|
| **训练成本** | 极低（仅需一次 AR 前向 + 3小时 8xA100） | 极高（需大量 DLM 采样构建数据集） |
| **数据需求** | 仅 1K 样本 | 高达 93K 自生成样本 |
| **通用性** | 可插拔于标准 MDLM 流程 | 依赖特定采样策略和后处理 |
| **效率 vs 性能平衡** | 几乎零额外开销，性能媲美蒸馏方法 | 高性能但资源消耗大 |

> 💡 TRIMS 实现了“以极低成本注入有效轨迹监督”，是首个完全避免 DLM 采样的轨迹优化 SFT 方法。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **训练数据**：`s1K` 数据集（Muennighoff et al., 2025），包含约 1K 条指令微调样本，侧重推理任务。
- **评估基准**（四大主流任务）：
  - **GSM8K**：数学应用题
  - **MATH**：竞赛级数学问题
  - **HumanEval**：代码生成
  - **MBPP**：Python 编程任务

> 所有任务均具有较长输出序列，适合测试并行解码能力。

---

### ⚙️ 实验设置
- **基础模型**：
  - `LLaDA-Instruct`（8B）
  - `Dream-Instruct`（7B）
- **训练细节**：
  - 序列长度：1024
  - Batch size: 32（梯度累积 factor=4）
  - 学习率：1e-4，cosine schedule，warmup ratio=0.03
  - 使用 LoRA（rank=32）进行参数高效微调
  - 训练平台：8×NVIDIA A100 80GB，DeepSpeed ZeRO-2
  - 总训练时间：约 3 小时
- **AR 教师模型**：`Qwen3-8B`

---

### 🎯 评估指标
- **Accuracy**：任务正确率（pass@1）
- **Parallelism**：**TPS (Tokens Per Step)** —— 每一步成功预测的非掩码 token 数量，越高表示并行性越好
- **Trade-off Curve**：准确率 vs TPS 曲线，越靠右上角越好

---

### 🆚 基线方法对比
分为四类：
1. **原始 DLM 模型**：
   - LLaDA / Dream（无加速）
2. **Train-Free 加速方法**：
   - Fast-dLLM（启用 KV Cache 和并行解码）
3. **Training-Based 方法**：
   - Fast-dLLM-v2, D2F
4. **Distillation-Based 方法（高成本）**：
   - dParallel, d3LLM（依赖 DLM 采样生成伪轨迹数据集）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1 & 2）

#### 在 **LLaDA-Instruct** 上的结果：
| Method | GSM8K Acc | TPS | MATH Acc | TPS |
|--------|-----------|-----|----------|-----|
| LLaDA | 72.6 | 1.00 | 32.2 | 1.00 |
| Fast-dLLM | 74.7 | 2.77 | 30.8 | 1.97 |
| d3LLM | 73.1 | **9.11** | 30.4 | **5.74** |
| **TRIMS (Ours)** | **74.9** | **6.26** | **34.3** | **4.72** |

> ✅ TRIMS 在保持更高准确率的同时，达到远超 baseline 的并行性（TPS ↑6倍）

#### 在 **Dream-Instruct** 上的结果：
| Method | MBPP Acc | TPS | HumanEval Acc | TPS |
|--------|----------|-----|----------------|-----|
| Dream | 57.2 | 1.00 | 55.2 | 1.00 |
| Fast-dLLM-v2 | 50.1 | 2.04 | 61.7 | 2.58 |
| d3LLM | 55.6 | 2.96 | 57.1 | 3.20 |
| **TRIMS (Ours)** | **56.6** | **6.31** | 57.3 | 2.21 |

> ✅ TRIMS 在 MBPP 上实现高达 **6.31 TPS**，相较 train-free 方法提升近 **3× 并行性**

---

### 🔬 消融实验结果（Ablation Studies）

#### （1）是否引入轨迹监督？
- 对比标准 MDLM 训练 vs TRIMS → TRIMS 显著优于所有基准，尤其在编码任务（HumanEval, MBPP）上同时提升 accuracy 和 TPS。

#### （2）不同的 token 排序策略：
| 策略 | 表现 |
|------|------|
| **Difficulty-descending（难→易）** | ✅ 最优，符合设计直觉 |
| Difficulty-ascending（易→难） | 次之 |
| Random bucket assignment | 仍优于标准训练，说明 bucket 结构本身有益 |

> 即使随机分配桶，也能带来增益，表明“结构化掩码”本身即有价值。

#### （3）桶的数量 $K$：
- 测试 $K \in \{4, 8, 16\}$，发现 $K=8$ 表现最佳
- 太少（4）则区分不足；太多（16）则过于细粒度，模型难以学习

#### （4）难度度量指标：
- 对比 **NLL** 与 **Entropy**
- 两者整体表现接近
- **NLL 在 coding 任务上略优**，因其更能捕捉语法关键 token 的置信度变化

> TRIMS 对 metric 选择鲁棒，但推荐使用 NLL

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **显式轨迹监督显著改善 accuracy-parallelism trade-off**  
   注入 token 揭示顺序的知识可有效缓解 train-inference mismatch，引导模型学习更高效的 decoding trajectory。

2. **Hard-to-Easy 解码优于其他顺序**  
   先解决困难 token 能锚定不确定性高的位置，使后续 token 更容易被并行预测。

3. **轻量级 AR 教师足以提供高质量信号**  
   不需要复杂的 DLM rollout 或采样，单次 teacher-forcing 即可获得有效的难度估计。

4. **TRIMS 实现“高性能+低成本”统一**  
   性能媲美蒸馏方法（d3LLM），但训练成本降低两个数量级（见 Table 3）：
   - TRIMS：0.6 GPU-hours 数据整理，1K 数据
   - d3LLM/dParallel：287 GPU-hours × N(model)，93K 数据

---

### ⚠️ 局限性
1. **依赖 AR 教师的质量**  
   若 AR 模型本身对某些领域（如数学）过强，则其难度评分可能偏离 DLM 实际预测行为。
2. **Dream-Instruct 上部分任务 TPS 提升有限**  
   可能源于其由 AR 模型改编而来，已带有一定顺序偏好，post-train 后轨迹优化空间较小。
3. **未探索更丰富的轨迹信号形式**  
   当前仅使用静态 NLL，未来可尝试动态不确定性、语义角色等高级特征。

---

### 🔮 未来工作方向
- 扩展至更大规模模型（如 70B 级别 DLM）
- 探索多模态或结构化任务中的轨迹建模
- 引入强化学习进一步优化 trajectory policy
- 结合 consistency models 或 flow matching 提升收敛速度

---

## ✅ 总结一句话
> **TRIMS 通过引入轻量级 AR 教师提供的 token 难度信号，并设计 trajectory-aware masking 策略，在几乎不增加训练成本的前提下，显著提升了 DLM 的并行解码效率与准确性，实现了性能与实用性的良好平衡。**

</details>

---

### 13. [A Decoupled Basis-Vector-Driven Generative Framework for Dynamic Multi-Objective Optimization](https://arxiv.org/abs/2604.00508)

**Authors**: Yaoming Yang, Shuai Wang, Bingdong Li, Peng Yang, Ke Tang  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.00508v1  

#### Abstract
Dynamic multi-objective optimization requires continuous tracking of moving Pareto fronts. Existing methods struggle with irregular mutations and data sparsity, primarily facing three challenges: the non-linear coupling of dynamic modes, negative transfer from outdated historical data, and the cold-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Decoupled Basis-Vector-Driven Generative Framework for Dynamic Multi-Objective Optimization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
动态多目标优化（**DMOPs**）面临三大挑战：
- **非线性耦合的动态模式**：环境变化复杂且不规则，传统线性预测模型难以捕捉。
- **负迁移（Negative Transfer）**：直接复用历史数据在非周期性或随机环境中可能导致误导。
- **冷启动问题（Cold-Start Problem）**：环境切换后缺乏有效初始种群，导致搜索效率低下。

### 提出了什么新方法或新思路
提出了一种**解耦的基向量驱动生成框架（DB-GEN）**，其核心思想是将动态优化范式从在线自回归预测转变为**零样本（zero-shot）潜在流形生成**。

#### 主要创新点：
- **频率解耦生成框架（Frequency-Decoupled Generative Framework）**  
  引入**离散小波变换（DWT）** 将进化轨迹分解为低频趋势和高频细节，降低学习复杂度，分离可预测模式与随机噪声。

- **基于可迁移基向量的跨问题结构学习（Cross-Problem Structure Learning via Transferable Bases）**  
  采用**稀疏字典学习（Sparse Dictionary Learning）** 学习通用的“物理基向量”（basis vectors），而非记忆具体历史实例。通过拓扑感知的对比约束（topology-aware contrastive loss）构建结构化潜在流形。

- **跨问题零样本生成搜索（Cross-Problem Zero-Shot Generative Search）**  
  模型在包含**1.2亿个解**的大规模离线数据集上预训练，无需在线微调即可执行推理。每次环境变化时，通过潜在空间中的高斯扰动和Tchebycheff筛选机制，**在约0.2秒内生成高质量初始种群**。

### 相比现有方法的优势
- **避免负迁移**：不依赖原始历史数据，而是提取可泛化的结构先验。
- **解决冷启动**：无需等待在线数据积累，实现即时响应。
- **高效性**：零样本推理仅需毫秒级时间，无需在线训练。
- **强泛化能力**：可在完全未见过的任务上进行跨问题生成。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **合成基准测试套件**：
  - **FDA 测试套件** [2]
  - **CEC 2018 套件（DF1–DF14）** [9]
- **真实世界启发问题**：
  - **动态资源分配（DRA）**
  - **动态路径规划（DPP）**

### 实验设置
- **变化配置**：三种标准设置 `(T, n)`：
  - (10,10)：标准变化
  - (5,10)：高频变化
  - (10,5)：剧烈变化
- **算法运行**：每个测试实例独立运行20次以减少统计偏差。
- **硬件平台**：AMD EPYC 7742 CPU、NVIDIA RTX 4090 GPU。

### 评估指标
- **MIGD（Mean Inverted Generational Distance）**：越小越好，衡量收敛性和多样性。
- **MHV（Mean Hypervolume）**：越大越好，衡量目标空间覆盖范围。

### 基线方法对比
选取四类代表性先进算法作为对比：
- **STT-MOEA/D** [22]：基于时空拓扑张量预测
- **DIP-DMOEA** [23]：基于方向改进预测
- **VARE** [21]：基于向量自回归演化
- **SIKT-DMOEA** [31]：基于相似性识别与知识迁移

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- 在 **57个动态环境配置** 中：
  - DB-GEN 在 **49个** 配置上取得最优 MIGD（Top-1）
  - 在 **5个** 配置上取得次优（Top-2）
  - **Top-2命中率达94.7%**
- **Friedman检验平均排名**：
  - MIGD 排名：**1.29**
  - MHV 排名：**1.20**
- 在极具挑战性的 **DF6 和 DF9** 上：
  - DF6 (n=10, T=5)：IGD 从基线最佳 **1.27E+00** 降至 **4.26E-01**（↓66.4%）
  - DF9 (n=10, T=5)：IGD 从 **2.78E-01** 降至 **8.00E-02**（↓71.2%）

### 与基线方法的对比结果
| 指标 | DB-GEN 表现 |
|------|------------|
| **MIGD** | 显著优于所有基线，在绝大多数问题上排名第一 |
| **MHV** | 在 FDA 子集上获得13/15最优结果 |
| **DRA 实际场景** | MIGD 达 **1.35E-02**，较第二名提升 **62.8%** |
| **DPP 实际场景** | 性能接近最优（VARE），显著优于其他 |

### 消融实验结果（Ablation Study）
在57个环境配置上对核心模块进行消融分析：

| 变体 | MIGD | 相对退化 | 胜/负/平 |
|------|------|---------|--------|
| 完整模型（Full Model） | **0.0832** | — | — |
| 移除高低频解耦（w/o High-Low Freq） | 0.0983 | ↑18.1% | 4/53/0 |
| 移除基向量学习（w/o Basis Learning） | 0.1205 | ↑44.8% | 14/43/0 |
| 移除解VAE（w/o Solution VAE） | 0.1028 | ↑23.6% | 23/34/0 |
| 移除三元组损失（w/o Triplet Loss） | 0.1048 | ↑26.0% | 15/42/0 |
| 移除分类损失（w/o Classify Loss） | 0.0934 | ↑12.3% | 17/40/0 |

> **结论**：基向量学习是最重要的模块；频率解耦、VAE 和辅助任务均显著提升性能。

---

## 4. 关键结论和发现

### 主要发现
1. **DB-GEN 实现了强大的跨问题零样本泛化能力**，即使面对与训练集无重叠的问题（如DF6），仍能通过组合已学基向量生成高质量解。
2. **潜在空间具有明确的拓扑语义结构**：t-SNE可视化显示不同问题形成连续轨迹，且几何相似问题被聚在一起。
3. **基向量具备可解释性**：部分基向量编码“形状神经元”（控制曲率）、“位移神经元”（控制位置偏移）、“强度神经元”（控制全局变化强度）。
4. **方法在剧烈变化下表现尤为突出**，而传统预测方法在此类场景中容易失效。

### 方法的局限性
- **静态超参数配置**：基向量数量 `K` 和扰动半径 `σ` 固定，无法自适应不同动态严重程度。
- **当前评估限于低维无约束环境**：尚未验证在高维或带约束 DMOPs 上的表现。
- **外分布（OOD）失败风险**：当新环境动态正交于历史基向量张成的空间时（如DF7），性能会下降。

### 未来工作方向
- 设计**自适应机制**，动态调整潜在流形结构和采样策略。
- 扩展至**高维和约束 DMOPs**，探索结合空间离散化或降维技术 [52]。
- 探索**因果推理**以增强对动态本质的理解与建模 [51]。
- 构建更通用的**跨问题动态优化基础模型**。

--- 

> ✅ **总结**：DB-GEN 是首个将大规模预训练与结构化解耦生成相结合的 DMOP 框架，实现了从“被动响应”到“主动生成”的范式转变，在准确性、鲁棒性和效率方面全面超越现有方法。

</details>

---

### 14. [A Safety-Aware Role-Orchestrated Multi-Agent LLM Framework for Behavioral Health Communication Simulation](https://arxiv.org/abs/2604.00249)

**Authors**: Ha Na Cho  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.00249v1  

#### Abstract
Single-agent large language model (LLM) systems struggle to simultaneously support diverse conversational functions and maintain safety in behavioral health communication. We propose a safety-aware, role-orchestrated multi-agent LLM framework designed to simulate supportive behavioral health dialogu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Safety-Aware Role-Orchestrated Multi-Agent LLM Framework for Behavioral Health Communication Simulation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **single-agent LLM** 的对话系统在行为健康（behavioral health）沟通中面临两大挑战：
- **功能单一性**：难以同时支持多种复杂的、相互依赖的对话功能（如共情、动机激励、认知重构、行动规划等）。
- **安全性不足**：缺乏持续、实时的安全审计机制，存在生成不当内容（如情感伤害、伦理风险）的风险。

这些问题限制了LLM在敏感医疗场景中的可靠部署。

### 🚀 提出的新方法与创新点
本文提出一个 **安全感知的、角色协同的多智能体LLM框架（safety-aware, role-orchestrated multi-agent LLM framework）**，其核心创新包括：

1. **模块化角色分工（Modular Role Decomposition）**
   - 将支持性对话功能分解为六个专用 agent：
     - `Empathizer`（共情）
     - `Motivator`（动机激励）
     - `Planner`（行动计划）
     - `Cognitive Restructurer`（认知重构）
     - `Director`（响应合成）
     - `Responsible Agent`（安全监督）
   - 每个 agent 具备明确的角色提示（role-specific prompt），实现功能可解释性和可控性。

2. **动态协调控制器（Dynamic Agent Controller）**
   - 基于用户输入和上下文，通过 **prompt-encoded transition rules** 动态激活相关 content-producing agents。
   - 非固定调度，而是上下文驱动的选择机制，提升灵活性。

3. **嵌入式安全架构（Embedded Safety Oversight）**
   - `Responsible Agent` 在每一轮对话中都进行 **持续的安全审计（continuous safety auditing）**，检查情绪适当性与伦理合规。
   - 安全不是后处理过滤，而是内建于生成流程中的结构性保障。

4. **可分析的模拟平台设计**
   - 强调系统设计的 **interpretability、transparency 和 reproducibility**，定位为研究工具而非临床干预产品。

### 🔍 相比现有方法的优势
| 维度 | 传统 single-agent 方法 | 本文 multi-agent 框架 |
|------|------------------------|-------------------------|
| 功能多样性 | 单一模型尝试覆盖所有功能 → 易混淆或遗漏 | 角色专业化 → 更高 functional diversity |
| 可控性与可解释性 | 黑箱行为，难追踪决策路径 | 模块化结构 + 日志记录 → 行为可观测 |
| 安全机制 | 多为事后过滤或规则屏蔽 | 内建 `Responsible Agent` 实现实时监控 |
| 协调机制 | 缺乏显式协作逻辑 | 控制器基于语义触发 agent 调用 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **DAIC-WOZ corpus**：
  - 包含189个半结构化心理 distress 面试录音及文本转录。
  - 采用 **Wizard-of-Oz** 协议收集，虚拟采访者由真人操控。
- 本研究仅使用 **participant utterances** 作为输入，排除系统回应。
- 从完整数据集中随机选取 **7名参与者**（ID: 300, 319, 361, 396, 446, 480, 492）用于系统级分析。

### ⚙️ 实验设置
- **LLM backend**：使用 `GPT-3.5-turbo` 和 `GPT-4-turbo` 进行 agent 推理与评估。
- **运行环境**：Apple M2 macOS 系统（11核CPU，18GB RAM）。
- **模拟轮次**：每个用户话语触发最多两轮内部 agent 交互（internal turns）。
- **上下文窗口**：最多保留前3条用户话语和 agent 输出，按角色过滤以减少冗余。

### 🎯 评估指标（Proxy-based Evaluation）
由于非临床应用，采用可扩展的代理评估框架（scalable proxy metrics）：

| 类别 | 指标 | 工具/方法 |
|------|------|-----------|
| **质量维度评分** | Empathy, Helpfulness, Coherence, Appropriateness, Role Alignment （5点Likert量表） | GPT-4-turbo 自动打分 |
| **意图分类** | Zero-shot 分类到12类治疗意图：<br>- Validation, Encouragement, Reflection, Psychoeducation, Coping Suggestion, Cognitive Reframing, Reassurance, Empowerment, Goal Orientation, Active Listening, Generic, Inappropriate | GPT-3.5-turbo pipeline |
| **语言多样性** | Word Count, Type-Token Ratio (TTR) | 文本统计 |
| **计算效率** | Generation Latency（延迟）、Token Usage（消耗） | 记录API响应时间与token数 |

### 🔁 基线对比
- 主要对比对象为 **single-agent baseline**（即统一prompt的单模型响应）。
- 并未直接比较其他 multi-agent 架构，而是强调自身在 **coordination design、safety integration 和 interpretability** 上的设计优势。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（见 Table II）

| Agent Role | Empathy | Helpfulness | Coherence | Appropriateness | Role Alignment | TTR |
|------------|--------|-------------|----------|------------------|----------------|-----|
| Empathizer | 4.80 | 3.80 | 5.00 | 5.00 | 5.00 | 0.13 |
| Motivator | 4.00 | 4.00 | 5.00 | 4.83 | 5.00 | 0.15 |
| Planner | 3.60 | 3.80 | 5.00 | 4.80 | 4.60 | 0.14 |
| Cognitive Restructurer | 4.00 | 4.00 | 5.00 | 5.00 | 5.00 | **0.24** |
| Director | 4.00 | 4.11 | 5.00 | 5.00 | 5.00 | 0.07 |
| Responsible | 3.86 | 4.00 | 5.00 | 4.93 | 5.00 | 0.08 |

> 注：所有角色在 **Coherence** 和 **Role Alignment** 上均达到满分或接近满分（5.00），表明系统输出结构良好且符合角色预期。

### 🔍 与基线方法的对比结果
- **相比 single-agent baseline**：
  - 展现出更清晰的 **role differentiation** 和 **inter-agent coordination**。
  - 实现更高的 **functional diversity**，能灵活组合不同 therapeutic intents。
  - 存在可预测的权衡（trade-off）：
    - 模块化带来更高安全性与可控性；
    - 但也导致略高的 **response latency**（尤其 Director 合成阶段平均延迟约 3.5 秒）。

### 🔄 Agent 激活与协调模式（Fig. 3 & Fig. 4）
- `Director` 和 `Responsible Agent` 每轮必激活（N=1370），体现其监督职责。
- 内容型 agent 激活频率依上下文而定：
  - `Empathizer`: 高频（情感表达时触发）
  - `Cognitive Restructurer`: 极少被调用 → 表明认知重构需求较低或控制器偏保守
- **Transition 分析显示**：
  - 多数 agent 输出流向 `Director`（如 Responsible → Director: 663次），说明合成机制有效。
  - `Cognitive Restructurer` 下游影响小 → 输出较少参与最终整合。

### 🧪 消融实验（隐含分析）
虽然未设正式消融实验，但文中通过以下方式验证设计有效性：
- 对比不同 **prompt 设计策略**：
  - 初始版本无跨 agent 引用 → 输出碎片化；
  - 加入“参考先前输出”指令后 → 显著改善 chaining 与 coherence。
- 分析 **intent 分布**：
  - 所有12类 intent 均出现至少一次 → 验证功能空间全覆盖。
  - 主导 intent 为：Psychoeducation (34%) > Empowerment (20%) > Encouragement (14%) → 反映 controller 倾向解决方案导向。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **角色专业化可行且有效**  
   明确的角色划分能够引导 agent 生成符合预期的功能性响应，实现 **high role alignment** 与 **functional diversity**。

2. **协调机制是关键设计变量**  
   agent 如何被激活、顺序如何、是否受监督，显著影响对话结构与质量。**orchestration behavior itself 是一等公民的设计要素**。

3. **安全可以架构化实现**  
   `Responsible Agent` 的持久存在使安全审计成为系统内在属性，而非附加组件，提升了系统的可信度与可审计性。

4. **情感能力仍集中于特定 agent**  
   共情主要由 `Empathizer` 承担，未在整个系统中扩散 → 当前设计偏向“情感模块化”，不利于深层次情感融合。

5. **控制器行为偏保守**  
   倾向生成信息性（psychoeducation）和激励性内容，可能忽视深层情感支持（如 reflection、validation），需引入自适应机制优化平衡。

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **仿真环境** | 所有实验基于历史文本回放，非真实用户互动，无法评估实际用户体验。 |
| **串行协调机制** | agent 之间为顺序执行，缺乏真正的并发或多视角协商（multi-perspective deliberation）。 |
| **上下文过滤损失敏感信号** | 删除短句或 disfluencies（如 um, uh）虽提高效率，但可能丢失情感线索。 |
| **依赖外部 LLM 性能** | 整体表现受限于 GPT 系列模型本身的能力与稳定性。 |

### 🔮 未来工作方向
1. **开发 adaptive orchestration 策略**  
   引入 context-aware 或 learning-based 控制器，动态调整 agent 权重与激活概率。

2. **增强 inter-agent interaction**  
   支持 agent 间直接通信与辩论（debate mechanisms），促进更丰富的观点整合。

3. **探索 affective signal propagation**  
   让情感状态能在 agent 间传递，使整个系统具备统一的情感基调，而非孤立响应。

4. **引入 human-in-the-loop evaluation**  
   结合临床专家或目标用户对生成内容进行主观评估，弥补 proxy metrics 的不足。

5. **扩展至多模态输入**  
   结合语音韵律、面部表情等非语言信号，进一步丰富情境理解能力。

---

> 💡 **总结一句话**：  
> 该论文提出了一个**以角色为中心、安全内建、可解释性强的 multi-agent LLM 框架**，为行为健康对话系统的建模、分析与决策支持提供了一个新的系统级范式，虽不适用于直接临床干预，但在 **healthcare simulation 与 system design research** 中具有重要价值。

</details>

---

### 15. [More Human, More Efficient: Aligning Annotations with Quantized SLMs](https://arxiv.org/abs/2604.00586)

**Authors**: Jiayu Wang, Junyoung Lee  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.00586v1  

#### Abstract
As Large Language Model (LLM) capabilities advance, the demand for high-quality annotation of exponentially increasing text corpora has outpaced human capacity, leading to the widespread adoption of LLMs in automatic evaluation and annotation. However, proprietary LLMs often exhibit systematic biase...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*More Human, More Efficient: Aligning Annotations with Quantized SLMs*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在自然语言处理中，**Large Language Model-as-a-Judge (LaaJ)** 和 **LLM-as-an-Annotator** 范式被广泛用于自动评估和标注任务。然而，依赖 **proprietary LLMs**（如 GPT 系列）存在以下核心问题：
- **系统性偏差**：如 position bias、verbosity bias、perplexity bias 等，导致评估结果偏离人类专家共识。
- **不可复现性**：商业模型为黑盒，API 版本更新频繁，缺乏透明度。
- **数据隐私风险**：敏感领域（如法律、医疗）的数据难以通过外部 API 处理。
- **成本高昂**：大规模调用 API 成本高，不利于可持续部署。

### 提出了什么新方法或新思路
本文提出一种基于 **quantized Small Language Model (SLM)** 的监督微调（supervised finetuning）框架，用于构建高度对齐人类标注的自动评估器。其核心创新包括：
- **任务特定对齐（Task-specific Alignment）**：在少量高质量人工标注数据上微调 SLM，使其专注于特定评估维度。
- **多维评分框架（Multi-dimensional Rubric Framework）**：设计了一个包含 **Completeness, Clarity, Interpretability, Conciseness, Accuracy, Relevance** 六个维度的细粒度评分体系，每个维度使用 [-2, 2] 的有序尺度。
- **轻量化与高效训练**：采用 **4-bit quantization** 和 **PEFT（Parameter-Efficient Fine-Tuning）** 技术，在消费级 GPU 上即可完成训练，支持本地化部署。
- **数据增强与正则化策略**：
  - **Prompt Paraphrasing**：提升模型对指令表述变化的鲁棒性。
  - **Component Permutation**：打乱输入顺序以缓解 position bias。
  - **Token Dropout**：随机掩码非关键 token，防止过拟合到特定词法模式。

### 相比现有方法的优势
- **更高的人类一致性**：相比最先进的 proprietary LLMs，实现了更高的 **inter-annotator agreement**（Krippendorff’s α）。
- **可复现、透明、安全**：开源模型 + 本地训练，解决黑盒与数据泄露问题。
- **更高效低成本**：仅需 1.7B 参数模型 + 4-bit 量化，可在单张 A100 或边缘设备运行。
- **通用性强**：在同一训练流程下，在不同任务（如情感分类）上也表现优异。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
1. **SPS Dataset（自建）**：
   - 来源：新加坡监狱服务局（Singapore Prison Service）官网文本。
   - 构成：97 个人工编写的问题，每个问题对应一个上下文段落，并由 7 个不同的开源 SLM 生成候选回答（共 679 条样本）。
   - 标注：由专家人工对每条回答在 6 个维度上打分，保留双套独立标注以反映人类分歧。

2. **GoEmotions Dataset（公开）**：
   - 来源：Demszky et al. (2020)，包含 Reddit 评论的情绪标签（27 类细粒度情绪）。
   - 用途：验证所提方法在标准分类任务上的泛化能力。

### 实验设置和评估指标
- **主模型**：`Qwen3-1.7B`，使用 **Unsloth** 库进行 **4-bit quantized SFT**。
- **训练目标**：将评分任务建模为 **causal language modeling**，直接生成 6 个分数字符串作为输出。
- **损失函数**：仅对 completion 部分计算 loss（completion-only loss）。
- **评估指标**：
  - 在 SPS 数据集上：使用 **Krippendorff’s Alpha (α)** 衡量模型输出与人类标注的一致性。
  - 在 GoEmotions 数据集上：使用 **Accuracy** 和 **Macro-F1**（因有明确 ground truth）。

### 基线方法对比
| 类型 | 模型 |
|------|------|
| **Zero-shot Baselines** | GPT-4o, GPT-5-nano, GPT-5-mini-2025-08-07, GPT-5.2-chat |
| **Few-shot Baselines** | GPT-4o, GPT-5.2-chat（使用 MIPROv2 + dspy 优化 prompt） |
| **消融实验变体** | 不加数据增强的 full finetuning / early stopping / LoRA dropout |

> 注：原始 `Qwen3-1.7B` 因无法生成有效标签未作为 baseline。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ SPS Dataset 结果（Krippendorff’s α）
| 方法 | α 值 |
|------|-----|
| **Our Method** | **0.5774** |
| Full finetuning w/o augmentation | 0.4304 |
| Early stopping w/o augmentation | 0.4380 |
| LoRA dropout (p=0.1) | 0.4067 |
| Zero-shot GPT-5-mini-2025-08-07 (**最佳 proprietary**) | 0.2462 |
| Zero-shot GPT-4o | 0.1964 |
| Few-shot GPT-4o | 0.0101 |

> 📌 **提升幅度**：相比最强 proprietary 模型（GPT-5-mini），**绝对提升 +0.3312 α points**，相对提升超过 **134%**。

#### ✅ GoEmotions Dataset 结果（泛化性验证）
| 方法 | Accuracy | Macro-F1 |
|------|---------|----------|
| **Our Method** | **0.8163** | **0.6380** |
| Full SFT w/o augmentation | 0.7819 | 0.4967 |
| Zero-shot GPT-4o | 0.4741 | 0.3732 |
| Zero-shot GPT-5.2-chat | 0.5062 | 0.4099 |

> 📌 在情感分类任务上，准确率接近 **82%**，几乎是 GPT-4o 的两倍。

### 消融实验结果
- 所有未使用数据增强的方法（包括 LoRA dropout）均显著低于完整方法（最大差距达 **+0.147 α**）。
- 训练曲线显示（见附录图2-4），加入数据增强后 validation loss 下降更稳定，无明显过拟合迹象，说明 **augmentation 显著提升了泛化能力**。

---

## 4. 关键结论和发现

### 主要发现
1. **小模型也能胜过大模型**：一个仅 1.7B 参数的 **quantized SLM**，经过任务特定微调后，在人类一致性方面远超 GPT-4o 和 GPT-5 系列等更大规模的 proprietary 模型。
2. **任务对齐优于零样本推理**：即使 proprietary LLMs 拥有更强的通用能力，但在抽象、多维、主观的评估任务中，缺乏针对性训练会导致严重偏差。
3. **数据质量 > 模型规模**：在高质量人工标注基础上进行微调，比依赖大规模预训练 + 零样本提示更有效。
4. **本地化 SLM 是可行替代方案**：结合 **4-bit quantization + PEFT + augmentation**，可在资源受限环境下实现高性能、可复现、低延迟的自动标注系统。

### 方法的局限性
- **未覆盖非 GPT 系列模型**：由于 API 访问限制，未与 Claude、Gemini 等其他主流 proprietary LLM 对比。
- **依赖人工标注质量**：方法效果受限于初始 human annotations 的质量和一致性。
- **当前仅适用于封闭域任务**：若目标任务分布变化较大，可能需要重新标注和微调。

### 未来工作方向
- 探索 **active learning** 或 **semi-supervised learning** 减少对大量人工标注的依赖。
- 将该框架扩展至更多任务类型，如 summarization evaluation、code review、legal document analysis 等。
- 研究如何将多个 SLM judges 进行集成（ensemble）以进一步提高稳定性。
- 开发自动化工具链，支持端到端的 “human-in-the-loop” annotation pipeline。

---

> 🔗 **代码开源地址**：[https://github.com/jylee-k/slm-judge](https://github.com/jylee-k/slm-judge)  
> 该项目为构建 **去中心化、可信、高效** 的 AI 评估基础设施提供了实用路径。

</details>

---

### 16. [LangMARL: Natural Language Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2604.00722)

**Authors**: Huaiyuan Yao, Longchao Da, Xiaoou Liu, Charles Fleming, Tianlong Chen, Hua Wei  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.00722v1  

#### Abstract
Large language model (LLM) agents struggle to autonomously evolve coordination strategies in dynamic environments, largely because coarse global outcomes obscure the causal signals needed for local policy refinement. We identify this bottleneck as a multi-agent credit assignment problem, which has l...

---

### 17. [Positional Cognitive Specialization: Where Do LLMs Learn To Comprehend and Speak Your Language?](https://arxiv.org/abs/2604.00923)

**Authors**: Luis Frentzen Salim, Lun-Wei Ku, Hsing-Kuo Kenneth Pao  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.00923v1  

#### Abstract
Adapting large language models (LLMs) to new languages is an expensive and opaque process. Understanding how language models acquire new languages and multilingual abilities is key to achieve efficient adaptation. Prior work on multilingual interpretability research focuses primarily on how trained ...

---

### 18. [Using predefined vector systems to speed up neural network multimillion class classification](https://arxiv.org/abs/2604.00779)

**Authors**: Nikita Gabdullin, Ilya Androsov  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.00779v1  

#### Abstract
Label prediction in neural networks (NNs) has O(n) complexity proportional to the number of classes. This holds true for classification using fully connected layers and cosine similarity with some set of class prototypes. In this paper we show that if NN latent space (LS) geometry is known and posse...

---

### 19. [Reconsidering Dependency Networks from an Information Geometry Perspective](https://arxiv.org/abs/2604.01117)

**Authors**: Kazuya Takabatake, Shotaro Akaho  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.01117v1  

#### Abstract
Dependency networks (Heckerman et al., 2000) provide a flexible framework for modeling complex systems with many variables by combining independently learned local conditional distributions through pseudo-Gibbs sampling. Despite their computational advantages over Bayesian and Markov networks, the t...

---

### 20. [Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts](https://arxiv.org/abs/2604.00901)

**Authors**: Sha Li, Naren Ramakrishnan  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.00901v1  

#### Abstract
Multi-agent Retrieval-Augmented Generation (RAG), wherein each agent takes on a specific role, supports hard queries that require multiple steps and sources, or complex reasoning. Existing approaches, however, rely on static agent behaviors and fixed orchestration strategies, leading to brittle perf...

---

### 21. [Is RISC-V Ready for Machine Learning? Portable Gaussian Processes Using Asynchronous Tasks](https://arxiv.org/abs/2604.00736)

**Authors**: Alexander Strack, Patrick Diehl, Dirk Pfl\"uger  
**Category**: cs.DC  
**Published**: 2026-04-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.00736v1  

#### Abstract
Gaussian processes are widely used in machine learning domains but remain computationally demanding, limiting their efficient scalability across diverse hardware platforms. The GPRat library targets these challenges with the help of the asynchronous many-task runtime system HPX. In this work, we ext...

---

### 22. [Evolution Strategies for Deep RL pretraining](https://arxiv.org/abs/2604.00066)

**Authors**: Adrian Mart\'inez, Ananya Gupta, Hanka Goralija, Mario Rico, Sa\'ul Fenollosa, Tamar Alphaidze  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.00066v1  

#### Abstract
Although Deep Reinforcement Learning has proven highly effective for complex decision-making problems, it demands significant computational resources and careful parameter adjustment in order to develop successful strategies. Evolution strategies offer a more straightforward, derivative-free approac...

---

### 23. [Convergence of Byzantine-Resilient Gradient Tracking via Probabilistic Edge Dropout](https://arxiv.org/abs/2604.00449)

**Authors**: Amirhossein Dezhboro, Fateme Maleki, Arman Adibi, Erfan Amini, Jose E. Ramirez-Marquez  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.00449v1  

#### Abstract
We study distributed optimization over networks with Byzantine agents that may send arbitrary adversarial messages. We propose \emph{Gradient Tracking with Probabilistic Edge Dropout} (GT-PD), a stochastic gradient tracking method that preserves the convergence properties of gradient tracking under ...

---

### 24. [Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions](https://arxiv.org/abs/2604.00499)

**Authors**: Haoyu Zheng, Yongqiang Zhang, Fangcheng Fu, Xiaokai Zhou, Hao Luo, Hongchao Zhu, Yuanyuan Zhu, Hao Wang, Xiao Yan, Jiawei Jiang  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.00499v1  

#### Abstract
To schedule LLM inference, the \textit{shortest job first} (SJF) principle is favorable by prioritizing requests with short output lengths to avoid head-of-line (HOL) blocking. Existing methods usually predict a single output length for each request to facilitate scheduling. We argue that such a \te...

---

### 25. [Predicting Dynamics of Ultra-Large Complex Systems by Inferring Governing Equations](https://arxiv.org/abs/2604.00599)

**Authors**: Qi Shao, Duxin Chen, Jiawen Chen, Yujie Zeng, Athen Ma, Wenwu Yu, Vito Latora, Wei Lin  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.00599v1  

#### Abstract
Predicting the behavior of ultra-large complex systems, from climate to biological and technological networks, is a central unsolved challenge. Existing approaches face a fundamental trade-off: equation discovery methods provide interpretability but fail to scale, while neural networks scale but ope...

---

### 26. [Full-Gradient Successor Feature Representations](https://arxiv.org/abs/2604.00686)

**Authors**: Ritish Shrirao, Aditya Priyadarshi, Raghuram Bharadwaj Diddigi  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.00686v1  

#### Abstract
Successor Features (SF) combined with Generalized Policy Improvement (GPI) provide a robust framework for transfer learning in Reinforcement Learning (RL) by decoupling environment dynamics from reward functions. However, standard SF learning methods typically rely on semi-gradient Temporal Differen...

---

### 27. [Performance of Neural and Polynomial Operator Surrogates](https://arxiv.org/abs/2604.00689)

**Authors**: Josephine Westermann, Benno Huber, Thomas O'Leary-Roseberry, Jakob Zech  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.00689v1  

#### Abstract
We consider the problem of constructing surrogate operators for parameter-to-solution maps arising from parametric partial differential equations, where repeated forward model evaluations are computationally expensive. We present a systematic empirical comparison of neural operator surrogates, inclu...

---

### 28. [Open, Reliable, and Collective: A Community-Driven Framework for Tool-Using AI Agents](https://arxiv.org/abs/2604.00137)

**Authors**: Hy Dang, Quang Dao, Meng Jiang  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.00137v1  

#### Abstract
Tool-integrated LLMs can retrieve, compute, and take real-world actions via external tools, but reliability remains a key bottleneck. We argue that failures stem from both tool-use accuracy (how well an agent invokes a tool) and intrinsic tool accuracy (the tool's own correctness), while most prior ...

---

### 29. [The Silicon Mirror: Dynamic Behavioral Gating for Anti-Sycophancy in LLM Agents](https://arxiv.org/abs/2604.00478)

**Authors**: Harshee Jignesh Shah (Independent Researcher)  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.00478v1  

#### Abstract
Large Language Models (LLMs) increasingly prioritize user validation over epistemic accuracy-a phenomenon known as sycophancy. We present The Silicon Mirror, an orchestration framework that dynamically detects user persuasion tactics and adjusts AI behavior to maintain factual integrity. Our archite...

---

### 30. [TR-ICRL: Test-Time Rethinking for In-Context Reinforcement Learning](https://arxiv.org/abs/2604.00438)

**Authors**: Wenxuan Jiang, Yuxin Zuo, Zijian Zhang, Xuecheng Wu, Zining Fan, Wenxuan Liu, Li Chen, Xiaoyu Li, Xuezhi Cao, Xiaolong Jin, Ninghao Liu  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.00438v1  

#### Abstract
In-Context Reinforcement Learning (ICRL) enables Large Language Models (LLMs) to learn online from external rewards directly within the context window. However, a central challenge in ICRL is reward estimation, as models typically lack access to ground-truths during inference. To address this limita...

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
