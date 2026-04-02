# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-02 06:58:46 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [From Skew to Symmetry: Node-Interconnect Multi-Path Balancing with Execution-time Planning for Modern GPU Clusters](https://arxiv.org/abs/2604.00317)

**Authors**: Jinghan Yao, Kaushik Kandadi, Bharath Ramesh, Hari Subramoni, Dhabaleswar K. Panda  
**Category**: cs.DC  
**Published**: 2026-04-02  
**Score**: 15.5  
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
现代基于GPU的高性能计算（HPC）和人工智能（AI）集群虽然具备极高的聚合通信带宽（通过NVLink、NVSwitch、多轨InfiniBand等异构互连），但在实际应用中，**通信流量的严重偏斜（traffic skew）** 导致部分链路过载而其他链路闲置，造成网络拥塞、尾延迟激增和可扩展性瓶颈。

典型场景包括：
- **Mixture-of-Experts (MoE)** 中的 `All-to-Allv` 通信
- 不规则图算法（如BFS）
- 自适应数值模拟中的边界热点
- 推荐系统中的稀疏聚合通信

现有通信库（如NCCL、MPI/UCX）依赖静态路径选择（如最短路径、静态哈希分轨），无法在运行时动态响应负载变化，导致资源利用率低下。

---

### 提出了什么新方法或新思路
作者提出 **NIMBLE**（Node-Interconnect Multi-path BaLancing with Execution-time orchestration），一个**端点驱动的运行时通信编排系统**，实现跨节点和片内互连的多路径动态负载均衡。

#### 核心创新点：
- **容量归一化的最小拥塞优化（Capacity-normalized Minimum Congestion Optimization）**
  - 使用 **multiplicative-weights update (MWU)** 快速求解近似最优路径分配
  - 路径成本基于“瓶颈链路”的负载，而非总和，更贴合流水线吞吐特性
- **GPU内核级RDMA流水线（CUDA-aware GPU kernel-based RDMA pipelining）**
  - 支持通过中间GPU进行多跳转发，无需主机干预
  - 利用P2P缓冲区和原子计数器实现细粒度同步
- **端点驱动的透明集成**
  - 无需修改应用程序代码
  - 与现有通信库（如NCCL）兼容，仅在 `Send/Recv` 和 `Alltoallv` 等非均匀操作中激活
- **实用策略保障正确性和低开销**
  - 大小阈值：小消息（≤1MB）不启用多路径以避免开销
  - 每目的地重排序队列：保证消息顺序
  - 滞后机制（hysteresis）：防止路径震荡

---

### 相比现有方法的优势
| 特性 | NCCL / MPI / UCX | NIMBLE |
|------|------------------|--------|
| 路径选择 | 静态（初始化时确定） | 动态运行时调整 |
| 多路径利用 | 有限（如PXN用于AlltoAll） | 全链路感知，支持GPU中继 |
| 负载感知 | 否 | 是（实时监控链路利用率） |
| 小消息处理 | 直接传输 | 启用大小阈值避免开销 |
| 透明性 | 高 | 高（无缝集成NCCL） |
| 性能提升（偏斜场景） | 基准 | **最高达5.2×** |

---

## 2. 核心实验方法和设置

### 实验硬件配置
- **节点配置**：每节点2颗Intel Xeon Platinum 8470 CPU，4颗NVIDIA H100 SXM5 GPU（94GB HBM2e）
- **片内互连**：全连接NVLink 4（理论峰值120 GB/s per link）
- **片间互连**：4个NDR400 InfiniBand HCAs（每卡50 Gb/s ≈ 45.1 GB/s实测），支持GPUDirect RDMA
- **拓扑**：2节点共8 GPU，用于MoE和All-to-Allv测试

---

### 软件与基线方法
- **基线方法**：
  - **NCCL v2.26**（主流GPU集合通信库）
  - **OpenMPI v5.0.7 + UCX v1.18.0**（支持CUDA-aware RDMA）
- **NIMBLE实现**：
  - 集成于NCCL框架，替换其底层路径调度逻辑
  - 使用kernel-level RDMA pipelining实现多跳转发

---

### 评估指标
- **吞吐量（Throughput）**：GB/s
- **端到端延迟（End-to-end Latency）**
- **尾延迟（p99 Latency）**
- **链路利用率均衡性**
- **算法决策开销（Orchestration Overhead）**

---

### 测试工作负载
1. **点对点通信（Point-to-Point）**
   - 不同大小的消息（1MB–256MB）
   - 模拟偏斜发送/接收模式
2. **偏斜All-to-Allv通信**
   - 控制“热点比例”（hotspot ratio）从0.4到1.0
   - 每个rank将固定比例的数据发往单一“热”目标
3. **端到端MoE推理块**
   - 使用真实LLM MoE架构参数
   - 分解为三个阶段：dispatch → compute → combine
   - 变化全局token数量（2K–64K）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）片内多路径聚合带宽（Intra-node Multi-path）
| 配置 | 峰值带宽 |
|------|---------|
| 直接NVLink（单路径） | 120 GB/s |
| 单中间GPU转发（2跳） | 213.1 GB/s |
| 双中间GPU转发（3路径） | **278.2 GB/s** |
- **提升达2.3×**，饱和于~64MB以上消息
- 小于1MB禁用多路径以避免开销

#### （2）片间多轨聚合带宽（Inter-node Multi-rail）
| 配置 | 峰值带宽 |
|------|---------|
| 单NIC（rail-matched） | 45.1 GB/s |
| 四NIC并行 | **170.0 GB/s** |
- **提升3.8×**，接近线性扩展
- 中继转发开销极小（<5%）

#### （3）偏斜All-to-Allv性能
- 在热点比率为0.7+时，NIMBLE相比NCCL：
  - **最高提速5.2×**
  - OpenMPI表现次之，因DMA引擎在小消息上更优
- 随着消息总量增加，加速比持续上升

#### （4）端到端MoE推理性能
- 在16K tokens、热点比0.9时：
  - **端到端加速1.35×**
  - dispatch和combine阶段显著缩短
  - compute阶段不变（公平比较）
- 平均加速比随热点比上升：
  - 0.4 → ~1.13×
  - 0.9 → ~1.26×，峰值1.35×

#### （5）算法开销（Orchestration Overhead）
| 消息大小 | 决策时间（ms） |
|----------|----------------|
| 16MB | 0.0321（intra），0.0374（inter） |
| 256MB | 0.0363 / 0.0480 |
- **远低于通信耗时**（如256MB通信需6.5ms），可忽略

---

### 与基线方法的对比结果
| 场景 | NIMBLE vs NCCL | NIMBLE vs OpenMPI |
|------|----------------|------------------|
| 偏斜All-to-Allv（高热点） | **↑5.2×** | **↑3.8×** |
| MoE端到端 | **↑1.35×** | 类似趋势 |
| 点对点（大消息） | ↑1.15–2.3× | ↑最高3.4× |
| 均衡流量 | 性能持平 | 性能持平 |

> ✅ **NIMBLE在偏斜场景下显著胜出，在均衡场景下无退化**

---

### 消融实验（Ablation Study）
- **禁用多跳转发**：性能回落至NCCL水平
- **移除大小阈值**：小消息性能下降（因转发开销）
- **关闭滞后机制**：路径震荡导致吞吐波动
- **使用静态哈希代替MWU**：无法适应动态偏斜，收益下降40%+

---

## 4. 关键结论和发现

### 主要发现
1. **通信偏斜是现代GPU集群的关键瓶颈**，即使总带宽充足，局部拥塞仍严重限制性能。
2. **动态、端点驱动的多路径编排可行且高效**：NIMBLE可在毫秒级完成路径重分配，开销可忽略。
3. **GPU中继转发 + 多轨NIC 可大幅提升有效带宽**：
   - 片内可达278.2 GB/s（2.3×）
   - 片间可达170.0 GB/s（3.8×）
4. **在真实MoE负载中，NIMBLE带来高达1.35×的端到端加速**，尤其适用于大规模、高偏斜推理。
5. **NIMBLE与现有生态兼容**，无需修改应用即可集成进NCCL。

---

### 方法的局限性
1. **依赖全连接片内拓扑**：
   - 在使用NVSwitch集中交换的DGX系统中，GPU间无直连链路，**无法进行片内多路径转发**
   - 但仍可用于片间多轨负载均衡
2. **多跳转发有启动开销**：
   - 仅对大消息（>64MB）有益
   - 小消息（≤1MB）默认走直接路径
3. **未解决跨作业干扰调度**：
   - NIMBLE聚焦单作业内部优化，依赖外部CC机制（如DCQCN/HPCC）保障多租户公平性

---

### 未来工作方向
1. **优化编排引擎**：
   - 引入预测模型预判偏斜趋势
   - 减少监控反馈延迟
2. **探索IBGDA（InfiniBand GPUDirect Async）**：
   - 实现完全异步、零CPU介入的RDMA操作
   - 进一步降低转发延迟和CPU占用
3. **扩展至更多Collective类型**：
   - 当前主要优化 `Send/Recv` 和 `Allto-Allv`
   - 未来可探索偏斜 `Reduce-Scatter` 或 `AllGather`
4. **支持异构设备混合集群**（如AMD + NVIDIA）

---

> **总体评价**：NIMBLE为现代异构GPU集群提供了一种灵活、高效、透明的通信负载均衡方案，显著缓解了由流量偏斜引起的性能瓶颈，推动了通信密集型AI/HPC应用的发展。

</details>

---

### 2. [Universal YOCO for Efficient Depth Scaling](https://arxiv.org/abs/2604.01220)

**Authors**: Yutao Sun, Li Dong, Tianzhu Ye, Shaohan Huang, Jianyong Wang, Furu Wei  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2604.01220v1  

#### Abstract
The rise of test-time scaling has remarkably boosted the reasoning and agentic proficiency of Large Language Models (LLMs). Yet, standard Transformers struggle to scale inference-time compute efficiently, as conventional looping strategies suffer from high computational overhead and a KV cache that ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Universal YOCO for Efficient Depth Scaling》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前主流的 **Transformer** 架构在进行 **test-time scaling**（推理时计算扩展）时面临显著效率瓶颈：
- **高计算开销**：传统循环机制（如 Universal Transformer）需要重复执行所有层，导致计算复杂度随深度线性增长。
- **KV Cache 膨胀**：每轮迭代都会生成新的 Key-Value 缓存，内存占用与模型深度成正比，严重限制长上下文场景下的可扩展性。

这些问题阻碍了 LLM 在复杂推理任务中高效地“思考更深”（think deeper）。

### 提出了什么新方法或新思路
本文提出 **Universal YOCO (YOCO-U)**，一种结合 **YOCO 架构** 与 **递归计算** 的新型高效深度扩展框架。其核心思想是：

- **YOCO 架构基础**：将模型分为两个模块：
  - **Self-Decoder**：使用 **Efficient Self-Attention (ESA)**（如滑动窗口注意力）处理输入，生成一个紧凑的全局 KV Cache。
  - **Cross-Decoder**：复用该共享 KV Cache 进行自回归生成，避免逐层缓存。
- **引入 Universal Self-Decoder**：在 Self-Decoder 模块上应用参数共享的多步递归计算（T 次迭代），提升表征能力而不增加参数量。
- **Partial Recursion 设计**：仅对浅层、高效的 Self-Decoder 模块进行递归，而 Cross-Decoder 保持不变。

> ✅ **核心创新**：实现了“**Think Deeper, Cache Once**”——通过局部递归增强推理深度，同时保持全局 KV Cache 仅需构建一次。

### 相比现有方法的优势
| 特性 | Standard Transformer | Universal Transformer | YOCO | **YOCO-U** |
|------|------------------------|------------------------|------|------------|
| 参数共享 | ❌ | ✅ | ❌ | ✅ |
| 全局 KV Cache 复用 | ❌ | ❌ | ✅ | ✅ |
| 推理预填充复杂度 | O(N²) | O(TN²) | O(N) | O(TN) |
| KV Cache 内存 | O(LND) | O(LTND) | O(ND) | **O(ND + WTL D)** |
| 递归开销 | — | 高（全注意力） | — | **低（仅 ESA 层）** |

> 💡 YOCO-U 在保持 YOCO 高效推理优势的同时，通过轻量级递归显著提升了模型能力，实现了更优的 **capability-efficiency tradeoff**。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **语言建模训练数据**：大规模通用文本语料（未明确命名，序列长度为 8192，batch size 4M tokens）
- **下游任务评估基准**：
  - **通用能力**：ARC-C, Winogrande, HellaSwag, MMLU, BBH, DROP
  - **数学推理**：GSM8K, MATH, SVAMP, ASDiv, MAWPS, CARP, TABMWP, Gaokao, OlympiadBench, CollegeMath, AMC23（共11个）
  - **长上下文建模**：Book 和 Code 数据上的长序列 perplexity 测试
  - **信息检索能力**：Needle In A Haystack (NIAH)

### 实验设置和评估指标
- **模型配置**：
  - 主要模型大小：10B 总参数，1.3B 激活参数（MoE 架构）
  - 层数：20 层（Self-Decoder 和 Cross-Decoder 各 10 层）
  - 默认递归次数 T=3 → 总 FLOPs 约为非递归基线的 2 倍
- **训练设置**：
  - 训练步数：75k 步（约 300B tokens）
  - 优化器：AdamW (β₁=0.9, β₂=0.95)，学习率 1e-3
  - 硬件：AMD MI300X GPUs
- **评估指标**：
  - **Perplexity (ppl)**：衡量语言建模质量
  - **Accuracy (acc)**：分类/选择题准确率
  - **Normalized Accuracy (acc_n)**：长度归一化的多选题评分
  - **Throughput (tokens/s)**：prefill 和 decode 阶段吞吐量
  - **KV Cache Memory (MB)**：缓存内存占用

### 基线方法对比
- **Non-recursive baselines**：
  - Standard Transformer
  - YOCO（非递归版本）
- **Recursive baselines**：
  - **Universal Transformer (UT)**：整网络循环
  - **RINS**：标准 Decoder-only 模型中的早期层递归
  - **ParScale**：并行扩展方法
- 所有方法在 FLOPs 上尽量对齐以公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 下游任务综合表现（Table 2）
| Model | ARC-C | Winogrande | HellaSwag | MMLU | BBH | GSM8K | Humaneval | DROP | **Average** |
|-------|-------|------------|-----------|------|-----|--------|-----------|------|-------------|
| YOCO | 46.50 | 61.72 | 63.44 | 49.59 | 33.13 | 38.06 | 9.15 | 32.62 | **41.78** |
| YOCO-U | 47.87 | 68.67 | 66.80 | 54.63 | 35.49 | 50.49 | 10.98 | 34.94 | **46.23** (+4.45) |

> 🔺 即使在相同 FLOPs 下，YOCO-U 也大幅超越基线，说明增益来自架构有效性而非单纯算力堆叠。

#### ✅ 数学推理能力（Figure 3）
- 在 **11 个数学基准** 上平均准确率提升 **+24.4%**
- 表明递归计算有效增强了隐式和显式推理能力，且与 test-time scaling 正交。

#### ✅ 长上下文建模（Figure 4）
- 在 Book 和 Code 数据上，随着前缀长度增加，YOCO-U 的 **perplexity 显著低于 Transformer 和 YOCO**
- 与 RINS 相当，表明其长程依赖建模能力未因高效注意力受损。

#### ✅ 信息检索能力（Table 4）
| Model | S-NIAH-1 | S-NIAH-2 |
|-------|----------|----------|
| Transformer | 0.87 | 0.82 |
| YOCO | 1.00 | 0.86 |
| RINS | 0.99 | 0.91 |
| **YOCO-U** | **1.00** | **0.95** |

> 🔍 YOCO-U 在超长序列中定位关键信息的能力最强，验证了其强大的上下文利用效率。

### 与基线方法的对比结果（Table 3）
| Model | Wiki.ppl | LMB.acc↑ | PIQA | OBQA | Hella | ARC-E | ARC-C | **Avg.acc** |
|-------|----------|----------|------|------|--------|--------|--------|-------------|
| Transformer | 22.52 | 38.4 | 69.6 | 22.6 | 45.7 | 57.1 | 59.6 | 47.1 |
| YOCO | 22.25 | 41.2 | 67.9 | 23.8 | 45.6 | 54.3 | 59.2 | 47.0 |
| RINS | 20.98 | 39.4 | 69.4 | 24.0 | 49.0 | 54.2 | 62.0 | 48.3 |
| **YOCO-U** | **21.01** | **41.2** | **68.7** | **24.6** | **48.9** | **55.3** | **62.2** | **48.3** |

> 🏆 YOCO-U 在多个指标上达到最优，尤其在 perplexity 上优于所有递归变体。

### 消融实验结果（Table 5）
| Model Variant | LMB.acc | Hella | ARC-C | **Avg.acc** |
|---------------|---------|--------|--------|-------------|
| YOCO (non-recursive) | 41.16 | 45.64 | 36.60 | 46.95 |
| YOCO-U (proposed) | 41.18 | 48.89 | 36.95 | **48.25** |
| Upper Loop (Cross-Decoder 循环) | 39.78 | 46.27 | 37.80 | 47.34 |
| Upper Loop w/o Shared KV | 38.21 | 45.69 | 35.75 | 46.41 |

> 🔍 发现：
> - 将递归应用于深层 Cross-Decoder 效果较差，支持“浅层递归更优”的设计。
> - 共享 KV Cache 对性能稳定至关重要。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Partial Recursion + Efficient Attention 是高效深度扩展的有效路径**：
   - 仅在浅层 ESA 模块中引入递归，即可显著提升模型能力，而不会带来高昂的 KV Cache 开销。
2. **“One-Piece KV Cache” 架构具有巨大潜力**：
   - YOCO-U 成功继承了 YOCO 的线性预填充和固定 KV Cache 优势，使其在长上下文场景下极具部署价值。
3. **递归计算能有效提升 token utility 和 parameter utility**：
   - 图 2 和图 5 显示，YOCO-U 在相同 FLOPs 或更少参数下能达到更好性能，甚至可用 **50% 更少参数** 达到同等效果。
4. **Representation 分析显示稳定性良好**：
   - 角距离分析（Figure 8）表明递归过程稳定收敛，且 Self-Decoder 与 Cross-Decoder 功能分离明显。

### 方法的局限性
- 当前实现仍局限于 **autoregressive language modeling**，尚未探索 encoder-decoder 或 multimodal 场景。
- 虽然递归开销小，但在极端长序列（>256K）下，局部 ESA 的缓存累积仍可能成为瓶颈（尽管远小于标准递归）。
- 实验集中在 MoE 和 Dense 架构，未覆盖所有主流 LLM 结构（如纯 Mamba 或 RetNet）。

### 未来工作方向
- 探索 **adaptive recursion depth**：根据输入难度动态调整迭代次数（类似 ETC 或 Think-once）。
- 将 YOCO-U 拓展至 **vision-language models** 或 **agent systems** 中，支持更复杂的推理流程。
- 结合 **test-time scaling** 技术（如 Chain-of-Thought, Tree-of-Thought），进一步释放“思考更深”的潜力。
- 研究如何将递归机制迁移到其他高效架构（如 RetNet, Mamba）中，形成新一代高效 backbone。

---

> ✅ **总体评价**：  
> YOCO-U 提出了一种优雅而实用的解决方案，在不牺牲推理效率的前提下实现了有效的深度扩展。它不仅在性能上全面超越基线，更重要的是为构建 **scalable, cost-effective, and long-context-capable LLMs** 提供了一个清晰可行的技术路线。该工作有望推动下一代高效大模型架构的发展。

</details>

---

### 3. [Agent Q-Mix: Selecting the Right Action for LLM Multi-Agent Systems through Reinforcement Learning](https://arxiv.org/abs/2604.00344)

**Authors**: Eric Hanchen Jiang, Levina Li, Rui Sun, Xiao Liang, Yubei Li, Yuchen Wu, Haozheng Luo, Hengli Li, Zhi Zhang, Zhaolu Kang, Kai-Wei Chang, Ying Nian Wu  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 10.0  
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
在基于 **Large Language Model (LLM)** 的多智能体系统（Multi-Agent Systems, MAS）中，如何高效地选择和连接多个智能体以解决复杂任务是一个关键挑战。现有的通信拓扑（communication topology）设计通常存在以下问题：
- **静态方法**（如链式、星型、全连接图）缺乏灵活性，无法根据任务难度动态调整通信模式。
- **自适应方法**大多依赖**集中式生成器**（centralized topology generator），导致执行时仍需全局控制，限制了去中心化决策能力。

因此，本文旨在解决：**如何实现一种可学习、去中心化且高效的通信拓扑选择机制**，从而提升多智能体系统的性能、鲁棒性和资源效率。

---

### 提出了什么新方法或新思路
作者提出 **Agent Q-Mix**，一个将通信拓扑学习建模为**合作型多智能体强化学习**（Cooperative Multi-Agent Reinforcement Learning, MARL）问题的框架，并采用 **QMIX** 进行值函数分解。

#### 核心思想：
- 将每个智能体的通信行为视为一组**离散动作**（如广播、查询、辩论等），联合决定每轮的通信图结构。
- 使用 **Centralized Training with Decentralized Execution (CTDE)** 范式，在训练时利用全局信息优化策略，部署时允许各智能体独立决策。
- 引入**拓扑感知的GNN编码器**（topology-aware GNN）、**GRU记忆模块** 和 **QMIX单调混合网络** 来实现去中心化的动作选择。

---

### 相比现有方法的优势
| 维度 | Agent Q-Mix 的优势 |
|------|---------------------|
| **去中心化决策** | 每个智能体自主选择通信动作，支持在线自适应，避免集中瓶颈 |
| **可解释性与表达力** | 六种通信动作（如 `broadcast`, `debate_check`）具有明确语义，可组合出多样拓扑 |
| **效率与鲁棒性** | 显式奖励函数平衡准确率与 token 成本；能自动隔离故障或对抗性智能体 |
| **跨模型泛化性** | 在 GPT-OSS:120B 和 Gemini-3.1-Flash-Lite 上均表现优异 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验覆盖三大领域共 **7个基准任务**：

| 领域 | 数据集 |
|------|--------|
| **编程** | LiveCodeBench v6, HumanEval |
| **推理** | MMLU-Pro |
| **数学** | AIME2025, AIME2026, Beyond-AIME, HMMT2025 |
| **额外挑战测试** | Humanity's Last Exam (HLE) —— 多领域综合MCQ测试 |

此外还进行了消融实验与对抗性攻击测试。

---

### 实验设置和评估指标

#### 模型配置
- **基础LLM**：GPT-OSS:120B 和 Gemini-3.1-Flash-Lite
- **智能体团队**：每任务使用3个智能体 + 1个 FinalRefer 决策节点
  - 编程：CodeWriter ×2 + AnalyzeAgent
  - 推理：ReasoningAgent ×2 + AnalyzeAgent
  - 数学：MathSolver ×2 + AnalyzeAgent
- **通信轮次**：数学任务 T=3，编程/推理任务 T=2

#### 架构组件
- **GNN Encoder**：2层，hidden size=128
- **GRU Memory**：hidden size=128
- **QMIX Mixing Network**：超网络输出维度64
- **训练参数**：Adam优化器，lr=5e-4，batch size=8，e-greedy探索（从1.0衰减至0.05）

#### 奖励函数
$$
R = w_{\text{acc}} \cdot \text{accuracy} - w_{\text{tok}} \cdot \min\left(\frac{\text{tokens\_used}}{\text{max\_tokens}}, 1\right)
$$
其中 `max_tokens=10,000`，权重根据模型调整。

#### 评估指标
- **主指标**：Accuracy (%)
- **辅助指标**：
  - Token 使用量（衡量效率）
  - 对抗攻击下的鲁棒性（robustness）
  - 消融实验中的性能变化

---

### 基线方法对比
分为四类基线进行比较：

| 类别 | 方法 |
|------|------|
| **单智能体基线** | Base (direct prompting) |
| **静态多智能体** | LLM-Debate |
| **自适应拓扑方法** | GPTSwarm, AgentDropout, G-Designer, MaAS, TopoDIM, GTD |
| **商业框架** | LangGraph, AutoGen, Microsoft Agent Framework, Lobster |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（GPT-OSS:120B）

| 方法 | 平均准确率 (%) |
|------|----------------|
| **Agent Q-Mix (Ours)** | **72.73** ✅ |
| GTD (最佳自适应方法) | 60.37 |
| AutoGen (最强商业框架) | 68.60 |

> 在所有七项任务上取得最高平均精度。

#### 各任务表现亮点：
- **数学任务大幅提升**：
  - HMMT2025: **53.33%** vs 第二名 43.33% (+10.00 pts)
  - Beyond-AIME: **42.00%** vs GTD (35.00%) / AutoGen (37.00%)
- **编程与推理保持领先**：
  - LiveCodeBench: **100.00%**
  - HumanEval: **97.56%**
  - MMLU-Pro: **92.86%**

---

### 与基线方法的对比结果

#### 在 Gemini-3.1-Flash-Lite 上的表现
| 方法 | 平均准确率 (%) |
|------|----------------|
| **Agent Q-Mix** | **66.90** ✅ |
| Microsoft Agent Framework | 64.88 |
| AutoGen | 63.41 |

> 即使在较小模型上也显著优于主流框架。

#### Humanity's Last Exam (HLE) 结果
- **Agent Q-Mix**: **20.8%**
- Microsoft Agent Framework: 19.2%
- LangGraph: 19.2%
- AutoGen / Lobster: ≤18.9%

✅ **首次在该高难度综合测试中突破20%门槛**

---

### 消融实验结果（见 Figure 5）

#### (a) 智能体数量影响
- Beyond-AIME 准确率随智能体数增加而上升，4个时达38%，10个时趋稳于41%
- HumanEval 始终接近100%，说明策略能抑制冗余通信

#### (b) 训练样本量影响
- 仅用 **15个训练样例/领域** 即可达95.73%准确率
- 证明 QMIX 策略具备强**样本效率**（sample efficiency）

#### (c) 奖励权重影响
- 提高 $w_{\text{acc}}$ 可提升准确率，但 token 成本增长缓慢
- 设定 $w_{\text{acc}}=1.5$（Gemini）和 $1.25$（GPT-OSS）为最优权衡点

#### (d) 通信轮次影响
- 数学任务需 T=3 达峰值（96.95%）
- 编程/推理任务 T=2 已足够（复杂度较低）

---

## 4. 关键结论和发现

### 主要发现
1. **去中心化的拓扑学习是可行且有效的**  
   Agent Q-Mix 首次实现了完全由智能体自主选择通信动作的机制，打破了对集中式控制器的依赖。

2. **QMIX + GNN 架构天然适配拓扑学习**  
   - QMIX 的 **IGM性质** 保证了去中心化贪婪策略的一致性
   - GNN 编码当前通信图结构，GRU 维持历史状态，形成“拓扑记忆”

3. **动态拓扑显著提升效率与鲁棒性**
   - **Token 效率极高**：在 MMLU-Pro 上仅用 **112K tokens**，远低于其他多智能体方法（471K–2.71M）
   - **对抗攻击下最稳健**：注入一个恶意智能体后，准确率仅下降 **2.86%**（LLM-Debate 下降8.57%）

4. **策略会根据任务自动调节通信密度**
   - 简单任务（如 LiveCode）倾向于 `solo_process`，减少通信开销
   - 困难任务（如 AIME）启用 `broadcast_all` 实现充分协作

---

### 方法的局限性
1. **动作空间固定为6种**，虽具通用性，但可能不足以覆盖极端复杂的交互模式。
2. 当前实验集中在中小规模团队（N=3），扩展到更大群体时通信组合爆炸问题仍待研究。
3. 所有实验基于 API 调用模式，未考虑本地部署延迟与成本建模。

---

### 未来工作方向
1. **动态扩展动作空间**：引入自然语言指令作为通信动作输入，增强表达能力。
2. **异构智能体支持**：不同角色拥有不同的动作子集，更贴近真实应用场景。
3. **长期记忆与元学习**：让智能体在多个任务间积累通信经验，实现跨任务迁移。
4. **结合工具调用与外部环境反馈**：构建更完整的 Agentic Workflow 生态。

---

> 🔗 **代码开源地址**：[https://github.com/ericjiang18/Agent-Q-Mix](https://github.com/ericjiang18/Agent-Q-Mix)

</details>

---

### 4. [Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention](https://arxiv.org/abs/2604.00754)

**Authors**: Zehao Jin, Yanan Sui  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.00754v1  

#### Abstract
The whole-brain connectome of a fruit fly comprises over 130K neurons connected with a probability of merely 0.02%, yet achieves an average shortest path of only 4.4 hops. Despite being highly structured at the circuit level, the network's long-range connections are broadly distributed across brain ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统 **Sliding Window Attention (SWA)** 虽然在计算复杂度上为 $O(nw)$，具有良好的效率，但由于其**局部性限制**，每层只能覆盖固定大小的邻域，导致经过 $l$ 层后最大感受野仅为 $lw$。当序列长度 $n \gg w$ 时，大量上下文无法被访问，严重限制了模型的表达能力（expressivity），尤其对需要长程依赖的任务（如 LAMBADA）表现不佳。

### 提出了什么新方法或新思路
受果蝇全脑连接组（**fruit fly connectome**）启发，作者提出 **Stochastic Attention (SA)**，一种无需额外可学习参数的 SWA 增强机制：

- **核心思想**：在每层 SWA 前，对 token 序列进行**随机排列（random permutation）**；在注意力计算后，再将输出恢复至原始顺序。
- 这使得原本固定的局部窗口在全局序列中变为一个**随机子集**，从而以相同 $O(nw)$ 成本实现“**随机全局路由**”。
- 多层堆叠下，独立采样的排列使感受野呈**指数增长**，仅需 $O(\log_w n)$ 层即可实现全序列覆盖（而 SWA 需 $O(n/w)$ 层）。
- 进一步提出 **gated SA + SWA** 结构，通过两个独立的 sigmoid gate 融合 SA 和 SWA 路径，模拟 connectome 中“局部聚类 + 长程捷径”的小世界（small-world）特性。

### 相比现有方法的优势
| 特性 | SA | SWA | Full Attention | MoBA |
|------|----|-----|----------------|-------|
| 时间复杂度 | $O(nw)$ | $O(nw)$ | $O(n^2)$ | $O(nw)$ |
| 感受野增长 | 指数级 ($O(\log n)$) | 线性 ($O(lw)$) | 全局（1层） | 受限于块选择策略 |
| 参数增加 | 无（单路径） | 无 | 无 | 有（top-k 路由逻辑） |
| 实现难度 | 极低（仅加 permute 操作） | 标准 | 标准 | 较高 |
| 可作为训练后替换 | ✅ 是 | ✅ 是 | ❌ 否 | ✅ 是 |

> ✅ **SA 是一种即插即用（drop-in）、无参数、硬件友好**的增强模块，兼容任何基于 windowed 或 linear attention 的架构。

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **预训练语言建模任务**：
   - 数据集：**SlimPajama** 子集（6B tokens，训练约 15B tokens）
   - 模型规模：~360M 参数的 decoder-only Transformer
2. **训练后推理任务（training-free inference）**：
   - 模型：**Qwen3-8B** 和 **Qwen3-30B-A3B**
   - 评估基准（通过 `lm-evaluation-harness`）：
     - **LAMBADA**（完形填空，需长程语义）
     - **MMLU**（多学科知识问答）
     - **BoolQ**, **ARC-Easy/Challenge**, **HellaSwag**, **HumanEval**（代码生成）

### 实验设置和评估指标
#### （1）预训练实验
- **序列长度**：2048
- **窗口大小 $w$**：256
- **层数**：24
- **评估方式**：zero-shot evaluation
- **指标**：
  - WikiText：**Perplexity (↓)**
  - 其他任务：**Accuracy (↑)**

#### （2）训练后推理实验
- 在已训练好的 Qwen3 模型上**仅修改 attention mask**，不更新权重
- **Prefill 阶段使用不同 attention 模式**，decode 阶段仍使用 full KV-cache attention
- **对比模式**：
  - Full Attention（基线）
  - SWA（滑动窗口）
  - **Stochastic (SA)**（本文方法）
  - **MoBA (k=2)**（Mixture of Block Attention，块级路由）
- **有效窗口大小**：从 16 到 512 不等
- **位置编码处理**：SA 中 RoPE 使用原始位置，而非排列后位置，确保与预训练一致

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### （1）预训练实验（360M 模型，zero-shot 平均准确率）

| Model | Wiki.ppl ↓ | LAMBADA ↑ | Avg Acc ↑ |
|-------|------------|-----------|-----------|
| Full Attention | 51.34 | 19.5 / 15.4 | 34.9 |
| SWA (w=256) | 57.05 | 21.3 / 17.0 | 35.1 |
| SA (w=256) | 75.83 | 20.1 / 14.1 | 34.3 |
| **SA+SWA (ours)** | **51.98** | **22.8 / 17.6** | **35.9** |

> 🔍 **关键发现**：
> - 单独 SA 表达能力强（下游任务尚可），但破坏局部连贯性，导致 **ppl 显著升高**
> - **SA+SWA 融合结构取得最优结果**：既保持了 SWA 的低 perplexity，又通过 SA 提升了 LAMBADA 等任务的表现，**平均准确率最高**

#### （2）训练后推理（Qwen3-8B / 30B-A3B）

##### 总体趋势（图4）：
- **Stochastic Attention 最快逼近 full attention 基线**
- 在 **weff=128 时，Qwen3-8B 达到 70.9%**（接近 full 的 71.5%），而 SWA 仅 62.2%
- 在 **weff=64 时，Qwen3-30B-A3B 上 SA 达 73.2%**，远超 SWA（47.0%）和 MoBA（66.3%）

##### 细分任务优势（Tables 4–5）：
- **MMLU、BoolQ、LAMBADA** 等需跨上下文整合的任务中，SA 明显优于 SWA 和 MoBA
- 尤其在 **极小窗口（weff=32）下，SWA 几乎崩溃（MMLU: 29.0 → 34.9）**，而 SA 仍维持较高性能（44.4 / 52.0），证明其**全局信息流动更鲁棒**

##### 对比 MoBA：
- 在相同有效窗口下，**SA 持续领先 MoBA 3–7 个百分点**
- 例如 Qwen3-8B @ weff=128：
  - SA: **70.9%**
  - MoBA: **70.7%**
  - SWA: 62.2%

> ✅ **SA 在更低或相等计算预算下，性能持平甚至超越 MoBA**

### 消融实验结果
- **单独 SA vs. 单独 SWA**：验证了二者互补性 —— SA 提供全局覆盖，SWA 提供局部一致性
- **SA+SWA 融合门控设计**：使用两个独立 sigmoid gate 效果优于 softmax gate，允许模型动态调节两条路径的重要性
- **理论分析支持**：
  - SA 是 uniform full attention 的近似无偏估计（bias $O(1/w)$，variance $O(B/w)$）
  - 多层组合下感受野呈指数扩展，$O(\log_w n)$ 层可达全连接

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **connectome 启发的设计是有效的**：果蝇大脑中“局部密集连接 + 分布式长程捷径”结构可通过随机排列机制在 attention 中复现。
2. ✅ **随机路由显著提升表达能力**：SA 在不增加参数和复杂度的前提下，使 SWA 从“确定性局部”跃迁为“随机全局”，极大增强了 long-range 建模能力。
3. ✅ **SA+SWA 实现小世界平衡**：结合局部（SWA）与随机全局（SA）路径，在 perplexity 与 downstream accuracy 之间取得最佳权衡。
4. ✅ **训练后即插即用有效**：即使在 full attention 预训练的模型上，SA 也能作为高效推理替代方案，快速恢复 full attention 性能。

### 方法的局限性
- **随机性引入方差**：单次 SA 输出存在波动，虽期望上接近 uniform attention，但在低 $w$ 下 variance 较大。
- **对极端短窗口仍有限制**：尽管优于 SWA，但在 $w < 16$ 时性能仍有明显下降。
- **未探索结构化排列**：当前使用完全随机排列，未来可研究更智能的分布（如基于 entropy 或 saliency）。

### 未来工作方向
- 探索 **structured stochastic routing**，例如基于 content-aware 的轻量级排列生成
- 将 SA 扩展至 **vision Transformer 和 multimodal 模型**
- 研究 SA 在 **long-sequence continual pre-training** 中的应用潜力
- 分析 SA 对 **KV cache 压缩与重用** 的影响，进一步优化推理效率

---

> 📌 **一句话总结**：  
> *Stochastic Attention* 通过 connectome 启发的随机排列机制，将 SWA 转化为具有指数级感受野的高效全局注意力，在保持 $O(nw)$ 成本的同时显著提升表达能力，是一种简单、通用且高效的 attention 增强范式。

</details>

---

### 5. [TENT: A Declarative Slice Spraying Engine for Performant and Resilient Data Movement in Disaggregated LLM Serving](https://arxiv.org/abs/2604.00368)

**Authors**: Feng Ren, Ruoyu Qin, Teng Ma, Shangming Cai, Zheng Liu, Chao Lei, Dejiang Zhu, Ke Yang, Zheming Li, Jialei Cui, Weixiao Huang, Yikai Zhao, Yineng Zhang, Hao Wu, Xiang Gao, Yuhao Fu, Jinlei Jiang, Yongwei Wu, Mingxing Zhang  
**Category**: cs.DC  
**Published**: 2026-04-02  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.00368v1  

#### Abstract
Modern GPU clusters are built upon a complex hierarchy of heterogeneous interconnects, ranging from multi-rail RDMA to proprietary fabrics such as Multi-Node NVLink and Ascend UB. Orchestrating these diverse links effectively remains a critical challenge in disaggregated LLM serving. Operating Moonc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TENT: A Declarative Slice Spraying Engine for Performant and Resilient Data Movement in Disaggregated LLM Serving

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

现代大规模 GPU 集群依赖复杂的异构互联架构（如 RDMA、NVLink、MNNVL、Ascend UB 等），但在**解耦式大语言模型（disaggregated LLM）服务**场景下，现有的数据传输引擎存在以下三大瓶颈：

1. **静态绑定导致通信孤岛（Static Binding Creates Silos）**  
   现有框架（如 Mooncake TE、NIXL、UCCL）在初始化时将应用与特定传输后端（如 NVLink 或 RDMA）绑定，导致不同硬件配置的集群无法互通，形成多个隔离的通信域。

2. **无状态条带化浪费多轨带宽（State-Blind Striping Wastes Bandwidth）**  
   多轨（multi-rail）设计本应提升总带宽，但现有系统采用轮询或哈希等静态分片策略，忽略链路拥塞和延迟变化，导致“最慢轨决定整体性能”（Head-of-Line Blocking），尾部延迟高。

3. **脆弱的执行机制需人工干预（Fragile Execution Requires Manual Intervention）**  
   当 NIC 故障、链路降级或设备重启时，现有引擎通常将错误暴露给上层应用，依赖手动恢复或全局重建通信器，造成服务中断和性能长期下降。

---

### **提出了什么新方法或新思路**

论文提出 **TENT** —— 一种**声明式切片喷洒（Declarative Slice Spraying）引擎**，实现高性能、弹性的数据移动。其核心思想是：**将传输意图（intent）与物理执行解耦**。

#### 主要创新点包括：

- ✅ **从静态绑定到动态编排（Dynamic Orchestration）**  
  TENT 将所有异构互连抽象为统一的资源池，在运行时根据拓扑和链路状态动态选择最优路径，支持跨设备的多跳中继路由（staged routing），无需修改应用代码。

- ✅ **从无状态条带化到遥测驱动的切片喷洒（Telemetry-Driven Slice Spraying）**  
  将大象流（elephant flows）拆分为细粒度切片（如 64KB），基于实时遥测（排队长度、完成时间）预测每条链路的服务时间，并动态调度切片至最快可用路径，避免 HoL 阻塞。

- ✅ **从脆弱执行到弹性自愈（Resilient Self-Healing）**  
  在数据平面内嵌入双层容错机制：
  - **链路层**：自动屏蔽故障轨，通过后台探测逐步恢复；
  - **传输层**：支持后端替换（如 NVLink → RDMA），透明切换而不中断传输。

- ✅ **声明式 API 接口**  
  应用只需声明“移动哪些数据段”，由 TENT 决定“如何移动”。这种解耦极大提升了可移植性和运维效率。

---

### **相比现有方法的优势**

| 维度 | 现有方法（Mooncake TE / NIXL / UCCL） | TENT |
|------|----------------------------------------|------|
| 路径选择 | 静态绑定，编译期/启动期确定 | 动态编排，请求级决策 |
| 分片策略 | 固定轮询或哈希，无视链路状态 | 遥测驱动，按预期完成时间调度 |
| 容错能力 | 控制面处理，需上层重试或重建 | 数据面自愈，子毫秒级重试 |
| 可移植性 | 不同 fabric 需不同镜像和配置 | 统一接口，仅更换 backend 配置 |
| 性能表现 | 易受最慢轨拖累，尾延迟高 | 充分利用聚合带宽，尾延迟低 |

---

## 2. 核心实验方法和设置

### **实验平台与硬件环境**

- **主测试床**：  
  - 节点配置：8× NVIDIA H800 GPU（80GB HBM）、8× 200 Gbps RoCE NIC、双路 Intel Xeon Platinum 8468V CPU
  - 互联结构：节点内通过 NVLink 连接，NIC 通过 PCIe 拓扑连接，跨 NUMA 域
  - 构成典型的异构 fabric：NVLink + PCIe + multi-rail RDMA

- **其他验证平台**：  
  - Multi-Node NVLink (MNNVL) 集群
  - Ascend UB/HIXL 集群
  - 支持 io_uring 的高性能 NVMe SSD 存储节点

---

### **评估指标**

| 类别 | 指标 |
|------|------|
| **性能** | 吞吐量（Throughput）、P90/P99 TTFT（Time-To-First-Token）、TPOT（Time-Per-Output-Token） |
| **微基准** | 读写吞吐、P99 延迟、并发扩展性 |
| **可靠性** | 故障恢复时间、吞吐波动持续时间、是否触发应用重试 |
| **可移植性** | 跨 fabric 性能损失、backend 实现复杂度 |

---

### **基线方法对比**

- **Mooncake Transfer Engine (TE)**：作者团队前代生产级引擎，使用 round-robin 条带化
- **NIXL**：NVIDIA 基于 UCX 的跨 NIC 传输框架
- **UCCL-P2P**：UCCL 的点对点传输后端
- 所有基线均运行在相同硬件和 RDMA 配置下

---

### **应用场景与负载**

1. **LLM 推理 + KVCache 复用（SGLang HiCache）**  
   - 模型：Qwen3-235B-A22B-Instruct-2507（TP=8）
   - 负载：60 客户端，每请求 2048 输入 token，10 轮对话
   - 缓存策略：600GB KVCache，启用 HiCache 多级缓存

2. **强化学习流水线中的参数更新（Moonshot Checkpoint Engine）**  
   - 场景：频繁模型权重同步（refresh window 要求短）
   - 模型：Qwen3-235B、GLM-4.5-Air、DeepSeek-V3.1、Kimi-K2-Instruct

3. **微基准测试（TEBench）**  
   - 测试模式：host-to-host、GPU-to-GPU
   - 参数范围：block size 4KB ~ 64MB，thread 数 1~64，batch size 1~128

---

## 3. 主要实验结果和性能指标

### **关键性能数据汇总**

| 场景 | 指标 | TENT 表现 | 对比基线（Mooncake TE） |
|------|------|-----------|--------------------------|
| LLM 推理（HiCache） | 输入吞吐（tok/s） | **78,759** | 58,006 (+35.7%) |
| | P90 TTFT（s） | **0.67** | 0.90 (-25.6%) |
| | 平均 TTFT（s） | **0.53** | 0.72 (-26.4%) |
| | 第10轮平均 TTFT（s） | **0.66** | 0.97 (-32.0%) |
| 参数更新（Checkpoint Engine） | 更新耗时（Qwen3-235B） | **10.34s** | 12.87s (-19.7%) |
| | 更新耗时（GLM-4.5-Air） | **5.30s** | 7.17s (-26.1%) |
| 微基准（host-to-host） | 写吞吐峰值 | **33.7% 更高** | vs Mooncake TE |
| | P99 延迟 | **降至 27.6%** | vs Mooncake TE |
| | GPU-to-GPU 写吞吐 | **2.1× 更高** | vs Mooncake TE |
| | P99 延迟 | **降至 46.7%** | vs Mooncake TE |

> 🔺 **最高收益**：在 KVCache 密集型推理中，TENT 实现 **1.36× 吞吐提升** 和 **26% P90 TTFT 下降**

---

### **与基线方法的对比结果**

#### ✅ **性能优势来源分析**

- **NVLink 优先利用**：TENT 自动识别并优先使用 NVLink 进行节点内 GPU-GPU 通信，而 Mooncake TE 默认走 RDMA。
- **动态负载均衡**：TENT 的调度器能感知 Tier-1 NIC 饱和情况，在大块传输时主动引入 Tier-2 NIC，实现带宽叠加。
- **避免 Hot Rail**：传统方法因固定映射导致某 NIC 成为瓶颈；TENT 通过遥测避开拥堵链路。

#### ✅ **并发扩展性优异**

- 在 16 线程时即达到峰值带宽（144 GB/s），远超 Mooncake TE 和 NIXL
- 单线程 + 大 batch 下仍接近理论极限（800 Gbps），说明调度开销极低

#### ✅ **消融实验（Sensitivity Analysis）**

- **调度参数鲁棒性强**：即使 NUMA 惩罚系数 `P1` 设置不当（过大或过小），性能下降有限，反馈机制可自动修正
- **默认 `P1=3` 达到最优平衡**：兼顾 Tier-1 优先与 Tier-2 利用

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **声明式 + 动态编排显著优于静态绑定**  
   将路径选择推迟到运行时，使系统能够适应异构、动态变化的硬件环境，打破通信孤岛。

2. ✅ **细粒度切片 + 遥测驱动调度有效消除 HoL 阻塞**  
   基于排队深度和历史完成时间的预测模型，实现了真正的“智能流量调度”，最大化多轨利用率。

3. ✅ **数据面自愈机制可将常见硬件故障转化为短暂性能扰动**  
   实验显示，单个 NIC 故障仅引起 <50ms 的吞吐下降，且无需应用层介入即可自动恢复。

4. ✅ **高度可移植且开销极低**  
   各 backend 实现 <800 LOC，核心逻辑复用，跨 fabric 性能损失可忽略（如 NVLink 达到理论带宽 84%）。

5. ✅ **已在工业界大规模部署验证**
   - 某金融科技公司：全企业级 LLM 推理底座，SLA：TTFT <1s，TPOT <30ms
   - 某头部 AI 服务商：千卡集群，峰值处理 >50M tokens/min

---

### **方法的局限性**

| 局限 | 说明 |
|------|------|
| **依赖底层 transport 支持** | 若某 fabric 不支持 RDMA 或 one-sided ops，则无法发挥零拷贝优势 |
| **调度模型需一定 warm-up 时间** | 初始阶段预测误差较大，但可通过周期性重置缓解 |
| **极端拥塞下可能误判健康状态** | 如整个交换机拥塞，可能导致健康链路被误排除，但概率较低 |

---

### **未来工作方向**

1. **支持更广泛的存储介质**  
   如集成 NVMe-oF、CXL.mem 设备作为统一 segment 抽象的一部分。

2. **跨集群广域调度**  
   将 TENT 的理念扩展到多数据中心场景，实现 geo-distributed LLM serving 的高效数据移动。

3. **与 MoE、EP 等计算范式深度融合**  
   结合 Expert Parallelism 中的动态路由需求，提供端到端的数据+计算协同优化。

4. **自动化参数调优**  
   引入 ML-based scheduler 替代当前启发式模型，进一步提升调度精度。

---

> 🌟 **总结一句话**：  
> **TENT 通过“声明式意图 + 动态切片喷洒 + 数据面自愈”，构建了一个面向未来 disaggregated AI 架构的高性能、高可靠、高可移植的数据移动引擎，已在多个千卡级生产环境中验证其优越性。**

> 🔗 开源地址：[https://github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)

</details>

---

### 6. [A Decoupled Basis-Vector-Driven Generative Framework for Dynamic Multi-Objective Optimization](https://arxiv.org/abs/2604.00508)

**Authors**: Yaoming Yang, Shuai Wang, Bingdong Li, Peng Yang, Ke Tang  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.00508v1  

#### Abstract
Dynamic multi-objective optimization requires continuous tracking of moving Pareto fronts. Existing methods struggle with irregular mutations and data sparsity, primarily facing three challenges: the non-linear coupling of dynamic modes, negative transfer from outdated historical data, and the cold-...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Decoupled Basis-Vector-Driven Generative Framework for Dynamic Multi-Objective Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **Dynamic Multi-Objective Optimization Problems (DMOPs)** 中存在的三大挑战提出解决方案：
- **非线性耦合动态模式**：环境变化通常是非线性的、复杂的拓扑变换，传统预测模型难以建模。
- **负迁移（Negative Transfer）**：记忆型方法在面对非周期性或随机环境时容易因历史数据误导而导致性能下降。
- **冷启动问题（Cold-Start Problem）**：环境切换后缺乏足够评估解，导致在线学习延迟。

这些问题共同限制了现有 **Dynamic Multi-Objective Evolutionary Algorithms (DMOEAs)** 在不规则突变、稀疏数据和复杂拓扑下的跟踪能力。

---

### 提出的新方法与新思路
作者提出了一个名为 **Decoupled Basis-Vector-Driven Generative Framework (DB-GEN)** 的生成式框架，其核心思想是将动态优化从“在线自回归预测”转变为“零样本（zero-shot）潜在流形生成”。

#### 主要创新点包括：

- **频率解耦机制（Frequency Decoupling）**
  - 引入 **Discrete Wavelet Transform (DWT)** 将进化轨迹分解为低频趋势（全局移动）和高频细节（局部扰动），从而降低学习复杂度并分离可预测模式与噪声。

- **基于可迁移基向量的结构学习（Transferable Basis Learning）**
  - 不直接记忆历史个体，而是通过 **Sparse Dictionary Learning** 学习一组通用的“物理基向量”（basis vectors），代表环境演化的基础动态模式。
  - 利用 **topology-aware contrastive loss**（基于 IGD 的三元组损失）对齐潜在空间几何结构，确保语义一致性。

- **跨任务零样本生成搜索（Zero-Shot Generative Search）**
  - 模型在包含 **1.2亿个历史解** 的大规模离线数据集上预训练，无需在线微调即可执行推理。
  - 在环境变化发生时，通过冻结模型进行 **surrogate-assisted decoding**，从潜在流形中采样高质量初始种群，实现毫秒级响应（约 0.2 秒/次切换）。

---

### 相比现有方法的优势
| 维度 | 传统方法（如记忆、预测、多群体） | DB-GEN |
|------|-------------------------------|--------|
| **泛化能力** | 局部适应，易受负迁移影响 | 跨问题泛化，基于组合式基向量重构 |
| **冷启动应对** | 需重新初始化或等待收敛 | 零样本生成，立即提供高质量种子 |
| **计算效率** | 多数需在线训练或重优化 | 离线训练 + 在线快速推断（~0.2s） |
| **鲁棒性** | 对非线性/不规则变化敏感 | 显著提升在 DF6、DF9 等难例上的表现 |

---

## 2. 核心实验方法和设置

### 数据集
- **合成基准测试套件**：
  - **FDA test suite** [Farina et al., 2004]
  - **CEC 2018 DMOP suite (DF1–DF14)** [Jiang et al., 2018]
- **真实世界启发问题**：
  - **Dynamic Resource Allocation (DRA)**
  - **Dynamic Path Planning (DPP)**

所有测试共涉及 **57 个动态配置**（不同 `T` 和 `n` 参数组合）。

---

### 实验设置与评估指标

#### 参数设置
- 变化频率 $ T \in \{5, 10\} $
- 变化强度 $ n \in \{5, 10\} $
- 每个算法运行 **20 次独立实验**

#### 评估指标
| 指标 | 含义 | 越小越好 / 越大越好 |
|------|------|---------------------|
| **MIGD (Mean Inverted Generational Distance)** | 衡量解集收敛性和多样性 | 越小越好 |
| **MHV (Mean Hypervolume)** | 衡量目标空间覆盖范围 | 越大越好 |

采用 **Wilcoxon 秩和检验** 和 **Friedman 检验** 进行统计显著性分析。

---

### 基线方法对比
选取四类主流 SOTA 方法作为对比：
1. **STT-MOEA/D**：基于时空拓扑张量预测
2. **DIP-DMOEA**：基于方向改进预测
3. **VARE**：向量自回归演化模型
4. **SIKT-DMOEA**：基于相似性识别的知识迁移

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ MIGD 结果（越小越好）
- 在 **57 个测试实例中**：
  - DB-GEN 获得 **49 次最优**（深灰），**5 次次优**（浅灰）
  - Top-2 占比高达 **94.7%**
- **平均排名为 1.29**（理论最优为 1.0），远超其他方法（2.82~4.30）

#### ✅ MHV 结果（越大越好）
- 在 FDA 子集上：
  - 平均排名达 **1.20**
  - 显著优于所有基线

---

### 与基线方法的对比结果

| 问题 | DB-GEN vs 最佳基线（DIP-DMOEA） |
|------|------------------------------|
| **DF6 ($n_t=10, T_t=5$)** | IGD 从 **1.27 → 0.426**，**降低 66.4%** |
| **DF9 ($n_t=10, T_t=5$)** | IGD 从 **0.278 → 0.080**，**降低 71.2%** |
| **DRA** | IGD 从 **3.63E-02 → 1.35E-02**，**降低 62.8%** |

> 特别是在高难度、非线性剧烈变化场景下，DB-GEN 显示出极强的鲁棒性。

---

### 消融实验结果（Ablation Study）

在 57 个环境中移除关键模块后的性能退化情况如下：

| 变体 | MIGD | 相对退化 | 胜/负/平 |
|------|------|----------|---------|
| **Full Model** | 0.0832 | — | — |
| w/o High-Low Freq | 0.0983 | +18.1% | 4/53/0 |
| w/o Basis Learning | 0.1205 | **+44.8%** | 14/43/0 |
| w/o Solution VAE | 0.1028 | +23.6% | 23/34/0 |
| w/o Triplet Loss | 0.1048 | +26.0% | 15/42/0 |
| w/o Classify Loss | 0.0934 | +12.3% | 17/40/0 |

> **结论**：
> - **Basis Learning 是最关键组件**，移除后性能下降最严重；
> - 所有模块均有正向贡献，验证了架构设计的整体必要性。

---

## 4. 关键结论和发现

### 主要发现
1. **频率解耦有效分离动态模式**  
   DWT 成功提取出低频趋势与高频波动，使模型能聚焦于可预测部分，避免被噪声干扰。

2. **基向量具有明确语义解释性**  
   - t-SNE 可视化显示不同问题形成连续轨迹；
   - 掩码实验证明某些维度专门控制“曲率”、“位移”等物理属性；
   - 支持 **组合式泛化**（compositional generalization），例如 DF6 可由 Problem 64（曲率）和 Problem 67（位移）插值得到。

3. **零样本生成显著缓解冷启动问题**  
   - 新环境下仅用 0.2 秒即可生成高质量初始种群；
   - IGD 从继承种群的 0.155 下降至 0.0488（DF1 示例）；
   - 相比逐个扰动旧个体策略（IGD→0.1106），**centroid-based perturbation 更稳定高效**。

4. **数据规模带来持续收益**  
   - 性能随历史数据量增加单调上升（$4\times10^7 \to 1.2\times10^8$）；
   - Friedman 检验 $p=2.48\times10^{-18}$，表明差异高度显著；
   - 表明更大的先验密度有助于构建更稳健的潜在流形。

---

### 方法的局限性
1. **无法处理完全正交的新动态模式（OOD Failure）**
   - 如 DF7 问题引入了超出字典张成空间的全新拓扑变化，导致性能下降；
   - 当前框架依赖已有 basis 的线性组合，不能创造全新 primitive。

2. **静态超参数配置**
   - 字典大小 $K$、扰动半径 $\sigma$、候选池大小 $N_{cand}$ 固定；
   - 缺乏自适应机制以应对不同严重程度的变化。

3. **当前评估集中于低维无约束问题**
   - 尚未扩展至高维、带约束的实际工程场景。

---

### 未来工作方向
1. **开发自适应调节机制**
   - 动态调整 $K$ 或 $\sigma$ 以匹配当前环境变化强度。

2. **拓展至高维与约束 DMOPs**
   - 结合 dimensionality reduction 或 space discretization 技术（如 [52] 所提方法）。

3. **增强 OOD 泛化能力**
   - 引入稀疏增量学习机制，在线更新字典以容纳新动态模式。

4. **应用于更多现实系统**
   - 如自动驾驶调度、电网负载均衡、LLM 参数融合等动态权衡场景。

---

> **总结一句话**：  
> DB-GEN 通过 **频率解耦 + 基向量学习 + 零样本生成** 的三阶段范式，实现了对复杂动态多目标问题的高效、鲁棒、可解释的求解，在多个维度上超越了现有 SOTA 方法，为下一代 DMOEA 提供了新的研究路径。

</details>

---

### 7. [SAGE: Subsurface AI-driven Geostatistical Extraction with proxy posterior](https://arxiv.org/abs/2604.00307)

**Authors**: Huseyin Tuna Erdinc, Ipsita Bhar, Rafael Orozco, Thales Souza, Felix J. Herrmann  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.00307v1  

#### Abstract
Recent advances in generative networks have enabled new approaches to subsurface velocity model synthesis, offering a compelling alternative to traditional methods such as Full Waveform Inversion. However, these approaches predominantly rely on the availability of large-scale datasets of high-qualit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SAGE: Subsurface AI-driven Geostatistical Extraction with proxy posterior》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
传统地震反演方法如 **Full Waveform Inversion (FWI)** 虽然能估计地下速度模型，但其高度非线性、病态且对初始模型敏感。近年来基于生成网络的方法通过学习地质先验来辅助反演，但这些方法严重依赖大规模、高质量、全观测的 **velocity models** 作为训练数据，而现实中这类数据稀缺，尤其是仅存在稀疏 **well logs** 和迁移后的 **seismic images (如 RTM 图像)**。

SAGE 正是为了解决“**在没有完整 velocity 模型的情况下如何学习地质统计先验**”这一关键挑战。

---

### 提出了什么新方法或新思路
提出 **SAGE (Subsurface AI-driven Geostatistical Extraction)** 框架，一种从不完全观测中学习代理后验（proxy posterior）的新范式：

- **学习目标**：在训练阶段利用成对的部分观测 velocity（即 well logs）和对应的 **migrated seismic image (RTM)** 来学习一个条件化的代理后验分布 $ p(x|y) $，其中 $ x $ 是 velocity 场，$ y $ 是 RTM 图像。
- **推理阶段**：仅需输入 migrated image 即可生成高分辨率、地质合理的 velocity 场，**well log 信息被隐式编码在训练学到的分布中**。
- **技术实现**：采用基于 **conditional score-based diffusion models** 的 simulation-based inference (SBI)，并引入双重掩码机制（原始掩码 + 子采样掩码）防止退化解（如直接复制观测值），从而实现全局一致的重建。

---

### 相比现有方法的优势
| 维度 | 传统方法 / 其他生成模型 | SAGE |
|------|------------------------|------|
| 数据需求 | 需要大量完整 velocity 模型 | 仅需稀疏 well logs + migrated images |
| 实用性 | 在真实场景受限于数据获取 | 更适用于实际勘探场景 |
| 推理效率 | 多数需迭代优化 | amortized inference，一次前向即可采样 |
| 下游应用支持 | 多为独立模型 | 可作为数据生成器用于训练其他任务（如 WISE） |

> ✅ **核心优势**：实现了**无需完整标签数据**下的地质先验学习，打通了从有限观测到高保真 velocity 合成的路径。

---

## 2. 核心实验方法和设置

### 使用的数据集
#### （1）合成数据集（Synthetic Dataset）
- 来源：基于 **Compass 模型** 切片得到的 2D 地球声波速度模型（共 1000 个 realization）
- 尺寸：256 × 512 网格，空间分辨率为 12.5 米（覆盖 3.2km × 6.4km）
- Seismic 模拟：
  - 16 个震源，256 个接收器
  - Ricker wavelet（主频 20Hz），记录时长 3.2 秒
  - 添加 10dB 彩色高斯噪声
- 成像：使用平滑背景模型进行 **Reverse-Time Migration (RTM)** 得到 migrated image
- 观测模拟：每条 velocity 模型只保留 **5 列（out of 256）** 作为 well-log 类似观测（约 99% 缺失）

#### （2）真实数据集（Field Data）
- 来源：英国 **UK National Data Repository** 提供的真实 seismic 和 well-log 数据
- 预处理流程：
  - Checkshot 时间-深度转换
  - Well tie 对齐
  - 提取 quasi-2D 测线
  - 运动学一致性下采样
- Well 数量：仅有 **40 口井**，极为稀疏
- 策略：采用 **fine-tuning** —— 在 synthetic Compass 模型上预训练的 SAGE 模型迁移到真实数据微调

---

### 实验设置和评估指标
- **网络架构**：U-Net 作为 denoiser
- **训练时间**：约 20 GPU 小时
- **输入形式**：concatenate noisy partial velocity + RTM image + mask + noise level
- **训练策略**：Algorithm 1 中描述的 subsampling-based objective，防止 trivial solution
- **评估方式**：
  - 定量：**SSIM (Structural Similarity Index Measure)** 衡量 posterior mean 与 ground truth 的结构相似性
  - 定性：可视化 posterior mean 与 standard deviation，分析不确定性分布
  - 下游任务验证：将 SAGE 生成的 samples 用于训练 **WISE** 框架，比较其性能下降程度

---

### 基线方法对比
本文未直接与其他端到端 velocity 生成模型对比（因多数需要完整训练数据），而是通过以下方式体现优越性：
- 与 **ground truth** 对比（尽管训练时不接触）
- 与 **WISE 使用真实 velocity 训练的结果** 对比，衡量 SAGE 生成样本的质量
- 展示 **real data 上的初步结果**，证明泛化能力

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 实验场景 | 性能指标 | 结果 |
|---------|--------|------|
| Synthetic 数据推理 | Posterior Mean vs GT 的 SSIM | **0.82 ~ 0.84** |
| WISE 使用 SAGE 样本训练 | Posterior Mean SSIM | **0.84** |
| WISE 使用真实 velocity 训练 | Posterior Mean SSIM | **0.88** |
| Real Data 推理 | —— | 定性显示生成 velocity 与 migrated image 和独立 well logs 一致 |

> 🔍 注：SSIM > 0.8 已表示良好的结构保持能力，尤其考虑到训练中从未见过完整模型。

---

### 与基线方法的对比结果
- **SAGE vs. Ground Truth Training (间接对比)**：
  - 当用于训练 WISE 时，使用 SAGE 生成数据导致 SSIM 仅从 **0.88 降至 0.84**，表明生成样本具有足够高的地质保真度。
- **SAGE 在 real data 上的表现**：
  - 生成的速度场展现出复杂地质结构，与 migrated image 特征吻合，并与未参与推理的独立 well logs 匹配良好（见 Figure 4 trace comparison）。
  - 显示出强大的跨域适应能力和 fine-tuning 有效性。

---

### 消融实验结果（如有）
文中虽未明确列出消融表，但在方法设计中有两个关键组件起到类似作用：
1. **Subsampling Mask ($\tilde{A}$)**：
   - 若无此机制，模型会收敛至 trivial 解（仅恢复已知列，忽略图像条件）。
   - 引入后迫使网络学习全局结构重建能力。
2. **Proxy Posterior Learning from Partial Observations**：
   - 成功证明即使只有 5/256 列观测，也能学习出有意义的后验分布，说明框架对极端稀疏观测鲁棒。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **可以从稀疏 well logs 和 migrated images 中有效学习 velocity 场的代理后验分布**，无需访问完整的 ground truth velocity 模型。
2. ✅ SAGE 能够生成**地质合理、结构一致、带不确定性量化**的 velocity realizations，在 synthetic 和 real 数据上均表现良好。
3. ✅ SAGE 生成的样本可用于训练下游任务（如 WISE），性能接近使用真实数据训练的效果，展示了其作为“**data surrogate generator**”的巨大潜力。
4. ✅ 框架支持 **efficient fine-tuning**，可在少量真实数据上快速适配，适合工业级部署。

---

### 方法的局限性
- ❌ 当前仅在 **2D 设置** 下验证，尚未扩展至三维（3D），而实际勘探多为 3D 场景。
- ❌ 对 migrated image 质量有一定依赖，若 migration 存在显著误差（如速度错误导致偏移不准），可能影响生成质量。
- ❌ 不确定性图反映的是模型认知不确定性，尚难完全解耦数据噪声与建模误差的影响。

---

### 未来工作方向
1. 🔄 扩展至 **3D SAGE** 框架，处理更大规模、更真实的三维地震数据。
2. 📦 构建更大规模的 **curated field datasets** 并开源，推动社区发展。
3. 🔗 探索与物理约束结合的方式（如嵌入波动方程残差），进一步提升生成模型的物理一致性。
4. 🤖 将 SAGE 集成进 end-to-end inversion pipeline，实现“learned prior + physics-based refinement”的混合范式。

---

> 📌 **总体评价**：  
> SAGE 是一项面向现实挑战的重要进展，它突破了传统生成模型对完整训练数据的依赖，开创了一种“**从不完整观测中自监督学习地质先验**”的新路径。该方法不仅本身具备实用价值，更为后续的 seismic imaging 与 inversion 提供了一个强大而灵活的 prior modeling 工具。  
> GitHub 开源地址：[https://github.com/slimgroup/SAGE](https://github.com/slimgroup/SAGE)

</details>

---

### 8. [Execution-Verified Reinforcement Learning for Optimization Modeling](https://arxiv.org/abs/2604.00442)

**Authors**: Runda Guan, Xiangqing Shen, Jiajun Zhang, Yifan Zhang, Jian Cheng, Rui Xia  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.00442v1  

#### Abstract
Automating optimization modeling with LLMs is a promising path toward scalable decision intelligence, but existing approaches either rely on agentic pipelines built on closed-source LLMs with high inference latency, or fine-tune smaller LLMs using costly process supervision that often overfits to a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Execution-Verified Reinforcement Learning for Optimization Modeling**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

自动化**优化建模**（Optimization Modeling）是将自然语言描述的实际决策问题转化为数学规划模型（如 LP、MILP）的关键步骤，广泛应用于供应链、能源系统等领域。然而，该任务高度专业化，传统方法面临以下挑战：

- **基于提示的方法**（Prompting-based）依赖闭源大模型（如 GPT-4o），推理成本高且延迟大。
- **监督微调方法**（SFT）依赖精细的**过程监督**（Process Supervision），即中间变量定义、公式推导等人工标注，标注成本高昂。
- 模型容易**过拟合特定求解器 API**（如 Gurobi），难以迁移到其他求解器（如 OR-Tools、COPT），缺乏跨求解器泛化能力。

---

### 🚀 提出的新方法：EVOM（Execution-Verified Optimization Modeling）

作者提出 **EVOM** ——一种基于**执行验证的强化学习框架**，其核心思想是：

> 将数学规划求解器（如 Gurobi）视为一个**确定性的交互式验证器**（Deterministic Interactive Verifier），通过“生成-执行-反馈-更新”闭环进行训练。

#### 核心机制：
- 输入：自然语言问题 `q` + 目标求解器 `s`
- 输出：包含 `<think>` 推理块 和 `<code>` 可执行代码块 的结构化响应
- 执行：在沙箱中运行生成的代码，获取求解状态（如 OPTIMAL、INFEASIBLE）、目标值 `v`
- 奖励：仅基于**最终执行结果是否匹配真实答案 `a`** 构造奖励信号（Outcome-only Reward）
- 优化：使用 GRPO 或 DAPO 进行无评论家策略优化（Critic-Free RL）

---

### 🔍 相比现有方法的优势

| 维度 | 传统方法（SFT） | EVOM（本文） |
|------|------------------|-------------|
| **监督信号** | 需要过程级标注（变量、约束、参考代码） | 仅需问题-答案对，无需中间标注 |
| **求解器迁移** | 过拟合特定求解器语法，零样本迁移失败 | 支持零样本求解器转移（Zero-shot Solver Transfer） |
| **适配新求解器** | 需重建数据集并重新训练 | 仅切换执行后端即可低代价适配（Low-cost Adaptation） |
| **可扩展性** | 数据构建成本高 | 利用执行反馈自动提供监督信号，更具可扩展性 |

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集

| 数据集 | 描述 |
|-------|------|
| **NL4OPT** | 来自 NeurIPS 2022 比赛，289 个线性规划问题，强调从自然语言到 LP 形式的转换 |
| **MAMO** | 包含 EasyLP 和 ComplexLP 子集，评估数学建模能力，涵盖 ODE、MILP 等 |
| **IndustryOR** | 工业级基准，来自 13 个行业的真实 OR 问题，更具实际意义 |
| **OptiBench** | 多样化的端到端优化问题，包含非线性规划和表格数据场景 |

> 所有数据均只保留 `(q, a)` 对，丢弃原始的过程级标注。

---

### ⚙️ 实验设置

- **基础模型**：Qwen2.5-7B（开源 LLM）
- **求解器支持**：Gurobi、OR-Tools、COPT
- **输出格式强制**：
  ```xml
  <think> ... 推理过程 ... </think>
  <code> ... 可执行 Python 代码 ... </code>
  ```
- **沙箱执行环境**：限制 10 秒超时、2GB 内存，捕获执行状态、日志、目标值
- **奖励函数设计**：
  - `R = τ_fmt(y) + τ_ans(o; a)`
  - `τ_fmt`: 格式合规奖励（标签完整性和正则匹配）
  - `τ_ans`: 结果正确性奖励（数值接近或状态一致）

---

### 📊 评估指标

- **Accuracy**：预测结果与真实答案 `a` 在相对误差容忍范围内（`ε_eval = 0.05` 或严格 `1e-4`）即为正确
- 正确条件：
  - 成功执行（无语法/运行错误）
  - 求解状态匹配（如 OPTIMAL）
  - 目标值满足精度要求

---

### 🆚 基线方法对比

| 类型 | 方法 |
|------|------|
| **Prompting-based** | DeepSeek-R1, OpenAI o1, GPT-4o（标准/Cot/CoE）, OptiMUS（代理系统） |
| **Training-based (SFT)** | ORLM（使用 OR-Instruct 全量过程标注微调） |

---

## 3. **主要实验结果和性能指标**

### 📈 主要性能对比（Table 1）

| 方法 | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndustryOR | Avg |
|------|-----------|--------|--------|--------|------------|-----|
| ORLM (SFT) | 60.96 | 84.89 | 88.34 | 35.71 | 27.00 | 59.38 |
| **EVOM (GRPO)** | **62.95** | **84.08** | **88.19** | **34.28** | **31.00** | **60.10** |

✅ **结论**：EVOM 在平均性能上**持平甚至略优于**依赖昂贵过程监督的 SFT 方法。

---

### 🔁 零样本求解器迁移（Zero-shot Solver Transfer）— Table 2

| 方法 | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndusOR |
|------|----------|--------|--------|--------|---------|
| ORLM (SFT) | 3.49 | 4.89 | 0.00 | 1.42 | 6.00 |
| **EVOM (GRPO)** | **54.31** | **77.55** | **84.81** | **22.27** | **24.00** |

📌 **关键发现**：
- SFT 模型因过拟合 Gurobi 语法，在 OR-Tools 上几乎完全失效；
- EVOM 凭借**结果导向的学习机制**，成功实现跨求解器泛化，展现出强大的“即插即用”能力。

---

### 🔧 低代价适配新求解器（Low-cost Adaptation）— Table 3

| 求解器 | 方法 | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndustryOR |
|--------|------|-----------|--------|--------|--------|------------|
| Gurobi | Base | 41.19 | 55.10 | 65.95 | 28.43 | 21.00 |
|        | **+EVOM** | **62.95 (+21.76)** | **84.08 (+28.98)** | **88.19 (+22.24)** | **34.28 (+5.85)** | **31.00 (+10.00)** |
| OR-Tools | Base | 40.86 | 59.18 | 53.83 | 18.48 | 20.00 |
|          | **+EVOM** | **52.49 (+11.63)** | **77.14 (+17.96)** | **82.82 (+28.99)** | **16.11 (-2.37)** | **31.00 (+11.00)** |

✅ **结论**：即使对于 OR-Tools 这类覆盖较少的求解器，EVOM 也能显著提升性能，尤其在 MAMO-E 上增益高达 **+28.99%**。

---

### 🔍 消融实验结果

#### （1）显式推理（`<think>` 块）的作用（Figure 2）

- 移除 `<think>` 后，性能在复杂任务（如 IndustryOR、MAMO-C）上大幅下降；
- 表明 `<think>` 不是事后解释，而是内部推理空间，对复杂逻辑建模至关重要；
- 在简单任务上略有提升，可能因避免“过度思考”。

#### （2）优化器选择（GRPO vs DAPO）— Figure 3

- GRPO 与 DAPO 性能曲线几乎重合；
- 表明在本框架下，**执行反馈的质量远比优化算法重要**，GRPO 已足够有效。

#### （3）小规模模型表现（Appendix G）

| 模型 | OptiBench ↑ | NL4OPT ↑ | MAMO-E ↑ |
|------|-------------|----------|---------|
| Qwen2.5-1.5B | 35.44 → **47.00** |
| Qwen2.5-3B | 28.40 → **55.81** |

✅ 强化学习对小模型有明显放大效应，**3B 模型经 EVOM 训练后接近 7B 基线水平**，适合资源受限部署。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **结果级监督足以学会优化建模**  
   无需过程标注，仅靠执行反馈即可让模型掌握复杂的数学建模能力。

2. **求解器可作为验证环境，实现跨求解器泛化**  
   模型学到的是**通用数学建模逻辑**，而非特定 API 语法，支持零样本迁移。

3. **低代价适配新求解器成为可能**  
   仅需切换执行后端，继续训练即可快速适应新求解器，无需重建数据集。

4. **显式推理提升建模质量**  
   `<think>` 块是有效的内部工作区，有助于分解复杂问题。

5. **强化学习显著减少变量与实现错误**（Error Analysis, Table 11）
   - **Variable Error** 下降最多（6.59% → 0.78%），说明模型更准确地声明整数/连续变量；
   - **Implementation Error** 显著降低，API 调用更规范；
   - **Constraint Error** 改进有限，仍是未来难点。

---

### ⚠️ 局限性

1. **冷启动问题**：对于预训练中极少出现的求解器（如 COPT），初始生成成功率极低，导致稀疏奖励，收敛慢。
   - 解决方案：提出两阶段策略 —— 先用少量翻译数据做冷启动 SFT，再接 RL。

2. **大规模 SFT 可能损害后续 RL 效果**（Appendix J）
   - 过多 SFT 导致模型僵化、熵降低，不利于 RL 探索；
   - 建议：**少量高质量冷启动数据 + RL** 是最优路径。

3. **复杂约束建模仍具挑战**
   - 当前框架对深层语义约束（如隐含逻辑）捕捉不足；
   - 错误分析显示 Constraint Error 占残差主导地位。

---

### 🔮 未来工作方向

1. **引入分层奖励机制**：对约束完整性、变量类型等中间属性设计细粒度奖励。
2. **结合过程提示的轻量引导**：在关键步骤注入少量模板，辅助复杂推理。
3. **探索多求解器联合训练**（Multi-Solver Joint Training）以增强通用建模能力（见 Appendix L）。
4. **扩展至更多领域**：如调度、博弈论、动态规划等问题。

---

## ✅ 总结

> **EVOM 开辟了一条低成本、高泛化、可扩展的优化建模自动化新路径**。

它摆脱了对昂贵人工标注和闭源模型的依赖，利用**求解器自身作为客观裁判**，通过强化学习实现了从自然语言到可执行优化代码的可靠映射。更重要的是，它首次系统验证了**跨求解器泛化**的可能性，为构建真正通用的 Decision Intelligence 系统提供了坚实基础。

</details>

---

### 9. [Reclaiming Idle CPU Cycles on Kubernetes: Sparse-Domain Multiplexing for Concurrent MPI-CFD Simulations](https://arxiv.org/abs/2604.00377)

**Authors**: Tianfang Xie  
**Category**: cs.DC  
**Published**: 2026-04-02  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.00377v1  

#### Abstract
When MPI-parallel simulations run on shared Kubernetes clusters, conventional CPU scheduling leaves the vast majority of provisioned cycles idle at synchronization barriers. This paper presents a multiplexing framework that reclaims this idle capacity by co-locating multiple simulations on the same ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Reclaiming Idle CPU Cycles on Kubernetes: Sparse-Domain Multiplexing for Concurrent MPI-CFD Simulations*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在共享的 **Kubernetes** 集群上运行 **MPI-parallel CFD** 模拟时，传统的 CPU 调度机制导致大量计算周期在 **MPI 同步屏障（如 `MPI_Allreduce`）** 处被浪费。由于域分解中近壁面区域（dense domains）计算密集而远场区域（sparse domains）计算轻量，后者在完成计算后会长时间空闲等待同步，造成高达 95% 的 CPU 周期闲置。

这一现象在云环境中尤为严重，因为：
- 每一个预留但未使用的 millicore 都直接转化为经济成本；
- 云平台存在 vCPU 配额限制，难以动态扩容；
- 工程 CFD 流程通常需要执行大量独立参数扫描任务，**集群吞吐量（throughput）** 比单次模拟的 wall-clock 时间更重要。

因此，核心问题是：**如何在不增加硬件开销的前提下，回收这些空闲 CPU 容量以提升 Kubernetes 上的 CFD 模拟吞吐量？**

---

### 提出的新方法与创新思路

本文提出了一种名为 **Sparse-Domain Multiplexing（稀疏域复用）** 的框架，通过以下四个关键贡献实现 CPU 利用率最大化：

#### ✅ 主要贡献

1. **基于 PMPI 的 per-rank duty-cycle 分析方法**
   - 使用 **PMPI profiling** 技术拦截 MPI 调用，在不修改应用代码的情况下测量每个 rank 的计算/通信占比。
   - 定义 **CPU duty cycle**：$ d_i = \frac{t_{\text{compute}}}{t_{\text{compute}} + t_{\text{MPI}}} $，量化每个 rank 的实际 CPU 占用比例。

2. **比例式 CPU 分配 + 并发共置调度（Proportional CPU Allocation & Co-location）**
   - 根据 duty cycle 动态设置每个 pod 的 CPU `requests`（无 `limits`），采用 **Burstable QoS** 类别避免 CFS 带宽限制造成的严重延迟放大。
   - 将多个 CFD 模拟部署在同一组物理节点上，让空闲的 sparse ranks 所释放的 CPU 容量被其他模拟的 ranks 利用。

3. **单参数解析吞吐模型（Analytical Throughput Model）**
   - 提出线性竞争模型：  
     $$
     T_N = T_1 (1 + \beta(p_N - p_1))
     $$
     其中 $\beta$ 是竞争系数，仅需一次双任务实验即可拟合，并用于预测任意并发数 $N$ 下的吞吐增益，误差控制在 ±4% 内。

4. **全自动动态控制器（Dynamic Controller）**
   - 实现端到端自动化流程：从 profiling → 基于 **KEP-1287 In-Place Pod Vertical Scaling** 动态调整 CPU 请求 → 自动部署新模拟 → 公平性监控。
   - **首次将 in-place CPU scaling 应用于正在运行的 MPI 工作负载**，无需重启 pod。

---

### 相比现有方法的优势

| 方面 | 本文方法 | 现有方法 |
|------|--------|---------|
| **CPU 分配粒度** | per-rank 差异化分配（67m–1005m） | 所有 rank 均匀分配（如 1000m） |
| **调度策略** | 利用 idle cycles 进行多作业 co-location | 单作业独占资源或静态划分 |
| **弹性能力** | 支持运行时垂直伸缩（in-place resize） | 多依赖 checkpoint/restart 或水平扩展 |
| **自动化程度** | 全自动 pipeline（profile → resize → pack） | 手动配置或部分自动化 |
| **适用场景** | 特别适合高通信/低计算比的 CFD 工作负载 | 更适用于计算密集型或松耦合任务 |

> ⭐ **核心优势**：在相同云预算下，显著提升吞吐量（最高达 3.74×），同时保持较低的 per-case 性能退化。

---

## 2. 核心实验方法和设置

### 数据集与测试案例
- **CFD 求解器**：OpenFOAM 10 中的 `rhoSimpleFoam`
- **几何模型**：NACA 0012 翼型（Mach 0.72）
- **网格规模**：498,834 六面体单元
- **湍流模型**：k-ω SST
- **边界条件**：自由来流速度 250 m/s，温度 298 K，攻角 0°

### 域分解方式
- **手动同心圆分区（Concentric Decomposition）**：
  - 16 个子域分为三个区：
    - **Dense ranks (12–15)**：权重 15，占 68% 单元（近壁层）
    - **Medium ranks (8–11)**：权重 5，占 23%
    - **Sparse ranks (0–7)**：权重 1，占 9%（远场）
- 同时验证了标准图分割工具 **Scotch** 的兼容性。

### 实验环境
- **平台**：Amazon EC2
- **节点配置**：
  - Worker Nodes：12 台 `c5.2xlarge`（每台 8 vCPU, 16 GiB RAM）
  - 总可用 vCPU：96
  - 控制平面：`t3.medium`
- **Kubernetes 发行版**：k3s v1.35.0（支持 KEP-1287 GA）
- **存储**：共享 Amazon EFS 卷用于输入输出文件
- **网络隔离**：禁用 Open MPI 的共享内存 BTL（`--mca btl tcp,self`），防止跨 job 干扰

### 评估指标
| 指标 | 描述 |
|------|------|
| **Wall-clock time** | 单个模拟完成时间（秒） |
| **Makespan** | 所有并发模拟中最晚完成的时间 |
| **Throughput Gain** | $ \frac{N}{T_N / T_1} $，即单位时间内完成的模拟数量相对于串行执行的倍数 |
| **Per-case Degradation** | 并发导致的单个模拟延长时间百分比 |
| **Scheduling Efficiency** | $ \text{Throughput} / N $，反映资源利用效率 |
| **Fairness (A/B ratio)** | 不同模拟间 wall-clock 时间的比例，理想为 1.0 |

### 基线方法对比
| 配置 | 描述 |
|------|------|
| **C-1E** | 单模拟，所有 rank 请求 1000m CPU（Equal allocation） |
| **C-1P** | 单模拟，按 duty cycle 设置差异化请求（Proportional allocation） |
| **C-2E** | 双模拟，均使用等量 CPU 请求 |
| **C-2P** | 双模拟，均使用比例式 CPU 请求（本文主推方案） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table 4 和 Figure 5）

| 配置 | Sim A (s) | Sim B (s) | Makespan (s) | Throughput | Degradation |
|------|-----------|-----------|--------------|------------|-------------|
| C-1E | 1179 ± 28 | — | 1179 | 1.00× | — |
| C-1P | 1249 ± 20 | — | 1249 | 1.00× | — |
| C-2P | 1331 ± 12 | 1410 ± 10 | **1410** | **1.77×** | +6% ~ +13% |
| C-2E | 1286 ± 14 | 1274 ± 22 | 1286 | 1.83× | ~+3% |

> 注：虽然 C-2E raw throughput 略高，但其消耗更多 CPU request 资源（32vCPU vs. 12vCPU），不利于更高密度共置。

---

### 并发密度扩展实验（N = 1–5）

| N | Throughput | Efficiency | Per-case Degradation | Makespan (s) |
|----|------------|-----------|------------------------|---------------|
| 1 | 1.00× | 100% | 0% | 1249 |
| 2 | 1.77× | 89% | +7% | 1410 |
| 3 | **2.59×** | **86%** | +16% | 1446 |
| 4 | 3.11× | 78% | +28% | 1604 |
| 5 | **3.74×** | 75% | **+34%** | 1670 |

- **Pareto Knee 在 N=3**：此时效率仍高达 86%，是性价比最优选择。
- 成本效益显著（Table 5）：
  - N=5 时每模拟成本降低 **73%**（从 \$1.43 → \$0.38）

---

### 消融实验与模型验证

#### ✅ 吞吐模型准确性
- 使用实测数据拟合 $\beta = 0.524$，预测值与实测误差 **全部在 ±4% 以内**。
- 仅用 N=2 数据训练模型，预测 N=3–5 结果仍保守准确（略微高估耗时），适合容量规划。

#### ✅ 分解方式无关性
- 使用默认 **Scotch 分区** 替代 concentric 分区：
  - duty cycle 范围：7.7%–13.1%（仍 >87% idle）
  - wall-clock 时间与 concentric 相差 <1%
  - 表明 multiplexing 效益对具体分区算法不敏感

#### ✅ 动态控制器表现（Section 5.4）
- 自动完成从 1 到 4 个并发模拟的全过程：
  - 包括 4 轮 PMPI profiling
  - **64 次 in-place pod resize**
  - **零 pod 重启**
- 最终达成 **3.25× 吞吐量**，优于静态配置（C-4P 的 3.11×）
- 全过程耗时 **53 分钟**

---

## 4. 关键结论和发现

### 主要发现
1. **MPI-CFD 模拟普遍存在极高 idle rate**：
   - 即使是最重载的 near-wall ranks，duty cycle 也仅为 ~19.4%
   - sparse ranks 平均仅 **5.0%**，意味着 **95% 的 CPU 周期处于空闲状态**

2. **Sparse-Domain Multiplexing 显著提升吞吐量**：
   - 双模拟共置带来 **1.77× 吞吐增益**
   - 最高可达 **3.74×（N=5）**
   - **N=3 是帕累托最优点**，兼顾效率与公平

3. **比例式 CPU 分配是关键前提**：
   - 低权重请求（如 67m）赋予 sparse ranks 更小的 CFS 权重，从而“腾出”CPU 给其他 job 使用
   - 若使用固定 1000m 请求，则无法支持高密度共置（受 ResourceQuota 限制）

4. **In-Place Pod Vertical Scaling 首次成功用于 MPI 工作负载**
   - 实现了真正的运行时弹性调度
   - 为未来智能调度系统奠定基础

---

### 局限性
1. **当前实验局限于单一 NUMA 架构和特定实例类型**（c5.2xlarge）
2. **未暴露硬件性能计数器（PMU）**，无法直接测量 LLC miss 或内存带宽争用
3. **动态控制器尚属 PoC 阶段**，缺乏容错机制（如 kubelet 压力、resize 失败处理）
4. **假设 workload duty cycle 较低（<20%）**，若 duty cycle 接近 50% 以上，multiplexing 增益将大幅下降
5. **未考虑跨节点通信带宽瓶颈**，在更大规模或更复杂拓扑中可能成为新瓶颈

---

### 未来工作方向
1. **异构工作负载配对（Heterogeneous Workload Pairing）**
   - 利用 Kubernetes 的 pod affinity 规则，主动将 **fine-mesh（dense）** 与 **coarse-mesh（sparse）** 模拟配对共置，进一步优化 CPU 利用率。

2. **燃烧化学加速（Combustion Chemistry Acceleration）**
   - 在反应流 CFD 中，反应区计算强度远高于非反应区（100–500× 子迭代）。
   - 结合神经网络代理模型（surrogate model）处理非刚性区域 + 本文的比例调度框架，有望同时降低 per-rank 开销并提高打包效率。

3. **集成到生产级调度器**
   - 当前控制器可与 **Volcano** 或 **Kueue** 等批处理调度器组合使用，前者负责 job 排队，后者负责 intra-job rank 级调度。

4. **支持 NUMA-aware 和 GPU-aware 调度**
   - 扩展至多 NUMA 节点和 GPU 加速场景，适应更广泛的工程仿真需求。

---

> 🔗 **开源信息**：所有实验脚本、Kubernetes 清单、PMPI 库、动态控制器及分析代码已公开于 GitHub：  
> https://github.com/Xieldor/K8s-CFD-Multiplexing

</details>

---

### 10. [A Safety-Aware Role-Orchestrated Multi-Agent LLM Framework for Behavioral Health Communication Simulation](https://arxiv.org/abs/2604.00249)

**Authors**: Ha Na Cho  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.00249v1  

#### Abstract
Single-agent large language model (LLM) systems struggle to simultaneously support diverse conversational functions and maintain safety in behavioral health communication. We propose a safety-aware, role-orchestrated multi-agent LLM framework designed to simulate supportive behavioral health dialogu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **single-agent LLM** 的对话系统在行为健康（behavioral health）沟通中面临两大挑战：
- **功能单一性**：难以同时支持多种复杂的、相互依赖的对话功能（如共情、动机激励、认知重构、行动规划等），导致对话缺乏多样性与深度。
- **安全性不足**：缺乏持续、实时的安全审计机制，在心理健康场景下存在生成不当内容（如情绪误导、伦理风险）的风险。

### 提出的新方法与新思路
作者提出了一种 **Safety-Aware Role-Orchestrated Multi-Agent LLM Framework**，其核心创新包括：

- **角色分工架构（Role-Differentiated Agents）**  
  将对话职责分解为六个专用 agent：
  - **Empathizer**（共情）
  - **Motivator**（动机鼓励）
  - **Planner**（行动计划）
  - **Cognitive Restructurer**（认知重构）
  - **Director**（响应合成）
  - **Responsible Agent**（安全监督）

- **动态控制器（Prompt-Based Controller）**  
  基于 prompt 编码的条件规则，动态激活相关 agent，实现上下文感知的角色调度，而非固定流程。

- **嵌入式安全机制（Embedded Safety Auditing）**  
  引入持久运行的 **Responsible Agent** 和 **Director** 进行每轮交互中的实时伦理与情感安全审查，将 safety 融入生成流程本身，而非事后过滤。

### 相比现有方法的优势
| 维度 | 传统 single-agent 或松耦合 multi-agent | 本文框架 |
|------|----------------------------------------|----------|
| 功能多样性 | 单一模型尝试覆盖所有功能 → 易混淆角色 | 角色专业化 → 更高 functional diversity |
| 可解释性 | 黑箱决策，难追踪功能来源 | 模块化设计 + 日志记录 → 行为可追溯、可分析 |
| 安全性 | 多为后处理检测或静态规则 | 实时、持续、由 dedicated agent 执行 |
| 系统可控性 | 难以精细控制特定对话策略 | 通过 prompt engineering 精确调控 agent 行为 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **DAIC-WOZ corpus**（Distress Analysis Interview Corpus Wizard-of-Oz）
  - 包含 189 场半结构化心理访谈录音与文本转录
  - 本研究仅使用 **participant utterances** 作为输入，模拟真实用户表达
  - 随机选取 **7 名参与者**（ID: 300, 319, 361, 396, 446, 480, 492）进行系统级分析
  - PHQ-8 分数范围广泛，涵盖不同抑郁严重程度的语言表现

### 数据预处理
- 移除非语言符号（笑声标记、特殊字符）
- 清理常见 disfluencies（如 "um", "uh"）
- 删除少于 3 个 token 的短语（如 “okay”, “yeah”）
- 保留最近最多 3 轮历史作为 context window

### 实验设置
- **LLM 模型**：GPT-3.5-turbo 与 GPT-4-turbo（用于评估）
- **硬件平台**：Apple M2（macOS, 11-core CPU, 18GB RAM）
- **每轮交互流程**（见 Table I）：
  1. 用户输入更新共享内存
  2. 控制器根据 context 决定激活哪些 agent
  3. 各 agent 生成角色特异性响应
  4. Director 合成统一输出
  5. Responsible Agent 审核安全性
  6. 返回最终响应并记录日志

### 评估指标（Proxy-Based Evaluation）
由于是模拟框架而非临床干预，采用可扩展的代理评估体系：

| 评估维度 | 方法 | 工具 | 指标 |
|--------|------|------|------|
| **质量评分** | Rubric-based scoring | GPT-4-turbo | 五维 5 分制 Likert 量表：<br>- Empathy<br>- Helpfulness<br>- Coherence<br>- Appropriateness<br>- Role Alignment |
| **意图分类** | Zero-shot intent classification | GPT-3.5-turbo | 12 类 therapeutic intent：<br>e.g., validation, encouragement, reflection, psychoeducation, coping suggestion, cognitive reframing, etc. |
| **语言多样性** | Linguistic analysis | 自动计算 | Word count, Type-Token Ratio (TTR) |

### 基线方法对比
- 主要对比对象为 **single-agent baseline**（即一个 LLM 承担全部功能）
- 不直接比较其他 multi-agent 系统，而是强调自身在 **coordination mechanism**, **safety integration**, 和 **interpretability** 上的设计优势

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）Agent 激活频率（Fig. 3）
- **Director & Responsible Agent**：每轮必激活（N = 1370），体现其 supervisory 角色
- **Empathizer**：高频选择性激活 → 对情绪表达敏感
- **Cognitive Restructurer**：极少被调用（“Rare” 激活模式）→ 认知重构需求较低或触发条件较严格

#### （2）跨 agent 转移模式（Fig. 4）
- 输出最常流向 **Director**（尤其是来自 Responsible Agent 的 663 次转移），表明合成路径清晰
- Empathizer → Director（341）、Motivator → Director（196）常见，说明情感类输出有效整合
- Cognitive Restructurer 下游影响小 → 其输出较少引发进一步交互

#### （3）计算效率（Latency & Token Usage）
| Agent | 平均延迟 | Token 数量 | 特征 |
|-------|---------|-----------|------|
| **Director** | ~3.5 秒 | 中等偏低 | 合成任务重但输出简洁 |
| **Empathizer/Motivator** | 较低 | 较高 | 生成更丰富的情感语言 |
| **Planner/Cognitive Restructurer** | 最低 | 最低 | 适合资源受限环境 |

#### （4）Rubric 评分结果（Table II）
| Agent Role | Empathy | Helpfulness | Coherence | Appropriateness | Role Alignment | TTR |
|----------|--------|------------|----------|----------------|---------------|-----|
| Empathizer | **4.80** | 3.80 | 5.00 | 5.00 | 5.00 | 0.13 |
| Motivator | 4.00 | 4.00 | 5.00 | 4.83 | 5.00 | 0.15 |
| Planner | 3.60 | 3.80 | 5.00 | 4.80 | 4.60 | 0.14 |
| Cognitive Restructurer | 4.00 | 4.00 | 5.00 | 5.00 | 5.00 | **0.24** |
| Director | 4.00 | 4.11 | 5.00 | 5.00 | 5.00 | 0.07 |
| Responsible | 3.86 | 4.00 | 5.00 | 4.93 | 5.00 | 0.08 |

> ✅ 所有 agent 在 **Coherence** 和 **Role Alignment** 上均达满分（5.00），证明角色一致性高  
> 🔺 Empathizer 在 empathy 上显著领先（4.80）  
> 💡 Cognitive Restructurer 拥有最高 **TTR (0.24)**，说明其词汇变化最大，符合“认知重塑”需灵活表达的特点  
> ⚠️ Supervisor agents（Director/Responsible）语言重复性强（TTR ≈ 0.07–0.08）

#### （5）意图分布（Intent Classification）
- 最常见意图：
  - **Psychoeducation**（34.0%）
  - **Empowerment**（20.0%）
  - **Encouragement**（14.0%）
- 较少见但重要：
  - Validation（8.0%）、Reflection（4.0%）、Coping suggestion（4.0%）、Cognitive reframing（6.0%）
- ✅ **所有 12 类 intent 均至少出现一次** → 功能空间完整覆盖

### 与基线方法对比结果
- 相比 single-agent baseline：
  - 更高的 **functional diversity**
  - 更清晰的 **role differentiation**
  - 更强的 **safety control**
  - 可预测的 **latency-functionality trade-off**（模块化带来轻微延迟增加，但换来可控性提升）

> ❗ 注：未报告端到端任务准确率或临床疗效，因该框架定位为 **simulation and analysis tool**，非 clinical intervention

---

## 4. 关键结论和发现

### 主要发现
1. **角色专业化可行且有效**  
   明确的角色划分能产生功能差异化的输出，且可通过 prompt engineering 实现稳定控制。

2. **协调机制本身是关键设计变量**  
   agent 如何被激活、串联、监督，直接影响对话结构与质量。prompt-encoded transition rules 成功实现了 context-aware orchestration。

3. **安全性可以架构化实现**  
   将 Responsible Agent 作为永久组件嵌入流程，使安全审计成为生成过程的一部分，提升了透明度与可审计性。

4. **情感能力仍集中于单一 agent**  
   当前设计中，empathy 主要由 Empathizer 承担，未在整个系统中扩散 → 存在“情感孤岛”现象。

5. **控制器偏向 solution-oriented 回应**  
   Psychoeducation 和 empowerment 占主导，可能抑制 reflective 或 deep emotional support，反映 controller policy 的保守倾向。

6. **系统具备完整的功能表达能力**  
   尽管某些 intent 出现频率低，但所有预定义类别均有覆盖，说明系统潜力未被完全释放。

---

### 局限性
- **仿真环境限制**：完全基于离线 transcript 模拟，无真实 human-in-the-loop 交互
- **顺序式协调**：agent 之间非真正并发对话，缺少 rich inter-agent negotiation
- **上下文过滤牺牲敏感性**：为效率删除短句和 fillers，可能丢失微弱但重要的情感信号
- **依赖 LLM 自身偏见与幻觉**：虽有 Responsible Agent 审核，但仍无法根除底层模型风险
- **评估为 proxy-based**：使用 GPT-4 评分而非 human judges，可能存在偏差

---

### 未来工作方向
1. **自适应 orchestration 策略**  
   引入 context-aware activation weights，动态调整各 agent 权重，避免固定调度偏差。

2. **增强 inter-agent interaction**  
   探索 agent 间的多轮协商、辩论或共识机制（debate mechanisms），提升观点多样性。

3. **情感信号共享机制**  
   设计 affective state propagation layer，让情绪理解成为全系统的共享属性，而非局限于 Empathizer。

4. **human-in-the-loop evaluation**  
   引入临床专家或真实用户参与测试，验证系统在实际应用中的有效性与接受度。

5. **轻量化部署优化**  
   针对 Planner 和 Restructurer 的高效特性，探索边缘设备上的低延迟部署方案。

---

> 📌 **总结定位**：  
> 本论文提出的框架并非用于替代心理咨询师的 **clinical tool**，而是一个面向 **behavioral health informatics** 与 **decision-support research** 的 **simulation and analysis platform**，旨在推动对多智能体对话系统在安全、可控、可解释方向的发展。

</details>

---

### 11. [Decision-Centric Design for LLM Systems](https://arxiv.org/abs/2604.00414)

**Authors**: Wei Sun  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.00414v1  

#### Abstract
LLM systems must make control decisions in addition to generating outputs: whether to answer, clarify, retrieve, call tools, repair, or escalate. In many current architectures, these decisions remain implicit within generation, entangling assessment and action in a single model call and making failu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Decision-Centric Design for LLM Systems**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前大多数 LLM 系统将控制决策（如是否回答、检索、澄清、调用工具等）隐式地嵌入在生成过程中，导致以下问题：
- 决策过程不可见，难以诊断失败原因；
- 控制逻辑与生成逻辑纠缠，无法模块化改进；
- 在复杂任务中，错误会累积且难以修复。

### **提出的新方法与新思路**
作者提出了 **Decision-Centric Framework**，其核心思想是：
> 将 **决策相关信号（decision-relevant signals）** 与 **决策策略（policy）** 显式分离，使控制层成为系统的一个可检查、可约束、可归因的独立组件。

该框架由三个核心元素构成：
- **Actions A**：候选动作集合（如 `clarify`, `execute`, `retrieve`, `backtrack`）；
- **Decision Context c**：决策所需的信息上下文（包括历史、检索结果、验证输出、不确定性估计等）；
- **Decision Function δ**：从上下文到动作的显式映射函数（可以是规则、学习模型或优化器）。

这种设计使得：
- 决策不再是生成的副产品，而是明确的系统接口；
- 支持单步（如路由、自适应推理）和多步序列决策（如交互式搜索）；
- 失败可归因于 **信号估计、决策策略 或 执行** 中的具体环节。

### **相比现有方法的优势**
| 维度 | 传统 Prompt-Based 方法 | Decision-Centric 方法 |
|------|------------------------|------------------------|
| **控制可见性** | 隐式，融合在生成中 | 显式，可追踪 |
| **失败归因** | 困难，黑箱行为 | 可定位至具体模块 |
| **模块化改进** | 修改提示影响全局 | 可单独优化信号或策略 |
| **约束执行** | 依赖模型遵循指令 | 可硬编码结构约束（如“失败后必须澄清”） |
| **跨任务通用性** | 每个任务需重新设计提示 | 统一抽象适用于多种控制场景 |

---

## **2. 核心实验方法和设置**

### **使用的数据集与任务设置**

#### **实验一：Calendar Scheduling（日程安排）**
- **任务**：从自然语言请求中提取完整会议信息（date, start_time, duration_min, attendees），缺失字段时应主动澄清。
- **数据构造**：基于固定事实生成 8 种场景，控制缺失字段数 $k \in \{0,1,2,3,4\}$ 和模糊类型（absent / unresolvable）。
- **Action Space**：`clarify` 或 `execute`。
- **评估指标**：
  - **Success Rate**：最终输出正确率；
  - **First-action Optimality**：首次动作是否最优；
  - **Wasted Executions**：无效执行次数；
  - **Clarification Turns**：澄清轮次。

#### **实验二：Graph Disambiguation（图谱消歧）**
- **任务**：在一个合成知识图中识别目标人物，通过 `clarify`, `execute`（访问节点）, `backtrack` 进行探索。
- **数据构造**：200 节点图，每个节点有 5 个属性；设计 5 个场景测试不同控制能力。
- **关键信号**：
  - $p_{suff}$：上下文充分性（候选数量倒数）；
  - $p_{corr}$：当前路径正确性（基于隐藏属性一致性）。
- **评估指标**：Success Rate, Wasted Traversals, Clarify/Backtrack 次数。

#### **实验三：Retrieval Control（检索控制）**
- **数据集**：Natural Questions (NQ) 的 150 个样本，构建 2000 文档 BM25 检索库。
- **任务**：决定是否停止检索或继续扩展（expand_k），直到能回答问题。
- **难度分桶**：
  - Easy：初始检索即命中；
  - Medium：需扩展一次才命中；
  - Hard：始终未命中。
- **评估指标**：
  - **Success Rate**：最终 passage 包含答案；
  - **Avg Retrieval Rounds (RR)**：效率指标。

---

### **基线方法对比**
| 方法 | 描述 |
|------|------|
| **Prompt** | 单一 LLM 调用完成状态理解与动作选择，决策隐式 |
| **Retry** | 失败后直接重试，不进行澄清或回溯 |
| **Prompt (w/ policy)** | 在 prompt 中描述策略逻辑，但仍为隐式执行 |
| **DC (Decision-Centric)** | 显式提取信号（如 $p_{suff}$）并应用确定性策略 |

所有方法共享相同的 LLM（如 `ibm/granite4:micro` 或 `LLaMA 3`）、执行模块和输入状态，仅控制机制不同。

---

## **3. 主要实验结果和性能指标**

### **实验一：Calendar Scheduling 结果**

| $k$ | Prompt-Clarify (Success) | DC (Success) |
|-----|--------------------------|-------------|
| 0   | 100%                     | 100%        |
| 1   | 100%                     | 100%        |
| 2   | 75%                      | **100%**    |
| 3   | 60%                      | **100%**    |
| 4   | 10%                      | **100%**    |

- **DC 完全避免了浪费的执行**（Wasted Executions 接近 0），而 Prompt 平均高达 2.2 次；
- 在 **unresolvable 场景** 下差距更大：Prompt 错误地将模糊引用当作有效信息提前执行；
- **消融分析**：在 LLaMA 上复现实验时发现 DC 性能非单调下降，但通过 **仅修改 question generator prompt**（不改 policy 或 estimator）即可修复，证明模块化优势。

> ✅ **关键发现**：显式控制显著提升鲁棒性，尤其在高不确定性场景。

---

### **实验二：Graph Disambiguation 结果**

| Scenario | Retry | Prompt | Prompt (w/ policy) | DC |
|---------|-------|--------|--------------------|----|
| S1: Clean | 100% | 100% | 100% | 100% |
| S2: Ambiguous ($p_{suff}$ only) | 45% | 85% | 95% | **100%** |
| S3: Unreliable ($p_{corr}$ only) | 100% | 100% | 100% | 100% |
| S4: Orthogonal (joint) | 65% | 95% | 100% | **100%** |
| **S5: Correlated (belief update)** | 60% | 35% | 35% | **100%** |

- **S5 是最强挑战**：一次失败 traversal 被动淘汰多个候选（correlated belief update），但 Prompt 类方法无法感知此状态变化；
- DC 成功的关键在于 **显式维护并更新 belief state**，而 prompt 方法即使写入策略也无法恢复丢失的状态；
- Prompt (w/ policy) 无提升，说明 **仅靠提示无法实现真正的状态跟踪**。

> ✅ **关键发现**：显式状态管理对复杂序列决策至关重要。

---

### **实验三：Retrieval Control 结果**

| Bucket | Method | Success | Avg RR |
|--------|--------|---------|--------|
| Easy | Prompt | 100% | 0.10 |
|      | DC-LLM | 100% | 0.62 |
| Medium | Prompt | 14% | 0.08 |
|        | DC-LLM | **88%** | 1.34 |
|        | DC-Dense | **90%** | 1.66 |
|        | DC-Composite | **94%** | 1.62 |
| Hard | All Methods | ~14–18% | ~1.6–1.8 |

- **DC-LLM vs Prompt**：两者使用相同 LLM 和 passage，唯一区别是是否显式输出 sufficiency signal $p$；
  - 在 Medium 任务上，Prompt 成功率仅 14%，而 DC-LLM 达 88%；
  - 分析显示：**64% 的 Prompt 失败案例中，模型自己承认“答案不在文中”，却仍选择 `stop`** —— 典型的“判断与行动脱节”。

- **信号组合优势**：
  - DC-Dense 倾向过扩展（relevance bias）；
  - DC-LLM 更保守（answerability bias）；
  - DC-Composite 结合二者，在 medium 上达到 **94% 成功率**。

> ✅ **关键发现**：显式信号支持模块化迭代与故障归因。

---

## **4. 关键结论和发现**

### **主要发现**
1. **显式决策层显著提升可靠性**：
   - 减少无效动作（wasted actions）；
   - 提高任务成功率，尤其在高不确定性或多步交互场景；
   - 在三种不同类型的任务中均取得一致优势。

2. **失败可归因（Failure Attribution）**：
   - 可区分问题是出在 **signal estimation**（如 LLM 判断不准）、**policy design**（阈值不合理）还是 **execution**（生成错误）；
   - 支持针对性修复（如只改 question generator）而不影响其他部分。

3. **模块化支持灵活迭代**：
   - 同一控制器下可替换不同信号源（LLM judge / Dense embedding / Composite）；
   - 新信号可在离线 trace 上评估，无需重新运行整个 pipeline。

4. **提示工程有根本局限**：
   - 即使在 prompt 中写明策略，也无法保证 LLM 正确执行，尤其是在状态动态变化的场景；
   - “评估正确但行动错误”是隐式控制的典型缺陷。

---

### **方法的局限性**
- 当前框架假设 **动作空间有限且可枚举**，对于开放域复杂行为可能难以建模；
- 信号估计本身仍依赖 LLM 或启发式方法，存在误差传播风险；
- 决策函数目前以确定性为主，尚未深入探索随机策略或强化学习集成；
- 实验集中在受控环境，真实世界复杂性（如用户对抗性输入）尚未充分验证。

---

### **未来工作方向**
1. **Hierarchical Decision Layers**：
   - 高层策略选择任务类型，底层策略执行细节；
   - 支持更复杂的 multi-step workflows。

2. **Bayesian & VoI Integration**：
   - 将 Value of Information (VoI) 作为 utility function 的一部分，实现理性信息采集；
   - 引入 posterior 更新机制，形成闭环 belief-state 推理。

3. **自动化信号构造**：
   - 学习自动构建 decision-relevant signals，而非人工设计；
   - 利用 meta-learning 适配不同任务。

4. **应用于 Multi-Agent Systems**：
   - 每个 Agent 暴露本地 decision context 和 policy；
   - 实现透明协作与协调控制。

---

> 🔚 **总体评价**：  
> 本文提出的 **Decision-Centric Design** 不是一种具体的算法，而是一个**架构级原则**，旨在将 LLM 系统的“大脑”从“嘴巴”中解放出来。它为构建 **可靠、可控、可诊断** 的 LLM 应用提供了坚实的基础框架，具有广泛的工程与研究价值。

</details>

---

### 12. [TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models](https://arxiv.org/abs/2604.00666)

**Authors**: Lingjie Chen, Ruizhong Qiu, Yuyu Fan, Yanjun Zhao, Hanghang Tong  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.00666v1  

#### Abstract
Diffusion language models (DLMs) offer a promising path toward low-latency generation through parallel decoding, but their practical efficiency depends heavily on the decoding trajectory. In practice, this advantage often fails to fully materialize because standard training does not provide explicit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
**Train-Inference Mismatch in DLMs**  
扩散语言模型（Diffusion Language Models, DLMs）理论上支持并行解码，从而实现低延迟生成。然而，在实践中，由于标准训练采用均匀随机掩码策略（uniform random masking），模型在训练时并未学习到有效的**token reveal order**（即解码轨迹）。这导致推理时模型可能退化为类似自回归（AR）的行为，或跳过困难 token，严重限制了实际的并行性。

此外，现有的轨迹优化方法（如基于 DLM 自身采样的蒸馏方法）依赖昂贵的扩散采样过程来获取伪轨迹（pseudo-trajectories），计算开销大、难以扩展。

### 提出了什么新方法或新思路
提出 **TRIMS (Trajectory-Ranked Instruction Masked Supervision)** —— 一种轻量级、轨迹引导的监督微调框架，用于增强 DLM 的解码效率。

核心思想是：  
- 利用一个**自回归教师模型**（AR teacher）对训练数据进行单次前向传播，计算每个 token 的预测难度（difficulty score，定义为 $-\log p(y_i | x, y_{<i})$）。
- 将这些难度分数通过分位数划分为多个有序桶（buckets），形成“从难到易”的离散化轨迹信号。
- 在训练中引入**轨迹感知的掩码策略**（trajectory-aware masking）：在每一步训练中，随机选择一个桶阈值 $k$，将更难的 token（bucket ≤ k）视为已知上下文，而更容易的 token（bucket > k）则以高概率保持掩码状态，迫使模型优先恢复困难 token。

该方法无需任何 DLM 采样，仅需一次 AR 模型的 teacher-forcing 推理，即可注入有效的轨迹监督。

### 相比现有方法的优势
| 维度 | TRIMS | 蒸馏类方法（如 dParallel, d3LLM） |
|------|-------|-------------------------------|
| **训练成本** | 极低（仅需 AR 模型一次前向 + 微调） | 高昂（需大量 DLM 采样构建蒸馏数据集） |
| **数据需求** | ~1K 样本 | ~93K 样本（两个数量级更多） |
| **通用性** | 不依赖特定 DLM 采样策略 | 依赖目标模型生成轨迹，泛化差 |
| **实现复杂度** | 简单，兼容标准 MDLM 流程 | 复杂，需额外轨迹生成与过滤流程 |

> ✅ **TRIMS 在几乎不增加训练成本的前提下，实现了与蒸馏方法相当甚至更好的 accuracy-parallelism trade-off。**

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **训练数据**：`s1K` 数据集（Muennighoff et al., 2025），包含约 1K 条指令-响应对，侧重推理任务。
- **评估基准**（四大公开测试集）：
  - **GSM8K**：小学数学应用题
  - **MATH**：高中及以上级别数学问题
  - **HumanEval**：Python 编程函数补全
  - **MBPP**：面向初学者的编程任务

### 实验设置和评估指标
- **骨干模型**：
  - `LLaDA-Instruct`（8B）
  - `Dream-Instruct`（7B）
- **训练细节**：
  - 序列长度：1024
  - Batch size：32（梯度累积 factor=4）
  - 学习率：1e-4，cosine 调度，warmup ratio=0.03
  - 使用 LoRA（rank=32）进行参数高效微调
  - 训练平台：8×NVIDIA A100 80GB GPU，DeepSpeed ZeRO-2
- **评估指标**：
  - **Accuracy**：任务完成准确率
  - **TPS (Tokens Per Step)**：每步预测的有效 token 数量，衡量并行性。越高越好。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **原始 DLM** | LLaDA, Dream |
| **无训练加速**（train-free） | Fast-dLLM, Fast-dLLM-v2 |
| **训练加速方法** | D2F, dParallel, d3LLM（均为蒸馏类） |

> 所有方法均使用相同的 backbone 和 sampling 策略（confidence-based sampling from Fast-dLLM）进行公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 & 2）

#### 在 LLaDA 上的结果（Table 1）
| Method | GSM8K Acc (%) | TPS | MATH Acc (%) | TPS |
|--------|----------------|-----|---------------|-----|
| LLaDA | 72.6 | 1.00 | 32.2 | 1.00 |
| Fast-dLLM | 74.7 | 2.77 | 30.8 | 1.97 |
| d3LLM | 73.1 | **9.11** | 30.4 | **5.74** |
| **TRIMS (Ours)** | **74.9** | **6.26** | **34.3** | **4.72** |

> ✅ TRIMS 在保持更高 accuracy 的同时，达到 **6.26× 并行加速**，显著优于大多数 baseline。

#### 在 Dream 上的结果（Table 2）
| Method | MBPP Acc (%) | TPS | HumanEval Acc (%) | TPS |
|--------|----------------|-----|--------------------|-----|
| Dream | 57.2 | 1.00 | 55.2 | 1.00 |
| Fast-dLLM-v2 | 50.1 | 2.04 | 61.7 | 2.58 |
| d3LLM | 55.6 | 2.96 | 57.1 | 3.20 |
| **TRIMS (Ours)** | **56.6** | **6.31** | **57.3** | **2.21** |

> ✅ TRIMS 在 MBPP 上实现高达 **6.31 TPS**，相比 train-free 方法提升超 **3× 并行性**，且精度未下降。

### 与基线方法的对比结果
- **vs. Train-free 方法（Fast-dLLM 等）**：TRIMS 显著提升了 TPS，尤其在长序列任务上优势明显。
- **vs. 蒸馏类方法（dParallel/d3LLM）**：
  - 性能接近甚至部分超越；
  - 但 TRIMS 的训练成本极低（见下表）。

#### 训练开销对比（Table 3）
| Method | Supervision Source | Curation Compute | Train Data Size |
|--------|---------------------|------------------|------------------|
| TRIMS | AR teacher | **0.6 GPU-hours** | **1K** |
| dParallel/d3LLM | DLM distillation | **287 GPU-hours × N(models)** | **93K** |

> 💡 TRIMS 的数据需求仅为蒸馏方法的 **~1%**，计算成本可忽略不计。

### 消融实验结果（Ablation Studies）

#### （1）轨迹监督的作用
- 移除 TRIMS 中的轨迹信号后，性能回落至标准 MDLM 水平。
- TRIMS 在所有 benchmark 上均优于标准训练，尤其在 coding 任务（HumanEval, MBPP）上增益显著。

#### （2）轨迹排序方式的影响
比较三种 bucket 分配策略：
- **Difficulty-descending**（难→易）✅ 最优
- Difficulty-ascending（易→难）
- Random assignment

> 结果显示，“先难后易”策略最有效，符合设计直觉：尽早确定高不确定性 token 可提升后续并行预测能力。

#### （3）桶的数量（K）影响
尝试 $K \in \{4, 8, 16\}$：
- $K=8$ 表现最佳
- $K=4$ 过于粗糙，$K=16$ 过细导致区分困难
> 方法对超参数具有鲁棒性。

#### （4）难度度量的选择
比较两种 AR 输出统计量作为难度指标：
- **Negative Log-Likelihood (NLL)** ✅ 更优
- Entropy

> 在 coding 任务中 NLL 效果更好，因其能更精细地区分语法关键 token。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **显式的轨迹监督至关重要**：标准 DLM 训练缺乏对解码顺序的建模，导致 train-inference mismatch；TRIMS 成功缓解此问题。
2. **“Hard-to-Easy” 解码策略更高效**：优先恢复困难 token 能更快降低整体不确定性，促进后续并行生成。
3. **轻量级信号足够有效**：仅需一个 AR teacher 的 NLL 输出即可指导 DLM 学习优质轨迹，无需复杂的 DLM 采样或强化学习。
4. **TRIMS 实现了高性能与低成本的统一**：在远低于蒸馏方法的成本下，达到与其相当的 accuracy-parallelism trade-off。

### 方法的局限性
- 当前依赖外部 AR teacher 模型（如 Qwen3-8B），若 teacher 偏差较大，可能传递错误难度信号。
- bucketing 策略仍为静态划分，未考虑动态上下文变化。
- 目前仅验证于中小规模 DLM（7B–8B），在更大模型上的可扩展性有待验证。

### 未来工作方向
- 扩展至更大规模 DLM（如 70B 级别）
- 探索更丰富的轨迹信号来源（如多步推理路径、思维链一致性）
- 动态调整 bucket 阈值或引入可学习的调度机制
- 将 TRIMS 思路应用于其他非自回归生成范式（如编辑、重写等）

---

> 🔚 **总结一句话**：  
> TRIMS 通过引入来自 AR teacher 的轻量级 token 难度信号，构建了一种简单高效的轨迹感知训练机制，在几乎零额外成本下显著提升了 DLM 的并行解码效率，为实用化低延迟文本生成提供了新路径。

</details>

---

### 13. [Performance of Neural and Polynomial Operator Surrogates](https://arxiv.org/abs/2604.00689)

**Authors**: Josephine Westermann, Benno Huber, Thomas O'Leary-Roseberry, Jakob Zech  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.00689v1  

#### Abstract
We consider the problem of constructing surrogate operators for parameter-to-solution maps arising from parametric partial differential equations, where repeated forward model evaluations are computationally expensive. We present a systematic empirical comparison of neural operator surrogates, inclu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Performance of Neural and Polynomial Operator Surrogates》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文研究了**参数到解映射**（parameter-to-solution maps）的**算子代理模型**（operator surrogates）在求解**参数化偏微分方程**（parametric PDEs）中的性能表现。这类问题常见于科学计算中的不确定性量化、反问题和优化任务，其中每次前向模拟（forward model evaluation）成本高昂，因此需要构建高效且准确的代理模型来替代原始高维PDE求解器。

### 提出的新方法与新思路
论文并未提出全新的算法，而是进行了**系统性的实证比较**，其核心创新在于：

- **统一的评估框架**：将神经算子（neural operators）与多项式代理方法（polynomial surrogates）置于同一基准下进行公平比较，涵盖从数据生成、训练/构造到评估的全流程成本。
- **多维度性能分析**：不仅比较精度，还分解了总成本为**数据生成成本**（data generation）、**构造成本**（setup cost）和**评估成本**（evaluation cost），并绘制**帕累托前沿**（Pareto frontiers）以揭示不同方法在“成本-精度”权衡上的优势区域。
- **正则性依赖分析**：系统地研究了输入场的**光滑性**（regularity，由谱系数衰减速率 $s$ 控制）对各类代理模型性能的影响，揭示了方法选择应与问题正则性匹配。

### 相比现有方法的优势
- **超越单一架构比较**：不同于以往仅比较不同神经算子架构的工作，本文将具有坚实逼近理论基础的**多项式方法**（如 sparse grid 和 tensor-train）纳入比较，提供了更全面的视角。
- **强调数据效率**：通过分析收敛速率与训练样本数的关系，突出了多项式方法在光滑问题上的**卓越数据效率**。
- **实用导向**：考虑了实际应用中的约束，如 Jacobian 信息是否可得，并评估了**导数知情训练**（derivative-informed training）的实际收益。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验基于两个典型的参数化PDE问题，均定义在单位正方形 $[0,1]^2$ 上，使用 $64 \times 64$ 的均匀网格离散：

1.  **线性参数扩散方程**（Linear parametric diffusion problem）：
    $$
    -\nabla \cdot (e^{x(c)} \nabla y) = 1, \quad y|_{\partial \Omega} = 0
    $$
    其中参数 $x(c)$ 是一个对数渗透率场。

2.  **非线性参数超弹性力学问题**（Nonlinear parametric hyperelasticity problem）：
    描述一个二维超弹性材料在应力作用下的变形，其本构关系是非线性的，导致求解的PDE也是非线性的。

参数 $x(c)$ 本身是一个随机场，通过控制其Matérn协方差特征值的衰减速率 $s$ 来调节其光滑性：
$$
x(c) = \sum_{j=1}^{d_{\text{true}}} c_j j^{-s} \phi_j, \quad c_j \in [-1, 1]
$$
$s$ 越大，输入场越光滑。

### 实验设置和评估指标
- **代理模型**：
  - **Neural Operators**:
    - `L2-RB-NO`: Reduced-Basis Neural Operator，使用 $L^2$ 目标函数训练。
    - `H1-RB-NO`: Reduced-Basis Neural Operator，使用包含导数项的 $H^1$ 目标函数（derivative-informed training）。
    - `FNO`: Fourier Neural Operator，使用 $L^2$ 目标函数训练。
  - **Polynomial Surrogates**:
    - `RB-SG`: Reduced-Basis Sparse-Grid surrogate。
    - `RB-TT`: Reduced-Basis Tensor-Train surrogate（仅用于线性扩散问题）。

- **评估指标**：
  1.  **$L^2$-error** ($e_{L^2}$): 相对均方根误差，衡量输出预测的准确性。
  2.  **$H^1$-error** ($e_{H^1}$): 包含雅可比矩阵（Jacobian）的相对误差，衡量梯度信息的准确性。
  3.  **训练数据量** ($n$): 用于训练或构造代理模型的PDE求解次数，代表**数据生成成本**。
  4.  **训练/构造时间** ($t_T$): 代表**构造成本**（不包括数据生成时间）。
  5.  **评估时间** ($t_E$): 应用代理模型进行一次前向预测的平均耗时，代表**在线评估成本**。

- **实验流程**：
  对每种代理模型，通过**调整超参数**（如网络深度/宽度、稀疏网格参数、TT秩等）生成一系列配置，然后在不同 $s$ 值下进行训练和测试，最终绘制出各方法的“成本-精度”帕累托前沿进行对比。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
实验结果表明，**没有一种方法在所有场景下都占优**，最优选择高度依赖于问题的特性。

#### (1) 数据效率与输入光滑性 ($s$) 的关系
- **对于光滑输入 ($s \geq 2$)**:
  - **`RB-SG` 和 `RB-TT` 表现出色**：它们的数据效率远高于神经算子，收敛速度与理论预测一致。特别是 `RB-SG` 在高精度要求下能达到更低的误差。
  - **`FNO` 和 `L2-RB-NO` 表现较差**：收敛缓慢，需要更多数据才能达到同等精度。
- **对于粗糙输入 ($s \leq 1$)**:
  - **`FNO` 收敛最快**：在低数据量下即能取得较好的近似效果，展现出对高频成分的良好捕捉能力。
  - **多项式方法 (`RB-SG`, `RB-TT`) 收敛极慢**，难以处理此类问题。

#### (2) 导数知情训练 (`H1-RB-NO`)
- **显著提升数据效率**：相比 `L2-RB-NO`，`H1-RB-NO` 在相同数据量下能获得更高的精度，尤其在**低数据量**和**粗糙输入** ($s \leq 1$) 场景下优势明显。
- **代价是更高的构造成本**：因为需要额外计算和利用Jacobian信息，其训练时间 $t_T$ 远高于 `L2-RB-NO`。

#### (3) 构造成本 ($t_T$)
- **多项式方法 (`RB-SG`) 极快**：其构造过程主要是线性代数运算和插值，无需迭代优化，因此 $t_T$ 非常低。
- **神经算子成本高昂**：`L2-RB-NO`, `H1-RB-NO`, `FNO` 都需要长时间的ADAM优化训练，$t_T$ 显著更高。
- **`RB-TT` 居中**：其ALS-cross算法有迭代过程，$t_T$ 高于 `RB-SG` 但低于神经算子。

#### (4) 评估成本 ($t_E$)
- **神经算子 (`FNO`) 评估最慢**：由于其复杂的架构（频域卷积、逐点非线性），单次评估耗时最长，且**批处理扩展性差**（batch size增大时，单样本耗时下降有限）。
- **`RB-SG` 和 `RB-TT` 评估非常快**：一旦构造完成，评估就是高效的张量运算或插值，$t_E$ 很低，且**批处理扩展性好**。
- **`L2/H1-RB-NO` 评估较快**：作为全连接网络，其评估效率尚可。

---

## 4. 关键结论和发现

### 主要发现
1.  **无普适最优方法**：代理模型的选择必须与具体问题的**正则性**（smoothness of the input field）相匹配。**“No single method is universally superior.”**
2.  **光滑问题首选多项式方法**：当输入场光滑 ($s \geq 2$) 时，`RB-SG` 和 `RB-TT` 在**数据效率**和**评估效率**上具有压倒性优势，是更优选择。
3.  **粗糙问题首选FNO**：当输入场粗糙 ($s \leq 1$) 时，`FNO` 凭借其频域建模能力，能实现最快的收敛。
4.  **导数信息是强大先验**：如果Jacobian信息可以低成本获取（例如通过伴随法），**`H1-RB-NO` 是一个极具竞争力的选项**，尤其是在低数据量和粗糙输入场景下，能极大提升数据效率。
5.  **成本构成差异巨大**：神经算子的主要瓶颈在**构造成本**（训练时间），而多项式方法的主要瓶颈在**数据成本**（所需样本数）。在数据生成极其昂贵的应用中，即使神经算子训练时间长，其可能仍是更好的选择。

### 方法的局限性
- **`RB-TT` 的稳定性问题**：在极高精度要求下，`RB-TT` 可能出现误差平台期（plateauing error），其数值稳定性有待进一步研究。
- **`FNO` 的灵活性限制**：`FNO` 严重依赖快速傅里叶变换（FFT），通常只适用于规则网格和矩形域，难以处理复杂几何形状。
- **`RB-SG` 的适用范围**：虽然简单稳定，但其有效性依赖于参数空间的低维或强各向异性结构。
- **`H1-RB-NO` 的数据依赖**：其优势完全建立在Jacobian信息可用的基础上，在无法高效计算导数的场景下不适用。

### 未来工作方向
- **混合方法**：探索结合神经算子和多项式方法优势的混合架构。
- **自适应方法**：开发能够自动感知输入场正则性 $s$ 并动态选择最优代理策略的框架。
- **复杂几何的推广**：改进 `FNO` 或发展新的神经算子，使其能更有效地处理非规则域。
- **`RB-TT` 的优化**：研究提高 `RB-TT` 在高精度下的数值稳定性和计算效率的方法，例如将其移植到GPU上运行。

</details>

---

### 14. [Adaptive Parallel Monte Carlo Tree Search for Efficient Test-time Compute Scaling](https://arxiv.org/abs/2604.00510)

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

### ✅ 解决的问题
Monte Carlo Tree Search (MCTS) 是一种有效的 **Test-time Compute Scaling (TTCS)** 方法，能显著提升大语言模型（LLM）在复杂推理任务中的准确性。然而，其**高度可变的执行时间**导致严重的长尾延迟（long-tail latency），尤其在高并发服务场景下，少数计算密集型请求会严重拖慢整体系统响应，限制了其实际部署能力。

现有优化如 **positive early exit** 能在早期发现高质量解时提前终止搜索，但在搜索长期无进展的情况下无效，无法解决“低效探索”带来的资源浪费。

---

### 🚀 提出的新方法与创新思路

本文从**系统级资源管理视角**重构 MCTS 推理流程，提出两个核心技术：

#### （1）**Negative Early Exit（负向早退机制）**
- **思想**：识别并剪枝那些**不可能产生高质量解的无前途搜索路径**。
- **实现**：基于轨迹评分聚合方式（如 cumulative product 或 min），若当前搜索树中所有叶节点的得分均低于接受阈值 $\tau$，则判定为“futile”，立即终止该请求的进一步 rollout。
- **优势**：避免对注定失败的任务进行冗余计算，直接减少尾部延迟。

#### （2）**Adaptive Boosting（自适应加速机制）**
- **思想**：将 negative early exit 释放出的计算资源**动态重新分配**给更有潜力的并发任务。
- **实现**：
  - 引入 **Parallelism Degree Control Policy**，根据任务等待时间和当前进度（是否接近正向早退阈值）动态调整每个任务的并行度（Degree of Parallelism, DOP）。
  - 对即将完成的任务进行“boosting”（即增加并行 rollout 数量），加快其完成速度，快速释放资源。
- **优势**：实现资源的高效再利用，在不增加总计算预算的前提下提升吞吐量、降低尾延迟。

此外，还引入了：
- **Selective Futility Check**：利用首步奖励与最终得分的相关性，跳过低初分分支的 futility 判断，提高检查效率。
- **WU-PUCT**：扩展 WU-UCT 思想至 PUCT 框架，支持使用“飞行中”统计量进行节点选择，提升并行 rollout 的协调性。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | 本工作的改进 |
|------|--------|--------------|
| Vanilla MCTS | 串行执行，尾延迟高 | 并行化 + 早退机制大幅降低延迟 |
| Positive Early Exit | 只能加速“成功快”的请求 | 新增 Negative Early Exit 加速“失败确定”的请求 |
| Parallel MCTS (e.g., WU-UCT) | 固定并行度，易造成资源争用 | 动态控制并行度，按需分配资源 |
| Beam Search | 计算效率低，生成冗余 token | 更精准的搜索控制，节省 token 和延迟 |

> ✅ **核心优势总结**：  
> 在保持 MCTS 高准确性的前提下，**系统性地缓解了长尾延迟问题**，实现了**低延迟、高吞吐、高资源利用率**的推理服务。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **Math500**：来自 MATH 数据集的 500 道数学题，用于评估逐步推理准确性。
- **AMC23**：2023 年美国数学竞赛题目，用于测试泛化能力（仅 40 题，因此性能测试时通过添加不同前缀扩增负载）。

### ⚙️ 实验设置
- **硬件平台**：单节点，4× NVIDIA H100-SXM 80GB GPU（NVLink 连接）
  - 2 GPUs 用于生成模型推理
  - 2 GPUs 用于 Reward Model 推理
- **生成模型**：
  - `Llama-3.1-8B-Instruct`
  - `Qwen2.5-14B-Instruct`
- **奖励模型**：`Qwen2.5-Math-PRM-7B`（Process Reward Model）
- **集成框架**：基于 **vLLM** 构建系统原型，支持高效的内存管理和调度。

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **p50 / p99 End-to-End Latency** | 请求端到端延迟的中位数与第99百分位数（关注尾延迟） |
| **Throughput (req/sec)** | 单位时间内处理的请求数量 |
| **Accuracy (%)** | 正确解答的比例（step-by-step solution matching） |
| **Avg. Generated Tokens** | 平均每请求生成的 token 数量（衡量计算开销） |

### 🔀 基线方法对比
| 配置 | 描述 |
|------|------|
| **Beam Search** | Beam Width=8，启用 Positive Early Exit |
| **Vanilla MCTS** | 串行 MCTS，无任何优化 |
| **PE (Positive Early Exit)** | 达到信心阈值后提前退出 |
| **PE+NE** | PE + Negative Early Exit |
| **PE+NE+Boosting** | 完整系统，含资源再分配机制 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（综合最差情况 vs Vanilla MCTS）

| 指标 | 提升幅度 |
|------|----------|
| **p99 Latency** | ↓ 最多 **2.83×** |
| **Throughput** | ↑ 最多 **2.44×** |
| **Token Generation** | 显著减少（见图2） |

---

### 🔁 与基线方法的对比结果（Figure 7 & 8）

#### （1）端到端延迟（p99）
- **Vanilla MCTS vs Beam Search**：平均降低 1.64× p99 延迟
- **PE vs Vanilla**：额外降低 1.38× p99
- **PE+NE vs PE**：再降 1.08× p99 → 表明 NE 有效处理难例
- **PE+NE+Boosting vs PE+NE**：再降 1.15× p99（除 `(amc23, Qwen)` 外普遍有效）

> 💡 在 `(Math500, Llama)` 上，p99 从 1,886ms 降至 1,277ms（↓1.47×）

#### （2）吞吐量（Throughput）
- **PE**：+1.3× 吞吐
- **PE+NE**：+1.36× 吞吐
- **PE+NE+Boosting**：
  - `(Math500, Llama)`：+1.37×
  - `(Math500, Qwen)`：+1.76×
  - `(amc23, Llama)`：+1.15×
- **例外**：`(amc23, Qwen)` 中 Boosting 导致吞吐下降至 0.8× NE —— 因多数请求简单，过度并行反而引发调度开销。

#### （3）准确性（Table 1）
| 方法 | Math500 (Llama) | AMC23 (Llama) |
|------|------------------|---------------|
| Beam Search | 72.9% | 52.5% |
| Vanilla MCTS | 75.3% | 57.5% |
| PE | 76.2% | 47.5% |
| **Ours (Full Suite)** | **74.8%** | **55.0%** |

> ✅ 准确率基本持平或略有波动，**未因优化而牺牲精度**

---

### 🔍 消融实验分析
- **Negative Early Exit 单独作用**：显著改善 p99 延迟，尤其在高负载下效果明显。
- **Boosting 的价值**：在存在大量可回收计算资源的场景下（如难题较多），能大幅提升吞吐；但在简单任务主导时可能因并行开销适得其反。
- **Selective Futility Check**：提升 negative exit 触发频率约 5% → 15%，增强资源回收效率。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MCTS 的部署瓶颈不在准确性，而在资源管理不均**。尾延迟本质上是系统级问题，需结合**动态调度 + 早退机制**解决。
2. **Negative Early Exit 是填补现有优化空白的关键机制**：它解决了 positive early exit 无法覆盖的“持续低质量探索”场景。
3. **资源再分配（Boosting）能显著放大早退收益**：将“省下来的算力”用于加速其他任务，形成正反馈循环。
4. 所提方法在真实推理负载下实现了 **p99 延迟 ↓2.83×、吞吐 ↑2.44×**，且**保持原有推理精度不变**。

---

### ⚠️ 方法的局限性
1. **依赖高质量 PRM（Process Reward Model）**：negative exit 的有效性建立在 reward 可预测最终成功率的基础上。若 PRM 不可靠，可能导致误剪枝。
2. **并行开销敏感**：在轻负载或简单任务为主的工作流中，过度并行可能因调度和通信开销反而降低性能（如 `(amc23, Qwen)` 场景）。
3. **当前实现在单节点内调度**：未考虑跨节点分布式 MCTS 的扩展性。

---

### 🔮 未来工作方向
1. **更智能的 exit 判定机制**：结合学习型模型预测 rollout 成功率，替代固定阈值判断。
2. **异构任务混合调度**：在同一系统中同时处理 MCTS、beam search、best-of-N 等多种 TTCS 策略的请求。
3. **面向分布式环境的扩展**：设计支持多机协同的 Adaptive Parallel MCTS 框架。
4. **自动化参数调优**：动态调整 $\tau$、T（proximity threshold）、最大并行度等超参以适应不同 workload。

---

> ✅ **总体评价**：  
> 本文成功将 MCTS 从一个“强但慢”的推理算法转变为一个**高效、可控、适合生产部署的系统组件**，为大规模 LLM 推理服务平台提供了重要的工程实践范式。

</details>

---

### 15. [More Human, More Efficient: Aligning Annotations with Quantized SLMs](https://arxiv.org/abs/2604.00586)

**Authors**: Jiayu Wang, Junyoung Lee  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.00586v1  

#### Abstract
As Large Language Model (LLM) capabilities advance, the demand for high-quality annotation of exponentially increasing text corpora has outpaced human capacity, leading to the widespread adoption of LLMs in automatic evaluation and annotation. However, proprietary LLMs often exhibit systematic biase...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《More Human, More Efficient: Aligning Annotations with Quantized SLMs》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
随着 Large Language Model (LLM) 在生成任务中的广泛应用，对大规模文本语料进行高质量人工标注的需求急剧增长，已远超人类标注能力。因此，业界广泛采用 LLM-as-a-Judge (LaaJ) 或 LLM-as-an-Annotator 范式实现自动评估与标注。

然而，**依赖 proprietary LLMs 存在以下系统性缺陷**：
- **系统性偏差**：如 position bias、verbosity bias、低 perplexity 偏好等；
- **缺乏可复现性**：API 黑箱、版本不透明；
- **数据隐私风险**：敏感领域（如法律、医疗）中数据外泄；
- **与人类专家共识偏离**：导致错误引导模型训练。

该论文旨在解决上述问题，提出一种更可靠、可复现、对齐人类判断的自动化标注方案。

---

### 🚀 提出的新方法与创新思路
作者提出了一种基于 **quantized Small Language Model (SLM)** 的监督微调 pipeline，核心创新包括：

1. **任务特定对齐（Task-Specific Alignment）**  
   在少量高质量 human-annotated 数据上 fine-tune 一个轻量级 SLM（Qwen3-1.7B），使其在特定评估维度上与人类专家高度一致。

2. **多维评分框架（Multi-Dimensional Rubric Framework）**  
   构建了一个分层的六维评价标准：
   - Completeness
   - Clarity
   - Interpretability
   - Conciseness
   - Accuracy
   - Relevance  
   每个维度使用 {-2, -1, 0, 1, 2} 的有序尺度打分，提升细粒度控制。

3. **定制化数据增强与正则化策略**  
   针对小样本场景设计三种技术防止过拟合：
   - **Prompt Paraphrasing**：改变指令表述形式，增强语法鲁棒性；
   - **Component Permutation**：随机打乱输入组件顺序（QUESTION/CONTEXT/ANSWER），缓解 position bias；
   - **Token Dropout**：训练时随机掩码非关键 token，作为结构化正则手段。

4. **4-bit Quantized PEFT 微调**  
   使用 Unsloth 库进行 4-bit 量化 + LoRA 参数高效微调，可在单张消费级 GPU 上运行，适合本地部署。

5. **Completion-Based Training Formulation**  
   将评分任务建模为 causal language modeling 问题：将 human annotations 附加为 completion string，让模型“生成”分数而非分类预测。

---

### 🔍 相比现有方法的优势
| 维度 | 本方法优势 |
|------|-----------|
| **对齐性** | 显著高于所有 tested proprietary LLMs 的 inter-annotator agreement |
| **可复现性** | 开源、确定性输出，避免黑盒不确定性 |
| **成本与效率** | SLM 可本地部署，无需昂贵 API 调用 |
| **隐私安全** | 支持离线处理，保障数据主权 |
| **灵活性** | 可适配不同 rubric 和任务类型 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

#### （1）SPS Dataset（主实验）
- 来源：新加坡监狱服务局（Singapore Prison Service, SPS）官网爬取内容
- 内容涵盖：职业信息、康复项目、年报等
- 构成：
  - 97 个人工编写的问题
  - 每个问题对应 7 个由不同 open-source SLMs 生成的回答（共 679 条候选响应）
- 回答来源模型见 Appendix A.2（如 Llama-3.2、Gemma-3、Qwen 等）

#### （2）GoEmotions Dataset（泛化性验证）
- 公开情感分类数据集（Demszky et al., 2020）
- 包含约 58k 社交媒体评论，标注 27 种细粒度情绪类别
- 用于验证方法在标准 classification 任务上的通用性

---

### 🧪 实验设置与评估指标

#### 评估指标
- **SPS Dataset**：使用 **Krippendorff’s α** 衡量 inter-annotator agreement  
  → 衡量模型评分与 human annotators 的一致性程度（考虑多评分者、有序尺度、差异幅度）
- **GoEmotions Dataset**：使用标准分类指标
  - Accuracy
  - Macro-F1

#### 微调配置（详见 Table 3）
| 参数 | 设置 |
|------|------|
| 模型 | Qwen3-1.7B |
| 量化方式 | 4-bit |
| 微调方式 | LoRA (r=16) |
| 精度 | bfloat16 |
| Epochs | 5 |
| Batch Size | 32 |
| Learning Rate | 5e-5 |
| LR Scheduler | Linear with 5% warmup |
| 硬件 | NVIDIA A100 |

---

### ⚖️ 基线方法对比

#### Proprietary LLM Baselines
- **Zero-shot**：
  - GPT-4o
  - GPT-5-nano
  - GPT-5-mini-2025-08-07
  - GPT-5.2-chat
- **Few-shot**（通过 dspy + MIPROv2 优化 prompt）：
  - GPT-4o
  - GPT-5.2-chat

#### Open-Source SLM Baselines（消融实验）
- Full fine-tuning without augmentation
- Early stopping without augmentation
- LoRA Dropout (p=0.1)

> 注：原始 Qwen3-1.7B 无法有效学习标签，未列入 baseline。

---

## 3. 主要实验结果和性能指标

### 📊 SPS Dataset 结果（Table 1）

| 方法 | Krippendorff’s α |
|------|------------------|
| **Our Method** | **0.5774** ✅ |
| Full finetuning w/o aug | 0.4304 |
| Early stopping w/o aug | 0.4380 |
| LoRA Dropout | 0.4067 |
| Zero-shot GPT-4o | 0.1964 |
| Zero-shot GPT-5-mini | 0.2462 |
| Few-shot GPT-4o | 0.0101 |

> 💡 **关键发现**：
- 本方法比最佳 proprietary 模型（GPT-5-mini）高出 **+0.3312 α points**
- 比 GPT-4o 提高 **+0.3810 α points**
- 所有 few-shot 商业模型表现极差，甚至低于随机水平（α ≈ 0）

---

### 📈 GoEmotions Dataset 结果（Table 2）

| 方法 | Accuracy | Macro-F1 |
|------|----------|----------|
| **Our Method** | **0.8163** ✅ | **0.6380** ✅ |
| Full SFT w/o aug | 0.7819 | 0.4967 |
| Zero-shot GPT-4o | 0.4741 | 0.3732 |
| Zero-shot GPT-5.2-chat | 0.5062 | 0.4099 |
| Few-shot GPT-5.2-chat | 0.4990 | 0.3926 |

> 💡 **关键发现**：
- 准确率接近 **82%**，几乎是 GPT-4o 的两倍
- 表明方法具有良好的 **task-agnostic generalizability**

---

### 🔍 消融实验结果（Ablation Study）

从 Table 1 和附录 Figure 2–4 可知：
- **数据增强显著提升性能**：
  - 加入 augmentation 后 α 提升约 **+0.14~0.17**
- **Loss 曲线显示更强泛化能力**：
  - 无增强模型出现明显过拟合（val loss 上升）
  - 使用 augmentation 后 train/val loss 更平稳收敛
- **LoRA Dropout 效果有限**，不如 proposed augmentation 有效

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **小型量化模型经任务对齐后可超越大型 proprietary LLMs**  
   即便只有 1.7B 参数，在高质量 human annotations 上 fine-tune 并结合合理 augmentation，其 human-alignment 性能远超 GPT-4o/GPT-5 系列。

2. **任务特定对齐 > 通用推理能力**  
   对于抽象、多维、主观性强的 annotation 任务，**focused specialization** 比 general-purpose reasoning 更有效。

3. **数据质量 > 模型规模**  
   高质量标注 + 定制 rubric 是成功的关键，而非盲目追求更大模型。

4. **本地化 SLM 是可行替代方案**  
   可解决 black-box 模型带来的 reproducibility、bias、privacy 等核心挑战，推动 democratized AI annotation。

---

### ⚠️ 局限性（Limitations）
- **仅测试 GPT 系列 API**：受限于访问权限，未包含 Claude、Gemini 等其他主流 proprietary models；
- **领域覆盖有限**：当前实验集中于政府信息服务与情感分析，需进一步验证在医学、法律等复杂领域的适用性；
- **rubric 设计依赖人工**：多维评分体系构建需要领域专家参与，自动化程度有待提高。

---

### 🔮 未来工作方向
- 探索自动化 rubric design 与动态权重调整机制；
- 扩展至更多 downstream labeling tasks（如 toxicity detection, fact-checking）；
- 研究如何将 human disagreement 显式建模进训练过程（参考 DisCo 方法）；
- 推动开源社区共建 aligned SLM annotator benchmarks。

---

## 🔗 开源信息
- 代码仓库公开地址：[https://github.com/jylee-k/slm-judge](https://github.com/jylee-k/slm-judge)  
- 支持快速部署与复现，促进透明、可信的 AI 评估生态建设。

</details>

---

### 16. [LangMARL: Natural Language Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2604.00722)

**Authors**: Huaiyuan Yao, Longchao Da, Xiaoou Liu, Charles Fleming, Tianlong Chen, Hua Wei  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.00722v1  

#### Abstract
Large language model (LLM) agents struggle to autonomously evolve coordination strategies in dynamic environments, largely because coarse global outcomes obscure the causal signals needed for local policy refinement. We identify this bottleneck as a multi-agent credit assignment problem, which has l...

---

### 17. [LLM REgression with a Latent Iterative State Head](https://arxiv.org/abs/2604.01206)

**Authors**: Yiheng Su, Matthew Lease  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.01206v1  

#### Abstract
We present RELISH (REgression with a Latent Iterative State Head), a novel, lightweight architecture designed for text regression with large language models. Rather than decoding numeric targets as text or aggregating multiple generated outputs, RELISH predicts scalar values directly from frozen LLM...

---

### 18. [Predicting Dynamics of Ultra-Large Complex Systems by Inferring Governing Equations](https://arxiv.org/abs/2604.00599)

**Authors**: Qi Shao, Duxin Chen, Jiawen Chen, Yujie Zeng, Athen Ma, Wenwu Yu, Vito Latora, Wei Lin  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.00599v1  

#### Abstract
Predicting the behavior of ultra-large complex systems, from climate to biological and technological networks, is a central unsolved challenge. Existing approaches face a fundamental trade-off: equation discovery methods provide interpretability but fail to scale, while neural networks scale but ope...

---

### 19. [Reconsidering Dependency Networks from an Information Geometry Perspective](https://arxiv.org/abs/2604.01117)

**Authors**: Kazuya Takabatake, Shotaro Akaho  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.01117v1  

#### Abstract
Dependency networks (Heckerman et al., 2000) provide a flexible framework for modeling complex systems with many variables by combining independently learned local conditional distributions through pseudo-Gibbs sampling. Despite their computational advantages over Bayesian and Markov networks, the t...

---

### 20. [Does Unification Come at a Cost? Uni-SafeBench: A Safety Benchmark for Unified Multimodal Large Models](https://arxiv.org/abs/2604.00547)

**Authors**: Zixiang Peng, Yongxiu Xu, Qinyi Zhang, Jiexun Shen, Yifan Zhang, Hongbo Xu, Yubin Wang, Gaopeng Gou  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.00547v1  

#### Abstract
Unified Multimodal Large Models (UMLMs) integrate understanding and generation capabilities within a single architecture. While this architectural unification, driven by the deep fusion of multimodal features, enhances model performance, it also introduces important yet underexplored safety challeng...

---

### 21. [Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts](https://arxiv.org/abs/2604.00901)

**Authors**: Sha Li, Naren Ramakrishnan  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.00901v1  

#### Abstract
Multi-agent Retrieval-Augmented Generation (RAG), wherein each agent takes on a specific role, supports hard queries that require multiple steps and sources, or complex reasoning. Existing approaches, however, rely on static agent behaviors and fixed orchestration strategies, leading to brittle perf...

---

### 22. [Full-Gradient Successor Feature Representations](https://arxiv.org/abs/2604.00686)

**Authors**: Ritish Shrirao, Aditya Priyadarshi, Raghuram Bharadwaj Diddigi  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.00686v1  

#### Abstract
Successor Features (SF) combined with Generalized Policy Improvement (GPI) provide a robust framework for transfer learning in Reinforcement Learning (RL) by decoupling environment dynamics from reward functions. However, standard SF learning methods typically rely on semi-gradient Temporal Differen...

---

### 23. [Using predefined vector systems to speed up neural network multimillion class classification](https://arxiv.org/abs/2604.00779)

**Authors**: Nikita Gabdullin, Ilya Androsov  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.00779v1  

#### Abstract
Label prediction in neural networks (NNs) has O(n) complexity proportional to the number of classes. This holds true for classification using fully connected layers and cosine similarity with some set of class prototypes. In this paper we show that if NN latent space (LS) geometry is known and posse...

---

### 24. [How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic Study](https://arxiv.org/abs/2604.00005)

**Authors**: Moran Sun, Tianlin Li, Yuwei Zheng, Zhenhong Zhou, Aishan Liu, Xianglong Liu, Yang Liu  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.00005v1  

#### Abstract
Emotion plays an important role in human cognition and performance. Motivated by this, we investigate whether analogous emotional signals can shape the behavior of large language models (LLMs) and agents. Existing emotion-aware studies mainly treat emotion as a surface-level style factor or a percep...

---

### 25. [The Silicon Mirror: Dynamic Behavioral Gating for Anti-Sycophancy in LLM Agents](https://arxiv.org/abs/2604.00478)

**Authors**: Harshee Jignesh Shah (Independent Researcher)  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.00478v1  

#### Abstract
Large Language Models (LLMs) increasingly prioritize user validation over epistemic accuracy-a phenomenon known as sycophancy. We present The Silicon Mirror, an orchestration framework that dynamically detects user persuasion tactics and adjusts AI behavior to maintain factual integrity. Our archite...

---

### 26. [Ontology-Constrained Neural Reasoning in Enterprise Agentic Systems: A Neurosymbolic Architecture for Domain-Grounded AI Agents](https://arxiv.org/abs/2604.00555)

**Authors**: Thanh Luong Tuan  
**Category**: cs.AI  
**Published**: 2026-04-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.00555v1  

#### Abstract
Enterprise adoption of Large Language Models (LLMs) is constrained by hallucination, domain drift, and the inability to enforce regulatory compliance at the reasoning level. We present a neurosymbolic architecture implemented within the Foundation AgenticOS (FAOS) platform that addresses these limit...

---

### 27. [A Taxonomy of Programming Languages for Code Generation](https://arxiv.org/abs/2604.00239)

**Authors**: Nishat Raihan, Christian Newman, Marcos Zampieri  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.00239v1  

#### Abstract
The world's 7,000+ languages vary widely in the availability of resources for NLP, motivating efforts to systematically categorize them by their degree of resourcefulness (Joshi et al., 2020). A similar disparity exists among programming languages (PLs); however, no resource-tier taxonomy has been e...

---

### 28. [From Baselines to Preferences: A Comparative Study of LoRA/QLoRA and Preference Optimization for Mental Health Text Classification](https://arxiv.org/abs/2604.00773)

**Authors**: Mihael Arcan  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.00773v1  

#### Abstract
Mental health text classification has rapidly adopted modern adaptation methods, yet practical guidance on which optimization strategy to use, when, and why remains limited. This paper presents a systematic comparative study of optimization pathways for a joint mental-health classification task, mov...

---

### 29. [Positional Cognitive Specialization: Where Do LLMs Learn To Comprehend and Speak Your Language?](https://arxiv.org/abs/2604.00923)

**Authors**: Luis Frentzen Salim, Lun-Wei Ku, Hsing-Kuo Kenneth Pao  
**Category**: cs.CL  
**Published**: 2026-04-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.00923v1  

#### Abstract
Adapting large language models (LLMs) to new languages is an expensive and opaque process. Understanding how language models acquire new languages and multilingual abilities is key to achieve efficient adaptation. Prior work on multilingual interpretability research focuses primarily on how trained ...

---

### 30. [Speeding Up Mixed-Integer Programming Solvers with Sparse Learning for Branching](https://arxiv.org/abs/2604.00094)

**Authors**: Selin Bayramo\u{g}lu, George L Nemhauser, Nikolaos V Sahinidis  
**Category**: cs.LG  
**Published**: 2026-04-02  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.00094v1  

#### Abstract
Machine learning is increasingly used to improve decisions within branch-and-bound algorithms for mixed-integer programming. Many existing approaches rely on deep learning, which often requires very large training datasets and substantial computational resources for both training and deployment, typ...

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
