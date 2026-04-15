# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-15 07:15:59 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving](https://arxiv.org/abs/2604.12171)

**Authors**: Xu Bai, Muhammed Tawfiqul Islam, Chen Wang, Adel N. Toosi  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.12171v1  

#### Abstract
Pipeline parallelism (PP) is widely used to partition layers of large language models (LLMs) across GPUs, enabling scalable inference for large models. However, existing systems rely on static PP configurations that fail to adapt to dynamic settings, such as serverless platforms and heterogeneous GP...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
当前的 **Pipeline Parallelism (PP)** 在 LLM 推理中被广泛用于跨 GPU 分割模型层，以实现可扩展推理。然而，**现有系统依赖静态 PP 配置**，无法适应动态工作负载（如 serverless 平台、异构 GPU 环境）的变化。

当需要重新配置 PP 时，传统方式需停止服务并重新部署，导致**分钟级停机时间**，破坏正在进行的推理任务。这在动态场景下是不可接受的。

此外，**实时原地（live in-place）PP 重配置面临三大挑战**：
1. **GPU 内存饱和**：模型权重和 KV Cache 已占满显存，难以为新层腾出空间。
2. **KV Cache 动态调整困难**：现有系统（如 vLLM）采用连续内存预分配，无法动态缩放 KV Cache。
3. **KV 状态一致性维护难**：直接“停止-复制”会导致长暂停；后台同步则可能因状态持续更新而出现不一致。

---

### ✅ **提出了什么新方法或新思路**

论文提出 **PIPELIVE**，一个支持高效、低干扰的 **live in-place PP reconfiguration** 的 LLM 服务系统，其核心创新包括：

#### 🔹 **统一的 KV Cache 管理机制**
- 扩展 **PageAttention** 支持**非连续 KV 块访问**，允许按块粒度动态分配和释放 KV Cache。
- 引入 **Layer Stacking** 技术：将多个层的 KV 块打包到同一个 GPU 物理内存块中，对齐 CUDA 最小分配粒度（通常 2 MiB），显著减少内部碎片。

#### 🔹 **增量式 KV Paching 机制**
- 受虚拟机热迁移启发，设计 **KV Patching**：在推理过程中**持续同步源与目标配置间的 KV 状态**。
- 通过周期性传输“脏块”（dirty blocks）来缩小差异，最终只需极短暂停即可完成切换。

#### 🔹 **协调协议与运行时控制**
- 设计 **Reconfiguration Coordinator** 和 **Worker-Level Components**，协同执行以下操作：
  - **KV Cache 缩容**：为加载新层预留内存。
  - **异步权重加载**：从 CPU 异步加载新层权重，避免阻塞推理。
  - **并发 KV 迁移与推理**：利用独立 NCCL 通信组，并引入**两阶段握手协议防止死锁**。
  - **安全切换点检测**：监控 KV 同步滞后，决定何时原子切换至新配置。

---

### ✅ **相比现有方法的优势**
| 维度 | 现有方法 | PIPELIVE |
|------|--------|---------|
| **重配置方式** | 静态配置，需重启 | 支持运行时动态切换 |
| **KV Cache 调整** | 固定大小，无法缩放 | 支持动态 resize |
| **服务中断时间** | 数秒甚至数分钟 | **<10ms** |
| **内存利用率** | 存在严重内部碎片 | 通过 layer stacking 显著提升 |
| **适用场景** | 同构、稳定负载 | 异构 GPU、动态负载 |

---

## 2. **核心实验方法和设置**

### 📚 **使用的模型**
- **Llama3-70B**
- **Qwen3-30B**

> 选择大模型以凸显内存压力下的优势。

---

### ⚙️ **实验设置**

#### **硬件环境**
- 异构双卡测试平台：
  - **NVIDIA A100 (80GB)**：高内存带宽（2039 GB/s）
  - **NVIDIA L40S (48GB)**：强计算能力（FP16/BF16 达 733 TFLOPS）
- 节点间通过 **InfiniBand** 互联，使用 **NCCL** 实现 GPU-to-GPU 通信。

#### **软件基础**
- 基于开源框架 **vLLM** 构建。
- 使用 **FlashAttention** 加速注意力计算。

#### **工作负载设计**
- **Pattern-Shifting Benchmark**：交替模拟两种典型负载：
  - **Prefill-heavy**：输入 512 tokens，输出 16 tokens
  - **Decode-heavy**：输入 128 tokens，输出 512 tokens
- 请求速率变化（1–5 req/s），共发送 200 个请求。

---

### 📊 **评估指标**
| 指标 | 描述 |
|------|------|
| **TTFT (Time-to-First-Token)** | 用户请求到首 token 返回的延迟 |
| **TPOT (Time-per-Output-Token)** | 每个生成 token 的平均延迟 |
| **Throughput (tok/s)** | 总体吞吐量（含输入和输出 token） |
| **Composite Score** | 对 TTFT、TPOT、Throughput 归一化后加权平均，综合评价性能 |

---

### 🆚 **基线方法对比**
1. **Static Baselines**：
   - **Prefill-Optimal**：针对 prefill-heavy 场景优化的固定配置
   - **Decode-Optimal**：针对 decode-heavy 场景优化的配置
   - **Balanced**：折中全局表现的静态配置
2. **PIPELIVE 变体（消融实验）**：
   - **无 KV Resize**：禁用 KV Cache 动态调整
   - **无 KV Patching**：关闭增量同步，改为全量复制
   - **无异步加载 + 无 KV Patch**

---

## 3. **主要实验结果和性能指标**

### 📈 **端到端性能提升**

| 模型 | 方法 | 性能得分提升 |
|------|------|-------------|
| Llama3-70B | PIPELIVE vs Balanced | **+36%** |
| Qwen3-30B | PIPELIVE vs Balanced | **+33%** |

> 表明动态切换最优配置可显著超越任何单一静态策略。

#### 🔍 具体指标改进（Llama3-70B）：
- **TTFT 最多降低 54.7%**
- **TPOT 最多降低 14.7%**
- 在高负载下仍保持稳定，而静态配置出现严重性能退化。

---

### 🔁 **KV Resize 的影响（图 10）**
- **禁用 KV Resize**：
  - 在负载切换后迅速发生 **KV Cache Overflow**
  - 导致 **TTFT 激增**（尤其在请求率 >1 时）
- **启用 KV Resize**：
  - 成功避免溢出
  - 在请求率达 2.5 时仍保持稳定
  - **高负载下吞吐提升超 45%**

> 结论：**KV Cache 动态调整是实现在不同配置间平滑切换的前提条件**。

---

### 🧱 **Layer Stacking 对内存效率的影响（图 11）**
- **无 stacking**：KV 利用率仅 **56%**，近半内存浪费于内部碎片。
- **stacking factor = 4**：利用率提升至 **~95%**
- 图 12 显示：stacking factor 过大会限制重配置灵活性，factor=4 是最佳平衡点。

> 结论：**layer stacking 在减少碎片的同时兼顾了重配置粒度**。

---

### ⏱️ **KV Patching 与异步加载的效果（图 13–14）**

| 设置 | Stop Time | Migration Time |
|------|----------|----------------|
| **Full PIPELIVE** | **~10ms** | 略长（因持续 patch） |
| **No KV Patch** | 数百 ms ~ 数秒 | 更短 |
| **No Async Load + No Patch** | 极长 | 短但完全阻塞 |

#### 性能对比（图 14）：
- 启用 KV Patch 单独带来：
  - **TTFT 最多改善 49.7%**
  - **TPOT 最多改善 29.5%**
- 完整 PIPELIVE：
  - **TTFT 改善达 72.4%**
  - **TPOT 改善达 26.7%**

> 结论：**KV Patching 将迁移从“阻塞操作”转变为“后台渐进过程”，极大降低服务质量下降风险**。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **动态 PP 重配置可行且必要**：
   - 异构 GPU 上，不同工作负载的最佳 PP 配置差异显著（如 prefill-heavy 偏好 L40S，decode-heavy 偏好 A100）。
   - 静态配置最多只能满足一种模式，性能损失可达 20–30%。

2. **KV Cache 必须支持动态 resize**：
   - 固定分配无法应对配置变更带来的容量需求波动。
   - PIPELIVE 的 block-level allocation + layer stacking 实现了高效 resize，**提升请求承载能力达 2.5×**。

3. **KV Patching 是低干扰的关键**：
   - 类似 VM live migration 的思想成功迁移到 LLM 推理领域。
   - 实现了 **<10ms 的服务中断时间**，几乎不影响用户体验。

4. **整体性能显著优于所有静态方案**：
   - 综合性能得分提升 **33–36%**
   - 在极端负载下依然稳健，展现出强大的弹性服务能力。

---

### ⚠️ **方法的局限性**
1. **未解决“何时触发重配置”问题**：
   - 当前假设已知源和目标配置，**决策逻辑留待未来工作**。
2. **依赖较强的网络带宽**：
   - KV 同步依赖高速互联（如 InfiniBand），在弱网环境下效果可能下降。
3. **layer stacking 限制了重配置粒度**：
   - stacking factor 越大，越难进行细粒度调整。
4. **目前仅支持 PP，未整合 TP 或 DP**：
   - 多维并行联合优化尚未探索。

---

### 🔮 **未来工作方向**
1. **构建智能重配置调度器**：
   - 基于实时负载特征（prefill/decode 比例、请求率等）自动选择最优 PP 配置。
2. **联合优化多种并行范式**：
   - 探索 PP + TP + DP 的动态协同重配置。
3. **扩展至多租户场景**：
   - 支持 tenant-aware 的资源隔离与动态调配。
4. **降低对高性能网络的依赖**：
   - 设计压缩或差分同步机制，适应普通数据中心环境。

---

> **总结一句话**：  
> **PIPELIVE 首次实现了真正意义上的“热插拔”式 PP 重配置，使 LLM 服务能够像操作系统进程一样，在运行时无缝切换执行布局，为构建自适应、高弹性的 LLM 推理引擎奠定了关键技术基础。**

</details>

---

### 2. [Three Birds, One Stone: Solving the Communication-Memory-Privacy Trilemma in LLM Fine-tuning Over Wireless Networks with Zeroth-Order Optimization](https://arxiv.org/abs/2604.12401)

**Authors**: Zhijie Cai, Yuhao Zheng, Haolong Chen, Dongzhu Liu, Bin Wang, Guangxu Zhu  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.12401v1  

#### Abstract
Federated Learning (FL) offers a promising pathway for collaboratively fine-tuning Large Language Models (LLMs) at the edge; however, this paradigm faces a critical bottleneck: the prohibitive communication and memory overheads incurred by exchanging high-dimensional gradients. Furthermore, recent s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Three Birds, One Stone: Solving the Communication-Memory-Privacy Trilemma in LLM Fine-tuning Over Wireless Networks with Zeroth-Order Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对在无线边缘网络中对 **Large Language Models (LLMs)** 进行 **Federated Learning (FL)** 微调时面临的“**通信-内存-隐私三难困境**”（communication-memory-privacy trilemma）提出解决方案：

- **Communication Bottleneck**：传统 FL 中传输高维梯度导致巨大的通信开销，难以适应带宽受限的无线边缘设备。
- **Memory Wall**：标准反向传播（backpropagation）需要存储中间激活值，超出边缘设备的内存容量。
- **Privacy Leakage**：即使不共享原始数据，攻击者仍可通过本地梯度重构用户敏感信息（如 gradient inversion attacks）。

### 提出的新方法：pAirZero
作者提出了 **pAirZero**，一种将 **Zeroth-Order (ZO) optimization** 与 **Over-the-Air (OTA) computation** 相结合的新型联邦微调框架，并进一步扩展为数字调制版本 **Sign-pAirZero**。

#### 核心创新点：
- **三重协同优化**：
  - 利用 ZO 估计梯度，避免 backpropagation，实现 **inference-level memory cost**。
  - 利用 OTA 的电磁波叠加特性，在物理层直接聚合信号，消除逐个解码带来的通信瓶颈。
  - 将无线信道噪声与人工注入噪声结合，作为实现 **differential privacy (DP)** 的天然机制，实现 **privacy-by-design** 架构。

- **极致资源效率**：
  - 每轮迭代仅需上传 **bit-level communication load**（Sign-pAirZero 为 1 bit，pAirZero 为 16 bits），与模型维度无关（O(1)）。
  - 内存消耗降低至推理级别，相比传统方法减少约 **75%**。

- **自适应功率控制与隐私保障**：
  - 设计了基于优化的功率分配策略（见 Theorem 3 和 Theorem 4），动态调整发射功率和人工噪声强度，在保证收敛性的同时满足 $(\epsilon, \delta)$-DP 要求。
  - 隐私保护不依赖于信道条件，确保一致性保护水平。

### 相比现有方法的优势
| 维度 | 传统方法 | pAirZero |
|------|--------|---------|
| **通信开销** | 与模型参数量成正比（如 238.88 MB for OPT-125M） | 固定低开销（1~16 bits） |
| **内存需求** | 需存储激活图，高达数百 MB（如 Adam 达 955.58 MB） | 推理级内存（~250 MB） |
| **隐私机制** | 后置添加 DP 噪声，增加计算负担且可能破坏性能 | 嵌入式设计，“in the air”完成隐私保护 |
| **同步要求** | OTA 方法通常要求严格时间同步 | ZO+OTA 减轻了同步压力 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **OPT-125M**：一个轻量级预训练语言模型，用于验证方法可行性。
- 下游任务：
  - **SST-2**：二分类情感分析任务（句子级）
  - **SQuAD v1.1**：问答任务，评估上下文理解和答案抽取能力

### 实验设置
- 客户端数量 $K = 5$
- 每轮使用 mini-batch 数据进行本地更新
- 总训练轮数 $T = 8000$
- 扰动尺度 $\mu = 0.001$
- DP 参数：$\epsilon = 5$, $\delta = 0.01$
- 学习率通过网格搜索确定（见 Table I）
- 重复 4 次独立运行取均值 ± 标准差

### 评估指标
- **主任务性能**：
  - SST-2：Accuracy
  - SQuAD：F1 Score
- **系统效率**：
  - 每轮上传数据量（bits）
  - 最小内存占用（MB）
- **消融研究**：
  - 不同功率分配策略的影响
  - 模拟 vs 数字调制性能差异
  - 参数敏感性分析（如 $e_o$, $\gamma$）

### 基线方法对比
- **Perfect**：理想无噪声聚合，作为性能上限
- **Static**：固定功率分配，非自适应
- **Reversed**：反转优化趋势的对照组
- **FO SGD / Adam**：传统一阶优化器作为资源消耗基准

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table II & Fig. 2）
| 方法 | 内存成本 | 每轮上传量 |
|------|--------|----------|
| FO SGD | ~600 MB | 238.88 MB |
| FO Adam | 955.58 MB | 238.88 MB |
| **pAirZero** | **~250 MB** | **16 bits** |
| **Sign-pAirZero** | **~250 MB** | **1 bit** |

> ✅ **内存节省约 75%，通信开销降低多个数量级**

### 与基线方法的对比结果（Fig. 2）
- 在 SST-2 和 SQuAD 上，**pAirZero 和 Sign-pAirZero 的最终性能接近 “Perfect” 理想情况**，显著优于传统方法在相同隐私约束下的表现。
- **模拟调制 (pAirZero)**：
  - 在高 SNR 下性能更优，尤其在复杂任务（如 SQuAD）上领先超过一个标准差。
  - 但在低 SNR 区域性能下降明显，受信道噪声影响较大。
- **数字调制 (Sign-pAirZero)**：
  - 表现更加稳定，跨 SNR 波动小，适合实际部署。
  - 因一比特压缩引入额外噪声，极限性能略低于模拟版本。

### 消融实验结果（Fig. 3）
- **Solution-based power allocation** 明显优于其他策略：
  - “Static” 方案因无法适应梯度变化而导致模型性能受损。
  - “Reversed” 方案验证了优化趋势的有效性。
- 自适应功率控制对于 ZO 类方法至关重要，尤其是在大规模聚合场景下。

### 参数研究（Section VII-D）
- **Sign Reversing Probability $e_o$**：实测最大值为 0.4968 < 0.5，支持理论假设。
- **Contraction Ratio $A$**：经验估计为 0.998，用于指导功率控制设计。
- **Clip Threshold $\gamma$**：设为 100 可覆盖 97% 以上的梯度投影值，平衡 clipping 频率与隐私预算。

---

## 4. 关键结论和发现

### 主要发现
1. **ZO + OTA 是解决 LLM 边缘微调三难问题的理想组合**：
   - ZO 消除 backpropagation，打破 memory wall；
   - OTA 利用物理层叠加，缓解 communication bottleneck；
   - 二者结合使 DP 可自然嵌入传输过程，实现 privacy-by-design。

2. **bit-level communication 与 inference-level memory 成为现实**：
   - 即使是百亿参数级别的 LLM，也可在资源极度受限的设备上参与联邦学习。

3. **自适应功率控制对性能至关重要**：
   - 动态调节 $c^{(t)}$ 和 $n^{(t)}$ 可最小化优化间隙，同时满足 DP 约束。

4. **数字调制更具鲁棒性，模拟调制潜力更高**：
   - Sign-pAirZero 更适合实际部署；pAirZero 在理想条件下具备更高上限。

### 方法的局限性
- 当前实验基于 **OPT-125M**，尚未验证在更大规模模型（如 OPT-1.3B 或 LLaMA 系列）上的可扩展性。
- 假设下行广播无噪声，实际中可能需考虑下行链路可靠性。
- 对随机方向 $z^{(t)}$ 的生成依赖 PRNG 同步，存在潜在实现复杂性。
- 数字调制引入的 sign reversal 噪声限制了其理论收敛速度。

### 未来工作方向
- 扩展到 **multi-modal models** 和 **larger-scale LLMs**。
- 探索 **asynchronous pAirZero** 以进一步降低同步要求。
- 结合 **PEFT 方法**（如 LoRA）与 pAirZero，实现双重压缩。
- 研究 **multi-cell OTA FL** 场景下的干扰管理与资源调度。

--- 

> 📌 **一句话总结**：  
> pAirZero 通过将 **Zeroth-Order Optimization** 与 **Over-the-Air Computation** 深度融合，首次实现了在无线边缘网络中对 LLM 进行高效、低内存、强隐私保护的联邦微调，真正做到了“一石三鸟”。

</details>

---

### 3. [Accelerating Microswimmer Simulations via a Heterogeneous Pipelined Parallel-in-Time Framework](https://arxiv.org/abs/2604.12083)

**Authors**: Ruixiang Huang, Weifan Liu  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.12083v1  

#### Abstract
Simulating large-scale microswimmer dynamics in viscous fluid poses significant challenges due to the coupled high spatial and temporal complexity. Conventional high-performance computing (HPC) methods often address these two dimensions in isolation, leaving a critical gap for synergistic accelerati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Accelerating Microswimmer Simulations via a Heterogeneous Pipelined Parallel-in-Time Framework

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

微泳者（microswimmer）在粘性流体中的大规模、长时间动力学模拟面临极高的计算复杂度，主要体现在两个方面：

- **空间复杂度**：基于拉格朗日框架（如 Method of Regularized Stokeslets, MRS）的流固耦合（FSI）模型需要计算所有离散点之间的相互作用，导致 $O(N^2)$ 的计算开销。
- **时间复杂度**：系统刚性强，显式积分器需极小时间步长以维持稳定性，导致长期模拟需数百万次迭代，严重受限于串行时间推进。

传统高性能计算（HPC）方法通常仅单独优化空间并行或时间并行，缺乏对时空协同加速的支持。

---

### **提出了什么新方法或新思路**

本文提出了一种**异构 CPU-GPU 流水线并行时域框架**（heterogeneous pipelined parallel-in-time framework），实现以下创新：

#### ✅ **双层级并行化策略**
1. **空间并行**：利用 GPU 高吞吐能力，设计高密度并行核函数，加速 MRS 中的 $O(N^2)$ 线性和角速度计算。
2. **时间并行**：采用 **pipelined Parareal 算法**，将时间域划分为多个子区间，在多 GPU 上实现粗粒度（coarse）与细粒度（fine）求解器的流水线重叠执行。

#### ✅ **流水线 Parareal 架构（Pipelined Parareal）**
- 改进标准 Parareal 的串行等待瓶颈：不再等待整个粗求解器完成后再启动细求解器，而是**一旦某段粗解完成，立即启动对应时间段的细求解**。
- 利用异步调度机制，使多个 GPU 并发运行不同阶段的求解任务，显著减少 GPU 空闲时间。

#### ✅ **GPU 优化的关键数值例程**
- 针对 Kirchhoff 杆模型中维持正交标架所需的 **matrix square root** 运算，提出一种基于 Rodrigues 公式的闭式解析解法，并针对 GPU 的 SIMT 架构进行优化，避免通用算法（如 `scipy.linalg.sqrtm`）在 GPU 上效率低下。

---

### **相比现有方法的优势**

| 维度 | 本方法优势 |
|------|-----------|
| **架构整合** | 同时融合空间（GPU）与时间（MPI + Parareal）并行，实现真正的时空协同加速 |
| **资源利用率** | 流水线设计有效重叠计算与通信，提升 GPU 利用率，降低空闲时间 |
| **扩展性** | 支持多 GPU 分布式部署，具备良好的弱缩放（weak scaling）和强缩放（strong scaling）性能 |
| **精度与效率平衡** | 在保持高精度（$10^{-12}$ 级误差）的同时，实现数量级加速 |

---

## 2. 核心实验方法和设置

### **使用的物理模型与仿真对象**

- **模型**：基于 **Kirchhoff rod model** 的细丝状微泳者（如精子、细菌鞭毛）
- **流体方程**：不可压缩 Stokes 方程，采用 **Method of Regularized Stokeslets (MRS)** 处理奇异性
- **边界条件**：半无限流体域，底部为无滑移平面壁面（no-slip wall）
- **运动驱动**：施加正弦波形的预设曲率（preferred strain-twist vector）

### **实验设置**

| 参数 | 设置 |
|------|------|
| 单根杆离散点数 $M$ | 51 |
| 正则化参数 $\epsilon$ | $4\Delta s$ |
| 时间步长（fine solver） | $10^{-6}$ |
| 总模拟时间 $T$ | 最长达 4 个单位时间 |
| 微泳者数量 | 1, 4, 12, 25 根杆 |
| GPU 平台 | NVIDIA A100 PCIe 40GB |
| CPU 平台 | Kunpeng-920 / AMD 7H12（用于 CPU-only 对比） |
| 实现语言 | Python + `numba.cuda`（GPU 核函数） |

### **评估指标**

| 指标 | 定义 |
|------|------|
| **Speedup** | $S = T_{\text{CPU}} / T_{\text{GPU}}$ |
| **Relative Error** | $\eta_k = \max_i \|x_i^k - x_i^{\text{true}}\| / \|x_i^{\text{true}}\|$ |
| **Relative Increment** | $\delta_k = \max_i \|x_i^k - x_i^{k-1}\| / \|x_i^{k-1}\|$（用于收敛判断） |
| **Weak Scaling** | 固定每 GPU 负载，增加总规模与 GPU 数量 |
| **Strong Scaling** | 固定问题规模，增加 GPU 数量 |
| **Idle Time Analysis** | 分析流水线 vs 标准 Parareal 的 GPU 等待时间差异 |

### **基线方法对比**

| 基线方法 | 描述 |
|--------|------|
| **CPU-only 串行求解** | 使用 CPU 实现完整时间序列积分，作为性能下限 |
| **标准 Parareal（非流水线）** | 经典 Parareal 方法，粗求解必须完全串行完成后才启动细求解 |
| **纯 GPU 空间并行** | 仅使用 GPU 加速空间计算，时间仍串行推进 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **空间并行加速效果（Table I）**

| 微泳者数 | 总体加速比（vs CPU） |
|---------|------------------|
| 1       | 23.8×            |
| 4       | 110.6×           |
| 12      | 399.5×           |
| 25      | **769.3×**       |

> - GPU 运行时间几乎不随杆数增加而上升（直到 25 根才略有增长），表明高度并行化吸收了负载。
> - 速度计算部分最高达 **2300×** 加速。

#### ✅ **矩阵平方根性能（Table II）**

| 指标 | 结果 |
|------|------|
| 平均加速比（vs `scipy.linalg.sqrtm`） | **2.14×** |
| 数值误差（$\|S^2 - R\|_F$） | 与 SciPy 相当甚至更优 |
| 不稳定区域 | $\theta \sim \pi$ 附近有波动，但不影响整体收敛 |

> 表明自定义 CUDA 实现不仅更快，且数值稳定。

#### ✅ **时间并行性能（Table III & Fig. 7）**

| 设置 | 流水线 vs 标准 Parareal 加速 |
|------|----------------------------|
| 2 GPUs, $r=2$ | 减少约 **25–30%** 运行时间 |
| 4 GPUs, $r=4$ | 减少约 **32%** |
| 随 $r$ 增大，优势减小；随 GPU 数增多，优势增强 |

> 实验验证了理论分析：当粗求解较慢（$r$ 小）或多 GPU 时，流水线减少的同步等待更显著。

#### ✅ **弱缩放性能（Table IV & Fig. 8a）**

| 时间长度 $T$ | GPU 数 | 流水线总时间（s） |
|---------------|--------|------------------|
| 0.5           | 1      | 4283             |
| 4             | 8      | 9232             |

> 问题规模扩大 8 倍，运行时间仅增加约 2.15 倍 → **良好弱缩放**

#### ✅ **强缩放性能（Table V & Fig. 8b）**

| GPU 数 | 总时间（s） | 加速比 | 并行效率 |
|-------|------------|--------|----------|
| 1     | 9597.82    | 1.00   | 100%     |
| 2     | 4834.20    | 1.99   | 99.5%    |
| 4     | 2492.66    | 3.85   | 96.3%    |
| 8     | 1555.91    | 6.17   | **77.1%** |

> 接近线性加速，8 GPU 下仍保持 77% 效率，优于多数 PinT 方法。

---

## 4. 关键结论和发现

### **主要发现**

1. **流水线 Parareal 显著降低 GPU 空闲时间**  
   理论与实验一致表明，流水线调度通过重叠粗/细求解，大幅减少同步等待，尤其在 $r$ 较小或 GPU 数较多时优势明显。

2. **GPU 主导的异构架构可实现数量级加速**  
   结合空间并行（MRS 核函数）与时间并行（pipelined Parareal），整体性能比 CPU-only 方法快 **数百倍至近 800×**。

3. **算法具备良好收敛性**  
   Parareal 在约 4 次迭代内即可达到 $10^{-12}$ 级精度，适用于高保真生物流体模拟。

4. **内存效率高**  
   25 根杆仅占用约 2GB GPU 内存，适合大规模部署。

---

### **方法的局限性**

1. **资源利用率仍有提升空间**  
   在大规模、长时间模拟中，部分 GPU 可能因任务依赖或负载不均而闲置。

2. **Python 实现限制峰值性能**  
   当前基于 `numba.cuda` 的实现未达底层语言（如 CUDA C++）的极限性能。

3. **参数敏感性**  
   参数 $r$（粗/细求解成本比）影响收敛性与效率：过大可能导致粗解不准、收敛变慢；过小则削弱加速效果。

4. **通信开销随节点增加上升**  
   多节点场景下，跨节点通信与同步成为瓶颈，限制进一步扩展。

---

### **未来工作方向**

1. **优化 GPU 调度策略**  
   设计动态负载均衡与任务映射机制，进一步减少空闲时间。

2. **迁移至低级语言**  
   使用 CUDA C/C++ 或 HIP 重构核心模块，挖掘更高性能潜力。

3. **增强对刚性问题的鲁棒性**  
   引入隐式或 IMEX 时间积分器，提升对高刚性系统的适应能力。

4. **集成更多 PinT 方法**  
   探索 PFASST 或 MGRIT 等多层级 PinT 方法与异构架构的结合。

5. **应用于真实生物集体行为模拟**  
   扩展至数千微泳者的群体自组织、涡旋形成等复杂涌现现象研究。

---

> **总结**：本文成功构建了一个高效、可扩展的异构时空并行框架，首次将 **pipelined Parareal** 与 **GPU 加速 MRS** 深度融合，为大规模生物流体模拟提供了强有力的工具，推动了微尺度流体力学仿真的前沿发展。

</details>

---

### 4. [Evolution of Optimization Methods: Algorithms, Scenarios, and Evaluations](https://arxiv.org/abs/2604.12968)

**Authors**: Tong Zhang, Jiangning Zhang, Zhucun Xue, Juntao Jiang, Yicheng Xu, Chengming Xu, Teng Hu, Xingyu Xie, Xiaobin Hu, Yabiao Wang, Yong Liu, Shuicheng Yan  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.12968v1  

#### Abstract
Balancing convergence speed, generalization capability, and computational efficiency remains a core challenge in deep learning optimization. First-order gradient descent methods, epitomized by stochastic gradient descent (SGD) and Adam, serve as the cornerstone of modern training pipelines. However,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Evolution of Optimization Methods: Algorithms, Scenarios, and Evaluations》核心总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文系统地解决了当前深度学习优化领域存在的三大核心挑战：
1.  **缺乏统一框架**：现有综述通常局限于一阶（FO）和二阶（SO）方法，忽略了快速发展的零阶（ZO）优化和面向特定场景（如分布式、隐私保护）的范式。
2.  **分类不严谨**：现有研究对算法的分类过于粗糙（如笼统的“自适应方法”），未能揭示不同方法间的内在联系和演化逻辑。
3.  **缺乏标准化基准**：缺少一个公平、大规模的实证基准来评估现代优化器在不同架构上的表现，导致文献中的结论常有冲突且难以复现。

### 提出的新方法或新思路
论文提出了一个全面的三支柱框架，旨在为优化领域提供一个统一的视角：

1.  **统一的数学分类法 (Unified Taxonomy)**：
    *   提出了一个基于梯度信息使用的严格数学分类体系，将优化方法分为四大类：**一阶 (First-Order, FO)**、**二阶 (Second-Order, SO)** 和 **零阶 (Zeroth-Order, ZO)**，以及一个覆盖性的 **面向场景的范式 (Scenario-Oriented Paradigms)**。
    *   引入了一个通用的数学公式 `g=8(f,0,S), 9=Tscenario(g), m=0(m-1,9), 0t+1=Pe(0-mM.-1m-m入0)` 来解耦优化过程，从四个维度（梯度估计器、预处理器、场景感知变换、结构投影）分析算法的演变。

2.  **面向场景的分析 (Scenario-Oriented Analysis)**：
    *   论文强调，现代优化器的演进本质上是从纯算法设计向**系统感知的工程解决方案**的转变。这些方法被重新架构以应对物理瓶颈，如分布式通信墙、严格的差分隐私（DP）约束和内存墙。
    *   这种视角突显了优化器如何通过权衡理论保证与实际约束（如计算成本、噪声鲁棒性）来实现平衡。

3.  **标准化的评估框架 (Standardized Evaluation)**：
    *   构建了一个严谨的测试平台，在多种架构代理（CNN和Transformer）上评估了23种主流优化器。
    *   该框架专注于分离算法性能与大规模工程优化，系统地考察了**学习率敏感性**、**长期训练可扩展性**和**跨架构泛化能力**等关键因素。

### 相比现有方法的优势
*   **更全面**：首次将FO、SO、ZO和场景导向框架整合到一个统一的分析框架中。
*   **更严谨**：提供了清晰的数学基础和分类标准，揭示了不同方法间的内在联系和演化路径。
*   **更实用**：通过大规模、标准化的实证评估，为研究人员和工程师选择优化器提供了可靠、可复现的指导，填补了理论与实践之间的鸿沟。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验涵盖了计算机视觉和自然语言处理两大领域的代表性任务：
*   **视觉任务**：使用 **ImageNet-1K** 数据集，模型为 **ResNet-50** 和 **ViT-Small (ViT-S)**。
*   **语言任务**：在 **WikiText-103** 数据集上从头开始训练一个 **60M参数的Llama** 模型。作者明确指出，这个小型Llama模型是作为现代自回归Transformer的**架构代理 (architectural proxy)**，用于研究优化器在因果注意力机制下的表现，而非复制大模型的涌现行为。

### 实验设置和评估指标
*   **超参数设置**：为了公平比较，所有优化器仅调整一个共同的超参数——**学习率 (learning rate)**。采用网格搜索策略，探索原始文献建议值的0.1x, 0.2x, 1.0x, 5.0x, 和10.0x倍数。
*   **评估指标**：
    *   **视觉任务**：使用 **Top-1 Accuracy** 作为主要指标。
    *   **语言任务**：使用 **困惑度 (Perplexity, PPL)** 作为主要指标。
*   **训练设置**：详细控制了训练配置，包括批量大小、学习率调度器（余弦退火）、权重衰减、梯度裁剪阈值等，确保评估聚焦于优化器本身的性能差异。

### 基线方法对比
论文对23种优化器进行了全面评估，涵盖了广泛的基线方法，包括但不限于：
*   **经典方法**：`SGD`, `RMSprop`, `Adam`, `AdamW`。
*   **先进一阶方法**：`Lion`, `Muon`, `MARS`, `Adan`, `LAMB`, `RAdam`。
*   **二阶方法**：`Kron`, `Sophia`。
*   **零阶方法**：`MeZO`。
*   **其他**：`Adafactor`, `Nadam`, `NovoGrad` 等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **跨架构泛化能力 (Cross-Architecture Generalization)**：
    *   **`Muon`** 和 **`MARS`** 在跨架构迁移时表现出色。当将在ViT-S上找到的最佳学习率直接应用于ResNet-50和Llama模型时，它们依然能保持稳定且优异的性能。
    *   `Muon` 的优势在于其**矩阵级正交化 (matrix-level orthogonalization)**，这使其更新不受各层梯度尺度差异的影响，从而在高度各向异性的大语言模型损失景观中也能稳健运行。
    *   `MARS` 的优势在于其**梯度校正 (gradient correction)** 机制，通过减去前一批次的梯度来抵消随机噪声，从而稳定了不同网络架构下的优化轨迹。

2.  **学习率敏感性 (Learning Rate Sensitivity)**：
    *   **`Muon`** 展现出最强的**超参数鲁棒性**。在ViT-S上，即使学习率缩放至0.2x到5.0x，其准确率仍能保持在75%以上，表明其极易调优。
    *   **`Lion`**, **`MADGRAD`**, **`MARS`** 表现出高下界稳定性。例如，`Lion` 只使用动量加权梯度的符号进行更新，这使得其步长与梯度幅值无关，从而避免了因学习率过大而导致的数值爆炸。
    *   相比之下，`SGD` 虽然在大语言模型上会崩溃，但在小学习率下对大倍数缩放具有意外的鲁棒性，因为它没有除以微小的二阶矩项。

3.  **长期训练可扩展性 (Long-term Training Scalability)**：
    *   `SGD` 系列优化器在长期训练（300轮）中表现出更强的可扩展性，因为它们不会像一些自适应方法那样过早地衰减步长，从而能在后期继续探索损失景观。
    *   `Muon` 等先进优化器虽然收敛速度快，但其在100轮后性能提升已趋于平缓，说明它们早期就已接近高质量的最优解。

4.  **大语言模型上的灾难性失败**：
    *   论文的关键发现之一是，**`SGD`及其变体在Llama架构上会发生灾难性训练崩溃**（如图9所示）。这是因为缺乏自适应的逐层或逐参数缩放机制，导致均匀的学习率在某些层引发梯度爆炸，而在另一些层则导致梯度消失，最终完全丧失模型的表征能力。

### 消融实验结果
论文通过相关性分析（图11）揭示了算法的内在行为：
*   在ViT-S上，由于其高度受限的训练流程（如长时间线性学习率预热），所有优化器的行为都趋于同质化，相关性极高。
*   在ResNet-50上，不同的优化器家族展现出明显的分化。例如，`Adam` 家族形成一个紧密相关的簇，而 `Kron` 和 `Muon` 因其结构性预处理和正交化技术，与前者有显著区别。

---

## 4. 关键结论和发现

### 主要发现
1.  **优化范式的根本转变**：现代优化器的演进不再是孤立的算法改进，而是针对**物理可行性危机**（内存墙、通信墙、隐私墙）的系统性工程响应。优化器已成为一种**系统感知的工程解决方案**。
2.  **`SGD`在大模型上的失效**：传统的 `SGD` 无法满足大语言模型的需求，因其在高度各向异性的损失景观中会不可避免地发生训练崩溃，这凸显了自适应缩放的绝对必要性。
3.  **鲁棒性与泛化的关键**：在 `Muon` 和 `MARS` 等新兴方法中，**矩阵级正交化**和**梯度校正**等机制是实现强跨架构泛化和超参数鲁棒性的关键。
4.  **评估的复杂性**：优化器的性能高度依赖于模型架构和训练方案。一个在ViT上表现优异的优化器可能在Llama上彻底失败，反之亦然。

### 方法的局限性
*   **计算成本高昂**：本文提出的标准化评估框架需要巨大的计算资源（文中估计超过1073个单卡A100小时），这对于大多数研究者来说是不可及的。
*   **代理模型的限制**：实验中使用的60M参数Llama是一个简化代理，其行为可能无法完全代表千亿参数级别的真实大语言模型。
*   **动态环境的缺失**：实验设置相对静态，未充分模拟现实世界中可能出现的非平稳目标函数或剧烈变化的硬件条件。

### 未来工作方向
1.  **自动化优化器设计**：发展能够自动为特定架构生成优化器的框架，减少对人工启发式调参的依赖。
2.  **硬件-算法协同设计**：推动优化器与底层硬件（如TPU、专用AI芯片）的协同设计，以实现全局效率的最大化。
3.  **精确的噪声消除**：开发能够精确抵消由随机扰动引入的固有噪声的框架，尤其是在零阶优化中。
4.  **理论与实践的桥梁**：建立一个统一的理论框架，能够同时保证在分布式、隐私保护和大规模场景下的通信效率、隐私效用和收敛性。

</details>

---

### 5. [TCL: Enabling Fast and Efficient Cross-Hardware Tensor Program Optimization via Continual Learning](https://arxiv.org/abs/2604.12891)

**Authors**: Chaoyao Shen, Linfeng Jiang, Yixian Shen, Tao Xu, Guoqing Li, Anuj Pathania, Andy D. Pimentel, Meng Zhang  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.12891v1  

#### Abstract
Deep learning (DL) compilers rely on cost models and auto-tuning to optimize tensor programs for target hardware. However, existing approaches depend on large offline datasets, incurring high collection costs and offering suboptimal transferability across platforms. In this paper, we introduce TCL, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：TCL: Enabling Fast and Efficient Cross-Hardware Tensor Program Optimization via Continual Learning

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前基于 **offline-trained cost model** 的深度学习（DL）编译器在跨硬件平台迁移时面临三大挑战：

1. **高数据收集成本**：为每个新硬件平台收集大规模 tensor program 性能数据耗时极长（如 CPU 需 40 天，GPU 超 60 天），严重制约可扩展性。
2. **cost model 架构效率低下**：Transformer 等模型虽能捕捉长程依赖，但具有 $O(n^2)$ 的计算和内存复杂度；LSTM 则并行性差、收敛慢。
3. **跨平台知识迁移能力不足**：现有方法多为单源到单目标的一对一迁移，而多任务学习（multi-task learning）存在参数爆炸和需同时访问所有平台数据的问题。

---

### **提出了什么新方法或新思路**

本文提出 **TCL**（Tensor Compiler via Continual Learning），一个高效、可转移的编译器框架，包含三个核心组件：

#### ✅ **(1) RDU Sampler**（Data-Efficient Active Learning）
- 一种主动学习采样策略，联合优化 **Representativeness（代表性）**、**Diversity（多样性）** 和 **Uncertainty（不确定性）**。
- 仅需采集 **10% 的数据** 即可达到接近全量数据训练的模型精度，大幅降低数据收集开销。

#### ✅ **(2) Mamba-based Cost Model**
- 采用 **Mamba Block** 替代传统的 Transformer 或 LSTM 架构。
- 利用 **Structured State Space Model (SSM)** 实现 $O(n)$ 时间复杂度，有效建模 schedule primitive 序列中的长程依赖。
- 参数更少、训练更快，且预测准确率更高。

#### ✅ **(3) Continuous Knowledge Distillation (CKD) Framework**
- 提出持续知识蒸馏机制，支持从多个源硬件平台渐进式积累知识。
- 包含两个模块：
  - **Knowledge Base (KB)**：共享的硬件知识库，冻结以防止遗忘旧知识。
  - **Hardware Active Column (AC)**：针对新硬件的学习模块，通过连接复用 KB 中的知识。
- 使用 **Elastic Weight Consolidation (EWC)** 抑制灾难性遗忘，实现无参数膨胀的跨平台知识迁移。

---

### **相比现有方法的优势**

| 方面 | TCL 优势 |
|------|---------|
| **数据效率** | 仅需 10% 数据即可媲美全量训练，远优于随机采样或传统 active learning 方法（如 ALT、BALTO）。 |
| **模型效率** | Mamba 架构比 Transformer 更轻量，推理速度快，内存占用低，适合部署于资源受限环境。 |
| **跨平台泛化能力** | 支持多源平台知识渐进融合，避免 multi-task learning 的参数爆炸问题，更具可扩展性。 |
| **实用性** | 不要求所有源平台数据同时可用，更适合实际场景下的增量更新。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- 主要基于开源的大规模数据集 **Tenset**，覆盖 6 种硬件平台（4 CPU + 2 GPU）。
- 补充自建数据：在 **Intel i7-12700F CPU** 和 **NVIDIA GeForce RTX 3080Ti GPU** 上采集的新数据。
- 每个平台约有 8.4M~8.6M 条样本，涵盖多种主流 DNN 模型（如 ResNet、MobileNet、BERT 等）。

---

### **实验设置和评估指标**

#### 📌 **评估方式分为两类**：

| 类型 | 指标说明 |
|------|----------|
| **Dataset-based Evaluation** | 使用 **Top-k Score** 评估 cost model 对 tensor program 排序的准确性：<br>$$ \text{Top-k} = \frac{\sum_{m,s} \min\_\text{latency}_{m,s} \cdot \text{weight}_{m,s}}{\sum_{m,s} \sum_{i=1}^{k} p\_\text{latency}_{m,s,i} \cdot \text{weight}_{m,s}} $$<br>值越接近 1 表示排序越准。 |
| **End-to-end Evaluation** | 在真实编译流程中测试性能：<br>- **Tuning Time**：达到相同推理延迟所需迭代次数。<br>- **Inference Latency**：固定调优轮次后的最优延迟表现。 |

#### 🧪 **训练配置**
- 使用 **TVM (v0.8.0)** 集成 TCL。
- Cost model 训练：Adam 优化器，初始学习率 0.0007，batch size 1024，训练 100 轮。
- RDU Sampler 设置采样率为 10%。

---

### **基线方法对比**

| 基线方法 | 类型 | 说明 |
|--------|------|------|
| **Tenset-MLP** | Fully-supervised | 使用 MLP 作为 cost model，代表早期监督学习方法。 |
| **MTL-TLP** | Multi-task Learning | 当前 SOTA 的多任务学习方法，联合学习多个平台。 |
| **Ansor** | Zero-shot | 无预训练 cost model，在线搜索，代表传统 auto-tuning 方法。 |
| **Felix** | Few-shot | 基于梯度下降的小样本调优器，仅适用于 GPU。 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔢 **端到端调优效率提升（vs. Tenset-MLP）**

| 平台 | 调优时间加速比 | 推理延迟降低 |
|------|----------------|-------------|
| **CPU (i7-12700F)** | **16.8×** 更快 | **1.20×** 更低延迟 |
| **GPU (RTX 3080Ti)** | **12.48×** 更快 | **1.13×** 更低延迟 |

> 即 TCL 只需 Tenset-MLP **1/16.8 的时间** 就能达到其最终优化水平，并进一步将推理延迟压缩至原来的 **83% 左右**。

#### 🔢 **与零样本/小样本方法对比（vs. Ansor / Felix）**

| 对比项 | 加速比 |
|-------|--------|
| vs. Ansor（CPU） | **19.98×** 调优加速 |
| vs. Ansor（GPU） | **34.78×** 调优加速 |
| vs. Felix（GPU） | **26.23×** 调优加速 |

> 显示 TCL 的离线 cost model 显著优于在线学习策略。

---

### **消融实验结果（Ablation Study）**

在 **i7-12700F CPU** 和 **RTX 3080Ti GPU** 上进行模块移除实验（Table 7）：

| 模块组合 | CPU Top-1 | GPU Top-1 |
|--------|-----------|----------|
| 无任何增强（Baseline） | 0.7414 | 0.6614 |
| + RDU Sampler | 0.9020 | 0.8349 |
| + Mamba Model | 0.9116 | 0.8533 |
| + CKD Framework | **0.9319** | **0.8675** |

✅ 结果表明：**三个模块均显著提升性能**，协同作用下达到最佳效果。

---

## 4. 关键结论和发现

### **主要发现**

1. **RDU Sampler 极大提升了数据利用效率**：  
   仅用 10% 数据即可实现接近甚至超越全量训练的效果，验证了主动学习在 DL 编译器中的巨大潜力。

2. **Mamba 架构优于传统序列模型**：  
   在保持 $O(n)$ 复杂度的同时，准确捕捉 schedule primitive 序列的长期依赖关系，尤其适合长序列建模任务。

3. **CKD 实现了高效的跨平台知识迁移**：  
   成功避免 multi-task learning 的“参数爆炸”问题，支持知识的渐进式积累，是迈向通用编译器的重要一步。

4. **同厂商硬件间知识迁移更有效**：  
   实验发现 Intel CPU 之间、NVIDIA GPU 之间的知识正向促进明显，而跨架构（如 AMD → Intel）可能引入干扰。

---

### **方法的局限性**

1. **未在 CPU 和 GPU 之间进行知识迁移**：  
   论文明确指出不尝试跨计算范式迁移（如 CPU ↔ GPU），限制了通用性。
   
2. **对有害知识过滤仍不彻底**：  
   尽管使用 EWC 和门控蒸馏机制，部分源平台的“负迁移”现象依然存在。

3. **依赖高质量 schedule primitive 表示**：  
   性能受限于特征工程的质量，尚未完全实现端到端表示学习。

---

### **未来工作方向**

1. 设计更细粒度的 **cost model** 以进一步提升预测精度。
2. 探索 **跨架构（CPU-GPU）的知识迁移机制**。
3. 引入 **自动化特征提取或图神经网络** 减少人工特征设计依赖。
4. 扩展至更多硬件类型（如 TPU、NPU、FPGA）构建真正的通用编译框架。

--- 

> ✅ **总结一句话**：  
> TCL 通过 **RDU Sampler + Mamba Cost Model + CKD Framework** 三位一体的设计，实现了**高速、高效、可扩展**的跨硬件 tensor program 优化，在数据效率、调优速度和推理性能上全面超越现有方法，为下一代 DL 编译器提供了新范式。

</details>

---

### 6. [BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining](https://arxiv.org/abs/2604.12241)

**Authors**: Haojie Ye, Arjun Laxman, Yichao Yuan, Krisztian Flautner, Nishil Talati  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.12241v1  

#### Abstract
Money laundering detection faces challenges due to excessive false positives and inadequate adaptation to sophisticated multi-stage schemes that exploit modern financial networks. Graph analytics and AI are promising tools, but they struggle with the fuzziness of laundering patterns, which exhibit s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*BlazingAML: High-Throughput Anti-Money Laundering (AML) via Multi-Stage Graph Mining*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 Anti-Money Laundering (AML) 系统面临以下挑战：
- **高误报率**：基于规则的方法产生大量 false positives。
- **难以适应复杂洗钱模式**：现代洗钱行为具有多阶段、结构模糊（structural fuzziness）和时间顺序灵活（temporal fuzziness）的特点，传统方法无法有效建模。
- **图挖掘效率低下**：现有的图模式匹配系统在处理模糊模式时需枚举所有变体，导致组合爆炸，运行开销巨大。

### 🚀 提出的新方法与创新思路
作者提出 **BlazingAML**，一个可扩展的 AML 系统，其核心创新包括：

#### （1）**多阶段框架（Multi-Stage Framework）**
- 将复杂的洗钱模式（如 scatter-gather、cycle）分解为一系列逻辑阶段（logical stages），每个阶段通过基本图操作（如 `out()`、`in()`、`intersection`）连接。
- 支持表达 **结构性模糊性**（中间账户数量可变）和 **时间模糊性**（非严格时间顺序，仅满足局部约束即可）。
- 统一抽象使不同模式可用相同原语描述，提升表达力与复用性。

#### （2）**领域专用编译器（Domain-Specific Compiler）**
- 接收高层声明式模式定义（如 YAML 配置），自动编译生成高性能 C++ 和 CUDA 内核代码。
- 自动进行多项优化：
  - **幂律感知内存访问**（Power-law-aware memory access）
  - **基于度数的工作负载均衡**（Degree-based workload balancing）
  - **CPU-GPU 流水线执行**
- 分离“模式逻辑”与“性能优化”，让金融分析师无需掌握并行编程即可部署高效算法。

#### （3）端到端 AML 流水线设计
- 图模式挖掘结果作为边特征（如 pattern count）输入下游分类器（如 XGBoost），形成 hybrid detection pipeline。
- 兼顾准确性与吞吐量，适用于实时大规模交易流检测。

### 🔍 相比现有方法的优势
| 方面 | BlazingAML | 传统方法（如 GFP） |
|------|------------|------------------|
| 模式表达能力 | 支持模糊结构与时间约束 | 固定模板，需枚举所有变体 |
| 性能 | 编译优化后实现极高吞吐 | 手工实现，效率低 |
| 可维护性 | 新模式只需修改配置文件 | 每种变体需独立开发 |
| 硬件适配 | 自动生成 CPU/OpenMP 与 GPU/CUDA 代码 | 通常只支持单一平台 |

---

## 2. 核心实验方法和设置

### 📊 数据集
使用 IBM Research 发布的合成 AML 数据集 [1]，共六类，按交易密度和规模划分：

| 数据集类别 | 节点数 | 边数 |
|----------|--------|-------|
| LI-Small ~ LI-Large | ~70万–207万 | ~700万–1.76亿 |
| HI-Small ~ HI-Large | ~51万–211万 | ~500万–1.8亿 |

其中：
- **LI**: Low Illicit（低欺诈密度）
- **HI**: High Illicit（高欺诈密度）

此外，在可扩展性测试中还使用了 **Trovares 合成图数据集**（从 10K 到 100M 条边），验证系统随图规模增长的表现。

### ⚙️ 实验设置
- **硬件平台**：
  - CPU：双路 Intel Xeon Platinum 8380（共 80 SMT 线程）
  - GPU：NVIDIA A40（最多四张）
- **并行配置**：BlazingAML 在 CPU 上测试 1–256 线程，在 GPU 上测试单卡性能。
- **评估指标**：
  - **F1 Score**：衡量分类准确性的主要指标（因数据高度不平衡）
  - **End-to-end Throughput**：每秒处理的边数（edges/sec），反映系统吞吐能力
  - **Speedup**：相对于基线的加速比

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **GFP [4]** | IBM 提出的 state-of-the-art 方法，提取子图特征 + XGBoost 分类器，是主要比较对象 |
| **FraudGT [19]** | 基于 Graph Transformer 的先进模型，用于对比 AI 架构与特征工程路线的权衡 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）F1 Score 表现（表 2 & 图 11）
- BlazingAML **完全复现 GFP 的特征输出**，因此在相同分类器下达到 **相同的 F1 Score**。
- 添加更多图模式特征（Fan → Degree → Cycle → Scatter-Gather）显著提升 F1：
  - 在 HI-Large 上，从 XGB-only 的 20.2 提升至 **58.1**
  - 表明结构化特征对检测至关重要
- HI 数据集表现远优于 LI，说明欺诈密度越高，模型越容易学习

#### （2）与 GFP 的性能对比（图 6–9）
BlazingAML 在保持精度的同时实现巨大性能提升：

| 模式 | CPU 加速比（vs GFP 64线程） | GPU 加速比 |
|------|----------------------------|-----------|
| **Scatter-Gather** | **210×**（平均） | **333×** |
| **Cycle** | 最高达 159× | — |
| **Fan-in/out** | ~11.4×（32线程） | 进一步优化可达更高 |
| **Stack** | 最高 25.8×（64线程） | **33.5×** |

> 即使单线程 CPU 版本也已超过 GFP 的 64 线程版本，显示编译优化的强大效果。

#### （3）与 FraudGT 的对比（图 12 & 表 4）
| 指标 | BlazingAML（128线程） | FraudGT |
|------|------------------------|---------|
| **平均吞吐量** | **4.9× 更高**（edges/sec） |
| **F1 Score** | 略低（如 HI-Medium: 51.1 vs 62.3）但更高效 |
| **适用场景** | 更适合大规模实时检测 |

> 结论：BlazingAML 以轻微精度代价换取 **数量级更高的吞吐**，更适合生产环境中的实时监控。

#### （4）可扩展性研究（图 10）
- 在 Trovares 数据集上（边数从 10K 到 100M）：
  - BlazingAML 展现出近乎线性的扩展性
  - 平均相对 GFP 实现 **27.5× 加速**
  - GPU 实现 **24.4× 加速**（Trovares-100M）
- 随着图规模增大，优势更加明显，符合“大图更受益于编译优化”的设计理念

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **模糊模式可通过统一的多阶段框架高效表达**  
   复杂洗钱行为（scatter-gather、cycle 等）可以被分解为通用图操作序列，自然支持结构与时间上的灵活性。

2. **领域专用编译器极大提升了图挖掘系统的生产力与性能**  
   无需手动编写并行代码，即可自动生成高度优化的 CPU/GPU 内核，实现数百倍加速。

3. **特征工程 + 轻量级分类器仍具强大竞争力**  
   尽管 FraudGT 使用先进的 Graph Transformer，BlazingAML 凭借高效的 pattern mining + XGBoost 实现了 **4.9× 更高的吞吐**，证明在实际部署中效率至关重要。

4. **系统具备优异的可扩展性与硬件适应性**  
   支持从单机多核到 GPU 的多种部署方式，且在超大规模图上仍保持高性能。

### ⚠️ 方法的局限性
- **依赖预定义模式**：仍属于“known pattern detection”，对于全新未知的洗钱策略可能不敏感。
- **未探索更复杂 AI 模型集成**：当前工作聚焦于图挖掘部分，下游仍使用 XGBoost；若换用 GNN 或 LLM 可能进一步提准，但会牺牲速度。
- **合成数据验证**：虽然 IBM 数据集被认为是 state-of-the-art 合成基准，但仍缺乏真实金融数据验证。

### 🔮 未来工作方向
- 扩展多阶段框架以支持 **动态图更新与增量挖掘**
- 集成 **Graph Neural Networks** 或 **LLM-based anomaly detection** 作为补充机制
- 探索 **unsupervised 或 self-supervised learning** 与 pattern mining 的结合，减少对标签依赖
- 将编译器扩展至其他图分析任务（如反欺诈、供应链风险检测）

---

> **总结一句话**：  
> BlazingAML 通过 **multi-stage 抽象 + 领域专用编译器**，实现了在不损失检测精度的前提下，将 AML 图模式挖掘的吞吐量提升 **两个数量级**，为构建高效、可扩展的实时反洗钱系统提供了新范式。

</details>

---

### 7. [Fast and principled equation discovery from chaos to climate](https://arxiv.org/abs/2604.11929)

**Authors**: Yuzheng Zhang, Weizhen Li, Rui Carvalho  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.11929v1  

#### Abstract
Our ability to predict, control, and ultimately understand complex systems rests on discovering the equations that govern their dynamics. Identifying these equations directly from noisy, limited observations has therefore become a central challenge in data-driven science, yet existing library-based ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast and principled equation discovery from chaos to climate

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文致力于解决**从稀疏且含噪声的观测数据中自动发现复杂系统控制方程**（governing equations）这一核心挑战。现有的基于库的稀疏回归方法（library-based sparse regression）在以下三个方面存在根本性权衡（trilemma）：
- **自动化**（Automation）：需要大量手动调参；
- **统计严谨性**（Statistical Rigor）：缺乏对模型选择和参数不确定性的量化；
- **计算效率**（Computational Efficiency）：高成本阻碍大规模应用。

### 提出的新方法：Bayesian-ARGOS
作者提出了一种名为 **Bayesian-ARGOS** 的混合框架，其核心思想是通过**策略性分解**来协调上述三难困境：
- **两阶段流程**：
  1. **Frequentist Screening Stage**（频繁主义筛选阶段）：采用快速、高效的双通自适应LASSO（adaptive lasso）结合交叉验证和BIC准则，对庞大的候选函数库进行激进降维。
  2. **Bayesian Inference Stage**（贝叶斯推断阶段）：在筛选后的精简模型空间上，利用Hamiltonian Monte Carlo (HMC) 进行贝叶斯后验采样，实现**有原则的不确定性量化**（principled uncertainty quantification）。

该方法巧妙地结合了两种范式的优点：**频繁主义方法的速度** 和 **贝叶斯方法的统计严谨性**。

### 相比现有方法的优势
- **更高的数据效率**（Data Efficiency）：在更少的数据量下即可成功识别方程。
- **更强的噪声鲁棒性**（Noise Robustness）：在高噪声条件下表现更优。
- **显著提升的计算效率**：相比基于bootstrap的全贝叶斯方法，计算成本降低约两个数量级（100倍加速）。
- **自动化程度高**：减少了对手动超参数调整的依赖。
- **提供丰富的诊断工具**：支持PSIS-LOO、VIF等标准统计诊断，可揭示模型失效模式。

---

## 2. 核心实验方法和设置

### 数据集
1. **基准混沌系统**（Benchmark Chaotic Systems）：
   - 使用了7个经典的三维混沌系统作为测试平台，包括：
     - Lorenz, Thomas, Rössler, Dadras, Aizawa, Sprott, Halvorsen。
   - 这些系统涵盖了多种非线性项（如三角函数、高阶多项式），被推荐用于评估系统辨识算法。
2. **高维现实世界数据**：
   - **全球海表温度**（Sea Surface Temperature, SST）数据，来自NOAA，时间跨度为1992–2019年，共1,400周数据，空间分辨率为180×360网格。

### 实验设置和评估指标
- **数据生成**：
  - 对每个混沌系统，随机选取100个初始条件，生成不同长度（`n`）和信噪比（SNR）的时间序列。
  - 添加高斯噪声以模拟真实观测环境。
- **评估指标**：
  - **成功率**（Success Rate）：在100次试验中，正确识别出所有真实项的比例。设定80%（Aizawa系统为70%）为可靠识别阈值。
  - **计算时间**（Runtime）：比较各方法的执行时间。
  - **预测误差**：在SST任务中，使用**均方根误差**（RMSE）评估重构场的准确性。

### 基线方法对比
- **SINDy**（Sparse Identification of Nonlinear Dynamics）：经典稀疏回归方法，速度快但缺乏不确定性量化。
- **ARGOS**：一种自动化的频繁主义管道，结合了自适应LASSO、信息准则和bootstrap置信区间，统计性较强但计算开销大。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
#### 在7个混沌系统上的综合表现（见图2, 3a, S1）
| 指标 | Bayesian-ARGOS 表现 |
|------|---------------------|
| **数据效率** | 在所有7个系统中，达到成功识别阈值所需的观测数均少于SINDy；在5/7系统中优于ARGOS。例如，Rössler、Dadras和Halvorsen系统仅需约 $10^{2.4}$ 至 $10^{2.7}$ 个观测点。 |
| **噪声鲁棒性** | 在固定 `n=5000` 下，5个系统在SNR=27dB时成功率超80%；Thomas和Halvorsen系统能容忍低至SNR=17dB的噪声。相比SINDy，在6/7系统中表现更优。 |
| **计算效率** | 相比ARGOS，**计算速度提升约两个数量级**（100倍）。例如，在 `n=10^5` 时，ARGOS耗时 >$10^{4.7}$ 秒，而Bayesian-ARGOS < $10^{2.5}$ 秒。SINDy最快，但无不确定性输出。 |

#### 在Aizawa系统上的特殊优势
- Aizawa系统的第三维方程包含高阶项，易导致设计矩阵严重**多重共线性**（multicollinearity）。
- Bayesian-ARGOS在所有 `n` 值下均优于两个基线方法，表现出对病态设计矩阵的**卓越鲁棒性**。

#### 高维SST任务结果（集成SINDy-SHRED框架）
- **有效率**（Valid Identification Rate）：
  - Bayesian-ARGOS：**77%**（82/107）
  - SINDy：**60%**（64/107）
- **预测精度**（在有效案例中）：
  - 平均潜空间MSE：0.263（Bayesian-ARGOS） vs. 0.334（SINDy）
  - 平均重构RMSE：1.055 vs. 1.282
- **长时预测稳定性**：Bayesian-ARGOS的误差增长更缓慢，长期预测更稳定（见图8a）。

### 消融分析与机制解释
- **为何更优？**
  - **SINDy** 易因库内相关性而在噪声下过选择（overselection）。
  - **ARGOS** 因强正则化可能遗漏弱项（false negative），尤其在中等样本量下（如Lorenz系统中的 `-x₂` 项）。
  - **Bayesian-ARGOS** 通过平衡的收缩（balanced shrinkage）避免了这两种极端。
- **异常行为诊断**：
  - **Aizawa系统在大数据量下性能下降**：由VIF诊断发现，随着 `n` 增加，候选特征间多重共线性加剧，导致回归病态。
  - **Dadras系统在大数据量下性能下降**：PSIS-LOO诊断发现存在多个**影响点**（influential observations），扭曲了后验推断。
  - **极低噪声下性能下降**：残差分析显示出现**异方差性**（heteroscedasticity），违反了同方差假设，导致过选择。

---

## 4. 关键结论和发现

### 主要发现
1. **三难困境可以被协调**：通过将“快速筛选”与“精确推断”分离，Bayesian-ARGOS成功实现了**自动化、统计严谨性和计算效率的统一**。
2. **更多数据/更少噪声并非总是更好**：在混沌系统辨识中，密集采样可能加剧特征冗余（多重共线性），而极低噪声会暴露模型假设的缺陷（如异方差性），反而导致性能下降。
3. **诊断透明性至关重要**：论文提供的统计诊断工具（PSIS-LOO, VIF, 残差分析）能够将“黑箱失败”转化为**可解释、可行动的诊断信号**，指导模型改进。
4. **模块化设计具有扩展性**：Bayesian-ARGOS可无缝集成到深度学习框架（如SINDy-SHRED）中，显著提升高维时空系统的潜动力学识别率和长期预测稳定性。

### 方法的局限性
- **依赖候选库的表达能力**：若真实动态项不在候选库中，则无法识别。方法本身不解决库的设计问题。
- **未完全处理选择偏差**（Selection Uncertainty）：目前的贝叶斯推断仅作用于筛选后的模型，尚未将筛选过程的不确定性纳入完整后验（post-selection inference 仍是开放问题）。
- **对特定系统在极端噪声下略逊于基线**：如在Sprott和Halvorsen系统中，Bayesian-ARGOS在某些低SNR条件下成功率略低于ARGOS，这是其“平衡收缩”策略的权衡结果。

### 未来工作方向
- 将不确定性量化扩展到整个流水线，包括频繁主义筛选阶段。
- 开发更智能的候选库构建策略，结合领域先验或符号回归。
- 探索与其他神经网络架构（如用于PDE或时变系统的模型）的集成。
- 应用于更多现实场景，如气候建模、神经科学和合成生物学等领域。

</details>

---

### 8. [RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair](https://arxiv.org/abs/2604.12820)

**Authors**: Jagadeesh Rachapudi, Pranav Singh, Ritali Vatsi, Praful Hambarde, Amit Shukla  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.12820v1  

#### Abstract
Large language models (LLMs) inherently absorb harmful knowledge, misinformation, and personal data during pretraining on large-scale web corpora, with no native mechanism for selective removal. While machine unlearning offers a principled solution, existing approaches are provider-centric, requirin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在预训练过程中会无差别地吸收有害知识、虚假信息和个人隐私数据，且目前缺乏**原生机制来选择性遗忘特定知识**。现有的机器遗忘（Machine Unlearning, MU）方法大多由模型服务提供商（MSP）主导，依赖完整的训练流程和保留数据集（retain dataset），普通用户无法自主控制自己的数据是否被遗忘。

这导致两个核心问题：
- **用户被动**：用户无法直接请求模型“忘记”关于自己的信息。
- **治理缺失**：违反了 GDPR 和 CCPA 等法规中的“被遗忘权”。

### 提出了什么新方法或新思路
本文提出了一种全新的范式——**Interactive Machine Unlearning (IMU)**，即**交互式机器遗忘**，允许终端用户通过自然语言指令在推理阶段让 LLM 忘记特定知识。

为实现 IMU，作者提出了 **RePAIR 框架**，其核心是 **STAMP（Steering Through Activation Manipulation with PseudoInverse）** 方法，包含三个模块：
1. **Mwatchdog**：检测用户的遗忘意图并提取需遗忘的 `(prompt, response)` 对。
2. **Msurgeon**：生成用于模型修复的代码。
3. **Mpatient**：原始模型，在运行时被动态修改参数以完成遗忘。

STAMP 的核心技术是在 MLP 层中通过伪逆（pseudoinverse）对激活值进行重定向，将其推向一个“拒绝子空间”（refusal subspace），从而实现无需梯度计算的知识移除。

还提出了低秩变体 **STAMP-LR**，显著降低计算复杂度。

### 相比现有方法的优势
| 特性 | 现有方法（如 GA, NPO, FLAT, ASU） | RePAIR / STAMP |
|------|-------------------------------|----------------|
| 是否训练自由（training-free） | ❌ 需要反向传播和优化 | ✅ 仅前向传播 + 伪逆更新 |
| 支持单样本遗忘（single-sample） | ❌ 多数需要批量数据 | ✅ 可处理单条请求 |
| 用户可参与 | ❌ 完全由 MSP 控制 | ✅ 用户可通过自然语言发起 |
| 推理时执行 | ❌ 需离线再训练 | ✅ 在设备上实时完成 |
| 计算效率 | 较高（O(d³) 或更高） | STAMP-LR 达到 ~3× 加速 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **WMDP-Bio**：1K 条生物医学领域的有害知识样本，用于评估**有害知识抑制**。
- **MMLU**：1K 条带有错误答案的多学科问题，用于评估**虚假信息纠正**。
- **合成个人资料数据集**：使用 Mistral-7B API 生成的 2K 条虚构人物传记，用于评估**个人数据擦除**。
- **TinyStories**：用于评估模型整体语言建模能力（utility preservation）。

每个数据集划分为等量的 `Df`（forget set）和 `Dr`（retain buffer ≤10% of total），另设 `Dref`（200 条“我不知道”类拒绝提示）构建拒绝子空间。

### 实验设置和评估指标
#### 主要任务：
- 有害知识遗忘
- 虚假信息修正
- 个人数据删除

#### 评估指标：
| 指标 | 含义 | 理想值 |
|------|------|--------|
| **Accf ↓** | 忘记准确率（越低越好） | 0.00 |
| **Accr ↑** | 保留准确率（越高越好） | 接近 Oracle |
| **F-RL ↓** | 忘记集合上的 ROUGE-L 分数（衡量记忆残留） | 0.00 |
| **R-RL ↑** | 保留集合上的 ROUGE-L 分数（衡量知识保留） | 越高越好 |
| **Perplexity ↓** | 在 TinyStories 上的语言模型困惑度（衡量通用性能） | 越低越好 |
| **RTE (Runtime Efficiency) ↓** | 总耗时（分钟） | 越短越好 |

### 基线方法对比
共比较六种 SOTA 方法：
- **GA** [25]
- **NPO** [28]
- **RMU** [17]
- **FLAT** [24]
- **WGA** [23]
- **ASU** [27]

所有方法均在 Llama-3-8B 上测试，作为 `Mpatient`；`Mwatchdog` 使用 Mistral-7B；`Msurgeon` 使用 Qwen2.5-Coder-7B-Instruct。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）
| 方法 | 有害知识 Accf | 保留 Accr | 困惑度 | RTE(min) |
|------|---------------|-----------|--------|----------|
| **Base** | 75.30 | 78.50 | 5.90 | N/A |
| **Oracle** | N/A | 77.37 | 6.10 | N/A |
| **STAMP** | **0.00** | 70.13 | 6.55 | 7.13 |
| **STAMP-LR** | **0.00** | **73.27** | 7.00 | **4.25** |

| 方法 | 虚假信息 Accf | 保留 Accr | 困惑度 | RTE(min) |
|------|----------------|-----------|--------|----------|
| **STAMP-LR** | **0.00** | **84.47** | 7.39 | **2.57** |

| 方法 | 个人数据 F-RL | R-RL | 困惑度 | RTE(min) |
|------|----------------|-------|--------|----------|
| **STAMP-LR** | **0.00** | **0.88** | 8.17 | **4.01** |

> ✅ 所有任务中，**STAMP 和 STAMP-LR 均达到 Accf = 0.00 或 F-RL = 0.00**，表示完全遗忘目标内容。

> ✅ **STAMP-LR 在保留性能（Accr/R-RL）上接近甚至优于部分基线**，同时保持较低困惑度。

> ⏱️ **STAMP-LR 实现高达 ~3× 的速度提升**，尤其在虚假信息任务中仅需 2.57 分钟。

### 与基线方法的对比结果
- **遗忘效果最好**：STAMP 系列在所有任务中实现了**近乎完美的遗忘**（Accf ≈ 0），而 WGA、FLAT 等仍有残余记忆（如 Accf 高达 2.47）。
- **保留能力更强**：相比 GA/NPO/WGA 等基于梯度的方法，STAMP 更少损害保留知识（Accr 更高）。
- **实用性更强**：训练自由 + 单样本支持 + 快速响应，适合部署于边缘设备或移动端。

### 消融实验结果（Ablation Studies）

#### （1）单层 vs 全层干预（Table 5）
| 设置 | F-RL | R-RL | 困惑度 | RTE(s) |
|------|------|------|--------|--------|
| Layer 7 only | 0.00 | 0.85 | 6.07 | **4.36** |
| All layers | 0.00 | 0.88 | 6.02 | 15.40 |

✅ **仅干预第7层即可获得接近全层的效果**，且速度快约 3.8×。  
👉 动机来自图5：Llama-3-8B 中 Layer 7 的 WMDP 与拒绝激活之间的余弦距离最大（0.867），最具区分性。

#### （2）低秩分解的秩 r 影响（Table 6）
| Rank(r) | R-RL ↑ | 困惑度 ↓ | RTE(min) |
|--------|--------|----------|----------|
| 64     | 0.85   | 6.10     | 4.01     |
| 128    | 0.88   | 6.07     | 5.24     |

✅ 当 `r ≥ 64` 时性能稳定；低于此阈值则遗忘不彻底。

#### （3）保留缓冲区大小影响（Table 7）
| Retain Ratio | R-RL | 困惑度 |
|-------------|------|--------|
| 0.10        | 0.88 | 8.83   |
| 1.00        | 0.90 | 7.07   |

✅ 即使只保留 10% 的 retain 数据，性能仍基本稳定，有利于边缘部署。

#### （4）单样本遗忘能力（Table 8）
当 `|Df|=1` 时：
- 所有训练型基线（GA/NPO/FLAT 等）**完全失效**（Accf = 100）
- **STAMP 和 STAMP-LR 依然实现 Accf = 0.00**

✅ 这是唯一能在**单样本场景下有效工作的方法**，验证了其对 IMU 场景的高度适配性。

---

## 4. 关键结论和发现

### 主要发现
1. **首次实现真正意义上的用户驱动遗忘**：通过 IMU 范式，用户可用自然语言直接命令模型“忘记某事”，无需依赖 MSP。
2. **STAMP 是首个满足“训练自由 + 单样本遗忘”的 LLM 遗忘方法**，基于伪逆的激活重定向机制高效且无需反向传播。
3. **STAMP-LR 显著提升效率**，将复杂度从 O(d³) 降至 O(r³ + r²·d)，支持在设备端快速执行。
4. **RePAIR 框架端到端可行**：Mwatchdog 能准确识别遗忘请求（>96%），Msurgeon 生成有效修复代码（>97% 成功率），Mpatient 成功转变为 Mhealed。
5. **遗忘质量媲美 Oracle**：在多项任务中达到近零遗忘分数，同时保留大部分原有能力。

### 方法的局限性
1. **仍需少量保留数据（retain buffer）**：尽管只需 10%，但在严格合规环境下存储任何历史数据都可能构成风险。
2. **依赖拒绝模板构建拒绝子空间**：当前利用“我不知道”等输入诱导一致性激活，泛化性有待进一步验证。
3. **资源限制下的扩展挑战**：虽然轻量化，但在极低端设备或多模态模型中应用仍需优化。
4. **未解决长期累积遗忘的影响**：连续多次遗忘操作可能导致参数漂移或功能退化。

### 未来工作方向
- 开发**完全无需 retain data 的遗忘方法**（retain-free unlearning）。
- 将 RePAIR 扩展至**多模态基础模型**（如 VLMs）。
- 研究**持续遗忘机制**，支持长期、多次、安全的用户编辑。
- 探索更鲁棒的**拒绝行为生成策略**，避免模式僵化。
- 结合联邦学习框架，实现去中心化的隐私保护推理系统。

--- 

> 📌 **总结一句话**：  
> RePAIR 首次实现了**用户可在推理时通过自然语言指令让 LLM “主动遗忘”特定知识**，提出的 STAMP 方法在**训练自由、单样本支持、高效性和遗忘效果**方面全面超越现有 SOTA，为实现透明、可控、合规的人工智能系统提供了重要路径。

</details>

---

### 9. [LoSA: Locality Aware Sparse Attention for Block-Wise Diffusion Language Models](https://arxiv.org/abs/2604.12056)

**Authors**: Haocheng Xi, Harman Singh, Yuezhou Hu, Coleman Hooper, Rishabh Tiwari, Aditya Tomar, Minjae Lee, Wonjun Kang, Michael Mahoney, Chenfeng Xu, Kurt Keutzer, Amir Gholami  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.12056v1  

#### Abstract
Block-wise diffusion language models (DLMs) generate multiple tokens in any order, offering a promising alternative to the autoregressive decoding pipeline. However, they still remain bottlenecked by memory-bound attention in long-context scenarios. Naive sparse attention fails on DLMs due to a KV I...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# LoSA: Locality Aware Sparse Attention for Block-Wise Diffusion Language Models —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **KV Inflation 问题**：在 block-wise Diffusion Language Models (DLMs) 中，尽管每个 query 只选择少量 prefix KV 位置进行 sparse attention，但由于 block 内不同 query 选择的 KV 位置差异大，导致实际需要加载的 **KV 位置并集（KV Union）非常大**，严重削弱了稀疏性带来的加速效果。
- 这使得传统的 sparse attention 方法（如 QUEST）在 DLM 上效率低下，无法有效降低内存带宽开销。

### ✅ 提出的新方法：LoSA (Locality-aware Sparse Attention)
- **核心洞察**：在连续的 denoising 步骤之间，只有少数 token（称为 **Active Tokens**）的 hidden states 发生显著变化，大多数 token（**Stable Tokens**）保持稳定。
- **新思路**：
  - 对于 **Stable Tokens**，直接复用上一步缓存的 prefix-attention 输出（`op`, `Lp`），避免重新计算。
  - 仅对 **Active Tokens** 执行 sparse attention，并基于其查询动态选择 KV 位置。
  - 利用 **online-softmax** 技术合并 prefix 和 block 内 attention 结果。

### ✅ 相比现有方法的优势
| 方面 | LoSA 优势 |
|------|----------|
| **准确性** | 显著优于 naive sparse attention（如 QUEST），在高稀疏度下仍接近 dense attention 精度，平均提升高达 **+9.01%**。 |
| **效率** | 大幅减少需访问的 KV 位置数量，实现最高 **4.14× 的 attention 速度提升**（RTX A6000）。 |
| **注意力密度** | 平均比 QUEST 低 **1.54× 的 attention density**，意味着更少的内存访问。 |
| **通用性** | 方法与具体 sparse selector（如 QUEST）正交，可与其他优化技术结合。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **LongBench**（主评测）：用于评估长上下文理解能力，包含多个子任务：
  - HotPotQA
  - TriviaQA
  - NarrativeQA
  - Qasper
  - MultiFieldQA
- **Commonsense Reasoning Benchmarks**（辅助验证）：
  - HellaSwag
  - WinoGrande
  - BoolQ

### ⚙️ 实验设置
- **模型**：
  - Trado-8B-Instruct
  - Trado-4B-Instruct
  - SDAR-8B-Chat
- **Block Size**：默认为 16，部分实验测试 32。
- **Context Length**：最长达 64K tokens。
- **Per-Query Budget**：sparse attention 的检索预算设为 128、256、512、1024。
- **硬件平台**：
  - 主要测试：NVIDIA RTX A6000
  - 新硬件验证：RTX 5090

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 在 LongBench 和 reasoning 任务上的准确率 |
| **KV-Cache Density (%)** | 被选中的 prefix token 占比，反映内存负载 |
| **Attention Speedup** | 相对于 dense attention 的前缀 attention 加速比 |
| **Latency Breakdown** | 分析各模块耗时（criticality estimation, locality pruning, attention computation 等） |

### 🔁 基线方法对比
| 方法 | 简介 |
|------|------|
| **Dense Attention** | 全量 attention，作为精度上限基准 |
| **QUEST** | 原为 autoregressive LLM 设计的 query-aware sparse attention，本文将其适配到 DLM 作为主要 baseline |
| **SparseD** | 在初始化时固定 sparse pattern 并跨步骤复用 |
| **Sparse-dLLM** | 动态淘汰 KV cache 条目以加速推理 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ LongBench 上的准确率表现（Trado-8B-Instruct）
| Budget | Method | Average Accuracy | Δ vs Baseline |
|--------|--------|------------------|---------------|
| 128 | QUEST | 31.54% | — |
| 128 | **LoSA** | **41.97%** | **+10.43** |
| 256 | QUEST | 34.64% | — |
| 256 | **LoSA** | **42.14%** | **+7.50** |
| 1024 | QUEST | 37.69% | — |
| 1024 | **LoSA** | **42.88%** | **+5.19** |

> ➕ LoSA 在极低预算（128）下即达到接近 dense 模型（44.86%）的性能。

#### ✅ KV-Cache 密度对比（平均降低 1.54×）
| Budget | QUEST (avg.) | LoSA (avg.) | Reduction |
|--------|--------------|-------------|---------|
| 128 | 4.24% | 2.58% | **1.64×** |
| 256 | 8.08% | 5.04% | **1.60×** |
| 512 | 14.78% | 9.71% | **1.52×** |
| 1024 | 25.66% | 18.23% | **1.41×** |

#### ✅ 推理延迟与加速比（TriviaQA, RTX A6000）
| 方法 | Speedup |
|------|--------|
| QUEST | 3.26× |
| **LoSA** | **4.14×** |

> 在 RTX 5090 上也取得 **3.67×** 加速，表明该优化在新一代 GPU 上依然有效。

#### ✅ 消融实验支持核心设计
- **Locality Pruning 必要性**：若不区分 active/stable tokens，性能大幅下降。
- **初始化使用 dense attention 更优**：首次 denoising 使用 dense attention 可为后续步骤提供高质量缓存，显著提升最终 accuracy。
- **Top-k 选择策略敏感性分析**：选取 top-5 变化最大的 token 作为 active tokens 效果最佳。

---

## 4. 关键结论和发现

### 🔍 主要发现
1. **Representation Change 具有强局部性（Locality）**：
   - 连续 denoising 步骤间，仅有少量 token 表示发生剧烈变化。
   - 该现象在 early 和 late 层更为明显（见 Figure 4）。
2. **KV Inflation 是 block-wise DLM 中 sparse attention 失效的根本原因**：
   - 不同 query 的 KV 选择缺乏重叠，导致 union 过大。
3. **LoSA 成功缓解 KV Inflation**：
   - 通过只对 active tokens 执行 sparse attention，将参与 selection 的 query 数从 `B` 降到 `|A|`，显著缩小 KV Union。
4. **效率与精度双提升**：
   - 不仅更快，而且更准 —— 因为 stable tokens 保留了 full dense attention 信息，而非受限于 sparse subset。

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **短上下文收益有限** | 当 context length < 1K 时，KV Inflation 不显著，LoSA 优势减弱（见 Table 5）。 |
| **Batch Size = 1 限制** | 当前实现未集成到批量服务系统（如 vLLM、TensorRT-LLM），影响实际部署场景适用性。 |
| **首次迭代需 dense attention** | 引入固定开销，虽然后续迭代可摊销，但在极短生成任务中可能不划算。 |

### 🔮 未来工作方向
- 将 LoSA 扩展至 **batched inference** 场景，支持高并发服务。
- 探索 **adaptive thresholding** 或 **learnable selection** 机制来自动生成 active token 集合。
- 结合其他 KV Cache 优化技术（如 PagedAttention、SnapKV）进一步提升端到端吞吐。
- 应用于 **non-diffusion non-autoregressive models** 中探索是否同样存在 representation locality。

---

## 总结一句话
> LoSA 通过挖掘 block-wise diffusion 模型中 **representation change 的局部性**，提出一种 **仅对 active tokens 执行 sparse attention + 缓存 stable tokens 输出** 的新范式，在几乎无精度损失的前提下实现了高达 **4.14× 的 attention 加速** 和 **1.54× 的 KV 访问压缩**，为高效 DLM 推理提供了新思路。

</details>

---

### 10. [Interpretable Relational Inference with LLM-Guided Symbolic Dynamics Modeling](https://arxiv.org/abs/2604.12806)

**Authors**: Xiaoxiao Liang, Juyuan Zhang, Liming Pan, Linyuan L\"u  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.12806v1  

#### Abstract
Inferring latent interaction structures from observed dynamics is a fundamental inverse problem in many-body interacting systems. Most neural approaches rely on black-box surrogates over trainable graphs, achieving accuracy at the expense of mechanistic interpretability. Symbolic regression offers e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Interpretable Relational Inference with LLM-Guided Symbolic Dynamics Modeling

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**从观测的动力学轨迹中推断潜在交互结构（latent interaction structures）**这一逆问题。在许多复杂系统（如物理、生物、流行病学等）中，实体间的交互图结构通常是未知或不可观测的，而节点状态的时间序列是可测量的。传统方法面临以下挑战：
- **神经网络方法**（如NRI）虽然能联合学习结构和动力学，但其“黑箱”特性导致缺乏**机制可解释性**（mechanistic interpretability），且易因过参数化而拟合虚假边。
- **符号回归**（Symbolic Regression, SR）虽能生成显式的动力学方程，但通常假设已知图结构或依赖固定的函数库（fixed function library），难以适应未知拓扑和复杂非线性。

### 提出的新方法：COSINE
作者提出了 **COSINE**（Co-Optimization of Symbolic Interactions and Network Edges），一个端到端可微分的框架，用于**联合发现交互图结构和稀疏符号动力学方程**。

#### 核心创新点：
1. **联合优化结构与符号动力学**  
   将动力学分解为**消息传递**（message）和**状态更新**（update）两个模块，分别通过共享的符号函数库进行稀疏线性组合建模。这使得图结构 $A$ 和动力学系数 $W$ 可以通过梯度下降共同优化。

2. **稀疏符号消息传递**（Sparse Symbolic Message Passing）  
   引入稀疏正则化（$l_1$ regularization）约束系数矩阵 $W_{msg}$ 和 $W_{upd}$，强制模型选择最简洁的函数组合。这种稀疏性作为一种功能约束，防止过参数化的动态掩盖虚假边，从而增强**结构可识别性**（structural identifiability）。

3. **LLM引导的函数库演化**（LLM-Guided Library Evolution）  
   这是最具突破性的设计。COSINE采用**双层闭环架构**：
   - **内环**：基于当前符号库 $\Phi(\cdot)$，通过梯度优化联合学习图结构和动力学系数。
   - **外环**：利用一个大型语言模型（LLM）作为“符号监督者”，根据内环反馈（如验证损失、残差模式、系数重要性）来**主动修剪冗余项**或**增补新的候选函数**，动态调整符号库。
   - 该策略解耦了**假设生成**（由LLM完成）与**假设选择**（由数值优化完成），确保最终发现的机制既新颖又数值可靠。

### 相比现有方法的优势
| 方面 | 传统方法（NRI, SINDy） | COSINE |
|------|------------------------|--------|
| **可解释性** | 黑箱，难以理解机制 | 显式符号方程，机制对齐 |
| **灵活性** | 固定函数库或固定结构 | 动态扩展/修剪函数库，适应未知结构 |
| **鲁棒性** | 易受噪声和过拟合影响 | 稀疏性+LLM反馈提升抗噪能力 |
| **数据效率** | 需大量数据训练神经网络 | 符号先验降低样本需求 |

---

## 2. 核心实验方法和设置

### 数据集
实验分为两部分：

#### （1）合成数据集（Synthetic Data）
- **图结构**：三种标准网络模型
  - **Erdos-Rényi (ER)**：随机图
  - **Barabasi-Albert (BA)**：无标度网络
  - **Watts-Strogatz (WS)**：小世界网络
- **动力学系统**：六类代表性非线性系统
  - **Michaelis-Menten (MM)**：生化反应动力学
  - **Diffusion (Diff)**：扩散过程
  - **Spring (Spr)**：弹簧网络（机械耦合）
  - **Kuramoto (Kura)**：相位同步振荡器
  - **Friedkin-Johnsen (FJ)**：意见动力学
  - **Coupled Map Network (CMN)**：混沌映射网络

#### （2）真实世界数据集
- **COVID-19 流行病数据**：来自美国四个州（Arizona, Connecticut, Illinois, Michigan）的县级每日确诊病例数。
- 构建地理邻接图，研究疫情传播的潜在机制。

### 实验设置与评估指标

#### 评估指标
- **关系推理性能**：使用 **AUC**（Area Under ROC Curve）衡量图结构恢复质量。
- **机制发现性能**：使用 **Term Accuracy** 衡量前 $K=3$ 个主导项是否覆盖真实机制中的基本原语（primitive coverage）。
- **预测性能**：使用 **PCC**（Pearson Correlation Coefficient）评估多步预测准确性。

#### 基线方法对比
| 类别 | 基线方法 |
|------|--------|
| **统计方法** | Granger Causality (GC), Mutual Information (MI), Transfer Entropy (TE) |
| **神经关系推理** | Neural Relational Inference (NRI), Graph Dynamics Prior (GDP) |
| **注意力模型** | Relational Inference via Variational Attention (RIVA) |

所有基线均使用公开实现并调优超参以保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 方法 | 平均 AUC (%) |
|------|-------------|
| GC | ~65 |
| MI | ~75 |
| TE | ~70 |
| NRI | ~85 |
| GDP | ~88 |
| **COSINE** | **~97** ✅ |

- 在几乎所有系统和图类型上，**COSINE 达到 SOTA 性能**，尤其在复杂非线性系统（如 Kuramoto on BA 图）上显著优于其他方法（AUC 99.85 vs. GDP 的 90.13）。
- 在低数据场景下（仅 10×10 样本），COSINE 仍保持高鲁棒性（Table 4），而神经方法严重退化。

### 与基线方法的对比结果
- **显著超越统计方法**：GC/MI/TE 在非线性系统中表现接近随机猜测。
- **优于深度学习模型**：NRI/GDP 虽然结构恢复较好，但在机制可读性和泛化性上不如 COSINE。
- **效率更高**：COSINE 使用稀疏线性回归替代重型 GNN，训练时间和 GPU 内存远低于 NRI（Figure 6）。

### 消融实验结果（Ablation Study）

#### （1）LLM 引导演化的必要性（Figure 4）
- **w/o update（固定库）**：性能大幅下降，尤其在 MM 和 CMN 系统上。
- **阈值剪枝（threshold-based pruning）**：无法有效引入新函数，陷入局部最优。
- **COSINE（LLM-guided）**：AUC 接近 1.0，证明 LLM 的主动假设生成至关重要。

#### （2）LLM 规模的影响（Table 3）
- 对简单系统（如 Spring, Kuramoto），小规模 LLM（8B）即可胜任。
- 对复杂非线性系统（如 MM, CMN），需要更大 LLM（14B~72B）才能稳定达到高性能。

#### （3）超参数敏感性分析（Figure 5）
- 模型对学习率有一定容忍范围，存在“性能高原”。
- 对正则化参数（$\beta_{KL}, \lambda_w$）高度鲁棒，即使强稀疏惩罚也能保持高 AUC。

---

## 4. 关键结论和发现

### 主要发现
1. **COSINE 实现了可解释的关系推理**：不仅能准确恢复图结构（AUC > 99%），还能发现**机制对齐的符号表达式**（如 Kuramoto 中的 `sin(xj - xi)`，扩散中的 `xj - xi`）。
2. **LLM 是突破“封闭世界”假设的关键**：传统符号回归受限于预定义库，而 LLM 能根据残差反馈主动探索新函数形式，实现真正的开放域假设搜索。
3. **稀疏性增强了结构可识别性**：通过稀疏回归抑制冗余项，避免了虚假边被复杂函数补偿的问题。
4. **在真实世界系统中揭示差异化机制**：在 COVID-19 数据中，COSINE 发现：
   - 所有地区都识别出 **mass-action 项 $x_i \cdot x_j$**，符合接触传播原理。
   - 密集城市区（AZ, CT）更依赖外部输入（`x·h`），而城乡混合区（IL, MI）表现出更强的本地非线性（如 `x(1-x)` 逻辑增长、负反馈）。

### 方法的局限性
- **对 LLM 能力有依赖**：极端复杂的动力学可能超出当前 LLM 的表达能力。
- **稀疏符号回归的表达能力有限**：对于高度复合或隐式定义的机制，可能无法完全捕捉。
- **符号发现可能存在非唯一性**：在噪声或数据不足时，多个函数可能近似等效，导致机制模糊。

### 未来工作方向
- 探索更高效的符号搜索策略（如结合 MCTS）。
- 将框架扩展至连续时间动态系统和异构网络。
- 应用于更多领域（如金融、气候、神经科学）以验证通用性。
- 研究如何进一步减少对 LLM 的依赖，例如通过自进化机制。

---

> **代码地址**：https://anonymous.4open.science/r/COSINE-6D43  
> **一句话总结**：COSINE 通过将 LLM 作为“假设生成引擎”与可微分符号回归结合，实现了**既准确又可解释**的复杂系统关系与机制联合发现。

</details>

---

### 11. [Enhancing Clustering: An Explainable Approach via Filtered Patterns](https://arxiv.org/abs/2604.12460)

**Authors**: Motaz Ben Hassine (CRIL), Sa\"id Jabbour (CRIL)  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12460v1  

#### Abstract
Machine learning has become a central research area, with increasing attention devoted to explainable clustering, also known as conceptual clustering, which is a knowledge-driven unsupervised learning paradigm that partitions data into $\theta$ disjoint clusters, where each cluster is described by a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Enhancing Clustering: An Explainable Approach via Filtered Patterns*

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对 **conceptual clustering** 中基于 **k-Relaxed Frequent Patterns (k-RFPs)** 的方法存在的一个关键瓶颈：  
多个不同的 k-RFPs 可能诱导出相同的 **k-cover**，导致候选模式集中存在大量**冗余的 symbolic representations**。这种冗余不仅扩大了搜索空间，还显著增加了后续 **ILP (Integer Linear Programming)** 求解器的计算复杂度和运行时间。

### 提出的新方法与新思路
作者提出了 **Optimized Conceptual Clustering Method (OCCM)**，其核心是一个**模式过滤框架**，旨在系统性地消除冗余模式。具体创新点如下：

- **理论分析**：形式化地刻画了不同 k-RFPs 诱导相同 k-cover 的条件（Proposition 2），为冗余检测提供了严格的理论基础。
- **过滤策略**：提出一种高效的 **Pattern Filtering Algorithm**，在将 k-RFPs 输入 ILP 模型之前，对每个唯一的 k-cover 仅保留一个代表性的模式。
- **代表性选择原则**：当多个模式共享同一个 k-cover 时，优先保留**最大的 itemset**（即 `argmax |L|`）。这一选择基于可解释性考虑——更大的模式提供更具体、信息量更丰富的集群描述。
- **可解释性评估**：引入两个新的理论度量来评估所选模式的**可解释性**和**代表性**：
  - **Shapley Value Variance (SVV)**：衡量模式内各 item 对集群代表性的贡献分布是否均衡。
  - **Average Cluster Stability (ACS)**：衡量从模式中移除单个 item 后，其诱导的集群的稳定性（通过 Jaccard Similarity）。

### 相比现有方法的优势
- **更高的计算效率**：通过减少输入 ILP 的决策变量数量，显著缩短了求解时间。
- **保持甚至提升聚类质量**：在多数数据集上保持了与基线方法相当的 F1-score，并在部分数据集（如 Mushroom, Primary-Tumor）上实现了提升。
- **增强的可解释性**：通过优先选择更大的模式，提升了集群描述的丰富性和表达能力。
- **理论严谨性**：为模式冗余现象提供了形式化的定义和证明。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在六个广泛使用的现实世界 transactional 数据集上进行：

| Dataset        | #Transactions | #Items | Density (%) |
|----------------|---------------|--------|-------------|
| Lymph          | 148           | 68     | 40          |
| Mushroom       | 8124          | 119    | 18          |
| Primary-Tumor  | 336           | 31     | 48          |
| Soybean        | 630           | 50     | 32          |
| Tic-tac-toe    | 958           | 27     | 33          |
| Vote           | 435           | 48     | 33          |

### 实验设置
- **松弛参数**：固定 `k = 1`。
- **聚类数**：固定 `θ = 2`（因所有数据集的真实划分均为两类）。
- **最小支持度阈值 `σ`**：
  - **Phase I**: 在 10% 到 40% 范围内变化以测试过滤效果。
  - **Phase II & III**: 为每个数据集选择最合适的 `σ` 值以最大化聚类质量。
- **超时限制**：ILP 求解最大运行时间为 1 小时。

### 评估指标
1. **模式数量**：比较过滤前 (`|A|`) 和过滤后 (`|Af|`) 的 k-RFP 数量。
2. **计算效率**：ILP 求解的 CPU 时间。
3. **聚类质量**：使用 **F1-score** 与真实标签（ground-truth clusters）进行比较。
4. **可解释性/代表性**：
   - **SVV** (Shapley Value Variance)
   - **ACS** (Average Cluster Stability)
   - 分析 SVV 与 ACS、模式大小与 ACS 之间的相关性。

### 基线方法对比
- **CCA-k-RFP-M1** (Hassine et al., 2024)：这是 OCCM 所要优化的原始方法，直接使用所有生成的 k-RFPs 进行 ILP 求解。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### (1) 模式过滤效果 (Phase I)
- 在所有数据集和所有 `σ` 设置下，均观察到冗余模式的存在。
- 过滤后模式数量 `|Af|` 始终小于过滤前 `|A|`，验证了冗余的普遍性。
- **最高压缩率**：在 `Tic-tac-toe` 数据集上，当 `σ=40%` 时，模式数量减少了 **26.67%**。
- 即使在高密度数据集（如 Lymph）上，也实现了约 **5-8%** 的减少。

#### (2) ILP 求解时间与聚类质量 (Phase II)
| Dataset        | Method         | |A| / |Af| | F1-score | CPU Time (s) |
|----------------|----------------|------------|----------|--------------|
| Lymph          | CCA-k-RFP-M1   | 91,888     | 0.71     | 49.66        |
|                | **OCCM**       | **85,470** | **0.71** | **29.74**    |
| Mushroom       | CCA-k-RFP-M1   | 19,712     | 0.34     | 3133.34      |
|                | **OCCM**       | **18,176** | **0.73** | **1144.46**  |
| Primary-Tumor  | CCA-k-RFP-M1   | 45,465     | 0.25     | 55.88        |
|                | **OCCM**       | **45,250** | **0.33** | **55.71**    |
| Soybean        | CCA-k-RFP-M1   | 11,900     | 0.29     | 18.29        |
|                | **OCCM**       | **11,664** | **0.29** | **16.99**    |
| Vote           | CCA-k-RFP-M1   | 280,386    | 0.51     | 1206.17      |
|                | **OCCM**       | **280,179**| **0.51** | **234.30**   |
| Tic-tac-toe    | Both           | —          | —        | Timeout (>1h)|

- **计算效率**：OCCM 在所有成功求解的数据集上均**显著降低了 CPU 时间**（例如，Mushroom 上从 3133s 降至 1144s，Vote 上从 1206s 降至 234s）。
- **聚类质量**：
  - 在 Lymph, Soybean, Vote 上，F1-score **完全持平**。
  - 在 Mushroom 和 Primary-Tumor 上，OCCM **显著优于** 基线方法（F1 从 0.34→0.73，0.25→0.33）。
  - 作者认为质量提升归功于过滤策略优先选择了更大的模式，从而获得了更具代表性的集群描述。

#### (3) 可解释性分析 (Phase III)
- **SVV 与 ACS 的关系**：在大多数数据集（Lymph, Mushroom, Soybean, Vote）上，观察到 **负相关**——贡献越均衡（SVV 越低）的模式，其诱导的集群越稳定（ACS 越高）。
- **模式大小与 ACS 的关系**：在所有数据集上均观察到**强正相关**——**模式越大，ACS 越高**，集群越稳定。
- **结论支持**：该发现验证了 OCCM 的设计选择——优先保留更大的模式，不仅能提升描述的丰富性，还能增强集群的鲁棒性和稳定性。

---

## 4. 关键结论和发现

### 主要发现
1. **冗余普遍存在**：在 k-RFP 框架下，多个不同模式诱导相同 k-cover 是一个普遍且重要的现象，是计算瓶颈的根源。
2. **过滤有效且必要**：提出的过滤算法能有效识别并去除冗余模式，显著减小搜索空间。
3. **效率与质量双赢**：OCCM 在**大幅提升计算效率**的同时，**保持甚至提升**了聚类质量。
4. **大模式更优**：实验证明，更大的模式不仅是更丰富的描述，也是更稳定、更具代表性的集群表示。这为模式选择提供了强有力的依据。

### 方法的局限性
- **Tic-tac-toe 数据集失败**：在高度复杂的 Tic-tac-toe 数据集上，即使经过过滤，ILP 求解仍超时，表明对于极端情况，仅靠过滤可能不足以解决可扩展性问题。
- **过滤作为后处理**：当前的过滤是在 SAT 生成 k-RFPs 之后进行的，属于后处理步骤，未能从根本上避免生成冗余模式的开销。
- **ILP 目标函数局限**：当前 ILP 仅最大化模式大小总和，可能未完全捕捉“最优”可解释性。

### 未来工作方向
1. **将去冗余机制集成到 SAT 生成过程**：设计更紧凑的约束，在生成阶段就避免产生冗余的 k-RFPs，进一步提升效率。
2. **改进 ILP 目标函数**：在目标函数中整合可解释性导向的标准（如 ACS 或 SVV 的优化），而不仅仅是模式大小。
3. **探索其他模式模型**：研究是否可以推广到其他类型的 relaxed patterns 或 clustering frameworks。

</details>

---

### 12. [RPRA: Predicting an LLM-Judge for Efficient but Performant Inference](https://arxiv.org/abs/2604.12634)

**Authors**: Dylan R. Ashley, Ga\"el Le Lan, Changsheng Zhao, Naina Dhingra, Zhipeng Cai, Ernie Chang, Mingchen Zhuge, Yangyang Shi, Vikas Chandra, J\"urgen Schmidhuber  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12634v1  

#### Abstract
Large language models (LLMs) face a fundamental trade-off between computational efficiency (e.g., number of parameters) and output quality, especially when deployed on computationally limited devices such as phones or laptops. One way to address this challenge is by following the example of humans a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：RPRA: Predicting an LLM-Judge for Efficient but Performant Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在推理质量和计算效率之间存在根本性权衡。大模型虽然性能强，但部署成本高、延迟大，难以在手机等边缘设备上运行；小模型虽高效，但在复杂任务上表现不稳定，容易产生“诡异”输出（如错误回答“strawberry中有几个r”）。  
本文旨在解决这一矛盾：**如何让小模型在自信能答好时自行响应，而在可能出错时主动请求大模型协助**，从而实现高效且可靠的推理。

### 提出了什么新方法或新思路
论文提出了两个核心范式，并围绕其构建了两种增强策略：

#### （1）**PA / RPRA 范式**（Predict-Answer / Reason-Predict-Reason-Answer）
- **PA（Predict-Answer）**：模型在生成回答前，先预测一个外部 **LLM Judge** 会如何评分其即将生成的回答（如 "great", "ok", "bad"）。
- **RPRA（Reason-Predict-Reason-Answer）**：在 PA 基础上增加多步推理能力，在预测前后都进行 Reasoning，提升判断准确性。
- 这种“预判自身表现”的机制，使模型具备“自我意识”，可作为路由决策依据。

#### （2）两种提升小模型预测能力的方法
- **Report Card（报告卡）系统**：
  - 为每个模型生成一份基于历史表现的“成绩单”，记录其在多个数据集上的典型评分（mode score）。
  - 在推理时将此报告卡作为上下文输入，帮助模型根据当前查询类型判断自身能力。
  - **无需训练**，适用于闭源模型（如 GPT-4）。
- **Supervised Fine-Tuning（监督微调）**：
  - 使用“hindsight trick”构造训练数据：用事后真实的 Judge 评分作为标签，对小模型进行微调，使其学会直接预测 Judge 评分。
  - 微调后模型可在无 Report Card 的情况下完成 PA 范式，**避免处理大量上下文 token**，提升推理效率。

### 相比现有方法的优势
| 维度 | 传统方法 | 本论文方法 |
|------|--------|----------|
| **评估方式** | 依赖人工标注或简单指标（如 BLEU） | 使用 **LLM Judge / Agent Judge**，更灵活、统一、贴近人类偏好 |
| **资源利用** | 固定使用大模型或静态路由 | 动态路由：小模型自信则自答，否则交由大模型，**显著节省计算资源** |
| **小模型可用性** | 小模型因不可靠而受限 | 通过 Report Card 或微调，**大幅提升小模型的可靠性和实用性** |
| **部署灵活性** | 难以适配不同评价标准 | LLM Judge 可通过修改 prompt 支持任意对齐目标（alignment），**高度可定制化** |

---

## 2. 核心实验方法和设置

### 使用的数据集
共使用 5 个多样化数据集，覆盖多种任务类型：
| 数据集 | 任务类型 | 描述 |
|-------|--------|------|
| **MedQA** | 医学诊断 | 来自医学执照考试的多项选择题 |
| **LongFact** | 事实性问答 | 涉及历史、政策、科学等领域的长尾事实问题 |
| **AIME 2024** | 数学竞赛 | 美国数学邀请赛级别的复杂数学题，需多步推理 |
| **MMLU-Pro** | 多学科理解 | 本科水平的跨学科选择题，涵盖哲学、经济、物理等 |
| **SciCode** | 科学编程 | 需结合科学知识编写代码的问题 |

### 实验设置和评估指标

#### 模型集合
共测试 **11 个模型**，参数量从 0.9B 到 120B 不等，包括：
- 小模型：`MobileLLM 0.9B`, `Llama 3.2 1B/3B`
- 中等模型：`Llama 3.1 8B`, `DeepSeek Distilled Qwen 14B/32B`
- 大模型：`Llama 3.3 70B`, `GPT OSS 120B`, `Llama 4 Scout`

#### Judge 设置
- 使用 **Llama 3.3 70B** 作为主 Judge 模型。
- 所有候选模型的回答被**同时提交给 Judge**，进行相对评估（减少偏见）。
- Judge 使用明确定义的 **rubric**（见 Prompt 11）进行打分，维度包括 Accuracy、Relevance、Completeness、Clarity、Instruction Following、Formatting。

#### 评估任务
模型需在不生成最终答案的前提下，预测 Judge 将对其回答给出的评分（"great"/"ok"/"bad"）。

#### 三种评估模式
| 方法 | 描述 |
|-----|------|
| **Zero-Shot** | 仅提供查询，无额外上下文 |
| **In-Context (Report Card)** | 提供该模型的历史表现报告卡作为上下文 |
| **Fine-Tuned** | 对小模型进行监督微调，专门用于预测 Judge 评分 |

#### 评估指标
- **预测准确率（Accuracy）**：模型预测的 Judge 分数是否与真实分数一致。
- 对比零样本、上下文增强、微调三种设置下的表现差异。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）Zero-Shot 表现
- **大模型 & 推理模型**（如 `DSQ32B`, `GPT120B`）在 zero-shot 下已有较好预测能力，尤其在 MedQA 和 LongFact 上。
- **小模型**（如 `M09B`, `L321B`）普遍表现接近随机猜测，甚至更差，表现出严重**校准偏差**（miscalibration），常**过度自信**（overconfident）。

#### （2）Report Card 效果（In-Context Learning）
- 显著提升小模型的预测准确率。
- 平均提升幅度达 **55%**（见 Table 3）。
- 例如：
  - `L321B` 在 AIME 2024 上从 ~20% 提升至 ~90%
  - `L318B` 在 MedQA 上从 ~30% 提升至 ~80%

#### （3）Fine-Tuning 效果
- 微调后的模型（如 `MP09B`, `LP318B`）预测准确率**达到或超过 Report Card 方法**。
- 平均提升 **52%**（见 Table 2）。
- 优势在于**无需携带冗长的 Report Card 上下文**，更适合实际部署。

#### （4）数据集间差异
- 在 **MedQA、LongFact、MMLU-Pro** 上效果最显著。
- 在 **SciCode** 上提升有限，作者推测是因为 Judge 自身对该类任务评估能力不足（Judge 不够专业）。
- 在 **AIME 2024** 上，尽管题目极难，但模型仍能有效判断自己是否“不会做”。

### 与基线方法的对比结果
| 方法 | 准确率（典型值） | 是否需要训练 | 是否引入额外上下文 |
|------|------------------|-------------|------------------|
| Zero-Shot | 低（小模型 ≈ 随机） | 否 | 否 |
| Report Card | 高（可达 80–90%） | 否 | 是（~100–200 tokens） |
| Supervised Fine-Tuning | 高（可达 80–90%） | 是 | 否 |

> ✅ 结论：**Report Card 和 Fine-Tuning 均远优于 Zero-Shot 基线**

### 消融实验结果

#### （1）独立 vs 联合评估 Judge
- 若 Judge 单独评估每个模型输出（非联合），其评分多样性降低，不利于公平比较。
- 联合评估（all-at-once）能更好地区分模型优劣，验证了实验设计合理性（Appendix E）。

#### （2）短版 Report Card（Short Feedback Template）
- 使用简化版报告卡（Prompt 6）导致预测性能明显下降。
- 说明**详细、结构化的反馈信息对性能至关重要**（Appendix H）。

#### （3）更换 Judge 模型（GPT OSS 120B）
- 使用 GPT OSS 120B 作为 Judge 时，其评分更严苛，但 Report Card 方法依然有效。
- 验证了方法对不同 Judge 的鲁棒性（Appendix G）。

#### （4）反转评分标准（Mischievous Rubric）
- 使用反向 rubric（把“正确”定义为“bad”）后，模型预测准确率下降，但仍高于随机水平。
- 说明模型不仅依赖表面规则，还利用了**对任务难度和自身能力的内在理解**（Appendix I）。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **大模型（尤其是具备推理能力的模型）可以在 zero-shot 下较好地预测 Judge 评分**。
2. ❌ **小模型在 zero-shot 下严重校准不良，通常过度自信**。
3. ✅ **Report Card 方法无需训练即可大幅（+55%）提升小模型的自我评估能力**，使其可用于 PA 路由。
4. ✅ **通过 hindsight + supervised fine-tuning，可训练小模型直接预测 Judge 评分，性能媲美 Report Card，且无上下文开销**。
5. 🔍 模型在**更难的任务上反而展现出更强的自我认知能力**，表明“感知难度”是重要信号。
6. 🧠 模型不仅能学习显式规则，还能捕捉隐含的能力边界，具备一定的“元认知”潜力。

### 方法的局限性
| 局限性 | 说明 |
|-------|------|
| **Report Card 引入额外 token 开销** | 虽然输入 token 成本低于输出，但仍增加延迟和成本 |
| **Fine-Tuning 需维护额外权重** | 需保存两套模型（原模型 + 预测头），或牺牲部分原始性能 |
| **依赖高质量 Judge** | 若 Judge 本身不可靠（如 SciCode 场景），预测也难以准确 |
| **单轮交互限制** | 当前框架未考虑多轮对话中的动态能力变化 |
| **泛化性待验证** | 实验集中于特定数据集，真实场景中效果需进一步验证 |

### 未来工作方向
1. **多轮对话中的动态路由**：扩展 PA/RPRA 至 multi-turn 场景，实时调整模型切换策略。
2. **端到端集成预测模块**：将预测头直接嵌入模型内部，共享 prompt 编码，提升效率。
3. **探索更高效的训练方式**：
   - 使用 **Reinforcement Learning from Judge Feedback (RLJF)**
   - 引入 **human-in-the-loop correction** 进行迭代优化
4. **支持任意对齐目标的预测**：利用 LLM Judge 的可塑性，预测模型在安全性、诚实性、无害性等方面的得分。
5. **轻量化 Report Card 表示**：研究如何压缩历史性能信息，降低上下文负担。

---

> **总体评价**：  
> 本文提出了一条通往“**高效、自知、可靠的小模型系统**”的新路径。通过让模型学会预测“别人会怎么评我”，实现了智能路由与资源节约。无论是即插即用的 Report Card，还是高性能的微调方案，都为未来边缘 AI 和混合规模推理系统提供了坚实基础。

</details>

---

### 13. [BEAM: Bi-level Memory-adaptive Algorithmic Evolution for LLM-Powered Heuristic Design](https://arxiv.org/abs/2604.12898)

**Authors**: Chuyang Xiang, Yichen Wei, Jiale Ma, Handing Wang, Junchi Yan  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12898v1  

#### Abstract
Large Language Model-based Hyper Heuristic (LHH) has recently emerged as an efficient way for automatic heuristic design. However, most existing LHHs just perform well in optimizing a single function within a pre-defined solver. Their single-layer evolution makes them not effective enough to write a...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BEAM: Bi-level Memory-adaptive Algorithmic Evolution for LLM-Powered Heuristic Design

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **Language Hyper-Heuristics (LHH)** 方法在自动启发式设计（Automatic Heuristic Design, AHD）中存在两大根本缺陷：
1. **结构与提示策略的局限性**：大多数 LHH 采用单层进化框架，将整个算法视为一个“个体”进行迭代优化，缺乏对高层算法结构（如流程控制、模块组合）的建模能力，导致难以生成复杂完整的求解器。
2. **知识增强（Knowledge Augmentation, KA）缺失或不足**：现有方法要么让 LLM 完全从零开始设计算法（易失败），要么依赖人工设计的模板函数（限制多样性），无法有效利用已有启发式组件。

### 提出的新方法与新思路
作者提出 **BEAM**（**Bi-level Memory-adaptive Algorithmic Evolution**），一种双层记忆自适应的算法演化框架，其核心思想是模仿人类专家的设计范式——将算法设计分解为**高层结构规划**与**底层函数实现**两个层次。

#### 主要创新点：
- **Bi-level Optimization Formulation**：
  - **外层（Exterior Layer）**：使用 **Genetic Algorithm (GA)** 进化算法的高层结构（如主循环、调用顺序、占位符函数 `func_1`, `func_2` 等），不关心具体实现。
  - **内层（Interior Layer）**：使用 **Monte Carlo Tree Search (MCTS)** 为每个结构中的占位函数寻找最优实现方案，通过多轮尝试、修复和校准生成高质量代码。
- **Adaptive Memory (AM)**：
  - 引入一个可复用的函数记忆池，将历史生成的优质函数存储起来，并允许后续的 LLM 直接调用（`import`），避免重复生成冗长代码。
  - 该机制提升了代码生成效率、稳定性和多样性，促进新旧组件的创新组合。
- **Knowledge Augmentation (KA) Pipeline**：
  - 构建两个外部知识库：
    - **HeuBase**：可调用的启发式函数库，包含 pip 可安装库和手写高级组件（如 `KaHIP`, `ARW`）。
    - **KnoBase**：基于任务标签检索并总结的文本型领域知识库。
  - 使 LHH 能够在已有知识基础上构建完整求解器，而非从零开始。

### 相比现有方法的优势
- **更强的表达能力**：双层结构支持生成完整的、复杂的求解器（Entire Algorithm Design），而不仅是单一函数。
- **更高的探索效率**：MCTS 在函数层面进行精细化搜索，GA 在结构层面进行宏观演化，分工明确。
- **更好的稳定性与复用性**：Adaptive Memory 避免了重复劳动，提升了演化过程的鲁棒性。
- **更贴近真实场景**：KA Pipeline 使得评估更有意义，推动 LHH 向实用化迈进。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖了多种优化问题类型，数据集如下：

| 问题类型 | 数据集 | 描述 |
|--------|-------|------|
| **TSP** | TSP-50, TSP-100, TSP-500 | TSPLIB 风格实例，用于测试路径规划 |
| **BPP** | Weibull5k, Weibull10k, Weibull100k | 在线装箱问题，物品大小服从 Weibull 分布 |
| **CAF** | Ackley, Rastrigin, Griewank 等 | 贝叶斯优化中的代价感知采集函数（Cost-aware Acquisition Function） |
| **MIS** | RB-200-300, RB-800-1200, SATLIB | 最大独立集问题，使用随机图模型 |
| **CVRP** | CVRP-100, CVRP-200, CVRP-500 | 容量约束车辆路径问题，坐标均匀采样于单位方格 |
| **BBOB** | Ellipsoidal, Rosenbrock, Sphere 等 | 黑盒优化基准套件，共 5 个标准函数 |
| **PMSP** | Randomly-Generated | 并行机器调度问题，源自 ZeroBubble 文献 |

### 实验设置和评估指标
- **LLM 模型**：主要使用 `DeepSeek-V3` 和 `DeepSeek-R1`，温度设为 0.7 或 1.0。
- **预算控制**：严格控制运行时间或 Token 消耗，确保公平比较（见 Table II）。
- **评估方式**：
  - 所有算法均在相同测试集上运行。
  - 性能指标主要为 **Optimality Gap ↓**（越小越好）或目标值（Objective Value）。
  - 报告多次运行的最佳结果（Best）及平均表现（Average）。
- **硬件环境**：Apple M3 CPU；部分依赖 GPU/CPU 的算法在对应设备上运行。

### 基线方法对比
对比了当前主流的 LHH 方法，包括：
- **ReEvo**, **EoH**, **MCTS-AHD**, **AlphaEvolve**, **FunSearch**, **LLaMEA-HPO** 等。
- 同时与传统 SOTA 求解器对比：
  - **KaMIS**（MIS）
  - **HGS**（CVRP）
  - **EACO-EDM**（TSP）

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 在 CVRP 上的整体提升
- BEAM 在混合算法设计任务中，相比现有 LHH 方法，在所有基准上的**平均最优性差距减少了 37.84%**。
- 具体表现（Table V）：
  - **CVRP-100**: BEAM 达到 `15.57`（Gap: `0.09%`），优于 ReEvo (`0.37%`) 和 EoH (`0.22%`)。
  - **CVRP-500**: BEAM 达到 `37.47`（Gap: `0.86%`），显著领先其他 LHH。

#### ✅ 在 MIS 上超越 SOTA
- 在 `RB-800-1200` 大规模图上，BEAM 设计的求解器性能达到甚至**超过 KaMIS**（SOTA 求解器）。
- 特别是在结合 `KaHIP` 与 `ARW` 的设置下，BEAM 生成的算法实现了 **-0.06% 的负 Gap**，表明其略优。

#### ✅ 在 TSP 上接近最优
- BEAM 生成的 ACO 框架结合 2-opt 局部搜索，在 TSP-500 上达到 `-10.66%` 的相对改进（优于 EACO-EDM）。

#### ✅ 在连续优化（BBOB）上接近 SOTA
- 在 BBOB 基准上，BEAM 的平均 Gap 仅为 **0.007**，远低于 ReEvo (0.957) 和 EoH (6.032)，接近专门优化的 LLaMEA-HPO (0.386)。

| 方法 | BBOB 平均 Gap |
|------|----------------|
| BEAM | **0.007** |
| LLaMEA-HPO | 0.386 |
| ReEvo | 0.957 |
| EoH | 6.032 |

### 消融实验结果（Ablation Study）

#### 🔍 Adaptive Memory (AM) 的作用（Table VII & VIII）
- 移除 AM 后（记为 BE），性能下降明显：
  - TSP 上性能从 `-9.55%` 降至 `-8.12%`
  - CVRP 上 Gap 从 `0.86%` 升至 `0.89%`
- **稳定性提升**：BEAM 的最终 Gap 方差（VAR）为 `0.01`，远低于 BE 的 `0.19`，说明 AM 显著增强了演化稳定性。

#### 🔍 教育方法比较（MCTS vs One-Shot）
- 内层使用 **MCTS** 搜索函数实现，效果普遍优于一次性填充（One-Shot）：
  - TSP: `3.05%` vs `3.63%`
  - CVRP: `0.86%` vs `1.07%`
- 表明 MCTS 能更有效地探索函数空间。

#### 🔍 模型泛化性测试
- 使用较小模型（如 GPT-3.5 Turbo）仍能生成高质量代码，Gap 增幅有限（TSP-500 上仅增加 `6.60%`），说明 **BEAM 对 LLM 尺寸依赖较低**，其架构本身是成功的关键。

---

## 4. 关键结论和发现

### 主要发现
1. **双层结构是关键**：将算法设计分解为“结构演化 + 函数实现”两阶段，显著提升了 LLM 在复杂代码生成任务中的成功率和质量。
2. **BEAM 是当前最强大的 LHH 框架之一**：在多个组合优化与连续优化任务上，不仅大幅超越现有 LHH 方法，甚至能设计出媲美或超越 SOTA 专用求解器的算法。
3. **Adaptive Memory 提升效率与稳定性**：通过记忆和复用优质函数，减少了冗余生成，提高了演化过程的收敛速度和鲁棒性。
4. **Knowledge Augmentation 至关重要**：没有外部知识支撑的“从零生成”模式不可靠；KA Pipeline 为 LHH 提供了坚实的基础，使其更具实用性。
5. **创新常来自已有组件的新组合**：BEAM 发现的高性能算法往往并非完全原创，而是巧妙地重组了 `KaHIP`, `ARW`, `Split`, `LS` 等已有技术。

### 方法的局限性
- **计算成本较高**：由于双层结构和 MCTS 搜索，首次生成有效个体所需的 Token 消耗较大（见 Fig. 8）。
- **对初始知识库依赖较强**：若 HeuBase 中缺少关键组件，可能限制搜索空间。
- **仍可能出现过度复杂化**：对于简单任务，BEAM 有时会生成不必要的复杂代码（如过多元函数），影响可读性。

### 未来工作方向
- 将 BEAM 扩展到更复杂的领域，如多目标优化、动态优化等。
- 探索更高效的 KA 获取方式，例如结合 RAG 动态检索。
- 研究如何自动扩展 HeuBase，形成“自我进化的启发式仓库”。
- 降低计算开销，例如引入 early stopping 或并行化 MCTS。

--- 

> **总结**：  
> BEAM 代表了 LLM-powered Heuristic Design 的一个重要跃迁——它不再只是“微调一个函数”，而是真正迈向“全自动设计完整求解器”的时代。其双层架构、记忆机制与知识增强理念，为未来智能算法自动化开辟了新的研究范式。

</details>

---

### 14. [Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score](https://arxiv.org/abs/2604.12196)

**Authors**: Manh Nguyen, Sunil Gupta, Hung Le  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12196v1  

#### Abstract
Large language models (LLMs) frequently generate multiple candidate responses for a given prompt, yet selecting the most reliable one remains challenging, especially when correctness diverges from surface-level majority agreement. Existing approaches, such as self-consistency, rely on discrete votin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLMs）在生成多个候选响应时，如何从 `Best-of-N` 的采样中选择最可靠的答案是一个关键挑战。传统方法如 **self-consistency**（基于多数投票）存在以下局限：
- 忽略语义相似性，仅依赖表面形式的一致性；
- 难以识别高质量但出现频率低的“少数正确答案”；
- 概率型方法（如 NLL、ANLL）往往无法捕捉答案间的语义关系，且对黑盒 API 不友好。

### 🆕 提出的新方法：Radial Consensus Score (RCS)
作者提出了一种简单、高效、无需训练的几何共识方法——**Radial Consensus Score (RCS)**，其核心思想是：
- 将每个候选答案通过 **sentence embedding model** 映射为向量；
- 计算这些嵌入的加权 **Fréchet mean** 作为语义中心（semantic center）；
- 根据各候选答案到该中心的 **radial distance**（径向距离）进行排序，距离越小表示与群体语义共识越一致；
- 最终选择距离最小的候选作为最优答案。

#### 支持多种权重方案（灵活集成信号）：
| Variant | Distribution P | 特点 |
|--------|----------------|------|
| `RCS_uni` | Uniform | 黑盒可用，纯几何共识 |
| `RCS_freq` | Frequency-based | 融合答案频次信号 |
| `RCS_prob` | Probability-weighted | 利用生成概率（白盒） |

### 🔍 相比现有方法的优势
- **超越 majority voting**：能恢复语义上正确但非主流的答案；
- **融合多源信号**：可结合 frequency 和 probability，提升鲁棒性；
- **计算轻量、模型无关**：复杂度为 $O(Nd)$，适用于 black-box 设置；
- **即插即用**：可直接替代 multi-agent debate 中的投票机制；
- **理论基础强**：基于 Fréchet mean 和 medoid 的闭式解，具有数学合理性。

---

## 2. 核心实验方法和设置

### 📚 数据集
共使用 **7 个基准数据集**，覆盖短问答与长推理任务：

| 类型 | 数据集 | 说明 |
|------|-------|------|
| **Short-form QA** | SciQ, GPQA Diamond | 开放式科学问答 |
| **Math Reasoning** | Arithmetics, GSM8K, AIME25 | 数学应用题与竞赛题 |
| **Long-form MCQA** | MMLU Formal Logic, MMLU-Pro | 形式逻辑与多任务理解 |

### 🧪 实验设置
- **模型**：5 个开源 LLMs  
  → Qwen2.5-3B/7B, Llama3.2-3B, Llama3.1-8B, Gemma2-9B
- **采样策略**：multinomial decoding, temperature=1, N ∈ {5,10,20,40}
- **Embedding 模型**：`all-MiniLM-L6-v2`（默认）
- **评估协议**：
  - 长文本任务：exact match
  - 短文本任务：ROUGE-L F1 > 0.3 视为正确
- **报告方式**：mean ± std over 3 seeds

### ⚔️ 基线方法对比
| 方法 | 类型 | 是否黑盒友好 |
|------|------|-------------|
| **Greedy / Oracle** | 参考基线 | —— |
| **NLL / ANLL** | 概率法（token-level likelihood） | ❌（需内部输出） |
| **Self-Consistency (SC)** | 多数投票 | ✅ |
| **Self-Certainty (CE)** | 结合频率与置信度的 Borda 投票 | ✅ |
| **RCS_medoid** | 离散 medoid 版本 | ✅（uniform only） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（N=10 平均准确率）

| 方法 | 平均准确率（Across 5 models & 5 datasets） |
|------|------------------------------------------|
| SC | ~53.5% |
| CE | ~53.6% |
| **RCS_uni** | **~57.0%** |
| **RCS_prob** | **~57.2%** |

👉 **RCS 超出最强基线 2–7%**，尤其在高采样预算下优势更明显。

### 🔺 与基线方法对比结果
- 在所有模型和任务上，`RCS_uni` 和 `RCS_prob` 均显著优于 SC 和 CE（p < 0.05）；
- 当 N 增大时（如 N=40），RCS 的增益进一步扩大（最高 +7%），说明其在噪声环境下更具扩展性；
- 在 **Formal Logic** 等语义多样性高的任务中表现尤为突出；
- 在 **multi-agent debate** 场景中，RCS 可作为 drop-in 替代 majority voting，带来 1–2% 提升。

### 🔍 消融实验结果

| 实验方向 | 发现 |
|--------|------|
| **Clean-answer setting**（去除空响应） | RCS 仍最优，但在噪声少时差距缩小 → 表明 RCS 对噪声更鲁棒 |
| **Black-box setting**（Cohere API） | 在 MMLU-Pro 上，`RCS_base` 和 `RCS_medoid` 明显优于 SC → 更适应多样且嘈杂的回答分布 |
| **Embedding model 容量影响** | 使用 all-MiniLM-L6-v2 / mpnet-base / roberta-large 几乎无差异 → 方法对 embedding 质量不敏感 |
| **Correctness threshold 敏感性测试** | 在 ROUGE-F1 阈值 0.1–0.5 区间内性能稳定 → 方法鲁棒性强 |
| **Full trajectory vs Final answer embeddings** | 使用完整生成路径嵌入会导致性能下降 → 存在“representation collapse”，验证了只用最终答案的合理性 |
| **Computational Complexity** |  
| 方法 | 复杂度 |
|------|--------|
| SC | O(N) |
| CE | O(NL\|V\|) |
| RCS_medoid | O(N²d) |
| **RCS_uni/freq** | **O(Nd)** ← 最佳平衡点 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **几何共识优于表面一致性**：  
   通过 embedding 空间中的 Fréchet mean 建模语义共识，能有效识别语义一致但形式不同的正确答案。
   
2. **RCS 是 majority voting 的自然推广**：  
   当所有答案语义接近时，RCS 自动退化为 frequency/probability 加权选择，保留了 self-consistency 的优点。

3. **随 N 增大而持续受益**：  
   与 SC 易被高频错误答案主导不同，RCS 能在更大采样集中发现稀疏但正确的语义簇。

4. **适用于黑盒场景**：  
   `RCS_uni` 和 `RCS_freq` 仅依赖输出文本和 embedding，完全兼容商业 API。

5. **即插即用能力强**：  
   成功应用于 multi-agent debate，提升决策稳定性。

### ⚠️ 局限性
- 依赖 sentence embedding model 的质量（尽管实验显示对主流模型不敏感）；
- `RCS_medoid` 具有 $O(N^2)$ 复杂度，不适合大规模 N；
- 对纯数值型答案（如 Arithmetics）增益有限，因语义空间区分度较低；
- `RCS_prob` 需要 calibrated probability 输出，在部分模型上可能不准。

### 🔮 未来工作方向
- 探索更高效的近似 medoid 计算方法以支持大 N；
- 扩展至多跳推理路径的分段共识建模；
- 将 RCS 与 verifier 或 reward model 结合用于迭代优化；
- 应用于 code generation、fact verification 等下游任务；
- 研究动态调整权重分布 $P$ 的自适应机制。

---

> **总结一句话**：  
> **Radial Consensus Score (RCS)** 提供了一个简洁、通用、高效的 Best-of-N 答案选择框架，通过将语义共识建模为嵌入空间中的几何中心，显著超越了传统的 majority voting 和概率方法，特别是在高噪声、高多样性或黑盒环境中展现出强大潜力。

</details>

---

### 15. [Latent-Condensed Transformer for Efficient Long Context Modeling](https://arxiv.org/abs/2604.12452)

**Authors**: Zeng You, Yaofo Chen, Qiuwu Chen, Ying Sun, Shuhai Zhang, Yingjian Li, Yaowei Wang, Mingkui Tan  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12452v1  

#### Abstract
Large language models (LLMs) face significant challenges in processing long contexts due to the linear growth of the key-value (KV) cache and quadratic complexity of self-attention. Existing approaches address these bottlenecks separately: Multi-head Latent Attention (MLA) reduces the KV cache by pr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLMs）在处理长上下文时面临两大瓶颈：
- **Key-Value (KV) Cache 的线性增长**：解码过程中缓存占用内存随序列长度线性增加。
- **Self-Attention 的二次计算复杂度**：注意力机制的时间复杂度为 $O(L^2)$，限制了长序列推理效率。

现有方法如 **Multi-head Latent Attention (MLA)** 能压缩 KV 缓存，但仍保留全量注意力计算；而稀疏注意力（Sparse Attention）虽能降低计算量，却无法直接作用于 MLA 的低维潜在空间，导致必须重建高维表示，丧失效率优势。

### **提出的新方法：Latent-Condensed Attention (LCA)**
本文提出 **Latent-Condensed Attention (LCA)**，一种可在 MLA 的**低维潜在空间中直接进行上下文压缩**的高效注意力机制。其核心思想是：
- 在 MLA 的 latent space 中对冗余上下文进行**结构化压缩**，而非在重建后的高维空间上做稀疏化。
- 显式区分语义信息（semantic latent vectors）和位置信息（positional keys），采用不同策略处理：
  - **语义向量**：通过 query-aware weighted pooling 进行加权聚合，保留所有 token 的语义贡献。
  - **位置编码**：通过 max-selection 选择每组中重要性最高的 token 作为 anchor，避免混合非线性 RoPE 信号。

该方法无需引入额外参数，即可同时减少 KV Cache 大小和注意力计算量。

### **相比现有方法的优势**
| 特性 | MLA | Sparse Attention (e.g., FlexPrefill, MInference) | LCA |
|------|-----|-----------------------------------------------|-----|
| KV Cache 压缩 | ✅（通过 latent projection） | ❌（需重建 KV） | ✅✅（原生 latent space 压缩） |
| 计算复杂度优化 | ❌（仍为 $O(L^2)$） | ✅（稀疏化） | ✅✅（latent 内聚合并压缩） |
| 是否引入新参数 | ✅（无） | ✅（通常无） | ✅（无） |
| 可扩展性 | — | 依赖高维 KV | ✅✅（architecture-agnostic，支持 GQA 等） |

> ✅ LCA 首次实现了在 latent space 中的**原生存量压缩**，联合优化了内存与计算开销。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **长上下文任务**：
  - **LongBench-E**：多语言、多任务长文本理解基准，涵盖问答、摘要、代码等共 21 项任务，最大上下文达 128K。
  - **RULER**：合成型长上下文评测集，测试检索、多跳推理等能力，覆盖 4K–128K 不同长度。
- **短上下文任务**（验证通用性）：
  - **MMLU**：跨学科知识理解。
  - **GSM8K**：小学数学题，考验多步推理。
  - **MBPP**：编程生成任务。

### **实验设置与评估指标**
- **主干模型**：
  - 主要基于 **DeepSeek-V2-Lite (16B)**，首个公开发布的 MLA 架构模型。
  - 后续也在 **MiniCPM3-4B** 和 **Distill-Qwen-7B (GQA)** 上验证泛化性。
- **实现细节**：
  - 使用 **Triton 自定义 kernel** 实现高效推理。
  - 微调 1000 步于 **SlimPajama 数据集**（最长 64K）。
  - 组大小 `g=16`，局部窗口 `w=1024`。
- **硬件平台**：8×H200 GPUs。
- **评估指标**：
  - **准确率（Accuracy / Score）**
  - **首 token 延迟（First-Token Latency, FTL）**
  - **KV Cache 内存占用（GPU Memory Footprint）**
  - **Prefilling Speedup**

### **基线方法对比**
| 方法 | 类型 | 是否适配 MLA |
|------|------|--------------|
| **MLA (Baseline)** | Latent Projection | 原始架构 |
| **MInference (Jiang et al., 2024)** | 动态稀疏注意力 | 适配至重建 KV |
| **FlexPrefill (Lai et al., 2025)** | 查询感知稀疏模式 | 适配至重建 KV |
| **KDA (Kimi et al., 2025b)** | 门控线性注意力变体 | 从头训练困难，本文微调对比 |

> 所有对比方法均被调整以兼容 MLA 输出，确保公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **在 LongBench-E 上的表现（128K 上 FTL 测试）**
| 方法 | Avg. Score | FTL (s) | Speedup |
|------|-----------|--------|--------|
| MLA | 29.51 | 3.20 | 1.0× |
| MInference | 19.71 | 2.99 | 1.1× |
| FlexPrefill | 21.05 | 2.51 | 1.3× |
| KDA | 27.15 | 2.47 | 1.3× |
| **LCA (Ours)** | **29.09** | **1.80** | **1.8×** |

> ✅ LCA 几乎保持原始 MLA 性能（仅下降 0.42 分），但获得 **1.8× 推理加速**。

#### **在 RULER 上的表现（128K 上 FTL 测试）**
| 方法 | RULER@128K | FTL (s) | Speedup |
|------|------------|--------|--------|
| MLA | 23.96 | 10.78 | 1.0× |
| MInference | 4.34 | 5.66 | 1.9× |
| FlexPrefill | 7.19 | 5.38 | 2.0× |
| KDA | 22.22 | 4.96 | 2.2× |
| **LCA (Ours)** | **24.38** | **4.40** | **2.5×** |

> ✅ LCA 在极端长度下反超 MLA，达到 **2.5× 预填充加速**，且性能更优。

#### **KV Cache 压缩效果**
- 在 128K 上：
  - KV Cache 从 **10.13 GB → 0.71 GB**，**减少 90%**。
  - 其他稀疏方法（如 MInference/FlexPrefill）**不减少 KV Cache**。
- 在 Distill-Qwen-7B 上甚至实现 **93% KV Cache 下降**。

#### **短上下文任务表现（无退化）**
| 方法 | MMLU | GSM8K | MBPP |
|------|------|-------|------|
| MLA | 57.12 | 41.47 | 54.09 |
| **LCA (Ours)** | **57.04** | **41.17** | **53.31** |

> ✅ 在标准任务上几乎无损，说明 LCA 不损害基础建模能力。

### **消融实验结果**

#### **Pooling 策略对比（Table 5）**
| Semantic Pooling \ Positional | MaxPool | MeanPool | **WeightedPool** |
|-------------------------------|--------|----------|------------------|
| MaxSel                        | 28.89  | 28.84    | **29.09**        |

> ✅ **Weighted Pooling + MaxSel** 最优，证明 query-aware 加权优于固定池化。

#### **组大小 `g` 影响（Figure 4）**
- `g=4`: 更精细压缩，性能更高但延迟上升。
- 默认 `g=16` 平衡性能与效率。

#### **窗口大小 `w` 影响（Figure 5）**
- `w=1024` 是默认折中点。
- 更大 `w` 提升长程任务精度（如 RULER@128K 提升至 25.53），但成本更高。

#### **查询数量影响（Table 8）**
- 使用最后 `g=16` 个 query 计算 summary query 效果最佳。
- 更多 query 收益有限，表明局部焦点已足够。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Latent Space 可原生压缩**：首次证明可在 MLA 的低维 latent space 中安全地进行上下文压缩，无需重建高维 KV。
2. **双路径设计至关重要**：
   - 语义信息适合加权聚合（weighted pooling）。
   - 位置信息需硬选择（max-selection）以保护 RoPE 结构。
3. **理论误差有界**：证明 LCA 的近似误差与上下文长度无关，仅取决于组内偏差。
4. **极致效率提升**：
   - 最高 **2.5× Prefilling Speedup**
   - **90% KV Cache Reduction**
   - 支持百万 token 序列（1M with Triton kernel）
5. **广泛适用性**：
   - 可迁移至 **MiniCPM3-4B (MLA)** 和 **Distill-Qwen-7B (GQA)**。
   - 在 GQA 上仍获 **3.25× 加速 + 93% KV 下降**。

### **局限性**
1. **工程部署门槛较高**：需要自定义 Triton kernel 才能发挥全部性能，普通框架难以高效实现。
2. **低精度支持不足**：当前实现针对 bfloat16/float16，未探索 int8 等量化格式下的优化。
3. **极端检索任务轻微退化**：
   - 在 multi-document QA 等需精确定位的任务上略有下降（11.15 → 8.90）。
   - 因为 condensation 会弱化极少数关键 token 的信号强度。

### **未来工作方向**
- 设计 **自适应 group/window 大小机制**，根据输入动态调整压缩强度。
- 探索 **更低精度下的 kernel 优化**，推动端侧部署。
- 将 LCA 应用于 **从头预训练（training from scratch）**，进一步释放潜力（已有初步实验证明可行）。
- 结合 **token pruning 或 early exiting**，构建多层次高效推理 pipeline。

---

> 📌 **总结一句话**：  
> **LCA 开创性地在 latent space 中实现了“语义聚合 + 位置锚定”的双路径压缩，在几乎不损失性能的前提下，达成高达 2.5× 加速与 90% KV 缓存缩减，为长上下文 LLM 提供了一种高效、通用且理论可靠的解决方案。**

</details>

---

### 16. [Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Sequence-Level Likelihood](https://arxiv.org/abs/2604.12736)

**Authors**: Xingyu Lin, Yilin Wen, Du Su, Jinchang Hou, En Wang, Wenbin Liu, Chenfu Bao, Zhonghou Lv  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12736v1  

#### Abstract
Group Relative Policy Optimization (GRPO) has significantly advanced the reasoning ability of large language models (LLMs), particularly in their mathemat ical reasoning performance. However, GRPO and related entropy regularization methods still struggle with token-level sparse-rewards, which is an ...

---

### 17. [PubSwap: Public-Data Off-Policy Coordination for Federated RLVR](https://arxiv.org/abs/2604.12160)

**Authors**: Anupam Nayak, Baris Askin, Muhammed Ustaomeroglu, Carlee Joe-Wong, Gauri Joshi  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12160v1  

#### Abstract
Reasoning post-training with reinforcement learning from verifiable rewards (RLVR) is typically studied in centralized settings, yet many realistic applications involve decentralized private data distributed across organizations. Federated training is a natural solution, but scaling RLVR in this reg...

---

### 18. [PrivEraserVerify: Efficient, Private, and Verifiable Federated Unlearning](https://arxiv.org/abs/2604.12348)

**Authors**: Parthaw Goswami, Md Khairul Islam, Ashfak Yeafi  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.12348v1  

#### Abstract
Federated learning (FL) enables collaborative model training without sharing raw data, offering a promising path toward privacy preserving artificial intelligence. However, FL models may still memorize sensitive information from participants, conflicting with the right to be forgotten (RTBF). To mee...

---

### 19. [LLM-HYPER: Generative CTR Modeling for Cold-Start Ad Personalization via LLM-Based Hypernetworks](https://arxiv.org/abs/2604.12096)

**Authors**: Luyi Ma, Wanjia Sherry Zhang, Zezhong Fan, Shubham Thakur, Kai Zhao, Kehui Yao, Ayush Agarwal, Rahul Iyer, Jason Cho, Jianpeng Xu, Evren Korpeoglu, Sushant Kumar, Kannan Achan  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12096v1  

#### Abstract
On online advertising platforms, newly introduced promotional ads face the cold-start problem, as they lack sufficient user feedback for model training. In this work, we propose LLM-HYPER, a novel framework that treats large language models (LLMs) as hypernetworks to directly generate the parameters...

---

### 20. [Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization](https://arxiv.org/abs/2604.12290)

**Authors**: Yizhe Chi, Deyao Hong, Dapeng Jiang, Tianwei Luo, Kaisen Yang, Boshi Zhang, Zhe Cao, Xiaoyan Fan, Bingxiang He, Han Hao, Weiyang Jin, Dianqiao Lei, Qingle Liu, Houde Qian, Bowen Wang, Situ Wang, Youjie Zheng, Yifan Zhou, Calvin Xiao, Eren Cai, Qinhuai Na  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12290v1  

#### Abstract
Current LLM agent benchmarks, which predominantly focus on binary pass/fail tasks such as code generation or search-based question answering, often neglect the value of real-world engineering that is often captured through the iterative optimization of feasible designs. To this end, we introduce Fro...

---

### 21. [KnowRL: Boosting LLM Reasoning via Reinforcement Learning with Minimal-Sufficient Knowledge Guidance](https://arxiv.org/abs/2604.12627)

**Authors**: Linhao Yu, Tianmeng Yang, Siyu Ding, Renren Jin, Naibin Gu, Xiangzhao Hao, Shuaiyi Nie, Deyi Xiong, Weichong Yin, Yu Sun, Hua Wu  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12627v1  

#### Abstract
RLVR improves reasoning in large language models, but its effectiveness is often limited by severe reward sparsity on hard problems. Recent hint-based RL methods mitigate sparsity by injecting partial solutions or abstract templates, yet they typically scale guidance by adding more tokens, which int...

---

### 22. [Think Through Uncertainty: Improving Long-Form Generation Factuality via Reasoning Calibration](https://arxiv.org/abs/2604.12046)

**Authors**: Xin Liu, Lu Wang  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12046v1  

#### Abstract
Large language models (LLMs) often hallucinate in long-form generation. Existing approaches mainly improve factuality through post-hoc revision or reinforcement learning (RL) with correctness-based rewards, but they do not teach the model to estimate which parts of its generation are reliable. As a ...

---

### 23. [Accelerating Speculative Decoding with Block Diffusion Draft Trees](https://arxiv.org/abs/2604.12989)

**Authors**: Liran Ringel, Yaniv Romano  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12989v1  

#### Abstract
Speculative decoding accelerates autoregressive language models by using a lightweight drafter to propose multiple future tokens, which the target model then verifies in parallel. DFlash shows that a block diffusion drafter can generate an entire draft block in a single forward pass and achieve stat...

---

### 24. [A Periodic Space of Distributed Computing: Vision & Framework](https://arxiv.org/abs/2604.12259)

**Authors**: Mohsen Amini Salehi, Adel N. Tousi, Hai Duc Nguyen, Murtaza Rangwala, Omar Rana, Tevfik Kosar, Valeria Cardellini, Rajkumar Buyya  
**Category**: cs.DC  
**Published**: 2026-04-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12259v1  

#### Abstract
Advances in networking and computing technologies throughout the early decades of the 21st century have transformed long-standing dreams of pervasive communication and computation into reality. These technologies now form a rapidly evolving and increasingly complex global infrastructure that will un...

---

### 25. [SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling](https://arxiv.org/abs/2604.12110)

**Authors**: Zikun Liu, Liang Luo, Qianru Li, Zhengyu Zhang, Wei Ling, Jingyi Shen, Zeliang Chen, Yaning Huang, Jingxian Huang, Abdallah Aboelela, Chonglin Sun, Feifan Gu, Fenggang Wu, Hang Qu, Huayu Li, Jill Pan, Kaidi Pei, Laming Chen, Longhao Jin, Qin Huang, Tongyi Tang, Varna Puvvada, Wenlin Chen, Xiaohan Wei, Xu Cao, Yantao Yao, Yuan Jin, Yunchen Pu, Yuxin Chen, Zijian Shen, Zhengkai Zhang, Dong Liang, Ellie Wen  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12110v1  

#### Abstract
Recent advances in recommendation scaling laws have led to foundation models of unprecedented complexity. While these models offer superior performance, their computational demands make real-time serving impractical, often forcing practitioners to rely on knowledge distillation-compromising serving ...

---

### 26. [Rethinking the Personalized Relaxed Initialization in the Federated Learning: Consistency and Generalization](https://arxiv.org/abs/2604.12768)

**Authors**: Li Shen, Yan Sun, Dacheng Tao  
**Category**: cs.LG  
**Published**: 2026-04-15  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.12768v1  

#### Abstract
Federated learning (FL) is a distributed paradigm that coordinates massive local clients to collaboratively train a global model via stage-wise local training processes on the heterogeneous dataset. Previous works have implicitly studied that FL suffers from the ``client-drift'' problem, which is ca...

---

### 27. [A hierarchical spatial-aware algorithm with efficient reinforcement learning for human-robot task planning and allocation in production](https://arxiv.org/abs/2604.12669)

**Authors**: Jintao Xue, Xiao Li, Nianmin Zhang  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.12669v1  

#### Abstract
In advanced manufacturing systems, humans and robots collaborate to conduct the production process. Effective task planning and allocation (TPA) is crucial for achieving high production efficiency, yet it remains challenging in complex and dynamic manufacturing environments. The dynamic nature of hu...

---

### 28. [Cycle-Consistent Search: Question Reconstructability as a Proxy Reward for Search Agent Training](https://arxiv.org/abs/2604.12967)

**Authors**: Sohyun An (Meta Superintelligence Labs, UCLA), Shuibenyang Yuan (Meta Superintelligence Labs), Hayeon Lee (Meta Superintelligence Labs), Cho-Jui Hsieh (UCLA), Alexander Min (Meta Superintelligence Labs)  
**Category**: cs.AI  
**Published**: 2026-04-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.12967v1  

#### Abstract
Reinforcement Learning (RL) has shown strong potential for optimizing search agents in complex information retrieval tasks. However, existing approaches predominantly rely on gold supervision, such as ground-truth answers, which is difficult to scale. To address this limitation, we propose Cycle-Con...

---

### 29. [Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Reasoning](https://arxiv.org/abs/2604.12282)

**Authors**: Houxing Ren, Mingjie Zhan, Zimu Lu, Ke Wang, Yunqiao Yang, Haotian Hou, Hongsheng Li  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.12282v1  

#### Abstract
Spreadsheets are central to real-world applications such as enterprise reporting, auditing, and scientific data management. Despite their ubiquity, existing large language model based approaches typically treat tables as plain text, overlooking critical layout cues and visual semantics. Moreover, re...

---

### 30. [Transforming External Knowledge into Triplets for Enhanced Retrieval in RAG of LLMs](https://arxiv.org/abs/2604.12610)

**Authors**: Xudong Wang, Chaoning Zhang, Qigan Sun, Zhenzhen Huang, Chang Lu, Sheng Zheng, Zeyu Ma, Caiyan Qin, Yang Yang, Hengtao Shen  
**Category**: cs.CL  
**Published**: 2026-04-15  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.12610v1  

#### Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucination in large language models (LLMs) by incorporating external knowledge during generation. However, the effectiveness of RAG depends not only on the design of the retriever and the capacity of the underlying model, but also on how retrieved ev...

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
