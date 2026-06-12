# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-06-12 09:52:30 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/abs/2606.12556)

**Authors**: Hakbeom Jang, Younghoon Min, Sunwoong Kim, Taeyoung Ahn, Hanyee Kim, Youngpyo Joo, Hoshik Kim, Jongryool Kim  
**Category**: cs.DC  
**Published**: 2026-06-12  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2606.12556v1  

#### Abstract
The rapid shift toward agentic and long-context workloads in Large Language Models (LLMs) is pushing the industry beyond the capacity of individual servers toward disaggregated shared storage to handle TB-scale context states. This movement has led to the emergence of specialized shared context laye...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories**

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
随着大语言模型（LLMs）向**长上下文（long-context）** 和**智能体式（agentic）工作流**演进，推理系统面临两大挑战：
- **内存容量瓶颈**：模型权重（如 OPT-175B 需要 325 GB FP16）和 KV Cache 足迹迅速增长至 TB 级别，远超单台服务器本地内存（GPU/HBM、Host DRAM）容量。
- **共享状态管理低效**：传统基于本地 SSD 或 DPU-JBOF 的 KV Cache 存储方案存在**碎片化、复用率低、软件开销高**等问题，难以支持跨节点、多轮次的上下文共享。

现有解决方案（如 DPU-based JBOF + NVMe-oF）虽能提供大容量存储，但依赖昂贵的 DPU 和复杂的软件栈，且无法提供**字节可寻址（byte-addressable）** 的低延迟访问能力。

---

### 提出了什么新方法或新思路
本文提出 **ITME（Inference Tiered Memory Expansion）**，一种基于 **CXL-Hybrid Memory** 的分层内存扩展架构，核心创新如下：

#### ✅ **ITME 架构设计**
- 利用 **CXL-Hybrid Memory** 设备构建一个远程的、TB 级、**字节可寻址的共享内存池（Tier T3.5）**，作为 GPU 服务器的直接内存扩展。
- 该设备由 **高性能 DRAM 缓存 + 大容量 NVMe SSD** 组成，通过 CXL 协议暴露为远程 NUMA 节点，支持标准 RDMA 访问。

#### ✅ **硬件级预取机制（Hardware-managed Prefetching）**
- 在 CXL-Hybrid 内部集成 **硬件级预取器（Hardware Prefetcher）**，自动将 SSD 数据预加载到 DRAM 缓存中，隐藏后端存储延迟。
- 支持 **用户级预取 API（chm_prefetch）**，允许 LLM 推理框架（如 vLLM）主动触发预取，实现应用感知的数据调度。

#### ✅ **多级 DMA 流水线预取（Multi-tier DMA-based Prefetching）**
- 构建从 **CXL-Hybrid → Host Memory → GPU Memory** 的端到端 DMA 流水线，利用 LLM 推理中**模型权重和前缀 KV Cache 的确定性访问模式**，提前发起数据迁移。
- 实现计算与通信重叠，有效掩盖网络和存储延迟。

#### ✅ **读优先 I/O 调度策略（Read-Priority I/O Scheduling）**
- 为避免后台写入（如 KV 块持久化）阻塞关键读请求，引入读优先调度。
- 在解码阶段插入受控的小批量写操作，确保下一轮“prefill”阶段前 SSD 恢复干净状态，保障峰值读带宽。

---

### 相比现有方法的优势
| 方面 | 现有方法（DPU-JBOF + NVMe-oF） | ITME |
|------|-------------------------------|------|
| **访问接口** | 块设备/文件系统，非字节可寻址 | 字节可寻址，支持直接 load/store |
| **硬件依赖** | 依赖昂贵 DPU（如 BlueField） | 使用低成本 RNIC，无需专用处理器 |
| **软件栈复杂度** | 需完整 NVMe-oF Target 软件栈 | 硬件直通，绕过内核协议栈 |
| **预取能力** | 软件级，开销高 | 硬件级 + 用户 API，高效可控 |
| **成本效率** | 高（DPU 成本占比大） | 更优（利用廉价 NAND + CXL 扩展） |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **ShareGPT Dataset**：用于构建多轮对话场景，模拟真实 LLM 推理负载，每会话最多 5 轮，最小 2000 tokens。
- **Mooncake Dataset**：专门用于分析 KV Cache 累积行为和预取效果的基准数据集。

### 实验设置
- **硬件平台**：
  - **GPU Server**：Dell R770，双路 Intel Xeon 6730，256GB DDR5，NVIDIA A100 80GB。
  - **Remote Server**：同型号服务器，搭载 **SK hynix CMM（CXL Memory Module）32GB DRAM + 2× KIOXIA PCIe Gen5 NVMe SSD（共 2TB）**。
  - **互联**：双 100Gbps RDMA（ConnectX-6），总带宽 200Gbps。
- **原型验证**：基于 **Intel Agilex 7 FPGA** 实现功能原型，验证硬件逻辑可行性。

### 评估指标
- **吞吐量（Throughput）**：tokens/sec
- **首 Token 延迟（TTFT, Time to First Token）**
- **KV Cache 命中率（Hit Rate）**
- **端到端推理速度提升（Speedup）**
- **带宽利用率（GB/s）**

### 基线方法对比
1. **Recomputation Only**：不缓存 KV，每次重新计算。
2. **CPU Offloading (with NVMe-oF)**：将 KV Cache 卸载至本地 NVMe-oF 存储，无预取优化。
3. **All-caching in GPU Memory**：理想情况，所有 KV Cache 存于 GPU HBM。
4. **Local NVMe-oF Baseline**：本地 NVMe-oF 存储，作为性能上限参考。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 场景 | ITME 性能表现 |
|------|------|----------------|
| **Llama-3.1 8B & 70B** | 多轮推理（Turn 3–5） | 较 Recomputation 提速 **1.81×** |
| **Llama-3.1 70B** | 扩展至 35 轮对话 | 较 CPU-offload 基线提升 **35.7% 吞吐量** |
| **Weight Prefetching** | 仅预取权重 | 与主机内存基线差距 < 5%，有效隐藏网络延迟 |
| **CXL-Hybrid 带宽** | FPGA 原型实测 | 读达 **18 GB/s**，写达 **12 GB/s**（DRAM hit） |

---

### 与基线方法的对比结果
#### 🔹 vs CPU Offloading（图10）
- 初始阶段（Turn 1–9）：性能接近，因工作集仍在 Host Memory。
- 中期（Turn 10–20）：ITME 开始出现轻微波动，因 I/O 竞争导致部分块未及时加载。
- 后期（Turn >21）：CPU-offload 耗尽 128GB 内存，命中率归零，退化为 recompute；而 ITME 凭借 TB 级扩展持续服务，**最高提升 35.7% 吞吐**。

#### 🔹 vs Local NVMe-oF（图11）
- 初始阶段性能一致。
- 从 Turn 5 起，NVMe-oF 因写入阻塞读取，性能急剧下降至 recompute 水平。
- ITME 通过 **读优先调度 + 预取流水线** 维持稳定读取，显著优于基线。

#### 🔹 vs 理想 GPU 缓存（All-caching）
- All-caching 是理论最优（本地 HBM 访问），在 Llama-3.1 70B 上达到 3.02× 加速。
- ITME 达到 **1.81× 加速**，表明其在远程访问条件下仍能实现高效推理。

---

### 消融实验结果
#### 🔹 权重预取影响（图12a）
- 仅启用权重预取时，ITME 已能接近主机内存基线性能（<5% 差距），证明其能有效隐藏存储与网络延迟。

#### 🔹 DRAM 缓存大小敏感性（图12b）
- 即使 CXL-Hybrid 的 DRAM 缓存仅为 **4GB**，也能维持 92% 的基准性能。
- 表明即使小缓存配合预取机制，仍可支撑大规模模型推理。

#### 🔹 FPGA 原型性能（图14）
- FPGA 实现带宽略低于 CMM 平台（约低 20–25%），但已验证硬件可行性。
- 当前瓶颈在于元数据更新延迟（3-cycle 锁机制），正在优化以逼近理论峰值 23.3 GB/s。

---

## 4. 关键结论和发现

### 主要发现
1. **CXL-Hybrid Memory 可作为高效的远程推理内存扩展方案**，支持 TB 级字节可寻址访问，简化软件栈。
2. **LLM 推理中的确定性访问模式（如逐层权重读取、前缀 KV 追加）是实现高效预取的关键前提**。
3. **多级 DMA 流水线 + 读优先调度** 能有效解决远程存储的延迟与 I/O 竞争问题。
4. **ITME 在长上下文、多轮推理场景下显著优于传统 CPU-offload 和 NVMe-oF 方案**，尤其在内存饱和后优势明显。

---

### 方法的局限性
- **依赖 CXL-Hybrid 硬件支持**：目前商用 CXL-Hybrid 产品尚不普及，部署门槛较高。
- **对突发随机访问适应性弱**：若 KV Cache 访问高度随机，预取收益可能下降。
- **FPGA 原型性能未达理论极限**：当前实现受限于元数据管理开销，需进一步优化硬件逻辑。
- **未考虑多租户隔离与安全性**：作为共享资源池，缺乏细粒度权限控制机制。

---

### 未来工作方向
- **支持更多异构内存类型**：如集成 PMEM、HBM 等，构建更灵活的分层池。
- **动态预取策略优化**：结合运行时 profiling 自适应调整预取粒度与时机。
- **跨集群全局内存池管理**：实现更大规模的 CXL 内存池编排与容错。
- **软硬协同优化推理框架**：将 ITME 集成进 vLLM、TensorRT-LLM 等主流引擎，实现端到端自动化管理。

--- 

> **总结一句话**：  
> ITME 通过 **CXL-Hybrid Memory + 硬件预取 + 多级 DMA 流水线**，实现了面向 LLM 推理的高效、可扩展、低成本的远程内存扩展，在长上下文场景下相较传统方案最高提升 **35.7% 吞吐量**，为下一代 disaggregated inference infrastructure 提供了可行路径。

</details>

---

### 2. [Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling](https://arxiv.org/abs/2606.12370)

**Authors**: Yucheng Li, Huiqiang Jiang, Yang Xu, Jianxin Yang, Yi Zhang, Yizhong Cao, Yuhao Shen, Fan Zhou, Rui Men, Jianwei Zhang, An Yang, Bowen Yu, Bo Zheng, Fei Huang, Junyang Lin, Dayiheng Liu, Jingren Zhou  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2606.12370v1  

#### Abstract
Reinforcement learning (RL) has become a key component in modern large language models, yet the rollout stage remains the key bottleneck in RL training pipelines. Although Multi-Token Prediction (MTP) offers a natural solution to accelerate rollouts through speculative decoding, many studies have ob...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Breaking Entropy Bounds: Accelerating RL Training via MTP with Rejection Sampling

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

在大型语言模型（LLM）的强化学习（RL）训练中，**rollout 阶段是主要计算瓶颈**。尽管 Multi-Token Prediction (MTP) 被提出用于加速推理，但在 RL 训练过程中，MTP 的接受率（acceptance rate）会显著下降，导致实际加速效果有限。

本文通过系统分析发现，**MTP 接受率的根本限制来自于策略模型（policy model）的熵波动（entropy fluctuation）**，而非传统认为的“分布不匹配”（distribution mismatch）。高熵使得目标分布更分散，从而降低了 draft 与 target 分布之间的重叠度，进而降低 MTP 的有效性。

---

### 提出的新方法与新思路

作者提出了 **Bebop** 框架，包含以下三个核心创新：

#### （1）揭示熵对 MTP 接受率的线性约束关系  
首次从理论上和实证上证明：  
> MTP 接受长度（acceptance length）与目标模型熵呈**负线性关系**，即 $ \alpha \sim a - b \cdot H(p) $。  
这一发现解释了为何在 RL 中熵上升时 MTP 性能退化。

#### （2）提出端到端 TV Loss（e2e TV Loss）  
传统 MTP 使用 Cross-Entropy (CE) 或 KL 散度作为训练目标，但这两种损失仅间接优化 TV 距离（Total Variation Distance），而 TV 距离才直接决定 rejection sampling 的接受率。

为此，作者提出：
- **TV Loss**：直接最小化 $ d_{\text{TV}}(p, q) = 1 - \sum \min(p(y), q(y)) $
- **End-to-End TV Loss**：进一步考虑多步 MTP 的乘积结构，直接优化期望接受长度：
  $$
  \mathcal{L}_{\text{e2e}} = 1 - \prod_{i=1}^\gamma (1 - d_{\text{TV}}(p_i, q_i))
  $$

该损失具有**梯度有界性**，训练更稳定，并能实现“概率比例型误差”（probability-proportional mismatch），使 draft 分布对熵变化鲁棒。

#### （3）预训练阶段完成 MTP 适配，无需在线更新  
通过结合 **rejection sampling + e2e TV loss**，可在 RL 开始前一次性完成 MTP 模块训练（pre-RL adaptation），在整个 RL 过程中保持高且稳定的接受率，**无需昂贵的在线 MTP 更新**。

---

### 相比现有方法的优势

| 维度 | 传统方法 | Bebop |
|------|--------|-------|
| **训练目标** | CE/KL Loss | e2e TV Loss（直接优化接受率） |
| **采样方式** | Target-Only Sampling | Rejection Sampling（更少受熵影响） |
| **MTP 更新策略** | 在线 co-training（高开销） | 一次式 pre-RL 训练（零额外开销） |
| **对熵的敏感性** | 强相关（接受率随熵升高而下降） | 几乎无关（>95% 减弱熵依赖） |
| **工程复杂度** | 高（需同步更新 MTP 头） | 低（冻结 MTP 头即可） |

---

## 2. 核心实验方法和设置

### 使用的数据集与任务

实验覆盖多种下游任务，包括：
- **数学推理**：HMMT25, AIME25
- **代码生成与修复**：LiveCodeBench, SWE-Bench（multi-turn code editing）
- **智能体任务（agentic tasks）**：Hybrid, Long-Horizon, Tool Use
- **通用对话能力**：MT-Bench（含 OOD 测试）

模型系列：Qwen3.5, Qwen3.6, Qwen3.7 系列（参数量从 3.5B 到 30B+）

---

### 实验设置与评估指标

#### 主要设置
- **MTP 步数**：$\gamma = 3$（每次验证 4 个 token）
- **训练流程**：
  - 先进行 SFT + MTP 头训练（使用不同 loss）
  - 再进入异步 RL 训练（async RL pipeline）
- **Rollout 引擎**：SGLang
- **RL 框架**：veRL-based async RL
- **Batch Size**：全局 batch 256，seq len 最高达 128K

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Acceptance Rate / Accept Length** | 每轮验证平均接受的 token 数，决定吞吐提升上限 |
| **Throughput Gain** | 实际推理吞吐提升倍数 |
| **End-to-End Speedup** | 完整 RL 训练时间加速比 |
| **Latency Reduction** | 单步 rollout 延迟降低 |
| **Decomposition of $\Delta \alpha$** | 将接受率变化分解为熵驱动 vs. 分布不匹配驱动 |

#### 基线方法对比
- **Baseline MTP Losses**：
  - CE Loss
  - KL Loss ($D_{\text{KL}}(p \| q)$)
  - Reverse KL Loss ($D_{\text{KL}}(q \| p)$)
  - Per-step TV Loss
- **Sampling 方法**：
  - Target-Only Sampling
  - Rejection Sampling
- **MTP 更新策略**：
  - 固定 MTP（no update）
  - 在线更新（online update with CE/TV loss）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 接受率大幅提升
| 方法 | 平均 Acceptance Rate 提升（vs. CE） |
|------|-------------------------------|
| TV Loss | +2.4% ~ +5.2% |
| **e2e TV Loss（本文）** | **+3.0% ~ +8.0%** |
| 最高达到 | **98.6%**（Agent 任务上的 Qwen3.7-Plus）|

> 在 agent 和 SWE 类任务上增益最大（+8%），说明结构化输出更容易被预测。

#### ✅ 吞吐与延迟显著改善
- **吞吐提升**：最高达 **25% 额外推理吞吐增益**
- **单步延迟降低**：**1.5× ~ 1.8× 减少**
- **Agentic RL 加速更明显**：某些场景下 rollout 阶段提速达 **2.4×**

#### ✅ 端到端训练加速
- **完整 RL 训练流程加速最高达 1.8×**
- 特别是在 Qwen3.7 系列的大规模训练中表现稳健

---

### 与基线方法的对比结果

| 对比项 | 结果 |
|-------|------|
| **TV Loss vs. CE/KL** | 所有任务上均优于 CE/KL，尤其在高熵任务（如 SWE）优势更大 |
| **Rejection Sampling vs. Target-Only** | 在大多数情况下 RS 更优；仅当 draft 质量极差时 TO 可能略好 |
| **e2e TV Loss vs. Per-step TV** | 多步累积效应更强，后期 MTP 步骤（Step 2/3）接受率高出 2.5–5% |
| **Pre-RL TV Training vs. Online Update** | 在线更新无显著收益，甚至用 CE loss 更新会导致性能回退 |

---

### 消融实验结果

#### （1）熵 vs. 分布不匹配的归因分析（Fig. 3）
- 在 rejection sampling + TV loss 下：
  - **熵变化引起的接受率下降占比 >90%**
  - **分布不匹配贡献几乎为零（Δα_mismatch ≈ 0）**
- 表明：只要解决熵问题，就无需担心 RL 更新带来的漂移

#### （2）是否需要在线更新 MTP？
- 实验表明：**从已训练好的 TV checkpoint 继续在线训练，接受率不会提升**
- 若改用 CE loss 更新，反而会使 draft 分布变平滑，破坏 TV 带来的 sharpness，导致接受率下降

#### （3）不同 loss 对 draft 分布的影响（Fig. 11）
- **TV Loss**：产生更 sharp 的 draft 分布（接近 target），但 KL 更大
- **CE Loss**：产生更平滑的分布，虽 KL 小但 TV 大，不利于 rejection sampling

---

## 4. 关键结论和发现

### 主要发现

1. 🔍 **MTP 接受率的核心瓶颈是熵，不是分布不匹配**  
   - 接受率与目标熵呈强负线性关系
   - 政策更新导致的权重漂移影响微弱

2. 🛠️ **Rejection Sampling 显著缓解熵敏感性**  
   - 其接受率取决于分布重叠（overlap），而非最大概率值
   - 比 target-only 更适合高熵 RL 场景

3. 💡 **e2e TV Loss 是最优训练目标**
   - 直接优化 TV 距离，梯度有界、训练稳定
   - 实现“概率比例误差”，天然抗熵扰动
   - 较 CE 基线提升约 **10% 接受率**

4. ✅ **Pre-RL MTP 训练足够，无需在线更新**
   - 一次训练 + rejection sampling 可维持全程高性能
   - 极大简化系统设计，节省内存与计算资源

5. 📈 **加速效果随模型增大而增强**
   - Qwen3.7 比 Qwen3.5 表现更好，表明 MTP 在大模型上更具潜力

---

### 方法的局限性

1. **熵不变性的前提是分布内（ID）数据**
   - 当 RL 探索导致策略熵远超 SFT 数据范围时，draft 可能遇到 OOD 分布，此时 TV 训练的优势减弱
   - 建议在这种情况下重启 MTP co-training with TV loss

2. **Full-vocab TV Loss 存在显存压力**
   - 虽然作者采用 fused kernel 优化，但在超大词表（>100K）上仍有挑战
   - Top-K 截断近似不稳定（见 Fig. 14b），不推荐使用

3. **未探索动态调整 γ（draft length）**
   - 当前固定 γ=3，未来可基于局部熵自适应调节以进一步提效

---

### 未来工作方向

- ✅ 将 TV Loss 应用于其他 speculative decoding 架构（如 early-exit, small model draft）
- ✅ 设计 entropy-aware MTP adapter，在 RL 中动态扩展 draft 能力
- ✅ 结合 temperature scheduling 与 MTP，平衡探索效率与 rollout 成本
- ✅ 探索 MTP 在 test-time computing、agent planning 中的应用

--- 

> **总结一句话**：  
> Bebop 揭示了 **Entropy 是 MTP 在 RL 中失效的本质原因**，并通过 **Rejection Sampling + e2e TV Loss + Pre-RL Training** 的组合拳，实现了高达 **1.8× 端到端加速**，为大规模 LLM 的 RL 训练提供了高效、实用的新范式。

</details>

---

### 3. [Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training](https://arxiv.org/abs/2606.12240)

**Authors**: Shilong Zong, Almuatazbellah Boker, Hoda Eldardiry  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2606.12240v1  

#### Abstract
Multivariate time-series data often exhibit complex temporal dependencies, irregular sampling, and heterogeneous dynamics across multiple time scales, making accurate sequence modeling particularly challenging. Traditional recurrent neural networks (RNNs), such as Long Short-Term Memory (LSTM) netwo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Multi-Rate Mixture of Experts for Accelerating Liquid Neural Network Training*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 RNN（如 LSTM）在处理**多变量时间序列**时存在以下局限：
- 依赖离散时间建模，难以捕捉连续、不规则采样的动态；
- 单一模型结构无法有效建模**异质性**和**多尺度时间动态**（fast-changing vs. slow-evolving patterns）；
- 标准 Liquid Neural Networks (LNNs) 虽然支持连续时间动态，但仍基于单一动力系统，表达能力受限。

此外，LNNs 因需数值积分微分方程，计算开销较高，训练效率低。

### 🚀 提出的新方法与创新思路
本文提出了一种新的架构：**Multi-Rate Mixture-of-Experts (MR-MoE)**，并进一步扩展为 **MR-MoE-Attention**，其核心创新包括：

| 创新点 | 描述 |
|--------|------|
| **Multi-Rate MoE 架构** | 多个基于 LNN 的专家（expert）在**不同时间尺度**上运行，分别捕获快变与慢变的时间模式。通过奇异摄动理论（singular perturbation theory），对快速专家采用准稳态近似以提升效率。 |
| **Gating Network 自适应选择专家** | 引入门控网络（gating network），根据输入条件动态分配各专家权重，实现专家专业化（expert specialization）。 |
| **双注意力机制集成** | 首次将 **feature-level attention** 和 **temporal attention** 同时引入 LNN-based MoE 框架：<br>• **Feature-level attention** 抑制噪声或无关变量；<br>• **Temporal attention** 改善长距离依赖建模，聚焦重要历史状态。 |
| **统一框架整合四大要素** | 将 **continuous-time dynamics (LNN)**、**multi-scale modeling (Multi-Rate)**、**expert specialization (MoE)** 与 **adaptive attention** 统一于一个端到端可训练架构中。据称是首个同时结合 MoE 与 attention 的 LNN 模型。 |

### 🔍 相比现有方法的优势
- 更强的表达能力：通过多专家分工 + 多时间尺度建模，能更好分离异质动态；
- 更高的鲁棒性：注意力机制抑制噪声传播，增强信号-噪声比；
- 更优的性能-效率权衡：相比 LSTM 显著降低内存占用，且 MR-MoE 设计提升了计算效率；
- 可解释性增强：注意力权重提供特征与时间步的重要性可视化。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **任务**：脓毒症预测（sepsis prediction）
- **来源**：重症监护病房（ICU）患者的多变量临床时间序列数据（来自 Moor et al. [2023]）
- **数据内容**：生命体征、实验室检测值等生理测量指标
- **预处理**：
  - 归一化
  - 缺失值前向填充（forward filling）
- **划分方式**：标准划分为训练集、验证集和测试集

### ⚙️ 实验设置
- **实现平台**：PyTorch
- **优化器**：Adam
- **学习率**：1e-3
- **Batch Size**：固定
- **模型配置**：
  - 所有模型隐藏层维度为 1500
  - MoE 类模型使用 $ K=3 $ 个专家，对应 fast / intermediate / slow 时间尺度
  - Gating network 为小型 MLP
  - Attention 模块：
    - Feature-level attention：两层 MLP
    - Temporal attention：标准点积注意力（dot-product attention）

### 📊 评估指标
针对类别不平衡的医学预测任务，采用以下指标：
- **AUROC**（Area Under the ROC Curve）
- **AUPRC**（Area Under the Precision-Recall Curve）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（Test Set）

| 模型 | AUROC | AUPRC |
|------|-------|--------|
| **LSTM** | ~0.53 | ~0.22 |
| **Monolithic LNN** | ~0.55 | ~0.32 |
| **MoE (LNN experts)** | ~0.58 | ~0.36 |
| **MR-MoE**（无 attention） | ~0.61 | ~0.42 |
| **MR-MoE-Attention**（完整模型） | **~0.65–0.68** | **~0.45** |

> ✅ 完整模型在 AUROC 上相较 LSTM 提升超过 **22%**，AUPRC 提升超过 **100%**

### 🔁 基线对比结果
- **MR-MoE > MoE > Monolithic LNN > LSTM**：逐级提升，表明多尺度设计优于单系统或多专家但同速率的设计；
- **MR-MoE-Attention 最优**：加入 attention 后性能进一步显著提升，说明 feature 和 temporal 注意力均带来增益；
- 在相同参数量下，MR-MoE 内存消耗低于 LSTM，且推理更高效（得益于快速专家的简化建模）。

### 🔍 消融实验分析（Ablation Study）
虽然未明确列出“ablation”小节，但从逐步构建过程可视为隐式消融：

| 模块添加 | 性能变化 | 分析 |
|---------|----------|------|
| 加入 MoE（vs. 单一 LNN） | AUROC ↑0.03, AUPRC ↑0.04 | 专家分解提升多样性与泛化能力 |
| 引入 Multi-Rate 结构 | AUROC ↑0.03, AUPRC ↑0.06 | 显式分离时间尺度显著改善建模效果 |
| 添加双 attention 机制 | AUROC ↑0.04, AUPRC ↑0.03 | 特征筛选与历史聚焦共同推动性能跃升 |

此外，图14显示：
- 在增加输入噪声的情况下，**MR-MoE-Attention 表现出最强鲁棒性**，性能下降最缓慢；
- 表明该模型在真实嘈杂医疗环境中更具实用性。

---

## 4. 关键结论和发现

### ✅ 主要结论
1. **多时间尺度建模至关重要**：现实世界时间序列常包含快慢交织的过程，显式建模 multi-rate dynamics 显著优于单一速率系统。
2. **MoE + LNN 是有效组合**：将 Mixture-of-Experts 与 Liquid Neural Networks 结合，能够实现专家间的功能分化，提升模型表达力。
3. **注意力机制与 MoE 协同增效**：feature-level 与 temporal attention 不仅提升准确率，还增强了模型鲁棒性和可解释性。
4. **性能与效率兼顾**：MR-MoE 在保持高预测性能的同时，通过准稳态近似减少计算负担，优于传统 LSTM 的内存效率。

### ⚠️ 方法的局限性
- **计算复杂度仍高于简单模型**：尽管进行了优化，但连续时间动态 + attention 导致训练成本高于普通 RNN；
- **超参数敏感性**：时间常数 $\tau_k$ 当前为手动设定，可能影响性能；
- **缺乏统计显著性检验**：由于算力限制，未报告误差条或 p-value；
- **尚未开源代码与数据**：目前未公开 code/data，影响复现性（见 checklist 第5项）。

### 🔮 未来工作方向
1. **Decoupled Multi-Time-Scale Training**  
   探索分层或交替训练策略，解耦快慢专家的优化过程，避免相互干扰。

2. **Learnable Time Constants**  
   将时间常数 $\tau_k$ 设为可学习参数，让模型自动从数据中发现最优时间尺度。

3. **提升可扩展性与灵活性**  
   开发更自适应的 multi-scale 架构，适用于更大规模、更多样化的时序任务。

4. **探索其他应用场景**  
   如金融、物联网、脑电信号等同样具有多尺度动态特性的领域。

---

## ✅ 总结一句话
本论文提出了 **MR-MoE** 框架——一种融合 **Liquid Neural Networks**、**Multi-Rate Dynamics**、**Mixture-of-Experts** 与 **Dual Attention Mechanisms** 的新型时间序列建模架构，在脓毒症预测任务上显著超越 LSTM、单体 LNN 和标准 MoE 模型，兼具高性能、高鲁棒性与良好效率，为复杂临床时序分析提供了有力工具。

</details>

---

### 4. [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)

**Authors**: Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Pengyu Zhao  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.13392v1  

#### Abstract
Ultra-long-context capability is becoming indispensable for frontier LLMs: agentic workflows, repository-scale code reasoning, and persistent memory all require the model to jointly attend over hundreds of thousands to millions of tokens, yet the quadratic cost of softmax attention makes this untena...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：MiniMax Sparse Attention**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前前沿大语言模型（LLMs）对**超长上下文能力**的需求日益增长，例如智能体工作流（agentic workflows）、代码库级推理和持久记忆等场景需要模型处理数十万甚至数百万 token 的上下文。然而，标准的 **softmax attention** 具有 $O(N^2)$ 的计算复杂度，在实际部署中面临严重的计算和内存瓶颈。

尽管已有多种稀疏注意力机制被提出，但多数存在以下问题：
- 实现复杂，难以高效部署；
- 依赖复杂的训练策略或额外模块；
- 难以在保持性能的同时实现显著的端到端加速。

---

### **提出的新方法：MiniMax Sparse Attention (MSA)**

MSA 是一种基于 **Grouped Query Attention (GQA)** 的块级稀疏注意力机制，其设计遵循奥卡姆剃刀原则（Occam’s Razor），仅保留最核心组件，具备高简洁性、可扩展性和可加速性。

#### **核心架构设计**
- **双分支结构**：
  - **Index Branch（索引分支）**：轻量级分支，为每个 GQA 组独立打分并选择 Top-k 的 KV Block。
    - 仅引入两个额外投影矩阵（$W_{idx}, W_{kdx}$）。
    - 使用 **blockwise max-pooling** 对 Key-Value 块进行打分。
    - 引入 **local block 强制保留机制**，确保稳定性。
  - **Main Branch（主分支）**：标准 GQA 注意力，但只在 Index Branch 选出的块上执行。
    - 实现精确的 block-sparse attention。
- **训练机制**：
  - 使用 **KL Alignment Loss** 对齐 Index Branch 与 Main Branch 在选中块上的分布。
  - 采用 **Gradient Detach** 和 **Indexer Warmup** 策略稳定训练。
  - 不依赖额外输出路径（如 Indexer 的 value head），简化结构。

#### **相比现有方法的优势**
| 特性 | MSA | 其他稀疏方法（如 NSA, MoBA, DSA） |
|------|-----|-------------------------------|
| 架构简洁性 | ✅ 极简，仅增加少量参数 | ❌ 多分支、复杂结构 |
| 可扩展性 | ✅ 支持从零训练和预训练模型转换 | ⚠️ 多数仅支持特定模式 |
| 执行效率 | ✅ 支持高效的 block-granular GPU 内核 | ⚠️ 多数未协同优化内核 |
| 模态兼容性 | ✅ 已验证于原生多模态模型 | ⚠️ 多数仅限文本 |
| 实际加速效果 | ✅ 显著端到端提速（见下文） | ⚠️ 理论 FLOPs 降低 ≠ 实际速度提升 |

> ✅ **关键优势总结**：MSA 在**不牺牲模型能力的前提下**，通过算法-硬件协同设计，将稀疏注意力的理论收益转化为**真实的 wall-clock 速度提升**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **预训练语料**：混合文本与图像/视频数据，总预算 **3T tokens**。
- **评估基准**：
  - **通用推理与问答**：MMLU, MMLU-Pro, BBH, GPQA Hard, ARC Challenge, TriviaQA, WinoGrande
  - **数学与代码**：GSM8K, MGSM, MathVista, OlymMATH, HumanEval, EvalPlus, BigCodeBench, MultiPL-E MBPP
  - **多模态能力**：
    - 图像：AI2D, ChartQA, MMMU, OCRBench v2, CharXiv, VisualWebBench, CVBench
    - 视频：EgoSchema, LongVideoBench, MLVU, MMVU, VideoMME, TemporalBench
  - **长上下文检索**：RULER, HELMET
  - **代理任务困惑度（PPL）**：TAU2, AgentCompany, Humanity's Last Exam (HLE), SWE-bench

---

### **实验设置**
- **模型规模**：109B 参数的 MoE 模型（激活参数约 6B/token）
- **层数**：41 层（前 3 层为 dense，其余为 MoE）
- **注意力配置**：
  - Query Heads: 64
  - KV Heads: 4 → GQA Ratio = 16
  - Head Dim: 128
  - RoPE Dim: 64
- **稀疏设置**：
  - Block Size $B_k = 128$
  - Top-k Blocks per Group: $k = 16$ → 每 query 最多访问 $16 \times 128 = 2048$ 个 KV tokens
- **训练方式对比**：
  - **Full Attention (GQA)**：全注意力基线
  - **MSA-PT**：从头开始训练 MSA
  - **MSA-CPT**：从已训练的 Full-Attention Checkpoint 转换后继续预训练

---

### **评估指标**
- **模型能力**：各基准任务准确率（↑越高越好）
- **语言建模质量**：下游任务的 Perplexity（↓越低越好）
- **效率指标**：
  - 理论 FLOPs 减少倍数
  - 实际 **prefill 和 decoding 阶段的 wall-clock 速度提升**

---

### **基线方法对比**
- 主要对比对象：**Grouped Query Attention (GQA)** 全注意力模型
- 同类稀疏方法参考对比（文中提及）：
  - **NSA**, **MoBA**, **DSA**, **InfLLM-V2** 等
- 本文强调 MSA 在**结构简洁性、训练稳定性、硬件适配性**方面优于这些方法。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

| 指标 | 数值 | 说明 |
|------|------|------|
| **FLOPs Reduction @ 1M context** | **28.4×** | 相比 GQA 的每 token attention 计算量减少 |
| **Prefill Speedup (H800)** | **14.2×** | 实际前向填充阶段加速 |
| **Decoding Speedup (H800)** | **7.6×** | 实际自回归生成阶段加速 |
| **Context Length Supported** | Up to **1M tokens** | 在极长上下文中仍保持高效 |

> 📌 注：上述加速是在 **109B MoE + 原生多模态训练** 的大规模设置下达成。

---

### **与基线方法的对比结果**

#### **表：代表性评估结果（部分摘录）**
| Benchmark | Full | MSA-PT | MSA-CPT |
|---------|------|--------|---------|
| MMLU | 67.0 | **67.2** | 66.8 |
| GSM8K | 76.2 | **77.7** | 73.7 |
| HumanEval | 61.0 | **64.0** | 57.9 |
| RULER-32K | 75.0 | **77.5** | 75.7 |
| AI2D | 68.3 | **70.6** | 67.3 |
| EgoSchema | 29.6 | **37.6** | 25.8 |
| TAU2 (PPL↓) | 1.155 | **1.148** | 1.150 |

> ✅ **结论**：MSA-PT 在多数任务上持平或超越 Full Attention；MSA-CPT 表现稳健，适合已有模型迁移。

#### **长上下文扩展实验（MSA-CPT + 140B 更长上下文训练）**
| Benchmark | Subset | Full | MSA-CPT | Δ |
|----------|--------|------|---------|----|
| HELMET-128K | Overall | 46.53 | 45.93 | -0.60 |
| RULER-128K | Overall | 72.00 | **72.12** | **+0.12** |
| RULER-128K | MK/MQ/MV | 96.63 | **98.87** | **+2.24** |

> ✅ 即使在 **128K 上下文** 下，MSA 依然能保持竞争力，尤其在知识检索子任务上表现优异。

---

### **消融实验结果**

#### **(1) Block Size 影响（固定总 token 数）**
| Block Size | 32 | 64 | 128 |
|-----------|----|----|-----|
| RULER-8K | 72.5 | 72.8 | **73.8** |
| PPL (TAU2) | 1.176 | 1.176 | 1.176 |

> 🔍 结论：更大的 block size 不影响性能，反而有利于 kernel 效率（提高 arithmetic intensity）。

#### **(2) 是否强制 sink/local window**
| 设置 | 强制 sink + local | 无强制 |
|------|------------------|--------|
| RULER-32K | 65.8 | 61.5 |
| PPL | ~1.23–1.30 | ~1.23–1.30 |

> 🔍 结论：移除硬编码规则后性能基本不变，说明模型能**自主学习 sink 和局部关注模式**。

#### **(3) Index Branch 是否使用 value head**
| 设置 | With-value | No-value |
|------|------------|----------|
| MMLU | 66.4 | **67.3** |
| GSM8K | **77.6** | 76.4 |
| RULER-32K | 79.7 | **80.4** |

> 🔍 结论：**无需 value head**，KL loss + warmup 已足够训练 indexer。

#### **(4) 动态选择 vs 固定滑动窗口**
- 在相同 FLOPs 预算下，**动态 MSA 明显优于固定滑动窗口**（lower PPL）。
- 表明 **content-dependent selection** 对 agent 类任务至关重要。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **MSA 可在不损失性能前提下大幅降低 attention 成本**：
   - 在 1M context 下实现 **28.4× FLOPs 减少** 和 **14.2× prefill / 7.6× decoding 实际加速**。
2. ✅ **双分支 + KL Alignment 设计有效且稳定**：
   - Index Branch 可被准确训练并与 Main Branch 对齐。
   - 可用于从零训练（MSA-PT）或已有模型转换（MSA-CPT）。
3. ✅ **支持原生多模态训练**：
   - 在图像、视频理解任务上表现强劲，证明其跨模态泛化能力。
4. ✅ **算法-硬件协同设计是关键**：
   - 自研 **exp-free Top-k kernel** 和 **KV-outer sparse attention kernel** 显著提升了 tensor-core 利用率。
5. ✅ **稀疏模式具有多样性**：
   - 不同 GQA group 学会了不同的 long-range stripe 模式，而非单一全局模式。
   - 注意力 sink 现象自然出现，无需显式设计。

---

### **方法的局限性**
1. ⚠️ **长上下文检索仍有微小差距**：
   - 尤其在 HELMET 和部分 RULER 子任务上略低于 Full Attention。
   - 可能因过紧的 selection budget（仅 2048 tokens）导致信息遗漏。
2. ⚠️ **极端稀疏可能导致信息丢失风险**：
   - 若 indexer 错误地忽略了关键历史块，可能影响推理连贯性。
3. ⚠️ 当前设计依赖 GQA 架构：
   - 虽然主流模型多用 GQA，但对 MQA/MHA 的适配需进一步验证。

---

### **未来工作方向**
1. **缩小长上下文检索差距**：
   - 探索更大 inference-time selection budget；
   - 引入更丰富的 indexer scoring function（如 hierarchical 或 cross-layer）。
2. **扩展至 RLPT 和 Agent Deployment**：
   - 在强化学习后训练（RLPT）和真实 agent 部署中应用 MSA，应对更严苛的成本约束。
3. **探索 zero-shot dense-to-sparse 切换**：
   - 如 InfLLM-V2 所示，实现无缝短上下文→长上下文切换。
4. **进一步优化 kernel 支持更灵活 block 结构**：
   - 支持 variable-size blocks 或 hierarchical sparse attention。

---

> 💡 **总结一句话**：  
> **MiniMax Sparse Attention (MSA)** 通过极简而高效的块级稀疏注意力设计，结合算法-硬件协同优化，在 **109B 级原生多模态模型** 上实现了与 GQA 相当的性能，并带来了高达 **14.2× 的实际推理加速**，为超长上下文 LLM 的实用化部署提供了强有力的技术路径。

</details>

---

### 5. [Re-evaluating Confidence Remasking in Masked Diffusion Language Models](https://arxiv.org/abs/2606.12232)

**Authors**: Stipe Frkovic, Metod Jazbec, Dan Zhang, Christian A. Naesseth, Ilija Bogunovic, Eric Nalisnick  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2606.12232v1  

#### Abstract
Masked diffusion language models (dLLMs) have recently emerged as a competitive alternative to autoregressive language models, with the promise of faster inference via parallel token generation. A notable limitation of the masked formulation, however, is that once a token has been unmasked it can no...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Re-evaluating Confidence Remasking in Masked Diffusion Language Models

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文重新审视了 **post-hoc confidence-based remasking** 在 **masked diffusion language models (dLLMs)** 中的有效性。尽管近期研究（如 WINO）声称通过基于置信度的 remasking 能够实现自我修正、提升生成质量，但这些结论缺乏在标准解码设置下的严谨比较。

本文指出，现有评估存在三大缺陷：
- 以较弱的 **high-confidence sampling** 为基准，而非更强的 **Fast-dLLM**；
- 主要关注大 block length (BL=128)，而实际应用中更常用小 block length (BL=32)；
- 忽略非贪婪解码（non-greedy decoding）场景下 remasking 的影响。

因此，论文旨在回答一个关键问题：**post-hoc remasking 是否真的带来了超越先进 unmasking 方法的实质性收益？**

### 提出了什么新方法或新思路
本文**并未提出新的 remasking 方法**，而是对代表性 post-hoc remasking 方法 **WINO** 进行了系统性的再评估，并提出了更全面的评估框架，强调需考虑以下因素：
- 不同 **block length**（BL=32 vs BL=128）
- 不同 **unmasking 策略**（Fast-dLLM vs dUltra）
- 不同 **解码方式**（greedy vs non-greedy）
- 更合理的效率衡量（**latency / throughput** 而不仅是 NFE）

### 相比现有方法的优势
- **批判性视角**：揭示了当前 remasking 方法收益可能被高估，尤其是在标准设置下。
- **系统性评估框架**：提出应综合考虑多种解码设置来评估 remasking 的真实价值。
- **深入归因分析**：通过 flip-flop 分析等手段，定位了 remasking 效果不佳的根本原因在于模型本身难以提出更好的替代 token。

---

## 2. 核心实验方法和设置

### 使用的数据集
在四个标准 benchmark 上进行评估：
- **GSM8k**：数学推理任务
- **MATH-500**：复杂数学问题求解
- **HumanEval**：代码生成能力测试
- **MBPP**：面向编程的自然语言到代码生成

### 实验设置和评估指标
#### 模型
- **LLaDA-8B-Instruct** [Nie et al., 2025]
- **Dream-v0-Instruct-7B** [Ye et al., 2025]

#### 解码策略对比
| 方法 | 类型 | 关键机制 |
|------|------|--------|
| **Fast-dLLM** | Baseline | 基于置信度阈值的 adaptive unmasking |
| **WINO** | Post-hoc remasking | 在 Fast-dLLM 基础上增加 shadow token 和 remasking 判断 |
| **dUltra** | Stochastic unmasking | 学习型 Bernoulli policy 决定 unmasking |

#### 变量控制
- **Block Length (BL)**：32（标准） vs 128（文献常用）
- **Sampling Temperature (T)**：0（greedy） vs 0.8 / 1.5（non-greedy）
- **Unmasking Threshold λ₁**：{0.5, 0.6, 0.7, 0.8, 0.9}
- **Remasking Threshold λ₂**：固定为 0.8

#### 评估指标
- **Accuracy / Pass@k**：衡量任务完成率（尤其 Pass@1, Pass@64）
- **Network Function Evaluations (NFEs)**：前向传播次数，反映计算成本
- **Latency / Throughput (tokens/sec)**：真实时间开销，更贴近实际部署
- **Flip-flop Frequency**：remask 后又预测回原 token 的比例，反映修正有效性

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 在标准设置下（BL=32, greedy decoding）：
- **WINO vs Fast-dLLM**：
  - 准确率提升极其有限：平均仅 **+0.4% (LLaDA)** 和 **+0.5% (Dream)**
  - 当考虑 **latency 开销**（shadow block 增加 FLOPs），WINO 的吞吐量更低，**性价比不具优势**
  - 图表显示 WINO 的帕累托前沿（accuracy vs NFE/latency）几乎与 Fast-dLLM 重合

> 🔍 结论：**在标准设置下，remasking 几乎没有带来额外收益**

#### ⚠️ 在大 block length 下（BL=128）：
- WINO 表现相对更好，准确率提升约 **+1.3%~1.5%**
- 但这主要是因为 **Fast-dLLM 在大 block 下性能严重下降**（EOS confidence corruption）
- 实际上，**WINO BL=128 的绝对性能仍低于 Fast-dLLM BL=32**

> 🔍 结论：WINO 的“增益”更多是**修复 Fast-dLLM 在非标准设置下的退化**，而非真正提升模型上限

#### 🔄 Flip-flop 分析（失败归因）：
- 高达 **75–90% (LLaDA)** 和 **85–95% (Dream)** 的 remask 操作最终导致 token 被重新预测为原来的样子
- 即使尝试扩展 remasking 范围（remask 邻居 token 或后续 token），也无法降低 flip-flop 率或提升性能
- 表明问题不在上下文依赖（cascading dependency），而在 **dLLM 自身无法生成更优替代 token**

> 🔍 结论：**模型的预测分布本身是瓶颈**

#### 🎲 在非贪婪解码（stochastic generation）下：
| 场景 | 发现 |
|------|------|
| **Non-greedy decoding (T > 0)** |  
&nbsp;&nbsp;• WINO 提升 **Pass@1**（平均 +2.6% @ T=0.8）  
&nbsp;&nbsp;• 但 **Pass@64 提升缩小至 +0.7%**，表明多样性进一步受限  
&nbsp;&nbsp;• 支持“diversity collapse”现象被 remasking 加剧 |
| **Stochastic unmasking (dUltra + T=0)** |  
&nbsp;&nbsp;• WINO 带来显著提升：**平均 +3.2% accuracy**  
&nbsp;&nbsp;• 仅增加 ~2 NFEs  
&nbsp;&nbsp;• 表明 remasking 对抗 **随机 unmasking 引入的噪声** 更有效 |

> 🔍 结论：**remasking 的价值高度依赖于 unmasking 策略和解码设置**

#### 🔁 其他 remasking 方法验证（Saber）
- 在相同设置下测试 **Saber** [Dong et al., 2025]
- 结果类似：**在 BL=32 下未能显著优于 Fast-dLLM**
- 支持本文结论具有普遍性，不限于 WINO

---

## 4. 关键结论和发现

### 主要发现
1. **Post-hoc remasking（如 WINO）在标准设置下（BL=32, greedy）几乎没有额外收益**，其微弱提升常被 latency 成本抵消。
2. **大 block length 下的增益主要源于补偿 Fast-dLLM 的退化**，而非提升模型本质能力。
3. **高 flip-flop 率表明模型无法提出更好 token**，说明当前 dLLMs 缺乏真正的“自我修正”能力。
4. **Remasking 在非贪婪或随机 unmasking 设置下更有潜力**，尤其能缓解由随机性引入的错误。
5. **Remasking 会加剧 confidence-based unmasking 已有的 diversity collapse 问题**。

### 方法的局限性
- **依赖 shadow token 的近似质量**：虽然 shadow token 是高效的 leave-one-out 近似，但在深层模型中仍存在信息泄露。
- **无法突破 dLLM 本身的建模限制**：remasking 不能解决模型训练时吸收过程（absorbing process）带来的根本性不可逆问题。
- **flip-flop 问题尚未解决**：即使识别出错误位置，模型仍倾向于重复错误。
- **效率代价明显**：shadow block 增加序列长度，显著提高 FLOPs 和内存占用。

### 未来工作方向
1. **开发更有效的 remasking 机制**：探索如何让模型在 remasking 后真正生成更优 token，例如引入外部记忆或迭代优化策略。
2. **转向 uniform diffusion 或 hybrid models**：放弃纯 absorbing 过程，允许自然 remasking（如 von Riutte et al., 2025）。
3. **结合训练时干预**：通过 fine-tuning 或 RL 训练模型具备真正的 self-correction 能力（如 Schiff et al., 2026）。
4. **建立标准化评估协议**：未来研究应报告多 setting（BL, T, unmasking policy）下的结果，避免片面结论。
5. **研究 remasking 对多样性和创造力的影响**：不仅看 accuracy，也需关注生成多样性、新颖性等维度。

---

> 💡 **一句话总结**：  
> 当前的 post-hoc confidence-based remasking（如 WINO）在标准 dLLM 解码设置下**增益甚微且代价高昂**，其效果高度依赖于具体解码策略；真正有效的自我修正可能需要**模型架构或训练方式的根本改变**。

</details>

---

### 6. [Accurate and Resource-Efficient Federated Continual Learning](https://arxiv.org/abs/2606.11480)

**Authors**: Jebacyril Arockiaraj, Dhruv Parikh, Jayashree Adivarahan, Rajgopal Kannan, Viktor Prasanna  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.11480v1  

#### Abstract
Federated continual learning (FCL) must learn from distributed task streams under limited resources, such as communication, computation, memory, and label availability. Existing FCL methods often rely on repeated local optimization, replay, and full supervision. Analytic alternatives avoid iterative...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Accurate and Resource-Efficient Federated Continual Learning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
联邦持续学习（Federated Continual Learning, FCL）面临多重资源约束，包括通信带宽有限、客户端计算能力弱、内存受限以及标签稀缺等问题。传统FCL方法通常依赖迭代优化、回放机制和全监督训练，导致高通信开销、计算成本大，并且在非独立同分布（non-IID）数据下表现不佳。

本文旨在设计一种**资源感知的FCL框架**，在保持高准确率的同时显著降低通信、计算和标签依赖。

---

### **提出的新方法：FedRAN**
作者提出了 **FedRAN**（Federated Random-feature Analytic Network），一个**资源感知的解析式联邦持续学习框架**，其核心思想是：

- **用紧凑的随机特征统计量替代梯度更新**：冻结预训练骨干网络，在客户端仅进行前向推理，提取固定维度的随机特征。
- **低秩谱摘要传输**：每个客户端不上传完整的二阶特征协方差矩阵（Gram矩阵），而是计算并上传其**截断SVD（truncated SVD）摘要** `(V, σ)`，将通信复杂度从 $O(M^2)$ 降至 $O(Mr)$，其中 $M$ 是随机特征维度，$r \ll M$ 是保留的秩。
- **两级OR-SVD子空间合并**：服务器端执行两层聚合：
  - **空间聚合**：跨客户端合并当前任务的局部SVD摘要；
  - **时间聚合**：跨任务序列合并历史全局子空间。
- **闭式分类器求解**：基于合并后的低秩Gram近似和精确的标签-特征统计 `B`，通过解析公式直接求解岭分类器（ridge classifier）。
- **支持半监督学习**：引入 **FedRAN-SSL** 变体，利用原型（prototype）相似性为高置信度无标签样本分配伪标签，提升标签稀缺场景下的性能。

---

### **相比现有方法的优势**

| 维度 | FedRAN优势 |
|------|-----------|
| **通信效率** | 比优化型FCL减少 **30.6–121.8×** 的每客户端通信量；避免多轮模型交换 |
| **计算效率** | 平均比梯度法快 **190.3×**；无需反向传播，仅需一次前向+统计构造 |
| **表示稳定性** | 冻结骨干网络，彻底消除“客户端诱导的特征漂移”（feature drift） |
| **准确性** | 在多个基准上平均准确率最高，优于最强基线达 **+4.8个百分点** |
| **标签效率** | 在仅有20%真实标签时，伪标签使平均准确率再提升 **高达6.61点** |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
在三个主流视觉分类基准上评估，涵盖标准、分布外和多样化领域任务：

| 数据集 | 类别数 | 任务数 | 特点 |
|--------|-------|--------|------|
| **CIFAR-100** | 100 | 5 / 10 | 标准图像分类，类增量学习 |
| **ImageNet-R** | 200 | 10 | Out-of-Distribution（OOD）渲染鲁棒性测试 |
| **VTAB** | 50 | 5 | 包含自然、医学、遥感等19个不同领域的通用视觉任务 |

---

### **实验设置**
- **客户端数量**：$K=5$
- **非IID划分**：使用Dirichlet分布 $\text{Dir}(\beta)$ 划分类别，$\beta \in \{0.1, 0.5, 1.0\}$ 控制偏斜程度（越小越不平衡）
- **骨干网络**：ResNet-18 和 ViT-B/16（均ImageNet预训练）
- **随机投影维度**：
  - ResNet：$M=8192$，秩 $r=2048$（CIFAR/ImageNet-R），$r=512$（VTAB）
  - ViT：$M=2048$，$r=512$
- **伪标签阈值**：$\tau=0.5$

---

### **评估指标**

#### **准确性指标**
- **最终准确率（Final Accuracy, $A_T$）**：所有任务完成后对全部已见类别的测试准确率
- **平均准确率（Average Accuracy, $A_{avg}$）**：各任务结束后的准确率平均值，衡量遗忘程度

#### **资源效率指标**
- **通信开销**：单任务中任一客户端最大上传字节数（MB）
- **运行时间**：完成一个任务所需的总墙钟时间（秒），含客户端与服务器计算
- **特征漂移（Feature Drift）**：衡量共享特征提取器是否因其他客户端更新而变化

---

### **基线方法对比**

#### **优化型FCL方法**
- **Finetune**：标准微调+FedAvg
- **FedLwF**, **FedEWC**, **FediCaRL**, **TARGET**：分别采用知识蒸馏、正则化、示例回放、合成数据等缓解遗忘

#### **分析型/轻量级FCL方法**
- **STSA**：空间-时间统计聚合，估计Gram矩阵
- **DualPrompt**, **CodaPrompt**, **PiLoRA**, **Fed-CPrompt**：基于提示（prompt）或LoRA的参数高效方法

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

| 方法 | $A_{avg}$ 提升（vs 最强基线） | 通信减少倍数 | 加速倍数 | 伪标签增益（20%标签） |
|------|-------------------------------|---------------|------------|--------------------------|
| **FedRAN** | **+4.8 pp**（CIFAR-100）<br>+4.24 pp（ImageNet-R）<br>+2.68 pp（VTAB） | **30.6–121.8×** vs 优化型<br>1.45–2.77× vs STSA | **190.3×** 平均加速<br>（最快达246.9×） | **+6.61 pp**（ImageNet-R） |

---

### **与基线方法的对比结果**

#### ✅ 准确性全面领先
- 在所有数据集和 $\beta$ 设置下，FedRAN均取得最高的 $A_{avg}$ 和 $A_T$。
- 相比最强基线 **STSA**，FedRAN在CIFAR-100上平均提升 **+2.3–4.8个百分点**。
- 优化型方法如Finetune出现严重灾难性遗忘，随任务增加准确率急剧下降。

#### ✅ 极致通信压缩
| 方法 | CIFAR-100 (MB) | ImageNet-R (MB) | VTAB (MB) |
|------|----------------|------------------|-----------|
| TARGET（代表优化型） | 4283.82 | 4306.32 | 4276.96 |
| STSA | 194.59 | 389.18 | 97.29 |
| **FedRAN** | **134.27** | **140.52** | **35.13** |

> FedRAN通信仅为TARGET的 **~1%**，且低于STSA（因其直接传输低秩结构而非估计量）

#### ✅ 超高速度
| 方法 | CIFAR-100 (s) | ImageNet-R (s) | VTAB (s) |
|------|----------------|------------------|-----------|
| Finetune | ~320 | ~560 | ~225 |
| STSA | 1.75 | 2.13 | 1.01 |
| **FedRAN** | **3.54** | **2.48** | **0.98** |

> 尽管略慢于STSA（因SVD开销），但仍比梯度法快两个数量级以上。

---

### **消融实验结果**

#### 🔍 **组件消融（Ablation Study）**
在CIFAR-100 + ViT上逐步添加组件：
- 原始ViT特征：87.71%
- + 随机投影（Random Projection）：→ 90.21%
- + ReLU非线性：→ 93.95%
- + 低秩SVD摘要：→ **93.96%**（无精度损失）

> 表明**低秩压缩几乎无损保留关键几何信息**

#### 🔍 **投影维度 $M$ 与秩 $r$ 影响**
- 增大 $r$ 或 $M$ 可提升准确率，但收益递减。
- 例如当 $M=2048$，$r$ 从256增至512带来+3.0点，但从1024到2048仅+1.0点，通信却翻倍。
- 结论：**适度的 $r$ 即可捕获大部分有用频谱信息**，实现精度-通信良好权衡。

#### 🔍 **伪标签效果（Proxy-Labeling）**
在标签稀疏条件下（20%标注）：
- CIFAR-100：+3.26 pp
- ImageNet-R：**+6.61 pp**
- VTAB：+5.76 pp

> 显示FedRAN-SSL能有效利用无标签数据，尤其在OOD场景更显著。

---

## **4. 关键结论和发现**

### **主要发现**
1. **解析式方法可以同时实现高性能与高效率**：通过紧凑的低秩统计聚合，FedRAN打破了“高准确率必须高资源消耗”的固有认知。
2. **保留主导Gram方向优于统计估计**：相较于STSA等通过一阶统计估计二阶结构的方法，FedRAN直接传输低秩SVD摘要，更稳定、偏差可控。
3. **冻结骨干+解析分类器是稳定高效的FCL范式**：避免了客户端更新带来的特征空间漂移，提升了non-IID下的鲁棒性。
4. **伪标签可在不引入额外训练的情况下增强半监督学习**：结合原型匹配与置信过滤，有效扩展可用训练信号。

---

### **方法的局限性**
- **依赖预训练骨干质量**：性能高度依赖于冻结特征提取器的泛化能力。
- **无法适应输入分布剧烈变化**：若新任务涉及全新模态或域偏移极大，固定特征可能不足。
- **理论边界为确定性误差界**：未建模伪标签噪声的影响，实际中可能存在标签错误累积风险。
- **当前聚焦于类增量设定**：尚未验证在域增量或任务增量等其他CL场景中的普适性。

---

### **未来工作方向**
1. **隐私保护扩展**：集成安全聚合（secure aggregation）以保护上传的统计量。
2. **推广至更多任务类型**：探索在回归、检测、分割等任务上的应用。
3. **纳入不确定性建模**：对伪标签引入置信度校准，防止噪声传播。
4. **动态秩调整机制**：根据任务难度自适应选择 $r$，进一步优化资源-精度平衡。
5. **理论分析伪标签噪声影响**：建立标签噪声与分类器性能之间的定量关系。

--- 

> 💡 **一句话总结**：  
> **FedRAN证明了在联邦持续学习中，“不做梯度更新”不仅可以大幅节省资源，还能获得更高准确率——通过精心设计的低秩随机特征统计聚合与闭式求解，实现了精度、速度、通信、标签效率的全面突破。**

</details>

---

### 7. [Physics-Distilled Neural Network enabled by Large Language Models for Manufacturing Process-Property Predictive Modeling](https://arxiv.org/abs/2606.11605)

**Authors**: Ge Song, Kiarash Naghavi Khanghah, Anandkumar Patel, Rajiv Malhotra, Hongyi Xu  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2606.11605v1  

#### Abstract
Predicting process-property relationships in manufacturing is often challenged by high experimental costs and the limited interpretability of complex 'black-box' models. This paper proposes a novel knowledge distillation framework designed to achieve high-accuracy predictions in data-scarce scenario...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
本论文旨在解决**制造过程-属性关系预测中普遍存在的两大挑战**：
- **数据稀缺性**：高精度制造过程的实验成本高昂，导致训练数据量小，传统数据驱动模型难以泛化。
- **模型可解释性与物理一致性差**：现有“黑箱”模型（如标准MLP、XGBoost）缺乏对物理规律的遵循，容易学习非物理相关的虚假关联。

此外，传统Physics-Informed Neural Networks (PINNs) 虽能引入物理约束，但其物理损失项依赖人工专家设计，难以扩展到新兴或复杂的制造工艺。

---

### 提出的新方法与新思路
作者提出了一种名为 **Physics-Distilled Neural Network** 的新型教师-学生框架，其核心创新在于将 **Large Language Models (LLMs)** 与 **Knowledge Distillation** 和 **Physics-Incorporated Architecture** 相结合：

1. **LLM 驱动的物理先验自动提取**  
   利用 **Retrieval-Augmented Generation (RAG)** 框架，从科学文献中自动检索并生成描述制造过程的**分析性物理方程**（analytical physics priors），作为初始物理知识。这消除了对人工推导物理方程的依赖。

2. **特权教师模型 (Privileged Teacher) 与 Graph-Masked Attention (GMA) 层**  
   教师模型不仅接受常规输入参数 $x$，还接受仅在训练时可用的“特权信息”（如高频时间序列数据 $y$）。其核心是 **GMA层**，该层利用LLM提取的物理方程的偏导数构建图邻接矩阵，**结构性地约束注意力机制**，确保特征交互符合物理规律。

3. **不对称潜空间知识蒸馏 (Latent-Manifold Distillation)**  
   将教师模型学到的、蕴含物理逻辑的**潜层表示 (latent representation)** 蒸馏到一个轻量级的学生模型中。学生模型仅需常规静态参数 $x$ 即可进行推理，实现了高性能与低部署成本的统一。

---

### 相比现有方法的优势
| 对比维度 | 本文方法 | 传统方法 |
|---------|--------|--------|
| **物理知识来源** | 自动从文献提取（LLM-RAG），无需人工干预 | 依赖专家手动编写物理方程 |
| **物理约束方式** | 结构性（通过GMA层的注意力掩码） | 通常为损失函数中的软约束（如PINNs） |
| **部署效率** | 学生模型轻量，推理速度 >6000 Hz，适合边缘部署 | 多数复杂模型难以实时部署 |
| **容错能力** | 对次优或不完整的LLM先验具有鲁棒性 | 若物理方程错误，PINNs性能急剧下降 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在**五个不同的制造过程**上进行，涵盖静态与动态输入：

| Benchmark | 过程名称 | 输入类型 | 输出目标 | 数据量 |
|----------|--------|--------|--------|------|
| 1 | FLIPMM (Flow Assisted Laser-Induced Plasma Micro-Machining) | 静态参数 | Heat-Affected Zone (HAZ) | 256 |
| 2 | MSLA (Masked Stereolithography) | 静态参数 | Ultimate Tensile Strength (UTS) | 216 |
| 3 | TADCR (Turn-Assisted Deep Cold Rolling) | 静态参数 | Hardness (HV) | 256 |
| 4 | 工业注塑成型 (Injection Molding) | 静态 + 时间序列 | Melt Cushion Volume (V) | 68 |
| 5 | MaRoReS 加工 | 静态 + 时间序列 | Radial Residual Stress (RS) | 68 |

其中，Benchmarks 1–3 使用合成数据（基于RSM方程），Benchmarks 4–5 使用真实实验数据。

---

### 实验设置与评估指标
- **训练策略**：
  - Benchmarks 1–3：采用**分层外推划分**，训练集仅取输入空间内75%区域，测试集为边界外样本，严格评估外推能力。
  - Benchmarks 4–5：因数据少，采用**5折交叉验证**以保证统计可靠性。
- **评估指标**：
  - **R² (Coefficient of Determination)**：越高越好，衡量模型解释的方差比例。
  - **RMSE (Root Mean Squared Error)**：越低越好，衡量预测偏差。
  - **推理频率 (Inference Frequency, Hz)**：衡量部署效率。

---

### 基线方法对比
与以下五类方法进行比较：
1. **纯物理模型 (Physics-only)**：直接使用LLM提取的 $f_{\text{init}}$ 或 $f_{\text{refine}}$ 方程。
2. **传统回归模型**：Random Forest (RF), XGBoost。
3. **传统神经网络**：MLP, 传统知识蒸馏 (KD)。
4. **物理融合模型**：Physics-Guided MLP (PG-MLP)，即带物理损失但无GMA层的MLP。
5. **本文提出的完整框架 (Proposed)**。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 3）
| Process | R² ↑ (fini/refine) | RMSE ↓ (fini/refine) |
|--------|-------------------|--------------------|
| Benchmark 1 (FLIPMM) | 0.949 / 0.957 | 2.410 / 2.204 |
| Benchmark 2 (MSLA) | 0.824 / 0.826 | 2.516 / 2.505 |
| Benchmark 3 (TADCR) | 0.957 / 0.964 | 3.240 / 3.109 |
| Benchmark 4 (注塑) | 0.963 / 0.989 | 0.169 / 0.099 |
| Benchmark 5 (MaRoReS) | 0.612 / 0.725 | 0.498 / 0.347 |

> 所有基准上，本文方法均取得最优性能，尤其在数据极少的Benchmark 5上优势显著。

---

### 与基线方法的对比结果
- 在 **FLIPMM (B1)** 上，即使使用**次优的初始物理方程**（$f_{\text{init}}, R^2=0.689$），本文方法仍能达到 **R²=0.949**，远超所有基线（MLP最高仅0.896）。
- 在 **MSLA (B2)** 上，物理方程本身性能较差（$f_{\text{refine}}, R^2=0.708$），但本文方法通过蒸馏提升了至 **0.826**，证明其能从弱先验中提取有效指导。
- 在 **MaRoReS (B5)** 上，传统MLP表现极不稳定（R²最低降至0.182），而本文方法保持稳定高精度（R²=0.725），且**interquartile range最窄**，显示卓越的泛化稳定性。

---

### 消融实验结果
通过对比不同变体，验证各组件作用：
- **vs KD (传统蒸馏)**：本文方法优于KD，说明**GMA层提供的结构性物理引导**比单纯的知识转移更有效。
- **vs PG-MLP**：本文方法优于PG-MLP，说明**特权信息 + 潜空间蒸馏**比仅加物理损失更能提升性能。
- **推理速度**：学生模型推理频率达 **>6000 Hz**（如FLIPMM达6537 Hz），满足实时工业监控需求。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM可作为有效的“物理知识提取器”**：通过RAG框架自动从文献中提取物理方程，可替代人工建模，显著提升方法的可扩展性。
2. **结构性物理约束优于损失级约束**：GMA层通过图注意力掩码强制模型学习物理变量间的正确依赖关系，比PINNs式的损失项更鲁棒。
3. **知识蒸馏可桥接训练与部署鸿沟**：利用特权信息（如传感器信号）训练教师模型，再蒸馏到仅需标准参数的学生模型，实现了**高性能与低成本部署的统一**。
4. **框架具有强容错性**：即使LLM提取的物理先验不准确或不完整，框架仍能通过数据驱动部分补偿误差，维持较高预测精度。

---

### 方法的局限性
- **输出为标量**：当前框架仅预测单一属性值，无法处理场量输出（如应力场、温度场）。
- **点预测无不确定性量化**：模型输出为确定值，未提供预测置信度，在高风险控制场景中存在隐患。
- **依赖文献质量**：若领域文献稀疏或错误，LLM提取的先验可能误导模型。

---

### 未来工作方向
1. 扩展至**场量预测任务**（如残余应力分布、变形图）。
2. 引入**不确定性量化机制**（如贝叶斯神经网络），支持风险感知的制造控制。
3. 探索多任务学习框架，实现多个制造属性的联合建模与优化。

</details>

---

### 8. [Mental-R1: Aligning LLM Reasoning for Mental Health Assessment](https://arxiv.org/abs/2606.13176)

**Authors**: Xin Wang, Boyan Gao, Yibo Yang, David A. Clifton  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.13176v1  

#### Abstract
Mental health problems such as anxiety, depression, and suicide remain urgent global challenges, where timely and accurate assessment is critical for effective intervention. Recently, large language models have been explored for mental health assessment. However, existing general-purpose post-traini...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Mental-R1: Aligning LLM Reasoning for Mental Health Assessment 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

当前基于 **Large Language Models (LLMs)** 的心理健康评估（Mental Health Assessment, MHA）研究多采用通用的后训练方法（如 Supervised Fine-Tuning, SFT 或标准 Reinforcement Learning, RL），这些方法未能模拟真实临床实践中**心理专家的认知推理过程**，导致模型推理路径不可靠、缺乏可解释性，并可能因依赖表面关键词而做出错误判断。

具体而言，现有方法存在以下不足：
- 忽视人类认知从“不确定性探索”到“确定性决策”的动态演化过程；
- 缺乏理论支撑的分阶段推理结构，难以捕捉个体对刺激的内在认知评价（cognitive appraisal）；
- 在多任务联合训练中面临类别与数据集不平衡问题。

### **提出了什么新方法或新思路**

本文提出 **Cognitive Relative Policy Optimization (CRPO)**，一种专为心理健康领域设计的强化学习框架，旨在将 LLM 的推理过程与人类认知机制对齐。其核心创新包括：

#### ✅ **Stage-wise Entropy Regularization (SER)**
- 引入**阶段依赖的熵正则化机制**，在策略优化目标中显式建模不同推理阶段的不确定性。
- 早期阶段鼓励高熵（多样化探索），后期阶段逐步降低熵值以促进确定性输出，模仿人类从“探索→收敛”的认知转变。

#### ✅ **Theory-Grounded Cognitive Reasoning Stages**
- 基于**Cognitive Appraisal Theory**构建五阶段推理流程：
  1. **Stimulus**：识别潜在刺激事件；
  2. **Primary Appraisal**：评估该刺激是否具有威胁性；
  3. **Secondary Appraisal**：判断个体是否有足够应对资源；
  4. **Reaction**：推断情绪与行为反应；
  5. **Mental State**：最终诊断心理状态。
- 此结构使模型避免仅凭负面词汇草率下结论，提升推理深度与可解释性。

#### ✅ **Balanced Answer Reward**
- 针对多任务训练中的**类别不平衡**与**数据集规模差异**，设计加权奖励函数：
  $$
  r_a \propto \sqrt{w_c \cdot w_d}
  $$
  其中 $w_c$ 和 $w_d$ 分别为类频率与数据集频率的倒数权重，确保稀有类与小数据集获得更强学习信号。

### **相比现有方法的优势**

| 维度 | CRPO优势 |
|------|--------|
| **推理可靠性** | 显式建模认知不确定性演化，增强复杂案例的鲁棒性 |
| **可解释性** | 输出遵循心理学理论支持的推理轨迹，便于人工审核 |
| **泛化能力** | 多任务联合训练 + 平衡奖励，缓解数据偏差影响 |
| **性能表现** | 在多个公开数据集上显著优于主流 SFT 与 RL 方法 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

共使用 **8个全公开的人工标注心理健康数据集**，涵盖五大心理障碍与多种任务形式：

| 数据集 | 任务类型 | 类别数 | 样本量 | 主要心理问题 |
|-------|--------|--------|--------|-------------|
| **Dreaddit** | 二分类 | 2 | 3,553 | Stress |
| **DATD** | 二分类 | 2 | 1,050 | Anxiety/Depression |
| **LT-EDI** | 多分类 | 3 | 10,251 | Depression Level |
| **DepSeverity** | 多分类 | 4 | 3,553 | Depression Severity |
| **SDCNL** | 二分类 | 2 | 1,895 | Suicide Ideation |
| **RSD** | 多分类 | 4 | 1,265 | Suicide Risk Severity |
| **FIG** | 二分类 | 2 | 5,633 | Loneliness |
| **LID** | 多分类 | 4 | 498 | Loneliness Intensity |

> 所有数据集均为开源，作者强调这是目前最全面且透明的 MHA benchmark。

### **实验设置**

- **基础模型**：Qwen3-8B
- **训练方式**：Reinforcement Learning（无 Critic 架构）
- **优化器**：Deepspeed ZeRO-2，bf16精度
- **超参数**：
  - 学习率：5e-6
  - Batch Size：每设备4，梯度累积16步
  - SER 参数：$M=0.06$, $T=3.5$
- **Prompt Length**：最大512 tokens
- **生成数量**：每个 prompt 采样4条输出用于相对比较

### **评估指标**

- **主指标**：**Weighted F1-score**（处理类别不平衡）
- **辅指标**：Accuracy
- 报告五次独立运行的均值±标准差

### **基线方法对比**

#### **RL 基线（同属 Critic-Free RL）**
- SFT（监督微调）
- RLOO, ReMax, Reinforce++, DAPO, GRPO

#### **LLM 基线**
- **专用模型**：Mental-LLM, Mental-GLM, Mental-Llama
- **开源模型**：Gemma-2-SFT, Llama-3.1-SFT
- **商用大模型**：GPT-3.5, GPT-4o, GPT-5, DeepSeek-V3/R1

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **CRPO vs. RL/SFT 基线（Table III）**

| 方法 | 平均 Accuracy ↑ | 平均 Weighted F1 ↑ |
|------|------------------|--------------------|
| SFT | — | 54.26% |
| GRPO（最强RL基线） | — | 53.05% |
| **CRPO（本文）** | **+12.7 pts** | **+13.6 pts** |

> CRPO 在所有8个数据集上均取得最佳性能，平均比最强 RL 基线 **DAPO 提升 10.4 F1 点**。

特别在不平衡严重的 **RSD**（自杀风险分级）和 **LID**（孤独强度）数据集上，分别提升 **17.5** 和 **16.7 F1 点**。

#### ✅ **Mental-R1 vs. 各类 LLM（Table IV）**

| 模型类别 | 平均 Weighted F1 提升 |
|----------|------------------------|
| 最佳专用模型（Mental-Llama） | +13.6 pts |
| 最佳开源模型（Llama-3.1-SFT） | +14.4 pts |
| 最佳商用模型（GPT-5） | +11.9 pts |

> **Mental-R1 是首个在心理健康任务上全面超越 GPT-4o/GPT-5 的开源定制模型**。

### **消融实验结果（Ablation Study）**

| 变体 | Weighted F1 下降幅度 |
|------|------------------------|
| w/o SER（移除阶段熵正则） | ⬇️ 最大下降（~7–8 pts） |
| w/o Stages（移除认知阶段） | ⬇️ 明显下降（~3–5 pts） |
| w/o Balanced Reward | ⬇️ 在 RSD/LID 上严重退化 |

> 表明 **Stage-wise Entropy Regularization 是最关键组件**，验证了“认知对齐”设计的有效性。

### **推理能力专项评测（Figure 3）**

- 使用 GPT-5 对测试样本打“推理难度分”，选取 top 20% 高难度样本构成 **reasoning-focused set**。
- 结果显示：**Mental-R1 在复杂样本上比 GPT-5 高出约 15.6 F1 点**。
- 小样本标准差更大，说明此设定更具挑战性，突显 CRPO 对深层推理的增强效果。

---

## 4. 关键结论和发现

### **主要发现**

1. 🔍 **认知对齐能显著提升 LLM 推理质量**  
   将人类认知的“不确定性→确定性”动态引入 RL 目标函数（即 SER），可有效引导模型先广泛探索线索、再聚焦整合证据，从而提高判断准确性。

2. 🧠 **理论驱动的推理结构增强可解释性与稳健性**  
   基于 Cognitive Appraisal Theory 设计的五阶段推理链，使模型不再依赖表面关键词（如“sad”、“alone”），而是理解个体如何解读事件及其应对能力，减少误判。

3. 📊 **平衡奖励机制对抗数据偏见**  
   在多任务联合训练中，balanced answer reward 成功缓解了小数据集与罕见类别的边缘化问题，尤其提升了 RSD 与 LID 的表现。

4. 🏆 **Mental-R1 成为新的 SOTA 模型**  
   不仅超越所有专用 Mental-LLM，还在多个任务上击败 GPT-4o 和 GPT-5，证明了领域定制化训练路径的巨大潜力。

### **方法的局限性**

- **依赖系统提示强制结构化输出**：需通过 `<think>`、`<stimulus>` 等标签约束生成格式，若用户绕过提示可能导致结构失效。
- **阶段划分固定**：五个认知阶段为预定义顺序，灵活性受限，无法适应非线性思维过程。
- **计算成本较高**：每次推理需生成完整推理链，延迟高于直接预测模型。
- **未涉及真实临床验证**：实验基于文本数据集，尚未在真实医疗场景中部署验证安全性与有效性。

### **未来工作方向**

1. **扩展至多模态输入**：结合语音、面部表情等非语言信号进行综合心理评估；
2. **个性化认知建模**：根据不同人格特质调整认知阶段权重；
3. **在线自适应推理**：允许模型根据上下文动态跳转或重复某些推理阶段；
4. **临床协作接口设计**：开发供心理医生使用的交互式辅助诊断工具；
5. **伦理与安全机制建设**：防止滥用、确保隐私保护与责任归属。

---

> 💡 **一句话总结**：  
> 本文提出的 **CRPO** 框架首次将人类认知动态与心理学理论系统地融入 LLM 强化学习过程，训练出的 **Mental-R1** 在多项心理健康任务上实现 SOTA 性能，不仅提升了准确率，更增强了模型的**可解释性**与**临床可信度**，为 AI 辅助精神健康干预提供了新范式。

</details>

---

### 9. [From Verdict to Process: Agentic Reinforcement Learning for Multi-Stage Fact Verification](https://arxiv.org/abs/2606.13262)

**Authors**: Rongxin Yang, Shenghong He, Siyuan Zhu, Chao Yu  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.13262v1  

#### Abstract
Recent approaches combining Large Language Models (LLMs) with retrieval-augmented reasoning have shown promise for automated fact verification. To process complex claims, these verification pipelines typically execute multi-stage workflows that coordinate tightly coupled modules, including claim dec...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*From Verdict to Process: Agentic Reinforcement Learning for Multi-Stage Fact Verification*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的多阶段事实验证系统（如 InFact、HerO）通常将**声明分解、证据检索、答案生成和最终判决**等模块独立优化或依赖固定启发式规则进行协调。这种分离导致中间决策（如问题生成）与最终目标（准确判断真实性）不一致，限制了各阶段之间的自适应协同，从而影响整体验证效果。

此外，传统方法仅依赖最终的真实性标签作为监督信号，面临**稀疏且延迟的奖励**问题，难以有效分配信用（credit assignment），即无法判断哪个中间步骤对成功或失败负责。

---

### 🚀 提出的新方法：**ProFact**
ProFact 是一个基于 **Agentic Reinforcement Learning** 的端到端优化框架，用于多阶段事实验证任务。其核心思想是将整个验证过程建模为一个长周期的决策过程（long-horizon decision-making），由统一策略（unified policy）控制所有阶段行为。

#### 主要创新点：
- **Agentic Verification Framework**  
  将事实验证视为一个代理（agent）在 MDP 框架下的交互过程，统一管理以下动作：
  - Claim decomposition → 生成验证问题（QUESTION stage）
  - Evidence retrieval → 调用搜索工具获取证据（SEARCH stage）
  - Answer generation → 基于证据生成回答
  - Verdict prediction → 输出最终真实性标签（VERDICT stage）

- **Process-Aware Trajectory Optimization**  
  引入**过程感知奖励函数**（process-aware reward），提供密集的阶段性反馈，解决稀疏奖励问题：
  - 在 QUESTION 阶段：使用 METEOR 分数衡量生成问题与黄金问题的匹配度
  - 在 SEARCH 阶段：衡量生成的问答对与标注过程的一致性
  - 在 VERDICT 阶段：使用指示函数判断预测标签是否正确
  
  这种设计将终端奖励转化为**阶段级监督信号**，显著提升信用分配能力。

- **End-to-End Joint Optimization**  
  使用 **Group-Relative Policy Optimization (GRPO)** 对完整验证轨迹进行联合优化，使早期决策（如问题分解）能主动适应后期目标（如判决准确性），实现跨阶段自适应协调。

---

### 🔍 相比现有方法的优势
| 方面 | 现有方法（如 InFact, HerO） | ProFact |
|------|----------------------------|--------|
| 架构 | 多个独立训练/提示的模块 | 单一统一策略端到端优化 |
| 协调机制 | 固定流程或启发式调度 | 动态学习最优路径 |
| 学习信号 | 仅最终判决标签（稀疏） | 各阶段密集的过程奖励 |
| 决策一致性 | 中间步骤可能偏离最终目标 | 所有行为服务于整体回报最大化 |
| 推理效率 | 流程冗长，上下文膨胀 | 更紧凑的推理路径，更低 token 消耗 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **AVeriTeC** [13]：真实世界声明验证基准数据集
  - 包含自然语言声明、真实性标签（`SUPPORTED`, `REFUTED`, `NOT ENOUGH EVIDENCE`, `CONFLICTING EVIDENCE`）
  - 提供人工标注的验证问题-答案对（gold QA pairs）
  - 自带静态网页文档知识库（evidence store），支持可复现的开放域检索

---

### ⚙️ 实验设置
- **Backbone Models**（共4个开源模型）：
  - Qwen2.5-3B-Instruct
  - Qwen2.5-7B-Instruct
  - Qwen3-4B-Instruct-2507
  - Qwen3-8B-Instruct
- **训练方式**：Post-training using GRPO
- **采样策略**：每条声明采样 8 条轨迹构成 group
- **Batch 设置**：mini-batch=32, micro-batch=4
- **KL 正则化系数**：β = 0.001
- **最大交互步数**：T = 12
- **解码策略（eval）**：deterministic decoding (temperature=0)
- **最多生成问题数**：5 个
- **每次检索返回 top-3 证据项**

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Q-only METEOR** | 衡量生成的验证问题与黄金问题的文本相似度，反映 claim decomposition 能力 |
| **Q&A METEOR** | 衡量生成的问答对与标注过程的整体对齐程度 |
| **Accuracy** | 最终真实性分类准确率 |
| **AVeriTeC Score** | 官方综合评分：要求证据质量得分 > 0.25 **且** 判决正确，联合评估证据充分性和决策准确性 |

---

### 🆚 基线方法对比
| 方法 | 类型 | 特点 |
|------|------|------|
| **Consistency** | Prompting-based | 使用固定黄金问答作为输入，通过一致性聚合预测判决 |
| **InFact** [12] | Workflow-based | 六阶段流水线，工程化 prompt 设计，独立模块执行 |
| **HerO** [21] | Strong LLM-based system | 使用假设文档增强检索，微调 LLM 进行判决预测 |
| **w/o PR** | Ablation of ProFact | 移除中间过程奖励，仅保留最终判决奖励 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1）

| Method | Backbone | Q-only METEOR | Q&A METEOR | Accuracy | AVeriTeC Score |
|--------|----------|----------------|-------------|-----------|----------------|
| **ProFact** | Qwen2.5-3B | **46.01** | **31.14** | **68.80** | **47.80** |
| **ProFact** | Qwen2.5-7B | 45.26 | 30.36 | **70.20** | **48.00** |
| **ProFact** | Qwen3-4B | 46.08 | 30.11 | 69.60 | 46.20 |
| **ProFact** | Qwen3-8B | 46.05 | 30.02 | 70.28 | 46.40 |
| HerO | Qwen3-8B | 44.29 | 30.54 | 69.20 | 44.40 |
| InFact | Qwen3-4B | 40.09 | 29.12 | 56.91 | 45.29 |
| Consistency | Qwen3-4B | 29.65 | 23.86 | 51.00 | 21.40 |

> ✅ **结论**：ProFact 在所有 backbone 上均取得最佳 Accuracy 和 AVeriTeC Score，尤其在较小模型上优势更明显。

---

### 🔬 消融实验结果（w/o PR）
移除过程奖励（process rewards）后性能显著下降：

| Method | Backbone | Q-only METEOR | Q&A METEOR | Accuracy | AVeriTeC Score |
|--------|----------|----------------|-------------|-----------|----------------|
| **ProFact** | Qwen2.5-3B | 46.01 | 31.14 | 68.80 | 47.80 |
| **w/o PR** | Qwen2.5-3B | 38.98 ↓ | 27.20 ↓ | 64.60 ↓ | 34.40 ↓ |

> ❗ **说明**：仅靠最终判决标签不足以有效指导中间行为的学习，**过程感知奖励对于信用分配至关重要**。

---

### ⏱️ 效率分析（Table 2）
| Method | Backbone | Time per Claim (s) | Total Input Tokens | Total Output Tokens |
|--------|----------|---------------------|--------------------|---------------------|
| InFact | Qwen3-8B | 288.00 | 34.04M | 12.63M |
| **ProFact** | Qwen3-8B | **22.06** | **6.59M** | **2.01M** |

> ✅ **ProFact 推理速度快约 13 倍，token 消耗减少超过 80%**  
> 原因：简化流程（去除重写）、上下文隔离、学习更高效检索路径。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **端到端轨迹优化优于分阶段独立优化**  
   ProFact 通过统一策略联合优化多阶段行为，实现了更好的跨阶段协调，提升了整体验证性能。

2. **过程感知奖励显著改善信用分配**  
   引入阶段级 METEOR-based 奖励，使得 RL 能够识别哪些中间行为有助于最终成功，解决了稀疏奖励难题。

3. **ProFact 在性能与效率上双重领先**  
   不仅 Accuracy 和 AVeriTeC Score 最高，而且推理时间短、token 成本低，具备更强实用性。

4. **大模型不一定更好**  
   实验显示 larger backbones 并未带来单调性能提升，甚至出现“逆向缩放”（inverse scaling）现象，表明参数知识可能干扰证据依赖推理。

5. **GRPO 是适合该任务的 RL 算法**  
   相比 PPO、DAPO、GiGPO，GRPO 在多阶段验证中表现最优，因其 group-relative 优势估计稳定且无需 critic network。

---

### ⚠️ 局限性
- 当前框架依赖预构建的知识库（kNN index），尚未完全集成实时搜索引擎。
- 过程奖励依赖黄金 QA 注释，在无标注场景下应用受限。
- 搜索动作目前为结构化调用，尚未扩展至复杂工具组合或多跳查询规划。

---

### 🔮 未来工作方向
- 扩展至动态网络搜索环境（real-time web search）
- 探索弱监督或无监督下的过程奖励建模
- 引入更复杂的工具调用机制（如多跳检索、反向验证）
- 应用于其他多阶段推理任务（如法律论证、科学假设检验）

---

## 总结
> **ProFact 成功地将事实验证从“以判决为中心”转向“以过程为中心”，通过 Agentic RL 实现了多阶段行为的端到端联合优化。其实验结果证明：过程感知的轨迹学习不仅能提高准确性，还能大幅提升推理效率，为下一代自动化事实核查系统提供了新范式。**

</details>

---

### 10. [Low-Latency Real-Time Audio Game Commentary System via LLM-Based Parallel Text Generation](https://arxiv.org/abs/2606.13322)

**Authors**: Ryota Kawamatsu, Anum Afzal, Yuki Saito, Shinnosuke Takamichi, Graham Neubig, Katsuhito Sudoh, Hiroya Takamura, Tatsuya Ishigaki  
**Category**: cs.CL  
**Published**: 2026-06-12  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2606.13322v1  

#### Abstract
We present a low-latency real-time audio game commentary system that generates spoken commentary directly from live gameplay video. In this end-to-end setting, a key bottleneck is accumulated waiting time; conventional pipelines capture frames, generate text, and synthesize speech sequentially for e...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Low-Latency Real-Time Audio Game Commentary System via LLM-Based Parallel Text Generation*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
在基于实时 gameplay 视频生成音频评论（audio commentary）的端到端系统中，**latency（延迟）** 是一个关键瓶颈。传统流水线采用**顺序处理模式**：必须等待当前语音播放结束后才开始下一帧的文本生成，导致**utterance 之间出现长时间沉默（inter-utterance silence）**，严重影响用户体验和评论节奏的自然性。

### 🚀 提出的新方法与创新思路
本文提出了一种**低延迟实时音频游戏评论系统**，其核心创新在于两个机制：

1. **Parallel Text Generation（并行文本生成）**
   - 在当前语音仍在播放时，系统即对新到达的视频片段进行 LLM 文本生成，并将多个候选 utterance 缓存在 buffer 中。
   - 当前语音一结束，立即从 buffer 中选择一条候选文本进行 TTS 合成，实现“无缝衔接”，避免等待。

2. **Lightweight Video Delay Control（轻量级视频延迟控制）**
   - 为缓解因生成延迟导致的音画不同步，在服务器端主动延迟视频流输出，使生成的语音与画面保持时间对齐。

> 💡 这种设计打破了传统 pipeline 的严格串行依赖，显著降低了感知延迟。

### ⚖️ 相比现有方法的优势
- **大幅减少沉默时间**：将平均 inter-utterance silence 从 9.6 秒降至 0.3 秒。
- **提升节奏自然度**：生成的 speaking/silence 模式更接近专业人类评论员（mIoU 提升超 40%）。
- **无需复杂调度策略**：三种简单的 buffer selection policy（Latest/Oldest/Random）均表现良好，说明性能增益主要来自并行机制本身。
- **适用于高动态游戏场景**：在快节奏游戏如 *Super Smash Bros. Ultimate* 上验证有效。

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **Smash Corpus [Saito et al., 2020]** 中的 8 段 gameplay 视频。
- 游戏目标为 *Super Smash Bros. Ultimate* —— 典型的高速动作对抗类游戏，适合测试低延迟能力。
- 收集了熟练评论员对该视频的手动评论作为 human reference。

### ⚙️ 实验设置
- **视频输入**：25 fps，每 32 帧组成一个 segment（$N=32$），实时上传至 commentary server。
- **模型配置**：
  - **LLM**: GPT-4.1-mini（base64 编码输入）
  - **TTS**: 采用 [Iura et al., 2025] 的 excitement-inducing TTS 系统
- **参数调节**：`max_new_tokens` 设置为 {20, 40, 60, 80, 100}，用于控制生成长度与时长匹配。

### 🎯 评估指标
| 类别 | 指标 | 描述 |
|------|------|------|
| **Timing Behavior** | Cumulative silence<br>Mean inter-utterance silence<br>Utterance length | 衡量沉默总时长与分布 |
| | mIoU（mean Intersection over Union） | 将 commentary 流建模为 1Hz 二值序列（1=说话, 0=静默），计算与 human pattern 的重合度 |
| **Content Adequacy** | ROUGE（Recall） | 在固定 10 秒窗口内比较生成文本与参考文本的内容相似性 |
| **主观质量** | MOS（Mean Opinion Score）<br>(Q1–Q3) | 用户研究评分：<br>Q1: 节奏自然性<br>Q2: 音画对齐<br>Q3: 整体质量 |

### 🔁 基线方法对比
| 方法 | 设计特点 |
|------|----------|
| **After-Audio (Baseline)** | 完全串行：等当前语音播放完再启动下一轮生成 |
| **After-Text (Semi-Sequential)** | 半并行：文本生成完成后即启动下一轮生成，但仍需等 TTS 完成才能播放 |
| **Our System (Parallel)** | 并行生成 + Buffering：<br>- Parallel Latest<br>- Parallel Oldest<br>- Parallel Random |

所有方法均启用相同的 **video delay control** 以公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 1 & Figure 3）

| Method | Cumulative Silence (s) | Mean Silence (s) | mIoU |
|--------|-------------------------|------------------|-------|
| Human | 59.7 ± 11.3 | 1.7 ± 0.4 | – |
| After-Audio | 134.0 ± 5.2 | **9.5 ± 0.9** | 0.01 ± 0.06 |
| After-Text | 125.5 ± 5.8 | 6.8 ± 0.9 | 0.10 ± 0.08 |
| **Parallel (Latest)** | 24.7 ± 6.6 | **0.4 ± 0.1** | 0.59 ± 0.04 |
| **Parallel (Oldest)** | **18.9 ± 3.1** | **0.3 ± 0.1** | **0.60 ± 0.04** |
| **Parallel (Random)** | 19.1 ± 3.6 | **0.3 ± 0.1** | **0.60 ± 0.03** |

> ✅ 结果显示：
- 并行方法将 **平均沉默时间降低 97%**（9.5s → 0.3s）
- mIoU 接近人类水平（~0.6 vs human ~0.7 estimated from figure）
- Oldest 与 Random 策略略优于 Latest，但差异不大

### 📈 内容质量与生成长度影响（Table 2 & Figure 2）
- `max_new_tokens=60` 时生成的 utterance 长度最接近 human（28.2 tokens vs 25.9）
- ROUGE Recall 显示并行方法在所有设置下均优于基线（Figure 2）
- 更长的生成允许更完整描述事件，但也可能引入冗余

### 👥 用户研究结果（Figure 3）
- 所有并行方法在 Q1（节奏自然性）、Q2（音画对齐）、Q3（整体质量）上显著优于基线（p < 0.01）
- MOS 提升明显，尤其在 Q1 上差距最大，表明减少沉默直接改善感知节奏

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **并行生成是降低 latency 的关键**：通过提前生成候选文本并缓存，可几乎消除 inter-utterance silence。
2. **selection policy 影响较小**：Latest / Oldest / Random 表现相近，说明系统鲁棒性强，无需复杂决策逻辑。
3. **video delay control 保障 temporal alignment**：轻微延迟视频即可吸收生成延迟，维持音画同步。
4. **节奏自然性可通过 mIoU 量化衡量**：该指标能有效反映 speaking/silence 模式的专业程度。

### ⚠️ 局限性
- 当前系统依赖高性能 LLM 和 TTS，部署成本较高。
- 缓冲机制可能导致轻微的信息过期（如 Latest 策略可能错过最新状态）。
- 未处理多说话人或情绪切换等高级表达维度。
- 实验集中在单一游戏类型（格斗类），泛化性有待验证。

### 🔮 未来工作方向
- 引入 **content planning 与 discourse management**，提升评论连贯性和叙事结构。
- 探索 **adaptive buffer management**，根据游戏状态动态调整生成频率与长度。
- 结合 **emotion-aware TTS** 实现更具感染力的 commentary。
- 扩展至更多游戏类型（如 MOBA、FPS）及真实直播平台集成。

---

> 🔗 **演示视频**：[https://youtu.be/pmrRUlvav8M](https://youtu.be/pmrRUlvav8M)  
> 📄 **原文链接**：arXiv:2606.13322 [cs.CL]

</details>

---

### 11. [Physics-Guided Spatiotemporal Learning for Coastal Wave Peak Period Estimation from Video](https://arxiv.org/abs/2606.13302)

**Authors**: Abubakar Hamisu Kamagata, Dharm Singh Jat, Attlee Munyaradzi Gamundani, Abhishek Srivastava, Paramasivam Saravanakumar  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.13302v1  

#### Abstract
Wave parameters in the nearshore are crucial for coastal engineering, shoreline protection, marine hazard assessment, and coastal management for climate resilience. Traditional monitoring systems like buoys and radar platforms offer accurate monitoring but can have high installation and maintenance ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Physics-Guided Spatiotemporal Learning for Coastal Wave Peak Period Estimation from Video

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统近岸波浪监测系统（如浮标、雷达）虽然精度高，但存在**成本高昂、维护困难、空间覆盖有限**等问题。基于视频的深度学习方法虽具潜力，但普遍存在以下缺陷：
- 缺乏物理可解释性（physically interpretable）
- 依赖中间代理变量（如 timestack、celerity inversion、spectral preprocessing）
- 对光照、相机角度等环境变化敏感
- 缺少标准化基准和专家标注数据集

本文旨在解决上述问题，提出一个**端到端（end-to-end）直接从原始单目海岸视频中估计波峰周期 $T_p$** 的框架。

---

### 🚀 提出的新方法与创新思路

作者提出了一个 **Physics-Guided Deep Spatiotemporal Learning Framework**，其三大核心创新为：

#### （1）自动化 ROI 检测（Automated Temporal-Variance-Based ROI Detection）
- 利用像素时间方差（temporal variance）自动识别活跃波区（surf zone），排除静态陆地/天空干扰。
- 方法具有**架构无关性（architecture-agnostic）**，适用于 CNN、RNN 和 Transformer 等多种模型。
- 显著提升信噪比，减少 RMSE（从 1.16s → 0.85s）。

#### （2）三阶段 Sim-to-Real 迁移学习策略（Tri-Phase Transfer Learning）
采用分阶段预训练策略缓解标注数据稀缺问题：
- **Phase 0（合成预训练）**：基于 **Airy Wave Theory** 合成理想波浪视频进行物理先验建模；
- **Phase 1（银标签预训练）**：在大规模“银集”（Silver Set）上使用光学流生成伪标签进行真实世界噪声适应；
- **Phase 2（金标签微调）**：在小规模高质量“金集”（Golden Set）上由专家标注进行精细对齐。

该策略实现从物理模拟到现实世界的平滑过渡，显著加速收敛并提高泛化能力。

#### （3）物理引导损失函数（Physics-Informed Regularization）
设计复合损失函数：
$$
\mathcal{L} = \mathcal{L}_{data} + \lambda \cdot \mathcal{L}_{physics}
$$
其中 $\mathcal{L}_{physics}$ 引入软约束，确保预测的 $T_p$ 落在物理合理范围 [8–20] 秒内（对应深水至中等水深波频 0.05–0.125 Hz）。通过调节 $\lambda$ 控制物理约束强度。

---

### ⚖️ 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **物理一致性** | 预测结果符合线性波理论，避免非物理解（如 <2s 或 >20s） |
| **端到端能力** | 不依赖 timestack 构造或频谱分析等中间步骤 |
| **鲁棒性** | 自动 ROI 抑制光照、视角变化影响；多阶段训练增强抗噪能力 |
| **实用性** | 模型轻量化（如 TinyWaveNet 仅 0.28M 参数），支持边缘设备实时部署（延迟低至 0.53ms） |

---

## 2. 核心实验方法和设置

### 📚 数据集

构建了两级数据策略以应对标注稀缺问题：

| 类型 | 名称 | 视频数量 | 片段数（60帧滑窗） | 标注方式 | 特点 |
|------|------|----------|---------------------|-----------|-------|
| Tier-1 | Golden Set（金集） | 13 场景 | 6,926 | 专家手动标注 + TimeStack 分析 | 高质量、多样化地貌（沙滩、岩岸、防波堤）、光照条件丰富 |
| Tier-2 | Silver Set（银集） | 20 场景 | 10,655 | 光学流自动生成伪标签 | 大规模但含噪声，用于 Phase 1 预训练 |

数据来源包括 GitHub、Zenodo、Surfline、Pixabay、iStockPhoto 等公开平台及实地拍摄。

---

### 🛠 实验设置与评估指标

#### 模型架构对比
- **主干网络**：
  - **Transformer-based**: LtViViT（轻量版 Video Vision Transformer）
  - **Recurrent-Convolutional**: PtAttnCNN (TinyWaveNet), PtLSTM, WaveConvLSTM
- **输入格式**：60帧 × 64×64 RGB 视频片段（约2秒）

#### 评估指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **RMSE**, **MAE** | 回归误差标准 | 衡量点对点预测精度 |
| **Scatter Index (SI)** | $ \text{SI} = \frac{\text{RMSE}}{\bar{T}_p} $ | 归一化误差，反映相对波动 |
| **Willmott Skill Score (WS)** | $ 1 - \frac{\sum{(T_{p,i}-\hat{T}_p)^2}}{\sum{(|T_{p,i}-\bar{T}_p| + |\hat{T}_p-\bar{T}_p|)^2}} $ | 衡量趋势拟合能力，越接近1越好 |

---

### 🔁 基线方法对比
- **传统方法**：基于多视图光流 + FFT 的频谱分析流程
- **深度学习基线**：
  - ResNet18 + LSTM（TinyWaveNet）
  - ConvLSTM / WaveConvLSTM
  - ViViT 变体（LtViViT）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| 模型 | RMSE (s) | MAE (s) | SI | WS | 参数量 (M) | 推理延迟 (ms) |
|------|----------|--------|-----|-----|------------|---------------|
| **LtViViT** | **0.7692** | 0.5580 | 0.1047 | 0.9691 | 2.36 | 1.20 |
| **PtAttnCNN ($\lambda=5.0$)** | 1.4898 | 1.7596 | **0.0892** | **0.9811** | 0.28 | **0.53** |
| PtAttnCNN (baseline) | 1.7599 | — | 0.0924 | 0.9767 | 0.28 | 0.63 |
| PtLSTM | 1.1566 | — | 0.0993 | 0.9747 | 12.52 | 3.61 |
| WaveConvLSTM | 1.7599 | — | 0.1068 | 0.9664 | 0.24 | 57.01 |

> 注：$\lambda=5.0$ 表示强物理正则化权重。

---

### 🔍 与基线方法对比结果

- **LtViViT 在瞬时预测精度上最优（最低 RMSE = 0.7692s）**  
  → 得益于 factorized temporal attention，能有效捕捉局部时空动态。

- **PtAttnCNN ($\lambda=5.0$) 在海洋学技能指标上表现最佳（最高 WS = 0.9811，最低 SI = 0.0892）**  
  → 更适合长期趋势跟踪任务，具备更强的操作可用性（operational oceanographic skill）。

- **WaveConvLSTM 尽管参数少，但推理延迟极高（57ms）**  
  → 不适用于实时边缘部署。

- **所有模型在引入物理正则化后均表现出更稳定的趋势跟随能力**，尤其在 $T_p \in [8,10]$s 区间误差显著降低。

---

### 🔬 消融实验结果（Ablation Study）

#### （1）物理正则化权重 $\lambda$ 敏感性分析（Table 7 & Fig. 13）

| $\lambda$ | SI | WS |
|----------|----|-----|
| 0.0（无约束） | 0.1099 | 0.9664 |
| 0.1（弱约束） | 0.0972 | 0.9707 |
| 1.0（中等） | 0.1010 | 0.9696 |
| **5.0（强约束）** | **0.0892** | **0.9811** |

- 发现非单调恢复现象：适度正则化（$\lambda=0.1$）即可提升性能，而 $\lambda=1.0$ 时略有下降，最终在 $\lambda=5.0$ 达到峰值。
- 表明**强物理约束有助于抑制噪声标签带来的偏差**，尤其是在高频震荡区域。

#### （2）组件消融（Table 8）

| 模型配置 | RMSE (s) | MAE (s) |
|---------|----------|--------|
| Baseline (no physics) | 2.4959 | 1.7136 |
| + Physics Loss ($\lambda=5.0$) | 2.4607 | 1.7596 |
| LtViViT (hybrid pretrained) | **0.7692** | **0.5580** |

- 单独添加物理损失即可将 RMSE 下降约 35ms，并消除大量 <8s 的非物理解。
- 最终性能是**合成预训练 + 混合真值 + 物理损失三者协同作用的结果**。

#### （3）预测分布对比（Fig. 3）
- **Standard CNN** 出现大量低于 8s 的预测（违反物理规律）；
- **Physics-Guided Model ($\lambda=5.0$)** 成功将预测集中在 [8–20]s 内，且峰值更尖锐集中于 8.5s 附近，表明模型学会关注真正水动力信号。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Transformer 架构在瞬时预测精度上占优（LtViViT RMSE 最低）**，适合需要高精度单点估计的应用场景。
2. **轻量级 CNN-RNN 模型（PtAttnCNN）结合强物理正则化，在整体趋势拟合能力上超越 Transformer**，更适合业务化运行。
3. **物理引导机制显著提升模型的物理一致性和稳定性**，防止出现非物理解释。
4. **自动化 ROI 检测提升约 27% 的精度**，并使模型无需人工干预即可适应不同摄像机角度。
5. **三阶段迁移学习有效缓解标注数据不足问题**，Phase 0 合成训练使验证 RMSE 在两轮内下降 60.3%，最终达 0.3138s。

---

### ⚠️ 局限性

1. **未使用严格独立测试集**：由于 Golden Set 仅含 13 场景，无法划分出完全独立的测试集，所有报告指标均为 Phase 2 微调过程中的表现。
2. **地理泛化待验证**：当前数据主要来自南非大西洋沿岸（Namibia），波气候以长周期涌浪为主，尚未验证风浪主导区域（如北海、地中海）的表现。
3. **仅估计 $T_p$，未扩展至其他参数**：如 $H_s$、波向、runup 等关键参数尚未纳入统一预测框架。

---

### 🔮 未来工作方向

1. **拓展至全球不同波候环境**：验证方法在风浪主导区（North Sea）、混合涌浪区（Pacific）的适用性。
2. **集成多参数联合估计**：同步预测 $H_s$、wave direction、runup 等，构建完整视频波浪观测系统。
3. **开发 Hybrid Transformer-Recurrent 架构**：兼顾精度与推理效率，进一步降低边缘设备延迟。
4. **扩大 Golden Set 并引入严格 hold-out 测试**：支持更严谨的模型比较与评估。
5. **探索 unsupervised/slef-supervised 学习范式**：减少对伪标签和专家标注的依赖。

---

## 总结

本研究成功构建了一个**低成本、高物理一致性、可实时部署**的海岸波峰周期估计框架。它通过 **automated ROI detection + Sim-to-Real transfer learning + physics-informed loss** 三重机制，实现了从原始视频到 $T_p$ 的端到端可靠预测。研究成果为替代传统昂贵传感器提供了一条可行路径，有望广泛应用于海岸工程、灾害预警与气候变化适应等领域。

</details>

---

### 12. [Eidola: Modeling Multi-GPU Network Communication Traffic in Distributed AI Workloads](https://arxiv.org/abs/2606.12638)

**Authors**: Ranganath R. Selagamsetty, Matthew Poremba, Bradford M. Beckmann, Joshua San Miguel, Mikko H. Lipasti  
**Category**: cs.DC  
**Published**: 2026-06-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.12638v1  

#### Abstract
As distributed AI workloads grow in scale, multi-GPU systems have become essential for training large models. Although techniques like kernel fusion and overlapping communication with computation help reduce delays, they also introduce irregular and transient traffic patterns that are difficult to m...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Eidola: Modeling Multi-GPU Network Communication Traffic in Distributed AI Workloads

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代分布式AI训练工作负载（如Transformer、LLMs）广泛采用**kernel fusion**（核融合）和**通信-计算重叠**技术以提升效率。然而，这些技术引入了高度不规则且瞬态的**multi-GPU网络通信流量**，导致细粒度同步压力增大，对互连带宽和延迟极为敏感。

现有的GPU模拟器（如gem5、gem5-gpu、GPGPU-Sim）缺乏对多GPU间通信行为的建模能力，难以支持对这类复杂交互的架构级研究。

### 提出的新方法与创新
作者提出了 **Eidola** —— 一个可扩展的 **gem5 扩展框架**，用于在大规模多GPU系统中进行高精度、周期级的通信流量建模。

#### 核心创新点：
- **轻量级抽象GPU模型（eidolon）**  
  非目标GPU被建模为“**eidolon**”（幽灵），仅保留必要的通信行为特征（如xGMI写操作的时间戳），而非完整微架构模拟，显著降低开销。
  
- **基于时间标注的通信回放机制**  
  利用真实应用运行时采集的**timing profile**（时间剖面），通过新的GPU伪操作 `register_write` 注册peer-to-peer的xGMI写事件，并在仿真中按精确cycle时机触发。

- **Write Tracking Table (WTT)**  
  在模拟器端维护一个优先队列结构，按唤醒时间排序所有待发通信事件，实现cycle-level精度的通信调度。

- **支持灵活的性能隔离分析**  
  允许研究人员独立控制每颗GPU的通信模式，便于研究不同拓扑、延迟、拥塞场景下的性能影响。

### 相比现有方法的优势
| 方法 | 局限性 | Eidola 的优势 |
|------|--------|---------------|
| **gem5 / gem5-gpu** | 缺乏原生multi-GPU通信建模 | 支持xGMI级peer-to-peer通信建模 |
| **GPGPU-Sim** | 单GPU微架构模拟为主 | 可扩展至数百GPU的大规模配置 |
| **MGPUSim** | 所有GPU均全细节模拟，扩展性差 | 抽象非目标GPU，实现亚线性扩展 |

> ✅ **核心优势：兼顾真实性与可扩展性**

---

## 2. 核心实验方法和设置

### 使用的数据来源
- **真实应用trace** 来自论文 [30] 中提出的 **fused GEMV+AllReduce kernel**。
- trace 包含每个GPU上发生的 **peer-to-peer xGMI写操作的时间戳**（即flag更新时间）。
- 这些trace由实际运行在4-GPU系统上的程序通过ROCm Profiler等工具收集。

### 实验设置
- **模拟平台**：gem5 + 自定义Eidola扩展
- **GPU模型**：基于AMD CDNA架构（MI100/MI200/MI300系列）
- **通信接口**：xGMI（external Global Memory Interconnect）
- **同步机制**：基于non-cacheable内存区域的flag polling
- **运行模式**：
  - Setup kernel：功能模式下注册所有通信事件
  - Main kernel：详细时序模式下执行目标GPU，其余GPU由eidolon回放通信行为

### 评估指标
| 指标 | 描述 |
|------|------|
| `Flag read count` | 自旋等待期间产生的无效内存读请求数量 |
| `Non-flag read count` | 正常计算过程中的有效内存访问 |
| `Simulation time (wall-clock)` | gem5仿真的总耗时，衡量可扩展性 |
| `Scaling behavior` | 输入规模M和eGPU数量增加时的性能变化趋势 |

### 基线方法对比
- **Baseline**：标准spin-wait同步方式（持续轮询flag变量）
- **Enhanced**：引入SyncMon-inspired机制，使用`monitor()`/`mwait()`实现spin-yield同步
- **对比维度**：
  - 内存流量随延迟的变化
  - 模拟时间随输入大小和GPU数目的增长趋势

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）通信延迟对内存流量的影响（图6）
- 当`wakeupTime`从0μs增至40μs：
  - **flag read数量线性增长**（~134 → ~660次）
  - 表明spin-wait行为被准确复现
  - 验证了Eidola能精细控制通信时序并反映其性能影响

#### （2）SyncMon机制的效果（图9）
- 启用`mwait`后：
  - **flag read数量稳定在728–788之间**，不再随延迟增长
  - **non-flag read保持约66K不变**
  - ➜ 成功实现了**spin-yield同步语义**，避免了无谓轮询

#### （3）模拟时间随输入规模扩展（图10）
- 输入矩阵维度M从1K到24K：
  - 总体simulation time呈**近似线性增长**（r²: 0.76–0.98）
  - 表明Eidola保持了原始应用的计算scaling特性
  - 加入`mwait`后开销极小，未破坏scaling规律

#### （4）模拟时间随eGPU数目扩展（图11）
- eGPU数量从3增至255：
  - 归一化simulation time仅增长 **7.3× 至 35.9×**
  - 显著低于理想全模拟的256×成本
  - ➜ 实现**sub-linear扩展**，验证了eidolon抽象的有效性

### 消融实验结果
- **WTT vs Event Queue设计选择**：
  - 当前WTT采用polling机制，每cycle一次比较，开销极低
  - 虽然event queue更高效，但当前设计更利于调试和透明性
- **CPU pseudo-op vs GPU pseudo-op**：
  - 曾尝试用CPU线程触发GPU通信，但因host参数干扰导致timing不可重现
  - 最终采用GPU侧`register_write`确保cycle-level准确性

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Eidola能够高保真地建模multi-GPU通信行为**，特别是fused kernel中的细粒度同步与xGMI流量。
2. ✅ **通过抽象非目标GPU为eidolon，实现了良好的可扩展性**，可在单台机器上模拟上百GPU系统的通信效应。
3. ✅ **支持灵活的“what-if”分析**，例如调整通信延迟、注入拥塞、测试新型同步原语（如SyncMon）。
4. ✅ **验证了spin-yield类同步机制的有效性**：相比传统spin-wait，可大幅减少无意义的memory traffic。

### 方法的局限性
- **依赖外部trace输入**：目前需要预先采集真实trace，无法完全脱离硬件运行生成初始profile。
- **仅建模通信事件本身**：未模拟底层network fabric的排队、拥塞、路由等动态行为（假设xGMI事务瞬间完成）。
- **重点在通信建模而非计算细节**：目标是分析通信对GPU性能的影响，而非替代全功能GPU模拟器。

### 未来工作方向
1. **扩展至更大规模系统验证**：将trace采集从单节点扩展到跨节点、多跳Infinity Fabric拓扑。
2. **支持更多fused kernel类型**：如embedding pooling + All-to-All、GEMM+All-to-All等非对称负载。
3. **集成event-driven backend**：利用gem5原生event queue替换WTT polling，进一步优化性能。
4. **支持合成traffic generation**：结合概率模型生成多样化通信模式，增强泛化能力。

---

> 🔚 **总结一句话**：  
> **Eidola填补了现有模拟器在大规模multi-GPU通信建模方面的空白，提供了一个真实、可控、可扩展的研究平台，为下一代AI加速器架构探索提供了关键工具支持。**

</details>

---

### 13. [To Intervene or Not: Guiding Inference-time Alignment with Probabilistic Model Blending](https://arxiv.org/abs/2606.11201)

**Authors**: Jin Gan, Xin Li, Jun Luo  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.11201v1  

#### Abstract
The wide deployment of LLMs has made model alignment necessary to make newly trained models safely and effectively respond to user instructions. Among different methods, inference-time alignment is often cheaper as it intervenes (i.e., offers guidances) only during output generation. Existing propos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*To Intervene or Not: Guiding Inference-time Alignment with Probabilistic Model Blending*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

- **Inference-time alignment 中的“质量盲区”（Quality Blindness）问题**：  
  现有方法（如 NUDGING、IVG、InferAligner）在推理时通过引入 **guidance model** 来引导未对齐的 **base model**，但这些方法普遍假设所有 guidance 都是有益的，缺乏对 guidance 质量的评估机制。
  
- **干预悖论（Intervention Paradox）**：  
  论文通过系统性实验证明：**高干预率（intervention rate）反而与更差的性能相关**，即过度干预通常意味着 guidance 不可靠，导致 cascading failures（级联错误），而非提升对齐效果。

### ✅ 提出的新方法：**BlendIn**

- **核心思想**：从“二元决策”转向“软集成”（soft integration）
  - 不再简单地接受或拒绝 guidance 的 token 建议（binary accept/reject），而是将 **guidance model 和 base model 的完整概率分布进行加权融合**，生成一个混合分布（hybrid distribution）。
  - 在每个生成步骤中，当 base model 置信度低（`max_prob < T`）时，触发融合机制。

- **关键机制**：
  - **自适应融合权重 α**：基于两个模型的置信度（top-1 probability）和 token 级别的一致性动态计算：
    $$
    \alpha = \text{clip}\left(\frac{p_g}{p_b + p_g} + \lambda \cdot P_b(t_g),\ 0,\ 1\right)
    $$
    其中 $p_g$ 是 guidance model 的最大概率，$p_b$ 是 base model 的，$\lambda=0.1$ 控制一致性奖励。

  - **混合分布采样**：
    $$
    P_{\text{blend}}(w) = \alpha \cdot P_g(w) + (1-\alpha) \cdot P_b(w)
    $$
    然后从中选择最高概率的 token（greedy decoding）。

### ✅ 相比现有方法的优势

| 方面 | 传统方法（如 NUDGING） | BlendIn |
|------|------------------------|--------|
| 决策方式 | 二元接受/拒绝 | 连续加权融合 |
| 质量感知 | ❌ 无，视为均有益 | ✅ 自适应调整权重 |
| 错误传播 | 易因错误 guidance 引发级联失败 | ✅ 可抑制不可靠建议的影响 |
| 性能稳定性 | 在高干预场景下显著下降 | ✅ 在高干预场景下仍能提升性能 |
| 诊断能力 | ❌ 无法预测失败 | ✅ 干预率可作为早期诊断信号 |

> ✅ **核心优势**：**在高干预、低质量 guidance 场景下实现最多达 50% 的性能提升，同时在低干预场景保持稳定。**

---

## 2. 核心实验方法和设置

### ✅ 数据集

共使用 **6 个基准数据集**，涵盖推理、事实性和安全性任务：

| 类型 | 数据集 | 说明 |
|------|--------|------|
| 推理 | **GSM8K** | 小学数学应用题，测试逻辑推理能力 |
| 事实性 | **TruthfulQA** | 测量模型是否模仿人类错误信念 |
| 安全性 | **XSTest** | 对抗性安全提示，测试有害输出控制 |
| 综合理解 | **MMLU**, **ARC-Challenge**, **JustEval-Safe** | 多项选择、常识推理、安全评分 |

> 所有任务以 **accuracy** 或 **average safety score** 为指标（0–1 或 1–5 scale）。

---

### ✅ 实验设置

- **模型组合**：
  - **Base Models**（未对齐）：Llama-3.1-8B, Gemma-2-9b, Qwen3-8B-Base
  - **Guidance Models**（对齐）：Llama-3.2-1B-Instruct, Gemma-3-1b-it, Qwen3-1.7B
  - 覆盖 **跨家族（cross-family）** 与 **同家族（within-family）** 组合，共 9 个模型。

- **干预阈值**：
  - 默认 `T = 0.4`：当 base model 最大预测概率 < 0.4 时触发融合。

- **评估方式**：
  - 报告 **task performance**（准确率/安全分）和 **intervention rate**（干预 token 占比）
  - 使用 **greedy decoding**（temperature=0）确保可复现性
  - 每个 benchmark 使用 **100 个样本子集** 进行主实验，部分扩展至全集验证

---

### ✅ 基线方法对比

| 方法 | 简介 |
|------|------|
| **Base Model** | 无任何 guidance，仅 base 模型独立生成 |
| **Guidance Model** | 仅用 guidance 模型生成，验证其自身能力 |
| **Aligned base model** | 经过 fine-tuning 的对齐版本，作为性能上界 |
| **NUDGING (Fei et al., 2025)** | 当 base 不确定时直接采纳 guidance 的 top token（主流 baseline） |
| **Intervention Capping** | 当干预率超过 15% 后停止所有 guidance（用于验证“数量限制无效”） |

---

## 3. 主要实验结果和性能指标

### ✅ 关键性能数据（来自 Table 1）

| 模型对 | 任务 | NUDGING | **Ours (BlendIn)** | 提升幅度 | 干预率 |
|--------|------|---------|------------------|----------|--------|
| Qwen → Llama | GSM8K | 0.27 | **0.31** | **+15%** | 22.2% |
| Qwen → Llama | TruthfulQA | 0.48 | **0.50** | **+4%** | 31.9% |
| Qwen → Llama | XSTest | 0.03 | **0.04** | **+33%** | 33.1% |
| Qwen → Gemma | GSM8K | 0.54 | **0.56** | +4% | 23.1% |
| Gemma → Llama | TruthfulQA | 0.45 | **0.50** | **+11%** | 19.2% |
| Llama → Gemma | XSTest | 0.10 | **0.15** | **+50%** | 20.3% |

> 🔍 **观察**：提升最显著出现在 **高干预率（>20%）且 baseline 表现差** 的组合中（如 Qwen 作为 guidance 时）。

---

### ✅ 与基线方法对比

- **相比 NUDGING**：
  - 在所有高干预对中实现一致提升（+4% ~ +50%）
  - 在低干预对中性能持平或略优，**无退化现象**
  - 成功缓解“干预越多，表现越差”的悖论

- **相比 Intervention Capping（表 3）**：
  - 强行限制干预率至 15% 导致全面性能下降：
    - Qwen→Gemma: 0.54 → 0.46 (**-15%**)
    - Gemma→Llama: 0.59 → 0.55 (**-7%**)
    - 即使原本低于 15% 的 Llama→Gemma 也从 0.67 → 0.58
  - 证明：**问题在于质量，而非数量**

- **消融实验支持**（Appendix A.3）：
  - 对比离散过滤策略（Agreement Filter + Confidence Competition）：
    - 表现不稳定，多数情况下不如 NUDGING
    - 证明 **soft blending 必要性强于 rule-based filtering**

---

### ✅ 其他重要发现

- **干预率是有效诊断信号**：
  - 图 3 显示：**intervention rate 与 performance 呈显著负相关**（GSM8K: r = -0.65, p=0.009）
  - >20% 干预率可作为快速判断模型不兼容的早期指标

- **词表重叠不影响性能**（图 4 & 表 4）：
  - 计算 top-50 / top-p 词表重叠率，发现与性能无显著相关性（|r| ≤ 0.35, p > 0.35）
  - 排除“tokenization mismatch”作为根本原因

- **跨家族 vs 同家族**：
  - Qwen 作为 guidance 时，**同家族内对齐也失败**，说明其 alignment 能力本身较弱
  - Llama 和 Gemma 在跨家族中表现更鲁棒

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **存在“干预悖论”**：  
   高干预率 ≠ 更好对齐，反而是 **不可靠 guidance 的症状**，会引发 cascading failures。

2. **现有方法存在“质量盲区”**：  
   缺乏对 guidance 质量的评估机制，导致盲目接受有害建议。

3. **软分布融合优于硬决策**：  
   BlendIn 通过 **质量感知的分布加权融合**，实现了更稳健、高效的 inference-time alignment。

4. **干预率可作为诊断工具**：  
   可在小样本上快速评估模型对是否兼容，避免昂贵的全量测试。

5. **性能提升集中在高干预场景**：  
   最多可达 **50% 的相对提升**，尤其在 safety 和 truthfulness 任务上。

---

### ⚠️ 方法的局限性

1. **不能完全替代 fine-tuning**：  
   尽管 BlendIn 提升明显，但仍无法达到 fully aligned model 的性能上限。

2. **仍需经验调参**：  
   虽然提出默认参数（T=0.4, α 自适应），但最优性能依赖 task-specific tuning（见 Appendix A.2）。

3. **无先验兼容性预测模型**：  
   无法在部署前预测哪组模型对会成功，仍需实测。

4. **计算开销略有增加**：  
   需查询两个模型的完整分布（或 top-k），虽可用 top-100 缓解，但仍高于 binary 方法。

---

### 🚀 未来工作方向

1. **构建 guidance quality predictor**：  
   利用干预率等信号训练轻量级模型，预测模型对的兼容性。

2. **动态调整融合策略**：  
   引入 context-aware blending，根据不同任务阶段自动调节 α。

3. **扩展到 representation-level alignment**：  
   在激活层而非 token 分布层面进行 soft blending。

4. **探索更大规模 guidance models**：  
   当前使用 1B 级 guidance 模型，预期更大模型能带来更显著收益。

5. **结合 speculative decoding 加速推理**：  
   将 BlendIn 与 fast decoding 技术结合，兼顾效率与效果。

---

> 💡 **总结一句话**：  
> **BlendIn 通过“质量感知的分布融合”解决了 inference-time alignment 中的“干预悖论”，在不牺牲效率的前提下，实现了对不可靠 guidance 的鲁棒处理，在最具挑战性的高干预场景中取得高达 50% 的性能提升。**

</details>

---

### 14. [Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching](https://arxiv.org/abs/2606.11583)

**Authors**: Zhuoyi Peng, Hanlin Gu, Lixin Fan, Yi Yang  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.11583v1  

#### Abstract
Text-attributed graphs (TAGs) underlie real-world applications such as citation networks, social media, and e-commerce. Few-shot graph learning on TAGs is hard: with only a handful of labels per class and the rest of the graph unannotated, neither GNNs nor LLMs can learn well on their own. GNNs read...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在**少样本图学习**（few-shot graph learning）场景下，文本属性图（Text-Attributed Graphs, TAGs）中每个类别仅有少量标注节点，其余大部分节点无标签。传统方法面临以下挑战：
- **GNNs** 依赖图拓扑结构，在低度节点（cold nodes）上表现差（邻居信息不足）。
- **LLMs** 仅依赖文本，在文本模糊或简短时难以准确分类。
- 现有 LLM-GNN 融合方法普遍采用“**黄金教师范式**”（golden-teacher assumption），即固定一个模型为“权威教师”，另一个作为学生进行模仿，这在稀疏监督下会将教师的盲区直接传递给学生。

### 🚀 提出的新方法：LLM-GNN Co-Teaching
作者提出一种**双向协同教学框架**（co-teaching），**不指定任何一方为“黄金教师”**，而是让 GNN 和 LLM 在多轮迭代中互相提供最自信的伪标签（pseudo-labels），共同进化。

#### 核心机制：
1. **双向伪标签交换**  
   每轮训练中，GNN 和 LLM 分别基于自身置信度（small-loss criterion）选择最可靠的预测结果，并将其作为监督信号传递给对方。
2. **Round-based Pseudo-Label Preference Optimization (RPL-PO)**  
   当某个节点从第 $t$ 轮的跨模型矛盾（disagreement）转变为第 $t+1$ 轮的一致（agreement）时，LLM 在该节点上的两个输出构成一个偏好对（preference pair）：
   - **rejected**: 第 $t$ 轮错误/矛盾的答案
   - **chosen**: 第 $t+1$ 轮被 GNN 认可的正确答案  
   利用 **DPO**（Direct Preference Optimization）更新 LLM，无需人工标注、奖励模型或外部评判。

### 🔍 相比现有方法的优势
| 特性 | 传统方法（如 GNN-as-Judge） | LLM-GNN Co-Teaching |
|------|-------------------------------|------------------------|
| 教学方向 | 单向（固定教师） | 双向协同 |
| 教师角色 | 固定不变 | 动态互换 |
| 错误修正能力 | 无法纠正教师错误 | 可通过轨迹自纠错 |
| 监督信号来源 | 外部伪标签或固定模型输出 | 来自训练过程中的动态一致性 |
| 是否需要人类标注/奖励模型 | 否（但依赖强教师） | 完全自监督 |

> ✅ **核心思想突破**：放弃“必须有一个可靠教师”的假设，在弱-弱协作中通过时间轨迹挖掘出可靠的监督信号。

---

## 2. 核心实验方法和设置

### 📚 数据集
在六个标准 **text-attributed graph** 上进行评估，涵盖不同规模与复杂度：
| 数据集 | 类型 | 节点数 | 边数 | 类别数 | 领域 |
|-------|------|--------|--------|--------|------|
| **Cora** | 引用网络 | 2,708 | 10,858 | 7 | 学术论文 |
| **Citeseer** | 引用网络 | 3,186 | 4,277 | 6 | 学术论文 |
| **PubMed** | 引用网络 | 19,717 | 88,670 | 3 | 医学文献 |
| **WikiCS** | 维基百科超链接 | 11,701 | 431,726 | 10 | 计算机科学 |
| **ogbn-arxiv** | arXiv 引用 | 169,343 | 1,166,243 | 40 | 计算机科学子领域 |
| **ogbn-products** | 亚马逊商品共购 | 54,025 | 72,319 | 47 | 商品分类 |

> 注：所有数据集均采用 **k-shot 设置**（每类仅 k 个标注节点），测试集随机采样 1,000 节点评估。

### 🧪 实验设置与评估指标
- **任务**：少样本半监督节点分类（few-shot semi-supervised node classification）
- **标签预算**：3-shot、5-shot、10-shot
- **评估指标**：**Accuracy (%)**
- **训练流程**：
  - 共运行 $T=20$ 轮协同教学
  - 每两轮执行一次 RPL-PO
  - 使用 LoRA 对 Llama-3-8B-Instruct 进行参数高效微调（PEFT）
  - GNN 使用 GCN 架构（也可替换为 GAT/SAGE）

### ⚔️ 基线方法对比
分为三类基线：
1. **经典 GNN 模型**：
   - GCN, GAT, GraphSAGE
2. **LLM-as-Predictor 方法**：
   - Zero-shot prompting
   - Graph-CoT（Chain-of-Thought）
   - Neighbor-augmented prompting
3. **LLM-GNN 联合方法**：
   - GLEM, TAPE（LLM-enhancer）
   - LLM-GNN, LLaGA, GraphGPT（LLM-predictor）
   - **GNN-as-Judge**（当前最优单向方法）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（3-shot 准确率 %）

| Method | Cora | Citeseer | PubMed | WikiCS | ogbn-arxiv | ogbn-products |
|--------|------|----------|--------|--------|------------|---------------|
| GCN | 70.72 | 55.14 | 67.72 | 60.33 | 39.83 | 60.14 |
| GNN-as-Judge | **77.89** | **73.59** | **87.12** | **67.50** | **62.21** | **81.02** |
| **LG Co-Teaching (Ours)** | **85.75** | **77.12** | **91.32** | **74.80** | **69.94** | **82.82** |

> ✅ **绝对提升最大达 +7.86%（Cora）和 +7.73%（ogbn-arxiv）**

#### 观察总结：
- 在所有 **6 个数据集、3 种 shot 数量** 下，LLM-GNN Co-Teaching 均取得 **SOTA 性能**
- 提升在 **小样本条件下更显著**，说明 RPL-PO 提供的有效监督特别适用于标签稀缺场景
- 在大规模细粒度分类任务（如 ogbn-arxiv 的 40 类）上优势明显，传统方法严重退化

### 🔬 消融实验结果（Ablation Study）

| 变体 | Cora | ogbn-arxiv | ogbn-products |
|------|------|-----------|----------------|
| Full Model | 85.75 | 69.94 | 82.82 |
| w/o bidirectional teaching | 78.66 | 65.50 | 79.78 |
| w/o RPL-PO | 83.03 | 66.77 | 79.73 |
| Fixed R=0.5 | 83.20 | 67.10 | 82.68 |
| w/o neighbor info | 85.08 | 68.76 | 80.89 |

#### 消融分析结论：
- **移除双向教学** → 性能大幅下降至接近 GNN-as-Judge，证明**双向互教至关重要**
- **移除 RPL-PO** → 平均下降约 2–3%，验证其额外增益
- **固定选择比例**不如动态升温策略有效
- **移除邻居信息**导致性能轻微下降，表明结构上下文对 LLM 至关重要

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **“黄金教师”假设在少样本下失效**  
   在标签极度稀疏的情况下，无论是 GNN 还是 LLM 都不可靠，强行指定一方为教师会导致错误传播。

2. **双向协同教学能实现弱-弱变强**  
   尽管初始两个模型都较弱，但因其**归纳偏置互补**（GNN 强于结构、LLM 强于语义），可通过置信度筛选实现高质量知识互授。

3. **RPL-PO 成功从训练轨迹中提取监督信号**  
   利用“从分歧到一致”的节点变化构建 preference pair，实现了完全自监督的偏好优化，无需人工干预。

4. **误差结构分析显示互补性增强**  
   - GNN 原本在低度节点上错误率高；
   - 经过多轮 co-teaching 后，GNN 在这些节点上的表现显著改善，得益于 LLM 提供的语义补充；
   - LLM 也因接收更干净的 GNN 伪标签而提升整体鲁棒性。

5. **具备良好的零样本跨数据集迁移能力**  
   在 **arxiv → Cora/Citeseer/PubMed** 上的 zero-shot transfer 表现优于所有基线，说明模型学到的是通用图推理能力，而非过拟合特定分布。

---

### ⚠️ 局限性（Limitations）
1. **计算开销较高**  
   多轮迭代带来更高的训练时间成本（约为单次训练的 $T$ 倍）。例如在 ogbn-arxiv 上需约 4.7 小时（T=20），尽管可通过早停缓解。

2. **依赖高质量预训练 LLM**  
   若使用能力较弱的 LLM（如 Vicuna-7B 替代 Llama-3-8B），性能显著下降（-4% ~ -7%），说明 LLM 是系统瓶颈。

3. **适用范围受限于文本质量**  
   当前方法假设节点文本具有描述性且信息丰富。对于文本缺失、噪声大或非自然语言的图（如分子图、交易图）尚未验证。

4. **潜在偏差放大风险**  
   LLM 生成的伪标签可能携带社会偏见，若未被 GNN 纠正，可能在迭代中被强化。

---

### 🔮 未来工作方向
1. **扩展至其他模态图**  
   探索图像、音频等多模态属性图上的 LLM-GNN 协同学习。
   
2. **降低计算成本**  
   设计更高效的采样策略或提前终止机制，实现在资源受限环境下的部署。

3. **引入去偏机制**  
   在伪标签选择阶段加入公平性约束，防止偏见扩散。

4. **推广至其他任务**  
   如链接预测、图生成、异常检测等，探索 co-teaching 范式的普适性。

5. **理论分析收敛性与一致性边界**  
   当前为经验驱动，未来可建立形式化理论解释为何弱模型能通过轨迹达成一致。

---

> 💡 **一句话总结**：  
> 本文打破了“必须有一个强教师”的思维定式，提出 **LLM-GNN Co-Teaching + RPL-PO** 框架，首次实现两个弱模型在无黄金教师情况下的相互促进，并在六大数据集上刷新 SOTA，为少样本图学习开辟了新的自监督路径。

</details>

---

### 15. [ICA Lens: Interpreting Language Models Without Training Another Dictionary](https://arxiv.org/abs/2606.11722)

**Authors**: Sida Liu, Feijiang Han  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.11722v1  

#### Abstract
Finding interpretable directions in language-model representations is critical for understanding and controlling model behavior. Sparse autoencoders (SAEs) have become the standard tool for this purpose, but using them as the default first lens often requires training, storing, and evaluating large ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ICA Lens: Interpreting Language Models Without Training Another Dictionary

## 1. 论文的主要贡献和创新点

### 解决的问题
当前对大语言模型（LLM）进行可解释性分析的主流方法是使用 **Sparse Autoencoders (SAEs)** 来学习一个过完备的特征字典。然而，这种方法存在显著的**计算瓶颈**：
- 需要为每个模型、每一层、每种稀疏度设置训练和存储庞大的 SAE 字典。
- 训练过程消耗大量计算资源、存储空间和时间（例如 Gemma Scope 项目消耗了超过 GPT-3 训练成本的 20%）。
- 这使得快速探索和初步分析变得不切实际。

因此，论文提出一个根本性问题：**在不训练另一个神经网络字典的情况下，仅从激活向量的几何结构中能直接观察到多少可解释的结构？**

### 提出的新方法和新思路
论文提出了 **ICA Lens (ICALens)**，一种将经典统计方法 **独立成分分析 (Independent Component Analysis, ICA)** 重新定位为 LLM 可解释性“第一透镜”的实用化工作流。

其核心思想是：
- **直觉**：许多可解释的方向在特定 token 上具有选择性，这种选择性会导致其激活分布呈现**非高斯性 (non-Gaussianity)**（如重尾分布）。
- **方法**：ICA 正是寻找一组在投影后最“非高斯”的方向的算法。通过最大化非高斯性（如峰度），可以直接发现这些潜在的可解释方向，而无需像 SAE 那样通过稀疏重建来隐式学习。

### 相比现有方法的优势
- **高效且低成本**：ICALens 不需要梯度下降训练，避免了 SAE 的巨大计算开销。
- **即插即用**：提供了一套完整的、稳定的工作流，使其成为快速探索的“第一透镜”。
- **互补而非替代**：它不是要取代 SAE，而是作为 SAE 之前的轻量级筛选工具，帮助研究者决定在哪些层或任务上投入昂贵的 SAE 训练是值得的。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **激活语料库 (Activation Corpus)**：在三个开源 LLM 上进行实验：
  - `GPT-2 Small` (d=768)
  - `Gemma 2 2B` (d=2304)
  - `Qwen 3.5 2B Base` (d=2048)
- 对每个模型，收集来自 **Pile-10k** 数据集的 100 万个残差流 (residual-stream) 激活向量，用于 ICA 拟合。
- 同时也对词嵌入层 (embedding layer) 进行了分析。

### 实验设置和评估指标
#### ICA 拟合流程 (ICALens Pipeline)
为了解决标准 ICA 在 LLM 激活上的不稳定性，论文引入了三项关键技术：
1.  **Row-Normalization (行归一化)**：在中心化和白化前，对每个激活向量进行 ℓ² 归一化，以减少异常值的影响。
2.  **Robust Convergence Acceptance (鲁棒收敛接受)**：采用 `p95-LIM` 规则，当 95% 的分量已收敛时即可接受该层，避免因少数难收敛分量而拒绝整个层。
3.  **Adaptive Refit (自适应重拟合)**：对于难以收敛的层，自动降低目标分量数，确保至少能获得一个可用的紧凑基底。

#### 评估指标
1.  **Sparse Probing**：衡量特征坐标是否浓缩了概念相关信息。使用 SAEBench 的标准流程，在 AG News 等数据集上训练稀疏探针。
2.  **Targeted Probe Perturbation (TPP)**：衡量特征干预能力。通过零消减 (zero-ablate) 特征并测量探针分数的变化，评估干预的选择性。
3.  **人类可解释性审计 (Human Inspection)**：
    - **随机组件审计**：随机抽样 ICA 组件，由专家标注其语义标签和置信度。
    - **二次对比审计**：由第二位专家通过构造正反例提示来验证初始标签的有效性。
4.  **与 SAE 的关系分析**：
    - **方向重叠 (Directional Overlap)**：计算 ICA 分量与公共 SAE 解码器方向的最大余弦相似度。
    - **激活模式比较 (Activation Pattern Comparison)**：可视化同一句子上 ICA 和 SAE 的响应模式差异。

### 基线方法对比
- **Public SAEs**：如 Gemma Scope、Qwen Scope 等公开发布的 SAE 字典（高容量基线）。
- **Matryoshka SAEs**：不同大小的 SAE 变体，用于测试在紧凑预算下的表现。
- **ITDA**：另一种无需训练的轻量级稀疏编码方法。
- **PCA**：经典的基于方差最大化的线性降维方法，作为传统统计基线。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **拟合效率与稳定性**：
    - 在 `GPT-2 Small` 上，ICALens 的改进使成功接受的层数增加了 **400.0%**，总迭代次数减少了 **21.5%**。

2.  **非高斯性验证**：
    - 图 3 显示，**ICA 方向的峰度 (excess kurtosis) 显著高于随机方向和公共 SAE 解码器方向**，证明了 ICA 成功找到了统计上异常的方向。

3.  **Sparse Probing 性能**：
    - 图 10 和 11 显示，**ICA 在稀疏探针任务上的表现与公共 SAE 相当，且始终优于 ITDA 和 PCA**。
    - 即使是更小的 Matryoshka SAE，其性能也低于 ICA。

4.  **Targeted Probe Perturbation (TPP) 性能**：
    - 图 12 显示，在**小到中等规模的干预预算下 (small-to-medium intervention budgets)**，**ICA 的 TPP 表现优于公共 SAE**。
    - 这表明少量 ICA 分量就能有效实现选择性干预，而 SAE 需要更大的 N 才能积累足够的相关潜变量。

5.  **人类可解释性审计**：
    - **随机审计**：在 150 个随机抽样的 ICA 组件中，**142 个获得了非“模糊”(unclear) 的标签，其中 127 个为高置信度**。
    - **二次审计**：在 127 个高置信度标签中，**121 个被完全支持，6 个部分支持，无一被拒绝**，且 112 个得分 ≥8，证明了标签的可靠性。

6.  **与 SAE 的关系**：
    - **方向重叠**：图 13 显示，大多数 ICA 分量与某个 SAE 特征有中等程度的重叠，但也有相当一部分是弱匹配或强匹配的，说明两者既有关联又不冗余。
    - **激活模式**：图 15 显示，**SAE 特征倾向于在单个 token 或短片段上出现尖峰激活，而 ICA 方向则在上下文相关的 token 序列上表现出更平滑、持续的响应**（如追踪“financial context”）。

---

## 4. 关键结论和发现

### 主要发现
1.  **ICA 被低估了**：由于早期实现不稳定和缺乏系统评估，ICA 在 LLM 可解释性中一直被视为弱基线。ICALens 证明了其潜力。
2.  **非高斯性是一个强大的信号**：直接优化非高斯性可以有效地发现可解释的方向，其效果甚至优于基于方差的 PCA。
3.  **ICA 是一个高效的“第一透镜”**：ICALens 提供了一种快速、低成本的方法来探索 LLM 内部表示，能够揭示从词法、句法到语义、话语等多种层次的可解释结构。
4.  **ICA 与 SAE 互补**：两者目标不同，导致发现的方向和激活模式存在差异。ICA 更适合快速探索和轻量级干预，而 SAE 更适合高分辨率的特征发现。

### 方法的局限性
- **容量限制**：标准 FastICA 最多只能返回 `d` 个分量，无法像 SAE 那样生成过完备的大型字典。
- **紧凑基底**：对于需要极高分辨率分析的任务，ICA 的紧凑性可能成为瓶颈。
- **重建限制**：当分量数小于 `d` 时，干预式的编辑依赖于伪逆重建。

### 未来工作方向
- **更高容量的 ICA 变体**：探索过完备 ICA、预条件 ICA、JADE、Infomax 等方法，以突破维度限制。
- **分析变换而非状态**：将 ICA 应用于 MLP 输出、注意力输出或残差更新，以理解层间的信息变换。
- **自动化应用**：利用 ICA 的低成本特性，开发针对特定任务或数据分布的自动化分析和 steering 工具。

</details>

---

### 16. [Efficient Time Series Clustering from Multiscale Reservoir Dynamics with Granular-Ball Anchoring Graph Optimization](https://arxiv.org/abs/2606.12077)

**Authors**: Yifan Wang, Lifeng Shen, Shuyin Xia, Yi Wang  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2606.12077v1  

#### Abstract
Time-series clustering remains challenging due to the inherent trade-off between clustering effectiveness and computational efficiency. Similarity-based methods often suffer from quadratic complexity caused by pairwise distance computations, while deep learning-based approaches typically rely on cos...

---

### 17. [Reward Modeling for Multi-Agent Orchestration](https://arxiv.org/abs/2606.13598)

**Authors**: King Yeung Tsang, Zihao Zhao, Vishal Venkataramani, Haizhou Shi, Zixuan Ke, Semih Yavuz, Shafiq Joty, Hao Wang  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.13598v1  

#### Abstract
Multi-Agent Systems (MAS) built on Large Language Models (LLMs) require effective orchestration to coordinate specialized agents, yet training such orchestrators is hindered by limited supervision and high computational cost. We propose Orchestration Reward Modeling (OrchRM), a self-supervised frame...

---

### 18. [SkillCAT: Contrastive Assessment and Topology-Aware Skill Self-Evolution for LLM Agents](https://arxiv.org/abs/2606.13317)

**Authors**: Kunfeng Chen, Qihuang Zhong, Juhua Liu, Bo Du  
**Category**: cs.CL  
**Published**: 2026-06-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.13317v1  

#### Abstract
Skill self-evolution methods for LLM agents aim to turn execution trajectories into reusable skill documents, but current pipelines typically learn from one trajectory per task, merge candidate skill patches before checking them, and load the full skill corpus before inference. We propose SkillCAT, ...

---

### 19. [JiRAIYA: A Reputation-Based Hierarchical Federated Learning Framework on Web3](https://arxiv.org/abs/2606.13180)

**Authors**: Venkata Raghava Kurada, Pallav Kumar Baruah  
**Category**: cs.DC  
**Published**: 2026-06-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.13180v1  

#### Abstract
Federated Learning(FL) is predominantly deployed in enterprise environments, where limited transparency and restricted auditability hinder broader adoption. Existing FL systems often suffer from opaque aggregation processes, making it unclear which model updates are accepted or discarded. Current mi...

---

### 20. [Harness In-Context Operator Learning with Chain of Operators](https://arxiv.org/abs/2606.12318)

**Authors**: Minghui Yang, Ling Guo, Liu Yang  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2606.12318v1  

#### Abstract
Neural operators approximate mappings between function spaces, but often generalize poorly to other operators and usually require fine-tuning or retraining. In-Context Operator Networks (ICON) addresses this issue by prompting the model with numerical context so that the model learns specific operat...

---

### 21. [Otters++: A Time-to-first-spike Based Energy Efficient Optical Spiking Transformer](https://arxiv.org/abs/2606.13016)

**Authors**: Zhanglu Yan, Jiayi Mao, Kaiwen Tang, Fanfan Li, Gang Pan, Tao Luo, Bowen Zhu, Qianhui Liu, Weng-Fai Wong  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.13016v1  

#### Abstract
Spiking neural networks (SNNs) are promising for energy-efficient inference, and time-to-first-spike (TTFS) coding is especially attractive because each neuron fires at most once. In practice, however, this benefit is often reduced by the cost of computing a temporal decay term and multiplying it by...

---

### 22. [Maestro: Workload-Aware Cross-Cluster Scheduling for LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2606.12950)

**Authors**: Jinghao Wang, Xiao Zhou, Xiaoyang Sun, Yihui Zhang, Yilong Li, Tianyu Wo, Xu Wang, Chunming Hu, Renyu Yang  
**Category**: cs.DC  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.12950v1  

#### Abstract
Large Language Model based Multi-Agent Systems (LLM-MAS) have emerged as a powerful paradigm for tackling complex tasks by breaking them into collaborative workflows of specialized LLM-powered agents. However, deploying such multi-agent workloads at scale poses significant system challenges. Each us...

---

### 23. [FlowBank: Query-Adaptive Agentic Workflows Optimization through Precompute-and-Reuse](https://arxiv.org/abs/2606.11290)

**Authors**: Lingzhi Yuan, Chenghao Deng, Fangxu Yu, Souradip Chakraborty, Mohammad Rostami, Furong Huang  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11290v1  

#### Abstract
Large Language Model (LLM)-based multi-agent systems are increasingly powerful, but current agentic workflow optimization paradigms make an unsatisfying trade-off. Task-level methods spend substantial offline compute yet deploy only a single workflow, leaving complementary candidates unused, while q...

---

### 24. [CRUMB: Efficient Prior Fitted Network Inference via Distributionally Matched Context Batching](https://arxiv.org/abs/2606.11473)

**Authors**: Jamie Heredge, Mattia J. Villani, Pranav Deshpande, Akshay Seshadri, Niraj Kumar  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11473v1  

#### Abstract
Prior-fitted networks (PFNs) are a promising class of tabular foundation models that perform in-context learning, whereby the entire labelled training set is supplied as context, and predictions for test queries are produced in a single forward pass. However, the quadratically scaling self-attention...

---

### 25. [DeMix: Debugging Training Data with Mixed Data Error Types by Investigating Influence Vectors](https://arxiv.org/abs/2606.11616)

**Authors**: Jiale Deng, Yanyan Shen, Xiaogang Shi, Chai Junjun  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.11616v1  

#### Abstract
High-quality training data is essential for the success of machine learning models. However, real-world datasets often contain mixed types of errors arising from systematic flaws in data preparation pipelines, including label errors, feature errors, and spurious correlations. Effective debugging of ...

---

### 26. [Reinforcement Learning Disrupts Gradient-Based Adversarial Optimization](https://arxiv.org/abs/2606.12251)

**Authors**: Xinhai Zou, Chang Zhao, Alireza Aghabagherloo, Dave Singel\'ee, Robin Degraeve, Bart Preneel  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2606.12251v1  

#### Abstract
Gradient-based adversarial attacks remain a dominant threat to deep neural networks (DNNs), as they exploit gradient information to efficiently optimize adversarial perturbations. To address this, we investigate whether reinforcement learning (RL) training can disrupt the gradient structure used by ...

---

### 27. [Structured Testbench Generation for LLM-Driven HDL Design and Verification-Oriented Data Curation](https://arxiv.org/abs/2606.12983)

**Authors**: En-Ming Huang, Yu-Hung Kao, Ren-Hao Deng, Wei-Po Hsin, Yao-Ting Hsieh, Cheng Liang, Hsiang-Yu Tsou, Mu-Chi Chen, Yu-Kai Hung, Shao-Chun Ho, Po-Hsuang Huang, Shih-Hao Hung, H. T. Kung  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.12983v1  

#### Abstract
Automated testbench generation has become a critical bottleneck in large language model (LLM)-driven Register Transfer Level (RTL) workflows, where large numbers of candidate designs must be verified rapidly and reliably. Existing prompt-based approaches treat testbench generation as unconstrained c...

---

### 28. [ERTS: Adversarial Robustness Testing of Ethical AI via Semantic Perturbation in a Bounded Consequence Space](https://arxiv.org/abs/2606.13282)

**Authors**: Pratyush Chaudhari  
**Category**: cs.AI  
**Published**: 2026-06-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.13282v1  

#### Abstract
As AI systems are deployed in high-stakes ethical contexts such as healthcare triage, autonomous vehicle control, and employment screening, formal methods for evaluating their robustness against adversarial manipulation of ethical reasoning remain underdeveloped. This paper introduces the Ethical Ro...

---

### 29. [Beyond Uniform Tokens: Adaptive Compression for Time Series Language Models](https://arxiv.org/abs/2606.13624)

**Authors**: Jialin Gan, Xin Qiu, Guangzhe Chen, Xue Wang  
**Category**: cs.CL  
**Published**: 2026-06-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.13624v1  

#### Abstract
Large language models (LLMs) have enabled time series (TS) analysis by jointly modeling numerical observations and textual context through a shared token interface. However, TS tokens and prompt tokens exhibit fundamentally different information structures, making uniform token processing inefficien...

---

### 30. [SwiftCTS: Fast Cross-Design Prediction and Pareto Optimization of Clock Tree Metrics via Few-Shot Calibration](https://arxiv.org/abs/2606.11348)

**Authors**: Barsat Khadka, Kawsher Roxy, Md Rubel Ahmed  
**Category**: cs.LG  
**Published**: 2026-06-12  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2606.11348v1  

#### Abstract
Clock Tree Synthesis (CTS) is a computationally expensive stage in the physical design flow, requiring iterative EDA tool invocations to navigate a vast configuration space for optimal power, wirelength, and timing skew. Existing machine learning approaches require computationally expensive retraini...

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
