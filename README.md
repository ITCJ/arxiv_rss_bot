# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-27 08:00:25 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [FlashSpread: IO-Aware GPU Simulation of Non-Markovian Epidemic Dynamics via Kernel Fusion](https://arxiv.org/abs/2604.22092)

**Authors**: Heman Shakeri, Behnaz Moradi-Jamei, Aram Vajdi, Ehsan Ardjmand  
**Category**: cs.DC  
**Published**: 2026-04-27  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.22092v1  

#### Abstract
Non-Markovian (renewal) epidemic simulation on multi-million-node contact networks is essential for realistic forecasting under general age-dependent holding-time distributions (log-normal, Weibull, Erlang, and similar), but the age-dependent hazard forces dense per-step updates that render the spar...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FlashSpread: IO-Aware GPU Simulation of Non-Markovian Epidemic Dynamics via Kernel Fusion

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统流行病模拟多基于**Markovian模型**（如SIS、SIR），假设个体状态转移时间服从指数分布，即“无记忆”特性。然而，真实传染病（如COVID-19）的潜伏期、传染期等通常服从**非指数分布**（如Weibull、Log-normal），具有明显的峰值和延迟特征。这种**非马尔可夫（Non-Markovian）** 动态在计算上极具挑战性，因为每个节点的转移速率随其“年龄”（在当前状态停留的时间）连续变化，导致每一步都需要对所有节点进行密集更新（dense updates），无法利用稀疏事件队列优化。

现有CPU方法（如NEXT-Net）虽能处理精确的非马尔可夫过程，但受限于串行瓶颈，难以扩展到超大规模网络（>10⁶ 节点）。而GPU并行计算面临内存带宽瓶颈和中间张量开销，常规方法效率低下。

### 提出的新方法与创新点
作者提出 **FLASHSPREAD**，一个面向非马尔可夫流行病动力学的**统一GPU框架**，其核心创新如下：

1. **IO感知的核融合（IO-aware Kernel Fusion）**
   - 将整个每步模拟流程（CSR遍历、erfcx-based hazard计算、Bernoulli tau-leaping采样、状态转移、下一时刻传染性写回）**融合为单个Triton内核**。
   - 所有中间变量保留在**流式多处理器（SM）寄存器**中，避免了多次全局内存读写，显著降低HBM流量。
   - 引入**块级标量跳过（block-scalar skip）**：若某线程块内无活跃节点（E或I状态），则跳过昂贵的`erfcx`计算，同时保持**CUDA Graph**的可捕获性。

2. **度感知的CSR调度（Degree-aware CSR Dispatch）**
   - 针对不同图结构自动选择最优的CSR遍历策略：
     - `thread`：每节点一线程（适用于均匀度图）
     - `warp`：每节点一warp（32线程协作）
     - `merge`：边分区合并负载均衡（适用于幂律图）
   - 通过 `Dmax/Davg` 自动决策，兼顾正则图与幂律图性能。

3. **混合精度存储（Mixed-Precision Storage）**
   - 在不牺牲数值稳定性的前提下，压缩状态存储：
     - `state`: int8
     - `age`, `infectivity`, `weights`: fp16 / bf16
     - 关键累加器（如压力、速率）仍用fp32
   - 显著减少每步内存占用，提升L2缓存命中率。

4. **主动节点压缩（Active-node Compaction）**
   - 利用疫情后期大量节点进入吸收态R的特性，动态缩小内核网格规模，仅对非R节点执行计算。
   - 采用**固定网格早退出模式（Fixed-Grid Early-Exit）**，兼容CUDA Graph批处理。

5. **双引擎架构（Dual-Engine Architecture）**
   - 同时支持**Markovian**（稀疏事件驱动）和**Renewal**（密集时间步进）两种动力学，分别采用最优策略。

### 相比现有方法的优势
- **首次开源端到端GPU框架**用于非马尔可夫网络流行病模拟。
- 实现**217倍硬件加速**（vs. 优化后的CPU tau-leaping）。
- 支持高达 **N=10⁸** 节点的单卡模拟（A100 40GB）。
- 通过核融合和调度优化，在真实复杂网络上实现接近内存带宽极限的吞吐。

---

## 2. 核心实验方法和设置

### 数据集与图结构
- **Erdős-Rényi (ER)**：均匀度图，平均度 d=8。
- **Barabási-Albert (BA)**：幂律图，参数 m=4，平均度 ~8，最大度达3870（Dmax/Davg ≈ 484）。
- 规模范围：N ∈ [10², 10⁸]。

### 模型设定
- **SEIR模型**，其中：
  - S→E：边介导，Markovian（常数β）
  - E→I 和 I→R：节点内，Non-Markovian，使用Log-normal分布（均值5.0/7.5天，中位数4.0/5.0天）。
- 使用**Bernoulli tau-leaping**，容忍度 ε ∈ [0.005, 0.1]，最大步长 τ_max=0.1。

### 评估指标
- **NUPS (Node-Updates Per Second)**：衡量密集计算负载的核心指标，定义为 `(N × steps) / wall-clock time`。
- **Events/sec**：用于稀疏事件驱动方法。
- **Fidelity Metrics**：
  - 峰值感染误差（Peak I error）
  - 最终攻击率误差（Final attack rate error）
  - 与精确Gillespie模拟的轨迹差异（L∞, L₂）

### 基线方法对比
| 方法 | 类型 | 平台 | 是否开源 |
|------|------|------|----------|
| c-GEMF, FastGEMF | 精确事件驱动（Markovian） | CPU | 是 |
| EoN | 非马尔可夫模拟 | CPU | 是 |
| NEXT-Net | 精确非马尔可夫模拟 | CPU | 是 |
| **CPU tau-leaping (8-core)** | 近似同步更新（本文实现） | CPU | 本文提供 |
| **FLASHSPREAD (unfused)** | 未融合GPU版本 | GPU | 本文 |

> 注：作者特别构建了一个强CPU基线（基于PyTorch向量化操作），包含相同的算法改进（如自适应tau-leaping、erfcx稳定计算等），以公平比较硬件加速效果。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 配置 | 性能（NUPS） | 加速比 |
|------|------------|--------|
| **Fused CG (ER, N=10⁶)** | **8.09 Giga-NUPS** | — |
| CPU tau-leaping (8-core) | 37.3 Mega-NUPS | **217×** |
| Unfused CG (GPU) | 1.87 Giga-NUPS | — |
| Fused CG (BA, merge) | 2.0 Giga-NUPS | — |

- 在**ER图**上，核融合带来 **4.3×** 加速，CUDA Graph批处理再增 **2.8×**。
- 在**BA图**上，默认`thread`策略性能骤降至0.45 Giga-NUPS，启用`merge`策略后恢复至 **2.0 Giga-NUPS**，实现 **4.5×** 提升。

### 与基线方法对比
- 相比**精确CPU方法**（如Phantom Process）：
  - 速度提升超过 **10⁵ 倍**（从~70 events/sec 到 5.86M events/sec）。
- 相比**优化CPU tau-leaping**：
  - 实现 **217× 严格硬件加速**（相同算法，仅平台差异）。
- 在**N=10⁷** 时，因超出L2缓存（~40MB），出现“L2缓存悬崖”，性能下降4.4×。
- 启用**混合精度**后，在N=10⁷时性能提升 **2.32×**，有效将L2可达规模扩大约3倍。

### 消融实验结果
#### （1）核融合与CUDA Graph
| 配置 | NUPS | 相对于前一项提升 |
|------|------|----------------|
| CPU tau-leaping | 37.3M | — |
| GPU unfused eager | 0.67G | 18× |
| GPU CG b=50 (unfused) | 1.87G | 2.8× |
| **GPU Fused CG b=50** | **8.09G** | **4.3×** |

> 表明**核融合**是最大性能增益来源。

#### （2）度感知调度（Table 2）
| 策略 | ER图 (G-NUPS) | BA图 (G-NUPS) | BA/ER比率 |
|------|---------------|---------------|-----------|
| thread | 7.88 | 0.45 | 5.7% |
| warp | 1.70 | 1.30 | 76.5% |
| **merge** | 3.92 | **2.00** | **51.0%** |

> `merge`在BA图上表现最佳，但在ER图上仅为`thread`的一半，凸显**自动调度必要性**。

#### （3）混合精度（Table 5）
| 图 | 基线 (G-NUPS) | 混合精度 (G-NUPS) | 加速比 |
|----|--------------|------------------|--------|
| ER d=8 (N=10⁶) | 7.15 | 8.16 | 1.14× |
| BA m=4 (N=10⁷) | 1.33 | 3.10 | **2.32×** |

> 在接近L2缓存极限时，混合精度收益巨大。

#### （4）主动节点压缩（Table 3）
| 图 | 基线 (G-NUPS) | 压缩后 (G-NUPS) | 加速比 |
|----|--------------|----------------|--------|
| ER d=8 | 7.52 | 7.60 | 0.99× |
| **BA m=4** | 0.44 | **0.67** | **1.53×** |

> 在高饱和度幂律图上效果显著，是“长尾相位”的重要优化。

---

## 4. 关键结论和发现

### 主要发现
1. **非马尔可夫模拟本质上是内存带宽受限任务**，必须通过**IO-aware核融合**消除中间张量开销。
2. **217倍硬件加速**证明了GPU在同步tau-leaping范式下的巨大潜力，远超CPU串行方法。
3. **结构性偏差存在下限**：即使ε→0，同步Bernoulli更新对精确Gillespie的误差仍有：
   - 峰值感染误差：~6%
   - 最终攻击率误差：~7%
   > 此偏差源于同步更新忽略了事件间的时序依赖，且**不随ε减小而消失**。
4. 该偏差远小于典型流行病学参数不确定性（常>20%），因此在实际预测中可接受。
5. **度异质性严重破坏默认线程映射**，需专用调度策略（如`merge`）恢复性能。
6. **混合精度与主动节点压缩**是扩展至超大规模（N≥10⁷）的关键技术。

### 方法的局限性
1. **结构性偏差不可消除**：同步更新固有缺陷，无法达到事件驱动的精度。
2. **单卡内存限制**：目前最大支持 N=10⁸（A100 40GB），更大规模需多GPU分解。
3. **静态拓扑假设**：不支持动态图（边独立形成/断裂），否则需O(E)级年龄跟踪。
4. **源节点近似限制**：假设传染性仅取决于源节点年龄，不适用于剂量累积或异质易感性场景。

### 未来工作方向
1. **多GPU域分解**：突破单卡内存墙，支持超大规模模拟。
2. **动态图支持**：引入边级年龄跟踪机制。
3. **更高级负载均衡**：如**分段规约（segmented reduction）** 减少原子冲突。
4. **形式化收敛分析**：建立结构性偏差与网络谱隙、度分布的理论关系。
5. **ROCm移植**：利用Triton的跨平台能力，支持AMD GPU。

> 代码已开源：[https://github.com/Shakeri-Lab/FlashSpread](https://github.com/Shakeri-Lab/FlashSpread)

</details>

---

### 2. [Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation](https://arxiv.org/abs/2604.22312)

**Authors**: Long Cheng, Ritchie Zhao, Timmy Liu, Mindy Li, Xianjie Qiao, Kefeng Duan, Yu-Jung Chen, Xiaoming Chen, Bita Darvish Rouhani, June Yang  
**Category**: cs.DC  
**Published**: 2026-04-27  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.22312v1  

#### Abstract
Sparse-attention decoders rely on exact Top-K selection to choose the most important key-value entries for each query token. In long-context LLM serving, this Top-K stage runs once per decode query and becomes a meaningful latency bottleneck even when the indexer and attention kernels are already hi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在长上下文 LLM 推理中，**sparse-attention 解码器**依赖于精确的 **Top-K 选择**来筛选最重要的 key-value 条目。然而，在序列长度达到 100K 以上时，尽管 indexer 和 attention 内核已高度优化，**Top-K 阶段仍成为显著的延迟瓶颈**。

传统 GPU Top-K 算法（如 radix-select）是分布无关的（distribution-agnostic），无法利用 LLM 自回归解码过程中特有的**时间相关性**（temporal correlation）——即相邻解码步之间 Top-K 索引集合高度重叠的现象。

### 提出了什么新方法或新思路
本文提出 **Guess-Verify-Refine (GVR)**，一种基于数据感知的、精确的 Top-K 算法，专为 NVIDIA Blackwell 架构上的 sparse-attention 解码设计。

#### 核心思想：
- **利用前一步的 Top-K 结果作为预测信号**（temporal prediction signal），指导当前步的 Top-K 选择。
- 将 Top-K 过程分解为四个阶段：
  1. **Guess**：基于前一步 Top-K 计算统计量（均值、最大/最小值），并通过 **secant-style 插值法**快速估计一个有效的阈值。
  2. **Verify**：通过一次全局扫描验证候选集是否包含真正的 Top-K 元素，并收集所有高于阈值的候选。
  3. **Refine**：在共享内存中对小规模候选集进行精确排序，选出最终的 K 个元素。

#### 创新机制：
- **Secant-style 阈值搜索**：将传统的多轮 radix 分解减少到仅需 **1–2 轮全局内存访问**（原方法需 3–4 轮）。
- **Ballot-free 收集机制**：避免使用 `__ballot_sync`，消除编译器屏障导致的 L2 加载流水线串行化。
- **Phase 2 计数缓存**：复用阈值搜索阶段的线程级计数结果，省去一次完整的 N 元素扫描。
- **共享内存内精炼**：在 SMEM 中完成直方图构建与 snap 迭代，极大降低同步开销。

### 相比现有方法的优势
| 维度 | GVR | Radix-Select (Baseline) |
|------|-----|------------------------|
| **全局内存访问次数** | 1–2 次 | 3–4 次 |
| **同步开销** | 极低（单 CTA 设计 + 无 ballot） | 高（原子操作竞争 hot bins） |
| **算法性质** | 数据感知（data-aware） | 分布无关（distribution-agnostic） |
| **输出精度** | Bit-exact（位精确） | Bit-exact |
| **适用场景** | 解码阶段（decode-phase） | 通用 |

> ✅ **优势总结**：GVR 在保持完全精确性的前提下，显著降低了 Top-K 的延迟，尤其适用于具有强时间稳定性的自回归解码任务。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **真实数据**：来自 `DeepSeek-V3.2` 在 **SWE-bench-derived LongSeqTasks** 上的实际推理日志。
  - 包括两个公开 prompt bucket：`swe_bench_64k.jsonl` 和 `swe_bench_100k.jsonl`
  - 上下文长度从 68K 到 100K 不等，生成长度约 2K tokens
- **合成数据**：模拟 RoPE + YaRN 编码下的 indexer scores，用于控制变量分析。

### 实验设置和评估指标
- **平台**：NVIDIA B200 GPU（Blackwell, sm_100）
- **框架集成**：嵌入 **TensorRT-LLM** 的 DSA（DeepSeek Sparse Attention）流程
- **评估层级**：
  - **单算子级别**：测量 `invokeIndexerTopKDecode` 内核延迟
  - **端到端级别**：在 TEP8 并行模式下测量 **Tokens Per Operation Time (TPOT)**
- **关键指标**：
  - 单算子延迟（ns）
  - 速度提升比（Speedup）
  - 端到端 TPOT 减少百分比
  - Draft Acceptance Rate (DAR)（用于 speculative decoding 场景）

### 基线方法对比
- **主基线**：生产环境中的 **radix-select kernel**（Zhang et al., 2023 的 Blackwell 优化版本）
  - 已比 `torch.topk` 快 7.4×
- **其他可选基线**：未直接比较（因缺乏 Blackwell 支持）：
  - RadiK (Li et al., 2024)
  - Sampling-based 或 approximate Top-K 方法（牺牲精度）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 单算子性能（真实解码数据，N ≈ 70,690）
| 层 | 平均 Speedup | 最大 Speedup |
|----|--------------|-------------|
| L0 | 1.59× | 1.78× |
| L1 | 1.71× | 1.90× |
| L20–L60 | 1.82–2.04× | 2.42× |
| **Overall Average** | **1.88×** | **2.42×** |

> 🔹 **最高提速达 2.42×**，平均 **1.88×**，且所有层均有正向增益。

#### 合成数据表现（随序列增长）
| N | GVR Latency (ns) | Baseline Latency (ns) | Speedup |
|----|------------------|------------------------|---------|
| 8,192 | 16,512 | 11,200 | 0.68× |
| 65,536 | 31,904 | 47,936 | 1.50× |
| 131,072 | 43,392 | 76,128 | **1.75×** |

> 🔹 在短序列下略有开销，但在长序列下优势明显，呈线性扩展优势。

#### 端到端 TPOT 提升（TEP8, OSL=1K）
| Context | MTP=0 | MTP=1 | MTP=3 |
|--------|-------|-------|-------|
| 64K | **5.47%↓** | 4.36%↓ | 2.40%↓ |
| 100K | **7.52%↓** | 6.30%↓ | 3.45%↓ |

> ✅ **上下文越长，收益越大**；即使在 speculative decoding 下仍有正向增益。

### 消融实验结果

#### 不同 `preIdx` 输入的质量影响（N≈70K）
| `preIdx` 类型 | Top-K 重叠率 α | 平均延迟 | Speedup |
|---------------|----------------|----------|---------|
| 无 (`radix`) | 0% | 43.7 μs | 1.00× |
| 随机索引 | ~2.9% | 30.9 μs | **1.44×** |
| L20–L60 前步 Top-K | ~44% | 22.7 μs | **1.94×** |
| L0–L1 前步 Top-K | ~1.5% | 27.2 μs | **1.65×** |

> 🔍 发现：
> - 即使随机预测也能获得 **1.44×** 加速，说明 GVR 的架构优化本身就有巨大价值。
> - 时间相关性质量是决定性因素，高相关层提速近 **2×**。

#### 各阶段耗时分解（Layer L21, N≈70K）
| 阶段 | 耗时 | 占比 |
|------|------|------|
| P1: PreIdx Stats | 2.2 μs | 10% |
| P2: Threshold Search | 7.5 μs | 32% |
| P3: Candidate Collect | 5.9 μs | 26% |
| P4: Refinement | 7.6 μs | 33% |

> 🔎 观察：
> - P3 是带宽受限，基本恒定；
> - P2 和 P4 成为主要瓶颈，分别受预测质量和局部分数分布影响。

---

## 4. 关键结论和发现

### 主要发现
1. **Top-K 存在显著的时间相关性**：
   - 在 DeepSeek-V3.2 中，深层（L20–L60）连续解码步的 Top-K 重叠率达 **35–50%**，最高可达 **60%**。
   - 理论基础源于 RoPE/YaRN 的 **Toeplitz 结构**：注意力得分仅依赖相对位置，使得查询移动时得分图谱平移而非剧烈变化。

2. **数据感知策略可大幅提升 Top-K 效率**：
   - 利用历史 Top-K 作为 warm-start，结合 secant 插值，能将全局扫描从 3–4 次降至 **1–2 次**。
   - **ballot-free 收集 + 计数缓存** 是实现低延迟的关键工程优化。

3. **性能增益随上下文长度单调增加**：
   - 因为 Top-K 的计算量随 N 线性增长，而 sparse MLA 固定为 K，故其在总延迟中的占比越来越高。
   - 在 100K 上下文下，TPOT 提升达 **7.52%**，表明其对长上下文服务至关重要。

4. **方法鲁棒性强**：
   - 即使在低相关层（L0/L1）或使用随机 `preIdx`，仍能保持正向加速（最低 1.59×）。
   - 输出与 `torch.topk` **bit-exact**，无精度损失。

### 方法的局限性
- **当前为单 CTA 设计**：占用 ~60KB SMEM，限制了 SM 上并发 CTA 数量（3 CTAs/SM），在小批量或超高吞吐场景下可能非最优。
- **对短序列不友好**：在 N < 16K 时存在启动开销，提速不明显甚至更慢。
- **依赖 temporal stability**：仅适用于 decode-phase；prefill 阶段无法使用。
- **内存占用较高**：需要额外 HBM 缓冲区存储 `prev_topk` 和 `scratch`。

### 未来工作方向
- **自适应切换机制**：根据序列长度自动启用 GVR 或 fallback 到 radix-select。
- **扩展至多 CTA 架构**：支持 ultra-long context 或更高 batch size。
- **增强预测信号**：探索 prefill 阶段或多步预测机制。
- **跨架构迁移**：验证在 Hopper、Ada 等其他 GPU 上的有效性。
- **推广至其他 sparse-attention 架构**：如 NSA、RocketKV、SAGE-KV 等，只要其 Top-K 具备 temporal stability 即可受益。

---

> 📌 **总结一句话**：  
> GVR 通过挖掘 LLM 解码中 Top-K 的**时间相关性**，提出了一种高效、精确的 data-aware Top-K 算法，在 Blackwell 上实现了 **平均 1.88× 的单算子加速** 和 **高达 7.52% 的端到端 TPOT 提升**，为长上下文稀疏注意力系统提供了新的性能边界。

</details>

---

### 3. [Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](https://arxiv.org/abs/2604.22345)

**Authors**: Weixu Zhang, Ye Yuan, Changjiang Han, Yuxing Tian, Zipeng Sun, Linfeng Du, Jikun Kang, Hong Kang, Xue Liu, Haolun Wu  
**Category**: cs.CL  
**Published**: 2026-04-27  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.22345v1  

#### Abstract
Large Language Models (LLMs) exhibit strong implicit personalization ability, yet most existing approaches treat this behavior as a black box, relying on prompt engineering or fine tuning on user data. In this work, we adopt a mechanistic interpretability perspective and hypothesize the existence of...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的 **personalization** 方法（如 prompt engineering、fine-tuning 或 retrieval-based 方法）通常将 LLM 视为黑箱，缺乏对模型内部如何表示和传播用户偏好的理解。这些方法虽然能提升表面一致性，但在 **interpretability（可解释性）、controllability（可控性）和 scalability（可扩展性）** 上存在不足。

本文提出一个根本性问题：**LLM 中的个性化行为究竟在哪些内部组件中产生？**

### 提出了什么新方法或新思路
作者从 **mechanistic interpretability（机制可解释性）** 的视角出发，提出了以下核心概念与框架：

- **Preference Heads（偏好头）**：假设存在一组稀疏的 attention heads，它们专门编码用户的风格和主题偏好，并对生成过程施加因果影响。
- **Preference Contribution Score (PCS)**：通过因果掩码分析量化每个 attention head 对用户对齐输出的因果贡献。
- **Differential Preference Steering (DPS)**：一种无需训练的解码时干预框架，利用识别出的 Preference Heads 实现可解释、可控的个性化生成。

> **创新点总结**：
> - 首次从机制层面系统分析 LLM 中的个性化现象；
> - 提出“Preference Heads”这一可解释的功能模块假设；
> - 设计了基于因果干预的 PCS 和 DPS 框架，实现参数不变的个性化控制。

### 相比现有方法的优势
| 维度 | 现有方法（如 Fine-tuning, Prompting） | DPS |
|------|----------------------------------------|-----|
| **是否需要训练** | 是（计算成本高） | 否（training-free） |
| **是否可解释** | 否（黑箱） | 是（定位到具体 attention heads） |
| **是否可控** | 弱（依赖外部输入） | 强（通过 γ 控制强度） |
| **是否高效** | 低（需重新训练） | 高（仅增加一次 forward pass） |
| **是否支持异构用户** | 有限 | 支持（cluster-aware 扩展） |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
所有实验基于 **LaMP benchmark**（Salemi et al., 2024），涵盖多种任务类型：

| 任务类别 | 具体任务 |
|--------|--------|
| **Generation（生成）** | News Headline Generation, Scholarly Title Generation, Tweet Paraphrasing |
| **Classification（分类）** | Citation Identification, Movie Tagging |
| **Regression（回归）** | Product Rating |

> 数据集统计见附录 Table 5：覆盖约 1K–2.5K 用户，总计超 10 万实例。

### 实验设置和评估指标

- **模型**：
  - `LLaMA-3-8B-Instruct`
  - `Mistral-7B-Instruct`
  - `Qwen2-7B-Instruct`

- **评估指标**：
  - **生成任务**：ROUGE-1, ROUGE-L, METEOR（↑越高越好）
  - **分类任务**：Accuracy, F1（↑）
  - **回归任务**：MAE, RMSE（↓越低越好）

- **Personalization Strength**：通过超参数 γ 控制 DPS 干预强度。

- **Head Selection**：选择 PCS 最高的前 K 个 heads 作为 Preference Heads（默认 K=32）。

### 基线方法对比
DPS 与其他主流 decoding-time 干预方法进行比较：

| 方法 | 简介 |
|------|------|
| **CAD (Context Aware Decoding)** | 通过对比上下文完整与削弱的 logits 来增强忠实性 |
| **DoLa (Decoding by Contrasting Layers)** | 利用深层 vs 浅层 logits 差异提升事实性 |
| **DeCoRe (Decoding by Contrasting Retrieval Heads)** | 对比 retrieval heads 来减少幻觉 |

> 所有 baseline 使用官方代码或忠实复现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Tables 1 & 2）

#### 表 1：生成任务表现（部分摘录）

| Model | Method | R-1 ↑ | R-L ↑ | METEOR ↑ |
|-------|--------|-------|-------|----------|
| LLaMA-3-8B | CAD | 0.1681 | 0.1498 | 0.1568 |
| LLaMA-3-8B | DeCoRe | 0.1768 | 0.1572 | 0.1626 |
| LLaMA-3-8B | **DPS (ours)** | **0.1787** | **0.1596** | **0.1650** |
| Qwen2-7B | **DPS (ours)** | 0.1627 | 0.1450 | **0.1318** |
| Mistral-7B | **DPS (ours)** | **0.1536** | **0.1366** | **0.1399** |

> ✅ DPS 在多个模型上取得 **最佳或接近最佳** 的生成质量，尤其在 News Headline 和 Tweet Paraphrasing 上优势明显。

#### 表 2：分类与回归任务表现

| Model | Method | Citation ID (F1) ↑ | Movie Tagging (F1) ↑ | Product Rating (RMSE) ↓ |
|-------|--------|---------------------|------------------------|----------------------------|
| LLaMA-3-8B | DeCoRe | 0.6200 | 0.4034 | 0.9458 |
| LLaMA-3-8B | **DPS (ours)** | **0.6288** | 0.3910 | **0.9278** |
| Qwen2-7B | DoLa | 0.6795 | 0.0958 | 0.6300 |
| Qwen2-7B | **DPS (ours)** | **0.7078** | **0.3202** | **0.6719** |

> ✅ DPS 显著优于所有 baseline，特别是在 Qwen2-7B 上实现了大幅跃升。

### 与基线方法的对比结果
- DPS 在 **绝大多数任务和模型组合中达到最优性能**；
- 尤其在 **异构性强的任务（如 Movie Tagging）** 上表现突出，说明其能有效捕捉多样化偏好模式；
- 性能在不同类型任务间保持一致，表明 DPS 是一种 **通用且鲁棒的个性化机制**。

### 消融实验结果（Ablation Studies）

#### （1）真实 Preference Heads vs 随机控制
- 使用随机选择的 heads 或随机 masking 模式替代真实 Preference Heads → 性能显著下降（Fig. 5）；
- 结果验证：**性能增益来源于语义上有意义的 heads，而非任意稀疏操作**。

#### （2）不同数量的 heads（K）的影响（Fig. 6）
- 性能随 K 增加先上升后饱和；
- 过多 heads 反而引入噪声，说明偏好信号集中在少数关键 heads 中；
- DPS 对 K 的选择具有较强鲁棒性。

#### （3）cluster-aware 路由策略比较（Fig. 8）
- **Hard routing**：在分类任务上略优；
- **Soft routing**：在生成任务上更稳定，提供平滑控制；
- 两者各有优势，可根据任务需求灵活选择。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **个性化确实在 LLM 内部由稀疏的 attention heads 实现**：
   - 存在一类称为 **Preference Heads** 的功能单元，它们因果地影响用户对齐的生成行为。
   
2. **Preference Heads 是稀疏且用户特异的**：
   - 不同用户的 Preference Heads 分布差异大（Fig. 4，Jaccard 重叠度低）；
   - 支持“个性化路径因人而异”的假设。

3. **可通过解码时干预实现高效个性化**：
   - DPS 无需微调即可实现高质量、可控的个性化输出；
   - 通过对比 masked / unmasked logits 放大偏好信号，机制清晰。

4. **cluster-aware 扩展提升了泛化能力**：
   - 将相似用户聚类并共享部分 Preference Heads，提高了小样本用户的稳定性。

5. **人类评估也支持 DPS 更好匹配用户偏好**：
   - 在 LaMP-4 新闻标题生成任务中，人工标注者更倾向于 DPS 输出（40% vs 34%）；
   - GPT-5.2 评分显示 DPS 在 Style 和 Alignment 上得分更高（Table 4）。

### 方法的局限性
- **不适用于黑盒 API 模型**：需要访问内部 attention heads 和中间激活；
- **推理开销翻倍**：每步需两次 forward pass（原始 + masked），尽管大部分 prefill 可共享；
- **可能放大不良偏差**：若用户 profile 包含噪声或刻板印象，DPS 可能过度强化这些模式；
- **依赖离线 head 发现**：PCS 计算需用户历史数据，在冷启动场景下受限。

### 未来工作方向
- 探索在 **black-box setting 下近似 DPS 效果**，例如通过轻量级 logit biasing 或 prompt engineering；
- 优化 inference efficiency，如 early-exit 或 heads approximation；
- 结合 feedback loop 动态更新 Preference Heads；
- 研究如何防止偏好信号被滥用或导致 echo chamber 效应。

---

> 🔗 **开源地址**：https://github.com/weixuzhang/DPS  
> 📄 **引用格式**：Zhang et al., *Preference Heads in LLMs*, 2025

</details>

---

### 4. [Adaptive Head Budgeting for Efficient Multi-Head Attention](https://arxiv.org/abs/2604.22583)

**Authors**: Bilal Faye, Abdoulaye Mbaye, Hanane Azzag, Mustapha Lebbah  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.22583v1  

#### Abstract
Transformers have become the dominant architecture across a wide range of domains, largely due to the effectiveness of multi-head attention in capturing diverse representation subspaces. However, standard multi-head attention activates all heads uniformly for every input, regardless of task requirem...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Adaptive Head Budgeting for Efficient Multi-Head Attention

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题  
标准的 **Multi-Head Attention (MHA)** 在每个输入上**均匀激活所有注意力头**，无论任务复杂度或输入特性如何。这种固定模式在许多场景下（如文本分类）会造成**计算资源浪费**，因为并非所有头都对最终决策有贡献。尤其在粗粒度任务中，全局语义信息可能仅需少量注意力头即可捕获。

因此，论文指出：
- 固定头数导致不必要的 **FLOPs 和内存开销**
- 头分配未根据输入动态调整，造成**效率低下**

---

### 🚀 提出了什么新方法或新思路  
提出 **BudgetFormer** —— 一种具备**自适应多头注意力机制**的 Transformer 架构，其核心思想是：

- **动态分配注意力头预算（Head Budget）**：对每个输入预测一个标量 $ s \in (0,1) $，表示需要激活的头的比例（即 $ k = \lfloor s \cdot H \rfloor $）
- **学习头部重要性分布**：通过可学习的评分网络生成各头的 relevance 分数，并结合温度控制的 softmax 进行选择
- **探索-利用训练策略（Exploration-Exploitation Trade-off）**：
  - 早期训练阶段鼓励模型探索不同头组合（高熵分布）
  - 后期收敛到高效、稀疏的头使用模式（低熵分布）

该方法实现了**细粒度的条件计算（conditional computation）**，在推理时只激活最相关的注意力头。

---

### 🔍 相比现有方法的优势  

| 方法类别 | 局限性 | BudgetFormer 的优势 |
|--------|------|------------------|
| **Head Pruning**（静态剪枝） | 剪枝策略离线确定，无法适应不同输入 | 动态按需激活，保留全部容量用于训练 |
| **Sparse Attention**（稀疏注意力） | 固定模式限制表达能力 | 不改变注意力结构，仅调节活跃头数量 |
| **Token/Depth-level Adaptivity** | 调整 token 数量或层数深度 | 在**头级别**实现更精细的资源调控 |
| **MoE / Early Exiting** | 主要作用于 FFN 或层间跳过 | 针对 MHA 内部结构优化，互补性强 |

> ✅ **核心优势**：在不牺牲模型表达力的前提下，显著降低推理成本（FLOPs、Memory、碳排放），同时性能持平甚至超越全头注意力。

---

## 2. **核心实验方法和设置**

### 📚 使用的数据集  
在五个主流文本分类基准上进行评估：

| 数据集 | 任务类型 | 类别数 | 训练样本数 |
|-------|---------|--------|------------|
| **DBpedia** | 本体分类 | 14 | 560,000 |
| **AG News** | 新闻主题分类 | 4 | 120,000 |
| **IMDB** | 影评情感分析 | 2 | 25,000 |
| **SNLI** | 自然语言推断 | 3 | 549,367 |
| **Yelp Full** | 用户评论星级预测 | 5 | 650,000 |

---

### ⚙️ 实验设置  
- **模型架构**：4层 Transformer Encoder，$ H=8 $ 个头，$ D=768 $
- **Baseline**：标准 MHA（所有头始终激活）
- **BudgetFormer**：替换为自适应头预算注意力机制
- **预算控制器 $ f_o $**：两层 FFN + Sigmoid 输出 $ s $
- **头评分函数 $ g_d $**：单一线性投影
- **训练策略**：
  - AdamW 优化器，LR=2e-5，batch size=16
  - 总训练 10 轮
  - 引入 entropy 正则项与 budget hinge loss 控制训练稳定性
- **推理方式**：Top-$k$ 选择（$ k=\lceil s\cdot H \rceil $），仅计算选中的头

---

### 📊 评估指标  
| 指标 | 描述 |
|-----|------|
| **Accuracy** | 分类准确率 |
| **FLOPs** | 推理总浮点运算量（测试集累计） |
| **Memory Usage** | 显存占用（理论与实测） |
| **Carbon Emission (gCO₂)** | 推理过程碳足迹估算 |
| **$ s_{\text{mean}} $** | 平均激活头比例（反映资源利用率） |

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（来自 Table II）

| Dataset | Accuracy (Transformer) | Accuracy (BudgetFormer) | Δ Acc | FLOPs Reduction | $ s_{\text{mean}} $ |
|--------|------------------------|--------------------------|-------|------------------|---------------------|
| DBpedia | 0.9830 | **0.9859** | **+0.29** | ~3.2% | **0.085** |
| AG News | 0.9099 | 0.9022 | -0.77 | ~2.7% | 0.212 |
| IMDB | 0.8354 | **0.8356** | +0.02 | ~10% | 0.601 |
| SNLI | 0.7835 | **0.8106** | **+2.71** | ~2.1% | 0.364 |
| Yelp | 0.5810 | **0.6190** | **+3.80** | ~20% | 0.198 |

> 💡 **关键观察**：
> - 在 **SNLI 和 Yelp** 上大幅提升准确率（+2.7 ~ +3.8 pts）
> - 在简单任务（如 AG News）略有下降但仍可控
> - 所有任务均实现 **FLOPs 下降**，最高达 **20%**
> - 碳排放同步减少（如 Yelp 从 1.075g → 0.859g CO₂）

---

### 🔬 消融实验结果

#### Ablation 1: 固定预算（Fixed Budget, 无 $ f_o $）
| s | DBpedia Acc | SNLI Acc | Yelp Acc |
|----|-------------|----------|----------|
| 0.1 | 0.9846 | 0.6704 | 0.5818 |
| 1.0 | 0.1670 | 0.3190 | 0.0743 |

> ❗ 结果显示：**固定高预算严重损害性能**，说明盲目增加头数会引入噪声；而 BudgetFormer 学习到最优小预算（如 DBpedia 仅用 8.5% 头）反而效果更好。

#### Ablation 2: 随机头选择（Random Gating, 无 $ g_d $）

| Dataset | Random $ g_d $ Acc | BudgetFormer Acc |
|--------|----------------------|------------------|
| DBpedia | 0.7511 | **0.9859** |
| SNLI | 0.3370 | **0.8106** |
| Yelp | 0.3244 | **0.6190** |

> ❗ 表明：**“选哪些头”比“选多少头”更重要**。随机选择即使预算正确也导致崩溃性性能下降。

> ✅ 结论：**$ f_o $（定量控制） + $ g_d $（定性选择）缺一不可**

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **注意力头存在显著冗余**：多数任务中仅需少数几个关键头即可完成决策。
2. **输入复杂度决定所需头数**：简单样本使用更少头（如 DBpedia 中命名实体识别），困难样本自动提升预算。
3. **动态头分配可提升泛化能力**：通过探索-利用机制，模型学会剔除冗余头，增强鲁棒性。
4. **效率与性能可兼得**：在降低高达 20% FLOPs 的同时，仍能取得更高 Accuracy。
5. **良好的可扩展性**：
   - 模型越大（更多层/头），$ s_{\text{mean}} $ 越低 → 更好地利用冗余
   - 数据越多，预算越小 → 学习更高效的表示

---

### ⚠️ 方法的局限性

1. **依赖全局池化（Global Pooling）**：
   - $ f_o $ 和 $ g_d $ 均基于 mean-pooling 的全局表示
   - 忽略 token-level 差异，在 QA、推理等细粒度任务中可能受限

2. **仅关注头级调制**：
   - 未结合 token pruning 或 layer skipping，未能全面挖掘冗余

3. **当前设计更适合粗粒度任务**：
   - 如分类、情感分析等全局决策任务表现优异
   - 对序列生成、机器阅读理解等尚未验证

---

### 🔮 未来工作方向

1. **拓展至复杂任务**：
   - 应用于 **Question Answering、Multi-hop Reasoning、Long-context Modeling**

2. **集成至大模型架构**：
   - 将 BudgetFormer 机制嵌入 **LLMs（如 Llama、BERT）**，验证其在真实场景下的可扩展性

3. **跨模态应用**：
   - 探索在 **Vision Transformers (ViT)** 或 **Multimodal Models（如 CLIP）** 中是否同样有效

4. **构建混合效率框架**：
   - 结合 **Token Pruning + Head Budgeting + Early Exiting**
   - 实现多层次、端到端的高效推理系统

5. **改进局部敏感机制**：
   - 设计能感知 token 重要性的轻量模块，替代全局池化，提升细粒度建模能力

---

> 🌱 **总体评价**：  
> BudgetFormer 提供了一种**原则性强、实现简洁且效果显著**的注意力效率优化路径。它揭示了 Transformer 中“**稀疏即智能**”的趋势，并为构建**绿色 AI（Green AI）** 提供了可行方案 —— 在保证性能的同时大幅降低计算与环境成本。

</details>

---

### 5. [Context-Fidelity Boosting: Enhancing Faithful Generation through Watermark-Inspired Decoding](https://arxiv.org/abs/2604.22335)

**Authors**: Weixu Zhang, Fanghua Ye, Qiang Gao, Jian Li, Haolun Wu, Yuxing Tian, Sijing Duan, Nan Du, Xiaolong Li, Xue Liu  
**Category**: cs.CL  
**Published**: 2026-04-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.22335v1  

#### Abstract
Large language models (LLMs) often produce content that contradicts or overlooks information provided in the input context, a phenomenon known as faithfulness hallucination. In this paper, we propose Context-Fidelity Boosting (CFB), a lightweight and general decoding-time framework that reduces such...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Context-Fidelity Boosting: Enhancing Faithful Generation through Watermark-Inspired Decoding

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题  
该论文针对 **Large Language Models (LLMs)** 在生成过程中出现的 **faithfulness hallucination**（忠实性幻觉）问题。这种现象指模型在生成文本时，忽略、矛盾或扭曲输入上下文中的事实信息，即使这些信息是明确提供的。这在 **retrieval-augmented generation (RAG)**、**question answering** 和 **summarization** 等高风险场景中尤为严重。

与常见的 **factuality hallucination**（事实性幻觉）不同，faithfulness hallucination 强调的是输出与**给定输入上下文的一致性**，而非是否符合真实世界知识。

---

### 🚀 提出的新方法：Context-Fidelity Boosting (CFB)

作者提出了一种轻量级、无需训练的 **decoding-time intervention** 方法——**Context-Fidelity Boosting (CFB)**，其核心思想受 **text watermarking** 中的 **logit-shaping** 技术启发：

- 不修改模型参数或架构
- 在推理阶段通过 **additive logit adjustment** 动态提升“被上下文支持”的 token 的生成概率
- 实现更忠实于输入内容的生成

#### 三层递进式 boosting 策略：
| 方法 | 描述 |
|------|------|
| **Static Boosting** | 对所有出现在上下文中的 token 统一增加固定偏置 $\delta$ |
| **Context-aware Boosting** | 根据上下文存在与否对 next-token 分布的影响程度（用 JSD 衡量），动态调整 boost 幅度：$\Delta(w) = \delta_{\text{min}} + (\delta_{\text{max}} - \delta_{\text{min}}) \cdot D$ |
| **Token-aware Boosting** | 进一步结合 **source-position attention** 和 **source-scoped semantic similarity**，为每个 source-supported token 分配不同的 boost 权重 |

---

### 🔍 相比现有方法的优势

| 特性 | CFB | 其他方法（如 CAD, ADACAD, COIECD） |
|------|-----|-----------------------------|
| 是否需要 retraining | ❌ 否 | 多数 ❌ |
| 模型通用性（model-agnostic） | ✅ 高 | ✅ |
| 推理效率 | ⚡ 轻量级（尤其 Static/Context-aware） | 一般较高开销 |
| 控制粒度 | ✅ 支持从全局到 token-level 的精细控制 | 多为整体分布对比或硬约束 |
| 设计理念 | 受 watermarking 启发，soft bias 更自然 | 常依赖 contrastive decoding 或复杂规则 |

> 💡 创新亮点：将 watermarking 中用于“嵌入可检测信号”的 logit-shaping 思想，**迁移至增强 context-faithfulness**，实现了可控且低扰动的生成校准。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 任务类型 | 数据集 | 说明 |
|--------|-------|------|
| **Summarization** | **CNN/DM**, **XSum** | 测试模型能否忠实总结原文内容；XSum 难度更高，摘要更简洁 |
| **Question Answering** | **NQ-Synth**, **NQ-Swap** | <br>- **NQ-Synth**: 上下文与模型先验一致（互补知识）<br>- **NQ-Swap**: 故意构造上下文与模型记忆冲突（知识冲突） |

---

### 🧪 实验设置与评估指标

| 设置项 | 内容 |
|-------|------|
| **模型** | Llama2-13B-chat-hf, Llama3-8B-Instruct, Mistral-7B-Instruct |
| **采样方式** | Top-p sampling（zero-shot setting） |
| **评估指标** | <br>- **ROUGE-L**: 衡量生成质量（lexical overlap）<br>- **FactKB**: 基于 LM 的事实一致性评分<br>- **BERT-P**: 语义相似度<br>- **Accuracy (%)**: QA 任务的答案准确率 |

---

### 🆚 基线方法对比

| 基线方法 | 简介 |
|--------|------|
| **CAD (Context-Aware Decoding)** | 固定超参调节输出分布 |
| **ADACAD (Adaptive CAD)** | 使用 JSD 动态调整 context 影响力度 |
| **COIECD** | 基于信息熵约束，在冲突/非冲突 token 上采用不同策略 |

> 所有方法均在同一解码框架下比较，确保公平性。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总（以 Llama3-8B 为例）

#### ✅ Summarization 结果（CNN/DM）
| 方法 | ROUGE-L ↑ | FactKB ↑ | BERT-P ↑ |
|------|-----------|----------|---------|
| CAD | 35.92 | 94.57 | 89.07 |
| Static CFB | **36.79** | 95.15 | 89.63 |
| Context-aware CFB | 36.78 | **97.23** | **89.85** |
| Token-aware CFB | 35.81 | 94.31 | 89.38 |

✅ **Context-aware CFB 在 FactKB 和 BERT-P 上显著领先**

#### ✅ QA 结果（NQ-Synth，上下文补充知识）
| 方法 | Accuracy ↑ | ROUGE-L ↑ |
|------|------------|-----------|
| CAD | 66.80 | 28.19 |
| Static CFB | 73.10 | 29.87 |
| Token-aware CFB | **73.40** | **32.90** |

✅ **CFB 显著提升 accuracy 和生成质量，尤其在 token-aware 下表现最佳**

#### ❌ QA 结果（NQ-Swap，上下文与先验冲突）
| 方法 | Accuracy ↑ |
|------|------------|
| ADACAD | **86.50** |
| Token-aware CFB | 32.43 |

⚠️ 在强知识冲突场景下，**ADACAD 等显式抑制 parametric prior 的方法更有效**，而 CFB 表现较弱

---

### 🔬 消融实验（Ablation Study）

在 `Llama3-8B + CNN/DM` 上进行消融（Token-aware CFB）：

| 变体 | ROUGE-L | FactKB | BERT-P |
|------|--------|--------|--------|
| 完整模型 | 35.81 | 94.31 | 89.38 |
| -w/o attention | 35.60 | 93.74 | 88.48 |
| -w/o semantic similarity | 34.45 | 66.84 | 67.68 |
| -w/o JSD scaling | 35.24 | 93.60 | 88.43 |

📌 发现：
- 移除 **semantic similarity** 导致崩溃 → 是稳定性和效果的关键组件
- 移除 **attention** 或 **JSD** 有影响但不致命 → 仍保留一定增益
- 表明：**全局自适应缩放 + 局部相关性信号** 协同作用最优

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **CFB 能有效提升 context-faithfulness**  
   在 summarization 和上下文补充型 QA（NQ-Synth）任务上，显著优于多种 decoding-time baseline，同时保持生成流畅性。

2. **无需训练即可部署，兼容性强**  
   作为 pure decoding-time 方法，适用于任意开放权重 LLM，易于集成到实际系统中。

3. **多层级控制提供灵活 trade-off**  
   用户可根据任务需求选择 Static / Context-aware / Token-aware 策略，在效率与精度之间平衡。

4. **human & LLM evaluation 验证有效性**  
   - human rating：Token-aware CFB 在 **faithfulness (4.31)** 和 **informativeness (4.12)** 上最高
   - GPT-4o judge：hallucination 数量最少（0.67），contradiction rate 最低（0.05）

5. **在低/中等冲突场景下表现稳健，在高冲突下受限**  
   当上下文与模型先验明显冲突时（如 NQ-Swap），需更强的 contrastive suppression 机制，此时 ADACAD 更优。

---

### ⚠️ 方法的局限性

| 限制 | 说明 |
|------|------|
| **依赖模型内部信息** | 需访问 logits、attention map、embeddings → 不适用于 black-box API（如 GPT-4） |
| **计算开销随精细度上升** | Token-aware variant 引入较多额外计算（见 Table 8），FLOPS 开销约 2.86e+08，高于其他变体 |
| **局部相关性估计可能不准** | attention 与 semantic similarity 的组合未必总能准确反映 token 重要性 |
| **未建模复杂推理过程** | 如 multi-hop reasoning（HotpotQA）任务上表现不佳，说明其优势集中在 context grounding 而非 reasoning |

---

### 🔮 未来工作方向

1. **Black-box approximation**  
   探索仅基于输出 token 序列推断 context-support 的方法，使 CFB 可用于 API 模型。

2. **更鲁棒的 token-level relevance modeling**  
   引入更好的语义匹配机制（如 cross-attention, sentence-level alignment）替代当前 heuristic 设计。

3. **降低 decoding overhead**  
   优化 attention 与 similarity 计算频率，例如缓存或稀疏化处理。

4. **结合 reasoning-aware decoding**  
   将 CFB 与 chain-of-thought 或 retrieval planning 结合，提升在 multi-hop QA 中的表现。

5. **探索与其他 watermarking 技术融合**  
   如 soft watermarking、semantic-invariant watermarking，进一步提升控制能力与隐蔽性。

---

## ✅ 总结一句话

> **Context-Fidelity Boosting (CFB)** 是一种受 watermarking 启发的轻量级 decoding-time 方法，通过 additive logit shaping 提升上下文支持 token 的生成概率，在 summarization 和 context-complementary QA 任务上显著改善 faithfulness，且无需 retraining，具备良好的实用性与扩展潜力。

</details>

---

### 6. [Decoding High-Dimensional Finger Motion from EMG Using Riemannian Features and RNNs](https://arxiv.org/abs/2604.22499)

**Authors**: Martin Colot, C\'edric Simar, Guy Cheron, Ana Maria Cebolla Alvarez, Gianluca Bontempi  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.22499v1  

#### Abstract
Continuous estimation of high-dimensional finger kinematics from forearm surface electromyography (EMG) could enable natural control for hand prostheses, AR/XR interfaces, and teleoperation. However, the complexity of human hand gestures and the entanglement of forearm muscles make accurate recognit...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Decoding High-Dimensional Finger Motion from EMG Using Riemannian Features and RNNs

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文致力于解决**从表面肌电图（sEMG）连续解码高维手指运动**的挑战，旨在实现更自然、直观的人机交互（如假肢控制、AR/XR 接口）。传统方法多依赖分类模型，限制了可控自由度（DOF），导致动作不连续；而回归任务因手部复杂性和个体差异极具挑战。

### 🚀 提出的新方法与创新
1. **可复现的数据采集框架**  
   - 提出了一种低成本、易部署的 EMG-FK 数据采集方案，仅需一个 **8通道消费级 EMG 手环（MindRove）** 和一个普通笔记本摄像头。
   - 引入自动同步机制（基于 Hilbert 变换与运动命令相关性），解决了无硬件同步下的时序对齐难题。

2. **新型轻量级回归模型：Temporal Riemannian Regressor (TRR)**  
   - 首次将 **multi-band Riemannian covariance features（CMTS）** 与 **GRU-based RNN 架构**结合用于 sEMG 到指关节角度的连续回归。
   - 特征提取采用在多个频段（5–40Hz, 40–80Hz, 80–150Hz）计算协方差矩阵并投影至切空间（tangent space），捕捉肌肉协同激活模式。

3. **开源资源发布**  
   - 发布了 **EMG-FK 数据集**（10小时，20名参与者，15个指关节角 + 8通道 EMG）
   - 开源代码与训练框架（GitHub + Zenodo）

### 🔍 相比现有方法的优势
| 方面 | TRR 的优势 |
|------|-----------|
| **准确性** | 在 intra- 和 cross-subject 设置下均优于 SOTA 方法（如 vemg2pose） |
| **实时性** | 模型轻量化设计，在 Raspberry Pi 5 上可达 ~10 预测/秒 |
| **部署友好性** | 运行功耗低，CPU 温升慢，适合嵌入式设备（如假肢控制器） |
| **泛化能力** | 支持无标定场景下的跨被试预测（cross-subject） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 描述 |
|--------|------|
| **EMG-FK（本文提出）** | 自建数据集，20 名被试，每人 30 分钟自由手势，采样率 500Hz，含 8 通道 sEMG 与 15 个指关节角。通过 MediaPipe 获取姿态。 |
| **emg2pose（公开基准）** | Meta Reality Labs 发布的大规模数据集，共 193 名被试。本研究使用其中前 30 名的第一大 session 子集进行对比。 |

### ⚙️ 实验设置
- **两种评估配置**：
  - **Intra-subject**：10 折交叉验证（不打乱时间顺序），每折用 10% 数据作验证集。
  - **Cross-subject**：Leave-One-Subject-Out（LOSO），测试对象完全未参与训练。
- **滑动窗口处理**：步长 100ms，窗口长度由模型决定（TRR 使用 10 个 300ms 窗口序列）。
- **预处理**：
  - EMG 经带通滤波（15–150Hz）与陷波滤波（50/100Hz）
  - 各被试独立标准化（standardization）

### 🎯 评估指标
- **主指标**：Normalized Mean Square Error (NMSE)
- **辅助解释指标**：平均绝对误差（Average Absolute Error, AE）以度数（°）表示（仅 EMG-FK 可计算）

### 🆚 基线方法对比
| 模型 | 类型 | 特征/架构 |
|------|------|----------|
| **vemg2pose** | SOTA 深度学习模型 | CNN + LSTM，输入原始高采样率 EMG（2000Hz） |
| **MLP on TDF** | 浅层模型 | 时间域特征（TDF）+ 多层感知机 |
| **MLP on CMTS** | 浅层模型 | Riemannian 协方差特征 + MLP |
| **CRNN on Raw/Envelope** | 深度模型 | 1D-CNN + GRU，输入原始信号或包络 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（EMG-FK 数据集）

| 方法 | Intra-subject AE (°) | Cross-subject AE (°) | Intra NMSE | Cross NMSE |
|------|------------------------|------------------------|------------|-------------|
| **TRR（本文）** | **9.79 ± 1.48** | **16.71 ± 3.97** | **0.43 ± 0.08** | **0.78 ± 0.12** |
| vemg2pose | 11.04 ± 1.51 | 16.76 ± 3.31 | 0.51 ± 0.09 | 0.79 ± 0.11 |
| MLP on CMTS | 11.41 ± 1.37 | 18.66 ± 3.12 | 0.53 ± 0.08 | 0.86 ± 0.10 |
| CRNN on Raw | 11.35 ± 1.56 | 17.87 ± 2.69 | 0.52 ± 0.09 | 0.82 ± 0.10 |

> ✅ **TRR 在所有设置下均取得最优性能**

### 📈 与其他方法对比结果
- **显著优于所有基线**：无论是否使用深度网络，TRR 均表现最佳。
- **CMTS 特征优势明显**：即使配合简单 MLP，CMTS 也优于 TDF 和端到端 CNN。
- **RNN 结构增益显著**：相比非循环模型（MLP、Gradient Boosting），GRU 能更好建模动态运动依赖关系。

### 🔬 消融实验结果（Ablation Study）
#### （A）特征提取方式影响
| 特征类型 | NMSE |
|--------|-------|
| CMTS（三频段） | **0.43** ✅ |
| CMTS（单频段 15–150Hz） | 0.45 |
| CMTS（无 shrinkage） | 0.47 |
| TDF | 0.58 |
| CRNN（Raw） | 0.52 |

> ➤ 多频段 CMTS 最优，说明不同频率携带互补信息。

#### （B）回归器架构比较
| 回归器 | NMSE |
|--------|-------|
| TRR（GRU-based） | **0.43** ✅ |
| MLP | 0.53 |
| Gradient Boosting | 0.61 |
| Ridge Regression | 0.68 |

> ➤ RNN 显著优于静态模型，体现时序建模的重要性。

#### （C）输入序列配置
| 序列长度 | NMSE |
|---------|-------|
| 10 × 300ms（步长 100ms） | **0.43** ✅ |
| 5 × 300ms | 0.46 |
| 单窗口 1200ms | 0.49 |

> ➤ 序列越长，上下文越充分，性能趋于收敛于 10 窗口。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Riemannian 特征（CMTS）在 sEMG 回归中具有强大表征能力**，优于传统 TDF 和端到端 CNN。
2. **TRR 模型实现了精度与效率的平衡**：
   - 准确性超越 SOTA（如 vemg2pose）
   - 推理速度快（Raspberry Pi 5 上达 10 Hz）
   - 功耗低，温度上升缓慢，适合边缘部署
3. **自由手势采集更贴近真实应用需求**，避免引导动作带来的偏差。
4. **跨被试性能仍有差距**，但 TRR 表现稳健，具备一定泛化潜力。

### ⚠️ 局限性
1. **应用场景受限**：
   - 数据采集排除握力、物体操作、手臂移动等现实干扰因素；
   - 未考虑长期使用中的信号漂移（如电极位移、疲劳、阻抗变化）。
2. **未探索量化压缩技术**：
   - 如量化（quantization）、剪枝等可进一步提升嵌入式部署效率的方法尚未尝试。
3. **训练数据量要求较高**：
   - 消融实验显示，TRR 性能随训练数据增加持续提升，暗示其对大数据依赖较强。

### 🔮 未来工作方向
1. **增强跨会话（cross-session）鲁棒性**：
   - 引入领域自适应（domain adaptation）、在线校准或持续学习策略应对信号漂移。
2. **扩展至实际任务环境**：
   - 在有负载、抓取物体、全身运动条件下验证模型有效性。
3. **探索更高效的模型变体**：
   - 尝试量化版 TRR 或轻量 Transformer 替代 GRU，进一步优化延迟与内存占用。
4. **闭环控制系统集成**：
   - 将 TRR 集成进假肢或 AR 设备中，开展用户研究评估可用性与用户体验。

---

> 💡 **总体评价**：本文通过“轻量特征 + 轻量模型”的设计哲学，在保证高精度的同时极大提升了实用性与可部署性，为下一代嵌入式 EMG 控制系统提供了可靠的技术路径。

</details>

---

### 7. [Fast Neural-Network Approximation of Active Target Search Under Uncertainty](https://arxiv.org/abs/2604.22254)

**Authors**: Bilal Yousuf, Zsofia Lendek, Lucian Busoniu  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.22254v1  

#### Abstract
We address the problem of searching for an unknown number of stationary targets at unknown positions with a mobile agent. A probability hypothesis density filter is used to estimate the expected number of targets under measurement uncertainty. Existing planners, such as Active Search (AS) and its In...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Fast Neural-Network Approximation of Active Target Search Under Uncertainty

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
本文针对**在不确定环境下使用移动智能体搜索未知数量且位置未知的静态目标**这一经典 Active Target Search（AS）问题，解决其**在线规划计算成本过高**的瓶颈。

传统基于模型的方法如 **Active Search (AS)** 和 **Active Search with Intermittent measurements (ASI)** 虽然能实现高精度检测，但依赖每一步的在线优化（online optimization），导致实时性差，难以部署于资源受限平台。

---

### 🚀 提出的新方法与创新思路

提出一种基于 **Convolutional Neural Network (CNN)** 的数据驱动策略，用于近似 AS 和 ASI 规划器的决策过程：

- **直接推理替代优化**：训练一个 CNN 模型，将环境状态编码为多通道空间网格（multi-channel grid），直接输出下一个最优 waypoint，避免重复求解优化问题。
- **输入表示设计**：构建四通道输入张量 $ Z_{\text{CNN}} \in \mathbb{R}^{n_g \times n_g \times 4} $，分别编码：
  1. **Visitation History**：访问频率（线性衰减记忆）
  2. **Target Belief Density**：PHD filter 的粒子权重经高斯平滑后的密度图
  3. **Agent Position**：智能体当前位置的一热编码（one-hot）
  4. **Boundary Proximity Mask**：边界区域掩码，防止传感器视野越界
- **探索机制革新**：引入“**无显式探索项**”版本的 AS/ASI，利用 PHD filter 中的 **birth intensity** 自然引导对新区域的探索，避免盲目探索带来的低效。

该方法属于 **explicit MPC** 思路，即用神经网络学习隐式的控制律，实现快速推断。

---

### 🔍 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **计算效率** | 推理速度提升**数个数量级**（AS: ~1.4ms → ~0.13ms；ASI: ~147ms → ~0.14ms） |
| **可扩展性** | CNN 推理时间不随候选 waypoint 数量增加而上升，而 AS/ASI 成本急剧上升 |
| **实用性** | 可部署于嵌入式系统或无人机等资源受限设备 |
| **性能保持** | 在检测率上与原始 AS/ASI **统计无差异**，仅初期略有滞后，后期反超 |

---

## 2. 核心实验方法和设置

### 📊 数据集与训练数据生成

- **非公开真实数据集**，而是通过**仿真环境自动生成训练数据**。
- 使用 **PHD filter + AS / ASI 规划器** 在多个独立仿真实验中收集决策样本。
- **训练数据构成**：
  - 输入：四通道空间网格 $ Z_{\text{CNN}} $
  - 输出标签：AS 或 ASI 选择的最优 waypoint 坐标（归一化至 [0,1]²）
- **训练样本总量**：
  - AS 数据集：60 次试验 × 250 步 = **9,120 样本**
  - ASI 数据集：60 次试验 × 50 决策步 = **3,120 样本**

---

### ⚙️ 实验设置

- **环境大小**：$ E = [0, 260]\text{m} \times [0, 260]\text{m} $
- **智能体模型**：简化线性模型（Parrot Mambo drone）
- **控制周期**：$ T_s = 0.005 $s，测量周期 $ \Delta = 10 \times T_s = 0.05 $s
- **初始位置固定**：$ q_0 = [10, 10] $
- **目标分布两种设定**：
  - **Uniform**：18 个目标均匀随机分布
  - **Clustered**：3 个簇，每簇 5–6 个目标，簇间分离
- **CNN 输入分辨率**：$ 26 \times 26 $ 网格（对应每格 10m×10m）

---

### 🎯 评估指标

| 指标 | 描述 |
|------|------|
| **平均检测目标数** | 随时间变化的目标发现数量，报告均值 ± 95% 置信区间 |
| **单步规划耗时** | 每次决策所花费的时间（CPU 时间） |
| **消融实验性能对比** | 移除某一输入通道后检测性能的变化 |
| **可扩展性测试** | 改变候选 waypoint 数量，观察计算开销增长趋势 |

---

### 🆚 基线方法对比

| 方法 | 类型 | 是否在线优化 |
|------|------|---------------|
| **AS** | Model-based, MPC-style | 是（每步优化） |
| **ASI** | Model-based, long-horizon + intermittent | 是（更复杂优化） |
| **CNN-based Planner** | Data-driven, direct inference | 否（前向传播即可） |

所有方法在同一仿真平台上运行，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ E1: CNN vs. AS（Uniform & Clustered）

- **检测性能**：CNN 与 AS **几乎完全重合**，最终检测数无显著差异
- **初期略慢**：前 50 步 CNN 略落后，但后期追上并更快找到最后几个目标
- **计算时间**：
  - AS：$ 1.4 \times 10^{-3} \pm 2.7 \times 10^{-3} $ s/step
  - CNN：$ 1.3 \times 10^{-4} \pm 9.0 \times 10^{-5} $ s/step  
  ➜ **加速约 10 倍**

#### ✅ E2: CNN vs. ASI

- **检测曲线高度一致**，尤其在 clustered 场景下匹配良好
- **计算时间差距巨大**：
  - ASI：$ 1.475 \times 10^{-1} \pm 3.3 \times 10^{-2} $ s/step （≈147 ms）
  - CNN：$ 1.38 \times 10^{-4} \pm 1.04 \times 10^{-4} $ s/step （≈0.14 ms）  
  ➜ **加速超过 1000 倍**

> 💡 注：实验平台为 Intel i3-1365U CPU，32GB RAM，MATLAB R2024

#### ✅ E3: 候选 waypoint 数量影响（可扩展性）

| 方法 | 候选数增加的影响 |
|------|----------------|
| **AS / ASI** | 计算时间随候选数**指数级上升**（boxplot 显示极大方差） |
| **CNN** | 推理时间**恒定不变**，不受候选集规模影响 |

- **AS with 16 directions**：比 8 方向更快发现目标 → CNN 能成功学习此精细动作空间策略
- **ASI with 144 waypoints (20m spacing)**：计算负担极重 → CNN 完全规避此问题

#### ✅ E4: 消融实验（Ablation Study）

| 输入通道移除 | 平均检测数（clustered） | 影响分析 |
|-------------|--------------------------|--------|
| **Original (4-channel)** | **15.0 ± 0.2** | 基准 |
| **No visitation history** | 7.67 ± 1.45 ❌ | 性能腰斩！说明历史信息至关重要 |
| **No Gaussian smoothing** | 8.87 ± 2.03 ❌ | 平滑显著提升稳定性 |
| **No agent position** | 10.87 ± 1.67 ⚠️ | 影响较小 → 位置可能被 visitation 编码隐含表达 |
| **No boundary mask** | 12.09 ± 1.20 ⚠️ | 导致 agent 漂向边缘，遗漏角落目标 |

👉 结论：**visitation history 和 Gaussian smoothing 最关键**，验证了输入设计的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **CNN 可高效逼近 AS/ASI 的决策行为**，在多种目标分布下达到**统计等效的检测性能**。
2. **计算效率飞跃**：从毫秒级优化降至亚毫秒级推理，适合实时应用。
3. **无需显式探索项**：通过 PHD filter 的 birth intensity 即可自然促进探索，在稀疏目标场景更高效。
4. **输入通道设计合理**：多通道空间编码有效融合 belief、history、position、boundary 信息，消融实验证明各组件均有贡献。

---

### ⚠️ 局限性

1. **泛化能力有限**：训练数据基于特定环境尺寸、传感器模型和目标分布，**换场景需重新训练**。
2. **静态环境假设**：未考虑动态障碍物或移动目标。
3. **单智能体框架**：尚未扩展到 multi-agent cooperation。
4. **误差分析缺失**：未量化 CNN 近似引入的策略偏差。

---

### 🔮 未来工作方向

1. **拓展至多智能体协同搜索**（multi-agent cooperative search）
2. **真实环境验证**：在实际无人机平台（如 Parrot Mambo）上部署测试
3. **增强泛化性**：研究 domain randomization 或元学习以提升跨场景适应能力
4. **误差建模与鲁棒性分析**：量化 CNN 替代带来的性能损失边界
5. **结合强化学习**：进一步优化端到端策略学习

---

## 总结

> 本文成功将 **data-driven CNN 方法引入 Active Target Search 领域**，实现了从“昂贵在线优化”到“快速直接推理”的范式转变。在保持 AS/ASI 高检测率的同时，**将计算成本降低 1–3 个数量级**，为将其部署于资源受限机器人系统提供了可行路径。其多通道输入设计和基于 PHD filter 的信念表示也为后续研究提供了重要参考。

</details>

---

### 8. [SpikingBrain2.0: Brain-Inspired Foundation Models for Efficient Long-Context and Cross-Platform Inference](https://arxiv.org/abs/2604.22575)

**Authors**: Yuqi Pan, Jinghao Zhuang, Yupeng Feng, Fangzhi Zhong, Siyu Ding, Xuerui Qiu, Shaowei Gu, Bohan Sun, Zhiyong Qin, Yibo Zhong, Lingtao Ouyang, Kun Yang, Zehao Liu, Yuhong Chou, Shurong Wang, Anjie Hu, Han Xu, Bo Xu, Guoqi Li  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.22575v1  

#### Abstract
Scaling context length is reshaping large-model development, yet full-attention Transformers suffer from prohibitive computation and inference bottlenecks at long sequences. A key challenge is to design foundation models that maintain performance and long-context efficiency with minimal training ove...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于 **full-attention Transformer** 的大模型在处理长上下文（long-context）时面临严重的效率瓶颈：
- **计算复杂度为 $O(n^2)$**，随序列长度增长呈二次方上升；
- **KV Cache 内存占用线性增长**，导致推理时内存受限；
- 难以部署于资源受限环境（如边缘设备、神经形态芯片）。

该论文旨在解决如何构建一种既能保持高性能，又能高效支持超长上下文推理，并具备跨平台兼容性的 **Foundation Model 架构**。

---

### 提出的新方法与创新思路

#### （1）Dual-Space Sparse Attention (DSSA)
提出一种新型混合注意力架构 DSSA，在层间非均匀地组合两种稀疏注意力机制：
- **Sparse Softmax Attention (MoBA)**：保留部分精确的 softmax 注意力模块，用于关键层的信息整合。
- **Sparse Linear Attention (SSE)**：采用线性注意力压缩状态表示，实现 $O(n)$ 复杂度。
- 比例为 **1:3**，即每4层中有1层使用 MoBA，其余使用 SSE。
- 最后一层仍保留 Full Attention (FA)，以稳定训练过程。

> ✅ **脑启发设计**：模拟生物记忆系统中的“选择性门控”与“竞争抑制”，减少噪声干扰，提升长期依赖建模能力。

#### （2）Dual-Path Activation Coding（双路径激活编码）
为适配不同硬件平台，提出两条量化路径：
- **INT8-Spiking Path**：
  - 将 INT8 激活值通过 **bitwise coding** 转换为稀疏脉冲序列；
  - 支持在 **neuromorphic chips** 上进行事件驱动（event-driven）计算；
  - 显著降低功耗与面积开销。
- **FP8 Path**：
  - 使用 **FP8 (E4M3)** 格式加速矩阵乘法；
  - 利用 NVIDIA Hopper GPU 的 Tensor Cores 实现实际推理加速；
  - 兼容主流 GPU 生态。

#### （3）优化的 Transformer-to-Hybrid (T2H) 转换流程
从预训练的 Qwen3-4B 出发，通过轻量级转换获得 SpB2.0：
- **LLM 路径**：四阶段流程 —— 短上下文蒸馏 → 长上下文扩展（至 512k）→ 通用 SFT → 推理导向 SFT。
- **VLM 路径**：两阶段流程 —— 知识蒸馏 + 指令微调。
- 总训练成本：<7k A100 GPU 小时（远低于重新预训练）。

---

### 相比现有方法的优势

| 维度 | SpB2.0 优势 |
|------|-------------|
| **长上下文效率** | 在 4M context 下达到 **10.13× TTFT 加速**；支持 >10M tokens 推理（基线 OOM） |
| **跨平台兼容性** | 同时支持 GPU（FP8）和神经形态芯片（INT8-Spiking）部署 |
| **能效表现** | INT8-Spiking 路径实现 **64.31% spike sparsity**，硬件仿真显示 **70.6% 面积缩减、46.5% 功耗下降** |
| **训练成本** | 总训练 <7k A100 小时，恢复 Qwen3-4B 大部分能力 |
| **性能保留** | 在多项基准上达到 Qwen3-4B 的 >95%，优于同规模开源模型 |

---

## 2. 核心实验方法和设置

### 使用的数据集

#### LLM 训练数据
- **Continual Pre-training (CPT)**：
  - `ProLong`（自然长文本）
  - `Nemotron-CC-v2` 高质量子集（比例 1:3）
- **SFT 数据**：
  - 数学、代码、对话任务混合数据
  - `DAPO-Math-17k` 用于 on-policy distillation

#### VLM 训练数据
- **知识蒸馏**：`LLaVA-NeXT-Data`
- **指令微调**：`LLaVA-OneVision-1.5-Instruct-Data` + OCR/PixMo 数据增强视觉理解

---

### 实验设置与评估指标

#### 主要评估维度
| 类别 | 指标 |
|------|------|
| **能力评估** | MMLU, ARC-C, GSM8K, HumanEval, LongBench, MMStar, MMMU 等 |
| **效率评估** | TTFT (Time to First Token), TPOT (Time Per Output Token), End-to-End Latency, Throughput |
| **硬件效率** | Area, Power (@250MHz / 500MHz), Spike Sparsity |
| **量化影响** | Accuracy Drop after INT8/FP8 Quantization |

#### 基线模型对比
- **LLM Baselines**：
  - Qwen3-4B, Qwen2.5-3B, Gemma3-4B, Llama3.2-3B
  - SpB1.0-7B（前代模型）
- **VLM Baselines**：
  - Qwen3-VL-4B, Qwen2.5-VL-3B, LLaVA-OneVision-7B, InternVL2-4B

#### 硬件测试平台
- **GPU 测试**：8× A100/H100，使用 HuggingFace + vLLM 框架
- **神经形态硬件仿真**：Synopsys Design Compiler + Gate-level Simulation（28nm 工艺）

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 结果 |
|------|------|
| **最大支持上下文长度** | >10M tokens（启用 chunked prefill） |
| **TTFT 加速比（4M context）** | **10.13×** vs Qwen3-4B（HuggingFace SP） |
| **vLLM 下综合加速（512k context）** | TTFT: 4.5×, TPOT: 1.12×, E2E: 4.3× |
| **请求并发数提升** | **3.17×** 更高并发（因 KV Cache 占用更低） |
| **FP8 推理加速（250k context）** | **2.52× TTFT speedup**，精度损失仅 0.24% |
| **INT8-Spiking 脉冲稀疏度** | **64.31%**（即 35.69% firing rate） |
| **硬件能效增益（vs INT8 baseline）** | 面积 ↓70.6%，功耗 ↓48.1%@250MHz / ↓46.5%@500MHz |
| **总训练成本** | <7k A100 GPU 小时（LLM: ~4.7k, VLM: ~2.0k） |

---

### 与基线方法的对比结果

#### LLM 性能对比（Post-SFT）
| 模型 | MMLU | GSM8K | HumanEval | LongBench-32k |
|------|-------|--------|------------|---------------|
| Qwen3-4B | 72.45 | 89.16 | 86.59 | 40.00 |
| SpB2.0-5B | **69.06** | **80.97** | **71.34** | **37.04** |
| Qwen2.5-3B | 66.43 | 74.07 | 74.39 | 28.22 |
| SpB1.0-7B | 66.34 | 66.87 | 40.24 | 27.50 |

> 🔹 SpB2.0-5B 明显优于 Qwen2.5-3B 和 SpB1.0-7B，接近 Qwen3-4B 表现。

#### VLM 性能对比
| 模型 | MMStar | MMMU(val) | ScienceQA | OCRBench |
|------|--------|-----------|------------|----------|
| Qwen3-VL-4B | 64.40 | 55.78 | 93.42 | 84.50 |
| SpB2.0-VL-5B | **55.40** | **50.33** | **81.55** | **75.10** |
| Qwen2.5-VL-3B | 55.80 | 51.11 | 79.54 | 82.60 |

> 🔹 SpB2.0-VL-5B 接近甚至超过同级别开源 VLM，有效恢复 Qwen3-VL 的多模态能力。

---

### 消融实验结果

#### （1）Layer Selection 策略对比（Table 1）
| 方法 | MMLU | LB-32k | IFEval |
|------|------|--------|--------|
| Uniform Interleaving | 55.23 | 13.65 | 40.53 |
| **Greedy Sensitivity Selection** | **60.32** | **14.06** | **47.72** |

> ✅ 基于敏感性分析的 layer selection 显著优于均匀交错。

#### （2）SSE 配置消融（Table 2）
| 配置 | Train Loss | MMLU |
|------|------------|--------|
| SSE-GLA | 0.955 | 44.97 |
| SSE-GDN | **0.871** | **49.67** |

> ✅ GDN 版本收敛更快、性能更强。

#### （3）SSE-SWA 分支消融（Table 22）
| 模式 | MMLU |
|------|------|
| Full Model | 69.38 |
| SSE-only | 62.15 |
| SWA-only | 23.69 |

> ✅ 辅助 SWA 分支不影响主 LA 学习；移除后性能大幅下降，说明其对训练稳定性有帮助。

---

## 4. 关键结论和发现

### 主要发现
1. **DSSA 架构实现了优异的性能-效率权衡**：
   - 在极低训练成本下恢复 Qwen3-4B 的大部分能力；
   - 支持超长上下文（>10M tokens），显著超越 full-attention 基线。

2. **双路径量化策略打通异构部署路径**：
   - FP8 路径适用于当前主流 GPU，提供实用加速；
   - INT8-Spiking 路径面向未来神经形态硬件，具备巨大节能潜力。

3. **T2H 转换范式高效可行**：
   - 不需要从头预训练即可完成架构迁移；
   - 使用全开源数据即可实现高质量转换。

4. **脑启发机制具有实际工程价值**：
   - 稀疏记忆（sparse memory）、事件驱动等机制不仅符合生物学原理，也能带来真实性能收益。

---

### 方法的局限性
- **AIME 等高难度数学任务仍有差距**：可能由于训练数据中缺乏足够难样本。
- **FP8 实现依赖特定硬件（Hopper）**：尚未广泛普及。
- **MoBA 的 paged attention 实现尚不完善**：限制了 chunked prefill 场景下的理论加速上限。
- **VLM 性能略逊于原生训练模型**：仍有进一步优化空间。

---

### 未来工作方向
1. **探索更大规模的 SpB 模型**（如 10B+），验证可扩展性；
2. **引入 RL/PPO 进行深度推理优化**：已有初步 on-policy distillation 成果；
3. **开发专用神经形态芯片支持 INT8-Spiking 推理**；
4. **进一步优化 MoBA 的稀疏调度算法与内存管理**；
5. **拓展到更多模态（音频、视频流）与 agent 应用场景**。

---

> 📌 **总体评价**：  
> **SpikingBrain2.0** 是首个将 **脑启发稀疏机制、高效注意力架构、双路径量化、轻量级转换训练** 完整融合的 Foundation Model 框架。它不仅在长上下文和跨平台推理方面取得突破，也为未来 **绿色 AI、边缘智能、类脑计算** 提供了切实可行的技术路径。

</details>

---

### 9. [$O(K)$-Approximation Coflow Scheduling in $K$-Core Optical Circuit Switching Networks](https://arxiv.org/abs/2604.22146)

**Authors**: Xin Wang, Hong Shen, Hui Tian, Ye Tao  
**Category**: cs.DC  
**Published**: 2026-04-27  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.22146v1  

#### Abstract
Coflow has emerged as a fundamental application-layer abstraction in distributed systems, representing communication dependencies and enabling collaborative management of related flows to enhance job completion efficiency. To meet the increasing bandwidth demands of modern data center networks (DCNs...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：$O(K)$-Approximation Coflow Scheduling in $K$-Core Optical Circuit Switching Networks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文研究了在**多核光电路交换（multi-core Optical Circuit Switching, OCS）网络**中，如何高效调度 **coflow** 以最小化总加权 **Coflow Completion Time (CCT)**。该问题面临三大挑战：
- **跨核流量分配**：多个独立 OCS 核并行运行，需合理分配流量。
- **端口互斥约束**：每个端口同一时间只能参与一个电路。
- **非零重构延迟（reconfiguration delay）**：电路切换带来不可忽略的时间开销。
- **异步重构模型（not-all-stop model）**：仅中断更新涉及的端口，其余传输继续，增加了调度复杂性。

此前针对此场景的研究极少，且缺乏理论性能保证。

---

### 提出的新方法与思路
作者提出了一种**基于线性规划引导的近似算法框架**，包含三个核心组件：
1. **LP-guided 全局 coflow 排序**：  
   通过求解一个基于顺序松弛的 **Linear Programming (LP) Relaxation**，获得每个 coflow 的完成时间下界 $T_m^{\text{LP}}$，并按其升序排列作为全局优先级。
   
2. **前缀感知的贪婪跨核流量分配（inter-core flow allocation）**：  
   按照排序后的顺序逐个处理 coflow，对每条 flow 分配到能使其所在 core 的“单核前缀下界” $T_B(D_{\leq m})$ 最小的 OCS 核上，避免某核成为瓶颈。

3. **核内贪婪最早可用端口匹配调度（intra-core circuit scheduling）**：  
   在每个 core 内部，按照全局顺序选择最早可建立电路的子流进行非抢占式调度，最大化资源利用率。

---

### 相比现有方法的优势
| 方面 | 本文优势 |
|------|--------|
| **理论保证** | 首次为 multi-core OCS 网络下的 coflow 调度提供 $O(K)$-approximation 性能保证，显著优于之前依赖于 $M \cdot \frac{w_{\max}}{w_{\min}} \cdot K$ 的弱保证 [Wang et al., 2026]。 |
| **适用性强** | 所提框架可自然推广至 multi-core EPS 网络，同样获得 $O(H)$-approximation。 |
| **模型更贴近现实** | 聚焦更具实用价值但更复杂的 **not-all-stop（异步）重构模型**，而非理想化的 all-stop 模型。 |
| **综合优化视角** | 同时考虑跨核耦合与核内约束，实现端到端协同优化。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Facebook Trace**：来自一个包含 3000 台机器、150 个机架的 MapReduce 集群的真实工作负载。
- 数据预处理：
  - 抽取 100 个 coflow 进行测试（从共 526 个中随机采样）。
  - 将接收者级别的通信信息转换为 sender-receiver flow 矩阵。
  - 映射到 $N$ 个逻辑源/目的端口（默认 $N=10$）。

---

### 实验设置
| 参数 | 默认值 |
|------|-------|
| 端口数 $N$ | 10 |
| coflow 数量 $M$ | 100 |
| OCS 核数量 $K$ | 3 |
| 核速率向量 | [10, 20, 30]（不平衡）或 [20,20,20]（平衡） |
| 总聚合速率 $R$ | 60 |
| 重构延迟 $\delta$ | 8 μs |

---

### 评估指标
- **归一化总加权 CCT（NormW）**：  
  $$
  \text{NormW}(A) = \frac{\sum w_m T_m(A)}{\sum w_m T_m(\text{OURS})}
  $$
  OURS 自身为基准（值为 1），越小越好。

- **尾部延迟指标**：p95 和 p99 CCT，反映长尾性能。

- **近似比（Approximation Ratio）**：  
  $$
  \text{Approx} = \frac{\sum w_m T_m(\text{OURS})}{\sum w_m T_m^{\text{LP}}}
  $$
  衡量实际性能与理论下界的差距。

---

### 基线方法（ablation baselines）
- **WSPT-ORDER**：用启发式优先级 $w_m / T_{\text{LB}}(D_m)$ 替代 LP-guided 排序。
- **SUNFLOW-S**：替换核内调度器为 Sunflow [20]。
- **BvN-S**：使用 Birkhoff-von Neumann 分解，在 all-stop 模型下运行。
- **LOAD-ONLY**：跨核分配只考虑负载均衡，忽略 reconfiguration 开销。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（默认设置下）
| 方法 | 归一化总加权 CCT | p95 CCT | p99 CCT |
|------|------------------|---------|---------|
| **OURS (本文)** | **1.00×** | **1.00×** | **1.00×** |
| WSPT-ORDER | **0.92×** | ~1.0 | ~1.0 |
| LOAD-ONLY | 1.37× | 1.33× | 1.32× |
| SUNFLOW-S | 1.38× | 2.22× | 2.26× |
| BvN-S | 4.34× | 6.89× | 7.07× |

> ✅ **OURs 在绝大多数情况下表现最优或接近最优**，尤其在尾部延迟方面远超其他方法。

---

### 不同配置下的稳定性表现
- **改变核数 $K=3,4,5$**：随着 $K$ 增大，OURs 相对于 LOAD-ONLY、SUNFLOW-S 和 BvN-S 的优势更加明显。
- **改变端口数 $N$（8~32）**：OURs 始终保持稳定领先。
- **改变重构延迟 $\delta$（2~12）**：OURs 对 $\delta$ 变化鲁棒，而 BvN-S 和 SUNFLOW-S 性能随 $\delta$ 上升急剧恶化。

---

### 消融实验结果
- **LOAD-ONLY vs OURs**：说明忽略 reconfiguration 开销会导致严重性能下降（+37% 加权 CCT），验证了联合建模的重要性。
- **SUNFLOW-S vs OURs**：表明即使使用先进调度器，若未与跨核分配协同设计，仍会因端口冲突导致尾部延迟激增。
- **BvN-S vs OURs**：all-stop 模型在异步环境中完全不适用，造成巨大性能损失（>300%）。
- **WSPT-ORDER 略优 OURs？**  
  在真实 trace 中，WSPT-ORDER 的加权 CCT 略低（0.92×），但这源于 workload 特性（heavy-tailed flow sizes），并不否定 LP-guided 方法的理论稳健性。

---

### 近似比分析
- 实际观测的 **approximation ratio** 在 2.5 ~ 5.0 之间，远低于理论最坏情况下的 $8K+1$（如 $K=3$ 时为 25）。
- **零释放时间** 下的 ratio 更低，符合预期（无时间约束更易调度）。
- 结论：**理论界限是保守的，但在实践中算法表现优异**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **首次为 multi-core OCS 网络中的 coflow 调度提供了 $O(K)$-approximation 保证**：
   - 零释放时间：$8K$-approximation
   - 任意释放时间：$(8K+1)$-approximation
   - 性能仅依赖架构参数 $K$，而非输入特征（如权重比、coflow 数量）。

2. ✅ **所提框架具有通用性**：可直接应用于 multi-core EPS 网络，获得 $4H$ 和 $(4H+1)$ 的 approximation ratio。

3. ✅ **LP-guided ordering 是支持理论分析的关键工具**，虽在特定 workload 下不如启发式，但提供了坚实的 worst-case 保障。

4. ✅ **跨核分配必须联合考虑传输负载与 reconfiguration 开销**，否则将导致严重性能退化。

5. ✅ **trace-driven 实验表明**：OURs 在总加权 CCT 和尾部延迟方面均显著优于代表性基线，验证了其实际有效性。

---

### 方法的局限性
- **离线假设**：当前算法基于所有 coflow 的 demand matrix 完全已知，属于 offline setting。
- **flow 不拆分**：为避免乱序和控制开销，flow 必须完整分配到单一 core，可能牺牲灵活性。
- **理论界限较松**：虽然实践表现好，但 $8K+1$ 的 bound 在大 $K$ 场景下仍偏高。

---

### 未来工作方向
- **Online Coflow Scheduling**：研究动态到达场景下的在线算法，并建立 competitive ratio 理论保证。
- **部分可观 demand matrix**：在 demand 信息不完整或预测误差存在的情况下设计鲁棒调度策略。
- **混合 OCS/EPS 多核架构扩展**：将本框架推广至更复杂的 hybrid multi-core 网络。
- **降低 approximation ratio**：探索更紧致的下界构造方式，缩小理论与实践差距。

</details>

---

### 10. [QuantClaw: Precision Where It Matters for OpenClaw](https://arxiv.org/abs/2604.22577)

**Authors**: Manyi Zhang, Ji-Fu Li, Zhongao Sun, Xiaohao Liu, Zhenhua Dong, Xianzhi Yu, Haoli Bai, Xiaobo Xia  
**Category**: cs.AI  
**Published**: 2026-04-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.22577v1  

#### Abstract
Autonomous agent systems such as OpenClaw introduce significant efficiency challenges due to long-context inputs and multi-turn reasoning. This results in prohibitively high computational and monetary costs in real-world development. While quantization is a standard approach for reducing cost and la...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《QuantClaw: Precision Where It Matters for OpenClaw》总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
自主智能体系统（如 **OpenClaw**）在实际部署中面临严重的效率挑战，主要体现在：
- **长上下文输入** 和 **多轮推理** 导致计算开销巨大；
- 当前系统通常采用**固定精度配置**（如 BF16 或 FP8），无论任务复杂度如何，造成资源浪费；
- 统一量化（quantization）虽能降低成本和延迟，但可能损害对精度敏感的任务性能。

因此，**如何在不牺牲任务性能的前提下，动态优化 agent 系统的计算成本与延迟**，成为一个关键问题。

### 🚀 提出了什么新方法或新思路
作者提出 **QuantClaw** —— 一种即插即用（plug-and-play）的**精度路由插件**（precision routing plugin），其核心思想是：

> 将 **precision 视为一种可动态分配的运行时资源**，而非固定的模型属性。

具体创新包括：
- **任务感知的精度分配机制**：基于任务类型自动选择执行精度（如 16-bit、8-bit、4-bit）；
- **动态路由策略**：轻量级任务走低精度路径以节省成本，高敏感任务保留高精度保障质量；
- **无需用户干预**：作为服务端透明层运行，用户无需感知或管理精度权衡。

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | QuantClaw |
|------|--------|---------|
| 精度策略 | 固定精度（all-BF16 或 all-INT4） | 动态按需分配 |
| 成本控制 | 被动压缩，可能导致性能下降 | 主动优化，在性能稳定前提下降本增效 |
| 用户体验 | 需手动选择模型/精度 | 完全透明，零额外复杂度 |
| 适用性 | 单一配置难以适应多样化任务 | 支持多种部署目标（latency-oriented / cost-oriented） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **Claw-Eval (v0.0.0)**：用于初步分析量化敏感性的基准套件。
  - 包含 **24 种任务类型**、**104 个人工验证任务**；
  - 覆盖领域广泛：服务编排、多模态感知、多轮对话等；
  - 评估维度：完成率、安全性、鲁棒性，并支持轨迹级审计。
- **PinchBench (v1.2.0 和 v2.0.0)**：主实验所用评测平台，更贴近真实 OpenClaw 工作流。

### ⚙️ 实验设置和评估指标
#### 模型集合（共6个）
| 模型 | 参数量 | 原生精度 |
|------|-------|----------|
| GLM-4.7-Flash | 30B | BF16 |
| GLM-5 | 744B | FP8 |
| MiniMax-M2.5 | 229B | BF16 |
| Qwen3.5-9B ~ Qwen3.5-397B-A17B | 9B–397B | BF16 |

所有模型均被量化至 **NVFP4** 或 **INT4** 进行对比。

#### 量化设置
- 对比从原生精度（BF16/FP8）到低精度（NVFP4/INT4）的影响；
- 成本估算依据行业惯例：INT4 token 价格设为 BF16 的 85%，NVFP4 为 80%。

#### 评估指标
| 指标 | 含义 |
|------|------|
| **Score** | 任务完成得分（越高越好） |
| **Cost ($)** | 推理总费用（越低越好） |
| **Time (s)** | 端到端延迟（越低越好） |
| **Throughput (tok/s)** | 输出吞吐量（越高越好） |

每组实验重复 6 次以减少随机性影响。

### 🔁 基线方法对比
- **All BF16 / All FP8**：高精度全量运行，代表高质量但高成本方案；
- **All INT4 / All NVFP4**：统一低精度运行，代表低成本但可能损失性能；
- **QuantClaw**：动态路由策略，结合两者优势。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 2）

#### 在 **GLM-4.7-Flash + PinchBench v1.2.0**
| 方法 | 平均 Score | 成本 ($) | 延迟 (s) |
|------|-----------|----------|--------|
| All BF16 | 81.26 | 0.001598 | 19.07 |
| All INT4 | 78.71 | 0.001422 | 21.80 |
| **QuantClaw** | **84.11** (+2.85) | **0.001252** (-21.7%) | **17.47** (-8.4%) |

✅ 结果：**性能提升 + 成本下降 + 延迟降低**

#### 在 **GLM-5 + PinchBench v2.0.0**
| 方法 | 平均 Score | 成本 ($) | 延迟 (s) |
|------|-----------|----------|--------|
| All FP8 | 87.08 | 0.0196 | 62.22 |
| All INT4 | 81.92 | 0.0169 | 58.99 |
| **QuantClaw** | **85.59** (+2.09) | **0.0154** (-21.4%) | **52.46** (-15.7%) |

✅ 结果：**超越两个极端基线**，实现“鱼与熊掌兼得”。

> 💡 总结：QuantClaw 在保持甚至提升任务性能的同时，实现了：
> - 最高 **21.4% 的成本节约**
> - 最高 **15.7% 的延迟降低**

### 🔍 消融实验结果（Table 3：任务检测方法比较）

| 检测方法 | Accuracy (%) | Macro F1 (%) | 查询耗时 (s) |
|--------|--------------|---------------|----------------|
| RuleDetector | 83.13 | 65.90 | 0.0017 |
| BGE-M3 | 89.76 | 86.56 | 0.0200 |
| GLM-5-FP8 | 92.17 | 89.72 | 0.1717 |
| **RuleDetector + BGE-M3** | **91.53** | **88.66** | **0.0149** |

📌 发现：
- 单独使用规则或大模型判断各有优劣；
- **混合策略（hybrid detection）** 在准确性和效率之间取得最佳平衡；
- 默认采用 `RuleDetector + BGE-M3` 作为任务分类器。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **量化敏感性高度任务依赖**（见 Figure 2）：
   - 高敏感任务（如代码生成、合规检查、安全决策）：对精度下降极为敏感；
   - 低敏感任务（如研究、理解、数据分析）：在低精度下仍稳健，甚至略有增益（正则化效应）；
   - 中等敏感任务（如重写、内容生成）：可接受混合精度。

2. **模型规模越大，量化鲁棒性越强**（见 Figure 1）：
   - 小模型（如 9B）量化后性能显著下降；
   - 大模型（>200B，如 GLM-5）几乎不受影响，部分反而提升；
   - 存在明显的 **scaling law**：△ ∝ N⁻⁰.²⁹³，表明大模型天然适合低精度部署。

3. **动态精度路由优于任何静态策略**：
   - 不是简单地“全部降精度”，而是“该省的地方省，该花的地方花”；
   - 实现了严格帕累托前沿改进（strictly better operating point）。

### ⚠️ 方法的局限性
- **依赖离线构建的任务-精度敏感性 profile**：需要预先进行大量实验建模；
- **任务检测模块引入额外开销**：虽然轻量，但在极端低延迟场景中仍需权衡；
- **当前仅适配 OpenClaw 生态**：扩展至其他 agent 框架需重新集成；
- **未考虑动态上下文变化**：当前基于初始 query 分类，未处理执行过程中任务漂移。

### 🔮 未来工作方向
1. **在线自适应 profile 更新**：通过反馈闭环持续优化精度映射策略；
2. **跨模型协同调度**：将 precision routing 扩展为 multi-model selection，实现能力与成本联合优化；
3. **支持更多量化格式**：如 FP6、Hifloat 等新兴低比特格式；
4. **应用于边缘设备或移动端 agent**：进一步推动轻量化个人 AI 系统发展；
5. **探索 MoE + Quantization 联合优化**：结合 Mixture-of-Experts 架构实现细粒度资源调配。

---

## 📌 总结一句话
> **QuantClaw 揭示了一个重要范式转变：在 agent 系统中，precision 不应是固定配置，而应是一种可根据任务需求动态调度的核心资源。** 通过任务感知的精度路由，它实现了性能、成本与延迟的三赢，为高效自治 agent 的规模化落地提供了实用解决方案。

</details>

---

### 11. [BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering](https://arxiv.org/abs/2604.22678)

**Authors**: Jinghong Chen, Jingbiao Mei, Guangyu Yang, Bill Byrne  
**Category**: cs.CL  
**Published**: 2026-04-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.22678v1  

#### Abstract
A common approach to question answering with retrieval-augmented generation (RAG) is to concatenate documents into a single context and pass it to a language model to generate an answer. While simple, this strategy can obscure the contribution of individual documents, making attribution difficult an...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统 **Concatenative RAG (ConcatRAG)** 存在以下关键问题：
- **“lost-in-the-middle”效应**：当相关文档位于长上下文中间时，模型容易忽略其内容。
- **计算开销大**：注意力机制的复杂度随上下文长度呈二次增长，尤其在多模态场景下（如视觉问答）更严重。
- **缺乏可解释性**：无法量化单个检索文档对生成答案的贡献，难以进行归因分析或重排序。

### 🚀 提出的新方法：**BERAG & BEFT**
作者提出了一套新的 RAG 框架：
- **Bayesian Ensemble RAG (BERAG)**：一种基于贝叶斯集成的推理方法，不再将所有文档拼接成一个长上下文，而是为每个文档独立生成 token 概率，并通过**文档后验概率**作为动态权重进行加权融合。
- **Bayesian Ensemble Fine-Tuning (BEFT)**：端到端训练 BERAG 的监督微调方法，利用 token 级别的损失函数优化文档先验与后验分布。

### 🔍 相比现有方法的优势
| 特性 | BERAG 优势 |
|------|------------|
| **可扩展性** | 支持并行处理多个短上下文，突破显存限制（例如可在单张 A100-80G 上运行 Top-K=50 推理） |
| **抗“lost-in-the-middle”** | 文档顺序无关，性能不受位置影响 |
| **可解释性** | 后验概率提供清晰的文档贡献度量，支持证据选择、归因和 deflection 判断 |
| **高效解码** | 可通过 Top-P 剪枝低概率分支实现比 ConcatRAG 更快的 decoding 速度 |
| **灵活集成** | 自然支持多文档联合推理，适用于 SlideVQA 等需跨页合成信息的任务 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
| 数据集 | 类型 | 描述 |
|-------|------|------|
| **E-VQA** | KB-VQA | 基于维基百科的知识型视觉问答，涉及动植物等细粒度知识 |
| **Infoseek** | KB-VQA | 视觉信息检索类问题，依赖外部知识回答图像相关问题 |
| **SlideVQA** | DocVQA | 基于 20 页 PPT 的多图文档问答，要求数值推理与跨页综合 |
| **MMNeedle (Multimodal Needle-in-a-Haystack)** | 多模态定位任务 | 在大量图像面板中精确定位目标子图（含坐标输出） |

### ⚙️ 实验设置与评估指标
| 方面 | 设置 |
|------|------|
| **模型架构** | 使用 Qwen2-VL-Instruct 和 LLaVA-Llama-3-8B 作为 VLM；PreFLMR-L 用于检索 |
| **输入长度** | 测试不同 Top-K 设置（K=1~50），部分超过 32K 上下文窗口 |
| **评估指标** | 
| - E-VQA：BERT Exact Match (BEM)  
| - Infoseek / SlideVQA：Exact Match (EM)  
| - MMNeedle：“Exact” 准确率（正确预测 panel index + row-col 坐标）  
| - Recall@K：衡量检索质量  
| - Deflection Accuracy/F1：判断是否应拒绝回答的能力 |
| **推理方式** | Greedy decoding，FP16 推理 |

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **Standard RAG (ConcatRAG)** | 将 Top-K 文档拼接输入 |
| **SFT (Supervised Fine-Tuning)** | 在 Top-5 文档上进行标准微调 |
| **DPO (Direct Preference Optimization)** | 基于偏好数据优化生成器 |
| **SoTA 系统** | 包括 MuKA、EchoSight、Reflectiva、GPT-4o-mini 等近期先进系统 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ E-VQA 和 Infoseek 性能（表1）
| 方法 | E-VQA (BEM) | Infoseek (EM) | Retriever Recall@5 |
|------|-------------|---------------|---------------------|
| MuKA (SoTA) | 63.1 | — | 64.9 |
| EchoSight (SoTA) | — | 41.8* | 74.0* |
| **BEFT (ours)** | **70.3** | **42.8** | 60.6 / 36.6 |

> 💡 BEFT 在两个任务上均超越当前最优系统，且仅使用 PreFLMR-L 检索器。

#### ✅ 不同 K 下的性能变化趋势（表2）
- **ConcatRAG 基线**：性能在中等 K（~5–15）达到峰值后下降，受“lost-in-the-middle”影响明显。
- **BEFT**：性能持续随 K 增大而提升，在 K=50 达到最佳，**E-VQA 提升超 5% vs DPO**。
- 即使总上下文超出模型原生窗口（>32K），BERAG 仍可正常运行。

#### ✅ SlideVQA 表现（表3）
| 方法 | Evidence Selection EM | QA EM |
|------|------------------------|--------|
| VDocRAG | 73.3 | 44.2* (F1) |
| AVIR / Eagle-2.5 | — | ~60–63 |
| Human | 97.7 | 89.8 |
| **BEFT (ours)** | **90.4** | **69.6** |

> ⬆️ +15.4 分 ES 准确率，+6.4 分 QA 分数，显著优于现有 Select-then-Generate 或 Direct Generation 方法。

#### ✅ MMNeedle 多模态定位表现（表4）
| 方法 | 1×1 | 2×2 | 4×4 | 8×8 |
|------|-----|-----|-----|-----|
| GPT-4o | 97.0 | 81.8 | 26.9 | 1.0 |
| **BEFT (LLaVA-Llama-3-8B)** | **97.1** | **86.8** | **41.4** | 0.0 |
| → w/ 分割 8×8 → 4×4×4 | — | — | — | **42.5** |

> ✅ BEFT 在 N ≤ 4 场景下全面超越 GPT-4o 和 Claude 3 Opus；通过将 8×8 分割为四个 4×4 面板，有效绕过视觉编码瓶颈。

#### ✅ “Lost-in-the-Middle” 效应验证（图1）
- **ConcatRAG 系列模型**：当 GT 文档位于中间位置（如第 5–12 位）时性能大幅下降。
- **BEFT (BERAG)**：性能完全不受文档排序影响，始终保持最高水平。

#### ✅ 解码延迟对比（表6）
| 方法 | K=50, ms/token | BEM |
|------|----------------|-----|
| RAG | 203.0 | 56.3 |
| Naive BERAG | 470.2 | 70.3 |
| **BERAG + Top-P Pruning** | **44.4** | **70.7** |

> ⚡ 经剪枝后，BERAG 解码速度**远快于标准 RAG**，同时保持更高准确率。

#### ✅ Deflection 能力测试（表5）
| 方法 | Deflection Acc @K=5 | VQA Score (Strict RAG) |
|------|------------------------|--------------------------|
| BEFT | 72.3% | 57.8 |
| **BEFT[w/z₀]** | **83.3%** | **72.0** |

> ✅ 引入空文档 $ z_0 $ 训练后，模型能更可靠地识别无足够依据的情况并主动拒绝回答。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **BERAG 成功缓解“lost-in-the-middle”问题**：由于采用文档级并行处理与贝叶斯集成，推理过程对文档顺序完全鲁棒。
2. **支持大规模检索列表的有效利用**：可在 K=50 下稳定运行并持续受益于高 Recall@K，突破传统 RAG 的上下文长度瓶颈。
3. **具备天然的可解释性和可控性**：
   - 文档后验概率可用于证据归因、reranking 和 deflection 决策。
   - 支持构建可信、安全的问答系统。
4. **可通过剪枝实现加速**：Top-P pruning 显著降低延迟，甚至快于 ConcatRAG。
5. **泛化能力强**：在 KB-VQA、DocVQA、NIAH 等多种任务上均取得 SOTA 或接近 SOTA 表现。

### ⚠️ 局限性
1. **需要专门训练（Not Training-Free）**：
   - 直接应用原始 VLM 进行 BERAG 推理效果差，必须通过 BEFT 微调才能发挥优势。
   - 当前主流 LLM/VLM 均未预训练以支持此类集成机制。
2. **边际化局限于单文档**：
   - 当前框架只对单个文档进行 marginalization，未考虑文档组合（powerset）之间的交互，可能限制复杂推理能力。
3. **推理基础设施尚未优化**：
   - 当前库（如 Transformers）高度优化 ConcatRAG，BERAG 的 prefill 阶段仍有重复计算（如 query 编码多次执行），需工程层面进一步优化以提升吞吐量。

### 🔮 未来工作方向
- 扩展至 **集合级别的 ensemble**（marginalize over subsets of documents）以增强多跳推理能力。
- 开发专用推理引擎，减少共享 multimodal query 的重复编码。
- 探索 **zero-shot 或轻量适配**方式使 BERAG 可应用于未经 BEFT 训练的模型。
- 将文档 posterior 应用于更多下游任务，如自动 citation、fact-checking 和 agent planning。

---

> ✅ **总结一句话**：  
> **BERAG 提供了一个更高效、可解释、鲁棒且高性能的 RAG 新范式，特别适合知识密集型、长上下文、多模态的视觉问答任务，是 ConcatRAG 的强有力替代方案。**

</details>

---

### 12. [Towards Adaptive Continual Model Merging via Manifold-Aware Expert Evolution](https://arxiv.org/abs/2604.22464)

**Authors**: Haiyun Qiu, Xingyu Wu, Kay Chen Tan  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.22464v1  

#### Abstract
Continual Model Merging (CMM) sequentially integrates task-specific models into a unified architecture without intensive retraining. However, existing CMM methods are hindered by a fundamental saturation-redundancy dilemma: backbone-centric approaches face parameter saturation and representation int...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Towards Adaptive Continual Model Merging via Manifold-Aware Expert Evolution

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**Continual Model Merging (CMM)** 中存在的两个核心挑战提出解决方案：

- **饱和-冗余困境 (Saturation-Redundancy Dilemma)**：
  - **Backbone-centric 方法**（如 OPCM）在固定容量骨干网络中合并模型，易导致参数饱和和表征干扰，引发灾难性遗忘。
  - **MoE-based 方法**（如 MINGLE）为每个任务分配独立专家，虽缓解干扰，但无差别扩展导致专家冗余和架构膨胀。

- **显式路由瓶颈 (Explicit Routing Bottleneck)**：
  - MoE 方法依赖可学习的门控网络（gating network）进行专家选择，需要额外训练和数据优化，违背了 CMM “无数据、免训练”的初衷。

---

### 提出的新方法：MADE-IT
作者提出了 **MADE-IT (Manifold-Aware Dynamic Expert Evolution and Implicit rouTing)**，一种自适应的 CMM 框架，其核心创新在于：

#### （1）流形感知的动态专家演化（Manifold-Aware Dynamic Expert Evolution）
- **核心思想**：将专家表示从原始参数空间转移到其**主子空间 (principal subspace)** 上，并将其视为 Grassmann 流形上的点。
- **关键技术**：
  - 使用 **truncated SVD** 提取模块级权重更新的低秩主子空间。
  - 设计 **投影式子空间亲和度 (Projection-Based Subspace Affinity)** 度量，量化专家间的几何相似性。
  - 引入**分布感知的自适应阈值机制**，指导专家是“创建”还是“合并”，实现多样性与简洁性的平衡。

#### （2）免训练隐式路由（Data-Free and Training-Free Implicit Routing）
- **核心思想**：无需参数化门控网络，通过输入特征与专家主子空间的对齐程度来激活专家。
- **关键技术**：
  - **特征投影对齐 (Feature Projection Alignment, FPA)**：计算中间特征向专家输入子空间投影后的余弦相似度。
  - **专家依赖图 (Expert Dependency Graph)** 和路径一致性机制，确保跨模块的语义连贯性，解决分支歧义。

---

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **架构效率** | 显著减少冗余专家数量，尤其在通用模块和浅层，提升参数利用率。 |
| **性能表现** | 在长序列和乱序任务下均取得 SOTA 准确率，且鲁棒性强。 |
| **部署友好** | 完全免数据、免训练，推理阶段无需额外优化，适合边缘场景。 |
| **理论基础** | 基于流形几何，解决了参数空间度量因排列不变性和旋转敏感性导致的相关性误判问题。 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **CLIP-ViT** 架构（ViT-B/32, ViT-B/16, ViT-L/14）作为 backbone。
- 在 **20 个公开图像分类数据集** 上进行微调并合并，包括：
  - `SUN397`, `Stanford Cars`, `RESISC45`, `EuroSAT`, `SVHN`, `GTSRB`, `MNIST`, `DTD`, `Flowers102`, `PCAM`, `FER2013`, `Oxford-IIIT Pets`, `STL-10`, `CIFAR-10/100`, `Food-101`, `Fashion-MNIST`, `EMNIST`, `KMNIST`, `RenderedSST-2` 等。

### 实验设置
- **任务序列**：构建三种基准：
  - **短程**（8 任务）、**中程**（14 任务）、**长程**（20 任务）
- **随机种子**：10 次运行（seed 42–51），以评估任务顺序敏感性。
- **超参数**：全局设定 `rank ratio p = 0.1`, `margin coefficient β = 1.0`。

### 评估指标
| 指标 | 定义 |
|------|------|
| **ACC (Average Accuracy)** | 所有任务上的平均准确率 |
| **BWT (Backward Transfer)** | 衡量灾难性遗忘：$ \text{BWT} = \frac{1}{T}\sum_{i=1}^{T} [A_T(i) - A_i(i)] $，越接近 0 越好 |

### 基线方法对比
分为三类共 **11 种基线**：

#### （1）非合并范式
- `Pre-Trained`: 零样本迁移
- `Fine-Tuned`: 单独微调（上界）
- `C. Fine-Tuned`: 连续微调（灾难性遗忘严重）

#### （2）传统合并方法的连续版本
- `Average (SWA)`
- `C. Task Arithmetic`
- `C. Ties-Merging`
- `C. LW AdaMerging`
- `C. LoRA-WEMoE`

#### （3）专门的 CMM 方法
- `OPCM`: 正交投影法
- `MINGLE`: MoE + Null-Space Gating

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 方法 | ViT-B/32 (20-task ACC) | ViT-B/16 (20-task ACC) | ViT-L/14 (20-task ACC) |
|------|------------------------|------------------------|------------------------|
| MINGLE (SOTA) | 77.1 ± 2.0 | 81.9 ± 0.9 | 85.5 ± 1.3 |
| **MADE-IT (Ours)** | **81.4 ± 1.3** (+4.3) | **83.0 ± 0.1** (+1.1) | **89.6 ± 0.7** (+4.1) |

> ✅ MADE-IT 在所有架构上均显著超越最强基线 MINGLE，尤其在 ViT-B/32 上提升达 **4.3%**。

#### BWT 对比（遗忘控制）
| 方法 | ViT-B/32 (20-task BWT) |
|------|------------------------|
| MINGLE | -2.2 ± 0.8 |
| **MADE-IT** | **-3.8 ± 3.1** |

> 尽管 BWT 数值略差，但 MADE-IT 在大幅提升 ACC 的同时仍保持合理遗忘水平。

---

### 与基线方法的对比结果
- **全面优于所有基线**：在 ACC 指标上，MADE-IT 在所有任务长度和模型规模下均达到 SOTA。
- **远超 backbone-centric 方法**：相比 OPCM，在 20-task 下提升高达 **15.7%**（ViT-B/32）。
- **优于 MoE 方法**：相比 MINGLE，不仅精度更高，且专家数量更少，结构更紧凑。

---

### 消融实验结果

#### （1）子空间亲和度 vs. 参数空间相似度（Table 2）
| 方法 | ViT-B/32 ACC | ViT-B/16 ACC | ViT-L/14 ACC |
|------|--------------|--------------|--------------|
| w/ Cosine Similarity | 84.0 ± 2.6 | 85.1 ± 2.2 | 92.1 ± 0.7 |
| **w/ Subspace Affinity (Ours)** | **87.1 ± 1.6** | **90.4 ± 0.6** | **93.1 ± 0.7** |

> ✅ 流形几何度量显著优于传统的 cosine similarity。

#### （2）专家演化分析（Fig. 3 & 7）
- **组件维度**：MLP 模块保留最多专家（~11–12），而其他模块压缩率近 95%，说明 MLP 是任务特异性知识的主要载体。
- **深度维度**：浅层专家高度共享（压缩率 ~80%），深层专家趋向专业化（压缩率 ~40%），符合“通用→特定”的层次演化规律。

#### （3）超参数敏感性分析（Fig. 6）
- **Rank Ratio `p`**：当 `p > 0.6` 后性能急剧下降，表明过大的秩会引入噪声，破坏隐式路由的有效性。
- **Margin Coefficient `β`**：`β = 1.0` 时达到最佳平衡，过大则导致冗余，过小则抑制多样性。

---

## 4. 关键结论和发现

### 主要发现
1. **参数空间度量不可靠**：传统的 cosine similarity 因高维正交性而失效，无法捕捉专家间的真实功能相关性。
2. **流形几何是可靠基础**：基于主子空间的 Grassmann 流形表示能有效提取专家的本质特征，支持精准的相似性判断。
3. **动态专家管理可行**：通过自适应阈值机制，可在不牺牲性能的前提下自动合并冗余专家，实现架构精简。
4. **隐式路由高效可行**：无需任何可学习门控，仅通过特征-子空间对齐即可实现高效且一致的专家激活。

### 方法的局限性
- **依赖低秩假设**：方法有效性建立在 fine-tuning updates 具有低秩特性之上，若任务差异极大或更新密集，可能受限。
- **SVD 开销**：虽然轻量，但在大规模模型上频繁执行 SVD 可能带来一定计算负担。
- **对极端任务冲突敏感**：若多个任务在相同模块产生高度冲突的子空间，仍可能出现性能波动。

### 未来工作方向
- 探索更高效的子空间追踪算法，避免重复 SVD。
- 将 MADE-IT 扩展至 **LLM** 场景下的 Continual Model Merging。
- 结合稀疏激活机制进一步降低推理成本。
- 研究如何在动态演化中保留历史任务的可解释性。

---

> **总结**：MADE-IT 通过引入**流形几何视角**，从根本上解决了 CMM 中的饱和-冗余困境和路由瓶颈，实现了**高性能、高效率、免训练**的持续模型融合，为构建可持续演化的 Foundation Models 提供了新范式。

</details>

---

### 13. [Dynamically Acquiring Text Content to Enable the Classification of Lesser-known Entities for Real-world Tasks](https://arxiv.org/abs/2604.22325)

**Authors**: Fahmida Alam, Ellen Riloff  
**Category**: cs.CL  
**Published**: 2026-04-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.22325v1  

#### Abstract
Existing Natural Language Processing (NLP) resources often lack the task-specific information required for real-world problems and provide limited coverage of lesser-known or newly introduced entities. For example, business organizations and health care providers may need to be classified into a var...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Dynamically Acquiring Text Content to Enable the Classification of Lesser-known Entities for Real-world Tasks

## 1. 论文的主要贡献和创新点

### 解决的问题
- **现实世界中对小众实体（lesser-known entities）分类的需求难以满足**：许多实际应用（如企业分类、医疗提供者归类）需要将组织或个人归入特定的分类体系（如 SIC Codes 或 Healthcare Taxonomy Codes），但现有 NLP 资源（如 Wikidata、DBpedia）通常缺乏这些任务所需的细粒度、领域特定的信息。
- **缺乏描述性文本资源**：对于大量不知名的企业或医疗从业者，互联网上的公开描述有限，且没有现成的标注数据集支持监督学习。

### 提出的新方法与思路
- **提出一个通用框架**：仅需输入实体名称和对应的 gold labels，即可自动完成从文本获取到分类器训练的全过程。
- **动态文本获取机制（Dynamic Text Acquisition）**：
  - 结合 **Web Retrieval**（通过 Google 搜索获取 top-k snippets） 和 **LLM-based Generation**（利用 GPT-4o mini 和 LLaMA 3.1-8B INSTRUCT 生成任务相关摘要）来构建训练用文本。
  - 文本来源独立于预编译的知识库或语料库，实现“零先验文本依赖”。
- **端到端可扩展架构**：适用于不同领域和任务，无需人工编写描述文本。

### 相比现有方法的优势
| 对比维度 | 现有方法 | 本文方法 |
|--------|--------|--------|
| 数据依赖 | 依赖已有 KB（如 Wikipedia）、静态语料库 | 不依赖任何结构化知识源或预存文本 |
| 文本来源 | 静态检索（如 RAG 中的 Wikipedia） | 动态 Web 搜索 + 多 LLM 生成 |
| 应用灵活性 | 通常为推理服务增强（如 QA） | 用于训练监督分类模型 |
| 可复现性与责任共享 | 可能涉及第三方内容分发风险 | 仅发布重建指令，不直接分享原始 snippet 或 LLM 输出 |

> ✅ **核心创新**：首次将 **web-retrieved text** 与 **LLM-generated text** 联合用于构建监督分类任务的训练数据，解决了“无描述文本”的冷启动问题。

---

## 2. 核心实验方法和设置

### 使用的数据集
构建了两个跨领域的 benchmark 数据集：

#### （1）SIC Code Classification Dataset
- **任务**：将企业按其主营业务分类至两位数 SIC Codes（共 27 类）
- **来源**：基于 Jiang et al. (2023) 的食品系统研究数据集扩展
- **实体特点**：多为鲜为人知的小型企业（如 Multi-Corp International Inc.），非 Google、IBM 等大众公司
- **规模**：5,400 条记录 → 分割为 2,700（train）、900（dev）、1,800（test）

#### （2）Healthcare Provider Taxonomy Code Dataset
- **任务**：将个体医疗提供者分类至 NUCC 维护的 17 个专业类别
- **来源**：从 NPPES 公共注册系统提取
- **构建方式**：每个类别选取至少有 200 名从业者的 taxonomy code，随机采样 200 人，共 3,400 条
- **分割比例**：1,700（train）、560（dev）、1,140（test）

> ⚠️ 所有数据均来自公开可访问资源，符合伦理规范；未发布原始 snippet 或 LLM 输出，仅提供重构指南。

### 实验设置与评估指标
- **输入文本类型对比**：
  1. `GSnip`：Google top-10 snippets 拼接
  2. `GPTSum`：GPT-4o mini 生成的任务摘要
  3. `LLaMASum`：LLaMA 3.1-8B INSTRUCT 生成的摘要
  4. `GSnip + GPTSum`
  5. `GSnip + LLaMASum`

- **模型架构**：
  - **Encoder-based Models**：
    - BERT (`bert-base-uncased`)
    - RoBERTa (`roberta-base`)
    - Longformer (`allenai/longformer-base-4096`)
  - **Generative Model**：
    - GPT-4o mini (`gpt-4o-mini-2024-07-18`) —— 微调用于分类

- **微调参数统一设置**（BERT/RoBERTa/Longformer）：
  - Epochs: 3
  - Optimizer: AdamW
  - Learning Rate: 5e-5
  - Warmup Steps: 500
  - Batch Size: 8 (train), 16 (eval)

- **评估指标**：
  - **Macro-averaged Precision (P)**, **Recall (R)**, **F1-score**
  - 强调各类别均衡表现，避免被高频类主导

### 基线方法对比
- **Prompting Baseline**：
  - 使用 GPT-4o mini 进行 zero-shot 分类
  - 设置两种 prompt：
    - 无上下文：仅给实体名
    - 有上下文：加入 GSnip / GPTSum / LLaMASum 作为 context
  - 目标是判断是否可通过记忆直接预测代码（验证非 trivial lookup）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Macro-F1）

| Model + Context | SIC Code F1 | Healthcare Taxonomy F1 |
|----------------|-------------|-------------------------|
| **Best Prompting (w/ GSnip)** | 60.1% | 25.6% |
| **Best Fine-tuned (GPT-4o mini + GSnip+LLaMASum)** | **82.3%** | **72.9%** |

> 📈 性能提升显著：相比最佳 prompting 方法，分别高出 **+22.2 pts** 和 **+47.3 pts**

### 详细结果趋势（见 Table 3）
- 在所有模型上，**组合文本 > 单一文本**：
  - 例如，BERT 使用 `GSnip+GPTSum` 在 Healthcare 任务中比单独使用 `GSnip` 提升 **20.2 F1 points**
- **GPT-4o mini 表现最优**，尤其在融合 LLaMASum 时达到峰值
- Encoder-based 模型中：
  - RoBERTa 最佳：SIC 上达 76.3% F1（GSnip+GPTSum）
  - Longformer 最佳：Healthcare 上达 69.4% F1

### 消融实验结果
#### （1）Snippet 数量影响（Table 4）
| Top-k Snippets | F1 Score |
|---------------|----------|
| 1             | 68.6%    |
| 5             | 76.8%    |
| **10**        | **81.7%** ✅ |
| 15            | 77.7%    |
| 20            | 81.0%    |

> 🔍 结论：top-10 snippets 达到性能峰值，更多片段引入噪声导致下降

#### （2）置信度阈值控制精度-召回权衡（Figure 6）
- 使用 RoBERTa 和 Longformer 提取预测置信度
- 设置不同 threshold 控制输出：
  - Threshold=0.85 → Precision=**96%**, Recall≈30%
  - Threshold=0.60 → Precision=**91%**, Recall≈60%

> ✅ 支持高精度自动化标注，在牺牲部分召回率的前提下实现接近全自动的知识填充

---

## 4. 关键结论和发现

### 主要发现
1. **LLM prompting 无法直接解决该任务**：
   - GPT-4o mini 在 zero-shot 下仅得 46.4%（SIC）和 2.1%（Healthcare）F1
   - 表明 SIC/Taxonomy codes 并未被充分 memorized，不是简单查表任务

2. **动态文本获取有效弥补信息缺口**：
   - Web snippets 提供真实、聚焦的关键事实
   - LLM summaries 提供结构化、归纳性强的内容
   - 二者互补，联合使用显著优于单一来源

3. **框架具有强泛化能力**：
   - 成功应用于完全不同的两个领域（industry vs healthcare）
   - 各类模型家族（encoder/generative）均受益
   - 不同 LLM 生成器（GPT vs LLaMA）效果一致

4. **Google Snippets 为何优于 LLM Summaries？**
   - LLM 可能因实体罕见而“拒绝回答”
   - LLM 倾向于叙述宽泛背景，忽略最相关的运营细节
   - Web snippets 更聚焦于显性事实，多样性更高

### 方法局限性
- **依赖外部 API**：Google 搜索受限于 SerpAPI 接口可用性和成本
- **LLM 生成存在幻觉风险**：尽管采用 factuality prompt 缓解，仍可能生成不准确信息
- **无法处理极度稀有的实体**：若 web 完全无提及，则 GSnip 为空，影响性能
- **GPT-4o mini 不返回置信分数**：限制了高精度自动标注的应用范围

### 未来工作方向
- 探索更低成本的搜索引擎替代方案（如 Bing、Perplexity）
- 引入多轮迭代检索-生成机制以优化文本质量
- 将框架扩展至其他实体类型（如地点、产品）
- 构建闭环系统：用高置信预测自动扩充训练集，持续迭代改进模型
- 研究如何量化并过滤 LLM 生成中的潜在偏差与错误

---

> 💡 **总体评价**：本文提出的 **dynamic text acquisition framework** 为现实场景下的小众实体分类提供了高效、灵活、可扩展的解决方案，推动了自动化知识获取与领域适配型 NLP 系统的发展。发布的两个 benchmark 数据集也为后续研究提供了宝贵资源。

</details>

---

### 14. [Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought](https://arxiv.org/abs/2604.22709)

**Authors**: Keshav Ramji, Tahira Naseem, Ram\'on Fernandez Astudillo  
**Category**: cs.CL  
**Published**: 2026-04-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.22709v1  

#### Abstract
While long, explicit chains-of-thought (CoT) have proven effective on complex reasoning tasks, they are costly to generate during inference. Non-verbal reasoning methods have emerged with shorter generation lengths by leveraging continuous representations, yet their performance lags behind verbalize...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
传统的 **Chain-of-Thought (CoT)** 推理虽然能显著提升大语言模型（LLM）在复杂任务上的表现，但其依赖**长篇幅、自然语言形式的中间推理过程**，导致以下问题：
- **高延迟与高成本**：生成大量推理 token 显著增加推理时间与计算开销。
- **冗余与低效**：许多推理步骤在语义上重复或可压缩。
- **不可控性**：显式的自然语言 CoT 难以进行长度控制或优化。

尽管已有研究尝试通过连续隐空间（如 Coconut）或填充 token（如 `<pause>`）来实现更高效的内部推理，但这些方法在**表达性、可控性或性能**上仍存在不足。

---

### **提出了什么新方法或新思路**
本文提出 **Abstract Chain-of-Thought (Abstract-CoT)**，一种**离散、潜变量形式的高效推理机制**，核心思想如下：

- 在 LLM 的后训练阶段（post-training），引入一个由 $ M $ 个**预留抽象 token**（如 `<TOKEN_A>`, `<TOKEN_B>`）组成的 **abstract codebook**。
- 模型不再生成自然语言 CoT，而是生成一个**短序列的抽象 token**（如 `<beginabstract> EC AEFA BB D ... <endabstract>`），作为“潜思维草稿”（latent scratchpad）。
- 最终答案仅依赖于该抽象序列，而非原始的自然语言 CoT。

该方法的关键创新在于：
- **完全离散化**：不同于连续隐空间方法，Abstract-CoT 使用**离散 token 序列**，便于集成到现有解码流程中。
- **后训练引入**：无需重新预训练，仅通过指令微调和强化学习即可学会使用新 token。
- **双阶段训练策略**：
  1. **Policy Iteration Warm-up**：交替执行“瓶颈化监督微调”（bottlenecked SFT）与“自蒸馏”（self-distillation），逐步引导模型将抽象 token 学习为有效的信息载体。
  2. **Warm-started RL**：在 warm-up 后启动强化学习（GRPO），进一步探索高质量的抽象推理路径。

---

### **相比现有方法的优势**
| 方法 | 抽象表示 | 是否需预训练 | 推理效率 | 可控性 | 性能 |
|------|----------|--------------|-----------|--------|-------|
| **Verbal CoT** | 自然语言 | 否 | 低 | 低 | 高 |
| **Pause/Filler Tokens** | 单一 token | 否 | 中 | 中 | 一般 |
| **Coconut (连续隐空间)** | 连续向量 | 是 | 高 | 低 | 中 |
| **Abstract-CoT (本文)** | **离散 token 序列** | **否** | **极高** | **高** | **高** |

- **token 效率提升高达 11.6×**，同时保持甚至超越 verbal CoT 的性能。
- **无需修改模型架构**，仅通过 token 扩展和训练策略实现。
- **保留了部分可解释性**：抽象 token 序列虽非人类可读，但仍为结构化中间表示，可用于监控与分析。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **训练数据**：
  - `Dolci-Think-SFT`：包含 prompt、gold verbal CoT 和答案，用于 warm-up 阶段。
  - `Dolci-Think-RL`：仅包含 prompt 和 gold answer，用于 RL 阶段。
- **测试基准**：
  - **MATH-500**：数学推理（可验证）
  - **AlpacaEval-LC-2.0**：通用指令遵循（非可验证，基于赢率）
  - **HotpotQA**：多跳问答（非可验证）
  - **AIME 25**, **GPQA-Diamond**：更具挑战性的推理任务

### **实验设置与评估指标**
- **模型**：
  - Qwen3-8B, Qwen3-4B, Granite-4.0-Micro (3B)，以及 Qwen3-32B（消融实验）
- **抽象词表配置**：
  - $ M = 64 $ 个抽象 token
  - 最大抽象序列长度 $ m_{\text{max}} = 128 $
- **训练流程**：
  - **Warm-up**：3 轮 policy iteration（每轮 3 epoch SFT + 3 epoch self-distillation）
  - **RL**：1M episodes，使用 GRPO + generative reward model（gpt-oss-20b）
- **评估指标**：
  - **性能**：Accuracy (MATH), Win-rate (AlpacaEval), F1 (HotpotQA)
  - **效率**：平均生成 token 数（含推理 + 回答）

### **基线方法对比**
| 基线方法 | 描述 |
|---------|------|
| **Baseline** | 直接生成答案，无 CoT |
| **Pause Tokens** | 插入 $ m_{\text{max}} $ 个 `<pause>` token |
| **Stepwise Internalization (ICoT-SI)** | 渐进式移除 CoT 步骤 |
| **SFT (no CoT)** | 仅用 (x, y) 微调 |
| **SFT (CoT)** | 用 (x, c, y) 微调 |
| **SFT + RL** | SFT 后接 RL（标准 verbal CoT 流程） |
| **Abstract-CoT (w/o warm-up)** | 仅 RL（冷启动） |
| **Abstract-CoT (warm-up only)** | 仅 warm-up |
| **Abstract-CoT (warm-up + RL)** | 完整方法 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（Qwen3-8B）**

| Method | MATH-500 Acc | Tokens | AlpacaEval Win-rate | Tokens | HotpotQA F1 | Tokens |
|--------|---------------|--------|----------------------|--------|-------------|--------|
| **SFT + RL (Verbal CoT)** | 92.6 | 1671 | 58.4 | 496 | 58.1 | 735 |
| **Abstract-CoT (Warm-up + RL)** | **90.8** | **144** | **60.8** | **225** | **58.8** | **171** |
| **Compression Ratio** | — | **11.6×** | — | **2.2×** | — | **4.3×** |

- 在 MATH-500 上，**仅用 144 个推理 token**（vs. 1671），达到接近最优性能（90.8 vs. 92.6）。
- 在 AlpacaEval 上，**性能反超**（60.8 > 58.4），且 token 数减少 2.2×。
- 在 HotpotQA 上，F1 提升至 58.8，token 减少 4.3×。

### **与其他模型家族一致**
在 Qwen3-4B 和 Granite-4.0-Micro 上均观察到类似趋势，表明方法具有**跨模型泛化能力**。

### **更具挑战性任务的结果（Qwen3-8B）**

| Method | GPQA-Diamond Acc | Tokens | AIME'25 Acc | Tokens |
|--------|--------------------|--------|------------|--------|
| **SFT + RL** | 51.5 | 1382 | 25.6 | 9343 |
| **Abstract-CoT (w+u+RL)** | **50.5** | **174** | **24.4** | **3438** |
| **Compression** | — | **7.9×** | — | **2.7×** |

- 在 GPQA 和 AIME 上，token 数分别减少 **7.9×** 和 **2.7×**，性能几乎持平。

### **消融实验结果**
#### **(1) 冷启动 vs. Warm-up**
- **RL-only (冷启动)**：性能远低于 baseline，说明随机初始化的抽象 token 无法有效学习。
- **Warm-up only**：已能超越 SFT (no CoT)，但弱于 SFT (CoT)。
- **Warm-up + RL**：显著提升，证明 warm-up 为 RL 提供了有效起点。

#### **(2) 词表大小 $ M $ 消融**
- $ M = 64 $ 时性能最优，过小（如 $ M=2 $）限制表达力，过大则饱和甚至下降。
- 图 5–7 显示：$ M=64 $ 在多个任务上达到峰值。

#### **(3) 截断敏感性分析（Truncation Ablation）**
- 将推理长度截断至 32 token：
  - **Verbal CoT**：性能大幅下降（如 MATH-500 从 92.6 → 80.8）
  - **Abstract-CoT**：下降平缓（90.8 → 84.6），说明其设计为**短而紧凑的推理路径**，对长度更鲁棒。

#### **(4) 置换敏感性分析（Permutation Ablation）**
- 对 CoT 步骤或抽象 token 序列进行随机打乱：
  - 两种方法性能均下降，但 **Abstract-CoT 下降幅度小于 Verbal CoT**。
  - 表明抽象 token 序列具有**一定的顺序敏感性和组合性**，支持“抽象推理语言”的形成。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **抽象 token 可作为高效推理媒介**：仅用少量离散 token 即可编码复杂的推理过程。
2. ✅ **Warm-up 是成功关键**：通过 bottlenecked SFT + self-distillation，使抽象 token 从“无意义”变为“有意义”。
3. ✅ **RL 可进一步优化抽象策略**：warm-started RL 成功探索出更优的抽象推理路径。
4. ✅ **出现类自然语言的幂律分布**：抽象 token 使用频率呈现 Zipf-like 分布，暗示模型学会了**重用高频概念**，形成“抽象推理语言”。
5. ✅ **跨任务、跨模型泛化性强**：在数学、指令、多跳问答等任务及不同规模模型上均有效。

### **方法的局限性**
- **依赖初始 verbal CoT 数据**：warm-up 阶段需要高质量的黄金 CoT。
- **抽象 token 不可解释**：虽结构化，但无法直接理解其语义，不利于调试与审计。
- **冷启动困难**：若无 warm-up，RL 难以从零学习抽象推理。
- **固定词表限制**：当前方法未支持动态扩展抽象词汇。

### **未来工作方向**
- **构建可解释的抽象词典**：通过反向映射或聚类分析，赋予抽象 token 语义标签。
- **分层抽象结构**：引入 hierarchy 或 subroutine，支持更复杂的推理模式。
- **预算自适应机制**：根据问题难度动态调整抽象序列长度。
- **结合连续与离散表示**：HybridCoT 风格，平衡效率与表达力。
- **用于 RLHF 中的思维监控**：利用抽象 trace 实现对模型内部推理过程的可审计性（chain-of-thought monitorability）。

---

> **总结一句话**：  
> **Abstract-CoT 证明了 LLM 可以后训练学会一种“无声的思考语言”——它既不像自然语言那样冗长，也不像黑箱那样不可控，而是以极高的 token 效率实现了与 verbal CoT 相当甚至更优的推理能力。**

</details>

---

### 15. [GICC: A High-Performance Runtime for GPU-Initiated Communication and Coordination in Modern HPC Systems](https://arxiv.org/abs/2604.22126)

**Authors**: Baodi Shan, Mauricio Araya-Polo, Barbara Chapman  
**Category**: cs.DC  
**Published**: 2026-04-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.22126v1  

#### Abstract
Distributed GPU applications increasingly rely on kernel-level, cross-node coordination to reduce launch overheads and improve compute-communication overlap, but such support is lacking. On OFI-based interconnects such as HPE Slingshot, which powers six of the top ten systems in the November 2025 To...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：GICC: A High-Performance Runtime for GPU-Initiated Communication and Coordination in Modern HPC Systems

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题

现代高性能计算（HPC）系统中，分布式 GPU 应用广泛依赖频繁的协调操作（如 barriers、halo exchanges），但当前主流运行时存在两大瓶颈：

- **在 OFI-based 互连（如 HPE Slingshot）上**：GPU 内核无法自主驱动跨节点协调，必须通过主机（CPU）介入完成通信启动和进度管理，导致显著的 **kernel-launch 开销** 和 **compute-communication 重叠受限**。
- **在 InfiniBand 上**：虽然支持 GPU-initiated 通信，但现有实现（如 NVSHMEM）引入了不必要的 **同步与锁开销**，影响性能。

此外，有限的 NIC 资源（如计数器、deferred work queue 容量）限制了持续性的 GPU-driven 协调能力。

### 提出了什么新方法或新思路

本文提出 **GICC** —— 一种面向现代 HPC 系统的高性能、GPU 驱动的通信与协调运行时框架，其核心创新包括：

- **统一的 GPU-triggered 执行模型**：
  - 在 **InfiniBand** 上，GPU 可直接提交 NIC 操作（doorbell write）；
  - 在 **OFI/CXI（如 Slingshot）** 上，主机预配置 NIC 工作项（DWQ entries），GPU 通过更新触发计数器来“门铃”式触发执行，实现无 CPU 快路径参与的协调。

- **解耦协调语义与数据移动机制**：
  - 协调逻辑由 GPU 控制流主导，而底层 RDMA 数据传输由 NIC 异步完成。
  - 支持 **active messages** 和 **barriers** 等原生 GPU-visible 协调原语。

- **异步资源回收机制（asynchronous resource reclamation）**：
  - 引入轻量级 host monitor thread，在后台处理 `libfabric` 手动进度（manual progress）、完成队列轮询、资源退役与重装。
  - 采用 **epoch-based handoff** 和 **double-buffered stage-ahead** 设计，确保 GPU 不阻塞于资源回收，同时避免 DWQ 满或计数器溢出。

- **滑动窗口式资源复用**：
  - 对 barrier 等重复操作，仅维护两个预置槽位（slot i mod 2），实现流水线式准备与执行，将 NIC 状态占用控制为常数级别。

### 相比现有方法的优势

| 方面 | GICC 优势 |
|------|----------|
| **低延迟协调** | 消除 host round-trip，协调延迟从 ~25 μs 降至 0.11 μs（最高 **229× 降低**） |
| **更强的 compute-communication 重叠** | GPU 可在内核中直接发起 halo exchange，无需等待 kernel 结束 |
| **跨平台可移植性** | 统一接口支持 InfiniBand 与 OFI/CXI，填补 Slingshot 上 GPU-driven 协调空白 |
| **资源效率高** | 显式管理有限 NIC 资源，避免因资源耗尽导致阻塞 |

---

## 2. 核心实验方法和设置

### 实验平台

- **Tioga 平台（OFI/CXI）**：
  - GPU：AMD MI250X（每节点 8 GCD）
  - NIC：HPE Cassini（Slingshot 11）
  - 软件栈：LibFabric 2.1, Cray MPICH 9.0.1, GASNet 2025.8.0
  - 用于评估 GICC 在 OFI 上的表现

- **Maple 平台（InfiniBand）**：
  - GPU：NVIDIA GH200
  - NIC：Mellanox ConnectX-7（HDR InfiniBand）
  - 软件栈：MLNX_OFED 24.07, NVSHMEM v3.3.24
  - 用于对比 GICC 与 NVSHMEM 性能

> 注：作者尝试启用 Cray MPICH 的 kernel-triggered 模式但未成功，故以标准 GPU-aware MPI 作为 host-driven 基线。

### 基线方法对比

| 基线 | 描述 |
|------|------|
| **Cray MPICH (GPU-aware)** | 主机驱动的一致性基准，使用 `MPI_Barrier`, `MPI_Put` 等 |
| **NVSHMEM** | 当前主流 GPU-initiated 通信库，支持 InfiniBand，但在 OFI 上仍为主机中介（host-mediated） |
| **GASNet-EX** | 支持 active message 的高性能通信层，用于 AM 微基准测试对比 |

### 评估指标

- **Per-coordination latency**：单次 barrier 或协调操作平均延迟
- **End-to-end runtime / Speedup / Efficiency**
  - 弱扩展效率 $ E_p = T_1 / T_p $
  - 加速比（Speedup）
- **Communication time**：总通信耗时占比
- **Put/get latency**：点对点 RDMA 操作往返延迟
- **Active Message (AM) latency**：短消息请求/响应延迟

### 使用的应用与微基准

- **Microbenchmarks**：
  - 固定计算量 + 变化协调频率（N phases）
  - P2P put/get 延迟测试（OSU-like）
  - Active Message ping-pong 测试

- **真实应用**：
  - **2D Jacobi stencil**：弱扩展性测试
  - **Matrix Multiplication（Cannon 算法）**：强扩展性测试
  - **Minimod**：工业级地震模拟代理应用，多 kernel 分解，高频 halo exchange

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）协调延迟大幅降低
- 在 Tioga（Slingshot）上：
  - Host-driven 方案：平均每次协调延迟 **25.2 μs**
  - GICC：降至 **0.11 μs** → **降低 229×**
  - 图 6(a) 显示端到端时间几乎不随协调次数增长，而 host-driven 明显上升

#### （2）点对点通信延迟优化
- **Tioga（Slingshot）**：
  - 小消息（≤2KB）下 GICC 比 Cray MPICH **快 7–14%**
  - 大消息趋同（带宽饱和）

- **Maple（InfiniBand）**：
  - 小消息（4B–1KB）GICC 比 NVSHMEM **快 1.62–1.95×**（即延迟低 38–49%）
  - 原因：绕过 NVSHMEM 内部 proxy thread 与软件队列
  - 大消息（>8KB）趋同；4MB 时均达 ~185 μs（~21–22 GB/s）

#### （3）Active Message 性能
- **Tioga（Slingshot）**：
  - GICC AM latency：ReqReq 模式 **4.01 μs**，ReqRep 模式 **3.58 μs**
  - 优于 GASNet-EX（4.36 / 4.01 μs）

- **Maple（InfiniBand）**：
  - GICC 表现不如 GASNet-EX（11.7 vs 3.28 μs）
  - 原因：NVIDIA 官方建议 CPU 构造 WQE 更高效；但该场景忽略 kernel launch 开销，实际中若需触发 GPU 计算，GICC 更优

#### （4）应用级性能提升

| 应用 | 指标 | GICC 结果 | 对比 MPI |
|------|------|-----------|---------|
| **Jacobi Stencil**（弱扩展） | 64 GPU 效率 | **76.1%** | MPI: **60.6%**（+25% 效率提升） |
| **Matrix Multiplication** | 最大加速比 | **1.064×**（小矩阵） | 大矩阵接近或略低于 MPI（因通信稀疏） |
| **Minimod**（64 AMD MI250X） | 并行效率 | **42.0%** | MPI: **35.4%** |
| | 通信时间 | **低 52%** | MPI 通信耗时高出超 52% |

> ✅ GICC 在 **phase-heavy、高频协调** 场景中优势最明显。

---

## 4. 关键结论和发现

### 主要发现

1. **GPU-driven 协调是必要的且可行的**：
   - 在当前 Top500 中占主导地位的 OFI-based 系统（如 Slingshot）上，现有运行时不支持真正的 GPU-triggered 协调。
   - GICC 成功填补这一空白，实现了跨平台一致的 GPU-visible 协调抽象。

2. **协调开销已成为现代 GPU 工作负载的主要瓶颈**：
   - 如图 3 所示，一个 200 阶段的工作负载中，**超过 32% 时间消耗在协调上**，远高于计算本身的增长。

3. **细粒度 compute-communication overlap 显著提升效率**：
   - 允许 GPU 在边界区域计算完成后立即发起 halo exchange，无需等待整个 kernel 完成，极大改善弱扩展性。

4. **有限 NIC 资源下的可持续协调是挑战也是突破口**：
   - GICC 的 epoch-handoff + double-buffering 设计有效解决了资源循环利用问题，避免传统方案中的 blocking flush。

### 方法的局限性

1. **OFI 上仍非完全“主动发起”**：
   - GPU 不能动态创建 NIC 工作项，只能触发主机预设的操作序列。
   - 因此适用于规则/半规则通信模式，难以应对完全动态生成的协调需求。

2. **编程模型可移植性受限**：
   - 同一代码在 InfiniBand（真 device-side）与 OFI（triggered pre-staged）上行为不同，可能导致资源耗尽或背压。

3. **当前实现假设可靠传输与 fail-stop 模型**：
   - 不支持容错、动态成员变更或消息丢失恢复。

### 未来工作方向

1. **扩展至更多互连架构**：
   - 如 AWS EFA（Elastic Fabric Adapter），其 SRD 传输也支持 triggered operations，适合 GICC 的资源感知设计。

2. **构建完整的 GPU-triggered collective 库**：
   - 将 barrier、allreduce 等复杂集体操作基于相同范式实现，形成完整生态。

3. **编译器辅助的自动 lowering**：
   - 开发编译器支持，使开发者以 fabric-agnostic 方式表达协调意图，自动映射到底层（如 InfiniBand 直接提交 vs OFI 触发序列）。

4. **探索更智能的资源调度策略**：
   - 动态调整 stage-ahead 深度、预测 host re-arming 延迟，缓解背压问题。

---

> 📌 **总结一句话**：  
> **GICC 首次在 OFI-based HPC 系统上实现了可持续、低延迟、GPU-driven 的分布式协调，显著降低了通信开销并提升了应用扩展性，尤其适用于 stencil、迭代求解器等高频协调场景。**

</details>

---

### 16. [A Brain-Inspired Deep Separation Network for Single Channel Raman Spectra Unmixing](https://arxiv.org/abs/2604.22324)

**Authors**: Gaoruishu Long, Jinchao Liu, Bo Liu, Jie Liu, Xiaolin Hu  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.22324v1  

#### Abstract
Raman spectra obtained in real world applications are often a noisy combination of several spectra of various substances in a tested sample. Unmixing such spectra into individual components corresponding to each of the substances is of great value and has been a longstanding challenge in Raman spect...

---

### 17. [Introducing Background Temperature to Characterise Hidden Randomness in Large Language Models](https://arxiv.org/abs/2604.22411)

**Authors**: Alberto Messina, Stefano Scotta  
**Category**: cs.AI  
**Published**: 2026-04-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.22411v1  

#### Abstract
Even when decoding with temperature $T=0$, large language models (LLMs) can produce divergent outputs for identical inputs. Recent work by Thinking Machines Lab highlights implementation-level sources of nondeterminism, including batch-size variation, kernel non-invariance, and floating-point non-as...

---

### 18. [Rethinking Math Reasoning Evaluation: A Robust LLM-as-a-Judge Framework Beyond Symbolic Rigidity](https://arxiv.org/abs/2604.22597)

**Authors**: Erez Yosef, Oron Anschel, Shunit Haviv Hakimi, Asaf Gendler, Adam Botach, Nimrod Berman, Igor Kviatkovsky  
**Category**: cs.AI  
**Published**: 2026-04-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.22597v1  

#### Abstract
Recent advancements in large language models have led to significant improvements across various tasks, including mathematical reasoning, which is used to assess models' intelligence in logical reasoning and problem-solving. Models are evaluated on mathematical reasoning benchmarks by verifying the ...

---

### 19. [Incentivizing Neuro-symbolic Language-based Reasoning in VLMs via Reinforcement Learning](https://arxiv.org/abs/2604.22062)

**Authors**: Karthic Palaniappan  
**Category**: cs.CL  
**Published**: 2026-04-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.22062v1  

#### Abstract
There are 7,407 languages in the world. But, what about the languages that are not there in the world? Are humans so narrow minded that we don't care about the languages aliens communicate in? Aliens are humans too! In the 2016 movie Arrival, Amy Adams plays a linguist, Dr. Louise Banks who, by lear...

---

### 20. [How Large Language Models Balance Internal Knowledge with User and Document Assertions](https://arxiv.org/abs/2604.22193)

**Authors**: Shuowei Li, Haoxin Li, Wenda Chu, Yi Fang  
**Category**: cs.CL  
**Published**: 2026-04-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.22193v1  

#### Abstract
Large language models (LLMs) often need to balance their internal parametric knowledge with external information, such as user beliefs and content from retrieved documents, in real-world scenarios like RAG or chat-based systems. A model's ability to reliably process these sources is key to system sa...

---

### 21. [Bridging the Long-Tail Gap: Robust Retrieval-Augmented Relation Completion via Multi-Stage Paraphrase Infusion](https://arxiv.org/abs/2604.22261)

**Authors**: Fahmida Alam, Mihai Surdeanu, Ellen Riloff  
**Category**: cs.CL  
**Published**: 2026-04-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.22261v1  

#### Abstract
Large language models (LLMs) struggle with relation completion (RC), both with and without retrieval-augmented generation (RAG), particularly when the required information is rare or sparsely represented. To address this, we propose a novel multi-stage paraphrase-guided relation-completion framework...

---

### 22. [Learning Coverage- and Power-Optimal Transmitter Placement from Building Maps: A Comparative Study of Direct and Indirect Neural Approaches](https://arxiv.org/abs/2604.22056)

**Authors**: \c{C}a\u{g}kan Yapar  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.22056v1  

#### Abstract
Optimal wireless transmitter placement is a central task in radio-network planning, yet exhaustive search becomes prohibitively expensive at scale. This paper studies the single-transmitter setting under a fixed learned propagation surrogate, where exhaustive per-pixel evaluation remains tractable a...

---

### 23. [Insect-inspired modular architectures as inductive biases for reinforcement learning](https://arxiv.org/abs/2604.22081)

**Authors**: Anne E. Staples  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.22081v1  

#### Abstract
Most reinforcement-learning (RL) controllers used in continuous control are architecturally centralized: observations are compressed into a single latent state from which both value estimates and actions are produced. Biological control systems are often organized differently. Insects, in particular...

---

### 24. [HubRouter: A Pluggable Sub-Quadratic Routing Primitive for Hybrid Sequence Models](https://arxiv.org/abs/2604.22442)

**Authors**: Abhinaba Basu  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.22442v1  

#### Abstract
We introduce HubRouter, a pluggable module that replaces O(n^2) attention layers with O(nM) hub-mediated routing, where M << n is a small number of learned hub tokens. We demonstrate it in two from-scratch architectures: a Jamba-style hybrid and a 12-layer Transformer; retrofit into pretrained model...

---

### 25. [MolClaw: An Autonomous Agent with Hierarchical Skills for Drug Molecule Evaluation, Screening, and Optimization](https://arxiv.org/abs/2604.21937)

**Authors**: Lisheng Zhang, Lilong Wang, Xiangyu Sun, Wei Tang, Haoyang Su, Yuehui Qian, Qikui Yang, Qingsong Li, Zhenyu Tang, Haoran Sun, Yingnan Han, Yankai Jiang, Wenjie Lou, Bowen Zhou, Xiaosong Wang, Lei Bai, Zhengwei Xie  
**Category**: cs.AI  
**Published**: 2026-04-27  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2604.21937v1  

#### Abstract
Computational drug discovery, particularly the complex workflows of drug molecule screening and optimization, requires orchestrating dozens of specialized tools in multi-step workflows, yet current AI agents struggle to maintain robust performance and consistently underperform in these high-complexi...

---

### 26. [Emergent Strategic Reasoning Risks in AI: A Taxonomy-Driven Evaluation Framework](https://arxiv.org/abs/2604.22119)

**Authors**: Tharindu Kumarage, Lisa Bauer, Yao Ma, Dan Rosen, Yashasvi Raghavendra Guduri, Anna Rumshisky, Kai-Wei Chang, Aram Galstyan, Rahul Gupta, Charith Peris  
**Category**: cs.AI  
**Published**: 2026-04-27  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2604.22119v1  

#### Abstract
As reasoning capacity and deployment scope grow in tandem, large language models (LLMs) gain the capacity to engage in behaviors that serve their own objectives, a class of risks we term Emergent Strategic Reasoning Risks (ESRRs). These include, but are not limited to, deception (intentionally misle...

---

### 27. [RouteLMT: Learned Sample Routing for Hybrid LLM Translation Deployment](https://arxiv.org/abs/2604.22520)

**Authors**: Yingfeng Luo, Hongyu Liu, Dingyang Lin, Kaiyan Chang, Chenglong Wang, Bei Li, Quan Du, Tong Xiao, Jingbo Zhu  
**Category**: cs.CL  
**Published**: 2026-04-27  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2604.22520v1  

#### Abstract
Large Language Models (LLMs) have achieved remarkable performance in Machine Translation (MT), but deploying them at scale remains prohibitively expensive. A widely adopted remedy is the hybrid system paradigm, which balances cost and quality by serving most requests with a small model and selective...

---

### 28. [Accelerating Intra-Node GPU-to-GPU Communication Through Multi-Path Transfers with CUDA Graphs](https://arxiv.org/abs/2604.22228)

**Authors**: Amirhossein Sojoodi, Yiltan Hassan Temucin, Amirreza Baratisedeh, Hamed Sharifian, Ahmad Afsahi  
**Category**: cs.DC  
**Published**: 2026-04-27  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2604.22228v1  

#### Abstract
Effective intra-node GPU communication is essential for optimizing performance in MPI-based HPC applications, especially when leveraging multiple communication paths. In this study, we propose a novel approach that integrates CUDA Graphs into the UCX framework to enhance intra-node multi-path point-...

---

### 29. [How LLMs Detect and Correct Their Own Errors: The Role of Internal Confidence Signals](https://arxiv.org/abs/2604.22271)

**Authors**: Dharshan Kumaran, Viorica Patraucean, Simon Osindero, Petar Velickovic, Nathaniel Daw  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2604.22271v1  

#### Abstract
Large language models can detect their own errors and sometimes correct them without external feedback, but the underlying mechanisms remain unknown. We investigate this through the lens of second-order models of confidence from decision neuroscience. In a first-order system, confidence derives from...

---

### 30. [TabSCM: A practical Framework for Generating Realistic Tabular Data](https://arxiv.org/abs/2604.22337)

**Authors**: Sven Jacob, Bardh Prenkaj, Weijia Shao, Gjergji Kasneci  
**Category**: cs.LG  
**Published**: 2026-04-27  
**Score**: 3.5  
**Type**: new  
**ArXiv ID**: 2604.22337v1  

#### Abstract
Most tabular-data generators match marginal statistics yet ignore causal structure, leading downstream models to learn spurious or unfair patterns. We present TabSCM, a mixed-type generator that preserves those causal dependencies. Starting from a Completed Partially Directed Acyclic Graph (CPDAG) f...

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
