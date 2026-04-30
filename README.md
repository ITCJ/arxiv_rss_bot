# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-30 07:59:53 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference](https://arxiv.org/abs/2604.26294)

**Authors**: Vasu Shyam, Anna Golubeva, Quentin Anthony  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.26294v1  

#### Abstract
We present tensor and sequence parallelism (TSP), a parallel execution strategy that folds tensor parallelism and sequence parallelism onto a single device axis. In conventional multi-dimensional parallelism layouts, tensor parallelism (TP) shards model weights while sequence parallelism (SP) shards...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在大规模 Transformer 模型的训练与推理中，**显存瓶颈**是核心挑战之一。传统多维并行策略（如 TP、SP、DP）将不同的并行维度分配到设备网格的不同正交轴上，导致以下问题：
- **内存复制开销大**：Tensor Parallelism (TP) 减少参数内存但不减少激活内存；Sequence Parallelism (SP) 减少激活内存但完全复制模型参数。
- **设备利用率低**：TP 和 SP 各自占用独立的 mesh 维度，消耗大量设备用于模型并行，减少了可用于 Data Parallelism (DP) 的副本数量。
- **通信拓扑不友好**：二维 mesh（如 $T \times S$）容易跨越节点边界，被迫使用带宽较低的 inter-node 连接（如 InfiniBand），而非高带宽的 intra-node 互连（如 NVLink / Infinity Fabric）。

### 提出了什么新方法或新思路
本文提出 **Tensor and Sequence Parallelism (TSP)** ——一种**折叠式并行执行策略**，其核心思想是：
> 将 TP 和 SP 折叠到**同一个设备轴**上，每个 rank 同时持有：
> - 一份 **weight shard**（来自 TP）
> - 一份 **sequence shard**（来自 SP）

通过这种方式，在单个并行组内同时实现参数和激活的分片，从而在不增加总设备数的前提下，获得两种并行方式的内存收益。

#### 创新设计亮点：
- **Attention 层调度**：采用 **broadcast-loop + all-gather K/V** 的方式。每轮广播一个 rank 的权重分片（WQ/WK/WV/WO），各 rank 在本地序列上计算 Q/K/V，然后对 K/V 进行 all-gather 并重排序以恢复因果上下文。
- **MLP 层调度**：采用 **ring communication** 循环传递权重分片（gate/up/down projection），局部累积输出，避免标准 TP 所需的 all-reduce。
- **zigzag 分区**：用于平衡因果注意力中的负载不均问题。

### 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **内存效率** | 是唯一能在单一并行轴上同时降低 `parameter` 和 `activation` 内存的方法，降幅均为 $1/D$。 |
| **硬件适配性** | 更易映射到高带宽 intra-node 域（如 MI300X 的 Infinity Fabric），避免跨节点通信。 |
| **资源灵活性** | 节省出的设备可扩展为更多 DP 副本，提升吞吐量或支持更大 batch size。 |
| **组合兼容性** | 可与其他并行范式（PP、EP、DP）正交组合，作为灵活的并行轴插入。 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
论文未明确使用具体自然语言处理任务的数据集（如 WikiText、C4 等）。  
实验聚焦于 **系统级性能建模与实测基准测试**，基于一个合成的 **7B 参数 dense decoder-only Transformer 模型**进行前向/反向传播的端到端测量。

### 实验设置和评估指标

#### 模型配置（7B Reference Model）
- Hidden dimension $h = 4096$
- Layers $L = 32$
- Query heads $n_h = 32$, KV heads $n_{ku} = 32$ (MHA, $g=1$)
- FFN expansion factor $F = 4$ (SwiGLU)
- Precision: bf16/fp16 for forward ($\beta_p = 2$), AdamW states in fp32 ($\beta_o = 4$)

#### 硬件平台
- **Node 架构**：每节点 8× AMD MI300X GPU，通过 **Infinity Fabric** 互联（intra-node 高带宽）
- **Cluster 架构**：最多使用 128 节点（共 1024 GPUs），采用 **rails-only topology**，配备 Pollara 400Gbps NIC 用于训练通信
- 支持 selective recomputation 和 full recomputation 两种模式

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Peak Memory (VRAM)** | 单卡峰值内存占用（GB） |
| **Throughput (tokens/s)** | 前向或前后向总吞吐量 |
| **Communication Volume (bytes)** | 每层每设备通信量理论分析 |
| **FLOPs** | 每设备计算量 |
| **Scaling Efficiency** | 不同 degree 下的性能扩展性 |

### 基线方法对比
- **DP**（Data Parallelism）
- **TP**（Tensor Parallelism）
- **SP**（Sequence Parallelism）
- **TP+SP**（传统二维 mesh，如 $T=2, S=4$ 或 $T=4, S=2$）
- **TSP**（本文提出）

所有方案均在相同设备总数下比较（例如 D=8）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 内存表现（图 9）
| 序列长度 | TSP (GB) | TP (GB) | SP (GB) | TP+SP (GB) |
|---------|--------|-------|-------|----------|
| 16K     | ~31.0  | ~31.5 | ~108  | ~85–108  |
| 128K    | ~67.5  | ~119  | ~38.8 | ~54–85   |

✅ **TSP 在所有序列长度下均取得最低峰值内存**，且优势随 context 增长而扩大。

#### 吞吐量表现（图 10–12）
- 在 $D=2,4,8$ 下，TSP 的 **forward throughput** 显著优于匹配的 TP+SP 配置。
- 随着 sequence length 增加（32K → 128K），TSP 的相对优势持续增强。
- 在 micro-batch size sweep 中，由于更低内存占用，TSP 支持更大的 batch size，带来更高吞吐。

#### 通信体积分析（图 5, 7）
- TSP 的理论通信量高于 TP（因需移动权重），但在满足条件 $BS > 8h$ 时，其通信成本趋于与 TP 相当。
- 实际运行中，**weight transfer 被有效 overlap 在 GEMM 计算之后**，未成为关键路径。

### 与基线方法的对比结果
| 对比项 | 结果 |
|--------|------|
| vs TP | 在短 context 下内存相近；在长 context 下显著更优（激活内存更低） |
| vs SP | 在长 context 下内存相当甚至更优（额外减少参数内存）；训练通信略高但可控 |
| vs TP+SP | 在相同 $D$ 下达到更低内存；无需二维 mesh，更易部署于单节点内 |

### 消融实验结果（隐含在分析中）
- **通信 overlap 设计至关重要**：MLP 使用 ring + async send/recv，Attention 使用 multi-stream pipeline（weight bcast / K/V all-gather / compute 重叠），确保通信不阻塞计算。
- **broadcast vs ring 权重传输选择合理**：Attention 使用 broadcast 因其 K/V all-gather 已主导通信；MLP 使用 ring 因其计算密集适合隐藏小消息延迟。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **TSP 成功实现了 TP 与 SP 的“折叠”融合**，在单一设备轴上同时削减参数和激活内存，突破了传统二维 mesh 的资源限制。
2. ✅ **内存节省带来的实际收益远超额外通信开销**：尽管 TSP 引入了运行时 weight movement，但由于通信可被有效 overlap，**并未显著影响 wall-clock time**。
3. ✅ **TSP 具有强拓扑适应性**：尤其适用于 intra-node 高带宽架构（如 MI300X + Infinity Fabric），能完全保留在高速域内运行，避免 inter-node 降速。
4. ✅ **TSP 是一种可组合的并行原语**：可无缝集成进现有的 multi-dimensional parallelism 框架，替代部分 TP/SP 组合，释放更多设备用于 DP 或 PP。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **通信开销增加** | 尤其在小 batch 或短 context 场景下，weight movement 开销较明显。 |
| **依赖 overlap 实现性能** | 若硬件无法高效重叠通信与计算（如低带宽 fabric 或 poor streaming support），性能可能下降。 |
| **当前仅验证于 dense model** | 虽提及可用于 MoE，但实验集中在 dense decoder-only 架构。 |
| **实现复杂度较高** | 需定制 kernel 调度逻辑（multi-phase loops, reorder, async P2P） |

### 未来工作方向
1. **扩展至 MoE 模型**：结合 Expert Parallelism (EP)，探索 TSP + EP 的折叠策略。
2. **自动并行策略搜索**：将 TSP 作为一个候选轴，构建自动化 parallelism mapper，根据模型结构与硬件 topology 动态选择最优组合。
3. **进一步优化通信协议**：研究 hybrid broadcast/ring 或 hierarchical TSP 以适应更大规模集群。
4. **支持动态 sequence length**：适配 variable-length batching 场景下的负载均衡与通信调度。

---

> **总结一句话**：  
> TSP 通过将 TP 与 SP 折叠到同一设备轴，在几乎不牺牲吞吐的情况下大幅降低显存占用，是一种面向 **long-context** 与 **memory-constrained** 场景的高效、实用且硬件友好的新型并行范式。

</details>

---

### 2. [When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?](https://arxiv.org/abs/2604.26412)

**Authors**: Tianyu Liu, Yuhao Shen, Xinyi Hu, Baolin Zhang, Hengxin Zhang, Jun Dai, Jun Zhang, Shuang Ge, Lei Chen, Yue Li, MingCheng Wan  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.26412v1  

#### Abstract
Speculative decoding accelerates LLM inference, but SOTA hidden-state-based drafters suffer from long-range decay: draft accuracy degrades as the speculative step increases. Existing work attributes this decay to train-inference mismatch and proposes test-time training (TTT) as a remedy, yet we obse...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该论文聚焦于**长程推测解码（long-range speculative decoding）中的“长程衰减”（long-range decay）问题**。当前主流的基于 **hidden-state reuse** 的 drafters（如 EAGLE-3 和 MTP）在推测步数 $k$ 增加时，draft 准确率显著下降。虽然已有研究将此归因于训练-推理不匹配（train-inference mismatch），并提出通过 **autoregressive Test-Time Training (TTT)** 缓解，但作者发现即使在 TTT 下，长程衰减依然存在。

### 🧠 提出的新思路与方法
作者从**上下文信息保留（context information preservation）** 的角度重新审视该问题，并提出以下核心观点与方法：

#### **KV-Reuse Hypothesis（KV重用假设）**
- **Hidden State 是一种有偏压缩（biased compression）**：目标模型的 hidden state 是通过 attention 聚合得到的，其聚合权重由当前查询 $q_t$ 决定，仅优化用于预测下一个 token。这会导致对后续预测重要的历史信息被弱化甚至丢弃。
- **KV Cache 是显式的完整上下文记忆**：KV cache 保存了每个位置的 key/value 对，在未经过 attention 聚合前是完整的。如果 draft model 可以直接访问这些 KV 对，它可以通过自己的查询进行 **re-attention**，从而恢复那些被目标模型压缩掉的信息。

> 因此，作者提出：**允许 draft model 复用目标模型的 KV cache，可以为长距离推测提供更丰富的条件信号。**

#### **KVShot：诊断框架**
为验证上述假设，作者构建了一个统一的诊断框架 **KVShot**，系统比较三种信息复用范式：
1. **Hidden-only Reuse**：标准 EAGLE-3/MTP 方式，复用 hidden states。
2. **KV-only Reuse**：完全依赖 cross-attention 到目标模型的 KV cache。
3. **Hybrid Reuse**：结合两者，hidden state 作为主干，KV cache 通过一个**门控增量规则（gated delta rule）** 提供修正项。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（EAGLE-3等） | 本文方法（KVShot） |
|------|------------------------|--------------------|
| **信息保留能力** | 弱，hidden state 已丢失部分历史信息 | 强，KV cache 保留完整 token 级表示 |
| **长程预测潜力** | 随 $k$ 增大迅速衰减 | 更平缓地衰减（支持 Prediction 1） |
| **理论视角创新** | 关注 train-inference mismatch | 引入 **信息压缩 vs. 显式记忆** 的新分析框架 |
| **设计灵活性** | 单一路径 | 支持 hybrid 架构探索 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **初步消融实验**：使用 `ShareGPT` 数据集（约 70k 样本，训练 3 轮）。
- **端到端评估**：扩展至 **280k 样本**（ShareGPT + UltraChat），且响应由目标模型 **Qwen3-8B 自身再生**，确保训练分布与推理一致。

### ⚙️ 实验设置
- **目标模型**：`Qwen3-8B`
- **训练方式**：所有 drafters 均采用与 EAGLE-3 一致的 **autoregressive TTT（Test-Time Training）** 目标进行训练。
- **采样策略**：KV cache 默认来自目标模型中均匀间隔的 3 层（$L_s=3$）。
- **架构细节**：
  - 所有 drafters 为单层或浅层 Transformer。
  - KV-only 使用 cross-attention 注入；hybrid 使用 self-attention（hidden path）+ cross-attention（KV path）+ gated fusion。

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| $\alpha_k$ | 第 $k$ 步的 draft 接受率（acceptance rate） |
| **MAT**（Mean Accepted Tokens） | 期望接受 token 数，衡量整体效率 |
| **Retention Ratio ($\alpha_6 / \alpha_0$)** | 衡量长程衰减程度，越高说明衰减越慢 |
| **HF-measured MAT** | 使用 HuggingFace 推测解码流水线测量的 MAT，含树形验证开销，更接近真实性能 |

### 🆚 基线方法
- **主要基线**：`1-layer EAGLE-3` drafter（hidden-only reuse）
- 其他对比：
  - 不同注入方式的 KV-only drafter（head concat vs. linear projection）
  - 不同深度的 KV-only drafter（1~4 层）
  - Hybrid drafter（warm-started vs. random init）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 方法 | $\alpha_0$ | $\alpha_3$ | $\alpha_6$ | Retention | MAT (step-wise) | HF MAT |
|------|------------|------------|------------|-----------|------------------|--------|
| **EAGLE-3 (baseline)** | 0.638 | 0.511 | 0.469 | 73.5% | 2.37 | 5.01 |
| **KV-only (1-layer)** | 0.494 | 0.381 | 0.353 | 71.5% | 1.84 | — |
| **KV-only (4-layer)** | 0.614 | 0.527 | 0.495 | 80.6% | 2.34 | — |
| **Hybrid (ckpt-init)** | 0.665 | 0.553 | 0.514 | 77.3% | **2.54** | **5.04** |

> ✅ **Step-wise MAT 提升 +0.17**（2.37 → 2.54），表明 KV 修正有效提升每步接受率。

> ❌ **HF-measured MAT 仅提升 +0.03**（5.01 → 5.04），且额外引入 **5–10% drafting latency**，导致端到端加速几乎不可见。

---

### 🔬 消融实验结果

#### （1）KV-only Reuse 消融（Table 1）
- 移除目标信息后 MAT 降至 1.31，证明 **target KV 提供有效信号**。
- **Linear projection > Head concat**：说明强制单查询跨多层 KV 空间难度高。
- RoPE 修复影响微弱，说明位置编码不是瓶颈。

#### （2）深度扩展实验（Table 2）
- 从 1 层 → 2 层 KV-only drafter，MAT **跃升 +0.39**（1.84 → 2.23），验证 **query estimation 是关键瓶颈（Prediction 2）**。
- 更深层数收益递减，4 层仍略低于 EAGLE-3。

#### （3）Hybrid Reuse 消融（Table 3）
- **Warm-starting from EAGLE-3 checkpoint 至关重要**：MAT 从 2.44（随机初始化）提升至 **2.54**。
- **Self-attention anchor 不可替代**：“Cross-only” 变体 MAT 下降至 ~2.35，说明 **KV 应作为补充而非替代**。
- **Retention 提升明显**：73.5% → 77.3%，支持 **KV 在长程更具优势（Prediction 1）**。

#### （4）端到端评估（Table 4）
- 尽管 step-wise 提升显著，但 **HF MAT 增益极小（+0.6%）**。
- 主要原因：
  - EAGLE-3 基线本身在更大数据下已大幅提升（4.43 → 5.01）；
  - Tree verification 压缩了不同 drafters 的差异；
  - 新增 cross-attention 带来计算开销。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Long-range decay 不仅源于 train-inference mismatch，也与 representation choice 密切相关**：
   - Hidden state 是 query-dependent 的压缩摘要，会丢弃对未来预测有用的历史信息。
   - KV cache 保留原始 token 级 KV 对，允许 draft model 进行 re-attention，理论上更适合长程推测。

2. **KV-Reuse Hypothesis 得到验证**（支持三项预测）：
   - **Prediction 1（长程优势）**：KV-only 和 hybrid 在 $k \geq 3$ 时衰减更慢，Retention 更高。
   - **Prediction 2（query estimation 瓶颈）**：增加 drafter 深度显著提升 KV-only 性能，说明 query 估计能力是制约因素。
   - **Prediction 3（短程劣势）**：KV-only 在 $k=0$ 时表现弱于 hidden-only，因其缺乏 rich semantic embedding。

3. **Hybrid 设计最优**：结合 hidden state（强短程 anchor）与 KV cache（长程修正），可在 step-wise 层面实现最佳性能。

---

### ⚠️ 方法的局限性
尽管 KV reuse 在理论上具有优势，但在当前 **autoregressive TTT pipeline** 下仍面临三大结构性瓶颈：

#### （1）Query Estimation 难度高
- 浅层 drafter 难以逼近深层目标模型生成的复杂 query。
- 即使 4 层 KV-only 也无法超越 1 层 EAGLE-3，说明 capacity 不足。

#### （2）Sparse Optimization of Draft-side KV Projections
- 训练中绝大多数 KV 来自目标模型 copy，只有少量 draft token 参与更新。
- 导致 $W_{\text{kv}}^{\text{draft}}$ 参数接收到稀疏梯度，难以充分学习。

#### （3）Gate-induced Gradient Starvation
- 在 warm-started hybrid 模型中，cross-attention 分支初始输出为噪声，gate 快速关闭（$g \to 0$）。
- 一旦关闭，cross-attention 分支梯度被抑制，陷入“低注入-难学习”的恶性循环。
- 实验显示平均 gate 值长期维持在 ~0.1，限制了 KV 通道的利用效率。

---

### 🔮 未来工作方向
作者指出，要真正释放 KV-aware decoding 的潜力，需跳出当前 **autoregressive TTT** 范式，转向更适合的训练架构：

#### ✅ 推荐方向：**Block-wise Training**
- 如 **DFlash (Chen et al., 2026)** 所示，block diffusion adapter 可并行生成多个 draft token。
- 优点：
  - 提供更深的 query 计算路径；
  - 每步生成更多 draft token，缓解 KV 投影的稀疏梯度问题；
  - 更适合 hybrid 架构的稳定训练。

#### 🧪 建议后续研究
- 设计专为 KV reuse 优化的非自回归训练目标；
- 探索更鲁棒的 fusion 机制（如动态门控、课程学习）；
- 在更长序列和复杂任务上验证 KV reuse 的泛化能力。

---

## 总结

| 维度 | 结论 |
|------|------|
| **核心思想** | 提出 **KV-Reuse Hypothesis**，认为 KV cache 比 hidden state 更利于长程推测解码 |
| **方法创新** | 构建 **KVShot 框架**，系统比较 hidden-only、KV-only、hybrid 三种范式 |
| **实验证明** | KV reuse 在 step-wise 层面确实改善 long-range acceptance（MAT +0.17） |
| **现实挑战** | 当前 autoregressive TTT 存在 **query estimation、sparse gradient、gate starvation** 三重瓶颈 |
| **最终判断** | **KV cache 确实携带有用信号，但现有训练 pipeline 无法高效利用它** |
| **未来出路** | 必须转向 **block-wise 或非自回归训练范式** 才能充分发挥 KV-aware decoding 的潜力 |

> 💡 **一句话总结**：  
> *“KV cache 能救长程推测，但现在的训练方式拖了后腿。”*

</details>

---

### 3. [Adaptive and Fine-grained Module-wise Expert Pruning for Efficient LoRA-MoE Fine-Tuning](https://arxiv.org/abs/2604.26340)

**Authors**: Weihang Li, Jianchun Liu, Hongli Xu  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.26340v1  

#### Abstract
LoRA-MoE has emerged as an effective paradigm for parameter-efficient fine-tuning, combining the low training cost of LoRA with the increased adaptation capacity of Mixture-of-Experts (MoE). However, existing LoRA-MoE frameworks typically adopt a fixed and uniform expert configuration across heterog...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Adaptive and Fine-grained Module-wise Expert Pruning for Efficient LoRA-MoE Fine-Tuning*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的 **LoRA-MoE** 框架在进行参数高效微调时存在两个关键缺陷：
- **粗粒度专家分配**：采用固定且统一的专家数量（如每模块8个专家），忽略了不同 Transformer 模块（如 attention 中的 `q_proj` 和 MLP 中的 `gate_proj`）在功能和容量需求上的异质性，导致部分模块过度配置、冗余严重。
- **持续负载均衡约束**：在整个训练过程中强制执行 `load-balancing loss`（Laux），虽然初期有助于防止路由崩溃，但在后期会抑制专家向特定任务的深度专业化。

### 提出的新方法：**DMEP**（Dynamic Module-wise Expert Pruning）
DMEP 是一种动态、细粒度的 LoRA-MoE 微调框架，包含三个阶段：
1. **Phase I: Load-Aware Dense Initialization**  
   初始阶段采用均匀密集结构，并引入轻量级在线追踪器记录每个 token 的硬路由决策（Top-k），积累各专家的实际利用率。
2. **Phase II: Fine-Grained Module-Wise Pruning**  
   基于模块级利用率分数（如 Gini 系数、Routing Entropy）对低效专家进行物理剪枝，仅保留高利用率专家，并同步清除其对应的 **optimizer states**（如 AdamW 的动量和方差张量）。
3. **Phase III: Relaxed Specialization**  
   剪枝后关闭 `load-balancing loss`，释放优化冲突，允许剩余专家专注于下游任务，实现更强的任务特异性专业化。

### 相比现有方法的优势
| 维度 | 传统 LoRA-MoE / 对称 MoE | DMEP |
|------|--------------------------|------|
| **专家分配粒度** | 层级或全模型统一 | **模块级自适应**（module-wise） |
| **架构灵活性** | 静态预设 | 动态在线学习、任务感知 |
| **剪枝方式** | 零掩码（pseudo-sparsity） | **物理切片**（structural sparsity），真正减少计算与内存开销 |
| **优化目标演化** | 全程使用 Laux | 后期禁用 Laux，促进专业化 |
| **效率提升来源** | 单一稀疏性 | 双重增益：**结构压缩 + 优化自由化** |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **ScienceQA**：多模态科学问答数据集，本文仅使用文本推理部分，转为多选题格式。
- **OpenBookQA**：常识与基础科学知识问答，处理为标准四选一任务。
- **GSM8K**：小学数学应用题，直接预测最终数值答案（通过 `####` 分隔符监督）。

### 实验设置
- **模型**：
  - `Qwen3-0.6B`：用于消融研究与详细分析。
  - `Qwen3-8B`：验证大规模下的有效性。
- **适配器配置**：
  - LoRA Rank $ r = 8 $，缩放因子 $ \alpha = 16 $
  - 应用于所有线性层：`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`（共7个模块/层）
  - Top-2 路由策略（$ k=2 $）
- **训练细节**：
  - 总共训练 5 轮（epochs）
  - 第一轮为探索阶段（$ E_w = 1 $），启用 Laux
  - 使用 AdamW，峰值学习率 $ 3\times10^{-4} $，cosine 退火
  - 混合精度训练（bfloat16），有效 batch size 固定为 128
- **硬件环境**：单张 NVIDIA RTX A6000（48GB VRAM）

### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 测试集准确率 |
| **Trainable Parameters (M)** | 可训练参数总量（剪枝前后对比） |
| **Training Throughput (tok/s)** | 每秒处理 token 数量，衡量训练速度 |
| **Gini Coefficient** | 衡量专家间负载不均程度（越高表示越集中） |
| **Routing Entropy** | 衡量路由分布平滑性（越低表示越专注） |
| **Routing Drift** | 连续步骤间的路由分布变化（判断稳定性） |

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Dense LoRA** | 标准 LoRA，无 MoE 结构（$ N=1 $） |
| **Symmetric MoE (N=8)** | 统一配置8个专家，全程启用 Laux |
| **DMEP (T=0.10)** | 本文提出的方法，默认阈值 $ T=0.10 $，$ K_{\min}=2 $ |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table I）

| Model | Dataset | Method | Params (M) | Throughput (tok/s) | Accuracy (%) |
|-------|--------|--------|------------|--------------------|--------------|
| Qwen3-0.6B | ScienceQA | Symmetric MoE (N=8) | 42.7 | 1368 | 91.32 |
| Qwen3-0.6B | ScienceQA | **DMEP (Ours)** | **24.2** | **1466** | **91.55** |
| Qwen3-0.6B | GSM8K | Symmetric MoE (N=8) | 42.7 | 1354 | 17.00 |
| Qwen3-0.6B | GSM8K | **DMEP (Ours)** | **25.3** | **1502** | **19.00** ✅↑ |
| Qwen3-8B | ScienceQA | Symmetric MoE (N=8) | 185.2 | 294 | 97.08 |
| Qwen3-8B | ScienceQA | **DMEP (Ours)** | **121.8** | **320** | 96.40 |
| Qwen3-8B | OpenBookQA | Symmetric MoE (N=8) | 185.2 | 196 | 95.00 |
| Qwen3-8B | OpenBookQA | **DMEP (Ours)** | **107.9** | **216** | **95.00** ✅持平 |
| Qwen3-8B | GSM8K | Symmetric MoE (N=8) | 185.2 | 288 | 43.00 |
| Qwen3-8B | GSM8K | **DMEP (Ours)** | **110.4** | **317** | **43.00** ✅持平 |

> ✅ **总体表现**：
> - **参数减少 35%–43%**
> - **训练吞吐提升约 10%**
> - 在多数任务上保持甚至超越对称 MoE 的准确率

### 与基线方法对比结果
- **vs Symmetric MoE (N=8)**：
  - 显著降低参数量（平均 ↓40%），同时维持或略微提升 accuracy。
  - 吞吐显著提高（↑10%+），源于更少的专家参与前向/反向传播及优化更新。
- **vs Dense LoRA**：
  - 参数虽高于 Dense LoRA（~5M），但 accuracy 提升巨大（如 ScienceQA 从 89.03 → 91.55）。
  - 实现了“以少量额外参数换取显著性能增益”的高效平衡。

### 消融实验结果（Table II，基于 Qwen3-0.6B + ScienceQA）

| 配置 | Params | Throughput | Accuracy |
|------|--------|-----------|---------|
| Symmetric MoE (N=8) | 42.7M | 1368 tok/s | 91.32% |
| Symmetric MoE (N=4) | 21.3M | 1439 tok/s | 90.29% |
| **DMEP (T=0.10)** | **31.2M** | **1438 tok/s** | **91.55%** ✅最优 |
| DMEP (T=0.15) | 13.1M | 1508 tok/s | 90.65% |
| Early Pruning (Step 50) | 26.7M | 1457 tok/s | 91.23% |
| **Optimal Pruning (Step 100)** | **31.2M** | **1438 tok/s** | **91.55%** ✅最佳时机 |
| Late Pruning (Step 250) | 35.3M | 1275 tok/s | 91.23% |

> 🔍 **关键发现**：
> - **动态剪枝优于静态对称缩减**：即使 DMEP(T=0.15) 参数更少（13.1M vs 21.3M），仍取得更高 accuracy（90.65% vs 90.29%），说明动态保留更有价值。
> - **剪枝阈值 T 控制权衡**：T 越大，压缩越强，但 accuracy 下降；T=0.10 是最佳折中点。
> - **剪枝时机至关重要**：第100步（即第一轮结束）是路由稳定点，此时剪枝效果最好。

---

## 4. 关键结论和发现

### 主要发现
1. **模块级异质性显著存在**：同一层内不同模块（如 `o_proj` vs `gate_proj`）的专家利用率差异极大（见 Fig. 2），统一配置必然造成资源浪费。
2. **路由模式早期即可收敛**：实验显示在第一个 epoch 结束时（Step 100），routing drift 已接近零，表明可安全进入剪枝阶段。
3. **后期负载均衡有害无益**：一旦路由稳定，继续施加 Laux 会人为拉高 entropy，阻碍专家专业化；关闭后反而有助于 accuracy 提升。
4. **物理剪枝带来真实加速**：相比 mask-based 方法，DMEP 的结构性移除真正减少了 optimizer state 更新成本，实现端到端提速。

### 方法的局限性
- **依赖初始探索阶段质量**：若探索不足或数据偏差大，可能导致错误剪枝。
- **一次性剪枝不可逆**：当前为 one-shot pruning，无法恢复被误删的专家。
- **超参数敏感性**：阈值 $ T $ 和 $ K_{\min} $ 需合理设置，否则可能欠拟合或过压缩。
- **小模型收益更大**：在 Qwen3-0.6B 上有 accuracy 提升，在大模型上主要是效率优化。

### 未来工作方向
- 引入 **迭代式渐进剪枝**（iterative pruning），逐步调整专家结构。
- 探索 **自适应阈值机制**，根据模块类型自动设定 $ T $。
- 扩展至 **多任务学习场景**，支持任务感知的专家共享与隔离。
- 将 DMEP 与其他 PEFT 方法结合（如 IA³、BitFit），构建更通用的高效微调体系。

---

> 📌 **总结一句话**：  
> **DMEP 通过“在线感知 → 模块级物理剪枝 → 放松正则”三阶段设计，在几乎不损失性能的前提下，将 LoRA-MoE 的参数量减少近 40%，训练速度提升 10%，实现了真正的高效与智能微调。**

</details>

---

### 4. [Efficient, VRAM-Constrained xLM Inference on Clients](https://arxiv.org/abs/2604.26334)

**Authors**: Aditya Ukarande, Deep Shekhar, Marc Blackstein, Ram Rangan  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.26334v1  

#### Abstract
To usher in the next round of client AI innovation, there is an urgent need to enable efficient, lossless inference of high-accuracy large language models (LLMs) and vision language models (VLMs), jointly referred to as xLMs, on client systems. To address this, we present pipelined sharding, a novel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient, VRAM-Constrained xLM Inference on Clients**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
本文旨在解决在**客户端系统**（如桌面PC、笔记本电脑）上高效运行高精度大模型（xLMs，包括LLMs和VLMs）时面临的**显存（VRAM）受限**问题。具体挑战包括：
- 支持交互式和批处理模式下的推理；
- 高分辨率视觉语言模型（VLM）推理；
- 密集型（dense）和混合专家（MoE）LLMs的统一支持；
- 动态适应系统条件（CPU线程数、PCIe带宽、VRAM预算）和推理阶段（prefill/decode）。

尽管已有CPU-GPU混合调度技术，但尚无单一方案能同时满足上述所有需求。

---

### **提出了什么新方法或新思路**
提出了一种名为 **Pipelined Sharding** 的新型、基于基准分析（benchmark-profile-guided）的CPU-GPU混合调度技术，其核心思想包括：

#### **Pipelined Sharding 的关键技术**
1. **子层级模型分片（Sub-layer Level Sharding）**  
   将Transformer层在Attention和FFN之间切分，实现更细粒度的控制。
   
2. **优先级张量放置（Prioritized Tensor Placement）**  
   按照重要性顺序将关键组件（如Attention、KV Cache）优先保留在VRAM中，其余部分可动态卸载到sysRAM。

3. **流水线拷贝-计算重叠（Pipelined Copy-Compute）**  
   在数据传输（如权重从CPU加载到GPU）的同时执行计算任务，隐藏PCIe延迟。

4. **Token Tier自适应调度机制**  
   根据当前批次的新token数量（new token count），选择最优的调度策略（GPU-only / Static / Dynamic），自动适应不同推理阶段和批量大小。

#### **针对VLM的优化：VLMOpt**
结合Pipelined Sharding，为VLM设计了三项工程优化：
- **Vision Tensor CPU Offloading**：将视觉编码器权重卸载至CPU内存；
- **Flash Attention for Vision Encoder**：在视觉编码器中启用FlashAttention以减少O(N²)注意力开销；
- **Vision-Language VRAM Overlap Avoidance**：串行化视觉与语言处理流程，避免两者峰值VRAM占用叠加。

---

### **相比现有方法的优势**
| 特性 | Pipelined Sharding | 其他方法（如TwinPilots、HeteGen等） |
|------|---------------------|-------------------------------|
| 支持TTFT优化 | ✅ | ❌ 多数不关注首token延迟 |
| 支持MoE模型 | ✅ | ❌ 或有限支持 |
| 自动适应batch size/context size | ✅ | ❌ 多为静态配置 |
| 支持VLM | ✅（+VLMOpt） | ❌ |
| 不损失精度（lossless） | ✅ | ⚠️ 多数量化/剪枝方法有损 |

此外，该方法完全自动化，无需手动调参（如`-ngl`参数），显著降低部署门槛。

---

## **2. 核心实验方法和设置**

### **使用的模型**
共测试六类模型，涵盖dense与MoE架构：
| 类型 | 模型名称 | 参数规模 | 存储大小 |
|------|--------|----------|---------|
| Dense LLM | `nemo4b`, `nemo8b` | 4B, 8B | 7.7GB, 15.7GB |
| MoE LLM | `qwen30b`, `qwen235b` | 30B, 235B | 16.4GB, 77.0GB |
| VLM | `vnemo4b`, `cr1` | —— | 8.4GB, 15.4GB |

其中CR1是NVIDIA Cosmos-Reason1 VLM，用于物理AI场景。

---

### **实验平台**
使用三类客户端设备进行评测：
| 设备 | GPU | VRAM | CPU | 内存带宽 | PCIe带宽 |
|------|-----|------|-----|-----------|------------|
| cli1 | RTX 3500 | 12GB | Ultra7 (16核) | 119.5 GB/s | 13 GB/s (Gen3) |
| cli2 | RTX 5070 Ti | 16GB | Ryzen7 (8核) | 57.6 GB/s | 50 GB/s (Gen5) |
| cli3 | RTX 5090 | 32GB | EPYC (16核) | 153.6 GB/s | 50 GB/s (Gen5) |

所有系统运行Windows 11。

---

### **评估指标**
- **TTFT**（Time-to-First-Token）：首token生成时间
- **TPS**（Tokens Per Second）：每秒输出token数
- **E2EL**（End-to-End Latency）：总端到端延迟 = TTFT + 100 / TPS
- **VRAM Usage**：峰值显存消耗
- **Speedup**：相对于baseline的加速比

---

### **基线方法对比**
- **LLM基线**：`llama.cpp-baseline`，通过手动调整`-ngl`参数找到最大可容纳层数，属于“aggressive baseline”。
- **VLM基线**：对CR1使用`vLLM`（仅限高VRAM）；低VRAM下仍用`llama.cpp-baseline`。
- 所有对比均确保公平（相同VRAM预算、量化格式等）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **LLM 性能提升（在cli3上）**
| 模型 | VRAM | Context | TPS | TTFT Speedup | TPS Speedup |
|------|------|--------|-----|---------------|--------------|
| qwen235b | 2G | 1K | 7.7 TPS | —— | —— |
| qwen235b | 2G | 16K | 5.2 TPS | 达到交互阈值（>5 TPS） | —— |
| 平均 | —— | —— | —— | **2× TTFT加速（最高6.7×）** | **3.7× TPS加速（最高30×）** |
| 最大 | —— | 64K | —— | —— | **TPS up to 30×** |

> 即使在仅2G VRAM下，77GB的qwen235b也能实现**>5 TPS**，满足人类阅读速度要求（约4–5 TPS）。

#### **批处理模式性能**
| 模型 | Batch Size | Context | TPS Speedup |
|------|------------|--------|-------------|
| qwen30b | 16 | 4K | **8.2×**（unified KV） |
| 平均 | —— | —— | **2.3× batch-wide TPS提升** |

表明Pipelined Sharding在batched场景下同样有效。

---

#### **VLM 性能表现（CR1）**
| 输入分辨率 | 基线所需VRAM | Pipelined Sharding + VLMOpt | VRAM Reduction |
|-----------|--------------|------------------------------|----------------|
| 1440p | ≥20GB | 可降至 **2GB** | **10× 显存压缩** |

> 成功实现了原本无法运行的高分辨率图像推理任务。

| 模型 | VRAM Budget | E2EL Speedup |
|------|-------------|---------------|
| vnemo4b | 6G | **1.78×** |
| cr1 | 多种配置 | **Baseline OOM，本方法成功运行** |

---

### **消融实验与敏感性分析**

#### **调度策略选择有效性（Oracle Comparison）**
- 在105个配置中，调度器**100%选中最优策略**（GPU-only / Static / Dynamic）；
- 不同条件下最优策略分布：
  - **Static**：72%（CPU资源充足）
  - **Dynamic**：18%
  - **GPU-only**：10%（CPU线程少）

验证了自适应调度的必要性。

#### **PCIe带宽影响**
- PCIe从Gen3 → Gen5，TTFT加速比从**1.2× 提升至 2.4×**；
- 表明更高的PCIe带宽进一步释放Pipelined Sharding潜力。

#### **CPU线程利用率**
- qwen30b在8G VRAM、16K context下：
  - 1线程：+10.3 TPS
  - 16线程：+24.0 TPS
- 显示框架能有效利用多核CPU资源。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Pipelined Sharding 能在极低VRAM预算下实现高效、无损推理**，支持dense/MoE/interactive/batched/VLM等多种场景。
2. ✅ **Token Tier机制是实现跨模式自适应的关键**，优于固定策略或人工调参。
3. ✅ **VLMOpt使高分辨率VLM推理成为可能**，CR1显存需求下降10倍。
4. ✅ **自动调度优于手动offloading**：在MoE FFN或KV缓存手动卸载的情况下，Pipelined Sharding仍能取得更高TPS和更低TTFT。
5. ✅ **性能随硬件升级而扩展**：更强的PCIe带宽和更多CPU线程带来更大收益，适用于未来客户端设备。

---

### **方法的局限性**
- 当前实现依赖于**llama.cpp**框架，虽已开源但仍需上游合并；
- **未考虑热节流（thermal throttling）** 对性能的影响；
- **暂未集成NPU或其他异构单元**，未来可扩展至更多硬件后端；
- 多进程并发调度仍需外部协调（如NVIDIA AMP），尚未内置QoS保障。

---

### **未来工作方向**
1. **支持视频输入**：当前VLM仅支持静态图像；
2. **自动检测驱动/固件更新并触发重新profiling**；
3. **扩展至移动端/NPU设备**；
4. **与游戏引擎深度集成**（via CiG / IGI SDK plugin）；
5. **构建统一的client AI co-scheduler**，实现LLM、图形、音频等多任务资源公平调度。

---

> **总结一句话**：  
> **Pipelined Sharding + VLMOpt 实现了真正意义上的“按需分配”的客户端大模型推理——无论你有多少VRAM，都能跑得动、跑得快、不丢精度。**

</details>

---

### 5. [COPUS: Co-adaptive Parallelism and Batch Size Selection in Large Language Model Training](https://arxiv.org/abs/2604.26687)

**Authors**: Akhmed Sakip, Erland Hilman Fuadi, Omar Sayedelahl, Zonghang Li, Jianshu She, Alham Fikri Aji, Steve Liu, Eric Xing, Qirong Ho  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.26687v1  

#### Abstract
Training large language models requires jointly configuring two interdependent aspects of the system: the global batch size, which governs statistical efficiency, and the 3D parallelism strategy, which governs hardware throughput. Existing approaches make these decisions independently: optimization ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# COPUS: Co-adaptive Parallelism and Batch Size Selection in Large Language Model Training

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在大规模语言模型（LLM）训练中，**全局批大小（global batch size, Bg）** 和 **3D并行策略（3D parallelism strategy, S）** 是两个相互依赖的关键配置参数。然而，现有方法通常将这两个决策孤立处理：
- **优化领域**的方法（如基于Gradient Noise Scale, GNS）仅自适应调整批大小以跟踪演化的临界批大小（critical batch size），但固定并行策略。
- **系统领域**的方法选择给定批大小下最快的并行策略，但不考虑批大小会随训练动态变化。

这种解耦导致次优：当批大小增长时，最优的并行策略可能已改变，而固定策略无法跟随这一变化，从而浪费硬件吞吐量。

### 提出了什么新方法或新思路
本文提出了 **Copus**，这是首个在单次LLM训练过程中**联合自适应地调优全局批大小（Bg）、微批大小（Bm）和3D并行策略（S）** 的系统。

其核心创新在于引入并最大化 **Goodput** 这一综合指标：
$$ \text{Goodput}(S, B_g, B_m, H) = T(S, B_g, B_m, H) \times SE(B_g) $$
其中：
- $T$ 是系统吞吐量（samples/sec）
- $SE$ 是统计效率（convergence per sample）

对于使用Adam优化器的场景，作者进一步提出了 **LR-aware Goodput**，显式地将学习率缩放规则（$\eta \propto \sqrt{B_g}$）纳入目标函数，使其更符合实际训练配方。

### 相比现有方法的优势
1. **联合优化**：首次将批大小和并行策略的决策统一到一个框架下，通过最大化Goodput实现协同进化。
2. **更鲁棒的决策**：相比直接用GNS估计值设定批大小（CBS-only），Goodput因同时考虑了吞吐量约束，对GNS校准因子的不确定性更不敏感。
3. **高效的在线重配置**：支持在训练过程中进行**在线状态重分片（online state resharding）**，无需全量检查点重启即可更改并行策略，显著降低了重配置开销（比checkpoint-restart快2-16倍）。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **WikiText-103** 数据集，序列长度为2048 tokens。
- 所有模型均使用其预训练的tokenizer。

### 实验设置和评估指标
- **硬件平台**：
  - **NVIDIA集群**：1-4个节点，每节点8块H100 GPU，通过NVLink和SR-IOV连接。
  - **AMD集群**：4个节点，每节点8块MI210 GPU，通过Infinity Fabric和Ethernet连接。
- **模型规模**：从3B到32B参数的Transformer模型（LLaMA-3.2-3B, LLaMA-2-13B, Qwen-2.5-32B, LLaMA-2-7B）。
- **评估指标**：
  - **主要指标**：达到特定损失值所需的**时间（Time-to-convergence）**。
  - **辅助指标**：Goodput分解分析（吞吐量、统计效率）。

### 基线方法对比
- **Static-GBS Baselines**：固定全局批大小，为每个固定批大小选择经过性能剖析的最优并行策略和微批大小。
- **CBS Baselines**：在线自适应批大小（选择最接近GNS估计的临界批大小的候选），但**固定并行策略**。文中测试了两种变体：
  - **悲观型**：针对初始小批大小（Bg=16）优化。
  - **乐观型**：针对训练后期的大批大小优化。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在所有四个配置（3B, 13B, 32B, 7B）上，Copus均取得了显著的速度提升：
- **平均加速**：相比各任务中最快的基线方法，**平均时间节省3.9%至8.0%**。
- **峰值加速**：最高可达 **11.1%** 的速度提升（在32B/4x8H100配置下）。
- **理想上限**：如果消除重配置开销，平均加速可提升至4.7%-11.4%，峰值达13.2%。

### 与基线方法的对比结果
- **优于所有静态基线**：Copus始终快于任何固定批大小的配置。
- **优于或匹配CBS基线**：当自适应批大小轨迹未跨越吞吐量最优策略边界时，Copus表现与最优CBS基线相当；当跨越边界时，Copus通过切换并行策略获得显著优势。
- **案例说明**：在3B模型上，一个固定的“大批次”训练配方（如LLaMA风格，固定Bg=2048）在47分钟后才达到某个损失值，而Copus在**不到2分钟内**就达到了相同损失，凸显了早期使用大批次的巨大浪费。

### 消融实验结果
- **重配置开销分析**（Table 2）：展示了每次并行策略变更的时间。在线重分片（Online Resharding）耗时30-56秒，而传统的检查点重启（Checkpoint-Restart）耗时231-809秒，**加速2.3x至15.7x**。
- **GNS校准验证**：实验表明，即使GNS存在系统性偏差，Copus的Goodput决策依然稳健，因为吞吐量曲线提供了独立的约束。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **批大小与并行策略强耦合**：实验证明，随着全局批大小的变化，吞吐量最优的3D并行策略会发生显著转移（例如，从小批时的PP-heavy策略转向大批时的DP-heavy策略）。
2. **Goodput是更优的优化目标**：单纯追求统计效率（SE）或吞吐量（T）都是片面的。最大化Goodput能有效平衡两者，在单位墙钟时间内实现最快收敛。
3. **协同自适应带来真实收益**：Copus通过在训练过程中动态调整批大小和并行策略，能够持续保持高Goodput，从而显著缩短总训练时间。

### 方法的局限性
1. **决策空间有限**：当前仅优化了Data, Tensor, Pipeline并行度以及批大小，未包含ZeRO-style optimizer sharding、context parallelism等其他重要维度。
2. **GNS校准问题**：仍需一个手动确定的线性校准因子`c`来将GNS映射到临界批大小，该关系可能是非线性的且依赖于模型和训练阶段。
3. **依赖离线性能剖析**：需要为每个（模型，硬件）组合预先生成一个吞吐量查找表（lookup table），增加了部署成本。
4. **规模限制**：实验在最多32块GPU上进行，更大规模集群上的效果尚待验证。

### 未来工作方向
1. **扩展决策空间**：将专家并行（expert parallelism）、上下文并行（context parallelism）等纳入联合优化范围。
2. **自动化GNS校准**：研究如何自动、准确地确定GNS到临界批大小的映射关系。
3. **在线吞吐量建模**：用在线性能预测模型替代离线性能剖析，使系统完全自包含。
4. **探索更复杂的调度**：研究在多作业共享集群中的协同调度策略。

</details>

---

### 6. [FloatSOM: GPU-Accelerated, Distributed, Topology-Flexible Self-Organizing Maps](https://arxiv.org/abs/2604.26555)

**Authors**: Tony Xu, Sarah Klamt, Katherine Turner, Anne Brustle, Felix Marsh-Wakefield, Givanna Putri  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.26555v1  

#### Abstract
GPU-accelerated Self-Organizing Map (SOM) implementations are among the most competitive options for large-scale SOM analysis, but growing dataset sizes increasingly challenge their practical use because workloads no longer fit cleanly within device-memory limits. We introduce FloatSOM, a SOM framew...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：FloatSOM: GPU-Accelerated, Distributed, Topology-Flexible Self-Organizing Maps**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前主流的 **Self-Organizing Map (SOM)** 实现面临以下挑战：
- **内存限制**：大多数实现要求整个数据集加载到 GPU 的 **VRAM** 中，难以处理超大规模数据。
- **计算扩展性差**：缺乏对多 GPU 和分布式训练的支持，限制了在高性能计算（HPC）环境中的应用。
- **拓扑僵化**：传统 SOM 多采用固定的矩形或六边形网格（regular lattice），无法灵活适应复杂、不规则的数据流形结构。

### **提出的新方法与创新**
作者提出了 **FloatSOM**，一个全新的、可扩展的 SOM 框架，具备以下核心创新：

| 创新维度 | 具体内容 |
|--------|--------|
| **系统架构** | 支持 **multi-GPU + 分布式训练**（基于 Ray + NCCL），可在多个 HPC 节点间并行执行。 |
| **内存管理** | 支持 **out-of-memory** 模式，通过磁盘流式加载（disk-backed streaming）处理超出 VRAM/RAM 的数据。 |
| **拓扑灵活性** | 引入两种新型图结构拓扑：<br>- **Minimum Spanning Tree (MST)**<br>- **Relative Neighborhood Graph (RNG)**<br>摆脱固定网格约束，动态构建节点邻接关系。 |
| **采样策略支持** | 支持多种采样方式：`full`, `random`, `HDSSSOM`，用于平衡精度与效率。 |
| **自动化调参** | 集成 **Optuna** 进行多目标超参数优化（multi-objective hyperparameter optimization），提升模型性能。 |

### **相比现有方法的优势**
| 对比项 | FloatSOM | 现有方法（如 XPySOM、Somoclu、GigaSOM） |
|------|---------|----------------------------------|
| GPU 扩展性 | ✅ 支持跨节点多 GPU 并行 | ❌ 多为单 GPU 或 CPU 分布式 |
| 内存容量 | ✅ 支持磁盘流式加载 | ❌ 必须全量载入内存 |
| 拓扑结构 | ✅ 支持 MST/RNG 动态图拓扑 | ❌ 仅支持固定 lattice |
| 性能表现 | ✅ 更低 Quantization Error (QE) | ❌ 固定拓扑表达能力受限 |
| 易用性 | ✅ 统一框架集成采样、拓扑、训练、调优 | ❌ 各模块割裂，需手动组合 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共使用 **14 个数据集**，涵盖合成与真实场景：
- **合成数据集**：`blobs`, `circles`, `moons`, `s_curve`, `swiss_roll`
- **真实数据集**：`iris`, `wine`, `digits`, `breast_cancer`, `diabetes`, `california_housing`, `covertype`, `kddcup99`, `olivetti_faces`

> 数据预处理：标准化（StandardScaler），70%/30% 划分训练/holdout 集。

此外，在速度基准测试中使用 **合成随机矩阵**（uniform [0,1]），样本数从 $10^6$ 到 $10^9$，维度从 50 到 5000 不等。

### **实验设置**
- **硬件平台**：澳大利亚国家计算基础设施（NCI）上的 **Gadi 超算**，使用 `NVIDIA V100 (32GB VRAM)` GPU。
- **并行配置**：测试 $G = \{1, 2, 4, 8\}$ GPU 的分布式性能。
- **运行环境**：基于 **Ray** 实现分布式调度，使用 **NCCL** 进行 GPU 间同步。
- **训练模式**：batch 模式为主，支持 full/random 采样。

### **评估指标**
| 指标 | 定义 | 说明 |
|-----|------|------|
| **Quantization Error (QE)** | 样本与其 BMU（Best Matching Unit）之间的平均距离 | 衡量 SOM 对数据分布的拟合程度 |
| **QET** | 训练集上的 QE | 反映训练拟合能力 |
| **QEH** | Holdout 集上的 QE | 反映泛化能力 |
| **QEB (Balanced QE)** | $(QET + QEH)/2$ | 综合评价指标，兼顾拟合与泛化 |
| **Scaling Efficiency** | $E_G = (T_1 / T_G) / G \times 100\%$ | 衡量多 GPU 加速效率，理想值为 100% |

### **基线方法对比**
- **XPySOM**：主流 GPU 加速 SOM 实现，作为主要对比基线（默认六边形拓扑）。
- **Hexagonal lattice**：作为所有拓扑比较的标准 baseline。
- **Full vs Random Sampling**：分析不同采样策略的影响。
- **MST vs RNG vs Hexagonal**：比较不同拓扑结构的表现。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 场景 | 结果 |
|------|------|
| **最大规模训练任务** | 在 **8 GPUs** 上训练 **10亿样本 × 50维特征** 的数据，使用 **1024 节点 RNG-SOM**，耗时仅 **6.16 分钟**（369.41 秒）。 |
| **最优 QE 提升** | 相比默认 XPySOM（hexagonal + untuned），**tuned FloatSOM-RNG** 实现：<br>- **QEB ↓14.5%**<br>- **QEH ↓9.1%**<br>- **QET ↓22.5%**（见 Fig. 13） |
| **多 GPU 扩展效率** | 在大负载下接近甚至超过线性加速（>100% efficiency），因多 GPU 可保持数据驻留 RAM，避免单卡进入磁盘模式。 |
| **拓扑运行时开销** | 在 grid size=64 时：<br>- Hexagonal: **32.54s**<br>- MST: **266.45s (8.19×)**<br>- RNG: **880.83s (27.07×)**（见 Supp. Table S11） |

### **与基线方法的对比结果**
#### **(1) 拓扑结构对比（Full Sampling）**
| 拓扑 | vs Hexagonal QE 表现 | p-value (QEB) |
|------|------------------|-------------|
| **MST** | QEB ↓，主要来自 QET 提升 | p = 1.12e-05 |
| **RNG** | QEB 显著 ↓，尤其在大型真实数据上 | p = 7.4e-10 |

> RNG 是唯一在 **QEB、QEH、QET** 三项均显著优于 hexagonal 的拓扑。

#### **(2) 采样策略对比**
- **Full vs Random**：
  - 当样本数 > 10,000 时，两者 QE 差异不显著。
  - 小样本下 random 更不稳定，**full sampling 更可靠**。
- **Full vs HDSSSOM**：
  - HDSSSOM 在 pilot 实验中全面劣于 full sampling（50/50 次比较失败），被排除为主要方案。

#### **(3) 超参数调优效果**
- **Tuned vs Untuned (XPySOM 默认)**：
  - 所有拓扑家族（hexagonal/MST/RNG）均显著受益于调优。
  - 全局平均 QE 改善达 **10–20%**（见 Fig. 8）。
  - 证明：**hyperparameter tuning 应视为方法的一部分而非可选步骤**。

### **消融实验结果**
| 实验 | 发现 |
|------|------|
| **不同采样策略对 QE 影响** | Full 最佳，Random 在大数据下可接受，HDSSSOM 表现差。 |
| **拓扑稳定性分析** | MST 和 RNG 的超参数在不同种子间更稳定，表明其对初始化鲁棒性强。 |
| **拓扑刷新频率影响** | 初期频繁更新拓扑，后期放缓，有助于收敛稳定。 |
| **多 GPU 扩展性分析** | 在 sample/dimension scaling 下扩展良好；但在 grid size scaling 下收益有限（瓶颈转向拓扑计算）。 |

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **RNG 拓扑是当前最优选择**：在表达能力和泛化性上全面超越传统六边形网格，尤其适合高维、非均匀分布的真实数据。
2. ✅ **FloatSOM 实现了真正的可扩展性**：支持超大规模数据（十亿级样本）、多 GPU 分布式训练、磁盘流式加载，填补了现有工具链空白。
3. ✅ **超参数调优至关重要**：即使在同一拓扑下，合理调参可带来 **>10% QE 改善**，且应结合拓扑特性进行“topology-aware tuning”。
4. ✅ **采样策略需按规模选择**：
   - 小数据（<10k）：优先使用 **full sampling** 保证稳定性。
   - 大数据：可用 **random sampling** 加速训练，QE 损失可忽略。
5. ✅ **系统设计影响实际性能**：多 GPU 不仅提升吞吐，还能延迟进入磁盘模式，从而间接提升效率（解释了 >100% scaling efficiency）。

### **方法的局限性**
- **RNG/MST 计算开销高**：尤其在 large grid size 场景下，拓扑重建成为瓶颈（RNG 是 hexagonal 的 27× 运行时间）。
- **不适合极小数据集**：启动开销（如 Ray 初始化、磁盘分片）在小任务中占比较高。
- **拓扑动态更新机制仍需权衡**：过于频繁更新影响收敛，过少则失去自适应优势。
- **未支持在线学习（online regime）**：目前仅支持 batch 模式。

### **未来工作方向**
- 开发更高效的 RNG/MST 近似算法以降低拓扑维护成本。
- 探索 **hybrid topology**（如局部 lattice + 全局 RNG）以平衡效率与表达力。
- 支持 **incremental learning / streaming data** 场景。
- 将 FloatSOM 集成至主流 ML 生态（如 PyTorch/TensorFlow 插件）。
- 探索在 **single-cell genomics**, **cytometry**, **spatial transcriptomics** 等领域的具体应用。

---

> **总结一句话**：  
> **FloatSOM 是首个将 topology-flexibility、distributed GPU training、out-of-core execution 和 automated hyperparameter tuning 统一整合的高性能 SOM 框架，在大规模数据分析中展现出显著优于现有方法的性能与实用性。**

</details>

---

### 7. [FaaSMoE: A Serverless Framework for Multi-Tenant Mixture-of-Experts Serving](https://arxiv.org/abs/2604.26881)

**Authors**: Minghe Wang, Trever Schirmer, Mohammadreza Malekabbasi, David Bermbach  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.26881v1  

#### Abstract
Mixture-of-Experts (MoE) models offer high capacity with efficient inference cost by activating a small subset of expert models per input. However, deploying MoE models requires all experts to reside in memory, creating a gap between the resource used by activated experts and the provisioned resourc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FaaSMoE: A Serverless Framework for Multi-Tenant Mixture-of-Experts Serving

## 1. 论文的主要贡献和创新点

### 解决的问题
Mixture-of-Experts (MoE) 模型虽然在推理时仅激活少量专家（sparse activation），但部署时仍需将**所有专家常驻内存**，导致资源利用率低下。这一问题在 **multi-tenant**（多租户）场景中尤为严重，因为每个租户独立部署完整模型会造成大量重复的专家副本，造成严重的内存浪费。

此外，现有优化方法（如 expert offloading、deduplication）大多聚焦于单个模型实例内的效率提升，未能解决跨租户间的专家共享与弹性伸缩问题。

---

### 提出的新方法与思路
作者提出 **FaaSMoE** —— 一种基于 **Function-as-a-Service (FaaS)** 的多租户 MoE 服务架构，其核心思想是：

- 将 MoE 中的 **experts 部署为无状态的 FaaS 函数**，实现按需调用（on-demand invocation）和 scale-to-zero。
- 构建一个轻量级的 **control plane（Orchestrator）** 负责非专家部分计算（如 tokenization、attention、gating 和结果聚合）。
- 支持 **可配置的 expert granularity**：多个专家可打包成一个 “expert block” 作为一个 FaaS 函数部署，以权衡调用开销与弹性粒度。

该设计实现了：
- ✅ **跨租户专家池共享**（shared expert pool）
- ✅ **真正的资源按需分配**
- ✅ **自动扩缩容与强隔离性**

---

### 相比现有方法的优势
| 方法 | 局限性 | FaaSMoE 的优势 |
|------|--------|----------------|
| Baseline（每租户独占全模型） | 冗余高，资源浪费严重 | 资源消耗降至三分之一以下 |
| MoESaic [6] | 仅去重已激活专家，未激活专家仍驻留内存 | 所有未被使用的专家均可 scale-to-zero |
| 分布式 MoE（如 Expert-as-a-Service [13]） | 依赖专用集群、高性能通信，缺乏通用性和弹性 | 利用 FaaS 平台天然支持 autoscaling 与多租户 |
| 单租户 Serverless MoE [12] | 仅优化计费成本，不支持跨租户共享 | 显著降低平均资源占用，支持多租户协同 |

> ✅ **核心优势总结**：FaaSMoE 是首个将 FaaS 用于构建**跨租户共享、弹性可扩展 MoE 推理系统**的框架，在保证模型正确性的前提下极大提升了资源效率。

---

## 2. 核心实验方法和设置

### 使用的数据集
- 实验任务来自 **BIG-Bench** 数据集中的五个异构任务（heterogeneous tasks）。
- 每个客户端提交 5 个不同任务请求，共 6 个并发客户端 → 总计 30 个请求 / 实验轮次。
- 请求具有多样性，能有效触发不同的 expert routing pattern。

---

### 实验设置
- **原型实现**：
  - 模型：`Qwen1.5-MoE-2.7B`（含 60 个专家 + 4 共享专家 / MoE layer）
  - FaaS 平台：`tinyFaaS`（开源边缘导向 FaaS 框架）
  - 运行环境：纯 CPU 服务器（300 GB 内存），无 GPU
- **Orchestrator 部署方式对比**：
  - `FaaSMoE-Shared`：单个中心化 Orchestrator，支持跨租户微批处理（micro-batching）
  - `FaaSMoE-Private`：每个租户拥有独立 Orchestrator，提高隔离性
- **Expert Block Size**：默认设为 20 个专家 / block，测试范围为 6–30

---

### 评估指标
| 指标 | 描述 |
|------|------|
| **CPU Usage (%)** | 每秒采样进程级 CPU 使用率，100% 表示一个核心满载 |
| **Memory Consumption (GB)** | 进程级内存占用均值 |
| **Resource Breakdown** | 区分 client（Orchestrator）、server（expert backend）资源消耗 |
| **Invocation Overhead** | 通过 block size 消融分析调用频率与运行时开销关系 |

---

### 基线方法对比
| 部署策略 | 描述 |
|---------|------|
| **Baseline** | 每租户部署完整的 MoE 模型（标准做法） |
| **Local Distribution** | 租户保留非专家模块，专家集中部署在本地共享服务器（Uvicorn），非弹性 |
| **FaaSMoE-Shared** | 共享 Orchestrator + FaaS 上的专家函数 |
| **FaaSMoE-Private** | 每租户私有 Orchestrator + FaaS 上的专家函数 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（block size = 20）

| 部署策略 | Avg CPU Usage | Avg Memory Usage |
|--------|---------------|------------------|
| **Baseline** | 1126.84% | 217.52 GB |
| **Local Distribution** | 428.67% | 50.38 GB |
| **FaaSMoE-Shared** | **326.40%** | **72.25 GB** |
| **FaaSMoE-Private** | 408.49% | 90.98 GB |

> 🔍 **相比 Baseline，FaaSMoE-Shared 资源节省超过 70%！**

- CPU 使用下降约 **71%**
- 内存使用下降约 **67%**
- 即使与更优的 Local Distribution 相比，FaaSMoE 在内存上略高，但在 **CPU 弹性与专家利用率方面更具优势**

---

### 与基线方法的对比结果
- **Baseline 最差**：因完全复制模型，资源开销巨大。
- **Local Distribution 内存最优**：专家集中管理，避免冗余；但 CPU 较高且无法 scale-to-zero。
- **FaaSMoE-Shared 表现最佳**：得益于跨租户 batching 和连接复用，CPU 开销最低。
- **FaaSMoE-Private 成本稍高**：由于每个租户维护 Orchestrator，带来额外序列化与调度开销，但提供更好隔离性。

> ✅ **结论**：FaaS-based expert pool 显著降低了跨租户的总体资源消耗，尤其适合稀疏、动态的工作负载。

---

### 消融实验结果（Expert Block Size 影响）

#### CPU Usage 趋势
- **Local Distribution**：随 block size 增大单调下降（减少并行组数）
- **FaaSMoE**：呈现 **非单调波动**，在 block size=20 处出现峰值，30 时下降
  - 原因：涉及 batching 效率、FaaS 调用次数、路由开销之间的复杂权衡

#### Memory Usage 趋势（U-shaped 曲线）
- block size 从 6 → 20：内存持续下降（减少运行时重复）
- block size 从 20 → 30：内存反而上升
  - 原因：过大的 block 导致单个函数内存驻留增加，且内部调度复杂度上升

> 📊 **最优 block size ≈ 20**：在调用开销与弹性之间取得良好平衡

---

## 4. 关键结论和发现

### 主要发现
1. **FaaS 是 MoE serving 的理想运行时平台**：
   - 其 event-driven、scale-to-zero 特性完美匹配 MoE 的稀疏激活模式。
2. **Decoupling control and compute plane 是关键**：
   - 将专家作为 FaaS 函数解耦执行，使得真正实现“用时才启”成为可能。
3. **跨租户专家共享显著提升资源效率**：
   - 相比每租户独占模型，FaaSMoE 将平均资源消耗降至 **不到三分之一**。
4. **Orchestrator 放置影响效率与隔离性权衡**：
   - 共享 Orchestrator 更高效（batching + 复用），私有则更安全可靠。
5. **Expert block granularity 是重要设计参数**：
   - 不应固定，而应根据 workload 特征进行调整或自适应。

---

### 方法的局限性
1. **延迟与网络开销**：
   - 控制平面与专家函数间频繁通信引入额外网络延迟，可能影响端到端推理速度。
2. **当前仅支持 CPU 执行**：
   - 主流 FaaS 平台对 GPU 支持有限，限制了大规模 backbone 的加速能力。
3. **冷启动问题未深入探讨**：
   - 新专家函数首次调用可能存在冷启动延迟，虽可通过预热缓解，但文中未量化分析。
4. **未考虑动态负载突增下的稳定性**：
   - 大规模突发请求可能导致 FaaS 平台调度瓶颈。

---

### 未来工作方向
1. **自适应 block granularity 策略**：
   - 根据访问热度动态合并“hot experts”，拆分冷门专家。
2. **低延迟通信优化**：
   - 结合 high-speed RPC 或流水线技术隐藏通信延迟。
3. **Hybrid Orchestrator Placement**：
   - 动态选择共享 vs 私有部署，基于负载特征自动切换。
4. **GPU-aware FaaS 扩展**：
   - 探索支持 GPU serverless 的 MoE 部署方案，兼顾性能与弹性。
5. **真实生产环境验证**：
   - 在 AWS Lambda、Google Cloud Functions 等主流平台上部署验证可行性。

---

> ✅ **最终结论**：  
> **FaaSMoE 为 multi-tenant MoE serving 提供了一条切实可行、高度资源高效的路径**。它利用 FaaS 的弹性特性打破了传统部署中“必须常驻全部专家”的桎梏，推动 MoE 模型向更经济、更可持续的大规模服务演进。

</details>

---

### 8. [Who Trains Matters: Federated Learning under Enrollment and Participation Selection Biases](https://arxiv.org/abs/2604.26604)

**Authors**: Gota Morishita  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.26604v1  

#### Abstract
Federated learning (FL) trains a shared model from updates contributed by distributed clients, often implicitly assuming that contributing clients are representative of the target population. In practice, this representativeness assumption can fail at two distinct stages, inducing selection bias. Fi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Who Trains Matters: Federated Learning under Enrollment and Participation Selection Biases**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

该论文指出，**联邦学习（Federated Learning, FL）中的客户端选择偏差（selection bias）不仅存在于每轮通信的参与阶段（participation bias），更根本地存在于初始的注册/入组阶段（enrollment bias）**。

- **Enrollment Bias（注册偏差）**：由于设备限制、软件版本、用户授权等原因，并非所有目标人群都能被注册进训练系统，导致可训练客户端群体本身就不具代表性。
- **Participation Bias（参与偏差）**：即使已注册，客户端是否参与某一轮训练还受电量、网络状态等动态因素影响。

现有研究大多关注**参与偏差**，但忽略了**注册偏差**这一更深层、持续性的偏差来源。这会导致即使完美纠正了参与偏差，模型优化的目标仍然是一个“错误”的目标——即偏向于已注册客户端分布，而非真正的目标人群（target population）。

---

### ✅ **提出了什么新方法或新思路**

#### （1）**两阶段选择建模框架（Two-stage Selection Model）**
将客户端纳入过程分解为两个因果有序的阶段：
- 第一阶段：`Enrollment` → 是否能被系统访问
- 第二阶段：`Participation | Enrollment` → 已注册客户中谁实际参与本轮训练

并证明在一定假设下，总体纳入概率可分解为：
$$
\pi_{i,r} = \pi_i^{\text{enroll}} \cdot \pi_{i,r}^{\text{part}}
$$

#### （2）**FedIPW：基于逆概率加权的聚合算法**
提出 **FedIPW**，一种改进版的 FedAvg 聚合机制，对每个客户端更新进行逆概率加权（Inverse Probability Weighting, IPW）：
$$
\Delta^{\text{FedIPW}} = \sum_{i \in S_r} \frac{1}{\hat{\pi}_{i,r}} \Delta_{i,r}
$$
其中 $\hat{\pi}_{i,r} = \hat{\pi}_i^{\text{enroll}} \cdot \hat{\pi}_{i,r}^{\text{part}}$ 是估计的两阶段联合纳入概率。

在 ignorability 和 positivity 假设下，该方法能无偏估计目标人群的平均梯度更新。

#### （3）**Aggregate Calibration：有限信息下的近似纠偏方法**
当无法获取未注册用户的个体级协变量时（常见于真实部署），提出一种仅利用**总体统计摘要**（如各地区/设备类型的占比）来重新加权已注册样本的方法，称为 aggregate calibration。

这是一种实用的退而求其次方案，适用于现实约束场景。

#### （4）**理论分析：残差加权误差导致“偏差地板”（bias floor）**
首次提供了一个**算法无关的优化误差分析框架**，表明：
- 若存在残差加权误差（residual weighting error）$\epsilon_w$，则最终收敛误差中会出现一个**非消失的偏差项**（bias floor），其大小为：
  $$
  \mathcal{O}\left(\frac{\epsilon_w^2 G^2}{\mu}\right)
  $$
  其中 $G^2$ 表示客户端间梯度异质性，$\mu$ 是强凸参数。
- 这意味着：**只要选择机制未完全纠正，无论训练多久都无法消除目标人群上的泛化差距**。

---

### ✅ **相比现有方法的优势**

| 方面 | 现有方法 | 本文方法 |
|------|--------|---------|
| 建模范式 | 多数只建模 round-level 参与偏差 | 显式区分 enrollment 与 participation 两阶段 |
| 纠偏完整性 | 忽略 enrollment bias 导致目标错位 | 完整纠正两阶段偏差才能对齐目标人群目标 |
| 实用性扩展 | 需要个体级注册数据 | 支持 aggregate calibration，在仅有总体统计时仍可部分纠偏 |
| 理论深度 | 缺乏对“纠偏不全”的长期影响分析 | 揭示了残差权重误差如何导致不可忽略的 bias floor |

---

## 2. **核心实验方法和设置**

### ✅ **使用了哪些数据集**

- 使用**合成数据集（synthetic dataset）** 构造的 **federated logistic regression** 任务。
- 数据生成方式可控，确保可以精确计算出**目标人群最优解 $\theta^*$**，从而直接衡量“目标人群损失”（population excess loss）。

> 注：虽然没有使用真实世界数据集（如 FEMNIST 或 CIFAR-100），但这种设计是为了**精准隔离 selection bias 的影响**，排除模型误设、非凸性等干扰因素。

---

### ✅ **实验设置和评估指标**

#### 📌 设置要点：
- 固定总客户端数量 $N$。
- 设计两个阶段的选择机制：
  - **Enrollment Mechanism**：基于预注册协变量 $Z_i$（如区域、设备类型）决定是否注册。
  - **Participation Mechanism**：在已注册客户端中，基于当前系统状态 $X_{i,r}$ 决定是否参与本轮。
- 控制 enrollment bias 强度（从 None 到 Severe），观察不同方法的表现变化。

#### 📌 评估指标：
- **Target-Population Excess Loss**: $F(\theta_R) - F^*$，即最终模型在目标人群上的损失超出最小值的程度。
- **Convergence to Oracle IPW**：是否接近使用真实纳入概率的理想 IPW 方法。
- **Sensitivity to Moment Noise**：测试 aggregate calibration 对输入总体统计噪声的鲁棒性。

---

### ✅ **基线方法对比**

比较了以下四种聚合策略：

| 方法 | 是否纠正参与偏差 | 是否纠正注册偏差 | 权重方式 |
|------|------------------|------------------|----------|
| **Naive FedAvg** | ❌ | ❌ | 均等权重 |
| **Round-only IPW** | ✅ | ❌ | 仅用 $\hat{\pi}_{i,r}^{\text{part}}$ 加权 |
| **FedIPW** | ✅ | ✅ | 用 $\hat{\pi}_i^{\text{enroll}} \cdot \hat{\pi}_{i,r}^{\text{part}}$ 加权 |
| **Oracle IPW** | ✅ | ✅ | 使用真实 $\pi_{i,r}$，理想上限 |

此外还包括：
- **Aggregate Calibration**：作为 FedIPW 在有限信息下的替代方案。

---

## 3. **主要实验结果和性能指标**

### ✅ **关键性能数据与对比结果**

#### 🔹 图 2A：两阶段选择下的收敛轨迹
- **FedAvg** 很快收敛，但停滞在一个较高的目标人群损失水平 → 收敛到“选中客户端”的最优解，而非目标人群最优。
- **Round-only IPW** 有所改善，但仍存在明显差距。
- **FedIPW** 几乎完全贴合 **Oracle IPW** 曲线 → 成功恢复目标人群目标。

#### 🔹 图 2B：随注册偏差强度增加的最终性能
| 方法 | 无偏差 | 轻微偏差 | 中等偏差 | 严重偏差 |
|------|-------|---------|---------|---------|
| FedAvg | ~0.03 | ~0.035 | ~0.045 | ~0.058 |
| Round-only IPW | ~0.03 | ~0.032 | ~0.040 | ~0.052 |
| **FedIPW / Oracle IPW** | **~0.03** | **~0.030** | **~0.030** | **~0.030** |

👉 结论：**只有 FedIPW 能稳定抵抗不断增强的 enrollment bias**；其他方法性能随偏差加剧显著下降。

#### 🔹 图 2C：aggregate calibration 效果（使用准确总体矩）
- 在无法获得个体级注册数据的情况下，**aggregate calibration 显著缩小了 round-only IPW 与目标之间的差距**。
- 尤其当注册偏差由可观测特征驱动时，效果显著。

#### 🔹 图 2D：对总体矩噪声的敏感性
- 当提供的总体统计含噪时，calibration 效果下降。
- 轻微噪声下仍有增益；高噪声下可能失效甚至有害。
- 表明：**aggregate calibration 是一种稳健但非万能的 fallback 策略**。

---

### ✅ **消融实验结果**

- **Corollary 6.2 & Proposition 6.3** 提供了理论层面的“消融”：
  - 若仅纠正 participation 而忽略 enrollment，则残差权重误差为 $\left|1 - \pi_i^{\text{enroll}}\right|$。
  - 此误差会进入 bias floor 项，且无法通过更多训练消除。
- 实验证实：**遗漏 enrollment correction 会造成不可压缩的性能天花板**。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **“谁可以训练”比“谁参与本轮训练”更重要**  
   → Enrollment bias 是一种结构性、持续性的偏差源，若不纠正，整个 FL 系统优化的就是一个错误目标。

2. **两阶段建模是必要的**  
   → 将 enrollment 与 participation 分开建模可避免模型误设（model misspecification），提升纠偏有效性。

3. **FedIPW 可有效恢复目标人群目标**  
   → 在知道或能估计纳入概率的前提下，IPW 加权能使聚合更新无偏于目标人群均值更新。

4. **纠偏不全会导致“偏差地板”**  
   → 存在残差加权误差时，无论训练多长时间，都会留下一个与 $\epsilon_w^2 G^2/\mu$ 成正比的误差下限。

5. **Aggregate Calibration 是一种实用折衷方案**  
   → 在缺乏个体级注册数据时，利用外部总体统计仍可实现部分纠偏。

---

### ⚠️ **方法的局限性**

| 局限性 | 说明 |
|--------|------|
| **依赖协变量质量** | FedIPW 需要高质量的 pre-enrollment covariates（如设备类型、地域）用于估计倾向得分。若关键变量缺失，纠偏效果受限。 |
| **需要外部人口统计** | Aggregate calibration 依赖可靠的 target-population summary statistics，这些在现实中可能难以获取或更新滞后。 |
| **强假设前提** | IPW 方法依赖 ignorability 和 positivity 假设，在复杂系统中可能不成立（例如存在隐藏混杂因子）。 |
| **仅验证于凸问题** | 理论分析基于 strongly convex 目标函数，对非凸深度学习任务的适用性需进一步验证。 |

---

### 🔮 **未来工作方向**

1. **开发无需显式建模 enrollment 的自适应纠偏方法**  
   → 如利用 representation learning 或 domain adaptation 技术隐式对齐分布。

2. **结合 drift/variance reduction 方法**  
   → 将 FedIPW 与 SCAFFOLD、FedProx 等方法结合，同时处理 selection bias 与 client drift。

3. **在线估计倾向得分**  
   → 设计能在运行时动态更新 $\hat{\pi}_i^{\text{enroll}}, \hat{\pi}_{i,r}^{\text{part}}$ 的轻量级模型。

4. **扩展至异构模型架构场景**  
   → 当客户端使用不同模型结构时，如何定义和纠正 selection bias？

5. **隐私保护下的纠偏机制**  
   → 如何在满足 DP 或 secure aggregation 的前提下实施 IPW 或 calibration？

---

> 💬 **一句话总结**：  
> **联邦学习的成功不仅取决于“谁参与了训练”，更取决于“谁能被允许训练”。本文揭示了 enrollment bias 的深远影响，并提供了 FedIPW 和 aggregate calibration 两种层次化的解决方案，强调了在系统设计初期就应考虑人群代表性的必要性。**

</details>

---

### 9. [EvoSelect: Data-Efficient LLM Evolution for Targeted Task Adaptation](https://arxiv.org/abs/2604.26170)

**Authors**: Ting-Wei Li, Sirui Chen, Jiaru Zou, Yingbing Huang, Tianxin Wei, Jingrui He, Hanghang Tong  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.26170v1  

#### Abstract
Adapting large language models (LLMs) to a targeted task efficiently and effectively remains a fundamental challenge. Such adaptation often requires iteratively improving the model toward a targeted task, yet collecting high-quality human-labeled data to support this process is costly and difficult ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# EvoSELECT: Data-Efficient LLM Evolution for Targeted Task Adaptation —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大型语言模型（LLM）的**目标任务适配**（targeted task adaptation）中，通常采用迭代的生成-训练循环（generation-training loop），即通过外部生成器合成数据并用于微调模型。然而，该过程面临两个核心挑战：
- **数据漂移**（data drift）：生成的数据可能逐渐偏离目标任务分布，导致对齐性下降。
- **冗余性**（redundancy）：连续迭代中生成的样本高度重叠，增加计算成本且稀释学习信号。

直接训练所有生成数据可能导致性能下降或无效学习。

### 🚀 提出的新方法：EvoSELECT
作者提出了一种新的范式：**迭代生成-选择-训练循环**（iterative generation-selection-training loop），并在其基础上设计了 **EvoSELECT** 框架，核心思想是：
> 在每次迭代中，先从候选生成数据中**联合建模任务对齐性**（task alignment）**和多样性**（diversity），再进行模型更新。

#### 创新点：
1. **基于最优传输的任务对齐机制**（Optimal Transport-based Alignment）
   - 使用 **Optimal Transport (OT)** 对齐生成数据与验证集之间的梯度表示分布。
   - 相比传统 attribution 方法仅匹配“平均中心”（centroid），OT 能捕捉**全局几何结构**，实现更全面的目标覆盖。

2. **显式的多样性正则化**（Diversity Regularization）
   - 引入基于内积相似性的 **diversity energy** 项，惩罚梯度空间中过于密集的样本选择。
   - 防止选中大量相似样本，提升数据利用效率。

3. **统一优化框架**
   - 将 OT 梯度（对齐信号）与多样性梯度结合，通过指数更新规则联合优化样本权重。
   - 实现了对齐性与多样性的**协同演化**（synergized），而非简单的加权组合。

### 🔍 相比现有方法的优势
| 方法类型 | 典型代表 | 缺陷 | EvoSELECT 如何改进 |
|--------|---------|------|------------------|
| Attribution-only | Xia et al. (2024) | 只关注与验证集相似性，集中在少数簇，缺乏多样性 | 使用 OT 进行分布级对齐，避免中心偏置 |
| Diversity-only | Jung et al. (2025) | 忽视任务相关性，可能选中无关但多样的噪声 | 显式建模任务对齐作为首要目标 |
| Heuristic 组合 | Attr-Div, TSDS | 多为串行处理或启发式加权，未真正融合两种信号 | 统一梯度优化框架，动态平衡两者 |

> ✅ **核心优势**：EvoSELECT 是首个将 OT 与多样性目标**直接融合于选择过程**的框架，实现了原则性更强、更高效的数据筛选。

---

## 2. 核心实验方法和设置

### 📚 数据集
共涵盖 **10个基准数据集**，分为三类：
- **科学推理**：ARC-Challenge, MMLU, OpenBookQA, ClimaQA
- **常识/逻辑推理**：CommonsenseQA, LogiQA, LogiQA2
- **生物医学/医疗**：Med-MCQA, MedQA, HeadQA

### ⚙️ 实验设置
- **模型架构**：
  - 主要使用 **Qwen2.5** 系列模型。
  - 基础模型（base model）：`Qwen2.5-3B-Instruct` 或 `14B-Instruct`
  - 数据生成器（generator）：固定为 `Qwen2.5-14B-Instruct`
  - 代理模型（proxy model）：`Qwen2.5-0.5B-Instruct`（用于高效梯度特征提取）

- **训练流程**：
  - 两轮迭代（Iter. 1 & 2）的 generation-selection-training 循环。
  - 使用 **LoRA** 进行参数高效微调（PEFT）。
  - 选择比例（selection ratio）设为 0.2 和 0.5。

- **评估指标**：
  - 主要指标：**准确率**（accuracy）在测试集上的表现。
  - 辅助分析指标：
    - **Vendi Score**：衡量所选数据的多样性（越高越好）。
    - **OT Distance**：反映分布对齐程度。

### 🆚 基线方法对比
| 类别 | 方法 | 描述 |
|------|------|------|
| 简单策略 | `All`, `Random` | 训练全部数据 / 随机采样 |
| 对齐优先 | `Attribution` (Xia et al., 2024) | 基于梯度相似性排序 |
| 多样性优先 | `Diversity` (Jung et al., 2025) | K-means聚类后从小簇中采样 |
| 启发式组合 | `Attr-Div` | 先按 attribution 筛掉底部25%，再应用 diversity 选择 |
| 先进方法 | `TSDS` (Liu et al., 2024) | 基于 OT + KDE 抑制冗余，采样选择 |

---

## 3. 主要实验结果和性能指标

### 📊 性能汇总（以 3B base model 为例）

#### 表格摘要（取部分平均值）：

| 方法 | Avg. Acc @0.2 | Avg. Acc @0.5 | 是否持续优于 Base？ |
|------|---------------|---------------|--------------------|
| Base | 0.7428 | 0.7428 | — |
| All | 0.7416 | 0.7416 | ❌ |
| Random | 0.7463 | 0.7463 | ⚠️ 不稳定 |
| Attribution | 0.7423 | 0.7504 | ⚠️ 有时退化 |
| Diversity | 0.7424 | 0.7498 | ⚠️ 有失败案例 |
| TSDS | 0.7495 | 0.7443 | ⚠️ 第二轮下降 |
| **EvoSELECT** | **0.7580** | **0.7599** | ✅ **始终提升** |

> 💡 **关键观察**：
> - EvoSELECT 在 **所有任务类别、选择比例和迭代轮次** 中均取得最佳或次佳性能。
> - 是**唯一一个在两轮迭代中都稳定超越 base model** 的方法（见 Figure 3）。

### 🔬 关键图表分析

#### Figure 3: 相对于 Base Model 的增益
- 所有其他方法在某些任务上出现负增益（红色），表明“有害适配”（harmful adaptation）。
- **只有 EvoSELECT 在所有设置下均为正增益**（绿色），说明其鲁棒性和有效性。

#### Figure 4: 性能增益 vs 任务难度
- 当 base model 准确率较低（即任务越难）时，EvoSELECT 的相对提升越大。
- 表明该方法在**高需求适应场景下更具价值**。

#### Figure 5: 相对于 Full-data Training 的表现
- 即使只选择 **20% 的数据**，EvoSELECT 也能**超过训练全部数据的效果**。
- 证明其具备强大的**高质量子集识别能力**。

#### Figure 6: 不同任务簇中的 Rank-1 Win Rate
- EvoSELECT 在三大任务簇（科学、常识/逻辑、医疗）中均获得最高胜率（>60%）。
- 显示其跨领域泛化能力强。

### 🔍 消融实验（隐含分析）
虽然未明确列出消融表，但从以下方面可推断组件重要性：
- **OT vs Centroid Matching**：Figure 1 显示 attribution 方法聚焦于单一簇，而 EvoSELECT 覆盖更广区域。
- **联合优化 vs 分离处理**：TSDS 虽也用 OT，但其局部多样性增强效果有限；EvoSELECT 的全局联合优化更优。
- **多样性影响**：Figure 2 显示 attribution 方法 Vendi Score 最低，而 EvoSELECT 平衡二者。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **生成数据需谨慎选择**：盲目训练所有合成数据不仅无益，反而可能导致性能下降。
2. **对齐性 + 多样性必须协同优化**：单独强调任一方面都会导致次优解；EvoSELECT 通过统一框架实现双赢。
3. **OT 是建模任务对齐的有效工具**：相比 centroid-based 方法，OT 更好地保留了目标分布的结构信息。
4. **EvoSELECT 具备强鲁棒性**：
   - 在弱/强生成器设置下均有效；
   - 跨不同任务类型、选择比例和迭代次数保持领先；
   - 是**唯一防止有害适配的方法**。

### ⚠️ 局限性
1. **依赖代理模型**（proxy model）进行梯度近似，虽降低开销，但仍引入一定偏差。
2. **计算复杂度较高**：OT 计算（尤其是 Sinkhorn 迭代）在大规模数据下仍有一定负担（时间复杂度 $O(Tnm\log(\max(n,m))\gamma^{-2})$）。
3. **当前仅适用于分类/问答类任务**，对生成式任务（如摘要、对话）的扩展尚待验证。

### 🔮 未来工作方向
1. **扩展至更多任务形式**：如文本生成、指令跟随等开放域任务。
2. **在线选择机制**：将选择过程嵌入到训练动态中，实现实时反馈调整。
3. **减少 OT 开销**：探索更高效的近似算法或稀疏化策略。
4. **理论分析**：提供关于“为何联合优化能防止有害适配”的收敛性或泛化界分析。

---

> ✅ **一句话总结**：  
> **EvoSELECT 通过将 Optimal Transport 与多样性正则化深度融合，在迭代 LLM 适配中实现了高效、稳健且始终有益的数据选择，显著优于现有方法。**

</details>

---

### 10. [Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving](https://arxiv.org/abs/2604.26837)

**Authors**: Zihan Zhao, Baotong Lu, Shengjie Lin, Yizou Chen, Jing Liu, Yanqi Zhang, Ziming Miao, Ming-Chang Yang, Haiying Shen, Qi Chen, Fan Yang  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.26837v1  

#### Abstract
Long-context LLM serving is bottlenecked by the cost of attending over ever-growing KV caches. Dynamic sparse attention promises relief by accessing only a small, query-dependent subset of the KV state per decoding step and extending the KV storage to CPU memory. In practice, however, these algorith...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving*

---

## 1. 论文的主要贡献和创新点

### **解决的问题**

现代大语言模型（LLM）在支持长上下文（long-context）推理、文档理解和代码生成等任务时，面临 **KV Cache**（Key-Value Cache）随序列长度线性增长带来的两大瓶颈：

- **GPU 内存容量压力**：KV Cache 占用大量 HBM（High Bandwidth Memory），限制并发请求数。
- **内存带宽瓶颈**：解码每一步需读取全部历史 KV 状态，导致计算受限于内存带宽。

虽然 **dynamic sparse attention** 算法通过仅访问关键 token 子集来缓解这一问题，并可将完整 KV Cache 存储在 CPU 内存中按需加载，但在系统层面却存在以下挑战：

1. **缺乏统一抽象**：不同稀疏算法（如 block-level、cluster-level）粒度不一，难以共享优化的内存管理、调度和内核。
2. **不规则数据访问开销大**：稀疏访问模式导致跨 GPU-CPU 边界的细粒度、非连续数据传输，严重浪费 PCIe 带宽。
3. **元数据开销爆炸**：head-wise 稀疏性和长上下文导致页表等元数据占用数十甚至上百 GB GPU 内存。

这些问题使得算法级的稀疏性优势无法转化为端到端的系统性能提升。

---

### **提出的新方法：SPIN**

作者提出了 **SPIN**，一个面向稀疏注意力的推理框架，通过软硬件协同设计解决上述系统瓶颈。其三大核心技术为：

#### ✅ **(1) 统一分区抽象（Unified Partition Abstraction）**
- 引入 **partition** 作为逻辑单元，统一表示不同稀疏算法中的块（block）、簇（cluster）或 token 组。
- 将算法特定的稀疏单元映射到底层基于 page 的 KV 管理机制上，实现“算法灵活性”与“系统效率”的解耦。
- 支持多种代表性算法（ShadowKV、RetroInfer、SeerAttention-R）以插件形式集成。

#### ✅ **(2) 局部性感知的 KV 缓存管理（Locality-aware KV Cache Manager）**
- 设计动态缓冲区，支持 per-request 的 HBM 配额弹性调整。
- 提出 **bucketed LRU** 替换策略：使用有限范围的时间戳（如 64 个桶）近似 LRU，避免全局排序开销，适合 GPU 并行执行。
- 利用自回归解码中关键 token 的时间局部性，缓存高频访问 partitions，显著减少 PCIe 往返次数。

#### ✅ **(3) 两级分层元数据布局（Two-level Hierarchical Metadata）**
- 元数据（如页表）不再按最坏情况预分配，而是采用操作系统风格的多级索引（multi-level indexing）。
- 元数据分布在 GPU 和 CPU 之间：
  - GPU 保留轻量级顶层目录；
  - CPU 存储实际页映射，通过 pinned memory 被 GPU 直接访问。
- 显著降低 GPU 上的元数据占用，释放更多 HBM 给 KV Cache 使用。

---

### **相比现有方法的优势**

| 方面 | 现有方法（如 vLLM、原始稀疏实现） | SPIN |
|------|-------------------------------|------|
| 抽象能力 | 缺乏通用接口，每个稀疏算法需重写内存管理 | 统一 pipeline + 可插拔算法模块 |
| 数据移动效率 | 不利用时间局部性，频繁 PCIe 传输 | bucketed LRU 提升缓存命中率 |
| 元数据开销 | 扁平化页表，内存占用随最大上下文线性增长 | 多级索引 + 分层存储，仅随工作集增长 |
| 系统复用性 | 各原型独立开发，难共享优化组件 | 共享 Offload/Retrieve 优化路径 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **LongBench-v2** [7]：真实场景下的长输入短输出任务（如问答、摘要），平均输入长度约 55K tokens。
- **LongGenBench** [36]：强调推理能力的任务，输入较短但输出更长（平均输出 12K tokens），体现生成阶段的压力。

### **实验平台**

| 类型 | 配置 |
|------|------|
| A100 Server | 4× NVIDIA A100 (80GB HBM), PCIe Gen4 (32 GB/s/GPU), AMD EPYC CPU, 850GB DRAM |
| B200 Server | 4× NVIDIA B200 (180GB HBM), PCIe Gen5 (64 GB/s/GPU), Intel Xeon CPU, 1.5TB DRAM |

### **模型**

- Qwen3-14B、Qwen3-32B、Llama-3.1-70B（覆盖不同规模和架构）

### **评估指标**

- **端到端吞吐量（end-to-end throughput）**：单位时间内生成的 output tokens 数量。
- **TTFT（Time to First Token）**：首 token 延迟。
- **TPOT（Time Per Output Token）**：单 token 解码延迟。
- **平均批大小（average batch size）**：反映系统并发能力。
- **元数据 HBM 消耗**：衡量内存效率。
- **消融实验**：验证各组件对性能的影响。

### **基线方法对比**

| 基线 | 描述 |
|------|------|
| **vLLM** | 当前主流 LLM 推理引擎，使用 PagedAttention 进行密集注意力管理 |
| **vLLM-Offload** | 扩展版 vLLM，支持在 HBM 不足时将 KV Cache 卸载至 CPU |
| **LServe** [69] | GPU-only 架构，利用稀疏性加速，但不支持跨设备分层存储 |
| **原始稀疏实现** | 如 ShadowKV、RetroInfer 的原生原型系统（无成熟调度器） |

> 注：SPIN 在 vLLM 基础上构建，集成了三种稀疏算法形成 SPIN-ShadowKV、SPIN-RetroInfer、SPIN-SeerAttention。

---

## 3. 主要实验结果和性能指标

### **端到端性能（Online Serving）**

#### 🔹 吞吐量提升显著
- 在 A100 上，SPIN 相比 vLLM 实现 **2.34–5.66× 更高的 end-to-end throughput**。
- 在高负载下仍能持续扩展，而 vLLM 因内存抖动迅速饱和。
- 在 B200 上也取得 **1.66–4.03×** 的吞吐增益。

#### 🔹 TTFT 大幅降低
- 在请求速率为 0.5 req/s 时，SPIN-ShadowKV 实现 **7–9× 更低的 TTFT**。
- 原因：更大批处理能力减少了排队延迟。

#### 🔹 批大小显著增加
- SPIN 支持的平均批大小是 vLLM 的 **2.5–3.3×**，说明其有效缓解了内存压力，提升了并发性。

#### 🔹 TPOT 对比
- SPIN 的 TPOT 略高于 vLLM（因引入 retrieval 开销），但保持稳定。
- SPIN-SeerAttention 因固定 retrieval budget，在长上下文中表现更好。

---

### **离线性能分析（Offline Evaluation）**

#### 🔹 Prefill 阶段
- SPIN 相比 vLLM 仅增加最多 1.32 秒（64K 上下文），且部分配置下反而更快（得益于异步索引与卸载）。
- LServe 因采用稀疏 prefill，在 prefill 阶段略优，但该优化正交于 SPIN，可融合。

#### 🔹 Decode 阶段
- SPIN 支持 **4–8× 更大的最大批大小**。
- Decode 吞吐量提升 **1.34–3.49×**。

#### 🔹 与原始稀疏实现对比
- SPIN 相比原始稀疏算法实现：
  - 减少 **21–58% 的 per-token decode latency**。
  - 最高带来 **2.39× 的吞吐提升**。
- 性能收益来源：
  - 对 ShadowKV/RetroInfer：主要来自 **PCIe retrieval 时间下降**（局部性缓存）。
  - 对 SeerAttention-R：主要来自 **更高效的 GPU kernels** 替代原低效算子。

---

### **消融实验（Ablation Studies）**

#### ✅ 动态缓存管理效果
- “Base”（无缓存）→ “+Mandatory”（保留当前 step 必需页）→ “+Buffering”（启用 bucketed LRU）
- 最终 decode throughput 提升 **49%**，证明跨步局部性利用的有效性。

#### ✅ 缓存大小影响
- 缓存命中率随缓冲区增大而上升，但在 **buffering : mandatory ≥ 4×** 后趋于饱和。
- 建议设置最小缓冲为 mandatory 的 **5×**，平衡命中率与内存占用。

#### ✅ 多级索引节省元数据
- 相比扁平页表设计，SPIN 的 multi-level indexing + tier-split 减少元数据 HBM 消耗：
  - **13.9–17.4×**（索引优化）
  - 再加上 CPU offload，总体达 **49–78× 减少**。
- 例如 Llama-3.1-70B 在 128K 上下文下，元数据从 100GB 降至 <2GB。

---

## 4. 关键结论和发现

### **主要发现**

1. **算法级稀疏 ≠ 系统级高效**：即使稀疏算法理论上节省 95% KV 访问，若无高效的数据管理和局部性利用，实际性能可能不如高度优化的 dense 系统。
2. **统一抽象至关重要**：通过 partition 抽象，SPIN 成功将多样化的稀疏算法纳入同一执行管道，实现了“一次优化，处处受益”。
3. **局部性是突破口**：自回归解码具有强时间局部性，bucketed LRU 策略能低成本捕捉此特性，极大减少 PCIe 通信。
4. **元数据不可忽视**：在超长上下文下，元数据本身成为瓶颈；SPIN 的分层、按需分配策略解决了这一“暗开销”。

---

### **方法的局限性**

1. **依赖外部稀疏算法准确性**：SPIN 不改进稀疏选择逻辑本身，其性能上限受制于底层算法（如 ShadowKV 是否准确识别 outlier）。
2. **PCIe 仍是潜在瓶颈**：尽管优化了传输效率，但在极端稀疏、极高并发场景下，PCIe 带宽仍可能成为限制。
3. **目前聚焦 decode 阶段**：prefill 阶段虽兼容，但未深度优化稀疏 prefill（如 LServe 所做）。

---

### **未来工作方向**

1. **结合稀疏 prefill**：将 prefill 阶段的稀疏性也纳入统一框架，进一步压缩早期计算成本。
2. **支持更多稀疏模式**：如 structured sparsity、activation sparsity 等，拓展 SPIN 的适用边界。
3. **探索新型硬件支持**：如 CXL 内存池、智能网卡辅助检索，进一步打破 GPU-CPU 数据墙。
4. **自动化调参**：根据 workload 特征自动调节 buffer size、replacement policy 参数。

---

> 📌 **总结一句话**：  
> SPIN 通过 **统一抽象 + 局部性感知缓存 + 分层元数据**，成功将 dynamic sparse attention 的算法潜力转化为实实在在的系统性能飞跃，在多个模型、硬件和 workload 下实现 **1.66–5.66× 吞吐提升** 和 **7–9× TTFT 降低**，为可扩展的长上下文 LLM serving 提供了一套完整的系统解决方案。

</details>

---

### 11. [DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference](https://arxiv.org/abs/2604.26074)

**Authors**: Shouxu Lin, Zhiyuan Guo, Jiaxin Lin  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.26074v1  

#### Abstract
LLM inference is constrained by GPU memory capacity and bandwidth. Tiered memory architectures mitigate this by allowing the GPU to offload memory to the remote tier. However, existing memory offloading frameworks rely on prefetching data into local GPU HBM. This approach underutilizes system resour...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
大型语言模型（LLM）推理面临严重的 **GPU memory capacity** 和 **bandwidth bottleneck**。传统 tiered memory 架构通过将部分权重和 KV Cache 卸载到远程内存（如 CPU 内存）来缓解该问题，但现有基于 **prefetching** 的方法存在以下缺陷：
- 引入 HBM contention，降低本地带宽利用率；
- 需要额外的 staging buffer，浪费 GPU HBM 容量；
- 粗粒度预取难以实现计算与通信的完美重叠，导致 pipeline bubbles。

### 🚀 提出的新方法与创新思路
论文提出 **DAK**（Direct Access Kernel），一种端到端的 **direct-access memory offloading 框架**，其核心思想是：
> 允许 GPU SM（Streaming Multiprocessor）直接从远程内存加载数据至 SMEM（Shared Memory），绕过 HBM 中转。

#### 主要创新点包括：

| 创新点 | 描述 |
|--------|------|
| **1. Direct Memory Offload Architecture** | 设计 warp-specialized SplitK_GEMM 和 SplitK_FlashAttn 内核，利用 TMA（Tensor Memory Accelerator）异步地从 host memory 直接拉取权重/KV Cache 到 SMEM，实现真正的带宽聚合（bandwidth aggregation）。 |
| **2. Optimal Per-operation Offload Ratio** | 提出一个 **贪心算法** 来为每个操作（operation）分配最优卸载比例（offloading ratio），考虑 compute-bound 与 memory-bound 操作的不同敏感性，理论证明其最优性。 |
| **3. Efficient TMA Access to Host Memory** | 引入主动拥塞控制（active congestion control）防止远程请求阻塞本地 HBM 访问；采用 **TMA multicast + host-locality-first 调度** 消除 uncachable host memory 导致的读放大（read amplification）。 |
| **4. Cross-Architecture Generality** | 支持多种硬件架构（NVLink-C2C / PCIe）、不同 GPU 架构（Hopper / Blackwell）和模型配置，具备良好的可移植性和鲁棒性。 |

### 🔍 相比现有方法的优势
| 维度 | 传统 prefetching 方法 | DAK |
|------|------------------------|-----|
| 数据路径 | `Host → HBM → SMEM`（两跳） | `Host → SMEM`（直通） |
| 带宽利用 | HBM contention，无法聚合系统总带宽 | 实现接近理论峰值的 aggregate bandwidth |
| 缓冲区开销 | 需静态 staging buffer，占用 HBM | 无需 staging buffer，释放更多 HBM 给 KV Cache |
| 并发粒度 | 层级预取（layer-wise），易产生 bubble | 细粒度 tile-level 并行访问，无缝重叠 |
| 性能优化 | 忽视 operation 特性统一卸载 | 分析性地按 operation 动态分配 offload ratio |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集与模型
- **模型**：
  - OPT 系列：OPT-6.7B、OPT-30B
  - Llama 系列：Llama-2-7B、LLaMA-70B
- **任务**：LLM 推理（decoding 和 prefilling）
- **输入长度**：prompt length 从 32 到 1024 不等
- **批大小**（batch size）：8 ~ 512

### ⚙️ 实验平台
| 平台 | GPU | Host-GPU Interconnect | GPU Memory | Host Memory |
|------|-----|------------------------|------------|-------------|
| **GH200**（Grace Hopper） | H100 GPU | NVLink-C2C（单向 900 GB/s） | 96 GB HBM3（4.0 TB/s） | 480 GB |
| **RTX6000 Pro Blackwell** | Blackwell GPU | PCIe Gen5（64 GB/s） | 96 GB GDDR7（1.8 TB/s） | 512 GB |

### 📈 评估指标
- **Effective Bandwidth (EB)**：总数据量 ÷ 单 token 延迟，反映实际达到的内存带宽
- **Time Per Output Token (TPOT)**：端到端解码延迟
- **Throughput**：每秒生成 token 数
- **Global Offload Ratio**：根据模型大小与可用 HBM 自动计算所需卸载比例

### 🆚 基线方法对比
| 基线 | 类型 | 说明 |
|------|------|------|
| **FlexGen** | Layer-wise prefetching | 双缓冲层间预取，经典 offloading 框架 |
| **vLLM-prefetch** | Asynchronous prefetch | 异步预取 KV blocks 和 weights |
| **vLLM-uvm** | UVM-based paging | 使用 NVIDIA Unified Virtual Memory，依赖 page fault 和迁移机制 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### 在 GH200 上的表现（NVLink-C2C）
- **最高达 3× 吞吐提升**（vs. state-of-the-art baselines）
- 在 OPT-30B 上，10% 卸载率即可实现 **3,300 GB/s** 的有效带宽，接近理论极限（HBM + NVLink-C2C）
- 对比 prefetching 方法，在所有 offload ratio 下均显著领先（**1.5× ~ 5× EB 提升**）

#### 在 RTX6000 Blackwell 上的表现（PCIe）
- 最高 **1.8× 性能增益**
- 在低中等 offload ratio（0–40%）下，EB 提升 **1.3× ~ 3×**
- 当 offload ratio > 40%，受限于 PCIe 带宽瓶颈，各方法趋于收敛

> 💡 注：vLLM-UVM 表现最差，因频繁 page fault 和 migration 开销过大。

### 🆚 与基线方法对比结果
| 场景 | DAK vs. FlexGen | DAK vs. vLLM-prefetch |
|------|------------------|------------------------|
| 小 batch（8），仅 weight offload | ↑ 1.5× ~ 5× | ↑ 2× ~ 4× |
| 大 batch（512），weight + KV cache offload | ↑ 2.1× | ↑ 1.8× |
| 极大上下文（1024 seq len） | ↑ 1.83× ~ 21× | — |

> 示例：在 LLaMA-70B、batch=128、seq=1024 场景下，全局 offload ratio 达 60%，DAK 比 FlexGen 快 **21倍**

### 🔬 消融实验结果

#### （1）Greedy Offload Algorithm vs. Uniform Offload
- 在 batch=512（混合 compute/memory-bound ops）场景下：
  - 当 offload ratio < 60%，**greedy 比 uniform 快 1.5×**
  - 超过 60% 后两者趋同（所有操作均受 interconnect 限制）
- 结论：**greedy 算法能精准识别 operation 敏感性并优先卸载 compute-bound 操作**

#### （2）Congestion Control 消融
- 移除拥塞控制后，SM-to-HBM 带宽下降高达 **22%**
- 最高带来 **1.22× 性能下降**
- 原因：过多 inflight TMA 请求引发 interconnect congestion，阻塞本地 HBM 访问

#### （3）TMA Multicast 效果
| 批大小 N | 无 multicast | 启用 multicast | 加速比 |
|---------|--------------|----------------|--------|
| 512     | Baseline     | +1.3×          | 1.3×   |
| 1024    | Baseline     | +2.5×          | 2.5×   |
- 原因：随着 batch 增大，uncacheable host memory 的 **read amplification** 严重恶化，multicast 显著减少冗余传输

#### （4）Kernel Alignment 优化
- 保证 tile 分配均匀，避免尾部延迟（tail latency）
- 最高提升 **1.2× 吞吐**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Direct access 优于 prefetching**  
   绕过 HBM staging 可以真正聚合系统总带宽（local HBM + interconnect），而 prefetching 因 HBM contention 本质上无法达到理论上限。

2. **TMA 可被用于跨层级数据移动**  
   首次将 TMA 用于 **host memory → SMEM** 的 direct access，充分发挥其异步、高并发优势。

3. **per-operation offload ratio 至关重要**  
   统一卸载策略会“饿死”memory-bound 操作或“拖慢”compute-bound 操作；**greedy 算法可分析性地找到最优分配**。

4. **uncacheable host memory 是重大挑战**  
   GPU L2 cache 不缓存 host-homed memory，导致严重 read amplification；必须通过 **TMA multicast + locality-aware scheduling** 解决。

5. **DAK 具备跨架构通用性**  
   在 NVLink-C2C 和 PCIe 系统上均表现优异，验证了设计的普适性。

### ⚠️ 方法的局限性
- **依赖 TMA 硬件支持**：目前仅 Hopper 及后续架构（如 Blackwell）支持 TMA，不适用于旧代 GPU（如 Ampere）。
- **需离线调优参数**：如 congestion window size、最优 SM 分配等需通过 profiling 预先确定。
- **对极低带宽互连收益有限**：当 interconnect 成为绝对瓶颈时（如高 offload ratio + PCIe），性能增益受限。

### 🔮 未来工作方向
- 扩展至 MoE（Mixture-of-Experts）模型的动态专家卸载
- 支持更复杂的 memory hierarchy（如 CXL pools）
- 动态 runtime 调度以适应变化的工作负载
- 探索在训练场景中的应用潜力

---

> ✅ **开源地址**：https://github.com/shouxulin/DirectAccessKernel.git  
> 📌 **一句话总结**：DAK 通过 **direct-access + TMA + greedy offload + multicast** 实现了 LLM 推理中 GPU memory offloading 的效率最优解，在多种架构上实现了高达 **3× 的性能提升**。

</details>

---

### 12. [SpecTr-GBV: Multi-Draft Block Verification Accelerating Speculative Decoding](https://arxiv.org/abs/2604.25925)

**Authors**: Yijun Lin, Jinhao Sheng, Qingyue Cai, Feng Zhou  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.25925v1  

#### Abstract
Autoregressive language models suffer from high inference latency due to their sequential decoding nature. Speculative decoding (SD) mitigates this by employing a lightweight draft model to propose candidate tokens, which are selectively verified by a larger target model. While existing methods eith...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：SpecTr-GBV: Multi-Draft Block Verification Accelerating Speculative Decoding**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLM）在推理阶段采用自回归（autoregressive）生成方式，逐个生成 token，导致**高延迟**。尽管 **Speculative Decoding (SD)** 通过引入轻量级 draft model 预测候选 token 来缓解该问题，但其效率仍受限于两个方面：
- **单草案限制**：标准 SD 仅使用一个 draft 序列，接受率低。
- **逐位置验证（position-by-position verification）**：即使后续 token 更匹配，一旦某个位置拒绝，整个序列即被截断。

现有方法要么改进为 **multi-draft**（如 SpecTr），要么改进为 **block verification**（如 GBV），但二者通常独立处理，未能协同优化。

---

### **提出的新方法：SpecTr-GBV**
本文提出 **SpecTr-GBV**，首次将 **multi-draft** 和 **greedy block verification (GBV)** 统一到一个框架中，实现协同加速。

#### **核心思想**
- 生成 $ K $ 个独立同分布（i.i.d.）的 draft 序列。
- 将验证过程建模为 **draft token blocks** 与 **target token block** 之间的 **Optimal Transport (OT)** 问题。
- 采用 **greedy block verification** 策略，联合验证多个 token 子块，选择最长可接受子块。

#### **算法流程简述**
1. 对每个 draft 序列，从当前最长已接受子块后开始，依次验证更长的子块。
2. 使用公式计算子块接受概率 $ h_{ik} $，决定是否接受。
3. 若某完整 block 被接受，则提前终止并采样下一个 token；否则记录未接受块，继续下一 draft。
4. 最终输出最长接受子块，并从残差分布中采样修正 token。

---

### **相比现有方法的优势**
| 方法 | 机制 | 局限性 | SpecTr-GBV 的改进 |
|------|------|--------|------------------|
| **SD** | 单草案 + 逐 token 验证 | 接受率低，效率有限 | ✅ 多草案 + 块验证 → 更高接受率 |
| **SpecTr** | 多草案 + 逐位置验证 | 未充分利用 block 结构 | ✅ 引入 GBV，提升期望接受长度 |
| **GBV** | 单草案 + 块验证 | 缺乏多草案多样性 | ✅ 扩展至 multi-draft 场景 |

- **理论优势**：证明 SpecTr-GBV 在 i.i.d. draft 生成下达到**最优期望接受长度**，且该上界随 $ K $ 增加而单调递增。
- **计算效率**：验证复杂度为 $ O(|\Omega|) $，优于 SpecTr 的 $ O(|\Omega|\log K) $。
- **分布保真**：通过 residual distribution 和 distribution modification 步骤，保证输出分布与目标模型一致。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共五类任务，覆盖编程、数学、语言建模和指令遵循：
- **HumanEval**：Python 编程问题（Chen et al., 2021）
- **GSM8K**：小学数学应用题（Cobbe et al., 2021）
- **MGSM**：多语言版 GSM8K（Shi et al., 2022）
- **LM1B**：十亿词基准语言建模数据集（Chelba et al., 2013）
- **Alpaca**：斯坦福指令跟随数据集（Taori et al., 2023）

---

### **实验设置**
- **主模型组合**：
  - Target Model：`DeepSeek-33B`, `DeepSeek-6.7B`, `CodeLlama-13B`, `Vicuna-13B`
  - Draft Model：`DeepSeek-1.3B`, `CodeLlama-7B`, `Vicuna-7B`
- **关键超参数**：
  - Draft Length $ L $: 8 或 12
  - Temperature $ T $: 0.4
  - Draft Number $ K $: 3
- **评估指标**：
  - **Block Efficiency (BE)**：每轮串行调用解码的平均 token 数  
    $$
    \text{BE} = \frac{\text{Total decoded tokens}}{\text{Number of serial calls to } M_b}
    $$
  - **Speedup Ratio (SR)**：相对于 autoregressive 推理的墙钟时间加速比  
    $$
    \text{SR} = \frac{T_{\text{autoregressive}}}{T_{\text{proposed}}}
    $$

---

### **基线方法对比**
1. **AR**：标准自回归解码（无 speculative）
2. **SD**：基础 speculative decoding
3. **SpecTr**：基于 optimal transport 的 multi-draft 方法
4. **GBV**：greedy block verification（单草案）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**

| Setting | Method | Avg BE | Avg SR |
|--------|--------|--------|--------|
| DeepSeek-33B / 1.3B | SD | 7.64 | 1.64 |
| | SpecTr | 8.39 | 1.96 |
| | GBV | 7.83 | 1.67 |
| | **SpecTr-GBV** | **8.59** | **2.12** |
| DeepSeek-6.7B / 1.3B | SD | 6.13 | 1.05 |
| | SpecTr | 6.69 | 1.11 |
| | GBV | 6.27 | 1.06 |
| | **SpecTr-GBV** | **6.84** | **1.20** |

#### **性能提升总结**
- 在 DeepSeek-33B 设置下：
  - BE 提升：相对 SD ↑12.4%，相对 SpecTr ↑2.3%，相对 GBV ↑9.7%
  - SR 提升：相对 SD ↑29.3%，相对 SpecTr ↑8.2%，相对 GBV ↑27.0%
- 在 DeepSeek-6.7B 设置下：
  - BE 提升：↑11.6% (vs SD)，↑2.2% (vs SpecTr)，↑9.1% (vs GBV)
  - SR 提升：↑14.3% (vs SD)，↑8.1% (vs SpecTr)，↑13.2% (vs GBV)

> ✅ **SpecTr-GBV 在所有数据集和模型组合上均取得最优 BE 和 SR**

---

### **消融实验结果**

#### **(1) Draft Length $ L $ 的影响（Table 3 & 5）**
- BE 随 $ L $ 增加持续上升（更多候选 token）
- SR 先升后降：过长的 draft 导致 draft model 开销过大，抵消收益
- **结论**：存在最优 $ L $ 平衡 draft 成本与 target 调用节省

#### **(2) Draft Number $ K $ 的影响（Figure 2a & 3a）**
- 接受率随 $ K $ 增加而提高（理论支持）
- SpecTr-GBV 在所有 $ K $ 下均优于 SpecTr
- 差距随 $ K $ 增大而扩大 → 表明其对多草案更具扩展性

#### **(3) Temperature $ T $ 的影响（Figure 2b & 3b）**
- BE 和 SR 在不同 $ T $（0.1, 0.4, 0.7）下保持稳定
- **结论**：SpecTr-GBV 对温度变化鲁棒性强

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **理论最优性**：SpecTr-GBV 在 i.i.d. draft 框架下实现了**物理可达的最优期望接受长度**。
2. ✅ **协同增益**：multi-draft 与 block verification 的结合不是简单叠加，而是产生**协同效应**，显著超越任一单独策略。
3. ✅ **高效实现**：算法复杂度 $ O(|\Omega|) $，验证开销比 SpecTr **降低超过 50%**（见 Table 2）。
4. ✅ **广泛适用性**：在多种 LLM 架构（DeepSeek, CodeLlama, Vicuna）和任务上均表现一致优越。

---

### **方法的局限性**
- **依赖高质量 draft model**：若 draft model 与 target 差异过大，多草案策略增益下降。
- **内存开销增加**：需存储 $ K $ 个 draft 序列及其评分，在极端 $ K $ 下可能成为瓶颈。
- **实现复杂度高于 SD/GBV**：需维护 block 接受状态和 distribution modification 逻辑。

---

### **未来工作方向**
1. **动态调整 $ K $ 和 $ L $**：根据上下文难度自适应选择草案数量和长度。
2. **非 i.i.d. 草案生成**：探索树状或多路径草案结构以进一步提升多样性。
3. **硬件感知优化**：结合 GPU 并行能力，优化 batched scoring 和 memory layout。
4. **与其他加速技术融合**：如与 **Lookahead Decoding** 或 **Distilled Draft Models** 结合。

---

> **总结**：SpecTr-GBV 是首个将 **multi-draft** 与 **block verification** 统一的 speculative decoding 框架，在理论最优性和实际性能上均取得突破，为 LLM 高效推理提供了新的范式。

</details>

---

### 13. [Shorthand for Thought: Compressing LLM Reasoning via Entropy-Guided Supertokens](https://arxiv.org/abs/2604.26355)

**Authors**: Zhenyu Zhao, Sander Land, Dan Bikel, Waseem Alshikh  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26355v1  

#### Abstract
Reasoning in Large Language Models incurs significant inference-time compute, yet the token-level information structure of reasoning traces remains underexplored. We observe that reasoning tokens split into two functional types: low-entropy \textit{structural} tokens (recurring phrases that scaffold...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Shorthand for Thought: Compressing LLM Reasoning via Entropy-Guided Supertokens*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
大型语言模型（LLM）在执行 **Chain-of-Thought (CoT)** 推理时会产生大量冗长的中间推理 token，导致显著的 **inference-time compute 开销**。尽管已有多种压缩方法，但大多通过删减内容（pruning）、抽象化（abstraction）或引入隐空间（latent space）来实现，牺牲了推理过程的可读性和可解释性。

本文指出：**推理 token 并非同质流，而是由两类功能不同的 token 构成**：
- **Structural tokens**：低熵、模式化的“骨架”短语（如 “Let’s verify”, “Wait, hold on”），组织推理流程但信息量低。
- **Organic tokens**：高熵、问题相关的“有机”内容（如数学表达式、变量绑定），驱动实际求解。

这种结构性冗余提示了一个新的压缩机会：**不删内容，而是在词汇层面合并高频结构短语为 supertokens**。

---

### 🚀 提出的新方法与创新思路
提出一种 **model-agnostic 的推理压缩 pipeline**，基于信息论指导下的 **supertoken 学习**：

1. **Entropy-Guided Supertoken Discovery**  
   在模型自身的 CoT 推理轨迹上应用 **cross-word BPE**（来自 SuperBPE），提取跨词边界、高频且低熵的多 token 模式作为 **supertoken**，并添加到原始 tokenizer 的词汇表中。

2. **轻量级微调（SFT）适配**  
   仅对 embedding 层、LM Head 和少量 transformer 层进行 **supervised fine-tuning (SFT)**，教会模型使用这些 supertokens，无需重新预训练。

3. **保留完整可读性**  
   与内容级压缩不同，该方法 **不删除任何推理步骤**，仅改变其 tokenization 方式，因此输出仍为人类可读的完整推理链。

---

### ⚖️ 相比现有方法的优势

| 方法类型 | 代表工作 | 压缩率 | 是否保留内容 | 可解释性 |
|--------|--------|-------|-------------|---------|
| **Content-level** | TokenSkip, ConMax, CtrlCoT | 30–73% | ❌ 删减/修改步骤 | ❌ |
| **Latent-level** | Coconut, Token Assorted | 高 | ❌ 替换为向量 | ❌ |
| **Vocabulary-level (本文)** | **Ours** | **~8.1%** | ✅ 完整保留 | ✅ |

> ✅ **正交性**：本方法与上述方法兼容，可在其基础上进一步压缩。

> ✅ **额外价值**：supertokens 天然成为 **可解释的推理动作标注（reasoning-move annotations）**，可用于分析模型策略。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

- **训练数据**：`OpenThoughts3`（Guha et al., 2025）上的 CoT 推理轨迹，用于提取 supertokens 和 SFT 微调。
- **评估数据集（5个数学推理基准）**：
  - `AIME 2024`, `AIME 2025`
  - `MATH-500` (Hendrycks et al., 2021)
  - `Minerva`
  - `OlympiadBench` (He et al., 2024)

所有评估均为 **zero-shot setting**。

---

### 🔧 实验设置

- **目标模型族（3个）**：
  - QwQ-32B
  - Qwen3-30B-A3B
  - DeepSeek-R1-Llama-70B-Distill

- **Supertoken 提取**：
  - 在各模型自己的推理轨迹上运行 **SuperBPE**。
  - 应用 **structural filter** 过滤无意义合并（如 “is the”, “in the”），保留如 “Let’s check”, “Wait, hold on” 等有意义推理短语。
  - 保留 top-250 最频繁 merges 作为 supertokens。

- **微调方式（SFT）**：
  - 三种策略对比：
    1. **Embedding-only**：只更新新 supertoken 的 embedding 和 LM head。
    2. **LoRA**（r=16, α=32）
    3. **Partial layer unfreezing**：解冻首尾若干层 + embedding。

- **评估指标**：
  - **平均推理 token 数（↓）**
  - **准确率（Acc%）**
  - **supertoken 采用率（%）**
  - **wall-clock latency（秒/样本）**

---

### 🆚 基线方法对比
在 Table 1 中与以下方法对比：

| 方法 | 类型 | 压缩率 | 准确率影响 | 可解释性 |
|------|------|--------|------------|----------|
| TokenSkip | Step pruning | 30–40% | <4% drop | ❌ |
| R1-Compress | Chunk compression | ~20% | 0.6% drop | ❌ |
| ConMax | RL compression | 43% | 0.7% drop | ❌ |
| Coconut | Latent space | 高 | varies | ❌ |
| **Ours** | **Vocabulary-level** | **8.1%** | **~0pp** | ✅ |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（见 Table 2）

| 模型 | 平均压缩率 | 平均准确率变化 |
|------|-----------|----------------|
| QwQ-32B | -6.4% | +1.3pp |
| Qwen3-30B-A3B | -6.3% | -1.3pp |
| DeepSeek-R1-70B | -11.5% | -0.8pp |

> ✅ **总体平均压缩率：8.1%**  
> ✅ **所有准确率变化均在 95% 置信区间内不显著（即无统计显著损失）**

- **最大压缩**：DeepSeek-R1 在 AIME 上达 **-17%**，且显著（p<0.05）。
- **延迟降低**：wall-clock latency 平均下降 **4.4–14.1%**，部分场景超过 token 减少比例（Table 9）。

---

### 🔍 消融实验结果

#### （1）不同 SFT 策略对比（Table 7）

| 方法 | Supertoken 采用 | 压缩效果 | 准确率 |
|------|------------------|----------|--------|
| Embedding-only | 有（~7–10%） | ❌ 无压缩 | 下降明显 |
| LoRA | 有（~10–13%） | ❌ 无压缩 | 保持良好 |
| **Partial unfreezing** | 有 | ✅ 显著压缩 | ✅ 保持 |

> 💡 结论：**仅学习使用 supertokens ≠ 实现压缩**；需要一定模型灵活性才能调整生成长度。

#### （2）LoRA Rank Ablation（Table 8）
- 不同 rank（1–16）下 supertoken 采用率稳定（~10–13%），但 **均未带来压缩**，再次验证上述结论。

#### （3）Cross-Model Entropy Validation（Table 10）
- 使用 Qwen3-30B 对 QwQ-32B 的推理轨迹重打分，发现：
  - **continuation tokens 仍是最低熵**（1.36 < 1.98）
  - **structural/organic 差距反而扩大**（0.62 > 0.26）
> ✅ 表明结构性低熵是文本本身的规律，而非模型自洽幻觉。

---

## 4. 关键结论和发现

### 🎯 主要发现

1. **推理 token 具有双峰结构**：
   - Structural tokens 低熵、可预测，适合压缩；
   - Organic tokens 高熵、承载核心信息，应保留。

2. **Supertokens 是有效的压缩与解释工具**：
   - 实现 **8.1% 平均 token 压缩**，**无显著准确率损失**。
   - supertokens 可分类为 **9类 reasoning moves**（如 Backtracking, Verification, Strategy Shift），形成“推理签名”。

3. **推理质量可通过 supertoken 转移模式诊断**：
   - **正确推理**：呈现 **productive recovery** 模式：
     - Problem Ref. → Strategy Shift (**3.0×**)
     - Verification → Strategy Shift (**2.1×**)
   - **错误推理**：陷入 **confusion cycles**：
     - Problem Ref. → Hedging (**2.1×**)
     - Contradiction → Re-read problem (**2.0×**)

4. **提出三个诊断指标**：
   - **Productive Recovery Rate**：正确轨迹高 34%
   - **Confusion Cycle Rate**：错误轨迹高 50%
   - **Verification Inflow Rate**：正确轨迹高 14%

> 这些信号可用于 **reward shaping** 或 **early stopping**，提升 RL-based reasoning 效率。

---

### ⚠️ 方法的局限性

1. **压缩率相对较低（~8.1%）**  
   相比 content-level 方法（30–70%）显得保守 —— 但这是**有意设计**：以适度压缩换取 **完全保留可读性与可解释性**。

2. **依赖 tokenizer 兼容性**  
   在 Llama-based tokenizer 上集成更困难（如 DeepSeek-R1），可能导致更大扰动。

3. **当前仅用于数学推理**  
   尚未验证在代码、规划等其他 CoT 场景中的泛化能力。

---

### 🔮 未来工作方向

1. **集成至 RL 训练框架**（如 GRPO）：
   - 使用 **confusion cycle detection** 触发 early stopping。
   - 使用 **productive recovery rate** 作为 reward shaping 信号。

2. **构建两层推理架构（Two-Tier Reasoning）**：
   - **显式结构层**：由 supertoken 控制流程（backtrack, verify, pivot）。
   - **隐式内容层**：autoregressive 生成具体计算与逻辑。
   > 实现更可控、高效的推理。

3. **跨任务、跨模型通用 supertoken 设计**  
   探索是否可构建通用的“推理语法”supertoken 集。

4. **实时监控与干预系统**  
   在推理过程中检测不良模式（如连续 hedging），动态注入 prompt 引导恢复。

---

## 总结

✅ 本文提出了一种 **简单、通用、无损** 的 LLM 推理压缩方法：  
通过 **entropy-guided supertokens** 合并低熵结构短语，在 **不删内容** 的前提下实现 **8.1% token 压缩** 与 **推理过程可视化**。

🧠 更重要的是，它揭示了：**CoT 不是自由文本，而是一种具有“语法”的推理行为**。  
supertokens 不仅是压缩工具，更是打开黑箱、理解模型“思维策略”的 **可解释性窗口**。

</details>

---

### 14. [MoRFI: Monotonic Sparse Autoencoder Feature Identification](https://arxiv.org/abs/2604.26866)

**Authors**: Dimitris Dimakopoulos, Shay B. Cohen, Ioannis Konstas  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26866v1  

#### Abstract
Large language models (LLMs) acquire most of their factual knowledge during the pre-training stage, through next token prediction. Subsequent stages of post-training often introduce new facts outwith the parametric knowledge, giving rise to hallucinations. While it has been demonstrated that supervi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# MoRFI: Monotonic Sparse Autoencoder Feature Identification 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文研究了**大型语言模型（LLMs）在监督微调（SFT）过程中因引入未知事实而导致幻觉（hallucinations）增加的现象**。尽管已有研究表明微调新知识会加剧幻觉，但其内部机制尚不明确。本文旨在从**激活空间的角度揭示这一现象的因果机制**，并识别出导致知识遗忘或访问失败的关键 latent 方向。

### 提出了什么新方法或新思路
作者提出了 **MoRFI（Monotonic Relationship Feature Identification）**，一种用于从稀疏自编码器（Sparse Autoencoders, SAEs）中识别因果相关特征的新算法。其核心思想是：
- 不依赖传统的“模型差分”（model diffing）方法（即比较两个固定状态），而是通过**控制微调条件的梯度变化**（如未知知识比例、训练轮数），追踪 latent 激活的**单调趋势**。
- 利用**引导采样（bootstrapping）与统计检验**（Spearman 秩相关 + Mann-Kendall 趋势检验）来筛选出具有稳健单调响应的 SAE 特征。

### 相比现有方法的优势
| 维度 | MoRFI 的优势 |
|------|--------------|
| **分析粒度** | 超越静态对比，捕捉连续变化过程中的动态趋势 |
| **抗干扰能力** | 控制数据主题多样性，避免窄域微调带来的虚假激活痕迹（narrow fine-tuning artifacts） |
| **因果性更强** | 结合 activation steering 验证干预效果，确保所选 latents 具有因果影响力 |
| **通用性** | 可应用于任何可控变量诱导的模型检查点序列 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主数据集**：`EntityQuestions`，基于 Wikidata 构建的闭卷问答（closed-book QA）数据集，包含多个关系类型（如 P17 国家、P36 首都等）。
- **知识划分标准**：使用 `Pcorrect(q,a;M,T)` 指标判断一个问题是否为模型已知（Known）或未知（Unknown）。该指标基于 base model 在 few-shot 设置下的准确率进行分类。
- **构造微调混合数据集**：对训练集按不同比例（0%, 10%, 25%, ..., 100%）混合未知样本，共构建 7 个不同的 fine-tuning 数据集 $ D_p $。

### 实验设置
- **模型**：Llama 3.1 8B、Gemma 2 9B、Mistral 7B v03 三种主流开源模型。
- **SAE 来源**：
  - Llama Scope（Llama 3.1）
  - Gemma Scope（Gemma 2）
  - Engels et al. (2025)（Mistral）
- **微调配置**：
  - 微调轮数：10–50 轮，另设早停组（early stop）
  - 每个模型 × 每种混合比例 → 35 次独立微调运行
- **目标维度**：
  - **个Unknown**：未知样本占比的变化
  - **个Epochs**：训练轮数的增长

### 评估指标
- **下游任务性能**：在测试集上的准确率（accuracy）
- **latents 影响力评估**：通过 **activation steering** 干预残差流（residual stream），观察 dev set 准确率提升
- **知识恢复率（Knowledge Recovery Rate, $ R_K $）**：
  $$
  R_K = P(M_{D0}=1 \mid M_{D100s}=1, M_{D100}=0)
  $$
  表示被干预后“找回”的答案中，原本属于 base model 已知知识的比例。

### 基线方法对比
- **Composite Direction（$ \delta_u $）**：使用 known vs unknown 微调模型之间的平均激活差异作为整体 steering 方向。
- **Control Group（非趋势 latents）**：选取在微调过程中无显著变化的 latents 进行 steering，验证 MoRFI 所选 latents 的特殊性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 最佳单 latent 干预增益（Δacc） | 知识恢复率 $ R_K $ |
|------|-------------------------------|---------------------|
| **Llama 3.1 8B** | +33.4（neg. steering） | 69.0% – 76.9% |
| **Gemma 2 9B** | +22.6（neg. steering） | 76.1% – 85.1% |
| **Mistral 7B v03** | +39.77（neg. steering） | 77.1% – 83.8% |

> 注：所有增益均相对于仅在未知数据上微调的最差模型（MD100）而言。

### 与基线方法的对比结果
- **MoRFI 单 latent > Composite Direction $ \delta_u $**  
  尽管 $ \delta_u $ 投影能带来一定提升，但 MoRFI 发现的单个 latents 效果更优（见 Figure 4），说明知识相关的信号是**稀疏分布**的，复合方向中包含噪声或抵消成分。
  
- **MoRFI latents >> Control Group**  
  对照组 latents 的 steering 几乎无效甚至有害，证明 MoRFI 所选特征并非随机扰动的结果（见 Figure 3 和附录图）。

### 消融实验与关键发现
#### （1）Steering 极性不对称性
- **负向 steering（抑制增长的 latents）效果远优于正向 steering（增强下降的 latents）**
  - Llama 上最佳负向增益：+33.4 vs 正向 +21.7
  - Mistral 上：+39.77 vs +24.91
  - 表明：**过度表达某些 latents 是导致知识访问失败的关键机制**

#### （2）下降型 latents 更具潜力
- 那些随未知知识增加而**激活减弱**的 latents，在被正向 steering 后也能显著恢复性能。
- 表明这些 latents 仍编码着原始知识路径，只是被“关闭”而非删除。

#### （3）跨模型一致性**
- 多个重要 latents（如 27191, 30382, 2378）在不同模型中均有出现，且功能相似（如地理实体识别）。
- 支持了 LLMs 中存在**通用的知识访问电路**的可能性。

#### （4）知识恢复集中在特定语义类别**
- 恢复最多的知识集中于地理类关系（P17 国家、P36 首都、P495 原产国），符合预期——这些概念共享底层表示结构。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **引入未知知识会破坏模型访问已有参数化知识的能力**，表现为特定 latent 方向的异常激活。
2. ✅ **MoRFI 成功识别出一组因果相关的 SAE latents**，这些 latents 的激活强度与未知知识比例呈单调关系。
3. ✅ **单 latent 干预即可大幅恢复性能**（最高达 +39.77 pts），且恢复的答案中 **69–85% 属于原模型已知知识**，表明“遗忘”实为“访问失败”，而非“知识擦除”。
4. ✅ **负向 steering（抑制）比正向 steering 更有效**，暗示 fine-tuning 引入了错误的“门控”机制，压制了正确路径。
5. ✅ **知识相关信号高度稀疏**，复合方向 $ \delta_u $ 不如精选的单个 latents 有效。

### 方法的局限性
- **依赖高质量预训练 SAEs**：若 SAE 未能解耦出语义清晰的 features，则 MoRFI 效果受限。
- **局限于中间层分析**：本研究只提取 middle layer 的激活，未探索多层协同机制。
- **静态干预策略**：steering 使用固定 scalar 强度，未考虑输入依赖的动态调整。
- **计算成本高**：需大量微调运行以构建平滑梯度，不适合大规模部署。

### 未来工作方向
- 🔄 **扩展至其他属性维度**：如毒性、偏见、逻辑推理能力退化等，构建通用的“行为归因框架”。
- 🔍 **多层联合分析**：研究不同网络深度中 latents 的交互模式。
- ⚙️ **开发动态 steering 控制器**：根据输入自动选择最优 latents 与强度。
- 🧭 **几何解释探索**：验证文中提出的“知识可及状态流形”假设，研究 latents 是否沿曲面导航。
- 🤝 **结合 RLHF/SFT 安全训练**：利用 MoRFI 发现的风险 latents 设计防御性微调策略。

---

> **总结一句话**：  
> MoRFI 揭示了 LLM 微调中“知识遗忘”的本质是**访问路径被干扰而非内容被覆盖**，并通过识别稀疏的因果 latents 实现了高效的 inference-time 知识恢复，为理解与缓解幻觉提供了新的机制性视角。

</details>

---

### 15. [SplitFT: An Adaptive Federated Split Learning System For LLMs Fine-Tuning](https://arxiv.org/abs/2604.26388)

**Authors**: Yimeng Shan, Zhaorui Zhang, Sheng Di, Yu Liu, Xiaoyi Lu, Benben Liu  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26388v1  

#### Abstract
Federated Split Learning has been identified as an efficient approach to address the computational resource constraints of clients in classical federated learning, while guaranteeing data privacy for distributed model training across data owners. However, it faces some critical challenges when such ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SplitFT: An Adaptive Federated Split Learning System For LLMs Fine-Tuning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前在 **Federated Learning (FL)** 中对 **Large Language Models (LLMs)** 进行微调时面临三大挑战：
1. **设备异构性**（Device Heterogeneity）：不同客户端计算资源差异大，固定模型切分（cutlayer）导致高性能设备需等待低性能设备，影响整体效率。
2. **数据异构性**（Data Heterogeneity）：各客户端数据分布非独立同分布（Non-IID），影响全局模型收敛和性能。
3. **通信开销高**：客户端需频繁传输中间“smashed data”，尤其在微调 LLMs 时通信负担严重。

现有方法未能有效解决上述问题，尤其是在结合 **Split Learning** 和 **Parameter-Efficient Fine-Tuning (PEFT)** 场景下的自适应优化。

---

### 🚀 提出的新方法与创新思路

作者提出 **SplitFT** —— 一种面向 LLMs 微调的**自适应联邦分割学习系统**，其核心创新包括：

#### （1）**自适应切层策略（Adaptive Cutlayer Allocation）**
- 允许每个客户端根据自身 **计算资源** 和 **本地模型性能** 动态调整其负责的模型层数（即 `lc,i`）。
- 引入动态权重机制：表现优于平均准确率的客户端承担更多训练任务，反之则减轻负载。
- 优化目标为平衡客户端计算负载与全局模型性能。

#### （2）**降低 Cutlayer 的 LoRA Rank 以减少通信开销**
- 在 cutlayer（客户端最后一层 / 服务器第一层）上使用更小的 LoRA rank（如 `rcut=8`），而在其他层保持较高 rank（如 `rothers=16`）。
- 显著减小 smashed data 和梯度的尺寸，从而降低通信成本，同时不影响整体模型质量。

#### （3）**基于长度的 Dirichlet 数据划分方法（Length-based Dirichlet Partitioning）**
- 针对 LLMs 微调场景设计新型 Non-IID 划分方式：
  - 先按输入序列长度将数据分为 K 类；
  - 再通过 Dirichlet 分布控制每类数据在客户端间的分配比例。
- 超参数 α 控制异构程度：α 越小，数据越偏斜（高度 Non-IID）；α 越大越接近 IID。

#### （4）系统模块化设计
- 支持灵活集成不同算法、采样策略、模型结构和 PEFT 方法。
- 包含 Main Server、Local FedAvg Server、Client 等组件，职责分明。

---

### ⚖️ 相比现有方法的优势

| 维度 | SplitFT 优势 |
|------|-------------|
| **灵活性** | 支持不同客户端设置不同的 cutlayer，适应设备异构性 |
| **效率** | 减少通信量（via 低秩 cutlayer）、提升训练速度 |
| **鲁棒性** | 在高度 Non-IID 下仍能稳定收敛，避免过拟合 |
| **隐私保护** | 客户端不掌握完整模型，仅交换中间表示，增强安全性 |
| **通用性** | 可扩展至多种 LLM 架构（GPT、OPT、Neo 等） |

> ❗ 本文是首个将 **adaptive layer splitting + LoRA rank 调整 + Non-IID 模拟** 结合用于联邦 Split Learning 微调 LLMs 的工作。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **Wikitext2-v1**：广泛使用的语言建模基准数据集。
  - 训练样本：36,700
  - 验证样本：3,760
  - 测试样本：4,360
  - 内容来自维基百科文章，适合 next-sentence prediction 任务。

---

### 🔧 实验设置

| 参数 | 设置 |
|------|------|
| **基础模型** | GPT2-small（12 layers）、OPT-125M、GPT-Neo 125M |
| **客户端数量** | 5 |
| **本地数据划分** | 每个客户端 12,000 个样本 |
| **Batch Size** | 4 |
| **Learning Rate** | $5 \times 10^{-5}$ |
| **最大序列长度** | 512 tokens |
| **LoRA 配置** | Attention 模块启用 LoRA；cutlayer rank 设为 8，其余设为 16 |
| **Cutlayer 位置** | 默认第 2 层，对比测试 2~10 层 |
| **训练轮数** | 最多 1200 global rounds |
| **硬件平台** | NVIDIA GeForce RTX 3090 GPU × PyTorch 2.3 |
| **框架实现** | 基于 PyTorch + Flower 构建 |

---

### 🎯 评估指标

| 指标 | 描述 |
|------|------|
| **Perplexity** | 主要评价指标，衡量语言模型预测能力，值越低越好 |
| **Max Accuracy / Max Perplexity Drop** | 衡量最终模型性能 |
| **Elapsed Time / Round Time** | 衡量训练效率 |
| **Communication Overhead (MB)** | 估计传输的梯度与激活值总量 |
| **Trainable Parameters** | 参数效率分析 |

---

### 🆚 基线方法对比

- **Baseline（Same Split）**：
  - 所有客户端固定使用前 2 个 GPT2Blocks（共12层中的前2层）作为 client-side；
  - 服务器处理剩余 10 层；
  - LoRA 设置相同（cutlayer rank=8, others=16）；
  - 使用 FedAvg 聚合。

> SplitFT 在相同配置下引入 **adaptive layer adjustment** 和 **rank reduction at cutlayer**，进行公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ 表格 I：不同 Cutlayer 位置的影响（GPT2-small）

| Cutlayer | Max Perplexity ↓ | Mean Elapsed Time (s) ↓ | Max Comm Overhead (MB) ↑ |
|---------|------------------|----------------------------|----------------------------|
| 2       | 0.0606           | 810.4                        | 3475.4                     |
| 4       | 0.0571           | 863.2                        | 3534.4                     |
| 6       | 0.0605           | 934.2                        | 3593.4                     |
| 8       | 0.0621           | 1113.4                       | 3652.4                     |
| 10      | **0.0629**       | 1104.7                       | **3711.4**                 |
| No Cut  | 0.0617           | **6897.3**                   | 3490.1                     |

> 💡 发现：较早的 cutlayer（如第2层）虽增加通信开销，但显著降低训练时间；而完全不分割（No Cut）耗时极高，不可行。

#### ✅ 表格 II：不同 LoRA Rank 的影响

| Rank | Max Acc. ↓ | Elapsed Time (s) ↓ | Comm Overhead (MB) ↓ | Trainable Params |
|------|------------|---------------------|------------------------|------------------|
| 1    | 0.0579     | 2277.5              | 3462.5                 | 0.08M            |
| 2    | 0.0585     | 2138.2              | 3464.3                 | 0.015M           |
| 4    | 0.0589     | 2293.4              | 3468.0                 | 0.031M           |
| 8    | **0.0606** | **1597.7**          | **3475.4**             | 0.062M           |

> 💡 发现：适当提高 rank 可加快收敛，但 `rcut=8`, `rothers=16` 是性能与效率的最佳折中。

---

### 📈 性能对比结果（vs Baseline）

#### 图 3(a)：Adaptive SplitFT vs Same Split（GPT2-small）
- **SplitFT（adaptive）**：
  - 收敛更平稳，未出现明显过拟合；
  - 最终 perplexity 更低（约下降 15%-20%）；
  - 尽管初期稍慢，但长期表现更优。
- **Baseline（fixed cutlayer）**：
  - 初期快速下降，但很快陷入过拟合；
  - 在 IID 和 Non-IID 下均劣于 SplitFT。

#### 图 3(b)(c)：不同 α 下的 Non-IID 表现
- 当 α = 0.1（高度 Non-IID）时，SplitFT 依然保持良好收敛；
- 相比之下，Baseline 在 Non-IID 下性能急剧下降；
- SplitFT 的动态调整机制有效缓解了梯度冲突。

#### 图 4：跨模型泛化能力验证（OPT-125M & GPT-Neo）
- 在三种主流 LLM 上均取得一致优越表现；
- 不论是 IID 还是 Non-IID 设置，SplitFT 均能更快收敛且达到更低 perplexity；
- 证明其具有良好的 **model generalizability**。

---

### 🔍 消融实验结果（Ablation Study）

| 配置 | Perplexity | 说明 |
|------|-----------|------|
| Full LoRA (rank=16 everywhere) | 0.0617 | 基准线，通信开销大 |
| One-side cutlayer rank↓ (only client) | ~0.0610 | 效果有限 |
| Two-side cutlayer rank↓ (`rcut=8`) | **0.0585** | 对称降秩效果最佳 |
| Fixed cutlayer (no adaptation) | 0.0606 | 不如 adaptive 版本 |
| Adaptive + Two-side LoRA ↓ | **0.0571** | 组合最优 |

> ✅ 结论：**同时采用 adaptive layer allocation 和 symmetric LoRA rank reduction** 是性能提升的关键。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **自适应切层显著提升性能与效率**  
   - 动态分配客户端训练层数可有效应对设备异构性和数据异构性；
   - 高性能客户端承担更多任务，加速整体收敛。

2. **降低 cutlayer 的 LoRA rank 可大幅减少通信开销而不牺牲性能**  
   - 对称地在 client 和 server 两侧同时降低 cutlayer 的 LoRA rank 效果最好；
   - 是实现高效通信的关键技术。

3. **SplitFT 在高度 Non-IID 场景下仍具强鲁棒性**  
   - 即使 α=0.1（极端偏斜分布），模型也能稳定训练；
   - 动态调整机制有助于缓解局部过拟合和梯度冲突。

4. **方法具备良好泛化性**  
   - 在 GPT2、OPT、GPT-Neo 多种架构上均有效；
   - 支持不同 LoRA rank 和 cutlayer 配置，适用性强。

---

### ⚠️ 局限性

1. **依赖 LoRA 假设**：目前仅支持 LoRA 类型的 PEFT，尚未扩展到 Prefix Tuning 或 Adapter Tuning。
2. **超参数敏感性**：Dirichlet 参数 α 和 layer adjustment 权重 γ 需要调优。
3. **未考虑极端资源受限设备**：假设客户端至少能运行若干 Transformer 层，无法覆盖极轻量终端（如 IoT）。
4. **缺乏真实世界部署测试**：实验基于模拟环境，实际网络延迟和带宽波动未充分建模。

---

### 🔮 未来工作方向

1. **支持更多 PEFT 方法**：集成 Prefix Tuning、Adapter、DoRA 等新型微调策略。
2. **自动化超参数调节**：引入强化学习或贝叶斯优化自动选择最优 cutlayer 和 LoRA rank。
3. **边缘设备适配**：进一步压缩 client-side 模型，支持手机、嵌入式设备等场景。
4. **跨模态扩展**：探索图像、语音等多模态任务中的联邦 Split Learning 应用。
5. **安全与攻击防御研究**：分析 SplitFT 是否易受模型反演、成员推断等隐私攻击，并提出防护机制。

---

## ✅ 总结

SplitFT 是一项针对 **LLMs 联邦微调** 的重要进展，首次系统性解决了 **设备异构、数据异构、通信瓶颈** 三大难题。通过 **自适应切层 + LoRA 降秩 + 新型 Non-IID 划分**，实现了更高效率、更强鲁棒性和更好性能，在多个主流 LLM 上验证了其优越性。该工作为未来在医疗、金融等隐私敏感领域部署个性化大模型提供了可行路径。

</details>

---

### 16. [MPI Malleability Validation under Replayed Real-World HPC Conditions](https://arxiv.org/abs/2604.26576)

**Authors**: S. Iserte, M. Madon, G. Da, J. Pierson, A. J. Pe\~na  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26576v1  

#### Abstract
Dynamic Resource Management (DRM) techniques can be leveraged to maximize throughput and resource utilization in computational clusters. Although DRM has been extensively studied through analytical workloads and simulations, skepticism persists among end administrators and users regarding their feas...

---

### 17. [Hierarchical adaptive control for real-time dynamic inference at the edge](https://arxiv.org/abs/2604.26470)

**Authors**: Francesco Daghero, Mahyar Tourchi Moghaddam, Mikkel Baun Kj{\ae}rgaard  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26470v1  

#### Abstract
Industrial systems increasingly depend on Machine Learning (ML), and operate on heterogeneous nodes that must satisfy tight latency, energy, and memory constraints. Dynamic ML models, which reconfigure their computational footprint at runtime, promise high energy efficiency and lower average latency...

---

### 18. [Hierarchical Multi-Persona Induction from User Behavioral Logs: Learning Evidence-Grounded and Truthful Personas](https://arxiv.org/abs/2604.26120)

**Authors**: Nayoung Choi, Haeyu Jeong, Changbong Kim, Hongjun Lim, Jinho D. Choi  
**Category**: cs.AI  
**Published**: 2026-04-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.26120v1  

#### Abstract
Behavioral logs provide rich signals for user modeling, but are noisy and interleaved across diverse intents. Recent work uses LLMs to generate interpretable natural-language personas from user logs, yet evaluation often emphasizes downstream utility, providing limited assurance of persona quality i...

---

### 19. [FlowBot: Inducing LLM Workflows with Bilevel Optimization and Textual Gradients](https://arxiv.org/abs/2604.26258)

**Authors**: Hongyeon Yu, Young-Bum Kim, Yoon Kim  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.26258v1  

#### Abstract
LLM workflows, which coordinate structured calls to individual LLMs (each augmented with varying instructions and tools) to achieve a particular goal, offer a promising path towards extending the capabilities of LLMs and building powerful systems that can tackle diverse tasks. However, existing appr...

---

### 20. [HealthNLP_Retrievers at ArchEHR-QA 2026: Cascaded LLM Pipeline for Grounded Clinical Question Answering](https://arxiv.org/abs/2604.26880)

**Authors**: Md Biplob Hosen, Md Alomgeer Hussein, Md Akmol Masud, Omar Faruque, Tera L Reynolds, Lujie Karen Chen  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.26880v1  

#### Abstract
Patient portals now give individuals direct access to their electronic health records (EHRs), yet access alone does not ensure patients understand or act on the complex clinical information contained in these records. The ArchEHR-QA 2026 shared task addresses this challenge by focusing on grounded q...

---

### 21. [Select to Think: Unlocking SLM Potential with Local Sufficiency](https://arxiv.org/abs/2604.26940)

**Authors**: Wenxuan Ye, Yangyang Zhang, Xueli An, Georg Carle, Yunpu Ma  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.26940v1  

#### Abstract
Small language models (SLMs) offer computational efficiency for scalable deployment, yet they often fall short of the reasoning power exhibited by their larger counterparts (LLMs). To mitigate this gap, current approaches invoke an LLM to generate tokens at points of reasoning divergence, but these ...

---

### 22. [DMRlib: Easy-coding and Efficient Resource Management for Job Malleability](https://arxiv.org/abs/2604.26624)

**Authors**: Sergio Iserte, Rafael Mayo, Enrique S. Quintana-Ort\'i, Antonio J. Pe\~na  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.26624v1  

#### Abstract
Process malleability has proved to have a highly positive impact on the resource utilization and global productivity in data centers compared with the conventional static resource allocation policy. However, the non-negligible additional development effort this solution imposes has constrained its a...

---

### 23. [Efficient and Interpretable Transformer for Counterfactual Fairness](https://arxiv.org/abs/2604.26188)

**Authors**: Panyi Dong, Zhiyu Quan  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.26188v1  

#### Abstract
The growing reliance of machine learning models in high-stakes, highly regulated domains such as finance and insurance has created a growing tension between predictive performance, interpretability, and regulatory fairness requirements. In these settings, models are expected not only to deliver reli...

---

### 24. [CoQuant: Joint Weight-Activation Subspace Projection for Mixed-Precision LLMs](https://arxiv.org/abs/2604.26378)

**Authors**: Zhe Ding, Su Pan, Duowei Pan  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.26378v1  

#### Abstract
Post-training quantization (PTQ) has become an important technique for reducing the inference cost of Large Language Models (LLMs). While recent mixed-precision methods improve ultra-low bit quantization by preserving critical subspaces in high precision, they typically construct these subspaces rel...

---

### 25. [PAINT: Partial-Solution Adaptive Interpolated Training for Self-Distilled Reasoners](https://arxiv.org/abs/2604.26573)

**Authors**: Zhiquan Tan, Yinrong Hong  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2604.26573v1  

#### Abstract
Improving large language model (LLM) reasoning requires supervision that is both aligned with the model's own test-time states and informative at the token level. Reinforcement learning with verifiable rewards provides on-policy exploration but offers sparse, high-variance credit; supervised fine-tu...

---

### 26. [Distill-Belief: Closed-Loop Inverse Source Localization and Characterization in Physical Fields](https://arxiv.org/abs/2604.26095)

**Authors**: Yiwei Shi, Zixing Song, Mengyue Yang, Cunjia Liu, Weiru Liu  
**Category**: cs.AI  
**Published**: 2026-04-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.26095v1  

#### Abstract
{Closed-loop inverse source localization and characterization (ISLC) requires a mobile agent to select measurements that localize sources and infer latent field parameters under strict time constraints.} {The core challenge lies in the belief-space objective: valid uncertainty estimation requires ex...

---

### 27. [CogRAG+: Cognitive-Level Guided Diagnosis and Remediation of Memory and Reasoning Deficiencies in Professional Exam QA](https://arxiv.org/abs/2604.25928)

**Authors**: Xudong Wang, Zilong Wang, Zhaoyan Ming  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.25928v1  

#### Abstract
Professional domain knowledge underpins human civilization, serving as both the basis for industry entry and the core of complex decision-making and problem-solving. However, existing large language models often suffer from opaque inference processes in which retrieval and reasoning are tightly enta...

---

### 28. [Budget-Constrained Causal Bandits: Bridging Uplift Modeling and Sequential Decision-Making](https://arxiv.org/abs/2604.26169)

**Authors**: Abhirami Pillai  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.26169v1  

#### Abstract
Treatment allocation under budget constraints is a central challenge in digital advertising: advertisers must decide which users to show ads to while spending a limited budget wisely. The standard approach follows a two-stage offline pipeline - first collect historical data to estimate heterogeneous...

---

### 29. [SWAN: World-Aware Adaptive Multimodal Networks for Runtime Variations](https://arxiv.org/abs/2604.26181)

**Authors**: Jason Wu, Shir-Kang Scott Jinn, Yuyang Yuan, Maggie Wigness, Lance M. Kaplan, Hang Qiu, Mani Srivastava  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.26181v1  

#### Abstract
Multimodal deep neural networks deployed in realistic environments must contend with runtime variations: changes in modality quality, overall input complexity, and available platform resources. Current networks struggle with such fluctuations -- adaptive networks cannot adhere to a strict compute bu...

---

### 30. [NeuroPlastic: A Plasticity-Modulated Optimizer for Biologically Inspired Learning Dynamics](https://arxiv.org/abs/2604.26297)

**Authors**: Douglas Jiang, Yuechen Wang, Jiayi Wang, Jiaying Geng, Qinglong Wang, Feng Tian  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2604.26297v1  

#### Abstract
Optimization algorithms are fundamental to modern deep learning, yet most widely used methods rely on update rules based primarily on local gradient statistics. We introduce NeuroPlastic, a plasticity-modulated optimizer that augments gradient-based updates with an adaptive multi-signal modulation m...

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
