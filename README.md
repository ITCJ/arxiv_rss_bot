# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-30 07:58:18 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?](https://arxiv.org/abs/2604.26412)

**Authors**: Tianyu Liu, Yuhao Shen, Xinyi Hu, Baolin Zhang, Hengxin Zhang, Jun Dai, Jun Zhang, Shuang Ge, Lei Chen, Yue Li, MingCheng Wan  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 13.5  
**Type**: new  
**ArXiv ID**: 2604.26412v1  

#### Abstract
Speculative decoding accelerates LLM inference, but SOTA hidden-state-based drafters suffer from long-range decay: draft accuracy degrades as the speculative step increases. Existing work attributes this decay to train-inference mismatch and proposes test-time training (TTT) as a remedy, yet we obse...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*When Hidden States Drift: Can KV Caches Rescue Long-Range Speculative Decoding?*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文聚焦于**长程衰减（long-range decay）**问题：在基于 **hidden-state reuse** 的 speculative decoding（如 EAGLE-3 和 MTP）中，随着推测步数 $k$ 增加，draft token 的接受率持续下降，限制了最大有效 draft 长度和端到端加速效果。

尽管已有工作通过 **autoregressive Test-Time Training (TTT)** 缓解了训练-推理不一致（train-inference mismatch），但实验证明长程衰减依然存在，说明该现象不能仅由“状态漂移”解释。

---

### 提出了什么新方法或新思路
作者提出了一种全新的视角——**上下文信息保留（context information preservation）**，并由此引出 **KV-Reuse Hypothesis**：

> **重用目标模型的 KV Cache 而非 hidden state，可以更好地保留前缀中的 token-level 信息，从而缓解 long-range decay。**

为验证这一假设，作者构建了一个统一的诊断框架：**KVShot**，系统比较三种表示复用范式：
- **Hidden-only reuse**：标准 EAGLE/MTP 范式，复用 hidden states。
- **KV-only reuse**：仅通过 cross-attention 注入目标模型的 KV Cache。
- **Hybrid reuse**：结合两者，以 hidden state 为主干，KV Cache 提供带门控的增量修正（gated delta correction）。

---

### 相比现有方法的优势
- **理论层面**：首次从“信息压缩 vs. 显式存储”的角度分析 long-range decay，指出 hidden state 是一种 query-dependent 的有损压缩，而 KV Cache 保留了完整的历史表征。
- **机制设计**：提出的 **gated delta fusion** 结构允许 KV 通路作为 long-range 补偿项，既保留了 hidden state 在短程预测上的优势，又增强了远距离建模能力。
- **可扩展性**：KVShot 框架可用于未来对不同表示复用策略的公平比较。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **初步消融实验**：`ShareGPT`（约 70k 样本，训练 3 轮）
- **端到端评估**：`ShareGPT + UltraChat`（共 280k 样本），且响应由 **Qwen3-8B 自身再生**，确保训练分布与推理对齐。

---

### 实验设置和评估指标

#### 目标模型
- **Qwen3-8B**（Qwen Team et al., 2025）

#### 训练方式
- 所有 drafters 均采用 **autoregressive TTT**（Test-Time Training）目标进行训练，与 EAGLE-3 保持一致。
- 复用目标模型中 **Ls=3 层均匀采样的 KV Cache**。

#### 评估指标
| 指标 | 含义 |
|------|------|
| $\alpha_k$ | 第 $k$ 步 draft token 的接受率（step-wise acceptance rate） |
| MAT | Expected Mean Accepted Tokens: $ \text{MAT} = 1 + 2\sum_{i=1}^{6}\prod_{j=1}^{i}\alpha_j $ |
| HF-MAT | 使用 HuggingFace 推理管道在 draft tree $(8,10,60)$ 下测量的 MAT，更接近真实端到端性能 |

---

### 基线方法对比
- **主基线**：1-layer **EAGLE-3** drafter（hidden-only reuse）
- 对比变体：
  - KV-only reuse（不同注入方式）
  - Hybrid reuse（gated delta fusion）
  - Cross-only（仅用 KV，无 self-attention 主干）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ 表 1 & 2：KV-only Reuse 性能
| 方法 | $\alpha_0$ | $\alpha_3$ | $\alpha_6$ | Retention ($\alpha_6/\alpha_0$) | MAT |
|------|------------|------------|------------|-------------------------------|-----|
| EAGLE-3 (baseline) | 0.638 | 0.511 | 0.469 | 73.5% | **2.37** |
| KV-only (1-layer) | 0.494 | 0.381 | 0.353 | 71.5% | 1.84 |
| KV-only (4-layer) | 0.614 | 0.527 | 0.495 | **80.6%** | 2.34 |

> 🔍 发现：
> - 单层 KV-only 效果差，$\alpha_0 < 0.5$，说明缺乏 rich semantic content。
> - 随着深度增加（1→4 层），MAT 显著提升（+0.5），**验证了 query estimation 是瓶颈**。
> - **长程保留率（Retention）随层数上升至 80.6% > EAGLE-3 的 73.5%**，支持 Prediction 1。

---

#### ✅ 表 3：Hybrid Reuse 性能（ShareGPT）
| 方法 | Init | $\alpha_0$ | $\alpha_3$ | $\alpha_6$ | Retention | MAT |
|------|------|------------|------------|------------|----------|-----|
| EAGLE-3 | — | 0.638 | 0.511 | 0.469 | 73.5% | 2.37 |
| Hybrid | random | 0.650 | 0.527 | 0.490 | 75.4% | 2.44 |
| Hybrid | **EAGLE-3 ckpt** | **0.665** | **0.553** | **0.514** | **77.3%** | **2.54** |
| Cross-only | EAGLE-3 ckpt | 0.637 | 0.495 | 0.442 | 69.4% | 2.35 |

> 🔍 发现：
> - **Warm-start from EAGLE-3 checkpoint 极其重要**：MAT 从 2.44 → 2.54。
> - **Self-attention 主干不可替代**：移除后（Cross-only）性能下降。
> - 改进集中在 long-range（$\alpha_6$ 提升 +9.6%），支持 hybrid 设计有效性。

---

#### ✅ 表 4：端到端 HF-MAT 评估（280k 数据）
| 设置 | HF-MAT |
|------|--------|
| EAGLE-3 (existing ckpt) | 4.43 |
| EAGLE-3 (train from scratch) | 4.76 |
| EAGLE-3 (train from ckpt) | 5.01 |
| **Hybrid (train from ckpt)** | **5.04** |

> ⚠️ 关键负结果：
> - 尽管 step-wise MAT 提升显著（+0.17），但 **HF-MAT 仅提升 +0.03**。
> - 分析表明：额外 cross-attention 引入 **5–10% 额外延迟**，导致实际加速几乎为零。

---

### 消融实验结果（Appendix D）

| 变体 | MAT | 结论 |
|------|-----|------|
| 4-layer KV-only | 2.24 vs. 2.23 | 增加 KV 层数收益极小 → 瓶颈不在输入丰富度 |
| MLP projector | 2.20 | 线性投影优于复杂结构 → 优化困难 |
| Hidden→KV 投影 | 2.04 | 直接学习从 hidden 重建 KV 效果差 → **预计算 KV 更优** |
| Offline TTT | 2.18 | TTT 对 KV-drafter 仍有帮助 |
| KV grad ×50 | 2.21 | 梯度稀疏问题无法靠缩放解决 |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **Long-range decay 不仅源于 train-inference mismatch，更与表示形式有关**：
   - Hidden state 是一种 **biased context compression**，会丢弃对未来预测重要的弱相关 token 信息。
   - KV Cache 提供显式访问机制，更适合 long-horizon 预测。

2. ✅ **KV-Reuse Hypothesis 得到部分验证**：
   - KV-only drafters 在 deep layers 下表现出更好的 long-range retention。
   - Hybrid drafter 在 step-wise 层面显著优于 EAGLE-3（MAT +0.17）。

3. ❌ **当前 autoregressive TTT 训练范式严重制约 KV Reuse 潜力发挥**：
   - **Query estimation 困难**：浅层 drafter 难以逼近深层 target query。
   - **Sparse gradient on KV projections**：只有少量 draft token 更新 KV 参数，信号稀疏。
   - **Gate-induced gradient starvation**：warm-start 后 gate 快速关闭，抑制 cross-attention 分支学习。

4. 🔄 **最佳实践是 hybrid reuse**：KV 应作为 hidden state 的补充而非替代，在 long-range 提供 re-attention 修正。

---

### 方法的局限性
| 问题 | 描述 |
|------|------|
| **端到端增益微弱** | 当前 pipeline 中，KV 带来的 MAT 提升被额外计算开销抵消，无实质加速。 |
| **依赖强 warm-start** | Hybrid 模型高度依赖 EAGLE-3 checkpoint 初始化，难以独立训练。 |
| **训练动态不稳定** | Gate 容易陷入局部最优（close-and-stay-closed），阻碍 KV 通路发展。 |
| **未突破 autoregressive 范式限制** | 仍受限于 token-by-token 生成模式，无法充分利用 block-level 并行性。 |

---

### 未来工作方向
1. **转向 block-wise training pipelines**（如 DFlash）：
   - 并行生成多个 draft token，提供密集梯度信号给 KV projections。
   - 更深的 draft model 可更好估计 future queries。

2. **改进训练机制设计**：
   - 设计 anti-starvation 初始化策略（如强制早期打开 gate）。
   - 引入 curriculum learning 或 auxiliary losses 引导 KV 通路学习。

3. **探索非门控融合结构**：
   - 如 dual-path attention、late fusion 等，避免 gate 控制造成的优化困境。

4. **构建 KV-aware 的专用 drafter 架构**：
   - 不再简单叠加 cross-attention，而是原生支持 KV 输入的轻量 decoder。

---

> 💡 **最终洞见**：  
> **KV Cache 确实携带对 long-range drafting 有用的信号，但当前训练范式（autoregressive TTT）无法高效利用它。**  
> KVShot 的价值在于揭示了“信息可用性”与“可学习性”之间的差距，为下一代 speculative decoding 架构指明了方向：**必须重新思考训练范式本身**。

</details>

---

### 2. [DAK: Direct-Access-Enabled GPU Memory Offloading with Optimal Efficiency for LLM Inference](https://arxiv.org/abs/2604.26074)

**Authors**: Shouxu Lin, Zhiyuan Guo, Jiaxin Lin  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 11.0  
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
大型语言模型（LLM）推理面临严重的 **GPU 内存容量** 和 **带宽瓶颈**。传统基于 **prefetching** 的内存卸载框架存在以下问题：
- 卸载数据需先复制到本地 HBM，造成 **HBM 带宽竞争**；
- 需要预留静态 **bounce buffer**，浪费 GPU 显存容量；
- 层级粒度的预取难以实现计算与通信的完美重叠，导致 **pipeline bubbles**。

这些问题导致系统无法充分利用 **aggregate system bandwidth**（HBM + interconnect），限制了性能提升。

---

### 🚀 提出的新方法与创新思路
作者提出 **DAK**（Direct-Access Kernel），一种端到端的 **direct-access memory offloading** 框架，其核心创新包括：

#### （1）**Direct Memory Offload Architecture**
- 利用 **Tensor Memory Accelerator (TMA)** 引擎，将远程主机内存中的权重和 KV Cache **直接异步加载到 GPU 的 SMEM（Shared Memory）**，绕过 HBM。
- 设计了 **warp-specialized SplitK_GEMM** 和 **SplitK_FlashAttn** 内核，实现细粒度的 **compute-communication overlap**。

#### （2）**Optimal Per-operation Offload Ratio**
- 提出一个 **贪心算法**，为每个操作（如 attention 或 linear 层）动态分配最优卸载比例。
- 区分 **memory-bound** 和 **compute-bound** 操作，前者优先利用远程带宽，后者可容忍更高卸载比例而不影响性能。

#### （3）**Efficient TMA Access to Host Memory**
- **主动拥塞控制**（Active Congestion Control）：限制每个 SM 的 in-flight TMA 请求数量（`N_inflight`）和访问主机内存的 SM 数量（`N_SM_host`），防止干扰本地 HBM 性能。
- **TMA Multicast + Host-Locality-First 调度**：解决 **uncacheable host memory** 导致的 **read amplification** 问题，多个 SM 共享一次远程读取。

#### （4）**跨架构通用性**
- 支持不同 interconnect（如 NVLink-C2C 和 PCIe）和 GPU 架构（Hopper 及后续），具备良好的可移植性。

---

### 🔍 相比现有方法的优势
| 维度 | 传统 Prefetching 方法 | DAK |
|------|------------------------|-----|
| 数据路径 | Host → HBM → SMEM（两跳） | Host → SMEM（直通） |
| 带宽利用 | HBM 与 prefetch 流量竞争 | 并行利用 HBM + Interconnect |
| 显存开销 | 需要 bounce buffer | 无需 staging buffer |
| 卸载粒度 | 层级（coarse-grained） | 操作级 + 矩阵分块（fine-grained） |
| 缓存行为 | 忽略 host memory 不可缓存特性 | 通过 multicast 消除冗余传输 |

> ✅ **DAK 实现了理论上的最大聚合带宽利用率，显著优于所有基于 prefetch 的系统。**

---

## 2. 核心实验方法和设置

### 🧪 实验平台
在两个硬件平台上进行测试：

| 平台 | GH200 (NVLink-C2C) | RTX6000 Pro Blackwell (PCIe) |
|------|--------------------|-------------------------------|
| GPU HBM | 96 GB, 4.0 TB/s | 96 GB GDDR7, 1.8 TB/s |
| Host Memory | 480 GB | 512 GB |
| Interconnect BW | 900 GB/s (bidirectional) | 64 GB/s (unidirectional) |

---

### 📚 测试模型与工作负载
- **模型**：OPT-30B, OPT-6.7B, Llama-2-7B
- **任务**：离线批量推理（batched inference）
- **序列长度**：从 32 到 1024 不等
- **批大小**：从 8 到 512
- **解码 token 数**：每请求生成 32 个 token

---

### 📊 评估指标
- **Effective Bandwidth (EB)**：总数据量 / 单 token 推理时间，反映系统实际带宽利用率
- **Time Per Output Token (TPOT)**：端到端延迟
- **Throughput**：tokens/sec

---

### ⚔️ 基线方法对比
1. **FlexGen**：基于双缓冲、逐层预取的内存管理框架
2. **vLLM-prefetch**：支持异步预取 KV Cache 和权重
3. **vLLM-uvm**：基于 NVIDIA Unified Virtual Memory (UVM)，依赖页错误和迁移机制

---

## 3. 主要实验结果和性能指标

### 📈 性能提升（vs. 基线）
| 系统 | 性能增益 |
|------|----------|
| **GH200 (NVLink-C2C)** | 最高 **3× 吞吐提升** |
| **RTX6000 (PCIe)** | 最高 **1.8× 吞吐提升** |

> 在 OPT-30B 上，当卸载比为 10% 时，DAK 达到 **3,300 GB/s** 的有效带宽，接近理论峰值（HBM + NVLink-C2C）。

---

### 🔬 关键实验结果图示分析

#### 图 8 & 9：不同卸载比例下的性能表现
- DAK 在所有卸载比例下均优于基线，尤其在中低卸载比（0–40%）优势明显。
- 在 PCIe 系统上，超过 40% 后性能趋于收敛，因受限于低带宽 interconnect。

#### 图 11：Greedy vs. Uniform 卸载策略对比（batch size=512）
- 在卸载比 <60% 时，**greedy 算法比 uniform 提升达 1.5×**。
- 当卸载比过高时，所有操作都受 interconnect 限制，差异消失，符合理论预期。

#### 图 13：TMA Multicast 效果
- 当 batch size 从 512 增至 1024，**multicast 带来 1.3× → 2.5× 的性能提升**。
- 因为更大的 batch 导致更多重复读取，而 multicast 完美消除 read amplification。

#### 图 12：消融实验
| 优化项 | 性能提升 |
|-------|---------|
| 拥塞控制（Congestion Control） | 最高 **1.22×** |
| 内核实对齐（Kernel Alignment） | 最高 **1.2×** |

---

### 🧩 大模型运行能力验证（Fig. 10 & 14）
- 在 **OPT-30B + batch=128 + seq=1024** 场景下，全局卸载比达 **60%**。
- DAK 相比 vLLM 提速 **1.06×–1.83×**，相比 FlexGen 提速 **1.53×–21×**。
- 表明 DAK 能高效支持超出 GPU 显存的大规模推理。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Direct Access 优于 Prefetching**  
   直接访问远程内存可实现真正的 **bandwidth aggregation**，避免 HBM 竞争和 staging buffer 开销。

2. **TMA 是实现高效 direct access 的关键硬件**  
   利用 TMA 实现异步、非阻塞的数据拉取，是绕过软件 paging 开销的有效手段。

3. **Per-operation 卸载策略至关重要**  
   统一卸载比例会浪费资源；区分 memory-bound 和 compute-bound 操作并动态分配，才能最大化性能。

4. **uncacheable host memory 是重大挑战**  
   若不处理，会导致严重 read amplification；TMA multicast 是解决该问题的理想方案。

5. **DAK 实现近似理论最优性能**  
   在多种配置下达到 **near-optimal aggregate bandwidth**，验证了设计有效性。

---

### ⚠️ 方法的局限性
- **依赖 TMA 硬件支持**：仅适用于 Hopper 及以后架构（如 Blackwell），不兼容旧 GPU。
- **调度复杂性增加**：需要精细控制 `N_inflight` 和 `N_SM_host`，依赖离线调优。
- **目前仅支持部分算子**：主要覆盖 GEMM 和 FlashAttention 类型操作，尚未扩展至所有 LLM 层。

---

### 🔮 未来工作方向
1. 扩展至更多算子类型（如 RMSNorm、SwiGLU 等）
2. 支持动态批处理（dynamic batching）和流式推理
3. 探索在 MoE 模型中的应用
4. 自动化参数调优（如 congestion window size）
5. 结合 CXL 内存池构建更大规模 tiered memory 系统

---

## ✅ 总结
**DAK 是首个将 TMA 用于远程内存 direct access 的系统**，通过 **细粒度卸载 + 异步传输 + 拥塞控制 + multicast**，实现了 LLM 推理中内存卸载的 **最优效率**。实验证明其在 NVLink-C2C 和 PCIe 系统上分别取得最高 **3× 和 1.8× 的性能提升**，是突破 LLM “memory wall” 的重要进展。

> 🔗 项目已开源：[https://github.com/shouxulin/DirectAccessKernel.git](https://github.com/shouxulin/DirectAccessKernel.git)

</details>

---

### 3. [Adaptive and Fine-grained Module-wise Expert Pruning for Efficient LoRA-MoE Fine-Tuning](https://arxiv.org/abs/2604.26340)

**Authors**: Weihang Li, Jianchun Liu, Hongli Xu  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.26340v1  

#### Abstract
LoRA-MoE has emerged as an effective paradigm for parameter-efficient fine-tuning, combining the low training cost of LoRA with the increased adaptation capacity of Mixture-of-Experts (MoE). However, existing LoRA-MoE frameworks typically adopt a fixed and uniform expert configuration across heterog...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Adaptive and Fine-grained Module-wise Expert Pruning for Efficient LoRA-MoE Fine-Tuning*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **LoRA-MoE** 框架在进行参数高效微调时存在两个关键缺陷：

1. **粗粒度的专家分配策略**：大多数方法采用**固定且统一的专家数量**（uniform expert allocation），即所有 Transformer 层和模块（如 attention 和 MLP）都分配相同数量的 LoRA 专家。然而，不同模块的功能角色和容量需求差异显著，这种“一刀切”的设计导致：
   - 部分模块**过度配置**（over-provisioning），专家利用率低；
   - 另一些模块则**容量不足**，限制模型表达能力。

2. **持续的负载均衡约束**（Persistent Load Balancing）：
   - 虽然训练初期引入辅助损失 $L_{\text{aux}}$ 有助于防止路由崩溃（routing collapse），促进探索；
   - 但在训练后期，当路由模式已趋于稳定后，继续强制负载均衡会**抑制专家专业化**（specialization），阻碍模型对下游任务的深度适应。

---

### 🚀 提出的新方法：**DMEP**（Dynamic Module-wise Expert Pruning）

作者提出 **DMEP** —— 一种动态、细粒度的模块级专家剪枝框架，其核心思想是：

> **专家容量不应预先设定或全局共享，而应根据训练过程中实际的路由利用情况在线自适应调整。**

#### 方法流程分为三个阶段：

| 阶段 | 内容 |
|------|------|
| **Phase I: Load-Aware Dense Initialization** | 初始化一个均匀密集的 LoRA-MoE 架构，在前一个 epoch 进行探索性训练，并通过轻量级追踪器记录每个模块中各专家的实际 token 分配频率（离散 Top-k 路由）。同时保留 $L_{\text{aux}}$ 以保证初始稳定性。 |
| **Phase II: Fine-Grained Module-wise Pruning** | 基于收集到的利用率分数（如 Gini 系数、Routing Entropy），对每个模块独立执行**物理剪枝**（physical pruning）：<br> - 移除利用率低于阈值 $T$ 的专家；<br> - 同步清除其对应的 **optimizer states**（momentum, variance）；<br> - 对权重和 gating network 进行结构化重构（structural slicing）。 |
| **Phase III: Relaxed Specialization** | 在剪枝后的异构架构上继续训练，**关闭 $L_{\text{aux}}$**，使剩余专家可自由专注于任务目标，发展更强的任务特异性专长。 |

---

### 🔍 相比现有方法的优势

| 维度 | DMEP 的优势 |
|------|-------------|
| **结构灵活性** | 支持**模块级差异化专家数量**，比 layer-wise 更精细，真正实现“按需分配”。 |
| **效率提升** | 物理移除参数和 optimizer states，而非 mask，显著降低显存占用与计算开销。 |
| **优化自由度** | 动态解除负载均衡，释放专家专业化潜力，避免正则化冲突。 |
| **自动化程度高** | 不依赖人工设定或离线分析，完全基于在线路由统计自动决策。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

在以下三个具有代表性的推理任务上进行评估：

| 数据集 | 任务类型 | 描述 |
|--------|--------|------|
| **ScienceQA** | 多模态科学问答（文本子集） | 过滤掉需图像输入的问题，转为标准多选题格式。 |
| **OpenBookQA** | 常识与基础科学知识问答 | 转换为严格的四选一选择题（A/B/C/D）。 |
| **GSM8K** | 数学应用题求解 | 模型直接输出最终数字答案（#### 后接答案），通过正则提取评估。 |

---

### ⚙️ 实验设置

| 项目 | 设置详情 |
|------|----------|
| **Base Models** | `Qwen3-0.6B` 和 `Qwen3-8B`，验证跨规模有效性 |
| **Targeted Modules** | 所有 attention 与 FFN 投影层：<br> `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`（共 7 个/层） |
| **LoRA Rank** | $r = 8$, 缩放因子 $\alpha = 16$ |
| **Routing Strategy** | Top-2（$k=2$） |
| **Training Epochs** | 5 epochs，warm-up epoch $E_w = 1$ |
| **Optimizer** | AdamW，峰值学习率 $3\times10^{-4}$，cosine 衰减，10% 线性预热 |
| **Batch Size** | 有效 batch size 固定为 128 |
| **Precision** | bfloat16 混合精度训练 |
| **Hardware** | 单张 NVIDIA RTX A6000（48GB VRAM） |

---

### 📊 评估指标

| 指标 | 说明 |
|------|------|
| **Accuracy (%)** | 测试集上的准确率 |
| **Trainable Parameters (M)** | 可训练参数总量（百万） |
| **Training Throughput (tok/s)** | 每秒处理的 token 数量，反映训练速度 |
| **Gini Coefficient** | 衡量专家间负载不均程度（越高越集中） |
| **Routing Entropy / Drift** | 分析路由分布的锐利性与稳定性 |

---

### 🆚 基线方法对比

| 方法 | 说明 |
|------|------|
| **Dense LoRA** | 标准 LoRA，无 MoE 结构（$N=1, k=1$） |
| **Symmetric MoE (N=8)** | 统一为每模块 8 个专家，启用 $L_{\text{aux}}$ |
| **DMEP (T=0.10)** | 本文方法，默认剪枝阈值 $T=0.10$，$K_{\min}=2$ |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总（来自 Table I）

| Model | Dataset | Method | Params (M) | Throughput (tok/s) | Accuracy (%) |
|-------|--------|--------|------------|---------------------|--------------|
| Qwen3-0.6B | ScienceQA | Symmetric MoE | 42.7 | 1368 | 91.32 |
| | | **DMEP (Ours)** | **24.2** | **1466** | **91.55** |
| Qwen3-0.6B | OpenBookQA | Symmetric MoE | 42.7 | 923 | 69.00 |
| | | **DMEP** | **24.2** | **1032** | **69.00** |
| Qwen3-0.6B | GSM8K | Symmetric MoE | 42.7 | 1354 | 17.00 |
| | | **DMEP** | **25.3** | **1502** | **19.00** |
| Qwen3-8B | ScienceQA | Symmetric MoE | 185.2 | 294 | 97.08 |
| | | **DMEP** | **121.8** | **320** | **96.40** |
| Qwen3-8B | OpenBookQA | Symmetric MoE | 185.2 | 196 | 95.00 |
| | | **DMEP** | **107.9** | **216** | **95.00** |
| Qwen3-8B | GSM8K | Symmetric MoE | 185.2 | 288 | 43.00 |
| | | **DMEP** | **110.4** | **317** | **43.00** |

---

### ✅ 对比结果总结

- **参数减少**：相比对称 MoE 基线，DMEP 平均减少 **35%–43%** 的可训练参数。
- **吞吐提升**：训练吞吐平均提高约 **10%**（最高达 15% 以上）。
- **精度表现**：
  - 在多数任务上保持甚至**超越基线精度**（如 GSM8K 上从 17.00% → 19.00%）；
  - 少数情况下略有下降（如 Qwen3-8B on ScienceQA：97.08% → 96.40%），但仍在合理范围内。

---

### 🔍 消融实验结果（Table II，Qwen3-0.6B on ScienceQA）

| 配置 | 参数量 | 吞吐 | 准确率 |
|------|--------|--------|--------|
| Symmetric MoE (N=8) | 42.7M | 1368 tok/s | 91.32% |
| Symmetric MoE (N=4) | 21.3M | 1439 tok/s | 90.29% |
| **DMEP (T=0.10)** | **31.2M** | **1438 tok/s** | **91.55%** |
| DMEP (T=0.15) | 13.1M | 1508 tok/s | 90.65% |
| DMEP (Step 50) | 26.7M | 1457 tok/s | 91.23% |
| DMEP (Step 250) | 35.3M | 1275 tok/s | 91.23% |

#### 发现：

- **动态剪枝优于对称压缩**：即使保留更多参数（31.2M vs 21.3M），DMEP 仍取得更高精度（91.55% vs 90.29%），说明“精准保留”比“全面削减”更有效。
- **剪枝时机影响大**：在第 100 步（第一轮结束）剪枝效果最佳；过早或过晚都会损害性能。
- **阈值控制权衡**：更高的 $T$ 导致更稀疏结构和更高吞吐，但可能牺牲精度，提供灵活的部署选项。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **模块级异质性真实存在**：
   - 注意力模块（如 `O_PROJ`）通常表现出高度偏斜的路由行为（高 Gini），适合大幅压缩；
   - MLP 模块（如 `GATE_PROJ`）路由更均匀，需保留较多专家。

2. **静态架构设计存在冗余**：
   - 固定、统一的专家分配造成严重资源浪费，尤其在浅层或低利用模块中。

3. **负载均衡应阶段性使用**：
   - 初期有益于探索，后期反而成为优化瓶颈，适时关闭可释放专家专业化能力。

4. **物理剪枝带来真实加速**：
   - 不仅减少参数，还清除 optimizer states 和计算图中的冗余部分，实现真正的结构稀疏与效率增益。

---

### ⚠️ 方法的局限性

1. **依赖路由稳定性判断**：
   - 剪枝时机依赖于 routing drift 的收敛，若某些任务收敛慢或不稳定，可能导致误剪。

2. **一次性剪枝不可逆**：
   - 当前采用 one-shot 剪枝策略，无法恢复已被删除的专家，缺乏弹性。

3. **小模型收益更大**：
   - 在较小模型（如 0.6B）上能提效并提点；在大模型（8B）上主要体现为效率提升，精度增益有限。

4. **超参数敏感性**：
   - 阈值 $T$ 和 $K_{\min}$ 需要根据任务调节，虽有一定鲁棒性，但仍影响最终性能。

---

### 🔮 未来工作方向

1. **支持迭代式动态剪枝与扩展**：
   - 引入可生长机制，在必要时重新激活专家或新增分支。

2. **结合 sensitivity analysis 自动确定阈值**：
   - 利用梯度敏感度等指标辅助决定剪枝强度，进一步降低人工干预。

3. **扩展至其他 PEFT 方法**：
   - 探索将 DMEP 思路应用于 Prefix-tuning、Adapter 等 MoE 化版本。

4. **面向推理阶段的部署优化**：
   - 将训练后得到的异构结构用于推理加速，研究其在边缘设备上的可行性。

---

## ✅ 总结

**DMEP** 是一项针对 **LoRA-MoE 微调范式结构性低效问题** 的重要改进。它通过 **模块级动态剪枝 + 阶段性解除负载均衡**，实现了：

> **更紧凑的结构、更高的训练效率、更强的专家专业化能力。**

其实验充分证明了该方法能在 **减少 35%-43% 参数** 的同时，**提升约 10% 吞吐**，并在多个复杂推理任务上 **持平或超越强基线**，显著提升了 LoRA-MoE 的实用性和性价比，为资源受限场景下的高效微调提供了新范式。

</details>

---

### 4. [Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training & Inference](https://arxiv.org/abs/2604.26294)

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

### 解决的问题
在大规模 Transformer 模型的训练与推理中，**显存瓶颈**是核心挑战，主要来自两个方面：
- **参数内存**（Parameters, Gradients, Optimizer States）：随模型规模增长。
- **激活内存**（Activations）：随序列长度平方级增长。

传统的多维并行策略（如 TP、SP、TP+SP）通过将不同并行维度分配到正交的设备网格轴上（如 `TP axis` 和 `SP axis`），虽然能分别缓解上述两类内存压力，但也带来了以下问题：
- 需要大量设备组成高维并行组，导致部分通信必须跨越低带宽的 **inter-node links**（如 InfiniBand）。
- 占用过多设备用于模型并行，减少了可用于 **Data Parallelism (DP)** 的副本数量，影响吞吐量。

### 提出的新方法：Tensor and Sequence Parallelism (TSP)
论文提出 **TSP** —— 一种将 **Tensor Parallelism (TP)** 和 **Sequence Parallelism (SP)** “折叠”到**单一设备轴**上的新型并行执行策略。

#### 核心思想
- 不再为 TP 和 SP 分配独立的 mesh 维度。
- 而是在一个大小为 $ D $ 的并行组中，每个 rank 同时持有：
  - 一份 **权重分片**（weight shard）
  - 一份 **序列分片**（sequence shard）
- 从而在单个维度上同时实现参数内存和激活内存的缩减。

#### 创新设计
- **Attention 层**：采用 **循环广播 + All-Gather K/V** 的调度方式。
  - 每轮广播一个 rank 的权重分片（WQ/WK/WV/WO）给所有 peer。
  - 各 rank 在本地序列上计算 Q/K/V。
  - 对 K/V 进行 All-Gather 并通过 **ZigZag Reorder** 恢复因果上下文。
  - 最后累加局部输出。
- **MLP 层**：采用 **Ring Communication**。
  - 权重分片在 ring 中循环传递（point-to-point send/recv）。
  - 每个 rank 积累部分输出，避免了传统 TP 所需的 All-Reduce。
  - 输出累加完全本地化。

### 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **内存效率** | 同时降低参数类内存（↓1/D）和激活内存（↓1/D），是唯一在单轴上实现双重缩减的方法。 |
| **拓扑友好性** | TSP 组大小为 $ D $，而传统 TP+SP 需要 $ T \times 2 $。更小的组更容易完全映射到高带宽的 **intra-node interconnect**（如 NVLink / Infinity Fabric），避免跨节点通信。 |
| **资源利用率** | 节省下来的设备可用于扩展 **Data Parallelism**，提升整体吞吐。 |
| **通信可重叠性** | MLP 的 ring 通信和 Attention 的多阶段流水线设计，使得大部分通信可以与 GEMM 计算重叠，不显著增加 wall-clock 时间。 |

---

## 2. 核心实验方法和设置

### 数据集
- 论文未使用具体自然语言任务数据集进行端到端微调或评测。
- 实验基于**合成数据**进行前向/反向传播的性能分析，关注的是系统级指标而非任务准确率。

### 实验设置
- **硬件平台**：
  - 使用 **MI300X GPU** 构建集群。
  - 单节点 8 GPUs，通过 **Infinity Fabric** 互联（高带宽 intra-node）。
  - 多节点间通过 **Pollara 400Gbps NICs** 互联（inter-node）。
- **模型配置**（7B Reference Model）：
  - Decoder-only Transformer，32 层，hidden dim = 4096，FFN expansion = 4，MHA。
  - 总参数约 8.6B（文中称 7B）。
- **并行度测试范围**：$ D = 2, 4, 8 $
- **序列长度范围**：从 16K 到 128K tokens
- **微批次大小**（micro-batch size）：$ B = 1 $ 至 $ B = 16 $

### 评估指标
| 指标 | 描述 |
|------|------|
| **Peak Memory per GPU** | 每张 GPU 的峰值显存占用（MB/GB） |
| **Throughput (tokens/s)** | 前向或前后向每秒处理的 token 数量 |
| **Communication Volume** | 每层每设备通信字节数（理论分析） |
| **FLOPs** | 每设备计算量（理论分析） |

### 基线方法对比
- **DP**（Data Parallelism）
- **TP**（Tensor Parallelism）
- **SP**（Sequence Parallelism）
- **TP+SP**（传统二维并行，如 T=2, 2=4 或 T=4, 2=2）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）显存表现（图 9）
- 在所有序列长度下，**TSP 显存最低**。
- **短序列（16K）**：
  - TSP ≈ 31.0 GB
  - TP ≈ 31.5 GB（接近，因参数主导）
  - SP ≈ 58.8 GB（未分片参数）
- **长序列（128K）**：
  - TSP ≈ 103.3 GB
  - TP ≈ 119.0 GB（激活未分片）
  - SP ≈ 108.5 GB（参数未分片）
  - TP+SP ≈ 108.5–119.0 GB（取决于拆分方式）

✅ **TSP 在长短序列均最优，且优势随序列增长而扩大**。

#### （2）吞吐量表现（图 10–12）
- **TSP 持续优于匹配的 TP+SP 基线**。
- 在 $ D=8 $、$ S=128K $ 下，TSP 吞吐比 TP+SP 高出约 **15–20%**。
- 微批次增大时，TSP 可支持更大的 batch size（得益于更低显存），进一步拉大吞吐优势。

✅ **额外通信未成为瓶颈，计算-通信重叠有效**。

#### （3）通信体积（图 5a, 表 III）
- **TSP 前向通信体积高于 TP 和 SP**，因其需移动权重。
- 但在满足 $ BS > 8h $ 时（即批量或序列足够大），其通信总量趋近于 TP。
- 实际运行中，由于通信被计算掩盖，**并未转化为显著的延迟增加**。

✅ **通信代价被可重叠性抵消，实际性能不受损**。

#### （4）消融实验（隐含于设计对比）
- **Attention 使用 Broadcast + All-Gather** vs Ring：
  - 因 K/V All-Gather 是主要通信项，使用全组参与的 collective 更适合 Infinity Fabric 的带宽特性。
- **MLP 使用 Ring** vs All-Reduce：
  - Ring 避免了输出 All-Reduce，通信量更小且易于与 GEMM 重叠。
- **ZigZag Partitioning**：
  - 用于平衡因果注意力的负载，确保各 rank 工作量均衡。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **TSP 是一种高效的“内存优先”并行范式**：
   - 在长上下文和显存受限场景下，**减少复制比最小化通信更重要**。
2. ✅ **通信体积 ≠ 通信成本**：
   - 尽管 TSP 引入更多通信，但通过精心设计的流水线和异步传输，**通信可被计算有效掩盖**。
3. ✅ **拓扑感知设计至关重要**：
   - TSP 更容易适配高带宽 intra-node fabric（如 MI300X 的 Infinity Fabric），避免跨节点通信。
4. ✅ **TSP 与其他并行策略正交兼容**：
   - 可作为单一 mesh 维度，灵活组合进 PP、EP、DP 等多维并行框架中。

### 方法的局限性
| 局限性 | 说明 |
|--------|------|
| **通信开销增加** | 相比纯 TP 或 SP，TSP 引入了运行时权重移动，对低带宽网络不友好。 |
| **实现复杂度高** | 需要复杂的调度逻辑（广播循环、ring 传递、ZigZag 重排）和多流控制。 |
| **依赖硬件特性** | 在非对称带宽拓扑（如 xGMI）中优势明显，但在 NVSwitch 等全连接架构中增益可能缩小。 |
| **仅适用于 Dense Layer** | 当前设计针对标准 Transformer，对 MoE 等稀疏结构需额外适配（但可结合 EP）。 |

### 未来工作方向
1. **扩展至 Mixture-of-Experts (MoE) 模型**：
   - 探索 TSP 与 Expert Parallelism (EP) 的协同优化。
2. **动态调整 TSP 度数**：
   - 根据序列长度自动选择是否启用 TSP 或切换回 TP/SP。
3. **支持更多通信原语优化**：
   - 如使用 NCCL/RCCL 的定制 collective 来进一步压缩 TSP 通信开销。
4. **编译器集成**：
   - 将 TSP 调度自动化，纳入深度学习编译器（如 TorchAO、PopXL）中。
5. **异构设备支持**：
   - 在 CPU-offload 或 multi-accelerator 场景下探索 TSP 的适用性。

---

> **总结一句话**：  
> TSP 通过将 TP 和 SP 折叠到同一设备轴，在几乎不牺牲吞吐的前提下，实现了**当前最紧凑的显存占用**，是一种面向长上下文和内存受限场景的**硬件感知、拓扑友好的新型并行范式**。

</details>

---

### 5. [Efficient, VRAM-Constrained xLM Inference on Clients](https://arxiv.org/abs/2604.26334)

**Authors**: Aditya Ukarande, Deep Shekhar, Marc Blackstein, Ram Rangan  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.26334v1  

#### Abstract
To usher in the next round of client AI innovation, there is an urgent need to enable efficient, lossless inference of high-accuracy large language models (LLMs) and vision language models (VLMs), jointly referred to as xLMs, on client systems. To address this, we present pipelined sharding, a novel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient, VRAM-Constrained xLM Inference on Clients**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
本文致力于解决在**客户端系统**（如PC、笔记本）上高效运行高精度大模型（xLMs，包括LLMs和VLMs）时面临的**显存（VRAM）受限**问题。具体挑战包括：
- 在有限的VRAM预算下实现**无损推理**（lossless inference）；
- 支持**交互式**（batch size = 1）和**批处理**（batch size > 1）两种模式；
- 适应不同的系统条件（CPU线程数、PCIe带宽、VRAM大小）和推理阶段（prefill vs decode）；
- 同时支持**密集模型**（dense LLMs）和**混合专家模型**（MoE LLMs），以及**高分辨率视觉语言模型**（VLMs）。

现有方法如量化、剪枝等虽能降低显存占用，但会牺牲模型精度；而现有的CPU-GPU混合调度方案无法全面兼顾上述所有需求。

---

### **提出了什么新方法或新思路**
提出了一种名为 **Pipelined Sharding** 的新型、基于**基准性能分析**（benchmark-profile-guided）的CPU-GPU混合调度技术，其核心思想包括：

1. **子层级分片**（sub-layer level sharding）  
   将Transformer层进一步拆分为Attention和FFN等子模块，实现更细粒度的计算图划分。

2. **优先级张量放置策略**（prioritized tensor placement）  
   根据子模块的重要性（Attention > KV Cache > FFN > Output）动态分配VRAM资源，确保关键部分优先驻留GPU。

3. **流水线拷贝-计算重叠**（pipelined copy-compute）  
   在数据传输（如权重从sysRAM到VRAM）的同时执行计算任务，隐藏PCIe延迟。

4. **Token Tier自适应调度机制**  
   预先为不同“新生成token数量”级别（如1, 4, 16, ..., 16K）生成多个调度计划，并选择最优者，使算法能灵活应对不同请求组合与上下文长度。

此外，针对VLMs，提出了 **VLMOpt** 优化套件，包含：
- **Vision Tensor CPU Offloading**：将视觉编码器权重卸载至CPU；
- **Flash Attention for Vision Encoder**：在视觉编码中启用FlashAttention以减少O(N²)注意力开销；
- **Vision-Language VRAM Allocation Avoidance**：避免视觉与语言部分的显存分配重叠，降低峰值显存。

---

### **相比现有方法的优势**
| 特性 | Pipelined Sharding | 其他方法（如TwinPilots、HeteGen等） |
|------|--------------------|-------------------------------|
| 优化TTFT | ✅ | ❌ |
| 支持Batch=1 | ✅ | ❌/部分 |
| 支持Dense LLMs | ✅ | ✅ |
| 支持MoE LLMs | ✅ | ❌/部分 |
| 区分Attention与FFN优先级 | ✅ | ❌ |
| KV Cache分片 | ✅ | ❌ |
| 支持VLMs | ✅ | ❌ |
| 客户端实测验证 | ✅ | ❌/有限 |

> ✅ 表示支持，❌ 表示不支持或未验证

该方法是目前唯一一个同时满足所有关键特性的解决方案。

---

## 2. **核心实验方法和设置**

### **使用的模型**
共测试六类模型，涵盖LLMs与VLMs：
| 类型 | 模型名称 | 参数规模 | 显存占用 |
|------|--------|---------|--------|
| Dense LLM | `nemo4b`, `nemo8b` | 4B, 8B | 7.7GB, 15.7GB |
| MoE LLM | `qwen30b`, `qwen235b` | 30B, 235B | 16.4GB, 77.0GB |
| VLM | `vnemo4b`, `cr1` | — | 8.4GB, 15.4GB |

其中CR1为NVIDIA Cosmos-Reason1推理VLM，用于物理AI场景。

---

### **实验设置**
- **硬件平台**：三类客户端设备
  - `cli1`: 笔记本（RTX 3500, 12GB VRAM）
  - `cli2`: 中端台式机（RTX 5070 Ti, 16GB VRAM）
  - `cli3`: 高端台式机（RTX 5090, 32GB VRAM）

- **软件框架**：基于 **llama.cpp** 实现（开源分支），支持GGUF格式模型加载。

- **评估指标**：
  - **TTFT**（Time-to-First-Token）：首token延迟
  - **TPS**（Tokens Per Second）：每秒输出token数
  - **E2EL**（End-to-End Latency）：总响应时间 = TTFT + 100 / TPS
  - **Peak VRAM Usage**：峰值显存消耗
  - **Speedup**：相对于baseline的加速比

- **VRAM预算范围**：从2G到32G（以1000MB为单位，非1024MB）

---

### **基线方法对比**
- **LLMs**：使用 **llama-cpp-baseline**，即手动调优`-ngl`参数（指定多少层放GPU）达到最大适配VRAM的配置。
- **CR1 VLM**：使用 **vLLM** 作为上限基线（仅在20G以上可用），低VRAM下仍用llama-cpp-baseline。

> 所有基线均为“aggressive”设定，力求公平比较。

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### **LLM 性能提升（在cli3上）**
| 模型 | 最小VRAM | TPS | TTFT加速倍数 | TPS加速倍数 |
|------|--------|-----|--------------|------------|
| `qwen235b` | 2G | 7.7 TPS (@1K ctx) | 最高 **6.7×** | 最高 **30×** |
| `qwen30b` | 2G | 20–26 TPS (@≤16K ctx) | 平均 **2×** | 平均 **3.7×** |
| `nemo8b` | 2G | 可达 >5 TPS | — | 达标交互体验 |

> 注：人类阅读速度约需 **4–5 TPS**，本文方法在极低VRAM下即可达标。

#### **批量推理性能（Batched Mode）**
- 批处理吞吐量平均提升 **2.3×**，最高达 **8.2×**（qwen30b @4K context, bs=16）。
- 支持统一KV缓存（unified KV cache）和非统一模式，均显著优于基线。

#### **VLM 性能表现**
| 模型 | 优化效果 |
|------|--------|
| `vnemo4b` | E2EL最高提速 **1.78×**，可在2G VRAM运行 |
| `cr1` | **显存需求下降10×**（从20G → 2G），支持1440p图像输入 |

> 在480p–1440p多分辨率测试中，CR1均可成功运行于低至2G VRAM。

---

### **与基线方法的对比结果**
- **TTFT**：平均加速 **2×**，最高 **6.7×**
- **TPS**：平均加速 **3.7×**，最高 **30×**（qwen235b @64K context）
- **E2EL**：平均加速 **2×**，最高 **4.3×**
- **Batched Throughput**：平均提升 **2.3×**，最高 **8.2×**

> 即使在高VRAM条件下，本文方法仍优于人工调参基线，因其能自动利用多余空间进行更优调度。

---

### **消融实验与敏感性分析**
#### **调度策略有效性验证**
- 在105个配置中，调度器选择最优策略的成功率为 **100%**。
- 不同条件下三种计划各有优势：
  - **GPU-only**：适合CPU线程少、PCIe带宽高
  - **Static**：适合CPU线程充足（占72%胜率）
  - **Dynamic**：适合中等负载，可重叠传输与计算

#### **资源敏感性研究**
- **CPU线程增加 → TPS持续上升**（qwen30b从1线程→16线程，TPS提升超2倍）
- **PCIe从Gen3升至Gen5 → TTFT加速近2倍**（因权重流更快）
- **并发游戏运行测试**：存在Pareto最优VRAM分配点，平衡LLM推理与游戏FPS

---

## 4. **关键结论和发现**

### **主要发现**
1. **Token Tier设计是通用高效的调度范式**  
   能自然覆盖从单请求交互到多请求批处理的所有场景，无需为不同模式单独优化。

2. **子层切分 + 优先级分配显著提升资源利用率**  
   Attention和KV Cache优先保留在VRAM中，对降低TTFT至关重要。

3. **无损推理完全可行且高效**  
   通过智能调度而非压缩/量化，在保持精度前提下实现大规模模型落地。

4. **VLM显存瓶颈可通过工程优化突破**  
   VLMOpt使得CR1这类高分辨率VLM能在消费级设备运行。

5. **自动化优于人工调参**  
   Pipelined Sharding在各种条件下均超越手动offloading策略，尤其在复杂混合负载中优势明显。

---

### **方法的局限性**
- 当前实现依赖 **NVIDIA CUDA GPU**，虽框架通用，但生产部署集中在CUDA生态。
- **未考虑热管理与功耗波动**，当前profile未包含动态电源状态的影响。
- 对 **全新算子类型** 需要更新benchmark suite才能准确建模。
- 多进程共享GPU时缺乏QoS保障，需依赖外部调度器（如AI Management Processor）。

---

### **未来工作方向**
1. **扩展至NPU/Mobile SoC架构**，适配更多异构硬件；
2. **集成动态profiling机制**，在驱动或固件更新后自动重新校准性能数据库；
3. **支持视频输入流推理**，拓展VLM应用场景；
4. **开发统一的任务共调度器**，实现LLM、图形、游戏等多任务间的QoS隔离与协同；
5. **探索更激进的内存复用策略**，如KV Cache压缩+分页管理结合。

---

> 💡 **一句话总结**：  
> **Pipelined Sharding + VLMOpt 实现了首个支持全场景、全模型类型、全系统条件的无损客户端xLM推理方案，在极端VRAM限制下仍能达到交互级性能，推动高精度AI真正走向终端。**

</details>

---

### 6. [EvoSelect: Data-Efficient LLM Evolution for Targeted Task Adaptation](https://arxiv.org/abs/2604.26170)

**Authors**: Ting-Wei Li, Sirui Chen, Jiaru Zou, Yingbing Huang, Tianxin Wei, Jingrui He, Hanghang Tong  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.26170v1  

#### Abstract
Adapting large language models (LLMs) to a targeted task efficiently and effectively remains a fundamental challenge. Such adaptation often requires iteratively improving the model toward a targeted task, yet collecting high-quality human-labeled data to support this process is costly and difficult ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EvoSELECT: Data-Efficient LLM Evolution for Targeted Task Adaptation**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在大型语言模型（LLM）的**目标任务适配**（targeted task adaptation）中，传统的迭代生成-训练范式（generation-training loop）面临两个核心挑战：
- **数据漂移**（data drift）：由外部数据生成器（data generator）合成的样本可能逐渐偏离目标任务分布。
- **冗余与噪声**：生成的数据高度重复或对齐性差，直接用于训练会稀释有效学习信号，甚至导致性能下降。

现有基于**数据选择**（data selection）的方法通常孤立地处理任务对齐（alignment）或多样性（diversity），缺乏统一框架，导致次优选择。

---

### **提出的新方法与新思路**
作者提出了 **EvoSELECT** —— 一种基于**迭代生成-选择-训练循环**（generation-selection-training loop）的新型数据高效 LLM 进化框架。

#### **核心创新点**：
1. **联合建模任务对齐与多样性**  
   EvoSELECT 在一个统一、原则性的框架中同时优化：
   - **Task Alignment**：通过 **Optimal Transport (OT)** 量化候选样本与目标任务验证集之间的对齐程度，避免仅聚焦于分布中心（centroid）而忽略全局几何结构。
   - **Diversity**：引入基于梯度相似性的正则项，抑制选择梯度方向高度相似的冗余样本，提升覆盖范围。

2. **基于代理模型的高效表示**  
   使用轻量级 **proxy model** 提取训练与验证样本的梯度特征，并通过投影压缩（如 Sparse Johnson-Lindenstrauss Transform）降低计算开销，实现可扩展的数据表示。

3. **统一优化流程**  
   采用指数更新（exponentiated update）交替优化 OT 对齐梯度与多样性梯度，动态调整样本权重，最终选出 top-k 高质量样本。

---

### **相比现有方法的优势**
| 方法类型 | 代表 | 缺陷 | EvoSELECT 改进 |
|--------|------|------|----------------|
| **Attribution-only** | Xia et al. (2024) | 聚焦平均梯度方向，导致样本集中、多样性低 | 使用 OT 尊重完整分布结构，提升覆盖 |
| **Diversity-only** | Jung et al. (2025) | 忽视任务相关性，选中的样本可能无关紧要 | 显式建模任务对齐，确保有效性 |
| **Heuristic Hybrid** | TSDS (Liu et al., 2024) | 先对齐后局部去重，解耦处理，表达能力受限 | 联合优化，协同增强两种信号 |

> ✅ **EvoSELECT 是首个将 OT 直接用于同步优化对齐与多样性的数据选择方法**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
涵盖三大类共 10 个知识密集型 QA 基准：

| 类别 | 数据集 |
|------|-------|
| **科学推理** | ARC-Challenge, MMLU, OpenBookQA, ClimaQA |
| **常识/逻辑推理** | CommonsenseQA, LogiQA, LogiQA2 |
| **生物医学/医疗推理** | Med-MCQA, MedQA, HeadQA |

---

### **实验设置**
- **生成-选择-训练循环**：运行两轮迭代（Iter. 1 & 2）
- **数据生成器**：固定为 `Qwen2.5-14B-Instruct`
- **被训练模型（base model）**：
  - 强生成器设置：`Qwen2.5-3B-Instruct`
  - 弱生成器设置：`Qwen2.5-14B-Instruct`（即生成器与训练模型同规模）
- **代理模型（proxy model）**：`Qwen2.5-0.5B-Instruct`
- **选择比例（selection ratio）**：0.2 和 0.5
- **训练方式**：LoRA 微调（SFT），使用 TRL 库
- **评估指标**：准确率（Accuracy），零样本评测（zero-shot evaluation）

---

### **基线方法对比**
| 基线方法 | 描述 |
|--------|------|
| **All** | 使用全部生成数据进行训练 |
| **Random** | 随机选择指定比例的数据 |
| **Attribution** | 基于梯度相似性选择最相关的样本（Xia et al., 2024） |
| **Diversity** | 基于聚类选择最多样的样本（Jung et al., 2025） |
| **Attr-Div** | 先按 Attribution 筛掉底部 25%，再应用 Diversity 选择 |
| **TSDS** | 当前最优混合方法，结合 OT 与 KDE 正则化（Liu et al., 2024） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（3B base model）**
> 所有数值为平均准确率（Avg. Accuracy），加粗为 Top-1 / 下划线为 Top-2

| 方法 \ Ratio | 0.2 | 0.5 |
|------------|-----|-----|
| **Base** | 0.7504 | 0.7504 |
| **All** | 0.7416 | 0.7416 |
| **Random** | 0.7509 | 0.7463 |
| **Attribution** | 0.7423 | 0.7504 |
| **TSDS** | 0.7495 | 0.7443 |
| **EvoSELECT** | **0.7580** | **0.7599** |

✅ **EvoSELECT 在所有设置下均优于其他方法**，且是**唯一始终超越 Base 模型**的选择策略。

---

### **与基线方法的对比结果**
- **一致提升**：如图 3 所示，在所有数据集和选择比例下，**只有 EvoSELECT 实现了持续正向增益**（绿色），其余方法在某些任务上出现负增益（红色），表明其可能导致“有害适配”（harmful adaptation）。
- **显著优势**：
  - 在 **LogiQA2** 上，EvoSELECT 达到 0.4198（vs Base 0.3836），相对提升超 9%。
  - 在 **MedQA** 上，达到 0.5334（vs Base 0.5043），提升明显。
- **弱生成器场景同样有效**：即使生成器不强于训练模型（14B vs 14B），EvoSELECT 仍保持领先（见 Appendix G 表格）。

---

### **消融实验与进一步分析**
- **Vendi Score 分析**（衡量多样性）：
  - Attribution 方法多样性最低（冗余严重）
  - Diversity 方法任务对齐最差
  - **EvoSELECT 平衡两者，实现高对齐 + 高多样性**
- **任务难度越高，优势越明显**（图 4）：
  - 在基础模型表现较差的任务上（如 ClimaQA、LogiQA），EvoSELECT 增益最大，说明其特别适合**困难任务的精准适配**。
- **胜率分析**（图 6）：
  - EvoSELECT 在所有任务类别（C1–C3）中均取得最高 Rank-1 胜率（>60%），远超第二名（TSDS ~25%），体现其**跨领域鲁棒性**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **盲目训练所有生成数据是有害的**：`All` 方法常低于 Base 性能，证明**数据效率至关重要**。
2. **单一目标无法满足需求**：仅考虑对齐或多样性都会导致性能瓶颈。
3. **EvoSELECT 实现稳定、高效的进化路径**：
   - 是**唯一在整个演化过程中持续提升性能**的方法；
   - 即使在低选择比（20%）下也能超越全量训练；
   - 在强/弱生成器设置下均表现优异，具有广泛适用性。

---

### **方法的局限性**
- **依赖验证集**：需要一个小规模高质量验证集来引导选择，若验证集有偏，则影响效果。
- **计算复杂度较高**：尽管使用 proxy model 加速，OT 与多样性梯度联合优化仍比简单方法更耗时（时间复杂度：O(Tnm log(max(n,m)) γ⁻²)）。
- **当前仅适用于监督微调**（SFT），尚未扩展至 RLHF 或长上下文任务。

---

### **未来工作方向**
- 探索无需验证集的自监督对齐机制。
- 将 EvoSELECT 框架应用于 **instruction tuning** 或 **domain adaptation** 中的大规模预训练阶段。
- 结合主动学习（active learning）动态构建种子集与验证集。
- 扩展至多模态 LLM 的数据选择任务。

---

> 📌 **总结一句话**：  
> **EvoSELECT 通过将 Optimal Transport 与多样性正则化联合优化，首次实现了在迭代生成中“既对齐又多样”的高质量数据选择，显著提升了 LLM 在目标任务上的适应效率与稳定性，是迈向高效、可控 LLM 演化的关键一步。**

</details>

---

### 7. [COPUS: Co-adaptive Parallelism and Batch Size Selection in Large Language Model Training](https://arxiv.org/abs/2604.26687)

**Authors**: Akhmed Sakip, Erland Hilman Fuadi, Omar Sayedelahl, Zonghang Li, Jianshu She, Alham Fikri Aji, Steve Liu, Eric Xing, Qirong Ho  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.26687v1  

#### Abstract
Training large language models requires jointly configuring two interdependent aspects of the system: the global batch size, which governs statistical efficiency, and the 3D parallelism strategy, which governs hardware throughput. Existing approaches make these decisions independently: optimization ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：COPUS: Co-adaptive Parallelism and Batch Size Selection in Large Language Model Training**

---

## **1. 主要贡献和创新点**

### **解决的问题**
在大规模语言模型（LLM）训练中，**全局批大小（global batch size, Bg）** 和 **3D 并行策略（3D parallelism strategy, S = (d, t, p)** 在数据并行（DP）、张量并行（TP）、流水线并行（PP）之间）是两个关键配置参数。传统方法通常将这两个决策**独立处理**：
- **优化领域**：自适应批大小（如基于 GNS 的方法）仅调整 Bg 以跟踪“临界批大小”（critical batch size），但固定并行策略。
- **系统领域**：并行优化器（如 Alpa、Galvatron）为固定批大小选择最优并行策略，但不考虑批大小会随训练动态变化。

这种解耦导致训练过程中长期处于**次优配置**：当批大小增长时，并行策略可能已不再高效。

### **提出的新方法**
本文提出了 **Copus**，首个在单次 LLM 训练过程中**联合自适应调整**以下三个参数的系统：
- 全局批大小 `Bg`
- 微批大小 `Bm`
- 3D 并行策略 `S`

其核心思想是：**批大小与并行策略是强耦合的**，应共同优化。

### **核心创新点**
1. **引入 Goodput 作为统一优化目标**  
   Goodput 定义为：
   ```
   Goodput = Throughput × Statistical Efficiency
   ```
   即单位时间内的有效收敛速度。它同时建模了硬件吞吐量和统计效率，避免了只优化单一维度的片面性。

2. **LR-aware Goodput 扩展**  
   针对 Adam 优化器的平方根学习率缩放规则（`η ∝ √Bg`），进一步扩展 Goodput 公式为：
   ```
   Goodput ∝ T × SE × √Bg
   ```
   这使得大批次在提升学习率方面的优势被显式建模。

3. **3D 并行感知的在线 GNS 估计**  
   改进传统假设纯数据并行的 GNS 估计器，使其能正确处理梯度累积（micro-batch）和 3D 并行下的梯度方差，实现低开销、高准确的噪声估计。

4. **支持运行时并行策略重配置（online resharding）**  
   实现了无需重启的并行策略切换，通过 CPU 内存暂存状态，在 GPU 上重建新的 process group 和分片布局，显著降低重配置延迟（相比 checkpoint-restart 快 2–16×）。

### **相比现有方法的优势**
- **打破批大小与并行的解耦设计**，实现端到端协同优化。
- **Goodput 指标更贴近真实训练目标**（快速收敛），而非单纯最大化吞吐或样本效率。
- **对 GNS 校准误差更具鲁棒性**：由于 Goodput 同时受吞吐约束，即使 GNS 估计不准，也能选出较稳健的配置。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **WikiText-103** 数据集，序列长度为 2,048 tokens。
- 使用各模型对应的预训练 tokenizer。

### **模型与硬件配置**
| 模型 | 参数量 | 硬件 |
|------|--------|------|
| LLaMA-3.2-3B | 3B | 1×8 H100 |
| LLaMA-2-13B | 13B | 2×8 H100 |
| Qwen-2.5-32B | 32B | 4×8 H100 |
| LLaMA-2-7B | 7B | 4×8 MI210 (AMD) |

### **评估指标**
- **Time to Target Loss**：达到指定损失值所需的时间（分钟）。
- **Goodput**：综合吞吐与统计效率的指标。
- **Speedup**：相对于最强基线的加速比。

### **基线方法对比**
1. **Static GBS Baselines**  
   固定全局批大小，使用离线 profiling 找到该批大小下最优的 `(S, Bm)`。
2. **CBS Baselines**  
   基于 GNS 动态调整批大小，但**固定并行策略**。测试两种变体：
   - **Pessimistic CBS**：针对小批大小优化初始并行策略。
   - **Optimistic CBS**：针对大批大小优化初始并行策略。

### **实验设置细节**
- 使用 **BF16 + AdamW**，学习率按 `√Bg` 缩放。
- GNS 估计使用指数移动平均（EMA）平滑。
- Copus 触发切换需满足：新配置 Goodput 提升 ≥10%。
- 离线生成 **throughput lookup table**，覆盖所有内存可行的 `(S, Bg, Bm)` 组合。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
| 配置 | 平均加速比（vs 最强 baseline） | 峰值加速比 |
|------|-------------------------------|------------|
| 3B / 1×8 H100 | **+7.9%** | +10.1% |
| 13B / 2×8 H100 | **+3.9%** | +6.1% |
| 32B / 4×8 H100 | **+8.0%** | **+11.1%** |
| 7B / 4×8 MI210 | **+5.0%** | +11.0% |

> ✅ **平均加速 3.9–8.0%，峰值达 11.1%**，且包含重配置开销。

### **与基线方法对比**
- **Copus 始终优于或等于最佳基线**。
- 在 32B 模型上收益最大，因其多节点拓扑提供更多并行策略选择，吞吐差异更大。
- 在 13B 上增益较小，因其 CBS 基线初始策略在整个训练中接近最优。
- **Fixed-large-batch（如 GBS=1024）严重拖慢早期训练**：图1显示 Copus 达到 loss=6.2 仅需 2 分钟，而固定大批次需 47 分钟。

### **消融与分析**
#### **Goodput 分解分析（图8）**
- Copus 能持续保持**高吞吐**与**高统计效率**。
- Static 小批大小虽统计效率高，但吞吐低；大批次反之。
- Copus 通过动态切换并行策略（如从小 PP 切换到全 DP），在批增大时维持高吞吐。

#### **重配置开销分析（表2）**
| 配置 | 重配置耗时（Copus） | Checkpoint-Restart | 加速倍数 |
|------|---------------------|---------------------|----------|
| 3B | 30–56 秒 | 231–238 秒 | **4.2–7.8×** |
| 32B | 52 秒 | 809 秒 | **15.7×** |

> ✅ 在线 resharding 显著降低重配置成本。

#### **GNS 校准敏感性分析（附录C）**
- CBS 方法：批大小选择与 GNS 校准误差呈**线性关系**。
- Goodput 方法：批大小选择与校准误差呈**平方根关系**，更鲁棒。

---

## **4. 关键结论和发现**

### **主要发现**
1. **批大小与并行策略存在强耦合**  
   不同批大小下，最优并行策略不同。例如：
   - 小批大小时，PP-heavy 策略更优（避免 DP 浪费）。
   - 大批大小时，DP-heavy 策略更优（同步开销被摊薄）。

2. **Goodput 是更合理的优化目标**  
   仅优化统计效率或吞吐都会导致整体收敛变慢。Copus 通过联合优化实现了**双赢**。

3. **动态重配置切实可行且必要**  
   即使重配置有开销（数十秒），只要长期 Goodput 增益足够，仍能带来净收益。

4. **跨平台有效性**  
   在 NVIDIA H100 和 AMD MI210 两种不同互联架构上均取得收益，表明方法具有通用性。

### **局限性**
- **决策空间有限**：未包含 ZeRO、context parallelism、sequence parallelism 等其他并行维度。
- **依赖离线 profiling**：需预先测量 throughput lookup table，无法完全自适应新硬件。
- **GNS 校准仍需人工设定**：当前使用固定比例因子 `c=2.0`，缺乏自动校准机制。
- **扩展性限制**：实验规模为 1–4 节点（8–32 GPUs），更大规模效果未知。

### **未来工作方向**
- 扩展决策空间至更多并行维度（如 expert parallelism）。
- 替换离线 profiling 为**在线吞吐建模**，实现完全自适应。
- 自动学习 GNS 到 CBS 的映射函数，替代线性校准。
- 探索更细粒度的学习率调度策略，适配不同批大小。
- 应用于短周期开发任务（如数据混合搜索），加速整个研发流程。

---

> **总结**：Copus 首次实现了 LLM 训练中批大小与并行策略的**协同自适应优化**，通过 Goodput 指导的动态配置选择，在多种模型与硬件上实现了 **3.9–8.0% 的平均加速**，验证了“批-并行耦合”优化的重要性，为未来全自动训练系统提供了新范式。

</details>

---

### 8. [FaaSMoE: A Serverless Framework for Multi-Tenant Mixture-of-Experts Serving](https://arxiv.org/abs/2604.26881)

**Authors**: Minghe Wang, Trever Schirmer, Mohammadreza Malekabbasi, David Bermbach  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.26881v1  

#### Abstract
Mixture-of-Experts (MoE) models offer high capacity with efficient inference cost by activating a small subset of expert models per input. However, deploying MoE models requires all experts to reside in memory, creating a gap between the resource used by activated experts and the provisioned resourc...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《FaaSMoE: A Serverless Framework for Multi-Tenant Mixture-of-Experts Serving》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
Mixture-of-Experts (MoE) 模型虽然在推理时仅激活少量专家（sparse activation），但部署时仍需将**所有专家常驻内存**，导致资源利用率低下。这一问题在 **multi-tenant 场景**下尤为严重——每个租户独立部署完整模型会造成大量重复的专家副本，造成严重的内存和计算资源浪费。

此外，现有优化方法（如 expert offloading、deduplication）多聚焦于单租户场景，无法有效实现跨租户的专家共享与弹性伸缩。

### 提出的新方法与创新思路
论文提出 **FaaSMoE**，一种基于 **Function-as-a-Service (FaaS)** 的多租户 MoE 推理服务架构，其核心思想是：

- **解耦控制平面与执行平面**：
  - 将 MoE 模型中的 **非专家组件**（如 tokenizer、attention、gating 网络）保留在轻量级的 **Orchestrator** 中集中管理；
  - 将 **专家模型** 部署为无状态的 FaaS 函数，按需调用、自动扩缩容（scale-to-zero）。

- **支持可配置的专家粒度（configurable expert granularity）**：
  - 允许多个专家打包成一个 **Expert Block** 作为一个 FaaS 函数部署；
  - 在 **调用开销**（invocation overhead）与 **专家弹性**（elasticity）之间进行权衡。

- **天然支持多租户共享**：
  - 多个租户共享同一组专家函数池，避免重复部署；
  - 利用 FaaS 平台的强隔离性和自动扩缩能力，适应动态负载。

### 相比现有方法的优势
| 对比维度 | 现有方法（如 MoESaic、本地分布） | FaaSMoE |
|--------|-------------------------------|-------|
| 资源效率 | 仍保留全模型实例或固定服务器部署 | 实现 scale-to-zero，仅使用实际需要的资源 |
| 弹性伸缩 | 缺乏细粒度弹性或依赖专用集群 | 利用 FaaS 自动扩缩，支持突发负载 |
| 租户间共享 | 有限去重（如 MoESaic） | 完全共享专家池，显著降低总体资源占用 |
| 部署灵活性 | 通常需手动配置资源 | 无需人工干预，serverless 特性即插即用 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **BIG-Bench**：从该基准中选取了 6 个客户端，每个客户端发出 5 个异构任务，共 30 个请求/轮次；
- 请求具有多样性，模拟真实 multi-tenant 工作负载下的专家访问模式。

### 实验设置
- **原型实现**：
  - 模型：`Qwen1.5-MoE-2.7B`（含多个 MoE 层，每层 60 个专家 + 4 个共享专家）；
  - FaaS 平台：开源边缘导向平台 **tinyFaaS**；
  - 运行环境：纯 CPU 服务器（300 GB 内存），无 GPU；
  - 通信方式：异步 HTTP 调用实现微批处理（micro-batching）以减少网络开销。

- **部署策略对比**（共四种）：
  1. **Baseline**：每个租户独占一个完整的 MoE 模型；
  2. **Local Distribution**：租户保留非专家模块，专家集中部署在本地 Uvicorn 服务器上（非 FaaS，不具弹性）；
  3. **FaaSMoE-Shared**：所有租户共享一个 Orchestrator，调用 FaaS 上的专家函数；
  4. **FaaSMoE-Private**：每个租户运行自己的 Orchestrator，分别调用远程专家。

- **评估指标**：
  - **CPU 使用率**（%）：进程级采样，100% 表示单核满载；
  - **内存消耗**（GB）：系统级监控；
  - 所有指标取平均值并附带标准差或置信区间。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Figure 3）
| 部署策略 | 平均总 CPU 使用率 | 平均总内存消耗 |
|---------|------------------|---------------|
| Baseline | 1126.84% | 217.52 GB |
| Local Distribution | 428.67% | **50.38 GB** |
| FaaSMoE-Shared | **326.40%** | **72.25 GB** |
| FaaSMoE-Private | 408.49% | 90.98 GB |

> ✅ **相比 Baseline，FaaSMoE-Shared 资源使用不到三分之一！**

### 与基线方法对比结果
- **内存节省显著**：
  - FaaSMoE-Shared 相比 Baseline 内存下降约 **67%**（217.52 → 72.25 GB）；
  - 尽管 Local Distribution 内存更低（50.38 GB），但缺乏 FaaS 提供的弹性与隔离性。
- **CPU 效率更高**：
  - FaaSMoE-Shared CPU 使用仅为 Baseline 的 ~29%，得益于跨租户微批处理与连接复用；
  - FaaSMoE-Private 因每个租户维护独立 Orchestrator，开销略高但仍远优于 Baseline。

### 消融实验结果：专家块大小（Expert Block Size）的影响（见 Figure 5）
研究了不同 **block size**（6~30 个专家/函数）对性能的影响：

#### CPU 使用趋势
- **Local Distribution**：随 block size 增大，CPU 单调下降（减少调用次数）；
- **FaaSMoE**：呈非单调变化，在 block size=20 附近出现峰值，size=30 时下降；
  - 原因：涉及 **batching 效率、FaaS 调用频率、路由开销** 的复杂权衡。

#### 内存使用趋势（U 形曲线）
- block size 从 6 增加到 20：内存持续下降（达到最优）；
- 继续增加至 30：内存反而上升；
  - 原因：过粗粒度导致单个函数内存驻留过高，且运行时开销增大。

> 🔍 最佳实践建议：将每层 MoE 分为约 **3 个 Expert Block**（如 60 个专家 → 每块 20 个），可在弹性与效率间取得良好平衡。

---

## 4. 关键结论和发现

### 主要发现
1. **FaaS 是 MoE 多租户服务的理想运行时**：
   - 其 event-driven、scale-to-zero、stateless 特性完美匹配 MoE 的稀疏激活特征；
   - 支持跨租户专家共享，大幅降低总体资源占用。

2. **控制平面与执行平面解耦有效提升资源效率**：
   - 将专家作为 FaaS 函数部署，实现了真正的“按需加载”；
   - 控制逻辑轻量化且可集中/分布式部署，灵活适配不同隔离需求。

3. **专家粒度是一个关键设计参数**：
   - 不应追求极致细粒度（如 1-expert-per-function），而应根据 workload 特征调整 block size；
   - 存在明显的 **U-shaped memory-cost trade-off**。

4. **Orchestrator 放置影响全局效率 vs 租户隔离**：
   - 共享 Orchestrator 更高效（支持跨租户 batching）；
   - 私有 Orchestrator 提升隔离性但带来额外开销；
   - 可考虑混合或自适应策略。

### 方法的局限性
- **延迟问题**：当前架构引入了 Orchestrator 与远程专家之间的频繁网络通信，可能增加端到端延迟；
- **纯 CPU 设计限制**：未利用 GPU 加速，虽符合主流 FaaS 现状，但在高性能场景下可能受限；
- **冷启动风险**：FaaS 函数冷启动可能影响响应时间，尤其对低频专家；
- **依赖特定 FaaS 平台特性**：tinyFaaS 为研究原型，生产级部署需适配 AWS Lambda、Azure Functions 等。

### 未来工作方向
1. **集成低延迟通信机制**：
   - 引入 high-speed RPC 或流水线技术隐藏通信延迟；
   - 探索 co-location（将 Orchestrator 与 FaaS worker 同机部署）减少跳数。

2. **自适应专家分组策略**：
   - 动态构建 “hot-expert” blocks，提高常用专家的调度效率；
   - 基于访问模式预测进行预热或缓存。

3. **支持 GPU-based FaaS 扩展**：
   - 随着 serverless GPU 的普及（如 Databricks Serverless GPU），探索异构加速支持。

4. **更复杂的 multi-tenant QoS 保障机制**：
   - 提供 SLA、优先级调度、资源配额等企业级功能。

5. **扩展至其他稀疏模型架构**：
   - 如 Task-Specific Heads、Conditional Computation Networks 等，验证通用性。

---

> 📌 **总结一句话**：  
> **FaaSMoE 通过将 MoE 专家部署为 FaaS 函数，首次实现了真正意义上的跨租户、弹性、高效的 MoE 推理服务，在资源利用率上相较传统部署方式提升了三倍以上，为大规模多租户 AI 服务提供了可行路径。**

</details>

---

### 9. [Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving](https://arxiv.org/abs/2604.26837)

**Authors**: Zihan Zhao, Baotong Lu, Shengjie Lin, Yizou Chen, Jing Liu, Yanqi Zhang, Ziming Miao, Ming-Chang Yang, Haiying Shen, Qi Chen, Fan Yang  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.26837v1  

#### Abstract
Long-context LLM serving is bottlenecked by the cost of attending over ever-growing KV caches. Dynamic sparse attention promises relief by accessing only a small, query-dependent subset of the KV state per decoding step and extending the KV storage to CPU memory. In practice, however, these algorith...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大语言模型（LLM）在处理长上下文任务时面临严重的性能瓶颈，主要体现在两个方面：
- **KV Cache 内存压力**：随着上下文长度增长到数十万甚至百万 token，Key-Value (KV) Cache 占用的 GPU 显存急剧增加，限制了并发请求能力。
- **内存带宽瓶颈**：解码过程中需要反复读取全部历史 KV 状态，导致 GPU 内存带宽成为性能瓶颈。

尽管动态稀疏注意力（Dynamic Sparse Attention）算法通过仅关注关键 token 来减少计算量并缓解显存压力，但在系统层面却难以实现预期收益，原因在于：
- 不同稀疏算法的**粒度不统一**，缺乏通用执行框架；
- 在分层存储（GPU-CPU）架构下，不规则、细粒度的 KV 数据访问模式导致 PCIe 传输效率低下；
- 元数据开销随上下文长度呈指数级增长。

### 提出的新方法：SPIN
本文提出 **SPIN**（Sparse-attention-aware inference framework），一个面向稀疏注意力的推理框架，通过软硬件协同设计解决上述问题。其三大核心技术为：

#### （1）统一的 Partition 抽象
- 将不同稀疏算法（如块级、簇级）的操作统一到“partition”这一逻辑单元上。
- 引入 **partition-to-page mapping**，将算法定义的稀疏粒度与底层硬件友好的 page-based KV 管理解耦，支持灵活集成多种算法。

#### （2）局部性感知的 KV 缓存管理
- 设计动态缓冲机制，区分“必选页”（mandatory pages）和“缓存页”（buffering pages）。
- 提出 **bucketed LRU** 替换策略：使用有限范围的时间戳代替全局排序，避免昂贵的排序操作，适合 GPU 并行化执行，有效利用 token 访问的时间局部性。

#### （3）两级分层元数据组织
- 采用操作系统风格的多级索引结构（two-level indexing），使元数据规模与物理工作集成正比，而非最坏情况下的逻辑地址空间。
- 元数据跨 GPU 和 CPU 分布（tier-split），关键控制信息驻留 GPU，容量主导部分存放于 CPU pinned memory，显著降低 HBM 开销。

### 相比现有方法的优势
- **通用性强**：提供标准化接口，支持 ShadowKV、RetroInfer、SeerAttention-R 等多种稀疏算法即插即用。
- **系统级优化**：从执行流水线、内存管理到底层内核全面优化，真正释放稀疏注意力的理论潜力。
- **高效数据移动**：通过局部性优化和 warp-based copy kernel 最大化 PCIe 利用率。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LongBench-v2** [7]：双语、多任务长上下文理解基准，输入长（32K–120K tokens）、输出短（平均 5K tokens）。
- **LongGenBench** [36]：强调复杂推理的生成任务，输入较短（平均 18K tokens），但输出更长（平均 12K tokens）。

### 实验设置
- **硬件平台**：
  - A100 服务器：4× NVIDIA A100 (80GB HBM)，PCIe Gen4 (32 GB/s)
  - B200 服务器：4× NVIDIA B200 (180GB HBM)，PCIe Gen5 (64 GB/s)
- **模型**：
  - Qwen3-14B、Qwen3-32B、Llama-3.1-70B
- **请求模式**：Poisson 过程模拟在线服务负载，测试不同请求到达率下的性能表现。

### 评估指标
- **端到端吞吐量**（end-to-end throughput）：单位时间内生成的 output tokens 数量。
- **TTFT**（Time to First Token）：首 token 延迟。
- **TPOT**（Time Per Output Token）：每个输出 token 的平均延迟。
- **元数据 HBM 消耗**：衡量系统资源占用。
- **缓存命中率**（cache hit ratio）

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **vLLM** | 支持 PagedAttention 的主流 LLM 推理引擎，全注意力机制 |
| **vLLM-Offload** | 扩展版 vLLM，支持 KV Cache 向 CPU 内存卸载 |
| **LServe** [69] | 专注于稀疏注意力的 GPU-only 推理系统 |
| **原始稀疏实现** | 如 ShadowKV、RetroInfer 等原作者发布的原型系统 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | 结果 |
|------|------|
| **端到端吞吐量提升** | 较 vLLM 提升 **1.66–5.66×** |
| **TTFT 降低** | 较 vLLM 降低 **7–9×** |
| **TPOT 降低** | 较原始稀疏实现最多降低 **58%**，相当于吞吐提升 **2.39×** |
| **元数据 HBM 消耗减少** | 减少 **49–78×**（相比 flat page table 设计） |
| **最大批大小支持** | 支持 4–8× 更大的 batch size |

### 与基线方法的对比结果
- 在 **LongBench-v2** 上：
  - SPIN 在高请求速率下持续扩展，而 vLLM 和 vLLM-Offload 因内存抖动迅速饱和。
  - SPIN-ShadowKV 在 A100 上达到 vLLM 的 **3.80× 吞吐**（1.5 req/s）。
- 在 **LongGenBench** 上：
  - SPIN 实现 **2.62–5.66× 吞吐提升**（A100），即使在 B200 高带宽环境下仍保持优势。
- 对比 LServe：
  - LServe 虽然在预填充阶段有优势（利用稀疏性），但受限于 GPU-only 架构，无法扩展批大小，最终吞吐低于 SPIN。

### 消融实验结果
#### （1）GPU 缓冲管理的影响（图 14）
- “Base”（无缓存）：吞吐很快饱和，PCIe 成为瓶颈。
- “+Mandatory”：保留当前步骤所需页，略有改善。
- “+Mandatory + Buffering”（完整 SPIN）：引入 bucketed LRU 后，decode 吞吐进一步提升 **49%**，验证时间局部性的重要性。

#### （2）缓存大小对命中率的影响（图 16）
- 缓存命中率随缓冲区增大而上升，但在 **buffering-to-mandatory 比例达 4–5× 后趋于饱和**。
- SPIN 默认设置为 **5×**，平衡命中率与 GPU 内存占用。

#### （3）多级索引效果（图 15）
- 对于 Llama-3.1-70B（128K context, batch=32）：
  - 扁平页表设计消耗高达 **100 GB HBM**。
  - 多级索引 + 元数据卸载后降至 **<2 GB**，节省 **>50×**。

---

## 4. 关键结论和发现

### 主要发现
1. **稀疏注意力的潜力远未被现有系统释放**：由于缺乏统一抽象和低效的数据管理，现有实现距离理想性能包络（ideal sparse-serving envelope）仍有巨大差距。
2. **系统设计必须与算法协同演进**：SPIN 证明，通过 partition 抽象可将多样化的稀疏算法纳入统一框架，并共享高效的 Offload/Retrieve 实现。
3. **局部性是分层存储的关键**：token 访问具有强时间局部性，bucketed LRU 等轻量替换策略能显著提升缓存命中率。
4. **元数据开销不可忽视**：在超长上下文场景下，元数据可能超过实际 KV 数据占用，必须采用 scalable metadata design。

### 方法的局限性
- 当前 SPIN 基于 vLLM 实现，依赖其调度器和执行引擎，对其他推理框架的适配需额外开发。
- bucketed LRU 是一种近似 LRU 策略，在极端访问模式下可能不如精确策略有效。
- 所有实验均假设 KV Cache 可完全放入 CPU 内存，若超出则需引入二级存储（如 NVMe），尚未验证。

### 未来工作方向
- 将 SPIN 的思想扩展至 **prefill 阶段的稀疏性利用**，进一步加速长输入处理。
- 探索 **异构设备上的部署**（如多个 GPU + CPU + SSD 组合）。
- 支持更多类型的稀疏模式，例如基于语义或任务自适应调整 partition 粒度。
- 结合 **weight sparsity** 或 **activation sparsity**，构建全方位高效的 LLM 推理系统。

--- 

> ✅ 总结一句话：  
> **SPIN 通过统一抽象、局部性感知缓存和可扩展元数据设计，首次实现了稀疏注意力从算法潜力到系统性能的完整转化，在真实负载下达成数倍于现有系统的吞吐与延迟优势。**

</details>

---

### 10. [FloatSOM: GPU-Accelerated, Distributed, Topology-Flexible Self-Organizing Maps](https://arxiv.org/abs/2604.26555)

**Authors**: Tony Xu, Sarah Klamt, Katherine Turner, Anne Brustle, Felix Marsh-Wakefield, Givanna Putri  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.26555v1  

#### Abstract
GPU-accelerated Self-Organizing Map (SOM) implementations are among the most competitive options for large-scale SOM analysis, but growing dataset sizes increasingly challenge their practical use because workloads no longer fit cleanly within device-memory limits. We introduce FloatSOM, a SOM framew...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# FloatSOM: GPU-Accelerated, Distributed, Topology-Flexible Self-Organizing Maps 论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前主流的 **Self-Organizing Map (SOM)** 实现面临两大瓶颈：
- **系统层面**：大多数实现受限于单GPU内存（VRAM），无法处理超大规模数据集，缺乏对分布式训练、内存外（out-of-memory）执行的支持。
- **拓扑层面**：传统SOM依赖固定的矩形或六边形网格（regular lattice），这种强几何先验可能无法有效捕捉复杂、不规则的数据流形结构。

### 提出了什么新方法或新思路
作者提出了 **FloatSOM**，一个全新的、面向大规模应用的SOM框架，具备以下核心特性：

- **GPU加速与分布式训练**：基于 **Ray** 和 **NCCL** 实现多GPU、跨节点的分布式并行训练，支持从消费级显卡到高性能计算集群（HPC）的部署。
- **内存外（Out-of-Core）支持**：当数据超出GPU VRAM甚至CPU RAM时，可自动切换至磁盘流式加载（disk-backed streaming），突破设备内存限制。
- **拓扑灵活（Topology-Flexible）**：首次在大规模场景下实现了非网格拓扑结构，引入两种动态图结构：
  - **Minimum Spanning Tree (MST)**：基于节点权重构建最小生成树作为邻域图。
  - **Relative Neighborhood Graph (RNG)**：更密集的图结构，允许节点间存在多个连接路径，能更好地保留局部数据结构。
- **拓扑感知的超参数调优**：结合 **Optuna** 进行多目标超参数优化，针对不同拓扑结构进行精细化调参。

### 相比现有方法的优势
| 特性 | FloatSOM | XPySOM | GigaSOM | Somoclu |
|------|--------|--------|---------|---------|
| 多GPU支持 | ✅ | ❌（仅单GPU） | ⚠️（侧重CPU） | ⚠️（有限） |
| 分布式训练 | ✅（Ray + NCCL） | ❌ | ✅（MPI） | ✅（MPI） |
| 内存外执行 | ✅（Disk-backed） | ❌（需全载入VRAM） | ❌ | ❌ |
| 非网格拓扑 | ✅（MST, RNG） | ❌ | ❌ | ❌ |
| 超参数自动化调优 | ✅（Optuna集成） | ❌ | ❌ | ❌ |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
实验分为两类基准测试：

#### (1) Optuna 质量基准（算法性能）
- **14个数据集**（10个真实 + 4个合成）：
  - 真实数据：`iris`, `wine`, `breast_cancer`, `digits`, `olivetti_faces`, `diabetes`, `california_housing`, `kddcup99`, `covertype`
  - 合成数据：`blobs`, `circles`, `moons`, `s_curve`, `swiss_roll`
- 所有输入经 **StandardScaler** 归一化，并划分为 70%/30% 的训练/验证集。

#### (2) 速度扩展性基准（系统性能）
- 使用 **合成随机矩阵**（uniform [0,1]），用于测试可扩展性：
  - 样本数：$10^6$ 到 $10^9$
  - 特征维度：50 到 5000
  - 网格大小：$8 \times 8$ 到 $64 \times 64$

### 实验设置和评估指标

#### 评估指标
- **主指标**：**Quantization Error (QE)**，衡量SOM对数据的表示精度。
  - $QET$：训练集QE
  - $QEH$：验证集QE（泛化能力）
  - $QEB$：平衡QE = $(QET + QEH)/2$，综合性能指标
- **运行时指标**：
  - 绝对训练时间（wall-clock time）
  - 加速效率 $E_G = (T_1 / T_G) / G \times 100\%$，衡量多GPU扩展性

#### 基线方法对比
- **XPySOM**：当前最先进的GPU加速SOM，作为主要对比基线（默认六边形拓扑）。
- **Hexagonal SOM**：作为固定网格拓扑的基准。
- **Full vs Random Sampling**：比较完整采样与随机子采样的效果。

#### 超参数优化
- 使用 **Optuna** 进行 **200 trials × 10 seeds** 的多目标优化（同时优化 $QET$ 和 $QEH$）。
- 调优参数包括：初始半径、初始化方式、衰减类型、动量等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### (1) 拓扑结构性能对比（QEB）
在 **Full Sampling + 超参数调优** 下：
- **RNG > MST > Hexagonal**
- RNG 在 **balanced QE** 上显著优于六边形拓扑（全局中位提升 **14.5%**，p=7.4e-10）
- 尤其在大型真实数据集上优势明显，说明其拓扑灵活性更能适应复杂数据分布。

#### (2) 超参数调优收益
- 调优后的配置相比 **XPySOM 默认参数** 显著提升性能：
  - 全局 $QEB$ 改善 **~15%**
  - $QET$ 改善 **~22.5%**
- 证明 **超参数调优是必要环节**，而非可选后处理。

#### (3) 最大尺度训练性能
- 在 **8 GPUs** 上训练一个 **1024节点** 的SOM网络，处理 **10亿样本 × 50维特征**：
  - 仅用 **6.16分钟（369.41秒）**
  - 此规模远超 XPySOM 的内存承载能力（后者在 $10^8$ 样本即失败）

#### (4) 多GPU扩展效率
- 在样本和维度扩展任务中，**8-GPU 效率可达 80–100%+**，部分情况因内存模式转变出现“超线性加速”。
- 但在 **Grid-Size Scaling** 中，当网格极大（如 $64 \times 64$）时，多GPU加速比下降至 **5.69%**，表明此时通信与拓扑刷新开销占主导。

#### (5) 拓扑运行时开销（8 GPUs）
| 拓扑 | 网格大小64平均耗时 | 相对Hexagonal倍数 |
|------|------------------|------------------|
| Hexagonal | 32.54 秒 | 1.0x |
| MST | 266.45 秒 | 8.19x |
| RNG | 880.83 秒 | 27.07x |

> 注：RNG 因需维护更复杂的图结构，计算开销显著更高。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **RNG 是最优拓扑**：在表示精度（QE）上全面超越传统六边形和MST，尤其适合高维、非合成数据。
2. **拓扑与超参数耦合**：不同拓扑需要不同的超参数配置，**必须进行拓扑感知的调优**。
3. **Full Sampling 更稳定**：在小数据集（<10k）上，随机采样会导致性能不稳定；大数据集上两者差异不大。
4. **FloatSOM 可扩展性强**：通过分布式+内存外机制，成功将SOM推向 **十亿级样本** 规模，且保持高吞吐。
5. **系统设计影响效率**：多GPU效率高度依赖工作负载类型，**数据并行易扩展，模型并行（大网格）难扩展**。

### 方法的局限性
- **RNG 计算开销大**：虽然精度高，但训练时间是六边形的 **27倍**，不适合对速度敏感的场景。
- **拓扑刷新成本高**：MST/RNG 需在训练中动态重建图结构，带来额外计算负担。
- **未支持所有图拓扑**：目前仅实现MST和RNG，其他图结构（如kNN图）尚未集成。
- **依赖高性能基础设施**：最佳性能需HPC环境支持，普通用户可能难以复现最大规模结果。

### 未来工作方向
- 开发更高效的图拓扑近似算法，降低RNG/MST的计算成本。
- 探索异步更新或分层拓扑刷新策略，减少通信与同步开销。
- 扩展支持更多图结构（如UMAP图、kNN图）作为SOM邻域。
- 集成在线学习与增量训练模式，适应流数据场景。
- 提供更友好的API和可视化工具，降低使用门槛。

---

> **总结**：FloatSOM 是首个将 **拓扑灵活性**、**分布式训练** 和 **内存外执行** 集于一体的SOM框架，不仅在算法性能上超越现有方法，更在系统层面实现了数量级的扩展能力，为大规模无监督学习提供了新的可能性。

</details>

---

### 11. [Who Trains Matters: Federated Learning under Enrollment and Participation Selection Biases](https://arxiv.org/abs/2604.26604)

**Authors**: Gota Morishita  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.26604v1  

#### Abstract
Federated learning (FL) trains a shared model from updates contributed by distributed clients, often implicitly assuming that contributing clients are representative of the target population. In practice, this representativeness assumption can fail at two distinct stages, inducing selection bias. Fi...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Who Trains Matters: Federated Learning under Enrollment and Participation Selection Biases**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题

该论文指出，在 **Federated Learning (FL)** 中，模型训练的客户端通常并非来自目标总体的代表性样本，从而导致 **selection bias（选择偏差）**。这种偏差在实践中通过两个阶段产生：

- **Enrollment Bias（注册偏差）**：由于设备限制、软件要求或用户同意等原因，只有部分客户端被允许加入训练系统。
- **Participation Bias（参与偏差）**：即使已注册，客户端是否实际参与每一轮训练还受电池状态、网络状况等动态因素影响。

现有研究大多关注第二阶段（round-level participation bias），而忽视了第一阶段的 **enrollment bias**，这会导致训练目标与真实目标总体之间存在 **持久性错配（persistent objective mismatch）**。

---

### 🚀 提出的新方法与新思路

作者提出了一个 **两阶段选择模型（two-stage selection model）** 来形式化上述过程，并基于此提出以下创新方法：

#### （1）**FedIPW（Federated Inverse Probability Weighting）**
- 一种基于逆概率加权（Inverse Probability Weighting, IPW）的聚合机制。
- 将总纳入概率分解为：
  $$
  \pi_{i,r} = \underbrace{P(E_i=1|Z_i)}_{\text{enrollment propensity}} \times \underbrace{P(A_{i,r}=1|E_i=1,Z_i,X_{i,r})}_{\text{participation propensity}}
  $$
- 在服务器端对每个客户端更新按其估计的纳入概率倒数进行加权，以恢复对 **target-population mean update** 的无偏估计。

#### （2）**Aggregate-Calibration 扩展（有限信息下的修正）**
- 当非注册客户端的个体协变量不可得时（常见于实际部署），无法直接估计 enrollment propensity。
- 引入一种 **aggregate-calibration 方法**：利用已知的目标总体汇总统计量（如各地区/设备类型的占比），调整已注册客户端的权重，使其加权分布匹配总体分布。
- 这是一种实用的近似修正手段，适用于“limited-information”场景。

#### （3）**算法无关的优化误差分析**
- 首次提供了在残差加权误差（residual weighting error）下的 FL 收敛性分析。
- 发现不完全的选择偏差修正确实会引入一个 **non-vanishing bias floor（非消失偏差底限）**，其大小取决于：
  - 残差权重误差 $ e_w $
  - 客户端梯度异质性 $ G $
  - 强凸参数 $ \mu $

---

### 🔍 相比现有方法的优势

| 方面 | 现有方法（如 FedAvg、Round-only IPW） | 本文方法（FedIPW） |
|------|----------------------------------------|---------------------|
| 考虑偏差类型 | 仅处理 participation bias 或忽略所有偏差 | 显式建模并联合纠正 enrollment + participation bias |
| 目标一致性 | 可能收敛到“可参与客户端”的最优解 | 更接近 **target-population objective** |
| 方法灵活性 | 多为 heuristic 客户端采样策略 | 基于因果推断框架，具有理论保证 |
| 实用扩展 | 缺乏对缺失注册数据的支持 | 提供 aggregate-calibration 应对现实约束 |

此外，FedIPW 是 **algorithm-agnostic** 的，可与其他 FL 方法（如 FedProx、SCAFFOLD）结合使用。

---

## 2. **核心实验方法和设置**

### 📊 数据集
- 使用 **合成数据集（synthetic dataset）** 构造的 **federated logistic regression** 任务。
- 设计目的：能够精确计算目标总体的全局最优解 $ \theta^* $，从而直接衡量 **objective mismatch**。

### 🧪 实验设置
- **客户端总数**：固定有限人口 $ N $
- **特征生成**：
  - 每个客户端有 pre-enrollment covariates $ Z_i $（如区域、设备类型）
  - $ Z_i $ 同时影响本地数据分布和被选中的概率
- **两阶段选择机制**：
  1. **Enrollment Mechanism**：基于 $ Z_i $ 决定是否注册（引入 enrollment bias）
  2. **Participation Mechanism**：在注册客户端中，基于 $ Z_i $ 和 $ X_{i,r} $（系统状态）决定每轮是否参与
- **通信轮次**：$ R = 1000 $ 轮
- **本地训练**：每轮 K 步 SGD

### 🎯 评估指标
- **Target-Population Excess Loss**：即 $ F(\theta_R) - F^* $，衡量最终模型相对于目标总体最优值的差距。
- **收敛轨迹可视化**：展示不同方法在训练过程中目标损失的变化。
- **敏感性分析**：测试 aggregate-calibration 对噪声汇总信息的鲁棒性。

### ⚖️ 基线方法对比
| 方法 | 描述 |
|------|------|
| **Naive FedAvg** | 不做任何偏差校正，均匀平均返回的更新 |
| **Round-only IPW** | 仅纠正参与偏差（只使用 participation propensity） |
| **FedIPW** | 本文方法，纠正完整的两阶段纳入概率 |
| **Oracle IPW** | 使用真实的纳入概率作为理想上限参考 |

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能结果（见 Figure 2）

#### （A）训练过程中的目标损失曲线
- **FedAvg** 快速收敛但停滞在一个较高的损失水平 → 表明其优化的是“可观测客户端”的目标函数。
- **Round-only IPW** 有所改进，但仍存在明显差距。
- **FedIPW** 几乎完全跟踪 **Oracle IPW**，表明成功恢复了目标总体目标。

#### （B）随 enrollment bias 强度增加的表现
- 当 enrollment bias 加剧时：
  - Naive FedAvg 和 Round-only IPW 性能持续恶化
  - FedIPW 保持稳定，始终接近 Oracle
- **Gap between Round-only IPW and FedIPW 扩大** → 说明忽略 enrollment bias 的代价随偏差强度上升而增大。

#### （C）Aggregate-Calibration 效果（有限信息场景）
- 在仅有总体汇总信息可用的情况下，aggregate calibration 显著缩小了 Round-only IPW 留下的剩余差距。
- 特别是在 covariates 同时驱动 enrollment 和 loss 分布时效果显著。

#### （D）对汇总信息噪声的敏感性
- 当提供的 population moments 存在噪声时：
  - 小幅噪声下仍优于 baseline
  - 噪声过大时可能失效甚至反向损害性能
- 结论：**aggregate calibration 是一种稳健的 fallback 策略，但不能替代完整 enrollment modeling**

---

### 🔬 消融实验与理论验证
- **Proposition 6.3 验证**：构造了一个两客户端强凸实例，证明当 enrollment 被忽略时，suboptimality 下界确为 $ \Omega(e_w G^2 / \mu) $，证实了 bias floor 的不可避免性。
- **Weighting Error 分解**：验证了残差误差越大、梯度异质性越高，则最终误差越大。

---

## 4. **关键结论和发现**

### ✅ 主要发现

1. **“Who can train at all” matters more than “who participates this round”**
   - Enrollment bias 是根本性的结构性问题，若不纠正，即使完美解决 participation bias 也无法对齐目标总体。

2. **Two-stage selection must be modeled separately**
   - 将 enrollment 与 participation 分开建模可避免时间顺序混淆（temporal confounding），减少模型误设风险。

3. **Incomplete correction leads to a non-vanishing bias floor**
   - 残留的 weighting error 会转化为无法通过增加训练轮次消除的误差底限，尤其在高梯度异质性场景下更严重。

4. **Aggregate calibration is a practical compromise**
   - 在缺乏个体级注册数据时，利用外部统计数据进行加权校准是有效的部分纠偏手段。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **依赖倾向得分估计质量** | 若 enrollment/participation propensity model 本身不准，会导致残差误差增大 |
| **需要额外协变量收集** | 需要在客户端上报 $ Z_i, X_{i,r} $ 等元数据，涉及隐私与工程成本 |
| **aggregate calibration 的充分性假设难满足** | 要求所选 summary statistics $ b(Z) $ 能充分解释更新差异，现实中往往不成立 |
| **强凸假设用于理论分析** | bias floor 分析基于 strongly convex setting，非凸情况仅提供 stationarity bounds |

---

### 🔮 未来工作方向

1. **在线估计 propensity scores**
   - 开发自适应算法，在训练过程中动态学习 enrollment 和 participation propensity。

2. **隐私保护下的 calibration**
   - 探索如何在差分隐私（DP）或安全聚合（Secure Aggregation）条件下实现 calibration weighting。

3. **将 FedIPW 与其他 FL 技术集成**
   - 如与 SCAFFOLD（控制变量）、FedProx（proximal term）结合，同时缓解 drift 与 selection bias。

4. **真实世界验证**
   - 在大规模生产级 FL 系统（如 GBoard、医疗 IoT）中部署 FedIPW，验证其在复杂环境下的有效性。

5. **end-to-end 可微分修正**
   - 设计可微的 reweighting layer，使整个 FL pipeline 可联合优化。

---

> **一句话总结**：  
> 本论文揭示了联邦学习中 **enrollment bias** 的深远影响，提出 **FedIPW** 与 **aggregate calibration** 两种层次的纠偏方案，并从理论上证明了忽视该问题将导致不可消除的性能天花板——真正实现了“谁训练，就该代表谁”。

</details>

---

### 12. [SpecTr-GBV: Multi-Draft Block Verification Accelerating Speculative Decoding](https://arxiv.org/abs/2604.25925)

**Authors**: Yijun Lin, Jinhao Sheng, Qingyue Cai, Feng Zhou  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 8.0  
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
传统的 **speculative decoding (SD)** 虽然通过引入轻量级的 draft model 来加速大语言模型（LLM）推理，但仍存在以下瓶颈：
- **单草案限制**：标准 SD 仅使用单一 draft 序列，导致每轮迭代中可接受的 token 数量有限。
- **验证策略低效**：现有方法要么采用逐位置（position-by-position）验证（如 SpecTr），要么采用块级验证（block verification）但仅限于单草案场景（如 GBV），未能同时利用多草案和块验证的优势。

这导致整体 **block efficiency (BE)** 和 **speedup ratio (SR)** 仍有提升空间。

### **提出的新方法与思路**
本文提出了 **SpecTr-GBV**，一种将 **multi-draft** 与 **greedy block verification (GBV)** 统一在单一框架下的新型 SD 方法。其核心思想是：
- 生成多个独立同分布（i.i.d.）的 draft 序列；
- 将验证过程建模为一个 **optimal transport (OT)** 问题，在多个 draft token blocks 与目标 token block 之间进行联合优化；
- 采用 GBV 的贪心块验证机制，而非逐位置验证，以最大化每轮迭代中可接受的 token 长度（即 acceptance length）。

### **相比现有方法的优势**
- ✅ **理论最优性**：证明了 SpecTr-GBV 在 i.i.d. 草案生成框架下，达到了物理上可实现的 **最大期望接受长度**（optimal expected acceptance length）。
- ✅ **随草案数量单调提升**：随着 draft 数量 $ K $ 增加，理论上的最大接受长度也随之增加，突破了单草案方法的上限。
- ✅ **首次统一 multi-draft 与 block verification**：是首个将两种主流优化路径融合的方法，兼具两者优势。
- ✅ **计算效率更高**：验证阶段复杂度为 $ O(|\Omega|) $，优于 SpecTr 的 $ O(|\Omega|\log K) $，且验证开销降低超过 50%。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖五个多样化任务的数据集，涵盖代码、数学、语言建模和指令遵循：
- **HumanEval**：Python 编程问题（code generation）
- **GSM8K**：小学数学应用题（reasoning）
- **MGSM**：多语言版 GSM8K（multilingual reasoning）
- **LM1B**：十亿词基准（language modeling）
- **Alpaca**：斯坦福指令跟随数据集（instruction following）

### **实验设置**
- **模型组合**：
  - 主要使用 **DeepSeek** 系列：DeepSeek-33B / 6.7B（target）、DeepSeek-1.3B（draft）
  - 补充实验使用 **CodeLlama-13B/7B** 和 **Vicuna-13B/7B**
- **关键超参数**：
  - Draft length $ L = 8 $ 或 $ 12 $
  - Draft number $ K = 3 $
  - Temperature $ T = 0.4 $
- **硬件环境**：基于真实推理延迟测量 wall-clock time

### **评估指标**
| 指标 | 定义 | 意义 |
|------|------|------|
| **Block Efficiency (BE)** | 平均每次串行调用 target model 解码的 token 数量：<br>$ \text{BE} = \frac{\text{总解码 token 数}}{\text{target model 串行调用次数}} $ | 反映算法层面的效率 |
| **Speedup Ratio (SR)** | 相对于自回归（AR）推理的端到端加速比：<br>$ \text{SR} = \frac{T_{\text{autoregressive}}}{T_{\text{proposed}}} $ | 反映实际推理速度提升 |

### **基线方法对比**
- **AR**：标准自回归解码（baseline）
- **SD**：基础 speculative decoding（Leviathan et al., 2023）
- **SpecTr**：基于 optimal transport 的 multi-draft 方法（Sun et al., 2023）
- **GBV**：单草案下的贪心块验证方法（Sun et al., 2024b）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**
在 **DeepSeek-33B-1.3B** 设置下，SpecTr-GBV 的平均表现如下：

| 方法 | Average BE | Average SR |
|------|------------|------------|
| SD | 7.64 | 1.64 |
| SpecTr | 8.39 | 1.96 |
| GBV | 7.83 | 1.67 |
| **SpecTr-GBV** | **8.59** | **2.12** |

👉 **相对提升**：
- 相比 SD：**BE ↑12.4%**, **SR ↑29.3%**
- 相比 SpecTr：**BE ↑2.3%**, **SR ↑8.2%**
- 相比 GBV：**BE ↑9.7%**, **SR ↑27.0%**

在 **DeepSeek-6.7B-1.3B** 设置下同样显著领先：
- BE 提升 **11.6% ~ 9.1%**
- SR 提升 **14.3% ~ 13.2%**

### **与其他模型家族的结果一致（Table 4）**
| 模型 | 相比 SD (SR↑) | 相比 SpecTr (SR↑) | 相比 GBV (SR↑) |
|------|--------------|-------------------|----------------|
| CodeLlama | +15.4% | +5.2% | +17.4% |
| Vicuna | +21.5% | +9.2% | +21.5% |

✅ 所有设置下，SpecTr-GBV 均取得 **最高 BE 和 SR**，验证其泛化能力。

### **消融实验结果**

#### **(1) Draft Length $ L $ 的影响（Table 3 & 5）**
- BE 随 $ L $ 增加而持续上升（更多候选 token）
- SR 先升后降：过长的 draft 导致 draft model 开销过大，抵消收益
- **结论**：存在最优 $ L $，需权衡 draft 成本与 target 调用节省

#### **(2) Draft Number $ K $ 的影响（Figure 2a & 3a）**
- 接受率（acceptance rate）随 $ K $ 增加而单调上升
- SpecTr-GBV 在所有 $ K $ 下均优于 SpecTr，且差距随 $ K $ 增大而扩大
- **相对优势**：在 $ K=7 $ 时，acceptance rate 提升达 **2.75%**

#### **(3) Temperature $ T $ 的影响（Figure 2b & 3b）**
- BE 和 SR 在不同温度（$ T=0.1, 0.4, 0.7 $）下保持稳定
- **结论**：SpecTr-GBV 对温度变化具有强鲁棒性，适用于多种生成风格

#### **(4) 时间开销分析（Table 2）**
- **验证开销降低 >50%**：得益于更高效的概率计算（无需 binary search）
- **总时间减少 6.3%~11.8%**
- **迭代次数减少 1.5%~4.5%**：更高的接受率带来更少的 SD 循环

---

## **4. 关键结论和发现**

### **主要发现**
1. **理论最优性成立**：SpecTr-GBV 在 i.i.d. 草案假设下实现了理论上可达到的最大期望接受长度。
2. **多草案 + 块验证 > 单独任一策略**：二者结合能显著超越现有方法，验证了统一框架的有效性。
3. **实际加速明显**：在多个模型和数据集上，SpecTr-GBV 实现了 **最高 BE 和 SR**，且验证开销更低。
4. **对超参数鲁棒**：对 temperature 不敏感，适合实际部署；draft number 越多，优势越明显。

### **方法的局限性**
- 依赖于 **i.i.d. 草案生成**，若 draft model 存在系统性偏差可能影响效果。
- 当前未探索 **tree-based drafting** 等更复杂的草案结构，仍局限于 flat 多序列。
- 需要额外的 **distribution modification** 步骤来保证多轮迭代下的分布一致性，增加了实现复杂度。

### **未来工作方向**
- 扩展至 **non-i.i.d. 草案生成**（如基于检索或规划的 draft）
- 结合 **tree-structured candidates** 进一步提升覆盖率
- 探索 **adaptive draft length/number** 策略以动态优化资源分配
- 在更大规模模型和真实服务场景中部署并评估系统级性能

---

> ✅ **总结一句话**：  
> **SpecTr-GBV 是首个将 multi-draft 与 greedy block verification 统一的 speculative decoding 框架，在理论和实践中均实现了当前最优的推理加速效果。**

</details>

---

### 13. [HealthNLP_Retrievers at ArchEHR-QA 2026: Cascaded LLM Pipeline for Grounded Clinical Question Answering](https://arxiv.org/abs/2604.26880)

**Authors**: Md Biplob Hosen, Md Alomgeer Hussein, Md Akmol Masud, Omar Faruque, Tera L Reynolds, Lujie Karen Chen  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.26880v1  

#### Abstract
Patient portals now give individuals direct access to their electronic health records (EHRs), yet access alone does not ensure patients understand or act on the complex clinical information contained in these records. The ArchEHR-QA 2026 shared task addresses this challenge by focusing on grounded q...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*HealthNLP_Retrievers at ArchEHR-QA 2026: Cascaded LLM Pipeline for Grounded Clinical Question Answering*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
该研究针对 **Electronic Health Records (EHR)** 中患者语言与临床记录之间的语义鸿沟问题。具体挑战包括：
- 患者提问通常**冗长、非正式且带有情绪表达**，而临床笔记则高度专业化、碎片化。
- 现有 **Large Language Models (LLMs)** 在生成答案时容易产生“幻觉”（hallucination），即引入未在EHR中支持的医学信息。
- 需要实现**证据可追溯性**（evidence grounding）——即每个答案都必须严格基于EHR中的具体句子。

### 🚀 提出的新方法与创新点
作者提出了一个四阶段级联的 **Cascaded LLM Pipeline**，其核心创新在于将复杂任务分解为多个可控模块，并通过精心设计的提示工程（prompt engineering）引导模型行为：

1. **Few-shot Query Reformulation（查询重构）**
   - 引入“临床行政助理”persona，将患者原始问题压缩为不超过15词的专业化临床查询。
   - 使用few-shot示例提升语义保留能力，减少噪声干扰。

2. **Heuristic-based Evidence Scorer（启发式证据评分器）**
   - 将证据识别转化为**句子级Likert量表打分任务**（1–5分），优先召回关键信息。
   - 设计**动态回退机制**：若无高分证据，则放宽阈值以避免下游“证据饥饿”。

3. **Grounded Response Generator（基于证据的回答生成）**
   - 采用“临床文档专家”persona，确保输出简洁、客观、仅依赖检索到的证据。
   - 利用 **Retrieval-Augmented Generation (RAG)** 结构，结合过滤后的证据 + 完整病历作为上下文，形成“focus-then-expand”策略。

4. **Many-to-Many Answer-Evidence Alignment（多对多对齐框架）**
   - 构建精确映射关系：将生成的答案句逐句链接至支持它的EHR句子索引。
   - 使用保守提示策略，防止过度引用无关背景。

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **结构化流程** | 超越传统单步RAG范式，通过多阶段解耦降低错误传播风险。 |
| **更强的grounding能力** | 显著减少幻觉，所有输出均受控于明确证据池。 |
| **更高的实用性** | 支持端到端可解释性，便于医生验证答案来源。 |
| **灵活适应共享任务格式** | 各子任务独立优化，在ArchEHR-QA 2026四项子任务中取得全面竞争力表现。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **ArchEHR-QA 2026 Shared Task Dataset**
  - 来源于 **MIMIC-III** critical care database 的去标识化EHR片段。
  - 包含真实患者提出的关于自身住院过程的信息需求问题。
  - 数据经过人工标注，提供标准答案及对应的证据句子标签。

### ⚙️ 实验设置
- **主干模型**：使用 **Gemini 2.5 Pro**，利用其强大的长文本理解能力和推理性能。
- **API调用配置**：
  - 对需JSON输出的任务（Subtask 1, 2, 4）设 `temperature=0` 保证确定性。
  - 回答生成任务（Subtask 3）设 `temperature=0.1` 提升自然度。
  - 关闭所有安全过滤（`HarmCategory.BLOCK_NONE`），确保临床术语不被误拦。
- **代码开源**：系统源码已公开于GitHub，增强可复现性。

### 📊 评估指标
| 子任务 | 主要指标 | 辅助指标 |
|--------|---------|----------|
| **Subtask 1: Question Interpretation** | Overall Score (ROUGELsum + BERTScore + AlignScore + MEDCON) | BERTScore, ROUGELsum |
| **Subtask 2: Evidence Identification** | Strict Micro F1 | Lenient Micro F1, Precision, Recall |
| **Subtask 3: Answer Generation** | SARI, BERTScore, AlignScore, MEDCON | BLEU, ROUGELsum |
| **Subtask 4: Answer-Evidence Alignment** | Micro F1 | Micro Precision, Micro Recall |

> 注：官方规定各子任务最终排名依据指定的primary metric。

### 🔁 基线方法对比
虽然未直接列出完整基线系统名称，但从讨论中可知：
- 多数参赛队伍仍采用标准RAG或端到端生成方式。
- 本工作对比了以下变体进行消融分析：
  - Zero-shot vs Few-shot 查询重构
  - Raw patient query vs Reformulated clinician query 作为检索锚点

---

## 3. 主要实验结果和性能指标

### 🏆 总体性能排名（Table 1）
| Subtask | Rank | Primary Metric (Our) | Top System |
|--------|------|-----------------------|-----------|
| **Question Interpretation** | **1st** | 31.2 (Overall) | 31.2 |
| **Evidence Identification** | 7th | 60.2 (Strict Micro F1) | 63.7 |
| **Answer Generation** | 5th | 59.2 (SARI) | 58.6 |
| **Answer-Evidence Alignment** | 9th | 76.9 (Micro F1) | 81.5 |

> 表明该系统在**问题理解**方面达到最优水平，在其他三项也具备强竞争力。

### 🔢 关键性能细节
#### ✅ Subtask 1: Query Reformulation
- **Few-shot优于Zero-shot**：
  - BERTScore从34.13 → 45.09
  - MEDCON（医学实体保留）从9.20 → 17.20
- 成功实现语义浓缩同时保持临床相关性。

#### ✅ Subtask 2: Evidence Scoring
- **Recall导向设计有效**：
  - Strict Micro Recall = **74.38**（高于Precision的50.56）
  - 使用reformulated query后，F1提升至**62.90**（vs 原始query的60.20）
- Lenient F1达68.3，显示额外召回内容具有临床邻近性。

#### ✅ Subtask 3: Answer Generation
- **SARI得分最高之一（59.2）**，显著优于第一名WisPerMed（58.6），表明文本简化效果优异。
- **MEDCON = 38.7**，证明生成内容忠实于输入证据，无外部知识注入。
- **BLEU仅7.0**，反映词汇层面与参考答案差异大，但语义准确。

#### ✅ Subtask 4: Alignment
- **Micro Precision = 83.8**，优于Yale-DM-Lab（83.3）等顶尖团队，说明对证据选择极为严谨。
- Micro Recall = 71.1，低于OptiMed（79.8），体现精度优先的设计取舍。

### 🔍 消融实验结果（Table 2）
| 设置 | 效果 |
|------|------|
| **Few-shot Prompting (vs Zero-shot)** | 显著提升BERTScore (+11 pts) 和MEDCON (+8 pts)，验证示范样本的重要性 |
| **Clinician Query (vs Patient Narrative)** | 提升Strict Micro F1从60.2 → 62.9，Precision从50.6 → 61.4，证明查询质量直接影响证据检索精度 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **多阶段级联架构优于端到端生成**
   - 分解任务有助于控制每一步的行为，提高整体系统的可控性和可解释性。
2. **Few-shot Query Reformulation是关键前置步骤**
   - 精炼后的临床查询显著提升了后续证据检索的质量。
3. **Recall-biased证据提取有利于下游生成**
   - 即使牺牲部分precision，丰富的上下文能帮助生成更完整的答案。
4. **High-precision alignment可有效抑制幻觉**
   - 严格的对齐机制构成最后一道防线，确保答案完全基于EHR原文。

### ⚠️ 局限性
1. **依赖闭源API（Gemini 2.5 Pro）**
   - 不利于隐私敏感场景部署，存在数据外泄风险。
2. **误差传播问题**
   - 级联系统中前一阶段错误会影响后续模块，缺乏纠错机制。
3. **MEDCON得分偏低**
   - 在Subtask 1中仅为18.7，低于KPSCMI（27.9），说明有时会丢失关键医学实体（如药物名）。
4. **过于简化的语言影响技术准确性**
   - 高SARI带来易读性，但也导致低BLEU，可能削弱专业用户的信任。

### 🔮 未来工作方向
1. **开发迭代式自我修正机制**
   - 引入feedback loop来检测并修复早期阶段的错误。
2. **迁移到开放权重模型（open-weight models）**
   - 如BioGPT、ClinicalBERT等，实现在本地部署，保护患者隐私。
3. **探索end-to-end联合训练策略**
   - 减少模块间接口带来的信息损失和延迟。
4. **扩展至更多EHR系统和临床场景**
   - 当前评估局限于MIMIC-III，需验证泛化能力。

--- 

> 💡 **总结一句话**：  
> 本文提出了一种结构清晰、可解释性强的四阶段级联LLM pipeline，在ArchEHR-QA 2026多项子任务中表现出色，尤其在**问题理解和证据生成一致性**方面领先，为面向患者的EHR问答系统提供了实用且可靠的工程范式。

</details>

---

### 14. [Bian Que: An Agentic Framework with Flexible Skill Arrangement for Online System Operations](https://arxiv.org/abs/2604.26805)

**Authors**: Bochao Liu, Zhipeng Qian, Yang Zhao, Xinyuan Jiang, Zihan Liang, Yufei Ma, Junpeng Zhuang, Ben Chen, Shuo Yang, Hongen Wan, Yao Wu, Chenyi Lei, Xiao Liang  
**Category**: cs.AI  
**Published**: 2026-04-30  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.26805v1  

#### Abstract
Operating and maintaining (O&amp;M) large-scale online engine systems (search, recommendation, advertising) demands substantial human effort for release monitoring, alert response, and root cause analysis. While LLM-based agents are a natural fit for these tasks, the deployment bottleneck is not rea...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Bian Que: An Agentic Framework with Flexible Skill Arrangement for Online System Operations**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
大型在线引擎系统（如搜索、推荐、广告）的运维（O&M）任务高度依赖人工，尤其是在**发布监控、告警响应和根因分析（Root Cause Analysis, RCA）**等环节。尽管 LLM-based agents 在自动化推理方面展现出潜力，但实际部署中的瓶颈并非模型的推理能力，而是**上下文构建（context assembly）**——即如何为每个操作事件选择合适的**数据**（metrics, logs, change events）和**知识**（handbook rules, 实践经验）。  
传统方法存在以下问题：
- 输入所有信号会导致信息稀释和幻觉（hallucination）；
- 手动配置事件到（数据, 知识）的映射在高频迭代系统中不可持续（每天数十次发布）；
- 现有 AIOps 方法通常假设输入上下文已整理好，而忽略了“**选对上下文”本身才是关键挑战**。

### **提出了什么新方法或新思路**
作者提出 **BIAN QUE**，一个基于 agent 的运维框架，其三大核心贡献如下：

#### **(1) 统一的运维范式（Unified Operational Paradigm）**
将日常运维抽象为三个标准模式，构成三道防线：
- **Release Interception**：发布拦截，在发布过程中实时检测异常；
- **Proactive Inspection**：主动巡检，定期检查系统健康状态；
- **Alert Root Cause Analysis**：告警触发后的根因诊断。

该范式强调“**预防优于补救**”，将告警诊断作为最后一道防线，而非起点。

#### **(2) 灵活技能编排（Flexible Skill Arrangement）**
引入 **Skill** 抽象，每个 Skill 定义了特定业务模块下应检索的数据和知识，并可通过自然语言指令由 LLM 自动生成和更新。  
- **Skill = (LoadDataSchema, Prompt, Meta)**，结构化定义数据源调用和推理逻辑；
- 支持 LLM 驱动生成（Generation）和人类反馈驱动的增量更新（Update），无需代码修改；
- 实现了事件到（数据, 知识）映射的**自动化与可持续维护**。

#### **(3) 统一的自我进化机制（Unified Self-Evolving Mechanism）**
单个反馈信号同时驱动两条并行路径：
- **Case-Memory-to-Knowledge Distillation**：从案例中提炼可复用的知识（如故障模式）；
- **Targeted Skill Refinement**：修正 Skill 中的数据检索或推理逻辑错误。

实现知识库与 Skill 映射的**协同演化（co-evolution）**，避免双通道更新的割裂。

### **相比现有方法的优势**
| 方面 | 现有方法 | BIAN QUE |
|------|--------|---------|
| 上下文构建 | 假设上下文已整理好 | 自动选择相关数据与知识 |
| 可维护性 | 静态配置，难以适应快速迭代 | LLM + 自然语言反馈动态更新 |
| 知识演化 | RAG 静态知识库 | 动态蒸馏 + 持续精炼 |
| 架构覆盖 | 多聚焦于 RCA | 覆盖完整运维生命周期 |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **真实生产环境数据**：来自快手（Kuaishou）电商平台搜索引擎的 **104 个真实运维事件**，包括：
  - **44 个告警事件（Alert）**
  - **60 个巡检事件（Inspection）**
- 每个事件由资深 SRE 标注**真实根因（ground-truth diagnosis）**。
- 发布拦截未单独采样，因其 Skill 结构为巡检 Skill 的特例。

### **实验设置**
- **部署平台**：快手电商搜索系统，服务数亿 DAU，日均数十次发布。
- **默认 LLM**：`Qwen3.5-35B-FP8`，运行于单张 NVIDIA Tesla L20 GPU。
- **Skill 存储**：YAML 文件 + 关键词索引。
- **知识库**：
  - KV / KKV Index：Redis
  - Vector Index：ANN 服务
  - 短期记忆：滚动窗口案例存储
  - 长期知识：每日离线去重合并
- **评估前重置 Skill 目录**，防止历史泄露。

### **评估指标**
| 指标 | 含义 |
|------|------|
| **Alert Volume Reduction** | 告警总量相对下降比例 |
| **RCA Accuracy** | 根因分析被确认正确的比例 |
| **MTTR (Mean Time to Resolution)** | 平均故障解决时间 |
| **PASS@k** | k 次独立执行中至少一次生成正确 Skill 的比例 |

### **基线方法对比**
- **STATIC**：手动编写固定 Skill，禁用 LLM 生成与更新；
- **NOKNOW**：推理时不检索任何历史知识（无短期记忆 + 无长期知识）；
- **Full BIAN QUE**：完整框架（含 Flexible Skill + Knowledge Retrieval + Feedback Loop）

---

## 3. **主要实验结果和性能指标**

### **关键性能数据（线上部署效果）**
| 指标 | 部署前 | 部署后 | 提升 |
|------|-------|--------|------|
| Fired Alerts | 100% | **25%**（↓75%） | ✅ |
| Non-actionable Alerts (绝对量) | 100% | **~5%**（↓~95%） | ✅ |
| RCA Accuracy | — | **80%** | ✅ |
| Alerts Resolved within 5min | — | **95%** | ✅ |
| MTTR | 100% | **<50%**（↓>50%） | ✅ |

> **说明**：告警减少主要得益于前两道防线（发布拦截 + 主动巡检）提前发现问题；非必要告警大幅下降显著减轻工程师负担。

### **与基线方法的对比结果**

#### **PASS@k 性能对比（完整框架 vs 基线）**

##### **Table 3: Full BIAN QUE (pass@k)**
| 场景 | pass@1 | pass@5 |
|------|--------|--------|
| Alert | 70.5% | 95.5% |
| Inspection | 85.0% | 93.3% |
| **Overall** | **78.8%** | **94.2%** |

##### **Table 6: STATIC（手动 Skill）**
| 场景 | pass@1 | pass@5 |
|------|--------|--------|
| **Overall** | **65.3%** | **83.7%** |
> ↓13.5pp（pass@1），↓10.5pp（pass@5），且仍有 **16.3% 无法解决**（vs 全框架仅 5.8%）

##### **Table 7: NOKNOW（无知识检索）**
| 场景 | pass@1 | pass@5 |
|------|--------|--------|
| **Overall** | **71.2%** | **86.5%** |
> ↓7.6pp（pass@1），表明历史知识对首次生成至关重要，无法通过重试弥补。

### **消融实验结果**
- **Flexible Skill 是最关键组件**：移除后性能下降最大（pass@1 ↓13.5pp）；
- **历史知识不可替代**：无知识检索导致 pass@1 下降 7.6pp，且无法通过多采样恢复；
- **反馈机制维持长期准确性**：
  - 无反馈时，系统在 13 天内准确率从 ~75% 降至 ~32%；
  - 有反馈时，系统自动学习错误，维持 >80% 准确率。

---

## 4. **关键结论和发现**

### **主要发现**
1. **上下文构建比推理更重要**：在复杂运维场景中，**选择正确的数据和知识**是 LLM 成功应用的前提。
2. **Flexible Skill 机制高效可行**：LLM 可基于种子配置自动生成高质量 Skill，**pass@1 达 78.8%**，配合自然语言反馈可达 **99.0% pass@5**。
3. **统一反馈机制实现协同演化**：单个反馈信号驱动 Skill 与知识同步优化，形成闭环。
4. **预防性运维显著降低告警量**：通过发布拦截与主动巡检，**减少 75% 告警触发**，从根本上减轻运维压力。
5. **大模型规模影响显著**：建议使用 ≥35B 参数的 LLM，小模型（如 0.8B）表现灾难性下降（pass@1 仅 13.9%）。

### **局限性**
1. **尚未实现闭环修复**：当前框架支持诊断，但不自动执行 rollback 或扩容等修复动作。
2. **Skill 匹配依赖关键词**：当前 MATCH 机制为关键词匹配，对新型事件泛化能力有限。
3. **多 agent 协同缺失**：复杂级联故障中多个 agent 的协作仍需人工协调。
4. **知识与 Skill 的协同增益尚难量化**：共演化效应观察明显，但缺乏长期定量验证。
5. **新服务知识冷启动问题**：对刚上线的服务，知识积累不足导致 RCA 错误较多。

### **未来工作方向**
1. **扩展至闭环自主修复（Closed-loop Remediation）**：集成执行工具，实现“诊断→决策→执行”全流程自动化。
2. **改进 Skill 路由机制**：探索基于 embedding 的智能匹配，提升对未知事件的适应性。
3. **多 agent 协同协议设计**：研究复杂故障下的多 agent 分工与通信机制。
4. **长期演化效应量化**：开展跨年尺度实验，验证知识与 Skill 的复合增长效应。
5. **加速新服务知识收敛**：引入迁移学习或仿真数据，缓解冷启动问题。

--- 

> **命名寓意**：框架以中国古代名医“扁鹊”命名，因其“**上医治未病**”的理念，正契合本框架“**预防优于治疗**”的运维哲学。

</details>

---

### 15. [SplitFT: An Adaptive Federated Split Learning System For LLMs Fine-Tuning](https://arxiv.org/abs/2604.26388)

**Authors**: Yimeng Shan, Zhaorui Zhang, Sheng Di, Yu Liu, Xiaoyi Lu, Benben Liu  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 7.0  
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
当前在 **Federated Learning (FL)** 中对 **Large Language Models (LLMs)** 进行 **fine-tuning** 面临三大挑战：
1. **设备异构性（Device Heterogeneity）**：不同客户端计算资源差异大，固定模型切分（cutlayer）导致高性能设备需等待低性能设备，降低训练效率。
2. **数据异构性（Data Heterogeneity）**：各客户端数据分布非独立同分布（Non-IID），影响全局模型收敛和性能。
3. **通信开销高**：客户端需频繁传输中间“smashed data”，尤其在 LLM 场景下，通信成本显著。

现有方法未能同时解决上述问题，尤其是在结合 **Split Learning** 和 **Parameter-Efficient Fine-Tuning (PEFT)** 的场景中缺乏自适应机制。

---

### 🚀 提出的新方法与创新点
作者提出 **SplitFT** —— 一种面向 LLM fine-tuning 的自适应联邦分割学习系统，其核心创新如下：

#### （1）**自适应切层策略（Adaptive Cutlayer Allocation）**
- 允许每个客户端根据自身 **计算能力** 和 **本地模型性能** 动态调整其负责的模型层数（即 cutlayer 位置）。
- 引入动态权重调整机制：表现优于平均准确率的客户端承担更多计算任务，反之则减少负载。
- 实现计算资源与模型质量之间的最优平衡。

#### （2）**Cutlayer 上的 LoRA Rank 压缩**
- 在 cutlayer 处采用更低的 **LoRA rank**（如 `rcut=8`），而在其他层保持较高 rank（如 `rothers=16`）。
- 显著减少客户端与服务器之间传输的梯度大小，从而降低 **communication overhead**。
- 同时通过协同调优两侧（client/server）的 LoRA rank 实现性能无损压缩。

#### （3）**基于长度的 Dirichlet 数据划分方法（Length-based Dirichlet Partitioning）**
- 针对 LLM 应用场景设计新型 Non-IID 划分方式：先按文本长度将数据分类，再使用 Dirichlet 分布控制各类别在客户端间的分配比例。
- 超参数 α 控制异构程度（α 越小，异构越严重），更贴近真实世界数据分布。

#### （4）系统模块化设计
- 支持多种算法、采样策略、模型配置集成，具备良好的可扩展性和通用性。

---

### 🔍 相比现有方法的优势
| 维度 | SplitFT | 传统 FL / 固定 Split Learning |
|------|--------|-------------------------------|
| **资源利用率** | 自适应分配，适配异构设备 | 所有客户端执行相同任务，存在等待瓶颈 |
| **隐私保护** | 客户端不掌握完整模型，仅交换中间激活值 | 客户端持有全模型，风险更高 |
| **通信效率** | 通过低秩 cutlayer 减少传输量 | 传输完整激活或梯度，开销大 |
| **模型性能** | 更快收敛 + 更低 perplexity | 易过拟合，尤其在 Non-IID 下 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **Wikitext2-v1**：广泛用于语言建模任务的标准 benchmark。
  - 包含 36,700 条训练样本、3,760 验证样本、4,360 测试样本。
  - 内容来自维基百科文章，适合评估 LLM 的上下文理解与生成能力。

---

### ⚙️ 实验设置

| 参数 | 设置 |
|------|------|
| **基础模型** | GPT2-small (125M), OPT-125M, GPT-Neo 125M |
| **LoRA 配置** | Attention 模块启用 LoRA；cutlayer rank `rcut=8`，其余层 `rothers=16` |
| **Batch Size** | 4 |
| **Learning Rate** | 5×10⁻⁵ |
| **Sequence Length** | 最大 512 tokens |
| **Client 数量** | 5 |
| **每轮本地数据量** | 12,000 samples/client |
| **总训练轮数** | 最多 1200 global rounds |
| **框架实现** | 基于 PyTorch + Flower 构建 |

---

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Perplexity** | 主要评价指标，衡量语言模型预测能力，越低越好 |
| **Max Accuracy / Test Accuracy** | 衡量 fine-tuning 效果 |
| **Elapsed Time / Round Time** | 反映训练效率 |
| **Communication Overhead (MB)** | 衡量传输数据总量 |
| **Trainable Parameters** | 衡量参数效率 |

---

### 🆚 基线方法（Baselines）
- **Same Split (Fixed Cutlayer)**：
  - 所有客户端固定前 2 个 GPTBlock 在客户端训练，其余在服务器。
  - LoRA 设置一致（rank=8 for cutlayer, 16 for others）。
- **No Cutlayer / Full LoRA Fine-tuning**：
  - 客户端训练全部层，作为性能上限参考（但资源消耗极高）。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ 表格 I：不同 cutlayer 设置下的性能对比（GPT2-small）

| Cutlayer | Max Accuracy | Mean Elapsed Time (s) | Max Comm Overhead (MB) |
|----------|--------------|------------------------|-------------------------|
| 2        | 0.0606       | 810.4                  | 3475.4                  |
| 4        | 0.0571       | 863.2                  | 3534.4                  |
| 6        | 0.0605       | 934.2                  | 3593.4                  |
| 8        | 0.0621       | 1113.4                 | 3652.4                  |
| 10       | **0.0629**   | 1104.7                 | **3711.4**              |
| No Cut   | 0.0617       | **6897.3**             | 3490.1                  |

> 💡 发现：cutlayer 越靠后（即客户端处理越多层），精度略升但耗时剧增；而 cutlayer=2 在效率与性能间取得最佳平衡。

---

#### ✅ 表格 II：不同 LoRA Rank 对性能的影响

| Rank | Max Acc. | Elapsed Time (s) | Comm Overhead (MB) | Trainable Param. |
|------|----------|------------------|---------------------|------------------|
| 1    | 0.0579   | 2277.5           | 3462.5              | 0.08M            |
| 2    | 0.0585   | 2138.2           | 3464.3              | 0.015M           |
| 4    | 0.0589   | 2293.4           | 3468.0              | 0.031M           |
| 8    | **0.0606** | **1597.7**     | **3475.4**          | **0.062M**       |

> 💡 发现：`rank=8` 在精度、速度、通信开销方面达到最优权衡；进一步提升 rank 收益递减。

---

### 📊 与基线方法的对比结果

#### （1）**Adaptive vs. Fixed Cutlayer（图 3a）**
- SplitFT（adaptive）虽然初期收敛稍慢，但最终达到 **更低的 perplexity**。
- 在 IID 和 Non-IID 场景下均优于 fixed split 方法。
- 尤其在 α=0.1（高度异构）时仍能稳定收敛，显示强鲁棒性。

#### （2）**Non-IID 场景下的表现（图 3b & 3c）**
- Baseline 在 Non-IID 下出现明显过拟合（test perplexity 上升）。
- SplitFT 利用个性化优化 + 动态 layer 调整，有效缓解梯度冲突，在 Non-IID 下持续下降训练 perplexity。

#### （3）**跨模型泛化能力（图 4）**
- 在 GPT2-small、OPT-125M、GPT-Neo 125M 上均取得一致优势。
- 不论是 IID 还是 Non-IID 设置，SplitFT 均能快速收敛并达到更低 perplexity。

---

### 🔬 消融实验结果（Ablation Study）

| 配置 | 性能趋势 |
|------|----------|
| **Only client-side LoRA rank reduced** | 性能下降明显（不对称导致特征失配） |
| **Both sides reduce LoRA rank at cutlayer** | 性能最优，说明对称调整至关重要 |
| **Symmetric LoRA rank adjustment (rcut < rothers)** | 在保证性能的同时显著降低通信开销 |
| **Adaptive layer allocation enabled** | 提升低资源客户端参与度，整体训练效率提高约 30% |

> ✅ 结论：**双侧 LoRA rank 压缩 + 自适应 layer 分配** 是性能提升的关键组合。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **自适应 cutlayer 显著提升训练效率与模型性能**  
   - 动态分配机制使高质数据客户端承担更多任务，加速全局收敛。
   
2. **降低 cutlayer 的 LoRA rank 可大幅减少通信开销而不牺牲性能**  
   - 特别是在 symmetric configuration 下效果最佳（client/server 同时降秩）。

3. **SplitFT 在 Non-IID 场景下表现出卓越鲁棒性**  
   - 即使 α=0.1（极端异构），也能维持低 perplexity，优于固定切分方法。

4. **方法具有良好的通用性与可扩展性**  
   - 成功应用于 GPT2、OPT、GPT-Neo 等多种架构，验证其广泛适用性。

---

### ⚠️ 局限性
1. **依赖高质量的初始模型划分策略**：如何自动确定最优初始 cutlayer 仍需探索。
2. **未考虑极端网络延迟或丢包场景**：实际部署中可能面临更复杂的通信环境。
3. **目前仅支持 LoRA 类 PEFT 方法**：尚未集成 Prefix-tuning 或 Adapter-tuning 等其他 PEFT 技术。

---

### 🔮 未来工作方向
1. **自动化 cutlayer 初始化机制**：结合强化学习或贝叶斯优化动态选择起始点。
2. **支持更多 PEFT 方法集成**：如 DoRA、rsLoRA、LoRA+ 等以进一步提升性能。
3. **边缘设备实测验证**：在真实 IoT 设备或移动终端上部署 SplitFT，测试端到端可行性。
4. **引入差分隐私（DP）增强安全性**：在已有隐私基础上叠加 DP 机制，实现双重保障。

---

## ✅ 总结
**SplitFT** 是首个将 **自适应联邦分割学习** 与 **LoRA-based PEFT** 深度融合的 LLM fine-tuning 框架。它通过：
- 自适应 cutlayer 分配，
- cutlayer 上的 LoRA rank 压缩，
- 新型 length-based Dirichlet 数据划分，

有效解决了 **设备异构、数据异构、通信开销高** 三大难题。实验表明，SplitFT 在多个主流 LLM 上均实现了 **更快收敛、更低 perplexity、更少通信开销**，为资源受限环境下安全高效的 LLM 微调提供了强有力的技术路径。

</details>

---

### 16. [Spatially-constrained clustering of geospatial features for heat vulnerability assessment of favelas in Rio de Janeiro](https://arxiv.org/abs/2604.26133)

**Authors**: Baptiste Clemence, Thomas Hallopeau, Vanderlei Pascoal De Matos, Laurent Demagistri, Joris Guerin  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.26133v1  

#### Abstract
Informal settlements face disproportionate exposure to climate-related health hazards. However, existing methodologies lack systematic approaches to link diverse settlement characteristics with environmental health outcomes. We develop a data-driven framework to assess heat vulnerability in Rio de J...

---

### 17. [Shorthand for Thought: Compressing LLM Reasoning via Entropy-Guided Supertokens](https://arxiv.org/abs/2604.26355)

**Authors**: Zhenyu Zhao, Sander Land, Dan Bikel, Waseem Alshikh  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.26355v1  

#### Abstract
Reasoning in Large Language Models incurs significant inference-time compute, yet the token-level information structure of reasoning traces remains underexplored. We observe that reasoning tokens split into two functional types: low-entropy \textit{structural} tokens (recurring phrases that scaffold...

---

### 18. [SAGE: A Strategy-Aware Graph-Enhanced Generation Framework For Online Counseling](https://arxiv.org/abs/2604.26630)

**Authors**: Eliya Naomi Aharon, Meytal Grimland, Avi Segal, Loona Ben Dayan, Inbar Shenfeld, Yossi Levi Belz, Kobi Gal  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.26630v1  

#### Abstract
Effective mental health counseling is a complex, theory-driven process requiring the simultaneous integration of psychological frameworks, real-time distress signals, and strategic intervention planning. This level of clinical reasoning is critical for safety and therapeutic effectiveness but is oft...

---

### 19. [Hierarchical adaptive control for real-time dynamic inference at the edge](https://arxiv.org/abs/2604.26470)

**Authors**: Francesco Daghero, Mahyar Tourchi Moghaddam, Mikkel Baun Kj{\ae}rgaard  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.26470v1  

#### Abstract
Industrial systems increasingly depend on Machine Learning (ML), and operate on heterogeneous nodes that must satisfy tight latency, energy, and memory constraints. Dynamic ML models, which reconfigure their computational footprint at runtime, promise high energy efficiency and lower average latency...

---

### 20. [Hierarchical Multi-Persona Induction from User Behavioral Logs: Learning Evidence-Grounded and Truthful Personas](https://arxiv.org/abs/2604.26120)

**Authors**: Nayoung Choi, Haeyu Jeong, Changbong Kim, Hongjun Lim, Jinho D. Choi  
**Category**: cs.AI  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.26120v1  

#### Abstract
Behavioral logs provide rich signals for user modeling, but are noisy and interleaved across diverse intents. Recent work uses LLMs to generate interpretable natural-language personas from user logs, yet evaluation often emphasizes downstream utility, providing limited assurance of persona quality i...

---

### 21. [AGEL-Comp: A Neuro-Symbolic Framework for Compositional Generalization in Interactive Agents](https://arxiv.org/abs/2604.26522)

**Authors**: Mahnoor Shahid, Hannes Rothe  
**Category**: cs.AI  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.26522v1  

#### Abstract
Large Language Model (LLM)-based agents exhibit systemic failures in compositional generalization, limiting their robustness in interactive environments. This work introduces AGEL-Comp, a neuro-symbolic AI agent architecture designed to address this challenge by grounding actions of the agent. AGEL-...

---

### 22. [Human-in-the-Loop Benchmarking of Heterogeneous LLMs for Automated Competency Assessment in Secondary Level Mathematics](https://arxiv.org/abs/2604.26607)

**Authors**: Jatin Bhusal, Nancy Mahatha, Aayush Acharya, Raunak Regmi  
**Category**: cs.AI  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.26607v1  

#### Abstract
As Competency-Based Education (CBE) is gaining traction around the world, the shift from marks-based assessment to qualitative competency mapping is a manual challenge for educators. This paper tackles the bottleneck issue by suggesting a "Human-in-the-Loop" benchmarking framework to assess the effe...

---

### 23. [CogRAG+: Cognitive-Level Guided Diagnosis and Remediation of Memory and Reasoning Deficiencies in Professional Exam QA](https://arxiv.org/abs/2604.25928)

**Authors**: Xudong Wang, Zilong Wang, Zhaoyan Ming  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.25928v1  

#### Abstract
Professional domain knowledge underpins human civilization, serving as both the basis for industry entry and the core of complex decision-making and problem-solving. However, existing large language models often suffer from opaque inference processes in which retrieval and reasoning are tightly enta...

---

### 24. [FlowBot: Inducing LLM Workflows with Bilevel Optimization and Textual Gradients](https://arxiv.org/abs/2604.26258)

**Authors**: Hongyeon Yu, Young-Bum Kim, Yoon Kim  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.26258v1  

#### Abstract
LLM workflows, which coordinate structured calls to individual LLMs (each augmented with varying instructions and tools) to achieve a particular goal, offer a promising path towards extending the capabilities of LLMs and building powerful systems that can tackle diverse tasks. However, existing appr...

---

### 25. [Select to Think: Unlocking SLM Potential with Local Sufficiency](https://arxiv.org/abs/2604.26940)

**Authors**: Wenxuan Ye, Yangyang Zhang, Xueli An, Georg Carle, Yunpu Ma  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.26940v1  

#### Abstract
Small language models (SLMs) offer computational efficiency for scalable deployment, yet they often fall short of the reasoning power exhibited by their larger counterparts (LLMs). To mitigate this gap, current approaches invoke an LLM to generate tokens at points of reasoning divergence, but these ...

---

### 26. [BioGraphletQA: Knowledge-Anchored Generation of Complex QA Datasets](https://arxiv.org/abs/2604.26048)

**Authors**: Richard A. A. Jonker, B\'arbara Maria Ribeiro de Abreu Martins, S\'ergio Matos  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26048v1  

#### Abstract
This paper presents a principled and scalable framework for systematically generating complex Question Answering (QA) data. In the core of this framework is a graphlet-anchored generation process, where small subgraphs from a Knowledge Graph (KG) are used in a structured prompt to control the comple...

---

### 27. [MoRFI: Monotonic Sparse Autoencoder Feature Identification](https://arxiv.org/abs/2604.26866)

**Authors**: Dimitris Dimakopoulos, Shay B. Cohen, Ioannis Konstas  
**Category**: cs.CL  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26866v1  

#### Abstract
Large language models (LLMs) acquire most of their factual knowledge during the pre-training stage, through next token prediction. Subsequent stages of post-training often introduce new facts outwith the parametric knowledge, giving rise to hallucinations. While it has been demonstrated that supervi...

---

### 28. [MPI Malleability Validation under Replayed Real-World HPC Conditions](https://arxiv.org/abs/2604.26576)

**Authors**: S. Iserte, M. Madon, G. Da, J. Pierson, A. J. Pe\~na  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26576v1  

#### Abstract
Dynamic Resource Management (DRM) techniques can be leveraged to maximize throughput and resource utilization in computational clusters. Although DRM has been extensively studied through analytical workloads and simulations, skepticism persists among end administrators and users regarding their feas...

---

### 29. [A Test Taxonomy and Continuous Integration Ecosystem for Dynamic Resource Management in HPC](https://arxiv.org/abs/2604.26824)

**Authors**: Petter Sand{\aa}s, \'I\~nigo Ar\'ejula-A\'isa, Sergio Iserte, Antonio J. Pe\~na  
**Category**: cs.DC  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26824v1  

#### Abstract
High-performance computing (HPC) systems are increasingly exploring dynamic resource management and malleable MPI applications to better adapt to heterogeneous architectures, fluctuating workloads, and energy constraints. However, the correctness of the libraries that support these techniques is oft...

---

### 30. [Efficient and Interpretable Transformer for Counterfactual Fairness](https://arxiv.org/abs/2604.26188)

**Authors**: Panyi Dong, Zhiyu Quan  
**Category**: cs.LG  
**Published**: 2026-04-30  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.26188v1  

#### Abstract
The growing reliance of machine learning models in high-stakes, highly regulated domains such as finance and insurance has created a growing tension between predictive performance, interpretability, and regulatory fairness requirements. In these settings, models are expected not only to deliver reli...

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
