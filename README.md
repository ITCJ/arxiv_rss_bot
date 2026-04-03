# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-03 06:54:30 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Universal YOCO for Efficient Depth Scaling](https://arxiv.org/abs/2604.01220)

**Authors**: Yutao Sun, Li Dong, Tianzhu Ye, Shaohan Huang, Jianyong Wang, Furu Wei  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2604.01220v1  

#### Abstract
The rise of test-time scaling has remarkably boosted the reasoning and agentic proficiency of Large Language Models (LLMs). Yet, standard Transformers struggle to scale inference-time compute efficiently, as conventional looping strategies suffer from high computational overhead and a KV cache that ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Universal YOCO for Efficient Depth Scaling

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Transformer** 架构在实现 **test-time scaling**（推理时计算扩展）时面临两大瓶颈：
- **高计算开销**：传统循环机制（如 Universal Transformer）需要重复执行所有层，导致计算复杂度随深度线性增长。
- **KV Cache 膨胀**：每轮迭代都会生成新的 Key-Value 缓存，内存占用随迭代次数 $T$ 和层数 $L$ 成比例增加，严重限制长上下文场景下的可扩展性。

这些问题阻碍了 LLM 在推理阶段高效地通过“多步思考”提升其 **reasoning** 和 **agentic** 能力。

---

### 🚀 提出的新方法：Universal YOCO (YOCO-U)

YOCO-U 是一种结合 **YOCO 架构** 与 **递归计算（recursion）** 的新型高效深度扩展框架，其核心思想是：

- 将模型分为两个模块：
  - **Self-Decoder**：处理输入表示，采用 **efficient attention**（如滑动窗口 attention, SWA），并进行 **T 次迭代计算**。
  - **Cross-Decoder**：复用 Self-Decoder 产生的全局 KV Cache 进行自回归生成。
- 引入 **Universal Self-Decoder**：在浅层模块中共享参数、多次迭代，增强表达能力而不增加参数量。

> 🔁 “Think Deeper, Cache Once” —— 更深的推理，仅一次缓存。

---

### ⚖️ 相比现有方法的优势

| 特性 | Standard Transformer | Universal Transformer | YOCO | YOCO-U |
|------|------------------------|------------------------|------|--------|
| 参数共享 | ❌ | ✅ | ❌ | ✅ |
| KV Cache 复用 | ❌ | ❌ | ✅ | ✅（仅局部增长） |
| 推理前填充复杂度 | $O(LN^2D)$ | $O(LTN^2D)$ | $O(\tilde{L}ND)$ | $O(\tilde{L}TND)$ |
| KV Cache 内存 | $O(LND)$ | $O(LTND)$ | $O((N + WL)D)$ | $O((N + WTL)D)$ |
| 长上下文支持 | 差 | 极差 | 好 | **优秀** |

> ✅ **优势总结**：
> - 实现了 **constant global KV cache**，避免了传统递归带来的内存爆炸；
> - 利用 **partial recursion**（仅对浅层 efficient-attention 模块递归），以极小额外开销换取显著性能增益；
> - 继承 YOCO 的线性 prefilling 和低内存 footprint，适合部署于长文本任务。

---

## 2. 核心实验方法和设置

### 📚 数据集与训练配置

#### 主要训练数据
- 自建大规模语料（未公开名称），训练总 token 数达 **300B**。
- 使用序列长度为 **8192**，batch size 为 **4M tokens**。
- 模型大小：**10B 总参数，激活参数 1.3B**（MoE 架构，64 专家中激活 8+1 共享专家）。

#### 模型结构细节
- 隐藏维度：`2560`
- 总层数：`20` → Self-Decoder 和 Cross-Decoder 各 `10` 层
- Self-Decoder 使用 **Sliding Window Attention (SWA)**，窗口大小 `512`
- Cross-Decoder 使用标准 full attention + **NoPE** 位置编码
- Self-Decoder 使用 **RoPE**

#### 基线对比方法
| 模型 | 类型 | 描述 |
|------|------|------|
| Transformer | 非递归 | 标准 decoder-only 模型 |
| YOCO | 非递归 | 原始 YOCO 架构，无递归 |
| Universal Transformer (UT) | 递归 | 整体网络权重共享，loop 2 次 |
| RINS | 递归 | 早期层递归的标准 Transformer |
| ParScale | 并行扩展 | 并行分支扩展计算量 |

---

### 📊 评估指标

| 类别 | 指标 |
|------|------|
| 语言建模 | Validation Loss, Perplexity (ppl) |
| 下游任务 | Accuracy (%) on ARC-C, Winogrande, HellaSwag, MMLU, BBH, GSM8K, Humaneval, DROP 等 |
| 数学推理 | 11 个 benchmark 平均准确率（GSM8K, MATH, SVAMP, OlympiadBench 等） |
| 长上下文能力 | Long-sequence perplexity（Book & Code 数据）、Needle-in-a-Haystack（NIAH）检索准确率 |
| 推理效率 | Prefill/Decode Throughput (tokens/sec)，KV Cache 占用（MB） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### （1）语言建模损失 vs FLOPs（图2）
- 在相同 FLOPs 下，YOCO-U 比非递归 YOCO **降低 △L=0.033 的 loss**，表明更强的计算利用率。
- 达到同等性能所需训练 token 减少 **62%**，说明更高的 **token utility**。

#### （2）下游任务表现（表2）
| Model | Average Score |
|-------|---------------|
| YOCO (non-recursive) | 41.78 |
| YOCO-U ($T=3$) | **46.23** (+4.45) |
| YOCO-U (equal FLOPs) | **47.08** |

> 💡 即使在控制 FLOPs 的情况下仍大幅领先，证明收益不仅来自更多计算，而是架构有效性。

#### （3）数学推理能力（图3）
- 在 **11 项数学基准测试** 上平均准确率提升 **+24.4%**。
- 表明递归机制有效增强了隐式与显式推理能力，且与 test-time scaling 正交互补。

#### （4）与其他架构比较（表3）
| Model | Avg Acc↑ | KV Cache↓ |
|-------|----------|-----------|
| Transformer | 47.1 | 高 |
| YOCO | 47.0 | 低 |
| Universal TRM | 47.8 | 极高 |
| RINS | **48.3** | 高 |
| **YOCO-U** | **48.3** | **极低（≈YOCO）** |

> ✅ YOCO-U 实现了与 RINS 相当的性能，但 KV Cache 开销仅为后者的 **~1/38**！

#### （5）长上下文建模（图4 & 表4）
- 在 **Book 和 Code 数据** 上，随着输入长度增加，YOCO-U 的 perplexity 显著低于 Transformer/YOCO，并与 RINS 持平。
- **Needle-in-a-Haystack 测试** 中，YOCO-U 在单针和双针任务上分别达到 **1.00 / 0.95** 准确率，优于或持平其他模型，验证其强大的长程依赖捕捉能力。

#### （6）消融实验（表5）
| 变体 | 结果分析 |
|------|---------|
| Deep instead of wide | 性能几乎不变 → 深度本身不直接带来增益 |
| Upper Loop (Cross-Decoder 递归) | 性能下降 → 不宜在深层递归 |
| Upper Loop w/o Shared KV | 明显退化 → 共享 KV 至关重要 |

> ✅ 验证了设计选择的合理性：**应在浅层 efficient-attention 模块中递归，并保持全局 KV 共享**。

---

## 4. 关键结论和发现

### 🔍 主要发现

1. **Partial Recursion + Efficient Attention 是高效深度扩展的关键路径**  
   YOCO-U 证明，在浅层模块中引入递归即可显著提升模型能力，而无需在整个网络中复制计算。

2. **“One-Piece KV Cache” 架构极大缓解内存压力**  
   全局 KV 缓存只需构建一次，不受迭代次数影响，使得 YOCO-U 在长上下文场景下极具部署优势。

3. **递归与 MoE、test-time scaling 等技术正交兼容**  
   实验显示 YOCO-U 可无缝集成 Thinking SFT，进一步释放数学推理潜力。

4. **参数利用更高效，scaling law 更优**（图5）
   - 在相同激活参数下，YOCO-U 可缩小甚至反超非递归模型；
   - 支持用 **50% 更少参数** 实现相近性能，具备成本优势。

---

### ⚠️ 方法的局限性

- 当前递归次数 $T$ 固定，缺乏动态调整机制（如根据问题难度自适应 loop 次数）。
- 依赖 hybrid attention 设计（local + global），工程实现略复杂于纯 Transformer。
- 所有实验基于自研数据与 MoE 架构，通用性需在更多开源模型上验证。

---

### 🔮 未来工作方向

1. **Dynamic Recursion**：探索基于 confidence 或 energy 的 early-exit 机制，实现 adaptive depth。
2. **Integration with Test-Time Scaling**：将 YOCO-U 作为 backbone，结合 CoT、ToT、RL 等推理策略。
3. **Extension to Multimodal Models**：应用于 vision-language 模型中的 cross-modal interaction 模块。
4. **Hardware-Aware Optimization**：针对 H100/A100 等硬件进一步优化 kernel fusion 与 memory layout。

---

## ✅ 总结

> **YOCO-U 提供了一条“高性能 + 高效率”的深度扩展新范式**：它通过将递归计算限制在轻量级的 Self-Decoder 模块中，实现了“更深的思考”与“更低的开销”之间的理想平衡。实验证明其在多项任务上超越主流架构，同时保持卓越的推理效率，是迈向 **scalable, cost-effective LLMs** 的重要一步。

</details>

---

### 2. [Application of parametric Shallow Recurrent Decoder Network to magnetohydrodynamic flows in liquid metal blankets of fusion reactors](https://arxiv.org/abs/2604.02139)

**Authors**: M. Lo Verso, C. Introini, E. Cervi, L. Savoldi, J. N. Kutz, A. Cammi  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2604.02139v1  

#### Abstract
Magnetohydrodynamic (MHD) phenomena play a pivotal role in the design and operation of nuclear fusion systems, where electrically conducting fluids (such as liquid metals or molten salts employed in reactor blankets) interact with magnetic fields of varying intensity and orientation, influencing the...

---

### 3. [Agent Q-Mix: Selecting the Right Action for LLM Multi-Agent Systems through Reinforcement Learning](https://arxiv.org/abs/2604.00344)

**Authors**: Eric Hanchen Jiang, Levina Li, Rui Sun, Xiao Liang, Yubei Li, Yuchen Wu, Haozheng Luo, Hengli Li, Zhi Zhang, Zhaolu Kang, Kai-Wei Chang, Ying Nian Wu  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.00344v1  

#### Abstract
Large Language Models (LLMs) have shown remarkable performance in completing various tasks. However, solving complex problems often requires the coordination of multiple agents, raising a fundamental question: how to effectively select and interconnect these agents. In this paper, we propose \textbf...

---

### 4. [Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention](https://arxiv.org/abs/2604.00754)

**Authors**: Zehao Jin, Yanan Sui  
**Category**: cs.CL  
**Published**: 2026-04-03  
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

### 解决的问题
传统 **Sliding Window Attention (SWA)** 虽然在计算复杂度上为 $O(nw)$，适合长序列建模，但由于其**局部性限制**，每层仅能覆盖窗口大小 $w$ 的邻域，导致多层后仍难以实现全局信息流动（接收域增长为 $lw$），当 $w \ll n$ 时大量上下文不可达。这限制了模型对远距离依赖的建模能力。

现有改进方案如引入 global tokens、hand-crafted sparse patterns 或 block-level routing（如 MoBA）会增加架构复杂性和参数量。

---

### 提出的新方法与思路
受果蝇全脑连接组（**Drosophila connectome**）启发，作者提出 **Stochastic Attention (SA)** ——一种**无参数、即插即用**的 SWA 增强机制：

- 在每一层中：
  1. **随机打乱** token 序列顺序；
  2. 在打乱后的空间执行标准 SWA；
  3. 再通过逆排列恢复原始顺序。
  
该操作将原本固定的局部窗口转化为**跨序列的随机子集**，形成类似“小世界网络”中的**随机长程捷径（stochastic shortcuts）**。

进一步地，提出 **gated SA + SWA** 结构：
- 并行运行 SA 和 SWA 分支；
- 使用两个独立的 sigmoid gate 动态融合输出；
- 实现“本地聚类 + 随机长程连接”的小世界特性。

---

### 相比现有方法的优势
| 特性 | SA | SWA | Full Attention | MoBA |
|------|----|-----|----------------|-------|
| 时间复杂度 | $O(nw)$ | $O(nw)$ | $O(n^2)$ | $O(nw)$ |
| 接收域增长 | 指数级 $O(\log_w n)$ | 线性 $O(lw)$ | 全局（1层） | 手工设计 |
| 参数增量 | 0（单路径） / 少量（双路径） | 0 | 0 | 引入额外路由逻辑 |
| 可插拔性 | ✅ 改动极小，兼容任意 SWA 实现 | — | — | ❌ 需专门实现 |
| 理论保障 | 有：期望连接概率均匀、覆盖深度 $O(\log n)$ | 无 | — | 依赖启发式 |

> ✅ **核心优势**：在保持 $O(nw)$ 成本的同时，实现了接近 full attention 的表达力，且无需重新训练即可作为推理阶段的增强模块。

---

## 2. 核心实验方法和设置

### 数据集
1. **预训练语言模型任务**：
   - 使用 **SlimPajama** 子集（6B tokens，共约15B训练token）
   - 模型规模：~360M 参数
2. **训练-free 推理测试**：
   - 在已训练好的大模型上直接替换 attention 模块进行评估：
     - **Qwen3-8B**
     - **Qwen3-30B-A3B**

### 评估基准（零样本评测）
- **通用理解与推理**：`MMLU`, `ARC-Easy`, `ARC-Challenge`
- **常识推理**：`HellaSwag`, `PIQA`, `WinoGrande`
- **长程依赖预测**：`LAMBADA`
- **问答能力**：`BoolQ`
- **代码生成**：`HumanEval`

### 实验设置
| 设置项 | 描述 |
|--------|------|
| 序列长度 | 2048（预训练）、可变（推理实验至32K） |
| 注意力窗口 $w$ | 256（预训练）；推理中扫描 $w_{\text{eff}} \in \{16,32,\dots,512\}$ |
| 对比方法 | Full Attention, SWA, SA, SA+SWA, MoBA ($k=2$) |
| 评估方式 | Zero-shot accuracy / perplexity ↓ |
| 实现细节 | 使用 FlexAttention 实现高效 mask 构造；RoPE 使用原始位置编码 |

---

## 3. 主要实验结果和性能指标

### （1）预训练实验结果（Table 1）

| Model | WikiPPL↓ | LAMBADA PPL↓ | Avg Acc↑ |
|-------|----------|--------------|-----------|
| Full Attention | 51.34 | 185.3 | 34.9 |
| SWA (w=256) | 57.05 | 156.1 | 35.1 |
| SA (w=256) | 75.83 | 260.1 | 34.3 |
| **SA+SWA (w=256)** | **51.98** | **131.7** | **35.9** |

✅ **关键发现**：
- 单独使用 SA 导致 perplexity 显著上升（破坏局部连贯性），但下游任务表现尚可 → 表明 SA 捕获了互补的全局信息。
- **SA+SWA 组合取得最佳平均准确率（35.9）和最优 LAMBADA 表现**，说明二者互补性强。
- 在 WikiText 上几乎追平 Full Attention，表明其有效逼近全局注意力的信息流。

---

### （2）训练-free 推理实验（Qwen3 系列）

#### 总体趋势（Figure 4）
- **Stochastic Attention 最快恢复 full attention 性能**：
  - Qwen3-8B @ $w=128$：SA 达到 **70.9%**，接近 full attention 的 71.5%，而 SWA 仅为 62.2%
  - Qwen3-30B-A3B @ $w=64$：SA 达到 **73.2%**，SWA 仅 47.0%，MoBA 为 66.3%

#### 关键窗口下的性能对比（Tables 4–5）

| 方法 | Qwen3-8B @ $w=256$ (Avg) | Qwen3-30B-A3B @ $w=256$ (Avg) |
|------|----------------------------|--------------------------------|
| SWA | 69.1 | 74.9 |
| MoBA (k=2) | 71.6 | 76.9 |
| **Stochastic (ours)** | **71.6** | **77.7** ✅ |
| Full Attention | 71.5 | 77.4 |

✅ **结论**：
- **SA 在相同计算预算下持平甚至超越 MoBA**，尤其在 MMLU、BoolQ、LAMBADA 等需要跨上下文整合的任务上优势明显。
- 在极小窗口（如 $w=32$）下，SWA 几乎崩溃（MMLU <35%），而 SA 仍维持较高性能（>50%），验证其强大的全局混合能力。

---

### （3）消融实验与理论分析支持
- **接收域增长速度**（Figure 2 左）：
  - SA：指数增长，$O(\log_w n)$ 层内实现全序列覆盖
  - SWA：线性增长，需 $O(n/w)$ 层
- **效率分析**（Table 2）：
  - 序列越长，SA 相对于 Full Attention 的加速越显著：
    - $n=2K$: 1.5×
    - $n=32K$: **28× 更快**
- **光谱分析**（Appendix A.4）：
  - 单层 SA 与 SWA 具有相同谱隙；
  - 多层独立排列打破循环结构慢模式，实现快速混合（rapid mixing）

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **神经科学启发有效**：果蝇 connectome 中“局部密集 + 随机长程连接”的小世界拓扑可在 attention 机制中复现，并带来实际性能提升。
2. ✅ **SA 是高效的 drop-in upgrade**：无需修改模型权重或重新训练，即可在推理时显著提升 SWA 的表达能力。
3. ✅ **SA + SWA 实现互补**：SWA 提供局部一致性，SA 提供全局覆盖，门控机制可自适应平衡两者。
4. ✅ **理论与实践一致**：SA 的接收域呈指数扩展，在 $O(\log n)$ 层内可达全部节点，解释其高效性。

---

### 方法的局限性
- **依赖足够深度**：浅层模型可能无法充分积累随机路径以实现全局连接。
- **极端短窗口下仍有损失**：尽管优于 SWA，但在 $w<32$ 时仍难完全替代 full attention。
- **随机性引入方差**：SA 输出具有随机波动（variance $O(B/w)$），需靠门控或集成缓解。
- **不适用于严格顺序敏感任务**？——文中未讨论，但打乱顺序可能影响某些结构化生成。

---

### 未来工作方向
1. **动态窗口分配**：结合 content-based routing 决定何时启用 SA 或调整 $w$
2. **更复杂的排列策略**：非均匀随机排列（如偏好语义相近区域）
3. **与其他稀疏 attention 方法结合**：如 SA + BigBird 或 SA + Sinkhorn
4. **应用于 Vision Transformer 或 Multimodal Models**
5. **硬件优化**：针对 permutation 操作设计专用 kernel 进一步降低开销

---

> 🔬 **最终启示**：  
> “Global information flow 不必依赖 dense all-to-all connectivity。”  
> —— 通过深度中累积的**结构化局部计算 + 随机长程跳接**，即可高效实现全局感知。这一原则不仅适用于 AI 架构设计，也再次印证了生物神经系统的设计智慧。

</details>

---

### 5. [Taming the Exponential: A Fast Softmax Surrogate for Integer-Native Edge Inference](https://arxiv.org/abs/2604.02292)

**Authors**: Dimitrios Danopoulos, Enrico Lupi, Michael Kagan, Maurizio Pierini  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.02292v1  

#### Abstract
Softmax can become a computational bottleneck in the Transformer model's Multi-Head Attention (MHA) block, particularly in small models under low-precision inference, where exponentiation and normalization incur significant overhead. As such, we suggest using Head-Calibrated Clipped-Linear Softmax (...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Taming the Exponential: A Fast Softmax Surrogate for Integer-Native Edge Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在基于 Transformer 的模型中，**Softmax** 是 Multi-Head Attention (MHA) 模块的关键组成部分，但在边缘设备（edge devices）上进行低精度推理时，其计算开销成为显著瓶颈。具体问题包括：
- **指数运算（exp）代价高昂**：尤其在 `int8` 量化模型中，标准 Softmax 需要将整数转换为浮点（如 bfloat16）以执行 exp 操作，带来额外的类型转换和计算延迟。
- **硬件利用率低下**：现有实现依赖 Look-Up Tables (LUTs) 或浮点运算单元，无法充分利用 AI Engine 中高效的 **int8 MAC（Multiply-Accumulate）流水线**。
- **归一化与同步开销大**：传统 Softmax 包含全局 reduction（max、sum）、exp 和除法操作，难以高效映射到整数向量处理架构。

### **提出的新方法：Head-Calibrated Clipped-Linear Softmax (HCCS)**
作者提出一种全新的 **Softmax 替代函数**——**HCCS**，专为 **int8 量化下的边缘推理**设计，具备以下特点：
- **无显式指数运算**：用一个**分段线性映射**替代 exp 函数，形式为 $ s_i = B_h - S_n \cdot \delta_i $，其中 $\delta_i = \min(\max x - x_i, D_{\text{max},h})$ 是最大值中心化并截断的距离。
- **单调且有界**：保证输出概率分布的有序性和数值稳定性。
- **每头校准（Per-head Calibration）**：引入轻量级参数 $(B_h, S_h, D_{\text{max},h})$，针对每个 attention head 单独离线优化，保留各 head 的统计特性差异。
- **完全整数运算支持**：整个流程可在 `int8`/`int16` 下完成，无需浮点转换，直接映射到 AMD Versal AI Engine 的 **native int8 MAC pipeline**。

### **相比现有方法的优势**
| 方面 | 现有方法（如 BF16 Softmax、LUT-based） | HCCS |
|------|----------------------------------------|------|
| **计算模式** | 浮点 exp 或 LUT 查表 | 完全整数运算（add, sub, clamp, MAC） |
| **硬件适配性** | 不适用于 int8 流水线，受限于 LUT 吞吐 | 天然匹配 AI Engine 的 int8 MAC 单元 |
| **吞吐效率** | 受限于 exp 和 reciprocal 延迟 | 显著提升，尤其短序列下 |
| **灵活性** | 固定函数形式 | 支持 per-head 参数校准，适应异构 head 分布 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **SST-2**：二分类情感分析任务，最大序列长度 64。
- **MNLI**：自然语言推断任务，输入为句子对，最大序列长度 128。

### **模型**
- **BERT-tiny**：2 层，2 头，hidden=128
- **BERT-small**：4 层，8 头，hidden=512  
> 聚焦小规模模型，反映边缘部署典型场景。

### **实验设置**
- **平台**：AMD Versal AI Engine（AIE-ML 和 AIE-MLv2），使用 **cycle-accurate AIE simulator**（Vitis 2025.2）
- **输入建模**：通过 PLIO 直接输入，排除 PS/DDR 传输开销，模拟流式推理。
- **量化方式**：`int8` 量化，采用 Quantization-Aware Training (QAT) 进行微调以恢复精度。
- **HCCS 参数校准**：
  - 在代表性数据集上离线进行网格搜索（grid search）
  - 优化目标：最小化 KL 散度（KL-divergence） between standard softmax 和 HCCS 输出（在 int16 空间比较）
  - 参数约束确保满足 `int8` 运算的安全范围（如非负、无溢出）

### **评估指标**
1. **任务准确率（Accuracy）**：下游任务性能（SST-2, MNLI）
2. **KL 散度**：衡量 attention 分布保真度
3. **吞吐量（Throughput）**：单位为 elements/s，衡量 kernel 执行效率
4. **消融实验**：不同粒度的参数共享（global/shared vs per-layer vs per-head）

### **基线方法对比**
- **AMD 官方参考实现**：BF16 Softmax kernel
  - AIE-ML：基于 LUT 的 exp 实现
  - AIE-MLv2：使用原生 BF16 exp 指令
- 所有对比均在同一硬件仿真环境下进行。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **任务准确率（Table I）**
| Model | Baseline (Float32) | HCCS (No Retrain) | HCCS (With QAT) | Δ (vs Baseline) |
|-------|--------------------|-------------------|------------------|-----------------|
| BERT-tiny (SST-2) | 0.825 | 0.619 | 0.822 | -0.003 |
| BERT-small (SST-2) | 0.893 | 0.766 | 0.878 | -0.015 |
| BERT-tiny (MNLI) | 0.653 | 0.480 | 0.639 | -0.013 |
| BERT-small (MNLI) | 0.742 | 0.602 | 0.723 | -0.019 |

> **结论**：经过 QAT 微调后，HCCS 模型可恢复至原始模型 **98%+ 的准确率**，最大误差仅 **1.9 个百分点**。

#### ✅ **吞吐量表现（Table III）**
| Sequence Length | Platform | BF16 Baseline | HCCS (i16+div) | Speedup | HCCS (i8+CLB) | Speedup |
|-----------------|----------|---------------|----------------|---------|--------------|---------|
| 32 | AIE-ML | 0.09 G/s | 0.41 G/s | 4.6× | **1.36 G/s** | **15.1×** |
| 64 | AIE-ML | 0.16 G/s | 0.78 G/s | 4.9× | **2.19 G/s** | **13.7×** |
| 128 | AIE-ML | 0.25 G/s | 1.37 G/s | 5.5× | **2.18 G/s** | **8.72×** |
| 32 | AIE-MLv2 | 0.24 G/s | 0.41 G/s | 1.7× | **1.46 G/s** | **6.1×** |
| 128 | AIE-MLv2 | 0.77 G/s | 1.41 G/s | 1.8× | **2.21 G/s** | **2.9×** |

> **最高达 15.1× 吞吐加速**，尤其在短序列和 AIE-ML 平台上优势明显。

#### ✅ **多核扩展性（Figure 3）**
- 在 AIE-MLv2 上扩展至 **184 个 AI Engine tiles**
- HCCS (i8+CLB) 达到 **407 G elements/s** 总吞吐
- **线性扩展良好**，验证了算法的高度并行性

#### ✅ **消融实验（Table II & Figure 2）**
| Calibration Granularity | SST-2 (BERT-tiny) | MNLI (BERT-tiny) |
|-------------------------|-------------------|------------------|
| Shared / Global | 0.817 | 0.416 |
| Per-layer | 0.819 | 0.552 |
| **Per-head (HCCS)** | **0.822** | **0.639** |

> **结论**：per-head 校准显著优于粗粒度共享，在异构 attention head 分布中尤为重要。

此外，注意力可视化显示：
- HCCS 成功保留了 **broad head**（广泛注意）和 **focused head**（聚焦注意）的结构特征
- 绝对概率值虽有差异，但相对排序和分布形态保持一致

---

## **4. 关键结论和发现**

### **主要发现**
1. **Softmax 可被高效替代**：在量化 MHA 中，**精确的 exp 运算并非必需**，可通过简单的 clipped-linear 映射实现高保真近似。
2. **per-head 校准至关重要**：不同 attention head 具有不同的统计特性，**细粒度参数调整能显著提升最终任务准确率**。
3. **硬件友好设计大幅提升效率**：HCCS 完全基于整数运算，**天然契合 AI Engine 的 int8 MAC 架构**，避免了浮点转换和 LUT 内存瓶颈。
4. **吞吐优势显著**：在 AIE-ML 上，HCCS 最高实现 **15.1× 于 BF16 baseline 的吞吐提升**，即使在 AIE-MLv2 上也有 2–6× 提升。
5. **精度损失可控**：经 QAT 微调后，准确率下降控制在 **<2%**，证明该 surrogate 在实际应用中的可行性。

### **方法的局限性**
- **需要离线校准**：虽然不增加推理开销，但仍需在部署前完成 per-head 参数搜索。
- **依赖 QAT**：若不进行量化感知训练，直接替换会导致严重精度下降（如 SST-2 上降 20+ pts）。
- **目前仅适用于 encoder-only 模型**：未在 decoder 或生成式任务中验证。
- **参数空间仍需人工约束**：如 $D_{\text{max},h} \leq 127$、$B_h - S_n D_{\text{max},h} \geq 0$ 等，限制了表达自由度。

### **未来工作方向**
- **可学习的 HCCS**：将 $(B_h, S_h, D_{\text{max},h})$ 作为可训练参数，在训练过程中联合优化。
- **扩展至其他硬件平台**：如 FPGA、ASIC、NPU 等支持 int8 MAC 的设备。
- **应用于更复杂的 attention 结构**：如稀疏 attention、long-range attention。
- **探索更多 surrogate 形式**：结合 piecewise-linear 或 low-degree polynomials 进一步提升拟合能力。

---

> **总结一句话**：  
> **HCCS 是首个面向 AMD Versal AI Engine 的 int8 原生 Softmax 替代方案，通过 head-calibrated clipped-linear 近似，在几乎无损任务精度的前提下，实现了高达 15× 的吞吐提升，为边缘端高效 Transformer 推理提供了新的硬件协同设计范式。**

</details>

---

### 6. [Apriel-Reasoner: RL Post-Training for General-Purpose and Efficient Reasoning](https://arxiv.org/abs/2604.02007)

**Authors**: Rafael Pardinas, Ehsan Kamalloo, David Vazquez, Alexandre Drouin  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2604.02007v1  

#### Abstract
Building general-purpose reasoning models using reinforcement learning with verifiable rewards (RLVR) across diverse domains has been widely adopted by frontier open-weight models. However, their training recipes and domain mixtures are often not disclosed. Joint optimization across domains poses si...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Apriel-Reasoner: RL Post-Training for General-Purpose and Efficient Reasoning*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Reinforcement Learning with Verifiable Rewards (RLVR)** 的多领域推理模型训练面临两大挑战：
1. **多域异步训练中的采样偏差**：不同任务（如数学、代码）在 rollout 长度、验证延迟和样本效率上差异显著，导致简单或快速领域在训练中被过度采样，破坏预设的领域混合比例，影响泛化能力。
2. **推理长度与效率的权衡**：RLVR 模型倾向于生成过长的 **chain-of-thought (CoT)** 路径，造成高推理成本和延迟，尤其在实际部署中不可接受。

### 提出的新方法与创新
本论文提出 **Apriel-Reasoner**，一个基于 15B 参数开源 LLM *Apriel-Base* 的通用高效推理模型，并引入两个关键技术：

#### ✅ 创新点 1：自适应领域采样（Adaptive Domain Sampling）
- **机制**：动态调整各领域的采样权重，以补偿因 rollout 速度不均导致的分布漂移。
- **公式**：  
  $$
  \alpha_d = \text{clip}\left(\frac{w_d}{n_d / N}, 0.1, 10.0\right), \quad p_d = \frac{w_d \alpha_d}{\sum_j w_j \alpha_j}
  $$
  其中 $ n_d $ 是领域 $ d $ 已完成的 rollout 数量，$ N $ 是总数。
- **优势**：无需手动重平衡数据，自动维持目标领域比例（如数学 40%，代码 25%），提升跨域泛化。

#### ✅ 创新点 2：难度感知长度惩罚（Difficulty-Aware Length Penalty, DAP）
- **机制**：将标准长度惩罚（Length Penalty）扩展为根据问题难度动态调节惩罚强度。
- **实现方式**：
  - 对正确回答，惩罚系数 $ \lambda = s^\gamma $，其中 $ s $ 是该问题组内的 solve rate（解决率），越低表示越难，惩罚越弱。
  - 对错误回答，保持固定强惩罚 $ \lambda_f $。
- **优势**：
  - 无额外训练开销（不修改 policy loss 或引入辅助模型）。
  - 鼓励模型对难题进行更长推理，对简单题保持简洁，实现“智能思考”而非“盲目长思”。

### 相比现有方法的优势
| 方面 | Apriel-Reasoner | 现有方法 |
|------|------------------|--------|
| **可复现性** | 完全公开训练配方、超参、数据源 | 多数前沿模型仅开源权重，未披露训练细节 |
| **训练策略** | 联合多域 RLVR + 自适应采样 | 多为串行训练（如 Nemotron-Cascade）或静态混合 |
| **长度控制** | 难度感知、无额外训练成本 | 固定预算、条件训练、或复杂路由机制 |
| **效率-精度权衡** | 显著缩短输出长度同时提升准确率 | 往往牺牲其一 |

---

## 2. 核心实验方法和设置

### 使用的数据集（训练环境）
五个公开、可验证奖励的领域数据集，构成联合 RLVR 训练环境：

| Domain | Dataset | Reward Type |
|--------|---------|-----------|
| **Mathematics** | Open-Reasoner-Zero (~129K) | 最终答案 exact match |
| **Code Generation** | TACO (~24K) | 沙箱执行通过所有测试用例 |
| **Instruction-Following** | IF-RLVR (~95K) | 输出约束满足比例 |
| **Logical Puzzles** | INTELLECT-3/SynLogic (~12K) | 任务特定程序化验证器 |
| **Function Calling** | BFCL v4 单轮任务 (~4K) | 函数名匹配 + 参数有效性 |

> 📌 所有数据集均为公开资源，支持完全复现。

### 实验设置
- **基础模型**：*Apriel-Base*（15B 参数），此前未经过任何 RL 或偏好优化，确保行为变化可归因于本次训练。
- **训练框架**：基于 **PipelineRL** 实现异步 on-policy RL，支持 in-flight weight updates。
- **优化算法**：采用 **Group Sequence Policy Optimization (GSPO)**，避免 critic 训练开销，稳定序列级优化。
- **输出限制**：训练时最大输出长度为 **16K tokens**，测试时放宽至 **32K tokens**，验证长度外推能力。

### 评估指标
- **主指标**：Pass@1 准确率（四个基准平均）。
- **效率指标**：平均每条输出的 **Mean Output Tokens**。
- **评估配置**：
  - 温度 = 0.6，top_p = 0.95
  - 所有模型统一在 32K token 预算下评估
  - 多次采样取平均以降低方差

### 基线方法对比
选取三个同规模（~14B）开源推理模型作为基线：
1. **Nemotron-Cascade-14B**：通过串行 RL 训练的推理模型
2. **Qwen3-14B**：结合 SFT 和 RL 的多阶段训练模型
3. **Phi-4-reasoning**：基于 o3-mini 推理轨迹微调 + RLVR

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）
| Model | AIME-25 Acc | Tok | GPQA Acc | Tok | MMLU-Pro Acc | Tok | LCB v5 Acc | Tok |
|-------|-------------|-----|----------|-----|--------------|-----|------------|-----|
| **Apriel-Reasoner (Ours)** | **78.3%** | 11.3k | **69.8%** | 5.8k | **77.3%** | 1.9k | **70.8%** | 7.4k |
| Apriel-Base | 73.3% | 16.6k | 68.8% | 10.5k | 76.4% | 3.5k | 61.4% | 14.9k |

> 💡 在所有四项任务上均超越基线，且输出长度减少 **30–50%**

### 与最强基线对比（Nemotron-Cascade）
- **AIME-25**：+2.3% 更高准确率，**少生成 41% tokens**（11.3k vs 19.0k）
- **LiveCodeBench**：+0.1% 更高准确率，**仅用 46% 的 token**（7.4k vs 16.0k）
- **GPQA**：最高准确率（69.8%），token 使用量约为一半
- **MMLU-Pro**：准确率领先，token 消耗仅为第二低模型的一半

👉 结果表明 Apriel-Reasoner 成功推动了 **accuracy vs. token budget** 的帕累托前沿。

### 消融实验结果（Ablation Studies）

#### 🔹 DAP vs. 标准长度惩罚（Table 2 后两行）
| 设置 | AIME-25 Acc | Tok | LCB Acc | Tok |
|------|-------------|-----|---------|-----|
| +RLVR w/ LP | 71.7% | 11.1k | 67.7% | 7.0k |
| +RLVR w/ DAP (Ours) | **78.3%** | 11.3k | **70.8%** | 7.4k |

✅ DAP 在几乎不增加 token 开销的情况下，带来 **+6.6% AIME** 和 **+3.1% LCB** 提升，证明其能有效分配推理资源。

#### 🔹 领域混合消融（Table 3）
比较三种混合策略：
- **Uniform (20%/20%/...)**：各项指标均低于所提方案
- **Math & Code Only (50%/50%)**：在 LCB 上大幅下降（67.2% → 70.8%），说明其他领域（如指令遵循、函数调用）对综合能力有贡献
- **Ours (40%/25%/15%/10%/10%)**：在所有任务上达到最佳平衡

➡️ 表明五领域联合训练 + 合理加权是关键。

---

## 4. 关键结论和发现

### 主要发现
1. **多域联合 RLVR 可显著提升通用推理能力**：在 Apriel-Base 上应用 RLVR 后，所有任务准确率均提升。
2. **效率提升源于表达更紧凑，而非推理变浅**：
   - 分析显示 Apriel-Reasoner 与 Apriel-Base 的 **推理步骤数相近**；
   - 但每步使用的 **tokens 少约 35%**，说明模型学会了“言简意赅”。
3. **非生产性步骤显著减少**：从 21% ↓ 至 14%，减少了重复陈述、无效元评论等“过思考”现象。
4. **非线性推理行为增多**：验证（verification）、回溯（backtracking）、子目标设定（subgoal setting）等高级认知行为占比从 11% ↑ 至 17%，表明 RL 训练促进了更智能的推理模式。

### 方法的局限性
- **依赖高质量可验证奖励**：仅适用于 reward 可编程验证的任务（如数学、代码），难以扩展到开放生成任务。
- **DAP 依赖 group 内 solve rate 估计**：在 group size 较小时估计可能不稳定。
- **未探索更大模型尺度的影响**：目前仅在 15B 规模验证，是否可迁移到百亿以上仍需研究。

### 未来工作方向
- 将 DAP 与 test-time budget 控制结合，实现动态推理调度。
- 探索在多模态或 agent 场景下的应用。
- 构建更多可验证的开放域推理数据集，扩大 RLVR 的适用范围。
- 研究如何将此类高效推理能力迁移到小模型（如 distillation）。

---

## 总结
**Apriel-Reasoner** 是一项在 **可复现性、通用性和效率** 上取得重要突破的工作。它不仅提出了实用的 **自适应领域采样** 和 **难度感知长度惩罚（DAP）** 技术，还在多个权威基准上实现了 **更高准确率 + 更短推理路径** 的双重优势，为构建高效、低成本的通用推理模型提供了清晰的技术路线图。

</details>

---

### 7. [DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72](https://arxiv.org/abs/2604.01621)

**Authors**: Wanqian Li, Jintao Peng, Zongfei Jing, Tianyu Zhang, Ze Long, Xianjie Qiao, Xiaoming Chen, Dongxu Yang, Kefeng Duan, June Yang  
**Category**: cs.DC  
**Published**: 2026-04-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.01621v1  

#### Abstract
Large language model (LLM) inference increasingly depends on multi-GPU execution, yet existing inference parallelization strategies require layer-wise inter-rank synchronization, making end-to-end performance sensitive to workload imbalance. We present DWDP (Distributed Weight Data Parallelism), an ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的主流 **LLM 推理并行化策略**（如 tensor parallelism、expert parallelism 和 pipeline parallelism）在每一层计算后都需要进行跨 GPU 的集体通信（collective communication），例如 all-to-all 或 all-gather。这种 **layer-wise inter-rank synchronization** 在真实负载中会导致严重的性能瓶颈，尤其是在以下两种不平衡场景下：
- **请求级不平衡**（request-level imbalance）：不同 rank 处理的输入序列长度（ISL）不同，KV-cache 命中率不一致。
- **权重级不平衡**（weight-level imbalance）：在 MoE 模型中，专家路由（routing）不均导致某些 GPU 负载更高。

这些不平衡会放大为全局等待时间，使得端到端吞吐受限于最慢的 rank。

### 提出了什么新方法或新思路
提出 **DWDP**（Distributed Weight Data Parallelism），一种全新的推理并行策略，其核心思想是：
- **保持 data-parallel 执行模式**：每个 rank 独立接收请求、独立执行推理。
- **分布式权重卸载**（distributed weight offloading）：将 MoE 层中的远程专家权重存储在 peer GPUs 上，并在需要时按需异步预取（asynchronous remote-weight prefetch）。
- **完全去同步化**（fully asynchronous execution）：通过 `cudaMemcpyAsync` 替代 NCCL 集体通信，避免 SM 资源占用和同步开销，实现无阻塞执行。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **性能** | 消除同步等待，显著提升 TPS/GPU，尤其在负载不平衡时增益更大 |
| **灵活性** | 支持任意 DWDP group size，不要求专家数量整除 group size，支持冗余放置以优化局部性 |
| **部署粒度** | 支持单 GPU 粒度资源调度，在 disaggregated serving 中更易实现 rate matching |
| **可扩展性** | 更适合长上下文（long context）场景，因 compute window 更大，易于隐藏 prefetch 开销 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **Artificial Analysis dataset**：用于 context-only 性能分析。
- **SemiAnalysis dataset**：用于 end-to-end 实验，模拟真实服务负载。
  - 最大输入长度：8K tokens
  - 输出长度：1K tokens
  - 输入范围：6.4K–8K tokens（input ratio = 0.8）

### 实验设置
- **硬件平台**：GB200 NVL72，具备高带宽 NVLink 连接，支持高效的 peer-to-peer 通信。
- **软件框架**：基于 **TensorRT-LLM** 实现 DWDP，集成至 PR #12136（上游合并中）。
- **模型**：**DeepSeek-R1**，使用 NVFP4 量化 MoE 权重，FP8 KV cache。
- **运行模式**：disaggregated serving 架构，DWDP 应用于 **context server**，generation server 配置固定。

### 评估指标
| 指标 | 定义 |
|------|------|
| **TPS/GPU** | Tokens Per Second per GPU，衡量硬件利用效率 |
| **TPS/user** | Tokens Per Second per user，衡量服务质量 |
| **TTFT** | Time to First Token（含排队延迟），反映响应延迟 |
| **End-to-end output TPS/GPU** | 端到端每 GPU 输出吞吐量 |
| **Pareto frontier** | 在相同 TPS/user 下比较输出吞吐是否更优 |

### 基线方法对比
- **DEP**（Data Parallelism with Expert Parallelism）：标准的 attention 数据并行 + MoE 专家并行方案，作为主要 baseline。
- 对比配置统一在相同的 runtime 和硬件约束下进行。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### ✅ Context-Only 实验结果（Table 3）
| 设置 | TPS/GPU Speedup | TTFT Speedup |
|------|------------------|-------------|
| ISL ∈ [1K, 32K] | 1.09–1.11× | 1.11–1.27× |
| MNT=32768, ISL=8K | — | 1.16× (TTFT) / 1.10× (TPS/GPU) |
| 工作负载不平衡增加 | 最高达 1.15× | 最高达 1.18× |
| DWDP3 vs DWDP4 | 几乎相同 (~1.09×) | DWDP3 TTFT 较差（部署规模小导致排队） |

> ⚠️ 注意：随着 ISL 增加，相对增益下降——因为 compute 占主导，同步开销占比降低。

#### ✅ End-to-End 实验结果（Figure 5 & Table 5）
在典型服务区间 **20–100 TPS/user** 内：
- **平均提升 end-to-end output TPS/GPU 达 8.8%**
- 在低负载区（20–30 TPS/user）表现最佳：
  - TPS/user 提升 1.15×
  - TPS/GPU 提升 1.10×
- 在高负载区（>170 TPS/user）出现轻微退化（TPS/GPU 0.97×），因系统已严重 generation-bottlenecked

#### ✅ 消融实验结果（Ablation Studies）

##### （1）Split-Weight Merge Elimination（§4.2）
- 引入 TensorList-based groupedGEMM kernel，直接支持多 buffer 输入
- **消除 ~34 μs 的 D2D copy 开销**
- 实现约 **3% 的额外 TPS/GPU 提升**

##### （2）Contention Mitigation via Time-Division Multiplexing（§4.3）
- 将远程权重传输切片（slice），采用 round-robin 调度避免 source-side 多对一拥塞
- 在短 compute window 场景下效果显著：
  - 如 MNT=16K, ISL ratio=0.5 时，从低于 baseline 提升至 **+8.1%**
- 在长窗口下增益较小（通信本就能被隐藏）

##### （3）通信-计算干扰分析（Appendix A）
- 干扰主因是 **power-induced frequency throttling**，而非内存带宽争用
- 当 attention kernel（功耗 >96% TDP）与通信并发时，总功耗超限 → DVFS 触发 → 频率下降 → kernel 延迟上升
- 实测 attention kernel 在 DWDP 中变慢 **1.19×**

---

## 4. 关键结论和发现

### 论文的主要发现
1. **同步开销是现实 LLM serving 中不可忽视的性能瓶颈**，尤其在负载不平衡时可达 12% 以上。
2. **DWDP 成功将同步操作移出关键路径**，实现 fully asynchronous inference，有效缓解负载不均衡带来的性能损失。
3. **在具备高带宽互联（如 NVL72）的系统上，remote weight prefetch 可被 compute window 隐藏**，从而获得净收益。
4. **DWDP 特别适用于 context phase**，因其具有较大的 layer-wise compute 开销，利于 overlap prefetch。
5. **灵活的专家部署能力使 DWDP 支持细粒度资源配置**，有助于 disaggregated serving 中的 rate matching。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖高带宽 P2P 通信** | 若 NVLink 带宽不足，prefetch 时间过长，则无法有效隐藏通信开销 |
| **prefetch 引发通信-计算干扰** | 尤其在 compute-intensive layer（如 attention）中，会引起 power throttling，降低频率 |
| **不适合极端 generation-bottlenecked 场景** | 当生成阶段成为瓶颈时，加速 context 效果有限甚至有害 |
| **当前未解决请求匹配问题** | 减少 context GPU 数量可能导致 rate mismatch，增加 TTFT |

### 未来工作方向
1. **进一步优化通信-计算 overlap 策略**，例如 coarse-grained overlap 以减少 power spike。
2. **结合更好的 request scheduling 机制**，改善 context 与 generation 阶段之间的 rate matching。
3. **探索在其他架构（非 NVL72）上的适配性**，研究如何在低带宽系统中仍能受益。
4. **将 DWDP 扩展至更多类型的模型结构**，如 dense 模型或其他稀疏激活架构。
5. **动态调整 prefetch 策略**，根据运行时负载自适应选择 slice size 和调度方式。

---

> 📌 **一句话总结**：  
> DWDP 通过 **分布式权重卸载 + 异步预取** 实现了 **无需同步的 fully asynchronous 推理**，在 GB200 NVL72 上对 DeepSeek-R1 的 context serving 提升了 **8.8% 的 end-to-end output TPS/GPU**，同时增强了部署灵活性，为高性能 LLM 推理提供了新的并行范式。

</details>

---

### 8. [Matching Accuracy, Different Geometry: Evolution Strategies vs GRPO in LLM Post-Training](https://arxiv.org/abs/2604.01499)

**Authors**: William Hoy, Binxu Wang, Xu Pan  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.01499v1  

#### Abstract
Evolution Strategies (ES) have emerged as a scalable gradient-free alternative to reinforcement learning based LLM fine-tuning, but it remains unclear whether comparable task performance implies comparable solutions in parameter space. We compare ES and Group Relative Policy Optimization (GRPO) acro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Matching Accuracy, Different Geometry: Evolution Strategies vs GRPO in LLM Post-Training

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
该论文系统地探讨了一个关键问题：  
当 **Evolution Strategies (ES)** 和基于梯度的强化学习方法（如 **GRPO**）在下游任务上达到相似准确率时，它们是否在参数空间中找到了“相同”的解？这些方法对模型的副作用（如遗忘、知识漂移）有何不同？

此前研究对 ES 在 LLM 对齐中的表现存在矛盾结论：一些认为其保留原始能力更强，另一些则指出其导致严重遗忘。本文旨在澄清这种差异背后的几何机制。

### 🧠 提出的新方法与新思路
- **首次从参数空间几何角度对比 ES 与 GRPO**，揭示了二者尽管任务性能相近，但在更新路径、方向、范数等方面存在根本性差异。
- 提出 **“信号-扩散分解”理论（signal-diffusion decomposition）**，统一解释了以下现象：
  - ES 更新为何具有巨大 $L_2$ 范数；
  - 为何 ES 与 GRPO 的更新方向几乎正交；
  - 为何两者之间存在线性模式连接（linear mode connectivity），且无损失壁垒（loss barrier）；
  - 为何 ES 在未训练任务上的行为类似于随机扰动。

该理论表明：**ES 的权重变化由两部分组成**：
1. **On-manifold component**：沿高曲率方向移动，对应任务相关信号，类似 GD；
2. **Off-manifold component**：在平坦子空间中进行各向同性的随机游走（random walk），不改变损失值，但显著增加参数位移。

### 🔍 相比现有方法的优势
| 维度 | ES | GRPO |
|------|----|-------|
| **优化方式** | 梯度自由（gradient-free） | 基于梯度（gradient-based） |
| **内存开销** | 更低（无需反向传播） | 较高（需存储计算图） |
| **并行性** | 极高（可完全并行化种群评估） | 受限于序列训练 |
| **更新特性** | 大范围、扩散式更新 | 小步长、精准定向更新 |
| **知识保留** | 在某些 holdout 任务上更好（如 IFEval） | 在其他任务更稳定（如 MMLU） |

> 💡 **核心洞见**：**相似的任务性能 ≠ 相似的内部解构**。两种方法可以达到相同的“山顶”，但走的是完全不同地形的路径。

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
实验基于四个多样化任务，涵盖不同推理类型：

| 任务 | 类型 | 训练样本数 | 测试样本数 |
|------|------|------------|------------|
| **Countdown** | 算术推理 | 200 | 2,000 |
| **Math** | 数学解题 | 200 | 500 |
| **SciKnowEval-Chemistry** | 化学知识问答 | 200 | 2,000 |
| **BoolQ** | 是非阅读理解 | 200 | 2,000 |

此外，使用两个**未参与训练的 holdout 任务**来衡量泛化能力和遗忘程度：
- **MMLU**：通用知识测试
- **IFEval (strict)**：指令遵循能力评测

### ⚙️ 实验设置
- **基础模型**：`Qwen3-4B-Instruct-2507`
- **微调方式**：全参数微调（full-parameter fine-tuning）
- **训练模式**：
  - **单任务训练（single-task）**
  - **顺序持续学习（sequential continual learning）**：依次训练四个任务
- **超参数选择**：
  - **ES**: $N=30$, $\sigma=0.0015$, $\alpha=0.00075$
  - **GRPO**: 学习率 $lr=1\times10^{-5}$, AdamW 优化器，禁用 KL 正则项（$\beta_{KL}=0$）

### 🎯 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 主要任务性能指标 |
| **$\|\Delta\theta\|_2$** | 权重更新的 $L_2$ 范数，反映参数空间移动距离 |
| **KL 散度增量** | 每个训练步骤后输出分布的变化，衡量对其他任务的影响 |
| **线性插值性能（Linear Mode Connectivity）** | 在 ES 与 GRPO 解之间进行线性插值得分，判断是否处于同一损失盆地 |
| **扰动方向分析** | 沿 ES/GRPO 方向逐步放大扰动，观察任务准确率变化曲线 |

### 🆚 基线方法对比
- **Evolution Strategies (ES)**：梯度自由搜索，通过种群扰动估计更新方向
- **Group Relative Policy Optimization (GRPO)**：去除了 critic 的 PPO 变体，已成为主流 LLM 后训练方法之一

> 所有比较均在公平条件下进行（相同模型、数据量、训练轮次等）

---

## 3. 主要实验结果和性能指标

### 📊 单任务训练性能（Table 1）

| Task | Base | ES (100) | ES (300) | GRPO |
|------|------|----------|-----------|--------|
| Countdown | 20.5 | 75.1 | **78.6** | 74.5 |
| Math | 61.8 | 72.4 | **74.0** | 68.8 |
| Chemistry | 63.4 | 68.1 | **76.5** | 74.9 |
| BoolQ | 83.7 | 87.2 | **88.1** | 86.6 |

✅ **结论**：  
- **ES 在单任务上匹配甚至超越 GRPO**，尤其在更多迭代下（300步）表现最佳。

---

### 🔁 顺序持续学习稳定性（Figure 1）

- **ES (300)**：出现明显灾难性遗忘（catastrophic forgetting），早期任务性能大幅下降。
- **ES (100) 与 GRPO**：均保持良好稳定性，适合持续学习场景。

➡️ 因此后续实验采用 **ES (100)** 进行对比。

---

### 🔄 忘记与泛化能力（Table 2）

| 阶段 | ES (MMLU) | GRPO (MMLU) | ES (IFEval) | GRPO (IFEval) |
|------|-----------|-------------|--------------|----------------|
| 初始 | 77.5 | 77.5 | 79.7 | 79.7 |
| 最终 | **73.8** | **78.3** | **81.9** | **78.4** |
| Δ | **-3.7** | **+0.8** | **+2.2** | **-1.3** |

📌 **关键发现**：
- **GRPO 更好地保持了 MMLU 表现**（+0.8%），而 ES 显著退化（-3.7%）
- **ES 显著提升了 IFEval 表现**（+2.2%），GRPO 下降（-1.3%）
- ➡️ 说明两种方法对不同类型能力的影响截然不同。

---

### 📏 参数空间几何差异（Table 3）

| 阶段 | ES 范数 | GRPO 范数 | 比例 |
|------|--------|----------|------|
| After Countdown | 87.28 | 1.00 | 87× |
| After Math | 122.76 | 1.15 | 107× |
| After Chemistry | 149.99 | 1.65 | 91× |
| After BoolQ | **173.00** | **1.84** | **94×** |

✅ **结论**：  
- **ES 引起的参数变化比 GRPO 大两个数量级**，即使任务性能相当。

---

### 🔗 线性模式连接性（Figure 3）
- 在 ES 与 GRPO 解之间进行线性插值时，**没有出现性能骤降**。
- 插值路径上的模型性能平滑过渡，有时甚至优于两端点。

➡️ 表明：**尽管几何路径不同，ES 与 GRPO 仍位于同一个损失盆地内**。

---

### 🧭 更新方向分析（Figure 4 & 5）
- **GRPO 方向**：准确率快速上升至峰值，随后迅速下降 → 高效但狭窄。
- **ES 方向**：准确率缓慢提升，需要更大扰动才能达到目标性能 → 扩散性强。
- **Holdout 任务表现**：ES 方向的行为与**随机方向高度相似**，而 GRPO 导致更快的通用能力退化。

➡️ 支持理论预测：**ES 的大部分位移发生在任务无关的平坦维度中**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **任务性能可匹配，但参数几何迥异**：
   - ES 与 GRPO 可实现相近甚至更高的任务准确率。
   - 但 ES 的参数更新范数是 GRPO 的 **~100 倍**，且方向几乎正交。

2. **ES 的本质是“信号+扩散”混合过程**：
   - **On-manifold 信号**：推动模型向最优解前进；
   - **Off-manifold 扩散**：在平坦维度中随机行走，积累大量无意义位移。

3. **理论预测与实证高度一致**：
   - 推导出的随机游走尺度 $\mathbb{E}[\|\Delta\theta\|^2] \propto \frac{\sigma^2 T d}{N}$ 与实验数据拟合度达 $R^2 > 0.99$。
   - 大多数权重矩阵表现出接近理想随机游走行为（effective dimension ratio ≈ 96.4%）。

4. **线性模式连接源于 off-manifold 不变性**：
   - 由于 off-manifold 分量不影响损失，连接 ES 与 GRPO 的直线仍处于低损失区域。

5. **不同的遗忘模式**：
   - ES 更利于提升指令跟随能力（IFEval ↑），但损害通用知识（MMLU ↓）；
   - GRPO 更稳定维持 MMLU，但在最后阶段损害 IFEval。

---

### ⚠️ 方法的局限性
- **理论假设简化**：将 ES 视为 Ornstein-Uhlenbeck 过程，忽略了实际奖励噪声、非二次损失结构等因素。
- **仅适用于特定设置**：结论基于固定种群大小（N=30）、特定噪声尺度下的 ES；更大 N 或自适应策略可能改变行为。
- **未探索所有 LLM 架构**：实验集中在 Qwen3-4B，是否推广到更大或稀疏模型有待验证。
- **缺乏多任务联合训练对比**：目前只做了顺序学习，未考察并行或多任务优化场景。

---

### 🔮 未来工作方向
1. **设计约束 off-manifold 扩散的新型 ES 变体**，例如引入方向先验或稀疏扰动。
2. **结合 GRPO 的精确性和 ES 的并行优势**，开发混合优化框架。
3. **将该几何分析工具应用于更多 LLM 微调算法**（如 LoRA、PPO、DPO）。
4. **研究如何利用 off-manifold 扰动增强鲁棒性或探索能力**，而非视为负面效应。
5. **构建“解流形”可视化工具**，帮助理解不同优化路径的拓扑结构。

---

## 总结（TL;DR）

> 🌟 **ES 和 GRPO 能达到同样高的任务准确率，却走了两条完全不同的路**：
> - **GRPO** 像外科手术刀，精准切入高梯度方向，小幅修改参数；
> - **ES** 像广域勘探队，在参数空间中大范围漫游，一边获取信号，一边在无数平坦维度中随机漂移。
>
> 尽管最终都登顶，但 ES 积累了百倍以上的参数变动，其路径更像随机方向。两者之所以能线性相连，是因为那些“多余”的移动并不影响损失函数。
>
> ❗ 这意味着：**不能仅凭任务性能判断微调方法优劣** —— 必须关注其在参数空间中的几何行为及其对模型整体知识结构的影响。

</details>

---

### 9. [DDCL: Deep Dual Competitive Learning: A Differentiable End-to-End Framework for Unsupervised Prototype-Based Representation Learning](https://arxiv.org/abs/2604.01740)

**Authors**: Giansalvo Cirrincione  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.01740v1  

#### Abstract
A persistent structural weakness in deep clustering is the disconnect between feature learning and cluster assignment. Most architectures invoke an external clustering step, typically k-means, to produce pseudo-labels that guide training, preventing the backbone from directly optimising for cluster ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：DDCL: Deep Dual Competitive Learning

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**deep clustering**领域中的一个根本性缺陷——**representation learning 与 cluster assignment 之间的结构断连（disconnect）问题**。

在传统方法如 **DeepCluster** 中，特征提取（由 CNN 完成）和聚类（由外部 k-means 完成）是两个分离的阶段。k-means 生成的伪标签（pseudo-labels）被用于监督训练骨干网络，但梯度无法通过 k-means 步骤反向传播。这导致骨干网络无法直接优化以提升聚类质量，只能被动地预测当前 k-means 迭代产生的标签。

### 提出的新方法
论文提出了 **Deep Dual Competitive Learning (DDCL)**，这是首个完全可微分（fully differentiable）、端到端（end-to-end）的无监督原型（prototype）表示学习框架。

其核心创新在于架构设计：
- **用内部的 Dual Competitive Layer (DCL) 替代了外部的 k-means**。
- DCL 通过对特征矩阵进行转置操作，使得**原型（prototypes）成为网络的原生输出（native differentiable outputs）**，而非存储在层内的权重向量。
- 这一“单次反转”（single inversion）使得从骨干网络特征提取、原型生成到软分配（soft cluster assignment）的整个流程都可以通过单一的统一损失函数 `Lq` 进行反向传播训练。

### 相比现有方法的优势
| 优势维度 | DDCL | 传统方法 (如 DeepCluster) |
| :--- | :--- | :--- |
| **可微分性** | ✅ 整个流程端到端可微，梯度可直达骨干网络 | ❌ 聚类步骤不可微，存在梯度断点 |
| **训练方式** | ✅ 单一、统一的损失函数，连续梯度流 | ❌ 两阶段交替优化，依赖伪标签 |
| **原型生成** | ✅ 原型是网络的显式输出 | ❌ 原型是外部参数或隐藏权重 |
| **理论分析** | ✅ 提供了严格的代数分解、稳定性定理等理论基础 | ❌ 缺乏对损失几何和动态的深入分析 |
| **高维鲁棒性** | ✅ DCL 梯度天然限制在数据子空间内，抗噪声能力强 | ❌ 在高维稀疏数据下易受噪声影响 |

## 2. 核心实验方法和设置

### 使用的数据集
论文进行了六组受控实验，使用了多种数据集来验证理论预测：
- **合成数据集**：Moons, Circles, Spiral, Blobs （用于可视化和验证基本性质）
- **经典基准**：`sklearn` 的 `load_digits` 数据集 (MNIST Digits, 1797 张 8x8 图像)
- **高维模拟**：MADELON-style 数据集 (n=100, k=2)，在 d ∈ {10, 50, 200, 500, 1000, 5000} 六个维度上测试
- **大规模图像**：**CIFAR-10** (50,000 张图像, k=10)，使用预训练的 ResNet-18 提取特征
- **流式数据**：在 MNIST Digits 上模拟流式学习（streaming），数据以 mini-batch 形式到达且不重复

### 实验设置和评估指标
- **目标**：并非追求 SOTA 性能，而是**严格验证理论推导的结构性预测**。
- **评估指标**：
  - **Clustering Accuracy (ACC)**：通过匈牙利算法匹配预测簇与真实标签后的最高准确率。
  - **Normalized Mutual Information (NMI)**：衡量预测与真实标签间的互信息，归一化后范围为 [0,1]。
  - **Adjusted Rand Index (ARI)**：衡量两个聚类结果的一致性，经随机性校正，完美一致时为 1。
- **实现**：部分实验使用纯 NumPy 实现以确保可复现性，部分使用 PyTorch。

### 基线方法对比
- **DDCL (Lq)**：提出的完整方法，使用软量化损失 `Lq`。
- **DDCL (LoLs)**：消融版本，仅使用重构误差 `LoLs`。
- **DeepCluster**：最相关的两阶段方法。
- **k-means** 和 **k-means+PCA**：经典的聚类基线。
- **MiniBatchKMeans**：用于流式场景的对比。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1. **端到端性能优势**：
    - 在 MNIST Digits 上联合训练骨干网络时，**DDCL (Lq) 的 ACC 达到 0.378±0.085**。
    - 相比其消融版本 **DDCL (LoLs)** (ACC 0.229±0.041)，**性能提升 +65%**。
    - 相比 **DeepCluster e2e** (ACC 0.170±0.042)，**性能提升 +122%**。

2. **高维鲁棒性**：
    - 在 MADELON-style 数据集上，当维度 `d` 超过样本数 `n` (即 `d/n > 1`) 时，传统方法（如 DeepCluster, k-means raw）性能急剧下降至随机水平（ACC ~ 0.5）。
    - **DDCL (Lq)** 则表现出优雅的退化（degrade gracefully），在 `d=200` 时仍保持 ACC=0.807±0.124，显著优于其他方法。

3. **稳定性与方差**：
    - 在 `load_digits` 数据集上，**DDCL 的方差 (±0.053) 显著低于 DeepCluster (±0.091)**，证明了其交替优化循环的不稳定性已被消除。

### 消融实验结果
1. **温度调度（Temperature Annealing）**：
    - 在 `load_digits` 上，使用温度退火策略 (`T` 从 2 降到 0.5) 的配置 (ACC=0.576) 显著优于固定温度 `T=0.5` (ACC=0.563) 或极端值 `T=0.1`/`T=2.0` 的配置，验证了理论分析中温度作为控制软硬分配的“主参数”的作用。

2. **显式分离项 `Lsep` 的作用**：
    - 实验表明，在 `load_digits` 上，添加显式分离项 `Lsep` 的变体 (ACC=0.549) 并未优于仅使用 `Lq` 的版本 (ACC=0.576)。
    - 这证实了**隐式分离力 `VpV` 已足够有效**，`Lsep` 仅作为小温度下的备用机制。

3. **冻结骨干网络 vs. 联合训练**：
    - 当骨干网络特征被冻结时（如在 CIFAR-10 上），`Lq` 和 `LoLs` 的性能几乎相同（ACC 均为 0.265±0.031）。
    - 一旦骨干网络参与联合训练，`Lq` 的性能优势立即显现（+65%）。这证明了**隐式分离力 `VpV` 的优势只有在骨干网络与原型协同适应（co-adapt）时才会传递到骨干网络的学习信号中**。

## 4. 关键结论和发现

### 主要发现
1. **理论驱动的设计是有效的**：DDCL 的核心思想——将原型变为网络的可微分输出——成功解决了 deep clustering 中的根本性结构断连问题。
2. **损失分解揭示了内在机制**：`Lq = LoLs + V` 的代数分解是核心洞见。其中 `V`（加权原型方差）的梯度 `VpV = 2PΣ_qn` 构成了一个**自调节的隐式分离力**，它能自动抵抗原型坍缩（prototype collapse），无需任何辅助目标。
3. **存在负反馈稳定环路**：系统中存在一个“原型分离度 `S` → 分配集中度 `K` → 隐式分离力 `VpV` → `S`”的负反馈循环，使系统能自我调节，避免坍缩和过度分散。
4. **全局稳定性得到保证**：对于编码器特征固定的简化系统，论文证明了**全局 Lyapunov 稳定性**，即所有轨迹有界，能量单调递减，并收敛到 KKT 平稳点集。

### 方法的局限性
1. **全系统全局稳定性仍是开放问题**：论文证明的全局稳定性仅适用于**编码器冻结的简化系统**。完整的、端到端的 DDCL 系统（骨干网络也在演化）的全局渐近稳定性尚未解决，这是一个非凸、多尺度的复杂动力学系统。
2. **大规模基准验证待完成**：本文的重点是理论和受控验证，**尚未在 CIFAR-10、STL-10 等标准视觉基准上进行大规模的、从头开始的端到端 GPU 训练**，以与 SCAN、DINO 等 SOTA 自监督方法进行全面比较。
3. **原型数量 `k` 需预先指定**：与大多数聚类方法一样，DDCL 需要预先知道或设定簇的数量 `k`。

### 未来工作方向
1. **大规模端到端 GPU 验证**：在标准视觉基准（如 CIFAR-10, STL-10）上，使用 ResNet 等骨干网络进行完整的端到端训练，是首要的未来工作。
2. **探索全系统的稳定性**：利用奇异摄动或双时间尺度随机逼近框架，研究完整 DDCL 系统的全局稳定性。
3. **与其他架构结合**：将 DDCL 扩展到 **Transformer**、**RNN** 等架构，并探索其与掩码建模（masked modeling）等预训练任务的兼容性。
4. **研究原型的可识别性**：探究在 `Lq` 损失下，原型配置的唯一性和可识别性问题。

</details>

---

### 10. [Execution-Verified Reinforcement Learning for Optimization Modeling](https://arxiv.org/abs/2604.00442)

**Authors**: Runda Guan, Xiangqing Shen, Jiajun Zhang, Yifan Zhang, Jian Cheng, Rui Xia  
**Category**: cs.AI  
**Published**: 2026-04-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.00442v1  

#### Abstract
Automating optimization modeling with LLMs is a promising path toward scalable decision intelligence, but existing approaches either rely on agentic pipelines built on closed-source LLMs with high inference latency, or fine-tune smaller LLMs using costly process supervision that often overfits to a ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Execution-Verified Reinforcement Learning for Optimization Modeling**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的优化建模自动化方法面临两大挑战：
- **高成本的过程监督**（Process Supervision）：许多训练方法依赖于精细标注的数据（如变量定义、公式推导、参考代码），这类数据构建成本高昂且难以扩展。
- **跨求解器泛化能力差**：模型在特定求解器（如 Gurobi）上训练后，迁移到其他求解器（如 OR-Tools 或 COPT）时性能急剧下降，因为其过拟合了特定 API 的语法模式。

### **提出的新方法：EVOM**
作者提出了 **Execution-Verified Optimization Modeling (EVOM)**，一种基于强化学习的框架，其核心思想是将数学规划求解器作为**确定性的交互式验证器**（deterministic, interactive verifier）。

#### **核心机制**
- 给定自然语言问题 `q` 和目标求解器 `s`，模型生成求解器专用代码；
- 在沙箱环境中执行该代码，获取运行结果（状态、目标值等）；
- 将执行结果与真实答案对比，转化为标量奖励（scalar reward）；
- 使用 **GRPO** 和 **DAPO** 等无评论家（critic-free）的强化学习算法进行策略更新。

#### **关键创新**
- **Outcome-only 学习范式**：仅依赖最终输出是否正确来提供反馈，无需中间步骤标注。
- **闭环生成-执行-反馈-更新流程**：通过试错自动对齐自然语言与数学逻辑。
- **求解器即验证环境**：将不同求解器视为不同的验证环境，实现低成本适配。

### **相比现有方法的优势**
| 维度 | 传统方法（SFT / Prompting） | EVOM |
|------|-------------------------------|------|
| 数据需求 | 需要大量过程级标注 | 仅需问题-答案对 |
| 推理延迟 | 多轮 Agent 自纠错导致高延迟 | 单次生成即可 |
| 跨求解器迁移 | 几乎无法零样本迁移 | 支持零样本转移和低代价微调 |
| 成本 | 高标注成本或高推理成本 | 低成本训练与部署 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **NL4OPT**：来自 NeurIPS 2022 比赛，包含 289 个线性规划问题，强调从自然语言到 LP 公式的转换。
- **MAMO**：评估 LLM 数学建模能力，分为 EasyLP 和 ComplexLP 子集。
- **IndustryOR**：首个工业级 OR 评测基准，涵盖 13 个行业的真实场景问题。
- **OptiBench**：多样化优化问题集合，支持端到端数值验证。

> 所有训练数据均来自 **OR-Instruct-3K**，但只保留问题描述 `q` 和真实答案 `a`，丢弃所有中间过程标注。

### **实验设置**
- **基础模型**：Qwen2.5-7B（也测试了 0.5B–3B 小模型）
- **求解器后端**：Gurobi、OR-Tools、COPT
- **训练流程**：
  - 使用沙箱执行代码（10秒超时，2GB内存限制）
  - 提取 `<code>` 块并执行，返回标准化观察 `o = (c, o_status, v_obj, l_log)`
- **输出格式协议**：
  ```text
  <think>推理过程...</think>
  <code>可执行Python代码...</code>
  ```

### **评估指标**
- **Accuracy**：预测结果与真实答案相对误差 ≤ 5%（主实验）或 ≤ 1e-4（严格测试）
- 判定条件：
  - 代码成功执行
  - 求解状态匹配（OPTIMAL/INFEASIBLE 等）
  - 目标值满足容忍度要求

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Prompting-based** | DeepSeek-R1, OpenAI o1, GPT-4o (CoT, CoE), OptiMUS |
| **Training-based (SFT)** | ORLM (SFT)，基于完整 OR-Instruct 数据集微调 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**
在 Gurobi 后端上的平均准确率（%）：

| Model | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndustryOR | Avg |
|-------|-----------|--------|--------|--------|----------|-----|
| ORLM (SFT) | 60.96 | 84.89 | 88.34 | 35.71 | 27.00 | **59.38** |
| **EVOM (GRPO)** | **62.95** | **84.08** | **88.19** | **34.28** | **31.00** | **60.10** |

✅ **EVOM 匹配甚至略优于 SFT 基线**，且完全不依赖过程监督。

---

### **零样本求解器迁移（Zero-shot Transfer）**
从 Gurobi 迁移到 OR-Tools（Table 2）：

| Model | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndusOR |
|-------|-----------|--------|--------|--------|---------|
| ORLM (SFT) | 3.49 | 4.89 | 0.00 | 1.42 | 6.00 |
| **EVOM (GRPO)** | **54.31** | **77.55** | **84.81** | **22.27** | **24.00** |

📌 **EVOM 实现有效零样本迁移**，而 SFT 方法几乎崩溃，说明其学到的是通用建模逻辑而非特定语法。

---

### **低代价适配（Low-cost Adaptation）**
切换执行后端继续训练（Table 3）：

| Solver | OptiBench ↑ | NL4OPT ↑ | MAMO-E ↑ | MAMO-C ↑ | IndustryOR ↑ |
|--------|-------------|----------|----------|----------|--------------|
| Gurobi | +21.76 | +28.98 | +22.24 | +5.85 | +10.00 |
| OR-Tools | +11.63 | +17.96 | +28.99 | -2.37 | +11.00 |

✅ **无需重建数据集即可快速适应新求解器**，尤其在原始性能较弱时提升显著。

---

### **消融实验结果**

#### **显式推理块的作用（Figure 2）**
- 移除 `<think>` 块导致性能大幅下降（尤其在 IndustryOR 和 MAMO-C 上）
- 表明 `<think>` 不是事后解释，而是内部推理空间，有助于复杂逻辑建模

#### **优化器选择敏感性（Figure 3）**
- GRPO 与 DAPO 性能几乎一致
- 说明性能主导因素是“求解器验证能力”而非优化器本身

#### **小模型表现（Table 6 & 7）**
- Qwen2.5-3B 经 EVOM 训练后接近甚至超过 7B 基础模型
- 证明该方法对小模型具有放大效应，适合资源受限场景

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Outcome-only 监督足以学会高质量优化建模**  
   无需过程标注，仅靠执行反馈即可驱动模型掌握变量定义、约束构造和目标函数设计。

2. ✅ **求解器可作为通用验证环境**  
   更换求解器只需切换执行后端，无需重新标注数据，极大降低迁移成本。

3. ✅ **支持零样本跨求解器迁移**  
   表明 EVOM 学到了与具体 API 无关的**通用数学建模原则**。

4. ✅ **强化学习显著改善变量类型和实现错误**  
   错误分析显示 Variable Error 下降最明显（-5.81%），说明执行反馈能纠正整数声明、边界设定等问题。

5. 🔁 **训练动态呈现“思考变长、代码变短”趋势**  
   模型逐渐将更多预算用于规划阶段，生成更简洁高效的代码。

---

### **方法的局限性**
- **冷启动问题**：对于预训练中极少出现的求解器（如 COPT），初始生成成功率极低，RL 收敛慢。
- **深层语义错误仍难消除**：Constraint Error 和 Objective Error 改进有限，因这些错误不易被执行结果直接暴露。
- **依赖沙箱稳定性**：若求解器行为不稳定或日志不可解析，会影响奖励准确性。

---

### **未来工作方向**
1. **引入分层奖励机制**：结合轻量级过程提示（如关键约束是否生成）以缓解稀疏奖励问题。
2. **多求解器联合训练**：同时在多个后端上训练，增强 solver-agnostic 能力。
3. **冷启动策略优化**：探索更高效的小样本翻译 + SFT 初始化方案。
4. **扩展至非凸/非线性问题**：研究如何处理局部最优干扰下的奖励信号可信度问题。

---

> 📌 **总体评价**：  
> EVOM 开辟了一条**低成本、高泛化性**的优化建模自动化路径。它利用“求解器即验证器”的特性，摆脱了对昂贵标注数据的依赖，并首次系统验证了 outcome-only 强化学习在跨求解器迁移中的有效性，为 Decision Intelligence 的规模化落地提供了新范式。

</details>

---

### 11. [Intelligent Cloud Orchestration: A Hybrid Predictive and Heuristic Framework for Cost Optimization](https://arxiv.org/abs/2604.02131)

**Authors**: Heet Nagoriya, Komal Rohit  
**Category**: cs.DC  
**Published**: 2026-04-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.02131v1  

#### Abstract
Cloud computing allows scalable resource provisioning, but dynamic workload changes often lead to higher costs due to over-provisioning. Machine learning (ML) approaches, such as Long Short-Term Memory (LSTM) networks, are effective for predicting workload patterns at a higher level, but they can in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Intelligent Cloud Orchestration: A Hybrid Predictive and Heuristic Framework for Cost Optimization*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代云环境中的动态工作负载导致资源管理困难，传统方法面临以下挑战：
- **Over-provisioning**：为应对流量高峰而过度分配资源，造成成本浪费。
- **Under-provisioning**：资源不足影响系统性能和 SLA 合规性。
- 单一方法局限：  
  - **Machine Learning (ML)** 虽能预测长期趋势，但存在 **inference latency**，难以响应突发流量。  
  - **Mathematical Heuristics**（如 Game Theory）响应快，但缺乏对未来负载的预判能力。

因此，如何在 **cost efficiency** 与 **real-time responsiveness** 之间取得平衡成为关键问题。

---

### 🚀 提出的新方法与创新思路
提出一种 **Hybrid Optimization-ML 框架**，将两种范式结合：
- **Macro-Level Predictive Scaling**：使用 **LSTM** 进行长期 workload 预测，指导集群规模调整（proactive provisioning）。
- **Micro-Level Heuristic Scheduling**：利用轻量级 **Game Theory** 或 **Simulated Annealing** 实现任务的实时、确定性调度。

> 🔗 核心思想：**“预测+反应”分层架构** —— ML 负责“战略规划”，Heuristics 负责“战术执行”。

---

### ⚖️ 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **Cost Efficiency** | 接近纯 ML 方法的成本优化水平，显著优于纯启发式方法 |
| **Latency & Responsiveness** | 响应速度接近启发式方法，避免了 ML 的推理延迟 |
| **SLA Compliance** | 在流量突增时仍能保持低延迟，满足严格的服务质量要求 |
| **部署可行性** | 减少对高算力硬件（如 GPU）的依赖，更适合实际生产环境 |

---

## 2. 核心实验方法和设置

### 📊 数据集与环境
- 使用模拟的 **dynamic cloud environment**，基于真实世界 workload trace 构建。
- 输入数据包括：
  - 历史 workload telemetry（CPU、内存使用率等）
  - 动态请求流量模式（含周期性和突发性负载）
- 模拟平台未明确指定，但从上下文推断可能基于 **CloudSim** 或自定义仿真器。

---

### ⚙️ 实验设置与评估指标

#### 实验配置
- 对比三种策略：
  1. **Standalone ML Model**（仅 LSTM 预测 + 扩容）
  2. **Mathematical Heuristic Only**（仅 Game Theory 调度）
  3. **Proposed Hybrid Framework**（LSTM + Game Theory 联合）

#### 评估指标
| 指标 | 描述 |
|------|------|
| **Total Operational Cost** | 包括实例运行费用、资源闲置开销等 |
| **Real-Time Task Latency** | 从任务提交到完成的时间，尤其关注流量 spike 期间的表现 |
| **SLA Violation Rate** | 是否因响应慢导致服务协议违约 |
| **Scalability & Convergence Speed** | 系统对快速变化负载的适应能力 |

---

### 🆚 基线方法对比
| 基线方法 | 类型 | 特点 |
|--------|------|------|
| LSTM-only | Predictive ML | 强于趋势预测，但 inference 延迟高 |
| Game Theory / Simulated Annealing | Optimization Heuristic | 响应快，无训练开销，但无法预见未来负载 |
| Serverless 架构方案（参考文献[32]） | Architectural | 成本低但有 cold-start 问题和 vendor lock-in 风险 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自图示与文本描述）

#### （1）总运营成本对比（Total Operational Cost）
- **Hybrid 方法** 的成本仅略高于 **pure ML** 模型，但远低于 **heuristic-only** 方法。
- 相比 heuristic 方法，**hybrid 节省约 30%-40% 成本**（依据 Fig. 中曲线估算）。

#### （2）流量突增时的任务延迟（Real-Time Task Latency）
| 方法 | 表现 |
|------|------|
| **Standalone ML** | 明显延迟（due to inference time），峰值延迟达 70ms+ |
| **Heuristic Only** | 响应最快（<30ms），但因频繁 over-provisioning 导致成本高昂 |
| **Proposed Hybrid** | **延迟控制在 ~35ms 左右**，接近 heuristic 水平，同时大幅降低成本 |

> ✅ 结论：Hybrid 在 **cost 和 latency 之间实现了最优权衡**

#### （3）SLA 合规性
- Hybrid 框架在整个测试周期内 **未出现明显 SLA violation**。
- 纯 ML 方法在突发流量初期因预测滞后导致短暂超时。

#### （4）消融实验（Ablation Study）
虽然文中未明确标注“ablation study”，但从对比设计中可视为隐式消融：
- 移除 LSTM 预测 → 成本上升（验证预测模块价值）
- 移除 Game Theory 实时调度 → 响应延迟增加（验证启发式调度必要性）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **单一范式无法胜任现代云成本优化**：
   - ML 擅长宏观预测但实时性差；
   - Heuristics 响应快但缺乏前瞻性。
2. **Hybrid 架构是未来方向**：
   - 分层协同（macro-prediction + micro-scheduling）可兼顾 **cost efficiency** 与 **real-time performance**。
3. **框架具备工程落地潜力**：
   - 不依赖复杂 DRL 模型，降低部署门槛；
   - 可集成进 Kubernetes、Terraform 等主流编排工具。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖历史数据质量** | LSTM 的预测精度受限于输入 telemetry 的完整性和代表性 |
| **冷启动问题未完全解决** | 若初始阶段无足够训练数据，预测效果下降 |
| **多云支持有限** | 当前框架聚焦单云场景，跨云调度需额外扩展 |
| **模型压缩尚未实现** | 尽管提出轻量化方向，但未实测 pruned 或 quantized LSTM 效果 |

---

### 🔮 未来工作方向
1. **轻量化 ML 模型研究**：
   - 探索 **Neural Network Pruning**、**Quantization** 技术以加速 inference。
   - 引入 **Federated Learning** 实现边缘节点分布式训练，减少中心化计算负担。

2. **构建 Vendor-Agnostic Orchestration Layer**：
   - 利用 **Kubernetes** 和 **Terraform** 开发统一调度层；
   - 接入多云 **Spot Pricing API**，实现自动成本感知迁移。

3. **闭环反馈机制增强**：
   - 将调度结果反馈给预测模型，形成 **online learning loop**，提升自适应能力。

4. **标准化评估基准建设**：
   - 呼吁建立统一的 cloud cost optimization benchmark dataset 和评测协议。

---

## 总结

> 💡 **本文核心洞见**：  
> “The future of cloud cost efficiency lies not in choosing between ML and optimization — but in orchestrating them together.”

该论文通过提出一个 **Hybrid Predictive-Heuristic 框架**，成功弥合了 **predictive accuracy** 与 **real-time agility** 之间的鸿沟，为构建智能、自治、低成本的云资源管理系统提供了切实可行的技术路径。

</details>

---

### 12. [UQ-SHRED: uncertainty quantification of shallow recurrent decoder networks for sparse sensing via engression](https://arxiv.org/abs/2604.01305)

**Authors**: Mars Liyao Gao, Yuxuan Bao, Amy S. Rude, Xinwei Shen, J. Nathan Kutz  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.01305v1  

#### Abstract
Reconstructing high-dimensional spatiotemporal fields from sparse sensor measurements is critical in a wide range of scientific applications. The SHallow REcurrent Decoder (SHRED) architecture is a recent state-of-the-art architecture that reconstructs high-quality spatial domain from hyper-sparse s...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：UQ-SHRED: uncertainty quantification of shallow recurrent decoder networks for sparse sensing via engression

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统的 **SHRED**（Shallow REcurrent Decoder）网络虽然在从超稀疏传感器测量中重建高维时空场方面表现出色，但它仅提供**点估计**（point estimation），缺乏对预测不确定性的量化能力。这在科学应用中是严重缺陷，尤其是在系统具有随机性、高频动态或数据稀缺的情况下，无法支持风险评估、异常检测和不确定性下的决策。

### 提出的新方法
本文提出了 **UQ-SHRED**，一种基于 **engression** 框架的分布学习方法，用于实现有效的 **不确定性量化**（Uncertainty Quantification, UQ）。其核心思想是：
- 在输入层注入**随机高斯噪声** $ \epsilon \sim \mathcal{N}(0, I_{d_\epsilon}) $
- 使用 **Energy Score** 作为训练损失函数，直接优化一个 proper scoring rule，以学习输出空间的完整条件分布 $ P(y_t | x_{t-L+1:t}) $

该方法无需额外网络结构或集成多个模型，仅通过单个网络即可生成预测分布。

### 相比现有方法的优势
| 方法 | 缺陷 | UQ-SHRED 的优势 |
|------|------|----------------|
| **Bayesian Neural Networks** | 计算昂贵，后验近似复杂 | 无须变分推断或采样，计算高效 |
| **MC-Dropout** | 缺乏理论一致性保证 | 具有严格的统计理论保障（严格适当评分规则） |
| **Deep Ensembles** | 需要训练多个独立网络，成本翻倍 | 单一网络，最小化计算开销 |
| **Heteroscedastic Regression** | 假设固定分布形式（如高斯） | 不假设输出分布形态，更灵活 |
| **Normalizing Flows / Diffusion Models** | 架构受限或推理过程迭代耗时 | 仅需一次前向传播生成样本 |

> ✅ **核心优势**：**保持 SHRED 的高效性的同时，实现了理论上可证明的有效不确定性估计**。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖五个来自不同科学领域的复杂真实世界数据集：

| 数据集 | 描述 | 维度/规模 |
|--------|------|-----------|
| **SST** (Sea-surface Temperature) | NOAA 海表温度再分析数据（1992–2019） | 180×360 网格，共 44,219 个空间点 |
| **ISO** (Isotropic Turbulent Flow) | JHTDB 中各向同性湍流压力场 | 350×350 空间切片，1,667 时间步 |
| **Neural Activity** | Allen Institute 小鼠视觉皮层 LFP 记录 | 多通道局部场电位信号 |
| **Solar Activity** | NASA Solar Dynamics Observatory 日冕图像（171 Å） | 274 帧太阳活动影像 |
| **1D RDE Transient Stage** | 旋转爆震发动机（Rotating Detonation Engine）瞬态模拟 | 一维压力与温度场，64 空间点，101 时间步 |

### 实验设置
- **传感器数量**：极稀疏设置，通常只用 **2–3 个随机放置的传感器**
- **时间滞后窗口 $ L $**：52–100 步（对应约一年或系统特征时间）
- **噪声维度 $ d_\epsilon $**：50–1000
- **Monte Carlo 样本数**：推理时生成 50–200 个噪声样本用于估计分布
- **网络架构**：沿用 SHRED 结构（GRU + 浅解码器），仅修改输入并更换损失函数

### 评估指标
| 指标 | 定义 | 目标 |
|------|------|------|
| **Calibration** | 观测覆盖率 vs 名义置信水平（如 95% CI 是否包含 95% 真值） | 接近对角线表示良好校准 |
| **CRPS** (Continuous Ranked Probability Score) | 衡量预测分布的整体质量（兼顾校准性和锐度） | 越低越好 |
| **Sharpness** | 预测区间宽度（如 95% CI 平均宽度） | 在正确校准前提下越窄越好 |
| **RMSE** | 中位数预测的均方根误差 | 衡量点估计精度 |

### 基线方法对比
- **Deterministic SHRED**：原始确定性版本，使用 MSE 损失
- **其他 UQ 方法**（隐含比较）：MC-Dropout、Ensemble、Heteroscedastic 回归等（文中指出其不足）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1 & 2）

#### 表 1：四大数据集上的 UQ-SHRED 性能汇总

| Dataset | RMSE | 95% Calibration (%) | CRPS | 95% CI Width |
|--------|------|--------------------|-------|--------------|
| **SST** | 0.656 | 90.8 | 0.343 | 2.791 |
| **ISO** | 0.656 | 85.7 | 0.032 | 0.209 |
| **Neural** | 3.54e-5 | 91.3 | 1.8e-5 | 1.14e-4 |
| **Solar** | 0.029 | 94.6 | 0.016 | 0.139 |

> 🔹 所有数据集上，**观测覆盖率接近名义水平**（如 95% CI 实际覆盖率为 90.8%~94.6%），表明高度**校准良好**（well-calibrated）  
> 🔹 **CRPS 较低**，说明概率预测质量高  
> 🔹 **CI 宽度随物理复杂性动态变化**：在剧烈变化区域（如激波、太阳耀斑）显著增宽

#### 表 2：1D RDE 上的外推泛化测试（Closer vs Further Trajectory）

| Test Case | RMSE (P/T) | 95% Coverage | CRPS | 95% CI Width |
|----------|------------|---------------|-------|--------------|
| Closer (Run 0) | 0.156 / 0.307 | 95.9% | 0.070 | 0.556 |
| Further (Run 1) | 0.194 / 0.398 | 91.5% | 0.096 | 0.569 |

> 🔹 即使面对较大初始扰动（distributional shift），仍保持良好覆盖能力  
> 🔹 不确定性带能准确反映**激波前沿**等高不确定性区域（见图 7 和 15）

### 与基线方法对比
- **Deterministic SHRED**：RMSE 略优（例如 Run 0 中 0.145 vs 0.156），但**完全无法提供不确定性信息**
- **外推稳定性对比**（图 9）：
  - Deterministic SHRED 在训练时间范围外预测发散严重，不同初始化之间方差大（平均高出 3.3 倍）
  - UQ-SHRED 预测分布稳定，体现 **engression 对外插任务的鲁棒性增强**

### 消融实验结果（Ablation Study on SST Data）

| 变量 | 发现 |
|------|------|
| **Temporal Lag $ L $** | RMSE 随 $ L $ 增加而下降（符合 Takens’ 嵌入定理），但 UQ 校准性对 $ L > 10 $ 不敏感 |
| **Sensor Count** | 更多传感器降低 RMSE，但 95% 覆盖率提升有限；建议增加 $ d_\epsilon $ 以匹配输入维度 |
| **Noise Dimension $ d_\epsilon $** | 过小导致预测过集中（under-dispersed）；过大增加计算负担但收益递减 |
| **Monte Carlo Sample Size** | ≥100 样本能有效减少分位数估计偏差；推荐至少 100 次采样 |
| **Training Epochs** | **关键发现**：即使在欠拟合或过拟合状态下，UQ-SHRED 仍能维持有效校准（95% 分位数实测覆盖率达 91.6%~91.8%）→ 显示能量得分损失具有**强健性** |

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **UQ-SHRED 成功实现了对 SHRED 的不确定性扩展**，能够在不牺牲效率的前提下输出**经过良好校准的概率预测**。
2. ✅ 所学不确定性具有**物理意义**：在系统快速变化、传感器约束弱的区域（如激波、季节转折点、神经尖峰），置信区间自动展宽。
3. ✅ 方法具备**强外推鲁棒性**：相比传统 MSE 训练，engression 损失促使模型在外推时产生一致且合理的不确定性估计。
4. ✅ **理论可证**：在正则条件下，Energy Score 最小化能收敛到真实条件分布（Thm. 1），且 MC 分位数估计具有一致性（Thm. 2）。

### 局限性
1. **有限样本下的校准可能偏离理想状态**：尤其在强分布偏移下（如 RDE Run 1），可能出现轻微**保守估计**（observed < expected）
2. **未提供有限样本覆盖保证**：当前理论基于总体最优，实际中可通过 **conformal calibration** 方法进一步改进
3. **噪声维度需手动调参**：尚无自动化机制决定最优 $ d_\epsilon $
4. **依赖 Takens’ 延迟嵌入有效性**：对于非平稳或记忆极长的动力系统，性能可能受限

### 未来工作方向
- 结合 **conformal prediction** 技术实现有限样本下的精确覆盖控制
- 探索自适应噪声维度选择策略
- 将 UQ-SHRED 应用于更多实时监测与控制系统（如气候预警、航天推进）
- 扩展至多尺度或多模态传感场景（如融合卫星与地面站数据）

---

> 📌 **总结一句话**：  
> **UQ-SHRED 是首个将 engression 引入稀疏感知任务的工作，在保持 SHRED 高效性的同时，提供了理论可靠、物理合理、易于部署的不确定性量化能力，为安全关键型科学机器学习应用开辟了新路径。**

🔗 **代码开源地址**：[https://github.com/gaoliyao/uq_shred](https://github.com/gaoliyao/uq_shred)

</details>

---

### 13. [Detecting Complex Money Laundering Patterns with Incremental and Distributed Graph Modeling](https://arxiv.org/abs/2604.01315)

**Authors**: Haseeb Tariq, Alen Kaja, Marwan Hassani  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.01315v1  

#### Abstract
Money launderers take advantage of limitations in existing detection approaches by hiding their financial footprints in a deceitful manner. They manage this by replicating transaction patterns that the monitoring systems cannot easily distinguish. As a result, criminally gained assets are pushed int...

---

### 14. [World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry](https://arxiv.org/abs/2604.01985)

**Authors**: Yuejiang Liu, Fan Feng, Lingjing Kong, Weifeng Lu, Jinzhou Tang, Kun Zhang, Kevin Murphy, Chelsea Finn, Yilun Du  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.01985v1  

#### Abstract
General-purpose world models promise scalable policy evaluation, optimization, and planning, yet achieving the required level of robustness remains challenging. Unlike policy learning, which primarily focuses on optimal actions, a world model must be reliable over a much broader range of suboptimal ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前的 **action-conditioned world model**（动作条件化世界模型）在机器人学习中面临一个核心挑战：**robust action following**（鲁棒的动作跟随）。  
- 与仅需建模最优动作的策略学习不同，world model 必须对**广泛的子最优、探索性甚至随机动作**都具备可靠的预测能力。
- 然而，大规模收集带动作标签的机器人交互数据成本高、耗时长且存在安全风险，导致在**未充分探索（under-explored）区域**，模型预测不可靠。

此外，现有的 **exploration for world model** 方法（如基于不确定性或学习进度的方法）通常在已探索区域表现良好，但在最需要验证的未知区域却不可靠。

---

### **提出的新方法与核心思想**
作者提出了 **World Action Verifier (WAV)**，一种通过 **forward-inverse asymmetry**（前向-逆向不对称性）实现自改进的世界模型框架。

#### **核心创新点：**
1. **将 world model 验证问题分解为两个更易处理的子问题：**
   - **State Plausibility**（状态合理性）：预测的状态是否在物理和视觉上合理？
   - **Action Reachability**（动作可达性）：从当前状态到预测状态的转移是否能由给定动作实现？

2. **利用两种关键的“不对称性”来分别验证这两个因素：**
   - **数据可用性不对称（Distribution Asymmetry）**：  
     - 无动作标签的视频数据（`D_vid`）远多于带标签的交互数据（`D_act`）。
     - 利用 `D_vid` 训练一个 **diverse subgoal generator** 来生成合理的未来状态作为参考。
   - **维度不对称（Dimensionality Asymmetry）**：  
     - 动作通常只影响状态空间中的少数关键特征（如机械臂末端位姿、物体运动）。
     - 训练一个 **sparse inverse model**，仅从这些低维相关特征中推断动作，显著降低逆向建模难度。

3. **构建自改进循环（Self-Improving Cycle）：**
   - **反向循环（Reverse Cycle）**：先由 subgoal generator 生成目标状态 → 由 inverse model 推断实现该转移所需动作 → 由 world model 执行前向预测 → 比较预测状态与生成目标状态的一致性。
   - 不一致性（discrepancy）被用作 **verification loss**，指导选择最具信息量的交互进行数据采集，从而实现模型自改进。

---

### **相比现有方法的优势**
- **更可靠的验证机制**：在未探索区域仍能提供可靠误差估计，克服了传统方法在未知区域失效的问题。
- **更高的样本效率**：通过精准选择信息量大的样本，以更少的数据实现更好的模型性能。
- **更强的泛化能力**：利用无标签视频数据和稀疏逆模型，提升了对新对象和新交互的泛化能力。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖了三个基准环境，涵盖网格世界和模拟机器人操作任务：
- **MiniGrid**：用于可控消融实验，研究泛化性和鲁棒性。
- **RoboMimic**：包含 `Lift`, `Can`, `Square` 三个模拟机器人操作任务。
- **ManiSkill**：包含 `PullCube`, `PokeCube`, `LiftPeg` 三个更具挑战性的操作任务。

---

### **实验设置与评估指标**

#### **训练流程**
1. **预热阶段**：在少量均匀采样的带标签数据（如 200 条轨迹）上预训练 world model。
2. **探索阶段**：执行多轮探索，每轮使用不同策略选择 100 条新轨迹进行数据采集，并更新模型。
3. **评估阶段**：在独立测试集上评估 world model 的预测精度和下游策略性能。

#### **评估指标**
- **World Model 性能**：
  - **Prediction MSE**：下一观测的均方误差。
  - **Dynamics Accuracy**：仅对动态变化部分（如物体位置、携带状态）的预测准确率。
- **数据选择质量**：
  - **Spearman/Kendall Rank Correlation**：与“Oracle”（基于真实误差排序）的相关性，衡量数据选择策略的有效性。
- **下游策略性能**：
  - **Average Reward**：使用 SAILOR 协议，在想象中规划并微调策略后的平均奖励。

---

### **基线方法对比**
- **Random**：随机选择候选样本。
- **Uncertainty**：选择预测不确定性最高的样本。
- **Progress**：选择学习进度（loss 下降最多）最大的样本。
- **Vanilla IDM**：不加稀疏约束的逆动力学模型。
- **Oracle**：使用真实标签计算误差，选择误差最大的样本（理论上限）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **样本效率提升**：WAV 在 **9 个任务**上实现了 **2× 更高的样本效率**。
- **下游策略性能提升**：相比最强基线，**平均奖励提高 18%**。

---

### **与基线方法的对比结果**
- **在 MiniGrid 上**（图 4）：
  - WAV 的数据选择策略与 Oracle 的 Spearman 相关系数最高，显著优于 Uncertainty 和 Progress。
  - 在第二轮探索后，WAV 的 world model 预测误差远低于其他方法。
- **在 RoboMimic 和 ManiSkill 上**（图 6）：
  - WAV 在所有数据预算下均优于基线，尤其在低数据 regime 下优势明显。
  - 例如，在 `Can` 和 `PokeCube` 任务上，WAV 的 MSE 显著更低。
- **下游策略性能**（图 7）：
  - 使用 WAV 训练的 world model 进行策略微调，最终性能仅次于 Oracle，远超其他基线。
  - 在接触复杂度高的任务（如 `Can`, `Square`, `PokeCube`）上提升尤为显著。

---

### **消融实验结果**
- **Sparse IDM vs. Vanilla IDM**（图 4 左）：
  - 在有限数据下，**sparse IDM** 在 out-of-distribution 泛化上表现更好，证明稀疏性增强了鲁棒性。
- **维度、噪声、样本量的影响**（图 3）：
  - **维度增加**（更多物体）：WM 性能急剧下降，IDM 保持稳定。
  - **环境噪声增加**（noisy floor）：WM 对噪声敏感，IDM 几乎不受影响。
  - **样本量减少**：IDM 在低数据下始终优于 WM，验证了其样本效率优势。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Forward-inverse asymmetry 是有效的**：  
   - 在高维、随机性强、标注数据稀缺的环境中，**从低维状态特征中推断动作（inverse）比预测完整未来状态（forward）更容易且更鲁棒**。
2. **WAV 实现了高效的自改进**：  
   - 通过 subgoal generator 和 sparse inverse model 构建的反向循环，能够可靠地识别 world model 的预测错误，并引导采集最具信息量的交互数据。
3. **显著提升样本效率与下游性能**：  
   - 在多个任务上实现了 **2× 样本效率** 和 **18% 策略性能提升**，验证了方法的通用性和有效性。

---

### **方法的局限性**
- **计算开销较大**：需要三次推理（subgoal → inverse → forward），比单次 forward 模型更慢。
- **依赖高质量的 subgoal generator**：若生成的状态不合理，会影响验证效果。
- **稀疏性假设可能不总是成立**：在某些任务中，动作可能影响广泛的状态特征，稀疏逆模型可能失效。
- **对反馈的依赖**：仍需环境的真实反馈来纠正错误，尚未实现完全的合成数据自改进。

---

### **未来工作方向**
- **提升计算效率**：通过共享中间表示或自适应计算机制降低推理成本。
- **扩展到更复杂的任务**：应用于长视野、具身智能体等更复杂的场景。
- **结合更强的生成先验**：集成更强大的视频生成模型（如 diffusion models）作为 subgoal generator。
- **探索完全自监督的改进路径**：减少对环境真实反馈的依赖，迈向真正的自我进化 world model。

</details>

---

### 15. [Batched Contextual Reinforcement: A Task-Scaling Law for Efficient Reasoning](https://arxiv.org/abs/2604.02322)

**Authors**: Bangji Yang, Hongbo Ma, Jiajun Fan, Ge Liu  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.02322v1  

#### Abstract
Large Language Models employing Chain-of-Thought reasoning achieve strong performance but suffer from excessive token consumption that inflates inference costs. Existing efficiency methods such as explicit length penalties, difficulty estimators, or multi-stage curricula either degrade reasoning qua...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Batched Contextual Reinforcement: A Task-Scaling Law for Efficient Reasoning

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在采用 Chain-of-Thought（CoT）推理时表现出强大的性能，但存在严重的**过度生成（overthinking）**问题，即产生冗长且重复的推理链，导致推理成本高昂、延迟增加。现有的效率优化方法如显式长度惩罚（explicit length penalties）、难度估计器（difficulty estimators）或多阶段课程学习（multi-stage curricula），要么会损害推理质量，要么需要复杂的训练流程。

### 提出的新方法：Batched Contextual Reinforcement (BCR)
作者提出了一种极简的单阶段训练范式——**Batched Contextual Reinforcement (BCR)**。其核心思想是通过一个简单的结构修改来激发模型的高效推理能力：
- **多任务批处理**：在训练时，将 `N` 个数学问题打包成一个“问题组”（problem group），放入同一个上下文窗口中，要求模型一次性连续解决所有问题。
- **共享资源约束**：为整个输出设定一个固定的 token 预算（token budget）。如果前面的问题用光了预算，后面的问题就无法完成，从而获得零分。
- **奖励机制**：仅基于每个问题的正确性（per-instance accuracy）进行奖励，**不引入任何显式的长度惩罚或辅助模型**。

这种方法创造了一个**隐式的资源竞争环境**，迫使模型自主学会如何分配推理深度、压缩冗余思考，并优先提高信息密度。

### 相比现有方法的优势
| 特性 | BCR | 显式长度惩罚 (e.g., L1, ShorterBetter) | 辅助模型 (e.g., ARM, SelfBudgeter) | 多阶段课程 (e.g., ProRL, BroRL) |
| :--- | :--- | :--- | :--- | :--- |
| **训练复杂度** | 单阶段，简单 | 单阶段，但有对抗梯度风险 | 多组件，需额外模型 | 多阶段，超参数调优复杂 |
| **是否需要长度监督** | 否 | 是 | 是 | 是 |
| **是否需要难度标签** | 否 | 否 | 是 | 通常需要 |
| **优化稳定性** | 高（无对抗梯度） | 低（易崩溃） | 中等 | 中等 |
| **可组合性** | 高（正交于其他技术） | 低 | 低 | 低 |

**核心优势**：BCR 无需任何显式的效率信号，仅通过结构激励就能稳定地诱导出高效的推理策略，是一种高度实用且稳定的替代方案。

## 2. 核心实验方法和设置

### 数据集
- **训练数据**：从 `DeepMath-103K` 数据集中采样构建 3,000 个问题组，每组包含 `N=3` 个问题。采用**分层采样**（stratified sampling）确保每组平均难度相近。
- **评估基准**：在五个主流数学推理基准上测试，涵盖不同难度：
  - `AIME25`, `AMC23`: 竞赛级数学
  - `Minerva`, `MATH-500`: 综合数学问题
  - `Olympiad`: 奥林匹克级别双语科学问题

### 实验设置
- **基础模型**：在两个不同规模的模型上验证：
  - `JustRL-DeepSeek-1.5B`
  - `Qwen3-4B-Thinking-2507`
- **训练算法**：使用 `GRPO`（Group Relative Policy Optimization）进行强化学习。
- **输入格式**：使用 Markdown 结构化提示（如 `### Problem X` 和 `AnswerX::\boxed{...}`），便于答案提取。
- **Token 预算**：
  - 1.5B 模型：`5120` tokens / 3 问题 ≈ 1707 tokens/问题
  - 4B 模型：`8000` tokens / 3 问题 ≈ 2667 tokens/问题
- **评估指标**：
  - **准确率 (Accuracy %)**：标准数学准确率。
  - **平均每问题生成 Token 数 (Avg. Tokens per problem)**：衡量推理效率。

### 基线方法对比
- **通用 LLM**: `Qwen3-1.7B`
- **纯数学 LLM**: `STILL-3-1.5B`
- **带长度控制的 LLM**: `BroRL-1.5B`, `e3-1.7B`
- **自适应推理模型**: `ARM-3B`, `Thinker-Q1.5B`
- **直接对比基线**: `JustRL-deepseek-1.5B`, `Qwen3-4B-Thinking-2507`

## 3. 主要实验结果和性能指标

### 关键性能数据（N=1 推理）
在标准单问题（N=1）推理下，BCR 在显著降低 Token 消耗的同时，保持甚至提升了准确率。

#### 与 `JustRL-1.5B` 对比
| Benchmark | 准确率变化 | Token 减少 |
| :--- | :--- | :--- |
| AIME25 | +3.3% | -62.6% |
| AMC23 | +2.5% | -53.8% |
| Minerva | +5.1% | -53.9% |
| MATH-500 | -3.8% | -39.8% |
| Olympiad | +0.8% | -58.0% |

#### 与 `Qwen3-4B` 对比
| Benchmark | 准确率变化 | Token 减少 |
| :--- | :--- | :--- |
| AIME25 | **+13.3%** | -15.8% |
| AMC23 | +2.5% | -31.8% |
| Minerva | +2.2% | -27.1% |
| MATH-500 | +0.4% | -27.7% |
| Olympiad | +1.9% | -18.0% |

> **关键观察**：在 4B 模型上，BCR 在所有 5 个基准上都实现了“免费午餐”（free lunch），即**准确率提升的同时，Token 消耗大幅下降**。

### 任务扩展定律（Task-Scaling Law）
当在推理时增加并发问题数 `N`，BCR 展现出一种新的扩展规律：
- **Per-problem Token Usage** 随 `N` 增加而**单调递减**。
- **准确率**随 `N` 增加而**缓慢下降**，远优于基线。

例如，在 `AIME25` 上，当 `N=4` 时，BCR 的 Token 消耗比基线减少 **75%**，同时准确率更高。

这表明 `N` 可作为一个可控的**吞吐量-准确率调节旋钮**，类似于传统计算中的 batch size。

### 消融实验结果
- **训练组大小 `N` 扫描**：在 `N=3,4,5` 上训练，发现在 `N=3` 时通常能取得最佳的准确率-效率平衡，且性能对 `N` 的选择具有鲁棒性。
- **隐式 vs. 显式长度控制**：显式长度惩罚（如 `r_len = -|y|/maxlen`）会导致**灾难性优化崩溃**（catastrophic training collapse），准确率迅速降至零。而 BCR 的隐式预算约束则保持了高度稳定的优化过程。
- **Token 预算大小**：预算过小（如 4096）会因过度压缩而损害准确率；预算过大（如 6144）则削弱了效率增益。`5120` 被证明是一个良好的折衷。

## 4. 关键结论和发现

### 主要发现
1. **任务扩展定律 (Task-Scaling Law)**：并发问题数 `N` 是一个新的推理效率扩展维度。增加 `N` 可系统性地降低每问题的 Token 消耗，且准确率下降平缓。
2. **“免费午餐”现象 (Free Lunch)**：在多个基准上，BCR 不仅大幅降低了 Token 消耗（最高达 62.6%），还**一致地维持或提升了准确率**。这挑战了传统的准确率-效率权衡假设，表明**冗余的推理步骤可能是有害的**。
3. **涌现的自我调节效率 (Emergent Self-Regulated Efficiency)**：模型自发地学会了消除元认知循环（如“等等，让我再检查一下…”）、直接选择最优策略、并防止灾难性退化。这是一种纯粹由结构激励引发的**语法级压缩**，而非语义上的跳步。
4. **隐式约束优于显式惩罚**：硬性的 token 预算约束从根本上避免了显式长度惩罚带来的对抗梯度问题，提供了更稳定、更优越的优化路径。
5. **效率源于结构，而非监督**：LLMs 本身具备高效推理的潜力，但标准的单问题训练未能激活它。BCR 通过多任务资源竞争的结构设计，成功解锁了这种**潜在的高密度推理模式**。

### 方法的局限性
- **已验证领域有限**：目前仅在 1.5B 和 4B 规模的数学推理任务上进行了验证。
- **未探索异构任务**：所有问题组内的任务都是同质的数学问题，未研究混合不同类型任务的效果。
- **依赖结构化解析**：需要精确的答案提取机制（如栈式解析器）来分离多个问题的答案。

### 未来工作方向
- 将 BCR 扩展到更大规模的模型（7B-70B）和其他推理领域，如代码生成、科学推理和多模态任务。
- 研究任务扩展定律的理论基础，以及它是否适用于异构任务混合。
- 探索将 BCR 与其他效率技术（如自适应 Token 分配、投机解码）结合，以实现进一步的性能提升。
- 研究环境结构如何替代显式监督，以解锁 LLMs 的更多潜在能力。

</details>

---

### 16. [A Safety-Aware Role-Orchestrated Multi-Agent LLM Framework for Behavioral Health Communication Simulation](https://arxiv.org/abs/2604.00249)

**Authors**: Ha Na Cho  
**Category**: cs.AI  
**Published**: 2026-04-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.00249v1  

#### Abstract
Single-agent large language model (LLM) systems struggle to simultaneously support diverse conversational functions and maintain safety in behavioral health communication. We propose a safety-aware, role-orchestrated multi-agent LLM framework designed to simulate supportive behavioral health dialogu...

---

### 17. [Decision-Centric Design for LLM Systems](https://arxiv.org/abs/2604.00414)

**Authors**: Wei Sun  
**Category**: cs.AI  
**Published**: 2026-04-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.00414v1  

#### Abstract
LLM systems must make control decisions in addition to generating outputs: whether to answer, clarify, retrieve, call tools, repair, or escalate. In many current architectures, these decisions remain implicit within generation, entangling assessment and action in a single model call and making failu...

---

### 18. [TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models](https://arxiv.org/abs/2604.00666)

**Authors**: Lingjie Chen, Ruizhong Qiu, Yuyu Fan, Yanjun Zhao, Hanghang Tong  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.00666v1  

#### Abstract
Diffusion language models (DLMs) offer a promising path toward low-latency generation through parallel decoding, but their practical efficiency depends heavily on the decoding trajectory. In practice, this advantage often fails to fully materialize because standard training does not provide explicit...

---

### 19. [DISCO-TAB: A Hierarchical Reinforcement Learning Framework for Privacy-Preserving Synthesis of Complex Clinical Data](https://arxiv.org/abs/2604.01481)

**Authors**: Arshia Ilaty, Hossein Shirazi, Amir Rahmani, Hajar Homayouni  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.01481v1  

#### Abstract
The development of robust clinical decision support systems is frequently impeded by the scarcity of high-fidelity, privacy-preserving biomedical data. While Generative Large Language Models (LLMs) offer a promising avenue for synthetic data generation, they often struggle to capture the complex, no...

---

### 20. [Learning from the Right Rollouts: Data Attribution for PPO-based LLM Post-Training](https://arxiv.org/abs/2604.01597)

**Authors**: Dong Shu, Denghui Zhang, Jessica Hullman  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.01597v1  

#### Abstract
Traditional RL algorithms like Proximal Policy Optimization (PPO) typically train on the entire rollout buffer, operating under the assumption that all generated episodes provide a beneficial optimization signal. However, these episodes frequently contain noisy or unfaithful reasoning, which can deg...

---

### 21. [Transformer self-attention encoder-decoder with multimodal deep learning for response time series forecasting and digital twin support in wind structural health monitoring](https://arxiv.org/abs/2604.01712)

**Authors**: Feiyu Zhou, Marios Impraimakis  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.01712v1  

#### Abstract
The wind-induced structural response forecasting capabilities of a novel transformer methodology are examined here. The model also provides a digital twin component for bridge structural health monitoring. Firstly, the approach uses the temporal characteristics of the system to train a forecasting m...

---

### 22. [SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization](https://arxiv.org/abs/2604.02268)

**Authors**: Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Chengcheng Han, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.02268v1  

#### Abstract
Agent skills, structured packages of procedural knowledge and executable resources that agents dynamically load at inference time, have become a reliable mechanism for augmenting LLM agents. Yet inference-time skill augmentation is fundamentally limited: retrieval noise introduces irrelevant guidanc...

---

### 23. [Adaptive Parallel Monte Carlo Tree Search for Efficient Test-time Compute Scaling](https://arxiv.org/abs/2604.00510)

**Authors**: Hongbeen Kim, Juhyun Lee, Sanghyeon Lee, Kwanghoon Choi, Jaehyuk Huh  
**Category**: cs.AI  
**Published**: 2026-04-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.00510v1  

#### Abstract
Monte Carlo Tree Search (MCTS) is an effective test-time compute scaling (TTCS) method for improving the reasoning performance of large language models, but its highly variable execution time leads to severe long-tail latency in practice. Existing optimizations such as positive early exit, reduce la...

---

### 24. [More Human, More Efficient: Aligning Annotations with Quantized SLMs](https://arxiv.org/abs/2604.00586)

**Authors**: Jiayu Wang, Junyoung Lee  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.00586v1  

#### Abstract
As Large Language Model (LLM) capabilities advance, the demand for high-quality annotation of exponentially increasing text corpora has outpaced human capacity, leading to the widespread adoption of LLMs in automatic evaluation and annotation. However, proprietary LLMs often exhibit systematic biase...

---

### 25. [LangMARL: Natural Language Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2604.00722)

**Authors**: Huaiyuan Yao, Longchao Da, Xiaoou Liu, Charles Fleming, Tianlong Chen, Hua Wei  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.00722v1  

#### Abstract
Large language model (LLM) agents struggle to autonomously evolve coordination strategies in dynamic environments, largely because coarse global outcomes obscure the causal signals needed for local policy refinement. We identify this bottleneck as a multi-agent credit assignment problem, which has l...

---

### 26. [LLM REgression with a Latent Iterative State Head](https://arxiv.org/abs/2604.01206)

**Authors**: Yiheng Su, Matthew Lease  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.01206v1  

#### Abstract
We present RELISH (REgression with a Latent Iterative State Head), a novel, lightweight architecture designed for text regression with large language models. Rather than decoding numeric targets as text or aggregating multiple generated outputs, RELISH predicts scalar values directly from frozen LLM...

---

### 27. [A Practical Two-Stage Framework for GPU Resource and Power Prediction in Heterogeneous HPC Systems](https://arxiv.org/abs/2604.02158)

**Authors**: Beste Oztop, Dhruva Kulkarni, Zhengji Zhao, Ayse Kivilcim Coskun, Kadidia Konate  
**Category**: cs.DC  
**Published**: 2026-04-03  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.02158v1  

#### Abstract
Efficient utilization of GPU resources and power has become critical with the growing demand for GPUs in high-performance computing (HPC). In this paper, we analyze GPU utilization and GPU memory utilization, as well as the power consumption of the Vienna ab initio Simulation Package (VASP), using t...

---

### 28. [Does Unification Come at a Cost? Uni-SafeBench: A Safety Benchmark for Unified Multimodal Large Models](https://arxiv.org/abs/2604.00547)

**Authors**: Zixiang Peng, Yongxiu Xu, Qinyi Zhang, Jiexun Shen, Yifan Zhang, Hongbo Xu, Yubin Wang, Gaopeng Gou  
**Category**: cs.AI  
**Published**: 2026-04-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.00547v1  

#### Abstract
Unified Multimodal Large Models (UMLMs) integrate understanding and generation capabilities within a single architecture. While this architectural unification, driven by the deep fusion of multimodal features, enhances model performance, it also introduces important yet underexplored safety challeng...

---

### 29. [Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts](https://arxiv.org/abs/2604.00901)

**Authors**: Sha Li, Naren Ramakrishnan  
**Category**: cs.AI  
**Published**: 2026-04-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.00901v1  

#### Abstract
Multi-agent Retrieval-Augmented Generation (RAG), wherein each agent takes on a specific role, supports hard queries that require multiple steps and sources, or complex reasoning. Existing approaches, however, rely on static agent behaviors and fixed orchestration strategies, leading to brittle perf...

---

### 30. [An Online Machine Learning Multi-resolution Optimization Framework for Energy System Design Limit of Performance Analysis](https://arxiv.org/abs/2604.01308)

**Authors**: Oluwamayowa O. Amusat, Luka Grbcic, Remi Patureau, M. Jibran S. Zuberi, Dan Gunter, Michael Wetter  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.01308v1  

#### Abstract
Designing reliable integrated energy systems for industrial processes requires optimization and verification models across multiple fidelities, from architecture-level sizing to high-fidelity dynamic operation. However, model mismatch across fidelities obscures the sources of performance loss and co...

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
