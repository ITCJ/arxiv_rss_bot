# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-03 06:55:52 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Taming the Exponential: A Fast Softmax Surrogate for Integer-Native Edge Inference](https://arxiv.org/abs/2604.02292)

**Authors**: Dimitrios Danopoulos, Enrico Lupi, Michael Kagan, Maurizio Pierini  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.02292v1  

#### Abstract
Softmax can become a computational bottleneck in the Transformer model's Multi-Head Attention (MHA) block, particularly in small models under low-precision inference, where exponentiation and normalization incur significant overhead. As such, we suggest using Head-Calibrated Clipped-Linear Softmax (...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Taming the Exponential: A Fast Softmax Surrogate for Integer-Native Edge Inference**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在基于Transformer的模型中，**Softmax** 是 Multi-Head Attention (MHA) 模块的关键组成部分，用于将注意力logits转换为概率分布。然而，在低精度边缘推理场景下（尤其是使用 int8 量化的小型模型），标准Softmax因涉及**指数运算（exp）和归一化操作**，成为计算瓶颈。  
具体问题包括：
- **高开销的浮点运算**：即使输入是 int8，Softmax 通常仍需转换为 bfloat16 或 FP32 进行 exp 计算，带来大量类型转换开销。
- **硬件利用率低下**：AMD Versal AI Engine 虽具备高效的 int8 MAC 单元，但现有Softmax实现依赖 LUT 或 bfloat16 exp 操作，无法充分利用其整数向量处理能力。
- **内存带宽受限**：LUT-based 方法受限于访存吞吐。

### **提出了什么新方法或新思路**
提出 **Head-Calibrated Clipped-Linear Softmax (HCCS)**，一种专为量化MHA设计的、**无需显式指数运算**的Softmax替代方案，具有以下特点：

- **Clipped-Linear 映射**：用分段线性函数替代 exp，形式为 $ s_i = B_h - S_n \cdot \delta_i $，其中 $\delta_i = \min(\max x - x_i, D_{\text{max},h})$，保证单调性和非负性。
- **Head-Wise Calibration**：每个 attention head 独立优化参数 $(B_h, S_n, D_{\text{max},h})$，通过离线网格搜索最小化 KL 散度，适配不同head的统计特性。
- **完全整数域实现**：所有计算（包括归一化）均在 int8/int16 下完成，避免浮点转换。
- **硬件友好设计**：直接映射到 AI Engine 的 int8 MAC 流水线，支持 CLB（leading-bit detection）加速倒数计算。

### **相比现有方法的优势**
| 对比维度 | HCCS | 现有方法（如 BF16 Softmax、LUT-based） |
|--------|------|-----------------------------|
| **计算类型** | 完全整数运算（int8 MAC） | 需要浮点 exp 或 LUT 查表 |
| **硬件适配性** | 充分利用 AI Engine 的 int8 向量单元 | 受限于 exp 单元或 LUT 带宽 |
| **吞吐量** | 显著更高（最高达 15.1×） | 较低，尤其在短序列时 |
| **部署灵活性** | 参数固定，可QAT后固化 | 多数不可微或需额外训练 |
| **能效潜力** | 更少的数据搬移和类型转换 | 类型转换开销大 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **SST-2**：二分类情感分析任务，最大序列长度 64。
- **MNLI**：自然语言推断任务，最大序列长度 128（因输入为句子对）。

### **模型**
- **BERT-tiny**：2层，2头，hidden=128
- **BERT-small**：4层，8头，hidden=512  
均为适用于边缘设备的小型Transformer模型。

### **实验设置**
- **硬件平台**：AMD Versal AI Engine（VEK280 和 VEK385），使用 **cycle-accurate AIE simulator**（Vitis 2025.2）进行测量。
- **输入建模**：数据通过 PLIO 直接输入，排除 PS/DDR 传输开销，符合流式推理场景。
- **量化方式**：int8 量化，采用 Quantization-Aware Training (QAT)。
- **输出格式**：
  - `i16+div`：int16 输出，使用精确整数除法归一化。
  - `i8+CLB`：int8 输出，使用 CLB（leading-bit detection）近似倒数。

### **评估指标**
- **任务准确率**（Validation Accuracy）：下游任务性能。
- **KL 散度**：衡量 HCCS 与 float32 Softmax 分布的差异。
- **吞吐量**（Throughput）：单位为 elements/s，反映 kernel 级效率。
- **消融实验**：验证 calibration granularity（全局 vs 层级 vs 头级）的影响。

### **基线方法对比**
- **AMD 官方参考实现**：BF16 Softmax kernel
  - 在 AIE-ML 上使用 LUT-based exp
  - 在 AIE-MLv2 上使用原生 BF16 exp 指令

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **任务准确率（Table I）**
| Task | Model | Baseline | HCCS (No Retrain) | HCCS (Retrained) | Δ (vs Baseline) |
|------|-------|----------|-------------------|------------------|-----------------|
| SST-2 | BERT-tiny | 0.825 | 0.619 | **0.822** | -0.003 |
| SST-2 | BERT-small | 0.893 | 0.766 | **0.878** | -0.015 |
| MNLI | BERT-tiny | 0.653 | 0.480 | **0.639** | -0.013 |
| MNLI | BERT-small | 0.742 | 0.602 | **0.723** | -0.019 |

✅ **结论**：经过 QAT 后，HCCS 模型准确率仅比 float32 基线低 **0.3–1.9个百分点**，恢复了绝大多数性能。

#### **吞吐量对比（Table III）**
| Sequence Length | Platform | BF16 (elems/s) | HCCS i16+div (Speedup) | HCCS i8+CLB (Speedup) |
|-----------------|----------|----------------|-------------------------|------------------------|
| 32 | AIE-ML | 0.09G | 4.6× | **15.1×** |
| 64 | AIE-ML | 0.16G | 4.9× | **13.7×** |
| 128 | AIE-ML | 0.25G | 5.5× | **8.72×** |
| 32 | AIE-MLv2 | 0.24G | 1.7× | **6.1×** |
| 128 | AIE-MLv2 | 0.77G | 1.8× | **2.9×** |

✅ **结论**：
- HCCS 在所有配置下均显著超越 BF16 基线。
- `i8+CLB` 版本最快，**最高达 15.1× 加速**（AIE-ML, seq=32）。
- 短序列增益更大，因 exp 开销占比更高。

#### **多核扩展性（Figure 3）**
- 支持 **multi-tile scaling**，吞吐随 AI Engine 数量线性增长。
- 最高可达：
  - HCCS (i16+div): **259 G elements/s**
  - HCCS (i8+CLB): **407 G elements/s**（184 tiles）
- 表明 HCCS 具备极强并行扩展能力。

#### **消融实验结果**
- **校准粒度影响（Table II）**
  | Calibration Mode | SST-2 (BERT-small) | MNLI (BERT-small) |
  |------------------|--------------------|---------------------|
  | Shared/global | 0.834 | 0.545 |
  | Per-layer | 0.842 | 0.602 |
  | **Per-head (HCCS)** | **0.878** | **0.723** |

✅ **结论**：**头级校准（per-head）效果最好**，尤其在异构性强的任务（如 MNLI）上优势明显。

- **归一化方式影响**
  - `i8+CLB` 与 `i16+div` 准确率几乎无差异，说明 CLB 近似对最终任务性能影响可忽略。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Softmax 可被高效整数代理替代**：HCCS 证明无需 exp 也能实现高质量注意力归一化。
2. ✅ **头级校准至关重要**：不同 attention head 分布差异大，统一参数会损害性能；**per-head calibration 显著提升准确率**。
3. ✅ **硬件映射决定效率上限**：将算法设计与 AI Engine 的 int8 MAC 架构对齐，是实现高吞吐的关键。
4. ✅ **QAT 是必要环节**：直接替换 HCCS 不重训会导致严重掉点（↓20%+），但轻量级 QAT 即可恢复性能。
5. ✅ **HCCS 是首个面向 AMD AI Engine 的 int8 优化 Softmax 替代方案**，填补了该平台在整数量化推理中的空白。

### **方法的局限性**
- **适用范围**：目前主要针对小型、encoder-only 模型（如 BERT-tiny/small），在大型 decoder 模型（如 LLM）上的泛化能力待验证。
- **静态参数**：校准参数在训练后固定，不支持动态调整；虽有利于部署，但牺牲了一定适应性。
- **KL 散度非唯一目标**：校准阶段使用 KL 散度作为代理目标，但最终仍需依赖下游任务微调。

### **未来工作方向**
- 探索 **可学习的 HCCS 参数**（differentiable calibration），在训练中联合优化。
- 扩展至 **decoder 结构和生成任务**（如机器翻译）。
- 将 HCCS 集成进完整的 **端到端 Transformer 编译器流程**（如 Vitis AI 或 hls4ml）。
- 探索与其他 **non-softmax attention 替代方案**（如 ReLA、entmax）的结合。

---

> **总结一句话**：  
> HCCS 提出了一种**硬件感知、头级校准、纯整数实现**的 Softmax 替代方案，在保持接近原始模型准确率的同时，**在 AMD Versal AI Engine 上实现了高达 15.1× 的吞吐提升**，为边缘侧高效 Transformer 推理提供了新的可行路径。

</details>

---

### 2. [Stochastic Attention: Connectome-Inspired Randomized Routing for Expressive Linear-Time Attention](https://arxiv.org/abs/2604.00754)

**Authors**: Zehao Jin, Yanan Sui  
**Category**: cs.CL  
**Published**: 2026-04-03  
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
传统 **Sliding Window Attention (SWA)** 虽然在计算复杂度上为 $O(nw)$，适合长序列建模，但由于其**局部性限制**，每层只能看到固定窗口内的上下文，导致多层堆叠后感受野增长缓慢（线性增长 $lw$），难以实现全局信息交互。这在需要长程依赖的任务（如语言建模）中成为瓶颈。

现有改进方案（如引入 global tokens、稀疏模式或 block-level routing）往往增加架构复杂性或参数量。

---

### 提出的新方法与核心思想
受果蝇全脑连接组（**fruit fly connectome**）启发，作者提出 **Stochastic Attention (SA)** ——一种**无参数、即插即用**的 SWA 增强机制。

#### 核心机制：
- 在每一层 **应用前随机打乱 token 顺序**（Random Permutation）
- 在打乱后的序列上执行标准的 **Sliding Window Attention**
- 注意力输出后再通过逆排列恢复原始顺序

> 这相当于将固定的局部窗口变为“**随机的全局窗口**”，每个 token 都有均等概率关注到任意其他位置。

#### 结合策略：SA + SWA
进一步提出一个**双通路门控融合结构**（gated SA + SWA）：
- 并行运行 SA 和 SWA 分支
- 使用两个独立的 sigmoid gate 动态加权两者的输出
- 保留 SWA 的局部连贯性 + SA 的随机长程跳跃能力

---

### 相比现有方法的优势
| 特性 | SA | SWA | Full Attention | MoBA |
|------|----|-----|----------------|-------|
| 复杂度 | $O(nw)$ | $O(nw)$ | $O(n^2)$ | $O(nw)$ |
| 参数增量 | 0 | 0 | 0 | 有 |
| 感受野增长 | **指数级** $O(\log_w n)$ | 线性 $O(lw)$ | 全局（1层） | 依赖路由策略 |
| 实现难度 | 极低（仅 index shuffle） | 标准 | 高 | 中等 |
| 可训练性 | 支持端到端 | 支持 | 支持 | 支持 |
| **是否可作为训练后替换** | ✅ 是 | ✅ 是 | ❌ 否 | ✅ 是 |

> ✅ SA 在保持 $O(nw)$ 成本的同时实现了接近 full attention 的表达能力，并且可以**无需重新训练直接用于预训练模型**。

---

## 2. 核心实验方法和设置

### 数据集
1. **预训练实验**：
   - **SlimPajama**：6B token 子集，共训练约 15B tokens
   - 序列长度：2048

2. **训练后推理实验（Training-free inference）**：
   - 使用已发布的 **Qwen3-8B** 和 **Qwen3-30B-A3B** 模型
   - 在以下基准测试中进行 zero-shot 或 generation 测试：
     - **HellaSwag**, **MMLU**, **ARC-Easy/Challenge**, **BoolQ**, **LAMBADA**, **HumanEval**

---

### 实验设置与评估指标
| 设置项 | 描述 |
|--------|------|
| 模型规模（预训练） | ~360M 参数，24 层，d=1024，H=16，w=256 |
| Attention 变体对比 | Full Attention, SWA, SA, SA+SWA |
| 优化器 | AdamW ($\beta_1=0.9, \beta_2=0.95$)，峰值学习率 $3\times10^{-4}$ |
| 位置编码 | RoPE，使用原始位置（非打乱后） |
| 评估方式 | Zero-shot accuracy / Perplexity |
| 推理实验 | 修改 attention mask，不更新权重；prefill 阶段使用 SA/SWA/MoBA，decode 使用 full KV cache |

---

### 基线方法对比
- **Full Attention**：标准因果注意力，理论最优但成本高
- **SWA**：滑动窗口注意力，高效但受限于局部性
- **MoBA (Mixture of Block Attention)**：基于相关性的块级路由，当前先进稀疏 attention 方法
- **Random Permutation + SWA (SA)**：本文提出的方法
- **Gated SA + SWA**：本文提出的融合结构

---

## 3. 主要实验结果和性能指标

### （1）预训练语言模型结果（Table 1）
| Model | Wiki.ppl↓ | LAMBADA ppl↓ | Avg Acc↑ |
|-------|-----------|--------------|----------|
| Full Attention | 51.34 | 185.3 | 34.9 |
| SWA | 57.05 | 156.1 | 35.1 |
| SA | 75.83 | 260.1 | 34.3 |
| **SA+SWA** | **51.98** | **131.7** | **35.9** |

> 🔍 **关键发现**：
- 单独使用 SA 导致 **ppl 显著上升**（75.83 vs 57.05），说明破坏了局部连续性不利于语言建模。
- 但下游任务表现尚可（avg 34.3），表明 SA 捕获了**互补的全局语义信息**。
- **SA+SWA 融合模型取得最佳综合性能**：ppl 接近 full attention，LAMBADA 准确率最高（acc 22.8/17.6），平均准确率达 **35.9**，显著优于其他变体。

---

### （2）训练后推理实验（Qwen3 系列）

#### 总体趋势（Figure 4）
- **Stochastic Attention 最快逼近 full attention 基线**
- 在较小有效窗口下（如 $w_{eff}=64$）：
  - Qwen3-30B 上，SA 达到 **73.2%** 平均精度
  - SWA 仅为 **47.0%**
  - MoBA 为 **66.3%**
- SA 在 **MMLU、BoolQ、LAMBADA** 等需跨上下文整合的任务上优势明显

#### 小窗口鲁棒性（$w=32$）
| Model | MMLU (8B) | MMLU (30B) |
|-------|-----------|------------|
| SWA | 29.0 | 34.9 |
| **Stochastic (ours)** | **44.4** | **52.0** |

> ✅ 表明即使窗口极小，SA 仍能维持有效全局信息流动。

---

### （3）消融实验与分析
- **门控机制有效性**：SA+SWA 中的 gate 学会平衡局部（SWA）与全局（SA）路径，在不同任务和输入中自适应调整权重。
- **理论覆盖深度验证**：
  - SA 实现 $O(\log_w n)$ 层内全序列覆盖
  - 当 $n=130K, w=256$ 时，预测约 4 层即可完成全连接，与果蝇 connectome 的平均路径长度 **4.4 hop** 高度吻合
- **偏差-方差分解**：
  - SWA 引入系统性偏差（无法看到远距离 token）
  - SA 引入随机方差（每次采样不同）
  - 门控机制可学习平衡二者

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **神经科学启发有效**：果蝇 connectome 的“小世界网络”特性（dense local clusters + sparse long-range shortcuts）可被迁移到 attention 设计中。
2. ✅ **随机重排是强大而廉价的全局混合机制**：仅通过随机 permutation 即可在 $O(nw)$ 时间内实现指数级感受野扩展。
3. ✅ **SA + SWA 实现“小世界 regime”**：
   - SWA 提供高聚类系数（local coherence）
   - SA 提供短路径长度（global shortcuts）
   - 两者结合达到类似 Watts-Strogatz 模型的小世界状态
4. ✅ **即插即用性强**：SA 不改变 attention 内核，仅需前后添加 `gather/scatter` 操作，兼容 FlexAttention 等现代框架。
5. ✅ **可用于训练后增强**：对已训练好的 full attention 模型（如 Qwen3），可在 inference 时用 SA 替代 SWA，以极低成本恢复大部分性能。

---

### 方法的局限性
1. ❗ **纯 SA 破坏局部结构**：单独使用会导致语言建模 perplexity 上升，不适合单一路由。
2. ❗ **依赖足够深度**：虽然理论覆盖快，但在浅层模型中可能未充分展开。
3. ❗ **硬件优化依赖底层支持**：虽然操作简单，但频繁的 index shuffle 可能影响内存访问效率，需良好 kernel 实现（如 FlexAttention）才能发挥优势。
4. ❗ **理论假设简化**：分析中使用 circular window 和 uniform permutation，实际中 causal masking 会略微降低连接概率。

---

### 未来工作方向
1. 🔄 探索更智能的 permutation 策略（如 data-adaptive shuffle）
2. ⚙️ 将 SA 思想推广至 **linear attention**、**state space models (SSMs)** 等新型 attention 架构
3. 🧠 进一步挖掘 connectome-inspired 架构设计原则（rich-club, reciprocity, motifs）
4. 💡 结合 routing-based 方法（如 MoE）构建更高效的混合系统
5. 📈 在超长序列（$n > 100K$）场景下验证 SA 的 scalability

---

> 🧠 **最终洞见**：  
> “Global information flow need not rely on dense all-to-all connectivity, but can emerge from the interplay of structured local computation and sparse long-range shortcuts accumulated through depth.”  
> —— 这正是果蝇大脑告诉我们的智慧。

</details>

---

### 3. [Universal YOCO for Efficient Depth Scaling](https://arxiv.org/abs/2604.01220)

**Authors**: Yutao Sun, Li Dong, Tianzhu Ye, Shaohan Huang, Jianyong Wang, Furu Wei  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.01220v1  

#### Abstract
The rise of test-time scaling has remarkably boosted the reasoning and agentic proficiency of Large Language Models (LLMs). Yet, standard Transformers struggle to scale inference-time compute efficiently, as conventional looping strategies suffer from high computational overhead and a KV cache that ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Universal YOCO for Efficient Depth Scaling**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
当前主流的 **Transformer** 架构在进行 **test-time scaling**（如推理时循环、深度扩展）以增强模型推理能力时，面临两个关键瓶颈：
- **高计算开销**：标准循环机制（如 Universal Transformer）需要重复执行所有层，导致计算复杂度随深度线性增长。
- **KV Cache 膨胀**：每轮循环都会生成新的 KV 缓存，内存占用随迭代次数 $T$ 和层数 $L$ 成倍增加，严重限制长上下文和高效推理。

这些问题阻碍了 LLM 在推理阶段有效扩展计算资源，尤其是在需要深度思考（deep reasoning）的任务中。

---

### 🚀 **提出了什么新方法或新思路**
本文提出 **Universal YOCO (YOCO-U)**，一种结合 **YOCO 架构** 与 **递归计算（recursion）** 的新型高效深度扩展架构。其核心思想是：

- **继承 YOCO 的“你只缓存一次”（You Only Cache Once）理念**：
  - 将模型分为 **Self-Decoder** 和 **Cross-Decoder** 两部分。
  - Self-Decoder 使用 **Efficient Self-Attention（如滑动窗口 SWA）** 处理输入，生成一个紧凑的全局 KV Cache。
  - Cross-Decoder 所有层共享该 KV Cache，避免逐层存储，实现 **常量级 KV Cache 开销**。

- **引入“通用自解码器”（Universal Self-Decoder）**：
  - 在 Self-Decoder 模块上应用 **参数共享的递归计算**，通过多次迭代提升表征能力，而不增加参数量。
  - 仅对浅层、高效的注意力模块进行递归，而非整个网络。

> 🔑 **关键创新**：将递归限制在 **高效注意力模块** 中，实现了“**Think Deeper, Cache Once**”——既能深度思考，又不增加缓存负担。

---

### ⚖️ **相比现有方法的优势**
| 方法 | 优势 |
|------|------|
| **vs. Standard Transformer** | 避免全层重复计算，KV Cache 不随深度增长 |
| **vs. Universal Transformer (UT)** | 仅递归浅层模块，避免冗余计算和优化困难；保持线性预填充（prefilling）效率 |
| **vs. RINS / Early-Layer Recursion** | 使用高效注意力（如 SWA），进一步降低递归开销；KV Cache 更小 |
| **vs. Parallel Scaling (e.g., ParScale)** | 提升的是建模深度而非宽度，更利于复杂推理任务 |

> ✅ YOCO-U 实现了 **能力-效率的更好权衡**：用额外计算换取更强表达力，同时维持低延迟和低内存占用。

---

## 2. **核心实验方法和设置**

### 📚 **使用了哪些数据集**
- **预训练语料**：大规模文本语料（未具体命名，类似 The Pile 或 Common Crawl）
- **下游评测基准**：
  - **通用语言理解**：ARC-C/E, Winogrande, HellaSwag, MMLU, BBH
  - **数学推理**：GSM8K, MATH, SVAMP, ASDiv, MAWPS, CARP, TABMWP, Gaokao, OlympiadBench, CollegeMath, AMC23（共11个）
  - **长上下文建模**：
    - Book 和 Code 数据上的 **困惑度（perplexity）**
    - **Needle In A Haystack (NIAH)** 测试信息检索能力
  - **消融实验**：WikiText-103, LAMBADA, PIQA, OBQA 等

---

### ⚙️ **实验设置和评估指标**

#### 模型配置
- 主要模型大小：**10B 总参数，激活 1.3B 参数**（MoE 结构）
- 层数：20 层（Self-Decoder 和 Cross-Decoder 各 10 层）
- 隐藏维度：2560
- Self-Decoder 使用 **Sliding Window Attention (SWA)**，窗口大小 512
- 位置编码：RoPE（Self-Decoder）、NoPE（Cross-Decoder）
- 训练长度：8192，batch size 4M tokens
- 递归次数 $T=3$ → 总 FLOPs 约为非递归版本的 2×
- 训练步数：75k 步（约 300B tokens）

#### 评估指标
- **语言建模损失（Loss）**
- **准确率（Accuracy）**
- **困惑度（Perplexity）**
- **推理效率**：
  - Prefilling Throughput（tokens/sec）
  - Decoding Throughput（tokens/sec）
  - KV Cache 内存占用（MB）

#### 基线方法对比
| 类型 | 对比模型 |
|------|--------|
| **非递归** | Transformer, YOCO |
| **递归式深度扩展** | Universal Transformer, RINS |
| **并行扩展** | ParScale |

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据**

#### （1）语言建模性能（图2 & 表2）
- 在相同 FLOPs 下，YOCO-U 比 YOCO **降低语言建模损失 △L=0.033**
- 在相同训练 token 数下，YOCO-U **只需 80B tokens 即达到 YOCO 用 210B tokens 的性能** → **节省 ~62% 数据量**

| Model | Average Score |
|-------|---------------|
| YOCO | 41.78 |
| YOCO-U | **46.23** (+4.45) |
| YOCO-U (equal FLOPs) | **47.08** |

> 💡 即使控制 FLOPs，YOCO-U 仍显著优于基线，说明收益不仅来自更多计算，而是架构有效性。

---

#### （2）数学推理能力（图3）
- 在 **11 个数学基准** 上全面超越 YOCO
- 平均准确率提升 **+24.4%**
- 显示递归计算增强了隐式推理能力，且与显式思维链（CoT）正交

---

#### （3）与其他架构对比（表3 & 图4）
| Model | Avg Acc ↑ | KV Cache ↓ | Prefill Speed ↑ |
|-------|-----------|------------|------------------|
| Transformer | 47.1 | 10240 MB (@256K) | 7.5k t/s |
| YOCO | 47.0 | **522 MB** | **220k t/s** |
| RINS | 48.3 | 20480 MB | 3.7k t/s |
| **YOCO-U** | **48.3** | **542 MB** | **76k t/s** |

> ✅ YOCO-U 达到与 RINS 相当的性能，但 **KV Cache 仅为 RINS 的 1/38**，**prefill 吞吐快 20 倍以上**

---

#### （4）长上下文建模（图4）
- 在 Book 和 Code 数据上，随着上下文增长，YOCO-U 困惑度持续下降，表现优于 Transformer 和 YOCO
- 在 **NIAH 测试** 中（表4）：
  - YOCO-U 在单针（S-NIAH-1）达到 **1.00 准确率**
  - 双针（S-NIAH-2）达 **0.95**，优于 RINS（0.91）和 Transformer（0.82）
  > 表明其具备强大的长程依赖捕捉和信息检索能力

---

### 🔍 **消融实验结果（表5）**

| 设计选择 | 结果分析 |
|--------|---------|
| **Deep instead of Wide** | 加深模型但不递归 → 性能无明显提升 |
| **Upper Loop (Cross-Decoder 递归)** | 性能下降 → 说明不应在高层递归 |
| **Upper Loop w/o Shared KV** | 性能更低 → 验证了共享 KV 的必要性 |
| **Deeper layout + YOCO-U** | 仍能受益于递归 → 架构可扩展性强 |

> ✅ 验证了“**在浅层 Self-Decoder 上递归 + 共享 KV**”是最优设计

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **递归计算与高效注意力可以协同增效**：
   - 将递归应用于 **浅层、高效注意力模块**，可在几乎不增加 KV Cache 的前提下显著提升模型能力。
2. **YOCO-U 实现了“能力-效率”的帕累托前沿突破**：
   - 在相同 FLOPs 下性能更强
   - 在相同性能下参数更少（图5：50% 更少参数即可媲美 YOCO）
   - 支持线性预填充、低内存占用，适合长上下文部署
3. **递归带来的收益是结构性的，而非单纯计算堆叠**：
   - 消融实验证明，位置和机制设计至关重要
4. **与 test-time scaling 正交**：
   - YOCO-U 提升的是训练阶段的内在推理能力，可与推理时 CoT、Tree-of-Thought 等方法结合

---

### ⚠️ **方法的局限性**
- 当前递归次数有限（通常 $T=3$），过多次迭代可能出现表征收敛（见图8：angular distance 趋于稳定）
- 依赖 YOCO 架构的前提假设（如 Self/Cross 分离），可能不适用于所有任务
- 实验集中在自回归语言模型，是否适用于 encoder-decoder 架构尚待验证

---

### 🔮 **未来工作方向**
- 探索 **动态递归次数**（adaptive depth），根据输入复杂度决定迭代步数
- 将 YOCO-U 应用于 **多模态模型** 和 **agent 系统**
- 结合 **test-time scaling** 进一步释放推理潜力
- 扩展至更大规模模型（>100B）验证可扩展性

---

## ✅ **总结一句话**
> **YOCO-U 通过在 YOCO 架构的浅层模块中引入递归计算，实现了“深度思考、一次缓存”，在几乎不增加推理成本的前提下显著提升了 LLM 的建模能力和推理效率，为 scalable LLM 设计提供了新范式。**

</details>

---

### 4. [Apriel-Reasoner: RL Post-Training for General-Purpose and Efficient Reasoning](https://arxiv.org/abs/2604.02007)

**Authors**: Rafael Pardinas, Ehsan Kamalloo, David Vazquez, Alexandre Drouin  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2604.02007v1  

#### Abstract
Building general-purpose reasoning models using reinforcement learning with verifiable rewards (RLVR) across diverse domains has been widely adopted by frontier open-weight models. However, their training recipes and domain mixtures are often not disclosed. Joint optimization across domains poses si...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Apriel-Reasoner: RL Post-Training for General-Purpose and Efficient Reasoning》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前基于 **Reinforcement Learning with Verifiable Rewards (RLVR)** 的多领域推理模型训练面临两大挑战：
1. **多域异构性导致训练分布偏移**：不同任务（如数学、代码）在 rollout 长度、验证延迟和样本效率上差异显著，导致简单或快速完成的领域在训练中被过度采样，破坏预设的领域混合比例，影响泛化能力。
2. **推理长度与效率的权衡**：RLVR 模型倾向于生成过长的 **chain-of-thought (CoT)** 跟踪，造成高推理成本和延迟，尤其在实际部署中不可接受。

此外，尽管许多前沿开源模型采用多域 RLVR，其训练配方（training recipes）、领域配比等细节往往未公开，限制了可复现性和研究进展。

---

### **提出的新方法与创新点**

#### ✅ **1. 可复现的多域 RL 后训练配方（Reproducible Multi-Domain RL Post-Training）**
- 在 **Apriel-Base**（一个 15B 参数、未经 RL 或偏好优化的开源 LLM）基础上，使用 **五个公开数据集** 进行联合多域 RLVR 训练。
- 完整公开训练配置、超参数、领域配比和系统设计，推动多域 RL 研究的透明化与可复现性。

#### ✅ **2. 自适应领域采样机制（Adaptive Domain Sampling）**
- 动态调整各领域的采样权重，以补偿 rollout 速度和成功率的不均衡。
- 通过监控已完成的 rollout 数量，实时计算调整因子，确保训练过程中维持目标领域比例。
- **优势**：无需手动重采样或静态划分，适用于异步、动态的多域训练环境。

#### ✅ **3. 难度感知长度惩罚（Difficulty-Aware Length Penalty, DAP）**
- 对标准长度惩罚（Length Penalty）进行扩展，使其强度随问题难度自适应变化。
- 利用同一 prompt 下多个 rollout 的 **solve rate**（正确率）估计难度：低 solve rate 表示难题，此时降低惩罚强度，允许更长推理；反之则鼓励简洁。
- **优势**：
  - 无额外训练开销（不修改 policy loss，无需辅助模型）。
  - 实现“难题深思，易题速答”的智能资源分配。

---

### **相比现有方法的优势**
| 维度 | 现有方法 | Apriel-Reasoner |
|------|--------|----------------|
| **领域平衡** | 静态采样或逐阶段训练（sequential），易忽略跨域交互 | 动态自适应采样，保持目标混合比例 |
| **长度控制** | 固定预算或复杂机制（如预算预测、路由） | 简单、高效、无额外模块的 DAP |
| **可复现性** | 多数模型仅开放权重，训练细节缺失 | 全流程公开：数据、代码、配置、超参 |
| **效率-精度权衡** | 往往牺牲精度换短输出，或反之 | 显著提升精度的同时减少 30–50% 输出 token |

---

## **2. 核心实验方法和设置**

### **使用的数据集（Training Domains）**
| 领域 | 数据集 | 规模 | 奖励机制 |
|------|-------|------|---------|
| **Mathematics** | Open-Reasoner-Zero | ~129K | 最终答案 exact match |
| **Code Generation** | TACO | ~24K | 沙箱执行通过所有测试用例 |
| **Instruction-Following** | IF-RLVR | ~95K | 满足约束的比例（0–1） |
| **Logical Puzzles** | INTELLECT-3/SynLogic | ~12K | 任务特定程序化验证器 |
| **Function Calling** | BFCL v4（单轮） | ~4K | 函数名精确匹配 + 参数合法 |

---

### **实验设置**
- **基础模型**：`Apriel-1.5-15b-Thinker`（15B 参数，仅解码器，无视觉编码器）
- **训练框架**：基于 **PipelineRL** 实现异步 on-policy RL，支持 in-flight 权重更新
- **优化算法**：**GSPO**（Group Sequence Policy Optimization），避免 token-level 高方差梯度
- **输出限制**：训练时最大输出长度为 **16K tokens**，推理时扩展至 **32K tokens**
- **领域配比**：数学 40%，代码 25%，逻辑谜题 15%，指令遵循 10%，函数调用 10%

---

### **评估指标**
- **主指标**：Pass@1 准确率（四个基准平均）
- **效率指标**：平均每条输出的 **token 数量**
- **评估配置**：
  - 温度 0.6，top-p 0.95
  - 所有模型统一在 **32K token 输出上限**下评估
  - 多次运行取平均以降低方差

---

### **基线方法对比**
| 模型 | 类型 | 参数量 | 特点 |
|------|------|--------|------|
| **Apriel-Base** | 基础模型 | 15B | 未经 RL 训练 |
| **Phi-4-reasoning** | RL 微调 | 14B | 基于 o3-mini 推理轨迹微调 |
| **Qwen3-14B** | 多阶段后训练 | 14B | SFT + RL，通用与推理结合 |
| **Nemotron-Cascade-14B** | 逐域 RL | 14B | 分阶段训练，强调级联推理 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 2）**
| 模型 | AIME-25 Acc | Tok | GPQA Acc | Tok | MMLU-Pro Acc | Tok | LCB v5 Acc | Tok |
|------|------------|-----|----------|-----|--------------|-----|------------|-----|
| **Phi-4-reasoning** | 57.3 | 12.5k | 64.8 | 3.5k | 77.1 | 3.4k | 56.9 | 14.4k |
| **Qwen3-14B** | 68.0 | 16.9k | 64.3 | 6.7k | 77.7 | 2.4k | 65.9 | 12.2k |
| **Nemotron-Cascade** | 76.0 | 19.0k | 68.4 | 10.6k | 76.8 | 3.6k | 70.7 | 16.0k |
| **Apriel-Base** | 73.3 | 16.6k | 68.8 | 10.5k | 76.4 | 3.5k | 61.4 | 14.9k |
| **Apriel-Reasoner (Ours)** | **78.3** | **11.3k** | **69.8** | **5.8k** | **77.3** | **1.9k** | **70.8** | **7.4k** |

> 📌 **结论**：Apriel-Reasoner 在所有四项任务上均达到最高准确率，同时输出 token 数减少 **30–50%**，显著优于同类模型。

---

### **与基线的对比亮点**
- **AIME-25**：准确率 **78.3%**（SOTA），比 Nemotron-Cascade 高 2.3%，但输出少 **41%**（11.3k vs 19.0k）
- **GPQA**：首次突破 **69.8%**，且仅用约一半 token（5.8k vs 10.6k）
- **MMLU-Pro**：准确率 **77.3%**，输出仅 **1.9k tokens**，为最高效模型（第二名为 Qwen3 的 2.4k）
- **LiveCodeBench**：准确率 **70.8%**，接近 Nemotron-Cascade（70.7%），但输出 token 不到其一半（7.4k vs 16.0k）

---

### **消融实验结果（Ablation Studies）**

#### 🔹 **DAP vs 标准长度惩罚**
| 设置 | AIME-25 Acc | Tok | LCB Acc | Tok |
|------|------------|-----|--------|-----|
| 标准 LP | 71.7 | 11.1k | 67.7 | 7.0k |
| **DAP (Ours)** | **78.3** | **11.3k** | **70.8** | **7.4k** |

> ✅ DAP 在几乎不增加 token 开销的情况下，带来 **+6.6% AIME** 和 **+3.1% LCB** 提升，证明其有效利用 token 资源。

#### 🔹 **领域配比消融（Table 3）**
| 配比策略 | AIME-25 Acc | LCB Acc | 结论 |
|--------|------------|--------|------|
| 均匀配比（Uniform） | 75.0 | 70.5 | 全面落后 |
| 仅数学+代码 | 75.7 | 67.2 | 忽视其他领域损害泛化 |
| **本文配比（40/25/15/10/10）** | **78.3** | **70.8** | 最优组合 |

> ✅ 所有五个领域均有贡献，特别是逻辑谜题和指令遵循对综合性能至关重要。

---

## **4. 关键结论和发现**

### **主要发现**
1. **多域联合 RLVR 可显著提升通用推理能力**，且可通过 **自适应采样** 有效缓解异构 rollout 带来的分布偏移。
2. **DAP 是一种轻量高效的长度控制机制**，能实现“按需思考”，在不增加训练复杂性的前提下优化 token 使用效率。
3. **Apriel-Reasoner 展现出强大的长度泛化能力**：尽管训练时限制为 16K tokens，但在 32K tokens 下仍表现优异。
4. **效率提升源于表达更紧凑而非推理变浅**：
   - 分析显示，Apriel-Reasoner 的推理步数与 Base 模型相近；
   - 但每步 token 数减少约 **35%**，且非生产性步骤从 21% 降至 14%；
   - 更多使用 **verification、backtracking、subgoal setting** 等高级认知行为。

---

### **方法的局限性**
- **依赖高质量可验证奖励**：仅适用于 reward 可编程验证的任务（如数学、代码），难以扩展到主观性强的生成任务。
- **DAP 依赖 group-level solve rate**：需要每个 prompt 生成多个 rollout 才能估计难度，在低资源场景可能受限。
- **未探索更大规模模型**：目前仅在 15B 模型上验证，是否可扩展至百亿以上参数尚待研究。

---

### **未来工作方向**
- 将 DAP 思想推广至 **test-time budget allocation**，实现动态推理深度控制。
- 探索 **跨领域迁移机制**，进一步增强小众领域（如逻辑谜题）的学习效率。
- 构建 **统一的多模态 RLVR 框架**，整合文本、代码、图像等多模态推理任务。
- 推动 **RLVR 训练标准化与开源生态建设**，促进社区协作与公平比较。

---

> ✅ **总结一句话**：  
> **Apriel-Reasoner 通过可复现的多域 RL 配方、自适应采样与难度感知长度惩罚，在提升通用推理能力的同时大幅降低推理成本，推动了 accuracy-vs-token-efficiency 的帕累托前沿。**

</details>

---

### 5. [Batched Contextual Reinforcement: A Task-Scaling Law for Efficient Reasoning](https://arxiv.org/abs/2604.02322)

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

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
大型语言模型（LLMs）在采用 Chain-of-Thought（CoT）推理时表现出强大性能，但存在严重的**推理效率低下**问题。模型倾向于生成冗长、重复的推理链，导致 token 消耗过高，显著增加推理成本。尽管已有多种效率优化方法（如显式长度惩罚、难度估计器、多阶段课程学习），但这些方法通常会损害推理质量、引入复杂训练流程或需要精细调参。

本文旨在解决一个根本性问题：**能否让 LLM 在不依赖任何显式长度监督的情况下，自主学会高效推理？**

### 提出了什么新方法或新思路
作者提出了 **Batched Contextual Reinforcement (BCR)**，一种极简的单阶段训练范式，其核心思想是通过**结构性修改输入格式**来隐式诱导高效推理。

- **方法核心**：将多个数学问题打包成一组（batch），共享一个固定的上下文窗口（context window），并要求模型在一次生成中连续解决所有问题。
- **奖励机制**：仅基于每个问题的解答正确性（per-instance accuracy）进行奖励，**不引入任何显式的长度惩罚项、辅助模型或难度标签**。
- **隐式约束**：固定 token 预算创建了一个“信息瓶颈”（information bottleneck）。模型必须在有限的 token 内解决所有问题，从而被迫自主学习如何分配推理深度、压缩冗余思考，并优先提高信息密度。

### 相比现有方法的优势
| 特性 | 现有方法（如 ShorterBetter, ARM, ProRL） | BCR |
| :--- | :--- | :--- |
| **训练复杂度** | 多阶段、需额外模型或复杂调度 | 单阶段、无需额外组件 |
| **监督信号** | 显式长度惩罚、难度标签 | 仅准确率奖励，无显式效率信号 |
| **稳定性** | 易因对抗梯度导致训练崩溃 | 优化稳定，避免了对抗梯度问题 |
| **通用性** | 通常针对特定任务设计 | 结构简单，正交于其他技术，易于组合 |

BCR 的优势在于其**极简主义**（minimalist）和**涌现性**（emergent）——高效推理能力并非被直接教导，而是在资源竞争的环境下自然涌现的。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **训练数据**：`DeepMath-103K`，一个大规模、去污染的数学数据集。
- **评估数据集**：涵盖不同难度级别的五个主流数学基准：
  - `AIME25`（美国数学邀请赛）
  - `AMC23`（美国数学竞赛）
  - `Minerva Math`
  - `MATH-500`
  - `Olympiad`（奥林匹克级别双语科学问题）

### 实验设置和评估指标
- **模型规模**：在两个不同规模的模型上验证，`JustRL-DeepSeek-1.5B` 和 `Qwen3-4B-Thinking-2507`。
- **训练配置**：
  - 将训练数据按难度分层采样，每组包含 `N=3` 个问题。
  - 设置全局 token 预算（如 5120 tokens for 1.5B model）。
  - 使用 `GRPO`（Group Relative Policy Optimization）进行训练。
- **评估指标**：
  - **准确性（Accuracy）**：解答正确的百分比。
  - **平均 token 数（Avg. Tokens per problem）**：衡量推理效率的关键指标。
  - 评估模式包括标准单问题推理（N=1）和多问题并发推理（N>1）。

### 基线方法对比
与以下几类基线方法进行了全面比较：
- **通用 LLM**：`Qwen3-1.7B`
- **数学专用 LLM**：`STILL-3-1.5B`
- **带长度控制的 LLM**：`BroRL-1.5B`, `e3-1.7B`
- **自适应推理模型**：`ARM-3B`, `Thinker-Q1.5B`

---

## 3. 主要实验结果和性能指标

### 关键性能数据
在标准单问题推理（N=1）下，BCR 取得了显著的“免费午餐”（free lunch）现象：

#### 对比 `JustRL-1.5B` 基线：
| 数据集 | 准确率变化 | Token 减少 |
| :--- | :--- | :--- |
| AIME25 | +3.3% | -62.6% |
| AMC23 | +2.5% | -53.8% |
| Minerva | +5.1% | -53.9% |
| MATH-500 | -3.8% | -39.8% |
| Olympiad | +0.8% | -58.0% |

#### 对比 `Qwen3-4B` 基线：
| 数据集 | 准确率变化 | Token 减少 |
| :--- | :--- | :--- |
| AIME25 | +13.3% | -15.8% |
| AMC23 | +2.5% | -31.8% |
| Minerva | +2.2% | -27.1% |
| MATH-500 | +0.4% | -27.7% |
| Olympiad | +1.9% | -18.0% |

> **关键发现**：BCR 在几乎所有基准上都实现了**准确率持平甚至提升的同时，大幅降低 token 消耗**（最高达 62.6%）。

### 与基线方法的对比结果
- **优于长度控制方法**：BCR 的 token 效率与 `BroRL` 等多阶段方法相当或更优，但实现方式更简单、更稳定。
- **远超自适应推理模型**：`ARM` 和 `Thinker` 虽然极度精简，但准确率暴跌（如 `ARM` 在 AIME25 上仅 3.3%），而 BCR 在保持高准确率的前提下实现高效。
- **“任务扩展定律”（Task-Scaling Law）**：在多问题并发推理（Nx）下，随着 `N` 增加，BCR 模型的每题 token 消耗单调下降，且准确率下降非常平缓。例如，在 `N=4` 时，BCR 在 `AIME25` 上比基线少用 75% 的 token 且准确率更高。

### 消融实验结果
- **训练组大小（N）消融**：在 `N=3,4,5` 下训练，均能获得稳定的效率增益，其中 `N=3` 在多数基准上表现最佳。
- **隐式 vs. 显式长度控制**：显式长度惩罚（如 `r_len = -|y|/maxlen`）会导致灾难性的训练崩溃（catastrophic training collapse），准确率迅速降至零。而 BCR 的隐式预算约束则保证了优化过程的稳定性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **任务扩展定律（Task-Scaling Law）**：并发处理的问题数 `N` 是一个新的推理效率扩展维度。增大 `N` 可系统性地提高推理密度。
2. **“免费午餐”现象**：BCR 挑战了传统的准确率-效率权衡假设，证明在不牺牲甚至提升准确率的前提下，可以实现大幅效率提升。
3. **涌现的自我调节效率**：模型自发学会了消除冗余的元认知循环（如“让我再检查一下”）、直接选择最优策略、防止灾难性退化，这表明 LLMs 具备潜在的高密度推理模式。
4. **约束优于惩罚**：隐式预算约束在优化稳定性上远超显式长度惩罚，为长度控制提供了更优越的范式。
5. **核心论断**：LLMs 本身具备高效推理的能力，传统单问题训练未能激活这种潜能。BCR 通过多问题资源竞争的结构化环境，成功解锁了这一潜能。

### 方法的局限性
- 当前研究集中在**数学推理**领域，尚未在代码生成、科学推理或多模态任务上验证。
- 已在 1.5B 和 4B 规模模型上验证，是否可扩展到更大模型（如 7B-70B）有待探索。
- 未结合其他效率技术（如投机解码）进行进一步优化。

### 未来工作方向
- 将 BCR 扩展到更大的模型（7B-70B）和其他推理领域（代码、科学、多模态）。
- 探索任务扩展定律的理论基础，以及其在异构任务混合中的适用性。
- 结合 BCR 与其他效率技术（如自适应 token 分配、推测解码）以追求进一步的性能突破。
- 研究环境结构如何替代显式监督，以激发模型的其他潜在能力。

</details>

---

### 6. [DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72](https://arxiv.org/abs/2604.01621)

**Authors**: Wanqian Li, Jintao Peng, Zongfei Jing, Tianyu Zhang, Ze Long, Xianjie Qiao, Xiaoming Chen, Dongxu Yang, Kefeng Duan, June Yang  
**Category**: cs.DC  
**Published**: 2026-04-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.01621v1  

#### Abstract
Large language model (LLM) inference increasingly depends on multi-GPU execution, yet existing inference parallelization strategies require layer-wise inter-rank synchronization, making end-to-end performance sensitive to workload imbalance. We present DWDP (Distributed Weight Data Parallelism), an ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

在当前的 **Large Language Model (LLM)** 推理系统中，主流的并行化策略（如 **expert parallelism**, **tensor parallelism**, **pipeline parallelism**）依赖于层间（layer-wise）的跨 GPU 同步通信（例如 all-to-all、all-gather），这导致：

- **同步开销显著**：由于请求长度（ISL）、KV-cache 命中率、专家路由不均等原因，各 rank 的计算负载天然不平衡；
- **最慢 rank 决定整体吞吐**：即使部分 GPU 已完成计算，仍需等待其他 rank 完成通信才能进入下一层，造成“长尾延迟”；
- **端到端吞吐受限**：尤其在高并发、长上下文场景下，这种同步机制成为性能瓶颈。

### 🚀 提出了什么新方法或新思路

提出 **DWDP (Distributed Weight Data Parallelism)** ——一种新型推理并行策略，其核心思想是：

- **保持 data-parallel 执行模式**：每个 rank 独立接收请求、独立执行推理；
- **将 MoE 权重分布式地 offload 到 peer GPUs 上**：仅保留本地专家（Local Experts），远程专家（Remote Experts）按需异步预取；
- **完全移除集体通信（collective communication）**：避免 NCCL-based all-to-all 等同步操作；
- **通过 cudaMemcpyAsync 实现点对点异步拉取（peer-to-peer pull）**：利用 copy engine 而非 SM，避免抢占计算资源；
- **通信与计算重叠（overlap）**：在当前层 MoE 和 attention 计算期间，异步预取下一层所需的远程专家权重。

> 💡 **本质创新**：将“模型并行中的权重分布”从“静态划分 + 集体通信”转变为“动态按需加载 + 异步拉取”，实现 **fully asynchronous inference**。

### 🔍 相比现有方法的优势

| 维度 | DEP / EP 等传统方法 | DWDP |
|------|---------------------|-------|
| **同步机制** | 层间同步（barrier-like） | 完全异步，无全局同步 |
| **负载均衡敏感性** | 极高（受 ISL、routing skew 影响大） | 显著降低（各 rank 可独立推进） |
| **通信开销位置** | 在关键路径上（critical path） | 可被计算窗口隐藏 |
| **资源分配灵活性** | 要求专家数能整除 group size | 支持任意 group size，支持冗余放置 |
| **部署粒度** | 粗粒度（必须满足并行约束） | 更细粒度，便于 disaggregated serving |

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

- **Artificial Analysis Dataset**：用于 context-only 性能分析；
- **SemiAnalysis Dataset**：用于 end-to-end 推理评估，模拟真实服务负载；
- 输入序列长度范围：**6.4K ~ 8K tokens**（input ratio = 0.8）；
- 输出序列长度：**1K tokens**。

### ⚙️ 实验设置

| 项目 | 设置 |
|------|------|
| **硬件平台** | **GB200 NVL72**（Blackwell 架构，高带宽 NVLink） |
| **软件框架** | **TensorRT-LLM**（基于 commit `3a89495` 实现 DWDP） |
| **模型** | **DeepSeek-R1**（MoE 模型，NVFP4 量化权重，FP8 KV cache） |
| **运行模式** | **Disaggregated Serving**（context server 使用 DWDP，decoding server 不变） |
| **DWDP Group Size** | 默认为 4（即每组 4 个 GPU 共享完整 MoE 权重） |

### 📈 评估指标

| 指标 | 描述 |
|------|------|
| **TPS/GPU** | Tokens Per Second per GPU，衡量硬件利用率 |
| **TPS/user** | Tokens Per Second per user，衡量服务质量 |
| **TTFT (Time to First Token)** | 中位响应延迟（含排队时间） |
| **End-to-End Throughput** | 整体输出吞吐能力 |
| **Pareto Frontier** | 分析 TPS/user 与 TPS/GPU 的权衡关系 |

### 🔁 基线方法对比

- **主基线（Baseline）**：**DEP (Data parallelism with Expert Parallelism)**  
  即传统的 attention 数据并行 + MoE 专家并行组合方案；
- 所有实验在同一硬件和运行时条件下进行，仅修改 context server 的并行策略。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ Context-Only 实验结果（表3）

| ISL (tokens) | TPS/GPU Speedup | TTFT Speedup |
|-------------|------------------|--------------|
| 1K          | 1.11×            | 1.27×        |
| 8K          | 1.10×            | 1.16×        |
| 16K         | 1.09×            | 1.12×        |
| 32K         | 1.09×            | 1.11×        |

> - 在长上下文（>8K）下仍保持约 **10% 的 TPS/GPU 提升**；
> - **TTFT 改善更明显**，说明首次响应更快。

#### ✅ 不同 MNT（Max Number of Tokens）的影响

| MNT (tokens) | TPS/GPU Speedup | TTFT Speedup |
|-------------|------------------|--------------|
| 16K         | 1.01×            | 1.07×        |
| 32K         | 1.10×            | 1.16×        |

> - 更大的 MNT 提供更大的 **compute window**，有助于隐藏 prefetch 开销 → 性能增益更高。

#### ✅ 负载不平衡影响（ISL STD 增加）

| ISL STD (tokens) | TPS/GPU Speedup | TTFT Speedup |
|------------------|------------------|--------------|
| 0                | 1.09×            | 1.12×        |
| 4096             | 1.15×            | 1.18×        |

> - **负载越不均衡，DWDP 优势越明显**，验证其抗 skew 能力。

#### ✅ DWDP Group Size 影响

| Group Size | TPS/GPU Speedup | TTFT Speedup |
|-----------|------------------|--------------|
| 3         | 1.093×           | 0.86×        |
| 4         | 1.091×           | 1.15×        |

> - TPS/GPU 基本持平，表明小规模部署也有效；
> - DWDP3 的 TTFT 更差，可能因总吞吐低导致排队延迟上升。

---

### 🔬 消融实验结果（Ablation Studies）

#### 表4：优化技术带来的提升（ISL=8K）

| ISL Ratio | MNT     | DEP (baseline) | DWDP + Merge Elim. | Full DWDP (w/ contention mitigation) |
|----------|---------|----------------|---------------------|--------------------------------------|
| 0.5      | 16384   | 1.000          | 0.995               | **1.081**                            |
| 0.5      | 32768   | 1.000          | 1.140               | **1.139**                            |
| 0.8      | 16384   | 1.000          | 1.039               | **1.053**                            |
| 0.8      | 32768   | 1.000          | 1.098               | **1.109**                            |

> - **Split-weight merge elimination**：消除 D2D copy 开销，带来 ~3–4% 提升；
> - **Contention mitigation（time-division multiplexing）**：在 compute window 较小时效果显著（如 MNT=16K），防止源端 copy engine 序列化阻塞。

---

### 🏁 End-to-End 实验结果（表5 & 图5）

#### 在不同 TPS/user 区间下的平均加速比

| TPS/user Range | Avg. TPS/user Speedup | Avg. TPS/GPU Speedup |
|----------------|------------------------|------------------------|
| 20–30          | 1.15×                  | **1.10×**              |
| 40–50          | 1.16×                  | 1.08×                  |
| 60–70          | 1.00×                  | 1.10×                  |
| 80–90          | 1.00×                  | 1.06×                  |
| 170–180        | 1.00×                  | 0.97×                  |

> - 在 **20–100 TPS/user** 典型服务区间内，**TPS/GPU 平均提升 8.8%**；
> - 高负载区域（>170 TPS/user）出现轻微退化，因系统已严重 generation-bottlenecked，context 加速无法释放。

#### 关键观察：
- DWDP 点位于原 Pareto frontier **上方**，表示在相同 TPS/user 下实现了更高的 TPS/GPU；
- 多数情况下使用 **更少的 context GPU** 即可达到相同性能 → 提升资源效率；
- 在低 TPS/user 区间，TTFT 有所增加（见表6），归因于 context 阶段服务速率下降引发的排队延迟。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **DWDP 成功消除了层间同步瓶颈**，特别适用于 **负载不均衡的真实生产环境**；
2. 在具备高带宽互联（如 NVLink on GB200）的系统上，**remote weight prefetch 可被有效隐藏**于 compute window 中；
3. **异步执行 + 动态权重加载** 的设计提升了部署灵活性，支持更细粒度的资源调度；
4. 在典型 8K/1K 推理场景下，**端到端 TPS/GPU 提升达 8.8%**，且不牺牲 TPS/user；
5. 两大优化（split-weight merge elimination 和 contention mitigation）对实际性能至关重要，尤其在短 compute window 场景下。

### ⚠️ 方法的局限性

| 限制 | 说明 |
|------|------|
| **依赖高带宽通信** | 若 NVLink 带宽不足（如 PCIe 系统），prefetch 难以隐藏，收益下降甚至为负 |
| **TTFT 可能恶化** | 当 context GPU 数量减少过多时，context stage 成为瓶颈，导致排队延迟上升 |
| **仅适用于 context phase** | 当前设计聚焦于 prefill 阶段；decode 阶段 token 数少，难以摊销 prefetch 开销 |
| **通信-计算干扰仍存在** | 尽管使用 copy engine，但仍共享 NoC、L2、DRAM 和 power budget，可能导致频率降频（DVFS） |

> 🔍 附录分析指出：**power-induced frequency throttling** 是 Attention kernel 变慢的主因，而非内存带宽竞争。

### 🔮 未来工作方向

1. **改进 request scheduling 与 rate matching**：缓解因 context GPU 减少导致的 stage mismatch 问题；
2. **探索 decode 阶段的轻量级 DWDP 变体**：例如缓存热点专家、预测性预取；
3. **进一步优化通信调度策略**：如智能 slice size 自适应、优先级控制；
4. **扩展至非 MoE 模型**：研究通用权重分块与按需加载机制；
5. **软硬协同设计**：定制 NoC 或 memory controller 以更好支持异步 weight fetching。

---

## 总结

> **DWDP 通过“去同步化 + 按需权重加载”的范式转变，在高带宽多 GPU 平台上实现了更高效、更鲁棒的 LLM 推理。它不仅提升了吞吐（TPS/GPU ↑8.8%），还增强了系统对现实负载波动的适应能力，为下一代 disaggregated LLM serving 架构提供了重要技术路径。**

</details>

---

### 7. [Application of parametric Shallow Recurrent Decoder Network to magnetohydrodynamic flows in liquid metal blankets of fusion reactors](https://arxiv.org/abs/2604.02139)

**Authors**: M. Lo Verso, C. Introini, E. Cervi, L. Savoldi, J. N. Kutz, A. Cammi  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.02139v1  

#### Abstract
Magnetohydrodynamic (MHD) phenomena play a pivotal role in the design and operation of nuclear fusion systems, where electrically conducting fluids (such as liquid metals or molten salts employed in reactor blankets) interact with magnetic fields of varying intensity and orientation, influencing the...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

**论文标题**: *Application of parametric Shallow Recurrent Decoder Network to magnetohydrodynamic flows in liquid metal blankets of fusion reactors*

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题
本研究针对核聚变反应堆中液态金属包层（如WCLL）内的 **Magnetohydrodynamic (MHD)** 流动建模难题。这类流动涉及强非线性、多物理场耦合（流体动力学、热传导、电磁场），其高保真数值模拟（Full-Order Models, FOMs）计算成本极高，难以满足实时监测、诊断或控制的需求。

此外，在实际工程场景中，传感器部署受限于高温、辐射和几何约束，通常只能获取稀疏、有限的测量数据（如温度），而无法直接观测速度、压力或磁场等关键变量。

因此，本文旨在解决以下挑战：
- 如何在仅有少量传感器（如3个温度探头）的情况下，实现对完整三维 MHD 状态（温度 $T$、速度 $\mathbf{u}$、压力 $p$）的高精度重建；
- 如何使模型具备泛化能力，能够预测训练集中未见过的磁参数配置（强度、方向、时间演化）；
- 如何实现实时、低计算开销的状态估计，适用于数字孪生和在线监控系统。

---

### ✅ 提出的新方法或新思路
提出并应用了一种名为 **parametric SHallow REcurrent Decoder (SHRED)** 的全数据驱动框架，用于 MHD 状态重建。

该方法结合了：
- **Singular Value Decomposition (SVD)** 进行降维压缩，提取主导空间模态；
- **Long Short-Term Memory (LSTM)** 捕捉时间序列动态依赖关系；
- **Shallow Decoder Network (SDN)** 将低维隐状态映射回高维物理空间；
- 利用 **Takens’ embedding theorem** 支持从延迟观测重构系统动力学。

特别地，这是首次将 SHRED 应用于：
- **三维真实包层几何构型**（围绕冷却管的 PbLi 流动）；
- **参数化 MHD 场景**，涵盖不同磁参数组合；
- **时间变化磁场** 下的动态推断任务。

---

### ✅ 相比现有方法的优势
| 方面 | SHRED 的优势 |
|------|--------------|
| **数据效率** | 仅需 **3个随机分布的温度传感器** 即可实现高质量重建，远少于传统 data assimilation 所需密集测量。 |
| **泛化能力** | 对未见磁参数（强度、方向、时间波形）具有强大外推与内插能力，无需重新训练。 |
| **计算效率** | 训练可在普通笔记本电脑完成（数分钟），在线重建耗时 <1秒，相比 FOM（5–15小时/HPC）提速数千倍。 |
| **鲁棒性** | 对传感器位置不敏感，适合复杂工程环境下的部署。 |
| **多功能性** | 可同时重建多物理场变量（$T$, $\mathbf{u}$, $p$），甚至能反演未知的时间变化磁场 $B(t)$。 |
| **轻量化设计** | 模型参数极少（<1000），增强可解释性和安全性，适用于核级应用。 |

---

## 2. **核心实验方法和设置**

### ✅ 数据集来源与生成方式
- 使用 **OpenFOAM** 中开发的 **magnetoHDFoam** 求解器进行高保真 MHD 数值模拟。
- 几何模型为 **WCLL 包层单元的一部分**：一个三维矩形通道内含中心圆柱形水冷管道，PbLi 在其中流动。
- 物理条件基于 DEMO 级反应堆参数设定，采用真实热物性参数。
- 总共生成三类磁配置的数据集：
  1. **恒定环向磁场**（Constant Toroidal Field）
  2. **环向+极向复合磁场**（Toroidal + Poloidal Fields）
  3. **随时间振荡的环向磁场**（Time-Varying Sinusoidal Field）

每组模拟运行 3 秒，采样间隔 0.025 秒，共 120 时间步。

---

### ✅ 实验设置
- **输入信号**：仅使用 **三个随机布置的温度传感器** 的时间序列作为输入。
- **预处理**：
  - 使用 SVD 压缩原始快照矩阵（rank=5）；
  - 采用 min-max 归一化处理所有变量。
- **模型架构**：
  - LSTM 层：2 层，每层 64 神经元，lag=30；
  - SDN 解码器：2 层（350 → 400 神经元）；
  - 输入维度大幅降低，提升训练效率。
- **训练/验证/测试划分**：
  - 多个参数组合用于训练和验证；
  - 测试案例选择在训练范围之外（如 $B=2.5\,\text{T}$ 超出训练上限 $2.0\,\text{T}$）以检验泛化能力。

---

### ✅ 评估指标
- **相对 $L_2$-误差**（Relative $L^2$-error）定义如下：
  $$
  \epsilon_{\phi}(t) = \frac{\|\phi_{\text{FOM}}(t) - \phi_{\text{SHRED}}(t)\|_2}{\|\phi_{\text{FOM}}(t)\|_2}, \quad \phi \in \{T, \mathbf{u}, p\}
  $$
- **可视化对比**：展示横截面上的场量分布及绝对误差图。
- **参数反演能力评估**：比较 SHRED 推测的 $B(t)$ 与真实时间曲线的一致性。

---

### ✅ 基线方法对比
文中虽未明确列出与其他深度学习模型（如 CNN、Transformer、Autoencoder）的定量对比，但通过以下方式体现优越性：
- 强调其相较于典型 deep learning 模型的 **参数量更小、训练更快、硬件要求更低**；
- 与传统 ROM 或 data assimilation 方法相比，SHRED 不需要显式构建物理方程或协方差矩阵，更具灵活性；
- 引用已有文献表明 SHRED 在其他领域（如裂变堆、自由表面流）已展现优于 POD-Galerkin 和 DMD 的表现。

---

## 3. **主要实验结果和性能指标**

### ✅ 关键性能数据

#### 📌 场量重建误差汇总（典型值）
| 测试场景 | 温度 $T$ 最大误差 | 速度 $\mathbf{u}$ 最大误差 | 压力 $p$ 最大误差 |
|---------|------------------|----------------------------|--------------------|
| 恒定磁场（$B_x=0.75\sim2.5\,\text{T}$） | < 4% | < 9%（初瞬），稳态 ~2% | < 4%，稳态 < 2% |
| 复合磁场（$B_{\text{tot}}=1.66\,\text{T}, \theta=15.7^\circ$） | ~3% | ~4%（初瞬），稳态 < 2% | < 2% |
| 时变磁场（Case A/B/C） | 4–6% | 初瞬可达 18%，后降至 ~2% | 初瞬可达 14%，后降至 ~2% |

> 注：误差随时间快速收敛，且最大值集中在局部区域，整体场形高度一致。

---

### ✅ 与基线方法的对比结果（隐含分析）
尽管没有直接表格对比，但从实验设计可得出：
- **相比传统 ROM 方法（如 POD-Galerkin）**：
  - SHRED 无需假设线性叠加或投影残差，避免了模型偏差；
  - 更容易处理非定常、非周期性激励（如任意 $B(t)$）。
- **相比端到端深度学习模型**：
  - SHRED 在极小数据下即可训练成功（仅数百个快照）；
  - 计算资源需求极低，可在个人电脑上完成训练；
  - 泛化能力强，无需大量覆盖参数空间的训练样本。

---

### ✅ 消融实验结果（文中体现）
虽然未设专门“消融”章节，但通过多场景测试间接验证了各组件作用：
- **SVD 压缩有效性**：使用 rank=5 的低秩表示仍能保持高精度重建，说明有效捕捉了主导模态；
- **传感器数量影响**：延续前作 [32] 结论，3 个传感器已达性能饱和，增加无显著增益；
- **传感器位置无关性**：传感器随机放置且固定，重建误差稳定，证明方法对布点无敏感依赖；
- **时间滞后长度（lag=30）足够捕获动态特征**：即使面对复杂 $B(t)$，也能准确追踪演化趋势。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **SHRED 能够仅凭 3 个温度传感器的时间序列，精确重建三维 MHD 全场状态**（$T$, $\mathbf{u}$, $p$），误差普遍低于 5%，具备工程可用性。
2. **模型具有强大的参数泛化能力**：
   - 成功外推至训练范围外的磁场强度（如 $B=2.5\,\text{T}$）；
   - 准确重建复合方向磁场下的倾斜流动结构；
   - 适应多种频率、相位、幅值的时变磁场。
3. **SHRED 可反演隐藏参数**：首次证明可通过温度测量 **间接推断时间变化的磁场 $B(t)$**，为聚变装置提供潜在的 **无侵入式磁场诊断工具**。
4. **整个流程高度高效**：训练几分钟，单次重建小于1秒，适合嵌入数字孪生系统。

---

### ✅ 方法的局限性
- **依赖高质量仿真数据集**：当前完全基于 FOM 数据训练，若模拟与实际存在偏差（如边界条件、材料属性），可能影响迁移效果。
- **尚未集成实验数据**：目前仍处于数值验证阶段，尚未在真实实验台架（如 KIT 的 DYNASTY）上部署验证。
- **长期预测能力未充分测试**：本文聚焦于状态重建而非长期预报，未来需评估其在长时间积分中的稳定性。
- **全局基底假设限制**：使用统一 SVD 基可能在极端参数跳跃时失效，未来可探索自适应基或局部 ROM 策略。

---

### ✅ 未来工作方向
1. **扩展至完整 WCLL 单元几何**，考虑更多冷却管和结构件的影响；
2. **集成 into closed-loop control systems**，用于实时调节冷却功率或反馈磁控策略；
3. **融合实验数据与迁移学习**，提升模型在真实环境中的适用性；
4. **应用于数字孪生平台**，支持聚变反应堆的运行监控与故障预警；
5. **探索多任务学习框架**，同步实现状态重建、参数识别与异常检测。

---

> 🔚 **总结一句话**：  
> 本研究成功将 **parametric SHRED** 应用于聚变包层 MHD 流动，展示了其在稀疏传感条件下实现 **高精度、强泛化、超实时** 状态重建的巨大潜力，是迈向智能、高效、安全可控核聚变系统的重要一步。

</details>

---

### 8. [Execution-Verified Reinforcement Learning for Optimization Modeling](https://arxiv.org/abs/2604.00442)

**Authors**: Runda Guan, Xiangqing Shen, Jiajun Zhang, Yifan Zhang, Jian Cheng, Rui Xia  
**Category**: cs.AI  
**Published**: 2026-04-03  
**Score**: 7.5  
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
- **高成本的过程监督**（Process Supervision）：许多训练方法依赖于精细标注的中间步骤（如变量定义、公式推导、参考代码），这类数据构建成本高昂且难以扩展。
- **跨求解器泛化能力差**：模型在特定求解器（如 Gurobi）上训练后，迁移到其他求解器（如 OR-Tools）时性能急剧下降，因为其过拟合了特定API的语法模式。

### **提出的新方法：EVOM**
作者提出了 **Execution-Verified Optimization Modeling (EVOM)**，一种基于执行验证的强化学习框架，核心思想是将数学规划求解器（solver）视为一个**确定性的交互式验证器**。

#### **核心机制**
- 给定自然语言问题 `q` 和目标求解器 `s`，模型生成求解器专用的代码。
- 在沙箱环境中执行该代码，获取执行结果（状态、目标值等）。
- 将执行结果与真实答案对比，转化为**标量奖励**（scalar reward）。
- 使用 GRPO 或 DAPO 等无critic的强化学习算法进行策略更新，形成“生成-执行-反馈-更新”的闭环。

### **相比现有方法的优势**
| 方面 | 传统方法 | EVOM |
|------|--------|------|
| **监督信号** | 需要过程级标注（SFT）或闭源LLM多轮推理 | 仅需问题-答案对（outcome-only），无需中间步骤 |
| **跨求解器能力** | 差，需重建数据集重训练 | 支持零样本迁移（zero-shot transfer）和低成本适配（low-cost adaptation） |
| **部署成本** | 高（依赖闭源LLM或多步agent） | 低（使用开源小模型 + 执行反馈） |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **NL4OPT**：NeurIPS 2022竞赛数据集，专注于从自然语言描述生成线性规划模型。
- **MAMO**：评估LLM数学建模能力的基准，分为 EasyLP 和 ComplexLP 子集。
- **IndustryOR**：首个工业级OR问题基准，涵盖13个行业的真实场景。
- **OptiBench**：端到端优化问题求解基准，包含线性和非线性问题。

> 所有训练数据均来自 **OR-Instruct-3K**，但只保留问题描述 `q` 和最终答案 `a`，丢弃所有中间过程标注。

### **实验设置**
- **基础模型**：Qwen2.5-7B（作为policy model）
- **求解器**：Gurobi, OR-Tools, COPT
- **输出格式**：严格要求 `<think>...</think><code>...</code>` 双块结构，便于解析与执行
- **执行环境**：沙箱运行，限制10秒超时、2GB内存
- **奖励函数**：
  - `r_fmt`：格式合规性奖励（标签完整性、正则匹配）
  - `r_ans`：结果正确性奖励（基于求解器返回的状态和目标值）

### **评估指标**
- **Accuracy**：预测结果在相对误差容忍度 `ε_eval` 内即为正确
  - 主要结果使用 `ε_eval = 0.05`（5%）
  - 严格评估使用 `ε_eval = 1e-4`

### **基线方法对比**
| 类型 | 代表方法 |
|------|---------|
| **Prompting-based** | DeepSeek-R1, OpenAI o1, GPT-4o (CoT, CoE), OptiMUS |
| **Training-based (SFT)** | ORLM (Supervised Fine-Tuning with process supervision) |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表1：在 Gurobi 上的主实验结果（Accuracy %）**

| Model | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndustryOR | Avg |
|-------|-----------|--------|--------|--------|------------|-----|
| **ORLM (SFT)** | 60.96 | 84.89 | 88.34 | 35.71 | 27.00 | 59.38 |
| **EVOM (GRPO)** | **62.95** | **84.08** | **88.19** | **34.28** | **31.00** | **60.10** |

✅ **结论**：EVOM 在平均性能上**超越或持平**于需要过程监督的 SFT 方法，尤其在更难的 IndustryOR 和 OptiBench 上表现更好。

---

#### **表2：零样本求解器迁移（Gurobi → OR-Tools）**

| Model | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndusOR |
|-------|-----------|--------|--------|--------|---------|
| **ORLM (SFT)** | 3.49 | 4.89 | 0.00 | 1.42 | 6.00 |
| **EVOM (GRPO)** | **54.31** | **77.55** | **84.81** | **22.27** | **24.00** |

✅ **结论**：EVOM 实现了强大的**零样本迁移能力**，而 SFT 方法几乎完全失效。

---

#### **表3：不同求解器下的适配效果提升**

| Solver | 模型 | OptiBench | NL4OPT | MAMO-E | MAMO-C | IndustryOR |
|--------|------|-----------|--------|--------|--------|------------|
| Gurobi | Base → EVOM | +21.76 | +28.98 | +22.24 | +5.85 | +10.00 |
| OR-Tools | Base → EVOM | +11.63 | +17.96 | +28.99 | -2.37 | +11.00 |

✅ **结论**：通过切换执行环境并继续训练，EVOM 能快速适应新求解器，无需重新构建标注数据集。

---

### **消融实验结果**

#### **显式推理（`<think>`块）的作用**
- 移除 `<think>` 块导致性能显著下降，尤其是在逻辑复杂的任务（如 IndustryOR、MAMO-C）上。
- 表明 `<think>` 不仅是事后解释，而是**内部推理工作区**，对复杂建模至关重要。

#### **优化器选择（GRPO vs DAPO）**
- 图3显示两者性能曲线几乎一致，说明在执行验证框架下，**优化器选择不是主导因素**，框架本身更具决定性。

#### **小规模模型上的有效性**
- 对 Qwen2.5-1.5B 和 3B 模型应用 EVOM 后，性能大幅提升。
- 3B 模型经训练后接近甚至超过未训练的 7B 基础模型，证明该方法对资源受限场景具有实用价值。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Outcome-only supervision 是可行的**：仅靠执行反馈即可让模型学会复杂的优化建模，无需昂贵的过程监督。
2. ✅ **求解器可作为验证器**：利用 solver 的确定性行为提供可靠奖励信号，实现闭环学习。
3. ✅ **支持跨求解器泛化**：
   - **零样本迁移**：在源求解器上训练的模型可直接用于目标求解器，性能保持良好。
   - **低成本适配**：只需切换执行后端并继续训练，即可高效适配新求解器。
4. ✅ **提升小模型潜力**：EVOM 显著放大了中小规模 LLM 的建模能力，适合部署在边缘或低资源环境。

### **方法的局限性**
- **冷启动问题**：对于预训练中极少出现的求解器（如 COPT），初始生成成功率极低，导致稀疏奖励问题。
- **深层语义错误难以纠正**：虽然语法和变量类型错误大幅减少，但**约束遗漏或目标函数错误**等深层逻辑问题仍存在。
- **对执行失败敏感**：若模型长期无法生成可执行代码，则无法获得有效反馈，训练停滞。

### **未来工作方向**
1. **改进冷启动策略**：结合少量跨求解器翻译数据 + SFT + RL，缓解冷启动问题（已在 Appendix H/I 中初步探索）。
2. **引入分层奖励机制**：设计更细粒度的奖励信号，引导模型逐步完善建模过程（如先鼓励正确识别变量，再鼓励写出约束）。
3. **增强推理能力**：探索如何让模型在 `<think>` 中生成更有用的中间表示，进一步提升复杂问题的建模质量。
4. **多求解器联合训练**：研究是否可以通过同时暴露多个求解器环境，训练出真正通用的“求解器无关”建模能力（见 Appendix L）。

---

> **总结一句话**：  
> **EVOM 成功地将“求解器”变成了“老师”，通过执行反馈教会 LLM 如何建模，摆脱了对人工标注和特定API的依赖，为可扩展、低成本、跨平台的决策智能提供了新路径。**

</details>

---

### 9. [Adaptive Parallel Monte Carlo Tree Search for Efficient Test-time Compute Scaling](https://arxiv.org/abs/2604.00510)

**Authors**: Hongbeen Kim, Juhyun Lee, Sanghyeon Lee, Kwanghoon Choi, Jaehyuk Huh  
**Category**: cs.AI  
**Published**: 2026-04-03  
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

### 解决的问题
Monte Carlo Tree Search (MCTS) 是一种有效的 **Test-time Compute Scaling (TTCS)** 方法，能够显著提升大语言模型（LLM）在复杂推理任务中的表现。然而，MCTS 存在严重的**长尾延迟（long-tail latency）**问题，因其执行时间高度可变且依赖于序列化搜索过程。这导致在高并发服务场景下资源争用严重，影响系统吞吐量和服务稳定性。

现有优化如 **positive early exit** 能在早期发现高质量解时提前终止搜索，但在难以求解的任务上无效，无法缓解尾部延迟。

### 提出的新方法与新思路
本文从**系统级资源管理**视角重构 MCTS 推理流程，提出两个核心机制：

- **Negative Early Exit (NE)**  
  识别并剪枝“无前途”的搜索路径：当当前搜索树的所有叶节点均为“futile”（即其累积奖励已不可能达到接受阈值）时，立即终止该请求的搜索，避免浪费计算资源。

- **Adaptive Boosting（自适应增强）**  
  将因 early exit（包括正向和负向）释放的 GPU 计算资源，动态重新分配给更有潜力的并发请求，优先加速接近完成或进展较快的任务，从而提高整体系统效率。

此外，作者设计了一个集成调度框架，结合：
- **WU-PUCT 并行策略**：基于未观测计数（unobserved counts）实现高效的并行 rollout；
- **Selective Futility Check**：利用首步奖励与最终得分的相关性，选择性跳过低潜力分支的 futility 判断，进一步提升效率。

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **延迟控制** | 显著降低 p99 端到端延迟，尤其针对难例和尾部请求；相比串行 MCTS 最多降低 **2.83×**。 |
| **吞吐量** | 通过资源再利用和动态并行度控制，最大提升 **2.44× 吞吐量**。 |
| **准确性保持** | 在减少计算的同时维持原有 MCTS 的推理准确率，未出现明显下降。 |
| **系统友好性** | 主动管理系统资源，防止长任务垄断 GPU，提升服务可预测性和容量利用率。 |

---

## 2. 核心实验方法和设置

### 数据集
- **Math500**：源自 MATH 数据集的 500 道数学题，用于评估逐步推理准确性。
- **AMC23**：2023 年美国数学竞赛题目，测试泛化能力。由于样本少（仅 40 题），为压测性能引入不同前缀进行负载扩展，但精度报告仍基于原始数据。

### 模型配置
- **生成模型**：
  - `Llama-3.1-8B-Instruct`
  - `Qwen2.5-14B-Instruct`
- **奖励模型（PRM）**：
  - `Qwen2.5-Math-PRM-7B`，用于打分中间推理步骤

硬件平台：单节点 4×NVIDIA H100-SXM 80GB GPU（2 用于生成，2 用于奖励评分）

### 评估指标
- **p50 / p99 端到端延迟**（End-to-end latency）
- **请求吞吐量**（Throughput, req/sec）
- **推理准确率**（Accuracy）
- **生成 token 数量**（衡量计算开销）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Beam Search** | 正向早退 + beam width=8，作为并行搜索基线 |
| **Vanilla MCTS** | 传统串行 MCTS，无任何优化 |
| **PE (Positive Early Exit)** | 达到置信阈值后提前退出 |
| **PE+NE** | 加入负向早退机制 |
| **PE+NE+Boosting** | 完整系统，含资源重分配与自适应并行 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 指标 | 提升幅度（vs Vanilla） | 备注 |
|------|------------------------|------|
| **p99 延迟降低** | 最高达 **2.83×** | 在 `(Math500, Llama)` 场景下 |
| **vs PE 单独使用** | 额外降低 **1.08×~1.15×** p99 延迟 | 表明 NE 和 Boosting 的叠加效应 |
| **吞吐量提升** | 最高达 **2.44×** | 特别是在高负载下效果显著 |
| **token 生成量减少** | MCTS 比 Beam Search 更高效，尤其启用 early exit 后 |

#### 具体对比结果（见 Figure 7 & 8）

- **Beam Search** 虽然速度快，但由于 quadratic 解码成本，在高并发时产生严重排队，**p99 延迟最高**。
- **Vanilla MCTS** 比 Beam Search 平均改善 1.64× p99 延迟，但仍存在明显尾部延迟。
- **PE** 可使简单请求快速完成，平均 p99 下降 1.38×，但对难例无效。
- **PE+NE** 进一步将 p99 降低 1.49×（vs Vanilla），直接削减尾部行为。
- **PE+NE+Boosting** 实现最佳综合表现，p99 再降 1.15×（vs PE），总降幅达 **2.83×**。

> ⚠️ 异常情况：在 `(amc23, Qwen)` 配置中，Boosting 导致轻微延迟上升和吞吐下降。原因分析为：多数请求本可用少量串行 rollout 解决，过度并行反而引发调度开销和资源争抢。

#### 消融实验结果
- **Negative Early Exit 单独作用**：约 5% 请求被触发，虽比例不高，但集中在最难样本上，有效遏制尾部延迟。
- **Selective Futility Check**：通过忽略首步得分极低的路径参与判断，加快 NE 触发速度，提升资源回收效率。
- **Adaptive Boosting**：是吞吐量提升的关键驱动因素，特别是在高请求速率下释放的资源能被高效复用。

#### 准确率表现（Table 1）
| 方法 | Math500 (Llama) | AMC23 (Llama) | 变化趋势 |
|------|------------------|----------------|---------|
| Vanilla MCTS | 75.3% | 57.5% | — |
| PE | 76.2% | 47.5% | 波动明显 |
| Ours (Full) | 74.8% | 55.0% | 保持基本持平 |

> ✅ 结论：所有优化均未显著损害准确率。AMC23 上波动较大归因于样本量小（仅 40 题），单个错误对整体影响大。

---

## 4. 关键结论和发现

### 主要发现
1. **MCTS 的部署瓶颈不在准确率，而在系统资源管理不均**。  
   长尾延迟本质上是由少数难例持续消耗大量计算资源所致。

2. **Negative Early Exit 是抑制尾部延迟的有效手段**。  
   利用轨迹评分上限性质（cumulative product/min aggregation），可在数学上证明某些搜索已无成功可能，安全提前退出。

3. **资源应动态再分配而非静态预留**。  
   “Adaptive Boosting” 将空闲资源投向更可能成功的任务，形成正反馈循环，极大提升系统利用率。

4. **并行化需受控，非越多越好**。  
   不加限制的并行会加剧资源争用，尤其在轻负载或易解任务中反而降低性能。

### 方法的局限性
- **依赖高质量 PRM**：NE 的有效性建立在 PRM 打分可靠的基础上，若 reward signal 噪声大，则 futility 判断可能误判。
- **对超难问题仍可能全预算运行**：若始终存在一个“看似有希望”的叶节点（score > τ），则不会触发 NE。
- **并行开销不可忽视**：在本就能快速解决的任务中，过度并行会导致调度和通信开销超过收益。

### 未来工作方向
- **更智能的 early exit 条件建模**：引入学习型 predictor 判断搜索前景，替代固定阈值。
- **跨请求资源共享机制**：探索多个 MCTS 任务之间的共享子树或知识迁移。
- **异构硬件适配**：将生成与 reward 模型部署在不同设备上，进一步优化 pipeline 效率。
- **在线自适应参数调整**：根据实时负载自动调节 τ、parallelism 上限等超参。

---

> 📌 **总结一句话**：  
> 本文提出了一种系统级优化框架，通过 **Negative Early Exit** 和 **Adaptive Boosting** 动态管理 MCTS 的计算资源，在几乎不损失准确性的前提下，实现了高达 **2.83× 的 p99 延迟降低** 和 **2.44× 的吞吐提升**，推动 MCTS 成为可用于生产环境的大规模推理服务方案。

</details>

---

### 10. [Agent Q-Mix: Selecting the Right Action for LLM Multi-Agent Systems through Reinforcement Learning](https://arxiv.org/abs/2604.00344)

**Authors**: Eric Hanchen Jiang, Levina Li, Rui Sun, Xiao Liang, Yubei Li, Yuchen Wu, Haozheng Luo, Hengli Li, Zhi Zhang, Zhaolu Kang, Kai-Wei Chang, Ying Nian Wu  
**Category**: cs.CL  
**Published**: 2026-04-03  
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

### ✅ 解决的问题
当前基于 **Large Language Model (LLM)** 的多智能体系统（Multi-Agent Systems, MAS）在解决复杂任务时面临通信拓扑（communication topology）设计的关键挑战。现有方法存在以下两大局限：

- **静态拓扑**（如链式、星型、全连接）无法适应不同难度的任务，导致简单任务浪费 token，复杂任务协作不足。
- **自适应方法**（如 G-Designer、GTD）依赖**集中式生成器**统一决定整个通信图，缺乏灵活性，且不支持去中心化执行（Decentralized Execution），限制了在线适应能力。

### 🚀 提出的新方法：Agent Q-Mix
本文提出 **Agent Q-Mix**，一种将通信拓扑学习建模为**合作式多智能体强化学习**（Cooperative Multi-Agent Reinforcement Learning, MARL）问题的框架，核心思想如下：

- 将每个 agent 的通信行为建模为**局部离散动作选择**（如广播、查询、辩论、独立处理等），联合动作诱导出每轮的通信图。
- 采用 **QMIX** 进行值函数分解，在训练时集中优化全局奖励，部署时实现**去中心化决策**（Centralized Training with Decentralized Execution, CTDE）。
- 架构包含：
  - **Topology-aware GNN Encoder**：编码当前通信图结构。
  - **GRU Memory**：维护跨轮次的历史状态。
  - **Per-agent Q-heads**：每个 agent 学习自己的 Q 函数。
  - **QMIX Mixing Network**：确保单调性约束，保障 IGM（Individual-Global-Max）性质。

### 🔍 相比现有方法的优势
| 维度 | Agent Q-Mix | 传统方法 |
|------|-----------|--------|
| **决策方式** | 去中心化、动态、按轮次调整 | 集中式或固定模式 |
| **可解释性** | 动作空间语义明确（6类通信行为） | 黑箱生成图 |
| **效率** | 自动抑制冗余通信，节省 token | 固定高开销 |
| **鲁棒性** | 能识别并隔离故障/对抗 agent | 易受错误传播影响 |

---

## 2. 核心实验方法和设置

### 📚 数据集
在 **7个核心基准测试** 上进行评估，涵盖三大领域：

| 类别 | 数据集 |
|------|-------|
| **Coding** | LiveCodeBench v6, HumanEval |
| **Reasoning** | MMLU-Pro, Humanity's Last Exam (HLE) |
| **Mathematics** | AIME2025, AIME2026, Beyond-AIME, HMMT2025 |

此外还额外评估了极具挑战性的 **Humanity’s Last Exam (HLE)**。

### ⚙️ 实验设置
- **基础模型**：GPT-OSS:120B 和 Gemini-3.1-Flash-Lite。
- **Agent 团队配置**：
  - 每个任务使用 3 个 domain-specialist agents + 1 个 FinalRefer 决策节点。
  - 角色包括：MathSolver、CodeWriter、ReasoningAgent、AnalyzeAgent 等。
- **通信轮数**：
  - 数学任务：T=3 轮
  - 编码与推理任务：T=2 轮
- **训练细节**：
  - 使用仅 **15 个样本/领域** 进行训练，验证小样本泛化能力。
  - 奖励函数平衡准确率与 token 成本：
    $$
    R = w_{\text{acc}} \cdot \text{accuracy} - w_{\text{tok}} \cdot \min\left(\frac{\text{tokens\_used}}{10000}, 1\right)
    $$

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (%)** | 主要性能指标，报告中位数（3次运行） |
| **Token Usage** | 总消耗 token 数，衡量效率 |
| **Robustness** | 注入对抗 agent 后的性能下降程度 |
| **Ablation Study** | 分析组件有效性（如 agent 数量、训练数据量、通信轮数等） |

### 🆚 基线方法对比
分为四类基线：

| 类型 | 方法 |
|------|------|
| **单智能体** | Base (direct prompting) |
| **静态拓扑** | LLM-Debate |
| **自适应拓扑** | GPTSwarm, AgentDropout, G-Designer, MaAS, TopoDIM, GTD |
| **商业框架** | LangGraph, AutoGen, Microsoft Agent Framework, Lobster |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（GPT-OSS:120B）
| 方法 | 平均准确率 (Avg) |
|------|----------------|
| **Agent Q-Mix** | **72.73%** ✅ |
| GTD (最佳自适应) | 60.37% |
| AutoGen (最强商业框架) | 68.60% |

> ➕ **相对提升 +12.36 pts vs GTD**, **+4.13 pts vs AutoGen**

#### 在数学任务上表现尤为突出：
| 任务 | Agent Q-Mix | 第二名 |
|------|------------|--------|
| HMMT2025 | **53.33%** | 43.33% (+10.00) |
| Beyond-AIME | **42.00%** | 37.00% (AutoGen) |

#### 在 HLE 上的表现（Gemini-3.1-Flash-Lite）：
| 方法 | 准确率 | Token 消耗 |
|------|--------|-------------|
| **Agent Q-Mix** | **20.8%** ✅ | **163K** ✅ |
| Microsoft Agent Framework | 19.2% | 3.42M |
| LangGraph | 19.2% | 88.81M |
| AutoGen | ~18.9% | >3.89M |

> ✅ **最高准确率 + 最低 token 消耗**

### 🔄 与基线方法对比总结
- 在所有 7 个 benchmark 中，**Agent Q-Mix 取得最高平均准确率**。
- 在 **token 效率方面显著优于其他多智能体方法**（见 Table 3）：
  - MMLU-Pro：仅用 **112K tokens**，远低于其他方法（471K–2.71M）。
  - Beyond-AIME：708K vs 其他 1.00M–2.68M。
- **鲁棒性强**：在注入一个对抗 agent 后，准确率仅下降 **2.86 pts**（从 92.86% → 90.00%），而 LLM-Debate 下降 8.57 pts，AutoGen 下降 10.00 pts。

### 🔬 消融实验结果（Ablation Studies）
使用 Gemini-3.1-Flash-Lite 进行分析：

| 实验 | 发现 |
|------|------|
| **Agent 数量** | Beyond-AIME 准确率随 agent 数增加而上升，4 agent 达峰值（38%），10 agent 趋于饱和（41%）；HumanEval 始终接近完美（>96%），说明策略能避免冗余通信。 |
| **训练样本数** | 仅需 **15 个训练样例/领域** 即可达 95.73% 准确率，与 100 样本效果相当，证明**高度样本高效**。 |
| **通信轮数 T** | 数学任务需 T=3 才达最优（96.95%），T=1 仅 84.28%；编码/推理任务 T=2 已足够。支持默认设置合理性。 |
| **奖励权重 w_acc** | 提升 w_acc 可提高准确率，但 token 成本缓慢增长；最终选定 w_acc=1.50 (Gemini), 1.25 (GPT-OSS) 作为效率-性能权衡点。 |

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **去中心化的拓扑学习是可行且高效的**：通过将通信动作本地化，Agent Q-Mix 实现了灵活、可扩展的协作机制。
2. **QMIX 的价值分解机制非常适合该场景**：其单调混合网络保障了 IGM 性质，使训练与部署解耦成为可能。
3. **学习到的通信策略具有任务自适应性**：
   - 数学难题 → 多轮广播与聚合（dense communication）
   - 简单编码任务 → 独立处理为主（sparse communication）
4. **兼具高性能、高效率与强鲁棒性**：不仅准确率领先，还能主动规避错误信息传播，提升系统稳定性。

### ⚠️ 局限性
- 当前动作空间固定为 6 种类型，虽具通用性，但在极端复杂场景下可能受限。
- 依赖 LLM API 调用成本较高，端到端训练仍受限于 inference 开销。
- 当前实验集中在特定角色分工，尚未探索完全动态的角色分配机制。

### 🔮 未来工作方向
- 扩展动作空间以支持更复杂的交互模式（如工具调用、记忆读写）。
- 探索基于 LLM 的 policy network 替代 handcrafted prompts，实现端到端学习。
- 将 Agent Q-Mix 应用于真实世界任务流（workflow automation）、软件工程全流程协作等工业级场景。
- 结合 memory-augmented architectures 实现长期记忆共享与演化。

---

> 💡 **一句话总结**：  
> **Agent Q-Mix 成功将 MARL 中的 QMIX 框架引入 LLM 多智能体系统的通信拓扑学习，实现了去中心化、高效、鲁棒且可解释的协作机制，在多个维度上超越现有方法，推动了 multi-agent reasoning 的边界。**

</details>

---

### 11. [Intelligent Cloud Orchestration: A Hybrid Predictive and Heuristic Framework for Cost Optimization](https://arxiv.org/abs/2604.02131)

**Authors**: Heet Nagoriya, Komal Rohit  
**Category**: cs.DC  
**Published**: 2026-04-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.02131v1  

#### Abstract
Cloud computing allows scalable resource provisioning, but dynamic workload changes often lead to higher costs due to over-provisioning. Machine learning (ML) approaches, such as Long Short-Term Memory (LSTM) networks, are effective for predicting workload patterns at a higher level, but they can in...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Intelligent Cloud Orchestration: A Hybrid Predictive and Heuristic Framework for Cost Optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现代云环境中的动态工作负载导致资源管理面临显著挑战。传统方法存在以下两类问题：
- **Over-provisioning**：为应对流量峰值而过度分配资源，造成成本浪费。
- **Under-provisioning**：资源不足导致性能下降、SLA 违约。

现有的解决方案分为两类：
- **Machine Learning (ML)** 方法（如 LSTM、DRL）能预测长期负载趋势，但推理延迟高，在突发流量下响应慢。
- **Mathematical Heuristics**（如 Game Theory、Simulated Annealing）响应迅速，但缺乏对未来负载的预判能力。

因此，单一方法难以在 **cost efficiency** 和 **real-time responsiveness** 之间取得平衡。

---

### **提出了什么新方法或新思路**
本文提出了一种 **Hybrid Optimization-ML Framework**，将两种范式结合：
- **Macro-Level Prediction (ML)**：使用 **LSTM** 进行长期 workload forecasting，指导集群级别的容量规划（proactive scaling）。
- **Micro-Level Scheduling (Heuristic)**：采用轻量级 **Game Theory** 调度器进行实时任务分配，确保低延迟响应。

该框架通过分层架构实现“预测+执行”的协同优化。

> 如图 Fig. 2 所示：  
> **ML 模块输出预测容量需求 → Heuristic 模块基于当前可用资源边界进行即时任务调度**

---

### **相比现有方法的优势**
| 维度 | 优势 |
|------|------|
| **Cost Efficiency** | 接近纯 ML 方法的成本节省水平，避免 heuristic 方法因反应滞后导致的 over-provisioning |
| **Latency & Responsiveness** | 响应速度接近纯 heuristic 方法，优于 ML 方法的 inference delay |
| **SLA Compliance** | 在流量突增时仍能维持较低任务延迟，保障服务质量 |
| **部署可行性** | 减少对大规模 GPU 集群的依赖，降低 ML 模型的运行开销 |

---

## **2. 核心实验方法和设置**

### **使用的数据集 / 环境**
- 实验基于 **simulated cloud environment** 构建。
- 输入 workload 数据来源于 **Google Cloud historical traces**（参考文献 [4][20]），具有典型的时间周期性和突发性特征。
- 使用 **CloudSim** 或类似仿真平台进行调度模拟。

---

### **实验设置和评估指标**

#### **实验设置**
- 对比三种策略：
  1. **Standalone ML Model**（仅使用 LSTM 预测并扩容）
  2. **Mathematical Heuristic Only**（仅使用 Game Theory 实时调度）
  3. **Proposed Hybrid Framework**（LSTM + Game Theory 联合调度）

- 工作负载包含 **diurnal patterns** 和 **sudden traffic spikes**，模拟真实场景。

#### **评估指标**
| 指标 | 描述 |
|------|------|
| **Total Operational Cost** | 包括 VM 租赁费用、Spot Instance 中断成本等 |
| **Real-Time Task Latency** | 从任务提交到完成的时间，反映系统响应能力 |
| **SLA Violation Rate** | 因延迟过高或资源不足导致的服务协议违约比例 |
| **Resource Utilization** | CPU/Memory 利用率，衡量资源利用效率 |

---

### **基线方法对比**
| 基线方法 | 类型 | 特点 |
|--------|-----|------|
| **LSTM-only** | Predictive ML | 强于趋势预测，但 inference latency 高 |
| **Game Theory Scheduler** | Deterministic Heuristic | 响应快，但无法预见负载变化 |
| **Simulated Annealing** | Metaheuristic | 全局搜索能力强，收敛慢 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
根据 Fig. VI 中的图表结果（Total Operational Cost 和 Task Latency 曲线）：

| 方法 | 总运营成本 | 平均任务延迟 | SLA 合规性 |
|------|-----------|--------------|------------|
| **Standalone ML** | 最低 (~$2,500) | 高（峰值 >70ms） | 差（突发时延迟严重） |
| **Heuristic Only** | 最高 (~$7,500) | 低（稳定 ~30ms） | 较好（响应快） |
| **Proposed Hybrid** | 接近 ML (~$2,800) | 接近 Heuristic (~35ms) | **最优**（兼顾两者） |

> 注：数值为根据图表估算值，单位为任意成本/时间单位。

---

### **与基线方法的对比结果**
- **成本方面**：Hybrid 框架比 heuristic 方法节省约 **60% 成本**，仅比纯 ML 多出约 10%。
- **延迟方面**：在 traffic spike 发生时，ML 方法出现明显延迟上升（due to inference delay），而 Hybrid 和 Heuristic 表现平稳。
- **综合表现**：Hybrid 在 **cost-performance trade-off** 上达到 Pareto 最优。

---

### **消融实验结果（如有）**
虽然文中未明确列出消融实验（ablation study），但从对比设计中可推断：
- 若移除 LSTM 预测模块 → 回归为 heuristic 方法，成本显著上升。
- 若移除 Game Theory 实时调度 → 完全依赖 ML 推理决策 → 响应延迟增加，SLA 违约风险提高。

这间接验证了两个组件的必要性。

---

## **4. 关键结论和发现**

### **论文的主要发现**
1. **No Silver Bullet**：单独使用 ML 或 heuristic 都无法在动态云环境中实现全面优化。
2. **Synergy Matters**：将 **predictive intelligence** 与 **deterministic agility** 结合是未来方向。
3. **Layered Design Effective**：宏观预测 + 微观调度的分层架构可在不牺牲性能的前提下大幅降低成本。
4. **Practical Feasibility**：Hybrid 框架降低了对高性能硬件的依赖，更适合工业部署。

---

### **方法的局限性**
| 局限性 | 说明 |
|-------|------|
| **依赖历史数据质量** | LSTM 的预测精度受限于训练数据的完整性和代表性 |
| **跨云异构支持有限** | 当前框架假设单一云环境，multi-cloud 场景需额外适配 |
| **冷启动问题未完全解决** | 尽管未聚焦 serverless，但在快速扩缩容中仍可能存在 container cold start 延迟 |
| **模型更新机制缺失** | 未讨论如何在线更新 LSTM 模型以适应长期模式漂移（concept drift） |

---

### **未来工作方向**
1. **Lightweight ML Models**：
   - 探索 **neural network pruning**, **quantization**, **knowledge distillation** 技术压缩模型，提升推理速度。
2. **Federated Learning for Cost Awareness**：
   - 利用 **FedCostAware** [5] 等框架实现分布式训练，减少中心化计算负担。
3. **Vendor-Agnostic Orchestration Layers**：
   - 基于 **Kubernetes** 和 **Terraform** 构建多云成本感知调度器，支持自动迁移至低价区域。
4. **Unified Hybrid Architectures**：
   - 推动 **Optimization-ML Integrated Frameworks** 的标准化，打破算法孤岛。
5. **Real-Time Pricing Integration**：
   - 接入 **Spot Market APIs** 动态调整调度策略，最大化利用低成本资源。

---

> ✅ **总结一句话**：  
> 本文提出的 **Hybrid Predictive and Heuristic Framework** 成功弥合了 ML 预测性与 heuristic 实时性的鸿沟，在保持低延迟的同时实现了接近最优的成本控制，为智能云编排提供了可行路径。

</details>

---

### 12. [Matching Accuracy, Different Geometry: Evolution Strategies vs GRPO in LLM Post-Training](https://arxiv.org/abs/2604.01499)

**Authors**: William Hoy, Binxu Wang, Xu Pan  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.01499v1  

#### Abstract
Evolution Strategies (ES) have emerged as a scalable gradient-free alternative to reinforcement learning based LLM fine-tuning, but it remains unclear whether comparable task performance implies comparable solutions in parameter space. We compare ES and Group Relative Policy Optimization (GRPO) acro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Matching Accuracy, Different Geometry: Evolution Strategies vs GRPO in LLM Post-Training

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统地探讨了一个关键问题：当 **Evolution Strategies (ES)** 和基于梯度的强化学习方法（如 **GRPO**）在下游任务上达到相似准确率时，它们是否在参数空间中找到了相似的解？更进一步，这些方法对模型原有能力的保留（如遗忘、知识漂移）有何不同？

尽管已有研究分别验证了 ES 和 GRPO 在 LLM 微调中的有效性，但二者在**几何路径、更新模式和泛化影响上的根本差异尚未被深入理解**。

### 提出了什么新方法或新思路
本文并未提出新的优化算法，而是通过**系统的实证分析与理论建模**，揭示了两种范式之间的本质区别，并提出了一个统一的理论框架来解释观察到的现象：

- **信号-扩散分解理论（Signal-Diffusion Decomposition）**：  
  将 ES 的更新分解为两个部分：
  - **On-manifold signal**：沿任务相关高曲率方向的有用更新（类似 GD）。
  - **Off-manifold diffusion**：在平坦、弱信息维度上的随机游走式扩散，虽不影响当前任务性能，但导致大规模参数漂移。

这一理论首次从数学上解释了为何 ES 能以“看似低效”的大步长实现与 GRPO 相当的任务精度。

### 相比现有方法的优势
- **理论深度**：不仅报告现象，还提供了解释机制（如随机游走主导、线性模式连接无损失壁垒）。
- **多维评估视角**：结合任务准确率、KL 散度、权重空间几何、插值分析等，全面刻画优化轨迹。
- **揭示反直觉事实**：相同性能 ≠ 相似解 —— 二者更新方向近正交，且 ES 更新范数高出两个数量级。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验基于四个多样化任务进行单任务与顺序持续学习训练与评估：

| 任务 | 类型 | 数据来源 | 训练样本数 | 测试样本数 |
|------|------|----------|------------|-------------|
| **Countdown** | 算术推理 | `Jiayi-Pan/Countdown-Tasks-3to4` | 200 | 2,000 |
| **Math** | 数学求解 | `nlile/hendrycks-MATH-benchmark` | 200 | 500 |
| **SciKnowEval-Chemistry** | 化学知识问答 | `hicai-zju/SciKnowEval v2` | 200 | 2,000 |
| **BoolQ** | 是非阅读理解 | `google/boolq` | 200 | 2,000 |

此外，使用以下**持外任务**评估通用能力保留情况：
- **MMLU**：通用知识（14,042 样本）
- **IFEval (strict)**：指令遵循能力（541 样本）

### 实验设置和评估指标

#### 模型
- 基础模型：**Qwen3-4B-Instruct-2507**（约 40 亿参数）
- 微调方式：全参数微调（full-parameter fine-tuning）

#### 对比方法
| 方法 | 类型 | 关键配置 |
|------|------|---------|
| **ES (Evolution Strategies)** | 梯度无关 | Population size $N=30$, noise scale $\sigma=0.0015$, step size $\alpha=\sigma/2=7.5\times10^{-4}$ |
| **GRPO (Group Relative Policy Optimization)** | 梯度基础 | Learning rate $1\times10^{-5}$, AdamW optimizer, clip ratio $\epsilon=0.2$, no KL penalty ($\beta_{KL}=0$) |

#### 评估指标
1. **任务准确率（Accuracy %）**
2. **增量 KL 散度**：$\mathrm{KL}(\text{after} \| \text{before})$，衡量输出分布变化
3. **权重更新范数**：$\|\Delta \theta\|_2 = \|\theta_{\text{trained}} - \theta_{\text{base}}\|_2$
4. **线性模式连接（Linear Mode Connectivity）**：检查 ES 与 GRPO 解之间是否存在损失壁垒
5. **扰动方向分析**：沿 $\Delta\theta_{\text{ES}}$ 或 $\Delta\theta_{\text{GRPO}}$ 方向移动，观察任务性能与泛化能力的变化

---

## 3. 主要实验结果和性能指标

### 单任务训练性能（Table 1）

| Task | Base | ES (100) | ES (300) | GRPO |
|------|------|----------|----------|-------|
| Countdown | 20.5 | 75.1 | **78.6** | 74.5 |
| Math | 61.8 | 72.4 | **74.0** | 68.8 |
| Chemistry | 63.4 | 68.1 | **76.5** | 74.9 |
| BoolQ | 83.7 | 87.2 | **88.1** | 86.6 |

✅ **结论**：ES 在单任务上**匹配甚至超越 GRPO**，尤其在迭代次数足够时（300 步）表现最佳。

---

### 顺序持续学习稳定性（Figure 1）

- **ES (300)**：出现明显**灾难性遗忘**，前期任务性能显著下降。
- **ES (100)** 与 **GRPO**：均保持稳定，在整个序列中未见大幅退化。

➡️ 因此后续实验采用 **ES (100)** 进行公平比较。

---

### 权重更新范数对比（Table 3）

| Checkpoint | ES norm | GRPO norm | Ratio |
|-----------|--------|----------|--------|
| After Countdown | 87.28 | 1.00 | **87×** |
| After Math | 122.76 | 1.15 | **107×** |
| After Chemistry | 149.99 | 1.65 | **91×** |
| After BoolQ | 173.00 | 1.84 | **94×** |

📌 **核心发现**：ES 引起的参数变化量级是 GRPO 的 **近百倍**，表明其更新极为“发散”。

---

### KL 散度分析（Figure 2）

- **GRPO**：更新高度局部化，仅在目标任务上有显著 KL 变化，其他任务几乎不变。
- **ES**：引起广泛的**跨任务 KL 漂移**（off-diagonal drift），例如训练 Math 导致 BoolQ 输出分布偏移达 0.23 nats（而 GRPO 仅为 0.01）。

➡️ 表明 ES 更容易破坏原有功能，支持其更强的“干扰性”。

---

### 线性模式连接（Figure 3）

- 在 ES 与 GRPO 最终模型间进行线性插值：
  $$
  \theta(\alpha) = (1-\alpha)\theta_{\text{ES}} + \alpha\theta_{\text{GRPO}},\quad \alpha \in [0,1]
  $$
- 结果显示：**中间点无性能崩溃**，准确率平滑过渡，甚至有时优于端点。

✅ **结论**：尽管几何路径迥异，ES 与 GRPO **处于同一损失盆地（loss basin）内**，说明两者可达相似最优。

---

### 扰动方向分析（Figure 4–5）

- 沿归一化更新方向逐步放大扰动：
  - **GRPO 方向**：任务准确率快速上升后迅速下降，说明其方向紧凑高效。
  - **ES 方向**：准确率缓慢提升，需更大位移才达峰值，行为更像**随机方向**。
- 在持外任务（MMLU, IFEval）上，ES 与随机方向的退化曲线高度一致。

➡️ 支持理论预测：ES 的大部分位移位于**任务无关的平坦子空间**。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **匹配准确率，不同几何**：
   - ES 与 GRPO 可达到**相当的任务性能**，但在参数空间中采取完全不同的路径。
   - ES 更新范数大两个数量级，方向与 GRPO **近乎正交**（expected cosine similarity ~ $O(\sqrt{r/d})$）。

2. ✅ **ES 的本质是“信号+扩散”混合过程**：
   - 有用信号集中在低维任务流形；
   - 大部分更新表现为在高维平坦空间中的**各向同性随机游走**，符合理论预测 $\mathbb{E}[\|\Delta\theta\|^2] \propto \frac{\sigma^2 T d}{N}$。

3. ✅ **线性模式连接源于“损失不变分量”**：
   - 由于 off-manifold 分量对损失无影响，连接 ES 与 GRPO 的直线仍位于低损失区域。

4. ✅ **遗忘机制不同**：
   - GRPO 忘记主因是集中于特定任务导致过拟合；
   - ES 忘记则源于**广泛的知识漂移**，即使未训练的任务也会受影响。

5. ✅ **理论与实证高度吻合**：
   - 权重漂移随训练步数呈线性增长（$R^2 > 0.99$）；
   - 大多数权重矩阵表现出接近理想随机游走的行为（$d_{\text{eff}}/d \approx 96.8\%$）。

---

### 方法的局限性

- **计算成本高**：ES 需要大量并行采样（population-based），实际部署资源消耗远高于 GRPO。
- **不可控漂移风险**：虽然最终可达好解，但过程中可能发生不可逆的功能退化。
- **理论假设简化**：分析基于二次/线性奖励景观，真实 LLM 损失面更为复杂。
- **适用范围有限**：在需要严格控制参数变化的场景（如医疗、法律）中，ES 的大更新可能不可接受。

---

### 未来工作方向

1. **设计约束型 ES**：引入正则项或投影机制，抑制 off-manifold 扩散，提升稳定性。
2. **混合优化策略**：结合 ES 的探索能力与 GRPO 的精确性，实现“先粗搜再精调”。
3. **动态噪声调度**：根据训练阶段自适应调整 $\sigma$，早期鼓励探索，后期收敛减少扩散。
4. **结构化扰动**：仅在特定模块（如 attention heads）施加 ES，降低整体扰动规模。
5. **扩展至更多任务与模型**：验证结论在更大模型（如 70B+）或多模态场景下的普适性。

---

> 📌 **一句话总结**：  
> **ES 与 GRPO 能通往同样准确的终点，却走过截然不同的道路 —— 前者是一场穿越高维平原的漫游，后者是一次精准的定向攀登。**

</details>

---

### 13. [Learning from the Right Rollouts: Data Attribution for PPO-based LLM Post-Training](https://arxiv.org/abs/2604.01597)

**Authors**: Dong Shu, Denghui Zhang, Jessica Hullman  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.01597v1  

#### Abstract
Traditional RL algorithms like Proximal Policy Optimization (PPO) typically train on the entire rollout buffer, operating under the assumption that all generated episodes provide a beneficial optimization signal. However, these episodes frequently contain noisy or unfaithful reasoning, which can deg...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
传统基于 **Proximal Policy Optimization (PPO)** 的强化学习（RL）后训练方法在优化大型语言模型（LLMs）时，通常对整个 rollout buffer 中的所有生成样本进行无差别训练。然而，这些样本中常包含以下两类低质量数据：
- **Unfaithful Reasoning**：尽管最终答案正确，但推理过程存在逻辑错误（如“False Positive”、跳跃式推理等）。
- **Redundant Episodes**：模型已掌握的知识路径，反复训练无法提供有效学习信号。

这类噪声数据会降低策略更新的质量，导致训练效率低下甚至性能下降。

### 提出的新方法：Influence-Guided PPO (I-PPO)
本文提出 **I-PPO**，一种将 **data attribution** 技术集成到 PPO 后训练循环中的新框架。其核心思想是：
- 利用梯度对齐（gradient alignment）计算每个 rollout episode 对验证集的“影响分数”（influence score）。
- 若某 episode 的影响分数为负，说明其梯度方向与高质量人类偏好推理（validation gradient）相反，应被过滤。
- 仅保留正影响的 episodes 进行策略更新，并对它们按影响分数重新加权。

该方法实现了动态的数据筛选机制，使模型更专注于高质量、有建设性的推理轨迹。

### 相比现有方法的优势
- **更高的训练效率**：通过动态减少 rollout buffer 大小，加速收敛，尤其在后期形成“内在早停机制”（intrinsic early stopping mechanism）。
- **更强的推理能力**：有效抑制 unfaithful reasoning，提升模型输出的可靠性和一致性。
- **无需额外奖励建模**：利用 data attribution 作为隐式的、免训练的 **process reward signal**，弥补了传统 outcome-based reward 无法捕捉推理质量的缺陷。

---

## 2. 核心实验方法和设置

### 数据集
实验覆盖数学、物理和社交常识三大推理领域，共使用五个数据集：
- **数学推理**：`GSM8K`, `CollegeMath`, `MATH`
- **物理推理**：`OlympiadBench`（奥赛级别）
- **社会常识推理**：`ECQA`

### 模型架构
为验证通用性，实验在五种不同规模的 SFT 模型上进行：
- `Rho-1B`, `Gemma-2-2B`, `Qwen2.5-3B`, `Phi-3-4B`, `LLaMA-3-8B`

### 评估指标
采用三项标准指标衡量性能：
- **Majority Vote (MV)**：对每个问题生成多个响应，取最频繁答案判断正确性，反映整体性能。
- **Exact Match (EM)**：单个响应的准确率，衡量样本级可靠性。
- **Pass@K (PK)**：至少有一个响应正确的概率，反映模型潜在能力上限。

### 基线方法对比
- **SFT Baseline**：监督微调后的初始模型。
- **Traditional PPO**：标准 PPO 算法，使用完整 rollout buffer。
- **GRPO**：Group Relative Policy Optimization，作为替代 PPO 的对比算法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）
在所有模型和数据集上，**I-PPO 在 Majority Vote 和 Exact Match 上均显著优于 SFT 和传统 PPO**。例如：
- 在 `Rho-1B` + `GSM8K` 上，MV 从 PPO 的 50.05 提升至 **51.93**，EM 从 41.11 提升至 **46.10**。
- 在 `LLaMA-3-8B` + `GSM8K` 上，MV 达到 **80.21**，EM 达到 **73.36**，全面领先。

### 与基线方法的对比
- **相比 SFT**：所有 RL 方法均有明显提升，表明 RL 能有效对齐模型行为。
- **相比 PPO**：I-PPO 在绝大多数设置下表现更优，尤其在 EM 指标上优势显著，说明其能更有效地提升单次推理的准确性。
- **相比 GRPO**：I-PPO 也全面超越 GRPO（见 Table 2），证明其在实例级梯度隔离上的设计更合理。

### 消融实验结果
- **移除重加权机制（Reweighting）**：若仅过滤负影响样本而不按影响分数加权，性能出现一致下降（见 Figure 3 和 Figure 8）。这表明影响分数不仅可用于二值过滤，其数值本身蕴含了样本价值的细粒度信息。
- **结论**：完整的 I-PPO 框架（过滤 + 加权）是实现最优性能的关键。

---

## 4. 关键结论和发现

### 主要发现
1. **I-PPO 是有效的推理质量过滤器**：
   - 定性分析显示，负影响 episodes 中包含大量 **False Positive**, **Nonsensical Reasoning**, 和 **Reasoning Shortcuts**。
   - 正影响 episodes 更倾向于展示 **step-by-step procedural arithmetic** 和 **clear algebraic formulations**。
2. **I-PPO 实现了内在早停机制**：
   - 随着训练推进，负影响 episodes 比例上升，rollout buffer 被动态压缩，训练速度加快（见 Figure 2 和 Figure 5）。
3. **I-PPO 提供了隐式的过程奖励信号**：
   - 尽管使用的是 outcome-based reward，I-PPO 仍能识别并过滤掉推理错误但答案正确的样本，证明其 data attribution 机制可作为免训练的 **process reward**。

### 方法的局限性
- **依赖验证集质量**：data attribution 的准确性高度依赖于验证集的质量。若验证集本身存在噪声或偏差，可能误导影响分数计算。
- **计算开销增加**：初期需计算每个 episode 的梯度与验证梯度的点积，带来额外计算成本（尽管后期因 buffer 缩减而抵消）。

### 未来工作方向
- 探索更高效的 data attribution 近似方法以进一步降低开销。
- 将 I-PPO 思路扩展至其他 RL 算法（如 DPO 变体）或非推理任务。
- 研究如何自适应构建和更新高质量验证集，以增强方法鲁棒性。

> **代码地址**：https://anonymous.4open.science/r/Influence_ppo-4C37

</details>

---

### 14. [LLM REgression with a Latent Iterative State Head](https://arxiv.org/abs/2604.01206)

**Authors**: Yiheng Su, Matthew Lease  
**Category**: cs.CL  
**Published**: 2026-04-03  
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
在大型语言模型（LLM）主导的自然语言处理时代，大多数任务被统一为“文本到文本”（text-to-text）范式。然而，**回归任务**（regression）——如语义相似度评分、机器翻译质量估计等——因其目标是连续的标量值而非文本序列，与该范式存在根本性不匹配。

现有方法面临以下挑战：
- **Autoregressive Decoding**：将数值作为文本生成，依赖token级别的交叉熵损失，无法感知数值间的距离（如0.9比0.1更接近1.0），导致预测误差大。
- **Regression-aware Inference (RAIL/RAFT)**：虽能通过贝叶斯最优决策规则（如后验均值）提升性能，但需多次采样或枚举候选值，推理成本高。
- **Predictive Head Methods**：直接从LLM表示预测数值，效率高，但通常采用简单的池化策略（如mean-pooling），难以充分提取回归相关的细粒度信息。

### 提出的新方法：RELISH
作者提出 **RELISH (REgression with a Latent Iterative State Head)**，一种轻量级、高性能的LLM回归架构。

其核心思想是：
- **冻结LLM主干**，仅训练一个小型的附加模块。
- 引入一个**可学习的潜在状态**（learned latent state）`r[0]`。
- 通过**迭代式的交叉注意力机制**（iterative cross-attention），让该潜在状态反复与LLM的token-level表示交互，逐步提炼和聚合信息。
- 最终，将优化后的潜在状态 `r[L]` 输入一个**线性回归器**（linear regressor）以输出标量预测。

### 相比现有方法的优势
- **性能更强**：超越了三大类基线方法（自回归解码、回归感知推理、现有预测头），成为新的SOTA。
- **效率更高**：单次前向传播即可完成推理，避免了多轮采样的计算开销。
- **参数更少**：仅需约 **3.4–3.7M** 可训练参数（占LLM总参数的 **0.01–0.04%**），远低于LoRA等微调方法（0.26–0.42%）。
- **设计更优**：用动态的、基于注意力的迭代提炼机制取代了静态的池化操作，能更灵活地捕捉分布式的、局部的回归信号。

---

## 2. 核心实验方法和设置

### 数据集
在五个涵盖两类经典文本回归任务的数据集上进行评估：
- **语义文本相似度 (Semantic Textual Similarity, STS)**：
  - **STS-Benchmark (STS-B)**：英文句子对，评分范围 [0, 5]。
  - **SICKR-STS**：涉及组合知识的句子对，评分范围 [1, 5]。
- **机器翻译质量估计 (Machine Translation Quality Estimation, WMT)**：
  - **WMT_EN_ZH**：英译中，评分范围 [0, 100]。
  - **WMT_RU_EN**：俄译英，评分范围 [0, 100]。
  - **WMT_SI_EN**：僧伽罗语译英，低资源场景，评分范围 [0, 100]。

### 实验设置
- **LLM 主干**：使用四个不同规模的LLM：
  - Llama 3.1 8B Instruct
  - Qwen3 8B
  - Qwen3 32B
  - Gemma 3 27B Instruct
- **训练模式**：在两种模式下评估：
  - **Frozen**：仅训练预测头，LLM主干冻结。
  - **Fine-tuned (LoRA/RAFT)**：使用LoRA进行参数高效微调，或使用RAFT进行回归感知微调。
- **评估指标**：
  - **Pearson 相关系数 (r)**：衡量线性相关性。
  - **Spearman 相关系数 (ρ)**：衡量排序一致性。
  - **归一化均方根误差 (NRMSE)**：`RMSE / (y_max - y_min)`，用于跨不同评分范围的数据集比较，越低越好。

### 基线方法对比
与来自三大类别的多种基线方法进行了全面对比：
- **Autoregressive Decoding**：
  - Zero-shot Prompting
  - Many-shot Prompting (128个示例)
- **Regression-aware Inference**：
  - RAIL (基于采样的后验均值)
  - RAFT (结合回归感知微调)
- **Predictive Head Methods**：
  - Linear：对mean-pooled表示接线性层。
  - MLP：对mean-pooled表示接两层MLP。
  - 所有基线均在frozen和LoRA微调版本下测试。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（表3宏观平均）
在五数据集、四LLM、三随机种子上的宏平均结果：

| 方法 | Pearson (r↑) | Spearman (ρ↑) | NRMSE↓ |
| :--- | :--- | :--- | :--- |
| **RELISH (Ours)** | **76.3** | **74.0** | **13.3** |
| RAFT (第二好) | 71.5 | 69.0 | 16.6 |

- RELISH相比第二好的基线（RAFT）：
  - Pearson 提升 **6.7%**
  - Spearman 提升 **7.2%**
  - NRMSE 降低 **19.9%**

### 与基线方法的对比结果
- **全面领先**：RELISH在几乎所有数据集和LLM组合上都取得了最佳性能。
- **显著优于简单预测头**：传统的Linear和MLP预测头表现最弱，甚至不如零样本提示（Zero-shot），验证了静态池化的“信息坍塌”问题。
- **优于高成本方法**：尽管RAIL/RAFT需要多轮采样且计算成本高，RELISH仍能超越它们，证明了其架构的有效性。
- **在困难任务上优势更大**：在机器翻译质量估计（尤其是WMT_EN_ZH）这类需要捕捉局部错误的任务上，RELISH的增益最为显著。

### 消融实验结果
#### (1) 损失函数影响（Huber vs. MSE）
- 对比RELISH使用Huber损失和MSE损失的性能。
- 结果显示，两种损失下RELISH均大幅优于基线。
- Huber损失带来**小幅额外增益**（尤其在相关系数上），但主要性能提升来源于**架构本身**，而非损失函数选择。

#### (2) 迭代深度（L）的影响
- 系统性地改变迭代次数 `L`（从1到5）。
- `L=1`（即单步注意力池化）已是一个强基线，优于传统池化。
- `L≥2` 后性能**显著提升**，证明了迭代提炼的必要性。
- 在 `L=3` 时达到较好平衡，之后收益递减。
- 不同LLM的最佳 `L` 可能不同，表明最优深度与模型内部表示几何有关。

---

## 4. 关键结论和发现

### 主要发现
1. **RELISH是当前最优的LLM回归方案**：它成功地在**性能**、**效率**和**参数量**之间取得了前所未有的平衡。
2. **迭代提炼机制至关重要**：相比于静态池化，通过交叉注意力对潜在状态进行多轮迭代更新，能够更有效地从LLM的隐藏状态中提取回归相关信息。
3. **性能瓶颈在于信息提取，而非知识获取**：对于许多回归任务，LLM的预训练表示中已蕴含足够信号，关键是如何“读取”这些信号。RELISH的轻量级设计证明，精细的信息提取机制比大规模微调更能释放LLM的回归潜力。
4. **在需要细粒度分析的任务上优势明显**：RELISH在机器翻译质量估计等任务上提升最大，因为它能通过注意力机制定位并整合输入中的局部错误。

### 方法的局限性
- **依赖LLM主干的质量**：RELISH作用于冻结的LLM之上，其性能上限受限于主干模型的表示能力。对于需要全新领域知识的任务，可能仍需微调主干。
- **序列长度扩展性**：其交叉注意力机制的时间复杂度与输入序列长度呈线性关系，对于超长文本（如整篇论文）可能成为瓶颈。
- **任务范围有限**：目前评估集中在成对输入的回归任务（如STS、MT QE）。对于单句情感分析或非成对任务的适用性有待验证。

### 未来工作方向
- **扩展至其他任务**：探索在奖励建模（Reward Modeling）、LLM-as-a-judge、价格预测等更多回归场景的应用。
- **不确定性量化**：通过将线性回归器替换为分位数回归头（quantile heads），使RELISH能够输出预测区间而不仅是点估计。
- **处理更长序列**：结合稀疏注意力或记忆压缩技术，以适应超长文档的回归需求。
- **探索更复杂的潜在空间动态**：研究不同的迭代更新机制（如RNN、ODE-inspired）是否能进一步提升性能。

</details>

---

### 15. [UQ-SHRED: uncertainty quantification of shallow recurrent decoder networks for sparse sensing via engression](https://arxiv.org/abs/2604.01305)

**Authors**: Mars Liyao Gao, Yuxuan Bao, Amy S. Rude, Xinwei Shen, J. Nathan Kutz  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 6.5  
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
传统的 **SHRED**（Shallow REcurrent Decoder）网络虽然在从超稀疏传感器测量中重建高维时空场方面表现出色，但它仅提供**点估计**（point estimation），缺乏对预测不确定性的量化能力。这在科学应用中是严重缺陷，尤其是在系统具有**随机性、高频动态或数据稀缺**的情况下，无法支持风险评估、异常检测和不确定性下的决策。

### 提出的新方法
本文提出 **UQ-SHRED**，一种基于 **engression** 框架的分布学习方法，用于实现有效的不确定性量化（Uncertainty Quantification, UQ）。其核心思想是：
- 在输入传感器数据时**注入高斯噪声**（noise injection）；
- 使用 **energy score** 作为训练损失函数，直接优化分布预测的 proper scoring rule；
- 通过单次前向传播生成多个带噪声的样本，从而得到输出的**经验预测分布**（empirical predictive distribution）。

该方法无需额外网络结构或集成训练，仅需一次训练即可生成完整的条件分布 $ P(y_t | x_{t-L+1:t}) $。

### 相比现有方法的优势
| 方法 | 局限性 | UQ-SHRED 的优势 |
|------|--------|----------------|
| **Deterministic SHRED** | 无不确定性估计 | 提供完整分布输出 |
| **Bayesian Neural Networks** | 计算昂贵，后验近似复杂 | 无需变分推断或采样，计算高效 |
| **Deep Ensembles** | 需要训练多个模型，成本高 | 单一网络，仅需输入噪声注入 |
| **MC-Dropout** | 缺乏理论一致性保证 | 具有严格的能量分数（energy score）理论支撑 |
| **Normalizing Flows / Diffusion Models** | 架构受限或推理过程迭代耗时 | 前馈式快速推理，适合实时应用 |

> ✅ **核心优势**：**最小计算开销 + 理论可证的有效UQ + 易于部署**

---

## 2. 核心实验方法和设置

### 使用的数据集
实验覆盖五个来自不同科学领域的复杂真实世界数据集，验证泛化能力：

| 数据集 | 描述 | 特点 |
|-------|------|------|
| **SST** (Sea-surface temperature) | NOAA 海表温度再分析数据（1992–2019） | 高维空间网格（180×360），长期季节变化 |
| **Isotropic Turbulent Flow** | JHTDB 各向同性湍流压力场模拟 | 强非线性、多尺度动力学 |
| **Neural Activity** | Allen Institute 小鼠视觉皮层 LFP 记录 | 生物信号噪声大，高频波动 |
| **Solar Activity** | NASA SDO 太阳极紫外图像（171 Å） | 太阳耀斑等瞬态事件导致剧烈变化 |
| **1D RDE Transient Stage** | 旋转爆震发动机（Rotating Detonation Engine）简化模型 | 初始条件扰动下波前传播建模，测试外推鲁棒性 |

### 实验设置
- **传感器数量**：通常仅使用 **2–3个随机放置的传感器**
- **时间滞后窗口 L**：52–100 步（捕捉足够延迟嵌入）
- **噪声维度 $ d_e $**：50–1000，随输入规模调整
- **Monte Carlo 样本数 K**：50–200（用于推理时生成预测分布）

### 评估指标
| 指标 | 定义 | 目标 |
|------|------|------|
| **Calibration** | 观测值落入预测区间的真实比例 vs 名义置信水平（如95%） | 接近对角线表示良好校准 |
| **CRPS** (Continuous Ranked Probability Score) | 衡量预测分布的整体质量（兼顾校准性和锐度） | 越低越好 |
| **Sharpness** | 平均预测区间宽度 | 在正确校准前提下越窄越好（更精确） |
| **RMSE** | 中位数预测与真值之间的均方根误差 | 衡量点估计精度 |

### 基线方法对比
- 主要对比对象为原始 **Deterministic SHRED**
- 所有实验保持相同网络架构，仅将输出头由回归改为分布建模
- 未与其他UQ方法（如ensemble）直接比较，强调“零额外成本”特性

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1 & 2）

#### 表1：四个主要数据集上的UQ-SHRED表现

| Dataset | RMSE | 95% Calibration (%) | CRPS | 95% CI Width |
|--------|------|---------------------|------|-------------|
| **SST** | 0.656 | 90.8 | 0.343 | 2.791 |
| **ISO (Turbulent Flow)** | 0.656 | 85.7 | 0.032 | 0.209 |
| **Neural** | 3.54e-5 | 91.3 | 1.8e-5 | 1.14e-4 |
| **Solar** | 0.029 | 94.6 | 0.016 | 0.139 |

> 🔹 所有数据集上，**观测覆盖率接近名义水平**（如95%置信区间的实际覆盖率为90.8%~94.6%），表明高度**校准良好**（well-calibrated）  
> 🔹 **CRPS较低**，说明预测分布兼具准确性与集中性  
> 🔹 **CI宽度随物理复杂度自适应变化**：在突变区域（如太阳耀斑、神经脉冲）显著增宽

#### 表2：1D RDE 外推场景下的表现（含分布偏移测试）

| Test Case | RMSE (Median) | 95% Coverage | CRPS | 95% CI Width |
|----------|---------------|--------------|------|--------------|
| Closer (Run 0) | 0.156 (P), 0.307 (T) | 95.9% | 0.070 | 0.556 |
| Further (Run 1) | 0.194 (P), 0.398 (T) | 91.5% | 0.096 | 0.569 |
| Deterministic SHRED (Run 0) | 0.145 (P), 0.274 (T) | — | — | — |

> 🔹 尽管 RMSE 略高于确定性模型，但 **UQ-SHRED在外推任务中展现出更强稳定性**
> 🔹 图9显示：**Deterministic SHRED 在训练时间边界后预测发散严重**，而 **UQ-SHRED 的预测分布保持一致**

---

### 消融实验结果（Ablation Study on SST）

| 因素 | 影响 |
|------|------|
| **Temporal Lag (L)** | RMSE 随 L 增加下降，符合 Takens’ 延迟嵌入定理；但 UQ 校准性对 L 不敏感（L ≥ 10 后稳定） |
| **Sensor Number** | 更多传感器降低 RMSE 和 CI 宽度，但 **95% 覆盖率提升有限**，建议同步增加噪声维度 $ d_e $ |
| **Noise Dimension $ d_e $** | 过小会导致预测过自信（underdispersed），过大则增加计算负担；推荐 $ d_e \propto $ 输入维度 |
| **Monte Carlo Sample Size K** | K ≥ 100 可有效减少分位数估计偏差；低于50时校准性下降明显 |
| **Training Epochs** | ⭐ **关键发现**：即使在欠拟合或过拟合状态下，UQ-SHRED 仍能产生**统计上有效的不确定性估计**（平均95%分位覆盖率达91.6%~91.8%） |

> ✅ 支持 **early stopping 可用于防止过度自信预测**

---

## 4. 关键结论和发现

### 主要发现
1. **UQ-SHRED 成功实现了对 SHRED 架构的有效扩展**，使其能够输出**经过良好校准的预测分布**，而无需牺牲计算效率。
2. **不确定性估计具有物理意义**：CI 宽度在系统动态剧烈变化区域（如爆震波前、太阳耀斑、神经高频活动）自动增大，反映出真实的重建模糊性。
3. **energy score 损失提供了强大的正则化效果**，使得模型在时间外推任务中比传统 MSE 训练更加稳健，避免初始化依赖导致的预测发散。
4. **消融实验证明 UQ-SHRED 对多种超参数设置具有鲁棒性**，尤其在训练不充分或过拟合情况下仍能维持有效UQ。

### 方法的局限性
1. **理论保证依赖于 population optimum**：有限数据下可能出现轻微校准偏差（如 RDE 实验中强分布偏移时 coverage 下降至91.5%）。
2. **未提供有限样本下的 coverage guarantee**：可结合 conformal calibration 方法进一步改进。
3. **当前框架假设噪声独立同分布**，未建模传感器噪声结构或异方差性。
4. **高维输出空间中 energy score 的估计可能存在偏差**，尤其当 MC 样本数不足时。

### 未来工作方向
- 结合 **conformal prediction** 技术进行后处理校准，提升有限样本下的 coverage 保证
- 扩展至 **multi-scale 或 transformer-based SHRED 变体**
- 探索 **structured noise injection**（如时空相关噪声）以更好匹配物理先验
- 应用于 **主动传感设计**（active sensing）和 **安全关键型控制** 场景

---

> 📌 **总结一句话**：  
> **UQ-SHRED 是首个为 SHRED 类稀疏感知模型提供理论支持、计算高效且物理可解释的不确定性量化框架，在多个科学领域展现了强大鲁棒性与实用性。**

</details>

---

### 16. [Detecting Complex Money Laundering Patterns with Incremental and Distributed Graph Modeling](https://arxiv.org/abs/2604.01315)

**Authors**: Haseeb Tariq, Alen Kaja, Marwan Hassani  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.01315v1  

#### Abstract
Money launderers take advantage of limitations in existing detection approaches by hiding their financial footprints in a deceitful manner. They manage this by replicating transaction patterns that the monitoring systems cannot easily distinguish. As a result, criminally gained assets are pushed int...

---

### 17. [World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry](https://arxiv.org/abs/2604.01985)

**Authors**: Yuejiang Liu, Fan Feng, Lingjing Kong, Weifeng Lu, Jinzhou Tang, Kun Zhang, Kevin Murphy, Chelsea Finn, Yilun Du  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.01985v1  

#### Abstract
General-purpose world models promise scalable policy evaluation, optimization, and planning, yet achieving the required level of robustness remains challenging. Unlike policy learning, which primarily focuses on optimal actions, a world model must be reliable over a much broader range of suboptimal ...

---

### 18. [TRIMS: Trajectory-Ranked Instruction Masked Supervision for Diffusion Language Models](https://arxiv.org/abs/2604.00666)

**Authors**: Lingjie Chen, Ruizhong Qiu, Yuyu Fan, Yanjun Zhao, Hanghang Tong  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.00666v1  

#### Abstract
Diffusion language models (DLMs) offer a promising path toward low-latency generation through parallel decoding, but their practical efficiency depends heavily on the decoding trajectory. In practice, this advantage often fails to fully materialize because standard training does not provide explicit...

---

### 19. [EXaCTz: Guaranteed Extremum Graph and Contour Tree Preservation for Distributed- and GPU-Parallel Lossy Compression](https://arxiv.org/abs/2604.01397)

**Authors**: Yuxiao Li, Mingze Xia, Xin Liang, Bei Wang, Hanqi Guo  
**Category**: cs.DC  
**Published**: 2026-04-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.01397v1  

#### Abstract
This paper introduces EXaCTz, a parallel algorithm that concurrently preserves extremum graphs and contour trees in lossy-compressed scalar field data. While error-bounded lossy compression is essential for large-scale scientific simulations and workflows, existing topology-preserving methods suffer...

---

### 20. [Massively Parallel Exact Inference for Hawkes Processes](https://arxiv.org/abs/2604.01342)

**Authors**: Ahmer Raza, Hudson Smith  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.01342v1  

#### Abstract
Multivariate Hawkes processes are a widely used class of self-exciting point processes, but maximum likelihood estimation naively scales as $O(N^2)$ in the number of events. The canonical linear exponential Hawkes process admits a faster $O(N)$ recurrence, but prior work evaluates this recurrence se...

---

### 21. [LI-DSN: A Layer-wise Interactive Dual-Stream Network for EEG Decoding](https://arxiv.org/abs/2604.01889)

**Authors**: Chenghao Yue, Zhiyuan Ma, Zhongye Xia, Xinche Zhang, Yisi Zhang, Xinke Shen, Sen Song  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.01889v1  

#### Abstract
Electroencephalography (EEG) provides a non-invasive window into brain activity, offering high temporal resolution crucial for understanding and interacting with neural processes through brain-computer interfaces (BCIs). Current dual-stream neural networks for EEG often process temporal and spatial ...

---

### 22. [SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization](https://arxiv.org/abs/2604.02268)

**Authors**: Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Chengcheng Han, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.02268v1  

#### Abstract
Agent skills, structured packages of procedural knowledge and executable resources that agents dynamically load at inference time, have become a reliable mechanism for augmenting LLM agents. Yet inference-time skill augmentation is fundamentally limited: retrieval noise introduces irrelevant guidanc...

---

### 23. [A Safety-Aware Role-Orchestrated Multi-Agent LLM Framework for Behavioral Health Communication Simulation](https://arxiv.org/abs/2604.00249)

**Authors**: Ha Na Cho  
**Category**: cs.AI  
**Published**: 2026-04-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.00249v1  

#### Abstract
Single-agent large language model (LLM) systems struggle to simultaneously support diverse conversational functions and maintain safety in behavioral health communication. We propose a safety-aware, role-orchestrated multi-agent LLM framework designed to simulate supportive behavioral health dialogu...

---

### 24. [More Human, More Efficient: Aligning Annotations with Quantized SLMs](https://arxiv.org/abs/2604.00586)

**Authors**: Jiayu Wang, Junyoung Lee  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.00586v1  

#### Abstract
As Large Language Model (LLM) capabilities advance, the demand for high-quality annotation of exponentially increasing text corpora has outpaced human capacity, leading to the widespread adoption of LLMs in automatic evaluation and annotation. However, proprietary LLMs often exhibit systematic biase...

---

### 25. [LangMARL: Natural Language Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2604.00722)

**Authors**: Huaiyuan Yao, Longchao Da, Xiaoou Liu, Charles Fleming, Tianlong Chen, Hua Wei  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.00722v1  

#### Abstract
Large language model (LLM) agents struggle to autonomously evolve coordination strategies in dynamic environments, largely because coarse global outcomes obscure the causal signals needed for local policy refinement. We identify this bottleneck as a multi-agent credit assignment problem, which has l...

---

### 26. [Positional Cognitive Specialization: Where Do LLMs Learn To Comprehend and Speak Your Language?](https://arxiv.org/abs/2604.00923)

**Authors**: Luis Frentzen Salim, Lun-Wei Ku, Hsing-Kuo Kenneth Pao  
**Category**: cs.CL  
**Published**: 2026-04-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.00923v1  

#### Abstract
Adapting large language models (LLMs) to new languages is an expensive and opaque process. Understanding how language models acquire new languages and multilingual abilities is key to achieve efficient adaptation. Prior work on multilingual interpretability research focuses primarily on how trained ...

---

### 27. [DDCL: Deep Dual Competitive Learning: A Differentiable End-to-End Framework for Unsupervised Prototype-Based Representation Learning](https://arxiv.org/abs/2604.01740)

**Authors**: Giansalvo Cirrincione  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.01740v1  

#### Abstract
A persistent structural weakness in deep clustering is the disconnect between feature learning and cluster assignment. Most architectures invoke an external clustering step, typically k-means, to produce pseudo-labels that guide training, preventing the backbone from directly optimising for cluster ...

---

### 28. [Graph Neural Operator Towards Edge Deployability and Portability for Sparse-to-Dense, Real-Time Virtual Sensing on Irregular Grids](https://arxiv.org/abs/2604.01802)

**Authors**: William Howes, Jason Yoo, Kazuma Kobayashi, Subhankar Sarkar, Farid Ahmed, Souvik Chakraborty, Syed Bahauddin Alam  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.01802v1  

#### Abstract
Accurate sensing of spatially distributed physical fields typically requires dense instrumentation, which is often infeasible in real-world systems due to cost, accessibility, and environmental constraints. Physics-based solvers address this through direct numerical integration of governing equation...

---

### 29. [annbatch unlocks terabyte-scale training of biological data in anndata](https://arxiv.org/abs/2604.01949)

**Authors**: Ilan Gold, Felix Fischer, Lucas Arnoldt, F. Alexander Wolf, Fabian J. Theis  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.01949v1  

#### Abstract
The scale of biological datasets now routinely exceeds system memory, making data access rather than model computation the primary bottleneck in training machine-learning models. This bottleneck is particularly acute in biology, where widely used community data formats must support heterogeneous met...

---

### 30. [Crystalite: A Lightweight Transformer for Efficient Crystal Modeling](https://arxiv.org/abs/2604.02270)

**Authors**: Tin Had\v{z}i Veljkovi\'c, Joshua Rosenthal, Ivor Lon\v{c}ari\'c, Jan-Willem van de Meent  
**Category**: cs.LG  
**Published**: 2026-04-03  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.02270v1  

#### Abstract
Generative models for crystalline materials often rely on equivariant graph neural networks, which capture geometric structure well but are costly to train and slow to sample. We present Crystalite, a lightweight diffusion Transformer for crystal modeling built around two simple inductive biases. Th...

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
