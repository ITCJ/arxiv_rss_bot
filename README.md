# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-17 07:17:23 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Material-Agnostic Zero-Shot Thermal Inference for Metal Additive Manufacturing via a Parametric PINN Framework](https://arxiv.org/abs/2604.14562)

**Authors**: Hyeonsu Lee, Jihoon Jeong  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2604.14562v1  

#### Abstract
Accurate thermal modeling in metal additive manufacturing (AM) is essential for understanding the process-structure-performance relationship. While prior studies have explored generalization across unseen process conditions, they often require extensive datasets, costly retraining, or pre-training. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**金属增材制造**（Metal Additive Manufacturing, AM）中热场建模的**跨材料零样本泛化**（cross-material zero-shot generalization）难题。传统方法在面对新材料时通常需要重新训练或依赖大量标注数据，而现有 PINN 框架多为非参数化设计，难以适应不同材料的热物理特性变化。

具体挑战包括：
- 不同材料具有显著差异的热行为（如热导率、比热容等），导致温度场分布迥异。
- 缺乏针对任意新材料的真实温度标签，使得监督学习不可行。
- 参数化 PINN 在跨材料推理时易出现训练不稳定、收敛困难等问题。

---

### 提出的新方法与创新思路
作者提出了一种**解耦式参数化 PINN 框架**（decoupled parametric PINN framework），实现无需重训练、预训练或标签数据的**材料无关零样本热推断**（material-agnostic zero-shot thermal inference）。其三大核心创新如下：

#### （1）解耦式参数化网络架构（Decoupled Parametric PINN Architecture）
- 将**spatiotemporal coordinates**（空间-时间坐标）与**material properties**（材料属性）分别由两个独立子网络编码。
- 通过 **FiLM-based conditional modulation** 机制融合二者：材料属性作为条件变量，对时空特征进行缩放（scale）和偏移（shift）。
- 这种设计更符合物理规律——材料参数在控制方程中以乘法形式作用于导数项，而非简单的加法拼接。

> ✅ **优势**：相比传统的单体式（monolithic）参数化结构，该架构提升了表示能力，增强了物理一致性，并有效缓解了梯度干扰问题。

#### （2）基于物理引导的输出缩放（Physics-Guided Output Scaling）
- 引入由 **Rosenthal 解析解** 推导出的峰值温度估计 $ T_{\text{max}}(\lambda) $ 作为动态缩放因子。
- 构造输出变换：  
  $$
  T_{\text{phys}} = T_\infty + K \cdot T_{\text{max}}(\lambda) \cdot \text{Softplus}(T_e)
  $$
  其中 $ K \geq 1 $ 是补偿因子，用于修正 Rosenthal 模型在 AM 场景下的低估偏差。

> ✅ **优势**：显著降低不同材料间温度量级差异带来的梯度不平衡，提升训练稳定性，避免手动调参。

#### （3）混合优化策略（Hybrid Optimization Strategy）
- 结合 **Adam** 与 **L-BFGS** 优化器：
  - 前期使用 Adam 快速探索参数空间；
  - 后期切换至带随机小批量采样的 L-BFGS 实现精细收敛。
- 引入周期性**collocation point resampling**，防止因固定采样导致的曲率停滞。

> ✅ **优势**：大幅缩短训练周期，克服传统 PINN 需要数万轮迭代才能收敛的问题。

---

### 相比现有方法的优势
| 维度 | 本方法优势 |
|------|-----------|
| **泛化能力** | 支持零样本跨材料推理，无需 retraining 或 fine-tuning |
| **效率** | 训练速度极快，在仅 **4.4% 的 baseline epoch 数**内即超越基准模型 |
| **准确性** | 在 ID 和 OOD 材料上均取得更低误差，最高相对 L2 error 下降 **64.2%** |
| **可扩展性** | 框架模块化，组件可迁移至其他 PINN 应用 |

---

## 2. 核心实验方法和设置

### 数据集与仿真环境
- 使用一个公开的 **bare-plate laser powder bed fusion**（LPBF）数值基准。
- 温度真值由开源高保真 **FEM 工具 JAX-AM** 生成。
- 考虑四种典型合金：
  - **In-distribution**（ID）:
    - Ti-6Al-4V
    - Inconel 718
    - SS 316L
  - **Out-of-distribution**（OOD）:
    - AlSi10Mg（$ k=150 $ W/m·K）
    - Copper（$ k=401 $ W/m·K）→ 显著超出训练范围上限（50 W/m·K）

> ⚠️ 所有材料属性在训练阶段仅从定义区间 $ \mathcal{M} = [\rho, C_p, k] $ 中均匀采样，不包含测试材料。

---

### 实验设置
| 项目 | 设置说明 |
|------|--------|
| **空间域** | $ 40 \times 10 \times 6 $ mm³ |
| **激光参数** | 功率 500W，扫描速度 10 mm/s，光斑半径 1.5 mm |
| **时间范围** | 0–3 秒，步长 0.1 s |
| **Collocation Points** | 总计约 55.5 万个，采用多分辨率策略集中于激光路径附近 |
| **输入变量** | $(x, y, z, t, \rho, C_p, k)$ |
| **归一化** | 所有输入（含材料属性）归一化到 $[-1,1]$ |

---

### 评估指标
- **Relative L2 Error (%)**：
  $$
  \text{L2 error} = \frac{\sqrt{\sum_n (T_{\text{pred}}^{(n)} - T_{\text{FEM}}^{(n)})^2}}{\sqrt{\sum_n (T_{\text{FEM}}^{(n)})^2}} \times 100\%
  $$

---

### 基线方法对比
| 模型 | 类型 | 描述 |
|------|------|------|
| **N-PINN** | Non-parametric PINN | 固定材料参数，每种材料需单独训练 |
| **P-PINN** | Monolithic Parametric PINN | 将材料属性与时空坐标拼接输入单一网络 |

> 注：所有实验运行 **5 次随机种子** 取平均值 ± 标准差，确保可复现性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（ID 材料）

| 材料 | 方法 | #Params | L2 Error (%) |
|------|------|--------|-------------|
| Ti-6Al-4V | N-PINN | 11,341 | 6.19 ± 1.00 |
|             | P-PINN | 11,521 | 7.59 ± 3.86 |
|             | **Proposed** | **9,641** | **2.71 ± 0.34** |
| Inconel 718 | N-PINN | — | 6.09 ± 0.83 |
|              | P-PINN | — | 6.12 ± 3.22 |
|              | **Proposed** | — | **2.18 ± 0.45** |
| SS 316L     | N-PINN | — | 5.94 ± 0.63 |
|              | P-PINN | — | 5.84 ± 2.59 |
|              | **Proposed** | — | **2.17 ± 0.53** |

✅ **性能提升**：
- 相比 N-PINN，L2 error 平均下降 **56–64%**
- 参数量减少 **15%**，且支持一次训练、多材料通用

---

### 训练效率对比
- **N-PINN** 需训练 **50,000 epochs** 达到稳定性能。
- **本方法** 在 **2,200 epochs** 内即达到同等甚至更优精度 → 仅为 **4.4% 的训练成本**。
- 在 10,000 epochs 时进一步拉开差距，验证其快速收敛 + 更高最终精度。

---

### OOD 材料表现（极端热导率）

| 材料 | 方法 | L2 Error (%) |
|------|------|-------------|
| AlSi10Mg ($k=150$) | N-PINN | 4.25 ± 0.35 |
|                    | P-PINN | 3.25 ± 0.39 |
|                    | **Proposed** | **1.75 ± 0.28** |
| Copper ($k=401$)   | N-PINN | 1.63 ± 0.13 |
|                    | P-PINN | **14.38 ± 23.92** ← 失败！严重发散 |
|                    | **Proposed** | **0.69 ± 0.14** |

✅ **关键发现**：
- P-PINN 在铜这种超高导材料上完全失效，表明传统参数化结构无法处理强外推场景。
- 本方法仍保持 <1% 的误差，体现强大的 OOD 泛化能力。

---

### 消融实验结果

#### （1）输出缩放策略对比（Table 6）
| 缩放方式 | Ti-6Al-4V L2 Error |
|---------|------------------|
| 无约束输出 $ T_e $ | 77.81 ± 31.91 |
| $ T_\infty + \text{Softplus}(T_e) $ | 38.59 ± 0.00 |
| 手动设定 $ T_{\text{max}}=3000 $ | 5.66 ± 1.45 |
| 学习网络预测 $ T_{\text{max}} $ | 38.50 ± 0.09 |
| **物理引导 $ K \cdot T_{\text{max}}(\lambda) $** | **2.71 ± 0.34** ✅ |

> ➤ 物理引导缩放效果最佳，且方差最小，证明其稳定性和有效性。

#### （2）缩放因子 $ K $ 敏感性分析（Table 7 & Figure 14）
- $ K=1.0 $：低估峰值 → 补偿压力大 → 训练不稳定（误差高达 9.67%）
- $ K=1.5 $：最优平衡点 → 最低误差（2.71%）
- $ K=2.0 $：轻微过估 → 影响较小 → 仍优于所有基线

> ➤ 验证了“合理覆盖”优于“精确匹配”，适度过估反而更鲁棒。

#### （3）优化策略消融（Table 8 & 9）
| 优化方式 | Ti-6Al-4V L2 Error |
|----------|------------------|
| Adam-only | 26.49 ± 8.05 |
| Adam + Full-batch L-BFGS | 18.21 ± 18.26 |
| **Adam + Stochastic L-BFGS (proposed)** | **2.71 ± 0.34** ✅ |

> ➤ 随机化 mini-batch L-BFGS 显著改善收敛，适用于各类 PINN 架构（见 Table 9）。

---

## 4. 关键结论和发现

### 主要发现
1. **参数化 PINN 可实现真正的材料无关零样本推理**，无需任何 retraining。
2. **解耦架构 + FiLM 调制** 比传统拼接式更契合物理机制，提升表示能力和泛化性。
3. **物理引导的输出缩放** 是稳定训练的关键，尤其在跨材料场景下不可或缺。
4. **混合优化策略** 显著加速收敛，将训练耗时压缩至传统方法的 **<5%**。
5. 该框架不仅自身优越，其组件（如输出缩放、优化策略）也具有广泛适用性，可用于改进其他 PINN 方法。

---

### 方法局限性
1. **假设材料属性为常数**，未考虑温度依赖性（temperature-dependent properties），可能影响高温区精度。
2. 当前仅模拟纯热传导过程，未包含熔池内的 **thermo-fluid dynamics** 或相变效应。
3. 使用统一的 collocation sampling 策略，未能根据材料热扩散特性自适应调整采样密度。
4. 仅限单材料 AM，尚未拓展至 **multi-material AM** 场景。

---

### 未来工作方向
1. 开发 **material-aware collocation sampling** 策略，依据热导率自动调节采样分辨率。
2. 扩展框架以纳入更多工艺参数（如 laser power, scanning speed）作为可调参数。
3. 引入 **temperature-dependent material properties** 建模，提高真实场景适用性。
4. 探索在 **multi-material AM** 中的应用，支持界面热传递建模。
5. 结合在线传感数据，发展 **real-time adaptive PINN** 框架。

---

> 🔚 **总体评价**：本文提出的 parametric PINN 框架为金属 AM 中的热建模提供了一个高效、灵活、可扩展的解决方案，推动了 PINN 向实际工业部署迈进一大步。

</details>

---

### 2. [RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](https://arxiv.org/abs/2604.14885)

**Authors**: Zihong Zhang, Zuchao Li, Lefei Zhang, Ping Wang, Hai Zhao  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.14885v1  

#### Abstract
Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess-and-verify strategy, but existing training-free variants face trade-offs: retrieval-based drafts break when no exact match ex...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决的问题
大型语言模型（LLMs）采用**autoregressive decoding**逐词生成文本，导致推理延迟高、效率低。尽管**Speculative Decoding (SD)** 能通过“猜测-验证”策略加速推理，但现有方法存在以下瓶颈：
- **Retrieval-based 方法**（如 PLD、REST）依赖精确的 token 序列匹配，当上下文中无完全匹配时失效。
- **Logits-based 方法**（如 LogitSpec、Token Recycling）虽能利用模型内部预测信号，但缺乏结构性引导，推测范围有限。

### 🚀 提出的新方法：RACER
提出 **RACER**（Retrieval-Augmented Contextual Rapid Speculative Decoding），一种**轻量级、无需训练**的 speculative decoding 方法，其核心思想是：
> **将 retrieval 提供的“可靠锚点”与 logits 提供的“动态外推”相结合，实现更丰富、准确的 speculative draft 构建。**

#### 创新设计：
- **Logits Tree**：基于 `copy-logit` 策略构建，复用相同 token 上一次出现时的 logits 来推测后续 token，形成具有语义一致性的扩展路径。
- **Retrieval Tree with LRU Eviction**：基于 Aho-Corasick 自动机维护一个容量受限的 n-gram 检索树，并引入 **LRU（Least Recently Used）淘汰机制**，确保高频、近期模式优先保留。
- **融合策略（Integration Strategy）**：在固定 draft 容量下，优先填充 retrieval 提供的高置信度候选，剩余空间由 Logits Tree 补充，最终合并为统一的 draft tree 进行验证。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **无需训练** | 不依赖额外 draft model，节省训练成本与存储开销 |
| **轻量高效** | 仅需少量内存维护检索结构，适合边缘部署 |
| **鲁棒性强** | 即使 retrieval 失败，logits 仍可提供有效推测；反之亦然 |
| **通用性好** | 在多种模型架构（Vicuna、LLaMA、Qwen）、任务类型（代码、数学、对话）上均表现优异 |

---

## 2. **核心实验方法和设置**

### 📚 数据集
实验覆盖三大类基准任务，涵盖多语言与多领域场景：
- **Spec-Bench**：综合评测集，包含 Multi-turn Conversation (MT)、Translation (Trans)、Summarization (Sum)、QA、Math、RAG 等子任务。
- **HumanEval**：代码生成任务，评估函数实现能力。
- **MGSM-ZH**：中文版 GSM8K，用于评估中文数学推理能力。

此外还补充了英文推理任务：
- **GSM8K**, **AIME**, **MATH**（见附录 G）

### ⚙️ 实验设置
- **目标模型**：
  - Vicuna 系列（7B, 13B, 33B）
  - LLaMA3.1-8B
  - OpenPangu-7B（推理优化模型）
  - Qwen3 系列（8B, 14B, 32B）
- **解码方式**：greedy decoding，batch size = 1，最大输出长度 1024
- **硬件环境**：
  - 7B/8B 模型：RTX 4090 (24GB) + 20 CPU cores
  - ≥13B 模型：A800 (80GB) + 64 CPU cores

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **MAT (Mean Accepted Tokens)** | 每次 speculative 步骤平均接受的 token 数量，反映 draft 质量 |
| **Speedup Ratio** | 相较于标准 autoregressive 解码的速度提升倍数 |

### 🆚 基线方法对比
| 类别 | 方法 | 特点 |
|------|------|------|
| **Retrieval-based** | PLD, REST | 依赖历史或外部语料中的 exact match |
| **Logits-involved** | Token Recycling (TR), LogitSpec | 利用 past logits 构建 speculative draft |
| **Model-based (SOTA)** | EAGLE-3 | 需要额外训练的 draft model，性能强但开销大 |

---

## 3. **主要实验结果和性能指标**

### 📈 关键性能数据（来自 Table 1 & 2）
| 模型 | 方法 | 平均 MAT | 平均 Speedup |
|------|------|--------|-------------|
| Vicuna-7B | RACER | **3.27** | **2.42×** |
| Vicuna-13B | RACER | **3.23** | **2.50×** |
| Vicuna-33B | RACER | **3.09** | **2.52×** |
| LLaMA3.1-8B | RACER | **3.12** | **2.72×** |
| Qwen3-8B | RACER | **2.82** | **2.21×** |
| OpenPangu-7B | RACER | **2.63** | **2.12×** |

> ✅ 所有模型上，RACER 均取得**最高 speedup**，且 MAT 显著优于其他 model-free 方法。

### 🔁 与基线方法对比
| 对比维度 | 结果 |
|---------|------|
| vs. **PLD / REST** | RACER MAT 提升 40%-80%，speedup 超过 2×，远超 retrieval-only 方法上限 |
| vs. **TR / LogitSpec** | RACER 在所有任务上全面超越，尤其在 MGSM-ZH 上优势明显（+0.4~0.7 MAT） |
| vs. **EAGLE-3** | 尽管 EAGLE-3 在 MAT 上有时更高，但由于其需运行额外 draft model，**实际 end-to-end speedup 更低**。RACER 以更低计算代价实现更强加速效果 |

> 💡 示例：在 Vicuna-7B 上，RACER 达到 2.42× 加速，而 EAGLE-3 仅为 2.21×，说明 RACER 更高效。

### 🔍 消融实验结果（Ablation Study）

#### （1）组件移除实验（Table 3）
| 设置 | 平均 MAT ↓ | 平均 Speedup ↓ |
|------|----------|----------------|
| RACER（完整） | 3.27 | 2.42× |
| w/o logits | 1.95 | 1.35× |
| w/o retrieval | 2.75 | 1.99× |

> ❗ 移除 logits 导致性能严重下降，说明其是 speculation 的主干；
>  
> ✅ 移除 retrieval 也造成显著退化，尤其是在 MGSM-ZH 上（MAT↓0.7），表明 retrieval 对结构化推理至关重要。

#### （2）集成策略比较（Table 5）
| 策略 | MAT | Speedup |
|------|-----|---------|
| **Merge (RACER)** | **3.00** | **2.18×** |
| Half（各占一半预算） | 2.69 | 1.97× |
| Hard（fallback 切换） | 2.77 | 2.11× |

> ✅ “Merge” 策略最优，证明 retrieval 与 logits 应协同而非割裂使用。

#### （3）参数鲁棒性（Figure 8）
- RACER 在不同 **draft size**（16–64）、**node capacity**（2.5K–20K）、**n-gram depth** 和 **top-k breadth** 下性能稳定。
- 最佳配置集中在：`n-gram depth ≈ 9–11`, `top-k breadth ≈ 8–10`，符合 copy-logit 第85百分位排名为9的经验规律。

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **Retrieval 与 logits 具有互补性**：
   - Retrieval 提供**结构锚点**，适用于重复模式丰富的任务（如代码、数学推理）；
   - Logits 提供**动态外推能力**，支撑长距离、非模板化生成。
2. **RACER 实现了两者的有机统一**：
   - 通过 LRU 管理的 AC automaton 实现高效、自适应的 retrieval；
   - 通过 copy-logit + 层级剪枝构建高质量 Logits Tree；
   - 二者融合后显著提升 MAT 与 speedup。
3. **无需训练也能达到甚至超越 SOTA model-based 方法的实际加速效果**：
   - RACER 在多数情况下优于 EAGLE-3 的 end-to-end 推理速度，尽管后者 MAT 更高。

### ⚠️ 局限性（Limitations）
- 当前仅验证于**纯文本任务**，未测试在多模态（vision/audio）场景下的适用性。
- 对于极长序列（>8k），检索结构可能面临更新延迟问题（failure link 懒重建）。
- 中文等非拉丁语系的表现依赖于 tokenizer 与上下文对齐质量。

### 🔮 未来工作方向
- 扩展至 **multimodal speculative decoding**，整合视觉/音频 token 的检索与 logits 预测。
- 引入 **adaptive depth control**，根据上下文复杂度动态调整 draft tree 深度。
- 探索与 **distributed / parallel decoding** 架构的结合，进一步释放吞吐潜力。
- 研究 **cross-lingual retrieval cues**，增强多语言场景下的泛化能力。

---

## ✅ 总结一句话
> **RACER 通过将 retrieval 的“结构稳定性”与 logits 的“语义外推力”有机结合，在无需任何训练的前提下，实现了轻量、通用且高效的 speculative decoding，显著提升了 LLM 的推理速度（普遍 >2×），并优于现有 model-free 与部分 model-based 方法。**

GitHub 开源地址：[https://github.com/hkr04/RACER](https://github.com/hkr04/RACER)

</details>

---

### 3. [Modular Continual Learning via Zero-Leakage Reconstruction Routing and Autonomous Task Discovery](https://arxiv.org/abs/2604.14375)

**Authors**: Noureddine Kermiche  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.14375v1  

#### Abstract
Catastrophic forgetting remains a primary hurdle in sequential task learning for artificial neural networks. We propose a silicon-native modular architecture that achieves structural parameter isolation using Task-Specific Experts and a distributed, outlier-based Gatekeeper. Moving beyond traditiona...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文旨在解决**持续学习**（Continual Learning, CL）中的三大核心挑战：
- **灾难性遗忘**（Catastrophic Forgetting, CF）：模型在学习新任务时会覆盖旧任务的知识。
- **隐私泄露风险**：传统方法如 Experience Replay 需要存储历史数据，违反 GDPR 等“被遗忘权”法规。
- **路由机制失效**：全局分类器或共享骨干网络在长期序列中易受干扰，导致错误路由。

此外，作者指出当前主流的“单体AI”（monolithic AI）范式已难以应对工业级连续适应需求，提出向模块化、去中心化的“后单体时代”（post-monolith era）演进。

---

### 提出的新方法与思路
论文提出了一个名为 **Modular Brain Architecture** 的硅原生模块化架构，其核心思想是通过**结构参数隔离**实现稳定且可扩展的持续学习。主要创新包括：

#### （1）Simultaneous Pipeline（同步管道）
- 在原始数据流存在的局部训练阶段，并行执行三项操作：
  - **Teacher 学习**：高容量教师网络快速吸收新任务知识。
  - **Student 蒸馏**：紧凑的学生专家从教师处一次性提取“Dark Knowledge”。
  - **Router 流形获取**：基于重建误差的路由器学习当前任务的数据流形特征。
- 数据仅在本地缓存用于收敛，在任务边界确认后立即永久删除，满足 **Zero-Leakage** 合规要求。

#### （2）Tight-Bottleneck Autoencoder (TB-AE)
- 针对标准 VAE 在高维稀疏 LLM 嵌入空间中出现的 **posterior collapse** 问题，提出一种确定性、极窄瓶颈的自动编码器。
- 通过强制结构压缩（如将 4096-D 嵌入压缩至 k=12 维），有效区分语义密集的潜在流形，提供鲁棒的无监督新颖性信号。

#### （3）Decentralized Reconstruction Routing（去中心化重建路由）
- 每个任务拥有独立的 **reconstruction-based router**，作为异常检测器判断输入是否属于其已知流形。
- 路由决策完全本地化，无需联合训练全局分类器，从根本上避免了路由层的灾难性遗忘。

#### （4）Autonomous Task Discovery（自主任务发现）
- 利用 **Familiarity Probe** 和动态阈值机制自动识别新任务的到来。
- 结合 **Minimum Viable Manifold (MVM)** 条件控制模块生成节奏，防止“专家爆炸”。

#### （5）Semi-Frozen Backbone 设计
- 冻结底层预训练骨干网络（F），确保中间隐变量 $ h = F(x) $ 不漂移，为路由器提供稳定的拓扑签名。
- 上层仍保持可塑性，作为 Persistent Teacher 实现前向迁移。

---

### 相比现有方法的优势

| 方法类别 | 典型代表 | 主要缺陷 | 本文优势 |
|--------|--------|--------|--------|
| 正则化法 | EWC, SI | 容量饱和、语义模糊 | 参数物理隔离，无容量上限 |
| 回放法 | Experience Replay | 违反 Zero-Leakage | 数据即时清除，合规性强 |
| 生成回放 | Generative Replay | 模式崩溃、噪声累积 | 放弃生成，直接实时蒸馏 |
| 知识蒸馏 | LwF | 共享表示漂移、软标签失真 | 单次 Live Distillation + 固定流形 |
| MoE 架构 | Standard MoE | 全局门控易遗忘 | 分布式、独立的 outlier-based 路由 |
| 自编码器路由 | TAMiL, CLARE | 后验坍缩、依赖回放缓冲区 | 引入 TB-AE + Simultaneous Pipeline |

> ✅ **综合优势**：实现了 **0.0% backward interference**、**zero-leakage compliance**、**autonomous routing without task ID**，适用于企业级隐私敏感场景。

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **计算机视觉领域**：
   - **Split-MNIST**：将 MNIST 分为两个任务（0–4 vs 5–9），用于基准比较。
2. **自然语言处理领域**：
   - **Synthetic "Crowded Manifold" Dataset**：模拟现代 4096-D LLaMA-3 嵌入空间特性。
     - 任务中心间距仅为 ±0.15 单位（高度重叠）
     - 内在流形维度 d=12
     - 用于测试高维空间下的流形分离能力。

---

### 实验设置与评估指标

#### 评估维度
| 指标 | 描述 |
|------|------|
| **Task Retention** | 旧任务准确率保留程度，衡量稳定性 |
| **Routing Accuracy** | 自主路由正确率，反映任务识别能力 |
| **Discrimination Ratio** | 已知 vs 未知任务的 MSE 对比，体现新颖性检测强度 |
| **Signal-to-Noise Ratio (SNR)** | 返回任务时的重建误差对比，验证长期记忆一致性 |
| **End-to-End Accuracy** | 整体系统在混合流上的表现（路由 + 推理） |

#### 基线方法对比
- Naive Sequential Training
- Learning without Forgetting (LwF)
- Elastic Weight Consolidation (EWC)
- Experience Replay (ER)

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 表1：Split-MNIST 上的任务保留能力（Task A 在 Task B 后的表现）

| 方法 | Task A Retention | CF 风险 | 泄露风险 |
|------|------------------|---------|----------|
| Naive | 19.40% | 严重 | 低 |
| LwF | 79.80% | 中等 | 低 |
| EWC | 84.00% | 中等 | 低 |
| Experience Replay | 95.10% | 低 | **高（违反 Zero-Leakage）** |
| **Ours (Simultaneous)** | **99.42%** | **零（隔离）** | **高（立即清除）** |

> 🔍 **分析**：本方法不仅超越所有非回放方法，甚至优于标准回放策略，同时实现 **zero-leakage**。

---

#### 表2：Live Distillation 的正则化效应

| 模块 | 准确率 |
|------|-------|
| Persistent Teacher | 99.10% |
| Frozen Student Expert | 99.42% |
| **Fidelity Gap** | **-0.31%** |

> 📌 **发现**：“Live Distillation” 起到了**自然正则化器**的作用，学生模型精度略高于教师，表明其成功提炼出更泛化的语义规则。

---

#### 表3：TB-AE 在 4096-D 空间中的消融实验（不同瓶颈维度 k）

| Bottleneck (k) | MSE Task A | MSE Task B | Discrimination Ratio |
|---------------|------------|------------|------------------------|
| k=4 (Narrow) | 0.0674 | 0.2476 | 3.67x |
| **k=12 (Tight)** | **0.0010** | **0.2109** | **203.78x** ✅ |
| k=32 (Relaxed) | 0.0012 | 0.2104 | 174.23x |
| k=64 (Wide) | 0.0011 | 0.2104 | 176.47x |

> 🔬 **结论**：当瓶颈维度接近任务内在维度（d=12）时，TB-AE 达到最优判别能力（>200x 区分度），过大或过小均降低效果。

---

#### 表4：自主任务检索信噪比（SNR）

| 输入流形 | 路由器 | 重建 MSE | 动作 |
|----------|--------|-----------|------|
| Task A（返回） | Task A Router | **0.0014** | RECOGNIZED → 路由至 Expert A |
| Task B（新） | Task A Router | **0.2105** | NOVELTY → 触发 Expert B 生成 |

> 🧠 **SNR = 145.34x**：系统能以极高置信度识别返回任务，避免冗余实例化。

---

#### 其他重要结果
- **端到端盲流准确率**：结合 99.42% 专家保留 + 96.10% 自主路由 → **约 95.54%**
- **前向迁移加速**：Persistent Teacher 提供 warm-start，使 Task B 训练提速 **16.2%**

---

## 4. 关键结论和发现

### 主要发现
1. **Simultaneous Pipeline 可实现“即时巩固”**：
   - 并行完成 Teacher 学习、Student 蒸馏、Router 获取，支持数据单遍扫描即删，符合 GDPR。
2. **TB-AE 是高维 LLM 嵌入空间下唯一有效的无监督新颖性探测器**：
   - 确定性结构瓶颈优于 B-VAE 等概率正则化方法，解决了 posterior collapse。
3. **去中心化 outlier detection 比全局分类更稳健**：
   - 每个 router 只需学会“我是谁”，无需知道“别人是谁”，彻底免疫灾难性遗忘。
4. **Live Distillation 是强正则化手段**：
   - 学生模型因必须压缩知识而被迫丢弃噪声，反而提升泛化性能（负保真度差距）。
5. **semi-frozen backbone 是必要设计**：
   - 类似生物大脑的进化硬编码，冻结基础感知层为高层认知功能提供不变参考系。

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **不适用于 Class-Incremental Learning** | 当前框架基于协变量偏移（covariate shift）检测任务边界，无法处理同一输入分布内的标签增量变化。 |
| **Forward Transfer 有限于 Teacher 上层** | 学生专家之间无结构共享，仅靠上层 Teacher 实现前向迁移，受限于任务相似性。 |
| **O(N) 路由复杂度** | 随着任务数量增长，需遍历所有 routers，计算开销线性上升。 |
| **False-Positive Familiarity 风险** | 高维空间中已有 manifold 并集可能覆盖新输入，造成误识别。 |
| **Block-Sequential 数据假设** | 要求连续批次构成完整任务；高度交错流会导致 provisional expert 无法收敛。 |
| **Zero-Leakage ≠ 绝对安全** | 虽消除回放缓冲区，但蒸馏权重仍可能遭受 Model Inversion Attack，需配合 DP-SGD 等防御机制。 |

---

### 未来工作方向
1. **Hierarchical Routing**：
   - 引入 K-Means 或 Taxonomic Clustering 构建层级路由树，降低搜索复杂度与碰撞概率。
2. **Dynamic Bottleneck Tuning**：
   - 将瓶颈维度 $ k $ 设为可调超参，根据任务复杂度自适应调整。
3. **Interleaved Stream Support**：
   - 开发解耦的累计缓存机制，支持更复杂的任务交错模式。
4. **Class-Incremental 扩展**：
   - 探索结合 contrastive learning 或 prompt tuning 技术，支持细粒度标签扩展。
5. **硬件部署优化**：
   - 针对边缘设备优化 LoRA + TB-AE 的推理效率，推动实际落地。

---

> ✅ **总体评价**：  
> 本文提出的 **Modular Continual Learning via Zero-Leakage Reconstruction Routing** 是一次面向工业级应用的重大架构革新。它不仅在技术上突破了传统 CL 方法的算法瓶颈，更在法律合规性（GDPR）、计算效率（parallel pipeline）、生物学合理性（CLS theory）等多个维度建立了新的标准，为构建可持续、可信赖的企业级 AI 系统提供了坚实基础。

</details>

---

### 4. [From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning](https://arxiv.org/abs/2604.15244)

**Authors**: Kiran Purohit, Ramasuri Narayanam, Soumyabrata Pal  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.15244v1  

#### Abstract
Speculative decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose outputs that a stronger target model verifies. However, its token-centric nature allows erroneous steps to propagate. Prior approaches mitigate this using external reward models, but...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **Speculative Decoding (SD)** 虽然能加速大语言模型（LLM）推理，但其**token-centric**（以词元为中心）的设计在多步推理任务中存在严重缺陷：
- 错误的中间步骤一旦被接受，会**持续传播**（error propagation），导致最终答案错误。
- 依赖外部 **Process Reward Models (PRMs)** 的改进方法（如 RSD）虽然提升了可靠性，但引入了额外的延迟、计算开销，并且泛化能力差。

### 🚀 提出的新方法：SPECGUARD
作者提出 **SPECGUARD** —— 一种**验证感知的推测解码框架**，实现高效且可靠的多步推理。其核心思想是：
> 将验证从“token级”提升到“step级”，并仅使用**模型内部信号**进行轻量级验证，避免依赖外部模型。

#### 主要创新点：
1. **Step-Level Verification（步骤级验证）**
   - 不再逐个 token 验证，而是对整个推理步骤（reasoning step）进行整体评估，更符合逻辑推理的本质。

2. **双信号集成验证器（Ensemble Verifier）**
   - **Attention-Based Grounding Verification (ABGV)**：通过注意力机制衡量生成步骤是否充分“扎根”于输入或已验证的前序步骤，防止“无根据”的幻觉。
   - **Log-Probability-Based Verification (LPBV)**：基于对数概率评估模型在 token 级别的置信度。
   - 两者结合形成一个**联合判别标准**：只有同时满足高置信度和强归因的步骤才会被接受。

3. **自一致性选择器（Self-Consistency Selector）**
   - 在每一步，让 draft model 生成多个候选步骤（k 个）。
   - 使用 Sentence Transformer 编码这些候选，选择与其他候选语义最一致的那个作为待验证步骤，增强鲁棒性。

4. **无需外部奖励模型**
   - 完全摆脱对 PRM 的依赖，降低延迟和部署复杂度，提升通用性。

### 🔍 相比现有方法的优势
| 方法 | 是否依赖 PRM | 验证粒度 | 效率 | 准确性 | 通用性 |
|------|--------------|----------|------|--------|--------|
| SD | ❌ | Token | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| RSD | ✅ | Step | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **SPECGUARD** | ❌ | **Step** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 2. 核心实验方法和设置

### 📚 数据集
实验在四个具有挑战性的多步推理基准上进行：
- **MATH500**：竞赛级数学题，涵盖代数、几何等。
- **GSM8K**：小学数学应用题，测试数值推理。
- **Gaokao-2023-En**：中国高考英语版数学题，风格正式、路径复杂。
- **OlympiadBench**：国际奥赛级别的科学问题，极具创造性与难度。

### 📊 评估指标
- **Exact Match (EM)**：预测答案与标准答案完全匹配的比例。

### ⚙️ 实验设置
- **模型**：
  - 主要使用 `Qwen2.5-Math-Instruct`（7B target / 1.5B draft）
  - 对比也使用 `Llama-3` 系列（8B / 1B）
- **推理设置**：
  - 每个推理步骤以 `\n\n` 分隔。
  - 温度 `temperature=0.7`，`top_p=0.8`，采样数量 `n=16`。
  - 阈值 `T=0.7`，权重 `β=0.3`（偏向 grounding 信号）。
- **硬件**：NVIDIA A100 GPU，使用 vLLM 作为推理后端。

### 🆚 基线方法
1. **Target-only / Draft-only**：单独使用目标或草稿模型。
2. **Best-of-N (BoN)**：从多个 draft 中选得分最高者（使用 PRM 打分）。
3. **Speculative Decoding (SD)**：标准 token-level 推测解码。
4. **Reward-guided SD (RSD)**：使用 PRM 进行 step-level 验证。
5. **Beam Search / Process Best-of-N**：搜索类方法。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 1 & 2）

| 方法 | MATH500 | GSM8K | Gaokao | Olympiad |
|------|---------|-------|--------|----------|
| Target Model (7B) | 83.0 | 94.7 | 66.8 | 40.6 |
| SD | 82.4 | 94.2 | 66.3 | 39.4 |
| RSD | 82.4 | 94.4 | 68.5 | 39.6 |
| **SPECGUARD** | **85.4** | **95.8** | **69.4** | **41.2** |

> ✅ **SPECGUARD 在所有基准上均超越 SOTA 方法**，平均**准确率提升达 3.6%**。

### ⏱️ 效率表现
- **延迟降低约 11%** 相比 RSD。
- 例如在 GSM8K 上：
  - SPECGUARD：**95.8% 准确率，耗时 34 分钟**
  - RSD Majority：**88.7% 准确率，耗时超 41 分钟**

### 🔬 消融实验结果（Ablation Studies）

#### （1）验证信号的有效性（Table 1）
- **SC + LPBV**（仅有置信度）：准确率低于完整 SPECGUARD → 表明**仅靠置信度无法过滤“合理但无依据”的错误步骤**。
- **SPECGUARD** 加入 ABGV 后显著提升 → 证明**归因验证至关重要**。

#### （2）层数选择（Figure 3 & 4）
- 使用**最后 3 层**注意力即可达到接近使用全部层的效果，但**延迟更低**。
- 使用前 3 层效果最差，说明深层注意力更适合 grounding 判断。

#### （3）注意力稀疏化
- 将注意力头中小于 0.01 的值置零，可**提升效率且不损失精度**，甚至略有增益。

#### （4）超参数敏感性（Appendix A.2.2）
- 方法对 `β` 和 `T` 具有较强鲁棒性。
- 最优配置：`β=0.3`, `T=0.7`

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Step-level verification 是多步推理的关键**：相比 token-level，step-level 更能捕捉逻辑连贯性。
2. **模型内部信号足以支撑可靠验证**：ABGV 和 LPBV 的组合可在不依赖 PRM 的情况下实现更强的错误检测能力。
3. **SPECGUARD 实现了准确性与效率的双重提升**：
   - **准确率 +3.6%**
   - **延迟 -11%**
   - 显著优于 SD 和 RSD。
4. **自一致性 + 多样本采样有效抑制噪声**：相比 greedy 生成，多候选选择大幅提升稳定性。

### ⚠️ 方法的局限性
1. **未覆盖开放域生成任务**：当前评估集中于结构化推理，未测试长文本生成、对话等场景。
2. **单实例推理优化**：未考虑大规模 batch 推理或硬件级优化，实际生产部署需进一步工程适配。
3. **仍可能产生幻觉**：尽管减少错误传播，但不能完全消除 LLM 固有的幻觉风险，需配合人工审核。

### 🔮 未来工作方向
1. 引入更多模型内部信号，如 **entropy-based measures** 或 **uncertainty calibration**，进一步提升验证器可靠性。
2. 将 SPECGUARD 扩展至 **long-form generation** 和 **dialogue systems**。
3. 探索与 **test-time compute scaling**（如 Best-of-N, Tree Search）的结合，实现更强大的推理系统。

---

> **总结**：  
> **SPECGUARD** 成功将 speculative decoding 从“token时代”推进到“step时代”，通过**轻量级、内生式的双信号验证机制**，在不牺牲效率的前提下显著提升了多步推理的可靠性，为高效、可信的 LLM 推理提供了新的范式。

</details>

---

### 5. [Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter](https://arxiv.org/abs/2604.15039)

**Authors**: Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu, Weimin Zheng, Mingxing Zhang  
**Category**: cs.DC  
**Published**: 2026-04-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.15039v1  

#### Abstract
Prefill-decode (PD) disaggregation has become the standard architecture for large-scale LLM serving, but in practice its deployment boundary is still determined by KVCache transfer. In conventional dense-attention models, prefill generates huge KVCache traffics that keep prefill and decode tightly c...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大规模 LLM 推理服务普遍采用 **Prefill-Decode (PD) disaggregation** 架构，将计算密集型的 *prefill* 阶段与内存带宽密集型的 *decode* 阶段分离。然而，这种架构在实践中仍受限于 **KVCache 跨节点传输开销**，导致 prefill 和 decode 必须部署在同一高带宽、低延迟的 RDMA 网络域内（如单个数据中心），难以实现真正的异构化部署和资源弹性扩展。

此外，传统 dense-attention 模型生成的 KVCache 大小随上下文长度线性增长，在长上下文场景下会带来巨大的网络压力，使得跨数据中心部署几乎不可行。

### 🚀 提出的新方法：**Prefill-as-a-Service (PrfaaS)**
论文提出了一种新的跨数据中心 LLM 服务架构 —— **Prefill-as-a-Service (PrfaaS)**，其核心思想是：

- 利用新一代 **hybrid-attention 模型** 显著降低 KVCache 大小；
- 将 **长上下文且未命中 prefix cache 的 prefill 请求** 动态卸载到远程专用的、计算密集型的 **PrfaaS 集群**；
- 通过 **普通以太网（commodity Ethernet）** 将生成的 KVCache 传回本地 PD 集群进行 decode；
- 不对所有请求都卸载，而是采用 **选择性卸载策略（selective offloading）**，避免短请求造成带宽浪费。

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | PrfaaS |
|------|--------|-------|
| **部署模式** | 单数据中心内 tight-coupled 部署，依赖 RDMA | 支持跨数据中心 loose-coupled 部署，仅需普通以太网 |
| **硬件灵活性** | 异构硬件必须共处同一 RDMA fabric | 可独立扩展 prefill 和 decode 容量，支持不同芯片类型分布于不同集群 |
| **资源利用率** | 固定 prefill/decode 硬件比例，易出现负载不均 | 动态调度，按需分配，提升整体吞吐 |
| **系统设计** | 仅依赖模型改进 | 结合模型侧（KVCache 减少）与系统侧优化（选择性卸载、带宽感知调度、缓存感知路由） |

> 💡 **核心创新点总结**：
> - 提出 **“跨数据中心 KVCache”** 新范式，突破传统 PD 架构的物理边界；
> - 设计 **PrfaaS 架构**，实现计算密集型 prefill 的云服务化；
> - 引入 **多层级调度机制（dual-timescale scheduling）**，兼顾短期拥塞控制与长期资源再平衡。

---

## 2. 核心实验方法和设置

### 📊 数据集与工作负载
- 使用内部真实业务流量建模的工作负载：
  - 输入长度服从截断对数正态分布：`μ=9.90, σ=1.00`，范围 `[128, 128K]`，平均约 **27K tokens**；
  - 输出长度固定为 **1024 tokens**；
  - 包含大量增量 prefill（incremental prefill）和 prefix cache hit 场景，符合 agent 类应用趋势。

### ⚙️ 实验设置
- **模型**：内部自研 **1T 参数 hybrid-attention 模型**，结构基于 Kimi Linear [26]，采用 KDA:MLA 层比 3:1；
- **硬件配置**：
  - **PrfaaS 集群**：32 × H200 GPU（高算力，专用于长上下文 prefill）；
  - **本地 PD 集群**：64 × H20 GPU，支持 PD disaggregation，RDMA 内联（800 Gbps/node）；
  - **基线对比**：
    - **Homogeneous PD**：96 × H20 GPU 组成的传统同构 PD 集群；
    - **Naive Heterogeneous PD**：所有 prefill 交给 H200，所有 decode 交给 H20，无智能调度；
- **网络连接**：两个集群间通过 VPC peering 连接，提供 **100 Gbps 聚合带宽** 的跨数据中心链路。

### 📈 评估指标
| 指标 | 含义 |
|------|------|
| `Amax` | 最大稳态吞吐量（requests/sec） |
| `TTFT`（Time to First Token） | 首 token 延迟，特别是 P90 TTFT |
| `KV Throughput (Pkv)` | KVCache 生成速率（Gbps） |
| `Bout` | PrfaaS 集群出口带宽消耗 |
| `Throughput Gain` | 相对于基线的吞吐提升倍数 |

### 🔁 基线方法对比
1. **Homogeneous PD**：纯同构部署，作为性能基准；
2. **Naive Heterogeneous PD**：简单地将 prefill 分配给高性能芯片（H200），decode 分配给内存带宽强芯片（H20），但缺乏调度优化；
3. **PrfaaS-PD（本文方法）**：结合选择性卸载、带宽感知调度与缓存感知路由。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 6）

| 指标 | PrfaaS-PD | Homogeneous PD | Naive Heterogeneous PD |
|------|-----------|----------------|-------------------------|
| **最大吞吐 `Amax` (req/s)** | **3.24** | 2.11 | 2.45 |
| **相对吞吐增益** | **1.54×** | 1.00× | 1.16× |
| **Mean TTFT (s)** | 2.22 | 4.44 | 1.74 |
| **P90 TTFT (s)** | **3.51** | 9.73 | 3.51 |
| **PrfaaS 出口带宽 `Bout`** | **13 Gbps** | — | — |

### 🔍 对比分析
- **相比 Homogeneous PD**：
  - 吞吐提升 **54%**（3.24 vs 2.11 req/s）；
  - P90 TTFT 降低 **64%**（3.51s vs 9.73s），显著改善用户体验；
  - 原因：长请求被卸载至高算力 PrfaaS 集群，缓解本地 prefill 拥塞。

- **相比 Naive Heterogeneous PD**：
  - 吞吐高出 **32%**（3.24 vs 2.45 req/s）；
  - 尽管 naive 方案也有异构优势，但由于缺乏调度优化，导致 prefill/decode 阶段严重失衡；
  - PrfaaS 通过动态调整 `routing threshold t=19.4K` 和实例配比 `(Np=3, Nd=5)` 实现最优负载均衡。

- **带宽效率极高**：
  - 平均仅消耗 **13 Gbps** 跨数据中心带宽（占总链路 13%），远低于 100 Gbps 上限；
  - 表明即使在 commodity Ethernet 下，**cross-datacenter KVCache transfer 是完全可行的**。

### 🔍 消融实验与关键参数分析（Figure 5）
- 通过网格搜索确定最优 `threshold t` 和 `Np/Nd` 比例：
  - 当 `t ≈ 19.4K` 时，PrfaaS 与 PD-P 吞吐达到平衡；
  - 当 `Np=3, Nd=5` 时，decode 阶段不再成为瓶颈；
- 结果显示：**过低的阈值会导致过多短请求涌入 PrfaaS，引发带宽瓶颈；过高则无法充分利用加速能力**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **KVCache-efficient 模型是前提，但不足以支撑跨数据中心部署**  
   hybrid-attention 架构虽能将 KV throughput 降低 **10–36×**，但仍需系统级优化才能避免拥塞。

2. **Selective Offloading 是关键**  
   并非所有 prefill 都应卸载。只有 **长上下文、未命中 prefix cache 的请求** 才值得付出传输代价。

3. **跨数据中心 PD disaggregation 成为现实可能**  
   在合理调度下，**commodity Ethernet 已足以承载 hybrid 模型的 KVCache 传输**，无需昂贵 RDMA fabric。

4. **PrfaaS 实现了更高的资源利用率与成本效益**  
   - 允许 prefill 与 decode 独立扩缩容；
   - 支持利用远程闲置或低成本计算资源；
   - 提升整体集群吞吐达 **54%**。

### ⚠️ 方法的局限性
- 当前实验基于模拟与建模，尚未在超大规模生产环境中验证；
- 依赖稳定的跨数据中心网络质量，若链路抖动剧烈可能影响调度稳定性；
- 对于极短请求（<1K tokens），本地处理仍是最优路径；
- 当前调度假设集群间状态可全局可见，实际中可能存在元数据同步延迟。

### 🔮 未来工作方向
1. **更细粒度的 KVCache 分片与并行传输机制**；
2. **结合 KVCache 压缩技术（如 KIVI、H2O）进一步降低带宽需求**；
3. **支持多租户场景下的 SLA-aware 调度与资源隔离**；
4. **探索 PrfaaS 作为公共云服务的可能性（类似 Function-as-a-Service）**；
5. **与 speculative decoding、distillation 等推理加速技术融合优化**。

---

> 📌 **一句话总结**：  
> 本文提出的 **PrfaaS** 架构通过 **“模型轻量化 + 系统智能化” 协同设计**，首次证明了 **跨数据中心的 KVCache 传输在现实中是高效且可行的**，为下一代 LLM 推理基础设施提供了全新的可扩展范式。

</details>

---

### 6. [CoCoDiff: Optimizing Collective Communications for Distributed Diffusion Transformer Inference Under Ulysses Sequence Parallelism](https://arxiv.org/abs/2604.14561)

**Authors**: Bin Ma, Xingjian Ding, Tekin Bicer, Pengfei Su, Dong Li  
**Category**: cs.DC  
**Published**: 2026-04-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.14561v1  

#### Abstract
Diffusion Transformers (DiTs) are increasingly adopted in scientific computing, yet growing model sizes and resolutions make distributed multi-GPU inference essential. Ulysses sequence parallelism scales DiT inference but introduces frequent all-to-all collectives that dominate latency. Overlapping ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**CoCoDiff: Optimizing Collective Communications for Distributed Diffusion Transformer Inference Under Ulysses Sequence Parallelism**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **分布式 DiT 推理中的通信瓶颈**：在使用 **Ulysses Sequence Parallelism** 进行大规模 Diffusion Transformer (DiT) 推理时，频繁的 **all-to-all collective communication** 成为性能瓶颈，尤其在高分辨率图像生成中，其开销可占总推理时间的 **80%以上**。
- **挑战来源**：
  - 数据依赖紧密，难以重叠通信与计算；
  - 通信量大且每步重复，无法有效摊销；
  - Aurora 超算等异构拓扑中存在带宽不对称（如 intra-GPU vs. inter-GPU）。

### 🚀 提出的新方法：CoCoDiff
提出了一种面向 DiT 分布式推理的高性能引擎 **CoCoDiff**，基于两个关键观察设计了三项核心技术：

#### 创新机制：
1. **Tile-Aware Parallel All-to-All (TAPA)**  
   - 将传统的扁平 all-to-all 分解为两个阶段：
     - **Phase 1**: 在 GPU 内部 tile 之间进行高速（185 GB/s）并行交换；
     - **Phase 2**: 在跨 GPU 的 Xe Link 上以双环结构并行执行 6-rank all-to-all。
   - 充分利用 Aurora 的层次化通信拓扑，提升聚合带宽利用率。

2. **V-First Scheduling**  
   - 利用 **QKV 处理不对称性（QKV Processing Asymmetry）**：
     - Q 和 K 需要额外的 RMSNorm 和 RoPE 操作，而 V 只需线性投影；
     - 因此 V 可提前完成，其通信可在 Q/K 计算期间被隐藏。
   - 实现 V 的 Phase 1 通信与 Q/K 的 Norm+RoPE 计算重叠，减少关键路径延迟。

3. **V-Major Selective Communication**  
   - 利用 **时间冗余（Temporal Redundancy）**：相邻去噪步骤产生的中间张量高度相似。
   - 引入选择性通信机制：
     - 基于前一步缓存的 `V` 向量计算变化度（L1 距离），识别“活跃”token；
     - 仅传输发生变化的部分（active projections），其余复用缓存；
     - 使用 **time-varying cache schedule** 动态调整缓存比例（从 0 到 1 线性增长），平衡精度与速度。

### 🔍 相比现有方法的优势
| 方面 | CoCoDiff 优势 |
|------|----------------|
| **通信效率** | 显著降低 all-to-all 开销，尤其是在多节点场景下；TAPA 比 oneCCL 快达 3.9× |
| **计算-通信重叠** | V-First 实现有效隐藏通信，突破传统阻塞式 all-to-all 限制 |
| **通信语义感知** | 不再将 all-to-all 视为黑盒，而是结合模型特性优化通信内容 |
| **硬件适配性** | 针对 Intel Aurora 架构定制，最大化利用 tile-level 并行性和带宽差异 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用来自 **Advanced Photon Source (APS)** 的同步辐射 X 射线显微断层扫描数据（synchrotron X-ray microtomography）
- 对象：小鼠脑组织切片（mouse brain tissue）
- 任务：科学图像修复（scientific inpainting），测试 block 和 center 两种掩码配置

### ⚙️ 实验设置
- **硬件平台**：Intel Aurora 超级计算机
  - 单节点：6 个 Intel GPU Max 1550（Ponte Vecchio），共 12 个 tile（每个 GPU 2 个 tile）
  - 多节点：扩展至 2、4、8 节点（最多 96 tiles）
  - 通信拓扑：intra-GPU（185 GB/s）、inter-GPU Xe Link（15 GB/s）、跨节点 Slingshot（25 GB/s）

- **软件栈**：
  - PyTorch 2.10.0 + Intel IPEX + oneCCL 2021.17
  - 基于 xDiT 和 Diffusers 库实现，约 20K 行代码扩展

- **模型与参数**：
  - 四个 DiT 模型：
    - FLUX.1-dev (12B)
    - Qwen-Image (20B)
    - Stable Diffusion 3.5 (SD3.5, 8B)
    - FLUX.2-dev (32B)
  - 分辨率：768×768、1536×1536、2304×2304
  - Batch size：1–32（根据内存动态调整）

- **评估指标**：
  - **端到端加速比（end-to-end speedup）**
  - 图像质量指标：
    - PSNR（越高越好）
    - SSIM（越高越好）
    - MSE（越低越好）
  - 缓存策略下的 **speedup-quality trade-off**

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **Flat** | 标准 Ulysses all-to-all，无任何优化 |
| **TAPA** | 仅启用 Tile-Aware Parallel All-to-All |
| **CoCoDiff** | 完整方案：TAPA + V-First + V-Major Selective Communication |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据
- **平均加速比**：相比 Flat 基线，**CoCoDiff 平均提速 3.6×**
- **峰值加速比**：在 SD3.5 模型、2304×2304 分辨率、4 节点环境下达到 **8.4× 加速**

#### 多节点 vs. 单节点表现
| 场景 | CoCoDiff 加速比 | TAPA 加速比 |
|------|------------------|-------------|
| 单节点（12 ranks） | ~2.5× | ~2.3× |
| 多节点（24–96 ranks） | **~5.1×** | ~2.2× |

> 💡 **说明**：V-Major 在多节点中效果显著，因慢速 inter-GPU/inter-node 通信占比更高，选择性通信节省更明显。

#### 消融实验（Ablation Study on SD3.5 @1536×1536）
| 组件 | 贡献（平均加速） | 说明 |
|------|------------------|------|
| TAPA | 2.2× | 所有配置下稳定增益，源于拓扑对齐 |
| V-First | 1.3×（多节点） | 在单节点增益小，在多节点因通信压力增大而显现价值 |
| V-Major | 2.9×（多节点） | 是多节点性能飞跃的关键驱动因素 |

> 🔻 注意：当节点数超过模型 attention head 数量上限后，加速比下降（如 Qwen-Image 在 4 节点后性能反转），因为超出部分转为 Ring Parallelism，不再受 CoCoDiff 优化影响。

#### 通信延迟对比（TAPA vs. oneCCL）
| 消息大小（每 rank） | oneCCL 延迟 | TAPA 总延迟 | 加速比 |
|---------------------|------------|-------------|--------|
| 48 MB | 25.5 ms | 6.0 ms | **3.9×** |
| 24 MB | ~18 ms | ~6.0 ms | ~3.0× |
| 6 MB | ~6.5 ms | ~6.0 ms | ~1.1× |

> ✅ TAPA 在大消息场景下优势显著，得益于 Phase 2 的并行 ring 结构。

#### 图像质量评估（Table II）
| 模型 | 方法 | PSNR (dB) | SSIM | MSE |
|------|------|----------|------|-----|
| SD3.5 | Flat / CoCoDiff | 23.05 / 22.77 | 0.8036 / 0.8042 | 0.0065 / 0.0081 |
| Qwen-Image | Flat / CoCoDiff | 29.97 / 29.17 | 0.9256 / 0.9229 | 0.0011 / 0.0012 |

> ✅ 差异极小（PSNR < 2dB，SSIM < 0.03），表明 **CoCoDiff 几乎不损失图像质量**。

#### 速度-质量权衡分析（Figure 10）
- **time-varying cache ratio**（动态从 0→1）优于固定缓存比率：
  - 在相同速度下提供更高 PSNR/SSIM；
  - 在相同质量下实现更大加速；
- 证明了自适应调度的有效性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **通信是 DiT 分布式推理的主要瓶颈**，尤其在高分辨率和多节点扩展时，all-to-all 占据高达 96% 的时间。
2. **QKV 处理不对称性** 和 **时间冗余性** 是可被系统级优化利用的重要特征。
3. **拓扑感知通信分解（TAPA）** 能有效缓解带宽不对称问题，大幅提升通信效率。
4. **V-First + V-Major** 实现了通信隐藏与体积压缩的双重优化，是实现高加速比的核心。
5. **CoCoDiff 在保持图像质量几乎不变的前提下**，实现了 **平均 3.6×、最高 8.4× 的端到端加速**。

### ⚠️ 局限性
1. **受限于 Ulysses Parallelism 的 head 数限制**：
   - 当设备数量超过模型 attention head 数时，多余设备进入 Ring Parallelism，不再受益于 CoCoDiff 优化。
2. **缓存机制带来额外内存开销**：
   - 尽管 per-tile 开销随规模线性下降，但在大 batch 或小规模部署时可能触发 OOM（如 FLUX.2-dev 在较大 batch 下失败）。
3. **当前仅针对 Intel GPU 集群优化**：
   - TAPA 设计强依赖 Aurora 的 tile-ring 架构，移植到其他平台（如 NVIDIA）需重新适配。

### 🔮 未来工作方向
1. **扩展至更多并行范式**：将 V-Major 思路应用于 Ring、Pipeline 或 Tensor Parallelism 中的通信优化。
2. **自动缓存策略学习**：引入轻量级 policy network 自适应决定哪些 token 应被缓存。
3. **支持异构集群通用化**：抽象通信拓扑建模，使 CoCoDiff 可自动适配不同硬件架构（如 NVIDIA DGX、AMD CDNA）。
4. **集成量化与稀疏化**：联合优化通信压缩与模型压缩技术（如 CacheQuant）。

---

> 🧩 **总体评价**：  
> CoCoDiff 是首个将 **应用语义（QKV asymmetry + temporal redundancy）** 与 **硬件拓扑（tile-aware hierarchy）** 深度协同设计的 DiT 推理引擎，代表了“**通信-计算-模型”协同优化**的新范式，具有广泛推广潜力。

</details>

---

### 7. [Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving](https://arxiv.org/abs/2604.14993)

**Authors**: Tingyang Sun, Ting He, I-Hong Hou  
**Category**: cs.DC  
**Published**: 2026-04-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.14993v1  

#### Abstract
As a current trend in Artificial Intelligence (AI), large foundation models are increasingly employed as the core of AI services. However, even after training, serving such models at scale remains a challenging task due to their heavy resource footprints, particularly in terms of GPU memory. While r...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**大规模基础模型（Large Foundation Models）在推理服务中的资源分配难题**，特别是由于模型巨大的内存占用（memory footprint）导致的 **GPU 内存瓶颈**问题。传统分布式计算系统通常以计算为瓶颈，而大模型服务则以**内存为瓶颈**，尤其是在存储模型参数和推理过程中的中间状态（如 KV Cache）时。

具体而言，论文聚焦于通过 **Pipeline Parallelism** 构建服务器链（server chain）来服务链式结构任务（chain-structured jobs）时的三个耦合子问题：
- **Block Placement**：如何将模型块（如 Transformer 层）放置到物理服务器上
- **Cache Allocation**：如何在服务器上分配缓存空间用于存储 KV Cache
- **Job Dispatching**：如何将请求分发到不同的服务器链上

### 提出的新方法与新思路
论文提出了一种**两时间尺度的优化框架**，将复杂的联合优化问题分解为可管理的部分：

1. **问题抽象与形式化**  
   首次将大模型推理服务抽象为“**Server Chain Composition via Block Placement and Cache Allocation**”这一新型组合优化问题，并证明其 **NP-hard** 性质。

2. **离线阶段：服务器链组成（Server Chain Composition）**
   - **Greedy Block Placement with Cache Reservation (GBP-CR)**：一种贪心算法，在块放置时预留缓存空间，优先将快速服务器组合成短链，以最小化服务时间。
   - **Greedy Cache Allocation (GCA)**：在块放置完成后，进一步利用剩余内存进行缓存分配，构建更多高吞吐量的服务链。

3. **在线阶段：负载均衡策略**
   - 提出 **Join-the-Fastest-Free-Chain (JFFC)**，是 JFFS（Join-the-Fastest-Free-Server）在支持并行处理场景下的扩展版本，能够动态选择最快且有容量的服务器链来处理新到达的请求。

4. **理论分析与性能边界**
   - 给出了在 JFFC 调度下系统的**稳态平均响应时间的闭式上下界**（closed-form upper/lower bounds），可用于指导参数调优（如缓存预留量 `c`）。

### 相比现有方法的优势
- **系统性建模**：不同于以往仅关注实现的系统工作（如 vLLM, PETALS），本文从理论层面揭示了根本性的资源管理挑战。
- **性能显著提升**：相比当前最先进的方案（如 PETALS 和 BPRR），提出的解决方案实现了 **63–77% 的平均响应时间降低**。
- **鲁棒性强**：即使在真实流量偏离理论假设（非泊松到达、非指数服务时间）的情况下仍表现优异。
- **可指导调参**：提出的理论界限能有效指导设计参数 `c` 的选择，优于启发式方法。

---

## 2. 核心实验方法和设置

### 数据集与模型
- **模拟环境**：基于自定义仿真器（代码开源），模拟 Pipeline-Parallel LLM 推理。
  - 模型：BLOOM-176B（70 层 Transformer）
  - 参数大小：每层 `sm = 1.32GB`，KV Cache 大小：`sc = 0.11GB`
- **真实系统实验**：基于修改版的 **PETALS** 分布式 LLM 推理系统
  - 模型：LLaMA-2-7B（32 层）
  - 硬件：3 × A100 (80GB)，使用 NVIDIA MIG 技术虚拟出 9 个 GPU 实例（3g.40gb 和 2g.20gb）
  - 流量来源：Azure LLM inference trace（2023年11月11日采集）
    - 平均请求速率：2.57 req/s
    - 输入长度：2048 tokens
    - 输出长度：28 tokens

### 实验设置
- **网络延迟**：使用 RIPE Atlas European network 的 RTT 数据模拟广域网通信开销。
- **通信模式**：Orchestrator 中继模式（类似 PETALS）。
- **缓存策略**：静态预分配（static cache allocation），按最大序列长度预分配 KV Cache。
- **评估指标**：
  - 平均/中位数/P95/P99 响应时间（Response Time）
  - 等待时间（Waiting Time）和服务时间（Service Time）分解
  - 改进百分比（Improvement vs. baseline）

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **PETALS [6]** | 当前 PETALS 系统使用的启发式块放置与路由策略 |
| **BPRR [29]** | 最近提出的一种两阶段资源分配方法，未显式构建服务器链 |
| **JFFC only** | 所有服务器都部署完整模型实例，仅用 JFFC 进行调度（无分片） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| 指标 | PETALS | BPRR | JFFC only | **Proposed** |
|------|--------|------|-----------|--------------|
| **Mean Response Time (s)** | 31.4 | 19.8 | 10.0 | **7.3** |
| **Median Response Time (s)** | 27.8 | 16.9 | 8.5 | **6.5** |
| **P95 Response Time (s)** | 68.5 | 44.2 | 22.1 | **15.2** |
| **Mean Waiting Time (s)** | 24.2 | 12.6 | 1.5 | **0.6** |
| **Mean Service Time (s)** | 7.2 | 7.2 | 8.5 | **6.7** |

### 与基线方法的对比结果
- 相比 **PETALS**：
  - 平均响应时间减少 **76.8%**
  - P95 响应时间减少 **77.8%**
  - 平均等待时间减少 **97.5%**
- 相比 **BPRR**：
  - 平均响应时间减少 **63.1%**
  - 显示出明显的性能代际优势
- 相比 **JFFC only**（全模型副本）：
  - 仍有 **27%** 的响应时间优势，说明合理的分片 + 缓存分配优于简单复制

### 消融实验与关键观察
- **等待时间大幅下降**：改进主要来自于**等待时间的压缩**（从 24.2s → 0.6s），表明 JFFC + 合理链构造显著提升了并发能力。
- **服务时间略有下降**：得益于更优的块放置（如避免慢节点瓶颈），服务时间也有所优化。
- **参数 `c` 的影响显著**（见 Figure 6 & 7）：
  - 存在一个最优的缓存预留量 `c*`
  - 使用理论下界（Lower Bound from Theorem 3.7）预测的 `c*` 与实际最优值高度一致
  - 启发式目标函数 `c·K(c)` 或上界预测效果较差
- **低资源环境下优势更大**（Figure 8）：
  - 在服务器数量少（J=10）或高性能 GPU 比例低（η=0.1）时，性能增益高达 **83%**

---

## 4. 关键结论和发现

### 主要发现
1. **内存是大模型服务的核心瓶颈**，必须联合考虑模型放置与 KV Cache 分配。
2. **显式构造“服务器链”并进行容量规划**（via GBP-CR + GCA）比动态调度策略（如 BPRR）更高效。
3. **JFFC 是适用于异构、并行化服务器链的理想负载均衡策略**，接近理论最优。
4. **理论推导的响应时间边界具有实际指导意义**，可用于自动调参，无需依赖大量试错。
5. 即使在**真实流量非理想分布**（bursty 到达、轻尾服务时间）下，该方法依然鲁棒有效。

### 方法的局限性
- 假设**静态缓存分配**，不支持动态扩展 KV Cache（如处理超长上下文）。
- 假设**无任务迁移或抢占**，因 KV Cache 迁移代价过高。
- 目前为**集中式调度架构**，未考虑完全去中心化的场景（如志愿者设备网络）。
- 块放置假设为连续区间（contiguous block placement），未探索更灵活的非连续分片。

### 未来工作方向
- 支持**动态缓存分配**与弹性扩容。
- 研究**时间变化的需求**下的自适应资源配置。
- 将框架扩展至**多模型共存**与**混合并行策略**（Pipeline + Tensor Parallelism）。
- 探索**去中心化协调机制**，适用于弱连接设备组成的边缘推理网络。
- 结合更高级的**Job Scheduling**策略（如 [21]）进一步优化队列行为。

> ✅ **总结一句话**：本文首次系统性地提出了面向大模型推理的“服务器链组成”优化框架，通过 GBP-CR + GCA + JFFC 的协同设计，在理论与实践中均实现了对现有方法的显著超越，为高效的大规模模型服务提供了新的范式。

</details>

---

### 8. [Rethinking AI Hardware: A Three-Layer Cognitive Architecture for Autonomous Agents](https://arxiv.org/abs/2604.13757)

**Authors**: Li Chen  
**Category**: cs.AI  
**Published**: 2026-04-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.13757v1  

#### Abstract
The next generation of autonomous AI systems will be constrained not only by model capability, but by how intelligence is structured across heterogeneous hardware. Current paradigms -- cloud-centric AI, on-device inference, and edge-cloud pipelines -- treat planning, reasoning, and execution as a mo...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Rethinking AI Hardware: A Three-Layer Cognitive Architecture for Autonomous Agents*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 AI 系统在部署上面临多重挑战：
- **云中心化系统**（Cloud-centric）存在高延迟（1–3秒）、能耗高、依赖网络连接；
- **边缘设备推理**（Edge-only）受限于算力，难以处理复杂任务；
- **混合边云流水线**缺乏对“认知功能”的合理划分，导致效率低下。

根本问题是：**将 planning、reasoning 和 execution 视为单一计算任务，未根据其时间尺度和硬件特性进行解耦**。

---

### 🚀 提出的新方法：Tri-Spirit Architecture
提出一种三层认知架构 **Tri-Spirit**，将智能体的认知过程显式分解为三个异构层，并通过异步消息总线协调：

| 层 | 功能 | 时间尺度 | 实现方式 |
|----|------|---------|--------|
| **Super Layer (Ls)** | 长期规划、目标生成、跨智能体协调 | 秒到分钟级 | 云端大模型（如 GPT-4） |
| **Agent Layer (La)** | 任务分解、决策制定、上下文感知推理 | 毫秒到秒级 | 设备端紧凑型 LLM（7B–13B 参数） |
| **Reflex Layer (Lr)** | 传感器响应、低延迟控制、习惯执行 | 微秒到毫秒级 | 有限状态机 FSM / 查表策略，无 LLM 推理 |

此外引入以下机制支持该架构：
- **异步消息总线 SpiritBus**：实现层间通信与解耦；
- **参数化路由策略 Routing Policy**：基于任务的 *latency urgency* 和 *cognitive complexity* 自动分配至合适层级；
- **Habit Compilation Mechanism**：将高频重复的 reasoning 路径编译为零推理开销的轻量级执行策略；
- **记忆模型与安全约束**：保障系统一致性与安全性。

---

### 🔍 相比现有方法的优势
| 对比维度 | Tri-Spirit | 现有方法（Cloud/Edge/Hybrid） |
|--------|-----------|-----------------------------|
| 架构设计 | 显式的**认知功能分解**（非仅计算分片） | 多为 DNN 层切分或简单 offloading |
| 延迟优化 | 支持微秒级 Reflex 响应，避免被 Planning 阻塞 | 无法兼顾实时性与复杂推理 |
| 能耗控制 | 减少不必要的 LLM 调用，尤其通过 Habit 编译实现零推理执行 | 每次调用均需完整 inference |
| 离线可用性 | 77.6% 任务可在离线状态下完成 | Cloud-centric 完全不可用 |
| 可扩展性 | 各层可独立优化硬件与算法，支持异构部署 | 单一模型或固定流水线限制灵活性 |

> ✅ 创新本质：从“模型规模驱动”转向“**认知结构驱动**”，强调系统级效率而非单纯提升参数量。

---

## 2. 核心实验方法和设置

### 📊 数据集与任务生成
- **合成任务集**（N = 2,000），模拟真实场景中的多样化请求；
- 每个任务具有两个关键属性：
  - `l ∈ [0,1]`：**latency urgency**（越小表示时限越紧）
  - `c ∈ [0,1]`：**cognitive complexity**（越大表示需要更多推理）

任务分为三类：
| 类型 | 占比 | 特征 |
|------|-----|------|
| Type A (Reactive) | 60% | 低延迟、低复杂度（如语音唤醒） |
| Type B (Reasoning) | 30% | 高复杂度、中等延迟要求（如行程规划） |
| Type C (Repeated) | 10% | 高频重复模式（如每日提醒执行） |

---

### ⚙️ 实验设置
- **仿真平台**：Python 模拟器 `simulate_v3.py`（开源附带）
- **随机种子固定**：`seed=42`，确保可复现性
- **Bootstrap 分析**：2,000 次重采样，报告 95% 置信区间（CI）

#### 延迟模型（均值 ± 标准差）
| 组件 | 延迟 |
|------|------|
| Lr (Reflex) | 5.5 ± 1.4² ms |
| La (Agent) | 155 ± 30² ms |
| Ls (Super) | 1920 ± 280² ms（含 LTE RTT） |
| Habit Policy | 2.1 ± 0.5² ms |

#### 能耗模型
| 组件 | 能耗 |
|------|------|
| Lr | 0.48 mJ |
| La | 10.2 mJ |
| Ls | 40.5 mJ（含约 30mJ 无线传输） |
| Habit | 0.09 mJ |

---

### 📈 评估指标
| 指标 | 描述 |
|------|------|
| **Mean Latency ↓** | 平均任务完成时间 |
| **Energy Consumption ↓** | 每任务平均能耗 |
| **LLM Invocations ↓** | 每任务平均调用次数（Ls 算 2 次） |
| **Offline Completability ↑** | 不依赖网络即可完成的任务比例 |
| **P95 Latency ↓** | 第95百分位延迟，反映尾部表现 |

---

### 🆚 基线方法对比
| 基线 | 描述 |
|------|------|
| **Cloud-Centric** | 所有任务发送至 Ls 处理 |
| **Edge-Only** | 所有任务由本地 La 处理（牺牲部分质量以保低延迟） |

---

## 3. 主要实验结果和性能指标

### 📉 关键性能数据（默认阈值下）

| System | Mean Latency (ms) | Energy (mJ) | LLM Calls | Offline (%) |
|--------|--------------------|-------------|-----------|--------------|
| **Cloud-Centric** | 2,146 [2,138–2,153] | 46.1 | 1.00 | 0.0 |
| **Edge-Only** | 179 [178–180] | 11.8 | 1.00 | 100.0 |
| **Tri-Spirit** | **523 [486–562]** | **13.3** | **0.70 [0.65–0.74]** | **77.6** |

> ✅ **核心提升**：
- **平均延迟降低 75.6%**（vs Cloud）
- **能耗减少 71.1%**
- **LLM 调用减少 30%**
- **离线可完成率达 77.6%**

---

### 🔍 按任务类型细分效果（图5）
| 任务类型 | Tri-Spirit 表现 |
|--------|----------------|
| **Type A (Reactive)** | 延迟从 ~1,909ms（Cloud）降至 **40ms**（主要由 Reflex Layer 处理） |
| **Type B (Reasoning)** | 改善有限（多数仍需 Ls），体现架构权衡 |
| **Type C (Repeated)** | 延迟从 155ms（Edge）降至 **2.1ms**，且 **零 LLM 调用**（Habit 编译生效） |

---

### 🔧 消融实验结果（Ablation Study）

| 变体 | Latency (ms) | Energy (mJ) | LLM Calls | Offline (%) |
|------|-------------|------------|-----------|-------------|
| **TS-Full**（完整版） | **523** | **13.3** | **0.67** | **77.6** |
| TS-NoReflex | 601 | 18.4 | 1.12 | 77.6 |
| TS-NoHabit | 533 | 13.9 | 0.72 | 77.6 |
| TS-RandomRoute | 523 | 13.2 | 0.67 | 77.6 |
| TS-LocalOnly | 93 | 6.2 | 0.50 | 100.0 |

#### 因果归因分析（Latency Saving 来源）
| 组件 | 延迟降低（ms） | 占总改进比例 |
|------|---------------|-------------|
| **Local Execution**（避免上云） | -2,053 | 95.7% |
| **Reflex Layer**（快速路径） | -78 | 3.6% |
| **Habit Compilation**（零推理） | -10 | 0.5% |
| **Intelligent Routing**（质量匹配） | ~0 | <0.1% |

> 💡 发现：
- 延迟优势主要来自“**本地执行**”，而非三层结构本身；
- **Reflex 与 Habit 是 Tri-Spirit 的差异化贡献**，虽占比小但在关键场景至关重要；
- 智能路由不影响延迟分布，但保障了输出质量（防止复杂任务降级处理）。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **认知分解是系统效率的关键驱动力**  
   相较于单纯扩大模型规模，合理的功能拆分更能显著提升整体性能。

2. **三层架构实现了多目标平衡**  
   在保持高质量推理能力的同时，满足低延迟、节能、离线运行等现实需求。

3. **Habit Compilation 具有长期增益价值**  
   尽管单次延迟节省有限（-10ms），但能持续降低能耗与 LLM 成本，适用于高频重复任务场景。

4. **系统具备优雅退化能力（Graceful Degradation）**  
   当 Ls 不可达时，La 与 Lr 仍可维持基本服务，提升鲁棒性。

5. **路由策略的价值在于质量保障而非延迟优化**  
   即使随机分配任务也能获得相近延迟，但会牺牲 ~22% 复杂任务的质量。

---

### ⚠️ 方法的局限性
1. **仿真假设偏理想化**
   - 延迟建模使用正态分布，实际系统常呈重尾分布（如热节流、队列堆积）；
   - 忽略了任务属性预测误差（实践中需快速分类器估计 `l`, `c`）；

2. **静态路由阈值**
   - 默认阈值基于先验设定，动态环境可能需要在线自适应调整；

3. **尚未实机验证**
   - 所有结果基于模拟器，缺少真实硬件上的功耗测量与用户体验评估。

---

### 🔮 未来工作方向
1. **开发自适应路由机制**  
   利用反馈信号动态调节 `Tr`, `Ta` 阈值，应对任务分布漂移。

2. **构建物理原型并实测验证**  
   在手机、可穿戴设备、机器人平台上部署 Tri-Spirit，收集真实性能数据。

3. **扩展 Habit Learning 能力**  
   引入 context-aware drift detection 与增量 recompilation，增强策略稳定性。

4. **探索与其他 Cognitive Architectures 的融合**  
   如结合 AutoGPT 的循环规划机制，或将 ROS2 控制器集成进 Reflex Layer。

5. **建立 Tri-Spirit Benchmark Suite**  
   推动标准化测试框架，促进硬件-软件协同优化研究。

---

> 📌 **最终结论**：  
> Tri-Spirit 提供了一个**硬件感知、认知结构化的 AI 系统设计范式**。它表明，在迈向真正自主智能体的过程中，我们不仅需要更强的模型，更需要更聪明的系统架构。**认知解耦 + 习惯固化 = 高效、可持续、贴近人类行为模式的 AI 硬件未来**。

</details>

---

### 9. [Awakening Dormant Experts:Counterfactual Routing to Mitigate MoE Hallucinations](https://arxiv.org/abs/2604.14246)

**Authors**: Wentao Hu, Yanbo Zhai, Xiaohui Hu, Mingkuan Zhao, Shanhong yu, Xue Liu, Kaidong Yu, Shuangyong Song, Xuelong Li  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.14246v1  

#### Abstract
Sparse Mixture-of-Experts (MoE) models have achieved remarkable scalability, yet they remain vulnerable to hallucinations, particularly when processing long-tail knowledge. We identify that this fragility stems from static Top-$k$ routing: routers tend to favor high-frequency patterns over rare fact...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Awakening Dormant Experts: Counterfactual Routing to Mitigate MoE Hallucinations*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
- **MoE模型中的“幻觉”问题**，尤其是在处理**长尾知识**（long-tail knowledge）时表现尤为严重。
- 作者指出，这一问题的根本原因在于标准的 **Top-k 路由机制**存在**相关性-因果性错位**（correlation-causality misalignment）：
  - 路由器倾向于选择高频模式（如常见语法结构），而忽视低频但关键的事实关联。
  - 导致拥有特定领域知识的“专家”（experts）被系统性地低估，成为“休眠专家”（dormant experts）——即模型“知道”事实，但无法“回忆”，因为路由未激活这些专家。

### 🚀 提出的新方法：**Counterfactual Routing (CoR)**
- 一种**无需训练**（training-free）、在推理阶段即可应用的框架，旨在“唤醒”这些休眠专家。
- 核心思想是通过**因果引导的资源重分配**（causal-guided resource reallocation），实现计算成本不变下的专家选择优化。

#### 创新机制包括：
1. **层间动态预算再分配**（Layer-wise Adaptive Budgeting）：
   - 使用**扰动分析**（perturbation analysis）结合**对比敏感度归一化**（Contrastive Sensitivity Normalization）识别“知识密集型层”（knowledge-intensive layers）。
   - 将更多专家预算从“语法主导层”转移到“知识密集层”，保持总激活专家数不变（compute-preserving）。

2. **专家级因果影响度量**（Expert-wise Causal Impact）：
   - 提出 **Counterfactual Expert Impact (CEI)** 指标，基于虚拟消融（virtual ablation）衡量某个专家对正确预测的实际因果贡献。
   - 高 CEI 但低路由得分的专家即为“休眠专家”，CoR 在推理时优先激活它们。

### ⭐ 相比现有方法的优势
| 方法 | 局限性 | CoR 的优势 |
|------|--------|-----------|
| **Retrieval-Augmented Generation (RAG)** | 需要外部检索、额外训练 | 无需训练、不依赖外部知识 |
| **DoLa / ITI** | 作用于输出分布或残差流，属于“事后修正” | **直接干预路由决策**，解决根本瓶颈 |
| **静态扩展策略**（如增加 Top-k） | 增加计算开销，收益递减 | **同等计算成本下性能更优**，效率更高 |

> ✅ **核心优势**：CoR 是首个专门针对 MoE 架构中**路由层面幻觉成因**进行干预的方法，实现了“精准激活”而非“盲目扩容”。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **校准数据集（用于离线因果分析）**：
  - `C4` (Colossal Clean Crawled Corpus)，随机采样约 1,000 个 token。
  - 分为“难样本”（hard tokens，高 NLL）和“易样本”（easy tokens，低 NLL）以区分知识强度。
  
- **评估基准（测试集）**：
  - **事实一致性**：`TruthfulQA`, `FACTOR`, `TriviaQA`
  - **通用能力**：`MMLU`, `ARC-C/E`, `GSM8K`

### ⚙️ 实验设置
- **模型**：在三种不同架构的 MoE 模型上验证泛化性：
  - `Qwen-3-30B-A3B`
  - `DeepSeek-V2-Lite`
  - `GPT-OSS-20B`
  - （附录还验证了更大规模的 `TeleChat3-105B-A4.7B`）
- **硬件**：单张 NVIDIA H100 80GB GPU
- **评估方式**：zero-shot setting

### 🎯 评估指标
- 主要指标：**平均事实准确率提升**（如 TruthfulQA 的 MC1/MC2/Gen，TriviaQA 的 Exact Match）
- 辅助指标：通用任务性能（MMLU、ARC、GSM8K）以验证是否损害整体能力
- 效率指标：FLOPs 对比（验证 compute-preserving 特性）

### 🔁 基线方法对比
| 基线 | 描述 |
|------|------|
| **Standard Top-k Routing** | 默认路由机制 |
| **Random Routing** | 控制组，随机选择专家 |
| **DoLa** | 解码时对比早晚期 logits 放大事实信号 |
| **ITI** | 推理时沿“真实性方向”调整激活值 |
| **Static Scaling (Top-9 ~ Top-12)** | 固定增加专家数量作为效率对比 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）
| Model | Method | Average Factuality Score |
|-------|--------|--------------------------|
| Qwen-3-30B-A3B | Standard | 48.01 |
| | **CoR (Ours)** | **49.95** (+1.94) |
| DeepSeek-V2-Lite | Standard | 38.12 |
| | **CoR (Ours)** | **43.57** (+5.45) |
| GPT-OSS-20B | Standard | 33.58 |
| | **CoR (Ours)** | **35.49** (+1.91) |

> ✅ **总体平均提升：+3.1%**，且**未增加推理预算**。

### 🔍 与基线方法对比
- **显著优于所有基线**，尤其在 DeepSeek 上提升达 **5.45%**。
- **DoLa 和 ITI 在 MoE 上效果有限**，甚至不如标准路由（如 GPT-OSS-20B 上仅微幅提升），说明其“事后修正”策略难以弥补错误的专家选择。
- **Random Routing 表现最差**，证明专家选择至关重要。

### 🔬 消融实验（Ablation Study, Table 2）
| 方法 | Average Score | 提升来源 |
|------|---------------|---------|
| Standard | 48.01 | — |
| Layer-wise Only | 48.42 | 深层知识集中，预算转移有效 |
| Expert-wise Only | 49.27 | CEI 成功识别并激活关键专家 |
| **CoR (Full)** | **49.95** | 两者协同增效 |

> ✅ 结论：**层间预算重分配** 与 **专家级因果选择** 是互补机制，联合使用效果最佳。

### 💡 效率分析（Figure 5）
- **Pareto 前沿分析显示**：CoR（等效于 Top-8 成本）的性能**超过静态 Top-12**。
- 证明：**MoE 事实性的瓶颈不是参数量，而是专家选择的精度**。

---

## 4. 关键结论和发现

### 🧠 主要发现
1. **“休眠专家”现象普遍存在**：
   - 大量专家具有高因果影响力（CEI），但在标准路由下获得极低门控分数。
   - 图 4 显示明显的“能力-信心差距”（competence-confidence gap）。

2. **路由决策是 MoE 幻觉的关键瓶颈**：
   - 即使模型内部存储了正确知识，若路由未能激活对应专家，仍会生成幻觉。
   - 现有 post-hoc 方法（如 DoLa）对此无能为力。

3. **因果导向的专家选择优于统计流行度**：
   - CoR 通过 CEI 度量打破频率偏见，实现更准确的知识召回。

4. **方法安全且可扩展**：
   - 在通用任务（MMLU、ARC、GSM8K）上性能稳定，未损害语言流畅性。
   - 在百B级 MoE 模型（TeleChat3-105B）上依然有效（Table 6）。

### ⚠️ 局限性
- **依赖模型已有知识**：CoR 只能优化“已知知识”的检索，**不能注入新知识**。
  - 若事实完全不在预训练语料中（out-of-pretraining knowledge），则无法纠正。
- **离线分析需少量计算**：CEI 计算虽为一次性，但仍需在小数据集上进行前向/反向传播。
- **超参数敏感性较低但存在**：λ 设置过高可能破坏语言流畅性。

### 🔮 未来工作方向
- **与 RAG 结合**：构建混合系统——CoR 用于内部知识检索，RAG 用于外部知识补充。
- **在线自适应 CEI 更新**：让模型在部署过程中持续更新因果影响评分。
- **扩展至其他稀疏架构**：如 Sparse Attention、Blockwise MoE 等。
- **探索多模态 MoE 中的休眠专家问题**。

---

> ✅ **总结一句话**：  
> **CoR 揭示了 MoE 幻觉的本质是“路由失灵”而非“知识缺失”，并通过因果引导的专家重路由，在不增加计算成本的前提下，显著提升了事实准确性，为构建可信的大模型提供了新范式。**

🔗 代码开源地址：[https://github.com/ZhaiYanbo/CoR](https://github.com/ZhaiYanbo/CoR)

</details>

---

### 10. [Generative Augmented Inference](https://arxiv.org/abs/2604.14575)

**Authors**: Cheng Lu, Mengxin Wang, Dennis J. Zhang, Heng Zhang  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.14575v1  

#### Abstract
Data-driven operations management often relies on parameters estimated from costly human-generated labels. Recent advances in large language models (LLMs) and other AI systems offer inexpensive auxiliary data, but introduce a new challenge: AI outputs are not direct observations of the target outcom...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Generative Augmented Inference 论文总结

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对 **data-driven operations management** 中一个核心挑战：人类标注数据（如购买决策、专家注释、调查反馈）成本高昂，难以大规模获取，而依赖这些标签进行参数估计会严重限制样本量和模型精度。

尽管 **Large Language Models (LLMs)** 和其他 AI 系统可以低成本生成大量辅助数据，但直接将其作为真实标签的替代品存在两大问题：
1.  **系统性偏差**：AI 输出与人类判断存在复杂且未知的关系，可能引入严重偏差。
2.  **形式差异**：AI 输出常为高维表示（如推理链、置信度分数、嵌入向量），而非简单的标签，无法直接用于传统方法。

现有方法（如 **Prediction-Powered Inference, PPI**）将 AI 预测视为有噪声的真实标签代理，当 AI 预测质量差或形式不同时，效果不佳甚至失效。

### 提出的新方法和新思路
论文提出了 **Generative Augmented Inference (GAI)** 框架，其核心创新在于**概念上的根本转变**：

- **核心思想**：**不将 AI 输出视为真实标签的“代理”（surrogate），而是将其视为预测人类标签的“信息特征”（informative features）**。
- **方法论**：GAI 利用 **Neyman 正交性**（Neyman orthogonality）构建了一个正交矩（orthogonal moment）。该框架分为两步：
  1.  **Nuisance Estimation**：在有标签的子集上，使用灵活的机器学习方法（如随机森林、神经网络）估计两个辅助函数：
      - `g(X, z)`：给定协变量 `X` 和 AI 输出 `z` 时，对人类标签 `y` 的最佳预测。
      - `e(X, z)`：观测到人类标签的概率（倾向得分）。
  2.  **Bias-Corrected Estimation**：利用估计出的 `g` 和 `e` 构建一个正交得分函数，整合有标签和无标签数据，进行最终的目标参数估计。

### 相比现有方法的优势
1.  **更广泛的适用性**：能处理任何形式的 AI 输出（离散标签、连续分数、高维嵌入、非结构化文本），而 PPI 仅适用于代理标签。
2.  **更强的鲁棒性和安全性**：在随机抽样标注条件下，GAI 具有 **“safe default” 属性**：它保证不会比仅使用人类数据的估计器更差，并且只要 AI 输出包含任何预测信息，就能严格提升效率。
3.  **更高的效率**：通过有效利用无标签数据中的信息，显著降低了对昂贵人类标注的需求。
4.  **有效的信息提取**：不仅能利用 AI 的额外预测信息，还能利用其强大的“表征能力”（representational power）来近似复杂的条件期望，即使该信息在原始协变量 `X` 中已存在。

## 2. 核心实验方法和设置

### 使用的数据集
论文在三个真实世界的商业应用中进行了评估，覆盖了不同的辅助数据场景：
1.  **疫苗联合分析 (Vaccine Conjoint Analysis)**：使用 Kreps et al. (2020) 的数据集，包含 1,971 名受访者对假想疫苗的选择。AI 输出来自 GPT-4o 等 LLM 的 **Chain-of-Thought (CoT) 推理**，具体为：
    - **Labels**：离散选择预测 `{abstain, option 1, option 2}`。
    - **Embeddings**：使用 OpenAI 的 `text-embedding-3-large` 模型将完整的推理文本转换为 **3072 维的向量**。
2.  **零售定价 (Retail Pricing)**：使用 Toubia et al. (2025) 的数据集，包含 2,058 名参与者对产品购买意向的调查。AI 输出是基于参与者详细画像（persona）构建的 **“数字孪生” (digital twins)** 生成的二元购买预测。
3.  **健康保险选择 (Health Insurance Choice)**：使用 Angelopoulos et al. (2023a) 构建的美国加州人口普查数据集，研究收入与私人医疗保险覆盖率的关系。AI 输出是由梯度提升分类器生成的**连续概率预测**，该模型使用了比目标模型更丰富的特征集。

### 实验设置和评估指标
- **实验设计**：采用重复抽样实验。对于每个主样本大小 `np`（人类标注数），固定一个较大的辅助样本大小 `nA`（AI 数据数），运行 50 次独立试验并报告平均值。
- **评估指标**：
    1.  **点估计 (Point Estimation)**：使用 **MAPE (Mean Absolute Percentage Error)** 衡量估计系数与真实值的平均百分比偏差。
    2.  **推断质量 (Inference Quality)**：
        - **置信区间覆盖率 (CI Coverage)**：95% 置信区间包含真实参数的比例，理想值为 95%。
        - **置信区间宽度 (CI Width)**：衡量推断的精确度。
        - **决策错误率 (Decision Errors)**：因置信区间导致的管理决策错误比例。

### 基线方法对比
论文将 GAI 与以下四种基准方法进行了比较：
1.  **Primary**：仅使用人类标注数据的最大似然估计。
2.  **Naive**：简单地将人类和 AI 标签混合后进行最大似然估计。
3.  **PPI**：Prediction-Powered Inference 的原始版本。
4.  **PPI++**：PPI 的计算高效变体。

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
GAI 在所有三个应用场景中均表现出色，实现了显著的性能提升：

| 应用场景 | 性能提升 | GAI vs. Best Baseline (PPI++) |
| :--- | :--- | :--- |
| **疫苗联合分析** | **MAPE 降低约 50%** <br> **人类标注需求减少 >75%** | GAI (Labels/Embeddings) 的 MAPE 约为 **16-17%**，而 Primary 在 `np=200` 时仍有 **19.0%**。GAI (`np=50`) 的表现优于 Primary (`np=200`)。 |
| **零售定价** | **MAPE 降低约 50%** <br> **人类标注需求减少 67%** | GAI 的 MAPE 为 **7-12%**，而 PPI++ 为 **9.7-22.2%**。GAI (`np=100`) 的精度与 Primary (`np=300`) 相当。 |
| **健康保险选择** | **MAPE 降低 >50%** <br> **人类标注需求减少 >90%** | GAI 的 MAPE 为 **140-160%**，而 Primary (`np=1000`) 仍有 **290%**。GAI (`np=100`) 的表现超越了 Primary (`np=1000`)。 |

### 推断质量
- **覆盖率 (Coverage)**：GAI 在几乎所有配置下都达到了或超过了 95% 的名义覆盖率。相比之下，PPI++ 经常出现**欠覆盖**（undercover），例如在健康保险分析中仅为 83-93%。
- **置信区间宽度 (Width)**：GAI 的置信区间宽度与 PPI++ 相当，有时更窄，且没有以牺牲覆盖率为代价。
- **决策错误 (Decision Errors)**：GAI 的决策错误率最低，在多个实验中接近或达到零。

### 消融实验结果
论文通过一系列严谨的实验验证了其理论：
1.  **公平比较 (Level Playing Field)**：在零售定价实验中，强制所有方法（包括 PPI）使用相同的二元 AI 预测。结果证明，GAI 的优势源于其**方法论本身**，而非访问了更丰富的数据。
2.  **不同数据形式**：在联合分析中，GAI 同时成功利用了低质量的离散标签和高维文本嵌入，展示了其对多种 AI 输出格式的适应性。
3.  **不同信息结构**：从 `ylz|X`（AI 无额外信息）到 `y⊥̸z|X`（AI 有额外信息）的不同场景下，GAI 均能稳定提效，验证了其理论分解的三种增益来源（样本扩展、表征能力、额外信息）。

## 4. 关键结论和发现

### 主要发现
1.  **范式转变的有效性**：将 AI 输出视为“特征”而非“代理标签”的范式转变是成功的。GAI 能够从有偏、弱相关甚至形式迥异的 AI 输出中提取有价值的信息。
2.  **“Safe Default” 属性成立**：在随机标注条件下，使用 GAI 是一个安全的选择，因为它永远不会比忽略 AI 数据更差。
3.  **巨大的成本节约潜力**：GAI 可以将对昂贵人类标注的需求减少 **67% 至 90% 以上**，同时保持甚至提高统计推断的精度和可靠性。
4.  **广泛适用性**：GAI 的优势在从近乎随机的预测到高度校准的预测等各种 AI 数据质量下都得到了证实，证明了其强大的鲁棒性。

### 方法的局限性
1.  **随机标注假设**：论文的“安全默认”性质依赖于人类标注是随机选择的。如果标注是有策略性的（例如，优先标注 AI 不确定的案例），这种保证可能会失效。
2.  **计算复杂性**：需要进行交叉拟合（cross-fitting）和辅助函数估计，相比简单方法计算开销更大。

### 未来工作方向
1.  **扩展至非随机标注**：将 GAI 的优势扩展到非随机、战略性标注的设计中。
2.  **实时决策系统**：开发在线版本的 GAI 框架，以集成到实时决策系统中。
3.  **更广泛的模型类别**：探索将 GAI 框架应用于更广泛的模型类别，而不仅仅是广义线性模型（GLMs）。

</details>

---

### 11. [Mean Flow Policy Optimization](https://arxiv.org/abs/2604.14698)

**Authors**: Xiaoyi Dong, Xi Sheryl Zhang, Jian Cheng  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.14698v1  

#### Abstract
Diffusion models have recently emerged as expressive policy representations for online reinforcement learning (RL). However, their iterative generative processes introduce substantial training and inference overhead. To overcome this limitation, we propose to represent policies using MeanFlow models...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Mean Flow Policy Optimization》核心总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
现有的基于 **diffusion models** 的强化学习策略虽然能建模复杂的多峰动作分布，从而提升探索能力，但其迭代生成过程（iterative generative process）导致训练和推理效率低下，通常需要 10–20 步采样，显著增加了计算开销。

此外，在 **maximum entropy RL (MaxEnt RL)** 框架下优化 diffusion policy 存在两个挑战：
- 动作似然（action likelihood）难以精确计算（涉及不可解积分）
- 软策略改进（soft policy improvement）因离散化误差而在少步数下失效

### **提出的新方法与新思路**
本文提出 **Mean Flow Policy Optimization (MFPO)**，将 **MeanFlow models** 引入在线强化学习作为策略表示，并结合 MaxEnt RL 框架进行优化。

#### **核心创新点：**
- **采用 MeanFlow 模型作为策略表示**  
  利用其对“平均速度场”（average velocity field）的学习，显著降低粗粒度时间离散下的误差，实现仅需 **2 步采样** 即可高质量生成动作，大幅提升推理效率。

- **引入 Average Divergence Network (ADN)**  
  为解决 MeanFlow 策略的动作似然评估难题，设计了一个专用网络来近似时间平均散度（time-averaged divergence），使策略熵和软 Q 函数估计变得可行且高效。

- **自适应瞬时速度估计（Adaptive Instantaneous Velocity Estimation）**  
  在策略改进阶段，目标分布由 Q 函数隐式定义，无法直接采样。为此提出一种基于 **self-normalized importance sampling (SNIS)** 的混合估计器，结合高斯提议分布和当前策略分布，并以 **Effective Sample Size (ESS)** 自适应加权，提高估计稳定性与效率。

### **相比现有方法的优势**
| 维度 | MFPO | Diffusion-based Baselines (e.g., DIME, MaxEntDP) |
|------|------|-----------------------------------------------|
| 采样步数 | **2 步** | 16–20 步 |
| 推理延迟 | 显著更低（见 Table 1） | 高延迟 |
| 策略表达力 | 支持多模态动作分布 | 同样支持 |
| MaxEnt 框架兼容性 | 可行且稳定 | 少步数下性能下降严重（如 DIME 下界变松） |
| 训练效率 | 提升约 50% | 较慢 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MuJoCo**：5 个标准连续控制任务  
  `Humanoid-v3`, `HalfCheetah-v3`, `Ant-v3`, `Walker2d-v3`, `Hopper-v3`
- **DeepMind Control Suite (DMC)**：6 个高难度任务  
  `Dog Run/Walk/Stand`, `Humanoid Run/Walk/Stand`

### **实验设置与评估指标**
- **训练框架**：off-policy, online RL
- **评估周期**：每个任务运行 5 个随机种子，报告平均学习曲线与标准差
- **主要指标**：
  - **环境回报（Return）**：episode 累积奖励
  - **归一化回报（Normalized Return）**：相对于 MFPO 最终表现的比率
  - **训练时间（Training Time）**
  - **推理延迟（Inference Latency）**
  - **采样步数（Sampling Steps）**

### **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Diffusion-based** | DIME, FlowRL, MaxEntDP, DACER, QVPO |
| **Classical** | SAC, TD3 |

所有方法统一网络结构、batch size、update-to-data ratio 等超参数以保证公平比较。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Fig. 2 & E3, Table 1）**

#### **表：不同算法的采样步数与推理延迟（MuJoCo）**

| ALGORITHM | SAMPLING STEPS | INFERENCE TIME (ms) |
|----------|----------------|------------------------|
| **MFPO** | **2** | **0.46** |
| DIME | 16 | 0.97 |
| FlowRL | 11 | 0.42 |
| MaxEntDP | 20 | 1.56 |
| DACER | 20 | 1.06 |
| QVPO | 20 | 1.68 |
| TD3 / SAC | 1 | ~0.15 |

> 注：尽管 SAC/TD3 更快，但其策略表达受限（单峰），在复杂任务中易陷入局部最优。

#### **性能表现总结**
- MFPO 在大多数任务上达到或超过所有 diffusion-based 和 classical 基线的表现。
- 在 **Humanoid-v3** 上，MFPO 的最终回报优于 DIME 和 MaxEntDP。
- 在 DMC 的 `Humanoid Run` 等难任务上，MFPO 表现领先。
- **训练时间减少约 50%**，显著缩小了 diffusion policy 与传统 one-step policy 的效率差距。

#### **消融实验结果（Ablation Studies）**
在 `HalfCheetah-v3` 上验证关键组件作用：

| 消融项 | 结果分析 |
|--------|---------|
| **移除 MeanFlow（改用标准 flow matching）** | 性能明显下降 → 表明平均速度建模对少步生成至关重要 |
| **移除 ADN（即不估计 divergence 积分）** | 学习不稳定，收敛缓慢 → 表明准确似然估计是稳定优化的关键 |
| **仅使用高斯提议分布（无自适应 SNIS）** | 性能下降；仅用策略提议失败 → 表明双提议 + ESS 加权更鲁棒 |
| **固定温度 vs Auto-tuning α** | 固定温度易导致早期探索不足或后期过度探索 → 自动调温更稳定 |
| **采样步数 T=2 vs T>2** | T=2 已接近最优性能，增加步数收益有限 → 验证了 MeanFlow 的高效性 |

---

## **4. 关键结论和发现**

### **主要发现**
1. **Few-step MeanFlow 是高效的策略表示**  
   相比 diffusion models，MeanFlow 在仅 **2 步采样** 下即可保持高表达力，同时大幅降低训练与推理成本。

2. **ADN 实现了高效且准确的动作似然估计**  
   所提平均散度网络可在几乎不增加训练负担（+5% 时间）的前提下提供可靠似然估计，支撑 MaxEnt RL 中的熵正则化与重要性采样。

3. **自适应 SNIS 提升策略改进稳定性**  
   融合高斯提议与当前策略提议，并通过 ESS 加权，有效缓解了提议分布失配带来的方差问题。

4. **MFPO 实现了性能与效率的平衡**  
   在多个复杂控制任务上，MFPO 不仅性能媲美甚至超越最先进的 diffusion policy 方法，还显著提升了训练和推理效率。

### **方法的局限性**
- 当前仍需 **至少 2 步采样**，尚未实现真正的 one-step 生成。
- 对超高维动作空间的扩展性和内存占用未充分讨论。
- ADN 和策略网络分离训练，可能影响端到端优化效果。

### **未来工作方向**
- 探索进一步减少采样步数至 **1 步** 的可能性，例如结合更强大的生成先验或改进架构。
- 将 MFPO 应用于 **offline RL** 或 **multi-agent RL** 场景。
- 研究如何联合优化策略网络与 ADN，提升训练一致性。
- 探索与其他快速生成模型（如 Consistency Models、Shortcut Models）的融合。

---

> ✅ **代码开源地址**：[https://github.com/MFPolicy/MFPO](https://github.com/MFPolicy/MFPO)

</details>

---

### 12. [TOPCELL: Topology Optimization of Standard Cell via LLMs](https://arxiv.org/abs/2604.14237)

**Authors**: Zhan Song, Yu-Tung Liu, Chen Chen, Guoheng Sun, Jiaqi Yin, Chia-tung Ho, Ang Li, Haoxing Ren, Cunxi Yu  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.14237v1  

#### Abstract
Transistor topology optimization is a critical step in standard cell design, directly dictating diffusion sharing efficiency and downstream routability. However, identifying optimal topologies remains a persistent bottleneck, as conventional exhaustive search methods become computationally intractab...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《TOPCELL: Topology Optimization of Standard Cell via LLMs》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
标准单元（Standard Cell）设计中的**晶体管拓扑优化**是影响下游布线可行性（routability）、布局密度和寄生参数的关键环节。传统方法如递归搜索或穷举探索（如 SO3-Cell）在晶体管数量增加时面临**指数级计算复杂度**，难以扩展到先进工艺节点（如 7nm、2nm），成为自动化流程的瓶颈。

### 提出的新方法与新思路
本文提出 **TOPCELL** —— 一种基于 **Large Language Models (LLMs)** 的新型可扩展框架，将高维的拓扑探索问题重构为一个**生成式任务**，通过强化学习驱动 LLM 自主发现物理感知（physically-aware）且可布线的晶体管排列。

核心创新点如下：

- **LLM 驱动的拓扑优化框架**  
  首次将 LLM 应用于标准单元级的晶体管网络拓扑生成与优化，利用其强大的逻辑推理能力处理复杂的电路网表（netlist）结构。

- **引入 GRPO 进行策略微调**  
  采用 **Group Relative Policy Optimization (GRPO)** 对 LLM 进行政策优化，使其策略与物理设计反馈（Placement & Routing, P&R）对齐。相比传统的 Supervised Fine-Tuning (SFT)，GRPO 能主动探索更优解空间，避免陷入局部最优。

- **确定性的拓扑重布线机制（LLM-Guided Topology Permutation）**  
  LLM 只负责选择“pivot net”进行局部交换，具体的拓扑变换由图算法执行，确保功能等价性和实现可靠性，同时降低 LLM 的推理负担。

- **端到端加速现有 SOTA 流程**  
  将 TOPCELL 集成进现有的自动化工具链（如 SO3-Cell），替代其耗时的拓扑搜索模块，在保持布局质量的同时实现**高达 85.91× 的平均加速**。

### 相比现有方法的优势
| 维度 | 传统方法（如 SO3-Cell） | TOPCELL |
|------|------------------------|---------|
| 可扩展性 | 差（随晶体管数指数增长） | 强（零样本泛化至 6 输入函数） |
| 运行时间 | 数小时甚至超时 | 秒级完成 |
| 物理感知能力 | 依赖显式建模 | 通过奖励模型隐式学习 |
| 泛化能力 | 限于训练配置 | 支持跨工艺节点（2nm → 7nm） |

---

## 2. 核心实验方法和设置

### 数据集构建
- **基础布尔函数集合**：穷举所有非平凡的三输入单输出布尔函数（共 254 个）。
- **拓扑枚举方式**：从每个函数的多级因式分解形式（multi-level factored form）出发，使用 ABC 工具生成初始 SPICE 网表，并通过 LLM-Guided Topology Permutation 枚举最多 $ \min(100, 2^n) $ 种拓扑变体（$ n $ 为有效 pivot 数量）。
- **最终数据集规模**：共生成 **7,918 个唯一标准单元网表**，其中 **2,039 个为不可布线（unroutable）设计**，专门用于训练模型修复坏拓扑。
- **技术节点**：训练基于 **2nm 工艺节点**下的 P&R 结果（使用 NVCell 2），测试则迁移到 **7nm 工艺库**以验证零样本泛化能力。

### 实验设置
- **基础模型**：
  - `Qwen2.5-Coder-3B`
  - `Qwen2.5-Coder-7B`
- **训练框架**：Verl + SGLang，运行于配备四块 80GB A100 GPU 的 NVIDIA DGX Station。
- **训练参数**：批大小 256，学习率 $10^{-6}$，训练 15 轮。
- **奖励模型**：训练一个基于 **PyTorch Geometric (PyG)** 的 **Graph Neural Network (GNN)** 模型作为轻量级代理，预测拓扑的 routability 分数，避免每次调用完整 P&R 流程。

### 评估指标
| 指标 | 描述 |
|------|------|
| **Routable Rate (%)** | 原本不可布线的设计中，经 TOPCELL 优化后变得可布线的比例 |
| **PDA Congestion (Mean)** | NVCell 2 提出的 DTCO 指标，衡量局部拥塞程度；值越低越好 |
| **Optimal Total Cost (OTC)** | SO3-Cell 中使用的综合成本目标，涵盖面积、延迟、功耗等 |
| **End-to-End Runtime** | 总耗时 = LLM 推理时间 ($t_{\text{LLM}}$) + 物理实现时间 ($t_{\text{P\&R}}$) |
| **Speedup (×)** | 相对于基线方法的运行时间加速比 |

### 基线方法对比
- **Foundation Models**：包括 Qwen、CodeLlama、Llama-3.3、DeepSeek-Coder、GPT-5、Claude Opus 等主流代码/通用 LLM，均未经过微调。
- **SOTA 自动化框架**：SO3-Cell —— 当前最先进的联合优化拓扑、布局与布线的标准单元生成系统。
- **消融实验基线**：SFT（监督微调）版本的 TOPCELL，用于验证 GRPO 的优越性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1 和 Table 2）

#### ✅ 在 2nm 上的 routability 优化表现（Table 1）
| 模型 | Routable Rate (%) | PDA Congestion |
|------|--------------------|----------------|
| **TOPCELL-7B** | **77.3** (+24.3 pts) | **3.90** (-8.02%) |
| **TOPCELL-3B** | **69.7** (+14.1 pts) | **4.06** (-6.02%) |
| Qwen2.5-Coder-7B (base) | 53.0 | 4.24 |
| GPT-5 | 56.1 | 4.19 |
| DeepSeek-Coder-33B | 58.6 | 4.10 |

> 💡 即使是较小的 3B 模型也显著优于更大规模的基础模型（如 33B、70B），说明**领域专用微调远胜于单纯扩大模型规模**。

#### ✅ 在 7nm 库生成任务中集成 SO3-Cell 的性能对比（Table 2）
| 指标 | SO3-Cell 平均 | TOPCELL 平均 | 加速比 |
|------|---------------|--------------|--------|
| OTC（布局质量） | 相当（除个别案例外一致） | 相当 | — |
| End-to-End Runtime | 数千秒级 | 数十秒级 | **85.91×** |
| 示例：AOI222_X1_SH（12T） | 35,488 sec | 63.143 sec | **562×** |

> ⚡️ 最快达到 **562 倍加速**，且布局质量完全匹配。

#### ✅ 零样本泛化能力
- 模型仅在 **3-input, 2nm 数据上训练**；
- 成功应用于 **4–6 输入、7nm 工艺**的标准单元（如 NAND4、AOI222_X1_SH）；
- 输出拓扑无需 dummy gate 插入，面积更小，pin access 更好（见 Figure 1）。

#### ✅ 消融实验：GRPO vs SFT（Figure 4）
- **SFT 方法**：快速饱和，最高仅达 **57% Routable Rate**；
- **GRPO 方法**：持续上升，最终突破 **77%**；
- 表明 GRPO 能够通过探索发现超越训练集中“黄金答案”的更优拓扑，而 SFT 仅能模仿已有模式。

---

## 4. 关键结论和发现

### 主要发现
1. **LLM 完全可以胜任标准单元级的晶体管拓扑优化任务**，尤其是在结合物理反馈的情况下。
2. **GRPO 是比 SFT 更适合该任务的对齐方式**，因其支持主动探索与相对优势判断，适用于无单一最优解的复杂设计空间。
3. **TOPCELL 展现出极强的零样本泛化能力**，能够跨越输入维度和技术节点迁移，表明其学到了可迁移的设计原则。
4. **将 LLM 作为“智能引导器”而非“完整生成器”是高效可靠的设计范式**—— LLM 提供高层决策（选 pivot），图算法保证底层正确性。

### 方法的局限性
- **依赖高质量的奖励模型（GNN）**：若 GNN 无法准确预测 routability，则 GRPO 训练会失效。
- **当前仍局限于标准单元级别**：尚未扩展到宏单元或完整模块级布局优化。
- **硬件资源需求较高**：虽然推理快，但 GRPO 训练需要多 GPU 支持，不适合小型团队部署。
- **对极端复杂拓扑可能仍有遗漏风险**：例如 deeply stacked structures 或特殊匹配要求场景。

### 未来工作方向
- 扩展至 **multi-row standard cell layout synthesis** 和 **macro cell optimization**。
- 探索 **multi-agent LLM collaboration**，分别负责 topology、placement、routing 子任务。
- 引入 **testbench-level feedback** 进行闭环优化（如 timing violation、IR drop）。
- 开发 **lightweight on-device deployment version**，便于集成进商业 EDA 工具流。

---

> 🔚 **总结一句话**：  
> **TOPCELL 成功将 LLM 引入标准单元拓扑优化这一经典 NP-hard 问题，通过 GRPO 实现了物理感知的自主拓扑发现，在保持 SOTA 布局质量的前提下实现了两个数量级的速度提升，为 AI-driven EDA 开辟了新路径。**

</details>

---

### 13. [Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](https://arxiv.org/abs/2604.14267)

**Authors**: Junzhe Wang, Zhiheng Xi, yajie yang, Hao Luo, Shihan Dou, Tao Gui, Qi Zhang  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.14267v1  

#### Abstract
Search agents extend Large Language Models (LLMs) beyond static parametric knowledge by enabling access to up-to-date and long-tail information unavailable during pretraining. While reinforcement learning has been widely adopted for training such agents, existing approaches face key limitations: pro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Models (LLMs)** 的搜索代理（Search Agents）在训练过程中面临两大挑战：
- **Outcome Supervision**（结果监督）：仅依赖最终答案的正确性作为奖励信号，导致稀疏奖励，难以进行细粒度的信用分配（credit assignment），无法区分关键检索轮次与冗余操作。
- **Process Supervision**（过程监督）：虽可提供每轮反馈，但通常依赖学习型价值函数（如critic），易受评估噪声影响，训练不稳定。

### 提出的新方法：**CW-GRPO**（Contribution-Weighted GRPO）
提出一种新型强化学习框架 **Contribution-Weighted Group Relative Policy Optimization (CW-GRPO)**，其核心思想是：
- **不直接优化过程奖励**，而是将过程级信号用于**重分配（reallocation）轨迹级优势函数（advantage）**。
- 利用一个冻结的 **LLM Judge** 对每个搜索轮次评估两个维度：
  - **Retrieval Utility**（检索效用）：是否获取了新颖且相关的信息。
  - **Reasoning Correctness**（推理正确性）：推理链是否逻辑一致并支持后续决策。
- 将这两个信号合成为**贡献分数（contribution score）**，作为缩放因子对标准 **GRPO** 中的 outcome advantage 进行加权重分配。

### 相比现有方法的优势
- **稳定性高**：保留了 GRPO 的 group-relative 优势计算机制，避免训练中价值函数估计偏差。
- **信用分配更精细**：实现**轮级（round-level）信用分配**，放大高质量轮次的学习信号，抑制低质量轮次的影响。
- **无需训练额外模型**：使用预训练 LLM Judge 而非训练 PRM（Process Reward Model），降低标注成本与泛化风险。
- **设计鲁棒**：对失败轨迹采用均匀权重，防止因模糊归因而引入噪声。

---

## 2. 核心实验方法和设置

### 数据集
实验涵盖两类知识密集型任务：
- **单跳问答（General QA）**：
  - Natural Questions (NQ)
  - TriviaQA
  - PopQA
- **多跳问答（Multi-Hop QA）**：
  - HotpotQA
  - 2WikiMultiHopQA
  - Musique
  - Bamboogle

此外，引入更具挑战性的测试集：
- **AgentGym-SearchQA-test**：400个样本，筛选自 Qwen2.5-72B-Instruct 模型也无法回答的问题，构成“hard-case”分布。

### 实验设置与评估指标
- **基础模型**：Qwen3-8B 和 Qwen3-1.7B
- **严格约束**：禁止使用参数化知识（parametric knowledge），强制依赖外部检索。
- **评估指标**：Avg@4 Exact Match (EM)，即每个问题采样4条响应取平均EM得分。
- **训练框架**：veRL + SGLang 推理引擎，最大上下文长度 9192 tokens，最多10轮交互。
- **检索系统**：基于 2018 Wikipedia dump，使用 E5 作为 dense retriever，每次返回3个文档。

### 基线方法对比
分为两类：
- **Outcome-Supervised Methods**：
  - Search-R1-PPO
  - Search-R1-GRPO
- **Process-Supervised Methods**：
  - R3-RAG
  - MT-PPO

同时对比多个强闭源模型（如 GPT-5.1、Gemini-3-Pro-Preview、Qwen3-Max）和开源模型（如 DeepSeek-V3.2、Qwen3-32B）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Avg@4 EM）
| 方法 | Qwen3-8B 总体 | Qwen3-1.7B 总体 |
|------|----------------|------------------|
| Search-R1-GRPO | 29.88 | 21.00 |
| **CW-GRPO** | **31.38** (+5.0%) | **22.31** (+6.3%) |

> ✅ 在两种规模模型上均取得显著提升，尤其在小模型上增益更大。

### 与基线方法的对比结果
- **全面超越所有基线**：在相同 backbone 下，CW-GRPO 是唯一在所有子任务上都优于 outcome- 和 process-supervised 方法的方案。
- **优于大型开源模型**：即使在 Qwen3-8B 上，CW-GRPO 也超过了 Qwen3-32B 和 DeepSeek-V3.2-Thinking。
- **多跳任务增益最明显**：在 HotpotQA、Musique、Bamboogle 上均有显著提升，表明其在长程推理和证据聚合方面更强。

### 消融实验结果
#### （1）Sharpness 参数 $\alpha$ 影响（控制优势集中程度）
| $\alpha$ | 总体 EM |
|---------|--------|
| 0（均匀） | 29.88 |
| 1       | 29.69 |
| 3       | 30.63 |
| 5       | 30.50 |
| ∞（完全集中于高贡献轮次） | **31.38** |

> 🔍 结果显示：将学习信号集中在少数高质量轮次效果最佳，说明成功轨迹中贡献高度集中。

#### （2）去除任一监督信号的影响
| 方法 | 总体 EM |
|------|--------|
| CW-GRPO（完整） | 31.38 |
| w/o Retrieval Utility | 29.19 |
| w/o Reasoning Correctness | 28.56 |

> ❌ 移除任一信号都会导致性能下降，证明两者缺一不可。

---

## 4. 关键结论和发现

### 主要发现
1. **搜索行为的成功源于少数关键轮次**：实验证明，成功的搜索轨迹中，只有约 13.8% 的轮次被判定为“有贡献”，说明任务进展是非均匀累积的。
2. **Credit Assignment 应聚焦于高质量轮次**：通过贡献加权重分配优势函数，能有效引导策略学习真正推动任务完成的行为。
3. **LLM Judge 可靠且实用**：经过校准后，LLM Judge 与人类标注一致性达 **95%**，可用于稳定生成过程信号。
4. **CW-GRPO 具有良好扩展性**：在不同模型尺寸下均表现稳健，尤其在小模型上提升更显著。

### 方法的局限性
1. **仅对成功轨迹进行细粒度归因**：失败轨迹仍采用统一权重，未能建模“为何失败”的中间原因。
2. **表达能力受限于二值门控机制**：采用 $p = u \land v$ 的合取形式虽增强鲁棒性，但也限制了贡献表示的丰富性。
3. **依赖外部 LLM Judge**：引入额外推理开销（训练时间增加约33%），且性能受 judge 质量影响。
4. **不支持 token-level 监督**：目前仅适用于轮级抽象，难以应用于更细粒度的任务分解。

### 未来工作方向
- 扩展至其他多步代理任务（如 WebShop、数学推理），探索通用 credit assignment 框架。
- 设计轻量化 judge（如蒸馏版或规则启发式），提升效率与部署可行性。
- 探索失败轨迹中的归因机制，例如通过反事实分析识别错误根源。
- 将方法推广到 step-level 或 token-level 的结构化输出任务中。

---

> 📌 **总结一句话**：  
> **CW-GRPO 通过 LLM Judge 评估每轮贡献，并以此动态重分配 GRPO 的优势函数，在保持训练稳定性的同时实现了精准的轮级信用分配，显著提升了 LLM-based Search Agents 的性能。**

</details>

---

### 14. [Quantization of Spiking Neural Networks Beyond Accuracy](https://arxiv.org/abs/2604.14487)

**Authors**: Evan Gibson Smith, Jacob Whitehill, Fatemeh Ganji  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.14487v1  

#### Abstract
Quantization is a natural complement to the sparse, event-driven computation of Spiking Neural Networks, reducing memory bandwidth and arithmetic cost for deployment on resource-constrained hardware. However, existing SNN quantization evaluation focuses almost exclusively on accuracy, overlooking wh...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Quantization of Spiking Neural Networks Beyond Accuracy**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **Spiking Neural Networks (SNNs)** 量化研究几乎完全聚焦于**准确率（accuracy）的保持**，而忽略了量化是否改变了网络的**放电行为（spiking dynamics）**。然而，SNN 的核心优势在于其稀疏、事件驱动的计算特性，而这些特性直接由神经元的放电分布（如放电率、死神经元比例、时间模式等）决定。

论文指出：  
> **两个量化模型可能具有相近的 accuracy，但其 firing distribution 可能差异巨大**，这会直接影响部署时的内存占用、状态存储开销和事件处理负载。

因此，仅用 accuracy 评估 SNN 量化是**不充分且具有误导性的**。

---

### **提出了什么新方法或新思路**
为弥补这一评估缺口，作者提出：

- **引入 Earth Mover's Distance (EMD)** 作为量化 SNN 的诊断指标，用于衡量量化模型与全精度（FP32）基准之间在**每神经元放电率分布**上的差异。
  - EMD 是一种对非正态、多峰、重尾分布敏感的距离度量，能够捕捉均值、方差、双峰性等变化，远优于简单的 scalar 统计（如平均放电率）。

- 将 EMD 系统应用于：
  - 不同量化方法（uniform vs. learned）
  - 不同比特宽度（2~8 bits）
  - 权重（weight）与膜电位（membrane potential）量化
  - 不同 clipping 范围（[-1,1] vs. [-10,10]）

- **首次将 LQ-Net 风格的 learned quantization 应用于 SNNs**，以探索其在保持放电行为方面的潜力。

---

### **相比现有方法的优势**
| 方面 | 传统方法 | 本文方法 |
|------|--------|---------|
| **评估标准** | 仅 accuracy | accuracy + EMD（行为保真度） |
| **量化方法分析** | 忽略 firing 分布变化 | 揭示相同 accuracy 下的行为差异 |
| **量化策略** | 多为 uniform quantization | 引入并验证 LQ-Net 在 SNN 中的有效性 |
| **适用性** | 适用于 ANN | 针对 SNN 特有的动态特性设计 |

> ✅ **核心优势**：揭示了“accuracy-preserving ≠ behavior-preserving”，并提供了可量化的工具（EMD）来检测这种差距。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **CIFAR-10**
- **CIFAR-100**

均为图像分类任务，适合中等规模 SNN 模型训练与系统化扫描。

---

### **模型架构**
- **SEW-ResNet8 和 SEW-ResNet18**
  - 使用 ADD-type 残差连接，支持稳定深层 SNN 训练
  - 采用 Leaky Integrate-and-Fire (LIF) 神经元，τ=2.0，阈值=1.0，重置=0.0
  - 时间步数 T = 4，使用 direct encoding

---

### **量化设置**
#### **量化对象**
1. **Weight Quantization**：网络权重
2. **Membrane Potential Quantization**：LIF 神经元的膜电位（递归状态变量）

#### **量化方法**
| 方法 | 描述 |
|------|------|
| **Uniform Quantization (STE)** | 对称均匀量化，使用 Straight-Through Estimator 进行梯度近似 |
| **Learned Quantization (LQ-Net)** | 学习一组 basis vectors，权重表示为二进制组合：<br> $ \hat{W} = \sum_{k=1}^n b_k \cdot q_k,\ q_k \in \{0,1\} $<br>通过闭式误差最小化联合优化 basis 和权重 |
| **Clipping Ranges** | [-1,1]（窄范围）、[-10,10]（宽范围）用于权重；<br>[−1.0,1.0]（有符号）、[0.0,1.0]（无符号）用于膜电位 |

#### **比特宽度**
- 2, 4, 6, 8 bits

---

### **评估指标**
| 指标 | 说明 |
|------|------|
| **Accuracy (%)** | 主要任务性能指标 |
| **Mean Firing Rate** | 所有通道平均放电率，反映整体活跃程度 |
| **Earth Mover's Distance (EMD)** | 每层 firing rate 分布与 FP32 基准之间的 Wasserstein-1 距离，衡量分布偏移 |
| **Dead Neurons (%)** | 放电率 < 0.05 的神经元占比，反映功能丧失 |
| **Saturated Neurons** | 高放电率群体（文中未单独列出，但分布图可见） |

---

### **基线方法对比**
- **Uniform Quantization (STE)**：主流 baseline
- **LQ-Net**：首次在 SNN 中应用，作为 learned quantization 的代表
- **不同 clipping 范围**：测试量化鲁棒性
- **Signed vs. Unsigned Membrane Quantization**：探索数值范围影响

---

## **3. 主要实验结果和性能指标**

### **关键性能数据汇总**

#### **表1：Weight Quantization on CIFAR-100 (SEW-ResNet18)**

| 方法 | Bits | Acc (%) | EMD | Dead (%) |
|------|------|--------|-----|----------|
| FP32 (baseline) | – | 75.2 | 0.0 | ~66 |
| Uniform [-1,1] | 6 | 74.9 | **0.0077** | 66.5 |
| Uniform [-10,10] | 6 | 73.7 | **0.0136** | 59.8 |
| LQ-Net [-1,1] | 6 | 75.3 | 0.0073 | 68.7 |
| LQ-Net [-10,10] | 6 | 75.3 | **0.0072** | 71.4 |

> 🔍 **观察**：尽管 accuracy 差异仅 1.2%，但 wide-clipped uniform 的 EMD 是 narrow-clipped 的 **1.76倍**，表明其 firing behavior 显著退化。

#### **极端案例（2-bit, wide clipping）**
| 方法 | Acc (%) | EMD |
|------|--------|-----|
| Uniform [-10,10] | 51.6 | 0.1206 |
| LQ-Net [-10,10] | 74.4 | **0.0121** |

> ⚠️ **EMD 差距达 10 倍以上！** 表明 uniform 严重破坏放电分布，而 LQ-Net 成功保留。

---

### **与基线方法的对比结果**

| 对比维度 | 结果 |
|--------|------|
| **Uniform vs. LQ-Net (weight)** | LQ-Net 在所有配置下均显著降低 EMD，尤其在低 bit-width 和 wide clipping 下优势明显；accuracy 更高且更稳定 |
| **Narrow vs. Wide Clipping (uniform)** | Wide clipping 导致更高的 EMD 和 dead neuron 比例，即使 accuracy 接近，firing behavior 仍劣化 |
| **LQ-Net vs. Clipping Sensitivity** | LQ-Net 几乎不受 clipping 范围影响，因其 level 自适应学习 |
| **Membrane Quantization** | LQ-Net 无法成功训练（训练崩溃），只能使用 uniform；需 ≥4 bits 才能有效保持 behavior |

---

### **消融实验结果**
- **Clipping Range 消融**：
  - Wide clipping 在 uniform 量化中导致显著 EMD 上升，但在 LQ-Net 中无此现象 → 证明 learned quantization 具有更强鲁棒性。
- **Bit-width 消融**：
  - 随着 bit-width 增加，accuracy 恢复快于 EMD → **accuracy 回升不代表 behavior 完全恢复**
  - EMD 曲线显示 dynamics lag behind accuracy recovery
- **Membrane Range 消融**：
  - **Unsigned ([0,1])** 在 2-bit 下优于 signed ([−1,1])，both in accuracy and EMD
    - CIFAR-100 ResNet18 @2bit: 71.9% vs. 67.1%，EMD=0.010 vs. 0.014
  - 推测原因：膜电位通常非负，unsigned 更符合实际分布

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **Accuracy is not enough**: 相同 accuracy 的量化模型可能具有截然不同的 firing distribution，仅靠 accuracy 无法识别这种差异。
2. ✅ **EMD is a powerful diagnostic tool**: 能有效捕捉 firing rate 分布的形状变化（如双峰、方差漂移、死神经元增加），是对 scalar 指标的有力补充。
3. ✅ **Uniform quantization induces distributional drift**, especially under:
   - Low bit-width (< 4 bits)
   - Wide clipping ranges
   - Membrane quantization
4. ✅ **LQ-Net preserves both accuracy and dynamics**:
   - 在 weight quantization 中表现优异，even at 2-bit
   - 对 clipping range 不敏感，更具实用性
5. ✅ **Membrane quantization is more challenging**:
   - 错误会在 timestep 间累积（recurrent dynamics）
   - 需要 ≥4 bits 才能可靠保持 behavior
   - Unsigned range ([0,1]) 更优，更适合低比特场景
   - LQ-Net 当前不可行，需专门设计

---

### **方法的局限性**
- **LQ-Net 无法扩展到 membrane quantization**：由于训练不稳定，尚未解决。
- **EMD 是 layer-wise 计算**，未建模跨层或时空 firing pattern 变化。
- 实验集中在 CIFAR 数据集和 SEW-ResNet 架构，泛化性有待验证。
- 未提供硬件部署的实际能耗或延迟测量，仅为 behavior 分析。

---

### **未来工作方向**
1. **Develop learned quantization for membrane potentials**：针对递归状态设计稳定的训练机制。
2. **Extend EMD to temporal spike trains**：不仅看 firing rate，也考虑 spike timing 和 burst 模式。
3. **Hardware-aware quantization co-design**：结合 EMD 与 energy/latency constraint 进行联合优化。
4. **Apply EMD to other SNN compression techniques**：如剪枝（pruning）、蒸馏（distillation）等，统一 behavior-preserving 评估框架。

---

> 📌 **一句话总结**：  
> 本文揭示了 SNN 量化中“accuracy-preserving ≠ behavior-preserving”的关键问题，提出使用 **EMD** 作为 firing distribution 偏移的诊断工具，并证明 **LQ-Net** 能有效保持放电行为，而传统 uniform 量化即使 accuracy 恢复也可能严重破坏网络动态特性。建议将 **behavior preservation** 作为 SNN 量化的新评估准则。

</details>

---

### 15. [Physics-Informed Machine Learning for Pouch Cell Temperature Estimation](https://arxiv.org/abs/2604.14566)

**Authors**: Zheng Liu  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.14566v1  

#### Abstract
Accurate temperature estimation of pouch cells with indirect liquid cooling is essential for optimizing battery thermal management systems for transportation electrification. However, it is challenging due to the computational expense of finite element simulations and the limitations of data-driven ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
- 针对采用**间接液冷**（indirect liquid cooling）的**pouch cell**电池在热管理系统中的**温度分布估计难题**。  
- 传统方法如有限元仿真（FE）计算成本高，难以用于迭代优化；而纯**data-driven模型**依赖大量训练数据，泛化能力差，且可能违反物理规律，导致不可靠预测。

### **提出了什么新方法或新思路**
- 提出了一种**Physics-Informed Machine Learning (PIML)** 框架，用于高效、可靠地估计pouch cell的**稳态温度分布**。
- 将**控制热传导的偏微分方程**（PDE）直接嵌入神经网络的损失函数中，构建**Physics-Informed Neural Network (PINN)**。
- 结合了**数据驱动学习**与**物理守恒定律**，实现少数据条件下的高保真建模。

### **相比现有方法的优势**
- **更高的预测精度**：尤其在远离冷却通道的区域表现更优。
- **更快的收敛速度**：仅需10个epoch即显著优于data-driven模型。
- **更强的泛化能力**：即使训练数据稀疏，仍能保持物理一致性。
- **更低的计算开销**：作为**surrogate model**，可替代高成本FE模拟用于设计优化。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 数据由**Finite Difference Method (FDM)** 仿真生成，共包含 **100个不同冷却通道几何构型**的样本。
- 每个样本包括：
  - 输入：二值化的冷却通道掩码 $ M(x,y) \in \{0,1\} $
  - 输出：对应的稳态温度场 $ T(x,y) $

- 电池尺寸简化为二维平面 $154 \times 203$ mm²，空间分辨率 $1\,\text{mm}$，网格大小 $154 \times 203$。

### **实验设置和评估指标**
- **训练/测试划分**：80%训练集（80 samples），20%测试集（20 samples）
- **输入预处理**：温度场标准化（基于训练集均值 $\mu_{\text{train}}$ 和标准差 $\sigma_{\text{train}}$）
- **输出后处理**：预测结果反标准化至原始温度单位（°C）

#### **评估指标**
- **Mean Squared Error (MSE)**：主评价指标，衡量预测温度场与FDM真值之间的差异。
- **Loss components**：
  - $L_{\text{MSE}}$: 数据拟合误差
  - $L_{\text{PDE}}$: PDE残差损失（确保满足热传导方程）
  - $L_{\text{BC}}$: 边界条件损失（绝热边界约束）

#### **优化目标（Total Loss）**
$$
\mathcal{L}_{\text{total}} = w_1 \cdot \mathcal{L}_{\text{MSE}} + w_2 \cdot \mathcal{L}_{\text{PDE}} + w_3 \cdot \mathcal{L}_{\text{BC}}
$$

---

### **基线方法对比**
| 方法 | 类型 | 描述 |
|------|------|------|
| **Data-Driven Model** | 全卷积网络（FCN） | 以通道掩码为输入，直接回归温度场，仅最小化MSE损失 |
| **PIML Model** | Physics-Informed CNN | 同样结构，但联合优化MSE + PDE + BC损失 |

> 注：两种模型均使用相同网络架构（ReLU激活前两层，最后一层线性）、Adam优化器（learning rate = 0.001），训练100 epochs。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- 在**仅训练10个epoch**时：
  - **PIML模型 MSE = 5.66**
  - **Data-Driven模型 MSE = 11.12**
  - → **相对提升达49.1%**

- 经过完整100 epoch训练后，PIML持续保持更低的测试MSE，表明其不仅收敛快，而且最终精度更高。

### **与基线方法的对比结果**
- **可视化对比（Fig. 2）显示**：
  - Data-Driven模型在远离冷却通道区域出现明显偏差（如高温区过估或欠估）。
  - PIML模型能更准确捕捉温度梯度，尤其是在**非直接受冷却影响的区域**。
- **误差图分析**：PIML的整体误差分布更均匀且幅值更低。

### **消融实验结果（隐含分析）**
- 虽未明确列出消融实验表格，但从损失函数设计可知：
  - 引入 $L_{\text{PDE}}$ 和 $L_{\text{BC}}$ 显著提升了模型的**物理一致性**与**外推能力**。
  - 即使在少量数据下，PIML也能通过物理约束“补全”缺失的信息，体现其**强归纳偏置**（strong inductive bias）优势。

---

## **4. 关键结论和发现**

### **论文的主要发现**
- **PIML显著优于纯data-driven方法**：在相同训练条件下，实现更快收敛与更高精度，尤其适用于**数据稀缺场景**。
- **物理嵌入提升可靠性**：通过将热传导PDE和边界条件编码进损失函数，确保预测结果符合基本物理规律，避免非物理解。
- **适用于BTMS设计优化**：所提框架可作为高效的**surrogate model**，支持快速评估多种冷却通道设计方案，降低对昂贵仿真的依赖。

### **方法的局限性**
- 当前研究基于**二维稳态假设**，忽略了瞬态效应和三维结构的影响。
- 冷却通道建模进行了简化（如固定厚度冷板、理想化对流系数），实际系统可能存在更多复杂因素（如流动不均、接触热阻等）。
- 模型尚未经过真实实验数据验证，仍处于**仿真驱动阶段**。

### **未来工作方向**
- 扩展PIML框架至**瞬态热行为建模**与**三维几何建模**。
- 联合优化**冷却通道设计参数**（如宽度、间距、布局），实现端到端的设计探索。
- 开展**实验验证**，采集真实pouch cell温度数据，进一步检验模型的预测能力和鲁棒性。
- 探索**multi-fidelity modeling**策略，融合低精度仿真与高精度实测数据，提升模型实用性。

---

> ✅ **总结一句话**：  
> 本文提出的PIML框架通过融合物理定律与深度学习，在极少数据下实现了对pouch cell温度场的高精度、快速估计，展现出在电池热管理系统（BTMS）设计优化中的巨大潜力。

</details>

---

### 16. [Constraint-based Pre-training: From Structured Constraints to Scalable Model Initialization](https://arxiv.org/abs/2604.14769)

**Authors**: Fu Feng, Yucheng Xie, Ruixiao Shi, Jing Wang, Xin Geng  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.14769v1  

#### Abstract
The pre-training and fine-tuning paradigm has become the dominant approach for model adaptation. However, conventional pre-training typically yields models at a fixed scale, whereas practical deployment often requires models of varying sizes, exposing its limitations when target model scales differ ...

---

### 17. [Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](https://arxiv.org/abs/2604.13488)

**Authors**: Ziwei Wang, Junjie Zheng, Leyang Yang, Sheng Zhou, Xiaoxuan Tang, Zhouhua Fang, Zhiwei Liu, Dajun Chen, Yong Li, Jiajun Bu  
**Category**: cs.AI  
**Published**: 2026-04-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.13488v1  

#### Abstract
Autonomous Graphical User Interface (GUI) agents powered by Multimodal Large Language Models (MLLMs) enable digital automation on end-user devices. While scaling both parameters and data has yielded substantial gains, advanced methods still suffer from prohibitive deployment costs on resource-constr...

---

### 18. [SeaAlert: Critical Information Extraction From Maritime Distress Communications with Large Language Models](https://arxiv.org/abs/2604.14163)

**Authors**: Tomer Atia, Yehudit Aperstein, Alexander Apartsin  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.14163v1  

#### Abstract
Maritime distress communications transmitted over very high frequency (VHF) radio are safety-critical voice messages used to report emergencies at sea. Under the Global Maritime Distress and Safety System (GMDSS), such messages follow standardized procedures and are expected to convey essential deta...

---

### 19. [Beyond Importance Sampling: Rejection-Gated Policy Optimization](https://arxiv.org/abs/2604.14895)

**Authors**: Ziwu Sun, Zhen Gao, Jiyong Zhang, Jiaheng Li  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.14895v1  

#### Abstract
We propose a new perspective on policy optimization: rather than reweighting all samples by their importance ratios, an optimizer should select which samples are trustworthy enough to drive a policy update. Building on this view, we introduce Rejection-Gated Policy Optimization (RGPO), which replace...

---

### 20. [How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations](https://arxiv.org/abs/2604.15273)

**Authors**: Nouhaila Innan, Antonello Rosato, Alberto Marchisio, Muhammad Shafique  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.15273v1  

#### Abstract
Node embeddings act as the information interface for graph neural networks, yet their empirical impact is often reported under mismatched backbones, splits, and training budgets. This paper provides a controlled benchmark of embedding choices for graph classification, comparing classical baselines w...

---

### 21. [Reward Design for Physical Reasoning in Vision-Language Models](https://arxiv.org/abs/2604.13993)

**Authors**: Derek Lilienthal, Manisha Mukherjee, Sameera Horawalavithana  
**Category**: cs.AI  
**Published**: 2026-04-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.13993v1  

#### Abstract
Physical reasoning over visual inputs demands tight integration of visual perception, domain knowledge, and multi-step symbolic inference. Yet even state-of-the-art Vision Language Models (VLMs) fall far short of human performance on physics benchmarks. While post-training algorithms such as Supervi...

---

### 22. [Hierarchical Retrieval Augmented Generation for Adversarial Technique Annotation in Cyber Threat Intelligence Text](https://arxiv.org/abs/2604.14166)

**Authors**: Filippo Morbiato, Markus Keller, Priya Nair, Luca Romano  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.14166v1  

#### Abstract
Mapping Cyber Threat Intelligence (CTI) text to MITRE ATT\&amp;CK technique IDs is a critical task for understanding adversary behaviors and automating threat defense. While recent Retrieval-Augmented Generation (RAG) approaches have demonstrated promising capabilities in this domain, they fundament...

---

### 23. [CROP: Token-Efficient Reasoning in Large Language Models via Regularized Prompt Optimization](https://arxiv.org/abs/2604.14214)

**Authors**: Deep Shah, Sanket Badhe, Nehal Kathrotia, Priyanka Tiwari  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.14214v1  

#### Abstract
Large Language Models utilizing reasoning techniques improve task performance but incur significant latency and token costs due to verbose generation. Existing automatic prompt optimization(APO) frameworks target task accuracy exclusively at the expense of generating long reasoning traces. We propos...

---

### 24. [SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](https://arxiv.org/abs/2604.14672)

**Authors**: Binxian Su, Haoye Lou, Shucheng Zhu, Weikang Wang, Ying Liu, Dong Yu, Pengyuan Liu  
**Category**: cs.CL  
**Published**: 2026-04-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.14672v1  

#### Abstract
Large language models (LLMs) are being increasingly used in urban planning, but since gendered space theory highlights how gender hierarchies are embedded in spatial organization, there is concern that LLMs may reproduce or amplify such biases. We introduce SPAGBias - the first systematic framework ...

---

### 25. [Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines](https://arxiv.org/abs/2604.15186)

**Authors**: Marcel Wagenl\"ander, Otto White, Britannio Jarrett, Pedro Silvestre, Yanda Tao, Guo Li, Huanzhou Zhu, Ll\'uis Vilanova, Peter Pietzuch  
**Category**: cs.DC  
**Published**: 2026-04-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.15186v1  

#### Abstract
Agentic workflows carry out complex tasks by orchestrating multiple large language models (LLMs) and tools. Serving such workflows at a target throughput with low latency is challenging because they can be defined using arbitrary agentic frameworks and exhibit unpredictable execution times: executio...

---

### 26. [Portfolio Optimization Proxies under Label Scarcity and Regime Shifts via Bayesian and Deterministic Students under Semi-Supervised Sandwich Training](https://arxiv.org/abs/2604.14206)

**Authors**: Adhiraj Chattopadhyay  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.14206v1  

#### Abstract
This paper proposes a machine learning assisted portfolio optimization framework designed for low data environments and regime uncertainty. We construct a teacher student learning pipeline in which a Conditional Value at Risk (CVaR) optimizer generates supervisory labels, and neural models (Bayesian...

---

### 27. [Explainable Graph Neural Networks for Interbank Contagion Surveillance: A Regulatory-Aligned Framework for the U.S. Banking Sector](https://arxiv.org/abs/2604.14232)

**Authors**: Mohammad Nasir Uddin  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.14232v1  

#### Abstract
The Spatial-Temporal Graph Attention Network (ST-GAT) framework was created to serve as an explainable GNN-based solution for detecting bank distress early warning signs and for conducting macro-prudential surveillance of the interbank system in the United States. The ST-GAT framework models 8,103 F...

---

### 28. [Catching Every Ripple: Enhanced Anomaly Awareness via Dynamic Concept Adaptation](https://arxiv.org/abs/2604.14726)

**Authors**: Jiaqi Zhu, Shaofeng Cai, Jie Chen, Fang Deng, Beng Chin Ooi, Wenqiao Zhang  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.14726v1  

#### Abstract
Online anomaly detection (OAD) plays a pivotal role in real-time analytics and decision-making for evolving data streams. However, existing methods often rely on costly retraining and rigid decision boundaries, limiting their ability to adapt both effectively and efficiently to concept drift in dyna...

---

### 29. [Learning Ad Hoc Network Dynamics via Graph-Structured World Models](https://arxiv.org/abs/2604.14811)

**Authors**: Can Karacelebi, Yusuf Talha Sahin, Elif Surer, Ertan Onur  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.14811v1  

#### Abstract
Ad hoc wireless networks exhibit complex, innate and coupled dynamics: node mobility, energy depletion and topology change that are difficult to model analytically. Model-free deep reinforcement learning requires sustained online interaction whereas existing model based approaches use flat state rep...

---

### 30. [Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization](https://arxiv.org/abs/2604.14853)

**Authors**: Zhiyuan Zhai, Bingcong Li, Bingnan Xiao, Ming Li, Xin Wang  
**Category**: cs.LG  
**Published**: 2026-04-17  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.14853v1  

#### Abstract
Test-time compute scaling, the practice of spending extra computation during inference via repeated sampling, search, or extended reasoning, has become a powerful lever for improving large language model performance. Yet deploying these techniques under finite inference budgets requires a decision t...

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
