# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-05-21 08:54:01 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs](https://arxiv.org/abs/2605.20315)

**Authors**: Haiquan Lu, Zigeng Chen, Gongfan Fang, Xinyin Ma, Xinchao Wang  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 14.5  
**Type**: new  
**ArXiv ID**: 2605.20315v1  

#### Abstract
LLM agents have recently emerged as a powerful paradigm for solving complex tasks through planning, tool use, memory retrieval, and multi-step interaction. However, these agentic workflows often introduce substantial input-side overhead, making the compute-intensive prefilling stage a key bottleneck...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Mix-Quant: Quantized Prefilling, Precise Decoding for Agentic LLMs**

---

## 1. **论文的主要贡献和创新点**

### ✅ 解决了什么问题
- **问题背景**：LLM agents 在执行复杂任务时（如工具调用、记忆检索、多步交互）会产生大量输入上下文，导致 **prefilling 阶段成为计算瓶颈**。该阶段需处理长序列且计算密集，而传统量化方法（如 W4A4 或 FP4）若统一应用于整个推理过程（prefill + decode），会导致 **decoding 阶段误差累积**，显著降低生成质量。
- **核心矛盾**：如何在加速 prefilling 的同时，避免 decoding 质量下降？

### ✅ 提出了什么新方法或新思路
- **提出 Mix-Quant**：一种 **phase-aware 量化框架**，对不同推理阶段采用不同的精度策略：
  - **Prefilling 阶段**：使用高吞吐的 **NVFP4 W4A4 量化**（低比特权重与激活值），以大幅提升计算效率。
  - **Decoding 阶段**：保持 **BF16 高精度**，防止 token-level 错误传播和雪崩效应。
- **硬件协同设计**：利用 NVIDIA Blackwell 架构原生支持的 **NVFP4** 格式，实现高效的低比特矩阵乘法（GEMM），充分发挥硬件加速潜力。

### ✅ 相比现有方法的优势
| 方法 | 缺陷 | Mix-Quant 改进 |
|------|------|----------------|
| Uniform PTQ / W4A4 | 全流程低精度 → decoding 误差累积严重 | 分阶段优化，decoding 保精度 |
| Weight-only Quantization (e.g., GPTQ, AWQ) | 仅压缩权重，prefilling 加速有限（因仍需高精度 activation 计算） | W4A4 全路径量化 → 显著提升 prefill 吞吐 |
| 单一 FP4 推理 | 性能下降明显，尤其在 long-trajectory 任务中 | 通过分离 phase，兼顾速度与稳定性 |

> 🔑 **核心思想**：**“Quantized Prefilling, Precise Decoding”** —— 利用 prefilling 的并行性和冗余性进行激进量化，保留 decoding 的数值稳定性。

---

## 2. **核心实验方法和设置**

### ✅ 使用的数据集
#### **Agentic Benchmarks（代理类任务）**
- **BFCL v4**：评估工具调用与函数调用能力
- **LongMemEval**：测试长期交互记忆管理能力
- **T2-bench**：评估状态化对话中的 agent 行为

#### **Long-Context & Reasoning Benchmarks（长上下文与推理）**
- **LongBench-V2**, **AA-LCR**：长文档理解、摘要与推理
- **Math500**, **AIME24/AIME25**：数学推理任务，检验多步逻辑能力

> 所有任务均涉及 **input-heavy 工作负载**，即输入远长于输出。

### ✅ 实验设置和评估指标
- **模型**：
  - Qwen3-8B, Qwen3.5-9B
  - Gemma-4-26B-A4B-it, Gemma-4-31B-it
- **上下文长度**：
  - Gemma-4: 256K
  - Qwen3.5: 262K
  - Qwen3-8B: 通过 YaRN 扩展至 131K
- **硬件平台**：
  - NVIDIA RTX 5090 和 B200 GPU（支持 Blackwell NVFP4）
- **服务框架**：
  - 基于 **vLLM**，使用 FlashInfer 和 NVFP4 GEMM 内核
  - 采用 **prefill-decode disaggregation** 架构，通过 NIXL 进行 KV-cache 传输
- **评估方式**：
  - 每个 benchmark 独立运行三次取平均分
  - 主要指标：**任务准确率 / 得分（Accuracy/Scores）** 和 **end-to-end prefill 延迟**

### ✅ 基线方法对比
| 方法 | 描述 |
|------|------|
| **BF16 Baseline** | 原始高精度模型，无量化 |
| **Uniform NVFP4** | 整个推理流程（prefill + decode）全部使用 NVFP4 W4A4 量化 |
| **Mix-Quant (Ours)** | Prefill 用 NVFP4，decode 用 BF16 |
| **P16D4 (Ablation)** | Prefill 用 BF16，decode 用 FP4（反向对照） |

---

## 3. **主要实验结果和性能指标**

### ✅ 关键性能数据

#### 📊 表格 1 & 2：Agentic 与 Long-Context 任务表现（平均得分）

| Model | BF16 | Uniform NVFP4 | **Mix-Quant** |
|-------|------|---------------|--------------|
| Qwen3-8B | 42.85 | 38.64 (-4.21) | **41.45 (-1.4)** |
| Qwen3.5-9B | 77.31 | 70.37 (-6.94) | **74.68 (-2.63)** |
| Gemma-4-26B-A4B-it | 66.07 | 55.95 (-10.12) | **61.67 (-4.4)** |
| Gemma-4-31B-it | 77.63 | 76.21 (-1.42) | **77.14 (-0.49)** |

> 💡 **结论**：Mix-Quant 显著恢复了 uniform NVFP4 丢失的性能，接近甚至逼近 BF16 水平。

#### 📈 Prefilling Speedup（图 4）
- 在 RTX 5090 上，Mix-Quant 实现：
  - **最高达 3.74× 的 prefill 速度提升**
  - 平均 **2–3× speedup**（跨序列长度和 batch size）
- 尤其在 Qwen3-8B 上增益更明显，说明小模型受益更大。

#### 🧪 消融实验（表 3）：Phase-wise Quantization Ablation
| 方法 | Qwen3-8B Avg | Gemma-4-26B Avg |
|------|-------------|----------------|
| BF16 | 40.42 | 63.81 |
| Uniform NVFP4 | 33.59 | 53.34 |
| P16D4 (Decode-only quantized) | 36.74 | 59.85 |
| **Mix-Quant (Prefill-only quantized)** | **38.32** | **60.18** |

> 🔍 **发现**：
- 仅量化 decoding（P16D4）也会造成较大性能损失
- 仅量化 prefill（Mix-Quant）是更优选择，验证了 “decoding 更敏感” 的假设

---

## 4. **关键结论和发现**

### ✅ 主要发现
1. **Agentic workflows 是 input-heavy 的**：
   - 输入 token 数量通常是输出的数十倍，使得 prefill 成为主要瓶颈。
2. **Prefilling 可安全量化**：
   - 输入固定、并行处理、注意力集中于少数 token（fig. 3 显示 top 3.125% tokens 占据 95.8% 注意力），因此量化误差影响可控。
3. **Decoding 对量化极其敏感**：
   - 自回归生成中，单个错误 token 可引发后续预测的“雪崩效应”，导致任务失败。
4. **Phase-aware 设计至关重要**：
   - 统一量化策略不适用于 agentic 场景；应根据阶段特性差异化优化。
5. **NVFP4 + BF16 混合路径可行且高效**：
   - 利用 disaggregated serving 架构可无缝部署双路径，无需频繁 kernel 切换或格式转换。

### ⚠️ 方法的局限性
- **依赖 Blackwell 硬件**：NVFP4 是 Blackwell 新引入格式，当前仅在最新 GPU 上可用，限制了部署范围。
- **KV-cache 精度兼容性要求**：prefill 输出的 KV-cache 必须与 decode 路径兼容（dtype、layout），需要 careful engineering。
- **不适用于极短上下文场景**：当 prefill 开销较小时，收益不显著。

### 🔮 未来工作方向
- **扩展到其他 phase-aware 技术**：如将 speculative decoding、MoE routing 等也纳入 phase-specific 优化。
- **动态切换机制**：根据上下文长度或任务类型自动决定是否启用 Mix-Quant。
- **支持更多 low-precision 格式**：探索适用于 older GPU 的类似策略（如 INT4 + BF16）。
- **与其他优化结合**：与 sparse attention（如 FlashPrefill）、KV-cache compression 联合使用，进一步降低长上下文成本。

---

## ✅ 总结一句话
> **Mix-Quant 通过“量化 prefill、保留 decode 精度”的 phase-aware 策略，在几乎不牺牲任务性能的前提下，实现了高达 3× 的 prefill 加速，为高效可靠的 LLM agent 推理提供了新的范式。**

🔗 **代码开源地址**：[https://github.com/haiquanlu/Mix-Quant](https://github.com/haiquanlu/Mix-Quant)

</details>

---

### 2. [Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization](https://arxiv.org/abs/2605.20295)

**Authors**: Jinghe Zhang, Daliang Xu, Chenghua Wang, Weikai Xie, Tao Qi, Yun Ma, Mengwei Xu, Gang Huang  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 14.0  
**Type**: new  
**ArXiv ID**: 2605.20295v1  

#### Abstract
Large language models (LLMs) are increasingly deployed on mobile devices, where Neural Processing Units (NPUs) necessitate fully static quantization for optimal inference efficiency. However, existing post-training quantization (PTQ) methods predominantly rely on dynamic activation quantization, ren...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代移动设备上的 **Large Language Models (LLMs)** 推理面临两大挑战：
- **硬件约束**：移动 **NPU**（如高通 Hexagon）为追求高能效，要求使用 **fully static quantization**（全静态量化），即所有量化参数在编译时确定，避免运行时动态计算（如动态缩放因子）。
- **精度损失**：现有的主流 **Post-Training Quantization (PTQ)** 方法大多依赖 **dynamic activation quantization**（动态激活量化），虽然精度较高，但无法满足 NPU 的静态部署需求，导致转换后出现严重精度下降。

因此，如何在 **NPU 友好的 fully static quantization** 条件下，实现高精度、低延迟的 on-device LLM 推理，是本文要解决的核心问题。

---

### 🚀 提出的新方法：Quant.npu
作者提出 **Quant.npu** —— 一个专为移动 NPU 设计的 **整数-only 全静态量化框架**，其核心创新包括：

#### （1）Rotation-and-bit-width-aware Initialization  
- **洞察**：量化参数的初始化对优化稳定性至关重要。不恰当的初始化会导致梯度不稳定或收敛缓慢。
- **方案**：根据张量是否经过 **rotation** 和目标 **bit-width**，自适应选择初始化策略：
  - **Rotated 激活**（分布近似高斯）→ 使用 **Mean-based 初始化**（基于均值±标准差）
  - **Unrotated 激活**（重尾分布）→ 使用 **Max-Min 初始化**（基于最大最小值范围）

#### （2）Distribution-aware Selective Optimization（两阶段量化流程）
- **洞察**：并非所有量化参数都适合联合学习。对重尾分布的张量（如 `output activation`、`KV cache`）进行可学习优化反而会破坏训练稳定性。
- **方案**：采用 **两阶段优化流程**：
  - **Stage One**：仅对 **linear layers 的输入激活和权重** 进行梯度优化，同时联合优化 **rotation matrices**。
  - **Stage Two**：对剩余张量（如输出激活、SiLU）直接使用 **static calibration**（静态校准），不再优化。

#### （3）Sensitivity-guided Adaptive Mixed-Precision
- **洞察**：完全移除在线旋转（如 R4）会导致 `down_proj` 输入激活出现严重离群值，难以用 8-bit 表示。
- **方案**：引入 **自适应混合精度策略**：
  - 定义 **quantization sensitivity metric**（相对量化误差）衡量各层敏感度。
  - 对最敏感的 **top-k% 的 down_proj 层** 升级到 16-bit，其余保持 8-bit。
  - 实现精度与效率的精细平衡。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法（如 SpinQuant） | Quant.npu |
|------|------------------------|-----------|
| **量化模式** | 动态激活量化（不可部署于 NPU） | **Fully Static Quantization**（NPU 可部署） |
| **优化方式** | 联合优化所有量化参数 | **Selective Optimization**（只优化关键部分） |
| **初始化** | 固定或随机初始化 | **Rotation/bit-width 自适应初始化** |
| **精度-效率权衡** | 固定 bit-width | **Adaptive Mixed-Precision**（按需提升精度） |
| **部署开销** | 需在线浮点矩阵乘法（R3/R4） | **仅使用 R1/R2，可离线融合，零运行时开销** |

> ✅ **最终优势**：在 **真实移动 NPU 上实现高精度 + 低延迟推理**，无需牺牲模型性能换取部署可行性。

---

## 2. 核心实验方法和设置

### 📚 数据集
- **量化校准数据集**：`WikiText-2`（用于激活分布分析和参数初始化）
- **语言建模评估**：`C4`（计算 **Perplexity, PPL**）
- **下游任务评估**（Zero-shot QA）：
  - PIQA
  - Winogrande
  - HellaSwag
  - ARC-E / ARC-C
  - LAMBADA
- **指令跟随能力评估**：`AlpacaEval 2.0`

---

### ⚙️ 实验设置
- **模型**：
  - 主要：`Llama-3.2-3B-Instruct`
  - 扩展验证：`Qwen2.5-3B-Instruct`, `SmolLM2-1.7B-Instruct`, `Qwen3-1.7B`, `Llama3-8B`
- **硬件平台**：
  - **主测试**：Qualcomm SM8650（真实移动 NPU）
  - **扩展测试**：SM8750（更新一代 NPU）
- **量化配置**：
  - **Weight**：per-channel symmetric quantization
  - **Activation**：per-tensor symmetric quantization
  - 支持多种组合：W4A8, W8A8, W4A4 等
- **评估指标**：
  - **PPL ↓**（越低越好）
  - **Zero-shot Accuracy ↑**（平均得分）
  - **End-to-end Latency ↓**（prefill & decode 速度，单位：tokens/s）
  - **Peak Memory Usage ↓**
  - **Energy Consumption ↓**

---

### 🆚 基线方法对比
| 方法 | 类型 | 是否支持 NPU 静态部署 |
|------|------|---------------------|
| **ExecuTorch** | PTQ（block-wise W4A16） | ✅ 是（但低效） |
| **QuaRot** | Rotation-based PTQ（固定旋转） | ❌ 含动态量化 |
| **SpinQuant** | Rotation-based PTQ（可学习旋转 + 动态量化） | ❌ 不兼容静态部署 |
| **MobileQuant** | Mobile-friendly PTQ | ❌ 原始设计非全静态 |

> 所有方法均在 **ExecuTorch 整数推理后端** 上统一实现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（以 Llama-3.2-3B-Instruct 为例）

#### （1）W8A8 设置下接近 FP32 性能
| 方法 | PPL (C4) | Avg Accuracy |
|------|----------|-------------|
| FP32 | 16.85 | 0.613 |
| Quant.npu | **16.42** | **0.6165** |
| SpinQuant | 17.18 | 0.6035 |

✅ **结论**：Quant.npu 在 W8A8 下甚至略微超越 FP32！

---

#### （2）W4A8 设置下显著优于基线
| 方法 | PPL | Avg Accuracy |
|------|-----|--------------|
| ExecuTorch (W4A8) | 26.39 | 0.5272 |
| SpinQuant (W4A8) | 28.78 | 0.4623 |
| **Quant.npu (W4A8)** | **19.16** | **0.5827** |

📌 **提升**：相比 SpinQuant，PPL ↓9.62，准确率 ↑12.04%

---

#### （3）W4A4 极端压缩下仍保持可用性
| 方法 | PPL | Avg Accuracy |
|------|-----|--------------|
| ExecuTorch | 6950.81 | 0.2984 |
| SpinQuant | 528.97 | 0.2940 |
| **Quant.npu** | **21.76** | **0.5525** |

📌 **惊人表现**：在 W4A4 下，Quant.npu 仍保持接近 W8A8 基线的精度，而其他方法几乎崩溃。

---

### ⏱️ 端到端延迟（SM8650 NPU）
| 方法 | Prefill Speed (tokens/s) | Decode Speed | 相对加速 |
|------|--------------------------|------------|---------|
| ExecuTorch (W4A16) | ~800 | ~28 | 1.0× |
| **Quant.npu (W4A8)** | **~990** | ~22.7 | **↑15.1%** |

✅ **结论**：在 **仅损失 2.58% 平均精度、PPL ↑1.23** 的前提下，实现 **最高 15.1% 的推理加速**。

---

### 🔬 消融实验（Ablation Study）

| 配置 | PPL | Avg Accuracy |
|------|-----|--------------|
| Baseline (S + Joint Opt) | 69.65 | 0.2969 |
| + Rotation-aware Init | 36.42 | 0.3796 |
| + Selective Optimization | 30.89 | 0.4015 |
| + Adaptive MP (10%) | **22.09** | **0.4733** |

📌 **结论**：
- 初始化贡献最大（PPL ↓33.23）
- 选择性优化进一步稳定训练
- 自适应混合精度以极小代价恢复精度（仅 10% 层升至 16-bit）

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **初始化决定优化起点质量**：Mean-based 初始化在旋转后激活上效果最佳，Max-Min 更适合未旋转的重尾分布。
2. **不是所有参数都该被优化**：对 `output activation` 和 `KV` 张量进行可学习优化反而有害，应使用静态校准。
3. **静态量化可以媲美动态量化精度**：通过合理的初始化 + 选择性优化 + 混合精度，Quant.npu 实现了与 SOTA 动态方法相当的精度。
4. **低比特 + 全静态 ≠ 低性能**：W4A8 配置下，Quant.npu 在真实 NPU 上实现 **高精度 + 高吞吐**，打破“静态必掉点”刻板印象。

---

### ⚠️ 方法的局限性
1. **仍保留部分 16-bit 操作**：如 SiLU 激活函数等非线性层使用 16-bit，限制了向全 8-bit 或更低比特的演进。
2. **依赖校准数据质量**：当前未优化 calibration dataset 的选择，若数据代表性不足可能影响泛化。
3. **未探索更复杂的 rotation 结构**：仅使用 R1/R2，未尝试结构化稀疏旋转或其他正交变换。

---

### 🔮 未来工作方向
- 探索 **fully low-bit execution pipeline**（如 INT4/INT2），减少对 16-bit 的依赖。
- 设计 **principled calibration data selection** 方法（如基于多样性或重要性采样）。
- 将 Quant.npu 扩展至 **vision-language models** 和 **real-time streaming inference** 场景。
- 结合 **sparse quantization** 或 **weight sharing** 进一步压缩模型。

---

## 总结
> **Quant.npu 成功弥合了高精度 PTQ 与 NPU 静态部署之间的鸿沟**。它通过 **rotation-aware 初始化**、**distribution-aware 选择性优化** 和 **adaptive mixed-precision** 三大技术，在真实移动 NPU 上实现了 **SOTA 级别的精度** 与 **高达 15.1% 的推理加速**，为 on-device LLM 的高效部署提供了实用且可扩展的解决方案。

</details>

---

### 3. [Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption](https://arxiv.org/abs/2605.19593)

**Authors**: Mert Yildiz, Pietro Spadaccino, Alexey Rolich, Francesca Cuomo, Andrea Baiocchi  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2605.19593v1  

#### Abstract
Modern deployments of Large Language Models (LLMs) increasingly require serving multiple models with diverse architectures, sizes, and specialization on shared, heterogeneous hardware. This setting introduces new challenges for resource allocation, dispatching, and scheduling, particularly under GPU...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Towards Multi-Model LLM Schedulers: Empirical Insights into Offloading and Preemption*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代 **Large Language Model (LLM)** 部署环境日益复杂，需在共享的异构硬件上同时服务多个不同架构、规模和用途的模型。这带来了新的调度挑战，尤其是在 **GPU 内存受限** 的情况下，需要进行 **CPU-GPU 层级卸载（layer offloading）** 和 **任务抢占（preemption）**。

然而，现有的主流系统（如 vLLM）主要针对单个模型优化吞吐量，缺乏对 **多模型共存场景下资源动态分配、卸载策略与抢占开销** 的深入理解。本文填补了这一空白，系统性地研究了 **multi-model LLM 调度中的关键性能影响因素**。

### 🚀 提出的新方法与新思路
本文并未提出一个全新的调度器，而是通过大规模实证研究，揭示了以下关键现象，并基于此提炼出下一代 **multi-model LLM scheduler** 必须具备的核心特征：

- **非线性卸载敏感性建模**：首次系统性地展示了不同模型在部分 offloading 下的 **decode throughput** 如何随 GPU 层占比变化，发现其关系是 **强非线性且模型依赖的**。
- **抢占开销的精确分解**：首次将完整的 **preempt-resume cycle** 开销分解为多个阶段（KV cache transfer、unload、reload 等），并量化各部分贡献。
- **提出“固定抢占代价”模型**：发现抢占总开销几乎与中断时的生成进度无关，主要由 **model reload 时间决定**，因此可建模为 **每个模型-硬件对的固定常数**。

### 🔍 相比现有方法的优势
| 方面 | 现有方法 | 本文贡献 |
|------|--------|--------|
| **Offloading 研究** | 多在固定配置下比较全 GPU vs 全 CPU，未系统扫描 offloading ratio | 扫描从 0% 到 100% 的 GPU 层占比，揭示连续非线性退化规律 |
| **Preemption 分析** | 假设抢占有开销，但未实测分解；或仅模拟 | 实测完整抢占流程，精确量化 reload 占主导（>98%） |
| **调度指导** | 缺乏对 multi-model 场景的细粒度指导 | 提出六大关键特征，为未来 scheduler 设计提供实证依据 |

---

## 2. 核心实验方法和设置

### 🧪 实验目标
- 量化 **partial CPU-GPU layer offloading** 对 decode throughput 的影响。
- 测量 **full job preemption** 的实际开销，并分解其构成。
- 探索模型大小、硬件平台、序列长度等因素的影响。

### 📦 使用的模型（Models）
| 实验类型 | 模型列表 | 精度 | 参数量 |
|---------|--------|-----|-------|
| **Offloading** | Llama 3 8B, Qwen3-32B, Llama 2 70B | Q4 量化 | 8B–70B |
| **Preemption** | Qwen2.5-3B, Qwen3-8B, Qwen2.5-14B | FP16 | 3B–14B |

### 💻 硬件平台（Hardware）
- **CPU**: AMD Threadripper PRO 5995WX (64 cores, 512GB DDR4)
- **GPU 1**: NVIDIA RTX 5000 Ada (32GB VRAM, PCIe Gen4 x16)
- **GPU 2**: NVIDIA RTX A6000 (48GB VRAM, PCIe Gen4 x16)

> 在两个平台上分别运行实验以评估硬件差异。

### 🛠️ 软件与工具
- **Offloading 实验**：使用 **Ollama v0.17.7** 控制 GPU 层数量。
- **Preemption 实验**：使用 **HuggingFace Transformers**，手动控制 KV cache 迁移与模型加载/卸载。

### 📊 工作负载（Workloads）
| 实验 | 设置 |
|------|------|
| **Offloading** | 3 模型 × 13–15 种 GPU 层配置 × 6 种输出长度（50–5000 tokens）× 3 次重复 |
| **Preemption** | Job A 生成 7000 tokens，在 9 个检查点中断，运行 500-token 的 Job B；测试 4 种模型组合 |

### 📈 评估指标（Metrics）
| 类型 | 指标 |
|------|------|
| **Offloading** | Decode throughput (tok/s), Normalized throughput (vs 100% GPU) |
| **Preemption** | Total preemption overhead (s), Breakdown by step (KV transfer, unload, reload), Overhead as % of baseline |

---

## 3. 主要实验结果和性能指标

### 📉 Offloading 实验结果
- **小模型对 offloading 更敏感**：
  - Llama3-8B 在 GPU 层接近 100% 时 throughput 急剧上升，表明 **轻微 CPU offloading 即导致显著性能下降**。
  - 大模型（如 Llama2-70B）则呈现更平缓的线性增长趋势，对 offloading 更鲁棒。
- **Normalized throughput 在 RTX 5000 上更高**：
  - 因为 RTX 5000 本身性能较低，CPU 与其差距小，offloading 惩罚更轻。
  - RTX A6000 性能更强，offloading 导致更大的相对性能损失。
- **长序列加剧 offloading 影响**：
  - 随着输出长度增加，KV cache 增大，decode 阶段效率下降被放大。

### ⚙️ Preemption 实验结果
#### 🔹 总体开销极低且恒定
| 模型 | RTX 5000 开销 | RTX A6000 开销 | 占总耗时比例 |
|------|---------------|----------------|-------------|
| Qwen2.5-3B (3B) | ~3.0 s | ~2.6 s | ~2.0% |
| Qwen3-8B (8B) | ~5.2 s | ~4.1 s | ~2.1% |
| Qwen2.5-14B (14B) | ~7.3 s | ~5.7 s | ~1.7% |

> **关键发现**：无论在生成 100 还是 5000 token 后中断，**preemption overhead 几乎不变**。

#### 🔹 开销构成分析（Table II）
| 步骤 | 占比（RTX 5000） | 占比（RTX A6000） |
|------|------------------|-------------------|
| **Model Swap (Unload + Reload)** | >99% | >98.5% |
| **KV Cache Transfer (双向)** | <1% | <1.5% |

- **Model reload 是绝对主导项**（例如 Qwen3-8B 占 4.7s / 5.16s）。
- **KV cache transfer 最高仅约 90ms**，即使在 5000 tokens 时也微不足道。
- **Preempting model 的大小决定开销，而非被抢占的 Job B 的大小**。

#### 🔹 PCIe 带宽利用率
- GPU→CPU 传输速率：~10–12 GB/s（RTX 5000），~9–10 GB/s（RTX A6000）
- CPU→GPU 传输速率：~13–16 GB/s
- 远低于 PCIe Gen4 x16 的理论峰值（31.5 GB/s），因 PyTorch 逐层拷贝引入额外开销。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Decode throughput 随 offloading 比例呈非线性下降**，且 **小模型比大模型更敏感**。
2. **Preemption overhead 几乎为常数**，主要由 **model reload 时间决定**，与已生成 token 数无关。
3. **KV cache transfer 开销极小**（<1.5%），即使在长上下文下也不构成瓶颈。
4. **硬件平台显著影响 offloading 效率与 reload 时间**：高端 GPU 受 offloading 惩罚更大，但 reload 更快。
5. **抢占代价可建模为 “per-model-per-hardware” 的固定值**，极大简化调度决策。

### ⚠️ 方法的局限性
- 所有实验基于 **single-request 执行模式**，未考虑 **continuous batching** 或高并发场景。
- 仅测试 **一次抢占**，未研究频繁抢占下的累积效应。
- 使用的是 **PCIe Gen4**，未来若采用 CXL 或 NVLink，KV transfer 可能变得更重要。
- 未涉及 **Mixture-of-Experts (MoE)** 或 **multi-modal models**。

### 🔮 未来工作方向
- 将 offloading 与 preemption 特性扩展到 **continuous batching** 和 **multi-request** 场景。
- 研究 **多次抢占** 下的 aggregate overhead 与调度策略优化。
- 构建 **workload-level simulation**，评估在真实请求到达模式下，preemption 是否真正提升整体吞吐。
- 探索 **CXL、SSD offloading** 等新型存储层级对 reload 时间的影响。
- 设计基于本文发现的 **prototype multi-model scheduler**，集成 offloading 敏感性、preemption 固定代价等特征。

---

> **一句话总结**：  
> 本文通过精细实证揭示，**multi-model LLM 调度中，offloading 性能退化是非线性的且模型相关，而抢占开销几乎是固定的、由 model reload 主导**——这些发现为设计高效、硬件感知的下一代 LLM 调度系统提供了坚实基础。

</details>

---

### 4. [EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design](https://arxiv.org/abs/2605.19743)

**Authors**: Gioele Molinari, Florian Felten, Soheyl Massoudi, Mark Fuge  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.19743v1  

#### Abstract
Large Language Model (LLM) agents are increasingly applied to engineering design tasks, yet existing evaluation frameworks do not adequately address multi-agent systems that combine simulation, retrieval, and manufacturing preparation. We introduce a benchmark suite with three evaluation dimensions:...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：EngiAI: A Multi-Agent Framework and Benchmark Suite for LLM-Driven Engineering Design**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 LLM agent 评估框架主要针对通用任务或单步工具调用，**无法有效评估多 agent 系统在复杂工程设计流程中的表现**，尤其是在结合仿真、检索增强生成（RAG）和高性能计算（HPC）训练编排等多阶段、长周期任务时存在明显不足。

具体挑战包括：
- 如何评估 agent 在**条件分支、语义消歧、工作记忆追踪**等认知需求下的表现；
- 如何**隔离并量化 RAG 对参数选择的贡献**，避免模型依赖先验知识“猜对”答案；
- 缺乏对 **end-to-end HPC ML training pipeline**（如 SLURM 集群上的模型训练）的自动化编排能力评估。

### **提出了什么新方法或新思路**
本文提出两个核心贡献：

#### **(1) ENGIAI：一个基于 LangGraph 的 Multi-Agent System (MAS) 参考实现**
- 采用**分层监督架构（hierarchical supervisor pattern）**，由一个中央 supervisor agent 路由请求至七个专业化 agent。
- 各 agent 分别负责：
  - **Engineering Agent**：执行拓扑优化、仿真、STL 导出（通过 EngiBench/EngiOpt）；
  - **RAG / ArXiv / Search Agents**：提供文档问答、论文检索与网络搜索；
  - **HPC / CLI / Prusa Agents**：管理远程集群作业、本地命令行执行、3D 打印机控制。
- 支持自然语言交互，并可通过 Web 界面访问。

#### **(2) 三维度 Benchmark Suite**
为系统化评估 LLM-driven 工程设计流程，构建了一个涵盖以下三个维度的基准套件：

| 维度 | 内容 |
|------|------|
| **Workflow Benchmark** | 设计七种 prompt style，测试不同认知能力：<br>• `FULL`（直接工具使用）<br>• `NATURAL`（模糊描述需澄清）<br>• `W-RAND`（随机导出参数）<br>• `W-DERIVED`（派生参数计算）<br>• `W-DISTRACT`（竞争性数值干扰）<br>• `W-COND`（条件分支决策）<br>• `W-MULTI`（多配置导出） |
| **RAG Benchmark** | 引入**门控评分机制（gated scoring）**，仅当 agent 显式调用 `search_documents` 工具且返回正确参数时才得分，防止模型靠记忆“蒙混过关”。 |
| **HPC Benchmark** | 测试 agent 是否能完成完整的 ML 训练 pipeline：<br>1. 生成 SLURM 脚本<br>2. 提交到远程集群<br>3. 监控训练状态<br>4. 下载模型并评估性能 |

### **相比现有方法的优势**
| 方面 | 优势 |
|------|------|
| **全面性** | 是首个同时覆盖 **simulation + retrieval + manufacturing + HPC training** 的工程设计 agent 框架与评估体系。 |
| **可解释性** | RAG 门控机制首次实现了对“检索是否真正起作用”的**因果分离评估**。 |
| **实用性** | 支持真实世界工具集成（如 Prusa 3D 打印机、SLURM 集群），具备实际部署潜力。 |
| **模块化** | 新功能可通过新增 agent 或 tool API 插入，支持持续扩展。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **Beams2D**：二维悬臂梁拓扑优化问题，目标是最小化 compliance，约束为体积分数（volfrac）、力作用位置（forcedist）、滤波半径（rmin）。有标准解可供对比。
- **Photonics2D**：二维光子器件逆向设计问题，最大化电磁场重叠率，用于测试跨物理域泛化能力。

以上均来自 **EngiBench** 框架提供的统一 Python API。

### **实验设置**
#### **LLM Backends（共4个）**
| 类型 | 模型 |
|------|------|
| Proprietary Cloud Models | `GPT-5-mini`, `Gemini-3-Flash` |
| Open-Source Local Models (via Ollama) | `Qwen3-4B`, `Qwen3.5-4B` |

所有实验设置 `temperature=0` 并固定 seed 以保证可复现性。

#### **评估指标**
##### **Workflow Performance**
综合得分公式：
$$
S_{\text{workflow}} = 0.65 \cdot S_{\text{design}} + 0.20 \cdot S_{\text{tool}} + 0.15 \cdot S_{\text{completion}}
$$
其中：
- $S_{\text{design}}$：设计质量（65%），加权平均 IoU、像素准确率、目标匹配、约束满足、连通性、水密性；
- $S_{\text{tool}}$：工具调用效率（20%），正确调用数 / 最优调用数；
- $S_{\text{completion}}$：任务完成率（15%），是否成功调用所有必需工具。

##### **RAG Evaluation**
采用**门控评分（gated scoring）**：
- 若未调用 `search_documents`，即使参数正确也得分为 0；
- 若调用但索引为空（Empty RAG），则检验是否盲目信任空结果。

##### **HPC Orchestration**
主要指标为加权复合得分：
$$
S_{\text{HPC}} = 0.70 \cdot S_{\text{step}} + 0.15 \cdot S_{\text{config}} + 0.15 \cdot S_{\text{eval}}
$$
- $S_{\text{step}}$：各步骤完成情况；
- $S_{\text{config}}$：脚本配置正确性；
- $S_{\text{eval}}$：能否提取评估指标（如 MMD, DPP, RVC, IOG/COG/FOG）。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **Workflow Task Completion Rate（Beams2D）**
| Prompt Style | GPT-5-mini | Gemini-3-Flash | Qwen3-4B | Qwen3.5-4B |
|------------|------------|----------------|----------|-------------|
| **Average TC** | **96%** | **97%** | 55% | **78%** |
| FULL | 100% | 93% | 0% | 73% |
| NATURAL | 87% | 100% | 0% | 33% |
| W-COND（条件分支） | 93% | 87% | 40% | 60% |

> ✅ **发现**：专有模型接近完美；开源模型中 Qwen3.5-4B 相比前代显著提升（+23%），显示快速代际进步。

#### **Photonics2D 泛化能力**
| Prompt Style | Best Model (Gemini-3-Flash) |
|------------|----------------------------|
| W-RAND / W-DISTRACT | 100% TC |
| **W-COND** | **仅 53% TC** |

> ❗ **难点突出**：条件推理在陌生领域（Photonics）难度剧增，失败主因是**分支反转**（branch inversion），即 agent 错误判断阈值比较方向。

#### **RAG Evaluation 结果**
| 条件 | 得分趋势 |
|------|--------|
| **RAG-on** | 接近满分（~1.0） |
| **RAG-off / Empty RAG** | 几乎为零（except P0 中的常见值 0.35 可能被记忆） |

> ✅ **验证了门控机制有效性**：只有通过检索获取的信息才被认可，证明 RAG 对后训练时期文献参数至关重要。

#### **HPC Training Pipeline 成功率**
| 模型 | Prompt Style | Step 4 完成率 |
|------|--------------|----------------|
| **Gemini-3-Flash** | Explicit / Natural | **100%** |
| **GPT-5-mini** | Explicit | 70% |
| **GPT-5-mini** | Natural | **50%** |

> ⚠️ **发现**：GPT-5-mini 在长流程中出现**指令衰减（instruction degradation）**，常遗漏最后一步 `evaluate_model` 调用，尤其在自然语言提示下更严重。

---

## **4. 关键结论和发现**

### **主要发现**
1. **RQ1 (Workflow Performance)**  
   - 专有模型（GPT-5-mini, Gemini-3-Flash）在 Beams2D 上达到 **96–97% 平均任务完成率**；
   - 开源 4B 模型仍有较大差距，但 **Qwen3.5-4B 比 Qwen3-4B 提升显著**（55% → 78%）；
   - **W-COND（条件分支）最难**，尤其在 Photonics2D 上最高仅 53%，暴露模型对非标准物理指标理解薄弱。

2. **RQ2 (Model Robustness)**  
   - 两大专有模型表现一致稳健；
   - 开源模型代际改进可部分弥补规模劣势。

3. **RQ3 (Tool Usage Efficiency)**  
   - 多余工具调用会降低综合得分（如从 0.70 降至 0.63），但不影响设计质量；
   - **Qwen3.5-4B 实现最优工具效率**，说明其行为更精准。

4. **RQ4 (RAG Improvement)**  
   - **RAG-on 得分接近 1.0，其余接近 0**，验证了门控评分机制的有效性；
   - 表明 agent 必须依赖检索内容而非先验知识才能完成任务。

5. **RQ5 (HPC Orchestration)**  
   - 当前专有模型可完成 end-to-end ML 训练 pipeline；
   - 但 **multi-step 指令遵循能力随流程延长而退化**，尤其在自然语言输入下更为明显。

---

### **局限性**
1. **问题范围有限**：仅测试 Beams2D 和 Photonics2D，未覆盖 EngiBench 全部问题；
2. **LLM 数量受限**：仅评测 4 个模型，受 API 成本限制；
3. **缺少人类干预研究**：无真实工程师参与的人在环实验；
4. **HPC 实验成本高**：仅限于两个专有模型，开源模型因推理慢未参与；
5. **缺乏单 agent 基线对比**：未进行 supervisor 架构的消融实验；
6. **未做统计显著性分析**：仅报告均值与标准差。

---

### **未来工作方向**
1. **扩大评估范围**  
   - 加入更多 EngiBench 问题、更大开源模型（如 70B）、其他模型家族（Llama, Mistral）；
   - 进行敏感性分析（温度、prompt 表述、tool 描述长度）。

2. **优化工具使用效率**  
   - 引入 **few-shot 示例** 或 **constrained decoding** 抑制不必要的工具调用；
   - 探索工具调用模式是否跨问题迁移。

3. **增强条件推理能力**  
   - 引入 **structured chain-of-thought**，要求显式写出模拟结果、比较逻辑与分支选择；
   - 解耦参数提取与工具调用，先生成结构化计划再执行。

4. **升级 RAG 与 HPC 评估**
   - 构建更大、含矛盾信息的文档集合，测试 **adversarial retrieval** 能力；
   - 要求 agent 自主编写并迭代训练代码，而非仅编排预设脚本；
   - 引入显式状态跟踪机制（如 checkpoint 记录已完成步骤），缓解长流程中的指令遗忘。

5. **探索工具生态系统扩展**
   - 研究随着可用工具数量增长（via MCP 或 tool APIs），agent 性能如何变化；
   - 开发专用 fine-tuning 数据集，利用本框架产生的 trace 优化小型开源模型的 tool-calling 准确率。

--- 

> 🔚 **总结**：该论文填补了 LLM 在**复杂工程设计流程**中缺乏系统性评估框架的空白，提出的 **ENGIAI 框架 + 三维度 benchmark** 为未来智能设计系统的发展提供了重要基础设施与评估标准。

</details>

---

### 5. [Instant GPU Efficiency Visibility at Fleet Scale](https://arxiv.org/abs/2605.20799)

**Authors**: Connor Pedersen, Dong H. Ahn, Michel Migdal, Collin Neale, Nik Konyuchenko  
**Category**: cs.DC  
**Published**: 2026-05-21  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.20799v1  

#### Abstract
We present Overall FLOP Utilization (OFU), a hardware-level, precision-agnostic GPU efficiency metric for AI workloads on HPC systems, derived from two on-chip performance counters: Tensor Pipe Activity and SM clock frequency. OFU requires no application instrumentation and works across GPU generati...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结  
**论文标题：** *Instant GPU Efficiency Visibility at Fleet Scale*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在大规模 AI 高性能计算（HPC）系统中，GPU 利用率（utilization）是决定经济可行性的关键因素。然而，现有的 GPU 利用率测量方法存在以下问题：
- **应用层 MFU（Model FLOPs Utilization）**：需要框架级集成，覆盖率低（文中指出仅约 20% 工作负载支持），且易因模型架构演进而出现 FLOPs 计算错误。
- **性能分析工具（如 Nsight Compute）**：精度高但需手动插桩，开销大，不适合持续、全集群监控。
- **硬件计数器缺乏系统性验证**：虽非侵入性强，但其准确性未被充分量化，难以作为可靠度量。

因此，亟需一种**无需应用修改、跨代兼容、精度可控、可扩展至整个 GPU 集群**的利用率度量方法。

### 提出了什么新方法或新思路
作者提出 **Overall FLOP Utilization (OFU)** —— 一种基于硬件性能计数器的 GPU 效率度量指标，具有以下特点：

- **定义方式**：  
  $$
  \text{OFU} = \text{Tensor Pipe Activity} \times \frac{f_{\text{SM}}}{f_{\text{max}}}
  $$
  其中：
  - `Tensor Pipe Activity` 是执行 Tensor Core 指令的周期占比；
  - $ f_{\text{SM}} / f_{\text{max}} $ 是当前 SM 时钟频率相对于最大频率的比例。

- **实现机制**：
  - 完全依赖 DCGM 提供的硬件性能计数器；
  - 不需要任何应用程序插桩或软件栈修改；
  - 支持所有 NVIDIA GPU 架构（H100、GB200 等）和多种数值精度（FP16、TF32、FP8、NVFP4）。

### 相比现有方法的优势
| 维度 | 应用层 MFU | Profiling 工具 | OFU |
|------|------------|----------------|-----|
| 是否需插桩 | ✅（需代码修改） | ✅（运行时开销） | ❌（零侵入） |
| 覆盖范围 | 有限（依赖框架支持） | 单任务级别 | ✅ 全集群、全工作负载 |
| 精度可靠性 | 易出错（公式过时） | 高 | 中高（经校正后接近真实值） |
| 可扩展性 | 差 | 差 | ✅ 支持自动化监控服务 |
| 多精度支持 | 手动调整 | 支持 | ✅ 自动感知（硬件无关） |

> ✅ **核心优势总结**：OFU 实现了“即时、全集群可见性”（instant fleet-wide visibility），为大规模 AI 训练提供了部署就绪的效率监测方案。

---

## 2. 核心实验方法和设置

### 使用的数据集与工作负载
- **控制实验**：使用 GEMM（矩阵乘法）内核进行基准测试，覆盖不同尺寸（N=128~32768）、形状（随机/对齐）和精度。
- **生产环境数据**：
  - **608 个真实训练任务**：基于 H100 GPU 的 Megatron-LM 训练作业，涵盖从 8 到 5888 张 GPU 的规模；
  - 包括 MoE、混合架构（Mamba + Transformer）、激活重计算等复杂场景。

### 实验设置
- **平台**：
  - H100 SXM 和 GB200 NVL 平台；
  - 使用 `nvcr.io/nvidia/pytorch:25.11-py3` 容器环境；
  - 性能采集通过 DCGM + Prometheus 实现，采样间隔为 30 秒。
- **对比指标**：
  - **Ground Truth**：App MFU（由 OneLogger 报告的应用级 MFU）；
  - **预测指标**：原始 OFU 与经过 tile quantization 校正后的 Adjusted OFU。

### 评估指标
- Pearson 相关系数（r）
- Mean Absolute Error (MAE)
- ≤2pp / ≤5pp 准确率（即误差小于等于 2 或 5 个百分点的任务比例）
- 生产案例中的相对误差（Relative Error）

### 基线方法对比
- 主要对比对象为 **应用层报告的 MFU**（来自 Megatron-LM 等框架）；
- 同时将 OFU 与 Nsight Compute 测得的真实 FLOPs 进行对比，用于验证其物理正确性。

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）GEMM 控制实验结果（Table II）
| GPU | Precision | Estimator | MAE (pp) | ≤2pp (%) | ≤5pp (%) |
|-----|-----------|----------|----------|----------|----------|
| H100 | FP16 | OFU | 1.90 | 64% | 96% |
| H100 | FP16 | Adj OFU | 0.06 | 100% | 100% |
| H100 | TF32 | OFU | 3.46 | 44% | 86% |
| H100 | TF32 | Adj OFU | 0.50 | 99% | 99% |
| GB200 | NVFP4 | OFU | 1.21 | 87% | 98% |
| GB200 | NVFP4 | Adj OFU | 1.15 | 95% | 100% |

✅ **结论**：经过 tile quantization 校正后，Adjust OFU 在绝大多数情况下误差 ≤2 个百分点，最高达 100% 准确率。

#### （2）生产训练任务验证（608 jobs）
- **整体相关性**：OFU 与 App MFU 的 Pearson r = **0.53**
- **排除异常任务后**（如 MoE 架构误报）：r 提升至 **0.78**
- **平均绝对误差（MAE）**：6.2%
- **79.4% 的任务误差 <10%**
- **显著改进于小规模任务**：大于 768 GPU 的任务基本实现 <5% 绝对误差

#### （3）消融实验：五大误差源分析
| 误差来源 | 影响程度 | 可解释性 | 校正手段 |
|--------|---------|----------|----------|
| **Tile Quantization** | 高（最大可达 33%，尤其 TF32） | ✅ 可建模 | 使用 `ceil()` 推导有效维度并修正 |
| **Floating-point Precision Scaling** | 中 | ✅ | OFU 天然捕捉实际吞吐变化 |
| **Clock Sampling Noise** | 低 | ✅ | 30s 内噪声 < ±0.22pp |
| **Tensor Core Clock Domain** | 中 | ✅ | 使用独立 TC clock（如 H100 为 1830MHz）而非 SM boost clock |
| **Non-tensor Undercounting** | 极低 | ✅ | 忽略 CUDA-core FLOPs（<0.2%）合理 |

> ✅ **关键发现**：主要误差来自 tile padding 和框架级 FLOPs 错误，而非 OFU 本身。

---

## 4. 关键结论和发现

### 主要发现
1. **OFU 是一个实用且可靠的 MFU 替代指标**：
   - 在无任何模型信息的前提下，OFU 能以 ≤2pp 的误差预测真实 MFU；
   - 特别适用于新型、复杂模型（如 MoE、Mamba、混合架构），这些场景下应用层 MFU 极易出错。

2. **硬件计数器可用于发现重大性能回归**：
   - 在实体智能体训练中，因误开启 `TORCH_DISTRIBUTED_DEBUG` 导致通信串行化，OFU 成功检测到利用率下降 **2.5×**；
   - 移除 debug flag 后，OFU 显示利用率恢复。

3. **OFU 能准确反映多精度训练的变化趋势**：
   - 在混合精度预训练（BF16/FP8/NVFP4）中，尽管各精度峰值不同，OFU 仍能同步反映利用率波动；
   - 当切换回 BF16-only 模式时，OFU 与 App MFU 均上升，且两者差异 <1pp。

4. **框架级 FLOPs 计算普遍存在严重错误**：
   - 发现两个典型错误：
     - MoE 中未考虑 latent-space projection，导致 FLOPs 高估 3×；
     - 混合架构中将 Mamba 层当作 Attention 层计数，MFU 被高估 57.5%；
   - 这些问题只有通过 OFU 与 App MFU 的偏差才得以暴露。

### 方法的局限性
- **依赖特定硬件信号**：目前仅适用于 NVIDIA GPU，尤其是具备 Tensor Core 的现代架构（Volta 及以后）。
- **Tile Quantization 仍需外部校正**：虽然可通过 NCU 获取真实 FLOPs 来修正，但在完全黑盒场景下无法自动完成。
- **瞬时采样噪声**：若采集频率太低（>30s），可能影响短期波动判断；建议结合滑动平均处理。

### 未来工作方向
1. **将 OFU 集成进自动化优化闭环**：
   - 结合 OFU 与 profiling 工具形成“预警 → 定位 → 修复”的自动化流程。
2. **扩展至推理场景**：
   - 验证 OFU 在低批量、动态输入下的有效性。
3. **推动硬件设计改进**：
   - 呼吁下一代 GPU 提供更精细的时间平均 clock counter 和 per-kernel FLOPs counter。
4. **标准化 OFU 输出接口**：
   - 推动 DCGM 或其他监控系统原生支持 OFU 指标输出。

---

> 🔚 **最终结论**：  
> **OFU 提供了一种简单、高效、可扩展的方式来实现大规模 GPU 集群的实时效率监控。它不仅是对传统 MFU 的补充，更是解决现代 AI 架构快速演进带来的度量失准问题的关键基础设施。**

</details>

---

### 6. [torchtune: PyTorch native post-training library](https://arxiv.org/abs/2605.21442)

**Authors**: Mark Obozov, Maxime Griot, Joseph Cummings, Evan Smothers, Felipe Mello, Rafi Ayub, Philip John Bontrager, Salman Mohammadi, Ariel Kwiatkowski, Nathan Azrak, Mircea Mironenco  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2605.21442v1  

#### Abstract
Modern LLMs typically require multistage training pipelines to achieve strong downstream performance, with post-training serving as the main interface for adapting open-weight models. We introduce torchtune, a PyTorch-native library designed to streamline the post-training lifecycle of LLMs, enablin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：torchtune: PyTorch native post-training library**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**

现代大型语言模型（LLMs）的**post-training阶段**（如 SFT、DPO、RLHF 等）已成为适配开源模型的关键环节。然而，现有的 fine-tuning 框架存在以下问题：

- **依赖复杂**：基于 `transformers` 的框架引入庞大的依赖栈，影响可复现性和部署。
- **抽象过重**：模型构建、训练逻辑、分布式策略等被封装在工厂类或高级 Trainer 中，难以进行细粒度修改。
- **性能路径不透明**：通用实现未能充分利用 `FSDP2`、`DTensor`、`torch.compile` 和内核优化，导致内存和吞吐次优。
- **多任务支持碎片化**：SFT、DPO、PPO、蒸馏等任务分散在不同库中，难以统一比较。
- **分布式训练组合性差**：跨节点、张量并行、上下文并行等支持不一致。

---

### ✅ **提出了什么新方法或新思路**

作者提出 **torchtune** —— 一个**原生 PyTorch 的 post-training 库**，其设计围绕以下核心思想：

#### 🔧 **模块化组件架构（Modular Components）**
- 所有模型构建通过显式的 `builder` 函数完成，模块组装透明。
- 支持在同一 recipe 下灵活切换：全参数微调、LoRA、QLoRA、量化训练等，仅需替换组件。

#### 🛠️ **YAML 驱动的可组合 Recipe 系统**
- 受 Hydra 启发，使用 YAML 配置定义训练流程（model/data/loss/optimizer 等）。
- 组件可独立替换（如将 `LinearCrossEntropyLoss` 换为标准 CE），无需重写训练循环。

#### ⚙️ **四大关键技术贡献**
1. **In-backward Optimizer Fusion（优化器后向融合）**
   - 在反向传播过程中即时执行 optimizer step，梯度计算完立即消费，显著减少梯度缓冲区生命周期。
   - 内存峰值降低，尤其对大模型（如 Llama 3.3 70B）至关重要。

2. **Linear Cross-Entropy Loss（LCE）**
   - 融合输出投影与交叉熵计算，避免生成完整的 `[B, S, V]` logits 张量。
   - 显著降低 loss 计算时的峰值内存，尤其适用于大词表模型。

3. **基于 DTensor 的可组合并行栈**
   - 支持 FSDP2、Tensor Parallelism、Sequence Parallelism、Expert Parallelism（MoE）、Loss Parallelism、Context Parallelism。
   - 利用 PyTorch 原生 `DTensor` 实现无缝扩展，从单卡到多节点集群无需改代码。

4. **异步 GRPO 训练 Recipe（async_grpo_full_finetune_distributed）**
   - 解耦 rollout 生成与策略更新，通过 Ray 队列 + Replay Buffer 实现高利用率。
   - 支持 on-policy 同步刷新与可控 off-policy 延迟两种模式。

---

### ✅ **相比现有方法的优势**

| 特性 | torchtune | Axolotl | Unsloth |
|------|---------|--------|--------|
| **透明性 & Hackability** | ✅ 高（直接操作 PyTorch 模块） | ❌ 中（依赖 transformers 抽象） | ❌ 低（黑盒 CUDA 内核） |
| **灵活性** | ✅ 极高（组件可自由组合） | ⚠️ 中等（配置驱动但耦合深） | ❌ 低（专注 LoRA/QLoRA） |
| **性能优化** | ✅ 全面（编译 + 并行 + 内存优化） | ⚠️ 一般 | ✅ 高（定制内核加速） |
| **多任务支持** | ✅ 统一接口（SFT/DPO/GRPO/KD） | ⚠️ 多库拼接 | ❌ 有限 |
| **分布式扩展性** | ✅ 原生 FSDP2 + DTensor | ⚠️ 依赖 DeepSpeed/DDP | ❌ 不支持 |

> **定位**：介于高层自动化框架（如 Lightning）与底层高性能内核（如 Unsloth）之间，强调**可复现性、可调试性与硬件效率的平衡**。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**

| 任务 | 数据集 |
|------|-------|
| **SFT（监督微调）** | Alpaca（instruction-following） |
| **DPO（偏好优化）** | Anthropic HH-RLHF（helpful subset） |
| **消融实验** | 自定义合成数据（用于 Context Parallel 测试） |

---

### ⚙️ **实验设置**

#### 硬件环境
- **单卡实验**：1×H100（80GB）
- **多卡实验**：8×H100（单节点），部分使用 GH200（96GB HBM3）

#### 模型规模
- Qwen3 系列：0.6B ~ 32B
- Llama3.1 / Llama3.3：8B ~ 70B

#### 序列长度与批处理
- `seq_len=2048`
- `micro_batch_size=2`，`gradient_accumulation_steps=8`（除非启用 Optim Bwd，则设为 1，batch_size 补至 16）

#### 评估指标
- **内存占用（VRAM per GPU）**
- **吞吐量（tokens/s/GPU）**
- **是否 OOM（Out-of-Memory）**

#### 对比基线
- **Axolotl**：主流社区配置工具，支持多种 PEFT 和 DeepSpeed。
- **Unsloth**：专注于 LoRA/QLoRA 的高效训练，使用 Triton 内核加速。

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据汇总**

#### **Table 1: 多 GPU 性能对比（FSDP2, 8×H100）**

| 方法 | Qwen3 32B<br>Memory (GB) | Tok/s | Llama3.3 70B<br>Memory (GB) | Tok/s |
|------|--------------------------|--------|----------------------------|--------|
| torchtune | 67.78 | 465.79 | OOM | OOM |
| +AC | 42.93 | 405.95 | 74.75 | 122.55 |
| +LCE | 38.43 | 398.56 | OOM | OOM |
| +Compile | 44.17 | 433.12 | 75.22 | 128.57 |
| **+Optim Bwd** | **60.41** | **581.63** | **74.89** | **352.11** |
| Axolotl | 40.9 | 218 | OOM | OOM |

> ✅ **Optim Bwd 是最大吞吐提升项**，在 Llama 70B 上实现 **352 tok/s**，远超 Axolotl（无法运行）。

---

#### **Table 2: 单卡消融实验（Qwen3, 1×H100）**

| 方法 | Qwen3 8B<br>Memory | Tok/s | 说明 |
|------|--------------------|--------|------|
| torchtune (baseline) | OOM | OOM | 未启用任何优化 |
| +AC | 64.04 GB | 3,037 | 激活检查点使训练可行 |
| +Compile | 64.04 → 64.04? | ↑~8k | 编译显著提升吞吐 |
| **+Optim Bwd** | **51.79 GB** | **3,773** | 内存下降 + 吞吐上升 |
| **+AdamW8Bit** | **31.07 GB** | 3,066 | 最大内存节省 |

> ✅ **Activation Checkpointing（AC）是大模型能否运行的关键**。

---

#### **Table 3: Sequence Packing 效果**

| 方法 | Qwen3 8B<br>Tok/s |
|------|------------------|
| Baseline (no packing) | 3,037 |
| +Packed 2048 | 8,094 |
| +Packed 4096 | OOM |

> ✅ **序列打包显著提升 token 利用率和吞吐**，但可能增加内存压力。

---

#### **Table 4: DPO 训练对比（GH200, 96GB）**

| 方法 | Llama3.1 8B<br>Memory | Tok/s |
|------|------------------------|--------|
| torchtune (DPO) | 81.82 GB | 745.0 |
| Axolotl (DPO) | OOM | – |
| Axolotl +8bit | 67.64 GB | 249.2 |

> ✅ **torchtune 可在标准 AdamW 下成功运行 DPO，而 Axolotl 必须降级到 8-bit 才能避免 OOM**。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **Optimization 是互补的**：
   - `torch.compile` 主要提升**吞吐**（尤其中小模型）。
   - `Activation Checkpointing` 和 `Optimizer in Backward` 决定**大模型是否可运行**。
   - `Linear Cross-Entropy` 有效缓解 loss 阶段的内存峰值。
   - `AdamW8Bit` 提供最大绝对内存节省，但可能牺牲吞吐。

2. **torchtune 在内存与性能上全面优于 Axolotl**：
   - 更少 OOM，更高吞吐，尤其在 DPO 和 70B 规模下优势明显。
   - 无需依赖 `bitsandbytes` 或特殊优化即可运行。

3. **Unsloth 在 LoRA 场景下吞吐领先，但灵活性极低**：
   - torchtune 在 LoRA 设置下仍保持竞争力（见 Table 2），且支持更多变体。

4. **组件化设计支持快速迭代**：
   - 更换 loss、optimizer、并行策略仅需修改 YAML 或一行命令行参数。

---

### ⚠️ **局限性**

1. **Gradient Accumulation 不兼容 Optim Bwd**：
   - 当前实现假设每 backward 一次就更新一次，因此必须提高 batch size 来补偿。

2. **与 ZeRO 类优化器集成需谨慎**：
   - Optimizer-in-backward 与全局 optimizer state 分片（如 DeepSpeed ZeRO）存在冲突风险。

3. **异步 GRPO 尚未进行端到端 reward 对比**：
   - 当前仅验证系统可行性，reward 质量与 policy lag 的权衡留待未来研究。

---

### 🔮 **未来工作方向**

1. **支持更多 RL 算法**：如 PPO、KTO、ORPO 等。
2. **增强量化训练能力**：与 TorchAO 深度整合，支持 INT8/FP8 训练。
3. **自动优化策略推荐**：基于模型大小、硬件配置自动选择最优组合（AC + Compile + Optim Bwd 等）。
4. **可视化调试工具**：集成 Profiler 与 Metric Logger，提供训练全流程可观测性。

---

## ✅ **总结**

**torchtune** 是一个面向 LLM **post-training 全流程**的 PyTorch 原生库，其核心价值在于：

> **以最小抽象暴露最大控制力，在保证高性能的同时实现极致的可复现性与可扩展性。**

它不是为了“一键训练”，而是为了“**让每一次实验都清晰、可控、可比较**”。  
对于需要深入研究 LLM 微调机制的研究者而言，torchtune 提供了一个**理想的基础平台**。

</details>

---

### 7. [Spectral Souping: A Unified Framework for Online Preference Alignment](https://arxiv.org/abs/2605.20408)

**Authors**: Yinlam Chow, Guy Tennenholtz, Ted Yun, James Harrison, Arthur Gretton, Andre Barreto, Bo Dai  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.20408v1  

#### Abstract
Reinforcement Learning from Human Feedback (RLHF) effectively aligns Large Language Models (LLMs) with aggregate human preferences but often fails to address the diverse and conflicting needs of individual users. To overcome this issue, we introduce Spectral Souping, a unified framework for efficien...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前主流的 **Reinforcement Learning from Human Feedback (RLHF)** 和 **Direct Preference Optimization (DPO)** 虽然能有效对齐大语言模型（LLM）与人类偏好，但其依赖于聚合反馈得到的单一奖励函数，难以满足**个体用户的多样化、甚至冲突的偏好需求**。为每个用户单独微调模型（fine-tuning）成本高昂且不可扩展。

本文旨在解决 **在线个性化偏好对齐（online personalized preference alignment）** 这一挑战，即如何在推理时高效、低成本地将一个通用LLM动态适配到任意新用户的特定偏好，而无需昂贵的在线重训练。

---

### **提出了什么新方法或新思路**
作者提出 **Spectral Souping**，一个统一的、高效的在线偏好对齐框架，其核心创新在于：

- **理论发现：通用谱表示（Universal Spectral Representation）**  
  首次证明，在语言马尔可夫决策过程（Language MDP）中，LLM策略的logits存在于一个由参考模型特征构成的**低维、结构化的潜空间（spectral space）**。这意味着任何个性化策略都可以被表示为一组基础策略的线性组合。

- **两阶段方法论**：
  1. **离线阶段（Offline Phase）**：预先训练一组 **K 个专业化策略（specialized policies）**，每个策略专注于一个细粒度的偏好维度（如“帮助性”、“诚实性”等）。
  2. **在线阶段（Online Phase）**：通过一个轻量级的“**汤化（souping）**”算法，在推理时动态地将这些基础策略**合并（merge）** 成一个针对当前用户的定制化策略。合并方式有两种：
     - **显式汤化（Explicit Souping）**：直接线性组合各策略的输出logits。
     - **隐式汤化（Implicit Souping）**：通过对参考策略进行拒绝采样（rejection sampling）实现。

- **理论保证**：推导出该方法的**次优性性能边界（sub-optimality bounds）**，从理论上证明了Spectral Souping的性能可以逼近一个完全微调的定制化策略，这是以往启发式模型合并方法所缺乏的。

---

### **相比现有方法的优势**
- **高效性**：避免了为每个用户进行耗时的在线微调，仅需学习一个低维的混合权重向量 $\lambda$。
- **可扩展性**：一套离线训练的基础策略库可服务于无数在线用户。
- **灵活性**：可通过增减基础策略来适应新的偏好维度。
- **理论严谨性**：提供了形式化的性能保证，而非仅凭经验验证。
- **性能优越**：实验证明其性能显著优于现有的个性化对齐方法。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
论文在三个真实世界的LLM个性化任务上进行了评估：

1. **UltraFeedback**：基于指令遵循的对话数据集，响应被标注了4D特征向量（helpfulness, honesty, instruction-following, truthfulness）。用于合成多样化的用户偏好。
2. **Text-to-Image (T2I) Generation**：在PASTA框架下进行5轮交互式文本生成图像任务。环境使用Stable Diffusion XL生成图像，代理使用Gemini 1.5 Flash进行提示词扩展。
3. **LLM Sleep Coaching**：基于LifeSnaps数据集构建的睡眠教练场景。合成用户具有来自真实个体的详细健康档案，并根据“大五人格”（Big Five）特质生成不同偏好的对话数据。

---

### **实验设置和评估指标**
- **离线阶段**：
  - 在UltraFeedback上训练 $K=30$ 个基础策略。
  - 在T2I上训练 $K=32$ 个基础策略。
  - 在Sleep Coaching上训练 $K=15$ 个基础策略（对应5种人格 × 3种强度）。
  - 使用 **Bradley-Terry logistic loss** 进行离线偏好对齐训练。
- **在线阶段**：
  - 测试模型对 **5个未见过的模拟用户（held-out users）** 的适应能力。
  - 使用在线用户反馈（偏好对或评分）来学习汤化权重 $\lambda$。
  - 采用 **凸优化方法**（如逻辑回归）高效更新 $\lambda$。
- **评估指标**：
  - 主要指标为 **测试时训练性能（Test-time Training Performance）** 和 **最终评估性能（Evaluation Performance）**，通常以归一化得分或与理想策略的接近程度衡量。
  - 性能以 **达到“定制化RLHF”（bespoke RLHF）性能的百分比** 来报告。

---

### **基线方法对比**
- **Bespoke RLHF**：为每个用户单独进行RLHF微调，作为性能上限（upper bound）。
- **P-SOUPS (Personalized Soups)**：训练多个专用策略并在参数空间进行加权平均合并。
- **PAD (Personalized Alignment at Decoding-time)**：在解码时通过偏好奖励向量引导生成。
- **PAD-SF (PAD with Successor Features)**：PAD的变体，使用类似优势函数的机制进行重加权。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- Spectral Souping 在所有三个基准上均表现出色，性能**逼近计算成本高昂的Bespoke RLHF**：
  - **UltraFeedback**: 达到Bespoke RLHF性能的 **83%**。
  - **T2I Generation**: 达到Bespoke RLHF性能的 **88%**。
  - **Sleep Coaching**: 达到Bespoke RLHF性能的 **72%**。

---

### **与基线方法的对比结果**
- **显著优于P-SOUPS**：即使调整了P-SOUPS的超参数，Spectral Souping 仍表现更优，证明了在**结构化的谱表示空间**内进行策略合并的有效性，远胜于在原始参数空间的盲目插值。
- **优于PAD和PAD-SF**：隐式Spectral Souping 比PAD系列方法更稳定、性能更好，表明使用最优Q函数权重指导策略混合，比使用原始奖励权重或启发式优势近似更有效。
- **显式 vs 隐式汤化**：两种方法性能相当，但**显式汤化（SS-Exp）略占优势**，因为它避免了隐式方法中因高拒绝率导致的采样效率下降问题。

---

### **消融实验结果**
- **基础策略数量（K）的影响**：
  - 性能随 $K$ 增加而提升，但在 $K$ 减少到某一阈值以下时出现**显著性能下降**：
    - UltraFeedback: $K < 7$
    - T2I: $K < 5$
    - Sleep Coaching: $K < 13$
  - 这表明存在一个**最小的基础策略集合**，足以张成偏好空间。
- **模型大小的影响**：
  - 更大的模型（如Gemma-V3 4B）在减少基础策略时性能下降更平缓，说明**更大的预训练模型能捕获更全面的谱表示**，使得基础策略更具表达力和鲁棒性。

---

## **4. 关键结论和发现**

### **论文的主要发现**
1. **存在通用谱表示**：LLM的个性化策略并非杂乱无章，而是存在于一个由参考模型定义的、低维的谱表示空间中。
2. **策略可线性组合**：任何个性化策略都可以通过少量基础策略的线性组合（“汤化”）来高效近似。
3. **高效且高性能**：Spectral Souping 提供了一种**计算高效、可扩展且性能优越**的在线个性化解决方案，性能逼近全量微调。
4. **理论与实践结合**：首次为模型“汤化”方法提供了**严格的理论性能边界**，填补了该领域理论空白。

---

### **方法的局限性**
- **性能边界非紧致**：理论上的次优性边界并不紧致（not tight），即使主误差项消失，剩余的奖励近似误差和负权重惩罚项仍会导致性能差距。
- **依赖基础策略覆盖度**：性能高度依赖于离线阶段训练的基础策略是否能充分覆盖潜在的用户偏好空间。
- **谱表示的发现是后验的**：目前的谱表示是从强大的预训练LLM中“发现”的，而非在预训练时主动学习。

---

### **未来工作方向**
1. **在预训练中学习最优谱表示**：将通用的偏好基础嵌入到基础模型本身，使其天生具备更强的可适配性。
2. **更复杂的在线适应算法**：探索非线性“汤化”技术或元学习（meta-learning）方法，以从稀疏反馈中更快地推断用户需求。
3. **扩展到其他模态**：将Spectral Souping的原理应用于多模态任务，如个性化的文本到图像生成，实现跨领域的鲁棒个性化。

</details>

---

### 8. [ShapeBench: A Scalable Benchmark and Diagnostic Suite for Standardized Evaluation in Aerodynamic Shape Optimization](https://arxiv.org/abs/2605.20763)

**Authors**: Shaghayegh Fazliani, Krissh Chawla, Jack Guo, Yiren Shen, Matthias Ihme, Madeleine Udell  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.20763v1  

#### Abstract
Rapid progress in aerodynamic shape optimization (ASO) has outpaced currently-available standardized evaluation frameworks. Fair comparison requires a unified benchmark spanning diverse shape classes, objective formulations, and matched-budget state-of-the-art baselines. We introduce ShapeBench, an ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ShapeBench: A Scalable Benchmark and Diagnostic Suite for Standardized Evaluation in Aerodynamic Shape Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前的 **Aerodynamic Shape Optimization (ASO)** 领域缺乏统一、标准化的评估框架，导致以下问题：
- 不同研究之间的比较不公平（因任务设定、预算、接口不一致）；
- 多数基准局限于二维空气动力学设计（如 airfoil），缺乏对三维复杂几何体（如 BWB、全机配置）的支持；
- 依赖 **Surrogate 模型** 加速搜索，但存在 **Surrogate Exploitation** 风险（即优化器找到在代理模型中表现优异但在真实 CFD 中无效的设计）；
- 缺乏对 **Large Language Model (LLM)-driven optimizers** 在 ASO 中系统性评估的标准协议。

### 提出的新方法与创新
作者提出了 **ShapeBench**——一个开源、可扩展、跨领域的 ASO 综合评估基准与诊断套件，其核心创新包括：

#### ✅ 主要贡献
1. **统一的 ASO 评估平台**
   - 包含 **103 个实例化任务**，覆盖 **8 种气动形状类别**（从 2D airfoil 到 3D 全机设计），涵盖单目标/多目标、多工况（multi-point）、混合变量（mixed-variable）等多种优化范式。
   - 提供统一的 Python API 接口，支持不同优化器无缝切换。

2. **成对的求解器配置（Surrogate + High-Fidelity CFD）**
   - 每个任务均配备经过验证的快速 **Surrogate 模型**（用于高效搜索）；
   - 尽可能提供对应的高保真 **CFD 流水线**（用于最终验证），实现 **Fidelity-Gap Analysis**（精度差距分析）。

3. **引入新型 LLM 专用优化器：ShapeEvolve**
   - 提出一种专为 ASO 设计的 LLM 进化算法 **ShapeEvolve**，结合了：
     - **ASO 结构化 Prompt** 和 **流场反馈注入**；
     - **视觉上下文输入**（几何与流场图）；
     - **持久反思记忆机制**（persistent reflection scratchpad）积累设计知识；
     - 第二阶段由 LLM 生成局部搜索脚本（Python optimizer code），提升搜索效率。

4. **诊断套件（Diagnostic Suite）**
   - 构建两阶段诊断流程：
     1. **确定性检查**：参数边界合规、网格完整性等；
     2. **LLM 综合判断器**：整合证据包、图像与运行上下文，输出可信度报告。
   - 可识别 **Surrogate Exploitation**、**几何失真**、**物理不可信** 等失败模式，并建议修复措施（如添加地面间隙约束）。

5. **标准化且公平的评估协议**
   - 所有优化器在相同计算预算下进行比较；
   - 支持经典优化器（Adjoint, L-BFGS-B, PSO, CMA-ES, Bayesian Optimization）与 LLM 驱动方法（OpenEvolve, ShinkaEvolve）的公平对比。

#### ✅ 相比现有方法的优势
| 方面 | 现有基准（如 ADODG, AFBench, EngiBench） | ShapeBench |
|------|----------------------------------------|-----------|
| 几何多样性 | 单一或少数几类（如仅 airfoil） | 覆盖 8 类，含 3D 飞机、汽车等 |
| 多保真支持 | 多为纯 CFD 或纯 Surrogate | 每任务配对 Surrogate + CFD 验证 |
| 评估一致性 | 各自为政，接口不统一 | 统一 API + 标准预算协议 |
| LLM 支持 | 无专门 LLM 基线 | 提供通用与领域专用 LLM 基线（ShapeEvolve） |
| 失败诊断 | 无 | 内置 Diagnostic Suite 检测不可信设计 |

---

## 2. 核心实验方法和设置

### 数据集
ShapeBench 整合并扩展了多个公开与自研数据集，主要包括：

| 名称 | 描述 | 来源 |
|------|------|------|
| **NeuralFoil** | 基于 XFOIL 的 airfoil 气动性能 Surrogate | [41] |
| **SuperWing** | 跨音速三维机翼数据集与 Surrogate | [52] |
| **BlendedNet** | 混合翼身（BWB）气动数据集与 Surrogate | [47] |
| **DrivAerStar** | 工业级乘用车外部气动 CFD 数据集（12,000 例） | [33] |
| **VortexNet** | Delta Wing 多保真 Surrogate | [45] |
| **COCOANet (Ours)** | 自研 Collaborative Combat Aircraft (CCA) Surrogate，基于 Transolver 架构训练于 3,570 个高保真 CFD 模拟 | 本文提出 |

此外还包括两个混合变量概念设计任务：
- **CERAS**：基于 A320 的中央参考飞机系统，含连续/离散/分类变量；
- **STA**：超音速运输机设计，含是否带前翼、T 尾等布尔决策。

### 实验设置与评估指标

#### 评估协议
- 所有优化器在 **固定评估次数预算** 下运行（通常为数千次函数调用）；
- 使用 **统一接口 `simulate()`** 获取目标值与约束；
- 所有结果记录完整配置文件以保证可复现性。

#### 性能指标
| 指标 | 定义 | 用途 |
|------|------|------|
| **Objective Value** | 最终优化目标函数值（如最小化 $C_D$ 或最大化 $L/D$） | 衡量优化效果 |
| **Rank Trajectory** | 不同预算下的方法排名变化 | 分析稳定性与收敛速度 |
| **Spearman Correlation** | 不同任务间优化器排名的相关性 | 判断泛化能力 |
| **Fidelity Gap** | Surrogate 预测 vs CFD 真实值的误差（如 % error in $L/D$） | 评估 Surrogate 可靠性 |
| **Physical Credibility Score** | 通过 Diagnostic Suite 输出的可信度等级 | 识别 Surrogate Exploitation |

#### 对比的基线方法
共测试 **8 种优化器**，分为三类：

| 类别 | 方法 | 实现说明 |
|------|------|----------|
| **Gradient-based** | Adjoint (IPOPT), L-BFGS-B | 使用自动微分或有限差分梯度 |
| **Derivative-free** | PSO, CMA-ES, Bayesian Optimization (BO) | 黑箱优化标准基线 |
| **LLM-driven** | OpenEvolve, ShinkaEvolve, **ShapeEvolve (ours)** | LLM 引导的进化搜索 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### 📊 跨任务性能差异显著（核心发现）
- 在所有任务上的平均 **成对 Spearman 秩相关系数仅为 0.013**，表明：
  > **单一任务的最优方法无法推广到其他任务**，传统“一个任务定胜负”的结论不可靠。

#### 🔁 优化器排名随任务与预算动态变化
- 图 4b 显示，在不同相对预算下，各优化器的中位排名剧烈波动；
- 例如：
  - **Bayesian Optimization** 在 CERAS 和 Delta Wing 上表现最好，但在 CCA 上最差；
  - **LLM 方法**（如 ShapeEvolve）在 CCA 上远超传统方法，但在其他任务上并无优势。

#### ⚙️ ShapeEvolve 表现突出（尤其在复杂任务）
- 在 **3D CCA 任务** 中：
  - **ShapeEvolve 达到 $L/D = 84.7$**，显著优于第二名（Bayesian Opt. $L/D = 43.2$）；
  - 传统方法普遍卡在 $L/D < 40$；
  - 归因于其更大的设计空间探索能力和 LLM 对复杂结构的理解。

#### 🧪 Fidelity Gap 分析
- 在 2D airfoil 多工况任务中：
  - Surrogate（NeuralFoil）预测的最佳设计在真实 CFD（XFOIL）中平均损失约 **9% 的 $L/D$**；
  - 某些极端案例误差高达 **18%**，说明 Surrogate 在边缘区域不可靠。

#### 🛠️ Diagnostic Suite 成功检测 Surrogate Exploitation
- 在 3D 汽车设计任务中：
  - 所有优化器均报告极低 $C_D$（~0.065），但诊断显示：
    - **WARNING**: 参数接近边界（boundary collapse）；
    - **REAR GROUND CLEARANCE 几乎消失**（物理上不可能）；
    - **LLM 报告判定为“HIGH Surrogate Exploitation Risk”**；
  - 建议增加显式几何约束（如 ground clearance > 5 cm）。

#### 💰 计算成本对比（每千次评估耗时）
| Optimizer | Airfoil (CPU-hr) | Car (CPU-hr) | CCA (CPU-hr) |
|----------|------------------|-------------|--------------|
| L-BFGS-B | 0.05–0.10 | 1.4–3.0 | 559 |
| Bayesian Opt. | 1.1–3.5 | 13–28 | 599 |
| PSO | 0.05–0.10 | ~2.7 | 688 |
| **ShapeEvolve** | **0.8–1.0** | **3.3–5.1** | **385** |
| OpenEvolve | 8.20 | 8.83 | 562 |

> ShapeEvolve 在多数任务中兼具高性能与较低计算开销。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **优化器性能高度依赖任务特性**  
   > “No single optimizer wins all” —— 不同任务中优化器排名不稳定（Spearman ρ ≈ 0.013），强调必须进行跨任务综合评估。

2. ✅ **Surrogate 模型虽快但易被利用**  
   > 即使 Surrogate 经过严格训练，仍可能出现 **Surrogate Exploitation**，产生物理上不合理但分数高的设计。

3. ✅ **高保真验证至关重要**  
   > Surrogate 与 CFD 之间存在显著 **Fidelity Gap**，仅靠代理模型评估可能导致误导性结论。

4. ✅ **LLM 方法展现潜力但需专用设计**  
   > 通用 LLM 优化器（如 OpenEvolve）表现不佳；而 **ShapeEvolve** 因引入领域知识与反馈机制，在复杂任务（如 CCA）中取得突破。

5. ✅ **诊断工具可有效提升研究可靠性**  
   > Diagnostic Suite 能自动识别边界坍塌、几何失真等问题，帮助研究人员避免误判。

---

### 局限性
1. ❌ **并非所有任务都具备 CFD 验证路径**  
   - 如 3D Passenger Car 仅有 Surrogate（DrivAerStar + Transolver），无法执行 fidelity-gap 分析。

2. ❌ **Surrogate Exploitation 仍是挑战**  
   - 即便使用诊断工具，也无法完全防止优化器向低置信度区域偏移。

3. ❌ **未涵盖制造约束、不确定性鲁棒性等现实因素**  
   - 当前任务主要关注气动性能，尚未集成工艺性、轻量化、噪声、不确定性等工程需求。

4. ❌ **LLM 方法依赖提示工程与模型选择**  
   - ShapeEvolve 的性能受 LLM 能力影响较大，目前未全面测试不同 LLM 的敏感性。

---

### 未来工作方向
1. ✅ **扩展高保真验证支持**  
   - 为更多任务（尤其是汽车）构建可调用的 CFD 回放流水线。

2. ✅ **增强 Surrogate 的不确定性建模与校准**  
   - 引入贝叶斯神经网络或集成学习估计预测置信度，指导更安全的搜索。

3. ✅ **纳入更强的物理一致性正则项**  
   - 在目标函数中加入对压力梯度、分离区、涡结构的惩罚项。

4. ✅ **推动社区共建与插件式任务扩展**  
   - 支持外部贡献者以插件形式添加新任务，保持基准演进活力。

5. ✅ **发展面向多学科优化（MDAO）的统一接口**  
   - 将结构、重量、控制、噪声等模块集成进 ShapeBench 框架。

---

> 🔗 **项目资源**  
> - GitHub: [https://github.com/ShapeBench/ShapeBench](https://github.com/ShapeBench/ShapeBench)  
> - Dataset: [https://huggingface.co/datasets/ShapeBench](https://huggingface.co/datasets/ShapeBench)

</details>

---

### 9. [Efficient Learning of Deep State Space Models via Importance Smoothing](https://arxiv.org/abs/2605.21108)

**Authors**: John-Joseph Brady, Nikolas Nusken, Yunpeng Li  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2605.21108v1  

#### Abstract
Latent state space systems are ubiquitous in statistical modelling, arising naturally when a time series is observed through a noisy measurement function, however training deep state space models (DSSM) at scale remains difficult. Two largely distinct strategies and literatures have developed around...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Learning of Deep State Space Models via Importance Smoothing**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
- **训练 Deep State Space Models (DSSMs)** 在大规模场景下仍然困难，尤其是在需要**判别任务**（supervised/semi-supervised）时。
- 现有两大主流方法存在明显缺陷：
  - **Auto-Encoding DSSMs**（基于 VAE 范式）：虽然可并行训练，但依赖于独立于动态过程的编码器，导致变分界松散，且难以支持监督学习。
  - **Differentiable Sequential Monte Carlo (DSMC)**：能自然支持监督损失并提供更紧的变分界，但由于其**顺序性前向传播**（sequential forward pass），在现代硬件上扩展性差，训练成本高。

### **提出的新方法：Parallel Variational Monte Carlo (PVMC)**
- **核心思想**：结合 VAE 和 DSMC 的优势，提出一种新的端到端可微粒子平滑器（end-to-end differentiable particle smoother）。
- **关键机制**：
  - 避免传统粒子滤波中的**重采样（resampling）** 步骤，从而消除时间上的顺序依赖。
  - 使用 **importance-weighted approximation** 来逼近**边际平滑后验分布**（marginal smoothing posterior），即每个时间步的状态在给定完整观测序列下的后验分布。
  - 利用 **associative prefix/suffix scans** 并行计算重要性权重，实现高效的并行化。

### **相比现有方法的优势**
| 维度 | PVMC | VAE-DSSM | DSMC |
|------|------|--------|------|
| **并行性** | ✅ 高（span 复杂度 $O(\log N \times \log T)$） | ✅ 高 | ❌ 低（顺序执行） |
| **支持监督学习** | ✅ 是 | ❌ 否 | ✅ 是 |
| **变分界紧致性** | ✅ 更紧（优于标准 VAE/IWAE） | ⚠️ 较松 | ✅ 紧 |
| **梯度无偏性** | ✅ 是 | ✅ 是 | ⚠️ 通常有偏（因重采样不可导） |
| **统计一致性** | ✅ 是（渐近收敛） | ✅ 是 | ⚠️ 不一定 |

- **速度提升显著**：在 NVIDIA RTX 4090 上，PVMC 比最快的 DSMC 方法快 **10×**，比无偏 DSMC 快 **100×**。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **Linear-Gaussian System**  
   - 用于验证 PVMC 是否能正确实现贝叶斯平滑。
   - 状态维度 $d_x = d_y = 5$，时间步 $T=501$。
   - 真实后验可通过 **RTS Smoother** 精确计算。

2. **Prey-Predator Model (Lotka-Volterra)**  
   - 非线性随机系统，模拟捕食者-猎物种群动态。
   - 观测为 Poisson 分布，状态不可直接观测。
   - 时间步 $T=256$，用于测试监督学习能力。

3. **Financial Time Series: SPX Returns**  
   - 真实世界金融时间序列（标普500指数日收益率）。
   - 数据长度约10年（2516天），训练窗口为120天。
   - 用于生成建模，评估合成数据的质量。

### **实验设置与评估指标**

#### **通用设置**
- 所有实验运行于 **NVIDIA RTX 4090 GPU**。
- 使用 **PyTorch** 实现，未启用 JIT 编译。
- 粒子数 $N=32$ 或 $64$，批量大小根据方法调整。

#### **评估指标**
| 任务 | 主要指标 |
|------|---------|
| **Linear-Gaussian** | - L2 误差（vs RTS 真实均值）<br>- Kernelized Stein Discrepancy (KSD)<br>- 运行时间 |
| **Prey-Predator** | - MSE（预测 vs 真实潜变量）<br>- Filtering MSE（用学习模型做 bootstrap PF）<br>- Squared Sliced 2-Wasserstein Distance (2-SWD)<br>- 训练失败次数（out of 20 runs） |
| **SPX Returns** | - 自相关函数（绝对收益与平方收益）<br>- Skewness 与 Kurtosis 分布<br>- 生成轨迹与真实数据的匹配程度 |

### **基线方法对比**
| 类型 | 基线方法 |
|------|--------|
| **Non-differentiable** | Kalman Filter, RTS Smoother, TFS, d-SMC |
| **DSMC-based** | Soft DPF, Stop-gradient DPF, Diffusion DPF, MDPS |
| **VAE-based** | DMM, TCVAE, P-VAE（消融） |

> 注：P-VAE 是 PVMC 架构但使用 VAE 目标函数的消融版本。

---

## **3. 主要实验结果和性能指标**

### **Linear-Gaussian State Estimation (Table 1)**
| Method | L2 Error | Time (s) | KSD |
|-------|----------|----------|-----|
| RTS Smoother (exact) | 0.0 | 0.14 | — |
| Kalman Filter | 0.132 | 0.13 | — |
| TFS | 0.501 | 25.9 | 0.410 |
| d-SMC | 0.44 | 4.00 | 2.21 |
| **PVMC (Kalman proposal)** | **0.054** | **1.88** | **0.200** |
| **PVMC (learned proposal)** | **0.052** | **1.50** | **0.199** |

- **结论**：PVMC 显著优于所有基线，在精度和速度上均领先。

### **Prey-Predator State Estimation (Table 2)**
| Method | MSE ↓ | Filtering MSE ↓ | 2-SWD ↓ | Time (m:s) | Failures ↓ |
|--------|------|------------------|----------|------------|-----------|
| Stop-gradient | 0.83±0.50 | 0.72±0.46 | 14.8±9.4 | 16:27 | 2 |
| Soft | 0.62±0.42 | 0.58±0.42 | 6.70±4.30 | 15:32 | 7 |
| Diffusion | 0.52±0.22 | 0.56±0.16 | 10.2±4.28 | 267:10 | 0 |
| MDPS | 1.20±0.55 | 1.32±0.64 | 13.5±10.0 | 26:23 | 14 |
| P-VAE | 0.43±0.06 | 1.21±0.11 | 20.9±2.6 | 1:49 | 0 |
| **PVMC** | **0.32±0.04** | **0.40±0.03** | **2.96±0.74** | **1:49** | **0** |

- **关键发现**：
  - PVMC 在所有指标上达到最优。
  - 训练稳定性远超 DSMC 方法（零失败 vs 最高达14次失败）。
  - P-VAE 虽然 MSE 尚可，但 Filtering MSE 差，说明仅学到了 proposal，未学到有效动态模型。

### **Financial Generative Modelling (SPX Returns)**
- **自相关结构**（图2）：
  - PVMC 最好地捕捉了绝对收益和平方收益的短期自相关（volatility clustering）。
  - DMM 和 Soft-DPF 几乎无法建模此特性。
- **分布形状**（图3）：
  - PVMC 生成的 skewness 和 kurtosis 分布最接近真实 SPX 数据。
  - TCVAE 和 P-VAE 倾向于低估偏度和峰度的幅度与方差。
- **直方图分析**（图8）：
  - DMM 和 Soft-DPF 生成的回报分布过宽，表明可能陷入局部最优（只拟合边缘分布，忽略路径结构）。

---

## **4. 关键结论和发现**

### **主要发现**
1. **PVMC 成功桥接了 VAE 与 DSMC 范式**：
   - 兼具 VAE 的并行效率和 DSMC 的判别能力与紧致界。
2. **首次实现了端到端可微、无偏、统计一致的并行粒子平滑器**：
   - 支持高效监督学习，同时保持理论严谨性。
3. **显著加速训练**：
   - 比最快 DSMC 快 **10×**，比无偏 DSMC 快 **100×**。
4. **更强的训练稳定性**：
   - 在复杂非线性系统中几乎不出现训练崩溃，而多数 DSMC 方法对随机种子敏感。

### **方法的局限性**
- **内存开销较高**：由于需存储中间 scan 结果用于反向传播，相比简单 VAE 内存占用更大。
- **当前假设 proposal 完全因子分解**：虽可在附录中推广至马尔可夫 proposal，但会增加计算负担。
- **对 proposal 网络设计敏感**：尽管实验中使用简单 CNN，但更优架构可能进一步提升性能（非本文重点）。

### **未来工作方向**
1. **扩展到更高维状态空间**：探索稀疏化或低秩近似以降低矩阵乘法复杂度。
2. **结合 structured inference**：将 PVMC 与 hierarchical 或 attention-based proposal 结合。
3. **应用于 real-time filtering**：研究如何将平滑目标迁移到在线推断场景。
4. **理论深化**：建立关于 dependent proposal 下收敛性的严格理论保证。

---

> **代码开源**：作者已将实现发布于 GitHub：  
> [https://github.com/John-JoB/parallel-variational-sequential-monte-carlo](https://github.com/John-JoB/parallel-variational-sequential-monte-carlo)

</details>

---

### 10. [Projecting Latent RL Actions: Towards Generalizable and Scalable Graph Combinatorial Optimization](https://arxiv.org/abs/2605.19721)

**Authors**: Franco Terranova (UL, LORIA, Inria), Guillermo Bernardez (UC Santa Barbara), Albert Cabellos-Aparicio (UPC), Nina Miolane (UC Santa Barbara), Abdelkader Lahmadi (LORIA, UL, Inria)  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.19721v1  

#### Abstract
Graph combinatorial optimization (GCO) has attracted growing interest, as many NP-hard problems naturally admit graph formulations, yet their combinatorial explosion renders exact methods computationally intractable. Recent advances in Reinforcement Learning (RL) combined with Graph Neural Networks ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Projecting Latent RL Actions: Towards Generalizable and Scalable Graph Combinatorial Optimization 总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
图组合优化（**Graph Combinatorial Optimization, GCO**）中的许多问题是 NP-hard 的，传统精确求解方法在大规模实例上计算不可行。现有的基于强化学习（**Reinforcement Learning, RL**）结合图神经网络（**Graph Neural Networks, GNNs**）的方法虽然取得进展，但仍面临两大挑战：

- **泛化能力差**：离散动作空间依赖于具体图实例，难以跨不同规模或分布的图迁移。
- **可扩展性不足**：迭代式动作评估（如 S2V-DQN）需要对每个候选动作进行独立前向传播，推理时间随动作空间线性增长，尤其在现实世界中具有超线性（super-linear）甚至指数级动作空间的任务中变得不可行。

### 提出的新方法：Projection Agents
本文提出了一种名为 **Projection Agent** 的新型 RL-GCO 框架，其核心思想是将决策过程从离散动作空间转移到连续的潜在动作嵌入空间（latent action embedding space），通过“预测-投影”机制实现高效且泛化的决策。

#### 核心创新点：
1. **连续潜在动作空间建模**  
   利用 GNN 对图元素（节点、边、路径等）生成语义丰富的嵌入，并构建一个由所有可能动作嵌入构成的连续潜在空间 $ \mathcal{U} $。

2. **单次前向传递 + 最近邻解码（One-shot Inference + NN Decoding）**  
   - **策略网络输出“目标嵌入”**：Agent 在一次前向传递中直接输出一个连续的“期望动作嵌入” $ \alpha_t \in \mathcal{U} $。
   - **最近邻搜索解码为有效动作**：通过在预构建的动作嵌入数据库中执行 **Nearest Neighbor (NN)** 搜索，找到最接近 $ \alpha_t $ 的合法离散动作作为最终决策。

3. **统一的观察与动作嵌入空间（Shared Embedding Space）**  
   使用无监督 GNN 预训练节点表示，使得观察空间和动作空间共享同一语义嵌入体系，促进公平比较与更好泛化。

4. **支持复杂结构化动作**  
   支持多变量、相互依赖的决策（如选择一条路径及其对应的链路权重），适用于真实场景（如流量工程、网络安全）。

### 相比现有方法的优势
| 特性 | 离散方法 (Discrete) | 迭代方法 (Iterative) | **Projection (Ours)** |
|------|---------------------|------------------------|--------------------------|
| 推理速度 | 快（单次前向） | 慢（$ O(|\mathcal{A}|) $） | **极快（单次前向 + 子线性检索）** |
| 泛化能力 | 差（依赖索引） | 中等（语义嵌入） | **强（共享嵌入 + 几何一致性）** |
| 动作空间支持 | 简单（节点/边） | 可处理复杂动作 | **支持超线性、结构化动作** |
| 是否支持插值 | 否 | 否 | **是（连续空间天然支持）** |
| 图大小不变性 | 部分（需填充） | 是 | **是（GNN 特性）** |

---

## 2. 核心实验方法和设置

### 数据集（Benchmark Environments）
共使用 **7 个 GCO 基准任务**，分为两类：

#### 经典基准（Classical Benchmarks）
- **TSP**（Traveling Salesman Problem）：寻找最短哈密顿回路。
- **MinVertex**（Minimum Vertex Cover）：最小顶点覆盖。
- **MaxCut**：最大化割边权值和。

#### 应用驱动的真实世界基准（Real-world Benchmarks）
- **Placement**：虚拟机放置，优化资源利用率与安全风险。
- **Cyber-Path**：网络攻击路径预测，涉及部分可观测性。
- **OSPF**：OSPF 路由协议下的链路权重调优以降低拥塞。
- **Traffic**：端到端流量工程，直接路由需求至可行路径。

> 表格 1 显示这些任务的动作空间复杂度呈超线性增长（如 Traffic 达 $ O(|V|^2 \cdot 2^{[\ldots]}) $）。

### 实验设置与评估指标

#### 评估范式：跨图泛化（Out-of-Distribution Generalization）
- 生成 **101 个多样化图实例**，按大小排序。
- 四种训练策略：
  - **S/M/L**：仅在小/中/大图上训练，在其余图上测试。
  - **V (Varied)**：K 折式训练，每 100 轮切换子集，提升多样性。

#### 主要评估指标
- **归一化得分（Normalized Score）**：  
  将解的质量映射到 [0,1] 区间，0 为最差启发式结果，1 为最优启发式结果。
- **Interquartile Mean (IQM)**：取最佳 100 次运行中第 25–75 百分位的平均值，减少异常值影响。
- **Train-Test Gap ($\Delta$)**：衡量泛化能力，越小越好。
- **推理延迟（Inference Latency）**：动作选择耗时 vs 图规模。
- **缩放指数 $\alpha$**：拟合 $ T(n) = c \cdot n^\alpha $，$\alpha$ 越低表示可扩展性越好。

### 基线方法对比
| 类型 | 方法 |
|------|------|
| **离散方法** | `P-Discrete`, `P-Discrete-M`（带掩码）、`G-Discrete`, `G-Discrete-M` |
| **迭代方法** | `Iterative`（基于 DQN 的逐动作评估） |
| **本文方法** | `Projection (Ours)`（PPO + z-score 归一化 + K=1 NN 解码） |

所有方法均基于 **Stable-Baselines3 (SB3)** 实现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| Benchmark | Best Baseline (IQM) | **Projection (IQM)** | Improvement |
|----------|----------------------|-------------------------|-------------|
| **TSP** | 0.99 (`Iterative`) | 0.78 | — |
| **MinVertex** | 0.95 (`Projection`) | **0.95** | ✅ 超越 |
| **MaxCut** | 0.95 (`Projection`) | **0.95** | ✅ 并列最优 |
| **Placement** | 0.78 (`G-Discrete-M`) | **0.91** | **↑ ~17%** |
| **Cyber-Path** | 0.88 (`Projection`) | **0.88** | ✅ 最优 |
| **OSPF** | 0.89 (`Projection`) | **0.89** | ✅ 最优 |
| **Traffic** | 0.84 (`Projection`) | **0.84** | ✅ 最优 |

> **结论**：`Projection` 在 **6/7 个基准上达到或超过最佳性能**，尤其在真实世界任务中显著领先。

### 泛化能力对比
- `Projection` 在 **MinVertex、MaxCut 和所有现实任务** 上表现出最强的跨图泛化能力。
- `Iterative` 在 TSP 上表现最好，但在动作空间超线性增长的任务（如 Traffic）上性能急剧下降。
- `Discrete` 方法普遍泛化能力弱，尤其在未见过的大图上。

### 可扩展性（Scalability）分析（Table 3 & Figure 6）
使用幂律模型 $ T(n) = c \cdot n^\alpha $ 拟合动作选择时间：

| Benchmark | Iterative ($\alpha$) | **Projection ($\alpha$)** |
|----------|------------------------|----------------------------|
| **TSP** | 0.25 | **0.17** |
| **Placement** | 1.81 | **0.22** |
| **Traffic** | 4.68 | **0.73** |

> **结论**：`Projection` 的推理时间增长远慢于 `Iterative`，尤其在 **超线性动作空间任务中提速高达 16.2×**。

### 消融实验与补充发现
- **解码策略**：即使使用最简单的 **K=1 NN 解码**，`Projection` 仍能取得优异性能，说明潜在空间几何结构良好。
- **训练成本**：`Projection` 训练开销较高（见 Table 5），但推理效率极高，适合部署。
- **动作空间密度**：UMAP 可视化（Figure 5）显示真实任务动作空间更密集、重叠更多，对精确匹配要求更高。

---

## 4. 关键结论和发现

### 主要发现
1. **Projection Agent 实现了高效与泛化的统一**：  
   通过在连续潜在空间中进行单次预测并解码，实现了 **快速推理** 与 **强跨图泛化能力** 的平衡。

2. **解决了超线性动作空间的可扩展性瓶颈**：  
   在 Traffic 等任务中，`Iterative` 方法因逐动作评估而无法扩展，而 `Projection` 保持稳定性能。

3. **共享嵌入空间提升了公平性与可比性**：  
   分离表示学习与策略学习，使不同 RL 方法可在相同语义基础上比较。

4. **开放工具促进社区发展**：  
   发布 **LaGCO-RL** 开源库，支持自动化潜在空间构建与新 GCO 任务集成。

### 局限性
1. **编码与解码策略受限**：目前采用单一嵌入方式和简单 NN 解码，未探索更复杂的聚合或学习型解码器。
2. **嵌入未微调**：GNN 编码器为无监督预训练，未针对下游 RL 任务进行微调，可能限制峰值性能。
3. **评估范围有限**：仅对比 RL 方法，未涵盖监督学习或混合方法；实验规模为 101 实例 × 20 运行。

### 未来工作方向
- 探索更强大的解码机制（如学习评分函数、beam search）。
- 引入自适应嵌入更新机制，在线优化潜在空间。
- 扩展至动态图环境与多智能体协同优化场景。
- 结合模仿学习或课程学习加速收敛。

--- 

> **总结**：该论文提出的 **Projection Agent** 为 RL-GCO 提供了一个**通用、可扩展且高性能**的新范式，特别适用于现实世界中复杂、高维、超线性增长的动作空间任务，同时发布的 **LaGCO-RL** 库将进一步推动该领域的标准化与可复现研究。

</details>

---

### 11. [LT2: Linear-Time Looped Transformers](https://arxiv.org/abs/2605.20670)

**Authors**: Chunyuan Deng, Yizhe Zhang, Rui-Jie Zhu, Yuanyuan Xu, Jiarui Liu, T. S. Eugene Ng, Hanjie Chen  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.20670v1  

#### Abstract
Looped Transformers (LT) have emerged as a powerful architecture by iterating their layers multiple times before decoding the final token. However, pairing them with full attention retains quadratic complexity, making them computationally expensive and slow. We introduce LT2 (Linear-Time Looped Tran...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LT2: Linear-Time Looped Transformers

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前的 **Looped Transformers (LT)** 虽然通过权重共享实现参数高效（parameter-efficient），但由于每轮循环中仍使用 **quadratic softmax attention**，其训练和推理的计算复杂度随序列长度呈平方增长。这导致在长上下文场景下，**FLOPs 和 KV-cache 内存开销急剧上升**，严重限制了模型的可扩展性和实用性。

### 提出了什么新方法或新思路
本文提出 **LT2 (Linear-Time Looped Transformers)**，一个将 **subquadratic token mixer** 引入 Looped 架构的新家族，从根本上解决注意力瓶颈问题。具体包括：

- **LT2-linear**：使用 **linear attention**（如 GDN、KDA）替代 softmax attention。
- **LT2-sparse**：使用 **sparse attention**（如 DSA）构建滑动窗口机制。
- **LT2-hybrid**：混合多种 attention 变体，在循环的不同阶段或深度层级中组合使用，例如：
  - `LT2-hybrid (GDN+DSA)`：结合线性与稀疏 attention，完全避免二次项。
  - `LT2-hybrid (Full+GDN)`：少量 full attention 层 + 大量 GDN 层，兼顾性能与效率。

此外，作者还展示了如何将预训练好的 Looped Transformer **蒸馏为 LT2-hybrid 模型**，无需从头训练即可获得线性时间效率。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **效率** | 推理解码吞吐量提升 **5.7×**（8k 上下文，batch=8），KV-cache 内存恒定，支持更长上下文（如 32k） |
| **性能** | 在语言建模、零样本任务上达到甚至超越标准 Looped Transformer |
| **可迁移性** | 支持对已有 Looped 模型进行高效转换，降低部署门槛 |
| **稳定性** | GDN 等结构具备更好的训练稳定性和梯度控制 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主预训练数据**：`FineWeb-Edu`（100B tokens）
- **下游评估基准**：
  - 零样本任务：`ARC-E/C`, `HellaSwag`, `Winogrande`, `MMLU`, `GSM8K`, `PIQA`, `OBQA`, `BoolQ`, `SciQ`
  - 长上下文检索：`SWDE`, `SQuAD`, `FDA`, `TriviaQA`, `Natural Questions`, `DROP`
  - 合成任务：State-tracking + Recall（自定义指针交换程序）
  - “大海捞针”测试：`NIAH-Single-1/2/3`（1k–4k 上下文）

### 实验设置和评估指标
| 设置项 | 描述 |
|--------|------|
| **模型规模** | 0.6B 和 1.3B 参数 |
| **循环次数 T** | 固定为 4 次（pre-training 阶段） |
| **序列长度** | 最高支持 32k tokens |
| **硬件平台** | 单张 H100 GPU（80GB） |
| **评估指标** |
| - Perplexity (PPL) | 语言建模能力 |
| - Zero-shot Accuracy (%) | 下游任务表现 |
| - Decode Throughput (tok/s) | 解码速度（batch=8） |
| - OOM Threshold | 最大可处理上下文长度 |
| - Gradient Norm / Loss Curve | 训练稳定性分析 |

### 基线方法对比
- **Standard Transformer**
- **Looped Transformer (ref)**：带 full attention 的标准循环架构
- **Non-looped subquadratic models**：如 RetNet、Mamba2、DSA 等非循环版本
- **Industry-level models**：Gemma3-1B, Qwen2.5-1.5B, Llama3.2-1B/3B 等开源小模型作为外部参考

---

## 3. 主要实验结果和性能指标

### 关键性能数据（1.3B 规模）

| 模型 | PPL ↓ | 平均零样本得分 ↑ | 解码吞吐量 (8k, bs=8) | 是否突破 32k OOM |
|------|-------|------------------|------------------------|------------------|
| Looped Transformer (ref) | 9.87 | 59.27% | 22 tok/s | ❌（OOM @8k） |
| **LT2-hybrid (GDN+DSA)** | **9.50** | **60.73%** | **125 tok/s** | ✅ |
| **LT2-hybrid (Full+GDN)** | **9.12** | **62.89%** | **110 tok/s** | ✅ |
| Ouro-hybrid-1.4B (converted) | – | ≈60%+ | ~linear-time | ✅ |

> 💡 **说明**：LT2-hybrid (GDN+DSA) 在完全无 full attention 的情况下，**匹配原 Looped Transformer 性能，同时提速 5.7×**；而 (Full+GDN) 版本则进一步将平均得分提升 **+2.1 pts**，并保持约 5× 的加速。

### 与基线方法的对比结果
- **相比标准 Looped Transformer**：
  - 所有 LT2 变体在相同参数量下均取得相当或更优的语言建模和零样本性能。
  - **Hybrid (Full+GDN)** 显著优于所有 baseline，在多个任务上接近甚至超过 4B 级别工业模型。
- **相比行业级 1B 模型**：
  - 蒸馏得到的 `Ouro-hybrid-1.4B` 在仅用 ~1B tokens 微调后，全面超越同级别 1B 模型，并媲美 3B–4B 模型。

### 消融实验结果（Table 3）

#### （1）Hybrid Ratio（Full : GDN）
| 比例 | PPL | 平均准确率 |
|------|-----|------------|
| 1:0 (Full-only) | 9.87 | 59.27 |
| 1:1 | 9.41 | 60.92 |
| **1:4 (default)** | **9.31** | **61.39** |
| 1:6 | 9.36 | 61.07 |
| 1:12 | 9.74 | 59.51 |

✅ **结论**：存在明显的 **inverse-U 曲线**，最优比例为 **1:4**，过多或过少 full attention 均会损害性能。

#### （2）Hybrid Pattern（固定 1:4）
| 模式 | PPL | 准确率 |
|------|-----|--------|
| Interleave (default) | 9.31 | 61.39 |
| Bookend (首尾 Full) | **9.27** | **61.52** |
| Front-loaded | 9.45 | 60.61 |
| Back-loaded | 9.53 | 60.43 |

✅ **结论**：“分散分布”优于“集中分布”，**bookend 设计略优**于均匀交错。

#### （3）Hybrid Level（depth vs loop）
| 方式 | PPL | 准确率 |
|------|-----|--------|
| Depth-level (fixed interleave) | 9.31 | 61.39 |
| Loop-level coarse→fine | 9.36 | 60.71 |
| Loop-level fine→coarse | 9.42 | 61.10 |
| Random-sample + vote (K=5) | **9.26** | **61.55** |

⚠️ **结论**：跨 loop 的动态混合收益有限，且随机投票需 5× 推理成本，**推荐使用固定的 depth-level interleaving**。

---

## 4. 关键结论和发现

### 主要发现
1. **Looping 与 subquadratic attention 具有独特协同效应**：
   - 对 **linear attention**：T 次循环将 rank-1 更新变为 **rank-T 更新**，增强表达能力。
   - 对 **sparse attention**：T 次循环使有效感受野从 `O(w)` 扩展到 `O(Tw)`，实现“compute-to-context”转化。

2. **Hybrid 架构开辟新的 Pareto 前沿**：
   - `LT2-hybrid (GDN+DSA)`：以纯线性代价 **匹配 full-attention 质量**
   - `LT2-hybrid (Full+GDN)`：以显著更快的速度 **超越 full-attention 性能**

3. **训练更稳定**：
   - GDN 中的 **data-dependent gating** 和 **delta rule** 有效抑制梯度爆炸和激活累积。
   - 引入 **SDPA output gate** 可缓解 attention sink 在循环中的累积效应。

4. **可高效蒸馏已有模型**：
   - 将 `Ouro-1.4B` 蒸馏为 `Ouro-hybrid-1.4B`，仅需约 1B tokens 训练，即可保留教师模型质量，同时继承 LT2 的线性效率。

### 方法的局限性
- **未探索 full loop-level hybridization**：不同循环迭代是否应使用不同的 mixer 类型仍未充分研究。
- **缺乏显式的跨 loop state carry mechanism**：当前状态传递依赖隐式残差连接，可能限制长期记忆复用。
- **Adaptive Computation Time (ACT)**：虽理论上更强大，但在大规模 pre-training 中存在优化不稳定、吞吐下降等问题，目前采用固定 T=4。

### 未来工作方向
- 设计更灵活的 **per-loop mixer scheduling** 策略。
- 探索 **explicit recurrent state carry** 机制以增强跨循环信息流动。
- 开发稳定的 **ACT 变体**，实现输入自适应的动态计算分配。
- 进一步推动 **small yet capable language models** 的发展，结合 LT2 架构打造高效推理引擎。

---

> 🔗 **代码与模型**：
> - GitHub: [https://github.com/chili-lab/LT2](https://github.com/chili-lab/LT2)
> - HuggingFace: [https://huggingface.co/chili-lab/Ouro-hybrid-1.4B](https://huggingface.co/chili-lab/Ouro-hybrid-1.4B)

</details>

---

### 12. [Optimized Federated Knowledge Distillation with Distributed Neural Architecture Search](https://arxiv.org/abs/2605.21322)

**Authors**: Chaimaa Medjadji, Sylvain Kubler, Yves Le Traon, Guilain Leduc, Sadi Alawadi, Feras M. Awaysheh  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2605.21322v1  

#### Abstract
Federated Learning (FL) enables collaborative model training without centralizing data. However, real-world deployments must simultaneously address statistical heterogeneity across client data (non-IID), system heterogeneity in device capabilities, and communication efficiency. Existing FL approache...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **Optimized Federated Knowledge Distillation with Distributed Neural Architecture Search**  
**核心结论与实验结果总结**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
该论文针对 **Federated Learning (FL)** 在现实部署中面临的三大挑战：
- **统计异质性 (Statistical heterogeneity)**：客户端数据非独立同分布（non-IID），导致模型漂移（client drift）和收敛不稳定。
- **系统异质性 (System heterogeneity)**：设备在计算能力、内存、能耗等方面差异巨大，难以统一部署相同模型架构。
- **通信效率 (Communication efficiency)**：频繁传输完整模型参数开销大，尤其在带宽受限场景下成为瓶颈。

现有方法通常假设客户端使用**固定架构**，无法同时适应数据复杂性和硬件约束，导致准确率与效率之间的次优权衡。

---

### **提出的新方法与创新思路**
作者提出了 **FedKD-NAS**，一个将 **Knowledge Distillation (KD)** 与 **Neural Architecture Search (NAS)** 结合的新型联邦学习框架。其核心创新在于：

- **将模型架构视为一等优化变量**：不再假设所有客户端使用相同架构，而是允许每个客户端**自主选择轻量级模型**，以适应本地数据分布和资源限制。
- **去中心化的 NAS 机制**：客户端在本地从预定义搜索空间中选择最优架构，无需全局协调或权重共享，降低了计算和通信开销。
- **基于预测的协作协议**：客户端仅向服务器上传在公共参考集上的软预测（soft predictions/logits），而非模型参数，实现**无参数共享的协作**。
- **稳定的知识传递机制**：服务器聚合客户端预测，并结合一个固定的教师模型（teacher model）进行平滑处理，生成稳定的蒸馏目标，有效缓解非IID下的客户端漂移。

---

### **相比现有方法的优势**
| 维度 | FedKD-NAS 优势 |
|------|----------------|
| **灵活性** | 支持异构客户端架构，适应不同设备能力。 |
| **通信效率** | 仅交换 logits，通信开销远低于参数聚合方法（如 FedAvg）。 |
| **鲁棒性** | 在非IID数据下表现更稳定，准确率下降更小。 |
| **资源效率** | 客户端模型更轻量化，降低 CPU 和内存占用。 |
| **可扩展性** | 无需超网（supernet）训练或全局架构同步，适合大规模联邦系统。 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共使用 **6 个数据集**，涵盖图像、时间序列等多种模态：
- **图像分类**：MNIST, FMNIST, EMNIST, CIFAR-10, CIFAR-100
- **时间序列活动识别**：CASA（真实世界传感器数据）

每个数据集均模拟三种划分方式以测试异质性影响：
- **IID**：数据均匀划分
- **Dirichlet (α=0.1)**：轻度非IID
- **Shards**：极端标签偏斜（label skew）

---

### **实验设置与评估指标**

#### **客户端模型配置**
根据不同数据集复杂度，客户端采用不同轻量级模型：
- MNIST/FMNIST/EMNIST：LeNet5 或 ResNet18
- CIFAR-10/100：MobileNetV2 或 ShuffleNetV2-x0.5
- CASA：DeepConvLSTM

服务器端使用预训练的教师模型提供指导。

#### **评估指标**
论文引入了多个复合指标以全面衡量部署效率：

| 指标 | 含义 | 方向 |
|------|------|------|
| **Acc** | 准确率 | ↑ |
| **Loss** | 测试损失 | ↓ |
| **CPU / MEM** | 客户端 CPU 使用率 / 内存占用 | ↓ |
| **Comm** | 通信体积（字节） | ↓ |
| **RES** | Resource Efficiency Score，综合 CPU/MEM/Comm | ↓ |
| **PQS** | Performance Quality Score，加权 Acc 和 Loss | ↑ |
| **CES** | Communication Efficiency Score，归一化通信成本倒数 | ↑ |
| **UES** | Unified Efficiency Score = PQS × CES / RES | ↑ |

---

### **基线方法对比**
与 **6 类代表性 FL 基线** 进行比较：
- **参数聚合类**：FedAvg, Ditto
- **个性化 FL**：Ditto
- **知识蒸馏类**：FedDistill, FedMD, FedDF, Local-KD
- **NAS + KD 类**：RaFL, AdaptFL（未复现，故未直接对比）

> 注：FedKD-NAS 是首个在联邦设置中**同时实现分布式 NAS 与 logit 级蒸馏**的方法。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
在 **CIFAR-10 非IID 设置下**，FedKD-NAS 表现显著优于基线：

| 指标 | FedKD-NAS | 最佳基线 | 提升幅度 |
|------|-----------|----------|---------|
| **Accuracy** | +7–11% | FedDistill | ↑ |
| **CPU Usage** | ~34% | ~47% | ↓ **28%** |
| **Communication Volume** | ~1.62 MB | ~71.6 MB (FedAvg) | ↓ **44×** |
| **UES** | 62.73 (Dirichlet) | 38.51 (FedMD) | ↑ **60%** |

在 **CASA 时间序列任务** 上：
- **准确率提升 15%**（0.9442 vs. 0.8206）
- **UES* 达到 14.72**，是 FedDistill 的 **2.5 倍**，FedAvg 的 **183 倍**

在 **真实设备部署**（RockPi, Jetson Nano, Raspberry Pi）上：
- **准确率 +8%~11%**
- **CPU 使用率最低（90.53%）**
- **通信减少 1.86×**
- **UES 达 1.88**，是 FedMD 的 **2.6 倍**，FedAvg 的 **6 倍**

---

### **与基线方法的对比结果**
- **在 35 个实验配置中，FedKD-NAS 在 31 个达到最高 UES**。
- **在非IID条件下优势最明显**，且随着异质性增加，其相对优势持续扩大。
- **在 IID 条件下略逊于 FedAvg**，但在实际场景中因异质性普遍存在，此劣势不构成主要问题。
- **在高类别数（如 EMNIST 47类）时，logit 通信可能大于模型参数**，此时 FedAvg 通信更优，但 FedKD-NAS 仍保持更高准确率。

---

### **消融分析与关键发现**
虽然未设独立“消融实验”章节，但通过以下观察可推断各组件作用：
- **NAS 控制器**：负责 PQS 提升，使模型适配本地数据，减少过拟合。
- **KD 蒸馏机制**：实现通信压缩，带来 44× 通信节省。
- **教师模型融合**：提供稳定监督信号，防止客户端预测发散。
- **EMA 平滑**：抑制轮间波动，提升收敛稳定性。

> 实验显示，**FedKD-NAS 的优势随异质性增强而增强**，表明其设计特别适合真实世界场景。

---

## **4. 关键结论和发现**

### **主要发现**
1. **架构灵活性是关键**：将模型架构作为优化变量，而非固定设计，能显著提升联邦系统的适应性与效率。
2. **NAS + KD 协同增益**：本地架构搜索与服务器端知识蒸馏协同，实现了准确率、效率与鲁棒性的帕累托前沿突破。
3. **通信优势依赖于 C/P 比例**：当类别数 C 相对于模型参数 P 较大时，logit 通信可能反超参数通信，需谨慎选择方法。
4. **真实世界泛化能力强**：在 CASA 和真实设备上的优异表现，验证了 FedKD-NAS 在非视觉、异构硬件场景下的通用性。

---

### **方法的局限性**
1. **依赖公共参考集 (DPub)**：若 DPub 与客户端数据域不匹配，性能可能下降。
2. **高类别数下通信劣势**：在 C >> P 场景（如 EMNIST），logit 通信体积可能超过模型本身。
3. **隐私风险**：尽管只传 logits，但仍可能泄露成员信息（membership inference）。
4. **缺乏形式化拜占庭鲁棒性**：对恶意客户端提交虚假预测缺乏理论防御机制。
5. **未提供绝对能耗测量**：依赖代理指标，缺乏真实碳足迹数据。

---

### **未来工作方向**
1. **替换公共数据集**：使用生成模型合成参考数据，避免域偏移问题。
2. **高类别数优化**：引入稀疏化、Top-K 或量化 logits 以降低通信开销。
3. **增强隐私保护**：集成差分隐私（DP）于预测共享过程。
4. **拜占庭鲁棒性**：引入 Trimmed Mean 或几何中位数聚合策略。
5. **绿色AI评估**：结合 CodeCarbon 等工具进行全生命周期碳排放测算。
6. **动态 NAS 策略**：根据任务复杂度自适应调整搜索空间大小。

---

> **总结**：FedKD-NAS 是首个将 **Distributed NAS** 与 **Federated Distillation** 成功结合的框架，在准确率、通信效率、资源消耗三个维度上实现了显著帕累托改进，为构建高效、鲁棒、可持续的联邦学习系统提供了新范式。

</details>

---

### 13. [Parallel LLM Reasoning for Bias-Resilient, Robust Conceptual Abstraction](https://arxiv.org/abs/2605.20194)

**Authors**: Aisvarya Adeseye, Jouni Isoaho, Adeyemi Adeseye  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.20194v1  

#### Abstract
Large language models (LLMs) have been increasingly used to analyze text. However, they are often plagued with contextual reasoning limitations when analyzing long documents. When long documents are processed sequentially, early or dominant concepts can overshadow less visible but meaningful interpr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

**论文标题**：*Parallel LLM Reasoning for Bias-Resilient, Robust Conceptual Abstraction*

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**

该研究针对 **Large Language Models (LLMs)** 在长文本分析中的两大结构性缺陷：

1. **累积性分析偏差 (Cumulative Analytical Bias)**  
   当文本被顺序分块处理时，早期或主导概念会作为隐含先验影响后续推理，导致次要但重要的主题被忽略，产生 **遗漏错误 (omission error)** 和 **位置主导效应 (positional dominance)**，例如“lost in the middle”现象。

2. **无依据综合 (Ungrounded Synthesis)**  
   分块独立输出在合并时若缺乏严格的证据约束，容易引入 **冗余、概念漂移 (conceptual drift)** 和 **不支持的断言 (unsupported claims)**，即模型“幻觉”。

这些问题在多阶段 LLM 推理流程中尤为严重，且仅靠扩大模型规模无法根本解决。

---

### **提出了什么新方法或新思路**

提出了一种名为 **Parallel Evidence-Constrained Independent Inference (PECII)** 的结构化框架，其核心创新在于：

- **并行分块独立推理 (Parallel Chunk-Level Independent Inference)**  
  将文本划分为语义连贯的块后，并行独立处理每个块，**消除执行顺序依赖**，防止早期输出对后续推理的干扰。

- **证据锚定的综合机制 (Evidence-Anchored Consolidation)**  
  所有生成的概念必须附带可追溯的原文引用（quote），并在合并时施加显式约束：
  - **证据多样性**：要求概念来自多个不同文档。
  - **冗余控制**：禁止近似重复的引文。
  - **合并合理性验证**：要求提供合并理由，增强可审计性。

- **六层模块化架构设计**  
  包括从文本摄入、语义分块、并行推理、证据验证、全局聚类到可靠性排序的完整流程，确保端到端的可复现性和可解释性。

---

### **相比现有方法的优势**

| 维度 | 传统方法 | PECII |
|------|--------|-------|
| **推理方式** | 顺序处理（易受顺序偏见） | 并行独立处理（消除顺序依赖） |
| **证据支持** | 弱或缺失，易产生幻觉 | 显式引文+语义对齐验证 |
| **综合过程** | 简单聚合，易漂移 | 受控聚类+合并理由生成 |
| **可审计性** | 低 | 高（全程 traceability） |
| **小模型表现** | 差（易遗漏） | 显著提升，接近大模型水平 |

PECII 表明：**方法论结构设计**（而非单纯模型规模）是实现可靠、可扩展概念抽象的关键。

---

## **2. 核心实验方法和设置**

### **使用的数据集**

- **数据来源**：82 名参与者的半结构化访谈记录。
- **组织背景多样**：NGO、私营企业、大学、政府机构、医疗机构。
- **主题**：组织环境中引入游戏化机制（gamification）引发的隐私担忧。
- **数据量**：每份访谈 45–60 分钟，转录文本约 8,000–13,000 词。
- **预处理**：完全匿名化，去除所有个人身份信息（PII）。

---

### **实验设置**

#### **三种执行策略对比**：

1. **Direct Full-Transcript Execution**  
   整篇文本一次性输入（受限于模型上下文窗口）。
2. **Sequential Chunk Execution**  
   分块后按顺序处理，前一块输出影响下一块。
3. **Parallel Chunk Execution (PECII)**  
   分块后并行独立处理，互不干扰。

#### **模型选择（共6个）**：

| 模型 | 规模 | 类型 |
|------|------|------|
| LLaMA-1B | 1B | 开源小模型 |
| Qwen-1.5B | 1.5B | 开源小模型 |
| LLaMA-3B | 3B | 中等模型 |
| Qwen-4B | 4B | 中等模型 |
| LLaMA-8B | 8B | 较大开源模型 |
| ChatGPT | ~5.2B+ | 商业闭源大模型 |

目的：检验 PECII 是否能缩小大小模型之间的性能差距。

#### **评估指标**

| 指标 | 公式/说明 | 目标 |
|------|---------|------|
| **Omission Error Rate (%)** | $1 - \frac{|\Omega \cap \Omega^*|}{|\Omega^*|}$ | 越低越好 |
| **Early-Chunk Dominance Index (ECDI)** | 早期块提取概念占比 | 越低越好（减少锚定偏见） |
| **Evidence Traceability Score (%)** | 支持概念的多源引文比例 | 越高越好 |
| **Unsupported Claim Rate (%)** | 无有效证据支持的断言比例 | 越低越好 |
| **Theme Compression Ratio (TCR)** | $\frac{\text{原始候选数}}{\text{最终概念数}}$ | 越高越好（去冗余） |
| **Cross-Theme Leakage (%)** | 被分配到多个主题的片段比例 | 越低越好 |
| **Merge Justification Quality (1–5)** | 专家盲评合并理由质量 | 越高越好 |

**黄金标准 (Gold Standard)**：由两位独立研究人员使用 NVivo 进行编码，通过结构化协商达成一致的主题集。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **表1：并行处理显著降低遗漏与主导效应**

| 模型 | Omission Error ↓ | ECDI ↓ | Interpretive Novelty ↑ |
|------|------------------|--------|------------------------|
| **平均（所有模型）** | **~84% reduction** vs baseline | **>80% reduction** | **+22.6% to +56.7%** |
| **LLaMA-1B** | 36.8% → **5.9%** | 0.51 → **0.09** | 3.1 → **4.7** |
| **ChatGPT** | 22.8% → **3.7%** | 0.36 → **0.07** | 4.2 → **4.8** |

> ✅ **并行执行使遗漏错误下降约 84%，早期块主导指数下降超 80%**。

#### **表2：证据锚定大幅提升可靠性**

| 指标 | 改进幅度（跨模型） | 示例（LLaMA-1B） |
|------|------------------|------------------|
| **Evidence Traceability** | **+36% 到 +132%** | 41% → **95%** (+131.7%) |
| **Unsupported Claim Rate** | **-78% 到 -91%** | 34% → **3%** (-91.2%) |
| **Theme Compression Ratio** | +6.2 → **14.8** | 提升约 **2.4 倍** |
| **Cross-Theme Leakage** | **↓66–79%** | 29% → **6%** |
| **Merge Justification Quality** | 2.8 → **4.8** | 接近满分 |

> ✅ **证据锚定后，各模型性能趋于收敛，小模型提升最大**。

---

### **与基线方法的对比结果**

| 方法 | Omission Error | Traceability | Unsupported Claims | 跨模型方差 |
|------|----------------|-------------|---------------------|------------|
| **No Chunking** | 高（22.8–36.8%） | 低（41–71%） | 高（9–34%） | 大（14%） |
| **Sequential Chunking** | 中（~15–23%） | 中（~60–80%） | 中（~2–11%） | 中（~6%） |
| **PECII (Parallel + Anchored)** | **极低（3.7–5.9%）** | **极高（95–97%）** | **极低（2–3%）** | **极小（<2.2%）** |

> ✅ **PECII 在所有维度上全面超越基线，尤其在小模型上优势明显**。

---

### **消融实验结果**

虽然未明确列出“消融实验”章节，但从以下发现可推断出各组件作用：

- **仅分块（Sequential）**：遗漏错误下降 ~36–40%，表明分块本身有益。
- **加入并行处理**：遗漏再降 ~84%，表明 **并行是减少偏差的核心**。
- **加入证据锚定**：不支持断言率下降 >90%，表明 **证据验证是抑制幻觉的关键**。
- **小模型受益最多**：LLaMA-1B 的遗漏错误从 36.8% 降至 5.9%，相对改进远超 ChatGPT。

> ✅ **结构设计（并行 + 锚定）起到了“正则化器”作用，补偿了小模型的能力不足**。

---

## **4. 关键结论和发现**

### **主要发现**

1. **方法论结构 > 模型规模**  
   在长文本概念抽象任务中，**推理架构的设计比模型参数量更能决定最终质量**。合理的结构可以显著缩小大小模型之间的差距。

2. **并行处理是缓解累积偏见的有效手段**  
   通过隔离分块推理，**打破了自回归模型中的“条件漂移”循环**，从根本上减少了遗漏和位置偏见。

3. **证据锚定实现了结构化可靠性提升**  
   强制要求每个断言附带可验证引文，并在合并时施加多样性与冗余约束，**将 LLM 推理从“生成”转变为“受控归纳”**。

4. **小模型在结构优化下可逼近大模型性能**  
   结构化方法使小模型（如 1B 参数）在遗漏、可追溯性等指标上接近甚至媲美大模型（如 ChatGPT），为资源受限场景提供了高效替代方案。

---

### **方法的局限性**

1. **计算开销增加**  
   并行执行带来更高的内存和并发需求，尽管作者认为收益远大于成本。

2. **严格过滤可能误删罕见但有效的解释**  
   若模型未能正确提取支持引文，即使概念合理也可能被过滤。

3. **依赖高质量的分块策略**  
   语义边界识别的准确性直接影响推理质量，当前方法仍基于启发式规则。

4. **人工验证成本较高**  
   黄金标准构建依赖专家手动编码，难以大规模推广。

---

### **未来工作方向**

1. **跨领域验证**  
   在政策、法律、临床叙事等不同类型文本中测试 PECII 的泛化能力。

2. **多语言支持**  
   扩展至非英语语料库，检验其在跨语言场景下的有效性。

3. **改进证据验证机制**  
   引入基于 **entailment 检查** 或 **verifier model** 的更精确验证方式。

4. **人机协同变体**  
   设计 **human-in-the-loop** 版本，在合并阶段引入专家审核证据链接。

5. **自适应分块策略**  
   研究动态调整分块大小和边界的算法，以适配不同文体风格。

6. **计算效率优化**  
   量化并行化策略的资源消耗，探索适用于边缘设备的轻量化部署方案。

---

> **总结**：PECII 通过 **并行独立推理 + 证据锚定综合** 的结构化设计，成功解决了 LLM 在长文本分析中的核心结构性缺陷。实验证明，该方法不仅大幅提升了分析的准确性、可追溯性和鲁棒性，还使得小模型在合理架构下也能达到接近大模型的性能水平，为构建 **可靠、可审计、可复现的 LLM 分析流水线** 提供了重要范式。

</details>

---

### 14. [Divide-Prompt-Refine: a Training-Free, Structure-Aware Framework for Biomedical Abstract Generation](https://arxiv.org/abs/2605.20628)

**Authors**: Sylvey Lin, Joe Menke, Shufan Ming, Dongin Nam, Neil Smalheiser, Halil Kilicoglu  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.20628v1  

#### Abstract
Biomedical abstracts play a critical role in downstream NLP applications, such as information retrieval, biocuration, and biomedical knowledge discovery. However, a non-trivial number of biomedical articles do not have abstracts, diminishing the utility of these articles for downstream tasks. We pro...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Divide-Prompt-Refine: a Training-Free, Structure-Aware Framework for Biomedical Abstract Generation》核心总结

---

## 1. 主要贡献和创新点

### ✅ 解决的问题
- **Biomedical Abstract Generation (BAG)** 任务旨在为没有摘要的生物医学全文文章自动生成结构化摘要。
- 当前挑战：
  - 大量 PubMed 文章（约 29%）缺失摘要，影响下游 NLP 任务（如信息检索、知识发现）。
  - 全文长度远超 LLM 上下文限制，导致传统模型出现 **信息遗漏** 和 **幻觉**（hallucinations）。
  - 现有方法存在 **提取偏差**（extractive bias），生成文本缺乏连贯性。

### 🚀 提出的新方法：DPR-BAG
提出 **DPR-BAG**（Divide, Prompt, and Refine for Biomedical Abstract Generation），一种无需训练、零样本（zero-shot）、结构感知的框架，其流程如下：

1. **Divide（分割）**  
   将全文按 **BOMRC** 结构（Background, Objective, Methods, Results, Conclusions + Others）切分为语义片段。
   - 使用 LLM-SSC 模型对段落首句进行 **rhetorical labeling**，实现细粒度结构划分。

2. **Prompt（并行提示）**  
   对每个结构片段独立调用 LLM 进行摘要生成（可选加入实体引导，如 TR-UMLS 或 CoT）。

3. **Refine（精炼）**  
   将各片段摘要拼接后，通过一个最终的 LLM refinement 阶段恢复全局话语连贯性和风格一致性。

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **无需训练** | 完全 zero-shot，不依赖 fine-tuning，适用于低资源场景。 |
| **结构感知** | 显式建模 BOMRC 结构，提升摘要科学性和可读性。 |
| **长文本处理** | 分治策略有效缓解上下文长度限制，避免截断损失。 |
| **抽象性更强** | 生成更少复制原文的“新颖”摘要，接近人类写作风格。 |
| **事实一致性高** | 在保持高度抽象的同时，仍能维持与原文的事实对齐。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **PMC-MAD**：作者构建的新数据集，包含 **46,309 篇** 缺失摘要的生物医学全文文章。
  - 来源：PubMed Central (PMC) Open Access 子集。
  - 时间范围：1987–2023。
  - 构建方式：基于 publication type 分层抽样，确保分布与真实无摘要文章一致。
- **PubMed Summarization Dataset (PubMedSum)**：用于验证泛化能力。

### ⚙️ 实验设置
- **主干模型**：`Llama-3.2:3B`（via Ollama），instruction-tuned 模式。
- **硬件限制**：NVIDIA Tesla V100 GPU (32GB)，标准 baseline 受限于 8,192 token 输入。
- **消融变量**：
  - 分割策略（FS vs. NS vs. SH）
  - 提示复杂度（BC vs. DI vs. SI）
  - 实体引导机制（TR-UMLS, CoT）

### 📊 评估指标
| 类别 | 指标 | 说明 |
|------|------|------|
| **抽象性 (Abstractiveness)** | Bigram/Trigram Novelty, Density | 衡量生成内容是否为原创表达而非直接复制。 |
| **事实一致性 (Factuality)** | AlignScore, MiniCheck, SummaC | 判断生成摘要是否忠实于原文内容。 |
| **语义对齐 (Semantic Alignment)** | BERTScore, DiscoScore (SENT, FOCUS) | 衡量与参考摘要的语义相似性和话语连贯性。 |
| **支持性指标** | ROUGE-L, UMLS Recall, Coverage, Compression | 辅助评价信息覆盖与压缩程度。 |

### 🆚 基线方法对比
| 类型 | 模型 |
|------|------|
| **Zero-shot 基线** | LED-Arxiv, LED-PubMed, LongT5 (均未微调) |
| **Fine-tuned 基线** | LED-PubMed (FT), LongT5 (FT) —— 在 PMC-MAD 上监督训练 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（在 PMC-MAD 上）

#### ✅ DPR-BAG (BC) vs. Fine-tuned Baselines
| 指标 | DPR-BAG | 最佳 Fine-tuned | 差距 |
|------|---------|----------------|-------|
| **Bigram Novelty** | 0.397 | 0.309 | **+28.5% 更抽象** |
| **Trigram Novelty** | 0.605 | 0.453 | **+33.5% 更抽象** |
| **Density ↓** | 4.690 | 11.369 | **显著更低 → 更少复制** |
| **AlignScore ↑** | **0.762** | 0.511 | **+41.7% 更事实一致** |
| **MiniCheck ↑** | **0.890** | 0.653 | **+36.3% 更可靠** |

> ✅ **结论**：DPR-BAG 在抽象性和事实一致性上全面超越 fine-tuned 模型（所有 p < 0.001）。

#### ❌ 代价：语义对齐略低
| 指标 | DPR-BAG | LED-PubMed (FT) |
|------|--------|------------------|
| **BERTScore** | 0.617 | 0.634 |
| **ROUGE-L** | 0.183 | 0.264 |
| **UMLS Recall** | 0.266 | 0.364 |

> ⚠️ 原因：这些指标偏好提取式输出，而 DPR-BAG 是强抽象模型，因此得分偏低属预期现象。

#### 🔁 泛化性测试（PubMedSum）
- 在 PubMedSum 上趋势一致：
  - DPR-BAG 保持更高的 **AlignScore** 和 **novelty**。
  - **SI+CoT** 配置表现更优，尤其在 AlignScore 和 Compression 上显著优于 BC。
  - MiniCheck 优势未转移，归因于该数据集本身抽象度更高，影响 NLI 类指标稳定性。

---

### 🔬 消融实验结果

#### A. 分割策略比较（RQ2）
| 策略 | AlignScore | Trigram Novelty |
|------|------------|------------------|
| **First Sentence Labeling (FS)** | **0.762** | 0.605 |
| Naive Splitting (NS) | 0.756 | 0.613 |
| Section Header (SH) | 0.636 | 0.746 |

> ✅ **发现**：FS 显著优于 SH（p < 0.001），说明仅靠章节标题不足以提供足够语境；而 NS 虽简单但接近 FS，表明结构划分的有效性。

#### B. 提示策略与实体引导（RQ3）
| 配置 | AlignScore ↓ | BERTScore ↑ | Trigram Novelty ↑ |
|------|-------------|--------------|--------------------|
| **BC (Basic Concise)** | **0.762** | 0.617 | 0.605 |
| DI (Detailed Instruction) | 0.642 | 0.638 | 0.725 |
| SI (Structural Instruction) | 0.630 | 0.636 | 0.736 |
| **SI + CoT** | 0.596 | **0.631** | **0.749** |

> ⚠️ **反直觉发现**：增加提示复杂度（DI/SI）反而 **降低事实一致性**（AlignScore 下降 >13%），可能引发“分心效应”（distraction effect）。
>
> 💡 **解释**：模型难以同时满足格式要求和事实准确性。

#### C. 实体引导效果
- **TR-UMLS**（注入 UMLS 实体）：
  - 未能恢复 DI 的事实一致性下降。
  - 甚至轻微恶化（DS-FOCUS 上升，表示名词焦点偏离）。
- **CoT**（两阶段推理）：
  - 提升 BERTScore 和抽象性。
  - 但进一步牺牲事实一致性（AlignScore 降至 0.596）。

> ❌ **结论**：在当前设置下，显式实体引导并未带来收益，反而可能加剧分心。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **结构感知 + 分治策略有效**  
   DPR-BAG 证明了无需训练即可生成高质量、事实一致且高度抽象的生物医学摘要。
   
2. **抽象性与事实性可以兼得**  
   通过“分块生成 + 全局精炼”机制，在提升抽象性的同时维持甚至超过 fine-tuned 模型的事实一致性。

3. **提示越复杂不一定越好**  
   **Counterintuitive Finding**：详细指令（DI/SI）和实体引导（TR-UMLS）会损害事实对齐，揭示了小规模 LLM 中“分心效应”的存在。

4. **无需微调也能超越监督模型**  
   在多个维度上，zero-shot DPR-BAG 超过了在相同数据上 fine-tuned 的先进模型。

### ⚠️ 局限性
1. **自动化指标依赖性强**  
   缺乏人工评估，部分指标（如 MiniCheck）在高抽象场景下可能不可靠。

2. **BOMRC 结构假设较强**  
   对非标准结构的文章（如 case report）适应性有限。

3. **过度压缩问题**  
   生成摘要平均压缩率高达 **50.037**（原始为 23.827），可能导致重要背景或次要发现丢失。

4. **实体关系混淆风险**  
   案例显示偶有基因功能错配等问题（如误将 target gene 视为 lncRNA）。

### 🔮 未来工作方向
- 引入 **publication-type-aware prompting**，根据不同文献类型调整生成策略。
- 探索 **动态 facet-level 长度控制**，缓解过度压缩问题。
- 设计更鲁棒的 **external grounding 机制**，真正提升事实一致性。
- 扩展至多语言或多模态生物医学文档生成。

---

## 🔗 开源资源
- **数据集**：[PMC-MAD on Hugging Face](https://huggingface.co/datasets/pmc-mad/PMC-MAD)
- **代码**：[GitHub - ScienceNLP-Lab/DPR-BAG](https://github.com/ScienceNLP-Lab/MultiTagger-v2/tree/main/DPR-BAG)

> “DPR-BAG 展示了无需训练即可实现高质量科学摘要生成的巨大潜力，特别是在低资源和长文本场景中。”

</details>

---

### 15. [MTR-Suite: A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks](https://arxiv.org/abs/2605.20729)

**Authors**: Junhao Ruan, Abudukeyumu Abudula, Bei Li, Yongjing Yin, Xinyu Liu, Kechen Jiao, Xin Chen, Jingang Wang, Xunliang Cai, Tong Xiao, Jingbo Zhu  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.20729v1  

#### Abstract
Accurate evaluation of conversational retrieval is pivotal for advancing Retrieval-Augmented Generation (RAG) systems. However, existing conversational retrieval benchmarks suffer from costly, sparse human annotation or rigid, unnatural automated heuristics. To address these challenges, we introduce...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：MTR-SUITE: A Framework for Evaluating and Synthesizing Conversational Retrieval Benchmarks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
传统 **conversational retrieval**（对话式检索）基准存在两大核心缺陷：
- **人工标注成本高且稀疏**：人类标注者受限于“认知边界”（cognitive boundary），只能基于局部文档生成查询，无法全局判断其他文档是否也能回答该问题，导致大量有效证据被遗漏（annotation sparsity）。
- **自动化合成方法僵化不自然**：现有自动化方法（如 CORAL）依赖静态启发式规则（如维基百科标题转问题），生成的对话缺乏真实用户的动态意图，导致语义对齐偏差。

### 提出的新方法与创新思路
作者提出 **MTR-SUITE**，一个统一框架，包含三个核心组件：

#### (1) MTR-EVAL：诊断型评估工具
- 首个细粒度评估机制，用于科学量化检索基准的质量。
- 通过 **LLM-as-a-Judge** 范式，在四个维度上审计现有基准：
  - **Query-Evidence Alignment**：检查标注文档是否真能回答查询（检测 annotation noise）
  - **Evidence Completeness**：通过判别测试（discriminability testing）识别被遗漏的相关文档（检测 annotation sparsity）
  - **Answer-Evidence Faithfulness**：验证答案是否忠实于文档
  - **Answer Quality**：评估回复的语言质量
- 实验表明其评分与人类判断高度相关（Pearson > 0.74），可作为可靠代理指标。

#### (2) MTR-PIPELINE：多智能体合成系统
- 全自动构建高质量多轮检索数据集的流程。
- 创新性地采用 **greedy traversal clustering** 算法进行语义聚类，确保：
  - 固定簇大小以适配 LLM 上下文窗口
  - 每个文档仅访问一次，避免冗余
  - 构建平滑的“语义路径”，模拟用户浏览轨迹
- 引入三智能体架构：
  - **Questioner**（用户模拟器）：生成基于上下文的问题
  - **Responder**（RAG 模拟器）：严格依据黄金文档生成回答
  - **Polisher**（润色器）：引入指代消解（coreference）、省略（ellipsis）等自然语言现象
- 成本仅为人工标注的 **1/400**（约 \$0.005/对话）

#### (3) MTR-BENCH：严苛通用领域基准
- 基于 MTR-PIPELINE 合成的大规模、通用领域对话检索基准。
- 显著提升挑战性，体现在：
  - **硬话题切换**（Hard Topic Switching）：模拟用户突然改变主题，造成上下文干扰
  - **生产级响应特征**：长文本回复（平均 87 token）、模糊决策风格（主动猜测而非澄清）
  - **工业级知识规模**：使用 2025 年初的 Wikipedia 数据，测试模型对最新知识的检索能力
- 具备更高的 **discriminative power**，能更好区分先进检索器的能力差异。

### 相比现有方法的优势
| 维度 | 传统方法 | MTR-SUITE |
|------|--------|---------|
| 标注成本 | 高（\$1.5–2.0/对话） | 极低（\$0.005/对话） |
| 数据真实性 | 局部视角导致稀疏性 | 全局感知减少假阴性 |
| 对话自然性 | 生硬、结构化 | 包含省略、指代等真实特征 |
| 可扩展性 | 难以扩展至私有领域 | 支持任意知识库（已验证金融领域） |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **MTR-BENCH**：本文提出的基准，基于 Wikipedia 2025-01 构建，包含 1,000 个对话、8,000 轮交互。
- **对比基准**：
  - QReCC、QuAC、Doc2Dial、TopiOCQA、INSCIT（人工或半自动标注）
  - CORAL（全自动生成）

### 实验设置
- **输入构造**：将对话历史序列化为 `"User: ... Agent: ..."` 形式，拼接当前查询作为检索输入。
- **检索模型**：
  - **SOTA 单轮检索器**：`bge-large-en-v1.5`, `gte-modernbert-base`, `stella_en_400m_v5`
  - **SOTA 对话密集检索器**（CDR）：`Dragon-ChatQA`, `Dragon-DocChat`
- **评估指标**：
  - **Recall@k**（R@5, R@20）
  - **MRR@20**, **NDCG@20**
  - **MTR-EVAL 四项指标**：Query-Evidence, Evidence Completeness, Answer Faithful, Answer Quality

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table 3）
在 **MTR-BENCH** 上，主流检索器表现显著低于其他基准：

| Model | R@5 | R@20 |
|-------|-----|------|
| `gte-modernbert-base` | 50.29 | 59.31 |
| `Dragon-ChatQA` | 43.84 | 50.96 |
| `bge-large-en-v1.5` | 30.16 | 35.04 |

> ⚠️ 注意：这些分数远低于其在 QReCC、QuAC 等数据集上的表现（普遍 >80 R@20），说明 MTR-BENCH 更具挑战性。

### 与其他基准的对比结果
- **性能差距大**：在其他基准上平均 R@20 比在 MTR-BENCH 上高出 **43.54 分**，说明后者尚未饱和，仍有改进空间。
- **增量收益小**：从 R@5 到 R@20 的提升仅 **8.68 分**（其他基准为 15.06 分），表明简单增加候选数量效果有限，需更强的语义理解能力。
- **判别力强**：图 2(b) 显示 MTR-BENCH 中不同模型间性能差距更大，具备更高 discriminative power。

### 消融实验结果
#### （1）移除质量过滤器的影响（Table 4）
| 设置 | Answer Quality | R@5 |
|------|----------------|-----|
| 完整流水线 | 4.91 | 0.37 |
| 移除过滤器 | 4.90 | 0.45 |

👉 结果显示：虽然召回率上升，但说明过滤器成功剔除了简单易检的内容，维持了基准难度。

#### （2）移除 Polisher 的影响
- 人类识别机器生成问题的准确率从 **62% 上升到 79%**
- 表明 **Polisher 显著提升了对话的自然性和隐蔽性**

#### （3）查询重写实验（Figure 7）
- 将原始对话查询重写为独立完整形式后，R@5 提升 **20%-40%**
- 证明性能下降主因是 **语言复杂性**（如省略、指代），而非知识不可达

#### （4）扩展 Recall@k 实验（Table 5）
即使在 k=1000 时：
- 最佳模型 `gte-Qwen2-7B` 也仅达到 **82.2% R@1000**
- `bge-large-en-v1.5` 饱和于 **47.0%**
👉 表明部分黄金文档完全不在检索模型的表示空间中，构成硬上限。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **现有基准普遍存在标注噪声与稀疏性**：即使是人工标注数据也受“认知边界”限制，导致有效检索被判为假阴性。
2. ✅ **MTR-EVAL 是有效的诊断工具**：其评分与人类高度一致，可用于自动审计基准质量（已帮助 CORAL 团队改进数据集）。
3. ✅ **MTR-BENCH 真正反映现实挑战**：其难度源于真实的语言复杂性（话题跳转、省略、长历史），而非知识冷门或模型偏见。
4. ✅ **自动化合成可行且高效**：MTR-PIPELINE 可在极低成本下生成高质量、多样化、可定制的基准数据，适用于企业私有知识库。

### 方法的局限性
- **聚焦检索而非端到端生成**：未评估最终生成质量（E2E），仅关注 retrieval 模块。
- **依赖 LLM 质量**：若生成模型本身有偏见或幻觉，可能影响合成数据质量（尽管通过指令控制缓解）。
- **领域依赖性**：虽验证了金融领域可行性，但在极端专业领域（如医学、法律）仍需进一步调优。

### 未来工作方向
- 扩展至多语言、跨模态检索场景
- 开发更高效的 **on-demand benchmarking** 流程，支持随知识库更新实时生成新测试集
- 探索将 MTR-EVAL 指标集成进训练过程，实现“自我净化”的数据增强
- 构建面向特定行业的垂直领域 MTR-Bench（如 MTR-FINANCE）

---

> 🔗 **代码与数据开源地址**：[https://github.com/rangehow/mtr-suite](https://github.com/rangehow/mtr-suite)  
> 📦 作者承诺将持续维护并发布基于 Qwen3.5 的 12 轮增强版 MTR-BENCH，推荐用于后续研究。

</details>

---

### 16. [PulseCol: Periodically Refreshed Column-Sparse Attention for Accelerating Diffusion Language Models](https://arxiv.org/abs/2605.20813)

**Authors**: Yanyi Lyu, Letian Chen, Futing Sun, Miao Zhang, Weili Guan, Liqiang Nie  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2605.20813v1  

#### Abstract
Inference in diffusion large language models (dLLMs) is computationally expensive, as full self-attention must be repeatedly executed at each step of the denoising process without KV cache. Recent sparse attention methods for dLLMs mitigate this cost via block-sparse computation, which is applied on...

---

### 17. [From SGD to Muon: Adaptive Optimization via Schatten-p Norms](https://arxiv.org/abs/2605.19781)

**Authors**: Thomas Massena (IRIT, DTIPG - SNCF, UT3), Corentin Friedrich (IRIT), Mathieu Serrurier (IRIT)  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.19781v1  

#### Abstract
Modern optimizers, like Muon, impose matrix-wise geometry constraints on their updates. These matrix-wise constraints can be unified under Linear Minimization Oracle (LMO) theory. However, all current methods impose fixed LMO geometries for the update rules, chosen by-design or empirically, which ar...

---

### 18. [ZEBRA: Zero-shot Budgeted Resource Allocation for LLM Orchestration](https://arxiv.org/abs/2605.20485)

**Authors**: May Hamri, Inbal Talgam-Cohen  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.20485v1  

#### Abstract
As autonomous agents increasingly execute end-to-end tasks under fixed monetary budgets, the pressing open question shifts from whether the budget is respected, to how to spend it effectively. Existing budget-aware methods typically control reasoning step-by-step within a single agent, or learn reso...

---

### 19. [DeCoR: Design and Control Co-Optimization for Urban Streets Using Reinforcement Learning](https://arxiv.org/abs/2605.21311)

**Authors**: Bibek Poudel, Lei Zhu, Kevin Heaslip, Sai Swaminathan, Weizi Li  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2605.21311v1  

#### Abstract
Modern vision systems can detect, track, and forecast urban actors at scale, yet translating perception outputs to urban design remains limited. We introduce DeCoR, a two-stage reinforcement learning framework that leverages flow observations to co-optimize crosswalk layout and network-level signal ...

---

### 20. [GraphRAG on Consumer Hardware: Benchmarking Local LLMs for Healthcare EHR Schema Retrieval](https://arxiv.org/abs/2605.20815)

**Authors**: Peter Fernandes, Ria Kanjilal  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.20815v1  

#### Abstract
Graph-based Retrieval Augmented Generation (GraphRAG) extends retrieval-augmented generation to support structured reasoning over complex corpora, but its reliability under resource-constrained, privacy-sensitive deployments remains unclear. In healthcare, where Electronic Health Record (EHR) data i...

---

### 21. [NanoCP: Request-Level Dynamic Context Parallelism for Data-Expert Parallel Decoding](https://arxiv.org/abs/2605.21100)

**Authors**: Jiefei Chen, Binbin Lin, Jinming Ma, Jiangfei Duan, Haojie Duanmu, Hao Liu, Qinxiu Cheng, Xiuhong Li, Zhilin Pei, Hui Wang, Xingcheng Zhang, Dahua Lin  
**Category**: cs.DC  
**Published**: 2026-05-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.21100v1  

#### Abstract
Modern serving systems for Mixture-of-Experts (MoE) models adopt hybrid data-expert parallelism: expert parallelism (EP) shards experts across GPUs to scale capacity, while data parallelism (DP) replicates attention layers across instances to process independent requests. Existing systems bind each ...

---

### 22. [Cloud-Native Operation of Roadside Infrastructure Enabling Demand-Driven Collective Perception via V2X](https://arxiv.org/abs/2605.21145)

**Authors**: Lukas Zanger, Fabian Thomsen, Guido Linden, Jean-Pierre Busch, Lennart Reiher, Lutz Eckstein  
**Category**: cs.DC  
**Published**: 2026-05-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.21145v1  

#### Abstract
Intelligent roadside infrastructure is a key enabler for cooperative intelligent transport systems (C-ITS), supporting vehicles equipped with automated driving systems (ADS), e.g., through enhanced environment perception. With a growing number and an expanding functional scope of roadside units, sca...

---

### 23. [The General Theory of Localization Methods](https://arxiv.org/abs/2605.20635)

**Authors**: Congwei Song  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.20635v1  

#### Abstract
This paper proposes a general machine learning framework called the localization method, which is fundamentally built on two core concepts: localization kernels and local means -- key components that underpin the self-attention mechanism. To establish a rigorous theoretical foundation, the framework...

---

### 24. [Fast and Stable Triangular Inversion for Delta-Rule Linear Transformers](https://arxiv.org/abs/2605.21325)

**Authors**: Aleksandros Sobczyk, Gioele Gottardo, Christos K. Matzoros, Mirko De Vita, Filip Skogh, Anastasios Zouzias, Jiawei Zhuang  
**Category**: cs.LG  
**Published**: 2026-05-21  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2605.21325v1  

#### Abstract
Linear attention has emerged as a cornerstone for efficient long-context architectures, as evidenced by its integration into state-of-the-art open-source models including Qwen3.5/3.6, Kimi Linear, and RWKV-7. Models that incorporate linear attention layers with the so-called Delta-Rule involve the i...

---

### 25. [Memory-Augmented Reinforcement Learning Agent for CAD Generation](https://arxiv.org/abs/2605.19748)

**Authors**: Yin Xiaolong, Liu Yu, Shen Jiahang, Lu Xingyu, Ni Jingzhe, Fan Fengxiao, Sang Fan  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19748v1  

#### Abstract
Automatic generation of computer-aided design (CAD) models is a core technology for enabling intelligence in advanced manufacturing. Existing generation methods based on large language models (LLMs) often fall short when handling complex CAD models characterized by long operation sequences, diverse ...

---

### 26. [CogScale: Scalable Benchmark for Sequence Processing](https://arxiv.org/abs/2605.19758)

**Authors**: Yannis Bendi-Ouis (Mnemosyne), Romain de Coudenhove (ENS-PSL), Xavier Hinaut (Mnemosyne)  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19758v1  

#### Abstract
The ability to maintain and manipulate information over time is a fundamental aspect of living beings and Artificial Intelligence. While modern models have achieved remarkable success in tasks like natural language processing, evaluating the capacity of novel architectures to process sequential info...

---

### 27. [OpenComputer: Verifiable Software Worlds for Computer-Use Agents](https://arxiv.org/abs/2605.19769)

**Authors**: Jinbiao Wei, Qianran Ma, Yilun Zhao, Xiao Zhou, Kangqi Ni, Guo Gan, Arman Cohan  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19769v1  

#### Abstract
We present OpenComputer, a verifier-grounded framework for constructing verifiable software worlds for computer-use agents. OpenComputer integrates four components: (1) app-specific state verifiers that expose structured inspection endpoints over real applications, (2) a self-evolving verification l...

---

### 28. [Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization](https://arxiv.org/abs/2605.19782)

**Authors**: Dmitry Redko (Applied AI Institute), Albert Fazlyev (AI Talent Hub, ITMO University), Konstantin Sozykin (Applied AI Institute), Maria Ivanova (YSDA, Applied AI Institute), Evgeny Burnaev (Applied AI Institute), Egor Shvetsov (Applied AI Institute)  
**Category**: cs.AI  
**Published**: 2026-05-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.19782v1  

#### Abstract
LLM discovery and optimization systems are increasingly applied across domains, implementing a common propose-evaluate-revise loop. Such optimization or discovery progresses via context conditioning on received feedback from an environment. However, as modern LLM agents are increasingly complex in t...

---

### 29. [Cross-lingual robustness of LLM-brain alignment and its computational roots](https://arxiv.org/abs/2605.21049)

**Authors**: Ni Yang, Rui He, Philipp Homan, Iris Sommer, Davide Staub, Wolfram Hinzen  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.21049v1  

#### Abstract
Large language models (LLMs) reliably predict neural activity during language comprehension and transformer depth has been interpreted as mirroring hierarchical cortical organization. However, it remains unclear whether such alignment extends to subcortical regions, overlaps spatially across languag...

---

### 30. [TextReg: Mitigating Prompt Distributional Overfitting via Regularized Text-Space Optimization](https://arxiv.org/abs/2605.21318)

**Authors**: Lucheng Fu, Ye Yu, Yiyang Wang, Yiqiao Jin, Haibo Jin, B. Aditya Prakash, Haohan Wang  
**Category**: cs.CL  
**Published**: 2026-05-21  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2605.21318v1  

#### Abstract
Large language models (LLMs) are highly sensitive to the prompts used to specify task objectives and behavioral constraints. Many recent prompt optimization methods iteratively rewrite prompts using LLM-generated feedback, but the resulting prompts often become longer, accumulate narrow sample-speci...

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
