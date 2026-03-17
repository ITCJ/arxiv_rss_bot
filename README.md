# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-17 06:47:25 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [MONET: Modeling and Optimization of neural NEtwork Training from Edge to Data Centers](https://arxiv.org/abs/2603.15002)

**Authors**: J\'er\'emy Morlier, Robin Geens, Stef Cuyckens, Arne Symons, Marian Verhelst, Vincent Gripon, Mathieu L\'eonardon  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2603.15002v1  

#### Abstract
While hardware-software co-design has significantly improved the efficiency of neural network inference, modeling the training phase remains a critical yet underexplored challenge. Training workloads impose distinct constraints, particularly regarding memory footprint and backpropagation complexity,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《MONET: Modeling and Optimization of neural NEtwork Training from Edge to Data Centers》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 DNN 加速器建模框架（如 Timeloop、Accelergy）主要聚焦于 **inference** 阶段的性能建模，而 **training** 阶段由于其更高的内存开销（如激活值存储、优化器状态）、复杂的反向传播依赖以及更丰富的调度策略（如 activation checkpointing），在硬件-软件协同设计中缺乏有效的建模工具。这导致为 inference 优化的硬件架构在 training 场景下可能表现不佳。

本文指出：**inference 和 training 在能效-延迟权衡上存在显著差异**（见图1），因此需要专门针对 training 进行建模与优化。

---

### **提出了什么新方法或新思路**
作者提出了 **MONET** —— 一个用于建模和优化神经网络在异构 dataflow 加速器上训练过程的框架。它是对已有推理建模工具 **Stream** 的扩展，首次实现了对完整训练流程（前向 + 反向 + 优化器更新）的端到端建模。

#### **核心创新点包括：**
- ✅ **支持完整的 training 工作流建模**  
  将 PyTorch 模型导出为 ONNX，并通过自定义的 ONNX 转换 pass 生成包含 forward、backward 和 optimizer 更新的完整计算图，使 Stream 能够模拟整个训练迭代。

- ✅ **引入细粒度的 layer-fusion 优化求解器**  
  基于约束优化（constraint programming）的方法自动搜索最优的 layer-fusion 配置，考虑内存容量、tiling 兼容性和算子类型限制，提升融合效率。

- ✅ **提出基于遗传算法的 activation checkpointing 优化方案**  
  发现传统 MILP 模型无法准确捕捉 fused-layer 架构下的非线性重计算代价，因此采用 **NSGA-II 遗传算法** 来联合优化 energy、latency 和 memory，获得 Pareto 最优解集。

- ✅ **模块化、可复用的设计**  
  所有转换均以 ONNX pass 形式实现，可集成进其他基于 ONNX 的工作流，具备良好的通用性。

---

### **相比现有方法的优势**

| 方法 | 是否支持 Training | 是否支持 Layer Fusion | 是否支持 Activation Checkpointing | 是否支持 Heterogeneous DA |
|------|------------------|------------------------|------------------------------------|----------------------------|
| Timeloop + Accelergy | ❌ | ❌ | ❌ | ⚠️（有限） |
| Dace-AD | ✅（部分） | ❌ | ✅ | ✅ |
| NVArchSim | ✅ | ❌ | ✅ | ❌（仅 GPU） |
| **MONET (Ours)** | ✅ | ✅（fine-grained） | ✅（genetic algo） | ✅ |

> ✅ MONET 是首个同时支持 **training-aware 建模、layer fusion 和 activation checkpointing** 在 **异构 dataflow 加速器** 上进行系统级探索的框架。

---

## **2. 核心实验方法和设置**

### **使用的模型与任务**
- **ResNet-18**：图像分类任务，输入尺寸 `(3, 32, 32)`（CIFAR-10 类似）
- **Small GPT-2**：自然语言处理任务，标准 Transformer 结构，固定序列长度与因果注意力掩码

### **硬件平台配置**
- **Edge TPU**（用于 ResNet-18）  
  基于 [19] 的 HDA 架构，包含 4×4 PE 数组，每个 PE 含 SIMD 单元和 weight-stationary 数据流核心。
  - 搜索空间见 Table II（如 PE 数量、SIMD 大小、本地内存等）

- **FuseMax**（用于 GPT-2）  
  一种输出驻留（output-stationary）的 attention 专用加速器，具有 MAC 阵列和向量单元。
  - 搜索空间见 Table III（如 PEs 规模、buffer 带宽、片外带宽等）

### **评估指标**
- **Latency**（cycles）
- **Energy Consumption**（pJ）
- **Peak Memory Usage**（GB）
- **Pareto Front 分布分析**（energy vs. latency, memory vs. latency）

### **基线方法对比**
- **Inference-only 分析结果**（作为 baseline 对照）
- **Manual layer-fusion configuration**（来自原始 Stream 的人工设计）
- **No activation checkpointing**（保存所有激活值）
- **Linear MILP 模型预测结果**（用于验证非线性现象）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) ResNet-18 on Edge TPU**
- 图 8 显示：
  - **Training 与 inference 的能效-延迟分布完全不同**：某些对 inference 有利的配置在 training 下表现差。
  - 更大的 PE 并不总是更好：对于 training latency，大 PE 不在 Pareto 前沿；而对于 inference，大 PE 更优。
  - 能耗方面，大 PE 在高资源预算下才高效，但整体不在最优前沿。

- 图 10：Layer-fusion 优化效果
  - 相比手动 fusion，MONET 的约束求解器在 **Limit6** 配置下将 latency 降低约 **15%**，energy 降低约 **12%**。
  - 自动化方法避免了繁琐的手动调参，适用于多种 workload/hardware 组合。

- 图 11 & 12：Activation Checkpointing 效果（batch size=1, 224×224）
  - 使用遗传算法可在仅增加 **4% latency 和 energy** 的前提下，节省高达 **13MB 内存**。
  - 存在某些配置可同时降低 memory 和 latency，但 energy 可能上升。

#### **(2) Small GPT-2 on FuseMax**
- 图 9：Energy-Latency Trade-off
  - 性能分布更集中，反映 GPT-2 结构的高度同质性。
  - 缓冲区带宽（buffer bandwidth）是关键瓶颈：更高带宽显著改善 energy 和 latency。
  - 训练能耗约为推理的 **4–5 倍**，凸显 training 的资源密集特性。

---

### **与基线方法的对比结果**
| 方面 | MONET vs. Baseline |
|------|--------------------|
| **Hardware Design Insight** | 揭示 training/inference 设计目标不一致，不能直接迁移 inference 优化结果 |
| **Layer Fusion** | 自动生成优于 manual fusion 的配置，减少人工干预 |
| **Activation Checkpointing** | 遗传算法比线性 MILP 更准确建模非线性效应，找到更优 trade-off |

---

### **消融实验结果**
- **Layer-fusion subgraph 长度限制的影响**（图10）：
  - `Limit=6` 达到最佳性能，过长反而无益（可能因调度复杂度上升）。
- **Activation Checkpointing 非线性验证**（图11）：
  - 重计算两个 activation 的总 cost ≠ 各自单独重计算之和 → 证明线性假设失效。
  - 原因：recomputation 改变了 layer fusion 的可能性（如 Op2 和 Op3 可被融合）。

---

## **4. 关键结论和发现**

### **主要发现**
1. 🔍 **Training-aware 建模至关重要**：  
   推理阶段得出的“最优”硬件配置在训练场景下往往次优甚至劣化，必须独立建模 training workload。

2. 🧩 **Layer fusion 在 training 中更具潜力但也更复杂**：  
   optimizer 中的 element-wise 操作适合与梯度计算融合，减少中间激活存储；但需系统化搜索策略。

3. 🔁 **Activation checkpointing 具有非线性行为**：  
   传统的线性 MILP 模型不足以描述 fused-layer 架构中的重计算代价，必须采用非线性优化方法（如 GA）。

4. ⚖️ **存在新的 Pareto 最优操作点**：  
   通过联合优化 hardware、mapping、fusion 和 checkpointing，可以发现兼顾 energy、latency 和 memory 的新设计空间。

---

### **方法的局限性**
- 当前框架仍基于 **analytical modeling**，未完全模拟 cache miss、pipeline stall 等微架构细节。
- 遗传算法虽有效，但收敛速度依赖参数设置，在超大规模模型上可能面临可扩展性挑战。
- 目前主要验证在 Edge TPU 和 FuseMax 上，尚未扩展至主流 GPU 架构（如 H100）。

---

### **未来工作方向**
- ✅ 将 MONET 扩展至 **GPU 架构建模**（类似 LLMCompass 对 inference 的支持）
- ✅ 探索 **专用 hardware design** 以支持 activation checkpointing 和 fused training ops
- ✅ 开发更高效的 **multi-objective mapping algorithms**，用于大规模 LLM training
- ✅ 引入 **learned cost models** 替代解析式估计，提高精度

---

> **总结一句话**：  
> MONET 填补了 **training-aware DNN 加速器建模** 的空白，揭示了 training 与 inference 的根本差异，并提供了自动化工具链来联合优化硬件架构、layer fusion 与 activation checkpointing，推动高效深度学习系统的 co-design。

</details>

---

### 2. [SVD Contextual Sparsity Predictors for Fast LLM Inference](https://arxiv.org/abs/2603.14110)

**Authors**: Georgii Serbin, Kirill Koshkin, Zhongao Sun, Anastasiya Bistrigova, C. C. Korikov  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.14110v1  

#### Abstract
Contextual sparsity is one of the approaches used to reduce computational complexity in the inference process of large language models (LLMs). Existing techniques for efficient LLM inference acceleration based on contextual sparsity with minimal accuracy degradation require training sparse pattern p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# SVD Contextual Sparsity Predictors for Fast LLM Inference 论文总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

大型语言模型（LLMs）在推理阶段计算开销巨大，尤其是在边缘设备上部署时面临**高延迟、高功耗和内存带宽瓶颈**的问题。尽管 ReGLU 架构中存在**动态稀疏性**（即每层 FFN 中仅有少量神经元被激活），但如何高效地预测这些稀疏模式以实现加速，同时保持精度，是一个挑战。

现有基于学习的稀疏性预测器（如 Deja Vu、PowerInfer）需要额外训练，且对硬件依赖性强，难以快速部署。

### **提出了什么新方法或新思路**

本文提出了一种**无需训练**（training-free）的稀疏模式预测框架——**SVD-based Sparsity Predictors (SVDP)**，用于加速 ReGLU-based LLM 的推理过程。其核心思想是：

- 利用**截断感知的奇异值分解**（truncation-aware SVD）对 FFN 中的 `W_gate` 投影矩阵进行低秩近似，构建轻量级稀疏性预测器。
- 引入**校准偏置项**（calibrated bias）来补偿因低秩近似导致的分布偏移，提升预测准确性。
- 在推理时采用**顺序执行管道**（sequential pipeline），先计算 gate 投影，再验证并剔除误激活神经元，进一步提高实际稀疏度。

该方法分为两个阶段：
1. **离线阶段**：通过 SVD 分解 + 数据白化 + 贪心偏置校准，构建每层的稀疏预测器。
2. **在线阶段**：在每个 FFN 前插入预测器，生成稀疏掩码，并由定制的 CUDA/CANN 执行器完成稀疏计算。

### **相比现有方法的优势**

| 方面 | 优势 |
|------|------|
| **无需训练** | 不需要额外训练数据或梯度优化，节省时间和资源，适合快速迁移和部署。 |
| **理论保障** | 提供了预测误差的理论上界（见附录 A），保证了方法的可靠性。 |
| **高效性** | 预测器仅需两层线性变换（`ABx + b`），计算开销极小。 |
| **通用性** | 支持 CUDA 和华为 CANN 平台，在多种硬件上均可实现显著加速。 |
| **可组合性** | 与已有的稀疏模型（如 ProSparse、TurboSparse）兼容，可直接集成。 |

---

## 2. 核心实验方法和设置

### **使用的模型**

在三个开源的稀疏化 7B 级 LLM 上进行评估：
- **ProSparse-LLaMA2-7B**（使用 ReLU 替换激活函数）
- **TurboSparse-Mistral-Instruct**
- **SparseQwen2-7B**

这些模型本身具有较高的激活稀疏性（平均约 90%），为上下文稀疏性提供了基础。

### **数据集与评估任务**

使用多个权威基准测试集，涵盖多类复杂任务：
- **数学推理**：GSM8K
- **代码生成**：HumanEval, MBPP
- **科学常识推理**：ARC-E / ARC-C
- **开放域问答**：TriviaQA
- **综合推理**：BBH
- **中文理解**：CMMLU

所有任务均采用**自由生成**（free-form generation）方式进行评估，而非 perplexity，更贴近真实应用场景。

### **实验设置**

- **硬件平台**：
  - GPU：NVIDIA RTX 3090（MSU 实验室）
  - NPU：Ascend 310P3（华为）
- **软件框架**：基于 vLLM 实现推理，不启用 tensor parallelism。
- **精度设置**：推理使用 FP16，离线构建使用 FP64。
- **评估指标**：
  - **End-to-End 推理延迟**（E2E Latency）：包含 prefill 和 200 token 生成时间。
  - **准确率/通过率**（Accuracy/pass@1）：各任务得分。
  - **加速比**（Speedup）：相对于稠密推理的速度提升倍数。

### **基线方法对比**

| 方法 | 类型 | 是否需训练 | 特点 |
|------|------|------------|------|
| **Dense baseline** | 稠密推理 | 否 | 性能上限参考 |
| **Deja Vu** | 学习型预测器 | 是 | Lookahead + 异步预测 |
| **PowerInfer** | 学习型预测器 | 是 | CPU/GPU 混合加速为主 |
| **GRIFFIN** | 无训练方法 | 否 | 基于 prompt 统计的静态稀疏 |
| **SVDP (Ours)** | 无训练方法 | 否 | 本文提出的方法 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 模型 | 方法 | 平均准确率下降 | 最大 E2E 加速比 |
|------|------|----------------|----------------|
| ProSparse-LLaMA2-7B | SVDP | < 0.15% | **1.71×** |
| TurboSparse-Mistral-Instruct | SVDP | < 1.12% | **1.84×** |
| SparseQwen2-7B | SVDP | < 1.35% | **1.86×** |

> ✅ **总体结论**：在平均激活稀疏度达 90% 的条件下，**最高实现 1.8× 的端到端推理加速**，同时**准确率损失控制在 1% 以内**。

### **与基线方法的对比结果**

#### 表格对比（以 ProSparse-LLaMA2-7B 为例）

| 方法 | 预测稀疏度 | 平均得分 | E2E 加速比 |
|------|-----------|----------|------------|
| Dense | — | 38.92 | 1.00× |
| Deja Vu (r=1024) | 20% | 32.86 | 1.38× |
| PowerInfer (r=1024) | 80% | 37.93 | 1.63× |
| GRIFFIN (50%) | 50% | 32.81 | 1.46× |
| **SVDP (seq., r=256)** | **50%** | **38.78** | **1.64×** |

- **SVDP 在更低的预测稀疏度下实现了更高的准确率和更快的加速**。
- 相比 PowerInfer，SVDP 准确率更高（38.78 vs 37.93），且无需训练。
- 相比 GRIFFIN，SVDP 更稳定，尤其在长序列生成中表现更好。

#### ROC-AUC 分析（图4）

- SVDP 在各层的 **ROC-AUC 显著高于 Deja Vu 和 PowerInfer**，说明其预测能力更强。
- 即使 rank 较低（如 r=256），SVDP 仍优于 r=1024 的学习型方法，表明其**利用权重结构的能力更强**。

### **消融实验结果**

#### 消融配置（表3 & 表9–11）

| 配置 | 相对于 Dense 的平均得分损失 |
|------|-------------------------------|
| Naive SVD (r=256) | > 20% 下降 |
| + Data Whitening | 下降至 ~8% |
| + Bias Calibration | 下降至 <1.5% |

- **Data Whitening** 显著提升了预测分离能力（见图5），特别是在低秩情况下。
- **Bias Calibration** 是关键组件，有效缓解了低秩近似带来的 false negative 问题。
- **Sequential Pipeline** 比 Parallel Pipeline 更优，能进一步过滤误激活神经元。

#### 预测器 Rank 影响（图7）

- 当 rank 达到中间维度 D 的约 **2%**（如 D=11008 → r≈256）时，ROC-AUC 进入平台期。
- 表明**小规模预测器即可达到良好效果**，有利于降低预测开销。

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **SVD 可作为高质量稀疏性预测的基础工具**：通过对 `W_gate` 的低秩 SVD 分解，可以捕捉到输入与激活之间的强相关性，无需训练即可获得高召回率的稀疏模式。

2. ✅ **Calibrated Bias 至关重要**：由于截断引入系统性偏差，简单的零阈值会带来大量 false negatives；而通过贪心算法校准 per-neuron bias，可在极小数据集上实现精准控制。

3. ✅ **顺序执行管道增强稀疏性**：gate/up 投影的串行执行允许运行时重新验证激活状态，从而提升最终稀疏度，弥补预测器轻量化带来的不足。

4. ✅ **跨平台高效执行成为可能**：
   - 在 CUDA 上通过自定义 kernel 实现确定性、高性能稀疏计算；
   - 在 CANN 上仅用 PyTorch 高级 API 即可获得高达 **30× 的 FFN 层加速**（表4，95% 稀疏时），证明了 DaVinci 架构对稀疏性的原生支持潜力。

5. ✅ **适用于边缘部署场景**：整个方法轻量、无需训练、内存友好，特别适合在资源受限的边缘设备上部署 LLM。

### **局限性**

1. ❌ **依赖 ReLU 类激活函数**：本方法依赖于 ReLU 的硬截断特性来安全跳过非正激活。对于 SiLU/Swish 等平滑激活函数（如原始 Qwen2），直接应用会导致严重精度下降（见表5，平均下降超 3.5%）。

2. ❌ **批处理稀疏性受限**：batched inference 会降低整体稀疏度，限制加速潜力；且当前方法主要用于 decoding 阶段，prefill 阶段无法有效利用稀疏性。

3. ❌ **对非常深层模型泛化待验证**：目前实验集中在 7B 规模模型，更大模型（如 70B）上的扩展性和稳定性尚需研究。

### **未来工作方向**

- 探索将 SVDP 扩展至非 ReLU 激活函数的方法，例如结合 soft masking 或 activation transformation。
- 设计适用于 prefill 阶段的稀疏机制，进一步压缩总延迟。
- 结合量化、KV Cache 压缩等技术，打造一体化的边缘 LLM 加速方案。
- 探索在 MoE 架构中的应用，预测 expert 激活路径。

--- 

> **一句话总结**：  
> 本文提出了一种无需训练、基于 SVD 的上下文稀疏性预测方法 SVDP，通过理论驱动的设计和高效的执行引擎，在三类稀疏 LLM 上实现了最高 **1.8× 的端到端推理加速**，且精度损失小于 1%，为 LLM 在边缘设备上的高效部署提供了新路径。

</details>

---

### 3. [LightningRL: Breaking the Accuracy-Parallelism Trade-off of Block-wise dLLMs via Reinforcement Learning](https://arxiv.org/abs/2603.13319)

**Authors**: Yanzhe Hu, Yijie Jin, Pengfei Liu, Kai Yu, Zhijie Deng  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.13319v1  

#### Abstract
Diffusion Large Language Models (dLLMs) have emerged as a promising paradigm for parallel token generation, with block-wise variants garnering significant research interest. Despite their potential, existing dLLMs typically suffer from a rigid accuracy-parallelism trade-off: increasing the number of...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：LightningRL: Breaking the Accuracy-Parallelism Trade-off of Block-wise dLLMs via Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **block-wise dLLMs**（diffusion Large Language Models）在并行解码时面临显著的 **accuracy-parallelism trade-off**（准确率-并行度权衡）：
- 为了提升推理速度（如提高 Tokens Per Forward, TPF），模型往往采用激进的并行解码策略；
- 但这会导致生成质量下降，尤其是在数学推理和代码生成等复杂任务上。

### 🚀 提出的新方法：LightningRL
作者提出一种基于 **Reinforcement Learning (RL)** 的后训练框架 —— **LightningRL**，旨在直接优化预训练 dLLM 的 **speed-quality frontier**（速度-质量前沿）。

其核心思想是：
> 不要求模型在所有采样路径上都进行高并行解码，而是引导模型找到那些“既高度可并行、又能输出正确结果”的特定轨迹。

为此，LightningRL 在 **Group Relative Policy Optimization (GRPO)** 框架基础上进行了三项关键改进：

#### 创新点 1：Per-reward Decoupled Normalization（奖励解耦归一化）
- 多目标 RL 中，accuracy reward（离散）和 TPF reward（连续）量级差异大，导致优化信号被主导。
- LightningRL 对每个 reward 分量独立归一化后再聚合，避免某一目标“淹没”其他目标，提升训练稳定性。

#### 创新点 2：Token-level NLL Regularization（词元级负对数似然正则）
- 引入在正确轨迹上的 token-level Negative Log-Likelihood (NLL) 损失，作为“自模仿锚点”（self-imitation anchor）。
- 防止策略漂移（policy drift），增强语言连贯性和准确性，缓解稀疏奖励下的 reward hacking。

#### 创新点 3：Dynamic Sampling with TPF-aware Filtering（TPF感知动态采样）
- 在训练中过滤掉组内 TPF 差异小或全为错误样本的 prompt。
- 确保每批次都有足够多样化的并行性信号和至少一个成功样本，提升梯度密度与学习效率。

### 🔍 相比现有方法的优势
| 方面 | LightningRL | 传统方法（如 Fast-dLLM, d3LLM, EAGLE-3） |
|------|-------------|----------------------------------------|
| 优化方式 | 显式联合优化 accuracy 和 parallelism | 多为训练后采样策略或蒸馏，未联合建模 |
| 训练稳定性 | 高（通过解耦+锚定机制） | 容易出现 reward collapse 或 accuracy 下降 |
| 性能表现 | 显著打破 accuracy-parallelism 权衡 | 提速常以牺牲 accuracy 为代价 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **数学推理任务**：
  - `GSM8K`（小学数学应用题）
  - `MATH500`（高中以上难度数学题）
- **代码生成任务**：
  - `MBPP`（Python 编程任务）
  - `HumanEval`（函数级代码生成）

### ⚙️ 实验设置
- **基础模型**：基于 `SDAR-8B-b32`（block size=32 的 8B 参数 block-wise dLLM）
- **训练配置**：
  - Batch size: 128 tasks × 32 rollouts per task
  - 采样温度：1.0（训练），greedy decoding（测试）
  - 使用低置信度 remasking（confidence threshold=0.9）
- **RL 设置**：
  - 使用 GRPO 框架扩展
  - 奖励设计：Accuracy（±1） + Speed（TPF）
  - 优化器：AdamW，学习率 1e-5（policy），5e-6（value，若存在）

### 🎯 评估指标
| 指标 | 含义 |
|------|------|
| **Acc (%)** | 任务准确率（answer matching） |
| **TPF (Tokens Per Forward)** | 平均每次前向传播生成的 token 数，衡量并行程度 |
| **AUP (Accuracy Under Parallelism)** | 综合指标，定义为 Acc × TPF 的积分或加权平均，反映整体 speed-quality 表现 |
| **TPS (Tokens Per Second)** | 实际推理吞吐量（硬件实测） |

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **dLLM 加速方法** | Fast-dLLM-v2, dParallel-LLaDA, d3LLM |
| **AR 模型加速方法** | EAGLE-3 (speculative decoding 类) |
| **其他 RL 方法** | TraceRL, GRPO(traj) |
| **原始模型** | SDAR-8B-b32（未微调） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 2）

| Model | GSM8K Acc (%) | TPF | AUP |
|-------|----------------|-----|-----|
| SDAR-8B-b32 | 88.9 | 2.85 | 252.5 |
| EAGLE-3 | 76.6 | 5.12 | 319.0 |
| d3LLM-LLaDA | 73.1 | 9.11 | 637.7 |
| **LightningRL-8B-b32** | **90.3** | **5.58** | **492.4** |

| Model | MBPP Acc (%) | TPF | AUP |
|-------|---------------|-----|-----|
| SDAR-8B-b32 | 58.0 | 2.44 | 81.1 |
| d3LLM-LLaDA | 40.6 | 4.21 | 88.4 |
| **LightningRL-8B-b32** | **58.3** | **11.10** | **641.6** |

> ✅ LightningRL 在保持甚至略微提升 accuracy 的同时，将 TPF 大幅提升至 **平均 7.32**，最高达 **11.10（MBPP）**，AUP 达到 **497.9**，远超所有基线。

### 🏁 与基线方法对比结果
- **相比 SDAR 基线**：
  - TPF 提升近 **2 倍**（2.85 → 5.58）
  - AUP 提升超过 **90%**
- **相比 EAGLE-3**（当前主流 AR 加速器）：
  - 准确率更高（90.3 vs 76.6）
  - 更优的 AUP（492.4 vs 319.0）
- **相比 d3LLM**（最强 dLLM 基线）：
  - 虽然 d3LLM 有更高 TPF（9.11），但 accuracy 下降严重（73.1%），而 LightningRL 实现了 **高 accuracy + 高 parallelism** 的平衡。

### 🔬 消融实验结果（Ablation Study）

#### （1）组件消融（Table 3）

| 消融变体 | Acc (%) | TPF | AUP |
|----------|---------|-----|-----|
| w/o NLL loss | 80.7 | 5.03 | 385.7 |
| w/o Decoupled Norm | 85.3 | 4.96 | 416.5 |
| w/o TPF-aware Filtering | 87.2 | 5.27 | 454.5 |
| **Full LightningRL** | **90.3** | **5.58** | **492.4** |

> 所有三个组件均不可或缺，尤其是 **token-level NLL loss** 对维持 accuracy 至关重要。

#### （2）损失归约策略消融（Table 4）

| 归约策略（Policy-KL-NLL） | Acc (%) | TPF | AUP |
|----------------------------|---------|-----|-----|
| Seq-Seq-Seq | 87.5 | 5.42 | 467.1 |
| Seq-Tok-Seq | 88.7 | 4.86 | 424.7 |
| **Seq-Tok-Tok** | **90.3** | **5.58** | **492.4** |
| Tok-Tok-Tok | 80.0 | 3.91 | 306.5 |

> 使用 **sequence-level policy loss + token-level KL/NLL loss**（即 Seq-Tok-Tok）效果最佳，说明 rollout 平等加权更稳定。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **成功打破 accuracy-parallelism trade-off**：
   - LightningRL 是首个在 block-wise dLLMs 上实现 **accuracy 与 parallelism 协同提升** 的 RL 框架。
   - 在多个 benchmark 上显著优于现有 dLLM 和 AR 加速方法。

2. **多目标 RL 设计至关重要**：
   - 单纯堆叠 TPF 或 accuracy reward 会导致训练崩溃；
   - **decoupled normalization + NLL anchoring** 构成了稳定的双目标优化基石。

3. **动态采样提升训练效率**：
   - TPF-aware filtering 有效缓解了 reward quantization 导致的梯度饥饿问题。

4. **无需价值网络也能稳定训练**：
   - 实验表明引入 value model 反而导致性能下降（见 Appendix A），推测因 block-wise 解码状态跳跃剧烈，critic 难以建模稳定 baseline。

### ⚠️ 局限性
- 当前仅验证于数学与代码任务，是否泛化到开放生成（如对话、创作）尚待研究；
- 模型规模目前限于 8B，更大模型（如 70B）的扩展性需进一步探索；
- 依赖 verifier-based reward（如答案匹配、unit test），难以应用于无明确标准答案的任务。

### 🔮 未来工作方向
- 探索 **larger context length** 下的 scaling laws；
- 将 LightningRL 框架推广至更多类型的 dLLM 架构；
- 结合 curriculum learning 进一步优化 high-TPF 路径的学习效率；
- 研究适用于非验证类任务的隐式 reward 建模方法。

---

> 🔗 项目代码已开源：[https://github.com/SJTU-DENG-Lab/LightningRL](https://github.com/SJTU-DENG-Lab/LightningRL)

</details>

---

### 4. [ExPosST: Explicit Positioning with Adaptive Masking for LLM-Based Simultaneous Machine Translation](https://arxiv.org/abs/2603.14903)

**Authors**: Yuzhe Shang, Pengzhi Gao, Yazheng Yang, Jiayao Ma, Wei Liu, Jian Luan, Jingsong Su  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.14903v1  

#### Abstract
Large language models (LLMs) have recently demonstrated promising performance in simultaneous machine translation (SimulMT). However, applying decoder-only LLMs to SimulMT introduces a positional mismatch, which leads to a dilemma between decoding efficiency and positional consistency. Existing appr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ExPosST: Explicit Positioning with Adaptive Masking for LLM-Based Simultaneous Machine Translation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在基于 **decoder-only LLM** 的 **Simultaneous Machine Translation (SimulMT)** 中，存在一个核心挑战：**Positional Mismatch（位置错位）**。

- 在 SimulMT 场景中，源语言句子是流式输入的，新的 `source token` 被动态插入上下文中。
- 这会导致已生成的目标 token 的 **position index 发生偏移**，从而破坏了 Key-Value (KV) Cache 中存储的位置信息。
- 结果是：要么需要 **重新计算 KV Cache**（牺牲效率），要么保留缓存但引入 **位置不一致**（损害准确性），形成 **效率 vs. 一致性** 的两难困境。

### 提出了什么新方法或新思路
作者提出 **ExPosST**（Explicit Position Allocation for Simultaneous Translation），一种通用且高效的框架，通过以下两个核心机制解决上述问题：

#### （1）Pre-allocated Positions Inference（预分配位置推理）
- 显式地为潜在的 incoming source tokens **预留固定长度的位置槽（position slot）**。
- 所有目标 token 的生成都从这些预留槽之后开始，确保其 **position index 在整个 READ/WRITE 周期中保持不变**。
- 因此，KV Cache 可以被直接复用而无需重新计算，实现 **zero-recomputation inference**。

#### （2）Policy-Consistent Fine-tuning（策略一致的微调）
- 在训练阶段模拟推理时的 slot 结构：
  - 将源句按 slot 容量进行分段；
  - 使用与实际 SimulMT 策略（如 wait-k 或 read-n）一致的 **attention mask**，限制模型只能看到当前应可见的源词。
- 确保训练与推理之间的行为完全对齐，避免分布偏移。

### 相比现有方法的优势
| 方法 | 局限性 | ExPosST 的优势 |
|------|--------|----------------|
| **SimulMask** | 依赖 ALiBi positional encoding，无法兼容主流 RoPE-based LLM（如 Llama/Qwen） | 兼容 RoPE 和 ALiBi，适配性强 |
| **GPE (Group Position Encoding)** | 引入非整数或重叠位置，偏离标准 LLM 预训练范式 | 使用标准整数位置编码，无需修改模型架构 |
| **Conversational SimulMT** | 依赖 user/assistant 角色切换，导致“prompt bloating”和注意力分散 | 无频繁角色切换，减少冗余标记，提升效率 |

✅ **综合优势**：  
ExPosST 实现了 **高推理效率（KV Cache 复用） + 严格位置一致性 + 广泛模型兼容性** 的统一，解决了现有方法难以兼顾三者的问题。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **IWSLT 2017** 数据集，涵盖五个语言对：
  - English → French (En-Fr)
  - English → German (En-De)
  - English → Dutch (En-Nl)
  - English → Italian (En-It)
  - English → Romanian (En-Ro)

### 实验设置
- **主干模型**：
  - `Llama-3.1-8B-Instruct`
  - `Qwen2.5-7B-Instruct`
  - `falcon-rw-1b`（用于与 SimulMask 对比）
- **微调方式**：采用 **LoRA (Low-Rank Adaptation)** 进行参数高效微调。
- **推理框架**：集成于 `Simul-LLM` 框架，并使用 `SimulEval` 工具包进行评估。

### 评估指标
- **质量指标**：
  - **BLEU**（使用 SacreBLEU 计算去分词 BLEU）
  - **COMET**（补充语义层面的质量评估）
- **延迟指标**：
  - **LAAL (Length-Adaptive Average Lagging)**：衡量翻译延迟的标准指标

### 基线方法对比
| 基线方法 | 描述 |
|---------|------|
| **GPE** | 分组位置编码，独立管理源/目标位置空间 |
| **Conversational SimulMT** | 使用对话模板，交替添加 source/target 片段 |
| **Offline** | 全句翻译模型，作为性能上限参考 |
| **SimulMask** | 改进 attention mask 与 ALiBi 编码配合的方法 |

此外，ExPosST 分别在两种策略下测试：
- **ExPosST(wait-k)**：与 GPE 对齐
- **ExPosST(read-n)**：与 Conversational SimulMT 对齐

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 在 Llama-3.1-8B-Instruct 上的表现（图4）
- **vs. GPE (wait-k)**：
  - 在相同 LAAL 下，**BLEU 提升达 2–3 分**；
  - 性能随 k 增大稳步逼近离线模型（offline）水平。
- **vs. Conversational SimulMT (read-n)**：
  - 全延迟范围内 **平均提升超过 1 BLEU 分**；
  - 表现出更优的 **quality-latency Pareto frontier**。

#### ✅ 在 Qwen2.5-7B-Instruct 上的结果（图5）
- 在 En-Fr 和 En-De 上均取得 **优于或持平基线** 的表现；
- 验证了 ExPosST 在不同 RoPE-based 架构上的 **强泛化能力**。

#### ✅ 在 falcon-rw-1b（ALiBi-based）上的表现（图6）
- 与专为 ALiBi 设计的 **SimulMask 性能相当**；
- 表明 ExPosST **可无缝迁移至不同类型 positional encoding 的模型**。

#### ✅ COMET 指标验证（附录图9）
- 在所有语言对上，ExPosST 同样显著优于各基线；
- 即使在低延迟（low LAAL）场景下也能保持较高的语义保真度。

### 消融实验结果（Ablation Study，图7）
在 En-De 任务上对两个核心组件进行消融：

| 变体 | 描述 | 性能影响 |
|------|------|----------|
| **w/o Masking** | 移除策略一致的 attention mask | 明显下降，说明模型未能学习增量生成行为 |
| **w/o Slot** | 移除预分配 slot 机制 | 性能大幅退化，证实位置错位严重干扰推理 |
| **完整 ExPosST** | 包含 slot + masking | 达到最优性能，二者缺一不可 |

👉 结论：**Pre-allocated slot 与 policy-consistent masking 相辅相成，共同保障高效且准确的 SimulMT 推理**。

---

## 4. 关键结论和发现

### 主要发现
1. **显式位置分配是解决 KV Cache 错位的关键**：
   - 通过预分配 position slot，彻底消除因流式输入引起的位置偏移问题。
2. **训练-推理一致性至关重要**：
   - Policy-Consistent Fine-tuning 成功弥合了传统 SFT 与 SimulMT 流式推理间的鸿沟。
3. **ExPosST 具备高度通用性**：
   - 兼容 RoPE 和 ALiBi 等多种 positional encoding；
   - 适用于 wait-k、read-n 等多种 decoding policy；
   - 在多个主流 LLM 上均表现优异。

### 方法的局限性
- **超参数敏感性**：预分配 slot 长度 $ L_{\text{slot}} $ 的选择会影响性能（见图3）。过小导致频繁切换，过大则增加 padding 开销。
- **训练-推理 slot 不匹配的风险**：若推理时使用的 slot 长度远小于训练设定值（如从16降到4），会显著降低 BLEU（表1）。
- **仍需针对特定 policy 微调模型**：目前未实现单一模型支持多 policy 切换。

### 未来工作方向
- 探索 **自适应 slot 长度调整机制**，根据输入动态优化 $ L_{\text{slot}} $。
- 研究 **multi-policy unified training**，使单个模型能灵活应对不同 wait-k / read-n 设置。
- 将 ExPosST 扩展至 **语音 SimulST（Simultaneous Speech Translation）** 场景。

---

> 📌 **总结一句话**：  
> **ExPosST 通过“显式位置预留 + 策略对齐微调”，首次实现了在不牺牲效率、不修改模型结构的前提下，将任意 RoPE/ALiBi-based LLM 高效部署于 SimulMT 任务中，推动了 LLM 在实时翻译中的实用化进程。**

</details>

---

### 5. [Distributed Acoustic Sensing for Urban Traffic Monitoring: Spatio-Temporal Attention in Recurrent Neural Networks](https://arxiv.org/abs/2603.13903)

**Authors**: Izhan Fakhruzi, Manuel Titos, Carmen Ben\'itez, Luz Garc\'ia  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.13903v1  

#### Abstract
Effective urban traffic monitoring is essential for improving mobility, enhancing safety, and supporting sustainable cities. Distributed Acoustic Sensing (DAS) enables large-scale traffic observation by transforming existing fiber-optic infrastructure into dense arrays of vibration sensors. However,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Distributed Acoustic Sensing for Urban Traffic Monitoring: Spatio-Temporal Attention in Recurrent Neural Networks

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究针对**城市交通监控中分布式声学传感（DAS）数据的高分辨率时空建模难题**。尽管 DAS 利用现有光纤基础设施实现了大范围、低成本的振动感知，但其信号受环境噪声、耦合条件异质性和事件重叠影响，导致在真实动态场景下进行连续交通事件识别（如车辆、公交车、噪音）仍极具挑战。

此外，传统方法多基于孤立事件或静态特征分类，难以有效捕捉连续交通流中的时序依赖关系。

---

### 提出的新方法与新思路
作者提出了一种结合 **spatio-temporal attention 机制的 Recurrent Neural Network（RNN）架构**，用于建模 DAS 数据中的复杂时空模式。具体创新包括：

- **系统性集成空间注意力（Spatial Attention, SA）与时间注意力（Temporal Attention, TA）模块于 bi-LSTM 架构中**，以分别聚焦关键空间位置（SPs）和重要时间片段。
- 设计并比较多种 attention 配置方式（前置/后置、单层/级联），探索最优结构。
- 提出 **SA-bi-TA 架构**：即先应用 SA 模块选择关键空间点，再通过 bi-LSTM 编码双向时序信息，最后由 TA 模块强调关键时间段，形成层次化关注机制。
- 引入 **attention heatmaps 可视化分析**，增强模型决策过程的可解释性，并辅助发现标注不一致问题。

---

### 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **性能与效率平衡** | 在保持高准确率的同时显著降低参数量（如 bi-TA 模型相比添加导数特征的方法减少 715% 参数增长）。 |
| **可解释性提升** | attention 权重图能直观反映模型对特定车道（如公交专用道）和事件持续期的关注，符合物理直觉。 |
| **跨站点泛化能力** | 所提 SA-bi-TA 模型展现出良好的 spatial transferability，在未见地点（Acera del Darro）实现高达 79.27% 的识别准确率，仅比训练地（Palacio de Congresos）下降约 8–18%，远优于需逐点标注的传统方案。 |
| **实用性增强** | 充分利用已有光纤网络，无需额外部署传感器，支持可持续智慧城市发展。 |

---

## 2. 核心实验方法和设置

### 数据集
- **来源**：西班牙格拉纳达市的真实城市光纤部署（Granada, Spain）
- **设备**：Aragón Photonics™ 的 HDAS 系统，基于 CP-OTDR 技术
- **监测路段**：
  - **Palacio de Congresos**：2车道单向通行，采集10小时数据
  - **Acera del Darro**：4车道双向通行，采集1小时数据
- **采样参数**：空间分辨率 6m/SP，通道间距 3m，采样频率 250Hz
- **标签方式**：视频同步记录 + 专家手动标注，定义三类事件：
  - `Noise`（行人或无事件）
  - `Car`
  - `Bus`（含卡车等大型车辆）

> 总共生成约 72,000 个窗口样本（表 I）

---

### 特征工程与预处理
1. 对原始应变率信号（△ε）进行去噪、去趋势、带通滤波（0.1–30 Hz）
2. 分段为 2秒滑动窗（步长0.5秒），加汉明窗
3. 提取每窗 **36维手工特征**（能量、熵、统计量等）
4. 增强特征维度：
   - 添加一阶与二阶时间导数 → 得到 108维特征向量（记作 +△）
   - 融合相邻两个 SP 的特征 → 最终达 324维（考虑空间上下文）

---

### 实验设置与评估指标
- **任务类型**：连续交通事件分类（非孤立事件检测）
- **输入单位**：1.5分钟长的时间序列片段
- **模型架构对比组**：
  - 基线：LSTM、bi-LSTM
  - 单 attention：SA-bi, bi-SA, TA-bi, bi-TA
  - 级联 attention：SA-bi-TA, TA-bi-SA 等六种组合
- **超参数优化**：使用 Optuna 进行贝叶斯搜索（层数、隐藏单元、dropout、学习率等）
- **训练策略**：5折交叉验证，早停 + L2 正则防止过拟合
- **评估指标**：
  - Accuracy（Acc）
  - F1-score（F1）
  - 可训练参数数量（#Param）
  - 相对于基线的相对改进（RI-Acc, RPI）

---

### 基线方法对比
| 模型 | 是否使用 Attention | 是否含导数特征 |
|------|---------------------|----------------|
| LSTM / bi-LSTM | ❌ | ✅/❌ |
| SA-bi / TA-bi | ✅（前接） | ✅/❌ |
| bi-SA / bi-TA | ✅（后接） | ✅/❌ |
| SA-bi-TA（主推） | ✅（前后皆有） | ✅/❌ |

> 所有模型均基于 handcrafted features 输入，非端到端 CNN 架构

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总（来自 Table II & VIII）

| 模型 | Acc (%) | F1 (%) | #Param (M) | RI-Acc (%) | RPI (%) |
|------|--------|-------|-----------|------------|---------|
| **LSTM** | 86.62 | 86.59 | 0.08 | — | — |
| **bi-LSTM (baseline)** | 88.08 | 88.10 | 0.41 | 0.00 | 0.00 |
| **bi-TA** | 88.47 | 88.49 | 0.25 | **+0.45** | **-38.85** |
| **SA-bi+△** | 89.05 | 89.05 | 0.84 | +1.10 | +103.50 |
| **SA-bi-TA** | **88.62** | **88.65** | **0.68** | **+0.62** | **+65.48** |
| **SA-bi-TA+△** | 89.05 | 89.03 | 1.13 | +1.10 | +174.15 |

> 注：负 RPI 表示参数更少；SA-bi-TA 在精度与复杂度间取得最佳平衡

---

### 与基线方法对比结果
- **bi-LSTM > LSTM**：双向结构显著提升性能（↑1.68% Acc），验证历史与未来上下文的重要性。
- **temporal derivatives (+△)** 可提升性能（LSTM+A 达 88.17%），但带来巨大参数开销（RPI ↑997%），性价比低。
- **bi-TA 模型** 在不含导数情况下达到 88.47% 准确率，且参数量低于基线（RPI = -38.85%），表明 **TA 比显式导数更高效地捕获短期动态**。
- **SA-bi-TA 模型** 实现 88.62% 准确率，优于多数单一 attention 或导数增强模型，同时参数控制良好。

---

### 消融实验结果
#### （1）Attention 放置位置的影响
| 配置 | 效果 |
|------|------|
| **bi-TA（后置 TA）** | 显著提升性能且参数最少，说明 TA 在高层语义上强化已有时序建模效果更优 |
| **TA-bi（前置 TA）** | 性能略降，注意力分布碎片化，缺乏连续性，因缺少上下文引导 |
| **SA-bi vs bi-SA** | SA 放在前面更能发挥空间筛选作用，配合后续 bi-LSTM 更有效 |

#### （2）级联 attention 是否更好？
- 多数级联配置（如 bi-SA-TA, TA-SA-bi）未带来明显增益，反而增加参数负担。
- **SA-bi-TA 是唯一表现优异的级联结构**，体现“先空间筛选 → 再双向编码 → 后时间聚焦”的合理流程。
- 结论：**并非越深越好，合理的模块顺序比堆叠更重要**

#### （3）是否需要 temporal derivatives？
- 加入 +△ 可小幅提升最高性能（至 89.05%），但代价高昂（参数翻倍以上）
- **SA-bi+△ 和 SA-bi-TA+△ 均可达 89.05%**，但后者参数更少（1.13M vs 0.84M），更具优势

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **bi-LSTM 是建模 DAS 交通事件的理想基线**：能稳定捕捉短时快速变化的车辆穿越信号。
2. ✅ **Temporal Attention（TA）比 temporal derivatives 更高效**：可在更低参数成本下实现相似甚至更好的性能，有助于缓解过拟合。
3. ✅ **Attention placement 至关重要**：
   - 后置 TA（bi-TA）能有效增强已有的时序表示；
   - 前置 SA（SA-bi）有助于提前过滤无关空间信息。
4. ✅ **SA-bi-TA 架构实现最佳权衡**：兼顾准确性、参数效率与可解释性，是推荐配置。
5. ✅ **Attention heatmaps 具备强可解释性**：
   - 成功定位公交车常行驶的车道（SP2）
   - 揭示了人类标注错误（模型预测 Bus，实际标为 Car），证明可用于数据质量审计
6. ✅ **具备良好 spatial transferability**：
   - 在 Acera del Darro 不同 SP 组上测试，最高达 **79.27% 准确率**
   - 尽管性能下降（较原地 87.65% 下降 ~8–18%），但仍具实用价值
   - UMAP 分析显示两地数据分布具有一定结构性相似性，支撑迁移可行性

---

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖手工特征** | 未采用 end-to-end CNN 提取特征，可能遗漏深层非线性模式 |
| **小规模标注数据** | 总共仅约 7.2万样本，限制深度模型潜力，易受噪声干扰 |
| **跨站点性能衰减明显** | 最差组（Group A）准确率仅 69.93%，提示需进一步适应机制 |
| **未解决极端环境变异** | 土壤属性、埋深、电磁干扰等因素仍可能导致模型失效 |
| **实时性未充分验证** | 推理延迟、部署能耗等工程问题尚未讨论 |

---

### 未来工作方向
1. **开发自监督/半监督学习框架**：减少对大规模人工标注的依赖，利用海量无标签 DAS 数据预训练。
2. **引入 domain adaptation 或 meta-learning**：提升模型在新地点的快速适应能力，缩小跨站点性能差距。
3. **探索 end-to-end 架构**：结合 CNN 自动提取局部特征，与 RNN+attention 联合优化。
4. **构建 DAS-based 城市级交通数字孪生系统**：整合多源数据（GPS、摄像头、DAS），实现全域实时感知。
5. **推动 foundation model for DAS**：借鉴大模型思想，训练通用振动理解模型，支持多任务迁移（交通、安防、地震等）。

---

> **总体评价**：该论文将 **spatio-temporal attention 机制系统引入 DAS 城市交通监控领域**，不仅提升了识别性能与模型效率，更重要的是增强了系统的**可解释性与可迁移性**，为构建**可扩展、智能化的城市感知基础设施**提供了坚实的技术路径。

</details>

---

### 6. [A Multi-Scale Graph Learning Framework with Temporal Consistency Constraints for Financial Fraud Detection in Transaction Networks under Non-Stationary Conditions](https://arxiv.org/abs/2603.14592)

**Authors**: Yiming Lei, Qiannan Shen, Junhao Song  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.14592v1  

#### Abstract
Financial fraud detection in transaction networks involves modeling sparse anomalies, dynamic patterns, and severe class imbalance in the presence of temporal drift in the data. In real-world transaction systems, a suspicious transaction is rarely isolated: rather, legitimate and suspicious transact...

---

### 7. [Federated Learning of Binary Neural Networks: Enabling Low-Cost Inference](https://arxiv.org/abs/2603.15507)

**Authors**: Nitin Priyadarshini Shankar, Soham Lahiri, Sheetal Kalyani, Saurav Prakash  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.15507v1  

#### Abstract
Federated Learning (FL) preserves privacy by distributing training across devices. However, using DNNs is computationally intensive at the low-powered edge during inference. Edge deployment demands models that simultaneously optimize memory footprint and computational efficiency, a dilemma where con...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Federated Learning of Binary Neural Networks: Enabling Low-Cost Inference

## 1. 论文的主要贡献和创新点

### 解决的问题
- **边缘设备推理成本高**：传统 Federated Learning (FL) 虽然保护了训练阶段的数据隐私，但训练出的 Deep Neural Networks (DNNs) 在资源受限的边缘设备上进行推理时，计算和内存开销巨大。
- **后训练量化精度损失严重**：现有的 post-training binarization 方法虽然能压缩模型大小，但由于量化误差导致严重的 accuracy drop。
- **联邦场景下的二值化挑战**：如何在保持模型轻量化的同时，在非独立同分布 (non-IID) 数据下实现高效且准确的联邦学习。

### 提出的新方法：FedBNN
本文提出 **FedBNN**，一种面向低开销推理的旋转感知二值神经网络框架，其核心思想是：
- **直接学习二值表示**：在本地训练过程中直接学习 Binary Neural Network (BNN)，而非先训练实值模型再量化。
- **旋转对齐机制 (Rotation-aware)**：借鉴 Lin et al. (2020) 的 rotated BNN 思想，引入可学习的旋转矩阵 $R_1$ 和 $R_2$，最小化实值权重与其二值化版本之间的角度偏差 (angular bias)，从而减少量化误差。
- **联邦融合策略 (Federated-aware Fusion)**：
  - 客户端使用插值权重 $w = \lambda w_{\text{local}} + (1-\lambda) w_{\text{server}}$ 进行旋转优化，结合本地与全局信息。
  - 引入可学习参数 $\alpha$ 和 $\beta$，自适应调整旋转方向和服务器权重的影响，增强对异构数据的鲁棒性。
- **双路径聚合策略**：
  - 服务器维护两个模型：标准的实值聚合模型（用于下发）和旋转对齐后的二值友好模型（仅用于评估和选模）。
  - 通过在旋转空间内聚合，提升全局模型的 sign consistency，选出更适合二值化的模型。

### 相比现有方法的优势
| 方法 | 通信效率 | 推理效率 | 二值化鲁棒性 | 优势说明 |
|------|----------|----------|--------------|----------|
| **FedAvg** | ❌ | ❌ | ❌ | 全精度模型，推理开销大；后量化性能骤降 |
| **FedBAT** | ✅ | ❌ | ❌ | 仅对**更新量**进行二值化通信，最终模型仍是实值，推理无收益 |
| **FedMUD / HGC** | ✅ | ❌ | ❌ | 侧重通信压缩，不解决推理复杂度 |
| **FedBNN (本文)** | ⚠️ (略增) | ✅✅✅ | ✅✅✅ | **唯一同时优化通信、推理效率，并在训练中集成二值化的方法** |

> **注**：FedBNN 因需传输旋转矩阵 $R_1, R_2$，通信开销略高于 FedAvg，但增加不足 1%，且随模型增大而降低（见附录 A.4），可忽略不计。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在多个主流 FL 基准数据集上进行，覆盖不同难度和规模：
- **FMNIST** (Fashion-MNIST)
- **SVHN** (Street View House Numbers)
- **CIFAR-10**
- **Tiny-ImageNet**
- **FEMNIST** (Federated Extended MNIST)

### 实验设置
- **客户端数量**：$N_c = 100$
- **每轮采样客户端数**：10
- **本地训练**：每轮 15 个 epoch（FMNIST/SVHN 为 5/10）
- **批量大小**：64
- **总通信轮数**：1500（FMNIST/SVHN 为 500）
- **优化器**：SGD，初始学习率 0.1（CNN4/ResNet）或 0.01（ConvNeXt-Tiny），按轮次衰减。

### 评估指标
- **Accuracy**：在测试集上的分类准确率（clean accuracy）。
- **Binarized Accuracy**：将最终模型的权重和激活全部二值化后的准确率，衡量模型对二值化的鲁棒性。
- **FLOPs**：推理阶段的浮点运算次数，衡量计算复杂度。
- **Memory**：模型存储所需内存（MB），衡量存储开销。

### 基线方法对比
- **FedAvg**：标准联邦平均算法。
- **FedBAT**：可学习的二值化更新通信方法。
- **FedMUD**：基于分解的低秩联邦学习方法。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Dataset (Model) | Method     | IID Acc (%) | Non-IID 1 (%) | Non-IID 2 (%) | FLOPs       | Memory (MB) | Binarized Acc (IID) |
|-----------------|------------|-------------|---------------|---------------|-------------|-------------|---------------------|
| FMNIST (CNN4)   | **FedBNN** | **88.46**   | **87.76**     | **83.38**     | **3.48e5**  | **0.0489**  | **88.46**           |
|                 | FedAvg     | 92.24       | 91.44         | 89.28         | 2.02e7      | 1.56        | 53.42               |
| SVHN (CNN4)     | **FedBNN** | **86.89**   | **85.63**     | **84.40**     | **5.19e5**  | **0.0498**  | **86.89**           |
|                 | FedAvg     | 92.10       | 90.60         | 89.34         | 3.00e7      | 1.60        | 28.01               |
| CIFAR10 (ResNet10)| **FedBNN** | **89.95**   | **82.84**     | **73.82**     | **1.11e7**  | **0.613**   | **89.95**           |
|                 | FedAvg     | 90.86       | 86.28         | 70.62         | 4.40e8      | 19.62       | 17.20               |
| CIFAR10 (ConvNeXt-Tiny)| **FedBNN** | **72.08**   | **67.08**     | **63.00**     | **6.07e7**  | **3.49**    | **72.08**           |
|                 | FedAvg     | 65.22       | 60.20         | 61.84         | 2.98e9      | 111.64      | 18.04               |

### 与基线方法的对比结果
- **推理效率**：
  - FedBNN 实现 **~58× FLOPs reduction** 和 **32× memory reduction**，显著优于所有基线（FedBAT/FedMUD 不降低推理开销）。
- **二值化鲁棒性**：
  - FedBNN 的 **clean accuracy ≈ binarized accuracy**，表明其训练即为二值化友好的。
  - FedAvg 后量化后性能暴跌（如 CIFAR10 上从 90.86% → 17.20%），而 FedBNN 保持不变。
- **准确性**：
  - 在多数情况下，FedBNN 准确率接近 FedAvg（差距 < 10%），且在某些非 IID 场景（如 CIFAR10 + ConvNeXt-Tiny）甚至**反超 FedAvg**。
  - 显著优于 FedBAT 和 FedMUD，尤其在 Non-IID 设置下。

### 消融实验结果
#### (1) 服务器对齐的重要性（Table 2）
移除服务器对齐项 ($\beta=1, \lambda=1$) 导致性能下降，尤其在异构数据下：
- CIFAR10 (ResNet10, Non-IID 2): 73.82% → 68.54% (-5.28%)
- Tiny-ImageNet (ResNet18, IID): 46.54% → 42.86% (-3.68%)

> **结论**：服务器对齐对维持全局一致性、提升模型稳定性至关重要。

#### (2) 聚合策略比较（Table 3）
- **先聚合旋转权重再构建辅助更新**（本文方法）性能更优。
- 尤其在复杂数据集（如 CIFAR10 ResNet10, Non-IID 2）上，替代方案性能大幅下降（73.82% vs 55.62%）。

#### (3) 与同等复杂度实值模型对比（Table 5）
将实值模型（如 ResNet）缩小至与 FedBNN 相当的 FLOPs/Memory：
- FedBNN 在所有设置下均**显著优于**这些“瘦身”实值模型。
- 例如 Tiny-ImageNet (Non-IID 2): FedBNN 45.74% vs FLOPs-matched ResNet 35.66%。

> **结论**：直接训练 BNN 比缩小实值模型更有效。

---

## 4. 关键结论和发现

### 主要发现
1. **训练即二值化是关键**：在 FL 中直接训练 BNN（FedBNN）相比后量化，能极大保留模型性能，实现真正的低开销部署。
2. **旋转对齐有效缓解量化误差**：通过旋转矩阵对齐实值与二值空间，显著提升了 BNN 在联邦环境下的表现。
3. **服务器对齐增强鲁棒性**：融合服务器权重的更新机制对处理 non-IID 数据至关重要。
4. **效率与精度的良好平衡**：FedBNN 以轻微的精度损失（通常 < 10%）换取了高达 58× 的 FLOPs 和 32× 的内存节省，极具实用价值。
5. **超越全精度模型**：在特定架构（如 ConvNeXt-Tiny）和异构数据下，FedBNN 的准确率甚至超过 FedAvg，表明二值化可能具有正则化效果。

### 方法的局限性
- **通信开销微增**：需额外传输小规模的旋转矩阵 $R_1, R_2$，尽管占比 <1%。
- **超参数敏感性**：$\lambda, \alpha, \beta$ 的学习动态影响收敛，需仔细调参。
- **理论分析有限**：缺乏对收敛性的严格证明。

### 未来工作方向
- 探索更高效的旋转矩阵表示或参数化方法，进一步降低通信开销。
- 扩展到更大规模的模型和更复杂的任务（如 NLP）。
- 设计替代的聚合策略，以更好地利用旋转对齐信息。
- 加强理论分析，提供收敛性保证。

</details>

---

### 8. [Supervised Fine-Tuning versus Reinforcement Learning: A Study of Post-Training Methods for Large Language Models](https://arxiv.org/abs/2603.13985)

**Authors**: Haitao Jiang, Wenbo Zhang, Jiarui Yao, Hengrui Cai, Sheng Wang, Rui Song  
**Category**: cs.AI  
**Published**: 2026-03-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.13985v1  

#### Abstract
Pre-trained Large Language Model (LLM) exhibits broad capabilities, yet, for specific tasks or domains their attainment of higher accuracy and more reliable reasoning generally depends on post-training through Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL). Although often treated as dis...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Supervised Fine-Tuning versus Reinforcement Learning: A Study of Post-Training Methods for Large Language Models*

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文系统地解决了当前大型语言模型（LLM）后训练领域中一个关键但被忽视的问题：**Supervised Fine-Tuning (SFT)** 和 **Reinforcement Learning (RL)** 这两种主流后训练范式之间的关系模糊、割裂，缺乏统一的理解框架。

尽管SFT和RL在实践中常被结合使用（如RLHF），但大多数研究将它们视为独立的方法进行探讨，缺少对二者内在联系、互补优势以及如何有效融合的深入分析。这导致了方法设计上的重复、低效，以及对“何时用SFT，何时用RL”这一基本问题缺乏指导。

### 提出了什么新方法或新思路
本论文并非提出单一的新算法，而是提出了一个**全面且统一的理论与实践框架**来理解、比较和整合SFT与RL。其核心新思路包括：

1.  **统一的目标函数视角**：论文从数学上论证了SFT可以被视为RL的一个特例。具体而言，SFT中最大化专家响应概率的目标，等价于在RL框架下，以一个指示函数（`1{y=y*}`）作为隐式的奖励信号。这为将两种方法置于同一优化框架下提供了坚实的理论基础。
2.  **互补性与集成框架**：基于上述统一视角，论文系统性地梳理并分类了多种利用SFT和RL互补优势的策略，例如：
    *   **Using SFT to Enhance RL**：利用SFT的离线示范数据来引导RL的在线探索，提高训练效率和稳定性（如SRFT, BREAD）。
    *   **Using RL to Enhance SFT**：借鉴RL的思想（如重要性采样、在线rollout）来改进SFT，缓解分布偏移（distribution shift）问题，提升泛化能力（如DFT, iw-SFT）。
    *   **Hybrid Training**：在同一训练流程中交替或加权结合SFT和RL目标，实现端到端的联合优化（如ReLIFT, HPT）。
3.  **趋势洞察与未来方向**：通过对2023至2025年的应用研究进行大规模分析，论文揭示了一个明确的趋势：后训练正从单一的SFT或RL，快速转向**混合的SFT-RL范式**，并且数据来源正从依赖API向由开源模型生成的数据转移。

### 相比现有方法的优势
相比以往孤立地研究SFT或RL的综述或技术论文，本文的优势在于：

*   **系统性与综合性**：首次提供了一个涵盖理论、算法、数据、应用和趋势的完整图景，填补了该领域的空白。
*   **理论深度**：通过将SFT形式化为RL的特例，为两种看似不同的方法建立了深刻的理论联系，为后续研究提供了新的思考角度。
*   **实践指导性**：提出的统一框架和分类法为研究人员选择和设计后训练策略提供了清晰的决策依据，有助于避免盲目尝试。
*   **前瞻性**：准确预测并总结了SFT-RL混合训练将成为主流范式的发展趋势，为社区指明了方向。

## 2. 核心实验方法和设置

需要特别指出的是，这篇论文是一篇**综述性研究 (survey/study)**，而非提出单一新算法的实证论文。因此，它本身不包含传统意义上的“实验”，而是对已有大量研究工作的**系统性分析、归纳和趋势验证**。

### 分析方法和数据集
论文的核心“实验”是对其所引用的大量文献进行内容分析和量化统计，其方法如下：

*   **文献范围**：聚焦于2023年至2025年发表的关于LLM后训练的研究论文。
*   **分析维度**：
    1.  **方法论分析**：对SFT和RL的算法（Algorithm-centric）和数据（Data-centric）层面的创新进行分类和总结。
    2.  **应用领域分析**：将应用研究分为四大类：**General QA Tasks**, **Mathematical Tasks**, **Agentic Tasks**, 和 **Code-based Tasks**，并分析各领域的技术特点。
    3.  **趋势量化分析**：通过关键词搜索arXiv预印本数据库，统计不同年份、不同领域、不同训练方法（SFT, RL, Both）的论文数量，以验证其提出的趋势假设。

### 评估指标
由于是综述，其“评估”是定性和定量相结合的：
*   **定性评估**：通过理论推导（如SFT作为RL特例的证明）、案例分析（如Table 1中的方法对比）来论证观点。
*   **定量评估**：使用**论文发表数量**作为衡量研究活跃度和趋势的代理指标。例如，通过统计提及特定基准数据集（如`hotpotqa`, `gsm8k`, `webarena`, `humaneval`）的论文数量，来反映各领域的研究热度。

### 基线方法对比
本文没有直接对比某个新模型的性能，而是将**SFT**、**RL** 和 **SFT+RL** 视为三种不同的“基线”训练范式，并通过分析历史数据来比较它们的相对流行度和发展轨迹。其对比结果显示，纯SFT的主导地位正在被SFT+RL的混合范式迅速取代。

## 3. 主要实验结果和性能指标

这里的“结果”指的是论文通过其分析得出的关键发现和统计数据。

### 关键性能数据（研究趋势数据）
论文通过分析arXiv数据，得出了以下关键趋势数据（见原文Figure 2和Appendix B.4）：

*   **研究总量激增**：所有领域的相关研究都在快速增长。
    *   **General QA**: 从2023年的292篇增长到2024年的652篇（+123%），预计2025年达983篇（+118%）。
    *   **Mathematical**: 从492篇增至1098篇（+123%），预计2025年达2399篇（近5倍增长）。
    *   **Agentic**: 从100篇增至174篇（+74%），预计2025年达261篇。
    *   **Code-based**: 从115篇增至428篇（+272%），预计2025年达786篇（+84%），增速最快。
*   **训练范式转变**：混合训练成为主流。
    *   **2023年**: SFT占73.3%，混合方法（Both）仅占20.0%。
    *   **2024年**: 混合方法飙升至73.8%，成为最主流方法，SFT降至19.1%。
    *   **2025年（预测）**: 混合方法占比稳定在70.6%，SFT进一步下降。
*   **数据来源转变**：从闭源API转向开源模型。
    *   **API-based models**的使用率从2023年的32.2%骤降至2025年的11.1%。
    *   **Open-weight models**的使用率从12.2%增长至25.0%。
    *   **Benchmarks**的使用率从48.9%稳步上升至61.1%。

### 与基线方法的对比结果
论文的分析表明，在性能和研究影响力上，“SFT+RL”混合范式已经超越了单一的SFT或RL：
*   **SFT vs. RL**: SFT更稳定，适合初始阶段注入知识；RL能更好地平衡性能与探索，提升泛化，但可能不稳定。
*   **SFT+RL vs. 单一方法**: 绝大多数高性能的现代LLM（如文中提到的Deepseek-R1, WizardMath等）都采用了先SFT再RL的两阶段流程，或更先进的混合训练。这表明**结合两者优势的范式取得了最高的报告性能**。

### 消融实验结果
本文未进行消融实验。但它总结了其他研究中的发现，间接支持其观点。例如，许多研究表明，跳过SFT直接进行RL通常效果不佳，而只进行SFT则可能导致模型过度拟合，泛化能力差。这从侧面印证了两种方法各有短板，需要互补。

## 4. 关键结论和发现

### 论文的主要发现
1.  **SFT与RL本质相连**：SFT可以被形式化地看作是一种特殊的RL，其中奖励信号是是否完美复现了专家演示。
2.  **互补而非对立**：SFT和RL不是相互竞争的方法，而是具有**互补优势**的工具。SFT提供稳定性和高质量的初始行为，RL促进探索和泛化。
3.  **混合范式是未来**：后训练的前沿已从单一方法演变为**SFT与RL的深度融合**。未来的高效、可扩展的LLM对齐将依赖于精心设计的混合训练管道。
4.  **开放化与标准化**：研究生态正朝着使用**开源权重模型**和**标准化基准**的方向发展，推动了研究的可复现性和透明度。

### 方法的局限性
*   **综述性质**：作为一篇综述，其结论依赖于所选文献的质量和覆盖面。作者承认，由于该领域进展极快，可能存在遗漏最新进展的风险。
*   **数据偏差**：用于趋势分析的论文筛选策略（基于数据集关键词）可能引入偏差，因为它可能低估了那些不使用主流benchmark的创新工作。
*   **理论简化**：将SFT完全视为RL的特例是一个有用的理论抽象，但在实际实现细节（如梯度计算、稳定性处理）上，两者仍有显著差异。

### 未来工作方向
论文在第7节明确指出了两个重要的未来方向：
1.  **样本与计算高效的后训练方法 (Sample- and compute-efficient methodologies)**：开发更高效的数据利用（如信息论指导的采样）和计算（如量化感知训练、部分rollout）技术，以降低SFT和RL的资源消耗。
2.  **稀疏或间接奖励信号下的SFT与RL (SFT and RL under sparse or indirect reward signals)**：探索在缺乏明确、稠密奖励信号的真实世界任务中如何进行对齐，例如利用用户的间接反馈（如点击、停留时间）或自我评估信号。

</details>

---

### 9. [PrototypeNAS: Rapid Design of Deep Neural Networks for Microcontroller Units](https://arxiv.org/abs/2603.15106)

**Authors**: Mark Deutel, Simon Geis, Axel Plinge  
**Category**: cs.AI  
**Published**: 2026-03-17  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.15106v1  

#### Abstract
Enabling efficient deep neural network (DNN) inference on edge devices with different hardware constraints is a challenging task that typically requires DNN architectures to be specialized for each device separately. To avoid the huge manual effort, one can use neural architecture search (NAS). Howe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：PrototypeNAS: Rapid Design of Deep Neural Networks for Microcontroller Units**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
在资源受限的边缘设备（如 **Microcontroller Units, MCUs**）上高效部署深度神经网络（DNN）是一个重大挑战。传统方法需要为不同硬件单独设计和训练大量模型，耗时且计算成本高。现有的 **Neural Architecture Search (NAS)** 方法虽然能自动化模型设计，但通常依赖从头训练大量候选模型，效率低下，且难以兼顾目标平台的硬件约束（如内存、算力限制）。

### **提出的新方法与创新点**
作者提出了 **PrototypeNAS**，一种**零样本（zero-shot）多目标优化（Multi-Objective Optimization, MOO）框架**，用于快速为MCU定制高效的DNN架构。其三大核心创新如下：

1. **联合搜索空间（Combined Search Space）**  
   将**架构选择**（如 MobileNetV2、ResNet）、**结构优化**（superblock depth、kernel size）、**剪枝（pruning）** 和 **量化（quantization）配置**统一在一个可优化的超参数空间中，实现端到端的联合优化，而非分阶段处理。

2. **零样本代理集成（Ensemble of Zero-Shot Proxies）作为多目标**  
   不采用单一零样本代理（如 SNIP、NASWOT），而是将多个代理（MeCo、ZiCo、NASWOT、SNIP）作为独立的优化目标，结合 **FLOPs** 构成多目标优化问题。这避免了对代理进行主观加权，减少偏差，提升鲁棒性。

3. **基于超体积子集选择（Hypervolume Subset Selection）的模型筛选**  
   在获得 Pareto 前沿后，使用进化算法驱动的 **Hypervolume subset selection** 从中选出最具代表性的 3–5 个模型，确保覆盖精度与效率之间的最优权衡，显著减少后续训练开销。

### **相比现有方法的优势**
- **极高的效率**：无需训练数百个模型，仅需训练最终选出的 3–5 个候选模型。
- **更强的泛化能力**：支持多种任务（图像分类、时间序列分类、目标检测）和多种基础架构。
- **更贴近真实部署**：优化过程中直接考虑 MCU 的 **RAM、ROM 和 FLOPs 限制**，生成的模型可直接部署。
- **避免代理偏差**：通过多代理竞争机制，缓解单一零样本代理不准确的问题。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
共评估了 **12 个数据集**，涵盖三类任务：
- **图像分类（8个）**：CIFAR10、CIFAR100、GTSRB、Flowers、Birds、Cars、Pets、ArxPhotos314
- **时间序列分类（3个）**：Daliac、MAFULDA、BitBrain Sleep
- **目标检测（1个）**：COCO 子集（person detection）

### **实验设置**
- **目标硬件平台**：ARM Cortex-M 系列 MCU（主测 iMXRT1062 Cortex-M7）
- **搜索阶段**：
  - 执行 **500 次采样** 进行多目标优化。
  - 使用 **Hypervolume subset selection** 选出 top-5 模型。
- **训练阶段**：
  - 图像任务：ImageNet 预训练 + 目标数据集微调（100 epochs）
  - 时间序列任务：随机初始化训练
  - 使用 **PyTorch Lightning** 和 **YOLOv5** 分别处理分类与检测任务
- **压缩策略**：
  - 图像与检测任务：Post-Training Quantization (PTQ)
  - 时间序列任务：Quantization-Aware Training (QAT)，因 PTQ 表现不佳

### **评估指标**
| 类别 | 指标 |
|------|------|
| **性能** | Test Accuracy (%)、mAP50（目标检测） |
| **资源消耗** | FLOPs、ROM (kB)、RAM (kB) |
| **运行时表现** | Latency (ms)、Energy (mJ) |
| **相关性分析** | Kendall’s τ（衡量零样本代理与真实精度的相关性） |

### **基线方法对比**
- **TinyNAS (MCUNet)**：专为MCU设计的两阶段NAS方法，基于MobileNet超网搜索。
- **NATS-Bench**：包含拓扑（TSS）和规模（SSS）两个搜索空间的大规模NAS基准。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **图像分类（CIFAR10 结果）**
| 方法 | 最高准确率 | 平均优势 |
|------|-----------|--------|
| **PrototypeNAS** | **93.7%** | —— |
| MCUNet (TinyNAS) | ~88.7% | ↓ ~5% |

- PrototypeNAS 在相似 FLOPs 下平均**高出约 5% 准确率**。
- 所有选出的模型均可在 iMXRT1062 上运行，**延迟最低达 224ms**，能耗低至 **77.6mJ**。

#### **时间序列分类**
- 在 Mafaulda 数据集上达到 **98.5% 准确率**，FLOPs 仅为 215.3 MFLOPs。
- 显示出良好的跨任务适应能力。

#### **目标检测（Person Detection on COCO）**
- 输入分辨率 128×128：在低计算量下实现合理 mAP50。
- 输入分辨率 320×320（Raspberry Pi 5）：验证了方法对更大设备的扩展性。
- 成功将 MbedNet 与 YOLOv5 检测头结合，证明搜索空间灵活性。

### **与基线方法对比结果**
| 对比项 | PrototypeNAS vs. MCUNet | PrototypeNAS vs. NATS-Bench |
|-------|--------------------------|----------------------------|
| **准确率** | ↑ **+5%**（CIFAR10） | 达到甚至超过其最佳模型 |
| **训练成本** | 仅需训练 **5 模型** | NATS-Bench 需训练 **48,393 模型** |
| **搜索效率** | 分钟级完成搜索 | 耗时数周训练全部候选 |
| **部署可行性** | 直接输出可部署模型 | 多数模型超出MCU容量 |

> ✅ **结论**：PrototypeNAS 在**极低训练成本下达到了与大规模NAS相当甚至更优的性能**。

### **消融实验与分析（隐含）**
尽管未明确命名“ablation”，但以下分析具有消融性质：

1. **零样本代理集成的有效性**（见 Table 4）：
   - 单个代理（如 SNIP、ZiCo）与真实精度的 Kendall’s τ 在不同数据集上波动剧烈，有时为负（误导搜索）。
   - **无单一代理始终优于 FLOPs**，但**至少有一个代理在每个数据集上表现良好**。
   - → 支持使用**代理集成 + MOO** 而非加权融合。

2. **Hypervolume Selection 的有效性**：
   - 选出的 top-5 模型覆盖从小到大、从低精到高精的完整权衡曲线。
   - 避免了人工筛选的主观性，提升了决策效率。

3. **QAT vs. PTQ**：
   - 时间序列任务中 PTQ 导致显著精度下降，而 QAT 可恢复性能。
   - → 表明**量化策略应根据任务特性动态选择**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **零样本代理虽不完美，但集成后可有效指导搜索**：通过将其作为多目标而非加权评分，能规避个体偏差，提升整体鲁棒性。
2. **联合优化架构、剪枝与量化显著提升搜索效率**：相比分步优化，统一搜索空间能发现更优的全局组合。
3. **Hypervolume subset selection 是连接搜索与训练的关键桥梁**：以极低成本提炼出最有价值的候选模型。
4. **PrototypeNAS 具备强任务通用性**：成功应用于图像、时间序列、目标检测三大类任务，展现广泛适用潜力。
5. **FLOPs 是延迟与能耗的良好代理**：实验显示二者与 FLOPs 呈线性关系，可在无硬件反馈的情况下用于优化。

### **方法的局限性**
- **依赖预定义的基础架构池**：无法生成完全新颖的网络结构（如全新注意力模块），局限于已有架构的变体。
- **零样本代理仍存在不确定性**：尽管集成缓解了问题，但在某些数据集上仍可能出现排序错误。
- **当前实现聚焦于特定MCU平台**：需针对不同硬件重新设定约束条件，自动化程度有待提升。
- **未探索动态输入或稀疏激活等高级压缩技术**。

### **未来工作方向**
- 扩展搜索空间以支持更多新型架构组件（如 Transformer blocks）。
- 引入轻量级仿真器预测不同MCU上的实际性能。
- 探索在线自适应NAS，在设备上持续优化模型。
- 将方法推广至更多模态（音频、传感器融合等）和应用场景（如 TinyML for robotics）。

---

> 📌 **总体评价**：  
> **PrototypeNAS 是一项面向实用化的高效NAS突破**。它不是追求“最强模型”，而是致力于“**最快找到足够好的可部署模型**”。对于工业界快速落地 TinyML 应用具有重要价值。

</details>

---

### 10. [SemantiCache: Efficient KV Cache Compression via Semantic Chunking and Clustered Merging](https://arxiv.org/abs/2603.14303)

**Authors**: Shunlong Wu, Hai Lin, Shaoshen Chen, Tingwei Lu, Yongqin Zeng, Shaoxiong Zhan, Hai-Tao Zheng, Hong-Gee Kim  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.14303v1  

#### Abstract
Existing KV cache compression methods generally operate on discrete tokens or non-semantic chunks. However, such approaches often lead to semantic fragmentation, where linguistically coherent units are disrupted, causing irreversible information loss and degradation in model performance. To address ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《SemantiCache: Efficient KV Cache Compression via Semantic Chunking and Clustered Merging》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现有的 **KV cache compression** 方法通常基于离散 token 或非语义 chunk 进行压缩，导致“**语义碎片化（semantic fragmentation）**”——即原本连贯的语言单元（如短语、从句、句子）被强行打断，造成不可逆的信息丢失，进而显著降低模型在长上下文任务中的表现。

### ✅ 提出的新方法与新思路
作者提出 **SemantiCache**，一种新型的 KV cache 压缩框架，其核心思想是：
- **对齐语言的语义层级结构**，通过“语义分块 + 聚类合并”的两阶段策略，在压缩过程中保持语义完整性。
- 受人类记忆长文本的认知机制启发：先按自然边界（标点等）切分为完整语义单元（chunking），再提炼每个单元的核心含义（clustering & merging）。

#### 主要技术组件：
1. **Semantic Chunking（语义分块）**  
   利用自然语言中的分隔符（如 `.` `,` `?` `\n` 等）作为语义边界，将 KV cache 分割为语义连贯的 chunks，并保留 delimiter 的 KV 状态作为结构锚点。

2. **Greedy Seed-Based Clustering (GSC)**  
   在每个 chunk 内部运行轻量级聚类算法：选取未分配 token 作为“种子”，贪婪吸收与其 key 向量相似度高于阈值 $ T $ 的后续 token，形成高内聚的 **semantic clusters**。

3. **Clustered Merging with Proportional Attention**  
   - 对每个 cluster 使用均值池化生成一个紧凑的 **semantic core**。
   - 引入 **Proportional Attention** 机制，在 attention 计算中对合并后的 core 添加 $ \log s $（$ s $ 为原 cluster 大小），使注意力权重与其原始规模成正比，缓解因合并带来的信息稀释问题。

### ✅ 相比现有方法的优势
| 维度 | 传统方法（Eviction/Merging） | SemantiCache |
|------|-------------------------------|-------------|
| 语义完整性 | ❌ 易产生语义断裂 | ✅ 保持 phrase/sentence 级结构 |
| 压缩效率 | ⚠️ 高但伴随性能下降 | ✅ 高效且性能损失极小 |
| 信息保留 | ❌ 永久删除或粗粒度合并 | ✅ 结构锚定 + 比例加权恢复影响力 |
| 推理加速 | 一般 | ✅ 最高达 **2.61× TPOT 加速** |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **LongBench [25]**：多任务、双语长上下文评测基准，涵盖以下任务类型：
  - 单文档问答（Single-document QA）
  - 多文档问答（Multi-document QA）
  - 摘要生成（Summarization）
  - 少样本学习（Few-shot Learning）
  - 合成任务（Synthetic Tasks）
  - 代码补全（Code Completion）
- **Needle-in-a-Haystack (NIAH) [26]**：专门测试模型在超长上下文中检索特定信息的能力，用于评估语义保真度。

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|--------|------|
| **模型** | Llama-3-8B-Instruct、Mistral-7B-Instruct-v0.2 |
| **硬件平台** | NVIDIA A100 80GB GPU |
| **上下文长度** | 最大支持至 32k tokens |
| **KV Cache Budget** | 设置为原始大小的 20%, 35%, 50% 等 |
| **Delimiter Set D** | `[ ".", ",", "?", "!", ";", ":", "", "\t", "\n" ]` |
| **相似度阈值 T** | 在 [0.5, 0.9] 范围内调参 |

#### 评估指标：
- **Accuracy / Average Score**：LongBench 上的平均得分；NIAH 中的命中准确率。
- **Efficiency Metrics**：
  - **TTFT (Time To First Token)**：prefill 阶段延迟
  - **TPOT (Time Per Output Token)**：解码阶段每 token 时间
  - **Memory Footprint**：显存占用（GB）

### 🔁 基线方法对比
| 类型 | 方法 |
|------|------|
| **Eviction-based** | StreamingLLM [14], H2O [15], SnapKV [13] |
| **Merging-based** | CaM [19], D2O [18] |
| **Full Model** | 不压缩的完整 KV cache（上限参考） |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ LongBench 平均得分（Table 1）
| 方法 | Llama-3-8B @50% | Mistral-7B @50% |
|------|------------------|------------------|
| Full Model | 34.88 | 42.01 |
| D2O | 31.86 | 40.02 |
| **Ours (SemantiCache)** | **32.34** | **40.87** |

👉 在相同压缩比例下，**全面超越所有基线**，接近甚至逼近全模型性能。

#### ✅ Needle-in-a-Haystack 准确率（Table 2）
| 方法 | L=8k (1024 budget) | L=32k (4096 budget) |
|------|--------------------|---------------------|
| Full Model | 97.56 | 93.85 |
| D2O | 93.21 | 90.29 |
| **Ours** | **94.38** | **91.15** |

👉 展现出更强的**长距离信息定位能力**，说明语义完整性保护有效。

#### ✅ 效率提升（Table 3，Llama-3-8B @32k context）
| 方法 | TTFT (s) | TPOT (s) | Speedup × | Memory (GB) |
|------|----------|----------|-----------|--------------|
| Full Model | 4.12 | 0.081 | 1.0× | 24.27 |
| StreamingLLM | 4.12 | 0.032 | 2.53× | 16.12 |
| **Ours** | **4.25** | **0.031** | **2.61×** | **15.94** |

👉 实现当前最优的 **2.61× 解码速度提升** 和最低内存占用。

### 🔍 消融实验结果（Ablation Study）

#### （1）不同分块策略对比（Table 4）
| 方法变体 | LongBench Avg. Score (@20%) |
|---------|------------------------------|
| 固定大小分块（64 tokens） | 28.22 |
| 无分块（全局聚类） | 26.94 |
| **语义分块（SemantiCache）** | **30.01** |

✅ 表明 **基于自然 delimiter 的语义分块至关重要**，能显著提升压缩后性能。

#### （2）相似度阈值 $ T $ 影响（Table 5）
| $ T $ | 平均得分 | 压缩率 (%) |
|-------|----------|------------|
| 0.50 | 27.21 | 86.3% |
| 0.70 | 30.12 | 79.8% |
| 0.80 | 32.87 | 51.2% |
| 0.90 | 34.05 | 16.7% |

✅ 存在明显 trade-off：**更高的 $ T $ → 更少合并 → 更高性能但更低压缩率**。可根据实际需求调节。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **语义碎片化是现有 KV cache 压缩方法性能下降的根本原因之一**。
2. **Aligning compression with linguistic hierarchy（将压缩与语言层次结构对齐）可显著提升压缩质量**。
3. **SemantiCache 在保持高压缩率的同时，几乎不牺牲模型性能**，尤其在需要精确语义理解的任务中优势明显。
4. **GSC + Proportional Attention 构成了高效且可扩展的轻量化压缩流程**，适用于主流开源 LLM。

### ⚠️ 方法的局限性
- **依赖 delimiter 的有效性**：对于缺乏明确标点的语言（如中文古文）或口语化文本，delimiter 可能不够可靠。
- **聚类顺序敏感性**：GSC 是单向贪心算法，可能受 token 顺序影响，存在局部最优风险。
- **静态阈值 $ T $**：未实现动态调整，难以适应不同语义密度的内容。

### 🔮 未来工作方向
- 扩展到更多语言（尤其是低资源语言）；
- 引入动态阈值选择机制（例如基于句法分析或注意力分布）；
- 探索与稀疏 attention、prefix tuning 等其他推理优化技术的联合应用；
- 在更大模型（如 Llama-3-70B）上验证泛化性。

---

> 💡 **一句话总结**：  
> SemantiCache 通过“**语义分块 + 种子聚类 + 比例注意力**”三步走策略，首次系统性解决了 KV cache 压缩中的 **semantic fragmentation** 问题，在实现高达 **2.61× 解码加速** 和 **显著降内存** 的同时，性能仍逼近全模型，代表了高效推理领域的重要进展。

</details>

---

### 11. [Efficient Document Parsing via Parallel Token Prediction](https://arxiv.org/abs/2603.15206)

**Authors**: Lei Li, Ze Zhao, Meng Li, Zhongwang Lun, Yi Yuan, Xingjing Lu, Zheng Wei, Jiang Bian, Zang Li  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.15206v1  

#### Abstract
Document parsing, as a fundamental yet crucial vision task, is being revolutionized by vision-language models (VLMs). However, the autoregressive (AR) decoding inherent to VLMs creates a significant bottleneck, severely limiting parsing speed. In this paper, we propose Parallel-Token Prediction (PTP...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Efficient Document Parsing via Parallel Token Prediction》核心总结**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
- **问题背景**：Document Parsing 是将非结构化或半结构化的文档（如PDF）转换为结构化、机器可读输出的关键任务，广泛应用于 RAG、文档分析等场景。
- **核心瓶颈**：当前主流基于 Vision-Language Models (VLMs) 的方法采用 **Autoregressive (AR) 解码机制**（即 Next-Token Prediction, NTP），逐个生成 token，导致推理速度慢，严重制约实际部署效率。

### ✅ **提出了什么新方法或新思路**
提出 **Parallel Token Prediction (PTP)**，一种**模型无关、即插即用、简单高效**的并行 token 预测框架，用于加速 VLM 在文档解析中的解码过程。

#### **核心技术思想**：
- 引入可学习的 **register tokens** 插入输入序列中。
- 在训练阶段，这些 register tokens 被训练来预测未来多个位置的 token（例如，`r_i+1` 预测 `x_i+2`）。
- 在推理时，每步解码附加 N 个 register tokens，模型可**并行生成 N+1 个新 token**，实现理论上的 N 倍加速。

> 🔧 **关键设计**：
> - Register tokens 共享同一嵌入向量，通过位置编码区分；
> - 设计特殊的 **causal attention mask**，确保 regular 和 register tokens 的训练独立性；
> - 推理时不丢弃 register tokens，而是利用其预测能力进行加速。

### ✅ **相比现有方法的优势**
| 对比维度 | PTP | 传统 AR 方法（NTP） | 多 token 预测（MTP） |
|--------|-----|------------------|------------------|
| **解码方式** | 并行多 token 输出 | 串行单 token 输出 | 自回归式多头预测 |
| **架构修改** | 无需改动主干 | 无 | 需额外 head/block |
| **训练效率** | 收敛快、稳定 | 正常 | 较慢，对 head 数敏感 |
| **推理效率** | 加速 1.6×–2.2×，接受率 100% | 慢 | 接受率仅 ~70% |
| **通用性** | 模型无关、任务可迁移 | — | 通常需定制 |

> 💡 **优势总结**：PTP 在不牺牲准确性的前提下，显著提升解码速度，并减少模型幻觉，具备强泛化性和兼容性。

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
- **OmniDocBench** [27]  
  当前最全面的文档解析 benchmark，涵盖 9 类 PDF 页面（学术论文、教科书、财务报告等）、多种布局与语言类型，评估文本与公式识别性能。
  
- **olmOCR-bench** [29]  
  包含 1,402 份真实 PDF 文档，分为 7 个子集（arXiv Math、Old Scans、Tables 等），用于细粒度评估。

此外，作者构建了一个高质量的 **layout-level document parsing dataset**，包含约 **180万样本**，来源包括：
- 开源数据集（DocLayNet、GNHK、CASIA-HWDB）
- 内部真实文档
- 合成生成数据（字体、样式增强）

### ⚙️ **实验设置**
- **基础模型**：`Qwen2.5-VL-3B-Instruct`
- **训练细节**：
  - 使用自建数据集微调
  - 最大 register token 数 `n=3`，损失权重 α=0.5
  - 冻结视觉编码器和 aligner，仅更新 LLM 权重
  - 学习率 2e-5，训练 1 轮，使用 8×A100 40GB GPU
- **推理平台**：H20 (90G) GPU，集成至 KsanaLLM [38]

### 🎯 **评估指标**
| 类别 | 指标 | 说明 |
|------|------|------|
| **准确性** | Edit Distance ↓ | 字符级编辑距离，越低越好 |
|           | CDM ↑ | Character Detection Matching，公式识别专用指标 |
|           | BLEU ↑ | 公式语义相似度 |
|           | Accuracy | VLU 任务准确率（如 ScienceQA） |
| **效率**   | Time Per Output Token (TPOT) ↓ | 单个输出 token 所需时间 |
|           | Inter-Token Latency (ITL) ↓ | 相邻 token 间延迟 |
|           | Throughput / QPS ↑ | 每秒查询数或输出 token 数 |
|           | Speedup Ratio | 相对于 NTP 的加速比 |
| **可靠性** | Hallucination Rate ↓ | 错误生成非图像内容的比例 |
|           | Acceptance Rate (%) | speculative decoding 中草案 token 接受比例 |

### 🆚 **基线方法对比**
| 类型 | 方法 |
|------|------|
| **Pipeline Tools** | Marker-1.8.2, MinerU2-pipeline, PP-StructureV3 |
| **General VLMs** | GPT-4o, Gemini-2.5 Pro, InternVL3-76B, Qwen2.5-VL |
| **Specialized VLMs** | Dolphin, olmOCR-7B, MonkeyOCR, dots.ocr, MinerU2.5 |

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据**

#### ✅ **在 OmniDocBench 上的表现（文本识别）**
| 模型 | Text Edit Distance ↓ | 相对提升 |
|------|------------------|---------|
| Qwen2.5-VL-3B-NTP | 0.0585 | — |
| **Qwen2.5-VL-3B-PTP-1** | **0.0431** | **↓26.3%** |
| Qwen2.5-VL-3B-PTP-2 | 0.0589 | ≈持平 |

> 🔍 **发现**：PTP-1 不仅未损失精度，反而提升了文本识别效果，表明 PTP 可增强上下文建模，降低幻觉。

#### ✅ **公式识别性能（OmniDocBench）**
| 模型 | CDM ↑ | BLEU ↑ | Edit Distance ↓ |
|------|-------|--------|---------------|
| NTP | 71.65 | 63.05 | 0.226 |
| **PTP-1** | **89.63** | **62.32** | **0.236** |
| PTP-2 | 77.23 | 57.92 | 0.284 |

> ✅ PTP-1 在公式识别上表现优异，接近最优水平；PTP-2 性能略有下降，但仍优于多数 baseline。

#### ✅ **效率提升（Decoding Speed）**
| 模型 | 加速比 (Speedup Ratio) | Throughput 提升 |
|------|---------------------|----------------|
| PTP-1 | **1.6×** | 显著降低 TPOT 和 ITL |
| PTP-2 | **2.2×** | 最高吞吐提升 |

> 📈 图 3 显示：PTP-1 和 PTP-2 分别达到 1.6× 和 2.2× 的端到端加速，且随 QPS 增加仍保持稳定。

#### ✅ **与其他加速方法比较**
| 方法 | 接受率 | 实际加速 | 是否需要辅助模块 |
|------|------|--------|----------------|
| MTP | ~70% | <1.5× | 是（额外 head） |
| **PTP** | **100%** | **1.6×–2.2×** | 否（仅插入 token） |

> ✅ PTP 实现更高接受率和更优加速，且无需额外参数。

#### ✅ **结合 Speculative Decoding 的泛化能力**
在 **ScienceQA** 等复杂 VLU 任务中：
| 模型 | 准确率 | Acceptance Rate |
|------|------|----------------|
| NTP | 92.21% | N/A |
| PTP-1 | 91.72% | 100% |
| PTP-1 + Self-Speculative | **92.21%** | **82%** |

> ✅ 表明 PTP 可无缝融合 speculative decoding，在保持精度的同时大幅提升推理效率。

---

### 🔍 **消融实验结果（Ablation Study）**

| 设置 | 结果分析 |
|------|----------|
| **共享 vs. 独立 register embedding** | 共享 embedding 效果略优，参数更少，收敛更快 |
| **连续 vs. 交错插入 register tokens** | 连续插入（continuous）优于交错（interleaved），尤其对 PTP-2 更有效，因能复用中间信息 |
| **是否替换 KV Cache** | 若不替换 register 的 KV cache，性能大幅下降（CDM 从 89.63 → 39.43）<br>✅ **证明 cache 替换机制至关重要** |

> 📌 **结论**：PTP 的设计细节（如 register 插入方式、KV cache 管理）对其性能有显著影响。

---

## 4. **关键结论和发现**

### ✅ **主要发现**
1. **文档解析本质是高确定性转录任务**，天然支持并行处理，而传统 AR 方法未能充分利用这一特性。
2. **PTP 成功将并行性“内化”进模型本身**，通过 register tokens 实现多 token 并行预测，打破 AR 瓶颈。
3. **PTP 不仅提速，还改善性能**：
   - 减少幻觉（hallucination mitigation）
   - 提升长程依赖建模能力
   - 数据利用率更高
4. **高度通用**：
   - 跨模型（Qwen、InternVL、MonkeyOCR 等均受益）
   - 跨任务（OCR、VLU、Math Reasoning）
   - 支持 extrapolation（训练 n=2，推理 n=3 仍有效）

### ⚠️ **局限性**
- **远距离预测误差传播**：当 register 预测过远 future token 时，准确性下降。
- **计算开销边际增加**：虽然内存受限不影响吞吐，但 register 引入少量额外计算。
- **对极端噪声敏感**：尽管优于 NTP，但在严重模糊/扭曲图像中仍有挑战。

### 🔮 **未来工作方向**
1. **动态 register 数量控制**：根据文档复杂度自适应调整并行度。
2. **与视觉 token 压缩技术联合优化**：结合 DeepEncoder [44] 或 token pruning [36] 进一步压缩输入。
3. **扩展至视频或多页文档理解**：探索跨页面的 register 传递机制。
4. **构建更大规模 layout-level 数据集**：推动社区发展更精细的文档理解基准。

---

> ✅ **总体评价**：  
> 本文提出的 **PTP 方法是一项简洁而强大的创新**，在不改变模型架构的前提下，实现了高达 **2.2× 的解码加速**，同时**保持甚至提升精度**，并有效缓解幻觉问题。它不仅适用于文档 OCR，也为所有以转录为核心的 VLM 应用提供了新的高效范式。

</details>

---

### 12. [DOS: Dependency-Oriented Sampler for Masked Diffusion Language Models](https://arxiv.org/abs/2603.15340)

**Authors**: Xueyu Zhou, Yangrong Hu, Jian Huang  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.15340v1  

#### Abstract
Masked diffusion language models (MDLMs) have recently emerged as a new paradigm in language modeling, offering flexible generation dynamics and enabling efficient parallel decoding. However, existing decoding strategies for pre-trained MDLMs predominantly rely on token-level uncertainty criteria, w...

---

### 13. [Committee Configuration Optimization for Parallel Byzantine Consensus in a Trusted Execution Environment](https://arxiv.org/abs/2603.14445)

**Authors**: Yifei Xie, Btissam Er-Rahmadi, Xiao Chen, Tiejun Ma, Jane Hillston  
**Category**: cs.DC  
**Published**: 2026-03-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.14445v1  

#### Abstract
Parallel Byzantine Fault Tolerant (BFT) protocols based on committee-based sharding improve scalability but weaken safety since smaller node groups are responsible for consensus. Recent approaches integrate trusted execution environments (TEEs) into parallel BFT frameworks to enhance safety. While t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Committee Configuration Optimization for Parallel Byzantine Consensus in a Trusted Execution Environment*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 Byzantine Fault Tolerant (BFT) 协议面临 **可扩展性差** 和 **安全性弱化** 的双重挑战：
- 经典 BFT（如 PBFT）通信复杂度为 $O(N)$，难以支持大规模节点；
- 并行 BFT（Parallel BFT）通过委员会分片（committee-based sharding）提升吞吐量，但小规模委员会降低了系统容错能力，削弱了安全性；
- 当前多数并行 BFT 系统采用随机分配节点到委员会的方式，导致性能不稳定，尤其在网络延迟高或节点故障率不均时表现更差；
- 引入 Trusted Execution Environment (TEE) 可增强安全性（允许从 3f+1 降至 2f+1 节点），但 TEE 硬件失效会触发 fallback 到传统模式，现有配置方法未对此进行优化。

### 🚀 提出的新方法与创新思路
本文提出 **Committee Configuration Optimization (CCO)** 模型，基于 **Mixed Integer Programming (MIP)** 对 TEE 支持下的并行 BFT 系统中的委员会成员配置进行优化：

- **首次将数学规划用于 TEE-based Parallel BFT 的委员会配置优化**，综合考虑通信延迟、节点故障率等因素；
- 设计了一个联合决策变量模型，包括：
  - **Partition Variables**：决定哪个节点作为 leader，以及其所属 committee；
  - **Connection Variables**：建模 leader 与 follower 之间的通信关系；
- 明确建模 **正常运行** 和 **TEE 失效后的 fallback 场景**，实现自适应重配置；
- 在目标函数中最小化单笔交易的端到端延迟 $T_{tr}$，涵盖 Pre-prepare、Prepare、Verify、Commit 等阶段的时间开销。

### 🔍 相比现有方法的优势
| 方面 | 传统方法 | 本文 CCO 模型 |
|------|--------|-------------|
| 配置方式 | 随机分配（randomized assignment） | 数学优化驱动（MIP-based） |
| 性能影响因素 | 忽略网络延迟与节点可靠性差异 | 显式建模 $d_{ij}$ 和失败率 $b_i, c_i$ |
| 容错机制 | 固定配置，fallback 无优化 | 自适应 fallback，仅受影响 committee 升级至 3f+1 |
| 安全性保障 | 依赖大委员会或随机性 | 结合 TEE 安全假设与动态安全边界 |

> ✅ **核心优势**：在保证安全性的前提下，显著提升系统吞吐量并降低共识延迟，特别是在 TEE 故障等异常场景下仍能维持高性能。

---

## 2. 核心实验方法和设置

### 💾 实验平台与环境
- **测试床实现语言**：C 语言
- **TEE 实现**：基于 Intel SGX 构建可信单调计数器（trusted monotonic counters）
- **部署平台**：Microsoft Azure 云服务
- **硬件资源**：5 台虚拟机（VM），每台 8 vCPU + 64 GB RAM
- **模拟节点规模**：最多 240 个节点实例分布在 5 个 VM 上
- **协议实现**：集成 CCO 模型到 TopBFT 协议框架中（来自文献 [6]）

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Throughput** | 吞吐量，单位为 operations per second (op/s)，衡量系统处理能力 |
| **Latency** | 延迟，单位为毫秒（ms），指从客户端发送请求组到收到所有有效回复的时间 |

### ⚖️ 基线方法对比
实验对比了以下代表性 BFT 协议：
1. **HotStuff**：现代主流 BFT，优于 BFT-SMaRt
2. **FastBFT**：基于 TEE 的高效 BFT，优于 MinBFT/CheapBFT
3. **GeoBFT**：地理感知的并行 BFT 方案
4. **TopBFT (default)**：原始 TopBFT，使用默认随机委员会配置（4 节点/committee）
5. **CCO-driven TopBFT**：本文提出的优化版本

此外，在 fallback 实验中还比较了：
- **Standard Fallback TopBFT**：所有 committee 强制切换到 3f+1 模式
- **CCO-driven Fallback TopBFT**：仅受 TEE 故障影响的 committee 动态升级

---

## 3. 主要实验结果和性能指标

### 📈 正常操作下的性能表现（Section 5.1 & 5.2）

#### ✅ 吞吐量提升（图 2a, 图 3a）
| 条件 | CCO-driven TopBFT vs Default TopBFT |
|------|-------------------------------|
| 节点数 = 240，payload = 1MB | **提升约 15%**（从 ~200 op/s → ~230 op/s） |
| payload = 1MB，固定 200 节点 | **最高提升达 29.4%** |

> 💡 原因：CCO 减少了跨节点通信延迟，并避免慢节点分散在多个 committee 中造成“木桶效应”。

#### ✅ 延迟降低（图 2b, 图 3b）
| 条件 | 性能改善 |
|------|---------|
| 节点数 > 200 时（验证委员会饱和） | 默认 TopBFT 延迟急剧上升；CCO 保持稳定 |
| payload = 1MB | CCO-driven TopBFT 延迟比标准 TopBFT **低 18.9%** |

> 📉 CCO 成功缓解了验证委员会瓶颈问题，提升了整体响应速度。

---

### 🔁 Fallback 场景下的性能表现（Section 5.3，图 4）

设定 **30% 节点发生 TEE 故障**，触发 fallback 机制：

| 指标 | CCO-driven vs Standard Fallback |
|------|------------------------------|
| **Throughput** | **提升约 21%** |
| **Latency** | 显著低于标准 fallback，说明优化后重配置效率更高 |

> ✅ 关键发现：CCO 实现了 **选择性 fallback** —— 只有包含故障 TEE 的 committee 才启用 3f+1 模式，其余继续以 2f+1 高效运行，从而减少整体性能损失。

---

### ❌ 消融实验（Ablation Study）
虽然文中未明确标注“ablation study”章节，但从设计逻辑可推断出以下隐含消融分析：
- 若移除对 $d_{ij}$ 的建模 → 将退化为随机配置，无法应对网络异构性；
- 若忽略 failure rate 约束（公式 12–13）→ 可能选中不稳定节点作 leader，降低系统鲁棒性；
- 若禁用 adaptive fallback（公式 15–16）→ 所有 committee 强制扩容，带来不必要的通信开销。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **委员会配置对并行 BFT 性能具有决定性影响**，简单的随机分配策略远非最优；
2. **基于 MIP 的 CCO 模型能有效整合网络延迟、节点可靠性、TEE 状态等多维信息**，生成接近全局最优的 committee 结构；
3. **在正常运行和 TEE 故障两种场景下，CCO 均带来显著性能增益**：
   - 吞吐量提升 **15%~29.4%**
   - 延迟降低 **近 19%**
   - fallback 场景下吞吐量额外提升 **21%**
4. **adaptive fallback 机制是关键创新**：它实现了细粒度的安全降级，避免“一刀切”带来的性能浪费。

### ⚠️ 局限性
1. **计算开销较高**：MIP 求解器（如 CPLEX）在大规模节点下可能引入配置延迟，不适合频繁动态调整；
2. **静态配置为主**：当前模型适用于相对稳定的网络拓扑，对极端动态变化（如高频节点进出）支持有限；
3. **依赖 TEE 普及程度**：若 TEE 支持不足或存在兼容性问题，fallback 触发频率增加，影响长期稳定性；
4. **未考虑恶意行为建模**：假设故障为 crash 或随机延迟，未深入探讨针对性攻击下的鲁棒性。

### 🔮 未来工作方向
1. **轻量化优化算法**：探索启发式或近似算法替代完整 MIP 求解，提升实时性；
2. **在线自适应配置**：结合监控反馈实现闭环控制，动态更新 committee 配置；
3. **跨层协同优化**：将 CCO 与区块打包大小、共识轮次调度等参数联合优化；
4. **扩展至 PoS 或混合机制**：结合 stake weight 或 reputation score 进行加权配置；
5. **实测于真实区块链系统**：集成进 Hyperledger Fabric、Diem-like 等生产级系统验证实用性。

---

## ✅ 总结一句话
> 本文提出了首个面向 **TEE-based Parallel BFT** 的 **基于 MIP 的委员会配置优化模型 CCO**，通过联合优化通信效率与容错能力，在正常和故障场景下均实现了 **显著的性能提升（吞吐 +15%~21%，延迟 ↓19%）**，为构建高安全、高性能的下一代分布式账本系统提供了重要技术路径。

</details>

---

### 14. [Machine Learning Models to Identify Promising Nested Antiresonance Nodeless Fiber Designs](https://arxiv.org/abs/2603.13302)

**Authors**: Rania A. Eltaieb, Sophie LaRochelle, Leslie A. Rusch  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.13302v1  

#### Abstract
Hollow-core fibers offer superior loss and latency characteristics compared to solid-core alternatives, yet the geometric complexity of nested antiresonance nodeless fibers (NANFs) makes traditional optimization computationally prohibitive. We propose a high-efficiency, two-stage machine learning fr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Machine Learning Models to Identify Promising Nested Antiresonance Nodeless Fiber Designs*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统优化方法在设计 **Nested Antiresonant Nodeless Fibers (NANFs)** 时面临巨大挑战：
- NANF 几何参数空间复杂，维度高；
- 使用有限元方法（如 COMSOL）进行仿真耗时极长，难以支持大规模搜索；
- 现有机器学习模型多为分类器，只能提供定性预测（如“低/中/高损耗”），无法精确预测 **Confinement Loss (CL)**。

因此，如何以**极小训练成本**实现对高性能 NANF 设计的高效、准确识别，是本文要解决的核心问题。

---

### 🚀 提出的新方法与创新思路

提出一种**两阶段神经网络框架**（two-stage ML framework）用于快速识别高性能 NANF 结构：

1. **Stage 1: 分类器 NN（Classifier NN）**
   - 输入：6维几何参数 `[Dclad, Dcore, Dnest, Dcap, Dcap(1−α), g]`
   - 输出：判断是否为“interesting design”
     - 定义标准：**单模性**（Suppression Ratio ≥ 50 dB）且 **CL < 10³ dB/km**
   - 使用 ReLU 激活函数 + Adam 优化器，输出层用 sigmoid 进行二分类。

2. **Stage 2: 回归器 NN（Regressor NN）**
   - 只对被分类器选中的“interesting”设计进行训练；
   - 预测目标：**CL 的常用对数 log₁₀(CL)**，而非原始 CL 值；
     - 创新点：通过取对数缓解 CL 动态范围过大带来的训练困难；
   - 最终将预测值转换回线性尺度（dB/km）用于评估。

该方法实现了从“粗筛”到“精估”的流程解耦，显著提升效率与精度。

---

### ⚖️ 相比现有方法的优势

| 方面 | 本文方法 | 先前方法（如 [13][14][17]） |
|------|----------|-----------------------------|
| 数据需求 | 极少（仅 **1,819 个样本**） | 数万至数十万样本（>60k–290k） |
| 预测粒度 | 回归模型 → **连续 CL 值预测** | 多为分类模型 → 仅区间估计 |
| 单模性考虑 | 显式建模（SR ≥ 50 dB） | 多数未联合建模模式抑制 |
| 泛化能力 | 成功外推至训练集之外区域（CL < 1 dB/km） | 多限于插值范围内 |
| 计算开销 | NN 推理速度比 FEM 快数个数量级 | 依赖大量 FEM 仿真生成数据 |

> ✅ **核心优势总结**：  
> 在**极小数据量下实现高精度、可泛化的 NANF 性能预测**，并能有效指导超大设计空间（~14e6）的快速探索。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集

- **主数据集（Master Dataset）**：
  - 总规模：**18,188 个 NANF 设计**
  - 来源：COMSOL Multiphysics 仿真，在中心波长 **1400 nm** 下计算：
    - Fundamental mode 的 **CL (dB/km)**
    - First higher-order mode 的 **CL₁ (dB/km)**
    - 抑制度（Suppression Ratio, SR = CL₁ − CL）
  - **剔除条件**：排除所有 CL < 1 dB/km 的设计（共 234 个），确保训练集中无“最优设计”，用于后续验证外推能力。

- **子数据集构建方式**：
  - 选取不同大小 `n ∈ {1,819, 3,000, ..., 18,188}` 的随机子集；
  - 每个子集划分为：训练集（80%）、验证集（10%）、测试集（10%）；
  - 多次随机采样以评估稳定性。

---

### 🔬 实验设置

#### 几何参数空间（4D 参数向量）
- `[Dcore, Dcap, α, Dnest]`
- 范围如下：
  - `Dcore`: [20.0 : 1.0 : 60.0] μm
  - `Dcap`: [25.8 : 0.5 : 54.3] μm
  - `α`（嵌入率）: [0.0 : 0.05 : 0.5]
  - `Dnest`: [0.1 : 0.05 : 0.6] × (1−α)Dcap
- 衍生参数：`Dclad`, `g` 由上述参数决定，并施加物理约束（如 `3μm < g < 6μm`）

#### 搜索空间（Search Space）
- 扩展后的待搜索设计总数：**~14e6**
- 采用均匀采样 + 网格扫描方式覆盖更密集的设计组合。

---

### 📈 评估指标

| 模块 | 指标 |
|------|------|
| **Classifier** | False Positive Rate (FPR), False Negative Rate (FNR) |
| **Regressor** | Absolute Relative Error (ARE): $\epsilon = \left|\frac{CL_t - CL_p}{CL_t}\right|$<br>Mean ARE over test samples with $CL_t \leq 6.3\,\text{dB/km}$<br>Probability Density Function (PDF) of ARE |
| **Overall Performance** | Best identified design’s **confirmed CLₜ** via COMSOL<br>Comparison between predicted vs. true CL |

---

### 🆚 基线方法对比

虽然文中未直接运行其他 ML 模型作为 baseline，但通过文献对比凸显优势：

| 文献 | 方法 | 数据量 | 是否回归 | 单模性建模 | 备注 |
|------|------|--------|---------|------------|------|
| [13] | 分类 NN | >290k | ❌ | ❌ | 定性预测 CL 区间 |
| [14] | 分类 CNN | 60k | ❌ | ❌ | 仅针对 NANF |
| [17] | 回归 NN (CNN) | 78,312 | ✅ | ❌ | 使用图像式输入，高维 |
| **本文** | **两阶段 NN（MLP）** | **1,819** | ✅ | ✅ | **最小数据 + 外推成功** |

> 💡 强调：本文方法不仅数据效率更高，且结构更简单（全连接网络即可胜任），更适合工程部署。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### A. 分类器性能（Fig. 4）
- 当数据集大小 ≥ 12,732 时，FNR 和 FPR 收敛；
- 即使在最小数据集（n=1,819）上：
  - **FNR ≈ 2.9%** → 能检测出 **≥97.1% 的优质设计**
  - **FPR ≈ 7%** → 少量误报，但不影响最终筛选（额外计算代价低）

> ✔️ 小数据足以支撑稳定分类。

#### B. 回归器性能（Fig. 5a & 5b）
- 对于真实 CL ≤ 6.3 dB/km 的设计：
  - 平均绝对相对误差（Mean ARE）：
    - n = 1,819：**6.9%**
    - n = 18,188：**5.4%**
  - PDF 分布显示即使小数据也能获得接近大样本的误差分布；
  - 相对误差标准差极低（<0.07），说明预测一致性好。

> ✔️ 小数据集仍可实现高精度 CL 回归预测。

#### C. 最优设计发现能力（Fig. 6–7）

| 数据集大小 | 最佳预测 CLₚ | 最佳确认 CLₜ | 外推幅度 |
|-----------|--------------|---------------|----------|
| n = 18,188（全集） | 0.18 dB/km | **0.25 dB/km** | 从 1→0.25 dB/km |
| n = 1,819（最小） | 0.22 dB/km | **~0.25 dB/km** | 同样突破 1 dB/km 下限 |

> 🔥 **关键突破**：尽管训练集中最低 CL 为 1 dB/km，模型成功外推并找到 **CLₜ = 0.25 dB/km** 的设计！

#### D. 不同数据集规模的影响（Fig. 7）
- 预测最佳 CLₚ 跨度：**0.18–0.29 dB/km**
- 真实最佳 CLₜ 跨度：仅 **0.25–0.29 dB/km**（变化极小）
- 表明：**即使预测有偏差，最终确认性能高度一致**

> ✅ 小数据训练的 NN 也能找到接近全局最优的设计。

---

### 🔍 消融实验与敏感性分析（Section VI）

#### A. 搜索空间大小影响（Fig. 8）
- 发现曲线极为陡峭 → 很快达到性能饱和；
- **6e6 搜索空间已足够**，进一步扩大收益有限；
- 但由于 NN 推理极快，使用 14e6 也无额外负担。

#### B. Top-18 设计的一致性（Fig. 9–10）
- Top 18 设计的标准差：
  - 预测 CLₚ：σ < 0.011 dB/km
  - 真实 CLₜ：略高，但仍非常集中
- 表明：NN 输出的“最佳候选群”具有高度一致性。

#### C. 多次试验验证（Fig. 11–12）
- 在 20 次独立训练（n=1,819）中：
  - 所有模型均能找到 CLₜ < 0.3 dB/km 的设计；
  - 预测误差最大约 0.1 dB/km（≈41% 相对误差），但在尾部概率低；
  - △CL 的方差仅为 **9.3e⁻⁵ (dB/km)²**，表明鲁棒性强。

> ✅ 方法在多次随机初始化下表现稳定，具备实用可靠性。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **极小数据即可训练高性能 ML 模型**：
   - 仅需 **1,819 个仿真样本**即可构建稳定、高精度的 NANF 性能预测模型；
   - 打破“大数据依赖”瓶颈，极大降低前期仿真成本。

2. **模型具备强外推能力**：
   - 成功识别出 **CL = 0.25 dB/km** 的设计，远低于训练集中最小值（1 dB/km）；
   - 表明 NN 学到了 **antiresonance guidance 的物理规律**，而非简单记忆数据。

3. **两阶段架构高效可靠**：
   - 分类先行 → 减少无效回归计算；
   - 对数域回归 → 解决动态范围问题；
   - 整体推理速度比 FEM 快 **数万倍以上**。

4. **搜索空间无需极致精细**：
   - 中等规模搜索（~6e6）即可捕获优质设计；
   - 实际应用中可用更大空间“暴力穷举”而不增加时间成本。

---

### ⚠️ 方法的局限性

1. **依赖高质量标签数据**：
   - 虽然数据量小，但仍需精确的 COMSOL 仿真结果作为监督信号；
   - 若初始数据噪声大或覆盖不均，可能影响泛化。

2. **局限于特定 NANF 结构**：
   - 当前模型适用于圆形、4毛细管结构的 NANF；
   - 拓展至 DNANF 或非对称结构需重新建模。

3. **未考虑制造公差与弯曲效应**：
   - 当前优化基于理想几何；
   - 实际光纤性能受工艺偏差、弯曲等因素影响，尚未纳入模型。

---

### 🔮 未来工作方向

1. **引入不确定性建模**：
   - 使用贝叶斯神经网络或集成学习量化预测置信度；
   - 辅助主动学习策略，智能选择最有价值的新仿真点。

2. **扩展至多波段联合优化**：
   - 当前训练集中在 1400 nm，未来可构建宽谱性能预测模型；
   - 如 Fig. 13 所示，所发现设计在 O/C/L-band 均表现良好，值得深入挖掘。

3. **闭环逆向设计系统**：
   - 将正向预测模型嵌入遗传算法或强化学习框架；
   - 实现全自动、端到端的 NANF 结构逆向设计。

4. **融合制造约束**：
   - 在损失函数中加入可制造性惩罚项；
   - 实现“设计即可用”的工业级优化。

---

## ✅ 总结一句话

> 本文提出了一种**高效、轻量、可外推**的两阶段神经网络框架，仅用 **1,819 个样本**就成功从 **1400 万个候选设计**中定位出 **CL 低至 0.25 dB/km** 的高性能 NANF 结构，为下一代超低损 hollow-core fibers 的快速研发提供了强有力的 AI 工具链。

</details>

---

### 15. [CAMD: Coverage-Aware Multimodal Decoding for Efficient Reasoning of Multimodal Large Language Models](https://arxiv.org/abs/2603.14745)

**Authors**: Huijie Guo, Jingyao Wang, Lingyu Si, Jiahuan Zhou, Changwen Zheng, Wenwen Qiang  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.14745v1  

#### Abstract
Recent advances in Multimodal Large Language Models (MLLMs) have shown impressive reasoning capabilities across vision-language tasks, yet still face the challenge of compute-difficulty mismatch. Through empirical analyses, we identify that existing decoding methods may waste compute on easy cases w...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CAMD: Coverage-Aware Multimodal Decoding for Efficient Reasoning of Multimodal Large Language Models

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Multimodal Large Language Models (MLLMs)** 在视觉-语言推理任务中表现出色，但在推理过程中存在 **compute-difficulty mismatch**（计算资源与实例难度不匹配）问题：
- 对于简单的样本，采用固定采样策略（如 Best-of-N）会造成 **过度计算（over-computation）**；
- 对于复杂的样本，则因计算不足导致 **推理覆盖不充分（insufficient coverage）**，影响准确性和效率。

这种“一刀切”的解码方式在面对多模态任务中广泛存在的 **重尾难度分布（heavy-tailed difficulty distribution）** 时尤为低效。

---

### 🚀 提出的新方法：Coverage-Aware Multimodal Decoding (CAMD)

CAMD 是一种**自适应推理机制**，通过动态分配计算资源来提升 MLLMs 的推理效率与可靠性。其核心思想是：  
> “难样本多算，易样本少算”，基于估计的不确定性动态调整采样预算。

#### 创新点包括：
1. **理论框架构建**  
   建立了一个连接 **sampling coverage（采样覆盖率）、instance difficulty（实例难度）和 residual risk（残余风险）** 的理论模型，揭示了多模态推理中失败概率主要由少数高难度/模糊样本主导（即重尾现象），从而证明固定采样策略本质上是非最优的。

2. **三阶段自适应解码流程**  
   CAMD 包含三个关键模块：
   - **Evidence-weighted Scoring**：综合生成置信度（generation confidence）、跨模态一致性（cross-modal consistency）和推理连贯性（reasoning coherence）对每个候选答案打分，作为单次尝试成功概率 $ s $ 的估计。
   - **Posterior Coverage Estimation**：将语义相似的答案聚类，利用加权得分估算当前采样集中包含正确答案的概率 $ p^* $。当 $ p^* \geq 1-\delta $ 时停止采样。
   - **Bayesian Adaptive Sampling**：使用 Dirichlet 后验更新不同语义簇的成功概率，并据此重新加权 token 分布，在下一轮采样中聚焦更有希望的方向。

3. **无需训练的即插即用设计**  
   CAMD 是一个 **plug-and-play decoding wrapper**，仅依赖模型输出的候选序列进行重排序与终止判断，**无需微调或额外训练**，易于集成到现有 MLLM 流程中。

---

### 🔍 相比现有方法的优势
| 维度 | 现有方法（如 Best-of-N, Self-consistency） | CAMD |
|------|----------------------------------------|-------|
| 采样策略 | 固定数量（uniform budget） | 动态调整（adaptive budget） |
| 资源分配 | 忽视实例差异 | 按难度分配计算 |
| 效率 | 易样本浪费资源，难样本覆盖不足 | 平衡精度与延迟 |
| 可扩展性 | 随 N 增大收益递减 | 收敛更快，边际效益更高 |

---

## 2. 核心实验方法和设置

### 📚 数据集
实验覆盖图像与视频两大模态，涵盖多种任务类型：

#### 图像任务：
- **综合性评测基准**：MMBench, LLaVA-Bench, MM-Vet
- **通用 VQA**：VizWiz, ScienceQA (SQA)
- **幻觉检测基准**：POPE-R, CHAIR
- **数学视觉推理**：MathVista（用于动机分析）

#### 视频任务：
- **零样本视频问答**：MSRVTT-QA, MSVD-QA, ActivityNet-QA
- **视频文本生成评测**：Video-Based Text Generation Benchmark（使用 GPT-3.5 Turbo 打分）

#### 自动评估：
- 使用 **GPT-4o** 对生成描述的质量进行评分，维度包括：
  - Accuracy（准确性）
  - Correctness（事实正确性）
  - Detailedness（细节丰富度）

---

### ⚙️ 实验设置
- **基础模型**：LLaVA-1.5, InstructBLIP, Video-LLaVA, Chat-UniVi, VILA, Video-LLaMA2 等主流 MLLMs。
- **解码参数**：
  - 温度 $ T=0.7 $, top-p=0.9, repetition penalty=1.05
  - 最大生成长度：16,384 tokens
- **token budget 范围**：128–2048，用于比较不同方法在有限预算下的表现
- **超参数设置**：
  - 权重系数：$ \lambda_g = 1, \lambda_c = 0.3 $
  - 置信阈值 $ \delta = 0.05 $, 温度 $ T=0.9 $

---

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| 固定采样 | Best-of-N ($ N=1,2,\dots,256 $) |
| 解码优化 | ICD, VCD, CGD, OPERA, FarSight |
| 推理增强 | Self-consistency, Tree-of-Thoughts（间接对比） |

---

## 3. 主要实验结果和性能指标

### 📊 性能提升汇总（来自 Tables 1 & 2）

#### 图像任务平均提升（vs Base Model）：
| 指标 | 提升幅度 |
|------|----------|
| Comprehensive Benchmarks (MMBench等) | **+2.2% ~ +4.3%** |
| General VQA (VizWiz, SQA) | **+1.5% ~ +3.5%** |
| Hallucination Metrics (↓CHAIRs/r, ↑POPE系列) | **-1.0 ~ -7.4 pts**, **+1.1% ~ +4.6%** |

> ✅ CAMD 显著降低幻觉率，同时提高整体准确率。

#### 视频任务平均提升（Table 2）：
| 指标 | 提升幅度 |
|------|----------|
| Accuracy (MSVD/ActivityNet) | **+0.8% ~ +3.7%** |
| Score (GPT-3.5评分) | **+0.1 ~ +0.5 pts** |
| Cr./Cs./De./Ct./Te.（五维） | **+0.16 ~ +0.75 pts** |

> ✅ 在复杂视频理解任务中也保持稳定增益。

---

### ⏱️ 效率优势（Figure 4）
- 在 **POPE-R 和 MSRVTT-QA** 上，CAMD 在更低的 token 预算下达到甚至超过其他方法的峰值性能。
- 例如，在 POPE-R 上，CAMD 在 **512 tokens** 即可逼近 FarSight 在 2048 tokens 的表现，节省约 **75% 计算量**。
- **无精度损失前提下显著减少延迟（latency）和生成 token 数量**。

---

### 🧪 消融实验（Ablation Study, Figure 6）
- 对证据加权中的两个超参数 $ \lambda_g $（对齐权重）和 $ \lambda_c $（连贯性权重）进行搜索。
- 最优配置为 $ \lambda_g = 0.9, \lambda_c = 0.7 $，验证了引入跨模态一致性和推理连贯性的必要性。
- 移除任一组件均导致性能下降，说明三者协同作用有效。

---

### 🤖 GPT-4o 辅助评估（Figure 5）
- CAMD 生成的回答在 **accuracy、correctness、detailedness** 上均优于基线。
- 尤其在 detailedness 上领先明显，表明其能生成更完整、细致且可靠的推理链。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **多模态推理具有显著的重尾难度分布**  
   少数高难度或模棱两可的样本主导了整体错误率，使得固定采样策略效率低下。

2. **自适应采样优于固定预算策略**  
   通过动态调整采样次数，可在保证高覆盖率的同时大幅节省计算资源。

3. **CAMD 实现帕累托前沿突破**  
   在多个 benchmark 上实现了 **精度与效率的双重提升**，尤其在抑制幻觉方面效果突出。

4. **无需训练即可部署**  
   作为一种 decoding-time 方法，CAMD 不改变模型结构或参数，具备良好的通用性和实用性。

---

### ⚠️ 局限性
1. **依赖高质量聚类与语义相似度判断**  
   当前使用 LLM 进行语义聚类，可能引入额外误差或延迟。
2. **超参数敏感性**  
   如 $ \delta $（置信阈值）、$ \lambda $ 系数需在验证集上调优，泛化能力有待进一步验证。
3. **未考虑视觉编码阶段开销**  
   实验假设视觉特征已缓存，仅语言解码阶段可变，实际端到端系统中视觉处理也可能成为瓶颈。

---

### 🔮 未来工作方向
1. **扩展至更多模态组合**  
   如音频+文本+图像的多模态联合推理场景。
2. **结合 planning-based reasoning 方法**  
   将 CAMD 与 ToT（Tree-of-Thoughts）、MCTS 等结构化搜索结合，实现更高效的探索。
3. **在线学习机制引入**  
   利用历史推理经验自动调整先验分布或超参数，实现持续优化。
4. **硬件感知的自适应调度**  
   结合 GPU 内存、带宽限制，设计资源受限下的最优采样策略。

---

## 总结
> **CAMD 从理论出发，揭示了 MLLMs 中 compute-difficulty mismatch 的本质问题，并提出了一套基于 coverage-aware 的自适应解码框架，在不增加训练成本的前提下，实现了更高效、更可靠、更少幻觉的多模态推理。**

该工作不仅提供了实用的解码工具，也为理解大规模模型的推理动态提供了新的理论视角，是迈向高效、可信多模态 AI 的重要一步。

</details>

---

### 16. [Lightweight User-Personalization Method for Closed Split Computing](https://arxiv.org/abs/2603.14958)

**Authors**: Yuya Okada, Takayuki Nishio  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.14958v1  

#### Abstract
Split Computing enables collaborative inference between edge devices and the cloud by partitioning a deep neural network into an edge-side head and a server-side tail, reducing latency and limiting exposure of raw input data. However, inference performance often degrades in practical deployments due...

---

### 17. [Think First, Diffuse Fast: Improving Diffusion Language Model Reasoning via Autoregressive Plan Conditioning](https://arxiv.org/abs/2603.13243)

**Authors**: Earl J St Sauver  
**Category**: cs.AI  
**Published**: 2026-03-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.13243v1  

#### Abstract
Diffusion large language models (dLLMs) generate text via iterative denoising but consistently underperform on multi-step reasoning. We hypothesize this gap stems from a coordination problem: AR models build coherence token-by-token, while diffusion models must coordinate all positions simultaneousl...

---

### 18. [Selective Fine-Tuning of GPT Architectures for Parameter-Efficient Clinical Text Classification](https://arxiv.org/abs/2603.14183)

**Authors**: Fariba Afrin Irany, Sampson Akwafuo  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.14183v1  

#### Abstract
The rapid expansion of electronic health record (EHR) systems has generated large volumes of unstructured clinical narratives that contain valuable information for disease identification, patient cohort discovery, and clinical decision support. Extracting structured knowledge from these free-text do...

---

### 19. [Attention Residuals](https://arxiv.org/abs/2603.15031)

**Authors**: Kimi Team, Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo Chen, Bohong Yin, Yutian Chen, Junjie Yan, Ming Wei, Y. Zhang, Fanqing Meng, Chao Hong, Xiaotong Xie, Shaowei Liu, Enzhe Lu, Yunpeng Tai, Yanru Chen, Xin Men, Haiqing Guo, Y. Charles, Haoyu Lu, Lin Sui, Jinguo Zhu, Zaida Zhou, Weiran He, Weixiao Huang, Xinran Xu, Yuzhi Wang, Guokun Lai, Yulun Du, Yuxin Wu, Zhilin Yang, Xinyu Zhou  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.15031v1  

#### Abstract
Residual connections with PreNorm are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer's contribution. We propose Attention Residuals (AttnRes), which...

---

### 20. [Linear Predictability of Attention Heads in Large Language Models](https://arxiv.org/abs/2603.13314)

**Authors**: Khalid Shaikh, Asmit Kumar Singh, Rebecca Christopher Dsouza, Shikhar Shiromani  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.13314v1  

#### Abstract
Large language model (LLM) inference is increasingly bottlenecked by the Key-Value (KV) cache, yet the fine-grained structure of attention-head activations remains poorly understood. We show that pretrained Transformers exhibit a pervasive inter-head linear structure: for a given token, the Query, K...

---

### 21. [Dataset Distillation Efficiently Encodes Low-Dimensional Representations from Gradient-Based Learning of Non-Linear Tasks](https://arxiv.org/abs/2603.14830)

**Authors**: Yuri Kinoshita, Naoki Nishikawa, Taro Toyoizumi  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.14830v1  

#### Abstract
Dataset distillation, a training-aware data compression technique, has recently attracted increasing attention as an effective tool for mitigating costs of optimization and data storage. However, progress remains largely empirical. Mechanisms underlying the extraction of task-relevant information fr...

---

### 22. [Joint Routing and Model Pruning for Decentralized Federated Learning in Bandwidth-Constrained Multi-Hop Wireless Networks](https://arxiv.org/abs/2603.15188)

**Authors**: Xiaoyu He, Weicai Li, Tiejun Lv, Xi Yu  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.15188v1  

#### Abstract
Decentralized federated learning (D-FL) enables privacy-preserving training without a central server, but multi-hop model exchanges and aggregation are often bottlenecked by communication resource constraints. To address this issue, we propose a joint routing-and-pruning framework that optimizes rou...

---

### 23. [PA-Net: Precipitation-Adaptive Mixture-of-Experts for Long-Tail Rainfall Nowcasting](https://arxiv.org/abs/2603.13818)

**Authors**: Xinyu Xiao, Sen Lei, Eryun Liu, Shiming Xiang, Hao Li, Cheng Yuan, Yuan Qi, Qizhao Jin  
**Category**: cs.AI  
**Published**: 2026-03-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.13818v1  

#### Abstract
Precipitation nowcasting is vital for flood warning, agricultural management, and emergency response, yet two bottlenecks persist: the prohibitive cost of modeling million-scale spatiotemporal tokens from multi-variate atmospheric fields, and the extreme long-tailed rainfall distribution where heavy...

---

### 24. [VTC-Bench: Evaluating Agentic Multimodal Models via Compositional Visual Tool Chaining](https://arxiv.org/abs/2603.15030)

**Authors**: Xuanyu Zhu, Yuhao Dong, Rundong Wang, Yang Shi, Zhipeng Wu, Yinlun Peng, YiFan Zhang, Yihang Lou, Yuanxing Zhang, Ziwei Liu, Yan Bai, Yuan Zhou  
**Category**: cs.AI  
**Published**: 2026-03-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.15030v1  

#### Abstract
Recent advancements extend Multimodal Large Language Models (MLLMs) beyond standard visual question answering to utilizing external tools for advanced visual tasks. Despite this progress, precisely executing and effectively composing diverse tools for complex tasks remain persistent bottleneck. Cons...

---

### 25. [PMIScore: An Unsupervised Approach to Quantify Dialogue Engagement](https://arxiv.org/abs/2603.13796)

**Authors**: Yongkang Guo, Zhihuan Huang, Yuqing Kong  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.13796v1  

#### Abstract
High dialogue engagement is a crucial indicator of an effective conversation. A reliable measure of engagement could help benchmark large language models, enhance the effectiveness of human-computer interactions, or improve personal communication skills. However, quantifying engagement is challengin...

---

### 26. [Towards Next-Generation LLM Training: From the Data-Centric Perspective](https://arxiv.org/abs/2603.14712)

**Authors**: Hao Liang, Zhengyang Zhao, Zhaoyang Han, Meiyi Qiang, Xiaochen Ma, Bohan Zeng, Qifeng Cai, Zhiyu Li, Linpeng Tang, Weinan E, Wentao Zhang  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.14712v1  

#### Abstract
Large language models (LLMs) have demonstrated remarkable performance across a wide range of tasks and domains, with data playing a central role in enabling these advances. Despite this success, the preparation and effective utilization of the massive datasets required for LLM training remain major ...

---

### 27. [Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks](https://arxiv.org/abs/2603.14864)

**Authors**: Zijian Yu, Kejun Xiao, Huaipeng Zhao, Tao Luo, Xiaoyi Zeng  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.14864v1  

#### Abstract
In e-commerce, LLM agents show promise for shopping tasks such as recommendations, budgeting, and bundle deals, where accurately capturing user preferences from long-term conversations is critical. However, two challenges hinder realizing this potential: (1) the absence of benchmarks for evaluating ...

---

### 28. [Covariance-Guided Resource Adaptive Learning for Efficient Edge Inference](https://arxiv.org/abs/2603.14577)

**Authors**: Ahmad N. L. Nabhaan, Zaki Sukma, Rakandhiya D. Rachmanto, Muhammad Husni Santriaji, Byungjin Cho, Arief Setyanto, In Kee Kim  
**Category**: cs.DC  
**Published**: 2026-03-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.14577v1  

#### Abstract
For deep learning inference on edge devices, hardware configurations achieving the same throughput can differ by 2$\times$ in power consumption, yet operators often struggle to find the efficient ones without exhaustive profiling. Existing approaches often rely on inefficient static presets or requi...

---

### 29. [SemRep: Generative Code Representation Learning with Code Transformations](https://arxiv.org/abs/2603.13640)

**Authors**: Weichen Li, Jiamin Song, Bogdan Alexandru Stoica, Arav Dhoot, Gabriel Ryan, Shengyu Fu, Kexin Pei  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.13640v1  

#### Abstract
Code transformation is a foundational capability in the software development process, where its effectiveness relies on constructing a high-quality code representation to characterize the input code semantics and guide the transformation. Existing approaches treat code transformation as an end-to-en...

---

### 30. [True 4-Bit Quantized Convolutional Neural Network Training on CPU: Achieving Full-Precision Parity](https://arxiv.org/abs/2603.13931)

**Authors**: Shivnath Tathe  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.13931v1  

#### Abstract
Low-precision neural network training has emerged as a promising direction for reducing computational costs and democratizing access to deep learning research. However, existing 4-bit quantization methods either rely on expensive GPU infrastructure or suffer from significant accuracy degradation. In...

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
