# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-17 06:47:36 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SVD Contextual Sparsity Predictors for Fast LLM Inference](https://arxiv.org/abs/2603.14110)

**Authors**: Georgii Serbin, Kirill Koshkin, Zhongao Sun, Anastasiya Bistrigova, C. C. Korikov  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2603.14110v1  

#### Abstract
Contextual sparsity is one of the approaches used to reduce computational complexity in the inference process of large language models (LLMs). Existing techniques for efficient LLM inference acceleration based on contextual sparsity with minimal accuracy degradation require training sparse pattern p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SVD Contextual Sparsity Predictors for Fast LLM Inference

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）在边缘设备上的推理面临高计算开销和内存带宽瓶颈。尽管 ReGLU-based 模型具有**动态稀疏性**（contextual sparsity），即每层中仅有少量神经元被激活，但如何高效、准确地预测这些激活模式以实现加速，仍是一个挑战。

现有基于学习的稀疏性预测器（如 Deja Vu, PowerInfer）需要额外训练，且对分布外输入敏感；而无需训练的方法（如 GRIFFIN, TDA）则精度较低或灵活性不足。

### 提出的新方法
本文提出了一种**无需训练**（training-free）、基于**截断感知奇异值分解**（truncation-aware SVD）的上下文稀疏性预测框架——**SVD-based Sparsity Predictors (SVDP)**，用于加速 ReGLU-based LLMs 的 FFN 推理。

#### 核心思想：
- 利用门控投影矩阵 $ W_{\text{gate}} $ 的低秩 SVD 近似作为轻量级预测器。
- 引入**校准偏置**（calibrated bias）补偿因低秩近似导致的输出分布偏移。
- 结合**数据白化技术**（data whitening）提升低秩近似质量。
- 在运行时采用**顺序执行流水线**（sequential pipeline），进一步剔除误报（false positives）。

### 相比现有方法的优势
| 特性 | SVDP | Deja Vu / PowerInfer | GRIFFIN / TDA |
|------|------|------------------------|----------------|
| 是否需训练 | ❌ 否（training-free） | ✅ 是 | ❌ 否 |
| 构建成本 | 极低（仅需少量token） | 高（需训练） | 低 |
| 准确性保障 | ✅ 理论误差上界 | ❓ 数据依赖 | ❌ 无理论保证 |
| 泛化能力 | ✅ 利用权重结构，鲁棒性强 | ⚠️ 易受OOD影响 | ✅ 统计驱动，较稳定 |
| 加速效果 | ✅ 高达1.8× E2E加速 | ✅ 高 | ⚠️ 中等 |

此外，SVDP 支持在 **CUDA 和 CANN** 平台高效部署，尤其在华为 Ascend NPU 上通过高级 PyTorch API 即可获得显著加速，无需手动优化底层指令。

---

## 2. 核心实验方法和设置

### 使用的数据集
用于构建预测器的**校准数据集**来自以下任务的 few-shot 样本（共约 20,000 tokens）：
- GSM8K（数学推理）
- ARC-E / ARC-C（科学常识推理）
- OpenBookQA
- QAsper
- CodeXGLUE

### 评估基准（Benchmark）
在多个开放源码评测集上进行性能评估，涵盖多种任务类型：
- **Reasoning**: GSM8K, ARC-E/C, BBH
- **Code Generation**: HumanEval, MBPP
- **Knowledge**: TriviaQA
- **Multilingual Understanding**: CMMLU（中文多任务理解）

所有任务均采用**自由生成**（free-form generation）方式评估，而非 perplexity，更贴近真实场景。

### 实验设置
- **硬件平台**：
  - GPU: NVIDIA RTX 3090（MSU 实验室独立完成）
  - NPU: Huawei Ascend 310P3
- **软件框架**：基于 **vLLM** 实现推理服务，使用半精度（half precision）运行，离线阶段使用双精度构建预测器。
- **模型**：三个开源的稀疏化 7B 模型：
  - ProSparse-LLaMA2-7B
  - SparseQwen2-7B
  - TurboSparse-Mistral-Instruct
- **评估指标**：
  - **End-to-End Latency**：包含 prefill 和 200 token 生成时间
  - **Speedup**：相对于 dense 模型的端到端加速比
  - **Accuracy Score**：各 benchmark 的平均得分（pass@1, acc 等）

### 基线方法对比
- **Dense Baseline**：原始密集模型
- **Trained Predictors**：
  - Deja Vu（learnable predictor）
  - PowerInfer（预训练 predictor）
- **Training-free Methods**：
  - GRIFFIN（基于 prompt 统计）
  - Naive SVD（无白化与偏置校准）

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 模型 | 方法 | 平均 Accuracy | E2E Speedup | 激活稀疏度 |
|------|------|---------------|-------------|------------|
| ProSparse-LLaMA2-7B | SVDP (s=0.5) | 38.78% (-0.14%) | **1.64×** | ~90% |
| TurboSparse-Mistral-Instruct | SVDP (s=0.7) | 53.44% (-0.86%) | **1.84×** | ~90% |
| SparseQwen2-7B | SVDP (s=0.7) | 64.49% (-1.35%) | **1.86×** | ~90% |

> ✅ **最高实现 1.8× 的端到端解码加速**，同时保持 <1% 的 benchmark 分数下降。

### 与基线方法对比结果（以 ProSparse-LLaMA2-7B 为例）

| 方法 | Predicted Sparsity | Average Score | E2E Speedup |
|------|--------------------|----------------|--------------|
| Dense | — | 38.92% | 1.00× |
| Deja Vu (r=1024) | 20% | 32.86% | 1.38× |
| PowerInfer (r=1024) | 80% | 37.93% | 1.63× |
| GRIFFIN (50%) | 50% | 32.81% | 1.46× |
| **SVDP (seq., r=256)** | **50%** | **38.78%** | **1.64×** |

- SVDP 在相同稀疏水平下**精度远超其他方法**。
- 尽管 PowerInfer 也达到相近加速，但其依赖大量训练数据和特定硬件调度。
- SVDP 在**更小 rank（256 vs 1024）** 下实现相当甚至更好的性能，说明其效率更高。

### 消融实验结果（Ablation Study）

在 `ProSparse-LLaMA2-7B` 上逐步添加组件的效果（Average Score）：

| 配置 | Average Score |
|------|----------------|
| Dense baseline | 38.92% |
| Naive SVD (parallel) | 34.08% |
| + Data Whitening | 37.10% |
| + Bias Calibration | **38.72%** |
| Sequential Pipeline + Full Stack | **38.78%** |

> 🔍 **关键发现**：
> - **Bias calibration 贡献最大**，弥补了低秩近似带来的系统性偏差。
> - **Data whitening 显著提升 ROC-AUC 分离能力**（见 Figure 5）。
> - **Sequential execution pipeline 可进一步提高实际稀疏度**，允许使用更轻量预测器。

---

## 4. 关键结论和发现

### 主要发现
1. **SVD-based predictor 具备强分离能力**：即使在极低秩（如 r=256）下，也能有效区分活跃与非活跃神经元（见 Figure 4, 7）。
2. **无需训练即可实现高性能稀疏推理**：利用 LLM 自身权重结构 + 输入统计信息，即可构建高精度预测器。
3. **理论可解释性强**：提供了预测误差的确定性上界分析（见 Appendix A），揭示了 SVD 截断误差与偏置项的关系。
4. **跨平台兼容性好**：
   - CUDA：通过自定义 kernel 实现细粒度控制
   - CANN（Ascend）：仅用高级 PyTorch API 即可达 **30× FFN 加速**（sparsity=95%）
5. **顺序执行优于并行执行**：通过先算 gate 再验证 up 投影，能动态过滤 false positives，提升最终稀疏度。

### 方法的局限性
- **依赖 ReLU 类激活函数**：仅适用于 ReGLU 或 dReLU 等能产生零值输出的结构。对于 SiLU/Swish 等平滑激活函数，直接应用会导致严重精度损失（见 Table 5，Qwen2-7B-Instruct 下降 3.5 pts）。
- **批处理受限**：batched processing 会降低稀疏性，难以在 prefill 阶段启用稀疏预测。
- **不支持 KV-cache 复用更新机制**：不同于 CLADA，无法在生成过程中动态调整历史 token 的稀疏模式。

### 未来工作方向
- 扩展至非 ReLU 架构：探索对中间激活进行变换（activation transformation）以适配 SVDP。
- 动态 rank 调整：根据不同层特性自适应选择 SVD rank。
- 支持 long-context 场景下的增量稀疏更新。
- 探索与其他压缩技术（如量化、LoRA）的联合优化。

---

> 📌 **总结一句话**：  
> 本文提出了一种**无需训练、理论可靠、跨平台高效的 SVD-based 稀疏预测器 SVDP**，在平均 **90% 激活稀疏度**下实现了高达 **1.8× 的端到端推理加速**，为 LLM 在边缘设备的部署提供了新的可行路径。

</details>

---

### 2. [MONET: Modeling and Optimization of neural NEtwork Training from Edge to Data Centers](https://arxiv.org/abs/2603.15002)

**Authors**: J\'er\'emy Morlier, Robin Geens, Stef Cuyckens, Arne Symons, Marian Verhelst, Vincent Gripon, Mathieu L\'eonardon  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2603.15002v1  

#### Abstract
While hardware-software co-design has significantly improved the efficiency of neural network inference, modeling the training phase remains a critical yet underexplored challenge. Training workloads impose distinct constraints, particularly regarding memory footprint and backpropagation complexity,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*MONET: Modeling and Optimization of neural NEtwork Training from Edge to Data Centers*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前大多数神经网络加速器设计和建模工具（如 Timeloop、Accelergy）聚焦于 **inference** 阶段，而 **training** 具有显著不同的计算和内存特性（如反向传播、梯度存储、optimizer states 和 activation checkpointing），导致为 inference 优化的硬件在 training 场景下表现不佳。  
本文指出：**inference-only 的建模无法准确反映 training 的能效与延迟行为**，因此需要专门针对训练阶段进行硬件-软件协同设计。

### ✅ 提出的新方法与创新点
作者提出了 **MONET** —— 一个支持端到端神经网络训练建模与优化的框架，其核心创新如下：

- **扩展 Stream 框架以支持 training 工作流**：
  - 在原有支持 layer fusion 的推理建模基础上，新增对 **forward pass、backward pass 和 optimizer 更新** 的完整 ONNX 图建模。
  - 引入专用的 ONNX transformation passes，将复合梯度算子（如 ConvGrad）分解为细粒度组件，便于精确调度与融合分析。

- **支持 activation checkpointing 的建模与优化**：
  - 将 activation checkpointing 建模为可配置的 ONNX 子图替换机制，实现对“存储 vs 重计算”权衡的系统性探索。

- **提出约束优化方法解决 layer-fusion 问题**：
  - 针对训练图中节点数量多、依赖复杂的特点，设计基于 **constraint programming** 的 layer-fusion 分区算法，考虑内存、tiling 兼容性和操作类型限制。

- **采用遗传算法求解非线性的 activation checkpointing 问题**：
  - 发现传统 MILP 模型因忽略 layer fusion 与重计算之间的耦合效应而不适用，转而使用 **NSGA-II 多目标遗传算法** 寻找 energy-latency-memory 的 Pareto 最优解。

### ✅ 相比现有方法的优势

| 特性 | MONET | 其他主流框架（如 Timeloop, Dace-AD, NVArchSim） |
|------|-------|---------------------------------------------|
| 支持 Training | ✅ 完整支持 | ❌ 不支持 或 仅部分支持 |
| 支持 Layer Fusion | ✅ 细粒度融合建模 | ❌ 通常只到 operator level |
| 支持异构架构（HDA） | ✅ 支持 | ⚠️ 多数仅限 GPU 或同构架构 |
| 支持 activation checkpointing | ✅ 内建建模与优化 | ❌ 无显式支持 |
| 多目标优化能力 | ✅ 支持 energy/latency/memory 权衡 | ⚠️ 多为单目标 |

> 📌 **优势总结**：MONET 是首个同时支持 **fine-grained layer fusion + full training workload modeling + heterogeneous dataflow accelerators (HDA)** 的开源建模框架，填补了 training-aware 硬件设计空间探索的空白。

---

## 2. 核心实验方法和设置

### 🔧 实验模型与硬件平台

| 类别 | 设置 |
|------|------|
| **神经网络模型** | - **ResNet-18**（图像分类，输入尺寸 3×32×32）<br>- **Small GPT-2**（NLP 任务，Transformer 架构） |
| **硬件平台** | - **Edge TPU**：用于 ResNet-18，模拟 4×4 PE 数组，weight-stationary 数据流<br>- **FuseMax**：用于 GPT-2，output-stationary MAC 阵列 + SIMD 向量单元 |
| **部署策略** | 使用 pipeline parallelism 和 tensor parallelism 进行跨核映射；支持 layer-fused scheduling |

### 📊 评估指标

- **Latency**：训练一次迭代所需的周期数（cycles）
- **Energy**：总能耗（单位：pJ），由 Accelergy 提供组件级能耗模型
- **Memory Consumption**：峰值激活内存占用（FP16 存储假设）
- **Pareto Front**：展示 energy-latency、latency-memory 等多维权衡关系

### 🔍 基线方法对比

| 基线 | 描述 |
|------|------|
| **Inference-only 设计** | 使用 Stream 对 inference 进行优化后的配置，直接用于 training 对比 |
| **Manual Layer Fusion** | 手动设计的 layer fusion 配置（来自原始 Stream） |
| **No Checkpointing** | 保存所有 activations 的 baseline |
| **Linear MILP Checkpointing** | 传统线性规划方法（作为对比，证明其局限性） |

---

## 3. 主要实验结果和性能指标

### 📈 ResNet-18 实验结果（Edge TPU）

#### ▶ Layer Fusion 优化效果（图10）
- **相比 manual fusion**：
  - **Latency 降低最多达 15%**
  - **Energy 下降约 12%**
- **最优 subgraph 长度为 6**，超过后收益递减
- 即使限制最大长度为 4，仍优于手动设计 → 表明自动化 fusion 更高效

#### ▶ Activation Checkpointing 效果（图11 & 图12）
- **非线性现象明显**：两个 activation 一起重计算的成本 ≠ 各自单独重计算之和
  - 原因：重计算改善了数据局部性，并促进了更多 layer fusion（如 Op2 与 Op3 可合并）
- **遗传算法生成 Pareto 前沿**：
  - 在 batch size=1, image size=224×224 下：
    - **节省高达 13MB 内存**
    - 代价仅为 **+4% latency 和 +4% energy**
  - 某些配置甚至 **同时降低 latency 和 memory**

> 💡 注：随着 batch size 或 sequence length 增大，checkpointing 的收益会更显著。

### 📈 Small GPT-2 实验结果（FuseMax）

#### ▶ 能效权衡分析（图9）
- **Inference vs Training 的 energy-latency 分布差异显著**
  - Training 更受 memory bandwidth 影响（颜色编码显示 buffer bandwidth 的影响）
  - 高带宽配置在 training 中更易进入 Pareto 前沿
- 结构同质性强 → 性能分布更集中，对硬件变化不敏感

#### ▶ 设计启示
- 对 LLM 类 workload，**buffer bandwidth 和 off-chip 通信效率** 成为 training 的关键瓶颈
- 推理友好的设计不一定适合训练，需重新权衡资源分配

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Training 与 Inference 的硬件偏好存在根本差异**  
   - 图1 显示：同一组硬件配置在 inference 和 training 下的 energy-latency 分布完全不同。
   - 为 inference 优化的架构不能直接迁移至 training 场景。

2. **Layer Fusion 在 Training 中更具潜力但也更复杂**  
   - optimizer 中的 element-wise 操作（如 Adam）非常适合与前层梯度融合，减少中间激活存储。
   - 自动化 fusion 算法（基于 constraint optimization）优于人工设计。

3. **Activation Checkpointing 具有强非线性效应**  
   - 重计算不仅带来额外 FLOPs，还会影响 layer fusion 机会和数据局部性。
   - 传统的线性 MILP 模型无法捕捉这些交互，必须采用非线性或多目标搜索方法（如 GA）。

4. **Pareto 权衡揭示新操作点**  
   - 通过遗传算法可在 memory、latency、energy 之间找到新的平衡点，例如“小幅增加开销换取大幅内存节省”。

### ⚠️ 局限性

- 当前框架尚未支持现代 GPU 架构（如 NVIDIA H100）的完整建模，虽可通过抽象逼近，但精度有限。
- 遗传算法虽然有效，但在超大规模模型上可能收敛较慢，缺乏理论最优保证。
- 缺少对分布式训练（data/model/tensor parallelism 联合优化）的全面建模。

### 🔮 未来工作方向

1. **扩展至 GPU 和大规模集群建模**  
   - 类似 LLMCompass 的方式，在 Stream/MONET 中集成 GPU 微架构抽象。

2. **开发更高效的 mapping 与 fusion 算法**  
   - 探索强化学习或图神经网络（GNN）驱动的自动 scheduler。

3. **联合优化 hardware design + algorithm choice**  
   - 将 activation checkpointing 策略、batch size、sequence length 纳入联合搜索空间。

4. **支持更多 optimizer 和 low-rank training 技术**  
   - 如 Galore（低秩梯度投影）、8-bit Adam 等内存优化技术的建模。

---

## 总结

📌 **MONET 的核心价值在于：首次实现了面向 training 的、支持 layer fusion 与 activation checkpointing 的异构加速器建模与优化闭环**。它不仅揭示了 training-aware 设计的重要性，也为未来的高效深度学习系统提供了可复现、可扩展的探索平台。

</details>

---

### 3. [LightningRL: Breaking the Accuracy-Parallelism Trade-off of Block-wise dLLMs via Reinforcement Learning](https://arxiv.org/abs/2603.13319)

**Authors**: Yanzhe Hu, Yijie Jin, Pengfei Liu, Kai Yu, Zhijie Deng  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 11.5  
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
现有的 **block-wise dLLMs**（如 SDAR、Fast-dLLM）在并行解码时面临严重的 **accuracy-parallelism trade-off**：  
- 为了提升推理速度（提高 TPF，Tokens Per Forward），模型需要一次性预测更多 token；
- 但这往往导致生成质量下降，尤其是在数学推理和代码生成等复杂任务上。

尽管已有训练后采样策略（如 confidence-driven decoding）或蒸馏方法试图缓解该问题，但通常以牺牲准确性为代价，难以同时兼顾 **高并行度** 和 **高准确率**。

---

### 🚀 提出的新方法：LightningRL
作者提出一种基于 **Reinforcement Learning (RL)** 的后训练框架 —— **LightningRL**，旨在直接优化预训练 dLLM 的 **speed-quality frontier**（速度-质量前沿）。

其核心思想是：
> 不要求模型在所有路径上都激进地并行解码，而是引导模型学会找到那些“既高度可并行化又能输出正确结果”的特定采样轨迹。

为此，LightningRL 在 **Group Relative Policy Optimization (GRPO)** 框架基础上进行了三项关键改进：

#### （1）Per-reward Decoupled Normalization（去耦奖励归一化）
- 传统 GRPO 对多目标奖励（如 accuracy + TPF）进行统一归一化，容易导致量级较大的奖励主导优化过程。
- LightningRL 将不同奖励分量（accuracy 和 TPF）分别独立归一化后再聚合，避免信号失衡，提升多目标优化稳定性。

#### （2）Token-level NLL Regularization（词元级负对数似然正则）
- 引入在**正确轨迹上的 token-level NLL 损失**作为锚定项（anchoring），防止策略漂移（policy drift）。
- 这种自模仿学习机制增强了语言连贯性和语义一致性，尤其在稀疏奖励场景下稳定训练。

#### （3）Dynamic Sampling with TPF-aware Filtering（TPF感知动态采样）
- 在每个训练批次中，仅保留满足以下条件的 prompt 组：
  - 至少有一个采样轨迹是正确的；
  - 不同轨迹间的 TPF 差异足够大（即存在明显的快慢路径差异）。
- 此举确保每组 rollout 都能提供有效的相对优势信号，提升样本效率和梯度密度。

---

### 🔍 相比现有方法的优势
| 方面 | LightningRL | 传统方法（如 Fast-dLLM, d3LLM） |
|------|-------------|-------------------------------|
| 优化方式 | 显式联合优化 accuracy 与 parallelism | 多为训练后加速策略或单目标优化 |
| 并行能力 | 主动学习“快速且正确”的路径 | 被动依赖固定解码规则 |
| 准确性保持 | 利用 NLL 锚定 + 去耦归一化维持高精度 | 高 TPF 下 accuracy 显著下降 |
| 训练稳定性 | 动态过滤 + 分离奖励 → 更鲁棒收敛 | 容易出现 reward collapse |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **数学推理任务**：
  - `GSM8K`（小学数学应用题）
  - `MATH500`（高等数学问题）
- **代码生成任务**：
  - `MBPP`（面向编程初学者的任务）
  - `HumanEval`（函数级 Python 编程挑战）

训练数据来自：
- MATH 和 GSM8K 的训练集
- PrimeIntellect（用于代码任务）

---

### ⚙️ 实验设置
- **基础模型**：基于 `SDAR-8B-b32`（8B 参数，block size=32）进行 post-training。
- **RL 设置**：
  - 每个 prompt 采样 `G=32` 条轨迹
  - 批大小：128 个任务
  - 温度 = 1.0，confidence threshold = 0.9
- **硬件**：使用 H200 GPU 多节点分布式训练（DeepSpeed ZeRO-1）

---

### 📊 评估指标
| 指标 | 含义 |
|------|------|
| **Acc (%)** | 最终任务准确率（通过 verifier 判断） |
| **TPF (Tokens Per Forward)** | 平均每次前向传播生成的 token 数，衡量并行程度 |
| **AUP (Accuracy Under Parallelism)** | 综合指标：Acc × TPF，反映速度与质量的整体权衡 |
| **TPS (Tokens Per Second)** | 实际推理吞吐量（wall-clock time 测量） |

---

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **dLLM 加速方法** | Dream, Fast-dLLM, dParallel-Dream/LLaDA, d3LLM |
| **AR 模型 + 推测解码** | EAGLE-3 (Llama-3.1-8B), Qwen2.5-7B-Instruct |
| **其他 RL 框架变体** | TraceRL, GRPO(traj) |
| **原始模型** | SDAR-8B-b32（未微调） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 2）

| Model | GSM8K Acc | GSM8K TPF | GSM8K AUP |
|-------|-----------|----------|----------|
| SDAR-8B-b32 | 88.9% | 2.85 | 252.5 |
| EAGLE-3 | 76.6% | 5.12 | 319.0 |
| d3LLM-LLaDA | 73.1% | 9.11 | 637.7 |
| **LightningRL-8B-b32** | **90.3%** | **5.58** | **492.4** |

| Model | MBPP Acc | MBPP TPF | MBPP AUP |
|--------|---------|--------|--------|
| SDAR-8B-b32 | 58.0% | 2.44 | 81.1 |
| d3LLM-LLaDA | 40.6% | 4.21 | 88.4 |
| **LightningRL-8B-b32** | **58.3%** | **11.10** | **641.6** |

> ✅ **LightningRL 在 MBPP 上达到最高 AUP（641.6）和 TPF（11.10）**

| Model | HumanEval Acc | TPF | AUP |
|--------|----------------|-----|-----|
| SDAR-8B-b32 | 73.5% | 2.39 | 123.8 |
| **LightningRL-8B-b32** | 72.6% | **6.30** | **450.1** |

> 💡 尽管 HumanEval 准确率略有下降，但 TPF 提升近 3 倍，AUP 提升超过 3 倍！

---

### 📉 与基线方法的对比结果
- **平均表现**：
  - LightningRL 达到 **平均 AUP = 497.9**，**平均 TPF = 7.32**
  - 显著优于：
    - EAGLE-3：AUP=276.1，TPF=5.63
    - d3LLM-family：虽 TPF 更高，但 accuracy 太低
    - SDAR baseline：AUP=189.2，TPF=3.12
- **推理速度实测（H100）**：
  - LightningRL 实现 **336.03 TPS**，远超 SDAR（105.55 TPS）和其他基线。

---

### 🔬 消融实验结果（见 Table 3 & 4）

#### 表 3：组件消融（GSM8K）

| 模型配置 | Acc (%) | TPF | AUP |
|--------|--------|-----|-----|
| 完整 LightningRL | **90.3** | **5.58** | **492.4** |
| w/o NLL loss | 80.7 | 5.03 | 385.7 |
| w/o Decoupled Norm | 85.3 | 4.96 | 416.5 |
| w/o TPF-aware Filtering | 87.2 | 5.27 | 454.5 |

> ❗ 所有三个模块均不可或缺，尤其是 **token-level NLL loss** 对 accuracy 影响最大。

#### 表 4：损失归约策略比较

| Reduction Strategy | Acc (%) | TPF | AUP |
|--------------------|--------|-----|-----|
| **Seq-Tok-Tok**（推荐） | **90.3** | **5.58** | **492.4** |
| Seq-Seq-Seq | 87.5 | 5.42 | 467.1 |
| Tok-Tok-Tok | 80.0 | 3.91 | 306.5 |

> ⚠️ 全局 token-level reduction 会导致严重性能退化，因其忽略了序列长度偏差问题。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **成功打破 accuracy-parallelism trade-off**：
   - LightningRL 能在显著提升 TPF 的同时维持甚至略微提升 accuracy。
   - 在多个 benchmark 上实现了当前最优的 AUP 指标。

2. **RL 是优化 dLLM 解码路径的有效手段**：
   - 通过多目标 RL 显式建模“快速且正确”的偏好，比传统 heuristic 或 distillation 方法更灵活高效。

3. **训练稳定性至关重要**：
   - 去耦归一化 + 动态采样有效缓解了 reward collapse 和梯度饥饿问题。
   - 引入 value model 反而会因状态跳跃剧烈而导致 critic 不稳定（见 Appendix A）。

4. **可扩展性强**：
   - 在不同模型规模（1.7B → 8B）和 block size（4 → 32）下均表现出良好泛化性（见 Table 6）。

---

### ⚠️ 方法的局限性
- **依赖高质量 verifier**：适用于有明确答案的任务（如数学、编程），但在开放生成任务中难以直接应用。
- **计算开销较高**：RL 训练需要大量 rollout 和 GPU 资源，不适合轻量部署场景。
- **目前仅适配 SDAR 架构**：尚未验证在其他 block-wise dLLM 上的迁移效果。

---

### 🔮 未来工作方向
1. **扩展至更大上下文和多模态任务**
2. **探索更高效的 RL 采样策略**（如重要性重采样）
3. **将 parallelism-aware RL 应用于 AR 模型的 drafters 设计**
4. **研究 offline RL 版本以降低训练成本**

---

> 🔗 项目代码已开源：[https://github.com/SJTU-DENG-Lab/LightningRL](https://github.com/SJTU-DENG-Lab/LightningRL)

</details>

---

### 4. [ExPosST: Explicit Positioning with Adaptive Masking for LLM-Based Simultaneous Machine Translation](https://arxiv.org/abs/2603.14903)

**Authors**: Yuzhe Shang, Pengzhi Gao, Yazheng Yang, Jiayao Ma, Wei Liu, Jian Luan, Jingsong Su  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.14903v1  

#### Abstract
Large language models (LLMs) have recently demonstrated promising performance in simultaneous machine translation (SimulMT). However, applying decoder-only LLMs to SimulMT introduces a positional mismatch, which leads to a dilemma between decoding efficiency and positional consistency. Existing appr...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ExPosST: Explicit Positioning with Adaptive Masking for LLM-Based Simultaneous Machine Translation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在基于 **decoder-only LLM** 的 **Simultaneous Machine Translation (SimulMT)** 中，存在一个核心挑战：**位置不一致（positional mismatch）**。

- 在 SimulMT 场景中，源语言句子是流式输入的，新的 `source token` 被动态插入上下文中。
- 这会导致已生成的 `target token` 的 **position index 发生偏移**，从而破坏 **Key-Value (KV) Cache** 的有效性。
- 传统做法要么重新计算 KV Cache（低效），要么引入复杂的提示工程（如对话格式），导致模型注意力分散或兼容性差。

该问题造成了 **推理效率** 与 **位置一致性** 之间的根本性矛盾。

---

### 🚀 提出的新方法：ExPosST

作者提出 **ExPosST**（Explicit Position Allocation for Simultaneous Translation），一种通用且高效的框架，通过 **显式的位置分配机制** 解决上述矛盾。

#### 核心思想：
- **预分配位置槽（Pre-allocated Position Slots）**：为潜在的源 token 预留固定长度的位置区间（称为 `slot`）。
- 所有目标 token 的生成从这些预留槽之后开始，其 **position index 在整个 READ/WRITE 流程中保持不变**。
- 因此，KV Cache 可以被直接复用而无需重计算（zero-recomputation），实现高效推理。

#### 关键组件：
1. **Pre-allocated Positions Inference**  
   - 推理时采用固定位置槽策略，确保 target token 的位置索引不变。
   - 支持动态扩展：当当前 slot 不足时，在输出后立即分配新 slot。

2. **Policy-Consistent Fine-tuning**  
   - 训练阶段模拟推理时的 slot 结构，对源句按 slot 分段。
   - 引入 **policy-consistent attention masking**，使训练时可见的 source token 与实际流式解码策略（如 wait-k 或 read-n）严格对齐。

---

### 🔍 相比现有方法的优势

| 方法 | 局限性 | ExPosST 如何改进 |
|------|--------|------------------|
| **SimulMask** | 依赖 ALiBi positional encoding，无法兼容主流 RoPE-based LLM（如 Llama/Qwen） | 兼容 RoPE 和 ALiBi，适用范围更广 |
| **GPE (Group Position Encoding)** | 使用非整数或重叠位置，偏离标准 LLM 预训练范式 | 使用标准整数位置编码，无额外修改 |
| **Conversational SimulMT** | 依赖频繁的角色切换（user/assistant），造成“prompt bloating”，增加计算开销 | 无需角色切换，减少冗余标记，降低计算成本 |

✅ **综合优势**：
- 实现 **高推理效率**（KV Cache 零重计算）
- 保证 **严格的位置一致性**
- 具备 **强模型兼容性**（支持 RoPE-based 和 ALiBi-based LLM）
- 不依赖特殊 prompt 格式或内部结构改造

---

## 2. 核心实验方法和设置

### 📚 数据集
- 使用 **IWSLT 2017** 多语言数据集进行评估：
  - 英语 → 法语 (**En-Fr**)
  - 英语 → 德语 (**En-De**)
  - 英语 → 荷兰语 (**En-Nl**)
  - 英语 → 意大利语 (**En-It**)
  - 英语 → 罗马尼亚语 (**En-Ro**)

---

### ⚙️ 实验设置

| 组件 | 设置 |
|------|------|
| **Backbone Models** | `Llama-3.1-8B-Instruct`, `Qwen2.5-7B-Instruct`, `falcon-rw-1b`（用于 ALiBi 对比） |
| **Fine-tuning 方法** | 使用 **LoRA**（Low-Rank Adaptation）进行参数高效微调 |
| **Slot Length (`L_slot`)** | 默认设为 16（经敏感性分析确定最优值） |
| **Decoding Policies** | 支持两种主流策略：<br>• `wait-k`（k ∈ {1,3,5,7}）<br>• `read-n & incremental decoding`（n ∈ {3,5,7,9,11,13}） |
| **Training Framework** | 基于 `Simul-LLM` 框架实现 fine-tuning |
| **Evaluation Toolkit** | 使用 `SimulEval` 工具包进行在线评估 |

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **BLEU Score** | 衡量翻译质量（使用 SacreBLEU 计算去分词 BLEU） |
| **LAAL (Length-Adaptive Average Lagging)** | 衡量延迟，数值越小表示响应越快 |
| **COMET Score** | 补充评估语义保真度（见 Appendix B） |
| **GFLOPs** | 推理过程中的累计浮点运算量，衡量计算效率 |

---

### 🆚 基线方法对比

| 基线方法 | 特点 |
|---------|------|
| **GPE** | 基于分组位置编码，避免 cache 重算，但位置设计非常规 |
| **Conversational SimulMT** | 将 SimulMT 视为对话任务，通过 user/assistant 角色追加 token |
| **Offline** | 全句翻译模型，作为性能上限参考 |
| **SimulMask** | 专为 ALiBi 设计的方法，用于跨 positional encoding 的兼容性测试 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（基于 Llama-3.1-8B-Instruct）

#### ✅ BLEU-LAAL 权衡表现（图 4）
- 在所有五个语言对上，**ExPosST 显著优于基线**：
  - 在 `wait-k` 策略下，相比 GPE 平均提升 **2–3 BLEU points**
  - 在 `read-n` 策略下，相比 Conversational SimulMT 提升 **>1 BLEU point**
- 曲线更靠近左上角，表明在相同延迟下质量更高，或在相同质量下延迟更低。
- 随着 `k` 增大，性能逐渐逼近 **offline 模型**，说明其潜力巨大。

#### ✅ 跨模型泛化能力（图 5）
- 在 `Qwen2.5-7B-Instruct` 上同样取得 **SOTA 性能**，验证了框架的模型兼容性。
- 在 `falcon-rw-1b`（ALiBi-based）上的实验显示，ExPosST 与 SimulMask 性能相当（图 6），证明其可适配不同 positional encoding 方案。

#### ✅ 计算效率（图 8）
- **ExPosST 的推理成本最低**：
  - 显著低于 Prefix Finetuning（因无需重计算）
  - 低于 Conversational SimulMT（因无 prompt bloating）
- 中位 GFLOPs 下降明显，尤其在 wait-3 和 read-5 策略下。

#### ✅ 消融实验（Ablation Study，图 7）
- 移除任一组件都会导致性能显著下降：
  - **w/o Masking**：未模拟流式可见性，模型无法学习增量生成行为。
  - **w/o Slot**：位置偏移导致 KV Cache 失效，影响推理效率与准确性。
- 结果证明两个组件缺一不可。

#### ✅ Slot 长度敏感性分析（图 3 & 表 1）
- 最优 `L_slot = 16`：
  - 更小 → 频繁 slot 切换，增加 special token 开销
  - 更大 → 过多 padding，浪费计算资源
- **训练与推理 slot 长度需一致**才能达到最佳性能（表 1）：
  - 若推理时 `L_slot < 16`，BLEU 明显下降
  - 若 `L_slot > 16`，性能相对稳定，但略有波动

#### ✅ COMET 评分补充结果（图 9）
- 在语义层面也全面领先：
  - 即使在低 LAAL（高实时性）区域仍保持高 COMET 分数
  - 缩小了 simultaneous 与 offline 翻译之间的语义差距

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **显式位置分配是解决 SimulMT 中 KV Cache 冲突的有效途径**  
   - 通过预分配位置槽，彻底消除因 token 插入引起的位置偏移问题。

2. **ExPosST 实现了效率、一致性与兼容性的统一**  
   - 支持主流 RoPE-based LLM（如 Llama/Qwen）
   - 无需修改模型架构或 positional encoding
   - 推理速度快，计算成本低

3. **Policy-Consistent Fine-tuning 至关重要**  
   - 训练与推理的行为必须对齐，否则会损害性能

4. **框架具有良好的泛化性和鲁棒性**  
   - 跨语言、跨模型、跨 decoding policy 均表现优异

---

### ⚠️ 方法的局限性

1. **需要预先设定 slot 长度 `L_slot`**
   - 虽然可通过经验选择（如 16），但仍是一个超参数
   - 极端长句可能需要多次 slot 扩展，带来轻微管理开销

2. **训练与推理配置需匹配**
   - 若训练与推理使用的 `L_slot` 不一致，可能导致性能下降（尤其是推理 slot 更小时）

3. **目前依赖 LoRA 微调**
   - 虽然参数高效，但在某些场景下全量微调可能进一步提升性能

---

### 🔮 未来工作方向

1. **自适应 slot 长度机制**
   - 动态预测每一步所需的 slot 容量，减少 padding 或切换开销

2. **扩展至语音 SimulST（Simultaneous Speech Translation）**
   - 结合 ASR 输出流，构建端到端流式翻译系统

3. **探索 zero-shot 或 few-shot SimulMT 能力**
   - 当前依赖 fine-tuning，未来可研究如何让 LLM 在不训练的情况下原生支持 ExPosST 范式

4. **集成到实际应用系统**
   - 如会议同传、直播字幕等低延迟场景，验证真实环境下的稳定性与用户体验

---

## 总结

> **ExPosST 是首个通过显式位置分配解决 LLM-based SimulMT 中 KV Cache 重计算难题的通用框架**。它在不牺牲翻译质量的前提下，实现了高效的零重计算推理，并展现出卓越的模型兼容性与跨任务泛化能力，为大规模部署实时机器翻译系统提供了坚实的技术基础。

</details>

---

### 5. [A Multi-Scale Graph Learning Framework with Temporal Consistency Constraints for Financial Fraud Detection in Transaction Networks under Non-Stationary Conditions](https://arxiv.org/abs/2603.14592)

**Authors**: Yiming Lei, Qiannan Shen, Junhao Song  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.14592v1  

#### Abstract
Financial fraud detection in transaction networks involves modeling sparse anomalies, dynamic patterns, and severe class imbalance in the presence of temporal drift in the data. In real-world transaction systems, a suspicious transaction is rarely isolated: rather, legitimate and suspicious transact...

---

### 6. [Supervised Fine-Tuning versus Reinforcement Learning: A Study of Post-Training Methods for Large Language Models](https://arxiv.org/abs/2603.13985)

**Authors**: Haitao Jiang, Wenbo Zhang, Jiarui Yao, Hengrui Cai, Sheng Wang, Rui Song  
**Category**: cs.AI  
**Published**: 2026-03-17  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.13985v1  

#### Abstract
Pre-trained Large Language Model (LLM) exhibits broad capabilities, yet, for specific tasks or domains their attainment of higher accuracy and more reliable reasoning generally depends on post-training through Supervised Fine-Tuning (SFT) or Reinforcement Learning (RL). Although often treated as dis...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Supervised Fine-Tuning versus Reinforcement Learning: A Study of Post-Training Methods for Large Language Models*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
本文系统性地解决了当前大型语言模型（LLM）后训练领域的一个核心问题：**Supervised Fine-Tuning (SFT)** 和 **Reinforcement Learning (RL)** 这两种主流后训练范式之间的关系模糊、研究割裂的问题。

尽管 SFT 和 RL 被广泛用于提升 LLM 在特定任务上的表现，但大多数研究将它们视为独立甚至对立的方法，缺乏一个统一的理论框架来理解其内在联系、互补优势以及如何有效结合。这导致了方法设计的碎片化和潜在性能提升的错失。

### 提出了什么新方法或新思路
本文并非提出单一的新算法，而是提出了一个**全面且统一的视角**，对 SFT 和 RL 进行整合分析。其核心创新点包括：

1.  **统一的理论框架**：首次明确指出，在特定形式下，**SFT 可以被视为 RL 的一个特例**。具体而言，SFT 中最大化专家响应概率的目标，等价于在 RL 框架下使用一个基于“是否完全匹配专家输出”的指示函数作为奖励信号。
2.  **互补性与集成路径的系统分类**：构建了一个清晰的分类体系，将 SFT 与 RL 的结合方式分为三类：
    *   **Using SFT to Enhance RL**：利用 SFT 的离线示范数据来引导 RL 的在线探索，例如通过前缀采样（prefix sampling）、分支回放（branched rollouts）等方式稳定训练。
    *   **Using RL to Enhance SFT**：借鉴 RL 的思想改进 SFT，例如通过重要性重加权（importance weighting）来修正 SFT 中隐含的奖励偏差，从而提升泛化能力。
    *   **Hybrid Training**：将 SFT 和 RL 的目标函数直接融合进行联合优化，如交替训练、动态加权或在同一训练流程中同时应用两种损失。
3.  **趋势洞察**：通过对 2023-2025 年间大量应用研究的分析，揭示了后训练领域的三大趋势：**任务领域的快速扩展、混合（SFT+RL）训练范式的迅速普及、以及从依赖 API 标注向使用开源权重模型生成数据的转变**。

### 相比现有方法的优势
- **综合性强**：这是首个系统性地比较、连接并统一 SFT 与 RL 的综述性研究，填补了该领域的空白。
- **理论深刻**：提出的“SFT 是 RL 的特例”这一观点，为理解两种方法的本质提供了深刻的理论洞见，使得一种方法下的技巧可以启发另一种方法的改进。
- **实践指导意义**：通过梳理和分类各种集成策略，为研究人员和工程师选择和设计后训练流程提供了清晰的路线图和决策依据。

---

## 2. 核心实验方法和设置

需要特别说明的是，本文是一篇**综述性研究（survey/study）**，而非提出单一新模型的实证论文。因此，它本身没有传统意义上的“实验”，而是对已有研究进行了大规模的**系统性分析和趋势总结**。

### 数据集
本文的“数据集”是**学术文献本身**。作者通过以下方式收集和分析数据：
- **文献来源**：主要从 arXiv 预印本平台获取 2023 年 1 月至 2025 年 6 月期间发表的计算机科学领域相关论文。
- **领域划分**：根据论文中使用的基准（benchmark）数据集，将其归类到四个主要应用领域：
  - **General QA**: `hotpotqa`, `strategyqa`, `triviaqa` 等。
  - **Mathematical**: `gsm8k`, `asdiv`, `svamp`, `aime` 等。
  - **Agentic**: `webshop`, `webarena`, `alfworld`, `scienceworld` 等。
  - **Code-based**: `swe-bench`, `humaneval`, `livecodebench`, `bird` 等。

### 实验设置和评估指标
这里的“实验”指的是对文献的**量化分析过程**：
1.  **文献检索**：使用上述领域相关的数据集名称作为关键词，在 arXiv 上进行搜索。
2.  **文献筛选**：采用“基准导向的论文搜索”（benchmark-oriented paper search）策略。一篇论文若在其文本中至少提及某个领域相关数据集 **5 次以上**，则被计入该领域。
3.  **趋势分析**：统计每个领域每年的论文数量，并分析训练方法（SFT, RL, Both）和模型类型（API, Open-weight, Benchmark, Human/Web）的占比变化。
4.  **评估指标**：主要的“性能指标”是**论文发表数量的增长率**和**不同方法/技术路线的采用比例**。

### 基线方法对比
本文不直接对比模型性能，而是将不同的**后训练范式**本身作为对比对象：
- **SFT**：仅使用监督微调。
- **RL**：仅使用强化学习（如 PPO, GRPO, DPO）。
- **Both (SFT+RL)**：混合使用 SFT 和 RL 的方法。
- **其他**：如 Prompt Optimization 等。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
通过对文献的量化分析，得出了以下关键数据（基于表5和附录B.4的趋势分析）：

| 指标 | 2023年 | 2024年 | 2025年 (预测) | 增长率 (23->24) | 增长率 (24->25) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **General QA 论文数** | 292 | 652 | 983 | +123% | +51% |
| **Mathematical 论文数** | 492 | 1,098 | 2,399 | +123% | +118% |
| **Agentic 论文数** | 100 | 174 | 261 | +74% | +50% |
| **Code-based 论文数** | 115 | 428 | 786 | +272% | +84% |

| 指标 | 2023年 | 2024年 | 2025年 (预测) |
| :--- | :--- | :--- | :--- |
| **训练方法 - SFT** | 73.3% | 19.1% | 17.8% |
| **训练方法 - RL** | 6.7% | 7.1% | 11.8% |
| **训练方法 - Both (SFT+RL)** | 20.0% | 73.8% | 70.6% |
| **模型类型 - API-based** | 32.2% | 19.9% | 11.1% |
| **模型类型 - Open-weight** | 12.2% | 17.5% | 25.0% |

### 与基线方法的对比结果
- **SFT vs. RL vs. Both**：纯 SFT 的主导地位正在迅速瓦解。**混合方法 (Both)** 已成为绝对主流，从 2023 年的 20% 急剧增长到 2024 年的 73.8%，表明社区已普遍认识到结合两者的重要性。
- **API vs. Open-weight**：对闭源 API 模型的依赖显著下降，而使用开源权重模型（open-weight model）的比例持续上升，反映了研究向更开放、可复现的方向发展。

### 消融实验结果
本文未进行消融实验。其分析是基于对已有研究的观察和归纳。

---

## 4. 关键结论和发现

### 论文的主要发现
1.  **SFT 与 RL 本质相连**：SFT 可以被形式化地看作是一种特殊的 RL，其中奖励信号是“是否完美复制专家演示”。这一发现为两种方法的融合提供了坚实的理论基础。
2.  **混合范式是未来**：**SFT 和 RL 的结合（hybrid SFT-RL pipelines）已成为后训练领域的主导范式**。纯粹的 SFT 或 RL 正在被更复杂的多阶段、集成化流程所取代。
3.  **开放与标准化是趋势**：研究社区正快速从依赖昂贵且不可控的 API 标注转向使用开源模型生成的数据和标准化的公开基准，推动了研究的民主化和可复现性。
4.  **互补优势明显**：SFT 提供稳定、高效的初始学习，适合注入先验知识；RL 则擅长通过探索和奖励信号来提升泛化能力和处理复杂、长程的任务。

### 方法的局限性
- **综述性质**：本文是分析和总结，不提供新的代码或可直接部署的模型。
- **文献覆盖范围**：由于 LLM 领域发展极其迅速，作者承认可能遗漏了一些最新的进展。
- **近似偏差**：通过“数据集提及次数”来分类论文的方法虽然实用，但可能存在偏差，例如可能忽略了一些重要的非基准驱动的研究。

### 未来工作方向
1.  **高效的方法学**：开发更**样本高效（sample-efficient）和计算高效（compute-efficient）** 的 SFT 和 RL 方法，以降低资源消耗和环境影响。
2.  **稀疏/间接奖励下的对齐**：探索在**奖励信号稀疏或间接**（如用户反馈、行为日志）的真实世界场景下，如何有效进行 SFT 和 RL。
3.  **更强大的统一框架**：进一步深化对 SFT 和 RL 统一目标的理解，设计出能自动平衡模仿与探索、记忆与泛化的下一代后训练算法。

</details>

---

### 7. [SemantiCache: Efficient KV Cache Compression via Semantic Chunking and Clustered Merging](https://arxiv.org/abs/2603.14303)

**Authors**: Shunlong Wu, Hai Lin, Shaoshen Chen, Tingwei Lu, Yongqin Zeng, Shaoxiong Zhan, Hai-Tao Zheng, Hong-Gee Kim  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 9.5  
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
现有的 KV Cache 压缩方法（如基于**eviction**或**merging**的方法）通常在离散 token 或非语义 chunk 上操作，忽略了语言的自然层次结构，导致“**语义碎片化（semantic fragmentation）**”——即连贯的语言单元（如短语、从句、句子）被割裂，造成不可逆的信息丢失，进而显著降低模型在长上下文任务中的表现。

### 🚀 提出的新方法与新思路
作者提出 **SemantiCache**，一种新的 KV Cache 压缩框架，其核心思想是**将压缩过程与语言的语义层级结构对齐**，从而保持语义完整性。该方法模拟人类记忆长文本的认知策略，分为三个阶段：

1. **Semantic Chunking（语义分块）**  
   利用自然语言中的标点符号（如 `.` `,` `?` `\n` 等）作为**语义边界（delimiters）**，将 KV Cache 分割为语义连贯的 chunk。这些 delimiter 的 KV 状态被保留作为“结构锚点”。

2. **Greedy Seed-Based Clustering (GSC)**  
   在每个 chunk 内部，设计了一种轻量级、高效的聚类算法 GSC：
   - 顺序扫描未分配 token，选作“种子”
   - 贪心地吸收与其 Key 向量相似度高于阈值 $ T $ 的后续 token
   - 形成高内聚性的**语义簇（semantic clusters）**

3. **Clustered Merging with Proportional Attention**  
   将每个 cluster 内的 KV 状态通过均值池化合并为一个**语义核心（semantic core）**，并引入 **Proportional Attention 机制**：
   - 在注意力计算中对 pre-softmax logits 加上 $\log s$（$s$ 为 cluster 大小）
   - 等价于按原始 token 数量比例放大 attention 权重，防止信息稀释

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | SemantiCache |
|------|--------|-------------|
| **语义完整性** | 忽视语言结构 → 易产生 fragmentation | 遵循语言层级 → 保持语义连贯 |
| **压缩粒度** | Token-level 或固定大小 chunk | 语义驱动的动态 chunking + clustering |
| **信息保留** | Eviction 导致永久丢失；Merging 导致稀释 | 保留 delimiters + Proportional Attention 补偿 |
| **效率 vs 性能平衡** | 高压缩率常伴随性能下降 | 高压缩下仍接近 full model 表现 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **LongBench [25]**：多任务、双语长上下文评测基准，涵盖以下任务类型：
  - 单文档问答（Single-document QA）
  - 多文档问答（Multi-document QA）
  - 摘要生成（Summarization）
  - 少样本学习（Few-shot Learning）
  - 合成任务（Synthetic Tasks）
  - 代码补全（Code Completion）
- **Needle-in-a-Haystack (NIAH) [26]**：专门测试模型在超长文档中检索特定“needle”句子的能力，用于评估信息保留能力。

### ⚙️ 实验设置与评估指标

| 设置项 | 描述 |
|-------|------|
| **模型** | Llama-3-8B-Instruct、Mistral-7B-Instruct-v0.2 |
| **硬件平台** | NVIDIA A100 80GB GPU |
| **上下文长度** | 最大支持 32k tokens |
| **KV Cache 预算** | 设置为原大小的 20%~50%，测试不同压缩比下的性能 |
| **Delimiter 集合 D** | `[".", ",", "?", "!", ";", ":", "\t", "\n"]` |
| **GSC 相似度阈值 T** | 在 `[0.5, 0.9]` 范围内调参 |

#### 评估指标
- **准确性指标**：
  - LongBench：平均得分（Average Score）
  - NIAH：准确率（Accuracy），衡量能否正确定位目标句子
- **效率指标**：
  - **TTFT**（Time To First Token）：prefill 阶段延迟
  - **TPOT**（Time Per Output Token）：解码阶段每 token 时间
  - **Memory Footprint**：KV Cache 占用内存（GB）

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **Eviction-based** | StreamingLLM [14], H2O [15], SnapKV [13] |
| **Merging-based** | CaM [19], D2O [18] |
| **Full Model** | 不进行任何压缩，作为性能上限参考 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ LongBench 平均得分（部分展示，KV Cache = 50%）
| 方法 | Llama-3-8B | Mistral-7B |
|------|------------|-----------|
| Full Model | 34.88 | 42.01 |
| D2O | 31.86 | 40.02 |
| **Ours (SemantiCache)** | **32.34** | **40.87** |

> ➤ 在相同压缩预算下，**全面超越所有基线方法**，最接近 full model 表现。

#### ✅ Needle-in-a-Haystack 准确率（L=32k, Cache Budget=4096）
| 方法 | 准确率（%） |
|------|------------|
| Full Model | 93.85 |
| D2O | 90.29 |
| **Ours (SemantiCache)** | **91.15** |

> ➤ 展现出更强的**长距离信息检索能力**，说明语义完整性更好。

#### ✅ 效率提升（Llama-3-8B, 32k context, 20% cache budget）
| 方法 | TTFT (s) | TPOT (s) | Memory (GB) |
|------|----------|----------|------------|
| Full Model | 4.12 | 0.081 | 24.27 |
| StreamingLLM | 4.12 | 0.032 | 16.12 |
| D2O | 4.29 | 0.038 | 16.91 |
| **Ours** | **4.25** | **0.031** | **15.94** |

> ➤ **TPOT 加速达 2.61×**（0.081 → 0.031），**内存占用最低**（↓34.3%），同时 TTFT 控制良好。

### 🔍 消融实验结果

#### （1）不同 chunking 策略对比（LongBench, 20% 缓存预算）
| 方法变体 | 平均得分 |
|--------|---------|
| **Semantic Chunking（完整方法）** | **30.01** |
| Fixed-size Chunking（64 tokens） | 28.22 |
| 无 chunking（全局聚类） | 26.94 |

> ➤ 证明了**基于自然 delimiter 的语义分块至关重要**，能有效避免跨句切割带来的语义破坏。

#### （2）GSC 相似度阈值 $ T $ 的影响
| $ T $ | 平均得分 | 压缩率（%） |
|--------|----------|------------|
| 0.50 | 27.21 | 86.3% |
| 0.70 | 30.12 | 79.8% |
| 0.80 | 32.87 | 51.2% |
| 0.90 | 34.05 | 16.7% |

> ➤ 存在明显的**精度-压缩率权衡**：更高的 $ T $ 产生更多小 cluster，压缩率低但精度高；较低 $ T $ 更激进但损失明显。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **语义碎片化是现有 KV Cache 压缩方法的根本缺陷**，直接导致信息丢失和性能下降。
2. **语言的层级结构可被有效利用于 KV Cache 压缩**：通过 delimiter 分块 + 内部聚类的方式，能够实现高效且保真的压缩。
3. **SemantiCache 在多个维度上实现 SOTA 表现**：
   - 推理速度提升高达 **2.61×**
   - 内存占用减少超过 **34%**
   - 在 LongBench 和 NIAH 上**显著优于各类 eviction 和 merging 基线**
4. **Proportional Attention 机制有效缓解了 merging 引起的信息稀释问题**，确保合并后的语义核心具有合理的影响力。

### ⚠️ 方法的局限性
- **依赖显式 delimiter**：对于缺乏明确标点的语言（如中文古文）或口语化文本，delimiter 可能不够可靠。
- **GSC 是贪心算法**：不具备全局最优性，可能受 token 顺序影响。
- **阈值 $ T $ 需要调优**：不同任务和模型可能需要不同的配置，自动化调节尚未探索。
- **目前仅适用于 decoder-only 架构**，对 encoder-decoder 模型适配未知。

### 🔮 未来工作方向
- 扩展到更复杂的语义边界检测（如句法解析器、语义分割模型）
- 设计自适应的 $ T $ 调整机制，根据上下文动态控制压缩强度
- 探索与训练联合优化的可能性（e.g., 训练时增强对压缩鲁棒性）
- 应用于多模态 LLM 中的 cross-modal KV Cache 压缩

---

> 💡 **一句话总结**：  
> **SemantiCache 通过“语义分块 + 聚类合并 + 比例注意力”的三级流水线，在大幅压缩 KV Cache 的同时最大程度保留语义完整性，实现了推理效率与模型性能的双赢。**

</details>

---

### 8. [Distributed Acoustic Sensing for Urban Traffic Monitoring: Spatio-Temporal Attention in Recurrent Neural Networks](https://arxiv.org/abs/2603.13903)

**Authors**: Izhan Fakhruzi, Manuel Titos, Carmen Ben\'itez, Luz Garc\'ia  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.13903v1  

#### Abstract
Effective urban traffic monitoring is essential for improving mobility, enhancing safety, and supporting sustainable cities. Distributed Acoustic Sensing (DAS) enables large-scale traffic observation by transforming existing fiber-optic infrastructure into dense arrays of vibration sensors. However,...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Distributed Acoustic Sensing for Urban Traffic Monitoring: Spatio-Temporal Attention in Recurrent Neural Networks**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本研究针对城市环境中基于 **Distributed Acoustic Sensing (DAS)** 的交通事件连续识别难题。传统方法在处理高噪声、异构光纤网络条件下的连续、重叠事件时存在挑战，且模型泛化能力受限于部署位置。

### 提出的新方法与创新思路
- **提出了一种融合时空注意力机制的 RNN 架构（SA-bi-TA）**，用于建模 DAS 数据中的复杂时空依赖关系。
- 在 **真实城市环境** 中开展大规模 DAS 交通监测实验，突破以往局限于孤立事件或受控场景的研究范式。
- 系统性地评估了多种 **spatio-temporal attention 配置**（如 SA-bi、TA-bi、SA-bi-TA 等），分析其对性能、参数效率和可解释性的影响。
- 探索了模型在不同传感位置之间的 **空间迁移能力（spatial transferability）**，验证了模型跨地点部署的可行性。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **性能与效率平衡** | 引入 TA 模块后，在提升准确率的同时显著降低参数量（如 bi-TA 参数减少 38.85%） |
| **可解释性增强** | 注意力热图（attention heatmaps）提供了物理意义明确的决策依据，揭示了关键 SP 和时间片段 |
| **部署灵活性** | SA-bi-TA 模型展现出良好的空间迁移能力，仅需少量标注即可应用于新地点 |
| **实用性提升** | 利用已有光纤基础设施，避免额外传感器安装成本和道路封闭 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **采集地点**：西班牙格拉纳达市（Granada）
  - **Palacio de Congresos**：2车道单向通行，3个有效 SP（Spatial Points）
  - **Acera del Darro**：4车道双向通行，7个 SP，分为三组（A: #1–#3, B: #3–#5, C: #5–#7）
- **数据规模**：
  - Palacio：10小时观测数据
  - Acera：1小时观测数据
- **事件类别**：`Noise`（行人或无事件）、`Car`、`Bus`（含卡车）
- **标签方式**：通过视频记录和人工注释同步生成精确起止时间，并与 DAS 信号对齐

### 实验设置
- **DAS 设备**：Aragón Photonics™ 的 HDAS 系统，基于 CP-OTDR 技术
- **采样参数**：
  - 时间分辨率：250 Hz
  - 空间分辨率：6 m / SP，通道间距 3 m
- **预处理流程**：
  1. 对 △ε 信号进行去噪、去趋势、带通滤波（0.1–30 Hz）
  2. 分段为 2 秒滑动窗口（步长 0.5 秒），加汉明窗
  3. 提取每窗口 36 维手工特征（能量、熵、统计量等）
  4. 增强特征维度：
     - 加入一阶和二阶时间导数（+△），形成 108 维特征
     - 融合相邻 SP 特征，最终输入为 324 维向量
  5. 使用多数投票法处理过渡帧标签

### 评估指标
| 指标 | 描述 |
|------|------|
| **Accuracy (Acc)** | 整体分类准确率 |
| **F1-Score (F1)** | 类别加权平均 F1 分数 |
| **#Param** | 可训练参数数量（单位：百万 M） |
| **RI-Acc** | 相对于基线的相对准确率提升 |
| **RPI** | 相对于基线的参数增长比例 |

### 基线方法对比
- **基础模型**：
  - `LSTM`
  - `bi-LSTM`（选定为基准）
- **注意力变体**：
  - 单注意力模块：`bi-SA`, `bi-TA`, `SA-bi`, `TA-bi`
  - 级联注意力模块：`SA-bi-TA`, `TA-bi-SA`, `bi-SA-TA` 等
- 所有模型均采用 **5折交叉验证**，并使用 **Optuna 进行超参优化**

---

## 3. 主要实验结果和性能指标

### 关键性能数据（以 bi-LSTM 为基准）

| 模型 | Acc (%) | F1 (%) | #Param (M) | RI-Acc (%) | RPI (%) |
|-------|---------|--------|------------|-------------|----------|
| **bi-LSTM** (baseline) | 88.08 | 88.10 | 0.41 | 0.00 | 0.00 |
| **bi-TA** | 88.47 | 88.49 | 0.25 | **+0.45** | **-38.85** |
| **SA-bi+△** | 89.05 | 89.05 | 0.84 | +1.10 | +103.50 |
| **SA-bi-TA** | 88.62 | 88.65 | 0.68 | +0.62 | +65.48 |
| **SA-bi-TA+△** | 89.05 | 89.03 | 1.13 | +1.10 | +174.15 |

> ✅ **亮点**：`bi-TA` 在参数更少的情况下实现了更高精度；`SA-bi-TA` 在性能与复杂度之间取得良好平衡。

### 与基线方法的对比结果
- **bi-LSTM > LSTM**：双向结构利用前后文信息，有效捕捉车辆穿越光纤的快速动态变化。
- **TA 后接 bi-LSTM (bi-TA)** 表现最优之一，且参数量低于基线（-38.85%），说明其高效捕获时间模式的能力。
- **SA-bi+△** 达到最高准确率（89.05%），但代价是参数大幅增加（+103.5%），适用于资源充足场景。
- **级联注意力未带来增益**：多数双注意力配置表现不如最优单注意力组合，表明存在“收益递减”现象。

### 消融实验结果
#### （1）注意力放置位置影响
| 配置 | 性能特点 |
|------|----------|
| **bi-SA / bi-TA**（后置） | 更关注高层语义特征，参数效率高 |
| **SA-bi / TA-bi**（前置） | 作用于原始特征，需配合 +△ 才能有效提升性能 |

#### （2）是否加入时间导数（+△）
- 加入 +△ 显著提升 LSTM 性能（从 86.62% → 88.17%），使其接近 bi-LSTM
- 但在已具强时序建模能力的 bi-LSTM 上，+△ 收益有限，反而大幅增加参数

#### （3）最佳配置选择
- **SA-bi-TA** 被选为最终推荐架构：
  - 平衡了性能（+0.62% RI-Acc）、参数量（+65.48% RPI）和可解释性
  - 注意力热图显示 SA 优先关注公交车道对应的 SP2，TA 强调 Car 和 Bus 的时间演化

---

## 4. 关键结论和发现

### 主要发现
1. **RNN + 注意力机制适合 DAS 交通事件识别任务**  
   bi-LSTM 能有效建模事件间的时序依赖，而注意力机制进一步增强了对关键时空区域的关注。

2. **注意力机制提升不仅是性能，更是可解释性和效率**  
   - 注意力权重可视化揭示了模型如何结合道路几何（如 SP2 对应公交专用道）做出判断
   - bi-TA 在减少参数的同时提高准确率，有助于防止过拟合并支持边缘部署

3. **SA-bi-TA 架构具有互补协同效应**  
   - SA 先筛选重要空间位置（如靠近车道中心的 SP）
   - TA 再聚焦这些位置上的关键时间片段
   - 形成“先定位、再追踪”的层次化推理过程

4. **模型具备较强的空间迁移能力**  
   - 将在 Palacio 训练的 SA-bi-TA 模型直接用于 Acera 地点：
     - Group B 最佳：**Acc = 79.27%**, F1 = 78.79%
     - 相比原地测试（87.65%）仅下降约 8–17%，体现一定鲁棒性
   - UMAP 分析显示两地数据分布相似，支持跨站点迁移可行性

5. **注意力可用于辅助数据质量检查**  
   - 当模型预测与人工标签不一致时，注意力热图常指向正确类别（如将误标为 Car 的 Bus 正确识别）
   - 表明注意力可作为 **dataset auditing tool** 发现潜在标注错误

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **依赖高质量手工标注** | 当前仍需大量人力进行视频同步标注，难以扩展到全城范围 |
| **跨站点性能下降明显** | 不同路段耦合条件、土壤特性差异导致信号变异，限制零样本迁移效果 |
| **未考虑实时推理延迟** | 实验侧重离线评估，未报告 inference latency，实际部署需进一步优化 |
| **事件类别有限** | 仅区分 Noise/Car/Bus，缺乏对车型细分、速度估计等功能扩展 |

### 未来工作方向
1. **发展自监督/弱监督学习框架**，降低对精细标注的依赖
2. **探索 Foundation Model 范式**，在多城市 DAS 数据上预训练通用表示
3. **引入自适应归一化或域自适应技术**，提升跨站点泛化能力
4. **结合 GNSS 或其他传感器数据**，实现多模态融合感知
5. **开发轻量化模型版本**，适配边缘计算设备实现实时监控

---

> 🔚 **总体评价**：该论文系统推进了 DAS 在智能交通中的应用边界，不仅提出了高性能且可解释的 SA-bi-TA 架构，更重要的是验证了其在真实复杂城市环境下的部署潜力与迁移能力，为构建可扩展的城市级交通感知网络奠定了方法论基础。

</details>

---

### 9. [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569)

**Authors**: Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.15569v1  

#### Abstract
Scaling inference-time compute has emerged as an important driver of LLM performance, making inference efficiency a central focus of model design alongside model quality. While the current Transformer-based models deliver strong model quality, their quadratic compute and linear memory make inference...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Mamba-3: Improved Sequence Modeling using State Space Principles

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前主流的 **Transformer** 模型在推理时面临 **二次计算复杂度（quadratic compute）** 和 **线性内存增长（KV cache）** 的瓶颈，导致长序列生成效率低下。虽然已有如 **Mamba-2** 和 **Gated DeltaNet (GDN)** 等基于 **State Space Model (SSM)** 的线性模型试图解决该问题，但仍存在以下缺陷：
- **模型表达能力不足**：难以完成状态追踪（state tracking）任务（如奇偶性判断、模运算）。
- **硬件利用率低**：理论上线性推理，但实际解码阶段算术强度（arithmetic intensity）低，GPU 计算单元空闲。
- **依赖额外模块**：需要引入短因果卷积（short causal convolution）等非核心组件来提升性能。

为此，论文提出 **Mamba-3**，从 **SSM 视角出发**，以“推理优先”（inference-first）为设计原则，系统性地改进 SSM 架构。

---

### 提出的新方法与创新点

Mamba-3 引入了三项核心方法论改进：

#### （1）指数-梯形离散化（Exponential-Trapezoidal Discretization）
- **动机**：Mamba-1/2 使用的启发式离散化缺乏理论依据，仅相当于一阶精度的 Euler 方法。
- **新方法**：提出更精确的二阶 **指数-梯形规则**，对 SSM 输入积分进行更高精度近似。
- **优势**：
  - 形成一个隐式的两步卷积（implicit 2-band convolution），可替代传统架构中显式的短卷积层。
  - 理论上更合理，实证上可移除短卷积而不损失性能。

#### （2）复数域状态更新（Complex-valued State Update）
- **动机**：实数域 SSM 的状态转移矩阵特征值受限于正实数，无法建模“旋转”动态（如周期性、循环行为），导致无法解决奇偶性等任务。
- **新方法**：将 SSM 扩展至复数域，其离散化后等价于在输入/输出投影 $B_t$ 和 $C_t$ 上应用 **data-dependent RoPE**（旋转位置编码）。
- **优势**：
  - 显著增强模型的状态追踪能力。
  - 通过 “RoPE Trick” 高效实现，计算开销极小。

#### （3）多输入多输出（MIMO）架构
- **动机**：标准 SISO（Single-Input Single-Output）SSM 在解码时是内存密集型（memory-bound），算术强度低，硬件利用率差。
- **新方法**：将状态更新从外积（outer product）改为矩阵乘法（matmul），推广为 MIMO 结构。
- **优势**：
  - 大幅提升算术强度（arithmetic intensity），更好利用 GPU tensor cores。
  - 在不增加状态大小（state size）的前提下提升建模能力，保持解码延迟不变。

---

### 相比现有方法的优势
| 维度 | Mamba-2 / GDN | Mamba-3 |
|------|----------------|---------|
| **理论基础** | 启发式离散化 | 有理论支撑的高阶离散化 |
| **表达能力** | 弱于状态追踪任务 | 显著增强，可解决奇偶性、模运算 |
| **硬件效率** | 解码算术强度 ~2.5 ops/byte | MIMO 版本可达 4× 更高 FLOPs |
| **架构简洁性** | 依赖短卷积 | 可移除短卷积，更简洁 |

---

## 2. 核心实验方法和设置

### 数据集
- **预训练数据**：`FineWeb-Edu`（100B tokens），用于语言建模预训练。
- **下游评估任务**：
  - **语言建模**：LAMBADA、HellaSwag、PIQA、ARC-E/C、WinoGrande、OpenBookQA。
  - **检索能力**：
    - 真实世界：SWDE、SQuAD、FDA、TriviaQA、NQ、DROP（cloze 格式）。
    - 合成任务：Needle-in-a-Haystack (NIAH)，测试不同上下文长度下的信息提取能力。
  - **状态追踪合成任务**：
    - Parity（奇偶性）
    - Modular Arithmetic（模运算，带/不带括号）

---

### 实验设置与评估指标

#### 模型规模
- 对比多个尺度：180M、440M、880M、1.5B 参数。
- 使用 `Llama-3.1` 架构作为主干，交替堆叠 Mamba-3 块与 SwiGLU 块。

#### 评估指标
| 指标 | 说明 |
|------|------|
| **Perplexity (ppl)** | 语言建模性能，越低越好 |
| **Accuracy (%)** | 下游任务平均准确率，越高越好 |
| **Retrieval Score** | 信息检索任务的命中率 |
| **State Tracking Accuracy** | 合成任务上的分类准确率 |
| **Decode Latency** | 单 token 解码延迟（ms） |
| **Arithmetic Intensity** | FLOPs / 内存访问量，衡量硬件利用率 |

#### 基线方法
- **Transformer**（带 RoPE）
- **Gated DeltaNet (GDN)**
- **Mamba-2**

---

## 3. 主要实验结果和性能指标

### 关键性能数据（1.5B 模型）

| 模型 | FW-Edu ppl ↓ | 平均准确率 ↑ |
|------|--------------|-------------|
| Transformer | 10.51 | 55.4 |
| GDN | 10.45 | 55.8 |
| Mamba-2 | 10.47 | 55.7 |
| **Mamba-3 (SISO)** | **10.35** | **56.4** |
| **Mamba-3 (MIMO)** | **10.24** | **57.6** |

> ✅ **Mamba-3 (MIMO)** 在 1.5B 尺度下：
> - 相比次优模型（GDN）**平均准确率提升 1.8 个百分点**
> - 相比 Mamba-2 提升 1.9 个百分点
> - 相比纯 Transformer 提升 2.2 个百分点

---

### 与基线方法的对比结果

#### （1）语言建模与下游任务
- Mamba-3 在所有尺度上均优于 Mamba-2 和 GDN。
- MIMO 版本进一步显著提升性能，尤其在逻辑推理类任务（如 ARC、WinoGrande）上表现突出。

#### （2）状态追踪能力
| 模型 | Parity | Mod. Arith (w/o brack.) | Mod. Arith (with brack.) |
|------|--------|--------------------------|----------------------------|
| Mamba-2 | 0.9% | 47.8% | 0.88% |
| GDN | 100% | 99.25% | 93.5% |
| **Mamba-3** | **100%** | **98.51%** | **87.75%** |

> ✅ Mamba-3 凭借复数状态机制，**首次在现代选择性 SSM 中恢复了强大的状态追踪能力**，接近 GDN[-1,1] 的水平。

#### （3）检索能力
- 在真实世界检索任务上仍弱于 Transformer（因无 KV cache）。
- 但在 **NIAH 合成任务** 上，Mamba-3 展现出更强的 **分布外泛化能力**，尤其在超长上下文（4K, 8K）下优于 Mamba-2。
- 在混合架构中（5:1 线性层:NoPE 注意力），加入 `grouped RMSNorm` 可显著提升长度外推能力。

#### （4）推理效率
| 指标 | 结果 |
|------|------|
| **MIMO 解码 FLOPs** | 较 Mamba-2 提升高达 **4×** |
| **解码延迟** | 与 Mamba-2 相当（wall-clock time 相似） |
| **算术强度** | MIMO 显著提高，更好匹配 GPU 计算能力 |

> ⚡ Mamba-3 MIMO 实现了 **“更多计算，相同延迟”**，突破了传统线性模型“低算力密度”的限制。

---

### 消融实验结果

#### （1）组件消融（440M 模型）
| 模型变体 | Perplexity ↓ |
|----------|-------------|
| Mamba-3（完整） | **15.72** |
| + bias only | 16.49 |
| + trap only | 16.68 |
| + bias + trap | 15.72 |
| + bias + trap + conv | 15.85 |

> 🔍 表明 **bias + exponential-trapezoidal** 联合使用即可达到最佳效果，并使短卷积变得冗余。

#### （2）MIMO 参数化影响
- 使用轻量级参数化策略（仅扩展 $B, C$ 投影，头特异性缩放），避免参数爆炸。
- 通过缩小 MLP 宽度实现参数匹配，确保公平比较。

#### （3）复数状态有效性
- 移除 RoPE 或使用标准 RoPE 均导致状态追踪任务性能崩溃（接近随机猜测）。
- 证明 **data-dependent RoPE** 是成功的关键。

---

## 4. 关键结论和发现

### 主要发现
1. **SSM 视角能指导高效架构设计**：从连续 SSM 动力学出发，可自然导出更优的离散化、复数扩展和 MIMO 结构。
2. **表达能力与效率可兼得**：Mamba-3 在提升建模能力的同时，通过 MIMO 提高硬件利用率，**推进了性能-效率帕累托前沿（Pareto frontier）**。
3. **短卷积非必需**：结合 bias 和 exponential-trapezoidal 后，可安全移除短卷积，简化架构。
4. **复数状态有效且高效**：通过 RoPE trick 实现的复数状态极大增强了状态追踪能力，且计算代价极低。
5. **MIMO 是推理优化的有效路径**：在不牺牲解码速度的前提下大幅提升 FLOPs 利用率。

---

### 方法的局限性
- **训练成本更高**：MIMO 版本训练 FLOPs 增加约 R 倍（R 为 rank），不适合资源受限场景。
- **检索能力仍有差距**：固定状态大小限制了对极长上下文的精确记忆，在强检索任务上仍不如 Transformer。
- **理论假设与实践差异**：ablation 显示无需强制满足误差界条件（如 $\lambda = \frac{1}{2} + O(\Delta t)$），说明理论最优 ≠ 实践最优。

---

### 未来工作方向
- 探索更复杂的 MIMO 参数化与训练策略，降低训练开销。
- 将 Mamba-3 思路应用于其他模态（如语音、视觉）。
- 设计更高效的 hybrid architecture，平衡 SSM 的高效性与注意力的记忆能力。
- 进一步探索复数 SSM 在算法推理、程序合成等任务中的潜力。

---

> 📌 **总结一句话**：  
> **Mamba-3 通过三项基于 SSM 理论的创新——指数-梯形离散化、复数状态更新和 MIMO 架构，在不牺牲推理效率的前提下，显著提升了线性序列模型的表达能力和硬件利用率，重新定义了高效语言模型的设计边界。**

</details>

---

### 10. [Committee Configuration Optimization for Parallel Byzantine Consensus in a Trusted Execution Environment](https://arxiv.org/abs/2603.14445)

**Authors**: Yifei Xie, Btissam Er-Rahmadi, Xiao Chen, Tiejun Ma, Jane Hillston  
**Category**: cs.DC  
**Published**: 2026-03-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.14445v1  

#### Abstract
Parallel Byzantine Fault Tolerant (BFT) protocols based on committee-based sharding improve scalability but weaken safety since smaller node groups are responsible for consensus. Recent approaches integrate trusted execution environments (TEEs) into parallel BFT frameworks to enhance safety. While t...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Committee Configuration Optimization for Parallel Byzantine Consensus in a Trusted Execution Environment

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
传统的 **Byzantine Fault Tolerant (BFT)** 协议面临可扩展性瓶颈，而基于委员会分片（committee-based sharding）的 **parallel BFT** 虽然提升了吞吐量，但因节点组规模变小而削弱了安全性。尽管引入 **Trusted Execution Environment (TEE)** 可以增强安全并降低容错下限（从 `3f+1` 降至 `2f+1`），但现有的委员会配置方法多采用随机分配，导致性能受限。

此外，在 TEE 发生故障时，系统需回退到传统 `3f+1` 模式以保证安全，但当前配置策略未考虑这一场景下的性能优化。

### 🚀 提出的新方法与创新
本文提出了一种基于 **Mixed Integer Programming (MIP)** 的 **Committee Configuration Optimization (CCO)** 模型，用于在 TEE 支持的 parallel BFT 系统中优化委员会配置。

#### 主要创新点：
- **首次将数学规划应用于 TEE-based parallel BFT 的委员会成员配置优化**，综合考虑通信延迟、节点故障率等因素。
- **支持自适应回退机制（adaptive fallback）**：当某个委员会中的 TEE 出现故障时，仅该委员会切换至 `3f+1` 模式，其余正常委员会仍保持高效运行，避免全局降级带来的性能损失。
- **最小化交易延迟为目标函数**，涵盖 Pre-prepare、Prepare、Verification、Commit 等阶段的时间建模，并排除客户端网络影响。

### 🔍 相比现有方法的优势
| 方面 | 现有方法 | 本文 CCO |
|------|--------|---------|
| 配置方式 | 随机分配或静态规则 | 数学优化驱动，动态适应网络状态 |
| 安全与性能权衡 | 忽略 TEE 故障对性能的影响 | 显式建模 TEE 故障并进行自适应重配置 |
| 性能目标 | 多关注吞吐量 | 综合优化延迟与吞吐，尤其改善高负载下的表现 |
| 适用场景 | 通用 BFT 或普通 sharding | 特别适用于 TEE-enabled parallel BFT 架构（如 TopBFT） |

---

## 2. 核心实验方法和设置

### 💾 实验平台与环境
- **测试床实现语言**：C 语言
- **TEE 实现**：基于 Intel SGX 构建可信单调计数器
- **部署平台**：Microsoft Azure 云服务
- **硬件资源**：5 台虚拟机（VM），每台 8 vCPU + 64 GB RAM
- **模拟节点数**：最多 240 个节点实例分布在 5 个 VM 上
- **共识协议基础**：集成于 **TopBFT** 协议框架中
- **求解器**：使用 **CPLEX** 求解 MIP 优化模型

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Throughput (op/s)** | 每秒处理的操作数量 |
| **Latency (ms)** | 从客户端发送请求组到接收到所有有效回复的总时间 |
| **Fallback Performance** | 在 30% 节点发生 TEE 故障情况下的吞吐与延迟表现 |

### 🔁 基线方法对比
实验对比了以下代表性 BFT 协议：
- **HotStuff**：高性能单委员会 BFT，优于 BFT-SMaRt
- **FastBFT**：基于 TEE 的高效 BFT，优于 MinBFT 和 CheapBFT
- **GeoBFT**：一种并行 BFT 方案，强调地理分布
- **TopBFT (default)**：原始 TopBFT 使用随机四节点委员会配置
- **CCO-driven TopBFT**：本文提出的优化版本

---

## 3. 主要实验结果和性能指标

### 📈 正常操作模式下的性能提升（Normal Case）

#### ✅ 吞吐量（Throughput）
- 当节点数为 240 时：
  - **CCO-driven TopBFT** 达到约 **230 op/s**
  - 原始 TopBFT 为约 **200 op/s**
  - **相对提升达 15%**

> 图 2a 显示，随着节点增加，CCO 模型能更有效地组织委员会，避免验证委员会成为瓶颈。

#### ⏱️ 延迟（Latency）
- 在 200+ 节点时，原始 TopBFT 因验证委员会过载导致延迟显著上升
- CCO 通过优化配置使延迟维持稳定水平
- **相比默认 TopBFT，延迟降低近 18.9%（在 1MB payload 下）**

> 图 2b 表明 CCO 有效缓解了热点拥塞问题。

#### 📦 不同操作负载大小下的表现（Operation Size: 0B ~ 1MB）
- 所有协议随 payload 增大吞吐下降、延迟上升
- **CCO-driven TopBFT 在大负载下优势更明显**：
  - 在 1MB 操作大小下，**吞吐高出标准 TopBFT 近 29.4%**
  - 延迟比其他协议（FastBFT、HotStuff、GeoBFT）显著更低

---

### 🔄 自适应回退机制性能（Fallback Process）

#### 设置条件
- 强制 **30% 节点出现 TEE 故障**
- 对比两种 fallback 策略：
  - **Standard Fallback**：所有委员会转为 `3f+1` 模式
  - **CCO-driven Fallback**：仅受影响委员会升级，其余保持 `2f+1`

#### 结果
- **吞吐量提升 21%**（见图 4a）
- **延迟更低且增长平缓**（见图 4b）
- 验证了 CCO 的 **选择性回退机制** 可大幅减少系统降级开销

> ❗ 尽管 fallback 场景整体性能低于正常运行，但 CCO 显著减轻了性能衰退。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **委员会配置对 parallel BFT 性能具有决定性影响**，简单的随机分配无法满足高性能需求。
2. **基于 MIP 的 CCO 模型能够联合优化通信延迟与容错能力**，显著提升 TopBFT 的吞吐与响应速度。
3. **在 TEE 故障场景下，自适应 fallback 机制至关重要** —— 全局切换会带来不必要的性能惩罚，而 CCO 实现了细粒度弹性恢复。
4. **真实云环境中实验验证了 CCO 的有效性**：在多种节点规模和负载条件下均表现出一致优越性。

### ⚠️ 局限性
- **计算开销**：MIP 模型依赖 CPLEX 求解，在大规模节点变动时可能引入配置延迟。
- **静态假设**：当前模型假设节点延迟和故障率相对稳定，未完全建模极端动态变化。
- **依赖 TEE 可用性**：若 TEE 普及率低或存在广泛攻击风险，其优势将受限。

### 🔮 未来工作方向
- 探索轻量化近似算法（如启发式或 ML-based）替代完整 MIP 求解，提升实时性。
- 扩展模型以支持动态节点加入/退出（dynamic membership）。
- 结合 reputation system 与 CCO，进一步提升抗恶意行为能力。
- 在更大规模跨地域网络中验证模型鲁棒性。

---

## 总结

✅ **CCO 是首个针对 TEE-based parallel BFT 的数学优化型委员会配置方案**，  
✅ **兼顾正常运行效率与故障恢复弹性**，  
✅ **实验证明其在吞吐（↑15%~21%）和延迟（↓~19%）方面全面超越现有方法**。

👉 该工作为构建**高安全、高性能、强弹性的下一代区块链共识系统**提供了重要理论与实践支撑。

</details>

---

### 11. [Lightweight User-Personalization Method for Closed Split Computing](https://arxiv.org/abs/2603.14958)

**Authors**: Yuya Okada, Takayuki Nishio  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.14958v1  

#### Abstract
Split Computing enables collaborative inference between edge devices and the cloud by partitioning a deep neural network into an edge-side head and a server-side tail, reducing latency and limiting exposure of raw input data. However, inference performance often degrades in practical deployments due...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Lightweight User-Personalization Method for Closed Split Computing*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Split Computing** 场景中，一个深度神经网络被分割为边缘设备端的 **head network** 和云端的 **tail network**，以降低延迟并保护原始输入数据隐私。然而，在实际部署中存在以下挑战：
- **用户个性化需求**：不同用户的输入数据分布（data distribution）差异大，导致通用模型性能下降。
- **通信不可靠**：无线信道中的 **packet loss** 会破坏传输的中间表示（latent features），影响推理准确性。
- **隐私泄露风险**：尽管不直接传输原始数据，但中间表示仍可能被用于 **inversion attack** 重构输入。
- **闭源模型限制（Closed-Model Constraints）**：许多实际系统中，head/tail 网络是黑盒（如API或专用硬件），无法进行传统 fine-tuning。

这些问题使得传统的参数微调（fine-tuning）方法失效。

### 提出了什么新方法或新思路
作者提出 **SALT (Split-Adaptive Lightweight Tuning)** ——一种轻量级、适用于闭源 Split Computing 环境的自适应框架。

#### 核心思想：
- 在客户端插入一个小型可训练的 **adapter 模块**，位于冻结的 head network 和服务器端的 tail network 之间。
- 该 adapter 对 head 输出的中间表示 $ z = H(x) $ 进行修正，生成 $ z' = z + \Delta z $，其中 $\Delta z = S(z)$ 是由 adapter 预测的残差向量。
- **仅训练 adapter 参数**，保持 head 和 tail 网络完全冻结且不可访问。
- 不改变中间表示维度，因此 **不增加通信开销**。

#### 支持多种适应目标（通过调整训练条件实现）：
- 用户个性化（personalization）
- 抗通信退化（robustness to packet loss）
- 抗噪声扰动（noise-robust inference，用于隐私保护）

### 相比现有方法的优势
| 特性 | SALT | 传统方法（Retrain/Fine-tune） |
|------|------|-------------------------------|
| 是否需要访问模型内部参数 | ❌ 否（black-box 兼容） | ✅ 是 |
| 是否修改原始模型结构 | ❌ 否 | ✅ 是 |
| 是否引入额外通信负载 | ❌ 否 | ❌（通常也不） |
| 训练成本 | 极低（仅训练 adapter） | 高（需反向传播至 head 层） |
| 多任务统一支持 | ✅ 单一架构支持多种目标 | ❌ 通常需分别设计 |

> ✅ **优势总结**：SALT 实现了在 **closed-model constraint** 下的高效、灵活、低成本模型适配，填补了理论研究与现实部署之间的鸿沟。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **CIFAR-10**：10类图像分类任务，用于主实验。
- **CIFAR-100**：100类细粒度分类任务，验证方法在更复杂场景下的有效性。

### 实验设置
- **骨干模型**：ResNet-18，预训练后固定作为 feature extractor。
- **Split Points**：考虑四个位置：
  - `BeforeBlock1`
  - `AfterBlock1`
  - `AfterBlock2`
  - `AfterBlock3`
- **Adapter 结构**：
  - 3层轻量卷积网络（Conv-BN-ReLU ×2 → 1×1 Conv）
  - 参数量约 443K，计算量 28.39 MMACs
  - 保持输入输出维度一致（no spatial/channel change）

### 评估指标
| 指标 | 描述 |
|------|------|
| **Top-1 Accuracy** | 主任务分类准确率 |
| **Training Latency** | 总训练时间（含前向/反向通信延迟） |
| **Inference Latency** | 推理延迟 |
| **SSIM (Structural Similarity Index)** | 评估 inversion attack 重建图像质量，衡量隐私保护效果 |
| **Robustness under Packet Loss** | 在不同丢包率（0.0–0.75）下的准确率表现 |
| **Noise Robustness** | 注入高斯噪声（$\sigma$ 从 0.0 到 1.5）时的准确率变化 |

### 基线方法对比
| 方法 | 是否适用闭源环境 | 可训练部分 | 训练成本 |
|------|------------------|------------|----------|
| **Original** | ✅ | 无 | None |
| **Retrain (Head Retraining)** | ❌ | 整个 head network | 高 |
| **Fine-tune (Head Tune)** | ❌ | head 的部分参数 | 中等 |
| **SALT (Insertion Adapter)** | ✅ | 插入模块 | 中低 |
| **SALT (Residual Adapter)** | ✅ | 残差修正模块 | 低 |

> 所有方法均在同一训练配置下运行（Adam, LR=1e-3, Batch=128, Epochs=100 或早停）

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📊 CIFAR-10 用户个性化结果（Table 4 & Fig. 5）
| 方法 | 准确率 | 训练轮数 | 总训练延迟（秒） |
|------|--------|----------|------------------|
| Original | 88.1% | – | – |
| Retrain | 92.4% | 54.0 | ~1967.9 |
| Fine-tune | 92.0% | 26.0 | ~906.9 |
| **SALT (Residual Adapter)** | **93.8%** | **20.5** | **~682.1** |

> ✅ **提升**：相比 retrain，准确率 ↑1.4%，训练延迟 ↓**65%+**

#### 📈 CIFAR-100 用户个性化（Fig. 6）
- SALT 达到 **90.2%** 准确率，显著优于 retrain（82.4%）和 fine-tune（88.9%）
- 仅用 **21.2 轮** 完成训练，远低于其他方法

#### 📉 通信鲁棒性测试（Fig. 8）
| 丢包率 | Original | SALT (Residual) |
|-------|---------|----------------|
| 0% | ~90% | ~93.8% |
| 25% | ~85% | ~93.5% |
| 50% | ~70% | ~92% |
| **75%** | **<50%** | **>90%** ✅ |

> 🔥 SALT 在 **75% packet loss** 下仍维持超过 90% 准确率，展现出极强鲁棒性。

#### 🔐 隐私保护兼容性测试（Fig. 10–12）
- **噪声注入强度 $\sigma = 1.0$**
  - 无 SALT：准确率降至 ~50%
  - **有 SALT**：准确率保持 **~88%**
- **SSIM 分析（图11）**
  - 有无 SALT 的 SSIM 曲线几乎重合
  - 表明 SALT **未削弱噪声带来的隐私保护能力**

> ✅ 实现“**高精度 + 强隐私**”双赢：SALT 补偿了噪声引起的性能损失，同时保留其防重构能力。

### 消融实验结果
- **Split Point 影响（Fig. 7）**
  - SALT 在所有 split point 上表现稳定，不受分割深度影响。
  - 表明其适应机制独立于特征抽象层级。
- **Adapter 类型比较**
  - **Residual Adapter** 比 Insertion Adapter 收敛更快（20.5 vs 25.3 epochs），训练成本更低。
  - 因其学习的是“增量修正”，更容易优化。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **SALT 可在完全闭源环境下实现高效个性化**：无需访问 head/tail 网络参数，即可完成用户域适配。
2. ✅ **单一轻量模块支持多目标联合优化**：通过调整训练条件，可同时应对数据偏移、通信退化、隐私扰动。
3. ✅ **显著降低训练成本**：相比 retrain，训练延迟减少 **60%以上**，适合资源受限边缘设备。
4. ✅ **强健的通信鲁棒性**：即使在 **75% packet loss** 下仍能维持 >90% 准确率。
5. ✅ **兼容隐私保护机制**：能有效补偿噪声注入导致的精度下降，且 **不损害其抗 inversion attack 能力**。

### 方法的局限性
- 当前实验基于 **ResNet-18 + CIFAR**，尚未在更大模型（如 ViT、Transformer）或真实视频流等动态数据上验证。
- Adapter 设计虽轻量，但在极端低功耗设备（如MCU）上仍可能存在部署挑战。
- 依赖 server 返回梯度进行训练，在某些安全策略严格的系统中可能受限（需可信 server）。

### 未来工作方向
1. 将 SALT 扩展至 **更大规模模型**（如 ResNet-50、ViT）及 **真实边缘云部署环境**。
2. 探索 **无监督/自监督训练方式**，减少对标注数据的依赖。
3. 从 **信息论角度分析** SALT 如何调节 feature transmission 与 channel impairment 的关系。
4. 研究 **multi-user adapter sharing** 或联邦式 SALT 架构，进一步提升泛化能力。

--- 

> 💡 **总体评价**：  
> SALT 提出了一种极具实用价值的 Split Computing 自适应范式，解决了“如何在不能改模型的前提下让模型更好用”的核心难题。其实验充分、设计巧妙，为边缘智能系统的个性化、鲁棒性和隐私保护提供了统一而高效的解决方案。

</details>

---

### 12. [Federated Learning of Binary Neural Networks: Enabling Low-Cost Inference](https://arxiv.org/abs/2603.15507)

**Authors**: Nitin Priyadarshini Shankar, Soham Lahiri, Sheetal Kalyani, Saurav Prakash  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.15507v1  

#### Abstract
Federated Learning (FL) preserves privacy by distributing training across devices. However, using DNNs is computationally intensive at the low-powered edge during inference. Edge deployment demands models that simultaneously optimize memory footprint and computational efficiency, a dilemma where con...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Federated Learning of Binary Neural Networks: Enabling Low-Cost Inference

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Federated Learning (FL)** 中，虽然训练过程保护了数据隐私，但部署阶段仍面临挑战：**现代深度神经网络（DNN）在资源受限的边缘设备上进行推理时，计算和内存开销巨大**。传统方法如后训练二值化（post-training binarization）虽能压缩模型，但会导致严重的精度损失。

此外，现有联邦学习中的通信优化方法（如 FedMUD、FedBAT）主要关注减少通信成本，而忽略了**推理阶段的运行时效率**，无法满足低功耗边缘设备的实际需求。

### 提出了什么新方法或新思路
本文提出 **FedBNN** —— 一种面向联邦学习的旋转感知二值神经网络（Rotation-aware Binary Neural Network）框架，其核心思想是：

- 在本地训练过程中直接学习二值权重（Binary Weights），而非仅对通信更新进行二值化。
- 引入 **可学习的旋转矩阵 $R_1$ 和 $R_2$** 来最小化实值权重与其二值化版本之间的角偏差（angular bias），从而缓解量化误差导致的性能下降。
- 设计了一个融合服务器与客户端权重的混合表示 $ \tilde{w} = \lambda w_{\text{client}} + (1-\lambda) w_{\text{server}} $，并在该表示上应用旋转操作，实现联邦场景下的几何对齐。
- 提出带全局记忆的自适应调整机制，通过可学习参数 $\alpha$ 和 $\beta$ 控制旋转方向和服务器模型的影响，增强稳定性。
- 使用 **Training-aware Approximation** 替代传统的 STE（Straight-Through Estimator），提升梯度流动性和训练稳定性。

### 相比现有方法的优势
| 方法 | 是否真正降低推理成本 | 是否集成训练时二值化 | 是否处理角偏差 | 联邦一致性机制 |
|------|------------------------|------------------------|----------------|----------------|
| FedAvg | ❌（全精度模型） | ❌ | ❌ | ✅ |
| FedBAT | ❌（仅通信二值化） | ⚠️（局部使用） | ❌ | ❌ |
| FedMUD | ❌（低秩分解） | ❌ | ❌ | ✅ |
| **FedBNN** | ✅（运行时 FLOPs ↓58×, 内存 ↓32×） | ✅（全程端到端二值训练） | ✅（旋转对齐） | ✅（服务器侧聚合评估） |

> ✅ FedBNN 是首个在联邦学习中实现“**训练即二值化**”并显著提升推理效率而不牺牲准确性的框架。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验在多个标准 FL 基准数据集上进行，涵盖图像分类任务：
- **FMNIST**（Fashion-MNIST）
- **SVHN**（Street View House Numbers）
- **CIFAR-10**
- **Tiny-ImageNet**
- **FEMNIST**（Extended MNIST，手写字符）

### 实验设置
- **客户端数量**：$N_c = 100$
- **每轮采样客户端数**：10
- **本地训练轮次（local epochs）**：15（部分为5或10）
- **批量大小（batch size）**：64
- **总通信轮次**：1500（部分为500）
- **学习率**：0.1（CNN4, ResNet）或 0.01（ConvNeXt-Tiny），按周期衰减
- **硬件平台**：PyTorch + 双 NVIDIA RTX PRO 6000 GPU

### 数据异构性设置（Non-IID）
为了验证鲁棒性，设计三种数据分布：
1. **IID**：数据均匀随机划分
2. **Non-IID 1**：基于 Dirichlet 分布（$\alpha=0.3$）划分标签，模拟统计异质性
3. **Non-IID 2**：极端标签偏斜，每个客户端只拥有少数类别（如 CIFAR-10 中每客户仅3类）

### 评估指标
- **Accuracy**（IID / Non-IID 1 / Non-IID 2）
- **Runtime FLOPs**（推理阶段浮点运算量）
- **Memory Usage**（模型存储空间）
- **Binarized Accuracy**（模型完全二值化后的测试精度）
- **Ablation Study**：消融服务器对齐项、旋转机制等组件

### 基线方法对比
- **FedAvg**：标准联邦平均算法（全精度）
- **FedBAT**：可学习二值化更新，但保留实值模型
- **FedMUD**：低秩分解优化通信效率

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 1）

| Dataset (Model) | Method | IID Acc (%) | Non-IID 1 (%) | Non-IID 2 (%) | FLOPs | Memory (MB) | Binarized Acc (%) |
|------------------|--------|-------------|---------------|---------------|--------|--------------|--------------------|
| FMNIST (CNN4) | FedAvg | 92.24 | 91.44 | 89.28 | 2.02e7 | 1.56 | 53.42 |
| | **FedBNN** | **88.46** | **87.76** | **83.38** | **3.48e5** (**↓58×**) | **0.049** (**↓32×**) | **88.46** |
| SVHN (CNN4) | FedAvg | 92.10 | 90.60 | 89.34 | 3.00e7 | 1.60 | 28.01 |
| | **FedBNN** | **86.89** | **85.63** | **84.40** | **5.19e5** | **0.050** | **86.89** |
| CIFAR-10 (ResNet10) | FedAvg | 90.86 | 86.28 | 70.62 | 4.40e8 | 19.62 | 17.20 |
| | **FedBNN** | **89.95** | **82.84** | **73.82** | **1.11e7** | **0.613** | **89.95** |
| Tiny-ImageNet (ResNet18) | FedAvg | 55.00 | 52.62 | 54.54 | 4.44e9 | 45.09 | ~1–2 |
| | **FedBNN** | **46.54** | **43.00** | **45.74** | **9.05e7** | **1.41** | **46.54** |
| FEMNIST (ResNet18) | FedAvg | 80.24 | 81.12 | 80.32 | 9.13e8 | 45.09 | ~2 |
| | **FedBNN** | **79.73** | **80.31** | **79.13** | **1.84e7** | **1.41** | **79.73** |

> 💡 **观察**：尽管 FedBNN 清晰精度略低于 FedAvg（差距约 3–5%），但在所有数据集上均显著优于其他二值化基线（FedBAT、FedMUD），且**二值化后性能几乎无损**。

### 与基线方法的对比结果
- **vs FedBAT**：FedBNN 在 SVHN 上高出 +0.88%～+8.62%，在 CIFAR-10 上高出 +10% 以上；FedBAT 二值化后性能崩溃至个位数。
- **vs FedMUD**：FedBNN 在 CIFAR-10 (ConvNeXt-Tiny) 上反超 FedAvg 和 FedMUD（72.08% vs 65.22%/53.92%），说明其在复杂架构+异构数据下更具优势。
- **通信开销分析（Appendix A.4）**：传输旋转矩阵 $R_1, R_2$ 的额外参数占比 <1%，且随模型增大而降低（如 ConvNeXt-Tiny 仅为 0.28%），可忽略不计。

### 消融实验结果（Ablation Study）

#### （1）移除服务器对齐（set $\beta=1, \lambda=1$）
| Dataset (Model) | FedBNN (完整) | FedBNN (no alignment) | 差距 |
|------------------|----------------|------------------------|-------|
| CIFAR-10 (ResNet10, Non-IID 2) | 73.82 | 68.54 | **+5.28%** |
| Tiny-ImageNet (ResNet18, IID) | 46.54 | 42.86 | **+3.68%** |

> 🔍 表明服务器对齐项对稳定聚合、提升泛化至关重要，尤其在高异构环境下。

#### （2）不同聚合策略比较（Appendix A.1）
将旋转对齐应用于聚合前 vs 聚合后：
- **先聚合旋转权重再构造辅助更新（FedBNN）** 效果更优。
- 特别是在 CIFAR-10 (ResNet10, Non-IID 2) 上，准确率从 55.62% 提升至 **73.82%**，表明旋转空间内的聚合更能保持符号一致性。

#### （3）与同等复杂度实值模型对比（Appendix A.3）
将 ResNet 结构缩小以匹配 FedBNN 的 FLOPs 和内存预算：
- FedBNN 在 CIFAR-10 (Non-IID 2) 上达到 73.82%，而 FLOPs-matched ResNet 仅为 66.16%，**领先 7.66%**。
- 在 Tiny-ImageNet 上也领先约 6–10 个百分点。

> ✅ 证明：**直接训练二值网络比压缩实值网络更高效有效**。

---

## 4. 关键结论和发现

### 主要发现
1. **FedBNN 成功实现了“训练即二值化”**，在整个联邦流程中维护一个高质量的二值模型，避免了后处理带来的严重精度退化。
2. **旋转对齐机制有效缓解了量化角偏差**，使得实值与二值表示之间的一致性大幅提升。
3. **服务器端双路径聚合策略（real-valued + rotation-aligned）提供了可靠的模型选择依据**，无需改变通信协议即可选出最优二值模型。
4. **在高度非独立同分布（Non-IID）场景下，FedBNN 表现出更强的鲁棒性**，甚至在某些情况下超越全精度 FedAvg（如 ConvNeXt-Tiny on CIFAR-10）。
5. **二值化后性能几乎无损**，而 FedAvg 等方法二值化后性能暴跌至个位数，凸显 FedBNN 的部署友好性。

### 方法的局限性
- 当前方法主要针对 CNN 架构设计，在 Transformer 类模型上的扩展需进一步研究。
- 旋转矩阵的学习增加了每层的计算负担（尽管总体仍远低于原始 FLOPs）。
- 对于极小模型（如 CNN4），旋转带来的增益相对有限，可能不如简单剪枝有效。

### 未来工作方向
- 探索更高效的旋转参数化方式（如低秩近似）以进一步降低开销。
- 将 FedBNN 扩展至更多模态（如语音、文本）和更大规模模型（如 ViT、LLMs）。
- 研究动态旋转机制，根据客户端数据分布自适应调整旋转强度。
- 结合稀疏化（sparsification）与二值化，构建更极致的轻量化联邦系统。

---

> 📌 **总结一句话**：  
> **FedBNN 是首个在联邦学习中实现端到端二值神经网络训练的方法，在几乎不损失精度的前提下，将推理阶段的 FLOPs 降低 58 倍、内存占用减少 32 倍，并展现出卓越的 Non-IID 鲁棒性和部署实用性。**

</details>

---

### 13. [A Dual-Path Generative Framework for Zero-Day Fraud Detection in Banking Systems](https://arxiv.org/abs/2603.13237)

**Authors**: Nasim Abdirahman Ismail, Enis Karaarslan  
**Category**: cs.AI  
**Published**: 2026-03-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.13237v1  

#### Abstract
High-frequency banking environments face a critical trade-off between low-latency fraud detection and the regulatory explainability demanded by GDPR. Traditional rule-based and discriminative models struggle with "zero-day" attacks due to extreme class imbalance and the lack of historical precedents...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Dual-Path Generative Framework for Zero-Day Fraud Detection in Banking Systems

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**高频率银行交易环境中的零日欺诈检测**（zero-day fraud detection）所面临的三大核心挑战：
- **极端类别不平衡**（fraudulent transactions 占比通常低于 0.17%），导致传统模型难以学习罕见攻击模式；
- **实时性要求高**（需在 <50ms 内完成决策），限制了复杂模型和解释性方法的应用；
- **监管合规需求**（如 GDPR Article 22 要求可解释性），要求模型具备事后解释能力（post-hoc interpretability）。

### 提出的新方法与创新思路
作者提出了一种 **Dual-Path Generative Framework**（双路径生成框架），其核心创新如下：

#### ✅ 双路径架构设计（Decoupled Real-Time & Offline Processing）
- **同步检测路径（Synchronous Detection Path）**：基于 **Variational Autoencoder (VAE)** 构建“合法交易流形”（legitimate transaction manifold），通过重建误差（reconstruction error）实现实时异常检测。
- **异步合成路径（Asynchronous Synthesis Path）**：采用 **Wasserstein GAN with Gradient Penalty (WGAN-GP)** 离线生成高熵欺诈样本，用于压力测试和边界扩展。

> 这种解耦设计实现了**实时检测**与**对抗训练**的分离，避免了在线推理延迟。

#### ✅ 针对离散金融特征的可微处理：Gumbel-Softmax Estimator
- 为解决 **Merchant Category Code (MCC)** 等离散类别变量不可导的问题，引入 **Gumbel-Softmax** 技术，使模型能够在反向传播中有效更新这些特征的嵌入表示。
- 保证了生成路径对业务逻辑的一致性和真实性。

#### ✅ 触发式可解释机制（Triggered Explainability with SHAP）
- 引入选择性激活的 **SHAP (Shapley Additive Explanations)** 模块，仅当交易的重建误差超过动态阈值 $ T $（即高不确定性案例）时才触发解释分析。
- 成功平衡了 **XAI 的计算开销** 与 **系统吞吐量**，满足 <50ms 推理延迟要求。

#### ✅ 人类闭环反馈机制（Human-in-the-Loop, HITL）
- 将专家判断纳入闭环流程：确认为欺诈的案例存入 **Adversarial Buffer**，用于后续 WGAN-GP 的再训练，实现持续自适应优化。

---

### 相比现有方法的优势
| 维度 | 传统方法（如 SMOTE + RF/SVM） | 本论文方法 |
|------|-------------------------------|-----------|
| 类别不平衡处理 | 线性插值（SMOTE），产生噪声，破坏数据流形 | WGAN-GP 实现非线性、高熵欺诈生成，保持分布一致性 |
| 实时性能 | 多数黑箱模型无法满足 <50ms 要求 | VAE 主路径轻量高效，延迟可控 |
| 可解释性 | 通常事后附加，影响性能 | 条件触发 SHAP，仅对可疑交易启用，兼顾效率与合规 |
| 对抗鲁棒性 | 缺乏主动边界探索机制 | WGAN-GP 主动“攻击”VAE 边界，提升泛化能力 |

---

## 2. 核心实验方法和设置

### 数据集
论文未明确列出具体公开数据集名称，但从上下文推断使用的是典型的信用卡交易日志，具有以下特性：
- 高频数字支付记录；
- 包含字段如：transaction amount, location, MCC, device ID, timestamp 等；
- 极端类别不平衡：fraud ratio ≈ 0.17%，符合真实银行业务场景。

> 注：实验基于模拟或合作金融机构提供的真实交易日志进行验证。

---

### 实验设置
- **预处理层**：使用 **Entity Embedding** 处理高维稀疏 MCC 字段，并结合 **Gumbel-Softmax** 实现可微采样。
- **VAE 结构**：编码器将输入映射至概率潜空间 $ z $，解码器重构原始交易；目标是最小化 ELBO 损失。
- **WGAN-GP 结构**：离线运行，以梯度惩罚确保训练稳定，生成逼真的欺诈向量用于增强检测边界。
- **SHAP 触发机制**：仅当 $ E(x) = ||x - \hat{x}|| > T $ 时启动 SHAP 分析，输出特征重要性报告。

### 评估指标
- **Detection Performance**：
  - Precision, Recall, F1-score
  - False Positive Rate (FPR)
  - AUC-ROC
- **Operational Metrics**：
  - Inference Latency (<50ms 是否达标)
  - SHAP computation frequency (仅作用于 <1% 的异常样本)
- **Adversarial Resilience**：对三类高级攻击的检测能力：
  1. **Salami Slicing Attack**（微额切片攻击）
  2. **Card-Not-Present (CNP) Velocity Attack**
  3. **Account Takeover (ATO)**

### 基线方法对比
- Rule-based systems（静态规则引擎）
- Discriminative models：Random Forest, SVM
- Oversampling-enhanced models：SMOTE + RF/SVM
- Pure GAN-based augmentation methods
- Monolithic VAE/GAN hybrid models

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自文中描述与推论）
| 指标 | 本方法表现 |
|------|----------|
| 平均推理延迟 | **<50ms**（满足高频交易要求） |
| SHAP 启用比例 | **<1% 的交易**触发解释模块 |
| Fraud Detection Recall | 显著高于基线（尤其对 zero-day 攻击） |
| False Positive Rate (FPR) | 低于 SMOTE 类方法（减少误拦合法交易） |
| AUC-ROC | 达到先进水平（优于传统判别模型） |

### 与基线方法的对比结果
- 在 **Salami Slicing 攻击检测** 上，VAE 能捕捉频率熵异常，而规则系统因金额过小无法识别；
- 在 **CNP Velocity Attacks** 中，Gumbel-Softmax + WGAN-GP 成功建模 MCC 跨域跳跃行为，检测率显著提升；
- 在 **ATO 场景** 下，HITL 机制结合 SHAP 提供行为偏移证据（如 IP-location mismatch），支持人工复核并驱动模型迭代。

### 消融实验（Ablation Study）结果（隐含分析）
尽管未提供显式表格，但文中通过多个维度论证组件必要性：
- **无 Gumbel-Softmax** → 离散特征无法参与端到端训练，生成质量下降；
- **无 WGAN-GP 路径** → 检测边界脆弱，易被新型攻击绕过；
- **全量启用 SHAP** → 计算复杂度达 $ O(2^n) $，严重拖慢系统吞吐；
- **无 HITL 反馈** → 无法应对概念漂移（concept drift），长期性能退化。

> 实验表明：各模块协同工作是实现高性能、高鲁棒、合规检测的关键。

---

## 4. 关键结论和发现

### 主要发现
1. **双路径架构能有效解耦实时检测与对抗训练**，在不牺牲延迟的前提下显著提升 zero-day 攻击识别能力。
2. **WGAN-GP 比 SMOTE 更适合金融数据增强**：它能生成符合真实分布的高熵欺诈样本，而非简单线性插值噪声。
3. **Gumbel-Softmax 是处理离散金融特征的关键技术**，解决了 MCC 等字段的不可导难题。
4. **条件触发的 SHAP 机制实现了 XAI 与低延迟的兼容**，满足 GDPR 合规要求而不影响主流程。
5. **HITL + 异步重训练机制形成闭环学习系统**，能够持续适应新型欺诈策略，缓解 concept drift。

---

### 方法的局限性
- **依赖高质量标注数据用于 HITL 验证**：若专家反馈延迟或错误，会影响模型进化速度；
- **WGAN-GP 存在 mode collapse 风险**：虽有梯度惩罚缓解，但仍可能生成有限类型的欺诈样本；
- **未完全开源或在标准 benchmark 上评测**：缺乏与其他 SOTA 方法的直接数值对比；
- **冷启动问题**：初始阶段缺乏足够合法交易数据构建准确的 “legitimate manifold”。

---

### 未来工作方向（作者明确提出）
1. **Privacy-Preserving Decentralization**  
   → 探索将框架迁移至 **Federated Learning** 环境，允许多家金融机构协作训练 WGAN-GP，同时保护原始交易隐私。

2. **Adaptive Thresholding via DRL**  
   → 使用 **Deep Reinforcement Learning (DRL)** 动态调整异常阈值 $ T $，响应网络拥堵或欺诈潮波动。

3. **Cross-Institutional Generalizability**  
   → 测试框架在不同银行系统的迁移能力，验证“合法交易流形”的通用性与鲁棒性。

---

> **总结一句话**：本文提出的 Dual-Path Generative Framework 在保证 <50ms 推理延迟的同时，通过 VAE+WGAN-GP+Gumbel-Softmax+Triggered SHAP 的协同机制，实现了对 zero-day 银行欺诈的高效、可解释、可持续演进的检测，为现代金融风控提供了兼具性能与合规性的新范式。

</details>

---

### 14. [PrototypeNAS: Rapid Design of Deep Neural Networks for Microcontroller Units](https://arxiv.org/abs/2603.15106)

**Authors**: Mark Deutel, Simon Geis, Axel Plinge  
**Category**: cs.AI  
**Published**: 2026-03-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.15106v1  

#### Abstract
Enabling efficient deep neural network (DNN) inference on edge devices with different hardware constraints is a challenging task that typically requires DNN architectures to be specialized for each device separately. To avoid the huge manual effort, one can use neural architecture search (NAS). Howe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：PrototypeNAS: Rapid Design of Deep Neural Networks for Microcontroller Units

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在资源受限的边缘设备（特别是 **Microcontroller Units, MCUs**）上部署高效的深度神经网络（DNN）是一个重大挑战。传统方法需要为每种硬件单独设计和训练大量模型，耗时且计算成本高昂。现有的 **Neural Architecture Search (NAS)** 方法虽然能自动化架构设计，但通常存在以下问题：
- 需要从头训练大量候选模型，资源消耗大；
- 忽略目标系统的实际资源约束（如 RAM、ROM、FLOPs）；
- 多数基于单一超网（supernet），搜索空间受限。

### 🚀 提出的新方法：PrototypeNAS
作者提出 **PrototypeNAS** —— 一种**零样本（zero-shot）多目标优化（MOO）框架**，用于快速、自动地为不同 MCU 设计专用 DNN 架构。

#### 创新点：
1. **三步解耦流程**：
   - 将 DNN 的设计探索（architecture exploration）与训练过程解耦，大幅减少训练开销。
   - 只需最终训练 **3–5 个候选模型**，即可获得高性能架构。

2. **统一的搜索空间（Search Space）**：
   - 同时优化多种因素：**baseline architecture 选择、结构优化（kernel & stride）、宽度缩放（width multiplier）、剪枝（pruning sparsity）、量化配置**。
   - 支持跨架构类型联合搜索（如 MobileNetV2 vs ResNet vs SqueezeNet），避免局限于单一超网。

3. **零样本代理集成（Ensemble of Zero-Shot Proxies）作为 MOO 目标**：
   - 使用四个不同的 zero-shot proxy（MeCo, ZiCo, NASWOT, SNIP）来预测未训练模型的潜在精度。
   - 不将多个 proxy 加权合并成单一分值，而是让它们作为**独立的优化目标参与 MOO**，提升鲁棒性和多样性。

4. **Hypervolume Subset Selection 进行 Pareto 剪枝**：
   - 在得到 Pareto 前沿后，使用基于进化算法的 Hypervolume 子集选择策略，从中提取最具代表性的 **3–5 个折中方案**，便于决策者选择。

### 🔍 相比现有方法的优势
| 特性 | PrototypeNAS | 典型 NAS 方法（如 TinyNAS / NATS-Bench） |
|------|---------------|------------------------------------------|
| 是否需要训练数百模型 | ❌ 否（仅训练 ~5 个） | ✅ 是（每个候选都需训练） |
| 搜索空间灵活性 | ✅ 跨架构 + 结构 + 压缩联合优化 | ⚠️ 通常限于单一超网内子网络 |
| 资源感知能力 | ✅ 强（直接建模 RAM/ROM/FLOPs 约束） | ✅ 部分支持 |
| 优化效率 | ⏱️ 分钟级完成搜索 | ⏳ 数小时至数天 |
| 准确率表现 | ✅ 更高（尤其在 CIFAR10 上 +5%） | ⬆️ 中等 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
共涵盖 **12 个数据集**，分布在三大任务中：

| 任务 | 数据集 |
|------|--------|
| **Image Classification** | CIFAR10, CIFAR100, GTSRB, Flowers, Birds, Cars, Pets, ArxPhotos314 |
| **Time Series Classification** | Daliac, MAFULDA, BitBrain Sleep |
| **Object Detection** | COCO（person detection 子集） |

所有实验均以 **ARM Cortex-M MCU**（具体为 iMXRT1062 Cortex-M7）为目标平台进行部署验证。

### ⚙️ 实验设置
- **搜索阶段**：
  - 执行 **500 次采样 trial** 的多目标优化（MOO）。
  - 目标函数：最小化 FLOPs，最大化 4 个 zero-shot proxy 得分。
  - 约束条件：RAM ≤ 256 kB, ROM ≤ 1 MB, FLOPs ≤ 200 MFLOPs（视情况调整）。
- **选择阶段**：
  - 对 Pareto 前沿应用 **Hypervolume Subset Selection**，选出 top-5 模型。
- **评估阶段**：
  - 对选出的 5 个模型进行完整训练、剪枝和量化（PTQ 或 QAT）。
  - 图像分类任务使用 **ImageNet 预训练 50 轮 + CIFAR 微调 100 轮**。
  - 时间序列任务采用随机初始化并使用 **QAT（Quantization-Aware Training）**。
  - 检测任务基于 **YOLOv5 + MbedNet backbone** 实现。

### 📊 评估指标
| 指标 | 描述 |
|------|------|
| **Test Accuracy (%)** | 量化后的测试准确率 |
| **FLOPs (MFLOPs)** | 推理计算量 |
| **ROM [kB]** / **RAM [kB]** | 模型存储大小与运行内存占用 |
| **Latency [ms]** | 在 iMXRT1062 上的平均推理延迟 |
| **Energy [mJ]** | 单次推理能耗 |
| **mAP50** | 物体检测任务中 IoU=0.5 下的平均精度 |

### 🔁 基线方法对比
- **TinyNAS (MCUNet)** [22]：专为 MCU 设计的两阶段 NAS 方法，基于 MobileNet 类型的超网。
- **NATS-Bench** [8]：包含两个搜索空间：
  - TSS（Topology Search Space）：关注层操作组合；
  - SSS（Size Search Space）：关注通道数量变化。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table 2 & 3）

#### ✅ 图像分类结果（CIFAR10 最佳模型）
| 指标 | 数值 |
|------|------|
| **Accuracy** | **93.7%** |
| **FLOPs** | 151.0 MFLOPs |
| **ROM** | 356.5 kB |
| **RAM** | 189.7 kB |
| **Latency** | 753.2 ms |
| **Energy** | 254.5 mJ |

> ✅ 所有模型均可成功部署于标准 Cortex-M MCU。

#### ✅ 时间序列分类最佳表现（Mafaulda）
| 指标 | 数值 |
|------|------|
| **Accuracy** | **98.5%** |
| **FLOPs** | 215.3 MFLOPs |
| **Latency** | 635.0 ms |

#### ✅ 目标检测（Person Detection on COCO）
- 输入分辨率 128×128 → 达到约 **0.4 mAP50**
- 输入分辨率 320×320 → 提升至 **>0.5 mAP50**
- 显示 PrototypeNAS 可扩展至非分类任务。

---

### 🔁 与基线方法对比（Fig. 6）

#### vs. **MCUNet (TinyNAS)** on CIFAR10
- PrototypeNAS 的五个模型在相同 FLOPs 水平下，**平均高出 5% 的准确率**。
- 表明其搜索空间更具表达力，能够找到“知识密度”更高的架构。

#### vs. **NATS-Bench**
- PrototypeNAS 找到的模型性能与 NATS-Bench 中最优者相当。
- 但 **训练成本极低**：
  - NATS-Bench：需训练 **48,393 个模型 × 最多 200 epoch**
  - PrototypeNAS：仅训练 **5 个模型 × 100 epoch**
- 效率提升超过 **三个数量级**。

---

### 🔍 消融实验与分析（Section 4.2 Proxy Ensemble Analysis）

#### 发现：单一 zero-shot proxy 不可靠
- 计算 **Kendall’s τ 相关系数** 发现：
  - 所有 proxy 与真实准确率的相关性不稳定；
  - 有些 proxy 在某些数据集上有负相关（如 SNIP 在 GTSRB 上 τ = -0.4）；
  - **没有任何一个 proxy 在所有任务上表现一致良好**。

#### 结论支持使用 **proxy ensemble**
- 多个 proxy 提供互补视角，降低偏差风险；
- 将其作为独立目标而非加权融合，更有利于发现多样化的高质量架构；
- **FLOPs 本身与准确率几乎无相关性（τ ≈ 0）**，说明不能仅靠轻量化选模型。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **零样本 NAS 完全可行且高效**：
   - 通过合理的搜索空间设计和 proxy ensemble，可在无需训练的情况下有效探索数百种架构。
2. **统一优化架构+压缩策略优于分步优化**：
   - 联合优化 architecture selection、pruning、quantization 配置可发现更好的 trade-off。
3. **Hypervolume Subset Selection 有效提炼 Pareto 解集**：
   - 成功将大型 Pareto 前沿压缩为 3–5 个有意义的选择，极大简化部署决策。
4. **跨架构搜索带来更高性能上限**：
   - 相比局限于单一超网的方法（如 MCUNet），PrototypeNAS 能发掘更优的知识分布模式。

### ⚠️ 局限性
1. **依赖预定义 baseline architectures**：
   - 无法生成全新结构（如完全不同于 CNN 的模块），仍属“微调”范畴。
2. **zero-shot proxy 仍有噪声**：
   - 尽管 ensemble 缓解了问题，但 proxy 与真实性能之间仍存在偏差。
3. **当前实现聚焦于特定 MCU 平台**：
   - 需针对不同硬件重新设定约束边界，尚未完全通用化。

### 🔮 未来工作方向
- 扩展至更多类型的传感器输入（如音频、雷达）；
- 引入硬件在环（hardware-in-the-loop）反馈进一步校准 proxy；
- 探索动态架构生成机制，突破固定 superblock 结构限制；
- 构建端到端工具链，支持一键式“数据 → MCU 部署”。

---

## 总结
📌 **PrototypeNAS 是一个高效、实用、面向 MCU 的零样本 NAS 框架**。它通过**三步解耦流程 + 多目标 proxy ensemble + Hypervolume 子集选择**，实现了在几分钟内完成大规模架构探索，并仅需训练极少数模型即可获得高性能、可部署的 DNN。实验证明其在图像、时间序列和检测任务上均达到 SOTA 水平，显著优于 TinyNAS 和 NATS-Bench，在准确率和效率之间取得了卓越平衡。

</details>

---

### 15. [Intelligent Co-Design: An Interactive LLM Framework for Interior Spatial Design via Multi-Modal Agents](https://arxiv.org/abs/2603.15341)

**Authors**: Ren Jian Lim, Rushi Dai  
**Category**: cs.AI  
**Published**: 2026-03-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.15341v1  

#### Abstract
In architectural interior design, miscommunication frequently arises as clients lack design knowledge, while designers struggle to explain complex spatial relationships, leading to delayed timelines and financial losses. Recent advancements in generative layout tools narrow the gap by automating 3D ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Intelligent Co-Design: An Interactive LLM Framework for Interior Spatial Design via Multi-Modal Agents*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题
在建筑室内设计中，客户与设计师之间常因**沟通不畅**而产生误解：  
- 客户缺乏专业知识，难以准确表达空间意图；  
- 设计师难以用非技术语言解释复杂的空间关系。  

这导致设计周期延长、成本增加，并限制了非专业人士参与设计过程（即 **participatory design** 的缺失）。

现有自动化工具如 **rule-based 系统**（如 Infinigen Indoors）或 **data-driven 模型**（如 SceneDreamer）虽能生成布局，但存在以下缺陷：
- Rule-based 方法依赖硬编码规则，灵活性差；
- Data-driven 方法需要大量训练数据，泛化能力弱；
- 多数工具忽略用户交互，无法实现迭代优化。

---

### 🚀 提出的新方法与创新思路
本文提出一个基于 **LLM 的多模态、多智能体框架（multi-modal, multi-agent system）**，将自然语言描述和参考图像动态转化为高质量的 3D 室内设计方案。

#### 核心架构包含四个专用 Agent：
| Agent | 功能 |
|------|------|
| **Reference Agent** | 分析输入图像，提取家具类型与空间关系作为上下文 |
| **Spatial Agent** | 结合用户需求与 RAG 获取的设计规范，生成家具选择、空间约束与评分项（spatial rules） |
| **Interactive Agent** | 将技术性输出转为通俗语言，支持实时人机对话与反馈 |
| **Grader Agent** | 利用视觉模型 LLaVa 对生成方案进行自动打分，用于闭环优化 |

该系统通过 **prompt engineering + Retrieval-Augmented Generation (RAG)** 实现无需额外训练即可集成专家知识，显著降低对标注数据的依赖。

---

### 🔍 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **包容性（Inclusivity）** | 支持自然语言交互，使无设计背景用户也能参与设计 |
| **灵活性（Flexibility）** | 不依赖大规模训练数据，适用于多种户型（resilient across typologies） |
| **可解释性（Explainability）** | 所有决策以文本形式记录，提升透明度 |
| **效率（Efficiency）** | 自动化重复任务，缩短设计周期 |
| **协作性（Collaboration）** | 支持“反思性实践”（reflection-in-action），允许持续迭代优化 |

> 💡 创新关键词：**Interactive LLM**, **Multi-Agent System**, **Participatory Design**, **RAG**, **Multimodal Input**

---

## 2. 核心实验方法和设置

### 📦 数据集与测试场景
未使用传统训练数据集，而是基于真实住宅户型图进行测试，来源包括：
- Alea Miami 公开户型目录（studio, 1B1B 等）
- 包括不同面积与类型的房间：  
  - 小户型（8–14 m²）  
  - 中等户型（22–23 m²）  
  - 多种功能需求（阅读、观影、用餐等）

共测试多个 floor plan 类型（见 Figure 4），验证框架的**跨类型适应能力**。

---

### ⚙️ 实验设置
- **前端界面**：Web UI 收集用户输入（文字 + 图像）
- **后端引擎**：基于 **Claude 4.0 Sonnet** 和 **LLaVa-1.5-7B-hf** 构建多 Agent 协作流程
- **集成渲染器**：接入开源 3D 合成工具 **Infinigen Indoors**，利用其模拟退火算法（simulated annealing）优化家具摆放
- **两种模式对比**：
  - **Interactive Mode**：用户参与每轮决策（对象选择、约束、评分项）
  - **Non-Interactive Mode**：仅由 LLM 内部推理完成，无用户干预

---

### 📊 评估指标
采用双轨评估策略：

#### （1）独立 LLM 评估（ChatGPT-o3）
由外部 LLM 对生成布局进行盲评，评分标准如下（每项满分10分）：

| 评估维度 | 描述 |
|--------|------|
| **User Intent Alignment** | 是否满足用户声明的功能需求（如“喜欢看书”） |
| **Aesthetic Coherence** | 风格统一性、比例协调、焦点明确 |
| **Functionality** | 家具尺寸合理、可达性良好、不影响门窗使用 |
| **Circulation** | 行走路径畅通，无阻塞或危险交叉 |

> 使用预定义 rubric 引导 GPT-o3 输出一致评分（见 Table 1）

#### （2）用户问卷调查
共收集 **53 名有效参与者** 反馈，涵盖设计师、工程师、医疗从业者等群体。  
问题覆盖五个方面：
1. 共创感（Sense of Co-Creation）
2. 透明性与可解释性
3. 易用性与表达自由度
4. 协作友好性
5. 反馈响应能力
6. 工具偏好（vs. SketchUp / RoomSketcher）

---

### 🆚 基线方法对比
| 方法 | 特点 |
|-----|------|
| **Proposed Method (Interactive)** | 用户参与 + 多 Agent + RAG + 自然语言交互 |
| **Baseline (Non-Interactive)** | 相同初始条件，但无用户反馈循环，完全由 LLM 自主决定 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 场景 | 方法 | 平均得分（/40） | 各项平均提升 |
|------|------|------------------|-------------|
| **Compact Living Room (14m²)** | Interactive | **30.0** | ↑ 1.25 |
| | Non-Interactive | 25.0 | — |
| **Spacious Living Room (23m²)** | Interactive | **31.0** | ↑ 1.25 |
| | Non-Interactive | 26.0 | — |
| **Micro Bedroom (8m²)** | Interactive | **27.0** | ↑ 1.25 |
| | Non-Interactive | 22.0 | — |

> 注：总分为四项指标（intent, aesthetics, functionality, circulation）之和，满分为40

#### 具体提升表现：
- **Circulation** 提升最明显（+2~3 分），说明用户反馈有效改善动线设计
- **User Intent Alignment** 显著增强，尤其在个性化需求（如“爱读书的人”）上体现更优理解
- **Aesthetic Coherence** 在互动模式下风格更统一，避免杂乱搭配

---

### 🗳️ 用户问卷结果
- **77% 的用户表示“满意”或“非常满意”**
- **89% 认为系统帮助他们摆脱专业术语表达偏好**
- 工具偏好分布：
  - **51% 偏好本框架**
  - 42% 无强烈偏好（部分愿结合传统软件使用）
  - 仅 **8% 坚持使用传统设计软件**

> 跨职业群体普遍认可：
> - 设计师中 **71% 愿意采用该框架**
> - 非设计背景用户（如医生、工程师）也表现出高接受度（60–71%）

---

### ❌ 无消融实验
文中未提供 ablation study（例如移除某个 Agent 或禁用 RAG 的影响），但通过对比交互 vs 非交互模式间接体现了用户反馈的价值。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **用户参与显著提升设计质量**  
   交互式流程使布局在 **intent alignment、functionality 和 circulation** 上全面优于纯 LLM 推理方案。

2. **多 Agent 架构有效分工协作**  
   专业化角色划分（Reference/Spatial/Interactive/Grader）提升了系统的模块化与可控性。

3. **RAG 减少数据依赖，增强实用性**  
   无需微调即可注入建筑设计规范，实现轻量级知识增强。

4. **自然语言接口促进包容性设计**  
   非专业人士可通过日常语言参与创作，推动 **democratization of design**。

5. **系统具备跨户型鲁棒性（resilience）**  
   在不同面积、形状、功能需求下均能生成合理布局，无需重新训练。

---

### ⚠️ 局限性
1. **视觉交互不足**  
   当前以文本为主，缺乏拖拽、草图、实时 3D 预览等直观操作方式。

2. **语言障碍**  
   缺乏多语言支持，限制全球适用性。

3. **过度依赖 simulated annealing**  
   优化算法固定，难以应对非常规空间逻辑（如悬浮家具、变形结构）。

4. **部分设计师希望保留控制权**  
   一些专业人士倾向“AI 辅助而非主导”，建议提供更多手动调节选项。

---

### 🔮 未来工作方向
1. **增强 multimodal interaction**  
   支持 sketch 输入、mood board 上传、语音指令等多元输入方式。

2. **扩展至 broader architectural scenarios**  
   应用于建筑外立面设计、城市规划、可持续设计等领域。

3. **改进 placement algorithm**  
   替代或增强 simulated annealing，引入强化学习或神经优化器。

4. **增加 multilingual support**  
   提升国际化可用性，服务更多语种用户。

5. **构建开放平台生态**  
   支持第三方插件、家具库集成、VR/AR 协同体验。

---

## 总结
> 本研究成功构建了一个**以人为中心、可交互、低数据依赖的 LLM 多智能体框架**，实现了从自然语言到高质量 3D 室内设计的端到端转化。实验证明，该方法不仅提升了设计效率与质量，更重要的是**让非专业人士真正参与到创造性过程中**，为未来的 **AI-driven participatory design** 提供了可行范式。

</details>

---

### 16. [Selective Fine-Tuning of GPT Architectures for Parameter-Efficient Clinical Text Classification](https://arxiv.org/abs/2603.14183)

**Authors**: Fariba Afrin Irany, Sampson Akwafuo  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.14183v1  

#### Abstract
The rapid expansion of electronic health record (EHR) systems has generated large volumes of unstructured clinical narratives that contain valuable information for disease identification, patient cohort discovery, and clinical decision support. Extracting structured knowledge from these free-text do...

---

### 17. [Towards Next-Generation LLM Training: From the Data-Centric Perspective](https://arxiv.org/abs/2603.14712)

**Authors**: Hao Liang, Zhengyang Zhao, Zhaoyang Han, Meiyi Qiang, Xiaochen Ma, Bohan Zeng, Qifeng Cai, Zhiyu Li, Linpeng Tang, Weinan E, Wentao Zhang  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.14712v1  

#### Abstract
Large language models (LLMs) have demonstrated remarkable performance across a wide range of tasks and domains, with data playing a central role in enabling these advances. Despite this success, the preparation and effective utilization of the massive datasets required for LLM training remain major ...

---

### 18. [Joint Routing and Model Pruning for Decentralized Federated Learning in Bandwidth-Constrained Multi-Hop Wireless Networks](https://arxiv.org/abs/2603.15188)

**Authors**: Xiaoyu He, Weicai Li, Tiejun Lv, Xi Yu  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.15188v1  

#### Abstract
Decentralized federated learning (D-FL) enables privacy-preserving training without a central server, but multi-hop model exchanges and aggregation are often bottlenecked by communication resource constraints. To address this issue, we propose a joint routing-and-pruning framework that optimizes rou...

---

### 19. [Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in Large Language Models](https://arxiv.org/abs/2603.14517)

**Authors**: Ying Xie  
**Category**: cs.AI  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.14517v1  

#### Abstract
Large language models (LLMs) suffer from proactive interference (PI): outdated information in the context window disrupts retrieval of current values. This interference degrades retrieval accuracy log-linearly as stale associations accumulate, a bottleneck that persists regardless of context length ...

---

### 20. [Efficient Document Parsing via Parallel Token Prediction](https://arxiv.org/abs/2603.15206)

**Authors**: Lei Li, Ze Zhao, Meng Li, Zhongwang Lun, Yi Yuan, Xingjing Lu, Zheng Wei, Jiang Bian, Zang Li  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.15206v1  

#### Abstract
Document parsing, as a fundamental yet crucial vision task, is being revolutionized by vision-language models (VLMs). However, the autoregressive (AR) decoding inherent to VLMs creates a significant bottleneck, severely limiting parsing speed. In this paper, we propose Parallel-Token Prediction (PTP...

---

### 21. [DOS: Dependency-Oriented Sampler for Masked Diffusion Language Models](https://arxiv.org/abs/2603.15340)

**Authors**: Xueyu Zhou, Yangrong Hu, Jian Huang  
**Category**: cs.CL  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.15340v1  

#### Abstract
Masked diffusion language models (MDLMs) have recently emerged as a new paradigm in language modeling, offering flexible generation dynamics and enabling efficient parallel decoding. However, existing decoding strategies for pre-trained MDLMs predominantly rely on token-level uncertainty criteria, w...

---

### 22. [Machine Learning Models to Identify Promising Nested Antiresonance Nodeless Fiber Designs](https://arxiv.org/abs/2603.13302)

**Authors**: Rania A. Eltaieb, Sophie LaRochelle, Leslie A. Rusch  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.13302v1  

#### Abstract
Hollow-core fibers offer superior loss and latency characteristics compared to solid-core alternatives, yet the geometric complexity of nested antiresonance nodeless fibers (NANFs) makes traditional optimization computationally prohibitive. We propose a high-efficiency, two-stage machine learning fr...

---

### 23. [Linear Predictability of Attention Heads in Large Language Models](https://arxiv.org/abs/2603.13314)

**Authors**: Khalid Shaikh, Asmit Kumar Singh, Rebecca Christopher Dsouza, Shikhar Shiromani  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.13314v1  

#### Abstract
Large language model (LLM) inference is increasingly bottlenecked by the Key-Value (KV) cache, yet the fine-grained structure of attention-head activations remains poorly understood. We show that pretrained Transformers exhibit a pervasive inter-head linear structure: for a given token, the Query, K...

---

### 24. [Generalization and Memorization in Rectified Flow](https://arxiv.org/abs/2603.13421)

**Authors**: Mingxing Rao, Daniel Moyer  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.13421v1  

#### Abstract
Generative models based on the Flow Matching objective, particularly Rectified Flow, have emerged as a dominant paradigm for efficient, high-fidelity image synthesis. However, while existing research heavily prioritizes generation quality and architectural scaling, the underlying dynamics of how RF ...

---

### 25. [Improving Channel Estimation via Multimodal Diffusion Models with Flow Matching](https://arxiv.org/abs/2603.13440)

**Authors**: Xiaotian Fan, Xingyu Zhou, Le Liang, Xiao Li, Shi Jin  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.13440v1  

#### Abstract
Deep generative models offer a powerful alternative to conventional channel estimation by learning complex channel distributions. By integrating the rich environmental information available in modern sensing-aided networks, this paper proposes MultiCE-Flow, a multimodal channel estimation framework ...

---

### 26. [PDE-SSM: A Spectral State Space Approach to Spatial Mixing in Diffusion Transformers](https://arxiv.org/abs/2603.13663)

**Authors**: Eshed Gal, Moshe Eliasof, Siddharth Rout, Eldad Haber  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.13663v1  

#### Abstract
The success of vision transformers-especially for generative modeling-is limited by the quadratic cost and weak spatial inductive bias of self-attention. We propose PDE-SSM, a spatial state-space block that replaces attention with a learnable convection-diffusion-reaction partial differential equati...

---

### 27. [CAMD: Coverage-Aware Multimodal Decoding for Efficient Reasoning of Multimodal Large Language Models](https://arxiv.org/abs/2603.14745)

**Authors**: Huijie Guo, Jingyao Wang, Lingyu Si, Jiahuan Zhou, Changwen Zheng, Wenwen Qiang  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.14745v1  

#### Abstract
Recent advances in Multimodal Large Language Models (MLLMs) have shown impressive reasoning capabilities across vision-language tasks, yet still face the challenge of compute-difficulty mismatch. Through empirical analyses, we identify that existing decoding methods may waste compute on easy cases w...

---

### 28. [OpenReservoirComputing: GPU-Accelerated Reservoir Computing in JAX](https://arxiv.org/abs/2603.14802)

**Authors**: Jan Williams, Dima Tretiak, Steven L. Brunton, J. Nathan Kutz, Krithika Manohar  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.14802v1  

#### Abstract
OpenReservoirComputing (ORC) is a Python library for reservoir computing (RC) written in JAX (Bradbury et al. 2018) and Equinox (Kidger and Garcia 2021). JAX is a Python library for high-performance numerical computing that enables automatic differentiation, just-in-time (JIT) compilation, and GPU/T...

---

### 29. [Dataset Distillation Efficiently Encodes Low-Dimensional Representations from Gradient-Based Learning of Non-Linear Tasks](https://arxiv.org/abs/2603.14830)

**Authors**: Yuri Kinoshita, Naoki Nishikawa, Taro Toyoizumi  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.14830v1  

#### Abstract
Dataset distillation, a training-aware data compression technique, has recently attracted increasing attention as an effective tool for mitigating costs of optimization and data storage. However, progress remains largely empirical. Mechanisms underlying the extraction of task-relevant information fr...

---

### 30. [Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling](https://arxiv.org/abs/2603.15279)

**Authors**: Aram Davtyan, Leello Tadesse Dadi, Volkan Cevher, Paolo Favaro  
**Category**: cs.LG  
**Published**: 2026-03-17  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.15279v1  

#### Abstract
Conditional Flow Matching (CFM), a simulation-free method for training continuous normalizing flows, provides an efficient alternative to diffusion models for key tasks like image and video generation. The performance of CFM in solving these tasks depends on the way data is coupled with noise. A rec...

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
