# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-01-19 06:04:05 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [HOSL: Hybrid-Order Split Learning for Memory-Constrained Edge Training](https://arxiv.org/abs/2601.10940)

**Authors**: Aakriti, Zhe Li, Dandan Liang, Chao Huang, Rui Li, Haibo Yang  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2601.10940v1  

#### Abstract
Split learning (SL) enables collaborative training of large language models (LLMs) between resource-constrained edge devices and compute-rich servers by partitioning model computation across the network boundary. However, existing SL systems predominantly rely on first-order (FO) optimization, which...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：HOSL: Hybrid-Order Split Learning for Memory-Constrained Edge Training

---

## 1. 论文的主要贡献和创新点

### 解决的问题
在 **Split Learning (SL)** 框架中，边缘设备（client）与服务器协同训练大型语言模型（LLMs），通过将模型分割实现隐私保护和计算卸载。然而，现有方法存在以下瓶颈：

- **First-Order (FO) 优化**：虽然收敛快、性能好，但需要存储前向传播的中间激活（activations）用于反向传播，导致客户端内存开销大，违背了模型分区以节省资源的初衷。
- **Zeroth-Order (ZO) 优化**：无需反向传播，仅依赖函数值估计梯度，显著降低内存占用，但在高维场景下收敛慢、性能差。

因此，**如何在保证低内存消耗的同时维持快速收敛和高性能**，是当前 SL 在边缘设备部署中的核心挑战。

---

### 提出的新方法：HOSL（Hybrid-Order Split Learning）
作者提出 **HOSL** ——一种混合阶次的 Split Learning 框架，其核心思想是：

> **在客户端采用 ZO 优化，在服务器端采用 FO 优化**，形成“**客户端轻量、服务端高效**”的异构优化策略。

#### 创新点：
1. **混合优化架构设计**：
   - 客户端使用 ZO-SGD 更新参数，避免存储激活和构建计算图，极大减少内存占用。
   - 服务器端使用标准 FO-SGD 进行反向传播，确保整体训练的快速收敛和高精度。

2. **通信与同步机制**：
   - 客户端对本地参数进行多次随机扰动（perturbation），执行多个前向传递，并将中间输出发送给服务器。
   - 服务器始终处于推理模式（inference mode），只返回标量损失值，不参与梯度计算。
   - 所有扰动完成后，客户端恢复原始参数并估算 ZO 梯度；随后发起一次含梯度跟踪的前向传递，触发服务器端的 FO 反向更新。

3. **理论支持**：
   - 给出了非凸目标下的收敛性分析，证明 HOSL 收敛速率为 $ O(\sqrt{d_c}/TQ) $，其中 $ d_c $ 是客户端参数维度，而非整个模型维度 $ d $。
   - 表明：**将更多计算卸载到服务器可提升收敛速度**。

---

### 相比现有方法的优势
| 方法 | 内存效率 | 收敛速度 | 性能表现 | 是否适合边缘设备 |
|------|----------|-----------|------------|------------------|
| FO-FO | ❌ 高内存需求 | ✅ 快 | ✅ 高 | 否 |
| ZO-ZO | ✅ 极低内存 | ❌ 慢 | ❌ 较低 | ✅ 是 |
| **HOSL (ZO-FO)** | ✅ 接近 ZO-ZO | ✅ 接近 FO-FO | ✅ 显著优于 ZO-ZO | ✅✅ 最优 |

> HOSL 成功平衡了 **memory efficiency** 与 **optimization effectiveness**，实现了“鱼与熊掌兼得”。

---

## 2. 核心实验方法和设置

### 使用的数据集
在 **GLUE 和 SuperGLUE** 子任务上进行评估，共6个下游任务：
- **SST-2**（情感分类）
- **CB**（自然语言推断）
- **WIC**（词义消歧）
- **WSC**（指代消解）
- **BoolQ**（是非问答）
- **RTE**（文本蕴含）

---

### 模型与训练配置
- **基础模型**：OPT 系列
  - OPT-125M（1.25亿参数）
  - OPT-1.3B（13亿参数）
- **微调方式**：
  - Full-parameter fine-tuning（全参数微调）
  - LoRA-based parameter-efficient fine-tuning（低秩适配）
- **模型切分位置**：第5层后分割
  - OPT-125M：客户端5层，服务器7层
  - OPT-1.3B：客户端5层，服务器19层
- **硬件平台**：NVIDIA A100 GPU（40GB显存）

---

### 评估指标
1. **测试准确率（Test Accuracy）**
2. **峰值 GPU 内存消耗（Peak GPU Memory Usage）**
   - 分别报告客户端（CGPU）和服务端（SGPU）
3. **准确性-内存权衡（Accuracy-Memory Trade-off）**

---

### 基线方法对比
比较三种优化组合：
| 方法 | 客户端优化器 | 服务器优化器 | 特点 |
|------|---------------|---------------|------|
| **ZO-ZO** | ZO-SGD | ZO-SGD | 内存最小，但性能最差 |
| **FO-FO** | FO-SGD | FO-SGD | 性能最好，但客户端内存高 |
| **HOSL (Ours)** | ZO-SGD | FO-SGD | 本文提出的方法 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### ✅ 客户端内存大幅下降
- **相比 FO-FO 基线，HOSL 减少客户端 GPU 内存最多达 3.7×**
  - OPT-125M 上 BoolQ 任务：从 **8.67 GB → 2.36 GB**
  - OPT-1.3B 上 BoolQ 任务：从 **15.41 GB → 7.23 GB**
- 内存水平与 ZO-ZO 相当，远低于 FO-FO

#### ✅ 准确率显著优于 ZO-ZO
- **最高提升达 15.55%**
  - OPT-1.3B + RTE 任务：ZO-ZO 为 58.73%，HOSL 达到 **74.28%**
- 在所有任务上均一致超越 ZO-ZO 基线

#### ✅ 与 FO-FO 基线差距极小
- 相比 FO-FO，准确率损失仅为：
  - **全参数微调**：0.41% ~ 4.23%
  - **LoRA 微调**：0.20% ~ 2.05%
- 实现了几乎无损的性能保留

#### ✅ LoRA 场景同样有效
- 即使在参数高效微调场景下，HOSL 仍保持优势：
  - 内存接近 ZO-ZO
  - 准确率明显高于 ZO-ZO（+0.60% ~ 13.16%）

---

### 消融实验与关键观察（隐含于主实验）
- **增加扰动次数 Q 可提升稳定性与收敛性**：验证了多点梯度估计的有效性。
- **减小客户端模型规模 $ d_c $ 可加速收敛**：符合理论预测 $ O(\sqrt{d_c}/TQ) $
- **KV Cache 成为主要内存项之一**：因多次前向传递需缓存 Key/Value，提示未来可优化缓存策略。

---

## 4. 关键结论和发现

### 主要发现
1. **HOSL 成功解决了 SL 中内存与性能之间的根本矛盾**：
   - 在客户端实现 ZO 级别的内存效率（≈ ZO-ZO）
   - 在全局达到接近 FO-FO 的收敛速度和最终性能

2. **理论收敛速率仅依赖客户端参数维度 $ d_c $**，而非完整模型维度 $ d $，说明：
   - 更多计算卸载至服务器有助于提升收敛速度
   - 支持“越小的客户端模型越好”的切分策略

3. **实验验证了 HOSL 的普适性和有效性**：
   - 跨两个模型尺度（125M / 1.3B）
   - 多种任务类型（分类、推理、问答等）
   - 不同微调范式（全参数 / LoRA）

---

### 方法的局限性
1. **训练时间可能增加**：
   - ZO 梯度估计需要 $ 2Q $ 次前向传递（如 $ Q=10 $ 则为20次），显著增加通信轮次和 wall-clock 时间。
   
2. **未考虑联邦学习中的数据异构性**：
   - 当前分析基于单客户端设定，尚未扩展到多客户端、Non-IID 数据分布场景。

3. **KV Cache 开销不可忽略**：
   - 多次扰动带来的重复前向传递增加了 KV 缓存压力，影响实际部署效率。

---

### 未来工作方向
1. **优化扰动机制**：
   - 探索更高效的梯度估计方法（如稀疏扰动、结构化扰动）以减少前向次数。
   
2. **引入压缩或量化技术**：
   - 对扰动种子或损失反馈进行编码压缩，降低通信成本。

3. **扩展至 Split Federated Learning（SplitFL）框架**：
   - 结合 HOSL 与联邦学习，解决多客户端下的异构性与通信效率问题。

4. **动态模型切分策略**：
   - 根据设备资源动态调整 $ d_c $，最大化性能-资源权衡。

---

## 总结
> **HOSL 是首个将 ZO 与 FO 优化有机结合的 Split Learning 框架，在理论上证明了其收敛优势，并在实践中实现了“低内存 + 高性能”的理想特性，为 LLM 在边缘设备上的高效微调提供了可行路径。**

</details>

---

### 2. [Mugi: Value Level Parallelism For Efficient LLMs](https://arxiv.org/abs/2601.10823)

**Authors**: Daniel Price, Prabhu Vellaisamy, John Shen, Di Wu  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2601.10823v1  

#### Abstract
Value level parallelism (VLP) has been proposed to improve the efficiency of large-batch, low-precision general matrix multiply (GEMM) between symmetric activations and weights. In transformer based large language models (LLMs), there exist more sophisticated operations beyond activation-weight GEMM...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Mugi: Value Level Parallelism For Efficient LLMs*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对当前大型语言模型（LLMs）推理效率低下的挑战，指出现有基于 **Value Level Parallelism (VLP)** 的硬件架构（如 Carat）存在以下四大局限：

1. **不支持非线性操作**：现有 VLP 架构仅优化 GEMM 运算，无法高效处理 LLM 中关键的非线性函数（如 softmax、SiLU、GELU）。
2. **不兼容非对称量化**：主流 LLM 推理采用 BF16-INT4 的非对称量化（如 WOQ 和 KVQ），而传统 VLP 设计（如 Carat）仅支持对称的 FP8。
3. **小批量输入效率低下**：LLM 解码阶段通常为小批量（如 batch=8），而传统 VLP 针对大批次设计，导致资源利用率低。
4. **可持续性差**：专用的非线性计算单元增加了芯片面积和制造过程中的 **embodied carbon**，不利于可持续计算。

### 提出的新方法与创新
作者提出 **Mugi**，一种全新的 VLP 架构，首次将 VLP 扩展到完整的 LLM 工作负载，其核心创新如下：

- **VLP 非线性近似（VLP Nonlinear Approximation）**：
  - 首次将 VLP 应用于非线性函数（softmax、SiLU、GELU）的硬件加速。
  - 采用 **input approximation**（输入近似）而非传统的 output approximation，结合 **value-centric** 设计，对重要值（如接近 0 的指数）分配更高精度。
  - 利用 **滑动窗口（sliding window）** 动态适应不同层的输入分布，提升精度。

- **优化非对称、小批量 GEMM**：
  - 支持 **BF16-INT4** 的非对称 GEMM，适配 WOQ 和 KVQ 优化。
  - 通过定制化映射（权重在行，激活在列）和广播机制，高效支持小批量输入和 **Grouped Query Attention (GQA)**。

- **统一架构设计**：
  - 复用同一套 VLP 计算阵列同时执行 GEMM 和非线性操作，避免专用向量单元，显著降低芯片面积和功耗。

### 相比现有方法的优势
- **更高的吞吐量和能效**：在非线性操作和完整 LLM 工作负载上均显著优于现有架构。
- **更好的可持续性**：通过资源共享减少芯片面积，同时降低 **operational carbon** 和 **embodied carbon**。
- **更强的通用性**：支持现代 LLM 的主流优化技术（WOQ、KVQ、GQA）。

---

## 2. 核心实验方法和设置

### 数据集
实验基于多个主流 LLM 和视觉 Transformer 模型，具体见 **Table 1**：
- **LLaMA2**：7B, 13B, 70B
- **Whisper**：tiny, large
- **SwinV2**：tiny, large
- **ViViT**：base

### 实验设置与评估指标
- **硬件模拟器**：基于公开的 Carat 架构构建事件驱动的周期级模拟器，RTL 综合于 45nm 工艺，频率 400MHz。
- **评估指标**：
  - **Perplexity / Loss**：衡量模型精度。
  - **Throughput (Tokens/s)**：吞吐量。
  - **Energy Efficiency (Tokens/s/pJ)**：能效。
  - **Power Efficiency (Tokens/s/W)**：功率效率。
  - **Area (mm²)**：芯片面积。
  - **Carbon Emissions**：操作碳排放（operational carbon）和制造碳排放（embodied carbon）。

### 基线方法对比
- **非线性近似基线**：
  - **Precise Vector Array (VA-FP)**：精确计算。
  - **Approximate Vector Array (VA-AP)**：近似计算。
  - **Piecewise Linear (PWL)**：分段线性近似。
  - **Taylor Series**：泰勒展开近似。
- **GEMM 加速器基线**：
  - **Carat**：原始 VLP 架构。
  - **Systolic Array (SA)** / **SIMD Array (SD)**：传统矩阵计算架构。
  - **FIGNA**：支持 FP-INT GEMM 的定制化 PE。
  - **Tensor Core**：NVIDIA Hopper GPU 的张量核。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 非线性操作性能（图 11）
- **Softmax**：
  - 吞吐量：**45×** 高于精确向量阵列（VA-FP）。
  - 能效：**750×** 高于 VA-FP。
- **SiLU**：
  - 吞吐量：**48×** 高于 VA-FP。
  - 能效：**668×** 高于 VA-FP。

#### 完整 LLM 工作负载性能（表 3，LLaMA2-70B + GQA）
- **吞吐量**：相比 16×16 Systolic Array，Mugi (256) 提升 **2.07×**。
- **能效**：提升 **3.11×**。
- **功率效率**：提升 **1.50×**。
- 在 4×4 NoC 多节点配置下，吞吐量达 **22.19 Tokens/s**，远超 Tensor Core（10.06）和 Systolic Array（10.74）。

#### 碳排放（图 15）
- **操作碳排放（Operational CO₂eq）**：降低 **1.45×**。
- **制造碳排放（Embodied CO₂eq）**：降低 **1.48×**。

### 与基线方法对比
- **vs Carat**：Mugi 在相同阵列大小下能效更高，且支持非线性操作。
- **vs Systolic/SIMD**：Mugi 通过 VLP 消除乘法器，能效优势显著。
- **vs Tensor Core**：尽管 Tensor Core 单节点性能高，但 Mugi 在多节点扩展中表现更优，且能效更高。

### 消融实验（隐含在分析中）
- **缓冲区最小化**：通过广播和输出缓冲区精简（output buffer leaning），缓冲区面积降低 **4.5×**。
- **统一架构优势**：Mugi-L（专用 LUT 处理非线性）面积更大、功耗更高，验证了共享 VLP 阵列的优越性（图 13）。

---

## 4. 关键结论和发现

### 主要发现
1. **VLP 可成功扩展至非线性操作**：通过 input approximation 和 value-centric 设计，Mugi 在保持高精度的同时极大提升了非线性函数的执行效率。
2. **统一架构显著提升能效与可持续性**：复用 GEMM 阵列处理非线性操作，避免了专用硬件开销，是实现高效、绿色 LLM 推理的关键。
3. **Mugi 全面适配现代 LLM 优化趋势**：对 WOQ、KVQ、GQA 等技术的良好支持，使其在真实场景中具有极强竞争力。

### 方法的局限性
- **不支持所有非线性操作**：如 Layer Normalization 和 Rotary Positional Embeddings (RoPE) 尚未原生支持，需外部处理。
- **MoE 和多模态模型未完全验证**：虽推测可扩展，但未在论文中进行完整实验。
- **离线预计算 LUT**：当前 LUT 为离线生成，runtime 分布漂移可能影响精度，未来可探索在线调整机制。

### 未来工作方向
- 支持更多非线性操作（如 RoPE）。
- 探索 **online LUT adaptation** 以应对运行时分布变化。
- 扩展至 **Mixture-of-Experts (MoE)** 和 **multi-modal models** 的全面验证与优化。
- 探索更激进的量化方案（如 1-bit 权重）与 Mugi 的结合。

---

</details>

---

### 3. [Toward Adaptive Grid Resilience: A Gradient-Free Meta-RL Framework for Critical Load Restoration](https://arxiv.org/abs/2601.10973)

**Authors**: Zain ul Abdeen, Waris Gill, Ming Jin  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2601.10973v1  

#### Abstract
Restoring critical loads after extreme events demands adaptive control to maintain distribution-grid resilience, yet uncertainty in renewable generation, limited dispatchable resources, and nonlinear dynamics make effective restoration difficult. Reinforcement learning (RL) can optimize sequential d...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Toward Adaptive Grid Resilience: A Gradient-Free Meta-RL Framework for Critical Load Restoration*

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**极端事件后配电网关键负荷恢复（Critical Load Restoration, CLR）**中的挑战，旨在解决以下核心难题：
- **不确定性高**：可再生能源（如光伏、风电）出力受天气影响，存在显著的预测误差。
- **泛化能力差**：传统强化学习（RL）方法在特定场景下训练后，难以适应新的停电配置、负荷分布或发电模式，需要大量重新训练或微调。
- **计算复杂度高**：基于优化的方法（如MPC）在大规模系统中求解混合整数非线性规划（MILP）问题计算成本高昂，难以满足实时性要求。

### 提出的新方法
作者提出了一种名为 **MGF-RL (Meta-guided Gradient-Free Reinforcement Learning)** 的新型框架，其核心创新点如下：
- **梯度无关的元强化学习框架**：将**进化策略（Evolution Strategies, ES-RL）**作为任务内的策略优化器，避免了对环境动力学或策略网络的梯度计算，适用于非可微仿真环境（如OpenDSS）。
- **一阶元更新机制**：采用**一阶元学习更新**（类似MAML的一阶近似），替代了传统元学习中昂贵的二阶梯度计算（如Hessian矩阵），大幅降低了计算开销。
- **元初始化与快速适应**：通过在多个历史停电任务上进行元训练，学习一个通用的“元初始化”（meta-initialization）。当面对新的、未见过的停电场景时，仅需少量微调（fine-tuning）即可快速适应，实现高效部署。

### 相比现有方法的优势
- **更强的泛化能力**：相比标准RL和MAML，MGF-RL能有效泛化到全新的停电场景和可再生能源出力模式。
- **更高的计算效率**：避免了二阶梯度计算和复杂的在线优化，训练和推理速度更快，更适合实时应用。
- **更好的鲁棒性**：在可再生能源预测误差较大的情况下，仍能保持稳定的性能。
- **更低的微调成本**：仅需2-4个微调episode即可适应新任务，而标准RL通常需要15-20个。

---

## 2. 核心实验方法和设置

### 数据集与测试系统
- **测试平台**：在两个标准配电网模型上进行验证：
  - **Modified IEEE 13-bus system**：包含15个关键负荷和4类DERs（电池储能、光伏、风机、微型燃气轮机）。
  - **Modified IEEE 123-bus system**：更大规模系统，用于验证可扩展性。
- **可再生能源数据**：使用 **NREL WIND Toolkit** 的历史数据生成真实的风光出力序列，并通过合成方法引入可控的预测误差（0%-25%）。
- **任务生成**：共生成60个不同的CLR任务，其中32个用于元训练，其余用于测试泛化能力。

### 实验设置
- **控制周期**：5分钟（T=1/12小时），总恢复时长为6小时（72个时间步）。
- **状态空间**：包含各负荷的恢复水平、DERs的荷电状态（SOC）、剩余燃料、可再生能源预测值、当前时间步等。
- **动作空间**：控制DERs的有功/无功功率设定点以及关键负荷的恢复量。
- **奖励函数**：综合考虑高优先级负荷恢复、电压越限惩罚和负荷频繁投切惩罚。

### 评估指标
- **学习效率**：微调过程中的累积奖励曲线、初始性能提升（Jump-start performance, △ini）、渐近奖励增益（Asymptotic reward gain, △R）。
- **系统可靠性**：
  - **SAIDI (System Average Interruption Duration Index)**：平均每个用户的停电持续时间。
  - **90%恢复时间**：达到90%负荷恢复所需的时间。
  - **总负荷恢复比例**。
- **理论分析**：推导了任务平均最优性差距（TAOG）的次线性遗憾界（sublinear regret bounds），证明了算法在静态和动态环境下的收敛性。

### 基线方法对比
- **ES-RL**：非元学习的梯度无关RL，作为基准。
- **Warm-start RL**：将前一任务的最终策略作为下一任务的初始化。
- **MAML-RL**：最先进的元强化学习方法，使用TRPO进行任务内优化。
- **AC-RL (Automated Curriculum-based RL)**：基于课程学习的元RL方法。
- **MPC (Model Predictive Control)**：基于优化的基准方法，包括No Reserve MPC (NR-MPC) 和 Reserve Considered MPC (RC-MPC)。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **SAIDI 改善**：MGF-RL 在15个测试场景上的平均SAIDI为 **135.3分钟**，相比MPC（184.7分钟）、Warm-start RL（230.0分钟）和MAML（203.5分钟）分别提升了 **33%、41% 和 33%**。
- **90% 负荷恢复**：MGF-RL 是唯一能在所有测试场景中于6小时内成功恢复 **90%以上负荷** 的方法，平均耗时约 **275分钟**；其他方法均未能在此时间内达标。
- **总负荷恢复率**：MGF-RL 平均恢复了 **96%** 的负荷，远高于MAML（44%）、Warm-start RL（68%）和MPC（82%）。
- **微调效率**：MGF-RL 仅需 **2-4个微调episode** 即可适应新任务，而标准RL需要15-20个。

### 与基线方法的对比结果
- **性能全面领先**：在图5的学习曲线中，MGF-RL 展现出最高的累积奖励和最低的方差，表明其稳定性和优越的泛化能力。
- **初始优势明显**：在表3中，MGF-RL 在所有测试任务上的 △ini 和 △R 均为正值，说明其元初始化提供了强大的“跳跃启动”能力，并能持续改进。而MAML和AC-RL 经常出现负值，表现不如从零开始训练的ES-RL。
- **决策更稳健**：如图7所示，MPC 采取激进策略，早期恢复大量负荷，但在风光出力下降时被迫甩负荷，导致运行不稳定。而MGF-RL 采取保守策略，优先储备能量，平滑地逐步恢复负荷，避免了负荷循环，电压波动也更小。

### 消融实验与鲁棒性分析
- **预测误差影响**：在表5中，随着预测误差（Er）增加至25%，MPC 性能急剧下降（奖励从19.98降至12.86），而ES-RL（特别是2小时预测头）表现更稳健（从18.95降至16.64）。这验证了模型无关方法在不确定性下的优势。
- **预测时域选择**：图8b显示，在低误差时，长时域预测（K=4,6）更有利；但在高误差时，短时域（K=1,2）反而更鲁棒，体现了“预见性”与“可靠性”的权衡。
- **可扩展性**：表6显示，从IEEE-13到IEEE-123系统，尽管状态/动作空间维度增加，但训练时间仅从~25分钟增至~30分钟，证明了MGF-RL良好的可扩展性。

---

## 4. 关键结论和发现

### 主要发现
1. **MGF-RL 显著提升了CLR的自适应性和韧性**：通过结合元学习与梯度无关优化，实现了对未知停电场景的快速、高效适应。
2. **一阶元更新足以实现高性能**：无需昂贵的二阶梯度计算，即可获得优于甚至媲美MAML的泛化效果。
3. **模型无关方法更具鲁棒性**：在可再生能源预测不准确的现实条件下，MGF-RL等免模型方法比依赖精确模型的MPC更具优势。
4. **自主学习保守策略**：MGF-RL 在高不确定性下会自动采取保守策略（如优先使用可调度资源、逐步恢复负荷），无需显式编程。

### 方法的局限性
- **依赖高质量的历史任务数据**：元学习的效果高度依赖于元训练任务的多样性和代表性。
- **理论假设较强**：遗憾界分析基于一些理想化假设（如最优策略已知），实际性能可能受近似误差影响。
- **未考虑拓扑动态变化**：当前框架假设故障后网络拓扑已固定，未涉及开关操作的联合优化。

### 未来工作方向
- 扩展至支持**动态负荷剖面**和**自适应网络拓扑重构**。
- 探索**元-RL与鲁棒优化的混合框架**，以实现物理信息引导的长时域规划。
- 将框架应用于更复杂的多微网协同恢复或多代理系统。

</details>

---

### 4. [AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](https://arxiv.org/abs/2601.11044)

**Authors**: Keyu Li, Junhao Shi, Yang Xiao, Mohan Jiang, Jie Sun, Yunze Wu, Shijie Xia, Xiaojie Cai, Tianze Xu, Weiye Si, Wenjie Li, Dequan Wang, Pengfei Liu  
**Category**: cs.AI  
**Published**: 2026-01-19  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2601.11044v1  

#### Abstract
Large Language Models (LLMs) based autonomous agents demonstrate multifaceted capabilities to contribute substantially to economic production. However, existing benchmarks remain focused on single agentic capability, failing to capture long-horizon real-world scenarios. Moreover, the reliance on hum...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts  
**核心结论与实验结果总结**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **Autonomous Agent** 评测基准存在两大瓶颈：
- **任务复杂度不足**：多数基准聚焦于短视距（short-horizon）、单一能力的任务（如工具调用或代码生成），无法反映真实世界中长期、多步骤、跨能力的复杂场景。
- **依赖人工反馈**：真实任务需要多轮交互和人类指导，导致评估过程难以自动化，限制了大规模测试和可复现性。

### 🚀 提出的新方法与创新
作者提出 **AGENCYBENCH**，一个面向真实世界长周期任务的综合性评测框架，其核心创新包括：

#### （1）构建高保真、长周期的真实任务集
- 基于 AI 开发者、研究人员和工程师的实际使用场景，设计了 **32 个真实世界场景**，涵盖 **6 大核心 agentic 能力**：
  - Game Development
  - Front-end Development
  - Back-end Development
  - Code Generation
  - Research
  - MCP Tool Use
- 共计 **138 项具体任务**，每个任务包含明确的 **Query（需求）**、**Deliverables（产出物）** 和 **Rubrics（评分标准）**。
- 平均每个场景需消耗 **100万 tokens**、执行 **90 次工具调用**，耗时数小时，显著提升任务难度与上下文长度。

#### （2）实现全自动化的评估流水线
为解决“人工反馈瓶颈”，设计了两个关键技术组件：
- **User Simulation Agent**：模拟用户行为，在任务未达标时提供基于 rubric 的迭代反馈，替代人工干预。
- **Docker-based Remote Sandbox**：在隔离环境中运行生成的代码（如前端页面），通过 UI 渲染、鼠标点击等操作捕获视觉输出（截图/录屏），用于功能验证。

#### （3）统一且可扩展的评估框架
- 所有任务最终由 **rule-based 或 LLM-as-Judge** 自动打分（0–10 分），支持文本与视觉双重判断。
- 整个流程完全自动化，确保可复现性和可扩展性。

### 🔍 相比现有方法的优势
| 特性 | AGENCYBENCH | 其他基准（如 GAIA2, Toolathlon, SWE-bench） |
|------|-------------|------------------------------------------|
| 上下文长度 | ~1M tokens | 最高仅 ~200k（UltraHorizon） |
| 工具调用次数 | 平均 90 次 | 多数 < 30 次 |
| 多能力覆盖 | ✅ 6 类能力 | ❌ 通常只专注某一领域 |
| 反馈机制 | 自动化 User Simulation Agent | 依赖人工反馈 |
| 评估方式 | 视觉 + 功能沙箱 + 自动评分 | 主要靠 LLM 判断或简单脚本 |

> 💡 **优势总结**：AGENCYBENCH 是目前最接近“真实经济生产环境”的 agent benchmark，首次实现了对百万 token 级别、多轮交互、跨模态任务的端到端自动化评估。

---

## 2. 核心实验方法和设置

### 📁 数据集
- **来源**：由 20 名专家（AI 研究员、开发者）从实际工作中提炼出 32 个真实场景。
- **结构**：每个场景包含 1–5 个递进式任务，共 138 个任务。
- **分布**（见 Table 5）：
  - Game: 50 tasks
  - Code: 29 tasks
  - Front-end / Back-end: 各 15 tasks
  - Research: 19 tasks
  - MCP: 10 tasks

### ⚙️ 实验设置
- **模型范围**：
  - **闭源模型（Proprietary）**：GPT-5.2, Claude-4.5-Opus, Gemini-3-Pro, Grok-4.1-Fast
  - **开源模型（Open-source）**：GLM-4.6, Qwen-3-235B-A22B-Thinking, Deepseek-V3.2, Kimi-K2-Thinking
- **访问方式**：通过 OpenRouter API，temperature=0.7
- **Agent Scaffold**：内置文件操作、shell 执行、web search、memory management 等工具集，运行于隔离 workspace。

### 🎯 评估指标
| 指标 | 定义 | 目标 |
|------|------|-------|
| **Average Score (SAvg)** | 所有任务得分平均值（满分 100%） | 衡量整体能力 |
| **Average Attempts (Att)** | 每个场景平均尝试轮次（含反馈修正） | 衡量自主纠错能力 |
| **Pass@1 / Pass@2** | 在 1 或 2 轮内通过任务的比例 | 衡量初始成功率与学习能力 |
| **Token Efficiency (Etok)** | $ \frac{SAvg}{Tok} $，单位 token 收益 | 衡量资源利用率 |
| **Attempt Efficiency (Eatt)** | $ \frac{SAvg}{Att} $，单位尝试收益 | 衡量纠错效率 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

| 模型 | SAvg (%) | Att | Pass@1 | Pass@2 |
|------|----------|-----|--------|--------|
| **GPT-5.2**（最强） | **56.5** | 1.46 | 28.1% | 53.1% |
| **Claude-4.5-Opus** | 47.7 | 1.54 | 15.6% | 28.1% |
| **Gemini-3-Pro** | 46.9 | 1.46 | 28.1% | 37.5% |
| **Grok-4.1-Fast** | 44.3 | 1.55 | 25.0% | 31.3% |
| **GLM-4.6**（最佳开源） | 38.6 | 1.54 | 28.1% | 37.5% |
| **Qwen-3...Thinking**（最弱） | 27.0 | 1.79 | 3.1% | 9.4% |

> 🔺 **总体差距**：闭源模型平均得分为 **48.4%**，而开源模型仅为 **32.1%**，差距显著。

### 🔍 与其他基准对比（Figure 1）
| Benchmark | Avg. Tokens | Avg. Turns |
|----------|--------------|------------|
| GAIA2 | 10k | 22.5 |
| Toolathlon | 15k | 26.8 |
| UltraHorizon | 200k | 60 |
| **AGENCYBENCH** | **1000k** | **90** |

> ✅ 显著拉高任务复杂度上限。

### 🔬 消融实验与关键分析

#### （1）反馈驱动自纠正能力（Table 2）
- GPT-5.2 在引入一次反馈后，Pass 率从 28.1% 提升至 53.1%，**Rise 达 88.9%**。
- Kimi-K2-Thinking 提升幅度最大（**300%**），说明其虽初始表现差，但能快速响应反馈。
- DeepSeek-V3.2 几乎无提升（0% Rise），表明缺乏有效自我修正机制。

#### （2）资源消耗分析（Table 3）
| 模型 | Tok (M) | Time (h) | Turns |
|------|---------|----------|-------|
| GPT-5.2 | 3.4 | 0.6 | 89.0 |
| Grok-4.1-Fast | 1.2 | 0.3 | 37.0 |
| GLM-4.6 | 2.4 | 0.6 | 105.0 |

- GPT-5.2 是“暴力推理”型，高 token 换高分；
- Grok-4.1-Fast 最高效，速度快、成本低；
- Kimi/Qwen 时间开销大（>1h），可能因内部推理链过长。

#### （3）效率排名（Figure 4）
- **Attempt Efficiency 最高**：GPT-5.2（38.7%）
- **Token Efficiency 最高**：Grok-4.1-Fast（37.2%）
- **最低 Token 效率**：Claude-4.5-Sonnet（11.4%），浪费严重。

#### （4）工具使用偏好（Figure 5 & Table 6）
| 模型 | 工具偏好 |
|------|---------|
| GPT-5.2 / Claude-Opus | 高频使用 `run_shell_command`（系统级操作） |
| Gemini-3-Pro | 唯一使用 `update_memory_bank`，具备外部记忆管理能力 |
| Qwen-3...Thinking | 极度依赖 `write_file`（77.6%），倾向于重写而非修改 |
| GLM-4.6 | 大量使用 `web_fetch`（96 次），依赖外部知识检索 |

> 🧠 揭示不同模型具有独特的“problem-solving personality”。

#### （5）Agentic Scaffold 影响（Table 4）
| 模型 | 在自家 SDK 下表现 |
|------|------------------|
| Claude-4.5-Opus | +20.5%（在 Claude-Agent-SDK 中） |
| GPT-5.2 | +1.3%（在 OpenAI-Agents-SDK 中） |
| GLM-4.6 | +10.6%（在 Claude-Agent-SDK 中） |
| Kimi-K2-Thinking | -12.8%（迁移到其他 SDK 性能暴跌） |

> 🔥 发现“**Home-field Advantage**”现象：模型在其原生生态中表现最优。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **闭源模型全面领先**：在复杂推理、自我修正、稳定性方面显著优于开源模型，尤其 GPT-5.2 综合最强。
2. **资源效率差异巨大**：部分模型（如 Grok-4.1-Fast）以极低成本达成较高性能，适合部署；而某些模型（如 Claude-Sonnet）token 浪费严重。
3. **反馈利用能力决定上限**：能否有效吸收 User Simulation Agent 的反馈是提升性能的关键。GPT-5.2 和 Kimi-K2-Thinking 展现出强学习能力。
4. **工具使用体现“认知风格”**：不同模型展现出截然不同的策略（如“外科医生”vs“重写者”、“导航者”vs“执行者”）。
5. **Scaffold 决定性能边界**：模型性能高度依赖其所运行的 agentic framework，“原生生态系统”带来显著增益。

### ⚠️ 局限性
- **模型覆盖有限**：受限于算力与预算，未能涵盖所有新兴模型（如特定 fine-tuned variants）。
- **领域局限**：目前仅适用于软件类数字代理（digital agents），不涉及机器人或物理世界交互（embodied agents）。
- **伦理风险**：允许 agent 执行 shell 命令存在安全隐患，需严格沙箱控制。

### 🔮 未来工作方向
- 扩展至更多垂直领域（如生物信息学、金融建模）。
- 引入更复杂的多智能体协作任务。
- 探索轻量化、高效率的开源 agent 架构优化。
- 构建通用 scaffold-agnostic agent，减少对特定 SDK 的依赖。

---

## 📦 开源信息
- **项目主页**：[https://github.com/GAIR-NLP/AgencyBench](https://github.com/GAIR-NLP/AgencyBench)
- 包含完整 benchmark 数据集、evaluation toolkit、prompt 模板与 sandbox 配置。

> 🌟 **一句话总结**：AGENCYBENCH 是首个实现百万 token 级真实任务自动化评估的综合性 agent benchmark，揭示了当前 autonomous agent 在长周期任务中的能力边界与优化方向，推动 agent 系统向真正的“经济生产力工具”迈进。

</details>

---

### 5. [BYOL: Bring Your Own Language Into LLMs](https://arxiv.org/abs/2601.10804)

**Authors**: Syed Waqas Zamir, Wassim Hamidouche, Boulbaba Ben Amor, Luana Marotti, Inbal Becker-Reshef, Juan Lavista Ferres  
**Category**: cs.CL  
**Published**: 2026-01-19  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2601.10804v1  

#### Abstract
Large Language Models (LLMs) exhibit strong multilingual capabilities, yet remain fundamentally constrained by the severe imbalance in global language resources. While over 7,000 languages are spoken worldwide, only a small subset (fewer than 100) has sufficient digital presence to meaningfully infl...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：BYOL: Bring Your Own Language Into LLMs

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLMs）虽然具备多语言能力，但其训练严重依赖于全球语言资源的分布，导致超过7000种语言中仅有少数高资源语言（如英语、中文）占据主导地位。这种**严重的资源不平衡**导致低资源语言（Low-Resource Languages, LRLs）在LLM中的表现显著落后，存在系统性性能下降、文化错位和可访问性差等问题。

### 提出的新方法：BYOL框架
本文提出了 **Bring Your Own Language (BYOL)** 框架，一个统一、可扩展且数据高效的LLM开发范式，旨在将低资源和极低资源语言系统地引入LLM。其核心创新点包括：

1.  **四层语言资源分类体系 (Four-Tier Language Resource Classification)**：
    *   基于 `FineWeb2` 数据集，根据每种语言的数字文本量（corpus size）将其划分为四个层级：
        *   **Extreme-Low-Resource (≤ 5×10⁶ words)**：数字存在感极低。
        *   **Low-Resource (5×10⁶ - 2×10⁹ words)**：有有限但可用的数据。
        *   **Mid-Resource (2×10⁹ - 10¹¹ words)**：有大量文本资源。
        *   **High-Resource (> 10¹¹ words)**：拥有海量高质量语料。
    *   这种分类为不同资源水平的语言提供了明确的集成路径。

2.  **分层适应策略 (Tier-Aware Integration Strategy)**：
    *   **低资源语言 (Low-Resource)**：采用“全栈”数据精炼与扩展管道，包括：
        *   **数据清洗与增强**：利用大型多语言LLM对原始文本进行引导式重写，提升清晰度和连贯性。
        *   **合成文本生成**：通过高质量的机器翻译（MT）系统，将英文语料（如FineWeb-Edu）翻译成目标语言，扩充训练数据。
        *   **持续预训练 (Continual Pretraining, CPT)** 和 **监督微调 (Supervised Finetuning, SFT)**。
        *   **权重空间模型合并 (Weight-Space Model Merging)**：将语言专家模型与通用多语言基线模型（如Gemma-3）在权重空间直接合并，既获得目标语言的专业能力，又保留了原有的多语言和安全特性。
    *   **极低资源语言 (Extreme-Low-Resource)**：提出“翻译中介”（translation-mediated）的接入路径，即 **Translate-Test 范式**：
        *   用户输入的目标语言文本先被翻译成英语。
        *   英语LLM处理请求。
        *   结果再从英语翻译回目标语言。
        *   此方法避免了直接为数据极少的语言训练模型的不可行性。

### 相比现有方法的优势
*   **系统性与可扩展性**：提供了一个清晰、统一的框架来处理不同资源水平的语言，而非“一刀切”的多语言扩展。
*   **数据高效**：通过合成数据生成和模型合并，有效缓解了低资源语言数据稀缺的问题。
*   **性能与兼容性兼顾**：模型合并技术成功解决了“多语言诅咒”（curse of multilinguality），即在提升特定语言性能的同时，不损害模型在其他语言上的表现和安全性。
*   **实用性**：为极低资源语言提供了切实可行的解决方案，使得这些语言也能间接享受先进LLM的能力。

---

## 2. 核心实验方法和设置

### 使用的数据集
*   **预训练数据**：
    *   `FineWeb2`：用于获取目标语言的真实文本。
    *   `FineWeb-Edu`：高质量的英文教育语料，通过MT系统翻译成目标语言以生成合成数据。
*   **指令微调数据**：
    *   `Aya` 数据集：包含少量目标语言的原生指令数据。
    *   `SmolTalk2`：高质量的英文指令数据，被翻译成目标语言。
*   **机器翻译数据**：
    *   对于 **Inuktitut**，使用了公开的 `Nunavut Hansard (NH 3.0)` 语料库以及来自儿童读物和新闻文章的内部数据集。
*   **评估数据集**：
    *   **基准测试**：`Global MMLU-Lite`, `ARC-Easy/Hard`, `MGSM`, `XCOPA`, `StoryCloze`, `PIQA`, `HellaSwag`, `XNLI-2.0`, `XWinograd`, `Belebele`, `FLORES-200`。
    *   **关键贡献**：作者发布了由专业人士人工翻译的 `Global MMLU-Lite` 在 **Chichewa**, **Maori**, 和 **Inuktitut** 上的版本，极大提升了对这些语言的评估可靠性。
    *   **自建评测集**：`RTTBench-Mono`，一个包含25个领域的单语英文数据集，用于无参考的往返翻译（Round-Trip Translation, RTT）评估MT和LLM性能。

### 实验设置和评估指标
*   **模型架构**：以 `Gemma-3` 作为基线模型，训练了参数规模为1B、4B和12B的BYOL模型。
*   **评估指标**：
    *   **任务性能**：准确率（Accuracy）、精确匹配（Exact Match, EM）。
    *   **翻译质量**：BLEU、chrF++。
    *   **综合评分**：将多个基准的分数归一化后计算平均分。
    *   **生成能力评估**：采用 **LLM-as-a-judge** 设置，使用 `GPT-5-chat` 作为裁判，对不同模型生成的回答进行成对比较（win-rate）。

### 基线方法对比
与以下主流LLM进行了对比：
*   `Llama-3.1`
*   `Qwen-3`
*   `GPT-OSS`
*   `Apertus`
*   `Gemma-3` (基线)
*   `GPT-4o`

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
1.  **低资源语言模型性能 (Chichewa & Maori)**：
    *   在12个基准测试上，BYOL模型（如 `BYOL-nya` 和 `BYOL-mri`）相比强大的多语言基线模型，实现了约 **12% 的平均性能提升**。
    *   一个显著的结果是，参数量仅为 **4B** 的 `BYOL-nya` 模型，在多项任务上**超越了参数量大7倍的 `Gemma-3(27B-IT)` 模型**（见图1）。
    *   在 `LLM-as-a-judge` 的评估中，`BYOL-nya(12B-M)` 和 `BYOL-mri(12B-M)` 的输出被裁判（GPT-5-chat）选择的胜率极高，其表现与 `GPT-4o` 相当，确立了在Chichewa和Maori上的新SOTA。

2.  **极低资源语言（Inuktitut）的翻译系统性能**：
    *   训练的神经机器翻译（NMT）系统在 `Inuktitut → English` 方向，相比商业基线（Azure Translator），取得了约 **+4 BLEU** 的提升。
    *   在 `Global MMLU-Lite` 基准上，直接用LLM处理Inuktitut文本性能极差（下降超50%）。而采用 **Translate-Test** 范式后，性能恢复明显，相比直接推理获得了约 **14% 的准确率增益**。

### 消融实验结果
*   **基线模型选择**：通过 `RTTBench-Mono` 的往返翻译评估，确定 `Gemma-3` 是最适合的基线模型。
*   **数据混合策略**：消融实验证明，结合了真实数据、清洗后的数据和合成翻译数据的混合策略（C4）效果最佳，显著优于仅使用原始数据的方案。
*   **全参数微调 vs. LoRA**：全参数微调（full-parameter tuning）的性能始终优于低秩适配（LoRA），尽管LoRA的性能随秩（rank）增加而提升。
*   **模型合并的影响**：
    *   未合并的专家模型（`BYOL-nya (IT)`）在Chichewa上表现优异，但在其他语言上性能严重下降。
    *   合并后的模型（`BYOL-nya (M)`）不仅保持了在Chichewa上的提升，还几乎完全恢复了基线模型在其他语言上的多语言能力（见图6）。
    *   安全性评估显示，合并模型的安全特性（bias/toxicity）更接近安全的基线模型，而未合并的专家模型安全性最差。

---

## 4. 关键结论和发现

### 主要发现
1.  **资源感知的框架至关重要**：针对不同资源水平的语言采取不同的适应策略（CPT+SFT 或 Translate-Test）是解决语言鸿沟的有效途径。
2.  **数据工程是关键**：对于低资源语言，通过数据清洗、合成生成和精心设计的训练流程，可以显著提升模型性能。
3.  **模型合并是平衡性能与兼容性的利器**：权重空间模型合并技术成功地将语言专家模型的专业知识注入到通用模型中，同时保留了其多语言能力和安全对齐，是实现“专精而不失通才”的关键技术。
4.  **翻译中介是极低资源语言的可行方案**：对于数据极少的语言，构建高质量的MT系统并通过Translate-Test范式接入LLM，是一种务实且有效的替代方案。

### 方法的局限性
1.  **对高质量MT系统的依赖**：无论是生成合成数据还是实现Translate-Test，都高度依赖于一个高性能的MT系统。如果MT系统本身质量不佳，整个流程的效果会大打折扣。
2.  **数据稀缺的根本挑战**：尽管方法能缓解问题，但低资源语言缺乏大规模、高质量的预训练文本、指令数据和安全评估集的根本问题依然存在。
3.  **安全性风险**：模型合并虽然保留了基线的安全性，但如何确保在低资源语言环境下模型的安全对齐仍然是一个开放问题。

### 未来工作方向
1.  **扩展至更多语言**：将BYOL框架应用于更多的低资源和极低资源语言。
2.  **扩展至多语言专家模型**：从单一语言的专家模型扩展到一组相关语言的专家模型，并探索更复杂的合并策略。
3.  **整合语音接口**：将健壮的自动语音识别（ASR）组件集成进来，为那些主要是口语的低资源语言提供端到端的语音-LLM交互。
4.  **社区驱动的数据创建**：优先推动由社区主导的、跨模态的数据创建工作，以从根本上解决数据稀缺问题。

</details>

---

### 6. [Towards Tensor Network Models for Low-Latency Jet Tagging on FPGAs](https://arxiv.org/abs/2601.10801)

**Authors**: Alberto Coppi, Ema Puljak, Lorenzo Borella, Daniel Jaschke, Enrique Rico, Maurizio Pierini, Jacopo Pazzini, Andrea Triossi, Simone Montangero  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2601.10801v1  

#### Abstract
We present a systematic study of Tensor Network (TN) models $\unicode{x2013}$ Matrix Product States (MPS) and Tree Tensor Networks (TTN) $\unicode{x2013}$ for real-time jet tagging in high-energy physics, with a focus on low-latency deployment on Field Programmable Gate Arrays (FPGAs). Motivated by ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Towards Tensor Network Models for Low-Latency Jet Tagging on FPGAs*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**高能物理实验中实时喷注识别（jet tagging）在硬件触发系统（Level-1 Trigger）中的低延迟、资源受限部署挑战**。传统深度学习模型虽然性能优越，但在 FPGA 上实现时面临推理延迟高、参数量大、可解释性差等问题。

### 提出的新方法与新思路
提出并系统研究了两种基于**张量网络（Tensor Network, TN）** 的机器学习模型用于喷注分类任务：
- **Matrix Product State (MPS)**：链状结构的张量分解模型。
- **Tree Tensor Network (TTN)**：树状层级结构的张量网络。

这些模型从量子多体物理中借鉴而来，具有以下特点：
- 利用**低秩张量分解**实现紧凑参数化；
- 使用**线性操作为主**，适合硬件加速；
- 支持**后训练量化（Post-Training Quantization, PTQ）** 以降低精度需求；
- 具备**内在可解释性**，可通过 Quantum Mutual Information (QMI) 分析输入特征间的相关性，指导模型压缩与架构设计。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **性能** | 在低层次喷注构成粒子特征上达到与当前先进深度学习模型（如 Transformer、MLP-Mixer）相当的分类性能。 |
| **效率** | 参数更少，计算复杂度更低，支持固定延迟、流水线化实现。 |
| **可解释性** | QMI 分析揭示了模型内部特征处理机制，可用于优化嵌入方式和减少冗余连接。 |
| **硬件友好性** | 支持低位宽定点数表示，在 FPGA 上实现**亚微秒级（sub-microsecond）推理延迟**，满足 HL-LHC L1 Trigger 的 12.5 μs 预算要求。 |

---

## 2. 核心实验方法和设置

### 数据集
使用公开的 **hls4ml jet dataset**（Zenodo 提供），其特点如下：
- 包含喷注的低层次粒子构成信息（low-level particle-based representation）；
- 每个喷注由最多 150 个粒子组成，每个粒子有 16 个运动学特征；
- 实验选取前 $ N \in \{8, 16, 32\} $ 个高 $ p_T $ 粒子；
- 每粒子保留三个关键特征：横向动量 $ p_T $、相对能量 $ E_{\text{rel}} $、距喷注中心距离 $ \Delta R $。

### 特征嵌入（Feature Embedding）
采用多项式映射将每粒子特征扩展为 7 维向量：
$$
\phi(\mathbf{x}) = C \cdot [1, p_T, E_{\text{rel}}, \Delta R, p_T^2, E_{\text{rel}}^2, \Delta R^2]
$$
最终输入为所有粒子嵌入的张量积形式，模拟量子态的可分态表示。

通过 **Quantum Mutual Information (QMI)** 分析不同特征排列对长程关联的影响，发现将同类特征集中排列（如所有 $ p_T $ 在前，随后是所有 $ \Delta R $）可显著降低所需 bond dimension，从而减小模型规模。

### 模型结构
| 模型 | 结构描述 |
|------|----------|
| **MPS** | 链式结构，$ N $ 个张量串联，每个张量物理维度 $ d=7 $，bond dimension $ D=10 $；中间张量带输出类索引。 |
| **TTN** | 二叉树结构，共 $ L=\log_2 N $ 层，底层连接嵌入向量，顶层输出分类结果；最大 bond dimension $ \chi=10 $。 |

### 实验设置
- **训练数据**：620,000 样本
- **测试数据**：260,000 样本
- **优化器**：Adam
- **损失函数**：
  - MPS：Cross-Entropy
  - TTN：Mean Squared Error（经验上更稳定）
- **软件框架**：
  - MPS：`tn4ml` Python 库
  - TTN：`qtealeaves`（原用于量子系统模拟，拓展至 ML）

### 量化策略
进行**后训练量化（PTQ）**：
- 固定整数位宽为 2 bits；
- 变化**小数位宽（Fractional Bits, FB）** 从 2 到 14；
- 对比量化前后准确率变化，评估数值鲁棒性。

### 硬件部署平台
目标 FPGA 芯片：**AMD-Xilinx XCVU13P**
- 时钟频率：250 MHz
- 实现方式：
  - MPS：使用 **Vitis HLS**（高层次综合，C++ 描述）
  - TTN：使用 **VHDL** 手动编写（精细控制资源与时序）

### 评估指标
| 类别 | 指标 |
|------|------|
| **模型性能** | Accuracy (%)、AUC (%)（按类别：q, W, Z, t） |
| **硬件性能** | Latency (ns)、DSP/LUT/FF/BRAM/CARRY8 占用率、Memory (kB) |
| **压缩效果** | 参数数量、bit width 影响 |

### 基线方法对比
文中未直接复现所有基线，但引用了多个前沿工作作为性能参照：
- **Permutation-Invariant NNs** [25]
- **JEDI-linear / LL-GNN** [20,27]
- **MLP-Mixer on FPGA** [24]
- **Transformer-based models** [16]

---

## 3. 主要实验结果和性能指标

### （1）全精度模型性能（Table I）

| N | 模型 | #params | Acc (%) | AUC_q | AUC_W | AUC_Z | AUC_t |
|----|-------|--------|--------|--------|--------|--------|--------|
| 8 | MPS | 6,678 | 66.1 | 89.1 | 87.1 | 85.4 | 91.2 |
|   | TTN | 4,460 | 65.3 | 89.3 | 87.0 | 84.3 | 92.4 |
| 16 | MPS | 12,278 | 72.0 | 90.4 | 91.6 | 89.0 | 92.9 |
|    | TTN | 10,420 | 72.5 | 91.3 | 91.7 | 89.7 | 93.9 |
| 32 | MPS | 23,478 | 74.8 | 90.8 | 93.7 | 91.5 | 93.5 |
|    | TTN | 22,340 | 77.1 | 92.6 | 94.3 | 93.1 | 94.7 |

> ✅ **关键观察**：
> - TTN 在多数情况下优于 MPS，尤其在 $ N=32 $ 时准确率达到 **77.1%**；
> - 性能接近甚至超过文献中报告的 MLP-Mixer 和 Transformer 模型；
> - 参数量远小于典型 DNN，且随 $ N $ 增长缓慢。

---

### （2）量化影响分析（Fig. 6 & 7）

#### MPS 量化敏感性
- 当 **FB < 8** 时，量化运算（qop）导致明显性能下降；
- 使用浮点运算但权重量化（fpop）仍保持较高精度，说明主要误差来自**运算过程而非权重存储**。

#### TTN 量化鲁棒性更强
- 在 **FB ≥ 6** 时即可维持高性能；
- 即使在 FB=6 下，accuracy 几乎无损；
- 表明 TTN 更适合低精度硬件部署。

✅ **结论**：TTN 对低精度算术更具容忍力，更适合资源受限场景。

---

### （3）FPGA 硬件性能（Table II）

| N | 模型 | FB | DSP (%) | Latency (ns) | Memory (kB) |
|----|-------|-----|---------|-------------|------------|
| 8 | MPS | 14 | 52% | 236 | 114 |
|   | TTN | 14 | 39.75% | 92 | 71 |
| 16 | MPS | 14 | 78% | 432 | 203 |
|    | TTN | 14 | 92.41% | 124 | 166 |
| 32 | MPS | 14 | **129%** ❌（不可合成） |
|    | MPS | 8 | **60%** ✅ | 708 | 239 |
|    | TTN | 14 | 99.27% | 156 | 357 |
|    | TTN | 6 | **0%** ✅（无需 DSP） | 156 | 178 |

> 🔍 **亮点发现**：
> - 所有配置均实现 **< 1 μs 推理延迟**（最快 92 ns），满足 L1 Trigger 要求；
> - TTN 在 FB=6 时完全避免使用 DSP，大幅节省关键资源；
> - MPS 在 N=32、FB=14 时超出 DSP 容量，但通过降精度（FB=8）成功部署；
> - 内存占用随 N 和 FB 增加而上升，但仍在合理范围内。

---

### （4）消融实验与分析
- **QMI 分析指导嵌入设计**：通过可视化输入间量子互信息，发现原始相邻嵌入（$ p_T, \Delta R $ 成对）引入不必要的长程相关，改用“特征分组”策略后显著降低建模难度；
- **bond dimension 影响**：较小的 $ D $ 或 $ \chi $ 已足够捕捉有效特征交互，验证了 TN 的高效表达能力；
- **TTN vs MPS 架构差异**：TTN 的层级聚合天然适合并行化，且更利于控制信息流路径。

---

## 4. 关键结论和发现

### 主要发现
1. **TN 模型可在 FPGA 上实现亚微秒级喷注分类**，满足 HL-LHC Level-1 Trigger 的严苛延迟约束（< 12.5 μs）；
2. **TTN 在准确性和硬件效率方面整体优于 MPS**，特别是在低精度下表现更稳健；
3. **后训练量化可显著压缩模型而不牺牲性能**，TTN 在仅 6-bit 小数精度下仍保持高精度；
4. **QMI 分析提供了一种可解释工具**，可用于指导特征工程与模型压缩；
5. **TN 天然适合 FPGA 实现**，尤其是 TTN 可完全规避 DSP 使用，提升资源利用率。

### 方法的局限性
- **当前仅适用于分类任务**，尚未扩展到回归或多任务学习；
- **输入长度固定**（需零填充），缺乏对变长序列的灵活处理能力；
- **训练速度较慢**，相比现代 DNN 缺乏自动微分支持，依赖专用库；
- **HLS 工具链不确定性**：MPS 使用 Vitis HLS 导致资源与延迟预测困难，不如 VHDL 精确可控。

### 未来工作方向
1. 探索更先进的量化技术，如**逐权重混合精度量化（per-weight mixed precision quantization）**；
2. 开发利用 TN **规范形式（canonical form）和规范自由度（gauge freedom）** 的专用 PTQ 方法；
3. 引入**动态剪枝或自适应结构优化**，进一步压缩模型；
4. 将 TN 应用于其他 HEP 实时任务，如异常检测、事件重建等；
5. 推动 TN 与量子启发算法结合，探索新型低延迟智能触发范式。

---

## 总结
该论文成功展示了 **Tensor Network 模型在高能物理实时喷注识别中的巨大潜力**。它不仅提供了媲美深度神经网络的分类性能，还具备**参数精简、可解释性强、易于硬件部署、支持低位宽运行**等独特优势。特别是 TTN 架构在 FPGA 上实现了 **92–156 ns 的极低延迟**，且可通过量化进一步降低资源消耗，为下一代触发系统的智能化升级提供了切实可行的技术路径。

</details>

---

### 7. [Reward Modeling for Scientific Writing Evaluation](https://arxiv.org/abs/2601.11374)

**Authors**: Furkan \c{S}ahinu\c{c}, Subhabrata Dutta, Iryna Gurevych  
**Category**: cs.CL  
**Published**: 2026-01-19  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2601.11374v1  

#### Abstract
Scientific writing is an expert-domain task that demands deep domain knowledge, task-specific requirements and reasoning capabilities that leverage the domain knowledge to satisfy the task specifications. While scientific text generation has been widely studied, its evaluation remains a challenging ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Reward Modeling for Scientific Writing Evaluation

## 1. 论文的主要贡献和创新点

### 解决的问题
科学写作是一种高度依赖领域专业知识、任务特定要求和复杂推理能力的专家级任务。现有的 **LLM-as-a-judge** 和 **reward models** 主要针对通用基准（如数学推理、代码生成）进行优化，难以适应科学写作中多样、动态且多维度的评价标准。具体挑战包括：
- 难以基于稀疏的领域知识进行推理；
- 对任务依赖、多方面的评分标准理解不足；
- 缺乏对显式评价准则（constitution）的实时适应能力；
- 为每个新任务单独微调成本高昂，尤其在低资源场景下不可行。

### 提出的新方法与思路
作者提出了 **ScIRM**（Scientific Writing Evaluation Reward Model）及其增强版本 **ScIRM-REF**，专为科学写作评估设计的开源、低成本奖励模型。其核心创新在于：

#### （1）两阶段训练框架（Two-stage Training Framework）
- **Stage 1**: 通过 **GRPO**（Group Relative Policy Optimization）优化模型对科学写作评估偏好的理解，学习如何依据给定的评价准则（criteria + scoring rubric）进行打分。
- **Stage 2**: 引入“自我反思”机制，鼓励模型重新审视其初始判断和推理过程，进一步提升其 **reasoning 能力** 和对评价规则的忠实度。

#### （2）多方面联合评估设计（Multi-aspect Evaluation Design）
- 不再输出单一总分，而是从多个独立维度（如 coherence, positioning type, actionability 等）分别评估，提高评估的细粒度、可靠性和可解释性。
- 支持不同任务使用不同的评分量表（scoring rubrics），增强了模型的鲁棒性和泛化能力。

#### （3）显式宪法条件化（Constitution-conditioned Evaluation）
- 将评价标准（criteria）、评分量表（rubric）和示例（examples）作为输入的一部分（in-context constitution），使模型能够在推理时动态适应不同的评价任务，避免了传统 Constitutional AI 中“固化宪法”的僵化问题。

### 相比现有方法的优势
| 维度 | 现有方法（如 Prometheus, DS-GRM） | ScIRM / ScIRM-REF |
|------|----------------------------------|-------------------|
| **任务泛化性** | 多为通用 judge 或固定任务 reward model | 可跨任务复用，无需重新训练 |
| **推理能力** | 推理能力有限，易忽略细节 | 显式建模推理过程，支持反思修正 |
| **灵活性** | 固定评分标准 | 支持动态、多样的评分量表 |
| **细粒度反馈** | 通常只输出总分 | 支持多维度独立评分 |
| **成本与开放性** | 部分依赖闭源大模型（如 o3-mini） | 完全开源，基于 7B 规模模型 |

---

## 2. 核心实验方法和设置

### 数据集
研究整合了来自多个来源的多样化科学写作任务数据，共 **65,357 条实例**（训练集 58,712，测试集 6,645）：

| 数据集 | 任务类型 | 评估方面（Aspects） | 评分量表 |
|--------|--------|---------------------|---------|
| **Sahinuc et al. (2025)** 扩展版 | Related Work 生成 | Coherence, Positioning Type, Positioning Consistency | 二分类（0-1） |
| **RevUtil (Sadallah et al., 2025)** | 科学论文评审 | Actionability, Grounding, Verifiability, Helpfulness | 五级量表（1-5） |

此外，在未见任务测试中使用了两个额外数据集：
- **Novelty Alignment Dataset (Afzal et al., 2026)**：评估两份新颖性评估是否达成一致（对齐与否），用于测试模型在高推理需求任务上的表现。
- **Scientific Revision Dataset (Jourdan et al., 2025)**：评估根据指令修改后的文本质量，关注 **Relatedness**（是否遵循指令）和 **Correctness**（是否正确改进原文）。

### 实验设置
- **基础模型**：Qwen 2.5-7B
- **训练方法**：采用 **LoRA** 进行参数高效微调，使用 **GRPO** 算法进行强化学习训练。
- **输出格式**：强制模型先输出 `<reasoning>` 推理过程，再输出 `<score>` 最终分数，便于提取奖励信号。
- **评估方式**：
  - 在 **已见任务**（related work, review utility）上测试性能；
  - 在 **未见方面**（unseen aspects）上测试泛化能力（如移除 Actionability 方面进行训练）；
  - 在 **未见任务**（unseen tasks）上测试跨任务迁移能力（novelty alignment, revision evaluation）。
- **重复次数**：每项测试运行 5 次（o3-mini 运行 3 次），报告均值与标准差。

### 基线方法对比
| 类型 | 基线模型 |
|------|--------|
| **通用 LLM** | Qwen2.5, Qwen3, Llama3.1, Granite3.3, DS-Qwen |
| **专用 Judge 模型** | Prometheus, Selene |
| **Reward Model** | DeepSeek-GRM, Skywork-Critic |
| **闭源强基线** | o3-mini（Proprietary） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 2）
| Model | RW | RU | Nov. | Rev. | **Avg.** |
|-------|----|----|------|------|----------|
| DS-Qwen | 0.70 | 0.44 | 0.41 | 0.72 | 0.57 |
| Qwen3 | 0.78 | 0.57 | 0.58 | 0.81 | 0.68 |
| o3-mini | 0.79 | 0.60 | **0.78** | **0.84** | **0.75** |
| **SciRM** | **0.83** | **0.71** | 0.61 | 0.78 | **0.73** |
| **SciRM-Ref** | **0.83** | 0.69 | **0.74** | 0.78 | **0.76** |

> 注：RW = Related Work, RU = Review Utility, Nov. = Novelty Alignment, Rev. = Revision Evaluation

### 与基线方法的对比结果
- **在所有四项任务上，ScIRM 和 ScIRM-Ref 均显著优于大多数开源基线模型**。
- 在 **相关工作评估（RW）** 上，ScIRM 达到接近完美的准确率（0.83），远超其他模型。
- 在 **评审效用评估（RU）** 上，ScIRM 在所有子任务上均取得最高分，表明其能有效处理复杂的多级评分标准。
- 在 **未见任务 novelty alignment** 上，**ScIRM-Ref 表现尤为突出**，得分高达 0.74，仅次于闭源的 o3-mini（0.78），显示出强大的推理泛化能力。
- **ScIRM-Ref 在平均得分上超越所有基线模型，成为最强的开源评估器**。

### 消融实验结果
#### （1）两阶段训练的有效性（Stage 2 的作用）
- **ScIRM-Ref vs ScIRM**：两者在大多数任务上表现相近，但在 **novelty alignment** 任务上，ScIRM-Ref 显著优于 ScIRM（0.74 vs 0.61）。
- 这表明第二阶段的“自我反思”训练显著提升了模型在 **高推理强度任务** 上的表现。

#### （2）未见方面测试（Unseen Aspect）
- 使用 **ScIRM-MASKED**（在训练中移除 Actionability 和 Grounding 方面）进行测试。
- 结果显示，尽管在这两个方面性能下降，但仍 **优于多数基线模型**，说明模型学到的是通用的评估模式，而非过拟合特定方面。

---

## 4. 关键结论和发现

### 主要发现
1. **两阶段训练显著提升科学写作评估性能**：特别是第二阶段的“自我反思”机制，能有效增强模型的推理能力和对评价标准的忠实度。
2. **多任务联合训练提升泛化能力**：通过在多个任务和评分体系上联合训练，模型能够更好地适应新任务和新评分标准。
3. **ScIRM-Ref 实现接近闭源模型的性能**：在完全开源的前提下，ScIRM-Ref 在平均性能上超越所有开源基线，并在关键推理任务上逼近 o3-mini，验证了其高效性和实用性。
4. **单个评估器可复用于多种任务**：无需为每个新任务重新训练，极大降低了部署成本。

### 局限性
1. **模型规模限制**：受限于计算资源，仅在 7B 规模模型上实验，更大模型可能捕捉更细微的评估差异。
2. **数据稀缺性**：科学写作评估数据仍较稀缺，多数评分量表为二元或五级，缺乏更精细的标注。
3. **评分一致性挑战**：部分数据集中存在人类标注者分歧，导致模型在边界案例上容易混淆相邻分数。

### 未来工作方向
1. 利用 ScIRM 提供的细粒度反馈，指导科学文本生成模型进行 **强化学习优化**。
2. 构建更丰富、更细粒度的科学写作评估数据集，支持更复杂的评分体系。
3. 探索将外部知识检索（retrieval）集成到评估流程中，进一步提升领域知识的利用效率。
4. 将该框架扩展至其他专家领域（如医学、法律）的文本评估任务。

> **总结**：本文提出的 ScIRM 系列模型为科学写作评估提供了一个**可泛化、可解释、低成本且高性能**的解决方案，推动了自动化科研评估向更可靠、更实用的方向发展。

</details>

---

### 8. [One LLM to Train Them All: Multi-Task Learning Framework for Fact-Checking](https://arxiv.org/abs/2601.11293)

**Authors**: Malin Astrid Larsson, Harald Fosen Grunnaleite, Vinay Setty  
**Category**: cs.CL  
**Published**: 2026-01-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2601.11293v1  

#### Abstract
Large language models (LLMs) are reshaping automated fact-checking (AFC) by enabling unified, end-to-end verification pipelines rather than isolated components. While large proprietary models achieve strong performance, their closed weights, complexity, and high costs limit sustainability. Fine-tuni...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：One LLM to Train Them All: Multi-Task Learning Framework for Fact-Checking

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前自动化事实核查（**Automated Fact-Checking, AFC**）系统通常依赖于多个独立训练的模型来完成不同子任务（如声明检测、证据重排序、立场判断），导致：
- **资源冗余**：需要维护多个专用模型；
- **高成本**：大参数量的闭源 LLM 虽然性能强，但计算开销大、不可持续；
- **缺乏跨任务知识共享**：各模块之间缺乏协同优化。

此外，Few-shot Prompting 在复杂任务上表现有限，而单任务微调（Single-Task Learning, STL）无法实现端到端统一建模。

---

### 🚀 提出的新方法与创新
本文提出一种基于 **多任务学习（Multi-Task Learning, MTL）** 的统一框架，用于联合微调一个小型开源 LLM 来同时执行 AFC 的三大核心任务：
1. **Claim Detection**（声明检测）
2. **Evidence Re-ranking**（证据重排序）
3. **Stance Detection**（立场判断）

#### 创新点包括：
- **首次在 decoder-only 开源 LLM 上实现全任务联合训练**：使用 Qwen3 系列模型（如 Qwen3-4B）结合 **QLoRA** 进行参数高效微调。
- **三种 MTL 架构对比研究**：
  - 分类头（Classification Head, CLS）
  - 因果语言建模头（Causal Language Modeling Head, CLM）
  - 指令微调（Instruction Tuning, IT）
- **统一输入表示与共享骨干网络**：冻结主干，仅更新 QLoRA 适配器和任务头，提升效率并减少灾难性遗忘。
- **提供可复现的完整配置与代码**：促进透明性和可持续 AI 发展。

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | 本文 MTL 方法 |
|------|--------|-------------|
| 模型数量 | 多个专用模型 | 单一统一模型 |
| 参数效率 | 高冗余 | 使用 QLoRA，<5% 可训练参数 |
| 推理部署 | 复杂流水线 | 端到端一体化 |
| 成本与能耗 | 高（尤其闭源 LLM） | 显著降低（小模型 + 开源） |
| 性能增益 | Few-shot 表现弱 | 在零样本/少样本基础上提升高达 **54%** |

> ✅ **优势总结**：在保持高性能的同时，显著提升了系统的**效率、可持续性、可解释性与部署便捷性**。

---

## 2. 核心实验方法和设置

### 📚 数据集
| 任务 | 数据集 | 描述 |
|------|-------|------|
| **Claim Detection** | CheckThat! 2024 英文子集 | 政治辩论中的语句标注为 checkworthy (T) / non-checkworthy (F) |
| **Evidence Re-ranking** | AVeriTeC | 真实世界声明 + Google 检索文档片段，人工标注是否包含支持证据 |
| **Stance Detection** | Politifact/Snopes/FullFact 合并数据集 | 四分类标签：`Supported`, `Partially-Supported`, `Partially-Refuted`, `Refuted` |

> 所有数据均为公开可用，确保可复现性。

---

### ⚙️ 实验设置
- **模型**：Qwen3 系列（0.6B, 1.7B, 4B, 8B），以 Qwen3-4B-Instruct 为主；
- **微调方式**：
  - 使用 **QLoRA**（r=64, α=16, 4-bit NF4 量化）
  - 冻结主干，仅训练 adapter 和任务头
- **训练策略**：
  - Batch size: 32
  - Epochs: 5
  - Optimizer: paged AdamW (32-bit)
  - Precision: bfloat16
  - Gradient checkpointing 启用
- **任务权重**：初始设为 1:1:1，后续探索加权影响

---

### 📊 评估指标
对每个任务报告：
- **F1-score per class**
- **Macro-F1**（平衡类别不均衡）
- **Weighted-F1**（反映真实分布）

---

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| **Zero/Few-shot Baselines** | Zero-shot prompting, Few-shot ICL（每类一个示例） |
| **Non-LLM Baselines** | XLM-RoBERTa-Large（用于 CD & SD）、BGE-reranker-v2-m3（用于 ER） |
| **Single-Task Learning (STL)** | 各任务单独微调 CLS/CLM/IT 版本的 Qwen3-4B |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Tables 2–3）

#### Claim Detection（Macro-F1）
| 方法 | Macro-F1 (%) |
|------|------------|
| Zero-shot | 75.9 |
| Few-shot ICL | 76.1 |
| XLM-RoBERTa-Large | 85.3 |
| **MTL-CLS (本文)** | **91.21** |
| STL-CLS（单任务最佳） | 92.36 |

> ➕ **相对提升达 44%**（vs zero-shot）

---

#### Evidence Re-ranking（Macro-F1）
| 方法 | Macro-F1 (%) |
|------|------------|
| Zero-shot | 65.9 |
| Few-shot ICL | 66.5 |
| BGE-reranker | 68.2 |
| **MTL-CLS (本文)** | **80.39** |
| STL-CLS | 81.39 |

> ➕ **相对提升达 54%**（vs zero-shot）

---

#### Stance Detection（Macro-F1）
| 方法 | Macro-F1 (%) |
|------|------------|
| Zero-shot | 50.2 |
| Few-shot ICL | 51.1 |
| XLM-RoBERTa-Large | 67.1 |
| **MTL-IT (本文)** | **68.50** |
| STL-CLS（最优） | 69.36 |

> ➕ **相对提升达 31%**（vs zero-shot）

---

### 🔬 消融实验结果

#### ✅ 不同 MTL 架构比较（RQ2）
| 架构 | 整体表现 | 推理速度 |
|------|---------|----------|
| **CLS Head** | ✅ 最佳性能（尤其 CD & ER） | 46 samples/s（最快） |
| CLM Head | 中等性能 | 15.6 samples/s |
| Instruction Tuning | 接近 CLS，适合泛化 | 5.46 samples/s（最慢） |

> **结论**：对于分类型任务，显式监督的 **CLS 头更优且高效**。

---

#### 🔢 损失权重的影响（Table 4）
调整损失权重（λ_CD : λ_ER : λ_SD）发现：
- 最佳组合之一：**(1,4,2)** 或 **(4,1,2)**
- 加权后：
  - Claim Detection Macro-F1 提升至 **93.99**
  - Stance Detection 达 **68.86**
- 但 Evidence Re-ranking 对更高权重无明显收益

> **发现**：强调更难或数据稀疏的任务有助于整体优化。

---

#### 🔄 训练顺序的影响（Table 5）
不同任务训练顺序影响性能：
| 顺序 | Claim Det. (Mac-F1) | Ev. Rank. (Mac-F1) | Stance (Mac-F1) |
|------|---------------------|--------------------|-----------------|
| C-R-S（声明→排序→立场） | **92.28** | 78.50 | 66.26 |
| R-S-C（排序→立场→声明） | 91.80 | **82.11** | 64.97 |

> **结论**：从检索 → 推理 → 分类的渐进式课程学习（curriculum）能更好促进迁移。

---

#### 📦 模型规模影响（Figure 3）
- **Qwen3-4B** 是最佳平衡点：
  - 在所有任务中表现稳定
  - 优于 0.6B / 1.7B
  - 与 8B 差距极小（几乎饱和）
- 更大规模未带来显著增益

> ✅ 小到中等模型即可胜任 AFC 多任务。

---

#### 📈 数据量扩展（Figure 4）
- 随着训练数据增加，性能持续上升（尤其 Stance 和 Ranking）
- Claim Detection 很快饱和（因其较简单）
- 表明高质量多样化数据对 MTL 泛化至关重要

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **MTL 可有效整合 AFC 流水线**：单一模型可同时处理三个核心任务，性能接近甚至超越单任务模型。
2. **CLS 架构最优**：相比 CLM 和 IT，分类头在精度和推理效率上均占优。
3. **显著超越零/少样本设置**：在 Macro-F1 上实现 **44%~54% 的相对提升**，证明微调必要性。
4. **小模型足够强大**：Qwen3-4B 即可达到优异性能，无需更大模型。
5. **训练调度与损失加权重要**：合理的任务顺序和损失比例可进一步提升性能。
6. **可持续性增强**：通过开源模型 + QLoRA + 单一部署，大幅降低计算与能源消耗。

---

### ⚠️ 局限性
1. **任务间存在轻微负迁移**：某些配置下某一任务提升会导致另一任务轻微下降。
2. **指令微调推理慢**：虽然灵活，但生成式输出延迟高，不适合实时场景。
3. **数据偏差问题**：
   - Claim Detection 存在类别不平衡（非声明远多于声明）
   - 错误分析显示长句、含数字语句更容易被误判
4. **隐式相关性识别困难**：Evidence Re-ranking 对间接支持（背景、因果）识别能力有限。

---

### 🔮 未来工作方向
1. **引入动态任务权重机制**：自动调节各任务损失权重（如 GradNorm）。
2. **探索更强的提示工程与思维链（CoT）集成**：提升指令微调效率。
3. **构建端到端多跳推理能力**：支持跨多个证据片段的综合判断。
4. **扩展至多语言与跨领域事实核查**：验证 MTL 的泛化能力。
5. **结合检索增强生成（RAG）**：将证据检索也纳入统一框架。

---

## ✅ 总结
该论文提出了一个**高效、可持续、可复现的多任务学习框架**，成功地将自动化事实核查的三大任务整合到一个小型开源 LLM 中。实验证明，该方法不仅在性能上显著优于零/少样本和非 LLM 基线，在资源利用方面也极具优势，为构建下一代透明、绿色、可扩展的事实核查系统提供了坚实基础。

</details>

---

### 9. [Latent Dynamics Graph Convolutional Networks for model order reduction of parameterized time-dependent PDEs](https://arxiv.org/abs/2601.11259)

**Authors**: Lorenzo Tomada, Federico Pichi, Gianluigi Rozza  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2601.11259v1  

#### Abstract
Graph Neural Networks (GNNs) are emerging as powerful tools for nonlinear Model Order Reduction (MOR) of time-dependent parameterized Partial Differential Equations (PDEs). However, existing methodologies struggle to combine geometric inductive biases with interpretable latent behavior, overlooking ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Latent Dynamics Graph Convolutional Networks for Model Order Reduction of Parameterized Time-Dependent PDEs

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文针对**参数化时变偏微分方程 (parameterized time-dependent PDEs)** 的模型降阶 (Model Order Reduction, MOR) 任务中，现有图神经网络 (GNN) 方法存在的两个关键缺陷：
- **缺乏动力学驱动的可解释性**：许多基于 GNN 的方法未能有效建模系统的内在动态演化过程，导致潜在空间轨迹难以解释。
- **几何信息与动力学耦合不清**：传统方法往往将空间结构信息与时间演化混合处理，无法实现对系统全局低维动态行为的独立学习。

此外，现有方法如 GCA-ROM 缺乏因果性（causal modeling），不适用于需要时间步进预测的场景。

---

### 提出的新方法：LD-GCN
作者提出了一种全新的、纯数据驱动且**无编码器 (encoder-free)** 的架构——**Latent Dynamics Graph Convolutional Network (LD-GCN)**，其核心思想是：
- **两分支解耦设计**：
  1. **NODE 分支**：在低维潜在空间中通过 Neural Ordinary Differential Equation (NODE) 显式建模并递推演化系统动态；
  2. **GCN 解码器分支**：利用 Graph Convolutional Network 将潜在状态一致地映射回具有复杂几何形状的物理域上，支持非结构化网格。
- **全局潜在表示**：学习一个受外部输入和参数条件影响的全局、低维动态系统表征。

该方法结合了**潜在动力学模型的可解释性**与**GNN 对几何结构的归纳偏置能力**，实现了高效且具物理意义的降阶建模。

---

### 相比现有方法的优势
| 特性 | LD-GCN | GCA-ROM | LDNet |
|------|--------|---------|-------|
| 是否有编码器 | ❌ (encoder-free) | ✅ | ❌ |
| 动力学建模方式 | 显式 NODE 时间推进 | 静态参数到潜变量映射 | 显式 NODE |
| 几何信息利用 | ✅ (GCN 解码器保留拓扑) | ✅ | ❌ (meshless) |
| 支持时间外推 | ✅ | ❌ | ✅ |
| 可解释性 | 高（规则潜轨迹） | 中等 | 高 |
| 参数量 | 更少 (~50%) | 多 | 少 |
| 支持零样本预测 | ✅（通过潜空间插值） | ❌ | ✅ |

> ✅ **优势总结**：
> - 实现了**动力学演化与几何重建的完全解耦**；
> - 在保持高精度的同时显著减少训练参数；
> - 支持**时间外推 (time extrapolation)** 和**零样本预测 (zero-shot prediction)**；
> - 潜在轨迹具备良好的正则性和可解释性，可用于发现复杂现象（如分岔）。

---

## 2. 核心实验方法和设置

### 使用的数据集
论文在四个具有不同复杂度的计算力学基准问题上进行了验证：

| 测试案例 | 方程类型 | 参数类型 | 网格节点数 $N_h$ | 时间步数 $N_t$ |
|--------|--------|--------|------------------|----------------|
| **Square Advection (SA)** | 对流扩散方程 | 物理参数 ($\mu \in [-1,1]^2$) | 1472 | 100 |
| **Moving Hole (MH)** | 对流扩散方程 | 几何参数（孔洞位置） | 1352 | 100 |
| **Lid-driven cavity flow** | Navier-Stokes 方程 | 边界条件参数（Fourier mode） | 10,024 | 19 |
| **Coanda effect** | Navier-Stokes 方程 | 黏度参数（引发分岔） | 2752 | 81 |

所有数据均通过有限元法 (FE) 高保真模拟生成，并构建为 $(t, \mu)$ 到全场解 $u_h(t;\mu)$ 的输入输出对。

---

### 实验设置与评估指标

#### 训练/测试划分
- **参数划分**：部分 $\mu$ 值用于训练，其余用于测试（评估泛化能力）；
- **时间划分**：仅使用前段（如 $T_{\text{train}}=1.5$）进行训练，后续用于时间外推评估。

#### 评估指标
定义相对误差如下：
$$
\text{erel}(t;\mu) = \frac{\|u_h(t;\mu) - u_{\text{sim}}(t;\mu)\|_2}{\|u_h(t;\mu)\|_2}, \quad E_{\text{mean}}, E_{\text{max}}
$$
对于腔体流动还使用 **Normalized Root Mean Square Error (NRMSE)**。

#### 基线方法对比
- **GCA-ROM**：基于图卷积自编码器的经典降阶方法；
- **LDNet**：无网格的潜在动力学网络，作为“无几何”对照；
- 所有模型统一使用相同训练策略（Adam + L-BFGS）、激活函数（ELU/tanh）和损失函数（MSE + $L^1$ 正则）。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 表 1：Square Advection 与 Moving Hole 结果（来自 Table 1）
| Test Case | Architecture | $n$ | $E_{\text{mean}}$ | $E_{\text{max}}$ | Trainable Params |
|----------|-------------|-----|--------------------|------------------|------------------|
| SA | LD-GCN | 3 | $7.86 \times 10^{-3}$ | $5.26 \times 10^{-2}$ | ~322k |
| SA | GCA-ROM | 3 | $1.87 \times 10^{-2}$ | $9.75 \times 10^{-2}$ | ~599k |
| MH | LD-GCN | 3 | $1.67 \times 10^{-2}$ | $1.54 \times 10^{-1}$ | ~298k |
| MH | GCA-ROM | 3 | $4.47 \times 10^{-2}$ | $2.14 \times 10^{-1}$ | ~547k |

> ✅ **结论**：LD-GCN 在两种情况下均优于 GCA-ROM，且参数量减少约 50%。

---

#### 表 2：Lid-driven Cavity Flow 结果（来自 Table 2）
| $n$ | NRMSE | Parameters |
|-----|--------|------------|
| 3 | $6.79 \times 10^{-5}$ | ~4.05M |
| 10 | $8.33 \times 10^{-3}$ | ~4.05M |

> ✅ 与原始 LDNet 报道的 $1.39 \times 10^{-3}$ 相当甚至更优，表明即使在简单几何下也能保持竞争力。

---

#### Coanda Effect 分岔检测结果
- 成功识别出黏度 $\nu^* \approx 0.96$ 处的分岔点；
- 潜空间相图 $(s_1, s_2)$ 清晰展示对称破缺行为；
- 定义潜变量振幅 $A(s_2)$ 构建的分岔图与高保真结果高度一致（见 Figure 14b）。

---

### 与其他方法的对比结果
- **vs GCA-ROM**：
  - 外推阶段误差降低一个数量级；
  - 更稳定，不易过拟合（因参数更少）；
  - 支持时间因果建模，适合长期预测。
- **vs LDNet**：
  - 在复杂几何问题（如 MH）中表现更好；
  - 能显式利用网格拓扑信息；
  - 代价是参数量随网格规模线性增长。

---

### 消融实验与分析（Subsection 2.5 & 3.1.4）

#### 潜空间插值实验（Zero-shot Prediction）
- 使用 **Gaussian Process Regression (GPR)** 和 **线性样条插值** 对潜轨迹进行跨参数/时间插值；
- 插值得到的 $s(t;\mu)$ 经解码后能准确重构全场解；
- 插值误差满足理论界：$\|u_h - u_{\text{interp}}\| \leq L \cdot \delta(m) + e(t;\mu)$；
- 即使在 $\mu=(0,0)$（对流项消失）这种极端情况仍表现良好（Figure 7）。

> 🔍 发现：潜轨迹具有强正则性，支持高质量插值，可用于加速在线推理。

---

## 4. 关键结论和发现

### 主要发现
1. **LD-GCN 成功融合了潜在动力学建模与图神经网络的优点**，实现了兼具效率、可解释性和几何适应性的降阶框架；
2. **潜空间轨迹具有高度规则性和物理一致性**，不仅能用于重建，还可直接用于系统行为分析（如分岔检测）；
3. **支持零样本预测与时间外推**，在未见过的参数和时间点上仍保持高精度；
4. **数学上可证**：提出了针对 encoder-free 架构的 Universal Approximation Theorem (UAT)，为方法提供了理论支撑；
5. 在多个复杂 PDE 问题中，**性能优于 GCA-ROM，媲美 LDNet**，同时参数更少。

---

### 方法的局限性
1. **参数量随网格分辨率线性增长**：由于全连接层作用于每个节点，难以扩展到极高自由度问题；
2. **依赖固定网格拓扑**：当前假设所有仿真共享相同的 $N_h$ 和连接关系，限制了几何大变形应用；
3. **初始条件固定**：encoder-free 设计要求每个参数对应唯一初态，难以处理多初值场景；
4. **训练成本较高**：需在整个时间轴上反向传播，训练速度慢于 GCA-ROM（约两倍时间）。

---

### 未来工作方向
1. **引入多保真度解码器**（如 GFN）以缓解大规模网格带来的参数爆炸；
2. **集成图池化 (graph pooling/unpooling)** 机制，支持变拓扑和变尺寸网格；
3. **拓展至 Operator Learning 框架**，支持任意初始场和强迫项的学习；
4. **发展带误差估计的理论体系**，特别是针对时变信号的情形；
5. 探索如何从潜空间恢复多个共存解（如分岔后的多稳态）。

---

> 📌 **总体评价**：  
> LD-GCN 是一种将 **latent dynamics modeling** 与 **GNN-based reconstruction** 成功结合的创新架构，在科学机器学习领域为参数化时变 PDE 的高效、可解释降阶提供了一个强有力的新工具。它不仅提升了预测性能，更为理解复杂系统的内在动态开辟了新的路径。

</details>

---

### 10. [ORBITFLOW: SLO-Aware Long-Context LLM Serving with Fine-Grained KV Cache Reconfiguration](https://arxiv.org/abs/2601.10729)

**Authors**: Xinyue Ma, Heelim Hong, Taegeon Um, Jongseop Lee, Seoyeong Choy, Woo-Yeon Lee, Myeongjae Jeon  
**Category**: cs.AI  
**Published**: 2026-01-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.10729v1  

#### Abstract
Serving long-context LLMs is challenging because request lengths and batch composition vary during token generation, causing the memory footprint to fluctuate significantly at runtime. Offloading KV caches to host memory limits effective memory usage, but existing static and predetermined offloading...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：ORBITFLOW: SLO-Aware Long-Context LLM Serving with Fine-Grained KV Cache Reconfiguration

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现代 **Long-Context LLM** 服务面临显著挑战：请求长度和批处理组成在生成过程中动态变化，导致 **GPU 内存占用剧烈波动**。传统的 **KV Cache Offloading** 方法采用静态、统一的策略（如每 N 层 offload 一次），无法适应运行时内存需求的变化，导致：
- 过度的 CPU-GPU 数据传输
- 频繁的延迟尖峰（latency spikes）
- 大量的 **SLO（Service-Level Objective）违反**

尤其是在高负载下，长请求会垄断 GPU 资源，拖慢短请求，造成系统级性能下降。

---

### 提出的新方法与创新思路
论文提出了 **ORBITFLOW** —— 一种细粒度、自适应的 KV Cache 管理系统，核心创新如下：

#### （1）Fine-Grained & Per-Request KV Cache Reconfiguration
- 不再对整个 batch 统一 offload，而是为每个请求独立决定其各层 KV Cache 的驻留策略。
- 使用一个轻量级的 **ILP（Integer Linear Programming）求解器**，在 GPU 内存约束下，为每个请求动态优化 offload 距离，以最小化 SLO 违反。

#### （2）Runtime Feedback-Driven 动态重配置
- 在 token 生成过程中持续监控运行时反馈（如计算时间、通信开销、KV 缓存增长）。
- 当当前计划变得次优时，触发重新规划，实现 **动态适应 token 和 batch 维度的漂移（drift）**。

#### （3）Fallback 机制保障 SLO
- 在严重过载时，引入 **Pause-Resume** 机制：
  - 暂停内存占用大的“长尾”请求，释放资源给其他请求。
  - 利用 **Token Deposit** 机制缓冲已生成的 token，并按 SLO 规定速率输出，维持用户体验连续性。

#### （4）SLO-Aware 的优化目标
- 将 SLO 违反作为硬约束纳入 ILP 模型，确保系统优先满足延迟要求。
- 引入 **decode window** 概念，由求解器自主决定多久重新规划一次，平衡稳定性与响应性。

---

### 相比现有方法的优势
| 特性 | ORBITFLOW | 传统方法（如 FlexGen, DeepSpeed） |
|------|-----------|-----------------------------|
| Offloading Granularity | **Per-request, fine-grained** | Batch-uniform, layer-wise |
| Adaptivity | **Runtime feedback-driven** | Static or only at batch boundary |
| SLO Awareness | **Explicitly modeled in ILP** | Often ignored or loosely handled |
| Handling Long Requests | **Pause-Resume + Token Deposit** | No fallback, suffer from straggler |
| Compute-Communication Overlap | **Optimized per request** | Suboptimal due to uniform policy |

---

## 2. 核心实验方法和设置

### 数据集
- 使用 **ShareGPT** 数据集合成的 **long-context 请求轨迹**。
- 序列长度最长达 **400K tokens**，覆盖从短到长的广泛输入输出分布。
- 请求到达模式为 **Poisson 分布**，模拟真实场景下的突发性和变异性。

### 实验设置
- **模型**：LLaMA3-8B（单卡）、LLaMA3-70B（多卡）
- **硬件**：
  - 单卡：NVIDIA RTX A5000（24GB GPU，PCIe 3.0×16）
  - 多卡：4×RTX A6000（48GB，PCIe 4.0×16）
- **批大小**：默认为 4
- **总序列长度限制**：32K tokens / batch（单卡）

### 评估指标
| 指标 | 含义 |
|------|------|
| **TBT SLO Attainment** | Time-Between-Tokens，衡量连续输出 token 的延迟是否超标 |
| **TPOT SLO Attainment** | Time-Per-Output-Token，平均每个 token 的延迟达标率 |
| **P95/P99 Latency** | 尾部延迟，反映最差情况性能 |
| **Throughput** | 每分钟完成的请求数 |
| **End-to-End Latency** | 单个请求的总完成时间 |
| **GPU Memory Utilization** | GPU KV Cache 利用率 |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **DeepSpeed-Inference** | 每层都 offload，仅保留当前层在 GPU |
| **FlexGen** | 离线预设最优 offload 距离，静态策略 |
| **FlexGen+** | 支持在 batch 变化时动态调整 offload 距离 |
| **SLO-aware Offloading** | 考虑 SLO 的动态 offloading，但仍为 batch-uniform |
| **Dynamic Heuristic** | 启发式动态调整，不使用求解器 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（最高提升）
| 指标 | 提升幅度 | 对比基线 |
|------|---------|----------|
| **TBT SLO Attainment** | ↑ **48%** | 最佳基线 |
| **TPOT SLO Attainment** | ↑ **66%** | 最佳基线 |
| **P95 TBT Latency** | ↓ **38%** | 现有 offloading 方法 |
| **Throughput** | ↑ **3.3×** | DeepSpeed-Inference |

---

### 详细对比结果

#### ✅ SLO 达成率（Fig. 8a）
- ORBITFLOW 在高到达率和紧 SLO 下仍保持 **>75% TBT SLO 达成率**。
- 其他方法在高负载下迅速下降至 ~50%，而 ORBITFLOW 更鲁棒。

#### ✅ 尾部延迟（P95/P99）
- **P95 TBT Latency**：比 FlexGen 和 DeepSpeed 分别低 **21.7%** 和 **68.6%**。
- **P99 TBT Latency**：比 DeepSpeed 低 **65%**，比其他基线低 **3–9%**。

#### ✅ 吞吐量（Fig. 9d）
- 实现 **最高吞吐量**，比 DeepSpeed 高 **3.3×**，比其他基线高 **4–52%**。
- 原因：Pause-Resume 机制避免了长请求拖累整个 batch。

#### ⚠️ 端到端延迟（E2E Latency）
- ORBITFLOW 的 E2E 延迟 **比多数基线高 12–21%**。
- 原因：Pause-Resume 机制会推迟部分长请求的执行。
- 但 **Dynamic Heuristic**（无 Pause-Resume）E2E 最低，说明该机制是可控权衡。

#### ✅ GPU 内存利用率（Fig. 9c）
- ORBITFLOW 实现 **接近满载的 GPU 利用率**，优于 DeepSpeed（极低）和 FlexGen（保守）。

---

### 消融实验结果（Ablation Study）

#### （1）组件贡献分析（Fig. 14a）
逐步添加组件后的 TBT SLO 达成率（SLO Scale=1）：
- **Batch-Uniform**：45.1%
- + **Request-Wise Offloading**：54.5%
- + **Pause-Resume**：71.0%
- + **Token Deposit**：**85.6%**

👉 结论：三项技术协同作用，显著提升 SLO 达成。

#### （2）Token Deposit 效果（Fig. 14b）
在宽松 SLO 下为各方法添加 Token Deposit：
- **FlexGen+**：↑18%
- **Dynamic Heuristic**：↑11%
- **ORBITFLOW**：↑5%（因其本身已高效）

但在紧 SLO 下，ORBITFLOW 仍能额外提升 **20%**。

#### （3）Fallback 策略选择（Fig. 14c）
比较暂停不同请求的效果：
- **暂停最长请求**：效果最好 → 释放最多 GPU 内存
- **随机暂停** 或 **暂停最短请求**：效果差

#### （4）Solver 开销（Table 1）
| 工作负载 | Solver 调用次数/请求 | Solver 时间占比（E2E） |
|----------|------------------|---------------------|
| Both Static | 1.25 | 0.27% |
| Token Dynamic | 11.92 | 0.37% |
| Both Dynamic | 48.33 | **0.64%** |

👉 即使在最动态场景下，solver 开销也 <1%，且通过提前启动隐藏在计算中。

---

## 4. 关键结论和发现

### 主要发现
1. **静态 KV offloading 无法应对 long-context 场景的动态性**，尤其在 batch composition 和 token length 持续变化时表现糟糕。
2. **细粒度、按请求优化的 offloading 策略** 显著优于 batch-uniform 方法，能更好平衡资源分配。
3. **Solver-driven 动态规划 + Token Deposit + Pause-Resume** 三者结合，可在不牺牲准确性的前提下，有效掩盖延迟尖峰，提升 SLO 达成率。
4. ORBITFLOW 在 **不同 context length（8K–128K）、batch size、burstiness、分布式 TP 设置下均保持优越性能**。

---

### 方法的局限性
1. **Pause-Resume 会增加长请求的 E2E 延迟**，虽提升整体 SLO，但可能影响特定用户。
2. **目前仅支持单节点多 GPU（Tensor Parallelism）**，未扩展到 Pipeline Parallelism 或跨节点场景。
3. **ILP 求解器虽轻量，但仍受限于搜索空间设计**（仅考虑等距 offload），理论上可能错过更优非规则布局（但实验证明影响很小）。

---

### 未来工作方向
1. 扩展支持 **Pipeline Parallelism 和 Data Parallelism**，适配更大规模分布式部署。
2. 探索 **与 KV Pruning、Compression 方法的结合**，进一步降低内存压力。
3. 引入 **更智能的调度策略**（如 SRTF）与 ORBITFLOW 协同优化，缓解长请求聚集问题。
4. 将框架应用于 **多模态 LLM** 或 **推理-检索混合任务** 中的缓存管理。

---

> 🔗 **开源信息**：  
> 代码、数据和 artifact 已公开在 GitHub：[https://github.com/Heelim-Hong/ORBITFLOW](https://github.com/Heelim-Hong/ORBITFLOW)

</details>

---

### 11. [CTHA: Constrained Temporal Hierarchical Architecture for Stable Multi-Agent LLM Systems](https://arxiv.org/abs/2601.10738)

**Authors**: Percy Jardine  
**Category**: cs.AI  
**Published**: 2026-01-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.10738v1  

#### Abstract
Recently, multi-time-scale agent architectures have extended the ubiquitous single-loop paradigm by introducing temporal hierarchies with distinct cognitive layers. While yielding substantial performance gains, this diversification fundamentally compromises the coordination stability intrinsic to un...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CTHA: Constrained Temporal Hierarchical Architecture for Stable Multi-Agent LLM Systems

---

## 1. 论文的主要贡献和创新点

### 解决的问题
现有的多时间尺度（multi-time-scale）**Temporal Hierarchy (TH)** 架构虽然提升了LLM代理系统的表征能力，但由于层间通信无约束，导致以下三大稳定性问题：
- **Inter-Layer Conflict**（层间冲突）：高层指令与底层已承诺行为矛盾，引发决策不一致。
- **Unbounded Error Propagation**（误差无限传播）：误差在层级间呈指数级放大，导致决策崩溃。
- **Authority Violation**（权限越界）：快层干预慢层长期计划，或慢层干涉快层实时响应。

此外，无约束架构还带来**通信开销高**（$O(n^2)$）、**可扩展性差**等问题。

### 提出的新方法：CTHA
作者提出 **Constrained Temporal Hierarchical Architecture (CTHA)**，通过将层间通信空间投影到结构化流形（structured manifolds），并引入仲裁机制，恢复协调稳定性。其三大核心约束机制为：

1. **Message Contract Constraints**  
   定义三种标准化消息格式：
   - `Summary`（上行）：快层向慢层汇报执行摘要。
   - `Plan`（下行）：慢层向快层下发子目标。
   - `Policy`（广播）：最慢层（Institutional）发布全局规则。
   所有消息需符合预定义的JSON Schema，确保结构化、可解析、低延迟。

2. **Authority Manifold Constraints**  
   每一层的决策空间被限制在其时间尺度范围内，防止越权操作。例如：
   - Reflex层不能修改战略目标。
   - Strategic层不能直接调用工具。
   通过**投影函数** $P_{\mathcal{A}_l}$ 将越界决策映射回合法动作空间。

3. **Arbiter Resolution Constraints**  
   引入一个轻量级 **Arbiter** 模块，接收所有层的行动提案，通过优先级函数解决冲突，输出唯一确定的最终行动 $a_{\text{final}}$。优先级基于：
   - 时间紧迫性（Urgency）
   - 层级权威（默认慢层更高）
   - 自信度（Confidence）
   - 学习组件（learned neural network）

### 相比现有方法的优势
- **稳定性强**：从根本上抑制层间冲突、误差爆炸和权限越界。
- **通信高效**：消息复杂度从 $O(n^2)$ 降至 $O(n)$。
- **可扩展性好**：支持更深的层级结构而不失稳。
- **性能更优**：在复杂任务上显著优于单层、多代理和无约束TH系统。

---

## 2. 核心实验方法和设置

### 数据集
实验覆盖九个多样化基准，涵盖多种能力维度：
| 数据集 | 任务类型 |
|--------|----------|
| **ToolBench** | 工具调用（API执行） |
| **WebArena** | 网页导航与交互 |
| **SWE-Bench Verified** | 软件工程（GitHub问题修复） |
| **T2-Bench** | 多轮对话代理 |
| **AgentBench** | 多环境综合测试（OS、DB、KG等） |
| **ALFWorld** | 文本环境中多步规划 |
| **HotpotQA** | 多跳推理与检索 |
| **GAIA** | 多模态现实世界任务 |
| **SafetyBench** | 安全对抗测试 |

### 实验设置与评估指标
- **模型配置**：使用异构LLM栈，按层需求分配不同模型：
  - Institutional: DeepSeek-V3.2-Speciale（强推理）
  - Strategic: Kimi-K2
  - Tactical: Qwen3-32B
  - Reflex: GLM-4.6-9B（低延迟）
- **温度分层**：慢层使用更高temperature以鼓励探索。
- **选择性激活**：仅在必要时触发高层（如目标完成、异常检测）。
- **评估指标**：
  - 任务成功率（Success Rate）
  - 准确率（Accuracy）、F1、EM
  - 攻击成功率（ASR，越低越好）
  - 帮助性保留（Helpfulness Preservation, HP）
  - 吞吐量（Tasks/Hour）
  - 协调失败率、误差放大因子

### 基线方法对比
| 类别 | 基线方法 |
|------|--------|
| **Single-Scale Agents** | ReAct, Reflexion, AutoGPT, LATS |
| **Multi-Agent Systems** | MetaGPT, AutoGen, AgentVerse |
| **Temporal Hierarchies** | Voyager, DEPS, Unconstrained TH（本文实现） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 13）
| 方法 | SWE-Bench Res. (%) | GAIA Acc. (%) | WebArena SR (%) | SafetyBench ASR (%) |
|------|---------------------|----------------|------------------|----------------------|
| **Unconstrained TH** | 51.2 | 44.8 | 19.8 | 24.6 |
| **CTHA-DS (Ours)** | **64.1** | **53.2** | **26.8** | **5.8** |
| **提升幅度** | **+12.9** | **+8.4** | **+7.0** | **↓76.4%** |

- 在 **SWE-Bench** 上提升 **+12.9%**，表明CTHA在复杂软件任务中优势显著。
- 在 **GAIA** 上提升 **+8.4%**，尤其在困难任务（Level 3）提升达 **+18.7%**。
- **安全性大幅提升**：攻击成功率从24.6%降至5.8%，帮助性从72.4%升至89.4%。

### 与基线方法的对比
- 相比最强单层基线 **LATS**：
  - SWE-Bench: +18.8%
  - GAIA: +11.9%
- 相比最佳多代理系统 **AutoGen**：
  - SWE-Bench: +19.3%
  - GAIA: +14.0%
- 相比无约束TH：**全面超越**，且稳定性显著提升。

### 消融实验结果（Table 17）
移除各组件对SWE-Bench性能的影响：
| 配置 | SWE-Bench Res. (%) | 相对下降 |
|------|---------------------|-----------|
| CTHA-Full | 64.1 | — |
| w/o Message Contracts | 58.7 | -5.4 |
| w/o Authority Manifolds | 55.2 | **-8.9** |
| w/o Arbiter Resolution | 57.4 | -6.7 |
| w/o All Constraints (Unconstrained TH) | 51.2 | -12.9 |

- **Authority Manifolds** 贡献最大，尤其在安全性和权限控制方面。
- 三者存在**协同效应**：组合使用效果远超单独移除之和。

### 效率分析（Table 20）
| 方法 | Tasks/Hour | Success Rate (%) | Successful Tasks/Hour |
|------|------------|------------------|-------------------------|
| Unconstrained TH | 52 | 51.2 | 26.6 |
| **CTHA-DS** | **108** | **64.1** | **69.2** |

- **吞吐量提升2.1倍**，成功任务数提升 **2.6倍**。
- 拜访选择性激活（平均仅1.8层活跃）和消息缓存所赐。

---

## 4. 关键结论和发现

### 主要发现
1. **无约束TH存在根本性稳定性缺陷**：误差增益可达 $10^3$，协调失败率高达62%。
2. **CTHA有效恢复协调稳定性**：
   - 协调失败减少 **89%**
   - 误差放大因子从47.3×降至1.12×
   - 权限越界减少94.2%
3. **结构化约束提升而非限制能力**：CTHA在性能、效率、安全性上均优于灵活但混乱的无约束系统。
4. **跨模型泛化性强**：在开源（CTHA-DS/Qwen）和闭源（CTHA-GPT）模型上均表现优异，证明改进源于架构而非特定模型。

### 方法的局限性
- **固定层级结构**：当前采用4层静态划分，缺乏动态适应任务复杂度的能力。
- **Arbiter训练数据依赖**：仲裁器在特定任务分布上训练，跨领域泛化可能受限。
- **简单任务开销**：对于短任务，多层架构引入不必要的复杂性（尽管仅增加12%延迟）。
- **上下文长度限制**：当前基于128K上下文，更长任务需额外记忆管理机制。

### 未来工作方向
1. **自适应层级学习**：让模型自动学习最优时间分解策略。
2. **多样化流形约束**：探索黎曼流形、李群等几何结构作为约束空间。
3. **多代理CTHA扩展**：多个CTHA代理协作，研究复杂环境中的涌现协调。
4. **形式化验证**：利用结构化协议对代理行为进行数学证明，确保安全性。
5. **持续学习**：利用Institutional层实现策略的在线更新与元学习。

---

> **总结**：CTHA通过引入**Message Contracts**、**Authority Manifolds** 和 **Arbiter Resolution** 三大约束机制，成功解决了多时间尺度LLM代理系统的稳定性难题，在保持高性能的同时实现了卓越的鲁棒性、安全性和效率，为构建可信赖的自主智能体提供了坚实架构基础。

</details>

---

### 12. [Efficient Protein Optimization via Structure-aware Hamiltonian Dynamics](https://arxiv.org/abs/2601.11012)

**Authors**: Jiahao Wang, Shuangjia Zheng  
**Category**: cs.AI  
**Published**: 2026-01-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.11012v1  

#### Abstract
The ability to engineer optimized protein variants has transformative potential for biotechnology and medicine. Prior sequence-based optimization methods struggle with the high-dimensional complexities due to the epistasis effect and the disregard for structural constraints. To address this, we prop...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Efficient Protein Optimization via Structure-aware Hamiltonian Dynamics**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
传统基于序列的蛋白质优化方法面临两大挑战：
- **高维复杂性与上位效应（epistasis）**：氨基酸突变之间的非线性相互作用导致适应度景观（fitness landscape）高度崎岖，难以有效探索。
- **忽略结构约束**：仅依赖序列信息的方法无法保证设计出的蛋白质具有稳定且功能相关的三维结构。

此外，现有黑箱优化方法在稀疏、噪声大的适应度景观中容易陷入局部最优，采样效率低下。

---

### **提出了什么新方法或新思路**
作者提出 **HADES**（**H**amiltonian **A**cquisition for **D**irected **E**volution in a **S**tructure-informed manner），一种结合贝叶斯优化与哈密顿动力学的结构感知蛋白质优化框架。其核心创新包括：

1. **结构感知的代理模型（Surrogate Model）**
   - 采用两阶段编码器-解码器架构：
     - 第一阶段学习突变体与野生型之间的 **RMSD**（Root Mean Square Deviation），利用 ESMFold 预测结构变化作为先验。
     - 第二阶段固定编码器，训练解码器预测适应度值。
   - 利用结构信息建模序列-结构-功能关系，构建更平滑、物理合理的适应度代理函数。

2. **基于哈密顿蒙特卡洛（HMC）的高效采样策略**
   - 在连续隐空间中引入 **Hamiltonian Dynamics** 进行远距离跳跃式采样，克服高维空间中的低接受率问题。
   - 引入动量变量加速向高适应度区域迁移，提升探索效率。

3. **离散化机制与虚拟边界（Virtual Barriers）**
   - 设计位置离散化过程，将连续状态转换为合法的一维蛋白序列（one-hot 表示）。
   - 提出“虚拟屏障”处理超出 [0,1] 范围的连续向量，通过反弹机制维持数值稳定性，减少离散误差。

4. **不确定性引导的候选选择（UCB）**
   - 使用模型集成估计预测不确定性，基于上置信界（Upper Confidence Bound, UCB）筛选候选序列，平衡探索与利用。

---

### **相比现有方法的优势**
| 维度 | HADES 的优势 |
|------|--------------|
| **搜索效率** | 借助 HMC 实现长距离跳跃，避免局部收敛，显著加快收敛速度 |
| **结构合理性** | 显式建模结构扰动（RMSD），确保生成序列对应稳定结构 |
| **泛化能力** | 结构先验缓解了上位效应带来的噪声影响，提升对未见突变组合的预测准确性 |
| **多样性保持** | 高 fDiv 分数表明能发现多个高适应度且序列差异大的解 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **GB1**：来自葡萄球菌蛋白 G 的 IgG 结合域，包含 149,361 个四点突变体，广泛用于测试多点突变优化。
- **PhoQ**：细菌组氨酸激酶受体，含 140,517 个四点突变体，适应度分布更稀疏，更具挑战性。

> 两个数据集均来自饱和突变实验，提供真实湿实验室测量的适应度标签，避免使用合成或模拟 oracle。

---

### **实验设置和评估指标**

#### **优化流程**
- 总共进行 $ N = 10 $ 轮查询，每轮调用 $ K = 100 $ 次真实适应度 oracle（即湿实验评估）。
- 每轮更新代理模型并生成新候选序列。

#### **评估指标**
| 指标 | 定义与意义 |
|------|-----------|
| **max fit.** | 所有已测样本中的最大适应度，衡量是否找到全局最优解 |
| **mean fit.** | 当前批次中平均适应度，反映整体优化质量 |
| **fDiv** | 修正的多样性得分，综合考虑序列编辑距离与成对平均适应度：<br>$$ \text{fDiv}(D) = \frac{\sum_{(x_i,x_j)\in D, i\neq j} d(x_i,x_j)(f(x_i)+f(x_j))}{|D|(|D|-1)} $$<br>防止低适应度但高多样性的虚假优势 |

所有结果基于 10 次独立运行取均值 ± 标准差。

---

### **基线方法对比**
| 方法 | 类型 | 简介 |
|------|------|------|
| **ESM2-zs** | Zero-shot | 使用预训练语言模型直接打分，无微调 |
| **BO (Bayesian Optimization)** | 序列空间贝叶斯优化 | 基于 Thompson Sampling 的标准 BO |
| **CMA-ES** | 进化算法 | 自适应协方差矩阵进化策略 |
| **AdaLead** | 贪婪搜索 | Hill-climbing 风格，选择预测最高者 |
| **PEX** | 局部探索 | 优先探索靠近野生型的邻近区域 |
| **EvoPlay** | 强化学习 | 自博弈 + MCTS 决策框架 |
| **HADES-L** | 对照版本 | 使用 Langevin Dynamics 替代 HMC |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（Table 1）**

| Task | Metric | ESM2-zs | BO | CMA-ES | AdaLead | PEX | EvoPlay | HADES-L | **HADES** |
|------|--------|--------|-----|--------|---------|-------|----------|----------|------------|
| **GB1** | max fit. | 1.00 | 0.57±0.15 | 0.69±0.16 | 0.84±0.15 | 0.83±0.17 | 0.91±0.09 | 0.93±0.14 | **1.00±0.00** |
|        | mean fit. | 0.03 | 0.08±0.01 | 0.28±0.07 | 0.49±0.05 | 0.51±0.06 | 0.54±0.03 | 0.51±0.06 | **0.59±0.02** |
|        | fDiv | 0.04 | 0.14±0.02 | 0.37±0.08 | 0.64±0.09 | 0.68±0.14 | 0.77±0.04 | 0.70±0.12 | **0.84±0.03** |
| **PhoQ** | max fit. | 0.32 | 0.27±0.05 | 0.47±0.23 | 0.62±0.21 | 0.46±0.05 | 0.69±0.22 | 0.72±0.24 | **0.80±0.25** |
|          | mean fit. | 0.01 | 0.05±0.01 | 0.13±0.03 | 0.19±0.03 | 0.20±0.01 | 0.18±0.02 | 0.20±0.02 | **0.22±0.02** |
|          | fDiv | 0.02 | 0.08±0.01 | 0.18±0.03 | 0.26±0.04 | 0.30±0.01 | 0.26±0.03 | 0.28±0.03 | **0.32±0.02** |

> ✅ **HADES 在所有任务和指标上均达到最佳表现**，尤其在 GB1 上实现了 **100% 找到全局最优解**（max fit. = 1.00）。

---

### **与基线方法的对比结果**
- **超越所有主流方法**：无论是进化算法（CMA-ES）、强化学习（EvoPlay）还是贪心策略（AdaLead），HADES 均显著领先。
- **优于结构启发式方法（PEX）**：说明全局结构感知比局部邻域探索更有效。
- **优于零样本模型（ESM2-zs）**：证明联合训练结构与功能模块的有效性。
- **优于 Langevin 版本（HADES-L）**：验证了 HMC 相较于 LMC 在长距离跳跃上的优越性。

---

### **消融实验结果（Table 2 & 3）**

| 变体 | GB1 (max fit.) | PhoQ (max fit.) | 说明 |
|------|----------------|------------------|------|
| **HADES (完整版)** | **1.00±0.00** | **0.80±0.25** | — |
| w/o HD（替换为随机突变） | 0.84±0.15 | 0.75±0.25 | HMC 对高效探索至关重要 |
| w/o structure（移除结构解码器） | 0.95±0.10 | 0.74±0.27 | 结构先验有助于提升鲁棒性和精度 |
| w/o ucb（不用不确定性估计） | 0.93±0.11 | 0.76±0.24 | UCB 显著增强探索能力 |
| w/o vb.（移除虚拟边界） | 0.97±0.09 | 0.63±0.25 | 虚拟边界有效防止梯度爆炸导致的离散失败 |

> 🔍 消融实验证明：**四个组件缺一不可**，尤其是虚拟边界在陡峭景观（如 PhoQ）中作用显著。

---

## **4. 关键结论和发现**

### **主要发现**
1. **结构信息是优化的关键先验**：通过建模突变引起的结构扰动（RMSD），可有效平滑适应度景观，提升模型泛化能力。
2. **HMC 极大提升了高维序列空间的采样效率**：相比传统 MH-MCMC 或 Langevin 动力学，HMC 能实现更大步长跳跃，快速逼近高适应度区域。
3. **连续隐空间 + 离散输出机制可行且高效**：将离散序列映射到连续空间进行梯度驱动优化，并通过离散化回投影，兼顾灵活性与可行性。
4. **HADES 能同时实现高性能与高多样性**：不仅找到最优解，还维持了解集的多样性（高 fDiv），有利于后续实验筛选。

---

### **方法的局限性**
- **依赖结构预测模型（ESMFold）**：虽然 ESMFold 快速可用，但仍存在预测误差，可能误导结构先验学习。
- **计算开销较高**：每次迭代需运行多次 HMC 轨迹（T=16）及 ESMFold 推理，在大规模任务中可能受限。
- **目前仅支持单目标优化**：尚未扩展至多目标（如稳定性 + 活性 + 表达量）联合优化。

---

### **未来工作方向**
1. **改进结构建模模块**：引入更精确的结构能量函数或联合训练结构生成器。
2. **多目标优化扩展**：结合 Pareto 优化或多任务学习框架。
3. **降低推理成本**：设计轻量化替代结构预测器，或缓存结构特征。
4. **应用于真实湿实验闭环系统**：与自动化实验平台对接，实现全自动蛋白质设计 pipeline。

---

> 📌 **总结一句话**：  
> **HADES 通过融合结构感知代理模型与哈密顿动力学采样，在蛋白质定向进化任务中实现了更高效率、更强鲁棒性与更好多样性，代表了当前 ML-guided protein design 的前沿进展。**

</details>

---

### 13. [NAACL: Noise-AwAre Verbal Confidence Calibration for LLMs in RAG Systems](https://arxiv.org/abs/2601.11004)

**Authors**: Jiayu Liu, Rui Wang, Qing Zong, Qingcheng Zeng, Tianshi Zheng, Haochen Shi, Dadi Guo, Baixuan Xu, Chunyang Li, Yangqiu Song  
**Category**: cs.CL  
**Published**: 2026-01-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.11004v1  

#### Abstract
Accurately assessing model confidence is essential for deploying large language models (LLMs) in mission-critical factual domains. While retrieval-augmented generation (RAG) is widely adopted to improve grounding, confidence calibration in RAG settings remains poorly understood. We conduct a systema...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：NAACL: Noise-AwAre Verbal Confidence Calibration for LLMs in RAG Systems

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文聚焦于 **Retrieval-Augmented Generation (RAG)** 系统中 **大语言模型 (LLMs)** 的 **verbal confidence calibration（口头置信度校准）** 问题。尽管 RAG 被广泛用于提升 LLM 的事实准确性，但研究发现，在存在检索噪声（如无关或反事实的上下文）时，LLMs 会表现出严重的 **过度自信（overconfidence）**，即其输出的置信度分数与实际正确率严重不匹配。

具体而言，当检索到的上下文中包含：
- **反事实（Counterfactual）** 证据（支持错误答案）
- **相关但无用（Relevant）** 或 **完全无关（Irrelevant）** 的段落

LLM 不仅可能生成错误答案，还会以高置信度“坚信”其错误，这在高风险应用中极具危害。

---

### **提出了什么新方法或新思路**

为解决上述问题，作者提出：

#### （1）**NAACL Rules（Noise-AwAre Confidence CaLibration Rules）**
一套指导模型在噪声环境下合理调整置信度的原则，包括三条核心规则：
- **Conflict Independence（冲突独立性）**：当多个高度相关的段落相互矛盾时，不应依赖外部上下文，应回退到模型自身的参数化知识（parametric knowledge），并降低置信度。
- **Noise Invariance（噪声不变性）**：对于无关段落，模型应明确忽略它们，推理结果不应受其影响。
- **Parametric Fallback（参数回退）**：当没有有用段落时，模型应回退到自身知识作答，并反映相应的不确定性。

#### （2）**NAACL 框架**
一个基于监督微调（SFT）的噪声感知校准框架，无需依赖更强的教师模型或强化学习。其核心流程如下：
- **自举训练数据构建**：利用约 2K 条 HotpotQA 数据，通过 **Best-of-N 采样** 和 **多阶段过滤**，生成高质量的训练样本。
- **显式中间判断**：要求模型在输出答案前，先对每个段落进行分类（高度相关 / 相关 / 无关）并判断组间一致性。
- **规则引导生成**：在提示中嵌入 NAACL Rules，引导模型在推理过程中显式应用这些规则。
- **监督微调（SFT）**：使用 LoRA 对模型进行微调，使其内化噪声感知能力。

---

### **相比现有方法的优势**

| 方面 | NAACL 的优势 |
|------|--------------|
| **方法范式** | 与依赖白盒信号（logits）或测试时采样的方法不同，NAACL 是一种 **黑盒、轻量级、可训练** 的解决方案，适用于闭源模型。 |
| **无需强教师模型** | 不依赖昂贵的教师模型（如 GPT-4）进行蒸馏或强化学习，成本更低，更易部署。 |
| **增强可解释性** | 通过强制模型输出 **段落判断（Passage Judgement）** 和 **规则应用过程**，提升了决策透明度。 |
| **泛化性强** | 在域内和域外均表现优异，且能适应不同数量的检索段落（如从 3 到 5）。 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

- **主评估数据集**（4个）：
  - **Natural Questions (NQ)**
  - **Bamboogle**
  - **StrategyQA**
  - **HotpotQA**

- **训练数据来源**：基于 **HotpotQA** 的 ~2K 示例，通过人工设计的提示（prompt）由 **Gemini-2.5-Pro** 合成噪声段落，构建训练集。

---

### **实验设置和评估指标**

#### **RAG 设置**
- **检索器**：BM25（稀疏）和 Contriever（稠密）
- **检索数量**：Top-3 段落
- **输入顺序**：随机打乱以避免位置偏差
- **基础模型**：4 个开源 LLMs
  - `Llama-3.1-8B-Instruct`
  - `Qwen2.5-7B-Instruct`
  - `DeepSeek-R1-Distill-Llama-8B`
  - `DeepSeek-R1-Distill-Qwen-7B`

#### **评估指标**
- **Expected Calibration Error (ECE)** ↓：衡量置信度与准确率之间的平均偏差，越低越好。
- **AUROC** ↑：衡量置信度区分正确/错误预测的能力，越高越好。

---

### **基线方法对比**

| 基线方法 | 描述 |
|---------|------|
| **Vanilla** | 直接生成答案和置信度 |
| **Chain-of-Thought (CoT)** | 引导模型逐步推理后再输出 |
| **Noise-aware Prompting** | 将 NAACL Rules 写入提示中，零样本应用 |
| **Ensemble** | 多次采样取平均置信度 |
| **Label-only SFT** | 仅使用 (answer, confidence) 对进行 SFT，不含中间推理过程 |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

| 方法 | 平均 ECE ↓ | 平均 AUROC ↑ |
|------|-----------|-------------|
| **NAACL** | **0.266** | **0.751** |
| Label-only SFT | 0.353 | 0.686 |
| Ensemble | 0.352 | 0.648 |
| CoT | 0.377 | 0.591 |
| Vanilla | 0.411 | 0.605 |

> 数据基于 `Llama-3.1-8B-Instruct` 在四个数据集上的平均表现（见 Table 2）。

- **NAACL 相比 CoT 提升显著**：
  - **ECE 下降 11%**（相对改善）
  - **AUROC 提升明显**，尤其在 Bamboogle 上达到 0.877

- **跨模型一致领先**：在所有 4 个 backbone 模型上，NAACL 均取得最佳或第二佳的 ECE 表现。

---

### **与基线方法的对比结果**

- **优于 Prompting 方法**：NAACL 显著优于 Vanilla、CoT 和 Noise-aware Prompting，说明仅靠提示无法根本解决噪声导致的校准失败。
- **优于 Ensemble**：NAACL 仅需一次推理即可超越需要多次采样的 Ensemble 方法，效率更高。
- **优于 Label-only SFT**：证明中间推理过程（如段落判断、规则应用）是性能提升的关键，而非简单拟合标签。

---

### **消融实验结果**

#### （1）**Noise-aware Prompting 的有效性**
- 即使不进行训练，仅将 NAACL Rules 写入提示，也能显著提升校准效果（平均 ECE 达 0.314），成为仅次于 NAACL 的第二好方法。
- 说明 **NAACL Rules 本身具有强大指导意义**。

#### （2）**Out-of-Distribution (OOD) 泛化能力**
- 在 NQ 和 Bamboogle 上将检索段落数从 3 增加到 5 进行测试：
  - NAACL 相比 Vanilla **ECE 降低 8%**
  - 表明模型学会了 **泛化的噪声识别能力**，而非过拟合固定格式。

#### （3）**对段落效用判断能力的提升**
- NAACL 显著增强了模型判断段落是否“高度相关”的能力：
  - 相比 Vanilla，**判断准确率提升约 10%**（instruction-tuned 模型）
  - 即使在 DeepSeek 蒸馏模型上也稳定提升约 5%

---

## 4. 关键结论和发现

### **主要发现**

1. **RAG 中的检索噪声是导致 LLM 置信度失校准的根本原因**：
   - 反事实段落导致 **虚假确定性（false certainty）**
   - 无关/相关段落也会系统性抬高置信度，造成 **信息膨胀诱导的过度自信**

2. **现有 prompting 方法无法有效缓解该问题**：
   - CoT、Vanilla 等方法在 RAG 场景下平均 ECE > 0.4，远高于可接受水平（>0.25 即视为差）

3. **NAACL 通过规则引导的监督训练，实现了本质性的改进**：
   - 不再是“拟合标签”，而是学会在推理中 **显式识别噪声 → 应用规则 → 调整置信度**
   - 提升了模型的 **认知可靠性（epistemic reliability）**

4. **可解释性增强**：
   - 用户可通过查看模型的 **段落分类** 和 **规则应用逻辑**，理解为何置信度高或低。

---

### **局限性**

1. **模型规模限制**：
   - 当前评估限于 7B–8B 参数的开源模型，未扩展至更大模型（如 70B+）或闭源模型（如 GPT-5）。

2. **噪声合成的局限性**：
   - 训练数据中的噪声为人工合成，真实世界中的检索错误可能更复杂、更细微。

3. **任务范围有限**：
   - 当前聚焦于短问答任务，尚未扩展到长文本生成（如摘要）、超长上下文或多跳推理代理场景。

4. **计算开销与注意力瓶颈**：
   - 在超长上下文或动态信息流中，逐段扫描和规则应用可能导致计算成本过高或出现 “lost-in-the-middle” 问题。

---

### **未来工作方向**

- 将 NAACL 扩展至 **长文本生成** 和 **Agent-based RAG** 场景。
- 探索更高效的 **噪声检测机制**，减少对完整推理链的依赖。
- 结合 **动态检索重排序** 或 **反馈机制**，实现端到端的鲁棒 RAG 系统。
- 研究在 **专业领域**（如医疗、法律）中的适用性和泛化能力。

---

> **总结**：NAACL 首次系统揭示了 RAG 中检索噪声对 LLM 置信度校准的破坏性影响，并提出了一套原则性、可训练、可解释的解决方案。它不仅显著提升了校准性能，也为构建 **可信、可靠、可解释的 RAG 系统** 提供了重要基础。

</details>

---

### 14. [Space-Optimal, Computation-Optimal, Topology-Agnostic, Throughput-Scalable Causal Delivery through Hybrid Buffering](https://arxiv.org/abs/2601.11487)

**Authors**: Paulo S\'ergio Almeida  
**Category**: cs.DC  
**Published**: 2026-01-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.11487v1  

#### Abstract
Message delivery respecting causal ordering (causal delivery) is one of the most classic and widely useful abstraction for inter-process communication in a distributed system. Most approaches tag messages with causality information and buffer them at the receiver until they can be safely delivered. ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Space-Optimal, Computation-Optimal, Topology-Agnostic, Throughput-Scalable Causal Delivery through Hybrid Buffering*

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

传统基于接收端缓冲（receiver-buffering）的 **causal delivery** 算法在大规模分布式系统中面临严重可扩展性问题。其主要瓶颈在于：

- **元数据开销大**：经典算法（如 RST、KS）需要 $O(n)$ 或 $O(n^2)$ 大小的元数据（如向量时钟、依赖矩阵），其中 $n$ 是进程总数，这在数千甚至上万进程的场景下不可接受。
- **计算复杂度高**：接收端处理消息需 $O(n)$ 或更高时间复杂度，限制了吞吐量。
- **纯发送端缓冲（sender-buffering）方法（如 MF、Cykas）虽元数据小，但吞吐不可扩展**：它们无法支持消息流水线（pipelining），吞吐受限于网络延迟。

此外，拓扑感知方法（如树形传播）虽然能避免元数据，但不适用于任意通信拓扑，缺乏通用性。

---

### **提出了什么新方法或新思路**

本文提出了一种全新的 **hybrid buffering**（混合缓冲）策略，结合发送端和接收端缓冲，以实现空间、计算和吞吐的全面优化。

#### **核心创新点：**

1. **提出 Sender Permission to Send (SPS) 策略**  
   定义了一个新的因果序保证机制：一个进程只有在所有“发生在前”的消息（由曾向它发过消息的其他进程发出）已被交付后，才允许将新消息 **network-send** 出去。  
   并证明：**SPS + FIFO ⇒ Causal Delivery**。

2. **引入 Hybrid Buffering 架构**  
   - **发送端缓冲（sender-buffering）**：用于实施 SPS，即延迟 `network-send` 直到满足许可条件。
   - **接收端缓冲（receiver-buffering）**：仅用于维护同一发送者的 FIFO 顺序，而非复杂的因果依赖。
   这种分工使得接收端无需存储全局因果状态，大幅降低元数据。

3. **设计了新型数据结构：sliding array 和 sliding map**  
   - 利用滑动窗口结构高效管理未确认（unacked）消息和未收到的许可（permits）。
   - 实现了 **摊销常数时间（amortized constant-time）** 的操作，确保计算最优。

4. **提出两个算法变体**：
   - **Basic Algorithm**：基于保守的 CSPS（Conservative SPS），易于理解和实现。
   - **SPS-Optimal Algorithm**：精确实现 SPS，在更多场景下减少延迟。

---

### **相比现有方法的优势**

| 特性 | 经典 Receiver-Buffering (RST/KS) | Sender-Buffering (MF/Cykas) | 本文 Hybrid 方法 |
|------|-------------------------------|-----------------------------|------------------|
| **元数据大小** | $O(n^2)$ 或 $O(n)$ | $O(1)$ | **$O(1)$** |
| **计算复杂度** | $O(n^2)$ 或 $O(n)$ | $O(1)$ | **摊销 $O(1)$** |
| **吞吐可扩展性** | ✅（支持流水线） | ❌（受限于RTT） | ✅（支持流水线） |
| **交付延迟** | 最优（最小） | 高（非因果延迟） | 非最优，但可控 |
| **拓扑无关性** | ✅ | ✅ | ✅ |
| **适用规模** | 小到中等规模 | 小规模 | **大规模（千级+进程）** |

> ✅ 本文是首个同时满足 **拓扑无关、元数据最优、计算最优、吞吐可扩展** 的因果交付算法。

---

## 2. 核心实验方法和设置

### **说明**

该论文 **未包含实际性能实验或仿真**。作者在第6节“Future Work”中明确指出：

> “After designing the new approach and algorithm, an obvious future work is performance evaluation.”  
> “Either way, a proper evaluation is beyond the current work, and will require a full paper…”

因此，以下内容为基于论文描述的 **理论分析与比较框架**，而非实测结果。

---

### **理论评估方法和设置**

#### **评估维度（Criteria）**

论文定义了五个关键评估标准，超越传统的“交付延迟最优”：

1. **Space Overhead**（空间开销）：消息元数据和进程内存占用。
2. **Delivery Latency**（交付延迟）：是否最优，是否存在非因果延迟。
3. **Throughput Scalability**（吞吐可扩展性）：是否受网络延迟限制，能否支持流水线。
4. **Liveness**（活性）：是否存在死锁或饥饿风险。
5. **Computation Time**（计算时间）：发送/接收端处理开销。

#### **对比基线方法**

- **Receiver-Buffering**：
  - RST [16]：经典算法，$O(n^2)$ 元数据。
  - KS [10]：元数据最优的 receiver-buffering 算法。
  - Newtop [9]：低元数据但引入无限延迟。
- **Sender-Buffering**：
  - MF [12]：经典发送端缓冲，单消息在途。
  - Cykas [20]：最新 sender-buffering，引入“许可”机制。

#### **系统假设**

- 异步不可靠网络（允许丢包、乱序、重复）。
- 进程不崩溃，但无全局拓扑知识。
- 支持 unicast 和 multicast。

---

## 3. 主要实验结果和性能指标

由于 **没有实际实验**，所有“结果”均为 **理论分析得出的复杂度和性质对比**。

### **关键性能指标（理论分析）**

| 指标 | 本文方法 | RST | KS | Cykas | MF |
|------|--------|-----|----|-------|-----|
| **消息元数据大小** | $O(1)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | $O(1)$ |
| **进程空间复杂度** | $O(i + o + b + p + u)$<br>（仅依赖入度出度） | $O(n^2 b)$ | $O(n^2 b)$ | $O(n o + b)$ | $O(b)$ |
| **发送端计算复杂度** | $O(1)$ | $O(1)$ | $O(n^2)$ | $O(n o)$ | $O(1)$ |
| **接收端计算复杂度** | $O(1)$ | $O(n^2)$ | $O(n^2)$ | $O(1)$ | $O(1)$ |
| **吞吐可扩展性** | ✅（支持多消息在途） | ✅ | ✅ | ❌（每目的地最多1条） | ❌（全局1条） |
| **是否支持流水线** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **交付延迟** | 非最优（sender-induced） | 最优 | 最优 | 非最优 | 非最优 |
| **活性保障** | ✅（无饥饿） | ✅ | ✅ | ❌（Cykas 存在 starvation） | ✅ |

> 注：$i$: 入度，$o$: 出度，$b$: 缓冲消息数，$p$: 缺失许可数，$u$: 未确认消息数。

### **消融分析（隐含）**

论文通过对比不同算法揭示了设计选择的影响：

- **纯 sender-buffering 无法支持流水线** → 必须引入 receiver-buffering 来解耦。
- **Cykas 的全局 `MODE` 变量导致信息丢失** → 本文使用 per-message permit tracking 避免此问题。
- **使用 sliding map 而非普通 map** → 实现摊销 $O(1)$ 查找，达成计算最优。

---

## 4. 关键结论和发现

### **主要发现**

1. **Sender-Buffering 单独不足以实现吞吐可扩展**：即使像 Cykas 这样的先进算法，因缺乏接收端缓冲，仍无法支持多消息流水线，吞吐受限于 RTT。
2. **Hybrid Buffering 是大规模系统的唯一可行路径**：在不牺牲拓扑无关性的前提下，必须结合 sender 和 receiver 缓冲才能兼顾元数据、计算和吞吐。
3. **SPS 是一个强大的新原语**：它提供了一种新的因果序构造方式，解耦了“交付”与“发送许可”，为新算法设计开辟了空间。
4. **本文算法是目前唯一满足四大属性的 topology-agnostic 算法**：
   - Space-Optimal ($O(1)$ metadata)
   - Computation-Optimal (amortized $O(1)$ ops)
   - Throughput-Scalable (supports pipelining)
   - Topology-Agnostic

---

### **方法的局限性**

- **交付延迟非最优**：相比 receiver-buffering 算法，存在 sender-induced 的非因果延迟。
- **算法复杂度较高**：尤其是 SPS-optimal 版本，涉及多个索引变量和统一缓冲区，实现难度大。
- **未考虑进程故障**：假设进程不崩溃，不适用于强容错场景。
- **缺乏实证验证**：所有优势均基于理论分析，尚未在真实系统中验证。

---

### **未来工作方向**

1. **性能评估**：在大规模微服务或 actor 系统中进行仿真或实测，评估延迟、吞吐和资源消耗。
2. **动态权衡机制**：设计运行时策略，在“发送延迟”和“发送少量元数据”之间动态权衡，例如对小消息使用 SPS，对大消息附加部分依赖信息。
3. **探索更多 Hybrid Algorithms**：研究如何利用 receiver-buffering 超越仅维护 FIFO，以进一步降低延迟。
4. **支持进程动态加入/退出**：增强系统的动态性和弹性。

---

## 总结

本文提出了一种革命性的 **hybrid buffering** 因果交付算法，首次实现了 **元数据最优、计算最优、吞吐可扩展且拓扑无关** 的完美组合。尽管牺牲了交付延迟最优性，但在大规模分布式系统（如微服务、actor 模型）中，其低开销和高吞吐特性使其成为最实用的选择。该工作为未来大规模因果一致性系统的设计奠定了重要基础。

</details>

---

### 15. [Multivariate LSTM-Based Forecasting for Renewable Energy: Enhancing Climate Change Mitigation](https://arxiv.org/abs/2601.10961)

**Authors**: Farshid Kamrani, Kristen Schell  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.10961v1  

#### Abstract
The increasing integration of renewable energy sources (RESs) into modern power systems presents significant opportunities but also notable challenges, primarily due to the inherent variability of RES generation. Accurate forecasting of RES generation is crucial for maintaining the reliability, stab...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Multivariate LSTM-Based Forecasting for Renewable Energy: Enhancing Climate Change Mitigation*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
可再生能源（RES）如光伏（PV）和风能具有显著的**时间变异性与不确定性**，这给现代电力系统的**可靠性、稳定性与经济调度（Economic Dispatch, ED）**带来了巨大挑战。传统的预测方法（如确定性模型和基于K-means聚类的随机规划）难以充分捕捉复杂的时空依赖关系，导致预测误差较大，进而引发实时运行中对高碳排放燃气机组的过度依赖。

### 提出的新方法与创新思路
本文提出了一种**多变量长短期记忆网络（Multivariate LSTM, M-LSTM）**用于可再生能源发电量的日前预测（Day-Ahead Forecasting），其核心创新包括：

- **多区域输入建模**：不仅利用目标区域的历史发电数据，还融合**邻近区域的RES历史数据**作为输入，以捕捉空间相关性。
- **考虑市场时序结构**：在预测设计中显式考虑了**日前市场关闭与实时市场启动之间的时间间隔**，提升实际应用场景下的适用性。
- **物理先验知识嵌入**：对于光伏发电，在预定义的“黑暗时段”（夜间）直接将输出设为零，避免无效预测，增强模型物理一致性。
- **端到端学习框架**：通过深度学习自动提取非线性特征与时序依赖，无需手动构造代表性场景。

### 相比现有方法的优势
相比传统方法（如K-means聚类生成典型日、月均值法），M-LSTM能够：
- 更好地捕捉**长期时间依赖性和跨区域的空间相关性**；
- 显著降低预测误差，从而减少因预测偏差引起的备用需求；
- 在系统层面实现更低的CO₂排放、更少的负荷削减（Load Shedding）和更高的可再生能源利用率。

---

## 2. 核心实验方法和设置

### 数据集
- 使用加拿大阿尔伯塔省（Alberta）三个不同规划区域的真实历史数据；
- 时间跨度：一年（8760小时），时间分辨率：每小时；
- 特征维度：每个样本包含3个区域的PV发电功率（共3个features）；
- 数据来源：[AESO Historical Generation Data](https://www.aeso.ca/market/market-and-system-reporting/data-requests/historical-generation-data)

### 实验设置
- **预测任务**：基于过去24小时（look-back window = 24）的多区域PV发电数据，预测目标区域下一时刻的PV出力；
- **模型结构**：
  - 两层LSTM：第一层64个unit，第二层32个unit，均使用ReLU激活函数；
  - Dropout层防止过拟合；
  - 全连接层输出最终预测；
  - 使用Adam优化器，损失函数为Mean Squared Error (MSE)；
  - 输入数据进行归一化处理，输出后反归一化。
- **训练策略**：滑动窗口方式构建样本，输入序列长度为24，预测步长m=1。

### 评估指标
- **Normalized Mean Absolute Error (NMAE)**：用于衡量预测精度；
- **系统级影响指标**：
  - 燃气机组出力（Gas-fired unit output）
  - CO₂排放量（kg）
  - 负荷削减量（Load Shedding, MW）
  - 可再生能源弃电（RES Spillage, MW）
  - 日前+实时总成本（DA+RT Cost, $）

### 基线方法对比
三种预测模型用于比较：
1. **Case 1**：基于K-means聚类生成代表性日的方法（Kamrani et al. [2021b]）
2. **Case 2**：按月份每小时取平均值（Monthly Average per Hour）
3. **Case 3**：本文提出的M-LSTM模型

所有预测结果均应用于**日前经济调度（DA ED）**，并在**实时经济调度（RT ED）**中根据真实值调整系统运行状态。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见Table 1）

| 指标 | Case 1 (K-means) | Case 2 (Monthly Avg) | Case 3 (M-LSTM) |
|------|------------------|------------------------|------------------|
| Gas-fired unit (MW) | 263.7 | 140.56 | **108.7** |
| CO₂ emission (kg) | 53,267 | 28,393 | **21,957** |
| Load Shedding (MW) | 121.7 | 5.79 | **0** |
| RES Spillage (MW) | 0 | 0 | 5 |
| DA+RT Cost ($) | 53,043 | 44,148 | **43,490** |
| NMAE | 1.57 | 0.59 | **0.46** |

### 与基线方法的对比结果
- **预测精度**：M-LSTM的NMAE仅为0.46，远低于K-means（1.57）和月均值法（0.59），表明其具备最强的预测能力。
- **减排效果**：相较于Case 1，M-LSTM实现了**58%的CO₂减排**（从53,267 kg降至21,957 kg）。
- **系统可靠性**：完全消除负荷削减（0 MW），而其他两种方法分别高达121.7 MW和5.79 MW。
- **运行成本**：总成本最低（$43,490），优于其他两个案例。
- **弃电分析**：M-LSTM出现少量弃电（5 MW），但这反映的是**准确预测下真实过剩发电**，而非保守估计所致；相比之下，其他方法无弃电是因为低估或过度保守调度。

> 图2显示，M-LSTM对应的DA与RT之间的功率偏差最小，说明其预测最接近真实值。

### 消融实验（未明确开展）
文中未提供正式的消融研究（ablation study），但通过以下设计体现了关键组件的作用：
- 引入邻区数据 → 提升空间感知能力；
- 夜间强制置零 → 提高物理合理性与预测一致性；
- 多变量输入 → 捕捉跨区域相关性。

这些设计共同提升了整体性能。

---

## 4. 关键结论和发现

### 主要发现
- M-LSTM能够有效建模多区域RES之间的**复杂时空依赖关系**，显著提升预测精度（NMAE下降69% vs K-means）；
- 高精度预测直接转化为**更低的燃气机组调用频率与出力水平**，从而大幅减少CO₂排放；
- 准确的日前预测有助于实现更优的经济调度决策，**几乎消除负荷削减**，提高供电可靠性；
- 尽管出现轻微弃电，但这是系统高效运行的表现——即尽可能多地使用可再生能源，并仅在无法消纳时放弃多余电量；
- 更精确的预测带来更低的整体系统运行成本。

### 方法的局限性
- 模型依赖高质量、高时间分辨率的历史数据，在数据缺失或噪声严重的情况下性能可能下降；
- 当前仅针对PV发电建模，尚未扩展至风能或其他混合RES组合；
- 模型未考虑天气预报等外部协变量（如辐照度、温度、风速），可能限制进一步提升潜力；
- 所有实验基于单一省份数据，泛化能力需在更多地理区域验证。

### 未来工作方向
- 引入气象数据作为额外输入特征，构建**多模态预测模型**；
- 探索Transformer架构（如Temporal Fusion Transformer）以进一步提升长期预测能力；
- 将该方法推广至**风电+光伏联合预测**及跨区域电网协同调度；
- 结合强化学习或在线学习机制，实现动态适应变化环境的能力；
- 在更大规模电网中部署并评估其对碳边际排放（Marginal CO₂ Emissions）的影响。

---

> ✅ **总结一句话**：  
> 本论文提出的M-LSTM模型通过融合多区域历史数据与物理先验知识，显著提升了可再生能源预测精度，并在系统层面实现了**更低排放、更高可靠性和更优经济性**，为AI驱动的清洁能源转型提供了有力支持。

</details>

---

### 16. [Factored Value Functions for Graph-Based Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2601.11401)

**Authors**: Ahmed Rashwan, Keith Briggs, Chris Budd, Lisa Kreusser  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.11401v1  

#### Abstract
Credit assignment is a core challenge in multi-agent reinforcement learning (MARL), especially in large-scale systems with structured, local interactions. Graph-based Markov decision processes (GMDPs) capture such settings via an influence graph, but standard critics are poorly aligned with this str...

---

### 17. [Extractive summarization on a CMOS Ising machine](https://arxiv.org/abs/2601.11491)

**Authors**: Ziqing Zeng, Abhimanyu Kumar, Chris H. Kim, Ulya R. Karpuzcu, Sachin S. Sapatnekar  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2601.11491v1  

#### Abstract
Extractive summarization (ES) aims to generate a concise summary by selecting a subset of sentences from a document while maximizing relevance and minimizing redundancy. Although modern ES systems achieve high accuracy using powerful neural models, their deployment typically relies on CPU or GPU inf...

---

### 18. [Japanese AI Agent System on Human Papillomavirus Vaccination: System Design](https://arxiv.org/abs/2601.10718)

**Authors**: Junyu Liu, Siwen Yang, Dexiu Ma, Qian Niu, Zequn Zhang, Momoko Nagai-Tanima, Tomoki Aoyama  
**Category**: cs.AI  
**Published**: 2026-01-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2601.10718v1  

#### Abstract
Human papillomavirus (HPV) vaccine hesitancy poses significant public health challenges, particularly in Japan where proactive vaccination recommendations were suspended from 2013 to 2021. The resulting information gap is exacerbated by misinformation on social media, and traditional ways cannot sim...

---

### 19. [Health Facility Location in Ethiopia: Leveraging LLMs to Integrate Expert Knowledge into Algorithmic Planning](https://arxiv.org/abs/2601.11479)

**Authors**: Yohai Trabelsi, Guojun Xiong, Fentabil Getnet, St\'ephane Verguet, Milind Tambe  
**Category**: cs.AI  
**Published**: 2026-01-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2601.11479v1  

#### Abstract
Ethiopia's Ministry of Health is upgrading health posts to improve access to essential services, particularly in rural areas. Limited resources, however, require careful prioritization of which facilities to upgrade to maximize population coverage while accounting for diverse expert and stakeholder ...

---

### 20. [DialDefer: A Framework for Detecting and Mitigating LLM Dialogic Deference](https://arxiv.org/abs/2601.10896)

**Authors**: Parisa Rabbani, Priyam Sahoo, Ruben Mathew, Aishee Mondal, Harshita Ketharaman, Nimet Beyza Bozdag, Dilek Hakkani-T\"ur  
**Category**: cs.CL  
**Published**: 2026-01-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2601.10896v1  

#### Abstract
LLMs are increasingly used as third-party judges, yet their reliability when evaluating speakers in dialogue remains poorly understood. We show that LLMs judge identical claims differently depending on framing: the same content elicits different verdicts when presented as a statement to verify ("Is ...

---

### 21. [Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](https://arxiv.org/abs/2601.11038)

**Authors**: Xuanming Zhang, Shwan Ashrafi, Aziza Mirsaidova, Amir Rezaeian, Miguel Ballesteros, Lydia B. Chilton, Zhou Yu, Dan Roth  
**Category**: cs.CL  
**Published**: 2026-01-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2601.11038v1  

#### Abstract
We study the reasoning behavior of large language models (LLMs) under limited computation budgets. In such settings, producing useful partial solutions quickly is often more practical than exhaustive reasoning, which incurs high inference costs. Many real-world tasks, such as trip planning, require ...

---

### 22. [Reasoning in Trees: Improving Retrieval-Augmented Generation for Multi-Hop Question Answering](https://arxiv.org/abs/2601.11255)

**Authors**: Yuling Shi, Maolin Sun, Zijun Liu, Mo Yang, Yixiong Fang, Tianran Sun, Xiaodong Gu  
**Category**: cs.CL  
**Published**: 2026-01-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2601.11255v1  

#### Abstract
Retrieval-Augmented Generation (RAG) has demonstrated significant effectiveness in enhancing large language models (LLMs) for complex multi-hop question answering (QA). For multi-hop QA tasks, current iterative approaches predominantly rely on LLMs to self-guide and plan multi-step exploration paths...

---

### 23. [CTest-Metric: A Unified Framework to Assess Clinical Validity of Metrics for CT Report Generation](https://arxiv.org/abs/2601.11488)

**Authors**: Vanshali Sharma, Andrea Mia Bejar, Gorkem Durak, Ulas Bagci  
**Category**: cs.CL  
**Published**: 2026-01-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2601.11488v1  

#### Abstract
In the generative AI era, where even critical medical tasks are increasingly automated, radiology report generation (RRG) continues to rely on suboptimal metrics for quality assessment. Developing domain-specific metrics has therefore been an active area of research, yet it remains challenging due t...

---

### 24. [Theoretically and Practically Efficient Resistance Distance Computation on Large Graphs](https://arxiv.org/abs/2601.11159)

**Authors**: Yichun Yang, Longlong Lin, Rong-Hua Li, Meihao Liao, Guoren Wang  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2601.11159v1  

#### Abstract
The computation of resistance distance is pivotal in a wide range of graph analysis applications, including graph clustering, link prediction, and graph neural networks. Despite its foundational importance, efficient algorithms for computing resistance distances on large graphs are still lacking. Ex...

---

### 25. [FORESTLLM: Large Language Models Make Random Forest Great on Few-shot Tabular Learning](https://arxiv.org/abs/2601.11311)

**Authors**: Zhihan Yang, Jiaqi Wei, Xiang Zhang, Haoyu Dong, Yiwen Wang, Xiaoke Guo, Pengkun Zhang, Yiwei Xu, Chenyu You  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2601.11311v1  

#### Abstract
Tabular data high-stakes critical decision-making in domains such as finance, healthcare, and scientific discovery. Yet, learning effectively from tabular data in few-shot settings, where labeled examples are scarce, remains a fundamental challenge. Traditional tree-based methods often falter in the...

---

### 26. [Latent Space Inference via Paired Autoencoders](https://arxiv.org/abs/2601.11397)

**Authors**: Emma Hart, Bas Peters, Julianne Chung, Matthias Chung  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2601.11397v1  

#### Abstract
This work describes a novel data-driven latent space inference framework built on paired autoencoders to handle observational inconsistencies when solving inverse problems. Our approach uses two autoencoders, one for the parameter space and one for the observation space, connected by learned mapping...

---

### 27. [QUPID: A Partitioned Quantum Neural Network for Anomaly Detection in Smart Grid](https://arxiv.org/abs/2601.11500)

**Authors**: Hoang M. Ngo, Tre' R. Jeter, Jung Taek Seo, My T. Thai  
**Category**: cs.LG  
**Published**: 2026-01-19  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2601.11500v1  

#### Abstract
Smart grid infrastructures have revolutionized energy distribution, but their day-to-day operations require robust anomaly detection methods to counter risks associated with cyber-physical threats and system faults potentially caused by natural disasters, equipment malfunctions, and cyber attacks. C...

---

### 28. [AdaMARP: An Adaptive Multi-Agent Interaction Framework for General Immersive Role-Playing](https://arxiv.org/abs/2601.11007)

**Authors**: Zhenhua Xu, Dongsheng Chen, Shuo Wang, Jian Li, Chengjie Wang, Meng Han, Yabiao Wang  
**Category**: cs.AI  
**Published**: 2026-01-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2601.11007v1  

#### Abstract
LLM role-playing aims to portray arbitrary characters in interactive narratives, yet existing systems often suffer from limited immersion and adaptability. They typically under-model dynamic environmental information and assume largely static scenes and casts, offering insufficient support for multi...

---

### 29. [ReCreate: Reasoning and Creating Domain Agents Driven by Experience](https://arxiv.org/abs/2601.11100)

**Authors**: Zhezheng Hao, Hong Wang, Jian Luo, Jianqing Zhang, Yuyan Zhou, Qiang Lin, Can Wang, Hande Dong, Jiawei Chen  
**Category**: cs.AI  
**Published**: 2026-01-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2601.11100v1  

#### Abstract
Large Language Model agents are reshaping the industrial landscape. However, most practical agents remain human-designed because tasks differ widely, making them labor-intensive to build. This situation poses a central question: can we automatically create and adapt domain agents in the wild? While ...

---

### 30. [Do We Always Need Query-Level Workflows? Rethinking Agentic Workflow Generation for Multi-Agent Systems](https://arxiv.org/abs/2601.11147)

**Authors**: Zixu Wang, Bingbing Xu, Yige Yuan, Huawei Shen, Xueqi Cheng  
**Category**: cs.AI  
**Published**: 2026-01-19  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2601.11147v1  

#### Abstract
Multi-Agent Systems (MAS) built on large language models typically solve complex tasks by coordinating multiple agents through workflows. Existing approaches generates workflows either at task level or query level, but their relative costs and benefits remain unclear. After rethinking and empirical ...

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
