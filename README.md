# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-04-29 07:45:21 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [CacheFlow: Efficient LLM Serving with 3D-Parallel KV Cache Restoration](https://arxiv.org/abs/2604.25080)

**Authors**: Sean Nian, Jiahao Fang, Qilong Feng, Zhiyu Wu, Fan Lai  
**Category**: cs.DC  
**Published**: 2026-04-29  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2604.25080v1  

#### Abstract
KV cache restoration has emerged as a dominant bottleneck in serving long-context LLM workloads, including multi-turn conversations, retrieval-augmented generation, and agentic pipelines. Existing approaches treat restoration as a per-request tradeoff between recomputation and I/O transfer, recomput...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《CacheFlow: Efficient LLM Serving with 3D-Parallel KV Cache Restoration》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题  
在长上下文 LLM 推理场景中（如多轮对话、RAG、智能体流水线），**KV cache restoration** 已成为影响 **Time-To-First-Token (TTFT)** 的关键瓶颈。现有方法通常将恢复过程简化为单个请求内的“重计算 vs I/O 加载”二元选择，存在以下问题：

- 忽略了 **token、layer 和 GPU 之间**的细粒度并行潜力；
- 未考虑 **批处理场景下的资源争用与 straggler 效应**；
- 重计算成本随序列长度呈超线性增长（因 attention 的 $ O(n^2) $ 复杂度），导致长序列效率低下。

### 提出了什么新方法或新思路  
提出 **CacheFlow** —— 一种将 KV cache 恢复重构为**三维并行执行问题**的新框架，其核心思想是：

- 将恢复任务分解为具有结构依赖关系的细粒度单元；
- 引入统一的 **3D 并行抽象**：
  - **Token-wise parallelism**：利用因果依赖，在前向重计算早期 token 的同时，反向从存储加载后期 token；
  - **Layer-wise parallelism**：利用模型层间依赖，底层向前重算，高层向后加载；
  - **Multi-GPU parallelism**：通过传递轻量级边界激活状态（boundary activations），实现各 GPU 分片独立并发恢复本地 KV cache。

此外，设计了一个 **batch-aware two-pointer scheduler**，动态调度多个请求间的 compute 和 I/O 资源，优先服务能带来最大重计算节省的请求。

### 相比现有方法的优势  
- 实现了 **fine-grained overlap of recomputation and I/O**，显著降低 TTFT；
- 支持批处理下全局优化，缓解资源争用；
- 在不同硬件（L40S/A100/H100）、带宽条件（10–80 Gbps）和模型架构（dense/MoE）下均表现鲁棒；
- 不修改底层模型或应用逻辑，可集成到 vLLM、LMCache 等主流 serving stack。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集  
构建了三个真实世界风格的工作负载，来源于公开数据集：

| 数据集 | 描述 |
|-------|------|
| **LMSYS-Chat** | 来自 ChatGPT 的多轮对话轨迹，反映典型 chatbot 场景中的长前缀复用模式 |
| **WildChat** | 开放域大规模对话语料，涵盖多种语言与任务，具有广泛的 prefix 长度分布 |
| **SWE-Bench** | 编程智能体基准测试，涉及对共享代码库上下文的重复工具调用，体现系统性前缀复用 |

这些工作负载覆盖了从短到超长上下文（最高达 32K tokens）的不同场景。

### 实验设置和评估指标  

#### 模型
- **Qwen3-8B**（dense）
- **Llama-3.1-8B**（dense）
- **Qwen3-30B-A3B**（MoE 架构，active 3B parameters）

#### 硬件环境
- 单卡或多卡部署：NVIDIA **L40S (46GB)**、**A100 (40GB)**、**H100 (80GB)**
- I/O 带宽模拟：**10 Gbps**（典型云间节点传输）、**40 Gbps**（SSD 读取）、**80 Gbps**（InfiniBand/RoCE）

#### 评估指标
- 主要指标：**Time-To-First-Token (TTFT)** —— 用户感知延迟的关键指标
- 辅助指标：
  - GPU 利用率（Compute Utilization）
  - I/O 带宽利用率
  - 批处理规模下的平均与尾部延迟（P90/P99）

### 基线方法对比  
与以下先进系统进行比较：

| 基线 | 类型 | 特点 |
|------|------|------|
| **vLLM** | 仅重计算（recomputation-only） | 标准 prefill 流程，代表 compute-bound 极端 |
| **LMCache** | 仅 I/O 加载（pure loading） | 代表 I/O-bound 极端，state-of-the-art offloading 系统 |
| **SGLang + HiCache** | 存储层级缓存扩展 | 支持跨内存层级的 KV 缓存管理 |
| **Cake** | 混合恢复（hybrid） | 当前最先进的混合策略，按 token 维度划分恢复区域 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据  
- **整体 TTFT 减少 10%–62%**，相比所有基线取得一致领先；
- 在 **LMSys-Chat 和 SWE-Bench** 上提升最明显（高达 **1.7× 更快**），因其包含更多长上下文请求；
- 尾部延迟（P90–P99）改善尤为显著，表明有效缓解了 straggler 问题。

### 与基线方法的对比结果  
| 对比项 | 结果 |
|--------|------|
| vs **vLLM** | 最高提速 **1.7×**，尤其在长序列（>20K tokens）时优势扩大 |
| vs **LMCache** | 在低带宽（10 Gbps）下仍优于纯加载方案，说明合理混合更优 |
| vs **Cake** | 进一步减少 TTFT **10%–25%**，得益于多维并行与批感知调度 |
| vs **SGLang** | 显著降低延迟，特别是在 MoE 模型上 |

> 图4显示：CacheFlow 的 CDF 曲线全面左移，意味着在所有百分位上都实现了更低的 TTFT。

### 消融实验结果（Ablation Studies）

| 实验 | 发现 |
|------|------|
| **禁用 Multi-GPU 并行**（图7） | 平均恢复延迟从 **0.21s → 0.29s**（↑38%），证明 3D 并行有效性 |
| **仅保留 2D 并行** | 仍比 vLLM 快 **24%**，说明两指针机制本身已具强效 |
| **不同 I/O 带宽测试**（图8） | 在 40 Gbps 下提速 **1.7×**，80 Gbps 下 **1.5×**，验证自适应能力 |
| **不同 GPU 硬件**（图9） | 在 L40S 和 A100 上分别提速 **1.6× 和 1.5×**，展现跨平台鲁棒性 |
| **批大小变化**（图10） | 批量越大（batch=8），相对增益越明显（**1.6×–2.6×**），凸显 batch-aware 调度价值 |

---

## 4. 关键结论和发现

### 论文的主要发现  
- **KV cache restoration 不应视为单一操作**，而是一个可分解、可并行化的多维调度问题；
- **token、layer、GPU 三个维度存在天然并行结构**，可通过两指针 meet-in-the-middle 策略高效协同；
- **批处理下的资源争用必须被显式建模**，简单的 per-request 优化无法最大化系统吞吐；
- **最优恢复时间受 harmonic mean bound 控制**：$ T^* \propto \frac{T_{\text{comp}} \cdot T_{\text{io}}}{T_{\text{comp}} + T_{\text{io}}} $，CacheFlow 接近该理论上限；
- **multi-GPU 并行可带来接近线性的加速比**（公式推导得 $ S $ 倍 speedup）；

### 方法的局限性  
- 依赖于 **pipeline parallelism** 的部署方式，对于 fully replicated 架构支持有限；
- 当前调度假设边界激活状态能快速获取，若这部分也成为瓶颈需进一步优化；
- 对 extremely sparse MoE 或非 transformer 架构的泛化尚未验证。

### 未来工作方向  
- 扩展至 **multi-agent pipeline 中的跨请求 KV 共享与预取**；
- 结合 **KV cache compression 技术**（如 DiffKV、BTP）进一步减小 I/O 开销；
- 动态调整 chunk size 与策略切换阈值 $ L_\Delta $，实现运行时自适应；
- 探索与 **prefetching** 和 **lifetime prediction**（如 Continuum）系统的联合优化。

--- 

> ✅ 总结一句话：**CacheFlow 将 KV cache 恢复从“重算 or 加载”的粗粒度决策升级为“何时、何地、如何并行恢复”的精细调度问题，在真实场景中实现了高达 62% 的 TTFT 降低，推动了长上下文 LLM serving 的效率边界。**

</details>

---

### 2. [FED-FSTQ: Fisher-Guided Token Quantization for Communication-Efficient Federated Fine-Tuning of LLMs on Edge Devices](https://arxiv.org/abs/2604.25421)

**Authors**: Changyu Li, Shuanghong Huang, Jiashen Liu, Ming Lei, Jidu Xing, Kaishun Wu, Lu Wang, Fei Luo  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2604.25421v1  

#### Abstract
Federated fine-tuning provides a practical route to adapt large language models (LLMs) on edge devices without centralizing private data, yet in mobile deployments the training wall-clock is often bottlenecked by straggler-limited uplink communication under heterogeneous bandwidth and intermittent p...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：FED-FSTQ: Fisher-Guided Token Quantization for Communication-Efficient Federated Fine-Tuning of LLMs on Edge Devices

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在 **Federated Learning (FL)** 场景下，对 **Large Language Models (LLMs)** 在边缘设备上进行微调时，面临严重的通信瓶颈：
- 上行链路带宽异构且有限（heterogeneous bandwidth），导致训练过程被“**straggler-limited**”（最慢客户端决定整体进度）。
- 即使采用 **Parameter-Efficient Fine-Tuning (PEFT)** 如 LoRA，每轮传输的数据量仍然巨大。
- 在 **Non-IID 数据分布** 下，均匀压缩（uniform compression）会丢失稀有但任务关键的 token（如医学文本中的否定词、代码中的分隔符），损害模型性能。

### 提出了什么新方法或新思路
提出 **FED-FSTQ**（Fisher-Spectrum-aware Token Quantization），一种基于 **Fisher 信息引导的 token 级量化系统原语**，其核心思想是：
- 利用轻量级的 **token-level Fisher 代理信号** 来估计每个 token 对损失函数的敏感度（即语义重要性）。
- 结合 **重要性感知的 token 选择** 与 **非均匀混合精度量化**（mixed-precision quantization），将高保真度分配给信息量大的 token，抑制冗余传输。

#### 主要创新点：
- **首次将 Fisher 信息作为通信控制原语**：不同于传统用于优化的 Fisher 用途，FED-FSTQ 将其用于动态调控通信资源分配。
- **模型无关、即插即用**：可无缝集成到标准联邦 PEFT 流程（如 FedAvg + LoRA）中，无需修改服务器聚合规则。
- **支持异构带宽客户端**：通过紧凑的稀疏消息打包（sparse message packing）适应不同网络条件。
- **双用途设计**：训练阶段的 Fisher 掩码也可用于推理阶段的 token 减少，进一步提升端到端效率。

### 相比现有方法的优势
| 维度 | 现有方法（如 QSGD, Top-k, Fed-ToMe） | FED-FSTQ |
|------|----------------------------------------|---------|
| 压缩依据 | 参数级、幅度驱动（magnitude-driven）、均匀或启发式 | **token 级、语义敏感度驱动（Fisher-guided）** |
| 语义保留 | 易丢弃稀有但关键 token（如 negation） | **优先保留高 Fisher 敏感度 token** |
| 系统兼容性 | 多数需特定解码/聚合机制 | **完全兼容 FedAvg 聚合，无需服务端改动** |
| 部署可行性 | 高内存/计算开销 | **轻量级实现，适用于 Jetson 等边缘设备** |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
- **Fed-Aya**：多语言指令/问答数据集，模拟极端语言异构性（Dirichlet α=0.1），涵盖 ar, en, es, fr, pt, ru, te, zh 等语言。
- **Fed-Med**：基于 PubMedQA 构建的医疗问答数据集，测试对稀有医学实体和逻辑操作符（如否定）的保留能力。
- **Fed-Code**：基于 CodeAlpaca-20k 的代码生成任务，评估语法正确性（Pass@1）。

### 实验设置和评估指标

#### 系统设置
- 客户端数量：K = 100，每轮采样 10 个客户端参与（10% 参与率）。
- 边缘测试平台：虚拟边缘测试床，模拟 **4G/LTE 异构上行链路**，包含两种 profile：
  - **Controlled LTE-20Mbps**：固定速率，用于分析负载影响。
  - **Heterogeneous LTE (straggler-tail)**：具有慢尾分布的随机带宽，用于模拟真实 straggler 效应。
- 硬件实测：NVIDIA Jetson Orin Nano (8GB)，测量实际计算延迟与能耗。

#### 评估指标
| 类别 | 指标 |
|------|------|
| **通信效率** | 累计上行流量（cumulative uplink traffic） |
| **端到端性能** | 墙钟时间到准确率（time-to-accuracy） |
| **系统开销** | 每轮延迟分解（计算 vs 通信）、能量消耗（E<sub>comp</sub>, E<sub>comm</sub>） |
| **语义可靠性** | Token Recall（高 Fisher token 保留率）、ROUGE-L、METEOR、LLM-as-a-judge |
| **资源可行性** | 峰值内存占用、是否可在 2GB 边缘设备运行 |

### 基线方法对比
| 类型 | 基线方法 |
|------|--------|
| **无压缩参考** | FedAvg-LoRA |
| **参数级压缩** | QSGD (4-bit), FedPAQ, Top-k Sparsification |
| **启发式数据压缩** | Fed-ToMe（注意力驱动的 token 合并） |
| **其他控制** | FedBAT（可学习二值化）、FedProx/SCAFFOLD（漂移校正） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据
| 指标 | FED-FSTQ 表现 | 提升幅度 |
|------|-------------|----------|
| **累计上行流量减少** | 达到相同质量阈值时减少 **46×** | vs. Fed-LoRA |
| **端到端时间到准确率提升** | 加快 **52%** | vs. Fed-LoRA |
| **推理端到端加速** | 在 NVIDIA Jetson 上达 **1.55×** | 可选启用 Fisher 掩码 |
| **每轮总耗时** | 从 414.60s → **61.05s** | **6.8× 加速** |
| **每轮总能耗** | 从 634.40J → **98.50J** | 低于 100J，显著节能 |

### 与基线方法的对比结果
- **通信-准确率帕累托前沿**（图4）：FED-FSTQ 显著优于所有基线，在极低通信成本下达到目标准确率。
- **多语言鲁棒性**（表 I）：在中文（zh）上通信成本降低 **52%**（4.35 → 2.08），表明对信息密集语言更有效。
- **抗非 IID 性能**（图7）：在 Dirichlet α=0.1 的极端异构下仍保持稳定（准确率 0.5120），而 FedAvg 和 QSGD 几乎崩溃。
- **抗丢包与掉线**（图9–10）：
  - 在 20% 包丢失下，准确率仅从 0.66 → 0.579（下降 0.081），而 FedAvg 从 0.65 → 0.342（下降 0.308）。
  - 在 70% 客户端掉线时，准确率下降仅 0.10，远优于 FedAvg 的 0.40。

### 消融实验结果
| 变体 | ROUGE-L | Payload (MB) | 发现 |
|------|--------|--------------|------|
| **Full FED-FSTQ** | 0.6610 | 153.6 | 基准 |
| **w/o Fisher (Random)** | 0.4215 | 153.6 | **质量暴跌**，证明 Fisher 指导至关重要 |
| **w/o Token Pruning** | 0.6650 | 512.0 | 质量略升但通信开销大增 |
| **w/o Quantization** | 0.6720 | 614.4 | 进一步验证混合精度必要性 |

> ✅ 结论：**Fisher 指导 + token 剪枝 + 混合精度量化** 三者协同才能实现最优权衡。

---

## 4. 关键结论和发现

### 主要发现
1. **通信瓶颈的本质是语义失真而非单纯带宽不足**：盲目压缩会破坏稀有但关键的语义结构（如否定、分隔符），FED-FSTQ 通过 Fisher 信号识别并保护这些 token。
2. **Fisher 信息可作为有效的通信控制信号**：轻量级 token-level Fisher 代理即可实现高效的重要性排序，无需复杂 Hessian 计算。
3. **适度计算换大幅通信收益是值得的**：额外 0.85s 计算换来数十秒通信节省，在移动场景下极具性价比。
4. **训练与推理可共享敏感度信号**：Fisher 掩码可用于推理阶段 token 减少，实现“一次估计，双重受益”。

### 方法的局限性
- 当前依赖同步 FL 框架，未探索异步协议下的适用性。
- Fisher 估计假设局部梯度稳定性，极端 Non-IID 或剧烈分布偏移下可能失效。
- 未直接集成 secure aggregation 或 DP，元数据（如掩码、比特标签）的安全性需额外考虑。

### 未来工作方向
1. 扩展至 **异步或部分同步联邦学习** 协议，平衡 straggler 缓解与更新陈旧性。
2. 加强在 **安全聚合（secure aggregation）与差分隐私（DP）** 下的鲁棒性，适配更严格的部署约束。
3. 探索 **Fisher 信号的跨层传播机制**，提升低秩适配器（如 LoRA）中参数耦合的准确性。
4. 将该范式推广至 **多模态 LLMs** 的联邦微调场景。

---

> 📌 **总体评价**：FED-FSTQ 是一个面向实际部署的、系统友好的联邦 LLM 微调通信优化方案，它将传统的“压缩即降维”思维升级为“**语义感知的率失真优化**”，为在真实边缘网络中高效、可靠地训练 LLMs 提供了新路径。

</details>

---

### 3. [CUDA Kernel Optimization and Counter-Free Performance Analysis for Depthwise Convolution in Cloud Environments](https://arxiv.org/abs/2604.25422)

**Authors**: Huriyeh Babak, Melanie Schaller  
**Category**: cs.DC  
**Published**: 2026-04-29  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2604.25422v1  

#### Abstract
Efficient GPU execution of convolution operators is governed by memory-access efficiency, on-chip data reuse, and execution mapping rather than arithmetic throughput alone. This paper presents a controlled operator-level study of CUDA kernel optimization for the depthwise convolution used in Structu...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*CUDA Kernel Optimization and Counter-Free Performance Analysis for Depthwise Convolution in Cloud Environments*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在云环境（如 Kaggle、Google Colab、AWS）中，由于缺乏对硬件性能计数器（hardware performance counters）和高级分析工具（如 Nsight Compute）的访问权限，难以进行深入的 GPU 内核级性能分析。这限制了在受限环境下对 CUDA kernel 行为的可复现、架构级理解。

本文旨在回答一个核心问题：**在没有硬件性能监控支持的情况下，能否获得有意义的 GPU 架构级性能洞察？**

---

### 🚀 提出的新方法与创新思路

#### **Counter-Free Performance Analysis Methodology（无计数器性能分析方法）**
提出了一种**云兼容、无需硬件性能计数器**的性能分析框架，结合以下技术：
- **CUDA-event timing**：精确测量 kernel 运行时间。
- **Execution-path decomposition**：分别分析前向传播（forward）、输入梯度（input-gradient）和权重梯度（weight-gradient）路径。
- **Analytical memory-traffic modeling**：基于算子结构建模理论内存流量。
- **Effective bandwidth estimation**：通过运行时间和估计的数据移动量计算有效带宽。
- **Roofline analysis**：构建无计数器的 Roofline 模型以判断是否受内存或计算限制。

该方法实现了**类 profiling 的架构洞察**，而仅依赖于可移植的运行时测量和解析建模。

---

### 🔍 相比现有方法的优势

| 方面 | 传统方法局限 | 本论文优势 |
|------|--------------|------------|
| **环境依赖性** | 依赖 Nsight Compute 等特权工具，在云平台不可用 | 完全基于 CUDA Events 和数学建模，适用于任何标准 CUDA 环境 |
| **分析粒度** | 多为端到端模型加速研究，忽略单个 kernel 差异 | 控制变量法隔离 CUDA kernel 实现影响，实现 operator-level 分析 |
| **执行路径感知** | 忽略 backward 路径中的 reduction 瓶颈 | 明确区分 forward/input-gradient 与 weight-gradient 路径的行为差异 |
| **可复现性** | 受驱动版本、库实现等干扰 | 固定 operator、model、dataset、training config，确保结果可比 |

> ✅ **核心创新在于将多种经典分析手段统一整合成一套可在受限环境中使用的、系统性的、执行路径感知的分析流程。**

---

## 2. 核心实验方法和设置

### 📊 数据集
- **ASHRAE Great Energy Predictor III (GEPIII)**  
  包含建筑能耗与气象特征的时间序列数据。
- 选择原因：低维度输入（F=4）、固定序列长度（L=48），使得 depthwise convolution 成为主要计算瓶颈，适合做 kernel-level 分析。

---

### ⚙️ 实验设置

| 项目 | 配置 |
|------|------|
| **GPU 硬件** | NVIDIA Tesla P100-PCIE-16GB（Pascal 架构，compute capability 6.0） |
| **CUDA 平台** | 使用原生 CUDA C++ 编写 kernels，不依赖 cuDNN 或其他闭源库 |
| **Operator** | S4ConvD 中的 depthwise 1D convolution（来自 Structured State Space Models） |
| **Model** | S4ConvD block，H=128, K=48 |
| **Training Config** | SGD with momentum 0.9, LR=1e-5, batch size=16,384, RMSLE loss |
| **数据加载** | 多进程预取，避免 I/O stall 影响测量准确性 |

---

### 📈 评估指标

| 指标 | 描述 |
|------|------|
| **Kernel Runtime** | 使用 `cudaEvent_t` 测量各 execution path 的运行时间（ms） |
| **Epoch Time** | 单轮训练总耗时（秒），用于衡量端到端加速效果 |
| **Effective Bandwidth (GB/s)** | 估算值 = 总数据移动量 / kernel runtime |
| **Arithmetic Intensity (FLOPs/Byte)** | 浮点操作数 / 内存访问字节数 |
| **Roofline Plot** | 判断 kernel 是否 memory-bound 或 compute-bound |
| **Speedup** | 相对于 naive baseline 的加速比 |

---

### 🔁 基线方法对比（CUDA Kernel Variants）

共实现并比较四种 CUDA kernel 实现：

| Kernel 类型 | 关键优化策略 | 特点 |
|-------------|----------------|------|
| **Naive CUDA** | 朴素实现，每线程输出一个元素，无共享内存重用 | 基准参考，冗余访存严重 |
| **GMC (Global-Memory Coalesced)** | 对齐 warp 级内存访问，提升 coalescing 效率 | 减少事务开销，但仍有重复读取 |
| **Shared-Memory Cache Blocked** | 将输入和 kernel 权重缓存在 shared memory 中 | 支持片上数据重用，减少全局内存访问 |
| **Warp-Tiled** | 每个 warp 处理一个 (b, h) 实例，完全片上数据驻留 | 最大化数据局部性和 warp 级协作 |

> 所有变体保持数学等价，仅改变执行映射与内存利用方式。

---

## 3. 主要实验结果和性能指标

### 📉 关键性能数据（见 Table II）

| Method | FWD (ms) | BWD_in (ms) | BWD_k (ms) | Conv Total (ms) | Epoch (s) |
|--------|----------|-------------|------------|------------------|-----------|
| Naive CUDA | 29.97 | 30.25 | 73.26 | **133.47** | 44.82 |
| GMC | 28.23 | 28.78 | 49.64 | 106.65 | 40.31 |
| Shared | 16.36 | 16.03 | 34.17 | 66.57 | 36.91 |
| **Warp-tiled** | **10.46** | **10.61** | **19.91** | **40.99** | **34.74** |

---

### 📊 性能对比结果

| 指标 | 结果 |
|------|------|
| **Kernel-level Speedup (vs Naive)** | **3.26×**（从 133.47ms → 40.99ms） |
| **End-to-End Training Speedup (vs Naive)** | **1.29×**（从 44.82s → 34.74s） |
| **GMC 加速比** | 仅 1.25×，说明单纯 coalescing 效益有限 |
| **Shared vs Naive** | 2.00× 加速，体现 shared memory 价值 |
| **Warp-tiled vs Naive** | 达到最大加速，得益于 full on-chip data reuse |

---

### 🔍 消融实验与路径分解分析（见 Fig. 8 & V.B）

#### 各路径加速效果对比：

| Execution Path | Warp-tiled Speedup | 原因分析 |
|----------------|--------------------|---------|
| **Forward Pass** | ~2.86× | 得益于 warp-aligned + shared memory 重用 |
| **Input Gradient** | ~2.84× | 类似前向，内存局部性改善显著 |
| **Weight Gradient** | **3.68×**（相对最高）但绝对仍最慢 | reduction 结构导致同步与累加开销大，仍是瓶颈 |

> 💡 尽管 weight-gradient 相对加速最大，但由于其原始耗时长且 reduction 开销固有，仍是整体 runtime 的主导部分。

---

### 📈 Counter-Free Effective Bandwidth（见 Table III）

| Variant | Eff. BW (GB/s) | Peak Util. |
|--------|----------------|-----------|
| GMC | ~42 | ~6% |
| Shared | ~75 | ~10% |
| **Warp-tiled** | **~115** | **~16%** |

- 趋势表明：随着片上重用增强，有效带宽持续上升。
- 尽管远低于 P100 的峰值 732 GB/s，但对于低算术强度 workload 是合理表现。
- **关键结论：性能提升主要来自减少冗余数据移动，而非提高峰值利用率。**

---

### 📊 Roofline Analysis（见 Fig. 10）

- 所有 kernel 均位于 **memory-bound 区域**，远离 compute roof。
- Naive kernel 算术强度最低（冗余访存多）。
- Shared 和 Warp-tiled kernel 向右上方移动 → 更高 arithmetic intensity 和更高 bandwidth utilization。
- Weight-gradient kernel 始终处于较低位置 → reduction overhead 抑制效率。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **内存访问效率 > 算术吞吐量**
   - Depthwise convolution 是典型的 memory-bound operator。
   - 性能由 **data movement** 和 **on-chip data reuse** 主导，而非 FLOPs。

2. **Coalescing 效益有限**
   - GMC 仅带来 1.25× 加速，因其未消除重叠窗口的重复读取。

3. **On-chip Data Reuse 是关键**
   - Shared memory 和 warp-tiled 设计通过显式数据重用大幅降低 global memory traffic，是性能跃升的根本原因。

4. **Execution Path Matters**
   - Forward 和 input-gradient 可被高效优化；
   - Weight-gradient 因 reduction 结构成为结构性瓶颈，即使优化后仍是 runtime 主体。

5. **Kernel Speedup ≠ End-to-End Speedup**
   - kernel 加速 3.26×，但端到端仅提速 1.29×。
   - 原因：当 kernel 变快后，**framework overhead、optimizer update、synchronization** 等非 kernel 成分占比上升。

6. **Occupancy 不等于高性能**
   - 高占用率 kernel 若存在冗余访存，仍可能低效；
   - 更重要的是 memory efficiency 和 locality。

7. **Counter-Free Analysis 是可行的**
   - 仅用 CUDA Events + analytical modeling 即可重建类似 profiling 的洞察。
   - 可识别出冗余访存、reduction 瓶颈、memory-bound 状态等关键问题。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **无法精确量化 cache miss rate** | 缺乏 PMCs 导致无法直接观测 L1/L2 miss，只能间接推断 |
| **Effective BW 为估算值** | 并非真实 DRAM throughput，而是反映相对内存效率 |
| **特定于 depthwise convolution 结构** | 虽然结论具有一般性，但具体优化策略需适配不同算子 |
| **未修改算法结构** | 如 hierarchical reduction、kernel fusion 等更深层优化未引入，以保证公平比较 |

---

### 🔮 未来工作方向

1. **Algorithmic Restructuring for Reductions**
   - 探索更高效的 reduction 策略（如 tree-based、warp-shuffle + atomic fusion）来打破 weight-gradient 瓶颈。

2. **Extension to Other Operators**
   - 将 counter-free 分析框架推广至 attention、transformer blocks、sparse conv 等复杂 operator。

3. **Automated Analysis Pipelines**
   - 构建自动化工具链，集成 event timing、path decomposition、roofline 绘制等功能，便于社区使用。

4. **Multi-Kernel Pipeline Analysis**
   - 扩展至整个 training pipeline，分析 kernel 间依赖与调度影响。

5. **跨平台可复现性研究**
   - 在不同云平台（AWS/GCP/Azure）验证该方法的一致性，推动标准化性能评测流程。

---

## ✅ 总结一句话

> 本文证明了：**即便在无法使用硬件性能计数器的云环境中，也能通过精心设计的 counter-free 方法，获得深度、可复现的 GPU kernel 架构级性能洞察** —— 关键在于控制变量、路径分解、结合运行时测量与解析建模，并强调 **on-chip data reuse** 比单纯的 memory coalescing 更能决定 memory-bound operator 的性能上限。

</details>

---

### 4. [Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](https://arxiv.org/abs/2604.23623)

**Authors**: Zichuan Fu, Xian Wu, Guojing Li, Yejing Wang, Yijun Chen, Zihao Zhao, Yixuan Luo, Hanyu Yan, Yefeng Zheng, Xiangyu Zhao  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.23623v1  

#### Abstract
Recent advancements in large language models (LLMs) have catalyzed the rise of reasoning-intensive inference paradigms, where models perform explicit step-by-step reasoning before generating final answers. While such approaches improve answer quality and interpretability, they incur substantial comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning**

---

## **1. 论文的主要贡献和创新点**

### **解决的问题**
当前大型语言模型（LLMs）在执行复杂推理任务时，通常采用“思维链”（Chain-of-Thought, CoT）或更长的“思考范式”（thinking paradigm），生成数千token的推理过程。这种做法虽然提升了答案质量和可解释性，但也带来了巨大的**计算开销**，表现为：
- 推理延迟高
- 运行成本昂贵
- 难以部署于实时或预算受限场景

此外，已有优化方法如强化微调（RFT）存在以下限制：
- 需要持续训练LLM，可能损害其通用能力
- 不适用于仅提供API访问的闭源模型

因此，核心问题是：**如何在保留LLM高质量推理优势的同时，显著降低计算成本？**

---

### **提出的新方法与新思路**
作者提出了 **Tandem** 框架，一种新颖的 **LLM-SLM 协作推理范式**，灵感来源于“导师-实习生”（mentor-intern）关系。

#### **核心思想**
- **LLM作为战略导师（Mentor）**：负责生成轻量级、结构化的关键**思维洞察（Thinking Insights）**，而非完整推理链。
- **SLM作为执行实习生（Intern）**：利用这些洞察完成详细推理并输出最终答案。
- **动态终止机制**：引入一个基于不确定性的**cost-aware 终止判断器**，决定何时LLM的指导已足够，从而实现早期停止。

#### **四大思维洞察（Thinking Insights）**
Tandem将LLM的推理分解为四个模块化组件，对应人类认知阶段（ACT-R架构）：
1. **Goal**：明确目标与约束
2. **Planning**：制定高层策略与子问题拆解
3. **Retrieval**：召回相关知识、公式或定义
4. **Action**：执行具体计算或逻辑步骤

通过传递这四类结构化信息，而非冗长文本，大幅减少传输负担。

---

### **相比现有方法的优势**
| 方面 | Tandem | 传统方法（如RFT、Budget Forcing） |
|------|--------|-------------------------------|
| **无需训练LLM** | ✅ 支持API调用的黑盒LLM | ❌ 通常需微调或修改LLM |
| **灵活性强** | ✅ 自适应控制LLM生成长度 | ❌ 固定预算或硬截断 |
| **效率更高** | ✅ 减少约40%计算成本 | ⚠️ 成本节省有限或牺牲精度 |
| **跨域迁移性好** | ✅ Sufficiency Classifier可在不同领域复用 | ❌ 多数方法依赖特定任务训练 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **MATH** (Hendrycks et al., 2021)  
  - 包含12.5K竞赛级别数学题，覆盖7个学科（代数、几何等）
  - 分为5个难度等级，测试集5K样本
- **GSM8K** (Cobbe et al., 2021)  
  - 小学数学应用题，共8.5K样本，强调多步推理
- **HumanEval** (Chen et al., 2021)  
  - 用于代码生成任务的基准，验证跨域泛化能力

---

### **实验设置与评估指标**

#### **模型配置**
- **LLM候选**：DeepSeek-R1-Distill-Qwen-32B（简称DeepSeek-32B）、Qwen3-32B、GPT-4o-mini、gpt-oss-120b
- **SLM候选**：DeepSeek-7B、Qwen3-8B
- 所有LLM运行在**确定性模式**下（temperature=0）

#### **Tandem三阶段设计**
| 阶段 | 努力程度 | 新增洞察 | 累积洞察 |
|------|----------|-----------|------------|
| Stage 1 | Low | Goal | Goal |
| Stage 2 | Medium | Planning, Retrieval | Goal + Planning + Retrieval |
| Stage 3 | High | Action | 全部四类 |

每个阶段后由SLM判断是否已“充分”，若充分则提前终止LLM生成。

#### **评估指标**
1. **Accuracy**：标准准确率（正确解答比例）
2. **Inference Length**：总生成token数（LLM + SLM）
3. **Computational Cost (TFLOPs)**：
   $$
   \text{Cost} = \frac{|\theta_L|}{10^{12}} \cdot L_L + \frac{|\theta_S|}{10^{12}} \cdot (L_L + L_S)
   $$
   其中 $\theta$ 为参数量，$L$ 为生成长度

#### **基线方法对比**
- **Single Model**：仅用7B或32B独立推理
- **Fixed-length Collaboration**：固定LLM生成长度（100/500/1000 tokens）
- **Budget Forcing**：强制截断32B模型的推理长度
- **LLM Cascade**：二分类路由决策，选择走SLM还是Full LLM路径

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **在MATH上的总体表现（Table 1）**
| 方法 | 平均Accuracy | 相对32B提升 | 计算成本 | 相对32B节省 |
|------|--------------|-------------|-----------|-------------|
| LLM (32B) | 80.90% | — | 168.35 TFLOPs | — |
| **Tandem (7B+32B)** | **83.46%** | **+2.56pp** | **99.72 TFLOPs** | **~40.8%↓** |

> 🔥 **Tandem不仅超越了单独LLM的表现，还节省了近41%的计算成本**

#### **在HumanEval上的跨域表现（Table 5）**
| 方法 | HumanEval Acc. |
|------|----------------|
| SLM (7B) | 65.24% |
| LLM (32B) | 89.02% |
| Tandem | **85.37%** |

> ✅ 即使**未重新训练分类器**（直接使用MATH上训练的sufficiency classifier），Tandem仍优于多数固定预算协作，并接近LLM上限。

---

### **与效率导向基线的对比（Table 6）**

| 方法 | MATH Acc. | Cost (TFLOPs) |
|------|-----------|---------------|
| LLM (32B) | 80.90% | 168.35 |
| Budget Forcing | 82.18% | 108.74 |
| LLM Cascade | 82.60% | 95.33 |
| **Tandem** | **83.46%** | **99.72** |

> 📈 **Tandem实现了最高准确率，且成本低于Budget Forcing，略高于Cascade但仍具竞争力**

---

### **消融实验与关键分析**

#### **(1) 跨家族协作有效性（Table 2）**
- DeepSeek-7B + Qwen3-32B 在MATH上达到 **79.96%**，超过任一单体模型
- 表明**结构化思维洞察具有良好的跨模型家族可理解性**

#### **(2) 模型大小组合的影响（Table 3）**
- 最佳协作发生在**能力差距适中**时（如7B+32B）
- 若SLM太小（如1.5B），难以解析高级推理 → 改进有限
- 若差距过小（如14B+32B），互补性弱 → 增益微弱

#### **(3) API可用LLM协作（Table 4）**
- 使用DeepSeek-7B本地运行 + API调用GPT-4o-mini/gpt-oss-120b
- 结果显示：**准确率提升同时成本下降**
- 证明Tandem完全兼容**闭源、API-only模型**

#### **(4) 非思考模式下的有效性（Table 7）**
- 即使LLM不启用“thinking mode”，Tandem依然有效
- 在non-thinking模式下，Tandem比32B单独运行**节省36.7%成本**，**准确率+2.58pp**
- 说明该框架**不限于特定推理范式**

#### **(5) Sufficiency Classifier分析（Table 9 & 14）**
- 分类器平均F1达 **0.832**，Precision > 0.82
- 错误案例中，93.1%是由于**提前停止（Premature Stop）或延迟停止（Late Stop）**
- Oracle上限可达 **98.86%**，表明进一步优化分类器仍有巨大空间

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **轻量级结构化指导 > 完整推理链**  
   LLM只需提供精炼的四类思维洞察（Goal/Planning/Retrieval/Action），即可有效引导SLM完成高质量推理。

2. ✅ **动态资源分配优于固定预算**  
   Tandem的cost-aware终止机制能根据问题难度自适应调整LLM参与深度，避免简单问题过度消耗资源。

3. ✅ **高效且可扩展的协作模式**  
   - 支持跨模型家族（DeepSeek ↔ Qwen）
   - 支持API接入模型（GPT系列）
   - 支持非thinking模式LLM

4. ✅ **sufficiency classifier具备强泛化能力**  
   在MATH上训练的分类器可直接迁移到HumanEval（代码生成），无需重训，准确率达85.37%

5. ✅ **实际部署友好**  
   - 推理延迟降低至 **1.8× speedup**（Table 13）
   - 判断机制开销极低（<2% 总时间）

---

### **局限性**
1. **领域泛化尚未全面验证**  
   当前实验集中于数学与编程，是否适用于常识推理、开放问答等任务仍待探索。

2. **仍需一定标注数据训练classifier**  
   尽管支持跨域迁移，但至少需要在一个领域有带标签数据进行初始训练。

3. **静态双模型架构**  
   当前仅为“一对一”协作，未考虑多模型协同、角色动态切换等更复杂的协作模式。

---

### **未来工作方向**
- 探索**无监督或弱监督方式训练sufficiency classifier**
- 构建**多层级、多角色的协作网络**（如多个SLM分工）
- 引入**反馈机制**让SLM主动请求特定类型的洞察
- 将Tandem思想应用于**多模态推理系统**

---

> 💡 **一句话总结**：  
> **Tandem通过“导师-实习生”式的LLM-SLM协作，用结构化思维洞察替代冗长推理链，结合动态终止机制，在数学与代码任务上实现了比纯LLM更高准确率、更低40%成本的高效推理，且支持API模型与跨域迁移，为大规模语言模型的实际部署提供了新范式。**

</details>

---

### 5. [Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence](https://arxiv.org/abs/2604.24954)

**Authors**: NVIDIA,  :, Amala Sanjay Deshmukh, Kateryna Chumachenko, Tuomas Rintamaki, Matthieu Le, Tyler Poon, Danial Mohseni Taheri, Ilia Karmanov, Guilin Liu, Jarno Seppanen, Arushi Goel, Mike Ranzinger, Greg Heinrich, Guo Chen, Lukas Voegtle, Philipp Fischer, Timo Roman, Karan Sapra, Collin McCarthy, Shaokun Zhang, Fuxiao Liu, Hanrong Ye, Yi Dong, Mingjie Liu, Yifan Peng, Piotr Zelasko, Zhehuai Chen, Nithin Rao Koluguri, Nune Tadevosyan, Lilit Grigoryan, Ehsan Hosseini Asl, Pritam Biswas, Leili Tavabi, Yuanhang Su, Zhiding Yu, Peter Jin, Alexandre Milesi, Netanel Haber, Yao Xu, Sarah Amiraslani, Nabin Mulepati, Eric Tramel, Jaehun Jung, Ximing Lu, Brandon Cui, Jin Xu, Zhiqi Li, Shihao Wang, Yuanguo Kuang, Shaokun Zhang, Huck Yang, Boyi Li, Hongxu Yin, Song Han, Pavlo Molchanov, Adi Renduchintala, Charles Wang, David Mosallanezhad, Soumye Singhal, Luis Vega, Katherine Cheung, Sreyan Ghosh, Yian Zhang, Alexander Bukharin, Venkat Srinivasan, Johnny Greco, Andre Manoel, Maarten Van Segbroeck, Suseella Panguliri, Rohit Watve, Divyanshu Kakwani, Shubham Pachori, Jeffrey Glick, Radha Sri-Tharan, Aileen Zaman, Khanh Nguyen, Shi Chen, Jiaheng Fang, Qing Miao, Wenfei Zhou, Yu Wang, Zaid Pervaiz Bhat, Varun Praveen, Arihant Jain, Ramanathan Arunachalam, Tomasz Kornuta, Ashton Sharabiani, Amy Shen, Wei Huang, Yi-Fu Wu, Ali Roshan Ghias, Huiying Li, Brian Yu, Nima Tajbakhsh, Chen Cui, Wenwen Gao, Li Ding, Terry Kong, Manoj Kilaru, Anahita Bhiwandiwalla, Marek Wawrzos, Daniel Korzekwa, Pablo Ribalta, Grzegorz Chlebus, Besmira Nushi, Ewa Dobrowolska, Maciej Jakub Mikulski, Kunal Dhawan, Steve Huang, Jagadeesh Balam, Yongqiang Wang, Nikolay Karpov, Valentin Mendelev, George Zelenfroynd, Meline Mkrtchyan, Qing Miao, Omri Almog, Bhavesh Pawar, Rameshwar Shivbhakta, Sudeep Sabnis, Ashrton Sharabiani, Negar Habibi, Geethapriya Venkataramani, Pamela Peng, Prerit Rodney, Serge Panev, Richard Mazzarese, Nicky Liu, Michael Fukuyama, Andrii Skliar, Roger Waleffe, Duncan Riach, Yunheng Zou, Jian Hu, Hao Zhang, Binfeng Xu, Yuhao Yang, Zuhair Ahmed, Alexandre Milesi, Carlo del Mundo, Chad Voegele, Zhiyu Cheng, Nave Assaf, Andrii Skliar, Daniel Afrimi, Natan Bagrov, Ran Zilberstein, Ofri Masad, Eugene Khvedchenia, Natan Bagrov, Borys Tymchenko, Tomer Asida, Daniel Afrimi, Parth Mannan, Victor Cui, Michael Evans, Katherine Luna, Jie Lou, Pinky Xu, Guyue Huang, Negar Habibi, Michael Boone, Pradeep Thalasta, Adeola Adesoba, Dina Yared, Christopher Parisien, Leon Derczynski, Shaona Ghosh, Wes Feely, Micah Schaffer, Radha Sri-Tharan, Jeffrey Glick, Barnaby Simkin, George Zelenfroynd, Tomasz Grzegorzek, Rishabh Garg, Aastha Jhunjhunwala, Sergei Kolchenko, Farzan Memarian, Haran Kumar, Shiv Kumar, Isabel Hulseman, Anjali Shah, Kari Briski, Padmavathy Subramanian, Joey Conway, Udi Karpas, Jane Polak Scowcroft, Annie Surla, Shilpa Ammireddy, Ellie Evans, Jesse Oliver, Tom Balough, Chia-Chih Chen, Sandip Bhaskar, Alejandra Rico, Bardiya Sadeghi, Seph Mard, Katherine Cheung, Meredith Price, Laya Sleiman, Saori Kaji, Wesley Helmholz, Wendy Quan, Michael Lightstone, Jonathan Cohen, Jian Zhang, Oleksii Kuchaiev, Boris Ginsburg, Jan Kautz, Eileen Long, Mohammad Shoeybi, Mostofa Patwary, Oluwatobi Olabiyi, Andrew Tao, Bryan Catanzaro, Udi Karpas  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2604.24954v1  

#### Abstract
We introduce Nemotron 3 Nano Omni, the latest model in the Nemotron multimodal series and the first to natively support audio inputs alongside text, images, and video. Nemotron 3 Nano Omni delivers consistent accuracy improvements over its predecessor, Nemotron Nano V2 VL, across all modalities, ena...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
Nemotron 3 Nano Omni 旨在构建一个**高效、开源且真正支持多模态输入（text, image, video, audio）的统一模型**，解决以下挑战：
- 现有视觉语言模型（VLMs）普遍缺乏对音频输入的原生支持；
- 多模态长序列处理效率低，推理延迟高；
- 跨模态对齐困难，尤其在引入新模态时易导致文本能力退化（catastrophic forgetting）；
- 长上下文理解能力不足，难以处理超长文档或长时间音视频。

### **提出的新方法与思路**
1. **Omni-Modal 架构设计**  
   - 首次在 Nemotron 系列中实现**原生音频支持**，集成 Parakeet-TDT-0.6B-v2 音频编码器。
   - 采用 **encoder-projector-decoder** 结构，结合 C-RADIOv4-H 视觉编码器、Parakeet 音频编码器与 Nemotron 3 Nano 30B-A3B MoE LLM。

2. **关键技术改进**
   - **Dynamic Image Resolution**：取代传统的图像分块策略，动态调整分辨率以保留原始宽高比，提升 OCR 和图表理解精度。
   - **Temporal Video Compression via Conv3D**：使用 3D 卷积压缩连续帧，将每两帧融合为一个“tubelet”，实现时间维度上 **2× token reduction**。
   - **Extended Context Length**：最大上下文长度从 128K 提升至 **256K tokens**，显著增强长文档和长视频的理解能力。
   - **Multimodal Token Reduction Techniques**：结合 Conv3D 与时序采样（EVS），大幅降低输入 token 数量，提高吞吐量。

3. **训练策略创新**
   - 采用**多阶段渐进式训练流程（multi-stage SFT + RL）**：
     - 先分别对视觉、音频 projector 进行 warmup；
     - 再逐步联合训练所有模态，并扩展上下文长度；
     - 最后通过强化学习（RL）优化推理与安全性。
   - 此策略有效缓解了跨模态对齐不稳定和灾难性遗忘问题。

### **相比现有方法的优势**
| 维度 | Nemotron 3 Nano Omni | 对比优势 |
|------|------------------------|----------|
| **模态支持** | 支持 text, image, video, audio 四种模态 | 是首个原生支持 audio 的 Nemotron 模型 |
| **架构效率** | MoE + Mamba-Transformer Hybrid | 更适合长序列建模，推理更高效 |
| **上下文长度** | 最大 256K tokens | 超越多数同类模型（通常 ≤128K） |
| **推理性能** | 支持 EVS + Conv3D + 量化（FP8/NVFP4） | 显著降低 TTFT 和提升 throughput |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
#### **SFT 阶段数据构成（总计约 434.1M 样本，466.9B tokens）**
| 阶段 | 主要数据来源 | 数据规模 | 任务类型 |
|------|-------------|---------|----------|
| Stage 0 | Vision projector warmup | 9.35M samples | 图像-文本对齐 |
| Stage 1 | Vision SFT | 86.3M samples | Captioning, OCR, VQA, GUI, 文档理解等 |
| Stage 2 | Audio projector warmup | 59.2M samples | ASR（Granary v1.1） |
| Stage 3 | Audio encoder + projector | 242.0M samples | ASR, Sound, Music, Speech 理解 |
| Stage 4 | Omni SFT @16K | 30.5M samples | 多模态 QA、captioning、安全数据 |
| Stage 5 | Omni SFT @48K | 6.08M samples | 中长视频、多步推理 |
| Stage 6 | Omni SFT @256K | 623K samples | 超长文档理解（学术论文、财报等） |

> 注：大量使用合成数据管道生成高质量 QA 对，利用 Qwen3-Omni、GPT-OSS 等模型进行蒸馏与过滤。

#### **评估基准**
| 类别 | 基准名称 | 描述 |
|------|--------|------|
| **视觉理解** | MMMU, MathVista-Mini, OCRBench-V2, MMLongBench-Doc, ChartQA, DocVQA, InfoVQA, AI2D, TextVQA |
| **GUI 与代理任务** | ScreenSpot(v2/Pro), OSWorld |
| **视频理解** | Video-MME, LongVideoBench |
| **音频理解** | OpenASR, TED-LIUM Longform, MMAU, VoiceBench |
| **音视频联合理解** | DailyOmni, WorldSense |
| **纯文本能力** | MMLU-Pro, GPQA, AIME-2025, LiveCodeBench, IFBench, TauBench, SciCode |

### **实验设置与评估指标**
- **推理框架**：vLLM backend（用于视觉/音频）、NeMo-Skills（用于文本）
- **量化格式测试**：BF16、FP8、NVFP4
- **硬件平台**：NVIDIA B200 GPU
- **关键指标**：
  - 准确率（Accuracy）
  - Word Error Rate（WER，越低越好）
  - Throughput（output tokens/sec）
  - Time-to-First-Token（TTFT）
  - Interactivity Target（如 150 tok/s/user）

### **基线方法对比**
- **Nemotron Nano V2 VL**（前代模型）
- **Qwen3-Omni**（通义千问系列 omni 模型）
- **Qwen3.5-Omni**（闭源模型，仅作参考）

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **视觉理解表现（Table 7）**
| 基准 | Nemotron 3 Nano Omni | Qwen3-Omni | 提升情况 |
|------|------------------------|------------|-----------|
| **MMMU (val)** | 70.8 | 75.6 | ↓（略差） |
| **MathVista-Mini** | 82.8 | 80.0 | ↑ +2.8 pts |
| **MMLongBench-Doc** | 57.5 | 49.5 | ↑ +8.0 pts |
| **OCRBench-V2 (EN/ZH)** | 67.0/52.7 | — | SOTA |
| **ChartQA (Test)** | 90.3 | 89.5 | ↑ |
| **DocVQA (Test)** | 95.6 | 95.3 | 持平 |
| **Video-MME (w/o sub)** | 72.2 | 70.5 | ↑ +1.7 pts |

> 在文档理解类任务上取得显著领先，尤其在 OCR-Reasoning 上从 33.9 提升到 **54.14**。

#### ✅ **GUI 与 Agent 任务（Table 7）**
| 基准 | Nemotron 3 Nano Omni | Qwen3-Omni | 提升情况 |
|------|------------------------|------------|-----------|
| **ScreenSpot-Pro** | 57.8 | 59.7 | 略低 |
| **OSWorld** | 47.4 | 29.0 | ↑ +18.4 pts |

> 在真实计算机操作任务 **OSWorld** 上远超 Qwen3-Omni，显示其强大的 agentic 能力。

#### ✅ **音频理解（Table 8）**
| 基准 | Nemotron 3 Nano Omni | Qwen3-Omni | 提升情况 |
|------|------------------------|------------|-----------|
| **OpenASR Avg WER↓** | 5.95 | 6.55 | ↓ -0.6 pts |
| **TED-LIUM Longform WER↓** | 3.11 | 2.4 | ↓（稍差） |
| **MMAU Avg↑** | 74.6 | 77.5 | ↓（稍弱） |
| **VoiceBench Avg↑** | 89.4 | 88.8 | ↑ +0.6 pts |

> 在语音助手交互任务 **VoiceBench** 上达到最先进水平，优于 Qwen3-Omni。

#### ✅ **音视频联合理解（Table 9）**
| 基准 | Nemotron 3 Nano Omni | Qwen3-Omni | 提升情况 |
|------|------------------------|------------|-----------|
| **DailyOmni (acc)** | 74.5 | 71.9 | ↑ +2.6 pts |
| **WorldSense (acc)** | 55.4 | — | 领先 |

> 在跨模态时序对齐与因果推理任务上表现优异。

#### ✅ **纯文本能力保持（Table 10）**
| 基准 | Nemotron 3 Nano Omni | Nemotron 3 Nano 30B-A3B | 差距 |
|------|------------------------|----------------------------|-------|
| **MMLU-Pro** | 77.3 | 78.3 | -1.0 pt |
| **GPQA (no tools)** | 72.2 | 73.0 | -0.8 pt |
| **LiveCodeBench** | 63.2 | 68.3 | -5.1 pt |

> 尽管增加了多模态能力，文本基础能力几乎完全保留，体现了良好的模态平衡。

---

### **消融实验结果**

#### 🔹 **推理预算控制（Reasoning Budget Control, Table 11）**
| 基准 | 无 reasoning budget | 含 reasoning budget | 提升 |
|------|--------------------|---------------------|------|
| MathVista-Mini | 80.3 | **82.8** | ↑ +2.5 |
| MMLongBench-Doc | 54.5 | **56.8** | ↑ +2.3 |
| CharXiv(RQ) | 61.8 | **64.0** | ↑ +2.2 |

> 合理设置推理预算可避免冗余思维链，提升准确率。

#### 🔹 **Conv3D + EVS 效率优化（Table 12 & 13）**
- **Token Reduction**：
  - 原始 512 帧视频 → ~141K tokens
  - 加 Conv3D → ~75K tokens（↓47%）
  - 加 Conv3D + EVS(q=0.5) → ~42K tokens（↓70%）
- **TTFT 降低**：
  - BF16 baseline: 7969 ms
  - Conv3D + EVS: **5313 ms（↓33%）**
- **精度损失极小**：平均仅下降约 0.5 pts

#### 🔹 **量化影响（Table 14）**
| 精度格式 | Size | bpw | 平均精度下降 |
|--------|------|-----|-------------|
| BF16 | 61.5 GB | 16.0 | 0% |
| FP8 | 32.8 GB | 8.5 | -0.37 pts |
| NVFP4 | 20.9 GB | 4.98 | -0.40 pts |

> 量化后模型体积缩小至 1/3（FP8）甚至 1/3（NVFP4），精度损失 <1%，性价比极高。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **首次实现原生音频支持的高效 omni-modal 模型**，在 VoiceBench、DailyOmni 等任务上达到 SOTA。
2. ✅ **通过 Conv3D + EVS 实现高效的 token 压缩机制**，在不牺牲太多精度的前提下，显著降低推理延迟与计算成本。
3. ✅ **多阶段训练策略成功维持了强大的文本能力**，同时大幅提升视觉与音频理解性能。
4. ✅ **在文档理解、GUI 操作、长视频理解等实际应用场景中全面超越 Qwen3-Omni**，特别是在 OSWorld 上表现突出。
5. ✅ **支持多种量化格式（FP8/NVFP4）并公开训练数据与代码**，推动社区发展。

### **方法的局限性**
- **音频理解仍有提升空间**：在 TED-LIUM Longform 和 MMAU 上略逊于最优模型。
- **依赖大规模合成数据**：虽然提升了数据多样性，但也可能引入噪声或偏差。
- **未开放全部训练数据**：仅发布部分数据集（如 Nemotron-Image-Training-v3），完整训练集不可用。
- **硬件依赖较强**：最佳性能需在 B200 等高端 GPU 上运行。

### **未来工作方向**
- 扩展更多模态（如传感器、触觉等）；
- 探索更细粒度的跨模态对齐机制；
- 优化音频编码器以进一步提升 ASR 与声音理解能力；
- 发布更大规模版本（如 Nemotron 3 Omni 70B）；
- 推动轻量化部署方案，适配边缘设备。

---

> 📢 **总结一句话**：  
> **Nemotron 3 Nano Omni 是一个兼具高性能、高效率与开放性的 omni-modal 模型，在文档理解、GUI agent、长音视频推理等任务上全面领先，是当前最实用的开源多模态智能体之一。**

</details>

---

### 6. [Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis](https://arxiv.org/abs/2604.23072)

**Authors**: Junyan Cheng, Kyle Richardson, Peter Chin  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.23072v1  

#### Abstract
Large language model (LLM) agents are increasingly tasked with complex real-world analysis (e.g., in financial forecasting, scientific discovery), yet their reasoning suffers from stochastic instability and lacks a verifiable, compositional structure. To address this, we introduce Analytica, a novel...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Model (LLM)** 的智能体在进行复杂现实世界分析（如金融预测、科学发现）时面临两大挑战：
- **推理过程不稳定**：LLM 的生成具有随机性（stochastic instability），导致相同任务多次运行结果不一致。
- **缺乏可验证的结构化推理**：传统 **Chain-of-Thought (CoT)** 等方法依赖自由文本推理，难以分解、验证和量化不确定性。

### 提出的新方法：Analytica 与 Soft Propositional Reasoning (SPR)
论文提出 **Analytica**，一种基于 **Soft Propositional Reasoning (SPR)** 的新型 LLM 智能体架构。其核心思想是将复杂分析重构为对“软真值”（soft truth value）的估计过程。

#### 核心创新点：
- **Soft Propositional Reasoning (SPR)**  
  将复杂命题分解为一系列子命题（sub-propositions），每个子命题被赋予一个介于 [0,1] 的“软真值”（如 P=0.7 表示 70% 可信度）。最终答案通过组合这些软真值得到，而非直接生成文本。
  
- **三阶段 Divide-and-Conquer 架构**：
  1. **Analysis Stage**：由 **Analyzer** 将根命题递归分解为树状子命题。
  2. **Grounding Stage**：由 **Grounder** 并行验证叶节点（leaf nodes），使用工具（如 Web Search、Jupyter Notebook）获取证据并打分。
  3. **Synthesis Stage**：由 **Synthesizer** 自底向上聚合软真值，使用鲁棒的合成规则（如线性模型）计算根命题的最终得分。

- **误差建模与优化**  
  借鉴统计学中的 **Mean Squared Error (MSE)** 分解，将预测误差分为 **Bias（偏差）** 和 **Variance（方差）**，并分别优化：
  - **降低 Bias**：通过深度分解，使叶节点更简单，易于验证。
  - **降低 Variance**：通过线性合成规则平均多个子路径的噪声，提升稳定性。

### 相比现有方法的优势
| 方面 | 传统方法（如 CoT, ToT） | Analytica |
|------|------------------------|----------|
| 推理结构 | 线性或图状文本流 | 结构化命题树 |
| 不确定性处理 | 隐式、不可量化 | 显式软真值 |
| 稳定性 | 低（高方差） | 高（低方差） |
| 可解释性 | 弱 | 强（每一步可追溯） |
| 成本效率 | 通常较高 | 支持高效并行与复用 |

---

## 2. 核心实验方法和设置

### 数据集
在 **736 个真实世界的经济与金融预测任务** 上进行评估，涵盖三大类：
- **Financial Market Tasks**：对股票、指数、商品等资产进行“长期持有 vs. 做空”的一年期预测。
- **Predictive Market Tasks**：来自 Kalshi 和 Polymarket 的预测市场选项（如“谁将赢得2024年美国总统大选？”）。
- **事件时间跨度**：从数周到超过一年，确保测试的是真实未来预测能力。

### 实验设置
- **基础模型**：统一使用 `o3-2025-04-16` 模型，知识截止日期为 2024 年 6 月 1 日。
- **温度参数**：设为 0.1，以减少生成随机性。
- **最大叶节点数**：10（控制分析深度）。
- **并行执行**：支持最多 20 个叶节点并行验证。

### 评估指标
| 指标 | 含义 |
|------|------|
| **Accuracy (Accu.)** | 正确选择最高效用选项的比例（Top-1 准确率） |
| **Soft Score** | 所有选项的软真值加权平均回报（反映置信度校准） |
| **Hard Score** | 最高软真值选项的实际回报 |
| **Brier Score (BS)** | 预测概率分布的均方误差，越低越好 |
| **Variance (Var.)** | 多次运行下 Hard Score 的方差，衡量稳定性 |
| **Cost & Time** | API 调用成本与响应时间，衡量效率 |

### 基线方法对比
- **基础基线**：
  - `Basic Search`：仅使用网络搜索的 LLM。
  - `Deep Research`：OpenAI 的深度研究智能体。
  - `Jupyter Notebook`：模拟分析师使用 Jupyter 进行代码驱动分析。
- **结构化推理基线**：
  - `Tree-of-Thoughts (ToT)`
  - `Graph-of-Thoughts (GoT)`
  - `Forest-of-Thoughts (FoT)`
- **Analytica 变体**：
  - `Analytica-V`：使用 LLM 直接合成（Vanilla）
  - `Analytica-S`：使用模糊逻辑规则合成（Simple Logic）
  - `Analytica-L`：使用线性模型合成（Linear）——主推方法

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

| 方法 | Accuracy (%) | Improvement (%) | Variance (%) | Cost ($) | Time (min) |
|------|-------------|---------------|-------------|---------|-----------|
| `Basic Search` | 53.94 | — | 10.30 | 0.02 | 0.54 |
| `Deep Research` | 63.04 | — | 9.28 | 4.02 | 7.60 |
| `Jupyter NB` | 61.96 | — | 12.28 | 0.07 | 2.61 |
| **`Analytica-L` (Best)** | **71.06** | **+12.72** | **6.02** | 14.10 | 30.01 |
| **`Analytica-L` (Jupyter)** | **70.11** | **+13.15** | **7.28** | **1.36** | **14.15** |

> ✅ **平均提升 15.84% 准确率**，最高达 **71.06%**，且方差最低（6.02%）。

### 与基线方法的对比结果
- **相比 Deep Research**：
  - `Analytica-L` 在其基础上再提升 **12.72%** 准确率。
  - 方差从 9.28% 降至 6.02%，显著更稳定。
- **相比 ToT/FoT**：
  - `Analytica-L` 比 `Forest-of-Thoughts` 高出 **10.89%**。
- **Jupyter Grounder 的性价比**：
  - 使用 `Jupyter Notebook` 作为 Grounder 的 `Analytica-L` 达到 **70.11%** 准确率。
  - 相比 `Deep Research`，**节省 90.35% 成本** 和 **52.85% 时间**。

### 消融实验结果
#### （1）不同合成规则对比（Table 2）
| 合成规则 | Accuracy (%) | Improvement (%) | Variance (%) |
|----------|------------|----------------|-------------|
| Vanilla (V) | 63.18 | +17.73 | 10.89 |
| Simple Logic (S) | 57.61 | +6.80 | 7.45 |
| **Linear (L)** | **65.62** | **+21.65** | **6.46** |

> ✅ **Linear 规则在准确率和稳定性上全面领先**，验证了其抗噪优势。

#### （2）不同 Grounder 的影响（Table 3）
- `Jupyter Notebook` + `Analytica-L` 准确率接近 `Deep Research` + `Analytica-L`（仅差 1.34%），但成本极低。
- 证明 **强大的 Grounder 是性能基石**，而 Analytica 能有效放大其能力。

#### （3）开放权重小模型上的表现（Table 6）
- 在 `OpenAI-OSS-20B`（21B 参数）上，`Analytica-L` 将准确率从 55.57% 提升至 **64.24%**（+15.59%）。
- 表明该框架能**显著缩小小模型与大模型之间的能力差距**。

---

## 4. 关键结论和发现

### 主要发现
1. **SPR 框架有效降低推理误差**：通过结构化分解与线性合成，同时降低 **Bias** 与 **Variance**，实现更鲁棒的预测。
2. **Analytica 显著优于现有方法**：在多个领域平均提升 **15.84%** 准确率，且稳定性更高。
3. **Jupyter Grounder 具备极高性价比**：结合代码执行与数据分析，以极低成本实现接近顶级模型的性能。
4. **线性合成规则最鲁棒**：相比模糊逻辑，线性模型对输入噪声更不敏感，避免“蝴蝶效应”。
5. **高度可扩展**：支持递归调用与并行执行，分析节点增长 54 倍时，计算时间仅增加 12 倍（近线性）。
6. **支持交互式“what-if”分析**：通过 **Resynthesis** 功能，可快速重算反事实场景（如“如果失业率飙升会怎样？”）。

### 局限性
1. **假设子命题独立**：实际中子命题可能存在相关性，未显式建模协方差。
2. **合成器可靠性依赖**：若 Synthesizer 学习的权重（β）不准，可能引入系统误差。
3. **固定 Grounder 策略**：目前对所有叶节点使用相同的 Grounder，未根据命题类型动态路由。
4. **上下文长度限制**：尽管采用递归设计缓解，极端复杂的树仍可能超出 LLM 上下文窗口。

### 未来工作方向
- 开发 **自适应 Grounder 路由机制**，根据不同命题选择最优工具。
- 引入 **Probabilistic Graphical Models (PGMs)** 或 **Bayesian Networks** 显式建模子命题间的依赖关系。
- 探索 **更复杂的合成策略**，如基于注意力机制的加权融合。
- 将 Analytica 应用于更多领域，如 **机器人决策**、**政策制定** 和 **医疗诊断**。

---

> 🔗 **开源信息**：作者已公开代码与数据，详见 GitHub 仓库：[https://github.com/chengjunyanl/analytica](https://github.com/chengjunyanl/analytica)

</details>

---

### 7. [Large Language Models Explore by Latent Distilling](https://arxiv.org/abs/2604.24927)

**Authors**: Yuanhao Zeng, Ao Lu, Lufei Li, Zheng Zhang, Yexin Li, Kan Ren  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.24927v1  

#### Abstract
Generating diverse responses is crucial for test-time scaling of large language models (LLMs), yet standard stochastic sampling mostly yields surface-level lexical variation, limiting semantic exploration. In this paper, we propose Exploratory Sampling (ESamp), a decoding approach that explicitly en...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Large Language Models Explore by Latent Distilling*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
标准的随机采样（stochastic sampling）在大语言模型（LLM）推理时扩展（test-time scaling）中存在**语义探索不足**的问题。尽管能生成词法上多样的文本，但其底层推理路径往往高度重复，导致候选解冗余，限制了后续选择机制（如 reranking 或 majority voting）的效果。

### 提出的新方法：Exploratory Sampling (ESamp)
提出了一种名为 **Exploratory Sampling (ESamp)** 的新型解码算法，通过**潜空间蒸馏（latent distillation）** 来鼓励语义多样性。

#### 核心思想
- 引入一个轻量级的 **Latent Distiller (LD)**，在测试时在线训练，学习从 LLM 的浅层隐藏表示预测深层隐藏表示。
- 利用 Distiller 的**预测误差**作为“新颖性信号”（novelty signal）：高误差表示当前上下文在语义上是未被充分探索的。
- 在解码过程中，将该新颖性信号用于重加权候选 token 的概率分布，从而引导生成走向更少被探索的语义区域。

### 相比现有方法的优势
| 方法类别 | 代表方法 | 局限性 | ESamp 的优势 |
|--------|--------|------|-------------|
| **Stochastic Sampling** | Top-p, Min-p, FIRE | 主要产生词法多样性，难以改变深层推理逻辑 | 显著提升**语义多样性**而非表面变化 |
| **Structured Search** | Tree of Thoughts, Beam Search | 计算开销大，延迟高，不适合高吞吐场景 | **异步实现**，端到端开销 < 5%，实用性强 |
| **Logit-Level Control** | Contrastive Decoding, OverRIDE | 在词表空间操作，可能忽略不同表达形式下的语义等价性 | 在**连续潜空间**检测冗余，对语义重复更敏感 |

此外，ESamp 支持**协作式探索**：多个并行生成序列共享同一个 Distiller，形成隐式的“先到先得”调度机制，避免重复探索相同语义模式。

---

## 2. 核心实验方法和设置

### 数据集
实验覆盖四个领域，验证泛化能力：
- **数学推理**：AIME 2024 / 2025（竞赛级数学题）
- **科学问答**：GPQA-Diamond（生物、物理、化学专家验证难题）
- **代码生成**：LiveCodeBench v5（LeetCode 类编程题，防数据污染）
- **创意写作**：BookCorpus（故事续写任务）

### 实验设置
- **模型**：涵盖多种架构与能力的模型
  - Qwen2.5-7B/32B-Instruct（指令微调）
  - Qwen3-8B（专为推理优化）
  - GPT-OSS-20B（其他模型家族）
- **批大小与采样数**：支持单请求（B=1,K=1）到大规模并行（B=32,K=16）
- **上下文长度**：数学任务设为 8192，其余为 4096

### 评估指标
| 指标 | 含义 |
|-----|------|
| **Pass@k** | k 个样本中至少有一个正确的概率（主指标） |
| **Embedding Similarity** | 生成结果间的平均余弦相似度，衡量语义接近程度 |
| **Vendi Score** | 谱系多样性指标，量化有效独立语义簇的数量 |
| **Perplexity (PPL)** | 使用小模型评估语言流畅性与连贯性 |

### 基线方法对比
分为三类进行比较：
1. **Stochastic Sampling**：Vanilla, Min-p, FIRE
2. **Structured Search**：Tree of Thoughts (ToT)
3. **Logit-Level Control**：Contrastive Decoding, OverRIDE

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Pass@k）

#### 数学任务（AIME25, Qwen2.5-7B-Instruct）
| 方法 | Pass@16 | Pass@64 |
|------|---------|---------|
| Vanilla | 30.3% | — |
| Min-p | 29.5% | — |
| OverRIDE | 27.7% | — |
| **ESamp (Ours)** | **31.7%** | — |

> ✅ ESamp 在所有方法中表现最佳，显著优于基线。

#### 高效性突破（GPT-OSS-20B）
- **ESamp @ Pass@8 ≈ Vanilla/FIRE @ Pass@64**
- 表明 ESamp 可以用极小的采样预算达到传统方法大量采样的效果。

### 与基线方法的对比结果
- **Pass@k 扩展效率更高**：随着 k 增加，ESamp 持续提升，而部分基线（如 FIRE）在高 k 下被 vanilla 超越。
- **跨任务鲁棒性强**：在数学、科学、代码、创作等多个领域均优于或媲美最强基线，无明显短板。
- **打破多样性-连贯性权衡**：在创意写作中，ESamp 同时实现了最高 Vendi Score 和最低 PPL，说明其既多样又高质量。

### 消融实验结果

#### 探索强度 β 的影响（AIME25）
| β 值 | Pass@16 | Pass@64 |
|------|---------|---------|
| 0.1 | 26.7% | 40.0% |
| **0.25 (默认)** | **31.7%** | **46.7%** |
| 0.5 | 23.8% | 30.0% |

> β 过低退化为 vanilla；过高则过度惩罚置信度高的 token，损害性能。

#### Logit 融合方式对比
| 融合公式 | Pass@64 |
|--------|--------|
| `logit_new = (1+β)logit_ref - β*logit_dist` | **46.7%** |
| `logit_new = logit_ref - β*logit_dist` | 13.3% |

> 提出的 `(1+β)` 形式能更好保留原始分布结构，防止语法错误。

#### 替代方案对比
- **随机噪声注入**：若用同模长的高斯噪声替代 Distiller 错误向量，性能回落至 vanilla 水平 → 证明**错误方向包含结构化信息**。
- **词表空间蒸馏**：在输出分布空间训练 Distiller 导致不稳定且性能差 → 证明**潜空间估计更稳定有效**。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **潜空间预测误差可作为有效的语义新颖性信号**，成功引导 LLM 探索不同的推理路径。
2. ✅ **ESamp 显著提升了 Pass@k 效率**，尤其在数学推理任务上优势明显。
3. ✅ **打破了多样性与连贯性的传统权衡**，在保持甚至提升语言质量的同时增强语义多样性。
4. ✅ **异步实现几乎零开销**：在典型服务场景下，吞吐下降仅 **1.2%~4.25%**，具备工业部署价值。
5. ✅ **支持自然组合**：可与 FIRE、Self-Consistency 等方法结合，进一步提升性能。

### 方法的局限性
- **依赖模型内部结构访问权限**：需要获取中间层隐藏状态，不适用于黑盒 API。
- **对某些任务可能存在交叉干扰**：共享 Distiller 在异构问题（如不同 AIME 题目）中可能引入轻微负迁移（见 C.10），建议未来采用自适应共享策略。
- **超参数敏感性虽低但仍存在**：虽然 β=0.25 在多数情况下表现良好，但在极端任务中仍需调整。

### 未来工作方向
- 设计**自适应共享机制**：动态决定是否共享 Distiller，平衡跨提示学习与任务隔离。
- 将 Latent Distiller 思路推广至更多 test-time learning 场景，如在线微调、激活编辑等。
- 探索更高效的 Distiller 架构与更新策略，进一步降低资源消耗。
- 结合 reward modeling 或 verifier 进行两级筛选：先用 ESamp 扩展解空间，再用 verifier 精排。

---

> 🔗 **开源地址**：https://github.com/LinesHogan/tLLM  
> 📦 包含完整实现与优化版本（tLLM 框架），支持高效部署。

</details>

---

### 8. [Marco-MoE: Open Multilingual Mixture-of-Expert Language Models with Efficient Upcycling](https://arxiv.org/abs/2604.25578)

**Authors**: Fan Jiang, Yu Zhao, Chenyang Lyu, Tianqi Shi, Yichao Du, Feihu Jiang, Longyue Wang, Weihua Luo  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.25578v1  

#### Abstract
We present Marco-MoE, a suite of fully open multilingual sparse Mixture-of-Experts (MoE) models. Marco-MoE features a highly sparse design in which only around 5\% of the total parameters are activated per input token. This extreme sparsity, combined with upcycling from dense models, enables efficie...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# Marco-MoE: Open Multilingual Mixture-of-Expert Language Models with Efficient Upcycling 论文总结

## 1. 主要贡献和创新点

### 解决的问题
该论文旨在解决**多语言大模型中的“多语言诅咒”（curse of multilinguality）**问题。这一问题指在固定参数预算下，增加模型的语言覆盖范围通常会导致单个语言性能下降，原因包括容量瓶颈和跨语言干扰。现有的小型多语言模型（如 Tiny-Aya）多采用密集架构（dense architectures），难以在语言广度和任务深度之间取得平衡。

### 提出的新方法和创新点
论文提出了 **Marco-MoE**，一个开源的、高度稀疏的多语言 **Mixture-of-Experts (MoE)** 模型系列。其核心创新点在于将 **MoE Upcycling** 范式首次应用于优化紧凑规模下的多语言性能，并引入了细粒度的专家专业化设计。

具体创新包括：
1.  **首个稀疏多语言 Upcycling**：首次利用 MoE Upcycling 技术，将预训练好的密集模型（如 Qwen3-0.6B）高效地转换为多语言 MoE 模型，显著降低了计算开销。
2.  **细粒度专家专业化**：摒弃了传统的粗粒度专家复制（复制整个 FFN 层），采用了**子矩阵分割（sub-matrix splitting）**技术来初始化大量细粒度专家。结合 **Drop-Upcycling** 策略，有效促进了专家的多样化和专业化，避免了冗余瓶颈。
3.  **完全透明与开放**：公开了完整的预训练数据集、数据合成方法和四阶段预训练课程（curriculum），为社区树立了高性能多语言 LLM 开发的新标准。

### 相比现有方法的优势
- **更高的效率**：通过 Upcycling 和稀疏激活，仅需约 5% 的总参数被激活，实现了高效的预训练（5T tokens）。
- **更强的性能**：在同等或更低的计算成本下，在英语和多语言基准测试上超越了同类模型。
- **更好的可扩展性**：能够无干扰地扩展到 64 种语言，解决了密集模型在扩展时常见的性能退化问题。

---

## 2. 核心实验方法和设置

### 使用的数据集
Marco-MoE 在总计 **5.1T tokens** 的高质量数据上进行了预训练，数据来源分为四个阶段：
- **高质量英文数据**：Nemotron-CC-v2（高质和合成部分）、内部网络爬取的英文语料（经 Fineweb-EDU 过滤）。
- **推理与指令数据**：Nemotron-Pretraining-SFT-v1, Nemotron-Pretraining-Specialized-v1, FineMath, MegaMath, OpenThoughts3-1.2M, FLAN 等。
- **高质量多语言数据**：
  - **网络爬取数据**：Fineweb-2 及其高质量变体 Fineweb-2-HQ。
  - **合成数据**：
    - **多语言 QA 数据**：将英文 Diverse QA 数据集翻译成 13 种语言，并利用 Qwen3-30B 模型翻译至其余语言。
    - **多语言 STEM 数据**：通过翻译管道将英文 STEM 内容（来自 Nemotron-Pretraining-SFT-v1, OpenMathInstruct-2）投射到 28 种目标语言。
    - **文化与区域数据**：通过识别高文化密度的网页文档，并生成多样化的区域多选题（Synthetic-Regional-MCQs）。

### 实验设置和评估指标
- **模型架构**：基于 Qwen 框架的解码器专用 Transformer 架构，采用 MoE 层替代传统 FFN 层。关键配置包括 **Grouped-Query Attention (GQA)**、**RMSNorm**、**SwiGLU** 激活函数和 **Rotary Positional Embeddings (RoPE)**。
- **评估框架**：使用 **Light-Eval** 框架进行标准化评估。
- **评估维度**：
  1. **英语能力**：MMLU, MMLU-Redux, MMLU-Pro, AGIEval, BBH, ARC, HellaSwag, WinoGrande, GSM8K 等。
  2. **多语言通用能力**：GlobalMMLU, MMMLU, BELEBELE, mHellaSwag, mARC, FLORES-200, WMT24++ 等。
  3. **多语言文化与区域知识**：INCLUDE, Global-PIQA, CMMLU, C-Eval, ArabicMMLU, TurkishMMLU 等。

### 基线方法对比
- **Base 模型对比**：Qwen3-1.7B/4B, Granite4-Tiny, Llama3.2-3B, SmolLM3-3B, Gemma3-4B, Tiny-Aya-3.35B, Trinity-Nano/Mini。
- **Instruct 模型对比**：Qwen3-1.7B/4B-Instruct, Ministral3-3B/8B-Instruct, Gemma3-12B-Instruct, Granite4-Tiny/Small, LFM2-8B-A1B/24B-A2B。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
- **Marco-Mini-Base** (0.86B 激活参数)：
  - 英语平均得分：**63.7**
  - 多语言通用平均得分：**50.9**
  - 多语言文化与区域平均得分：**65.0**
- **Marco-Nano-Base** (0.6B 激活参数)：
  - 英语平均得分：**57.5**
  - 多语言通用平均得分：**42.3**
  - 多语言文化与区域平均得分：**55.6**
- **Marco-Mini-Instruct** (0.86B 激活参数)：
  - 英语平均得分：**75.5**
  - 多语言通用平均得分：**50.8**
  - 多语言文化与区域平均得分：**71.0**

### 与基线方法的对比结果
- **性能-计算比领先**：如图2所示，Marco-MoE Base 模型在性能-计算比上显著优于所有基线，尤其是在长尾和低资源语言上优势明显。
- **超越更大模型**：如图1所示，尽管激活参数仅为 0.6B 和 0.86B，**Marco-Nano-Instruct** 和 **Marco-Mini-Instruct** 的性能持续匹配或超越了拥有 **3-14倍更多激活参数** 的竞争模型。
- **多语言能力卓越**：在 BELEBELE 和 MGSM 等多语言基准上，Marco-MoE 模型取得了最佳成绩，证明了其强大的跨语言推理能力。

### 消融实验结果
- **四阶段预训练的有效性**：表8显示，随着预训练阶段的推进，模型在所有评估维度上的性能都稳步提升，特别是在第3和第4阶段，多语言能力得到显著增强。
- **数据成分的重要性**：表2-4的消融研究表明，加入多语言 QA、STEM 和文化区域数据均能带来一致的性能增益，尤其对低资源语言帮助巨大。
- **级联蒸馏（Cascaded Distillation）的效果**：表12表明，从 30B-A3B 教师模型切换到更强的 80B-A3B 教师模型后，学生模型在所有基准上都有进一步提升，验证了该策略的有效性。

---

## 4. 关键结论和发现

### 主要发现
1.  **高效且强大的多语言模型**：Marco-MoE 成功打破了多语言模型的容量瓶颈，证明了通过 **细粒度 MoE Upcycling** 可以构建出兼具高性能和高效率的多语言模型。
2.  **结构化的专家激活模式**：模型学习到了与语言学家族结构相似的专家激活模式。相关语言（如罗曼语族、斯拉夫语族）共享专家池，而孤立语言（如泰语、阿拉伯语）则使用专门的专家，这有助于减少干扰并促进正向迁移。
3.  **可扩展性强**：Marco-MoE 框架支持大规模语言扩展（已扩展至64种语言），且不会像密集模型那样产生严重的性能干扰。
4.  **计算效率优越**：Marco-MoE 系列模型在激活参数和训练计算量远低于许多基线的情况下，实现了顶尖的性能，树立了新的性能-计算比标杆。

### 方法的局限性
- **特定领域表现不足**：在高度本地化的考试风格知识（如 CMMLU 和 C-Eval 上的中文知识）方面，仍落后于专门为此类任务优化的模型（如 Qwen3-4B）。
- **依赖合成数据**：模型性能的提升部分依赖于翻译和合成数据，这些数据在真实性和多样性上可能存在局限。
- **语言扩展方式**：目前需要重新训练整个模型来集成新语言，缺乏模块化、增量式的扩展能力。

### 未来工作方向
- **扩大语言覆盖面**：将模型的覆盖范围扩展到更多全球语言，特别是那些资源极度匮乏的语言。
- **改进路由机制**：研究更先进的路由机制，以增强对极长尾语言的专家专业化。
- **开发模块化扩展**：探索无需重新训练即可高效添加新语言的方法，例如模块化、增量式学习。

</details>

---

### 9. [Heterogeneous Variational Inference for Markov Degradation Hazard Models: Discretized Mixture with Interpretable Clusters](https://arxiv.org/abs/2604.24818)

**Authors**: Takato Yasuno  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2604.24818v1  

#### Abstract
Bayesian finite mixture models can identify discrete risk clusters (low-risk vs. high-risk equipment), but face three critical bottlenecks: (1) insufficient degradation signals from coarse state discretization, (2) unstable cluster identification when data inherently supports fewer clusters than exp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Heterogeneous Variational Inference for Markov Degradation Hazard Models: Discretized Mixture with Interpretable Clusters*

---

## 1. 论文的主要贡献和创新点

### 解决的问题
工业基础设施（如泵、压缩机等）的退化模式具有显著的**个体异质性**（heterogeneity），传统生存分析模型假设所有设备共享相同的退化速率（homogeneous hazard rates），无法捕捉实际中设备间高达 **17× 到 34×** 的退化速度差异。这导致维护决策缺乏针对性。

尽管贝叶斯有限混合模型（finite mixture models）可用于识别高风险/低风险设备集群，但在实践中面临三大瓶颈：
1. **信号不足**：粗粒度状态离散化（如4个健康等级）导致退化事件稀少（仅1.3%），难以支撑稳定聚类；
2. **模型选择不稳定**：标准信息准则（如WAIC）倾向于选择复杂但不可解释的模型（如K=5），产生极小且无操作意义的簇；
3. **计算不可行**：MCMC（尤其是NUTS）推理耗时长达 **7小时以上**，不适用于生产环境迭代。

---

### 提出的新方法与创新思路

作者提出一个**面向生产的综合框架**，系统性解决上述挑战：

#### ✅ Solution 1: 8-state global percentile discretization  
采用**全局百分位数**将连续健康指标划分为 **8个离散状态**（12.5%~87.5%分位点），相比传统的4状态划分：
- 平衡各状态分布（每类约11.8%-13.1%）
- 将退化事件率从 **1.3% 提升至 2.4%**（+83%）
- 显著增强状态转移多样性，提升混合模型稳定性

#### ✅ Solution 2: Comprehensive feature engineering（30维特征）
融合多源信号构建高维协变量向量 $X \in \mathbb{R}^{30}$：
- **统计趋势**（22维）：90天窗口内的均值、方差、偏度、回归斜率、波动率等
- **连续健康指标**（2维）：归一化测量值、设备年龄
- **文本嵌入**（3维）：对巡检文本评论使用Sentence-BERT编码后经PCA压缩至3维，保留语义信息

#### ✅ Solution 3: Interpretable model selection rules
在WAIC基础上引入三条可解释性约束，防止过拟合：
1. **WAIC容忍阈值**：$\Delta\text{WAIC} \leq 50$
2. **最小簇占比**：$\min(\text{cluster share}) \geq 5\%$（至少14台设备/簇）
3. **最小簇分离度**：$\min(\Delta\mu) \geq 0.15$（对应exp(0.15)≈1.16倍风险差异）

#### ✅ Solution 4: Full-rank ADVI 替代 NUTS
使用**自动微分变分推断**（ADVI）替代传统MCMC进行后验估计：
- 使用 full-rank 高斯近似以捕获参数相关性
- 执行时间从 **7h40min（NUTS）降至5分钟**（84×加速）
- 在随机效应模型上验证其准确性（r > 0.99 vs NUTS）

---

### 相比现有方法的优势

| 维度 | 本文方法 | 传统方法 |
|------|---------|--------|
| **状态划分** | 8-state + 全局百分位 → 更丰富信号 | 通常4-state固定阈值 → 信号稀疏 |
| **特征工程** | 融合统计+连续+文本信号 | 多依赖简单协变量或忽略文本 |
| **模型选择** | 引入可解释性规则防过拟合 | 单纯依赖WAIC/BIC易选复杂模型 |
| **推理效率** | ADVI（5分钟）支持快速迭代 | NUTS（>7h）难用于生产部署 |
| **收敛稳定性** | ADVI无label switching，结果稳定 | NUTS存在label switching与发散风险 |

---

## 2. 核心实验方法和设置

### 数据集
- **设备类型**：工业泵（pump equipment）
- **规模**：280台设备，共 **104,703条检查记录**
- **时间跨度**：1991–2025年（34年）
- **采样频率**：月度至季度（平均间隔91天）
- **观测内容**：
  - 连续健康指标（振动/温度，归一化到[0,1]）
  - 巡检文本评论（82,416条，经Sentence-BERT嵌入为1024D → PCA压缩至3D）

---

### 实验设置

#### 模型配置
| 模型 | 类型 | 推理方法 | 参数 |
|------|------|----------|------|
| `mcl` | Random Effect（连续随机效应） | NUTS | 4 chains, 2000 draws, 1000 tune |
| `mc2` | Random Effect | ADVI（full-rank） | 20k iterations, 3k samples |
| `mix1` | Finite Mixture（K=2~5） | ADVI（grid search） | 同上 |
| `mix2` | Finite Mixture（K=2） | NUTS | 6 chains, init='advi' |

#### 评估指标
- **统计拟合**：WAIC（Widely Applicable Information Criterion）
- **收敛诊断**：
  - $\hat{r}$（Gelman-Rubin statistic）：<1.01 表示收敛
  - ESS（Effective Sample Size）：>400 为佳
- **可解释性指标**：
  - 最小簇占比（min cluster share）
  - 簇均值间距（$\Delta\mu$）
- **执行时间**：wall-clock runtime
- **相关性检验**：Pearson $r$ 与 RMSE 对比不同方法估计的 $u_i$

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ ADVI vs NUTS：随机效应模型（验证基准）
| 指标 | NUTS (`mcl`) | ADVI (`mc2`) | 对比 |
|------|-------------|--------------|------|
| 执行时间 | 45 min | **3 min** | **15× 加速** |
| $u_i$ 范围 | [-3.52, +2.85] | [-3.56, +2.86] | 几乎一致 |
| Pearson $r(u)$ | — | **0.997** | 极高一致性 |
| 显著异质性泵数 | 223/280 (79.6%) | 相同 | 完全匹配 |
| $\hat{r}$ | 1.0032 | N/A | 收敛良好 |

> ✔️ 结论：ADVI 在随机效应模型上与 NUTS 几乎完全一致，且速度快15倍。

---

#### ✅ Mixture Model：最优簇数选择（ADVI grid search）
| K | WAIC | 最小簇占比 | $\Delta\mu$ | 是否通过可解释性规则 |
|----|-------|------------|-----------|------------------|
| 2 | **19,788** | **27.1%** | **0.98** | ✅ 全部通过 |
| 3 | 19,814 (+26) | 2.9% | 0.56 | ❌ 不满足 min share ≥5% |
| 4 | 19,842 | 1.8% | 0.42 | ❌ |
| 5 | 19,875 | 0.7% | 0.29 | ❌ |

> ✔️ 最终选择 **K=2**：低风险（72.9%，降解慢2.7×）与高风险（27.1%，基准速率）

---

#### ✅ ADVI vs NUTS：混合模型（K=2）对比
| 指标 | ADVI (`mix1`) | NUTS (`mix2`) | 对比分析 |
|------|---------------|---------------|----------|
| 执行时间 | **5 min** | 7h40min (**460 min**) | **84× 更快** |
| $\mu_1$（低风险均值） | -0.98 | -3.52 | NUTS 极端异常 |
| $\mu_2$（高风险均值） | 0.00 | +0.88 | NUTS 失真严重 |
| $\Delta\mu$ 分离度 | 0.98 | 4.40 | 物理上不合理 |
| $\hat{r}(\mu)$ | N/A | **1.19–1.28** | **严重发散** |
| Min ESS | N/A | **17** | 样本无效 |
| 簇比例 | 72.9% / 27.1% | 39.6% / 60.4% | 因 label switching 倒置 |
| Label switching | 无 | 有 | 导致结果不可靠 |
| 生产可用性 | ✅ 稳定可复现 | ❌ 不可靠 | |

> ⚠️ 惊人发现：**NUTS 在混合模型上未能收敛，而 ADVI 反而更稳定！**

---

### 消融实验结果

#### （1）状态离散化影响（4-state vs 8-state）
| 指标 | 4-state（baseline） | 8-state（proposed） |
|------|--------------------|---------------------|
| 退化事件数 | 1,371 | **2,512** |
| 事件率 | 1.3% | **2.4%**（↑83%） |
| K=3 稳定性 | 出现空簇 | 存在但占比仅2.9% |
| K=2 可靠性 | 边缘通过 | **稳健通过** |

> ✔️ 证明：细粒度状态划分是混合模型稳定的前提。

#### （2）特征工程影响（7维 vs 30维）
| 模型 | 特征维度 | K=2 最小占比 | K=3 是否可行 |
|------|----------|----------------|----------------|
| Base | 7（基础） | 5.2%（勉强通过） | ❌ 空簇 |
| Full | 30（含统计+文本） | 27.1% | ✅ 存在但仍不满足 min share |

> ✔️ 发现：特征工程能提升信号质量，但不能“强行”制造不存在的簇结构（data inherently supports K=2）

---

## 4. 关键结论和发现

### 主要发现

1. 🔬 **细粒度状态离散化至关重要**  
   8-state 全局百分位划分使退化事件增加83%，是实现稳定混合建模的前提条件。

2. 🧪 **Full-rank ADVI 在混合模型上优于 NUTS**  
   - **速度**：84× 加速（5分钟 vs 7h40min）
   - **稳定性**：无 label switching，收敛可靠
   - **准确性**：在随机效应模型上与 NUTS 高度一致（r > 0.99）
   - **实用性**：唯一适合生产部署的方法

3. 📏 **可解释性规则有效防止过拟合**  
   单纯依赖 WAIC 会选出 K=3 模型，但因最小簇仅占2.9%，不符合运维实践需求。加入 min share ≥5% 和 min_gap ≥0.15 规则后，正确锁定 K=2。

4. 🛠️ **特征工程增强鲁棒性但不改变本质结构**  
   添加22个统计特征消除了空簇问题，但仍无法支持 K=3，说明数据本身只存在两个自然风险组。

5. 🏭 **生产级部署成为可能**  
   整个流程可在 **5分钟内完成**，支持每月重训练、动态风险评分、反事实模拟与CMMS集成。

---

### 局限性

1. ❌ **未建模故障（failure）而是退化（degradation）**  
   当前模型预测的是状态转移概率，而非最终故障时间。缺少泵更换/大修数据（right-censored）限制了寿命预测能力。

2. ❌ **未显式建模时间自相关性**  
   虽然使用90天滑动窗提取趋势，但未引入GP或AR结构来建模同一设备多次测量间的依赖关系。

3. ❌ **协变量选择未优化**  
   30个协变量中仅有6个显著（95% CI 不包含0），未来可用 spike-and-slab 或 horseshoe prior 进行稀疏选择。

4. ❌ **单设备类型验证**  
   当前结论基于泵类设备，是否推广至涡轮机、压缩机等其他设备尚需验证。

5. ❌ **文本嵌入利用不足**  
   仅用PCA压缩至3维，丢失大量语义细节。未来可微调领域专用语言模型（如 Maintenance-BERT）提取更丰富的语义特征。

---

### 未来工作方向

#### 近期方向：
- **Failure prediction extension**：结合退化路径与故障时间数据，构建联合模型
- **Causal inference**：引入维修干预数据，估计维护动作对退化轨迹的因果效应
- **Multi-equipment hierarchical models**：跨设备类型进行知识迁移（如泵→压缩机）

#### 长期方向：
- **Adaptive ADVI for online learning**：开发增量式变分更新机制，实现秒级模型刷新
- **Decision-theoretic optimization**：结合成本函数与不确定性，进行贝叶斯决策优化
- **Domain-specific language models**：训练专用于巡检日志的Transformer模型，提取故障模式、操作行为等深层语义

---

### 总结陈述

> 本文首次系统证明：**细粒度状态划分 + 可解释性正则化 + Full-rank ADVI** 是实现**可信赖、高效、可部署**的工业退化建模的关键组合。它不仅解决了传统MCMC方法在混合模型中的收敛难题，还颠覆了“MCMC一定优于VI”的固有认知，在真实工业场景中展现出更强的实用价值。该框架为智能资产管理系统提供了坚实的算法基础。

</details>

---

### 10. [STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator](https://arxiv.org/abs/2604.24544)

**Authors**: Alessio Sordo, Lingxiao Du, Meeka-Hanna Lenisa, Evgeny Bogdanov, Maxim Romanovsky  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.24544v1  

#### Abstract
The increasing reliance on Large Language Models (LLMs) across diverse sectors highlights the need for robust domain-specific and language-specific evaluation datasets; however, the collection of such datasets is challenging due to privacy concerns, regulatory restrictions, and the time cost for man...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：STELLAR-E: a Synthetic, Tailored, End-to-end LLM Application Rigorous Evaluator**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
当前对 **Large Language Models (LLMs)** 的评估严重依赖人工标注的真实数据集，这在实践中面临诸多挑战：
- **隐私与合规限制**：在金融、医疗等受监管领域，真实数据难以获取或使用；
- **创建成本高**：手动构建高质量数据集耗时且昂贵；
- **多语言支持不足**：大多数基准测试以英语为主，非英语语言缺乏可靠资源；
- **可扩展性差**：现有自动化方法通常基于已有数据增强或翻译，无法实现完全合成、定制化生成。

这些问题导致 LLM 应用的持续监控（如 LLMOps 中的 CI/CD）变得困难。

---

### **提出了什么新方法或新思路**
本文提出 **STELLAR-E** —— 一个**全自动化、端到端的合成指令-答案（Instruction-Answer, I&A）数据生成与评估系统**，用于评估多语言、领域特定的 LLM 应用。

其核心架构分为两个阶段：
1. **合成数据引擎**：基于改进的 **TGRT Self-Instruct 框架**，通过可控提示生成高质量、多样化的 I&A 对；
2. **评估流水线**：结合统计指标和 **LLM-as-a-Judge** 方法，自动评估生成数据的有效性和挑战性。

关键创新机制包括：
- **DVE (Diversity Enhancement)**：利用嵌入模型进行语义去重，提升数据多样性；
- **DFE (Difficulty Enhancement)**：通过对抗性改写提升指令难度，防止生成“过于简单”的样本；
- **反馈循环机制（Feedback Loop）**：低质量输出由独立 LLM 提供反馈并重新生成，显著提高数据保真度。

---

### **相比现有方法的优势**
| 特性 | STELLAR-E | 其他方法（如 YourBench, OmniEval, BENCHAGENTS） |
|------|-----------|---------------------------------------------|
| 是否依赖真实文档 | ❌ 不依赖 | ✅ 多数依赖输入文档 |
| 可控性与定制化 | ✅ 支持语言、领域、格式灵活控制 | ⚠️ 定制能力有限 |
| 数据完全合成 | ✅ 是 | ⚠️ 部分依赖真实数据或翻译 |
| 多语言原生支持 | ✅ 内建多语言生成与评估 | ❌ 主要面向英语 |
| 可扩展性 | ✅ 支持大规模批量生成 | ⚠️ 生成数量受限，无修复机制 |
| 质量保障机制 | ✅ 含 DVE/DFE + 反馈循环 | ⚠️ 多为过滤而非优化 |

> ✅ **优势总结**：STELLAR-E 实现了**无需真实数据、高度可配置、可扩展、高质量的端到端评估流程**，特别适用于企业级 LLM 应用的质量保证。

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **真实基准数据集**：`Mintaka`（一个多语言、复杂的端到端问答数据集），使用其英文（`mintaka_en_real`）和意大利文版本（`mintaka_it_real`）作为黄金标准。
- **机器翻译对照组**：将 `Mintaka` 英文集通过 API 自动翻译成意大利文（`mintaka_it_translated`），用于比较翻译 vs 合成效果。
- **合成数据集**：
  - `mintaka_en_synthetic` 和 `mintaka_it_synthetic`：由 STELLAR-E 在相同领域下生成的英/意双语数据集。
- 所有数据集最终均采样至 **1,500 条 I&A 对** 进行公平比较。

---

### **实验设置**
- **生成参数**：
  - 使用 8 种 Question Types (QTs) 控制领域；
  - 每轮迭代生成 50 条指令，共 50 轮；
  - G-Eval 判定阈值设为 **T=8/10**；
  - DVE 相似度阈值为 **0.3**。
- **模型选择**：
  - **生成模型**：Gemini 1.5 Pro 002；
  - **评估模型（Judge LLM）**：Gemini 2.5 Pro；
  - **过滤/打分模型**：Gemini 2.0 Flash 001；
  - **嵌入模型**：bge-m3（用于多语言语义距离计算）。

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **G-Eval** | 基于 Chain-of-Thought 的 LLM-as-a-Judge 指标，综合评分 Accuracy, Relevance, Completeness； |
| **ROUGE-L** | 衡量生成答案与参考答案之间的最长公共子序列，反映句法相似性； |
| **BERTScore F1** | 基于上下文嵌入的语义相似度指标，适合跨语言比较； |
| **Answer Relevance** | 评估回答是否紧扣问题，避免冗余或偏离主题； |

> 所有指标均在强模型（Gemini 2.5 Flash）和弱模型（Llama 2 Chat 13B）上分别测试，验证泛化性。

---

### **基线方法对比**
- **Real Dataset**：原始人类标注的 Mintaka 数据集（理想情况）；
- **Translated Dataset**：机器翻译版 Mintaka（常见替代方案）；
- **Synthetic Dataset (w/o DVE/DFE)**：仅基础生成；
- **Synthetic Dataset (w/ DVE)**：加入多样性增强；
- **Synthetic Dataset (w/ DVE & DFE)**：完整版本（本文方法）；

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（见 Table 3 & 4）**

#### **在强模型上的表现（Gemini 2.5 Flash）**
| 数据集 | G-Eval 分数 | 相对于 Real 的差距 |
|--------|-------------|------------------|
| Real EN | 8.17 | — |
| Synthetic EN (DVE+DFE) | 8.74 | **+5.7%** |
| Real IT | 8.00 | — |
| Synthetic IT (DVE+DFE) | 8.59 | **+5.8%** |

✅ **结论**：合成数据平均仅高出真实数据 **+5.7%**，表明其具有相当的评估效力。

#### **在弱模型上的表现（Llama 2 Chat 13B）**
| 数据集 | G-Eval 分数 | 相对于 Real 的差距 |
|--------|-------------|------------------|
| Real EN | 5.69 | — |
| Synthetic EN (DVE+DFE) | 6.78 | **+10.9%** |
| Real IT | 4.28 | — |
| Synthetic IT (DVE+DFE) | 4.35 | **+0.7%** |

⚠️ **观察**：小模型在合成数据上得分更高，说明合成数据对其而言**相对更易回答**，可能存在“捷径”线索。

---

### **与基线方法的对比结果**
- **相比纯翻译数据**：
  - 意大利语翻译数据（`Translated IT`）G-Eval 得分为 8.23（vs Real 8.00），说明翻译后任务反而变简单（可能因去除歧义）；
  - 而合成数据（8.59）更接近真实复杂度，且能保留文化语境；
- **相比未优化合成数据**：
  - 无 DVE/DFE 的合成数据 G-Eval 高达 9.43（+12.6%），明显“放水”；
  - 加入 DVE 和 DFE 后显著降低偏差，逼近真实水平。

---

### **消融实验结果**
| 配置 | G-Eval 差距（EN） | 效果分析 |
|------|------------------|---------|
| No DVE / No DFE | +12.6% | 数据太简单，模型轻松答对 |
| +DVE only | +9.7% | 多样性提升，但仍偏容易 |
| +DVE + DFE | **+5.7%** | 显著缩小差距，难度更合理 |

📌 **发现**：**DFE 是最关键模块**，它有效提升了指令的挑战性，使合成数据更具评估价值。

此外：
- **ROUGE-L 在 DVE&DFE 下最低** → 回答句式更多样，不照搬模板；
- **BERTScore F1 接近甚至略低于真实数据** → 语义表达差异更大，考验模型理解力；
- **Answer Relevance 维持高位** → 回答始终聚焦问题本身。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **合成数据可以达到接近真实数据的评估质量**：
   - 平均 G-Eval 差距仅为 **+5.7%**，证明 STELLAR-E 能生成具备中等挑战性的高质量测试集。
2. ✅ **DVE 和 DFE 显著提升数据质量**：
   - 尤其是 DFE，能有效防止生成“过于简单”的指令，增强对 LLM 能力的探测。
3. ✅ **多语言原生生成优于机器翻译**：
   - 翻译数据存在“翻译腔”（translationese）和文化错位风险，而 STELLAR-E 可直接生成符合本地语用习惯的内容。
4. ⚠️ **小模型更容易被合成数据“误导”**：
   - 弱模型在合成数据上得分增幅更大（+10.9%），暗示当前合成策略可能仍含有一些模式线索，需进一步优化。

---

### **方法的局限性**
1. **仍依赖 LLM 作为评判者（LLM-as-a-Judge）**：
   - 存在潜在的自增强偏见（self-enhancement bias），即生成模型与评判模型同源时可能导致评分膨胀。
2. **尚未经过人类专家验证**：
   - 缺乏 native speaker 对意大利语等非英语数据的文化适配性评估。
3. **仅在一个基准（Mintaka）上验证**：
   - 泛化性有待在更多领域（如金融、法律）和数据集上检验。
4. **未处理多轮对话场景**：
   - 当前仅支持单轮 I&A 对，未来可拓展至 multi-turn RAG 场景。

---

### **未来工作方向**
1. **扩大元评估范围**：
   - 引入更多模型家族（如 Llama、Qwen、Claude）参与生成与评判，减少偏见；
   - 构建生成器-评判器异构组合（cross-family evaluation）。
2. **引入人工评估环节**：
   - 对少量合成样本进行 native speaker 审查，分析文化相关性与语言自然度。
3. **扩展至 RAG 场景**：
   - 生成合成 source documents，并构建基于这些文档的 grounded I&A 对，用于评估 RAG 系统。
4. **模块复用与开放集成**：
   - 各组件（如 Topic Generator、DFE Module）可独立使用，可用于训练数据合成、强化学习微调等任务。
5. **探索动态难度调节机制**：
   - 根据被测模型的能力动态调整 DFE 强度，实现个性化压力测试。

---

> 🔚 **总结一句话**：  
> **STELLAR-E 成功实现了无需真实数据、全自动、可定制的 LLM 应用评估框架，在质量和效率之间取得了良好平衡，为工业级 LLMOps 提供了一种高效可行的新范式。**

</details>

---

### 11. [BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate](https://arxiv.org/abs/2604.25203)

**Authors**: Arnon Mazza, Elad Levi  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.25203v1  

#### Abstract
Deploying guardrails for custom policies remains challenging, as generic safety models fail to capture task-specific requirements, while prompting LLMs suffers from inconsistent boundary-case performance and high inference costs. Training custom classifiers achieves both accuracy and efficiency, yet...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate》核心总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在部署 **Custom Policy Guardrails**（自定义策略防护）时面临以下挑战：
- **通用安全模型**（如 LlamaGuard、ShieldGemma）无法适应特定任务的政策需求。
- **直接提示 LLM 进行判断**（LLM-as-a-Judge）存在边界案例表现不稳定、推理成本高、标签噪声大等问题。
- **训练定制分类器**虽然准确且高效，但依赖大量人工标注数据，成本高昂。

因此，如何在**仅有任务描述和少量无标签样本**的情况下，生成高质量、多样化的训练数据来训练高效的定制化 guardrail 模型，是一个关键难题。

### 提出了什么新方法或新思路
本文提出 **BARRED**（Boundary Alignment Refinement through REflection and Debate），一个基于合成数据的框架，用于训练自定义策略防护模型。其核心创新包括：

1. **维度分解**（Dimension Decomposition）  
   将任务领域空间按语义相关维度进行系统性拆解（如“语气风格”、“违规类型”、“句法变换”等），并通过 **Verbalized Sampling** 技术对每个维度实例化，确保生成数据覆盖全面、避免模式坍缩（mode collapse）。

2. **多智能体辩论验证机制**（Multi-Agent Debate Validation）  
   引入由 **Advocate** 和多个 **Judges** 组成的辩论系统：
   - **Advocate** 固定支持生成样本的标签并提供推理；
   - **Judges** 独立评估并在多轮讨论中更新判断；
   - 只有当所有 Judges 达成共识且与目标标签一致时，样本才被接受。
   此机制有效提升了标签的**保真度**（faithfulness）。

3. **迭代精炼流程**（Iterative Refinement）  
   对未通过辩论验证的样本，利用 Judges 的反馈进行结构化修正，而非简单丢弃，从而提高数据质量与利用率。

### 相比现有方法的优势
| 方面 | 现有方法 | BARRED |
|------|--------|-------|
| 数据来源 | 依赖人工标注或混合增强 | 完全从合成数据训练，无需人工标注 |
| 准确性 | 动态方法（如 DynaGuard）精度较低 | 接近甚至超越 SOTA LLM |
| 效率 | LLM-as-a-Judge 推理延迟高 | 训练小型模型（SLM），部署低延迟 |
| 泛化能力 | 静态模型难以适配新政策 | 支持任意策略定义，快速迁移 |

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
论文在四个不同领域的 guardrail 任务上进行了评估，涵盖对话、代理输出、医疗合规场景：

| 数据集 | 类型 | 描述 |
|-------|-----|------|
| **PE Repetition** | Dialogue | 判断客服对话中是否出现用户重复提问超过3次需引导 |
| **PE Privacy** | Dialogue | 是否泄露员工 GPS 坐标位置信息 |
| **Plan Verification** | Structured Plan | AI Assistant 输出的研究计划是否符合指令格式要求（如正确使用 `<end_plan>`） |
| **Health Compliance** | Q&A | 回答是否包含“健康建议”（health advice），涉及监管合规 |

> 所有测试集均包含两部分：**人类标注样本**（Human）和**人工验证过的合成样本**（Synth），总计约 800+ 测试样例（见 Table 1）。

### 实验设置和评估指标
- **评估指标**：**Accuracy**（准确率）
- **训练方式**：使用 BARRED 生成 1000 个合成训练样本 → 微调多种规模的语言模型（从 1.5B 到 14B 参数）
- **推理配置**：微调后的模型仅输出 `0` 或 `1`，不带解释（prompt 见 Appendix A.4）

### 基线方法对比
分为两类强基线：

#### （1）LLM-as-a-Judge
直接用前沿 LLM 对输入进行分类：
- GPT-4.1-nano, GPT-4.1-mini, GPT-4.1, GPT-5-mini（reasoning model）
- Qwen2.5-14B（开源基线）

#### （2）通用 Guardrail 模型
专为支持自定义策略设计的安全模型：
- **OSS-Safeguard-20B**：OpenAI 推出的安全推理模型
- **Glider**：3.8B 参数通用评估模型，训练于 685 个领域

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 3）
在所有任务和测试集上，**基于 BARRED 合成数据微调的小型模型显著优于各类基线**：

| 模型 | 平均 Accuracy（Human + Synth） |
|------|-------------------------------|
| 最佳基线（GPT-5-mini） | ~0.87 |
| **ft-Qwen14B（BARRED）** | **~0.94** |
| **ft-4.1-nano（BARRED）** | **~0.95** |

> 即使是 **3B 参数的 ft-Qwen3B**，也普遍优于 **20B 的 OSS-Safeguard** 和 **GPT-4.1**。

#### 典型结果示例（Health Compliance - Synth Set）：
- GPT-4.1: 0.97
- OSS-Safeguard-20B: 0.81
- **ft-Qwen14B (BARRED)**: **0.98**

表明：**更小的模型 + 高质量合成数据 > 更大的通用模型**

### 与基线方法的对比结果
- 在所有四项任务中，**finetuned models using BARRED 均取得第一或第二名成绩**，且多数情况下大幅领先。
- 特别是在复杂任务（如 Privacy、Health）上，优势更为明显。
- 所有 LLM-as-a-Judge 方法在 human-annotated 数据上表现波动较大，说明存在泛化问题。

### 消融实验结果（Ablation Studies）

#### （1）辩论验证机制的影响（Table 4）
| 设置 | Human Acc | Synth Acc |
|------|-----------|----------|
| **完整 BARRED（含 Debate）** | **0.85** | **0.99** |
| 无验证（No verification） | 0.58 | 0.65 |
| 自我精炼（Self-Refine） | 0.53 | 0.65 |

> 移除辩论验证导致 **准确率下降达 27%**；而 Self-Refine 表现更差，说明单模型容易陷入确认偏误。

#### （2）维度分解的作用（Figure 3）
- 随着维度实例化数量增加，**测试集覆盖率**和**模型准确率**均提升。
- 准确率呈对数增长趋势，表明适度的维度即可捕获大部分任务空间。
- 证明了该机制能有效提升数据多样性。

#### （3）模型规模影响（Figure 2）
- 简单任务（Repetition）在小模型（1.5B）即饱和；
- 复杂任务（Privacy, Health）随模型容量增大持续增益；
- 但即使是 1.5B 模型也能达到竞争性性能，体现合成信号的有效性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. ✅ **仅靠合成数据可训练出高性能定制 guardrail 模型**，无需任何人工标注。
2. ✅ **维度分解 + 多智能体辩论** 是保证数据**多样性**与**标签保真度**的关键。
3. ✅ 微调后的 **小型语言模型（SLM）可超越大型 LLM 和专用 guardrail 模型**，实现更高精度与更低延迟。
4. ✅ 该范式适用于多种任务类型（对话、结构化输出、合规审查），具有良好的**通用性**。

### 方法的局限性
- **生成阶段计算开销较高**：需要多次调用 LLM 进行维度提取、样本生成与多轮辩论。
- 当前依赖较强 LLM（如 GPT-5-mini）作为 generator 和 judge，可能限制完全开源复现。
- 尚未处理多标签或多层级分类任务。
- Debate 机制的成功依赖于 judge 的多样性与推理能力（见 Wu et al., 2025 引用）。

### 未来工作方向
- 扩展至 **multi-label 和 hierarchical classification** 场景。
- 探索 **跨任务间合成数据迁移**（transfer of synthetic data）的可能性。
- 融合 **human feedback loop** 实现迭代优化。
- 降低生成成本，探索轻量化 debate 架构。

---

> **Impact Statement**：BARRED 降低了组织构建高质量内容审核系统的门槛，有望让资源有限的机构也能部署精准、高效的 AI 安全防护机制，推动 AI 安全技术的民主化。

</details>

---

### 12. [Backtranslation Augmented Direct Preference Optimization for Neural Machine Translation](https://arxiv.org/abs/2604.25702)

**Authors**: Mehrdad Ghassabi, Spehr Rajabi, Hamidreza Baradaran Kashani, Sadra Hakim, Mahshid Keivandarian  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.25702v1  

#### Abstract
Contemporary neural machine translation (NMT) systems are almost exclusively built by training on supervised parallel data. Despite the tremendous progress achieved, these systems still exhibit persistent translation errors. This paper proposes that a post-training paradigm based on reinforcement le...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Backtranslation Augmented Direct Preference Optimization for Neural Machine Translation*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 Neural Machine Translation (NMT) 模型严重依赖大规模平行语料进行监督训练，导致模型容易学习到训练数据中的系统性偏差，产生“**translationese**”（非自然、机械感强的译文）。尽管已有进步，这些模型在**流畅性、充分性和语义一致性**方面仍存在持续性错误。

此外，现有的基于 Reinforcement Learning (RL) 或 Direct Preference Optimization (DPO) 的后训练方法在机器翻译任务中应用有限，且往往缺乏对偏好数据质量的有效控制。

### 🚀 提出的新方法与创新思路
本文提出了一种**基于 DPO 的新型后训练框架**，结合 **backtranslation** 自动生成高质量的偏好对 (preference pairs)，用于优化预训练 NMT 模型。其核心流程如下：

1. 利用一个 **expert translator**（可以是人类或更强的 AI 模型）将源语言句子翻译为目标语言；
2. 将该目标语言翻译结果输入学生模型 $T_g$ 进行 **backtranslation** 回源语言；
3. 若回译结果与原始源句差异较大（通过 BLEU 和 COMET 评分判断），则构成一个偏好对：
   - **Winner**: 原始正确源句
   - **Loser**: 学生模型生成的有缺陷回译
4. 使用这些偏好对，采用 **DPO** 对学生模型进行 fine-tuning，提升其翻译能力。

> 🔍 **关键创新点**：
- 首次将 **backtranslation 作为诊断工具** 来识别翻译缺陷，并自动生成可用于 DPO 的偏好数据。
- 不依赖人工标注的平行偏好数据，仅需单语语料 + 专家翻译器即可构建高质量训练信号。
- 引入 **双层过滤机制**（BLEU + COMET 肘部检测）确保偏好对具有清晰的质量区分度，增强训练稳定性。

### ⭐ 相比现有方法的优势
| 方面 | 优势 |
|------|------|
| 数据效率 | 无需大量人工标注的偏好数据，利用 monolingual corpus 即可生成训练样本 |
| 成本与可扩展性 | 可适配低资源语言或专业领域，降低对平行语料的依赖 |
| 训练稳定性 | DPO 替代传统 RL，避免奖励模型过拟合；配合 LoRA 实现参数高效微调 |
| 效果显著 | 在高资源语言对（en→de）上仍取得明显提升，说明方法普适性强 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集
- **WMT14** 英德翻译数据集：标准 benchmark，包含高质量英文原文及其专业德语翻译。

### ⚙️ 实验设置
- **基线模型**：`gemma3-1b` —— 一个中等规模的多语言小模型（Small Language Model）
- **学生模型**：以 `gemma3-1b` 为起点，经过 DPO 微调后命名为 `amestris-1b`
- **专家翻译器**：使用更强的语言模型模拟 expert translator 生成目标语言翻译
- **backtranslation 执行者**：学生模型自身（即 `gemma3-1b`）
- **偏好数据构建流程**：
  1. 对每个源句 $s$，由 expert 生成翻译 $t$
  2. 学生模型将 $t$ 回译成 $\hat{s}$
  3. 若 $\text{BLEU}(s, \hat{s}) <$ 阈值 或 $\text{COMET}(s, \hat{s}) <$ 肘点（knee point = 0.7233），保留为候选偏好对
- **DPO 微调配置**：
  - 使用 **LoRA**（Low-Rank Adaptation）进行参数高效微调
  - 冻结主干模型权重，只训练 adapter 层
  - 超参数见 Table I（如 DPO 温度 $\beta=0.1$, LoRA rank=32）

### 📊 评估指标
| 指标 | 类型 | 含义 |
|------|------|------|
| **BLEU ↑** | N-gram 匹配精度 | 衡量输出与参考译文的词汇重叠程度 |
| **COMET-DA ↑** | 神经评估指标（基于 WMT22） | 与人工评分高度相关，反映整体翻译质量 |
| **COMET-QE ↑** | QE-based 神经评估（unbabel/cometkiwi-da） | 更关注语义一致性和流畅性 |
| **METEOR ↑** | 改进的精确率/召回率 | 考虑同义词和词干匹配 |
| **TER ↓** | 编辑距离 | 数值越低越好，表示修改次数少 |
| **chrF++ ↑** | 字符级 F-score | 对形态丰富的语言更敏感 |

### 🔁 基线对比
- **Baseline**: `gemma3-1b`（未经 DPO 微调）
- **Proposed Method**: `amestris-1b`（经 backtranslation-augmented DPO 微调）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table II）

| Metric | gemma3-1b (baseline) | amestris-1b (ours) | 变化趋势 |
|--------|------------------------|--------------------|----------|
| **BLEU ↑** | 0.1572 | 0.1500 | ↓（轻微下降） |
| **COMET-DA ↑** | 0.7698 | **0.7810** | ↑ +0.0112 |
| **COMET-QE ↑** | 0.7031 | **0.7476** | ↑ **+0.0445** |
| **METEOR ↑** | 0.3861 | **0.3969** | ↑ +0.0108 |
| **TER ↓** | 77.65 | **76.21** | ↓ -1.44 |
| **chrF++ ↑** | 41.93 | **43.82** | ↑ +1.89 |

> 💡 **核心亮点**：
- **COMET-QE 提升达 +0.0445**，表明模型在语义保真度和自然度上有显著改善。
- 尽管 BLEU 略有下降，但在所有其他更贴近人类判断的指标（COMET、METEOR、chrF++）上均实现一致提升，说明 DPO 成功提升了**语义充分性和流畅性**，而非简单追求 n-gram 匹配。
- TER 下降也说明译文更接近参考文本，编辑成本更低。

### ❌ 消融实验（Ablation Study）
文中未明确列出独立的消融实验表格，但从方法设计中可推断以下关键组件的作用：
- **Backtranslation + 双重过滤（BLEU + COMET 肘部检测）** 是保证偏好对质量的核心，若仅依赖 BLEU 易引入噪声。
- **DPO + LoRA** 构成了稳定高效的优化路径，避免了传统 RL 中 reward hacking 和训练不稳定问题。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **DPO 可有效用于 NMT 后训练**：即使在高资源语言对（en→de）上，也能显著提升翻译质量，尤其是在 COMET 等高级评估指标上表现突出。
2. **Backtranslation 是有效的诊断机制**：通过让学生模型回译专家翻译，能自动识别其语义理解缺陷，并生成有针对性的训练信号。
3. **偏好数据质量至关重要**：引入 COMET 分布的肘部检测作为筛选标准，确保 winner-loser 对之间存在明显质量差距，从而强化 DPO 的学习效果。
4. **无需人工标注偏好数据**：整个流程仅需单语语料和一个 expert translator（AI 或人），具备良好的可扩展性和实用性。

### ⚠️ 方法的局限性
- 当前实验集中在 **English→German** 这一高资源语言对，尚未验证在低资源或远距语言对上的泛化能力。
- 依赖一个“更强”的 expert translator 来生成初始翻译，若 expert 性能不足会影响整个流程质量。
- 未进行人工评估（如 MQM），结果主要依赖自动指标，尤其是 COMET 的代理效应需谨慎解读。

### 🔮 未来工作方向（Future Research）
1. **Domain Adaptation**：将该框架应用于医学、法律等专业领域，结合领域专家反馈进一步提升术语准确性和减少幻觉。
2. **改进偏好构造策略**：探索不同的采样方式、DPO 变体（如 SimPO、ORPO）以及更高效的参数微调技术（如 QLoRA）。
3. **完全去中心化的偏好生成**：研究无需固定 expert model 的 self-improvement 机制，实现闭环优化。
4. **跨语言迁移能力研究**：探究在一个语言对上训练的偏好模式是否可迁移到其他语言对。

---

> 📢 **补充信息**：  
作者已公开全部代码、模型和实验资源，仓库地址：[github.com/mehrdadghassabi/Amestris](https://github.com/mehrdadghassabi/Amestris)，便于复现与社区拓展。

</details>

---

### 13. [Nautile-370M: Spectral Memory Meets Attention in a Small Reasoning Model](https://arxiv.org/abs/2604.24809)

**Authors**: Maixent Chenebaux  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.24809v1  

#### Abstract
We present Nautile-370M, a 371-million-parameter small language model designed for efficient reasoning under strict parameter and inference budgets. Nautile-370M uses a hybrid backbone in which two SeqCond Attention (SCA) layers, a linear-time spectral sequence operator inspired by SeqCondenser, alt...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Nautile-370M: Spectral Memory Meets Attention in a Small Reasoning Model

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文旨在解决**小规模语言模型在严格参数和推理预算下进行高效推理**的挑战。传统 transformer 模型虽然表达能力强，但在长上下文场景中计算复杂度为 $O(L^2)$，难以部署于资源受限环境。而结构化序列模型（如 SSM）虽具线性效率，却缺乏 attention 的灵活 token-to-token 路由能力。

Nautile-370M 的目标是设计一个兼具**高效性与强推理能力**的小模型，在有限算力条件下实现高质量的 reasoning 表现。

---

### 提出的新方法与新思路

#### （1）**SeqCond Attention (SCA)**  
一种受 **characteristic function 导数**启发的新型序列建模机制：
- 将历史 token 序列表示为其经验特征函数的梯度：$\nabla_\theta \phi_X(\theta) = i\mathbb{E}[X e^{i\langle\theta,X\rangle}]$
- 利用该导数作为“spectral memory”来编码前缀信息
- 通过 **Hermitian inner product** 进行读取，形式上类比 attention，但查询的是频谱空间中的分布摘要而非显式 key-value 对

#### （2）**理论可表达性证明**
论文从理论上证明了 SCA 的强大表达能力：
- **Theorem 1**: 可以精确恢复任意单个历史 token（exact token retrieval）
- **Corollary 3**: 可恢复带权分布 $\sum_k p_k h_k$
- **Corollary 4**: 可以复现任意 softmax attention 输出（即 SCA 至少与 full self-attention 同样 expressive）

> ✅ 关键洞见：使用特征函数的**梯度**而非函数本身，使得嵌入向量 $h_k$ 成为相位指数的乘子，从而可通过线性读出直接提取；若仅用 $\phi_X(\theta)$，则只能恢复权重 $p_k$，无法获取 $h_k$。

#### （3）**混合架构设计：SCA + Transformer**
提出一种 **2:1 混合堆叠结构**：
- 每两个 SCA 层后接一个标准 transformer 层 → 总共 16 个 SCA 层 + 8 个 transformer 层
- 设计动机：大部分语言建模任务适合增量状态更新（SCA 擅长），少数需精确比较的任务（如指代消解）由 attention 处理

#### （4）**训练流程优化**
针对小模型强化学习中的失败模式提出改进：
- **Gradient-balanced GRPO**：对正负优势梯度分别归一化，防止低成功率时负梯度主导
- **Scored self-distillation**：在策略自蒸馏中，仅保留具有正 advantage 的正确推理轨迹，并按优势加权监督损失，形成自动课程学习

---

### 相比现有方法的优势

| 维度 | 优势 |
|------|------|
| **效率** | SCA 支持 $O(1)$ 推理状态更新，训练时使用 parallel scan，优于标准 attention 的 $O(L)$ |
| **表达力** | 在连续极限下至少等价于 full self-attention，且更通用（无非负性和归一化约束） |
| **结构互补性** | SCA 擅长统计累积，attention 擅长稀疏精准匹配，二者协同增强整体推理能力 |
| **训练稳定性** | 针对小模型 RL 提出 gradient-balanced GRPO 和 scored self-distillation，显著提升收敛性 |

---

## 2. 核心实验方法和设置

### 数据集

| 类型 | 数据集 | 规模 | 描述 |
|------|--------|------|------|
| 预训练 | **FineWeb-Edu** | ~350B tokens | 教育导向网页文本，提供广泛事实与语言知识 |
| 预训练 | **SYNTH [11]** | ~250B tokens | 合成 chain-of-thought 数据集，强调逻辑推理与指令遵循 |
| 增强数据 | **Template-distilled synthetic data** | ~4M documents | 基于多种 teacher 模型生成并格式对齐至 SYNTH 风格，用于提升响应一致性 |

---

### 实验设置

- **模型规模**：371M 参数（decoder-only）
- **上下文长度**：1024
- **骨干结构**：24 层，交替 `SCA → SCA → Transformer` 结构
- **训练平台**：
  - Pretraining & SFT：Google TRC 提供的单个 TPU v4-64 pod slice
  - Reinforcement Learning：NVIDIA DGX Spark
- **Tokenizer**：cl100k_base
- **Weight tying**：启用

---

### 评估指标

- 主要 benchmark：**GSM8K (pass@1)** — 数学应用题准确率
- 其他 zero-shot 评测：
  - OpenBookQA, ARC, CommonsenseQA（常识推理）
  - PIQA, IFEval（物理/指令遵循）
  - TriviaQA, MMLU, MMLU-Pro（知识问答）
  - MATH500, GPQA Diamond（高难度推理）
- 所有评测采用 **strict scoring**：若输出多个答案，则判错

---

### 基线方法对比

对比同类规模公开模型：
- **Qwen2.5-0.5B**
- **Granite-350M**
- **SmolLM2-360M**

> 注：这些模型训练 token 数远超 Nautile（最高达 28T vs ~0.8T），突显其数据效率优势。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）

| Benchmark | Nautile-370M | 最佳基线 |
|----------|--------------|---------|
| **GSM8K (0-shot)** | **33.4%** | 33.0% (Granite-350M) |
| **MATH500** | 2.4% | 18.8% (Qwen2.5) |
| **GPQA Diamond** | **27.3%** | 26.3% (Granite-350M) |
| **Average Accuracy** | **35.7%** | 33.3% (Granite-350M) |

> 💡 尽管总训练 token 较少（~0.8T vs 其他 4T–28T），Nautile 在多数推理任务上仍取得领先或第二的成绩，尤其在 GSM8K 和 GPQA 上表现突出。

---

### 强化学习阶段消融实验（Table 1）

| 训练阶段 | GSM8K 准确率 |
|--------|-------------|
| After SFT (baseline) | 27.98% |
| + Dr. GRPO (Stage 1) | 28.96% (+0.98pp) |
| + Gradient-balanced GRPO (Stage 2) | 31.36% (+2.40pp) |
| + Scored self-distillation (Stage 3) | **33.43%** (+2.07pp) |

> 🔍 发现：
> - 格式对齐（Stage 1）几乎不提升推理能力
> - 标准 GRPO 失败，因负梯度主导导致退化
> - **scored self-distillation 是最大增益来源**，表明“自我学习正确推理路径”极为有效

---

## 4. 关键结论和发现

### 主要发现

1. **SCA 是一种理论上 sound 且实践中高效的 attention 替代方案**
   - 具备与 self-attention 相当甚至更强的表达能力
   - 实现 $O(1)$ 推理更新，适合长序列建模
   - 与 transformer 形成结构性互补

2. **混合架构设计有效平衡效率与精度**
   - “2 SCA + 1 Transformer” 结构无需系统消融即表现出竞争力
   - SCA 承担主流上下文传播，attention 处理关键 token 匹配

3. **小模型 RL 存在特殊失败模式**
   - 当 success rate 较低时，标准 GRPO 中 negative advantage 占据主导，反而抑制已有推理结构
   - 因模型已在 SFT 阶段吸收大量 CoT 数据，错误多源于知识缺失而非推理缺陷

4. **Scored self-distillation 是简单而强大的提升手段**
   - 模型能从自身验证正确的推理链中学到更一致、更可靠的推理模式
   - 自动形成 curriculum：难样本获得更高权重

---

### 方法的局限性

- **SCA 的离散近似限制表达力**：实际只采样 $M=2$ 个 spectral points，不足以完全模拟任意 attention 分布（尤其当 $t > KM$）
- **未探索最优层比例**：2:1 的 SCA:Transformer 比例基于经验设定，缺乏系统调优
- **应用场景受限**：明确不适用于对话或代码生成，定位为轻量级推理引擎
- **依赖外部 reward model / verifier**：self-distillation 需要正确性判断信号，限制其在无标签任务上的扩展

---

### 未来工作方向

- 探索不同模型尺度下的最佳 SCA/Transformer 比例
- 研究更大 $M$（spectral samples）对表达力的影响及性价比
- 将 SCA 应用于 encoder-decoder 架构或多模态任务
- 开发无需外部 verifier 的内部一致性检测机制，推动 fully autonomous self-improvement
- 探索如何将此类高效推理模型用于大规模 synthetic population modeling 或边缘设备部署

---

> 🧠 **总体评价**：  
> Nautile-370M 展示了一条**以数学原理驱动架构创新**的道路——将 characteristic function 这一经典统计工具引入神经网络设计，并结合现代 RL 技术，在极小模型上实现了超越同级甚至更大模型的推理能力。它不仅是工程实践的成功，更是“first principles thinking”在 AI 架构设计中的典范。

</details>

---

### 14. [A Comparative Analysis on the Performance of Upper Confidence Bound Algorithms in Adaptive Deep Neural Networks](https://arxiv.org/abs/2604.24810)

**Authors**: Grigorios Papanikolaou, Ioannis Kontopoulos, Konstantinos Tserpes  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.24810v1  

#### Abstract
Edge computing environments impose strict constraints on energy consumption and latency, making the deployment of deep neural networks a significant challenge. Therefore, smart and adaptive inference strategies that dynamically balance computational cost or latency with predictive accuracy are criti...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：A Comparative Analysis on the Performance of Upper Confidence Bound Algorithms in Adaptive Deep Neural Networks

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在边缘计算（edge computing）场景中，深度神经网络（DNNs）面临严格的**能效（energy consumption）** 和 **延迟（latency）** 约束。传统的 Early-Exit DNNs 虽然通过动态提前退出（early exit）减少计算开销，但其阈值选择策略通常依赖固定的或简单的启发式规则，难以在不同输入难度下实现最优的精度-效率权衡。

现有研究大多仅采用标准的 **UCB1** 算法作为 Multi-Armed Bandit（MAB）框架中的探索-利用策略来动态选择 confidence threshold，忽略了其他更先进的 UCB 变体可能带来的性能提升。

### 🚀 提出的新方法与创新思路
本文首次系统地将四种不同的 **Upper Confidence Bound（UCB）算法** 引入到 Adaptive Deep Neural Networks（ADNNs）中用于动态阈值选择，具体包括：

- **UCB-V**：考虑奖励方差（variance-aware），对高方差臂给予更多探索。
- **UCB-Tuned**：基于方差进行调优，防止早期过度自信。
- **UCB-Bayes**：引入贝叶斯先验，建模奖励分布的不确定性（Bayesian uncertainty）。
- **UCB-BwK**：结合成本感知机制，在决策中显式考虑每条臂的计算代价。

这些算法被集成至 MAB 框架中，以在线方式为每个输入样本自适应地选择最优 confidence threshold，从而决定是否提前退出。

### 🔍 相比现有方法的优势
- **超越单一 UCB1 的局限性**：揭示了不同 UCB 策略在 accuracy-latency 和 accuracy-energy 权衡上的显著差异。
- **更精细的探索控制**：如 UCB-V 和 UCB-Tuned 利用方差信息加速收敛；UCB-Bayes 利用贝叶斯推断提高稳定性。
- **首次全面比较**：这是首个对多种 UCB 策略在 ADNNs 中性能进行全面对比分析的工作，填补了该领域的空白。

---

## 2. 核心实验方法和设置

### 📊 使用的数据集
- **CIFAR-10**：50k 训练图像 + 10k 测试图像，共 10 类。
- **CIFAR-100**：50k 训练图像 + 10k 测试图像，共 100 类。
- **CIFAR-10.1v6**：CIFAR-10 的扩展测试集，含 2k 新样本，用于评估轻微分布偏移下的鲁棒性。

### ⚙️ 实验设置
#### 模型架构
- **CNN 架构**：
  - ResNet18、ResNet34、ResNet50
- **CNN-Transformer 混合架构**：
  - xxs-MobileViT（轻量级视觉 Transformer）

#### Early Exit 设计
- 在 ResNet 中每块 residual block 后添加一个 exit branch（共 4 个出口）。
- 在 MobileViT 中每个 conv + Transformer 块后添加出口（共 3 个出口）。
- 使用 **gating network** 输出可靠性评分（reliability score），并与 confidence 结合构建 reward 函数。

#### Reward 函数设计
$$
\text{reward} = C_{\text{EEDNN}} \times (1 - C_{\text{Gating}}) - \lambda \times \text{cost}
$$
其中：
- $ C_{\text{EEDNN}} $：模型预测置信度（如 softmax 最大值）
- $ C_{\text{Gating}} $：gating network 输出的不可靠性分数
- $ \text{cost} $：退出层级索引（越深 cost 越高）
- $ \lambda $：风险控制系数，设为 0.01 / #exits

#### 基线方法对比
| 类别 | 方法 |
|------|------|
| 静态策略 | 默认 ResNet/MobileViT（无 early exit） |
| 静态阈值 | Early-Exit 版本（固定 threshold） |
| 动态策略 | UCB1（当前主流）、UCB-V、UCB-Tuned、UCB-Bayes、UCB-BwK |

#### 评估指标
- **Accuracy**：Top-1 分类准确率
- **Energy Consumption**：使用 CodeCarbon 库估算（单位：kWh × 1e⁻⁴）
- **Latency / Inference Time**：处理整个测试集所需时间（秒）
- **Cumulative Regret**：衡量算法收敛速度的关键指标
- **Pareto Frontier**：可视化 accuracy vs. energy 和 accuracy vs. latency 的最优权衡边界

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据与对比结果

#### ✅ Pareto Frontier 表现（图 2–4）
- **UCB-Tuned 和 UCB-V 显著主导 Pareto Frontier**：
  - 在 ResNet 和 MobileViT 上均实现了最佳的 **accuracy-energy** 与 **accuracy-latency** 权衡。
  - 尤其在参数量较大的 ResNet34/50 上优势更为明显。
- **UCB-Bayes 提供更高 accuracy**，但以更高的 energy 和 latency 为代价。
- **UCB-BwK 在 MobileViT 上表现优异**，因其 reward variance 较小，成本敏感策略更具优势。

#### ✅ 累积遗憾（Cumulative Regret）分析（图 5–6）
- 所有 UCB 变体均表现出 **sub-linear cumulative regret**，满足理论风险控制要求。
- 收敛速度排序（从快到慢）：
  1. **UCB-Bayes**：最快收敛，尤其在 ~4000–6000 步后 regret 增长趋缓。
  2. **UCB-Tuned / UCB-V**：次之，稳定下降。
  3. **UCB1 / UCB-BwK**：最慢，平均 regret 更高。

#### ✅ 不同模型的表现趋势
| 模型 | 最佳 UCB 策略 | 观察现象 |
|------|---------------|----------|
| ResNet 系列 | UCB-Tuned、UCB-V | 参数越多，动态阈值增益越大 |
| MobileViT | UCB-BwK、UCB-Tuned | 因结构高效，reward variance 小，成本感知更重要 |

#### ✅ 消融实验（隐含于多组设置中）
- **静态 threshold vs. 动态 MAB**：
  - 所有动态 UCB 方法在相同 accuracy 下显著降低 energy 和 latency。
- **UCB1 vs. 其他 UCB**：
  - UCB-Tuned 和 UCB-V 在 Pareto 前沿上全面优于 UCB1。
  - UCB-Bayes 收敛更快，但推理延迟较高，不适合实时性要求高的场景。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **并非所有 UCB 算法都等效**：尽管 UCB1 是当前主流，但 **UCB-Tuned 和 UCB-V 在实际部署中综合表现最优**，特别是在 accuracy-energy 和 accuracy-latency 权衡方面占据 Pareto 前沿。
2. **UCB-Bayes 收敛最快**，适合需要快速学习最优策略的场景，但由于其计算复杂度较高，导致推理延迟增加，**不适用于低延迟边缘设备**。
3. **模型结构影响最佳 UCB 选择**：
   - 对于传统 CNN（如 ResNet），**方差感知型算法（UCB-V/Tuned）更优**。
   - 对于高效混合架构（如 MobileViT），**成本感知型算法（UCB-BwK）更具竞争力**。
4. **动态阈值显著优于静态策略**：MAB 框架下的动态 threshold 选择能有效适应输入复杂度变化，实现“简单样本早退出，困难样本深计算”。

### ⚠️ 方法的局限性
- **UCB-Bayes 的计算开销较大**：由于需维护 posterior 参数更新，增加了推理时延，限制其在资源极度受限设备上的应用。
- **未考虑非平稳环境**：假设 reward 分布相对稳定，未测试概念漂移（concept drift）场景下的鲁棒性。
- **仅限于图像分类任务**：结论尚未推广至 NLP 或语音等其他模态任务。

### 🔮 未来工作方向
- 探索更多先进 UCB 变体，例如 **LinUCB**（结合 contextual features）或 **Thompson Sampling**。
- 引入神经网络驱动的 bandit agent，实现更复杂的 context-aware 决策。
- 扩展至多模态和连续学习场景，支持在线 adaptation。
- 进一步优化 UCB-Bayes 的近似推理过程，降低其部署开销。

---

## 总结一句话
> 本文首次系统比较了多种 UCB 算法在 Adaptive DNNs 中的性能，发现 **UCB-Tuned 和 UCB-V 在精度-效率权衡上表现最优，而 UCB-Bayes 收敛最快**，为边缘智能中的动态推理提供了重要的算法选型指导。

</details>

---

### 15. [QFlash: Bridging Quantization and Memory Efficiency in Vision Transformer Attention](https://arxiv.org/abs/2604.25306)

**Authors**: Sehyeon Oh, Yongin Kwon, Jemin Lee  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2604.25306v1  

#### Abstract
FlashAttention improves efficiency through tiling, but its online softmax still relies on floating-point arithmetic for numerical stability, making full quantization difficult. We identify three main obstacles to integer-only FlashAttention: (1) scale explosion during tile-wise accumulation, (2) ine...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《QFlash: Bridging Quantization and Memory Efficiency in Vision Transformer Attention》总结**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
现有的 **FlashAttention** 虽然通过 **tiling** 和 **fused kernel** 设计显著减少了 **off-chip memory** 访问，提升了计算效率，但其 **online softmax** 仍依赖 **floating-point arithmetic** 来保证数值稳定性，这导致无法实现 **fully integer-only** 的注意力计算。

此外，已有量化方法如 **I-ViT**、**QAttn**、**INT-FlashAttention** 存在以下问题：
- 仅对 **MatMul** 进行量化，**softmax** 仍在 FP32 中运行；
- 缺乏完整的 kernel fusion，无法解决内存瓶颈；
- 混合精度设计引入额外通信开销，不利于低功耗部署。

因此，如何在 **tile-based fused attention** 架构中实现 **端到端的整数量化（integer-only）**，同时保持数值稳定性和高性能，是一个尚未解决的关键挑战。

### **提出了什么新方法或新思路**
本文提出 **QFlash** —— 一种 **端到端整数量化** 的 **FlashAttention** 实现，首次实现了在 **INT8/INT32** 域内完成整个注意力计算流程（包括 softmax），并以单个 **Triton kernel** 形式运行。

#### **三大核心技术突破：**
1. **Scale Release 机制**  
   针对 tile-wise 积累过程中因整数指数近似导致的 **scale explosion** 问题，提出 **Scale Release** 策略，在每一步积累后释放 scale，避免 scale 不断累积导致溢出和精度下降。

2. **GPU 友好的 ShiftExp2 实现**  
   传统基于 shift 的指数近似需要 **integer division** 分离整数和小数部分，而 GPU 上该操作极慢。QFlash 使用 **fixed-point multiplication + shift** 替代除法，大幅提升效率。

3. **Per-tensor Quantization Granularity**  
   为支持跨 tile 的整数比较与累加，采用 **per-tensor 量化粒度**，确保所有 tile 使用统一 scale，避免 dequantization 开销，实现真正的整数融合。

---

### **相比现有方法的优势**
| 维度 | QFlash | 其他方法（如 I-ViT, INT-FlashAttention） |
|------|--------|----------------------------------------|
| **量化完整性** | ✅ 所有算子（含 softmax）均为 INT-only | ❌ softmax 仍用 FP |
| **Kernel Fusion** | ✅ 单一 Triton kernel，完全融合 | ❌ 多 kernel 或部分融合 |
| **内存效率** | ✅ 减少 off-chip memory 访问 | ⚠️ 未充分优化 |
| **硬件通用性** | ✅ 纯软件方案，适用于通用 GPU | ⚠️ 部分需专用硬件 |
| **能效比** | ✅ 更高 IMMA 利用率，更低能耗 | ⚠️ 混合精度增加指令开销 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 实验不直接在完整数据集上训练模型，而是从 **ImageNet-1K** 上预训练的 **ViT、DeiT、Swin** 模型中提取注意力层进行 **operator-level benchmarking**。
- 输入分辨率统一为 **224×224**。

### **实验设置**
- **硬件平台**：NVIDIA GeForce RTX 5090
- **软件栈**：
  - PyTorch 2.7.1 + CUDA 12.8
  - TVM 0.8
  - Triton 3.3.1
- **量化配置**：
  - **INT8** 输入（Q/K/V）
  - **INT32** 累加
  - **Per-tensor quantization**
  - 动态缩放因子（dynamic scaling）

### **评估指标**
| 指标 | 描述 |
|------|------|
| **Latency** | 单次 attention forward 的执行时间 |
| **Speedup** | 相对于基线方法的加速比 |
| **Energy Consumption** | 使用 `nvidia-smi` 测量功耗，积分得能量（μJ） |
| **Tensor Core Utilization** | IMMA / HMMA 利用率（占峰值吞吐比例） |
| **Accuracy** | Top-1 准确率（ImageNet-1K） |
| **SQNR / MSE** | 信号量化噪声比与均方误差，衡量量化保真度 |

### **基线方法对比**
| 方法 | 类型 | 是否整数 | 是否融合 | Softmax 精度 |
|------|------|----------|-----------|----------------|
| **FlashAttention-2** | Baseline | ❌ | ✅ | FP16 |
| **I-ViT** | Integer-only | ✅ | ❌ | INT-only |
| **QAttn** | Mixed-precision | ❌ | ✅ | FP32 |
| **INT-FlashAttention (Full/Half)** | Mixed | ❌ | ✅ | FP32 |
| **Torch (FP32)** | Full precision | ❌ | ❌ | FP32 |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **延迟（Latency）与加速比**
在 **7 个典型 attention workload（A1–A7）** 上测试，涵盖 **ViT/DeiT/Swin** 不同阶段：

| 模型 | Batch Size | Speedup vs I-ViT |
|------|------------|------------------|
| ViT/DeiT (A1–A3) | 1 | 最高 **4.54×** |
| ViT/DeiT (A1–A3) | 8 | 最高 **6.73×** |
| Swin (A4–A7) | 1 | 最高 **7.69×** |
| Swin (A4–A7) | 8 | 最高 **8.69×** |

> 💡 **说明**：QFlash 在 batch 较大时优势更明显，得益于更好的 kernel fusion 和 IMMA 并行利用率。

#### **Tensor Core 利用率**
| 方法 | IMMA (%) | HMMA (%) |
|------|----------|----------|
| FlashAttention-2 | – | 13.08% (A2), 4.69% (A7) |
| QFlash (Ours) | **8.00% (A2)**, **2.55% (A7)** | – |

虽然 IMMA 利用率低于 FlashAttention-2 的 HMMA，但由于 **RTX 5090 上 IMMA 峰值吞吐是 HMMA 的两倍**（348K vs 174K ops/cycle），QFlash 实际性能更高。

#### **能效表现**
在 **Workload A2, Batch=8** 下测量：
| 方法 | 能耗 (μJ) |
|------|-----------|
| FP16 FlashAttention-2 | 929.6 |
| QFlash (Ours) | **754.6** |

✅ **降低 18.8% 能耗**，主要归因于：
1. 更短的执行时间；
2. IMMA 比 HMMA 具有更高的 **performance per watt**。

#### **量化精度（SQNR/MSE）**
| 方法 | A2 (SQNR) | A7 (SQNR) |
|------|-----------|-----------|
| I-ViT | 25.80 dB | 25.22 dB |
| QFlash (Ours) | **32.50 dB** | **31.02 dB** |

✅ **SQNR 提升高达 6.7 dB**，表明 QFlash 的整数 softmax 数值稳定性优于 I-ViT。

---

### **消融实验结果（Ablation Study）**
来自附录 A 的逐步优化分析（A2, batch=1024）：

| 版本 | 改进内容 | 相对 VO 加速比 |
|------|--------|----------------|
| V0 | FP16 FlashAttention-2（Baseline） | 1.00× |
| V1 | QK MatMul → INT8 | 1.41× |
| V2 | QK + PV MatMul → INT8 | 1.46× |
| V3 | 引入整数指数近似 | 1.61× |
| V4 | 完整整数累加（QFlash） | **1.62×** |

👉 表明：**完整整数路径（含 softmax 和 accumulation）是性能提升的关键**。

---

## **4. 关键结论和发现**

### **主要发现**
1. ✅ **整数域可以高效实现 FlashAttention**：通过 **Scale Release** 和 **ShiftExp2 优化**，可在不牺牲精度的前提下完成全整数 attention。
2. ✅ **IMMA 比 HMMA 更具能效潜力**：尽管利用率较低，但更高的峰值吞吐使 INT8 在实际性能和能耗上全面胜出。
3. ✅ **kernel fusion + integer-only = 更高效率**：减少 off-chip memory 访问 + 消除浮点单元依赖，特别适合边缘设备和低功耗场景。
4. ✅ **per-tensor quantization 是可行折衷**：在 accuracy 和 efficiency 之间取得良好平衡，尤其适用于 fused kernel。

### **方法的局限性**
1. **Top-1 准确率在 Swin 上略有下降**：由于 Swin 使用 **window partitioning**，且 QFlash 采用 **per-tensor scaling**，局部统计差异被放大，导致轻微精度损失。
2. **依赖 Triton 编程模型**：目前实现基于 Triton，对开发者有一定门槛。
3. **未支持动态序列长度优化**：tile size 固定，可能影响变长输入效率。

### **未来工作方向**
1. 探索 **per-head quantization** 以进一步提升 Swin 等模型的准确率；
2. 将 QFlash 扩展至 **LLM attention** 场景；
3. 结合 **sparse attention** 或 **approximate attention** 进一步压缩计算；
4. 开发自动化的 **Triton kernel generator** 降低部署复杂度。

---

> 🔗 **代码开源地址**：https://github.com/EfficientCompLab/qflash

> 📌 **一句话总结**：  
> **QFlash 是首个实现端到端整数量化的 FlashAttention 方案，在保持 Top-1 准确率的同时，最高实现 8.69× 速度提升和 18.8% 能耗降低，为高效、低功耗 Transformer 推理提供了实用解决方案。**

</details>

---

### 16. [PhySE: A Psychological Framework for Real-Time AR-LLM Social Engineering Attacks](https://arxiv.org/abs/2604.23148)

**Authors**: Tianlong Yu, Yang Yang, Ziyi Zhou, Jiaying Xu, Siwei Li, Tong Guan, Kailong Wang, Ting Bi  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.23148v1  

#### Abstract
The emerging threat of AR-LLM-based Social Engineering (AR-LLM-SE) attacks (e.g. SEAR) poses a significant risk to real-world social interactions. In such an attack, a malicious actor uses Augmented Reality (AR) glasses to capture a target visual and vocal data. A Large Language Model (LLM) then ana...

---

### 17. [MetaGAI: A Large-Scale and High-Quality Benchmark for Generative AI Model and Data Card Generation](https://arxiv.org/abs/2604.23539)

**Authors**: Haoxuan Zhang, Ruochi Li, Yang Zhang, Zhenni Liang, Junhua Ding, Ting Xiao, Haihua Chen  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.23539v1  

#### Abstract
The rapid proliferation of Generative AI necessitates rigorous documentation standards for transparency and governance. However, manual creation of Model and Data Cards is not scalable, while automated approaches lack large-scale, high-fidelity benchmarks for systematic evaluation. We introduce Meta...

---

### 18. [From Syntax to Emotion: A Mechanistic Analysis of Emotion Inference in LLMs](https://arxiv.org/abs/2604.25866)

**Authors**: Bangzhao Shu, Arinjay Singh, Mai ElSherief  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.25866v1  

#### Abstract
Large language models (LLMs) are increasingly used in emotionally sensitive human-AI applications, yet little is known about how emotion recognition is internally represented. In this work, we investigate the internal mechanisms of emotion recognition in LLMs using sparse autoencoders (SAEs). By ana...

---

### 19. [Subspace Optimization for Efficient Federated Learning under Heterogeneous Data](https://arxiv.org/abs/2604.25467)

**Authors**: Shuchen Zhu, Zhengyang Huang, Yuqi Xu, Peijin Li  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2604.25467v1  

#### Abstract
Federated learning increasingly operates in a large-model regime where communication, memory, and computation are all scarce. Typically, non-IID client data induce drift that degrades the stability and performance of local training. Existing remedies such as SCAFFOLD introduce heterogeneity-correcti...

---

### 20. [Agentic Adversarial Rewriting Exposes Architectural Vulnerabilities in Black-Box NLP Pipelines](https://arxiv.org/abs/2604.23483)

**Authors**: Mazal Bethany, Kim-Kwang Raymond Choo, Nishant Vishwamitra, Peyman Najafirad  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.23483v1  

#### Abstract
Multi-component natural language processing (NLP) pipelines are increasingly deployed for high-stakes decisions, yet no existing adversarial method can test their robustness under realistic conditions: binary-only feedback, no gradient access, and strict query budgets. We formalize this strict black...

---

### 21. [LLM-Guided Agentic Floor Plan Parsing for Accessible Indoor Navigation of Blind and Low-Vision People](https://arxiv.org/abs/2604.23970)

**Authors**: Aydin Ayanzadeh, Tim Oates  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.23970v1  

#### Abstract
Indoor navigation remains a critical accessibility challenge for the blind and low-vision (BLV) individuals, as existing solutions rely on costly per-building infrastructure. We present an agentic framework that converts a single floor plan image into a structured, retrievable knowledge base to gene...

---

### 22. [CGU-ILALab at FoodBench-QA 2026: Comparing Traditional and LLM-based Approaches for Recipe Nutrient Estimation](https://arxiv.org/abs/2604.25774)

**Authors**: Wei-Chun Chen, Yu-Xuan Chen, I-Fang Chung, Ying-Jia Lin  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.25774v1  

#### Abstract
Accurate nutrient estimation from unstructured recipe text is an important yet challenging problem in dietary monitoring, due to ambiguous ingredient terminology and highly variable quantity expressions. We systematically evaluate models spanning a wide range of representational capacity, from lexic...

---

### 23. [Spark Policy Toolkit: Semantic Contracts and Scalable Execution for Policy Learning in Spark](https://arxiv.org/abs/2604.25061)

**Authors**: Zeyu Bai  
**Category**: cs.DC  
**Published**: 2026-04-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.25061v1  

#### Abstract
Custom policy-learning pipelines in Spark fail for two coupled systems reasons: rowwise Python execution makes inference impractical, and driver-side candidate materialization makes split search fragile at feature scale. We present Spark Policy Toolkit, a semantics-governed systems toolkit for scala...

---

### 24. [Comparative Study of Bending Analysis using Physics-Informed Neural Networks and Numerical Dynamic Deflection in Perforated nanobeam](https://arxiv.org/abs/2604.24768)

**Authors**: Ramanath Garai, Iswari Sahu, S. Chakraverty  
**Category**: cs.LG  
**Published**: 2026-04-29  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2604.24768v1  

#### Abstract
In this chapter, we investigate the bending behavior of a perforated nanobeam subjected to sinusoidal loading using an efficient and computationally robust Physics-Informed Functional Link Constrained Framework with Domain Mapping (DFL-TFC) method. Our aim is to determine the relationship between st...

---

### 25. [Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines](https://arxiv.org/abs/2604.23178)

**Authors**: Sadman Kabir Soumik  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.23178v1  

#### Abstract
LLM-as-a-Judge has become the dominant paradigm for evaluating language model outputs, yet LLM judges exhibit systematic biases that compromise evaluation reliability. We present a comprehensive empirical study comparing nine debiasing strategies across five judge models from four provider families ...

---

### 26. [Discovering Agentic Safety Specifications from 1-Bit Danger Signals](https://arxiv.org/abs/2604.23210)

**Authors**: V\'ictor Gallego  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.23210v1  

#### Abstract
Can large language model agents discover hidden safety objectives through experience alone? We introduce EPO-Safe (Experiential Prompt Optimization for Safe Agents), a framework where an LLM iteratively generates action plans, receives sparse binary danger warnings, and evolves a natural language be...

---

### 27. [Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models](https://arxiv.org/abs/2604.23460)

**Authors**: Sharan Ramjee  
**Category**: cs.AI  
**Published**: 2026-04-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.23460v1  

#### Abstract
Chain-of-Thought (CoT) reasoning has emerged as a key technique for eliciting complex reasoning in Large Language Models (LLMs). Although interpretable, its dependence on natural language limits the model's expressive bandwidth. Continuous thought models address this bottleneck by reasoning in laten...

---

### 28. [ADE: Adaptive Dictionary Embeddings -- Scaling Multi-Anchor Representations to Large Language Models](https://arxiv.org/abs/2604.24940)

**Authors**: Orhan Demirci, Sezer Aptourachman  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.24940v1  

#### Abstract
Word embeddings are fundamental to natural language processing, yet traditional approaches represent each word with a single vector, creating representational bottlenecks for polysemous words and limiting semantic expressiveness. While multi-anchor representations have shown promise by representing ...

---

### 29. [FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](https://arxiv.org/abs/2604.25135)

**Authors**: Amir Saeidi, Venkatesh Mishra, Souradeep Mukhopadhyay, Gaowen Liu, Ali Payani, Jayanth Srinivasa, Chitta Baral  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.25135v1  

#### Abstract
Large Language Models are being increasingly deployed as the decision-making core of autonomous agents capable of effecting change in external environments. Yet, in conversational benchmarks, which simulate real-world customer-centric issue resolution scenarios, these agents frequently fail due to t...

---

### 30. [Frictive Policy Optimization for LLMs: Epistemic Intervention, Risk-Sensitive Control, and Reflective Alignment](https://arxiv.org/abs/2604.25136)

**Authors**: James Pustejovsky, Nikhil Krishnaswamy  
**Category**: cs.CL  
**Published**: 2026-04-29  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2604.25136v1  

#### Abstract
We propose Frictive Policy Optimization (FPO), a framework for learning language model policies that regulate not only what to say, but when and how to intervene in order to manage epistemic and normative risk. Unlike standard alignment methods that optimize surface-level preference or task utility,...

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
