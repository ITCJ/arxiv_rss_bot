# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-16 07:02:20 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [KernelFoundry: Hardware-aware evolutionary GPU kernel optimization](https://arxiv.org/abs/2603.12440)

**Authors**: Nina Wiedemann, Quentin Leboutet, Michael Paulitsch, Diana Wofk, Benjamin Ummenhofer  
**Category**: cs.DC  
**Published**: 2026-03-16  
**Score**: 12.5  
**Type**: new  
**ArXiv ID**: 2603.12440v1  

#### Abstract
Optimizing GPU kernels presents a significantly greater challenge for large language models (LLMs) than standard code generation tasks, as it requires understanding hardware architecture, parallel optimization strategies, and performance profiling outputs. Most existing LLM-based approaches to kerne...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：KernelFoundry: Hardware-aware evolutionary GPU kernel optimization**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
优化 GPU kernel 是一项高度专业化且复杂的任务，需要深入理解硬件架构、内存层次结构和并行编程范式。尽管大型语言模型（LLMs）在代码生成方面取得了进展，但在**高性能 GPU kernel 生成**上仍面临巨大挑战：
- LLMs 缺乏对硬件细节的理解，导致生成的 kernel 性能不佳。
- 现有基于 prompt + 反馈循环的方法容易陷入“模式崩溃”（mode collapse），即反复生成相似变体，缺乏多样性探索。
- 随着迭代次数增加，“上下文污染”（context degradation）问题严重，失败历史占据提示空间，影响后续生成质量。

### **提出了什么新方法或新思路**
作者提出 **KernelFoundry**，一个面向 GPU kernel 优化的**进化式框架**，结合 LLM 与进化算法，具备以下三大核心机制：

#### **(1) MAP-Elites 质量-多样性搜索（Quality-Diversity Search）**
- 引入 **MAP-Elites** 算法，将 kernel 设计空间划分为多个行为单元（behavioral cells），每个单元独立维护最优解。
- 定义了三个**领域特定的行为维度**来刻画 kernel 特征：
  - `dmem`：内存访问模式（从标量访问到多级缓存优化）
  - `dalgo`：算法结构（从直接翻译到融合/重构算法）
  - `dsync`：并行协调机制（从无同步到全局原子操作）
- 这种结构化划分有效防止了模式崩溃，确保在不同优化路径上的持续探索。

#### **(2) 元提示进化（Meta-Prompt Evolution）**
- 提示本身也被视为可进化的对象，通过一个专用的 **meta-prompter LLM** 动态调整提示内容。
- 维护一个独立的“提示档案库”，记录哪些指导策略更有效。
- 支持四个可演化区域：
  - 优化哲学（如优先考虑带宽利用率）
  - 优化策略（具体技术清单）
  - 常见陷阱（避免 bank conflict 等反模式）
  - 分析引导（预编码推理模板）
- 显著缓解了上下文污染问题，并实现任务特定优化策略的自动发现。

#### **(3) 模板化参数优化（Template-based Parameter Optimization）**
- 将算法设计与硬件参数调优分离。
- 引导 LLM 生成**模板化 kernel**（templated kernel），其中 block size、tile size 等作为模板参数。
- 框架自动枚举参数组合并进行独立评测，选出最佳配置。
- 实现对硬件依赖性强的超参的系统性搜索。

### **相比现有方法的优势**
| 方面 | 传统方法 | KernelFoundry |
|------|--------|-------------|
| 探索多样性 | 低（易陷局部最优） | 高（MAP-Elites 结构保障） |
| 上下文管理 | 固定提示，易污染 | 动态进化提示，抗污染 |
| 参数调优 | 依赖 LLM 直接猜测 | 显式模板 + 自动搜索 |
| 跨平台支持 | 多为 CUDA 专有 | 支持 SYCL（跨厂商）、CUDA、Triton |
| 用户灵活性 | 限于标准 benchmark | 支持自定义任务与复杂测试框架 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- **KernelBench** [Ouyang et al., 2025b]：主流 LLM kernel 生成基准，包含 250 个任务（单算子、融合模式、完整架构）。
- **robust-kbench** [Lange et al., 2025b]：强调鲁棒性验证的任务集，包含前向/反向传播操作。
- **自定义任务**：用于展示实际应用场景，例如优化 Llama3 中的 rotary positional embedding。

> 注：作者对 KernelBench 进行了过滤，排除存在“奖励欺骗”风险或基线效率过低的任务，最终使用 **111 个任务**（含代表性子集 40 个）。

### **实验设置和评估指标**

#### **编程语言与硬件平台**
- 主要目标语言：**SYCL**（开放标准，支持 Intel/NVIDIA/AMD）
- 对比语言：**CUDA**
- 硬件环境：
  - Intel Arc B580（离散 GPU）
  - Intel Arc 140V（集成 GPU，简称 LNL）
  - NVIDIA RTX A6000（Ampere 架构）

#### **评估指标**
| 指标 | 定义 |
|------|------|
| **Correctness Rate** | 正确编译且数值正确的 kernel 比例 |
| **Fast@p** | 加速比超过 p 的任务占比（如 Fast@1 表示加速 >1x） |
| **Average Speedup** | 相对于 PyTorch Eager 执行时间的平均加速比 |
| **Geometric Mean Speedup** | 更稳健的平均加速度量方式 |
| **Hardware-Speedup (hws)** | 在目标硬件上优化的 kernel vs. 在其他硬件上优化的 kernel 的加速比，用于衡量硬件感知能力 |

#### **基线方法对比**
- **AI CUDA Engineer** [Lange et al., 2025a]：基于进化的多智能体框架
- **robust-kbench** [Lange et al., 2025b]：强调验证鲁棒性的方法
- **Kernelsseum**：KernelBench 官方榜单结果
- **OpenEvolve**：开源版 AlphaEvolve，仅用通用进化策略，无领域定制

> 所有对比均在相同硬件（NVIDIA A6000）上重测以保证公平性。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **在 CUDA 上的表现（vs. 基线）**
| 方法 | Avg Speedup (L1) | Avg Speedup (L2) | Fast@2 (L2) |
|------|------------------|------------------|------------|
| AI CUDA Engineer (re-eval) | 1.005 | 1.606 | 25% |
| **Ours** | **1.241** | **2.104** | **45%** |

> ➤ 在 L2 任务上平均加速达 **2.1×**，显著优于基线（+31%），且 Fast@2 提升明显。

#### ✅ **在 SYCL 上的表现（首次系统性评估）**
| 方法 | Dataset | Correct Rate | Avg Speedup | Fast@2 |
|------|-------|--------------|-------------|--------|
| **Ours** | KernelBench (111 tasks) | **97%** | **2.32×** | 42% |
| robust-kbench (CUDA) | — | — | 1.49× | — |
| **Ours + param opt** | KernelBench (40 iters) | 1.0 | **2.732×** | 45% |

> ➤ 首次证明 LLM 可高效生成高性能 **SYCL kernel**，平均加速 **2.32×**，远超已有 CUDA 方法。

#### ✅ **与 OpenEvolve 对比（消融意义）**
| 方法 | Iterations | Avg Speedup (L2) |
|------|-----------|------------------|
| OpenEvolve | 10 | 1.483 |
| **Ours** | **10** | **2.059** |
| OpenEvolve | 40 | 2.535 |
| **Ours + param opt** | **40** | **2.732** |

> ➤ **仅 10 轮迭代即超越 OpenEvolve 40 轮表现**，说明领域定制机制极大提升了收敛速度。

#### ✅ **硬件感知能力验证（交叉测试）**
进行“交叉优化”实验：分别在 B580 和 LNL 上运行 KernelFoundry，然后互换测试。

| 测试场景 | hws > 1 的比例 | 平均 hws |
|---------|---------------|----------|
| LNL 上优化的 kernel 在 LNL 测试 | **70%** | **1.537×** |
| B580 上优化的 kernel 在 B580 测试 | **70%** | **1.109×** |

> ➤ 明确表明 KernelFoundry 能生成**针对特定硬件优化的 kernel**，而非通用方案。

#### ✅ **真实案例：Llama3 rotary embedding 加速**
- 任务：优化 Llama3.2 1B 模型中的 `rotary positional embedding`
- 结果：10 轮内找到正确 kernel，实现 **7.9× 加速**
- 影响：端到端前向推理时间减少 **8%**（0.413s → 0.38s）

---

## **4. 关键结论和发现**

### **主要发现**
1. **领域定制的进化搜索优于通用 prompt 循环**  
   MAP-Elites 结合 domain-specific behavioral descriptors 能有效维持多样性，避免早熟收敛。

2. **提示也可以被进化**  
   Meta-prompt evolution 成功解决了上下文污染问题，并实现了优化知识的积累与迁移。

3. **SYCL 是可行且有前景的跨平台选择**  
   尽管 LLM 对 SYCL 不熟悉，KernelFoundry 仍能生成高性能 kernel，推动真正可移植的 GPU 编程。

4. **硬件感知是可学习的**  
   框架能在不同 GPU 上学到不同的优化策略，生成针对性强的 kernel。

5. **适用于真实世界任务**  
   支持自定义输入格式与复杂测试流程，已在 Llama3 等实际模型中成功应用。

### **方法的局限性**
- **依赖高质量 LLM**：使用 GPT-OSS 20B 时正确率仅为 65%，显示对模型能力敏感。
- **计算成本高**：需多次编译、执行、分析，适合离线优化而非实时生成。
- **行为维度设计依赖专家知识**：当前 `dmem/dalgo/dsync` 需人工定义，尚未完全自动化。
- **未解决形式化验证问题**：仍可能存在 reward hacking，需更强验证机制。

### **未来工作方向**
- 扩展模板化 kernel 以适应多种输入形状和张量大小。
- 引入形式化验证工具链，彻底杜绝错误 kernel。
- 深化硬件特异性建模，如集成 microarchitecture simulator。
- 探索基于 RL 的 fine-tuning，结合可验证奖励信号训练专用模型。
- 构建公开的 kernel 数据库与 benchmark，促进社区协作与复现。

---

> 🔚 **总结一句话**：  
> **KernelFoundry 通过“行为空间划分 + 提示进化 + 参数模板”的三重机制，实现了高效、多样、硬件感知的 GPU kernel 自动生成，在 SYCL 和 CUDA 上均显著超越现有方法，为自动化高性能计算开辟了新路径。**

</details>

---

### 2. [TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition](https://arxiv.org/abs/2603.12465)

**Authors**: Prabhu Vellaisamy, Shreesh Tripathi, Vignesh Natarajan, Surya Santhan Thenarasu, Shawn Blanton, John P. Shen  
**Category**: cs.DC  
**Published**: 2026-03-16  
**Score**: 12.0  
**Type**: new  
**ArXiv ID**: 2603.12465v1  

#### Abstract
Large Language Model (LLM) inference is widely used in interactive assistants and agentic systems. In latency-sensitive deployments, inference time can become dominated by host-side overheads. Existing approaches typically expose this cost only as an aggregate residual or a launch/queue metric, whic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition**

---

## **1. 论文的主要贡献和创新点**

### **解决了什么问题**
在大语言模型（LLM）推理中，尤其是在延迟敏感的应用（如对话系统、Agent系统）中，**host-side（CPU端）的调度开销**（orchestration overhead）已成为影响端到端延迟的关键瓶颈。然而，现有的分析工具通常将这些开销表示为一个聚合残差（aggregate residual）或仅关注 kernel launch/queue 时间（如 TKLQT），无法精确识别开销来自哪一层软件栈（框架、CUDA库、驱动等）。这导致优化方向不明确。

此外，对于 **Mixture-of-Experts (MoE)** 这类动态性强、kernel 数量多的模型，host-side 开销更为显著，但传统指标难以揭示其根本原因。

### **提出了什么新方法或新思路**
本文提出 **TaxBreak** —— 一种基于 trace 驱动的 **host-side 推理开销分解方法**，将总开销细分为三个正交且互斥的组件：

1. **△FT (Framework Translation)**  
   - 包括 Python 调度开销（`T_py`）和不可约减的 ATen 派发成本（`T_dispatch_base`）
   - 反映框架层（如 PyTorch）的处理时间

2. **△CT (CUDA-Library Translation)**  
   - 指通过 vendor library（如 cuBLAS、cuDNN）时产生的额外前端处理时间
   - 仅对 library-mediated kernel 收费

3. **△KT (Kernel Launch Path)**  
   - 从 `cudaLaunchKernel` 到 GPU 实际启动之间的硬件级延迟（`T_floor_sys`）
   - 通过空 kernel profiling 测量，作为不可再压缩的底层开销

同时，作者提出了一个新的诊断指标：**Host-Device Balance Index (HDBI)**，定义为：

$$
\text{HDBI} = \frac{T_{\text{DeviceActive}}}{T_{\text{DeviceActive}} + T_{\text{Orchestration}}}
$$

- HDBI → 1：设备受限（device-bound）
- HDBI → 0：主机受限（host-bound）

该指数帮助判断优化应聚焦于软件栈还是设备侧计算。

### **相比现有方法的优势**
| 对比维度 | 以往方法（如 Framework Tax, TKLQT） | TaxBreak |
|--------|-------------------------------|--------|
| 开销粒度 | 聚合残差或仅 launch-path | 分解至 framework / library / launch 三层 |
| 诊断能力 | 仅能判断 host/device-bound | 可定位具体瓶颈层 |
| 适用场景 | 多为静态 dense 模型 | 支持 MoE、动态 eager 模式 |
| 指标设计 | 单一指标（如 TKLQT） | 引入 HDBI + 分解项联合诊断 |

> ✅ **优势总结**：TaxBreak 提供了**机制级归因**（mechanism-level attribution），使开发者能够精准决定是优化运行时编译、减少 kernel 数量，还是改进调度路径。

---

## **2. 核心实验方法和设置**

### **使用的模型与工作负载**
- **Dense 模型**：
  - Llama-3.2-1B / -3B
  - GPT-2 (124M)
- **MoE 模型**：
  - OLMoE-1B/7B
  - Qwen1.5-MoE-A2.7B
- 数据格式：BFloat16
- 执行模式：默认使用 **eager mode**（更贴近真实动态场景）

### **实验平台**
- **H100 平台**：
  - CPU: Intel Xeon 8480C (56 cores @ 2.0/3.8 GHz)
  - GPU: NVIDIA H100 (80GB)，DGX H100 系统
- **H200 平台**：
  - CPU: Intel Xeon Gold 6538Y+ (32 cores)
  - GPU: NVIDIA H200 NVL (141GB)

> ⚠️ 注意：H200 GPU 主频更低（-9.9%），但 CPU 单核性能更强，用于验证 CPU 性能的影响。

### **评估指标**
- **端到端延迟**（End-to-end latency）：TTFT（Time to First Token）、TPOT（Time Per Output Token）
- **GPU 利用率**（GPU utilization）
- **idle fraction**：$(T_{e2e} - T_{\text{DeviceActive}})/T_{e2e}$
- **HDBI**：主客观平衡指数
- **kernel 发射统计**：总数、唯一名称数、每 token kernel 数

### **基线对比**
- **Framework Tax [14]**：仅提供 aggregate host-side 残差
- **TKLQT [30]**：只测量 kernel launch 和 queue 时间
- **FlashAttention-2**：作为优化案例进行对比分析

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### **(1) MoE 模型的 kernel 爆炸现象**
在 decode 阶段（BS=4, SL=2048, m=10）：

| 模型 | 总 kernel 数 | 每 token kernel 数 | 是 Dense 模型的倍数 |
|------|-------------|---------------------|--------------------|
| Llama-3.2-1B | 8,475 | 847.5 | 1× |
| OLMoE-1B/7B | 93,053 | 9,305.3 | **11×** |
| Qwen1.5-MoE-A2.7B | 66,951 | 6,695.1 | **8×** |

> 🔥 **结论**：MoE 模型由于路由和专家调度，产生大量小而频繁的 kernel，导致严重 host-bound。

#### **(2) GPU 利用率与 idle fraction**
- **Llama-3.2-3B**（dense）：
  - Prefill 时 GPU 利用率达 67.6%，idle fraction < 1%（大 batch 下）
  - Decode 时 idle fraction 上升至 ~78%（BS=1），但可通过 batching 缓解
- **Qwen1.5-MoE-A2.7B**（MoE）：
  - Decode 时 idle fraction 高达 **81.5% → 73.3%**（即使 BS=16）
  - GPU 利用率仅 **27.7%**，远低于 dense 模型

#### **(3) HDBI 结果**
- **Dense 模型**：
  - Prefill：HDBI ≈ 0.37–0.41（较平衡）
  - Small decode：HDBI ≈ 0.23–0.24（host-visible）
  - Large decode：随 batch/context 增大，HDBI 回升至 >0.9（device-dominant）
- **MoE 模型**：
  - Prefill：HDBI ≈ 0.15
  - Decode：HDBI 持续低于 **0.1–0.15**，始终处于 **host-bound** 状态

#### **(4) TaxBreak 分解结果**
- **GEMM kernels**（如 cuBLAS）有较高 △CT（+1.7–1.88 μs）
- 多数 elementwise/reduction kernels 接近 `T_floor_sys`（~4.7 μs）
- **△KT_fw**（框架 enqueue 开销）在 GEMM 中可达 1.18–1.88 μs，表明存在优化空间

#### **(5) CPU 单线程性能影响（H100 vs H200）**
尽管 H200 GPU 主频更低（-9.9%），但由于 CPU 更快（Emerald Rapids），**host-side 开销降低 10–29%**：

| 场景 | T_orchestration 下降 | 端到端延迟改善 |
|------|------------------------|----------------|
| Llama-3.2-1B Decode | ↓29% | ↑14% |
| Qwen1.5-MoE-A2.7B | ↓13–14%（end-to-end） | 尽管 T_device_active 略高仍更快 |

> 💡 **关键洞察**：对于 host-bound 工作负载（尤其是 MoE decode），**CPU 单核性能是 first-order 参数**。

#### **(6) FlashAttention-2 对比**
- 在长序列（BS=8, SL=2048）下：
  - 端到端延迟下降 **68.6%**
  - T_device_active 大幅下降（内存访问减少）
  - T_orchestration 仅下降 24%
  - HDBI 从 0.96 → 0.90（变得更 host-visible）

> ✅ 表明 FA2 是典型的 **device-side 优化**，而非降低 host 开销。

---

## **4. 关键结论和发现**

### **主要发现**
1. **aggregate latency、GPU idle 或 boundedness ratio 单独使用会误导优化方向**。例如：
   - 高 idle fraction 可能源于 host 调度、device 依赖链或内存停顿，需进一步归因。
2. **MoE 模型在 decode 阶段持续 host-bound**，即使增大 batch size 也无法缓解，因其每 token 的 kernel 数量是 dense 模型的 **8–11×**。
3. **TaxBreak 能准确区分优化目标**：
   - 若 △FT + △CT 主导 → 应优化 framework 或 library dispatch（如使用 `torch.compile`）
   - 若 $N \cdot T_{\text{floor\_sys}}$ 主导 → 应融合 kernel 或使用 CUDA Graphs
   - 若 △KT_fw 显著 → 应优化 driver/runtime 路径
4. **CPU 单线程性能对 host-bound LLM 推理至关重要**：
   - 快速 CPU 可使 T_orchestration 降低 **10–29%**
   - 即使搭配更慢的 GPU，也能实现 **最高 14% 的端到端加速**

### **方法的局限性**
- 当前仅支持 **NVIDIA GPU + CUDA 生态**，依赖 nsys tracing 接口
- 对高度动态、autotuned 或同步密集型 kernel 的 replay attribution 可能不完全准确
- HDBI 是诊断性指标，不能直接作为优化目标，需结合绝对延迟解读

### **未来工作方向**
- 扩展至更多 AI 工作负载（如 vision models、retrieval-augmented generation）
- 支持跨平台（AMD MI300X、NVIDIA GB200/GB300）
- 构建自动化优化建议引擎，基于 TaxBreak 输出推荐 kernel fusion、compilation 策略

---

> 📌 **一句话总结**：  
> **TaxBreak 揭示了 LLM 推理中被忽视的 host-side 开销结构，证明在 MoE 和小批量 decode 场景下，CPU 单核性能和软件栈效率可能比 GPU 算力更重要**。

</details>

---

### 3. [Efficient and Interpretable Multi-Agent LLM Routing via Ant Colony Optimization](https://arxiv.org/abs/2603.12933)

**Authors**: Xudong Wang, Chaoning Zhang, Jiaquan Zhang, Chenghao Li, Qigan Sun, Sung-Ho Bae, Peng Wang, Ning Xie, Jie Zou, Yang Yang, Hengtao Shen  
**Category**: cs.AI  
**Published**: 2026-03-16  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.12933v1  

#### Abstract
Large Language Model (LLM)-driven Multi-Agent Systems (MAS) have demonstrated strong capability in complex reasoning and tool use, and heterogeneous agent pools further broaden the quality--cost trade-off space. Despite these advances, real-world deployment is often constrained by high inference cos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient and Interpretable Multi-Agent LLM Routing via Ant Colony Optimization

---

## 1. 论文的主要贡献和创新点

### 解决的问题
当前基于 **Large Language Model (LLM)** 的 **Multi-Agent Systems (MAS)** 在实际部署中面临三大挑战：
- **高推理成本**（inference cost）和 **延迟**（latency）
- **路由策略缺乏透明度**，难以在医疗、金融等高风险领域建立信任
- 现有路由机制（如基于LLM的selector或静态规则）对动态负载和混合意图适应能力差，导致资源利用效率低、性能不稳定

### 提出的新方法：AMRO-S
作者提出 **AMRO-S**（Ant Colony Optimization for Multi-Agent LLM Routing with Supervision），一种高效且可解释的多智能体路由框架，其核心思想是将 MAS 路由建模为一个**语义条件下的路径选择问题**，并引入受蚁群优化（ACO）启发的机制。

#### 三大创新机制：
1. **SFT增强的小型语言模型（SLM）语义路由器**
   - 使用经过监督微调（SFT）的轻量级SLM（如 Llama-3.2-1B-Instruct）进行查询意图识别
   - 输出任务混合分布 $ w(q) $，作为后续路由决策的“语义锚点”
   - 显著降低路由开销，同时实现对混合意图的细粒度感知

2. **任务特定的信息素专家（Task-specific Pheromone Specialists）**
   - 将全局信息素矩阵分解为多个任务专用的 $ \tau_t $ 矩阵（如数学、代码、通用推理）
   - 在推理时通过 $ \tau(q) = \sum w_t(q)\cdot\tau_t $ 进行加权融合
   - 有效减少跨任务干扰，提升混合负载下的路径选择质量

3. **质量门控的异步更新机制（Quality-Gated Asynchronous Update）**
   - 推理与学习解耦：在线服务路径不进行实时更新
   - 异步地从缓冲区采样请求，由轻量级 **LLM-Judge** 判断输出质量（$ g \in \{0,1\} $）
   - 仅对高质量轨迹（$ g=1 $）进行信息素强化，避免噪声污染
   - 实现无延迟增长的持续优化

### 相比现有方法的优势
| 维度 | AMRO-S优势 |
|------|-----------|
| **效率** | 路由开销极低（SLM + 非实时更新），支持高并发 |
| **准确性** | 平均得分显著高于最强基线（+1.9 pts） |
| **成本效益** | 更优的质量-成本权衡，token 和 latency 更低 |
| **可解释性** | 信息素模式提供可追踪的路由证据 |
| **稳定性** | 高并发下仍保持准确率稳定 |

---

## 2. 核心实验方法和设置

### 数据集
在五个公开基准上进行全面评估：
- **GSM8K**：小学数学应用题（Exact Match）
- **MATH**：数学竞赛题（Exact Match）
- **MMLU**：涵盖57个领域的知识问答（Accuracy）
- **HumanEval**：代码生成（Pass@1）
- **MBPP**：Python编程任务（Pass@1）

### 实验设置
- **Agent Pool**：异构LLM池，包含 `gpt-4o-mini`, `gemini-1.5-flash`, `claude-3.5-haiku`, `llama-3.1-70b`
- **Semantic Router Backbone**：`Llama-3.2-1B-Instruct` 和 `Qwen2.5-1.5B`（经SFT训练）
- **统一推理预算**：限制最大交互轮次和总调用次数，确保公平比较
- **成本核算**：基于官方API定价计算 token 成本
- **硬件平台**：单张 NVIDIA A100 (80GB)

### 评估指标
- 主要指标：**Pass@1 / Accuracy**
- 辅助指标：**平均得分（Avg.）**、**推理成本（Cost）**、**端到端延迟（Latency）**、**吞吐量（Speedup）**

### 基线方法对比
| 类型 | 方法 |
|------|------|
| 单模型基线 | GPT-4o, Claude-3.5-Sonnet |
| 单模型+推理链 | CoT, ToT, GoT, AoT |
| 多智能体系统（无路由） | LLM-Debate, GPTSwarm, AFlow |
| 动态路由方法 | RouteLLM, RouterDC, MasRouter |

此外还测试了 AMRO-S 在 **MacNet**, **GPTSwarm**, **HEnRY** 等主流MAS框架中的即插即用能力。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（Table I）
| 方法 | Avg. Score |
|------|----------|
| Vanilla (GPT-4o) | 87.76 |
| MasRouter (SOTA baseline) | 85.93 |
| **AMRO-S (Ours)** | **87.83** ✅ |

- 在 **MATH** 上从 75.42 → **78.15**
- 在 **MBPP** 上从 84.0 → **86.3**
- 在 **HumanEval** 上达到 **92.2**

👉 表明 AMRO-S 能更精准匹配任务语义与模型能力，在复杂推理和编码任务上提升显著。

### 与基线方法对比结果
- **相比 MasRouter**：平均分高出 **1.9 分**
- **相比静态路由或多智能体协作**：避免因能力错配导致的性能下降
- **在成本控制方面**：在 Table II 中显示，AMRO-S 在所有框架中均以**最低成本获得最高准确率**
  - 例如在 MacNet + GSM8K 上，成本从 \$2.14 降至 \$2.00

### 高并发压力测试（Table V）
| 并发数 | AMRO-S 时间(s) | Speedup | 准确率 |
|--------|------------------|---------|--------|
| 20     | 3849.60          | 1.0×    | 96.10% |
| 1000   | **823.21**       | **4.7×**| **96.40%** |

- **AMRO-S** 实现 **4.7倍加速**，且准确率**稳定甚至略有上升**
- 对比基线 **Weighted Round-Robin (WRR)**：准确率从 96.00% 下降到 88.20%，表明其无法维持语义感知路由

### 消融实验结果（Table III）
| 设置 | Avg. Score |
|------|------------|
| Random Routing | 79.64 |
| w/o SFT (Llama-3.2-1B) | 83.42 |
| w/o SFT (GPT-4o-mini) | 86.48 |
| w/ SFT (Qwen2.5-1.5B) | 87.63 |
| **AMRO-S (SFT + Llama-3.2-1B)** | **87.83** |

👉 结论：
- 多智能体协作本身不足以带来稳定增益
- **SFT 显著提升小型路由器的意图识别精度**（见 Table IV，SFT后达 97.9%）
- “SFT + 信息素专家”组合是性能跃升的关键

---

## 4. 关键结论和发现

### 主要发现
1. **语义感知路由至关重要**：仅靠多智能体协作无法保证性能提升，必须结合任务意图理解。
2. **信息素分解机制有效隔离任务记忆**：任务特定的信息素专家能防止历史经验污染，提升混合负载鲁棒性。
3. **异步质量门控更新保障稳定性**：避免低质量样本误导学习过程，实现安全的在线进化。
4. **AMRO-S 是即插即用的通用路由层**：可在不修改原有架构的前提下集成进多种MAS框架，并持续提效降本。
5. **信息素图谱具有强可解释性**（Figure 3）：
   - 数学任务偏好早期分解、后期精确计算的路径
   - 编码任务聚焦于最终执行阶段的可靠模型
   - 通用任务则呈现均衡分布

### 方法的局限性
- 当前任务类别需预先定义，尚未完全支持开放域动态任务发现
- 信息素初始化依赖离线暖启动，冷启动场景可能需要额外设计
- LLM-Judge 虽轻量但仍引入一定计算负担，极端边缘场景需进一步简化

### 未来工作方向
- 扩展至动态任务空间，支持自动任务聚类与信息素自动生成
- 探索联邦式信息素共享机制，支持跨系统协同学习
- 结合强化学习进一步优化探索-利用平衡
- 在真实工业场景（如客服、运维自动化）中验证长期运行效果

---

> **总结一句话**：  
> AMRO-S 通过 **SLM语义路由 + 信息素专家 + 质量门控异步更新**，实现了**高效、低成本、高准确率且可解释**的多智能体LLM路由，在多项基准和高并发场景中全面超越现有方法。

</details>

---

### 4. [ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](https://arxiv.org/abs/2603.13115)

**Authors**: Jie Ji, Gen Li, Kaiyuan Deng, Fatemeh Afghah, Xiaolong Ma  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.13115v1  

#### Abstract
Deep learning models, despite their impressive achievements, suffer from high computational costs and memory requirements, limiting their usability in resource-constrained environments. Sparse neural networks significantly alleviate these constraints by dramatically reducing parameter count and comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training —— 核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **sparse training** 在高稀疏度下存在以下关键挑战：
- **梯度信号混乱且噪声大**：由于大量参数被剪枝，剩余参数承担过重学习负担，导致梯度方差增大、训练不稳定。
- **SAM（Sharpness-Aware Minimization）虽能提升泛化能力，但计算开销过大**：标准 SAM 需要两次反向传播（backpropagation），使计算成本翻倍，在资源受限场景中难以应用。

这些问题在边缘设备、移动端等 **resource-constrained environments** 中尤为突出。

---

### 🚀 提出的新方法：ZO-SAM
作者提出 **Zero-Order Sharpness-Aware Minimization (ZO-SAM)**，一种新型优化框架，将 **Zero-Order (ZO) Optimization** 与 **SAM** 创新性结合。

#### 核心思想：
- 在 SAM 的 **perturbation step**（扰动步骤）中使用 **zero-order gradient estimation**（零阶梯度估计），避免一次完整的 backpropagation。
- 在后续的 **gradient update step** 中仍保留 **first-order gradient**（一阶梯度），确保更新精度和稳定性。

> 💡 这是一种“混合策略”：用 ZO 加速扰动，用 FGSM-style 精确更新。

#### 使用的技术细节：
- 采用 **Random Gradient Estimation (RGE)** 而非 Coordinate-wise Gradient Estimation (CGE)，因为：
  - CGE 需 $ d $ 次前向传播（$ d $ 为参数维度），代价极高；
  - RGE 只需 $ m \ll d $ 次函数评估，显著降低计算开销。

---

### 🔍 相比现有方法的优势
| 维度 | 优势说明 |
|------|----------|
| **计算效率** | 相比传统 SAM 减少 50% backpropagation 开销，仅需单次反向传播，速度接近 SGD。 |
| **收敛稳定性** | 显著降低梯度方差（见 Figure 1），尤其在 90%~98% 高稀疏度下表现更优。 |
| **泛化性能** | 引导模型进入更平坦的极小值（flat minima），提高测试准确率与鲁棒性。 |
| **兼容性强** | 可作为插件式模块集成到多种主流 sparse training 方法中（如 LTH, SNIP, RigL, MEST 等）。 |
| **部署友好** | 特别适用于边缘计算、低功耗设备等对算力敏感的应用场景。 |

---

## 2. 核心实验方法和设置

### 📚 数据集
- **图像分类任务**：
  - **CIFAR-10 / CIFAR-100**：用于主实验验证。
  - **ImageNet-1K**：用于大规模 Transformer 架构验证。
- **鲁棒性测试集**：
  - **CIFAR-10-C**：包含 19 种常见 corruption（模糊、噪声、天气变化等），共 5 个严重等级，用于评估分布偏移下的鲁棒性。

---

### ⚙️ 实验设置
| 项目 | 设置详情 |
|------|---------|
| **模型架构** | ResNet-32, ResNet-50, WideResNet-28-10, DeiT-Tiny/Small（Vision Transformer） |
| **稀疏度水平** | 90%, 95%, 98%（静态与动态稀疏训练均覆盖） |
| **评估指标** |  
| - 主要指标 | 测试准确率（Test Accuracy %）  
| - 效率指标 | 每秒处理图像数（images/sec）、epoch 数（收敛速度）  
| - 鲁棒性指标 | Clean Accuracy vs. Corrupted Accuracy 差值 $ \Delta = \text{Clean} - \text{CIFAR-10-C} $，越小越好  
| - 可视化 | 损失曲面可视化（loss surface）、feature map 对比 |

---

### 🆚 基线方法对比
#### （1）Sparse Training 方法（基础算法）
- **Static Pruning**: LTH (Lottery Ticket Hypothesis), SNIP, GraSP, SynFlow
- **Dynamic Sparse Training (DST)**: SET, DSR, RigL, MEST, SViTE

#### （2）SAM 家族优化器对比
- **SAM** [13]
- **ESAM** [9]: 选择性扰动子集以减少计算
- **LookSAM (LS)** [25]: 利用时间一致性周期性复用扰动方向
- **GSAM** [48]: 添加正交上升步以缩小代理间隙

> 所有方法均在同一设置下进行公平比较，并报告均值 ± 标准差（3 次重复实验）。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

#### ✅ 表格 1：ResNet-32 on CIFAR-10/100 @ 不同稀疏度（节选代表性结果）

| 方法 | Sparsity | CIFAR-10 Acc (%) | Gain vs Base | CIFAR-100 Acc (%) | Gain vs Base |
|------|----------|------------------|---------------|--------------------|---------------|
| SNIP | 90% | 92.59 | — | 68.89 | — |
| SNIP + ZO-SAM | 90% | **93.38** | ↑0.79 | **69.54** | ↑0.65 |
| RigL | 95% | 91.83 | — | 68.22 | — |
| RigL + ZO-SAM | 95% | **92.21** | ↑0.38 | **70.58** | ↑2.36 |
| MEST | 98% | 89.22 | — | 70.44 | — |
| MEST + ZO-SAM | 98% | **91.53** | ↑2.31 | **72.20** | ↑1.76 |

> ✅ **ZO-SAM 在所有稀疏度下均带来稳定增益，最高达 +2.31%（CIFAR-10）和 +2.54%（CIFAR-100）**

---

#### ✅ 表格 2：Transformer on ImageNet-1K

| 方法 | Model | Sparsity | Accuracy (%) | Gain |
|------|-------|----------|--------------|------|
| RigL | DeiT-Tiny | 50% | 70.79 | — |
| RigL + ZO-SAM | DeiT-Tiny | 50% | **71.32** | ↑1.14 |
| MEST | DeiT-Small | 70% | 78.04 | — |
| MEST + ZO-SAM | DeiT-Small | 70% | **79.16** | ↑1.17 |

> ✅ 即使在 Vision Transformers 上也能有效提升性能。

---

#### ✅ 收敛速度对比（Table 3 & Figures 5–6）
| 方法 | Sparsity | Epochs to Reach 90% Acc (CIFAR-10) |
|------|----------|-------------------------------|
| SGD | 90% | 104 |
| SAM | 90% | 84 |
| ZO-SAM | 90% | **70** ✅ |
| SGD | 95% | 131 |
| ZO-SAM | 95% | **88** ✅ |

> 🔥 ZO-SAM 收敛最快，且仅用单次 backpropagation，效率远超其他 SAM 变体。

---

#### ✅ 计算效率对比（Table 4）

| 方法 | ResNet-32 (img/sec) | WRN-28-10 (img/sec) | 相对 SGD (%) |
|------|---------------------|----------------------|----------------|
| SGD | 5673.95 | 752.30 | 100% |
| SAM | 2704.84 | 354.94 | ~47.7% |
| ZO-SAM | **4349.53** | **576.01** | **76.7%** ✅ |

> ⚡ ZO-SAM 的吞吐量是 SAM 的约 **1.6 倍**，接近原始 SGD 的 77%，远优于其他高效 SAM 方法。

---

#### ✅ 鲁棒性提升（Table 5: CIFAR-10-C）

| 方法 | Clean Acc (%) | CIFAR-10-C Acc (%) | Δ = Clean − Corrupted |
|------|----------------|----------------------|------------------------|
| SNIP | 92.59 | 59.60 | 32.99 |
| SNIP + ZO-SAM | 93.38 | **62.70** | **30.68** ✅ |
| 提升幅度 | — | ↑3.10 | ↓2.31 |

> 🛡️ ZO-SAM 显著增强模型对分布偏移的鲁棒性，误差下降超过 2.3 个百分点。

---

#### ✅ 消融实验与可视化支持
- **Loss Surface Visualization (Figure 4)**：ZO-SAM 得到更宽、更平滑的损失盆地，表明其成功引导至 flat minima。
- **Feature Map Analysis (Figure 7)**：ZO-SAM 的激活图更清晰、聚焦于物体边界与语义区域，表示内部表征更稳定。
- **Gradient Variance Plot (Figure 1a)**：ZO-SAM 的梯度波动明显低于 SGD，尤其在高稀疏度下优势显著。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **稀疏训练中的核心瓶颈是梯度噪声而非单纯参数减少**，而 SAM 类方法可通过寻找 flat minima 来缓解该问题。
2. **传统 SAM 因双 backpropagation 成本过高，难以用于稀疏训练**，成为其落地障碍。
3. **ZO-SAM 成功打破“性能 vs. 效率”的权衡困境**：
   - 通过引入 zero-order estimation 于 perturbation step，
   - 实现与 SAM 相当的泛化能力，
   - 同时将计算开销压缩至接近 SGD 水平。
4. **ZO-SAM 是即插即用的通用优化器**，可广泛适配各类 sparse training pipeline，带来一致性的精度、收敛速度与鲁棒性提升。

---

### ⚠️ 方法的局限性
| 局限 | 说明 |
|------|------|
| **Zero-Order Estimator 的近似误差** | 尽管 RGE 控制了采样次数，但仍引入一定噪声，可能影响极端高维任务的表现。 |
| **超参数敏感性** | 如 finite difference step size $ \delta $、random direction 数量 $ m $ 需调优以平衡效率与精度。 |
| **不适用于完全不可微场景** | 虽然减少了 backprop，但仍依赖部分 first-order gradient，不能完全替代黑盒优化。 |

---

### 🔮 未来工作方向
1. **自适应选择 ZO 采样数量 $ m $**：根据层重要性或稀疏模式动态调整，进一步优化效率。
2. **扩展至联邦学习、LLM fine-tuning 等场景**：利用 ZO 特性实现 memory-efficient 或 communication-efficient 训练。
3. **理论分析 ZO-SAM 的收敛性与泛化界**：建立更坚实的理论支撑。
4. **探索与其他正则化技术（如 dropout, mixup）的协同效应**。

---

## ✅ 总结一句话
> **ZO-SAM 是首个将 Zero-Order Optimization 成功嵌入 SAM 框架的工作，在几乎不牺牲泛化性能的前提下，将 SAM 的计算成本降低一半，使其真正可用于高效的 sparse training，为资源受限环境下的深度模型训练提供了实用且强大的新工具。**

</details>

---

### 5. [Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](https://arxiv.org/abs/2603.12996)

**Authors**: Bumjun Kim, Dongjae Jeon, Moongyu Jeon, Albert No  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.12996v1  

#### Abstract
Parallel decoding for diffusion LLMs (dLLMs) is difficult because each denoising step provides only token-wise marginal distributions, while unmasking multiple tokens simultaneously requires accounting for inter-token dependencies. We propose Dependency-Aware Parallel Decoding (DAPD), a simple, trai...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
Diffusion LLMs (dLLMs) 虽然理论上支持并行解码（parallel decoding），但由于模型在训练时仅学习每个掩码位置的**token-wise marginal distributions**，而未建模跨 token 的联合依赖关系，导致直接并行采样多个 token 会产生 **joint-marginal mismatch** —— 即局部合理但全局不一致的输出。

现有基于 marginal confidence 的方法（如 Fast-dLLM、KLASS）虽然能筛选“高置信度”token 进行并行解码，但忽略了 token 之间的强耦合关系，无法从根本上解决该问题。

### 提出了什么新方法或新思路
本文提出 **Dependency-Aware Parallel Decoding (DAPD)**，一种无需额外训练的并行解码方法，其核心思想是：

- 利用 dLLM 内部的 **self-attention** 机制作为代理信号，构建一个 **Markov Random Field (MRF)** 来显式建模被掩码 token 之间的条件依赖结构。
- 在每一步中，将并行解码问题转化为图上的 **独立集选择问题**（Independent Set Selection），即只同时解码那些在 MRF 图中无边连接（表示弱依赖）的 token。
- 采用受 **Welsh-Powell 图着色算法** 启发的启发式策略，优先处理高 degree 的“hub”节点，以简化后续图结构，提升长期并行效率。

### 相比现有方法的优势
- **无需训练或辅助模型**：完全基于推理时可用的内部 attention 信号，保持了 training-free 的优势。
- **更准确地捕捉依赖关系**：相比仅依赖 marginal confidence 的方法，DAPD 显式考虑了 token 间的交互强度。
- **实现真正的全局并行**：能够跨序列不同区域并发解码独立子任务，而非局限于局部连续块，更好地利用了 dLLMs 的 any-order 生成能力。
- **更高的解码效率与质量平衡**：在更少的 decoding steps 下达到更高或相当的 accuracy。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主任务基准**：
  - 数学推理：`GSM8K`, `Math500`
  - 代码生成：`HumanEval`, `MBPP`
  - 指令遵循：`IFEval`
- **并行解码压力测试专用基准**：
  - `ParallelBench`：设计用于暴露 joint-marginal mismatch，包含多种复杂依赖结构的任务（如排序、谜题、复制等）
- **分析性实验数据集**：
  - `TriviaQA`：从中抽取 5 个独立问题拼接成单个 prompt，用于观察多任务并行解码行为。

### 实验设置和评估指标
- **模型**：
  - `LLaDA-8B-Instruct`
  - `Dream-7B-Instruct`
- **评估框架**：`lm-eval-harness`
- **最大生成长度**：256 tokens
- **核心指标**：
  - **Accuracy / Score**：各任务的标准评估指标（如 pass@1）
  - **Decoding Steps (NFE)**：前向传播次数，衡量推理延迟
  - **Speed-up**：相对于 step-by-step 解码的速度提升倍数
  - **Accuracy-Steps Trade-off**：综合评估效率与性能的关键曲线

### 基线方法对比
- **EB-Sampler** (Ben-Hamu et al., 2025)：基于熵约束选择可安全解码的位置
- **KLASS** (Kim et al., 2025c)：结合 confidence 和 KL 散度稳定性信号
- **Fast-dLLM** (Wu et al., 2025)：设定 confidence 阈值进行选择性并行
- 所有 baseline 均使用原论文推荐超参，并在必要时启用 block-wise decoding 和 EOS suppression 以保证公平比较。

---

## 3. 主要实验结果和性能指标

### 关键性能数据
#### 在 LLaDA 上的结果（Table 2 & Figure 3）
| Method | Acc. (%) | Steps | Speed-up |
|--------|----------|-------|----------|
| Original (step-by-step) | 52.64 | 256.0 | 1.0× |
| Fast-dLLM | 52.12 | 124.4 | 2.06× |
| KLASS | 52.20 | 177.4 | 1.44× |
| EB-Sampler | 51.20 | 131.3 | 1.95× |
| **DAPD (Ours)** | **52.08** | **66.2** | **3.87×** |

> ✅ DAPD 在几乎不损失 accuracy 的前提下，实现了 **3.87× 的速度提升**，远超其他 baseline。

#### 在 Dream 上的结果（Figure 3）
- DAPD 在所有任务上均取得最优的 **accuracy-steps trade-off**，尤其在 `HumanEval` 和 `IFEval` 上显著优于 baseline。
- 所有方法均运行于 single-block 模式，验证了 DAPD 在标准设置下的鲁棒性和普适性。

#### 在 ParallelBench 上的表现（Figure 4）
- DAPD 在大多数任务上实现了更优的 **score-steps trade-off**，表明其能更有效地识别低依赖 token 组并行更新，即使在复杂依赖结构下也表现稳健。

### 与基线方法的对比结果
- **速度优势**：DAPD 的平均 decoding steps 比第二快的方法（Fast-dLLM）还少一半以上。
- **解码模式差异**（Figure 1 & 5）：
  - Baseline 方法呈现高度**顺序化、局部聚集**的解码轨迹（类似双向 AR）。
  - DAPD 展现出**空间分散、全局并发**的解码行为，早期即可并行推进多个独立问题。
- **段落数量演化**（Segmentation Count）：
  - DAPD 的 segment count 先上升后下降，体现“分治-合并”策略；
  - Baseline 始终维持少量 segments，反映其串行瓶颈。

### 消融实验结果（文中隐含分析）
- **注意力层选择**：使用顶层 transformer 的 attention（最后两层）效果最佳，因其整合了更多全局上下文。
- **阈值 $T_t$ 设计**：采用保守小阈值（如 0.01），确保非边代表真正可忽略的依赖，避免冲突。
- **degree-confidence 加权排序**：相比纯 degree 或纯 confidence 排序，加权方式更能平衡结构重要性与预测可靠性。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Self-Attention 是有效的依赖探测器**：实验证明 attention weights 可靠地反映了 token 间的 conditional dependence 结构（AUC 达 0.928）。
2. **MRF + Independent Set 是合理的建模范式**：将并行解码视为动态图上的独立集选择，为缓解 joint-marginal mismatch 提供了理论清晰且高效的解决方案。
3. **全局并行是 dLLMs 的天然优势**：DAPD 成功释放了 dLLMs 的 any-order 生成潜力，实现了真正意义上的跨区域并发生成。
4. **无需额外训练也能实现高性能并行**：DAPD 完全利用已有模型内部信号，在无需 retraining 或 auxiliary planner 的情况下超越现有方法。

### 方法的局限性
- **依赖 attention 的质量**：若模型 attention 未能准确反映语义依赖（如低质量模型或特定领域），性能可能下降。
- **图构造近似性**：MRF 构造基于 attention thresholding，是一种经验性近似，非严格概率图模型推断。
- **对极端长程依赖敏感**：虽然优于 baseline，但在极复杂、深层嵌套依赖场景中仍可能存在误判。

### 未来工作方向
- 将 DAPD 思想扩展到 **image/video diffusion models** 中的空间 patch 并行生成。
- 探索 **learnable thresholding mechanisms** 或轻量级 adapter 来优化图结构估计。
- 结合 **speculative decoding** 或 **distillation** 进一步加速残差图的收敛。
- 研究如何将 DAPD 应用于 **streaming generation** 场景，实现低延迟在线输出。

--- 

> **总结一句话**：  
> DAPD 通过将 self-attention 视为依赖探测工具，首次实现了**基于结构感知的 training-free 并行解码**，在保持 accuracy 的同时大幅提升 dLLMs 的推理效率，并揭示了“全局并发”才是发挥 diffusion language models 潜力的正确范式。

</details>

---

### 6. [98$\times$ Faster LLM Routing Without a Dedicated GPU: Flash Attention, Prompt Compression, and Near-Streaming for the vLLM Semantic Router](https://arxiv.org/abs/2603.12646)

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.12646v1  

#### Abstract
System-level routers that intercept LLM requests for safety classification, domain routing, and PII detection must be both fast and operationally lightweight: they should add minimal latency to every request, yet not require a dedicated GPU -- an expensive resource better used for LLM inference itse...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：`98× Faster LLM Routing Without a Dedicated GPU`

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题

在大型语言模型（LLM）推理服务系统中，**语义路由层**（Semantic Router）用于拦截请求并执行安全分类、意图识别、PII检测和模型选择等任务。然而，这类系统面临三大瓶颈：

1. **显存爆炸（Memory Explosion）**：标准的 `scaled dot-product attention`（SDPA）具有 $O(n^2)$ 的注意力掩码内存开销，在处理长上下文（如 8K–32K tokens）时极易发生 OOM。
2. **延迟过高**：CPU 推理虽避免 GPU 显存压力，但延迟随长度超线性增长（例如 8K tokens 达到 4.9 秒），远超可接受范围。
3. **序列化开销大**：通过 Envoy 的 `ext_proc` 处理完整 HTTP body 需要全量反序列化和 JSON 解析，带来显著额外开销。

此外，若为路由层单独配备专用 GPU，则资源利用率低且成本高昂。

---

### 🚀 提出的新方法与创新思路

作者提出了 **三个阶段优化策略**，共同实现 **98× 端到端加速**，使语义路由器可在不使用独立 GPU 的情况下高效运行，并能共享 vLLM 服务所用的 GPU。

#### 主要贡献如下：

| 贡献 | 内容 |
|------|------|
| **1. CK Flash Attention for ONNX Runtime on ROCm** | 在 AMD ROCm 平台上首次将 FlashAttention 集成进 ONNX Runtime，通过自定义算子、图重写和 HIP 内核，将注意力内存从 $O(n^2)$ 降至 $O(n)$，解决 SDPA 在长文本下的 OOM 问题。 |
| **2. Neural-inference-free Prompt Compression** | 提出一种无需神经网络推理的提示压缩管道，结合 **TextRank、位置加权（U-shaped）、TF-IDF 和新颖性评分（Novelty Scoring）**，将任意长度输入压缩至约 512 tokens，且无精度损失。 |
| **3. Near-Streaming Body Processing** | 设计了一种自适应流式处理机制，基于首个 chunk 判断是否需要分类；对 `"auto"` 请求增量预处理，其余直接零拷贝透传，消除全量 JSON 序列化开销。 |
| **4. GPU Co-location with LLM Serving** | 整体方案使得路由模块总 GPU 占用 <800MB，可与 vLLM 共享同一块 GPU（如 MI300X），无需专用加速器，提升集群 GPU 密度与性价比。 |

---

### 🔍 相比现有方法的优势

| 对比维度 | 现有方法（如 RouteLLM、NVIDIA Blueprint） | 本文方法 |
|--------|------------------------------------------|---------|
| **硬件依赖** | 多需专用 GPU（~0.6–16GB） | 可共用服务 GPU，无需专用卡 |
| **显存效率** | SDPA 或普通 BERT 分类器显存占用高 | FlashAttention + 压缩 → 显存恒定小规模 |
| **延迟表现** | CPU 推理慢，GPU 上无法处理 >8K tokens | 支持 32K tokens，E2E 延迟仅 108ms @16K |
| **计算开销** | 多数压缩方法依赖 LLM 推理（如 LLMLingua） | 完全基于经典 NLP 技术，零神经推理开销 |
| **部署灵活性** | 固定缓冲模式，难以扩展 | 自适应流控，支持 near-realtime 路由 |

---

## 2. 核心实验方法和设置

### 📚 数据集与测试负载

- **合成数据生成**：构建不同长度（500、2K、8K、16K tokens）的 OpenAI 格式 prompt，嵌入以下信号以确保触发所有分类器：
  - Jailbreak 前缀（如 `"Ignore all previous instructions..."`）
  - PII 内容（SSN、邮箱、信用卡号）
  - 特定领域技术内容（如计算机科学）
- **真实内容评估**：使用来自 8 篇完整 Wikipedia 文章的离线测试集（共 384 个样本），涵盖 8 个 domain，每个 prompt 包含控制的位置组合（起始/中间/结尾）。

---

### ⚙️ 实验设置

| 组件 | 配置 |
|------|------|
| **硬件平台** | AMD Instinct MI300X GPU（192GB HBM3） |
| **软件栈** | ROCm 7.0, ONNX Runtime 1.22.1, Envoy v1.33 |
| **模型** | mmBERT-32K（Fine-tuned ModernBERT，270M 参数，FP16） |
| **并发配置** | 单节点上运行两个 vLLM 实例 + 语义路由器，共享 GPU |
| **分类任务** | 并行执行三项：Domain Classification、Jailbreak Detection、PII Detection |

---

### 📊 评估指标

| 指标 | 描述 |
|------|------|
| **End-to-End (E2E) Latency** | 从客户端发送请求到返回响应头的时间（curl 测量） |
| **Per-signal Extraction Latency** | 各分类信号提取耗时（Prometheus counter 记录） |
| **GPU Memory Usage** | 分类器会话的峰值显存占用 |
| **Classification Accuracy** | 是否正确捕获 jailbreak、PII、domain 信号 |
| **Throughput (req/s)** | 单位时间内可处理的请求数量（基于 E2E 延迟倒数） |

---

### 🔁 基线方法对比

| 基线 | 描述 |
|------|------|
| **ONNX CPU (Baseline)** | 使用 ONNX Runtime 在 CPU 上运行 mmBERT，Buffered 模式 |
| **Candle CPU** | Rust 框架 Candle 运行压缩版模型（截断至 512 tokens） |
| **ONNX GPU (SDPA)** | GPU 上运行原生 SDPA 注意力（未启用 FlashAttention） |
| **GPU + FA (Stage I)** | 引入 CK Flash Attention 后的 GPU 加速版本 |
| **+ Prompt Compression (Stage II)** | 添加经典 NLP 压缩后输入固定为 ~512 tokens |
| **+ Near-Streaming (Stage III)** | 支持流式 chunk 处理，减少 JSON 开销 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table VIII）

| 配置 | 输入长度 | E2E 延迟 (ms) | 相对于 CPU 的加速倍数 |
|------|----------|----------------|------------------------|
| ONNX CPU (Baseline) | 8K | 4,918 | 1.0× |
| GPU + CK FlashAttention | 8K | 127 | **38.7×** |
| + Prompt Compression | 8K | 62 | **79.3×** |
| + Near-Streaming | 8K | **50** | **98.4×** |
| 扩展至 16K tokens | 16K | **108** | ——（CPU 不可达） |

> ✅ **累计达 98× 加速**，16K tokens 下仍稳定在 108ms，而 CPU 基线在 8K 已达 4.9s。

---

### 🔬 消融实验分析（Ablation Study）

#### （1）各阶段加速效果分解（@8K tokens）

| 阶段 | 功能 | 延迟下降 | 加速比 |
|------|------|---------|-------|
| Stage I: GPU + FA | 替换 SDPA → FlashAttention | 4,918 → 127 ms | 38.7× |
| Stage II: Prompt Compression | 输入压缩至 ~512 tokens | 127 → 62 ms | 2.0× |
| Stage III: Near-Streaming | 零拷贝 JSON + 自适应流控 | 62 → 50 ms | 1.2× |

> ✅ 三阶段叠加呈乘法效应，最终实现 **98× 性能提升**。

---

#### （2）Prompt Compression 效果（Table VI）

| 输入长度 | Jailbreak 分类延迟（无压缩） | 压缩后 |
|--------|-------------------------------|--------|
| 500 tokens | 10.1 ms | 9.3 ms |
| 16K tokens | 126.6 ms | **10.4 ms**（↓12×） |

> ✅ 压缩后无论原始长度如何，分类延迟基本恒定（~10–11ms），极大缓解长文本压力。

---

#### （3）压缩开销本身（Table VII）

| 输入长度 | 输出长度 | 压缩比率 | 压缩耗时（CPU-only） |
|--------|----------|-----------|---------------------|
| 2K | ~510 | 25.1% | 2 ms |
| 8K | ~512 | 6.4% | 9 ms |
| 16K | ~512 | 3.2% | **19 ms** |

> ✅ 压缩过程完全在 CPU 上完成，无 GPU 开销，且其耗时远小于节省的 GPU 推理时间（如 16K 下节省 ~240ms）。

---

#### （4）并发负载下的可扩展性（Table X）

| 原始长度 | C=1（延迟） | C=20（有效 E2E 延迟） |
|--------|-------------|----------------------|
| 512 tokens | 17 ms | 140 ms |
| 16K tokens | 108 ms | 231 ms |
| 32K tokens（估计） | 125 ms | **248 ms** |

> ✅ 即使在 C=20 并发下，32K tokens 的有效延迟仍低于 250ms，且 **无任何 OOM 发生**。

---

#### （5）分类准确率对比（Table XI）

| 指标 | 原始（未压缩） | 压缩后 |
|------|---------------|--------|
| Domain Classification | 53.1% | **61.2%** ↑ |
| PII Detection | 78.5% | **92.4%** ↑ |
| Jailbreak Detection | 70.8% | 56.6% ↓ |

> ✅ **压缩反而提升了多数任务的准确性**！原因在于去除了冗余信息（“noise”），增强了信号密度（signal-to-noise ratio）。  
> ⚠️ Jailbreak 准确率下降是设计取舍：生产环境中该任务仍在原始 prompt 上运行，压缩仅用于 domain routing。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **FlashAttention 是长上下文路由的前提条件**：在 AMD 平台上通过自定义 CK 算子成功集成 FlashAttention，解决了 $O(n^2)$ 显存瓶颈，使 32K tokens 成为可能。
2. **Prompt Compression 不仅提速还能提准**：基于经典 NLP 的非神经压缩方法不仅零推理开销，还起到“去噪”作用，显著提升 domain 和 PII 分类准确率。
3. **Near-Streaming 极大降低 I/O 开销**：通过早期判断 `"model"` 字段决定是否透传，大多数请求实现接近零拷贝转发，大幅削减 JSON 序列化成本。
4. **三阶段优化具有乘法叠加效应**：GPU 加速 × 输入缩短 × I/O 优化 = **98× 端到端加速**。
5. **真正实现 GPU 共享部署**：整个路由系统 GPU 占用 <800MB，可与 vLLM 共享 MI300X，无需专用 GPU，每年每节点节省高达 \$8,800–\$10,100。

---

### ⚠️ 局限性

| 限制 | 说明 |
|------|------|
| **Tokenizer Approximation** | 当前压缩使用字符长度估算 token 数，而非精确 tokenizer，可能导致预算偏差。 |
| **Accumulate Path 仍需缓存全文** | 在 `"auto"` 模式下，仍需在内存中累积整个 body，虽减少 JSON 操作，但未降低最大内存占用。 |
| **AMD 特定优化 Stage 1 不通用** | CK FlashAttention 仅适用于 AMD ROCm，NVIDIA 用户已有 cuDNN 支持，但 Stage 2&3 仍普适。 |

---

### 🔮 未来工作方向

1. **集成精确 Tokenizer**：引入目标模型的真实 tokenizer 实现更精准的 token 预算控制。
2. **进一步轻量化压缩算法**：探索更低延迟的句子评分机制，支持更大规模文档实时处理。
3. **双向流式支持**：扩展至输出侧流控，实现端到端 near-streaming pipeline。
4. **跨平台统一接口封装**：将三阶段优化打包为通用 SDK，支持多种 backend（CUDA/ROCm/CPU）自动适配。
5. **动态压缩率调整**：根据负载情况动态调节压缩强度，在精度与速度间做弹性平衡。

---

## 📌 总结一句话

> 本文提出一套 **免专用 GPU 的高性能 LLM 语义路由方案**，通过 **CK FlashAttention（AMD）、无神经提示压缩、近流式 body 处理** 三阶段优化，实现了 **98× 端到端加速**，支持 32K tokens 实时分类，同时提升准确率并允许与 vLLM 共享 GPU，显著降低部署成本。

</details>

---

### 7. [Disentangled Latent Dynamics Manifold Fusion for Solving Parameterized PDEs](https://arxiv.org/abs/2603.12676)

**Authors**: Zhangyong Liang, Ji Zhang  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.12676v1  

#### Abstract
Generalizing neural surrogate models across different PDE parameters remains difficult because changes in PDE coefficients often make learning harder and optimization less stable. The problem becomes even more severe when the model must also predict beyond the training time range. Existing methods u...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Disentangled Latent Dynamics Manifold Fusion for Solving Parameterized PDEs

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对**参数化偏微分方程（parameterized PDEs）求解中的两个核心挑战**：

- **参数泛化能力差**：传统 PINNs 在面对未见过的 PDE 参数配置时表现不佳，需重新训练。
- **时间外推（temporal extrapolation）困难**：标准模型将时间作为静态输入坐标处理，缺乏内在动力学建模，导致在训练时间窗口之外预测迅速失效。

此外，现有基于 latent dynamics 的方法（如 PIDO、DINO）依赖于测试阶段的迭代 auto-decoding，计算开销大且破坏了解空间的连续几何结构。

---

### **提出了什么新方法或新思路**

作者提出了一种名为 **Disentangled Latent Dynamics Manifold Fusion (DLDMF)** 的新型物理信息神经网络框架，其核心思想是：

- **空间-时间-参数解耦表示（Space-Time-Parameter Disentanglement）**  
  将空间坐标 $x$、时间动态 $t$ 和 PDE 参数 $\mu$ 分别通过独立的前馈编码器映射到不同的隐流形（manifold），避免耦合优化难题。

- **显式参数条件化神经常微分方程（Parameter-Conditioned Neural ODE）**  
  时间演化由一个受 PDE 参数调制的 Neural ODE 显式建模，实现对连续时间轨迹的积分而非逐点回归。

- **动态流形融合机制（Dynamic Manifold Fusion）**  
  在共享的隐式神经表示（INR）解码器中融合空间嵌入 $h_x$、参数嵌入 $h_\mu$ 和随时间演化的潜态 $z_t$，生成物理一致的时空解。

- **免去测试时迭代优化（Amortized Feed-Forward Initialization）**  
  参数直接通过确定性前馈映射初始化潜态 $z_0 = g_{\text{init}}(h_\mu)$，完全规避了 auto-decoding 所带来的反问题瓶颈。

---

### **相比现有方法的优势**

| 方面 | DLDMF | 现有方法（如 P²INN, PIDO, MAD） |
|------|-------|-------------------------------|
| **参数泛化** | 强，通过连续参数嵌入支持 out-of-distribution 参数 | 部分支持，但受限于离散编码或浮点向量 |
| **时间外推** | 内生动力学建模，支持任意长时外推 | 外推能力弱，或依赖滚动预测误差累积 |
| **推理效率** | 前馈 + ODE 积分，无迭代优化，速度快 | Auto-decoding 需梯度优化，延迟高 |
| **几何一致性** | 维持参数空间的连续流形结构 | Auto-decoding 导致离散码本，破坏连续性 |
| **稳定性** | 结构上解耦参数刚度与时间动态，更鲁棒 | 易受谱偏置（spectral bias）和刚度影响 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### （1）**1D 参数化对流-扩散-反应方程（CDR）**
$$
\frac{\partial u}{\partial t} + \beta \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} - \rho u(1 - u)
$$
- 参数 $(\beta, \nu, \rho)$ 控制对流、扩散、反应强度
- 设计多种组合以模拟“易”、“难”物理场景（如强对流/高频率震荡）
- 初始条件包括 Gaussian 分布和 $1+\sin(x)$

#### （2）**2D Navier-Stokes 流体动力学（forced turbulence setting）**
- 模拟不可压缩流场
- 将雷诺数（Reynolds number）作为可变参数
- 考察高维时空下的参数泛化与长期预测能力

---

### **实验设置和评估指标**

#### **训练/测试划分**
- **时间维度**：训练区间 $[0, T_r]$，测试外推至 $T_s > T_r$
  - In-t: $t \leq T_r$
  - Out-t: $t > T_r$
- **参数维度**：
  - Interpolation: 参数在训练范围凸包内
  - Extrapolation: 参数超出训练分布边界

#### **评估指标**
| 指标 | 定义 | 用途 |
|------|------|------|
| **L2 相对误差** | $\|u - \hat{u}\|_2 / \|u\|_2$ | 主要精度衡量 |
| **L2 绝对误差** | $\|u - \hat{u}\|_2$ | 数值偏差大小 |
| **最大误差（Max Error）** | $\max |u - \hat{u}|$ | 局部尖峰捕捉能力 |
| **解释方差得分（Explained Variance Score）** | $1 - \frac{\mathrm{Var}(u - \hat{u})}{\mathrm{Var}(u)}$ | 拟合质量，越接近1越好 |

#### **外推协议**
- 对非原生支持外推的方法（如 P²INN），采用 auto-regressive rolling forecast
- DLDMF 直接通过 ODE 积分扩展至任意时间点

---

### **基线方法对比**

| 方法 | 类型 | 是否支持参数泛化 | 是否支持时间外推 | 是否需要 auto-decoding |
|------|------|------------------|------------------|------------------------|
| **P²INN** | 参数化 PINN | ✅ | ❌（失败明显） | ❌ |
| **PI-DeepONet** | Operator Learning | ✅ | ⭕（滚动预测） | ❌ |
| **DINO** | Latent Dynamics + INR | ⭕ | ✅ | ✅（instance-wise） |
| **PIDO** | Physics-Informed Latent ODE | ✅ | ✅ | ✅ |
| **MAD** | Meta-Auto-Decoding | ✅ | ⭕ | ✅ |

所有基线均在相同容量、训练预算下复现，确保公平比较。

---

## 3. 主要实验结果和性能指标

### **关键性能数据（来自 Table 1）**

| Model | Dataset | In-t L2 Rel. Err (%) | Out-t L2 Rel. Err (%) |
|-------|---------|------------------------|------------------------|
| **DLDMF** | — | **1.89** | **4.21** |
| P²INN | — | 21.34 | 32.87 |
| PIDO | — | 5.67 | 8.94 |
| DINO | 100% | 4.89 | 6.52 |
| DINO | 50% | 5.25 | 8.76 |

> ✅ **DLDMF 在 In-t 和 Out-t 上均显著优于所有基线**，尤其在外推任务中误差仅为 P²INN 的约 **1/8**

---

### **与基线方法的对比结果**

- **P²INN**：虽然参数泛化能力强，但在 $t=5.0$ 和 $t=10.0$ 时严重偏离真实解（见 Figure 2），无法保持平坦稳态，说明缺乏动力学建模。
- **PIDO/DINO**：具备一定外推能力，但由于依赖 auto-decoding 初始化潜态，在参数剧烈变化时不稳定，误差增长较快。
- **DLDMF**：即使在参数外推 + 时间外推双重压力下仍保持稳定预测，验证了其强大的联合泛化能力。

---

### **消融实验结果（Ablation Studies）**

| 变体 | 描述 | Out-t L2 Rel. Err (%) | 分析 |
|------|------|------------------------|------|
| Full DLDMF | 完整模型 | **4.21** | — |
| w/o Latent ODE | 用时间编码替代 ODE | ~15–20 | 缺乏动态建模导致外推崩溃 |
| w/o Manifold Fusion | 输入拼接代替解耦融合 | ~7–9 | 耦合引入优化困难 |
| w/ Auto-decoding | 替换为迭代优化初始化 | ~6.5 | 推理慢，且性能下降，证明前馈初始化更优 |

> 🔍 消融实验证明：**latent dynamics + disentangled fusion + feed-forward init** 是性能提升的关键三要素。

---

## 4. 关键结论和发现

### **主要发现**

1. **静态坐标回归无法支撑可靠的时间外推**  
   即使像 P²INN 这样优秀的参数化模型，在训练时间窗外也会快速退化，表明必须引入**内在动力学建模**。

2. **auto-decoding 存在结构性缺陷**  
   - 是一个非凸逆问题，初始值敏感；
   - 破坏参数空间的连续几何结构；
   - 推理成本高昂，不适合实时或多查询场景。

3. **DLDMF 实现了结构解耦与功能统一**  
   - 通过 feed-forward 参数编码建立连续参数流形；
   - 利用 parameter-conditioned Neural ODE 实现稳定时间演化；
   - 动态流形融合保留表达力同时增强训练稳定性。

4. **误差传播被有效抑制**  
   在长时间外推中，DLDMF 的误差呈亚线性增长，而 P²INN 呈近似线性上升（见 Figure 4），说明其动力学建模能约束误差积累。

---

### **方法的局限性**

- 当前实验集中在规则域和周期性边界条件，尚未验证复杂几何或非均匀网格上的表现。
- 对极高维参数空间（如空间依赖系数场）的扩展尚不明确。
- Neural ODE 积分可能在极端刚性系统中面临数值稳定性挑战。

---

### **未来工作方向**

1. **拓展至非结构化网格与复杂边界条件**
   - 结合 GNN 或 Mesh-based INRs 支持任意几何。
   
2. **集成自适应 ODE 求解器**
   - 根据局部刚度自动调整步长与精度，提高效率。

3. **多尺度建模与层次化 latent dynamics**
   - 构建 hierarchical latent space 以捕捉不同尺度的动力学行为。

4. **实际工程系统部署**
   - 应用于气候模拟、材料设计等大规模科学计算任务。

---

> 📌 **总结一句话**：  
> **DLDMF 是首个将 parameter manifold encoding、continuous-time latent dynamics 与 amortized inference 统一于单一几何框架下的物理信息学习方法，在参数泛化与长期时间外推方面实现了质的飞跃。**

</details>

---

### 8. [Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](https://arxiv.org/abs/2603.13085)

**Authors**: Jose Marie Antonio Mi\~noza, Paulo Mario P. Medina, Sebastian C. Iba\~nez  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.13085v1  

#### Abstract
Understanding the theoretical foundations of attention mechanisms remains challenging due to their complex, non-linear dynamics. This work reveals a fundamental trade-off in the learning dynamics of linearized attention. Using a linearized attention mechanism with exact correspondence to a data-depe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics**

---

## 1. **论文的主要贡献和创新点**

### ✅ **解决了什么问题**
该论文聚焦于**注意力机制（attention）在学习动态上的理论基础缺失**问题。尽管注意力机制在实践中表现出强大的灵活性和性能，但其复杂的非线性动态使得传统的 Neural Tangent Kernel (NTK) 理论难以适用。现有研究多关注初始化结构或最终性能，而忽略了训练过程中表示如何演化。

本文揭示了一个根本性现象：**即使在网络宽度很大时，线性化注意力（linearized attention）也无法收敛到其无限宽度的 NTK 极限**，这挑战了“宽网络趋于进入 kernel regime”的普遍认知。

---

### ✅ **提出了什么新方法或新思路**

#### （1）提出 **Influence Malleability（影响可塑性）**  
这是一个全新的量化指标，用于衡量模型对训练样本依赖关系的动态调整能力：
- 定义为：当训练数据被扰动（如加入对抗样本）时，模型对特定训练样本的“影响力”（influence）发生符号翻转的比例。
- 高 malleability 表示模型能快速重新评估哪些样本是有帮助/有害的，反映了其对数据质量变化的敏感性和适应性。

#### （2）建立 **线性化注意力与数据依赖核（data-dependent Gram-induced kernel）之间的精确对应关系**  
证明了线性化注意力 $ f^{\text{att}}(X) = X X^\top X $ 对应一个由 Gram 矩阵 $ G = X X^\top $ 诱导的四阶核：
$$
K_{\text{LinAttn}}(x_i, x_j) = \sum_{k,l} (x_i^\top x_k)(x_k^\top x_l)(x_l^\top x_j)
$$
该核具有**传递相似性传播**（transitive similarity）特性，即影响可通过中间样本间接传递。

#### （3）从谱放大角度解释 NTK 不收敛  
提出 **Spectral Amplification（谱放大）定理**：注意力变换将 Gram 矩阵的条件数立方化：
$$
\kappa(\tilde{G}) = \kappa(G)^3
$$
因此，要使有限宽度 NTK 收敛，所需宽度为 $ m = \Omega(\kappa(G)^6) $，远超实际可行范围（例如在 CIFAR-10 上需 $ m > 10^{24} $），从而形式化地解释了为何注意力始终处于 feature learning regime。

---

### ✅ **相比现有方法的优势**

| 方面 | 优势说明 |
|------|----------|
| **理论深度** | 首次为线性化注意力提供精确的 kernel 表达，并将其不收敛归因于谱放大机制，而非经验观察。 |
| **新视角** | 引入 *influence malleability* 作为衡量架构敏感性的统一指标，连接了泛化能力与鲁棒性之间的权衡。 |
| **双重含义框架** | 揭示注意力的强大与脆弱同源——都源于其脱离 kernel regime 的能力：既能更好拟合任务结构（降低 bias），也更易受对抗攻击。 |

---

## 2. **核心实验方法和设置**

### 📚 **使用的数据集**
- **MNIST**
- **CIFAR-10**
- **Fashion-MNIST**（补充材料）

所有输入均进行单位范数归一化（$ \|x_i\|_2 = 1 $）以满足理论假设。

---

### ⚙️ **实验设置**

#### **模型架构对比**
| 模型 | 结构描述 |
|------|---------|
| **2L-ReLU** | 两层 ReLU MLP，直接处理原始输入 |
| **MLP-Attn** | 线性化注意力预处理 $ X \mapsto X X^\top X $，再接相同结构的两层 ReLU MLP |

> 注意：两种模型的 MLP 部分完全一致，确保比较公平。

#### **训练配置**
- 优化器：Adam ($ \eta = 10^{-3} $)
- 批大小：128
- 正则化：L2 ($ \lambda = 10^{-3} $)
- 训练轮数：500
- 宽度范围：$ m \in \{4, 8, ..., 4096\} $

---

### 🎯 **评估指标**

| 指标 | 含义 |
|------|------|
| **NTK 距离** $ \|f_m - f_{\text{NTK}}\| $ | 有限宽度模型输出与无限宽度 NTK 预测之间的差异，用于判断是否进入 kernel regime |
| **Influence Flip Rate** | 在前 10% 最具影响力的训练样本上施加对抗扰动后，影响力符号反转的比例 |
| **Spearman’s Rank Correlation $ \rho $** | 原始与扰动后的影响力排序相关性，衡量稳定性 |
| **Top-K Stability** | 扰动前后 Top-K 影响样本集合的交集比例 |

---

### 🔍 **基线方法对比**
- 主要对比对象是标准 **2L-ReLU** 网络。
- 使用相同的 MLP 容量和训练流程，唯一区别在于是否引入线性化注意力模块。

---

## 3. **主要实验结果和性能指标**

### 📊 **关键性能数据汇总**

#### （1）**NTK 距离趋势（Table 1 & Figure 1）**

| Model / Width | m=16 | m=1024 | m=4096 |
|---------------|------|--------|--------|
| **2L-ReLU (MNIST)** | 45.1 | 39.9 | 39.2 |
| **MLP-Attn (MNIST)** | 10.3 | 33.3 | 43.4 |
| **2L-ReLU (CIFAR-10)** | 246.2 | 101.7 | 56.9 |
| **MLP-Attn (CIFAR-10)** | 3.7 | 10.4 | 12.6 |

✅ **结论**：
- 2L-ReLU 显示单调下降 → 收敛至 NTK 极限（lazy training）
- MLP-Attn 显示非单调或上升趋势 → **永不进入 kernel regime**

---

#### （2）**Influence Flip Rate（Table 2 & 3）**

##### 多类分类（10-class）

| Dataset | Model | FGSM | PGD | MIM |
|--------|-------|------|-----|-----|
| MNIST | 2L-ReLU | 4.1% | 3.3% | 3.4% |
| MNIST | MLP-Attn | 34.6% | **28.9%** | 21.9% |
| CIFAR-10 | 2L-ReLU | 3.3% | 3.1% | 3.2% |
| CIFAR-10 | MLP-Attn | 26.4% | **19.1%** | 20.5% |

➡️ **MLP-Attn 的 flip rate 是 ReLU 的 6–9 倍！**

##### 二元分类（Binary）

| Dataset | Model | PGD Flip Rate |
|--------|-------|----------------|
| MNIST | 2L-ReLU | 8.4% |
| MNIST | MLP-Attn | **41.0%** |
| CIFAR-10 | 2L-ReLU | 15.5% |
| CIFAR-10 | MLP-Attn | 14.0% |

➡️ 在简单任务（MNIST）中优势显著，在复杂任务（CIFAR-10）中减弱，符合谱放大理论预测（$ \kappa(G) $ 更大 → 效果更强）

---

#### （3）**对抗训练的影响（Table 4）**

| Dataset | Model | Standard | Adv-Trained |
|--------|-------|----------|-------------|
| MNIST | 2L-ReLU | 3.3% | **43.4%** |
| MNIST | MLP-Attn | 28.9% | 42.2% |
| CIFAR-10 | 2L-ReLU | 3.1% | **36.5%** |
| CIFAR-10 | MLP-Attn | 19.1% | 38.6% |

✅ **发现**：
- 对抗训练大幅提高 ReLU 的 malleability，迫使其“模仿”注意力的行为。
- 但 **MLP-Attn 在标准训练下已具备高 malleability**，说明这是其**架构固有属性**（architectural induction），而非训练诱导。

---

### 🔬 **消融实验与补充分析**
- **不同扰动方式（FGSM/PGD/MIM）**：均显示 MLP-Attn 更敏感。
- **干预策略**：
  - Curated（移除高影响力样本）
  - Transformed（替换为对抗样本）
  - Adversarial（全数据扰动）
- **Rank correlation 分析**：MLP-Attn 的 $ \rho $ 更低，表明其影响力排序持续演变，体现 feature learning 特征。

---

## 4. **关键结论和发现**

### ✅ **主要发现**

1. **线性化注意力不收敛于 NTK 极限**  
   即使在极大宽度下，NTK 距离也不减小，反而增加或波动，说明其始终处于 **feature learning regime**。

2. **根本原因是 Spectral Amplification**  
   注意力操作 $ X \mapsto X X^\top X $ 将 Gram 矩阵条件数立方化，导致收敛所需的宽度呈指数级增长，远超现实可能。

3. **提出 Influence Malleability 作为核心度量**  
   - MLP-Attn 的 malleability 比 2L-ReLU 高 **6–9 倍**
   - 反映了其对训练数据质量的高度敏感性

4. **双刃剑效应（Dual Implications）**
   - ✅ **正面**：数据依赖核可与任务结构对齐，降低 approximation error（bias）
   - ❌ **负面**：同样机制使其极易受到对抗样本操纵，影响稳定性

5. **注意力的灵活性是“天生的”**  
   不需要对抗训练即可获得高 malleability，而 ReLU 必须通过强正则化才能逼近。

---

### ⚠️ **方法的局限性**

| 局限 | 说明 |
|------|------|
| **线性化近似** | 使用 $ \text{softmax}(A) \approx 1 + A $ 近似，未包含 softmax 的竞争性归一化效应，可能低估真实注意力的非线性程度 |
| **参数自由设计** | 使用 $ W_Q=W_K=W_V=I $，虽经证明不影响结构性质，但仍简化了实际 Transformer 中 QKV 投影的学习动态 |
| **小规模实验** | 实验限于 MNIST/CIFAR-10 和宽度 ≤ 4096，尚未扩展到大规模语言模型或 Vision Transformer |
| **计算复杂度高** | NTK 和 influence 计算为 $ O(n^3) $，难以应用于百万级数据集 |

---

### 🔮 **未来工作方向**

1. **扩展至完整 Softmax Attention**  
   探索 row-wise softmax 归一化是否会进一步加剧谱放大效应。

2. **多头注意力与深层堆叠分析**  
   分析 multi-head 和 stacked attention 层的组合 kernel 结构及其对 malleability 的累积影响。

3. **结合 Layer Normalization 的理论建模**  
   当前分析忽略 LN，但它在稳定训练中起关键作用。

4. **指导鲁棒注意力设计**  
   利用低秩截断（truncated attention）等手段控制 $ \kappa(G) $，主动恢复 kernel regime，提升鲁棒性。

5. **发展 scalable influence estimation 方法**  
   如 Nyström 近似、迭代求解器等，以支持更大规模验证。

---

## 🧩 总结一句话

> **注意力机制的强大表达能力和高度敏感性，源自同一个根源——它本质上无法进入 kernel regime，而是始终处于 feature learning 状态；这种“非收敛性”由谱放大机制驱动，并可通过 influence malleability 定量刻画。**

</details>

---

### 9. [OpenACMv2: An Accuracy-Constrained Co-Optimization Framework for Approximate DCiM](https://arxiv.org/abs/2603.13042)

**Authors**: Yiqi Zhou, Yue Yuan, Yikai Wang, Bohao Liu, Qinxin Mei, Zhuohua Liu, Shan Shen, Wei Xing, Daying Sun, Li Li, Guozhu Liu  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.13042v1  

#### Abstract
Digital Compute-in-Memory (DCiM) accelerates neural networks by reducing data movement. Approximate DCiM can further improve power-performance-area (PPA), but demands accuracy-constrained co-optimization across coupled architecture and transistor-level choices. Building on OpenYield, we introduce Ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：OpenACMv2: An Accuracy-Constrained Co-Optimization Framework for Approximate DCiM

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前在 **Approximate Digital Compute-in-Memory (ADCiM)** 设计中存在以下挑战：
- **设计空间巨大且高度耦合**：涉及架构级（如压缩器组合、SRAM宏配置）和电路级（晶体管尺寸调整）的联合优化。
- **缺乏准确性感知的快速评估机制**：传统EDA流程（综合+时序分析+功耗仿真）耗时严重，难以支持大规模“what-if”探索。
- **缺少统一、开源的端到端优化框架**：现有工具（如AutoDCIM、SynDCIM等）多聚焦于精确计算或单一层次优化，无法实现精度约束下的跨层协同优化。

### 🚀 提出的新方法与创新
本文提出 **Accuracy-Constrained Co-Optimization (ACCO)** 框架，并基于此构建 **OpenACMv2** ——一个开源、可复现的两阶段优化框架：

#### 主要贡献：
1. **Two-Level Optimization for DCiM**
   - **Level 1（架构级搜索）**：在显式的应用级准确率预算下（如MRED/NMED），进行压缩器组合选择与SRAM宏参数配置。
   - **Level 2（电路级调优）**：对选定设计执行PVT/工艺变异感知的Monte Carlo晶体管尺寸优化，提升鲁棒性与PPA表现。
   - 通过解耦策略降低复杂度，保证收敛稳定性。

2. **算法集成灵活性**
   - 支持多种经典单目标（SA、PSO、CBO）与多目标优化器（MOEA/D、NSGA-II、SMAC、MOBO），验证其在ACCO流程中的有效性。

3. **Fast GNN-Based Approximate Multiplier Modeling（PEA-GNN）**
   - 构建图神经网络模型PEA-GNN，将乘法器结构编码为分层图，预测MRED、NMED、Delay、Power、Area等指标。
   - 实现比传统EDA快 **142×~464×** 的推理速度，同时保持高保真度（R² > 0.95）。

4. **开放性和兼容性**
   - 完全开源，兼容FreePDK45与OpenROAD生态，支持可复现评估与社区扩展。

### 🔍 相比现有方法的优势
| 对比维度 | OpenACMv2 | 其他方法（如OpenACM、AutoDCIM、ARCTIC） |
|--------|---------|------------------------------------|
| 优化层级 | 联合架构+晶体管级 | 多为单一层次优化 |
| 准确性建模 | 显式误差约束 + 快速GNN代理模型 | 缺乏精度感知或依赖慢速仿真 |
| 探索效率 | GNN加速，支持百万次查询 | 单次查询需数十秒至分钟级 |
| 开放程度 | 完整开源，支持OpenROAD集成 | 多闭源或部分开源 |
| 变异感知 | Level-2支持PVT/Monte Carlo分析 | 多忽略工艺波动影响 |

---

## 2. 核心实验方法和设置

### 📚 数据集与生成方式
- **非公开真实数据集**，但所有训练/测试样本由 **OpenACMv2框架自动生成**。
- 基于 **Nangate45nm标准单元库** 自动生成大量近似乘法器设计实例（含不同压缩器组合、位宽、拓扑结构）。
- 输入激励采用均匀网格采样（uniform grid）用于MRED/NMED计算。

### ⚙️ 实验设置
- **平台环境**：
  - CPU: Intel Xeon Gold 6330 @ 2.00GHz
  - GPU: NVIDIA A100 (40GB)
- **EDA基准流程**：
  - 使用 **OpenROAD + OpenSTA + VCS** 进行逻辑综合与时序/功耗分析作为“真实值”（ground truth）。
- **GNN训练细节**：
  - 模型：PyTorch + PyG 实现 PEA-GNN
  - 优化器：Adam
  - 特征提取：基于truth table与信号概率融合节点特征
  - 输出激活函数：Softplus确保物理量非负

### 🎯 评估指标
| 类别 | 指标 | 描述 |
|------|------|------|
| **准确性** | MRED（Mean Relative Error Distance）<br>NMED（Normalized Mean Error Distance） | 衡量近似计算引入的相对/绝对误差 |
| **物理性能** | Delay（延迟）<br>Dynamic Power（动态功耗）<br>Area（面积）<br>PDP（Power-Delay Product） | 综合PPA指标 |
| **系统质量** | PSNR（图像融合任务）<br>Top-1 / Top-5 Accuracy（CIFAR-10分类） | 应用层任务表现 |
| **效率** | Runtime（运行时间）<br>R² / MSE | 模型预测精度与加速比 |

### 🔀 基线方法对比
- **架构级优化器对比**：
  - MOEA/D, NSGA-II, SMAC, MOBO（用于乘法器设计）
  - CBO, PSO, SA（用于SRAM宏配置）
- **设计基线**：
  - **Base design**：使用Yang1压缩器的标准精确乘法器
  - 各Case代表不同的MRED预算下的最优解

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### ✅ PEA-GNN 预测精度（表1）
| 指标 | 8-bit MSE/R² | 16-bit MSE/R² | 加速比 |
|------|---------------|----------------|--------|
| MRED | 0.52e-4 / 0.998 | 2.94e-4 / 0.977 | 142× |
| NMED | 9.25e-4 / 0.996 | 2.66e-4 / 0.959 | — |
| Delay | 6.52e-4 / 0.969 | 7.39e-4 / 0.917 | — |
| Area | 1.77e-4 / 0.991 | 3.65e-4 / 0.978 | — |
| Power | 2.34e-4 / 0.989 | 0.86e-4 / 0.968 | 464× |

> ✔️ 所有指标R²均高于0.91，尤其MRED/NMED误差控制在2.1%~4.7%，满足精度敏感场景需求。

#### ✅ 8-bit乘法器性能（表2 & 图5）
- 在不同MRED预算下，ACCO成功找到一系列PDP更优的设计：
  - Case1（MRED=5.88e-2）→ PDP=191 fJ（较Base下降60%+）
  - Case5（MRED=4.25e-3）→ PDP=229 fJ，PSNR达64.31dB
- Level-2晶体管调优进一步降低PDP（平均↓5~10%），且不牺牲PSNR。

#### ✅ 16-bit乘法器与CIFAR-10推理（表3 & 图6）
- 尽管MRED跨度达两个数量级（1.66e-3 → 1.77e-4），Top-1准确率变化小于1%（65.1%~66.5%），体现NN强容错能力。
- 最佳设计PDP从3874 fJ降至1289 fJ（↓66.7%），显著节能。
- Level-2调优持续改善PDP（如Case1从1216→1203 fJ）。

#### ✅ 晶体管级优化效果（图7）
- 在Sabetz压缩器上，MOEA/D获得最佳Pareto前沿。
- 所有8种压缩器经MOEA/D调优后，PDP-Area曲线全面左移，表明设备尺寸优化带来一致增益。
- **关键发现**：Level-2调优不改变MRED，符合ACCO“不破坏精度约束”的要求。

#### ✅ SRAM宏优化（图8 & 图9）
- 架构级选择主导PPA表现：小阵列低功耗但高延迟，大阵列反之；最优配置出现在“elbow point”附近。
- 晶体管级调优收益有限——受限于外围电路（periphery）的非理想行为，安全调优窗口窄。
- 结论：**SRAM优化应优先关注架构配置，器件调优仅作微调**。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **ACCO框架有效实现了精度约束下的跨层协同优化**：
   - 两阶段解耦策略兼顾搜索效率与收敛稳定性。
   - Level-1快速筛选候选，Level-2精细调优，整体流程高效可靠。

2. **PEA-GNN是支撑大规模探索的关键使能技术**：
   - 数百倍加速使得原本不可行的“百万次what-if”探索成为可能。
   - 高保真预测保障了最终设计的质量。

3. **NN对算术误差具有天然容忍性**：
   - 即使MRED较高，也能维持接近原始的Top-k准确率，为近似计算提供理论基础。

4. **架构决策 > 器件调优**：
   - 对于SRAM宏，架构配置的影响远大于bitcell尺寸调整。
   - 对于乘法器，两者均有贡献，但Level-1决定主趋势，Level-2提供增量改进。

### ⚠️ 局限性
1. **Surrogate模型覆盖范围有限**：
   - 当前PEA-GNN未充分建模PVT角下的性能漂移，泛化能力有待加强。
2. **目标函数范围较窄**：
   - 仅关注PDP与accuracy，忽略了throughput、leakage power、IR drop、routing congestion等实际因素。
3. **后端签核流程集成不足**：
   - 缺少完整的布局布线（P&R）、寄生抽取、串扰分析等signoff级验证。
4. **SRAM宏优化深度不够**：
   - bitcell与外围电路（sense amp, decoder）未联合优化，限制了潜力挖掘。

### 🔮 未来工作方向
- 扩展PEA-GNN以支持PVT-aware预测。
- 引入更多物理效应建模（如crosstalk、variability propagation）。
- 实现bitcell-periphery co-design flow。
- 将ACCO应用于更大规模的CiM阵列系统级优化。
- 探索面向特定DNN模型（如ResNet、Transformer）的定制化近似策略。

---

> 💡 **总结一句话**：  
> **OpenACMv2通过“GNN代理模型 + 两层解耦优化”，首次实现了在严格精度约束下、高效可扩展的ADCiM跨层协同设计自动化，推动了近似计算向实用化与开源化迈进一大步。**

</details>

---

### 10. [RTD-Guard: A Black-Box Textual Adversarial Detection Framework via Replacement Token Detection](https://arxiv.org/abs/2603.12582)

**Authors**: He Zhu, Yanshu Li, Wen Liu, Haitian Yang  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.12582v1  

#### Abstract
Textual adversarial attacks pose a serious security threat to Natural Language Processing (NLP) systems by introducing imperceptible perturbations that mislead deep learning models. While adversarial example detection offers a lightweight alternative to robust training, existing methods typically re...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《RTD-Guard: A Black-Box Textual Adversarial Detection Framework via Replacement Token Detection》总结

---

## 1. 论文的主要贡献和创新点

### 解决的问题
文本对抗攻击通过微小且人类难以察觉的扰动误导深度学习模型，对 NLP 系统构成严重安全威胁。现有的**对抗样本检测方法**通常依赖以下至少一种不切实际的假设：
- 需要已知的对抗样本进行训练；
- 白盒访问目标模型（如梯度或内部特征）；
- 大量查询受害者模型。

这些限制使其在真实场景中难以部署，尤其是在商业 NLP API、资源受限或隐私敏感环境中。

### 提出的新方法与新思路
本文提出 **RTD-Guard**，一种全新的**黑盒（black-box）文本对抗检测框架**，其核心思想是：
> 利用预训练语言模型中的 **Replaced Token Detection (RTD)** 任务所具备的能力来识别对抗性扰动。

具体而言：
- 对抗攻击中的词替换操作与 RTD 任务的目标高度相似：即识别“被替换的、上下文不一致的 token”。
- 因此，可以直接使用一个未经微调的现成 RTD 判别器（如 ELECTRA 的 discriminator），定位输入中“可疑”的被替换 token。
- 通过对这些 token 进行掩码（masking），观察受害者模型预测置信度的变化，从而判断是否为对抗样本。

该方法完全无需对抗样本训练、无需白盒访问、仅需两次黑盒查询。

### 相比现有方法的优势
| 维度 | RTD-Guard 的优势 |
|------|----------------|
| **实用性** | 满足严格黑盒条件：无训练数据、无模型参数/梯度访问、极低查询开销（仅 2 次） |
| **效率** | 查询复杂度为常数 $O(1)$，运行时间最短，在生产系统中可实时部署 |
| **通用性** | 不依赖特定攻击模式，对多种 state-of-the-art 攻击均有效 |
| **模块化设计** | 可作为“插件式防护层”（plug-and-play guard）集成到任意 NLP 系统前端 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验基于标准对抗基准 RDE (Yoo et al., 2022) 构建，涵盖三个主流文本分类数据集：

| 数据集 | 任务 | 类别数 | 测试样本数（原始 / 对抗） | 中位长度 |
|--------|------|--------|----------------------------|----------|
| **IMDB** | 影评情感分类 | 2 | 25K / 10K | 161 |
| **AG-News** | 新闻主题分类 | 4 | 7.6K / 7.6K | 44 |
| **Yelp** | 餐厅评论情感分类 | 2 | 38K / 5K | 152 |

所有对抗样本均由 TEXTATTACK 框架生成，针对 fine-tuned BERT 模型发起攻击。

### 实验设置和评估指标

#### 攻击方法（Word-Level）
- **TextFooler**
- **PWWS**
- **BAE**
- **TF-adj**

选择理由：这类攻击在隐蔽性、成功率和语义保持方面达到最佳平衡，是最具现实威胁的攻击形式。

#### 评估指标
- **AUC**：ROC 曲线下面积，衡量整体区分能力
- **F1-Score**：精确率与召回率的调和平均
- **TPR@10%FPR (TPR10)**：在 10% 假阳性率下的真阳性率，反映低误报场景下的检测灵敏度

#### 基线方法对比
| 类型 | 方法 | 是否满足黑盒？ | 是否需要训练数据？ | 查询复杂度 |
|------|------|----------------|--------------------|------------|
| 外部 LM | **PPL** (GPT-2 Perplexity) | ✅ | ❌ | $O(1)$ |
| 静态启发式 | **FGWS** (频率替换) | ✅ | ❌ | $O(1)$ |
| 白盒密度估计 | **MLE**, **RDE** | ❌ | ❌ | $O(1)$ |
| 白盒梯度 | **GradMask** | ❌ | ❌ | $O(1)$ |
| 黑盒高查询 | **WDR**, **VoteTRANS** | ✅ | ❌ | $O(L)$ 或 $O(L \cdot T)$ |

> 注：RTD-Guard 与 PPL 和 FGWS 同属“实用约束下可部署”的方法类别。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（摘自 Table 2）

在 **12 个数据集 × 攻击组合中，RTD-Guard 在 11 个上取得最优性能**，显著优于所有基线：

| 方法 | 平均 AUC ↑ | 平均 F1 ↑ | 平均 TPR10 ↑ |
|------|-----------|-----------|--------------|
| **RTD-Guard** | **98.0** | **95.2** | **98.0** |
| WDR | 96.5 | 94.3 | 96.8 |
| GradMask | 96.0 | 93.8 | 96.2 |
| RDE | 96.3 | 93.0 | 95.8 |
| FGWS | 84.2 | 87.3 | 84.7 |
| PPL | 76.8 | 53.1 | 39.8 |

> ✅ 特别是在语义保留更强的攻击（如 BAE、TF-adj）上，PPL 和 FGWS 性能急剧下降，而 RTD-Guard 依然稳健。

### 与基线方法的对比结果
- **vs. PPL & FGWS**：虽同为轻量级黑盒方法，但 RTD-Guard 性能远超二者（AUC 提升 +10–20 pts），证明其机制更本质地捕捉到了对抗扰动的语言学特征。
- **vs. 白盒方法 (RDE, GradMask)**：尽管后者拥有内部信息优势，RTD-Guard 仍全面超越，说明基于上下文不一致性（contextual inconsistency）的信号比梯度或特征密度更具判别力。
- **vs. 高查询方法 (WDR)**：性能相当甚至更优，但查询成本从 $O(L)$ 降至 $O(1)$，使其真正适用于高吞吐服务。

### 消融实验结果

#### RQ1: 干预策略的影响（Table 3）
比较不同 token 干预方式：
- **[MASK]**（默认）
- **[UNK]**
- **删除 (DEL)**
- **MLM 填充**

✅ 结果显示：**只要定位准确，任何干预都能引发显著 confidence shift**，验证了“精准定位”是关键，“如何干预”次之。

#### RQ2: RTD vs. 梯度定位（Figure 3 & 分析）
- **GradMask** 依赖梯度重要性，在干净样本中会错误屏蔽语义核心词（如情感形容词），导致 clean accuracy 下降。
- **RTD-Guard** 定位的是“上下文不自然”的 token，在干净文本中通常是专有名词或连接词，不影响语义主干。
- ➕ 当增加屏蔽数量 $k$ 时，GradMask 性能下降，而 RTD-Guard 稳定甚至提升 → 表明 RTD 更适合多 token 攻击。

#### RQ3: RTD 模型规模影响（Table 4）
使用不同大小的 ELECTRA discriminator：
- **Small (14M)**：已有良好表现（AUC ~97%）
- **Base (110M)** / **Large (335M)**：逐步提升性能，Large 达到最优
- ⚖️ 权衡建议：可根据部署环境选择合适规模，在精度与延迟间权衡。

---

## 4. 关键结论和发现

### 主要发现
1. **对抗扰动的本质是“上下文不一致性”**  
   成功的对抗替换破坏了语言的自然分布，这种“不自然”正是 RTD 任务所训练识别的模式。

2. **RTD discriminator 是天然的对抗扰动探测器**  
   其预训练目标与对抗攻击机制存在结构性对称，因此无需微调即可迁移用于检测。

3. **信心变化（confidence shift）是可靠检测信号**  
   对抗样本对关键 token 的屏蔽极为敏感，而正常样本鲁棒性强，这一差异构成了强判别依据。

4. **无需复杂机制也能实现 SOTA 效果**  
   仅通过“定位 + 屏蔽 + 观察信心变化”三步，即可构建高效、稳定、通用的防御体系。

### 方法的局限性
- **语言依赖性**：当前依赖单语 RTD 模型（如英文 ELECTRA），尚不能直接推广至多语言或跨语言场景。
- **非替换类攻击可能失效**：对于字符级扰动（typo）或句级改写（paraphrase）等非 token 替换攻击，效果未知。
- **需要高质量 RTD 模型**：性能依赖于外部 RTD 判别器的质量，若无合适模型则无法应用。

### 未来工作方向
1. **扩展至大语言模型（LLM）安全场景**
   - 探索用于检测 **prompt injection** 和 **jailbreak attacks**
   - 利用一致性信号识别隐藏指令或角色劫持意图

2. **幻觉检测（Hallucination Detection）**
   - 结合反事实掩码与响应差异，识别生成内容中的事实不一致或不可验证主张

3. **多语言与跨语言 RTD-Guard**
   - 开发支持 multilingual RTD 判别器，提升国际化适用性

4. **与其他防御机制融合**
   - 将 RTD-Guard 作为前置过滤模块，与 robust training 或 input reconstruction 联合使用，形成纵深防御体系

--- 

> ✅ **总结一句话**：  
> RTD-Guard 成功将 PLM 内在的 **linguistic capability** 转化为 **security capability**，提出了一种简洁、高效、无需训练、真正可用于现实世界的黑盒对抗检测方案。

</details>

---

### 11. [Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking](https://arxiv.org/abs/2603.12831)

**Authors**: Zizhao Mo, Junlin Chen, Huanle Xu, Chengzhong Xu  
**Category**: cs.DC  
**Published**: 2026-03-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.12831v1  

#### Abstract
Nowadays, service providers often deploy multiple types of LLM services within shared clusters. While the service colocation improves resource utilization, it introduces significant interference risks for latency-sensitive (LS) services-which have strict SLO requirements for inference latency-and se...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking

## 1. 论文的主要贡献和创新点

### 解决的问题
现代大型语言模型（LLM）服务提供商通常在共享集群中部署多种类型的 LLM 服务，包括**Latency-Sensitive (LS)** 和 **Best-Effort (BE)** 两类负载。LS 服务对推理延迟有严格的 SLO 要求（如在线聊天），而 BE 服务则优先级较低、无严格延迟约束（如后台批处理任务）。然而，现有系统（如 Llumnix）通过预留 GPU 内存来隔离两类服务，存在以下问题：

- **资源利用率低**：内存预留机制过于粗粒度，无法有效缓解计算资源（如 SM、Memory Bandwidth）的竞争。
- **性能干扰严重**：BE 服务仍会显著影响 LS 服务的延迟，导致 SLO 违规。
- **BE 吞吐受限**：由于 GPU 可用内存有限，BE 请求难以充分利用空闲算力。

### 提出的新方法：OmniServe
本文提出 **OmniServe**，一种新型的 LLM 推理服务系统，旨在高效利用 CPU 和 GPU 异构资源，在保障 LS 服务 SLO 的前提下最大化 BE 服务吞吐量。其核心创新包括：

#### 创新点 1：Attention Piggybacking 机制
- **核心思想**：将 BE 服务的 **Attention 计算异步卸载到 CPU** 上执行，而 Dense 模块（如 MLP、proj）仍在 GPU 上执行。
- **解耦执行流**：GPU 不再等待 CPU 返回 Attention 结果，而是继续处理 LS 请求或其他已就绪的请求，避免因 PCIe 通信延迟阻塞 GPU。
- **层级别批处理（Layer-wise Batching）**：当后续层需要执行 Dense 模块时，若对应的 CPU 已完成 Attention 计算，则“顺带”（piggyback）将其加入当前 GPU 批次中处理，实现非同步融合计算。

> ✅ 优势：打破传统 token-wise batching 的同步依赖，允许 CPU 和 GPU 并行运行不同阶段的计算，显著降低干扰。

#### 创新点 2：动态批处理控制策略
- **模块级延迟建模**：构建高精度的 **fA(·)**（Attention）和 **fD(·)**（Dense）延迟预测模型，支持细粒度资源调度。
- **显式 SLO 控制**：基于延迟模型动态决定每批次可容纳的 LS 和 BE 请求数量，并控制可被 piggyback 的 BE 请求上限。
- **分层调度策略**：
  - 优先级顺序：LS decoding > LS chunk prefill > BE chunk prefill > BE decoding
  - 支持 admission control 和 chunk prefill 控制，确保 TTFT 和 TPOT SLO 得到满足。

> ✅ 优势：实现 SLO-aware 的弹性调度，在波动负载下仍能维持高性能与稳定性。

---

## 2. 核心实验方法和设置

### 数据集
- **LS 服务（Latency-Sensitive）**：
  - 模拟真实对话场景，使用 **ShareGPT** 中提取的查询长度分布。
  - 请求以泊松到达模式提交，默认速率变化范围为 1–8 req/s（Yi-34B）、1–5 req/s（Llama-70B）。
- **BE 服务（Best-Effort）**：
  - **LongBench-v2**：平均输入/输出长度为 8,952 / 136 tokens，最大上下文达 12K。
  - **DailyMails**：平均输入/输出长度为 1,964 / 397 tokens，模拟中等长度文本生成任务。

### 实验设置
- **硬件平台**：
  - 主节点：配备 4× A100 80GB GPU + Intel Xeon Gold 6342 CPU（24核）
  - 辅助节点：4 个纯 CPU 服务器（同款 CPU，各 400GB RAM）
  - 网络互联：默认 100 Gbps RoCE，部分测试使用 10 Gbps LAN
- **模型配置**：
  - **Yi-34B**：采用 2-way Tensor Parallelism（TP）
  - **Llama-2-70B**：采用 4-way TP
- **精度格式**：BF16（GPU 和 CPU 均使用）

### 评估指标
| 服务类型 | 主要指标 |
|--------|---------|
| LS 服务 | **SLO Attainment Rate**（基于 TTFT ≤ 2s / 3s，TPOT ≤ 0.2s / 0.25s） |
| BE 服务 | **Token Generation Throughput**（decoding 阶段） |

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Baseline A: Llumnix + vLLM on CPU** | 在 GPU 上运行 Llumnix 实现内存隔离；超出容量的 BE 请求卸载至 CPU 上的 vLLM |
| **Baseline B: NEO** | 将所有解码阶段的 Attention 卸载到 CPU，采用流水线方式执行，但未区分 LS/BE 优先级 |
| **Baseline C: Sarathi-Serve** | 纯 GPU 架构下的 SLO 最优方案，不使用 CPU，仅排队处理溢出 BE 请求 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（综合表现）
| 指标 | OmniServe vs. 最佳基线 |
|------|------------------------|
| **LS 服务 SLO 达成率提升** | 最高达 **1.48×**（TPOT SLO 下降至 0.15s 时） |
| **BE 服务吞吐提升** | 最高达 **9.85×**（重负载 + LongBench-v2） |
| **轻负载下 BE 吞吐提升** | 约 **1.2×**（GPU 资源充足） |
| **低带宽环境（10Gbps）下 BE 吞吐** | 仍可达 **9.1×** 提升，证明通信开销极小 |

### 详细对比结果

#### （1）SLO 保证能力（Fig. 10–14）
- 在各种请求到达率和 SLO 约束下，OmniServe 的 SLO 达成率始终接近 **Sarathi-Serve（纯 GPU 最优）**，仅下降约 **0.6%**。
- 相比之下，**Llumnix** 在高负载下 SLO 达成率急剧下降（尤其在 TPOT < 0.2s 时），最高降幅超过 30%。
- **NEO** 因将所有 Attention 卸载至 CPU，受 PCIe 带宽限制明显，SLO 表现更差，且随 CPU 数量增加反而恶化。

#### （2）BE 吞吐能力（Fig. 15–17）
- 在 **1G4C 配置**（1 GPU Server + 4 CPU Servers）下：
  - 使用 **LongBench-v2** 数据集，BE decoding 吞吐达到 **9.85×** 于 Llumnix。
  - 即使只启用 GPU 主机上的 CPU，也能获得 **3.47×** 提升。
- 在 **DailyMails** 场景下（较短上下文），仍实现 **最高 9.1×** 的吞吐增益。
- 在 **10 Gbps 网络**下性能损失极小，说明 OmniServe 的通信设计高效。

#### （3）消融实验结果

##### ✅ 模型准确性（Table 2）
- 延迟预测模型平均准确率达：
  - **Yi-34B**: **95.7%**
  - **Llama-70B**: **94.5%**
- P90 准确率均高于 **92.7%**，表明模型具有良好的一致性与泛化能力。

##### ✅ Admission Control 效果（Fig. 19b-c）
- 启用 admission control 后，TTFT SLO 达成率从不足 60% 提升至 **94.1%**。
- LS decoding 吞吐几乎不受影响（差异 < 6%），说明调度策略高效平衡了公平性与利用率。

##### ✅ Attention Piggybacking 开销分析（Fig. 19a）
- 辅助操作（queue 读写、residual 存取）延迟极低：
  - queue 操作 ≤ 75 μs（400 请求并发）
  - residual 加载 ~0.5 ms（非连续访问所致）
- 总体开销远小于推理时间，且频率受控，实际影响可忽略。

##### ✅ 分布式 CPU 扩展性（Fig. 18a）
- 随着接入更多 CPU 节点，BE 吞吐呈近线性增长，最多实现 **3.13×** 加速。
- LS 服务 TPOT 延迟保持稳定（Fig. 18b），验证了异步机制的有效性。

---

## 4. 关键结论和发现

### 主要发现
1. **CPU 资源可用于高效增强 BE 服务**：尽管 CPU 在 Dense 计算上远弱于 GPU（Table 1 显示差距达 498.1×），但其丰富的内存和闲置算力非常适合承担 BE 的 Attention 计算。
2. **异步解耦是关键**：传统的同步卸载方式会导致 GPU 阻塞。OmniServe 的 **Attention Piggybacking** 机制通过异步执行 + 层级批处理，彻底解除依赖，实现了真正的并行。
3. **细粒度建模驱动 SLO 保障**：基于模块级别的延迟建模，使得系统能够精确量化干扰，从而在动态负载下依然维持 LS 服务的 SLO。
4. **OmniServe 具备良好兼容性**：
   - 支持 **Prefill-Decode disaggregation**（PD 分离架构）
   - 兼容 **MoE、LoRA** 等先进模型结构
   - 可集成进主流框架（如 vLLM）

### 方法的局限性
- **依赖 PCIe 带宽**：虽然通信量小，但在极端低带宽环境下（如 < 10Gbps）可能成为瓶颈。
- **不适合全 CPU 推理场景**：假设 GPU 可承载完整模型参数，不适用于参数无法放入 GPU 显存的情况（此类问题由 HeteGen 等系统解决）。
- **复杂性增加**：引入 CPU-GPU 协同逻辑增加了系统实现和调试难度。

### 未来工作方向
- **支持多优先级队列**：扩展至多个 SLO 等级的服务共存，按等级设定不同的 piggyback 延迟容忍窗口。
- **结合 AMX/AVX-512 加速 Dense 计算**：在具备高级指令集的 CPU 上进一步加速 BE 的部分 Dense 模块。
- **跨数据中心扩展**：探索在广域网环境下进行远程 CPU 卸载的可能性。
- **自动调优 Piggybacking 参数**：引入 RL 或 ML 方法动态优化每一层的 piggyback 数量。

---

> 📌 **总结一句话**：  
> OmniServe 通过 **异步 Attention Piggybacking + 动态层批处理**，首次实现了在保障 LS 服务 SLO 的同时，将 BE 服务吞吐提升近一个数量级，为混合 LLM 负载的高效调度提供了全新范式。

</details>

---

### 12. [No More DeLuLu: Physics-Inspired Kernel Networks for Geometrically-Grounded Neural Computation](https://arxiv.org/abs/2603.12276)

**Authors**: Taha Bouhsine  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.12276v1  

#### Abstract
We introduce the yat-product, a kernel operator combining quadratic alignment with inverse-square proximity. We prove it is a Mercer kernel, analytic, Lipschitz on bounded domains, and self-regularizing, admitting a unique RKHS embedding. Neural Matter Networks (NMNs) use yat-product as the sole non...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：No More DeLuLu: Physics-Inspired Kernel Networks for Geometrically-Grounded Neural Computation

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现代神经网络将**几何计算**（如点积）与**非线性激活**（如 ReLU）分离，这种设计导致以下问题：
- **信息丢失**：负值被截断为零（如 ReLU），破坏输入间的连续关系。
- **依赖额外模块**：需要 BatchNorm、LayerNorm 等显式归一化层来稳定训练，增加复杂性和内存开销。
- **高维饱和**：传统 RBF 类核函数在高维空间中响应迅速衰减，难以扩展。

### 提出了什么新方法或新思路
提出 **E-product** 和基于其构建的 **Neural Matter Networks (NMNs)**：

#### ✅ E-product（核心算子）
定义如下：
$$
E(w, x) = \frac{(w^\top x)^2}{\|w - x\|^2 + \epsilon}
$$
- 结合了**对齐性**（squared dot product）与**邻近性**（inverse-square distance），形成“势阱”（potential well）结构。
- 是一个 **Mercer kernel**，具备理论完备性（analytic, Lipschitz, self-regularizing）。
- 非线性来源于其**几何结构本身**，而非独立的激活函数。

#### ✅ Neural Matter Networks (NMNs)
- 使用 E-product 作为唯一的非线性机制，替代传统的 `Linear + Activation + Normalization` 模块。
- 在 Transformer 中实现为 **Aether-GPT2**：用 E-based attention 和 E-MLP 替代原始组件。
- **无需任何 LayerNorm 或显式归一化层**，因 E-product 自带内在归一化（self-regulation）。

### 相比现有方法的优势
| 维度 | 优势 |
|------|------|
| **架构简化** | 移除 ReLU/GELU 和 LayerNorm，减少模块数量 |
| **内存效率** | 减少 15–25% 内存占用（无需存储激活值） |
| **梯度稳定性** | 梯度随距离衰减（vanish for outliers），天然抗扰动 |
| **表达能力** | 单个 E-unit 可解决 XOR（非线性可分问题） |
| **理论基础** | 具备 RKHS 表示、通用逼近性、NTK 分析支持 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 任务 | 数据集 |
|------|--------|
| 图像分类 | MNIST（手写数字识别） |
| 极端分类 | Eurlex-4K（法律文本多标签分类） |
| 语言建模 | FineWeb（2.5B tokens，用于训练 GPT-2/Aether-GPT2） |

### 实验设置和评估指标

| 任务 | 设置 | 评估指标 |
|------|------|----------|
| **MNIST** | 10-neuron 分类器，Adam(lr=0.001)，5 轮训练 | Accuracy, prototype evolution, superposition robustness |
| **Eurlex-4K** | 极端分类基准测试 | P@1, P@3, P@5, PSP@1-5（propensity-scored precision） |
| **Language Modeling** | Aether-GPT2 (124M params)，训练 2.5B tokens | Train/Val Loss, Throughput (tok/s), Calibration, NTK spectrum |
| **消融实验** | 对比是否加入 LayerNorm | Validation Loss 收敛情况 |

### 基线方法对比
| 方法 | 对比对象 |
|------|---------|
| E-product Classifier | Linear classifier（相同结构，仅替换为点积） |
| Aether-GPT2 | GPT-2（参数量相近，相同超参） |
| E-MHA + NMN | Scaled Dot-Product Attention + GeLU MLP |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### ✅ MNIST 分类结果（10-neuron classifier）
| 方法 | Accuracy | △Prototype Magnitude | Scaling Factor $ \sigma $ |
|------|----------|------------------------|----------------------------|
| Linear | 92.08% | +13.8%（增长） | —— |
| E-product (NMN) | **92.38%** | **-4.5%**（收缩） | 1 → **2.68** |

- **原型演化更稳定**：E-product 权重不会无限增长，符合 self-regulation 理论预测。
- **符号翻转鲁棒性强**（Superposition Robustness）：
  - Linear：反转权重后准确率从 92.04% → **0.01%**
  - E-product：从 92.18% → **87.87%**

#### ✅ 极端分类（Eurlex-4K）
| 方法 | P@1 | P@3 | P@5 | PSP@1 | PSP@3 | PSP@5 |
|------|-----|-----|-----|-------|-------|-------|
| E-product | **0.6465** | **0.5114** | **0.4271** | 1.1542 | 0.9664 | 0.8443 |
| Inner Product | 0.6235 | 0.5041 | 0.4125 | **1.2117** | **1.0215** | **0.8587** |

- E-product 在主流指标 **P@k 上全面领先**，说明其更适合 top-k 检索任务。
- Inner Product 在 PSP@k 更高，表明其对稀有类别倾向估计更强，但整体精度较低。

#### ✅ 语言模型（Aether-GPT2 vs GPT-2）
| 指标 | GPT-2 | Aether-GPT2 | 提升 |
|------|--------|--------------|------|
| Final Train Loss | 4.1969 | **4.0479** | ↓ 3.5% |
| Final Val Loss | 4.6417 | **4.5747** | ↓ **1.45%** |
| Memory Usage | — | **↓15–25%** | 显著降低 |
| Throughput | comparable | comparable | 无损失 |
| LayerNorm | Yes | **No** | 完全移除 |

- **验证损失更低**，且训练更稳定（尤其在 BF16 混合精度下）。
- **校准性更好**（calibration curve 更接近对角线），预测置信度更可靠。

#### ✅ 消融实验：LayerNorm 影响
| 配置 | Val Loss | 状态 |
|------|----------|------|
| Aether (no LayerNorm) | 4.5747 | 收敛成功 |
| Aether + Pre-LN | >10 | 发散 |
| Aether + Post-LN | >10 | 发散 |

> ❗ **LayerNorm 与 E-product 不兼容**：因其强制输入归一化会破坏 E-product 所依赖的 proximity 几何结构。

---

## 4. 关键结论和发现

### 主要发现
1. **E-product 是一种物理启发的统一算子**：
   - 将 alignment 与 proximity 融合，模拟物理中的逆平方律作用力（如引力、静电力）。
   - 天然具备非线性、局部响应、边界敏感等特性。

2. **NMNs 实现了“归一化内嵌”**：
   - 通过分母 $\|w - x\|^2 + \epsilon$ 实现自调节（self-regulation），输出有界。
   - **无需 LayerNorm/BatchNorm**，反而引入它们会导致训练崩溃。

3. **理论性质优越**：
   - 是 Mercer kernel ⇒ 存在唯一 RKHS 表示。
   - 具备 universal approximation 能力。
   - 梯度稳定、Lipschitz 连续、analytic，在优化和泛化上有保障。

4. **实际性能媲美甚至超越标准模型**：
   - 在 MNIST 上达到线性分类器水平；
   - 在极端分类中提升 P@k；
   - 在语言建模中以相同参数量取得更低 loss。

### 方法的局限性
| 局限 | 说明 |
|------|------|
| **FLOPs 开销较高** | E-product 前向计算约为 Linear+ReLU 的 **2× FLOPs**，虽可通过优化缓解，但仍高于传统线性层。 |
| **对 $\epsilon$ 敏感** | 正则项 $\epsilon$ 控制响应尖锐程度，需调优；太小易数值不稳定，太大削弱局部性。 |
| **尚未验证更大规模模型** | 当前实验集中于 124M GPT-2 规模，未展示在百亿级以上模型的表现。 |
| **不适用于所有任务范式** | 如需严格概率输出或熵最大化场景，可能仍需配合 softmax 或其他机制。 |

### 未来工作方向
1. **探索更多物理启发算子**：如结合电磁场张量、波动方程等构造新型 kernel operators。
2. **扩展到图神经网络与扩散模型**：利用 E-product 的几何敏感性改进 GNN 消息传递或生成模型 latent dynamics。
3. **硬件友好实现**：开发专用 kernel fusion 和低精度加速方案，进一步缩小与传统线性层的速度差距。
4. **动态 $\epsilon$ 调整机制**：设计自适应 $\epsilon$ 策略，根据输入分布或训练阶段自动调整稳定性与灵敏度。
5. **与其他 kernel 方法融合**：研究如何将 E-product 与 Random Fourier Features 或 Nystrom 方法结合，用于大规模 kernel approximation。

---

> 🔚 **总结一句话**：  
> 本文提出了一种**将物理规律融入神经计算**的新范式——通过 E-product 统一对齐与邻近性，使神经网络在保持强大表达力的同时，实现**架构极简、梯度稳定、无需归一化**，为下一代几何感知神经架构提供了坚实基础。

</details>

---

### 13. [Adaptive Diffusion Posterior Sampling for Data and Model Fusion of Complex Nonlinear Dynamical Systems](https://arxiv.org/abs/2603.12635)

**Authors**: Dibyajyoti Chakraborty, Hojin Kim, Romit Maulik  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.12635v1  

#### Abstract
High-fidelity numerical simulations of chaotic, high dimensional nonlinear dynamical systems are computationally expensive, necessitating the development of efficient surrogate models. Most surrogate models for such systems are deterministic, for example when neural operators are involved. However, ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Adaptive Diffusion Posterior Sampling for Data and Model Fusion of Complex Nonlinear Dynamical Systems

---

## 1. 论文的主要贡献和创新点

### 解决的问题
高维、混沌的非线性动力系统（如湍流）的高保真数值模拟计算成本极高，传统**确定性 surrogate 模型**（如基于神经算子的方法）虽然加速了预测，但存在以下根本缺陷：
- 无法捕捉混沌系统的**内在分布不确定性**；
- 长期预测中误差呈指数增长；
- 缺乏统一框架进行**自适应传感器放置**与**实时数据同化**。

此外，现有传感器布局方法多为静态或依赖昂贵在线优化，难以在复杂几何和动态演化系统中实现高效闭环反馈。

### 提出的新方法与创新思路
本文提出一个**基于扩散模型的统一概率建模框架**，实现了长期稳定预测、自适应传感与无需重训练的数据融合。其四大核心贡献如下：

#### (i) 多步自回归扩散训练目标（Multi-step Autoregressive Diffusion Objective）
- **方法**：在训练阶段引入 $K$-step rollout 的多步损失函数，而非传统的单步去噪目标。
- **优势**：显著提升长期 rollout 的稳定性，缓解误差累积问题。

#### (ii) 多尺度图变换器架构（Multi-scale Graph Transformer with Diffusion Preconditioning）
- **方法**：采用基于图神经网络的 U-Net 架构，结合：
  - **EDM preconditioning**（提升训练稳定性）
  - **AdaLN-Zero conditioning**（增强条件控制）
  - **Voxel-grid pooling**（处理非结构化网格上的多尺度特征）
- **优势**：适用于任意拓扑结构（如三角形有限元网格），支持不规则域上的物理场建模。

#### (iii) 自适应传感器放置策略（Adaptive Sensor Placement）
提出两种互补策略：
- **不确定性驱动法（Uncertainty-driven）**：利用扩散模型生成的**集成预测方差**作为空间不确定性度量，指导传感器部署。
- **学习型误差预测法（Predictive Model-based）**：训练一个轻量级元模型（error predictor network）来估计主扩散模型的重建误差，用于快速决策。

#### (iv) 拓扑感知贪婪选择算法（Topology-aware Greedy Selection）
- 在传感器选址过程中加入**空间抑制机制**（spatial suppression），确保最小传感器间距，防止在局部高不确定区域过度聚集，最大化信息增益。

### 相比现有方法的优势
| 维度 | 本工作 | 传统方法 |
|------|--------|----------|
| 不确定性建模 | ✅ 完整后验分布采样 | ❌ 仅点预测（point prediction） |
| 长期预测稳定性 | ✅ 多步训练减少误差累积 | ⚠️ 单步训练易发散 |
| 数据同化 | ✅ Diffusion Posterior Sampling（无需重训练） | ❌ 需微调或重新训练 |
| 传感器放置 | ✅ 动态、自适应、闭环 | ❌ 固定位置或离线设计 |
| 几何适应性 | ✅ 支持非结构化网格（graph-based） | ⚠️ 多限于规则网格 |

---

## 2. 核心实验方法和设置

### 使用的数据集
1. **二维受迫各向同性湍流（2D Forced Homogeneous Isotropic Turbulence, HIT）**
   - Reynolds 数：Re = 1000
   - 域大小：$L=2$，周期边界
   - 网格分辨率：原始 512×512 → 下采样至 64×64
   - 时间间隔：$\Delta t = 256 \Delta t_{\text{DNS}}$
   - 快照数量：共 800 帧，前 100 帧丢弃，使用后续 700 帧

2. **后向台阶流动（Flow over a Backward-Facing Step, BFS）**
   - Reynolds 数：Re = 26,000
   - 数值方法：Large Eddy Simulation (LES) + Smagorinsky SGS model
   - 网格类型：非结构化三角形 Taylor-Hood 元素（共 5,253 个）
   - 时间步长：$\Delta t = 1\times10^{-4}$ s
   - 快照频率：每 $10^{-4}$ 秒保存一次，共 800 帧，前 100 帧丢弃

### 实验设置与评估指标
| 设置项 | 描述 |
|-------|------|
| **模型架构** | Diffusion-based Graph Transformer + EDM preconditioning |
| **训练方式** | 对比 single-step vs multi-step rollout 训练 |
| **传感器预算** | 固定数量 $s \in \{20, 50, 100, 200\}$ |
| **最小间距约束** | $g = 10$ 或 $g = 15$（HIT）；$g = 5$ 或 $g = 7$（BFS） |
| **评估指标** | - **Mean Absolute Error (MAE)**：预测速度场与 DNS 的差异<br>- **Reynolds Stress Profiles**：高阶统计量恢复能力<br>- **Sampling Time**：推理耗时比较 |

### 基线方法对比
- **Random Placement**：随机选择传感器位置（基准）
- **No Sensors**：无观测输入
- **Std-based (Uncertainty-driven)**：基于扩散集成的标准差选择
- **Predicted Sensors**：基于误差预测网络的选择
- **Single-step Diffusion**：标准单步训练扩散模型

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果

#### ✅ 多步训练显著优于单步训练
- **HIT 场景（Fig. 2）**：
  - Multi-step 在 $t=50$ 时 MAE ≈ 0.25，而 Single-step 超过 1.75
- **BFS 场景（Fig. 9）**：
  - Multi-step 错误持续下降并趋于平稳，Single-step 在 $t > 40$ 后迅速发散

> **结论**：multi-step autoregressive objective 显著增强了长期 rollout 的稳定性。

#### ✅ 自适应传感器提升预测精度
- **HIT（Fig. 4）**：
  - Std-based 和 Predicted placement 均优于 Random
  - Std-based 表现最佳（MAE 最低）
- **BFS（Fig. 11）**：
  - 所有带传感器配置均明显优于无传感器
  - Std-based 与 Predicted 明显优于 Random，尤其在早期阶段

> **结论**：将观测融入 posterior sampling 可有效校正预测漂移。

#### ✅ 不同传感器密度与间距的影响
- **更多传感器 + 更大间隔 ⇒ 更好性能**
  - HIT 中，$s=200$, $g=15$ 时误差最低（Fig. 5）
- **但 BFS 存在权衡**：
  - 当 $s$ 较小时（如 20），较大 $g=7$ 更优
  - 当 $s$ 较大时（如 100），较小 $g=5$ 更能覆盖局部结构（Fig. 12）

> **结论**：最优 $g$ 依赖于流场的空间相干性和传感器预算。

#### ✅ 高阶统计量恢复更敏感于传感器策略
- **Reynolds 应力剖面（Fig. 15）**：
  - 所有传感器策略均改善应力峰值定位
  - **Std-based > Predicted > Random**
  - 差异在 mean velocity 上不明显，但在 turbulence statistics 上显著

> **结论**：不确定性引导的传感器更能捕捉瞬态涡旋结构。

#### ✅ 消融实验：传感器策略有效性验证
| 策略 | 性能 | 推理开销 |
|------|------|---------|
| **Random** | 最差 | 低 |
| **Predicted (error net)** | 中等偏上 | **极低**（无需 ensemble） |
| **Std-based (ensemble)** | **最优** | 高（需运行多个样本） |

- **Fig. 16 显示**：两者采样时间随 $s$ 近似线性增长，但 Predicted 方法因免去 ensemble 生成，在资源受限场景更具优势。

---

## 4. 关键结论和发现

### 主要发现
1. **多步扩散训练是实现长期稳定预测的关键**  
   相比单步训练，multi-step rollout 显著减缓误差传播，使模型可在数十步内保持物理一致性。

2. **扩散模型天然适合作为贝叶斯先验用于数据同化**  
   利用 **Diffusion Posterior Sampling (SDA/DPS)** 可无缝融合稀疏观测，且**无需重训练模型**，适合实时应用。

3. **不确定性可直接用于闭环主动推理（active inference）**  
   扩散集成方差提供了空间分辨的不确定性地图，成为传感器放置的自然依据。

4. **学习型误差预测器是高效的轻量化替代方案**  
   尽管 ensemble 方差更准确，但训练一个小型 error predictor network 可以近似其行为，并大幅降低计算成本。

5. **拓扑感知抑制机制防止“传感器簇拥”**  
   强制最小间距提升了空间覆盖率，避免冗余测量，从而最大化信息增益。

### 方法的局限性
- **计算开销较高**：尤其是 uncertainty-driven 方法需要运行多个 diffusion samples。
- **当前实验局限于二维**：三维扩展尚未验证，尽管架构本身支持。
- **未嵌入强物理约束**：虽使用 FEM 数据，但模型未显式编码守恒律（如质量/动量守恒）。
- **传感器动作不可逆**：当前为开环部署，未考虑动态增删传感器的闭环控制。

### 未来工作方向
- 扩展至 **3D 复杂几何与多物理场耦合系统**（如燃烧、多相流）
- 引入 **Physics-Informed Diffusion Models**，在训练或采样阶段加入 PDE 残差约束
- 探索 **可微分传感器重定位策略**，实现完全端到端的 adaptive sensing
- 结合 **real-world sensor noise models**，增强鲁棒性
- 开发 **low-rank approximation techniques** 以加速 posterior sampling

---

> 🔗 **代码开源地址**：[https://github.com/ISCLPurdue/chaos_gen](https://github.com/ISCLPurdue/chaos_gen)

</details>

---

### 14. [3DTCR: A Physics-Based Generative Framework for Vortex-Following 3D Reconstruction to Improve Tropical Cyclone Intensity Forecasting](https://arxiv.org/abs/2603.13049)

**Authors**: Jun Liu, Xiaohui Zhong, Kai Zheng, Jiarui Li, Yifei Li, Tao Zhou, Wenxu Qian, Shun Dai, Ruian Tie, Yangyang Zhao, Hao Li  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.13049v1  

#### Abstract
Tropical cyclone (TC) intensity forecasting remains challenging as current numerical and AI-based weather models fail to satisfactorily represent extreme TC structure and intensity. Although intensity time-series forecasting has achieved significant advances, it outputs intensity sequences rather th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
热带气旋（Tropical Cyclone, TC）强度预报长期以来面临挑战，主要原因如下：
- 当前的数值模型（如 ECMWF-HRES）和 AI-based 天气预测模型（如 FuXi）在表示极端 TC 结构和强度方面存在系统性低估。
- 强度时间序列预测虽有进展，但输出仅为一维强度序列，无法提供三维内核精细结构及驱动 TC 演化的物理机制。
- 高分辨率数值模拟（如 WRF）虽然能捕捉细尺度特征，但计算成本高昂，难以用于大规模业务化应用。

### 提出的新方法与思路
本文提出 **3DTCR**（3D Tropical Cyclone Reconstruction），一种基于物理约束的生成式框架，用于实现涡旋跟随式的三维重建以提升 TC 强度预报能力。其核心创新包括：

- **Region-adaptive vortex-following 架构**：采用动态移动区域策略，在一个以 TC 中心为中心的 $10^\circ \times 10^\circ$ 局部区域内进行高分辨率重建，避免对整个大域进行昂贵模拟。
- **Conditional Flow Matching (CFM)**：基于 Rectified Flow 的生成模型，高效学习从低分辨率输入到高分辨率 TC 内核结构的非线性映射。
- **Two-stage Transfer Learning + Latent Domain Adaptation**：
  - 第一阶段：使用 ERA5 数据预训练，学习基础物理降尺度关系；
  - 第二阶段：引入 MMD（Maximum Mean Discrepancy）损失函数，实现潜空间中的领域对齐，缓解因预报误差累积导致的空间模式偏移（pattern shift）。
- **Physics-informed 生成建模**：训练数据来自高分辨率 WRF 动力模拟（3-km），使模型隐式学习符合动力平衡的 TC 结构，保证生成场的热力学与运动学一致性。

### 相比现有方法的优势
| 方面 | 传统方法局限 | 3DTCR 改进 |
|------|---------------|------------|
| **分辨率** | 全球模型（~25 km）无法解析眼墙、RMW 等关键结构 | 实现 3-km 分辨率重建，恢复细尺度梯度 |
| **效率** | WRF 移动嵌套模拟需数百小时 CPU 时间 | 单 GPU 上分钟级完成重建，加速数个数量级 |
| **极端值建模** | 确定性模型易过平滑，低估 RI 和 Category 5 风暴 | CFM 联合 MMD 提升长尾分布建模能力，显著改善极端风速再现 |
| **实用性** | AI 模型受“double penalty”影响严重 | 通过潜空间适配解耦强度与相位学习，增强鲁棒性 |

---

## 2. 核心实验方法和设置

### 使用的数据集
| 数据类型 | 名称 | 描述 |
|--------|------|------|
| **高分辨率真值** | **WRF-3km** | 自建数据集，基于 WRF 模型在 $10^\circ \times 10^\circ$ 动态移动区域上运行，空间分辨率为 3 km（约 0.027°），时间覆盖 2018–2024 年，共 1,903 个样本，包含 93 个变量，最终选取 **21 个多层气象变量**作为目标字段。 |
| **低分辨率输入** | **FuXi** | 来自阿里巴巴通义千问团队的全球 AI 天气预报系统，分辨率为 0.25°，用作主要输入源之一。 |
| | **ERA5** | ECMWF 再分析数据，0.25° 分辨率，用于预训练阶段。 |
| **观测基准** | **IBTrACS** | 国际最佳路径档案，用于验证 TC 路径和最大风速（WS10M）。 |
| | **ECMWF-HRES** | 欧洲中期天气预报中心高分辨率预报系统，作为业务级对比基线。 |

### 实验设置
- **训练/测试划分**：2018–2023 年为训练集，2024 年西北太平洋 TC 季节为独立测试集（含 2,242 个样本）。
- **网络架构**：U-Net 主干 + Attention 模块，输入维度为 $42 \times 370 \times 370$（噪声 + 条件场），输出 $21 \times 370 \times 370$ 高分辨率场。
- **条件变量**：包括 10 米风速分量（U10M/V10M）、海平面气压（MSL）、2 米温度（T2M）、各层温湿风等共 21 变量；同时引入 forecast lead time 作为全局条件。

### 评估指标
| 指标 | 用途 |
|-----|------|
| **RMSE / Bias (WS10M)** | 衡量最大 10 米风速的整体误差 |
| **CSI (Critical Success Index)** | 在多个阈值下（如 >32.7 m/s）评估极端风事件检测能力，更关注长尾表现 |
| **Energy Spectrum / PDF Analysis** | 分析多尺度能量分布与风速概率密度，检验是否恢复高频细节 |
| **Azimuthal Fourier Decomposition** | 评估对称与非对称结构（如螺旋雨带）的重建质量 |
| **Scatter Plot vs IBTrACS** | 可视化预测与实测风速的相关性 |

### 基线方法对比
- **FuXi**：当前先进的 AIWP 模型
- **ERA5**：常用再分析数据
- **ECMWF-HRES**：主流业务数值预报系统
- **WRF-3km**：高分辨率仿真结果，视为“伪真实”

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）
在 **120 小时（5天）预报时效内**，3DTCR 相较于主要基线显著提升性能：

| 方法 | 24h RMSE | 48h RMSE | 72h RMSE | 96h RMSE | 120h RMSE |
|------|----------|----------|----------|----------|-----------|
| FuXi | 20.19 | 21.64 | 22.80 | 22.94 | 23.91 |
| ERA5 | 18.92 | 19.77 | 20.04 | 20.17 | 21.20 |
| ECMWF-HRES | 13.96 | 14.90 | 15.63 | 16.68 | **19.12** |
| **3DTCR-SFT(FuXi)** | **12.75** | **13.37** | **15.72** | **16.34** | **14.83** |
| WRF (目标) | 10.95 | 11.47 | 11.89 | 11.28 | 12.13 |

> ✅ **相对 FuXi 输入，RMSE 平均降低 36.5%，Bias 减少 60.7%**  
> ✅ **相比 ECMWF-HRES，RMSE 平均降低 8.9%，Bias 减少 28.2%**

### 与基线方法的对比结果
- 在所有 lead times 下，3DTCR 输出的散点图更接近 1:1 线，尤其在高强度区间（>50 m/s）明显减少低估现象（见 Fig. 1）。
- 误差增长缓慢：在 24–120 小时窗口中，3DTCR 的 RMSE 维持在 13–14 m/s，远低于 FuXi 快速上升的趋势（超过 20 m/s）。
- 成功重建 Super Typhoon KONG-REY（2024）的结构，包括清晰的眼区、紧密的眼壁、小尺度螺旋带等（见 Fig. 3）。

### 消融实验结果（Ablation Study）
#### （1）Two-stage Training vs End-to-End
- **3DTCR-E2E(w/MMD)**：端到端训练，尽管使用多任务优化，在极端阈值下 CSI 显著下降。
- **3DTCR-SFT(FuXi)**：两阶段微调模型在所有 CSI 阈值下均优于 E2E，特别是在 **WS10M > 32.7 m/s（Category 1 上限）** 时优势明显（见 Fig. 5）。

#### （2）Latent Domain Adaptation（MMD）的作用
- **3DTCR-SFT(w/o MMD)**：未使用 MMD 对齐潜空间，CSI 性能大幅下降，说明缺乏领域适应会导致极端结构重建失败。
- 加入 MMD 后，模型能在输入严重偏移的情况下仍保持对眼墙等关键结构的准确重建。

> 🔍 结论：**两阶段训练 + MMD 潜空间对齐 是突破“双罚问题”和提升极端事件建模的关键设计。**

---

## 4. 关键结论和发现

### 主要发现
1. **3DTCR 成功实现了物理一致的高分辨率 TC 内核重建**，在保持轨迹稳定性的同时显著提升了强度预报精度。
2. **该方法打破了传统 NWP 与 AI 模型之间的效率-精度权衡**：相比 WRF 模拟提速百倍以上，且性能优于 ECMWF-HRES。
3. **通过将 AI 生成效率与物理约束深度融合**，模型能够合理恢复 TC 的三维环流结构、径向风梯度、非对称扰动及能量谱特性。
4. **Latent domain adaptation 有效缓解了长期预报中的 pattern shift 问题**，使得即使在长 lead time 下也能稳定重建极端强度。

### 方法的局限性
1. **Ground Truth 仍是近似值**：WRF 模拟依赖参数化方案，不同配置会影响强度模拟效果，与真实大气状态仍有差距。
2. **未能完全消除平滑效应**：尽管已有改进，但在极端偏差场景下仍可能出现轻微过度平滑。
3. **工程部署复杂**：当前流程需结合全球 AI 模型 → 区域裁剪 → TC 追踪 → 3DTCR 推理，尚未实现统一 end-to-end 架构。

### 未来工作方向
- 探索 **end-to-end 架构**，直接由粗分辨率场输出精细化 TC 强度与结构，简化部署流程。
- 引入更多 **multi-modal 观测数据**（如卫星、雷达）进一步约束生成过程。
- 扩展至 **其他极端天气系统**（如大气河流、温带气旋）的高分辨率重建。
- 结合不确定性量化（Uncertainty Quantification）发展 **probabilistic 3DTCR** 版本，支持风险预警。

---

> 📌 **总结一句话**：  
> **3DTCR 是首个将物理约束、生成式 AI 与涡旋跟随思想融合的高效三维 TC 重建框架，在显著降低计算成本的同时，全面超越主流业务系统在强度预报上的表现，为下一代 AI-enhanced 气象预报提供了新范式。**

</details>

---

### 15. [Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](https://arxiv.org/abs/2603.12577)

**Authors**: Jia-Chen Zhang, Zhen-Wei Yan, Yu-Jie Xiong, Chun-Ming Xia  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.12577v1  

#### Abstract
Parameter-Efficient Fine-Tuning (PEFT) has become a dominant paradigm for deploying LLMs in multi-task scenarios due to its extreme parameter efficiency. While Mixture-of-Experts (MoE) based LoRA variants have achieved promising results by dynamically routing tokens to different low-rank experts, th...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 《Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation》论文总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
现有的基于 **Mixture-of-Experts (MoE)** 的 **Parameter-Efficient Fine-Tuning (PEFT)** 方法（如 MoE-LoRA）在多任务场景中表现良好，但存在以下关键缺陷：

- **忽略任务复杂度的层次性**：大多数方法采用统一架构的专家（uniform experts），即所有 LoRA 专家具有相同的秩（rank）和容量，无法适应不同任务对特征粒度的不同需求。
- **参数冗余与效率低下**：独立训练多个 LoRA 模块导致参数重复、存储开销大，且难以共享通用语言知识。
- **负迁移（Negative Transfer）严重**：简单任务与复杂任务共用相同结构的适配器，容易造成优化冲突。

例如，Table 1 显示不同任务在不同 LoRA 秩下的最优性能各异，说明“一刀切”的设计不适用于多任务学习。

---

### 提出了什么新方法或新思路
作者提出 **Expert Pyramid Tuning (EPT)**，一种新颖的参数高效微调框架，其核心思想是将计算机视觉中的 **Feature Pyramid Network (FPN)** 思想引入 PEFT，构建一个“参数金字塔”。

#### 主要创新组件包括：

1. **共享元知识子空间（Shared Meta-Knowledge Subspace）**
   - 引入低维可学习矩阵 $ Z_{\text{meta}} = B \cdot A $，作为所有任务共享的语言基础表示。
   - 所有专家从此子空间派生，避免重复学习通用模式。

2. **金字塔投影机制（Pyramid Projection Mechanism）**
   - 使用不同尺度的 **deconvolutional kernels** 将 $ Z_{\text{meta}} $ 投影到多个维度，形成多尺度专家（multi-scale experts）。
   - 小核捕捉局部细粒度语法模式，大核建模全局语义依赖。

3. **自适应 LoRA 剪枝器（Adaptive LoRA Pruner, ALP）**
   - 动态裁剪共享子空间以匹配目标层维度，并通过维度感知缩放因子 $ d_t / T $ 平衡共享参数与任务特定参数的更新频率，提升训练稳定性。

4. **基于对比学习的任务嵌入模块（Contrastive Task Embedding Module）**
   - 为每个任务分配可学习的嵌入向量 $ e_t $。
   - 使用对比损失最大化样本特征与其任务原型之间的互信息，增强路由准确性。

5. **Top-k 路由机制**
   - 动态选择最合适的多尺度专家组合进行推理，实现按需资源分配。

---

### 相比现有方法的优势

| 维度 | EPT 的优势 |
|------|-----------|
| **参数效率** | 参数量显著低于多数 MoE-LoRA 变体（如 EPT: 0.41M vs. MoELoRA: 0.81M per task），同时性能更高 |
| **表达能力** | 多尺度结构能更好适应从简单分类到复杂推理的不同任务需求 |
| **知识共享与区分** | 共享子空间促进正向迁移，任务嵌入增强任务间判别力，缓解负迁移 |
| **训练稳定性** | ALP 和频率补偿机制有效平衡参数更新节奏 |

> ✅ **一句话总结**：EPT 不再为每个任务训练独立 LoRA，而是构建一个“参数金字塔”，通过共享+多尺度重构的方式实现更高效、更灵活的多任务适配。

---

## 2. 核心实验方法和设置

### 使用的数据集

#### （1）自然语言理解任务（NLU）
基于 **GLUE benchmark**：
- **CoLA**（语言可接受性判断）
- **SST-2**（情感分析）
- **MRPC/QQP**（句子对相似性）
- **STS-B**（语义相关性回归）
- **MNLI/QNLI/RTE**（自然语言推理）

#### （2）常识推理任务
- **BoolQ**（是非问答）
- **OBQA**（开放书本问答）
- **ARC-E / ARC-C**（科学常识问答，Easy & Challenge）

数据集统计详见附录 A.1 和 A.2。

---

### 实验设置和评估指标

| 设置项 | 配置 |
|-------|------|
| **主干模型** | T5-base（GLUE）、LLaMA2-7B（常识推理） |
| **优化器** | AdamW |
| **学习率** | 3×10⁻⁴，线性衰减，500步预热 |
| **训练轮数** | 5 epochs |
| **Batch Size** | Global batch size = 32 |
| **序列长度** | 最长 128 tokens |
| **LoRA 配置** | Rank = 8, Alpha = 32 |
| **专家数量** | N = 8，kernel sizes = [2,2,4,4,6,6,8,8] |
| **Top-k 路由** | k = 2 |
| **对比损失权重** | λ = 0.1，温度 T = 0.05 |

#### 评估指标
- **Accuracy**：除特别说明外
- **Matthews Correlation (Mcc)**：CoLA
- **Pearson Correlation**：STS-B

---

### 基线方法对比

| 方法类别 | 对比方法 |
|--------|---------|
| **全量微调** | Fine-tuning |
| **经典 PEFT** | Adapters, PT, LoRA |
| **MoE-LoRA 类** | MultiLoRA, MixLoRA, MoELoRA, MoRE, HyperFormer, MPT |
| **其他先进方法** | AlphaLoRA, HydraLoRA, DoRA 等 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

#### 📊 在 GLUE 上的表现（T5-base，每任务仅 0.41M 参数）

| 方法 | AVG Score | 参数量（per task） |
|------|------------|------------------|
| Fine-tuning | 83.8 | 28M |
| LoRA (r=8) | 85.1 | 0.39M |
| MoELoRA | 86.2 | 0.81M |
| **EPT** | **87.0** | **0.41M** |

✅ **亮点**：
- EPT 在 **6/8 个任务上取得最佳成绩**（MNLI, QNLI, SST-2, MRPC, RTE, CoLA）
- 平均得分高出 MoELoRA **+0.8 pts**，同时参数减少一半以上
- 即使相比参数更多的方法（如 MultiLoRA: 1.56M），仍保持领先

#### 🧠 在常识推理任务上的表现（LLaMA2-7B）

| 方法 | AVG Accuracy |
|------|--------------|
| LoRA | 73.1 |
| MoRE | 74.9 |
| **EPT** | **75.5** |

✅ EPT 在更大模型上依然保持最强性能，验证其**跨模型鲁棒性**

---

### 消融实验结果（Table 4 & 5）

#### 消融配置对比（Table 5）

| 模块 | AB init | Top-K | ALP | AVG |
|------|--------|-------|-----|-----|
| 基线 | × | × | × | 86.0 |
| +ALP | × | × | ✓ | 86.2 |
| +AB init | ✓ | × | ✓ | 86.5 |
| +Top-K | ✓ | ✓ | ✓ | **87.0** |

#### 结论：
1. **AB init（随机初始化 A/B）**：带来 +0.3 提升 → 初始非退化表示至关重要
2. **Top-K 路由**：+0.5 提升，尤其在 RTE (+1.4) 和 QNLI 上明显 → 多尺度融合有效
3. **Adaptive LoRA Pruner (ALP)**：防止共享知识被覆盖，显著提升 CoLA 和 SST-2 稳定性

> 🔍 特别地，在固定专家维度的对比中（Table 4），**EPT-2468（混合尺度）优于所有单一尺度版本（EPT-2~8）**，证明多尺度设计必要。

---

## 4. 关键结论和发现

### 主要发现

1. ✅ **任务复杂度具有层次性**，需要不同粒度的特征表示 —— 简单任务适合低维抽象，复杂任务需高维精细建模。
2. ✅ **共享+多尺度分解优于独立专家堆叠** —— EPT 通过共享元空间大幅降低参数冗余。
3. ✅ **动态专家选择必须结合任务语义建模** —— 对比学习的任务嵌入显著提升路由质量。
4. ✅ **EPT 实现了性能与效率的双重突破** —— 更少参数 + 更高精度 + 更强鲁棒性。

---

### 方法的局限性

1. ❗ **专家维度配置为静态超参**：当前 kernel sizes（如 [2,2,4,4,...]）需手动设定，缺乏自动化搜索或动态调整机制。
2. ❗ **未在大规模预训练阶段验证**：目前实验集中于下游微调任务，尚未探索 EPT 在从头预训练中的扩展性和稳定性。
3. ❗ **计算资源限制**：部分实验受限于 GPU 数量，未能在更大模型（如 LLaMA3）上全面测试。

---

### 未来工作方向

1. **动态维度分配机制**：设计可学习的 gating 网络自动决定每个任务所需的专家尺度。
2. **扩展至 Foundation Model Training**：研究 EPT 是否可用于大规模预训练中的参数高效更新。
3. **跨模态应用**：将 EPT 推广至视觉-语言或多模态模型的适配任务中。
4. **硬件友好型实现**：进一步优化 deconvolutional projection 的推理延迟，推动实际部署。

---

> 💡 **总体评价**：  
> EPT 是一次将 **CV 中的多尺度思想成功迁移到 NLP 参数高效微调领域** 的典范工作。它不仅提出了结构新颖、理论清晰的新范式，还在多个基准上实现了 **SOTA 性能与卓越参数效率的统一**，为未来多任务 LLM 部署提供了重要思路。

</details>

---

### 16. [Rethinking Multiple-Choice Questions for RLVR: Unlocking Potential via Distractor Design](https://arxiv.org/abs/2603.12826)

**Authors**: Xu Guo, Qiming Ge, Jian Tong, Kedi Chen, Jin Zhang, Xiaogui Yang, Xuan Gao, Haijun Lv, Zhihui Lu, Yicheng Zou, Qipeng Guo  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.12826v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) significantly enhances the reasoning capabilities of Large Language Models. When applied to RLVR, Multiple-Choice Questions (MCQs) offer a scalable source of verifiable data but risk inducing reward hacking, where models shortcut reasoning via ra...

---

### 17. [A common parallel framework for LLP combinatorial problems](https://arxiv.org/abs/2603.13147)

**Authors**: David Ribeiro Alves, Vijay K. Garg  
**Category**: cs.DC  
**Published**: 2026-03-16  
**Score**: 6.5  
**Type**: new  
**ArXiv ID**: 2603.13147v1  

#### Abstract
Traditional lock-free parallel algorithms for combinatorial optimization problems, such as shortest paths, stable matching, and job scheduling require programmers to write problem-specific routines and synchronization code. We propose a general-purpose lock-free runtime, LLP-FW that can solve all co...

---

### 18. [EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning](https://arxiv.org/abs/2603.12698)

**Authors**: Chi Ruan, Dongfu Jiang, Huaye Zeng, Ping Nie, Wenhu Chen  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.12698v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) is a promising approach for improving code generation in large language models, but its effectiveness is limited by weak and static verification signals in existing coding RL datasets. In this paper, we propose a solution-conditioned and adversar...

---

### 19. [Adaptive Vision-Language Model Routing for Computer Use Agents](https://arxiv.org/abs/2603.12823)

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.12823v1  

#### Abstract
Computer Use Agents (CUAs) translate natural-language instructions into Graphical User Interface (GUI) actions such as clicks, keystrokes, and scrolls by relying on a Vision-Language Model (VLM) to interpret screenshots and predict grounded tool calls. However, grounding accuracy varies dramatically...

---

### 20. [PISmith: Reinforcement Learning-based Red Teaming for Prompt Injection Defenses](https://arxiv.org/abs/2603.13026)

**Authors**: Chenlong Yin, Runpeng Geng, Yanting Wang, Jinyuan Jia  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 6.0  
**Type**: new  
**ArXiv ID**: 2603.13026v1  

#### Abstract
Prompt injection poses serious security risks to real-world LLM applications, particularly autonomous agents. Although many defenses have been proposed, their robustness against adaptive attacks remains insufficiently evaluated, potentially creating a false sense of security. In this work, we propos...

---

### 21. [AI Planning Framework for LLM-Based Web Agents](https://arxiv.org/abs/2603.12710)

**Authors**: Orit Shahnovsky, Rotem Dror  
**Category**: cs.AI  
**Published**: 2026-03-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.12710v1  

#### Abstract
Developing autonomous agents for web-based tasks is a core challenge in AI. While Large Language Model (LLM) agents can interpret complex user requests, they often operate as black boxes, making it difficult to diagnose why they fail or how they plan. This paper addresses this gap by formally treati...

---

### 22. [ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](https://arxiv.org/abs/2603.12740)

**Authors**: Shuo Yang, Soyeon Caren Han, Yihao Ding, Shuhe Wang, Eduard Hoy  
**Category**: cs.AI  
**Published**: 2026-03-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.12740v1  

#### Abstract
Large Language Model (LLM) agents are increasingly applied to complex, multi-step tasks that require interaction with diverse external tools across various domains. However, current LLM agent tool planning methods typically rely on greedy, reactive tool selection strategies that lack foresight and f...

---

### 23. [Steve-Evolving: Open-World Embodied Self-Evolution via Fine-Grained Diagnosis and Dual-Track Knowledge Distillation](https://arxiv.org/abs/2603.13131)

**Authors**: Zhengwei Xie, Zhisheng Chen, Ziyan Weng, Tingyu Wu, Chenglong Li, Vireo Zhang, Kun Wang  
**Category**: cs.AI  
**Published**: 2026-03-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.13131v1  

#### Abstract
Open-world embodied agents must solve long-horizon tasks where the main bottleneck is not single-step planning quality but how interaction experience is organized and evolved. To this end, we present Steve-Evolving, a non-parametric self-evolving framework that tightly couples fine-grained execution...

---

### 24. [SpectralGuard: Detecting Memory Collapse Attacks in State Space Models](https://arxiv.org/abs/2603.12414)

**Authors**: Davi Bonetto  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.12414v1  

#### Abstract
State Space Models (SSMs) such as Mamba achieve linear-time sequence processing through input-dependent recurrence, but this mechanism introduces a critical safety vulnerability. We show that the spectral radius rho(A-bar) of the discretized transition operator governs effective memory horizon: when...

---

### 25. [A Spectral Revisit of the Distributional Bellman Operator under the Cram\'er Metric](https://arxiv.org/abs/2603.12576)

**Authors**: Keru Wang, Yixin Deng, Yao Lyu, Stephen Redmond, Shengbo Eben Li  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.12576v1  

#### Abstract
Distributional reinforcement learning (DRL) studies the evolution of full return distributions under Bellman updates rather than focusing on expected values. A classical result is that the distributional Bellman operator is contractive under the Cram\'er metric, which corresponds to an $L^2$ geometr...

---

### 26. [Context-Enriched Natural Language Descriptions of Vessel Trajectories](https://arxiv.org/abs/2603.12287)

**Authors**: Kostas Patroumpas, Alexandros Troupiotis-Kapeliaris, Giannis Spiliopoulos, Panagiotis Betchavas, Dimitrios Skoutas, Dimitris Zissis, Nikos Bikakis  
**Category**: cs.AI  
**Published**: 2026-03-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.12287v1  

#### Abstract
We address the problem of transforming raw vessel trajectory data collected from AIS into structured and semantically enriched representations interpretable by humans and directly usable by machine reasoning systems. We propose a context-aware trajectory abstraction framework that segments noisy AIS...

---

### 27. [Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation](https://arxiv.org/abs/2603.13099)

**Authors**: Wayner Barrios, SouYoung Jin  
**Category**: cs.AI  
**Published**: 2026-03-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.13099v1  

#### Abstract
We introduce **CRYSTAL** (*__C__lear __R__easoning via __Y__ielded __S__teps, __T__raceability and __L__ogic*), a diagnostic benchmark with 6,372 instances that evaluates multimodal reasoning through verifiable intermediate steps. We propose two complementary metrics: *Match F1*, which scores step-l...

---

### 28. [GONE: Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping](https://arxiv.org/abs/2603.12275)

**Authors**: Chahana Dahal, Ashutosh Balasubramaniam, Zuobin Xiong  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.12275v1  

#### Abstract
Unlearning knowledge is a pressing and challenging task in Large Language Models (LLMs) because of their unprecedented capability to memorize and digest training data at scale, raising more significant issues regarding safety, privacy, and intellectual property. However, existing works, including pa...

---

### 29. [Neuron-Aware Data Selection In Instruction Tuning For Large Language Models](https://arxiv.org/abs/2603.13201)

**Authors**: Xin Chen, Junchao Wu, Shu Yang, Runzhe Zhan, Zeyu Wu, Min Yang, Shujian Huang, Lidia S. Chao, Derek F. Wong  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.13201v1  

#### Abstract
Instruction Tuning (IT) has been proven to be an effective approach to unlock the powerful capabilities of large language models (LLMs). Recent studies indicate that excessive IT data can degrade LLMs performance, while carefully selecting a small subset of high-quality IT data can significantly enh...

---

### 30. [Multi-objective Genetic Programming with Multi-view Multi-level Feature for Enhanced Protein Secondary Structure Prediction](https://arxiv.org/abs/2603.12293)

**Authors**: Yining Qian, Lijie Su, Meiling Xu, Xianpeng Wang  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.12293v1  

#### Abstract
Predicting protein secondary structure is essential for understanding protein function and advancing drug discovery. However, the intricate sequence-structure relationship poses significant challenges for accurate modeling. To address these, we propose MOGP-MMF, a multi-objective genetic programming...

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
