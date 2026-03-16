# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-16 07:03:19 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [KernelFoundry: Hardware-aware evolutionary GPU kernel optimization](https://arxiv.org/abs/2603.12440)

**Authors**: Nina Wiedemann, Quentin Leboutet, Michael Paulitsch, Diana Wofk, Benjamin Ummenhofer  
**Category**: cs.DC  
**Published**: 2026-03-16  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.12440v1  

#### Abstract
Optimizing GPU kernels presents a significantly greater challenge for large language models (LLMs) than standard code generation tasks, as it requires understanding hardware architecture, parallel optimization strategies, and performance profiling outputs. Most existing LLM-based approaches to kerne...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：KernelFoundry: Hardware-aware evolutionary GPU kernel optimization**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
当前基于 LLM 的 GPU kernel 生成方法在优化高性能计算内核时面临以下挑战：
- **缺乏硬件感知能力**：大多数方法仅通过简单的提示（prompting）和反馈循环进行优化，未能深入理解 GPU 架构特性（如内存层次、并行模式）。
- **搜索空间探索不足**：容易陷入“模式坍缩”（mode collapse），即反复生成相似变体，无法有效探索多样化的优化策略。
- **上下文退化**（context degradation）：随着迭代次数增加，失败的历史记录充斥提示上下文，导致生成质量下降。

### **提出了什么新方法或新思路**
作者提出 **KernelFoundry** ——一个面向 GPU kernel 优化的进化式框架，其三大核心机制为：

#### ✅ **(1) MAP-Elites 质量多样性搜索（Quality-Diversity Search）**
- 引入 **domain-specific behavioral descriptors** 将 kernel 设计空间划分为多个行为单元格（behavioral cells），每个单元格独立保存最优解。
- 定义三个 kernel 特定的行为维度：
  - `dmem`：内存访问模式（从标量访问到多级存储重用）
  - `dalgo`：算法结构（从直接翻译到新型算法重构）
  - `dsync`：并行协调方式（从无同步到全局原子操作）
- 该设计确保系统能持续探索不同优化路径，避免局部收敛。

#### ✅ **(2) 元提示演化（Meta-Prompt Evolution）**
- 提示本身也成为可进化的对象，维护一个独立的 prompt archive。
- 专用的 **meta-prompter LLM** 分析生成结果，动态更新提示中的四个可演化区域：
  - 优化哲学（optimization philosophy）
  - 优化策略（strategies）
  - 常见陷阱（pitfalls）
  - 分析引导（analysis guidance）
- 实现提示与代码的协同进化，缓解 context degradation。

#### ✅ **(3) 模板化参数优化（Template-based Parameter Optimization）**
- 针对 block size、tile size 等硬件敏感参数，允许 LLM 输出模板化 kernel，并自动枚举配置组合进行调优。
- 分离算法结构优化与参数调整，提升搜索效率。

### **相比现有方法的优势**
| 维度 | KernelFoundry | 传统 LLM 方法 |
|------|---------------|----------------|
| 探索多样性 | 显式维持多种优化路径（QD） | 容易陷入局部最优 |
| 上下文管理 | 动态演化的 meta-prompt 减少噪声积累 | 历史失败记录污染上下文 |
| 参数调优 | 支持模板化参数搜索 | 依赖 LLM “猜测”最佳值 |
| 可移植性 | 支持 SYCL（跨平台） | 多数局限于 CUDA |
| 实际应用扩展性 | 支持自定义任务输入格式（PyTest + YAML） | 通常限于标准 benchmark |

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **KernelBench** [Ouyang et al., 2025b]：包含 250 个任务的标准 benchmark，涵盖单算子、融合模式和完整模型架构。
- **robust-kbench** [Lange et al., 2025b]：更严格的验证集，强调反 Reward Hacking 和前向/后向一致性。
- **自定义任务**：包括 oneDNN 对标测试及 Llama3 中 rotary positional embedding 的实际加速案例。

> 注：作者对 KernelBench 进行了过滤，排除存在 baseline 不合理或输出精度过低的任务，最终保留 **111 个高质量任务**。

### **实验设置**
- **目标语言**：同时支持 **SYCL**（主）、CUDA 和 Triton。
- **硬件平台**：
  - Intel Arc B580（discrete GPU）
  - Intel Arc 140V（integrated LNL GPU）
  - NVIDIA RTX A6000（Ampere 架构）
- **LLM 后端**：
  - 主要使用 GPT-4.1、GPT-5-mini、Claude Sonnet 4.5
  - 对比实验中复现基线所用模型（如 GPT-o3-mini）

### **评估指标**
| 指标 | 定义 |
|------|------|
| **Correctness Rate** | 编译成功且数值正确的 kernel 比例（采用相对误差 `< 0.01` 在 99% 输出上成立） |
| **Cosine Similarity** | 输出张量间的夹角余弦，衡量语义正确性 |
| **Speedup** | 相对于 PyTorch Eager 执行时间的速度提升倍数 |
| **fast_p** | 达到 speedup > p 的任务占比（如 fast₁ 表示 speedup > 1） |
| **Average / Geometric Speedup** | 算术平均与几何平均速度提升 |
| **Hardware-Speedup (hws)** | 在交叉硬件测试中，本地优化 kernel 相对于异地优化 kernel 的性能优势 |

### **基线方法对比**
| 基线名称 | 方法类型 | 是否开源 | 关键特点 |
|--------|--------|----------|---------|
| **AI CUDA Engineer** [Lange et al., 2025a] | Evolutionary + Prompting | 是 | 使用 DeepSeek/GPT/Sonnet 多模型协作 |
| **robust-kbench** [Lange et al., 2025b] | Evolutionary | 是 | 强调鲁棒性和防作弊机制 |
| **Kernelsseum** | Prompting Ensemble | 是 | 多模型投票生成 |
| **OpenEvolve** [CodeLion, 2025] | Evolutionary Code Gen | 开源实现 | 类似 AlphaEvolve，但无 kernel-specific 设计 |

---

## 3. **主要实验结果和性能指标**

### **关键性能数据汇总**

#### 🔹 **表1：在 CUDA 上与主流方法对比（KernelBench 子集）**

| 方法 | Avg Speedup (L1) | Avg Speedup (L2) | fast₂ (%) |
|------|------------------|------------------|-----------|
| AI CUDA Engineer (re-eval) | 1.005 | 1.606 | 25% |
| **Ours (KernelFoundry)** | **1.241** | **2.104** | **45%** |
| robust-kbench (re-eval) | – | – | 58% |
| **Ours** | – | – | **67%** |

> ✅ KernelFoundry 在所有任务上均达到 **100% 正确率**，并在 L2 层次实现 **平均 2.1× 加速**，显著优于基线。

#### 🔹 **表2：SYCL 内核生成性能（首次系统性评估）**

| 方法 | Dataset | Avg Speedup | fast₁ (%) | fast₂ (%) |
|------|-------|-------------|-----------|-----------|
| **Ours (SYCL)** | KernelBench (filtered, n=111) | **2.32** | 71% | 42% |
| robust-kbench (CUDA) | – | 1.49 | – | – |
| OpenEvolve (40 iters) | repr. set L2 | 2.535 | 70% | 40% |
| **Ours (40 iters + param opt)** | repr. set L2 | **2.732** | **80%** | **45%** |

> 📌 即使 SYCL 更难被 LLM 理解，KernelFoundry 仍实现了 **97% 正确率** 和 **2.32× 平均加速**，超越 CUDA 基线。

#### 🔹 **硬件感知能力验证（Crossover Experiment）**

| 测试场景 | hws₁ (%) | avg hws | geom hws |
|--------|---------|--------|---------|
| LNL-optimized kernels on LNL | 70% | **1.537** | 1.297 |
| B580-optimized kernels on B580 | 70% | 1.109 | 1.038 |

> ✅ 显示出明显的硬件特异性优化能力：为特定设备优化的 kernel 在该设备上表现更好。

#### 🔹 **与 oneDNN 高性能库对比（真实世界竞争力）**

| Operation | Speedup vs oneDNN |
|----------|--------------------|
| concat(x, layernorm(x)) | **1.79×** |
| softmax (FlashAttention-inspired) | **1.70×** |
| matmul + relu | 0.35× |
| maxpool + linear | 0.72× |
| sum reduction | 1.10× |

> ✅ 在部分操作中超过 hand-tuned assembly-level oneDNN 实现，证明其工业实用性。

#### 🔹 **消融实验与组件有效性分析**
- **Meta-prompt evolution**：显著提升长期稳定性，减少无效尝试。
- **Template-based tuning**：在部分任务中带来额外 **10–20% 性能增益**。
- **Gradient-informed selection**：加快收敛速度，在前 10 轮迭代中明显领先 OpenEvolve。

---

## 4. **关键结论和发现**

### **主要发现**
1. ✅ **KernelFoundry 是首个将 quality-diversity 搜索与 meta-prompt evolution 结合用于 kernel 优化的工作**，有效克服了 mode collapse 与 context degradation。
2. ✅ **SYCL 是可行的跨平台替代方案**：尽管生态不如 CUDA 成熟，但通过 KernelFoundry 可生成高性能、可移植的 kernel。
3. ✅ **硬件感知是可学习的**：通过在不同 GPU 上独立运行，系统能自动发现适应特定架构的优化策略。
4. ✅ **框架具备实际工程价值**：成功应用于 Llama3 模型中的 rotary embedding 加速，实现 **7.9× 单项加速**，整体前向延迟降低 **8%**。

### **方法的局限性**
- 当前依赖较强的闭源 LLM（如 GPT-4.1），在弱模型（如 GPT-OSS 20B）上成功率较低（仅 65% 正确率）。
- 模板参数搜索目前仅覆盖常见参数（block/tile size），尚未支持复杂调度策略。
- 编译失败的错误恢复机制有限，仍需人工干预处理极端情况。

### **未来工作方向**
- 扩展模板化 kernel 支持更多输入形状与 tensor layout 自适应。
- 引入形式化验证以彻底消除 reward hacking 风险。
- 探索基于 RL 的 fine-tuning 方案，结合 verifiable reward signals。
- 构建开放的 kernel database 与 benchmark registry，推动社区共建。

---

> 💡 **总体评价**：  
> KernelFoundry 不只是一个“更好的 prompt + loop”工具，而是构建了一个完整的 **LLM-driven evolutionary kernel engineering pipeline**，兼具科学深度与工程实用价值。它标志着从“LLM 写代码”迈向“LLM 做系统级性能工程”的重要一步。

</details>

---

### 2. [TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition](https://arxiv.org/abs/2603.12465)

**Authors**: Prabhu Vellaisamy, Shreesh Tripathi, Vignesh Natarajan, Surya Santhan Thenarasu, Shawn Blanton, John P. Shen  
**Category**: cs.DC  
**Published**: 2026-03-16  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.12465v1  

#### Abstract
Large Language Model (LLM) inference is widely used in interactive assistants and agentic systems. In latency-sensitive deployments, inference time can become dominated by host-side overheads. Existing approaches typically expose this cost only as an aggregate residual or a launch/queue metric, whic...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition*

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在 **LLM 推理**（尤其是延迟敏感场景）中，端到端延迟常受 **host-side orchestration overhead**（主机侧编排开销）主导。然而，现有工具（如 *framework tax* 或 *TKLQT*）仅提供聚合指标，无法定位开销来源的具体执行层（框架、CUDA 库、内核启动路径），导致优化目标模糊。

### 提出的新方法与新思路
本文提出 **TaxBreak**，一种基于 trace 驱动的 **host-side 开销分解方法**，将总主机开销 $ T_{\text{orchestration}} $ 分解为三个互斥且完备的组成部分：

- **△FT (Framework Translation)**：Python 调度与 ATen 框架调度开销  
- **△CT (CUDA-Library Translation)**：vendor library（如 cuBLAS/cuDNN）前端处理开销  
- **△KT (Kernel Launch Path)**：从 `cudaLaunchKernel` 到 GPU 内核实际启动的时间（硬件底层延迟）

同时引入 **Host-Device Balance Index (HDBI)**：
$$
\text{HDBI} = \frac{T_{\text{DeviceActive}}}{T_{\text{DeviceActive}} + T_{\text{orchestration}}} \in (0,1)
$$
- HDBI → 0：完全 host-bound  
- HDBI → 1：完全 device-bound  

该指数用于量化系统瓶颈倾向，并指导优化策略选择。

### 相比现有方法的优势

| 方法 | 缺陷 | TaxBreak 改进 |
|------|------|----------------|
| Framework Tax [14] | 仅提供聚合残差，无分层分解 | 提供三层细粒度归因 |
| TKLQT [30] | 仅关注 launch/queue 路径 | 包含 framework 和 library 层开销 |
| GPU Utilization | 忽略 host-side 影响 | 显式建模 host-device 平衡 |

> ✅ **优势总结**：TaxBreak 将“是否 host-bound”诊断升级为“**哪一层是瓶颈**”的机制级归因，实现精准优化导向。

---

## 2. 核心实验方法和设置

### 使用的模型与工作负载（Workloads）
- **Dense Models**:
  - Llama-3.2-1B / -3B
  - GPT-2 (124M)
- **Mixture-of-Experts (MoE) Models**:
  - OLMoE-1B/7B
  - Qwen1.5-MoE-A2.7B
- 所有实验运行于 **BFloat16** 精度，采用 **eager mode**（动态控制流场景）

### 实验平台（Hardware Setup）
- **H100 平台**：
  - CPU: Intel Xeon 8480C (56 cores @ 2.0/3.8 GHz)
  - GPU: NVIDIA H100 (80GB)，DGX H100 系统
- **H200 平台**：
  - CPU: Intel Xeon Gold 6538Y+ (32 cores)
  - GPU: NVIDIA H200 NVL (141GB)

> ⚠️ 注意：H200 GPU 主频更低（1785 MHz vs. 1980 MHz），但 CPU 单核性能更强，形成理想对照组。

### 软件栈
- Python 3.13, PyTorch 2.10, CUDA 12.6
- 使用 **NVIDIA Nsight Systems (nsys)** 进行 kernel-level tracing
- 所有测量均进行 **W=50 次预热 + R=150 次采样**，确保稳定性（95% CI < 0.34ms）

### 评估指标
- **End-to-end Latency**（TTFT & TPOT）
- **Torchestration**（分解后三项之和）
- **TDeviceActive**（GPU 内核执行时间总和）
- **HDBI**
- **GPU Idle Fraction** = $(T_{e2e} - T_{\text{DeviceActive}})/T_{e2e}$
- **Kernel Launch Count**, **Kernel Diversity Ratio**

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### （1）MoE 模型显著增加 kernel 数量
在 decode 阶段（BS=4, SL=2048, m=10）：

| Model | Total Kernel Launches | Kernels per Token | GPU Util (%) |
|-------|------------------------|--------------------|---------------|
| Llama-3.2-1B | 8,475 | 847.5 | 58.9 |
| OLMoE-1B/7B | 93,053 | **9,305.3** | **15.5** |
| Qwen1.5-MoE-A2.7B | 66,951 | **6,695.1** | **27.7** |

> 🔴 MoE 模型每 token 发射 **8–11× 更多 kernels**，源于路由逻辑与专家调用碎片化。

#### （2）HDBI 揭示不同模型的 boundedness 演变趋势
- **Dense 模型**：
  - Prefill：HDBI ≈ 0.37–0.41（适度平衡）
  - Small decode：HDBI ↓ 至 ~0.23（短暂 host-visible）
  - Large batch/context decode：HDBI ↑ 回 0.9+（重新 device-dominant）
- **MoE 模型**：
  - Prefill：HDBI ≈ 0.15
  - Decode：HDBI 持续低于 0.15，**始终 host-bound**，即使增大 batch 或 seq len 也无法缓解

#### （3）Null-kernel Launch Floor 测量（$ T_{\text{floor}}^{\text{sys}} $）
| GPU | avg (μs) | p50 (μs) |
|-----|----------|----------|
| H100 | 4.71 | 4.58 |
| H200 | 4.50 | 4.45 |

表明 launch 路径底层延迟极小且稳定，可用于归一化分析。

#### （4）FlashAttention-2 对比实验（Llama-3.2-1B on H200）
| Config | E2E Latency (ms) | $ T_{\text{orchestration}} $ (ms) | HDBI | GPU Util (%) |
|--------|------------------|-------------------------------|------|--------------|
| Eager (BS=8/SL=2048) | 303.99 | 11.7 | 0.96 | 97.9 |
| FA2 (same) | **95.5** | **8.9** (-24%) | **0.90** | 96.5 |

> ✅ FA2 主要通过减少 device-side memory traffic 提升性能，而非降低 host overhead；TaxBreak 成功识别其本质为 **device-side 优化**。

---

### 与基线方法对比结果

| 方法 | 是否支持分层归因 | 是否覆盖 prefill+decode | 是否支持 MoE | 是否跨平台可比 |
|------|------------------|-------------------------|-------------|----------------|
| Framework Tax [14] | ❌（仅残差） | ✅ | ⚠️有限 | ❌ |
| TKLQT [30] | ❌（仅 launch） | ✅ | ⚠️未验证 | ✅（GH200） |
| GPU Inference Characterization [31] | ❌ | ❌（prefill-centric） | ❌ | ✅ |
| **TaxBreak (Ours)** | ✅（△FT/△CT/△KT） | ✅ | ✅ | ✅ |

---

### 消融实验与归因分析（Ablation Insights）

#### （1）GPT-2 on H200：验证 host-device 转折点
- HDBI 从 BS=1 的 0.25 上升至 BS=16 的 0.74
- $ T_{\text{orchestration}} $ 几乎不变（~5.04–5.52ms），而 $ T_{\text{DeviceActive}} $ 从 1.66ms 增至 15.43ms
- 表明：**batch 增大提升的是 device work，host 开销线性增长但 per-kernel 成本恒定**

#### （2）Eager vs. FA2：揭示优化本质
- FA2 减少 kernel 数量（-7% ~ -19%）
- $ N \cdot T_{\text{floor}} $ 下降 0.3–0.8ms，与观测一致
- HDBI 下降说明：虽然绝对 $ T_{\text{orchestration}} $ 下降，但 **device work 下降更快**，导致 host 占比上升 —— 若无 TaxBreak，可能误判为“host 退步”

---

## 4. 关键结论和发现

### 主要发现（Key Takeaways）

1. **Aggregate 指标会误导优化方向**  
   单看 latency、GPU idle 或 boundedness ratio 可能掩盖真实瓶颈。例如 FA2 使 HDBI 下降，看似更 host-bound，实则是 device 性能大幅提升所致。

2. **MoE 模型在 decode 中长期处于 host-bound 状态**  
   其高 kernel 发射密度（8–11× dense）导致 $ T_{\text{orchestration}} $ 主导延迟，**批大小或更快 GPU 难以缓解此问题**。

3. **CPU 单线程性能是 host-bound 场景的一阶设计参数**  
   在 H200 平台上，尽管 GPU 更慢，但由于 CPU 单核更强（Emerald Rapids），**end-to-end latency 反而降低 11–14%**：
   - 对 MoE decode：$ T_{\text{orchestration}} $ 下降 **29%**
   - 对 dense decode：下降 **14%**
   - 效果随 HDBI 增加而减弱（device-bound 场景收益小）

4. **TaxBreak 可指导优化策略选择**
   - 若 △FT + △CT 主导 → 优化软件栈（如启用 `torch.compile`）
   - 若 $ N \cdot T_{\text{floor}} $ 主导 → 优先 kernel fusion 或 CUDA Graphs
   - 若 △KT_fw 显著 → 优化 driver/runtime 路径

---

### 方法的局限性

- **依赖 CUDA tracing 接口**：目前仅适用于 NVIDIA GPU，难以直接迁移到 AMD 或其他架构。
- **回放式测量对高度动态 kernel 不完美**：autotuning、同步密集型操作可能导致 replay 结果偏差。
- **HDBI 是诊断指标，非优化目标本身**：不能单独作为 loss function 使用，需结合绝对延迟解读。
- **未涵盖分布式 MoE 场景**：当前分析限于单卡，multi-node 通信开销未被建模。

---

### 未来工作方向

- 扩展至更多 AI 工作负载（vision, speech, retrieval-augmented generation）
- 支持跨平台（NVIDIA GB200/GB300, AMD MI300A/MI300X）
- 构建自动化优化建议引擎，基于 TaxBreak 输出推荐最佳 tuning 策略
- 探索在 compile-time 阶段预测并最小化 $ T_{\text{orchestration}} $

--- 

> 💡 **总结一句话**：  
> **TaxBreak 把“LLM 推理为何慢”的问题，从“是不是主机拖累”深化为“到底是框架、库还是启动路径在拖后腿”，为系统优化提供了精准导航图。**

</details>

---

### 3. [ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](https://arxiv.org/abs/2603.13115)

**Authors**: Jie Ji, Gen Li, Kaiyuan Deng, Fatemeh Afghah, Xiaolong Ma  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.13115v1  

#### Abstract
Deep learning models, despite their impressive achievements, suffer from high computational costs and memory requirements, limiting their usability in resource-constrained environments. Sparse neural networks significantly alleviate these constraints by dramatically reducing parameter count and comp...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training —— 论文总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前 **sparse training** 在高稀疏度下普遍存在以下挑战：
- **梯度信号混乱且噪声大**（chaotic and noisy gradient signals），导致训练不稳定、收敛困难；
- **Sharpness-Aware Minimization (SAM)** 虽能提升泛化能力并引导模型进入平坦极小值（flat minima），但其计算开销高昂——每步需两次反向传播（backpropagation），使得在资源受限场景中难以应用。

### 🚀 提出的新方法：**ZO-SAM**
作者提出 **Zero-Order Sharpness-Aware Minimization (ZO-SAM)**，一种将 **zero-order (ZO) optimization** 与 **SAM** 结合的新型优化框架，专为高效稀疏训练设计。

#### 创新机制：
- 在 SAM 的 **perturbation step** 中使用 **zero-order gradient estimation**（如 RGE）来估计扰动方向；
- 在后续的 **gradient update step** 中仍保留精确的一阶梯度（first-order gradient）进行参数更新。

> 这种“混合策略”实现了效率与稳定性的平衡。

### 🔍 相比现有方法的优势
| 方面 | 优势说明 |
|------|----------|
| **计算效率** | 只需一次 backpropagation，相比传统 SAM 减少 50% 的计算成本；相比 CGE 类 ZO 方法更高效（采用 RGE，仅需 $ m \ll d $ 次函数评估） |
| **稳定性与收敛性** | 显著降低梯度方差（gradient variance），加速收敛速度（见 Figure 5, 6） |
| **泛化性能** | 引导模型找到更平坦的损失曲面（flatter minima），提高测试准确率和鲁棒性 |
| **兼容性强** | 可无缝集成到多种主流稀疏训练方法中（如 LTH, SNIP, RigL, MEST 等） |

---

## 2. 核心实验方法和设置

### 📊 数据集
- **图像分类任务**：
  - **CIFAR-10 / CIFAR-100**：用于主实验验证；
  - **ImageNet-1K**：用于 Transformer 架构上的扩展实验；
  - **CIFAR-10-C**：用于评估分布偏移下的鲁棒性（robustness under distribution shift）。

### ⚙️ 实验设置
| 组件 | 配置 |
|------|------|
| **模型架构** | ResNet-32, ResNet-50, WideResNet-28-10, DeiT-Tiny/Small |
| **稀疏度水平** | 90%, 95%, 98% （静态与动态稀疏训练均覆盖） |
| **硬件平台** | 4×NVIDIA A6000 GPU |
| **重复次数** | 所有实验运行 3 次，报告均值 ± 标准差 |
| **评估指标** | 测试准确率（Test Accuracy）、收敛速度（Epoch to 90% Acc）、推理吞吐量（images/sec）、鲁棒性（△ = Clean Acc − Corrupted Acc） |

### 🔁 基线方法对比
#### 主要对比类别：
| 类型 | 方法 |
|------|------|
| **基础稀疏训练法** | LTH, SNIP, GraSP, SET, DSR, RigL, MEST |
| **SAM 家族变体** | SAM, ESAM, LookSAM (k=5/10), GSAM |
| **本文方法** | 各基线 + ZO-SAM（即 `Method + ZO-SAM`） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ 在 ResNet-32 上的表现（Table 1）
| 方法 | 稀疏度 | CIFAR-10 Acc (%) | 提升幅度 |
|------|--------|------------------|---------|
| LT | 90% → 98% | 91.06 → 88.78 | +1.25 → +0.68 |
| **LT + ZO-SAM** | 90% → 98% | **92.69 → 89.46** | ↑1.63 → ↑0.68 |
| SNIP | 90% → 98% | 93.38 → 88.22 | +0.79 → +0.71 |
| **SNIP + ZO-SAM** | 90% → 98% | **93.38 → 88.22** | ↑0.79 → ↑0.71 |
| RigL | 90% → 98% | 93.66 → 90.61 | +0.59 → +1.61 |
| **RigL + ZO-SAM** | 90% → 98% | **93.66 → 90.61** | ↑0.59 → ↑1.61 |
| MEST | 90% → 98% | 94.37 → 93.53 | +0.31 → +0.66 |
| **MEST + ZO-SAM** | 90% → 98% | **94.37 → 93.53** | ↑0.31 → ↑0.66 |

> 💡 在 CIFAR-10 上最高提升达 **2.31%**（MEST@98%），CIFAR-100 上最高提升 **2.54%**（RigL@95%）。

#### ✅ 在 ResNet-50 上的结果（Appendix Table .6）
- 一致地提升了各方法精度，在 CIFAR-10 上最大提升 **1.04%**，CIFAR-100 上达 **1.56%**，表明方法具有良好的跨架构泛化能力。

#### ✅ Transformer 在 ImageNet-1K 上的表现（Table 2）
| 模型 | 方法 | 稀疏度 | 准确率 (%) | 提升 |
|------|------|--------|------------|------|
| DeiT-Tiny | MEST | 50% | 69.69 | — |
| | **MEST + ZO-SAM** | 50% | **70.41** | ↑0.72 |
| DeiT-Small | RigL | 70% | 77.99 | — |
| | **RigL + ZO-SAM** | 70% | **79.16** | ↑1.17 |

> 表明 ZO-SAM 对 Transformer 同样有效。

---

### ⏱️ 收敛速度对比（Table 3 & Figures 5–6）
| 方法 | 达到 90% 准确率所需 epoch 数（越低越好） |
|------|----------------------------------------|
| SGD (sp=0.9) | 104 |
| ESAM | 75 |
| LookSAM (k=5) | 79 |
| GSAM | 84 |
| **ZO-SAM** | **70** ✅ |

> ZO-SAM 是所有方法中收敛最快的，在 **95% 稀疏度下也仅需 88 个 epoch**，远快于 SGD 的 131。

---

### ⚖️ 计算效率对比（Table 4）
| 方法 | ResNet-32 (img/sec) | WRN-28-10 (img/sec) | 效率占比 (%) |
|------|--------------------|---------------------|---------------|
| SGD | 5673.95 | 752.30 | 100% |
| SAM | 2704.84 | 354.94 | ~47.7% |
| ESAM | 2297.23 | 305.76 | ~40.5% |
| LookSAM (k=10) | 4272.22 | 593.02 | ~75.3% |
| **ZO-SAM** | **4349.53** | **576.01** | **~76.7%** ✅ |

> ZO-SAM 在保持接近 SAM 泛化能力的同时，吞吐量是 SAM 的 **1.6 倍以上**，显著优于其他高效 SAM 变体。

---

### 🛡️ 鲁棒性测试（Table 5, CIFAR-10-C）
| 方法 | Clean Acc (%) | Corrupted Acc (%) | △ = Clean − Corrupted |
|------|----------------|--------------------|------------------------|
| SNIP | 92.59 | 59.60 | 32.99 |
| SNIP + GSAM | 93.71 | 61.65 | 32.06 |
| **SNIP + ZO-SAM** | **93.38** | **62.70** | **30.68** ✅ |

> ZO-SAM 将 corruption 下的性能提升 **3.10%**，且 △ 最小，说明其具备更强的分布外鲁棒性。

---

### 🔍 特征图可视化分析（Figure 7）
- ZO-SAM 学得的特征图更清晰、局部响应更强：
  - **早期层**：边缘检测更连贯；
  - **中期层**：纹理表示更干净；
  - **深层**：语义激活更集中于目标区域；
- 对比之下，原始 DST 方法出现模糊、分散的激活模式。

> 佐证了 ZO-SAM 能学习更具判别性和稳定的内部表征。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **梯度噪声是稀疏训练不稳定的根源**，尤其在高稀疏度下加剧；
2. **SAM 有助于寻找平坦极小值，改善泛化，但代价过高**；
3. **ZO-SAM 成功融合 ZO 与 SAM 的优点**：
   - 利用 ZO 在 perturbation step 中避免额外 backprop；
   - 保留 first-order gradient 保证更新精度；
   - 实现 **单次反向传播 + 接近 SAM 的性能**；
4. ZO-SAM 在多个维度上实现最优权衡：
   - ✅ 更高的准确率
   - ✅ 更快的收敛
   - ✅ 更强的鲁棒性
   - ✅ 更高的计算效率

---

### ⚠️ 局限性
1. **依赖随机采样方向（RGE）的质量**：若方向数 $ m $ 过小可能导致扰动估计偏差；
2. **对超参数敏感性未充分探讨**：如 finite difference step $ \delta $、perturbation radius $ \rho $ 的选择可能影响性能；
3. **目前主要验证在 CV 领域**，在 NLP 或生成模型中的表现尚待探索。

---

### 🔮 未来工作方向
1. **自适应调节 ZO 参数**（如动态调整 $ m $ 或 $ \delta $）以进一步提升效率；
2. **扩展至联邦学习、边缘设备部署等低资源场景**；
3. **结合其他正则化技术**（如 dropout, mixup）构建更强大的稀疏训练 pipeline；
4. **应用于 LLM sparse fine-tuning**，利用 ZO 的前向-only 特性减少显存占用。

---

## 总结
> **ZO-SAM 是首个将 zero-order optimization 引入 SAM 框架的工作，解决了 SAM 在稀疏训练中因双反向传播带来的计算瓶颈问题。它通过“扰动用零阶、更新用一阶”的巧妙设计，在几乎不牺牲性能的前提下大幅提升了训练效率，并增强了模型的稳定性、收敛速度和鲁棒性，为资源受限环境下的高效深度学习提供了实用且可扩展的新范式。**

</details>

---

### 4. [Efficient and Interpretable Multi-Agent LLM Routing via Ant Colony Optimization](https://arxiv.org/abs/2603.12933)

**Authors**: Xudong Wang, Chaoning Zhang, Jiaquan Zhang, Chenghao Li, Qigan Sun, Sung-Ho Bae, Peng Wang, Ning Xie, Jie Zou, Yang Yang, Hengtao Shen  
**Category**: cs.AI  
**Published**: 2026-03-16  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.12933v1  

#### Abstract
Large Language Model (LLM)-driven Multi-Agent Systems (MAS) have demonstrated strong capability in complex reasoning and tool use, and heterogeneous agent pools further broaden the quality--cost trade-off space. Despite these advances, real-world deployment is often constrained by high inference cos...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Efficient and Interpretable Multi-Agent LLM Routing via Ant Colony Optimization

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前基于 **Large Language Model (LLM)** 的 **Multi-Agent Systems (MAS)** 在实际部署中面临三大挑战：
- **高推理成本**（inference cost）和 **延迟**（latency）
- **路由策略缺乏透明度**，难以在医疗、金融等高风险领域建立信任
- 现有路由机制多依赖昂贵的 LLM selector 或静态规则，在动态负载和混合意图下表现不稳定，资源利用率低

### 🚀 提出的新方法：AMRO-S
作者提出 **AMRO-S**（Ant Colony Optimization-based Multi-Agent Routing with Supervision），一种高效且可解释的 MAS 路由框架，其核心思想是将 MAS 路由建模为“语义条件下的路径选择”问题，并引入生物启发机制。

#### 主要创新点：
1. **语义感知的小模型路由器（SFT-enhanced SLM Router）**
   - 使用一个经过 **Supervised Fine-Tuning (SFT)** 的小型语言模型（如 Llama-3.2-1B-Instruct）进行 **intent inference**
   - 输出查询的 **任务混合分布**（task-mixture distribution），作为后续路由决策的语义锚点
   - 显著降低路由开销，同时支持对混合意图的细粒度响应

2. **任务特定的信息素专家（Task-specific Pheromone Specialists）**
   - 将传统 ACO 中的全局信息素矩阵分解为多个 **任务专属的信息素矩阵**（如 `T_math`, `T_code`）
   - 通过 **query-conditioned fusion** 动态融合不同任务的信息素，减少跨任务干扰
   - 实现更精准的任务适配路径选择

3. **质量门控的异步更新机制（Quality-Gated Asynchronous Update）**
   - 推理与学习解耦：在线服务路径不进行实时更新，避免增加延迟
   - 异步后台使用轻量级 **LLM-Judge** 对执行轨迹进行二元质量评估（g ∈ {0,1}）
   - 仅当质量达标时才触发信息素强化，确保学习稳定性与可控性

### 🔍 相比现有方法的优势
| 维度 | AMRO-S | 传统方法（如 RouteLLM, MasRouter） |
|------|--------|-------------------------------|
| 成本 | 极低（SLM 路由器） | 高（常需大模型 selector） |
| 可解释性 | 高（信息素模式可视化） | 低（黑盒决策） |
| 动态适应性 | 强（异步在线进化） | 弱（静态或全量训练） |
| 多任务兼容性 | 强（任务隔离记忆） | 弱（易发生交叉污染） |

---

## 2. 核心实验方法和设置

### 📚 数据集
在五个公开基准上进行全面评估：
- **GSM8K**：小学数学应用题（数学推理）
- **MMLU**：涵盖57个领域的知识问答（通识理解）
- **MATH**：数学竞赛题（复杂推理）
- **HumanEval**：代码生成任务（函数实现 + 单元测试）
- **MBPP**：基础编程任务（Python 编程）

### ⚙️ 实验设置
- **Agent Pool**：异构 LLM 池，包含 `gpt-4o-mini`, `gemini-1.5-flash`, `claude-3.5-haiku`, `llama-3.1-70b`
- **Semantic Router**：基于 `Llama-3.2-1B-Instruct` 和 `Qwen2.5-1.5B` 进行 SFT，训练集含 3,000 条 GPT-4o 生成的多样化指令
- **统一预算约束**：固定最大交互轮次（T_max）和总调用次数（I_max），确保公平比较
- **成本计算**：基于官方 API 定价模型，综合 token 消耗、延迟、节点负载

### 📊 评估指标
- **主指标**：Pass@1（Exact Match for 数学 / Unit Test Pass for 代码）
- **平均得分**：五项任务 Pass@1 的均值
- **效率指标**：端到端延迟、并发吞吐量、成本（美元/请求）
- **可解释性分析**：信息素热力图可视化

### 🆚 基线方法对比
| 类型 | 方法 |
|------|------|
| 单模型基线 | GPT-4o, Claude-3.5-Sonnet |
| 单模型 + 推理链 | CoT, ToT, GoT, AoT |
| 多智能体无路由 | LLM-Debate, GPTSwarm, AFlow |
| 动态路由方法 | RouteLLM, RouterDC, MasRouter |
| 插件式集成 | AMRO-S 替换 MacNet / GPTSwarm / HEnRY 的原生路由策略 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（来自 Table I）

| 方法 | MMLU | GSM8K | MATH | HumanEval | MBPP | **Avg.** |
|------|------|-------|------|-----------|--------|---------|
| Vanilla (GPT-4o) | 88.7 | 96.1 | 76.6 | 90.2 | 87.2 | 87.76 |
| MasRouter (SOTA) | 84.25 | 95.45 | 75.42 | 90.62 | 84.0 | **85.93** |
| **AMRO-S (Ours)** | **86.10** | **96.40** | **78.15** | **92.20** | **86.30** | **87.83** |

✅ **提升显著**：相比最强基线 MasRouter，**平均分提高 1.9 分**，尤其在难度较高的 MATH 和 MBPP 上分别提升 **+2.73** 和 **+2.3**

### 🔁 插件式集成效果（Table II）
在 MacNet、GPTSwarm、HEnRY 框架中替换原有路由策略后：
- 所有场景下 **准确率更高**
- 同时 **推理成本更低**（例如 GSM8K 成本从 \$2.14 → \$2.00）
- 表明 AMRO-S 是一个 **即插即用、低成本增益** 的通用路由层

### ⏱️ 高并发压力测试（Table V）
在 **1000 并发进程** 下：
- **速度提升达 4.7×**（运行时间从 3849s → 823s）
- **准确率稳定在 96.4%**，几乎无下降
- 对比基线 WRR（加权轮询）：准确率从 96.0% 降至 88.2%，说明其无法维持语义一致性

### 🔍 消融实验（Table III & IV）

#### 路由组件消融（ID I vs 其他）
| 设置 | Avg. Score |
|------|------------|
| Random Routing | 79.64 |
| w/o SFT (Llama-3.2-1B) | 83.42 |
| w/o SFT (GPT-4o-mini) | 86.48 |
| w/ SFT (Qwen2.5-1.5B) | 87.63 |
| **AMRO-S (SFT + Llama-3.2-1B)** | **87.83** |

👉 结论：**SFT 显著提升小模型语义识别能力**，即使使用紧凑模型也能接近最优性能

#### SLM 意图识别准确率（Table IV）
| 模型 | Math | Code | General | **Avg.** |
|------|------|------|---------|---------|
| Llama-3.2-1B (zero-shot) | 78.5% | 82.1% | 85.4% | 82.0% |
| Qwen2.5-1.5B (zero-shot) | 84.2% | 88.5% | 89.1% | 87.26% |
| **Llama-3.2-1B (SFT)** | **98.1%** | **97.9%** | **97.8%** | **97.93%** |

👉 SFT 使轻量级 SLM 达到接近饱和的意图识别精度，验证了低开销高精度接口的可行性

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **AMRO-S 显著提升了 MAS 的质量-成本权衡**，在多个任务上超越 SOTA 路由方法
2. **任务特定的信息素设计有效缓解了跨任务干扰**，实现了更稳定的混合负载处理
3. **质量门控机制保障了在线进化的可靠性**，避免噪声轨迹导致性能退化
4. **信息素模式具有强可解释性**，可通过热力图直观展示不同任务的最优协作拓扑（见 Fig. 3）
   - 如 `T_code` 倾向于选择特定 backbone 在最终阶段执行
   - `T_math` 展现出阶段性分工：前期重分解，后期重精确计算

### ⚠️ 局限性
- 当前信息素更新仍依赖人工定义的 **任务类别集合 T**，尚未完全自动化任务发现
- 异步更新存在一定延迟，可能影响极端快速变化环境下的响应速度
- 信息素空间随 agent 数量平方增长，大规模系统中需考虑稀疏化或压缩

### 🔮 未来工作方向
- 自动化任务聚类与动态信息素专家生成
- 结合 Meta-Learning 实现 zero-shot 路由迁移
- 在边缘设备或联邦学习场景中部署轻量化 AMRO-S
- 将信息素机制扩展至非结构化协作流程（如自由对话型 MAS）

---

## 总结
> **AMRO-S 是首个将 Ant Colony Optimization 与 LLM 多智能体路由深度融合的工作**，通过 **SFT-SLM 语义感知 + 任务隔离信息素 + 质量门控异步进化** 三重机制，在保持极低延迟的同时实现了高性能、高可解释性的动态路由。其实验充分验证了该方法在准确性、效率、稳定性与可维护性上的全面优势，为构建可信、高效的 LLM-MAS 提供了新范式。

</details>

---

### 5. [Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](https://arxiv.org/abs/2603.12996)

**Authors**: Bumjun Kim, Dongjae Jeon, Moongyu Jeon, Albert No  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.12996v1  

#### Abstract
Parallel decoding for diffusion LLMs (dLLMs) is difficult because each denoising step provides only token-wise marginal distributions, while unmasking multiple tokens simultaneously requires accounting for inter-token dependencies. We propose Dependency-Aware Parallel Decoding (DAPD), a simple, trai...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs

## 1. 论文的主要贡献和创新点

### 解决的问题
- **Parallel decoding** 是 diffusion-based LLMs (dLLMs) 的一个关键优势，理论上可以显著降低推理延迟。
- 然而，现有 dLLMs 通常只建模每个掩码位置的 token-wise **conditional marginal** 分布，直接并行采样多个 token 会忽略 token 间的依赖关系，导致 **joint-marginal mismatch**，产生局部合理但全局不一致的输出。

### 提出的新方法：DAPD (Dependency-Aware Parallel Decoding)
- **核心思想**：利用模型内部的 **self-attention** 机制作为条件独立性的代理信号，构建一个动态的 **Markov Random Field (MRF)** 来显式建模被掩码 token 之间的依赖关系。
- **方法流程**：
  1. 在每一步解码时，计算所有被掩码 token 对之间的 **attention score** $s_{ij}$ 作为依赖强度的度量。
  2. 根据一个阈值 $T_t$ 构建图 $G_t=(V, E)$，其中强注意力连接代表强依赖（有边），弱连接代表可忽略依赖（无边）。
  3. 将并行解码问题转化为在该图上选择一个 **independent set**（即图中任意两点间无边相连的节点集合），然后并行地对这些 token 进行 unmasking。
  4. 实现上采用受 **Welsh-Powell 算法**启发的启发式策略：优先选择度数高的“hub”节点，以简化后续步骤的残差图。
- **改进版本**：引入 **confidence-weighted degree** $d_i \cdot \text{conf}_i$ 作为排序标准，既考虑结构重要性也考虑预测置信度。

### 相比现有方法的优势
- **训练免费 (Training-free)**：仅依赖推理时可用的模型内部信号（attention 和 marginal confidence），无需额外训练、微调或辅助模型。
- **更优的准确性-步数权衡 (Accuracy-steps trade-off)**：相比基于边际置信度的方法，DAPD 能更好地缓解 joint-marginal mismatch，从而在更少的解码步数内达到更高的准确率。
- **实现真正的全局并行化**：基线方法倾向于按连续块（contiguous clusters）顺序解码，类似于双向自回归；而 DAPD 能够跨整个序列识别并行解码独立区域，充分利用 dLLMs 的 any-order 生成能力。
- **避免强耦合 token 的共更新**：通过图结构明确避免同时更新高度相关的 token，提升了生成质量。

---

## 2. 核心实验方法和设置

### 使用的数据集
- **主任务基准**：
  - **数学推理**：GSM8K, Math500
  - **代码生成**：MBPP, HumanEval
  - **指令跟随**：IFEval
- **并行解码压力测试**：**ParallelBench**，专门设计用于暴露强 token 依赖性和 joint-marginal mismatch。
- **解码行为分析**：使用 **TriviaQA** 数据集构造的多独立查询合成数据集（将 5 个独立问题合并为一个 prompt）。

### 实验设置和评估指标
- **模型**：在两个开源 dLLMs 上进行评估：
  - **LLaDA-8B-Instruct**
  - **Dream-7B-Instruct**
- **评估框架**：`lm-eval`，最大生成长度为 256 tokens。
- **核心指标**：
  - **Accuracy**：各任务的标准准确率。
  - **Decoding Steps / NFE (Number of Function Evaluations)**：衡量推理效率的关键指标，越少越好。
  - **Speed Up**：相对于逐 token 步进解码的速度提升倍数。
- **消融分析**：通过可视化 **unmasking 轨迹** 和统计 **segment count**（未掩码连续段的数量）来分析解码模式。

### 基线方法对比
- **EB-Sampler** (Ben-Hamu et al., 2025)：基于熵的边界选择解码位置。
- **KLASS** (Kim et al., 2025c)：结合置信度和稳定性（KL 散度）信号。
- **Fast-dLLM** (Wu et al., 2025)：应用选择性并行，仅解码置信度超过阈值的 token。
- 所有基线均采用其原论文报告的最佳超参数。

---

## 3. 主要实验结果和性能指标

### 关键性能数据与对比结果
- **准确性-步数权衡 (见 Figure 3)**：
  - DAPD 在 **LLaDA** 和 **Dream** 两个模型上均显著优于所有基线方法，其曲线位于左上角（高准确率，低步数）。
  - 在 **MBPP** 和 **IFEval** 任务上，DAPD 的准确率显著高于采用分块解码（block decoding）的基线。
  - 在其他任务上，DAPD 能以 **相当的准确率** 实现 **大幅加速**。
- **速度提升 (见 Table 2)**：
  - 在 TriviaQA 多查询任务上，DAPD 仅需 **66.2 步** 即可完成解码，而基线方法需要 124.4~131.3 步。
  - 相对于标准的逐 token 解码（256 步），DAPD 实现了 **3.87× 的速度提升**。
  - 相对于其他基线，DAPD 至少实现了 **1.8× 的额外加速**。
- **ParallelBench 结果 (见 Figure 4)**：
  - DAPD 在大多数 ParallelBench 任务上表现出更优的 **score-steps trade-off**，证明其能更有效地识别和并行更新低依赖性 token 集合。

### 消融实验结果
- **解码行为分析 (见 Figure 5 和 Figure 1)**：
  - **Baseline 方法 (如 Fast-dLLM)**：展现出高度 **顺序化 (sequential)** 的解码模式，倾向于从序列两端向内扩展，形成少量连续的已解码片段（segment count 保持低位）。
  - **DAPD**：展现出 **空间分散 (spatially dispersed)** 的解码模式。早期阶段 segment count 快速上升，表明它在不同问题上并行解码；后期随着上下文完善，片段逐渐合并。这验证了其真正利用了全局并行能力。
- **图结构有效性验证 (见 Section 3.2)**：
  - 在合成的 MRF 数据集上，attention score 能以 **0.928 的 AUC** 区分真实存在的边和非边。
  - 基于 attention sum 的节点度估计具有极低的 **Order Violation Rate (OVR=0.04)**，证明其是可靠的结构重要性代理。

---

## 4. 关键结论和发现

### 主要发现
- **Self-attention 是有效的依赖性代理**：Transformer 模型中的 self-attention 机制能够可靠地反映 token 间的条件依赖结构，可用于指导并行解码。
- **图视角是有效的**：将并行解码问题建模为在动态 MRF 图上寻找 **independent set**，是一个强大且直观的框架，能够有效缓解 joint-marginal mismatch。
- **全局并行是可行的**：DAPD 成功地将 dLLM 的解码范式从“类自回归”的局部聚类行为转变为“全局分散”的任意顺序生成，充分释放了 diffusion 模型的潜力。

### 方法的局限性
- **依赖于 attention 的质量**：方法的有效性建立在 attention 能准确反映语义依赖的假设之上。如果模型的 attention 机制存在偏差或噪声，可能会影响性能。
- **启发式算法**：独立集的选择采用的是贪心启发式算法（Welsh-Powell），并非最优解，可能存在进一步优化的空间。
- **阈值选择**：边缘阈值 $T_t$ 的设定对性能有一定影响，虽然文中通过实验确定了保守值，但仍是一个需要调整的超参数。

### 未来工作方向
- 探索更复杂的图神经网络 (GNN) 或强化学习方法来优化独立集的选择。
- 将此框架应用于其他类型的生成任务，如图像或音频生成。
- 研究如何将 DAPD 与需要额外训练的 learnable parallel decoding 方法相结合，以获得更好的性能。
- 进一步理论分析 attention 与条件独立性之间的精确关系。

</details>

---

### 6. [98$\times$ Faster LLM Routing Without a Dedicated GPU: Flash Attention, Prompt Compression, and Near-Streaming for the vLLM Semantic Router](https://arxiv.org/abs/2603.12646)

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.12646v1  

#### Abstract
System-level routers that intercept LLM requests for safety classification, domain routing, and PII detection must be both fast and operationally lightweight: they should add minimal latency to every request, yet not require a dedicated GPU -- an expensive resource better used for LLM inference itse...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**98× Faster LLM Routing Without a Dedicated GPU**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在大型语言模型（LLM）推理服务系统中，**语义路由层**（Semantic Router）负责在请求到达GPU推理端点前进行安全分类、意图识别、PII检测和模型路由。然而，这类系统面临三大瓶颈：

1. **显存爆炸**：标准的 scaled dot-product attention (SDPA) 具有 $O(n^2)$ 的注意力掩码内存开销，在处理长上下文（8K–32K tokens）时极易 OOM。
2. **延迟过高**：CPU 推理虽避免 GPU 内存压力，但延迟随长度超线性增长（如 8K tokens 达 4.9 秒），远超可接受范围。
3. **序列化开销**：Envoy 代理的 `BUFFERED` 模式需完整反序列化整个 HTTP 请求体，带来显著的 `json.Unmarshal` 和 `json.Marshal` 开销。

此外，若为路由层单独配备 GPU，成本高昂且资源利用率低。

---

### 🚀 提出的新方法与创新点

作者提出了 **三个阶段的优化策略**，共同实现 **98× 加速**，并支持与 vLLM 共享 GPU 资源：

#### **Stage 1: CK Flash Attention for ONNX Runtime on ROCm**
- **创新**：首次将 FlashAttention 集成到 **AMD ROCm 平台上的 ONNX Runtime**。
- 实现自定义算子 `CKFlashAttention`，通过 HIP 内核调用 AMD Composable Kernel (CK) 库中的 FMHA 算子。
- 将注意力内存从 $O(n^2)$ 降至 $O(n)$，使 8K–32K token 分类成为可能。

#### **Stage 2: Neural-Inference-Free Prompt Compression**
- **创新**：提出一种无需神经网络推理的提示压缩管道，结合：
  - **TextRank**（句子中心性）
  - **U-shaped position weighting**（首尾位置加权）
  - **TF-IDF**（信息密度）
  - **Novelty scoring**（离群内容检测）
- 所有输入压缩至约 **512 tokens**，极大降低计算负载。
- 完全基于经典 NLP 技术，无模型调用，压缩延迟仅 ~19ms（16K tokens）。

#### **Stage 3: Near-Streaming Body Processing**
- **创新**：设计自适应流式处理机制，基于首个 chunk 判断是否需要分类：
  - 若模型已指定（非 `"auto"`），直接零拷贝透传（zero-copy passthrough）。
  - 若为 `"auto"` 模型，则增量积累并预处理文本。
- 引入 **gjson/sjson** 实现字段级提取与原地改写，避免完整 JSON 序列化。

---

### 🔍 相比现有方法的优势

| 方面 | 本文方案 | 现有方案 |
|------|---------|--------|
| **硬件依赖** | 可共用 LLM 推理 GPU（总占用 <800MB） | 多数需专用 GPU（如 RouteLLM、NVIDIA Blueprint） |
| **长上下文支持** | 支持 32K tokens | SDPA 在 8K 即 OOM；CPU 后端延迟达秒级 |
| **压缩方式** | 无神经推理，速度快、保真度高 | 如 LLMLingua 需小模型推理（耗时数百毫秒至数秒） |
| **流式能力** | 自适应 chunk 处理，减少冗余解析 | 多数采用全缓冲（full buffering） |

---

## 2. 核心实验方法和设置

### 🧪 数据集
- 使用合成生成的 OpenAI 格式请求，包含：
  - 不同长度 prompt：~500, 2K, 8K, 16K, 32K tokens
  - 嵌入真实信号：jailbreak 前缀（如 "Ignore all previous instructions"）、PII（SSN、信用卡号）、领域特定内容（计算机科学等）
- 离线评估使用 **384 个测试用例**，来自 8 篇维基百科文章，覆盖 8 个领域 × 4 种长度 × 12 种信号组合。

---

### ⚙️ 实验设置

| 组件 | 配置 |
|------|------|
| **硬件平台** | AMD Instinct MI300X GPU（192GB HBM3, gfx942） |
| **软件栈** | ROCm 7.0, ONNX Runtime 1.22.1, Envoy v1.33 |
| **模型** | mmBERT-32K（270M 参数，FP16），部署为 3 个并发 ONNX 会话（domain/jailbreak/PII） |
| **部署架构** | vLLM Semantic Router 作为 Envoy ext_proc 过滤器运行于 Kubernetes 数据平面 |

---

### 📊 评估指标

- **End-to-End (E2E) Latency**：从客户端发送请求到收到响应头的时间（curl 测量）
- **Per-signal Extraction Latency**：各分类任务（domain/jailbreak/PII）执行时间
- **Memory Usage**：GPU 显存占用（特别是 attention mask）
- **Classification Accuracy**：domain、PII、jailbreak 检测准确率
- **Throughput (req/s)**：单流与并发下的吞吐能力
- **Compression Ratio & Overhead**：压缩后 token 数及 CPU 开销

---

### 🔁 基线方法对比

| 基线 | 描述 |
|------|------|
| **ONNX CPU (baseline)** | ONNX Runtime + CPU 推理，`BUFFERED` 模式 |
| **Candle CPU** | Rust 框架 Candle 的 CPU 推理（8K 截断至 512） |
| **ONNX GPU (SDPA)** | GPU 上标准 attention，无 FlashAttention |
| **ONNX GPU (FA)** | 本工作的 Stage 1 |
| **+ Prompt Compression** | Stage 1 + 2 |
| **+ Near-Streaming** | Stage 1 + 2 + 3 |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（8K tokens）

| 配置 | E2E 延迟 (ms) | 加速比 |
|------|----------------|--------|
| ONNX CPU (baseline) | 4,918 | 1.0× |
| ONNX GPU + FA (Stage 1) | 127 | 38.7× |
| + Prompt Compression (Stage 2) | 62 | 79.3× |
| + Near-Streaming (Stage 3) | **50** | **98.4×** |

> ✅ **累计加速达 98×，E2E 延迟从 4.9s 降至 50ms**

---

### 🔬 更多关键结果

#### ✅ **16K tokens 性能**
- 最终配置下 E2E 延迟为 **108ms**
- 此时 CPU 后端无法运行（延迟 >1.8s 或截断）

#### ✅ **FlashAttention 效果（Table V）**
| 序列长度 | SDPA | FA | Speedup |
|--------|------|----|--------|
| 4K | 167ms | 51ms | 3.3× |
| 8K | OOM | 105ms | — |
| 32K | OOM | 756ms | — |

> 💡 FA 不仅解决 OOM，还显著提升速度

#### ✅ **Prompt Compression 效果（Table VI）**
- 16K → 512 tokens，jailbreak 分类延迟从 **126.6ms → 10.4ms（12× 加速）**
- 压缩本身仅耗时 **19ms（CPU-only）**，远低于节省的 GPU 时间（>100ms）

#### ✅ **Near-Streaming 效果**
- 在 8K tokens 下，从 buffered 到 streamed 减少 **12ms（62→50ms）**
- 在 16K tokens 下减少 **34ms（142→108ms）**
- 对短请求影响小，对长请求收益显著

#### ✅ **并发负载表现（Table X）**
- 在 C=20 并发、32K tokens 场景下：
  - 无压缩时 FA 延迟高达 **9.9 秒**
  - 使用压缩后，有效 E2E 延迟仅为 **248ms**
  - 实现 **40× 以上加速**

#### ✅ **吞吐量（Table XII）**
| 配置 | req/s |
|------|-------|
| CPU baseline (8K) | 0.2 |
| GPU+FA+comp+stream (8K) | **20.0** |
| GPU+FA+comp+stream (16K) | **9.3** |

---

### 🔍 消融实验结果

- **Stage 1 alone**：解决 OOM，实现 38.7× 加速
- **Stage 2 加入后**：进一步压缩输入，使延迟再降 2×（127→62ms）
- **Stage 3 加入后**：消除 JSON 开销，最终达到 98×
- 三者**叠加效应明显**，尤其在长上下文场景

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **FlashAttention 是长上下文路由的前提**  
   没有 O(n) 注意力机制，8K+ tokens 的分类根本不可行。

2. **Prompt Compression 不仅不损失精度，反而提升准确率**  
   - **Domain classification**：53.1% → **61.2%**
   - **PII detection**：78.5% → **92.4%**
   - 原因是去除了“中间稀释”效应（lost in the middle），提升了信噪比。

3. **Jailbreak detection 准确率下降（70.8% → 56.6%）是可控的**  
   因为生产环境中该任务仍作用于原始未压缩 prompt，压缩仅用于 domain routing。

4. **可实现真正的 GPU 共享部署**  
   - 总 GPU 占用 < **800MB**
   - 成功与 vLLM 共享 MI300X，无需专用加速卡
   - 每节点节省一张 GPU，集群级成本大幅降低

5. **Stage 2 和 Stage 3 具备硬件通用性**  
   即使在 NVIDIA 平台（已有 FlashAttention），prompt compression 与 near-streaming 依然有效。

---

### ⚠️ 局限性

1. **Tokenizer 使用近似估算**  
   当前按字符长度粗略估计 token 数，未来应集成实际 tokenizer 以精确控制预算。

2. **Streaming 中 accumulate 路径仍需缓存全文**  
   虽然避免了 JSON 拷贝，但内存中仍保留完整 body，未完全实现流式内存友好。

3. **压缩粒度为句子级别**  
   无法像 LLMLingua 那样做到 token 级删减，压缩率受限（~3–6% 输出比例）。

4. **目前仅验证于 AMD 平台**  
   Stage 1 为填补 ROCm 生态空白而设计，但在 NVIDIA 上需依赖已有 FA 支持。

---

### 🔮 未来工作方向

1. **集成模型专属 Tokenizer**  
   实现更精准的 token 预算控制与压缩效果优化。

2. **完全流式压缩（Streaming Compression）**  
   在数据流入过程中动态选择关键句子，避免全文驻留内存。

3. **扩展至更多路由信号类型**  
   如模态判断、工具调用建议、缓存命中预测等。

4. **跨平台统一优化框架**  
   将 prompt compression 与 near-streaming 封装为通用中间件，适配多种 LLM 路由系统。

5. **探索强化学习驱动的动态压缩策略**  
   根据任务类型自动调整 TextRank、position、TF-IDF 权重。

---

> 📌 **开源地址**：[https://github.com/vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)  
> 📌 **模型发布**：[HuggingFace - llm-semantic-router](https://huggingface.co/llm-semantic-router)

</details>

---

### 7. [Disentangled Latent Dynamics Manifold Fusion for Solving Parameterized PDEs](https://arxiv.org/abs/2603.12676)

**Authors**: Zhangyong Liang, Ji Zhang  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 7.0  
**Type**: new  
**ArXiv ID**: 2603.12676v1  

#### Abstract
Generalizing neural surrogate models across different PDE parameters remains difficult because changes in PDE coefficients often make learning harder and optimization less stable. The problem becomes even more severe when the model must also predict beyond the training time range. Existing methods u...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文《Disentangled Latent Dynamics Manifold Fusion for Solving Parameterized PDEs》总结**

---

## **1. 主要贡献和创新点**

### **解决的问题**
该论文旨在解决**参数化偏微分方程（parameterized PDEs）求解中的两大挑战**：
- **参数泛化能力差**：传统 PINNs 在面对未见过的 PDE 参数配置时表现不佳，需重新训练。
- **时间外推能力弱**：标准坐标回归模型（如 P²INN）将时间 $ t $ 视为静态输入，缺乏内在动力学建模，导致在训练时间窗口之外（Out-t）预测迅速失效。

此外，现有基于 latent dynamics 的方法（如 PIDO、DINO）依赖于测试阶段的 **auto-decoding**（迭代优化），计算成本高且破坏了解空间的连续几何结构。

---

### **提出的新方法：DLDMF**
作者提出了 **Disentangled Latent Dynamics Manifold Fusion (DLDMF)**，一种新型物理信息神经网络框架，其核心思想是通过**空间-时间-参数解耦表示**与**动态流形融合机制**统一参数化求解与长期时间外推。

#### **关键创新点**：
1. **空间-时间-参数解耦架构（Space-Time-Parameter Disentanglement）**
   - 分别使用独立的前馈编码器处理：
     - 空间坐标 $ x $
     - 时间演化状态 $ z_t $
     - PDE 参数 $ \mu $
   - 避免将时间作为普通坐标输入，而是通过 latent Neural ODE 显式建模连续时间动态。

2. **确定性前馈初始化（Amortized Feed-Forward Latent Initialization）**
   - 不依赖测试阶段的 auto-decoding，而是通过一个确定性映射 $ z_0 = g_{\theta_o}(h_\mu) $ 直接从参数嵌入生成初始 latent state。
   - 消除了反问题求解瓶颈，提升推理效率与稳定性。

3. **动态流形融合机制（Dynamic Manifold Fusion）**
   - 将解耦的空间编码 $ h_x $、参数编码 $ h_\mu $ 和随时间演化的 latent state $ z_t $ 融合到共享的 INR 解码器中。
   - 实现物理一致的时空解重建，并结构性地分离参数刚度（stiffness）与时间动态。

4. **SVD 调制实现快速微调（Efficient Fine-Tuning via SVD Modulation）**
   - 对预训练模型的 decoder 层进行奇异值分解（SVD），仅优化选定的奇异向量基底。
   - 支持对特定目标参数配置的高效适应，无需全模型重训。

---

### **相比现有方法的优势**
| 方法 | 缺陷 | DLDMF 如何改进 |
|------|------|----------------|
| **P²INN / PINN** | 时间静态建模，外推失败 | 引入 latent Neural ODE 实现连续动态积分 |
| **PIDO / DINO** | 依赖 auto-decoding，计算昂贵，几何不连续 | 使用前馈映射避免迭代优化，保持流形连续性 |
| **MAD** | 实例级 latent 变量，无全局参数流形 | 构造显式的参数条件流形，支持跨参数泛化 |

> ✅ **DLDMF 是首个同时满足以下三点的框架**：
> - 显式参数流形编码
> - 内在连续时间 latent dynamics
> - 完全消除测试阶段的迭代优化

---

## **2. 核心实验方法和设置**

### **使用的数据集**
1. **1D 参数化对流-扩散-反应方程（Convection-Diffusion-Reaction, CDR）**
   $$
   \frac{\partial u}{\partial t} + \beta \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} - \rho u(1 - u)
   $$
   - 参数 $ (\beta, \nu, \rho) $ 控制对流、扩散、反应强度。
   - 包含多个子任务：纯对流、纯扩散、纯反应、混合项等。
   - 初始条件测试了多种设定（高斯分布、$1+\sin(x)$）。

2. **2D 流体动力学（Navier-Stokes 方程）**
   - 强制湍流场景下的不可压缩流动。
   - Reynolds 数作为可变参数，用于评估参数外推能力。

---

### **实验设置**
- **训练/测试划分**：
  - 时间维度：训练区间 $[0, T_r]$，测试外推至 $T_s > T_r$（Out-t）
  - 参数维度：训练参数集 $\mathcal{U}_{tr}$，测试包含插值（In-u）和外推（Out-u）参数
- **联合压力测试**：同时考察 $t > T_r$ 且 $\mu \notin \mathcal{U}_{tr}$ 的极端情况

---

### **评估指标**
| 指标 | 描述 |
|------|------|
| **L2 相对误差** $ \|u - \hat{u}\|_2 / \|u\|_2 $ | 主要精度指标 |
| **L2 绝对误差** | 补充绝对偏差度量 |
| **最大误差（Max Error）** | 捕捉局部尖峰误差 |
| **解释方差得分（Explained Variance Score）** | 衡量模型拟合程度，越接近 1 越好 |

分别报告 **In-t**（$t \leq T_r$）和 **Out-t**（$t > T_r$）性能。

---

### **基线方法对比**
| 类别 | 方法 |
|------|------|
| **参数化 PINN** | P²INN, PI-DeepONet |
| **连续时间 latent dynamics** | PIDO, DINO |
| **元学习方法** | MAD (Meta-Auto-Decoding) |

所有基线均在相同设置下复现或使用官方实现。

---

## **3. 主要实验结果和性能指标**

### **关键性能数据（来自 Table 1）**
| Model | Dataset | In-t L2 Rel. (%) | Out-t L2 Rel. (%) |
|-------|--------|------------------|--------------------|
| **DLDMF** | — | **1.89** | **4.21** |
| P²INN | — | 21.34 | 32.87 |
| PIDO | — | 5.67 | 8.94 |
| DINO | 100% | 4.89 | 6.52 |

> 🔺 **DLDMF 在 In-t 和 Out-t 上均显著优于所有基线**，尤其在外推任务上误差降低超过 **50%**。

---

### **与基线方法的对比结果**
- **P²INN**：虽然具备良好的参数泛化能力，但在 $t=5.0$ 和 $t=10.0$ 外推时完全失真（见 Fig. 2），验证了“静态时间建模”的根本缺陷。
- **PIDO/DINO**：得益于 latent ODE，在外推上有一定优势，但仍受限于 auto-decoding 的不稳定性和计算开销。
- **MAD**：需要大量 per-instance latent 存储和优化，难以扩展。

> 📉 图 4 显示 P²INN 的误差随外推时间呈近似线性增长，而 DLDMF 增长缓慢，表明其具有更强的动力学一致性。

---

### **消融实验结果**
作者设计了多个 ablation study 来验证各组件作用：

| 变体 | 修改 | 性能变化 |
|------|------|----------|
| **No Latent Dynamics** | 用直接时间编码替代 latent ODE | Out-t 误差急剧上升 → 证明显式动力学建模必要 |
| **Entangled Fusion** | 合并部分变量（如空间+参数） | 泛化性和稳定性下降 → 解耦更优 |
| **With Auto-decoding** | 替换为迭代 latent 推断 | 推理延迟增加 10×，且精度波动大 |

> ✅ 结果证实：**解耦架构 + 前馈初始化 + latent dynamics** 共同构成了性能提升的关键。

---

## **4. 关键结论和发现**

### **主要发现**
1. **静态时间建模无法支撑长期外推**  
   即使有良好参数泛化能力（如 P²INN），仍将因缺乏内在动力系统而导致 Out-t 性能崩溃。

2. **auto-decoding 是性能与部署的双重瓶颈**  
   - 计算成本高（每次推理需多步梯度更新）
   - 破坏参数空间的连续几何结构
   - 导致外推轨迹不稳定

3. **DLDMF 成功实现了“动态流形融合”**  
   - 将时间视为 latent manifold 上的轨迹而非坐标
   - 通过参数条件 Neural ODE 实现稳定积分
   - 前馈初始化保障了实时性与一致性

4. **参数诱导的刚度可通过结构解耦缓解**  
   - 参数影响被隔离到 $h_\mu$ 和 $z_0$ 中
   - 动态演化由 $f_o(z_t, h_\mu)$ 自适应调节
   - 减轻了 MLP 的 spectral bias 问题

---

### **方法的局限性**
- 当前实验集中在规则域和周期/Dirichlet 边界条件下。
- 对复杂几何或非结构网格的支持尚未验证。
- SVD 微调虽高效，但适用范围依赖于预训练流形的质量。

---

### **未来工作方向**
1. 扩展至 **非规则边界条件** 和 **复杂几何域**（如使用 FEM-based sampling）。
2. 探索 **更高维真实物理系统**（如大气模拟、多相流）。
3. 结合 **自适应 collocation 策略** 进一步提升 stiff regime 下的鲁棒性。
4. 开发 **在线增量学习机制**，支持持续参数域扩展。

---

> 💡 **总结一句话**：  
> **DLDMF 通过“解耦表示 + 前馈初始化 + latent dynamics + 流形融合”，首次实现了无需测试时优化即可兼具强参数泛化与可靠长时外推的物理信息建模范式，为科学机器学习提供了新的统一视角。**

</details>

---

### 8. [Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](https://arxiv.org/abs/2603.12577)

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

# 《Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation》核心总结

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **Parameter-Efficient Fine-Tuning (PEFT)** 方法如 **LoRA** 在多任务场景下存在以下局限：
- **统一架构设计缺陷**：大多数 MoE-LoRA 变体采用结构一致的专家（uniform experts），忽视了不同任务对特征粒度的需求差异（例如简单分类 vs 复杂推理）。
- **负迁移问题（Negative Transfer）**：在多任务学习中，冲突梯度导致性能下降。
- **参数冗余与效率瓶颈**：独立训练多个 LoRA 模块造成参数膨胀，且缺乏共享机制。

这些问题限制了模型在多样化、复杂度各异的任务中的适应能力与参数效率。

---

### 🆕 提出的新方法：Expert Pyramid Tuning (EPT)

作者提出 **Expert Pyramid Tuning (EPT)**，一种受计算机视觉中 **Feature Pyramid Network (FPN)** 启发的新型 PEFT 架构，其核心思想是构建一个“参数金字塔”来实现多尺度特征适配。

#### 主要创新点：
1. **共享元知识子空间（Shared Meta-Knowledge Subspace）**
   - 引入低维可学习矩阵 $ Z_{\text{meta}} = B \cdot A $，作为所有任务共有的语言模式基础。
   - 所有专家从此共享种子出发进行重构，提升参数利用率并促进知识迁移。

2. **金字塔投影机制（Pyramid Projection Mechanism）**
   - 使用不同尺寸的 **deconvolutional kernels** 将 $ Z_{\text{meta}} $ 投影到多个维度空间，形成多尺度专家。
   - 小核捕捉局部细粒度语法结构，大核建模全局语义依赖。

3. **自适应 LoRA 剪枝器（Adaptive LoRA Pruner, ALP）**
   - 动态切片共享矩阵以匹配目标层维度，并引入频率补偿因子 $ d_t / T $ 平衡共享参数与任务特定参数的更新频率，增强训练稳定性。

4. **基于对比学习的任务嵌入模块（Contrastive Task Embedding Module）**
   - 为每个任务分配可学习的嵌入向量 $ e_t $。
   - 通过对比损失（contrastive loss）最大化样本特征与其任务原型的一致性，优化路由选择精度。

---

### 🔍 相比现有方法的优势
| 维度 | EPT优势 |
|------|--------|
| **参数效率** | 显著减少训练参数量（仅需 0.41M/任务），优于多数 MoE-LoRA 方法 |
| **性能表现** | 在 GLUE 和常识推理任务上达到 SOTA 性能 |
| **结构灵活性** | 支持动态分配不同粒度的表示能力，避免过参数化或欠拟合 |
| **知识共享与区分** | 共享元知识 + 对比任务嵌入 → 更好平衡共享与特异性 |

---

## 2. 核心实验方法和设置

### 📚 数据集
#### （1）自然语言理解任务（NLU）
使用 **GLUE benchmark** 包含 8 个任务：
- **CoLA**（语言可接受性判断）
- **SST-2**（情感分析）
- **MRPC/QQP**（句子对相似性）
- **STS-B**（语义相关性回归）
- **MNLI/QNLI/RTE**（自然语言推理）

> 数据统计见附录 Table 6

#### （2）常识推理任务
使用三大常识问答数据集：
- **BoolQ**（二分类是非题）
- **OBQA**（开放书本问答）
- **ARC-E / ARC-C**（AI2 Reasoning Challenge，Easy & Challenge 子集）

> 数据统计见附录 Table 7

---

### ⚙️ 实验设置
| 设置项 | 配置 |
|-------|------|
| **主干模型** | T5-base（GLUE）、LLaMA2-7B（常识推理） |
| **优化器** | AdamW |
| **学习率** | 3e-4，线性衰减，500 步 warmup |
| **训练周期** | 5 epochs |
| **Batch Size** | Global batch size = 32 |
| **序列长度** | Max 128 tokens |
| **LoRA 参数** | Rank=8, Alpha=32，应用于 Q/K/V/O 和 MLP 层 |
| **MoE 设置** | Top-2 路由，专家核大小配置为 `[2,2,4,4,6,6,8,8]` |
| **对比学习温度** | T = 0.05 |
| **对比损失权重** | λ = 0.1 |

> 完整超参见 Table 8

---

### 🆚 基线方法对比
参与比较的方法包括：
- **Full Fine-tuning**
- **Adapters**
- **P-Tuning (PT)**
- **LoRA (r=8, r=16)**
- **HyperFormer**
- **MPT**
- **MultiLoRA**
- **MixLoRA**
- **MOELoRA**
- **MoRE**

重点关注 MoE-LoRA 类方法的公平比较（相同参数预算下）。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 2 & Table 3）

#### （1）GLUE 上的表现（T5-base，每任务仅 0.41M 参数）
| 方法 | AVG Score | 参数量/任务 |
|------|-----------|------------|
| Finetuning | 83.8 | 28M |
| LoRA (r=16) | 85.6 | 0.78M |
| MOELoRA | 86.2 | 0.81M |
| **EPT** | **87.0** | **0.41M** |

✅ **EPT 以不到一半的参数量超越所有基线，在 8 项任务中拿下 6 项第一（MNLI, QNLI, SST-2, MRPC, RTE, CoLA）**

#### （2）常识推理任务（LLaMA2-7B）
| 方法 | AVG Accuracy | 参数量/任务 |
|------|--------------|-------------|
| LoRA | 73.1 | 2.1M |
| MoRE | 74.9 | 4.5M |
| **EPT** | **75.5** | **3.3M** |

✅ **EPT 再次取得最高平均准确率，证明其在更大模型上的鲁棒性和泛化能力**

---

### 🔬 消融实验结果（Ablation Study）

#### （1）不同专家维度配置对比（Table 4）
| 配置 | AVG Score |
|------|----------|
| EPT-2（全小核） | 86.5 |
| EPT-8（全大核） | 86.3 |
| **EPT-2468（混合尺度）** | **87.0** |

➡️ 结果表明：**多尺度结构显著优于单一尺度专家池**，验证了金字塔设计的有效性。

#### （2）组件消融（Table 5）
| 模块缺失情况 | AVG Score | 影响 |
|-------------|----------|------|
| 无 AB 初始化 | 86.2 → 86.5↑ | 高斯初始化提供更丰富初始表征 |
| 无 Top-K 路由 | 86.5 → 87.0↑ | 自适应融合多尺度特征至关重要 |
| 无 ALP（剪枝器） | 86.7 → 87.0↑ | 维度感知缩放稳定训练过程 |

✅ **完整 EPT 框架（含 AB init + Top-K + ALP）达到最佳性能 87.0**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **任务复杂度具有层次性**：不同任务需要不同粒度的特征表达，统一架构无法最优适配。
2. **共享 + 分解 > 独立复制**：将任务适配分解为“共享元知识 + 多尺度重建”，大幅提升参数效率与性能。
3. **动态路由需高质量任务表示**：通过对比学习获得的任务嵌入能有效指导专家选择，缓解任务间干扰。
4. **EPT 实现高效与高性能双赢**：
   - 更少参数（↓50%+）
   - 更高性能（↑1~2 pts）
   - 更强鲁棒性（跨模型、跨任务稳定）

---

### ⚠️ 方法的局限性
1. **静态维度配置**：当前专家尺度（如 kernel size）为预设超参数，未实现动态调整。
2. **评估集中于下游微调**：尚未验证该结构在大规模预训练阶段的效果与稳定性。
3. **计算资源受限**：实验未扩展至更大规模模型（如 LLaMA3）或多模态场景。

---

### 🔮 未来工作方向
1. **动态维度分配机制**：探索基于输入或任务难度自动调节专家尺度。
2. **集成至预训练流程**：研究 EPT 在持续学习、课程学习等场景下的潜力。
3. **跨模态扩展**：将参数金字塔思想推广至视觉-语言联合建模（如 VL-MoE）。
4. **硬件友好设计**：进一步压缩投影核、支持稀疏激活，提升推理效率。

---

## ✅ 总结一句话
> **EPT 通过构建“共享元知识 + 多尺度专家金字塔”的新型 PEFT 架构，在显著降低参数消耗的同时实现了多任务性能的新 SOTA，为高效、灵活的大模型适配提供了新范式。**

</details>

---

### 9. [OpenACMv2: An Accuracy-Constrained Co-Optimization Framework for Approximate DCiM](https://arxiv.org/abs/2603.13042)

**Authors**: Yiqi Zhou, Yue Yuan, Yikai Wang, Bohao Liu, Qinxin Mei, Zhuohua Liu, Shan Shen, Wei Xing, Daying Sun, Li Li, Guozhu Liu  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 6.0  
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
传统 Digital Compute-in-Memory (DCiM) 设计中，**架构级选择**（如压缩器组合、SRAM 宏配置）与 **晶体管级优化**（标准单元和 SRAM bitcell 尺寸调整）高度耦合，且受制于工艺波动（PVT/variation）和精度约束（如 MRED/NMED）。现有工具缺乏统一框架来联合优化这些层次，并在满足应用级精度预算的前提下实现最优 PPA（Power-Performance-Area）权衡。

此外，精确评估近似乘法器需耗时的 EDA 流程（综合、时序分析、功耗仿真），导致无法支持大规模“what-if”探索。

### 🚀 提出的新方法与思路
本文提出 **Accuracy-Constrained Co-Optimization (ACCO)** 框架，并基于此构建开源工具链 **OpenACMv2**，其核心是**两层解耦优化流程**：

1. **Level-1：架构级搜索**
   - 在显式精度约束下（如 NMED ≤ ε），快速探索：
     - 近似压缩器（compressor）的组合方式
     - 部分积列（PP columns）的选择性近似策略
     - SRAM 宏参数（行/列划分、阵列数、mux ratio）
   - 使用 **PEA-GNN** —— 一个图神经网络（GNN）代理模型，实现对 PPA 和误差的**快速高保真预测**。

2. **Level-2：电路级晶体管尺寸调整**
   - 对 Level-1 筛选出的设计进行精细化调优：
     - 标准单元中压缩器的晶体管 sizing
     - SRAM bitcell 的尺寸调整
   - 采用 **Monte Carlo 仿真**，考虑 PVT 变化和工艺波动，确保鲁棒性和良率。

该方法实现了从系统级精度需求到物理实现的端到端协同优化。

### 🔍 相比现有方法的优势
| 方面 | OpenACMv2 的优势 |
|------|------------------|
| **统一性** | 首个将架构探索与晶体管级 sizing 联合建模并置于同一 ACCO 框架下的开源工具 |
| **效率** | 引入 PEA-GNN 替代传统 EDA 工具流，速度提升达 **142×（8-bit）至 464×（16-bit）** |
| **准确性感知** | 显式引入 MRED/NMED 作为硬约束，避免过设计或精度违规 |
| **开放性与可复现性** | 兼容 FreePDK45 和 OpenROAD，全流程开源，支持社区验证与扩展 |

---

## 2. 核心实验方法和设置

### 📊 数据集与生成方式
- **无外部数据集**：所有训练和测试样本由 OpenACMv2 自动生成。
- 基于 **Nangate45nm 开放单元库** 构建近似乘法器设计空间。
- 输入为不同压缩器组合及 sizing 参数，输出通过 OpenROAD + OpenSTA + VCS 进行 EDA 仿真获取真实值用于训练 PEA-GNN。

### ⚙️ 实验设置
- **平台**：Intel Xeon Gold 6330 CPU + NVIDIA A100 GPU
- **时钟周期**：5ns
- **负载电容**：10fF
- **目标位宽**：8-bit 与 16-bit 近似乘法器
- **优化算法支持**：
  - 多目标：MOEA/D、NSGA-II、SMAC、MOBO
  - 单目标：CBO、PSO、SA

### 🎯 评估指标
| 类别 | 指标 |
|------|------|
| **精度相关** | MRED（Mean Relative Error Distance）、NMED（Normalized Mean Error Distance） |
| **物理性能** | Delay（延迟）、Dynamic Power（动态功耗）、Area（面积） |
| **综合指标** | PDP（Power-Delay Product）、FOM（Figure of Merit: `-log(Pmax × √Abank × Dmax)`） |
| **任务质量** | 图像融合任务中的 PSNR、CIFAR-10 分类任务中的 Top-1 / Top-5 准确率 |

### 🆚 基线方法对比
- **Base design**：使用 Yang1 压缩器的标准设计（高精度但高能耗）
- **其他优化器**：比较 MOEA/D、NSGA-II、SMAC、MOBO 在 Pareto 前沿收敛性上的表现
- **传统流程**：直接使用 EDA 工具链（OpenROAD/OpenSTA/VCS）进行逐个评估（作为 GNN 的 ground truth）

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据

#### ✅ PEA-GNN 模型精度与加速效果（表1）
| 指标 | 8-bit MSE | 8-bit R² | 16-bit MSE | 16-bit R² | 推理时间（vs EDA） |
|-------|-----------|----------|------------|-----------|--------------------|
| MRED | 0.52×10⁻⁴ | 0.998 | 2.94×10⁻⁴ | 0.977 | **0.26s vs 37s → 142× 加速** |
| NMED | 9.25×10⁻⁴ | 0.996 | 2.66×10⁻⁴ | 0.959 | |
| Delay | 6.52×10⁻⁴ | 0.969 | 7.39×10⁻⁴ | 0.917 | |
| Area | 1.77×10⁻⁴ | 0.991 | 3.65×10⁻⁴ | 0.978 | |
| Power | 2.34×10⁻⁴ | 0.989 | 0.86×10⁻⁴ | 0.968 | **0.25s vs 116s → 464× 加速** |

> ✔️ 结论：PEA-GNN 在保持接近 EDA 精度的同时，实现**数百倍的速度提升**，支撑高效 Pareto 探索。

#### ✅ 近似乘法器优化结果（图5、图6、表2、表3）

##### 8-bit 乘法器（图像融合任务）
| 设计 | MRED | PDP (fJ) | PSNR (dB) | Level-2 调优后 PDP ↓ |
|------|--------|----------|-----------|---------------------|
| Base | 2.40×10⁻³ | 484 | 70.59 | — |
| Case5（紧约束） | 4.25×10⁻³ | 240 → **229** | 64.31 | **↓4.6%** |
| Case1（松约束） | 5.88×10⁻² | 193 → **191** | 46.01 | **↓1.0%** |

> ✔️ ACCO 成功在更小 PDP 下满足精度要求，且 Level-2 sizing 不破坏 MRED 约束。

##### 16-bit 乘法器（CIFAR-10 分类）
| 设计 | MRED | PDP (fJ) | Top-1 Acc (%) | Level-2 后 PDP ↓ |
|------|--------|----------|---------------|------------------|
| Base | 1.27×10⁻¹⁰ | 3874 | 66.6 | — |
| Case5（最严） | 1.77×10⁻⁴ | 1337 → **1289** | 65.7 | **↓3.6%** |
| Case1（最松） | 1.66×10⁻³ | 1216 → **1203** | 65.7 | **↓1.1%** |

> ✔️ 所有设计 Top-1 准确率变化 <1%，表明神经网络具有强误差容忍能力；同时 PDP 显著降低（**最高降幅超70%**）。

#### ✅ 晶体管级优化结果（图7）
- 对 Sabetz 等 8 种压缩器分别进行 sizing：
  - MOEA/D 表现出最佳 Pareto 收敛性
  - 所有优化后的 Pareto 前沿均左移（PDP 更低）
  - **器件级调优带来额外 3–10% 的 PDP 改善**

#### ✅ SRAM 宏优化结果（图8、图9）
- 架构级选择主导 PPA 性能：
  - 最佳配置集中在“外围开销”与“位线开关功耗”的平衡点附近
- 晶体管级 sizing 提升有限：
  - 受限于非理想外围电路的安全调优窗口
  - **主要作用为微调，难以改变整体 Pareto 趋势**

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **ACCO 框架有效解耦复杂设计空间**：
   - 将跨层级优化分解为“架构探索 + 器件调优”，兼顾效率与性能。
2. **PEA-GNN 实现高效准确建模**：
   - 支持大规模多目标搜索，在数千候选方案中快速定位 Pareto 最优解。
3. **精度约束可指导能量最优设计**：
   - 利用 NN 的误差容忍性，在可控误差下大幅削减 PDP（**最高减少 >70%**）。
4. **Level-1 决定大局，Level-2 精雕细琢**：
   - 架构选择决定主要性能边界；
   - 晶体管 sizing 提供进一步优化空间，但受限于外围电路限制。

### ⚠️ 局限性
1. **代理模型覆盖范围有限**：
   - PEA-GNN 当前仅针对特定压缩器家族训练，泛化至新拓扑需重新训练。
2. **目标函数较窄**：
   - 聚焦于 PDP 和精度，未考虑吞吐量、漏电流（leakage）、IR drop 或布线拥塞。
3. **后端签核集成不足**：
   - 缺少完整的 place-and-route、寄生提取、串扰分析等 signoff 级验证。
4. **SRAM 宏优化深度不够**：
   - bitcell 与外围电路协同优化不充分，存在进一步挖掘空间。

### 🔮 未来工作方向
1. 扩展 PEA-GNN 至更多近似算子（如加法器、激活函数）。
2. 引入多目标贝叶斯优化（MOBO）以更好处理离散-连续混合空间。
3. 集成 OpenROAD 完整流程，实现 RTL-to-GDSII 的闭环评估。
4. 探索面向特定 DNN 模型的任务级误差传播建模，替代手工设定 MRED/NMED 阈值。
5. 开发支持多种 PDK（如 Skywater130）的跨工艺迁移能力。

---

> 💡 **总结一句话**：  
> OpenACMv2 通过 **ACCO 两层解耦框架 + PEA-GNN 快速代理模型**，首次实现了在严格精度约束下的 **高效、可复现、开源的 Approximate DCiM 协同优化**，显著提升了设计效率与 PPA 表现，推动了近似计算在 CIM 领域的实用化进程。

</details>

---

### 10. [ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](https://arxiv.org/abs/2603.12740)

**Authors**: Shuo Yang, Soyeon Caren Han, Yihao Ding, Shuhe Wang, Eduard Hoy  
**Category**: cs.AI  
**Published**: 2026-03-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.12740v1  

#### Abstract
Large Language Model (LLM) agents are increasingly applied to complex, multi-step tasks that require interaction with diverse external tools across various domains. However, current LLM agent tool planning methods typically rely on greedy, reactive tool selection strategies that lack foresight and f...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning**

---

## **1. 主要贡献和创新点**

### **解决的问题**
当前的 **LLM Agent** 在进行多步任务时，通常依赖于 **greedy-based** 或 **reactive** 的工具选择策略，缺乏对长程依赖和工具间协同关系的考量。这导致：
- 早期错误决策无法回溯，影响后续步骤；
- 缺乏探索机制，容易陷入局部最优；
- 工具调用效率低，计算资源浪费严重。

此外，现有的 **search-based** 方法虽然引入了搜索树，但往往基于假设推理而非实际执行反馈，导致评分与真实效用脱节。

---

### **提出的新方法与创新思路**
本文提出了 **ToolTree**，一种受 **Monte Carlo Tree Search (MCTS)** 启发的新型工具规划框架，其核心创新在于：

#### ✅ **Dual-Feedback 机制**
- **Pre-evaluation (`r_pre`)**：在工具调用前，由 LLM Judge 预测该动作的潜在价值，作为 MCTS 中的选择策略先验。
- **Post-evaluation (`r_post`)**：在工具执行后，基于实际输出评估其贡献，用于反向传播更新节点价值。

> 这种“前瞻+回看”机制使 Agent 能够动态调整策略，兼具探索与利用能力。

#### ✅ **Bidirectional Pruning（双向剪枝）**
- **Pre-pruning**：在扩展节点前，若 `r_pre < T_pre`，则提前剪除不 promising 的分支，减少无效探索。
- **Post-pruning**：执行后若 `r_post < T_post`，则标记为不可扩展，防止继续浪费预算。

> 显著压缩搜索空间，提升单位计算下的准确率。

#### ✅ **无需训练的通用框架**
ToolTree 是一个 **plug-and-play** 模块，不依赖额外微调，可直接集成到不同 LLM 和 Agent 架构中，具备良好的泛化性和实用性。

---

### **相比现有方法的优势**
| 特性 | Greedy 方法 (如 ReAct) | Search-based 方法 (如 ToT, A*) | **ToolTree (Ours)** |
|------|------------------------|-------------------------------|--------------------|
| 是否有长程规划 | ❌ 单步决策 | ⚠️ 有搜索但无执行反馈 | ✅ 双重反馈驱动 |
| 是否支持回溯纠错 | ❌ | ⚠️ 有限 | ✅ 支持 |
| 是否高效利用计算资源 | ❌ 浪费严重 | ⚠️ 分支爆炸风险 | ✅ 双向剪枝控制 |
| 是否依赖训练 | ✅ 否 | ✅ 否 | ✅ 否 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
实验覆盖 **closed-set** 与 **open-set** 两类典型场景：

| 类型 | 数据集 | 描述 |
|------|-------|------|
| **Closed-set** | **GTA**, **m&m** | 固定小规模工具集（14~33个），强调多跳组合与输入一致性 |
| **Open-set** | **ToolBench**, **RestBench** | 大规模真实 API 库（上万级），需先检索再规划，贴近现实应用 |

---

### **实验设置与评估指标**

#### 🔧 **模型后端**
- 主要使用 **GPT-4o** 和轻量版 **GPT-4o-mini** 进行对比。

#### 📊 **评估指标**
| 场景 | 指标 |
|------|------|
| Closed-set (GTA/m&m) | **Tool F1**, **Argument F1**, **Planning F1**, **Execution F1**, **AVG F1** |
| Open-set (ToolBench/RestBench) | **Pass Rate**, **Win Rate**, **AVG Score** |

#### ⏱️ **统一控制变量**
- 所有方法共享相同的：
  - Tool schemas 与描述
  - Type pre-gating pipeline
  - Tool output caching 策略
  - Compute budget 与 rollout 限制（R_max=60）
- 确保公平比较仅反映 **planning strategy** 的差异。

---

### **基线方法对比**
涵盖从零样本到先进搜索算法的完整谱系：

| 类别 | 基线方法 |
|------|---------|
| **No Planning** | Zero-shot |
| **Greedy Reactive** | ReAct, Chain-of-Thought (CoT) |
| **Search-based** | Best-First, Tree-of-Thought (ToT), A*, LATS |
| **Open-set Specific** | DFSDT |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **Closed-set 结果（Table 1）**
| 方法 | GTA (GPT-4o) AVG | m&m (GPT-4o) AVG |
|------|------------------|------------------|
| Zero-shot | 57.78 | 80.58 |
| ReAct | 58.46 | 81.46 |
| ToT | 60.40 | 83.91 |
| A* | 62.52 | 84.74 |
| LATS | 64.78 | 86.45 |
| **ToolTree (Ours)** | **66.95** | **88.61** |

> ➤ 平均领先最强 baseline **约 2.2~2.2 分**，相对提升 **~10%**。

#### ✅ **Open-set 结果（Table 2）**
| 方法 | ToolBench (GPT-4o) AVG | RestBench-TMDB (GPT-4o) AVG |
|------|------------------------|------------------------------|
| Zero-shot | 48.79 | 53.14 |
| ReAct | 57.89 | 64.30 |
| DFSDT | 61.73 | 67.82 |
| LATS | 66.55 | 71.35 |
| **ToolTree (Ours)** | **69.04** | **74.50** |

> ➤ 在大规模 API 场景下仍保持显著优势，平均提升 **+2.5~3.1 分**。

---

### **消融实验结果（Table 3）**

| 变体 | Accuracy | Token Cost ↓ |
|------|----------|-------------|
| **Full ToolTree** | **76.44** | **18.2k** |
| -Pre-pruning | 75.28 | 20.4k |
| -Pre-evaluation | 71.80 | 21.1k |
| -Post-pruning | 75.82 | 22.9k |
| -Post-evaluation | 68.94 | 22.9k |
| -Both Pruning | 74.58 | 24.1k |
| -Both Evaluation | 66.70 | 24.3k |

> ➤ **Post-evaluation 移除导致最大下降（-7.5 pts）**，说明执行反馈至关重要；  
> ➤ **双向机制共同作用才能实现最高效率与精度平衡**。

---

## **4. 关键结论和发现**

### **主要发现**
1. **Dual-feedback 显著提升规划质量**  
   结合 `r_pre` 与 `r_post` 的 MCTS 框架能有效引导搜索方向，并通过真实反馈修正误判，避免“纸上谈兵”。

2. **Bidirectional Pruning 极大提高效率**  
   Pre-pruning 减少无效扩展，Post-pruning 快速终止失败路径，在固定预算下实现更优性能-时间权衡。

3. **方法具有强可扩展性与鲁棒性**
   - 在工具库从 14 增加到 **10,014** 时，性能仅下降 **1.62%**（Table 11），表明 pre-evaluation 能有效过滤噪声工具。
   - 对不同 LLM judge（GPT-4o, Gemini, LLaMA）表现出良好兼容性（Table 10），未出现严重过拟合。

4. **适用于多种 Agent 架构**
   ToolTree 可作为模块插入 LangChain、MetaGPT 等系统，平均带来 **+7 pts** 提升（Table 7），验证其即插即用价值。

---

### **局限性**
1. **依赖 LLM Judge 的稳定性**  
   尽管实验证明对 judge 错误有一定容忍度（Table 9），但极端情况下仍可能影响收敛。
   
2. **延迟高于纯 greedy 方法**  
   虽然效率优于多数 search 方法（Figure 3b），但仍高于 ReAct 等单路径方法，不适合极低延迟场景。

3. **未完全解决语义歧义问题**  
   当工具功能高度相似或描述模糊时，pre-evaluation 可能难以区分。

---

### **未来工作方向**
1. **动态调整 pruning threshold**  
   根据任务复杂度自适应调节 `T_pre` 和 `T_post`，进一步优化资源分配。

2. **引入多模态 feedback**  
   利用视觉、结构化日志等非文本信号增强 post-evaluation 的准确性。

3. **结合强化学习进行 meta-level 控制**  
   学习何时启用 ToolTree，何时退化为快速响应模式，构建混合智能体。

4. **部署至真实生产环境测试**  
   在真实 API 生态中验证长期稳定性与安全性。

---

> **总结**：ToolTree 提出了一种新颖且高效的 LLM Agent 工具规划范式，通过将 **MCTS** 与 **dual-feedback + bidirectional pruning** 相结合，在多个 benchmark 上实现了 **state-of-the-art 性能**，同时保持高计算效率，为未来复杂 Agent 系统的设计提供了重要参考。

</details>

---

### 11. [Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](https://arxiv.org/abs/2603.13085)

**Authors**: Jose Marie Antonio Mi\~noza, Paulo Mario P. Medina, Sebastian C. Iba\~nez  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 5.5  
**Type**: new  
**ArXiv ID**: 2603.13085v1  

#### Abstract
Understanding the theoretical foundations of attention mechanisms remains challenging due to their complex, non-linear dynamics. This work reveals a fundamental trade-off in the learning dynamics of linearized attention. Using a linearized attention mechanism with exact correspondence to a data-depe...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文总结：Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics**

---

## 1. **论文的主要贡献和创新点**

### **解决了什么问题**
该论文旨在从理论层面解释 **attention 机制的学习动态特性**，特别是其为何在实践中表现出高度灵活性的同时也容易受到对抗样本的影响。传统基于 **Neural Tangent Kernel (NTK)** 的分析框架假设足够宽的网络会收敛到一个固定的核函数（即“lazy training” regime），但这一假设在 attention 架构上失效。本文系统地揭示并解释了这种 **非收敛现象** 及其根本原因。

### **提出了什么新方法或新思路**
- **提出 “Influence Malleability”（影响可塑性）作为量化指标**  
  定义为模型对训练样本依赖程度随数据质量变化而动态调整的能力。高 malleability 表示模型能快速重新评估哪些训练样本是有帮助或有害的。
  
- **建立线性化 attention 与数据依赖核（data-dependent Gram-induced kernel）之间的精确对应关系**  
  证明线性化 attention $ f^{\text{att}}(X) = XXTX $ 对应于一个由 Gram 矩阵立方构成的核 $ K_{\text{LinAttn}} = (XX^T)^3 $，从而可以使用 NTK 框架进行严格分析。

- **揭示 spectral amplification（谱放大效应）是 NTK 非收敛的根本机制**  
  证明 attention 操作将 Gram 矩阵的条件数 $ \kappa(G) $ 放大为 $ \kappa(G)^3 $，导致需要宽度 $ m = \Omega(\kappa(G)^6) $ 才能收敛至无限宽 NTK 极限——远超实际可行范围。

### **相比现有方法的优势**
- **首次形式化地解释了 attention 不进入 kernel regime 的理论根源**，填补了 NTK 理论与现代架构之间的鸿沟。
- **引入 influence malleability 这一可测量量**，统一解释了 attention 的“双刃剑”特性：既能更好拟合任务结构（降低偏差），又更易受对抗攻击。
- **理论与实验紧密结合**，通过严格的数学推导和大量实验证明结论的普适性和稳健性。

---

## 2. **核心实验方法和设置**

### **使用的数据集**
- **MNIST**
- **CIFAR-10**
- （补充材料中还包括 Fashion-MNIST）

所有输入均归一化为单位 $ \ell_2 $ 范数以满足理论假设。

### **实验设置和评估指标**

#### **模型架构对比**
| 模型 | 描述 |
|------|------|
| **2L-ReLU** | 两层全连接 ReLU 网络，直接处理原始输入 |
| **MLP-Attn** | 在相同 MLP 前加入 **线性化 attention 层** $ f^{\text{att}}(X) = XXTX $，输出再经行归一化后送入 MLP |

> 注意：attention 是参数自由（parameter-free）且在整个训练集上一次性计算，属于 transductive 设计。

#### **训练配置**
- 优化器：Adam ($ \eta = 10^{-3} $)
- Batch size: 128
- Epochs: 500
- 正则化：$ \lambda = 10^{-3} $
- 初始权重服从 $ \mathcal{N}(0, 0.01^2) $

#### **评估指标**
1. **NTK Distance**: $ \|f_m - f_{\text{NTK}}\|_2 $，衡量有限宽度模型与无限宽度 NTK 预测器之间的差异。
2. **Influence Flip Rate**：在 top-10% 最具影响力的训练样本上施加对抗扰动后，其影响符号发生翻转的比例。
3. **Rank Correlation (Spearman’s ρ)**：扰动前后影响排序的相关性，越低表示 malleability 越强。
4. **Adversarial Training 效果**：比较标准训练 vs. PGD-AT 下的 malleability 变化。

#### **对抗扰动方式**
- **FGSM**（Fast Gradient Sign Method）
- **PGD**（Projected Gradient Descent）
- **MIM**（Momentum Iterative Method）
- 扰动预算 $ \epsilon = 0.3 $，采用 $ \ell_\infty $ 范数约束

#### **数据干预策略**
- **Curated**：移除 top-T 影响样本
- **Transformed**：用对抗样本替换 top-T 样本
- **Adversarial**：对全部训练数据施加 PGD 扰动

---

## 3. **主要实验结果和性能指标**

### **关键性能数据**

#### ✅ **NTK Distance 验证非收敛性（Table 1 & Figure 1）**

| Model / Width | m=16 | m=1024 | m=4096 |
|--------------|-------|--------|--------|
| **MNIST**     |       |        |        |
| 2L-ReLU      | 45.1  | 39.9   | 39.2   |
| MLP-Attn     | 10.3  | 33.3   | 43.4   |
| **CIFAR-10**  |       |        |        |
| 2L-ReLU      | 246.2 | 101.7  | 56.9   |
| MLP-Attn     | 3.7   | 10.4   | 12.6   |

> 观察：2L-ReLU 显示单调下降，符合 NTK 收敛预期；MLP-Attn 的 NTK distance **不降反升或非单调**，表明其始终未进入 kernel regime。

#### ✅ **Influence Flip Rate 展示高 malleability（Table 2 & 3）**

##### 多类分类（10-class）下的 PGD Flip Rate：

| Dataset     | Model       | PGD Flip Rate |
|-------------|-------------|----------------|
| MNIST       | 2L-ReLU     | 3.3%           |
|             | MLP-Attn    | **28.9%**      |
| CIFAR-10    | 2L-ReLU     | 3.1%           |
|             | MLP-Attn    | **19.1%**      |

👉 **MLP-Attn 的 influence malleability 达到 ReLU 网络的 6–9 倍！**

##### 二元分类下（Table 3）：
- 在 MNIST 上仍保持优势（~4×）
- 在 CIFAR-10 上差距缩小（~1×），与理论预测一致（因 $ \kappa(G) $ 更小）

#### ✅ **对抗训练显著提升 ReLU 的 malleability（Table 4）**

| Dataset     | Model       | Standard | Adv-Trained |
|-------------|-------------|----------|--------------|
| MNIST       | 2L-ReLU     | 3.3%     | **43.4%**    |
|             | MLP-Attn    | 28.9%    | 42.2%        |
| CIFAR-10    | 2L-ReLU     | 3.1%     | **36.5%**    |
|             | MLP-Attn    | 19.1%    | 38.6%        |

> 发现：**对抗训练使 ReLU 网络“被迫”获得类似 attention 的敏感性**，说明 malleability 可通过训练诱导；而 MLP-Attn 是**天生具备**此性质。

---

## 4. **关键结论和发现**

### **主要发现**
1. 🔴 **线性化 attention 不收敛于其无限宽度 NTK 极限**，即使在网络非常宽时依然如此。
2. 📌 **根本原因是 spectral amplification**：attention 将 Gram 矩阵的 condition number 立方放大，导致收敛所需宽度 $ m = \Omega(\kappa(G)^6) $，对于自然图像（如 CIFAR-10 的 $ \kappa(G)\sim 10^3 $）意味着 $ m > 10^{18} $，完全不可实现。
3. 💡 **influence malleability 是 attention 敏感性的量化体现**，其值比 ReLU 网络高出 6–9 倍。
4. ⚖️ **存在双重含义（dual implications）**：
   - ✅ **正面**：数据依赖核能更好地对齐任务结构，降低 approximation error（bias）；
   - ❌ **负面**：同一机制使其更容易被对抗样本操控，增加脆弱性。
5. 🔗 **architecture 决定 regime**：
   - 2L-ReLU → lazy training（稳定但僵化）
   - MLP-Attn → feature learning（灵活但敏感）

### **方法的局限性**
| 局限 | 说明 |
|------|------|
| **线性化近似** | 使用 $ f^{\text{att}}(X) = XXTX $ 替代完整 softmax attention，忽略了 row-wise normalization 的非线性影响 |
| **参数自由设计** | QKV 投影设为恒等映射，虽有理论推广（Proposition B.4），但仍简化了真实 Transformer 动态 |
| **小规模实验** | 实验限于 MNIST/CIFAR-10 和宽度 ≤ 4096，尚未扩展到大规模 vision transformers |
| **transductive 架构** | attention 在整个训练集上运行，不符合 inductive 学习设定，可能限制泛化解释力 |

### **未来工作方向**
1. **扩展至完整 softmax attention**：研究 row-wise softmax 归一化是否进一步加剧非收敛行为。
2. **多头 attention 与深层堆叠分析**：探索 multi-head 和 deep stack 如何累积 spectral amplification 效应。
3. **layer normalization 的作用建模**：理解 LN 是否起到稳定 kernel 或调节 malleability 的作用。
4. **开发鲁棒的 attention 架构**：利用低秩截断（truncated attention）等手段控制 condition number，恢复收敛性。
5. **intrinsic influence sensitivity 的实用诊断工具开发**：将 $ S(K,\lambda,y) = L_K \|y\|^2/\lambda^2 $ 应用于模型选择与防御设计。

---

> **一句话总结**：  
> Attention 的强大表达能力与其对抗脆弱性源于同一个根源——它脱离了 kernel regime，进入了 feature learning 区域；这种“脱离”由 **spectral amplification** 引发，并可通过 **influence malleability** 准确量化。

</details>

---

### 12. [Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking](https://arxiv.org/abs/2603.12831)

**Authors**: Zizhao Mo, Junlin Chen, Huanle Xu, Chengzhong Xu  
**Category**: cs.DC  
**Published**: 2026-03-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.12831v1  

#### Abstract
Nowadays, service providers often deploy multiple types of LLM services within shared clusters. While the service colocation improves resource utilization, it introduces significant interference risks for latency-sensitive (LS) services-which have strict SLO requirements for inference latency-and se...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文《Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking》总结

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
在现代数据中心中，服务提供商通常在同一集群上**共置（co-locate）多种类型的 LLM 服务**，主要包括两类：
- **Latency-Sensitive (LS) 服务**：如在线聊天机器人，对推理延迟有严格的 SLO 要求（如 TTFT 和 TPOT）。
- **Best-Effort (BE) 服务**：如后台任务（benchmarking、表单处理等），无严格 SLO，优先级较低。

然而，这种混合部署会引入严重的**资源干扰（interference）风险**：
- LS 服务因 BE 请求占用 GPU 资源而违反 SLO；
- BE 服务受限于 GPU 内存，无法充分利用闲置计算能力。

现有系统（如 Llumnix）通过为 LS 服务预留 GPU 内存（headroom）来隔离资源，但该策略存在两个根本缺陷：
1. **粗粒度控制**：仅管理内存，忽视了 GPU SM 和内存带宽的竞争，导致 LS 延迟波动大；
2. **资源浪费**：当 LS 负载低时，大量 GPU 计算资源闲置，而 BE 服务仍受内存限制。

---

### 提出了什么新方法或新思路
本文提出 **OmniServe** —— 一种新型的 LLM 推理系统，其核心是 **Attention Piggybacking** 机制，结合动态批处理调度策略，实现高效异构资源利用。

#### 主要创新点：

1. ✅ **Attention Piggybacking 机制**
   - 将 BE 服务的 **Attention 计算卸载到 CPU** 上执行；
   - 允许 GPU 在不等待 CPU 结果的情况下继续处理 LS 请求（**异步执行流**）；
   - 当 CPU 完成 Attention 后，其结果被“搭便车”（piggybacked）回 GPU，在后续层的 Dense 模块中以 **layer-wise batching** 方式合并执行。

2. ✅ **Layer-wise Batching 动态批处理策略**
   - 区别于传统的 token-wise batching，允许不同请求在不同模型层间动态进出批次；
   - 支持更细粒度的负载调控，提升 BE 吞吐的同时保障 LS 的 SLO。

3. ✅ **显式延迟建模与动态调度**
   - 构建模块级（module-level）延迟模型（`fA`, `fD`），精确预测 Attention 和 Dense 模块的执行时间；
   - 基于此进行 admission control 和 chunk prefill 控制，确保 SLO 不被违反。

4. ✅ **异步 CPU-GPU 协作架构**
   - 引入 CPU Attention Input/Output Queue 实现生产者-消费者模式；
   - 配合残差管理（residual store）保证计算正确性；
   - 支持分布式 CPU 集群协同处理大规模 KV 缓存。

---

### 相比现有方法的优势
| 维度 | 现有方法（如 Llumnix） | OmniServe |
|------|------------------------|----------|
| 干扰控制 | 仅靠 GPU 内存预留，粗粒度 | 利用 CPU 分担 Attention，减少 GPU 竞争 |
| 资源利用率 | GPU 计算常空闲，CPU 几乎不用 | 充分利用 CPU + GPU 异构资源 |
| 执行同步 | 必须同步等待所有 Attention 输出 | 异步解耦，避免 GPU 被阻塞 |
| 批处理灵活性 | 固定 token-wise 批次 | layer-wise 动态批处理，灵活调度 |
| SLO 保障能力 | 易受 BE 请求长度影响 | 显式建模 + 动态控制，稳定达标 |

---

## 2. 核心实验方法和设置

### 使用的数据集
- **LS 服务模拟**：基于 ShareGPT 数据集构建的在线聊天场景，采用 Poisson 分布提交请求。
- **BE 服务基准**：
  - **LongBench-v2**：平均输入/输出长度为 8,952 / 136 tokens，最大上下文达 12K，用于测试长文本生成；
  - **DailyMails**：平均输入/输出长度为 1,964 / 397 tokens，来自 Azure Public Datasets 的真实提交模式。

---

### 实验设置
- **硬件环境**：
  - 主节点：1 台 GPU 服务器，配备 **4×A100 80GB GPU**，Intel Xeon Gold 6342 CPU（24核），400GB RAM；
  - 辅助节点：4 台纯 CPU 服务器（同配置）；
  - 网络互联：默认 100 Gbps RoCE，部分实验测试 10 Gbps LAN；
- **模型规模**：
  - **Yi-34B**（2-way Tensor Parallelism）
  - **Llama-2-70B**（4-way TP）
- **精度格式**：BF16
- **运行时长**：每次实验持续 30 分钟

---

### 评估指标
| 服务类型 | 主要指标 |
|---------|---------|
| **LS 服务** | **SLO Attainment Rate**：<br>• TTFT（Time to First Token）达标率<br>• TPOT（Time Per Output Token）达标率 |
| **BE 服务** | **Token Generation Throughput**：<br>• 解码阶段吞吐量（tokens/sec） |

---

### 基线方法对比
| 基线 | 描述 |
|------|------|
| **Baseline A: Llumnix + vLLM-on-CPU** | Llumnix 在 GPU 上运行混合负载，超限 BE 请求交给 CPU 上的 vLLM 处理 |
| **Baseline B: NEO** | 将所有解码 Attention 卸载至 CPU，采用流水线方式执行；本文对其增强以支持 SLO 控制 |
| **Baseline C: Sarathi-Serve** | 纯 GPU 调度，始终优先 LS 请求，代表 SLO 最优上限（无 CPU 协同） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据

#### 📈 LS 服务 SLO 达标率提升
| 场景 | OmniServe vs. Llumnix (A) | 提升倍数 |
|------|----------------------------|---------|
| Llama-70B, LongBench-v2, 100Gbps | TPOT 达标率从 ~62% → **91.6%**（SLO=0.15s） | **1.48×** |
| Yi-34B, DailyMails, 动态负载 | SLO 达标率最高达 **1.42×** | 1.42× |
| Bursty 流量（5s/10s 变化） | SLO 达标率优于 Llumnix **1.23×**, 优于 NEO **1.13×** | 最高 1.23× |

> ✅ OmniServe 在各种负载下均接近 Sarathi-Serve 的 SLO 表现，说明其未牺牲 LS 性能。

---

#### 🚀 BE 服务吞吐显著提升
| 场景 | OmniServe vs. Baseline | 提升倍数 |
|------|------------------------|---------|
| Llama-70B, LongBench-v2, 重负载 | 吞吐从极低水平提升至 **9.85×** 超过 Llumnix | **9.85×** |
| Yi-34B, LongBench-v2, 1G4C 配置 | 平均提升 **~7–9×** | 最高 9.1× |
| DailyMails 数据集 | 吞吐提升 **高达 9.1×** | 9.1× |
| 仅使用 GPU 本地 CPU（1台） | 仍可达到 **3.47×** 提升 | 3.47× |

> ✅ 即使在网络带宽受限（10Gbps）或 CPU 数量有限条件下，OmniServe 依然表现出色。

---

### 消融实验结果

#### 🔍 建模准确性（Modeling Accuracy）
| 模型 | 平均准确率 | P90 准确率 |
|------|-----------|------------|
| Yi-34B | **95.7%** | ≥93.3% |
| Llama-70B | **94.5%** | ≥92.1% |

> 高精度建模支撑了有效的 admission control 和调度决策。

#### 🔒 Admission Control 效果
- **无 admission control**：高负载下 TTFT SLO 达标率急剧下降至 <50%；
- **启用 admission control**：维持 **94.1% TTFT SLO 达标率**，较基线提升 **43.3%**；
- 同时保持 LS 解码吞吐几乎不变（差异 ≤6%），证明资源利用高效。

#### ⚙️ 运行开销分析
- **队列读写 + 残差存储**：≤75 μs（400 请求并发）；
- **残差加载（非连续访问）**：约 0.5 ms（可接受）；
- **RAY 框架引入延迟**：<1%，可忽略；
- **Piggybacking 对 LS 影响**：SLO 仅下降 **0.3% 左右**，随 CPU 数量增加基本稳定。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **CPU 资源可用于缓解 GPU 干扰**：将 BE 的 Attention 卸载至 CPU 是可行且高效的，尤其适用于内存敏感型任务；
2. ✅ **异步执行可打破传统同步瓶颈**：Attention Piggybacking 成功解耦 CPU-GPU 流水线，避免 GPU 被慢速 CPU 阻塞；
3. ✅ **layer-wise batching 比 token-wise 更灵活**：支持跨层动态调度，极大提升了 BE 请求的容纳能力和系统弹性；
4. ✅ **显式延迟建模是 SLO 保障的关键**：精准预测各模块耗时，使得 admission control 和 piggybacking 控制成为可能；
5. ✅ **OmniServe 可无缝集成主流技术**：兼容 Prefill-Decode disaggregation、chunk prefill、MoE、LoRA 等先进架构。

---

### 方法的局限性
1. ❗ **依赖 PCIe 带宽**：虽然通信量小，但在极端低带宽环境下（<10Gbps）仍可能成为瓶颈；
2. ❗ **KV cache 迁移成本**：频繁的 CPU-GPU cache swap 会影响性能，需配合智能缓存管理；
3. ❗ **当前聚焦双优先级场景**：虽可在讨论中扩展至多优先级，但尚未实现在生产中验证；
4. ❗ **未考虑 CXL 或其他新型互连技术**：未来若普及，可进一步优化数据迁移效率。

---

### 未来工作方向
1. ➕ **支持多优先级 SLO 服务**：基于 SLO 要求设定不同的 piggybacking 延迟窗口；
2. ➕ **结合 CXL/NVM 技术**：将 KV cache 存储于共享内存池，降低迁移开销；
3. ➕ **自适应 offloading 策略**：根据实时负载自动选择是否卸载 Attention 或部分 Dense 模块；
4. ➕ **面向边缘设备部署**：适配资源受限环境下的轻量化版本；
5. ➕ **安全与容错机制增强**：应对 CPU 节点故障、网络中断等情况下的鲁棒性设计。

---

> 💡 **总结一句话**：  
> OmniServe 通过 **Attention Piggybacking + layer-wise batching + 显式延迟建模**，首次实现了在保障 LS 服务 SLO 的前提下，将 BE 服务吞吐提升近一个数量级，为 LLM 混合负载调度提供了全新的高效解决方案。

</details>

---

### 13. [No More DeLuLu: Physics-Inspired Kernel Networks for Geometrically-Grounded Neural Computation](https://arxiv.org/abs/2603.12276)

**Authors**: Taha Bouhsine  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 5.0  
**Type**: new  
**ArXiv ID**: 2603.12276v1  

#### Abstract
We introduce the yat-product, a kernel operator combining quadratic alignment with inverse-square proximity. We prove it is a Mercer kernel, analytic, Lipschitz on bounded domains, and self-regularizing, admitting a unique RKHS embedding. Neural Matter Networks (NMNs) use yat-product as the sole non...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：No More DeLuLu: Physics-Inspired Kernel Networks for Geometrically-Grounded Neural Computation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代神经网络将**几何计算**（如点积）与**非线性激活**（如ReLU）分离，导致以下问题：
- **信息丢失**：负值被置零（如ReLU），破坏输入间的连续关系；
- **依赖额外模块**：需引入 BatchNorm、LayerNorm 或 Attention 来恢复表达能力；
- **训练不稳定**：梯度爆炸/消失、对初始化敏感；
- **架构冗余**：多层组合（Linear + Activation + Norm）增加复杂性。

本文旨在统一神经计算中的**对齐（alignment）** 和 **邻近性（proximity）**，提出一种更简洁、物理启发式的神经算子。

---

### 🚀 提出的新方法与核心思想

#### **E-product 算子**
定义如下：
$$
E(w, x) = \frac{(w^\top x)^2}{\|w - x\|^2 + \epsilon}
$$
该算子融合了两个关键因素：
- **分子 $(w^\top x)^2$**：衡量向量间的方向对齐（alignment）；
- **分母 $\|w - x\|^2 + \epsilon$**：基于逆平方律的距离衰减机制，形成“势阱”（potential well）。

> 受物理学中引力、静电场等**inverse-square laws**启发，响应在权重原型 $w$ 附近高且局部化。

---

#### **Neural Matter Networks (NMNs)**
以 E-product 作为唯一非线性操作构建的新型网络架构：
- 替代传统 `Linear -> Activation -> Norm` 流程；
- 所有层均使用 E-product 构建，无需显式激活函数（如 GeLU/ReLU）；
- **内在自正则化（self-regularizing）**：输出天然有界，无需 LayerNorm。

---

### 🔍 相比现有方法的优势

| 特性 | 传统网络（如 GPT-2） | NMN / Aether-GPT2 |
|------|------------------------|--------------------|
| 非线性来源 | 显式激活函数（ReLU, GeLU） | 内生于 E-product 结构 |
| 正则化机制 | LayerNorm, Dropout | 分母隐含归一化（self-regulation） |
| 梯度稳定性 | 可能出现梯度爆炸/消失 | 梯度随距离自然衰减（gradient decay） |
| 数学性质 | 黑箱映射 | Mercer kernel，具备 RKHS 表示 |
| 内存开销 | 存储激活值 | 减少 15–25%（无激活存储） |
| 可解释性 | 较弱 | 原型具有几何意义，支持分析 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 任务 | 数据集 | 规模 |
|------|--------|------|
| 图像分类 | MNIST | 60k 训练，10k 测试 |
| 极端分类 | Eurlex-4K | 多标签文本分类，约 4k 类 |
| 语言建模 | FineWeb | 2.5B tokens，用于训练 Aether-GPT2 |

---

### ⚙️ 实验设置与评估指标

#### **通用设置**
- 所有模型使用 Adam/AdamW 优化器；
- 在相同参数预算下比较（如 Aether-GPT2 ≈ 124M params）；
- 使用 BF16 混合精度训练，确保公平性。

#### **评估指标**

| 任务 | 主要指标 |
|------|----------|
| MNIST | Accuracy, Prototype Evolution ($\Delta \|w\|$), Superposition Robustness |
| Eurlex-4K | P@1, P@3, P@5, PSP@1–5（propensity-scored precision） |
| 语言建模 | Train/Val Loss, Throughput (tok/s), Calibration Curve |
| 消融实验 | 是否加入 LayerNorm 后的收敛情况 |

---

#### **基线方法对比**
- **图像分类**：线性分类器（Linear + Softmax）
- **极端分类**：Inner Product Classifier
- **语言模型**：标准 GPT-2 架构（Scaled Dot-Product Attention + MLP with GeLU）

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据

#### **MNIST 分类（10-neuron classifier）**

| 方法 | Accuracy | $\Delta \|w\|$（原型变化） | Scaling Factor $s$ |
|------|----------|----------------------------|---------------------|
| Linear | 92.08% | +13.8%（持续增长） | — |
| **NMN (E-product)** | **92.38%** | **-4.5%（收缩稳定）** | 2.68 |

> ✅ NMN 不仅准确率略优，且原型演化受控，体现 **bounded response** 优势。

#### **Superposition Robustness（符号翻转测试）**

| 输入变换 | Linear Classifier | NMN (E-product) |
|---------|-------------------|------------------|
| 原始输入 | 92.04% | 92.18% |
| $w \to -w$（原型取反） | **0.01%（崩溃）** | **87.87%（保持鲁棒）** |

> ✅ 因为 E-product 使用 $(w^\top x)^2$，对符号不敏感，具备天然超位置不变性。

---

#### **Extreme Classification (Eurlex-4K)**

| 方法 | P@1 | P@3 | P@5 | PSP@1 | PSP@3 | PSP@5 |
|------|-----|-----|-----|-------|-------|-------|
| Inner Product | 0.6235 | 0.5041 | 0.4125 | 1.2117 | 1.0215 | 0.8587 |
| **E-product** | **0.6465** | **0.5114** | **0.4271** | 1.1542 | 0.9664 | 0.8443 |

> ✅ E-product 在主流 Top-K Precision 上显著领先，适合 top-label retrieval；
> ❗ 虽然 PSP 指标稍低，但反映的是倾向性偏差校正后的表现，不影响主任务优势。

---

#### **语言建模：Aether-GPT2 vs GPT-2**

| 指标 | GPT-2 | Aether-GPT2 | 改进 |
|------|--------|--------------|------|
| Final Train Loss | 4.1969 | 4.0479 | ↓ 1.49% |
| **Final Val Loss** | **4.6417** | **4.5747** | **↓ 1.45%** |
| Throughput | Comparable | Comparable | ✅ |
| Memory Usage | — | **↓ 15–25%** | ✅（无激活存储） |
| LayerNorm | Yes | **No** | ✅ 架构简化 |

> ✅ Aether-GPT2 在相同参数量下取得更低验证损失，且无需 LayerNorm。

---

#### **消融实验：LayerNorm 兼容性**

| 配置 | Validation Loss | 状态 |
|------|----------------|------|
| Aether (no LayerNorm) | 4.5747 | ✅ 收敛 |
| Aether + Pre-LN | >10 | ❌ 发散 |
| Aether + Post-LN | >10 | ❌ 发散 |

> ⚠️ **LayerNorm 与 E-product 不兼容**：其强制归一化会破坏 E-product 依赖的原始尺度信息（如 $\|x\|^2$, $\|w\|^2$），导致训练崩溃。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **E-product 是一个有效的 Mercer kernel**
   - 满足对称性、连续性和正定性；
   - 存在唯一的 RKHS 表示，支持理论分析与泛化边界推导。

2. **NMNs 实现了“几何接地”的神经计算**
   - 将 alignment 与 proximity 统一于单一物理启发算子；
   - 形成围绕原型的“势阱”，实现局部化响应。

3. **内在自正则化取代外部归一化**
   - 输出自然有界（$\lim_{\|x\|\to\infty} E(w,x) = \|w\|^2 \cos^2\theta$）；
   - 梯度随距离衰减（$\nabla_x E \to 0$ 当 $\|x\| \to \infty$）；
   - **因此可安全移除所有 LayerNorm 层**。

4. **更强的表示学习能力**
   - 单个 E-unit 可解决 XOR 问题（非线性可分）；
   - 在 MNIST 上原型更具结构性（见 Fig. 6）；
   - NTK 分析显示：不同类输入趋于正交时，NTK 响应趋零 → **隐式类别分离**。

5. **高效实用**
   - 前向/反向传播复杂度仍为 $O(Bnd)$；
   - 实际内存减少 15–25%；
   - BF16 下数值稳定，无需梯度裁剪。

---

### ⚠️ 方法的局限性

| 局限 | 说明 |
|------|------|
| **FLOPs 开销较高** | 每 neuron 约为 Linear+ReLU 的 2×，因涉及更多乘加运算；可通过硬件优化缓解。 |
| **对 $\epsilon$ 敏感** | 过小可能导致数值不稳定，过大削弱局部性；建议按噪声水平设置 $\epsilon^* \propto d \sigma^2$。 |
| **暂未扩展到 CNN/ViT 主干** | 当前验证集中于 MLP 和 Transformer；卷积形式尚待探索。 |
| **缺乏大规模下游任务评测** | 如 GLUE、SQuAD 等 NLP benchmark 尚未报告。 |

---

### 🔮 未来工作方向

1. **设计 E-Conv 层**：将 E-product 推广至局部感受野，应用于视觉骨干网络；
2. **动态 $\epsilon$ 调整机制**：根据输入分布或训练阶段自动调节正则项；
3. **结合 Diffusion 或 Energy-Based Models**：利用势场结构建模生成过程；
4. **探索更深的纯 kernel 架构**：研究多层 E-layer 的复合特性；
5. **部署优化**：开发专用 CUDA kernel 以加速 E-product 计算。

---

## 总结

> **Aether-GPT2 成功证明：我们可以不再依赖 “DeLuLu”（Dropout, LayerNorm, ReLU）堆叠来训练深层模型。通过引入物理启发的 E-product 算子，NMNs 实现了更简洁、更稳定、更具解释性的神经网络新范式。**

✅ **一句话总结**：  
**用一个融合对齐与距离的 kernel 算子（E-product），替代三大人工组件（Activation + Norm + Attention），构建出更简单、更强大、更 grounded 的 Neural Matter Networks。**

</details>

---

### 14. [Steve-Evolving: Open-World Embodied Self-Evolution via Fine-Grained Diagnosis and Dual-Track Knowledge Distillation](https://arxiv.org/abs/2603.13131)

**Authors**: Zhengwei Xie, Zhisheng Chen, Ziyan Weng, Tingyu Wu, Chenglong Li, Vireo Zhang, Kun Wang  
**Category**: cs.AI  
**Published**: 2026-03-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.13131v1  

#### Abstract
Open-world embodied agents must solve long-horizon tasks where the main bottleneck is not single-step planning quality but how interaction experience is organized and evolved. To this end, we present Steve-Evolving, a non-parametric self-evolving framework that tightly couples fine-grained execution...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Steve-Evolving: Open-World Embodied Self-Evolution via Fine-Grained Diagnosis and Dual-Track Knowledge Distillation

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前的 **open-world embodied agent**（如在 Minecraft 中运行的智能体）虽然能通过 LLM 进行任务分解与执行，但在面对**长视野、多依赖子目标的复杂任务**时，性能远低于人类玩家。其瓶颈不在于单步决策质量，而在于如何有效组织和演化交互经验。

现有方法（如 Jarvis-1、Optimus-1）大多仅将成功轨迹作为检索实例存储，或将失败经验简单丢弃，缺乏对失败原因的**细粒度归因机制**，导致无法从错误中提炼出可复用的安全约束。

---

### 🚀 提出的新方法与核心思路
作者提出 **Steve-Evolving** —— 一种**非参数化（non-parametric）的自演化框架**，通过以下三个阶段实现经验的持续进化：

#### （1）Experience Anchoring（经验锚定）
- 将每次子目标尝试固化为结构化的经验元组 `e = (pre-state, action, diagnosis-result, post-state)`。
- 构建三层经验空间（summary → index → document），支持高效、可审计的召回。
- 引入多维索引（条件签名、空间哈希、语义标签）提升检索效率。

#### （2）Dual-Track Experience Distillation（双轨经验蒸馏）
- **正向轨道（Skill Extraction）**：从成功轨迹中提取可复用技能（`Kskill`），包含前置条件、操作流程和验证标准。
- **负向轨道（Guardrail Extraction）**：从失败案例中提炼防御性“护栏”规则（`Kguard`），明确禁止在特定触发条件下执行高风险动作。

#### （3）Knowledge-Driven Closed-Loop Control（知识驱动闭环控制）
- 在规划阶段，将检索到的 `Kskill` 和 `Kguard` 注入 LLM 上下文，引导更安全、高效的计划生成。
- 执行过程中若连续失败达到阈值，则触发 **Local Replanning**，动态更新约束并重新规划路径。
- 整个过程无需模型参数更新，实现真正的**无监督自我演化**。

---

### 🔍 相比现有方法的优势
| 方法 | 是否保留失败经验 | 是否有细粒度诊断 | 是否提取结构化知识 | 是否支持动态重规划 |
|------|------------------|------------------|--------------------|---------------------|
| Jarvis-1 | ❌ 仅存成功轨迹 | ❌ 二值反馈 | ❌ 实例级检索 | ❌ 静态检索 |
| Optimus-1 | ✅ 存储失败记录 | ❌ 无诊断信号 | ⭕ 图谱式知识 | ❌ 无主动干预机制 |
| **Steve-Evolving** | ✅ 完整记录 | ✅ 13类状态观测 + 11种失败归因 | ✅ 技能 + 护栏双轨蒸馏 | ✅ 动态注入 + 局部重规划 |

> ✅ **优势总结**：
> - 实现了从“经验积累”到“经验演化”的范式转变；
> - 利用细粒度诊断信号建立精准因果链，使失败可解释、可预防；
> - 支持跨任务的知识迁移与累积，形成可持续增长的能力体系。

---

## 2. 核心实验方法和设置

### 🧪 数据集
- 使用 **MineRLHumanSurvival-v0** 环境，基于 Project Malmo 和 **MCU (minestudio)** 接口构建。
- 采用 **MCU tech-tree task suite**，共 70 个任务，分为 7 组，覆盖工具制造、采矿、交易、修理等典型长视野任务。
  
| 任务组 | 示例任务 |
|--------|----------|
| Wooden | 收集原木、制作木板 |
| Iron | 开采铁矿、村民交易 |
| Diamond | 挖掘钻石、修复装备 |
| Armor | 制作皮革/铁甲、头盔修复 |

> 💡 特点：任务具有明显的先决条件链（prerequisite chain），适合测试长期依赖下的能力演化。

---

### 📊 实验设置与评估指标

#### 评估协议
- 每轮 episode 在任务成功或预算耗尽后终止。
- 报告 **Success Rate (SR, %)**，即多个随机种子下的平均成功率。
- 所有方法共享相同环境配置、起始状态和每轮交互预算。

#### 对比基线
| 基线方法 | 描述 |
|---------|------|
| **Jarvis-1** | 成功轨迹存入多模态记忆库，CLIP 向量相似性检索 |
| **Optimus-1** | 分离“知识图谱”与“抽象经验池”，支持自由探索学习 |
| **STEVE-1** | 不使用 LLM 规划器的基准 |
| **Human-level** | 人类玩家表现（理想上限） |

#### 主干 LLM
使用五种主流 LLM 进行公平比较：
- `qwen3.5-flash`, `qwen3.5-plus`, `GLM-4.7`, `gemini-3-flash`, `gemini-3-pro`

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table 2）

| Method | Overall SR (%) | Iron SR (%) | Diamond SR (%) | Armor SR (%) |
|--------|----------------|-------------|----------------|--------------|
| Jarvis-1 (+qwen3.5-plus) | 42.59 | 35.57 | 9.03 | 15.58 |
| Optimus-1 (+qwen3.5-plus) | 47.42 | 46.06 | 11.42 | 19.09 |
| **Steve-Evolving (+qwen3.5-plus)** | **52.52** | **55.83** | **17.06** | **27.63** |
| Human-level | 100.00 | 100.00 | 100.00 | 100.00 |

> ✅ **结论**：
> - Steve-Evolving 在所有主干 LLM 上均取得最高总体成功率；
> - 在后期任务（Iron 及以上）中优势尤为显著，表明其特别适用于**长依赖、易失败场景**；
> - 性能增益稳定跨越不同规模 LLM，说明改进来自框架本身而非模型容量。

---

### 🔬 消融实验结果（见 Figure 4）

在 `qwen3.5-plus` 上进行消融研究，聚焦于四个困难任务组（Iron/Redstone/Diamond/Armor）：

| 设置 | Iron SR (%) | Redstone SR (%) | Diamond SR (%) | Armor SR (%) |
|------|-------------|------------------|----------------|---------------|
| Full Model (Steve-Evolving) | 55.4 | 30.6 | 17.1 | 27.4 |
| w/o KnowledgeVisibility | 21.9 | 12.6 | 3.1 | 9.5 |
| w/o GuardDistill | 45.4 | 21.7 | 10.1 | 18.3 |
| w/o SkillDistill | 52.4 | 29.0 | 14.6 | 24.1 |
| Planning Only | 0.0 | 0.0 | 0.0 | 0.0 |

> 🔍 **关键发现**：
> - 移除 `KnowledgeVisibility` 导致最大下降（-33.5% Iron SR），证明**知识注入是闭环生效的关键**；
> - `GuardDistill` 蒸馏失败约束对防重复犯错至关重要；
> - `SkillDistill` 提供辅助增益，体现技能复用价值；
> - 单纯依赖 LLM 规划器几乎无法完成这些任务。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **经验不应只是被积累，更应被演化**  
   Steve-Evolving 将原始交互逐步提炼为结构化知识（技能 + 护栏），再反哺至规划环节，实现了“经验生命周期”。

2. **细粒度诊断是根因归因的基础**  
   传统的二值成败判断不足以支撑可靠的知识蒸馏。本文设计了包含 **13 类状态观测** 和 **11 种失败归因** 的诊断系统，确保失败模式可识别、可纠正。

3. **双轨蒸馏机制实现正负反馈闭环**  
   - 正向：成功 → 技能 → 加速执行；
   - 负向：失败 → 护栏 → 避免重蹈覆辙；
   - 二者共同构成“试错—学习—优化”的自演化循环。

4. **性能随经验积累持续上升**  
   实验显示，随着任务执行次数增加，Steve-Evolving 的成功率呈明显上升趋势，而静态检索方法则趋于饱和。

---

### ⚠️ 方法的局限性
1. **依赖高质量诊断信号输入**  
   若底层执行器无法提供足够细粒度的状态反馈（如缺少 GUI/block 状态监控），则蒸馏效果受限。

2. **知识表示仍以文本为主**  
   当前技能与护栏以自然语言形式存储，尚未引入程序化表达（code-as-policy），限制了精确控制能力。

3. **跨域泛化能力待验证**  
   当前实验集中在 Minecraft 场景，是否适用于其他开放世界（如 Web、机器人）尚需进一步探索。

---

### 🔮 未来工作方向
1. **引入程序化知识表示**  
   结合 Voyager 的代码技能库思想，将部分技能编码为可执行函数，提升自动化程度。

2. **扩展多模态诊断能力**  
   融合视觉异常检测（如 lava 区域识别）、声音提示（mob 出现）等信号，增强环境感知维度。

3. **构建通用型经验演化引擎**  
   将 Steve-Evolving 抽象为通用框架，适配于 WebAgent、Robotics、Game AI 等多种 embodied 场景。

4. **探索社会性经验共享机制**  
   多个 Steve-Evolving agent 之间共享知识库，模拟群体智慧演化路径。

---

> 📌 **一句话总结**：  
> Steve-Evolving 通过 **fine-grained diagnosis + dual-track distillation + knowledge-driven control** 的闭环机制，首次实现了开放世界具身智能体的**无参数自我演化**，为构建可持续成长的通用智能体提供了新范式。

</details>

---

### 15. [RTD-Guard: A Black-Box Textual Adversarial Detection Framework via Replacement Token Detection](https://arxiv.org/abs/2603.12582)

**Authors**: He Zhu, Yanshu Li, Wen Liu, Haitian Yang  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 4.5  
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
文本对抗攻击通过微小、人类难以察觉的扰动误导深度学习模型，对 NLP 系统构成严重安全威胁。现有的**对抗样本检测方法**通常依赖以下至少一种不切实际的假设：
- 需要已知的对抗样本进行训练；
- 白盒访问目标模型（如梯度或内部特征）；
- 大量查询受害者模型。

这些限制使其在真实场景中难以部署，尤其是在商业 API 或资源受限环境中。

RTD-Guard 的目标是设计一个**严格黑盒、无需对抗训练数据、低查询开销**的高效检测框架，以应对现实世界中的部署挑战。

---

### 提出的新方法与核心思想
作者提出 **RTD-Guard**，一种基于 **Replaced Token Detection (RTD)** 的零样本、黑盒对抗检测框架，其核心洞察如下：

> 文本对抗攻击中的词替换操作，在结构上与 RTD 预训练任务高度相似：两者都涉及将原始词替换为其他词，并判断哪些位置被修改过。

因此，一个在 RTD 任务上预训练好的判别器（如 ELECTRA 的 discriminator），天然具备识别“上下文不一致”词的能力——而这正是对抗扰动的关键特征。

#### RTD-Guard 四步流程：
1. **Perturbation Localization**：使用冻结的 RTD 判别器计算每个 token 被替换的概率（即上下文异常程度）。
2. **Intervention via Masking**：选择 top-k 最可疑的 token，用 `[MASK]` 替换。
3. **Confidence Shift Measurement**：向受害模型发送原始输入和掩码后的输入，测量预测置信度的变化。
4. **Detection Decision**：若置信度变化超过阈值 $ T $，则判定为对抗样本。

该过程仅需 **两次黑盒查询** 和一次前向传播，完全无需微调或访问模型内部参数。

---

### 相比现有方法的优势
| 维度 | RTD-Guard | 现有方法局限 |
|------|-----------|----------------|
| **是否需要对抗训练数据** | ❌ 否（zero-shot） | ✅ 多数方法（如 DISP, ADFAR）需要 |
| **是否白盒访问** | ❌ 否（strict black-box） | ✅ GradMask, MLE 等需梯度或隐藏层 |
| **查询复杂度** | ✅ $ O(1) $（仅 2 次查询） | ❌ WDR ($ O(L) $), VoteTRANS ($ O(L \cdot T) $) 开销大 |
| **通用性与鲁棒性** | ✅ 对多种攻击（TextFooler, BAE, TF-adj 等）均有效 | ❌ PPL 在语义保持攻击下失效；FGWS 依赖高频词规则 |

RTD-Guard 是首个将 **RTD 预训练能力直接迁移用于对抗检测** 的工作，实现了高性能与高实用性的统一。

---

## 2. 核心实验方法和设置

### 使用的数据集
实验基于 **TEXTATTACK** 提供的标准对抗基准，在三个主流分类数据集上进行评估：

| 数据集 | 任务 | 类别数 | 中位长度 | 测试样本数（原/对抗） |
|--------|------|--------|----------|------------------------|
| **IMDB** | 情感分类（电影评论） | 2 | 161 | 25K / 10K |
| **AG-News** | 新闻主题分类 | 4 | 44 | 7.6K / 7.6K |
| **Yelp** | 情感分类（餐厅评论） | 2 | 152 | 38K / 5K |

所有对抗样本由四种 state-of-the-art 攻击生成，针对 fine-tuned BERT 模型。

---

### 攻击方法
聚焦于最具威胁的 **word-level 攻击**，因其在隐蔽性、成功率和语义保留之间达到最佳平衡：
- **TextFooler**
- **PWWS**
- **BAE**
- **TF-adj**

---

### 基线方法对比
涵盖四类典型检测范式：

| 类型 | 方法 | 特点 |
|------|------|------|
| **外部语言模型评分** | PPL | 使用 GPT-2 的困惑度作为异常分数 |
| **静态启发式规则** | FGWS | 替换低频词为其最常见同义词 |
| **白盒特征密度估计** | MLE, RDE | 基于输出分布或隐藏表示建模密度 |
| **白盒梯度引导** | GradMask | 掩码高梯度重要性 token 并观察置信变化 |
| **黑盒查询密集型** | WDR | 删除每个词并训练分类器，$ O(L) $ 查询 |
| **黑盒投票机制** | VoteTRANS（效率分析中提及） | 多次变换后硬标签投票，$ O(L \cdot T) $ 查询 |

---

### 评估指标
采用三项广泛使用的检测性能指标：
- **AUC**：ROC 曲线下面积，衡量整体区分能力
- **F1-Score**：精确率与召回率的调和平均
- **TPR@10%FPR (TPR10)**：在 10% 假阳性率下的真阳性率，反映低误报条件下的敏感性

越高越好。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2）
RTD-Guard 在绝大多数设置下取得 **SOTA 性能**：

| 指标 | 表现 |
|------|------|
| **TPR10** | 在 11/12 种 dataset-attack 组合中排名第一 |
| **F1-Score** | 多数情况下最优或接近最优 |
| **AUC** | 显著优于轻量级方法（PPL, FGWS），且媲美甚至超越白盒方法 |

例如在 **AG-News + TextFooler** 上：
- RTD-Guard: **TPR10=97.5**, F1=94.9, AUC=98.4
- 第二名 WDR: TPR10=96.5, F1=93.8, AUC=97.2

在更具挑战性的 **BAE 和 TF-adj**（强调流畅性和语法正确性）攻击下，PPL 和 FGWS 性能急剧下降，而 RTD-Guard 仍保持强劲表现。

---

### 与基线方法的对比结果
- **vs. PPL & FGWS**：尽管满足相同黑盒约束（无训练数据、低查询），但 RTD-Guard 显著更优，说明利用 **模型内在语言能力** 比依赖外部 LM 或静态规则更有效。
- **vs. 白盒方法（MLE, RDE, GradMask）**：虽然部分白盒方法性能尚可，但 RTD-Guard 在多数情况下持平甚至反超，证明了其**无需内部访问即可实现强检测力**。
- **vs. 查询密集型方法（WDR）**：WDR 性能优异但需 $ O(L) $ 查询，无法实时部署；RTD-Guard 仅需 2 次查询，效率高出一个数量级。

---

### 消融实验结果

#### RQ1: 不同干预策略的影响（Table 3）
比较四种干预方式：
- `[MASK]`（默认）
- `[UNK]`
- 删除（DEL）
- MLM 填充

结果表明：**不同干预策略性能差异极小**，说明检测效果主要取决于 **定位精度** 而非具体干预形式。这验证了核心假设：只要准确找到扰动位置，任何破坏其影响的操作都能引发显著置信度偏移。

#### RQ2: RTD vs. Gradient-based 定位（Figure 3 & 分析）
- **GradMask** 依赖梯度重要性，在 clean 示例中会误伤语义核心词（如情感形容词），导致增加 $ k $ 时 clean accuracy 下降、F1 恶化。
- **RTD-Guard** 基于上下文一致性打分，对 clean 文本中的“最不自然”词通常是专有名词或连接词，不影响语义主干。因此随 $ k $ 增加仍能稳定提升检测效果。

这一现象揭示了“信号纠缠”（signal entanglement）问题：梯度信号无法区分**恶意扰动**与**正常语义重要性**，而 RTD 信号则正交于此。

#### RQ3: RTD 模型规模影响（Table 4）
使用 ELECTRA-small (14M), base (110M), large (335M) 进行测试：
- 所有尺寸均可有效检测；
- 更大规模模型带来持续性能增益（如 AUC 提升 0.5–1.5%）；
- 时间成本线性增长（Small: 28.8s → Large: 35.3s on AG-News/TextFooler）；
  
表明存在 **精度-延迟权衡**，可根据部署需求灵活选择。

---

## 4. 关键结论和发现

### 主要发现
1. **对抗扰动的本质是上下文不一致**：成功的 word-substitution 攻击虽语义相近，但在局部语言模式上引入可检测的“违和感”，这正是 RTD 任务所擅长捕捉的。
2. **预训练模型蕴含隐式安全能力**：无需额外训练，仅通过 repurpose RTD 判别器即可构建强大检测器，体现了 PLM 的多功能潜力。
3. **黑盒检测可以既高效又准确**：RTD-Guard 证明了在仅有 API 访问权限的情况下，也能实现媲美白盒方法的检测性能。
4. **检测应关注“扰动本身”而非“决策边界”**：传统方法试图模拟攻击者逆向工程，易受 label shift 影响；RTD-Guard 转向直接刻画扰动特征，更具稳定性。

---

### 方法的局限性
- **语言依赖性强**：当前 RTD 模型多为单语训练（如英文 ELECTRA），跨语言泛化能力有限。
- **多语言/跨语言输入支持不足**：缺乏匹配语言的 RTD 判别器时，检测效果可能下降。
- **对非替换类攻击敏感度未知**：目前聚焦于 word-substitution 攻击，对 paraphrasing 或插入类攻击的效果有待进一步验证。

---

### 未来工作方向
1. **扩展至 LLM 安全场景**：
   - 应用于 **prompt injection** 和 **jailbreak detection**，识别隐藏指令或越权意图。
   - 利用一致性信号检测角色劫持（role hijacking）等高级攻击。

2. **生成式系统中的幻觉检测（Hallucination Detection）**：
   - 结合 RTD 的上下文一致性信号与响应差异（如 counterfactual masking），识别事实错误或来源不可靠的内容。

3. **多语言与跨领域适配**：
   - 构建 multilingual RTD 判别器，提升跨语言对抗检测能力。
   - 探索 domain-adaptive 机制，使检测器适应特定垂直领域文本。

4. **与其他防御机制集成**：
   - 作为前置过滤模块，与 robust training 或 input reconstruction 方法结合，形成纵深防御体系。

---

> **总结一句话**：  
> RTD-Guard 成功将 **ELECTRA 的 RTD 预训练能力**转化为一种**高效、实用、免训练的黑盒对抗检测方案**，在性能、效率与部署可行性之间取得了卓越平衡，为现实世界 NLP 系统的安全防护提供了新的范式。

</details>

---

### 16. [SteerRM: Debiasing Reward Models via Sparse Autoencoders](https://arxiv.org/abs/2603.12795)

**Authors**: Mengyuan Sun, Zhuohao Yu, Weizheng Gu, Shikun Zhang, Wei Ye  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.12795v1  

#### Abstract
Reward models (RMs) are critical components of alignment pipelines, yet they exhibit biases toward superficial stylistic cues, preferring better-presented responses over semantically superior ones. Existing debiasing methods typically require retraining or architectural modifications, while direct a...

---

### 17. [Rethinking Multiple-Choice Questions for RLVR: Unlocking Potential via Distractor Design](https://arxiv.org/abs/2603.12826)

**Authors**: Xu Guo, Qiming Ge, Jian Tong, Kedi Chen, Jin Zhang, Xiaogui Yang, Xuan Gao, Haijun Lv, Zhihui Lu, Yicheng Zou, Qipeng Guo  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.12826v1  

#### Abstract
Reinforcement Learning with Verifiable Rewards (RLVR) significantly enhances the reasoning capabilities of Large Language Models. When applied to RLVR, Multiple-Choice Questions (MCQs) offer a scalable source of verifiable data but risk inducing reward hacking, where models shortcut reasoning via ra...

---

### 18. [3DTCR: A Physics-Based Generative Framework for Vortex-Following 3D Reconstruction to Improve Tropical Cyclone Intensity Forecasting](https://arxiv.org/abs/2603.13049)

**Authors**: Jun Liu, Xiaohui Zhong, Kai Zheng, Jiarui Li, Yifei Li, Tao Zhou, Wenxu Qian, Shun Dai, Ruian Tie, Yangyang Zhao, Hao Li  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 4.5  
**Type**: new  
**ArXiv ID**: 2603.13049v1  

#### Abstract
Tropical cyclone (TC) intensity forecasting remains challenging as current numerical and AI-based weather models fail to satisfactorily represent extreme TC structure and intensity. Although intensity time-series forecasting has achieved significant advances, it outputs intensity sequences rather th...

---

### 19. [Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation](https://arxiv.org/abs/2603.13099)

**Authors**: Wayner Barrios, SouYoung Jin  
**Category**: cs.AI  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.13099v1  

#### Abstract
We introduce **CRYSTAL** (*__C__lear __R__easoning via __Y__ielded __S__teps, __T__raceability and __L__ogic*), a diagnostic benchmark with 6,372 instances that evaluates multimodal reasoning through verifiable intermediate steps. We propose two complementary metrics: *Match F1*, which scores step-l...

---

### 20. [GONE: Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping](https://arxiv.org/abs/2603.12275)

**Authors**: Chahana Dahal, Ashutosh Balasubramaniam, Zuobin Xiong  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.12275v1  

#### Abstract
Unlearning knowledge is a pressing and challenging task in Large Language Models (LLMs) because of their unprecedented capability to memorize and digest training data at scale, raising more significant issues regarding safety, privacy, and intellectual property. However, existing works, including pa...

---

### 21. [EvolveCoder: Evolving Test Cases via Adversarial Verification for Code Reinforcement Learning](https://arxiv.org/abs/2603.12698)

**Authors**: Chi Ruan, Dongfu Jiang, Huaye Zeng, Ping Nie, Wenhu Chen  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.12698v1  

#### Abstract
Reinforcement learning with verifiable rewards (RLVR) is a promising approach for improving code generation in large language models, but its effectiveness is limited by weak and static verification signals in existing coding RL datasets. In this paper, we propose a solution-conditioned and adversar...

---

### 22. [Adaptive Vision-Language Model Routing for Computer Use Agents](https://arxiv.org/abs/2603.12823)

**Authors**: Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.12823v1  

#### Abstract
Computer Use Agents (CUAs) translate natural-language instructions into Graphical User Interface (GUI) actions such as clicks, keystrokes, and scrolls by relying on a Vision-Language Model (VLM) to interpret screenshots and predict grounded tool calls. However, grounding accuracy varies dramatically...

---

### 23. [Mending the Holes: Mitigating Reward Hacking in Reinforcement Learning for Multilingual Translation](https://arxiv.org/abs/2603.13045)

**Authors**: Yifeng Liu, Siqi Ouyang, Yatish Hosmane Revanasiddappa, Lei Li  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.13045v1  

#### Abstract
Large Language Models (LLMs) have demonstrated remarkable capability in machine translation on high-resource language pairs, yet their performance on low-resource translation still lags behind. Existing post-training methods rely heavily on high-quality parallel data, which are often scarce or unava...

---

### 24. [Neuron-Aware Data Selection In Instruction Tuning For Large Language Models](https://arxiv.org/abs/2603.13201)

**Authors**: Xin Chen, Junchao Wu, Shu Yang, Runzhe Zhan, Zeyu Wu, Min Yang, Shujian Huang, Lidia S. Chao, Derek F. Wong  
**Category**: cs.CL  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.13201v1  

#### Abstract
Instruction Tuning (IT) has been proven to be an effective approach to unlock the powerful capabilities of large language models (LLMs). Recent studies indicate that excessive IT data can degrade LLMs performance, while carefully selecting a small subset of high-quality IT data can significantly enh...

---

### 25. [A common parallel framework for LLP combinatorial problems](https://arxiv.org/abs/2603.13147)

**Authors**: David Ribeiro Alves, Vijay K. Garg  
**Category**: cs.DC  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.13147v1  

#### Abstract
Traditional lock-free parallel algorithms for combinatorial optimization problems, such as shortest paths, stable matching, and job scheduling require programmers to write problem-specific routines and synchronization code. We propose a general-purpose lock-free runtime, LLP-FW that can solve all co...

---

### 26. [Sinkhorn-Drifting Generative Models](https://arxiv.org/abs/2603.12366)

**Authors**: Ping He, Om Khangaonkar, Hamed Pirsiavash, Yikun Bai, Soheil Kolouri  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.12366v1  

#### Abstract
We establish a theoretical link between the recently proposed "drifting" generative dynamics and gradient flows induced by the Sinkhorn divergence. In a particle discretization, the drift field admits a cross-minus-self decomposition: an attractive term toward the target distribution and a repulsive...

---

### 27. [Learning Pore-scale Multiphase Flow from 4D Velocimetry](https://arxiv.org/abs/2603.12516)

**Authors**: Chunyang Wang, Linqi Zhu, Yuxuan Gu, Robert van der Merwe, Xin Ju, Catherine Spurin, Samuel Krevor, Rex Ying, Tobias Pfaff, Martin J. Blunt, Tom Bultreys, Gege Wen  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.12516v1  

#### Abstract
Multiphase flow in porous media underpins subsurface energy and environmental technologies, including geological CO$_2$ storage and underground hydrogen storage, yet pore-scale dynamics in realistic three-dimensional materials remain difficult to characterize and predict. Here we introduce a multimo...

---

### 28. [Curriculum Sampling: A Two-Phase Curriculum for Efficient Training of Flow Matching](https://arxiv.org/abs/2603.12517)

**Authors**: Pengwei Sun  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.12517v1  

#### Abstract
Timestep sampling $p(t)$ is a central design choice in Flow Matching models, yet common practice increasingly favors static middle-biased distributions (e.g., Logit-Normal). We show that this choice induces a speed--quality trade-off: middle-biased sampling accelerates early convergence but yields w...

---

### 29. [Lyapunov Stable Graph Neural Flow](https://arxiv.org/abs/2603.12557)

**Authors**: Haoyu Chu, Xiaotong Chen, Wei Zhou, Wenjun Cui, Kai Zhao, Shikui Wei, Qiyu Kang  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.12557v1  

#### Abstract
Graph Neural Networks (GNNs) are highly vulnerable to adversarial perturbations in both topology and features, making the learning of robust representations a critical challenge. In this work, we bridge GNNs with control theory to introduce a novel defense framework grounded in integer- and fraction...

---

### 30. [Sobolev--Ricci Curvature](https://arxiv.org/abs/2603.12652)

**Authors**: Kyoichi Iwasaki, Tam Le, Hideitsu Hino  
**Category**: cs.LG  
**Published**: 2026-03-16  
**Score**: 4.0  
**Type**: new  
**ArXiv ID**: 2603.12652v1  

#### Abstract
Ricci curvature is a fundamental concept in differential geometry for encoding local geometric structure, and its graph-based analogues have recently gained prominence as practical tools for reweighting, pruning, and reshaping network geometry. We propose Sobolev-Ricci Curvature (SRC), a graph Ricci...

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
