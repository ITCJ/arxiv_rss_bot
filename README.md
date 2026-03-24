# arXiv Papers Bot 🤖

This repository automatically fetches and displays relevant papers from arXiv based on configured criteria.

## RSS Vercel Deployment [![An example of deployed RSS Server using vercel](https://img.shields.io/badge/Deployed-Example-blue)](https://arxiv.tachicoma.top/)

You can click this to deploy yours 

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/maydomine/arxiv_rss_bot)
## 📊 Statistics

- **Last Updated**: 2026-03-24 06:48:19 UTC
- **Total Papers Found**: 30
- **Categories Monitored**: cs.AI, cs.CL, cs.DC, cs.LG

## 📚 Recent Papers

### 1. [SparseDVFS: Sparse-Aware DVFS for Energy-Efficient Edge Inference](https://arxiv.org/abs/2603.21908)

**Authors**: Ziyang Zhang, Zheshun Wu, Jie Liu, Luca Mottola  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 11.5  
**Type**: new  
**ArXiv ID**: 2603.21908v1  

#### Abstract
Deploying deep neural networks (DNNs) on power-sensitive edge devices presents a formidable challenge. While Dynamic Voltage and Frequency Scaling (DVFS) is widely employed for energy optimization, traditional model-level scaling is often too coarse to capture intra-inference variations, whereas fin...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：SparseDVFS: Sparse-Aware DVFS for Energy-Efficient Edge Inference

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
在边缘设备上部署深度神经网络（DNN）面临严峻的能效挑战。传统的 **DVFS**（Dynamic Voltage and Frequency Scaling）策略存在以下两大瓶颈：
- **模型级DVFS**（Model-level DVFS）粒度过粗，无法捕捉推理过程中不同算子间的计算强度差异；
- **算子级DVFS**（Operator-level DVFS）虽细粒度，但由于硬件频率切换延迟（通常为5–20ms），其开销远超轻量算子执行时间（如ReLU仅几毫秒），导致端到端延迟剧增，能效反而下降。

此外，现有方法缺乏对DNN内部特征（尤其是**sparsity**）的感知能力，难以实现精准节能。

---

### 🚀 提出的新方法与核心思想
本文提出 **SparseDVFS** ——一种**稀疏性感知的块级DVFS框架**，通过将算子聚合成“超级块”（super-blocks），在保留细粒度优化潜力的同时摊销切换开销。

#### 主要创新点：
1. **以算子稀疏性作为频率调节的关键信号**
   - 发现：**compute-bound** 算子（如Conv2d、Linear）适合高频运行；而 **memory-bound** 稀疏算子（如ReLU、LayerNorm）即使提高GPU频率也无性能提升，应降频节能。
   - 利用 sparsity 作为指导硬件频率配置的一等指标。

2. **离线建模器（Offline Modeler）**
   - 构建基于物理原理的白盒模型（white-box timeline analysis + thermal-aware power model）
   - 建立从算子稀疏性 → 最优CPU/GPU/EMC频率三元组的确定性映射
   - 轻量化（仅32字节），适用于边缘部署，避免黑盒学习方法的冷启动问题

3. **运行时图划分器（Runtime Graph Partitioner）**
   - 使用贪心合并算法动态构建 super-blocks
   - 引入**延迟摊销约束**（latency amortization constraint）：确保每个块的执行时间 $ T_{\text{block}} > N \times T_{\text{switch}} $
   - 平衡优化粒度与切换开销

4. **统一协同控制器（Unified Co-Governor）**
   - 实现 **FUSE**（Frequency Unified Scaling Engine）策略，同步控制CPU、GPU、内存频率
   - 消除独立控制器之间的**对抗效应**（antagonistic effect），例如CPU因利用率低被降频后导致GPU等待数据
   - 采用**前向预取机制**（look-ahead instruction queue）隐藏DVFS切换延迟

---

### 🔍 相比现有方法的优势
| 维度 | 传统方法 | SparseDVFS |
|------|--------|-----------|
| 粒度 | 过粗（model-level）或过细不可行（operator-level） | 自适应块级（block-level） |
| 特征感知 | 忽略sparsity等DNN特性 | 显式利用sparsity驱动DVFS决策 |
| 开销控制 | 忽视切换延迟影响 | 显式建模并摊销切换开销 |
| 控制协调性 | 多组件独立调控引发冲突 | 统一调度，消除对抗效应 |
| 可扩展性 | 黑盒模型需重新训练 | 白盒模型无需重训，跨模型/平台更易迁移 |

---

## 2. 核心实验方法和设置

### 📚 数据集与模型
- **数据集**：ImageNet-2012 验证集（约5万张图像）
- **测试模型**：
  - CNN类：ResNet-18（21 ops）、ResNet-101（105 ops）
  - Transformer类：ViT-B16（38 ops）、ViT-L16（74 ops）
- 所有模型均转换为ONNX格式并通过TensorRT优化

### 💻 实验平台
- **硬件**：NVIDIA Jetson Orin Nano（8GB RAM）
- **软件栈**：PyTorch 2.1, TensorRT, CUDA, JetPack 6.0 (Linux Kernel 5.15)
- **监控工具**：`jetson_stats` (`jtop.power()` API) 实时采集功耗

### ⚙️ 实验设置
- **CPU频率档位**：20级（115MHz ~ 1510MHz）
- **GPU频率档位**：5级（306 / 408 / 510 / 612 / 624.75 MHz）
- **EMC（内存）频率**：可调范围覆盖带宽变化

### 📊 评估指标
| 指标 | 定义 |
|------|------|
| **Energy Efficiency Gain** | 相对于Default DVFS的能耗降低百分比 |
| **Cost-Gain Ratio** | 每单位能耗节省所付出的延迟代价（越低越好） |
| **End-to-End Latency** | 单次推理总耗时 |
| **Power Consumption** | 平均/峰值功耗 |
| **Thermal Stability** | 温度波动与是否触发thermal throttling |
| **Switching Latency** | 频率切换带来的额外延迟 |

### 🆚 基线方法对比
1. **Default DVFS**：Linux `schedutil`（CPU） + NVIDIA `simple_ondemand`（GPU）
2. **nvpmodel (MAX-N)**：静态最高性能模式（锁频至最大）
3. **GearDVFS** [36]：基于DRL的学习型模型级DVFS
4. **Ascend-DVFS** [67]：遗传算法驱动的算子级精细DVFS

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

| 指标 | SparseDVFS 表现 | 对比优势 |
|------|------------------|----------|
| **平均能效增益** | **+78.17%**（vs Default DVFS） | 远超GearDVFS (+20.25%) 和 Ascend-DVFS (+11.33%) |
| **Cost-Gain Ratio** | **14%** | 显著优于GearDVFS (48%) 和 Ascend-DVFS (68%) |
| **端到端延迟** | 比MAX-N高约12.8%，但优于Default DVFS | 在节能前提下保持竞争力 |
| **功耗水平** | 动态调节于 **7W–10W** | MAX-N恒定15W，GearDVFS波动于10–12W |
| **切换延迟摊销效果** | 减少 **7.0×~8.5×** 总DVFS切换延迟 | 如ResNet-18减少7.0倍，ViT-B16达8.5倍 |

---

### 🔬 详细对比结果

#### （1）能效增益（Figure 13）
- **SparseDVFS** 在所有模型上均取得显著领先：
  - ResNet-18: ~75%
  - ViT-L16: 超过80%
- **nvpmodel** 因快速完成任务获得一定能效（+46.86%），但依赖高功耗运行
- **Ascend-DVFS** 因频繁切换带来巨大开销，能效提升微弱（仅+11.33%）

#### （2）成本收益分析（Figure 14）
- **SparseDVFS 成本增益比最低（14%）**，意味着每节省1焦耳能量所增加的延迟最小
- GearDVFS 和 Ascend-DVFS 为节能付出了不成比例的延迟代价

#### （3）消融实验（Ablation Study, Figure 15）
使用ViT-B16在DOTA-v1.0和VisDrone数据集上验证各模块作用：
- **仅GPU缩放**：存在严重CPU-GPU对抗效应，能效差
- **+CPU Lock**：缓解GPU饥饿，性能改善
- **完整FUSE策略**：实现最优协同控制，达到最佳能效

#### （4）热稳定性表现（Figure 16）
- **nvpmodel 和 Default DVFS** 很快触发热节流（thermal throttling），帧率剧烈抖动
- **SparseDVFS** 平均功耗更低，温升缓慢，未触发节流，适合被动散热设备

#### （5）聚合因子 $N$ 影响研究（Figures 17–19）
- $N$ 控制块大小与切换频率的权衡
- **过小（N=1）**：切换频繁，延迟主导
- **过大（N=10）**：错过稀疏相位降频机会，能耗上升
- **中等值（N=5）** 达成最佳平衡，体现U形能耗曲线

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **Operator Sparsity 是决定硬件频率配置的核心因素**
   - 高稀疏算子多为memory-bound，无需高频运行，降频即可大幅节能
   - 低稀疏算子为compute-bound，需高频保障性能

2. **块级DVFS是可行且高效的中间路径**
   - 既避免模型级DVFS的粗糙性，又规避算子级DVFS的切换灾难
   - 通过 super-block 设计成功摊销 $T_{\text{switch}}$

3. **统一控制至关重要**
   - 独立的CPU/GPU控制器会导致pipeline stall
   - FUSE策略有效消除对抗效应，提升系统整体效率

4. **前向预取机制可有效隐藏切换延迟**
   - 利用流水线重叠计算与频率切换，几乎完全消除切换惩罚

5. **白盒建模优于黑盒学习**
   - 轻量、可解释、无需再训练，更适合边缘场景

---

### ⚠️ 局限性
1. **Sparsity建模仍较简化**
   - 当前仅使用稀疏比率（scalar ratio），未区分**结构化稀疏**（structured）与**非结构化稀疏**（unstructured）
   - 后者可能引起随机访存，降低带宽效率

2. **缺乏实时内存访问反馈**
   - 内存row activation延迟等因素未纳入动态调整逻辑

3. **设备依赖性强**
   - 离线建模需针对特定SoC进行profiling，跨平台迁移需重新校准

---

### 🔮 未来工作方向
1. **精细化稀疏模式识别**
   - 区分结构化/非结构化稀疏，动态调整EMC频率应对访问效率差异

2. **引入内存控制器性能计数器**
   - 实时监测bus contention、row conflicts等，增强runtime partitioner决策能力

3. **支持跨平台迁移**
   - 探索**Transfer Learning**技术，将在一个设备（如Jetson Orin Nano）上学得的V/F映射迁移到其他设备（如Google Edge TPU）

4. **结合Early Exit等动态推理机制**
   - 与E4、PowerInfer等技术融合，进一步提升长序列LLM推理能效

---

## ✅ 总结
**SparseDVFS** 是首个将**算子稀疏性**作为DVFS核心调控信号的工作，提出了一个**细粒度、可扩展、低开销**的能量优化框架。其实验结果表明，在真实边缘平台上实现了**平均78.17%的能效增益**，同时维持较低的成本增益比（14%），显著优于当前最先进的DVFS方案。该工作为边缘AI系统的软硬协同设计提供了重要范式。

</details>

---

### 2. [DRTriton: Large-Scale Synthetic Data Reinforcement Learning for Triton Kernel Generation](https://arxiv.org/abs/2603.21465)

**Authors**: Siqi Guo, Ming Lin, Tianbao Yang  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 11.0  
**Type**: new  
**ArXiv ID**: 2603.21465v1  

#### Abstract
Developing efficient CUDA kernels is a fundamental yet challenging task in the generative AI industry. Recent researches leverage Large Language Models (LLMs) to automatically convert PyTorch reference implementations to CUDA kernels, significantly reducing the engineering efforts. State-of-the-art ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# DRTriton: Large-Scale Synthetic Data Reinforcement Learning for Triton Kernel Generation —— 核心总结

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

在生成式AI产业中，开发高效的CUDA内核是降低推理成本的关键，但对人类专家而言仍极具挑战性（如FlashAttention耗时数年）。尽管已有研究尝试利用 **Large Language Models (LLMs)** 将PyTorch代码自动转换为CUDA/Triton内核，但当前最先进的模型（如GPT-5.2、Claude-Sonnet-4.5）在此任务上表现不佳，尤其是在**正确性和运行效率**方面。

此外，现有基于真实代码库的方法受限于：
- **训练数据规模有限**（仅约1万级样本）
- **复杂度不可控、质量不一致**
- **稀疏奖励下难以有效学习**

### **提出了什么新方法或新思路**

本文提出 **DRTriton**，一个完全基于**大规模合成数据**进行强化学习的可扩展框架，用于训练LLM将PyTorch程序高效转化为优化的Triton内核。其三大核心组件构成主要创新：

#### ✅ **(i) CSP-DAG 合成算法**
- 将PyTorch程序生成建模为**有向无环图上的约束满足问题 (Constraint Satisfaction Problem on DAGs)**。
- 利用CP-SAT求解器确保生成的程序语法合法且张量形状兼容。
- 支持**受控难度采样**，实现操作符空间中的**全覆盖与均匀分布**。

#### ✅ **(ii) 解耦奖励的课程强化学习 (Curriculum RL with Decoupled Rewards)**
- 引入 **DRPO (Decoupled Reward Policy Optimization)** 框架，将奖励分为两个独立部分：
  - `Correctness Reward`：保证功能等价
  - `Speed Reward`：鼓励生成更快的内核
- 在早期阶段缓解稀疏奖励问题，提升梯度信号质量。
- 采用三阶段课程学习（Level 1 → Level 2 → Level 5），逐步增加任务复杂度。

#### ✅ **(iii) 推理时搜索策略 (Test-time Search)**
- 针对复杂程序无法单个内核容纳的问题，设计系统性融合策略搜索机制。
- 分解程序为多个子片段 → 生成并验证每个片段的Triton内核 → 构造混合执行方案 → 选择最快者。
- 显著提升长程序的成功率与性能。

---

### **相比现有方法的优势**

| 维度 | DRTriton | 现有方法（如AutoTriton、KernelLLM） |
|------|----------|-------------------------------|
| 数据来源 | 完全合成，无限扩展 | 依赖有限的真实仓库数据（~10k） |
| 学习方式 | Curriculum RL + Decoupled Reward | SFT 或普通RL（GRPO） |
| 性能目标 | 同时优化**正确性 + 执行速度** | 多聚焦于正确性 |
| 泛化能力 | 能泛化到现实世界架构（MobileNet/VGG等） | 对分布外代码泛化差 |
| 可控性 | 困难等级可控，避免早期过拟合 | 样本顺序随机，易陷入局部最优 |

---

## 2. 核心实验方法和设置

### **使用的数据集**

#### 📌 **合成基准 (Synthetic Benchmark)**
- 包含406个未见的PyTorch程序，按难度分层：
  - Level 1: 单算子（106个）
  - Level 2: 双算子序列（100个）
  - Level 5: 五算子组合（100个）
  - Level 20: 二十算子复杂程序（100个）
- 操作符来自53种常用PyTorch函数（见Appendix A）

#### 📌 **现实世界基准 (KernelBench)**
- 来自Ouyang et al. (2025) 的标准测试集，共250个任务：
  - Level 1: 单内核操作（卷积、矩阵乘、激活函数）
  - Level 2: 多算子融合模式（如conv+bias+ReLU）
  - Level 3: 完整模型架构（MobileNet、VGG、MiniGPT）

> ⚠️ 注意：DRTriton训练数据**全部为合成数据**，未见过任何KernelBench样本。

---

### **实验设置和评估指标**

#### ✅ **基础模型**
- 使用 `Qwen-2.5-Coder-7B-Instruct` 作为初始化模型。

#### ✅ **训练流程**
1. **Supervised Fine-Tuning (SFT)**  
   - 数据：2,026个单算子PyTorch-Triton对（由DeepSeek-R1/GPT-5.2生成并通过验证）
   - Epochs: 10，LR: 2e-6，Batch Size: 64

2. **Curriculum Reinforcement Learning (RL)**  
   - 方法：DRPO
   - 三阶段训练：
     - Stage 1: Level 1（20k样本）
     - Stage 2: Level 2（60k样本）
     - Stage 3: Level 5（20k样本）
   - 每阶段1 epoch，LR: 1e-6，Rollouts: 8 per prompt

#### ✅ **评估指标**
| 指标 | 定义 |
|------|------|
| **Acc** | 正确通过验证的比例（语法 + 忠实性 + 数值一致性） |
| **Faster1** | 正确内核中，执行时间比原PyTorch快 **>1倍** 的比例 |
| **Avg. Speedup** | 所有正确内核相对于PyTorch的几何平均加速比 |

#### ✅ **验证流程**
1. **Syntax Validation**：检查是否包含 `@triton.jit` 并能编译成功
2. **Faithfulness Validation**：通过monkey-patch检测是否真正使用Triton计算（而非调用PyTorch）
3. **Correctness Validation**：在5组随机输入上比较输出是否一致（容忍浮点误差）

---

### **基线方法对比**

| 基线模型 | 类型 |
|--------|------|
| GPT-5.2 | 商业闭源LLM |
| Claude-Sonnet-4.5 | 商业闭源LLM |
| DeepSeek-R1 | 开源LLM |
| Qwen-3-Coder-480B | 开源LLM |
| AutoTriton | 专用RL训练模型（Li et al., 2025d） |

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### 🔹 在 **Synthetic Benchmark** 上的结果（Table 1）

| Model | Level 1 Acc | Level 2 Acc | Level 5 Acc | Level 20 Acc | Avg. Speedup |
|-------|-------------|-------------|-------------|--------------|---------------|
| GPT-5.2 | 53.8% | 43.0% | 5.0% | 0% | 1.34x |
| Claude-Sonnet-4.5 | 67.9% | 49.0% | 7.0% | 0% | 0.68x |
| **DRTriton** | **86.8%** | **75.0%** | **15.0%** | **0%** | **1.20x** |
| **+ test-time search** | 86.8% | **96.0%** | **99.0%** | **99.0%** | **1.57x** |

> 💡 发现：**test-time search极大提升了复杂程序的表现**，甚至在Level 20达到99%成功率。

#### 🔹 在 **KernelBench** 上的结果（Table 2）

| Model | Level 1 Acc | Level 2 Acc | Level 3 Acc | Level 2 Faster1 (vs TE) | Level 3 Faster1 (vs TC) |
|-------|-------------|-------------|-------------|-------------------------|--------------------------|
| GPT-5.2 | 40% | 32% | 30% | 23% | 10% |
| Claude-Sonnet-4.5 | 46% | 36% | 36% | 19% | 12% |
| **DRTriton (w/ test-time search)** | **69%** | **96%** | **76%** | **92%** | **34%** |

> ✅ 特别亮点：
> - 在Level 2任务中，**92%的生成内核比原始PyTorch更快**
> - 在Level 3完整模型中，仍有**54%优于PyTorch Eager，34%优于torch.compile**

---

### **消融实验结果**

#### 🔍 **Ablation on RL Algorithm (DRPO vs GRPO)**（Figure 5）
- 在相同SFT起点和Stage 1数据下：
  - DRPO在所有指标上均显著优于GRPO
  - 例如，在Level 1上，DRPO的Faster1达41.0%，而GRPO仅为31.1%
- 表明**解耦奖励机制更利于引导模型关注性能优化**

#### 🔍 **Ablation on Speed Reward Function**（Table 4）
- 比较不同速度奖励函数形式：
  - Power形式（α=0.25~1.0）效果较差
  - **Logarithmic形式 (`rs(o)=log(t_torch / t_triton)`)** 效果最佳
  - 达到 Acc=42.3%, Faster1=18.6%

> 结论：**对数形式的速度奖励提供了更稳定的梯度信号**

---

## 4. 关键结论和发现

### **主要发现**

1. ✅ **仅用合成数据即可超越真实数据训练的模型**
   - DRTriton虽未接触任何真实CUDA内核，却能在KernelBench上全面超越GPT-5.2、Claude等商业模型及专用模型AutoTriton。

2. ✅ **课程学习 + 解耦奖励显著稳定训练过程**
   - DRPO有效应对早期稀疏奖励问题，使模型从简单任务开始渐进掌握复杂映射。

3. ✅ **推理时搜索大幅提升复杂程序性能**
   - 对于难以单内核处理的程序，test-time search通过模块化融合找到最优组合，反而是越复杂的程序增益越大。

4. ✅ **性能涌现现象**
   - 模型并未被显式教授如何写Triton代码，但在RLVR（Reinforcement Learning with Verifiable Rewards）框架下，“自主学会”了高性能内核编写技能。

---

### **方法的局限性**

1. ❗ **依赖高质量的自动重写工具**
   - 对于高度封装或动态控制流的PyTorch代码（如条件分支、循环），目前的`functional rewrite`工具可能失效。

2. ❗ **Triton语言表达能力限制**
   - Triton本身不支持某些高级特性（如动态共享内存分配），导致部分复杂优化无法实现。

3. ❗ **硬件特异性未充分建模**
   - 当前方法未针对特定GPU架构（如H100 vs A100）定制优化策略，未来可引入硬件感知反馈。

---

### **未来工作方向**

1. ➕ **引入多智能体协作机制**
   - 如结合planning agent分析瓶颈，verification agent指导修复，进一步提升鲁棒性。

2. ➕ **构建硬件闭环反馈系统**
   - 将实际profiling延迟作为reward信号，形成“生成 → 编译 → 测速 → 更新”的闭环优化。

3. ➕ **扩展至其他DSL**
   - 将该范式推广至CUDA C++、HIP、SYCL等底层语言生成。

4. ➕ **探索zero-shot迁移能力**
   - 研究合成数据多样性与现实任务之间的泛化边界。

---

> 🎯 **总结一句话**：  
> **DRTriton证明了“纯合成数据 + 强化学习 + 推理时搜索”可以构建出超越商业LLM和传统编译器的自动化GPU内核生成系统，为AI驱动的系统优化开辟了新路径。**

</details>

---

### 3. [ARYA: A Physics-Constrained Composable & Deterministic World Model Architecture](https://arxiv.org/abs/2603.21340)

**Authors**: Seth Dobrin, Lukasz Chmiel  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 10.5  
**Type**: new  
**ArXiv ID**: 2603.21340v1  

#### Abstract
This paper presents ARYA, a composable, physics-constrained, deterministic world model architecture built on five foundational principles: nano models, composability, causal reasoning, determinism, and architectural AI safety. We demonstrate that ARYA satisfies all canonical world model requirements...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**ARYA: A Physics-Constrained Composable & Deterministic World Model Architecture**

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前主流的 **World Model** 架构（如基于 Transformer 或 RNN 的单体模型）存在以下根本性挑战：
- **计算效率低**：随着任务复杂度增加，模型参数呈超线性增长。
- **缺乏物理一致性**：依赖统计学习，容易违反物理规律（如能量守恒）。
- **黑箱决策**：难以审计、解释，尤其在高风险领域（医疗、航天）不可接受。
- **冷启动问题**：需要大量历史数据才能部署，无法零样本迁移。
- **安全失控风险**：自主系统可能绕过安全机制进行自我改进。

ARYA 针对上述问题，提出了一种全新的世界模型架构范式。

---

### 🚀 提出的新方法与核心创新

#### （1）**Nano Model 架构（Composable Intelligence Layer）**
- 将传统“大而全”的单体模型替换为 **数以万计的小型专用 nano model**（每个 10K–100K 参数）。
- 每个 nano model 负责一个特定功能（如热传导系数、振动共振、光学透射等），具备高保真度。
- 所有 nano model 通过 **AARA（Autonomous Research Agent）** 动态编排组合，形成完整模拟。

> 🔍 类比：如同乐高积木，而非整块雕塑。

#### （2）**物理约束优先（Physics-First Design）**
- 物理定律不是从数据中“学习”，而是作为 **硬性过滤器（Hard Filters）** 写入系统。
- 所有预测必须经过 **First-Principles Solvers**（如欧拉-伯努利梁方程、傅里叶定律）验证。
- 若违反物理规则，输出将被拒绝或替换。

#### （3）**可组合性与零样本部署（Zero-Shot Deployment）**
- 新客户无需提供历史训练数据。
- 只需输入技术规格（CAD 模型、材料属性、操作参数），即可立即生成可靠预测。
- 成功应用于 **ARYA-Fold** 蛋白质折叠任务，在无训练数据下达到 AlphaFold2 级别精度。

#### （4）**确定性与透明性（Deterministic & Glassbox/CDAI）**
- 支持完全可追溯的推理路径（W3C PROV-DM 血缘追踪）。
- 安全关键路径强制使用 **Rules Engine / Physics Solver**，禁用黑盒神经网络。
- 实现 **Constrained Deterministic AI™ (CDAI)** 合规。

#### （5）**不可解雇的安全内核（Unfireable Safety Kernel）**
- 一个独立运行的服务，所有状态变更都需其加密签名授权。
- **无法被关闭、绕过或篡改**，即使来自系统的自我改进引擎。
- 提供形式化验证（Z3）、沙箱执行、回归测试五阶段 **Safety Gauntlet Pipeline**。

#### （6）**自改进能力（Recursive Self-Improvement, RSI）**
- 系统能自动提出、评估、验证并部署自身架构优化。
- 在 **A6 级别**实现开放式的自我演化，但仍受 Safety Kernel 控制。
- 区别于 RL：修改的是代码结构本身，而非策略参数。

---

### ⚖️ 相比现有方法的优势

| 维度 | 传统 World Models (DreamerV3, JEPA) | ARYA |
|------|-------------------------------|------|
| 架构 | 单体神经网络 | 系统-of-systems of nano models |
| 物理约束 | 软惩罚（loss penalty） | 硬过滤（architectural filter） |
| 透明性 | 黑箱 | Glassbox/CDAI 可审计 |
| 推理模式 | 统计近似 | 数学确定性 |
| 扩展方式 | 超线性重训 | 线性添加 nano model |
| 激活方式 | 密集激活 | 稀疏激活（仅调用相关模型） |
| 自我改进 | 不支持 | 支持 + 形式化验证 |
| 安全机制 | 可配置策略 | 不可解雇内核（architecturally immutable） |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集与基准测试

论文在 **9 个外部公开基准** 和 **7 个实际行业部署节点** 上进行了验证：

#### 外部基准（9项）
| Benchmark | 领域 | 主要评估目标 |
|---------|------|-------------|
| **CLadder** | 因果推理 | 结构因果建模能力 |
| **PhysReason** | 物理推理 | 对物理规律的理解与应用 |
| **FrontierScience** | 博士级科学问题 | 科学深度与抽象推理 |
| **WoW (World of Workflows)** | 企业流程 | 复杂系统中的级联效应预测 |
| **WorldArena** | 具身规划 | 动作序列的功能实用性 |
| **BigCodeBench** | 代码生成 | 结构化编程任务 |
| **SWE-bench** | 软件工程 | 开放式代码修复 |
| **CausalBench** | 因果发现 | 从数据推断因果图 |
| **AI Safety Index** | 安全性 | 系统抗攻击与控制保持能力 |

#### 行业部署节点（7个）
- Aerospace（NASA EXCITE 数字孪生）
- Pharma Manufacturing（制药生产过程建模）
- Oil & Gas（上游产量优化）
- Smart Cities（城市基础设施联动分析）
- Biotech（精准医学与药物发现）
- Defense（导弹制导系统）
- Medical Devices（数字孪生设备）

> 💡 所有部署均已在生产环境中运行，共涉及 **111,572 个 nano models**。

---

### 🧪 实验设置与评估指标

#### 评估维度
| 指标类别 | 具体指标 |
|--------|--------|
| **性能** | Accuracy, Pass@1, nDTW, GDT-TS, TM-score |
| **效率** | Inference Latency (P50/P99), Training Time, Model Size |
| **安全性** | Bypass Attempt Success Rate, Z3 Verification Latency |
| **可扩展性** | Sparse Activation Ratio, Memory Reduction |
| **通用性** | Cross-domain Transfer, Zero-shot Performance |

#### 基线对比模型
- **GPT-5.2**
- **Claude Opus 4.6**
- **V-JEPA 2**
- **AlphaFold2**（用于蛋白质折叠）
- **DreamerV3**（作为世界模型模板）

> 所有 ARYA 结果均为 **zero-shot prompting**，不使用任何提示工程或微调。

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据汇总

| Benchmark | ARYA 性能 | 最佳基线 | 排名 |
|----------|-----------|----------|-----|
| **CLadder** (因果推理) | **99.89%** | GPT-4: 76.4% | #1 |
| **PhysReason** (物理推理) | **73.3** | DeepSeek-R1: 56.8 | #1 |
| **FrontierScience** (博士级科学) | **37.5%** | GPT-5.2: 25.8% | #1 |
| **WoW** (企业流程) | **30.5% 完美匹配率** | Claude Sonnet: 28.1% | #1 |
| **WorldArena** (具身规划) | **9.006 nDTW** | Claude Opus: 82.2 | #2 |
| **BigCodeBench** (代码生成) | **80.5% Pass@1** | o3-mini: 61.4% | Top 3 |
| **AI Safety Index** | **100.0%** | Claude Sonnet: 25% | #1 |

> ✅ ARYA 在 **6/9** 个基准上取得 SOTA，且全部使用 **zero neural network parameters**（纯符号/物理求解器）。

---

### 🔍 消融实验与系统级指标（System Metrics）

| 指标 | 目标值 | 实际表现 | 说明 |
|------|-------|---------|------|
| **Inference Latency (P50)** | <200ms | **0.0002ms** | 子微秒级响应 |
| **Inference Latency (P99)** | <200ms | **0.0007ms** | 极端延迟极低 |
| **Accuracy (mean)** | >95% | **99.34%** | 跨领域平均准确率 |
| **Training Time** | <20s | **1.2s ~ 18.9s** | 所有 nano model 均达标 |
| **Sparse Activation** | N/A | **12.5% 激活率** | 内存节省 **94.3%** |
| **Safety Kernel Bypass Attempts** | 0 success | **0/40 成功** | “不可解雇”得到实证 |
| **Z3 Formal Verification Latency** | 100–500ms | **P50: 2.11ms** | 远快于设计目标 |

> 🧩 **稀疏激活**使内存占用从理论上的 **2,475 MB**（密集）降至 **25 MB**（稀疏）。

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **世界模型不必是单体神经网络**  
   - 通过 **composable nano models + Context Network** 同样可以满足所有七项 canonical world model requirements。
   - 并且在 **物理一致性、可解释性、安全性** 上显著优于传统方法。

2. **物理先验知识比海量数据更高效**  
   - 在多个领域（航空航天、制药、生物）实现 **zero-shot deployment**，无需客户历史数据。
   - 如 ARYA-Fold 在无训练数据情况下，仅用物理规则即达到 AlphaFold2 水平。

3. **安全可以是架构级保障，而非政策层补丁**  
   - “Unfireable Safety Kernel” 是首个真正意义上 **无法被系统自身绕过的安全机制**。
   - 通过 **Ed25519 加密签名 + 独立进程边界 + Z3 形式验证** 实现。

4. **自主智能与人类控制并非对立**  
   - 即使在 A6 级别的开放式自我改进中，人类仍可通过 Safety Kernel 保留最终否决权。
   - 自主性增强的同时，**可控性和可审计性同步提升**。

5. **企业场景存在“动态盲区”（Dynamics Blindness）**  
   - LLMs 在 WoW 基准中频繁引发“静默违规”——完成任务却破坏下游流程。
   - ARYA 的 **Simulation Unit + Context Network** 显式建模级联影响，有效避免此类问题。

---

### ⚠️ 局限性

1. **不适合开放生成任务**
   - 在 **SWE-bench** 上得分为 0%，表明 ARYA 不适用于无约束的软件开发代理任务。
   - 优势集中在 **结构化、物理约束强的任务**。

2. **依赖高质量领域建模**
   - 虽然物理模块可复用，但初始 domain node 的构建仍需专家定义 solver 和 constraint。
   - 自动化程度低于端到端训练的 LLM。

3. **orchestration 复杂性较高**
   - 管理数十万个 nano model 的依赖关系、版本控制、冲突解决带来额外工程负担。
   - 但作者认为这是“可管理的复杂性”，已被脑启发的层级架构缓解。

---

### 🔮 未来工作方向

1. **扩展 domain node 覆盖范围**  
   - 当前已覆盖 7 个领域，计划拓展至量子计算、气候建模、金融系统。

2. **增强跨域迁移机制**  
   - 提升物理模式（如热力学、流体力学）在不同行业间的自动识别与复用能力。

3. **加速形式化验证性能**  
   - 进一步降低 Z3 验证延迟，支持更大规模的实时验证。

4. **发展量子就绪架构（Quantum-Ready Architecture）**  
   - 探索将量子算法集成进 nano model 生态的可能性。

5. **向更高自治等级演进**  
   - 在确保安全的前提下，推动 A6 级 RSI 更广泛的应用。

---

## 📌 总结

**ARYA** 不是一个新的 LLM 或世界模型变体，而是一种 **重新思考 AI 构建方式的基础架构革命**。它证明了：

> ✅ **高性能 ≠ 黑箱**  
> ✅ **自主性 ≠ 失控**  
> ✅ **通用智能 ≠ 单一模型**

通过 **nano model composability + physics-first design + unfireable safety kernel**，ARYA 在 **因果推理、物理理解、企业流程建模、AI 安全** 等关键领域实现了 SOTA，并已在 **航天、医药、能源、国防** 等高风险行业落地应用。

这标志着一种新型 AI 范式的崛起：**Deterministic, Auditable, Governed, and Safe-by-Architecture**。

</details>

---

### 4. [AI-Driven Multi-Agent Simulation of Stratified Polyamory Systems: A Computational Framework for Optimizing Social Reproductive Efficiency](https://arxiv.org/abs/2603.20678)

**Authors**: Yicai Xing  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.20678v1  

#### Abstract
Contemporary societies face a severe crisis of demographic reproduction. Global fertility rates continue to decline precipitously, with East Asian nations exhibiting the most dramatic trends -- China's total fertility rate (TFR) fell to approximately 1.0 in 2023, while South Korea's dropped below 0....

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：AI-Driven Multi-Agent Simulation of Stratified Polyamory Systems

## 1. 论文的主要贡献和创新点

### 解决了什么问题
该论文针对当代社会面临的**双重结构性危机**：
- **生育率持续下降**（如韩国TFR < 0.72，中国约1.0）
- **婚姻制度功能失灵**：高学历女性因“母亲惩罚”（motherhood penalty）理性拒绝婚育，底层男性面临系统性“性资源剥夺”（sexlessness）与社会边缘化。

传统解决方案（如生育补贴、道德劝说）效果有限，作者认为根本原因在于**monogamy**在当前生产力条件下的资源配置低效。

---

### 提出的新方法与新思路
提出一个名为 **Stratified Polyamory System (SPS)** 的计算社会科学框架，其核心是：
- **制度设计**：允许个体拥有1个法定配偶（spouse）+ 最多2个法律承认的伴侣（companions），后者享有性关系、生育权和有限育儿权，但无财产共有或继承权。
- **分层模型**：引入A/B/C三类基于复合吸引力（mate value, economic resources, fertility等）的异质性agent，模拟现实中的择偶市场分层。
- **社会工程机制**：
  - **Grace Decree Effect**：通过增加高资源个体（A-tier）的合法后代数量，实现财富代际分散，对抗`r > g`导致的贫富固化。
  - **Socialized Child-Rearing**：公共育儿体系消除母亲惩罚，保障性别平等。
- **AI驱动仿真平台**：整合多种前沿技术构建可验证政策影响的计算框架。

---

### 相比现有方法的优势
| 维度 | 现有研究（CNM / 社会政策） | 本论文（SPS + 计算框架） |
|------|--------------------------|----------------------------|
| 分析层次 | 微观个体关系质量 | 宏观人口级社会效率优化 |
| 政策形式 | 非正式实践或经济激励 | 正式法律权利区分的制度设计 |
| 评估方式 | 观察性研究、问卷调查 | 可控、可重复的ABM/MARL仿真 |
| 技术整合 | 单一方法为主 | 多模态融合（ABM + MARL + LLM + GNN） |
| 公平性保障 | 缺乏系统建模 | 显式算法公平约束（envy-free, tier parity） |

> ✅ **创新点总结**：首次将**agent-based modeling**, **multi-agent reinforcement learning**, 和 **LLM-empowered generative agents** 用于模拟复杂非传统婚恋系统的社会演化效应，为制度变革提供“数字孪生”测试平台。

---

## 2. 核心实验方法和设置

### 数据集与参数来源
- **未使用真实世界行为数据集**，而是基于以下来源进行参数化建模：
  - 美国综合社会调查（GSS）关于性活跃率的数据
  - 进化心理学实证（如Schmitt, 2005的48国研究）
  - 匹配市场理论（Becker, 1973；Greenwood et al., 2014）
  - 生物人类学证据（Murdock’s Ethnographic Atlas）
  - 代理属性初始化参考美国人口普查分布

---

### 实验设置
| 参数 | 设置值 |
|------|--------|
| 模拟人数 | 10,000 agents（50% male/female） |
| Tier分布 | A:B:C = 15:60:25 |
| 时间跨度 | 100年（每步≈1年） |
| Partner限制 | SPS组：1 spouse + ≤2 companions；Monogamy组：仅1 spouse |
| Agent维度 | `(v, r, f, s, g, l)`：mate value, resources, fertility, social capital, gender, life stage |
| 学习算法 | **PPO with CTDE**（Centralized Training, Decentralized Execution） |
| LLM代理比例 | 1%（100个LLM驱动的“生成型代理”） |
| 折扣因子 γ | 0.95 |

---

### 评估指标
- **Aggregate Welfare**: 加权总效用 `R = Σ[αW(t) + βF(t) + δS(t)]`
- **Fertility Rate (TFR)**: 总和生育率变化
- **Wealth Inequality**: 跨代Gini系数变化
- **Network Properties**:
  - 小世界特征（small-worldness）
  - 跨层级连接（cross-tier bridges）
  - 社区结构（community detection）
- **Algorithmic Fairness Metrics**:
  - Individual Rationality（无人比monogamy更差）
  - Envy-freeness（宽松版）
  - Tier Parity（各阶层福利差距缩小）
  - Gender Symmetry

---

### 基线方法对比
| 方法 | 描述 |
|------|------|
| **Strict Monogamy Baseline** | 一对一婚姻制，作为对照组 |
| **Unregulated CNM Practice** | 无法律框架支持的consensual non-monogamy（隐含比较） |
| **Traditional Pro-Natalist Policies** | 如北欧福利、匈牙利贷款减免（文献引用对比） |

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自初步原型模拟 N=1,000, 50年）
| 指标 | Monogamy | SPS | 提升幅度 |
|------|---------|-----|--------|
| 平均个体Welfare | 基准 | ↑18–25% | +21.5% avg |
| C-tier男性Welfare | 极低 | ↑140% | 最大受益群体 |
| TFR（稳定值） | 1.1–1.3 | 1.7–1.9 | ↑~50% |
| 跨三代财富Gini | 下降缓慢 | ↓8–12% | 加速财富分散 |
| 网络小世界特性 | 弱 | 显著呈现 | 高聚类+稀疏跨层边 |

---

### 与基线方法的对比结果
- **相比Monogamy**：
  - 所有tier预期效用均提升（支持Pareto Improvement假说）
  - C-tier男性从近乎零亲密关系跃升至部分满足
  - B-tier获得向上/向下兼容弹性
  - A-tier维持主配偶同时扩展伴侣网络
- **相比纯CNM实践**：
  - SPS提供**法律稳定性**（companion rights界定清晰）
  - 减少地下性交易风险
  - 更强的儿童权益保护机制

---

### 消融实验结果（文中提及但未详列）
- **移除Socialized Child-Rearing** → 母亲惩罚依然存在 → TFR提升不显著
- **取消Tier流动性建模**（静态tier）→ 系统僵化，无法反映博士生阶段性贬值现象
- **关闭Grace Decree机制** → 财富集中趋势未逆转 → 社会稳定性改善有限

> 🔁 结论：三大支柱（SPS制度 + 公共育儿 + 继承改革）缺一不可。

---

## 4. 关键结论和发现

### 主要发现
1. **SPS可实现Pareto改进**：所有社会阶层在预期效用上均优于strict monogamy，尤其C-tier男性获益最大。
2. **能有效恢复TFR至近更替水平**（1.7–1.9），主要驱动力为：
   - 消除母亲惩罚 → 提升高知女性生育意愿
   - 扩展C-tier生殖机会 → 增加总体生育基数
3. **促进财富代际分散**（Grace Decree Effect）：通过增加高资源者的合法子女人数，自然稀释资本集中，缓解`r > g`问题。
4. **增强社会稳定**：降低“bare branches”（过剩未婚男性）引发暴力犯罪的风险（Hudson & den Boer, 2004）。
5. **心理可行性**：基于CNM实证研究（Conley et al., 2013; Moors et al., 2017），嫉妒可通过文化建构管理，“compersion”等情绪可被培养。
6. **历史与跨文化支持**：全球83.5%的社会曾允许某种形式的polygyny（Murdock, 1967）；Mosuo走婚、藏族兄弟共妻等表明多元婚制可行。

---

### 方法的局限性
1. **A/B/C tier为简化分析工具**：现实中吸引力是多维连续谱，非刚性分类。
2. **福利预测依赖假设**：图8的效用增益为理论推导，需更多经验校准。
3. **政治与文化接受度不确定**：实施路径依赖强，需长期渐进改革（见Phase I–IV roadmap）。
4. **潜在负外部性未充分建模**：
   - 多方家庭中的儿童心理发展（尽管Kibbutz和Nordic研究支持正常发展）
   - 财产纠纷复杂性上升
   - intergroup conflict风险（Koos & Neupert-Wentz, 2020）
5. **计算挑战**：
   - MARL中的non-stationarity问题
   - LLM代理的行为真实性与prompt敏感性
   - sim-to-real gap（仿真到现实迁移）

---

### 未来工作方向
1. **全规模仿真实验**：运行完整N=10,000、100年的ABM+MARL联合训练。
2. **LLM代理深度集成**：扩大LLM-driven agent比例，捕捉文化语境下的决策多样性。
3. **Policy Optimization via Evolutionary Algorithm**：使用遗传算法自动搜索最优SPS参数组合（如伴侣上限、tier阈值、补贴水平）。
4. **跨文化情境迁移测试**：在不同初始财富/性别比/文化规范下验证鲁棒性。
5. **结合真实数据校准模型**：接入在线约会平台行为日志（如OkCupid）、生育登记数据等。
6. **伦理治理框架设计**：制定知情同意协议、退出机制、反剥削条款等配套制度。

---

> 📌 **最终结论**：  
> 当前monogamy作为私有制的文化伴生物，已难以适应知识经济时代的生产关系。SPS并非终极答案，但它代表了一种**以计算社会科学为基础、由AI辅助设计的新型制度探索方向**——更加尊重人性复杂性、更高效配置情感与生殖资源、更具包容性的未来社会形态。  
> “人类终将视monogamy如同农奴制一般，只是特定历史阶段的产物，而非自然法则。”

</details>

---

### 5. [ConsRoute:Consistency-Aware Adaptive Query Routing for Cloud-Edge-Device Large Language Models](https://arxiv.org/abs/2603.21237)

**Authors**: Haoyu Qiao, Hao Zhang, Shanwen Mao, Siyao Cheng, Jie Liu  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.21237v1  

#### Abstract
Large language models (LLMs) deliver impressive capabilities but incur substantial inference latency and cost, which hinders their deployment in latency-sensitive and resource-constrained scenarios. Cloud-edge-device collaborative inference has emerged as a promising paradigm by dynamically routing ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# **论文核心结论与实验结果总结**

## **1. 论文的主要贡献和创新点**

### **解决的问题**
大型语言模型（LLM）在云-边-端协同推理场景中面临显著的**推理延迟高、计算成本大**的问题。传统方法在资源受限的设备上部署小模型（DLM）虽能降低延迟，但牺牲了响应质量；而依赖云端大模型（CLM）则带来高昂的通信开销和延迟。

现有路由方法存在以下局限：
- 依赖**粗粒度的质量差距预测**（如reward score差值），忽略了输出之间的细粒度语义不一致性；
- 使用额外的编码器（如BERT）提取查询表示，增加设备端计算负担；
- 采用**全局静态阈值**进行路由决策，难以适应异构查询分布和动态网络环境。

---

### **提出的新方法：ConsRoute**
ConsRoute 是一种**轻量级、语义感知、自适应的查询路由框架**，其核心创新如下：

#### ✅ **创新点 1：基于语义一致性的软监督信号（Semantic Consistency Supervision）**
- 不再依赖标量质量分数差异作为训练目标；
- 引入 **reranker 模型**（如 Qwen3-Reranker-4B）直接衡量不同层级模型输出间的**语义相似度**，生成更精细的软标签（soft labels）；
- 结合 LLM judge 或规则增强标签（data augmentation），提升监督信号的鲁棒性和覆盖范围。

#### ✅ **创新点 2：复用 DLM 预填充隐藏状态进行轻量表示提取（Prompt-Guided Representation Reuse）**
- 在设备端将用户查询拼接一个固定指令提示（prompt）后输入 DLM；
- 利用 DLM **prefill 阶段最后一层 [EOS] token 的隐藏状态**作为语义表示；
- 避免引入额外编码器或前向传播，显著降低设备端开销；
- 提示设计引导模型关注“自身与更强模型的一致性”，实现任务对齐。

#### ✅ **创新点 3：基于聚类与贝叶斯优化的动态阈值机制（Cluster-Based Adaptive Thresholding）**
- 将查询按语义表示聚类（K-means），为每个簇学习独立的路由阈值；
- 使用 **Bayesian Optimization（BO）** 在线优化每个簇的 `(T1, T2)` 阈值，以最大化效用函数 `U = λ₁·Acc − λ₂·Latency − λ₃·Cost`；
- 支持**在线自适应更新**，应对查询分布漂移和网络条件变化。

---

### **相比现有方法的优势**
| 维度 | ConsRoute | 现有方法（如 RouteLLM, Zooter, MixLLM） |
|------|---------|----------------------------|
| **监督信号** | 基于语义一致性（fine-grained） | 基于质量/奖励分差（coarse-grained） |
| **表示提取** | 复用 DLM 隐藏状态，无额外开销 | 使用 BERT/embedding API，高内存/算力消耗 |
| **阈值策略** | 聚类+贝叶斯优化，动态可调 | 单一全局阈值，静态不可变 |
| **部署友好性** | 完全轻量化，适合移动端 | 依赖外部模型/API，不适合实时场景 |

---

## **2. 核心实验方法和设置**

### **使用的数据集**
- 主要基准：**RouterBench**（36.5K 查询，涵盖中英文）
- 子任务包括：
  - **MMLU**：通用知识问答
  - **GSM8K**：数学推理
  - **HumanEval**：代码生成
  - **MT-Bench**：多轮对话（由 GPT-4o 打分）

### **模型配置与部署环境**
| 层级 | 模型 | 硬件平台 |
|------|------|--------|
| **Device (DLM)** | Qwen3-1.7B | i5-12500H + RTX 3050 |
| **Edge (ELM)** | Qwen3-14B | RTX A6000 |
| **Cloud (CLM)** | Qwen3-32B | 双 RTX A6000 |

> 同时验证了跨家族部署：LLaMA-3.2-3B（device）、Qwen3-14B（edge）、DeepSeek-V3（cloud）

### **网络模拟设置**
- “Good” 条件：低延迟、高带宽（如 edge RTT=60ms）
- “Bad” 条件：高延迟、低带宽、丢包率高（如 cloud RTT=450ms）
- “Bad→Good”：模拟网络恢复过程

### **评估指标**
- **Accuracy**：相对于 CLM-only 的准确率百分比
- **Latency (%)**：归一化端到端延迟（DLM=0%, CLM=100%）
- **Cost (%)**：归一化推理成本（参数激活数 × 生成token数）
- **Utility Function**：综合权衡准确性、延迟、成本

### **基线方法对比**
| 方法 | 类型 | 是否支持预测路由 | 是否需额外编码器 |
|------|------|------------------|------------------|
| **DLM-only / Edge-only / CLM-only** | 固定路由 | ❌ | ❌ |
| **RouteLLM (BERT)** | 学习路由 | ✅ | ✅（BERT-base） |
| **RouteLLM (SW Ranking)** | 检索式路由 | ✅ | ✅（OpenAI Embedding API） |
| **MixLLM** | 多目标路由 | ✅ | ✅（Tag-enhanced BERT） |
| **ConsRoute (Ours)** | 一致性感知路由 | ✅ | ❌（复用 DLM 表示） |

---

## **3. 主要实验结果和性能指标**

### **关键性能数据**
- **达到 ≥95% 的云端模型性能水平**
- **端到端延迟降低约 40%**
- **推理成本降低约 30–40%**
- 在相同准确率下，比最优基线（MixLLM）节省 **10–20% 延迟**

> 示例：在 GSM8K 上，ConsRoute 达到 95% CLM 准确率仅需 60–65% 的延迟，而 MixLLM 需要 70–85%

---

### **与基线方法的对比结果**
| 方法 | 相对准确率 | 延迟（%） | 成本（%） | 设备开销 |
|------|------------|----------|----------|----------|
| **CLM-only** | 100% | 100% | 100% | 高 |
| **DLM-only** | ~70% | <10% | <10% | 极低 |
| **MixLLM** | ~93% | ~75% | ~70% | 中等（BERT） |
| **RouteLLM-BERT** | ~91% | ~80% | ~75% | 高（BERT） |
| **ConsRoute (Ours)** | **≥95%** | **~60%** | **~60%** | **极低（仅5M MLP）** |

> ConsRoute 在所有任务上均优于基线，在 MT-Bench 和 HumanEval 上优势尤为明显。

---

### **消融实验结果**
#### ✅ **Label Augmentation 的影响**
- 使用 reranker-only 监督 → 准确率较低
- 加入 reference-based 或 LLM judge 增强标签 → 显著提升性能
- **结论**：多源监督信号提高一致性判断可靠性

#### ✅ **Dynamic Thresholding 的作用**
- 固定全局阈值（Fixed Threshold）→ 性能下降明显
- 聚类+BO 动态阈值 → 更好适配复杂/敏感查询
- **结论**：个性化阈值显著改善长尾表现

#### ✅ **Prompting 策略比较（Table III）**
| 方法 | 准确率 | 路由延迟（ms） |
|------|-------|---------------|
| 显式选择（无CoT） | 85.79% | 36.2 |
| 显式选择（有CoT） | 86.44% | **1868.2** |
| 隐式选择（无prompt） | 82.34% | 20.4 |
| **隐式选择（prompted）✅** | **87.92%** | **20.9** |

> **ConsRoute 的 prompt-guided 隐式表示在精度和效率之间取得最佳平衡**

---

## **4. 关键结论和发现**

### **主要发现**
1. **语义一致性是比质量差距更可靠的路由监督信号**  
   → Reranker-based similarity 与人工标注一致性相关性更高（图4）

2. **复用 DLM 内部状态可实现零额外编码开销的高效表示提取**  
   → 路由延迟仅增加 ~20ms，远低于 RouteLLM 等需调用 BERT/API 的方案

3. **动态、聚类感知的阈值机制能有效适应异构查询和网络波动**  
   → ConsRoute-online 在分布漂移和网络恶化时仍保持高性能（图10–11）

4. **系统可在不同偏好间平滑调节**  
   → 通过调整效用权重 `λ₁/λ₂/λ₃` 控制质量 vs 效率权衡（图12）

---

### **方法的局限性**
- 当前 reranker 和 LLM judge 本身有一定延迟，限制了离线标签构建速度；
- 聚类数量 `K` 和初始化依赖历史数据，冷启动阶段可能表现不稳定；
- 对极端短文本或模糊查询的语义区分能力仍有挑战；
- 实验基于模拟环境，真实移动网络中的稳定性有待验证。

---

### **未来工作方向**
- 扩展至更多应用场景（如多模态、语音助手）；
- 探索更高效的在线反馈机制（如用户点击、满意度评分）；
- 研究更大规模的真实世界部署，测试跨厂商、跨架构模型组合下的泛化能力；
- 引入不确定性估计，进一步提升路由鲁棒性。

---

> **总结一句话**：  
> **ConsRoute 通过“语义一致性监督 + DLM 隐藏状态复用 + 贝叶斯驱动的动态阈值”，实现了高质量、低延迟、低成本的云-边-端 LLM 协同推理路由，在多项指标上全面超越现有方法。**

</details>

---

### 6. [EvoIdeator: Evolving Scientific Ideas through Checklist-Grounded Reinforcement Learning](https://arxiv.org/abs/2603.21728)

**Authors**: Andreas Sauter, Yuyue Zhao, Jacopo Urbani, Wenxiang Hu, Zaiqiao Meng, Lun Zhou, Xiaohui Yan, Yougang Lyu  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.21728v1  

#### Abstract
Scientific idea generation is a cornerstone of autonomous knowledge discovery, yet the iterative evolution required to transform initial concepts into high-quality research proposals remains a formidable challenge for Large Language Models (LLMs). Existing Reinforcement Learning (RL) paradigms often...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：EvoIdeator: Evolving Scientific Ideas through Checklist-Grounded Reinforcement Learning

---

## 1. 论文的主要贡献和创新点

### 解决了什么问题
当前在 **scientific idea generation**（科学创意生成）任务中存在一个“双重鸿沟”（dual gap）：
- **基于强化学习（RL）的方法** 通常依赖于标量奖励（scalar rewards），虽然能优化整体质量，但缺乏细粒度反馈（如具体哪部分需要修改、如何修改），无法指导模型进行可操作的迭代改进。
- **基于语言反馈（language feedback）的方法** 虽然能在推理时提供具体的文本级批评（span-level critiques），但这些方法多停留在 inference-time prompting 阶段，所使用的模型并未被显式训练以理解和利用此类反馈。

这导致了训练目标与推理过程之间的 **misalignment**，限制了模型自我演进能力的发展。

### 提出了什么新方法或新思路
提出 **EvoIdeator** 框架，首次将 **train-time RL** 与 **checklist-grounded language feedback** 显式对齐，实现端到端的科学创意演化系统。

其核心机制是双信号反馈循环：
1. **Lexicographic Rewards（字典序奖励）**  
   将科学质量分解为多个维度（如 Grounding、Feasibility、Method 等），并设定优先级顺序（primary vs. secondary objectives），通过非线性的字典序函数生成更合理的多目标优化信号，避免传统 MORL 中线性加权带来的稀释问题。

2. **Actionable Language Feedback（可执行的语言反馈）**  
   利用一个结构化的 **judge model**，基于预定义的 9 项 checklist 对生成的想法进行逐项评估，并输出：
   - 二值评分（0/1）
   - 若未达标，则提供 **span-level 的反馈指令**：`span_text`, `issue`, `suggestion`，明确指出原文中哪一段有问题、问题是什么、应如何修改。

这两个信号共同嵌入到 RL 训练循环中，使得策略模型不仅学会“得分高”，还学会“如何根据语言反馈进行精确修订”。

### 相比现有方法的优势
- ✅ **训练-推理一致性（Train-Inference Alignment）**：模型在训练阶段就接触语言反馈，在推理时能更有效地利用外部批评。
- ✅ **细粒度控制与可解释性增强**：相比单一标量奖励，checklist + span-level 反馈提供了更强的指导性和透明度。
- ✅ **小模型超越大模型**：基于 Qwen3-4B 的 EvoIdeator 在多项关键指标上显著优于更大的前沿模型（如 Gemini 3 Flash 和 DeepSeek-V3.2）。
- ✅ **泛化性强**：训练后的策略能够零样本迁移到不同来源的 judge 模型提供的反馈，展现出“插即用”（plug-and-play）潜力。

---

## 2. 核心实验方法和设置

### 使用了哪些数据集
作者构建了一个新的训练种子数据集，流程如下：
1. **Seed Paper Sampling**：从 OpenAlex 中随机采样 1,000 篇 2025 年发表的高质量论文（期刊/会议，非撤稿）。
2. **Query and Keyword Generation**：使用 Llama-3 为每篇 seed paper 生成一个研究问题（query）和检索关键词。
3. **Literature Review Synthesis**：基于 Semantic Scholar API 检索相关文献，再由 LLM 合成一段 1–2 段的综述，描述领域现状与 gap，不提出新方案。

最终得到 `(query, literature_review)` 对作为输入上下文 $ p_0 $。测试集包含 **96 个独立样本**，与训练集分离。

### 实验设置和评估指标
#### 评估框架
采用两阶段评估协议：
- **Generation Step**：仅输入 query 和 literature review，模型一次性输出初始想法（zero-shot）。
- **Refinement Step**：模型接收自身前一轮输出及来自 judge 的语言反馈，生成改进版本。

#### 评估指标
使用 **9 项 checklist** 进行打分（均为 binary 0/1）：
| 类别 | 指标 |
|------|------|
| **Primary Objectives**（科学严谨性） | Grounding, Feasibility, Problem, Risk, Method |
| **Secondary Objectives**（格式与新颖性） | Writing, Innovation, Length, Layout |

> 注：Layout 所有模型表现接近完美，故主表中省略；实际评估共 9 项。

报告 **平均得分 ± 95% 置信区间**，剔除无效 `<idea>` 输出。

### 基线方法对比
- **Qwen-4B**：未对齐的基础模型（zero-shot baseline）
- **DeepSeek R1 Distill**：经过 process supervision 优化的推理模型
- **DeepSeek-V3.2**：大规模开源 LLM，用于评分 judge 和对比
- **Gemini 3 Flash**：代表前沿闭源大模型性能上限

---

## 3. 主要实验结果和性能指标

### 关键性能数据（见 Table 1）

| 方法 | Grounding ↑ | Feasibility ↑ | Problem ↑ | Risk ↑ | Method ↑ | Writing ↑ | Innovation ↑ | Length ↑ |
|------|-------------|----------------|-----------|--------|----------|------------|---------------|-----------|
| **Qwen-4B (base)** | .85±.07 | .09±.06 | .05±.05 | .01±.02 | .08±.06 | .75±.09 | .21±.08 | .41±.10 |
| **Gemini 3 Flash** | .87±.07 | .20±.08 | .07±.05 | .00±.00 | .10±.06 | .90±.06 | .55±.10 | .81±.08 |
| **EvoIdeator (Ours)** | **.99±.02** | **.19±.08** | **.06±.05** | **.03±.04** | **.18±.08** | **.96±.04** | **.32±.10** | **.37±.10** |
| → 经过 **Refinement Step** 后 | **.99±.02** | **.31±.09** | **.94±.05** | **.35±.10** | **.58±.10** | **.99±.02** | **.47±.10** | **.18±.08** |

> ✅ 表示 EvoIdeator 在该列排名第一；下划线表示第二。

#### 核心发现：
- 在 **Refinement Step** 中，EvoIdeator 在所有 **Primary Objectives** 上均取得最高分：
  - **Grounding**: 0.99（几乎满分）
  - **Problem**: 0.94 > Gemini 的 0.90
  - **Risk**: 0.35 > Gemini 的 0.16
  - **Method**: 0.58 > Gemini 的 0.48
- 即使基础模型仅为 **4B 参数量**，仍全面超越 **Gemini 3 Flash** 和 **DeepSeek-V3.2** 等更大模型。

### 消融实验结果（RQ2 分析）

通过比较四种配置验证各组件作用：
- **Informed**：完整 EvoIdeator（带 feedback 训练）
- **Non-Informed**：无 feedback 的 RL 版本
- **Base + feedback**：基础模型 + 推理时 feedback
- **Base only**：基础模型无 feedback

#### 结果（图 2）：
- **RL 训练提升初始生成质量**：无论是否使用 feedback，EvoIdeator 的初始生成（Step 0）均远超 base model，说明 lexicographic reward 成功内化了科学标准。
- **Language Feedback 实现有效 refinement**：只有接受 feedback 的模型（Informed 和 Base+fb）在 Step 1 显著提升；而依赖 self-correction 的模型停滞不前。
- **双重机制具有加性增益（additive effect）**：EvoIdeator 同时具备高起点（来自 RL）和强改进斜率（来自 feedback），二者叠加带来显著优势。

### 跨 judge 泛化能力（RQ3）

| Feedback Provider | Generation Step | Refinement Step |
|-------------------|------------------|------------------|
| DS R1 14B Distill | 4.07±.19 | 5.64±.22 |
| DS R1 70B Distill | 4.09±.17 | 5.81±.22 |
| DS v3.2           | 4.05±.18 | **6.02±.23** |
| **Gemini 3 Flash** | 3.97±.19 | **5.13±.25** ↓↓ |

- ✅ 在 DeepSeek 家族内部，随着 judge 能力增强，refinement 效果单调上升 → 显示出良好的 **lineage 内泛化性**。
- ❌ 使用 Gemini 作为 feedback provider 时性能大幅下降 → 表明 EvoIdeator 对 feedback 的 **“语体风格”敏感**，跨家族迁移受限。

> 说明：语言反馈本质上是一种 learned communication protocol，需风格匹配才能生效。

---

## 4. 关键结论和发现

### 论文的主要发现
1. **Train-Inference Alignment 是关键**：将训练目标与推理时的 feedback loop 显式对齐，可显著提升模型对批评的理解与执行能力。
2. **Dual-Signal Design 具有加性收益**：Lexicographic reward 提供全局优化方向，language feedback 提供局部编辑路径，两者结合优于任一单独信号。
3. **小模型也能达到 SOTA 性能**：基于 Qwen3-4B 的 EvoIdeator 在 primary scientific criteria 上超越更大模型，证明了高效训练范式的潜力。
4. **Feedback Protocol 可泛化但具风格依赖性**：模型可在同一体系内无缝升级至更强 judge，但跨体系需统一 feedback format 才能适配。

### 方法的局限性
1. **Secondary Objectives 被牺牲**：由于字典序优先保障 primary objectives，导致 **Innovation** 和 **Length Compliance** 得分偏低。
2. **Feedback Dialect Sensitivity**：目前无法很好地处理来自不同训练谱系（如 Gemini vs. DeepSeek）的 feedback，限制了通用性。
3. **依赖 LLM-based Judge**：尽管自动化评估已是常态，但仍存在 judge bias 风险（如 self-preference bias）。
4. **固定 refinement 步数**：实验仅使用单步 refinement，未探索多轮动态演化的最优策略。

### 未来工作方向
- 探索 **adaptive weighting** 或 **Pareto-based MORL** 策略，动态平衡 primary 与 secondary objectives。
- 设计 **标准化 feedback schema**，打破 model family 间的通信壁垒，提升跨 judge 兼容性。
- 引入 **human-in-the-loop evaluation**，增强评估结果的可信度。
- 研究 **longer-horizon refinement trajectories** 与 compute allocation 策略，推动 fully autonomous 科学发现 agent 发展。

--- 

> 📌 **一句话总结**：  
> EvoIdeator 通过将 checklist-grounded language feedback 与 lexicographic RL reward 融合，首次实现了训练与推理一致的科学创意演化框架，在小模型上达成 SOTA，并展示了强大的 feedback 利用与跨 judge 泛化潜力，为构建自进化 AI Scientist 提供了一条可行路径。

</details>

---

### 7. [Benchmarking Message Brokers for IoT Edge Computing: A Comprehensive Performance Study](https://arxiv.org/abs/2603.21600)

**Authors**: Tapajit Chandra Paul, Pawissanutt Lertpongrujikorn, Hai Duc Nguyen, Mohsen Amini Salehi  
**Category**: cs.DC  
**Published**: 2026-03-24  
**Score**: 10.0  
**Type**: new  
**ArXiv ID**: 2603.21600v1  

#### Abstract
Asynchronous messaging is a cornerstone of modern distributed systems, enabling decoupled communication for scalable and resilient applications. Today's message queue (MQ) ecosystem spans a wide range of designs, from high-throughput streaming platforms to lightweight protocols tailored for edge and...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Benchmarking Message Brokers for IoT Edge Computing: A Comprehensive Performance Study*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
当前在 **IoT 和边缘计算** 场景中，选择合适的 **Message Broker** 非常困难。尽管已有大量关于消息队列系统（如 MQTT）的基准测试研究，但这些研究存在以下局限：
- 多数仅聚焦单一协议（如 MQTT），缺乏跨协议比较；
- 主要关注吞吐量和延迟，忽略 **CPU 和内存占用** 等资源效率指标；
- 实验环境固定，未考虑边缘设备常见的资源约束（如 1 vCPU / 2GB RAM）。

这导致开发者难以在实际部署中做出合理选择。

### 🛠 提出的新方法：`mq-bench`
作者提出并开源了一个统一的基准测试框架 —— **mq-bench**，其核心创新包括：
- **跨协议支持**：支持 MQTT、AMQP、NATS、RESP（Redis）、Zenoh 等多种协议；
- **高精度测量**：基于 Rust + Tokio 构建，实现纳秒级延迟采样；
- **资源监控集成**：通过 Docker Stats API 实时采集 CPU 和内存（RSS）使用情况；
- **网络故障注入**：集成 Toxiproxy 支持模拟 TCP RST 断连等真实网络异常。

### 🔍 相比现有方法的优势
| 维度 | 现有研究 | 本文工作 |
|------|--------|---------|
| 协议覆盖 | 单一协议（如仅 MQTT） | 跨协议统一比较（MQTT, AMQP, NATS, Zenoh, Redis） |
| 指标维度 | 吞吐量、延迟为主 | 增加 CPU 利用率、内存占用、尾部延迟（p95） |
| 硬件配置 | 固定高性能服务器 | 模拟边缘场景（1–4 vCPU, 2–8 GB RAM） |
| 可复现性 | 工具不统一或闭源 | 开源 `mq-bench` 框架，确保可复现 |

> 💡 **优势总结**：首次在一个标准化、可复现的框架下，对 **8 种主流 Message Broker** 在边缘计算典型资源限制下的性能进行全面横向评测。

---

## 2. 核心实验方法和设置

### 🧪 实验对象（Brokers）
共评估 8 个广泛使用的 Message Broker：
| Broker | 类型 | 实现语言 | 协议 |
|-------|------|----------|------|
| Mosquitto | MQTT 轻量级 | C | MQTT |
| EMQX | MQTT 高并发 | Erlang | MQTT |
| HiveMQ CE | 企业级 MQTT | Java | MQTT |
| RabbitMQ | 多协议 | Erlang | AMQP/MQTT |
| ActiveMQ Artemis | 多协议 | Java | MQTT/AMQP |
| NATS Server | 云原生 | Go | NATS |
| Redis (Pub/Sub) | 内存数据库 | C | RESP |
| Zenoh Router | 数据中心 | Rust | Zenoh |

所有 Broker 均以单节点模式运行于 Docker 容器中，避免集群协调开销。

### 💻 实验平台与资源配置
使用 Chameleon Cloud 提供的裸金属节点构建测试床，分为两个角色：
- **Workload Generator Machine**：运行 `mq-bench` 并生成客户端负载；
- **Broker Execution Machine**：托管 Broker 容器，隔离执行环境。

#### VM 配置（模拟边缘硬件）
| 配置 | 描述 | 典型对应设备 |
|------|------|-------------|
| 1 vCPU / 2 GB RAM | 资源极度受限 | Raspberry Pi 类网关 |
| 2 vCPU / 4 GB RAM | 中端边缘节点 | 小型工业 PC |
| 4 vCPU / 8 GB RAM | 高性能雾节点 | 边缘服务器 |

### 📊 实验设计与评估指标

#### 三大实验场景：
| 实验 | 目标 | 参数 |
|------|------|------|
| **Experiment 1**: Latency vs Payload Size | 测试不同消息大小对延迟的影响 | 1KB, 16KB, 1MB；10 pub-sub 对；QoS 0 |
| **Experiment 2**: Throughput & Resource Scaling | 测试连接数增长下的吞吐与资源消耗 | 500–10,000 pub-sub 对；每发布者 10 msg/s |
| **Experiment 3**: QoS Reliability under Network Failures | 测试网络中断下 QoS 保证能力 | MTTF=30s, MTTR=5s；测试 QoS 0/1/2 |

#### 评估指标
| 指标 | 测量方式 |
|------|--------|
| **Latency** | 发送时间戳嵌入消息头，接收端计算 end-to-end 延迟；报告 p50 和 p95 |
| **Throughput** | 成功接收的消息总数 / 稳定期持续时间（msg/s） |
| **CPU Utilization** | Docker Stats API 采样，单位为 vCPU 百分比 |
| **Memory Footprint** | RSS（Resident Set Size），单位 MB |
| **Reliability** | 消息丢失率（尤其在 QoS 1/2 下是否实现“至少一次”或“恰好一次”） |

### 🔁 基线对比策略
- 所有 Broker 在相同硬件、相同负载条件下测试；
- 使用统一工具链（mq-bench）控制实验流程；
- 不进行调优“作弊”，仅启用基本优化（如调整线程池匹配 vCPU 数）；
- 排除无法完成测试的配置（如 RabbitMQ-AMQP 在高并发下早期崩溃）。

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据汇总

#### ✅ Experiment 1: Latency vs Payload Size（4 vCPU, 8GB）
| Broker | 1KB p50 (ms) | 16KB p50 (ms) | 1MB p50 (ms) |
|--------|---------------|----------------|----------------|
| **NATS** | **0.21** | **0.39** | **8.27** |
| **Redis** | 0.26 | 0.41 | 8.58 |
| **Zenoh** | 0.35 | 0.42 | 7.96 |
| Mosquitto | 0.27 | 0.39 | 11.18 |
| EMQX | 0.40 | 0.52 | 12.42 |
| HiveMQ | 0.76 | 1.24 | 17.41 |
| Artemis | 0.69 | 1.18 | 21.60 |

> 🔎 **发现**：对于小消息（1KB），各 Broker 表现接近；随着 payload 增大，**JVM-based（HiveMQ, Artemis）延迟显著上升（2–3×）**，而 native 实现（NATS, Zenoh）保持领先。

#### ✅ Experiment 2: Throughput & Scaling（最大可持续吞吐）

| Broker | 最大吞吐量（msg/s） | 触发条件 | 内存占用峰值 |
|--------|--------------------|----------|--------------|
| **Zenoh** | **~90K** | 9000 pub-sub 对 | 1.7 GB |
| **NATS** | **~90K** | 9000 pub-sub 对 | 1.7 GB |
| **Artemis** | ~70K | 7000 pub-sub 对 | 5.5 GB |
| **EMQX** | ~47K | 5000 pub-sub 对 | 3.7 GB |
| **RabbitMQ-MQTT** | ~50K | 6000 pub-sub 对 | 2.1 GB |
| **Mosquitto** | ~45K | 10000 pub-sub 对 | 620 MB |
| **Redis** | ~30K | 10000 pub-sub 对 | 90 MB |
| **HiveMQ** | ~40K | 4000 pub-sub 对 | 5.5 GB |

> ⚠️ 注意：**Mosquitto 是单线程架构，在多核环境下无法扩展**，即便增加到 4 vCPU，性能提升有限。

#### ✅ Experiment 2: 资源效率对比
| 类别 | 内存表现 |
|------|--------|
| **Native Brokers (C/Rust/Go)** | Redis 仅占 **66–90MB**，Zenoh/NATS 约 1.1–1.7GB |
| **JVM-based Brokers (Java)** | HiveMQ/Artemis 占用 **1.6–5.5GB**，是 native 的 **10–50×** |

> 🔥 特别指出：**Artemis 在 4 vCPU 下达到 70K msg/s，超过 Redis 和 Mosquitto**，说明“管理型运行时”在资源充足时也能表现出色。

#### ✅ Experiment 4: Fanout 拓扑（广播场景）
- 场景：1 个 publisher → N 个 subscribers，目标吞吐 = N × 100 msg/s
- 结果：
  - **Zenoh** 实现 **850K msg/s @ 10,000 subscribers**，CPU 仅用 ~2 cores；
  - **NATS** 在 4000 订阅者后急剧下降至 167K msg/s，尽管 CPU 已饱和；
  - 其他多数 Broker（Mosquitto, Redis, EMQX）均卡在 ~100K msg/s。

> 🔄 **反转现象**：在点对点拓扑中 NATS 与 Zenoh 性能相当，但在 fanout 场景中 **Zenoh 明显胜出**，表明其广播路径高度优化。

#### ✅ Experiment 3: QoS 可靠性与延迟
| QoS Level | 消息丢失率 | 关键观察 |
|----------|------------|---------|
| QoS 0 | ~6.5% | 所有 Broker 表现一致，无重传机制 |
| QoS 1 / 2 | **0%** | 所有 MQTT Broker 均能通过 session persistence 实现可靠传递 |
| 尾部延迟（p95/p99） | 475–665ms / ~3s | 主要由 MTTR（5s）期间缓存引起，与 Broker 架构无关 |

> ⏱️ **延迟代价分析**：
- 多线程 Broker（EMQX, RabbitMQ）处理 ACK 异步，QoS 2 下延迟仅轻微上升；
- **Mosquitto（单线程）在 QoS 2 下延迟飙升至 7.93ms（+3.6×）**，因其必须串行处理握手。

---

## 4. 关键结论和发现

### ✅ 主要发现总结

| 发现 | 详细说明 |
|------|---------|
| **1. 轻量级 native Broker 更适合边缘设备** | 如 Mosquitto、Redis 在低资源下稳定且内存极低，适合 Raspberry Pi 等场景 |
| **2. 多线程 native Broker（Zenoh, NATS）扩展性强** | 随 vCPU 增加线性提升吞吐，Zenoh 在 fanout 场景表现尤为突出 |
| **3. JVM-based Broker 资源消耗巨大** | HiveMQ、Artemis 内存占用达 GB 级，需谨慎用于内存受限边缘节点 |
| **4. 单线程架构成为瓶颈** | Mosquitto 虽稳定，但无法利用多核优势，大负载下易饱和 |
| **5. QoS 保障依赖 session 持久化** | QoS 1/2 下所有 MQTT Broker 均实现零丢包，但单线程 Broker 延迟显著增加 |
| **6. 拓扑影响性能排名** | 在 fanout 场景中，Zenoh 凭借高效广播机制反超 NATS，体现架构差异的重要性 |

### ⚠️ 方法的局限性
- **仅测试单节点部署**：未涵盖集群、分布式场景下的容错与一致性行为；
- **未测试持久化存储性能**：所有实验假设消息无需落盘，忽略了磁盘 IO 影响；
- **负载模型较理想化**：采用均匀速率发送，未模拟突发流量或真实 IoT 数据分布；
- **缺少 request-reply 语义测试**：目前仅覆盖 pub/sub 模式。

### 🔮 未来工作方向
- 扩展 `mq-bench` 支持 **multi-node 集群部署** 与自动扩缩容测试；
- 引入更复杂的 **workload pattern**，如 fan-in、request-reply、流控背压等；
- 在广域网（WAN）条件下测试 **跨区域延迟与分区容忍性**；
- 探索 AI-driven 自适应 Broker 选型推荐系统。

---

## 🧭 实践建议（Deployment Guidelines）

| 场景 | 推荐 Broker | 理由 |
|------|-------------|------|
| **资源极度受限边缘设备** | Mosquitto, Redis | 内存占用小，稳定性好 |
| **中高端边缘节点（2–4 vCPU）** | Zenoh, NATS | 多线程架构充分利用多核，吞吐高 |
| **低延迟关键应用** | NATS, Zenoh, Redis | 原生实现，p50 < 1ms |
| **需要高级功能的企业部署** | RabbitMQ, Artemis | 支持复杂路由、事务、插件生态 |
| **大规模广播/通知场景** | **Zenoh** | 在 fanout 拓扑中表现最优，CPU 利用率低 |
| **要求 QoS 可靠且低延迟** | EMQX, RabbitMQ | 多线程处理 ACK，不影响主路径 |

> 📌 **一句话总结**：  
> “**没有最好的 Broker，只有最适合的场景。**”  
> 选择应综合考虑 **资源预算、消息模式、拓扑结构、可靠性需求** 和 **延迟敏感度**。

</details>

---

### 8. [MemDLM: Memory-Enhanced DLM Training](https://arxiv.org/abs/2603.22241)

**Authors**: Zehua Pei, Hui-Ling Zhen, Weizhe Lin, Sinno Jialin Pan, Yunhe Wang, Mingxuan Yuan, Bei Yu  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 9.5  
**Type**: new  
**ArXiv ID**: 2603.22241v1  

#### Abstract
Diffusion Language Models (DLMs) offer attractive advantages over Auto-Regressive (AR) models, such as full-attention parallel decoding and flexible generation. However, they suffer from a notable train-inference mismatch: DLMs are trained with a static, single-step masked prediction objective, but ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*MemDLM: Memory-Enhanced DLM Training*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决了什么问题

Diffusion Language Models (DLMs) 虽然在生成效率、双向上下文感知等方面优于传统的 Auto-Regressive (AR) 模型，但存在显著的 **train-inference mismatch**（训练-推理不匹配）问题：

- **训练阶段**：采用静态单步掩码预测（Static Masked Diffusion Language Modeling, MDLM），模型直接从高噪声状态 $ x_t $ 预测原始文本 $ x_0 $。
- **推理阶段**：通过多步渐进去噪（iterative denoising）逐步恢复文本，每一步依赖自身先前的预测。

由于训练中从未模拟这种“自回归式”的去噪轨迹，导致模型对中间噪声敏感，出现 **exposure bias**（暴露偏差），影响长上下文理解与信息检索能力。

---

### ✅ 提出了什么新方法或新思路

作者提出 **MemDLM**（Memory-Enhanced DLM），其核心思想是通过 **Bi-level Optimization** 在训练过程中嵌入一个模拟的渐进去噪过程，并引入 **Parametric Memory**（参数化记忆）机制来缓解上述问题。

#### 创新点如下：

1. **Bi-level Optimization 架构**：
   - **内层循环（Inner Loop）**：更新一组轻量级的 **fast weights**（如 LoRA adapter），模拟局部去噪轨迹，形成样本特定的 **Parametric Memory**。
   - **外层循环（Outer Loop）**：基于该记忆更新主模型参数 $ \theta $，使基础模型学习更鲁棒的表示。

2. **Anchor-Consistent Local Trajectory 设计**：
   - 内环分两步进行：
     - **Pre-Anchor Alignment**：从更嘈杂的状态 $ x_{t'} $（$ t' > t $）向锚定状态 $ x_t $ 去噪。
     - **Anchor-to-Target**：从 $ x_t $ 向最终目标 $ x_0 $ 去噪。
   - 这种设计确保 fast weights 显式捕捉以 $ x_t $ 为中心的局部轨迹经验。

3. **推理时可激活的适应路径**：
   - 推理时可重新启用内环，在输入 prompt 上执行 fast-weight 更新，实现 **prompt-specific adaptation**。
   - 此过程被解释为一种 emergent **in-weight retrieval mechanism**（权重内检索机制），有助于缓解 token-level attention bottleneck。

---

### ✅ 相比现有方法的优势

| 方面 | MemDLM 的优势 |
|------|----------------|
| **优化对齐性** | 缩小了训练与推理之间的分布差距，降低 exposure bias。 |
| **上下文建模能力** | 减少对脆弱 token 表示的依赖，提升长上下文信息保持能力。 |
| **训练效率** | 更快收敛，更低训练损失（见 Figure 4）。 |
| **零样本泛化** | 即使关闭推理时 adaptation（Train-Only），仍显著优于标准 MDLM。 |
| **灵活性** | 支持长度外推（如 16K/32K），且可通过 inference-time adaptation 进一步增强性能。 |

---

## 2. 核心实验方法和设置

### 📚 使用的数据集

| 数据集 | 类型 | 描述 |
|--------|------|------|
| **LongAlpaca** | 训练数据 | 用于 instruction tuning，强调长上下文理解和生成能力，最大长度限制为 4K tokens。 |
| **RULER** | 测试基准 | “Needle-in-a-Haystack”（NIAH）任务，测试模型在超长文本中定位关键信息的能力，包含：<br>- MV (Multi-Value)<br>- VT (Variable Tracking)<br>- CWE (Common Words Extraction) |
| **BABILong** | 测试基准 | 多跳推理任务，要求模型在数千 token 上下文中追踪事实并回答问题。 |
| **LongBench** | 综合评估 | 包含多文档问答、摘要、代码补全等真实场景任务，全面评估 long-context generalization。 |

---

### ⚙️ 实验设置和评估指标

| 设置项 | 配置说明 |
|-------|----------|
| **Backbone 模型** | - LLaDA-MoE-7B-A1B-Base<br>- LLaDA2.1-mini（简称 LLaDA-MoE 和 LLaDA2.1） |
| **训练策略** | - 使用 4-bit 量化加载 base model<br>- 外环使用 LoRA（rank=32, α=64）<br>- 内环 fast weights 也为 LoRA，仅作用于最后 10% 的 FFN 层 |
| **优化器** | - 外环：AdamW，lr=2e-5，cosine scheduler，warmup 0.1<br>- 内环：SGD，lr=0.1，clip=1.0，单轮更新 |
| **评估方式** | 所有模型使用相同 generation config，公平比较 |

---

### 🔁 基线方法对比

| 方法 | 描述 |
|------|------|
| **Standard MDLM** | 基线方法，仅使用标准单步 MDLM 目标函数，无任何轨迹模拟或 fast weights。 |
| **MemDLM (Train-Only)** | 训练时启用内环，推理时不启用（zero-shot）。 |
| **MemDLM (Train & Inference)** | 训练 + 推理均启用内环，允许 inference-time adaptation。 |

---

## 3. 主要实验结果和性能指标

### 📊 关键性能数据（来自 Table 1）

#### 在 **LLaDA-MoE** 上的表现（8K context）：

| 方法 | RULER-VT ↑ | BABILong ↑ |
|------|------------|-----------|
| Standard MDLM | 78.84 | 41.80 |
| MemDLM (Train-Only) | **94.84** (+16.0) | **45.60** (+3.8) |
| MemDLM (Train & Inference) | **95.80** (+16.96) | **47.00** (+5.2) |

> → 在最难的 Variable Tracking 任务上提升近 **17个百分点**！

#### 在 **LLaDA2.1** 上的表现（8K context）：

| 方法 | RULER-VT ↑ | BABILong ↑ |
|------|------------|-----------|
| Standard MDLM | 43.72 | 47.40 |
| MemDLM (Train-Only) | **59.15** (+15.43) | **55.00** (+7.6) |
| MemDLM (Train & Inference) | **60.72** (+17.0) | **57.00** (+9.6) |

> → 在 BABILong 上从 47.4 提升至 57.0，绝对增益达 **9.6**。

---

### 🔍 长度外推能力（Table 2）

在 **16K 和 32K context** 下测试（超出训练长度）：

| 方法 | RULER-VT @16K | RULER-VT @32K |
|------|---------------|---------------|
| Standard MDLM | 52.56 | 9.48 |
| MemDLM (Train-Only) | 55.30 | 10.35 |
| MemDLM (Train & Inference) | **56.84** | **11.44** |

> → 即使在极端长度下，MemDLM 依然保持相对稳定，证明其更强的 **length extrapolation** 能力。

---

### 🧪 消融实验结果（Ablation Studies）

| 实验维度 | 发现 |
|---------|------|
| **Trajectory Consistency**（图9） | 一致性的 anchor-centered 轨迹设计至关重要，不一致版本性能下降明显（0.684 vs 0.604）。 |
| **Two-Stage Inner Loop**（图10） | 必须同时包含 Pre-Anchor 和 Anchor-to-Target 两个阶段，缺一不可；完整设计达到 0.684，单独阶段仅 ~0.62–0.65。 |
| **Multiple Pre-Anchor Steps**（图11） | 增加步骤虽降低训练损失，但损害下游性能（3-step: 0.644, 4-step: 0.590），表明过拟合辅助路径有害。 |
| **Adaptation Scope**（图7） | 最佳效果出现在 **FFN-only + 最后10%层** 的设定，全参数更新反而更差（0.602），说明适度约束更有利。 |
| **Gradient Normalization** | 局部归一化（per-parameter）优于全局归一化（0.684 vs 0.632），而梯度裁剪阈值影响较小。 |
| **Inference Anchor Ratio**（图8） | 对 mask ratio 不敏感（0.2–0.8 效果接近），说明设计鲁棒性强。 |

---

## 4. 关键结论和发现

### ✅ 主要发现

1. **Train-inference mismatch 是制约 DLM 性能的关键瓶颈**，尤其在长上下文任务中表现明显。
2. **Parametric Memory 可有效缓解此问题**：将局部轨迹经验写入 fast weights 参数空间，减少对易受污染的 token 表示的依赖。
3. **MemDLM 的收益主要来自训练阶段**：即使不开启推理 adaptation（Train-Only），也能获得显著提升，说明 base model 学到了更鲁棒的表示。
4. **推理时 adaptation 提供额外增益**：表现为一种 emergent **in-weight retrieval** 机制，进一步强化 prompt 内容的记忆与利用。
5. **方法具备良好的长度外推能力**：在 16K/32K 上仍优于 baseline，显示其泛化潜力。

---

### ⚠️ 方法的局限性

| 局限性 | 说明 |
|--------|------|
| **计算开销增加** | 内环需额外前向/反向传播，带来约 15–20% 的训练时间增长（尽管 fast weights 很轻量）。 |
| **依赖高质量 backbone** | 当前实验集中在较强 DLM 架构上，是否适用于弱模型尚待验证。 |
| **尚未集成到 decoding loop 中** | 当前 inference-time adaptation 仅应用于 prompt，未在生成过程中动态调整，仍有改进空间。 |
| **理论分析有限** | 虽然实证有效，但关于为何 fast weights 能形成有效 memory 的机理仍缺乏深入解释。 |

---

### 🔮 未来工作方向

1. **Dynamic Inference-Time Adaptation**：在每一步生成中动态运行内环，实现真正的“边生成边记忆更新”。
2. **Cross-task Generalization**：探索 MemDLM 是否可用于 few-shot 或 domain adaptation 场景。
3. **Efficient Implementation**：进一步压缩 fast weights 开销，使其更适合大规模部署。
4. **结合 RL 或 Planning**：将 trajectory-aware training 与 planner-alignment 方法结合，构建更智能的 DLM 推理流程。
5. **扩展至 Vision & Multimodal**：将 Parametric Memory 思路迁移到 diffusion vision models 中。

---

## 总结一句话

> **MemDLM 通过 Bi-level Optimization 将渐进去噪轨迹“内化”为参数空间中的 Parametric Memory，不仅显著提升了 DLM 的长上下文理解与信息检索能力，还揭示了一条通过 memory-aware training 来弥合 train-inference gap 的新路径。**

</details>

---

### 9. [CurvZO: Adaptive Curvature-Guided Sparse Zeroth-Order Optimization for Efficient LLM Fine-Tuning](https://arxiv.org/abs/2603.21725)

**Authors**: Shuo Wang, Ziyu Chen, Ming Tang  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.21725v1  

#### Abstract
Fine-tuning large language models (LLMs) with backpropagation achieves high performance but incurs substantial memory overhead, limiting scalability on resource-constrained hardware. Zeroth-order (ZO) optimization provides a memory-efficient alternative by relying solely on forward passes, yet it ty...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：CurvZO: Adaptive Curvature-Guided Sparse Zeroth-Order Optimization for Efficient LLM Fine-Tuning

---

## 1. 论文的主要贡献和创新点

### 解决的问题
大型语言模型（LLM）通过 **backpropagation** 进行微调虽然性能优越，但会带来巨大的内存开销，限制了在资源受限硬件上的可扩展性。**Zeroth-Order (ZO) optimization** 虽然通过仅依赖前向传播实现了内存高效，但其梯度估计具有高方差，导致收敛慢且不稳定。

此外，现有的稀疏 ZO 方法（如 Sparse-MeZO、SubZero）缺乏有效的机制来识别应被扰动的关键参数，而像 SensZOQ 这类依赖预计算统计量的方法又引入了额外计算成本，违背了 ZO 方法“轻量、简洁”的初衷。

---

### 提出的新方法：CurvZO
本文提出 **Adaptive Curvature-Guided Sparse Zeroth-Order Optimization (CurvZO)**，一种无需依赖预计算信息、即插即用的稀疏 ZO 微调框架，其核心思想是：

- **在线追踪曲率信号（curvature signal）**：从每次 ZO 查询返回的标量反馈（scalar response）中提取局部损失曲面的几何信息。
- **基于曲率指导稀疏扰动**：利用这些信号构建参数级采样分布，优先扰动高曲率方向的参数。
- **自适应调整扰动预算（perturbation budget）**：根据曲率信号分布的演化动态调整每次更新中扰动的参数数量，平衡探索与聚焦。

---

### 相比现有方法的优势
| 方面 | CurvZO 的优势 |
|------|----------------|
| **内存效率** | 与 MeZO 同级，远优于 full fine-tuning（FT），适用于单卡训练大模型（如 OPT-6.7B）。 |
| **收敛速度** | 显著快于 MeZO 和 DiZO，在部分任务上实现约 2× 加速。 |
| **最终性能** | 准确率最高提升达 **4.4 个百分点**，且在分类与生成任务上均表现优异。 |
| **无需额外信息** | 不依赖预训练阶段的梯度统计（如 Fisher 信息），完全在线学习，保持 ZO 方法的简洁性。 |
| **通用性强** | 在 OPT 和 Llama 系列模型上均有效，支持与 LoRA 结合使用。 |

---

## 2. 核心实验方法和设置

### 使用的数据集
实验基于 **SuperGLUE** 基准及其扩展，涵盖多种 NLP 任务：
- **分类任务**：SST-2（情感分析）、RTE、CB、BoolQ、WSC、WIC
- **生成任务**：SQuAD、DROP

所有任务均采用 **1,000 条训练样本、500 验证、1,000 测试** 的小样本设定，符合 LLM 微调研究惯例。

---

### 实验设置和评估指标
| 设置项 | 说明 |
|--------|------|
| **模型系列** | OPT（2.7B, 6.7B）、Llama2（7B, 13B） |
| **评估指标** | 主要为 **Accuracy (%)**；辅以 **GPU 小时数（GPU hours）**、**收敛步数**、**内存占用（GB）** |
| **训练步数** | 固定为 20,000 步 |
| **扰动方式** | 使用 block-wise curvature scores（每个 block 对应一个原生参数张量）以降低开销 |
| **代码开源** | 已公开：https://anonymous.4open.science/r/CurvZO-9F35 |

---

### 基线方法对比
| 基线方法 | 类型 | 特点 |
|----------|------|------|
| **FT (Full Fine-tuning)** | First-Order (FO) | 使用 backpropagation 的标准微调，性能上限参考 |
| **LoRA** | Parameter-Efficient FO | 低秩适配，减少可训练参数 |
| **MeZO** | ZO Baseline | 经典 ZO 方法，全参数扰动，内存高效但收敛慢 |
| **DiZO** | ZO Baseline | 分层调节 ZO 更新，提升收敛性 |
| **MeZO/DiZO + LoRA** | Hybrid | ZO 与 LoRA 结合，兼顾稀疏性与参数效率 |

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

#### 表格结果摘要（Accuracy %）
| 模型 | 方法 | Average Accuracy ↑ |
|------|------|--------------------|
| OPT-2.7B | MeZO | 64.2 |
| OPT-2.7B | DiZO | 61.4 |
| OPT-2.7B | **CurvZO** | **66.8** ✅ |
| OPT-6.7B | MeZO | — |
| OPT-6.7B | **CurvZO** | **显著优于 MeZO 和 DiZO**（见图2） |
| Llama2-7B / 13B | **CurvZO** | 在所有任务上一致超越 MeZO 和 DiZO（见 Figure 2） |

> ⭐ 最高提升达 **4.4 pts**（如 WSC 上提升 14.3%）

---

### 与基线方法的对比结果
| 指标 | 结果 |
|------|------|
| **准确率提升** | 在 8 个任务中，CurvZO 在 7 个任务上优于所有 ZO 基线；在 LoRA 设置下也全面领先。 |
| **收敛速度** | 在 BoolQ 上比 MeZO 快 **2.4×**，在 RTE 上快 **2.0×**（Figure 3）。 |
| **GPU 时间节省** | 总 GPU 小时数最多减少 **59%（BoolQ）** 和 **51%（RTE）**（Figure 4），相当于 **接近 2× 速度提升**。 |
| **内存消耗** | 与 MeZO 几乎持平（OPT-2.7B：5.91 GB；OPT-6.7B：~13.95 GB），远低于 FT（>80 GB）（Table 3）。 |

---

### 消融实验结果（Ablation Study）
在 `Table 4` 中验证两个核心组件的作用：

| 方法 | Average Acc (no LoRA) | 说明 |
|------|------------------------|------|
| **US+AB**（均匀采样 + 自适应预算） | 62.5 | 仅用自适应预算，效果有限 |
| **CurvZO(B=0.4d)**（固定预算 + 曲率采样） | 64.5 | 固定预算下已优于 US+AB |
| **CurvZO**（完整方法） | **66.8** ✅ | 完整结合曲率引导 + 自适应预算，性能最优 |

> 🔍 发现：
> - **曲率引导采样** 是性能提升的主因；
> - **自适应预算机制** 进一步优化了不同训练阶段的扰动策略，二者协同增效。

---

## 4. 关键结论和发现

### 主要发现
1. **局部曲率是指导稀疏 ZO 扰动的有效代理信号**：即使在无法获取梯度的情况下，也能从标量反馈中在线估计曲率，并用于参数选择。
2. **高曲率参数更值得扰动**：CurvZO 通过 `s_i = Δ²u_i` 构造 curvature score，并结合 EMA 和归一化稳定估计。
3. **动态调整扰动预算至关重要**：早期需要更大预算以维持探索，后期可聚焦于少数高曲率方向，CurvZO 利用 **effective support size** 和 **score sharpness** 实现自适应控制。
4. **理论保障**：论文提供了收敛性分析，证明 CurvZO 在适当条件下能达到 $ O(1/T) $ 收敛速率，方差项受扰动预算 $ B $ 控制。

---

### 方法的局限性
- **仅反映局部敏感性**：curvature score 是 per-coordinate 或 block-wise 的，未建模参数间的耦合关系（interaction），可能在某些强交互场景下采样次优。
- **对超参有一定依赖**：如平滑系数 $ \beta $、预算范围 $ [B_{min}, B_{max}] $ 等需合理设置。
- **仍属 heuristics 近似**：尽管有理论支撑，但 curvature score 并非真实 Fisher 对角线，存在一定偏差。

---

### 未来工作方向
- 探索 **跨 block 的结构化扰动模式**，例如考虑注意力头或 MLP 层之间的相关性。
- 将 CurvZO 与其他高效微调技术（如 AdaLoRA、BitFit）进一步融合。
- 扩展至 **多模态模型** 或 **强化学习微调** 场景，验证其泛化能力。
- 研究如何将 curvature signal 用于 **early stopping** 或 **layer-wise learning rate scheduling**。

---

> ✅ **总结一句话**：  
> **CurvZO 是首个完全在线、无需预计算的曲率感知稀疏 ZO 方法，它通过智能分配扰动资源，在不牺牲内存效率的前提下，显著提升了 ZO 微调的收敛速度与最终性能，为大规模模型在边缘设备上的高效适配提供了新路径。**

</details>

---

### 10. [Incremental GNN Embedding Computation on Streaming Graphs](https://arxiv.org/abs/2603.20622)

**Authors**: Qiange Wang, Haoran Lv, Yanfeng Zhang, Weng-Fai Wong, Bingsheng He  
**Category**: cs.DC  
**Published**: 2026-03-24  
**Score**: 9.0  
**Type**: new  
**ArXiv ID**: 2603.20622v1  

#### Abstract
Graph Neural Network (GNN) on streaming graphs has gained increasing popularity. However, its practical deployment remains challenging, as the inference process relies on Runtime Embedding Computation (RTEC) to capture recent graph changes. This process incurs heavyweight multi-hop graph traversal o...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Incremental GNN Embedding Computation on Streaming Graphs

---

## 1. 论文的主要贡献和创新点

### **解决了什么问题**

该论文针对 **Streaming Graphs 上的 GNN 推理效率瓶颈**，特别是 **Runtime Embedding Computation (RTEC)** 中存在的冗余计算问题。

在动态图场景中，传统 RTEC 方法需要对受影响顶点的整个 L-hop 邻域进行重新计算，即使其中大部分中间结果并未改变。这种“邻居爆炸”（neighbor explosion）导致计算开销巨大，即使只有 0.1% 的边更新，也可能达到全图重计算成本的 80%。

### **提出了什么新方法或新思路**

作者提出了一种 **通用且高效的增量式 RTEC 框架 NeutronRT**，其核心思想是：

- **细粒度算子解耦（Fine-grained Operator Decoupling）**  
  将 GNN 的 `message-aggregate-update` 范式进一步分解为四个可独立处理的组件：
  - `msg_local()`：边级局部消息计算
  - `nbr_ctx()`：邻域上下文聚合（如度、注意力总和）
  - `msg_cbn()`：结合上下文的消息归一化
  - `aggregate()` 和 `update()`

- **安全重排序（Safe Reordering）**  
  利用算子的**结合性**（associativity）和**可分发性**（distributivity），将完整的多跳传播转化为仅在受影响子图上的增量更新，从而避免对未受影响部分的重复计算。

- **GPU-CPU 协同系统 NeutronRT**  
  设计了一个支持大规模图的异构系统，通过以下机制提升效率：
  - 将中间 embedding 存储于 CPU 内存
  - 使用通信优化的调度策略减少 GPU-CPU 数据传输
  - 支持 chunked task scheduling 以适应有限 GPU 显存

### **相比现有方法的优势**

| 方法 | 缺陷 | NeutronRT 的优势 |
|------|------|----------------|
| **RTEC-Full (Full-neighbor)** | 完全重计算，冗余高达 95% | 减少 64%-99% 的计算量 |
| **RTEC-NS (Neighbor Sampling)** | 采样导致精度下降，尤其小 fanout 时 | 保持与原模型等效的精度 |
| **RTEC-UER (Unaffected Embedding Reuse)** | 仍需全邻域聚合，无法消除边级冗余 | 实现真正的边级增量更新 |

此外，NeutronRT 是首个能**统一支持复杂 GNN 模型**（如 GCN、GAT）的增量 RTEC 框架，而此前工作仅适用于简单聚合函数（如 sum）。

---

## 2. 核心实验方法和设置

### **使用的数据集**

| 数据集 | 类型 | 规模（节点数 / 边数） | 特征维度 |
|--------|------|------------------------|----------|
| **ogbn-arxiv (AX)** | 引用网络 | 0.17M / 1.2M | 128 |
| **reddit (RD)** | 社交帖子 | 0.23M / 114M | 602 |
| **ogbn-products (PT)** | 商品共购 | 2.4M / 62M | 100 |
| **Twitter (TW)** | 社交媒体 | 41M / 1.5B | 128 |
| **ogbn-paper (PR)** | 引用网络 | 111M / 1.6B | 128 |
| **friendster (FS)** | 社交网络 | 65.6M / 2.5B | 128 |

涵盖从小到超大规模、不同结构特性的图。

### **实验设置和评估指标**

- **任务**：Node Classification
- **GNN 模型**：GCN, GraphSAGE, GIN, MoNet, GAT, AGNN
- **更新模式**：边插入/删除混合，批量大小为 `0.01%|E|`（小图）或 `0.001%|E|`（大图）
- **硬件平台**：NVIDIA A6000 GPU + 512GB CPU 内存
- **评估指标**：
  - **响应时间（Response Time）**
  - **吞吐量（Throughput, edges/sec）**
  - **访问体积（Vertex/Edge Access Volume）**
  - **内存消耗（Memory Usage）**
  - **准确率（Accuracy）**

### **基线方法对比**

| 基线 | 描述 |
|------|------|
| **RTEC-Full** | 全邻域重计算 |
| **RTEC-NS5 / NS10** | 邻居采样（fanout=5/10） |
| **RTEC-UER** | 复用未受影响顶点 embedding |
| **Helios [36]** | 分布式动态图采样系统 |
| **Grapher [9]** | Serverless 平台下的 RTEC-UER 实现 |
| **InkStream [48]** | CPU-GPU 混合增量系统 |
| **Ripple [29]** | CPU 上的增量 RTEC 系统 |

所有基线均在 NeutronRT 框架下公平复现。

---

## 3. 主要实验结果和性能指标

### **关键性能数据**

#### ✅ **速度提升（Speedup）**

| 场景 | 速度提升范围 |
|------|-------------|
| 小图（in-memory） | **1.7x – 19.8x** vs. RTEC-Full |
| 大图（out-of-memory） | **37.8x – 145.8x** vs. RTEC-Full |
| vs. RTEC-NS10 | **4.2x – 26.1x** |
| vs. RTEC-UER | **10.0x – 57.1x** |

> 在 ogbn-paper 上，NeutronRT 将每批处理时间从 **324.3s（RTEC-Full）** 降至 **2.2s**。

#### ✅ **计算量减少**

- **冗余计算减少 64% – 99%**
- **边访问量减少 4.9x – 168.5x**
- **高阶顶点（top 20% 度）贡献 72%-85% 的优化收益**

#### ✅ **吞吐量（Throughput）**

| 图规模 | 吞吐量（edges/sec） |
|--------|---------------------|
| 小图（如 arXiv） | 76.7K – 154.3K |
| 大图（如 Twitter） | 681.8K – 872.5K |

### **与基线方法的对比结果**

| 对比项 | 结果 |
|--------|------|
| **vs. RTEC-NS** | 在低/中度图上显著更快；在高度图（如 Reddit）上略慢但精度更高 |
| **vs. RTEC-UER** | 所有场景下全面超越，速度提升达数十倍 |
| **vs. InkStream/Ripple** | 在大图上快 **5.3x–7.7x**，因完全 GPU 化执行 |

### **消融实验结果**

#### 🔬 **不同 batch size 下的表现**

- 当 `|ΔE| < 10` 时，NeutronRT 因构建受影响子图的开销略慢；
- 当 `|ΔE| = 1K` 时，达到峰值加速 **285.0x**；
- 当 `|ΔE| = 100M`（占原图 16%）时，仍保持 **3.8x** 加速。

#### 🔬 **不同层数的影响**

| 层数 | 速度提升趋势 |
|------|--------------|
| 2层 | 最佳表现（如 Paper 上 122.6x） |
| 3层 | 性能下降但仍显著（如 Paper 上 20.4x） |

原因：随着层数增加，受影响子图呈指数扩张。

#### 🔬 **On-Demand Embedding Computation (ODEC)**

- 当查询顶点数 `|V₀| ≥ 1K` 时，NeutronRT 显著优于非增量方法；
- 若查询集中在受影响区域，性能增益接近完整 RTEC。

---

## 4. 关键结论和发现

### **主要发现**

1. **绝大多数 RTEC 冗余来自未受影响子图**，占比可达 **65%-95%**，这是增量优化的根本前提。
2. **通过算子解耦与重排序，可在理论上保证增量计算与全邻域计算的等价性**，满足特定条件即可严格保准。
3. **NeutronRT 可泛化至多种复杂 GNN 模型**（包括 GCN、GAT），突破了以往仅支持 sum/max/min 聚合的限制。
4. **GPU-CPU 协同设计使十亿级图上的实时推理成为可能**，单卡即可处理 billion-scale streaming graphs。

### **方法的局限性**

1. **对“目标依赖型”消息函数（如 GAT 中含目标节点嵌入）的支持受限**：这类模型需对部分顶点回退到全邻域计算，称为“约束增量模型”（Constrained Incremental Models），带来额外开销（约 1.2x–1.7x 慢于纯增量）。
2. **极端密集更新下优势减弱**：当更新覆盖大部分图时，增量优势消失，退化为全图计算。
3. **CPU 内存压力大**：若图过大超出单机内存，需缓存高阶顶点，可能导致性能急剧下降（>10x 慢）。

### **未来工作方向**

1. **扩展至更深的 GNN 模型**（>3 层），探索更高效的多跳增量传播机制。
2. **支持分布式部署**，实现跨节点的增量同步与缓存一致性。
3. **结合 LLM 自动化生成增量程序**：利用 LLM + SMT solver 自动生成满足条件的 `msg_local`, `nbr_ctx` 等函数。
4. **探索近似增量方法**：在允许轻微误差的前提下进一步压缩计算与通信。

---

> **总结一句话**：  
> NeutronRT 通过**算子解耦 + 增量重计算 + GPU-CPU 协同**，首次实现了对复杂 GNN 模型的高效、精确、可扩展的 Streaming Graph 推理，在多个真实世界图上实现了 **最高 145.8x 的端到端加速**，为实时 GNN 应用提供了坚实基础。

</details>

---

### 11. [Optimizing Multi-Agent Weather Captioning via Text Gradient Descent: A Training-Free Approach with Consensus-Aware Gradient Fusion](https://arxiv.org/abs/2603.21673)

**Authors**: Shixu Liu  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.21673v1  

#### Abstract
Generating interpretable natural language captions from weather time series data remains a significant challenge at the intersection of meteorological science and natural language processing. While recent advances in Large Language Models (LLMs) have demonstrated remarkable capabilities in time seri...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：Optimizing Multi-Agent Weather Captioning via Text Gradient Descent

## 1. 论文的主要贡献和创新点

### 解决的问题
该论文针对**天气时间序列数据的自然语言描述生成**（weather time series captioning）中存在的挑战，提出了一种新的解决方案。现有方法存在以下问题：
- 数值预测模型（如GraphCast、Pangu-Weather）缺乏可解释性，无法生成人类可理解的文本描述；
- 单一LLM生成的天气描述往往过于泛化或技术性强，难以兼顾统计规律、物理机制与气象业务意义；
- 多视角信息融合困难，不同领域知识（统计、物理、操作）难以协同优化。

### 提出的新方法与思路
作者提出了 **WeatherTGD** —— 一种**无需训练的多智能体框架**，将文本梯度下降（Text Gradient Descent, TGD）引入天气描述生成任务中，其核心思想是：
- 将多个LLM作为专业化Agent，分别从三个角度生成“文本梯度”（即改进方向的反馈）：
  - **Statistical Analyst**：关注趋势、分布、异常等统计特征；
  - **Physics Interpreter**：分析变量间的因果关系和物理机制（如气压梯度驱动风场）；
  - **Meteorology Expert**：提供业务相关性解读（如冷锋过境、降水概率）。
- 设计了**Consensus-Aware Gradient Fusion**机制，融合多Agent输出时既提取共识信息，又保留独特见解；
- 构建了一个类比于传统梯度下降的**迭代优化循环**，逐步提升caption质量。

### 相比现有方法的优势
| 维度 | WeatherTGD优势 |
|------|----------------|
| **无需训练** | 完全基于prompt engineering和反馈机制，zero-shot运行，部署成本低； |
| **多视角融合更优** | 显式区分共识与独特观点，避免简单平均导致的信息稀释； |
| **优化过程可控** | 引入语义相似度阈值实现收敛判断，防止无限迭代； |
| **效率更高** | 并行执行Agent + 早期停止策略，token消耗仅为baseline的~3.5倍，低于多数multi-agent系统（如Self-Consistency达5.0×）。 |

---

## 2. 核心实验方法和设置

### 数据集
- 使用真实世界地面气象站采集的**500个天气时间序列样本**；
- 时间跨度：24至168小时（1–7天），每小时观测一次；
- 变量维度：温度（℃）、气压（hPa）、湿度（%）、风速（m/s）、降水量（mm）；
- 每条样本配有由**资深气象预报员**（≥10年经验）撰写的参考caption；
- 划分方式：60%训练、20%验证、20%测试（按气候区分层抽样）；
- 注：数据集将在论文接受后公开。

### 实验设置
- **LLM Backbones**（用于Agent和评估）：
  - DeepSeek-V3.2（671B）
  - MiniMax-01（456B，中英双语优化）
  - Qwen3-Next-80B-A3B-Instruct（80B）
- 所有模型通过OpenRouter API调用，temperature=0.2，max_tokens=2048。

### 评估指标
#### （1）LLM Judge Evaluation（GPT-4o作为裁判）
在四个维度打分（1–10分制）：
- **Statistical Accuracy (SA)**：定量描述准确性；
- **Physical Coherence (PC)**：物理解释合理性；
- **Meteorological Relevance (MR)**：是否符合气象业务规范；
- **Overall Quality (OQ)**：综合评分。

#### （2）Human Expert Evaluation
- 邀请5位博士级气象专家（≥5年预报经验）独立评分；
- 使用标准化评分协议，最终取平均；
- **Inter-annotator agreement**（Krippendorff’s α）为0.78，表明信度高。

#### （3）Reference-Based Metrics
- BLEU-4、ROUGE-L、BERTScore（与人工标注caption对比）

### 基线方法对比
共比较6种主流MAS（Multi-Agent System）baseline，均适配到天气captioning任务：
1. **AutoGen**：可对话Agent，灵活交互；
2. **CAMEL**：角色扮演通信Agent；
3. **LLM-Debate**：多轮辩论机制；
4. **Self-Consistency (SC)**：多路径推理+投票；
5. **AgentVerse**：动态招募Agent；
6. **MAD**：结构化辩论协议。

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自Table 1 & Table 2）
| 方法 | LLM Judge OQ | Human Expert OQ | BLEU-4 | ROUGE-L | BERTScore | Token × |
|------|---------------|------------------|--------|---------|-----------|----------|
| Vanilla（单Agent） | 5.54 | 5.40 | 0.327 | 0.397 | 0.751 | 1.0 |
| AgentVerse（最佳基线） | 7.01 | 6.86 | 0.399 | 0.465 | 0.802 | 3.8 |
| **WeatherTGD（本文）** | **8.50** | **8.34** | **0.467** | **0.527** | **0.845** | **3.5** |

> ✅ **提升显著**：相比最强基线AgentVerse，OQ提升 **+1.49** 分（LLM裁判），+1.48分（人类专家），接近满分尺度下的质变水平。

### 与基线方法的对比结果
- 在所有三个LLM backbone上，WeatherTGD均**全面超越所有基线**；
- 改进幅度稳定：DeepSeek-V3 (+1.55), MiniMax-01 (+1.52), Qwen3-Next (+1.39)；
- 性能-成本权衡图显示，WeatherTGD位于**左上方高效区域**，兼具高质量与较低计算开销；
- LLM裁判与人类专家评分高度一致（Pearson r = 0.94），验证自动评估可靠性。

### 消融实验结果（Ablation Study，Table 3）
使用Qwen3-Next backbone进行消融分析（OQ得分）：

| 变体 | OQ得分 | 相对下降 |
|------|--------|----------|
| Full WeatherTGD | 8.69 | —— |
| w/o Consensus Fusion（简单平均） | 7.92 | -0.77 |
| w/o Unique View Integration | 8.18 | -0.51 |
| w/o Physics Agent | 7.85 | -0.84 |
| w/o Meteorology Agent | 7.72 | -0.97 |
| w/o Statistical Agent | 7.58 | **-1.11** |
| w/o Iterative Refinement（单次生成） | 8.12 | -0.57 |
| w/o Length Constraint | 8.48 | -0.21 |

> 🔍 发现：
> - **Statistical Agent贡献最大**，说明用户最看重准确的数据概括；
> - **Consensus-Aware Fusion至关重要**，优于简单平均；
> - 迭代优化带来明显增益（+0.57），支持TGD有效性；
> - 所有组件均有正向作用，体现系统设计完整性。

---

## 4. 关键结论和发现

### 主要发现
1. **Text Gradient Descent可用于复杂domain captioning任务**，首次将其应用于天气时间序列理解和多视角描述生成；
2. **三类专业Agent分工明确且互补**，联合生成的文本梯度能有效指导caption向更全面、准确的方向演化；
3. **Consensus-Aware Gradient Fusion机制优于传统聚合方式**，能在保留共识的同时不丢失专家独到见解；
4. **迭代优化确实有效**：前3–4轮快速提升，第5轮基本收敛，配合early stopping实现高效优化；
5. **人类评估与LLM裁判高度一致**，证明GPT-4o作为judge在该任务上的可信性。

### 方法的局限性
- 当前仅支持**单一地点的时间序列描述**，未考虑空间拓扑或区域天气系统联动；
- 所有Agent依赖外部LLM API，存在延迟和成本波动风险；
- 文本梯度的质量受限于LLM自身能力，若某Agent出现幻觉会影响整体结果；
- 尚未支持多语言输出（如中英文双语caption）；
- 融合机制仍需预设超参数（如Tcons=0.8, Tunique=0.6），缺乏自适应调节。

### 未来工作方向（原文Section 5）
- **Multilingual Captioning**：扩展至中文、西班牙语等多种语言；
- **Knowledge Graph Integration**：结合气象知识图谱增强物理机制推理；
- **Adaptive Agent Selection**：根据输入数据动态启用/关闭特定Agent；
- **Integration with NWP Systems**：与数值天气预报系统对接，实现实时可解释输出；
- 探索更复杂的**gradient weighting机制**，例如基于Agent置信度加权。

---

> 📌 **总结一句话**：  
> WeatherTGD首次将**Text Gradient Descent**范式应用于**天气时间序列captioning**，通过**三专业Agent协作 + 共识感知梯度融合 + 迭代优化机制**，实现了无需训练即可生成高质量、多视角、可解释的天气描述，在多项指标上大幅超越现有multi-agent系统，同时保持良好效率。

</details>

---

### 12. [Communication-Avoiding SpGEMM via Trident Partitioning on Hierarchical GPU Interconnects](https://arxiv.org/abs/2603.21444)

**Authors**: Julian Bellavita, Lorenzo Pichetti, Thomas Pasquali, Flavio Vella, Giulia Guidi  
**Category**: cs.DC  
**Published**: 2026-03-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.21444v1  

#### Abstract
The multiplication of two sparse matrices, known as SpGEMM, is a key kernel in scientific computing and large-scale data analytics, underpinning graph algorithms, machine learning, simulations, and computational biology, where sparsity is often highly unstructured. The unstructured sparsity makes ac...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*Communication-Avoiding SpGEMM via Trident Partitioning on Hierarchical GPU Interconnects*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
现代大规模分布式多GPU系统具有**层次化互联架构**（hierarchical GPU interconnects），即：
- **节点内通信**（intranode）通过高带宽、低延迟的互连（如NVLink）进行；
- **节点间通信**（internode）则依赖较慢的网络（如InfiniBand或Slingshot）。

然而，现有的分布式 **SpGEMM**（Sparse General Matrix Multiplication）算法通常将通信视为均质的，忽略了这种带宽差异，导致：
- 过度的**节点间通信开销**；
- 无法充分利用高速的节点内互连；
- 可扩展性和性能受限。

该论文针对这一问题，提出了一种新的层次感知（hierarchy-aware）SpGEMM算法。

---

### 🚀 提出的新方法：TRIDENT
TRIDENT 是一种**通信规避型**（communication-avoiding）、**异步执行**的分布式 SpGEMM 算法，其核心是 **Trident Partitioning**（三叉分区）策略。

#### 创新设计：
- **Hybrid 2D-1D 分区机制**：
  - 跨节点采用 **2D 分区**（coarse-grained 2D tiles），每个计算节点负责一个大块；
  - 在节点内部对每个 2D tile 进一步划分为 **1D slices**，分布到该节点内的多个 GPUs 上。
  - 形成 $\sqrt{P/\lambda} \times \sqrt{P/\lambda} \times \lambda$ 的三维进程网格（$\lambda$: 每节点GPU数）。

- **两阶段通信优化**：
  1. **节点间通信**：仅交换一次所需 tile（via GI）；
  2. **节点内聚合**：利用高速 LI（如NVLink）通过 `Allgatherv` 局部重建完整子矩阵，避免重复跨节点传输。

- **异步 C-stationary 执行模型**：
  - 输出矩阵 $C$ 固定分配（C-stationary），各 tile 可独立推进；
  - 使用 **非阻塞点对点通信** 和 **RDMA 请求队列** 实现异步进度，减少同步等待。

---

### 🔍 相比现有方法的优势
| 方面 | TRIDENT | 传统方法（如Sparse SUMMA） |
|------|--------|-----------------------------|
| **通信模式** | 区分 GI/LI，优先使用 LI | 忽略层级，统一处理 |
| **通信量** | 显著降低 internode 通信体积 | 高频全局广播（如 `MPI_Bcast`） |
| **可扩展性** | 更好地随 GPU 数量扩展 | 同步瓶颈明显 |
| **适用场景** | 特别适合 unstructured sparse matrices | 对 structured 矩阵更优 |

> 💡 TRIDENT 是首个专为 **hierarchical GPU interconnects** 设计的分布式 SpGEMM 算法。

---

## 2. 核心实验方法和设置

### 📊 数据集
使用来自真实世界的大型稀疏矩阵，涵盖生物信息学、图分析等领域：
- 来源：SuiteSparse [19] 和 HipMCL [3]
- 主要矩阵包括：
  - `HV15R`, `mouse_gene`, `archaea`, `eukarya`
  - `isolates_subgraph4/5`, `cage15`, `reddit`, `uniparc` 等
- 多数为 **unstructured sparse matrices**，非零元分布均匀且无规则结构。

> 表格摘要（部分）：
>
> | Matrix | Size (rows) | NNZ | Density | Imbalance (max/avg) |
> |--------|-------------|-----|---------|---------------------|
> | HV15R | 2.0M | 283M | 6.95e-5 | 15.44 |
> | eukarya | 3.2M | 359M | 3.42e-5 | 1.03 |
> | isolates_subgraph4 | 4.4M | 264M | 1.38e-5 | 1.02 |

---

### ⚙️ 实验设置
- **硬件平台**：NERSC 的 **Perlmutter 超级计算机**
  - 每节点 4 个 NVIDIA A100 GPU
  - 节点内：NVLink 3.0（~900 GB/s）
  - 节点间：Slingshot-11 Dragonfly（~400 Gb/s）
- **软件栈**：
  - MPI: Cray MPICH 8.30（支持 GPU-aware）
  - Intranode: NCCL 2.26
  - Local SpGEMM: KoKKOsKERNELS + cuSPARSE
- **规模测试**：从 4 到 256（部分至 400）GPUs

---

### 🎯 评估指标
- **运行时间**（runtime）
- **加速比**（speedup）
- **通信开销分解**（breakdown by phase）
- **节点间通信量**（internode communication volume）
- **在实际应用中的表现**：Markov Clustering（MCL）的 expansion 步骤

---

### 🆚 基线方法对比
1. **Trilinos (TPETRA)**  
   - 1D row-wise 分布式 SpGEMM
   - 支持 sparsity-aware 通信
   - 使用非阻塞 `MPI_Isend/Irecv`

2. **CombBLAS Sparse SUMMA**
   - 经典 2D 算法（Bulk-synchronous）
   - 使用 `MPI_Bcast` 进行通信
   - 已知存在主机端拷贝瓶颈

3. **Improved Sparse SUMMA**（本文实现）
   - 改进版 2D 算法
   - 数据保留在设备内存，合并操作也在 GPU 完成
   - 使用与 TRIDENT 相同的库（KoKKOsKERNELS）
   - 作为主要对比 baseline

---

## 3. 主要实验结果和性能指标

### 📈 性能加速比（Speedup）

| 对比对象 | 最大加速比 | 几何平均加速比（Geomean Speedup） |
|--------|-----------|-------------------------------|
| vs. **Trilinos** | **5.95×** (`mouse_gene`, 256 GPUs) | **2.96×** (@256 GPUs) |
| vs. **Improved Sparse SUMMA** | **2.38×** | **1.54×** (@256 GPUs) |

> ✅ TRIDENT 在所有测试矩阵上均优于或持平于 Improved Sparse SUMMA，在大规模下优势显著。

---

### 📉 通信量减少
- **节点间通信体积最多减少 2×**
- 如 Figure 10 所示，`isolates_subgraph4` 上每进程发送数据量下降约 50%
- 即使在某些情况下单次通信略增，但由于通信次数少且可重用，总体通信时间仍更低

---

### ⏱️ 运行时分解（Runtime Breakdown）
以 `isolates_subgraph4` 和 `mouse_gene` 为例（Figure 9）：
- **TRIDENT 的通信开销增长缓慢**，尤其 internode communication 占比小；
- **Intranode communication（Allgatherv）几乎可忽略**，得益于 NCCL 高效实现；
- 相比之下，Improved Sparse SUMMA 的 broadcast 开销随规模上升明显。

> 结论：TRIDENT 的通信可扩展性更好。

---

### 🔬 消融实验与有效性验证
虽然未明确命名“ablation study”，但以下分析体现了消融思想：
- **随机打乱结构化矩阵**（如 HV15R）后，TRIDENT 性能反超 Trilinos 达 ~6×，说明其更适合 unstructured 场景；
- **MCL 应用中早期迭代加速明显**（因矩阵稠密），体现 TRIDENT 在高通信压力下的优势；
- **异步设计容忍负载不均衡**（如 `mouse_gene` imbalance=4.65），而同步算法易受拖累。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **层次化通信架构必须被显式建模**  
   忽视 intranode/internode 带宽差异会导致严重的性能浪费。

2. **Trident Partitioning 显著降低 internode 通信量**  
   通过“一次跨节点 + 多次本地聚合”策略，有效复用高速 LI。

3. **TRIDENT 在 unstructured sparse matrices 上全面领先**
   - 尤其在大规模（256+ GPUs）时优势扩大；
   - 异步执行缓解了同步瓶颈和负载不均影响。

4. **在真实应用（MCL）中同样有效**
   - 在 Markov Clustering 中实现最高 **2× 加速**；
   - 表明 TRIDENT 不仅适用于微基准，也具备实用价值。

---

### ⚠️ 方法的局限性
- **依赖每节点固定数量的 GPUs**（如 λ=4 或 8），灵活性受限；
- 当前实现假设系统可划分为同构的 LI 组，复杂拓扑可能需调整；
- 内部使用 `Allgatherv` 导致节点内轻微同步，虽影响小但仍存在；
- 对 extremely structured matrices（如带状矩阵）不如 1D 方法高效。

---

### 🔮 未来工作方向
- 探索 **3D + Trident 混合分区** 以进一步减少通信；
- 扩展至其他稀疏原语（sparse primitives），如 SpMV、SpTSV；
- 支持动态负载均衡与自适应分区；
- 在 AMD（RCCL）和其他异构平台上的移植与优化；
- 结合 sparsity-aware 技术与 hierarchy-aware 架构设计。

---

## ✅ 总结
TRIDENT 提出了一种全新的 **hierarchy-aware** 分布式 SpGEMM 架构，通过 **Trident Partitioning** 和 **异步通信调度**，首次系统性地解决了现代多GPU超级计算机中 **intranode/internode 带宽失配** 导致的通信瓶颈问题。实验证明其在多种 unstructured 矩阵上实现了高达 **2.38×** 的加速，并显著降低了 internode 通信量。这是迈向高效、可扩展稀疏计算的重要一步。

</details>

---

### 13. [Joint Surrogate Learning of Objectives, Constraints, and Sensitivities for Efficient Multi-objective Optimization of Neural Dynamical Systems](https://arxiv.org/abs/2603.20984)

**Authors**: Frithjof Gressmann, Ivan Georgiev Raikov, Seung Hyun Kim, Mattia Gazzola, Lawrence Rauchwerger, Ivan Soltesz  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.20984v1  

#### Abstract
Biophysical neural system simulations are among the most computationally demanding scientific applications, and their optimization requires navigating high-dimensional parameter spaces under numerous constraints that impose a binary feasible/infeasible partition with no gradient signal to guide the ...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文核心结论与实验结果总结

## 1. 论文的主要贡献和创新点

### 解决的问题
- **高维、昂贵且受约束的多目标优化难题**：在计算神经科学中，生物物理神经网络模型（如多室神经元模型）具有大量难以通过实验或解析确定的参数，其仿真极其耗时（supercomputing scale），且需满足多个硬性约束（如放电频率范围、膜时间常数等）。
- **传统优化方法失效**：标准采样策略（如Monte Carlo、LHS）无法有效探索复杂目标空间；而基于高斯过程（Gaussian Process, GP）的代理模型因计算复杂度随样本量超线性增长，在大规模问题上不可扩展。
- **梯度信号缺失**：约束条件通常为二值化（可行/不可行），导致搜索空间被划分为不连续区域，缺乏平滑梯度指导搜索。

### 提出的新方法：DMOSOPT 框架
作者提出 **DMOSOPT**（Distributed Multi-Objective Surrogate-Assisted Optimization），一个可扩展的分布式多目标代理辅助优化框架，其核心创新在于：

- **联合学习的统一可微代理模型（Joint Surrogate Model）**：
  - 使用深度神经网络（如 ResNet 和 FT-Transformer）构建一个共享的、端到端可微的代理模型 $f: \mathbb{R}^n \rightarrow \mathbb{R}^{q+k}$，同时预测 $q$ 个目标值和 $k$ 个约束满足概率。
  - 该模型输出一个**统一的代理梯度**（surrogate gradient）：
    $$
    g_{\text{sopt}} = \nabla_x [\text{hypervolume loss}(y(x)) + (\text{constraint loss}(c(x)) - 1)]
    $$
    此梯度能同时引导搜索向**更低的目标值**和**更高的约束满足度**方向前进。

- **敏感性感知的探索机制（Sensitivity-informed Exploration）**：
  - 利用代理模型的偏导数 $\frac{\partial f}{\partial x_i}$ 提供每参数的敏感性估计（sensitivity indices），无需额外仿真开销。
  - 将这些敏感性映射到 MOEA（如 NSGA-II）的分布指数（distribution indices），使敏感参数经历较小扰动（精细调优），非敏感参数进行更广探索。

- **混合搜索策略（Hybrid Search）**：
  - 不直接优化代理模型，而是将代理梯度作为“有信息的搜索方向”注入到多目标进化算法（MOEA）中。
  - MOEA 的种群多样性、变异与重组操作提供探索压力，弥补代理模型不确定性量化能力的不足。

### 相比现有方法的优势
| 维度 | 传统方法（如 GPR/MEGP） | DMOSOPT（c+o-FTTransformer） |
|------|--------------------------|-------------------------------|
| **可扩展性** | $O(n^3)$ 训练成本，难处理大样本 | 神经网络架构天然支持大数据集和高维输出 |
| **信息整合** | 目标与约束分离建模，信息割裂 | 联合学习共享表示，捕捉目标-约束耦合关系 |
| **搜索效率** | 缺乏对不可行区域的有效逃离机制 | 代理约束梯度主动引导进入可行区 |
| **敏感性分析** | 需专用方法（FAST/DGSM），额外仿真预算 | 代理梯度副产品，零额外成本 |
| **HPC 支持** | 通常单机运行 | 原生支持 MPI 分布式执行 |

---

## 2. 核心实验方法和设置

### 数据集与基准问题
实验覆盖从单细胞到全脑区网络的多层次神经建模任务：

1. **CA1 海马中间神经元单细胞模型（Single-cell）**
   - **数量**：9 种形态各异的 CA1 中间神经元类型（如 PVBC, OLM, NGFC 等）
   - **参数维度**：每类 10–15 个 biophysical parameters
   - **目标函数**：4 项电生理特性误差（输入电阻、膜时间常数、f-I 曲线、峰电位幅度）
   - **约束条件**：7–8 条二值约束（单调 f-I、无自发放电、ISI 特性等）

2. **脊髓运动神经元模型（Motoneuron）**
   - **目的**：测试极端约束场景下的可行性求解能力
   - **配置**：
     - *窄范围*：生物学启发的合理参数边界（较易满足约束）
     - *宽范围*：人为扩大边界，其中 3/7 约束随机采样下满足率为 0%，形成组合不可行问题

3. **CA1 海马网络全尺度模拟（Large-scale Network）**
   - **规模**：约 836,970 个神经元，近 10 亿突触连接
   - **优化对象**：142 个突触权重参数
   - **目标**：36 维目标空间（12 个群体的平均放电率、活跃细胞比例及其标准差）
   - **约束**：12 条正放电率约束
   - **计算代价**：每次仿真 ~42.3 秒 @ 300 CPU cores（~5.4 节点）

> 注：由于完整网络仿真成本过高，实际优化在保留全部细胞类型和连接模式的“切片”（slice, ~8,700 神经元）上进行，结果迁移至全网验证。

### 实验设置
- **优化轮次**：
  - 单细胞与运动神经元：25 epochs，每轮 100 次评估
  - 网络级：50 epochs，每轮 100 次评估
- **初始采样**：Symmetric Latin Hypercube (SLHC)
- **控制器-工作者架构**：基于 MPI 实现分布式并行仿真与优化
- **开源实现**：`dmosopt` 框架已公开于 GitHub（github.com/dmosopt）

### 评估指标
| 指标 | 公式/说明 | 含义 |
|------|---------|------|
| **Normalized Hypervolume (HV)** | $\text{Vol}\left(\bigcup_{a \in A} [a, r]\right)$, $r = 1.1 \times \text{nadir}$ | 衡量帕累托前沿覆盖体积，越高越好 |
| **HV-AUC** | 曲线下面积（梯形法） | 反映收敛速度，越大越快 |
| **Inverted Generational Distance (IGD)** | $\frac{1}{|R|} \sum_{r \in R} \min_{a \in A} \|r - a\|$ | 与真实前沿距离，越小越好 |
| **Additive ε-indicator** | $\max_b \min_a \max_j (a_j - b_j)$ | 若 ≤0，则 A 完全支配 B |
| **NRMSE** | $\sqrt{\frac{\sum (y - \hat{y})^2}{\max y - \min y}}$ | 代理模型预测精度 |

### 基线方法对比
- **Gaussian Process Regression (GPR)**：Matérn 5/2 核，仅用于目标预测
- **Multi-Expert GP (MEGP)**：每个目标独立拟合 GP，支持多输出
- **Random Sampling**：SLH, LH, MC, Sobol 等被动采样
- 所有 GP 基线仅在可行样本上训练，以避免无效数据污染模型。

---

## 3. 主要实验结果和性能指标

### 关键性能数据汇总

| 任务 | 方法 | 性能表现 |
|------|------|--------|
| **单细胞优化** | c+o-FTTransformer | 平均 IGD 排名第1，HV-AUC 第2；ε-indicator < 2% |
| **高度约束问题**（Motoneuron, 宽范围） | c+o-FTTransformer | 在随机采样完全失败（0% 可行性）的情况下，成功找到可行解，HV 收敛速度快于 GPR 2–3 倍 |
| **大规模网络优化** | c+o-FTTransformer | 达到同等解质量时：<br>• 比 MEGP 节省 **2×** 计算资源<br>• 比 GPR 节省 **5×** 计算资源<br>• 提前 **54%**（vs MEGP）、**80%**（vs GPR）轮次收敛 |

### 与基线方法的对比结果
- **优于所有采样策略**：
  - 图2B 显示，即使数千次评估后，SLH/LH/MC/Sobol 仍远未达到理论最大 HV（~1.4641），而 GPR 已接近。
- **神经网络代理显著超越 GP 基线**：
  - 图2C：所有 NN 架构（尤其是 c+o-FTTransformer）的 ε-indicator 均 < 2%，表明其帕累托前沿几乎完全支配无代理基线。
  - 图2D：NN 的 NRMSE 显著低于 GP，预测更准确。
  - 图2E：NN 在更高精度的同时保持相当甚至更低推理延迟，性价比更高。

- **联合建模（joint learning）优于分离建模**：
  - 图3B：c+o-FTTransformer 在最终解质量（IGD）上排名第一，证明联合学习提升帕累托前沿质量。
  - trade-off：o- 架构（仅目标）初期收敛更快（HV-AUC 更高），但最终解可能不够可行；c+o 稍慢但更稳健。

- **约束梯度是突破“零可行性”瓶颈的关键**：
  - 图4A：在宽参数范围内，随机采样完全找不到任何可行解（3 条约束满足率 0%）。
  - 引入 $\nabla_x c(x)$ 后，优化器能快速进入可行区，实现有效收敛。

- **代理梯度敏感性媲美传统方法**：
  - 图3C：sgrad（代理梯度法）得到的 IGD 和 HV 与 FAST、DGSM 相当，但无需额外仿真预算。

### 消融实验结果
- **梯度增强有效性**（图9）：
  - 对难优化的 BS 细胞，基础 c+o-FTTransformer 收敛慢于无代理基线。
  - 加入梯度下降步骤（特别是 V(c+o) 联合优化）后，性能反超，证明梯度引导可修复冷启动问题。
- **轨迹采样提升训练效率**（图4B）：
  - 使用梯度路径上的多样化样本（trace samples）微调代理模型，仅需 10 个样本即可大幅降低 NRMSE，优于随机补充。
- **超前沿分析**（图16）：
  - 合并所有方法的帕累托最优解后，c+o-FTTransformer 贡献了最大份额（多数群体 >40%），说明其生成的解最具竞争力。

---

## 4. 关键结论和发现

### 主要发现
1. ✅ **联合学习产生质变**：将目标、约束、敏感性统一在一个可微代理模型中，使得三者之间的耦合依赖得以显式建模（例如同一离子通道影响目标也决定约束是否满足），从而在约束相关区域获得更准预测。
2. ✅ **代理梯度是高效搜索的核心驱动力**：即使原始问题是黑箱且无梯度，通过代理模型提供的光滑梯度，可以实现类似梯度下降的定向搜索，极大加速收敛。
3. ✅ **神经网络代理优于 GP 在大规模场景**：尽管失去校准的不确定性估计，但其卓越的可扩展性和表达能力使其更适合 HPC 级别的复杂优化任务。
4. ✅ **框架通用性强**：虽源于计算神经科学，但适用于任何带约束的多目标黑箱优化问题，如工程设计、药物发现、材料科学等。

### 方法的局限性
- **过拟合风险**：早期训练数据可能导致代理模型陷入虚假最优（spurious optima）。文中采用交叉验证选代、周期重训、精英保留缓解。
- **平滑近似可能过度软化约束边界**：对于存在尖锐可行边界的场景，当前平滑分类头可能不够精确。
- **高维目标下的 HV 计算瓶颈**：HV 指标随目标数指数级增长，限制了其在非常多目标（many-objective）场景的应用。
- **缺乏严格信任域机制**：目前依赖 MOEA 的内在探索平衡利用与探索，未来可引入更系统的信任域策略。

### 未来工作方向
- 引入不确定性量化（Uncertainty Quantification）：
  - 如 evidential deep learning 或 deep ensembles，结合贝叶斯优化的探索原则。
- 自适应平滑或约束特定头设计：更好处理陡峭的可行性边界。
- 多目标扩展：
  - 开发适用于高维目标空间的替代指标或分解策略。
- 预训练与迁移学习：
  - 在相似问题库上预训练代理模型，实现 warm-start。
- 与可微分模拟器结合：
  - 对部分可微系统，融合解析梯度进一步提升效率。

> **总结**：DMOSOPT 通过**联合学习 + 可微代理 + 梯度引导 + 敏感性反馈**的闭环设计，实现了对极端复杂神经动力系统的高效多目标优化，将原本需要数百 CPU-day 的任务压缩至数十天，并为跨学科的受限优化问题提供了新的范式。

</details>

---

### 14. [In-network Attack Detection with Federated Deep Learning in IoT Networks: Real Implementation and Analysis](https://arxiv.org/abs/2603.21596)

**Authors**: Devashish Chaudhary, Sutharshan Rajasegarar, Shiva Raj Pokhrel, Lei Pan, Ruby D  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.21596v1  

#### Abstract
The rapid expansion of the Internet of Things (IoT) and its integration with backbone networks have heightened the risk of security breaches. Traditional centralized approaches to anomaly detection, which require transferring large volumes of data to central servers, suffer from privacy, scalability...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：*In-network Attack Detection with Federated Deep Learning in IoT Networks: Real Implementation and Analysis*

---

## 1. 论文的主要贡献和创新点

### ✅ 解决的问题
物联网（IoT）网络中设备数量激增，导致攻击面扩大。传统的**集中式异常检测系统**存在以下问题：
- **隐私风险**：原始数据需上传至中心服务器；
- **通信开销大**：大量数据传输增加带宽负担；
- **延迟高、扩展性差**：难以在资源受限的边缘设备上实时运行。

此外，现有联邦学习（Federated Learning, FL）研究多停留在仿真阶段，缺乏在真实 IoT 设备上的部署验证。

---

### 🚀 提出的新方法与创新思路
本文提出了一种**基于联邦深度学习的轻量级 in-network 攻击检测框架**，其核心创新包括：

1. **On-device 自编码器模型设计**  
   在 Raspberry Pi 等资源受限的边缘设备上部署轻量级 Autoencoder 模型，实现本地化异常检测，避免原始数据外传。

2. **Federated Learning + Transfer Learning 联合架构**  
   - 使用 **FedAvg** 算法聚合各节点的模型权重，保护数据隐私；
   - 引入 **Transfer Learning** 加速训练收敛，提升异构设备下的泛化能力。

3. **真实 IoT Testbed 构建与攻击模拟**  
   - 搭建由 9 个 Raspberry Pi 组成的真实分层网络（含 Coordinator、Router、Edge Device 和 Attacker）；
   - 利用 ZigBee 协议中的 AT 命令实施 **Redirection Attacks**，生成真实的恶意流量。

4. **特征工程面向边缘优化**  
   提取延迟统计（mean delay, quartiles）、Shannon entropy、通信频次、平均跳数等适用于边缘计算的低维特征，并进行 MinMax 归一化处理。

---

### 🔍 相比现有方法的优势
| 方面 | 本文方案优势 |
|------|---------------|
| **隐私性** | 仅传输模型参数，不上传原始数据，符合 GDPR 等隐私规范 |
| **通信效率** | 通信开销从 4.5MB（集中式）降至 378KB（联邦式），减少约 **92%** |
| **实时性** | 支持 on-device 实时检测，响应更快 |
| **可扩展性** | 分布式训练适应大规模 IoT 部署 |
| **真实性** | 基于真实硬件 testbed 和实际攻击场景，非纯仿真 |

---

## 2. 核心实验方法和设置

### 📊 数据集来源与构建方式
- **自建真实数据集**：通过搭建的 IoT testbed 收集正常与攻击状态下的网络日志。
- **采集平台**：Raspberry Pi 3B+ + Digi XBee S2C ZigBee 模块。
- **数据记录格式**：每条日志包含时间戳、源/目的节点 ID、状态码（如 `S:0` 表示成功）。
- **时间窗口**：以 1 分钟为单位提取特征，共收集 5 小时正常数据 + 多轮攻击数据。

#### 攻击场景分类（共三类）：
| 场景 | 描述 | 数量 |
|------|------|-------|
| **Scenario I** | 边缘设备被重定向到其他路由器 | 12 次 |
| **Scenario II** | 路由器之间路径被篡改 | 5 次 |
| **Scenario III** | 所有设备被引导向攻击者节点发送数据 | 7 次 |

每次攻击持续 5 分钟，前后分别记录 20 分钟正常行为和 10 分钟恢复期。

---

### ⚙️ 实验设置
- **模型结构**：
  - Autoencoder 架构：输入层 31 维 → 编码器（32→16 神经元，ReLU）→ 解码器（16→32→31，Sigmoid）
  - 使用 Keras Functional API 实现，Adam 优化器，MSE 损失函数
  - 训练参数：batch size=32，epochs=100，learning rate=0.001

- **联邦学习流程**（见 Algorithm 1）：
  1. 初始化全局模型并下发至各 Router；
  2. 各节点使用本地数据训练；
  3. 完成后将权重上传给 Coordinator；
  4. Coordinator 执行 FedAvg 聚合更新全局模型；
  5. 更新后的模型再广播回所有节点。

- **阈值设定**：
  使用公式：  
  $$
  \text{Threshold} = \text{Mean} + k \times \text{Standard Deviation}
  $$  
  其中均值与标准差来自验证集（1小时正常数据），$k$ 取值范围为 1~4。

---

### 🎯 评估指标
| 指标 | 公式 |
|------|------|
| Accuracy | $(TP + TN) / (TP + TN + FP + FN)$ |
| Precision | $TP / (TP + FP)$ |
| Recall | $TP / (TP + FN)$ |
| F1-Score | $2 \cdot (Precision \cdot Recall) / (Precision + Recall)$ |

同时比较：
- **Federated vs Centralized** 学习性能
- 不同 $k$ 值对检测效果的影响
- 通信与计算开销分析

---

### 🔀 基线方法对比
| 方法 | 类型 | 是否 on-device | 是否 FL | 是否真实部署 |
|------|------|----------------|---------|--------------|
| 本文方法 | Federated Autoencoder | ✅ 是 | ✅ 是 | ✅ 是 |
| Centralized Autoencoder | 集中式模型 | ❌ 否 | ❌ 否 | ✅ 是（用于对比） |
| Fed-ANIDS [10] | 联邦异常检测 | ❌ 未实测 | ✅ 是 | ❌ 仅仿真 |
| Kanthuru et al. [6] | ML-based detection | ❌ 离线分析 | ❌ 否 | ✅ 是（但无 FL） |

---

## 3. 主要实验结果和性能指标

### 📈 关键性能数据（见 Table III）

| 设备 | 方法 | $k$ | Acc (%) | Prec (%) | Rec (%) | F1-Score |
|------|------|-----|--------|----------|---------|-----------|
| R1 | Federated | 4 | 97.41 | 89.26 | 90.00 | **0.8963** |
| R1 | Centralized | 4 | 97.41 | 85.71 | 95.00 | **0.9012** |
| R2 | Federated | 1 | 93.33 | 77.42 | 80.54 | **0.7895** |
| R2 | Centralized | 4 | 90.52 | 64.50 | 86.58 | **0.7393** |
| R3 | Federated | 2 | 80.97 | 29.08 | 86.90 | **0.4358** |
| R3 | Centralized | 4 | 88.52 | 42.19 | 96.43 | **0.5870** |

> 注：F1-score 是选择最优 $k$ 的依据。

---

### 🔬 对比结果分析
- **总体表现接近集中式模型**：
  - Federated 方法在 R1 上达到 **F1=0.8963**，与 Centralized 的 **0.9012** 几乎持平；
  - 表明联邦学习可在不牺牲太多精度的前提下实现隐私保护。

- **不同设备间存在差异性**：
  - R3 性能较低，可能与其在网络拓扑中位置有关（连接较少，数据多样性不足）；
  - Federated 模型普遍具有更高 Recall（更敏感），而 Centralized 更偏向 Precision（误报少）。

- **最优 $k$ 值因设备而异**：
  - Federated 模型需个性化调参：R1 最优 $k=4$，R2 最优 $k=1$，R3 最优 $k=2$；
  - Centralized 模型统一在 $k=4$ 表现最佳。

---

### 🔻 通信与计算开销对比

| 指标 | Centralized | Federated |
|------|-------------|-----------|
| 总通信量（5小时） | **4.5 MB** | **378 KB**（↓92%） |
| 单次通信频率 | 每秒传输数据包 | 每60分钟传输一次模型权重（12.6KB） |
| 数据类型 | 原始日志数据 | 模型参数（weights） |
| 隐私保障 | 差 | 强（无需共享数据） |

✅ 显著降低通信成本，适合带宽受限的无线 IoT 网络。

---

### 🧪 消融实验（隐含分析）
虽然未明确列出消融实验表格，但从设计中可推断以下关键组件作用：
- **Transfer Learning**：预训练模型初始化显著加快收敛速度，尤其对资源有限设备；
- **Feature Selection**：选取 delay、entropy、hop count 等高层特征有效捕捉路由异常；
- **Local Model Personalization**：允许每个 router 设置不同的 $k$ 阈值，提升了灵活性与适应性。

---

## 4. 关键结论和发现

### ✅ 主要发现
1. **联邦学习可用于真实 IoT 环境中的 in-network 攻击检测**，且性能媲美集中式方法。
2. **通信开销大幅降低**（92%↓），特别适合低功耗、低带宽的边缘网络。
3. **本地化检测增强隐私性与安全性**，防止单点数据泄露引发连锁攻击。
4. **Redirection Attacks 能被有效识别**，表现为延迟突变、熵升高、路径跳跃异常等特征变化。
5. **模型性能受设备角色影响**，需结合拓扑结构进行个性化配置（如阈值调整）。

---

### ⚠️ 局限性
1. **攻击类型有限**：目前只测试了 Redirection Attacks，尚未涵盖 DoS、DDoS、Packet Injection 等复杂攻击；
2. **设备异构性未充分建模**：所有 router 使用相同模型结构，未考虑 CPU/GPU/内存差异带来的训练偏差；
3. **静态阈值机制**：采用固定 $k$ 值，无法动态适应网络环境变化（如负载波动）；
4. **规模较小**：testbed 仅包含 9 个节点，需进一步验证在更大规模网络中的可扩展性。

---

### 🔮 未来工作方向
1. **扩展攻击类型测试**：引入 DoS、MITM、Spoofing 等攻击，验证通用性；
2. **探索 Adaptive Thresholding 与 Incremental Learning**：支持模型在线更新与自动调参；
3. **研究设备异构下的 Federated Optimization**：根据设备能力分配不同模型复杂度（如 Split Learning）；
4. **集成 Homomorphic Encryption 或 Differential Privacy**：进一步加强模型更新过程的安全性；
5. **部署于工业级 IoT 系统**：如智能电网、车联网等真实应用场景中验证鲁棒性。

---

## ✅ 总结
本论文成功实现了首个基于 **Federated Deep Learning** 的 **真实部署型 in-network 攻击检测系统**。它不仅证明了联邦学习在资源受限 IoT 设备上的可行性，还展示了其在保持高检测精度的同时，显著降低通信开销并增强隐私保护的能力。该工作为构建**可扩展、安全、去中心化的下一代 IoT 安全体系**提供了重要实践基础。

</details>

---

### 15. [Data-Free Layer-Adaptive Merging via Fisher Information for Long-to-Short Reasoning LLMs](https://arxiv.org/abs/2603.21705)

**Authors**: Tian Xia  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 8.5  
**Type**: new  
**ArXiv ID**: 2603.21705v1  

#### Abstract
Model merging has emerged as a practical approach to combine capabilities of specialized large language models (LLMs) without additional training. In the Long-to-Short (L2S) scenario, merging a base model with a long-chain-of-thought reasoning model aims to preserve reasoning accuracy while reducing...

<details>
<summary><strong>🤖 AI Summary (by qwen-long)</strong> - Click to expand</summary>

# 论文总结：**Data-Free Layer-Adaptive Merging via Fisher Information for Long-to-Short Reasoning LLMs**

---

## 1. 论文的主要贡献和创新点

### 解决的问题
本文聚焦于 **Long-to-Short (L2S)** 场景下的大语言模型（LLM）合并问题，即如何将一个擅长长链推理（long chain-of-thought, long-CoT）的模型与一个基础简洁响应模型合并，在保留高推理准确率的同时显著缩短输出长度。  
传统方法如 **Task Arithmetic** 假设模型行为随参数变化呈线性关系，但在 L2S 这种跨任务、大参数偏移场景中该假设被系统性违反，导致性能下降。

### 提出的新方法与新思路
作者提出 **FIM-Merging**，一种无需训练、无需领域特定校准数据（calibration data）的层自适应合并框架，其核心思想是：
- **理论驱动设计**：首次从理论上证明了 Task Arithmetic 的合并误差受每层 Hessian 范数控制（Proposition 1），并利用 **Fisher-Hessian Equivalence** 在局部最优处成立的性质，提出使用对角 **Fisher Information Matrix (FIM)** 作为 Hessian 的可计算代理。
- **数据自由计算**：通过在 base model 上使用 **随机 token 输入**（uniform from vocabulary）前向传播并反向计算梯度平方均值，即可估计对角 FIM，完全避免依赖下游任务数据。
- **层自适应系数分配**：结合 FIM 和每层 task vector 的 L2 范数 $ \|\delta\|^2 $ 构建重要性得分 $ \text{FIM} \times \|\delta\|^2 $，用于生成每层不同的 merging coefficient $\alpha_l$，实现“难改层保守、易改层激进”的策略。

具体实例化为两种变体：
- **FIM-TA**：FIM 指导的 Task Arithmetic（无剪枝）
- **FIM-TIES**：FIM 指导 + TIES-Merging 的增强版本（含 sign agreement、delta pruning、gate protection 和 residual norm calibration）

### 相比现有方法的优势
| 维度 | ACM [23] | FIM-Merging (Ours) |
|------|----------|---------------------|
| 层重要性信号 | Mutual Information（需校准数据） | Diagonal FIM × $\|\delta\|^2$（无需数据） |
| 是否需要 calibration data | 是 | 否 ✅ |
| 理论依据 | 无 | 有（Proposition 1 + Fisher-Hessian）✅ |
| 与误差关联 | 间接 | 直接（FIM ~ Hessian）✅ |
| 计算开销 | 高（多次前传） | 低（仅 8 次随机 fwd/bwd）✅ |
| 超参敏感性 | 中等（需调 $\theta$） | 无（sharpness 参数自适应）✅ |

> ✅ **核心优势总结**：**理论可解释 + 数据免费 + 性能领先**

---

## 2. 核心实验方法和设置

### 使用的数据集
本研究不涉及训练或微调，因此未使用原始训练数据；评估基于以下标准数学推理 benchmark：
- **GSM8K**
- **MATH500**
- **Minerva Math**
- **OlympiadBench**
- **CollegeMath**
- **AIME24**

所有评估均采用官方 Qwen2.5-Math 工具包进行一致性评测。

### 实验设置与评估指标
- **模型规模**：
  - **1.5B**：Base: `Qwen2.5-Math-1.5B`，Fine-tuned: `DeepSeek-R1-Distill-Qwen-1.5B`
  - **7B**：同上对应 7B 版本
- **FIM 计算细节**：
  - 使用 **N=8** 条随机 token 序列（vocab uniform sampling）
  - 序列长度 64，无需真实标签
  - 对 base model 执行 backward 得到梯度，取各层参数梯度平方平均作为对角 FIM
- **TIES 阈值设定**：
  - 1.5B：保留 top 20% 的 delta entries
  - 7B：保留 top 40%（反映更大冗余度）
- **评估指标**：
  - **Accuracy (%)**：各 benchmark 上的答案正确率
  - **Average Response Length (tokens)**：输出长度（越短越好）
  - **Self-consistency decoding**：n=16, temp=0.3 下的 AIME24 表现

### 基线方法对比
- **Task Arithmetic** [7]
- **TIES-Merging** [20]
- **AIM** [13]
- **Sens-Merging** [11]
- **ACM-TA / ACM-TIES** [23]（当前 SOTA 层自适应方法）

---

## 3. 主要实验结果和性能指标

### 关键性能数据（来自 Table 2 & 3）

#### 📊 1.5B 尺度结果（Table 2）
| Method | Avg Acc (%) | Avg Length (tokens) |
|--------|-------------|---------------------|
| DeepSeek-R1-1.5B (long-CoT) | 40.5 | 5,671 |
| ACM-TIES | 43.3 | 1,489 |
| **FIM-TIES (ours)** | **47.3** (+3.9) | **411** (-92.6%) |

> ✅ 在全部六个 benchmark 上均优于 ACM-TIES，尤其 AIME24 从 10.0 → 20.0（+10.0），且响应长度压缩至原模型的 **7.2%**

#### 📊 7B 尺度结果（Table 3，greedy decoding）
| Method | MATH500 | Olympiad | AIME24 (greedy) |
|--------|--------|---------|------------------|
| ACM-TIES | 84.0 | 46.4 | 33.3 |
| **FIM-TIES (ours)** | **90.2** (+6.2) | **47.9** (+1.5) | 26.7 |

> ✅ 在 **5/6 benchmarks 达到 SOTA**，MATH500 提升巨大（+6.2 pts）

#### 🔍 结合 Self-Consistency 解码（Table 4）
| Method | n=1 (greedy) | n=16 (self-consistency) |
|-------|--------------|--------------------------|
| ACM-TIES | 33.3 | — |
| **FIM-TIES (ours)** | 26.7 | **36.7** (+3.4) ✅ |

> ⭐ 即使 greedy 模式略逊，FIM-TIES 合并后的模型仍保留足够推理多样性，经 self-consistency 可超越依赖校准数据的 ACM-TIES

### 消融实验结果（Ablation Study）

#### ❌ Weight Norm Only 失败
- 若仅用 task vector 的 Frobenius norm 作为层重要性信号，得到近乎统一的 $\alpha_l \approx 0.53$
- 性能低于标准 Task Arithmetic → 说明单一信号无效

#### ✅ FIM × $\|\delta\|^2$ 是关键
- 在 1.5B 上 FIM-only 与 full product 效果接近（因 $\|\delta\|^2$ 变化小）
- 在 7B 上 full product 明显更优（$\|\delta\|^2$ 跨层差异 >5×）→ 支持理论预测

#### 🔁 TIES Threshold 影响
| Threshold | GSM8K (7B) | AIME24 (7B) |
|----------|------------|------------|
| 0.1 | 91.2 | 16.7 |
| 0.2 | 91.2 | 26.7 |
| **0.4 (ours)** | **92.2** | **26.7** |

> 更高的保留比例（0.4）更适合 7B 模型，体现参数冗余度影响

---

## 4. 关键结论和发现

### 主要发现
1. **Layer-adaptive merging 的必要性源于非线性**：
   - Task Arithmetic 的误差由每层 Hessian 范数主导（Proposition 1）
   - 早期 layer 的 FIM 和 NL Score 更高，表明其非线性强，应保守合并
2. **FIM 是理论正确的 proxy**：
   - 利用 Fisher-Hessian equivalence，FIM 成为衡量“合并难度”的自然选择
   - 实验验证 NL Score 与相对合并误差高度相关（Pearson r=0.972）
3. **数据自由可行且高效**：
   - 使用随机输入计算的 FIM 与真实数据下趋势一致
   - 仅需 8 次 fwd/bwd，可在 CPU 上完成（约 20–30 分钟 for 7B）
4. **FIM-TIES 实现精度与效率双赢**：
   - 在多个 benchmark 上超越依赖校准数据的方法
   - 输出长度减少超 90%，适合部署

### 方法的局限性
- 在 **greedy decoding 下 AIME24 表现暂时落后于 ACM-TIES**（26.7 vs 33.3），说明极端复杂推理任务可能仍受益于任务感知信号
- 当前方法假设 base model 接近最优（满足 Fisher-Hessian equivalence），若 fine-tuned model 偏离较大，FIM 代理效果可能减弱
- 未探索结构化稀疏或其他 geometry-aware merging 方式

### 未来工作方向
- 将 FIM-Merging 扩展到多模型融合（multi-way merging）
- 探索 FIM 与其他 sensitivity signals（如 gradient covariance）的组合
- 应用于 vision-language models 或 preference alignment 场景
- 开发更轻量化的 FIM 估计器以支持千亿级模型

---

> ✅ **一句话总结**：  
> 本文提出了首个**理论驱动、无需数据**的层自适应模型合并方法 **FIM-Merging**，通过 **Fisher Information Matrix** 准确捕捉每层的合并风险，在 **L2S 场景下实现了精度与效率的双重突破**，为训练自由、校准自由的模型集成提供了新范式。

</details>

---

### 16. [AgenticGEO: A Self-Evolving Agentic System for Generative Engine Optimization](https://arxiv.org/abs/2603.20213)

**Authors**: Jiaqi Yuan, Jialu Wang, Zihan Wang, Qingyun Sun, Ruijie Wang, Jianxin Li  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.20213v1  

#### Abstract
Generative search engines represent a transition from traditional ranking-based retrieval to Large Language Model (LLM)-based synthesis, transforming optimization goals from ranking prominence towards content inclusion. Generative Engine Optimization (GEO), specifically, aims to maximize visibility ...

---

### 17. [Modeling Epistemic Uncertainty in Social Perception via Rashomon Set Agents](https://arxiv.org/abs/2603.20750)

**Authors**: Jinming Yang, Xinyu Jiang, Xinshan Jiao, Xinping Zhang  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.20750v1  

#### Abstract
We present an LLM-driven multi-agent probabilistic modeling framework that demonstrates how differences in students' subjective social perceptions arise and evolve in real-world classroom settings, under constraints from an observed social network and limited questionnaire data. When social informat...

---

### 18. [Can we automatize scientific discovery in the cognitive sciences?](https://arxiv.org/abs/2603.20988)

**Authors**: Akshay K. Jagadish, Milena Rmus, Kristin Witte, Marvin Mathony, Marcel Binz, Eric Schulz  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.20988v1  

#### Abstract
The cognitive sciences aim to understand intelligence by formalizing underlying operations as computational models. Traditionally, this follows a cycle of discovery where researchers develop paradigms, collect data, and test predefined model classes. However, this manual pipeline is fundamentally co...

---

### 19. [LongCat-Flash-Prover: Advancing Native Formal Reasoning via Agentic Tool-Integrated Reinforcement Learning](https://arxiv.org/abs/2603.21065)

**Authors**: Jianing Wang, Jianfei Zhang, Qi Guo, Linsen Guo, Rumei Li, Chao Zhang, Chong Peng, Cunguang Wang, Dengchang Zhao, Jiarong Shi, Jingang Wang, Liulin Feng, Mengxia Shen, Qi Li, Shengnan An, Shun Wang, Wei Shi, Xiangyu Xi, Xiaoyu Li, Xuezhi Cao, Yi Lu, Yunke Zhao, Zhengyu Chen, Zhimin Lin, Wei Wang, Peng Pei, Xunliang Cai  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.21065v1  

#### Abstract
We introduce LongCat-Flash-Prover, a flagship 560-billion-parameter open-source Mixture-of- Experts (MoE) model that advances Native Formal Reasoning in Lean4 through agentic tool-integrated reasoning (TIR). We decompose the native formal reasoning task into three independent formal capabilities, i....

---

### 20. [Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference](https://arxiv.org/abs/2603.20224)

**Authors**: Patrick Wilhelm, Thorsten Wittkopp, Odej Kao  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.20224v1  

#### Abstract
Large Language Models (LLMs) demonstrate exceptional performance across diverse tasks but come with substantial energy and computational costs, particularly in request-heavy scenarios. In many real-world applications, the full scale and capabilities of LLMs are often unnecessary, as Small Language M...

---

### 21. [CALVO: Improve Serving Efficiency for LLM Inferences with Intense Network Demands](https://arxiv.org/abs/2603.21257)

**Authors**: Weiye Wang (Shanghai Jiao Tong University), Chen Chen (Shanghai Jiao Tong University), Junxue Zhang (University of Science and Technology of China), Zhusheng Wang (Huawei), Hui Yuan (Huawei), Zixuan Guan (Huawei), Xiaolong Zheng (Huawei), Qizhen Weng (Institute of Artificial Intelligence), Yin Chen (Institute of Artificial Intelligence), Minyi Guo (Shanghai Jiao Tong University)  
**Category**: cs.DC  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.21257v1  

#### Abstract
Distributed prefix caching has become a core technique for efficient LLM serving. However, for long-context requests with high cache hit ratios, retrieving reusable KVCache blocks from remote servers has emerged as a new performance bottleneck. Such network-intensive LLM inference is expected to bec...

---

### 22. [LLM-ODE: Data-driven Discovery of Dynamical Systems with Large Language Models](https://arxiv.org/abs/2603.20910)

**Authors**: Amirmohammad Ziaei Bideh, Jonathan Gryak  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.20910v1  

#### Abstract
Discovering the governing equations of dynamical systems is a central problem across many scientific disciplines. As experimental data become increasingly available, automated equation discovery methods offer a promising data-driven approach to accelerate scientific discovery. Among these methods, g...

---

### 23. [Model Evolution Under Zeroth-Order Optimization: A Neural Tangent Kernel Perspective](https://arxiv.org/abs/2603.21169)

**Authors**: Chen Zhang, Yuxin Cheng, Chenchen Ding, Shuqi Wang, Jingreng Lei, Runsheng Yu, Yik-Chung WU, Ngai Wong  
**Category**: cs.LG  
**Published**: 2026-03-24  
**Score**: 8.0  
**Type**: new  
**ArXiv ID**: 2603.21169v1  

#### Abstract
Zeroth-order (ZO) optimization enables memory-efficient training of neural networks by estimating gradients via forward passes only, eliminating the need for backpropagation. However, the stochastic nature of gradient estimation significantly obscures the training dynamics, in contrast to the well-c...

---

### 24. [A Framework for Low-Latency, LLM-driven Multimodal Interaction on the Pepper Robot](https://arxiv.org/abs/2603.21013)

**Authors**: Erich Studerus, Vivienne Jia Zhong, Stephan Vonschallen  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.21013v1  

#### Abstract
Despite recent advances in integrating Large Language Models (LLMs) into social robotics, two weaknesses persist. First, existing implementations on platforms like Pepper often rely on cascaded Speech-to-Text (STT)->LLM->Text-to-Speech (TTS) pipelines, resulting in high latency and the loss of paral...

---

### 25. [AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling](https://arxiv.org/abs/2603.21357)

**Authors**: Liang Ding  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.21357v1  

#### Abstract
LLM agents fail on the majority of real-world tasks -- GPT-4o succeeds on fewer than 15% of WebArena navigation tasks and below 55% pass@1 on ToolBench (Zhou et al., 2024; Qin et al., 2024) -- yet every failed trajectory is routinely discarded, wasting the dominant source of collected experience. We...

---

### 26. [Unified-MAS: Universally Generating Domain-Specific Nodes for Empowering Automatic Multi-Agent Systems](https://arxiv.org/abs/2603.21475)

**Authors**: Hehai Lin, Yu Yan, Zixuan Wang, Bo Xu, Sudong Wang, Weiquan Huang, Ruochen Zhao, Minzhi Li, Chengwei Qin  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.21475v1  

#### Abstract
Automatic Multi-Agent Systems (MAS) generation has emerged as a promising paradigm for solving complex reasoning tasks. However, existing frameworks are fundamentally bottlenecked when applied to knowledge-intensive domains (e.g., healthcare and law). They either rely on a static library of general ...

---

### 27. [Counterfactual Credit Policy Optimization for Multi-Agent Collaboration](https://arxiv.org/abs/2603.21563)

**Authors**: Zhongyi Li, Wan Tian, Yikun Ban, Jinju Chen, Huiming Zhang, Yang Liu, Fuzhen Zhuang  
**Category**: cs.AI  
**Published**: 2026-03-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.21563v1  

#### Abstract
Collaborative multi-agent large language models (LLMs) can solve complex reasoning tasks by decomposing roles and aggregating diverse hypotheses. Yet, reinforcement learning (RL) for such systems is often undermined by credit assignment: a shared global reward obscures individual contributions, infl...

---

### 28. [FinReflectKG -- HalluBench: GraphRAG Hallucination Benchmark for Financial Question Answering Systems](https://arxiv.org/abs/2603.20252)

**Authors**: Mahesh Kumar, Bhaskarjit Sarmah, Stefano Pasquali  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.20252v1  

#### Abstract
As organizations increasingly integrate AI-powered question-answering systems into financial information systems for compliance, risk assessment, and decision support, ensuring the factual accuracy of AI-generated outputs becomes a critical engineering challenge. Current Knowledge Graph (KG)-augment...

---

### 29. [RLVR Training of LLMs Does Not Improve Thinking Ability for General QA: Evaluation Method and a Simple Solution](https://arxiv.org/abs/2603.20799)

**Authors**: Kaiyuan Li, Jing-Cheng Pang, Yang Yu  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.20799v1  

#### Abstract
Reinforcement learning from verifiable rewards (RLVR) stimulates the thinking processes of large language models (LLMs), substantially enhancing their reasoning abilities on verifiable tasks. It is often assumed that similar gains should transfer to general question answering (GQA), but this assumpt...

---

### 30. [DiscoUQ: Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles](https://arxiv.org/abs/2603.20975)

**Authors**: Bo Jiang  
**Category**: cs.CL  
**Published**: 2026-03-24  
**Score**: 7.5  
**Type**: new  
**ArXiv ID**: 2603.20975v1  

#### Abstract
Multi-agent LLM systems, where multiple prompted instances of a language model independently answer questions, are increasingly used for complex reasoning tasks. However, existing methods for quantifying the uncertainty of their collective outputs rely on shallow voting statistics that discard the r...

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
